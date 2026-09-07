import { SOLID_WORLD_BRICK_CELLS, SOLID_WORLD_TERRAIN_MATERIAL_ID, type SolidWorld } from "../../core/solid-world";
import { compileRetainedSceneFineMeans, integrateRetainedSceneDensity,
  type RetainedSceneDensity, type RetainedScenePoint } from "./sparse-cm12-retained-scene-density";

export interface RetainedOpenSceneFineMeans {
  readonly fieldGeneration: number;
  readonly dimensions: readonly [number, number, number];
  readonly cellSize: number;
  /** Integral of q over the open part, divided by the full finest-cell volume. */
  readonly effectiveMeans: Float32Array;
  /** Open geometric volume divided by full finest-cell volume. */
  readonly openFractions: Float32Array;
  /** Canonical SolidWorld q8 occupancy after ordered region overrides. */
  readonly solidFractions: Uint8Array;
  readonly receipt: Readonly<{
    cells: number; closedCells: number; fractionalCells: number; clippedIntegrals: number;
    estimatedAbsoluteError: number; maximumEstimatedOpenMeanError: number;
  }>;
}

export interface RetainedOpenSceneOptions {
  /** Reuse the accepted unrestricted seed moments; never reconstruct from current resident means. */
  readonly seedMeans?: Float32Array;
  readonly absoluteMeanTolerance?: number;
  readonly maximumRectangles?: number;
}

function checkedLattice(dimensions: readonly [number, number, number], cellSize: number) {
  const h = Math.fround(cellSize), count = dimensions[0] * dimensions[1] * dimensions[2];
  if (!(h > 0) || !Number.isFinite(h) || dimensions.some(n => !Number.isSafeInteger(n) || n < 1)
    || !Number.isSafeInteger(count) || count > 0x1000_0000) throw new Error("Invalid retained open-domain lattice");
  return { h, count };
}

/** SolidWorld geometry is canonical, including page/region ordering. Filled
 * and cleared edit voxels are whole boxes. Its only fractional constructor is
 * terrain, which fills each column voxel from its bottom. The retained
 * geometry uses that stored q8 fraction as the slab height; the separately
 * quantized signed distance does not supersede the authoritative measure.
 * This does not claim to recover the original unquantized terrain surface. */
export function compileRetainedSceneSolidFractions(dimensions: readonly [number, number, number], world: SolidWorld): Uint8Array {
  checkedLattice(dimensions, 1);
  const count = dimensions[0] * dimensions[1] * dimensions[2], solid = new Uint8Array(count), materials = new Uint16Array(count);
  const at = (x: number, y: number, z: number) => x + dimensions[0] * (y + dimensions[1] * z);
  for (const page of world.pages) {
    const origin = page.coordinate.map(v => v * SOLID_WORLD_BRICK_CELLS);
    for (let z = 0; z < 8; z++) for (let y = 0; y < 8; y++) {
      const wy = origin[1] + y, wz = origin[2] + z;
      if (wy < 0 || wy >= dimensions[1] || wz < 0 || wz >= dimensions[2]) continue;
      for (let x = 0; x < 8; x++) {
        const wx = origin[0] + x; if (wx < 0 || wx >= dimensions[0]) continue;
        const source = x + 8 * (y + 8 * z), target = at(wx, wy, wz);
        solid[target] = page.solidFraction[source]; materials[target] = page.materialId[source];
      }
    }
  }
  for (const region of world.regions ?? []) {
    const lo = region.minimum.map((v, axis) => Math.max(0, Math.min(dimensions[axis], v)));
    const hi = region.maximumExclusive.map((v, axis) => Math.max(0, Math.min(dimensions[axis], v)));
    for (let z = lo[2]; z < hi[2]; z++) for (let y = lo[1]; y < hi[1]; y++) {
      const start = at(lo[0], y, z), end = start + Math.max(0, hi[0] - lo[0]);
      solid.fill(region.operation === "fill" ? 255 : 0, start, end);
      materials.fill(region.operation === "fill" ? region.materialId ?? 1 : 0, start, end);
    }
  }
  for (let i = 0; i < count; i++) if (solid[i] > 0 && solid[i] < 255 && materials[i] !== SOLID_WORLD_TERRAIN_MATERIAL_ID) {
    throw new Error(`Fractional SolidWorld voxel ${i} has no declared subvoxel geometry (material ${materials[i]})`);
  }
  return solid;
}

function compileMoments(field: RetainedSceneDensity, dimensions: readonly [number, number, number],
  h: number, solidFractions: Uint8Array, options: RetainedOpenSceneOptions): RetainedOpenSceneFineMeans {
  const count = solidFractions.length, tolerance = options.absoluteMeanTolerance ?? 2e-7;
  if (!(tolerance > 0) || !Number.isFinite(tolerance)) throw new Error("Invalid retained open integral tolerance");
  const seed = options.seedMeans ?? compileRetainedSceneFineMeans(field, dimensions, h, options);
  if (seed.length !== count) throw new Error("Retained seed mean snapshot does not match the open-domain lattice");
  const effectiveMeans = seed.slice(), openFractions = new Float32Array(count).fill(1), volume = h ** 3;
  let closedCells = 0, fractionalCells = 0, clippedIntegrals = 0, error = 0, maximumMeanError = 0;
  for (let i = 0; i < count; i++) {
    const solid = solidFractions[i];
    if (solid === 0) continue;
    if (solid === 255) { effectiveMeans[i] = 0; openFractions[i] = 0; closedCells++; continue; }
    fractionalCells++;
    const fraction = (255 - solid) / 255;
    openFractions[i] = fraction;
    const x = i % dimensions[0], yz = Math.floor(i / dimensions[0]), y = yz % dimensions[1], z = Math.floor(yz / dimensions[1]);
    const lower: RetainedScenePoint = [field.domain.lower[0] + x * h,
      field.domain.lower[1] + (y + solid / 255) * h, field.domain.lower[2] + z * h];
    const upper: RetainedScenePoint = [field.domain.lower[0] + (x + 1) * h,
      field.domain.lower[1] + (y + 1) * h, field.domain.lower[2] + (z + 1) * h];
    const integral = integrateRetainedSceneDensity(field, { lower, upper }, {
      absoluteTolerance: volume * fraction * tolerance, maximumRectangles: options.maximumRectangles });
    if (!integral.toleranceMet) throw new Error(`Retained clipped integral did not converge at ${x},${y},${z}`);
    effectiveMeans[i] = integral.amount / volume; clippedIntegrals++;
    error += integral.estimatedAbsoluteError;
    maximumMeanError = Math.max(maximumMeanError, integral.estimatedAbsoluteError / (volume * fraction));
  }
  return Object.freeze({ fieldGeneration: field.generation,
    dimensions: Object.freeze([...dimensions]) as unknown as RetainedOpenSceneFineMeans["dimensions"], cellSize: h,
    effectiveMeans, openFractions, solidFractions,
    receipt: Object.freeze({ cells: count, closedCells, fractionalCells, clippedIntegrals,
      estimatedAbsoluteError: error, maximumEstimatedOpenMeanError: maximumMeanError }) });
}

/** Integrate the retained field on actual canonical open boxes. Averaging q
 * first then multiplying by the voxel's open fraction is not equivalent when
 * the fluid transition intersects a partial terrain voxel. */
export function compileRetainedOpenSceneFineMeans(field: RetainedSceneDensity,
  dimensions: readonly [number, number, number], cellSize: number, world: SolidWorld,
  options: RetainedOpenSceneOptions = {}): RetainedOpenSceneFineMeans {
  const { h } = checkedLattice(dimensions, cellSize);
  return compileMoments(field, dimensions, h, compileRetainedSceneSolidFractions(dimensions, world), options);
}

export interface RetainedOpenSceneEditMoments {
  readonly next: RetainedOpenSceneFineMeans;
  /** Intersection of the old and new open domains. For qCurrent=a*qSeed+b,
   * its surviving amount/fullVolume is a*survivingSeedMeans+b*survivingOpenFractions. */
  readonly survivingSeedMeans: Float32Array;
  readonly survivingOpenFractions: Float32Array;
  readonly newlyOpenedFractions: Float32Array;
  readonly newlyClosedFractions: Float32Array;
}

/** Geometric receipts for a live boundary transaction. These do not reseed
 * newly exposed space or fit new a,b coefficients: either would invent liquid
 * or alter the field remaining in the old open region. Consumers can integrate
 * the old accepted field on the intersection, account for displaced mass,
 * and explicitly author how newly opened support becomes occupied. */
export function compileRetainedOpenSceneEditMoments(field: RetainedSceneDensity,
  previous: RetainedOpenSceneFineMeans, world: SolidWorld,
  options: RetainedOpenSceneOptions = {}): RetainedOpenSceneEditMoments {
  if (field.generation !== previous.fieldGeneration) throw new Error("Retained open edit uses a different seed field generation");
  const reusedOptions = { ...options, seedMeans: options.seedMeans
    ?? compileRetainedSceneFineMeans(field, previous.dimensions, previous.cellSize, options) };
  const next = compileRetainedOpenSceneFineMeans(field, previous.dimensions, previous.cellSize, world, reusedOptions);
  const intersectionSolid = next.solidFractions.map((q, i) => Math.max(q, previous.solidFractions[i]));
  const intersection = compileMoments(field, previous.dimensions, previous.cellSize, intersectionSolid, reusedOptions);
  const newlyOpenedFractions = new Float32Array(next.solidFractions.length), newlyClosedFractions = new Float32Array(next.solidFractions.length);
  for (let i = 0; i < next.solidFractions.length; i++) {
    newlyOpenedFractions[i] = Math.max(0, previous.solidFractions[i] - next.solidFractions[i]) / 255;
    newlyClosedFractions[i] = Math.max(0, next.solidFractions[i] - previous.solidFractions[i]) / 255;
  }
  return Object.freeze({ next, survivingSeedMeans: intersection.effectiveMeans,
    survivingOpenFractions: intersection.openFractions, newlyOpenedFractions, newlyClosedFractions });
}
