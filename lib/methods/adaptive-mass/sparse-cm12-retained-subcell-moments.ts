import type { SolidWorld } from "../../core/solid-world";
import { compileRetainedSceneSolidFractions } from "./sparse-cm12-retained-open-density";
import { integrateRetainedSceneDensity, retainedSceneDensityRange,
  type RetainedSceneDensity, type RetainedScenePoint } from "./sparse-cm12-retained-scene-density";

export interface RetainedSceneSubcellMoments {
  readonly fieldGeneration: number;
  readonly dimensions: readonly [number, number, number];
  readonly cellSize: number;
  /** Each contribution is divided by FULL finest-cell volume h³. The eight
   * consecutive octants use x bit0, y bit1, z bit2; a whole open octant is1/8. */
  readonly seedAmounts: Float32Array;
  readonly openVolumes: Float32Array;
  readonly receipt: Readonly<{ cells: number; subcells: number; bytes: number;
    integratedSubcells: number; constantSubcells: number; reflectedSubcells: number; estimatedAbsoluteError: number;
    maximumEstimatedSubcellMeanError: number }>;
}

/** Fixed physical h/2 subboxes for the declared rigid occupancy measure.
 * Rigid geometry may classify their centers, but density is integrated over
 * the entire selected open subbox, never approximated by its center sample.
 * The static SolidWorld terrain slab is intersected before integration.
 * This full initial-domain cache permits later body motion without rebuilding
 * seed geometry; callers may defer its allocation until rigid coupling exists. */
export function compileRetainedSceneSubcellMoments(field: RetainedSceneDensity,
  dimensions: readonly [number, number, number], cellSize: number, world: SolidWorld,
  options: { solidFractions?: Uint8Array; absoluteMeanTolerance?: number;
    maximumRectangles?: number; maximumBytes?: number } = {}): RetainedSceneSubcellMoments {
  const h = Math.fround(cellSize), count = dimensions[0] * dimensions[1] * dimensions[2];
  const bytes = count * 8 * 4 * 2, maximum = options.maximumBytes ?? 256 * 1024 * 1024;
  const tolerance = options.absoluteMeanTolerance ?? 2e-7;
  if (!(h > 0) || !Number.isFinite(h) || dimensions.some(n => !Number.isSafeInteger(n) || n < 1)
    || !Number.isSafeInteger(count) || !Number.isSafeInteger(bytes) || !(tolerance > 0) || !Number.isFinite(tolerance)
    || !Number.isSafeInteger(maximum) || maximum < 0) throw new Error("Invalid retained subcell lattice or budget");
  if (bytes > maximum) throw new Error(`Retained rigid subcell moments require ${bytes} bytes; budget ${maximum}`);
  const solid = options.solidFractions ?? compileRetainedSceneSolidFractions(dimensions, world);
  if (solid.length !== count) throw new Error("Retained rigid static geometry does not match its lattice");
  const seedAmounts = new Float32Array(8 * count), openVolumes = new Float32Array(8 * count);
  const indexAt = (x: number, y: number, z: number) => x + dimensions[0] * (y + dimensions[1] * z);
  const reflection = [false, false, false];
  for (const axis of [0, 2]) {
    const center = (field.domain.lower[axis] + field.domain.upper[axis]) / 2;
    reflection[axis] = Math.abs(dimensions[axis] * h - (field.domain.upper[axis] - field.domain.lower[axis])) <= dimensions[axis] * h * 1e-12
      && field.primitives.every(p => p.kind === "ellipsoid" ? p.center[axis] === center
        : p.kind === "box" ? p.lower[axis] + p.upper[axis] === 2 * center
          : p.curvature[axis] === 0 || p.center[axis] === center);
    if (!reflection[axis]) continue;
    for (let z = 0; z < dimensions[2] && reflection[axis]; z++) for (let y = 0; y < dimensions[1] && reflection[axis]; y++) for (let x = 0; x < dimensions[0]; x++) {
      const opposite = [x, y, z]; opposite[axis] = dimensions[axis] - 1 - opposite[axis];
      if (solid[indexAt(x, y, z)] !== solid[indexAt(opposite[0], opposite[1], opposite[2])]) { reflection[axis] = false; break; }
    }
  }
  const volume = h ** 3;
  let integratedSubcells = 0, constantSubcells = 0, reflectedSubcells = 0, error = 0, maximumMeanError = 0;
  for (let z = 0; z < dimensions[2]; z++) for (let y = 0; y < dimensions[1]; y++) for (let x = 0; x < dimensions[0]; x++) {
    const index = indexAt(x, y, z), at = 8 * index;
    const sourceX = reflection[0] ? Math.min(x, dimensions[0] - 1 - x) : x;
    const sourceZ = reflection[2] ? Math.min(z, dimensions[2] - 1 - z) : z;
    if (sourceX !== x || sourceZ !== z) {
      const source = 8 * indexAt(sourceX, y, sourceZ), flip = (sourceX !== x ? 1 : 0) | (sourceZ !== z ? 4 : 0);
      for (let octant = 0; octant < 8; octant++) {
        seedAmounts[at + octant] = seedAmounts[source + (octant ^ flip)];
        openVolumes[at + octant] = openVolumes[source + (octant ^ flip)];
      }
      reflectedSubcells += 8; continue;
    }
    if (solid[index] === 255) { constantSubcells += 8; continue; }
    const origin = [field.domain.lower[0] + x * h, field.domain.lower[1] + y * h, field.domain.lower[2] + z * h] as const;
    const upper = origin.map(v => v + h) as unknown as RetainedScenePoint;
    const range = retainedSceneDensityRange(field, { lower: origin, upper });
    const bottom = solid[index] / 255;
    for (let octant = 0; octant < 8; octant++) {
      const lowerY = Math.max((octant & 2) ? .5 : 0, bottom), upperY = (octant & 2) ? 1 : .5;
      const open = Math.max(0, upperY - lowerY) / 4;
      openVolumes[at + octant] = open;
      if (open === 0 || range[1] === 0 || range[0] === 1) {
        seedAmounts[at + octant] = range[0] === 1 ? open : 0; constantSubcells++; continue;
      }
      const lower: RetainedScenePoint = [origin[0] + ((octant & 1) ? .5 : 0) * h,
        origin[1] + lowerY * h, origin[2] + ((octant & 4) ? .5 : 0) * h];
      const upper: RetainedScenePoint = [origin[0] + ((octant & 1) ? 1 : .5) * h,
        origin[1] + upperY * h, origin[2] + ((octant & 4) ? 1 : .5) * h];
      const integral = integrateRetainedSceneDensity(field, { lower, upper }, {
        absoluteTolerance: volume * open * tolerance, maximumRectangles: options.maximumRectangles });
      if (!integral.toleranceMet) throw new Error(`Retained rigid subcell integral did not converge at ${x},${y},${z}/${octant}`);
      seedAmounts[at + octant] = integral.amount / volume; integratedSubcells++;
      error += integral.estimatedAbsoluteError;
      maximumMeanError = Math.max(maximumMeanError, integral.estimatedAbsoluteError / (volume * open));
    }
  }
  return Object.freeze({ fieldGeneration: field.generation,
    dimensions: Object.freeze([...dimensions]) as unknown as RetainedSceneSubcellMoments["dimensions"], cellSize: h,
    seedAmounts, openVolumes,
    receipt: Object.freeze({ cells: count, subcells: 8 * count, bytes, integratedSubcells, constantSubcells, reflectedSubcells,
      estimatedAbsoluteError: error * (reflection[0] ? 2 : 1) * (reflection[2] ? 2 : 1), maximumEstimatedSubcellMeanError: maximumMeanError }) });
}
