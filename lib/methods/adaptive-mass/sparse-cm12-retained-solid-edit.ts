import { SOLID_WORLD_BRICK_CELLS, SOLID_WORLD_TERRAIN_MATERIAL_ID, sampleSolidWorld,
  type SolidWorld, type SolidWorldCoordinate, type SolidWorldVoxelPatch } from "../../core/solid-world";
import { assertRetainedSceneIsotropicLattice, integrateRetainedSceneDensity, retainedSceneDensityRange,
  type RetainedSceneDensity, type RetainedScenePoint } from "./sparse-cm12-retained-scene-density";

export interface RetainedSolidEditDelta {
  readonly fieldGeneration: number;
  /** Compact planes: entry j belongs to full-domain cell indices[j]. */
  readonly indices: Uint32Array;
  readonly previousSolidFractions: Uint8Array;
  readonly solidFractions: Uint8Array;
  readonly effectiveMeans: Float32Array;
  readonly openFractions: Float32Array;
  readonly subcells?: {
    readonly seedAmounts: Float32Array;
    readonly openVolumes: Float32Array;
    readonly estimatedCellAbsoluteErrors: Float64Array;
  };
  readonly changedCellRanges: readonly { readonly firstCell: number; readonly cellCount: number; readonly compactOffset: number }[];
  readonly receipt: { readonly visitedCells: number; readonly changedCells: number;
    readonly integratedBoxes: number; readonly integrationRectangles: number;
    readonly estimatedAbsoluteError: number; readonly maximumEstimatedOpenMeanError: number;
    readonly maximumEstimatedSubcellMeanError: number };
}

export interface RetainedSolidEditOptions {
  /** Immutable unrestricted initial moments; never current transported means. */
  readonly seedMeans: Float32Array;
  readonly rigid?: boolean;
  readonly absoluteMeanTolerance?: number;
  readonly maximumRectangles?: number;
  readonly maximumTotalRectangles?: number;
  readonly maximumVisitedCells?: number;
  readonly maximumChangedCells?: number;
}

function sameRegion(a: SolidWorldVoxelPatch, b: SolidWorldVoxelPatch): boolean {
  return a === b || a.operation === b.operation
    && a.minimum.every((v, axis) => v === b.minimum[axis])
    && a.maximumExclusive.every((v, axis) => v === b.maximumExclusive[axis]);
}

/** Prepare a bounded transaction; never allocate/copy a dense domain snapshot,
 * mutate accepted caches, or touch GPU resources. Canonical ordered regions
 * are evaluated after sparse pages, exactly like the full initial compiler.
 * Unchanged page identity skips all 512 voxel samples. Any failure occurs while
 * this proposal is still private, before the owner can commit its ranges. */
export function compileRetainedSolidEditDelta(field: RetainedSceneDensity,
  dimensions: readonly [number, number, number], cellSize: number,
  previous: SolidWorld, next: SolidWorld, options: RetainedSolidEditOptions): RetainedSolidEditDelta {
  assertRetainedSceneIsotropicLattice(field, dimensions, cellSize);
  const count = dimensions[0] * dimensions[1] * dimensions[2], h = Math.fround(cellSize), volume = h ** 3;
  const tolerance = options.absoluteMeanTolerance ?? 2e-7;
  const visitLimit = options.maximumVisitedCells ?? 262144, changeLimit = options.maximumChangedCells ?? 32768;
  const rectangleLimit = options.maximumRectangles ?? 8192, totalRectangleLimit = options.maximumTotalRectangles ?? 8192;
  if (!Number.isSafeInteger(count) || count > 0xffff_ffff || options.seedMeans.length !== count
    || !(tolerance > 0) || !Number.isFinite(tolerance)
    || [visitLimit, changeLimit, rectangleLimit, totalRectangleLimit].some(n => !Number.isSafeInteger(n) || n < 1)) {
    throw new Error("Invalid retained solid edit lattice, seed moments or work budget");
  }
  const candidates = new Set<number>();
  let visitedCells = 0;
  const visitBox = (minimum: readonly number[], maximum: readonly number[]) => {
    const lo = minimum.map((v, axis) => Math.max(0, Math.min(dimensions[axis]!, v)));
    const hi = maximum.map((v, axis) => Math.max(0, Math.min(dimensions[axis]!, v)));
    const visits = hi.reduce((product, v, axis) => product * Math.max(0, v - lo[axis]!), 1);
    if (!Number.isSafeInteger(visits) || visitedCells + visits > visitLimit) {
      throw new RangeError("Live solid edit exceeds its bounded retained-geometry work budget; use a smaller edit.");
    }
    visitedCells += visits;
    for (let z = lo[2]!; z < hi[2]!; z++) for (let y = lo[1]!; y < hi[1]!; y++) for (let x = lo[0]!; x < hi[0]!; x++) {
      candidates.add(x + dimensions[0] * (y + dimensions[1] * z));
    }
  };
  const visitPage = (coordinate: SolidWorldCoordinate) => {
    const lo = coordinate.map(v => v * SOLID_WORLD_BRICK_CELLS);
    visitBox(lo, lo.map(v => v + SOLID_WORLD_BRICK_CELLS));
  };
  for (const page of next.pages) {
    const oldIndex = previous.directory.lookup(page.coordinate);
    if (oldIndex === undefined || previous.pages[oldIndex] !== page) visitPage(page.coordinate);
  }
  for (const page of previous.pages) if (next.directory.lookup(page.coordinate) === undefined) visitPage(page.coordinate);
  const oldRegions = previous.regions ?? [], newRegions = next.regions ?? [];
  let common = 0;
  while (common < oldRegions.length && common < newRegions.length && sameRegion(oldRegions[common]!, newRegions[common]!)) common++;
  for (const region of [...oldRegions.slice(common), ...newRegions.slice(common)]) visitBox(region.minimum, region.maximumExclusive);
  const changes: { index: number; point: RetainedScenePoint; oldQ8: number; q8: number }[] = [];
  for (const index of candidates) {
    const x = index % dimensions[0], yz = Math.floor(index / dimensions[0]);
    const point: RetainedScenePoint = [x, yz % dimensions[1], Math.floor(yz / dimensions[1])];
    const old = sampleSolidWorld(previous, point), current = sampleSolidWorld(next, point);
    const oldQ8 = Math.round(old.solidFraction * 255), q8 = Math.round(current.solidFraction * 255);
    if (q8 > 0 && q8 < 255 && current.materialId !== SOLID_WORLD_TERRAIN_MATERIAL_ID) {
      throw new Error(`Fractional SolidWorld voxel ${index} has no declared subvoxel geometry (material ${current.materialId})`);
    }
    if (oldQ8 === q8) continue;
    if (changes.length >= changeLimit) throw new RangeError("Live solid edit changes too many retained cells; use a smaller edit.");
    changes.push({ index, point, oldQ8, q8 });
  }
  changes.sort((a, b) => a.index - b.index);
  const indices = new Uint32Array(changes.length), previousSolidFractions = new Uint8Array(changes.length);
  const solidFractions = new Uint8Array(changes.length), effectiveMeans = new Float32Array(changes.length), openFractions = new Float32Array(changes.length);
  const subcells = options.rigid ? { seedAmounts: new Float32Array(8 * changes.length),
    openVolumes: new Float32Array(8 * changes.length), estimatedCellAbsoluteErrors: new Float64Array(changes.length) } : undefined;
  let integratedBoxes = 0, integrationRectangles = 0, estimatedAbsoluteError = 0;
  let maximumEstimatedOpenMeanError = 0, maximumEstimatedSubcellMeanError = 0;
  const integrate = (lower: RetainedScenePoint, upper: RetainedScenePoint, fraction: number) => {
    const remaining = totalRectangleLimit - integrationRectangles;
    if (remaining < 1) throw new RangeError("Live solid edit exceeds its retained integration work budget; use a smaller edit.");
    const result = integrateRetainedSceneDensity(field, { lower, upper }, {
      absoluteTolerance: volume * fraction * tolerance, maximumRectangles: Math.min(rectangleLimit, remaining) });
    if (!result.toleranceMet) throw new Error("Retained live solid integral did not converge within its unchanged numerical tolerance.");
    integratedBoxes++; integrationRectangles += result.rectangles; estimatedAbsoluteError += result.estimatedAbsoluteError;
    return result;
  };
  for (let j = 0; j < changes.length; j++) {
    const { index, point, oldQ8, q8 } = changes[j]!;
    indices[j] = index; previousSolidFractions[j] = oldQ8; solidFractions[j] = q8;
    const origin = point.map((v, axis) => field.domain.lower[axis]! + v * h) as unknown as RetainedScenePoint;
    const upper = origin.map(v => v + h) as unknown as RetainedScenePoint;
    const open = (255 - q8) / 255; openFractions[j] = open;
    if (q8 === 0) effectiveMeans[j] = options.seedMeans[index]!;
    else if (q8 !== 255) {
      const result = integrate([origin[0], origin[1] + q8 / 255 * h, origin[2]], upper, open);
      effectiveMeans[j] = result.amount / volume;
      maximumEstimatedOpenMeanError = Math.max(maximumEstimatedOpenMeanError, result.estimatedAbsoluteError / (volume * open));
    }
    if (!subcells || q8 === 255) continue;
    const range = retainedSceneDensityRange(field, { lower: origin, upper });
    for (let octant = 0; octant < 8; octant++) {
      const bottom = Math.max(octant & 2 ? .5 : 0, q8 / 255), top = octant & 2 ? 1 : .5;
      const fraction = Math.max(0, top - bottom) / 4, offset = 8 * j + octant;
      subcells.openVolumes[offset] = fraction;
      if (fraction === 0 || range[1] === 0 || range[0] === 1) {
        subcells.seedAmounts[offset] = range[0] === 1 ? fraction : 0; continue;
      }
      const lower: RetainedScenePoint = [origin[0] + (octant & 1 ? .5 : 0) * h, origin[1] + bottom * h, origin[2] + (octant & 4 ? .5 : 0) * h];
      const hi: RetainedScenePoint = [origin[0] + (octant & 1 ? 1 : .5) * h, origin[1] + top * h, origin[2] + (octant & 4 ? 1 : .5) * h];
      const result = integrate(lower, hi, fraction);
      subcells.seedAmounts[offset] = result.amount / volume;
      subcells.estimatedCellAbsoluteErrors[j] += result.estimatedAbsoluteError;
      maximumEstimatedSubcellMeanError = Math.max(maximumEstimatedSubcellMeanError, result.estimatedAbsoluteError / (volume * fraction));
    }
  }
  const changedCellRanges: { firstCell: number; cellCount: number; compactOffset: number }[] = [];
  for (let j = 0; j < indices.length; j++) {
    const last = changedCellRanges.at(-1);
    if (last && last.firstCell + last.cellCount === indices[j]) last.cellCount++;
    else changedCellRanges.push({ firstCell: indices[j]!, cellCount: 1, compactOffset: j });
  }
  return { fieldGeneration: field.generation, indices, previousSolidFractions, solidFractions, effectiveMeans, openFractions,
    subcells, changedCellRanges, receipt: { visitedCells, changedCells: changes.length, integratedBoxes,
      integrationRectangles, estimatedAbsoluteError, maximumEstimatedOpenMeanError, maximumEstimatedSubcellMeanError } };
}
