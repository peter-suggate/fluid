/**
 * CPU reference storage for an arbitrary sparse atlas of equal-world dyadic
 * bricks. This is a topology/numerics oracle, not the eventual GPU page pool.
 * Empty bricks have no payload and no directory entry.
 */

import {
  damBreakBoxContains,
  initialFluidBrickCoordinates,
  initialFluidSeedBrickCoordinates,
  initialLiquidFractionAtCell,
  sceneDamBreakBox,
  sceneInitialLiquidVolumes,
} from "../../core/initial-fluid";
import type { SceneDescription } from "../../core/model";
import { sceneRefinementRegions } from "../../core/refinement-regions";
import { sampleSolidWorld, solidWorldForScene, SOLID_WORLD_BRICK_CELLS,
  type SolidWorld } from
  "../../core/solid-world";
import {
  applySparseCM12RefinementRegionResolutionBounds,
  packSparseCM12RefinementRegions,
  sparseCM12RefinementRegionResolutionBoundsForBrick,
} from "./sparse-cm12-refinement-regions";

/** Supported construction-time finest resolution of one fixed-world brick. */
export type SparseBrickFineResolution = 4 | 8 | 16;
/** Any rung in the union of the supported complete dyadic ladders. */
export type SparseBrickResolution = 1 | 2 | 4 | 8 | 16;
export type SparseBrickVec3 = readonly [number, number, number];

export interface SparseBrickLadder {
  readonly fineResolution: SparseBrickFineResolution;
  readonly coarseResolution: SparseBrickResolution;
  readonly resolutions: readonly SparseBrickResolution[];
  readonly cellCapacity: number;
}

export const DEFAULT_BRICK_FINE_RESOLUTION: SparseBrickFineResolution = 8;
/** @deprecated Use atlas.brickFineResolution for geometry. */
export const BRICK_FINE_RESOLUTION = DEFAULT_BRICK_FINE_RESOLUTION;

const SPARSE_BRICK_LADDERS: Readonly<Record<SparseBrickFineResolution, SparseBrickLadder>> =
  Object.freeze({
    4: Object.freeze({ fineResolution: 4, coarseResolution: 2,
      resolutions: Object.freeze([1, 2, 4] as const), cellCapacity: 4 ** 3 }),
    8: Object.freeze({ fineResolution: 8, coarseResolution: 4,
      resolutions: Object.freeze([1, 2, 4, 8] as const), cellCapacity: 8 ** 3 }),
    16: Object.freeze({ fineResolution: 16, coarseResolution: 8,
      resolutions: Object.freeze([1, 2, 4, 8, 16] as const), cellCapacity: 16 ** 3 }),
  });

export function sparseBrickLadder(
  fineResolution: SparseBrickFineResolution = DEFAULT_BRICK_FINE_RESOLUTION,
): SparseBrickLadder {
  const ladder = SPARSE_BRICK_LADDERS[fineResolution];
  if (!ladder) throw new RangeError("brickFineResolution must be 4, 8, or 16");
  return ladder;
}

export function isSparseBrickResolution(
  resolution: number,
  fineResolution: SparseBrickFineResolution,
): resolution is SparseBrickResolution {
  return sparseBrickLadder(fineResolution).resolutions.includes(resolution as SparseBrickResolution);
}

export interface SparseAdaptiveMassBrick {
  readonly key: number;
  /** Minimum coordinate in the atlas-selected fixed-brick lattice. */
  readonly coordinate: SparseBrickVec3;
  /** Dyadic edge length in fixed bricks. Omitted means one legacy brick. */
  readonly spanBricks?: number;
  readonly resolution: SparseBrickResolution;
  /** Intensive CM12 surface density, x-major within the brick. */
  readonly density: Float64Array;
  readonly gamma: Float64Array;
}

export interface SparseAdaptiveMassAtlas {
  readonly dimensions: SparseBrickVec3;
  readonly brickFineResolution: SparseBrickFineResolution;
  /** Stable per-brick cell-id stride; equal to brickFineResolution^3. */
  readonly brickCellCapacity: number;
  readonly ladder: SparseBrickLadder;
  readonly brickDimensions: SparseBrickVec3;
  readonly bricks: readonly SparseAdaptiveMassBrick[];
  readonly directory: ReadonlyMap<number, SparseAdaptiveMassBrick>;
  /** Exact-origin directories, one per dyadic span; never expanded by volume. */
  readonly directoriesBySpan: ReadonlyMap<number, ReadonlyMap<number, SparseAdaptiveMassBrick>>;
  readonly maximumSpanBricks: number;
  readonly generation: number;
}

export interface SparseBrickAtlasInitializationOptions {
  readonly finestDimensions: SparseBrickVec3;
  readonly brickFineResolution?: SparseBrickFineResolution;
  /** Optional caller-owned construction guard; Sparse CM12 itself has no cell-count cap. */
  readonly maximumFinestCells?: number;
  /** Largest dyadic macro edge, in ordinary bricks. One disables macro leaves. */
  readonly maximumMacroSpanBricks?: number;
  readonly emptyEpsilon?: number;
  /** Count of occupied face-distance rings held at the finest rung around the interface. */
  readonly surfaceFineRings?: number;
  /** Optional fixed-policy override for every nonempty brick, including interfaces. */
  readonly resolutionForBrick?: (input: {
    readonly coordinate: SparseBrickVec3;
    readonly brickDimensions: SparseBrickVec3;
  }) => SparseBrickResolution;
}

export interface SparseBrickAtlasStats {
  readonly generation: number;
  readonly logicalBrickCount: number;
  readonly residentBrickCount: number;
  readonly omittedEmptyBrickCount: number;
  readonly fineBrickCount: number;
  readonly coarseBrickCount: number;
  /** Unique resident 8^3-to-4^3 brick pairs sharing a complete brick face. */
  readonly fineCoarseFaceConnectedPairCount: number;
  readonly leafCount: number;
  readonly equivalentFinestCellCount: number;
  readonly leafCompressionRatio: number;
  readonly integratedMassFineCells: number;
}

export interface SparseAtlasLeaf {
  readonly id: number;
  readonly brickKey: number;
  readonly brickResolution: SparseBrickResolution;
  readonly localIndex: number;
  readonly centerFine: SparseBrickVec3;
  /** Cell measure in finest-cell-volume units, clipped at domain edges. */
  readonly volumeFineCells: number;
  readonly density: number;
  readonly gamma: number;
}

export function sparseBrickSpan(brick: SparseAdaptiveMassBrick): number {
  return brick.spanBricks ?? 1;
}

function positiveDimensions(value: SparseBrickVec3): void {
  for (const component of value) {
    if (!Number.isSafeInteger(component) || component <= 0) {
      throw new RangeError("finestDimensions must contain positive integers");
    }
  }
}

export function sparseBrickKey(
  coordinate: SparseBrickVec3,
  brickDimensions: SparseBrickVec3,
): number {
  return coordinate[0] + brickDimensions[0]
    * (coordinate[1] + brickDimensions[1] * coordinate[2]);
}

export function sparseBrickCoordinate(
  key: number,
  brickDimensions: SparseBrickVec3,
): [number, number, number] {
  const xy = brickDimensions[0] * brickDimensions[1];
  const z = Math.floor(key / xy);
  const remainder = key - z * xy;
  const y = Math.floor(remainder / brickDimensions[0]);
  return [remainder - y * brickDimensions[0], y, z];
}

export function createSparseAdaptiveMassAtlas(
  dimensions: SparseBrickVec3,
  bricks: readonly SparseAdaptiveMassBrick[],
  generation = 1,
  brickFineResolution: SparseBrickFineResolution = DEFAULT_BRICK_FINE_RESOLUTION,
): SparseAdaptiveMassAtlas {
  positiveDimensions(dimensions);
  const ladder = sparseBrickLadder(brickFineResolution);
  const brickDimensions = dimensions.map((value) =>
    Math.ceil(value / brickFineResolution)) as [number, number, number];
  const directory = new Map<number, SparseAdaptiveMassBrick>();
  const directoriesBySpan = new Map<number, Map<number, SparseAdaptiveMassBrick>>();
  let maximumSpanBricks = 1;
  for (const brick of bricks) {
    const span = sparseBrickSpan(brick);
    if (!Number.isSafeInteger(span) || span < 1 || (span & (span - 1)) !== 0) {
      throw new RangeError(`brick ${brick.key} span must be a positive power of two`);
    }
    if (brick.coordinate.some((value) => value % span !== 0)) {
      throw new Error(`brick ${brick.key} origin must be aligned to span ${span}`);
    }
    if (!isSparseBrickResolution(brick.resolution, brickFineResolution)) {
      throw new RangeError(`brick ${brick.key} resolution is not on the ${ladder.resolutions.join("/")} ladder`);
    }
    const count = brick.resolution ** 3;
    if (brick.density.length !== count || brick.gamma.length !== count) {
      throw new RangeError(`brick ${brick.key} payload does not match ${brick.resolution}^3`);
    }
    if (brick.key !== sparseBrickKey(brick.coordinate, brickDimensions)) {
      throw new Error(`brick ${brick.key} coordinate/key mismatch`);
    }
    if (directory.has(brick.key)) throw new Error(`duplicate sparse brick ${brick.key}`);
    directory.set(brick.key, brick);
    let spanDirectory = directoriesBySpan.get(span);
    if (!spanDirectory) directoriesBySpan.set(span, spanDirectory = new Map());
    spanDirectory.set(brick.key, brick);
    maximumSpanBricks = Math.max(maximumSpanBricks, span);
  }
  // Accepted atlases are strongly graded: every resident face adjacency is
  // same-level or one rung apart across the selected complete dyadic ladder.
  for (const brick of directory.values()) for (let axis = 0; axis < 3; axis += 1) {
    const coordinate = [...brick.coordinate] as [number, number, number];
    coordinate[axis] += 1;
    if (coordinate[axis] >= brickDimensions[axis]) continue;
    const neighbor = directory.get(sparseBrickKey(coordinate, brickDimensions));
    if (neighbor && Math.max(brick.resolution, neighbor.resolution)
      / Math.min(brick.resolution, neighbor.resolution) > 2) {
      throw new Error(`brick face ${brick.key}/${neighbor.key} exceeds 2:1 grading`);
    }
  }
  return {
    dimensions, brickFineResolution, brickCellCapacity: ladder.cellCapacity,
    ladder, brickDimensions, bricks: [...bricks], directory,
    directoriesBySpan, maximumSpanBricks, generation,
  };
}

/** Resolve the leaf containing one fixed-brick coordinate in O(log domain). */
export function sparseBrickContainingCoordinate(
  atlas: SparseAdaptiveMassAtlas,
  coordinate: SparseBrickVec3,
): SparseAdaptiveMassBrick | undefined {
  if (coordinate.some((value, axis) => value < 0 || value >= atlas.brickDimensions[axis])) {
    return undefined;
  }
  for (let span = 1; span <= atlas.maximumSpanBricks; span *= 2) {
    const origin = coordinate.map((value) => Math.floor(value / span) * span) as
      [number, number, number];
    const brick = atlas.directoriesBySpan.get(span)?.get(
      sparseBrickKey(origin, atlas.brickDimensions),
    );
    if (brick) return brick;
  }
  return undefined;
}

function initialDensityAt(
  scene: SceneDescription,
  dimensions: SparseBrickVec3,
  x: number,
  y: number,
  z: number,
): number {
  const [nx, ny, nz] = dimensions;
  if (scene.systems?.fluid === false
    || x < 0 || y < 0 || z < 0 || x >= nx || y >= ny || z >= nz) return 0;
  const dam = sceneDamBreakBox(scene);
  const baseWet = scene.fluid.initialCondition === "tank-fill"
    ? (y + 0.5) / ny <= scene.container.fillFraction
    : damBreakBoxContains(dam, (x + 0.5) / nx, (y + 0.5) / ny, (z + 0.5) / nz);
  let world = initialSolidWorldCache.get(scene);
  if (!world) {
    world = solidWorldForScene(scene);
    initialSolidWorldCache.set(scene, world);
  }
  return (1 - sampleSolidWorld(world, [x, y, z]).solidFraction)
    * initialLiquidFractionAtCell(scene, x, y, z, dimensions, baseWet);
}

const initialSolidWorldCache = new WeakMap<SceneDescription, SolidWorld>();

/** Maximum fine-field RMS discarded by one static-solid restriction rung. */
export const SPARSE_CM12_SOLID_RESTRICTION_TOLERANCE = 0.08;

function solidWorldMayAffectBrick(
  world: SolidWorld,
  origin: SparseBrickVec3,
  brickFineResolution: SparseBrickFineResolution,
): boolean {
  const minimum = origin.map((value) => value - 1) as [number, number, number];
  const maximumExclusive = origin.map((value) =>
    value + brickFineResolution + 1) as [number, number, number];
  for (const region of world.regions ?? []) {
    if (region.minimum.every((value, axis) =>
      value < maximumExclusive[axis]!
        && region.maximumExclusive[axis]! > minimum[axis]!)) return true;
  }
  const pageMinimum = minimum.map((value) =>
    Math.floor(value / SOLID_WORLD_BRICK_CELLS));
  const pageMaximum = maximumExclusive.map((value) =>
    Math.floor((value - 1) / SOLID_WORLD_BRICK_CELLS));
  for (let pageZ = pageMinimum[2]!; pageZ <= pageMaximum[2]!; pageZ += 1)
    for (let pageY = pageMinimum[1]!; pageY <= pageMaximum[1]!; pageY += 1)
      for (let pageX = pageMinimum[0]!; pageX <= pageMaximum[0]!; pageX += 1) {
        const page = world.directory.lookup([pageX, pageY, pageZ]);
        if (page !== undefined && world.pages[page]!.solidFraction.some(
          (fraction) => fraction !== 0)) return true;
      }
  return false;
}

function sparseCM12StaticSolidSampleCube(
  world: SolidWorld,
  origin: SparseBrickVec3,
  brickFineResolution: SparseBrickFineResolution,
): Float64Array | null {
  if (!solidWorldMayAffectBrick(world, origin, brickFineResolution)) return null;
  const width = brickFineResolution + 2;
  const samples = new Float64Array(width ** 3);
  for (let z = -1; z <= brickFineResolution; z += 1)
    for (let y = -1; y <= brickFineResolution; y += 1)
      for (let x = -1; x <= brickFineResolution; x += 1) {
        samples[(x + 1) + width * ((y + 1) + width * (z + 1))] =
          sampleSolidWorld(world, [origin[0] + x, origin[1] + y,
            origin[2] + z]).solidFraction;
      }
  return samples;
}

/**
 * Measure the SolidWorld information discarded by direct restriction to one
 * candidate rung. Cell occupancy and each oriented face aperture are compared
 * only with their own conservative macro average. An axis-aligned plane is
 * therefore exact, as is the intersection of two or three such planes: the
 * different wall orientations are never incorrectly averaged together.
 */
export function sparseCM12StaticSolidRestrictionError(
  scene: SceneDescription,
  dimensions: SparseBrickVec3,
  coordinate: SparseBrickVec3,
  brickFineResolution: SparseBrickFineResolution,
  candidateResolution: SparseBrickResolution,
  cachedSamples?: Float64Array | null,
): number {
  if (candidateResolution >= brickFineResolution) return 0;
  if (brickFineResolution % candidateResolution !== 0) {
    throw new RangeError("candidateResolution must divide brickFineResolution");
  }
  let world = initialSolidWorldCache.get(scene);
  if (!world) {
    world = solidWorldForScene(scene);
    initialSolidWorldCache.set(scene, world);
  }
  const origin = coordinate.map((value) => value * brickFineResolution) as
    [number, number, number];
  const samples = cachedSamples === undefined
    ? sparseCM12StaticSolidSampleCube(world, origin, brickFineResolution)
    : cachedSamples;
  if (samples === null) return 0;
  const sample = (x: number, y: number, z: number) => {
    const width = brickFineResolution + 2;
    return samples[(x + 1) + width * ((y + 1) + width * (z + 1))]!;
  };
  const span = brickFineResolution / candidateResolution;
  let maximumError = 0;
  const rms = (sum: number, squareSum: number, count: number) => {
    const mean = sum / count;
    return Math.sqrt(Math.max(0, squareSum / count - mean * mean));
  };

  for (let macroZ = 0; macroZ < candidateResolution; macroZ += 1)
    for (let macroY = 0; macroY < candidateResolution; macroY += 1)
      for (let macroX = 0; macroX < candidateResolution; macroX += 1) {
        let sum = 0, squareSum = 0, count = 0;
        for (let z = 0; z < span; z += 1)
          for (let y = 0; y < span; y += 1)
            for (let x = 0; x < span; x += 1) {
              const value = sample(macroX * span + x,
                macroY * span + y, macroZ * span + z);
              sum += value; squareSum += value * value; count += 1;
            }
        maximumError = Math.max(maximumError, rms(sum, squareSum, count));
  }

  for (let axis = 0; axis < 3; axis += 1) {
    for (let face = 0; face <= candidateResolution; face += 1)
      for (let macroV = 0; macroV < candidateResolution; macroV += 1)
        for (let macroU = 0; macroU < candidateResolution; macroU += 1) {
          let sum = 0, squareSum = 0, count = 0;
          for (let v = 0; v < span; v += 1)
            for (let u = 0; u < span; u += 1) {
              const plane = face * span;
              const uCoordinate = macroU * span + u;
              const vCoordinate = macroV * span + v;
              let px = uCoordinate, py = vCoordinate, pz = plane;
              if (axis === 0) {
                px = plane; py = uCoordinate; pz = vCoordinate;
              } else if (axis === 1) {
                px = vCoordinate; py = plane; pz = uCoordinate;
              }
              const value = 1 - Math.max(
                sample(px - Number(axis === 0), py - Number(axis === 1),
                  pz - Number(axis === 2)), sample(px, py, pz),
              );
              sum += value; squareSum += value * value; count += 1;
            }
          maximumError = Math.max(maximumError, rms(sum, squareSum, count));
        }
  }
  return maximumError;
}

export function sparseCM12StaticSolidResolutionFloor(
  scene: SceneDescription,
  dimensions: SparseBrickVec3,
  coordinate: SparseBrickVec3,
  brickFineResolution: SparseBrickFineResolution,
): SparseBrickResolution {
  const ladder = sparseBrickLadder(brickFineResolution);
  let world = initialSolidWorldCache.get(scene);
  if (!world) {
    world = solidWorldForScene(scene);
    initialSolidWorldCache.set(scene, world);
  }
  const origin = coordinate.map((value) => value * brickFineResolution) as
    [number, number, number];
  const samples = sparseCM12StaticSolidSampleCube(
    world, origin, brickFineResolution,
  );
  if (samples === null) return 1;
  for (const resolution of ladder.resolutions) {
    if (resolution === brickFineResolution) break;
    if (sparseCM12StaticSolidRestrictionError(scene, dimensions, coordinate,
      brickFineResolution, resolution, samples)
      <= SPARSE_CM12_SOLID_RESTRICTION_TOLERANCE) {
      return resolution;
    }
  }
  return brickFineResolution;
}

function brickHasInterface(
  scene: SceneDescription,
  dimensions: SparseBrickVec3,
  coordinate: SparseBrickVec3,
  epsilon: number,
  brickFineResolution: SparseBrickFineResolution,
): boolean {
  const origin = coordinate.map((value) => value * brickFineResolution) as number[];
  for (let z = 0; z < brickFineResolution; z += 1)
    for (let y = 0; y < brickFineResolution; y += 1)
      for (let x = 0; x < brickFineResolution; x += 1) {
        const gx = origin[0] + x, gy = origin[1] + y, gz = origin[2] + z;
        if (gx >= dimensions[0] || gy >= dimensions[1] || gz >= dimensions[2]) continue;
        const own = initialDensityAt(scene, dimensions, gx, gy, gz);
        if (own > epsilon && own < 1 - epsilon) return true;
        for (const [dx, dy, dz] of [[-1, 0, 0], [1, 0, 0], [0, -1, 0],
          [0, 1, 0], [0, 0, -1], [0, 0, 1]] as const) {
          if (gx + dx < 0 || gy + dy < 0 || gz + dz < 0
            || gx + dx >= dimensions[0] || gy + dy >= dimensions[1]
            || gz + dz >= dimensions[2]) continue;
          const neighbor = initialDensityAt(
            scene, dimensions, gx + dx, gy + dy, gz + dz,
          );
          if ((own > epsilon) !== (neighbor > epsilon)) return true;
        }
      }
  return false;
}

function initialBrick(
  scene: SceneDescription,
  dimensions: SparseBrickVec3,
  coordinate: SparseBrickVec3,
  resolution: SparseBrickResolution,
  brickFineResolution: SparseBrickFineResolution,
): SparseAdaptiveMassBrick {
  const factor = brickFineResolution / resolution;
  const density = new Float64Array(resolution ** 3);
  const gamma = new Float64Array(resolution ** 3).fill(1);
  for (let z = 0; z < resolution; z += 1)
    for (let y = 0; y < resolution; y += 1)
      for (let x = 0; x < resolution; x += 1) {
        let rho = 0, count = 0;
        for (let dz = 0; dz < factor; dz += 1)
          for (let dy = 0; dy < factor; dy += 1)
            for (let dx = 0; dx < factor; dx += 1) {
              const gx = coordinate[0] * brickFineResolution + x * factor + dx;
              const gy = coordinate[1] * brickFineResolution + y * factor + dy;
              const gz = coordinate[2] * brickFineResolution + z * factor + dz;
              if (gx >= dimensions[0] || gy >= dimensions[1] || gz >= dimensions[2]) continue;
              rho += initialDensityAt(scene, dimensions, gx, gy, gz);
              count += 1;
            }
        density[x + resolution * (y + resolution * z)] = count > 0 ? rho / count : 0;
      }
  const brickDimensions: SparseBrickVec3 = [
    Math.ceil(dimensions[0] / brickFineResolution),
    Math.ceil(dimensions[1] / brickFineResolution),
    Math.ceil(dimensions[2] / brickFineResolution),
  ];
  return {
    key: sparseBrickKey(coordinate, brickDimensions),
    coordinate,
    resolution,
    density,
    gamma,
  };
}

function uniformInitialBrick(
  coordinate: SparseBrickVec3,
  spanBricks: number,
  resolution: SparseBrickResolution,
  brickDimensions: SparseBrickVec3,
): SparseAdaptiveMassBrick {
  return {
    key: sparseBrickKey(coordinate, brickDimensions),
    coordinate,
    spanBricks,
    resolution,
    density: new Float64Array(resolution ** 3).fill(1),
    gamma: new Float64Array(resolution ** 3).fill(1),
  };
}

/**
 * Structural span-one leaves whose topology may become fluid-active later.
 *
 * Besides authored liquid seeds, retain the mixed (solid/open) pages of the
 * canonical SolidWorld.  GPU-grown pages deliberately own only a fixed-B8
 * graph; if a dry wall page is omitted here, later wetting can never publish
 * the ordinary 8/4/2/1 ladder.  Mixed pages are surface-shaped geometry, so
 * this does not turn the sparse world into a domain-volume catalogue.
 */
function structuralSeedBrickCoordinates(
  scene: SceneDescription,
  dimensions: SparseBrickVec3,
  brickDimensions: SparseBrickVec3,
  brickFineResolution: SparseBrickFineResolution,
): SparseBrickVec3[] {
  const maximumMixedSolidPages = 2048;
  const authoredWidth = scene.voxelDomain.brickSize_cells;
  const result = new Map<number, SparseBrickVec3>();
  for (const authored of initialFluidSeedBrickCoordinates(scene, dimensions, authoredWidth)) {
    const lower = authored.map((value) => Math.floor(
      value * authoredWidth / brickFineResolution,
    )) as [number, number, number];
    const upper = authored.map((value, axis) => Math.min(brickDimensions[axis], Math.ceil(
      Math.min(dimensions[axis], (value + 1) * authoredWidth) / brickFineResolution,
    ))) as [number, number, number];
    for (let z = lower[2]; z < upper[2]; z += 1)
      for (let y = lower[1]; y < upper[1]; y += 1)
        for (let x = lower[0]; x < upper[0]; x += 1) {
          const coordinate = [x, y, z] as const;
          result.set(sparseBrickKey(coordinate, brickDimensions), coordinate);
        }
  }
  let world = initialSolidWorldCache.get(scene);
  if (!world) {
    world = solidWorldForScene(scene);
    initialSolidWorldCache.set(scene, world);
  }
  const mixedSolidCoordinates = new Map<number, SparseBrickVec3>();
  let mixedSolidCatalogueFits = true;
  for (const page of world.pages) {
    let hasSolid = false, hasOpen = false;
    for (const fraction of page.solidFraction) {
      hasSolid ||= fraction > 0;
      hasOpen ||= fraction < 255;
      if (hasSolid && hasOpen) break;
    }
    if (!hasSolid || !hasOpen) continue;
    const pageOffsets = [[0, 0, 0], [-1, 0, 0], [1, 0, 0], [0, -1, 0],
      [0, 1, 0], [0, 0, -1], [0, 0, 1]] as const;
    for (const offset of pageOffsets) {
      const fineLower = page.coordinate.map((value, axis) =>
        (value + offset[axis]!) * SOLID_WORLD_BRICK_CELLS) as
          [number, number, number];
      const fineUpper = fineLower.map((value) =>
        value + SOLID_WORLD_BRICK_CELLS) as [number, number, number];
      const lower = fineLower.map((value) => Math.floor(
        value / brickFineResolution)) as [number, number, number];
      const upper = fineUpper.map((value, axis) => Math.min(brickDimensions[axis],
        Math.ceil(Math.min(dimensions[axis], value) / brickFineResolution))) as
          [number, number, number];
      for (let z = Math.max(0, lower[2]); z < upper[2]; z += 1)
        for (let y = Math.max(0, lower[1]); y < upper[1]; y += 1)
          for (let x = Math.max(0, lower[0]); x < upper[0]; x += 1) {
            const coordinate = [x, y, z] as const;
            mixedSolidCoordinates.set(
              sparseBrickKey(coordinate, brickDimensions), coordinate,
            );
            if (mixedSolidCoordinates.size > maximumMixedSolidPages) {
              // The resident all-rung template ABI has the same 2,048-leaf
              // ceiling.  Beyond it, adding dry structural leaves would disable
              // every host candidate slot while making initialization scale with
              // the remote boundary. Keep the ordinary SparseWorld growth path.
              mixedSolidCatalogueFits = false;
              break;
            }
          }
      if (!mixedSolidCatalogueFits) break;
    }
    if (!mixedSolidCatalogueFits) break;
  }
  if (mixedSolidCatalogueFits) for (const [key, coordinate] of mixedSolidCoordinates) {
    result.set(key, coordinate);
  }
  return [...result.values()];
}

function initialResolutionWithRefinementRegionBounds(
  packed: ArrayBuffer,
  dimensions: SparseBrickVec3,
  brickCoordinate: SparseBrickVec3,
  spanBricks: number,
  requestedResolution: number,
  brickFineResolution: SparseBrickFineResolution,
): SparseBrickResolution {
  const origin = brickCoordinate.map((value) => value * brickFineResolution) as
    [number, number, number];
  const extent = origin.map((value, axis) => Math.max(0,
    Math.min(spanBricks * brickFineResolution, dimensions[axis]! - value))) as
    [number, number, number];
  const bounds = sparseCM12RefinementRegionResolutionBoundsForBrick(
    packed, origin, extent, brickFineResolution, spanBricks * brickFineResolution);
  const bounded = applySparseCM12RefinementRegionResolutionBounds(
    requestedResolution, bounds);
  if (!isSparseBrickResolution(bounded, brickFineResolution)) {
    throw new Error(`refinement region selected invalid Sparse CM12 rung ${bounded}`);
  }
  return bounded;
}

/**
 * Cover an analytic tank fill with maximal graded octree leaves. The traversal
 * visits octree boundary nodes, not every wet fixed brick. Partial top bricks
 * are deliberately left to a surface-area pass because their payload is cut.
 */
function hierarchicalTankFillBricks(
  scene: SceneDescription,
  dimensions: SparseBrickVec3,
  brickDimensions: SparseBrickVec3,
  brickFineResolution: SparseBrickFineResolution,
  surfaceFineRings: number,
  maximumMacroSpanBricks: number,
  refinementRegionParameters: ArrayBuffer,
): SparseAdaptiveMassBrick[] | undefined {
  if (scene.systems?.fluid === false) return [];
  if (scene.fluid.initialCondition !== "tank-fill"
    || initialFluidBrickCoordinates(scene, dimensions, scene.voxelDomain.brickSize_cells)
    || sceneInitialLiquidVolumes(scene).length > 0) {
    return undefined;
  }
  const fullFineY = Math.max(0, Math.min(dimensions[1],
    Math.floor(scene.container.fillFraction * dimensions[1] + 0.5)));
  let world = initialSolidWorldCache.get(scene);
  if (!world) {
    world = solidWorldForScene(scene);
    initialSolidWorldCache.set(scene, world);
  }
  // Macro leaves are valid only for unobstructed liquid bulk. Inspect the one
  // static authority directly; this covers terrain, a spherical shell and any
  // authored voxel edit without naming any of those authoring concepts.
  for (const page of world.pages) for (let local = 0;
    local < page.solidFraction.length; local += 1) {
    if (page.solidFraction[local] === 0) continue;
    const lx = local % SOLID_WORLD_BRICK_CELLS;
    const ly = Math.floor(local / SOLID_WORLD_BRICK_CELLS)
      % SOLID_WORLD_BRICK_CELLS;
    const lz = Math.floor(local / (SOLID_WORLD_BRICK_CELLS ** 2));
    const x = page.coordinate[0] * SOLID_WORLD_BRICK_CELLS + lx;
    const y = page.coordinate[1] * SOLID_WORLD_BRICK_CELLS + ly;
    const z = page.coordinate[2] * SOLID_WORLD_BRICK_CELLS + lz;
    if (x >= 0 && x < dimensions[0] && y >= 0 && y < fullFineY
      && z >= 0 && z < dimensions[2]) return undefined;
  }
  const fullBrickY = Math.floor(fullFineY / brickFineResolution);
  // A cut surface brick is itself distance ring zero. Without this offset the
  // last completely flooded brick was also assigned ring zero, producing two
  // fine structural bands whenever the fill height was not brick-aligned.
  const fractionalSurfaceRing = fullFineY % brickFineResolution === 0 ? 0 : 1;
  const wetMaximum: SparseBrickVec3 = [brickDimensions[0], fullBrickY, brickDimensions[2]];
  const hasFreeSurface = fullFineY < dimensions[1];
  let rootSpan = 1;
  while (rootSpan < Math.max(...brickDimensions)) rootSpan *= 2;
  const bricks: SparseAdaptiveMassBrick[] = [];
  const visit = (origin: [number, number, number], span: number): void => {
    const outside = origin.some((value, axis) => value >= wetMaximum[axis])
      || origin.some((value) => value < 0);
    if (outside) return;
    const inside = origin.every((value, axis) => value + span <= wetMaximum[axis]);
    if (inside) {
      const edgeFine = span * brickFineResolution;
      const clearanceRings = hasFreeSurface
        ? Math.max(0, wetMaximum[1] - origin[1] - span
          - (surfaceFineRings - 1) + fractionalSurfaceRing)
        : Number.POSITIVE_INFINITY;
      // Strong 2:1 grading doubles the admissible cell width once per complete
      // brick ring below the fine surface band: 1, 2, 4, and so on to B. The old
      // linear `clearanceFine / 4` approximation plateaued at 4 for both the
      // second and third submerged rings. On a 40-cell-deep tank that authored
      // the bottom span-two macro at width 4; macro leaves are intentionally
      // immutable at runtime, so selecting Surface distance could never reach
      // the valid bottom rung visible in the UI.
      const allowedCellWidth = hasFreeSurface
        ? Math.min(brickFineResolution, edgeFine, 2 ** clearanceRings)
        : edgeFine;
      // A cubic macro cannot express two vertical distance rungs. Split it
      // while its closest and deepest logical-brick layers require different
      // cell widths; once both saturate at B it is safe and useful to retain
      // the macro across arbitrarily deep bulk.
      const deepestClearanceRings = hasFreeSurface
        ? Math.max(0, wetMaximum[1] - origin[1] - 1
          - (surfaceFineRings - 1) + fractionalSurfaceRing)
        : Number.POSITIVE_INFINITY;
      const deepestAllowedCellWidth = hasFreeSurface
        ? Math.min(brickFineResolution, edgeFine, 2 ** deepestClearanceRings)
        : edgeFine;
      const crossesResolutionBands = span > 1
        && deepestAllowedCellWidth !== allowedCellWidth;
      const requiredResolution = initialResolutionWithRefinementRegionBounds(
        refinementRegionParameters, dimensions, origin, span,
        edgeFine / allowedCellWidth,
        brickFineResolution);
      // A macro at B^3 would refine all tangential directions merely to grade
      // one normal face. Split that last rung into base bricks instead; this
      // keeps the page census surface-shaped and substantially smaller.
      if (!crossesResolutionBands && span <= maximumMacroSpanBricks
        && requiredResolution <= (span > 1
          ? sparseBrickLadder(brickFineResolution).coarseResolution
          : brickFineResolution)) {
        bricks.push(uniformInitialBrick(
          origin, span, requiredResolution as SparseBrickResolution, brickDimensions,
        ));
        return;
      }
    }
    if (span === 1) {
      if (inside) bricks.push(uniformInitialBrick(
        origin, 1, initialResolutionWithRefinementRegionBounds(
          refinementRegionParameters, dimensions, origin, 1,
          brickFineResolution, brickFineResolution), brickDimensions,
      ));
      return;
    }
    const childSpan = span / 2;
    for (let dz = 0; dz <= childSpan; dz += childSpan)
      for (let dy = 0; dy <= childSpan; dy += childSpan)
        for (let dx = 0; dx <= childSpan; dx += childSpan)
          visit([origin[0] + dx, origin[1] + dy, origin[2] + dz], childSpan);
  };
  if (fullBrickY > 0) visit([0, 0, 0], rootSpan);

  // Only a fractional surface sheet is sampled. This is surface-shaped work,
  // independent of the liquid depth.
  if (fullFineY % brickFineResolution !== 0 && fullBrickY < brickDimensions[1]) {
    for (let z = 0; z < brickDimensions[2]; z += 1)
      for (let x = 0; x < brickDimensions[0]; x += 1)
        bricks.push(initialBrick(
          scene, dimensions, [x, fullBrickY, z], initialResolutionWithRefinementRegionBounds(
            refinementRegionParameters, dimensions, [x, fullBrickY, z], 1,
            brickFineResolution, brickFineResolution), brickFineResolution,
        ));
  }

  const provisional = createSparseAdaptiveMassAtlas(
    dimensions, bricks, 1, brickFineResolution,
  );
  for (const coordinate of structuralSeedBrickCoordinates(
    scene, dimensions, brickDimensions, brickFineResolution,
  )) {
    if (sparseBrickContainingCoordinate(provisional, coordinate)) continue;
    bricks.push(initialBrick(
      scene, dimensions, coordinate, initialResolutionWithRefinementRegionBounds(
        refinementRegionParameters, dimensions, coordinate, 1,
        brickFineResolution, brickFineResolution), brickFineResolution,
    ));
  }
  return bricks.sort((left, right) => left.key - right.key);
}

function candidateInitialBrickCoordinates(
  scene: SceneDescription,
  dimensions: SparseBrickVec3,
  brickDimensions: SparseBrickVec3,
  brickFineResolution: SparseBrickFineResolution,
): Iterable<SparseBrickVec3> {
  if (scene.systems?.fluid === false) return [];
  const candidates = new Map<number, SparseBrickVec3>();
  const addBrick = (coordinate: SparseBrickVec3) => {
    if (coordinate.some((value, axis) => value < 0 || value >= brickDimensions[axis])) return;
    candidates.set(sparseBrickKey(coordinate, brickDimensions), coordinate);
  };
  const addCellBounds = (minimum: readonly [number, number, number],
    maximumExclusive: readonly [number, number, number]) => {
    const lower = minimum.map((value) => Math.max(0,
      Math.floor(value / brickFineResolution))) as [number, number, number];
    const upper = maximumExclusive.map((value, axis) => Math.min(brickDimensions[axis],
      Math.ceil(value / brickFineResolution))) as [number, number, number];
    for (let z = lower[2]; z < upper[2]; z += 1)
      for (let y = lower[1]; y < upper[1]; y += 1)
        for (let x = lower[0]; x < upper[0]; x += 1) addBrick([x, y, z]);
  };
  const addNormalizedBounds = (minimum: readonly [number, number, number],
    maximum: readonly [number, number, number]) => {
    if (maximum.some((value, axis) => value <= minimum[axis])) return;
    // Expand by one finest cell because analytic volumes use eight samples at
    // +/-0.4 h around the centre. The local occupancy test below removes the
    // conservative boundary false positives.
    addCellBounds(
      minimum.map((value, axis) => Math.floor(value * dimensions[axis]) - 1) as
        [number, number, number],
      maximum.map((value, axis) => Math.ceil(value * dimensions[axis]) + 1) as
        [number, number, number],
    );
  };

  const authored = initialFluidBrickCoordinates(
    scene, dimensions, scene.voxelDomain.brickSize_cells,
  );
  for (const coordinate of structuralSeedBrickCoordinates(
    scene, dimensions, brickDimensions, brickFineResolution,
  )) addBrick(coordinate);

  // Replacement seeds suppress the procedural base; additive or absent seeds
  // retain it. A tank fill legitimately scales with its wet volume, while a
  // local dam contributes only its own bounded box.
  if (!authored) {
    if (scene.fluid.initialCondition === "tank-fill") {
      addNormalizedBounds([0, 0, 0], [1, scene.container.fillFraction, 1]);
    } else {
      const dam = sceneDamBreakBox(scene);
      addNormalizedBounds([dam.min.x, dam.min.y, dam.min.z],
        [dam.max.x, dam.max.y, dam.max.z]);
    }
  }

  const container = scene.container;
  const normalized = (point: readonly [number, number, number]) => [
    (point[0] + 0.5 * container.width_m) / container.width_m,
    point[1] / container.height_m,
    (point[2] + 0.5 * container.depth_m) / container.depth_m,
  ] as const;
  for (const volume of sceneInitialLiquidVolumes(scene)) {
    const minimum = volume.shape === "box"
      ? [volume.min_m.x, volume.min_m.y, volume.min_m.z] as const
      : volume.shape === "cylinder"
        ? [volume.center_m.x - volume.radius_m, volume.center_m.y - volume.radius_m,
          volume.center_m.z - volume.halfHeight_m] as const
        : [volume.center_m.x - volume.radius_m, volume.center_m.y - volume.radius_m,
          volume.center_m.z - volume.radius_m] as const;
    const maximum = volume.shape === "box"
      ? [volume.max_m.x, volume.max_m.y, volume.max_m.z] as const
      : volume.shape === "cylinder"
        ? [volume.center_m.x + volume.radius_m, volume.center_m.y + volume.radius_m,
          volume.center_m.z + volume.halfHeight_m] as const
        : [volume.center_m.x + volume.radius_m, volume.center_m.y + volume.radius_m,
          volume.center_m.z + volume.radius_m] as const;
    addNormalizedBounds(normalized(minimum), normalized(maximum));
  }
  return [...candidates.entries()].sort((left, right) => left[0] - right[0])
    .map((entry) => entry[1]);
}

/** Build one payload by exact volume averaging from the bounded finest lattice. */
export function sparseBrickFromDense(
  key: number,
  coordinate: SparseBrickVec3,
  resolution: SparseBrickResolution,
  denseDensity: ArrayLike<number>,
  dimensions: SparseBrickVec3,
  denseGamma?: ArrayLike<number>,
  brickFineResolution: SparseBrickFineResolution = DEFAULT_BRICK_FINE_RESOLUTION,
): SparseAdaptiveMassBrick {
  const density = new Float64Array(resolution ** 3);
  const gamma = new Float64Array(resolution ** 3).fill(1);
  const factor = brickFineResolution / resolution;
  for (let z = 0; z < resolution; z += 1)
    for (let y = 0; y < resolution; y += 1)
      for (let x = 0; x < resolution; x += 1) {
        let rho = 0, g = 0, count = 0;
        for (let dz = 0; dz < factor; dz += 1)
          for (let dy = 0; dy < factor; dy += 1)
            for (let dx = 0; dx < factor; dx += 1) {
              const gx = coordinate[0] * brickFineResolution + x * factor + dx;
              const gy = coordinate[1] * brickFineResolution + y * factor + dy;
              const gz = coordinate[2] * brickFineResolution + z * factor + dz;
              if (gx >= dimensions[0] || gy >= dimensions[1] || gz >= dimensions[2]) continue;
              const index = gx + dimensions[0] * (gy + dimensions[1] * gz);
              rho += denseDensity[index];
              g += denseGamma?.[index] ?? 1;
              count += 1;
            }
        const local = x + resolution * (y + resolution * z);
        density[local] = count > 0 ? rho / count : 0;
        gamma[local] = count > 0 ? g / count : 1;
      }
  return { key, coordinate, resolution, density, gamma };
}

export function initializeSparseBrickAtlasFromScene(
  scene: SceneDescription,
  options: SparseBrickAtlasInitializationOptions,
): SparseAdaptiveMassAtlas {
  positiveDimensions(options.finestDimensions);
  const brickFineResolution = options.brickFineResolution
    ?? DEFAULT_BRICK_FINE_RESOLUTION;
  const ladder = sparseBrickLadder(brickFineResolution);
  const maximumMacroSpanBricks = options.maximumMacroSpanBricks
    ?? Number.POSITIVE_INFINITY;
  if (maximumMacroSpanBricks !== Number.POSITIVE_INFINITY
    && (!Number.isSafeInteger(maximumMacroSpanBricks)
      || maximumMacroSpanBricks < 1
      || !Number.isInteger(Math.log2(maximumMacroSpanBricks)))) {
    throw new RangeError("maximumMacroSpanBricks must be a positive power of two");
  }
  const cellCount = options.finestDimensions.reduce((product, value) => product * value, 1);
  if (options.maximumFinestCells !== undefined
    && cellCount > options.maximumFinestCells) {
    throw new RangeError(
      `bounded finest lattice has ${cellCount} cells; caller cap is ${options.maximumFinestCells}`,
    );
  }
  const epsilon = options.emptyEpsilon ?? 1e-12;
  const surfaceFineRings = typeof options.surfaceFineRings === "number"
    && Number.isFinite(options.surfaceFineRings)
    ? Math.min(8, Math.max(1, Math.round(options.surfaceFineRings))) : 1;
  const brickDimensions = options.finestDimensions.map((value) =>
    Math.ceil(value / brickFineResolution)) as [number, number, number];
  const container = scene.container;
  const refinementRegionParameters = packSparseCM12RefinementRegions(
    sceneRefinementRegions(scene), {
      dimensions: options.finestDimensions,
      cellSize_m: [container.width_m / options.finestDimensions[0],
        container.height_m / options.finestDimensions[1],
        container.depth_m / options.finestDimensions[2]],
      origin_m: { x: -0.5 * container.width_m, y: 0, z: -0.5 * container.depth_m },
    });
  if (!options.resolutionForBrick) {
    const hierarchical = hierarchicalTankFillBricks(
      scene, options.finestDimensions, brickDimensions, brickFineResolution, surfaceFineRings,
      maximumMacroSpanBricks, refinementRegionParameters,
    );
    if (hierarchical) {
      return createSparseAdaptiveMassAtlas(
        options.finestDimensions, hierarchical, 1, brickFineResolution,
      );
    }
  }
  const structuralCoordinates = structuralSeedBrickCoordinates(
    scene, options.finestDimensions, brickDimensions, brickFineResolution,
  );
  const structuralKeys = new Set(structuralCoordinates.map((coordinate) =>
    sparseBrickKey(coordinate, brickDimensions)));
  const candidates: Array<{
    readonly coordinate: SparseBrickVec3;
    readonly key: number;
    readonly interfaceBrick: boolean;
  }> = [];
  for (const coordinate of candidateInitialBrickCoordinates(
    scene, options.finestDimensions, brickDimensions, brickFineResolution,
  )) {
    let nonempty = false;
    for (let z = 0; z < brickFineResolution && !nonempty; z += 1)
      for (let y = 0; y < brickFineResolution && !nonempty; y += 1)
        for (let x = 0; x < brickFineResolution; x += 1)
          if (initialDensityAt(
            scene, options.finestDimensions,
            coordinate[0] * brickFineResolution + x,
            coordinate[1] * brickFineResolution + y,
            coordinate[2] * brickFineResolution + z,
          ) > epsilon) {
            nonempty = true; break;
          }
    const key = sparseBrickKey(coordinate, brickDimensions);
    if (!nonempty && !structuralKeys.has(key)) continue;
    candidates.push({
      coordinate,
      key,
      interfaceBrick: brickHasInterface(
        scene, options.finestDimensions, coordinate, epsilon, brickFineResolution,
      ),
    });
  }

  // Seed an exact face-distance transform from the free surface. This makes
  // the initial accepted topology agree with the GPU activity policy's full
  // 8/4/2/1 ladder. Previously every saturated interior brick was fixed at
  // 4^3; later GPU requests for 2^3 and 1^3 remained candidate-only, so deep
  // pools such as ocean-seiche could never visibly or physically coarsen.
  const candidateByKey = new Map(candidates.map((candidate) =>
    [candidate.key, candidate] as const));
  const interfaceDistance = new Map<number, number>();
  const queue: typeof candidates = [];
  for (const candidate of candidates) {
    if (!candidate.interfaceBrick) continue;
    interfaceDistance.set(candidate.key, 0);
    queue.push(candidate);
  }
  const faceDirections = [
    [-1, 0, 0], [1, 0, 0], [0, -1, 0],
    [0, 1, 0], [0, 0, -1], [0, 0, 1],
  ] as const;
  for (let cursor = 0; cursor < queue.length; cursor += 1) {
    const candidate = queue[cursor]!;
    const distance = interfaceDistance.get(candidate.key)!;
    for (const direction of faceDirections) {
      const coordinate = candidate.coordinate.map((value, axis) =>
        value + direction[axis]) as [number, number, number];
      if (coordinate.some((value, axis) =>
        value < 0 || value >= brickDimensions[axis])) continue;
      const key = sparseBrickKey(coordinate, brickDimensions);
      const neighbor = candidateByKey.get(key);
      if (!neighbor || interfaceDistance.has(key)) continue;
      interfaceDistance.set(key, distance + 1);
      queue.push(neighbor);
    }
  }

  const resolutionByKey = new Map<number, SparseBrickResolution>();
  for (const candidate of candidates) {
    const { coordinate } = candidate;
    const distance = interfaceDistance.get(candidate.key);
    // A component without a free surface (for example a completely full,
    // closed tank) is quiescent bulk and therefore starts at 1^3.
    const distanceRung = distance === undefined
      ? 0 : Math.max(0, ladder.resolutions.length - 1 - Math.max(0, distance - surfaceFineRings + 1));
    const adaptiveResolution = (distance !== undefined && distance < surfaceFineRings
      ? brickFineResolution : ladder.resolutions[distanceRung]!) as SparseBrickResolution;
    const staticSolidResolutionFloor = sparseCM12StaticSolidResolutionFloor(
      scene, options.finestDimensions, coordinate, brickFineResolution,
    );
    const policySelected = options.resolutionForBrick?.({
      coordinate, brickDimensions,
    }) ?? adaptiveResolution;
    const evidenceSelected = Math.max(policySelected,
      staticSolidResolutionFloor) as SparseBrickResolution;
    const selected = initialResolutionWithRefinementRegionBounds(
      refinementRegionParameters, options.finestDimensions, coordinate, 1,
      evidenceSelected, brickFineResolution);
    if (!isSparseBrickResolution(selected, brickFineResolution)) {
      throw new RangeError(
        `resolutionForBrick must return a rung on ${ladder.resolutions.join("/")}`,
      );
    }
    resolutionByKey.set(candidate.key, selected);
  }
  // Boundary promotion is a hard floor, then the ordinary strong-grading
  // closure propagates only as far as necessary into the liquid component.
  let changed = true;
  while (changed) {
    changed = false;
    for (const candidate of candidates) for (const direction of faceDirections) {
      const coordinate = candidate.coordinate.map((value, axis) =>
        value + direction[axis]) as [number, number, number];
      if (coordinate.some((value, axis) => value < 0
        || value >= brickDimensions[axis])) continue;
      const neighborKey = sparseBrickKey(coordinate, brickDimensions);
      if (!candidateByKey.has(neighborKey)) continue;
      const own = resolutionByKey.get(candidate.key)!;
      const neighbor = resolutionByKey.get(neighborKey)!;
      if (own > 2 * neighbor) {
        resolutionByKey.set(neighborKey, (own / 2) as SparseBrickResolution);
        changed = true;
      }
    }
  }
  const bricks = candidates.map((candidate) => initialBrick(
    scene, options.finestDimensions, candidate.coordinate,
    resolutionByKey.get(candidate.key)!, brickFineResolution,
  ));
  return createSparseAdaptiveMassAtlas(
    options.finestDimensions, bricks, 1, brickFineResolution,
  );
}

export function sparseAtlasLeaves(atlas: SparseAdaptiveMassAtlas): SparseAtlasLeaf[] {
  const leaves: SparseAtlasLeaf[] = [];
  for (const brick of atlas.bricks) {
    const factor = atlas.brickFineResolution * sparseBrickSpan(brick) / brick.resolution;
    for (let z = 0; z < brick.resolution; z += 1)
      for (let y = 0; y < brick.resolution; y += 1)
        for (let x = 0; x < brick.resolution; x += 1) {
          const localIndex = x + brick.resolution * (y + brick.resolution * z);
          const minimum = [brick.coordinate[0] * atlas.brickFineResolution + x * factor,
            brick.coordinate[1] * atlas.brickFineResolution + y * factor,
            brick.coordinate[2] * atlas.brickFineResolution + z * factor] as const;
          const spans = minimum.map((value, axis) =>
            Math.max(0, Math.min(factor, atlas.dimensions[axis] - value))) as [number, number, number];
          const volumeFineCells = spans[0] * spans[1] * spans[2];
          if (volumeFineCells <= 0) continue;
          leaves.push({
            id: brick.key * atlas.brickCellCapacity + localIndex,
            brickKey: brick.key,
            brickResolution: brick.resolution,
            localIndex,
            centerFine: [minimum[0] + 0.5 * spans[0], minimum[1] + 0.5 * spans[1],
              minimum[2] + 0.5 * spans[2]],
            volumeFineCells,
            density: brick.density[localIndex],
            gamma: brick.gamma[localIndex],
          });
        }
  }
  return leaves;
}

export function materializeSparseBrickAtlasDensity(atlas: SparseAdaptiveMassAtlas): Float32Array {
  const output = new Float32Array(atlas.dimensions.reduce((product, value) => product * value, 1));
  for (const brick of atlas.bricks) {
    const spanFine = atlas.brickFineResolution * sparseBrickSpan(brick);
    const factor = spanFine / brick.resolution;
    for (let z = 0; z < spanFine; z += 1)
      for (let y = 0; y < spanFine; y += 1)
        for (let x = 0; x < spanFine; x += 1) {
      const gx = brick.coordinate[0] * atlas.brickFineResolution + x,
        gy = brick.coordinate[1] * atlas.brickFineResolution + y,
        gz = brick.coordinate[2] * atlas.brickFineResolution + z;
      if (gx >= atlas.dimensions[0] || gy >= atlas.dimensions[1] || gz >= atlas.dimensions[2]) continue;
      const local = Math.floor(x / factor) + brick.resolution
        * (Math.floor(y / factor) + brick.resolution * Math.floor(z / factor));
      output[gx + atlas.dimensions[0] * (gy + atlas.dimensions[1] * gz)] = brick.density[local];
        }
  }
  return output;
}

/**
 * Initial low-activity policy: retain fine bricks for small connected features,
 * but conservatively coarsen large quiescent bodies before the first step.
 * This is component-local, so many disconnected droplets do not cause one
 * another to lose detail. Dynamic activity/camera promotion can replace this
 * bootstrap policy without changing atlas storage or the projection operator.
 */
export function coarsenLargeQuiescentComponents(
  atlas: SparseAdaptiveMassAtlas,
  maximumFineComponentBricks = 8,
): SparseAdaptiveMassAtlas {
  if (!Number.isSafeInteger(maximumFineComponentBricks)
    || maximumFineComponentBricks < 1) {
    throw new RangeError("maximumFineComponentBricks must be a positive integer");
  }
  const visited = new Set<number>();
  const components: SparseAdaptiveMassBrick[][] = [];
  const directions = [
    [-1, 0, 0], [1, 0, 0], [0, -1, 0],
    [0, 1, 0], [0, 0, -1], [0, 0, 1],
  ] as const;
  for (const seed of [...atlas.bricks].sort((left, right) => left.key - right.key)) {
    if (visited.has(seed.key)) continue;
    const component: SparseAdaptiveMassBrick[] = [];
    const queue = [seed];
    visited.add(seed.key);
    for (let cursor = 0; cursor < queue.length; cursor += 1) {
      const brick = queue[cursor]!;
      component.push(brick);
      for (const direction of directions) {
        const coordinate = brick.coordinate.map((value, axis) =>
          value + direction[axis]) as [number, number, number];
        if (coordinate.some((value, axis) =>
          value < 0 || value >= atlas.brickDimensions[axis])) continue;
        const key = sparseBrickKey(coordinate, atlas.brickDimensions);
        const neighbor = atlas.directory.get(key);
        if (!neighbor || visited.has(key)) continue;
        visited.add(key);
        queue.push(neighbor);
      }
    }
    components.push(component);
  }
  if (components.every((component) => component.length <= maximumFineComponentBricks)) {
    return atlas;
  }
  const denseDensity = materializeSparseBrickAtlasDensity(atlas);
  const bricks = components.flatMap((component) => {
    if (component.length <= maximumFineComponentBricks) return component;
    return component.map((brick) =>
      sparseBrickSpan(brick) > 1 || brick.resolution === atlas.ladder.coarseResolution
        ? brick
        : sparseBrickFromDense(
          brick.key,
          brick.coordinate,
          atlas.ladder.coarseResolution,
          denseDensity,
          atlas.dimensions,
          undefined,
          atlas.brickFineResolution,
        ));
  });
  return createSparseAdaptiveMassAtlas(
    atlas.dimensions,
    bricks.sort((left, right) => left.key - right.key),
    atlas.generation,
    atlas.brickFineResolution,
  );
}

export function sparseBrickAtlasStats(atlas: SparseAdaptiveMassAtlas): SparseBrickAtlasStats {
  const leaves = sparseAtlasLeaves(atlas);
  const logicalBrickCount = atlas.brickDimensions.reduce((product, value) => product * value, 1);
  const representedLogicalBricks = atlas.bricks.reduce((sum, brick) => {
    const span = sparseBrickSpan(brick);
    const clipped = brick.coordinate.map((value, axis) =>
      Math.max(0, Math.min(span, atlas.brickDimensions[axis] - value)));
    return sum + clipped[0]! * clipped[1]! * clipped[2]!;
  }, 0);
  const equivalentFinestCellCount = atlas.dimensions.reduce((product, value) => product * value, 1);
  return {
    generation: atlas.generation,
    logicalBrickCount,
    residentBrickCount: atlas.bricks.length,
    omittedEmptyBrickCount: logicalBrickCount - representedLogicalBricks,
    fineBrickCount: atlas.bricks.filter((brick) =>
      brick.resolution === atlas.brickFineResolution).length,
    coarseBrickCount: atlas.bricks.filter((brick) =>
      brick.resolution < atlas.brickFineResolution).length,
    fineCoarseFaceConnectedPairCount: atlas.bricks.reduce((count, brick) => {
      // Positive axes count every undirected adjacency exactly once.
      for (let axis = 0; axis < 3; axis += 1) {
        const coordinate = [...brick.coordinate] as [number, number, number];
        coordinate[axis] += 1;
        if (coordinate[axis] >= atlas.brickDimensions[axis]) continue;
        const neighbor = atlas.directory.get(sparseBrickKey(coordinate, atlas.brickDimensions));
        if (neighbor && neighbor.resolution !== brick.resolution) count += 1;
      }
      return count;
    }, 0),
    leafCount: leaves.length,
    equivalentFinestCellCount,
    leafCompressionRatio: equivalentFinestCellCount / Math.max(1, leaves.length),
    integratedMassFineCells: leaves.reduce(
      (sum, leaf) => sum + leaf.volumeFineCells * leaf.density, 0,
    ),
  };
}
