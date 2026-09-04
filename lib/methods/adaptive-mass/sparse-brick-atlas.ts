/**
 * CPU reference storage for an arbitrary sparse atlas of equal-world dyadic
 * bricks. This is a topology/numerics oracle, not the eventual GPU page pool.
 * Empty bricks have no payload and no directory entry.
 */

import {
  baseInitialLiquidFractionAtCell,
  initialFluidBrickCoordinates,
  initialFluidSeedBrickCoordinates,
  initialLiquidFractionAtCell,
  sceneDamBreakBox,
  sceneInitialLiquidVolumes,
} from "../../core/initial-fluid";
import type { SceneDescription } from "../../core/model";
import {
  clampRefinementRegionCellSize,
  refinementRegionCellBounds,
  sceneRefinementRegions,
  type RefinementRegionLattice,
} from "../../core/refinement-regions";
import { sampleSolidWorld, solidWorldForScene, SOLID_WORLD_BRICK_CELLS,
  type SolidWorld } from
  "../../core/solid-world";
import {
  applySparseCM12RefinementRegionResolutionBounds,
  packSparseCM12RefinementRegions,
  sparseCM12RefinementRegionResolutionBoundsForBrick,
} from "./sparse-cm12-refinement-regions";
import { SPARSE_CM12_VELOCITY_EXTENSION_DEPTH } from
  "./sparse-cm12-velocity-extension";

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
  /** Reuse the caller's canonical fluid-collider world when it is already built. */
  readonly solidWorld?: SolidWorld;
  /** Optional caller-owned construction guard; Sparse CM12 itself has no cell-count cap. */
  readonly maximumFinestCells?: number;
  /** Largest dyadic macro edge, in ordinary bricks. One disables macro leaves. */
  readonly maximumMacroSpanBricks?: number;
  readonly emptyEpsilon?: number;
  /** Count of occupied face-distance rings held at the finest rung around the interface. */
  readonly surfaceFineRings?: number;
  /**
   * Construction-only shift toward coarser interface rungs. Activity mode uses
   * one ring so a reset frame starts from its ordinary calm-surface B4 floor;
   * measured motion and thinness remain runtime promotion authorities.
   */
  readonly initialSurfaceCoarseningBiasRings?: number;
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

/** Exact resident leaves sharing a complete or partial face with one leaf. */
export function sparseBrickFaceNeighbors(
  atlas: SparseAdaptiveMassAtlas,
  brick: SparseAdaptiveMassBrick,
): readonly SparseAdaptiveMassBrick[] {
  const neighbors = new Map<number, SparseAdaptiveMassBrick>();
  const span = sparseBrickSpan(brick);
  for (let axis = 0; axis < 3; axis += 1) for (const sign of [-1, 1]) {
    const tangents = [0, 1, 2].filter((candidate) => candidate !== axis);
    for (let v = 0; v < span; v += 1) for (let u = 0; u < span; u += 1) {
      const coordinate = [...brick.coordinate] as [number, number, number];
      coordinate[axis] += sign < 0 ? -1 : span;
      coordinate[tangents[0]!] += u;
      coordinate[tangents[1]!] += v;
      const neighbor = sparseBrickContainingCoordinate(atlas, coordinate);
      if (neighbor && neighbor.key !== brick.key) neighbors.set(neighbor.key, neighbor);
    }
  }
  return [...neighbors.values()].sort((left, right) => left.key - right.key);
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
  const baseFraction = baseInitialLiquidFractionAtCell(
    scene, x, y, z, dimensions,
  );
  let world = initialSolidWorldCache.get(scene);
  if (!world) {
    world = solidWorldForScene(scene);
    initialSolidWorldCache.set(scene, world);
  }
  return (1 - sampleSolidWorld(world, [x, y, z]).solidFraction)
    * initialLiquidFractionAtCell(scene, x, y, z, dimensions, baseFraction);
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
 * Structural span-one leaves explicitly authored as future liquid capacity.
 *
 * A small wall catalogue is useful: it closes the dyadic grading graph for a
 * compact vessel. It must not, however, make a local reservoir inherit every
 * mixed terrain page in a large world. `maximumMixedSolidPages` is supplied
 * from the already bounded fluid/policy claim and rejects the solid catalogue
 * as a whole if it would dominate that working set.
 */
const SPARSE_CM12_STRUCTURAL_TO_FLUID_PAGE_RATIO = 3;

function structuralSeedBrickCoordinates(
  scene: SceneDescription,
  dimensions: SparseBrickVec3,
  brickDimensions: SparseBrickVec3,
  brickFineResolution: SparseBrickFineResolution,
  maximumMixedSolidPages = 0,
): SparseBrickVec3[] {
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
  if (maximumMixedSolidPages <= 0) return [...result.values()];
  let world = initialSolidWorldCache.get(scene);
  if (!world) {
    world = solidWorldForScene(scene);
    initialSolidWorldCache.set(scene, world);
  }
  // One SolidWorld page can fan out to several neighboring fluid pages. If
  // the source catalogue already exceeds the fluid-derived budget, do not pay
  // an O(world complexity) scan merely to discover that the result is too big.
  if (world.pages.length > maximumMixedSolidPages) return [...result.values()];
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

function prolongSparseBrick(
  brick: SparseAdaptiveMassBrick,
  resolution: SparseBrickResolution,
): SparseAdaptiveMassBrick {
  if (resolution === brick.resolution) return brick;
  if (resolution < brick.resolution || resolution % brick.resolution !== 0) {
    throw new Error(`cannot prolong brick ${brick.key} from B${brick.resolution}`
      + ` to B${resolution}`);
  }
  const factor = resolution / brick.resolution;
  const sample = (values: Float64Array, x: number, y: number, z: number) => values[
    Math.floor(x / factor) + brick.resolution * (Math.floor(y / factor)
      + brick.resolution * Math.floor(z / factor))
  ]!;
  return {
    ...brick,
    resolution,
    density: Float64Array.from({ length: resolution ** 3 }, (_, local) => {
      const x = local % resolution;
      const y = Math.floor(local / resolution) % resolution;
      const z = Math.floor(local / (resolution ** 2));
      return sample(brick.density, x, y, z);
    }),
    gamma: Float64Array.from({ length: resolution ** 3 }, (_, local) => {
      const x = local % resolution;
      const y = Math.floor(local / resolution) % resolution;
      const z = Math.floor(local / (resolution ** 2));
      return sample(brick.gamma, x, y, z);
    }),
  };
}

function restrictSparseBrick(
  brick: SparseAdaptiveMassBrick,
  resolution: SparseBrickResolution,
): SparseAdaptiveMassBrick {
  if (resolution === brick.resolution) return brick;
  if (resolution > brick.resolution || brick.resolution % resolution !== 0) {
    throw new Error(`cannot restrict brick ${brick.key} from B${brick.resolution}`
      + ` to B${resolution}`);
  }
  const factor = brick.resolution / resolution;
  const sample = (values: Float64Array, x: number, y: number, z: number) => {
    let sum = 0;
    for (let dz = 0; dz < factor; dz += 1)
      for (let dy = 0; dy < factor; dy += 1)
        for (let dx = 0; dx < factor; dx += 1) {
          const sx = x * factor + dx;
          const sy = y * factor + dy;
          const sz = z * factor + dz;
          sum += values[sx + brick.resolution * (sy + brick.resolution * sz)]!;
        }
    return sum / factor ** 3;
  };
  return {
    ...brick,
    resolution,
    density: Float64Array.from({ length: resolution ** 3 }, (_, local) => {
      const x = local % resolution;
      const y = Math.floor(local / resolution) % resolution;
      const z = Math.floor(local / (resolution ** 2));
      return sample(brick.density, x, y, z);
    }),
    gamma: Float64Array.from({ length: resolution ** 3 }, (_, local) => {
      const x = local % resolution;
      const y = Math.floor(local / resolution) % resolution;
      const z = Math.floor(local / (resolution ** 2));
      return sample(brick.gamma, x, y, z);
    }),
  };
}

/**
 * Close a sparse leaf set to strong 2:1 grading without refining through an
 * authored minimum-cell-size floor. The finer side of an offending face is
 * restricted until its physical cell width is at least half its neighbour's.
 */
function stronglyGradeSparseBricksByCoarsening(
  dimensions: SparseBrickVec3,
  bricks: readonly SparseAdaptiveMassBrick[],
  brickFineResolution: SparseBrickFineResolution,
): SparseAdaptiveMassBrick[] {
  if (bricks.length < 2) return [...bricks];
  const brickDimensions = dimensions.map((value) =>
    Math.ceil(value / brickFineResolution)) as [number, number, number];
  const directoriesBySpan = new Map<number, Map<number, number>>();
  let maximumSpan = 1;
  for (let index = 0; index < bricks.length; index += 1) {
    const brick = bricks[index]!;
    const span = sparseBrickSpan(brick);
    let directory = directoriesBySpan.get(span);
    if (!directory) directoriesBySpan.set(span, directory = new Map());
    directory.set(brick.key, index);
    maximumSpan = Math.max(maximumSpan, span);
  }
  const containing = (coordinate: SparseBrickVec3): number | undefined => {
    if (coordinate.some((value, axis) => value < 0
      || value >= brickDimensions[axis])) return undefined;
    for (let span = 1; span <= maximumSpan; span *= 2) {
      const origin = coordinate.map((value) =>
        Math.floor(value / span) * span) as [number, number, number];
      const index = directoriesBySpan.get(span)?.get(
        sparseBrickKey(origin, brickDimensions));
      if (index !== undefined) return index;
    }
    return undefined;
  };
  const pairs = new Map<string, readonly [number, number]>();
  for (let index = 0; index < bricks.length; index += 1) {
    const brick = bricks[index]!;
    const span = sparseBrickSpan(brick);
    for (let axis = 0; axis < 3; axis += 1) {
      const tangents = [0, 1, 2].filter((candidate) => candidate !== axis);
      for (let v = 0; v < span; v += 1) for (let u = 0; u < span; u += 1) {
        const coordinate = [...brick.coordinate] as [number, number, number];
        coordinate[axis] += span;
        coordinate[tangents[0]!] += u;
        coordinate[tangents[1]!] += v;
        const neighbor = containing(coordinate);
        if (neighbor === undefined || neighbor === index) continue;
        const low = Math.min(index, neighbor), high = Math.max(index, neighbor);
        pairs.set(`${low}/${high}`, [low, high]);
      }
    }
  }
  const resolutions = bricks.map((brick) => brick.resolution);
  let changed = true;
  while (changed) {
    changed = false;
    for (const [left, right] of pairs.values()) {
      const width = (index: number) => brickFineResolution
        * sparseBrickSpan(bricks[index]!) / resolutions[index]!;
      const leftWidth = width(left), rightWidth = width(right);
      if (Math.max(leftWidth, rightWidth) <= 2 * Math.min(leftWidth, rightWidth)) continue;
      const finer = leftWidth < rightWidth ? left : right;
      const coarser = finer === left ? right : left;
      let next = resolutions[finer]!;
      while (next > 1 && width(coarser) > 2 * brickFineResolution
        * sparseBrickSpan(bricks[finer]!) / next) next /= 2;
      if (next === resolutions[finer]) {
        throw new Error(`cannot strongly grade sparse brick face ${
          bricks[left]!.key}/${bricks[right]!.key} by coarsening`);
      }
      resolutions[finer] = next as SparseBrickResolution;
      changed = true;
    }
  }
  return bricks.map((brick, index) =>
    restrictSparseBrick(brick, resolutions[index]!));
}

function matchedAirSupportResolution(
  brick: SparseAdaptiveMassBrick,
  brickFineResolution: SparseBrickFineResolution,
): SparseBrickResolution {
  // A macro brick's resolution spans its whole physical edge. Convert that
  // to the equivalent resolution of one fixed brick so the exterior column
  // continues the surface cell width instead of jumping to B8.
  const resolution = Math.max(1,
    brick.resolution / sparseBrickSpan(brick));
  if (!isSparseBrickResolution(resolution, brickFineResolution)) {
    throw new Error(`brick ${brick.key} has no matching fixed-brick air rung`);
  }
  return resolution;
}

function matchedAirSupportLayerCount(
  brick: SparseAdaptiveMassBrick,
  brickFineResolution: SparseBrickFineResolution,
): number {
  return Math.ceil((SPARSE_CM12_VELOCITY_EXTENSION_DEPTH + 1)
    / matchedAirSupportResolution(brick, brickFineResolution));
}

/**
 * Add the gas domain required by interface transport. The recurrence reaches
 * eight cells from its liquid seed and the transport stencil needs the next
 * receiver cell. Each face-normal column continues the neighboring surface
 * cell width and contains enough bricks for those nine cells. Omitted air is
 * a valid far-field boundary only beyond this band.
 */
function atlasWithInitialAirSupport(
  scene: SceneDescription,
  dimensions: SparseBrickVec3,
  bricks: readonly SparseAdaptiveMassBrick[],
  brickFineResolution: SparseBrickFineResolution,
  refinementRegionParameters: ArrayBuffer,
): SparseAdaptiveMassAtlas {
  let atlas = createSparseAdaptiveMassAtlas(
    dimensions, bricks, 1, brickFineResolution,
  );
  let world = initialSolidWorldCache.get(scene);
  if (!world) {
    world = solidWorldForScene(scene);
    initialSolidWorldCache.set(scene, world);
  }
  const hasOpenVoxel = (coordinate: SparseBrickVec3): boolean => {
    const origin = coordinate.map((value) => value * brickFineResolution);
    for (let z = 0; z < brickFineResolution; z += 1)
      for (let y = 0; y < brickFineResolution; y += 1)
        for (let x = 0; x < brickFineResolution; x += 1) {
          const fine = [origin[0]! + x, origin[1]! + y, origin[2]! + z] as const;
          if (fine.some((value, axis) => value >= dimensions[axis])) continue;
          if (sampleSolidWorld(world!, fine).solidFraction < 1) return true;
        }
    return false;
  };
  const liquid = atlas.bricks.filter((brick) =>
    brick.density.some((density) => density > 0));
  const maximumLayerCount = liquid.reduce((maximum, brick) => Math.max(maximum,
    matchedAirSupportLayerCount(brick, brickFineResolution)), 0);
  for (let layer = 0; layer < maximumLayerCount; layer += 1) {
    const supportCoordinates = new Map<number, {
      readonly coordinate: SparseBrickVec3;
      readonly resolution: SparseBrickResolution;
    }>();
    for (const brick of liquid) {
      const resolution = matchedAirSupportResolution(brick, brickFineResolution);
      if (layer >= matchedAirSupportLayerCount(brick, brickFineResolution)) continue;
      const span = sparseBrickSpan(brick);
      for (let axis = 0; axis < 3; axis += 1) for (const sign of [-1, 1]) {
        const tangents = [0, 1, 2].filter((candidate) => candidate !== axis);
        for (let v = 0; v < span; v += 1) for (let u = 0; u < span; u += 1) {
          const coordinate = [...brick.coordinate] as [number, number, number];
          coordinate[axis] += sign < 0 ? -(layer + 1) : span + layer;
          coordinate[tangents[0]!] += u;
          coordinate[tangents[1]!] += v;
          if (coordinate.some((value, component) => value < 0
            || value >= atlas.brickDimensions[component])) continue;
          const owner = sparseBrickContainingCoordinate(atlas, coordinate);
          if (owner) continue;
          const key = sparseBrickKey(coordinate, atlas.brickDimensions);
          const previous = supportCoordinates.get(key);
          if ((!previous || previous.resolution < resolution)
            && hasOpenVoxel(coordinate)) {
            supportCoordinates.set(key, { coordinate, resolution });
          }
        }
      }
    }
    if (supportCoordinates.size > 0) {
      const support = [...supportCoordinates.values()].map((request) => initialBrick(
        scene, dimensions, request.coordinate,
        initialResolutionWithRefinementRegionBounds(
          refinementRegionParameters, dimensions, request.coordinate, 1,
          request.resolution,
          brickFineResolution,
        ),
        brickFineResolution,
      ));
      const combined = new Map<number, SparseAdaptiveMassBrick>(atlas.bricks.map((brick) =>
        [brick.key, brick] as const));
      for (const brick of support) combined.set(brick.key, brick);
      const hasAuthoredMinimum = new Uint32Array(
        refinementRegionParameters, 0, 4,
      )[0]! > 0;
      if (!hasAuthoredMinimum) {
        const resolutionByKey = new Map([...combined].map(([key, brick]) =>
          [key, brick.resolution] as const));
        const queued = new Set(support.map((brick) => brick.key));
        const queue = [...queued];
        const neighbors = (brick: SparseAdaptiveMassBrick) => {
          const found = new Map<number, SparseAdaptiveMassBrick>();
          const span = sparseBrickSpan(brick);
          for (let axis = 0; axis < 3; axis += 1) for (const sign of [-1, 1]) {
            const tangents = [0, 1, 2].filter((candidate) => candidate !== axis);
            for (let v = 0; v < span; v += 1) for (let u = 0; u < span; u += 1) {
              const coordinate = [...brick.coordinate] as [number, number, number];
              coordinate[axis] += sign < 0 ? -1 : span;
              coordinate[tangents[0]!] += u;
              coordinate[tangents[1]!] += v;
              if (coordinate.some((value, component) => value < 0
                || value >= atlas.brickDimensions[component])) continue;
              const key = sparseBrickKey(coordinate, atlas.brickDimensions);
              const neighbor = combined.get(key)
                ?? sparseBrickContainingCoordinate(atlas, coordinate);
              if (neighbor && neighbor.key !== brick.key) found.set(neighbor.key, neighbor);
            }
          }
          return found.values();
        };
        for (let cursor = 0; cursor < queue.length; cursor += 1) {
          const brick = combined.get(queue[cursor]!)!;
          for (const neighbor of neighbors(brick)) {
            const own = resolutionByKey.get(brick.key)!;
            const other = resolutionByKey.get(neighbor.key)!;
            const ownWidth = brickFineResolution * sparseBrickSpan(brick) / own;
            const otherWidth = brickFineResolution * sparseBrickSpan(neighbor) / other;
            if (Math.max(ownWidth, otherWidth)
              <= 2 * Math.min(ownWidth, otherWidth)) continue;
            const coarseKey = ownWidth > otherWidth ? brick.key : neighbor.key;
            const promoted = (2 * resolutionByKey.get(coarseKey)!) as
              SparseBrickResolution;
            if (!isSparseBrickResolution(promoted, brickFineResolution)) {
              throw new Error(`cannot grade brick ${coarseKey} beyond B${promoted / 2}`);
            }
            if (promoted <= resolutionByKey.get(coarseKey)!) continue;
            resolutionByKey.set(coarseKey, promoted);
            if (!queued.has(coarseKey)) {
              queued.add(coarseKey);
              queue.push(coarseKey);
            }
          }
        }
        atlas = createSparseAdaptiveMassAtlas(
          dimensions, [...combined.values()].map((brick) => prolongSparseBrick(
            brick, resolutionByKey.get(brick.key)!,
          )).sort((left, right) => left.key - right.key),
          1, brickFineResolution,
        );
        continue;
      }
      atlas = createSparseAdaptiveMassAtlas(
        dimensions, stronglyGradeSparseBricksByCoarsening(
          dimensions, [...combined.values()], brickFineResolution,
        ).sort((left, right) => left.key - right.key),
        1, brickFineResolution,
      );
    }
  }
  return atlas;
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
    const surfaceResolution = (surfaceFineRings > 0
      ? brickFineResolution
      : sparseBrickLadder(brickFineResolution).coarseResolution) as
      SparseBrickResolution;
    for (let z = 0; z < brickDimensions[2]; z += 1)
      for (let x = 0; x < brickDimensions[0]; x += 1)
        bricks.push(initialBrick(
          scene, dimensions, [x, fullBrickY, z], initialResolutionWithRefinementRegionBounds(
            refinementRegionParameters, dimensions, [x, fullBrickY, z], 1,
            surfaceResolution, brickFineResolution), brickFineResolution,
        ));
  }

  const hasAuthoredMinimum = new Uint32Array(
    refinementRegionParameters, 0, 4,
  )[0]! > 0;
  const gradedBricks = hasAuthoredMinimum
    ? stronglyGradeSparseBricksByCoarsening(
      dimensions, bricks, brickFineResolution,
    ) : bricks;
  const provisional = createSparseAdaptiveMassAtlas(
    dimensions, gradedBricks, 1, brickFineResolution,
  );
  for (const coordinate of structuralSeedBrickCoordinates(
    scene, dimensions, brickDimensions, brickFineResolution,
    SPARSE_CM12_STRUCTURAL_TO_FLUID_PAGE_RATIO * gradedBricks.length,
  )) {
    if (sparseBrickContainingCoordinate(provisional, coordinate)) continue;
    gradedBricks.push(initialBrick(
      scene, dimensions, coordinate, initialResolutionWithRefinementRegionBounds(
        refinementRegionParameters, dimensions, coordinate, 1,
        brickFineResolution, brickFineResolution), brickFineResolution,
    ));
  }
  return (hasAuthoredMinimum
    ? stronglyGradeSparseBricksByCoarsening(
      dimensions, gradedBricks, brickFineResolution,
    ) : gradedBricks).sort((left, right) => left.key - right.key);
}

function refinementRegionBrickCoordinates(
  scene: SceneDescription,
  dimensions: SparseBrickVec3,
  brickDimensions: SparseBrickVec3,
  brickFineResolution: SparseBrickFineResolution,
): SparseBrickVec3[] {
  const container = scene.container;
  const lattice: RefinementRegionLattice = {
    dimensions,
    cellSize_m: [container.width_m / dimensions[0],
      container.height_m / dimensions[1],
      container.depth_m / dimensions[2]],
    origin_m: { x: -0.5 * container.width_m, y: 0,
      z: -0.5 * container.depth_m },
  };
  const coordinates = new Map<number, SparseBrickVec3>();
  for (const region of sceneRefinementRegions(scene)) {
    const bounds = refinementRegionCellBounds(region, lattice);
    const floor = clampRefinementRegionCellSize(region.minimumCellSize_cells);
    const ownMaximum = Math.max(1, Math.min(brickFineResolution,
      Math.floor(brickFineResolution / floor)));
    // Only the rungs strictly between the regional cap and ordinary Bmax need
    // pre-catalogued support. The first unconstrained Bmax brick may remain a
    // fixed sparse-world page once its ratio to the last mapped rung is 2:1.
    const gradingHalo = Math.max(0,
      Math.ceil(Math.log2(brickFineResolution / ownMaximum)) - 1);
    const lower = bounds.min.map((value) => Math.max(0,
      Math.floor(value / brickFineResolution) - gradingHalo)) as
      [number, number, number];
    const upper = bounds.max.map((value, axis) => Math.min(brickDimensions[axis],
      Math.ceil(value / brickFineResolution) + gradingHalo)) as
      [number, number, number];
    for (let z = lower[2]; z < upper[2]; z += 1)
      for (let y = lower[1]; y < upper[1]; y += 1)
        for (let x = lower[0]; x < upper[0]; x += 1) {
          const coordinate = [x, y, z] as const;
          coordinates.set(sparseBrickKey(coordinate, brickDimensions), coordinate);
        }
  }
  return [...coordinates.entries()].sort((left, right) => left[0] - right[0])
    .map((entry) => entry[1]);
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
  // An enforcement region is topology capacity, not merely an initial
  // resolution hint. Catalogue every intersecting base brick now so later
  // transport inside the box never has to fall back to a fixed-B8 sparse-world
  // page that cannot obey a coarser hard minimum.
  for (const coordinate of refinementRegionBrickCoordinates(
    scene, dimensions, brickDimensions, brickFineResolution,
  )) addBrick(coordinate);
  for (const coordinate of structuralSeedBrickCoordinates(
    scene, dimensions, brickDimensions, brickFineResolution,
    SPARSE_CM12_STRUCTURAL_TO_FLUID_PAGE_RATIO * candidates.size,
  )) addBrick(coordinate);
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
  // The adaptive solver has already constructed the canonical SolidWorld for
  // resident upload. Reusing it avoids a second terrain bake and also ensures
  // fluid-collider regions participate in atlas sampling.
  if (options.solidWorld) initialSolidWorldCache.set(scene, options.solidWorld);
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
  const authoredSurfaceFineRings = typeof options.surfaceFineRings === "number"
    && Number.isFinite(options.surfaceFineRings)
    ? Math.min(8, Math.max(1, Math.round(options.surfaceFineRings))) : 1;
  const initialSurfaceCoarseningBiasRings =
    typeof options.initialSurfaceCoarseningBiasRings === "number"
      && Number.isFinite(options.initialSurfaceCoarseningBiasRings)
      ? Math.min(authoredSurfaceFineRings, Math.max(0,
        Math.round(options.initialSurfaceCoarseningBiasRings))) : 0;
  const surfaceFineRings = authoredSurfaceFineRings
    - initialSurfaceCoarseningBiasRings;
  const brickDimensions = options.finestDimensions.map((value) =>
    Math.ceil(value / brickFineResolution)) as [number, number, number];
  const container = scene.container;
  const refinementRegions = sceneRefinementRegions(scene);
  const refinementLattice: RefinementRegionLattice = {
    dimensions: options.finestDimensions,
    cellSize_m: [container.width_m / options.finestDimensions[0],
      container.height_m / options.finestDimensions[1],
      container.depth_m / options.finestDimensions[2]],
    origin_m: { x: -0.5 * container.width_m, y: 0, z: -0.5 * container.depth_m },
  };
  const refinementRegionParameters = packSparseCM12RefinementRegions(
    refinementRegions, refinementLattice);
  if (!options.resolutionForBrick) {
    // A macro leaf may be rerung, but it cannot be spatially split after it is
    // packed into the resident catalogue. A partial minimum-cell-size box is
    // nevertheless safe: an intersecting macro is conservatively coarsened as
    // a whole, so no cell inside the box can slip below the authored minimum.
    // A partial maximum-cell-size box is different because a crossing macro
    // must split to honour its finer ceiling. Keep that case on the base-brick
    // builder below, which evaluates the complete envelope per brick and runs
    // the ordinary refine-only 2:1 closure across every boundary.
    const regionBoundsEpsilon = 1e-4;
    const hierarchicalCompatibleRefinementEnvelope = refinementRegions.every((region) => {
      const bounds = refinementRegionCellBounds(region, refinementLattice);
      const coversDomain = bounds.min.every((value) => value <= regionBoundsEpsilon)
        && bounds.max.every((value, axis) =>
          value >= options.finestDimensions[axis]! - regionBoundsEpsilon);
      return region.maximumCellSize_cells === undefined || coversDomain;
    });
    // A hard region must also catalogue its currently dry volume so future
    // wetting cannot allocate an incompatible fixed-B8 frontier page. The
    // macro tank shortcut represents only initial liquid/structure and is
    // therefore valid only when no enforcement envelope is authored.
    const hierarchical = hierarchicalCompatibleRefinementEnvelope
      ? hierarchicalTankFillBricks(
        scene, options.finestDimensions, brickDimensions, brickFineResolution,
        surfaceFineRings, maximumMacroSpanBricks, refinementRegionParameters,
      ) : undefined;
    if (hierarchical) {
      return atlasWithInitialAirSupport(
        scene, options.finestDimensions, hierarchical, brickFineResolution,
        refinementRegionParameters,
      );
    }
  }
  const candidateCoordinates = [...candidateInitialBrickCoordinates(
    scene, options.finestDimensions, brickDimensions, brickFineResolution,
  )];
  const structuralCoordinates = structuralSeedBrickCoordinates(
    scene, options.finestDimensions, brickDimensions, brickFineResolution,
    SPARSE_CM12_STRUCTURAL_TO_FLUID_PAGE_RATIO * candidateCoordinates.length,
  );
  const structuralKeys = new Set(structuralCoordinates.map((coordinate) =>
    sparseBrickKey(coordinate, brickDimensions)));
  // Region capacity must survive the empty-brick filter below. These leaves
  // are structural policy support even before liquid reaches them.
  for (const coordinate of refinementRegionBrickCoordinates(
    scene, options.finestDimensions, brickDimensions, brickFineResolution,
  )) structuralKeys.add(sparseBrickKey(coordinate, brickDimensions));
  const candidates: Array<{
    readonly coordinate: SparseBrickVec3;
    readonly key: number;
    readonly interfaceBrick: boolean;
  }> = [];
  for (const coordinate of candidateCoordinates) {
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

  const policyTileForCoordinate = (coordinate: SparseBrickVec3) => {
    const origin: SparseBrickVec3 = [coordinate[0] * brickFineResolution,
      coordinate[1] * brickFineResolution, coordinate[2] * brickFineResolution];
    const extent: SparseBrickVec3 = [Math.max(0, Math.min(brickFineResolution,
      options.finestDimensions[0] - origin[0])), Math.max(0, Math.min(
      brickFineResolution, options.finestDimensions[1] - origin[1])),
    Math.max(0, Math.min(brickFineResolution,
      options.finestDimensions[2] - origin[2]))];
    const bounds = sparseCM12RefinementRegionResolutionBoundsForBrick(
      refinementRegionParameters, origin, extent, brickFineResolution,
      brickFineResolution,
    );
    const scale = Math.max(1, Math.round(
      brickFineResolution / bounds.maximumResolution,
    ));
    const tileOrigin: SparseBrickVec3 = [Math.floor(coordinate[0] / scale) * scale,
      Math.floor(coordinate[1] / scale) * scale,
      Math.floor(coordinate[2] / scale) * scale];
    return { scale, tileOrigin, key: `${scale}:${tileOrigin.join(",")}` };
  };

  // Sparse omission is part of the physical operator: a represented coarse
  // brick contains both its wet cells and the adjacent dry pressure faces. A
  // finer authored scene whose two-brick policy tile contains only its wet
  // sub-bricks otherwise drops those faces, even though restriction produces
  // the same density and RHS. Close residency over the policy tile so the
  // finer partition represents exactly the same physical support.
  const candidateByKey = new Map(candidates.map((candidate) =>
    [candidate.key, candidate] as const));
  if (sceneRefinementRegions(scene).length > 0) {
    for (const candidate of [...candidates]) {
      const tile = policyTileForCoordinate(candidate.coordinate);
      if (tile.scale <= 1) continue;
      for (let z = 0; z < tile.scale; z += 1) {
        for (let y = 0; y < tile.scale; y += 1) {
          for (let x = 0; x < tile.scale; x += 1) {
            const coordinate: SparseBrickVec3 = [tile.tileOrigin[0] + x,
              tile.tileOrigin[1] + y, tile.tileOrigin[2] + z];
            if (coordinate.some((value, axis) => value < 0
              || value >= brickDimensions[axis])) continue;
            const siblingTile = policyTileForCoordinate(coordinate);
            if (siblingTile.key !== tile.key) continue;
            const key = sparseBrickKey(coordinate, brickDimensions);
            if (candidateByKey.has(key)) continue;
            const sibling = {
              coordinate,
              key,
              interfaceBrick: brickHasInterface(
                scene, options.finestDimensions, coordinate, epsilon,
                brickFineResolution,
              ),
            };
            candidateByKey.set(key, sibling);
            candidates.push(sibling);
          }
        }
      }
    }
    candidates.sort((left, right) => left.key - right.key);
  }

  // Seed an exact face-distance transform from the free surface. This makes
  // the initial accepted topology agree with the GPU activity policy's full
  // 8/4/2/1 ladder. Previously every saturated interior brick was fixed at
  // 4^3; later GPU requests for 2^3 and 1^3 remained candidate-only, so deep
  // pools such as ocean-seiche could never visibly or physically coarsen.
  const faceDirections = [
    [-1, 0, 0], [1, 0, 0], [0, -1, 0],
    [0, 1, 0], [0, 0, -1], [0, 0, 1],
  ] as const;

  // A minimum-cell-size region is also a physical policy scale. For example,
  // a two-fine-cell floor on a 32^3 scene must classify the same 0.4 m policy
  // tile as one B8 brick in the physically identical 16^3 scene. Classifying
  // each of the eight smaller authored bricks independently made only the
  // brick that actually crossed the interface stay fine, so the adaptive
  // topology depended on the otherwise irrelevant authoring lattice.
  const policyScaleByKey = new Map<number, number>();
  const policyTileByBrick = new Map<number, string>();
  const policyTiles = new Map<string, {
    readonly bricks: typeof candidates;
    interfaceBrick: boolean;
    readonly neighbors: Set<string>;
  }>();
  for (const candidate of candidates) {
    const { scale, key: tileKey } = policyTileForCoordinate(candidate.coordinate);
    policyScaleByKey.set(candidate.key, scale);
    policyTileByBrick.set(candidate.key, tileKey);
    let tile = policyTiles.get(tileKey);
    if (!tile) {
      tile = { bricks: [], interfaceBrick: false, neighbors: new Set() };
      policyTiles.set(tileKey, tile);
    }
    tile.bricks.push(candidate);
    tile.interfaceBrick ||= candidate.interfaceBrick;
  }
  for (const candidate of candidates) for (const direction of faceDirections) {
    const coordinate = candidate.coordinate.map((value, axis) =>
      value + direction[axis]) as [number, number, number];
    if (coordinate.some((value, axis) =>
      value < 0 || value >= brickDimensions[axis])) continue;
    const neighbor = candidateByKey.get(sparseBrickKey(coordinate, brickDimensions));
    if (!neighbor) continue;
    const ownTile = policyTileByBrick.get(candidate.key)!;
    const neighborTile = policyTileByBrick.get(neighbor.key)!;
    if (ownTile !== neighborTile) policyTiles.get(ownTile)!.neighbors.add(neighborTile);
  }
  const interfaceDistance = new Map<string, number>();
  const queue: string[] = [];
  for (const [key, tile] of policyTiles) if (tile.interfaceBrick) {
    interfaceDistance.set(key, 0);
    queue.push(key);
  }
  for (let cursor = 0; cursor < queue.length; cursor += 1) {
    const key = queue[cursor]!;
    const distance = interfaceDistance.get(key)!;
    for (const neighbor of policyTiles.get(key)!.neighbors) {
      if (interfaceDistance.has(neighbor)) continue;
      interfaceDistance.set(neighbor, distance + 1);
      queue.push(neighbor);
    }
  }

  const resolutionByKey = new Map<number, SparseBrickResolution>();
  const gradingMaximumByKey = new Map<number, SparseBrickResolution>();
  for (const candidate of candidates) {
    const { coordinate } = candidate;
    const tileKey = policyTileByBrick.get(candidate.key)!;
    const distance = interfaceDistance.get(tileKey);
    const policyFineResolution = brickFineResolution
      / policyScaleByKey.get(candidate.key)!;
    // A component without a free surface (for example a completely full,
    // closed tank) is quiescent bulk and therefore starts at 1^3.
    const distanceRung = distance === undefined ? 0 : Math.max(0,
      Math.log2(policyFineResolution) - Math.max(0,
        distance - surfaceFineRings + 1));
    const adaptiveResolution = (distance !== undefined && distance < surfaceFineRings
      ? policyFineResolution : 2 ** distanceRung) as SparseBrickResolution;
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
    const origin: SparseBrickVec3 = [
      coordinate[0] * brickFineResolution,
      coordinate[1] * brickFineResolution,
      coordinate[2] * brickFineResolution,
    ];
    const extent: SparseBrickVec3 = [
      Math.max(0, Math.min(brickFineResolution,
        options.finestDimensions[0] - origin[0])),
      Math.max(0, Math.min(brickFineResolution,
        options.finestDimensions[1] - origin[1])),
      Math.max(0, Math.min(brickFineResolution,
        options.finestDimensions[2] - origin[2])),
    ];
    const regionBounds = sparseCM12RefinementRegionResolutionBoundsForBrick(
      refinementRegionParameters, origin, extent, brickFineResolution,
      brickFineResolution);
    if (!isSparseBrickResolution(selected, brickFineResolution)) {
      throw new RangeError(
        `resolutionForBrick must return a rung on ${ladder.resolutions.join("/")}`,
      );
    }
    resolutionByKey.set(candidate.key, selected);
    gradingMaximumByKey.set(candidate.key,
      regionBounds.maximumResolution as SparseBrickResolution);
  }
  // Propagate authored minimum-size caps outward before closing the requested
  // topology. This is the reverse of ordinary refine-only grading: a hard
  // coarse region coarsens its fine neighbours as far as 2:1 requires, rather
  // than allowing those neighbours to refine back through the region floor.
  let gradingChanged = true;
  while (gradingChanged) {
    gradingChanged = false;
    for (const candidate of candidates) for (const direction of faceDirections) {
      const coordinate: SparseBrickVec3 = [
        candidate.coordinate[0] + direction[0],
        candidate.coordinate[1] + direction[1],
        candidate.coordinate[2] + direction[2],
      ];
      if (coordinate.some((value, axis) => value < 0
        || value >= brickDimensions[axis])) continue;
      const neighborKey = sparseBrickKey(coordinate, brickDimensions);
      if (!candidateByKey.has(neighborKey)) continue;
      const ownMaximum = gradingMaximumByKey.get(candidate.key)!;
      const neighborMaximum = gradingMaximumByKey.get(neighborKey)!;
      const mappedMaximum = Math.min(ownMaximum,
        2 * neighborMaximum) as SparseBrickResolution;
      if (mappedMaximum < ownMaximum) {
        gradingMaximumByKey.set(candidate.key, mappedMaximum);
        gradingChanged = true;
      }
    }
  }
  for (const candidate of candidates) {
    resolutionByKey.set(candidate.key, Math.min(
      resolutionByKey.get(candidate.key)!,
      gradingMaximumByKey.get(candidate.key)!,
    ) as SparseBrickResolution);
  }

  // Boundary promotion remains a hard resolution floor, but is now bounded by
  // the propagated authored cap. The cap construction guarantees that every
  // required 2:1 support rung is representable without violating a region.
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
        const promoted = Math.min(own / 2,
          gradingMaximumByKey.get(neighborKey)!) as SparseBrickResolution;
        if (promoted > neighbor) {
          resolutionByKey.set(neighborKey, promoted);
          changed = true;
        }
      }
    }
  }
  const bricks = candidates.map((candidate) => initialBrick(
    scene, options.finestDimensions, candidate.coordinate,
    resolutionByKey.get(candidate.key)!, brickFineResolution,
  ));
  return atlasWithInitialAirSupport(
    scene, options.finestDimensions, bricks, brickFineResolution,
    refinementRegionParameters,
  );
}

/**
 * Generation-zero membership includes an authored air support layer and is
 * closed over authored refinement-policy tiles.
 *
 * Uniform CM12 transports and extends face velocity through air cells beside
 * the liquid interface. Sparse CM12 needs the same represented domain: an
 * inactive authored leaf is absent from transport, characteristic tracing,
 * sharpening return, and velocity extension, rather than merely being a
 * zero-density pressure boundary. Keep the dry resident face band covering
 * velocity-extension depth plus its transport receiver active. Runtime
 * activity uses the same face-adjacent predicate to retain and advance it.
 *
 * A coarse brick also represents dry cells adjacent to its liquid inside the
 * same pressure stencil. When that physical brick is authored as a group of
 * finer sparse bricks, activating only wet sub-bricks removes those dry-side
 * faces. Keep every resident sibling in a minimum-cell-size policy tile active
 * whenever any sibling participates in the initial liquid/support topology.
 */
export function sparseCM12InitialActiveBrickKeys(
  scene: SceneDescription,
  atlas: SparseAdaptiveMassAtlas,
): ReadonlySet<number> {
  const active = new Set(atlas.bricks.filter((brick) =>
    brick.density.some((density) => density > 0)).map((brick) => brick.key));
  const addDrySupportLayer = (layer: number, sourceKeys: readonly number[]) => {
    for (const key of sourceKeys) {
      const brick = atlas.directory.get(key);
      if (!brick) continue;
      if (layer >= matchedAirSupportLayerCount(brick,
        atlas.brickFineResolution)) continue;
      const span = sparseBrickSpan(brick);
      for (let axis = 0; axis < 3; axis += 1) for (const sign of [-1, 1]) {
        const tangents = [0, 1, 2].filter((candidate) => candidate !== axis);
        for (let v = 0; v < span; v += 1) for (let u = 0; u < span; u += 1) {
          const coordinate = [...brick.coordinate] as [number, number, number];
          coordinate[axis] += sign < 0 ? -(layer + 1) : span + layer;
          coordinate[tangents[0]!] += u;
          coordinate[tangents[1]!] += v;
          const neighbor = sparseBrickContainingCoordinate(atlas, coordinate);
          if (neighbor?.density.every((density) => density <= 0)) {
            active.add(neighbor.key);
          }
        }
      }
    }
  };
  const supportLayerCountFor = (keys: readonly number[]) => keys.reduce(
    (maximum, key) => {
      const brick = atlas.directory.get(key);
      return brick ? Math.max(maximum, matchedAirSupportLayerCount(brick,
        atlas.brickFineResolution)) : maximum;
    }, 0);
  if (sceneRefinementRegions(scene).length === 0) {
    const liquid = [...active];
    for (let layer = 0; layer < supportLayerCountFor(liquid); layer += 1) {
      addDrySupportLayer(layer, liquid);
    }
    return active;
  }
  const container = scene.container;
  const packed = packSparseCM12RefinementRegions(
    sceneRefinementRegions(scene), {
      dimensions: atlas.dimensions,
      cellSize_m: [container.width_m / atlas.dimensions[0],
        container.height_m / atlas.dimensions[1],
        container.depth_m / atlas.dimensions[2]],
      origin_m: { x: -0.5 * container.width_m, y: 0,
        z: -0.5 * container.depth_m },
    });
  const closePolicyTiles = () => {
    for (const key of [...active]) {
      const brick = atlas.directory.get(key);
      if (!brick || sparseBrickSpan(brick) !== 1) continue;
      const origin: SparseBrickVec3 = [brick.coordinate[0] * atlas.brickFineResolution,
        brick.coordinate[1] * atlas.brickFineResolution,
        brick.coordinate[2] * atlas.brickFineResolution];
      const extent: SparseBrickVec3 = [Math.max(0, Math.min(atlas.brickFineResolution,
        atlas.dimensions[0] - origin[0])), Math.max(0, Math.min(
        atlas.brickFineResolution, atlas.dimensions[1] - origin[1])),
      Math.max(0, Math.min(atlas.brickFineResolution,
        atlas.dimensions[2] - origin[2]))];
      const bounds = sparseCM12RefinementRegionResolutionBoundsForBrick(
        packed, origin, extent, atlas.brickFineResolution,
        atlas.brickFineResolution,
      );
      const scale = Math.max(1, Math.round(
        atlas.brickFineResolution / bounds.maximumResolution,
      ));
      if (scale <= 1) continue;
      const tileOrigin: SparseBrickVec3 = [
        Math.floor(brick.coordinate[0] / scale) * scale,
        Math.floor(brick.coordinate[1] / scale) * scale,
        Math.floor(brick.coordinate[2] / scale) * scale,
      ];
      for (let z = 0; z < scale; z += 1) for (let y = 0; y < scale; y += 1) {
        for (let x = 0; x < scale; x += 1) {
          const coordinate = [tileOrigin[0] + x, tileOrigin[1] + y,
            tileOrigin[2] + z] as SparseBrickVec3;
          if (coordinate.some((value, axis) =>
            value >= atlas.brickDimensions[axis])) continue;
          const sibling = atlas.directory.get(sparseBrickKey(
            coordinate, atlas.brickDimensions,
          ));
          if (sibling) active.add(sibling.key);
        }
      }
    }
  };
  // Close the liquid's physical policy tiles before finding their exterior,
  // then close the resulting support tiles as the same physical volumes.
  closePolicyTiles();
  const liquidTiles = [...active];
  for (let layer = 0; layer < supportLayerCountFor(liquidTiles); layer += 1) {
    addDrySupportLayer(layer, liquidTiles);
    closePolicyTiles();
  }
  return active;
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
