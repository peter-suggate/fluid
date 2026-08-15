/**
 * CPU reference storage for an arbitrary sparse atlas of equal-world 4^3/8^3
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
import {
  classifyFineBoxAgainstSphericalContainer,
  sphericalContainerFineGeometry,
  sphericalContainerOpenFractionAtCell,
  type SphericalContainerFineGeometry,
} from "../../core/spherical-container";
import { sceneHasTerrain, terrainHeightAt } from "../../core/terrain";

/** Power-of-two cells per fixed 8-fine-cell brick edge. */
export type SparseBrickResolution = 1 | 2 | 4 | 8;
export type SparseBrickVec3 = readonly [number, number, number];

export interface SparseAdaptiveMassBrick {
  readonly key: number;
  readonly coordinate: SparseBrickVec3;
  readonly resolution: SparseBrickResolution;
  /** Intensive CM12 surface density, x-major within the brick. */
  readonly density: Float64Array;
  readonly gamma: Float64Array;
}

export interface SparseAdaptiveMassAtlas {
  readonly dimensions: SparseBrickVec3;
  readonly brickDimensions: SparseBrickVec3;
  readonly bricks: readonly SparseAdaptiveMassBrick[];
  readonly directory: ReadonlyMap<number, SparseAdaptiveMassBrick>;
  readonly generation: number;
  /** Embedded closed boundary carried by every topology/field generation. */
  readonly boundary?: SphericalContainerFineGeometry;
}

export interface SparseBrickAtlasInitializationOptions {
  readonly finestDimensions: SparseBrickVec3;
  /** Optional caller-owned construction guard; Sparse CM12 itself has no cell-count cap. */
  readonly maximumFinestCells?: number;
  readonly emptyEpsilon?: number;
  /** Count of occupied face-distance rings held at 8^3 around the initial interface. */
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

export const BRICK_FINE_RESOLUTION = 8;

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
  boundary?: SphericalContainerFineGeometry,
): SparseAdaptiveMassAtlas {
  positiveDimensions(dimensions);
  const brickDimensions = dimensions.map((value) =>
    Math.ceil(value / BRICK_FINE_RESOLUTION)) as [number, number, number];
  const directory = new Map<number, SparseAdaptiveMassBrick>();
  for (const brick of bricks) {
    if (brick.resolution !== 1 && brick.resolution !== 2
      && brick.resolution !== 4 && brick.resolution !== 8) {
      throw new RangeError(`brick ${brick.key} resolution must be 1, 2, 4, or 8`);
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
  }
  // Accepted atlases are strongly graded: every resident face adjacency is
  // same-level or one rung apart across the complete 1/2/4/8 ladder.
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
  return { dimensions, brickDimensions, bricks: [...bricks], directory, generation, boundary };
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
  if (sceneHasTerrain(scene)) {
    const worldX = -0.5 * scene.container.width_m
      + (x + 0.5) * scene.container.width_m / nx;
    const worldZ = -0.5 * scene.container.depth_m
      + (z + 0.5) * scene.container.depth_m / nz;
    if ((y + 0.5) * scene.container.height_m / ny
      <= terrainHeightAt(scene.terrain, worldX, worldZ)) return 0;
  }
  const baseWet = scene.fluid.initialCondition === "tank-fill"
    ? (y + 0.5) / ny <= scene.container.fillFraction
    : damBreakBoxContains(dam, (x + 0.5) / nx, (y + 0.5) / ny, (z + 0.5) / nz);
  return Math.min(
    sphericalContainerOpenFractionAtCell(scene, x, y, z, dimensions),
    initialLiquidFractionAtCell(scene, x, y, z, dimensions, baseWet),
  );
}

function brickRequiresCutBoundaryResolution(
  boundary: SphericalContainerFineGeometry | undefined,
  coordinate: SparseBrickVec3,
): boolean {
  if (!boundary) return false;
  const minimum = coordinate.map((value) => value * BRICK_FINE_RESOLUTION) as
    [number, number, number];
  const maximum = coordinate.map((value) => (value + 1) * BRICK_FINE_RESOLUTION) as
    [number, number, number];
  return classifyFineBoxAgainstSphericalContainer(boundary, minimum, maximum) === "cut";
}

function brickHasInterface(
  scene: SceneDescription,
  dimensions: SparseBrickVec3,
  coordinate: SparseBrickVec3,
  epsilon: number,
): boolean {
  const origin = coordinate.map((value) => value * BRICK_FINE_RESOLUTION) as number[];
  for (let z = 0; z < BRICK_FINE_RESOLUTION; z += 1)
    for (let y = 0; y < BRICK_FINE_RESOLUTION; y += 1)
      for (let x = 0; x < BRICK_FINE_RESOLUTION; x += 1) {
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
): SparseAdaptiveMassBrick {
  const factor = BRICK_FINE_RESOLUTION / resolution;
  const density = new Float64Array(resolution ** 3);
  const gamma = new Float64Array(resolution ** 3).fill(1);
  for (let z = 0; z < resolution; z += 1)
    for (let y = 0; y < resolution; y += 1)
      for (let x = 0; x < resolution; x += 1) {
        let rho = 0, count = 0;
        for (let dz = 0; dz < factor; dz += 1)
          for (let dy = 0; dy < factor; dy += 1)
            for (let dx = 0; dx < factor; dx += 1) {
              const gx = coordinate[0] * 8 + x * factor + dx;
              const gy = coordinate[1] * 8 + y * factor + dy;
              const gz = coordinate[2] * 8 + z * factor + dz;
              if (gx >= dimensions[0] || gy >= dimensions[1] || gz >= dimensions[2]) continue;
              rho += initialDensityAt(scene, dimensions, gx, gy, gz);
              count += 1;
            }
        density[x + resolution * (y + resolution * z)] = count > 0 ? rho / count : 0;
      }
  const brickDimensions: SparseBrickVec3 = [
    Math.ceil(dimensions[0] / BRICK_FINE_RESOLUTION),
    Math.ceil(dimensions[1] / BRICK_FINE_RESOLUTION),
    Math.ceil(dimensions[2] / BRICK_FINE_RESOLUTION),
  ];
  return {
    key: sparseBrickKey(coordinate, brickDimensions),
    coordinate,
    resolution,
    density,
    gamma,
  };
}

function candidateInitialBrickCoordinates(
  scene: SceneDescription,
  dimensions: SparseBrickVec3,
  brickDimensions: SparseBrickVec3,
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
      Math.floor(value / BRICK_FINE_RESOLUTION))) as [number, number, number];
    const upper = maximumExclusive.map((value, axis) => Math.min(brickDimensions[axis],
      Math.ceil(value / BRICK_FINE_RESOLUTION))) as [number, number, number];
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

  const authored = initialFluidBrickCoordinates(scene, dimensions, BRICK_FINE_RESOLUTION);
  for (const coordinate of initialFluidSeedBrickCoordinates(
    scene, dimensions, BRICK_FINE_RESOLUTION,
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
): SparseAdaptiveMassBrick {
  const density = new Float64Array(resolution ** 3);
  const gamma = new Float64Array(resolution ** 3).fill(1);
  const factor = BRICK_FINE_RESOLUTION / resolution;
  for (let z = 0; z < resolution; z += 1)
    for (let y = 0; y < resolution; y += 1)
      for (let x = 0; x < resolution; x += 1) {
        let rho = 0, g = 0, count = 0;
        for (let dz = 0; dz < factor; dz += 1)
          for (let dy = 0; dy < factor; dy += 1)
            for (let dx = 0; dx < factor; dx += 1) {
              const gx = coordinate[0] * 8 + x * factor + dx;
              const gy = coordinate[1] * 8 + y * factor + dy;
              const gz = coordinate[2] * 8 + z * factor + dz;
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
    Math.ceil(value / 8)) as [number, number, number];
  const boundary = sphericalContainerFineGeometry(scene, options.finestDimensions);
  const candidates: Array<{
    readonly coordinate: SparseBrickVec3;
    readonly key: number;
    readonly interfaceBrick: boolean;
  }> = [];
  for (const coordinate of candidateInitialBrickCoordinates(
    scene, options.finestDimensions, brickDimensions,
  )) {
    let nonempty = false;
    for (let z = 0; z < 8 && !nonempty; z += 1)
      for (let y = 0; y < 8 && !nonempty; y += 1)
        for (let x = 0; x < 8; x += 1)
          if (initialDensityAt(
            scene, options.finestDimensions,
            coordinate[0] * 8 + x,
            coordinate[1] * 8 + y,
            coordinate[2] * 8 + z,
          ) > epsilon) {
            nonempty = true; break;
          }
    if (!nonempty) continue;
    candidates.push({
      coordinate,
      key: sparseBrickKey(coordinate, brickDimensions),
      interfaceBrick: brickHasInterface(
        scene, options.finestDimensions, coordinate, epsilon,
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
    const adaptiveResolution: SparseBrickResolution = distance !== undefined
      && distance < surfaceFineRings ? 8
      : distance === surfaceFineRings ? 4
        : distance === surfaceFineRings + 1 ? 2 : 1;
    const selected = brickRequiresCutBoundaryResolution(boundary, coordinate) ? 8
      : options.resolutionForBrick?.({
      coordinate, brickDimensions,
    }) ?? adaptiveResolution;
    if (selected !== 1 && selected !== 2 && selected !== 4 && selected !== 8) {
      throw new RangeError("resolutionForBrick must return 1, 2, 4, or 8");
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
    resolutionByKey.get(candidate.key)!,
  ));
  return createSparseAdaptiveMassAtlas(options.finestDimensions, bricks, 1, boundary);
}

export function sparseAtlasLeaves(atlas: SparseAdaptiveMassAtlas): SparseAtlasLeaf[] {
  const leaves: SparseAtlasLeaf[] = [];
  for (const brick of atlas.bricks) {
    const factor = 8 / brick.resolution;
    for (let z = 0; z < brick.resolution; z += 1)
      for (let y = 0; y < brick.resolution; y += 1)
        for (let x = 0; x < brick.resolution; x += 1) {
          const localIndex = x + brick.resolution * (y + brick.resolution * z);
          const minimum = [brick.coordinate[0] * 8 + x * factor,
            brick.coordinate[1] * 8 + y * factor,
            brick.coordinate[2] * 8 + z * factor] as const;
          const spans = minimum.map((value, axis) =>
            Math.max(0, Math.min(factor, atlas.dimensions[axis] - value))) as [number, number, number];
          const volumeFineCells = spans[0] * spans[1] * spans[2];
          if (volumeFineCells <= 0) continue;
          leaves.push({
            id: brick.key * 512 + localIndex,
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
    const factor = 8 / brick.resolution;
    for (let z = 0; z < 8; z += 1) for (let y = 0; y < 8; y += 1) for (let x = 0; x < 8; x += 1) {
      const gx = brick.coordinate[0] * 8 + x, gy = brick.coordinate[1] * 8 + y,
        gz = brick.coordinate[2] * 8 + z;
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
      brick.resolution === 4
        ? brick
        : sparseBrickFromDense(
          brick.key,
          brick.coordinate,
          4,
          denseDensity,
          atlas.dimensions,
        ));
  });
  return createSparseAdaptiveMassAtlas(
    atlas.dimensions,
    bricks.sort((left, right) => left.key - right.key),
    atlas.generation,
    atlas.boundary,
  );
}

export function sparseBrickAtlasStats(atlas: SparseAdaptiveMassAtlas): SparseBrickAtlasStats {
  const leaves = sparseAtlasLeaves(atlas);
  const logicalBrickCount = atlas.brickDimensions.reduce((product, value) => product * value, 1);
  const equivalentFinestCellCount = atlas.dimensions.reduce((product, value) => product * value, 1);
  return {
    generation: atlas.generation,
    logicalBrickCount,
    residentBrickCount: atlas.bricks.length,
    omittedEmptyBrickCount: logicalBrickCount - atlas.bricks.length,
    fineBrickCount: atlas.bricks.filter((brick) => brick.resolution === 8).length,
    coarseBrickCount: atlas.bricks.filter((brick) => brick.resolution < 8).length,
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
