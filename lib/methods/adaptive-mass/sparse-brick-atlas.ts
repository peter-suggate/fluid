/**
 * CPU reference storage for an arbitrary sparse atlas of equal-world 4^3/8^3
 * bricks. This is a topology/numerics oracle, not the eventual GPU page pool.
 * Empty bricks have no payload and no directory entry.
 */

import { damBreakBoxContains, initialLiquidFractionAtCell, sceneDamBreakBox } from "../../core/initial-fluid";
import type { SceneDescription } from "../../core/model";
import { sceneHasTerrain, terrainHeightAt } from "../../core/terrain";

export type SparseBrickResolution = 4 | 8;
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
}

export interface SparseBrickAtlasInitializationOptions {
  readonly finestDimensions: SparseBrickVec3;
  /** Hard construction ceiling. Defaults to 16 million finest cells. */
  readonly maximumFinestCells?: number;
  readonly emptyEpsilon?: number;
  /** Keep saturated bricks fine on one configurable half of the domain. */
  readonly fineHalf?: {
    readonly axis: 0 | 1 | 2;
    readonly side: "negative" | "positive";
  };
  /** Optional policy for non-interface, nonempty bricks. Interface always wins at 8^3. */
  readonly resolutionForBrick?: (input: {
    readonly coordinate: SparseBrickVec3;
    readonly brickDimensions: SparseBrickVec3;
    readonly defaultResolution: SparseBrickResolution;
  }) => SparseBrickResolution;
}

export interface SparseBrickAtlasStats {
  readonly generation: number;
  readonly logicalBrickCount: number;
  readonly residentBrickCount: number;
  readonly omittedEmptyBrickCount: number;
  readonly fineBrickCount: number;
  readonly coarseBrickCount: number;
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

const BRICK_FINE_RESOLUTION = 8;

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
): SparseAdaptiveMassAtlas {
  positiveDimensions(dimensions);
  const brickDimensions = dimensions.map((value) =>
    Math.ceil(value / BRICK_FINE_RESOLUTION)) as [number, number, number];
  const directory = new Map<number, SparseAdaptiveMassBrick>();
  for (const brick of bricks) {
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
  return { dimensions, brickDimensions, bricks: [...bricks], directory, generation };
}

function initialDensity(scene: SceneDescription, dimensions: SparseBrickVec3): Float64Array {
  const [nx, ny, nz] = dimensions;
  const density = new Float64Array(nx * ny * nz);
  if (scene.systems?.fluid === false) return density;
  const dam = sceneDamBreakBox(scene);
  for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1) for (let x = 0; x < nx; x += 1) {
    if (sceneHasTerrain(scene)) {
      const worldX = -0.5 * scene.container.width_m + (x + 0.5) * scene.container.width_m / nx;
      const worldZ = -0.5 * scene.container.depth_m + (z + 0.5) * scene.container.depth_m / nz;
      if ((y + 0.5) * scene.container.height_m / ny
        <= terrainHeightAt(scene.terrain, worldX, worldZ)) continue;
    }
    const baseWet = scene.fluid.initialCondition === "tank-fill"
      ? (y + 0.5) / ny <= scene.container.fillFraction
      : damBreakBoxContains(dam, (x + 0.5) / nx, (y + 0.5) / ny, (z + 0.5) / nz);
    density[x + nx * (y + ny * z)] = initialLiquidFractionAtCell(
      scene, x, y, z, dimensions, baseWet,
    );
  }
  return density;
}

function denseValue(
  values: ArrayLike<number>, dimensions: SparseBrickVec3, x: number, y: number, z: number,
): number {
  if (x < 0 || y < 0 || z < 0
    || x >= dimensions[0] || y >= dimensions[1] || z >= dimensions[2]) return 0;
  return values[x + dimensions[0] * (y + dimensions[1] * z)];
}

function brickHasInterface(
  dense: ArrayLike<number>, dimensions: SparseBrickVec3, coordinate: SparseBrickVec3,
  epsilon: number,
): boolean {
  const origin = coordinate.map((value) => value * BRICK_FINE_RESOLUTION) as number[];
  for (let z = 0; z < BRICK_FINE_RESOLUTION; z += 1)
    for (let y = 0; y < BRICK_FINE_RESOLUTION; y += 1)
      for (let x = 0; x < BRICK_FINE_RESOLUTION; x += 1) {
        const gx = origin[0] + x, gy = origin[1] + y, gz = origin[2] + z;
        if (gx >= dimensions[0] || gy >= dimensions[1] || gz >= dimensions[2]) continue;
        const own = denseValue(dense, dimensions, gx, gy, gz);
        if (own > epsilon && own < 1 - epsilon) return true;
        for (const [dx, dy, dz] of [[-1, 0, 0], [1, 0, 0], [0, -1, 0],
          [0, 1, 0], [0, 0, -1], [0, 0, 1]] as const) {
          if (gx + dx < 0 || gy + dy < 0 || gz + dz < 0
            || gx + dx >= dimensions[0] || gy + dy >= dimensions[1]
            || gz + dz >= dimensions[2]) continue;
          const neighbor = denseValue(dense, dimensions, gx + dx, gy + dy, gz + dz);
          if ((own > epsilon) !== (neighbor > epsilon)) return true;
        }
      }
  return false;
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
  const maximum = options.maximumFinestCells ?? 16_000_000;
  if (cellCount > maximum) throw new RangeError(`bounded finest lattice has ${cellCount} cells; cap is ${maximum}`);
  const epsilon = options.emptyEpsilon ?? 1e-12;
  const dense = initialDensity(scene, options.finestDimensions);
  const brickDimensions = options.finestDimensions.map((value) =>
    Math.ceil(value / 8)) as [number, number, number];
  const bricks: SparseAdaptiveMassBrick[] = [];
  for (let bz = 0; bz < brickDimensions[2]; bz += 1)
    for (let by = 0; by < brickDimensions[1]; by += 1)
      for (let bx = 0; bx < brickDimensions[0]; bx += 1) {
        const coordinate = [bx, by, bz] as const;
        let nonempty = false;
        for (let z = 0; z < 8 && !nonempty; z += 1)
          for (let y = 0; y < 8 && !nonempty; y += 1)
            for (let x = 0; x < 8; x += 1)
              if (denseValue(dense, options.finestDimensions, bx * 8 + x, by * 8 + y, bz * 8 + z) > epsilon) {
                nonempty = true; break;
              }
        if (!nonempty) continue;
        const interfaceBrick = brickHasInterface(
          dense, options.finestDimensions, coordinate, epsilon,
        );
        let defaultResolution: SparseBrickResolution = 4;
        if (options.fineHalf) {
          const { axis, side } = options.fineHalf;
          const onPositiveHalf = coordinate[axis] + 0.5 >= 0.5 * brickDimensions[axis];
          if ((side === "positive") === onPositiveHalf) defaultResolution = 8;
        }
        const selected = options.resolutionForBrick?.({
          coordinate, brickDimensions, defaultResolution,
        }) ?? defaultResolution;
        if (selected !== 4 && selected !== 8) {
          throw new RangeError("resolutionForBrick must return 4 or 8");
        }
        const resolution: SparseBrickResolution = interfaceBrick ? 8 : selected;
        const key = sparseBrickKey(coordinate, brickDimensions);
        bricks.push(sparseBrickFromDense(
          key, coordinate, resolution, dense, options.finestDimensions,
        ));
      }
  return createSparseAdaptiveMassAtlas(options.finestDimensions, bricks);
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
    coarseBrickCount: atlas.bricks.filter((brick) => brick.resolution === 4).length,
    leafCount: leaves.length,
    equivalentFinestCellCount,
    leafCompressionRatio: equivalentFinestCellCount / Math.max(1, leaves.length),
    integratedMassFineCells: leaves.reduce(
      (sum, leaf) => sum + leaf.volumeFineCells * leaf.density, 0,
    ),
  };
}
