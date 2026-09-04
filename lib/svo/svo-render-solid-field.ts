import type { SceneDescription } from "../core/model";
import { solidWorldVoxelPatchBounds_m } from "../core/solid-world";
import { terrainHeightAt } from "../core/terrain";
import type { SparseBrickCoordinate } from "./sparse-brick-octree";

/**
 * Renderer-owned terrain samples. This is deliberately not a SolidWorld: its
 * lattice follows presentation quality and it is never published to physics.
 */
export interface SvoRenderTerrainField {
  readonly origin_m: readonly [number, number];
  readonly cellSize_m: readonly [number, number];
  readonly dimensions: readonly [number, number];
  readonly heights_m: Float32Array<ArrayBuffer>;
  readonly materialId: number;
  /** Ordered metre-space projection of canonical SolidWorld edits. */
  readonly patches: readonly SvoRenderSolidPatch[];
}

export interface SvoRenderSolidPatch {
  readonly operation: "fill" | "clear";
  readonly minimum_m: readonly [number, number, number];
  readonly maximum_m: readonly [number, number, number];
  readonly materialId: number;
}

/** Rows per cooperative offer; comfortably below one frame on the depth-3 garden. */
const TERRAIN_ROWS_PER_OFFER = 8;

export function* buildSvoRenderTerrainFieldSteps(
  scene: Pick<SceneDescription, "container" | "terrain" | "voxelDomain" | "solidVoxels">,
  cellSize_m: readonly [number, number, number],
  materialId: number,
): Generator<unknown, SvoRenderTerrainField | undefined, undefined> {
  if (!scene.terrain) return undefined;
  const nx = Math.max(1, Math.round(scene.container.width_m / cellSize_m[0]));
  const nz = Math.max(1, Math.round(scene.container.depth_m / cellSize_m[2]));
  const originX = -0.5 * scene.container.width_m;
  const originZ = -0.5 * scene.container.depth_m;
  const heights_m = new Float32Array(new ArrayBuffer(nx * nz * Float32Array.BYTES_PER_ELEMENT));
  for (let z = 0; z < nz; z += 1) {
    const worldZ = originZ + (z + 0.5) * cellSize_m[2];
    for (let x = 0; x < nx; x += 1) {
      const worldX = originX + (x + 0.5) * cellSize_m[0];
      heights_m[x + nx * z] = Math.min(scene.container.height_m,
        Math.max(0, terrainHeightAt(scene.terrain, worldX, worldZ)));
    }
    if ((z + 1) % TERRAIN_ROWS_PER_OFFER === 0) yield;
  }
  return {
    origin_m: [originX, originZ],
    cellSize_m: [cellSize_m[0], cellSize_m[2]],
    dimensions: [nx, nz],
    heights_m,
    materialId,
    patches: scene.solidVoxels.map((patch) => {
      const bounds = solidWorldVoxelPatchBounds_m(scene, patch);
      return {
        operation: patch.operation,
        minimum_m: bounds.minimum,
        maximum_m: bounds.maximum,
        materialId: patch.materialId ?? 1,
      };
    }),
  };
}

interface TerrainRangeLevel {
  readonly width: number;
  readonly depth: number;
  readonly minimum: Float32Array<ArrayBuffer>;
  readonly maximum: Float32Array<ArrayBuffer>;
}

/**
 * Min/max hierarchy aligned to SVO brick coordinates. It answers whether the
 * terrain/air interface crosses a node without scanning fine columns during
 * the octree walk.
 */
export function createSvoRenderTerrainRefinement(options: {
  readonly field: SvoRenderTerrainField;
  readonly worldOrigin_m: readonly [number, number, number];
  readonly renderCellSize_m: readonly [number, number, number];
  readonly refinedBrickDimensions: readonly [number, number, number];
  readonly nodeEdge_m: readonly (readonly number[])[];
  readonly brickSize: number;
  readonly maximumDepth: number;
}): { refineEnvironmentLeaf(level: number, coordinate: SparseBrickCoordinate): boolean } {
  const { field, worldOrigin_m, renderCellSize_m, refinedBrickDimensions,
    nodeEdge_m, brickSize, maximumDepth } = options;
  const width = refinedBrickDimensions[0], depth = refinedBrickDimensions[2];
  const minimum = new Float32Array(new ArrayBuffer(width * depth * 4));
  const maximum = new Float32Array(new ArrayBuffer(width * depth * 4));
  minimum.fill(Number.POSITIVE_INFINITY);
  maximum.fill(Number.NEGATIVE_INFINITY);
  const [nx, nz] = field.dimensions;
  for (let z = 0; z < nz; z += 1) for (let x = 0; x < nx; x += 1) {
    const worldX = field.origin_m[0] + (x + 0.5) * field.cellSize_m[0];
    const worldZ = field.origin_m[1] + (z + 0.5) * field.cellSize_m[1];
    const bx = Math.floor((worldX - worldOrigin_m[0]) / (renderCellSize_m[0] * brickSize));
    const bz = Math.floor((worldZ - worldOrigin_m[2]) / (renderCellSize_m[2] * brickSize));
    if (bx < 0 || bz < 0 || bx >= width || bz >= depth) continue;
    const index = bx + width * bz;
    const height = field.heights_m[x + nx * z]!;
    minimum[index] = Math.min(minimum[index]!, height);
    maximum[index] = Math.max(maximum[index]!, height);
  }
  const levels: TerrainRangeLevel[] = [{ width, depth, minimum, maximum }];
  while (levels.at(-1)!.width > 1 || levels.at(-1)!.depth > 1) {
    const prior = levels.at(-1)!;
    const nextWidth = Math.ceil(prior.width / 2), nextDepth = Math.ceil(prior.depth / 2);
    const nextMinimum = new Float32Array(new ArrayBuffer(nextWidth * nextDepth * 4));
    const nextMaximum = new Float32Array(new ArrayBuffer(nextWidth * nextDepth * 4));
    nextMinimum.fill(Number.POSITIVE_INFINITY);
    nextMaximum.fill(Number.NEGATIVE_INFINITY);
    for (let z = 0; z < prior.depth; z += 1) for (let x = 0; x < prior.width; x += 1) {
      const target = (x >> 1) + nextWidth * (z >> 1), source = x + prior.width * z;
      nextMinimum[target] = Math.min(nextMinimum[target]!, prior.minimum[source]!);
      nextMaximum[target] = Math.max(nextMaximum[target]!, prior.maximum[source]!);
    }
    levels.push({ width: nextWidth, depth: nextDepth, minimum: nextMinimum, maximum: nextMaximum });
  }
  return {
    refineEnvironmentLeaf(level, coordinate) {
      if (level >= maximumDepth) return false;
      const range = levels[Math.min(levels.length - 1, maximumDepth - level)]!;
      if (coordinate.x < 0 || coordinate.z < 0
        || coordinate.x >= range.width || coordinate.z >= range.depth) return false;
      const index = coordinate.x + range.width * coordinate.z;
      const low = range.minimum[index]!, high = range.maximum[index]!;
      if (!Number.isFinite(low) || !Number.isFinite(high)) return false;
      const edgeY = nodeEdge_m[level]![1]!;
      const nodeLow = worldOrigin_m[1] + coordinate.y * edgeY;
      const nodeHigh = nodeLow + edgeY;
      return high >= nodeLow && low <= nodeHigh;
    },
  };
}

/** Refine the boundary, rather than the volume, of ordered SolidWorld edits. */
export function createSvoRenderPatchRefinement(options: {
  readonly patches: readonly SvoRenderSolidPatch[];
  readonly worldOrigin_m: readonly [number, number, number];
  readonly nodeEdge_m: readonly (readonly number[])[];
  readonly maximumDepth: number;
}): { refineEnvironmentLeaf(level: number, coordinate: SparseBrickCoordinate): boolean } {
  return {
    refineEnvironmentLeaf(level, coordinate) {
      if (level >= options.maximumDepth) return false;
      const edge = options.nodeEdge_m[level];
      if (!edge) return false;
      const minimum = [
        options.worldOrigin_m[0] + coordinate.x * edge[0]!,
        options.worldOrigin_m[1] + coordinate.y * edge[1]!,
        options.worldOrigin_m[2] + coordinate.z * edge[2]!,
      ];
      const maximum = minimum.map((value, axis) => value + edge[axis]!);
      return options.patches.some((patch) => {
        const lo = patch.minimum_m, hi = patch.maximum_m;
        if (maximum.some((value, axis) => value < lo[axis]!)
          || minimum.some((value, axis) => value > hi[axis]!)) return false;
        return lo.some((plane, axis) => plane >= minimum[axis]! && plane <= maximum[axis]!)
          || hi.some((plane, axis) => plane >= minimum[axis]! && plane <= maximum[axis]!);
      });
    },
  };
}
