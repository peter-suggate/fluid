import type { SparseBrickCoordinate } from "./sparse-brick-octree";
import {
  TERRAIN_CEILING_MARGIN_M,
  terrainCeiling,
  terrainColumnHeightsForLattice,
  type TerrainDescription,
} from "./terrain";

/**
 * The ground, as ordinary voxel coverage.
 *
 * Terrain is the one piece of authored content in this engine that is special
 * everywhere: it is the only render-ABI kind that is not a finite record, and
 * that single fact suppresses the candidate BVH scene-wide
 * (`lib/svo-scene-primitives.ts`), throws in live proxy updates
 * (`lib/webgpu-sparse-scene-proxies.ts`), and carries a bespoke marcher with a
 * CPU mirror that has to agree with it. Worst of all it never reached the
 * octree, so the largest surface in the hero garden was drawn by primary
 * visibility and invisible to every cone, shadow and GI ray that reads the
 * opacity pyramid.
 *
 * This module is the correction, and it is deliberately *not* a new kind. There
 * is no terrain record, no tenth shape number, no new ABI surface: the ground
 * becomes a column-height field the per-voxel rebuild consults alongside its
 * binned candidates, and downstream of the scene lanes nothing can tell terrain
 * from a voxelized box. Everything that made terrain special stays where it is
 * — authoring, collision, the analytic primary — and can be retired later
 * without another format.
 *
 * The lattice is the octree's own finest cell lattice over its own domain, so a
 * voxel's column index is its world cell coordinate and no origin, spacing or
 * transform has to be transmitted or agreed. That is the whole reason the field
 * costs three header words.
 */
export interface SparseSceneTerrainField {
  /** Columns along x and z, at the octree's finest cell size. */
  readonly dimensions: readonly [number, number];
  /** World-space Y of the ground at each column centre, row-major x + nx * z. */
  readonly heights_m: Float32Array<ArrayBuffer>;
  /** Lowest and highest column, for bounds and for brick selection. */
  readonly minimumHeight_m: number;
  readonly maximumHeight_m: number;
  /**
   * The world box the ground occupies inside the domain: the whole footprint,
   * from the domain floor up to the ceiling every marcher already brackets at.
   *
   * This is what a publication dirties when the ground changes, and it is a
   * *region* rather than a record's bounds precisely because there is no record.
   */
  readonly bounds: {
    readonly minimum: readonly [number, number, number];
    readonly maximum: readonly [number, number, number];
  };
}

/** The octree domain a terrain field is baked onto. */
export interface SparseSceneTerrainDomain {
  readonly worldOrigin_m: readonly [number, number, number];
  readonly cellSize_m: readonly [number, number, number];
  /** Finest-level cells along each axis; the field is one column per (x, z). */
  readonly dimensionsCells: readonly [number, number, number];
}

/** The same emptiness test `sceneHasTerrain` applies, without needing a scene. */
export function sparseSceneTerrainPresent(terrain: TerrainDescription | undefined): boolean {
  return !!terrain && (terrain.baseHeight_m > 0 || terrain.features.length > 0 || !!terrain.grid);
}

/**
 * Bake the ground onto the octree's finest cell lattice.
 *
 * Undefined when the scene has no ground, which is the encoding for "this
 * voxelizer has no terrain field" all the way down to the shader: a zero
 * dimension in the header block switches the whole path off, so a scene without
 * terrain allocates nothing, writes nothing and evaluates nothing.
 */
export function planSparseSceneTerrainField(
  terrain: TerrainDescription | undefined,
  domain: SparseSceneTerrainDomain,
): SparseSceneTerrainField | undefined {
  if (!sparseSceneTerrainPresent(terrain)) return undefined;
  const [nx, , nz] = domain.dimensionsCells;
  if (!Number.isSafeInteger(nx) || !Number.isSafeInteger(nz) || nx < 1 || nz < 1) {
    throw new RangeError("Terrain field needs a positive integer cell footprint");
  }
  const domainTop_m = domain.worldOrigin_m[1] + domain.dimensionsCells[1] * domain.cellSize_m[1];
  const heights_m = terrainColumnHeightsForLattice(terrain, {
    originX_m: domain.worldOrigin_m[0], originZ_m: domain.worldOrigin_m[2],
    cellX_m: domain.cellSize_m[0], cellZ_m: domain.cellSize_m[2],
    nx, nz, maximumHeight_m: domainTop_m,
  });
  let minimumHeight_m = Infinity, maximumHeight_m = -Infinity;
  for (const height of heights_m) {
    if (height < minimumHeight_m) minimumHeight_m = height;
    if (height > maximumHeight_m) maximumHeight_m = height;
  }
  if (!Number.isFinite(minimumHeight_m)) { minimumHeight_m = 0; maximumHeight_m = 0; }
  // The ceiling every marcher already brackets at, so the voxelized ground and
  // the analytic one agree about where ground can possibly be. Clamped into the
  // domain because a region outside it dirties nothing and only widens the
  // invalidation dispatch.
  const ceiling_m = Math.min(domainTop_m,
    Math.max(maximumHeight_m + TERRAIN_CEILING_MARGIN_M, terrainCeiling(terrain)));
  return {
    dimensions: [nx, nz],
    heights_m,
    minimumHeight_m,
    maximumHeight_m,
    bounds: {
      minimum: [domain.worldOrigin_m[0], domain.worldOrigin_m[1], domain.worldOrigin_m[2]],
      maximum: [
        domain.worldOrigin_m[0] + nx * domain.cellSize_m[0],
        ceiling_m,
        domain.worldOrigin_m[2] + nz * domain.cellSize_m[2],
      ],
    },
  };
}

/** The ground height at one finest cell column, clamped to the field's edge. */
export function sparseSceneTerrainColumnHeight(field: SparseSceneTerrainField, cellX: number, cellZ: number): number {
  const [nx, nz] = field.dimensions;
  const x = Math.min(nx - 1, Math.max(0, cellX));
  const z = Math.min(nz - 1, Math.max(0, cellZ));
  return field.heights_m[x + nx * z];
}

/** Lowest and highest ground inside one finest-cell box footprint, inclusive. */
export function sparseSceneTerrainColumnRange(
  field: SparseSceneTerrainField,
  firstCellX: number, firstCellZ: number, lastCellX: number, lastCellZ: number,
): { minimum_m: number; maximum_m: number } {
  let minimum_m = Infinity, maximum_m = -Infinity;
  for (let z = firstCellZ; z <= lastCellZ; z += 1) for (let x = firstCellX; x <= lastCellX; x += 1) {
    const height = sparseSceneTerrainColumnHeight(field, x, z);
    if (height < minimum_m) minimum_m = height;
    if (height > maximum_m) maximum_m = height;
  }
  return { minimum_m, maximum_m };
}

/**
 * The finest bricks the ground occupies, for the octree plan.
 *
 * Every brick from the domain floor up to the tallest column inside its own
 * footprint, so the set is the solid the ground actually is rather than the box
 * that contains it — a garden whose ground rises on one side does not claim the
 * sky on the other. The interior bricks are claimed alongside the surface ones
 * deliberately: a coarse mip level averages its eight children, and a hollow
 * ground reads as half-transparent to a cone sampling it two levels up.
 */
export function sparseSceneTerrainBrickCoordinates(
  field: SparseSceneTerrainField,
  domain: SparseSceneTerrainDomain,
  brickSize: number,
): SparseBrickCoordinate[] {
  const brickDimensions = domain.dimensionsCells.map((cells) => Math.ceil(cells / brickSize));
  const brickEdge = domain.cellSize_m.map((size) => size * brickSize);
  const coordinates: SparseBrickCoordinate[] = [];
  for (let z = 0; z < brickDimensions[2]; z += 1) for (let x = 0; x < brickDimensions[0]; x += 1) {
    const range = sparseSceneTerrainColumnRange(field,
      x * brickSize, z * brickSize, (x + 1) * brickSize - 1, (z + 1) * brickSize - 1);
    if (!Number.isFinite(range.maximum_m)) continue;
    const top = Math.min(brickDimensions[1] - 1,
      Math.floor((range.maximum_m - domain.worldOrigin_m[1]) / brickEdge[1]));
    for (let y = 0; y <= top; y += 1) coordinates.push({ x, y, z });
  }
  return coordinates;
}
