import { createContentMemo } from "../../../core/content-memo";
import type { SceneDescription } from "../../../core/model";
import type { SolidWorldCoordinate } from "../../../core/solid-world";
import { terrainColumnHeightsForLatticeSteps, terrainContentStamp, terrainWorldContainer } from "../../../core/terrain";
import type { SparseBrickCoordinate } from "../construction/sparse-brick-octree";

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
}

/**
 * The fields of the last few (terrain, footprint, cell) a world was built on.
 *
 * A field is a function of the terrain's content, the footprint it covers and
 * the cell it is sampled on, and of nothing else in the scene, so a rebuild
 * that changed none of them takes the samples it already has. Shared between
 * builds: no holder writes to `heights_m`.
 */
const terrainFields = createContentMemo<SvoRenderTerrainField>(2);

export function* buildSvoRenderTerrainFieldSteps(
  scene: Pick<SceneDescription, "container" | "terrain" | "voxelDomain" | "solidVoxels" | "scenery">,
  cellSize_m: readonly [number, number, number],
  materialId: number,
): Generator<unknown, SvoRenderTerrainField | undefined, undefined> {
  if (!scene.terrain || scene.scenery?.terrainSurface === "scenery") return undefined;
  const terrainContainer = terrainWorldContainer(scene);
  const nx = Math.max(1, Math.round(terrainContainer.width_m / cellSize_m[0]));
  const nz = Math.max(1, Math.round(terrainContainer.depth_m / cellSize_m[2]));
  const originX = -0.5 * terrainContainer.width_m;
  const originZ = -0.5 * terrainContainer.depth_m;
  const key = JSON.stringify([terrainContentStamp(scene.terrain), terrainContainer.width_m,
    terrainContainer.height_m, terrainContainer.depth_m, cellSize_m[0], cellSize_m[2], materialId]);
  const held = terrainFields.get(key);
  if (held) return held;
  // The column bake every lattice shares: one height per cell centre, clamped
  // to the footprint's height, from the description's own nodes where a sweep
  // this size pays for deriving them.
  const heights_m = yield* terrainColumnHeightsForLatticeSteps(scene.terrain, {
    originX_m: originX, originZ_m: originZ,
    cellX_m: cellSize_m[0], cellZ_m: cellSize_m[2],
    nx, nz, maximumHeight_m: terrainContainer.height_m,
  });
  return terrainFields.set(key, {
    origin_m: [originX, originZ],
    cellSize_m: [cellSize_m[0], cellSize_m[2]],
    dimensions: [nx, nz],
    heights_m,
    materialId,
  });
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
    // A centre sample influences one terrain cell on either side of its
    // column. Include that halo in neighbouring brick ranges as well.
    const bx0 = Math.floor((worldX - field.cellSize_m[0] - worldOrigin_m[0]) / (renderCellSize_m[0] * brickSize));
    const bx1 = Math.floor((worldX + field.cellSize_m[0] - worldOrigin_m[0]) / (renderCellSize_m[0] * brickSize));
    const bz0 = Math.floor((worldZ - field.cellSize_m[1] - worldOrigin_m[2]) / (renderCellSize_m[2] * brickSize));
    const bz1 = Math.floor((worldZ + field.cellSize_m[1] - worldOrigin_m[2]) / (renderCellSize_m[2] * brickSize));
    const height = field.heights_m[x + nx * z]!;
    for(let bz=bz0;bz<=bz1;bz++)for(let bx=bx0;bx<=bx1;bx++){
      if(bx<0||bz<0||bx>=width||bz>=depth)continue;
      const index=bx+width*bz;
      minimum[index]=Math.min(minimum[index]!,height);
      maximum[index]=Math.max(maximum[index]!,height);
    }
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

/**
 * Refine around the pages an authored surface can cross
 * (`terrainEditSurfacePages`), rather than through the volume of a fill or of
 * a dig. A node is refined when a surface page lies within a voxel of it.
 *
 * The pages are kept as a pyramid of occupied page coordinates, so a node of
 * any size is answered by at most eight lookups at the level its span fits.
 */
export function createSvoRenderEditPageRefinement(options: {
  readonly pages: readonly SolidWorldCoordinate[];
  /** The solid lattice the pages are addressed on. */
  readonly lattice: { readonly origin_m: readonly [number, number, number]; readonly cellSize_m: readonly [number, number, number] };
  readonly worldOrigin_m: readonly [number, number, number];
  readonly nodeEdge_m: readonly (readonly number[])[];
  readonly maximumDepth: number;
}): { refineEnvironmentLeaf(level: number, coordinate: SparseBrickCoordinate): boolean } {
  const PAGE_CELLS = 8, LEVELS = 16, BIAS = 0x8000, SPAN = 0x1_0000;
  const key = (x: number, y: number, z: number): number => (x + BIAS) + SPAN * ((y + BIAS) + SPAN * (z + BIAS));
  const occupied: Set<number>[] = Array.from({ length: LEVELS + 1 }, () => new Set<number>());
  for (const [x, y, z] of options.pages) {
    if (Math.max(Math.abs(x), Math.abs(y), Math.abs(z)) >= BIAS) {
      throw new RangeError(`Terrain edit page ${x},${y},${z} is outside the addressable lattice`);
    }
    for (let level = 0; level <= LEVELS; level += 1) occupied[level]!.add(key(x >> level, y >> level, z >> level));
  }
  const { lattice } = options;
  return {
    refineEnvironmentLeaf(level, coordinate) {
      if (level >= options.maximumDepth) return false;
      const edge = options.nodeEdge_m[level];
      if (!edge) return false;
      const node = [coordinate.x, coordinate.y, coordinate.z];
      const low = [0, 0, 0], high = [0, 0, 0];
      let span = 1;
      for (let axis = 0; axis < 3; axis += 1) {
        const minimum = options.worldOrigin_m[axis]! + node[axis]! * edge[axis]!;
        const page = PAGE_CELLS * lattice.cellSize_m[axis]!;
        // One voxel either side: the reach of a face the node only touches.
        low[axis] = Math.floor((minimum - lattice.origin_m[axis]!) / page - 1 / PAGE_CELLS);
        high[axis] = Math.floor((minimum + edge[axis]! - lattice.origin_m[axis]!) / page + 1 / PAGE_CELLS);
        if (high[axis]! < -BIAS || low[axis]! >= BIAS) return false;
        low[axis] = Math.max(-BIAS, low[axis]!); high[axis] = Math.min(BIAS - 1, high[axis]!);
        span = Math.max(span, high[axis]! - low[axis]! + 1);
      }
      // At the level whose cells are at least the span, the range covers two an axis.
      const pyramid = Math.min(LEVELS, Math.ceil(Math.log2(span)));
      const cells = occupied[pyramid]!;
      for (let z = low[2]! >> pyramid; z <= high[2]! >> pyramid; z += 1) {
        for (let y = low[1]! >> pyramid; y <= high[1]! >> pyramid; y += 1) {
          for (let x = low[0]! >> pyramid; x <= high[0]! >> pyramid; x += 1) {
            if (cells.has(key(x, y, z))) return true;
          }
        }
      }
      return false;
    },
  };
}
