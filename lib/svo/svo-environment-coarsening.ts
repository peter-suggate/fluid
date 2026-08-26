/**
 * How coarse an environment leaf may be on a scene the solver owns.
 *
 * A simulated container pins every one of its bricks at the solver's own level,
 * so for the whole life of the wet render path there was exactly one resolution
 * knob — `finestCellSize_m` — and it applied to the entire domain. The set is
 * authored in metres and paid for in *those* cells, which is why an 8 m tank at
 * 25 mm could not be staged at all: `ocean-seiche`'s floor is a 21.7 m plate,
 * and drawing it at the water's own 25 mm cost 62 752 environment bricks, a
 * 1 959 MiB octree arena and 79 507 opacity pages, against 324 bricks and 410
 * pages for the same set around the 1.2 m reference tank. The catalog's answer
 * was `studioStageFits`: over a brick budget the scene kept a bare shell and no
 * floor at all.
 *
 * That trade was never between a good picture and a cheap one. The set is one
 * similarity transform of itself at every container size — the pool covers the
 * plan diagonal, the floor is 1.8 pool radii — so a floor ten times larger
 * wants voxels ten times larger, and drawing it at the *solver's* cell buys a
 * resolution nothing in the frame can use. What was missing is the statement of
 * that: a rule saying how fine each authored solid actually has to be drawn.
 *
 * ---------------------------------------------------------------------------
 * The rule
 * ---------------------------------------------------------------------------
 * A primitive is never drawn at a voxel coarser than the smallest feature the
 * voxel payload still owns divided by {@link SVO_ENVIRONMENT_FEATURE_VOXELS}.
 * For ordinary solids that is the geometric minimum. For an admitted planar
 * terminal the exact record owns thickness, so the payload owns only its
 * smaller in-plane feature. The floor therefore coarsens while a lamp stem
 * beside it does not. Nodes no authored solid reaches terminate at the coarsest
 * offered level.
 *
 * The size is the *primitive's*, not its bounding box's. A rotated capsule's
 * box is larger than the capsule and a cone's box says nothing about its
 * throat, and both of those are questions about how finely the surface has to
 * be recorded rather than about where it is.
 *
 * ---------------------------------------------------------------------------
 * Why the ceiling is `log2(brickSize)` and not a bigger number
 * ---------------------------------------------------------------------------
 * A leaf `p` levels above the finest holds voxels of `2^p` cells, and its own
 * origin is a whole number of `2^p` bricks from the scene lattice minimum —
 * which `planSparseSceneDomain` aligns to `brickSize`. So a coarse leaf's voxel
 * planes are at multiples of `2^p` cells from that minimum, and every plane the
 * *brick* lattice has is still a plane of the coarse leaf exactly while
 * `2^p` divides `brickSize`.
 *
 * That is what keeps the stage's own load-bearing detail: its floor's top face
 * is authored flush with `y = 0`, and `y = 0` is a whole number of bricks from
 * the lattice minimum by construction. At `p <= log2(brickSize)` the face stays
 * on a voxel plane and the slab is recorded exactly; one power further and the
 * top of the boards lands up to half a voxel away from the plane the water
 * meets. Half a voxel of floor is not a resolution difference, it is the floor
 * in the wrong place, so the ladder stops where the arithmetic does.
 *
 * ---------------------------------------------------------------------------
 * Scope
 * ---------------------------------------------------------------------------
 * The wet path only. A solverless world already has two ladders — extra levels
 * below the lattice (`environmentRefinementDepth`) and buried-ground coarsening
 * — each with its own predicate and its own gate, and mixing a third meaning
 * into one power is what the planner's `minimumEnvironmentLevel` comment warns
 * against. This is the ladder for the path that had none.
 */
import type { EnvironmentProxyPrimitive } from "../core/voxel-environments";
import type { SceneDescription } from "../core/model";
import { sceneCellSizes_m } from "../core/scene-lattice";
import {
  SOLID_WORLD_BRICK_CELLS,
  SOLID_WORLD_TERRAIN_MATERIAL_ID,
  type SolidWorld,
} from "../core/solid-world";
import { SPARSE_BRICK_GPU_LAYOUT, type SparseBrickCoordinate } from "./sparse-brick-octree";
import { isSvoPlanarBoundaryProxy } from "./svo-planar-boundary";

/**
 * Voxels the smallest feature of an authored solid must span.
 *
 * Four samples across an unresolved feature. Planar terminals do not apply it
 * to thickness: their exact slab record carries both faces independent of the
 * leaf's voxel width.
 *
 * Not a quality slider. Raising it costs leaves everywhere and changes no
 * silhouette; lowering it puts authored solids under the sampling floor of the
 * thing that draws them.
 */
export const SVO_ENVIRONMENT_FEATURE_VOXELS = 4;

/**
 * The smallest feature still represented by voxels, in metres.
 *
 * Read off the primitive's own parameters rather than its AABB, because the box
 * is an extent and this is a *size*: a rotated capsule's box is larger than the
 * capsule, and a truncated cone's box says nothing about the throat that is the
 * finest thing on it.
 *
 * The two procedural kinds return zero — "resolve me at the finest level there
 * is". Their envelope is a bound on where the solid may be and not a
 * description of what it is: a cluster's lobes and a field program's tape are
 * both arbitrarily finer than the ellipsoid or box that contains them, so the
 * envelope would answer a question about placement with a claim about detail.
 * Over-refining is the safe direction here exactly as it is for their bounds.
 */
export function environmentProxyFeatureSize_m(primitive: EnvironmentProxyPrimitive): number {
  switch (primitive.kind) {
    case "box": {
      const dimensions = [primitive.halfSize_m.x, primitive.halfSize_m.y,
        primitive.halfSize_m.z].sort((left, right) => left - right);
      // A promoted slab retains its thin dimension analytically. Its resolution
      // floor is therefore the smaller in-plane feature, not thickness.
      return 2 * (isSvoPlanarBoundaryProxy(primitive) ? dimensions[1] : dimensions[0]);
    }
    case "cylinder":
      return 2 * Math.min(primitive.radius_m, primitive.halfHeight_m);
    case "ellipsoid":
      return 2 * Math.min(primitive.radius_m.x, primitive.radius_m.y, primitive.radius_m.z);
    case "capsule":
      return 2 * primitive.radius_m;
    case "torus":
      return 2 * primitive.minorRadius_m;
    case "cone":
      return 2 * Math.min(primitive.baseRadius_m, primitive.topRadius_m, primitive.halfHeight_m);
    case "cluster":
    case "field-program":
      return 0;
  }
}

/**
 * Levels above the finest an environment leaf may terminate at.
 *
 * The alignment ceiling from the header, as a function of the one number it
 * depends on. Every level of it is only *offered*: {@link
 * createSvoEnvironmentCoarsening} is what decides how much of it each node
 * takes, and a scene whose solids are all fine takes none.
 */
export function svoEnvironmentCoarseningPower(brickSize: number): number {
  if (!Number.isSafeInteger(brickSize) || brickSize < 1 || (brickSize & (brickSize - 1)) !== 0) {
    throw new RangeError("Environment coarsening brick size must be a positive power of two");
  }
  return Math.log2(brickSize);
}

/** A claim that is not an authored proxy, with the feature size it has to resolve. */
export interface SvoEnvironmentCoarseningRegion {
  readonly minimum_m: readonly [number, number, number];
  readonly maximum_m: readonly [number, number, number];
  /** Zero pins the region's nodes at the finest level. */
  readonly feature_m: number;
}

/**
 * Finest-level refinement hints for pages that own a terrain/air interface.
 *
 * A coarsened SolidWorld leaf samples one source cell at the centre of each
 * coarse render voxel. That is conservative for buried volume, but it turns a
 * heightfield into one large step per coarse voxel. Only exposed terrain pages
 * need to retain the source lattice; pages below them may still coarsen.
 */
export function solidWorldTerrainSurfaceCoarseningRegions(
  scene: SceneDescription,
  world: SolidWorld,
): SvoEnvironmentCoarseningRegion[] {
  if (!scene.terrain) return [];
  const cell = sceneCellSizes_m(scene);
  const origin = [-0.5 * scene.container.width_m, 0,
    -0.5 * scene.container.depth_m] as const;
  return world.pages.flatMap((page) => {
    let exposed = false;
    for (let local = 0; local < page.materialId.length && !exposed; local += 1) {
      if (page.materialId[local] !== SOLID_WORLD_TERRAIN_MATERIAL_ID
        || page.solidFraction[local] === 0) continue;
      const y = Math.floor(local / SOLID_WORLD_BRICK_CELLS)
        % SOLID_WORLD_BRICK_CELLS;
      if (y + 1 < SOLID_WORLD_BRICK_CELLS) {
        exposed = page.solidFraction[local + SOLID_WORLD_BRICK_CELLS] === 0;
        continue;
      }
      const x = local % SOLID_WORLD_BRICK_CELLS;
      const z = Math.floor(local / (SOLID_WORLD_BRICK_CELLS ** 2));
      const abovePage = world.directory.lookup([
        page.coordinate[0], page.coordinate[1] + 1, page.coordinate[2],
      ]);
      exposed = abovePage === undefined
        || world.pages[abovePage]!.solidFraction[
          x + SOLID_WORLD_BRICK_CELLS * (SOLID_WORLD_BRICK_CELLS * z)
        ] === 0;
    }
    if (!exposed) return [];
    const pageMinimum = page.coordinate.map((value, axis) => origin[axis]!
      + value * SOLID_WORLD_BRICK_CELLS * cell[axis]!) as [number, number, number];
    const pageMaximum = pageMinimum.map((value, axis) => value
      + SOLID_WORLD_BRICK_CELLS * cell[axis]!) as [number, number, number];
    return [{
      // These are refinement hints, not conservative geometry bounds. Keep
      // them inside their source page so inclusive overlap does not also pin
      // every face-touching neighbour.
      minimum_m: pageMinimum.map((value, axis) =>
        value + 0.25 * cell[axis]!) as [number, number, number],
      maximum_m: pageMaximum.map((value, axis) =>
        value - 0.25 * cell[axis]!) as [number, number, number],
      feature_m: 0,
    }];
  });
}

export interface SvoEnvironmentCoarseningOptions {
  readonly primitives: readonly EnvironmentProxyPrimitive[];
  /** Static rigid bodies and anything else claiming bricks outside the catalog. */
  readonly regions?: readonly SvoEnvironmentCoarseningRegion[];
  readonly worldOrigin_m: readonly [number, number, number];
  /** Node edge in metres, indexed by level. The planner's own table. */
  readonly nodeEdge_m: readonly (readonly number[])[];
  readonly brickSize: number;
  readonly maximumDepth: number;
  /** Overlapping primitive boxes above which the node splits regardless. */
  readonly crowdingTarget: number;
}

export interface SvoEnvironmentCoarseningStatistics {
  /** Predicate invocations. */
  nodes: number;
  /** Nodes that stayed coarse with no authored solid in them. */
  emptyLeaves: number;
  /** Nodes that stayed coarse holding solids this level resolves. */
  resolvedLeaves: number;
  /** Nodes split because a solid in them is finer than this level records. */
  featureSplits: number;
  /** Nodes split because more solids overlap them than a leaf binds. */
  crowdingSplits: number;
}

export interface SvoEnvironmentCoarsening {
  refineEnvironmentLeaf(level: number, coordinate: SparseBrickCoordinate): boolean;
  readonly statistics: SvoEnvironmentCoarseningStatistics;
}

/**
 * The predicate the planner descends with, on a scene the solver owns.
 *
 * Three reasons to split, unioned in their evaluation order.
 *
 *   1. A solid whose smallest feature this level cannot record.
 *   2. More overlapping solids than the voxeliser binds per leaf — the surplus
 *      is dropped silently, so a coarse leaf that gathers a crowd is worse than
 *      the leaves it replaced.
 *   3. A non-catalog claim finer than this level, for the same reason as 1.
 *
 * Everything else stays coarse, which on a staged tank is the plate, the air
 * over it and nothing else.
 */
export function createSvoEnvironmentCoarsening(
  options: SvoEnvironmentCoarseningOptions,
): SvoEnvironmentCoarsening {
  const {
    primitives, regions = [], worldOrigin_m: worldOrigin, nodeEdge_m,
    brickSize, maximumDepth, crowdingTarget,
  } = options;
  const statistics: SvoEnvironmentCoarseningStatistics = {
    nodes: 0, emptyLeaves: 0, resolvedLeaves: 0,
    featureSplits: 0, crowdingSplits: 0,
  };
  /**
   * Bounds and feature sizes flattened once.
   *
   * The predicate is asked per node and scans this per ask, so the loop body is
   * seven typed-array reads rather than an object dereference and six property
   * loads. Same reason `createSvoEnvironmentRefinement` flattens its own.
   */
  const claims = [
    ...primitives.map((primitive) => ({
      min: [primitive.aabb_m.min.x, primitive.aabb_m.min.y, primitive.aabb_m.min.z],
      max: [primitive.aabb_m.max.x, primitive.aabb_m.max.y, primitive.aabb_m.max.z],
      feature_m: environmentProxyFeatureSize_m(primitive),
    })),
    ...regions.map((region) => ({
      min: [region.minimum_m[0], region.minimum_m[1], region.minimum_m[2]],
      max: [region.maximum_m[0], region.maximum_m[1], region.maximum_m[2]],
      feature_m: region.feature_m,
    })),
  ];
  const count = claims.length;
  const bounds = new Float64Array(count * 6);
  const features = new Float64Array(count);
  for (let index = 0; index < count; index += 1) {
    const claim = claims[index];
    bounds[index * 6] = claim.min[0]; bounds[index * 6 + 1] = claim.min[1]; bounds[index * 6 + 2] = claim.min[2];
    bounds[index * 6 + 3] = claim.max[0]; bounds[index * 6 + 4] = claim.max[1]; bounds[index * 6 + 5] = claim.max[2];
    features[index] = claim.feature_m;
  }

  return {
    statistics,
    refineEnvironmentLeaf(level: number, coordinate: SparseBrickCoordinate): boolean {
      statistics.nodes += 1;
      const edge = nodeEdge_m[level];
      // The largest axis: the leaf's voxels are one per brick cell on each axis,
      // and a feature has to survive the coarsest of the three.
      const voxel_m = Math.max(edge[0], edge[1], edge[2]) / brickSize;
      const resolves_m = voxel_m * SVO_ENVIRONMENT_FEATURE_VOXELS;
      const loX = worldOrigin[0] + coordinate.x * edge[0], hiX = loX + edge[0];
      const loY = worldOrigin[1] + coordinate.y * edge[1], hiY = loY + edge[1];
      const loZ = worldOrigin[2] + coordinate.z * edge[2], hiZ = loZ + edge[2];
      let overlapping = 0;
      for (let index = 0; index < count; index += 1) {
        const base = index * 6;
        if (bounds[base + 3] < loX || bounds[base] > hiX) continue;
        if (bounds[base + 4] < loY || bounds[base + 1] > hiY) continue;
        if (bounds[base + 5] < loZ || bounds[base + 2] > hiZ) continue;
        if (features[index] < resolves_m) {
          statistics.featureSplits += 1;
          return true;
        }
        overlapping += 1;
        if (overlapping > crowdingTarget) {
          statistics.crowdingSplits += 1;
          return true;
        }
      }
      if (overlapping === 0) statistics.emptyLeaves += 1;
      else statistics.resolvedLeaves += 1;
      return false;
    },
  };
}

/**
 * The voxel payload one leaf brick costs, over the five lanes it is stored in.
 *
 * Two geometry lanes, one velocity lane and two material/owner lanes — the same
 * arithmetic `tools/svo-fine-voxel-capacity.ts` prices a lattice with, read off
 * the layout rather than restated, so a lane added to the ABI moves both.
 */
export function svoEnvironmentLeafPayloadBytes(brickSize: number): number {
  return brickSize ** 3 * (
    2 * SPARSE_BRICK_GPU_LAYOUT.geometryStrideBytes
    + SPARSE_BRICK_GPU_LAYOUT.velocityStrideBytes
    + 2 * SPARSE_BRICK_GPU_LAYOUT.materialOwnerStrideBytes);
}

/**
 * What an authored set will cost, before any of it is built.
 *
 * The octree is the honest answer and it is far too expensive to be the one a
 * viewport asks every frame, so this is the cheap ceiling that stands in for it:
 * each solid claims its own bounding box, at the leaf size the legibility floor
 * gives *that* solid. Summing per solid rather than measuring the union's span
 * is the whole point — the union of a 25 m plate and a 3 m lamp is a 25 m cube
 * whichever rung either of them lands on, while the sum charges the plate for a
 * plate and the lamp for a lamp, which is what the planner actually does.
 *
 * It is a ceiling in three directions and a floor in none: a box's bounding box
 * is the box, but a cone's is three times the cone, empty leaves inside a
 * closed shell are counted, and two solids sharing a leaf are counted twice.
 * That is the safe direction for a gate whose failure mode is building a set
 * too large for the device and getting no picture at all.
 *
 * Reported in bytes because bytes are what runs out. The count of leaves is not
 * a quantity anything else in the system is denominated in, and the two device
 * limits this has to stay under — `maxBufferSize` for the payload allocation
 * and `maxStorageBufferBindingSize` for the lanes bound inside it — are both
 * byte limits.
 */
export function svoEnvironmentPayloadBytes(
  primitives: readonly EnvironmentProxyPrimitive[],
  options: { readonly cellSize_m: number; readonly brickSize: number },
): number {
  const { cellSize_m, brickSize } = options;
  const ceiling = svoEnvironmentCoarseningPower(brickSize);
  let leaves = 0;
  for (const primitive of primitives) {
    const feature_m = environmentProxyFeatureSize_m(primitive);
    // A solid with no feature size of its own — a cluster, a field program —
    // resolves at the finest level there is, exactly as the predicate has it.
    const power = feature_m > 0
      ? Math.min(ceiling, Math.max(0, Math.floor(
        Math.log2(feature_m / (SVO_ENVIRONMENT_FEATURE_VOXELS * cellSize_m)))))
      : 0;
    const leafEdge_m = cellSize_m * 2 ** power * brickSize;
    const { min, max } = primitive.aabb_m;
    leaves += Math.max(1, Math.ceil((max.x - min.x) / leafEdge_m))
      * Math.max(1, Math.ceil((max.y - min.y) / leafEdge_m))
      * Math.max(1, Math.ceil((max.z - min.z) / leafEdge_m));
  }
  return leaves * svoEnvironmentLeafPayloadBytes(brickSize);
}
