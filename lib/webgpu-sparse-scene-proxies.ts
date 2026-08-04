import {
  SPARSE_BRICK_INVALID_INDEX,
  SPARSE_BRICK_NO_OWNER,
  packMaterialOwner,
  sparseBrickDispatchDimensions,
  type SparseBrickOctreeGPU,
} from "./sparse-brick-octree";
import { SVO_BRICK_LIFECYCLE, SVO_BRICK_OCCUPANCY } from "./svo-brick-occupancy";
import type { EnvironmentProxyPrimitive } from "./voxel-environments";
import {
  sampleSvoPrimitive,
  svoClusterFeatureRadius_m,
  svoPrimitiveWGSL,
  type SvoPrimitiveDescriptor,
  type SvoSmoothUnionClusterPacking,
} from "./svo-primitive-abi";
import { SVO_CLUSTER_ARENA_BLOCK, packSvoClusterArena, svoClusterArenaDecodeWGSL } from "./svo-cluster-arena";
import type { SparseSceneTerrainField } from "./sparse-scene-terrain-field";
import { VOXEL_MATERIAL_IDS } from "./voxel-scene";

/**
 * The identity a ground voxel publishes: the shared terrain material and no
 * owner, exactly as the solver's own dense terrain bake does.
 *
 * No owner is the load-bearing half. `traceLeafPayload` upgrades a solid voxel
 * to its record's exact surface and rejects `SPARSE_BRICK_NO_OWNER`, so ground
 * voxels are invisible to primary visibility and the analytic marcher stays the
 * sole depth authority for the ground — the pyramid gains the ground without
 * the frame gaining a blocky one.
 */
export const SPARSE_SCENE_TERRAIN_MATERIAL_OWNER =
  packMaterialOwner(VOXEL_MATERIAL_IDS.terrain, SPARSE_BRICK_NO_OWNER);

export type SparseSceneVector3 = readonly [number, number, number];
export type SparseSceneQuaternion = readonly [number, number, number, number];

export const SPARSE_SCENE_PRIMITIVE_STRIDE_BYTES = 48;

/**
 * Voxelization keeps its own compact vocabulary rather than sharing the render
 * ABI's records: this pass evaluates every primitive at every candidate voxel,
 * so it wants the cheapest distance that still classifies a cell, where the
 * renderer wants the exact one. Tags are its own; only the shapes are shared.
 *
 * Shapes 1–6 are that bargain, and they keep it: a box, a cylinder, an
 * ellipsoid, a capsule, a torus and a cone all have a closed-form conservative
 * distance that costs a handful of instructions, and the renderer's exact form
 * has the same zero set. Nothing was ever lost by evaluating the cheap one.
 *
 * Shapes 7–9 exist because three kinds *were* being lost. They had no cheap
 * equivalent, so the adapters below downgraded them to a nearby shape — and the
 * downgrade was the problem, not the cost:
 *
 *  - `smooth-union-cluster` became its solid envelope ellipsoid, which turned
 *    the bonsai's fissured crown into a blob in voxel space and therefore in
 *    every shadow and GI ray that reads the opacity pyramid. This is the
 *    expensive one: the hero garden publishes 208 aggregates out of 501
 *    records, and the crown alone is 65 % of the frame's field evaluations.
 *  - `round-cone` became a truncated `cone` on the scenery path (and its widest
 *    capsule on the descriptor path). The truncated form's extent stops at the
 *    authored half height, so it was cutting the swept endpoint spheres out of
 *    voxel space entirely — 40 mm of the garden's coping runs.
 *  - `rounded-cylinder` became its unfilleted outer, which over-occludes the
 *    rim. Conservative, and wrong.
 *
 * These three evaluate the *render ABI's own* distance (`svoPrimitiveDistance_m`),
 * against the real packing, so voxel content and drawn surface finally describe
 * the same solid.
 */
export const SPARSE_SCENE_PRIMITIVE_TYPES = Object.freeze({
  box: 1,
  cylinder: 2,
  ellipsoid: 3,
  capsule: 4,
  torus: 5,
  cone: 6,
  "smooth-union-cluster": 7,
  "round-cone": 8,
  "rounded-cylinder": 9,
} as const);

/**
 * Aggregate packings a live publication may carry.
 *
 * Must equal the renderer's own cluster arena, `SVO_DRY_SCENE_CLUSTER_CAPACITY`,
 * so a scene that draws cannot fail to voxelize for want of a block. The two are
 * separate literals rather than one derived constant **because importing the
 * renderer here is a cycle** — `webgpu-svo-dry-scene` reaches this module through
 * `webgpu-octree-sparse-bricks`, and the import resolves `undefined` at module
 * init (a TDZ throw on `SPARSE_SCENE_MAINTENANCE_STATE_WORDS`, if you try it).
 * The equality is therefore enforced by the `cluster-arena-capacity` check in
 * `tools/run-svo-dry-render-smoke.ts`, which can import both safely.
 *
 * They had already drifted once at 1_024 apiece: the 10x acceptance scene
 * publishes 948 aggregates, so one more tile would have thrown here for a scene
 * the renderer could draw. The hero garden publishes 208 blocks against 501
 * records, and a block is 192 bytes to a record's 48, so the arena is sized
 * rather than assumed away.
 */
export const SPARSE_SCENE_CLUSTER_CAPACITY = 4_096;

/** Type word layout: the low byte is the shape, the rest is a cluster slot. */
const PRIMITIVE_TYPE_MASK = 0xff;
const PRIMITIVE_CLUSTER_SLOT_SHIFT = 8;

interface SparseScenePrimitiveBase {
  center: SparseSceneVector3;
  materialId: number;
  ownerId?: number;
}

export interface SparseSceneBoxPrimitive extends SparseScenePrimitiveBase {
  kind: "box";
  halfExtents: SparseSceneVector3;
  /** Quaternion order is xyzw. */
  orientation?: SparseSceneQuaternion;
}

export interface SparseSceneCylinderPrimitive extends SparseScenePrimitiveBase {
  kind: "cylinder";
  radius: number;
  halfHeight: number;
  /** Local cylinder axis is +Y. Quaternion order is xyzw. */
  orientation?: SparseSceneQuaternion;
}

export interface SparseSceneEllipsoidPrimitive extends SparseScenePrimitiveBase {
  kind: "ellipsoid";
  radii: SparseSceneVector3;
  /** Quaternion order is xyzw. */
  orientation?: SparseSceneQuaternion;
}

/** Swept circle along the local +Y segment. Quaternion order is xyzw. */
export interface SparseSceneCapsulePrimitive extends SparseScenePrimitiveBase {
  kind: "capsule";
  radius: number;
  halfLength: number;
  orientation?: SparseSceneQuaternion;
}

/** Ring swept about the local +Y axis. Quaternion order is xyzw. */
export interface SparseSceneTorusPrimitive extends SparseScenePrimitiveBase {
  kind: "torus";
  majorRadius: number;
  minorRadius: number;
  orientation?: SparseSceneQuaternion;
}

/** Truncated cone about the local +Y axis: `baseRadius` at -Y, `topRadius` at +Y. */
export interface SparseSceneConePrimitive extends SparseScenePrimitiveBase {
  kind: "cone";
  baseRadius: number;
  topRadius: number;
  halfHeight: number;
  orientation?: SparseSceneQuaternion;
}

/**
 * A tapered capsule: the render ABI's `round-cone`, not the truncated `cone`
 * this pass used to substitute. The substitution was not merely coarse, it was
 * *narrow* — a truncated cone's extent stops at the authored half height, so
 * the swept endpoint spheres were outside every dirty region that named it.
 */
export interface SparseSceneRoundConePrimitive extends SparseScenePrimitiveBase {
  kind: "round-cone";
  baseRadius: number;
  topRadius: number;
  halfHeight: number;
  orientation?: SparseSceneQuaternion;
}

/** A cylinder with a filleted rim. `radius`/`halfHeight` are the outer extents. */
export interface SparseSceneRoundedCylinderPrimitive extends SparseScenePrimitiveBase {
  kind: "rounded-cylinder";
  radius: number;
  halfHeight: number;
  edgeRadius: number;
  orientation?: SparseSceneQuaternion;
}

/**
 * A procedural field clipped to an ellipsoidal envelope — the bonsai's crown,
 * the stone sets, the swept tubes.
 *
 * `lobeRadii` is the envelope and remains what every bounds formula sees, so
 * nothing about dirty regions, binning or topology changes by voxelizing the
 * field instead of the envelope. Only the per-voxel distance changes, and it
 * changes from "the whole ellipsoid is solid" to what the renderer draws.
 */
export interface SparseSceneSmoothUnionClusterPrimitive extends SparseScenePrimitiveBase {
  kind: "smooth-union-cluster";
  lobeRadii: SparseSceneVector3;
  packing: SvoSmoothUnionClusterPacking;
  orientation?: SparseSceneQuaternion;
}

export type SparseScenePrimitive =
  | SparseSceneBoxPrimitive
  | SparseSceneCylinderPrimitive
  | SparseSceneEllipsoidPrimitive
  | SparseSceneCapsulePrimitive
  | SparseSceneTorusPrimitive
  | SparseSceneConePrimitive
  | SparseSceneRoundConePrimitive
  | SparseSceneRoundedCylinderPrimitive
  | SparseSceneSmoothUnionClusterPrimitive;

/** The kinds this pass evaluates through the render ABI rather than its own SDF library. */
export function sparseSceneAbiTruePrimitive(
  primitive: SparseScenePrimitive,
): primitive is SparseSceneRoundConePrimitive | SparseSceneRoundedCylinderPrimitive | SparseSceneSmoothUnionClusterPrimitive {
  return primitive.kind === "round-cone" || primitive.kind === "rounded-cylinder"
    || primitive.kind === "smooth-union-cluster";
}

/**
 * The render descriptor an ABI-true primitive evaluates as.
 *
 * One function, used by both the CPU mirror and the arena packer, so the
 * dimension lanes cannot mean one thing in a test and another on the GPU. The
 * identity fields are placeholders: this pass carries material and owner in its
 * own packed word and never reads them back through the ABI.
 */
function abiDescriptorForPrimitive(
  primitive: SparseSceneRoundConePrimitive | SparseSceneRoundedCylinderPrimitive | SparseSceneSmoothUnionClusterPrimitive,
  clusterReference = 0,
): SvoPrimitiveDescriptor {
  const [x, y, z, w] = normalizedQuaternion(primitive.orientation);
  const identity = {
    primitiveId: 0,
    materialId: primitive.materialId,
    ownerId: primitive.ownerId ?? SPARSE_BRICK_NO_OWNER,
    center_m: { x: primitive.center[0], y: primitive.center[1], z: primitive.center[2] },
    orientation: { x, y, z, w },
  };
  if (primitive.kind === "round-cone") {
    return {
      ...identity, kind: "round-cone",
      baseRadius_m: primitive.baseRadius, topRadius_m: primitive.topRadius, halfHeight_m: primitive.halfHeight,
    };
  }
  if (primitive.kind === "rounded-cylinder") {
    return {
      ...identity, kind: "rounded-cylinder",
      radius_m: primitive.radius, halfHeight_m: primitive.halfHeight, edgeRadius_m: primitive.edgeRadius,
    };
  }
  return {
    ...identity, kind: "smooth-union-cluster",
    lobeRadii_m: { x: primitive.lobeRadii[0], y: primitive.lobeRadii[1], z: primitive.lobeRadii[2] },
    clusterReference,
    packing: primitive.packing,
  };
}

/** Keyed transform/content replacement for one primitive in the live sparse scene. */
export interface SparseScenePrimitiveUpdate {
  readonly key: string;
  readonly primitive: SparseScenePrimitive;
}

/** Adapt the analytic render descriptor into the shared live voxel vocabulary. */
export function sparseScenePrimitiveForSvoDescriptor(descriptor: SvoPrimitiveDescriptor): SparseScenePrimitive {
  if (descriptor.kind === "terrain-heightfield") throw new RangeError("Terrain heightfields are not finite live primitive updates");
  const center: SparseSceneVector3 = [descriptor.center_m.x, descriptor.center_m.y, descriptor.center_m.z];
  const orientation: SparseSceneQuaternion | undefined = "orientation" in descriptor && descriptor.orientation
    ? [descriptor.orientation.x, descriptor.orientation.y, descriptor.orientation.z, descriptor.orientation.w]
    : undefined;
  const base = { center, orientation, materialId: descriptor.materialId, ownerId: descriptor.ownerId };
  if (descriptor.kind === "sphere") return { ...base, kind: "ellipsoid", radii: [descriptor.radius_m, descriptor.radius_m, descriptor.radius_m] };
  if (descriptor.kind === "box") return { ...base, kind: "box", halfExtents: [descriptor.halfExtents_m.x, descriptor.halfExtents_m.y, descriptor.halfExtents_m.z] };
  if (descriptor.kind === "capsule") return { ...base, kind: "capsule", radius: descriptor.radius_m, halfLength: descriptor.segmentHalfLength_m };
  if (descriptor.kind === "cylinder") return { ...base, kind: "cylinder", radius: descriptor.radius_m, halfHeight: descriptor.halfHeight_m };
  if (descriptor.kind === "rounded-cylinder") return {
    ...base, kind: "rounded-cylinder", radius: descriptor.radius_m,
    halfHeight: descriptor.halfHeight_m, edgeRadius: descriptor.edgeRadius_m,
  };
  if (descriptor.kind === "torus") return { ...base, kind: "torus", majorRadius: descriptor.majorRadius_m, minorRadius: descriptor.minorRadius_m };
  if (descriptor.kind === "cone") return { ...base, kind: "cone", baseRadius: descriptor.baseRadius_m, topRadius: descriptor.topRadius_m, halfHeight: descriptor.halfHeight_m };
  if (descriptor.kind === "round-cone") return {
    ...base, kind: "round-cone", baseRadius: descriptor.baseRadius_m,
    topRadius: descriptor.topRadius_m, halfHeight: descriptor.halfHeight_m,
  };
  // An aggregate voxelizes as the field the renderer draws, not as its
  // envelope. The envelope was the old answer, and its justification was that a
  // single centre sample under a planar coverage law *under*-reports a fissured
  // packing, which is indistinguishable from empty space downstream. That
  // argument is sound about the coverage law and wrong about the fix: the
  // cluster distance is a 1-Lipschitz lower bound, so a cell holding any solid
  // has a centre sample no further than the cell radius, and the conservative
  // sampling below turns that into occupancy the field actually has. Reporting
  // the solid lobe instead over-occluded a canopy that is mostly air, which is
  // what the shadows and the GI have been reading.
  if (descriptor.kind === "smooth-union-cluster") {
    if (!descriptor.packing) {
      throw new RangeError("A live aggregate primitive must carry its packing; an unresolved cluster has no shape");
    }
    return {
      ...base, kind: "smooth-union-cluster",
      lobeRadii: [descriptor.lobeRadii_m.x, descriptor.lobeRadii_m.y, descriptor.lobeRadii_m.z],
      packing: descriptor.packing,
    };
  }
  return { ...base, kind: "ellipsoid", radii: [descriptor.radii_m.x, descriptor.radii_m.y, descriptor.radii_m.z] };
}

/**
 * Adapt one authored scenery proxy into this pass's vocabulary. Kept beside the
 * shapes themselves so a new one cannot reach the render ABI while silently
 * missing from voxelization, which would publish a surface nothing owns.
 */
export function sparseScenePrimitiveForProxy(
  proxy: EnvironmentProxyPrimitive,
  identity: Pick<SparseScenePrimitiveBase, "materialId" | "ownerId">,
): SparseScenePrimitive {
  const center: SparseSceneVector3 = [proxy.center_m.x, proxy.center_m.y, proxy.center_m.z];
  // Scenery authors wxyz; this pass packs xyzw.
  const orientation: SparseSceneQuaternion | undefined = proxy.orientation
    ? [proxy.orientation.x, proxy.orientation.y, proxy.orientation.z, proxy.orientation.w]
    : undefined;
  const base = { ...identity, center, orientation };
  if (proxy.kind === "box") return { ...base, kind: "box", halfExtents: [proxy.halfSize_m.x, proxy.halfSize_m.y, proxy.halfSize_m.z] };
  // The same split the render publication makes: a fillet is a different SDF,
  // not a cylinder with a note attached.
  if (proxy.kind === "cylinder") {
    return proxy.edgeRadius_m !== undefined && proxy.edgeRadius_m > 0
      ? { ...base, kind: "rounded-cylinder", radius: proxy.radius_m, halfHeight: proxy.halfHeight_m, edgeRadius: proxy.edgeRadius_m }
      : { ...base, kind: "cylinder", radius: proxy.radius_m, halfHeight: proxy.halfHeight_m };
  }
  if (proxy.kind === "capsule") return { ...base, kind: "capsule", radius: proxy.radius_m, halfLength: proxy.halfLength_m };
  if (proxy.kind === "torus") return { ...base, kind: "torus", majorRadius: proxy.majorRadius_m, minorRadius: proxy.minorRadius_m };
  if (proxy.kind === "cone") {
    return proxy.roundedEnds
      ? { ...base, kind: "round-cone", baseRadius: proxy.baseRadius_m, topRadius: proxy.topRadius_m, halfHeight: proxy.halfHeight_m }
      : { ...base, kind: "cone", baseRadius: proxy.baseRadius_m, topRadius: proxy.topRadius_m, halfHeight: proxy.halfHeight_m };
  }
  // A cluster voxelizes as its field, clipped to the `radius_m` envelope — the
  // same solid the render ABI draws. It used to voxelize as the envelope
  // itself, which is where the bonsai's crown became a blob; see
  // `sparseScenePrimitiveForSvoDescriptor` for why that trade is off.
  if (proxy.kind === "cluster") {
    return {
      ...base, kind: "smooth-union-cluster",
      lobeRadii: [proxy.radius_m.x, proxy.radius_m.y, proxy.radius_m.z], packing: proxy.packing,
    };
  }
  return { ...base, kind: "ellipsoid", radii: [proxy.radius_m.x, proxy.radius_m.y, proxy.radius_m.z] };
}

export interface SparseSceneCellSample {
  solidSignedDistance: number;
  solidFraction: number;
  materialOwner: number;
}

export interface SparseSceneProxyVoxelizerOptions {
  cellSize: SparseSceneVector3;
  /** World-space position of cell-coordinate (0, 0, 0)'s minimum corner. */
  worldOrigin?: SparseSceneVector3;
  /** Maximum topology level. */
  finestLevel?: number;
  /** Fixed live primitive arena capacity. */
  primitiveCapacity: number;
  /** Maximum old/new world-space regions coalesced into one scene publication. */
  dirtyRegionCapacity: number;
  /** Maximum bricks repaired by one maintenance publication. */
  dirtyBrickCapacity: number;
  /** Fixed candidate bin capacity for each dirty brick. */
  candidatesPerDirtyBrick: number;
  /** Fixed aggregate-packing arena capacity; defaults to {@link SPARSE_SCENE_CLUSTER_CAPACITY}. */
  clusterCapacity?: number;
  /**
   * Coarse spatial index over the authored records, and the dirty-region brick
   * lattice invalidation is dispatched over.
   *
   * Omitting it keeps the original maintenance, which is O(leaves x regions) to
   * invalidate and O(dirty bricks x records) to bin. Both are fine at a few
   * hundred records and neither survives ten times that, which is the whole
   * point of the index; it is optional only so the two can be measured against
   * each other on one device.
   */
  recordIndex?: SparseSceneRecordIndexOptions;
  /**
   * Columns the terrain field may hold, or zero for a world with no ground.
   *
   * One f32 per finest cell of the domain footprint — 24 KB for the hero garden
   * — and it is a capacity rather than the field itself so that a sculpting
   * stroke republishes heights without reallocating anything.
   */
  terrainFieldCapacityColumns?: number;
  /**
   * Maximum dirty bricks one encoded frame repairs. Zero — the default —
   * repairs a publication in one go, exactly as this pass always has.
   *
   * A budgeted publication converges over frames like the light cache, and is
   * *not* observable as complete until its last chunk lands: `completedRevision`
   * is stored by the chunk that reaches the end of the dirty list and by no
   * other. See {@link SPARSE_SCENE_MAINTENANCE_INCOMPLETE_OVERFLOW} for why the
   * distinction between "incomplete" and "complete with less detail" is the one
   * thing this pass may not get wrong.
   */
  bricksPerFrameBudget?: number;
  label?: string;
}

export interface SparseSceneRecordIndexOptions {
  /** Grid dimensions over the declared brick domain. Each axis must be below 1024. */
  dimensions: readonly [number, number, number];
  /**
   * World extent of one grid cell, which must be a whole number of finest
   * bricks so that the lattice shares the octree's alignment and a leaf's cell
   * range can be read off its bounds without ulp sensitivity.
   */
  cellExtent_m: readonly [number, number, number];
  /** Scattered slots. Defaults to sixteen per record. */
  itemCapacity?: number;
  /**
   * A record covering more cells than this is tested against every dirty brick
   * instead of being scattered. The ground plane and the room shell cover the
   * whole grid; scattering them would make the index a copy of the record list.
   */
  maximumCellsPerRecord?: number;
}

export interface SparseSceneAxisAlignedBounds {
  minimum: SparseSceneVector3;
  maximum: SparseSceneVector3;
}

export interface SparseScenePublication {
  primitives: readonly SparseScenePrimitive[];
  /** Union coverage of every old and new primitive bound changed by this revision. */
  dirtyRegions: readonly SparseSceneAxisAlignedBounds[];
  /** Strictly increasing, nonzero render-scene revision. */
  revision: number;
  /**
   * Whether this revision may converge over frames under the configured brick
   * budget. Defaults to true; a caller sets it false for the publication that
   * brings a scene into existence, because there is nothing to show while that
   * one converges and every consumer would read a scene generation of zero.
   */
  budgeted?: boolean;
}

/**
 * Maintenance state, in words, at `stateOffsetBytes` of the maintenance arena.
 *
 * `chunkBegin`/`chunkEnd`/`chunkCursor` are the per-frame voxelization budget.
 * They occupy three of the words the dispatch triples already left spare, so a
 * budgeted publication costs the arena nothing: the state block is still twenty
 * words and is still cleared whole when a *new* revision starts — which is
 * exactly the distinction the budget turns on. A continuation frame must not
 * clear it, because the dirty brick list, the overflow mask and the requested
 * revision are what it is continuing.
 */
export const SPARSE_SCENE_MAINTENANCE_STATE_WORDS = Object.freeze({
  dirtyBrickCount: 0,
  overflowFlags: 1,
  requestedRevision: 2,
  completedRevision: 3,
  primitiveCount: 4,
  dirtyRegionCount: 5,
  /** First dirty brick this frame's chunk repairs. */
  chunkBegin: 6,
  /** One past the last dirty brick this frame's chunk repairs. */
  chunkEnd: 7,
  binDispatch: 8,
  rebuildDispatch: 11,
  finalizeDispatch: 14,
  /** Dirty bricks already repaired by earlier chunks of this revision. */
  chunkCursor: 17,
  /**
   * The busiest brick's candidate count this revision, capped or not.
   *
   * The overflow flag says a ceiling was crossed; this says by how much, which
   * is the difference between "one brick lost a primitive" and "the densest
   * brick binds 442 against 64 slots and most of its geometry is missing from
   * the opacity pyramid". A lane cannot ask for headroom it cannot see.
   */
  maximumBrickCandidates: 18,
  wordCount: 20,
} as const);

/**
 * Flags carried in the spare `cell.w` lane of the parameter block.
 *
 * The uniform is a fixed 96 bytes and both spare lanes are now spent — the
 * budget in `worldOrigin.w`, these flags here — rather than growing the struct
 * for two integers. Everything else the record index needs is *derived* from
 * the arena offsets the block already carries, or read from the index header
 * block inside the arena itself; see `indexHeaderOffset` in the shader.
 */
export const SPARSE_SCENE_MAINTENANCE_FLAGS = Object.freeze({
  /** Invalidation is driven by the dirty regions' brick cells, not by every leaf. */
  indexedInvalidation: 1,
  /** Binning reads the coarse record grid, not every record. */
  indexedBinning: 2,
  /**
   * A terrain column field is configured, so the derived header describing it
   * exists and may be read.
   *
   * A flag rather than a zero test on the header's own dimension words, because
   * the header is not allocated for a world that has neither an index nor
   * terrain — and an out-of-bounds storage read is *clamped*, not zeroed, so
   * "no ground" would have been decided by whatever the arena's last word
   * happened to hold.
   */
  terrainField: 4,
} as const);

/**
 * Words of the derived-block header, which sits immediately after the aggregate
 * arena. Its offset is derived on the GPU from the parameter block's cluster
 * offset and capacity, so everything described here costs no uniform lane.
 *
 * It began as the record index's own header and now also carries the terrain
 * column field, because the uniform's two spare lanes were already spent and
 * these are the only words in the arena whose address the shader can reach
 * without being told. It is allocated unconditionally for the same reason: a
 * world may have terrain and no record index, or the reverse, and a header that
 * exists only sometimes cannot describe either.
 */
export const SPARSE_SCENE_INDEX_HEADER = Object.freeze({
  gridDimensionX: 0,
  gridDimensionY: 1,
  gridDimensionZ: 2,
  globalRecordCount: 3,
  invalidationCellCount: 4,
  gridItemCapacity: 5,
  /** Grid cell extent in metres, as three bitcast f32 lanes. */
  gridExtentX: 6,
  gridExtentY: 7,
  gridExtentZ: 8,
  /** First word of the terrain column field, or anything when it is absent. */
  terrainFieldOffset: 9,
  /**
   * Columns along x and z, at the finest cell size. Zero switches the whole
   * terrain path off — there is no separate enable flag, because a field with
   * no columns and a field that is not there are the same thing.
   */
  terrainFieldDimensionX: 10,
  terrainFieldDimensionZ: 11,
  wordCount: 16,
} as const);

/** Bits per axis in a packed grid-cell coordinate; three of them share one word. */
export const SPARSE_SCENE_GRID_COORDINATE_BITS = 10;

/** Per-region record of the dirty brick-cell box invalidation is dispatched over. */
export const SPARSE_SCENE_REGION_CELL_WORDS = 8;

export const SPARSE_SCENE_MAINTENANCE_OVERFLOW = Object.freeze({
  dirtyBricks: 1,
  candidates: 2,
  topology: 4,
} as const);

/**
 * The overflow flags that mean the revision did not happen, as opposed to the
 * one that means it happened with less detail than was asked for.
 *
 * `dirtyBricks` truncated the brick list and `topology` skipped a relocating
 * brick, so in both cases some brick still holds the previous revision's voxels
 * and no consumer may be told the new one is current. `candidates` is not like
 * that: `rebuildDirtyBrickPayload` writes every voxel of an over-subscribed
 * brick from the first `candidatesPerBrick()` primitives, so the brick *is* the
 * new revision, minus whichever primitives lost the race.
 *
 * Reading all three the same way is what made a density budget into an outage.
 * Three bricks of the hero garden's pebble bands hold more than 64 primitives
 * at its 25 mm lattice (docs/SVO_FINE_VOXEL_CAPACITY.md §4), so every hero frame
 * raised `candidates`, `completedRevision` was never stored, `finalizeScene`
 * never advanced the live generation, and the one-shot pass that certifies the
 * node-mip pages as empty read that zero and certified nothing. Cone visibility
 * then failed closed over exactly the octree domain and the garden rendered
 * with a hard-edged black slab across the container footprint.
 */
export const SPARSE_SCENE_MAINTENANCE_INCOMPLETE_OVERFLOW =
  SPARSE_SCENE_MAINTENANCE_OVERFLOW.dirtyBricks | SPARSE_SCENE_MAINTENANCE_OVERFLOW.topology;

/**
 * CPU mirror of the completion gate the two finalize shaders apply. It exists so
 * the rule has one statement rather than a repeated bit test, and so a test can
 * assert the rule rather than the WGSL text that spells it.
 */
export function sparseSceneRevisionIncomplete(overflowFlags: number): boolean {
  return (overflowFlags & SPARSE_SCENE_MAINTENANCE_INCOMPLETE_OVERFLOW) !== 0;
}

function finiteVector(values: readonly number[], name: string): void {
  if (values.some((value) => !Number.isFinite(value))) throw new RangeError(`${name} must contain finite values`);
}

function positiveVector(values: SparseSceneVector3, name: string): void {
  finiteVector(values, name);
  if (values.some((value) => value <= 0)) throw new RangeError(`${name} must contain positive values`);
}

function validateIdentity(materialId: number, ownerId = SPARSE_BRICK_NO_OWNER): void {
  if (!Number.isInteger(materialId) || materialId < 1 || materialId > 0xffff) {
    throw new RangeError("Scene proxy material ID must be a nonzero uint16");
  }
  if (!Number.isInteger(ownerId) || ownerId < 0 || ownerId > 0xffff) {
    throw new RangeError("Scene proxy owner ID must fit uint16");
  }
}

function normalizedQuaternion(value: SparseSceneQuaternion | undefined): SparseSceneQuaternion {
  if (value === undefined) return [0, 0, 0, 1];
  finiteVector(value, "Primitive orientation");
  const length = Math.hypot(...value);
  if (length <= 1e-12) throw new RangeError("Primitive orientation must have nonzero length");
  return [value[0] / length, value[1] / length, value[2] / length, value[3] / length];
}

function primitiveExtent(primitive: SparseScenePrimitive): SparseSceneVector3 {
  if (primitive.kind === "box") {
    positiveVector(primitive.halfExtents, "Box half extents");
    return primitive.halfExtents;
  }
  if (primitive.kind === "cylinder") {
    if (!Number.isFinite(primitive.radius) || primitive.radius <= 0 ||
        !Number.isFinite(primitive.halfHeight) || primitive.halfHeight <= 0) {
      throw new RangeError("Cylinder radius and half height must be positive");
    }
    return [primitive.radius, primitive.halfHeight, primitive.radius];
  }
  if (primitive.kind === "capsule") {
    if (!Number.isFinite(primitive.radius) || primitive.radius <= 0 ||
        !Number.isFinite(primitive.halfLength) || primitive.halfLength < 0) {
      throw new RangeError("Capsule radius must be positive and its half length non-negative");
    }
    return [primitive.radius, primitive.halfLength, primitive.radius];
  }
  if (primitive.kind === "torus") {
    if (!Number.isFinite(primitive.majorRadius) || !Number.isFinite(primitive.minorRadius)
        || primitive.minorRadius <= 0 || !(primitive.minorRadius < primitive.majorRadius)) {
      throw new RangeError("Torus minor radius must be positive and smaller than its major radius");
    }
    return [primitive.majorRadius, primitive.minorRadius, 0];
  }
  if (primitive.kind === "cone" || primitive.kind === "round-cone") {
    if (!Number.isFinite(primitive.halfHeight) || primitive.halfHeight <= 0
        || !(primitive.baseRadius >= 0) || !(primitive.topRadius >= 0)
        || !(Math.max(primitive.baseRadius, primitive.topRadius) > 0)) {
      throw new RangeError("Cone half height must be positive with a positive radius at one end");
    }
    return [primitive.baseRadius, primitive.halfHeight, primitive.topRadius];
  }
  if (primitive.kind === "rounded-cylinder") {
    if (!Number.isFinite(primitive.radius) || primitive.radius <= 0
        || !Number.isFinite(primitive.halfHeight) || primitive.halfHeight <= 0
        || !Number.isFinite(primitive.edgeRadius) || primitive.edgeRadius < 0
        || primitive.edgeRadius > Math.min(primitive.radius, primitive.halfHeight)) {
      throw new RangeError("Rounded cylinder extents must be positive and contain their fillet");
    }
    return [primitive.radius, primitive.halfHeight, primitive.edgeRadius];
  }
  if (primitive.kind === "smooth-union-cluster") {
    positiveVector(primitive.lobeRadii, "Cluster envelope radii");
    return primitive.lobeRadii;
  }
  positiveVector(primitive.radii, "Ellipsoid radii");
  return primitive.radii;
}

function primitiveType(primitive: SparseScenePrimitive): number {
  return SPARSE_SCENE_PRIMITIVE_TYPES[primitive.kind];
}

function primitiveLocalBounds(primitive: SparseScenePrimitive): SparseSceneVector3 {
  if (primitive.kind === "torus") {
    primitiveExtent(primitive);
    return [primitive.majorRadius + primitive.minorRadius, primitive.minorRadius, primitive.majorRadius + primitive.minorRadius];
  }
  if (primitive.kind === "capsule") {
    primitiveExtent(primitive);
    return [primitive.radius, primitive.halfLength + primitive.radius, primitive.radius];
  }
  if (primitive.kind === "cone") {
    primitiveExtent(primitive);
    const radius = Math.max(primitive.baseRadius, primitive.topRadius);
    return [radius, primitive.halfHeight, radius];
  }
  // Deliberately the render ABI's own extents for the kinds it now evaluates,
  // so a bound cannot say one shape and the distance another. A round cone
  // sweeps its endpoint spheres past the segment; a cluster is its envelope,
  // which is why nothing about dirty regions changed when the field replaced it.
  if (primitive.kind === "round-cone") {
    primitiveExtent(primitive);
    const radius = Math.max(primitive.baseRadius, primitive.topRadius);
    return [radius, primitive.halfHeight + radius, radius];
  }
  if (primitive.kind === "rounded-cylinder") {
    primitiveExtent(primitive);
    return [primitive.radius, primitive.halfHeight, primitive.radius];
  }
  return primitiveExtent(primitive);
}

/** Conservative world-space bounds used to invalidate and bin live scene geometry. */
export function sparseScenePrimitiveBounds(primitive: SparseScenePrimitive): SparseSceneAxisAlignedBounds {
  finiteVector(primitive.center, "Primitive center");
  validateIdentity(primitive.materialId, primitive.ownerId);
  const [x, y, z, w] = normalizedQuaternion(primitive.orientation);
  const local = primitiveLocalBounds(primitive);
  // Absolute rotation matrix maps an oriented local AABB to its conservative world AABB.
  const xx = x * x; const yy = y * y; const zz = z * z;
  const xy = x * y; const xz = x * z; const yz = y * z;
  const wx = w * x; const wy = w * y; const wz = w * z;
  const extent: SparseSceneVector3 = [
    Math.abs(1 - 2 * (yy + zz)) * local[0] + Math.abs(2 * (xy - wz)) * local[1] + Math.abs(2 * (xz + wy)) * local[2],
    Math.abs(2 * (xy + wz)) * local[0] + Math.abs(1 - 2 * (xx + zz)) * local[1] + Math.abs(2 * (yz - wx)) * local[2],
    Math.abs(2 * (xz - wy)) * local[0] + Math.abs(2 * (yz + wx)) * local[1] + Math.abs(1 - 2 * (xx + yy)) * local[2],
  ];
  return {
    minimum: [primitive.center[0] - extent[0], primitive.center[1] - extent[1], primitive.center[2] - extent[2]],
    maximum: [primitive.center[0] + extent[0], primitive.center[1] + extent[1], primitive.center[2] + extent[2]],
  };
}

/**
 * Half-open lattices do not exist here: `boundsOverlap` is inclusive, so a bound
 * lying exactly on a cell face belongs to the cell on both sides of it. Every
 * conversion from metres to a lattice index below is nudged by this much of a
 * cell so that it says so, and always in the direction that includes more.
 *
 * The direction matters more than the magnitude. A widened index box dirties a
 * brick that needed nothing and bins a record that covers nothing, which costs
 * maintenance. A narrowed one drops geometry out of invalidation and out of the
 * opacity pyramid, silently, which is the failure this whole pass is about.
 */
const SPARSE_SCENE_LATTICE_EPSILON = 1e-6;

function latticeSpan(
  minimum: number, maximum: number, origin: number, extent: number, limit: number,
): readonly [number, number] {
  const low = Math.floor((minimum - origin) / extent - SPARSE_SCENE_LATTICE_EPSILON);
  const high = Math.floor((maximum - origin) / extent + SPARSE_SCENE_LATTICE_EPSILON);
  return [Math.max(0, Math.min(limit - 1, low)), Math.max(-1, Math.min(limit - 1, high))];
}

/** The finest brick-cell boxes a publication's dirty regions cover. */
export interface SparseSceneRegionCellPlan {
  /** `SPARSE_SCENE_REGION_CELL_WORDS` per region: base xyz, span xyz, running prefix, cell count. */
  readonly words: Uint32Array<ArrayBuffer>;
  /**
   * Total cells, which is both the invalidation dispatch size and a strict upper
   * bound on the number of leaves the regions can dirty.
   *
   * Strict because leaves are disjoint unions of finest brick cells: a leaf that
   * overlaps a region contains at least one cell of that region's index box, and
   * two leaves never contain the same one. That is what lets the CPU decide how
   * many budgeted frames a publication needs without reading anything back.
   */
  readonly cellCount: number;
}

/**
 * Expand each dirty region into the finest brick lattice.
 *
 * Invalidation used to run one thread per *leaf* and test it against every
 * region — a cost that depends on how big the tree is and how many records
 * moved, and on nothing about how much of the world actually changed. This is
 * the scatter side of the same question: the regions are known on the CPU
 * (`publish` already packs their bounds), so the cells they cover are known too,
 * and the GPU can descend to the leaf that owns each one.
 */
export function planSparseSceneRegionCells(
  regions: readonly SparseSceneAxisAlignedBounds[],
  worldOrigin: SparseSceneVector3,
  cellSize: SparseSceneVector3,
  brickSize: number,
  finestLevel: number,
): SparseSceneRegionCellPlan {
  const limit = finestLevel >= 31 ? 0x7fffffff : 2 ** finestLevel;
  const words = new Uint32Array(new ArrayBuffer(regions.length * SPARSE_SCENE_REGION_CELL_WORDS * 4));
  let prefix = 0;
  regions.forEach((region, index) => {
    const base = index * SPARSE_SCENE_REGION_CELL_WORDS;
    let cells = 1;
    for (let axis = 0; axis < 3; axis += 1) {
      const [low, high] = latticeSpan(
        region.minimum[axis], region.maximum[axis], worldOrigin[axis], cellSize[axis] * brickSize, limit,
      );
      const span = Math.max(0, high - low + 1);
      words[base + axis] = low;
      words[base + 3 + axis] = span;
      cells *= span;
    }
    words[base + 6] = prefix;
    words[base + 7] = cells;
    prefix += cells;
  });
  return { words, cellCount: prefix };
}

/** A coarse uniform grid over the authored records, plus the records too large to sit in it. */
export interface SparseSceneRecordGridPlan {
  /** `gridCells + 1` running offsets into `items`. */
  readonly offsets: Uint32Array<ArrayBuffer>;
  /** Record indices, grouped by cell. Only the used prefix. */
  readonly items: Uint32Array<ArrayBuffer>;
  /** Records tested against every dirty brick: too large to scatter, or arriving after the arena filled. */
  readonly globalRecords: Uint32Array<ArrayBuffer>;
  /**
   * Two words per record: the packed minimum and maximum grid cell it was
   * scattered into.
   *
   * The GPU reads the range the CPU actually used rather than re-deriving it,
   * because a leaf spanning several cells finds the same record once per cell
   * and has to accept exactly one of them. The rule is "accept in the
   * componentwise first cell shared by the record and the brick", which is only
   * exact if both sides agree on the record's range to the cell — and an f32
   * floor of a bound lying on a lattice face does not reliably agree with an f64
   * one. Transmitting it costs eight bytes a record and removes the question.
   */
  readonly cellRanges: Uint32Array<ArrayBuffer>;
}

function packGridCoordinate(x: number, y: number, z: number): number {
  const bits = SPARSE_SCENE_GRID_COORDINATE_BITS;
  return (x | (y << bits) | (z << (2 * bits))) >>> 0;
}

/**
 * Bin the publication's records into the coarse grid a dirty brick will consult.
 *
 * A counting sort, on the CPU, because `publish` is already O(records): it packs
 * every record and every record's bounds each revision. What it must not be is
 * O(dirty bricks x records) on the *GPU*, which is what this removes.
 *
 * Two tiers rather than one. A record whose bounds cover more than
 * `maximumCellsPerRecord` cells — the ground plane, the room shell, the water
 * volume — is not localized by any grid, and scattering it would put a copy of
 * it in most cells and turn the index back into the record list. Those go to a
 * global list every brick reads, which is sound and, for the scenes this is for,
 * short.
 */
export function planSparseSceneRecordGrid(
  bounds: readonly SparseSceneAxisAlignedBounds[],
  worldOrigin: SparseSceneVector3,
  index: SparseSceneRecordIndexOptions,
  itemCapacity: number,
): SparseSceneRecordGridPlan {
  const [dimX, dimY, dimZ] = index.dimensions;
  const cells = dimX * dimY * dimZ;
  const maximumCells = index.maximumCellsPerRecord ?? 64;
  const counts = new Uint32Array(cells);
  const spans = new Int32Array(bounds.length * 6);
  const cellRanges = new Uint32Array(new ArrayBuffer(bounds.length * 2 * 4));
  const global: number[] = [];
  let used = 0;
  for (let record = 0; record < bounds.length; record += 1) {
    let covered = 1;
    for (let axis = 0; axis < 3; axis += 1) {
      const [low, high] = latticeSpan(
        bounds[record].minimum[axis], bounds[record].maximum[axis],
        worldOrigin[axis], index.cellExtent_m[axis], index.dimensions[axis],
      );
      spans[record * 6 + axis] = low;
      spans[record * 6 + 3 + axis] = high;
      covered *= Math.max(0, high - low + 1);
    }
    if (covered === 0 || covered > maximumCells || used + covered > itemCapacity) {
      global.push(record);
      spans[record * 6 + 3] = -1;
      continue;
    }
    used += covered;
    cellRanges[record * 2] = packGridCoordinate(spans[record * 6], spans[record * 6 + 1], spans[record * 6 + 2]);
    cellRanges[record * 2 + 1] = packGridCoordinate(spans[record * 6 + 3], spans[record * 6 + 4], spans[record * 6 + 5]);
    for (let z = spans[record * 6 + 2]; z <= spans[record * 6 + 5]; z += 1) {
      for (let y = spans[record * 6 + 1]; y <= spans[record * 6 + 4]; y += 1) {
        for (let x = spans[record * 6]; x <= spans[record * 6 + 3]; x += 1) counts[x + dimX * (y + dimY * z)] += 1;
      }
    }
  }
  const offsets = new Uint32Array(new ArrayBuffer((cells + 1) * 4));
  let running = 0;
  for (let cell = 0; cell < cells; cell += 1) { offsets[cell] = running; running += counts[cell]; }
  offsets[cells] = running;
  const cursor = offsets.slice(0, cells);
  const items = new Uint32Array(new ArrayBuffer(running * 4));
  for (let record = 0; record < bounds.length; record += 1) {
    if (spans[record * 6 + 3] < spans[record * 6]) continue;
    for (let z = spans[record * 6 + 2]; z <= spans[record * 6 + 5]; z += 1) {
      for (let y = spans[record * 6 + 1]; y <= spans[record * 6 + 4]; y += 1) {
        for (let x = spans[record * 6]; x <= spans[record * 6 + 3]; x += 1) {
          const cell = x + dimX * (y + dimY * z);
          items[cursor[cell]] = record;
          cursor[cell] += 1;
        }
      }
    }
  }
  return { offsets, items, globalRecords: new Uint32Array(global), cellRanges };
}

/**
 * Slot each aggregate's parameter block will occupy in the companion arena.
 *
 * Publication order over the cluster subsequence, derived the same way by the
 * record packer and the arena packer so the two cannot disagree about which
 * block a record names. Non-cluster primitives get `-1` and never look.
 */
export function sparseSceneClusterSlots(primitives: readonly SparseScenePrimitive[]): readonly number[] {
  let next = 0;
  return primitives.map((primitive) => (primitive.kind === "smooth-union-cluster" ? next++ : -1));
}

/** Aggregate packings of a publication, in the order `sparseSceneClusterSlots` assigns. */
export function sparseSceneClusterPackings(
  primitives: readonly SparseScenePrimitive[],
): readonly SvoSmoothUnionClusterPacking[] {
  return primitives.flatMap((primitive) => (primitive.kind === "smooth-union-cluster" ? [primitive.packing] : []));
}

/** The companion arena a publication's aggregates are evaluated against. */
export function packSparseSceneClusterArena(
  primitives: readonly SparseScenePrimitive[],
  capacity = SPARSE_SCENE_CLUSTER_CAPACITY,
): Uint32Array<ArrayBuffer> {
  const packings = sparseSceneClusterPackings(primitives);
  if (packings.length > capacity) {
    throw new RangeError(`Live scene publishes ${packings.length} aggregates, above the arena's ${capacity}`);
  }
  return packSvoClusterArena(packings, capacity);
}

/**
 * Pack three vec4 words per primitive. The fourth lane of the first two vec4s
 * is bitcast on GPU to preserve exact type and material/owner integer identity:
 * `{ center.xyz, type }, { extent.xyz, packedIdentity }, { quaternion.xyzw }`.
 *
 * The type lane carries a second field in its upper 24 bits: the aggregate's
 * block slot in the companion cluster arena. It rides there rather than in a
 * fourth vec4 because the record is otherwise full and widening the stride
 * would cost every primitive sixteen bytes to carry a number only aggregates
 * read — and because a slot is a *property of the shape*, so a record that
 * names a shape with no arena block cannot be constructed by forgetting a lane.
 */
export function packSparseScenePrimitives(primitives: readonly SparseScenePrimitive[]): Uint32Array<ArrayBuffer> {
  const words = new Uint32Array(new ArrayBuffer(primitives.length * SPARSE_SCENE_PRIMITIVE_STRIDE_BYTES));
  const floats = new Float32Array(words.buffer);
  const clusterSlots = sparseSceneClusterSlots(primitives);
  for (let index = 0; index < primitives.length; index += 1) {
    const primitive = primitives[index];
    finiteVector(primitive.center, "Primitive center");
    validateIdentity(primitive.materialId, primitive.ownerId);
    const extent = primitiveExtent(primitive);
    const orientation = normalizedQuaternion(primitive.orientation);
    const base = index * (SPARSE_SCENE_PRIMITIVE_STRIDE_BYTES / 4);
    const slot = clusterSlots[index];
    if (slot >= SPARSE_SCENE_CLUSTER_CAPACITY) throw new RangeError("Live scene aggregate arena capacity exceeded");
    floats.set(primitive.center, base);
    words[base + 3] = primitiveType(primitive) | (slot < 0 ? 0 : slot << PRIMITIVE_CLUSTER_SLOT_SHIFT);
    floats.set(extent, base + 4);
    words[base + 7] = packMaterialOwner(primitive.materialId, primitive.ownerId);
    floats.set(orientation, base + 8);
  }
  return words;
}

function inverseRotate(point: SparseSceneVector3, quaternion: SparseSceneQuaternion): SparseSceneVector3 {
  const [qx, qy, qz, qw] = quaternion;
  // Apply conjugate(q) * point * q using the cross-product quaternion form.
  const tx = 2 * (-qy * point[2] + qz * point[1]);
  const ty = 2 * (-qz * point[0] + qx * point[2]);
  const tz = 2 * (-qx * point[1] + qy * point[0]);
  return [
    point[0] + qw * tx + (-qy * tz + qz * ty),
    point[1] + qw * ty + (-qz * tx + qx * tz),
    point[2] + qw * tz + (-qx * ty + qy * tx),
  ];
}

function boxDistance(point: SparseSceneVector3, halfExtents: SparseSceneVector3): number {
  const q = point.map((value, axis) => Math.abs(value) - halfExtents[axis]) as unknown as SparseSceneVector3;
  const outside = Math.hypot(Math.max(q[0], 0), Math.max(q[1], 0), Math.max(q[2], 0));
  return outside + Math.min(Math.max(q[0], q[1], q[2]), 0);
}

function cylinderDistance(point: SparseSceneVector3, radius: number, halfHeight: number): number {
  const radial = Math.hypot(point[0], point[2]) - radius;
  const vertical = Math.abs(point[1]) - halfHeight;
  return Math.hypot(Math.max(radial, 0), Math.max(vertical, 0)) + Math.min(Math.max(radial, vertical), 0);
}

function ellipsoidDistance(point: SparseSceneVector3, radii: SparseSceneVector3): number {
  const k0 = Math.hypot(point[0] / radii[0], point[1] / radii[1], point[2] / radii[2]);
  const k1 = Math.hypot(point[0] / (radii[0] ** 2), point[1] / (radii[1] ** 2), point[2] / (radii[2] ** 2));
  return k1 > 1e-12 ? k0 * (k0 - 1) / k1 : -Math.min(...radii);
}

function capsuleDistance(point: SparseSceneVector3, radius: number, halfLength: number): number {
  const segmentY = Math.max(-halfLength, Math.min(halfLength, point[1]));
  return Math.hypot(point[0], point[1] - segmentY, point[2]) - radius;
}

function torusDistance(point: SparseSceneVector3, majorRadius: number, minorRadius: number): number {
  return Math.hypot(Math.hypot(point[0], point[2]) - majorRadius, point[1]) - minorRadius;
}

function coneDistance(point: SparseSceneVector3, baseRadius: number, halfHeight: number, topRadius: number): number {
  const radial = Math.hypot(point[0], point[2]);
  const capRadius = point[1] < 0 ? baseRadius : topRadius;
  const cap: readonly [number, number] = [radial - Math.min(radial, capRadius), Math.abs(point[1]) - halfHeight];
  const slope: readonly [number, number] = [topRadius - baseRadius, 2 * halfHeight];
  const toApex: readonly [number, number] = [topRadius - radial, halfHeight - point[1]];
  const projection = Math.min(1, Math.max(0, (toApex[0] * slope[0] + toApex[1] * slope[1]) / (slope[0] ** 2 + slope[1] ** 2)));
  const side: readonly [number, number] = [radial - topRadius + slope[0] * projection, point[1] - halfHeight + slope[1] * projection];
  const inside = side[0] < 0 && cap[1] < 0;
  return (inside ? -1 : 1) * Math.sqrt(Math.min(cap[0] ** 2 + cap[1] ** 2, side[0] ** 2 + side[1] ** 2));
}

/** CPU mirror of the WGSL primitive SDF, useful for topology planning and tests. */
export function sparseScenePrimitiveSignedDistance(
  primitive: SparseScenePrimitive,
  worldPoint: SparseSceneVector3,
): number {
  finiteVector(worldPoint, "World point");
  finiteVector(primitive.center, "Primitive center");
  validateIdentity(primitive.materialId, primitive.ownerId);
  const localOffset: SparseSceneVector3 = [
    worldPoint[0] - primitive.center[0],
    worldPoint[1] - primitive.center[1],
    worldPoint[2] - primitive.center[2],
  ];
  const local = inverseRotate(localOffset, normalizedQuaternion(primitive.orientation));
  if (primitive.kind === "box") {
    positiveVector(primitive.halfExtents, "Box half extents");
    return boxDistance(local, primitive.halfExtents);
  }
  if (primitive.kind === "cylinder") {
    primitiveExtent(primitive);
    return cylinderDistance(local, primitive.radius, primitive.halfHeight);
  }
  if (primitive.kind === "capsule") {
    primitiveExtent(primitive);
    return capsuleDistance(local, primitive.radius, primitive.halfLength);
  }
  if (primitive.kind === "torus") {
    primitiveExtent(primitive);
    return torusDistance(local, primitive.majorRadius, primitive.minorRadius);
  }
  if (primitive.kind === "cone") {
    primitiveExtent(primitive);
    return coneDistance(local, primitive.baseRadius, primitive.halfHeight, primitive.topRadius);
  }
  // Three kinds have no cheap conservative equivalent, so they are evaluated
  // through the render ABI's own CPU sample — the same function the exactness
  // contract in `tests/webgpu-svo-primitive-exact.test.ts` holds against the
  // WGSL this pass now calls. This is a mirror of the GPU path, not a second
  // opinion about the shape.
  if (sparseSceneAbiTruePrimitive(primitive)) {
    primitiveExtent(primitive);
    const descriptor = abiDescriptorForPrimitive(primitive);
    return sampleSvoPrimitive(descriptor, { x: worldPoint[0], y: worldPoint[1], z: worldPoint[2] },
      undefined,
      primitive.kind === "smooth-union-cluster" ? () => primitive.packing : undefined).signedDistance_m;
  }
  positiveVector(primitive.radii, "Ellipsoid radii");
  return ellipsoidDistance(local, primitive.radii);
}

/**
 * The smallest solid a primitive draws, or `Infinity` when every feature it has
 * is the shape itself.
 *
 * Only the field family answers finitely: a lattice's finest lobe, a lobe set's
 * blend, a sweep's thinnest section. It is what decides whether a cell needs
 * more than its centre sample.
 */
export function sparseScenePrimitiveFeatureRadius(primitive: SparseScenePrimitive): number {
  if (primitive.kind !== "smooth-union-cluster") return Number.POSITIVE_INFINITY;
  return svoClusterFeatureRadius_m(
    { x: primitive.lobeRadii[0], y: primitive.lobeRadii[1], z: primitive.lobeRadii[2] },
    primitive.packing,
  );
}

/**
 * Offsets, in units of half a cell, at which a thin-featured primitive is
 * sampled in addition to the cell centre.
 *
 * The eight corners of the cell. The centre sample alone is *sound* for
 * occupancy — every kind here is a 1-Lipschitz lower bound, so a cell holding
 * solid has a centre distance no greater than its own radius — but soundness is
 * not the whole contract: the planar coverage law reads that distance as a
 * fraction, and a field whose features are finer than the cell has neither a
 * planar surface nor a unit gradient there, so a canopy of two-centimetre
 * florets in twenty-five-millimetre cells reports a fraction near zero across a
 * region that is substantially solid. Taking the minimum over the corners
 * recovers the feature that is inside the cell but away from its centre, which
 * is exactly the ≤1.5-cell case `lib/svo-scene-coverage.ts:129` flags as
 * "vulnerable to camera-dependent disappearance".
 *
 * Corners rather than faces because a lattice's lobes sit on a jittered grid:
 * face centres share two coordinates with the cell centre and miss a diagonal
 * lobe that the corners bracket.
 */
export const SPARSE_SCENE_CONSERVATIVE_OFFSETS: readonly SparseSceneVector3[] = Object.freeze([
  [-1, -1, -1], [1, -1, -1], [-1, 1, -1], [1, 1, -1],
  [-1, -1, 1], [1, -1, 1], [-1, 1, 1], [1, 1, 1],
] as const);

/** Whether a primitive's features are fine enough for the cell to need corner samples. */
export function sparseSceneNeedsConservativeSampling(
  primitive: SparseScenePrimitive,
  cellSize: SparseSceneVector3,
): boolean {
  return sparseScenePrimitiveFeatureRadius(primitive) < Math.max(cellSize[0], cellSize[1], cellSize[2]);
}

/** Conservative cell-center occupancy mirror used by the GPU voxelization pass. */
export function sampleSparseScenePrimitiveCell(
  primitives: readonly SparseScenePrimitive[],
  cellCenter: SparseSceneVector3,
  cellSize: SparseSceneVector3,
): SparseSceneCellSample {
  positiveVector(cellSize, "Cell size");
  // Two accumulators on purpose. The distance lane stays the field's own value
  // at the cell centre, because the derived builder differentiates it for
  // radiance normals and a minimum over offset samples is not that field's
  // distance anywhere. Coverage is what the extra samples are for.
  let distance = Number.POSITIVE_INFINITY;
  let coverage = Number.POSITIVE_INFINITY;
  let identity = packMaterialOwner(0, SPARSE_BRICK_NO_OWNER);
  for (const primitive of primitives) {
    const centre = sparseScenePrimitiveSignedDistance(primitive, cellCenter);
    let nearest = centre;
    if (sparseSceneNeedsConservativeSampling(primitive, cellSize)) {
      for (const offset of SPARSE_SCENE_CONSERVATIVE_OFFSETS) {
        const point = cellCenter.map((value, axis) => value + 0.5 * offset[axis] * cellSize[axis]) as unknown as SparseSceneVector3;
        nearest = Math.min(nearest, sparseScenePrimitiveSignedDistance(primitive, point));
      }
    }
    distance = Math.min(distance, centre);
    if (nearest < coverage) {
      coverage = nearest;
      identity = packMaterialOwner(primitive.materialId, primitive.ownerId);
    }
  }
  if (primitives.length === 0) return { solidSignedDistance: distance, solidFraction: 0, materialOwner: identity };
  const cellRadius = 0.5 * Math.hypot(...cellSize);
  const fraction = Math.max(0, Math.min(1, 0.5 - coverage / (2 * cellRadius)));
  return {
    solidSignedDistance: distance,
    solidFraction: fraction,
    materialOwner: fraction > 0 ? identity : packMaterialOwner(0, SPARSE_BRICK_NO_OWNER),
  };
}

/**
 * Live scene maintenance. The scene primitive arena is authoritative; sparse
 * payloads are a revisioned acceleration cache rebuilt only for affected
 * bricks. All variable-length records share one fixed arena to stay below the
 * portable storage-binding limit.
 */
export const sparseSceneProxyVoxelizationShader = /* wgsl */ `
${svoPrimitiveWGSL}
struct ScenePrimitive {
  centerType: vec4f,
  extentIdentity: vec4f,
  rotation: vec4f,
}
struct Params {
  worldOrigin: vec4f,
  cell: vec4f,
  publication: vec4u,
  capacities: vec4u,
  offsets: vec4u,
  lanes: vec4u,
}
@group(0) @binding(0) var<storage, read_write> structure: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read_write> payload: array<u32>;
@group(0) @binding(2) var<storage, read> primitives: array<ScenePrimitive>;
@group(0) @binding(3) var<storage, read_write> maintenance: array<atomic<u32>>;
@group(0) @binding(4) var<uniform> params: Params;

const BRICK_ACTIVE:u32=${SVO_BRICK_LIFECYCLE.activeBit}u;
const BRICK_DIRTY:u32=${SVO_BRICK_LIFECYCLE.dirtyBit}u;
const BRICK_QUEUED:u32=${SVO_BRICK_LIFECYCLE.queuedBit}u;
const BRICK_RELOCATING:u32=${SVO_BRICK_LIFECYCLE.relocatingBit}u;
const OCCUPANCY_READY:u32=${SVO_BRICK_OCCUPANCY.readyBit}u;
const OCCUPANCY_OCCUPIED:u32=${SVO_BRICK_OCCUPANCY.occupiedBit}u;
const DIRTY_BRICK_OVERFLOW:u32=${SPARSE_SCENE_MAINTENANCE_OVERFLOW.dirtyBricks}u;
const CANDIDATE_OVERFLOW:u32=${SPARSE_SCENE_MAINTENANCE_OVERFLOW.candidates}u;
const INCOMPLETE_OVERFLOW:u32=${SPARSE_SCENE_MAINTENANCE_INCOMPLETE_OVERFLOW}u;
const TOPOLOGY_INCOMPLETE:u32=${SPARSE_SCENE_MAINTENANCE_OVERFLOW.topology}u;
const NO_MATERIAL_OWNER:u32=0xffff0000u;
const INVALID_INDEX:u32=${SPARSE_BRICK_INVALID_INDEX}u;
const INDEXED_BINNING:u32=${SPARSE_SCENE_MAINTENANCE_FLAGS.indexedBinning}u;
const TERRAIN_FIELD:u32=${SPARSE_SCENE_MAINTENANCE_FLAGS.terrainField}u;
const CHUNK_BEGIN:u32=${SPARSE_SCENE_MAINTENANCE_STATE_WORDS.chunkBegin}u;
const CHUNK_END:u32=${SPARSE_SCENE_MAINTENANCE_STATE_WORDS.chunkEnd}u;
const CHUNK_CURSOR:u32=${SPARSE_SCENE_MAINTENANCE_STATE_WORDS.chunkCursor}u;
const MAXIMUM_BRICK_CANDIDATES:u32=${SPARSE_SCENE_MAINTENANCE_STATE_WORDS.maximumBrickCandidates}u;

fn primitiveCount()->u32{return params.publication.x;}
fn dirtyRegionCount()->u32{return params.publication.y;}
fn finestLevel()->u32{return params.publication.z;}
fn sceneRevision()->u32{return params.publication.w;}
fn dirtyBrickCapacity()->u32{return params.capacities.x;}
fn candidatesPerBrick()->u32{return params.capacities.y;}
fn primitiveBoundsOffset()->u32{return params.capacities.z;}
fn dirtyRegionOffset()->u32{return params.capacities.w;}
fn dirtyBrickOffset()->u32{return params.offsets.x;}
fn candidateOffset()->u32{return params.offsets.y;}
fn stateOffset()->u32{return params.offsets.z;}
fn sceneGeometryOffset()->u32{return params.lanes.x;}
fn sceneMaterialOffset()->u32{return params.lanes.y;}
fn topologyOffset()->u32{return params.lanes.z;}
fn controlLoad(word:u32)->u32{return atomicLoad(&structure[word]);}
fn topologyLoad(word:u32)->u32{return atomicLoad(&structure[topologyOffset()+word]);}
fn topologyStore(word:u32,value:u32){atomicStore(&structure[topologyOffset()+word],value);}
fn topologyAnd(word:u32,value:u32)->u32{return atomicAnd(&structure[topologyOffset()+word],value);}
fn topologyOr(word:u32,value:u32)->u32{return atomicOr(&structure[topologyOffset()+word],value);}
fn loadArenaF32(word:u32)->f32{return bitcast<f32>(atomicLoad(&maintenance[word]));}
fn loadArenaWord(word:u32)->u32{return atomicLoad(&maintenance[word]);}
// The aggregate arena rides in the two lanes the params struct already left
// spare, rather than in a seventh vec4: this pass holds itself to four storage
// bindings and a fixed 96-byte uniform, and neither ceiling is worth breaking
// for two integers.
fn clusterArenaOffset()->u32{return params.offsets.w;}
fn clusterArenaCapacity()->u32{return params.lanes.w;}
// The last two spare uniform lanes. Everything the record index needs beyond
// these is derived from offsets the block already carries, or read from the
// index header block below, so the index costs the uniform nothing.
fn brickBudget()->u32{return bitcast<u32>(params.worldOrigin.w);}
fn maintenanceFlags()->u32{return bitcast<u32>(params.cell.w);}
/**
 * Layout of everything after the aggregate arena, derived rather than passed.
 *
 * The arena is a fixed sequence of fixed-capacity blocks, and every capacity in
 * it is already visible here: the dirty-region *bounds* arena spans
 * \`dirtyBrickOffset()-dirtyRegionOffset()\` words, and the region *cell* arena
 * holds one equally-sized record per region, so the two are the same length by
 * construction. A capacity that had to be transmitted could disagree with the
 * allocation; one that is subtracted out of the allocation cannot.
 */
fn indexHeaderOffset()->u32{
  return clusterArenaOffset()+clusterArenaCapacity()*${SVO_CLUSTER_ARENA_BLOCK.strideWords}u;
}
fn indexWord(word:u32)->u32{return loadArenaWord(indexHeaderOffset()+word);}
fn indexFloat(word:u32)->f32{return bitcast<f32>(indexWord(word));}
fn regionCellOffset()->u32{return indexHeaderOffset()+${SPARSE_SCENE_INDEX_HEADER.wordCount}u;}
fn leafClaimOffset()->u32{return regionCellOffset()+(dirtyBrickOffset()-dirtyRegionOffset());}
fn gridOffsetsOffset()->u32{return leafClaimOffset()+dirtyBrickCapacity();}
fn gridDimensions()->vec3u{
  return vec3u(indexWord(${SPARSE_SCENE_INDEX_HEADER.gridDimensionX}u),
    indexWord(${SPARSE_SCENE_INDEX_HEADER.gridDimensionY}u),
    indexWord(${SPARSE_SCENE_INDEX_HEADER.gridDimensionZ}u));
}
fn gridExtent()->vec3f{
  return vec3f(indexFloat(${SPARSE_SCENE_INDEX_HEADER.gridExtentX}u),
    indexFloat(${SPARSE_SCENE_INDEX_HEADER.gridExtentY}u),
    indexFloat(${SPARSE_SCENE_INDEX_HEADER.gridExtentZ}u));
}
fn gridCellCount()->u32{let d=gridDimensions();return d.x*d.y*d.z;}
fn gridItemsOffset()->u32{return gridOffsetsOffset()+gridCellCount()+1u;}
fn globalRecordsOffset()->u32{
  return gridItemsOffset()+indexWord(${SPARSE_SCENE_INDEX_HEADER.gridItemCapacity}u);
}
/**
 * The ground, as a column field rather than as a shape.
 *
 * There is no terrain primitive, no tenth type number and no terrain record:
 * the lattice is the octree's own finest cell lattice over its own domain, so a
 * voxel's column is its world cell coordinate and nothing about origin, spacing
 * or orientation has to be transmitted or agreed. Zero columns is the encoding
 * for "no ground", which is why there is no separate enable bit.
 */
fn terrainFieldDimensions()->vec2u{
  return vec2u(indexWord(${SPARSE_SCENE_INDEX_HEADER.terrainFieldDimensionX}u),
    indexWord(${SPARSE_SCENE_INDEX_HEADER.terrainFieldDimensionZ}u));
}
fn terrainFieldPresent()->bool{return (maintenanceFlags()&TERRAIN_FIELD)!=0u;}
fn terrainColumnHeight(dimensions:vec2u,column:vec2u)->f32{
  let clamped=min(column,dimensions-vec2u(1u));
  return bitcast<f32>(loadArenaWord(indexWord(${SPARSE_SCENE_INDEX_HEADER.terrainFieldOffset}u)
    +clamped.y*dimensions.x+clamped.x));
}
fn primitiveCapacity()->u32{return (dirtyRegionOffset()-primitiveBoundsOffset())/8u;}
fn recordCellRangeOffset()->u32{return globalRecordsOffset()+primitiveCapacity();}
fn unpackGridCoordinate(word:u32)->vec3u{
  let mask=${(1 << SPARSE_SCENE_GRID_COORDINATE_BITS) - 1}u;
  return vec3u(word&mask,(word>>${SPARSE_SCENE_GRID_COORDINATE_BITS}u)&mask,
    (word>>${2 * SPARSE_SCENE_GRID_COORDINATE_BITS}u)&mask);
}
/** The grid cell a record was scattered into first, exactly as the CPU placed it. */
fn recordCellMinimum(primitiveIndex:u32)->vec3u{
  return unpackGridCoordinate(loadArenaWord(recordCellRangeOffset()+primitiveIndex*2u));
}
fn gridCoordinate(point:vec3f)->vec3u{
  let dimensions=gridDimensions();
  let raw=floor((point-params.worldOrigin.xyz)/gridExtent());
  return vec3u(clamp(raw,vec3f(0.0),vec3f(dimensions)-vec3f(1.0)));
}
fn gridCellIndex(coordinate:vec3u)->u32{
  let dimensions=gridDimensions();
  return coordinate.x+dimensions.x*(coordinate.y+dimensions.y*coordinate.z);
}
fn regionCellWord(region:u32,word:u32)->u32{
  return loadArenaWord(regionCellOffset()+region*${SPARSE_SCENE_REGION_CELL_WORDS}u+word);
}
/** The region owning a linear cell index, by binary search over the packed prefixes. */
fn regionForCell(cellIndex:u32)->u32{
  var low=0u;
  var high=dirtyRegionCount()-1u;
  loop{
    if(low>=high){break;}
    let mid=(low+high+1u)/2u;
    if(regionCellWord(mid,6u)<=cellIndex){low=mid;}else{high=mid-1u;}
  }
  return low;
}
fn popcountBelow(mask:u32,octant:u32)->u32{return countOneBits(mask&((1u<<octant)-1u));}
/**
 * The leaf owning one finest brick coordinate, or \`INVALID_INDEX\`.
 *
 * The same descent \`mutateTopology\` uses to insert one, read-only: child mask
 * in node word 3, first child in word 4, terminal leaf in word 6. It stops at
 * the first level whose octant is absent, so a coordinate in empty space costs
 * one or two loads rather than the full depth — which is what keeps a
 * whole-domain dirty region affordable on a sparse tree.
 */
fn leafAtBrickCoordinate(coordinate:vec3u)->u32{
  let maximumDepth=finestLevel();
  if(maximumDepth>21u||controlLoad(0u)==0u){return INVALID_INDEX;}
  var node=0u;
  for(var level=1u;level<=maximumDepth;level+=1u){
    let mask=topologyLoad(node*8u+3u)&0xffu;
    let bit=maximumDepth-level;
    let octant=((coordinate.x>>bit)&1u)|(((coordinate.y>>bit)&1u)<<1u)|(((coordinate.z>>bit)&1u)<<2u);
    if((mask&(1u<<octant))==0u){return topologyLoad(node*8u+6u);}
    node=topologyLoad(node*8u+4u)+popcountBelow(mask,octant);
  }
  return topologyLoad(node*8u+6u);
}
fn dirtyBrickTotal()->u32{return min(atomicLoad(&maintenance[stateOffset()]),dirtyBrickCapacity());}
fn chunkBegin()->u32{return atomicLoad(&maintenance[stateOffset()+CHUNK_BEGIN]);}
fn chunkEnd()->u32{return atomicLoad(&maintenance[stateOffset()+CHUNK_END]);}
${svoClusterArenaDecodeWGSL({
  functionName: "sceneClusterPacking",
  loadWord: "loadArenaWord",
  baseWordExpression: "clusterArenaOffset()",
  capacityExpression: "clusterArenaCapacity()",
})}

fn keyBit(low: u32, high: u32, bit: u32) -> u32 {
  if (bit >= 32u) { return (high >> (bit - 32u)) & 1u; }
  return (low >> bit) & 1u;
}
fn decodeMorton(low: u32, high: u32, level: u32) -> vec3u {
  var result = vec3u(0u);
  for (var bit = 0u; bit < level; bit += 1u) {
    let scale = 1u << bit;
    result.x += keyBit(low, high, 3u * bit) * scale;
    result.y += keyBit(low, high, 3u * bit + 1u) * scale;
    result.z += keyBit(low, high, 3u * bit + 2u) * scale;
  }
  return result;
}
fn leafBounds(leafIndex:u32)->mat2x3f{
  let leafBase=controlLoad(16u)+leafIndex*4u;
  let nodeIndex=topologyLoad(leafBase);
  let level=topologyLoad(nodeIndex*8u+2u);
  let brick=decodeMorton(topologyLoad(leafBase+2u),topologyLoad(leafBase+3u),level);
  var scale=1u;
  if(finestLevel()!=0xffffffffu&&finestLevel()>level){scale=1u<<(finestLevel()-level);}
  let brickSize=controlLoad(11u);
  let minimum=params.worldOrigin.xyz+vec3f(brick*brickSize*scale)*params.cell.xyz;
  let maximum=minimum+vec3f(f32(brickSize*scale))*params.cell.xyz;
  return mat2x3f(minimum,maximum);
}
fn boundsOverlap(a:mat2x3f,b:mat2x3f)->bool{
  return all(a[0]<=b[1])&&all(b[0]<=a[1]);
}
fn arenaBounds(word:u32)->mat2x3f{
  return mat2x3f(
    vec3f(loadArenaF32(word),loadArenaF32(word+1u),loadArenaF32(word+2u)),
    vec3f(loadArenaF32(word+4u),loadArenaF32(word+5u),loadArenaF32(word+6u)));
}
fn inverseRotate(point: vec3f, quaternion: vec4f) -> vec3f {
  let vector = -quaternion.xyz;
  let twiceCross = 2.0 * cross(vector, point);
  return point + quaternion.w * twiceCross + cross(vector, twiceCross);
}
fn boxDistance(point: vec3f, halfExtents: vec3f) -> f32 {
  let q = abs(point) - halfExtents;
  return length(max(q, vec3f(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}
fn cylinderDistance(point: vec3f, radius: f32, halfHeight: f32) -> f32 {
  let q = vec2f(length(point.xz) - radius, abs(point.y) - halfHeight);
  return length(max(q, vec2f(0.0))) + min(max(q.x, q.y), 0.0);
}
fn capsuleDistance(point: vec3f, radius: f32, halfLength: f32) -> f32 {
  let segmentY = clamp(point.y, -halfLength, halfLength);
  return length(vec3f(point.x, point.y - segmentY, point.z)) - radius;
}
fn torusDistance(point: vec3f, majorRadius: f32, minorRadius: f32) -> f32 {
  return length(vec2f(length(point.xz) - majorRadius, point.y)) - minorRadius;
}
fn coneDistance(point: vec3f, baseRadius: f32, halfHeight: f32, topRadius: f32) -> f32 {
  let radial = length(point.xz);
  let capRadius = select(topRadius, baseRadius, point.y < 0.0);
  let cap = vec2f(radial - min(radial, capRadius), abs(point.y) - halfHeight);
  let slope = vec2f(topRadius - baseRadius, 2.0 * halfHeight);
  let toApex = vec2f(topRadius - radial, halfHeight - point.y);
  let projection = clamp(dot(toApex, slope) / dot(slope, slope), 0.0, 1.0);
  let side = vec2f(radial - topRadius, point.y - halfHeight) + slope * projection;
  let inside = side.x < 0.0 && cap.y < 0.0;
  return select(1.0, -1.0, inside) * sqrt(min(dot(cap, cap), dot(side, side)));
}
fn ellipsoidDistance(point: vec3f, radii: vec3f) -> f32 {
  let k0 = length(point / radii);
  let k1 = length(point / (radii * radii));
  return select(-min(radii.x, min(radii.y, radii.z)), k0 * (k0 - 1.0) / k1, k1 > 1e-8);
}
fn scenePrimitiveType(primitive: ScenePrimitive) -> u32 {
  return bitcast<u32>(primitive.centerType.w) & ${PRIMITIVE_TYPE_MASK}u;
}
fn scenePrimitiveClusterSlot(primitive: ScenePrimitive) -> u32 {
  return bitcast<u32>(primitive.centerType.w) >> ${PRIMITIVE_CLUSTER_SLOT_SHIFT}u;
}
/**
 * The three kinds with no cheap conservative equivalent, as the render ABI's
 * own record. Built here rather than stored, because the voxelizer's record is
 * a third the size and the two lanes that differ — the identity word and the
 * arena reference — are exactly the two this pass owns differently.
 */
fn scenePrimitiveAbiRecord(primitive: ScenePrimitive, kind: u32) -> SvoPrimitiveRecord {
  return SvoPrimitiveRecord(
    vec4u(bitcast<vec3u>(primitive.centerType.xyz), kind),
    vec4u(bitcast<vec3u>(primitive.extentIdentity.xyz), bitcast<u32>(primitive.extentIdentity.w)),
    primitive.rotation,
    vec4u(0u, 0u, 0u, 0u));
}
fn primitiveDistance(primitive: ScenePrimitive, world: vec3f) -> f32 {
  let primitiveType = scenePrimitiveType(primitive);
  let offset = world - primitive.centerType.xyz;
  let local = inverseRotate(offset, primitive.rotation);
  if (primitiveType == 1u) { return boxDistance(local, primitive.extentIdentity.xyz); }
  if (primitiveType == 2u) { return cylinderDistance(local, primitive.extentIdentity.x, primitive.extentIdentity.y); }
  if (primitiveType == 3u) { return ellipsoidDistance(local, primitive.extentIdentity.xyz); }
  if (primitiveType == 4u) { return capsuleDistance(local, primitive.extentIdentity.x, primitive.extentIdentity.y); }
  if (primitiveType == 5u) { return torusDistance(local, primitive.extentIdentity.x, primitive.extentIdentity.y); }
  if (primitiveType == 6u) { return coneDistance(local, primitive.extentIdentity.x, primitive.extentIdentity.y, primitive.extentIdentity.z); }
  // Shapes 7-9 are the render ABI's, evaluated by the render ABI. Terrain height
  // is unreachable here: a heightfield is not a finite live primitive and the
  // adapters refuse it before it can be packed.
  if (primitiveType == ${SPARSE_SCENE_PRIMITIVE_TYPES["smooth-union-cluster"]}u) {
    return svoPrimitiveDistance_m(scenePrimitiveAbiRecord(primitive, SVO_KIND_SMOOTH_UNION_CLUSTER), world, 0.0,
      sceneClusterPacking(scenePrimitiveClusterSlot(primitive)));
  }
  if (primitiveType == ${SPARSE_SCENE_PRIMITIVE_TYPES["round-cone"]}u) {
    return svoPrimitiveDistance_m(scenePrimitiveAbiRecord(primitive, SVO_KIND_ROUND_CONE), world, 0.0,
      svoInvalidClusterPacking());
  }
  if (primitiveType == ${SPARSE_SCENE_PRIMITIVE_TYPES["rounded-cylinder"]}u) {
    return svoPrimitiveDistance_m(scenePrimitiveAbiRecord(primitive, SVO_KIND_ROUNDED_CYLINDER), world, 0.0,
      svoInvalidClusterPacking());
  }
  return 1e20;
}
/**
 * The smallest solid this primitive draws, in metres, or a large number when
 * its only feature is the shape itself. Mirrors
 * \`sparseScenePrimitiveFeatureRadius\`.
 */
fn primitiveFeatureRadius(primitive: ScenePrimitive) -> f32 {
  if (scenePrimitiveType(primitive) != ${SPARSE_SCENE_PRIMITIVE_TYPES["smooth-union-cluster"]}u) { return 1e20; }
  return svoClusterFeatureRadius_m(primitive.extentIdentity.xyz,
    sceneClusterPacking(scenePrimitiveClusterSlot(primitive)));
}
/**
 * Nearest solid this primitive puts anywhere in the cell, not just at its
 * centre. See \`SPARSE_SCENE_CONSERVATIVE_OFFSETS\` for why a field finer than
 * the cell needs its corners sampled and a plain convex shape does not.
 */
fn primitiveCellCoverageDistance(primitive: ScenePrimitive, world: vec3f, cell: vec3f, centre: f32) -> f32 {
  if (primitiveFeatureRadius(primitive) >= max(cell.x, max(cell.y, cell.z))) { return centre; }
  var nearest = centre;
  for (var corner = 0u; corner < 8u; corner += 1u) {
    let sign = vec3f(f32(corner & 1u), f32((corner >> 1u) & 1u), f32((corner >> 2u) & 1u)) * 2.0 - vec3f(1.0);
    nearest = min(nearest, primitiveDistance(primitive, world + 0.5 * sign * cell));
  }
  return nearest;
}
fn linearIndex64(gid:vec3u,groups:vec3u)->u32{
  return gid.x+gid.y*groups.x*64u+gid.z*groups.x*groups.y*64u;
}
fn linearIndex256(gid:vec3u,groups:vec3u)->u32{
  return gid.x+gid.y*groups.x*256u+gid.z*groups.x*groups.y*256u;
}
fn writeDispatch(offset:u32,workItems:u32,workgroupSize:u32){
  let blocks=(workItems+workgroupSize-1u)/workgroupSize;
  let x=min(blocks,65535u);
  var y=1u;
  if(x>0u){y=(blocks+x-1u)/x;}
  atomicStore(&maintenance[offset],x);
  atomicStore(&maintenance[offset+1u],y);
  atomicStore(&maintenance[offset+2u],1u);
}

/** The header every invalidation entry point publishes before it does anything else. */
fn storeRequestedRevision(){
  atomicStore(&maintenance[stateOffset()+2u],sceneRevision());
  atomicStore(&maintenance[stateOffset()+4u],primitiveCount());
  atomicStore(&maintenance[stateOffset()+5u],dirtyRegionCount());
}

/** Queue one leaf for repair. Shared so the two invalidation paths cannot diverge. */
fn markDirtyBrick(leafIndex:u32){
  let leafBase=controlLoad(16u)+leafIndex*4u;
  let nodeIndex=topologyLoad(leafBase);
  topologyAnd(nodeIndex*8u+7u,~OCCUPANCY_READY);
  topologyOr(nodeIndex*8u+7u,BRICK_DIRTY|BRICK_QUEUED);
  let dirtyIndex=atomicAdd(&maintenance[stateOffset()],1u);
  if(dirtyIndex>=dirtyBrickCapacity()){
    atomicOr(&maintenance[stateOffset()+1u],DIRTY_BRICK_OVERFLOW);
    return;
  }
  let record=dirtyBrickOffset()+dirtyIndex*4u;
  atomicStore(&maintenance[record],leafIndex);
  atomicStore(&maintenance[record+1u],0u);
  atomicStore(&maintenance[record+2u],0u);
  atomicStore(&maintenance[record+3u],0u);
}

/**
 * The unindexed invalidation: one thread per leaf, tested against every region.
 *
 * Retained beside the indexed one because it is the only sound answer when the
 * dirty regions cover more of the lattice than the tree has leaves, and because
 * a claim about complexity that cannot be measured against the thing it
 * replaced is not a measurement.
 */
@compute @workgroup_size(64)
fn invalidateDirtyBricks(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) groups:vec3u){
  let leafIndex=linearIndex64(gid,groups);
  if(leafIndex>=controlLoad(1u)){return;}
  if(leafIndex==0u){storeRequestedRevision();}
  let brickBounds=leafBounds(leafIndex);
  var affected=false;
  for(var regionIndex=0u;regionIndex<dirtyRegionCount();regionIndex+=1u){
    if(boundsOverlap(brickBounds,arenaBounds(dirtyRegionOffset()+regionIndex*8u))){affected=true;break;}
  }
  if(!affected){return;}
  markDirtyBrick(leafIndex);
}

/**
 * The indexed invalidation: one thread per dirty *brick cell*, descending to the
 * leaf that owns it.
 *
 * Cost tracks how much of the world changed, not how large the tree is nor how
 * many records moved. The claim word is what makes it a set: many finest cells
 * resolve to one coarse leaf, and overlapping regions name the same brick, so
 * the first thread to stamp the current revision onto a leaf owns it and the
 * rest fall away. Revisions are strictly increasing and nonzero, so a stamp from
 * an earlier one never reads as current.
 */
@compute @workgroup_size(64)
fn invalidateDirtyBricksFromRegions(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) groups:vec3u){
  let cellIndex=linearIndex64(gid,groups);
  if(cellIndex==0u){storeRequestedRevision();}
  if(dirtyRegionCount()==0u||cellIndex>=indexWord(${SPARSE_SCENE_INDEX_HEADER.invalidationCellCount}u)){return;}
  let region=regionForCell(cellIndex);
  let span=vec3u(regionCellWord(region,3u),regionCellWord(region,4u),regionCellWord(region,5u));
  if(span.x==0u||span.y==0u||span.z==0u){return;}
  let local=cellIndex-regionCellWord(region,6u);
  let coordinate=vec3u(regionCellWord(region,0u),regionCellWord(region,1u),regionCellWord(region,2u))
    +vec3u(local%span.x,(local/span.x)%span.y,local/(span.x*span.y));
  let leafIndex=leafAtBrickCoordinate(coordinate);
  if(leafIndex==INVALID_INDEX||leafIndex>=controlLoad(1u)){return;}
  if(atomicExchange(&maintenance[leafClaimOffset()+leafIndex],sceneRevision())==sceneRevision()){return;}
  markDirtyBrick(leafIndex);
}

/**
 * Bound this frame's chunk of the dirty list, and size the three indirect
 * dispatches to it.
 *
 * The cursor is the whole of the per-frame budget: it lives in the state block,
 * which a *new* revision clears and a continuation frame does not, so a
 * publication walks its own dirty list across as many frames as the budget
 * needs. Completion is stored by the chunk that reaches the end and by no
 * other — a half-repaired publication is never observable as current.
 */
@compute @workgroup_size(1)
fn prepareMaintenanceDispatch(){
  let dirtyCount=dirtyBrickTotal();
  let cursor=min(atomicLoad(&maintenance[stateOffset()+CHUNK_CURSOR]),dirtyCount);
  var end=dirtyCount;
  if(brickBudget()!=0u){end=min(dirtyCount,cursor+brickBudget());}
  atomicStore(&maintenance[stateOffset()+CHUNK_BEGIN],cursor);
  atomicStore(&maintenance[stateOffset()+CHUNK_END],end);
  atomicStore(&maintenance[stateOffset()+CHUNK_CURSOR],end);
  let chunk=end-cursor;
  if((maintenanceFlags()&INDEXED_BINNING)!=0u){
    // One workgroup per dirty brick: the brick's candidates are whatever sits in
    // its one grid cell plus the global list, and neither length is known here.
    writeDispatch(stateOffset()+${SPARSE_SCENE_MAINTENANCE_STATE_WORDS.binDispatch}u,chunk,1u);
  }else{
    writeDispatch(stateOffset()+${SPARSE_SCENE_MAINTENANCE_STATE_WORDS.binDispatch}u,chunk*primitiveCount(),256u);
  }
  let brickSize=controlLoad(11u);
  writeDispatch(stateOffset()+${SPARSE_SCENE_MAINTENANCE_STATE_WORDS.rebuildDispatch}u,chunk*brickSize*brickSize*brickSize,256u);
  writeDispatch(stateOffset()+${SPARSE_SCENE_MAINTENANCE_STATE_WORDS.finalizeDispatch}u,chunk,64u);
  // Only invalidation has run at this point, so the mask can only be carrying
  // DIRTY_BRICK_OVERFLOW here; it is spelled the same way as the two later
  // stores so the completion rule has exactly one form in this shader.
  if(chunk==0u&&end>=dirtyCount&&(atomicLoad(&maintenance[stateOffset()+1u])&INCOMPLETE_OVERFLOW)==0u){
    atomicStore(&maintenance[stateOffset()+3u],sceneRevision());
  }
}

@compute @workgroup_size(256)
fn binDirtyBrickCandidates(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) groups:vec3u){
  if(primitiveCount()==0u){return;}
  let index=linearIndex256(gid,groups);
  let dirtyIndex=chunkBegin()+index/primitiveCount();
  let primitiveIndex=index-(index/primitiveCount())*primitiveCount();
  if(dirtyIndex>=chunkEnd()){return;}
  let record=dirtyBrickOffset()+dirtyIndex*4u;
  let leafIndex=atomicLoad(&maintenance[record]);
  if(!boundsOverlap(leafBounds(leafIndex),arenaBounds(primitiveBoundsOffset()+primitiveIndex*8u))){return;}
  let localSlot=atomicAdd(&maintenance[record+1u],1u);
  atomicMax(&maintenance[stateOffset()+MAXIMUM_BRICK_CANDIDATES],localSlot+1u);
  if(localSlot<candidatesPerBrick()){
    atomicStore(&maintenance[candidateOffset()+dirtyIndex*candidatesPerBrick()+localSlot],primitiveIndex);
  }else{
    atomicStore(&maintenance[record+2u],1u);
    atomicOr(&maintenance[stateOffset()+1u],CANDIDATE_OVERFLOW);
  }
}

fn binCandidate(record:u32,dirtyIndex:u32,primitiveIndex:u32,brick:mat2x3f){
  if(!boundsOverlap(brick,arenaBounds(primitiveBoundsOffset()+primitiveIndex*8u))){return;}
  let localSlot=atomicAdd(&maintenance[record+1u],1u);
  atomicMax(&maintenance[stateOffset()+MAXIMUM_BRICK_CANDIDATES],localSlot+1u);
  if(localSlot<candidatesPerBrick()){
    atomicStore(&maintenance[candidateOffset()+dirtyIndex*candidatesPerBrick()+localSlot],primitiveIndex);
  }else{
    atomicStore(&maintenance[record+2u],1u);
    atomicOr(&maintenance[stateOffset()+1u],CANDIDATE_OVERFLOW);
  }
}

/**
 * The indexed binning: one workgroup per dirty brick, reading only the grid
 * cells that brick occupies.
 *
 * A finest leaf occupies one cell and a coarse one occupies a box of them, so
 * the same record can be found several times. Exactly one occurrence is
 * accepted: the one in the componentwise first cell the record and the brick
 * share, which is max(recordMinimum, brickMinimum) and is always inside their
 * intersection. The record's minimum comes from the CPU's own scatter rather
 * than from re-deriving it here, because that is the one place an f32 floor and
 * an f64 floor of a bound sitting on a lattice face could disagree — and the
 * disagreement would drop a record out of a brick's candidate list silently.
 *
 * The brick's own range is read a quarter of a finest brick inside its faces,
 * which is not a tolerance so much as a statement: a leaf is at least one brick
 * across and a cell is at least one brick across, so a quarter brick is
 * unambiguously interior on both counts.
 */
@compute @workgroup_size(64)
fn binDirtyBrickCandidatesIndexed(
  @builtin(workgroup_id) workgroup:vec3u,
  @builtin(num_workgroups) groups:vec3u,
  @builtin(local_invocation_index) lane:u32,
){
  if(primitiveCount()==0u){return;}
  let slot=workgroup.x+workgroup.y*groups.x+workgroup.z*groups.x*groups.y;
  let dirtyIndex=chunkBegin()+slot;
  if(dirtyIndex>=chunkEnd()){return;}
  let record=dirtyBrickOffset()+dirtyIndex*4u;
  let leafIndex=atomicLoad(&maintenance[record]);
  let brick=leafBounds(leafIndex);
  let globalCount=indexWord(${SPARSE_SCENE_INDEX_HEADER.globalRecordCount}u);
  for(var slotIndex=lane;slotIndex<globalCount;slotIndex+=64u){
    binCandidate(record,dirtyIndex,loadArenaWord(globalRecordsOffset()+slotIndex),brick);
  }
  let inset=0.25*params.cell.xyz*f32(controlLoad(11u));
  let low=gridCoordinate(brick[0]+inset);
  let high=max(gridCoordinate(brick[1]-inset),low);
  for(var z=low.z;z<=high.z;z+=1u){
    for(var y=low.y;y<=high.y;y+=1u){
      for(var x=low.x;x<=high.x;x+=1u){
        let coordinate=vec3u(x,y,z);
        let cell=gridCellIndex(coordinate);
        let first=loadArenaWord(gridOffsetsOffset()+cell);
        let last=loadArenaWord(gridOffsetsOffset()+cell+1u);
        for(var item=first+lane;item<last;item+=64u){
          let primitiveIndex=loadArenaWord(gridItemsOffset()+item);
          if(any(max(recordCellMinimum(primitiveIndex),low)!=coordinate)){continue;}
          binCandidate(record,dirtyIndex,primitiveIndex,brick);
        }
      }
    }
  }
}

@compute @workgroup_size(256)
fn rebuildDirtyBrickPayload(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) groups:vec3u){
  let index=linearIndex256(gid,groups);
  let brickSize = controlLoad(11u);
  let voxelsPerBrick = brickSize * brickSize * brickSize;
  let chunkSlot=index/voxelsPerBrick;
  let dirtyIndex=chunkBegin()+chunkSlot;
  if(dirtyIndex>=chunkEnd()){return;}
  let localIndex=index-chunkSlot*voxelsPerBrick;
  let record=dirtyBrickOffset()+dirtyIndex*4u;
  let leafIndex=atomicLoad(&maintenance[record]);
  let local = vec3u(localIndex % brickSize, (localIndex / brickSize) % brickSize, localIndex / (brickSize * brickSize));
  let leafBase = controlLoad(16u) + leafIndex * 4u;
  let nodeIndex = topologyLoad(leafBase);
  let voxelOffset = topologyLoad(leafBase + 1u);
  let level = topologyLoad(nodeIndex * 8u + 2u);
  let brick = decodeMorton(topologyLoad(leafBase + 2u),topologyLoad(leafBase + 3u),level);
  var scale = 1u;
  if (finestLevel() != 0xffffffffu && finestLevel() > level) { scale = 1u << (finestLevel() - level); }
  let worldCell = (brick * brickSize + local) * scale;
  let world = params.worldOrigin.xyz + (vec3f(worldCell) + 0.5 * f32(scale)) * params.cell.xyz;

  // Two accumulators, exactly as in \`sampleSparseScenePrimitiveCell\`: the
  // distance lane keeps the field's own value at the voxel centre because the
  // derived builder differentiates it for radiance normals, while coverage may
  // read a nearer sample elsewhere in the cell.
  let cellExtent = params.cell.xyz * f32(scale);
  let cellRadius = 0.5 * length(cellExtent);
  var bestDistance = 1e20;
  var bestCoverage = 1e20;
  var bestIdentity = NO_MATERIAL_OWNER;
  // The ground, folded in before the candidates and on exactly the same terms.
  //
  // It is not a candidate and never occupies a slot: a heightfield covers the
  // whole footprint, so binning it per brick would put one copy of it in every
  // brick of the domain and buy nothing — every brick that has ground has *the*
  // ground. Coverage is the exact column fraction rather than a distance read
  // through the planar law, expressed as the pseudo-distance that law inverts
  // to it: \`0.5 - coverage/(2*cellRadius)\` is the fraction, so handing it
  // \`cellRadius*(1 - 2f)\` yields f. That keeps one reduction for every source
  // of coverage in this pass instead of a second, terrain-shaped one, and the
  // two agree at both ends (+cellRadius is empty, -cellRadius is full).
  //
  // The distance lane keeps \`y - h\`, which is the field the derived builder
  // differentiates for radiance normals; the fraction takes the tallest column
  // under the voxel's own footprint so a ridge crossing a coarse voxel is not
  // lost to a centre sample.
  if(terrainFieldPresent()){
    let dimensions=terrainFieldDimensions();
    let first=worldCell.xz;
    let last=first+vec2u(scale-1u);
    let centre=first+vec2u(scale>>1u);
    bestDistance=world.y-terrainColumnHeight(dimensions,centre);
    var tallest=terrainColumnHeight(dimensions,centre);
    tallest=max(tallest,terrainColumnHeight(dimensions,first));
    tallest=max(tallest,terrainColumnHeight(dimensions,vec2u(last.x,first.y)));
    tallest=max(tallest,terrainColumnHeight(dimensions,vec2u(first.x,last.y)));
    tallest=max(tallest,terrainColumnHeight(dimensions,last));
    // Mirrors terrainCellSolidFraction in lib/terrain.ts.
    let fraction=clamp((tallest-(world.y-0.5*cellExtent.y))/cellExtent.y,0.0,1.0);
    bestCoverage=cellRadius*(1.0-2.0*fraction);
    bestIdentity=${SPARSE_SCENE_TERRAIN_MATERIAL_OWNER}u;
  }
  let candidateCount=min(atomicLoad(&maintenance[record+1u]),candidatesPerBrick());
  for(var slot=0u;slot<candidateCount;slot+=1u){
    let primitiveIndex=atomicLoad(&maintenance[candidateOffset()+dirtyIndex*candidatesPerBrick()+slot]);
    let primitive = primitives[primitiveIndex];
    let candidate = primitiveDistance(primitive, world);
    bestDistance = min(bestDistance, candidate);
    let coverage = primitiveCellCoverageDistance(primitive, world, cellExtent, candidate);
    if (coverage < bestCoverage) {
      bestCoverage = coverage;
      bestIdentity = bitcast<u32>(primitive.extentIdentity.w);
    }
  }

  let output = voxelOffset + localIndex;
  let geometryBase = sceneGeometryOffset() + output * 4u;
  let materialOffset = sceneMaterialOffset() + output;
  let primitiveFraction = clamp(0.5 - bestCoverage / (2.0 * cellRadius), 0.0, 1.0);
  // The scene lane is exclusively owned by this transaction. Fluid and
  // velocity live in disjoint payload lanes, so every material ID—including
  // terrain and rigid-body IDs below the scenery range—can update atomically.
  payload[geometryBase+1u]=bitcast<u32>(bestDistance);
  payload[geometryBase+2u]=bitcast<u32>(primitiveFraction);
  payload[materialOffset]=select(NO_MATERIAL_OWNER,bestIdentity,primitiveFraction>0.0);
}

@compute @workgroup_size(64)
fn finalizeDirtyBricks(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) groups:vec3u){
  let chunkSlot=linearIndex64(gid,groups);
  let dirtyIndex=chunkBegin()+chunkSlot;
  let dirtyCount=dirtyBrickTotal();
  // The revision completes with its *last* chunk. Under a per-frame budget the
  // earlier chunks have repaired real bricks and published nothing, which is
  // the contract: a partially repaired publication reports the previous
  // generation, never a current one with holes in it.
  let lastChunk=chunkEnd()>=dirtyCount;
  if(chunkSlot==0u&&dirtyCount==0u&&lastChunk&&(atomicLoad(&maintenance[stateOffset()+1u])&INCOMPLETE_OVERFLOW)==0u){
    atomicStore(&maintenance[stateOffset()+3u],sceneRevision());
  }
  if(dirtyIndex>=chunkEnd()){return;}
  let record=dirtyBrickOffset()+dirtyIndex*4u;
  let leafIndex=atomicLoad(&maintenance[record]);
  let leafBase=controlLoad(16u)+leafIndex*4u;
  let nodeIndex=topologyLoad(leafBase);
  let lifecycle=topologyLoad(nodeIndex*8u+7u);
  if((lifecycle&BRICK_RELOCATING)!=0u){
    atomicOr(&maintenance[stateOffset()+1u],TOPOLOGY_INCOMPLETE);
    return;
  }
  var packed=0u;
  if(controlLoad(11u)==8u){
    let voxelOffset=topologyLoad(leafBase+1u);
    var macroMask=0u;
    var minimum=vec3u(7u);
    var maximum=vec3u(0u);
    var occupied=false;
    for(var localIndex=0u;localIndex<512u;localIndex+=1u){
      let sceneMaterial=payload[sceneMaterialOffset()+voxelOffset+localIndex]&0xffffu;
      let fluidMaterial=payload[controlLoad(18u)+voxelOffset+localIndex]&0xffffu;
      if(sceneMaterial==0u&&fluidMaterial==0u){continue;}
      let local=vec3u(localIndex&7u,(localIndex>>3u)&7u,localIndex>>6u);
      minimum=min(minimum,local);maximum=max(maximum,local);
      let macroCoordinate=local>>vec3u(2u);
      macroMask|=1u<<(macroCoordinate.x|(macroCoordinate.y<<1u)|(macroCoordinate.z<<2u));
      occupied=true;
    }
    packed=OCCUPANCY_READY|macroMask;
    if(occupied){
      packed|=OCCUPANCY_OCCUPIED;
      packed|=(minimum.x<<8u)|(minimum.y<<11u)|(minimum.z<<14u);
      packed|=(maximum.x<<17u)|(maximum.y<<20u)|(maximum.z<<23u);
    }
  }
  // Occupancy and payload become current together. ACTIVE is retained while
  // DIRTY/QUEUED are cleared only after the complete rebuild.
  //
  // An over-subscribed brick is summarized here like any other. Its voxels were
  // rebuilt from the first candidatesPerBrick() primitives, so the occupancy
  // word describes what the brick now holds; withholding it left the payload
  // current and the summary stale, which is a worse state than either.
  topologyStore(nodeIndex*8u+7u,packed|(lifecycle&BRICK_ACTIVE));
  if(chunkSlot==0u&&lastChunk&&(atomicLoad(&maintenance[stateOffset()+1u])&INCOMPLETE_OVERFLOW)==0u){
    atomicStore(&maintenance[stateOffset()+3u],sceneRevision());
  }
}
`;

function nonNegativeInteger(value: number, name: string): number {
  if (!Number.isSafeInteger(value) || value < 0) throw new RangeError(`${name} must be a non-negative integer`);
  return value;
}

function positiveInteger(value: number, name: string): number {
  if (!Number.isSafeInteger(value) || value <= 0) throw new RangeError(`${name} must be a positive integer`);
  return value;
}

function checkedArenaWords(value: number, name: string): number {
  if (!Number.isSafeInteger(value) || value < 0 || value * 4 > 0xffffffff) throw new RangeError(`${name} exceeds WebGPU buffer limits`);
  return value;
}

function validateBounds(bounds: SparseSceneAxisAlignedBounds, name: string): void {
  finiteVector(bounds.minimum, `${name} minimum`);
  finiteVector(bounds.maximum, `${name} maximum`);
  if (bounds.minimum.some((value, axis) => value > bounds.maximum[axis])) throw new RangeError(`${name} bounds must be ordered`);
}

function packBounds(bounds: readonly SparseSceneAxisAlignedBounds[]): Uint32Array<ArrayBuffer> {
  const words = new Uint32Array(new ArrayBuffer(bounds.length * 8 * 4));
  const floats = new Float32Array(words.buffer);
  bounds.forEach((bound, index) => {
    validateBounds(bound, `Bounds ${index}`);
    floats.set(bound.minimum, index * 8);
    floats.set(bound.maximum, index * 8 + 4);
  });
  return words;
}

export interface SparseSceneMaintenanceBinding {
  buffer: GPUBuffer;
  stateOffsetBytes: number;
  dirtyBrickOffsetBytes: number;
  dirtyBrickCapacity: number;
}

/** Maintenance stages, in encode order, as timestamped by {@link SparseSceneMaintenanceTimestamps}. */
export const SPARSE_SCENE_MAINTENANCE_STAGES = Object.freeze([
  "invalidate", "prepare", "bin", "rebuild", "finalize",
] as const);

/**
 * Two queries per stage, so a benchmark can price invalidation and binning
 * separately against a record count. A claim that maintenance became sub-linear
 * is a claim about these two numbers and nothing else.
 */
export interface SparseSceneMaintenanceTimestamps {
  querySet: GPUQuerySet;
  /** First of `2 * SPARSE_SCENE_MAINTENANCE_STAGES.length` consecutive query indices. */
  baseIndex: number;
}

/** Fixed-capacity, render-revision-driven maintenance of live scene geometry. */
export class SparseSceneProxyVoxelizer {
  primitiveCount = 0;
  sceneRevision = 0;
  readonly allocatedBytes: number;
  readonly maintenanceBinding: SparseSceneMaintenanceBinding;

  private readonly device: GPUDevice;
  private readonly tree: SparseBrickOctreeGPU;
  private readonly primitiveBuffer: GPUBuffer;
  private readonly clusterCapacity: number;
  private readonly maintenanceArena: GPUBuffer;
  private readonly maintenanceDispatch: GPUBuffer;
  private readonly paramsBuffer: GPUBuffer;
  private invalidatePipeline!: GPUComputePipeline;
  private indexedInvalidatePipeline?: GPUComputePipeline;
  private preparePipeline!: GPUComputePipeline;
  private binPipeline!: GPUComputePipeline;
  private indexedBinPipeline?: GPUComputePipeline;
  private rebuildPipeline!: GPUComputePipeline;
  private finalizePipeline!: GPUComputePipeline;
  private shaderModule!: GPUShaderModule;
  private readonly pipelineLayout: GPUPipelineLayout;
  private readonly label: string;
  private readonly bindGroup: GPUBindGroup;
  private readonly options: Readonly<SparseSceneProxyVoxelizerOptions>;
  private readonly primitiveBoundsOffsetWords = 0;
  private readonly dirtyRegionOffsetWords: number;
  private readonly dirtyBrickOffsetWords: number;
  private readonly candidateOffsetWords: number;
  private readonly stateOffsetWords: number;
  private readonly clusterOffsetWords: number;
  private readonly indexHeaderOffsetWords: number;
  private readonly regionCellOffsetWords: number;
  private readonly gridItemCapacity: number;
  private readonly globalRecordCapacity: number;
  private readonly terrainFieldOffsetWords: number;
  private readonly terrainFieldCapacityColumns: number;
  private readonly derivedHeaderAllocated: boolean;
  /** Columns of the configured ground, or `undefined` for a world without one. */
  private terrainFieldDimensions?: readonly [number, number];
  private readonly brickBudget: number;
  private pending = false;
  private destroyed = false;
  /**
   * Frames of chunked repair still owed for the pending revision.
   *
   * Planned on the CPU from a strict upper bound on the dirty brick count (the
   * dirty regions' brick-cell volume), never from a readback. Over-planning
   * costs an empty frame whose indirect dispatches are all zero; under-planning
   * would stop encoding before the last chunk stored `completedRevision`, and a
   * publication that never completes is the black slab.
   */
  private chunksRemaining = 0;
  private plannedChunks = 0;
  private indexedInvalidation = false;
  private lastRegionCellCount = 0;
  private lastGlobalRecordCount = 0;
  private lastGridItemCount = 0;

  constructor(
    device: GPUDevice,
    tree: SparseBrickOctreeGPU,
    options: SparseSceneProxyVoxelizerOptions,
  ) {
    positiveVector(options.cellSize, "Cell size");
    const worldOrigin = options.worldOrigin ?? [0, 0, 0];
    finiteVector(worldOrigin, "World origin");
    const primitiveCapacity = positiveInteger(options.primitiveCapacity, "Primitive capacity");
    const dirtyRegionCapacity = positiveInteger(options.dirtyRegionCapacity, "Dirty region capacity");
    const dirtyBrickCapacity = positiveInteger(options.dirtyBrickCapacity, "Dirty brick capacity");
    const candidatesPerDirtyBrick = positiveInteger(options.candidatesPerDirtyBrick, "Candidates per dirty brick");
    if (options.finestLevel !== undefined && (!Number.isInteger(options.finestLevel) || options.finestLevel < 0 || options.finestLevel > 21)) {
      throw new RangeError("Finest topology level is invalid");
    }
    this.device = device;
    this.tree = tree;
    this.options = Object.freeze({ ...options, worldOrigin });
    const primitiveBytes = primitiveCapacity * SPARSE_SCENE_PRIMITIVE_STRIDE_BYTES;
    // Zero by default, and deliberately: a world with no aggregates should not
    // carry their arena, and a caller that publishes one without asking for the
    // capacity is refused loudly in `publish` rather than drawing nothing.
    const clusterCapacity = nonNegativeInteger(options.clusterCapacity ?? 0, "Cluster capacity");
    this.dirtyRegionOffsetWords = checkedArenaWords(primitiveCapacity * 8, "Primitive bounds arena");
    this.dirtyBrickOffsetWords = checkedArenaWords(this.dirtyRegionOffsetWords + dirtyRegionCapacity * 8, "Dirty region arena");
    this.candidateOffsetWords = checkedArenaWords(this.dirtyBrickOffsetWords + dirtyBrickCapacity * 4, "Dirty brick arena");
    this.stateOffsetWords = checkedArenaWords(this.candidateOffsetWords + dirtyBrickCapacity * candidatesPerDirtyBrick, "Candidate arena");
    this.clusterOffsetWords = checkedArenaWords(this.stateOffsetWords + SPARSE_SCENE_MAINTENANCE_STATE_WORDS.wordCount, "Maintenance state");
    // Everything past here is the record index, and every block of it is zero
    // words when the index is absent — so a voxelizer configured without one
    // allocates, clears and writes exactly what it always did.
    this.indexHeaderOffsetWords = checkedArenaWords(
      this.clusterOffsetWords + clusterCapacity * SVO_CLUSTER_ARENA_BLOCK.strideWords, "Aggregate arena");
    const index = options.recordIndex;
    if (index) {
      if (options.finestLevel === undefined) throw new RangeError("A record index needs the finest topology level to place its lattice");
      for (const [axis, value] of index.dimensions.entries()) {
        if (!Number.isSafeInteger(value) || value < 1 || value >= 1 << SPARSE_SCENE_GRID_COORDINATE_BITS) {
          throw new RangeError(`Record index dimension ${axis} must be a positive integer below ${1 << SPARSE_SCENE_GRID_COORDINATE_BITS}`);
        }
      }
      positiveVector(index.cellExtent_m as SparseSceneVector3, "Record index cell extent");
    }
    this.gridItemCapacity = index ? positiveInteger(index.itemCapacity ?? primitiveCapacity * 16, "Record index item capacity") : 0;
    this.globalRecordCapacity = index ? primitiveCapacity : 0;
    const gridCells = index ? index.dimensions[0] * index.dimensions[1] * index.dimensions[2] : 0;
    // The header is the one place in the arena whose address the shader derives
    // rather than being told, so it exists whenever *anything* it describes
    // does — and a world may have terrain without a record index, or the
    // reverse. A world with neither still allocates and writes exactly what it
    // always did.
    this.terrainFieldCapacityColumns = nonNegativeInteger(
      options.terrainFieldCapacityColumns ?? 0, "Terrain field capacity");
    this.derivedHeaderAllocated = index !== undefined || this.terrainFieldCapacityColumns > 0;
    this.regionCellOffsetWords = checkedArenaWords(
      this.indexHeaderOffsetWords + (this.derivedHeaderAllocated ? SPARSE_SCENE_INDEX_HEADER.wordCount : 0),
      "Derived block header");
    const leafClaimOffsetWords = checkedArenaWords(
      this.regionCellOffsetWords + (index ? dirtyRegionCapacity * SPARSE_SCENE_REGION_CELL_WORDS : 0), "Dirty region cell arena");
    const gridOffsetsOffsetWords = checkedArenaWords(
      leafClaimOffsetWords + (index ? dirtyBrickCapacity : 0), "Leaf claim arena");
    const gridItemsOffsetWords = checkedArenaWords(
      gridOffsetsOffsetWords + (index ? gridCells + 1 : 0), "Record grid offsets");
    const globalRecordsOffsetWords = checkedArenaWords(
      gridItemsOffsetWords + this.gridItemCapacity, "Record grid items");
    const recordCellRangeOffsetWords = checkedArenaWords(
      globalRecordsOffsetWords + this.globalRecordCapacity, "Unlocalized record list");
    // Last block in the arena, and the only one whose offset is transmitted
    // rather than derived — in the header, because both spare uniform lanes are
    // already spent. Putting it last keeps every existing offset untouched.
    this.terrainFieldOffsetWords = checkedArenaWords(
      recordCellRangeOffsetWords + (index ? primitiveCapacity * 2 : 0), "Unlocalized record cell ranges");
    const arenaWords = checkedArenaWords(
      this.terrainFieldOffsetWords + this.terrainFieldCapacityColumns, "Scene maintenance arena");
    this.brickBudget = nonNegativeInteger(options.bricksPerFrameBudget ?? 0, "Bricks per frame budget");
    const maintenanceDispatchBytes = 3 * 3 * 4;
    this.allocatedBytes = primitiveBytes + arenaWords * 4 + maintenanceDispatchBytes + 96;
    const label = options.label ?? "Sparse scene proxies";
    this.label = label;
    this.primitiveBuffer = device.createBuffer({
      label: `${label} primitives`, size: primitiveBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.clusterCapacity = clusterCapacity;
    this.maintenanceArena = device.createBuffer({
      label: `${label} fixed maintenance arena`, size: arenaWords * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    this.maintenanceDispatch = device.createBuffer({
      label: `${label} maintenance dispatch arguments`, size: maintenanceDispatchBytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.INDIRECT,
    });
    this.paramsBuffer = device.createBuffer({
      label: `${label} parameters`, size: 96,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const layout = device.createBindGroupLayout({
      label: `${label} live maintenance layout`,
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      ],
    });
    this.pipelineLayout = device.createPipelineLayout({ label: `${label} live maintenance pipeline layout`, bindGroupLayouts: [layout] });
    this.bindGroup = device.createBindGroup({
      label: `${label} live maintenance bind group`, layout,
      entries: [
        { binding: 0, resource: { buffer: tree.structure } },
        { binding: 1, resource: { buffer: tree.payload } },
        { binding: 2, resource: { buffer: this.primitiveBuffer } },
        { binding: 3, resource: { buffer: this.maintenanceArena } },
        { binding: 4, resource: { buffer: this.paramsBuffer } },
      ],
    });
    this.maintenanceBinding = Object.freeze({
      buffer: this.maintenanceArena,
      stateOffsetBytes: this.stateOffsetWords * 4,
      dirtyBrickOffsetBytes: this.dirtyBrickOffsetWords * 4,
      dirtyBrickCapacity,
    });
  }

  async initializePipelines(): Promise<void> {
    if (this.invalidatePipeline) return;
    this.shaderModule = this.device.createShaderModule({
      label: `${this.label} live maintenance shader`, code: sparseSceneProxyVoxelizationShader,
    });
    const pipeline = (entryPoint: string, stage: string) => this.device.createComputePipelineAsync({
      label: `${this.label} ${stage} pipeline`, layout: this.pipelineLayout,
      compute: { module: this.shaderModule, entryPoint },
    });
    this.invalidatePipeline = await pipeline("invalidateDirtyBricks", "invalidation");
    this.preparePipeline = await pipeline("prepareMaintenanceDispatch", "bounded dispatch preparation");
    this.binPipeline = await pipeline("binDirtyBrickCandidates", "candidate binning");
    this.rebuildPipeline = await pipeline("rebuildDirtyBrickPayload", "payload rebuild");
    this.finalizePipeline = await pipeline("finalizeDirtyBricks", "finalization");
    if (this.options.recordIndex) {
      this.indexedInvalidatePipeline = await pipeline("invalidateDirtyBricksFromRegions", "indexed invalidation");
      this.indexedBinPipeline = await pipeline("binDirtyBrickCandidatesIndexed", "indexed candidate binning");
    }
  }

  /**
   * Upload the ground, or clear it.
   *
   * Independent of `publish` on purpose: the ground is static in every scene
   * that has one, so it is written once and read by every rebuild from then on,
   * and a still scene stays at zero maintenance dispatches because nothing about
   * terrain is republished per revision. A sculpting stroke calls this again and
   * separately dirties the field's bounds — the caller owns that pairing,
   * because the caller owns the revision the dirty region belongs to.
   */
  configureTerrainField(field: SparseSceneTerrainField | undefined): void {
    if (this.destroyed) throw new Error("Cannot configure terrain on a destroyed scene voxelizer");
    if (!field) {
      this.terrainFieldDimensions = undefined;
      this.writeDerivedHeaderTerrain();
      return;
    }
    const [nx, nz] = field.dimensions;
    const columns = nx * nz;
    if (!Number.isSafeInteger(columns) || columns < 1) throw new RangeError("Terrain field must hold at least one column");
    if (columns !== field.heights_m.length) throw new RangeError("Terrain field dimensions do not match its heights");
    if (columns > this.terrainFieldCapacityColumns) {
      throw new RangeError(`Terrain field needs ${columns} columns but the fixed arena holds ${this.terrainFieldCapacityColumns}`);
    }
    this.terrainFieldDimensions = [nx, nz];
    this.device.queue.writeBuffer(this.maintenanceArena, this.terrainFieldOffsetWords * 4, field.heights_m);
    this.writeDerivedHeaderTerrain();
  }

  /** Whether the ground reaches the scene lanes, and how wide its lattice is. */
  get terrainFieldStatus(): { present: boolean; columns: number; capacityColumns: number } {
    const dimensions = this.terrainFieldDimensions;
    return {
      present: dimensions !== undefined,
      columns: dimensions ? dimensions[0] * dimensions[1] : 0,
      capacityColumns: this.terrainFieldCapacityColumns,
    };
  }

  /** The three header words describing the ground, in the layout the shader reads. */
  private terrainHeaderWords(): Uint32Array<ArrayBuffer> {
    const dimensions = this.terrainFieldDimensions;
    return new Uint32Array([
      this.terrainFieldOffsetWords,
      dimensions?.[0] ?? 0,
      dimensions?.[1] ?? 0,
    ]);
  }

  private writeDerivedHeaderTerrain(): void {
    // A world with neither an index nor a terrain capacity has no header block
    // at all, and writing one would run off the end of the arena. Nothing reads
    // these words in that configuration — the uniform flag is what decides
    // whether the ground exists — so there is nothing to write.
    if (!this.derivedHeaderAllocated) return;
    this.device.queue.writeBuffer(this.maintenanceArena,
      (this.indexHeaderOffsetWords + SPARSE_SCENE_INDEX_HEADER.terrainFieldOffset) * 4,
      this.terrainHeaderWords());
  }

  /** Whether a budgeted publication is still converging, and how far it has to go. */
  get budgetStatus(): { budget: number; plannedChunks: number; chunksRemaining: number; converging: boolean } {
    return {
      budget: this.brickBudget,
      plannedChunks: this.plannedChunks,
      chunksRemaining: this.chunksRemaining,
      converging: this.chunksRemaining > 0 || this.pending,
    };
  }

  /** What the last publication asked of the record index, for lanes that measure it. */
  get indexStatus(): {
    enabled: boolean;
    indexedInvalidation: boolean;
    regionCells: number;
    gridItems: number;
    globalRecords: number;
  } {
    return {
      enabled: this.options.recordIndex !== undefined,
      indexedInvalidation: this.indexedInvalidation,
      regionCells: this.lastRegionCellCount,
      gridItems: this.lastGridItemCount,
      globalRecords: this.lastGlobalRecordCount,
    };
  }

  /** Hot-publish one authoritative render-scene revision without reallocating GPU resources. */
  publish(publication: SparseScenePublication): void {
    if (this.destroyed) throw new Error("Cannot publish to a destroyed scene voxelizer");
    if (this.pending) throw new Error("Encode the pending scene revision before publishing another");
    if (!Number.isSafeInteger(publication.revision) || publication.revision <= this.sceneRevision || publication.revision > 0xffffffff) {
      throw new RangeError("Scene revision must be a strictly increasing nonzero uint32");
    }
    if (publication.primitives.length > this.options.primitiveCapacity) throw new RangeError("Live scene primitive capacity exceeded");
    if (publication.dirtyRegions.length === 0) throw new RangeError("A scene publication must cover its old/new dirty bounds");
    if (publication.dirtyRegions.length > this.options.dirtyRegionCapacity) throw new RangeError("Live scene dirty-region capacity exceeded");
    const packed = packSparseScenePrimitives(publication.primitives);
    // Republished whole with every revision. A record names its block by slot,
    // and slots are publication order over the aggregate subsequence, so one
    // added or removed aggregate renumbers every block after it — writing only
    // the changed ones would leave the rest naming their predecessors' shapes.
    const aggregates = sparseSceneClusterPackings(publication.primitives).length;
    if (aggregates > this.clusterCapacity) {
      throw new RangeError(`Live scene publishes ${aggregates} aggregates but the fixed arena holds ${this.clusterCapacity}`);
    }
    const clusterArena = aggregates > 0
      ? packSparseSceneClusterArena(publication.primitives, this.clusterCapacity)
      : undefined;
    const bounds = publication.primitives.map(sparseScenePrimitiveBounds);
    const primitiveBounds = packBounds(bounds);
    const dirtyBounds = packBounds(publication.dirtyRegions);
    const worldOrigin = (this.options.worldOrigin ?? [0, 0, 0]) as SparseSceneVector3;
    // The dirty regions' brick-cell volume is a strict upper bound on the leaves
    // they can dirty (leaves are disjoint unions of those cells), so it decides
    // both the indexed invalidation dispatch and how many budgeted frames this
    // revision needs — without a readback and without ever under-planning.
    const regionCells = this.options.finestLevel === undefined ? undefined : planSparseSceneRegionCells(
      publication.dirtyRegions, worldOrigin, this.options.cellSize, this.tree.brickSize, this.options.finestLevel,
    );
    const upperBoundDirtyBricks = Math.min(
      this.options.dirtyBrickCapacity, regionCells?.cellCount ?? this.options.dirtyBrickCapacity,
    );
    const budgeted = (publication.budgeted ?? true) && this.brickBudget > 0;
    this.plannedChunks = budgeted ? Math.max(1, Math.ceil(upperBoundDirtyBricks / this.brickBudget)) : 1;
    // Only worth descending per cell when there are fewer cells than the leaf
    // scan would touch anyway; a region covering more of the lattice than the
    // tree has leaves is exactly what the unindexed path is still here for.
    const index = this.options.recordIndex;
    this.indexedInvalidation = index !== undefined && regionCells !== undefined
      && regionCells.cellCount <= 8 * this.options.dirtyBrickCapacity;
    this.lastRegionCellCount = regionCells?.cellCount ?? 0;
    if (packed.byteLength > 0) this.device.queue.writeBuffer(this.primitiveBuffer, 0, packed);
    if (clusterArena) this.device.queue.writeBuffer(this.maintenanceArena, this.clusterOffsetWords * 4, clusterArena);
    if (primitiveBounds.byteLength > 0) this.device.queue.writeBuffer(this.maintenanceArena, this.primitiveBoundsOffsetWords * 4, primitiveBounds);
    this.device.queue.writeBuffer(this.maintenanceArena, this.dirtyRegionOffsetWords * 4, dirtyBounds);
    if (index && regionCells) this.writeRecordIndex(index, bounds, regionCells, worldOrigin);
    const parameterData = new ArrayBuffer(96);
    const floats = new Float32Array(parameterData);
    const uints = new Uint32Array(parameterData);
    floats.set(worldOrigin, 0);
    floats.set(this.options.cellSize, 4);
    uints[3] = budgeted ? this.brickBudget : 0;
    uints[7] = (index ? SPARSE_SCENE_MAINTENANCE_FLAGS.indexedBinning : 0)
      | (this.terrainFieldDimensions ? SPARSE_SCENE_MAINTENANCE_FLAGS.terrainField : 0);
    uints.set([publication.primitives.length, publication.dirtyRegions.length, this.options.finestLevel ?? 0xffffffff, publication.revision], 8);
    uints.set([this.options.dirtyBrickCapacity, this.options.candidatesPerDirtyBrick, this.primitiveBoundsOffsetWords, this.dirtyRegionOffsetWords], 12);
    uints.set([this.dirtyBrickOffsetWords, this.candidateOffsetWords, this.stateOffsetWords, this.clusterOffsetWords], 16);
    uints.set([
      this.tree.sceneGeometryOffsetBytes / 4,
      this.tree.sceneMaterialOwnerOffsetBytes / 4,
      this.tree.topologyOffsetBytes / 4,
      this.clusterCapacity,
    ], 20);
    this.device.queue.writeBuffer(this.paramsBuffer, 0, parameterData);
    this.primitiveCount = publication.primitives.length;
    this.sceneRevision = publication.revision;
    this.pending = true;
  }

  /** Header, dirty-region cell boxes, coarse record grid, and the oversized records' list. */
  private writeRecordIndex(
    index: SparseSceneRecordIndexOptions,
    bounds: readonly SparseSceneAxisAlignedBounds[],
    regionCells: SparseSceneRegionCellPlan,
    worldOrigin: SparseSceneVector3,
  ): void {
    const grid = planSparseSceneRecordGrid(bounds, worldOrigin, index, this.gridItemCapacity);
    if (grid.globalRecords.length > this.globalRecordCapacity) {
      throw new RangeError("Live scene record index cannot hold every unlocalized record");
    }
    this.lastGridItemCount = grid.items.length;
    this.lastGlobalRecordCount = grid.globalRecords.length;
    // The header and the region cells are contiguous, so they ride one write.
    const header = new Uint32Array(new ArrayBuffer(
      (SPARSE_SCENE_INDEX_HEADER.wordCount + regionCells.words.length) * 4));
    const headerFloats = new Float32Array(header.buffer);
    header[SPARSE_SCENE_INDEX_HEADER.gridDimensionX] = index.dimensions[0];
    header[SPARSE_SCENE_INDEX_HEADER.gridDimensionY] = index.dimensions[1];
    header[SPARSE_SCENE_INDEX_HEADER.gridDimensionZ] = index.dimensions[2];
    header[SPARSE_SCENE_INDEX_HEADER.globalRecordCount] = grid.globalRecords.length;
    header[SPARSE_SCENE_INDEX_HEADER.invalidationCellCount] = regionCells.cellCount;
    header[SPARSE_SCENE_INDEX_HEADER.gridItemCapacity] = this.gridItemCapacity;
    headerFloats.set(index.cellExtent_m, SPARSE_SCENE_INDEX_HEADER.gridExtentX);
    // The header is shared, and this write covers all sixteen words: restating
    // the terrain lanes here is what stops a publication from zeroing a ground
    // it knows nothing about.
    header.set(this.terrainHeaderWords(), SPARSE_SCENE_INDEX_HEADER.terrainFieldOffset);
    header.set(regionCells.words, SPARSE_SCENE_INDEX_HEADER.wordCount);
    this.device.queue.writeBuffer(this.maintenanceArena, this.indexHeaderOffsetWords * 4, header);
    const gridOffsetsOffsetWords = this.regionCellOffsetWords
      + this.options.dirtyRegionCapacity * SPARSE_SCENE_REGION_CELL_WORDS + this.options.dirtyBrickCapacity;
    const globalRecordsOffsetWords = gridOffsetsOffsetWords + grid.offsets.length + this.gridItemCapacity;
    this.device.queue.writeBuffer(this.maintenanceArena, gridOffsetsOffsetWords * 4, grid.offsets);
    if (grid.items.length > 0) {
      this.device.queue.writeBuffer(this.maintenanceArena, (gridOffsetsOffsetWords + grid.offsets.length) * 4, grid.items);
    }
    if (grid.globalRecords.length > 0) {
      this.device.queue.writeBuffer(this.maintenanceArena, globalRecordsOffsetWords * 4, grid.globalRecords);
    }
    if (grid.cellRanges.length > 0) {
      this.device.queue.writeBuffer(this.maintenanceArena,
        (globalRecordsOffsetWords + this.globalRecordCapacity) * 4, grid.cellRanges);
    }
  }

  /**
   * Encode invalidation, brick-local binning, authoritative rebuild, then
   * finalization — or, on a continuation frame, everything but invalidation.
   *
   * The distinction is the budget. A new revision clears the state block and
   * rebuilds the dirty list; a continuation must not, because the dirty list,
   * the overflow mask and the requested revision are precisely what it is
   * carrying forward. The cursor in that state block is what tells the GPU which
   * chunk of the list this frame owns.
   */
  encodeMaintenance(encoder: GPUCommandEncoder, timestamps?: SparseSceneMaintenanceTimestamps): boolean {
    if (this.destroyed) return false;
    const fresh = this.pending;
    if (!fresh && this.chunksRemaining <= 0) return false;
    if (fresh) {
      encoder.clearBuffer(this.maintenanceArena, this.stateOffsetWords * 4, SPARSE_SCENE_MAINTENANCE_STATE_WORDS.wordCount * 4);
      this.chunksRemaining = this.plannedChunks;
    }
    const stageWrites = (stage: (typeof SPARSE_SCENE_MAINTENANCE_STAGES)[number]): GPUComputePassTimestampWrites | undefined => {
      if (!timestamps) return undefined;
      const slot = timestamps.baseIndex + 2 * SPARSE_SCENE_MAINTENANCE_STAGES.indexOf(stage);
      return { querySet: timestamps.querySet, beginningOfPassWriteIndex: slot, endOfPassWriteIndex: slot + 1 };
    };
    const run = (
      stage: (typeof SPARSE_SCENE_MAINTENANCE_STAGES)[number],
      label: string, pipeline: GPUComputePipeline, workItems: number, workgroupSize: number,
    ) => {
      const pass = encoder.beginComputePass({ label, timestampWrites: stageWrites(stage) });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, this.bindGroup);
      pass.dispatchWorkgroups(...sparseBrickDispatchDimensions(Math.ceil(workItems * 256 / workgroupSize)));
      pass.end();
    };
    if (fresh) {
      const indexed = this.indexedInvalidation ? this.indexedInvalidatePipeline : undefined;
      if (indexed) run("invalidate", "Invalidate live scene dirty bricks", indexed, Math.max(1, this.lastRegionCellCount), 64);
      else run("invalidate", "Invalidate live scene dirty bricks", this.invalidatePipeline, this.tree.leafCapacity, 64);
    }
    run("prepare", "Prepare live scene maintenance dispatches", this.preparePipeline, 1, 1);
    encoder.copyBufferToBuffer(
      this.maintenanceArena,
      (this.stateOffsetWords + SPARSE_SCENE_MAINTENANCE_STATE_WORDS.binDispatch) * 4,
      this.maintenanceDispatch,
      0,
      3 * 3 * 4,
    );
    const runIndirect = (
      stage: (typeof SPARSE_SCENE_MAINTENANCE_STAGES)[number],
      label: string, pipeline: GPUComputePipeline, stateWord: number,
    ) => {
      const pass = encoder.beginComputePass({ label, timestampWrites: stageWrites(stage) });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, this.bindGroup);
      pass.dispatchWorkgroupsIndirect(
        this.maintenanceDispatch,
        (stateWord - SPARSE_SCENE_MAINTENANCE_STATE_WORDS.binDispatch) * 4,
      );
      pass.end();
    };
    runIndirect("bin", "Bin live scene primitives into dirty bricks",
      this.indexedBinPipeline ?? this.binPipeline, SPARSE_SCENE_MAINTENANCE_STATE_WORDS.binDispatch);
    runIndirect("rebuild", "Rebuild live scene dirty brick payloads", this.rebuildPipeline, SPARSE_SCENE_MAINTENANCE_STATE_WORDS.rebuildDispatch);
    runIndirect("finalize", "Finalize live scene dirty bricks", this.finalizePipeline, SPARSE_SCENE_MAINTENANCE_STATE_WORDS.finalizeDispatch);
    this.pending = false;
    this.chunksRemaining -= 1;
    return true;
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.primitiveBuffer.destroy();
    this.maintenanceArena.destroy();
    this.maintenanceDispatch.destroy();
    this.paramsBuffer.destroy();
  }
}
