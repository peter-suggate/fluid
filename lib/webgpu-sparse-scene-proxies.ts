import {
  SPARSE_BRICK_NO_OWNER,
  packMaterialOwner,
  sparseBrickDispatchDimensions,
  type SparseBrickOctreeGPU,
} from "./sparse-brick-octree";
import { SVO_BRICK_LIFECYCLE, SVO_BRICK_OCCUPANCY } from "./svo-brick-occupancy";
import type { EnvironmentProxyPrimitive } from "./voxel-environments";
import type { SvoPrimitiveDescriptor } from "./svo-primitive-abi";

export type SparseSceneVector3 = readonly [number, number, number];
export type SparseSceneQuaternion = readonly [number, number, number, number];

export const SPARSE_SCENE_PRIMITIVE_STRIDE_BYTES = 48;

/**
 * Voxelization keeps its own compact vocabulary rather than sharing the render
 * ABI's records: this pass evaluates every primitive at every candidate voxel,
 * so it wants the cheapest distance that still classifies a cell, where the
 * renderer wants the exact one. Tags are its own; only the shapes are shared.
 */
export const SPARSE_SCENE_PRIMITIVE_TYPES = Object.freeze({
  box: 1,
  cylinder: 2,
  ellipsoid: 3,
  capsule: 4,
  torus: 5,
  cone: 6,
} as const);

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

export type SparseScenePrimitive =
  | SparseSceneBoxPrimitive
  | SparseSceneCylinderPrimitive
  | SparseSceneEllipsoidPrimitive
  | SparseSceneCapsulePrimitive
  | SparseSceneTorusPrimitive
  | SparseSceneConePrimitive;

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
  if (descriptor.kind === "torus") return { ...base, kind: "torus", majorRadius: descriptor.majorRadius_m, minorRadius: descriptor.minorRadius_m };
  if (descriptor.kind === "cone") return { ...base, kind: "cone", baseRadius: descriptor.baseRadius_m, topRadius: descriptor.topRadius_m, halfHeight: descriptor.halfHeight_m };
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
  if (proxy.kind === "cylinder") return { ...base, kind: "cylinder", radius: proxy.radius_m, halfHeight: proxy.halfHeight_m };
  if (proxy.kind === "capsule") return { ...base, kind: "capsule", radius: proxy.radius_m, halfLength: proxy.halfLength_m };
  if (proxy.kind === "torus") return { ...base, kind: "torus", majorRadius: proxy.majorRadius_m, minorRadius: proxy.minorRadius_m };
  if (proxy.kind === "cone") {
    return { ...base, kind: "cone", baseRadius: proxy.baseRadius_m, topRadius: proxy.topRadius_m, halfHeight: proxy.halfHeight_m };
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
  label?: string;
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
}

export const SPARSE_SCENE_MAINTENANCE_STATE_WORDS = Object.freeze({
  dirtyBrickCount: 0,
  overflowFlags: 1,
  requestedRevision: 2,
  completedRevision: 3,
  primitiveCount: 4,
  dirtyRegionCount: 5,
  binDispatch: 8,
  rebuildDispatch: 11,
  finalizeDispatch: 14,
  wordCount: 20,
} as const);

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
  if (primitive.kind === "cone") {
    if (!Number.isFinite(primitive.halfHeight) || primitive.halfHeight <= 0
        || !(primitive.baseRadius >= 0) || !(primitive.topRadius >= 0)
        || !(Math.max(primitive.baseRadius, primitive.topRadius) > 0)) {
      throw new RangeError("Cone half height must be positive with a positive radius at one end");
    }
    return [primitive.baseRadius, primitive.halfHeight, primitive.topRadius];
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
 * Pack three vec4 words per primitive. The fourth lane of the first two vec4s
 * is bitcast on GPU to preserve exact type and material/owner integer identity:
 * `{ center.xyz, type }, { extent.xyz, packedIdentity }, { quaternion.xyzw }`.
 */
export function packSparseScenePrimitives(primitives: readonly SparseScenePrimitive[]): Uint32Array<ArrayBuffer> {
  const words = new Uint32Array(new ArrayBuffer(primitives.length * SPARSE_SCENE_PRIMITIVE_STRIDE_BYTES));
  const floats = new Float32Array(words.buffer);
  for (let index = 0; index < primitives.length; index += 1) {
    const primitive = primitives[index];
    finiteVector(primitive.center, "Primitive center");
    validateIdentity(primitive.materialId, primitive.ownerId);
    const extent = primitiveExtent(primitive);
    const orientation = normalizedQuaternion(primitive.orientation);
    const base = index * (SPARSE_SCENE_PRIMITIVE_STRIDE_BYTES / 4);
    floats.set(primitive.center, base);
    words[base + 3] = primitiveType(primitive);
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
  positiveVector(primitive.radii, "Ellipsoid radii");
  return ellipsoidDistance(local, primitive.radii);
}

/** Conservative cell-center occupancy mirror used by the GPU voxelization pass. */
export function sampleSparseScenePrimitiveCell(
  primitives: readonly SparseScenePrimitive[],
  cellCenter: SparseSceneVector3,
  cellSize: SparseSceneVector3,
): SparseSceneCellSample {
  positiveVector(cellSize, "Cell size");
  let distance = Number.POSITIVE_INFINITY;
  let identity = packMaterialOwner(0, SPARSE_BRICK_NO_OWNER);
  for (const primitive of primitives) {
    const candidate = sparseScenePrimitiveSignedDistance(primitive, cellCenter);
    if (candidate < distance) {
      distance = candidate;
      identity = packMaterialOwner(primitive.materialId, primitive.ownerId);
    }
  }
  if (primitives.length === 0) return { solidSignedDistance: distance, solidFraction: 0, materialOwner: identity };
  const cellRadius = 0.5 * Math.hypot(...cellSize);
  const fraction = Math.max(0, Math.min(1, 0.5 - distance / (2 * cellRadius)));
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
fn primitiveDistance(primitive: ScenePrimitive, world: vec3f) -> f32 {
  let primitiveType = bitcast<u32>(primitive.centerType.w);
  let offset = world - primitive.centerType.xyz;
  let local = inverseRotate(offset, primitive.rotation);
  if (primitiveType == 1u) { return boxDistance(local, primitive.extentIdentity.xyz); }
  if (primitiveType == 2u) { return cylinderDistance(local, primitive.extentIdentity.x, primitive.extentIdentity.y); }
  if (primitiveType == 3u) { return ellipsoidDistance(local, primitive.extentIdentity.xyz); }
  if (primitiveType == 4u) { return capsuleDistance(local, primitive.extentIdentity.x, primitive.extentIdentity.y); }
  if (primitiveType == 5u) { return torusDistance(local, primitive.extentIdentity.x, primitive.extentIdentity.y); }
  if (primitiveType == 6u) { return coneDistance(local, primitive.extentIdentity.x, primitive.extentIdentity.y, primitive.extentIdentity.z); }
  return 1e20;
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

@compute @workgroup_size(64)
fn invalidateDirtyBricks(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) groups:vec3u){
  let leafIndex=linearIndex64(gid,groups);
  if(leafIndex>=controlLoad(1u)){return;}
  if(leafIndex==0u){
    atomicStore(&maintenance[stateOffset()+2u],sceneRevision());
    atomicStore(&maintenance[stateOffset()+4u],primitiveCount());
    atomicStore(&maintenance[stateOffset()+5u],dirtyRegionCount());
  }
  let brickBounds=leafBounds(leafIndex);
  var affected=false;
  for(var regionIndex=0u;regionIndex<dirtyRegionCount();regionIndex+=1u){
    if(boundsOverlap(brickBounds,arenaBounds(dirtyRegionOffset()+regionIndex*8u))){affected=true;break;}
  }
  if(!affected){return;}
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

@compute @workgroup_size(1)
fn prepareMaintenanceDispatch(){
  let dirtyCount=min(atomicLoad(&maintenance[stateOffset()]),dirtyBrickCapacity());
  writeDispatch(stateOffset()+${SPARSE_SCENE_MAINTENANCE_STATE_WORDS.binDispatch}u,dirtyCount*primitiveCount(),256u);
  let brickSize=controlLoad(11u);
  writeDispatch(stateOffset()+${SPARSE_SCENE_MAINTENANCE_STATE_WORDS.rebuildDispatch}u,dirtyCount*brickSize*brickSize*brickSize,256u);
  writeDispatch(stateOffset()+${SPARSE_SCENE_MAINTENANCE_STATE_WORDS.finalizeDispatch}u,dirtyCount,64u);
  // Only invalidation has run at this point, so the mask can only be carrying
  // DIRTY_BRICK_OVERFLOW here; it is spelled the same way as the two later
  // stores so the completion rule has exactly one form in this shader.
  if(dirtyCount==0u&&(atomicLoad(&maintenance[stateOffset()+1u])&INCOMPLETE_OVERFLOW)==0u){
    atomicStore(&maintenance[stateOffset()+3u],sceneRevision());
  }
}

@compute @workgroup_size(256)
fn binDirtyBrickCandidates(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) groups:vec3u){
  if(primitiveCount()==0u){return;}
  let index=linearIndex256(gid,groups);
  let dirtyIndex=index/primitiveCount();
  let primitiveIndex=index-dirtyIndex*primitiveCount();
  let dirtyCount=min(atomicLoad(&maintenance[stateOffset()]),dirtyBrickCapacity());
  if(dirtyIndex>=dirtyCount){return;}
  let record=dirtyBrickOffset()+dirtyIndex*4u;
  let leafIndex=atomicLoad(&maintenance[record]);
  if(!boundsOverlap(leafBounds(leafIndex),arenaBounds(primitiveBoundsOffset()+primitiveIndex*8u))){return;}
  let localSlot=atomicAdd(&maintenance[record+1u],1u);
  if(localSlot<candidatesPerBrick()){
    atomicStore(&maintenance[candidateOffset()+dirtyIndex*candidatesPerBrick()+localSlot],primitiveIndex);
  }else{
    atomicStore(&maintenance[record+2u],1u);
    atomicOr(&maintenance[stateOffset()+1u],CANDIDATE_OVERFLOW);
  }
}

@compute @workgroup_size(256)
fn rebuildDirtyBrickPayload(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) groups:vec3u){
  let index=linearIndex256(gid,groups);
  let brickSize = controlLoad(11u);
  let voxelsPerBrick = brickSize * brickSize * brickSize;
  let dirtyIndex=index/voxelsPerBrick;
  let dirtyCount=min(atomicLoad(&maintenance[stateOffset()]),dirtyBrickCapacity());
  if(dirtyIndex>=dirtyCount){return;}
  let localIndex=index-dirtyIndex*voxelsPerBrick;
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

  var bestDistance = 1e20;
  var bestIdentity = NO_MATERIAL_OWNER;
  let candidateCount=min(atomicLoad(&maintenance[record+1u]),candidatesPerBrick());
  for(var slot=0u;slot<candidateCount;slot+=1u){
    let primitiveIndex=atomicLoad(&maintenance[candidateOffset()+dirtyIndex*candidatesPerBrick()+slot]);
    let primitive = primitives[primitiveIndex];
    let candidate = primitiveDistance(primitive, world);
    if (candidate < bestDistance) {
      bestDistance = candidate;
      bestIdentity = bitcast<u32>(primitive.extentIdentity.w);
    }
  }

  let output = voxelOffset + localIndex;
  let geometryBase = sceneGeometryOffset() + output * 4u;
  let materialOffset = sceneMaterialOffset() + output;
  let cellRadius = 0.5 * length(params.cell.xyz * f32(scale));
  let primitiveFraction = clamp(0.5 - bestDistance / (2.0 * cellRadius), 0.0, 1.0);
  // The scene lane is exclusively owned by this transaction. Fluid and
  // velocity live in disjoint payload lanes, so every material ID—including
  // terrain and rigid-body IDs below the scenery range—can update atomically.
  payload[geometryBase+1u]=bitcast<u32>(bestDistance);
  payload[geometryBase+2u]=bitcast<u32>(primitiveFraction);
  payload[materialOffset]=select(NO_MATERIAL_OWNER,bestIdentity,primitiveFraction>0.0);
}

@compute @workgroup_size(64)
fn finalizeDirtyBricks(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) groups:vec3u){
  let dirtyIndex=linearIndex64(gid,groups);
  let dirtyCount=min(atomicLoad(&maintenance[stateOffset()]),dirtyBrickCapacity());
  if(dirtyIndex==0u&&dirtyCount==0u&&(atomicLoad(&maintenance[stateOffset()+1u])&INCOMPLETE_OVERFLOW)==0u){
    atomicStore(&maintenance[stateOffset()+3u],sceneRevision());
  }
  if(dirtyIndex>=dirtyCount){return;}
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
  if(dirtyIndex==0u&&(atomicLoad(&maintenance[stateOffset()+1u])&INCOMPLETE_OVERFLOW)==0u){
    atomicStore(&maintenance[stateOffset()+3u],sceneRevision());
  }
}
`;

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

/** Fixed-capacity, render-revision-driven maintenance of live scene geometry. */
export class SparseSceneProxyVoxelizer {
  primitiveCount = 0;
  sceneRevision = 0;
  readonly allocatedBytes: number;
  readonly maintenanceBinding: SparseSceneMaintenanceBinding;

  private readonly device: GPUDevice;
  private readonly tree: SparseBrickOctreeGPU;
  private readonly primitiveBuffer: GPUBuffer;
  private readonly maintenanceArena: GPUBuffer;
  private readonly maintenanceDispatch: GPUBuffer;
  private readonly paramsBuffer: GPUBuffer;
  private invalidatePipeline!: GPUComputePipeline;
  private preparePipeline!: GPUComputePipeline;
  private binPipeline!: GPUComputePipeline;
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
  private pending = false;
  private destroyed = false;

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
    this.dirtyRegionOffsetWords = checkedArenaWords(primitiveCapacity * 8, "Primitive bounds arena");
    this.dirtyBrickOffsetWords = checkedArenaWords(this.dirtyRegionOffsetWords + dirtyRegionCapacity * 8, "Dirty region arena");
    this.candidateOffsetWords = checkedArenaWords(this.dirtyBrickOffsetWords + dirtyBrickCapacity * 4, "Dirty brick arena");
    this.stateOffsetWords = checkedArenaWords(this.candidateOffsetWords + dirtyBrickCapacity * candidatesPerDirtyBrick, "Candidate arena");
    const arenaWords = checkedArenaWords(this.stateOffsetWords + SPARSE_SCENE_MAINTENANCE_STATE_WORDS.wordCount, "Scene maintenance arena");
    const maintenanceDispatchBytes = 3 * 3 * 4;
    this.allocatedBytes = primitiveBytes + arenaWords * 4 + maintenanceDispatchBytes + 96;
    const label = options.label ?? "Sparse scene proxies";
    this.label = label;
    this.primitiveBuffer = device.createBuffer({
      label: `${label} primitives`, size: primitiveBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
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
    const primitiveBounds = packBounds(publication.primitives.map(sparseScenePrimitiveBounds));
    const dirtyBounds = packBounds(publication.dirtyRegions);
    if (packed.byteLength > 0) this.device.queue.writeBuffer(this.primitiveBuffer, 0, packed);
    if (primitiveBounds.byteLength > 0) this.device.queue.writeBuffer(this.maintenanceArena, this.primitiveBoundsOffsetWords * 4, primitiveBounds);
    this.device.queue.writeBuffer(this.maintenanceArena, this.dirtyRegionOffsetWords * 4, dirtyBounds);
    const parameterData = new ArrayBuffer(96);
    const floats = new Float32Array(parameterData);
    const uints = new Uint32Array(parameterData);
    floats.set(this.options.worldOrigin ?? [0, 0, 0], 0);
    floats.set(this.options.cellSize, 4);
    uints.set([publication.primitives.length, publication.dirtyRegions.length, this.options.finestLevel ?? 0xffffffff, publication.revision], 8);
    uints.set([this.options.dirtyBrickCapacity, this.options.candidatesPerDirtyBrick, this.primitiveBoundsOffsetWords, this.dirtyRegionOffsetWords], 12);
    uints.set([this.dirtyBrickOffsetWords, this.candidateOffsetWords, this.stateOffsetWords, 0], 16);
    uints.set([
      this.tree.sceneGeometryOffsetBytes / 4,
      this.tree.sceneMaterialOwnerOffsetBytes / 4,
      this.tree.topologyOffsetBytes / 4,
      0,
    ], 20);
    this.device.queue.writeBuffer(this.paramsBuffer, 0, parameterData);
    this.primitiveCount = publication.primitives.length;
    this.sceneRevision = publication.revision;
    this.pending = true;
  }

  /** Encode invalidation, brick-local binning, authoritative rebuild, then finalization. */
  encodeMaintenance(encoder: GPUCommandEncoder): boolean {
    if (this.destroyed || !this.pending) return false;
    encoder.clearBuffer(this.maintenanceArena, this.stateOffsetWords * 4, SPARSE_SCENE_MAINTENANCE_STATE_WORDS.wordCount * 4);
    const run = (label: string, pipeline: GPUComputePipeline, workItems: number, workgroupSize: number) => {
      const pass = encoder.beginComputePass({ label });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, this.bindGroup);
      pass.dispatchWorkgroups(...sparseBrickDispatchDimensions(Math.ceil(workItems * 256 / workgroupSize)));
      pass.end();
    };
    run("Invalidate live scene dirty bricks", this.invalidatePipeline, this.tree.leafCapacity, 64);
    run("Prepare live scene maintenance dispatches", this.preparePipeline, 1, 1);
    encoder.copyBufferToBuffer(
      this.maintenanceArena,
      (this.stateOffsetWords + SPARSE_SCENE_MAINTENANCE_STATE_WORDS.binDispatch) * 4,
      this.maintenanceDispatch,
      0,
      3 * 3 * 4,
    );
    const runIndirect = (label: string, pipeline: GPUComputePipeline, stateWord: number) => {
      const pass = encoder.beginComputePass({ label });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, this.bindGroup);
      pass.dispatchWorkgroupsIndirect(
        this.maintenanceDispatch,
        (stateWord - SPARSE_SCENE_MAINTENANCE_STATE_WORDS.binDispatch) * 4,
      );
      pass.end();
    };
    runIndirect("Bin live scene primitives into dirty bricks", this.binPipeline, SPARSE_SCENE_MAINTENANCE_STATE_WORDS.binDispatch);
    runIndirect("Rebuild live scene dirty brick payloads", this.rebuildPipeline, SPARSE_SCENE_MAINTENANCE_STATE_WORDS.rebuildDispatch);
    runIndirect("Finalize live scene dirty bricks", this.finalizePipeline, SPARSE_SCENE_MAINTENANCE_STATE_WORDS.finalizeDispatch);
    this.pending = false;
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
