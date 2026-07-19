import type { Quaternion } from "./model";
import {
  intersectSvoPrimitive,
  type SvoFinitePrimitiveDescriptor,
  type SvoPrimitiveRay,
  type SvoPrimitiveRayHit,
} from "./svo-primitive-abi";
import { SVO_TEMPORAL_HISTORY_THRESHOLDS } from "./svo-temporal-history";
import { packMaterialOwner, SPARSE_BRICK_NO_OWNER, unpackMaterialOwner } from "./sparse-brick-octree";
import type { SvoAabb, SvoVec3 } from "./webgpu-svo-traversal";

/** Separate sidecar: the established 64-byte primitive record remains unchanged. */
export const SVO_PRIMITIVE_MOTION_STRIDE_BYTES = 128;
export const SVO_PRIMITIVE_MOTION_WORDS = SVO_PRIMITIVE_MOTION_STRIDE_BYTES / Uint32Array.BYTES_PER_ELEMENT;

export const SVO_PRIMITIVE_MOTION_FLAGS = Object.freeze({
  velocityValid: 1 << 0,
  staticMotion: 1 << 1,
  shortestArcFlip: 1 << 2,
  revisionContinuous: 1 << 3,
  generationContinuous: 1 << 4,
  teleport: 1 << 5,
} as const);

export interface SvoPrimitivePublishedTransform {
  position_m: SvoVec3;
  orientation: Quaternion;
  revision: number;
  localTopologyGeneration: number;
}

export interface SvoPrimitiveMotionInput {
  primitive: SvoFinitePrimitiveDescriptor;
  previous: SvoPrimitivePublishedTransform;
  current: SvoPrimitivePublishedTransform;
  deltaTime_s: number;
  /** Smallest local SVO cell used by temporal rejection. */
  cellSize_m: number;
}

export type SvoPrimitiveMotionContinuityReason =
  | "continuous"
  | "generation-change"
  | "revision-discontinuity"
  | "teleport";

export interface SvoPrimitiveMotionRecord {
  currentPosition_m: SvoVec3;
  previousPosition_m: SvoVec3;
  currentOrientation: Quaternion;
  /** Normalized and sign-adjusted to the shortest arc from previous to current. */
  previousOrientation: Quaternion;
  deltaTime_s: number;
  boundingRadius_m: number;
  linearVelocity_m_s: SvoVec3;
  angularVelocity_rad_s: SvoVec3;
  angularDisplacement_rad: number;
  maximumSurfaceDisplacement_m: number;
  temporalMotionLimit_m: number;
  primitiveId: number;
  materialId: number;
  ownerId: number;
  currentRevision: number;
  previousRevision: number;
  currentLocalTopologyGeneration: number;
  previousLocalTopologyGeneration: number;
  flags: number;
  velocityValid: boolean;
  continuityReason: SvoPrimitiveMotionContinuityReason;
}

export interface SvoPrimitiveMotionVelocity {
  velocity_m_s: SvoVec3;
  valid: boolean;
}

export interface SvoPrimitiveMotionRayHit extends SvoPrimitiveRayHit {
  surfaceVelocity_m_s: SvoVec3;
  motionValid: boolean;
  localTopologyGeneration: number;
}

const EPSILON = 1e-12;
const UINT32_MAX = 0xffff_ffff;

function finiteVec3Tuple(value: SvoVec3, label: string): void {
  if (value.length !== 3 || value.some((component) => !Number.isFinite(component))) throw new RangeError(`${label} must contain three finite components`);
}

function uint32(value: number, label: string): number {
  if (!Number.isInteger(value) || value < 0 || value > UINT32_MAX) throw new RangeError(`${label} must fit uint32`);
  return value >>> 0;
}

function normalizedQuaternion(value: Quaternion, label: string): Quaternion {
  if (![value.w, value.x, value.y, value.z].every(Number.isFinite)) throw new RangeError(`${label} must be finite`);
  const magnitude = Math.hypot(value.w, value.x, value.y, value.z);
  if (!(magnitude > EPSILON)) throw new RangeError(`${label} must have nonzero length`);
  return { w: value.w / magnitude, x: value.x / magnitude, y: value.y / magnitude, z: value.z / magnitude };
}

function dotQuaternion(left: Quaternion, right: Quaternion): number {
  return left.w * right.w + left.x * right.x + left.y * right.y + left.z * right.z;
}

function negateQuaternion(value: Quaternion): Quaternion {
  return { w: -value.w, x: -value.x, y: -value.y, z: -value.z };
}

function multiplyQuaternion(left: Quaternion, right: Quaternion): Quaternion {
  return {
    w: left.w * right.w - left.x * right.x - left.y * right.y - left.z * right.z,
    x: left.w * right.x + left.x * right.w + left.y * right.z - left.z * right.y,
    y: left.w * right.y - left.x * right.z + left.y * right.w + left.z * right.x,
    z: left.w * right.z + left.x * right.y - left.y * right.x + left.z * right.w,
  };
}

function inverseUnitQuaternion(value: Quaternion): Quaternion {
  return { w: value.w, x: -value.x, y: -value.y, z: -value.z };
}

function descriptorIdentity(primitive: SvoFinitePrimitiveDescriptor) {
  const primitiveId = uint32(primitive.primitiveId, "Motion primitive ID");
  if (!Number.isInteger(primitive.materialId) || primitive.materialId < 1 || primitive.materialId > 0xffff) {
    throw new RangeError("Motion material ID must be a nonzero uint16");
  }
  const ownerId = primitive.ownerId ?? SPARSE_BRICK_NO_OWNER;
  if (!Number.isInteger(ownerId) || ownerId < 0 || ownerId > 0xffff) throw new RangeError("Motion owner ID must fit uint16");
  return { primitiveId, materialId: primitive.materialId, ownerId };
}

/** Rotation-independent radius used for conservative swept residency bounds. */
export function svoPrimitiveBoundingRadius(primitive: SvoFinitePrimitiveDescriptor): number {
  if (primitive.kind === "sphere") return primitive.radius_m;
  if (primitive.kind === "box") return Math.hypot(primitive.halfExtents_m.x, primitive.halfExtents_m.y, primitive.halfExtents_m.z);
  if (primitive.kind === "capsule") return primitive.segmentHalfLength_m + primitive.radius_m;
  if (primitive.kind === "cylinder") return Math.hypot(primitive.radius_m, primitive.halfHeight_m);
  return Math.max(primitive.radii_m.x, primitive.radii_m.y, primitive.radii_m.z);
}

export function svoPrimitiveTemporalMotionLimit(cellSize_m: number): number {
  if (!Number.isFinite(cellSize_m) || cellSize_m <= 0) throw new RangeError("Primitive motion cell size must be finite and positive");
  return Math.min(
    SVO_TEMPORAL_HISTORY_THRESHOLDS.maximumRigidMotion_m,
    SVO_TEMPORAL_HISTORY_THRESHOLDS.maximumRigidMotion_cell * cellSize_m,
  );
}

function modularRevisionContinuous(previous: number, current: number): boolean {
  const difference = (current - previous) >>> 0;
  return difference === 0 || difference === 1;
}

function sameTransform(
  previousPosition: SvoVec3,
  currentPosition: SvoVec3,
  previousOrientation: Quaternion,
  currentOrientation: Quaternion,
): boolean {
  return Math.hypot(
    currentPosition[0] - previousPosition[0], currentPosition[1] - previousPosition[1], currentPosition[2] - previousPosition[2],
  ) <= EPSILON && 1 - Math.abs(dotQuaternion(previousOrientation, currentOrientation)) <= EPSILON;
}

/** Validate, shortest-arc canonicalize, derive velocities, and classify continuity. */
export function createSvoPrimitiveMotionRecord(input: SvoPrimitiveMotionInput): SvoPrimitiveMotionRecord {
  const identity = descriptorIdentity(input.primitive);
  finiteVec3Tuple(input.previous.position_m, "Previous primitive position");
  finiteVec3Tuple(input.current.position_m, "Current primitive position");
  if (!Number.isFinite(input.deltaTime_s) || input.deltaTime_s <= 0) throw new RangeError("Primitive motion delta time must be finite and positive");
  let previousOrientation = normalizedQuaternion(input.previous.orientation, "Previous primitive orientation");
  const currentOrientation = normalizedQuaternion(input.current.orientation, "Current primitive orientation");
  const shortestArcFlip = dotQuaternion(previousOrientation, currentOrientation) < 0;
  if (shortestArcFlip) previousOrientation = negateQuaternion(previousOrientation);
  const previousRevision = uint32(input.previous.revision, "Previous primitive revision");
  const currentRevision = uint32(input.current.revision, "Current primitive revision");
  const previousLocalTopologyGeneration = uint32(input.previous.localTopologyGeneration, "Previous local topology generation");
  const currentLocalTopologyGeneration = uint32(input.current.localTopologyGeneration, "Current local topology generation");
  const revisionContinuous = modularRevisionContinuous(previousRevision, currentRevision)
    && (currentRevision !== previousRevision || sameTransform(input.previous.position_m, input.current.position_m, previousOrientation, currentOrientation));
  const generationContinuous = previousLocalTopologyGeneration === currentLocalTopologyGeneration;
  const deltaPosition: SvoVec3 = [
    input.current.position_m[0] - input.previous.position_m[0],
    input.current.position_m[1] - input.previous.position_m[1],
    input.current.position_m[2] - input.previous.position_m[2],
  ];
  const translation = Math.hypot(...deltaPosition);
  const deltaRotation = normalizedQuaternion(
    multiplyQuaternion(currentOrientation, inverseUnitQuaternion(previousOrientation)),
    "Primitive delta orientation",
  );
  const vectorMagnitude = Math.hypot(deltaRotation.x, deltaRotation.y, deltaRotation.z);
  const angularDisplacement_rad = 2 * Math.atan2(vectorMagnitude, Math.max(0, deltaRotation.w));
  const angularAxis: SvoVec3 = vectorMagnitude > EPSILON
    ? [deltaRotation.x / vectorMagnitude, deltaRotation.y / vectorMagnitude, deltaRotation.z / vectorMagnitude]
    : [0, 0, 0];
  const rawLinearVelocity = deltaPosition.map((component) => component / input.deltaTime_s) as [number, number, number];
  const rawAngularVelocity = angularAxis.map((component) => component * angularDisplacement_rad / input.deltaTime_s) as [number, number, number];
  const boundingRadius_m = svoPrimitiveBoundingRadius(input.primitive);
  if (!Number.isFinite(boundingRadius_m) || boundingRadius_m <= 0) throw new RangeError("Primitive bounding radius must be finite and positive");
  const maximumSurfaceDisplacement_m = translation + 2 * boundingRadius_m * Math.sin(0.5 * angularDisplacement_rad);
  const temporalMotionLimit_m = svoPrimitiveTemporalMotionLimit(input.cellSize_m);
  const teleport = maximumSurfaceDisplacement_m > temporalMotionLimit_m;
  const velocityValid = revisionContinuous && generationContinuous && !teleport;
  const staticMotion = sameTransform(input.previous.position_m, input.current.position_m, previousOrientation, currentOrientation);
  const continuityReason: SvoPrimitiveMotionContinuityReason = !generationContinuous ? "generation-change"
    : !revisionContinuous ? "revision-discontinuity"
      : teleport ? "teleport" : "continuous";
  const linearVelocity_m_s: SvoVec3 = velocityValid ? rawLinearVelocity : [0, 0, 0];
  const angularVelocity_rad_s: SvoVec3 = velocityValid ? rawAngularVelocity : [0, 0, 0];
  const flags = (velocityValid ? SVO_PRIMITIVE_MOTION_FLAGS.velocityValid : 0)
    | (staticMotion ? SVO_PRIMITIVE_MOTION_FLAGS.staticMotion : 0)
    | (shortestArcFlip ? SVO_PRIMITIVE_MOTION_FLAGS.shortestArcFlip : 0)
    | (revisionContinuous ? SVO_PRIMITIVE_MOTION_FLAGS.revisionContinuous : 0)
    | (generationContinuous ? SVO_PRIMITIVE_MOTION_FLAGS.generationContinuous : 0)
    | (teleport ? SVO_PRIMITIVE_MOTION_FLAGS.teleport : 0);
  return {
    currentPosition_m: [...input.current.position_m], previousPosition_m: [...input.previous.position_m],
    currentOrientation, previousOrientation, deltaTime_s: input.deltaTime_s, boundingRadius_m,
    linearVelocity_m_s, angularVelocity_rad_s, angularDisplacement_rad,
    maximumSurfaceDisplacement_m, temporalMotionLimit_m,
    ...identity, currentRevision, previousRevision,
    currentLocalTopologyGeneration, previousLocalTopologyGeneration,
    flags, velocityValid, continuityReason,
  };
}

/** Capsule of the two centres enlarged by the rotation-independent radius. */
export function svoPrimitiveSweptBounds(record: SvoPrimitiveMotionRecord): SvoAabb {
  const radius = record.boundingRadius_m;
  return {
    minimum: record.currentPosition_m.map((value, axis) => Math.min(value, record.previousPosition_m[axis]) - radius) as [number, number, number],
    maximum: record.currentPosition_m.map((value, axis) => Math.max(value, record.previousPosition_m[axis]) + radius) as [number, number, number],
  };
}

/** Translation plus omega cross radius at the exact current surface point. */
export function svoPrimitiveSurfaceVelocity(
  record: SvoPrimitiveMotionRecord,
  worldSurfacePosition_m: SvoVec3,
): SvoPrimitiveMotionVelocity {
  finiteVec3Tuple(worldSurfacePosition_m, "Primitive surface position");
  if (!record.velocityValid) return { velocity_m_s: [0, 0, 0], valid: false };
  const radius: SvoVec3 = worldSurfacePosition_m.map((value, axis) => value - record.currentPosition_m[axis]) as [number, number, number];
  const angular = record.angularVelocity_rad_s;
  const rotational: SvoVec3 = [
    angular[1] * radius[2] - angular[2] * radius[1],
    angular[2] * radius[0] - angular[0] * radius[2],
    angular[0] * radius[1] - angular[1] * radius[0],
  ];
  return {
    velocity_m_s: record.linearVelocity_m_s.map((component, axis) => component + rotational[axis]) as [number, number, number],
    valid: true,
  };
}

function currentDescriptor(
  primitive: SvoFinitePrimitiveDescriptor,
  record: SvoPrimitiveMotionRecord,
): SvoFinitePrimitiveDescriptor {
  const identity = descriptorIdentity(primitive);
  if (identity.primitiveId !== record.primitiveId || identity.materialId !== record.materialId || identity.ownerId !== record.ownerId) {
    throw new RangeError("Primitive motion identity does not match the exact primitive record");
  }
  const transform = {
    center_m: { x: record.currentPosition_m[0], y: record.currentPosition_m[1], z: record.currentPosition_m[2] },
    orientation: record.currentOrientation,
  };
  if (primitive.kind === "sphere") return { ...primitive, center_m: transform.center_m };
  return { ...primitive, ...transform } as SvoFinitePrimitiveDescriptor;
}

/** Exact current hit plus motion at that exact surface point. */
export function intersectSvoPrimitiveMotion(
  primitive: SvoFinitePrimitiveDescriptor,
  record: SvoPrimitiveMotionRecord,
  ray: SvoPrimitiveRay,
): SvoPrimitiveMotionRayHit | null {
  const hit = intersectSvoPrimitive(currentDescriptor(primitive, record), ray);
  if (!hit) return null;
  const velocity = svoPrimitiveSurfaceVelocity(record, [hit.position_m.x, hit.position_m.y, hit.position_m.z]);
  return {
    ...hit,
    surfaceVelocity_m_s: velocity.velocity_m_s,
    motionValid: velocity.valid,
    localTopologyGeneration: record.currentLocalTopologyGeneration,
  };
}

/** Eight 16-byte lanes, directly indexable beside existing primitive records. */
export function packSvoPrimitiveMotionRecords(inputs: readonly SvoPrimitiveMotionInput[]): Uint32Array<ArrayBuffer> {
  const buffer = new ArrayBuffer(inputs.length * SVO_PRIMITIVE_MOTION_STRIDE_BYTES);
  const words = new Uint32Array(buffer);
  const floats = new Float32Array(buffer);
  inputs.forEach((input, index) => {
    const record = createSvoPrimitiveMotionRecord(input);
    const base = index * SVO_PRIMITIVE_MOTION_WORDS;
    floats.set([...record.currentPosition_m, record.deltaTime_s], base);
    floats.set([...record.previousPosition_m, record.boundingRadius_m], base + 4);
    floats.set([record.currentOrientation.x, record.currentOrientation.y, record.currentOrientation.z, record.currentOrientation.w], base + 8);
    floats.set([record.previousOrientation.x, record.previousOrientation.y, record.previousOrientation.z, record.previousOrientation.w], base + 12);
    floats.set([...record.linearVelocity_m_s, record.maximumSurfaceDisplacement_m], base + 16);
    floats.set([...record.angularVelocity_rad_s, record.angularDisplacement_rad], base + 20);
    words.set([record.primitiveId, packMaterialOwner(record.materialId, record.ownerId), record.currentRevision, record.previousRevision], base + 24);
    words.set([record.currentLocalTopologyGeneration, record.previousLocalTopologyGeneration, record.flags, 0], base + 28);
    floats[base + 31] = record.temporalMotionLimit_m;
  });
  return words;
}

/** Unpack capture/readback data; continuity reason follows the published flags. */
export function unpackSvoPrimitiveMotionRecords(packed: Uint32Array): SvoPrimitiveMotionRecord[] {
  if (packed.length % SVO_PRIMITIVE_MOTION_WORDS !== 0) throw new RangeError("Packed primitive motion data has a partial record");
  const words = new Uint32Array(packed);
  const floats = new Float32Array(words.buffer);
  const result: SvoPrimitiveMotionRecord[] = [];
  for (let base = 0; base < words.length; base += SVO_PRIMITIVE_MOTION_WORDS) {
    const identity = unpackMaterialOwner(words[base + 25]);
    const flags = words[base + 30];
    const velocityValid = (flags & SVO_PRIMITIVE_MOTION_FLAGS.velocityValid) !== 0;
    const generationContinuous = (flags & SVO_PRIMITIVE_MOTION_FLAGS.generationContinuous) !== 0;
    const revisionContinuous = (flags & SVO_PRIMITIVE_MOTION_FLAGS.revisionContinuous) !== 0;
    const teleport = (flags & SVO_PRIMITIVE_MOTION_FLAGS.teleport) !== 0;
    const continuityReason: SvoPrimitiveMotionContinuityReason = !generationContinuous ? "generation-change"
      : !revisionContinuous ? "revision-discontinuity" : teleport ? "teleport" : "continuous";
    const boundingRadius_m = floats[base + 7];
    const angularDisplacement_rad = floats[base + 23];
    const maximumSurfaceDisplacement_m = floats[base + 19];
    result.push({
      currentPosition_m: [floats[base], floats[base + 1], floats[base + 2]],
      deltaTime_s: floats[base + 3],
      previousPosition_m: [floats[base + 4], floats[base + 5], floats[base + 6]],
      boundingRadius_m,
      currentOrientation: { x: floats[base + 8], y: floats[base + 9], z: floats[base + 10], w: floats[base + 11] },
      previousOrientation: { x: floats[base + 12], y: floats[base + 13], z: floats[base + 14], w: floats[base + 15] },
      linearVelocity_m_s: [floats[base + 16], floats[base + 17], floats[base + 18]],
      maximumSurfaceDisplacement_m,
      angularVelocity_rad_s: [floats[base + 20], floats[base + 21], floats[base + 22]],
      angularDisplacement_rad,
      temporalMotionLimit_m: floats[base + 31],
      primitiveId: words[base + 24], materialId: identity.materialId, ownerId: identity.ownerId,
      currentRevision: words[base + 26], previousRevision: words[base + 27],
      currentLocalTopologyGeneration: words[base + 28], previousLocalTopologyGeneration: words[base + 29],
      flags, velocityValid, continuityReason,
    });
  }
  return result;
}

/** Binding-free sidecar, surface velocity, and conservative swept-bounds helpers. */
export const svoPrimitiveMotionWGSL = /* wgsl */ `
const SVO_PRIMITIVE_MOTION_VALID:u32=1u;const SVO_PRIMITIVE_MOTION_STATIC:u32=2u;const SVO_PRIMITIVE_MOTION_SHORTEST_FLIP:u32=4u;
const SVO_PRIMITIVE_MOTION_REVISION_CONTINUOUS:u32=8u;const SVO_PRIMITIVE_MOTION_GENERATION_CONTINUOUS:u32=16u;const SVO_PRIMITIVE_MOTION_TELEPORT:u32=32u;
struct SvoPrimitiveMotionRecord{currentPositionDt:vec4f,previousPositionRadius:vec4f,currentOrientation:vec4f,previousOrientation:vec4f,linearVelocityDisplacement:vec4f,angularVelocityAngle:vec4f,identityRevision:vec4u,publication:vec4u}
struct SvoPrimitiveMotionVelocity{velocity_m_s:vec3f,valid:u32}
fn svoPrimitiveMotionQuaternionNormalize(value:vec4f)->vec4f{let magnitude=length(value);return select(vec4f(0.0,0.0,0.0,1.0),value/magnitude,magnitude>1e-12);}
fn svoPrimitiveMotionShortestPrevious(previousIn:vec4f,currentIn:vec4f)->vec4f{let previous=svoPrimitiveMotionQuaternionNormalize(previousIn);let current=svoPrimitiveMotionQuaternionNormalize(currentIn);return select(previous,-previous,dot(previous,current)<0.0);}
fn svoPrimitiveMotionVelocityAt(record:SvoPrimitiveMotionRecord,worldSurfacePosition_m:vec3f)->SvoPrimitiveMotionVelocity{if((record.publication.z&SVO_PRIMITIVE_MOTION_VALID)==0u){return SvoPrimitiveMotionVelocity(vec3f(0.0),0u);}let radius=worldSurfacePosition_m-record.currentPositionDt.xyz;return SvoPrimitiveMotionVelocity(record.linearVelocityDisplacement.xyz+cross(record.angularVelocityAngle.xyz,radius),1u);}
fn svoPrimitiveMotionSweptBounds(record:SvoPrimitiveMotionRecord)->mat2x3f{let radius=vec3f(record.previousPositionRadius.w);return mat2x3f(min(record.currentPositionDt.xyz,record.previousPositionRadius.xyz)-radius,max(record.currentPositionDt.xyz,record.previousPositionRadius.xyz)+radius);}
fn svoPrimitiveMotionMaterialId(record:SvoPrimitiveMotionRecord)->u32{return record.identityRevision.y&0xffffu;}fn svoPrimitiveMotionOwnerId(record:SvoPrimitiveMotionRecord)->u32{return record.identityRevision.y>>16u;}fn svoPrimitiveMotionGeneration(record:SvoPrimitiveMotionRecord)->u32{return record.publication.x;}
`;
