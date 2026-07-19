import type { Quaternion, RigidBodyDescription, Vec3 } from "./model";
import { packMaterialOwner, SPARSE_BRICK_NO_OWNER, unpackMaterialOwner } from "./sparse-brick-octree";
import { terrainHeightAt, terrainNormalAt, type TerrainDescription } from "./terrain";
import { materialIdForRigidShape } from "./voxel-scene";

/** Four 16-byte lanes, directly usable as a WebGPU storage-buffer array. */
export const SVO_PRIMITIVE_RECORD_STRIDE_BYTES = 64;
export const SVO_PRIMITIVE_RECORD_WORDS = SVO_PRIMITIVE_RECORD_STRIDE_BYTES / Uint32Array.BYTES_PER_ELEMENT;
export const SVO_PRIMITIVE_INVALID_REFERENCE = 0xffff_ffff;
/** Fixed bisection ceiling shared by the CPU oracle and f32 WGSL closest-point solve. */
export const SVO_ELLIPSOID_CLOSEST_POINT_ITERATIONS = 64;

/** Stable on-GPU primitive tags. Zero remains reserved for an invalid/empty record. */
export const SVO_PRIMITIVE_KINDS = Object.freeze({
  sphere: 1,
  box: 2,
  capsule: 3,
  cylinder: 4,
  ellipsoid: 5,
  terrainHeightfield: 6,
} as const);

export const SVO_PRIMITIVE_FLAGS = Object.freeze({
  exactDistance: 1,
  hardFeatures: 2,
  externalTerrain: 4,
} as const);

/** Stable feature IDs returned beside shading normals. */
export const SVO_PRIMITIVE_FEATURES = Object.freeze({
  smooth: 0,
  boxFaceX: 1,
  boxFaceY: 2,
  boxFaceZ: 3,
  cylinderSide: 4,
  cylinderCap: 5,
  terrain: 6,
} as const);

interface SvoPrimitiveIdentity {
  /** Stable scene-local primitive identity. */
  primitiveId: number;
  /** Stable material-table index. Zero is reserved for empty space. */
  materialId: number;
  /** Stable scene owner, or 0xffff when the primitive has no selectable owner. */
  ownerId?: number;
}

interface SvoLocatedPrimitive extends SvoPrimitiveIdentity {
  center_m: Vec3;
}

interface SvoOrientedPrimitive extends SvoLocatedPrimitive {
  /** Repository quaternion order is wxyz; packed GPU order is xyzw. */
  orientation?: Quaternion;
}

export interface SvoSpherePrimitive extends SvoLocatedPrimitive {
  kind: "sphere";
  radius_m: number;
}

export interface SvoBoxPrimitive extends SvoOrientedPrimitive {
  kind: "box";
  halfExtents_m: Vec3;
}

export interface SvoCapsulePrimitive extends SvoOrientedPrimitive {
  kind: "capsule";
  radius_m: number;
  segmentHalfLength_m: number;
}

export interface SvoCylinderPrimitive extends SvoOrientedPrimitive {
  kind: "cylinder";
  radius_m: number;
  halfHeight_m: number;
}

export interface SvoEllipsoidPrimitive extends SvoOrientedPrimitive {
  kind: "ellipsoid";
  radii_m: Vec3;
}

/**
 * A terrain record references the shared scene heightfield table. Variable-size
 * terrain features deliberately do not live in every primitive record.
 */
export interface SvoTerrainHeightfieldPrimitive extends SvoPrimitiveIdentity {
  kind: "terrain-heightfield";
  terrainReference: number;
  normalEpsilon_m?: number;
}

export type SvoPrimitiveDescriptor =
  | SvoSpherePrimitive
  | SvoBoxPrimitive
  | SvoCapsulePrimitive
  | SvoCylinderPrimitive
  | SvoEllipsoidPrimitive
  | SvoTerrainHeightfieldPrimitive;

export interface SvoPrimitiveSample {
  signedDistance_m: number;
  normal: Vec3 | null;
  featureId: number;
}

export type SvoFinitePrimitiveDescriptor = Exclude<SvoPrimitiveDescriptor, SvoTerrainHeightfieldPrimitive>;

/** World-space ray whose interval is measured in metres along its normalized direction. */
export interface SvoPrimitiveRay {
  origin_m: Vec3;
  direction: Vec3;
  tMin_m?: number;
  tMax_m?: number;
}

export interface SvoPrimitiveRayHit {
  /** Physical distance from origin_m, independent of direction magnitude. */
  t_m: number;
  position_m: Vec3;
  normal: Vec3;
  /** Smooth primitives interpolate their analytic gradient; hard-feature primitives select one authored feature. */
  normalPolicy: "smooth" | "hard-feature";
  featureId: number;
  primitiveKind: SvoFinitePrimitiveDescriptor["kind"];
  primitiveId: number;
  materialId: number;
  ownerId: number;
}

export type SvoTerrainResolver = (terrainReference: number) => TerrainDescription | undefined;

const NORMAL_EPSILON = 1e-12;

function finiteVec3(value: Vec3, label: string): void {
  if (![value.x, value.y, value.z].every(Number.isFinite)) throw new RangeError(`${label} must be finite`);
}

function positive(value: number, label: string): void {
  if (!Number.isFinite(value) || value <= 0) throw new RangeError(`${label} must be positive`);
}

function nonNegative(value: number, label: string): void {
  if (!Number.isFinite(value) || value < 0) throw new RangeError(`${label} must be non-negative`);
}

function uint32(value: number, label: string): number {
  if (!Number.isInteger(value) || value < 0 || value > 0xffff_ffff) throw new RangeError(`${label} must fit uint32`);
  return value >>> 0;
}

function normalizedOrientation(value: Quaternion | undefined): Quaternion {
  if (value === undefined) return { w: 1, x: 0, y: 0, z: 0 };
  if (![value.w, value.x, value.y, value.z].every(Number.isFinite)) throw new RangeError("Primitive orientation must be finite");
  const magnitude = Math.hypot(value.w, value.x, value.y, value.z);
  if (!(magnitude > NORMAL_EPSILON)) throw new RangeError("Primitive orientation must have nonzero length");
  return { w: value.w / magnitude, x: value.x / magnitude, y: value.y / magnitude, z: value.z / magnitude };
}

function validateIdentity(descriptor: SvoPrimitiveDescriptor): void {
  uint32(descriptor.primitiveId, "Primitive ID");
  if (!Number.isInteger(descriptor.materialId) || descriptor.materialId < 1 || descriptor.materialId > 0xffff) {
    throw new RangeError("Primitive material ID must be a nonzero uint16");
  }
  const owner = descriptor.ownerId ?? SPARSE_BRICK_NO_OWNER;
  if (!Number.isInteger(owner) || owner < 0 || owner > 0xffff) throw new RangeError("Primitive owner ID must fit uint16");
}

function dimensions(descriptor: SvoPrimitiveDescriptor): Vec3 {
  if (descriptor.kind === "sphere") {
    positive(descriptor.radius_m, "Sphere radius");
    return { x: descriptor.radius_m, y: 0, z: 0 };
  }
  if (descriptor.kind === "box") {
    finiteVec3(descriptor.halfExtents_m, "Box half extents");
    positive(descriptor.halfExtents_m.x, "Box X half extent");
    positive(descriptor.halfExtents_m.y, "Box Y half extent");
    positive(descriptor.halfExtents_m.z, "Box Z half extent");
    return { ...descriptor.halfExtents_m };
  }
  if (descriptor.kind === "capsule") {
    positive(descriptor.radius_m, "Capsule radius");
    nonNegative(descriptor.segmentHalfLength_m, "Capsule segment half length");
    return { x: descriptor.radius_m, y: descriptor.segmentHalfLength_m, z: 0 };
  }
  if (descriptor.kind === "cylinder") {
    positive(descriptor.radius_m, "Cylinder radius");
    positive(descriptor.halfHeight_m, "Cylinder half height");
    return { x: descriptor.radius_m, y: descriptor.halfHeight_m, z: 0 };
  }
  if (descriptor.kind === "ellipsoid") {
    finiteVec3(descriptor.radii_m, "Ellipsoid radii");
    positive(descriptor.radii_m.x, "Ellipsoid X radius");
    positive(descriptor.radii_m.y, "Ellipsoid Y radius");
    positive(descriptor.radii_m.z, "Ellipsoid Z radius");
    return { ...descriptor.radii_m };
  }
  uint32(descriptor.terrainReference, "Terrain reference");
  if (descriptor.terrainReference === SVO_PRIMITIVE_INVALID_REFERENCE) throw new RangeError("Terrain reference may not use the invalid sentinel");
  const epsilon = descriptor.normalEpsilon_m ?? 0.02;
  positive(epsilon, "Terrain normal epsilon");
  return { x: epsilon, y: 0, z: 0 };
}

function kindCode(kind: SvoPrimitiveDescriptor["kind"]): number {
  if (kind === "sphere") return SVO_PRIMITIVE_KINDS.sphere;
  if (kind === "box") return SVO_PRIMITIVE_KINDS.box;
  if (kind === "capsule") return SVO_PRIMITIVE_KINDS.capsule;
  if (kind === "cylinder") return SVO_PRIMITIVE_KINDS.cylinder;
  if (kind === "ellipsoid") return SVO_PRIMITIVE_KINDS.ellipsoid;
  return SVO_PRIMITIVE_KINDS.terrainHeightfield;
}

function primitiveFlags(kind: SvoPrimitiveDescriptor["kind"]): number {
  if (kind === "box" || kind === "cylinder") return SVO_PRIMITIVE_FLAGS.exactDistance | SVO_PRIMITIVE_FLAGS.hardFeatures;
  if (kind === "sphere" || kind === "capsule" || kind === "ellipsoid") return SVO_PRIMITIVE_FLAGS.exactDistance;
  if (kind === "terrain-heightfield") return SVO_PRIMITIVE_FLAGS.externalTerrain;
  return 0;
}

function descriptorCenter(descriptor: SvoPrimitiveDescriptor): Vec3 {
  if (descriptor.kind === "terrain-heightfield") return { x: 0, y: 0, z: 0 };
  finiteVec3(descriptor.center_m, "Primitive centre");
  return descriptor.center_m;
}

function descriptorOrientation(descriptor: SvoPrimitiveDescriptor): Quaternion {
  if (descriptor.kind === "sphere" || descriptor.kind === "terrain-heightfield") return normalizedOrientation(undefined);
  return normalizedOrientation(descriptor.orientation);
}

/** Validate and canonicalize a descriptor before hashing, packing, or upload. */
export function canonicalSvoPrimitive(descriptor: SvoPrimitiveDescriptor): SvoPrimitiveDescriptor {
  validateIdentity(descriptor);
  const d = dimensions(descriptor);
  const ownerId = descriptor.ownerId ?? SPARSE_BRICK_NO_OWNER;
  if (descriptor.kind === "sphere") return { ...descriptor, center_m: { ...descriptorCenter(descriptor) }, ownerId, radius_m: d.x };
  if (descriptor.kind === "box") return { ...descriptor, center_m: { ...descriptorCenter(descriptor) }, ownerId, orientation: descriptorOrientation(descriptor), halfExtents_m: d };
  if (descriptor.kind === "capsule") return { ...descriptor, center_m: { ...descriptorCenter(descriptor) }, ownerId, orientation: descriptorOrientation(descriptor), radius_m: d.x, segmentHalfLength_m: d.y };
  if (descriptor.kind === "cylinder") return { ...descriptor, center_m: { ...descriptorCenter(descriptor) }, ownerId, orientation: descriptorOrientation(descriptor), radius_m: d.x, halfHeight_m: d.y };
  if (descriptor.kind === "ellipsoid") return { ...descriptor, center_m: { ...descriptorCenter(descriptor) }, ownerId, orientation: descriptorOrientation(descriptor), radii_m: d };
  return { ...descriptor, ownerId, normalEpsilon_m: d.x };
}

/**
 * Pack `{center.xyz, kind}`, `{dimensions.xyz, material|owner}`, quaternion
 * `xyzw`, then `{primitive, terrain-reference, flags, reserved}`.
 */
export function packSvoPrimitiveRecords(descriptors: readonly SvoPrimitiveDescriptor[]): Uint32Array<ArrayBuffer> {
  const buffer = new ArrayBuffer(descriptors.length * SVO_PRIMITIVE_RECORD_STRIDE_BYTES);
  const words = new Uint32Array(buffer);
  const floats = new Float32Array(buffer);
  descriptors.forEach((input, index) => {
    const descriptor = canonicalSvoPrimitive(input);
    const base = index * SVO_PRIMITIVE_RECORD_WORDS;
    const center = descriptorCenter(descriptor);
    const d = dimensions(descriptor);
    const orientation = descriptorOrientation(descriptor);
    floats.set([center.x, center.y, center.z], base);
    words[base + 3] = kindCode(descriptor.kind);
    floats.set([d.x, d.y, d.z], base + 4);
    words[base + 7] = packMaterialOwner(descriptor.materialId, descriptor.ownerId);
    floats.set([orientation.x, orientation.y, orientation.z, orientation.w], base + 8);
    words.set([
      descriptor.primitiveId >>> 0,
      descriptor.kind === "terrain-heightfield" ? descriptor.terrainReference >>> 0 : SVO_PRIMITIVE_INVALID_REFERENCE,
      primitiveFlags(descriptor.kind),
      0,
    ], base + 12);
  });
  return words;
}

function descriptorFromRecord(words: Uint32Array, floats: Float32Array, base: number): SvoPrimitiveDescriptor {
  const center_m = { x: floats[base], y: floats[base + 1], z: floats[base + 2] };
  const d = { x: floats[base + 4], y: floats[base + 5], z: floats[base + 6] };
  const orientation = { x: floats[base + 8], y: floats[base + 9], z: floats[base + 10], w: floats[base + 11] };
  const { materialId, ownerId } = unpackMaterialOwner(words[base + 7]);
  const identity = { primitiveId: words[base + 12], materialId, ownerId };
  const kind = words[base + 3];
  if (kind === SVO_PRIMITIVE_KINDS.sphere) return { ...identity, kind: "sphere", center_m, radius_m: d.x };
  if (kind === SVO_PRIMITIVE_KINDS.box) return { ...identity, kind: "box", center_m, orientation, halfExtents_m: d };
  if (kind === SVO_PRIMITIVE_KINDS.capsule) return { ...identity, kind: "capsule", center_m, orientation, radius_m: d.x, segmentHalfLength_m: d.y };
  if (kind === SVO_PRIMITIVE_KINDS.cylinder) return { ...identity, kind: "cylinder", center_m, orientation, radius_m: d.x, halfHeight_m: d.y };
  if (kind === SVO_PRIMITIVE_KINDS.ellipsoid) return { ...identity, kind: "ellipsoid", center_m, orientation, radii_m: d };
  if (kind === SVO_PRIMITIVE_KINDS.terrainHeightfield) {
    return { ...identity, kind: "terrain-heightfield", terrainReference: words[base + 13], normalEpsilon_m: d.x };
  }
  throw new RangeError(`Unknown SVO primitive kind ${kind}`);
}

/** Deterministic CPU unpack mirror used by tests, diagnostics, and capture tools. */
export function unpackSvoPrimitiveRecords(packed: Uint32Array): SvoPrimitiveDescriptor[] {
  if (packed.length % SVO_PRIMITIVE_RECORD_WORDS !== 0) throw new RangeError("Packed SVO primitive data has a partial record");
  const copied = new Uint32Array(packed);
  const floats = new Float32Array(copied.buffer);
  const result: SvoPrimitiveDescriptor[] = [];
  for (let base = 0; base < copied.length; base += SVO_PRIMITIVE_RECORD_WORDS) {
    result.push(canonicalSvoPrimitive(descriptorFromRecord(copied, floats, base)));
  }
  return result;
}

/** Map the repository's existing rigid dimension semantics into the render ABI. */
export function svoPrimitiveForRigidBody(
  body: Pick<RigidBodyDescription, "shape" | "dimensions_m" | "position_m" | "orientation">,
  primitiveId: number,
  ownerId: number,
  materialId = materialIdForRigidShape(body.shape),
): Exclude<SvoPrimitiveDescriptor, SvoEllipsoidPrimitive | SvoTerrainHeightfieldPrimitive> {
  const identity = { primitiveId, materialId, ownerId, center_m: { ...body.position_m } };
  if (body.shape === "sphere") return canonicalSvoPrimitive({ ...identity, kind: "sphere", radius_m: body.dimensions_m.x }) as SvoSpherePrimitive;
  if (body.shape === "box") return canonicalSvoPrimitive({
    ...identity, kind: "box", orientation: { ...body.orientation },
    halfExtents_m: { x: body.dimensions_m.x / 2, y: body.dimensions_m.y / 2, z: body.dimensions_m.z / 2 },
  }) as SvoBoxPrimitive;
  if (body.shape === "capsule") return canonicalSvoPrimitive({
    ...identity, kind: "capsule", orientation: { ...body.orientation },
    radius_m: body.dimensions_m.x, segmentHalfLength_m: body.dimensions_m.y / 2,
  }) as SvoCapsulePrimitive;
  return canonicalSvoPrimitive({
    ...identity, kind: "cylinder", orientation: { ...body.orientation },
    radius_m: body.dimensions_m.x, halfHeight_m: body.dimensions_m.y / 2,
  }) as SvoCylinderPrimitive;
}

function rotate(q: Quaternion, v: Vec3): Vec3 {
  const tx = 2 * (q.y * v.z - q.z * v.y);
  const ty = 2 * (q.z * v.x - q.x * v.z);
  const tz = 2 * (q.x * v.y - q.y * v.x);
  return {
    x: v.x + q.w * tx + (q.y * tz - q.z * ty),
    y: v.y + q.w * ty + (q.z * tx - q.x * tz),
    z: v.z + q.w * tz + (q.x * ty - q.y * tx),
  };
}

function inverseRotate(q: Quaternion, v: Vec3): Vec3 {
  return rotate({ w: q.w, x: -q.x, y: -q.y, z: -q.z }, v);
}

function normalize(v: Vec3): Vec3 | null {
  const length = Math.hypot(v.x, v.y, v.z);
  return length > NORMAL_EPSILON ? { x: v.x / length, y: v.y / length, z: v.z / length } : null;
}

function localPoint(descriptor: Exclude<SvoPrimitiveDescriptor, SvoTerrainHeightfieldPrimitive>, worldPoint_m: Vec3): { point: Vec3; orientation: Quaternion } {
  finiteVec3(worldPoint_m, "Primitive query point");
  const orientation = descriptorOrientation(descriptor);
  return {
    point: inverseRotate(orientation, {
      x: worldPoint_m.x - descriptor.center_m.x,
      y: worldPoint_m.y - descriptor.center_m.y,
      z: worldPoint_m.z - descriptor.center_m.z,
    }),
    orientation,
  };
}

function worldNormal(orientation: Quaternion, normal: Vec3 | null): Vec3 | null {
  return normal ? normalize(rotate(orientation, normal)) : null;
}

interface CanonicalPrimitiveRay {
  origin_m: Vec3;
  direction: Vec3;
  tMin_m: number;
  tMax_m: number;
}

interface LocalPrimitiveRayHit {
  t_m: number;
  normal: Vec3;
  featureId: number;
}

function canonicalPrimitiveRay(ray: SvoPrimitiveRay): CanonicalPrimitiveRay {
  finiteVec3(ray.origin_m, "Primitive ray origin");
  finiteVec3(ray.direction, "Primitive ray direction");
  const direction = normalize(ray.direction);
  if (!direction) throw new RangeError("Primitive ray direction must be non-zero");
  const tMin_m = ray.tMin_m ?? 0;
  const tMax_m = ray.tMax_m ?? Number.POSITIVE_INFINITY;
  if (!Number.isFinite(tMin_m) || tMin_m < 0) throw new RangeError("Primitive ray minimum must be a non-negative finite metre distance");
  if (!(Number.isFinite(tMax_m) || tMax_m === Number.POSITIVE_INFINITY) || tMax_m < tMin_m) {
    throw new RangeError("Primitive ray maximum must be at least its minimum");
  }
  return { origin_m: { ...ray.origin_m }, direction, tMin_m, tMax_m };
}

/** Sorted roots of a*t^2 + 2*b*t + c, retaining exact tangent contact. */
function quadraticRoots(a: number, b: number, c: number): readonly number[] {
  if (!(a > NORMAL_EPSILON)) return [];
  const discriminant = b * b - a * c;
  const tolerance = 1e-12 * Math.max(1, Math.abs(b * b), Math.abs(a * c));
  if (discriminant < -tolerance) return [];
  const root = Math.sqrt(Math.max(0, discriminant));
  if (root === 0) return [-b / a];
  // This form avoids losing the near root when b and sqrt(discriminant) are close.
  const q = -b - Math.sign(b || 1) * root;
  const first = q / a;
  const second = c / q;
  return first <= second ? [first, second] : [second, first];
}

function inPrimitiveRayRange(t_m: number, ray: CanonicalPrimitiveRay): boolean {
  const tolerance = 1e-10 * Math.max(1, Math.abs(t_m), Math.abs(ray.tMin_m), Number.isFinite(ray.tMax_m) ? Math.abs(ray.tMax_m) : 1);
  return Number.isFinite(t_m) && t_m >= ray.tMin_m - tolerance && t_m <= ray.tMax_m + tolerance;
}

function nearestLocalHit(candidates: readonly LocalPrimitiveRayHit[], ray: CanonicalPrimitiveRay): LocalPrimitiveRayHit | null {
  let nearest: LocalPrimitiveRayHit | null = null;
  for (const candidate of candidates) {
    if (!inPrimitiveRayRange(candidate.t_m, ray)) continue;
    const t_m = Math.max(ray.tMin_m, candidate.t_m);
    if (!nearest || t_m < nearest.t_m) nearest = { ...candidate, t_m };
  }
  return nearest;
}

function localPrimitiveRay(
  descriptor: SvoFinitePrimitiveDescriptor,
  ray: CanonicalPrimitiveRay,
): { origin: Vec3; direction: Vec3; orientation: Quaternion } {
  const orientation = descriptorOrientation(descriptor);
  const offset = {
    x: ray.origin_m.x - descriptor.center_m.x,
    y: ray.origin_m.y - descriptor.center_m.y,
    z: ray.origin_m.z - descriptor.center_m.z,
  };
  return { origin: inverseRotate(orientation, offset), direction: inverseRotate(orientation, ray.direction), orientation };
}

function sphereRayCandidates(origin: Vec3, direction: Vec3, radius_m: number, centerY_m = 0): readonly number[] {
  const offset = { x: origin.x, y: origin.y - centerY_m, z: origin.z };
  return quadraticRoots(
    direction.x ** 2 + direction.y ** 2 + direction.z ** 2,
    offset.x * direction.x + offset.y * direction.y + offset.z * direction.z,
    offset.x ** 2 + offset.y ** 2 + offset.z ** 2 - radius_m ** 2,
  );
}

function intersectSphereLocal(
  descriptor: SvoSpherePrimitive,
  origin: Vec3,
  direction: Vec3,
  ray: CanonicalPrimitiveRay,
): LocalPrimitiveRayHit | null {
  const candidates = sphereRayCandidates(origin, direction, descriptor.radius_m).map((t_m) => {
    const point = { x: origin.x + direction.x * t_m, y: origin.y + direction.y * t_m, z: origin.z + direction.z * t_m };
    return { t_m, normal: normalize(point)!, featureId: SVO_PRIMITIVE_FEATURES.smooth };
  });
  return nearestLocalHit(candidates, ray);
}

function intersectBoxLocal(
  descriptor: SvoBoxPrimitive,
  origin: Vec3,
  direction: Vec3,
  ray: CanonicalPrimitiveRay,
): LocalPrimitiveRayHit | null {
  const axes = ["x", "y", "z"] as const;
  let enter = Number.NEGATIVE_INFINITY;
  let exit = Number.POSITIVE_INFINITY;
  let enterAxis: typeof axes[number] = "x";
  let exitAxis: typeof axes[number] = "x";
  for (const axis of axes) {
    const extent = descriptor.halfExtents_m[axis];
    if (Math.abs(direction[axis]) <= NORMAL_EPSILON) {
      if (origin[axis] < -extent || origin[axis] > extent) return null;
      continue;
    }
    let near = (-extent - origin[axis]) / direction[axis];
    let far = (extent - origin[axis]) / direction[axis];
    if (near > far) [near, far] = [far, near];
    // Strict comparisons preserve the stable X -> Y -> Z feature tie.
    if (near > enter) { enter = near; enterAxis = axis; }
    if (far < exit) { exit = far; exitAxis = axis; }
    if (exit < enter) return null;
  }
  const candidates = ([{ t_m: enter, axis: enterAxis }, { t_m: exit, axis: exitAxis }]).map(({ t_m, axis }) => {
    const coordinate = origin[axis] + direction[axis] * t_m;
    return {
      t_m,
      normal: {
        x: axis === "x" ? Math.sign(coordinate || -direction.x || 1) : 0,
        y: axis === "y" ? Math.sign(coordinate || -direction.y || 1) : 0,
        z: axis === "z" ? Math.sign(coordinate || -direction.z || 1) : 0,
      },
      featureId: axis === "x" ? SVO_PRIMITIVE_FEATURES.boxFaceX
        : axis === "y" ? SVO_PRIMITIVE_FEATURES.boxFaceY : SVO_PRIMITIVE_FEATURES.boxFaceZ,
    };
  });
  return nearestLocalHit(candidates, ray);
}

function intersectCapsuleLocal(
  descriptor: SvoCapsulePrimitive,
  origin: Vec3,
  direction: Vec3,
  ray: CanonicalPrimitiveRay,
): LocalPrimitiveRayHit | null {
  const candidates: LocalPrimitiveRayHit[] = [];
  const radialA = direction.x ** 2 + direction.z ** 2;
  const radialB = origin.x * direction.x + origin.z * direction.z;
  const radialC = origin.x ** 2 + origin.z ** 2 - descriptor.radius_m ** 2;
  for (const t_m of quadraticRoots(radialA, radialB, radialC)) {
    const y = origin.y + direction.y * t_m;
    if (y >= -descriptor.segmentHalfLength_m && y <= descriptor.segmentHalfLength_m) {
      const point = { x: origin.x + direction.x * t_m, y: 0, z: origin.z + direction.z * t_m };
      candidates.push({ t_m, normal: normalize(point)!, featureId: SVO_PRIMITIVE_FEATURES.smooth });
    }
  }
  for (const sign of [-1, 1] as const) {
    const centerY = sign * descriptor.segmentHalfLength_m;
    for (const t_m of sphereRayCandidates(origin, direction, descriptor.radius_m, centerY)) {
      const point = {
        x: origin.x + direction.x * t_m,
        y: origin.y + direction.y * t_m - centerY,
        z: origin.z + direction.z * t_m,
      };
      if ((sign < 0 && point.y <= 0) || (sign > 0 && point.y >= 0)) {
        candidates.push({ t_m, normal: normalize(point)!, featureId: SVO_PRIMITIVE_FEATURES.smooth });
      }
    }
  }
  return nearestLocalHit(candidates, ray);
}

function intersectCylinderLocal(
  descriptor: SvoCylinderPrimitive,
  origin: Vec3,
  direction: Vec3,
  ray: CanonicalPrimitiveRay,
): LocalPrimitiveRayHit | null {
  const candidates: LocalPrimitiveRayHit[] = [];
  const radialA = direction.x ** 2 + direction.z ** 2;
  const radialB = origin.x * direction.x + origin.z * direction.z;
  const radialC = origin.x ** 2 + origin.z ** 2 - descriptor.radius_m ** 2;
  for (const t_m of quadraticRoots(radialA, radialB, radialC)) {
    const y = origin.y + direction.y * t_m;
    if (y >= -descriptor.halfHeight_m && y <= descriptor.halfHeight_m) {
      const point = { x: origin.x + direction.x * t_m, y: 0, z: origin.z + direction.z * t_m };
      candidates.push({ t_m, normal: normalize(point)!, featureId: SVO_PRIMITIVE_FEATURES.cylinderSide });
    }
  }
  if (Math.abs(direction.y) > NORMAL_EPSILON) {
    for (const sign of [-1, 1] as const) {
      const t_m = (sign * descriptor.halfHeight_m - origin.y) / direction.y;
      const x = origin.x + direction.x * t_m;
      const z = origin.z + direction.z * t_m;
      if (x * x + z * z <= descriptor.radius_m ** 2 * (1 + 1e-12)) {
        // Caps are appended after sides, then promoted on an exact rim tie below.
        candidates.push({ t_m, normal: { x: 0, y: sign, z: 0 }, featureId: SVO_PRIMITIVE_FEATURES.cylinderCap });
      }
    }
  }
  const hit = nearestLocalHit(candidates, ray);
  if (!hit || hit.featureId === SVO_PRIMITIVE_FEATURES.cylinderCap) return hit;
  const tiedCap = candidates.find((candidate) => candidate.featureId === SVO_PRIMITIVE_FEATURES.cylinderCap
    && inPrimitiveRayRange(candidate.t_m, ray) && Math.abs(candidate.t_m - hit.t_m) <= 1e-10 * Math.max(1, hit.t_m));
  return tiedCap ? { ...tiedCap, t_m: hit.t_m } : hit;
}

function intersectEllipsoidLocal(
  descriptor: SvoEllipsoidPrimitive,
  origin: Vec3,
  direction: Vec3,
  ray: CanonicalPrimitiveRay,
): LocalPrimitiveRayHit | null {
  const scaledOrigin = {
    x: origin.x / descriptor.radii_m.x,
    y: origin.y / descriptor.radii_m.y,
    z: origin.z / descriptor.radii_m.z,
  };
  const scaledDirection = {
    x: direction.x / descriptor.radii_m.x,
    y: direction.y / descriptor.radii_m.y,
    z: direction.z / descriptor.radii_m.z,
  };
  const roots = quadraticRoots(
    scaledDirection.x ** 2 + scaledDirection.y ** 2 + scaledDirection.z ** 2,
    scaledOrigin.x * scaledDirection.x + scaledOrigin.y * scaledDirection.y + scaledOrigin.z * scaledDirection.z,
    scaledOrigin.x ** 2 + scaledOrigin.y ** 2 + scaledOrigin.z ** 2 - 1,
  );
  const candidates = roots.map((t_m) => {
    const point = { x: origin.x + direction.x * t_m, y: origin.y + direction.y * t_m, z: origin.z + direction.z * t_m };
    return {
      t_m,
      normal: normalize({
        x: point.x / descriptor.radii_m.x ** 2,
        y: point.y / descriptor.radii_m.y ** 2,
        z: point.z / descriptor.radii_m.z ** 2,
      })!,
      featureId: SVO_PRIMITIVE_FEATURES.smooth,
    };
  });
  return nearestLocalHit(candidates, ray);
}

function intersectCanonicalSvoPrimitive(
  descriptor: SvoFinitePrimitiveDescriptor,
  ray: CanonicalPrimitiveRay,
): SvoPrimitiveRayHit | null {
  const { origin, direction, orientation } = localPrimitiveRay(descriptor, ray);
  const localHit = descriptor.kind === "sphere" ? intersectSphereLocal(descriptor, origin, direction, ray)
    : descriptor.kind === "box" ? intersectBoxLocal(descriptor, origin, direction, ray)
      : descriptor.kind === "capsule" ? intersectCapsuleLocal(descriptor, origin, direction, ray)
        : descriptor.kind === "cylinder" ? intersectCylinderLocal(descriptor, origin, direction, ray)
          : intersectEllipsoidLocal(descriptor, origin, direction, ray);
  if (!localHit) return null;
  const normal = worldNormal(orientation, localHit.normal);
  if (!normal) return null;
  return {
    t_m: localHit.t_m,
    position_m: {
      x: ray.origin_m.x + ray.direction.x * localHit.t_m,
      y: ray.origin_m.y + ray.direction.y * localHit.t_m,
      z: ray.origin_m.z + ray.direction.z * localHit.t_m,
    },
    normal,
    normalPolicy: descriptor.kind === "box" || descriptor.kind === "cylinder" ? "hard-feature" : "smooth",
    featureId: localHit.featureId,
    primitiveKind: descriptor.kind,
    primitiveId: descriptor.primitiveId,
    materialId: descriptor.materialId,
    ownerId: descriptor.ownerId ?? SPARSE_BRICK_NO_OWNER,
  };
}

/** Exact analytic finite-primitive hit oracle. Terrain is intentionally handled by its separate heightfield tracer. */
export function intersectSvoPrimitive(input: SvoFinitePrimitiveDescriptor, rayInput: SvoPrimitiveRay): SvoPrimitiveRayHit | null {
  const descriptor = canonicalSvoPrimitive(input);
  if (descriptor.kind === "terrain-heightfield") throw new TypeError("Terrain heightfield intersection uses the separate terrain tracer");
  return intersectCanonicalSvoPrimitive(descriptor, canonicalPrimitiveRay(rayInput));
}

/** Nearest exact finite-primitive hit. Input order is the deterministic tie-breaker. */
export function intersectSvoPrimitives(inputs: readonly SvoPrimitiveDescriptor[], rayInput: SvoPrimitiveRay): SvoPrimitiveRayHit | null {
  const ray = canonicalPrimitiveRay(rayInput);
  let nearest: SvoPrimitiveRayHit | null = null;
  for (const input of inputs) {
    const descriptor = canonicalSvoPrimitive(input);
    if (descriptor.kind === "terrain-heightfield") continue;
    const hit = intersectCanonicalSvoPrimitive(descriptor, ray);
    if (hit && (!nearest || hit.t_m < nearest.t_m)) nearest = hit;
  }
  return nearest;
}

/** Unpack the stable 64-byte ABI and select its nearest finite analytic hit. */
export function intersectPackedSvoPrimitiveRecords(packed: Uint32Array, ray: SvoPrimitiveRay): SvoPrimitiveRayHit | null {
  return intersectSvoPrimitives(unpackSvoPrimitiveRecords(packed), ray);
}

interface SvoEllipsoidClosestPoint {
  point: Vec3;
  ambiguous: boolean;
}

function ellipsoidClosestEquation(extents: readonly number[], point: readonly number[], lambda: number): number {
  let sum = 0;
  for (let axis = 0; axis < extents.length; axis += 1) {
    const extentSquared = extents[axis] ** 2;
    const ratio = extents[axis] * point[axis] / (extentSquared + lambda);
    sum += ratio * ratio;
  }
  return sum - 1;
}

/**
 * Exact Euclidean closest point in the positive orthant. The active-axis
 * reduction handles interior medial-axis singularities; every root solve uses
 * the fixed public bisection ceiling mirrored by WGSL.
 */
function closestEllipsoidPositive(extents: readonly number[], point: readonly number[]): { point: number[]; ambiguous: boolean } {
  if (extents.length === 1) return { point: [extents[0]], ambiguous: point[0] <= NORMAL_EPSILON };
  const last = extents.length - 1;
  const smallestExtent = extents[last];
  const coordinateTolerance = NORMAL_EPSILON * Math.max(1, extents[0]);
  if (point[last] <= coordinateTolerance) {
    const candidate = new Array<number>(extents.length).fill(0);
    let surfaceSum = 0;
    let valid = true;
    for (let axis = 0; axis < last; axis += 1) {
      const denominator = extents[axis] ** 2 - smallestExtent ** 2;
      if (Math.abs(denominator) <= NORMAL_EPSILON * Math.max(1, extents[axis] ** 2)) {
        if (point[axis] > coordinateTolerance) valid = false;
        continue;
      }
      candidate[axis] = extents[axis] ** 2 * point[axis] / denominator;
      surfaceSum += (candidate[axis] / extents[axis]) ** 2;
    }
    if (valid && surfaceSum <= 1 + 32 * Number.EPSILON) {
      candidate[last] = smallestExtent * Math.sqrt(Math.max(0, 1 - surfaceSum));
      return { point: candidate, ambiguous: candidate[last] > coordinateTolerance };
    }
    const reduced = closestEllipsoidPositive(extents.slice(0, last), point.slice(0, last));
    return { point: [...reduced.point, 0], ambiguous: reduced.ambiguous };
  }

  const equationAtZero = ellipsoidClosestEquation(extents, point, 0);
  if (Math.abs(equationAtZero) <= 32 * Number.EPSILON) return { point: [...point], ambiguous: false };
  let lower = equationAtZero < 0 ? -(smallestExtent ** 2) : 0;
  let upper = equationAtZero < 0 ? 0 : Math.max(1, extents[0] * Math.hypot(...point));
  for (let iteration = 0; iteration < SVO_ELLIPSOID_CLOSEST_POINT_ITERATIONS; iteration += 1) {
    const middle = 0.5 * (lower + upper);
    if (ellipsoidClosestEquation(extents, point, middle) > 0) lower = middle;
    else upper = middle;
  }
  const lambda = 0.5 * (lower + upper);
  return {
    point: extents.map((extent, axis) => extent ** 2 * point[axis] / (extent ** 2 + lambda)),
    ambiguous: false,
  };
}

function closestEllipsoidPoint(radii: Vec3, point: Vec3): SvoEllipsoidClosestPoint {
  const extents = [radii.x, radii.y, radii.z];
  const coordinates = [point.x, point.y, point.z];
  const axes = [0, 1, 2].sort((a, b) => extents[b] - extents[a]);
  const sortedExtents = axes.map((axis) => extents[axis]);
  const positivePoint = axes.map((axis) => Math.abs(coordinates[axis]));
  const result = closestEllipsoidPositive(sortedExtents, positivePoint);
  const closest = [0, 0, 0];
  for (let sortedAxis = 0; sortedAxis < axes.length; sortedAxis += 1) {
    const originalAxis = axes[sortedAxis];
    closest[originalAxis] = Math.sign(coordinates[originalAxis] || 1) * result.point[sortedAxis];
  }
  return { point: { x: closest[0], y: closest[1], z: closest[2] }, ambiguous: result.ambiguous };
}

/** CPU numerical mirror of the WGSL evaluation and hard-feature normal policy. */
export function sampleSvoPrimitive(
  input: SvoPrimitiveDescriptor,
  worldPoint_m: Vec3,
  terrainResolver?: SvoTerrainResolver,
): SvoPrimitiveSample {
  const descriptor = canonicalSvoPrimitive(input);
  if (descriptor.kind === "terrain-heightfield") {
    if (!terrainResolver) throw new Error("Terrain primitive evaluation requires a terrain resolver");
    finiteVec3(worldPoint_m, "Primitive query point");
    const terrain = terrainResolver(descriptor.terrainReference);
    return {
      signedDistance_m: worldPoint_m.y - terrainHeightAt(terrain, worldPoint_m.x, worldPoint_m.z),
      normal: terrainNormalAt(terrain, worldPoint_m.x, worldPoint_m.z, descriptor.normalEpsilon_m),
      featureId: SVO_PRIMITIVE_FEATURES.terrain,
    };
  }
  const { point, orientation } = localPoint(descriptor, worldPoint_m);
  if (descriptor.kind === "sphere") {
    const length = Math.hypot(point.x, point.y, point.z);
    return { signedDistance_m: length - descriptor.radius_m, normal: worldNormal(orientation, normalize(point)), featureId: SVO_PRIMITIVE_FEATURES.smooth };
  }
  if (descriptor.kind === "box") {
    const q = {
      x: Math.abs(point.x) - descriptor.halfExtents_m.x,
      y: Math.abs(point.y) - descriptor.halfExtents_m.y,
      z: Math.abs(point.z) - descriptor.halfExtents_m.z,
    };
    const outside = { x: Math.max(q.x, 0), y: Math.max(q.y, 0), z: Math.max(q.z, 0) };
    const signedDistance_m = Math.hypot(outside.x, outside.y, outside.z) + Math.min(Math.max(q.x, q.y, q.z), 0);
    // Select exactly one authored face. Ties are stable X -> Y -> Z and never
    // average normals across a sharp edge or corner.
    let axis: "x" | "y" | "z" = "x";
    if (q.y > q.x) axis = "y";
    if (q.z > q[axis]) axis = "z";
    const localNormal = { x: axis === "x" ? Math.sign(point.x || 1) : 0, y: axis === "y" ? Math.sign(point.y || 1) : 0, z: axis === "z" ? Math.sign(point.z || 1) : 0 };
    const featureId = axis === "x" ? SVO_PRIMITIVE_FEATURES.boxFaceX : axis === "y" ? SVO_PRIMITIVE_FEATURES.boxFaceY : SVO_PRIMITIVE_FEATURES.boxFaceZ;
    return { signedDistance_m, normal: worldNormal(orientation, localNormal), featureId };
  }
  if (descriptor.kind === "capsule") {
    const segmentY = Math.max(-descriptor.segmentHalfLength_m, Math.min(descriptor.segmentHalfLength_m, point.y));
    const offset = { x: point.x, y: point.y - segmentY, z: point.z };
    return {
      signedDistance_m: Math.hypot(offset.x, offset.y, offset.z) - descriptor.radius_m,
      normal: worldNormal(orientation, normalize(offset)), featureId: SVO_PRIMITIVE_FEATURES.smooth,
    };
  }
  if (descriptor.kind === "cylinder") {
    const radialLength = Math.hypot(point.x, point.z);
    const radialDistance = radialLength - descriptor.radius_m;
    const capDistance = Math.abs(point.y) - descriptor.halfHeight_m;
    const signedDistance_m = Math.hypot(Math.max(radialDistance, 0), Math.max(capDistance, 0)) + Math.min(Math.max(radialDistance, capDistance), 0);
    const capWins = capDistance >= radialDistance;
    const localNormal = capWins
      ? { x: 0, y: Math.sign(point.y || 1), z: 0 }
      : normalize({ x: point.x, y: 0, z: point.z });
    return {
      signedDistance_m, normal: worldNormal(orientation, localNormal),
      featureId: capWins ? SVO_PRIMITIVE_FEATURES.cylinderCap : SVO_PRIMITIVE_FEATURES.cylinderSide,
    };
  }
  const closestResult = closestEllipsoidPoint(descriptor.radii_m, point);
  const delta = {
    x: point.x - closestResult.point.x,
    y: point.y - closestResult.point.y,
    z: point.z - closestResult.point.z,
  };
  const distance = Math.hypot(delta.x, delta.y, delta.z);
  const inside = Math.hypot(
    point.x / descriptor.radii_m.x,
    point.y / descriptor.radii_m.y,
    point.z / descriptor.radii_m.z,
  ) < 1;
  const signedDistance_m = inside ? -distance : distance;
  const onSurface = distance <= NORMAL_EPSILON * Math.max(1, descriptor.radii_m.x, descriptor.radii_m.y, descriptor.radii_m.z);
  const normal = onSurface
    ? normalize({
      x: point.x / descriptor.radii_m.x ** 2,
      y: point.y / descriptor.radii_m.y ** 2,
      z: point.z / descriptor.radii_m.z ** 2,
    })
    : closestResult.ambiguous ? null : normalize(inside
      ? { x: -delta.x, y: -delta.y, z: -delta.z }
      : delta);
  return { signedDistance_m, normal: worldNormal(orientation, normal), featureId: SVO_PRIMITIVE_FEATURES.smooth };
}

/**
 * Shared WGSL declaration/evaluation library. Terrain height and normal are
 * supplied by the scene's existing terrain evaluator using metadata.y as its
 * stable table reference. Box/cylinder normals select one feature; they never
 * average across hard boundaries.
 */
export const svoPrimitiveWGSL = /* wgsl */ `
const SVO_KIND_SPHERE: u32 = 1u;
const SVO_KIND_BOX: u32 = 2u;
const SVO_KIND_CAPSULE: u32 = 3u;
const SVO_KIND_CYLINDER: u32 = 4u;
const SVO_KIND_ELLIPSOID: u32 = 5u;
const SVO_KIND_TERRAIN: u32 = 6u;
const SVO_FEATURE_SMOOTH: u32 = 0u;
const SVO_FEATURE_BOX_X: u32 = 1u;
const SVO_FEATURE_BOX_Y: u32 = 2u;
const SVO_FEATURE_BOX_Z: u32 = 3u;
const SVO_FEATURE_CYLINDER_SIDE: u32 = 4u;
const SVO_FEATURE_CYLINDER_CAP: u32 = 5u;
const SVO_FEATURE_TERRAIN: u32 = 6u;
const SVO_SAMPLE_NORMAL_VALID: u32 = 1u;
const SVO_PRIMITIVE_RAY_MISS: u32 = 0u;
const SVO_PRIMITIVE_RAY_HIT: u32 = 1u;
const SVO_PRIMITIVE_RAY_INVALID: u32 = 2u;
const SVO_PRIMITIVE_RAY_INFINITY: f32 = 3.402823e38;

struct SvoPrimitiveRecord {
  centerKind: vec4u,
  dimensionsIdentity: vec4u,
  orientation: vec4f,
  metadata: vec4u,
}

struct SvoPrimitiveSample {
  signedDistance_m: f32,
  featureId: u32,
  flags: u32,
  _padding: u32,
  normal: vec4f,
}

struct SvoPrimitiveRayResult {
  t_m: f32,
  featureId: u32,
  status: u32,
  _padding: u32,
  normal: vec4f,
}

struct SvoPrimitiveQuadraticRoots {
  values: vec2f,
  count: u32,
  _padding: u32,
}

fn svoPrimitiveCenter_m(record: SvoPrimitiveRecord) -> vec3f { return bitcast<vec3f>(record.centerKind.xyz); }
fn svoPrimitiveKind(record: SvoPrimitiveRecord) -> u32 { return record.centerKind.w; }
fn svoPrimitiveDimensions_m(record: SvoPrimitiveRecord) -> vec3f { return bitcast<vec3f>(record.dimensionsIdentity.xyz); }
fn svoPrimitiveMaterialId(record: SvoPrimitiveRecord) -> u32 { return record.dimensionsIdentity.w & 0xffffu; }
fn svoPrimitiveOwnerId(record: SvoPrimitiveRecord) -> u32 { return record.dimensionsIdentity.w >> 16u; }
fn svoPrimitiveId(record: SvoPrimitiveRecord) -> u32 { return record.metadata.x; }
fn svoPrimitiveTerrainReference(record: SvoPrimitiveRecord) -> u32 { return record.metadata.y; }

fn svoQuaternionRotate(q: vec4f, point: vec3f) -> vec3f {
  let twiceCross = 2.0 * cross(q.xyz, point);
  return point + q.w * twiceCross + cross(q.xyz, twiceCross);
}

fn svoPrimitiveLocalPoint(record: SvoPrimitiveRecord, worldPoint_m: vec3f) -> vec3f {
  let q = record.orientation;
  return svoQuaternionRotate(vec4f(-q.xyz, q.w), worldPoint_m - svoPrimitiveCenter_m(record));
}

fn svoPrimitiveNoRayHit(status: u32) -> SvoPrimitiveRayResult {
  return SvoPrimitiveRayResult(SVO_PRIMITIVE_RAY_INFINITY, SVO_FEATURE_SMOOTH, status, 0u, vec4f(0.0));
}

// Sorted roots of a*t^2 + 2*b*t + c. The stable q form retains near roots.
fn svoPrimitiveQuadraticRoots(a: f32, b: f32, c: f32) -> SvoPrimitiveQuadraticRoots {
  if (!(a > 1e-8)) { return SvoPrimitiveQuadraticRoots(vec2f(0.0), 0u, 0u); }
  let discriminant = b * b - a * c;
  let tolerance = 8e-6 * max(1.0, max(abs(b * b), abs(a * c)));
  if (discriminant < -tolerance) { return SvoPrimitiveQuadraticRoots(vec2f(0.0), 0u, 0u); }
  let root = sqrt(max(0.0, discriminant));
  if (root == 0.0) { return SvoPrimitiveQuadraticRoots(vec2f(-b / a, 0.0), 1u, 0u); }
  let q = -b - select(-root, root, b >= 0.0);
  let first = q / a;
  let second = c / q;
  return SvoPrimitiveQuadraticRoots(vec2f(min(first, second), max(first, second)), 2u, 0u);
}

fn svoPrimitiveRayInRange(t_m: f32, tMin_m: f32, tMax_m: f32) -> bool {
  let tolerance_m = 8e-6 * max(1.0, max(abs(t_m), max(abs(tMin_m), abs(tMax_m))));
  return t_m >= tMin_m - tolerance_m && t_m <= tMax_m + tolerance_m;
}

/** Exact bounded analytic hit for every finite primitive kind in the shared ABI. */
fn svoIntersectPrimitiveExact(
  record: SvoPrimitiveRecord,
  worldOrigin_m: vec3f,
  worldDirectionIn: vec3f,
  tMin_m: f32,
  tMax_m: f32,
) -> SvoPrimitiveRayResult {
  let directionLength = length(worldDirectionIn);
  let orientationLength = length(record.orientation);
  if (!(directionLength > 1e-8) || !(orientationLength > 1e-8) || !(tMin_m >= 0.0) || !(tMax_m >= tMin_m)) {
    return svoPrimitiveNoRayHit(SVO_PRIMITIVE_RAY_INVALID);
  }
  let worldDirection = worldDirectionIn / directionLength;
  let q = record.orientation / orientationLength;
  let inverse = vec4f(-q.xyz, q.w);
  let localOrigin = svoQuaternionRotate(inverse, worldOrigin_m - svoPrimitiveCenter_m(record));
  let localDirection = svoQuaternionRotate(inverse, worldDirection);
  let dimensions_m = svoPrimitiveDimensions_m(record);
  let kind = svoPrimitiveKind(record);
  let finiteKind = kind >= SVO_KIND_SPHERE && kind <= SVO_KIND_ELLIPSOID;
  let dimensionsValid = select(
    dimensions_m.x > 0.0,
    all(dimensions_m > vec3f(0.0)),
    kind == SVO_KIND_BOX || kind == SVO_KIND_ELLIPSOID,
  ) && select(true, dimensions_m.y >= 0.0, kind == SVO_KIND_CAPSULE)
    && select(true, dimensions_m.y > 0.0, kind == SVO_KIND_CYLINDER);
  if (!finiteKind || !dimensionsValid) { return svoPrimitiveNoRayHit(SVO_PRIMITIVE_RAY_INVALID); }

  var bestT_m = SVO_PRIMITIVE_RAY_INFINITY;
  var bestNormal = vec3f(0.0);
  var bestFeature = SVO_FEATURE_SMOOTH;

  if (kind == SVO_KIND_SPHERE) {
    let roots = svoPrimitiveQuadraticRoots(
      dot(localDirection, localDirection),
      dot(localOrigin, localDirection),
      dot(localOrigin, localOrigin) - dimensions_m.x * dimensions_m.x,
    );
    for (var rootIndex = 0u; rootIndex < 2u; rootIndex += 1u) {
      if (rootIndex >= roots.count) { break; }
      let candidate = roots.values[rootIndex];
      if (svoPrimitiveRayInRange(candidate, tMin_m, tMax_m)) {
        bestT_m = max(tMin_m, candidate);
        bestNormal = normalize(localOrigin + localDirection * bestT_m);
        break;
      }
    }
  } else if (kind == SVO_KIND_ELLIPSOID) {
    let scaledOrigin = localOrigin / dimensions_m;
    let scaledDirection = localDirection / dimensions_m;
    let roots = svoPrimitiveQuadraticRoots(
      dot(scaledDirection, scaledDirection),
      dot(scaledOrigin, scaledDirection),
      dot(scaledOrigin, scaledOrigin) - 1.0,
    );
    for (var rootIndex = 0u; rootIndex < 2u; rootIndex += 1u) {
      if (rootIndex >= roots.count) { break; }
      let candidate = roots.values[rootIndex];
      if (svoPrimitiveRayInRange(candidate, tMin_m, tMax_m)) {
        bestT_m = max(tMin_m, candidate);
        let point_m = localOrigin + localDirection * bestT_m;
        bestNormal = normalize(point_m / (dimensions_m * dimensions_m));
        break;
      }
    }
  } else if (kind == SVO_KIND_BOX) {
    var enter = -SVO_PRIMITIVE_RAY_INFINITY;
    var exit = SVO_PRIMITIVE_RAY_INFINITY;
    var enterAxis = 0u;
    var exitAxis = 0u;
    var valid = true;
    for (var axis = 0u; axis < 3u; axis += 1u) {
      if (abs(localDirection[axis]) <= 1e-8) {
        if (localOrigin[axis] < -dimensions_m[axis] || localOrigin[axis] > dimensions_m[axis]) { valid = false; }
      } else {
        let first = (-dimensions_m[axis] - localOrigin[axis]) / localDirection[axis];
        let second = (dimensions_m[axis] - localOrigin[axis]) / localDirection[axis];
        let nearT = min(first, second);
        let farT = max(first, second);
        if (nearT > enter) { enter = nearT; enterAxis = axis; }
        if (farT < exit) { exit = farT; exitAxis = axis; }
        if (exit < enter) { valid = false; }
      }
    }
    let useEnter = svoPrimitiveRayInRange(enter, tMin_m, tMax_m);
    let candidate = select(exit, enter, useEnter);
    let featureAxis = select(exitAxis, enterAxis, useEnter);
    if (valid && svoPrimitiveRayInRange(candidate, tMin_m, tMax_m)) {
      bestT_m = max(tMin_m, candidate);
      let point_m = localOrigin + localDirection * bestT_m;
      bestNormal[featureAxis] = select(-1.0, 1.0, point_m[featureAxis] >= 0.0);
      bestFeature = SVO_FEATURE_BOX_X + featureAxis;
    }
  } else {
    let radialRoots = svoPrimitiveQuadraticRoots(
      dot(localDirection.xz, localDirection.xz),
      dot(localOrigin.xz, localDirection.xz),
      dot(localOrigin.xz, localOrigin.xz) - dimensions_m.x * dimensions_m.x,
    );
    for (var rootIndex = 0u; rootIndex < 2u; rootIndex += 1u) {
      if (rootIndex >= radialRoots.count) { break; }
      let candidate = radialRoots.values[rootIndex];
      let y_m = localOrigin.y + localDirection.y * candidate;
      if (abs(y_m) <= dimensions_m.y && svoPrimitiveRayInRange(candidate, tMin_m, tMax_m) && candidate < bestT_m) {
        bestT_m = max(tMin_m, candidate);
        let point_m = localOrigin + localDirection * bestT_m;
        bestNormal = normalize(vec3f(point_m.x, 0.0, point_m.z));
        bestFeature = select(SVO_FEATURE_CYLINDER_SIDE, SVO_FEATURE_SMOOTH, kind == SVO_KIND_CAPSULE);
      }
    }
    if (kind == SVO_KIND_CAPSULE) {
      for (var capIndex = 0u; capIndex < 2u; capIndex += 1u) {
        let capSign = select(-1.0, 1.0, capIndex != 0u);
        let capCenter = vec3f(0.0, capSign * dimensions_m.y, 0.0);
        let offset = localOrigin - capCenter;
        let roots = svoPrimitiveQuadraticRoots(
          dot(localDirection, localDirection), dot(offset, localDirection),
          dot(offset, offset) - dimensions_m.x * dimensions_m.x,
        );
        for (var rootIndex = 0u; rootIndex < 2u; rootIndex += 1u) {
          if (rootIndex >= roots.count) { break; }
          let candidate = roots.values[rootIndex];
          let normalPoint = offset + localDirection * candidate;
          if (capSign * normalPoint.y >= 0.0 && svoPrimitiveRayInRange(candidate, tMin_m, tMax_m) && candidate < bestT_m) {
            bestT_m = max(tMin_m, candidate);
            bestNormal = normalize(offset + localDirection * bestT_m);
          }
        }
      }
    } else if (abs(localDirection.y) > 1e-8) {
      for (var capIndex = 0u; capIndex < 2u; capIndex += 1u) {
        let capSign = select(-1.0, 1.0, capIndex != 0u);
        let candidate = (capSign * dimensions_m.y - localOrigin.y) / localDirection.y;
        let point_m = localOrigin + localDirection * candidate;
        let tieTolerance_m = 8e-6 * max(1.0, abs(candidate));
        if (dot(point_m.xz, point_m.xz) <= dimensions_m.x * dimensions_m.x * 1.000008
          && svoPrimitiveRayInRange(candidate, tMin_m, tMax_m) && candidate <= bestT_m + tieTolerance_m) {
          bestT_m = max(tMin_m, candidate);
          bestNormal = vec3f(0.0, capSign, 0.0);
          bestFeature = SVO_FEATURE_CYLINDER_CAP;
        }
      }
    }
  }

  if (!(bestT_m < SVO_PRIMITIVE_RAY_INFINITY)) { return svoPrimitiveNoRayHit(SVO_PRIMITIVE_RAY_MISS); }
  let worldNormal = normalize(svoQuaternionRotate(q, bestNormal));
  return SvoPrimitiveRayResult(bestT_m, bestFeature, SVO_PRIMITIVE_RAY_HIT, 0u, vec4f(worldNormal, 0.0));
}

fn svoBoxDistance_m(point: vec3f, halfExtents_m: vec3f) -> f32 {
  let q = abs(point) - halfExtents_m;
  return length(max(q, vec3f(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}

fn svoCapsuleDistance_m(point: vec3f, dimensions_m: vec3f) -> f32 {
  let closestY = clamp(point.y, -dimensions_m.y, dimensions_m.y);
  return length(vec3f(point.x, point.y - closestY, point.z)) - dimensions_m.x;
}

fn svoCylinderDistance_m(point: vec3f, dimensions_m: vec3f) -> f32 {
  let q = vec2f(length(point.xz) - dimensions_m.x, abs(point.y) - dimensions_m.y);
  return length(max(q, vec2f(0.0))) + min(max(q.x, q.y), 0.0);
}

fn svoEllipsoidEquation2(extents_m: vec2f, point_m: vec2f, lambda: f32) -> f32 {
  let squared = extents_m * extents_m;
  let ratio = extents_m * point_m / (squared + vec2f(lambda));
  return dot(ratio, ratio) - 1.0;
}

fn svoEllipsoidEquation3(extents_m: vec3f, point_m: vec3f, lambda: f32) -> f32 {
  let squared = extents_m * extents_m;
  let ratio = extents_m * point_m / (squared + vec3f(lambda));
  return dot(ratio, ratio) - 1.0;
}

// xy is the positive-orthant closest point; z marks an ambiguous medial-axis normal.
fn svoEllipsoidClosest2(extents_m: vec2f, point_m: vec2f) -> vec3f {
  let tolerance = 1e-6 * max(1.0, extents_m.x);
  if (point_m.y <= tolerance) {
    let denominator = extents_m.x * extents_m.x - extents_m.y * extents_m.y;
    if (abs(denominator) > 1e-6 * max(1.0, extents_m.x * extents_m.x)) {
      let candidateX = extents_m.x * extents_m.x * point_m.x / denominator;
      let surfaceSum = candidateX * candidateX / (extents_m.x * extents_m.x);
      if (surfaceSum <= 1.000004) {
        let candidateY = extents_m.y * sqrt(max(0.0, 1.0 - surfaceSum));
        return vec3f(candidateX, candidateY, select(0.0, 1.0, candidateY > tolerance));
      }
    } else if (point_m.x <= tolerance) {
      return vec3f(0.0, extents_m.y, 1.0);
    }
    return vec3f(extents_m.x, 0.0, select(0.0, 1.0, point_m.x <= tolerance));
  }
  let equationAtZero = svoEllipsoidEquation2(extents_m, point_m, 0.0);
  if (abs(equationAtZero) <= 4e-6) { return vec3f(point_m, 0.0); }
  var lower = select(0.0, -extents_m.y * extents_m.y, equationAtZero < 0.0);
  var upper = select(max(1.0, extents_m.x * length(point_m)), 0.0, equationAtZero < 0.0);
  for (var iteration = 0u; iteration < ${SVO_ELLIPSOID_CLOSEST_POINT_ITERATIONS}u; iteration += 1u) {
    let middle = 0.5 * (lower + upper);
    if (svoEllipsoidEquation2(extents_m, point_m, middle) > 0.0) { lower = middle; } else { upper = middle; }
  }
  let lambda = 0.5 * (lower + upper);
  let squared = extents_m * extents_m;
  return vec3f(squared * point_m / (squared + vec2f(lambda)), 0.0);
}

// xyz is the signed-octant closest point; w marks an ambiguous medial-axis normal.
fn svoEllipsoidClosestPoint_m(radii_m: vec3f, point_m: vec3f) -> vec4f {
  var extents_m = radii_m;
  var positivePoint_m = abs(point_m);
  var axes = vec3u(0u, 1u, 2u);
  if (extents_m.x < extents_m.y) {
    let extent = extents_m.x; extents_m.x = extents_m.y; extents_m.y = extent;
    let coordinate = positivePoint_m.x; positivePoint_m.x = positivePoint_m.y; positivePoint_m.y = coordinate;
    let axis = axes.x; axes.x = axes.y; axes.y = axis;
  }
  if (extents_m.y < extents_m.z) {
    let extent = extents_m.y; extents_m.y = extents_m.z; extents_m.z = extent;
    let coordinate = positivePoint_m.y; positivePoint_m.y = positivePoint_m.z; positivePoint_m.z = coordinate;
    let axis = axes.y; axes.y = axes.z; axes.z = axis;
  }
  if (extents_m.x < extents_m.y) {
    let extent = extents_m.x; extents_m.x = extents_m.y; extents_m.y = extent;
    let coordinate = positivePoint_m.x; positivePoint_m.x = positivePoint_m.y; positivePoint_m.y = coordinate;
    let axis = axes.x; axes.x = axes.y; axes.y = axis;
  }

  let tolerance = 1e-6 * max(1.0, extents_m.x);
  var sortedClosest_m = vec3f(0.0);
  var ambiguous = 0.0;
  if (positivePoint_m.z <= tolerance) {
    let squared = extents_m * extents_m;
    let denominator = squared.xy - vec2f(squared.z);
    var reducedValid = true;
    var candidate = vec2f(0.0);
    for (var axis = 0u; axis < 2u; axis += 1u) {
      if (abs(denominator[axis]) <= 1e-6 * max(1.0, squared[axis])) {
        if (positivePoint_m[axis] > tolerance) { reducedValid = false; }
      } else {
        candidate[axis] = squared[axis] * positivePoint_m[axis] / denominator[axis];
      }
    }
    let surfaceSum = dot(candidate / extents_m.xy, candidate / extents_m.xy);
    if (reducedValid && surfaceSum <= 1.000004) {
      let candidateZ = extents_m.z * sqrt(max(0.0, 1.0 - surfaceSum));
      sortedClosest_m = vec3f(candidate, candidateZ);
      ambiguous = select(0.0, 1.0, candidateZ > tolerance);
    } else {
      let reduced = svoEllipsoidClosest2(extents_m.xy, positivePoint_m.xy);
      sortedClosest_m = vec3f(reduced.xy, 0.0);
      ambiguous = reduced.z;
    }
  } else {
    let equationAtZero = svoEllipsoidEquation3(extents_m, positivePoint_m, 0.0);
    if (abs(equationAtZero) <= 4e-6) {
      sortedClosest_m = positivePoint_m;
    } else {
      var lower = select(0.0, -extents_m.z * extents_m.z, equationAtZero < 0.0);
      var upper = select(max(1.0, extents_m.x * length(positivePoint_m)), 0.0, equationAtZero < 0.0);
      for (var iteration = 0u; iteration < ${SVO_ELLIPSOID_CLOSEST_POINT_ITERATIONS}u; iteration += 1u) {
        let middle = 0.5 * (lower + upper);
        if (svoEllipsoidEquation3(extents_m, positivePoint_m, middle) > 0.0) { lower = middle; } else { upper = middle; }
      }
      let lambda = 0.5 * (lower + upper);
      let squared = extents_m * extents_m;
      sortedClosest_m = squared * positivePoint_m / (squared + vec3f(lambda));
    }
  }
  var closest_m = vec3f(0.0);
  closest_m[axes.x] = select(-sortedClosest_m.x, sortedClosest_m.x, point_m[axes.x] >= 0.0);
  closest_m[axes.y] = select(-sortedClosest_m.y, sortedClosest_m.y, point_m[axes.y] >= 0.0);
  closest_m[axes.z] = select(-sortedClosest_m.z, sortedClosest_m.z, point_m[axes.z] >= 0.0);
  return vec4f(closest_m, ambiguous);
}

fn svoEllipsoidDistance_m(point: vec3f, radii_m: vec3f) -> f32 {
  if (any(radii_m <= vec3f(0.0))) { return 3.402823e38; }
  let closest = svoEllipsoidClosestPoint_m(radii_m, point);
  let distance_m = length(point - closest.xyz);
  return select(distance_m, -distance_m, dot(point / radii_m, point / radii_m) < 1.0);
}

fn svoPrimitiveDistance_m(record: SvoPrimitiveRecord, worldPoint_m: vec3f, terrainHeight_m: f32) -> f32 {
  let kind = svoPrimitiveKind(record);
  let dimensions_m = svoPrimitiveDimensions_m(record);
  if (kind == SVO_KIND_TERRAIN) { return worldPoint_m.y - terrainHeight_m; }
  let point = svoPrimitiveLocalPoint(record, worldPoint_m);
  if (kind == SVO_KIND_SPHERE) { return length(point) - dimensions_m.x; }
  if (kind == SVO_KIND_BOX) { return svoBoxDistance_m(point, dimensions_m); }
  if (kind == SVO_KIND_CAPSULE) { return svoCapsuleDistance_m(point, dimensions_m); }
  if (kind == SVO_KIND_CYLINDER) { return svoCylinderDistance_m(point, dimensions_m); }
  if (kind == SVO_KIND_ELLIPSOID) { return svoEllipsoidDistance_m(point, dimensions_m); }
  return 3.402823e38;
}

fn svoBoxFeatureNormal(point: vec3f, halfExtents_m: vec3f) -> vec4f {
  let q = abs(point) - halfExtents_m;
  var axis = 0u;
  if (q.y > q.x) { axis = 1u; }
  if (q.z > q[axis]) { axis = 2u; }
  var normal = vec3f(0.0);
  normal[axis] = select(-1.0, 1.0, point[axis] >= 0.0);
  return vec4f(normal, f32(SVO_FEATURE_BOX_X + axis));
}

fn svoCylinderFeatureNormal(point: vec3f, dimensions_m: vec3f) -> vec4f {
  let radialDistance = length(point.xz) - dimensions_m.x;
  let capDistance = abs(point.y) - dimensions_m.y;
  if (capDistance >= radialDistance) {
    return vec4f(0.0, select(-1.0, 1.0, point.y >= 0.0), 0.0, f32(SVO_FEATURE_CYLINDER_CAP));
  }
  let radial = point.xz / max(length(point.xz), 1e-8);
  return vec4f(radial.x, 0.0, radial.y, f32(SVO_FEATURE_CYLINDER_SIDE));
}

fn svoPrimitiveLocalNormal(record: SvoPrimitiveRecord, point: vec3f) -> vec4f {
  let kind = svoPrimitiveKind(record);
  let dimensions_m = svoPrimitiveDimensions_m(record);
  if (kind == SVO_KIND_SPHERE) { return vec4f(normalize(point), f32(SVO_FEATURE_SMOOTH)); }
  if (kind == SVO_KIND_BOX) { return svoBoxFeatureNormal(point, dimensions_m); }
  if (kind == SVO_KIND_CAPSULE) {
    let closestY = clamp(point.y, -dimensions_m.y, dimensions_m.y);
    return vec4f(normalize(vec3f(point.x, point.y - closestY, point.z)), f32(SVO_FEATURE_SMOOTH));
  }
  if (kind == SVO_KIND_CYLINDER) { return svoCylinderFeatureNormal(point, dimensions_m); }
  if (kind == SVO_KIND_ELLIPSOID) {
    if (any(dimensions_m <= vec3f(0.0))) { return vec4f(0.0); }
    let closest = svoEllipsoidClosestPoint_m(dimensions_m, point);
    let delta = point - closest.xyz;
    let distance_m = length(delta);
    let surfaceTolerance_m = 1e-6 * max(1.0, max(dimensions_m.x, max(dimensions_m.y, dimensions_m.z)));
    if (distance_m <= surfaceTolerance_m) {
      return vec4f(normalize(point / (dimensions_m * dimensions_m)), f32(SVO_FEATURE_SMOOTH));
    }
    if (closest.w > 0.5) { return vec4f(0.0); }
    let outward = select(delta, -delta, dot(point / dimensions_m, point / dimensions_m) < 1.0);
    return vec4f(normalize(outward), f32(SVO_FEATURE_SMOOTH));
  }
  return vec4f(0.0);
}

fn svoEvaluatePrimitive(record: SvoPrimitiveRecord, worldPoint_m: vec3f, terrainHeight_m: f32, terrainNormal: vec3f) -> SvoPrimitiveSample {
  let distance_m = svoPrimitiveDistance_m(record, worldPoint_m, terrainHeight_m);
  if (svoPrimitiveKind(record) == SVO_KIND_TERRAIN) {
    let normalLength = length(terrainNormal);
    return SvoPrimitiveSample(distance_m, SVO_FEATURE_TERRAIN, select(0u, SVO_SAMPLE_NORMAL_VALID, normalLength > 1e-8), 0u, vec4f(terrainNormal / max(normalLength, 1e-8), 0.0));
  }
  let localPoint = svoPrimitiveLocalPoint(record, worldPoint_m);
  let local = svoPrimitiveLocalNormal(record, localPoint);
  let localLength = length(local.xyz);
  let worldNormal = svoQuaternionRotate(record.orientation, local.xyz / max(localLength, 1e-8));
  return SvoPrimitiveSample(distance_m, u32(local.w), select(0u, SVO_SAMPLE_NORMAL_VALID, localLength > 1e-8), 0u, vec4f(worldNormal, 0.0));
}
`;
