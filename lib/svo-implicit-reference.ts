import type { Quaternion, RigidBodyDescription, Vec3 } from "./model";
import { terrainHeightAt, terrainNormalAt, type TerrainDescription } from "./terrain";

/**
 * Deterministic CPU reference for the implicit surfaces consumed by the SVO
 * renderer. Distances are expressed in world metres and are negative inside.
 * A null normal means the signed-distance gradient is not unique at the query
 * point (for example, at the centre of a sphere or on a box's medial plane).
 */
export interface SvoImplicitSample {
  signedDistance_m: number;
  normal: Vec3 | null;
}

interface LocatedImplicitReference {
  center_m: Vec3;
}

interface OrientedImplicitReference extends LocatedImplicitReference {
  /** Local primitive axis is +Y. Omitted orientations are the identity. */
  orientation?: Quaternion;
}

export interface SvoSphereReference extends LocatedImplicitReference {
  kind: "sphere";
  radius_m: number;
}

export interface SvoBoxReference extends OrientedImplicitReference {
  kind: "box";
  halfExtents_m: Vec3;
}

export interface SvoCapsuleReference extends OrientedImplicitReference {
  kind: "capsule";
  radius_m: number;
  /** Half-length of the cylindrical segment between the two hemispheres. */
  segmentHalfLength_m: number;
}

export interface SvoCylinderReference extends OrientedImplicitReference {
  kind: "cylinder";
  radius_m: number;
  halfHeight_m: number;
}

export interface SvoEllipsoidReference extends OrientedImplicitReference {
  kind: "ellipsoid";
  radii_m: Vec3;
}

export interface SvoTerrainHeightfieldReference {
  kind: "terrain-heightfield";
  terrain?: TerrainDescription;
  /** Central-difference spacing used by the existing terrain normal model. */
  normalEpsilon_m?: number;
}

export type SvoImplicitReference =
  | SvoSphereReference
  | SvoBoxReference
  | SvoCapsuleReference
  | SvoCylinderReference
  | SvoEllipsoidReference
  | SvoTerrainHeightfieldReference;

const NORMAL_EPSILON = 1e-12;

function assertFiniteVec3(value: Vec3, label: string): void {
  if (![value.x, value.y, value.z].every(Number.isFinite)) throw new RangeError(`${label} must be finite`);
}

function assertPositive(value: number, label: string): void {
  if (!Number.isFinite(value) || value <= 0) throw new RangeError(`${label} must be positive`);
}

function assertNonNegative(value: number, label: string): void {
  if (!Number.isFinite(value) || value < 0) throw new RangeError(`${label} must be non-negative`);
}

function normalizedOrientation(value: Quaternion | undefined): Quaternion {
  if (value === undefined) return { w: 1, x: 0, y: 0, z: 0 };
  if (![value.w, value.x, value.y, value.z].every(Number.isFinite)) {
    throw new RangeError("Implicit orientation must be finite");
  }
  const magnitude = Math.hypot(value.w, value.x, value.y, value.z);
  if (!(magnitude > NORMAL_EPSILON)) throw new RangeError("Implicit orientation must have nonzero length");
  return { w: value.w / magnitude, x: value.x / magnitude, y: value.y / magnitude, z: value.z / magnitude };
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

function normalized(v: Vec3): Vec3 | null {
  const magnitude = Math.hypot(v.x, v.y, v.z);
  if (!(magnitude > NORMAL_EPSILON)) return null;
  return { x: v.x / magnitude, y: v.y / magnitude, z: v.z / magnitude };
}

function localPoint(reference: LocatedImplicitReference, worldPoint_m: Vec3, orientation?: Quaternion): Vec3 {
  assertFiniteVec3(reference.center_m, "Implicit centre");
  assertFiniteVec3(worldPoint_m, "Implicit query point");
  const offset = {
    x: worldPoint_m.x - reference.center_m.x,
    y: worldPoint_m.y - reference.center_m.y,
    z: worldPoint_m.z - reference.center_m.z,
  };
  return orientation ? inverseRotate(orientation, offset) : offset;
}

function worldSample(sample: SvoImplicitSample, orientation: Quaternion | undefined): SvoImplicitSample {
  if (!sample.normal || orientation === undefined) return sample;
  return { ...sample, normal: rotate(orientation, sample.normal) };
}

function sampleSphere(reference: SvoSphereReference, worldPoint_m: Vec3): SvoImplicitSample {
  assertPositive(reference.radius_m, "Sphere radius");
  const point = localPoint(reference, worldPoint_m);
  const distanceFromCentre = Math.hypot(point.x, point.y, point.z);
  return {
    signedDistance_m: distanceFromCentre - reference.radius_m,
    normal: distanceFromCentre > NORMAL_EPSILON
      ? { x: point.x / distanceFromCentre, y: point.y / distanceFromCentre, z: point.z / distanceFromCentre }
      : null,
  };
}

function sampleBox(reference: SvoBoxReference, worldPoint_m: Vec3): SvoImplicitSample {
  assertPositive(reference.halfExtents_m.x, "Box X half extent");
  assertPositive(reference.halfExtents_m.y, "Box Y half extent");
  assertPositive(reference.halfExtents_m.z, "Box Z half extent");
  const orientation = normalizedOrientation(reference.orientation);
  const point = localPoint(reference, worldPoint_m, orientation);
  const q = {
    x: Math.abs(point.x) - reference.halfExtents_m.x,
    y: Math.abs(point.y) - reference.halfExtents_m.y,
    z: Math.abs(point.z) - reference.halfExtents_m.z,
  };
  const outside = { x: Math.max(q.x, 0), y: Math.max(q.y, 0), z: Math.max(q.z, 0) };
  const outsideDistance = Math.hypot(outside.x, outside.y, outside.z);
  const signedDistance_m = outsideDistance + Math.min(Math.max(q.x, q.y, q.z), 0);
  let normal: Vec3 | null;
  if (outsideDistance > NORMAL_EPSILON) {
    normal = {
      x: Math.sign(point.x) * outside.x / outsideDistance,
      y: Math.sign(point.y) * outside.y / outsideDistance,
      z: Math.sign(point.z) * outside.z / outsideDistance,
    };
  } else {
    const maximum = Math.max(q.x, q.y, q.z);
    const scale = Math.max(1, ...Object.values(reference.halfExtents_m));
    const tiedAxes = (["x", "y", "z"] as const).filter((axis) => Math.abs(q[axis] - maximum) <= NORMAL_EPSILON * scale);
    const axis = tiedAxes.length === 1 ? tiedAxes[0] : null;
    normal = axis && Math.abs(point[axis]) > NORMAL_EPSILON
      ? { x: axis === "x" ? Math.sign(point.x) : 0, y: axis === "y" ? Math.sign(point.y) : 0, z: axis === "z" ? Math.sign(point.z) : 0 }
      : null;
  }
  return worldSample({ signedDistance_m, normal }, orientation);
}

function sampleCapsule(reference: SvoCapsuleReference, worldPoint_m: Vec3): SvoImplicitSample {
  assertPositive(reference.radius_m, "Capsule radius");
  assertNonNegative(reference.segmentHalfLength_m, "Capsule segment half length");
  const orientation = normalizedOrientation(reference.orientation);
  const point = localPoint(reference, worldPoint_m, orientation);
  const segmentY = Math.max(-reference.segmentHalfLength_m, Math.min(reference.segmentHalfLength_m, point.y));
  const closestOffset = { x: point.x, y: point.y - segmentY, z: point.z };
  const distanceToSegment = Math.hypot(closestOffset.x, closestOffset.y, closestOffset.z);
  return worldSample({
    signedDistance_m: distanceToSegment - reference.radius_m,
    normal: distanceToSegment > NORMAL_EPSILON
      ? { x: closestOffset.x / distanceToSegment, y: closestOffset.y / distanceToSegment, z: closestOffset.z / distanceToSegment }
      : null,
  }, orientation);
}

function sampleCylinder(reference: SvoCylinderReference, worldPoint_m: Vec3): SvoImplicitSample {
  assertPositive(reference.radius_m, "Cylinder radius");
  assertPositive(reference.halfHeight_m, "Cylinder half height");
  const orientation = normalizedOrientation(reference.orientation);
  const point = localPoint(reference, worldPoint_m, orientation);
  const radialLength = Math.hypot(point.x, point.z);
  const radialDistance = radialLength - reference.radius_m;
  const capDistance = Math.abs(point.y) - reference.halfHeight_m;
  const outsideRadial = Math.max(radialDistance, 0);
  const outsideCap = Math.max(capDistance, 0);
  const outsideDistance = Math.hypot(outsideRadial, outsideCap);
  const signedDistance_m = outsideDistance + Math.min(Math.max(radialDistance, capDistance), 0);
  const radialNormal = radialLength > NORMAL_EPSILON
    ? { x: point.x / radialLength, y: 0, z: point.z / radialLength }
    : null;
  const capNormal = Math.abs(point.y) > NORMAL_EPSILON ? { x: 0, y: Math.sign(point.y), z: 0 } : null;
  let normal: Vec3 | null;
  if (outsideRadial > 0 && outsideCap > 0) {
    normal = radialNormal ? {
      x: radialNormal.x * outsideRadial / outsideDistance,
      y: (capNormal?.y ?? 0) * outsideCap / outsideDistance,
      z: radialNormal.z * outsideRadial / outsideDistance,
    } : capNormal;
  } else if (outsideRadial > 0) normal = radialNormal;
  else if (outsideCap > 0) normal = capNormal;
  else {
    const tieScale = Math.max(1, reference.radius_m, reference.halfHeight_m);
    if (Math.abs(radialDistance - capDistance) <= NORMAL_EPSILON * tieScale) normal = null;
    else normal = radialDistance > capDistance ? radialNormal : capNormal;
  }
  return worldSample({ signedDistance_m, normal }, orientation);
}

interface ClosestEllipsoidPoint {
  point: number[];
  ambiguous: boolean;
}

function ellipsoidEquation(extents: readonly number[], point: readonly number[], lambda: number): number {
  let sum = 0;
  for (let axis = 0; axis < extents.length; axis += 1) {
    const extentSquared = extents[axis] ** 2;
    const ratio = extents[axis] * point[axis] / (extentSquared + lambda);
    sum += ratio * ratio;
  }
  return sum - 1;
}

/** Exact closest point in the positive orthant of an axis-aligned ellipsoid. */
function closestEllipsoidPoint(extents: readonly number[], point: readonly number[]): ClosestEllipsoidPoint {
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
    const reduced = closestEllipsoidPoint(extents.slice(0, last), point.slice(0, last));
    return { point: [...reduced.point, 0], ambiguous: reduced.ambiguous };
  }

  const equationAtZero = ellipsoidEquation(extents, point, 0);
  if (Math.abs(equationAtZero) <= 32 * Number.EPSILON) return { point: [...point], ambiguous: false };
  let lower: number;
  let upper: number;
  if (equationAtZero < 0) {
    lower = -(smallestExtent ** 2);
    upper = 0;
  } else {
    lower = 0;
    upper = Math.max(1, extents[0] ** 2);
    while (ellipsoidEquation(extents, point, upper) > 0) upper *= 2;
  }
  for (let iteration = 0; iteration < 128; iteration += 1) {
    const middle = 0.5 * (lower + upper);
    if (ellipsoidEquation(extents, point, middle) > 0) lower = middle;
    else upper = middle;
  }
  const lambda = 0.5 * (lower + upper);
  return {
    point: extents.map((extent, axis) => extent ** 2 * point[axis] / (extent ** 2 + lambda)),
    ambiguous: false,
  };
}

function sampleEllipsoid(reference: SvoEllipsoidReference, worldPoint_m: Vec3): SvoImplicitSample {
  assertPositive(reference.radii_m.x, "Ellipsoid X radius");
  assertPositive(reference.radii_m.y, "Ellipsoid Y radius");
  assertPositive(reference.radii_m.z, "Ellipsoid Z radius");
  const orientation = normalizedOrientation(reference.orientation);
  const point = localPoint(reference, worldPoint_m, orientation);
  const radii = [reference.radii_m.x, reference.radii_m.y, reference.radii_m.z];
  const coordinates = [point.x, point.y, point.z];
  const axes = [0, 1, 2].sort((a, b) => radii[b] - radii[a]);
  const extents = axes.map((axis) => radii[axis]);
  const positivePoint = axes.map((axis) => Math.abs(coordinates[axis]));
  const result = closestEllipsoidPoint(extents, positivePoint);
  const closest = [0, 0, 0];
  for (let sortedAxis = 0; sortedAxis < axes.length; sortedAxis += 1) {
    const originalAxis = axes[sortedAxis];
    closest[originalAxis] = Math.sign(coordinates[originalAxis] || 1) * result.point[sortedAxis];
  }
  const delta = { x: point.x - closest[0], y: point.y - closest[1], z: point.z - closest[2] };
  const distance = Math.hypot(delta.x, delta.y, delta.z);
  const scaledRadius = Math.hypot(point.x / radii[0], point.y / radii[1], point.z / radii[2]);
  const inside = scaledRadius < 1;
  let normal: Vec3 | null;
  if (distance <= NORMAL_EPSILON * Math.max(1, ...radii)) {
    normal = normalized({ x: point.x / radii[0] ** 2, y: point.y / radii[1] ** 2, z: point.z / radii[2] ** 2 });
  } else if (result.ambiguous) normal = null;
  else normal = normalized(inside ? { x: -delta.x, y: -delta.y, z: -delta.z } : delta);
  return worldSample({ signedDistance_m: inside ? -distance : distance, normal }, orientation);
}

function sampleTerrain(reference: SvoTerrainHeightfieldReference, worldPoint_m: Vec3): SvoImplicitSample {
  assertFiniteVec3(worldPoint_m, "Implicit query point");
  const epsilon_m = reference.normalEpsilon_m ?? 0.02;
  assertPositive(epsilon_m, "Terrain normal epsilon");
  const height_m = terrainHeightAt(reference.terrain, worldPoint_m.x, worldPoint_m.z);
  return {
    // The renderer's terrain implicit is an exact vertical height residual in
    // metres. It has the correct sign and zero set; it is not an exact
    // Euclidean distance away from sloping terrain.
    signedDistance_m: worldPoint_m.y - height_m,
    normal: terrainNormalAt(reference.terrain, worldPoint_m.x, worldPoint_m.z, epsilon_m),
  };
}

export function sampleSvoImplicit(reference: SvoImplicitReference, worldPoint_m: Vec3): SvoImplicitSample {
  if (reference.kind === "sphere") return sampleSphere(reference, worldPoint_m);
  if (reference.kind === "box") return sampleBox(reference, worldPoint_m);
  if (reference.kind === "capsule") return sampleCapsule(reference, worldPoint_m);
  if (reference.kind === "cylinder") return sampleCylinder(reference, worldPoint_m);
  if (reference.kind === "ellipsoid") return sampleEllipsoid(reference, worldPoint_m);
  return sampleTerrain(reference, worldPoint_m);
}

/** Map the repository's rigid-body dimension semantics to renderer primitives. */
export function svoImplicitReferenceForRigidBody(
  body: Pick<RigidBodyDescription, "shape" | "dimensions_m" | "position_m" | "orientation">,
): Exclude<SvoImplicitReference, SvoEllipsoidReference | SvoTerrainHeightfieldReference> {
  const dimensions = body.dimensions_m;
  if (body.shape === "sphere") {
    return { kind: "sphere", center_m: { ...body.position_m }, radius_m: dimensions.x };
  }
  if (body.shape === "box") {
    return {
      kind: "box", center_m: { ...body.position_m }, orientation: { ...body.orientation },
      halfExtents_m: { x: dimensions.x / 2, y: dimensions.y / 2, z: dimensions.z / 2 },
    };
  }
  if (body.shape === "capsule") {
    return {
      kind: "capsule", center_m: { ...body.position_m }, orientation: { ...body.orientation },
      radius_m: dimensions.x, segmentHalfLength_m: dimensions.y / 2,
    };
  }
  return {
    kind: "cylinder", center_m: { ...body.position_m }, orientation: { ...body.orientation },
    radius_m: dimensions.x, halfHeight_m: dimensions.y / 2,
  };
}
