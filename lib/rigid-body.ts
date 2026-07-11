import { add, cross, dot, length, normalize, scale, sub } from "./math";
import type { Quaternion, RigidBodyDescription, RigidShape, SceneDescription, Vec3 } from "./model";

export interface MassProperties {
  volume_m3: number;
  mass_kg: number;
  inertiaBody_kg_m2: Vec3;
}

export interface RigidBodyState {
  description: RigidBodyDescription;
  position_m: Vec3;
  orientation: Quaternion;
  linearVelocity_m_s: Vec3;
  angularVelocity_rad_s: Vec3;
  angularMomentum_kg_m2_s: Vec3;
  mass_kg: number;
  inverseMass_kg: number;
  inertiaBody_kg_m2: Vec3;
  inverseInertiaBody_kg_m2: Vec3;
  netForce_N: Vec3;
  netTorque_N_m: Vec3;
  collisionImpulse_N_s: Vec3;
  collisionAngularImpulse_N_m_s: Vec3;
  contactCount: number;
  maxPenetration_m: number;
  quaternionNormError: number;
}

export interface RigidStepDiagnostics {
  contactCount: number;
  maxPenetration_m: number;
  kineticEnergy_J: number;
  potentialEnergy_J: number;
  linearMomentum_kg_m_s: Vec3;
  angularMomentum_kg_m2_s: Vec3;
  nanCount: number;
  quaternionMaxNormError: number;
}

const ZERO = (): Vec3 => ({ x: 0, y: 0, z: 0 });

export function primitiveVolume(shape: RigidShape, dimensions: Vec3): number {
  if (shape === "sphere") return (4 / 3) * Math.PI * dimensions.x ** 3;
  if (shape === "box") return dimensions.x * dimensions.y * dimensions.z;
  if (shape === "cylinder") return Math.PI * dimensions.x ** 2 * dimensions.y;
  return Math.PI * dimensions.x ** 2 * dimensions.y + (4 / 3) * Math.PI * dimensions.x ** 3;
}

export function massProperties(body: RigidBodyDescription): MassProperties {
  const { shape, dimensions_m: d, density_kg_m3: density } = body;
  const volume = primitiveVolume(shape, d);
  const mass = density * volume;
  let inertia: Vec3;
  if (shape === "sphere") {
    const value = (2 / 5) * mass * d.x ** 2;
    inertia = { x: value, y: value, z: value };
  } else if (shape === "box") {
    inertia = {
      x: mass * (d.y ** 2 + d.z ** 2) / 12,
      y: mass * (d.x ** 2 + d.z ** 2) / 12,
      z: mass * (d.x ** 2 + d.y ** 2) / 12
    };
  } else if (shape === "cylinder") {
    inertia = {
      x: mass * (3 * d.x ** 2 + d.y ** 2) / 12,
      y: 0.5 * mass * d.x ** 2,
      z: mass * (3 * d.x ** 2 + d.y ** 2) / 12
    };
  } else {
    const radius = d.x;
    const cylinderLength = d.y;
    const cylinderMass = density * Math.PI * radius ** 2 * cylinderLength;
    const sphereMass = density * (4 / 3) * Math.PI * radius ** 3;
    const axial = 0.5 * cylinderMass * radius ** 2 + (2 / 5) * sphereMass * radius ** 2;
    const transverseCaps = sphereMass * ((83 / 320) * radius ** 2 + (cylinderLength / 2 + 3 * radius / 8) ** 2);
    const transverse = cylinderMass * (3 * radius ** 2 + cylinderLength ** 2) / 12 + transverseCaps;
    inertia = { x: transverse, y: axial, z: transverse };
  }
  return { volume_m3: volume, mass_kg: mass, inertiaBody_kg_m2: inertia };
}

export function quaternionNormalize(q: Quaternion): Quaternion {
  const norm = Math.hypot(q.w, q.x, q.y, q.z);
  if (!(norm > 0)) return { w: 1, x: 0, y: 0, z: 0 };
  return { w: q.w / norm, x: q.x / norm, y: q.y / norm, z: q.z / norm };
}

export function quaternionMultiply(a: Quaternion, b: Quaternion): Quaternion {
  return {
    w: a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
    x: a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
    y: a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
    z: a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w
  };
}

export function quaternionRotate(q: Quaternion, vector: Vec3): Vec3 {
  const u = { x: q.x, y: q.y, z: q.z };
  const uv = cross(u, vector);
  const uuv = cross(u, uv);
  return add(vector, add(scale(uv, 2 * q.w), scale(uuv, 2)));
}

export function quaternionInverseRotate(q: Quaternion, vector: Vec3): Vec3 {
  return quaternionRotate({ w: q.w, x: -q.x, y: -q.y, z: -q.z }, vector);
}

function bodyInertiaMultiply(body: RigidBodyState, vectorWorld: Vec3): Vec3 {
  const local = quaternionInverseRotate(body.orientation, vectorWorld);
  return quaternionRotate(body.orientation, {
    x: local.x * body.inertiaBody_kg_m2.x,
    y: local.y * body.inertiaBody_kg_m2.y,
    z: local.z * body.inertiaBody_kg_m2.z
  });
}

function bodyInverseInertiaMultiply(body: RigidBodyState, vectorWorld: Vec3): Vec3 {
  const local = quaternionInverseRotate(body.orientation, vectorWorld);
  return quaternionRotate(body.orientation, {
    x: local.x * body.inverseInertiaBody_kg_m2.x,
    y: local.y * body.inverseInertiaBody_kg_m2.y,
    z: local.z * body.inverseInertiaBody_kg_m2.z
  });
}

export function initializeRigidBody(description: RigidBodyDescription): RigidBodyState {
  const normalized = { ...description, orientation: quaternionNormalize(description.orientation) };
  const properties = massProperties(normalized);
  const inverseInertia = {
    x: 1 / properties.inertiaBody_kg_m2.x,
    y: 1 / properties.inertiaBody_kg_m2.y,
    z: 1 / properties.inertiaBody_kg_m2.z
  };
  const partial: RigidBodyState = {
    description: JSON.parse(JSON.stringify(normalized)) as RigidBodyDescription,
    position_m: { ...normalized.position_m },
    orientation: { ...normalized.orientation },
    linearVelocity_m_s: { ...normalized.linearVelocity_m_s },
    angularVelocity_rad_s: { ...normalized.angularVelocity_rad_s },
    angularMomentum_kg_m2_s: ZERO(),
    mass_kg: properties.mass_kg,
    inverseMass_kg: 1 / properties.mass_kg,
    inertiaBody_kg_m2: properties.inertiaBody_kg_m2,
    inverseInertiaBody_kg_m2: inverseInertia,
    netForce_N: ZERO(),
    netTorque_N_m: ZERO(),
    collisionImpulse_N_s: ZERO(),
    collisionAngularImpulse_N_m_s: ZERO(),
    contactCount: 0,
    maxPenetration_m: 0,
    quaternionNormError: 0
  };
  partial.angularMomentum_kg_m2_s = bodyInertiaMultiply(partial, partial.angularVelocity_rad_s);
  return partial;
}

export function initializeRigidBodies(descriptions: RigidBodyDescription[]): RigidBodyState[] {
  return descriptions.map(initializeRigidBody);
}

export function cloneRigidBodies(bodies: RigidBodyState[]): RigidBodyState[] {
  return bodies.map((body) => ({
    ...body,
    description: JSON.parse(JSON.stringify(body.description)) as RigidBodyDescription,
    position_m: { ...body.position_m }, orientation: { ...body.orientation },
    linearVelocity_m_s: { ...body.linearVelocity_m_s }, angularVelocity_rad_s: { ...body.angularVelocity_rad_s },
    angularMomentum_kg_m2_s: { ...body.angularMomentum_kg_m2_s }, inertiaBody_kg_m2: { ...body.inertiaBody_kg_m2 },
    inverseInertiaBody_kg_m2: { ...body.inverseInertiaBody_kg_m2 }, netForce_N: { ...body.netForce_N },
    netTorque_N_m: { ...body.netTorque_N_m }, collisionImpulse_N_s: { ...body.collisionImpulse_N_s }, collisionAngularImpulse_N_m_s: { ...body.collisionAngularImpulse_N_m_s }
  }));
}

export function boundingRadius(body: RigidBodyState | RigidBodyDescription): number {
  const description = "description" in body ? body.description : body;
  const d = description.dimensions_m;
  if (description.shape === "sphere") return d.x;
  if (description.shape === "box") return 0.5 * Math.hypot(d.x, d.y, d.z);
  if (description.shape === "cylinder") return Math.hypot(d.x, d.y / 2);
  return d.y / 2 + d.x;
}

function supportRadius(body: RigidBodyState, directionWorld: Vec3): number {
  const direction = normalize(directionWorld);
  const local = quaternionInverseRotate(body.orientation, direction);
  const d = body.description.dimensions_m;
  let support: Vec3;
  if (body.description.shape === "sphere") {
    support = scale(normalize(local), d.x);
  } else if (body.description.shape === "box") {
    support = { x: Math.sign(local.x || 1) * d.x / 2, y: Math.sign(local.y || 1) * d.y / 2, z: Math.sign(local.z || 1) * d.z / 2 };
  } else if (body.description.shape === "cylinder") {
    const radialLength = Math.hypot(local.x, local.z);
    support = {
      x: radialLength > 0 ? d.x * local.x / radialLength : 0,
      y: Math.sign(local.y || 1) * d.y / 2,
      z: radialLength > 0 ? d.x * local.z / radialLength : 0
    };
  } else {
    support = add(scale(normalize(local), d.x), { x: 0, y: Math.sign(local.y || 1) * d.y / 2, z: 0 });
  }
  return Math.max(0, dot(quaternionRotate(body.orientation, support), direction));
}

function velocityAt(body: RigidBodyState, offset: Vec3): Vec3 {
  return add(body.linearVelocity_m_s, cross(body.angularVelocity_rad_s, offset));
}

function effectiveAngularTerm(body: RigidBodyState, offset: Vec3, direction: Vec3): number {
  return dot(cross(bodyInverseInertiaMultiply(body, cross(offset, direction)), offset), direction);
}

function applyImpulse(body: RigidBodyState, impulse: Vec3, offset: Vec3): void {
  body.linearVelocity_m_s = add(body.linearVelocity_m_s, scale(impulse, body.inverseMass_kg));
  body.angularMomentum_kg_m2_s = add(body.angularMomentum_kg_m2_s, cross(offset, impulse));
  body.collisionAngularImpulse_N_m_s = add(body.collisionAngularImpulse_N_m_s, cross(offset, impulse));
  body.angularVelocity_rad_s = bodyInverseInertiaMultiply(body, body.angularMomentum_kg_m2_s);
  body.collisionImpulse_N_s = add(body.collisionImpulse_N_s, impulse);
}

function solvePlaneContact(body: RigidBodyState, normal: Vec3, offset: number): void {
  const radius = supportRadius(body, normal);
  const extreme = dot(normal, body.position_m) - radius;
  const penetration = offset - extreme;
  if (penetration <= 0) return;
  body.contactCount += 1;
  body.maxPenetration_m = Math.max(body.maxPenetration_m, penetration);
  body.position_m = add(body.position_m, scale(normal, penetration + 1e-10));
  const arm = scale(normal, -radius);
  let relativeVelocity = velocityAt(body, arm);
  const normalSpeed = dot(relativeVelocity, normal);
  if (normalSpeed >= 0) return;
  const restitution = -normalSpeed > 0.5 ? body.description.restitution : 0;
  const denominator = body.inverseMass_kg + effectiveAngularTerm(body, arm, normal);
  const normalImpulseMagnitude = -(1 + restitution) * normalSpeed / Math.max(denominator, 1e-15);
  applyImpulse(body, scale(normal, normalImpulseMagnitude), arm);

  relativeVelocity = velocityAt(body, arm);
  const tangentVelocity = sub(relativeVelocity, scale(normal, dot(relativeVelocity, normal)));
  const tangentSpeed = length(tangentVelocity);
  if (tangentSpeed <= 1e-12) return;
  const tangent = scale(tangentVelocity, 1 / tangentSpeed);
  const tangentDenominator = body.inverseMass_kg + effectiveAngularTerm(body, arm, tangent);
  const unclamped = -tangentSpeed / Math.max(tangentDenominator, 1e-15);
  const tangentImpulseMagnitude = Math.max(-body.description.friction * normalImpulseMagnitude, Math.min(body.description.friction * normalImpulseMagnitude, unclamped));
  applyImpulse(body, scale(tangent, tangentImpulseMagnitude), arm);
}

function solveBodyContact(a: RigidBodyState, b: RigidBodyState): void {
  const delta = sub(b.position_m, a.position_m);
  const distance = length(delta);
  const normal = distance > 1e-12 ? scale(delta, 1 / distance) : { x: 1, y: 0, z: 0 };
  const radiusA = boundingRadius(a);
  const radiusB = boundingRadius(b);
  const penetration = radiusA + radiusB - distance;
  if (penetration <= 0) return;
  a.contactCount += 1; b.contactCount += 1;
  a.maxPenetration_m = Math.max(a.maxPenetration_m, penetration);
  b.maxPenetration_m = Math.max(b.maxPenetration_m, penetration);

  const totalInverseMass = a.inverseMass_kg + b.inverseMass_kg;
  a.position_m = add(a.position_m, scale(normal, -penetration * a.inverseMass_kg / totalInverseMass));
  b.position_m = add(b.position_m, scale(normal, penetration * b.inverseMass_kg / totalInverseMass));
  const armA = scale(normal, radiusA);
  const armB = scale(normal, -radiusB);
  let relativeVelocity = sub(velocityAt(b, armB), velocityAt(a, armA));
  const normalSpeed = dot(relativeVelocity, normal);
  if (normalSpeed >= 0) return;
  const restitution = -normalSpeed > 0.5 ? Math.min(a.description.restitution, b.description.restitution) : 0;
  const denominator = totalInverseMass + effectiveAngularTerm(a, armA, normal) + effectiveAngularTerm(b, armB, normal);
  const normalImpulseMagnitude = -(1 + restitution) * normalSpeed / Math.max(denominator, 1e-15);
  applyImpulse(a, scale(normal, -normalImpulseMagnitude), armA);
  applyImpulse(b, scale(normal, normalImpulseMagnitude), armB);

  relativeVelocity = sub(velocityAt(b, armB), velocityAt(a, armA));
  const tangentVelocity = sub(relativeVelocity, scale(normal, dot(relativeVelocity, normal)));
  const tangentSpeed = length(tangentVelocity);
  if (tangentSpeed <= 1e-12) return;
  const tangent = scale(tangentVelocity, 1 / tangentSpeed);
  const tangentDenominator = totalInverseMass + effectiveAngularTerm(a, armA, tangent) + effectiveAngularTerm(b, armB, tangent);
  const unclamped = -tangentSpeed / Math.max(tangentDenominator, 1e-15);
  const friction = Math.sqrt(a.description.friction * b.description.friction);
  const tangentImpulseMagnitude = Math.max(-friction * normalImpulseMagnitude, Math.min(friction * normalImpulseMagnitude, unclamped));
  applyImpulse(a, scale(tangent, -tangentImpulseMagnitude), armA);
  applyImpulse(b, scale(tangent, tangentImpulseMagnitude), armB);
}

export function rigidDiagnostics(bodies: RigidBodyState[], gravity: Vec3): RigidStepDiagnostics {
  let contacts = 0;
  let penetration = 0;
  let kinetic = 0;
  let potential = 0;
  let linearMomentum = ZERO();
  let angularMomentum = ZERO();
  let nanCount = 0;
  let quaternionError = 0;
  for (const body of bodies) {
    contacts += body.contactCount;
    penetration = Math.max(penetration, body.maxPenetration_m);
    linearMomentum = add(linearMomentum, scale(body.linearVelocity_m_s, body.mass_kg));
    angularMomentum = add(angularMomentum, add(body.angularMomentum_kg_m2_s, cross(body.position_m, scale(body.linearVelocity_m_s, body.mass_kg))));
    kinetic += 0.5 * body.mass_kg * dot(body.linearVelocity_m_s, body.linearVelocity_m_s) + 0.5 * dot(body.angularVelocity_rad_s, body.angularMomentum_kg_m2_s);
    potential += body.mass_kg * Math.max(0, -gravity.y) * body.position_m.y;
    quaternionError = Math.max(quaternionError, body.quaternionNormError);
    const scalars = [body.position_m.x, body.position_m.y, body.position_m.z, body.linearVelocity_m_s.x, body.linearVelocity_m_s.y, body.linearVelocity_m_s.z, body.orientation.w, body.orientation.x, body.orientation.y, body.orientation.z, kinetic, potential];
    nanCount += scalars.filter((value) => !Number.isFinite(value)).length;
  }
  return { contactCount: contacts, maxPenetration_m: penetration, kineticEnergy_J: kinetic, potentialEnergy_J: potential, linearMomentum_kg_m_s: linearMomentum, angularMomentum_kg_m2_s: angularMomentum, nanCount, quaternionMaxNormError: quaternionError };
}

export function advanceRigidBodies(bodies: RigidBodyState[], scene: Pick<SceneDescription, "container" | "fluid">, dt: number, collisionIterations = 6): RigidStepDiagnostics {
  if (!(dt > 0) || !Number.isFinite(dt)) throw new Error("Rigid-body time step must be finite and positive");
  for (const body of bodies) {
    body.contactCount = 0; body.maxPenetration_m = 0; body.quaternionNormError = 0;
    body.collisionImpulse_N_s = ZERO(); body.collisionAngularImpulse_N_m_s = ZERO(); body.netTorque_N_m = ZERO();
    body.netForce_N = scale(scene.fluid.gravity_m_s2, body.mass_kg);
    body.linearVelocity_m_s = add(body.linearVelocity_m_s, scale(scene.fluid.gravity_m_s2, dt));
    body.position_m = add(body.position_m, scale(body.linearVelocity_m_s, dt));
    body.angularVelocity_rad_s = bodyInverseInertiaMultiply(body, body.angularMomentum_kg_m2_s);
    const omegaQuaternion = { w: 0, ...body.angularVelocity_rad_s };
    const derivative = quaternionMultiply(omegaQuaternion, body.orientation);
    const unnormalized = {
      w: body.orientation.w + 0.5 * dt * derivative.w,
      x: body.orientation.x + 0.5 * dt * derivative.x,
      y: body.orientation.y + 0.5 * dt * derivative.y,
      z: body.orientation.z + 0.5 * dt * derivative.z
    };
    body.quaternionNormError = Math.abs(Math.hypot(unnormalized.w, unnormalized.x, unnormalized.y, unnormalized.z) - 1);
    body.orientation = quaternionNormalize(unnormalized);
    body.angularVelocity_rad_s = bodyInverseInertiaMultiply(body, body.angularMomentum_kg_m2_s);
  }

  const c = scene.container;
  const planes: Array<[Vec3, number]> = [
    [{ x: 1, y: 0, z: 0 }, -c.width_m / 2], [{ x: -1, y: 0, z: 0 }, -c.width_m / 2],
    [{ x: 0, y: 0, z: 1 }, -c.depth_m / 2], [{ x: 0, y: 0, z: -1 }, -c.depth_m / 2],
    [{ x: 0, y: 1, z: 0 }, 0]
  ];
  if (c.top === "closed") planes.push([{ x: 0, y: -1, z: 0 }, -c.height_m]);

  for (let iteration = 0; iteration < collisionIterations; iteration += 1) {
    for (const body of bodies) for (const [normal, offset] of planes) solvePlaneContact(body, normal, offset);
    for (let i = 0; i < bodies.length; i += 1) for (let j = i + 1; j < bodies.length; j += 1) solveBodyContact(bodies[i], bodies[j]);
  }
  for (const body of bodies) {
    body.netForce_N = add(body.netForce_N, scale(body.collisionImpulse_N_s, 1 / dt));
    body.netTorque_N_m = scale(body.collisionAngularImpulse_N_m_s, 1 / dt);
  }
  return rigidDiagnostics(bodies, scene.fluid.gravity_m_s2);
}

export function createBodyDescription(shape: RigidShape, index: number, containerHeight: number): RigidBodyDescription {
  const radius = 0.075;
  const dimensions = shape === "box" ? { x: 0.15, y: 0.12, z: 0.13 } : shape === "sphere" ? { x: radius, y: radius, z: radius } : { x: radius, y: 0.14, z: radius };
  return {
    id: `body-${shape}-${index}`,
    name: `${shape[0].toUpperCase()}${shape.slice(1)} ${index}`,
    shape,
    dimensions_m: dimensions,
    density_kg_m3: shape === "sphere" ? 650 : 1100,
    position_m: { x: 0, y: containerHeight + 0.24 + index * 0.025, z: 0 },
    orientation: { w: 1, x: 0, y: 0, z: 0 },
    linearVelocity_m_s: ZERO(), angularVelocity_rad_s: { x: 0.4, y: 0.8, z: 1.1 },
    restitution: 0.3, friction: 0.45
  };
}
