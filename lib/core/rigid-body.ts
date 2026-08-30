import { add, cross, dot, length, normalize, scale, sub } from "./math";
import type { Quaternion, RigidBodyDescription, RigidShape, SceneDescription, Vec3 } from "./model";
import { sceneShape } from "./scene-shape";
import { sceneHasTerrain, terrainHeightAt, terrainNormalAt } from "./terrain";

export interface MassProperties {
  volume_m3: number;
  mass_kg: number;
  inertiaBody_kg_m2: Vec3;
}

export interface RigidBodyState {
  description: RigidBodyDescription;
  /**
   * In the user's hand: the pose is commanded, not integrated.
   *
   * Distinct from `motion === "static"`, which is what the body *is*. This is
   * what is being done to it, so it lives on the runtime state rather than on
   * the description and never reaches the document. Consumers must keep reading
   * its displaced volume — a carried cup is dipped precisely so the water sees
   * it — and must stop applying gravity, contacts and pair impulses to it, or a
   * body held still would sink out of the hand holding it.
   */
  held?: boolean;
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
  buoyantForce_N: Vec3;
  hydrodynamicForce_N: Vec3;
  hydrodynamicTorque_N_m: Vec3;
  displacedFluidVolume_m3: number;
}

export interface RigidExternalLoad {
  force_N: Vec3;
  torque_N_m: Vec3;
  buoyantForce_N?: Vec3;
  hydrodynamicForce_N?: Vec3;
  displacedFluidVolume_m3?: number;
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

/** Displaced volume, from the one table that declares what each shape is. */
export function primitiveVolume(shape: RigidShape, dimensions: Vec3): number {
  return sceneShape(shape).volume_m3(dimensions);
}

export function massProperties(body: RigidBodyDescription): MassProperties {
  const { shape, dimensions_m: d, density_kg_m3: density } = body;
  const kind = sceneShape(shape);
  const volume = kind.volume_m3(d);
  return {
    volume_m3: volume,
    mass_kg: density * volume,
    // Density rather than mass, because a composite shape — the capsule, the
    // cup — weighs each of its parts separately and folding that into a
    // per-mass ratio reorders the arithmetic. Blessed trajectories here are
    // compared bit-identically, so the shape arm keeps the operation order the
    // integrator was blessed with.
    inertiaBody_kg_m2: kind.inertia_kg_m2(d, density),
  };
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
  const fixed = normalized.motion === "static";
  const inverseInertia = {
    x: fixed ? 0 : 1 / properties.inertiaBody_kg_m2.x,
    y: fixed ? 0 : 1 / properties.inertiaBody_kg_m2.y,
    z: fixed ? 0 : 1 / properties.inertiaBody_kg_m2.z
  };
  const partial: RigidBodyState = {
    description: JSON.parse(JSON.stringify(normalized)) as RigidBodyDescription,
    position_m: { ...normalized.position_m },
    orientation: { ...normalized.orientation },
    linearVelocity_m_s: { ...normalized.linearVelocity_m_s },
    angularVelocity_rad_s: { ...normalized.angularVelocity_rad_s },
    angularMomentum_kg_m2_s: ZERO(),
    mass_kg: properties.mass_kg,
    inverseMass_kg: fixed ? 0 : 1 / properties.mass_kg,
    inertiaBody_kg_m2: properties.inertiaBody_kg_m2,
    inverseInertiaBody_kg_m2: inverseInertia,
    netForce_N: ZERO(),
    netTorque_N_m: ZERO(),
    collisionImpulse_N_s: ZERO(),
    collisionAngularImpulse_N_m_s: ZERO(),
    contactCount: 0,
    maxPenetration_m: 0,
    quaternionNormError: 0
    , buoyantForce_N: ZERO(), hydrodynamicForce_N: ZERO(), hydrodynamicTorque_N_m: ZERO(), displacedFluidVolume_m3: 0
  };
  partial.angularMomentum_kg_m2_s = bodyInertiaMultiply(partial, partial.angularVelocity_rad_s);
  return partial;
}

export function initializeRigidBodies(descriptions: RigidBodyDescription[]): RigidBodyState[] {
  return descriptions.map(initializeRigidBody);
}

/**
 * A body's description by value, in a fixed field order.
 *
 * Documents are replaced rather than mutated, so a body nobody touched still
 * arrives as a fresh object on every edit. Comparing by reference would
 * re-derive every body's mass properties each time and discard the contact and
 * force state each one is carrying.
 *
 * The orientation is normalized first because `initializeRigidBody` normalizes
 * the description it retains: an authored quaternion and the one a live body is
 * holding describe the same rotation while differing in the last bit, and
 * without this every body would read as edited on every commit.
 */
function describedBody(description: RigidBodyDescription): string {
  return JSON.stringify([
    description.shape, description.dimensions_m, description.density_kg_m3,
    description.position_m, quaternionNormalize(description.orientation),
    description.linearVelocity_m_s, description.angularVelocity_rad_s,
    description.restitution, description.friction, description.motion,
  ]);
}

/**
 * Take an edited roster onto a running simulation.
 *
 * A body that is still in the document keeps the state it has reached, which is
 * the whole point: dropping a second sphere must not teleport the first one
 * back to where it was authored two seconds ago. A body that is new starts from
 * its description, and one that is gone goes.
 *
 * When a description itself changed, the mass properties are re-derived from it
 * while the run's pose and velocity are carried across — so a resized crate
 * stays where it fell instead of snapping back to where it was dropped. The
 * document's authored pose is restored onto the result afterwards because that
 * is what the GPU rigid system signs a body with; only the simulated state
 * above came from the run.
 */
export function adoptRigidBodyRoster(
  live: readonly RigidBodyState[],
  descriptions: readonly RigidBodyDescription[],
): RigidBodyState[] {
  const byId = new Map(live.map((body) => [body.description.id, body]));
  return descriptions.map((description) => {
    const previous = byId.get(description.id);
    if (!previous) return initializeRigidBody(description);
    if (describedBody(previous.description) === describedBody(description)) return previous;
    const adopted = initializeRigidBody({
      ...description,
      position_m: previous.position_m,
      orientation: previous.orientation,
      linearVelocity_m_s: previous.linearVelocity_m_s,
      angularVelocity_rad_s: previous.angularVelocity_rad_s,
    });
    adopted.description = JSON.parse(JSON.stringify(description)) as RigidBodyDescription;
    return adopted;
  });
}

export function cloneRigidBodies(bodies: RigidBodyState[]): RigidBodyState[] {
  return bodies.map((body) => ({
    ...body,
    description: JSON.parse(JSON.stringify(body.description)) as RigidBodyDescription,
    position_m: { ...body.position_m }, orientation: { ...body.orientation },
    linearVelocity_m_s: { ...body.linearVelocity_m_s }, angularVelocity_rad_s: { ...body.angularVelocity_rad_s },
    angularMomentum_kg_m2_s: { ...body.angularMomentum_kg_m2_s }, inertiaBody_kg_m2: { ...body.inertiaBody_kg_m2 },
    inverseInertiaBody_kg_m2: { ...body.inverseInertiaBody_kg_m2 }, netForce_N: { ...body.netForce_N },
    netTorque_N_m: { ...body.netTorque_N_m }, collisionImpulse_N_s: { ...body.collisionImpulse_N_s }, collisionAngularImpulse_N_m_s: { ...body.collisionAngularImpulse_N_m_s },
    buoyantForce_N: { ...body.buoyantForce_N }, hydrodynamicForce_N: { ...body.hydrodynamicForce_N }, hydrodynamicTorque_N_m: { ...body.hydrodynamicTorque_N_m }
  }));
}

export function boundingRadius(body: RigidBodyState | RigidBodyDescription): number {
  const description = "description" in body ? body.description : body;
  return sceneShape(description.shape).boundingRadius_m(description.dimensions_m);
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
  } else if (body.description.shape === "cylinder" || body.description.shape === "cup") {
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

export function advanceRigidBodies(bodies: RigidBodyState[], scene: Pick<SceneDescription, "container" | "fluid" | "terrain">, dt: number, collisionIterations = 6, externalLoads?: ReadonlyMap<string, RigidExternalLoad>): RigidStepDiagnostics {
  if (!(dt > 0) || !Number.isFinite(dt)) throw new Error("Rigid-body time step must be finite and positive");
  for (const body of bodies) {
    body.contactCount = 0; body.maxPenetration_m = 0; body.quaternionNormError = 0;
    body.collisionImpulse_N_s = ZERO(); body.collisionAngularImpulse_N_m_s = ZERO();
    const load = externalLoads?.get(body.description.id);
    body.buoyantForce_N = load?.buoyantForce_N ? { ...load.buoyantForce_N } : ZERO();
    body.hydrodynamicForce_N = load?.hydrodynamicForce_N ? { ...load.hydrodynamicForce_N } : ZERO();
    body.hydrodynamicTorque_N_m = load?.torque_N_m ? { ...load.torque_N_m } : ZERO();
    body.displacedFluidVolume_m3 = load?.displacedFluidVolume_m3 ?? 0;
    // A held body is on the hand's clock, not gravity's. It keeps the displaced
    // volume assigned above — a carried cup still pushes water aside — but it
    // integrates nothing, and the contact pass below rolls back anything that
    // manages to shove it.
    if (body.description.motion === "static" || body.held) {
      body.netForce_N = ZERO(); body.netTorque_N_m = ZERO(); body.linearVelocity_m_s = ZERO(); body.angularVelocity_rad_s = ZERO(); body.angularMomentum_kg_m2_s = ZERO();
      continue;
    }
    body.netForce_N = add(scale(scene.fluid.gravity_m_s2, body.mass_kg), load?.force_N ?? ZERO());
    body.netTorque_N_m = load?.torque_N_m ? { ...load.torque_N_m } : ZERO();
    body.linearVelocity_m_s = add(body.linearVelocity_m_s, scale(body.netForce_N, body.inverseMass_kg * dt));
    if (load) body.angularMomentum_kg_m2_s = add(body.angularMomentum_kg_m2_s, scale(load.torque_N_m, dt));
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
  const terrain = sceneHasTerrain(scene) ? scene.terrain : undefined;

  // Where the hand put them. The contact solver has no notion of an immovable
  // body — a plane pushes anything it penetrates, and a pair correction moves
  // both — so a held body's pose is taken back afterwards rather than defended
  // inside every contact. It still pushes its neighbours; nothing pushes back.
  const heldPoses = bodies
    .filter((body) => body.held)
    .map((body) => ({ body, position_m: { ...body.position_m }, orientation: { ...body.orientation } }));

  for (let iteration = 0; iteration < collisionIterations; iteration += 1) {
    for (const body of bodies) {
      for (const [normal, offset] of planes) solvePlaneContact(body, normal, offset);
      if (terrain) {
        // Local tangent-plane contact against the ground heightfield, sampled
        // under the body each iteration so it tracks the slope as it moves.
        const normal = terrainNormalAt(terrain, body.position_m.x, body.position_m.z);
        const surfaceY = terrainHeightAt(terrain, body.position_m.x, body.position_m.z);
        solvePlaneContact(body, normal, normal.x * body.position_m.x + normal.y * surfaceY + normal.z * body.position_m.z);
      }
    }
    for (let i = 0; i < bodies.length; i += 1) for (let j = i + 1; j < bodies.length; j += 1) solveBodyContact(bodies[i], bodies[j]);
  }
  for (const held of heldPoses) {
    held.body.position_m = held.position_m; held.body.orientation = held.orientation;
    held.body.linearVelocity_m_s = ZERO(); held.body.angularVelocity_rad_s = ZERO(); held.body.angularMomentum_kg_m2_s = ZERO();
  }
  for (const body of bodies) {
    body.netForce_N = add(body.netForce_N, scale(body.collisionImpulse_N_s, 1 / dt));
    body.netTorque_N_m = add(body.netTorque_N_m, scale(body.collisionAngularImpulse_N_m_s, 1 / dt));
  }
  return rigidDiagnostics(bodies, scene.fluid.gravity_m_s2);
}

/**
 * A cup large enough to scoop with, and light enough to float empty.
 *
 * Sized against the default 1.2 x 0.8 x 0.8 m tank rather than against the
 * other primitives: a cup is an instrument, so it is the water it can lift —
 * about nine litres here — that has to read, not its footprint next to a
 * sphere. The 50 mm wall is two cells at the 25 mm lattice a cup scene wants;
 * see `cupResolutionAdvice`, which is what the editor states when a scene
 * carries a cup its cell size cannot hold water in.
 */
const CUP_DIMENSIONS_M: Vec3 = { x: 0.15, y: 0.26, z: 0.05 };

/**
 * The size a shape arrives at, before anything resizes it.
 *
 * Split out of `createBodyDescription` because the editor now *shows* these:
 * the strip's placement row draws the next body's own dimensions, and it has to
 * be able to ask what they are without making a body to read them off. One
 * answer either way, so a default changed here changes what the row says.
 */
export function defaultBodyDimensions_m(shape: RigidShape): Vec3 {
  const radius = 0.075;
  return shape === "cup" ? { ...CUP_DIMENSIONS_M }
    : shape === "box" ? { x: 0.15, y: 0.12, z: 0.13 }
    : shape === "sphere" ? { x: radius, y: radius, z: radius }
    : { x: radius, y: 0.14, z: radius };
}

export function createBodyDescription(shape: RigidShape, index: number, containerHeight: number): RigidBodyDescription {
  const dimensions = defaultBodyDimensions_m(shape);
  return {
    id: `body-${shape}-${index}`,
    name: `${shape[0].toUpperCase()}${shape.slice(1)} ${index}`,
    shape,
    dimensions_m: dimensions,
    // A cup has to float when empty and sink when full, or dipping it is a
    // gesture with one outcome. Six hundred puts the empty shell well above
    // the surface and the filled one just under it.
    density_kg_m3: shape === "sphere" ? 650 : shape === "cup" ? 600 : 1100,
    position_m: { x: 0, y: containerHeight + 0.24 + index * 0.025, z: 0 },
    orientation: { w: 1, x: 0, y: 0, z: 0 },
    // A cup enters the scene upright. Spinning it is the one thing that empties
    // it before the user has touched it.
    linearVelocity_m_s: ZERO(),
    angularVelocity_rad_s: shape === "cup" ? ZERO() : { x: 0.4, y: 0.8, z: 1.1 },
    restitution: 0.3, friction: 0.45
  };
}
