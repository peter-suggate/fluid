import {
  cloneRigidBodies,
  initializeRigidBodies,
  type RigidBodyState,
  type RigidExternalLoad,
} from "../../../core/rigid-body";
import { quaternionInverseRotate } from "../../../core/rigid-body";
import { boundingRadius, primitiveVolume } from "../../../core/rigid-body";
import type { RigidBodyDescription, Vec3 } from "../../../core/model";
import { rigidMotionLane } from "../../../core/webgpu-rigid-body";
import { sceneCellSizes_m } from "../../../core/scene-lattice";
import { sceneShape, sceneShapeCode } from "../../../core/scene-shape";
import { sampleSolidWorld, type SolidWorld } from "../../../core/solid-world";
import type { SliceRigidPose } from "./slice-dynamic-geometry";
import type { SliceNumericalFields, SliceNumericalTopology } from "./slice-stage-numerics";
import type { SliceSceneSeed } from "./slice-scene-seed";

/**
 * CPU authority for production rigid poses used by the centre-plane model.
 * The roster, packed state/exchange layouts, shape dispatch, load expressions,
 * mass/inertia, gravity, six contact iterations, restitution/friction and
 * body-pair ordering mirror the live GPU authority.
 */
export interface SliceRigidAuthority {
  readonly previous: readonly RigidBodyState[];
  readonly current: RigidBodyState[];
  readonly loads: ReadonlyMap<string, RigidExternalLoad>;
  readonly exchange: readonly Int32Array[];
}

export function createSliceRigidAuthority(seed: SliceSceneSeed): SliceRigidAuthority {
  const bodies = initializeRigidBodies(
    [...(seed.production?.scene.rigidBodies ?? [])].slice(0, 12),
  );
  return { previous: cloneRigidBodies(bodies), current: bodies, loads: new Map(),
    exchange: bodies.map(() => new Int32Array(12)) };
}

export interface AdvanceSliceRigidOptions {
  /** Reaction loads from the accepted fluid frame, keyed by authored body id. */
  readonly externalLoads?: ReadonlyMap<string, RigidExternalLoad>;
}

export function advanceSliceRigidAuthority(
  authority: SliceRigidAuthority,
  seed: SliceSceneSeed,
  dt: number,
  options: AdvanceSliceRigidOptions = {},
): SliceRigidAuthority {
  const scene = seed.production?.scene;
  if (!scene) return authority;
  const previous = cloneRigidBodies(authority.current);
  const cellSize = sceneCellSizes_m(scene);
  const packed = authority.current.map(body => packSliceRigidBody(body, seed.densityKgM3));
  const next = stepSliceRigidBodies({ bodies: packed,
    descriptions: authority.current.map(body => body.description),
    exchange: authority.exchange, dt, densityKgM3: seed.densityKgM3,
    gravity: scene.fluid.gravity_m_s2,
    cellVolumeM3: cellSize[0] * cellSize[1] * cellSize[2], snapshotCount: 1,
    solidWorld: seed.production?.solidWorld,
    solidOrigin_m: [seed.viewport.originX, seed.viewport.originY,
      -0.5 * scene.container.depth_m],
    solidCellSize_m: [cellSize[0], cellSize[1], cellSize[2]] });
  next.forEach((record, index) => applyPackedSliceRigidBody(authority.current[index]!, record));
  return { previous, current: authority.current, loads: options.externalLoads ?? authority.loads,
    exchange: authority.current.map(() => new Int32Array(12)) };
}

const f = Math.fround;
type FVec3 = readonly [number, number, number];
const fv = (x = 0, y = 0, z = 0): FVec3 => [f(x), f(y), f(z)];
const vadd = (a: FVec3, b: FVec3): FVec3 =>
  fv(f(a[0] + b[0]), f(a[1] + b[1]), f(a[2] + b[2]));
const vsub = (a: FVec3, b: FVec3): FVec3 =>
  fv(f(a[0] - b[0]), f(a[1] - b[1]), f(a[2] - b[2]));
const vmul = (a: FVec3, scalar: number): FVec3 =>
  fv(f(a[0] * scalar), f(a[1] * scalar), f(a[2] * scalar));
const vdot = (a: FVec3, b: FVec3): number =>
  f(f(f(a[0] * b[0]) + f(a[1] * b[1])) + f(a[2] * b[2]));
const vcross = (a: FVec3, b: FVec3): FVec3 => fv(
  f(f(a[1] * b[2]) - f(a[2] * b[1])),
  f(f(a[2] * b[0]) - f(a[0] * b[2])),
  f(f(a[0] * b[1]) - f(a[1] * b[0])),
);
const vlength = (a: FVec3): number => f(Math.sqrt(vdot(a, a)));

function qmul(a: readonly [number, number, number, number],
  b: readonly [number, number, number, number]): [number, number, number, number] {
  const av = fv(a[1], a[2], a[3]), bv = fv(b[1], b[2], b[3]);
  const vector = vadd(vadd(vmul(bv, a[0]), vmul(av, b[0])), vcross(av, bv));
  return [f(f(a[0] * b[0]) - vdot(av, bv)), vector[0], vector[1], vector[2]];
}

function qrotate(q: readonly [number, number, number, number], value: FVec3): FVec3 {
  const vector = fv(q[1], q[2], q[3]), uv = vcross(vector, value);
  return vadd(value, vmul(vadd(vmul(uv, q[0]), vcross(vector, uv)), 2));
}

function inverseInertia(record: Float32Array, value: FVec3, density: number): FVec3 {
  const q = [record[8]!, record[9]!, record[10]!, record[11]!] as const;
  const local = qrotate([q[0], f(-q[1]), f(-q[2]), f(-q[3])], value);
  return qrotate(q, fv(
    f(local[0] * record[21]! / Math.max(density, 1e-9)),
    f(local[1] * record[22]! / Math.max(density, 1e-9)),
    f(local[2] * record[23]! / Math.max(density, 1e-9)),
  ));
}

export function packSliceRigidBody(body: RigidBodyState, densityKgM3: number): Float32Array {
  const result = new Float32Array(32), d = body.description.dimensions_m;
  result.set([body.position_m.x, body.position_m.y, body.position_m.z,
    sceneShapeCode(body.description.shape), d.x, d.y, d.z, boundingRadius(body.description),
    body.orientation.w, body.orientation.x, body.orientation.y, body.orientation.z,
    body.linearVelocity_m_s.x, body.linearVelocity_m_s.y, body.linearVelocity_m_s.z,
    (body.held ? 0 : body.inverseMass_kg) * densityKgM3,
    body.angularVelocity_rad_s.x, body.angularVelocity_rad_s.y, body.angularVelocity_rad_s.z,
    body.description.density_kg_m3,
    (body.held ? 0 : body.inverseMass_kg) * densityKgM3,
    (body.held ? 0 : body.inverseInertiaBody_kg_m2.x) * densityKgM3,
    (body.held ? 0 : body.inverseInertiaBody_kg_m2.y) * densityKgM3,
    (body.held ? 0 : body.inverseInertiaBody_kg_m2.z) * densityKgM3,
    body.angularMomentum_kg_m2_s.x, body.angularMomentum_kg_m2_s.y,
    body.angularMomentum_kg_m2_s.z, body.description.restitution,
    body.description.friction, 0, rigidMotionLane(body), 0]);
  return result;
}

function applyPackedSliceRigidBody(body: RigidBodyState, record: Float32Array): void {
  body.position_m = { x: record[0]!, y: record[1]!, z: record[2]! };
  body.orientation = { w: record[8]!, x: record[9]!, y: record[10]!, z: record[11]! };
  body.linearVelocity_m_s = { x: record[12]!, y: record[13]!, z: record[14]! };
  body.angularVelocity_rad_s = { x: record[16]!, y: record[17]!, z: record[18]! };
  body.angularMomentum_kg_m2_s = { x: record[24]!, y: record[25]!, z: record[26]! };
}

export interface StepSliceRigidBodiesInput {
  readonly bodies: readonly Float32Array[];
  readonly descriptions: readonly RigidBodyDescription[];
  /** Exact 12-lane signed integer exchange records consumed by the GPU step. */
  readonly exchange?: readonly Int32Array[];
  readonly dt: number;
  readonly densityKgM3: number;
  readonly gravity: Vec3;
  readonly cellVolumeM3: number;
  readonly snapshotCount?: number;
  readonly solidWorld?: SolidWorld;
  readonly solidOrigin_m?: FVec3;
  readonly solidCellSize_m?: FVec3;
}

function supportRadiusPacked(body: Float32Array, description: RigidBodyDescription,
  direction: FVec3): number {
  const length = vlength(direction), normalized = length > 1e-8
    ? vmul(direction, f(1 / length)) : fv(1, 0, 0);
  const q = [body[8]!, body[9]!, body[10]!, body[11]!] as const;
  const local = qrotate([q[0], f(-q[1]), f(-q[2]), f(-q[3])], normalized);
  return f(sceneShape(description.shape).supportRadius_m(
    { x: body[4]!, y: body[5]!, z: body[6]! },
    { x: local[0], y: local[1], z: local[2] }));
}

function velocityAtPacked(body: Float32Array, arm: FVec3): FVec3 {
  return vadd(fv(body[12], body[13], body[14]),
    vcross(fv(body[16], body[17], body[18]), arm));
}

function angularTermPacked(body: Float32Array, arm: FVec3, direction: FVec3,
  density: number): number {
  return vdot(vcross(inverseInertia(body, vcross(arm, direction), density), arm), direction);
}

function applyImpulsePacked(body: Float32Array, impulse: FVec3, arm: FVec3,
  density: number): void {
  const inverseMass = f(body[20]! / Math.max(density, 1e-9));
  body.set(vadd(fv(body[12], body[13], body[14]), vmul(impulse, inverseMass)), 12);
  const angularMomentum = vadd(fv(body[24], body[25], body[26]), vcross(arm, impulse));
  body.set(angularMomentum, 24);
  body.set(inverseInertia(body, angularMomentum, density), 16);
}

function resolveStaticContact(body: Float32Array, normal: FVec3, penetration: number,
  radius: number, density: number): void {
  const inverseMass = f(body[20]! / Math.max(density, 1e-9));
  if (!(inverseMass > 0 && penetration > 0)) return;
  const position = vadd(fv(body[0], body[1], body[2]),
    vmul(normal, f(penetration + 1e-7)));
  body.set(position, 0);
  const arm = vmul(normal, f(-radius));
  let relative = velocityAtPacked(body, arm), normalSpeed = vdot(relative, normal);
  if (normalSpeed >= 0) return;
  const restitution = -normalSpeed > 0.5 ? body[27]! : 0;
  const denominator = Math.max(f(inverseMass
    + angularTermPacked(body, arm, normal, density)), 1e-9);
  const normalMagnitude = f(f(-(1 + restitution) * normalSpeed) / denominator);
  applyImpulsePacked(body, vmul(normal, normalMagnitude), arm, density);
  relative = velocityAtPacked(body, arm);
  const tangentVelocity = vsub(relative, vmul(normal, vdot(relative, normal)));
  const tangentSpeed = vlength(tangentVelocity);
  if (tangentSpeed <= 1e-8) return;
  const tangent = vmul(tangentVelocity, f(1 / tangentSpeed));
  const tangentDenominator = Math.max(f(inverseMass
    + angularTermPacked(body, arm, tangent, density)), 1e-9);
  const tangentMagnitude = clamp(f(-tangentSpeed / tangentDenominator),
    f(-body[28]! * normalMagnitude), f(body[28]! * normalMagnitude));
  applyImpulsePacked(body, vmul(tangent, tangentMagnitude), arm, density);
}

function voxelInteriorContact(center: FVec3, minimum: FVec3, maximum: FVec3):
  { normal: FVec3; distance: number } {
  const distances = [f(center[0] - minimum[0]), f(maximum[0] - center[0]),
    f(center[1] - minimum[1]), f(maximum[1] - center[1]),
    f(center[2] - minimum[2]), f(maximum[2] - center[2])];
  const normals: FVec3[] = [fv(-1, 0, 0), fv(1, 0, 0), fv(0, -1, 0),
    fv(0, 1, 0), fv(0, 0, -1), fv(0, 0, 1)];
  let chosen = 0;
  for (let i = 1; i < distances.length; i += 1) if (distances[i]! < distances[chosen]!) chosen = i;
  return { normal: normals[chosen]!, distance: Math.max(0, distances[chosen]!) };
}

function normalize(value: FVec3, fallback: FVec3): FVec3 {
  const length = vlength(value);
  return length > 1e-8 ? vmul(value, f(1 / length)) : fallback;
}

function packedSolidWorldContact(body: Float32Array, description: RigidBodyDescription,
  world: SolidWorld, origin: FVec3, cellSize: FVec3, density: number): void {
  if (f(body[20]! / density) <= 0) return;
  const position = fv(body[0], body[1], body[2]), broadRadius = Math.max(body[7]!, 1e-7);
  let bestPenetration = 0, bestNormal = fv(0, 1, 0), bestRadius = 0;
  const consider = (q: readonly [number, number, number], fraction: number,
    minimum: FVec3, maximum: FVec3) => {
    const closest = fv(clamp(position[0], minimum[0], maximum[0]),
      clamp(position[1], minimum[1], maximum[1]),
      clamp(position[2], minimum[2], maximum[2]));
    const delta = vsub(position, closest), separation = vlength(delta);
    if (separation > broadRadius) return;
    let normal = fv(0, 1, 0), penetration = 0, radius = 0;
    if (fraction < 1) {
      const sdfQ8 = (x: number, y: number, z: number) => f(clamp(
        sampleSolidWorld(world, [x, y, z]).signedDistance_cells * 256, -512, 512));
      const sx = f(f(sdfQ8(q[0] + 1, q[1], q[2]) - sdfQ8(q[0] - 1, q[1], q[2]))
        / cellSize[0]);
      const sy = f(f(sdfQ8(q[0], q[1] + 1, q[2]) - sdfQ8(q[0], q[1] - 1, q[2]))
        / cellSize[1]);
      const sz = f(f(sdfQ8(q[0], q[1], q[2] + 1) - sdfQ8(q[0], q[1], q[2] - 1))
        / cellSize[2]);
      const centre = vmul(vadd(minimum, maximum), 0.5);
      normal = normalize(fv(sx, sy, sz), normalize(vsub(position, centre), fv(0, 1, 0)));
      const sdfMetres = f(f(clamp(sampleSolidWorld(world, q).signedDistance_cells * 256,
        -512, 512) / 256) * Math.min(...cellSize));
      const surface = vsub(centre, vmul(normal, sdfMetres));
      radius = supportRadiusPacked(body, description, normal);
      penetration = f(radius - f(vdot(normal, position) - vdot(normal, surface)));
    } else if (separation > 1e-7) {
      normal = vmul(delta, f(1 / separation));
      radius = supportRadiusPacked(body, description, normal);
      penetration = f(radius - separation);
    } else {
      const interior = voxelInteriorContact(position, minimum, maximum);
      normal = interior.normal; radius = supportRadiusPacked(body, description, normal);
      penetration = f(radius + interior.distance);
    }
    if (penetration > bestPenetration) {
      bestPenetration = penetration; bestNormal = normal; bestRadius = radius;
    }
  };
  for (const page of world.pages) {
    for (let voxel = 0; voxel < page.solidFraction.length; voxel += 1) {
      const fraction = page.solidFraction[voxel]! / 255;
      if (fraction <= 0) continue;
      const q = [page.coordinate[0] * 8 + (voxel & 7),
        page.coordinate[1] * 8 + ((voxel >> 3) & 7),
        page.coordinate[2] * 8 + (voxel >> 6)] as const;
      const minimum = fv(origin[0] + q[0] * cellSize[0],
        origin[1] + q[1] * cellSize[1], origin[2] + q[2] * cellSize[2]);
      const maximum = vadd(minimum, cellSize);
      if ([0, 1, 2].some(axis => position[axis]! + broadRadius < minimum[axis]!
        || position[axis]! - broadRadius > maximum[axis]!)) continue;
      consider(q, fraction, minimum, maximum);
    }
  }
  for (const region of world.regions ?? []) {
    if (region.operation !== "fill") continue;
    const minimum = fv(origin[0] + region.minimum[0] * cellSize[0],
      origin[1] + region.minimum[1] * cellSize[1],
      origin[2] + region.minimum[2] * cellSize[2]);
    const maximum = fv(origin[0] + region.maximumExclusive[0] * cellSize[0],
      origin[1] + region.maximumExclusive[1] * cellSize[1],
      origin[2] + region.maximumExclusive[2] * cellSize[2]);
    consider(region.minimum, 1, minimum, maximum);
  }
  if (bestPenetration > 0) resolveStaticContact(body, bestNormal, bestPenetration,
    bestRadius, density);
}

function solvePackedBodyPair(a: Float32Array, b: Float32Array, density: number): void {
  const inverseA = f(a[20]! / density), inverseB = f(b[20]! / density);
  const inverseTotal = f(inverseA + inverseB); if (inverseTotal <= 0) return;
  const delta = vsub(fv(b[0], b[1], b[2]), fv(a[0], a[1], a[2]));
  const distance = vlength(delta), normal = distance > 1e-8
    ? vmul(delta, f(1 / distance)) : fv(1, 0, 0);
  const radiusA = a[7]!, radiusB = b[7]!, penetration = f(radiusA + radiusB - distance);
  if (penetration <= 0) return;
  a.set(vsub(fv(a[0], a[1], a[2]), vmul(normal,
    f(penetration * inverseA / inverseTotal))), 0);
  b.set(vadd(fv(b[0], b[1], b[2]), vmul(normal,
    f(penetration * inverseB / inverseTotal))), 0);
  const armA = vmul(normal, radiusA), armB = vmul(normal, -radiusB);
  let relative = vsub(velocityAtPacked(b, armB), velocityAtPacked(a, armA));
  const normalSpeed = vdot(relative, normal);
  if (normalSpeed >= 0) return;
  const restitution = -normalSpeed > 0.5 ? Math.min(a[27]!, b[27]!) : 0;
  const denominator = Math.max(f(f(inverseTotal
    + angularTermPacked(a, armA, normal, density))
    + angularTermPacked(b, armB, normal, density)), 1e-9);
  const normalMagnitude = f(f(-(1 + restitution) * normalSpeed) / denominator);
  applyImpulsePacked(a, vmul(normal, -normalMagnitude), armA, density);
  applyImpulsePacked(b, vmul(normal, normalMagnitude), armB, density);
  relative = vsub(velocityAtPacked(b, armB), velocityAtPacked(a, armA));
  const tangentVelocity = vsub(relative, vmul(normal, vdot(relative, normal)));
  const tangentSpeed = vlength(tangentVelocity); if (tangentSpeed <= 1e-8) return;
  const tangent = vmul(tangentVelocity, f(1 / tangentSpeed));
  const friction = f(Math.sqrt(Math.max(0, f(a[28]! * b[28]!))));
  const tangentDenominator = Math.max(f(f(inverseTotal
    + angularTermPacked(a, armA, tangent, density))
    + angularTermPacked(b, armB, tangent, density)), 1e-9);
  const magnitude = clamp(f(-tangentSpeed / tangentDenominator),
    f(-friction * normalMagnitude), f(friction * normalMagnitude));
  applyImpulsePacked(a, vmul(tangent, -magnitude), armA, density);
  applyImpulsePacked(b, vmul(tangent, magnitude), armB, density);
}

/** Pure f32 image of the live GPU rigid integration kernel before contacts. */
export function stepSliceRigidBodies(input: StepSliceRigidBodiesInput): readonly Float32Array[] {
  const dt = f(input.dt), density = f(Math.max(input.densityKgM3, 1e-9));
  const snapshots = f(Math.max(input.snapshotCount ?? 1, 1));
  const gravity = fv(input.gravity.x, input.gravity.y, input.gravity.z);
  const previous = input.bodies.slice(0, 12).map(source => Float32Array.from(source));
  const result = input.bodies.slice(0, 12).map((source, index) => {
    const body = Float32Array.from(source), exchange = input.exchange?.[index] ?? new Int32Array(12);
    if (!(body[30]! > 0.5 && body[30]! < 1.5)) return body;
    const wet = f(f(exchange[6]! / 65536) / snapshots);
    const volume = f(primitiveVolume(input.descriptions[index]!.shape,
      { x: body[4]!, y: body[5]!, z: body[6]! }));
    const displaced = f(Math.min(Math.max(0, f(wet * f(input.cellVolumeM3))), volume));
    const impulse = fv(f(exchange[0]! * 1e-6), f(exchange[1]! * 1e-6),
      f(exchange[2]! * 1e-6));
    const angularImpulse = fv(f(exchange[3]! * 1e-6), f(exchange[4]! * 1e-6),
      f(exchange[5]! * 1e-6));
    const weighted = vmul(fv(f(exchange[7]! * 1e-4), f(exchange[8]! * 1e-4),
      f(exchange[9]! * 1e-4)), f(1 / snapshots));
    const velocityWeight = f(f(exchange[11]! / 65536) / snapshots);
    const mean = velocityWeight > 1e-8 ? vmul(weighted, f(1 / velocityWeight)) : fv();
    const mass = body[20]! > 0 ? f(density / body[20]!) : f(1e30);
    const immersed = f(clamp(f(displaced / Math.max(volume, 1e-9)), 0, 1));
    const velocity = fv(body[12], body[13], body[14]), relative = vsub(velocity, mean);
    const speed = vlength(relative);
    const dragScale = f(f(f(f(f(-0.5 * density) * 0.9) * f(Math.PI))
      * f(body[7]! * body[7]!)) * immersed * speed);
    const drag = vmul(relative, dragScale);
    const buoyancy = exchange[10] !== 0 ? fv() : vmul(gravity, f(-density * displaced));
    const added = f(f(0.5 * density) * displaced);
    const numerator = vadd(vadd(vmul(gravity, mass),
      vmul(impulse, f(1 / Math.max(dt, 1e-8)))), vadd(drag, buoyancy));
    const acceleration = vmul(numerator, f(1 / Math.max(f(mass + added), 1e-8)));
    const nextVelocity = vadd(velocity, vmul(acceleration, dt));
    body.set(nextVelocity, 12);
    body[24] = f(body[24]! + angularImpulse[0]);
    body[25] = f(body[25]! + angularImpulse[1]);
    body[26] = f(body[26]! + angularImpulse[2]);
    body[0] = f(body[0]! + f(nextVelocity[0] * dt));
    body[1] = f(body[1]! + f(nextVelocity[1] * dt));
    body[2] = f(body[2]! + f(nextVelocity[2] * dt));
    const omega = inverseInertia(body, fv(body[24], body[25], body[26]), density);
    body.set(omega, 16);
    const q = [body[8]!, body[9]!, body[10]!, body[11]!] as const;
    const derivative = qmul([0, omega[0], omega[1], omega[2]], q);
    const unnormalized = [f(q[0] + f(0.5 * dt * derivative[0])),
      f(q[1] + f(0.5 * dt * derivative[1])),
      f(q[2] + f(0.5 * dt * derivative[2])),
      f(q[3] + f(0.5 * dt * derivative[3]))] as const;
    const length = f(Math.sqrt(f(f(f(unnormalized[0] ** 2 + unnormalized[1] ** 2)
      + unnormalized[2] ** 2) + unnormalized[3] ** 2)));
    body[8] = f(unnormalized[0] / length); body[9] = f(unnormalized[1] / length);
    body[10] = f(unnormalized[2] / length); body[11] = f(unnormalized[3] / length);
    return body;
  });
  for (let iteration = 0; iteration < 6; iteration += 1) {
    if (input.solidWorld && input.solidOrigin_m && input.solidCellSize_m) {
      for (let index = 0; index < result.length; index += 1) {
        packedSolidWorldContact(result[index]!, input.descriptions[index]!, input.solidWorld,
          input.solidOrigin_m, input.solidCellSize_m, density);
      }
    }
    for (let a = 0; a < result.length; a += 1) for (let b = a + 1; b < result.length; b += 1) {
      solvePackedBodyPair(result[a]!, result[b]!, density);
    }
  }
  for (let index = 0; index < result.length; index += 1) {
    if (result[index]![30]! > 1.5) result[index] = previous[index]!;
  }
  return result;
}

export function sliceRigidPoses(
  bodies: readonly RigidBodyState[],
): readonly SliceRigidPose[] {
  return bodies.map(body => ({
    description: body.description,
    position_m: body.position_m,
    orientation: body.orientation,
    linearVelocity_m_s: body.linearVelocity_m_s,
    angularVelocity_rad_s: body.angularVelocity_rad_s,
  }));
}

export interface SliceRigidCouplingReceipt {
  readonly bodyId: string;
  readonly displacedVolume_m3: number;
  readonly meanFluidVelocity_m_s: { readonly x: number; readonly y: number; readonly z: number };
  readonly force_N: { readonly x: number; readonly y: number; readonly z: number };
}

const exchangeByLoads = new WeakMap<object, readonly Int32Array[]>();

const clamp = (value: number, low: number, high: number): number =>
  Math.max(low, Math.min(high, value));

/** WGSL `round`: nearest integer with exact half cases resolved to even. */
function roundTiesEven(value: number): number {
  const floor = Math.floor(value), fraction = value - floor;
  if (fraction < 0.5) return floor;
  if (fraction > 0.5) return floor + 1;
  return floor % 2 === 0 ? floor : floor + 1;
}

function bodyContains(body: RigidBodyState,
  point: { x: number; y: number; z: number }): boolean {
  const local = quaternionInverseRotate(body.orientation, {
    x: point.x - body.position_m.x, y: point.y - body.position_m.y,
    z: point.z - body.position_m.z,
  });
  return sceneShape(body.description.shape).inside(body.description.dimensions_m, local);
}

function cellBodyCoverage(seed: SliceSceneSeed, body: RigidBodyState,
  cell: SliceNumericalTopology["cells"][number]): number {
  const h = seed.viewport.sourceCellSize;
  let covered = 0, samples = 0;
  for (let y = Math.round(cell.center[1] - 0.5 * cell.widths[1]);
    y < Math.round(cell.center[1] + 0.5 * cell.widths[1]); y += 1) {
    for (let x = Math.round(cell.center[0] - 0.5 * cell.widths[0]);
      x < Math.round(cell.center[0] + 0.5 * cell.widths[0]); x += 1) {
      for (const dx of [-0.4, 0.4]) for (const dy of [-0.4, 0.4]) {
        const point = {
          x: seed.viewport.originX + (x + 0.5 + dx) * h,
          y: seed.viewport.originY + (y + 0.5 + dy) * h,
          z: seed.viewport.centerZ,
        };
        if (bodyContains(body, point)) covered += 1;
        samples += 1;
      }
    }
  }
  return samples ? covered / samples : 0;
}

/**
 * Dimensionally reduce production `coupleCells` to the centre slab. The same
 * strict greatest-coverage owner, wet factor and fixed-point exchange lanes are
 * used. Multiplying fine area by h^3 treats the slice as its containing source
 * voxel slab; this is the only dimensional reduction in the load receipt.
 */
export function sliceRigidCouplingLoads(
  seed: SliceSceneSeed,
  topology: SliceNumericalTopology,
  fields: SliceNumericalFields,
  bodies: readonly RigidBodyState[],
): { readonly loads: ReadonlyMap<string, RigidExternalLoad>;
  readonly receipts: readonly SliceRigidCouplingReceipt[];
  readonly exchange: readonly Int32Array[] } {
  const count = Math.min(12, bodies.length);
  const exchange = Array.from({ length: count }, () => new Int32Array(12));
  const h = seed.viewport.sourceCellSize;
  for (const cell of topology.cells) {
    let best = 0, owner = -1;
    for (let body = 0; body < count; body += 1) {
      const coverage = cellBodyCoverage(seed, bodies[body]!, cell);
      if (coverage > best) { best = coverage; owner = body; }
    }
    if (owner < 0 || best <= 0) continue;
    const open = fields.capacity[cell.id]!;
    const wet = clamp(fields.density[cell.id]! / Math.max(open, 0.125), 0, 1);
    if (wet <= 0) continue;
    const displacedWeight = wet * best * cell.area;
    const qVolume = roundTiesEven(displacedWeight * 65536);
    exchange[owner]![6] += qVolume;
    exchange[owner]![11] += qVolume;
    exchange[owner]![7] += roundTiesEven(displacedWeight
      * fields.cellVelocity[2 * cell.id]! * h * 10000);
    exchange[owner]![8] += roundTiesEven(displacedWeight
      * fields.cellVelocity[2 * cell.id + 1]! * h * 10000);
    // The omitted centre-slab solver carries no z velocity authority.
  }
  const density = seed.densityKgM3;
  const gravity = seed.production?.scene.fluid.gravity_m_s2 ?? { x: 0, y: 0, z: 0 };
  const loads = new Map<string, RigidExternalLoad>();
  const receipts: SliceRigidCouplingReceipt[] = [];
  for (let index = 0; index < count; index += 1) {
    const body = bodies[index]!;
    const displacedFine = exchange[index]![6]! / 65536;
    const displaced = Math.min(Math.max(0, displacedFine * h * h * h),
      primitiveVolume(body.description.shape, body.description.dimensions_m));
    const velocityWeight = exchange[index]![11]! / 65536;
    const mean = velocityWeight > 1e-8 ? {
      x: exchange[index]![7]! * 1e-4 / velocityWeight,
      y: exchange[index]![8]! * 1e-4 / velocityWeight,
      z: 0,
    } : { x: 0, y: 0, z: 0 };
    const relative = { x: body.linearVelocity_m_s.x - mean.x,
      y: body.linearVelocity_m_s.y - mean.y,
      z: body.linearVelocity_m_s.z - mean.z };
    const speed = Math.hypot(relative.x, relative.y, relative.z);
    const immersed = clamp(displaced / Math.max(
      primitiveVolume(body.description.shape, body.description.dimensions_m), 1e-9), 0, 1);
    const radius = boundingRadius(body.description);
    const dragScale = -0.5 * density * 0.9 * Math.PI * radius * radius * immersed * speed;
    const drag = { x: dragScale * relative.x, y: dragScale * relative.y,
      z: dragScale * relative.z };
    const buoyancy = { x: -density * displaced * gravity.x,
      y: -density * displaced * gravity.y, z: -density * displaced * gravity.z };
    const added = 0.5 * density * displaced;
    const mass = body.mass_kg;
    // Encode the production added-mass denominator as an equivalent external
    // force accepted by the shared CPU pose/contact integrator.
    const denominator = Math.max(mass + added, 1e-8);
    const desired = {
      x: (mass * gravity.x + drag.x + buoyancy.x) / denominator,
      y: (mass * gravity.y + drag.y + buoyancy.y) / denominator,
      z: (mass * gravity.z + drag.z + buoyancy.z) / denominator,
    };
    const force = { x: mass * (desired.x - gravity.x),
      y: mass * (desired.y - gravity.y), z: mass * (desired.z - gravity.z) };
    loads.set(body.description.id, { force_N: force, torque_N_m: { x: 0, y: 0, z: 0 },
      buoyantForce_N: buoyancy, hydrodynamicForce_N: drag,
      displacedFluidVolume_m3: displaced });
    receipts.push({ bodyId: body.description.id, displacedVolume_m3: displaced,
      meanFluidVelocity_m_s: mean, force_N: force });
  }
  exchangeByLoads.set(loads, exchange);
  return { loads, receipts, exchange };
}

export function withSliceRigidLoads(authority: SliceRigidAuthority,
  loads: ReadonlyMap<string, RigidExternalLoad>): SliceRigidAuthority {
  return { ...authority, loads, exchange: exchangeByLoads.get(loads) ?? authority.exchange };
}
