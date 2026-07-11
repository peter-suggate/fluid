import { add, cross, length, scale, sub } from "./math";
import type { SceneDescription, Vec3 } from "./model";
import { boundingRadius, primitiveVolume, quaternionInverseRotate, type RigidBodyState, type RigidExternalLoad } from "./rigid-body";

export interface CouplingFluid {
  sampleOccupancy(position: Vec3): number;
  sampleVelocity(position: Vec3): Vec3;
  samplePressure(position: Vec3): number;
  applyImpulseAt(position: Vec3, impulse_N_s: Vec3, radius_m?: number): boolean;
}

export interface CouplingDiagnostics {
  displacedVolume_m3: number;
  bodyImpulse_N_s: Vec3;
  fluidReactionImpulse_N_s: Vec3;
  momentumClosureError_N_s: number;
  coupledBodyCount: number;
}

function insidePrimitive(body: RigidBodyState, point: Vec3) {
  const p = quaternionInverseRotate(body.orientation, sub(point, body.position_m)), d = body.description.dimensions_m;
  if (body.description.shape === "sphere") return length(p) <= d.x;
  if (body.description.shape === "box") return Math.abs(p.x) <= d.x / 2 && Math.abs(p.y) <= d.y / 2 && Math.abs(p.z) <= d.z / 2;
  if (body.description.shape === "cylinder") return p.x * p.x + p.z * p.z <= d.x * d.x && Math.abs(p.y) <= d.y / 2;
  const cy = Math.max(-d.y / 2, Math.min(d.y / 2, p.y)); return Math.hypot(p.x, p.y - cy, p.z) <= d.x;
}

export function computeFluidLoads(scene: SceneDescription, fluid: CouplingFluid, bodies: RigidBodyState[], samplesPerAxis = 7) {
  const loads = new Map<string, RigidExternalLoad>(); let displacedTotal = 0;
  for (const body of bodies) {
    const radius = boundingRadius(body), volume = primitiveVolume(body.description.shape, body.description.dimensions_m);
    let inside = 0, wet = 0, centroid = { x: 0, y: 0, z: 0 };
    for (let iz = 0; iz < samplesPerAxis; iz += 1) for (let iy = 0; iy < samplesPerAxis; iy += 1) for (let ix = 0; ix < samplesPerAxis; ix += 1) {
      const p = { x: body.position_m.x + ((ix + 0.5) / samplesPerAxis * 2 - 1) * radius, y: body.position_m.y + ((iy + 0.5) / samplesPerAxis * 2 - 1) * radius, z: body.position_m.z + ((iz + 0.5) / samplesPerAxis * 2 - 1) * radius };
      if (!insidePrimitive(body, p)) continue; inside += 1;
      const occupancy = fluid.sampleOccupancy(p); if (occupancy <= 0) continue; wet += occupancy; centroid = add(centroid, scale(p, occupancy));
    }
    const fraction = inside > 0 ? Math.max(0, Math.min(1, wet / inside)) : 0, displaced = volume * fraction;
    const centreOfBuoyancy = wet > 0 ? scale(centroid, 1 / wet) : body.position_m;
    const buoyant = scale(scene.fluid.gravity_m_s2, -scene.fluid.density_kg_m3 * displaced);
    const fluidVelocity = fluid.sampleVelocity(centreOfBuoyancy), relative = sub(body.linearVelocity_m_s, fluidVelocity), speed = length(relative);
    const area = Math.PI * radius * radius * fraction, drag = speed > 0 ? scale(relative, -0.5 * scene.fluid.density_kg_m3 * 0.9 * area * speed) : { x: 0, y: 0, z: 0 };
    const force = add(buoyant, drag), torque = cross(sub(centreOfBuoyancy, body.position_m), buoyant);
    loads.set(body.description.id, { force_N: force, torque_N_m: torque, buoyantForce_N: buoyant, hydrodynamicForce_N: force, displacedFluidVolume_m3: displaced });
    displacedTotal += displaced;
  }
  return { loads, displacedTotal };
}

export function applyFluidReactions(fluid: CouplingFluid, bodies: RigidBodyState[], loads: ReadonlyMap<string, RigidExternalLoad>, dt: number): CouplingDiagnostics {
  let bodyImpulse = { x: 0, y: 0, z: 0 }, fluidImpulse = { x: 0, y: 0, z: 0 }, displaced = 0, coupled = 0;
  for (const body of bodies) {
    const load = loads.get(body.description.id); if (!load) continue;
    const impulse = scale(load.force_N, dt), reaction = scale(impulse, -1);
    bodyImpulse = add(bodyImpulse, impulse); displaced += load.displacedFluidVolume_m3 ?? 0;
    if (fluid.applyImpulseAt(body.position_m, reaction)) { fluidImpulse = add(fluidImpulse, reaction); coupled += 1; }
  }
  return { displacedVolume_m3: displaced, bodyImpulse_N_s: bodyImpulse, fluidReactionImpulse_N_s: fluidImpulse, momentumClosureError_N_s: length(add(bodyImpulse, fluidImpulse)), coupledBodyCount: coupled };
}
