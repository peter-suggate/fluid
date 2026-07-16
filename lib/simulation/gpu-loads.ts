import { add, length, scale, sub } from "../math";
import type { SceneDescription } from "../model";
import { boundingRadius, primitiveVolume, type RigidBodyState, type RigidExternalLoad } from "../rigid-body";
import { consumeGPURigidLoad, type GPURigidLoad } from "../webgpu-eulerian";
import type { CouplingDiagnostics } from "../fluid-rigid-coupling";

export const GPU_RIGID_DRAG_COEFFICIENT = 0.9;
export const GPU_RIGID_ADDED_MASS_COEFFICIENT = 0.5;

/** Assemble body loads from the latest GPU exchange snapshot. The impulse
 * channel retains its existing interval amortization; buoyancy, form drag,
 * and added mass are evaluated from the latest immersed-volume snapshot. */
export function externalLoadsFromGPU(scene: Pick<SceneDescription, "fluid">, gpuLoads: GPURigidLoad[], dt: number, bodies: readonly RigidBodyState[]) {
  const loads = new Map<string, RigidExternalLoad>(), bodiesById = new Map(bodies.map((body) => [body.description.id, body]));
  let displacedVolume_m3 = 0, bodyImpulse_N_s = { x: 0, y: 0, z: 0 }, fluidReactionImpulse_N_s = { x: 0, y: 0, z: 0 }, coupledBodyCount = 0;
  const fluidDensity = scene.fluid.density_kg_m3;

  for (const gpuLoad of gpuLoads) {
    const { impulse_N_s: stepImpulse, angularImpulse_N_m_s: stepAngularImpulse } = consumeGPURigidLoad(gpuLoad, dt);
    const body = bodiesById.get(gpuLoad.bodyId);
    if (!body) continue;

    const hydrodynamicImpulseForce = scale(stepImpulse, 1 / dt);
    const hydrodynamicTorque = scale(stepAngularImpulse, 1 / dt);
    const bodyVolume = primitiveVolume(body.description.shape, body.description.dimensions_m);
    const displaced = Math.max(0, gpuLoad.displacedVolume_m3);
    const immersedFraction = Math.max(0, Math.min(1, displaced / Math.max(bodyVolume, Number.EPSILON)));
    const buoyantForce = scale(scene.fluid.gravity_m_s2, -fluidDensity * displaced);
    const relativeVelocity = sub(body.linearVelocity_m_s, gpuLoad.meanFluidVelocity_m_s);
    const relativeSpeed = length(relativeVelocity);
    const referenceArea = Math.PI * boundingRadius(body) ** 2 * immersedFraction;
    const dragForce = relativeSpeed > 0 && referenceArea > 0
      ? scale(relativeVelocity, -0.5 * fluidDensity * GPU_RIGID_DRAG_COEFFICIENT * referenceArea * relativeSpeed)
      : { x: 0, y: 0, z: 0 };
    const hydrodynamicForce = add(hydrodynamicImpulseForce, dragForce);
    const fluidForce = add(buoyantForce, hydrodynamicForce);

    // advanceRigidBodies divides by the body's own mass. Re-express the total
    // force so its existing integrator produces (m*g + Ffluid)/(m + ma).
    const addedMass = GPU_RIGID_ADDED_MASS_COEFFICIENT * fluidDensity * displaced;
    const gravityForce = scale(scene.fluid.gravity_m_s2, body.mass_kg);
    const force = addedMass > 0
      ? sub(scale(add(gravityForce, fluidForce), body.mass_kg / (body.mass_kg + addedMass)), gravityForce)
      : fluidForce;

    loads.set(gpuLoad.bodyId, {
      force_N: force,
      torque_N_m: hydrodynamicTorque,
      buoyantForce_N: buoyantForce,
      hydrodynamicForce_N: hydrodynamicForce,
      displacedFluidVolume_m3: displaced
    });
    displacedVolume_m3 += displaced;
    bodyImpulse_N_s = add(bodyImpulse_N_s, scale(force, dt));
    // The VOS blend's sampled impulse has an opposite fluid reaction. Analytic
    // buoyancy and CPU-side drag do not; do not fabricate reactions for either.
    fluidReactionImpulse_N_s = add(fluidReactionImpulse_N_s, scale(stepImpulse, -1));
    if (displaced > 0) coupledBodyCount += 1;
  }

  const diagnostics: CouplingDiagnostics = {
    displacedVolume_m3,
    bodyImpulse_N_s,
    fluidReactionImpulse_N_s,
    momentumClosureError_N_s: length(add(bodyImpulse_N_s, fluidReactionImpulse_N_s)),
    coupledBodyCount
  };
  return { loads, diagnostics };
}
