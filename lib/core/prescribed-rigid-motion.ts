import type { PrescribedOrbitMotion, Vec3 } from "./model";
import { quaternionMultiply, type RigidBodyState } from "./rigid-body";

/** Integral of smoothstep, clamped at both ends of a ramp. */
function rampIntegral(time: number, duration: number): number {
  const u = Math.max(0, Math.min(1, time / duration));
  return duration * (u ** 3 - 0.5 * u ** 4);
}

function prescribedOrbitAngle(motion: PrescribedOrbitMotion, time_s: number): number {
  const t = Math.max(0, time_s - motion.startTime_s);
  const down = Math.max(0, Math.min(motion.rampDown_s, t - motion.rampUp_s - motion.steady_s));
  const travel = rampIntegral(t, motion.rampUp_s)
    + Math.max(0, Math.min(motion.steady_s, t - motion.rampUp_s))
    + down - rampIntegral(down, motion.rampDown_s);
  return motion.phase_rad + motion.angularSpeed_rad_s * travel;
}

export function prescribedOrbitPosition(motion: PrescribedOrbitMotion, time_s: number): Vec3 {
  const angle = prescribedOrbitAngle(motion, time_s);
  return {
    x: motion.center_m.x + motion.radius_m * Math.cos(angle),
    y: motion.center_m.y,
    z: motion.center_m.z + motion.radius_m * Math.sin(angle),
  };
}

/** Absolute simulation time makes pause, restart and catch-up deterministic.
 * Use the step's displacement velocity so the moving solid and its fluid
 * boundary agree, including the first and last partial ramp steps. */
export function samplePrescribedRigidBody(body: RigidBodyState, time_s: number, dt_s: number): RigidBodyState {
  const motion = body.description.prescribedMotion;
  if (!motion || body.description.motion !== "static") return body;
  const position_m = prescribedOrbitPosition(motion, time_s);
  const before = prescribedOrbitPosition(motion, time_s - Math.max(0, dt_s));
  const scale = dt_s > 0 ? 1 / dt_s : 0;
  // Positive XZ travel is a negative rotation about Y. Keep angles unwrapped
  // so crossing a full turn cannot introduce a boundary-velocity spike.
  const angle = prescribedOrbitAngle(motion, time_s);
  const yaw = -(angle - motion.phase_rad);
  const angularSpeed = motion.rotateWithOrbit
    ? -(angle - prescribedOrbitAngle(motion, time_s - Math.max(0, dt_s))) * scale : 0;
  const orientation = motion.rotateWithOrbit
    ? quaternionMultiply({ w: Math.cos(yaw / 2), x: 0, y: Math.sin(yaw / 2), z: 0 }, body.description.orientation)
    : { ...body.description.orientation };
  return { ...body, position_m, orientation,
    linearVelocity_m_s: { x: (position_m.x - before.x) * scale, y: 0, z: (position_m.z - before.z) * scale },
    angularVelocity_rad_s: { x: 0, y: angularSpeed, z: 0 } };
}
