import type { PrescribedOrbitMotion, RigidPoseTrack, Vec3 } from "./model";
import { quaternionMultiply, quaternionNormalize, quaternionRotate, type RigidBodyState } from "./rigid-body";

/** Linear translation and shortest-arc spherical rotation, held outside the track. */
export function sampleRigidPoseTrack(track: RigidPoseTrack, time_s: number) {
  const frames = track.keyframes;
  let right = frames.findIndex(frame => frame.time_s >= time_s);
  if (right < 0) right = frames.length - 1;
  const b = frames[right]!, a = frames[Math.max(0, right - 1)]!;
  const t = a === b ? 0 : Math.max(0, Math.min(1, (time_s - a.time_s) / (b.time_s - a.time_s)));
  const qa = a.orientation, qb = b.orientation;
  const dot = qa.w * qb.w + qa.x * qb.x + qa.y * qb.y + qa.z * qb.z;
  const sign = dot < 0 ? -1 : 1, angle = Math.acos(Math.min(1, Math.abs(dot)));
  const wa = angle < 1e-6 ? 1 - t : Math.sin((1 - t) * angle) / Math.sin(angle);
  const wb = sign * (angle < 1e-6 ? t : Math.sin(t * angle) / Math.sin(angle));
  const orientation = quaternionNormalize({ w: wa * qa.w + wb * qb.w, x: wa * qa.x + wb * qb.x,
    y: wa * qa.y + wb * qb.y, z: wa * qa.z + wb * qb.z });
  const offset = quaternionRotate(orientation, track.localOffset_m ?? { x: 0, y: 0, z: 0 });
  const position_m = { x: a.position_m.x + t * (b.position_m.x - a.position_m.x) + offset.x,
    y: a.position_m.y + t * (b.position_m.y - a.position_m.y) + offset.y,
    z: a.position_m.z + t * (b.position_m.z - a.position_m.z) + offset.z };
  return { position_m, orientation };
}

function sampleTrackedBody(body: RigidBodyState, time_s: number, dt_s: number): RigidBodyState {
  const track = body.description.poseTrack!;
  const now = sampleRigidPoseTrack(track, time_s), before = sampleRigidPoseTrack(track, time_s - Math.max(0, dt_s));
  const q = before.orientation;
  const delta = quaternionMultiply(now.orientation, { w: q.w, x: -q.x, y: -q.y, z: -q.z });
  const length = Math.hypot(delta.x, delta.y, delta.z), scale = dt_s > 0 ? 1 / dt_s : 0;
  const angular = length > 1e-12 ? (delta.w < 0 ? -1 : 1) * 2 * Math.atan2(length, Math.abs(delta.w)) * scale / length : 0;
  return { ...body, ...now,
    linearVelocity_m_s: { x: (now.position_m.x - before.position_m.x) * scale,
      y: (now.position_m.y - before.position_m.y) * scale, z: (now.position_m.z - before.position_m.z) * scale },
    angularVelocity_rad_s: { x: delta.x * angular, y: delta.y * angular, z: delta.z * angular } };
}

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
  if (body.description.poseTrack && body.description.motion === "static") return sampleTrackedBody(body, time_s, dt_s);
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
