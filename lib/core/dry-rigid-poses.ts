/**
 * Host posing of prescribed rigid bodies in a dry scene. A scene with no fluid
 * solver has no rigid solver either, so nothing poses its bodies: the renderer
 * would draw a pose-tracked or orbiting body at its t = 0 pose while the cloth
 * collides with it on its track. The presentation instead poses every
 * prescribed body itself, at the time the presented state reached (the cloth's
 * clock, which poses the same track for its contact), and publishes those
 * drawn poses as a resident solver's readback is published, so picking, cuts
 * and gizmos see the frame on screen.
 */
import type { SceneDescription } from "./model";
import { samplePrescribedRigidBody } from "./prescribed-rigid-motion";
import type { RigidBodyState } from "./rigid-body";
import { GPU_RIGID_BODY_CAPACITY } from "./rigid-capacity";
import { planSceneRuntime } from "./scene-runtime";
import type { DrawnRigidBodyPose } from "./webgpu-rigid-body";

/** Whether a body's pose is prescribed (a pose track or an orbit on a static body). */
export function rigidBodyIsPrescribed(body: RigidBodyState): boolean {
  return body.description.motion === "static" && !!(body.description.poseTrack || body.description.prescribedMotion);
}

/**
 * The roster at `time_s` with every prescribed body posed, and the poses to publish for the drawn frame (the
 * first `capacity` bodies, as the GPU roster holds). Undefined when the scene has a fluid solver (its rigid solver
 * owns and reads back the poses) or no body is prescribed.
 */
export function dryPrescribedRigidPoses(scene: SceneDescription, bodies: readonly RigidBodyState[], time_s: number,
  capacity = GPU_RIGID_BODY_CAPACITY): { bodies: RigidBodyState[]; poses: DrawnRigidBodyPose[] } | undefined {
  if (planSceneRuntime(scene).fluidSolver || !bodies.some(rigidBodyIsPrescribed)) return undefined;
  const posed = bodies.map(body => samplePrescribedRigidBody(body, time_s, scene.numerics.fixedDt_s));
  return { bodies: posed, poses: posed.slice(0, capacity).map(body => ({ id: body.description.id, position_m: body.position_m, orientation: body.orientation })) };
}
