/** A scene's bodies as probes drive them without a rigid solver: every
 * motor-attached fixture follows its authored drive (rigidMotorPose) with the
 * drive's velocity, and every other body is held where authored (a moored
 * boat's wheels). Without these the fluid sees the bodies stand still. */
import { initializeRigidBodies, quaternionRotate, type RigidBodyState } from "../lib/core/rigid-body";
import { rigidMotorAngle, rigidMotorPose, rigidMotorStrength } from "../lib/core/rigid-motor";
import type { SceneDescription } from "../lib/core/model";

export function mooredMotorBodies(scene: SceneDescription, dt: number): (time_s: number) => RigidBodyState[] {
  const bodies = initializeRigidBodies(scene.rigidBodies);
  const parents = new Map(bodies.map(body => [body.description.id, body] as const));
  return time => {
    for (const body of bodies) {
      const m = body.description.motorAttachment;
      if (!m) { body.held = true; continue; }
      const parent = parents.get(m.parentId);
      if (!parent) throw new Error(`Motor fixture ${body.description.id} has no parent ${m.parentId}`);
      const drive = { ...m, angularSpeed_rad_s: rigidMotorStrength(parent.description, m.channel) * m.angularSpeed_rad_s };
      const now = rigidMotorPose(parent, drive, time), before = rigidMotorPose(parent, drive, time - dt);
      body.position_m = now.position_m; body.orientation = now.orientation;
      body.linearVelocity_m_s = { x: (now.position_m.x - before.position_m.x) / dt, y: (now.position_m.y - before.position_m.y) / dt, z: (now.position_m.z - before.position_m.z) / dt };
      const omega = (rigidMotorAngle(drive, time) - rigidMotorAngle(drive, time - dt)) / dt;
      const axis = quaternionRotate(parent.orientation, m.axis);
      body.angularVelocity_rad_s = { x: omega * axis.x, y: omega * axis.y, z: omega * axis.z };
    }
    return bodies;
  };
}
