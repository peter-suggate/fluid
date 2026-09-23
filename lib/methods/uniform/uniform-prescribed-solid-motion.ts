import { quaternionMultiply, type RigidBodyState } from "../../core/rigid-body";

type Pose = Pick<RigidBodyState, "position_m" | "orientation" | "held"> & { fixed: boolean };

/** Pose commands are sampled at simulation advances, not pointer-event rate.
 * Only prescribed bodies are differentiated: free bodies are GPU-owned. */
export class UniformPrescribedSolidMotion {
  private previous = new Map<string, Pose>();

  sample(bodies: readonly RigidBodyState[], dt: number): RigidBodyState[] {
    const next = new Map<string, Pose>();
    const result = bodies.map(body => {
      const fixed = body.description.motion === "static";
      const old = this.previous.get(body.description.id);
      next.set(body.description.id, {
        position_m: { ...body.position_m }, orientation: { ...body.orientation },
        held: body.held, fixed,
      });
      // Starting a grab may follow a long GPU simulation whose CPU roster
      // has stale poses. Use the explicit initial command at that transition.
      if (!(dt > 0) || !old || !(fixed || (body.held && old.held))) return body;
      if (fixed && !old.fixed && !old.held) return body;
      const q = quaternionMultiply(body.orientation, {
        w: old.orientation.w, x: -old.orientation.x,
        y: -old.orientation.y, z: -old.orientation.z,
      });
      const sign = q.w < 0 ? -1 : 1;
      const length = Math.hypot(q.x, q.y, q.z);
      const scale = length > 1e-12 ? sign * 2 * Math.atan2(length, Math.abs(q.w)) / (length * dt) : 0;
      return {
        ...body,
        linearVelocity_m_s: {
          x: (body.position_m.x - old.position_m.x) / dt,
          y: (body.position_m.y - old.position_m.y) / dt,
          z: (body.position_m.z - old.position_m.z) / dt,
        },
        angularVelocity_rad_s: { x: q.x * scale, y: q.y * scale, z: q.z * scale },
      };
    });
    this.previous = next;
    return result;
  }
}
