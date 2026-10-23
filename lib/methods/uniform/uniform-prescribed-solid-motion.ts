import type { RigidBodyState } from "../../core/rigid-body";
import { samplePrescribedRigidBody } from "../../core/prescribed-rigid-motion";

type Pose = Pick<RigidBodyState, "position_m" | "orientation" | "held"> & { fixed: boolean; sample: number };
type Derived = { readonly body: RigidBodyState; readonly linear: RigidBodyState["linearVelocity_m_s"]; readonly angular: RigidBodyState["angularVelocity_rad_s"] };

/** Pose commands are sampled at simulation advances, not pointer-event rate.
 * Only prescribed bodies are differentiated: free bodies are GPU-owned.
 * Authored motors are evaluated at absolute simulation time. Other pose
 * samples build nothing once their roster has been seen: the poses of the
 * sample before last, the differentiated bodies and the returned roster are
 * rewritten in place, so a result is valid until the next sample. */
export class UniformPrescribedSolidMotion {
  private previous = new Map<string, Pose>();
  private spare = new Map<string, Pose>();
  private samples = 0;
  private readonly derived: Derived[] = [];
  private readonly result: RigidBodyState[] = [];
  /** Drops a pose its roster no longer names (the sample that left it is two back). */
  private readonly expire = (pose: Pose, id: string, poses: Map<string, Pose>): void => { if (pose.sample !== this.samples) poses.delete(id); };

  sample(bodies: readonly RigidBodyState[], dt: number, time_s = 0): RigidBodyState[] {
    const next = this.spare, result = this.result, sample = ++this.samples;
    result.length = bodies.length;
    for (let i = 0; i < bodies.length; i++) {
      const body = bodies[i]!;
      const fixed = body.description.motion === "static";
      const old = this.previous.get(body.description.id);
      let pose = next.get(body.description.id);
      if (!pose) next.set(body.description.id, pose = { position_m: { x: 0, y: 0, z: 0 }, orientation: { w: 1, x: 0, y: 0, z: 0 }, held: undefined, fixed, sample });
      const p = pose.position_m, o = pose.orientation;
      p.x = body.position_m.x; p.y = body.position_m.y; p.z = body.position_m.z;
      o.w = body.orientation.w; o.x = body.orientation.x; o.y = body.orientation.y; o.z = body.orientation.z;
      pose.held = body.held; pose.fixed = fixed; pose.sample = sample;
      result[i] = body;
      if ((body.description.prescribedMotion || body.description.poseTrack) && fixed) {
        result[i] = samplePrescribedRigidBody(body, time_s, dt);
        continue;
      }
      // Starting a grab may follow a long GPU simulation whose CPU roster
      // has stale poses. Use the explicit initial command at that transition.
      if (!(dt > 0) || !old || !(fixed || (body.held && old.held))) continue;
      if (fixed && !old.fixed && !old.held) continue;
      // body.orientation times the old orientation's conjugate (quaternionMultiply's terms).
      const a = body.orientation, bw = old.orientation.w, bx = -old.orientation.x, by = -old.orientation.y, bz = -old.orientation.z;
      const qw = a.w * bw - a.x * bx - a.y * by - a.z * bz;
      const qx = a.w * bx + a.x * bw + a.y * bz - a.z * by;
      const qy = a.w * by - a.x * bz + a.y * bw + a.z * bx;
      const qz = a.w * bz + a.x * by - a.y * bx + a.z * bw;
      const sign = qw < 0 ? -1 : 1;
      const length = Math.hypot(qx, qy, qz);
      const scale = length > 1e-12 ? sign * 2 * Math.atan2(length, Math.abs(qw)) / (length * dt) : 0;
      const slot = this.derived[i] ??= { body: {} as RigidBodyState, linear: { x: 0, y: 0, z: 0 }, angular: { x: 0, y: 0, z: 0 } };
      const derived = slot.body;
      if (body.held === undefined && derived.held !== undefined) delete derived.held;
      Object.assign(derived, body);
      slot.linear.x = (body.position_m.x - old.position_m.x) / dt;
      slot.linear.y = (body.position_m.y - old.position_m.y) / dt;
      slot.linear.z = (body.position_m.z - old.position_m.z) / dt;
      slot.angular.x = qx * scale; slot.angular.y = qy * scale; slot.angular.z = qz * scale;
      derived.linearVelocity_m_s = slot.linear; derived.angularVelocity_rad_s = slot.angular;
      result[i] = derived;
    }
    if (next.size !== bodies.length) next.forEach(this.expire);
    this.spare = this.previous; this.previous = next;
    return result;
  }
}
