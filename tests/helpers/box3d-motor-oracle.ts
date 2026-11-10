import asset from '../../lib/core/imported-assets/paddleboats.generated.json';
import type { RigidBodyDescription, SceneDescription, Vec3 } from '../../lib/core/model';
import { add, cross, dot, normalize, scale } from '../../lib/core/math';
import { boundingRadius, massProperties, quaternionInverseRotate, quaternionRotate } from '../../lib/core/rigid-body';
import { rigidAppearanceSkin } from '../../lib/core/rigid-appearance';
import { sceneShapeRenderHalfExtent_m } from '../../lib/core/scene-shape';
import { Box3DWorld } from '../../lib/interaction/adapters/box3d/world';
import { GPU_RIGID_BODY_CAPACITY } from '../../lib/core/rigid-capacity';

const corners = (lo: readonly number[], hi: readonly number[]): Vec3[] =>
  Array.from({ length: 8 }, (_, i) => ({ x: (i & 1 ? hi : lo)[0], y: (i & 2 ? hi : lo)[1], z: (i & 4 ? hi : lo)[2] }));

/** One contact envelope per axle, enclosing every phase of its rendered parts.
 * The axle is rigidly attached to the parent: a wall impulse acts on the boat,
 * not on the prescribed, massless fluid fixtures. */
export function motorContactEnvelope(body: RigidBodyDescription): Vec3[] {
  const m = body.motorAttachment!;
  const half = rigidAppearanceSkin(body)?.half ?? sceneShapeRenderHalfExtent_m(body.shape, body.dimensions_m);
  const localPoints = body.appearance === 'paddleboat-wheel'
    ? asset.wheel.nodes.filter(n => n.part >= 0).flatMap(n => corners(n.lo, n.hi))
    : corners(half.map(v => -v), half);
  const points = localPoints.map(p => add(m.offset_m, quaternionRotate(m.orientation, p)));
  const axis = normalize(m.axis);
  const along = points.map(p => dot(p, axis));
  const radius = Math.max(...points.map((p, i) => Math.hypot(...Object.values(add(p, scale(axis, -along[i]))))));
  const u = normalize(cross(axis, Math.abs(axis.y) < .9 ? { x: 0, y: 1, z: 0 } : { x: 1, y: 0, z: 0 }));
  const v = cross(axis, u), n = 16, r = radius / Math.cos(Math.PI / n);
  return [Math.min(...along), Math.max(...along)].flatMap(z => Array.from({ length: n }, (_, i) =>
    add(m.pivot_m, add(scale(axis, z), add(scale(u, r * Math.cos(i * 2 * Math.PI / n)), scale(v, r * Math.sin(i * 2 * Math.PI / n)))))));
}

/** TEST ORACLE ONLY. The CPU Box3D motor-assembly contact solve that the
 * paddleboats scene used to run after a per-step pose readback; production
 * now resolves the same contacts inside the GPU rigid kernel
 * (lib/core/rigid-assembly-contact.ts). Kept here so the kernel's mirror can
 * be compared against an independent rigid-body engine. Native CCD/contact/
 * drift with substeps bounded by travel; the core collider alone carries mass
 * and inertia, the hull and swept wheel envelopes carry none. */
export class Box3DMotorContacts {
  private types = new Map<string, string>();
  private constructor(private readonly world: Box3DWorld) {}
  static async create(scene: SceneDescription, descriptions: readonly RigidBodyDescription[]) {
    const roots = descriptions.filter(b => !b.motorAttachment);
    const nativeScene = { ...scene, rigidBodies: roots, fluid: { ...scene.fluid, gravity_m_s2: { x: 0, y: 0, z: 0 } } };
    const world = await Box3DWorld.create(nativeScene, [], 1, GPU_RIGID_BODY_CAPACITY);
    try {
      for (const root of roots) {
        const native = world.bodies.get(root.id)!.native;
        const envelopes: Vec3[][] = [];
        if (root.appearance === 'paddleboat-red' || root.appearance === 'paddleboat-teal') {
          // Exported BVH leaf bounds enclose all original analytic parts,
          // including the bow and stern supports outside the buoyancy box.
          envelopes.push(asset.hull.nodes.filter(n => n.part >= 0).flatMap(n => corners(n.lo, n.hi)));
        }
        const axles = new Set<string>();
        for (const child of descriptions.filter(b => b.motorAttachment?.parentId === root.id)) {
          const envelope = motorContactEnvelope(child), key = JSON.stringify(envelope);
          if (!axles.has(key)) { envelopes.push(envelope); axles.add(key); }
        }
        for (const points of envelopes) {
          const shape = native.createHull({ points, density: 0, friction: root.friction, restitution: root.restitution });
          const valid = shape.isValid(); shape.delete();
          if (!valid) throw new Error(`Invalid motor contact envelope for ${root.id}`);
        }
      }
      return new Box3DMotorContacts(world);
    } catch (error) { world.destroy(); throw error; }
  }
  updateSolids(scene: SceneDescription): void { this.world.updateSolids(scene); }
  /** Mutates only the pose/velocity/momentum lanes of the GPU state image. */
  step(values: Float32Array, descriptions: readonly RigidBodyDescription[], dt: number): void {
    const vector = (at: number): Vec3 => ({ x: values[at], y: values[at+1], z: values[at+2] });
    for (const [i, description] of descriptions.entries()) {
      if (description.motorAttachment) continue;
      const body = this.world.bodies.get(description.id)!.native, at = i * 32;
      const type = values[at+30] === 1 ? 'dynamic' : 'kinematic';
      if (this.types.get(description.id) !== type) { body.setType(type); this.types.set(description.id, type); }
      const p = body.getPosition(), q = body.getRotation();
      // setTransform invalidates native contact caches, even for the same pose.
      // The GPU kick preserves these lanes; only an explicit edit/drag teleports.
      if ([p.x,p.y,p.z,q.w,q.x,q.y,q.z].some((v, i) => Math.abs(v - values[at + (i < 3 ? i : i+5)]) > 1e-7))
        body.setTransform(vector(at), { w: values[at+8], x: values[at+9], y: values[at+10], z: values[at+11] });
      body.setLinearVelocity(values[at+30] === 1 ? vector(at+12) : { x: 0, y: 0, z: 0 });
      body.setAngularVelocity(values[at+30] === 1 ? vector(at+16) : { x: 0, y: 0, z: 0 });
    }
    this.world.ensureCoverage();
    // Box3D refreshes broadphase pairs once per step (solver substeps alone
    // do not refresh them). Bound travel so a thin pool wall cannot be crossed
    // by a massless outboard contact shape before its pair exists.
    let speed = 0;
    for (const [i, description] of descriptions.entries()) {
      if (description.motorAttachment || values[i*32+30] !== 1) continue;
      const at = i*32;
      const reach = Math.max(boundingRadius(description), rigidAppearanceSkin(description)?.radius ?? 0,
        ...(descriptions.filter(b => b.motorAttachment?.parentId === description.id)
        .map(b => Math.hypot(...Object.values(b.motorAttachment!.pivot_m)) + (rigidAppearanceSkin(b)?.radius ?? 1))));
      speed = Math.max(speed, Math.hypot(values[at+12],values[at+13],values[at+14]) + reach*Math.hypot(values[at+16],values[at+17],values[at+18]));
    }
    const steps = Math.max(1, Math.ceil(speed*dt/.01));
    for (let i=0; i<steps; i++) this.world.step(dt/steps);
    for (const [i, description] of descriptions.entries()) {
      const at = i*32;
      if (description.motorAttachment || values[at+30] !== 1) continue;
      const body = this.world.bodies.get(description.id)!.native;
      const p = body.getPosition(), q = body.getRotation(), v = body.getLinearVelocity(), w = body.getAngularVelocity();
      const inertia = massProperties(description).inertiaBody_kg_m2, local = quaternionInverseRotate(q, w);
      const momentum = quaternionRotate(q, { x: local.x*inertia.x, y: local.y*inertia.y, z: local.z*inertia.z });
      values.set([p.x,p.y,p.z], at); values.set([q.w,q.x,q.y,q.z], at+8);
      values.set([v.x,v.y,v.z], at+12); values.set([w.x,w.y,w.z], at+16);
      values.set([momentum.x,momentum.y,momentum.z], at+24);
    }
  }
  destroy(): void { this.world.destroy(); }
}
