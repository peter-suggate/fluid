import type { EntityKindPlugin } from '../../contracts';
import type { PhysicsBehaviorFactory } from '../../runtime/physics';
import { behavior, box, controls, instance, metadata, numbers, scalar } from '../shared';
const schema = numbers({ width: .5, height: .5, depth: .5, density: 600, friction: .6, restitution: .05 }, { friction: [0, 1], restitution: [0, 1] });
export const rigidProp: EntityKindPlugin = { id: 'fluid.crate', schema, editor: { ...metadata('Crate', 'Props', 1), controls: controls(schema, {width:'rebuild',height:'rebuild',depth:'rebuild',density:'rebuild',friction:'rebuild',restitution:'rebuild'}) }, placement: { alignment: 'surface', overlap: 'reject' }, requires: ['transform'], preview: box, compose: () => [instance('fluid.rigid-body')] };
export const rigidBody = { ...behavior('fluid.rigid-body', 'Rigid body', ['transform', 'grabbable']), requiresHost: ['box3d'], phases: [{ phase: 'intent' as const }], actions: [{ id: 'grab', label: 'Hold to grab', kind: 'button' as const, bindings: [{ device: 'keyboard' as const, code: 'KeyG', value: true }] }] };
export const rigidRuntime: PhysicsBehaviorFactory = { type: rigidBody.id, create: () => ({ tick: (_phase, ctx) => {
  const active = ctx.input.actions.some(a => a.entityId === ctx.entity.id && a.behaviorId === ctx.behavior.id && a.actionId === 'grab' && a.value === true);
  const p = ctx.viewOrigin_m ?? { x: 0, y: 0, z: 0 }, d = ctx.viewDirection ?? { x: 0, y: 0, z: -1 };
  ctx.enqueue({ kind: 'grab', entityId: ctx.entity.id, active, target_m: { x: p.x + d.x * 2, y: p.y + d.y * 2, z: p.z + d.z * 2 } });
}, dispose() {} }), body: e => ({ shape: box(e.config), motion: 'dynamic', density: scalar(e, 'density'), friction: scalar(e, 'friction'), restitution: scalar(e, 'restitution') }) };
