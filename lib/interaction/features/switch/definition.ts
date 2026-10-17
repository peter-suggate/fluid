import type { EntityKindPlugin } from '../../contracts';
import type { PhysicsBehaviorFactory } from '../../runtime/physics';
import { activePort, behavior, box, controls, instance, metadata, numbers } from '../shared';
const schema = numbers({ width: .25, height: .25, depth: .25 });
export const sceneSwitch: EntityKindPlugin = { id: 'fluid.switch', schema, editor: { ...metadata('Switch', 'Mechanisms', 2), controls: controls(schema) }, placement: { alignment: 'surface', overlap: 'reject' }, requires: ['transform'], preview: box, compose: () => [{ ...instance('fluid.static-body'), id: 'body' }, instance('fluid.switch-toggle')] };
export const staticBody = { ...behavior('fluid.static-body', 'Static body', ['transform']), requiresHost: ['box3d'] };
export const staticBodyRuntime: PhysicsBehaviorFactory = { type: staticBody.id, body: e => ({ shape: box(e.config), motion: 'static', density: 1000, friction: .6 }), create: () => ({ tick() {}, dispose() {} }) };
export const switchToggle = { ...behavior('fluid.switch-toggle', 'Switch'), phases: [{ phase: 'intent' as const }], outputs: [activePort], actions: [{ id: 'use', label: 'Toggle', kind: 'button' as const, bindings: [{ device: 'keyboard' as const, code: 'KeyE', value: true }] }] };
export const switchRuntime: PhysicsBehaviorFactory = { type: switchToggle.id, create: () => {
  let active = false;
  return { tick: (_phase, ctx) => { if (ctx.input.actions.some(a => a.entityId === ctx.entity.id && a.behaviorId === ctx.behavior.id && a.actionId === 'use' && a.edge === 'pressed')) { active = !active; ctx.emit('active', active); } }, dispose() {} };
} };
