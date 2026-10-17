import type { EntityKindPlugin } from '../../contracts';
import type { PhysicsBehaviorFactory } from '../../runtime/physics';
import { activePort, behavior, box, controls, instance, metadata, numbers } from '../shared';
const schema = numbers({ width: 1, height: 1, depth: 1 });
export const triggerVolume: EntityKindPlugin = { id: 'fluid.trigger', schema, editor: { ...metadata('Trigger volume', 'Triggers', 4), controls: controls(schema) }, placement: { alignment: 'upright', overlap: 'allow' }, requires: ['transform'], preview: box, compose: () => [instance('fluid.trigger-sensor')] };
export const triggerSensor = { ...behavior('fluid.trigger-sensor', 'Trigger sensor', ['transform']), requiresHost: ['box3d'], phases: [{ phase: 'after-physics' as const }], outputs: [activePort] };
export const triggerRuntime: PhysicsBehaviorFactory = { type: triggerSensor.id, body: e => ({ shape: box(e.config), motion: 'static', sensor: true, density: 1000, friction: 0 }), create: () => {
  const visitors = new Set<string>();
  return { tick: (_phase, ctx) => {
    const before = visitors.size > 0;
    for (const contact of ctx.sensors ?? []) if (contact.sensorEntityId === ctx.entity.id) { if (contact.entered) visitors.add(contact.visitorEntityId); else visitors.delete(contact.visitorEntityId); }
    if (before !== (visitors.size > 0)) ctx.emit('active', visitors.size > 0);
  }, dispose() { visitors.clear(); } };
} };
