import type { EntityKindPlugin } from '../../contracts';
import type { PhysicsBehaviorFactory } from '../../runtime/physics';
import { activePort, behavior, box, controls, instance, metadata, numbers, scalar } from '../shared';
const schema = numbers({ width: 1.5, height: .2, depth: 1.5, travelX: 0, travelY: 1, travelZ: 0, speed: .6 }, { travelX: [-100, 100], travelY: [-100, 100], travelZ: [-100, 100] });
export const movingPlatform: EntityKindPlugin = { id: 'fluid.platform', schema, editor: { ...metadata('Moving platform', 'Mechanisms', 3), controls: controls(schema, {speed:'live',width:'rebuild',height:'rebuild',depth:'rebuild'}) }, placement: { alignment: 'upright', overlap: 'reject' }, requires: ['transform'], preview: box, compose: () => [instance('fluid.platform-motion')] };
export const platformMotion = { ...behavior('fluid.platform-motion', 'Platform motion', ['transform']), requiresHost: ['box3d'], phases: [{ phase: 'before-physics' as const }], inputs: [activePort] };
export const platformRuntime: PhysicsBehaviorFactory = { type: platformMotion.id, body: e => ({ shape: box(e.config), motion: 'kinematic', density: 1000, friction: .8 }), create: () => {
  let active = false, progress = 0;
  return { tick: (_phase, ctx) => {
    for (const event of ctx.events) if (event.target.port === 'active') active = event.value === true;
    const delta = { x: scalar(ctx.entity, 'travelX'), y: scalar(ctx.entity, 'travelY'), z: scalar(ctx.entity, 'travelZ') };
    const distance = Math.hypot(delta.x, delta.y, delta.z);
    progress = Math.max(0, Math.min(1, progress + (active ? 1 : -1) * scalar(ctx.entity, 'speed') * ctx.dt_s / Math.max(.001, distance)));
    const body = ctx.snapshot.bodies.find(b => b.entityId === ctx.entity.id);
    if (body) ctx.enqueue({ kind: 'kinematic-target', bodyId: body.bodyId, bodyGeneration: body.bodyGeneration, transform: { position_m: { x: ctx.entity.transform.position_m.x + delta.x * progress, y: ctx.entity.transform.position_m.y + delta.y * progress, z: ctx.entity.transform.position_m.z + delta.z * progress }, orientation: ctx.entity.transform.orientation } });
  }, dispose() {} };
} };
