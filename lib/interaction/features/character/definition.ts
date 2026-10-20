import type { EntityKindPlugin, BehaviorPlugin } from '../../contracts';
import type { PhysicsBehaviorFactory } from '../../runtime/physics';
import { behavior, controls, instance, metadata, numbers, scalar } from '../shared';
const schema = numbers({ radius: .3, height: 1.7, speed: 3, runMultiplier: 1.6, jumpHeight: .6, density: 800, acceleration: 18, maxSlope: 50, stepHeight: .2 }, { maxSlope: [0, 80], stepHeight: [0, .6], jumpHeight: [0, 5] });
const validate = schema.validate;
const TURN_RATE_RAD_S = 2.6;
export const character: EntityKindPlugin = {
  id: 'fluid.character', schema: { ...schema, validate: c => [...validate(c), ...(Number(c.height) < Number(c.radius) * 2 ? ['height must be at least twice radius'] : [])] },
  editor: { ...metadata('Character', 'Characters', 0), controls: controls(schema, {speed:'live',runMultiplier:'live',jumpHeight:'live',acceleration:'live',maxSlope:'live',stepHeight:'live',density:'rebuild',radius:'rebuild',height:'rebuild'}) },
  placement: { alignment: 'upright', overlap: 'reject' }, requires: ['transform', 'controllable'],
  preview: c => ({ shape: 'capsule', radius_m: c.radius as number, height_m: c.height as number }),
  compose: () => [instance('fluid.character-motor')],
};
export const characterMotor: BehaviorPlugin = {
  ...behavior('fluid.character-motor', 'Character movement', ['transform', 'controllable']), requiresHost: ['box3d'], phases: [{ phase: 'intent' }],
  actions: [
    { id: 'move', label: 'Move', kind: 'axis2', bindings: [
      { device: 'keyboard', code: 'KeyW', value: [0, -1] }, { device: 'keyboard', code: 'KeyS', value: [0, 1] },
      { device: 'keyboard', code: 'KeyA', value: [-1, 0] }, { device: 'keyboard', code: 'KeyD', value: [1, 0] },
      { device: 'keyboard', code: 'ArrowUp', value: [0, -1] }, { device: 'keyboard', code: 'ArrowDown', value: [0, 1] },
      { device: 'keyboard', code: 'ArrowLeft', value: [-1, 0] }, { device: 'keyboard', code: 'ArrowRight', value: [1, 0] },
    ] },
    { id: 'jump', label: 'Jump', kind: 'button', bindings: [{ device: 'keyboard', code: 'Space', value: true }] },
    { id: 'run', label: 'Run', kind: 'button', bindings: [{ device: 'keyboard', code: 'ShiftLeft', value: true }] },
    { id: 'somersault', label: 'Somersault', kind: 'button', bindings: [{ device: 'keyboard', code: 'KeyF', value: true }] },
  ],
};
export const characterRuntime: PhysicsBehaviorFactory = {
  type: characterMotor.id,
  body: e => ({ shape: character.preview(e.config), motion: 'dynamic', density: scalar(e, 'density'), friction: .15,
    ...(e.kind === 'fluid.robot' ? { appearance: 'bub' as const } : {}),
    character: { speed: scalar(e, 'speed'), jumpHeight: scalar(e, 'jumpHeight'), acceleration: scalar(e, 'acceleration'), maxSlope: scalar(e, 'maxSlope'), stepHeight: scalar(e, 'stepHeight') } }),
  create: () => ({ tick: (_phase, ctx) => {
    const actions = ctx.input.actions.filter(a => a.entityId === ctx.entity.id && a.behaviorId === ctx.behavior.id);
    const move = actions.find(a => a.actionId === 'move')?.value;
    const [x, z] = Array.isArray(move) ? move : [0, 0];
    const run = actions.some(a => a.actionId === 'run' && a.value === true);
    const speed = scalar(ctx.entity, 'speed') * (run ? scalar(ctx.entity, 'runMultiplier') : 1);
    // Steered in the character's own frame: W/S walk along its facing, A/D turn it.
    const ahead = Math.max(-1, Math.min(1, -z)), turn = -Math.max(-1, Math.min(1, x)) * TURN_RATE_RAD_S;
    ctx.enqueue({ kind: 'character-move', entityId: ctx.entity.id, velocity_m_s: { x: 0, y: 0, z: ahead * speed }, turn_rad_s: turn, jump: actions.some(a => a.actionId === 'jump' && a.edge === 'pressed'), climb: actions.some(a => a.actionId === 'jump' && a.value === true), somersault: actions.some(a => a.actionId === 'somersault' && a.edge === 'pressed') });
  }, dispose() {} }),
};
