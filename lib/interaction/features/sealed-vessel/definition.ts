import type { EntityKindPlugin } from '../../contracts';
import type { PhysicsBehaviorFactory, BodyRecipe } from '../../runtime/physics';
import { behavior, box, controls, inertRuntime, instance, metadata, numbers, scalar } from '../shared';

const numeric = numbers({ width: 1.6, height: 1.44, depth: 1.6, wallThickness: .16,
  density: 600, friction: .6, restitution: .05, velocityZ: 0, angularX: 0 },
  { friction: [0, 1], restitution: [0, 1], velocityZ: [-10, 10], angularX: [-10, 10] });
const schema = { ...numeric, validate: (value: Parameters<typeof numeric.validate>[0]) => {
  const errors = [...numeric.validate(value)];
  if (2 * Number(value.wallThickness) >= Math.min(Number(value.width), Number(value.height), Number(value.depth))) errors.push('Walls must leave an open interior');
  return errors;
} };
export const sealedVessel: EntityKindPlugin = { id: 'fluid.sealed-vessel', schema,
  editor: { ...metadata('Sealed vessel', 'Props', 2), controls: controls(schema) },
  placement: { alignment: 'surface', overlap: 'reject' }, requires: ['transform'], preview: box,
  compose: () => [instance('fluid.compound-vessel')] };
export const compoundVessel = { ...behavior('fluid.compound-vessel', 'Dynamic sealed vessel', ['transform']), requiresHost: ['box3d'] };
export const vesselRuntime: PhysicsBehaviorFactory = { type: compoundVessel.id, create: inertRuntime, body: e => {
  const x = scalar(e, 'width'), y = scalar(e, 'height'), z = scalar(e, 'depth'), t = scalar(e, 'wallThickness');
  // Disjoint volumes: Box3D computes shell mass/inertia without counting corners twice.
  const panels = [
    ['floor', [0, -(y-t)/2, 0], [x, t, z]], ['lid', [0, (y-t)/2, 0], [x, t, z]],
    ['left', [-(x-t)/2, 0, 0], [t, y-2*t, z]], ['right', [(x-t)/2, 0, 0], [t, y-2*t, z]],
    ['back', [0, 0, -(z-t)/2], [x-2*t, y-2*t, t]], ['front', [0, 0, (z-t)/2], [x-2*t, y-2*t, t]],
  ] as const;
  return { shape: box(e.config), motion: 'dynamic', density: scalar(e, 'density'), friction: scalar(e, 'friction'),
    restitution: scalar(e, 'restitution'), renderHidden: true,
    fluidInterior_m: { x: x-2*t, y: y-2*t, z: z-2*t },
    linearVelocity_m_s: { x: 0, y: 0, z: scalar(e, 'velocityZ') }, angularVelocity_rad_s: { x: scalar(e, 'angularX'), y: 0, z: 0 },
    colliders: panels.map(([id, p, d]) => ({ id, offset_m: { x: p[0], y: p[1], z: p[2] }, dimensions_m: { x: d[0], y: d[1], z: d[2] } })),
  } satisfies BodyRecipe;
} };
