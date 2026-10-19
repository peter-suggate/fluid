import type { EntityKindPlugin } from '../../contracts';
import { character } from '../character/definition';

/** Uses the standard character motor and an independently animated Blender skin. */
export const robot: EntityKindPlugin = {
  ...character,
  id: 'fluid.robot',
  schema: { ...character.schema, defaults: { ...character.schema.defaults,
    radius: .38, height: 1, speed: 2.2, runMultiplier: 1.5, jumpHeight: .85,
    density: 650, acceleration: 22, stepHeight: .32 } },
  editor: { ...character.editor, label: 'Bub · pool robot', order: 1,
    hint: 'A little poolside droid. WASD / arrows move, Space hops, hold Space at a ladder to climb, F somersaults.' },
};
