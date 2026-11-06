import test from 'node:test';
import assert from 'node:assert/strict';
import { defaultScene, type SceneDescription } from '../lib/core/model';
import { assertFreeSpaceSeedMargin, FREE_SPACE_SEED_MARGIN_CELLS } from '../lib/core/initial-liquid-lattice';
import { sceneFluidLatticeDimensions, sceneLatticeDimensions } from '../lib/core/scene-lattice-dimensions';

function scene(boundary: 'free-space' | undefined, liquid?: { lo: number[]; hi: number[] }): SceneDescription {
 const s = structuredClone(defaultScene);
 s.container = {width_m: .64, height_m: .64, depth_m: .64, fillFraction: liquid ? 0 : .75, top: 'open', fluidWallMode: 'free-slip', ...(boundary ? {boundary} : {})};
 s.voxelDomain = {finestCellSize_m: .01, brickSize_cells: 8}; s.nominalResolution = {length_m: .01};
 s.solidVoxels = []; s.rigidBodies = []; delete s.scenery; delete s.terrain; delete s.solidBoundary;
 s.fluid = {density_kg_m3: 1000, dynamicViscosity_Pa_s: 0, surfaceTension_N_m: 0, gravity_m_s2: {x: 0, y: -9.80665, z: 0}, initialCondition: 'tank-fill',
  ...(liquid ? {initialLiquidVolumes: [{shape: 'box' as const, min_m: {x: liquid.lo[0]!, y: liquid.lo[1]!, z: liquid.lo[2]!}, max_m: {x: liquid.hi[0]!, y: liquid.hi[1]!, z: liquid.hi[2]!}}]} : {})};
 return s;
}
const check = (s: SceneDescription) => assertFreeSpaceSeedMargin(s, sceneLatticeDimensions(s, Number.MAX_SAFE_INTEGER), sceneFluidLatticeDimensions(s, Number.MAX_SAFE_INTEGER));

test('free-space initial liquid inside the seed window\'s far-air margin is refused, with its face and distance', () => {
 const m = FREE_SPACE_SEED_MARGIN_CELLS * .01;
 // A free-space tank fill touches four sides and the floor of the window.
 assert.throws(() => check(scene('free-space')), /low x face, inside the .* far-air margin/);
 // The same fill in a walled tank has walls there, not far air.
 assert.doesNotThrow(() => check(scene(undefined)));
 // Exactly the margin from every face is accepted; a cell less on one face is not.
 assert.doesNotThrow(() => check(scene('free-space', {lo: [-.32 + m, m, -.32 + m], hi: [.32 - m, .64 - m, .32 - m]})));
 assert.throws(() => check(scene('free-space', {lo: [-.32 + m, m, -.32 + m], hi: [.32 - m, .64 - m, .32 - m + .01]})), /high z face/);
 assert.throws(() => check(scene('free-space', {lo: [-.32 + m, m - .01, -.32 + m], hi: [.32 - m, .64 - m, .32 - m]})), /low y face/);
 // A window with no initial liquid has nothing to certify.
 const dry = scene('free-space'); dry.container.fillFraction = 0;
 assert.doesNotThrow(() => check(dry));
});
