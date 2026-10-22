import assert from 'node:assert/strict';
import test from 'node:test';
import { createEmptyScene, minimalRoomSceneryGraph } from '../../core/empty-scene';
import { bakeProceduralSolid } from '../../core/procedural-solid';
import { initializeRigidBody } from '../../core/rigid-body';
import { sceneWithSolidStroke } from '../../core/solid-world';
import { compileSolidChunks, SolidCollisionCache } from '../adapters/solid-world/collision';
import { assertNoSolidOverlap } from '../adapters/solid-world/overlap';

function scene() {
  const s = createEmptyScene({ extents_m: { x: 6, y: 4, z: 6 }, finestCellSize_m: .25 });
  s.scenery = undefined; s.solidVoxels = []; s.solidBoundary = undefined; s.terrain = undefined;
  return s;
}

test('collision cache retains untouched pages and removes emptied chunks before undo', () => {
  const base = scene();
  base.solidVoxels = [
    { operation: 'fill', minimum: [0, 0, 0], maximumExclusive: [8, 2, 8] },
    { operation: 'fill', minimum: [16, 0, 0], maximumExclusive: [24, 2, 8] },
  ];
  const cache = new SolidCollisionCache(), original = compileSolidChunks(base, [], cache);
  const draw = sceneWithSolidStroke(base, [{ operation: 'fill', minimum: [1, 2, 1], maximumExclusive: [3, 4, 3] }]);
  const drawn = compileSolidChunks(draw, [], cache);
  assert.deepEqual(drawn, compileSolidChunks(draw));
  assert.equal(drawn.find(c => c.key === '2,0,0'), original.find(c => c.key === '2,0,0'));
  assert.notEqual(drawn.find(c => c.key === '0,0,0'), original.find(c => c.key === '0,0,0'));
  const cleared = sceneWithSolidStroke(draw, [{ operation: 'clear', minimum: [0, 0, 0], maximumExclusive: [8, 8, 8] }]);
  const cut = compileSolidChunks(cleared, [], cache);
  assert.deepEqual(cut, compileSolidChunks(cleared));
  assert.equal(cut.some(c => c.key === '0,0,0'), false);
  assert.deepEqual(compileSolidChunks(base, [], cache), original);
  const resized = { ...base, container: { ...base.container, width_m: 8 } };
  assert.deepEqual(compileSolidChunks(resized, [], cache), compileSolidChunks(resized));
});

test('sampled collision cache invalidates rounded cuts, node replacements and undo', () => {
  const base = createEmptyScene({ extents_m: { x: 4, y: 2, z: 2 }, finestCellSize_m: .25 });
  base.scenery = undefined;
  const baked = bakeProceduralSolid([16, 8, 8], [.25, .25, .25], p => p[1] - 1 - p[0] * .2);
  base.solidVoxels = baked.solidVoxels; base.solidBoundary = baked.solidBoundary;
  const cache = new SolidCollisionCache(), original = compileSolidChunks(base, [], cache);
  const cut = sceneWithSolidStroke(base, [{ operation: 'clear', kind: 'ellipsoid', affectsScenery: true,
    minimum: [6, 0, 2], maximumExclusive: [10, 8, 6] }]);
  assert.deepEqual(compileSolidChunks(cut, [], cache), compileSolidChunks(cut));
  assert.notDeepEqual(compileSolidChunks(cut, [], cache), original);
  assert.deepEqual(compileSolidChunks(base, [], cache), original);
  const shifted = { ...base, solidBoundary: { ...baked.solidBoundary,
    bricks: baked.solidBoundary.bricks.map(b => ({ ...b, values: b.values.map(v => v - .05) })) } };
  assert.deepEqual(compileSolidChunks(shifted, [], cache), compileSolidChunks(shifted));
  assert.notDeepEqual(compileSolidChunks(shifted, [], cache), original);
});

test('ground collision cache follows body coverage and invalidates changed terrain', () => {
  const base = createEmptyScene({ extents_m: { x: 16, y: 4, z: 8 }, finestCellSize_m: .5 });
  base.scenery = undefined; base.solidVoxels = []; base.terrain = { baseHeight_m: .5, features: [] };
  const cache = new SolidCollisionCache(), left = [{ x: -5, y: 1, z: 0 }], right = [{ x: 5, y: 1, z: 0 }];
  const first = compileSolidChunks(base, left, cache), moved = compileSolidChunks(base, right, cache);
  assert.ok(first.length && moved.length);
  assert.deepEqual(moved, compileSolidChunks(base, right));
  assert.ok(first.some(chunk => !moved.some(other => chunk.key === other.key)));
  const equal = compileSolidChunks(base, right, cache);
  assert.equal(equal[0], moved[0]);
  const raised = { ...base, terrain: { baseHeight_m: 1, features: [] } };
  assert.deepEqual(compileSolidChunks(raised, right, cache), compileSolidChunks(raised, right));
  assert.notDeepEqual(compileSolidChunks(raised, right, cache), moved);
});

test('overlap broad phase uses current body pose and still rejects nearby fill and undo', () => {
  const base = scene();
  const body = initializeRigidBody({ id: 'probe', name: 'Probe', shape: 'sphere', motion: 'dynamic', dimensions_m: { x: .2, y: .2, z: .2 },
    position_m: { x: 0, y: 1, z: 0 }, orientation: { x: 0, y: 0, z: 0, w: 1 }, density_kg_m3: 1000,
    friction: .5, restitution: 0, linearVelocity_m_s: { x: 0, y: 0, z: 0 }, angularVelocity_rad_s: { x: 0, y: 0, z: 0 } });
  // Build the base image before appending, preserving unchanged page identities.
  compileSolidChunks(base);
  const remote = sceneWithSolidStroke(base, [{ operation: 'fill', minimum: [19, 3, 11], maximumExclusive: [21, 5, 13] }]);
  assert.doesNotThrow(() => assertNoSolidOverlap(remote, [body], base));
  assert.throws(() => assertNoSolidOverlap(remote, [{ ...body, position_m: { x: 2, y: 1, z: 0 } }], base), /overlaps/);
  const filled = sceneWithSolidStroke(base, [{ operation: 'fill', minimum: [11, 3, 11], maximumExclusive: [13, 5, 13] }]);
  assert.throws(() => assertNoSolidOverlap(filled, [body], base), /overlaps/);
  const cut = sceneWithSolidStroke(filled, [{ operation: 'clear', minimum: [11, 3, 11], maximumExclusive: [13, 5, 13] }]);
  assert.doesNotThrow(() => assertNoSolidOverlap(cut, [body], filled));
  assert.throws(() => assertNoSolidOverlap(filled, [body], cut), /overlaps/);
});

test('scenery collider changes invalidate cached chunks even without written voxel pages', () => {
  const base = scene();
  const template = minimalRoomSceneryGraph.nodes[0]!;
  if (template.kind !== 'room-shell') throw new Error('Expected the empty room fixture shell');
  const shell = { ...template, faces: [] };
  const box = { kind: 'box' as const, id: 'collider', tags: ['fluid-collider'],
    place: { position: { x: 0, y: .5, z: 0 }, units: 'metres' as const },
    halfSize: { x: .5, y: .5, z: .5 }, material: { colorLinear: [.5, .5, .5] as const } };
  base.scenery = { palettes: minimalRoomSceneryGraph.palettes, nodes: [shell, box] };
  const cache = new SolidCollisionCache(), original = compileSolidChunks(base, [], cache);
  assert.ok(original.length);
  const moved = { ...base, scenery: { ...base.scenery, nodes: [shell, { ...box, place: { ...box.place, position: { x: 2, y: .5, z: 0 } } }] } };
  assert.deepEqual(compileSolidChunks(moved, [], cache), compileSolidChunks(moved));
  assert.notDeepEqual(compileSolidChunks(moved, [], cache), original);
  const removed = { ...base, scenery: { ...base.scenery, nodes: [shell] } };
  assert.deepEqual(compileSolidChunks(removed, [], cache), []);
  assert.deepEqual(compileSolidChunks(base, [], cache), original);
});
