import assert from 'node:assert/strict';
import test from 'node:test';
import { defaultScene, markSceneRevision, sceneRevision } from '../lib/core/model';
import { interactionScene } from '../lib/interaction/adapters/scene';
import { interactionCatalog } from '../lib/interaction/installed';

function interactiveScene() {
  return { ...defaultScene, interaction: { schemaVersion: 1 as const, connections: [], entities: [
    interactionCatalog.createEntity({ kind: 'fluid.crate', id: 'crate', name: 'Crate', transform: {
      position_m: { x: 0, y: 1, z: 0 }, orientation: { x: 0, y: 0, z: 0, w: 1 },
    } }),
  ] } };
}

test('interactive render derivatives preserve versioned publication without changing the authored roster', () => {
  const source = markSceneRevision(interactiveScene());
  const derived = interactionScene(source);
  assert.notEqual(derived, source);
  assert.ok(sceneRevision(derived) !== undefined, 'steady draws must not serialize the document');
  assert.notEqual(sceneRevision(derived), sceneRevision(source));
  assert.equal(interactionScene(source), derived);
  assert.equal(interactionScene(derived), derived);
  assert.equal(derived.rigidBodies.length, source.rigidBodies.length + 1);
  const edited = markSceneRevision({ ...source, surfaceStyle: 'voxel-flat' });
  assert.notEqual(sceneRevision(interactionScene(edited)), sceneRevision(derived), 'edits still publish');
});

test('unsupported interaction documents also retain their revision; external inputs keep fallback', () => {
  const unsupported = interactiveScene();
  const entity = unsupported.interaction.entities[0]!;
  unsupported.interaction.entities = [{ ...entity, behaviors: [
    { ...entity.behaviors[0]!, type: 'unsupported' },
  ] }];
  const source = markSceneRevision(unsupported);
  assert.ok(sceneRevision(interactionScene(source)) !== undefined);
  assert.deepEqual(interactionScene(source).rigidBodies, source.rigidBodies);
  assert.equal(sceneRevision(interactionScene(interactiveScene())), undefined);
});
