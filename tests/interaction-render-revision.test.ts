import assert from 'node:assert/strict';
import test from 'node:test';
import { defaultScene, markSceneRevision, sceneRevision } from '../lib/core/model';
import { sceneHoldsSolidWorld, sceneWithSolidStroke, solidWorldForScene } from '../lib/core/solid-world';
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


test('interaction render derivatives retain existing solid snapshots and unchanged stroke pages', () => {
  const source = markSceneRevision({ ...interactiveScene(), solidVoxels: [
    { operation: 'fill' as const, minimum: [0, 0, 0] as const, maximumExclusive: [1, 1, 1] as const },
    { operation: 'fill' as const, minimum: [16, 0, 0] as const, maximumExclusive: [17, 1, 1] as const },
  ] });
  const world = solidWorldForScene(source);
  assert.equal(solidWorldForScene(interactionScene(source)), world);
  const edited = markSceneRevision(sceneWithSolidStroke(source, [
    { operation: 'fill', minimum: [1, 0, 0], maximumExclusive: [2, 1, 1] },
  ]));
  const accepted = solidWorldForScene(edited);
  assert.equal(accepted.pages[1], world.pages[1]);
  // Replace the incidental global last-world cache, as an independent render
  // consumer can do; the derivative must carry its own accepted image.
  solidWorldForScene({ ...source, solidVoxels: [] });
  assert.equal(solidWorldForScene(interactionScene(edited)), accepted);
});

test('interaction derivatives do not compose solids eagerly and share a later startup image back', () => {
  const source = markSceneRevision({ ...interactiveScene(), solidVoxels: [
    { operation: 'fill' as const, minimum: [0, 0, 0] as const, maximumExclusive: [1, 1, 1] as const },
  ] });
  const derived = interactionScene(source);
  assert.equal(sceneHoldsSolidWorld(source), false);
  assert.equal(sceneHoldsSolidWorld(derived), false);
  const startup = solidWorldForScene(derived);
  assert.equal(interactionScene(source), derived);
  assert.equal(solidWorldForScene(source), startup);
});
