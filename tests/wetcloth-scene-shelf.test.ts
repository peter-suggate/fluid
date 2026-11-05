import assert from 'node:assert/strict';
import test from 'node:test';
import {readFile} from 'node:fs/promises';
import '../lib/methods';
import {findSceneDefinition, getScenePreset, cameraForPreset, SCENE_CATALOG} from '../lib/core/scenes';
import {sceneCardForDefinition, sceneDefinitionPrepared, duplicateSceneDefinitionIds} from '../lib/core/scene-definition';
import {sceneSections, savedSceneCard} from '../lib/core/scene-cards';
import {loadSceneFromLibrary, saveSceneToLibrary, readSceneLibrary, type SceneLibraryStorage} from '../lib/core/scene-library';
import {parseQueryState, serializeQueryState} from '../lib/core/url-state';
import {prepareWetClothSource} from '../lib/cloth/scenes/definitions';
import {WETCLOTH_SHELF, wetClothShelfDefinitions} from '../lib/cloth/scenes/shelf';
import {sourceCameraForScene} from '../lib/cloth/source-camera';
import {WETCLOTH_UPSTREAM_COMMIT} from '../lib/cloth/scenes/sources';
import {clothSurfaceAt} from '../lib/cloth/scene-geometry';
import {sceneWithClothStroke} from '../lib/cloth/scene-edit';
import {cameraBasis, cameraPosition} from '../lib/core/math';

const request: typeof fetch = async url => new Response(await readFile(`public${String(url)}`));
const SIMPLE = 'wetcloth-source-simple-cloth';

test('the wet-cloth shelf is one study shelf, small fixtures first, each card attributed and honest', () => {
  assert.deepEqual(duplicateSceneDefinitionIds(SCENE_CATALOG), []);
  const section = sceneSections([]).find(s => s.id === 'study')!;
  const shelf = section.shelves.find(s => s.shelf === WETCLOTH_SHELF)!;
  assert.ok(shelf, 'study section offers the wet-cloth shelf');
  assert.deepEqual(shelf.cards.slice(0, 8).map(c => c.id), [
    'simple-cloth', 'simple-plastic', 'simple-yarn', 'simple-yarn-loop', 'capillary-suction',
    'dam-break-nonlinear-drag-small', 'buoyancy-test-with', 'buoyancy-test-without'].map(n => `wetcloth-source-${n}`));
  assert.equal(shelf.cards.length, wetClothShelfDefinitions.length);
  for (const card of shelf.cards) {
    assert.ok(card.attribution!.url.includes(`/tree/${WETCLOTH_UPSTREAM_COMMIT}/`) || card.attribution!.url.includes(WETCLOTH_UPSTREAM_COMMIT), card.id);
    assert.match(card.status!, /not reproduced/);
    assert.ok(card.prepare && card.prepared);
  }
});

test('an unloaded wet-cloth scene refuses to build and never blocks synchronous preset reads', () => {
  const definition = findSceneDefinition('wetcloth-source-tighten-towel')!;
  assert.equal(sceneDefinitionPrepared(definition), false);
  assert.throws(() => sceneCardForDefinition(definition).open(), /prepareWetClothSource/);
  const preset = getScenePreset(definition.id);
  assert.equal(preset.presentationMode, 'full-scene');
  assert.ok(cameraForPreset(preset));
  // A link to it before its asset arrives falls back rather than throwing.
  assert.notEqual(parseQueryState(`?scene=${definition.id}`).presetId, definition.id);
});

test('opening a prepared card builds the source scene under its released camera, and links round-trip', async () => {
  const definition = findSceneDefinition(SIMPLE)!;
  await prepareWetClothSource(SIMPLE, request);
  assert.ok(sceneDefinitionPrepared(definition));
  const opening = sceneCardForDefinition(definition).open();
  assert.equal(opening.presetId, SIMPLE);
  assert.equal(opening.scene.systems?.fluid, false);
  assert.equal(opening.scene.cloth?.[0]?.geometry.kind, 'inline');
  const source = sourceCameraForScene(opening.scene)!.camera;
  assert.deepEqual(opening.camera!.target_m, source.target_m);
  assert.equal(opening.camera!.distance_m, source.distance_m);
  assert.equal(opening.camera!.roll_rad, source.roll_rad);
  assert.equal(opening.methodProfile?.overrides?.timeStep, 'scene');

  const parsed = parseQueryState(`?scene=${SIMPLE}`);
  assert.equal(parsed.presetId, SIMPLE);
  assert.deepEqual(parsed.scene.cloth, opening.scene.cloth);
  const search = serializeQueryState('', {presetId: SIMPLE, scene: parsed.scene},
    {methodId: 'uniform-narrow-band-flip', quality: 'high', overrides: {}});
  assert.equal(new URLSearchParams(search).get('scene'), SIMPLE);
  // An unedited imported scene writes no document keys: the link is its name.
  assert.deepEqual([...new URLSearchParams(search).keys()].filter(k => k.startsWith('scene.')), []);
});

test('a saved, cut wet-cloth scene round-trips through the library with its inline topology and framing', async () => {
  await prepareWetClothSource(SIMPLE, request);
  const definition = findSceneDefinition(SIMPLE)!;
  const scene = sceneCardForDefinition(definition).open().scene;
  const topology = scene.cloth![0]!.geometry.kind === 'inline' ? scene.cloth![0]!.geometry.data : undefined;
  assert.ok(topology);
  // Aim the released camera at the cloth centroid and pick it.
  const view = sourceCameraForScene(scene)!.camera, eye = cameraPosition(view);
  // A triangle's centroid, so the ray cannot pass through a gap in the sheet.
  const p = topology.restPositions_m, t = topology.triangles.slice(3 * 20, 3 * 21);
  const c = [0, 1, 2].map(a => t.reduce((s, v) => s + p[3 * v + a]!, 0) / 3);
  const d = [c[0]! - eye.x, c[1]! - eye.y, c[2]! - eye.z], len = Math.hypot(...d);
  const hit = clothSurfaceAt(scene, {origin: eye, direction: {x: d[0]! / len, y: d[1]! / len, z: d[2]! / len}});
  assert.ok(hit, 'the released camera sees the cloth');
  assert.ok(cameraBasis(view));
  // A local box cut through the hit point opens the cloth and keeps the rest.
  const at = [eye.x + d[0]! / len * hit!.distance_m, eye.y + d[1]! / len * hit!.distance_m, eye.z + d[2]! / len * hit!.distance_m];
  const r = 0.02;
  const cut = sceneWithClothStroke(scene, [{kind: 'box', box: {minimum_m: at.map(v => v - r) as [number, number, number], maximum_m: at.map(v => v + r) as [number, number, number]}}]);
  assert.notEqual(cut, scene);
  const cutTopology = cut.cloth![0]!.geometry.kind === 'inline' ? cut.cloth![0]!.geometry.data : undefined;
  assert.ok(cutTopology && cutTopology.triangles.length < topology.triangles.length, 'the cut removes local triangles');

  const store = new Map<string, string>();
  const storage: SceneLibraryStorage = {getItem: k => store.get(k) ?? null, setItem: (k, v) => { store.set(k, v); }};
  const saved = saveSceneToLibrary(storage, 'Cut cloth', cut, SIMPLE, {savedAt_ms: 1});
  assert.ok(saved.persisted);
  const entry = readSceneLibrary(storage).find(e => e.id === saved.entry.id)!;
  const restored = loadSceneFromLibrary(entry);
  assert.deepEqual(restored.cloth, JSON.parse(JSON.stringify(cut.cloth)));
  assert.deepEqual(restored.wetClothSource, JSON.parse(JSON.stringify(cut.wetClothSource)));
  const opening = savedSceneCard(entry).open();
  assert.deepEqual(opening.camera!.target_m, view.target_m);
});
