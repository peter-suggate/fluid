import assert from "node:assert/strict";
import test from "node:test";
import { cloneScene, defaultScene, serializeScene } from "../lib/core/model";
import { SCENE_LIBRARY_STORAGE_KEY, SCENE_LIBRARY_LIMIT, SCENE_AUTOSAVE_ENTRY_ID, renameSceneInLibrary, deleteSceneFromLibrary, saveSceneToLibrary, readSceneLibrary, loadSceneFromLibrary, type SceneLibraryStorage } from "../lib/core/scene-library";

function quotaStorage(limit: number) {
  const values = new Map<string, string>();
  let quota = limit;
  const storage: SceneLibraryStorage = {
    getItem: key => values.get(key) ?? null,
    setItem(key, value) {
      if (value.length * 2 > quota) throw new Error("QuotaExceededError");
      values.set(key, value);
    },
  };
  return { storage, values, setQuota: (value: number) => { quota = value; } };
}

test("large scene and identical autosave share one compact document under quota", () => {
  const scene = cloneScene(defaultScene);
  // A large numeric payload models authored terrain without a synthetic compression advantage.
  scene.sceneId = "terrain-fixture";
  scene.terrain = { baseHeight_m: 0, features: [], grid: {
    kind: "grid", origin_m: { x: -1, z: -1 }, spacing_m: .01,
    size: { nx: 256, nz: 512 },
    heights_m: Array.from({ length: 256 * 512 }, (_, index) => (index % 997) / 9970),
  } };
  const compact = JSON.stringify(scene);
  assert.ok(compact.length > 2_000_000);
  const fixture = quotaStorage((compact.length + 1500) * 2);
  const first = saveSceneToLibrary(fixture.storage, "Terrain", scene, "terrain", { savedAt_ms: 1 });
  assert.equal(first.persisted, true);
  const second = saveSceneToLibrary(fixture.storage, "Working", scene, "terrain", { savedAt_ms: 2, replaceId: SCENE_AUTOSAVE_ENTRY_ID });
  assert.equal(second.persisted, true);
  const raw = fixture.values.get(SCENE_LIBRARY_STORAGE_KEY)!;
  const stored = JSON.parse(raw);
  assert.equal(stored.documents.length, 1);
  assert.equal(stored.documents[0], compact);
  assert.equal(readSceneLibrary(fixture.storage).length, 2);
  assert.deepEqual(loadSceneFromLibrary(first.entry), loadSceneFromLibrary(second.entry));
  assert.ok(serializeScene(scene).includes("\n  "), "portable exports remain formatted");
});

test("legacy pretty entries migrate atomically without removing names or changing scenes", () => {
  const fixture = quotaStorage(Infinity);
  const scene = cloneScene(defaultScene);
  const legacy = { id: "legacy", name: "Keep me", savedAt_ms: 1, presetId: "fixture", scene: serializeScene(scene) };
  fixture.storage.setItem(SCENE_LIBRARY_STORAGE_KEY, JSON.stringify([legacy]));
  const saved = saveSceneToLibrary(fixture.storage, "Second", scene, "fixture", { savedAt_ms: 2 });
  assert.equal(saved.persisted, true);
  const reopened = readSceneLibrary(fixture.storage);
  assert.deepEqual(reopened.map(entry => entry.name), ["Second", "Keep me"]);
  assert.deepEqual(loadSceneFromLibrary(reopened[1]!), loadSceneFromLibrary(legacy));
  assert.equal(JSON.parse(fixture.values.get(SCENE_LIBRARY_STORAGE_KEY)!).documents.length, 1);
});

test("quota rejection reports failure and preserves every byte of previous library", () => {
  const fixture = quotaStorage(Infinity);
  saveSceneToLibrary(fixture.storage, "Keep", defaultScene, "fixture", { savedAt_ms: 1 });
  const before = fixture.values.get(SCENE_LIBRARY_STORAGE_KEY);
  fixture.setQuota(1);
  const failed = saveSceneToLibrary(fixture.storage, "Cannot fit", defaultScene, "fixture", { savedAt_ms: 2 });
  assert.equal(failed.persisted, false);
  assert.equal(fixture.values.get(SCENE_LIBRARY_STORAGE_KEY), before);
  assert.deepEqual(failed.entries.map(entry => entry.name), ["Keep"]);
  assert.equal(saveSceneToLibrary(undefined, "No storage", defaultScene, "fixture", { savedAt_ms: 3 }).persisted, false);
});

test("autosave retries unchanged document after failed quota write", async () => {
  await import("../lib/methods");
  const { createSceneAutosave } = await import("../lib/core/scene-autosave");
  const fixture = quotaStorage(1);
  const autosave = createSceneAutosave({ storage: fixture.storage });
  const working = { scene: defaultScene, presetId: "fixture" };
  autosave.request(working); autosave.flush();
  assert.equal(readSceneLibrary(fixture.storage).length, 0);
  fixture.setQuota(Infinity);
  autosave.request(working); autosave.flush();
  assert.equal(readSceneLibrary(fixture.storage)[0]?.id, SCENE_AUTOSAVE_ENTRY_ID);
  autosave.cancel();
});


test("renaming and deleting a pooled save preserve its sibling and Continue document", async () => {
  await import("../lib/methods");
  const { sceneResume } = await import("../lib/core/scene-autosave");
  const fixture = quotaStorage(Infinity);
  const named = saveSceneToLibrary(fixture.storage, "Named", defaultScene, "fixture", { savedAt_ms: 1 });
  saveSceneToLibrary(fixture.storage, "Working", defaultScene, "fixture", { savedAt_ms: 2, replaceId: SCENE_AUTOSAVE_ENTRY_ID });
  const renamed = renameSceneInLibrary(fixture.storage, named.entry.id, "Renamed");
  assert.equal(renamed.find(entry => entry.id === named.entry.id)?.name, "Renamed");
  assert.equal(renamed.find(entry => entry.id === named.entry.id)?.savedAt_ms, 1);
  const afterDelete = deleteSceneFromLibrary(fixture.storage, named.entry.id);
  assert.deepEqual(afterDelete.map(entry => entry.id), [SCENE_AUTOSAVE_ENTRY_ID]);
  const resume = sceneResume(afterDelete);
  assert.equal(resume?.autosaved, true);
  assert.deepEqual(resume?.card.open().scene, loadSceneFromLibrary(afterDelete[0]!));
  assert.equal(JSON.parse(fixture.values.get(SCENE_LIBRARY_STORAGE_KEY)!).documents.length, 1);
});

test("full library rejects additions without eviction but permits overwrite and deletion", () => {
  const fixture = quotaStorage(Infinity);
  for (let index = 0; index < SCENE_LIBRARY_LIMIT; index++) {
    assert.equal(saveSceneToLibrary(fixture.storage, `Scene ${index}`, defaultScene, "fixture", { savedAt_ms: index }).persisted, true);
  }
  const before = fixture.values.get(SCENE_LIBRARY_STORAGE_KEY);
  assert.equal(saveSceneToLibrary(fixture.storage, "Overflow", defaultScene, "fixture", { savedAt_ms: 100 }).persisted, false);
  assert.equal(fixture.values.get(SCENE_LIBRARY_STORAGE_KEY), before);
  assert.equal(saveSceneToLibrary(fixture.storage, "Scene 0", defaultScene, "fixture", { savedAt_ms: 101 }).persisted, true);
  const entries = readSceneLibrary(fixture.storage);
  assert.equal(entries.length, SCENE_LIBRARY_LIMIT);
  assert.equal(deleteSceneFromLibrary(fixture.storage, entries[0]!.id).length, SCENE_LIBRARY_LIMIT - 1);
});

test("malformed pooled references cannot masquerade as usable entries", () => {
  const fixture = quotaStorage(Infinity);
  const entry = { id: "valid", name: "Valid", savedAt_ms: 1, presetId: "fixture", document: 0 };
  fixture.storage.setItem(SCENE_LIBRARY_STORAGE_KEY, JSON.stringify({ version: 2, documents: [JSON.stringify(defaultScene)], entries: [
    entry, { ...entry, id: "missing", document: 2 }, { ...entry, id: "negative", document: -1 },
    { ...entry, id: "fraction", document: .5 },
  ] }));
  assert.deepEqual(readSceneLibrary(fixture.storage).map(entry => entry.id), ["valid"]);
});
