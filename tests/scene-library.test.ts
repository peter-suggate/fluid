import assert from "node:assert/strict";
import test from "node:test";
import { cloneScene, defaultScene, serializeScene } from "../lib/model";
import {
  deleteSceneFromLibrary,
  loadSceneFromLibrary,
  normalizeSceneName,
  readSceneLibrary,
  renameSceneInLibrary,
  saveSceneToLibrary,
  SCENE_LIBRARY_LIMIT,
  SCENE_LIBRARY_STORAGE_KEY,
  SCENE_NAME_MAXIMUM_LENGTH,
  type SceneLibraryStorage,
} from "../lib/scene-library";

function memoryStorage(initial?: string): SceneLibraryStorage & { raw(): string | null } {
  let value = initial ?? null;
  return {
    getItem: () => value,
    setItem: (_key, next) => { value = next; },
    raw: () => value,
  };
}

test("saved scenes round-trip byte-identically to the file export", () => {
  const storage = memoryStorage();
  const scene = cloneScene(defaultScene);
  const { entry } = saveSceneToLibrary(storage, "Pond", scene, "garden-pond", { savedAt_ms: 1_000 });
  assert.equal(entry.scene, serializeScene(scene), "an entry is exactly what the download would contain");
  const loaded = loadSceneFromLibrary(entry);
  assert.equal(serializeScene(loaded), serializeScene(scene));
  assert.equal(entry.presetId, "garden-pond");
});

test("saving the same name replaces rather than duplicating", () => {
  const storage = memoryStorage();
  const scene = cloneScene(defaultScene);
  saveSceneToLibrary(storage, "Pond", scene, "garden-pond", { savedAt_ms: 1_000 });
  const second = cloneScene(defaultScene);
  second.randomSeed = 77;
  const { entries } = saveSceneToLibrary(storage, "  pond  ", second, "garden-pond", { savedAt_ms: 2_000 });
  assert.equal(entries.length, 1, "case and whitespace differences are the same name");
  assert.equal(loadSceneFromLibrary(entries[0]!).randomSeed, 77, "the newer document wins");

  assert.equal(entries[0]?.name, "pond", "re-saving adopts the newly typed name, so casing can be corrected");

  saveSceneToLibrary(storage, "Other", second, "garden-pond", { savedAt_ms: 3_000 });
  const listed = readSceneLibrary(storage);
  assert.deepEqual(listed.map((item) => item.name), ["Other", "pond"], "newest first");
});

test("names are normalized and bounded", () => {
  assert.equal(normalizeSceneName("  a   b  "), "a b");
  assert.equal(normalizeSceneName("   "), "Untitled scene");
  assert.equal(normalizeSceneName("x".repeat(500)).length, SCENE_NAME_MAXIMUM_LENGTH);
});

test("rename and delete operate on the stored list", () => {
  const storage = memoryStorage();
  const scene = cloneScene(defaultScene);
  const { entry } = saveSceneToLibrary(storage, "Pond", scene, "garden-pond", { savedAt_ms: 1_000 });
  assert.equal(renameSceneInLibrary(storage, entry.id, " Lake  ")[0]?.name, "Lake");
  assert.deepEqual(deleteSceneFromLibrary(storage, entry.id), []);
  assert.deepEqual(readSceneLibrary(storage), []);
  assert.deepEqual(deleteSceneFromLibrary(storage, "missing"), [], "deleting an absent id is inert");
});

test("the library is bounded and drops the oldest entries", () => {
  const storage = memoryStorage();
  const scene = cloneScene(defaultScene);
  for (let index = 0; index < SCENE_LIBRARY_LIMIT + 5; index += 1) {
    saveSceneToLibrary(storage, `scene ${index}`, scene, "garden-pond", { savedAt_ms: 1_000 + index });
  }
  const entries = readSceneLibrary(storage);
  assert.equal(entries.length, SCENE_LIBRARY_LIMIT);
  assert.equal(entries[0]?.name, `scene ${SCENE_LIBRARY_LIMIT + 4}`);
  assert.equal(entries.at(-1)?.name, "scene 5", "the five oldest were evicted");
});

test("unusable or corrupt storage degrades to an empty library", () => {
  assert.deepEqual(readSceneLibrary(undefined), []);
  assert.deepEqual(readSceneLibrary(memoryStorage("not json")), []);
  assert.deepEqual(readSceneLibrary(memoryStorage(JSON.stringify({ nope: true }))), []);
  assert.deepEqual(readSceneLibrary(memoryStorage(JSON.stringify([{ id: "a" }]))), [], "malformed entries are skipped");

  // Saving with no storage still returns the list the caller should render.
  const { entries } = saveSceneToLibrary(undefined, "Pond", cloneScene(defaultScene), "garden-pond", { savedAt_ms: 1 });
  assert.equal(entries.length, 1);

  const throwing: SceneLibraryStorage = {
    getItem: () => { throw new Error("blocked"); },
    setItem: () => { throw new Error("quota"); },
  };
  assert.deepEqual(readSceneLibrary(throwing), []);
  assert.equal(saveSceneToLibrary(throwing, "Pond", cloneScene(defaultScene), "p", { savedAt_ms: 1 }).entries.length, 1);
});

test("a stored scene that fails validation reports instead of corrupting the session", () => {
  const broken = { id: "x", name: "x", savedAt_ms: 1, presetId: "p", scene: JSON.stringify({ ...cloneScene(defaultScene), schemaVersion: "9.9.9" }) };
  assert.throws(() => loadSceneFromLibrary(broken), /schema version/i);
  assert.equal(SCENE_LIBRARY_STORAGE_KEY, "fluid-lab.scene-library.v1");
});
