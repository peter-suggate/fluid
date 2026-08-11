import assert from "node:assert/strict";
import test from "node:test";
import { setTimeout as delay } from "node:timers/promises";
import { getMethod, resolveMethodValues } from "../lib/methods";
import { canonicalScene, cloneScene, serializeScene, validateScene, type SceneDescription } from "../lib/model";
import { allSceneCards, sceneCardPreview, sceneSections } from "../lib/scene-cards";
import {
  createSceneAutosave,
  readSceneAutosave,
  sceneAutosaveName,
  sceneResume,
  startSceneAutosave,
  writeSceneAutosave,
  SCENE_AUTOSAVE_NAME,
} from "../lib/scene-autosave";
import {
  loadSceneFromLibrary,
  readSceneLibrary,
  saveSceneToLibrary,
  savedSceneEntries,
  SCENE_AUTOSAVE_ENTRY_ID,
  SCENE_LIBRARY_STORAGE_KEY,
  type SceneLibraryStorage,
} from "../lib/scene-library";
import { getScenePreset, POWER2017_FACTOR4_BENCHMARK_METHOD_PROFILE } from "../lib/scenes";
import { useSceneStore } from "../lib/stores/scene-store";
import { useShellStore } from "../lib/stores/shell-store";
import { useMethodStore } from "../lib/stores/method-store";

/** In-memory storage that counts writes, so a debounce is observable. */
function memoryStorage(initial?: string): SceneLibraryStorage & { writes: number } {
  let value = initial ?? null;
  const storage = {
    writes: 0,
    getItem: () => value,
    setItem: (_key: string, next: string) => { value = next; storage.writes += 1; },
  };
  return storage;
}

function pond(patch: Partial<SceneDescription> = {}): SceneDescription {
  return { ...cloneScene(getScenePreset("garden-pond").create()), ...patch };
}

test("a drag's worth of edits collapses to one write", async () => {
  const storage = memoryStorage();
  const autosave = createSceneAutosave({ storage, delay_ms: 10 });
  for (let index = 0; index < 20; index += 1) {
    autosave.request({ scene: pond({ randomSeed: index }), presetId: "garden-pond" });
  }
  assert.equal(storage.writes, 0, "nothing is written while the gesture is still moving");

  await delay(60);
  assert.equal(storage.writes, 1);
  const entry = readSceneAutosave(readSceneLibrary(storage));
  assert.equal(loadSceneFromLibrary(entry!).randomSeed, 19, "the write is the document as it was left");

  // A debounce, not a once: work after the pause is still saved.
  autosave.request({ scene: pond({ randomSeed: 42 }), presetId: "garden-pond" });
  await delay(60);
  assert.equal(storage.writes, 2);

  // A document that has not changed is not worth a write.
  autosave.request({ scene: pond({ randomSeed: 42 }), presetId: "garden-pond" });
  await delay(60);
  assert.equal(storage.writes, 2);
});

test("flush is what a closing tab gets", () => {
  const storage = memoryStorage();
  const autosave = createSceneAutosave({ storage, delay_ms: 10_000 });
  autosave.request({ scene: pond({ randomSeed: 5 }), presetId: "garden-pond" });
  autosave.flush();
  assert.equal(loadSceneFromLibrary(readSceneAutosave(readSceneLibrary(storage))!).randomSeed, 5);

  const cancelled = memoryStorage();
  const second = createSceneAutosave({ storage: cancelled, delay_ms: 10_000 });
  second.request({ scene: pond(), presetId: "garden-pond" });
  second.cancel();
  second.flush();
  assert.equal(cancelled.writes, 0);
});

test("the working document round-trips to a scene the product accepts", () => {
  const storage = memoryStorage();
  const scene = pond({ randomSeed: 11 });
  const entry = writeSceneAutosave(storage, { scene, presetId: "garden-pond" }, 1_000);
  assert.equal(entry.id, SCENE_AUTOSAVE_ENTRY_ID);
  assert.equal(entry.presetId, "garden-pond", "the origin is what restores the camera and the profile");
  assert.equal(entry.scene, serializeScene(scene), "an autosave is the same artifact as the file download");

  const loaded = loadSceneFromLibrary(readSceneAutosave(readSceneLibrary(storage))!);
  assert.deepEqual(validateScene(loaded), []);
  assert.equal(canonicalScene(loaded), canonicalScene(scene));

  // Re-saving keeps the one reserved entry rather than growing the library.
  writeSceneAutosave(storage, { scene: pond({ randomSeed: 12 }), presetId: "garden-pond" }, 2_000);
  assert.equal(readSceneLibrary(storage).length, 1);
});

test("an autosave is named after the scene it was opened from", () => {
  assert.equal(sceneAutosaveName("garden-pond"), getScenePreset("garden-pond").name);
  assert.equal(sceneAutosaveName("starter:empty-room"), SCENE_AUTOSAVE_NAME);
  assert.equal(sceneAutosaveName("a-scene-from-an-older-build"), SCENE_AUTOSAVE_NAME);
});

test("the autosave is the Continue affordance, not a card on the shelf", () => {
  const storage = memoryStorage();
  saveSceneToLibrary(storage, "Pond experiment", pond(), "garden-pond", { savedAt_ms: 1_000 });
  writeSceneAutosave(storage, { scene: pond({ randomSeed: 3 }), presetId: "garden-pond" }, 2_000);
  const entries = readSceneLibrary(storage);
  assert.equal(entries.length, 2);

  const mine = sceneSections(entries).find(({ id }) => id === "mine");
  const shelved = mine?.shelves.flatMap((shelf) => shelf.cards) ?? [];
  assert.deepEqual(shelved.map(({ name }) => name), ["Pond experiment"]);
  assert.equal(allSceneCards(entries).some(({ id }) => id === `saved:${SCENE_AUTOSAVE_ENTRY_ID}`), false);

  // A library holding nothing but the working document has no shelf at all.
  const onlyAutosave = memoryStorage();
  writeSceneAutosave(onlyAutosave, { scene: pond(), presetId: "garden-pond" }, 1_000);
  assert.equal(sceneSections(readSceneLibrary(onlyAutosave)).some(({ id }) => id === "mine"), false);
  assert.deepEqual(savedSceneEntries(readSceneLibrary(onlyAutosave)), []);
});

test("Continue resolves to where the reader was, not to their newest save", () => {
  const storage = memoryStorage();
  writeSceneAutosave(storage, { scene: pond({ randomSeed: 3 }), presetId: "garden-pond" }, 1_000);
  saveSceneToLibrary(storage, "Pond experiment", pond(), "garden-pond", { savedAt_ms: 9_000 });

  const resume = sceneResume(readSceneLibrary(storage));
  assert.equal(resume?.autosaved, true, "the save is newer; the working document is still where they were");
  assert.equal(resume?.entry.id, SCENE_AUTOSAVE_ENTRY_ID);

  const opening = resume!.card.open();
  assert.equal(opening.scene.randomSeed, 3);
  assert.equal(opening.presetId, "garden-pond");
  assert.ok(opening.camera, "reopening restores the framing of the scene it came from");
  assert.equal(sceneResume([]), undefined);
});

test("Continue restores a selected Power 2017 factor-4 method profile", () => {
  const storage = memoryStorage();
  writeSceneAutosave(storage, {
    scene: cloneScene(getScenePreset("symmetric-expansion").create()),
    presetId: "symmetric-expansion",
    methodProfile: POWER2017_FACTOR4_BENCHMARK_METHOD_PROFILE,
  }, 1_000);

  const opening = sceneResume(readSceneLibrary(storage))!.card.open();
  assert.deepEqual(opening.methodProfile, POWER2017_FACTOR4_BENCHMARK_METHOD_PROFILE);
  assert.equal(opening.methodProfile?.overrides.coarseBackend, "power2017");
  assert.equal(opening.methodProfile?.overrides.globalFineLevelSetFactor, "4");
});

test("an explicit save is never written over by the autosave", () => {
  const storage = memoryStorage();
  // The names collide on purpose: both are "Garden pond" from the same origin.
  writeSceneAutosave(storage, { scene: pond({ randomSeed: 3 }), presetId: "garden-pond" }, 1_000);
  const { entry: saved } = saveSceneToLibrary(
    storage, sceneAutosaveName("garden-pond"), pond({ randomSeed: 7 }), "garden-pond", { savedAt_ms: 2_000 });
  assert.notEqual(saved.id, SCENE_AUTOSAVE_ENTRY_ID, "saving by name must not adopt the reserved entry");

  writeSceneAutosave(storage, { scene: pond({ randomSeed: 5 }), presetId: "garden-pond" }, 3_000);
  const entries = readSceneLibrary(storage);
  assert.equal(entries.length, 2);
  assert.equal(loadSceneFromLibrary(entries.find(({ id }) => id === saved.id)!).randomSeed, 7,
    "the reader's save still holds the document they saved");
  assert.equal(loadSceneFromLibrary(readSceneAutosave(entries)!).randomSeed, 5);
});

test("a corrupt autosave loses Continue rather than the library", () => {
  const corrupt = {
    id: SCENE_AUTOSAVE_ENTRY_ID, name: "Untitled scene", savedAt_ms: 9_000,
    presetId: "garden-pond", scene: "{ not a scene",
  };
  const alone = memoryStorage(JSON.stringify([corrupt]));
  assert.equal(sceneResume(readSceneLibrary(alone)), undefined);
  assert.equal(sceneSections(readSceneLibrary(alone)).some(({ id }) => id === "mine"), false);

  // The grid still draws an unreadable entry; only opening it reports why.
  const withSave = memoryStorage();
  saveSceneToLibrary(withSave, "Pond experiment", pond({ randomSeed: 4 }), "garden-pond", { savedAt_ms: 1_000 });
  withSave.setItem(SCENE_LIBRARY_STORAGE_KEY,
    JSON.stringify([corrupt, ...readSceneLibrary(withSave)]));
  const resume = sceneResume(readSceneLibrary(withSave));
  assert.equal(resume?.autosaved, false, "their own newest save is a better answer than nothing");
  assert.equal(resume?.entry.name, "Pond experiment");
  assert.equal(sceneCardPreview({ ...resume!.card, id: "saved:corrupt-fixture", savedAt_ms: 9_001 })!.randomSeed, 4);
});

test("nothing is autosaved until a scene has actually been opened", async () => {
  const storage = memoryStorage();
  useShellStore.setState({ view: "library", studioEntered: false });
  useMethodStore.setState(useMethodStore.getInitialState());
  const stop = startSceneAutosave({ storage, delay_ms: 5 });

  useSceneStore.getState().setScene(pond({ randomSeed: 1 }), "garden-pond");
  await delay(40);
  assert.equal(storage.writes, 0, "the default document is not the reader's work");

  // Entering the studio is itself the trigger: a scene opened and left alone is
  // still where the reader is.
  useShellStore.getState().enterStudio();
  await delay(40);
  assert.equal(storage.writes, 1);
  assert.equal(loadSceneFromLibrary(readSceneAutosave(readSceneLibrary(storage))!).randomSeed, 1);

  useMethodStore.getState().setParam("octree", "coarseBackend", "power2017");
  await delay(40);
  assert.equal(
    readSceneAutosave(readSceneLibrary(storage))?.methodProfile?.overrides.coarseBackend,
    "power2017",
    "method selection itself must update the resumable working document",
  );
  const resumedProfile = sceneResume(readSceneLibrary(storage))!.card.open().methodProfile!;
  const resumedValues = resolveMethodValues(getMethod(resumedProfile.methodId),
    resumedProfile.quality, resumedProfile.overrides);
  assert.equal(resumedValues.globalFineLevelSetFactor, "4");

  useSceneStore.getState().setScene(pond({ randomSeed: 2 }), "garden-pond");
  await delay(40);
  assert.equal(loadSceneFromLibrary(readSceneAutosave(readSceneLibrary(storage))!).randomSeed, 2);

  stop();
  useSceneStore.getState().setScene(pond({ randomSeed: 3 }), "garden-pond");
  await delay(40);
  assert.equal(loadSceneFromLibrary(readSceneAutosave(readSceneLibrary(storage))!).randomSeed, 2,
    "a stopped autosave writes nothing further");
});
