import assert from "node:assert/strict";
import test from "node:test";
import { canonicalScene, serializeScene, validateScene } from "../lib/model";
import {
  allSceneCards,
  matchSceneCards,
  sceneCardPreview,
  sceneSections,
  savedSceneCards,
  starterSceneCards,
} from "../lib/scene-cards";
import { sceneDefinitionCamera } from "../lib/scene-definition";
import type { SceneLibraryEntry } from "../lib/scene-library";
import { getSceneDefinition, getScenePreset } from "../lib/scenes";

function entry(patch: Partial<SceneLibraryEntry> = {}): SceneLibraryEntry {
  return {
    id: "pond-experiment",
    name: "Pond experiment",
    savedAt_ms: 1_000,
    presetId: "garden-pond",
    scene: serializeScene(getScenePreset("garden-pond").create()),
    ...patch,
  };
}

test("the sections read in the order the page is read", () => {
  const sections = sceneSections([]);
  assert.deepEqual(sections.map(({ id }) => id), ["explore", "study", "validation"]);
  // The oracles are how the physics is trusted; they stay, disclosed.
  assert.equal(sections.find(({ id }) => id === "validation")?.disclosed, true);
  for (const section of sections.filter(({ id }) => id !== "validation")) {
    assert.equal(section.disclosed, false, `${section.id} is offered, not disclosed`);
  }
});

test("starters are a hero, not a shelf, and are still cards", () => {
  // Creating is a peer of choosing, so the room sizes sit beside *Continue*
  // rather than in a section below the scenes. That must not cost them their
  // card identity: they are still openable and still findable.
  // `SceneSectionId` no longer admits a starters section at all, so the type is
  // the guard; what needs asserting is that the cards survived losing it.
  const cards = allSceneCards([]);
  for (const starter of starterSceneCards()) {
    assert.ok(cards.some(({ id }) => id === starter.id), `${starter.id} must remain in the card set`);
    assert.ok(matchSceneCards(cards, starter.name).some(({ id }) => id === starter.id),
      `${starter.id} must stay reachable by search`);
  }
});

test("an empty shelf is dropped rather than rendered as a reproach", () => {
  assert.equal(sceneSections([]).some(({ id }) => id === "mine"), false);
  const sections = sceneSections([entry()]);
  assert.deepEqual(sections.map(({ id }) => id), ["explore", "study", "mine", "validation"],
    "someone's own scenes sit above the oracles, not below them");
});

test("every section holds at least one card", () => {
  for (const section of sceneSections([entry()])) {
    const count = section.shelves.reduce((total, shelf) => total + shelf.cards.length, 0);
    assert.ok(count > 0, `${section.id} must not be an empty heading`);
    for (const shelf of section.shelves) assert.ok(shelf.cards.length > 0);
  }
});

test("saved scenes are newest first", () => {
  const cards = savedSceneCards([
    entry({ id: "older", name: "Older", savedAt_ms: 10 }),
    entry({ id: "newest", name: "Newest", savedAt_ms: 900 }),
    entry({ id: "middle", name: "Middle", savedAt_ms: 500 }),
  ]);
  assert.deepEqual(cards.map(({ name }) => name), ["Newest", "Middle", "Older"]);
  assert.deepEqual(cards.map(({ savedAt_ms }) => savedAt_ms), [900, 500, 10]);
});

test("a saved scene reopens with the framing and the profile it was authored under", () => {
  // Loading a saved document used to apply neither, so a saved validation scene
  // ran under whatever method happened to be selected.
  const validation = getSceneDefinition("minimal-power-dam-break");
  const [card] = savedSceneCards([entry({
    presetId: validation.id,
    scene: serializeScene(getScenePreset(validation.id).create()),
  })]);
  const opening = card.open();
  assert.deepEqual(opening.methodProfile, validation.methodProfile);
  assert.deepEqual(opening.camera, sceneDefinitionCamera(validation));
  assert.equal(opening.presetId, validation.id);
});

test("a saved scene from a scene this build no longer has keeps the camera it has", () => {
  // Falling back to the first catalog entry would frame it with an unrelated
  // scene's camera and pin an unrelated scene's solver profile.
  const [card] = savedSceneCards([entry({ presetId: "a-scene-from-an-older-build" })]);
  const opening = card.open();
  assert.equal(opening.camera, undefined);
  assert.equal(opening.methodProfile, undefined);
  assert.equal(opening.presetId, "a-scene-from-an-older-build", "identity is preserved, not rewritten");
  assert.match(card.blurb, /saved in this browser/i);
});

test("starters create an empty document rather than loading one", () => {
  const cards = starterSceneCards();
  assert.ok(cards.length > 0);
  for (const card of cards) {
    assert.equal(card.source, "starter");
    const opening = card.open();
    assert.deepEqual(validateScene(opening.scene), []);
    assert.equal(opening.scene.systems?.fluid, false, "a fresh scene asks for no solver");
    assert.equal(opening.scene.rigidBodies.length, 0);
    assert.equal(opening.presetId, card.id, "a starter's identity is its own, not a preset's");
  }
});

test("card identities never collide across sources", () => {
  const ids = allSceneCards([entry(), entry({ id: "second", name: "Second" })]).map(({ id }) => id);
  assert.equal(new Set(ids).size, ids.length);
});

test("search reaches every shelf, including the disclosed one", () => {
  const cards = allSceneCards([entry()]);
  assert.equal(matchSceneCards(cards, "   ").length, cards.length, "an empty query is not a filter");
  assert.ok(matchSceneCards(cards, "ceiling drop").some(({ id }) => id === "ceiling-slab-drop"),
    "an oracle must be findable by name even while its section is collapsed");
  // Matching the blurb is what makes a search for a thing you remember seeing
  // work without a tag vocabulary nobody would maintain.
  assert.ok(matchSceneCards(cards, "mushrooms").some(({ id }) => id === "garden-pond"));
  assert.ok(matchSceneCards(cards, "Garden").length > 1, "a shelf name is a query too");
  assert.deepEqual(matchSceneCards(cards, "zzzz-no-such-scene"), []);
});

test("a preview is built once per identity and follows a re-save", () => {
  const [card] = savedSceneCards([entry()]);
  const first = sceneCardPreview(card);
  assert.ok(first);
  assert.equal(sceneCardPreview(card), first, "the same identity must not rebuild its document");

  // A re-save keeps the entry id, so the save time is what tells the preview
  // that the document under it changed.
  const resaved = savedSceneCards([entry({
    savedAt_ms: 2_000,
    scene: serializeScene(getScenePreset("ocean-seiche").create()),
  })])[0];
  const second = sceneCardPreview(resaved);
  assert.ok(second);
  assert.notEqual(canonicalScene(second), canonicalScene(first));
});

test("an unreadable saved scene lists without taking the library down", () => {
  const [card] = savedSceneCards([entry({ id: "corrupt", scene: "{ not a scene" })]);
  assert.equal(sceneCardPreview(card), undefined, "the grid still draws; the failure surfaces on open");
  assert.throws(() => card.open(), "opening it is where the reader learns why");
});
