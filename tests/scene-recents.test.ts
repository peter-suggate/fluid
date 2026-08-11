import assert from "node:assert/strict";
import test from "node:test";
import { allSceneCards } from "../lib/scene-cards";
import type { SceneLibraryStorage } from "../lib/scene-library";
import {
  readSceneRecents,
  recentSceneCards,
  recordSceneOpen,
  SCENE_RECENTS_LIMIT,
  SCENE_RECENTS_STORAGE_KEY,
} from "../lib/scene-recents";

function memoryStorage(seed: Record<string, string> = {}): SceneLibraryStorage & { data: Record<string, string> } {
  const data = { ...seed };
  return {
    data,
    getItem: (key) => data[key] ?? null,
    setItem: (key, value) => { data[key] = value; },
  };
}

test("an open moves to the front rather than duplicating", () => {
  const storage = memoryStorage();
  recordSceneOpen(storage, "a", 1);
  recordSceneOpen(storage, "b", 2);
  const entries = recordSceneOpen(storage, "a", 3);
  assert.deepEqual(entries.map(({ cardId }) => cardId), ["a", "b"],
    "the row is a working set, not a click log");
  assert.deepEqual(readSceneRecents(storage).map(({ cardId }) => cardId), ["a", "b"]);
});

test("the list is bounded", () => {
  const storage = memoryStorage();
  for (let index = 0; index < SCENE_RECENTS_LIMIT + 5; index += 1) {
    recordSceneOpen(storage, `scene-${index}`, index);
  }
  const entries = readSceneRecents(storage);
  assert.equal(entries.length, SCENE_RECENTS_LIMIT);
  assert.equal(entries[0].cardId, `scene-${SCENE_RECENTS_LIMIT + 4}`, "newest first");
});

test("malformed storage reads as empty rather than throwing into the UI", () => {
  assert.deepEqual(readSceneRecents(undefined), []);
  assert.deepEqual(readSceneRecents(memoryStorage({ [SCENE_RECENTS_STORAGE_KEY]: "{ not json" })), []);
  assert.deepEqual(readSceneRecents(memoryStorage({ [SCENE_RECENTS_STORAGE_KEY]: '{"cardId":"a"}' })), []);
  assert.deepEqual(
    readSceneRecents(memoryStorage({ [SCENE_RECENTS_STORAGE_KEY]: '[{"cardId":"a"},{"cardId":"b","openedAt_ms":4}]' }))
      .map(({ cardId }) => cardId),
    ["b"],
    "entries are validated one by one",
  );
});

test("recents resolve against the current card set and stale ids drop out", () => {
  const cards = allSceneCards([]);
  const known = cards[0];
  const alsoKnown = cards[1];
  const row = recentSceneCards(
    [
      { cardId: known.id, openedAt_ms: 3 },
      { cardId: "saved:deleted-long-ago", openedAt_ms: 2 },
      { cardId: alsoKnown.id, openedAt_ms: 1 },
    ],
    cards,
    6,
  );
  assert.deepEqual(row.map(({ id }) => id), [known.id, alsoKnown.id],
    "an id this build cannot open must not become an unopenable card");
});

test("the row is cut to its limit after resolution, not before", () => {
  const cards = allSceneCards([]);
  const recents = [
    { cardId: "gone-1", openedAt_ms: 9 },
    ...cards.slice(0, 4).map((card, index) => ({ cardId: card.id, openedAt_ms: 8 - index })),
  ];
  const row = recentSceneCards(recents, cards, 3);
  assert.deepEqual(row.map(({ id }) => id), cards.slice(0, 3).map(({ id }) => id),
    "a stale id must not thin the row below its limit");
});
