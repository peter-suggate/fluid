import assert from "node:assert/strict";
import test from "node:test";
import { cloneScene, defaultScene, type SceneDescription } from "../lib/model";
import {
  EDITOR_HISTORY_COALESCE_MS,
  EDITOR_HISTORY_LIMIT,
  useEditorHistoryStore,
} from "../lib/stores/history-store";

function sceneWithSeed(randomSeed: number): SceneDescription {
  return { ...cloneScene(defaultScene), randomSeed };
}

function snapshot(randomSeed: number, label = `edit-${randomSeed}`) {
  return { label, scene: sceneWithSeed(randomSeed), presetId: "water-box-dam-break" };
}

function reset() {
  useEditorHistoryStore.setState(useEditorHistoryStore.getInitialState(), true);
}

test("undo and redo walk the document back and forward", () => {
  reset();
  const history = useEditorHistoryStore.getState();
  history.record(snapshot(1, "first"));
  history.record(snapshot(2, "second"));

  const undone = useEditorHistoryStore.getState().undo(snapshot(3));
  assert.equal(undone?.scene.randomSeed, 2, "undo returns the document before the last edit");
  const undoneAgain = useEditorHistoryStore.getState().undo(snapshot(2));
  assert.equal(undoneAgain?.scene.randomSeed, 1);
  assert.equal(useEditorHistoryStore.getState().undo(snapshot(1)), undefined, "an empty stack yields nothing");

  assert.equal(useEditorHistoryStore.getState().redo(snapshot(1))?.scene.randomSeed, 2);
  assert.equal(useEditorHistoryStore.getState().redo(snapshot(2))?.scene.randomSeed, 3);
  assert.equal(useEditorHistoryStore.getState().redo(snapshot(3)), undefined);
  reset();
});

test("a new edit discards the redo branch", () => {
  reset();
  useEditorHistoryStore.getState().record(snapshot(1));
  useEditorHistoryStore.getState().undo(snapshot(2));
  assert.equal(useEditorHistoryStore.getState().future.length, 1);
  useEditorHistoryStore.getState().record(snapshot(9));
  assert.equal(useEditorHistoryStore.getState().future.length, 0);
  reset();
});

test("one gesture coalesces into one entry, a different field does not", () => {
  reset();
  const history = useEditorHistoryStore.getState();
  history.record(snapshot(1), { coalesceKey: "body:ball:density", now_ms: 1_000 });
  history.record(snapshot(2), { coalesceKey: "body:ball:density", now_ms: 1_100 });
  history.record(snapshot(3), { coalesceKey: "body:ball:density", now_ms: 1_200 });
  assert.equal(useEditorHistoryStore.getState().past.length, 1, "a swept slider is one undo");
  assert.equal(useEditorHistoryStore.getState().past[0]?.scene.randomSeed, 1, "the pre-gesture document is kept");

  history.record(snapshot(4), { coalesceKey: "body:ball:friction", now_ms: 1_250 });
  assert.equal(useEditorHistoryStore.getState().past.length, 2, "a different field starts a new entry");

  history.record(snapshot(5), { coalesceKey: "body:ball:friction", now_ms: 1_250 + EDITOR_HISTORY_COALESCE_MS + 1 });
  assert.equal(useEditorHistoryStore.getState().past.length, 3, "a paused gesture is a separate edit");

  history.record(snapshot(6), { now_ms: 1_260 + EDITOR_HISTORY_COALESCE_MS });
  history.record(snapshot(7), { now_ms: 1_261 + EDITOR_HISTORY_COALESCE_MS });
  assert.equal(useEditorHistoryStore.getState().past.length, 5, "keyless edits never coalesce");
  reset();
});

test("history is bounded and never aliases live store state", () => {
  reset();
  for (let index = 0; index < EDITOR_HISTORY_LIMIT + 12; index += 1) {
    useEditorHistoryStore.getState().record(snapshot(index));
  }
  const past = useEditorHistoryStore.getState().past;
  assert.equal(past.length, EDITOR_HISTORY_LIMIT);
  assert.equal(past[0]?.scene.randomSeed, 12, "the oldest entries are dropped, not the newest");

  const live = snapshot(99);
  useEditorHistoryStore.getState().record(live);
  live.scene.randomSeed = 1234;
  live.scene.container.width_m = 99;
  const stored = useEditorHistoryStore.getState().past.at(-1);
  assert.equal(stored?.scene.randomSeed, 99, "snapshots are deep-cloned on record");
  assert.notEqual(stored?.scene.container.width_m, 99);
  reset();
});
