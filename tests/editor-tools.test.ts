import assert from "node:assert/strict";
import test from "node:test";
import {
  bodySelection,
  CLICK_SLOP_PX,
  DEFAULT_EDITOR_TOOL,
  EDITOR_TOOLS,
  editorToolForShortcut,
  editorToolIsActive,
  emptySpaceClickDeselects,
  getEditorTool,
  selectedBodyIdOf,
} from "../lib/editor-tools";
import { useUIStore } from "../lib/stores/ui-store";

test("editor tool shortcuts are unique and resolve case-insensitively", () => {
  const shortcuts = EDITOR_TOOLS.map((tool) => tool.shortcut);
  assert.equal(new Set(shortcuts).size, shortcuts.length, "two tools cannot share one key");
  for (const tool of EDITOR_TOOLS) {
    assert.equal(editorToolForShortcut(tool.shortcut), tool.id);
    assert.equal(editorToolForShortcut(tool.shortcut.toUpperCase()), tool.id);
  }
  assert.equal(editorToolForShortcut("]"), undefined);
});

test("select is the default tool and is always implemented", () => {
  assert.equal(DEFAULT_EDITOR_TOOL, "select");
  assert.equal(editorToolIsActive("select"), true);
  assert.equal(useUIStore.getInitialState().activeTool, "select");
});

test("the viewport opens on an empty selection so the first drag is the camera", () => {
  const initial = useUIStore.getInitialState();
  assert.equal(initial.selection, undefined);
  assert.equal(initial.selectedBodyId, undefined);
});

test("a background click deselects, a camera drag keeps the selection", () => {
  // A click never travels far; anything beyond the slop is a camera gesture
  // and must not throw away what the user is holding.
  assert.equal(emptySpaceClickDeselects("orbit", 0, 0), true);
  assert.equal(emptySpaceClickDeselects("orbit", CLICK_SLOP_PX, 0), true);
  assert.equal(emptySpaceClickDeselects("orbit", -CLICK_SLOP_PX, 0), true);
  assert.equal(emptySpaceClickDeselects("orbit", CLICK_SLOP_PX + 1, 0), false);
  assert.equal(emptySpaceClickDeselects("orbit", 0, CLICK_SLOP_PX + 1), false);
  // Diagonal travel is measured as distance, not per-axis.
  assert.equal(emptySpaceClickDeselects("orbit", CLICK_SLOP_PX, CLICK_SLOP_PX), false);

  // Pan is asked for explicitly with shift or the middle button, so it is
  // navigation rather than a pick and never touches the selection.
  assert.equal(emptySpaceClickDeselects("pan", 0, 0), false);
});

test("the select tool tells the user how to let go of a selection", () => {
  assert.match(getEditorTool("select").hint, /deselect/);
});

test("unimplemented tools declare the plan phase that lands them", () => {
  for (const tool of EDITOR_TOOLS) {
    if (tool.status === "active") assert.equal(tool.phase, undefined);
    else assert.ok(tool.phase, `${tool.id} must name the phase that implements it`);
  }
  assert.throws(() => getEditorTool("not-a-tool" as never), /Unknown editor tool/);
});

test("selection generalizes the body id without losing the body projection", () => {
  const initial = useUIStore.getInitialState();
  useUIStore.setState(initial, true);

  useUIStore.getState().selectBody("ball-1");
  assert.deepEqual(useUIStore.getState().selection, { kind: "body", id: "ball-1" });
  assert.equal(useUIStore.getState().selectedBodyId, "ball-1");

  // A non-body selection must clear the body projection the renderer reads,
  // otherwise a terrain feature would keep highlighting a rigid body.
  useUIStore.getState().select({ kind: "terrain-feature", id: "pond" });
  assert.deepEqual(useUIStore.getState().selection, { kind: "terrain-feature", id: "pond" });
  assert.equal(useUIStore.getState().selectedBodyId, undefined);

  useUIStore.getState().selectBody(undefined);
  assert.equal(useUIStore.getState().selection, undefined);
  assert.equal(useUIStore.getState().selectedBodyId, undefined);

  assert.deepEqual(bodySelection("a"), { kind: "body", id: "a" });
  assert.equal(bodySelection(undefined), undefined);
  assert.equal(selectedBodyIdOf({ kind: "inflow", id: "hose" }), undefined);

  useUIStore.setState(initial, true);
});
