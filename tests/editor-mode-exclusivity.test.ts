import "../lib/methods";
import assert from "node:assert/strict";
import test from "node:test";
import { createPaneSession } from "../lib/core/session/session";
import { voxelTools } from "../lib/core/voxel-editor/registry";

/**
 * One thing in the hand at a time, stated once in the store. The regressions:
 * putting a tool away flipped LOOK back to INTERACT, and LOOK left the sculpt
 * tool armed underneath it while it put the gesture down.
 */
test("a tool and a gesture exclude each other; the camera mode puts both down; disarming is not a mode change", () => {
  const ui = createPaneSession("a").ui;
  const tool = voxelTools.get("pen")?.id;
  assert.ok(tool);
  ui.getState().setVoxelTool(tool);
  assert.equal(ui.getState().viewportMode, "interact");
  ui.getState().setArmedGesture("fluid-ball");
  assert.equal(ui.getState().voxelToolId, undefined);
  ui.getState().setVoxelTool(tool);
  assert.equal(ui.getState().armedGesture, undefined);
  ui.getState().setViewportMode("camera");
  assert.equal(ui.getState().voxelToolId, undefined);
  assert.equal(ui.getState().armedGesture, undefined);
  ui.getState().setVoxelTool(undefined);
  assert.equal(ui.getState().viewportMode, "camera");
});
