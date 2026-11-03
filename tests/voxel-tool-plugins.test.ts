import assert from "node:assert/strict";
import test from "node:test";
import { interpolateCells, mirrorPatchX, shapePatches } from "../lib/core/voxel-editor/geometry";
import { EDITOR_GESTURES } from "../lib/core/editor-gesture-catalog";
import { SCENE_INSTRUMENT_ORDER } from "../lib/core/scene-instruments";
import { EDITOR_AXES } from "../lib/core/editor-axis-constraint";
import { CAMERA_FRAMINGS } from "../lib/core/editor-camera-framing";
import { voxelRailKeyForTool, voxelRailToolForKey, voxelToolHint, voxelToolKeyControls, voxelToolLineControls,
  voxelToolRail } from "../lib/core/editor-voxel-tool-actions";
import { RESERVED_EDITOR_KEYS, createVoxelToolRegistry, toolValues, type VoxelToolPlugin } from "../lib/core/voxel-editor/plugin";
import { materialControl, mirrorControls, operationControl } from "../lib/core/voxel-editor/controls";
import { voxelTools } from "../lib/core/voxel-editor/registry";
import { createEmptyScene } from "../lib/core/empty-scene";


test("fast strokes cover negative coordinates continuously and mirroring is involutive", () => {
  const points = interpolateCells([-9, 0, 3], [10, 0, 3]);
  assert.equal(points.length, 20);
  assert.deepEqual(points.map((p) => p[0]), Array.from({ length: 20 }, (_, i) => i - 9));
  assert.throws(() => interpolateCells([0, 0, 0], [10000, 0, 0]));
  const patch = shapePatches([-3, 0, 0], [-1, 1, 1], "fill", "box")[0]!;
  assert.deepEqual(mirrorPatchX(mirrorPatchX(patch, 32), 32), patch);
});

const tool = (id: string, ui: Partial<VoxelToolPlugin["ui"]> = {}): VoxelToolPlugin => ({
  id, version: 1, unavailable: () => undefined, begin: () => undefined,
  ui: { label: id, hint: id, group: "Brush", order: 0, icon: "M0 0", controls: [], ...ui },
});

test("the reserved keys cover every letter the editor itself answers to while editing", () => {
  const editor = [...EDITOR_GESTURES.flatMap((gesture) => gesture.shortcut ? [gesture.shortcut] : []),
    ...SCENE_INSTRUMENT_ORDER.map((instrument) => instrument.shortcut), ...EDITOR_AXES, "o", "r", "f"];
  for (const key of editor) assert.ok(RESERVED_EDITOR_KEYS.includes(key), `"${key}" must be reserved`);
  assert.ok(CAMERA_FRAMINGS.every((framing) => !/^[a-z]$/.test(framing.key)), "framings stay on digits");
});

test("the registry refuses a tool key that is reserved, taken by a tool, or driven by any control", () => {
  assert.doesNotThrow(() => createVoxelToolRegistry([tool("a", { key: "p" }), tool("b", { key: "u" })]));
  assert.throws(() => createVoxelToolRegistry([tool("a", { key: "p" }), tool("b", { key: "p" })]), /cannot take the key/);
  assert.throws(() => createVoxelToolRegistry([tool("a", { key: "r" })]), /reserved/);
  assert.throws(() => createVoxelToolRegistry([tool("a", { key: "P" })]), /malformed/);
  // A letter that drives a control under one tool would shadow the arming letter of another.
  assert.throws(() => createVoxelToolRegistry([tool("a", { key: "m" }), tool("b", { controls: [...mirrorControls] })]),
    /also drives a control/);
  assert.throws(() => createVoxelToolRegistry([tool("a", { hints: { armed: { keys: ["control:nope"] } } })]), /does not have/);
  assert.throws(() => createVoxelToolRegistry([tool("a", { hints: { later: { text: "x" } } as never })]), /Invalid hint/);
  assert.throws(() => createVoxelToolRegistry([tool("a", { controls: [{ ...operationControl, tier: "advanced" }] })]));
});

test("rail tools arm on their letters and their digits; a letter is never a control's key", () => {
  const rail = voxelToolRail(createEmptyScene(), "uniform-volume").flatMap((group) => group.tools);
  assert.equal(voxelRailToolForKey("p"), "pen");
  assert.equal(voxelRailToolForKey("P"), "pen", "caps lock still arms");
  assert.equal(voxelRailKeyForTool("pen"), "p");
  assert.equal(rail.find((entry) => entry.id === "pen")?.key, "P");
  rail.forEach((entry, index) => { if (index < 9) assert.equal(voxelRailToolForKey(String(index + 1)), entry.id); });
  for (const plugin of voxelTools.tools) {
    const letter = plugin.ui.key;
    if (!letter) continue;
    for (const armed of voxelTools.tools) {
      const keys = voxelToolKeyControls(armed.id).flatMap((control) => control.kind === "choice"
        ? control.options.map((option) => option.shortcut) : [control.shortcut]);
      assert.ok(!keys.includes(letter), `${plugin.id}'s "${letter}" is a control key under ${armed.id}`);
    }
  }
});

test("the pen shows operation, diameter and material; mirrors wait behind more unless one is on", () => {
  const pen = voxelTools.get("pen")!;
  const collapsed = voxelToolLineControls(pen, toolValues(pen), false);
  assert.deepEqual(collapsed.shown.map((control) => control.id), ["size", "material"],
    "the operation is the rail's switch, drawn there");
  assert.equal(collapsed.advanced, 3);
  assert.deepEqual(voxelToolLineControls(pen, toolValues(pen), true).shown.map((control) => control.id),
    ["size", "material", "mirror", "mirrorY", "mirrorZ"]);
  const mirrored = voxelToolLineControls(pen, toolValues(pen, { mirrorY: 1 }), false);
  assert.deepEqual(mirrored.shown.map((control) => control.id), ["size", "material", "mirrorY"], "a mirror that is on stays in view");
  const erasing = voxelToolLineControls(pen, toolValues(pen, { operation: 1 }), false);
  assert.ok(!erasing.shown.includes(materialControl), "erasing writes no material");
});

test("hints teach the next thing by phase, naming only keys the tool has", () => {
  const pen = voxelTools.get("pen")!;
  const values = (operation: number) => toolValues(pen, { operation });
  assert.equal(voxelToolHint("pen", "armed", values(0)), "T R G attach · erase · paint · [ ] diameter · drag across any surface");
  assert.equal(voxelToolHint("pen", "after", values(0)), "[ ] diameter · Alt inverts · pass again to build higher");
  assert.equal(voxelToolHint("pen", "after", values(1)), "[ ] diameter · Alt inverts · pass again to carve deeper");
  assert.match(voxelToolHint("pen", "after", values(2))!, /Alt-click picks a colour/);
  assert.equal(voxelToolHint("pen", "active", values(0)), "X Y Z lock an axis · ⇧ a plane · esc cancels · leave a spot and come back to go further");
  for (const plugin of voxelTools.tools) for (const phase of ["armed", "active", "after"] as const) {
    const hint = voxelToolHint(plugin.id, phase, toolValues(plugin));
    if (hint) assert.ok(hint.length <= 90, `${plugin.id} ${phase} hint stays short: ${hint}`);
  }
});
