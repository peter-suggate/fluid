import assert from "node:assert/strict";
import test from "node:test";
import { cloneScene, defaultScene } from "../lib/core/model";
import { sceneCellSizes_m } from "../lib/core/scene-lattice";
import { liveFluidEditDistance } from "../lib/core/live-fluid-edit";
import { toolValues } from "../lib/core/voxel-editor/plugin";
import { voxelTools } from "../lib/core/voxel-editor/registry";
import { fluidShapeBounds, fluidShapeHighlight, fluidShapeVolume } from "../lib/core/voxel-editor/fluid-geometry";

const plugins = voxelTools.tools.filter(plugin => plugin.execution === "release");
const ray = (x = 0, z = 0) => ({ origin: { x, y: 10, z }, direction: { x: 0, y: -1, z: 0 } });
const scene = () => {
  const result = cloneScene(defaultScene);
  result.terrain = undefined;
  result.container = { ...result.container, width_m: 4, height_m: 4, depth_m: 4 };
  return result;
};

test("fluid plugins own their executable declarations and previews propose no solid/document edits", () => {
  assert.equal(plugins.length, 3);
  for (const plugin of plugins) {
    const base = scene();
    const before = JSON.stringify(base);
    const values = toolValues(plugin);
    const gesture = plugin.begin({ scene: base, ray: ray(), values })!;
    const result = gesture.update(ray())!;
    assert.equal(result.action?.kind, "fluid", plugin.id);
    assert.equal(result.action.edit.operation, "add");
    assert.deepEqual(result.patches, []);
    assert.equal(JSON.stringify(base), before);
    assert.ok(plugin.ui.controls.some(c => c.presentation === "primary"));
    assert.equal(plugin.unavailable({ scene: base, methodId: "adaptive-volume" }), undefined);
    assert.ok(plugin.unavailable({ scene: { ...base, systems: { fluid: false } }, methodId: "adaptive-volume" }));
    assert.ok(plugin.unavailable({ scene: base, methodId: "other" }));
    const remove = plugin.begin({ scene: base, ray: ray(), values: { ...values, remove: 1 } })!.update(ray())!;
    assert.equal(remove.action!.edit.operation, "remove");
    assert.deepEqual(remove.highlight, result.highlight);
  }
});

test("drop height measures the shape bottom and stamps follow a fixed plane without accumulating", () => {
  for (const plugin of plugins) {
    const base = scene();
    const values = toolValues(plugin, { height: 3 });
    const voxel = Math.min(...sceneCellSizes_m(base));
    const gesture = plugin.begin({ scene: base, ray: ray(), values })!;
    const initial = gesture.update(ray())!.action!.edit;
    const moved = gesture.update(ray(.25, .25))!.action!.edit;
    assert.equal(initial.center_m.y, moved.center_m.y);
    assert.notEqual(initial.center_m.x, moved.center_m.x);
    assert.ok(Math.abs(fluidShapeBounds(moved).min.y - 3 * voxel) < 1e-10);
    assert.deepEqual(gesture.update(ray())!.action!.edit, initial);
    assert.throws(() => gesture.update(ray(100)), /inside the tank/);
  }
});

test("torus constraints preserve a resolved hole and preview paths lie on exact emitted surfaces", () => {
  const base = scene();
  for (const plugin of plugins) {
    const result = plugin.begin({ scene: base, ray: ray(), values: toolValues(plugin) })!.update(ray())!;
    const edit = result.action!.edit;
    const highlight = fluidShapeHighlight(edit);
    if (highlight.kind === "paths") {
      assert.ok(highlight.paths.length >= 3);
      for (const path of highlight.paths) {
        for (const point of path) assert.ok(Math.abs(liveFluidEditDistance(edit, point)) < 1e-9, `${plugin.id}: surface path`);
        assert.ok(Math.hypot(path[0]!.x - path.at(-1)!.x, path[0]!.y - path.at(-1)!.y, path[0]!.z - path.at(-1)!.z) < 1e-9);
      }
    } else assert.equal(highlight.kind, "box");
    const volume = fluidShapeVolume(edit);
    if (edit.shape === "torus") {
      assert.equal(volume.shape, "torus");
      assert.ok(liveFluidEditDistance(edit, edit.center_m) > 0);
      assert.equal(edit.tubeRadius_m, Math.min(...sceneCellSizes_m(base)));
      assert.ok(edit.tubeRadius_m! < edit.radius_m / 2);
      assert.throws(() => plugin.begin({ scene: base, ray: ray(), values: toolValues(plugin, { size: 4, thickness: 2 }) }), /keep the ring open/);
    }
  }
});

test("scene-aware defaults float drops above a pool, match execution, and respect explicit user heights", () => {
  const base = scene();
  base.container.fillFraction = 1 / 3;
  const voxel = Math.min(...sceneCellSizes_m(base));
  for (const plugin of plugins) {
    const values = toolValues(plugin, {}, base);
    const update = plugin.begin({ scene: base, ray: ray(), values })!.update(ray())!;
    const bounds = fluidShapeBounds(update.action!.edit);
    assert.ok(Math.abs(bounds.min.y - values.height * voxel) < 1e-9);
    assert.ok(bounds.min.y >= base.container.fillFraction * base.container.height_m + voxel);
    assert.ok(bounds.max.y <= base.container.height_m + 1e-9);
    assert.equal(toolValues(plugin, { height: 1 }, base).height, 1);
    const enlarged = toolValues(plugin, { size: 24, thickness: 10 }, base);
    const result = plugin.begin({ scene: base, ray: ray(), values: enlarged })!.update(ray())!;
    assert.ok(fluidShapeBounds(result.action!.edit).max.y <= base.container.height_m + 1e-9);
    assert.equal(toolValues(plugin, {}, { ...base, container: { ...base.container, fillFraction: 0 } }).height, 2);
  }
});

test("a single quick click in a newly enabled empty room dispatches one fluid command", async () => {
  const { createEmptyScene } = await import("../lib/core/empty-scene");
  const { defaultCamera } = await import("../lib/core/model");
  const { viewportRayForPointer } = await import("../lib/core/webgpu-camera");
  const { beginToolTransaction } = await import("../lib/core/voxel-editor/transaction");
  const base = createEmptyScene();
  base.systems = { ...base.systems, fluid: true };
  const plugin = voxelTools.get("fluid-ball")!;
  const click = viewportRayForPointer(defaultCamera, 730, 460, { left: 0, top: 0, width: 1571, height: 850 });
  const commands: unknown[] = [];
  let cleared = false;
  const transaction = beginToolTransaction(plugin, {
    scene: () => base,
    publish: async () => { assert.fail("a transient water drop cannot author the scene"); },
    execute: async action => { commands.push(action); },
    begin: () => {}, finish: () => assert.fail("water motion cannot enter scene history"), cancel: () => { cleared = true; },
  }, click, { size: 6, height: 8 })!;
  const sample = transaction.update(click);
  const release = transaction.finish();
  await Promise.all([sample, release]);
  assert.equal(commands.length, 1);
  assert.equal(cleared, true);
  assert.deepEqual(commands[0], { kind: "fluid", edit: { operation: "add", shape: "ball",
    center_m: { x: .30000000000000004, y: .55, z: .5 }, radius_m: .15000000000000002 } });
});
