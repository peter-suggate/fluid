import "../lib/methods";
import assert from "node:assert/strict";
import test from "node:test";
import { defaultScene, parseScene, serializeScene } from "../lib/core/model";
import { isEditableOak, oakTreeControlGroups, withOakParameters, withOakMaterial, withTreePreviewDepth, OAK_MATERIALS } from "../lib/core/oak-tree-controls";
import { createSceneryNodeAt, sceneryEntity, scenerySelectionId } from "../lib/core/editor-scenery";
import { findSceneryNode, withSceneryPlacement } from "../lib/core/scenery-edit";
import { validateSceneryGraph, type SceneryNode, type SceneryRecursiveShapeNode, type SceneryTaperedSweepClusterNode } from "../lib/core/scenery-graph";
import { OAK_V2_CONTROLS, OAK_V2_DEFAULTS, OAK_V2_PRESETS, oakPrimitiveCount, type OakV2Parameters } from "../lib/core/voxel-scenery/oak-v2-parameters";
import { oakFoliageDescriptor, planOakV2 } from "../lib/core/voxel-scenery/oak-v2";
import { sampleSvoPrimitive, validateSvoClusterPacking } from "../lib/svo/contracts/svo-primitive-abi";
import { createEditorHistoryStore } from "../lib/core/stores/history-store";
import { sceneEditRequiresReset } from "../lib/core/webgpu-renderer";

const spec = { key: "oak", seed: 4258, bark: OAK_MATERIALS[0].bark, foliage: OAK_MATERIALS[0].foliage };
const shell: SceneryNode = { kind: "room-shell", id: "shell", materialModel: "room", halfSize: { x: 1.45, y: 1, z: 1.45 }, floor: { colorLinear: [1, 1, 1] }, wall: { colorLinear: [1, 1, 1] }, ceiling: { colorLinear: [1, 1, 1] } };
const scene = () => ({ ...defaultScene, scenery: { palettes: {}, nodes: [planOakV2(spec).node, shell] } });
function checkGeometry(parameters: Partial<OakV2Parameters>) {
  const plan = planOakV2({ ...spec, ...parameters });
  const count = oakPrimitiveCount(plan.node.oak!.parameters);
  const wood = (plan.node.children[0] as { children: readonly SceneryTaperedSweepClusterNode[] }).children;
  const pads = (plan.node.children[1] as { children: readonly SceneryRecursiveShapeNode[] }).children;
  assert.equal(count.branches, wood.length);
  assert.equal(count.sprays, pads.length);
  assert.ok(wood.length + pads.length <= 3448);
  assert.deepEqual(validateSceneryGraph({ palettes: {}, nodes: [plan.node, shell] }), []);
  const byId = new Map(plan.branches.map(branch => [branch.id, branch]));
  const areas = new Map<string, number>();
  for (const branch of plan.branches) {
    if (branch.parent) {
      const parent = byId.get(branch.parent)!;
      assert.deepEqual(branch.from, parent.to);
      areas.set(branch.parent, (areas.get(branch.parent) ?? 0) + branch.baseRadius ** 2);
    }
    assert.ok(branch.baseRadius >= branch.tipRadius);
    assert.ok(Object.values(branch.to).every(Number.isFinite));
  }
  for (const [id, area] of areas) assert.ok(Math.abs(area - byId.get(id)!.tipRadius ** 2) < 1e-10);
  for (const node of wood) validateSvoClusterPacking({ field: "tapered-sweep", seed: node.seed, smoothRadius_m: node.smoothRadius,
    points: node.points.map(point => ({ position_m: point.position, radius_m: point.radius })) }, node.lobe);
  for (const pad of pads) assert.ok(sampleSvoPrimitive(oakFoliageDescriptor(pad), pad.place!.position!).signedDistance_m < 0);
}
test("all exposed control endpoints generate finite, connected, budgeted geometry", () => {
  for (const [key, control] of Object.entries(OAK_V2_CONTROLS)) {
    for (const value of [control.min, control.max]) {
      try { checkGeometry({ [key]: value }); }
      catch (error) { throw new Error(`${key}=${value}: ${String(error)}`, { cause: error }); }
    }
  }
});
test("presets and combined extremes retain fork area and occupied leaf attachments", () => {
  for (const preset of OAK_V2_PRESETS) checkGeometry(preset.values);
  for (const limit of ["min", "max"] as const) checkGeometry(Object.fromEntries(Object.entries(OAK_V2_CONTROLS).map(([key, control]) => [key, control[limit]])));
});
test("growth edits survive save/import and preserve placement, materials and unrelated scene state", () => {
  const initial = withSceneryPlacement(scene(), "oak", { units: "metres", position: { x: .2, y: .3, z: -.4 }, scale: .7, orientation: { w: 1, x: 0, y: 0, z: 0 } });
  const edited = withOakParameters(initial, "oak", { crownWidth: 1.4, seed: 65, leafScale: .6 });
  const node = findSceneryNode(edited, "oak");
  assert.ok(isEditableOak(node));
  assert.deepEqual(node.place, findSceneryNode(initial, "oak")!.place);
  assert.deepEqual(node.oak.bark, spec.bark);
  assert.equal(edited.fluid, initial.fluid);
  assert.equal(edited.solidVoxels, initial.solidVoxels);
  const restored = parseScene(serializeScene(edited));
  assert.deepEqual(restored.scenery, edited.scenery);
  assert.deepEqual(withOakParameters(restored, "oak", {}).scenery, edited.scenery);
  assert.equal(sceneEditRequiresReset(initial, edited, "sparse-cm12"), false);
});
test("undo/redo restores the exact recipe and generated children", () => {
  const initial = scene(), edited = withOakParameters(initial, "oak", { twigDepth: 1, showFoliage: 0 });
  const history = createEditorHistoryStore();
  history.getState().record({ scene: initial, label: "Changed tree", presetId: "test" });
  assert.deepEqual(history.getState().undo({ scene: edited, label: "current", presetId: "test" })!.scene.scenery, initial.scenery);
  assert.deepEqual(history.getState().redo({ scene: initial, label: "current", presetId: "test" })!.scene.scenery, edited.scenery);
});
test("voxel comparison preserves authored geometry, terrain, paint and seed across all depths", () => {
  const initial = { ...withOakParameters(scene(), "oak", { leafScale: .55, seed: 65 }), systems: { fluid: false } };
  for (const depth of [0, 1, 2, 3, 0]) {
    const next = withTreePreviewDepth(initial, depth);
    assert.equal(next.scenery, initial.scenery);
    assert.equal(next.terrain, initial.terrain);
    assert.equal(next.solidVoxels, initial.solidVoxels);
    assert.equal(next.voxelDomain.finestCellSize_m, initial.voxelDomain.finestCellSize_m);
    assert.equal(next.voxelDomain.detailCellSize_m, initial.voxelDomain.finestCellSize_m / 2 ** depth);
    assert.deepEqual(parseScene(serializeScene(next)).scenery, initial.scenery);
  }
  assert.throws(() => withTreePreviewDepth(scene(), 1), /Turn water off/);
  assert.throws(() => withTreePreviewDepth(initial, 9), RangeError);
});
test("all growth fields and presets use the same scene editor declarations", () => {
  const initial = scene();
  const groups = oakTreeControlGroups(initial, "oak");
  assert.equal(groups.flatMap(group => group.fields ?? []).length, 20);
  const entity = sceneryEntity.find!({ scene: initial, bodies: [] }, scenerySelectionId("oak"));
  assert.deepEqual(entity?.groups?.map(group => group.id), groups.map(group => group.id));
  const width = groups.flatMap(group => group.fields ?? []).find(field => field.id === "oak-crownWidth")!;
  const edited = { ...initial, ...width.apply(1.4) };
  const preset = oakTreeControlGroups(edited, "oak")[0].choices![0].options[0];
  const restored = findSceneryNode({ ...edited, ...preset.apply() }, "oak");
  assert.ok(isEditableOak(restored));
  assert.deepEqual(restored.oak.parameters, OAK_V2_DEFAULTS);
});
test("planting makes uniquely identified selectable oaks at the picked base", () => {
  const initial = scene(), point = { x: .3, y: .12, z: -.2 };
  const node = createSceneryNodeAt(initial, "oak-v2", point, { x: 0, y: 1, z: 0 });
  assert.ok(isEditableOak(node));
  assert.deepEqual(node.place!.position, point);
  const next = { ...initial, scenery: { ...initial.scenery, nodes: [...initial.scenery.nodes, node] } };
  assert.notEqual(createSceneryNodeAt(next, "oak-v2", point, { x: 0, y: 1, z: 0 }).id, node.id);
  assert.deepEqual(validateSceneryGraph(next.scenery), []);
});
test("colour edits preserve geometry and survive the next regeneration", () => {
  const initial = scene(), recoloured = withOakMaterial(initial, "oak", OAK_MATERIALS[1]);
  const before = findSceneryNode(initial, "oak"), after = findSceneryNode(recoloured, "oak");
  assert.ok(isEditableOak(before) && isEditableOak(after));
  assert.deepEqual(after.children.map(group => group.id), before.children.map(group => group.id));
  const regenerated = findSceneryNode(withOakParameters(recoloured, "oak", { seed: 12 }), "oak");
  assert.ok(isEditableOak(regenerated));
  assert.deepEqual(regenerated.oak.foliage, OAK_MATERIALS[1].foliage);
});
test("invalid imported recipes are rejected; ordinary legacy scenery remains editable", () => {
  const initial = scene();
  for (const bad of [{ twigDepth: 9 }, { sitesPerBough: 5.5 }, { leafScale: -1 }, { scale_m: Infinity }, { seed: NaN }]) {
    assert.throws(() => withOakParameters(initial, "oak", bad), RangeError);
    const input = JSON.parse(serializeScene(initial));
    Object.assign(input.scenery.nodes[0].oak.parameters, bad);
    assert.throws(() => parseScene(JSON.stringify(input)), /Oak v2/);
  }
  const legacy = { ...initial, scenery: { palettes: {}, nodes: initial.scenery.nodes.map(node => { if (!isEditableOak(node)) return node; const { oak, ...legacy } = node; assert.equal(oak.version, 1); return legacy; }) } };
  assert.deepEqual(oakTreeControlGroups(legacy, "oak"), []);
  assert.deepEqual(parseScene(serializeScene(legacy)).scenery, legacy.scenery);
});


test("plugin groups declare compact readouts and safe scene-wide voxel choices", () => {
  const wet = scene();
  const groups = oakTreeControlGroups(wet, "oak");
  assert.equal(groups.length, 6);
  assert.ok(groups.every(group => group.readout && !group.defaultOpen));
  assert.equal(groups.find(group => group.id === "oak-twigs")?.readout, "3 forks");
  const comparison = groups.find(group => group.id === "oak-voxel-comparison")!;
  assert.equal(comparison.tag, "Voxels");
  assert.ok(comparison.choices![0].options.every(option => option.enabled === false));
  const dry = { ...wet, systems: { ...wet.systems, fluid: false } };
  for (const option of oakTreeControlGroups(dry, "oak").at(-1)!.choices![0].options) {
    assert.equal(option.enabled, true);
    const patch = option.apply();
    assert.equal(patch.scenery, dry.scenery);
    assert.equal(patch.voxelDomain!.detailCellSize_m, dry.voxelDomain.finestCellSize_m / 2 ** Number(option.id));
  }
  const changed = withOakParameters(dry, "oak", { twigDepth: 1, showFoliage: 0 });
  const updated = oakTreeControlGroups(changed, "oak");
  assert.equal(updated.find(group => group.id === "oak-twigs")?.readout, "1 fork");
  assert.equal(updated.find(group => group.id === "oak-foliage")?.readout, "Bare");
});
