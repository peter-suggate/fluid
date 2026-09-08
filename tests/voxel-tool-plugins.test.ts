import assert from "node:assert/strict";
import test from "node:test";
import { cloneScene, defaultScene, parseScene, serializeScene } from "../lib/core/model";
import { sceneCellSizes_m, solidVoxelShellForScene } from "../lib/core/scene-lattice";
import { sampleSolidWorld, solidWorldForScene, sceneWithSolidStroke, withSolidWorldPatches } from "../lib/core/solid-world";
import { interpolateCells, mirrorPatchX, shapePatches } from "../lib/core/voxel-editor/geometry";
import { createVoxelToolRegistry, toolValues } from "../lib/core/voxel-editor/plugin";
import { voxelTools } from "../lib/core/voxel-editor/registry";
import { beginToolTransaction } from "../lib/core/voxel-editor/transaction";
import { createWebgpuSolidWorldPageLayout, writeWebgpuSolidWorldPages } from "../lib/core/webgpu-solid-world-pages";

const solidTools = voxelTools.tools.filter(plugin => plugin.execution !== "release");

function empty() {
  const scene = cloneScene(defaultScene);
  scene.solidVoxels = []; scene.terrain = undefined;
  return scene;
}
function ray(scene: ReturnType<typeof empty>, x: number, z: number) {
  const h = sceneCellSizes_m(scene);
  return { origin: { x: -scene.container.width_m / 2 + (x + .5) * h[0], y: 10,
    z: -scene.container.depth_m / 2 + (z + .5) * h[2] }, direction: { x: 0, y: -1, z: 0 } };
}

test("every registered tool owns its UI controls, icon and executable gesture", () => {
  const scene = empty();
  for (const plugin of solidTools) {
    assert.ok(plugin.ui.icon && plugin.ui.hint && plugin.ui.group);
    const initial = ray(scene, -2, 1);
    const gesture = plugin.begin({ scene, ray: initial, values: toolValues(plugin) });
    const result = gesture?.update(initial);
    assert.ok(result?.patches.length, plugin.id);
    const saved = sceneWithSolidStroke(scene, result.patches);
    assert.deepEqual(parseScene(serializeScene(saved)).solidVoxels, saved.solidVoxels);
  }
  assert.throws(() => createVoxelToolRegistry([voxelTools.tools[0]!, voxelTools.tools[0]!]));
  const extension = { ...voxelTools.tools[0]!, id: "third-party-brush" };
  assert.equal(createVoxelToolRegistry([extension]).get(extension.id), extension);
  const externalControl = { ...extension.ui.controls[0]!, presentation: undefined };
  assert.doesNotThrow(() => createVoxelToolRegistry([{ ...extension,
    ui: { ...extension.ui, controls: [externalControl] } }]));
  assert.throws(() => createVoxelToolRegistry([{ ...extension,
    ui: { ...extension.ui, controls: [{ ...externalControl,
      presentation: "hidden" as "advanced" }] } }]), /Invalid control/);
});

test("fast strokes cover negative coordinates continuously and mirroring is involutive", () => {
  const points = interpolateCells([-9, 0, 3], [10, 0, 3]);
  assert.equal(points.length, 20);
  assert.deepEqual(points.map((p) => p[0]), Array.from({ length: 20 }, (_, i) => i - 9));
  assert.throws(() => interpolateCells([0, 0, 0], [10000, 0, 0]));
  const patch = shapePatches([-3, 0, 0], [-1, 1, 1], "fill", "box")[0]!;
  assert.deepEqual(mirrorPatchX(mirrorPatchX(patch, 32), 32), patch);
});

test("sphere and oriented drill have curved cross-sections and exact bounds", () => {
  const scene = empty();
  const sphere = solidWorldForScene(sceneWithSolidStroke(scene, shapePatches([-2, -2, -2], [3, 3, 3], "fill", "sphere")));
  assert.equal(sampleSolidWorld(sphere, [0, 0, 0]).solidFraction, 1);
  assert.equal(sampleSolidWorld(sphere, [-2, -2, -2]).solidFraction, 0);
  assert.equal(sampleSolidWorld(sphere, [3, 0, 0]).solidFraction, 0);
  const cylinder = solidWorldForScene(sceneWithSolidStroke(scene,
    shapePatches([-4, -2, -2], [5, 3, 3], "fill", "cylinder", 0)));
  assert.equal(sampleSolidWorld(cylinder, [-4, 0, 0]).solidFraction, 1);
  assert.equal(sampleSolidWorld(cylinder, [4, 0, 0]).solidFraction, 1);
  assert.equal(sampleSolidWorld(cylinder, [0, -2, -2]).solidFraction, 0);
});

test("box retraction restores its base, freehand accumulates, and both persist", () => {
  for (const id of ["box", "build"]) {
    const scene = empty(); const plugin = voxelTools.get(id)!;
    const start = ray(scene, 0, 0);
    const gesture = plugin.begin({ scene, ray: start, values: toolValues(plugin) })!;
    gesture.update(start); gesture.update(ray(scene, 6, 0));
    const result = gesture.update(ray(scene, 1, 0))!;
    const world = solidWorldForScene(sceneWithSolidStroke(scene, result.patches));
    assert.equal(sampleSolidWorld(world, [0, 0, 0]).solidFraction, 1);
    assert.equal(sampleSolidWorld(world, [6, 0, 0]).solidFraction, id === "box" ? 0 : 1);
  }
});

test("a transaction publishes before release, records once, cancels live and rejects atomically", async () => {
  let scene = empty(); const base = scene;
  let begins = 0, commits = 0, cancels = 0, reject = false;
  const transaction = () => beginToolTransaction(voxelTools.get("build")!, {
    scene: () => scene,
    async publish(next) { if (reject) throw new Error("capacity"); scene = next; },
    begin() { begins++; }, finish() { commits++; }, cancel() { cancels++; },
  }, ray(scene, 0, 0))!;
  const stroke = transaction();
  await stroke.update(ray(scene, 0, 0));
  assert.notEqual(scene, base); assert.equal(commits, 0);
  const accepted = scene;
  reject = true;
  await assert.rejects(stroke.update(ray(scene, 2, 0)), /capacity/);
  assert.equal(scene, accepted);
  reject = false;
  await stroke.finish(true);
  assert.equal(scene, base); assert.equal(cancels, 1); assert.equal(commits, 0);
  const second = transaction();
  await second.update(ray(scene, 1, 0)); await second.finish(); await second.finish();
  assert.equal(commits, 1); assert.equal(begins, 2);
});

test("small edits copy and upload only changed page payloads", () => {
  const scene = empty();
  scene.solidVoxels = shapePatches([0, 0, 0], [16, 1, 1], "fill", "box");
  const base = solidWorldForScene(scene);
  const next = withSolidWorldPatches(base, [{ operation: "clear", minimum: [0, 0, 0], maximumExclusive: [1, 1, 1] }]);
  assert.notEqual(next.pages[0], base.pages[0]);
  assert.equal(next.pages[1], base.pages[1]);
  assert.equal(sampleSolidWorld(base, [0, 0, 0]).solidFraction, 1);
  assert.equal(sampleSolidWorld(next, [0, 0, 0]).solidFraction, 0);
  const layout = createWebgpuSolidWorldPageLayout({ baseWords: 0, authoredPageCount: 2 });
  const writes: number[] = [];
  const queue = { writeBuffer(_buffer: unknown, offset: number) { writes.push(offset); } } as unknown as GPUQueue;
  writeWebgpuSolidWorldPages(queue, {} as GPUBuffer, layout, next, [0, 0, 0], undefined, base);
  assert.equal(writes.filter((offset) => offset >= 4 * layout.pageBaseWords).length, 1);
});

function faceRay(scene: ReturnType<typeof empty>, at: readonly number[], axis: number, sign: number) {
  const h = sceneCellSizes_m(scene);
  const origin = [-scene.container.width_m / 2, 0, -scene.container.depth_m / 2];
  const position = at.map((v, a) => origin[a]! + (v + .5 + (a === axis ? 20 * sign : 0)) * h[a]!);
  const direction = [0, 0, 0]; direction[axis] = -sign;
  return { origin: { x: position[0]!, y: position[1]!, z: position[2]! },
    direction: { x: direction[0]!, y: direction[1]!, z: direction[2]! } };
}
function cells(patches: ReturnType<typeof shapePatches> | readonly ReturnType<typeof shapePatches>[number][]) {
  const occupied = new Set<string>();
  for (const p of patches) for (let z = p.minimum[2]; z < p.maximumExclusive[2]; z++)
    for (let y = p.minimum[1]; y < p.maximumExclusive[1]; y++)
      for (let x = p.minimum[0]; x < p.maximumExclusive[0]; x++) occupied.add([x, y, z].join(","));
  return occupied;
}
const subtractTools = new Set(["carve", "cut", "drill", "channel"]);

test("all eight tools extrude outward or cut inward on every signed face and persist exact occupancy", () => {
  for (const axis of [0, 1, 2]) for (const sign of [-1, 1]) for (const plugin of solidTools) {
    const scene = empty();
    scene.solidVoxels = shapePatches([10, 10, 10], [17, 17, 17], "fill", "box");
    const anchor = [13, 13, 13]; anchor[axis] = sign > 0 ? 16 : 10;
    const input = faceRay(scene, anchor, axis, sign);
    const subtract = subtractTools.has(plugin.id);
    const values = toolValues(plugin, { size: 3, depth: 3, shell: 1 });
    const gesture = plugin.begin({ scene, ray: input, values });
    const result = gesture?.update(input);
    assert.ok(result, `${plugin.id} axis ${axis} sign ${sign}`);
    const affected = cells(result.patches);
    const positions = [...affected].map((p) => Number(p.split(",")[axis]));
    const first = anchor[axis]! + (subtract ? 0 : sign);
    const final = first + 2 * sign * (subtract ? -1 : 1);
    assert.equal(Math.min(...positions), Math.min(first, final), plugin.id);
    assert.equal(Math.max(...positions), Math.max(first, final), plugin.id);
    const serialized = parseScene(serializeScene(sceneWithSolidStroke(scene, result.patches)));
    const world = solidWorldForScene(serialized);
    for (const key of affected) {
      const coordinate = key.split(",").map(Number) as [number, number, number];
      assert.equal(sampleSolidWorld(world, coordinate).solidFraction, subtract ? 0 : 1, `${plugin.id} ${key}`);
    }
    const unedited = [...anchor] as [number, number, number];
    unedited[axis] -= sign * 3;
    assert.equal(sampleSolidWorld(world, unedited).solidFraction, 1, plugin.id);
  }
});

test("empty-space construction planes support negative heights and parallel or backward rays are ignored", () => {
  const scene = empty();
  for (const plugin of solidTools) {
    const input = ray(scene, -5, -5);
    const gesture = plugin.begin({ scene, ray: input, values: toolValues(plugin, { plane: -3 }) })!;
    const result = gesture.update(input)!;
    assert.equal(Math.min(...result.patches.map((p) => p.minimum[1])), -3, plugin.id);
    assert.equal(gesture.update({ ...input, direction: { x: 1, y: 0, z: 0 } }), undefined);
    assert.equal(gesture.update({ ...input, direction: { x: 0, y: 1, z: 0 } }), undefined);
  }
});

test("each tool mirrors the complete proposal across the container centre", () => {
  const scene = empty(); const centre = Math.round(scene.container.width_m / sceneCellSizes_m(scene)[0]);
  for (const plugin of solidTools) {
    const input = ray(scene, -5, -5);
    const values = toolValues(plugin, { size: 3, depth: 2, mirror: 1 });
    const gesture = plugin.begin({ scene, ray: input, values })!;
    gesture.update(input);
    const affected = cells(gesture.update(ray(scene, -2, -5))!.patches);
    for (const key of affected) {
      const [x, y, z] = key.split(",").map(Number);
      assert.ok(affected.has([centre - x! - 1, y, z].join(",")), `${plugin.id} ${key}`);
    }
  }
});

test("box, cut, wall, channel and movable stamps retract while freehand edits accumulate", () => {
  for (const plugin of solidTools) {
    const scene = empty(); const start = ray(scene, -9, -5);
    const gesture = plugin.begin({ scene, ray: start, values: toolValues(plugin, { size: 1, depth: 1 }) })!;
    gesture.update(start); gesture.update(ray(scene, -3, -5));
    const affected = cells(gesture.update(ray(scene, -8, -5))!.patches);
    assert.equal(affected.has("-3,0,-5"), ["build", "carve"].includes(plugin.id), plugin.id);
    assert.equal(affected.has("-9,0,-5"), !["sphere", "drill"].includes(plugin.id), plugin.id);
    assert.ok(affected.has("-8,0,-5"), plugin.id);
  }
});

test("brush and line interpolation stay connected across diagonal pointer jumps", () => {
  const scene = empty();
  for (const id of ["build", "carve", "wall", "channel"]) {
    const plugin = voxelTools.get(id)!; const start = ray(scene, -20, -20);
    const gesture = plugin.begin({ scene, ray: start, values: toolValues(plugin) })!;
    const affected = cells(gesture.update(ray(scene, -10, -15))!.patches);
    for (const at of interpolateCells([-20, 0, -20], [-10, 0, -15])) assert.ok(affected.has(at.join(",")), id);
  }
});

test("size and stroke caps reject atomically, allowing subsequent smaller updates", () => {
  assert.throws(() => shapePatches([0, 0, 0], [33, 33, 33], "fill", "box"), /smaller region/);
  assert.throws(() => shapePatches([0, 0, 0], [0, 1, 1], "fill", "sphere"), /smaller region/);
  const scene = empty();
  for (const id of ["box", "cut", "build", "carve", "wall", "channel"]) {
    const plugin = voxelTools.get(id)!; const start = ray(scene, -100, -100);
    const gesture = plugin.begin({ scene, ray: start, values: toolValues(plugin, { depth: 32, size: 16 }) })!;
    const accepted = gesture.update(start)!;
    assert.throws(() => gesture.update(ray(scene, 500, 500)), /region|span|full/, id);
    assert.deepEqual(gesture.update(start)!.patches, accepted.patches, id);
  }
});

test("shell picking passes through tank walls by default, can target walls, and keeps floor usable", () => {
  const scene = empty();
  scene.solidVoxels = [...solidVoxelShellForScene(scene), ...shapePatches([2, 2, 2], [5, 5, 5], "fill", "box")];
  const plugin = voxelTools.get("build")!;
  const input = faceRay(scene, [-1, 3, 3], 0, -1);
  const update = (shell: number) => plugin.begin({ scene, ray: input, values: toolValues(plugin, { shell }) })!.update(input)!;
  assert.ok(cells(update(0).patches).has("1,3,3"));
  assert.ok(cells(update(1).patches).has("-2,3,3"));
  const floorRay = ray(scene, 8, 8);
  const floorUpdate = plugin.begin({ scene, ray: floorRay, values: toolValues(plugin) })!.update(floorRay)!;
  assert.ok(cells(floorUpdate.patches).has("8,0,8"));
});

test("stroke work caps account for mirrored and overlapping stamp workloads", () => {
  const scene = empty(); const plugin = voxelTools.get("build")!;
  const input = ray(scene, -30, -30);
  const gesture = plugin.begin({ scene, ray: input, values: toolValues(plugin, { size: 16, depth: 32, mirror: 1 }) })!;
  const accepted = gesture.update(input)!;
  assert.throws(() => gesture.update(ray(scene, -28, -30)), /Stroke is full/);
  assert.deepEqual(gesture.update(input)!.patches, accepted.patches);
});

test("long thin freehand strokes enforce the patch cap without discarding accepted geometry", () => {
  const scene = empty(); const plugin = voxelTools.get("build")!;
  const input = ray(scene, -10000, -10000);
  const gesture = plugin.begin({ scene, ray: input, values: toolValues(plugin) })!;
  let accepted = gesture.update(input)!;
  for (let i = 1; i <= 15; i++) accepted = gesture.update(ray(scene, -10000 + i * 256, -10000))!;
  assert.equal(accepted.patches.length, 3841);
  assert.throws(() => gesture.update(ray(scene, -10000 + 16 * 256, -10000)), /Stroke is full/);
  assert.deepEqual(gesture.update(ray(scene, -10000 + 15 * 256, -10000))!.patches, accepted.patches);
});
