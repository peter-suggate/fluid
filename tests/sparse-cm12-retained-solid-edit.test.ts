import assert from "node:assert/strict";
import test from "node:test";
import { createSolidWorld, withSolidWorldPatches, SOLID_WORLD_TERRAIN_MATERIAL_ID, type SolidWorld, type SolidWorldVoxelPatch } from "../lib/core/solid-world";
import { retainedSceneDensity, compileRetainedSceneFineMeans } from "../lib/methods/adaptive-mass/sparse-cm12-retained-scene-density";
import { compileRetainedOpenSceneFineMeans } from "../lib/methods/adaptive-mass/sparse-cm12-retained-open-density";
import { compileRetainedSceneSubcellMoments } from "../lib/methods/adaptive-mass/sparse-cm12-retained-subcell-moments";
import { compileRetainedSolidEditDelta } from "../lib/methods/adaptive-mass/sparse-cm12-retained-solid-edit";
const dimensions = [8, 8, 8] as const, h = .125;
const field = retainedSceneDensity({ generation: 1, transitionWidth: h,
  domain: { lower: [-.5, 0, -.5], upper: [.5, 1, .5] },
  primitives: [{ kind: "quadratic-height", center: [0, h / 2, 0], curvature: [0, 0, 0] }] });
const seedMeans = compileRetainedSceneFineMeans(field, dimensions, h);
const fill: SolidWorldVoxelPatch = { operation: "fill", minimum: [1, 0, 1], maximumExclusive: [3, 1, 2] };
function compare(previous: SolidWorld, next: SolidWorld) {
  const before = seedMeans.slice();
  const delta = compileRetainedSolidEditDelta(field, dimensions, h, previous, next, { seedMeans, rigid: true });
  const old = compileRetainedOpenSceneFineMeans(field, dimensions, h, previous, { seedMeans });
  const full = compileRetainedOpenSceneFineMeans(field, dimensions, h, next, { seedMeans });
  const sub = compileRetainedSceneSubcellMoments(field, dimensions, h, next, { solidFractions: full.solidFractions });
  const expected = Array.from(full.solidFractions.keys()).filter(i => full.solidFractions[i] !== old.solidFractions[i]);
  assert.deepEqual(Array.from(delta.indices), expected);
  for (let j = 0; j < delta.indices.length; j++) {
    const i = delta.indices[j]!;
    assert.equal(delta.previousSolidFractions[j], old.solidFractions[i]);
    assert.equal(delta.solidFractions[j], full.solidFractions[i]);
    assert.ok(Math.abs(delta.effectiveMeans[j]! - full.effectiveMeans[i]!) < 1e-7);
    assert.equal(delta.openFractions[j], full.openFractions[i]);
    for (let k = 0; k < 8; k++) {
      assert.ok(Math.abs(delta.subcells!.seedAmounts[8 * j + k]! - sub.seedAmounts[8 * i + k]!) < 1e-7);
      assert.equal(delta.subcells!.openVolumes[8 * j + k], sub.openVolumes[8 * i + k]);
    }
  }
  assert.deepEqual(seedMeans, before);
  return delta;
}
test("compact fill and undo match exact initial compilers, including rigid octants", () => {
  const empty = createSolidWorld(), filled = withSolidWorldPatches(empty, [fill]);
  const added = compare(empty, filled), removed = compare(filled, empty);
  assert.equal(added.receipt.visitedCells, 512);
  assert.equal(removed.indices.length, 2);
  assert.deepEqual(removed.changedCellRanges, [{ firstCell: 65, cellCount: 2, compactOffset: 0 }]);
  assert.equal(removed.subcells!.seedAmounts.length, 16);
});
test("fractional terrain uses the exact open slab and partial octant integrals", () => {
  const empty = createSolidWorld(), terrain = createSolidWorld([fill]);
  terrain.pages[0]!.solidFraction[65] = 96;
  terrain.pages[0]!.materialId[65] = SOLID_WORLD_TERRAIN_MATERIAL_ID;
  const result = compare(empty, terrain);
  assert.ok(result.receipt.integratedBoxes > 0);
  assert.ok(result.receipt.maximumEstimatedOpenMeanError <= 2e-7);
  compare(terrain, empty);
});
test("ordered regions override changed pages and equal reconstructed regions cause no work", () => {
  const base = createSolidWorld();
  const clear: SolidWorldVoxelPatch = { ...fill, operation: "clear", maximumExclusive: [2, 1, 2] };
  const old: SolidWorld = { ...base, regions: [fill, clear] };
  const equal: SolidWorld = { ...base, regions: old.regions!.map(r => ({ ...r, minimum: [...r.minimum], maximumExclusive: [...r.maximumExclusive] })) };
  assert.equal(compare(old, equal).receipt.visitedCells, 0);
  assert.equal(compare(old, { ...base, regions: old.regions!.map(r => ({ ...r, materialId: 7 })) }).receipt.visitedCells, 0);
  const reordered: SolidWorld = { ...base, regions: [clear, fill] };
  assert.equal(compare(old, reordered).indices.length, 1);
  const page = createSolidWorld([fill]);
  assert.equal(compare(old, { ...page, regions: old.regions }).indices.length, 0);
});
test("material-only edits do not integrate moments and work budgets reject without mutation", () => {
  const filled = createSolidWorld([fill]);
  const material = withSolidWorldPatches(filled, [{ ...fill, materialId: 7 }]);
  const unchanged = compare(filled, material);
  assert.equal(unchanged.indices.length, 0);
  assert.equal(unchanged.receipt.integratedBoxes, 0);
  const before = filled.pages[0]!.solidFraction.slice();
  assert.throws(() => compileRetainedSolidEditDelta(field, dimensions, h, filled, createSolidWorld(), { seedMeans, maximumVisitedCells: 511 }), /work budget/);
  assert.throws(() => compileRetainedSolidEditDelta(field, dimensions, h, filled, createSolidWorld(), { seedMeans, maximumChangedCells: 1 }), /too many/);
  const broad: SolidWorld = { ...createSolidWorld(), regions: [{ operation: "fill", minimum: [-100, -100, -100], maximumExclusive: [100, 100, 100] }] };
  assert.throws(() => compileRetainedSolidEditDelta(field, dimensions, h, filled, broad, { seedMeans, maximumVisitedCells: 600 }), /work budget/);
  assert.deepEqual(filled.pages[0]!.solidFraction, before);
});

test("immutable preparation overlays replace repeated edits and materialize exact geometry", async () => {
  const { compileRetainedScenePreparationCache, withRetainedSolidEdit } = await import("../lib/methods/adaptive-mass/sparse-cm12-retained-preparation-cache");
  const empty = createSolidWorld(), filled = createSolidWorld([fill]);
  const initial = compileRetainedScenePreparationCache(field, dimensions, h, empty, { rigid: true });
  const add = compileRetainedSolidEditDelta(field, dimensions, h, empty, filled, { seedMeans, rigid: true });
  const staged = withRetainedSolidEdit(initial, add);
  assert.equal(staged.openMeans, initial.openMeans);
  assert.equal(staged.subcellMoments, initial.subcellMoments);
  assert.equal(initial.solidEditPages, undefined);
  const materialized = compileRetainedScenePreparationCache(field, dimensions, h, filled, { previous: staged, rigid: true });
  assert.equal(materialized.openMeans.solidFractions[65], 255);
  assert.equal(initial.openMeans.solidFractions[65], 0);
  const bounded = withRetainedSolidEdit(initial, { ...add, receipt: { ...add.receipt,
    estimatedAbsoluteError: 1e-12, maximumEstimatedOpenMeanError: 1e-8,
    maximumEstimatedSubcellMeanError: 2e-8 } });
  const boundedMaterialized = compileRetainedScenePreparationCache(field, dimensions, h, filled, { previous: bounded, rigid: true });
  assert.ok(boundedMaterialized.openMeans.receipt.estimatedAbsoluteError >= 1e-12);
  assert.ok(boundedMaterialized.openMeans.receipt.maximumEstimatedOpenMeanError >= 1e-8);
  assert.ok(boundedMaterialized.subcellMoments!.receipt.maximumEstimatedSubcellMeanError >= 2e-8);
  const undo = compileRetainedSolidEditDelta(field, dimensions, h, filled, empty, { seedMeans, rigid: true });
  const undone = withRetainedSolidEdit(staged, undo);
  assert.equal(undone.solidEditPages!.size, 1);
  assert.equal(undone.solidEditPages!.get(0)!.size, 2);
  assert.notEqual(undone.solidEditPages!.get(0), staged.solidEditPages!.get(0));
  const restored = compileRetainedScenePreparationCache(field, dimensions, h, empty, { previous: undone, rigid: true });
  assert.deepEqual(restored.openMeans.effectiveMeans, initial.openMeans.effectiveMeans);
  assert.deepEqual(restored.subcellMoments!.seedAmounts, initial.subcellMoments!.seedAmounts);
});

test("resident preflight rejects before publication and commits only compact ranges without recompilation", async () => {
  const { WebGPUSparseCM12Resident } = await import("../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident");
  const { createSparseCM12SolidOccupancyLayout } = await import("../lib/methods/adaptive-mass/sparse-cm12-solid-occupancy");
  const { compileRetainedScenePreparationCache } = await import("../lib/methods/adaptive-mass/sparse-cm12-retained-preparation-cache");
  const empty = createSolidWorld(), filled = createSolidWorld([fill]);
  const stateBuffer = {}, writes: { offset: number; bytes: number }[] = [];
  const support = { dimensions, seedMeanBaseWords: 1000, openFractionBaseWords: 2000 };
  const state = Object.assign(Object.create(WebGPUSparseCM12Resident.prototype), {
    destroyed: false, dimensions, currentSolidWorld: empty,
    solidOccupancyLayout: createSparseCM12SolidOccupancyLayout({ baseWords: 0, authoredPageCount: 0 }),
    replacementConfiguration: { retainedDensity: field, finestCellSize_m: h },
    retainedDensityLayout: { support }, retainedUnrestrictedMeans: seedMeans,
    retainedPreparationCache: compileRetainedScenePreparationCache(field, dimensions, h, empty),
    state: stateBuffer, topologyArena: {}, templateCellCount: 512, bindGroup: {},
    pipelines: { refreshRetainedDensityNativeIntegralImage: {} },
    writeParameters() {}, encodeSolidWorldApertureRefresh() {},
    device: { queue: { submit() {}, writeBuffer(buffer: object, offset: number, _data: unknown, _start: number, bytes: number) {
      if (buffer === stateBuffer) writes.push({ offset, bytes });
    } }, createCommandEncoder() { return { clearBuffer() {}, finish() { return {}; }, beginComputePass() {
      return { setBindGroup() {}, setPipeline() {}, dispatchWorkgroups() {}, end() {} };
    } }; } },
  });
  const host = state as import("../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident").WebGPUSparseCM12Resident;
  const invalid = createSolidWorld([fill]); invalid.pages[0]!.solidFraction[65] = 128;
  assert.throws(() => host.validateSolidWorld(invalid), /no declared subvoxel geometry/);
  assert.equal(state.currentSolidWorld, empty);
  assert.equal(writes.length, 0);
  host.validateSolidWorld(filled);
  const prepared = state.preparedSolidEdit;
  host.validateSolidWorld(filled);
  assert.equal(state.preparedSolidEdit, prepared);
  // Any repeated sampling at commit would hit this trap.
  const lookup = filled.directory.lookup;
  filled.directory.lookup = () => { throw new Error("duplicate preparation"); };
  host.setSolidWorld(filled);
  filled.directory.lookup = lookup;
  assert.deepEqual(writes, [{ offset: 4 * (1000 + 65), bytes: 8 }, { offset: 4 * (2000 + 65), bytes: 8 }]);
  assert.equal(state.currentSolidWorld, filled);
  assert.equal(state.preparedSolidEdit, undefined);
});

test("fluid collider wrappers retain identity from scene preflight to commit", async () => {
  const { cloneScene, defaultScene } = await import("../lib/core/model");
  const { fluidSolidWorldForScene, sceneWithSolidStroke } = await import("../lib/core/solid-world");
  const scene = cloneScene(defaultScene);
  const before = fluidSolidWorldForScene(scene);
  assert.equal(fluidSolidWorldForScene(scene), before);
  const edited = sceneWithSolidStroke(scene, [fill]);
  const after = fluidSolidWorldForScene(edited);
  assert.notEqual(after, before);
  assert.equal(fluidSolidWorldForScene(edited), after);
});
