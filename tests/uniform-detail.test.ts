import "../lib/methods";
import assert from "node:assert/strict";
import test from "node:test";
import type { FluidRefinementRegion } from "../lib/core/model";
import type { RefinementRegionLattice } from "../lib/core/refinement-regions";
import { createMethodStore, resolvedMethodValues } from "../lib/core/stores/method-store";
import { parseMethodQueryState, serializeQueryState, parseQueryState } from "../lib/core/url-state";
import { structuralMethodValues } from "../lib/core/webgpu-renderer";
import { createDetailFocusCoalescer } from "../lib/core/solver-detail";
import { createUniformMixedLayout, mixedCellWidth } from "../lib/methods/uniform/uniform-mixed-layout";
import { UNIFORM_GEOMETRIC_NATIVE_PARAMS } from "../lib/methods/uniform/uniform-geometric-parameters";
import { uniformVolumeMethod } from "../lib/methods/uniform/uniform-volume-method";
import { UNIFORM_DETAIL_PARAM_KEYS, migrateUniformDetailOverrides, uniformDetailSettings, uniformDetailSensitivityScale, uniformDetailImportance, uniformDetailValues } from "../lib/methods/uniform/uniform-detail-policy";
import { buildUniformDetailRequests, uniformDetailDomain, uniformRegionDetailTier, type DetailRequest } from "../lib/methods/uniform/uniform-detail-requests";
import { UniformDetailPlanner, UNIFORM_DETAIL_REASON } from "../lib/methods/uniform/uniform-detail-planner";
import { UniformDetailPool, UNIFORM_DETAIL_NO_SLOT } from "../lib/methods/uniform/uniform-detail-pool";

/**
 * The detail contract of docs/plans/uniform-4h-first-implementation-handoff-2026-10-03.md
 * (WS1): request precedence, the planner's tile mask and support closure,
 * hysteresis on accepted steps, deferred slot reclamation, and the retired
 * `coarsening` control's migration. CPU-only; the solver consumer is Dawn's.
 */

// 64³ cells of 1/64 m: 16³ tiles, 2³ patches of 8³ tiles (32³ cells).
const lattice: RefinementRegionLattice = { dimensions: [64, 64, 64], cellSize_m: [1 / 64, 1 / 64, 1 / 64], origin_m: { x: 0, y: 0, z: 0 } };
const domain = uniformDetailDomain(lattice);
const tile_m = 4 / 64;
const box = (id: string, lo: number, hi: number, cells: [number, number?], tiles = false): FluidRefinementRegion => {
  const s = tiles ? tile_m : 1;
  return { id, rule: "minimum-cell-size", minimumCellSize_cells: cells[0], maximumCellSize_cells: cells[1],
    min_m: { x: lo * s, y: lo * s, z: lo * s }, max_m: { x: hi * s, y: hi * s, z: hi * s } } as FluidRefinementRegion;
};
const fine = (id: string, lo: number, hi: number, tiles = true) => box(id, lo, hi, [1, 1], tiles);
const settings = (values: Record<string, string | number> = {}) => uniformDetailSettings(values);

test("Surface distance defaults to zero, clamps to 0..3 and round-trips", () => {
  assert.equal(settings().surfaceDistance, 0);
  for (const [input, expected] of [[-1, 0], [0, 0], [1, 1], [2, 2], [3, 3], [4, 3], [1.6, 2]]) {
    const resolved = settings({ detailPolicy: "dynamic", detailSurface: "on", detailSurfaceDistance: input! });
    assert.equal(resolved.surfaceDistance, expected);
    assert.equal(uniformDetailImportance(resolved, 512).surfaceDistance, expected);
    assert.equal(uniformDetailSettings(uniformDetailValues(resolved)).surfaceDistance, expected);
  }
});

test("Surface is an opt-in Dynamic filter and survives settings serialization", () => {
  assert.equal(settings().surfaceOnly, false);
  const dynamic = settings({ detailPolicy: "dynamic", detailSurface: "on" });
  assert.equal(uniformDetailImportance(dynamic, 512).surfaceOnly, true);
  assert.equal(uniformDetailSettings(uniformDetailValues(dynamic)).surfaceOnly, true);
  for (const detailPolicy of ["full", "requested"]) {
    assert.equal(uniformDetailImportance(settings({ detailPolicy, detailSurface: "on" }), 512).surfaceOnly, false);
  }
});
const plan = (planner: UniformDetailPlanner, regions: FluidRefinementRegion[], step: number, values: Record<string, string | number> = {},
  extra: { focus?: { x: number; y: number; z: number }; activity?: DetailRequest[]; capacityPatches?: number } = {}) => {
  const s = settings(values);
  const set = buildUniformDetailRequests({ settings: s, regions, domain, acceptedStep: step, activity: extra.activity,
    focus: extra.focus && { revision: 1, position_m: extra.focus } });
  const p = planner.plan({ requests: set.requests, suppressions: set.suppressions, automaticSources: set.automaticSources,
    acceptedStep: step, budgetPercent: s.budgetPercent, capacityPatches: extra.capacityPatches });
  planner.commit(p);
  return p;
};
const count = (mask: Uint8Array) => mask.reduce((n, v) => n + v, 0);
const tileKey = (x: number, y: number, z: number) => x + 16 * (y + 16 * z);

test("legacy cell-size bounds map onto the two held tiers", () => {
  assert.equal(uniformRegionDetailTier(box("a", 0, 1, [1, 1])), "fine");
  assert.equal(uniformRegionDetailTier(box("b", 0, 1, [1, 2])), "fine");
  assert.equal(uniformRegionDetailTier(box("c", 0, 1, [4, 4])), "coarse");
  assert.equal(uniformRegionDetailTier(box("d", 0, 1, [2])), "coarse");
  assert.equal(uniformRegionDetailTier(box("e", 0, 1, [8, 32])), "coarse");
  // Both sizes allowed: the region enforces nothing and requests nothing.
  assert.equal(uniformRegionDetailTier(box("f", 0, 1, [1])), "automatic");
  assert.equal(uniformRegionDetailTier(box("g", 0, 1, [1, 4])), "automatic");
});

test("Requested h tiles are exactly the mixed layout's Fine tiles on a 4h background", () => {
  // Off-grid bounds, a Fine box over a Coarse one, and an unconstrained box.
  const regions = [box("f1", 0.11, 0.37, [1, 1]), box("c1", 0.2, 0.6, [4, 4]), box("f2", 0.55, 0.58, [1, 1]),
    box("u", 0.7, 0.9, [1]), box("c2", 0.62, 0.99, [8, 32])];
  const p = plan(new UniformDetailPlanner(lattice), regions, 0, { detailPolicy: "requested" });
  const layout = createUniformMixedLayout(lattice, regions, 4);
  assert.deepEqual([...p.fine], [...layout.tiles].map(word => mixedCellWidth(word) === 1 ? 1 : 0));
  assert.ok(count(p.fine) > 0);
  assert.equal(p.diagnostics.reasons.region, count(p.fine));
});

test("support closure is the admitted tiles dilated one tile, reported apart from requests", () => {
  const centre = plan(new UniformDetailPlanner(lattice), [fine("one", 5, 6)], 0, { detailPolicy: "requested" });
  assert.equal(centre.diagnostics.admittedTiles, 1);
  assert.equal(centre.diagnostics.supportTiles, 26);
  assert.equal(centre.diagnostics.residentPatches, 1);
  assert.equal(centre.diagnostics.supportCells, 26 * 64);
  assert.equal(centre.diagnostics.wastedCells, 32 ** 3 - 27 * 64);
  const corner = plan(new UniformDetailPlanner(lattice), [fine("one", 0, 1)], 0, { detailPolicy: "requested" });
  assert.equal(corner.diagnostics.supportTiles, 7);
  // A tile on a patch corner: its closure touches all eight patches.
  const straddle = plan(new UniformDetailPlanner(lattice), [fine("one", 7, 8)], 0, { detailPolicy: "requested" });
  assert.equal(straddle.diagnostics.supportTiles, 26);
  assert.equal(straddle.diagnostics.residentPatches, 8);
  assert.deepEqual([...straddle.patches], [0, 1, 2, 3, 4, 5, 6, 7]);
});

test("manual requests are never budget-clipped; hard capacity defers a whole request", () => {
  const all = plan(new UniformDetailPlanner(lattice), [fine("big", 0, 12)], 0, { detailPolicy: "requested", detailBudgetPercent: 0 });
  assert.equal(all.diagnostics.admittedTiles, 12 ** 3);
  assert.equal(all.diagnostics.budgetClippedTiles, 0);
  const capped = plan(new UniformDetailPlanner(lattice), [fine("a", 1, 3), fine("b", 12, 14)], 0, { detailPolicy: "requested" }, { capacityPatches: 1 });
  assert.deepEqual(capped.deferred, ["region:b"]);
  assert.equal(capped.diagnostics.admittedTiles, 8);
  assert.equal(capped.diagnostics.deferredTiles, 8);
  assert.equal(capped.fine[tileKey(12, 12, 12)], 0);
});

test("expired requests drop by accepted step", () => {
  const activity: DetailRequest[] = [{ id: "splash", bounds_m: { min: { x: 0, y: 0, z: 0 }, max: { x: tile_m, y: tile_m, z: tile_m } },
    targetSpacing: "h", source: "activity", priority: 200, expiresAfterStep: 3 }];
  const s = settings({ detailPolicy: "dynamic" });
  assert.equal(buildUniformDetailRequests({ settings: s, regions: [], domain, acceptedStep: 3, activity }).requests.length, 1);
  assert.equal(buildUniformDetailRequests({ settings: s, regions: [], domain, acceptedStep: 4, activity }).requests.length, 0);
  assert.equal(buildUniformDetailRequests({ settings: settings({ detailPolicy: "dynamic", detailActivity: "off" }), regions: [], domain, acceptedStep: 0, activity }).requests.length, 0);
});

test("sensitivity scales the estimator thresholds by 2^(2-4s)", () => {
  assert.equal(uniformDetailSensitivityScale(0.5), 1);
  assert.equal(uniformDetailSensitivityScale(0), 4);
  assert.equal(uniformDetailSensitivityScale(1), 0.25);
});

test("the focus coalescer bumps its revision only when the target moves", () => {
  const focus = createDetailFocusCoalescer();
  const a = focus({ x: 0, y: 1, z: 2 });
  assert.equal(a?.revision, 1);
  assert.equal(focus({ x: 0, y: 1, z: 2 }), a);
  assert.equal(focus(undefined), a);
  assert.equal(focus({ x: 0, y: 1, z: 3 })?.revision, 2);
});

test("pool ids are world keys; slots carry generations; packed rank is not identity", () => {
  const pool = new UniformDetailPool({ domainPatches: 16, slotBytes: 1, byteBudget: 1000, readers: ["simulation", "render"] });
  const g1 = pool.prepare([9, 3]);
  pool.commit(g1);
  assert.equal(g1.capacity, 2);
  assert.ok(g1.resize);
  assert.equal(g1.directory[9], g1.promotions[0]!.slot);
  // Execution is world-key order: key 3 runs first though it was requested second.
  assert.deepEqual([...g1.execution], [g1.directory[3], g1.directory[9]]);
  assert.equal(g1.directory[0], UNIFORM_DETAIL_NO_SLOT);
  const before = pool.slotOf(9);
  const g2 = pool.prepare([9, 3, 12]);
  pool.commit(g2);
  // Geometric growth keeps live slot indices; the key still names the patch.
  assert.equal(pool.slotOf(9), before);
  assert.equal(g2.capacity, 4);
  assert.deepEqual(g2.resize?.copies.map(c => c.from === c.to), [true, true]);
  assert.equal(new Set(g2.slotGenerations).size, 4);
});

test("a retired slot is reused only after every reader finishes the generation that last read it", () => {
  const pool = new UniformDetailPool({ domainPatches: 64, slotBytes: 1, byteBudget: 1000, readers: ["simulation", "render"] });
  pool.commit(pool.prepare([1, 2, 3, 4]));
  // Generation 1 is the last to read key 4's slot; generation 2 retires it.
  const retired = pool.slotOf(4), generation = pool.acceptedGeneration;
  pool.commit(pool.prepare([1, 2, 3]));
  assert.equal(pool.capacity, 4);
  pool.acknowledge("simulation", generation);
  // One reader has not finished generation 1: the slot is not free, so a new key needs new storage.
  const blocked = pool.prepare([1, 2, 3, 40]);
  assert.ok(blocked.resize);
  assert.throws(() => pool.acknowledge("render", 99), /past accepted/);
  assert.throws(() => pool.acknowledge("compositor", 1), /Unknown/);
  pool.acknowledge("render", generation);
  const reused = pool.prepare([1, 2, 3, 40]);
  assert.equal(reused.resize, undefined);
  assert.equal(reused.directory[40], retired);
  assert.ok(reused.promotions[0]!.slotGeneration > 4);
  pool.commit(reused);
  assert.throws(() => pool.commit(blocked), /prepared on/);
});

test("pool growth plans the old+new peak within the byte budget and defers what does not fit", () => {
  const pool = new UniformDetailPool({ domainPatches: 64, slotBytes: 10, byteBudget: 50, readers: ["simulation"] });
  pool.commit(pool.prepare([0, 1, 2, 3]));
  assert.equal(pool.bytes, 40);
  const tx = pool.prepare([0, 1, 2, 3, 4, 5]);
  assert.equal(tx.resize, undefined);
  assert.deepEqual(tx.deferred, [4, 5]);
  assert.ok(tx.peakBytes <= 50);
  // Room for a resize: the old pool is held until its readers finish, then released.
  const roomy = new UniformDetailPool({ domainPatches: 64, slotBytes: 10, byteBudget: 200, readers: ["simulation"] });
  roomy.commit(roomy.prepare([0, 1, 2, 3]));
  const grow = roomy.prepare([0, 1, 2, 3, 4, 5]);
  assert.equal(grow.capacity, 8);
  assert.equal(grow.peakBytes, 120);
  roomy.commit(grow);
  assert.equal(roomy.residentBytes, 120);
  assert.deepEqual(roomy.takeReleasable(), []);
  roomy.acknowledge("simulation", 1);
  assert.deepEqual(roomy.takeReleasable(), [{ capacity: 4, bytes: 40 }]);
  assert.equal(roomy.residentBytes, 80);
  // No detail: the pool shrinks to nothing.
  roomy.commit(roomy.prepare([]));
  assert.equal(roomy.capacity, 0);
});

test("the retired coarsening control opens as Requested unless detailPolicy was saved", () => {
  for (const old of ["regions", "dynamic", "octree"]) {
    const migrated = migrateUniformDetailOverrides({ coarsening: old, pressureResidualTolerance: 5 });
    assert.deepEqual(migrated.overrides, { pressureResidualTolerance: 5 });
    assert.match(migrated.notice ?? "", /Requested detail/);
  }
  assert.deepEqual(migrateUniformDetailOverrides({ coarsening: "dynamic", detailPolicy: "full" }), { overrides: { detailPolicy: "full" } });
  assert.deepEqual(migrateUniformDetailOverrides({}), { overrides: {} });
  // Through the address: the old key is gone, the notice is said, and the
  // canonical write emits only the new keys.
  const old = parseMethodQueryState("?method=uniform-volume&param.uniform-volume.coarsening=dynamic");
  assert.equal(old.overrides["uniform-volume"], undefined);
  assert.match(old.methodNotice ?? "", /Coarsening: Dynamic/);
  const explicit = parseMethodQueryState("?method=uniform-volume&param.uniform-volume.coarsening=regions&param.uniform-volume.detailPolicy=full");
  assert.deepEqual(explicit.overrides["uniform-volume"], { detailPolicy: "full" });
  assert.equal(explicit.methodNotice, undefined);
  const base = parseQueryState("");
  const written = new URLSearchParams(serializeQueryState("?param.uniform-volume.coarsening=regions",
    { presetId: base.presetId, scene: base.scene }, { methodId: "uniform-volume", quality: "balanced", overrides: explicit.overrides }));
  assert.equal(written.get("param.uniform-volume.coarsening"), null);
  assert.equal(written.get("param.uniform-volume.detailPolicy"), "full");
  assert.equal(parseMethodQueryState(`?${written}`).overrides["uniform-volume"]?.detailPolicy, "full");
});

test("the app opens Requested, an explicit saved policy wins, and reset restores Requested", () => {
  const store = createMethodStore();
  store.getState().setMethodId("uniform-volume");
  assert.equal(resolvedMethodValues(store.getState()).detailPolicy, "requested");
  store.getState().setParam("uniform-volume", "detailPolicy", "dynamic");
  assert.equal(resolvedMethodValues(store.getState()).detailPolicy, "dynamic");
  store.getState().resetParam("uniform-volume", "detailPolicy");
  assert.equal(resolvedMethodValues(store.getState()).detailPolicy, "requested");
  // A retired key in saved overrides changes nothing it resolves to.
  store.getState().setParam("uniform-volume", "coarsening", "dynamic");
  assert.equal(resolvedMethodValues(store.getState()).detailPolicy, "requested");
});

test("detail controls are live: runtime keys, out of every structural key and the native parameter list", () => {
  const values = resolvedMethodValues({ methodId: "uniform-volume", quality: "balanced", overrides: {} });
  const structural = structuralMethodValues({ methodId: "uniform-volume", quality: "balanced", values });
  for (const key of UNIFORM_DETAIL_PARAM_KEYS) {
    assert.ok(uniformVolumeMethod.runtimeParamKeys?.includes(key), key);
    assert.ok(!(key in structural), key);
    assert.ok(!UNIFORM_GEOMETRIC_NATIVE_PARAMS.some(p => p.key === key), key);
    assert.ok(key in values, key);
  }
  assert.ok(!uniformVolumeMethod.params.some(p => p.key === "coarsening"));
  const composed = uniformVolumeMethod.resolveComposition!({ ...values, detailPolicy: "full" });
  assert.equal(composed.variants.find(v => v.point === "simulation.uniform-volume.algorithms.detailPolicy")?.update, "live");
  assert.throws(() => uniformVolumeMethod.resolveComposition!({ detailPolicy: "octree" }), /supported variant/);
  assert.equal(UNIFORM_DETAIL_REASON.region | UNIFORM_DETAIL_REASON.focus, 5);
});
