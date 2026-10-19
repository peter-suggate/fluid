import assert from "node:assert/strict";
import test from "node:test";
import { fitCost, SpatialCostModel, type SpatialWorkCell } from "../../core/spatial-performance";
import { performanceViewRecords, DEFAULT_PERFORMANCE_VIEW, readPerformanceView } from "./view";
import { decodeUniformSpatialWork, SurfaceWorkCompiler } from "../../methods/uniform/uniform-spatial-performance";
import { decodeGPUTimestampPartition } from "../../core/performance-trace";
const plugin = { id: "test", revision: 1, stages: [{ id: "transport", label: "Transport", phases: ["advect"], features: ["active"] }, { id: "pressure", label: "Pressure", phases: ["pressure"], features: ["liquid"] }] };
const cells: SpatialWorkCell[] = [
  { origin: [0, 0, 0], size: [4, 4, 4], owners: 64, region: 0, tier: 2, features: { active: 64, liquid: 1 } },
  { origin: [4, 0, 0], size: [4, 4, 4], owners: 1, region: 0, tier: 0, features: { active: 0, liquid: 1 } },
  { origin: [16, 0, 0], size: [4, 4, 4], owners: 1, region: 1, tier: 1, features: { active: 1, liquid: 0 } },
];
test("spatial accounting closes actual wall time, excludes overlap, retains unmodelled time", () => {
  const s = new SpatialCostModel(plugin).account(cells, { frame: 8, wall_ms: 10, phases: [{ label: "advect", duration_ms: 4 }, { label: "pressure", duration_ms: 2 }, { label: "unmodelled", duration_ms: 1 }] });
  assert.equal(s.report.attributed_ms + s.report.shared_ms, 10);
  assert.equal(s.report.shared_ms, 4);
  assert.equal(s.costs[1]![1], 0, "cold transport has no active work");
  assert.ok(s.costs[2]![1]! > 0, "cold pressure is not free");
  assert.equal(s.costs[2]![2], 0);
  const overlap = new SpatialCostModel(plugin).account(cells, { frame: 9, wall_ms: 3, phases: [{ label: "advect", duration_ms: 4 }, { label: "pressure", duration_ms: 2 }] });
  assert.equal(overlap.report.attributed_ms, 3);
});
test("fit separates a fixed overhead from work and tests unseen observations", () => {
  const samples = Array.from({ length: 60 }, (_, i) => ({ x: [i * 10], y: 2 + i * 0.3 }));
  const fit = fitCost(samples, 1);
  assert.ok(Math.abs(fit.overhead - 2) < 0.01);
  assert.ok(Math.abs(fit.coefficients[0]! - 0.03) < 0.001);
  assert.ok(fit.validated && fit.validationError < 0.01);
  const adversarial = samples.map((s, i) => i % 5 === 4 ? { ...s, y: s.y * 3 } : s);
  assert.equal(fitCost(adversarial, 1).validated, false);
  assert.equal(fitCost(Array.from({ length: 60 }, () => ({ x: [10], y: 2 })), 1).validated, false, "constant work cannot identify variable cost");
});
test("aggregation preserves contribution and cell division never multiplies a tile's time", () => {
  const snapshot = new SpatialCostModel(plugin).account(cells, { frame: 1, wall_ms: 8, phases: [{ label: "advect", duration_ms: 4 }, { label: "pressure", duration_ms: 2 }] });
  for (const grain of ["cell", "tile", "region"] as const) {
    const data = performanceViewRecords(snapshot, { ...DEFAULT_PERFORMANCE_VIEW, grain });
    let sum = 0;for (let i = 0; i < data.count; i++) sum += data.records[12 * i + 8]!;
    assert.ok(Math.abs(sum - 6) < 1e-5);
    assert.equal(data.entities, grain === "cell" ? 66 : grain === "tile" ? 3 : 2);
  }
  assert.equal(performanceViewRecords(snapshot, { ...DEFAULT_PERFORMANCE_VIEW, stage: "missing" }).count, 0);
});
test("decode uses physical block octants, skips free/sentinel blocks and translates signed windows", () => {
  const tiles = new Uint32Array(4 * 512);tiles[512 + 4] = 0x80000000;
  const tiers = new Uint32Array(32);tiers[8] = 0;tiers[9] = 1;tiers[10] = 2;
  const block = (63 | (1 << 6) | (2 << 12));
  const data = decodeUniformSpatialWork({ tiles, tiers, particles: new Uint32Array(tiles.length), blocks: new Uint32Array([0xffffffff, block, 0xffffffff, 0xffffffff]), framePage: new Int32Array([-2, 0, 1]), origin: [-64, 0, 32], h: [1, 1, 1] });
  assert.equal(data.length, 512);assert.deepEqual(data[0]!.origin, [-32, 32, 64]);
  assert.equal(data[4]!.tier, 1);assert.equal(data[4]!.owners, 64);assert.equal(data[4 * 8]!.tier, 2);
});
test("bad timings and overlapping stage declarations fail closed", () => {
  assert.throws(() => new SpatialCostModel({ ...plugin, stages: [plugin.stages[0]!, plugin.stages[0]!] }));
  assert.throws(() => new SpatialCostModel(plugin).account(cells, { frame: 0, wall_ms: NaN, phases: [] }));
  assert.deepEqual(readPerformanceView({ grain: "bad", dimension: "nonsense", stage: 7 }), DEFAULT_PERFORMANCE_VIEW);
});

test("compiled predictions do not train on themselves and reject unseen work", () => {
  const model = new SpatialCostModel({ id: "linear", revision: 2, stages: [plugin.stages[0]!] });
  for (let i = 0; i < 60; i++) model.account([{ ...cells[0]!, features: { active: i * 10 } }], { frame: i, wall_ms: 30, phases: [{ label: "advect", duration_ms: 2 + i * .3 }] });
  const current = [{ ...cells[0]!, features: { active: 300 } }];
  assert.equal(model.canPredict(current), true);
  assert.equal(model.canPredict([{ ...cells[0]!, features: { active: 6000 } }]), false);
  const predicted = model.predict(current, 61, 20);
  assert.equal(predicted.report.evidence, "compiled-model");
  assert.equal(predicted.report.samples, 60);
  assert.ok(Math.abs(predicted.report.stages[0]!.total_ms - 11) < .02);
  assert.ok(Math.abs(predicted.report.attributed_ms - 9) < .02, "fixed overhead stays shared");
  assert.equal(model.profile().revision, 2);
});
test("spatial evidence rejects non-finite features and invalid geometry", () => {
  const timing = { frame: 1, wall_ms: 10, phases: [] };
  assert.throws(() => new SpatialCostModel(plugin).account([{ ...cells[0]!, features: { active: NaN } }], timing));
  assert.throws(() => new SpatialCostModel(plugin).account([{ ...cells[0]!, size: [0, 1, 1] }], timing));
});
test("surface mode attribution conserves measured regional stores including cold identity work", () => {
  const compiler = new SurfaceWorkCompiler();
  const group: SpatialWorkCell[] = cells.slice(0, 2).map(c => ({ ...c, tier: 0 as const, features: { ...c.features } }));
  compiler.apply(group, new Uint32Array([130, 0, 0, 0, 0]));
  assert.equal(group[0]!.features.identityVertices, 128);
  assert.equal(group[1]!.features.identityVertices, 2);
});
test("skipped GPU segments accept equal observed boundaries but never missing timestamps", () => {
  const base = { sampleId: 1, lane: "physics" as const, context: "cold", phases: [{ id: "other" as const, label: "skipped" }] };
  assert.equal(decodeGPUTimestampPartition({ ...base, timestamps: [10n, 10n] }), undefined);
  const empty = decodeGPUTimestampPartition({ ...base, timestamps: [10n, 10n], allowZeroDuration: true });
  assert.equal(empty?.total_ms, 0);
  assert.equal(decodeGPUTimestampPartition({ ...base, timestamps: [0n, 0n], allowZeroDuration: true }), undefined);
});

test("timestamp partition roundoff is harmless but negative measured work is rejected", () => {
  const model = new SpatialCostModel(plugin);
  assert.equal(model.account(cells, { frame: 1, wall_ms: 10, phases: [{ label: "advect", duration_ms: 4 }, { label: "Other measured work", duration_ms: -1e-15 }] }).report.stage_ms, 4);
  assert.throws(() => model.account(cells, { frame: 2, wall_ms: 10, phases: [{ label: "advect", duration_ms: -.001 }] }));
});

test("a fitted intercept without a zero-work reference does not erase spatial contribution", () => {
  const model = new SpatialCostModel({ ...plugin, stages: [plugin.stages[0]!] });
  let result;
  for (let i = 0; i < 60; i++) result = model.account([{ ...cells[0]!, features: { active: 100 + i * 10 } }], { frame: i, wall_ms: 30, phases: [{ label: "advect", duration_ms: 5 + i * .1 }] });
  assert.equal(model.profile().stages.transport!.overheadIdentified, false);
  assert.equal(result!.report.shared_ms, 30 - 10.9);
});
