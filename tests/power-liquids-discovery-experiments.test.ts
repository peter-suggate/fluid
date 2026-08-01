import assert from "node:assert/strict";
import test from "node:test";
import { fineLevelSetReasonConesRequested } from
  "../lib/webgpu-octree-fine-levelset-topology";
import { analyzeCostAndChange } from "../tools/analyze-power-liquids-cost-change";
import {
  POWER_DAM_LANE_ENVIRONMENT,
  powerDamLaneWithDt,
} from "../tools/power-dam-lane-environment";
import { analyzeGPUCriticalPath } from "../tools/power-liquids-critical-path";
import type { GPUDataFlowManifest } from "../tools/webgpu-data-flow-manifest";
import { detectFrames } from "../tools/xctrace-frame-report";

test("X-1 hydrostatic lane uses the authored offset oracle and large-lane numerics", () => {
  const still = POWER_DAM_LANE_ENVIRONMENT.hydrostatic;
  const churn = POWER_DAM_LANE_ENVIRONMENT.large;
  assert.equal(still.FLUID_SCENE, "hydrostatic-power-large-offset");
  assert.equal(still.FLUID_MAX_DT, churn.FLUID_MAX_DT);
  assert.equal(still.FLUID_MAXIMUM_LEAF_SIZE, churn.FLUID_MAXIMUM_LEAF_SIZE);
  assert.equal(still.FLUID_OCTREE_INTERFACE_BAND, churn.FLUID_OCTREE_INTERFACE_BAND);
  assert.equal(still.FLUID_OCTREE_GLOBAL_FINE_FACTOR, churn.FLUID_OCTREE_GLOBAL_FINE_FACTOR);
  assert.equal(still.FLUID_ORACLE_STEPS, "240");
});

test("X-7 timestep overrides preserve simulated time and exact step count", () => {
  const environment = powerDamLaneWithDt("large", 0.001, 2);
  assert.equal(environment.FLUID_TARGET_S, "2");
  assert.equal(environment.FLUID_MAX_DT, "0.001");
  assert.equal(environment.FLUID_ORACLE_STEPS, "2000");
  assert.equal(environment.FLUID_EXPECT_EXACT_STEPS, "2000");
  assert.throws(() => powerDamLaneWithDt("large", 0.003, 2), /integer multiple/);
});

test("the sound reason-cone control is explicit and fails closed", () => {
  assert.equal(fineLevelSetReasonConesRequested({}), true);
  assert.equal(fineLevelSetReasonConesRequested({ FLUID_FINE_REASON_CONES: "0" }), false);
  assert.equal(fineLevelSetReasonConesRequested({ FLUID_FINE_REASON_CONES: "membership" }), true);
});

test("X-6 reconstructs the longest RAW/WAW dependency path", () => {
  const pass = (label: string, total_ms: number) => ({
    label, samples: 1, dispatches: 1, dispatchesPerAdvance: 1, total_ms,
    totalPerAdvance_ms: total_ms, uniqueBoundBytesUpperBound: 0,
    readBoundBytesUpperBound: 0, writableBoundBytesUpperBound: 0,
    authorityBufferIds: [], paths: [],
  });
  const manifest: GPUDataFlowManifest = {
    schemaVersion: 1, measuredAdvances: 1,
    limitations: { boundBytes: "binding-range-upper-bound",
      indirectLogicalCount: "gpu-authored-not-read-back" },
    buffers: [], passes: [pass("seed", 2), pass("dependent", 3), pass("independent", 7)],
    sequence: [
      { ordinal: 0, label: "seed", pipeline: "seed", entryPoint: "main",
        readBufferIds: [], writtenBufferIds: [1] },
      { ordinal: 1, label: "dependent", pipeline: "dependent", entryPoint: "main",
        readBufferIds: [1], writtenBufferIds: [2] },
      { ordinal: 2, label: "independent", pipeline: "independent", entryPoint: "main",
        readBufferIds: [3], writtenBufferIds: [4] },
    ],
  };
  const report = analyzeGPUCriticalPath(manifest, 10);
  assert.equal(report.work_msPerAdvance, 12);
  assert.equal(report.criticalPath_msPerAdvance, 7);
  assert.equal(report.criticalPathToWall, 0.7);
  assert.equal(report.decision, "inconclusive");
});

test("X-8 joins progress windows to generation census samples", () => {
  const records: Record<string, unknown>[] = [];
  for (let sample = 1; sample <= 4; sample += 1) records.push({
    phase: "octree-row-delta-census-sample", sample,
    dirty: sample * 2, added: sample, retired: 0,
    fine: { support: sample * 10, displacement: sample * 3 },
  });
  records.push({ record: "progress", steps: 4, windowSteps: 2, windowWallPerStep_ms: 9 });
  const report = analyzeCostAndChange(records);
  assert.deepEqual(report.points, [{ step: 4, wall_ms: 9, bandPages: 35,
    dirtyRows: 7, membershipChanges: 3.5, displacement: 10.5 }]);
});

test("X-8 removes bootstrap sample offset before joining progress windows", () => {
  const records: Record<string, unknown>[] = [
    { phase: "octree-row-delta-census-sample", sample: 1, dirty: 100,
      fine: { support: 100, displacement: 100 } },
    { phase: "octree-row-delta-census-sample", sample: 2, dirty: 100,
      fine: { support: 100, displacement: 100 } },
  ];
  for (let step = 1; step <= 4; step += 1) records.push({
    phase: "octree-row-delta-census-sample", sample: step + 2,
    dirty: step * 2, added: step, retired: 0,
    fine: { support: step * 10, displacement: step * 3 },
  });
  records.push({ record: "progress", steps: 4, windowSteps: 2, windowWallPerStep_ms: 9 });
  const report = analyzeCostAndChange(records);
  assert.deepEqual(report.points, [{ step: 4, wall_ms: 9, bandPages: 35,
    dirtyRows: 7, membershipChanges: 3.5, displacement: 10.5 }]);
});

test("X-8 treats unbounded bootstrap displacement as state, not a count", () => {
  const records: Record<string, unknown>[] = [
    { phase: "octree-row-delta-census-sample", sample: 1, dirty: 1, added: 0, retired: 0,
      fine: { support: 10, displacement: 0xffff_ffff } },
    { phase: "octree-row-delta-census-sample", sample: 2, dirty: 1, added: 0, retired: 0,
      fine: { support: 10, displacement: 2 } },
    { record: "progress", steps: 2, windowSteps: 2, windowWallPerStep_ms: 3 },
  ];
  assert.equal(analyzeCostAndChange(records).points[0]?.displacement, 2);
});

test("X-1 trace reduction keeps a semantic boundary clipped by one metadata edge", () => {
  const interval = (start: number, label: string) => ({ start, duration: 1, label,
    encoders: [label], channel: "Compute", merged: false });
  const intervals = Array.from({ length: 7 }, (_, frame) =>
    interval(frame * 100, "Open coupled topology ready-commit gate"));
  for (const label of ["interior-a", "interior-b", "interior-c"]) {
    intervals.push(...Array.from({ length: 6 }, (_, frame) => interval(frame * 100 + 20, label)));
  }
  const detected = detectFrames(intervals);
  assert.equal(detected.anchor, "Open coupled topology ready-commit gate");
  assert.equal(detected.boundaries.length, 7);
});
