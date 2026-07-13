import assert from "node:assert/strict";
import test from "node:test";
import { decodeGPUStageTimings, planGPUAdvance } from "../lib/webgpu-eulerian";

test("GPU advance catches up more than one maxDt while bounding every substep", () => {
  assert.deepEqual(planGPUAdvance(1 / 60, 0.008, 0.02, 16), {elapsed_s:1/60,substeps:3,dt_s:1/180});
  assert.deepEqual(planGPUAdvance(0.2, 0.008, 0.02, 16), {elapsed_s:0.128,substeps:16,dt_s:0.008});
});

test("GPU advance respects a tighter stability limit", () => {
  const plan=planGPUAdvance(0.02,0.008,0.003,16);
  assert.equal(plan.substeps,7);
  assert.ok(plan.dt_s<=0.003);
});

test("GPU timings sum disjoint stage ranges across substeps", () => {
  const times = new BigUint64Array([0n,2_000_000n,2_000_000n,3_000_000n,3_000_000n,5_000_000n,5_000_000n,6_000_000n,6_000_000n,6_000_000n,7_000_000n,9_000_000n,9_000_000n,10_000_000n,10_000_000n,13_000_000n,13_000_000n,14_000_000n,14_000_000n,14_000_000n,15_000_000n,16_000_000n]);
  assert.deepEqual(decodeGPUStageTimings(times, 2), {advection_ms:4,control_ms:2,pressure_ms:5,projection_ms:2,rigidCoupling_ms:0,diagnostics_ms:1,overhead_ms:2,total_ms:16});
});

test("GPU timings clamp inverted timestamp pairs instead of bigint underflow", () => {
  const times = new BigUint64Array([10n,5n,10n,20n,20n,30n,30n,30n,30n,30n,10n,40n]);
  const timing = decodeGPUStageTimings(times, 1);
  assert.equal(timing?.advection_ms, 0);
  assert.ok((timing?.total_ms ?? 0) < 1);
});
