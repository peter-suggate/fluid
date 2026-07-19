import assert from "node:assert/strict";
import test from "node:test";

import { SVO_BASELINE_CASES } from "../tools/svo-baseline-cases";
import { SVO_BASELINE_REQUIRED_LIMITS, type SVOBaselineAdapterObservation } from "../tools/svo-baseline-contract";
import {
  SVO_BENCHMARK_SCHEMA_VERSION,
  aggregateSVOBenchmarkObservations,
  buildSVOBenchmarkPlan,
  type SVOBenchmarkFrameObservation,
  type SVOBenchmarkObservationBundle,
  type SVOBenchmarkPlan,
  type SVOBenchmarkRunObservation,
} from "../tools/svo-benchmark-contract";

const baseline = SVO_BASELINE_CASES[0];
const adapter: SVOBaselineAdapterObservation = {
  name: "Apple M3 Max",
  vendor: "Apple",
  backend: "metal",
  features: ["timestamp-query"],
  limits: { ...SVO_BASELINE_REQUIRED_LIMITS },
};

function benchmarkPlan(pairCount = 2): SVOBenchmarkPlan {
  return buildSVOBenchmarkPlan({
    revision: "deadbeef",
    adapterId: "apple-m3-max-metal",
    resetToken: "capture-session-7",
    captureNotBeforeUnixMs: 1_000_000,
    pairCount,
    cases: [baseline],
    internalResolution: {
      raster: { width: 1280, height: 720 },
      svo: { width: 960, height: 540 },
    },
  });
}

function observationForPlan(plan: SVOBenchmarkPlan): SVOBenchmarkObservationBundle {
  const runs: SVOBenchmarkRunObservation[] = plan.runs.map((run) => ({
    runId: run.id,
    sequenceIndex: run.sequenceIndex,
    revision: run.revision,
    adapterId: run.adapterId,
    baselineCanonical: run.baselineCanonical,
    quality: run.quality,
    outputResolution: run.outputResolution,
    internalResolution: run.internalResolution,
    resetToken: run.resetToken,
    adapter,
    equivalence: {
      sceneStateIdentity: "scene-sha256:abc",
      cameraStateIdentity: "camera-sha256:def",
      simulationStateIdentity: "solver-sha256:123",
      simulatedTime_s: baseline.checkpoint.simulatedTime_s,
      stepCount: baseline.checkpoint.stepCount,
    },
    frames: Array.from({ length: run.warmupFrames + run.measuredFrames }, (_, frameIndex): SVOBenchmarkFrameObservation => ({
      frameIndex,
      sampledAtUnixMs: run.captureNotBeforeUnixMs + frameIndex,
      resetToken: run.resetToken,
      requestedMode: run.requestedMode,
      effectiveMode: run.requestedMode,
      fallbackReason: null,
      gpuRenderTimingAvailable: true,
      cpuFrame_ms: frameIndex,
      gpuRender_ms: frameIndex / 10,
      gpuDryScene_ms: frameIndex / 20,
      rendererOwnedBytes: 1_000 + frameIndex,
    })),
  }));
  return { schemaVersion: SVO_BENCHMARK_SCHEMA_VERSION, runs };
}

function replaceRun(
  bundle: SVOBenchmarkObservationBundle,
  index: number,
  replacement: SVOBenchmarkRunObservation,
): SVOBenchmarkObservationBundle {
  const runs = [...bundle.runs];
  runs[index] = replacement;
  return { ...bundle, runs };
}

test("benchmark plan alternates raster/SVO order and gives every run a distinct reset identity", () => {
  const plan = benchmarkPlan();
  assert.deepEqual(plan.runs.map(({ renderer }) => renderer), ["raster", "svo", "svo", "raster"]);
  assert.deepEqual(plan.runs.map(({ sequenceIndex }) => sequenceIndex), [0, 1, 2, 3]);
  assert.equal(new Set(plan.runs.map(({ resetToken }) => resetToken)).size, plan.runs.length);
  assert.ok(plan.runs.every(({ warmupFrames, measuredFrames }) => warmupFrames === 30 && measuredFrames === 120));
  assert.match(plan.captureInstructions[0], /external/);
  assert.match(plan.captureInstructions[0], /does not claim browser automation/);
});

test("aggregation excludes warmups and retains p50/p95/max plus every raw frame", () => {
  const plan = benchmarkPlan(1);
  const report = aggregateSVOBenchmarkObservations(plan, observationForPlan(plan), [baseline]);
  assert.equal(report.runs.length, 2);
  assert.deepEqual(report.runs[0].cpuFrame_ms, { p50: 89.5, p95: 143.05, maximum: 149 });
  assert.ok(Math.abs(report.runs[0].gpuRender_ms!.p50 - 8.95) < 1e-12);
  assert.ok(Math.abs(report.runs[0].gpuDryScene_ms!.p95 - 7.1525) < 1e-12);
  assert.deepEqual(report.runs[0].rendererOwnedBytes, { p50: 1089.5, p95: 1143.05, maximum: 1149 });
  assert.equal(report.runs[0].rawFrames.length, 150);
  assert.equal(report.runs[0].timestampQueriesAvailable, true);
  assert.equal(report.runs[0].effectiveMode, "raster");
  assert.equal(report.runs[1].effectiveMode, "svo");
  assert.equal(report.pairs[0].equivalenceValidated, true);
  assert.equal(report.pairs[0].gpuRenderP95RatioSvoToRaster, 1);
  assert.equal(report.aggregates.length, 2);
  assert.deepEqual(report.aggregates.map(({ renderer }) => renderer), ["raster", "svo"]);
  assert.equal(report.aggregates[0].runIds.length, 1);
  assert.deepEqual(report.aggregates[0].gpuRender_ms, report.runs[0].gpuRender_ms);
});

test("stale reset tokens, pre-reset frames, discontinuities, and missing samples fail clearly", () => {
  const plan = benchmarkPlan(1);
  const original = observationForPlan(plan);
  const run = original.runs[0];
  assert.throws(() => aggregateSVOBenchmarkObservations(plan, replaceRun(original, 0, {
    ...run, frames: [{ ...run.frames[0], resetToken: "old-session" }, ...run.frames.slice(1)],
  }), [baseline]), /stale reset token/);
  assert.throws(() => aggregateSVOBenchmarkObservations(plan, replaceRun(original, 0, {
    ...run, frames: [{ ...run.frames[0], sampledAtUnixMs: plan.captureNotBeforeUnixMs - 1 }, ...run.frames.slice(1)],
  }), [baseline]), /predates the reset/);
  assert.throws(() => aggregateSVOBenchmarkObservations(plan, replaceRun(original, 0, {
    ...run, frames: run.frames.map((frame, index) => index === 40 ? { ...frame, frameIndex: 39 } : frame),
  }), [baseline]), /stale or discontinuous/);
  assert.throws(() => aggregateSVOBenchmarkObservations(plan, replaceRun(original, 0, {
    ...run, frames: run.frames.slice(0, -1),
  }), [baseline]), /exactly 150 are required/);
});

test("revision, renderer fallback, and raster/SVO state inequivalence are rejected", () => {
  const plan = benchmarkPlan(1);
  const original = observationForPlan(plan);
  assert.throws(() => aggregateSVOBenchmarkObservations(plan, replaceRun(original, 0, {
    ...original.runs[0], revision: "stale-revision",
  }), [baseline]), /stale revision/);
  assert.throws(() => aggregateSVOBenchmarkObservations(plan, replaceRun(original, 1, {
    ...original.runs[1], frames: original.runs[1].frames.map((frame, index) => index === 50
      ? { ...frame, effectiveMode: "raster", fallbackReason: "missing-structural-source" }
      : frame),
  }), [baseline]), /effective renderer\/fallback mismatch/);
  assert.throws(() => aggregateSVOBenchmarkObservations(plan, replaceRun(original, 1, {
    ...original.runs[1], equivalence: { ...original.runs[1].equivalence, simulationStateIdentity: "different-solver-state" },
  }), [baseline]), /renderer-equivalence mismatch/);
});

test("timestamp unavailability is explicit, while mixed measured availability is invalid", () => {
  const plan = benchmarkPlan(1);
  const original = observationForPlan(plan);
  const unavailable = {
    ...original.runs[0],
    frames: original.runs[0].frames.map((frame) => ({
      ...frame,
      gpuRenderTimingAvailable: false,
      gpuRender_ms: null,
      gpuDryScene_ms: null,
    })),
  };
  const report = aggregateSVOBenchmarkObservations(plan, replaceRun(original, 0, unavailable), [baseline]);
  assert.equal(report.runs[0].timestampQueriesAvailable, false);
  assert.equal(report.runs[0].gpuRender_ms, null);
  assert.equal(report.runs[0].gpuDryScene_ms, null);

  const mixedFrames = unavailable.frames.map((frame, index) => index === plan.runs[0].warmupFrames
    ? { ...frame, gpuRenderTimingAvailable: true, gpuRender_ms: 1, gpuDryScene_ms: .5 }
    : frame);
  assert.throws(() => aggregateSVOBenchmarkObservations(plan, replaceRun(original, 0, {
    ...unavailable, frames: mixedFrames,
  }), [baseline]), /timestamp availability changed/);
});
