import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
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
    frames: Array.from({ length: run.warmupFrames + run.measuredFrames }, (_, frameIndex): SVOBenchmarkFrameObservation => {
      const context = `octree:${run.quality}:smooth:${run.requestedMode}:running`;
      return {
        frameIndex,
        sampledAtUnixMs: run.captureNotBeforeUnixMs + frameIndex,
        resetToken: run.resetToken,
        requestedMode: run.requestedMode,
        effectiveMode: run.requestedMode,
        fallbackReason: null,
        performance: {
          methodId: "octree",
          context,
          capturedAt_ms: frameIndex,
          cpu: {
            sampleId: run.sequenceIndex * 1_000 + frameIndex + 1,
            domain: "cpu",
            lane: "main-thread",
            context,
            capturedAt_ms: frameIndex,
            total_ms: frameIndex,
            phases: [{ id: "frame-control", label: "Frame control", duration_ms: frameIndex }],
          },
          presentation: {
            sampleId: run.sequenceIndex * 1_000 + frameIndex + 1,
            domain: "gpu",
            lane: "presentation",
            context,
            capturedAt_ms: frameIndex,
            total_ms: frameIndex / 10,
            phases: [
              { id: "dry-scene", label: "Dry scene", duration_ms: frameIndex / 20 },
              { id: "present", label: "Present", duration_ms: frameIndex / 20 },
            ],
          },
        },
        rendererOwnedBytes: 1_000 + frameIndex,
      };
    }),
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
  assert.deepEqual(report.runs[0].cpuTotal_ms, { p50: 89.5, p95: 143.05, maximum: 149 });
  assert.ok(Math.abs(report.runs[0].presentationTotal_ms!.p50 - 8.95) < 1e-12);
  assert.ok(Math.abs(report.runs[0].dryScene_ms!.p95 - 7.1525) < 1e-12);
  assert.deepEqual(report.runs[0].rendererOwnedBytes, { p50: 1089.5, p95: 1143.05, maximum: 1149 });
  assert.equal(report.runs[0].rawFrames.length, 150);
  assert.equal(report.runs[0].presentationTracesAvailable, true);
  assert.equal(report.runs[0].effectiveMode, "raster");
  assert.equal(report.runs[1].effectiveMode, "svo");
  assert.equal(report.pairs[0].equivalenceValidated, true);
  assert.equal(report.pairs[0].presentationTotalP95RatioSvoToRaster, 1);
  assert.equal(report.aggregates.length, 2);
  assert.deepEqual(report.aggregates.map(({ renderer }) => renderer), ["raster", "svo"]);
  assert.equal(report.aggregates[0].runIds.length, 1);
  assert.deepEqual(report.aggregates[0].presentationTotal_ms, report.runs[0].presentationTotal_ms);
  assert.equal(report.aggregates[0].adapterId, "apple-m3-max-metal");
  assert.deepEqual(report.aggregates[0].adapter, adapter);
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

test("presentation-trace unavailability is explicit, while mixed measured availability is invalid", () => {
  const plan = benchmarkPlan(1);
  const original = observationForPlan(plan);
  const unavailable = {
    ...original.runs[0],
    frames: original.runs[0].frames.map((frame) => ({
      ...frame,
      performance: { ...frame.performance, presentation: undefined },
    })),
  };
  const report = aggregateSVOBenchmarkObservations(plan, replaceRun(original, 0, unavailable), [baseline]);
  assert.equal(report.runs[0].presentationTracesAvailable, false);
  assert.equal(report.runs[0].presentationTotal_ms, null);
  assert.equal(report.runs[0].dryScene_ms, null);

  const mixedFrames = unavailable.frames.map((frame, index) => index === plan.runs[0].warmupFrames
    ? { ...frame, performance: { ...frame.performance, presentation: original.runs[0].frames[index].performance.presentation } }
    : frame);
  assert.throws(() => aggregateSVOBenchmarkObservations(plan, replaceRun(original, 0, {
    ...unavailable, frames: mixedFrames,
  }), [baseline]), /presentation trace availability changed/);
});

test("cached presentation traces and performance-context drift are rejected", () => {
  const plan = benchmarkPlan(1);
  const original = observationForPlan(plan);
  const run = original.runs[0];
  assert.throws(() => aggregateSVOBenchmarkObservations(plan, replaceRun(original, 0, {
    ...run,
    frames: run.frames.map((frame, index) => index === 40
      ? { ...frame, performance: {
        ...frame.performance,
        presentation: { ...frame.performance.presentation!, sampleId: run.frames[39].performance.presentation!.sampleId },
      } }
      : frame),
  }), [baseline]), /repeats a cached presentation trace/);
  assert.throws(() => aggregateSVOBenchmarkObservations(plan, replaceRun(original, 0, {
    ...run,
    frames: run.frames.map((frame, index) => index === 40
      ? { ...frame, performance: { ...frame.performance, context: `${frame.performance.context}:changed` } }
      : frame),
  }), [baseline]), /performance context changed/);
});

test("GPU traversal benchmark compares the clean expansion cutover and retains isolated Morton decode", () => {
  const source = readFileSync(new URL("../tools/benchmark-svo-traversal-gpu.ts", import.meta.url), "utf8");
  assert.match(source, /FLUID_SVO_TRAVERSAL_COMPARISON \?\? "expansion"/,
    "the current candidate-expansion implementation must be the default comparison");
  assert.match(source, /comparison ".*" was retired:[^]*pre-carried-bounds traversal/,
    "the benchmark must reject comparisons backed by deleted traversal implementations");
  assert.match(source, /return expansionBaselineTraversalWGSL\(\)/,
    "the default must isolate candidate expansion without restoring legacy traversal");
  assert.match(source, /comparison === "morton-decode"[\s\S]*mortonDecodeBaselineTraversalWGSL/,
    "the opt-in baseline must replace only the production Morton decoder");
  assert.match(source, /decodeSample < \$\{amplifiedMortonDecodesPerTraversal\}u[\s\S]*svoDecodeMorton\(keyLow, keyHigh, 21u\)/,
    "the isolated mode must amplify deepest supported decodes");
  assert.match(source, /nodeHash = \(\(nodeHash \* 16777619u\) \^ decoded\.x\)/,
    "decoded coordinates must participate in the bit-equivalence result");
  assert.match(source, /phase: "svo-traversal-gpu-benchmark",\s*comparison,/,
    "machine-readable output must identify the selected comparison");
  assert.doesNotMatch(source, /carriedBoundsBaselineTraversalWGSL|parentBoundsBaselineTraversalWGSL/,
    "deleted traversal implementations must not survive as benchmark-only legacy code");
  assert.match(source, /FLUID_SVO_TRAVERSAL_FIXTURE \?\? "dense"/,
    "the historical dense fixture must remain the default while allowing a deep sparse chain");
  assert.match(source, /baselineStackEntryBytes:[\s\S]*optimizedStackEntryBytes:/,
    "the benchmark must report the local stack footprint tradeoff");
});
