import assert from "node:assert/strict";
import test from "node:test";

import {
  requireNativeCellwiseStageTimings,
  requireNativeWorldStageTimings,
  summarizeNativeStageTimings,
} from "../tools/native-cellwise-stage-timings";

const worldTiming = (frameTotal: number) => ({
  available: true,
  totalAdvance: frameTotal,
  fieldBuild: 1,
  primaryPressure: 2,
  supportPlanningTransfer: 3,
  postSupportPressure: 4,
  transport: 12,
  postTransport: 5,
  resolutionPublication: 6,
  other: frameTotal - 33,
});
const remapTiming = (totalRemap = 10) => ({
  available: true,
  baseFieldBuild: 1,
  receiverBandClosure: 1,
  streamfunctionExtension: 1,
  harmonicFill: 1,
  diagnostics: 1,
  trace: 1,
  geometryRefinement: 1,
  gather: 1,
  commit: 1,
  other: 1,
  totalRemap,
});

test("native timing receipt requires every common and remap stage", () => {
  const world = requireNativeWorldStageTimings({ stageTimings: worldTiming(40) }, 1);
  const remap = requireNativeCellwiseStageTimings({ timings: remapTiming() }, 1);
  const summary = summarizeNativeStageTimings([world], [remap]) as any;
  assert.equal(summary.world.totalAdvance.meanNanoseconds, 40);
  assert.equal(summary.cellwiseRemap.harmonicFill.meanNanoseconds, 1);
});

test("baseline timing remains comparable without a remap split", () => {
  const first = requireNativeWorldStageTimings({ stageTimings: worldTiming(40) }, 1);
  const second = requireNativeWorldStageTimings({ stageTimings: worldTiming(50) }, 2);
  const summary = summarizeNativeStageTimings([first, second]) as any;
  assert.equal(summary.cellwiseRemap, null);
  assert.equal(summary.world.totalAdvance.meanNanoseconds, 45);
  assert.equal(summary.world.totalAdvance.p95Nanoseconds, 50);
});

test("missing, unavailable, or overlapping remap timings fail closed", () => {
  assert.throws(() => requireNativeWorldStageTimings({ stageTimings: {
    ...worldTiming(40),
    available: false,
  } }, 1), /unavailable/);
  const missing = remapTiming() as Record<string, unknown>;
  delete missing.trace;
  assert.throws(() => requireNativeCellwiseStageTimings({ timings: missing }, 1),
    /missing cellwiseRemap\.trace/);
  assert.throws(() => requireNativeCellwiseStageTimings({
    timings: remapTiming(11),
  }, 1), /timing split 10 != totalRemap 11/);
});
