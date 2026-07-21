import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { decodeRenderStageTimestamps, hasResolvedRenderTimestampSample, RENDER_TIMESTAMP_QUERY_COUNT } from "../lib/webgpu-renderer";

const timestamps = [
  0n, 2_000_000n,
  3_000_000n, 6_000_000n,
  7_000_000n, 11_000_000n,
  12_000_000n, 17_000_000n,
  18_000_000n, 24_000_000n,
  25_000_000n, 32_000_000n,
  33_000_000n, 35_000_000n,
  36_000_000n, 39_000_000n
];

test("raster timestamps retain each presentation stage", () => {
  assert.deepEqual(decodeRenderStageTimestamps(timestamps, true), {
    total_ms: 32,
    surfaceExtraction_ms: 2,
    dryScene_ms: 3,
    interfaceFront_ms: 4,
    interfaceBack_ms: 5,
    interfaces_ms: 9,
    sprayFront_ms: 2,
    sprayBack_ms: 3,
    sprayRender_ms: 5,
    opticalComposite_ms: 6,
    upscale_ms: 7
  });
});

test("unchanged surfaces contribute no stale extraction time", () => {
  const result = decodeRenderStageTimestamps(timestamps, false);
  assert.equal(result.surfaceExtraction_ms, undefined);
  assert.equal(result.total_ms, 30);
});

test("disabled spray contributes no stale render time", () => {
  const result = decodeRenderStageTimestamps(timestamps, true, false);
  assert.equal(result.sprayFront_ms, 0);
  assert.equal(result.sprayBack_ms, 0);
  assert.equal(result.sprayRender_ms, 0);
  assert.equal(result.total_ms, 27);
});

test("SVO temporal resolve has an independent timestamp interval", () => {
  const withTemporal = [...timestamps, 40_000_000n, 44_000_000n];
  assert.equal(RENDER_TIMESTAMP_QUERY_COUNT, 22);
  const result = decodeRenderStageTimestamps(withTemporal, true, true, true);
  assert.equal(result.dryScene_ms, 3);
  assert.equal(result.svoTemporal_ms, 4);
  assert.equal(result.total_ms, 36);
  assert.equal(decodeRenderStageTimestamps(withTemporal, true, true, false).svoTemporal_ms, undefined,
    "raster frames must never report stale values from the reserved SVO interval");
});

test("caustics and optional inspection overlays contribute to the complete presentation total", () => {
  const complete = [
    ...timestamps,
    40_000_000n, 44_000_000n,
    45_000_000n, 47_000_000n,
    48_000_000n, 51_000_000n,
  ];
  const result = decodeRenderStageTimestamps(complete, true, true, true, true, true);
  assert.equal(result.caustics_ms, 2);
  assert.equal(result.overlays_ms, 3);
  assert.equal(result.total_ms, 41);
  const inactive = decodeRenderStageTimestamps(complete, true, true, true, false, false);
  assert.equal(inactive.caustics_ms, undefined);
  assert.equal(inactive.overlays_ms, undefined);
  assert.equal(inactive.total_ms, 36);
});

test("an all-zero query resolve is unavailable rather than a measured zero-time frame", () => {
  const unresolved = decodeRenderStageTimestamps(Array<bigint>(RENDER_TIMESTAMP_QUERY_COUNT).fill(0n), false, false, false);
  assert.equal(unresolved.total_ms, 0);
  assert.equal(hasResolvedRenderTimestampSample(unresolved), false);
  assert.equal(hasResolvedRenderTimestampSample({ ...unresolved, upscale_ms: 0.000001, total_ms: 0.000001 }), true,
    "positive sub-millisecond samples must remain exact rather than being rounded away");
});

test("renderer-mode epochs reject stale asynchronous timing readbacks", () => {
  const renderer = readFileSync(new URL("../lib/webgpu-renderer.ts", import.meta.url), "utf8");
  const controller = readFileSync(new URL("../lib/simulation/controller.ts", import.meta.url), "utf8");
  const panel = readFileSync(new URL("../components/PerformancePanel.tsx", import.meta.url), "utf8");
  assert.match(renderer, /timingContext = `\$\{config\.methodId\}:\$\{config\.quality\}:shadow-\$\{[^}]+\}:ao-\$\{[^}]+\}:temporal-\$\{[^}]+\}:lighting-\$\{svoLightingMode\}:\$\{voxelRenderMode\}:\$\{svoRenderMode\}`/);
  assert.match(renderer, /beginRenderTimingEpoch\(\)[^]*this\.renderTimingEpoch \+= 1/);
  assert.match(renderer, /beginRenderTimingEpoch\(\)[^]*this\.lastRenderQueryAt = -Infinity;[^]*this\.gpuRender_ms = undefined/,
    "a mode epoch must bypass the 250 ms cadence and sample its first presentation");
  assert.match(renderer, /setSimulationRunning\(running: boolean\)[^]*running !== this\.simulationRunning[^]*this\.beginRenderTimingEpoch\(\)/,
    "play/pause transitions must reject in-flight samples encoded under the previous presentation semantics");
  assert.match(renderer, /if \(this\.presentationPending\) return \{[^}]*\.\.\.this\.currentRenderTimingMetrics\(config\.methodId\)/,
    "a non-submitting paced callback must not replace the live renderer epoch with a synthetic legacy context");
  assert.match(renderer, /sampledEpoch=this\.renderTimingEpoch/);
  assert.match(renderer, /this\.renderTimingContext===sampledContext&&this\.renderTimingEpoch===sampledEpoch/);
  assert.match(renderer, /this\.renderTimingSampleId\+=1/,
    "accepted readbacks need a monotonic identity so capture never duplicates cached telemetry");
  assert.match(renderer, /currentTimingEpoch&&hasResolvedRenderTimestampSample\(stage\)/,
    "an all-zero query resolve must not replace a valid timing sample");
  assert.match(renderer, /pausedTimingRetryEpoch!==sampledEpoch[^]*this\.lastRenderQueryAt=-Infinity[^]*this\.pausedPresentationRevision\+=1/,
    "paused all-zero resolves get one current-epoch retry without an unbounded redraw loop");
  assert.match(renderer, /!this\.simulationRunning&&this\.pausedTimingNotificationEpoch!==sampledEpoch[^]*this\.pausedPresentationRevision\+=1/,
    "a paused accepted readback must request one frame that publishes its asynchronous metrics");
  assert.match(renderer, /this\.renderTimingEpoch!==sampledEpoch[^]*this\.lastRenderQueryAt=-Infinity[^]*this\.pausedPresentationRevision\+=1/,
    "a stale in-flight readback must request one current-epoch retry in paused mode");
  assert.match(renderer, /!this\.renderReadbackPending\|\|this\.renderReadbackPendingEpoch!==this\.renderTimingEpoch/,
    "an unresolved query from an old run-state or mode epoch must not starve the current epoch");
  assert.match(renderer, /if\(this\.renderReadbackPendingEpoch===sampledEpoch\)\{this\.renderReadbackPending=false/,
    "an old readback completion must not clear the current epoch's in-flight guard");
  assert.match(controller, /sameRenderContext = previous\.renderTimingContext === renderTimingContext/);
  assert.match(panel, /sample\.renderTimingContext === liveSnapshot\.renderTimingContext/,
    "rolling averages must never mix raster and SVO timing epochs");
  assert.match(panel, /renderTimed \? `Last presentation \$\{renderPerFrame_ms\.toFixed\(2\)\} ms on GPU` : "Awaiting presentation timestamp"/,
    "the panel must not present an unavailable timestamp as a real 0.00 ms measurement");
});
