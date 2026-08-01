import assert from "node:assert/strict";
import { existsSync, readdirSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import test from "node:test";

const productionRoots = ["../app", "../components", "../lib", "../tools", "../native/Sources"].map((path) =>
  fileURLToPath(new URL(path, import.meta.url)));

function productionSourceFiles(directory: string): string[] {
  return readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const path = `${directory}/${entry.name}`;
    if (entry.isDirectory()) return productionSourceFiles(path);
    return /\.(?:ts|tsx|js|jsx|swift)$/.test(entry.name) ? [path] : [];
  });
}

const retiredSymbols: readonly [string, RegExp][] = [
  ["flat CPU frame timings", /\b(?:cpuFrame_ms|cpuPhysicsSubmit_ms|cpuDataUpload_ms|cpuRenderEncode_ms)\b/],
  ["flat GPU presentation timings", /\b(?:gpuRender_ms|gpuSurfaceExtraction_ms|gpuCaustics_ms|gpuDryScene_ms|gpuSvoTemporal_ms|gpuInterfaceFront_ms|gpuInterfaceBack_ms|gpuInterfaces_ms|gpuSprayFront_ms|gpuSprayBack_ms|gpuSprayRender_ms|gpuOpticalComposite_ms|gpuUpscale_ms|gpuOverlays_ms)\b/],
  ["renderer timing epochs and decoders", /\b(?:RenderStageTimings|RENDER_TIMESTAMP_QUERY_COUNT|decodeRenderStageTimestamps|hasResolvedRenderTimestampSample|renderTimingContext|renderTimingEpoch|renderTimingSampleId|gpuRenderTimestampAvailable)\b/],
  ["profiler readback toggle", /\b(?:performanceReadbacksEnabled|setPerformanceReadbacksEnabled)\b/],
  ["retired stage capture", /\b(?:gpuStageCapture|encodeGPUStageTextureCapture|PendingGPUStageCapture)\b/],
  ["flat physics and queue timings", /\b(?:GPUPhysicsTimings|gpuTimings|gpuStep_ms|gpuBatchWall_ms|gpuAdvanceWall_ms|gpuPresentationWall_ms|gpuQueueWall_ms|gpuTelemetryWall_ms|gpuProfilerWallTotal_ms|cpuAdvanceEncode_ms)\b/],
  ["legacy quadtree pack wall timing", /\bgpuWall_ms\b/],
  ["intrusive pressure replay timing", /\bencodePressureWallProbe\b/],
  ["retired performance model imports", /(?:performance-averaging|performance-stage-model|performance-scheduling|presentation-frame-rate|gpu-stage-capture)/],
  ["retired frame-rate visualization", /\b(?:FrameRateCounter|recordPresentedFrame|Sparkline)\b/],
];

test("production source contains no legacy timing, readback, or performance visualization surface", () => {
  for (const file of productionRoots.flatMap(productionSourceFiles)) {
    const contents = readFileSync(file, "utf8");
    for (const [label, pattern] of retiredSymbols) {
      assert.doesNotMatch(contents, pattern, `${label} survived in ${file}`);
    }
  }
});

/**
 * The generic recorders own every timestamp query the app can encode. The Dawn
 * smoke and atomic-clock feasibility tool keep isolated timestamp queries:
 * they are diagnostic harnesses, not application instrumentation.
 */
const timestampQueryOwners = [
  "/lib/performance-trace.ts",
  "/tools/webgpu-smoke-gpu-audits.ts",
  "/tools/experiment-webgpu-atomic-clock.ts",
];

test("measurement never changes the command graph the frame would submit anyway", () => {
  const recorder = readFileSync(new URL("../lib/performance-trace.ts", import.meta.url), "utf8");
  const start = recorder.indexOf("export class GPUStageTimestampRecorder");
  const stage = recorder.slice(start, recorder.indexOf("\nexport ", start + 1));
  assert.ok(start >= 0 && stage.length > 0);
  assert.match(stage, /instrument\(encoder: GPUCommandEncoder\): GPUCommandEncoder/);
  assert.match(stage, /beginningOfPassWriteIndex/);
  assert.doesNotMatch(stage, /onSubmittedWorkDone/,
    "a stage recorder must never introduce a queue fence");
  // The proxy forwards real passes; only the closing boundary needs a pass.
  assert.equal((stage.match(/encoder\.beginComputePass|target\.beginComputePass/g) ?? []).length, 2,
    "one forwarded compute pass and exactly one marker pass of the recorder's own");
});

test("retired performance modules remain physically deleted", () => {
  for (const path of [
    "../components/FrameRateCounter.tsx",
    "../lib/gpu-stage-capture.ts",
    "../lib/performance-averaging.ts",
    "../lib/performance-scheduling.ts",
    "../lib/performance-stage-model.ts",
    "../lib/presentation-frame-rate.ts",
  ]) assert.equal(existsSync(new URL(path, import.meta.url)), false, path);
});
