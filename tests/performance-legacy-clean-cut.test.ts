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

test("the generic recorder exclusively owns GPU timestamp queries", () => {
  for (const file of productionRoots.flatMap(productionSourceFiles)) {
    if (file.endsWith("/lib/performance-trace.ts")) continue;
    assert.doesNotMatch(
      readFileSync(file, "utf8"),
      /\b(?:timestampWrites|GPUComputePassTimestampWrites|createQuerySet|resolveQuerySet)\b/,
      `standalone timestamp instrumentation survived in ${file}`,
    );
  }
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
