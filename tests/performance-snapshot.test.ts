import assert from "node:assert/strict";
import test from "node:test";
import { emptyPerformance, measuredGPUTime_ms, type PerformanceSnapshot } from "../lib/stores/diagnostics-store";

test("GPU performance totals exclude unavailable or stale timer values", () => {
  const stale: PerformanceSnapshot = {
    ...emptyPerformance,
    gpuAdvection_ms: 7,
    gpuPressure_ms: 11,
    gpuRender_ms: 13
  };
  assert.equal(measuredGPUTime_ms(stale), 0);
  assert.equal(measuredGPUTime_ms({ ...stale, gpuRenderTimingAvailable: true }), 13);
  assert.equal(measuredGPUTime_ms({ ...stale, gpuPhysicsTimingAvailable: true }), 18);
  assert.equal(measuredGPUTime_ms({ ...stale, gpuPhysicsTimingAvailable: true, gpuRenderTimingAvailable: true }), 31);
});

test("empty performance identifies the renderer without claiming a sample", () => {
  assert.equal(emptyPerformance.waterRenderMode, "rasterized");
  assert.equal(emptyPerformance.gpuRenderTimestampSupported, false);
  assert.equal(emptyPerformance.gpuRenderTimingAvailable, false);
  assert.equal(emptyPerformance.gpuSurfaceExtraction_ms, 0);
  assert.equal(emptyPerformance.gpuUpscale_ms, 0);
});
