/**
 * Dawn acceptance: does one Sparse CM12 advance publish a hardware stage
 * partition, and does every SIM-panel node get a figure from it?
 *
 * The panel degrades silently — a rejected sample retires hardware boundaries
 * for the run and every stage node falls back to "—" under one queue-wall
 * total — so the failure this catches is invisible in the app.
 *
 * Run:
 *   WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js \
 *   node --import tsx tools/probe-sparse-cm12-stage-trace.ts
 */
import assert from "node:assert/strict";
import { fileURLToPath, pathToFileURL } from "node:url";
import {
  fluidPipelinePhaseCosts,
  measureFluidPipelineStage,
} from "../lib/core/fluid-pipeline";
import { resolveMethodValues } from "../lib/core/method-contract";
import type { PerformanceTrace } from "../lib/core/performance-trace";
import { createMinimalPowerDamBreak32Scene } from "../lib/core/scenes";
import { usePerformanceInstrumentationStore } from "../lib/core/stores/performance-instrumentation-store";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import {
  acquireWebGPUExclusiveLock,
  releaseWebGPUExclusiveLock,
} from "../lib/harness/webgpu-smoke-isolation";
import { ADAPTIVE_MASS_FLUID_PIPELINE } from
  "../lib/methods/adaptive-mass/adaptive-mass-frame-pipeline";
import { adaptiveMassMethod } from "../lib/methods/adaptive-mass/method";

const SAMPLED_ADVANCES = 24;

await acquireWebGPUExclusiveLock("dawn-probe", "tools/probe-sparse-cm12-stage-trace.ts");
try {
  usePerformanceInstrumentationStore.getState().setMode("timeline");
  const modulePath = process.env.WEBGPU_NODE_MODULE
    ?? fileURLToPath(new URL("../node_modules/webgpu/index.js", import.meta.url));
  const { create, globals } = await import(pathToFileURL(modulePath).href) as {
    create(options: string[]): GPU;
    globals: Record<string, unknown>;
  };
  Object.assign(globalThis, globals);
  const gpu = create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
  Object.defineProperty(globalThis, "navigator", { configurable: true, value: { gpu } });
  const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
  assert.ok(adapter, "WebGPU did not expose an adapter");
  const features: GPUFeatureName[] = [];
  if (adapter.features.has("timestamp-query")) features.push("timestamp-query");
  const device = await adapter.requestDevice({
    requiredFeatures: features,
    requiredLimits: requiredFluidDeviceLimits(adapter.limits),
  });
  const validationErrors: string[] = [];
  device.addEventListener("uncapturederror", (event) => {
    validationErrors.push(event.error.message);
  });

  const scene = createMinimalPowerDamBreak32Scene();
  const values = resolveMethodValues(adaptiveMassMethod, "balanced", { timeStep: "scene" });
  const solver = await adaptiveMassMethod.createSolverAsync!(
    device, scene, "balanced", values, undefined, () => {});
  const dt_s = scene.numerics.maxDt_s;
  const sources = new Set<string>();
  let trace: PerformanceTrace | undefined;
  for (let frame = 1; frame <= SAMPLED_ADVANCES; frame += 1) {
    while (!solver.advanceTo(frame * dt_s, [])) await new Promise(setImmediate);
    const info = await solver.readStats();
    if (info.physicsTrace) {
      trace = info.physicsTrace;
      sources.add(info.physicsTrace.measurementSource ?? "unknown");
    }
    // The capture cadence is time-based; without a gap every advance declines.
    await new Promise((resolve) => setTimeout(resolve, 12));
  }
  assert.ok(trace, "no physics trace reached the panel");
  const costs = fluidPipelinePhaseCosts(trace);
  const stages = ADAPTIVE_MASS_FLUID_PIPELINE.stages.map((stage) => {
    const measurement = measureFluidPipelineStage(
      stage, ADAPTIVE_MASS_FLUID_PIPELINE.stages, costs, trace.total_ms, "on");
    return {
      stage: stage.id,
      kind: measurement.kind,
      duration_ms: Number((measurement.duration_ms ?? 0).toFixed(4)),
    };
  });
  const accounted_ms = trace.phases.reduce((sum, phase) => sum + phase.duration_ms, 0);
  console.log(JSON.stringify({
    phase: "sparse-cm12-advance-stage-trace",
    timestampQuery: features.length > 0,
    measurementSources: [...sources],
    total_ms: Number(trace.total_ms.toFixed(4)),
    closureError_ms: Number((trace.total_ms - accounted_ms).toFixed(6)),
    phases: trace.phases.map((phase) => ({
      label: phase.label, duration_ms: Number(phase.duration_ms.toFixed(4)),
    })),
    stages,
    validationErrors,
  }, null, 2));
  solver.destroy();

  assert.deepEqual(validationErrors, []);
  if (features.length > 0) {
    assert.equal(trace.measurementSource, "gpu-hardware-timestamp",
      "the advance fell back to a queue-wall total, so no stage node can show a figure");
    assert.ok(Math.abs(trace.total_ms - accounted_ms) < 1e-6,
      "the stage partition must close on the advance exactly");
    // Sub-tick stages round to zero on a quantized timestamp counter; the
    // expensive half of the advance must always be attributed.
    assert.ok(stages.filter((stage) => stage.kind === "measured").length >= 10,
      `only ${stages.filter((stage) => stage.kind === "measured").length} stages were attributed`);
  }
} finally {
  releaseWebGPUExclusiveLock();
}
