/**
 * Where does a Sparse CM12 advance spend its time, per hardware stage,
 * averaged over many advances and on a scene large enough that non-pressure
 * stages clear Dawn's 65.5 us timestamp tick?
 *
 * `probe-sparse-cm12-stage-trace.ts` is the acceptance gate for the partition
 * existing at all; it samples the mini 32^3 scene and reports one trace. This
 * is the measurement lane: pick a scene, sample every advance, and report the
 * median stage cost with the pressure solve separated from everything else.
 *
 * Run:
 *   WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js \
 *   node --import tsx tools/probe-sparse-cm12-stage-cost.ts --scene=long-dam
 */
import assert from "node:assert/strict";
import { fileURLToPath, pathToFileURL } from "node:url";
import {
  fluidPipelinePhaseCosts,
  measureFluidPipelineStage,
} from "../lib/core/fluid-pipeline";
import { resolveMethodValues } from "../lib/core/method-contract";
import type { PerformanceTrace } from "../lib/core/performance-trace";
import {
  createMinimalPowerDamBreak32Scene,
  createMinimalPowerDamBreak64Scene,
  createSparseCM12LongDamBreakScene,
  createSymmetricExpansionScene,
} from "../lib/core/scenes";
import { usePerformanceInstrumentationStore } from "../lib/core/stores/performance-instrumentation-store";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import {
  acquireWebGPUExclusiveLock,
  releaseWebGPUExclusiveLock,
} from "../lib/harness/webgpu-smoke-isolation";
import { ADAPTIVE_MASS_FLUID_PIPELINE } from
  "../lib/methods/adaptive-mass/adaptive-mass-frame-pipeline";
import { adaptiveMassMethod } from "../lib/methods/adaptive-mass/method";

const argument = (name: string, fallback: string): string =>
  process.argv.slice(2).find((value) => value.startsWith(`--${name}=`))
    ?.slice(name.length + 3) ?? fallback;

const sceneName = argument("scene", "long-dam");
const warmup = Number(argument("warmup", "8"));
const sampled = Number(argument("frames", "40"));
const buildScene = sceneName === "mini32" ? createMinimalPowerDamBreak32Scene
  : sceneName === "mini64" ? createMinimalPowerDamBreak64Scene
  : sceneName === "long-dam" ? createSparseCM12LongDamBreakScene
  : createSymmetricExpansionScene;

const median = (values: number[]): number => {
  if (values.length === 0) return Number.NaN;
  const ordered = [...values].sort((left, right) => left - right);
  return ordered[Math.floor((ordered.length - 1) / 2)]!;
};

await acquireWebGPUExclusiveLock("dawn-probe", "tools/probe-sparse-cm12-stage-cost.ts");
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
  const device = await adapter.requestDevice({
    requiredFeatures: adapter.features.has("timestamp-query")
      ? ["timestamp-query" as GPUFeatureName] : [],
    requiredLimits: requiredFluidDeviceLimits(adapter.limits),
  });
  const validationErrors: string[] = [];
  device.addEventListener("uncapturederror", (event) => {
    validationErrors.push(event.error.message);
  });

  const scene = buildScene();
  const values = resolveMethodValues(adaptiveMassMethod, "balanced", { timeStep: "scene" });
  const solver = await adaptiveMassMethod.createSolverAsync!(
    device, scene, "balanced", values, undefined, () => {});
  const dt_s = scene.numerics.fixedDt_s ?? scene.numerics.maxDt_s;

  const stageSamples = new Map<string, number[]>();
  const totals: number[] = [];
  const committedSamples: number[] = [];
  const phaseSamples = new Map<string, number[]>();
  let seen = 0;
  let lastTrace: PerformanceTrace | undefined;
  for (let frame = 1; frame <= warmup + sampled * 3; frame += 1) {
    while (!solver.advanceTo(frame * dt_s, [])) await new Promise(setImmediate);
    const info = await solver.readStats();
    const trace = info.physicsTrace;
    await new Promise((resolve) => setTimeout(resolve, 6));
    if (!trace || trace.measurementSource !== "gpu-hardware-timestamp") continue;
    if (trace === lastTrace) continue;
    lastTrace = trace;
    if (frame <= warmup) continue;
    const committed = (info as { adaptiveTopologyCommittedBrickCount?: number })
      .adaptiveTopologyCommittedBrickCount ?? 0;
    const costs = fluidPipelinePhaseCosts(trace);
    for (const stage of ADAPTIVE_MASS_FLUID_PIPELINE.stages) {
      const measurement = measureFluidPipelineStage(
        stage, ADAPTIVE_MASS_FLUID_PIPELINE.stages, costs, trace.total_ms, "on");
      const key = committed > 0 ? `${stage.id}|changed` : `${stage.id}|quiescent`;
      for (const name of [stage.id, key]) {
        const bucket = stageSamples.get(name) ?? [];
        bucket.push(measurement.duration_ms ?? 0);
        stageSamples.set(name, bucket);
      }
    }
    for (const phase of trace.phases) {
      const key = committed > 0 ? `${phase.label}|changed` : `${phase.label}|quiescent`;
      for (const name of [phase.label, key]) {
        const bucket = phaseSamples.get(name) ?? [];
        bucket.push(phase.duration_ms);
        phaseSamples.set(name, bucket);
      }
    }
    totals.push(trace.total_ms);
    committedSamples.push(committed);
    seen += 1;
    if (seen >= sampled) break;
  }

  const stages = [...stageSamples].map(([id, samples]) => ({
    stage: id, median_ms: Number(median(samples).toFixed(4)),
  })).sort((left, right) => right.median_ms - left.median_ms);
  const total = median(totals);
  const pressure = stages.find((stage) => stage.stage === "pressure-solve")?.median_ms ?? 0;
  console.log(JSON.stringify({
    probe: "sparse-cm12-stage-cost", scene: sceneName, samples: seen,
    medianAdvance_ms: Number(total.toFixed(4)),
    pressureSolve_ms: Number(pressure.toFixed(4)),
    nonPressure_ms: Number((total - pressure).toFixed(4)),
    committedBricksPerFrame: committedSamples,
    quiescentFrames: committedSamples.filter((value) => value === 0).length,
    phases: [...phaseSamples].map(([label, samples]) => ({
      label, median_ms: Number(median(samples).toFixed(4)), samples: samples.length,
    })).sort((left, right) => right.median_ms - left.median_ms),
    stages, validationErrors,
  }, null, 2));
  solver.destroy();
  assert.deepEqual(validationErrors, []);
} finally {
  releaseWebGPUExclusiveLock();
}
