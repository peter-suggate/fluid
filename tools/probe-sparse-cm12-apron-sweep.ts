/**
 * Every non-pressure Sparse CM12 stage is dispatched over the accepted cell or
 * row worklist, and roughly two thirds of that set is dry receiver apron. This
 * sweeps the two policy knobs that size the apron and reports, per arm, the
 * accepted population, the hardware phase partition, and a dam-front receipt so
 * a cheaper arm that has stopped resolving the front is visible immediately.
 *
 * Run:
 *   WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js \
 *   node --import tsx tools/probe-sparse-cm12-apron-sweep.ts
 */
import assert from "node:assert/strict";
import { fileURLToPath, pathToFileURL } from "node:url";
import type { MethodParamValues } from "../lib/core/method-contract";
import { resolveMethodValues } from "../lib/core/method-contract";
import { createSparseCM12LongDamBreakScene } from "../lib/core/scenes";
import { usePerformanceInstrumentationStore } from "../lib/core/stores/performance-instrumentation-store";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import {
  acquireWebGPUExclusiveLock,
  releaseWebGPUExclusiveLock,
} from "../lib/harness/webgpu-smoke-isolation";
import { adaptiveMassMethod } from "../lib/methods/adaptive-mass/method";
import { WebGPUAdaptiveMassSolver } from
  "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";

const FRAMES = 40;
const WARMUP = 8;
const ARMS: readonly { name: string; overrides: MethodParamValues }[] = [
  { name: "lookahead-4 (default)", overrides: {} },
  { name: "lookahead-2", overrides: { frontLookaheadSteps: 2 } },
  { name: "lookahead-1", overrides: { frontLookaheadSteps: 1 } },
];
const median = (values: number[]): number => {
  if (values.length === 0) return Number.NaN;
  const ordered = [...values].sort((left, right) => left - right);
  return ordered[Math.floor((ordered.length - 1) / 2)]!;
};

await acquireWebGPUExclusiveLock("dawn-probe", "tools/probe-sparse-cm12-apron-sweep.ts");
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
  const [nx, ny, nz] = [192, 96, 32];
  const results: unknown[] = [];
  for (const arm of ARMS) {
    const scene = createSparseCM12LongDamBreakScene();
    const values = resolveMethodValues(adaptiveMassMethod, "balanced",
      { timeStep: "scene", ...arm.overrides });
    const solver = await adaptiveMassMethod.createSolverAsync!(
      device, scene, "balanced", values, undefined, () => {}) as WebGPUAdaptiveMassSolver;
    const dt_s = scene.numerics.fixedDt_s ?? scene.numerics.maxDt_s;
    const accepted: number[] = []; const acceptedRows: number[] = [];
    const pressure: number[] = []; const totals: number[] = [];
    const phase = new Map<string, number[]>();
    let lastTrace: unknown;
    for (let frame = 1; frame <= WARMUP + FRAMES; frame += 1) {
      while (!solver.advanceTo(frame * dt_s, [])) await new Promise(setImmediate);
      const info = await solver.readStats();
      await new Promise((resolve) => setTimeout(resolve, 6));
      if (frame <= WARMUP) continue;
      accepted.push(info.adaptiveAcceptedCellCount ?? 0);
      acceptedRows.push(info.adaptiveAcceptedRowCount ?? 0);
      pressure.push(info.adaptivePressureCellCount ?? 0);
      const trace = info.physicsTrace;
      if (!trace || trace.measurementSource !== "gpu-hardware-timestamp") continue;
      if (trace === lastTrace) continue;
      lastTrace = trace;
      totals.push(trace.total_ms);
      for (const entry of trace.phases) {
        const bucket = phase.get(entry.label) ?? [];
        bucket.push(entry.duration_ms);
        phase.set(entry.label, bucket);
      }
    }
    const density = (await solver.readDiagnosticFields()).density;
    let front = -1; let mass = 0;
    for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1)
      for (let x = 0; x < nx; x += 1) {
        const rho = density[x + nx * (y + ny * z)]!;
        mass += Math.max(0, rho);
        if (rho >= 0.5) front = Math.max(front, x);
      }
    const phases = [...phase].map(([label, samples]) =>
      ({ label, median_ms: Number(median(samples).toFixed(4)) }));
    const solve = phases.find((entry) => entry.label.includes("MGPCG"))?.median_ms ?? 0;
    const sum = phases.reduce((total, entry) => total + entry.median_ms, 0);
    results.push({
      arm: arm.name,
      medianAcceptedCells: median(accepted),
      medianAcceptedRows: median(acceptedRows),
      medianPressureCells: median(pressure),
      medianAdvance_ms: Number(median(totals).toFixed(4)),
      pressureSolve_ms: Number(solve.toFixed(4)),
      nonPressure_ms: Number((sum - solve).toFixed(4)),
      liquidFrontX: front,
      totalMass: Number(mass.toFixed(2)),
      phases: phases.sort((left, right) => right.median_ms - left.median_ms),
    });
    solver.destroy();
  }
  console.log(JSON.stringify({ probe: "sparse-cm12-apron-sweep", results }, null, 2));
} finally {
  releaseWebGPUExclusiveLock();
}
