/**
 * Queue-complete frame timing for the uninstrumented Sparse CM12 production path.
 *
 * Unlike the stage-cost probe, this benchmark does not insert timestamp seams or
 * read diagnostics inside the measured window. Each sample covers command
 * encoding, submission, and completion of exactly one simulation advance.
 * Construction, warmup, and the terminal diagnostic readback are excluded.
 */
import assert from "node:assert/strict";
import { writeFile } from "node:fs/promises";
import { fileURLToPath, pathToFileURL } from "node:url";

import { resolveMethodValues } from "../lib/core/method-contract";
import {
  createMinimalPowerDamBreakScene,
  createMinimalPowerDamBreak64Scene,
} from "../lib/core/scenes";
import { usePerformanceInstrumentationStore } from
  "../lib/core/stores/performance-instrumentation-store";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { BROWSER_GPU_THROUGHPUT_DEPTH } from "../lib/core/webgpu-renderer";
import {
  acquireWebGPUExclusiveLock,
  releaseWebGPUExclusiveLock,
} from "../lib/harness/webgpu-smoke-isolation";
import {
  createProcessRetainedDawnGPU,
  type NodeDawnProvider,
} from "../lib/harness/node-dawn-provider";
import { adaptiveMassMethod } from "../lib/methods/adaptive-mass/method";
import { fingerprintSparseCM12RepositorySources } from
  "./sparse-cm12-source-content-fingerprint";

const argument = (name: string, fallback: string): string =>
  process.argv.slice(2).find((value) => value.startsWith(`--${name}=`))
    ?.slice(name.length + 3) ?? fallback;

if (process.argv.includes("--help") || process.argv.includes("-h")) {
  console.log(`Sparse CM12 uninstrumented production-frame benchmark

Options:
  --scene=mini16|mini64             Scene (default mini16)
  --brick-fine=4|8|16              Sparse brick ladder (default 8)
  --presentation-page=4|8|16       Presentation page size (default 8)
  --minimum-cell-size=N            Full-domain minimum cell size (default 0)
  --warmup=N                       Untimed queue-complete frames (default 8)
  --frames=N                       Measured queue-complete frames (default 40)
  --queue-depth=N                  Advances per completion fence (default browser depth)
  --out=PATH                       Write JSON receipt

Every measured sample is one shipping command graph with performance
instrumentation disabled. Queue depth amortizes the diagnostic host fence; the
reported frame time is queue-complete production throughput. Construction,
warmup, and diagnostics are excluded.`);
  process.exit(0);
}

const sceneName = argument("scene", "mini16");
const brickFineResolution = Number(argument("brick-fine", "8"));
const presentationPageResolution = Number(argument("presentation-page", "8"));
const minimumCellSize = Number(argument("minimum-cell-size", "0"));
const warmupFrames = Number(argument("warmup", "8"));
const measuredFrames = Number(argument("frames", "40"));
const queueDepth = Number(argument("queue-depth", String(BROWSER_GPU_THROUGHPUT_DEPTH)));
const outputPath = argument("out", "");

if (sceneName !== "mini16" && sceneName !== "mini64") {
  throw new RangeError("scene must be mini16 or mini64");
}
for (const [name, value] of Object.entries({
  brickFineResolution, presentationPageResolution, warmupFrames, measuredFrames, queueDepth,
})) {
  if (!Number.isSafeInteger(value) || value < 1) {
    throw new RangeError(`${name} must be a positive integer`);
  }
}
if (![4, 8, 16].includes(brickFineResolution)
  || ![4, 8, 16].includes(presentationPageResolution)
  || presentationPageResolution > brickFineResolution
  || brickFineResolution % presentationPageResolution !== 0) {
  throw new RangeError("brick-fine and presentation-page must be compatible values in 4, 8, 16");
}
if (!Number.isSafeInteger(minimumCellSize) || minimumCellSize < 0
  || (minimumCellSize > 0 && (minimumCellSize & (minimumCellSize - 1)) !== 0)) {
  throw new RangeError("minimum-cell-size must be zero or a positive power of two");
}

const median = (values: readonly number[]): number => {
  const ordered = [...values].sort((left, right) => left - right);
  const middle = Math.floor(ordered.length / 2);
  return ordered.length % 2 === 0
    ? (ordered[middle - 1]! + ordered[middle]!) / 2
    : ordered[middle]!;
};
const percentile = (values: readonly number[], quantile: number): number => {
  const ordered = [...values].sort((left, right) => left - right);
  return ordered[Math.ceil(quantile * ordered.length) - 1]!;
};
const rounded = (value: number): number => Number(value.toFixed(4));

const root = fileURLToPath(new URL("..", import.meta.url));
const sourceContentFingerprint = await fingerprintSparseCM12RepositorySources(root);

await acquireWebGPUExclusiveLock(
  "dawn-benchmark", "tools/benchmark-sparse-cm12-production-frame.ts");
let device: GPUDevice | undefined;
let solver: Awaited<ReturnType<NonNullable<typeof adaptiveMassMethod.createSolverAsync>>> | undefined;
try {
  // This is the defining difference from the attribution probe.
  usePerformanceInstrumentationStore.getState().setMode("off");
  const modulePath = process.env.WEBGPU_NODE_MODULE
    ?? fileURLToPath(new URL("../node_modules/webgpu/index.js", import.meta.url));
  const dawn = await import(pathToFileURL(modulePath).href) as NodeDawnProvider;
  Object.assign(globalThis, dawn.globals);
  const gpu = createProcessRetainedDawnGPU(dawn, [
    `backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`,
  ]);
  Object.defineProperty(globalThis, "navigator", { configurable: true, value: { gpu } });
  const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
  assert.ok(adapter, "WebGPU did not expose an adapter");
  device = await adapter.requestDevice({
    requiredLimits: requiredFluidDeviceLimits(adapter.limits),
  });
  const validationErrors: string[] = [];
  device.addEventListener("uncapturederror", (event) => {
    validationErrors.push(event.error.message);
  });

  const scene = sceneName === "mini16"
    ? createMinimalPowerDamBreakScene() : createMinimalPowerDamBreak64Scene();
  if (minimumCellSize > 0) {
    scene.fluid.refinementRegions = [{
      id: `production-frame-minimum-cell-${minimumCellSize}`,
      rule: "minimum-cell-size",
      minimumCellSize_cells: minimumCellSize,
      min_m: {
        x: -0.5 * scene.container.width_m,
        y: 0,
        z: -0.5 * scene.container.depth_m,
      },
      max_m: {
        x: 0.5 * scene.container.width_m,
        y: scene.container.height_m,
        z: 0.5 * scene.container.depth_m,
      },
    }];
  }
  const values = resolveMethodValues(adaptiveMassMethod, "balanced", {
    timeStep: "scene",
    brickFineResolution: String(brickFineResolution),
    presentationPageResolution: String(presentationPageResolution),
  });
  solver = await adaptiveMassMethod.createSolverAsync!(
    device, scene, "balanced", values, undefined, () => {});
  solver.applyRuntimeValues?.(values);
  const readySolver = solver as typeof solver & { waitForSimulationReady?: () => Promise<void> };
  await readySolver.waitForSimulationReady?.();
  await device.queue.onSubmittedWorkDone();

  const dt_s = scene.numerics.fixedDt_s ?? scene.numerics.maxDt_s;
  const encode = async (frame: number): Promise<void> => {
    while (!solver!.advanceTo(frame * dt_s, [])) {
      await new Promise<void>((resolve) => setImmediate(resolve));
    }
  };
  const advanceAndFence = async (frame: number): Promise<void> => {
    await encode(frame);
    await device!.queue.onSubmittedWorkDone();
    // Let the already-completed pressure receipt map callback publish before
    // the next production frame decides its encoded pressure ceiling.
    await new Promise<void>((resolve) => setImmediate(resolve));
  };

  for (let frame = 1; frame <= warmupFrames; frame += 1) {
    await advanceAndFence(frame);
  }
  const batches: Array<{ frames: number; total_ms: number; perFrame_ms: number }> = [];
  let submitted = 0;
  while (submitted < measuredFrames) {
    const batchFrames = Math.min(queueDepth, measuredFrames - submitted);
    const startedAt_ms = performance.now();
    for (let offset = 1; offset <= batchFrames; offset += 1) {
      await encode(warmupFrames + submitted + offset);
    }
    await device.queue.onSubmittedWorkDone();
    const total_ms = performance.now() - startedAt_ms;
    batches.push({ frames: batchFrames, total_ms, perFrame_ms: total_ms / batchFrames });
    submitted += batchFrames;
    await new Promise<void>((resolve) => setImmediate(resolve));
  }
  const samples_ms = batches.map((batch) => batch.perFrame_ms);
  const throughputMean_ms = batches.reduce((sum, batch) => sum + batch.total_ms, 0)
    / measuredFrames;
  const info = await solver.readStats();
  const report = {
    benchmark: "sparse-cm12-production-frame",
    scene: sceneName,
    configuration: {
      brickFineResolution,
      presentationPageResolution,
      minimumCellSize: minimumCellSize || undefined,
      regionScope: minimumCellSize > 0 ? "domain" : undefined,
      timeStep: "scene",
      dt_s,
      performanceInstrumentation: "off",
      queueDepth,
      measurement: "command encoding + submit + amortized queue completion per advance",
      constructionExcluded: true,
      diagnosticsExcluded: true,
    },
    sourceContentFingerprint,
    warmupFrames,
    measuredFrames,
    frame_ms: {
      mean: rounded(throughputMean_ms),
      median: rounded(median(samples_ms)),
      p95: rounded(percentile(samples_ms, 0.95)),
      minimum: rounded(Math.min(...samples_ms)),
      maximum: rounded(Math.max(...samples_ms)),
      samples: samples_ms.map(rounded),
      batches: batches.map((batch) => ({
        frames: batch.frames,
        total: rounded(batch.total_ms),
        perFrame: rounded(batch.perFrame_ms),
      })),
    },
    terminalWork: {
      acceptedCells: info.adaptiveAcceptedCellCount,
      acceptedRows: info.adaptiveAcceptedRowCount,
      pressureCells: info.adaptivePressureCellCount,
      pressureRows: info.adaptivePressureActiveRowCount,
      pressureIterationsExecuted: info.pressureIterationsExecuted,
      pressureIterationsEncoded: info.pressureIterationsEncoded,
      topologyCommitted: info.adaptiveTopologyCommittedBrickCount,
    },
    validationErrors,
  };
  console.log(JSON.stringify(report, null, 2));
  if (outputPath) await writeFile(outputPath, `${JSON.stringify(report, null, 2)}\n`, "utf8");
  assert.deepEqual(validationErrors, []);
} finally {
  try {
    await device?.queue.onSubmittedWorkDone();
  } finally {
    solver?.destroy();
    device?.destroy();
    await new Promise<void>((resolve) => setImmediate(resolve));
    await releaseWebGPUExclusiveLock();
  }
}
