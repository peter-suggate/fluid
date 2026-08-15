/**
 * Matched serialized-frame performance gate for Uniform CM12 vs Sparse CM12.
 *
 * Run:
 *   WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js \
 *   node --import tsx tools/benchmark-sparse-cm12-frame.ts
 *
 * Construction and warmup are excluded. Arms use the same authored scene,
 * 32x16x32 finest lattice and 0.004 s step, and alternate order to reduce
 * drift. The hard ratio is defined beside the adaptive method implementation.
 */
import assert from "node:assert/strict";
import { fileURLToPath, pathToFileURL } from "node:url";
import type {
  GPUSolverInstance,
  MethodParamValues,
  SimulationMethod,
} from "../lib/core/method-contract";
import { resolveMethodValues } from "../lib/core/method-contract";
import type { PerformanceTrace } from "../lib/core/performance-trace";
import {
  createMinimalPowerDamBreak32Scene,
  createSymmetricExpansionScene,
} from "../lib/core/scenes";
import { usePerformanceInstrumentationStore } from "../lib/core/stores/performance-instrumentation-store";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import {
  acquireWebGPUExclusiveLock,
  releaseWebGPUExclusiveLock,
} from "../lib/harness/webgpu-smoke-isolation";
import { adaptiveMassMethod } from "../lib/methods/adaptive-mass/method";
import {
  evaluateSparseCM12Performance,
  SPARSE_CM12_MINI_DAM_32_PERFORMANCE_ACCEPTANCE,
  SPARSE_CM12_PERFORMANCE_ACCEPTANCE,
  type SparseCM12BenchmarkArm,
  type SparseCM12TopologySample,
} from "../lib/methods/adaptive-mass/adaptive-mass-performance";
import { uniformMethod } from "../lib/methods/uniform/method";

const positiveInteger = (name: string, fallback: number): number => {
  const prefix = `--${name}=`;
  const inline = process.argv.slice(2).find((value) => value.startsWith(prefix));
  const value = Number(inline?.slice(prefix.length) ?? fallback);
  if (!Number.isSafeInteger(value) || value < 1) {
    throw new RangeError(`${name} must be a positive integer`);
  }
  return value;
};

const warmupFrames = positiveInteger("warmup", 5);
const timedFrames = positiveInteger("frames", 40);
const sceneArgument = process.argv.slice(2)
  .find((value) => value.startsWith("--scene="))?.slice("--scene=".length)
  ?? "symmetric";
if (sceneArgument !== "symmetric" && sceneArgument !== "mini32") {
  throw new RangeError("scene must be symmetric or mini32");
}
const sparseResolutionArgument = process.argv.slice(2)
  .find((value) => value.startsWith("--sparse-resolution="))
  ?.slice("--sparse-resolution=".length) ?? "all-fine";
if (sparseResolutionArgument !== "all-fine" && sparseResolutionArgument !== "adaptive") {
  throw new RangeError("sparse-resolution must be all-fine or adaptive");
}
const buildScene = sceneArgument === "mini32"
  ? createMinimalPowerDamBreak32Scene
  : createSymmetricExpansionScene;
const acceptance = sceneArgument === "mini32"
  ? SPARSE_CM12_MINI_DAM_32_PERFORMANCE_ACCEPTANCE
  : SPARSE_CM12_PERFORMANCE_ACCEPTANCE;
const scene = buildScene();
const dimensions = acceptance.finestDimensions;
const dt_s = scene.numerics.fixedDt_s ?? scene.numerics.maxDt_s;

interface FrameStateSample {
  readonly time_s: number;
  readonly duration_ms: number;
  readonly activeCells: number;
  readonly residentBricks: number;
  readonly mixedSeams: number;
  readonly pressureIterations: number;
  readonly maximumCfl: number;
}

interface MutableArm {
  readonly method: SimulationMethod;
  readonly solver: GPUSolverInstance;
  readonly endToEndFrame_ms: number[];
  readonly cpuTraces: PerformanceTrace[];
  readonly gpuTraces: PerformanceTrace[];
  readonly frameState: FrameStateSample[];
  initialTopology?: SparseCM12TopologySample;
  readonly evolvedTopology: SparseCM12TopologySample[];
  lastCPUTraceSampleId: number;
  lastGPUTraceSampleId: number;
}

function topologySample(info: Awaited<ReturnType<GPUSolverInstance["readStats"]>>): SparseCM12TopologySample {
  return {
    fineBricks: info.adaptiveFineBrickCount ?? 0,
    coarseBricks: info.adaptiveCoarseBrickCount ?? 0,
    fineCoarseFaceConnectedPairs: info.adaptiveFineCoarseFaceConnectedPairCount ?? 0,
    mixedSeamRows: info.adaptiveMixedSeamFaceCount ?? 0,
  };
}

function collectTrace(
  destination: PerformanceTrace[],
  trace: PerformanceTrace | undefined,
  lastSampleId: number,
): number {
  if (!trace || trace.sampleId === lastSampleId) return lastSampleId;
  destination.push(trace);
  return trace.sampleId;
}

async function advanceOne(arm: MutableArm, targetTime_s: number, timed: boolean) {
  const startedAt_ms = performance.now();
  while (!arm.solver.advanceTo(targetTime_s, [])) {
    await new Promise((resolve) => setImmediate(resolve));
  }
  const info = await arm.solver.readStats();
  if (timed) {
    const duration_ms = performance.now() - startedAt_ms;
    arm.endToEndFrame_ms.push(duration_ms);
    arm.frameState.push({
      time_s: info.completedTime_s ?? targetTime_s,
      duration_ms,
      activeCells: info.activeSampleCount ?? info.cellCount,
      residentBricks: info.fluidBrickResidentCount ?? 0,
      mixedSeams: info.adaptiveMixedSeamFaceCount ?? 0,
      pressureIterations: info.pressureIterations,
      maximumCfl: info.maxComponentCfl ?? 0,
    });
    if (arm.method.id === "adaptive-mass") {
      arm.evolvedTopology.push(topologySample(info));
    }
  }
  arm.lastCPUTraceSampleId = collectTrace(
    arm.cpuTraces,
    info.physicsCPUTrace,
    arm.lastCPUTraceSampleId,
  );
  arm.lastGPUTraceSampleId = collectTrace(
    arm.gpuTraces,
    info.physicsTrace,
    arm.lastGPUTraceSampleId,
  );
}

async function createArm(
  device: GPUDevice,
  method: SimulationMethod,
): Promise<MutableArm> {
  const overrides: MethodParamValues = method.id === "uniform"
    ? { timeStep: "scene", densityPostProcessing: "off" }
    : { timeStep: "scene", resolutionMode: sparseResolutionArgument };
  const values = resolveMethodValues(method, "balanced", overrides);
  const solver = await method.createSolverAsync!(
    device,
    buildScene(),
    "balanced",
    values,
    undefined,
    () => {},
  );
  await device.queue.onSubmittedWorkDone();
  assert.deepEqual(
    [solver.info.nx, solver.info.ny, solver.info.nz],
    dimensions,
    `${method.id} did not construct the matched finest lattice`,
  );
  const initialInfo = await solver.readStats();
  return {
    method,
    solver,
    endToEndFrame_ms: [],
    cpuTraces: [],
    gpuTraces: [],
    frameState: [],
    initialTopology: method.id === "adaptive-mass"
      ? topologySample(initialInfo) : undefined,
    evolvedTopology: [],
    lastCPUTraceSampleId: 0,
    lastGPUTraceSampleId: 0,
  };
}

function frozenArm(arm: MutableArm): SparseCM12BenchmarkArm {
  return {
    methodId: arm.method.id as SparseCM12BenchmarkArm["methodId"],
    sceneId: scene.sceneId,
    finestDimensions: dimensions,
    dt_s,
    constructionExcluded: true,
    endToEndFrame_ms: arm.endToEndFrame_ms,
    cpuTraces: arm.cpuTraces,
    gpuTraces: arm.gpuTraces,
    initialTopology: arm.initialTopology,
    evolvedTopology: arm.evolvedTopology,
  };
}

function summarizeFrameState(samples: readonly FrameStateSample[]) {
  const ordered = (select: (sample: FrameStateSample) => number) =>
    samples.map(select).sort((left, right) => left - right);
  const summary = (select: (sample: FrameStateSample) => number) => {
    const values = ordered(select);
    return {
      initial: select(samples[0]!),
      final: select(samples.at(-1)!),
      median: values[Math.floor((values.length - 1) / 2)]!,
      p95: values[Math.ceil(0.95 * values.length) - 1]!,
      maximum: values.at(-1)!,
    };
  };
  return {
    activeCells: summary((sample) => sample.activeCells),
    residentBricks: summary((sample) => sample.residentBricks),
    mixedSeams: summary((sample) => sample.mixedSeams),
    pressureIterations: summary((sample) => sample.pressureIterations),
    maximumCfl: summary((sample) => sample.maximumCfl),
    worstFrames: [...samples]
      .sort((left, right) => right.duration_ms - left.duration_ms)
      .slice(0, 5),
  };
}

await acquireWebGPUExclusiveLock(
  "dawn-benchmark",
  "tools/benchmark-sparse-cm12-frame.ts",
);
let uniform: MutableArm | undefined;
let sparse: MutableArm | undefined;
try {
  usePerformanceInstrumentationStore.getState().setMode("timeline");
  const modulePath = process.env.WEBGPU_NODE_MODULE
    ?? fileURLToPath(new URL("../node_modules/webgpu/index.js", import.meta.url));
  const { create, globals } = await import(pathToFileURL(modulePath).href) as {
    create(options: string[]): GPU;
    globals: Record<string, unknown>;
  };
  Object.assign(globalThis, globals);
  const gpu = create([
    `backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`,
    ...(process.env.FLUID_WEBGPU_ADAPTER
      ? [`adapter=${process.env.FLUID_WEBGPU_ADAPTER}`] : []),
  ]);
  Object.defineProperty(globalThis, "navigator", {
    configurable: true,
    value: { gpu },
  });
  const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
  assert.ok(adapter, "WebGPU did not expose an adapter");
  const requestedFeatures: GPUFeatureName[] = [];
  if (adapter.features.has("timestamp-query")) requestedFeatures.push("timestamp-query");
  const device = await adapter.requestDevice({
    requiredFeatures: requestedFeatures,
    requiredLimits: requiredFluidDeviceLimits(adapter.limits),
  });
  const validationErrors: string[] = [];
  device.addEventListener("uncapturederror", (event) => {
    validationErrors.push(event.error.message);
  });

  // Both are built before timing, and the order of advances alternates.
  uniform = await createArm(device, uniformMethod);
  sparse = await createArm(device, adaptiveMassMethod);
  for (let frame = 1; frame <= warmupFrames; frame += 1) {
    const targetTime_s = frame * dt_s;
    if (frame % 2 === 0) {
      await advanceOne(sparse, targetTime_s, false);
      await advanceOne(uniform, targetTime_s, false);
    } else {
      await advanceOne(uniform, targetTime_s, false);
      await advanceOne(sparse, targetTime_s, false);
    }
  }
  // Warmup traces are excluded just as construction is.
  uniform.cpuTraces.length = uniform.gpuTraces.length = 0;
  sparse.cpuTraces.length = sparse.gpuTraces.length = 0;

  for (let sample = 1; sample <= timedFrames; sample += 1) {
    const targetTime_s = (warmupFrames + sample) * dt_s;
    if (sample % 2 === 0) {
      await advanceOne(sparse, targetTime_s, true);
      await advanceOne(uniform, targetTime_s, true);
    } else {
      await advanceOne(uniform, targetTime_s, true);
      await advanceOne(sparse, targetTime_s, true);
    }
  }

  assert.equal(sparse.solver.info.hostFluidAuthority, "gpu-resident",
    "Sparse CM12 frame authority must remain GPU-resident");
  assert.equal(sparse.solver.info.hostSimulationSizedWorkItems, 0,
    "Sparse CM12 must not perform simulation-sized host work per frame");
  assert.equal(sparse.solver.info.hostSchedulingUsesReadback, false,
    "Sparse CM12 frame scheduling must not depend on GPU readback");
  if (sparseResolutionArgument === "all-fine") {
    assert.equal(sparse.solver.info.cellCount, uniform.solver.info.cellCount,
      "all-fine A/B must compare the same represented cell count");
  }

  const verdict = evaluateSparseCM12Performance(
    frozenArm(uniform),
    frozenArm(sparse),
    acceptance,
  );
  const report = {
    phase: "sparse-cm12-frame-performance",
    scene: scene.sceneId,
    finestDimensions: dimensions,
    dt_s,
    warmupFrames,
    timedFrames,
    sparseResolutionMode: sparseResolutionArgument,
    matchedRepresentation: {
      uniformCells: uniform.solver.info.cellCount,
      sparseLeaves: sparse.solver.info.cellCount,
      exactDegreesOfFreedomMatch:
        uniform.solver.info.cellCount === sparse.solver.info.cellCount,
    },
    validationErrors,
    stateGrowth: {
      uniform: summarizeFrameState(uniform.frameState),
      sparse: summarizeFrameState(sparse.frameState),
    },
    topologyAcceptance: {
      initial: sparse.initialTopology,
      evolvedMinimumMixedSeamRows: Math.min(...sparse.evolvedTopology.map(
        (sample) => sample.mixedSeamRows,
      )),
      evolvedMinimumFineCoarseFaceConnectedPairs: Math.min(
        ...sparse.evolvedTopology.map((sample) => sample.fineCoarseFaceConnectedPairs),
      ),
    },
    ...verdict,
  };
  console.log(JSON.stringify(report, null, 2));
  if (validationErrors.length > 0 || !verdict.passed) process.exitCode = 1;
} finally {
  uniform?.solver.destroy();
  sparse?.solver.destroy();
  await releaseWebGPUExclusiveLock();
}
