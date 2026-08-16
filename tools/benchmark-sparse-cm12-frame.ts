/**
 * Matched serialized-frame performance gate for Uniform CM12 vs Sparse CM12.
 *
 * Run:
 *   NODE_OPTIONS=--max-old-space-size=8192 \
 *   WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js \
 *   node --import tsx tools/benchmark-sparse-cm12-frame.ts
 *
 * Construction and warmup are excluded. Arms use the same authored scene,
 * 32x16x32 finest lattice and matched step, and alternate order to reduce
 * drift. The hard ratio is defined beside the adaptive method implementation.
 */
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { fileURLToPath, pathToFileURL } from "node:url";
import type {
  GPUSolverInstance,
  MethodParamValues,
  SimulationMethod,
} from "../lib/core/method-contract";
import { resolveMethodValues } from "../lib/core/method-contract";
import { CM12_PAPER_DT_S } from "../lib/core/cm12-numerics";
import type { PerformanceTrace } from "../lib/core/performance-trace";
import {
  createMinimalPowerDamBreak32Scene,
  createMinimalPowerDamBreak64Scene,
  createSparseCM12LongDamBreakScene,
  createSymmetricExpansionScene,
  SPARSE_CM12_LONG_DAM_METHOD_PROFILE,
} from "../lib/core/scenes";
import { usePerformanceInstrumentationStore } from "../lib/core/stores/performance-instrumentation-store";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import {
  acquireWebGPUExclusiveLock,
  releaseWebGPUExclusiveLock,
} from "../lib/harness/webgpu-smoke-isolation";
import { adaptiveMassMethod } from "../lib/methods/adaptive-mass/method";
import { WebGPUAdaptiveMassSolver } from
  "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";
import {
  evaluateSparseCM12Performance,
  SPARSE_CM12_MINI_DAM_32_PERFORMANCE_ACCEPTANCE,
  SPARSE_CM12_LONG_DAM_PERFORMANCE_ACCEPTANCE,
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

const optionalPositiveInteger = (name: string): number | undefined => {
  const prefix = `--${name}=`;
  const inline = process.argv.slice(2).find((value) => value.startsWith(prefix));
  if (!inline) return undefined;
  const value = Number(inline.slice(prefix.length));
  if (!Number.isSafeInteger(value) || value < 1) {
    throw new RangeError(`${name} must be a positive integer`);
  }
  return value;
};

const warmupFrames = positiveInteger("warmup", 5);
const timedFrames = positiveInteger("frames", 40);
const prepareBricksPerFrame = optionalPositiveInteger("prepare-bricks");
const surfaceFineRings = optionalPositiveInteger("surface-fine-rings");
const sceneArgument = process.argv.slice(2)
  .find((value) => value.startsWith("--scene="))?.slice("--scene=".length)
  ?? "symmetric";
if (sceneArgument !== "symmetric" && sceneArgument !== "mini32"
  && sceneArgument !== "mini64"
  && sceneArgument !== "long-dam") {
  throw new RangeError("scene must be symmetric, mini32, mini64, or long-dam");
}
const sparseResolutionArgument = process.argv.slice(2)
  .find((value) => value.startsWith("--sparse-resolution="))
  ?.slice("--sparse-resolution=".length) ?? "all-fine";
if (sparseResolutionArgument !== "all-fine" && sparseResolutionArgument !== "adaptive") {
  throw new RangeError("sparse-resolution must be all-fine or adaptive");
}
const timeStepArgument = process.argv.slice(2)
  .find((value) => value.startsWith("--time-step="))?.slice("--time-step=".length)
  ?? "scene";
if (timeStepArgument !== "scene" && timeStepArgument !== "paper") {
  throw new RangeError("time-step must be scene or paper");
}
const buildScene = sceneArgument === "mini32" || sceneArgument === "mini64"
  ? sceneArgument === "mini64"
    ? createMinimalPowerDamBreak64Scene
    : createMinimalPowerDamBreak32Scene
  : sceneArgument === "long-dam"
    ? createSparseCM12LongDamBreakScene
    : createSymmetricExpansionScene;
const acceptance = sceneArgument === "mini32" || sceneArgument === "mini64"
  ? sceneArgument === "mini64"
    ? { ...SPARSE_CM12_MINI_DAM_32_PERFORMANCE_ACCEPTANCE,
      sceneId: "minimal-power-dam-break-64",
      finestDimensions: [64, 64, 64] as const }
    : SPARSE_CM12_MINI_DAM_32_PERFORMANCE_ACCEPTANCE
  : sceneArgument === "long-dam"
    ? SPARSE_CM12_LONG_DAM_PERFORMANCE_ACCEPTANCE
    : SPARSE_CM12_PERFORMANCE_ACCEPTANCE;
const scene = buildScene();
const dimensions = acceptance.finestDimensions;
const dt_s = timeStepArgument === "paper"
  ? CM12_PAPER_DT_S : scene.numerics.fixedDt_s ?? scene.numerics.maxDt_s;

interface FrameStateSample {
  readonly time_s: number;
  readonly duration_ms: number;
  readonly activeCells: number;
  readonly residentBricks: number;
  readonly mixedSeams: number;
  readonly pressureIterations: number;
  readonly maximumCfl: number;
  readonly topologyPrepared: number;
  readonly topologyCommitted: number;
  readonly topologyDeferred: number;
  readonly fineBricks: number;
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

async function readDensity(
  device: GPUDevice,
  solver: GPUSolverInstance,
): Promise<Float32Array> {
  if (solver instanceof WebGPUAdaptiveMassSolver) {
    return (await solver.readDiagnosticFields()).density;
  }
  const [nx, ny, nz] = dimensions;
  const bytesPerRow = Math.ceil(nx * 4 / 256) * 256;
  const readback = device.createBuffer({
    label: "Sparse CM12 long-dam final density readback",
    size: bytesPerRow * ny * nz,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  try {
    const encoder = device.createCommandEncoder();
    encoder.copyTextureToBuffer(
      { texture: solver.volumeTexture },
      { buffer: readback, bytesPerRow, rowsPerImage: ny },
      dimensions,
    );
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const source = new Float32Array(readback.getMappedRange());
    const stride = bytesPerRow / 4;
    const density = new Float32Array(nx * ny * nz);
    for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1) {
      density.set(source.subarray(stride * (y + ny * z), stride * (y + ny * z) + nx),
        nx * (y + ny * z));
    }
    return density;
  } finally {
    if (readback.mapState === "mapped") readback.unmap();
    readback.destroy();
  }
}

function frontReceipt(density: Float32Array) {
  const [nx, ny, nz] = dimensions;
  const columnMass = new Array<number>(nx).fill(0);
  const furthest = { above1e3: -1, above005: -1, liquid: -1 };
  for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1)
    for (let x = 0; x < nx; x += 1) {
      const rho = density[x + nx * (y + ny * z)]!;
      columnMass[x] += rho;
      if (rho > 1e-3) furthest.above1e3 = Math.max(furthest.above1e3, x);
      if (rho > 0.05) furthest.above005 = Math.max(furthest.above005, x);
      if (rho >= 0.5) furthest.liquid = Math.max(furthest.liquid, x);
    }
  return {
    furthestFineCellX: furthest,
    occupiedColumnMass: columnMass.map((mass, x) => ({ x, mass }))
      .filter(({ mass }) => mass > 1e-3),
  };
}

function matchedDensityReceipt(uniform: Float32Array, sparse: Float32Array) {
  let differenceL1 = 0, differenceSquared = 0;
  let referenceL1 = 0, referenceSquared = 0, maximumAbsolute = 0;
  for (let index = 0; index < uniform.length; index += 1) {
    const difference = sparse[index]! - uniform[index]!;
    differenceL1 += Math.abs(difference);
    differenceSquared += difference * difference;
    referenceL1 += Math.abs(uniform[index]!);
    referenceSquared += uniform[index]! * uniform[index]!;
    maximumAbsolute = Math.max(maximumAbsolute, Math.abs(difference));
  }
  return {
    relativeL1: differenceL1 / Math.max(Number.MIN_VALUE, referenceL1),
    relativeL2: Math.sqrt(differenceSquared
      / Math.max(Number.MIN_VALUE, referenceSquared)),
    maximumAbsolute,
  };
}

function rawFloatHash(values: Float32Array) {
  return {
    elements: values.length,
    bytes: values.byteLength,
    sha256: createHash("sha256").update(new Uint8Array(
      values.buffer, values.byteOffset, values.byteLength,
    )).digest("hex"),
  };
}

function pressureStages(stages: Readonly<Record<string, {
  readonly samples: number; readonly median_ms: number; readonly p95_ms: number;
}>>) {
  return Object.fromEntries(Object.entries(stages).filter(([label]) =>
    label.startsWith("pressure-system:") || label.startsWith("pressure-solve:")
      || label.startsWith("velocity-projection:")));
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
      topologyPrepared: info.adaptiveTopologyPreparedBrickCount ?? 0,
      topologyCommitted: info.adaptiveTopologyCommittedBrickCount ?? 0,
      topologyDeferred: info.adaptiveTopologyDeferredBrickCount ?? 0,
      fineBricks: info.adaptiveFineBrickCount ?? 0,
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
    ? { timeStep: timeStepArgument, densityPostProcessing: "off" }
    : {
      ...(sceneArgument === "long-dam"
        ? SPARSE_CM12_LONG_DAM_METHOD_PROFILE.overrides : {}),
      timeStep: timeStepArgument,
      resolutionMode: sparseResolutionArgument,
      ...(prepareBricksPerFrame === undefined ? {} : { prepareBricksPerFrame }),
      ...(surfaceFineRings === undefined ? {} : { surfaceFineRings }),
    };
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
    topologyPrepared: summary((sample) => sample.topologyPrepared),
    topologyCommitted: summary((sample) => sample.topologyCommitted),
    topologyDeferred: summary((sample) => sample.topologyDeferred),
    fineBricks: summary((sample) => sample.fineBricks),
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

  const performanceAcceptance = sparseResolutionArgument === "all-fine" ? {
    sceneId: acceptance.sceneId,
    finestDimensions: acceptance.finestDimensions,
    dt_s: acceptance.dt_s,
    maximumMedianFrameRatio: acceptance.maximumMedianFrameRatio,
    targetMedianFrameRatio: acceptance.targetMedianFrameRatio,
    minimumTimedFrames: acceptance.minimumTimedFrames,
    constructionExcluded: acceptance.constructionExcluded,
  } : acceptance;
  const verdict = evaluateSparseCM12Performance(
    frozenArm(uniform),
    frozenArm(sparse),
    performanceAcceptance,
  );
  // QA readbacks occur strictly after the timed range. They fingerprint the
  // exact state associated with the pressure/control receipt without charging
  // either arm's frame timing or feeding any scheduling decision.
  const [uniformDensity, sparseDensity, finalUniformInfo, finalSparseInfo] =
    await Promise.all([
      readDensity(device, uniform.solver),
      readDensity(device, sparse.solver),
      uniform.solver.readStats(),
      sparse.solver.readStats(),
    ]);
  const sceneFront = {
    uniform: frontReceipt(uniformDensity),
    sparse: frontReceipt(sparseDensity),
  };
  const longDamFront = sceneArgument === "long-dam" ? {
    ...sceneFront,
    sparseResidentMaximumFineCellX: Math.max(
      ...((await (sparse.solver as WebGPUAdaptiveMassSolver)
        .readGPUActivityPolicy()).bricks.filter((brick) => brick.active)
        .map((brick) => 8 * (brick.coordinate[0] + 1) - 1)),
    ),
  } : undefined;
  const frontFailures: string[] = [];
  if (longDamFront
    && longDamFront.uniform.furthestFineCellX.above005
      > longDamFront.sparseResidentMaximumFineCellX
    && longDamFront.sparse.furthestFineCellX.above005
      <= longDamFront.sparseResidentMaximumFineCellX) {
    frontFailures.push(
      `Sparse CM12 front is pinned at resident boundary x=${
        longDamFront.sparseResidentMaximumFineCellX}; Uniform reached x=${
        longDamFront.uniform.furthestFineCellX.above005}`,
    );
  }
  if (longDamFront
    && longDamFront.uniform.furthestFineCellX.liquid === dimensions[0] - 1
    && longDamFront.sparse.furthestFineCellX.liquid !== dimensions[0] - 1) {
    frontFailures.push(
      `Sparse CM12 liquid front stopped at x=${
        longDamFront.sparse.furthestFineCellX.liquid}; the matched Uniform front reached the far wall`,
    );
  }
  const densityAgreement = matchedDensityReceipt(uniformDensity, sparseDensity);
  const forcedFineDensityAgreement = sparseResolutionArgument === "all-fine"
    ? densityAgreement : undefined;
  const performanceGateApplied = timeStepArgument === "scene";
  const agreementFailures: string[] = [];
  if (forcedFineDensityAgreement && forcedFineDensityAgreement.relativeL1 > 0.05) {
    agreementFailures.push(
      `forced-all-fine Sparse CM12 density relative L1 ${
        forcedFineDensityAgreement.relativeL1} exceeds 0.05 against Uniform`,
    );
  }
  const passed = (!performanceGateApplied || verdict.passed) && frontFailures.length === 0
    && agreementFailures.length === 0;
  const report = {
    phase: "sparse-cm12-frame-performance",
    scene: scene.sceneId,
    finestDimensions: dimensions,
    dt_s,
    warmupFrames,
    timedFrames,
    timeStep: timeStepArgument,
    performanceGateApplied,
    sparseResolutionMode: sparseResolutionArgument,
    topologyExperiment: {
      prepareBricksPerFrame: prepareBricksPerFrame ?? "method-default",
      surfaceFineRings: surfaceFineRings ?? "method-default",
    },
    matchedRepresentation: {
      uniformCells: uniform.solver.info.cellCount,
      uniformCoordinateFaceRows: (dimensions[0] - 1) * dimensions[1] * dimensions[2]
        + dimensions[0] * (dimensions[1] - 1) * dimensions[2]
        + dimensions[0] * dimensions[1] * (dimensions[2] - 1),
      sparseConstructionLeaves: sparse.solver.info.cellCount,
      sparseAcceptedCells: finalSparseInfo.adaptiveAcceptedCellCount,
      sparseAcceptedRows: finalSparseInfo.adaptiveAcceptedRowCount,
      exactDegreesOfFreedomMatch:
        uniform.solver.info.cellCount === finalSparseInfo.adaptiveAcceptedCellCount,
    },
    exactStateHashes: {
      uniformDensity: rawFloatHash(uniformDensity),
      sparseDensity: rawFloatHash(sparseDensity),
    },
    densityAgreement,
    pressureControl: {
      uniform: {
        solver: finalUniformInfo.pressureSolver,
        iterations: finalUniformInfo.pressureIterations,
        cpuStages: pressureStages(verdict.uniform.cpuStages),
        gpuStages: pressureStages(verdict.uniform.gpuStages),
      },
      sparse: {
        solver: finalSparseInfo.pressureSolver,
        iterations: finalSparseInfo.pressureIterations,
        relativeResidual: finalSparseInfo.pressureRelativeResidual,
        maximumDivergence_s: finalSparseInfo.maxDivergenceAfter_s,
        acceptedCells: finalSparseInfo.adaptiveAcceptedCellCount,
        acceptedRows: finalSparseInfo.adaptiveAcceptedRowCount,
        constructionLeafCount: sparse.solver.info.cellCount,
        cpuStages: pressureStages(verdict.sparse.cpuStages),
        gpuStages: pressureStages(verdict.sparse.gpuStages),
      },
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
    sceneFront,
    longDamFront,
    forcedFineDensityAgreement,
    ...verdict,
    passed,
    targetMet: (!performanceGateApplied || verdict.targetMet) && frontFailures.length === 0
      && agreementFailures.length === 0,
    failures: [...(performanceGateApplied ? verdict.failures : []),
      ...frontFailures, ...agreementFailures],
  };
  console.log(JSON.stringify(report, null, 2));
  if (validationErrors.length > 0 || !passed) process.exitCode = 1;
} finally {
  uniform?.solver.destroy();
  sparse?.solver.destroy();
  await releaseWebGPUExclusiveLock();
}
