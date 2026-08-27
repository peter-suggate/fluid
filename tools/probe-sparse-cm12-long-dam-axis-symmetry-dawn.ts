import assert from "node:assert/strict";
import { pathToFileURL } from "node:url";

import { CM12_PAPER_DT_S } from "../lib/core/cm12-numerics";
import { resolveMethodValues } from "../lib/core/method-contract";
import {
  createSparseCM12LongDamBreakScene,
  SPARSE_CM12_LONG_DAM_METHOD_PROFILE,
} from "../lib/core/scenes";
import { managedGPUDevice } from "../lib/core/gpu-compilation-manager";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import {
  acquireWebGPUExclusiveLock,
  releaseWebGPUExclusiveLock,
} from "../lib/harness/webgpu-smoke-isolation";
import {
  adaptiveMassMethod,
  adaptiveMassSolverOptions,
} from "../lib/methods/adaptive-mass/method";
import {
  SPARSE_CM12_RESIDENT_STAGES,
  type SparseCM12DiagnosticFields,
  type SparseCM12ResidentStageId,
} from "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident";
import { WebGPUAdaptiveMassSolver } from
  "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";

const modulePath = process.env.WEBGPU_NODE_MODULE;
if (!modulePath) throw new Error("set WEBGPU_NODE_MODULE to Dawn's webgpu module");

const steps = Number(process.env.FLUID_LONG_DAM_SYMMETRY_STEPS ?? 10);
if (!Number.isSafeInteger(steps) || steps < 1) {
  throw new RangeError("FLUID_LONG_DAM_SYMMETRY_STEPS must be a positive integer");
}
const resolutionMode = process.env.FLUID_LONG_DAM_SYMMETRY_RESOLUTION ?? "adaptive";
if (resolutionMode !== "adaptive" && resolutionMode !== "all-fine") {
  throw new RangeError(
    "FLUID_LONG_DAM_SYMMETRY_RESOLUTION must be adaptive or all-fine",
  );
}
const requestedStageLimit = process.env.FLUID_LONG_DAM_SYMMETRY_STAGE_LIMIT;
if (requestedStageLimit !== undefined
  && !SPARSE_CM12_RESIDENT_STAGES.includes(
    requestedStageLimit as SparseCM12ResidentStageId,
  )) {
  throw new RangeError(`unknown Sparse CM12 stage ${requestedStageLimit}`);
}
const stageLimit = requestedStageLimit as SparseCM12ResidentStageId | undefined;
const requestedSharpeningPhase = process.env.FLUID_LONG_DAM_SYMMETRY_SHARPENING_PHASE;
if (requestedSharpeningPhase !== undefined
  && !["setup", "transform", "finalize", "capacity"].includes(
    requestedSharpeningPhase,
  )) {
  throw new RangeError(`unknown sharpening phase ${requestedSharpeningPhase}`);
}
const sharpeningPhase = requestedSharpeningPhase as
  "setup" | "transform" | "finalize" | "capacity" | undefined;
const requestedTransportPhase = process.env.FLUID_LONG_DAM_SYMMETRY_TRANSPORT_PHASE;
if (requestedTransportPhase !== undefined
  && !["setup", "trace", "scatter", "gather"].includes(requestedTransportPhase)) {
  throw new RangeError(`unknown transport phase ${requestedTransportPhase}`);
}
const transportPhase = requestedTransportPhase as
  "setup" | "trace" | "scatter" | "gather" | undefined;
const phase1QA = process.env.FLUID_LONG_DAM_SYMMETRY_PHASE1_QA === "1";
const statsQA = process.env.FLUID_LONG_DAM_SYMMETRY_STATS === "1";
const requestedMacroSpan = process.env.FLUID_LONG_DAM_SYMMETRY_MACRO_SPAN;
const maximumMacroSpanBricks = requestedMacroSpan === undefined
  ? undefined : Number(requestedMacroSpan);
if (maximumMacroSpanBricks !== undefined
  && (!Number.isSafeInteger(maximumMacroSpanBricks) || maximumMacroSpanBricks < 1
    || !Number.isInteger(Math.log2(maximumMacroSpanBricks)))) {
  throw new RangeError(
    "FLUID_LONG_DAM_SYMMETRY_MACRO_SPAN must be a positive power of two",
  );
}
const requestedSurfaceSharpening =
  process.env.FLUID_LONG_DAM_SYMMETRY_SURFACE_SHARPENING ?? "on";
if (requestedSurfaceSharpening !== "on" && requestedSurfaceSharpening !== "off") {
  throw new RangeError(
    "FLUID_LONG_DAM_SYMMETRY_SURFACE_SHARPENING must be on or off",
  );
}
const focusCell = (process.env.FLUID_LONG_DAM_SYMMETRY_FOCUS ?? "21,33,15")
  .split(",").map(Number);
if (focusCell.length !== 3 || focusCell.some((value) => !Number.isSafeInteger(value))) {
  throw new RangeError("FLUID_LONG_DAM_SYMMETRY_FOCUS must be x,y,z integers");
}

type Activity = Awaited<ReturnType<
  WebGPUAdaptiveMassSolver["readGPUActivityPolicy"]
>>;

type Metric = {
  comparedValues: number;
  nonFiniteCount: number;
  mismatchAboveToleranceCount: number;
  maximumAbsoluteError: number;
  squaredErrorSum: number;
  worst?: Readonly<Record<string, unknown>>;
};

function metric(): Metric {
  return { comparedValues: 0, nonFiniteCount: 0, mismatchAboveToleranceCount: 0,
    maximumAbsoluteError: 0, squaredErrorSum: 0 };
}

function observe(target: Metric, source: number, reflected: number, tolerance: number,
  detail: Readonly<Record<string, unknown>>): void {
  target.comparedValues += 1;
  if (!Number.isFinite(source) || !Number.isFinite(reflected)) {
    target.nonFiniteCount += 1;
    return;
  }
  const error = Math.abs(reflected - source);
  target.squaredErrorSum += error * error;
  if (error > tolerance) target.mismatchAboveToleranceCount += 1;
  if (error > target.maximumAbsoluteError) {
    target.maximumAbsoluteError = error;
    target.worst = Object.freeze({ ...detail, source, reflected, absoluteError: error });
  }
}

function finish(source: Metric) {
  const finite = Math.max(1, source.comparedValues - source.nonFiniteCount);
  return Object.freeze({ comparedValues: source.comparedValues,
    nonFiniteCount: source.nonFiniteCount,
    mismatchAboveToleranceCount: source.mismatchAboveToleranceCount,
    mismatchAboveToleranceFraction: source.mismatchAboveToleranceCount / finite,
    maximumAbsoluteError: source.maximumAbsoluteError,
    rmsError: Math.sqrt(source.squaredErrorSum / finite), worst: source.worst });
}

function xAxisReflection(fields: SparseCM12DiagnosticFields,
  dimensions: readonly [number, number, number]) {
  const [nx, ny, nz] = dimensions;
  const tolerance = Object.freeze({ density: 1e-5, velocity_m_s: 1e-5,
    densityWeightedVelocity_m_s: 1e-5, pressure: 1e-3, divergence_s: 1e-5,
    pressureRhs: 1e-5, solidOpenFraction: 1e-6 });
  const density = metric(), velocity = metric(), pressure = metric(), pressureRhs = metric();
  const divergence = metric();
  const densityWeightedVelocity = metric();
  const solidOpenFraction = metric(), sharpeningDelta = metric();
  const sharpeningReceiptMass = metric();
  for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1) {
    for (let x = 0; x < nx; x += 1) {
      const source = x + nx * (y + ny * z);
      const reflectedZ = nz - 1 - z;
      const target = x + nx * (y + ny * reflectedZ);
      const detail = { sourceCell: [x, y, z], reflectedCell: [x, y, reflectedZ] };
      observe(density, fields.density[source]!, fields.density[target]!, tolerance.density, detail);
      observe(pressure, fields.pressure[source]!, fields.pressure[target]!, tolerance.pressure,
        detail);
      observe(pressureRhs, fields.pressureRhs[source]!, fields.pressureRhs[target]!,
        tolerance.pressureRhs, detail);
      observe(divergence, fields.divergence[source]!, fields.divergence[target]!,
        tolerance.divergence_s, detail);
      observe(solidOpenFraction, fields.solidOpenFraction[source]!,
        fields.solidOpenFraction[target]!, tolerance.solidOpenFraction, detail);
      observe(sharpeningDelta, fields.sharpeningDelta[source]!,
        fields.sharpeningDelta[target]!, tolerance.density, detail);
      observe(sharpeningReceiptMass, fields.sharpeningReceiptMass[source]!,
        fields.sharpeningReceiptMass[target]!, tolerance.density, detail);
      for (let axis = 0; axis < 3; axis += 1) {
        const expected = (axis === 2 ? -1 : 1) * fields.velocity[4 * source + axis]!;
        observe(velocity, expected, fields.velocity[4 * target + axis]!, tolerance.velocity_m_s,
          { ...detail, axis, sourceValue: fields.velocity[4 * source + axis]!, expected });
        const expectedMomentum = expected * fields.density[source]!;
        const reflectedMomentum = fields.velocity[4 * target + axis]!
          * fields.density[target]!;
        observe(densityWeightedVelocity, expectedMomentum, reflectedMomentum,
          tolerance.densityWeightedVelocity_m_s,
          { ...detail, axis, sourceValue: expectedMomentum, reflectedMomentum });
      }
    }
  }
  return Object.freeze({ transform: "reflect-z-about-longitudinal-x-axis", tolerance,
    density: finish(density), velocity: finish(velocity), pressure: finish(pressure),
    pressureRhs: finish(pressureRhs),
    densityWeightedVelocity: finish(densityWeightedVelocity),
    divergence: finish(divergence), solidOpenFraction: finish(solidOpenFraction),
    sharpeningDelta: finish(sharpeningDelta),
    sharpeningReceiptMass: finish(sharpeningReceiptMass) });
}

function focusedPhysicalPair(fields: SparseCM12DiagnosticFields,
  dimensions: readonly [number, number, number],
  sourceCell: readonly [number, number, number] = [21, 33, 15]) {
  const [nx, ny, nz] = dimensions;
  const reflectedCell = [sourceCell[0], sourceCell[1], nz - 1 - sourceCell[2]] as const;
  const sample = (cell: readonly [number, number, number]) => {
    const at = cell[0] + nx * (cell[1] + ny * cell[2]);
    return Object.freeze({ cell,
      density: fields.density[at], gamma: fields.gamma[at],
      sharpeningDelta: fields.sharpeningDelta[at],
      sharpeningReceiptMass: fields.sharpeningReceiptMass[at],
      velocity_m_s: Array.from(fields.velocity.slice(4 * at, 4 * at + 3)),
      pressureRhs: fields.pressureRhs[at], pressure: fields.pressure[at],
      divergence_s: fields.divergence[at] });
  };
  return Object.freeze({ source: sample(sourceCell), reflected: sample(reflectedCell) });
}

function reflectedPhysicalResolution(
  activity: Activity,
  brickDimensions: readonly [number, number, number],
) {
  const byCoordinate = new Map(activity.bricks.map((brick) =>
    [brick.coordinate.join(","), brick] as const));
  const meanDensity = metric();
  let comparedBricks = 0;
  let missingCounterpartCount = 0;
  let activeMismatchCount = 0;
  let acceptedResolutionMismatchCount = 0;
  let firstMissingCounterpart: readonly number[] | undefined;
  let firstActiveMismatch: Readonly<Record<string, unknown>> | undefined;
  let firstAcceptedResolutionMismatch: Readonly<Record<string, unknown>> | undefined;
  for (const brick of activity.bricks) {
    const [x, y, z] = brick.coordinate;
    if (x < 0 || x >= brickDimensions[0] || y < 0 || y >= brickDimensions[1]
      || z < 0 || z >= brickDimensions[2]) continue;
    const reflectedCoordinate = [x, y, brickDimensions[2] - 1 - z] as const;
    const reflected = byCoordinate.get(reflectedCoordinate.join(","));
    if (!reflected) {
      missingCounterpartCount += 1;
      firstMissingCounterpart ??= brick.coordinate;
      continue;
    }
    comparedBricks += 1;
    const detail = { sourceBrick: brick.coordinate, reflectedBrick: reflectedCoordinate };
    observe(meanDensity, brick.meanDensity, reflected.meanDensity, 1e-6, detail);
    if (brick.active !== reflected.active) {
      activeMismatchCount += 1;
      firstActiveMismatch ??= { ...detail, source: brick.active, reflected: reflected.active };
    }
    if (brick.acceptedResolution !== reflected.acceptedResolution) {
      acceptedResolutionMismatchCount += 1;
      firstAcceptedResolutionMismatch ??= {
        ...detail, source: brick.acceptedResolution, reflected: reflected.acceptedResolution,
      };
    }
  }
  return Object.freeze({
    transform: "reflect-z-about-longitudinal-x-axis",
    brickDimensions,
    residentBrickCount: activity.residentBrickCount,
    comparedBricks,
    missingCounterpartCount,
    firstMissingCounterpart,
    activeMismatchCount,
    firstActiveMismatch,
    acceptedResolutionMismatchCount,
    firstAcceptedResolutionMismatch,
    meanDensity: finish(meanDensity),
  });
}

await acquireWebGPUExclusiveLock("dawn-acceptance",
  "tools/probe-sparse-cm12-long-dam-axis-symmetry-dawn.ts");
let device: GPUDevice | undefined;
let solver: WebGPUAdaptiveMassSolver | undefined;
try {
  const dawn = await import(pathToFileURL(modulePath).href) as {
    create(options: string[]): GPU;
    globals: Record<string, unknown>;
  };
  Object.assign(globalThis, dawn.globals);
  const gpu = dawn.create([
    `backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`,
    "enable-dawn-features=disable_blob_cache",
  ]);
  const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
  assert.ok(adapter, "Dawn must expose a WebGPU adapter");
  device = managedGPUDevice(await adapter.requestDevice({
    requiredLimits: requiredFluidDeviceLimits(adapter.limits),
  }), { requireWorkerRealm: false, maximumConcurrentBundles: 1 });
  const validationErrors: string[] = [];
  device.addEventListener("uncapturederror", (event) => {
    event.preventDefault(); validationErrors.push(event.error.message);
  });
  const scene = createSparseCM12LongDamBreakScene();
  const pressureRelativeTolerance = Number(
    process.env.FLUID_LONG_DAM_SYMMETRY_PRESSURE_TOLERANCE ?? 1e-3,
  );
  const values = resolveMethodValues(adaptiveMassMethod, "balanced",
    { ...SPARSE_CM12_LONG_DAM_METHOD_PROFILE.overrides, resolutionMode,
      pressureRelativeTolerance,
      maximumMacroSpanBricks: maximumMacroSpanBricks ?? "auto",
      surfaceSharpening: requestedSurfaceSharpening });
  solver = phase1QA
    ? await WebGPUAdaptiveMassSolver.createPhase1TransportReceiptOracleForQA(
      device, scene, "balanced", undefined, adaptiveMassSolverOptions(values), () => {},
    )
    : await adaptiveMassMethod.createSolverAsync!(
      device, scene, "balanced", values, undefined, () => {},
    ) as WebGPUAdaptiveMassSolver;
  await solver.waitForSimulationReady();
  const dimensions = [solver.info.nx, solver.info.ny, solver.info.nz] as const;
  assert.deepEqual(dimensions, [192, 96, 32]);
  const brickDimensions = dimensions.map((value) => value / 8) as unknown as
    readonly [number, number, number];
  const [initial, initialActivity] = await Promise.all([
    solver.readDiagnosticFields(true), solver.readGPUActivityPolicy(),
  ]);
  console.log(JSON.stringify({ phase: "sparse-cm12-long-dam-axis-symmetry", step: 0,
    time_s: 0, dt_s: CM12_PAPER_DT_S, resolutionMode, dimensions,
    maximumMacroSpanBricks: maximumMacroSpanBricks ?? "auto",
    surfaceSharpening: requestedSurfaceSharpening,
    physics: xAxisReflection(initial, dimensions),
    focus: focusedPhysicalPair(initial, dimensions,
      focusCell as unknown as readonly [number, number, number]),
    physicalResolution: reflectedPhysicalResolution(initialActivity, brickDimensions) }));
  for (let step = 1; step <= steps; step += 1) {
    const target_s = step * CM12_PAPER_DT_S;
    const stageLimitedFinalStep = step === steps
      && Boolean(stageLimit || sharpeningPhase || transportPhase);
    solver.sparseWorldTrace.setStageLimitForQA(step === steps ? stageLimit : undefined);
    solver.sparseWorldTrace.setSharpeningPhaseLimitForQA(
      step === steps ? sharpeningPhase : undefined,
    );
    solver.sparseWorldTrace.setTransportPhaseLimitForQA(
      step === steps ? transportPhase : undefined,
    );
    assert.equal(solver.advanceTo(target_s, []), true,
      `long-dam advance ${step} was not accepted`);
    await device.queue.onSubmittedWorkDone();
    const [fields, activity] = await Promise.all([
      solver.readDiagnosticFields(true, stageLimitedFinalStep ? "candidate" : "accepted"),
      solver.readGPUActivityPolicy(),
    ]);
    const transportReceipt = phase1QA
      ? await solver.readPhase1TransportReceiptQA(stageLimitedFinalStep) : undefined;
    const stats = statsQA ? await solver.readStats() : undefined;
    const velocityExtension = phase1QA ? await solver.readVelocityExtensionQA() : undefined;
    let betaWorstEffectiveVelocity: readonly Readonly<Record<string, unknown>>[] | undefined;
    if (transportReceipt && velocityExtension) {
      const worst = transportReceipt.reflectedZ.betaFixed.worst as {
        sourceContributions?: { receivers?: readonly Readonly<Record<string, unknown>>[] };
        reflectedContributions?: { receivers?: readonly Readonly<Record<string, unknown>>[] };
      } | undefined;
      const ids = new Map<number, Readonly<Record<string, unknown>>>();
      for (const receipt of [...(worst?.sourceContributions?.receivers ?? []),
        ...(worst?.reflectedContributions?.receivers ?? [])]) {
        if (Number(receipt.normalizedWeight) < 0.01) continue;
        ids.set(Number(receipt.receiver), receipt);
        if (receipt.reflectedReceiver !== undefined) {
          ids.set(Number(receipt.reflectedReceiver), {
            coordinate: receipt.reflectedCoordinate,
            reflectedOf: receipt.receiver,
          });
        }
      }
      const velocity = new Float32Array(velocityExtension.velocityBits.buffer,
        velocityExtension.velocityBits.byteOffset, velocityExtension.velocityBits.length);
      betaWorstEffectiveVelocity = Object.freeze(Array.from(ids, ([cell, receipt]) =>
        Object.freeze({ cell, ...receipt, acceptedDepth: velocityExtension.acceptedDepth[cell],
          velocityFineCells_s: Array.from(velocity.slice(4 * cell, 4 * cell + 3)),
          velocity_m_s: Array.from(velocity.slice(4 * cell, 4 * cell + 3),
            (value) => value * solver!.fluidDomain.cellSize_m[0]) })));
    }
    console.log(JSON.stringify({ phase: "sparse-cm12-long-dam-axis-symmetry", step,
      time_s: target_s, dt_s: CM12_PAPER_DT_S, resolutionMode,
      maximumMacroSpanBricks: maximumMacroSpanBricks ?? "auto",
      surfaceSharpening: requestedSurfaceSharpening,
      stageLimit: step === steps ? stageLimit : undefined,
      sharpeningPhase: step === steps ? sharpeningPhase : undefined, dimensions,
      transportPhase: step === steps ? transportPhase : undefined,
      physics: xAxisReflection(fields, dimensions),
      focus: focusedPhysicalPair(fields, dimensions,
        focusCell as unknown as readonly [number, number, number]),
      transportReceipt,
      betaWorstEffectiveVelocity,
      pressureSolve: stats ? {
        relativeResidual: stats.pressureRelativeResidual,
        initialTrueRelativeResidual: stats.pressureInitialTrueRelativeResidual,
        trueResidualMaximum: stats.pressureTrueResidualMaximum,
        iterationsExecuted: stats.pressureIterationsExecuted,
        iterationsEncoded: stats.pressureIterationsEncoded,
        convergenceReason: stats.pressureConvergenceReason,
        curvatureBreakdown: stats.pressureCurvatureBreakdown,
        residualDrift: stats.pressureResidualDrift,
      } : undefined,
      physicalResolution: reflectedPhysicalResolution(activity, brickDimensions) }));
  }
  assert.deepEqual(validationErrors, []);
} finally {
  solver?.destroy();
  device?.destroy();
  await releaseWebGPUExclusiveLock();
}
