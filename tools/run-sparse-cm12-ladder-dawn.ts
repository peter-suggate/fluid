import { spawn, spawnSync } from "node:child_process";
import { pathToFileURL } from "node:url";

import { resolveMethodValues } from "../lib/core/method-contract";
import { CM12_PAPER_DT_S } from "../lib/core/cm12-numerics";
import {
  getScenePreset,
  SPARSE_CM12_COMPLEXITY_LADDER_METHOD_PROFILE,
  SPARSE_CM12_COMPLEXITY_SCENES,
  type SparseCM12ComplexitySceneId,
} from "../lib/core/scenes";
import { managedGPUDevice } from "../lib/core/gpu-compilation-manager";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import {
  acquireWebGPUExclusiveLock,
  releaseWebGPUExclusiveLock,
} from "../lib/harness/webgpu-smoke-isolation";
import { adaptiveMassMethod } from "../lib/methods/adaptive-mass/method";
import { SPARSE_CM12_RESIDENT_STAGES, type SparseCM12ResidentStageId } from
  "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident";
import type { WebGPUAdaptiveMassSolver } from
  "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";

const isWorker = process.env.FLUID_CM12_LADDER_WORKER === "1";
const rung = (process.env.FLUID_CM12_LADDER_RUNG ?? "empty-16") as
  SparseCM12ComplexitySceneId;
const rungDefinition = SPARSE_CM12_COMPLEXITY_SCENES.find((entry) => entry.id === rung);
if (!rungDefinition) {
  throw new Error(`Unknown Sparse CM12 ladder rung ${rung}`);
}
const steps = Number(process.env.FLUID_CM12_LADDER_STEPS ?? 1);
if (!Number.isSafeInteger(steps) || steps < 0) {
  throw new RangeError("FLUID_CM12_LADDER_STEPS must be a non-negative integer");
}
const stageLimit = process.env.FLUID_CM12_LADDER_STAGE as
  SparseCM12ResidentStageId | undefined;
if (stageLimit && !SPARSE_CM12_RESIDENT_STAGES.includes(stageLimit)) {
  throw new Error(`Unknown Sparse CM12 frame stage ${stageLimit}`);
}
const stageLimitStep = Number(process.env.FLUID_CM12_LADDER_STAGE_STEP ?? 0);
if (!Number.isSafeInteger(stageLimitStep) || stageLimitStep < 0) {
  throw new RangeError("FLUID_CM12_LADDER_STAGE_STEP must be a non-negative integer");
}
const activityPhase = process.env.FLUID_CM12_LADDER_ACTIVITY_PHASE as
  "scalar" | "topology" | "masks" | "measure" | "history" | "census"
  | "allocation" | "synthesis" | "connection" | undefined;
if (activityPhase && !["scalar", "topology", "masks", "measure", "history",
  "census", "allocation", "synthesis", "connection"]
  .includes(activityPhase)) {
  throw new Error(`Unknown Sparse CM12 activity phase ${activityPhase}`);
}
const transportPhase = process.env.FLUID_CM12_LADDER_TRANSPORT_PHASE as
  "setup" | "trace" | "scatter" | "gather" | undefined;
if (transportPhase && !["setup", "trace", "scatter", "gather"].includes(transportPhase)) {
  throw new Error(`Unknown Sparse CM12 transport phase ${transportPhase}`);
}
const pressureTopologyPhase = process.env.FLUID_CM12_LADDER_PRESSURE_TOPOLOGY_PHASE as
  "setup" | "cells" | "rows" | "fine" | "coarse-plan" | "coarse-indirect"
  | "coarse-edge" | "coarse-work" | "coarse"
  | "hierarchy" | undefined;
if (pressureTopologyPhase && !["setup", "cells", "rows", "fine", "coarse-plan",
  "coarse-indirect", "coarse-edge", "coarse-work", "coarse", "hierarchy"]
  .includes(pressureTopologyPhase)) {
  throw new Error(`Unknown Sparse CM12 pressure-topology phase ${pressureTopologyPhase}`);
}

async function worker(): Promise<void> {
  await acquireWebGPUExclusiveLock("dawn-ladder", `Sparse CM12 ${rung}`);
  let device: GPUDevice | undefined;
  let solver: Awaited<ReturnType<NonNullable<
    typeof adaptiveMassMethod.createSolverAsync>>> | undefined;
  const started = performance.now();
  try {
    const modulePath = process.env.WEBGPU_NODE_MODULE
      ?? `${process.cwd()}/node_modules/webgpu/index.js`;
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
    if (!adapter) throw new Error("Dawn did not expose a WebGPU adapter");
    const rawDevice = await adapter.requestDevice({
      requiredLimits: requiredFluidDeviceLimits(adapter.limits),
    });
    device = managedGPUDevice(rawDevice, {
      requireWorkerRealm: false,
      maximumConcurrentBundles: 1,
    });
    let lost: GPUDeviceLostInfo | undefined;
    const validationErrors: string[] = [];
    void device.lost.then((info) => { lost = info; });
    device.addEventListener("uncapturederror", (event) => {
      event.preventDefault();
      validationErrors.push(event.error.message);
    });
    // Use the catalog document, not the raw preset body. This is the same
    // construction path as the UI and is where the ordinary SolidWorld voxel
    // shell is authored after the final lattice is known.
    const scene = getScenePreset(`sparse-cm12-ladder-${rung}`).create();
    const values = resolveMethodValues(adaptiveMassMethod, "balanced", {
      ...(SPARSE_CM12_COMPLEXITY_LADDER_METHOD_PROFILE.overrides ?? {}),
      ...(process.env.FLUID_CM12_LADDER_PRESSURE_ITERATIONS
        ? { pressureIterations: Number(
          process.env.FLUID_CM12_LADDER_PRESSURE_ITERATIONS) }
        : {}),
      ...(process.env.FLUID_CM12_LADDER_SURFACE_SHARPENING === "off"
        ? { surfaceSharpening: "off" as const } : {}),
    });
    const constructionStarted = performance.now();
    solver = await adaptiveMassMethod.createSolverAsync!(
      device, scene, "balanced", values, undefined, (progress) => {
        console.log(JSON.stringify({ event: "construction-progress", rung,
          ...progress }));
      },
    );
    await (solver as typeof solver & {
      waitForSimulationReady(): Promise<void>;
    }).waitForSimulationReady();
    await device.queue.onSubmittedWorkDone();
    console.log(JSON.stringify({ phase: "constructed", rung,
      wall_ms: performance.now() - constructionStarted,
      allocatedBytes: solver.info.allocatedBytes }));
    for (let step = 1; step <= steps; step += 1) {
      const frameStarted = performance.now();
      const frameStageLimit = stageLimitStep === 0 || stageLimitStep === step
        ? stageLimit : undefined;
      (solver as WebGPUAdaptiveMassSolver).sparseWorldTrace
        .setStageLimitForQA(frameStageLimit);
      (solver as WebGPUAdaptiveMassSolver).sparseWorldTrace
        .setActivityPhaseLimitForQA(activityPhase);
      (solver as WebGPUAdaptiveMassSolver).sparseWorldTrace
        .setTransportPhaseLimitForQA(stageLimitStep === 0 || stageLimitStep === step
          ? transportPhase : undefined);
      (solver as WebGPUAdaptiveMassSolver).sparseWorldTrace
        .setPressureTopologyPhaseLimitForQA(stageLimitStep === 0 || stageLimitStep === step
          ? pressureTopologyPhase : undefined);
      if (!solver.advanceTo(step * CM12_PAPER_DT_S, [])) {
        throw new Error(`solver refused ladder frame ${step}`);
      }
      await device.queue.onSubmittedWorkDone();
      if (lost) throw new Error(`device lost: ${lost.reason} ${lost.message}`);
      if (validationErrors.length > 0) {
        throw new Error(`validation: ${validationErrors.join("; ")}`);
      }
      console.log(JSON.stringify({ phase: "frame", rung, step,
        stageLimit: frameStageLimit ?? "complete",
        activityPhase: activityPhase ?? "complete",
        wall_ms: performance.now() - frameStarted,
        submittedTime_s: solver.info.submittedTime_s }));
    }
    const growth = await (solver as WebGPUAdaptiveMassSolver).readWorldGrowthReceiptQA();
    console.log(JSON.stringify({ phase: "world-growth", rung, ...growth }));
    if (rung !== "frontier-create" && rung !== "frontier-advance"
      && rung !== "retire-reuse") {
      const diagnostics = await (solver as WebGPUAdaptiveMassSolver)
        .sparseWorldTrace.readDiagnostics();
      console.log(JSON.stringify({ phase: "pressure", rung,
        relativeResidual: diagnostics.pressureRelativeResidual,
        iterationsExecuted: diagnostics.pressureIterationsExecuted,
        converged: diagnostics.pressureSolveConverged,
        reason: diagnostics.pressureConvergenceReason,
        curvatureBreakdown: diagnostics.pressureCurvatureBreakdown,
        curvatureRecoveryCount: diagnostics.pressureCurvatureRecoveryCount,
        recursiveToTrueResidualRatio: diagnostics.pressureRecursiveToTrueResidualRatio,
        residualDrift: diagnostics.pressureResidualDrift,
        maximumDivergence_s: diagnostics.maximumDivergence_s,
        maximumMixedSeamDivergence_s: diagnostics.maximumMixedSeamDivergence_s,
      }));
    }
    if (rung === "frontier-create" || rung === "frontier-advance"
      || rung === "retire-reuse") {
      const fields = await (solver as WebGPUAdaptiveMassSolver).readDiagnosticFields();
      const [nx, ny] = [8, 12];
      let samples = 0, wetSamples = 0;
      let densitySum = 0, velocityXSum = 0;
      let velocityXMinimum = Number.POSITIVE_INFINITY;
      let velocityXMaximum = Number.NEGATIVE_INFINITY;
      let pressureMinimum = Number.POSITIVE_INFINITY;
      let pressureMaximum = Number.NEGATIVE_INFINITY;
      let maximumDensity = 0, overfullDensityCellCount = 0;
      let maximumDensityToOpenFraction = 0, overCapacityCellCount = 0;
      let maximumOpenFraction = 0;
      let nonfiniteDensityCellCount = 0;
      for (let cell = 0; cell < fields.density.length; cell += 1) {
        const density = fields.density[cell]!;
        const openFraction = fields.solidOpenFraction[cell]!;
        if (!Number.isFinite(density)) nonfiniteDensityCellCount += 1;
        else {
          maximumDensity = Math.max(maximumDensity, density);
          overfullDensityCellCount += Number(density > 1.001);
          maximumOpenFraction = Math.max(maximumOpenFraction, openFraction);
          maximumDensityToOpenFraction = Math.max(maximumDensityToOpenFraction,
            density / Math.max(openFraction, 1e-12));
          overCapacityCellCount += Number(density > openFraction + 0.001);
        }
      }
      for (let z = 5; z < 11; z += 1) for (let y = 0; y < 4; y += 1) {
        const cell = 7 + nx * (y + ny * z);
        const density = fields.density[cell]!;
        const velocityX = fields.velocity[4 * cell]!;
        const pressure = fields.pressure[cell]!;
        samples += 1;
        wetSamples += Number(density >= 0.5);
        densitySum += density;
        velocityXSum += velocityX;
        velocityXMinimum = Math.min(velocityXMinimum, velocityX);
        velocityXMaximum = Math.max(velocityXMaximum, velocityX);
        pressureMinimum = Math.min(pressureMinimum, pressure);
        pressureMaximum = Math.max(pressureMaximum, pressure);
      }
      console.log(JSON.stringify({ phase: "frontier-seam-host", rung, samples, wetSamples,
        meanDensity: densitySum / samples, meanVelocityX_m_s: velocityXSum / samples,
        velocityXMinimum_m_s: velocityXMinimum, velocityXMaximum_m_s: velocityXMaximum,
        pressureMinimum_Pa: pressureMinimum, pressureMaximum_Pa: pressureMaximum,
        maximumDensity, overfullDensityCellCount, maximumOpenFraction,
        maximumDensityToOpenFraction, overCapacityCellCount,
        nonfiniteDensityCellCount }));
      const transportIndirect = await (solver as WebGPUAdaptiveMassSolver)
        .sparseWorldTrace.readTransportPacketIndirectQA();
      const persistentPressureCacheIndirect = await (solver as WebGPUAdaptiveMassSolver)
        .sparseWorldTrace.readPersistentPressureCacheIndirectQA();
      const candidateEffectsTransaction = await (solver as WebGPUAdaptiveMassSolver)
        .sparseWorldTrace.readCandidateEffectsTransactionQA();
      const pressureDiagnostics = await (solver as WebGPUAdaptiveMassSolver)
        .sparseWorldTrace.readDiagnostics();
      const dynamicTransportPackets = await (solver as WebGPUAdaptiveMassSolver)
        .sparseWorldTrace.readDynamicTransportPacketsQA();
      const finalScalarMasks = await (solver as WebGPUAdaptiveMassSolver)
        .sparseWorldTrace.readFinalScalarMaskHeaderQA();
      const velocityExtension = await (solver as WebGPUAdaptiveMassSolver)
        .sparseWorldTrace.readVelocityExtensionQA();
      const cellsPerPage = 8 ** 3;
      const dynamicCellBase = velocityExtension.acceptedDepth.length
        - cellsPerPage * (growth.capacity - growth.initialLeaves);
      let dynamicExtendedCells = 0;
      let dynamicMaximumAbsExtendedVelocityFineCells_s = 0;
      const velocityFloats = new Float32Array(velocityExtension.velocityBits.buffer,
        velocityExtension.velocityBits.byteOffset, velocityExtension.velocityBits.length);
      for (let cell = Math.max(0, dynamicCellBase);
        cell < velocityExtension.acceptedDepth.length; cell += 1) {
        if (velocityExtension.acceptedDepth[cell] !== 0xffff_ffff) {
          dynamicExtendedCells += 1;
        }
        dynamicMaximumAbsExtendedVelocityFineCells_s = Math.max(
          dynamicMaximumAbsExtendedVelocityFineCells_s,
          Math.abs(velocityFloats[4 * cell]!), Math.abs(velocityFloats[4 * cell + 1]!),
          Math.abs(velocityFloats[4 * cell + 2]!),
        );
      }
      console.log(JSON.stringify({ phase: "frontier-transport-authority", rung,
        transportIndirect, persistentPressureCacheIndirect, candidateEffectsTransaction,
        pressureTopologyRepair: pressureDiagnostics.pressureTopologyRepair,
        pressureCutoverAuthorities: pressureDiagnostics.pressureCutoverAuthorities,
        pressureSolve: {
          relativeResidual: pressureDiagnostics.pressureRelativeResidual,
          iterationsExecuted: pressureDiagnostics.pressureIterationsExecuted,
          converged: pressureDiagnostics.pressureSolveConverged,
          reason: pressureDiagnostics.pressureConvergenceReason,
          curvatureBreakdown: pressureDiagnostics.pressureCurvatureBreakdown,
          curvatureRecoveryCount: pressureDiagnostics.pressureCurvatureRecoveryCount,
          recursiveToTrueResidualRatio:
            pressureDiagnostics.pressureRecursiveToTrueResidualRatio,
          residualDrift: pressureDiagnostics.pressureResidualDrift,
          maximumDivergence_s: pressureDiagnostics.maximumDivergence_s,
          maximumMixedSeamDivergence_s:
            pressureDiagnostics.maximumMixedSeamDivergence_s,
        },
        dynamicTransportPackets,
        finalScalarMasks, dynamicExtendedCells,
        dynamicMaximumAbsExtendedVelocityFineCells_s,
        velocityExtensionDispatchPacketCount: velocityExtension.dispatchPacketCount,
        velocityExtensionCellCapacity: velocityExtension.acceptedDepth.length,
        dynamicCellBase }));
    }
    const activity = await (solver as WebGPUAdaptiveMassSolver).readGPUActivityPolicy();
    const activityRecords = rung === "frontier-create"
      ? (await (solver as WebGPUAdaptiveMassSolver).sparseWorldTrace
        .readActivitySnapshot()).records : [];
    console.log(JSON.stringify({ phase: "activity", rung,
      acceptedTopologyGeneration: activity.acceptedTopologyGeneration,
      faultFlags: activity.faultFlags,
      newlyActivatedBrickCount: activity.newlyActivatedBrickCount,
      preparedBrickCount: activity.preparedBrickCount,
      committedBrickCount: activity.committedBrickCount,
      commitFailed: activity.commitFailed,
      ...(rung === "frontier-create" ? { records: activityRecords.map((record, brick) => ({
        brick, active: record.active, reasons: record.reasons,
        plannedResolution: record.plannedResolution,
        acceptedResolution: record.acceptedResolution,
        candidateResolution: record.candidateResolution,
        candidateStatus: record.candidateStatus,
        planReasons: record.planReasons,
      })) } : {}) }));
    if (activity.commitFailed) {
      const [candidateEffectsTransaction, diagnostics, frameControl, snapshot] =
        await Promise.all([
        (solver as WebGPUAdaptiveMassSolver).sparseWorldTrace
          .readCandidateEffectsTransactionQA(),
        (solver as WebGPUAdaptiveMassSolver).sparseWorldTrace.readDiagnostics(),
        (solver as WebGPUAdaptiveMassSolver).sparseWorldTrace.readFrameControlQA(),
        (solver as WebGPUAdaptiveMassSolver).sparseWorldTrace.readActivitySnapshot(),
      ]);
      console.log(JSON.stringify({ phase: "candidate-failure", rung,
        candidateEffectsTransaction,
        pressureTopologyRepair: diagnostics.pressureTopologyRepair,
        pressureCutoverAuthorities: diagnostics.pressureCutoverAuthorities,
        frameControl,
        candidates: snapshot.records.map((record, brick) => ({ brick,
          active: record.active, plannedResolution: record.plannedResolution,
          acceptedResolution: record.acceptedResolution,
          candidateResolution: record.candidateResolution,
          candidateStatus: record.candidateStatus,
          planReasons: record.planReasons,
        })).filter((record) => record.candidateStatus !== 0
          || record.plannedResolution !== record.acceptedResolution),
      }));
    }
    console.log(JSON.stringify({ phase: "complete", rung, steps,
      wall_ms: performance.now() - started }));
  } finally {
    solver?.destroy();
    device?.destroy();
    await releaseWebGPUExclusiveLock();
  }
}

if (isWorker) {
  await worker();
} else {
  const timeout_ms = Number(process.env.FLUID_CM12_LADDER_TIMEOUT_MS ?? 45_000);
  if (!Number.isSafeInteger(timeout_ms) || timeout_ms < 1_000) {
    throw new RangeError("FLUID_CM12_LADDER_TIMEOUT_MS must be at least 1000");
  }
  console.error("SAFETY: every browser WebGPU tab must be closed during this isolated Dawn rung.");
  const child = spawn(process.execPath, ["--import", "tsx", import.meta.filename], {
    cwd: process.cwd(),
    env: { ...process.env, FLUID_CM12_LADDER_WORKER: "1" },
    stdio: "inherit",
  });
  const monitor = setInterval(() => {
    if (!child.pid) return;
    const rss = spawnSync("ps", ["-o", "rss=", "-p", String(child.pid)], {
      encoding: "utf8",
    }).stdout.trim();
    if (rss) console.error(JSON.stringify({ phase: "resource", rung,
      childPid: child.pid, rss_kib: Number(rss) }));
  }, 1_000);
  let timedOut = false;
  let forcedExit: ReturnType<typeof setTimeout> | undefined;
  const deadline = setTimeout(() => {
    timedOut = true;
    clearInterval(monitor);
    console.error(`Sparse CM12 ${rung} exceeded ${timeout_ms} ms; terminating isolated Dawn child`);
    child.kill("SIGTERM");
    setTimeout(() => {
      child.kill("SIGKILL");
      // A wedged Metal/Dawn child may remain in the macOS exiting state even
      // after SIGKILL. Bound the supervisor too; retain the worker-owned lock
      // so no later GPU run can overlap a process the OS has not reaped.
      forcedExit = setTimeout(() => {
        console.error(`Sparse CM12 ${rung} child was not reaped after SIGKILL; leaving the exclusive GPU lock in place`);
        child.unref();
        process.exit(124);
      }, 2_000);
      forcedExit.unref();
    }, 2_000).unref();
  }, timeout_ms);
  const exitCode = await new Promise<number>((resolve, reject) => {
    child.once("error", reject);
    child.once("exit", (code, signal) => resolve(signal ? 1 : code ?? 1));
  });
  clearInterval(monitor);
  clearTimeout(deadline);
  if (forcedExit) clearTimeout(forcedExit);
  process.exitCode = timedOut ? 124 : exitCode;
}
