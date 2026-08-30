import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";

import { CM12_PAPER_DT_S } from "../lib/core/cm12-numerics";
import { cloneScene } from "../lib/core/model";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import {
  acquireWebGPUExclusiveLock,
  releaseWebGPUExclusiveLock,
} from "../lib/harness/webgpu-smoke-isolation";
import { WebGPUAdaptiveMassSolver } from
  "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";
import {
  SPARSE_CM12_ACTIVITY_POLICY,
  SPARSE_CM12_SHARPENING_DISTANCE_CELLS,
  SPARSE_CM12_SHARPENING_TRACE_STEPS,
} from "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident";
import type { AdaptiveMassSolverOptions } from
  "../lib/methods/adaptive-mass/method";

const dawnModule = process.env.WEBGPU_NODE_MODULE;
const dawnTest = dawnModule ? test : test.skip;

function fieldMetrics(fields: Awaited<ReturnType<
  WebGPUAdaptiveMassSolver["readDiagnosticFields"]>>): Readonly<{
    maximumSpeedFineCells_s: number;
    rmsSpeedFineCells_s: number;
    kineticEnergyFineUnits: number;
    maximumDivergence_s: number;
    rmsDivergence_s: number;
    densityVariation: number;
    maximumSpeedCell: number;
    maximumSpeedVelocity: readonly [number, number, number];
    rmsVelocityByAxis: readonly [number, number, number];
    maximumSpeedCellState: Readonly<{
      density: number; open: number; pressure: number; rhs: number; diagonal: number;
      divergence: number;
    }>;
  }> {
  let maximumSpeedFineCells_s = 0;
  let speedSquared = 0;
  let kineticEnergyFineUnits = 0;
  let maximumDivergence_s = 0;
  let divergenceSquared = 0;
  let minimumWetDensity = Number.POSITIVE_INFINITY;
  let maximumWetDensity = Number.NEGATIVE_INFINITY;
  let wet = 0;
  let maximumSpeedCell = -1;
  let maximumSpeedVelocity: [number, number, number] = [0, 0, 0];
  const velocitySquaredByAxis = [0, 0, 0];
  for (let cell = 0; cell < fields.density.length; cell += 1) {
    const density = fields.density[cell]!;
    if (density <= 0.05) continue;
    const vx = fields.velocity[4 * cell]!;
    const vy = fields.velocity[4 * cell + 1]!;
    const vz = fields.velocity[4 * cell + 2]!;
    const squared = vx * vx + vy * vy + vz * vz;
    const divergence = Math.abs(fields.divergence[cell]!);
    const speed = Math.sqrt(squared);
    if (speed > maximumSpeedFineCells_s) {
      maximumSpeedFineCells_s = speed;
      maximumSpeedCell = cell;
      maximumSpeedVelocity = [vx, vy, vz];
    }
    velocitySquaredByAxis[0]! += vx * vx;
    velocitySquaredByAxis[1]! += vy * vy;
    velocitySquaredByAxis[2]! += vz * vz;
    speedSquared += squared;
    kineticEnergyFineUnits += density * squared;
    maximumDivergence_s = Math.max(maximumDivergence_s, divergence);
    divergenceSquared += divergence * divergence;
    minimumWetDensity = Math.min(minimumWetDensity, density);
    maximumWetDensity = Math.max(maximumWetDensity, density);
    wet += 1;
  }
  return {
    maximumSpeedFineCells_s,
    rmsSpeedFineCells_s: Math.sqrt(speedSquared / Math.max(1, wet)),
    kineticEnergyFineUnits,
    maximumDivergence_s,
    rmsDivergence_s: Math.sqrt(divergenceSquared / Math.max(1, wet)),
    densityVariation: wet === 0 ? 0 : maximumWetDensity - minimumWetDensity,
    maximumSpeedCell,
    maximumSpeedVelocity,
    rmsVelocityByAxis: velocitySquaredByAxis.map((value) =>
      Math.sqrt(value / Math.max(1, wet))) as [number, number, number],
    maximumSpeedCellState: maximumSpeedCell < 0 ? {
      density: 0, open: 0, pressure: 0, rhs: 0, diagonal: 0, divergence: 0,
    } : {
      density: fields.density[maximumSpeedCell]!,
      open: fields.solidOpenFraction[maximumSpeedCell]!,
      pressure: fields.pressure[maximumSpeedCell]!,
      rhs: fields.pressureRhs[maximumSpeedCell]!,
      diagonal: fields.pressureDiagonal[maximumSpeedCell]!,
      divergence: fields.divergence[maximumSpeedCell]!,
    },
  };
}

dawnTest("Sparse CM12 hydrostatic equilibrium damps a microscopic gravity perturbation", {
  timeout: 240_000,
}, async () => {
  await acquireWebGPUExclusiveLock("dawn-test",
    "tests/sparse-cm12-hydrostatic-perturbation-dawn.test.ts");
  let device: GPUDevice | undefined;
  let solver: WebGPUAdaptiveMassSolver | undefined;
  try {
    const dawn = await import(pathToFileURL(dawnModule!).href) as {
      create(options: string[]): GPU;
      globals: Record<string, unknown>;
    };
    Object.assign(globalThis, dawn.globals);
    const gpu = dawn.create([
      `backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`,
      "enable-dawn-features=disable_blob_cache",
    ]);
    const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
    assert.ok(adapter);
    device = await adapter.requestDevice({
      requiredLimits: requiredFluidDeviceLimits(adapter.limits),
    });
    const validationErrors: string[] = [];
    device.addEventListener("uncapturederror", (event) => {
      event.preventDefault();
      validationErrors.push(event.error.message);
    });

    const arm = process.env.FLUID_HYDROSTATIC_PERTURB_ARM ?? "fine";
    const sceneId = process.env.FLUID_HYDROSTATIC_SCENE
      ?? "hydrostatic-power-two-level";
    const scene = sceneDocument(getSceneDefinition(sceneId));
    if (arm === "fine-zero-gravity") {
      scene.fluid.gravity_m_s2 = { x: 0, y: 0, z: 0 };
    }
    if (arm.startsWith("fine")) {
      scene.fluid.refinementRegions = [{
        id: "hydrostatic-fixed-b8",
        rule: "minimum-cell-size",
        minimumCellSize_cells: 1,
        maximumCellSize_cells: 1,
        min_m: { x: -0.5 * scene.container.width_m, y: 0,
          z: -0.5 * scene.container.depth_m },
        max_m: { x: 0.5 * scene.container.width_m, y: scene.container.height_m,
          z: 0.5 * scene.container.depth_m },
      }];
    }
    const options: AdaptiveMassSolverOptions = {
      resolutionMode: arm === "adaptive" ? "adaptive" : "all-fine",
      brickFineResolution: 8,
      surfaceFineRings: 1,
      timeStep: arm === "fine-scene-step" ? "scene" : "paper",
      activityPolicy: { ...SPARSE_CM12_ACTIVITY_POLICY, activitySignals: true,
        prepareBricksPerFrame: 256 },
      pressureIterations: 128,
      pressureRelativeTolerance: 0,
      sharpeningDistance: SPARSE_CM12_SHARPENING_DISTANCE_CELLS,
      sharpeningTraceSteps: SPARSE_CM12_SHARPENING_TRACE_STEPS,
      surfaceSharpeningEnabled: arm !== "fine-sharpen-off",
      gammaDiffusionEnabled: arm !== "fine-gamma-off",
    };
    solver = await WebGPUAdaptiveMassSolver.createAsync(
      device, scene, "balanced", undefined, options, () => {},
    );
    await solver.waitForSimulationReady();
    if (arm.startsWith("fine")) solver.setForcedSurfaceResolutionForQA(8);

    const settleSteps = Number(process.env.FLUID_HYDROSTATIC_SETTLE_STEPS ?? 10);
    const observeSteps = Number(process.env.FLUID_HYDROSTATIC_OBSERVE_STEPS ?? 10);
    const sampleSteps = Number(process.env.FLUID_HYDROSTATIC_SAMPLE_STEPS ?? 1);
    const relativeGravityChange = Number(
      process.env.FLUID_HYDROSTATIC_GRAVITY_EPSILON ?? 1e-6,
    );
    const trace = process.env.FLUID_HYDROSTATIC_PERTURB_TRACE === "1";
    let perturbed = false;
    let transitionCount = 0;
    const accepted = new Map<string, number>();
    const samples: Array<ReturnType<typeof fieldMetrics>> = [];
    const perturbedSamples: Array<ReturnType<typeof fieldMetrics>> = [];
    const pressureActiveRowCounts: number[] = [];
    const topologyGenerations: number[] = [];
    const surfaceScores: number[] = [];

    for (let step = 1; step <= settleSteps + observeSteps; step += 1) {
      if (!perturbed && step === settleSteps + 1) {
        const changed = cloneScene(scene);
        changed.fluid.gravity_m_s2.y *= 1 + relativeGravityChange;
        solver.applySceneUniforms(changed);
        perturbed = true;
      }
      assert.equal(solver.advanceTo(step * CM12_PAPER_DT_S, []), true);
      if (step % sampleSteps !== 0 && step !== settleSteps
        && step !== settleSteps + 1) continue;
      await device.queue.onSubmittedWorkDone();
      const [fields, candidateFields, activity, stats, frameControl,
        frameControlIndirect] = await Promise.all([
        solver.readDiagnosticFields(),
        process.env.FLUID_HYDROSTATIC_BANK_TRACE === "1"
          ? solver.readDiagnosticFields(false, "candidate") : Promise.resolve(undefined),
        solver.readGPUActivityPolicy(), solver.readStats(),
        trace ? solver.readFrameControlQA() : Promise.resolve(undefined),
        trace ? solver.readFrameControlIndirectQA() : Promise.resolve(undefined),
      ]);
      const metrics = fieldMetrics(fields);
      samples.push(metrics);
      if (step > settleSteps) perturbedSamples.push(metrics);
      if (stats.adaptivePressureActiveRowCount !== undefined) {
        pressureActiveRowCounts.push(stats.adaptivePressureActiveRowCount);
      }
      topologyGenerations.push(activity.acceptedTopologyGeneration);
      const transitions: Array<{ coordinate: string; from: number; to: number }> = [];
      for (const brick of activity.bricks.filter((candidate) => candidate.active)) {
        const coordinate = brick.coordinate.join(",");
        const previous = accepted.get(coordinate);
        if (previous !== undefined && previous !== brick.acceptedResolution) {
          transitions.push({ coordinate, from: previous, to: brick.acceptedResolution });
          transitionCount += 1;
        }
        accepted.set(coordinate, brick.acceptedResolution);
      }
      const surface = activity.bricks.filter((brick) => brick.active
        && (brick.reasons & 1) !== 0);
      surfaceScores.push(Math.max(0, ...surface.map((brick) => brick.scoreByte)));
      if (trace) {
        process.stderr.write(`[hydrostatic-perturbation] ${JSON.stringify({
          arm, step, time_s: step * CM12_PAPER_DT_S,
          phase: step <= settleSteps ? "settle" : "perturbed",
          relativeGravityChange, ...metrics,
          candidateMetrics: candidateFields ? fieldMetrics(candidateFields) : undefined,
          pressureRelativeResidual: stats.pressureRelativeResidual,
          pressureIterations: stats.pressureIterationsExecuted,
          pressureInitialRelativeResidual: stats.pressureInitialTrueRelativeResidual,
          pressureCanonicalMembership: stats.adaptivePressureCanonicalMembership,
          pressureTopologyRepair: stats.adaptivePressureTopologyRepair,
          pressureTopologyAttribution: stats.adaptivePressureTopologyAttribution,
          activityMeasuredBrickCount: stats.adaptiveActivityMeasuredBrickCount,
          frameControl,
          frameControlIndirect,
          topologyGeneration: activity.acceptedTopologyGeneration,
          residentBricks: activity.residentBrickCount,
          newlyActivatedBricks: activity.newlyActivatedBrickCount,
          preparedBricks: activity.preparedBrickCount,
          committedBricks: activity.committedBrickCount,
          surfaceB4: surface.filter((brick) => brick.acceptedResolution === 4).length,
          surfaceB8: surface.filter((brick) => brick.acceptedResolution === 8).length,
          surfaceMaximumTravel: Math.max(0, ...surface.map((brick) =>
            brick.maximumVelocityTravelFineCells)),
          surfaceMaximumScore: Math.max(0, ...surface.map((brick) => brick.scoreByte)),
          surfaceReasonMask: surface.reduce((mask, brick) => mask | brick.reasons, 0),
          transitions,
        })}\n`);
      }
    }

    assert.ok(samples.every((sample) => Number.isFinite(sample.maximumSpeedFineCells_s)
      && Number.isFinite(sample.rmsSpeedFineCells_s)
      && Number.isFinite(sample.kineticEnergyFineUnits)
      && Number.isFinite(sample.maximumDivergence_s)),
      `non-finite hydrostatic metrics in ${arm}`);
    const activity = await solver.readGPUActivityPolicy();
    assert.equal(activity.faultFlags, 0);
    assert.equal(activity.commitFailed, false);
    assert.deepEqual(validationErrors, []);
    if (arm === "fine" && sceneId === "hydrostatic-power-two-level") {
      assert.equal(transitionCount, 0,
        "the fixed-B8 hydrostatic control must not publish topology changes");
      assert.equal(new Set(topologyGenerations).size, 1,
        "the fixed-B8 hydrostatic control changed accepted topology");
      assert.equal(new Set(pressureActiveRowCounts).size, 1,
        `hydrostatic pressure-row membership changed: ${pressureActiveRowCounts.join(",")}`);
      assert.ok(perturbedSamples.length > 0);
      assert.ok(Math.max(...perturbedSamples.map((sample) =>
        sample.maximumSpeedFineCells_s)) < 1e-4,
      `microscopic gravity perturbation generated macroscopic motion: ${JSON.stringify(
        perturbedSamples.at(-1))}`);
      assert.ok(Math.max(...samples.map((sample) => sample.densityVariation)) < 1e-6,
        "the controlled hydrostatic density field changed");
      assert.ok(Math.max(0, ...surfaceScores) <= 1,
        `stationary surface remained activity-hot: ${surfaceScores.join(",")}`);
    }
    if (trace) process.stderr.write(`[hydrostatic-perturbation-summary] ${JSON.stringify({
      arm, relativeGravityChange, transitionCount, final: samples.at(-1),
    })}\n`);
  } finally {
    solver?.destroy();
    device?.destroy();
    await releaseWebGPUExclusiveLock();
  }
});
