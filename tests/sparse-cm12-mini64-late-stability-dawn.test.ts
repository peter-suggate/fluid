import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";

import { CM12_PAPER_DT_S } from "../lib/core/cm12-numerics";
import { createMinimalPowerDamBreak64Scene } from "../lib/core/scenes";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from
  "../lib/harness/webgpu-smoke-isolation";
import { WebGPUAdaptiveMassSolver } from
  "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";

const dawnModule = process.env.WEBGPU_NODE_MODULE;
const dawnTest = dawnModule ? test : test.skip;

function densityReceipt(density: Float32Array) {
  let mass = 0;
  let maximum = 0;
  let maximumIndex = 0;
  let nonFinite = 0;
  for (let index = 0; index < density.length; index += 1) {
    const value = density[index]!;
    if (!Number.isFinite(value)) nonFinite += 1;
    mass += Math.max(0, Number.isFinite(value) ? value : 0);
    if (Number.isFinite(value) && value > maximum) {
      maximum = value;
      maximumIndex = index;
    }
  }
  const edge = Math.round(Math.cbrt(density.length));
  return { mass, maximum, maximumCoordinate: [maximumIndex % edge,
    Math.floor(maximumIndex / edge) % edge,
    Math.floor(maximumIndex / (edge * edge))], nonFinite };
}

dawnTest("Sparse CM12 mini64 remains bounded through the late wall-impact window",
  { timeout: 600_000 }, async () => {
    await acquireWebGPUExclusiveLock("dawn-test",
      "tests/sparse-cm12-mini64-late-stability-dawn.test.ts");
    let device: GPUDevice | undefined;
    let solver: WebGPUAdaptiveMassSolver | undefined;
    try {
      const dawn = await import(pathToFileURL(dawnModule!).href) as {
        create(options: string[]): GPU;
        globals: Record<string, unknown>;
      };
      Object.assign(globalThis, dawn.globals);
      const gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
      const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
      assert.ok(adapter, "Dawn must expose a WebGPU adapter");
      device = await adapter.requestDevice({
        requiredLimits: requiredFluidDeviceLimits(adapter.limits),
      });
      const validationErrors: string[] = [];
      device.addEventListener("uncapturederror", (event) => {
        event.preventDefault();
        validationErrors.push(event.error.message);
      });

      const scene = createMinimalPowerDamBreak64Scene();
      scene.duration_s = 3;
      const paperTimeStep = process.env.FLUID_MINI64_LATE_PAPER_DT === "1";
      solver = await WebGPUAdaptiveMassSolver.createAsync(
        device, scene, "balanced", undefined, {
          resolutionMode: process.env.FLUID_MINI64_LATE_ALL_FINE === "1"
            ? "all-fine" : "adaptive",
          brickFineResolution: 8,
          timeStep: paperTimeStep ? "paper" : "scene",
          gammaDiffusionEnabled:
            process.env.FLUID_MINI64_LATE_NO_CONDITIONING === "1" ? false : undefined,
          surfaceSharpeningEnabled:
            process.env.FLUID_MINI64_LATE_NO_CONDITIONING === "1" ? false : undefined,
        }, () => {},
      );
      await solver.waitForSimulationReady();
      const initialDensity = densityReceipt((await solver.readDiagnosticFields()).density);
      const initialWorld = await solver.readWorldGrowthReceiptQA();
      const initialMass = initialDensity.mass + initialWorld.dynamicLiquidMassFineCells;
      let peakDensity = initialDensity.maximum;
      let peakVelocityTravel = 0;
      let peakDynamicFaceVelocity = 0;
      let peakRelativeMassError = 0;
      let peakDivergence_s = 0;
      let pressureCurvatureBreakdown = false;
      let failedHostIncidences = 0;
      const requestedSteps = Number(process.env.FLUID_MINI64_LATE_STEPS);
      const steps = Number.isSafeInteger(requestedSteps) && requestedSteps > 0
        ? requestedSteps : 135;
      const requestedSampleEvery = Number(process.env.FLUID_MINI64_LATE_SAMPLE_EVERY);
      const sampleEvery = Number.isSafeInteger(requestedSampleEvery)
        && requestedSampleEvery > 0 ? requestedSampleEvery
        : process.env.FLUID_MINI64_LATE_FINE_TRACE === "1" ? 1 : 45;
      const stepDt_s = paperTimeStep ? CM12_PAPER_DT_S : scene.numerics.maxDt_s;
      for (let step = 1; step <= steps; step += 1) {
        assert.equal(solver.advanceTo(step * stepDt_s, []), true);
        if (step % 2 === 0) await device.queue.onSubmittedWorkDone();
        if (step % sampleEvery !== 0 && step !== steps) continue;
        await device.queue.onSubmittedWorkDone();
        const [fields, world, activity, stats] = await Promise.all([
          solver.readDiagnosticFields(), solver.readWorldGrowthReceiptQA(),
          solver.readGPUActivityPolicy(), solver.readStats(),
        ]);
        const density = densityReceipt(fields.density);
        peakDensity = Math.max(peakDensity, density.maximum);
        peakVelocityTravel = Math.max(peakVelocityTravel,
          ...activity.bricks.map((brick) => brick.maximumVelocityTravelFineCells));
        peakDynamicFaceVelocity = Math.max(peakDynamicFaceVelocity,
          world.dynamicMaximumAbsFaceVelocityFineCells_s);
        peakRelativeMassError = Math.max(peakRelativeMassError, Math.abs(
          (density.mass + world.dynamicLiquidMassFineCells) / initialMass - 1));
        peakDivergence_s = Math.max(peakDivergence_s, stats.maxDivergenceAfter_s ?? 0);
        pressureCurvatureBreakdown ||= Boolean(stats.pressureCurvatureBreakdown);
        failedHostIncidences = Math.max(failedHostIncidences,
          world.failedHostIncidences);
        if (process.env.FLUID_MINI64_LATE_TRACE === "1") {
          const authoredCoordinates = new Set(activity.bricks.map((brick) =>
            brick.coordinate.join(",")));
          const expectedConnectedHostIncidences = 64
            * world.publishedTopologyPageCoordinates.reduce((count, coordinate) =>
              count + [[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0],
                [0, 0, -1], [0, 0, 1]].filter((offset) =>
                authoredCoordinates.has(coordinate.map((value, axis) =>
                  value + offset[axis]!).join(","))).length, 0);
          process.stderr.write(`${JSON.stringify({ step,
            time_s: step * stepDt_s,
            mass: density.mass + world.dynamicLiquidMassFineCells,
            relativeMass: (density.mass + world.dynamicLiquidMassFineCells) / initialMass,
            maximumDensity: density.maximum,
            maximumDensityCoordinate: density.maximumCoordinate,
            nonFiniteDensity: density.nonFinite,
            maximumVelocityTravelFineCells: Math.max(...activity.bricks.map(
              (brick) => brick.maximumVelocityTravelFineCells)),
            dynamicMaximumAbsFaceVelocityFineCells_s:
              world.dynamicMaximumAbsFaceVelocityFineCells_s,
            pressureRelativeResidual: stats.pressureRelativeResidual,
            maximumDivergence_s: stats.maxDivergenceAfter_s,
            faultFlags: activity.faultFlags,
            worldLeaves: `${world.liveLeaves}/${world.capacity}`,
            worldCapacityFaults: world.capacityFaults,
            connectedHostIncidences: world.connectedHostIncidences,
            failedHostIncidences: world.failedHostIncidences,
            expectedConnectedHostIncidences,
            topologyGeneration: activity.acceptedTopologyGeneration,
            newlyActivated: activity.newlyActivatedBrickCount,
            prepared: activity.preparedBrickCount,
            committed: activity.committedBrickCount,
            commitFailed: activity.commitFailed,
            pressureIterations: stats.pressureIterationsExecuted,
            pressureConverged: stats.pressureSolveConverged,
            pressureCapReached: stats.pressureIterationCapReached,
            pressureReason: stats.pressureConvergenceReason,
            pressureCurvatureBreakdown: stats.pressureCurvatureBreakdown,
            pressureCurvatureRecoveryCount: stats.pressureCurvatureRecoveryCount,
            pressureResidualDrift: stats.pressureResidualDrift,
            pressureRecursiveToTrueResidualRatio:
              stats.pressureRecursiveToTrueResidualRatio,
            pressureTopology: stats.adaptivePressureTopologyAttribution,
            pressureRepair: stats.adaptivePressureTopologyRepair,
            pressureMembership: stats.adaptivePressureCanonicalMembership,
          })}\n`);
        }
      }
      assert.deepEqual(validationErrors, []);
      assert.ok(Number.isFinite(peakDensity));
      assert.ok(Number.isFinite(peakVelocityTravel));
      assert.ok(Number.isFinite(peakDynamicFaceVelocity));
      if (!paperTimeStep) {
        assert.equal(failedHostIncidences, 0,
          "every recycled dynamic page must recover the canonical host incidence");
        assert.equal(pressureCurvatureBreakdown, false,
          "the late host-page topology must keep the pressure operator SPD");
        assert.ok(peakDivergence_s < 1,
          `late projection divergence grew to ${peakDivergence_s}/s`);
        assert.ok(peakRelativeMassError < 0.01,
          `represented mass drifted by ${100 * peakRelativeMassError}%`);
        assert.ok(peakDensity < 2.1,
          `scene-step density spiked to ${peakDensity}`);
        assert.ok(peakVelocityTravel < 2,
          `late velocity travelled ${peakVelocityTravel} fine cells`);
      }
    } finally {
      solver?.destroy();
      device?.destroy();
      await releaseWebGPUExclusiveLock();
    }
  });
