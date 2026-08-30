import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";

import { CM12_PAPER_DT_S } from "../lib/core/cm12-numerics";
import { resolveMethodValues } from "../lib/core/method-contract";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import {
  acquireWebGPUExclusiveLock,
  releaseWebGPUExclusiveLock,
} from "../lib/harness/webgpu-smoke-isolation";
import { adaptiveMassMethod } from "../lib/methods/adaptive-mass/method";
import type { WebGPUAdaptiveMassSolver } from
  "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";

const dawnModule = process.env.WEBGPU_NODE_MODULE;
const dawnTest = dawnModule ? test : test.skip;

function furthestLiquidX(density: Float32Array, dimensions: readonly number[]): number {
  const [nx, ny, nz] = dimensions;
  let furthest = -1;
  for (let z = 0; z < nz!; z += 1) for (let y = 0; y < ny!; y += 1) {
    for (let x = 0; x < nx!; x += 1) {
      if (density[x + nx! * (y + ny! * z)]! >= 0.05) furthest = Math.max(furthest, x);
    }
  }
  return furthest;
}

dawnTest("mini32 keeps its moving front fine and later consumes settled proofs", {
  timeout: 240_000,
}, async () => {
  await acquireWebGPUExclusiveLock("dawn-test",
    "tests/sparse-cm12-mini32-surface-coarsening-dawn.test.ts");
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
    assert.ok(adapter, "Dawn must expose a WebGPU adapter");
    device = await adapter.requestDevice({
      requiredLimits: requiredFluidDeviceLimits(adapter.limits),
    });
    const validationErrors: string[] = [];
    device.addEventListener("uncapturederror", (event) => {
      event.preventDefault();
      validationErrors.push(event.error.message);
    });

    const scene = sceneDocument(getSceneDefinition("minimal-power-dam-break-32"));
    const values = resolveMethodValues(adaptiveMassMethod, "balanced", {
      brickFineResolution: "8",
      resolutionMode: "adaptive",
      selectorMode: "activity",
      surfaceFineRings: 1,
      topologyCadenceSteps: 1,
      timeStep: "paper",
      ...(process.env.FLUID_MINI32_SURFACE_SHARPENING === "off"
        ? { surfaceSharpening: "off" } : {}),
      ...(process.env.FLUID_MINI32_GAMMA_DIFFUSION === "off"
        ? { gammaDiffusion: "off" } : {}),
    });
    solver = await adaptiveMassMethod.createSolverAsync!(
      device, scene, "balanced", values, undefined, () => {},
    ) as WebGPUAdaptiveMassSolver;
    await solver.waitForSimulationReady();
    const forceFine = process.env.FLUID_MINI32_SURFACE_FORCE_FINE === "1";
    if (forceFine) solver.setForcedSurfaceResolutionForQA(8);
    const initialFields = await solver.readDiagnosticFields();
    const initialFront = furthestLiquidX(initialFields.density,
      [solver.info.nx, solver.info.ny, solver.info.nz]);

    const duration_s = Number(process.env.FLUID_MINI32_SURFACE_DURATION_S ?? 6);
    const sampleSteps = Number(process.env.FLUID_MINI32_SURFACE_SAMPLE_STEPS ?? 15);
    const trace = process.env.FLUID_MINI32_SURFACE_TRACE === "1";
    const steps = Math.round(duration_s / CM12_PAPER_DT_S);
    const frontHistory: string[] = [];
    const surfaceHistory = new Map<string, Array<Readonly<{
      step: number;
      accepted: number;
      planned: number;
      reasons: number;
      velocityTravel: number;
      proofEpochs: number;
      representable: number;
      proofFailure: number;
      meanDensity: number;
    }>>>();
    let movingFrontFineSurfaces = 0;
    for (let step = 1; step <= steps; step += 1) {
      assert.equal(solver.advanceTo(step * CM12_PAPER_DT_S, []), true,
        `advance ${step}`);
      if (step % sampleSteps === 0) {
        await device.queue.onSubmittedWorkDone();
        const [fields, snapshot] = await Promise.all([
          solver.readDiagnosticFields(), solver.readGPUActivityPolicy(),
        ]);
        const representedSurface = snapshot.bricks.filter((brick) => brick.active
          && (brick.reasons & 1) !== 0);
        if (step === sampleSteps) movingFrontFineSurfaces = representedSurface.filter((brick) =>
          brick.acceptedResolution === 8).length;
        frontHistory.push(`${(step * CM12_PAPER_DT_S).toFixed(1)}s:x${
          furthestLiquidX(fields.density,
            [solver.info.nx, solver.info.ny, solver.info.nz])}/B4${
          representedSurface.filter((brick) => brick.acceptedResolution === 4).length
          }/B8${representedSurface.filter((brick) => brick.acceptedResolution === 8).length
          }/v${Math.max(0, ...representedSurface.map((brick) =>
            brick.maximumVelocityTravelFineCells)).toFixed(2)}`);
        for (const brick of representedSurface) {
          const key = brick.coordinate.join(",");
          const history = surfaceHistory.get(key) ?? [];
          history.push({ step, accepted: brick.acceptedResolution,
            planned: brick.plannedResolution, reasons: brick.reasons,
            velocityTravel: brick.maximumVelocityTravelFineCells,
            proofEpochs: brick.surfaceProofEpochs,
            representable: brick.representableNextResolution ?? 0,
            proofFailure: brick.representabilityFailure,
            meanDensity: brick.meanDensity });
          surfaceHistory.set(key, history);
        }
        if (trace) {
          const transitions = [...surfaceHistory].flatMap(([coordinate, history]) => {
            const previous = history.at(-2);
            const current = history.at(-1)!;
            return previous && previous.accepted !== current.accepted
              ? [{ coordinate, from: previous.accepted, to: current.accepted,
                ...current }] : [];
          });
          process.stderr.write(`[mini32-surface] ${JSON.stringify({
            time_s: step * CM12_PAPER_DT_S,
            front: furthestLiquidX(fields.density,
              [solver.info.nx, solver.info.ny, solver.info.nz]),
            surface: representedSurface.map((brick) => ({
              coordinate: brick.coordinate, accepted: brick.acceptedResolution,
              planned: brick.plannedResolution, reasons: brick.reasons,
              velocityTravel: brick.maximumVelocityTravelFineCells,
              proofEpochs: brick.surfaceProofEpochs,
              representable: brick.representableNextResolution ?? 0,
              proofFailure: brick.representabilityFailure,
              meanDensity: brick.meanDensity,
            })),
            transitions,
          })}\n`);
        }
      }
    }
    await device.queue.onSubmittedWorkDone();

    const [activity, finalFields] = await Promise.all([
      solver.readGPUActivityPolicy(), solver.readDiagnosticFields(),
    ]);
    const finalFront = furthestLiquidX(finalFields.density,
      [solver.info.nx, solver.info.ny, solver.info.nz]);
    assert.ok(finalFront >= 31,
      `mini32 dam front stalled: x=${initialFront} -> ${finalFront}; ${
        frontHistory.join(" ")}`);
    assert.ok(movingFrontFineSurfaces > 0,
      `mini32 coarsened its complete accelerating surface by 0.5 s; ${
        frontHistory.join(" ")}`);
    const surface = activity.bricks.filter((brick) => brick.active
      && (brick.reasons & 1) !== 0);
    assert.ok(surface.length > 0, "mini32 must retain a represented surface");
    const coarse = surface.filter((brick) => brick.acceptedResolution === 4);
    assert.deepEqual(validationErrors, []);
    assert.equal(activity.faultFlags, 0);
    assert.equal(activity.commitFailed, false);
    if (!forceFine) {
      assert.ok(coarse.length > 0,
        `mini32 never consumed a settled B4 proof by ${duration_s} seconds; ${
          frontHistory.join(" ")}`);
    }
  } finally {
    solver?.destroy();
    device?.destroy();
    await releaseWebGPUExclusiveLock();
  }
});
