import assert from "node:assert/strict";
import { fileURLToPath, pathToFileURL } from "node:url";
import "../lib/methods";
import { getScenePreset } from "../lib/core/scenes";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { adaptiveMassMethod } from "../lib/methods/adaptive-mass/method";
import type { WebGPUAdaptiveMassSolver } from
  "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";
import {
  acquireWebGPUExclusiveLock,
  releaseWebGPUExclusiveLock,
} from "../lib/harness/webgpu-smoke-isolation";

const requestedSteps = Number(process.env.FLUID_SOLID_WORLD_CM12_STEPS ?? 2);
assert.ok(Number.isSafeInteger(requestedSteps) && requestedSteps >= 0);
const brickFineResolution = Number(process.env.FLUID_SOLID_WORLD_CM12_BRICK_FINE ?? 8);
assert.ok(brickFineResolution === 4 || brickFineResolution === 8
  || brickFineResolution === 16);
const checkpoints = new Set([0, 1, 2, 4, 8, 16, 24, requestedSteps]
  .filter((step) => step <= requestedSteps));

await acquireWebGPUExclusiveLock("dawn-benchmark", "probe-water-box-solid-world-cm12");
let device: GPUDevice | undefined;
try {
  const modulePath = process.env.WEBGPU_NODE_MODULE
    ?? fileURLToPath(new URL("../node_modules/webgpu/index.js", import.meta.url));
  const dawn = await import(pathToFileURL(modulePath).href) as {
    create(options: string[]): GPU;
    globals: Record<string, unknown>;
  };
  Object.assign(globalThis, dawn.globals);
  const gpu = dawn.create([
    `backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`,
    ...(process.env.FLUID_WEBGPU_ADAPTER
      ? [`adapter=${process.env.FLUID_WEBGPU_ADAPTER}`] : []),
  ]);
  Object.defineProperty(globalThis, "navigator", { configurable: true, value: { gpu } });
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
  const scene = getScenePreset("water-box-dam-break").create();
  const shellFaces = ["yLow", "xLow", "xHigh", "zLow", "zHigh", "yHigh"] as const;
  const values = {
    ...adaptiveMassMethod.presetFor("balanced"),
    resolutionMode: "adaptive",
    brickFineResolution: String(brickFineResolution),
    presentationPageResolution: String(brickFineResolution),
    surfaceFineRings: 1,
    timeStep: "paper",
  };
  const solver = await adaptiveMassMethod.createSolverAsync!(
    device, scene, "balanced", values, undefined, () => {},
  ) as WebGPUAdaptiveMassSolver;
  try {
    await solver.waitForSimulationReady();
    const [nx, ny, nz] = [solver.info.nx, solver.info.ny, solver.info.nz];
    assert.deepEqual([nx, ny, nz], [24, 16, 16]);
    const sample = async (step: number) => {
      await device!.queue.onSubmittedWorkDone();
      const [activity, fields, stats] = await Promise.all([
        solver.readGPUActivityPolicy(), solver.readDiagnosticFields(), solver.readStats(),
      ]);
      const active = activity.bricks.filter((brick) => brick.active);
      const outsideBricks = active.filter((brick) => brick.coordinate.some(
        (coordinate, axis) => coordinate < 0
          || coordinate >= Math.ceil([nx, ny, nz][axis]! / brickFineResolution),
      ));
      const faceWet = (coordinate: readonly number[], face: string) => {
        const lo = coordinate.map((value) => value * brickFineResolution);
        const hi = lo.map((value, axis) => Math.min(
          [nx, ny, nz][axis]!, value + brickFineResolution));
        for (let z = lo[2]!; z < hi[2]!; z += 1)
          for (let y = lo[1]!; y < hi[1]!; y += 1)
            for (let x = lo[0]!; x < hi[0]!; x += 1) {
              if ((face === "xLow" && x !== 0) || (face === "xHigh" && x !== nx - 1)
                || (face === "yLow" && y !== 0) || (face === "yHigh" && y !== ny - 1)
                || (face === "zLow" && z !== 0) || (face === "zHigh" && z !== nz - 1)) {
                continue;
              }
              if (fields.density[x + nx * (y + ny * z)]! > 0.01) return true;
            }
        return false;
      };
      const walls = Object.fromEntries(shellFaces.map((face) => {
        const touching = active.filter((brick) => {
          const [x, y, z] = brick.coordinate;
          return (face === "xLow" && x === 0)
            || (face === "xHigh" && x === Math.ceil(nx / brickFineResolution) - 1)
            || (face === "yLow" && y === 0)
            || (face === "yHigh" && y === Math.ceil(ny / brickFineResolution) - 1)
            || (face === "zLow" && z === 0)
            || (face === "zHigh" && z === Math.ceil(nz / brickFineResolution) - 1);
        }).filter((brick) => faceWet(brick.coordinate, face));
        return [face, {
          wetBricks: touching.length,
          coarseWetBricks: touching.filter((brick) =>
            brick.acceptedResolution < brickFineResolution).length,
          resolutions: touching.map((brick) => brick.acceptedResolution),
        }];
      }));
      return {
        step,
        topologyGeneration: activity.acceptedTopologyGeneration,
        faults: activity.faultFlags,
        activeBricks: active.length,
        outsideBricks: outsideBricks.map((brick) => brick.coordinate),
        resolutionCounts: Object.fromEntries([1, 2, 4, 8].map((resolution) => [resolution,
          active.filter((brick) => brick.acceptedResolution === resolution).length])),
        walls,
        massFineCells: fields.density.reduce((sum, density) => sum + density, 0),
        representedVolumeDrift: stats.representedVolumeDrift,
        pressureSolveConverged: stats.pressureSolveConverged,
        pressureIterationCapReached: stats.pressureIterationCapReached,
        maximumDivergence_s: stats.maxDivergence_s,
      };
    };
    const report = [];
    report.push(await sample(0));
    const dt_s = 1 / 30;
    for (let step = 1; step <= requestedSteps; step += 1) {
      assert.equal(solver.advanceTo(step * dt_s, []), true, `advance ${step}`);
      if (checkpoints.has(step)) report.push(await sample(step));
    }
    await device.queue.onSubmittedWorkDone();
    console.log(JSON.stringify({ phase: "water-box-solid-world-sparse-cm12", report,
      validationErrors }));
    const initialMassFineCells = report[0]!.massFineCells;
    for (const receipt of report) {
      assert.equal(receipt.faults, 0, `topology faults at step ${receipt.step}`);
      assert.deepEqual(receipt.outsideBricks, [],
        `SolidWorld shell leaked resident pages at step ${receipt.step}`);
      assert.ok(receipt.walls.yLow.wetBricks > 0,
        `step ${receipt.step} must retain wet SolidWorld floor contact`);
      assert.ok(receipt.pressureSolveConverged,
        `pressure must converge at step ${receipt.step}`);
      assert.equal(receipt.pressureIterationCapReached, false,
        `pressure cap reached at step ${receipt.step}`);
      assert.equal(receipt.representedVolumeDrift, 0,
        `represented volume drift at step ${receipt.step}`);
      assert.ok(Math.abs(receipt.massFineCells - initialMassFineCells)
        / initialMassFineCells < 1e-5,
      `materialized mass drift at step ${receipt.step}`);
    }
    assert.deepEqual(validationErrors, []);
  } finally {
    solver.destroy();
    await device.queue.onSubmittedWorkDone();
  }
} finally {
  device?.destroy();
  await releaseWebGPUExclusiveLock();
}
