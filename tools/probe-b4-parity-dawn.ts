/** Matched production-policy B4/B8 snapshots, including the accepted clock and volume. */
import assert from "node:assert/strict";
import { mkdir, writeFile } from "node:fs/promises";
import { pathToFileURL } from "node:url";
import { CM12_PAPER_DT_S } from "../lib/core/cm12-numerics";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { resolveMethodValues } from "../lib/core/method-contract";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { adaptiveMassMethod, adaptiveMassSolverOptions } from "../lib/methods/adaptive-volume/method";
import { WebGPUAdaptiveMassSolver } from "../lib/methods/adaptive-volume/webgpu-adaptive-mass-solver";

await acquireWebGPUExclusiveLock("dawn-test", "b4-b8-parity");
let device: GPUDevice | undefined;
let solver: WebGPUAdaptiveMassSolver | undefined;
try {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE ?? `${process.cwd()}/node_modules/webgpu/index.js`).href);
  Object.assign(globalThis, dawn.globals);
  const gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
  const adapter = await gpu.requestAdapter(); assert.ok(adapter);
  const gpuDevice: GPUDevice = await adapter.requestDevice({ requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
  device = gpuDevice;
  const errors: string[] = [];
  gpuDevice.addEventListener("uncapturederror", event => { event.preventDefault(); errors.push(event.error.message); });
  const sceneId = process.env.FLUID_PARITY_SCENE ?? "minimal-power-dam-break-32";
  const steps = Number(process.env.FLUID_PARITY_STEPS ?? 24);
  const out = process.env.FLUID_PARITY_OUT ?? "artifacts/b4-parity/numerical";
  await mkdir(out, { recursive: true });
  const bricks = process.env.FLUID_PARITY_BRICKS?.split(",").map(Number) ?? [4, 8];
  assert.ok(bricks.every(b => b === 4 || b === 8));
  for (const b of bricks) {
    const scene = sceneDocument(getSceneDefinition(sceneId));
    const values = resolveMethodValues(adaptiveMassMethod, "balanced", {
      brickFineResolution: String(b),
      ...(process.env.FLUID_PARITY_SURFACE_RINGS ? { surfaceFineRings: Number(process.env.FLUID_PARITY_SURFACE_RINGS) } : {}),
      ...(process.env.FLUID_PARITY_SELECTOR ? { selectorMode: process.env.FLUID_PARITY_SELECTOR } : {}),
      ...(process.env.FLUID_PARITY_PRESSURE ? { pressureIterations: Number(process.env.FLUID_PARITY_PRESSURE) } : {}),
    });
    solver = await WebGPUAdaptiveMassSolver.createAsync(gpuDevice, scene, "balanced", undefined,
      adaptiveMassSolverOptions(values), () => {});
    await solver.waitForSimulationReady();
    for (let step = 0; step <= steps; step++) {
      if (step > 0) {
        while (!solver.advanceTo(step * CM12_PAPER_DT_S, [])) await new Promise(setImmediate);
        await solver.awaitFrameCompletion();
      }
      const stats = await solver.readStats();
      console.log(JSON.stringify({ b, step, time: stats.simulatedTime_s, residual: stats.pressureRelativeResidual,
        iterations: stats.pressureIterationsExecuted, speed: stats.maxSpeed_m_s, cells: stats.activeSampleCount }));
      if (step % 4 === 0) {
        const fields = await solver.readDiagnosticFields(true);
        const volume = await solver.readAcceptedGeometricVolumeQA();
        const phi = await solver.readAdaptiveLevelSetQA(true);
        const activity = await solver.readGPUActivityPolicy();
        await writeFile(`${out}/${sceneId}-b${b}-${step}.json`, JSON.stringify({ b, step, values, stats, volume, phi, activity,
          density: Array.from(fields.density), velocity: Array.from(fields.velocity) }));
      }
    }
    solver.destroy(); solver = undefined;
  }
  assert.deepEqual(errors, []);
} finally { solver?.destroy(); device?.destroy(); await releaseWebGPUExclusiveLock(); }
