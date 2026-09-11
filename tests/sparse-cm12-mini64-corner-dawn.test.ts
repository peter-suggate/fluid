import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { getSceneDefinition } from "../lib/core/scenes";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { sparseCM12DawnDefaultValues } from "../lib/harness/sparse-cm12-dawn-defaults";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { adaptiveMassMethod } from "../lib/methods/adaptive-volume/method";
import type { WebGPUAdaptiveMassSolver } from "../lib/methods/adaptive-volume/webgpu-adaptive-mass-solver";

import { sceneDocument } from "../lib/core/scene-definition";
import { decodeSparseCM12SignedPresentationKey } from "../lib/methods/adaptive-volume/webgpu-sparse-cm12-resident";

const liveGPUs = new Set<GPU>();
Object.assign(globalThis, { mini64CornerGPUs: liveGPUs });
const dawnModule = process.env.WEBGPU_NODE_MODULE;
(dawnModule ? test : test.skip)("mini64 material front reaches the far corner with production defaults", { timeout: 240_000 }, async () => {
  await acquireWebGPUExclusiveLock("dawn-test", "mini64-corner");
  let device: GPUDevice | undefined, solver: WebGPUAdaptiveMassSolver | undefined;
  try {
    const dawn = await import(pathToFileURL(dawnModule!).href);
    Object.assign(globalThis, dawn.globals);
    const gpu: GPU = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
    liveGPUs.add(gpu);
    const adapter = await gpu.requestAdapter(); assert.ok(adapter);
    device = await adapter.requestDevice({ requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
    const errors: string[] = [];
    device.addEventListener("uncapturederror", event => { event.preventDefault(); errors.push(event.error.message); });
    const scene = sceneDocument(getSceneDefinition("minimal-power-dam-break-64")); scene.duration_s = 1;
    solver = await adaptiveMassMethod.createSolverAsync!(device, scene, "balanced", sparseCM12DawnDefaultValues(), undefined, () => {}) as WebGPUAdaptiveMassSolver;
    await solver.waitForSimulationReady();
    for (let step = 0; step <= 30; step++) {
      if (step) {
        while (!solver.advanceTo(step / 30, [])) await new Promise(setImmediate);
        await solver.awaitFrameCompletion?.();
        if (step % 2 === 0) { await device.queue.onSubmittedWorkDone(); await solver.assertSimulationHealthy(); }
      }
      if (step % 2) continue;
      const fields = await solver.readDiagnosticFields(true);
      const world: Awaited<ReturnType<WebGPUAdaptiveMassSolver["readWorldGrowthReceiptQA"]>> = await solver.readWorldGrowthReceiptQA();
      const activity: Awaited<ReturnType<WebGPUAdaptiveMassSolver["readGPUActivityPolicy"]>> = await solver.readGPUActivityPolicy();
      const source = solver.sparseWorld.presentation().fineLevelSet;
      const read = async (buffer: GPUBuffer) => {
        const staging = device!.createBuffer({ size: buffer.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
        const encoder = device!.createCommandEncoder(); encoder.copyBufferToBuffer(buffer, 0, staging, 0, buffer.size); device!.queue.submit([encoder.finish()]);
        await staging.mapAsync(GPUMapMode.READ); const words = new Uint32Array(staging.getMappedRange()).slice(); staging.unmap(); staging.destroy(); return words;
      };
      const directory = await read(source.worklist), metadata = await read(source.metadata), samples = await read(source.samples);
      let publishedCornerSamples = 0;
      for (const page of directory.slice(7, 7 + directory[1]!)) {
        const q = decodeSparseCM12SignedPresentationKey(metadata[4 * page + 1]!);
        if (q[0] !== 7 || q[2] !== 7) continue;
        for (const sample of samples.subarray(page * source.plan.samplesPerBrick, (page + 1) * source.plan.samplesPerBrick)) if (((sample >>> 16) & 16) !== 0) publishedCornerSamples++;
      }
      let mass = 0, cornerMass = 0, frontX = -1, frontZ = -1, diagonalFront = -1, liquidDiagonalFront = -1;
      for (let z = 0; z < 64; z++) for (let y = 0; y < 64; y++) for (let x = 0; x < 64; x++) {
        const rho = fields.density[x + 64 * (y + 64 * z)]!;
        assert.ok(Number.isFinite(rho) && rho >= 0);
        mass += rho;
        if (x >= 60 && z >= 60) cornerMass += rho;
        if (x === z && y < 8 && rho > 0.05) diagonalFront = Math.max(diagonalFront, x);
        if (x === z && y < 8 && rho > 0.5) liquidDiagonalFront = Math.max(liquidDiagonalFront, x);
        if (rho > 0.05) { frontX = Math.max(frontX, x); frontZ = Math.max(frontZ, z); }
      }
      console.log(JSON.stringify({ step, time_s: step / 30, mass, cornerMass, diagonalFront, liquidDiagonalFront, publishedCornerSamples, frontX, frontZ, capacityFaults: world.capacityFaults, leaves: [world.liveLeaves, world.capacity], pages: world.activeTopologyPages, faults: activity.faultFlags, commitFailed: activity.commitFailed }));
      assert.equal(world.capacityFaults, 0, "frontier allocation must not exhaust the page pool");
      assert.equal(world.insertionFaults, 0);
      assert.equal(activity.faultFlags, 0);
      assert.equal(activity.commitFailed, false);
      if (step === 12) {
        assert.ok(cornerMass >= 1, `front must materially reach far corner; mass ${cornerMass}`);
        assert.ok(publishedCornerSamples >= 64, "renderer must publish material in the far-corner page");
      }
    }
    assert.deepEqual(errors, []);
  } finally { solver?.destroy(); device?.destroy(); liveGPUs.clear(); await releaseWebGPUExclusiveLock(); }
});
