/** Equal-time slab comparison of conservative volume and the published phi contour. */
import assert from "node:assert/strict";
import { writeFileSync, mkdirSync } from "node:fs";
import { dirname } from "node:path";
import { pathToFileURL } from "node:url";
import { getScenePreset } from "../lib/core/scenes";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { sparseCM12DawnDefaultOptions } from "../lib/harness/sparse-cm12-dawn-defaults";
import { createProcessRetainedDawnGPU, type NodeDawnProvider } from "../lib/harness/node-dawn-provider";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { WebGPUAdaptiveMassSolver } from "../lib/methods/adaptive-volume/webgpu-adaptive-mass-solver";
import { readPublishedCM12Field } from "./sparse-cm12-published-field";
const arg = (key: string, fallback: string) => process.argv.find(v => v.startsWith(`--${key}=`))?.slice(key.length + 3) ?? fallback;
const sceneId = arg("scene", "water-box-dam-break-slab");
const steps = Number(arg("steps", "10"));
const distanceSweeps = Number(arg("distance-sweeps", "8"));
const returnPasses = Number(arg("return-passes", "4"));
const output = arg("output", "artifacts/level-set-volume/sharpening-slab.json");
assert.ok(Number.isInteger(steps) && steps >= 0);
await acquireWebGPUExclusiveLock("dawn-probe", "sparse geometric sharpening");
let device: GPUDevice | undefined, solver: WebGPUAdaptiveMassSolver | undefined;
const report: Record<string, unknown> = { sceneId, steps, distanceSweeps, returnPasses, dt: 1 / 30, checkpoints: [] };
const errors: string[] = [];
try {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href) as NodeDawnProvider;
  Object.assign(globalThis, dawn.globals);
  const gpu = createProcessRetainedDawnGPU(dawn, ["backend=metal", "enable-dawn-features=disable_blob_cache"]);
  const adapter = await gpu.requestAdapter(); assert.ok(adapter);
  device = await adapter.requestDevice({ requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
  device.addEventListener("uncapturederror", e => { e.preventDefault(); errors.push(e.error.message); });
  const scene = getScenePreset(sceneId).create();
  scene.numerics.fixedDt_s = scene.numerics.maxDt_s = 1 / 30;
  solver = await WebGPUAdaptiveMassSolver.createCompiledTopologyTransport(device, scene, "balanced",
    undefined, { ...sparseCM12DawnDefaultOptions(), distanceSweeps, returnPasses }, () => {});
  await solver.waitForSimulationReady();
  report.dimensions = [solver.info.nx, solver.info.ny, solver.info.nz];
  const frameTimes: number[] = [];
  for (let step = 0; step <= steps; step++) {
    if (step > 0) {
      const started = performance.now();
      while (!solver.advanceTo(step / 30, [])) await new Promise<void>(setImmediate);
      await solver.awaitFrameCompletion(); await device.queue.onSubmittedWorkDone();
      frameTimes.push(performance.now() - started);
    }
    if (![0, 1, 3, 5, 10, 20, steps].includes(step)) continue;
    const [fields, phi, volume, transport, activity] = await Promise.all([
      solver.readDiagnosticFields(true), readPublishedCM12Field(device, solver),
      solver.readAcceptedGeometricVolumeQA(), solver.readGeometricVolumeTransportReceiptQA(),
      solver.readGPUActivityPolicy(),
    ]);
    let phiNegativeFineCells = 0;
    let airSideVolume = 0, deepAirVolume = 0, fractionalFineCells = 0, densitySum = 0;
    for (let i = 0; i < fields.density.length; i++) {
      const density = fields.density[i]!; densitySum += density;
      if (phi.values[i]! < 0) phiNegativeFineCells++;
      if (density > 1e-5 && density < 1 - 1e-5) fractionalFineCells++;
      if (phi.values[i]! > 0) airSideVolume += density;
      if (phi.values[i]! > scene.voxelDomain.finestCellSize_m) deepAirVolume += density;
    }
    const active = activity.bricks.filter(brick => brick.active);
    const adaptivity = {
      activeBricks: active.length,
      resolutions: Object.fromEntries([1, 2, 4, 8, 16].map(r => [r, active.filter(b => b.acceptedResolution === r).length])),
      thinBricks: active.filter(b => (b.reasons & 256) !== 0).length,
      surfaceBricks: active.filter(b => (b.reasons & 1) !== 0).length,
      bricks: active,
    };
    (report.checkpoints as unknown[]).push({ step, volume, adaptivity, coupling: transport.coupling,
      transportFault: transport.fault, phiNegativeFineCells, airSideVolume, deepAirVolume, fractionalFineCells, densitySum });
  }
  report.frameTimesMs = frameTimes;
  report.completed = true;
} catch (e) {
  report.completed = false; report.error = String(e); process.exitCode = 1;
} finally {
  report.validationErrors = errors;
  solver?.destroy(); device?.destroy();
  await releaseWebGPUExclusiveLock();
  mkdirSync(dirname(output), { recursive: true }); writeFileSync(output, JSON.stringify(report, null, 2) + "\n");
  console.log(JSON.stringify({ output, completed: report.completed, error: report.error }));
}
