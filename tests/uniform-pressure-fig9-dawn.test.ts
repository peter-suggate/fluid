import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { writeFileSync } from "node:fs";
import { createProcessRetainedDawnGPU, type NodeDawnProvider } from "../lib/harness/node-dawn-provider";
import { managedGPUDevice } from "../lib/core/gpu-compilation-manager";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { uniformVolumeMethod } from "../lib/methods/uniform/uniform-volume-method";
import { resolveMethodValues } from "../lib/core/method-contract";
import type { WebGPUUniformReferenceSolver } from "../lib/methods/uniform/webgpu-uniform-reference";
const modulePath = process.env.WEBGPU_NODE_MODULE;
(modulePath ? test : test.skip)("3D Figure 9 retains liquid and bounded pressure through six seconds", { timeout: 240_000 }, async () => {
  await acquireWebGPUExclusiveLock("dawn-test", "uniform geometric Figure 9 pressure stability");
  let device: GPUDevice | undefined, solver: WebGPUUniformReferenceSolver | undefined;
  try {
    const dawn = await import(pathToFileURL(modulePath!).href) as NodeDawnProvider;
    Object.assign(globalThis, dawn.globals);
    const gpu = createProcessRetainedDawnGPU(dawn, [`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
    const adapter = await gpu.requestAdapter(); assert.ok(adapter);
    device = managedGPUDevice(await adapter.requestDevice({ requiredLimits: requiredFluidDeviceLimits(adapter.limits) }), { requireWorkerRealm: false });
    const errors: string[] = [];
    device.addEventListener("uncapturederror", e => { e.preventDefault(); errors.push(e.error.message); });
    solver = await uniformVolumeMethod.createSolverAsync!(device, sceneDocument(getSceneDefinition("mass-conserving-figure-9-dam-break")), "balanced", resolveMethodValues(uniformVolumeMethod, "balanced", {}), undefined, () => {}) as WebGPUUniformReferenceSolver;
    assert.deepEqual([solver.info.nx, solver.info.ny, solver.info.nz], [128, 128, 64]);
    const receipts: Record<string, unknown>[] = [];
    for (let frame = 1; frame <= 180; frame++) {
      while (!solver.advanceTo(frame / 30, [])) await new Promise(setImmediate);
      const s = await solver.readStats();
      const pressure = s as unknown as Record<string, number | boolean>;
      assert.ok(Number.isFinite(s.maxSpeed_m_s) && s.maxSpeed_m_s! < 100, `frame ${frame}: speed ${s.maxSpeed_m_s}`);
      assert.ok(Number.isFinite(pressure.uniformCM11aFineResidualInfinity));
      assert.ok(Number(pressure.uniformPressureAcceptedResidual) <= Number(pressure.uniformPressureInitialResidual), `frame ${frame}: only non-worsening pressure is published`);
      assert.ok(Number(pressure.uniformPressureRecoverySweeps) <= 64);
      assert.ok(Math.abs(s.rawVolumeDrift!) < 1e-4, `frame ${frame}: conserved mass`);
      assert.ok(s.representedVolumeCellSum! > 0.5 * s.initialVolumeCellSum!, `frame ${frame}: surface must survive`);
      receipts.push({ frame, speed: s.maxSpeed_m_s, volumeDrift: s.rawVolumeDrift, representedVolume: s.representedVolumeCellSum,
        initialResidual: pressure.uniformPressureInitialResidual, residual: pressure.uniformCM11aFineResidualInfinity,
        rejected: pressure.uniformPressureRejectedCycles, recoverySweeps: pressure.uniformPressureRecoverySweeps, recoveryExhausted: pressure.uniformPressureRecoveryExhausted });
    }
    assert.deepEqual(errors, []);
    if (process.env.FLUID_UNIFORM_STABILITY_REPORT) writeFileSync(process.env.FLUID_UNIFORM_STABILITY_REPORT, JSON.stringify({ dimensions: [128, 128, 64], frames: 180, receipts }, null, 2) + "\n");
    console.log(JSON.stringify({ scene: "figure9", frames: 180, peakSpeed: Math.max(...receipts.map(r => Number(r.speed))), last: receipts.at(-1) }));
  } finally { solver?.destroy(); device?.destroy(); await releaseWebGPUExclusiveLock(); }
});
