import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { createProcessRetainedDawnGPU, type NodeDawnProvider } from "../lib/harness/node-dawn-provider";
import { managedGPUDevice } from "../lib/core/gpu-compilation-manager";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { solidVoxelShellForScene } from "../lib/core/scene-lattice";
import { WebGPUUniformReferenceSolver } from "../lib/methods/uniform/webgpu-uniform-reference";

const modulePath = process.env.WEBGPU_NODE_MODULE;
(modulePath ? test : test.skip)("pressure safety rejects corrupt cycles before velocity projection", { timeout: 180_000 }, async t => {
  await acquireWebGPUExclusiveLock("dawn-test", "uniform pressure safety fault injection");
  let device: GPUDevice | undefined;
  try {
    const dawn = await import(pathToFileURL(modulePath!).href) as NodeDawnProvider;
    Object.assign(globalThis, dawn.globals);
    const gpu = createProcessRetainedDawnGPU(dawn, [`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
    const adapter = await gpu.requestAdapter(); assert.ok(adapter);
    const raw = await adapter.requestDevice({ requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
    let fault = "";
    let hits = 0;
    const original = raw.createShaderModule.bind(raw);
    raw.createShaderModule = descriptor => {
      const dense = "textureStore(mgPressureOut,id,vec4f(mgP(id)+textureLoad(mgResidualIn,id,0).x));";
      const paged = "mgPressureOutStore(id,vec4f(mgP(id)+mgResidualInLoad(id).x));";
      const from=descriptor.code.includes(dense)?dense:paged;
      if (fault && descriptor.code.includes(from)) {
        hits++;
        const value=from===dense?fault:fault.replace("textureLoad(mgResidualIn,id,0)","mgResidualInLoad(id)");
        const replacement=from===dense?`textureStore(mgPressureOut,id,vec4f(${value}));`:`mgPressureOutStore(id,vec4f(${value}));`;
        return original({ ...descriptor, code: descriptor.code.replace(from, replacement) });
      }
      return original(descriptor);
    };
    device = managedGPUDevice(raw, { requireWorkerRealm: false });
    const errors: string[] = [];
    device.addEventListener("uncapturederror", e => { e.preventDefault(); errors.push(e.error.message); });
    const scene = structuredClone(sceneDocument(getSceneDefinition("minimal-power-dam-break-32")));
    scene.container.width_m = scene.container.height_m = scene.container.depth_m = 0.4;
    scene.voxelDomain.finestCellSize_m = 0.05;
    scene.fluid.initialCondition = "tank-fill";
    scene.container.fillFraction = 0.5;
    scene.fluid.initialLiquidVolumes = [];
    scene.solidVoxels = [...solidVoxelShellForScene(scene)];
    for (const pageDomain of [false, true]) for (const kind of ["finite", "nan", "infinity", "after-accepted-cycle"] as const) await t.test(`${pageDomain ? "GPU indirect" : "reference"}: ${kind}`, async () => {
      fault = kind === "finite" ? "1e20" : kind === "nan" ? "bitcast<f32>(0x7fc00000u+atomicLoad(&mgState.convergence[17]))" : kind === "infinity" ? "bitcast<f32>(0x7f800000u+atomicLoad(&mgState.convergence[17]))"
        : "select(mgP(id)+textureLoad(mgResidualIn,id,0).x,1e20,atomicLoad(&mgState.convergence[17])>0u)";
      const before = hits;
      const solver = await WebGPUUniformReferenceSolver.createAsync(device!, scene, "balanced", undefined, {
        geometricVolume: true, activeRegion: false, pageDomain, pressureCycleBudget: "fixed",
        pressureCycleDispatch: pageDomain ? "indirect" : "direct",
        pressureSchedule: { fullCycles: 3, vCycles: 4, preSweeps: 6, postSweeps: 6, residualTolerance: kind === "after-accepted-cycle" ? 0 : 0.1 },
      }, () => {});
      try {
        assert.ok(hits > before, "fault must reach the actual pressure shader");
        for (let frame = 1; frame <= 2; frame++) {
          assert.ok(solver.advanceTo(frame / 30));
          const stats = await solver.readStats() as unknown as Record<string, number | boolean>;
          assert.equal(stats.uniformPressureRejectedCycles, 1, "one rejection per step, counters reset");
          assert.ok(Number(stats.uniformPressureRecoverySweeps) > 0 && Number(stats.uniformPressureRecoverySweeps) <= 64);
          assert.ok(Number(stats.uniformPressureAcceptedResidual) <= Number(stats.uniformPressureInitialResidual));
          assert.ok(Number.isFinite(stats.uniformCM11aFineResidualInfinity));
          assert.ok(Number(stats.maxSpeed_m_s) < 1, `corrupt pressure cannot reach velocity projection: speed=${stats.maxSpeed_m_s}, frame=${frame}, residual=${stats.uniformPressureAcceptedResidual}, initial=${stats.uniformPressureInitialResidual}`);
          if (kind === "after-accepted-cycle") {
            assert.equal(stats.uniformCM11aFullCyclesExecuted, 2);
            assert.equal(stats.uniformPressureRecoverySweeps, 64);
            assert.equal(stats.uniformPressureRecoveryExhausted, true);
          }
        }
      } finally { solver.destroy(); }
    });
    assert.deepEqual(errors, []);
  } finally { device?.destroy(); await releaseWebGPUExclusiveLock(); }
});
