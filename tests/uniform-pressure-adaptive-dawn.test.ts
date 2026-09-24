import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { managedGPUDevice, gpuCompilationManagerFor } from "../lib/core/gpu-compilation-manager";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { solidVoxelShellForScene } from "../lib/core/scene-lattice";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { WebGPUUniformReferenceSolver } from "../lib/methods/uniform/webgpu-uniform-reference";
import { uniformGeometricSolverOptions } from "../lib/methods/uniform/uniform-geometric-options";

const modulePath = process.env.WEBGPU_NODE_MODULE;
(modulePath ? test : test.skip)("adaptive pressure meets current-frame tolerance before publication", {timeout: 240000}, async t => {
  await acquireWebGPUExclusiveLock("dawn-test", "adaptive pressure continuation");
  let device: GPUDevice | undefined;
  try {
    const dawn = await import(pathToFileURL(modulePath!).href); Object.assign(globalThis, dawn.globals);
    const gpu = createProcessRetainedDawnGPU(dawn, ["backend=metal"]);
    const adapter = await gpu.requestAdapter(); assert.ok(adapter);
    device = managedGPUDevice(await adapter.requestDevice({requiredLimits: requiredFluidDeviceLimits(adapter.limits)}), {requireWorkerRealm:false});
    const errors: string[] = [];
    device.addEventListener("uncapturederror", e => { e.preventDefault(); errors.push(e.error.message); });
    for (const sceneId of ["coarse-first-pool-impact-half-slab", "cm12-figure-9", "minimal-power-dam-break-64"]) await t.test(sceneId, async () => {
      const scene = sceneDocument(getSceneDefinition(sceneId));
      const solver = await WebGPUUniformReferenceSolver.createAsync(device!, scene, "balanced", undefined,
        uniformGeometricSolverOptions({}, scene), () => {});
      let continuations = 0;
      let cheapContinuations = 0;
      try {
        for (let frame = 1; frame <= (sceneId === "minimal-power-dam-break-64" ? 150 : 90); frame++) {
          assert.ok(solver.advanceTo(frame / 30));
          assert.equal(solver.framePending, true);
          assert.equal(solver.presentationPending, true);
          assert.equal(solver.advanceTo((frame + 1) / 30), false, "no overlapping physical frames");
          await solver.awaitFrameCompletion();
          assert.equal(solver.presentationPending, false);
          assert.ok(Math.abs(solver.info.completedTime_s! - frame/30) < 1e-9);
          const info = await solver.readStats();
          assert.equal(info.uniformPressureCyclesConverged, true, `frame ${frame}`);
          assert.ok(info.uniformCM11aFineResidualInfinity! <= 10, `frame ${frame}: ${info.uniformCM11aFineResidualInfinity}`);
          assert.equal(info.uniformPressureRecoveryExhausted, false);
          assert.ok(Number.isFinite(info.maxSpeed_m_s));
          if (info.uniformPressureCyclesEncoded! > 1) continuations++;
          if (info.uniformCM11aVCyclesExecuted! > 1 && info.uniformCM11aFullCyclesExecuted === 0) cheapContinuations++;
          if (frame === 1) {
            assert.equal(info.uniformCM11aFullCyclesExecuted, 0, "starts with a V-cycle");
            assert.equal(info.uniformPressureRecoverySweeps, 0);
            assert.ok(info.uniformPressurePassesEncoded! < 250, "unused cycles and recovery are omitted");
          }
        }
        if (sceneId === "cm12-figure-9") assert.ok(continuations > 0, "impact requests same-frame continuation");
        if (sceneId === "minimal-power-dam-break-64") assert.ok(cheapContinuations > 100,
          "mini64 should refine with V-cycles instead of routinely escalating to Full-Cycles");
      } finally { solver.destroy(); }
    });
    await t.test("stalled V progress skips unused V-cycles safely", async () => {
      const scene = sceneDocument(getSceneDefinition("coarse-first-pool-impact-half-slab"));
      const compiler = gpuCompilationManagerFor(device!);
      const original = compiler.createShaderModule;
      let hits = 0;
      compiler.createShaderModule = function(descriptor) {
        const needle = "let candidate=atomicLoad(&mgState.convergence[15]);";
        if (descriptor.code.includes(needle)) {
          hits++;
          // Report an initial 100 -> V-cycle 75 stall above tolerance;
          // Full-Cycle and final fine residuals retain their real values.
          descriptor = {...descriptor, code:descriptor.code.replaceAll(needle,
            "let candidate=select(select(atomicLoad(&mgState.convergence[15]),bitcast<u32>(100.0),mg.control.z==0u),bitcast<u32>(75.0),mg.control.z==3u);")};
        }
        return original.call(this, descriptor);
      };
      let solver: WebGPUUniformReferenceSolver | undefined;
      try {
        solver = await WebGPUUniformReferenceSolver.createAsync(device!, scene, "balanced", undefined,
          uniformGeometricSolverOptions({}, scene), () => {});
        compiler.createShaderModule = original;
        assert.ok(hits > 0);
        assert.ok(solver.advanceTo(1/30));
        await solver.awaitFrameCompletion();
        const info = await solver.readStats();
        assert.equal(info.uniformCM11aVCyclesExecuted, 1);
        assert.equal(info.uniformCM11aFullCyclesExecuted, 1);
        assert.equal(info.uniformPressureCyclesEncoded, 2);
        assert.ok(info.uniformCM11aFineResidualInfinity! <= 10);
      } finally { compiler.createShaderModule = original; solver?.destroy(); }
    });
    await t.test("a rejected correction runs recovery in the same frame", async () => {
      const scene = structuredClone(sceneDocument(getSceneDefinition("minimal-power-dam-break-32")));
      // A broad, still pool whose injected correction is recoverable within
      // the unchanged 64-sweep recovery cap at the strict 0.1 target.
      scene.container.width_m = scene.container.height_m = scene.container.depth_m = 1.6;
      scene.voxelDomain.finestCellSize_m = 0.2;
      scene.fluid.initialCondition = "tank-fill";
      scene.container.fillFraction = 0.5;
      scene.fluid.initialLiquidVolumes = [];
      scene.solidVoxels = [...solidVoxelShellForScene(scene)];
      const compiler = gpuCompilationManagerFor(device!);
      const original = compiler.createShaderModule;
      let hits = 0;
      compiler.createShaderModule = function(descriptor) {
        const needle = "textureStore(mgPressureOut,id,vec4f(mgP(id)+textureLoad(mgResidualIn,id,0).x));";
        if (descriptor.code.includes(needle)) {
          hits++;
          descriptor = {...descriptor,code:descriptor.code.replaceAll(needle,"textureStore(mgPressureOut,id,vec4f(1e20));")};
        }
        return original.call(this, descriptor);
      };
      let solver: WebGPUUniformReferenceSolver | undefined;
      try {
        solver = await WebGPUUniformReferenceSolver.createAsync(device!, scene, "balanced", undefined,
          {...uniformGeometricSolverOptions({},scene), pressureSchedule:{fullCycles:1,vCycles:0,preSweeps:6,postSweeps:6,residualTolerance:0.1}}, () => {});
        compiler.createShaderModule = original;
        assert.ok(hits > 0);
        assert.ok(solver.advanceTo(1/30));
        await solver.awaitFrameCompletion();
        const info = await solver.readStats();
        assert.equal(info.uniformPressureRejectedCycles, 1);
        assert.ok(info.uniformPressureRecoverySweeps! > 0);
        assert.equal(info.uniformPressureCyclesConverged, true);
        assert.ok(info.uniformCM11aFineResidualInfinity! <= 0.1);
        assert.ok(info.maxSpeed_m_s! < 1, "corrupt pressure cannot reach projection");
        assert.ok(info.uniformPressureFinishPassesEncoded! > 3);
      } finally { compiler.createShaderModule = original; solver?.destroy(); }
    });
    await t.test("an exhausted budget withholds publication and latches its failure", async () => {
      const scene = sceneDocument(getSceneDefinition("coarse-first-pool-impact-half-slab"));
      const options = uniformGeometricSolverOptions({}, scene);
      const solver = await WebGPUUniformReferenceSolver.createAsync(device!, scene, "balanced", undefined,
        {...options, pressureSchedule: {fullCycles:0, vCycles:1, preSweeps:1, postSweeps:1, residualTolerance:1e-12}}, () => {});
      try {
        assert.ok(solver.advanceTo(1/30));
        solver.applyRuntimeValues({pressureResidualTolerance:100});
        await assert.rejects(solver.awaitFrameCompletion(), /projection withheld/, "an in-flight tolerance edit cannot rescue the current solve");
        assert.equal(solver.presentationPending, true);
        assert.equal(solver.framePending, false);
        assert.equal(solver.info.completedTime_s ?? 0, 0);
        assert.equal(solver.info.simulatedTime_s, 0);
        assert.equal(solver.advanceTo(2/30), false);
        await assert.rejects(solver.assertSimulationHealthy(), /projection withheld/);
      } finally { solver.destroy(); }
    });
    assert.deepEqual(errors, []);
  } finally { device?.destroy(); await releaseWebGPUExclusiveLock(); }
});
