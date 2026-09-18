import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { getScenePreset } from "../lib/core/scenes";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import { sparseCM12DawnDefaultOptions } from "../lib/harness/sparse-cm12-dawn-defaults";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { WebGPUAdaptiveMassSolver } from "../lib/methods/adaptive-volume/webgpu-adaptive-mass-solver";

// A coarse disk has an initial SDF/material mismatch. Global phi feedback used
// to distribute that error onto the disconnected resting pool, whose coarse
// air-gap redistance then amplified the manufactured rise on every frame.
(process.env.WEBGPU_NODE_MODULE ? test : test.skip)(
  "min8 disk volume mismatch does not inflate the disconnected resting pool",
  { timeout: 120_000 }, async () => {
    await acquireWebGPUExclusiveLock("dawn-test", "min8 pool/disk phi separation");
    let device: GPUDevice | undefined;
    let solver: WebGPUAdaptiveMassSolver | undefined;
    try {
      const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href);
      Object.assign(globalThis, dawn.globals);
      const gpu = createProcessRetainedDawnGPU(dawn, [`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
      const adapter = await gpu.requestAdapter(); assert.ok(adapter);
      device = await adapter.requestDevice({ requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
      assert.ok(device);
      const scene = getScenePreset("coarse-first-pool-impact-half-slab").create();
      scene.numerics.fixedDt_s = scene.numerics.maxDt_s = .009;
      scene.fluid.refinementRegions = [{
        id: "min8", rule: "minimum-cell-size", minimumCellSize_cells: 8,
        min_m: { x: -3.2, y: 0, z: -.4 }, max_m: { x: 3.2, y: 4.8, z: .4 },
      }];
      solver = await WebGPUAdaptiveMassSolver.createCompiledTopologyTransport(
        device, scene, "balanced", undefined,
        { ...sparseCM12DawnDefaultOptions(), timeStep: "scene" }, () => {});
      await solver.waitForSimulationReady();
      const initial = await solver.readDiagnosticFields(true);
      const initialVolume = initial.density.reduce((sum, value) => sum + value, 0);
      for (let frame = 1; frame <= 9; frame++) {
        while (!solver.advanceTo(frame * .009, [])) await new Promise<void>(resolve => setImmediate(resolve));
        await solver.awaitFrameCompletion(); await solver.waitForTopologyReady();
        const sdf: Awaited<ReturnType<WebGPUAdaptiveMassSolver["readAdaptiveLevelSetQA"]>> =
          await solver.readAdaptiveLevelSetQA(true);
        assert.equal(sdf.fault, 0);
        assert.ok(sdf.activeCells <= 48, "the test must retain the min8 coarse representation");
        const vertices = sdf.vertices; assert.ok(vertices);
        const phi = (y: number, z: number) => {
          const vertex = vertices.find(v => v.positionFine[0] === 32
            && v.positionFine[1] === y && v.positionFine[2] === z);
          assert.ok(vertex, `missing centre vertex at y=${y}, z=${z}`);
          return vertex.phiFine;
        };
        for (const z of [0, 8]) {
          const y = phi(16, z) >= 0 ? 8 : 16;
          const a = phi(y, z), b = phi(y + 8, z);
          assert.ok(a <= 0 && b >= 0, `frame ${frame}: pool crossing disappeared`);
          const height = .1 * (y - 8 * a / (b - a));
          // 2 cm is a fifth of one finest cell; the old feedback moves the
          // surface by 2.8 cm in frame 1 and roughly 60 cm by frame 8.
          assert.ok(Math.abs(height - 1.6) < .02,
            `frame ${frame}, z=${z}: disconnected pool moved to ${height} m`);
          assert.ok(phi(24, z) > 0 && phi(32, z) < 0, "disk and pool stay separated");
        }
        const fields = await solver.readDiagnosticFields(true);
        const volume = fields.density.reduce((sum, value) => sum + value, 0);
        assert.ok(Math.abs(volume - initialVolume) / initialVolume < 1e-6,
          "surface stability must retain conservative material");
      }
    } finally { solver?.destroy(); device?.destroy(); await releaseWebGPUExclusiveLock(); }
  });
