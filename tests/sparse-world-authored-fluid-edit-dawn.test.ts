import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";

import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { sceneAtContainerExtents } from "../lib/core/scene-scale";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from
  "../lib/harness/webgpu-smoke-isolation";
import { WebGPUAdaptiveMassSolver } from
  "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";

const dawnModule = process.env.WEBGPU_NODE_MODULE;
const dawnTest = dawnModule ? test : test.skip;

dawnTest("authored fluid moves and shape changes retain the resident world without GPU allocation",
  { timeout: 240_000 }, async () => {
    await acquireWebGPUExclusiveLock("dawn-test",
      "tests/sparse-world-authored-fluid-edit-dawn.test.ts");
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

      const scene = sceneAtContainerExtents(
        sceneDocument(getSceneDefinition("water-box-tank-fill")),
        { width_m: 1.6, height_m: 2.4, depth_m: 1.6 },
      );
      scene.rigidBodies = [];
      scene.container.fillFraction = 0.2;
      scene.voxelDomain.finestCellSize_m = 0.05;
      scene.fluid.surfaceTension_N_m = 0;
      solver = await WebGPUAdaptiveMassSolver.createAsync(
        device, scene, "balanced", undefined, {
          resolutionMode: "adaptive",
          brickFineResolution: 8,
          surfaceFineRings: 1,
          timeStep: "paper",
          pressureIterations: 32,
          pressureRelativeTolerance: 0,
        }, () => {});
      await solver.waitForSimulationReady();

      assert.equal(solver.advanceTo(1 / 30, []), true);
      await device.queue.onSubmittedWorkDone();
      const world = solver.sparseWorld;
      const time = solver.info.submittedTime_s;
      const sample = (field: Float32Array, x: number, y: number, z: number) => {
        const ix = Math.floor((x + .8) / .05), iy = Math.floor(y / .05), iz = Math.floor((z + .8) / .05);
        return field[ix + 32 * (iy + 48 * iz)]!;
      };
      let current = scene;
      const centers = [-.4, .4];
      const shapes = ["box", "sphere", "cylinder", "hemisphere", "torus"] as const;
      let allocations = 0;
      const durations: number[] = [];
      const originalCreateBuffer = device.createBuffer.bind(device);
      device.createBuffer = (...args) => { allocations++; return originalCreateBuffer(...args); };
      for (const shape of shapes) for (const x of centers) {
        const next = { ...current, fluid: { ...current.fluid, initialLiquidVolumes: [
          shape === "box" ? { shape, min_m: { x: x - .15, y: 1.15, z: -.15 }, max_m: { x: x + .15, y: 1.45, z: .15 } }
          : { shape, center_m: { x, y: 1.3, z: 0 }, radius_m: .15,
            ...(shape === "cylinder" ? { halfHeight_m: .15 } : {}),
            ...(shape === "hemisphere" ? { outwardNormal: { x: 0, y: 1, z: 0 } } : {}),
            ...(shape === "torus" ? { tubeRadius_m: .05 } : {}) },
        ] } } as typeof scene;
        const beforeAllocations = allocations;
        const start = performance.now();
        solver.applySceneUniforms(structuredClone(next));
        await solver.refreshSceneTopology();
        await device.queue.onSubmittedWorkDone();
        durations.push(performance.now() - start);
        assert.equal(allocations, beforeAllocations, "body editing must not allocate GPU buffers");
        assert.equal(solver.sparseWorld, world);
        assert.equal(solver.info.submittedTime_s, time, "editing must retain the simulation clock");
        const field: Float32Array = (await solver.readDiagnosticFields(true)).density;
        assert.ok(sample(field, x + .1, 1.275, 0) > 0, `${shape} must wet its new position`);
        assert.equal(sample(field, -x, 1.3, 0), 0, `${shape} must clear the old position`);
        assert.ok(sample(field, 0, .2, 0) > .9, "the untouched reservoir stays wet");
        current = next;
      }
      const movedReservoir = structuredClone(current);
      movedReservoir.fluid.initialCondition = "dam-break";
      movedReservoir.fluid.initialDamBreakOrigin_m = { x: 1, y: .6, z: 1 };
      movedReservoir.fluid.initialDamBreakDimensions_m = { x: .4, y: .3, z: .4 };
      const beforeMoveAllocations = allocations;
      solver.applySceneUniforms(movedReservoir);
      await solver.refreshSceneTopology();
      assert.equal(allocations, beforeMoveAllocations, "moving and resizing the base reservoir retains GPU storage");
      const movedField: Float32Array = (await solver.readDiagnosticFields(true)).density;
      assert.equal(sample(movedField, 0, .2, 0), 0, "the old reservoir footprint is cleared");
      assert.ok(sample(movedField, .4, .7, .4) > .9, "the resized reservoir exists at its new position");
      console.log("Authored fluid edit submit-to-completion ms:", durations.map(v => v.toFixed(2)).join(", "));
      device.createBuffer = originalCreateBuffer;
      assert.deepEqual(validationErrors, []);
    } finally {
      solver?.destroy();
      device?.destroy();
      await releaseWebGPUExclusiveLock();
    }
  });
