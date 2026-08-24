import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";

import { CM12_PAPER_DT_S } from "../lib/core/cm12-numerics";
import { createSymmetricExpansionScene } from "../lib/core/scenes";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from
  "../lib/harness/webgpu-smoke-isolation";
import { WebGPUAdaptiveMassSolver } from
  "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";

const dawnModule = process.env.WEBGPU_NODE_MODULE;
const dawnTest = dawnModule ? test : test.skip;

dawnTest("symmetric expansion allocates and wets sparse corner tiles",
  { timeout: 240_000 }, async () => {
    await acquireWebGPUExclusiveLock("dawn-test",
      "tests/sparse-cm12-symmetric-corner-expansion-dawn.test.ts");
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

      const horizontalCells = 32;
      const scene = createSymmetricExpansionScene();
      scene.voxelDomain.finestCellSize_m = scene.container.width_m / horizontalCells;
      const brickSize = scene.voxelDomain.brickSize_cells;
      const brickDimensions = [horizontalCells / brickSize,
        horizontalCells / (2 * brickSize), horizontalCells / brickSize] as const;
      scene.fluid.initialBrickSeeds_m = [];
      for (let z = brickDimensions[2] / 4; z < 3 * brickDimensions[2] / 4; z += 1)
        for (let y = 0; y < brickDimensions[1] / 2; y += 1)
          for (let x = brickDimensions[0] / 4; x < 3 * brickDimensions[0] / 4; x += 1) {
            scene.fluid.initialBrickSeeds_m.push({
              x: -0.5 * scene.container.width_m
                + (x + 0.5) * brickSize * scene.voxelDomain.finestCellSize_m,
              y: (y + 0.5) * brickSize * scene.voxelDomain.finestCellSize_m,
              z: -0.5 * scene.container.depth_m
                + (z + 0.5) * brickSize * scene.voxelDomain.finestCellSize_m,
            });
          }
      scene.numerics.fixedDt_s = scene.numerics.maxDt_s = CM12_PAPER_DT_S;

      solver = await WebGPUAdaptiveMassSolver.createCompiledTopologyTransport(
        device, scene, "balanced", undefined, {
          resolutionMode: "adaptive",
          brickFineResolution: 8,
          presentationPageResolution: 8,
          timeStep: "paper",
          pressureIterations: 64,
        }, () => {},
      );
      await solver.waitForSimulationReady();

      const horizontalCorner = (coordinate: readonly number[]) =>
        (coordinate[0] === 0 || coordinate[0] === brickDimensions[0] - 1)
        && (coordinate[2] === 0 || coordinate[2] === brickDimensions[2] - 1);
      const initial = await solver.readGPUActivityPolicy();
      assert.equal(initial.bricks.filter((brick) => brick.active).length, 4);
      assert.equal(initial.bricks.filter((brick) =>
        brick.active && horizontalCorner(brick.coordinate)).length, 0,
      "corner tiles must begin absent rather than hiding a preallocated apron");

      for (let step = 1; step <= 20; step += 1) {
        assert.equal(solver.advanceTo(step * CM12_PAPER_DT_S, []), true);
        await device.queue.onSubmittedWorkDone();
        if (step !== 1) continue;
        const first = await solver.readGPUActivityPolicy();
        const corners = first.bricks.filter((brick) =>
          brick.active && horizontalCorner(brick.coordinate));
        assert.equal(corners.length, 8,
          "all horizontal edge/corner tiles must publish before expansion reaches them");
        assert.ok(corners.every((brick) => brick.acceptedResolution === 8),
          "each corner must activate a complete B8 receiver tile");
      }

      const density = (await solver.readDiagnosticFields()).density;
      let cornerMass = 0;
      for (let z = 0; z < horizontalCells; z += 1)
        for (let y = 0; y < horizontalCells / 2; y += 1)
          for (let x = 0; x < horizontalCells; x += 1) {
            if (!horizontalCorner([Math.floor(x / brickSize), Math.floor(y / brickSize),
              Math.floor(z / brickSize)])) continue;
            cornerMass += density[x + horizontalCells * (y + horizontalCells / 2 * z)]!;
          }
      assert.ok(cornerMass > 1e-3,
        `allocated corner tiles must accept transported liquid; measured ${cornerMass}`);
      assert.deepEqual(validationErrors, []);
    } finally {
      solver?.destroy();
      device?.destroy();
      await releaseWebGPUExclusiveLock();
    }
  });
