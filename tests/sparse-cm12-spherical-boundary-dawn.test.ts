import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";

import { CM12_PAPER_DT_S } from "../lib/core/cm12-numerics";
import { createCm12Figure8, createCm12Figure12 } from
  "../lib/core/cm12-paper-scenes";
import { sceneLatticeDimensions } from "../lib/core/scene-lattice";
import { sphericalContainerFineGeometry,
  sphericalContainerOpenFractionAtFineBox } from "../lib/core/spherical-container";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from
  "../lib/harness/webgpu-smoke-isolation";
import { WebGPUAdaptiveMassSolver } from
  "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";

const dawnModule = process.env.WEBGPU_NODE_MODULE;
const dawnTest = dawnModule ? test : test.skip;

dawnTest("Dawn advances a CM12 spherical scene with the sparse boundary",
  { timeout: 240_000 }, async () => {
    await acquireWebGPUExclusiveLock("dawn-test",
      "tests/sparse-cm12-spherical-boundary-dawn.test.ts");
    let device: GPUDevice | undefined;
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
      const uncaptured: string[] = [];
      device.addEventListener("uncapturederror", (event) => {
        event.preventDefault();
        uncaptured.push(event.error.message);
      });

      const requestedScene = process.env.FLUID_SCENE;
      // Keep one scene per Dawn process. Repeated destruction/reconstruction
      // of this large resident arena is a node-webgpu teardown hazard.
      const sceneFactories = requestedScene === "cm12-figure-12"
        ? [createCm12Figure12] : [createCm12Figure8];
      for (const makeScene of sceneFactories) {
        const scene = makeScene();
        const dimensions = sceneLatticeDimensions(scene);
        const boundary = sphericalContainerFineGeometry(scene, dimensions);
        assert.ok(boundary);
        device.pushErrorScope("validation");
        const solver = await WebGPUAdaptiveMassSolver.createAsync(
          device, scene, "balanced", undefined,
          {
            resolutionMode: "adaptive",
            fineTileResolution: 8,
            coarseTileResolution: 4,
            timeStep: "paper",
          },
          () => {},
        );
        try {
          assert.deepEqual([solver.info.nx, solver.info.ny, solver.info.nz], dimensions);
          assert.equal(solver.advanceTo(CM12_PAPER_DT_S, []), true);
          await device.queue.onSubmittedWorkDone();
          const fields = await solver.readDiagnosticFields();
          let maximumClosedDensity = 0, minimumClosedPressure = Infinity;
          for (let z = 0; z < dimensions[2]; z += 1)
            for (let y = 0; y < dimensions[1]; y += 1)
              for (let x = 0; x < dimensions[0]; x += 1) {
                const index = x + dimensions[0] * (y + dimensions[1] * z);
                const density = fields.density[index]!;
                assert.ok(Number.isFinite(density));
                assert.ok(Number.isFinite(fields.pressure[index]!));
                assert.ok(Number.isFinite(fields.divergence[index]!));
                if (sphericalContainerOpenFractionAtFineBox(
                  boundary, [x + 0.5, y + 0.5, z + 0.5], [1, 1, 1],
                ) === 0) {
                  maximumClosedDensity = Math.max(maximumClosedDensity, density);
                  minimumClosedPressure = Math.min(minimumClosedPressure,
                    fields.pressure[index]!);
                }
              }
          assert.ok(maximumClosedDensity <= 1e-6,
            `fully solid finest cells retained density ${maximumClosedDensity}`);
          assert.ok(minimumClosedPressure >= -1e-6,
            `separating pressure violated p >= 0 with ${minimumClosedPressure}`);
        } finally {
          solver.destroy();
        }
        const validation = await device.popErrorScope();
        assert.equal(validation?.message, undefined);
        assert.deepEqual(uncaptured, []);
      }
    } finally {
      device?.destroy();
      releaseWebGPUExclusiveLock();
    }
  });
