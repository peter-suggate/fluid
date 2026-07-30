import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";

import { fluidExecutionDeviceFeatures } from "../lib/gpu-startup";
import { octreeMethod, octreeSolverOptions } from "../lib/methods/octree";
import { svoFluidCoverageFromSignedDistance } from "../lib/svo-fluid-coverage";
import { WATER_OPTICS } from "../lib/webgpu-lighting";
import { requiredFluidDeviceLimits } from "../lib/webgpu-device-limits";
import { WebGPUUniformEulerianSolver } from "../lib/webgpu-uniform-eulerian";
import { WebGpuSvoFluidCoverage } from "../lib/webgpu-svo-fluid-coverage";
import { readFloatTexture3D } from "../tools/webgpu-smoke-readbacks";
import { createSmokeScenario } from "../tools/webgpu-smoke-scenarios";

const retainedNativeGPUs: GPU[] = [];
const retainedDevices: GPUDevice[] = [];

/**
 * End-to-end shadow strength against a real solver.
 *
 * Unit tests proved the coverage volume matches its CPU reference; they cannot
 * say whether the resulting shadow is visible, because that depends on how much
 * water a light ray actually crosses in a real scene. This measures exactly
 * that: it steps the production octree solver, resamples its coarse level set
 * the way the renderer does, and reports the transmittance a vertical shadow
 * cone would return through the deepest water column.
 */
test("water thick enough to see produces a shadow a viewer can see", {
  skip: !process.env.WEBGPU_NODE_MODULE && "set WEBGPU_NODE_MODULE for the Dawn shadow-strength gate",
  timeout: 300_000,
}, async (t) => {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href) as {
    create(options: string[]): GPU;
    globals: Record<string, unknown>;
  };
  Object.assign(globalThis, dawn.globals);
  const workerDescriptor = Object.getOwnPropertyDescriptor(globalThis, "Worker");
  Reflect.deleteProperty(globalThis, "Worker");
  t.after(() => { if (workerDescriptor) Object.defineProperty(globalThis, "Worker", workerDescriptor); });

  const nativeGpu = dawn.create([`backend=${process.env.WEBGPU_BACKEND ?? "metal"}`]);
  retainedNativeGPUs.push(nativeGpu);
  const adapter = await nativeGpu.requestAdapter();
  assert.ok(adapter, "no Dawn adapter");
  const device = await adapter.requestDevice({
    requiredLimits: requiredFluidDeviceLimits(adapter.limits),
    requiredFeatures: fluidExecutionDeviceFeatures(adapter.features),
  });
  retainedDevices.push(device);
  const uncaptured: string[] = [];
  device.addEventListener("uncapturederror", (event) => { uncaptured.push(event.error.message); });

  const scenario = createSmokeScenario("dam-break-ui");
  const values = octreeMethod.presetFor("balanced");
  const solver = await WebGPUUniformEulerianSolver.createAsync(
    device, scenario.scene, "balanced", undefined,
    octreeSolverOptions(scenario.scene, "balanced", values),
    () => { /* construction progress is not what this gate measures */ },
  );
  t.after(() => solver.destroy());

  // Enough simulated time for the dam column to collapse into a pool with real
  // depth; a thin sheet would understate the shadow this measures.
  for (let step = 1; step <= 60; step += 1) {
    solver.advanceTo(step * 0.004);
    await device.queue.onSubmittedWorkDone();
  }

  const { nx, ny, nz } = solver.info;
  const field = solver.surfaceFieldTexture ?? solver.volumeTexture;
  const phi = await readFloatTexture3D(device, field, nx, solver.info.storedNy, nz);
  const finite = [...phi].filter((value) => Number.isFinite(value));
  const inside = finite.filter((value) => value < 0);
  console.log(JSON.stringify({
    record: "coarse-level-set",
    grid: [nx, ny, nz], samples: finite.length, insideSamples: inside.length,
    minimum: Math.min(...finite), maximum: Math.max(...finite),
  }));
  assert.ok(inside.length > 0,
    "the coarse level set must contain water; a field with no negative values is occupancy, not a signed distance");

  const container = scenario.scene.container;
  const cellSize = [container.width_m / nx, container.height_m / ny, container.depth_m / nz] as const;
  const coverage = new WebGpuSvoFluidCoverage(device, {
    fieldDimensions: [nx, ny, nz],
    worldOrigin_m: [-container.width_m / 2, 0, -container.depth_m / 2],
    cellSize_m: cellSize,
    cellsPerTexel: 1,
  }, { coarsePhi: field.createView({ dimension: "3d" }) });
  t.after(() => coverage.destroy());

  const encoder = device.createCommandEncoder();
  coverage.encode(encoder);
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  assert.deepEqual(uncaptured, [], "the fill must encode against a live solver without validation errors");

  // Deepest vertical water column, measured from the same coverage the shadow
  // cone reads. A cone toward an overhead light accumulates this path length.
  const cellDiagonal = Math.hypot(...cellSize);
  let deepestColumn_m = 0;
  let deepestAt: readonly [number, number] = [0, 0];
  for (let z = 0; z < nz; z += 1) {
    for (let x = 0; x < nx; x += 1) {
      let column_m = 0;
      for (let y = 0; y < ny; y += 1) {
        const value = phi[(z * solver.info.storedNy + y) * nx + x];
        column_m += svoFluidCoverageFromSignedDistance(value, cellDiagonal) * cellSize[1];
      }
      if (column_m > deepestColumn_m) { deepestColumn_m = column_m; deepestAt = [x, z]; }
    }
  }
  const transmittance = WATER_OPTICS.absorption.map((coefficient) => Math.exp(-coefficient * deepestColumn_m));
  const darkestChannelLoss = 1 - Math.min(...transmittance);
  console.log(JSON.stringify({
    record: "shadow-strength",
    deepestColumn_m: Number(deepestColumn_m.toFixed(4)), deepestAt,
    transmittance: transmittance.map((value) => Number(value.toFixed(4))),
    darkestChannelLoss: Number(darkestChannelLoss.toFixed(4)),
  }));

  assert.ok(deepestColumn_m > 0, "the coverage must find water somewhere");
  // Roughly the threshold below which a shading difference is invisible against
  // a lit surface. Absorption alone over a shallow pool falls under it, which is
  // why the shadow term cannot be absorption alone.
  assert.ok(darkestChannelLoss >= 0.08,
    `the deepest water column (${deepestColumn_m.toFixed(3)} m) attenuates light by only ${(darkestChannelLoss * 100).toFixed(1)}%, which is below the visible threshold`);
});
