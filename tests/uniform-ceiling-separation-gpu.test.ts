import assert from "node:assert/strict";
import { pathToFileURL } from "node:url";
import test from "node:test";

import { uniformMethod } from "../lib/methods/uniform";
import { scaleScene } from "../lib/scene-scale";
import { getScenePreset } from "../lib/scenes";
import { requiredFluidDeviceLimits } from "../lib/webgpu-device-limits";

const webgpuModulePath = process.env.WEBGPU_NODE_MODULE;

test("CM11a ceiling separation releases the 4x paper-step mini-dam film", {
  skip: !webgpuModulePath && "set WEBGPU_NODE_MODULE for GPU validation",
  timeout: 120_000,
}, async () => {
  const dawn = await import(pathToFileURL(webgpuModulePath!).href) as {
    create(options: string[]): GPU;
    globals: Record<string, unknown>;
  };
  Object.assign(globalThis, dawn.globals);
  const gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
  const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
  assert.ok(adapter);
  const requiredFeatures: GPUFeatureName[] = ["subgroups"];
  if (adapter.features.has("timestamp-query")) requiredFeatures.push("timestamp-query");
  const device = await adapter.requestDevice({
    requiredFeatures,
    requiredLimits: requiredFluidDeviceLimits(adapter.limits),
  });

  const base = getScenePreset("minimal-power-dam-break-64").create();
  const twice = scaleScene(base, "world", 2);
  assert.ok(twice);
  const scene = scaleScene(twice, "world", 2);
  assert.ok(scene);
  const dt = 1 / 30;
  scene.numerics.fixedDt_s = dt;
  scene.numerics.maxDt_s = dt;

  const solver = await uniformMethod.createSolverAsync!(device, scene, "balanced", {
    densityPostProcessing: "scene",
    timeStep: "scene",
  }, undefined, () => {});
  try {
    assert.deepEqual([solver.info.nx, solver.info.ny, solver.info.nz], [64, 64, 64]);
    const surfaceField = solver.surfaceFieldTexture;
    assert.ok(surfaceField);
    assert.notEqual(surfaceField, solver.volumeTexture,
      "the mini dam should render the Sec. 3.8 reconstruction");
    for (let step = 1; step <= 45; step += 1) {
      assert.equal(solver.advanceTo(step * dt, []), true);
    }
    await device.queue.onSubmittedWorkDone();

    const [nx, ny, nz] = [solver.info.nx, solver.info.ny, solver.info.nz];
    const bytesPerRow = Math.ceil(nx * 4 / 256) * 256;
    const staging = device.createBuffer({
      label: "Uniform ceiling-separation readback",
      size: bytesPerRow * ny * nz,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const renderStaging = device.createBuffer({
      label: "Uniform thin-sheet render readback",
      size: bytesPerRow * ny * nz,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = device.createCommandEncoder();
    encoder.copyTextureToBuffer({ texture: solver.volumeTexture }, {
      buffer: staging,
      bytesPerRow,
      rowsPerImage: ny,
    }, { width: nx, height: ny, depthOrArrayLayers: nz });
    encoder.copyTextureToBuffer({ texture: surfaceField }, {
      buffer: renderStaging,
      bytesPerRow,
      rowsPerImage: ny,
    }, { width: nx, height: ny, depthOrArrayLayers: nz });
    device.queue.submit([encoder.finish()]);
    await Promise.all([
      staging.mapAsync(GPUMapMode.READ),
      renderStaging.mapAsync(GPUMapMode.READ),
    ]);
    const raw = new Uint8Array(staging.getMappedRange());
    const rendered = new Uint8Array(renderStaging.getMappedRange());
    let mass = 0;
    let lidMass = 0;
    let lidWetCells = 0;
    let highestWetLayer = -1;
    let reconstructedCeilingWetCells = 0;
    for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1) {
      const row = new Float32Array(raw.buffer, raw.byteOffset + bytesPerRow * (y + ny * z), nx);
      const renderRow = new Float32Array(
        rendered.buffer, rendered.byteOffset + bytesPerRow * (y + ny * z), nx);
      for (let x = 0; x < nx; x += 1) {
        const density = row[x]!;
        mass += density;
        if (density >= 0.5) highestWetLayer = Math.max(highestWetLayer, y);
        if (y === ny - 1) {
          lidMass += density;
          if (density >= 0.5) lidWetCells += 1;
        }
        if (y >= ny - 4 && renderRow[x]! >= 0.5) reconstructedCeilingWetCells += 1;
      }
    }
    staging.unmap();
    renderStaging.unmap();
    staging.destroy();
    renderStaging.destroy();

    assert.ok(Math.abs(mass - 94_400) < 1,
      `conservative density drifted to ${mass}`);
    assert.equal(lidWetCells, 0,
      `the released film left ${lidWetCells} visible cells stuck to the lid`);
    assert.ok(lidMass < 350,
      `the released film left ${lidMass} density-cell units in the lid layer`);
    assert.ok(highestWetLayer <= 60,
      `the visible surface remained at layer ${highestWetLayer}`);
    assert.ok(reconstructedCeilingWetCells > 2_000,
      `Sec. 3.8 exposed only ${reconstructedCeilingWetCells} top-band thin-sheet cells`);
  } finally {
    solver.destroy();
    device.destroy();
  }
});
