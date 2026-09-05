import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { createSolidWorld } from "../lib/core/solid-world";
import { createSparseAdaptiveMassAtlas } from "../lib/methods/adaptive-mass/sparse-brick-atlas";
import { buildSparseAtlasCompositeGrid } from "../lib/methods/adaptive-mass/sparse-atlas-composite-projection";
import { WebGPUSparseCM12Resident } from "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident";
const dawnModule = process.env.WEBGPU_NODE_MODULE;
(dawnModule ? test : test.skip)("running resident fields survive an isolated generation replacement", { timeout: 120_000 }, async () => {
 await acquireWebGPUExclusiveLock("dawn-test", "sparse-cm12-resident-generation-dawn");
 let device: GPUDevice | undefined;
 const residents: WebGPUSparseCM12Resident[] = [];
 try {
  const dawn = await import(pathToFileURL(dawnModule!).href);
  Object.assign(globalThis, dawn.globals);
  const gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
  const adapter = (await gpu.requestAdapter())!;
  device = await adapter.requestDevice({ requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
  assert.ok(device);
  device.pushErrorScope("validation");
  const atlas = (r: 1 | 2 | 4 | 8) => createSparseAdaptiveMassAtlas([13, 15, 11], [{
   key: 0, coordinate: [0, 0, 0], spanBricks: 2, resolution: r,
   density: new Float64Array(r ** 3).fill(0.5), gamma: new Float64Array(r ** 3).fill(1),
  }], 0, 8);
  const first = atlas(4);
  const resident = await WebGPUSparseCM12Resident.create(device, first,
   buildSparseAtlasCompositeGrid(first), 0.05, createSolidWorld(), undefined,
   undefined, undefined, 8, undefined, 0);
  residents.push(resident); await resident.waitForSimulationPipelines();
  const source = await resident.captureGenerationTransferSource();
  assert.equal(source.cellIds.length, buildSparseAtlasCompositeGrid(first).cells.length);
  assert.ok(source.rowIds.every(id => id !== 0xffffffff), "every live face has a source");
  const next = await resident.prepareTransferredGeneration(source,
   createSparseAdaptiveMassAtlas(source.atlas.dimensions, source.atlas.bricks.map(b => ({...b,
    resolution:2, density:new Float64Array(8).fill(0.5),gamma:new Float64Array(8).fill(1)})), 1, 8, true), source.active, 0.05);
  residents.push(next);
  const fields = await next.readDiagnosticFields();
  assert.ok(fields.density.every(value => value === 0.5));
  assert.ok(fields.gamma.every(value => value === 1));
  assert.equal((await next.readActivitySnapshot()).faultFlags, 0);
  assert.equal(await device.popErrorScope(), null);
 } finally { for (const resident of residents) resident.destroy(); device?.destroy(); await releaseWebGPUExclusiveLock(); }
});
