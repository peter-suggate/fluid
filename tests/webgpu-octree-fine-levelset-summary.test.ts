import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { FineLevelSetBrickOracle, packFineLevelSetBrickKey, planFineLevelSetBricks } from
  "../lib/octree-fine-levelset-bricks";
import { WebGPUFineLevelSetBricks } from "../lib/webgpu-octree-fine-levelset-bricks";
import { FINE_LEVELSET_SUMMARY_VALID, WebGPUFineLevelSetSummaries } from
  "../lib/webgpu-octree-fine-levelset-summary";

test("Dawn publishes factor-4/factor-8 sparse fine summaries across moving interface generations", {
  skip: !process.env.WEBGPU_NODE_MODULE && "set WEBGPU_NODE_MODULE",
}, async () => {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href) as {
    create(options: string[]): GPU; globals: Record<string, unknown>;
  };
  Object.assign(globalThis, dawn.globals);
  const gpu = dawn.create([`backend=${process.env.WEBGPU_BACKEND ?? "metal"}`]);
  const adapter = await gpu.requestAdapter(); assert.ok(adapter); const device = await adapter.requestDevice();
  for (const factor of [4, 8] as const) {
    const plan = planFineLevelSetBricks({ domainOrigin: [0, 0, 0], finestCellDimensions: [4, 2, 2],
      finestCellWidth: 1, fineFactor: factor, brickResolution: 4,
      maximumResidentBricks: factor === 4 ? 16 : 128 });
    const owner = new WebGPUFineLevelSetBricks(device, plan); const summaries = new WebGPUFineLevelSetSummaries(device, plan);
    const oracle = new FineLevelSetBrickOracle(plan);
    for (const plane of [1.25, 2.25]) {
      const keys: number[] = []; const x = Math.floor(plane / (plan.brickResolution * plan.fineCellWidth));
      for (let z = 0; z < plan.brickDimensions[2]; z += 1) for (let y = 0; y < plan.brickDimensions[1]; y += 1) {
        keys.push(packFineLevelSetBrickKey(plan, [Math.min(plan.brickDimensions[0] - 1, x), y, z]));
      }
      oracle.publishInterfaceAndRing(keys, ([px]) => px - plane);
      const source = owner.uploadGeneration(oracle.exportGPUGeneration());
      const encoder = device.createCommandEncoder(); summaries.encode(encoder, source); device.queue.submit([encoder.finish()]);
    }
    await device.queue.onSubmittedWorkDone();
    const readback = device.createBuffer({ size: summaries.plan.directoryBytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = device.createCommandEncoder(); encoder.copyBufferToBuffer(summaries.directory, 0, readback, 0,
      summaries.plan.directoryBytes); device.queue.submit([encoder.finish()]); await device.queue.onSubmittedWorkDone();
    await readback.mapAsync(GPUMapMode.READ); const words = new Uint32Array(readback.getMappedRange().slice(0)); readback.unmap();
    assert.equal(words[0], 0); assert.equal(words[1], 2); assert.ok(words[2] > 0); assert.equal(words[9], FINE_LEVELSET_SUMMARY_VALID);
    const topKey = summaries.plan.levelOffsets[summaries.plan.maximumLevel]; let top: number | undefined;
    for (let slot = 0; slot < summaries.plan.hashCapacity; slot += 1) if (words[16 + slot * 8] === topKey) { top = 16 + slot * 8; break; }
    assert.notEqual(top, undefined); assert.ok(words[top! + 4] > 0); assert.ok(words[top! + 5] > 0);
    summaries.destroy(); owner.destroy(); readback.destroy();
  }
  device.destroy();
});
