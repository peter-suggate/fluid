import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { createSparseAdaptiveMassAtlas } from "../lib/methods/adaptive-mass/sparse-brick-atlas";
import {
  createSparseCM12WorldDirectoryInitialWords, createSparseCM12WorldDirectoryLayout,
  createSparseCM12WorldDirectoryWGSL, sparseCM12WorldCoordinateHash,
  SPARSE_CM12_WORLD_DIRECTORY_HEADER as H,
  SPARSE_CM12_WORLD_DIRECTORY_ENTRY_WORDS as ENTRY_WORDS,
} from "../lib/methods/adaptive-mass/sparse-cm12-world-directory";

const dawnModule = process.env.WEBGPU_NODE_MODULE;
(dawnModule ? test : test.skip)("contended world directory allocation stays unique across tombstone reuse", async () => {
  await acquireWebGPUExclusiveLock("dawn-test", "CM12 world-directory contention");
  let device: GPUDevice | undefined;
  const buffers: GPUBuffer[] = [];
  try {
    const dawn = await import(pathToFileURL(dawnModule!).href);
    Object.assign(globalThis, dawn.globals);
    const gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
    device = await (await gpu.requestAdapter())!.requestDevice();
    assert.ok(device);
    device.pushErrorScope("validation");
    const layout = createSparseCM12WorldDirectoryLayout({ initialLeaves: 0, growthLeaves: 64, maximumSpanLog: 0 });
    const words = createSparseCM12WorldDirectoryInitialWords(layout,
      createSparseAdaptiveMassAtlas([8, 8, 8], [], 0, 8));
    const arena = device.createBuffer({ size: words.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const coords = device.createBuffer({ size: 32 * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const readback = device.createBuffer({ size: words.byteLength, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    buffers.push(arena, coords, readback);
    device.queue.writeBuffer(arena, 0, new Uint32Array(words));
    // All coordinates share the starting bucket. Thousands of repeated sources
    // contend for them; retries must eventually publish exactly one leaf each.
    const coordinates: number[] = [];
    for (let x = 1; coordinates.length < 32 * 4; x++) {
      if ((sparseCM12WorldCoordinateHash([x, -2, -2], 0) & (layout.capacity - 1)) === 1)
        coordinates.push(x, -2, -2, 0);
    }
    device.queue.writeBuffer(coords, 0, new Int32Array(coordinates));
    const module = device.createShaderModule({ code: `
      @group(0) @binding(0) var<storage,read_write> topologyArena:array<atomic<u32>>;
      @group(0) @binding(1) var<storage,read> coordinates:array<vec4i>;
      ${createSparseCM12WorldDirectoryWGSL(layout)}
      @compute @workgroup_size(64)
      fn allocate(@builtin(global_invocation_id) gid:vec3u){
        let leaf=cm12WorldAllocateExact(coordinates[gid.x%32u].xyz,0u);
      }
      @compute @workgroup_size(64)
      fn retire(@builtin(global_invocation_id) gid:vec3u){
        if(gid.x<64u){let released=cm12WorldReleaseLeaf(gid.x);}
      }
    ` });
    const bindingLayout = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
    ] });
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bindingLayout] });
    const pipelines = new Map<string, GPUComputePipeline>();
    for (const entryPoint of ["allocate", "finalizeSparseWorldDirectoryAllocations", "retire"])
      pipelines.set(entryPoint, await device.createComputePipelineAsync({ layout: pipelineLayout, compute: { module, entryPoint } }));
    const bindings = device.createBindGroup({ layout: bindingLayout, entries: [
      { binding: 0, resource: { buffer: arena } }, { binding: 1, resource: { buffer: coords } },
    ] });
    for (let cycle = 0; cycle < 6; cycle++) {
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setBindGroup(0, bindings);
      const dispatch = (name: string, count: number) => { pass.setPipeline(pipelines.get(name)!); pass.dispatchWorkgroups(count); };
      if (cycle > 0) dispatch("retire", 1);
      for (let retry = 0; retry < 64; retry++) {
        dispatch("allocate", 64);
        dispatch("finalizeSparseWorldDirectoryAllocations", Math.ceil(layout.capacity / 64));
      }
      pass.end();
      encoder.copyBufferToBuffer(arena, 0, readback, 0, words.byteLength);
      device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const result = new Uint32Array(readback.getMappedRange());
      const entries: number[] = [];
      for (let slot = 0; slot < layout.capacity; slot++) {
        const at = layout.entryBaseWords + slot * ENTRY_WORDS;
        assert.notEqual(result[at], 1, "no unpublished reservation survives the finalize dispatch");
        if (result[at] === 2) entries.push(result[at + 2]!);
      }
      assert.equal(entries.length, 32, `cycle ${cycle}: one entry per coordinate`);
      assert.equal(new Set(entries).size, 32, `cycle ${cycle}: no duplicate coordinates`);
      assert.equal(result[H.liveCount], 32);
      assert.equal(result[H.capacityFaults], 0);
      assert.equal(result[H.insertionFaults], 0);
      readback.unmap();
    }
    assert.equal(await device.popErrorScope(), null);
  } finally {
    for (const buffer of buffers) buffer.destroy();
    device?.destroy();
    await releaseWebGPUExclusiveLock();
  }
});
