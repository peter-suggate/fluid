import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { createSparseAdaptiveMassAtlas } from "../lib/methods/adaptive-volume/sparse-brick-atlas";
import {
  createSparseCM12WorldDirectoryInitialWords, createSparseCM12WorldDirectoryLayout,
  createSparseCM12WorldDirectoryWGSL, sparseCM12WorldCoordinateHash,
  SPARSE_CM12_WORLD_DIRECTORY_HEADER as H,
  SPARSE_CM12_WORLD_DIRECTORY_ENTRY_WORDS as ENTRY_WORDS,
} from "../lib/methods/adaptive-volume/sparse-cm12-world-directory";

const dawnModule = process.env.WEBGPU_NODE_MODULE;
for (const uniqueRequest of [false, true]) (dawnModule ? test : test.skip)(uniqueRequest
  ? "unique world-directory requests complete every colliding key in one dispatch across tombstone reuse"
  : "contended world directory allocation stays unique across tombstone reuse", async () => {
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
    const keyCount = uniqueRequest ? 64 : 32;
    const layout = createSparseCM12WorldDirectoryLayout({ initialLeaves: 0, growthLeaves: 64, maximumSpanLog: 0 });
    const words = createSparseCM12WorldDirectoryInitialWords(layout,
      createSparseAdaptiveMassAtlas([8, 8, 8], [], 0, 8));
    const arena = device.createBuffer({ size: words.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const coords = device.createBuffer({ size: keyCount * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const readback = device.createBuffer({ size: words.byteLength, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    buffers.push(arena, coords, readback);
    device.queue.writeBuffer(arena, 0, new Uint32Array(words));
    // All coordinates share the starting bucket. Thousands of repeated sources
    // contend for them; retries must eventually publish exactly one leaf each.
    const coordinates: number[] = [];
    for (let x = 1; coordinates.length < keyCount * 4; x++) {
      if ((sparseCM12WorldCoordinateHash([x, -2, -2], 0) & (layout.capacity - 1)) === 1)
        coordinates.push(x, -2, -2, 0);
    }
    device.queue.writeBuffer(coords, 0, new Int32Array(coordinates));
    const shader = device.createShaderModule({ code: `
      @group(0) @binding(0) var<storage,read_write> topologyArena:array<atomic<u32>>;
      @group(0) @binding(1) var<storage,read> coordinates:array<vec4i>;
      ${createSparseCM12WorldDirectoryWGSL(layout)}
      @compute @workgroup_size(64)
      fn allocate(@builtin(global_invocation_id) gid:vec3u){
        ${uniqueRequest ? `if(gid.x>=${keyCount}u){return;}` : ""}
        let leaf=${uniqueRequest ? "cm12WorldAllocateUniqueExact" : "cm12WorldAllocateExact"}(coordinates[gid.x%${keyCount}u].xyz,0u);
      }
      @compute @workgroup_size(64)
      fn baseline(@builtin(global_invocation_id) gid:vec3u){
        if(gid.x>=${keyCount}u){return;}
        let leaf=cm12WorldAllocateExact(coordinates[gid.x].xyz,0u);
      }
      @compute @workgroup_size(64)
      fn verify(@builtin(global_invocation_id) gid:vec3u){
        if(gid.x>=${keyCount}u){return;}
        if(cm12WorldLookupExact(coordinates[gid.x].xyz,0u)==CM12_WDR_INVALID){
          atomicAdd(&topologyArena[CM12_WDR_BASE+${H.insertionFaults}u],1u);
        }
      }
      @compute @workgroup_size(1)
      fn exhaust(){
        let leaf=cm12WorldAllocateUniqueExact(coordinates[0].xyz+vec3i(0,1,0),0u);
        if(leaf!=CM12_WDR_INVALID){atomicAdd(&topologyArena[CM12_WDR_BASE+${H.insertionFaults}u],1u);}
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
    for (const entryPoint of ["allocate", "baseline", "finalizeSparseWorldDirectoryAllocations", "verify", "exhaust", "retire"])
      pipelines.set(entryPoint, await device.createComputePipelineAsync({ layout: pipelineLayout, compute: { module: shader, entryPoint } }));
    const bindings = device.createBindGroup({ layout: bindingLayout, entries: [
      { binding: 0, resource: { buffer: arena } }, { binding: 1, resource: { buffer: coords } },
    ] });
    if (uniqueRequest) {
      // The original duplicate-capable insertion publishes only one of these
      // distinct keys in one epoch. This is the missing-support mechanism,
      // rather than a test that merely mirrors the repaired implementation.
      const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
      pass.setPipeline(pipelines.get("baseline")!); pass.setBindGroup(0, bindings); pass.dispatchWorkgroups(1); pass.end();
      encoder.copyBufferToBuffer(arena, 0, readback, 0, words.byteLength); device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      assert.equal(new Uint32Array(readback.getMappedRange())[H.liveCount], 1,
        "the original reservation rule must reproduce incomplete same-epoch support");
      readback.unmap(); device.queue.writeBuffer(arena, 0, new Uint32Array(words));
    }
    for (let cycle = 0; cycle < 6; cycle++) {
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setBindGroup(0, bindings);
      const dispatch = (name: string, count: number) => { pass.setPipeline(pipelines.get(name)!); pass.dispatchWorkgroups(count); };
      if (cycle > 0) dispatch("retire", 1);
      for (let retry = 0; retry < (uniqueRequest ? 1 : 64); retry++) {
        dispatch("allocate", uniqueRequest ? 1 : 64);
        dispatch("finalizeSparseWorldDirectoryAllocations", Math.ceil(layout.capacity / 64));
      }
      dispatch("verify", 1);
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
      assert.equal(entries.length, keyCount, `cycle ${cycle}: one entry per coordinate`);
      assert.equal(new Set(entries).size, keyCount, `cycle ${cycle}: no duplicate coordinates`);
      assert.equal(result[H.liveCount], keyCount);
      assert.equal(result[H.capacityFaults], 0);
      assert.equal(result[H.insertionFaults], 0);
      readback.unmap();
    }
    if (uniqueRequest) {
      const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
      pass.setPipeline(pipelines.get("exhaust")!); pass.setBindGroup(0, bindings); pass.dispatchWorkgroups(1); pass.end();
      encoder.copyBufferToBuffer(arena, 0, readback, 0, words.byteLength); device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ); const result = new Uint32Array(readback.getMappedRange());
      assert.equal(result[H.capacityFaults], 1, "capacity exhaustion must be explicit");
      assert.equal(result[H.liveCount], keyCount, "a rejected allocation cannot create or overwrite a leaf");
      assert.equal(result[H.insertionFaults], 0); readback.unmap();
    }
    assert.equal(await device.popErrorScope(), null);
  } finally {
    for (const buffer of buffers) buffer.destroy();
    device?.destroy();
    await releaseWebGPUExclusiveLock();
  }
});
