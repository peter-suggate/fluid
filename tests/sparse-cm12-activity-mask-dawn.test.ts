import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from
  "../lib/harness/webgpu-smoke-isolation";

const source = readFileSync(new URL(
  "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.wgsl.ts", import.meta.url), "utf8");
const unpack = source.match(/fn cm12UnpackActivityFlags\([\s\S]*?\n}/)?.[0];
assert.ok(unpack);
const packing = source.match(/activityMasks\[lane\]=(vec2u\([\s\S]*?);/)?.[1];
assert.ok(packing);
const live = new Set<GPU>();
const dawnTest = process.env.WEBGPU_NODE_MODULE ? test : test.skip;

dawnTest("transport direction bits never become surface or occupancy flags", async () => {
  await acquireWebGPUExclusiveLock("dawn-test", "activity-mask-flags");
  let gpu: GPU | undefined, device: GPUDevice | undefined;
  try {
    const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href);
    Object.assign(globalThis, dawn.globals);
    gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]); live.add(gpu!);
    const adapter = await gpu!.requestAdapter(); assert.ok(adapter);
    device = await adapter.requestDevice();
    const shader = device.createShaderModule({ code: `
@group(0)@binding(0)var<storage,read_write>result:array<vec4u>;
${unpack}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id)gid:vec3u){
  let activityFlags=gid.x/32u;let direction=gid.x%32u;
  let supportMask=select(1u<<direction,0x07ffffffu,direction>=27u);
  let sweptSupportMask=select(supportMask,supportMask^0x07ffffffu,(activityFlags&1u)!=0u);
  let packed=${packing};
  result[gid.x]=vec4u(cm12UnpackActivityFlags(packed),
    packed.x&0x07ffffffu,packed.y&0x07ffffffu,
    (packed.x>>27u)|(packed.y>>22u));
}` });
    assert.deepEqual((await shader.getCompilationInfo()).messages.filter(m => m.type === "error").map(m => m.message), []);
    const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module: shader, entryPoint: "main" } });
    const bytes = 256 * 32 * 16;
    const result = device.createBuffer({ size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const copy = device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: result } }] });
    const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
    pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(128); pass.end();
    encoder.copyBufferToBuffer(result, 0, copy, 0, bytes); device.queue.submit([encoder.finish()]);
    await copy.mapAsync(GPUMapMode.READ);
    const words = new Uint32Array(copy.getMappedRange()); let oldFailures = 0;
    for (let flags = 0; flags < 256; flags++) for (let direction = 0; direction < 32; direction++) {
      const at = 4 * (32 * flags + direction);
      const support = direction < 27 ? 1 << direction : 0x07ffffff;
      const swept = flags & 1 ? support ^ 0x07ffffff : support;
      assert.equal(words[at], flags, `flags ${flags}, direction ${direction}`);
      assert.equal(words[at + 1], support); assert.equal(words[at + 2], swept);
      oldFailures += Number(words[at + 3] !== flags);
    }
    assert.ok(oldFailures > 0, "the fixture must expose the previous direction-to-flag leak");
    copy.unmap(); copy.destroy(); result.destroy();
  } finally {
    device?.destroy(); await releaseWebGPUExclusiveLock(); if (gpu) live.delete(gpu);
  }
});
