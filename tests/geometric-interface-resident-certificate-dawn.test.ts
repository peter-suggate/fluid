import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";

import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from
  "../lib/harness/webgpu-smoke-isolation";
import { geometricInterfaceResidentWGSL } from
  "../lib/methods/adaptive-volume/geometric-interface-resident.wgsl";

const productionFunction = geometricInterfaceResidentWGSL.match(
  /fn geometricResidentCertifiedFill\([\s\S]*?\n}/,
)?.[0];
assert.ok(productionFunction);

const dawnTest = process.env.WEBGPU_NODE_MODULE ? test : test.skip;
dawnTest("production interface certificate clamps roundoff and rejects non-finite fills", async () => {
  await acquireWebGPUExclusiveLock("dawn-test", "geometric-interface-certificate");
  let gpu: GPU | undefined, device: GPUDevice | undefined;
  try {
    const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href);
    Object.assign(globalThis, dawn.globals);
    gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
    const adapter = await gpu.requestAdapter(); assert.ok(adapter);
    device = await adapter.requestDevice();
    const module = device.createShaderModule({ code: `
@group(0) @binding(0) var<storage,read> input:array<f32>;
@group(0) @binding(1) var<storage,read_write> output:array<vec2f>;
fn geometricResidentFill(cell:u32,densityOffset:u32)->f32{_=densityOffset;return input[cell];}
${productionFunction}
@compute @workgroup_size(8) fn main(@builtin(global_invocation_id) gid:vec3u){
  if(gid.x<8u){output[gid.x]=geometricResidentCertifiedFill(gid.x,0u);}
}` });
    assert.deepEqual((await module.getCompilationInfo()).messages.filter(m => m.type === "error"), []);
    const pipeline = await device.createComputePipelineAsync({ layout: "auto",
      compute: { module, entryPoint: "main" } });
    const values = new Float32Array([-(2 ** -29), 0, 1, 1 + 2 ** -22,
      -(2 ** -19), 1 + 2 ** -19, Number.NaN, Number.POSITIVE_INFINITY]);
    const input = device.createBuffer({ size: values.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const output = device.createBuffer({ size: 8 * 8,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const readback = device.createBuffer({ size: output.size,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    device.queue.writeBuffer(input, 0, values);
    const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: input } }, { binding: 1, resource: { buffer: output } },
    ] });
    const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
    pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(1); pass.end();
    encoder.copyBufferToBuffer(output, 0, readback, 0, output.size);
    device.queue.submit([encoder.finish()]); await readback.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(readback.getMappedRange()).slice();
    assert.deepEqual(Array.from(result), [0, 1, 0, 1, 1, 1, 1, 1,
      0, 0, 0, 0, 0, 0, 0, 0]);
    readback.unmap(); input.destroy(); output.destroy(); readback.destroy();
  } finally { device?.destroy(); await releaseWebGPUExclusiveLock(); assert.ok(gpu); }
});
