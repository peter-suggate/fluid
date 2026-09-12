import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";

import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from
  "../lib/harness/webgpu-smoke-isolation";
import { geometricInterfaceResidentWGSL } from
  "../lib/methods/adaptive-volume/geometric-interface-resident.wgsl";

const productionFunction = geometricInterfaceResidentWGSL.match(
  /fn geometricResidentIntegrationSupported\([\s\S]*?\n}/,
)?.[0];
assert.ok(productionFunction);

const dawnTest = process.env.WEBGPU_NODE_MODULE ? test : test.skip;
dawnTest("production ELVIRA axis predicate rejects unsigned zero under reflection", async () => {
  await acquireWebGPUExclusiveLock("dawn-test", "geometric-interface-integration-axis");
  let gpu: GPU | undefined, device: GPUDevice | undefined;
  try {
    const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href);
    Object.assign(globalThis, dawn.globals);
    gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
    const adapter = await gpu.requestAdapter(); assert.ok(adapter);
    device = await adapter.requestDevice();
    const module = device.createShaderModule({ code: `
@group(0) @binding(0) var<storage,read_write> output:array<u32>;
${productionFunction}
@compute @workgroup_size(4) fn main(@builtin(global_invocation_id) gid:vec3u){
  let normals=array<vec3f,4>(vec3f(0.0,1.0,0.0),vec3f(-0.0,1.0,0.0),
    vec3f(1e-20,1.0,0.0),vec3f(-1e-20,1.0,0.0));
  output[gid.x]=select(0u,1u,geometricResidentIntegrationSupported(normals[gid.x],0u));
}` });
    assert.deepEqual((await module.getCompilationInfo()).messages.filter(m => m.type === "error"), []);
    const pipeline = await device.createComputePipelineAsync({ layout: "auto",
      compute: { module, entryPoint: "main" } });
    const output = device.createBuffer({ size: 16,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const readback = device.createBuffer({ size: 16,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: output } },
    ] });
    const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
    pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(1); pass.end();
    encoder.copyBufferToBuffer(output, 0, readback, 0, 16); device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    assert.deepEqual(Array.from(new Uint32Array(readback.getMappedRange())), [0, 0, 1, 1]);
    readback.unmap(); output.destroy(); readback.destroy();
  } finally { device?.destroy(); await releaseWebGPUExclusiveLock(); assert.ok(gpu); }
});
