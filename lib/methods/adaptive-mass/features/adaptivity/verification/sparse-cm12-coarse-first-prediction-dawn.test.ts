import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../../../../../harness/webgpu-smoke-isolation";
import { sparseCM12CoarseFirstPredictionWGSL } from "../sparse-cm12-coarse-first-prediction.wgsl";

const dawnModule = process.env.WEBGPU_NODE_MODULE;
(dawnModule ? test : test.skip)("coarse-first prediction sizes relative entry motion, preserving real impacts", async () => {
  await acquireWebGPUExclusiveLock("dawn-test", "coarse-first-prediction");
  let device: GPUDevice | undefined;
  try {
    const dawn = await import(pathToFileURL(dawnModule!).href);
    Object.assign(globalThis, dawn.globals);
    const gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
    const adapter = await gpu.requestAdapter(); assert.ok(adapter);
    device = await adapter.requestDevice();
    const cases = [
      { name: "slow lateral entry with fast tangential travel", delta: [8, 0, 0], source: [.1, -12, 0], receiver: [0, 0, 0], expected: .05 },
      { name: "coherent translating liquid", delta: [8, 0, 0], source: [20, -12, 0], receiver: [20, -12, 0], expected: 0 },
      { name: "mini32 neighbour recedes in receiver frame", delta: [8, 0, 0], source: [.1046, -11.1942, .1046], receiver: [.261, -12.0382, .0622], expected: 0 },
      { name: "fast separated impact", delta: [0, -24, 0], source: [0, -64, 0], receiver: [0, 0, 0], expected: 32 },
      { name: "impact on moving liquid", delta: [0, -24, 0], source: [0, -80, 0], receiver: [0, -16, 0], expected: 32 },
      { name: "outside horizon", delta: [0, -24, 0], source: [0, -16, 0], receiver: [0, 0, 0], expected: 0 },
      { name: "pure tangent to touching face", delta: [8, 0, 0], source: [0, -64, 0], receiver: [0, 0, 0], expected: 0 },
      { name: "leaves tangent slab before normal entry", delta: [16, 0, 0], source: [32, 80, 0], receiver: [0, 0, 0], expected: 0 },
      { name: "diagonal touching faces use slower entry", delta: [8, 8, 0], source: [32, 4, 0], receiver: [0, 0, 0], expected: 2 },
      { name: "later entry face sets travel", delta: [16, 12, 0], source: [32, 12, 0], receiver: [0, 0, 0], expected: 6 },
    ];
    // Exercise every axis and reflection: geometry must not depend on which
    // direction happens to be vertical in the mini dam fixture.
    const queries: number[] = [], expected: { name: string; value: number }[] = [];
    for (const c of cases) for (let axis = 0; axis < 3; axis++) for (const sign of [-1, 1]) {
      const rotate = (v: number[]) => v.map((_, i) => sign * v[(i + axis) % 3]!);
      queries.push(...rotate(c.delta), 8, ...rotate(c.source), .5, ...rotate(c.receiver), 0);
      expected.push({ name: `${c.name} axis=${axis} sign=${sign}`, value: c.expected });
    }
    const module = device.createShaderModule({ code: `
@group(0)@binding(0)var<storage,read>input:array<vec4f>;
@group(0)@binding(1)var<storage,read_write>output:array<f32>;
${sparseCM12CoarseFirstPredictionWGSL}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id)gid:vec3u){
  if(gid.x>=arrayLength(&output)){return;}
  let d=input[3u*gid.x];let s=input[3u*gid.x+1u];let r=input[3u*gid.x+2u];
  output[gid.x]=coarseFirstApproachTravel(d.xyz,s.w*(s.xyz-r.xyz),d.w);
}` });
    const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "main" } });
    const input = device.createBuffer({ size: queries.length * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(input, 0, new Float32Array(queries));
    const output = device.createBuffer({ size: expected.length * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const readback = device.createBuffer({ size: output.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const bindings = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [input, output].map((buffer, binding) => ({ binding, resource: { buffer } })) });
    const encoder = device.createCommandEncoder(); const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline); pass.setBindGroup(0, bindings); pass.dispatchWorkgroups(Math.ceil(expected.length / 64)); pass.end();
    encoder.copyBufferToBuffer(output, 0, readback, 0, output.size); device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const actual = new Float32Array(readback.getMappedRange());
    for (const [i, c] of expected.entries()) assert.ok(Math.abs(actual[i]! - c.value) < 1e-5,
      `${c.name}: ${actual[i]} != ${c.value}`);
    readback.unmap(); input.destroy(); output.destroy(); readback.destroy();
  } finally { device?.destroy(); await releaseWebGPUExclusiveLock(); }
});
