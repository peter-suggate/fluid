import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { pathToFileURL } from "node:url";
import {
  acquireWebGPUExclusiveLock,
  releaseWebGPUExclusiveLock,
} from "../lib/harness/webgpu-smoke-isolation";

// Exercise the emitted implementation rather than a JavaScript copy of it.
const resident = readFileSync(new URL(
  "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.wgsl.ts", import.meta.url,
), "utf8");
const functionSource = (name: string) => {
  const source = resident.match(new RegExp(`fn ${name}\\([\\s\\S]*?\\n}`))?.[0];
  assert.ok(source, `production WGSL function ${name}`);
  return source;
};
// Dawn's device does not keep its native instance alive during readback.
const live = new Set<GPU>();
const dawnTest = process.env.WEBGPU_NODE_MODULE ? test : test.skip;

dawnTest("coarse presentation and refinement reproduce affine cell averages", async () => {
  await acquireWebGPUExclusiveLock("dawn-test", "presentation-linear-reproduction");
  let gpu: GPU | undefined, device: GPUDevice | undefined;
  try {
    const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href);
    Object.assign(globalThis, dawn.globals);
    gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
    live.add(gpu!);
    const adapter = await gpu!.requestAdapter();
    assert.ok(adapter);
    device = await adapter.requestDevice();
    const gpuErrors: string[] = [];
    device.addEventListener("uncapturederror", (event) => gpuErrors.push(event.error.message));
    device.pushErrorScope("validation");
    const shader = `
struct Params { dimensions:vec4u }
const p=Params(vec4u(128));
var<workgroup>presentationInterpolationCoefficients:array<vec4f,128>;
@group(0)@binding(0)var<storage,read_write>result:array<vec4f>;
// Exact averages of one affine field over physical control volumes.
fn analytic(position:vec3f)->f32{return .2+dot(position,vec3f(.001,-.0007,.0003));}
fn restrictedPresentationDensityAt(lower:vec3i,scale:i32,offset:u32)->f32{
  _=offset;return analytic(vec3f(lower)+vec3f(.5*f32(scale)));
}
fn presentationStencilDensityAt(q:vec3i,scale:u32,first:vec3i,dims:vec3u,fits:bool,offset:u32)->f32{
  _=first;_=dims;_=fits;return restrictedPresentationDensityAt(q*i32(scale),i32(scale),offset);
}
${functionSource("presentationLimitedSlope")}
${functionSource("preparePresentationInterpolationCache")}
${functionSource("interpolatedPresentationDensityAt")}
${functionSource("smoothedPresentationDensityAt")}
${functionSource("directSmoothedPresentationDensityAt")}
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index)lane:u32,@builtin(workgroup_id)group:vec3u){
  let scale=2u<<group.x;let first=vec3i(i32(32u/scale));let dims=vec3u(8u/scale);
  preparePresentationInterpolationCache(lane,scale,first,dims,first-vec3i(1),dims+vec3u(2),0u,true);
  for(var i=lane;i<512u;i+=64u){
    let local=vec3u(i%8u,(i/8u)%8u,i/64u);
    let q=vec3i(vec3u(32)+local);
    let expected=analytic(vec3f(q)+vec3f(.5));
    let cached=smoothedPresentationDensityAt(q,scale,first,dims);
    let direct=directSmoothedPresentationDensityAt(q,scale,0u);
    result[group.x*512u+i]=vec4f(cached,direct,expected,0);
  }
}`;
    const pipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module: device.createShaderModule({ code: shader }), entryPoint: "main" },
    });
    const output = device.createBuffer({
      size: 3 * 512 * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const readback = device.createBuffer({
      size: output.size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const group = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: output } }],
    });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(3);
    pass.end();
    encoder.copyBufferToBuffer(output, 0, readback, 0, output.size);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const values = new Float32Array(readback.getMappedRange()).slice();
    readback.unmap();
    readback.destroy();
    output.destroy();
    const validationError = await device.popErrorScope();
    assert.equal(validationError, null, validationError?.message);
    assert.deepEqual(gpuErrors, [], "GPU execution must succeed");
    let maximumError = 0;
    for (let i = 0; i < values.length; i += 4) {
      const local = (i / 4) % 512;
      const expected = .2 + (32.5 + local % 8) * .001
        - (32.5 + Math.floor(local / 8) % 8) * .0007
        + (32.5 + Math.floor(local / 64)) * .0003;
      assert.ok(Math.abs(values[i + 2]! - expected) < 1e-6, "shader must produce the analytic fixture");
      assert.ok(Math.abs(values[i]! - values[i + 1]!) < 1e-7,
        "continuous display and conservative refinement reproduce the same affine field");
      maximumError = Math.max(maximumError, Math.abs(values[i]! - values[i + 2]!));
    }
    assert.ok(maximumError < 1e-6, `an affine field must survive reconstruction: error ${maximumError}`);
    // Every rung reconstructs the same physical fine samples, including points
    // on both sides of parent-cell boundaries. Changing the source width at a
    // 2:1 seam must not change the reconstructed affine field.
    for (let rung = 1; rung < 3; rung += 1) {
      for (let i = 0; i < 512; i += 1) {
        assert.ok(Math.abs(values[(rung * 512 + i) * 4]! - values[i * 4]!) < 1e-6, "the same position must agree across 2:1 source widths");
      }
    }
    for (let rung = 0; rung < 3; rung += 1) {
      const width = 2 << rung;
      for (let z = 0; z < 8; z += width) {
        for (let y = 0; y < 8; y += width) {
          for (let x = 0; x < 8; x += width) {
            let sum = 0;
            let expected = 0;
            for (let dz = 0; dz < width; dz += 1) {
              for (let dy = 0; dy < width; dy += 1) {
                for (let dx = 0; dx < width; dx += 1) {
                  const at = 4 * (rung * 512 + x + dx + 8 * (y + dy + 8 * (z + dz)));
                  sum += values[at]!;
                  expected += values[at + 2]!;
                }
              }
            }
            assert.ok(Math.abs(sum - expected) / width ** 3 < 1e-7, "each reconstructed parent must retain its mean");
          }
        }
      }
    }
  } finally {
    device?.destroy();
    await releaseWebGPUExclusiveLock();
    if (gpu) live.delete(gpu);
  }
});
