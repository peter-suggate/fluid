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

dawnTest("native volume scalar preserves planar subcell waterlines at every coarse rung", async () => {
  await acquireWebGPUExclusiveLock("dawn-test", "presentation-native-volume");
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
struct Params { dimensions:vec4u, frame:vec4f }
const p=Params(vec4u(128),vec4f(0,.05,0,0));
const CM12_LIQUID_ISOVALUE=.5;
const BRICK_FINE_RESOLUTION=8u;
const INVALID=0xffffffffu;
const cm12PresentationBrick=0u;
fn brickHasUnclippedWorldGeometry(brick:u32)->bool{_=brick;return false;}
fn cm12WorldOwnerAt(q:vec3i)->u32{_=q;return INVALID;}
fn cm12WorldFloorToSpan(q:i32,span:i32)->i32{return i32(floor(f32(q)/f32(span)))*span;}
var<private>fixtureHeight:f32;
@group(0)@binding(0)var<storage,read_write>result:array<vec4f>;
fn presentationStencilDensityAt(q:vec3i,scale:u32,first:vec3i,dims:vec3u,fits:bool,offset:u32)->f32{
  _=first;_=dims;_=fits;_=offset;
  return clamp((fixtureHeight-f32(q.y)*f32(scale))/f32(scale),0.0,1.0);
}
${functionSource("presentationResolvedColumnPhi")}
${functionSource("presentationCanonicalCoarseCoordinate")}
${functionSource("presentationInteriorColumnPhi")}
${functionSource("presentationColumnContinuation")}
${functionSource("presentationContinuationWeights")}
${functionSource("presentationCoarseColumnPhi")}
${functionSource("presentationVolumeWeights")}
${functionSource("presentationInterpolatedVolumePhi")}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id)gid:vec3u){
  if(gid.x>=4u*1025u){return;}
  let rung=gid.x/1025u;let scale=1u<<rung;
  fixtureHeight=8.0+f32(scale)*f32(gid.x%1025u)/1024.0;
  let lower=i32(floor(fixtureHeight-.5));
  let lo=presentationInterpolatedVolumePhi(vec3i(32,lower,32),scale,vec3i(0),vec3u(0),false,0u);
  let hi=presentationInterpolatedVolumePhi(vec3i(32,lower+1,32),scale,vec3i(0),vec3u(0),false,0u);
  result[gid.x]=vec4f(lo,hi,fixtureHeight,f32(lower)+.5);
}`;
    const pipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module: device.createShaderModule({ code: shader }), entryPoint: "main" },
    });
    const output = device.createBuffer({
      size: 4 * 1025 * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
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
    pass.dispatchWorkgroups(Math.ceil(4 * 1025 / 64));
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
    for (let i = 0; i < values.length; i += 4) {
      const low = values[i]!, high = values[i + 1]!;
      const height = values[i + 2]!, lowerCenter = values[i + 3]!;
      const crossing = lowerCenter - low / (high - low);
      assert.ok(Math.abs(crossing - height) < 1e-5,
        `native width ${1 << Math.floor(i / 4 / 1025)} moved height ${height}: ${crossing}`);
      assert.ok(Math.abs(low - .05 * (lowerCenter - height)) < 1e-6);
      assert.ok(Math.abs(high - .05 * (lowerCenter + 1 - height)) < 1e-6,
        "all rungs must publish the same physical signed-distance units");
    }
  } finally {
    device?.destroy();
    await releaseWebGPUExclusiveLock();
    if (gpu) live.delete(gpu);
  }
});
