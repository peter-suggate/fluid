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
  "../lib/methods/adaptive-volume/webgpu-sparse-cm12-resident.wgsl.ts", import.meta.url,
), "utf8");
const functionSource = (name: string) => {
  const source = resident.match(new RegExp(`fn ${name}\\([\\s\\S]*?\\n}`))?.[0];
  assert.ok(source, `production WGSL function ${name}`);
  return source;
};
// Dawn's device does not keep its native instance alive during readback.
const live = new Set<GPU>();
const dawnTest = process.env.WEBGPU_NODE_MODULE ? test : test.skip;

dawnTest("coarse display interpolates a continuous bounded density field", async () => {
  await acquireWebGPUExclusiveLock("dawn-test", "presentation-density-continuity");
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
struct Params { dimensions:vec4u, dispatch:vec4u }
const p=Params(vec4u(128),vec4u(1));
const BRICK_FINE_RESOLUTION=8u;
const INVALID=0xffffffffu;
var<private>fixtureOffset:vec3i;
fn brickHasUnclippedWorldGeometry(brick:u32)->bool{_=brick;return any(fixtureOffset!=vec3i(0));}
fn cm12WorldOwnerAt(q:vec3i)->u32{_=q;return INVALID;}
fn cm12WorldFloorToSpan(q:i32,span:i32)->i32{return i32(floor(f32(q)/f32(span)))*span;}
fn brickActive(brick:u32)->bool{_=brick;return true;}
fn brickSpan(brick:u32)->u32{_=brick;return 1u;}
fn acceptedBrickResolution(brick:u32)->u32{_=brick;return 2u;}
fn cm12WorldLeafCoordinate(brick:u32)->vec3i{_=brick;return fixtureOffset/8;}
fn compactOwnerCellAt(q:vec3i)->vec3u{_=q;return vec3u(0);}
fn presentationIntegratedAdaptiveFloorHeight(x:i32,z:i32,offset:u32)->vec2f{
  _=offset;return vec2f(8.0*fixture(vec3i(floor(vec3f(f32(x),0,f32(z))/4.0))),1);
}
fn presentationIntegratedColumnHeight(brick:u32,x:i32,z:i32,offset:u32)->vec2f{
  _=brick;return presentationIntegratedAdaptiveFloorHeight(x,z,offset);
}
var<workgroup>presentationInterpolationCoefficients:array<vec4f,128>;
@group(0)@binding(0)var<storage,read_write>result:array<vec4f>;
fn fixture(q:vec3i)->f32{
  let x=clamp(q.x-fixtureOffset.x/4-2,0,3);
  let profile=array<f32,4>(.1,.2,.8,.9);
  return profile[u32(x)]+.01*f32(q.z-fixtureOffset.z/4-3);
}
fn presentationStencilDensityAt(q:vec3i,scale:u32,first:vec3i,dims:vec3u,fits:bool,offset:u32)->f32{
  _=scale;_=first;_=dims;_=fits;_=offset;return fixture(q);
}
${functionSource("interpolatedPresentationDensityAt")}
${functionSource("presentationLimitedSlope")}
${functionSource("preparePresentationInterpolationCache")}
${functionSource("smoothedPresentationDensityAt")}
${functionSource("presentationCanonicalCoarseCoordinate")}
${functionSource("presentationContinuousColumnHeight")}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id)gid:vec3u){
  let quadrant=gid.x/16u;
  fixtureOffset=vec3i(select(0,select(-32,160,quadrant==2u),quadrant>0u),0,
    select(0,select(-32,160,quadrant>=2u),quadrant>0u));
  let scale=4u;let q=vec3i(i32(gid.x%16u)+8,12,12)+fixtureOffset;
  let first=vec3i(2,3,3)+fixtureOffset/4;let dims=vec3u(4,1,1);
  preparePresentationInterpolationCache(gid.x,scale,first,dims,first-vec3i(1),dims+vec3u(2),0u,true);
  let actual=smoothedPresentationDensityAt(q,scale,first,dims);
  let position=(vec3f(q)+.5)/f32(scale)-.5;
  let lower=vec3i(floor(position));let t=fract(position);
  let expected=mix(mix(fixture(lower),fixture(lower+vec3i(1,0,0)),t.x),
    mix(fixture(lower+vec3i(0,0,1)),fixture(lower+vec3i(1,0,1)),t.x),t.z);
  let height=presentationContinuousColumnHeight(0u,q.x,q.z,0u,(gid.x&1u)==0u);
  result[gid.x]=vec4f(actual,expected,height.x,height.y);
}`;
    const pipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module: device.createShaderModule({ code: shader }), entryPoint: "main" },
    });
    const output = device.createBuffer({
      size: 64 * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
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
    pass.dispatchWorkgroups(1);
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
      assert.ok(Math.abs(values[i]! - values[i + 1]!) < 1e-6,
        `display must interpolate neighboring density means at sample ${i / 4}`);
      assert.ok(values[i]! >= .09 - 1e-6 && values[i]! <= .91 + 1e-6,
        "display must not create new density extrema");
      assert.ok(Math.abs(values[i + 2]! - 8 * values[i + 1]!) < 1e-6,
        "height receipts must interpolate on the same native X/Z centers");
      assert.equal(values[i + 3], 1, "valid neighboring receipts remain valid");
    }
    // The steep interval passes through a native coarse-cell face at x=16.
    // There must be no face jump beyond the continuous interpolant's slope.
    assert.ok(Math.abs((values[8 * 4]! - values[7 * 4]!) - .15) < 1e-6);
    for (let group = 1; group < 4; group++) for (let sample = 0; sample < 16; sample++) {
      const before = 4 * sample, translated = 4 * (16 * group + sample);
      assert.ok(Math.abs(values[translated]! - values[before]!) < 1e-6);
      assert.ok(Math.abs(values[translated + 2]! - values[before + 2]!) < 1e-6,
        "floor and ordinary height receipts must be invariant under signed XZ translation");
    }
  } finally {
    device?.destroy();
    await releaseWebGPUExclusiveLock();
    if (gpu) live.delete(gpu);
  }
});
