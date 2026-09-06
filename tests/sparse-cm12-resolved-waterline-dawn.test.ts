import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from
  "../lib/harness/webgpu-smoke-isolation";

const source = readFileSync(new URL(
  "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.wgsl.ts", import.meta.url), "utf8");
function production(name: string): string {
  const body = source.match(new RegExp(`fn ${name}\\([\\s\\S]*?\\n}`))?.[0];
  assert.ok(body, `production function ${name}`);
  return body;
}
const live = new Set<GPU>();
const dawnTest = process.env.WEBGPU_NODE_MODULE ? test : test.skip;

dawnTest("resolved waterlines preserve sharp cell volume without column-mode switches", async () => {
  await acquireWebGPUExclusiveLock("dawn-test", "resolved-waterline");
  let gpu: GPU | undefined, device: GPUDevice | undefined;
  try {
    const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href);
    Object.assign(globalThis, dawn.globals);
    gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
    live.add(gpu!);
    const adapter = await gpu!.requestAdapter(); assert.ok(adapter);
    device = await adapter.requestDevice();
    const errors: string[] = [];
    device.addEventListener("uncapturederror", event => errors.push(event.error.message));
    const shader = device.createShaderModule({ code: `
struct Parameters { frame:vec4f, dispatch:vec4u, injectionCenter:vec4f }
const p=Parameters(vec4f(0,.05,0,0),vec4u(0,0,0,3),vec4f(0));
const CM12_LIQUID_ISOVALUE:f32=.5;
const BRICK_FINE_RESOLUTION:u32=8u;
fn acceptedBrickResolution(brick:u32)->u32{return select(8u,4u,brick==1u);}
fn cm12WorldLeafCoordinate(brick:u32)->vec3i{return vec3i(0,select(2,0,brick==2u),0);}
@group(0)@binding(0)var<storage,read_write>result:array<vec4f>;
${production("presentationResolvedColumnPhi")}
${production("presentationResolvedFineColumnPhi")}
${production("presentationLimitedSlope")}
${production("presentationHeightPolicyEnabled")}
fn sampleSharp(y:i32,f:f32)->f32{return clamp(f-f32(y),0.0,1.0);}
fn sharpPhi(y:i32,f:f32)->f32{
  var rho:array<f32,5>;
  for(var i=0u;i<5u;i+=1u){rho[i]=sampleSharp(y+i32(i)-2,f);}
  return presentationResolvedColumnPhi(rho);
}
fn refinedDensity(y:i32,height:f32)->f32{
  let position=(f32(y)+.5)/2.0;let parent=floor(position);
  let center=clamp((height-2.0*parent)/2.0,0.0,1.0);
  let back=clamp((height-2.0*(parent-1.0))/2.0,0.0,1.0);
  let forward=clamp((height-2.0*(parent+1.0))/2.0,0.0,1.0);
  return center+presentationLimitedSlope(back,center,forward)*(fract(position)-.5);
}
fn finePhi(y:i32,height:f32,refined:bool)->f32{
  var rho:array<f32,7>;
  for(var j=0u;j<7u;j+=1u){
    let sample=y+i32(j)-3;
    rho[j]=select(sampleSharp(sample,height),refinedDensity(sample,height),refined);
  }
  return presentationResolvedFineColumnPhi(rho);
}
fn fineHeight(height:f32,refined:bool)->f32{
  let lower=i32(floor(height-.5));let lo=finePhi(lower,height,refined);
  let hi=finePhi(lower+1,height,refined);
  return f32(lower)+.5-lo/(hi-lo);
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id)gid:vec3u){
  let i=gid.x;if(i>1024u){return;}let f=f32(i)/1024.0;
  let middle=sharpPhi(0,f);
  let left=sharpPhi(-1,f);let right=sharpPhi(1,f);
  // A full cell, a cell containing f, and air. Cell centres are -.5,.5,1.5.
  let height=select(-.5-left/(middle-left),.5-middle/(right-middle),f>=.5);
  result[i]=vec4f(height,middle,sharpPhi(0,1.0-f),
    select(0.0,1.0,presentationHeightPolicyEnabled(i%4u)));
  let center=.4+.2*f;
  let affine=presentationResolvedColumnPhi(array<f32,5>(center+.1,center+.05,center,center-.05,center-.1));
  let diffuse=presentationResolvedColumnPhi(array<f32,5>(min(1.0,.98+.04*f),.95,.6,.2,.01));
  let thin=presentationResolvedColumnPhi(array<f32,5>(0.0,0.0,f,0.0,0.0));
  result[1025u+i]=vec4f(affine,4.0*.05*(.5-center),diffuse,thin);
  let fineAffine=presentationResolvedFineColumnPhi(array<f32,7>(center+.15,
    center+.1,center+.05,center,center-.05,center-.1,center-.15));
  let fineThin=presentationResolvedFineColumnPhi(array<f32,7>(0.0,0.0,0.0,f,0.0,0.0,0.0));
  result[2050u+i]=vec4f(fineHeight(f,false),fineHeight(2.0*f,true),fineAffine,fineThin);
  let belowGap=presentationResolvedFineColumnPhi(array<f32,7>(1.0,0.0,0.0,f,0.0,0.0,0.0));
  let aboveGap=presentationResolvedFineColumnPhi(array<f32,7>(0.0,0.0,0.0,f,0.0,0.0,1.0));
  let poolUnderDrop=presentationResolvedFineColumnPhi(array<f32,7>(1.0,1.0,1.0,f,0.0,0.0,1.0));
  result[3075u+i]=vec4f(belowGap,aboveGap,poolUnderDrop,0.0);
}` });
    const info = await shader.getCompilationInfo();
    assert.deepEqual(info.messages.filter(m => m.type === "error"), []);
    const pipeline = device.createComputePipeline({ layout: "auto", compute: { module: shader, entryPoint: "main" } });
    const output = device.createBuffer({ size: 4 * 1025 * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const copy = device.createBuffer({ size: output.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: output } }] });
    const encoder = device.createCommandEncoder(); const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(17); pass.end();
    encoder.copyBufferToBuffer(output, 0, copy, 0, output.size); device.queue.submit([encoder.finish()]);
    await copy.mapAsync(GPUMapMode.READ);
    const values = new Float32Array(copy.getMappedRange()).slice();
    copy.unmap(); copy.destroy(); output.destroy();
    for (let i = 0; i <= 1024; i++) {
      assert.ok(Math.abs(values[4 * i]! - i / 1024) < 2e-7,
        `sharp waterline must preserve cell volume at fill ${i / 1024}`);
      assert.ok(Math.abs(values[4 * i + 1]! + values[4 * i + 2]!) < 1e-7,
        "liquid/air reflection must commute with reconstruction");
      assert.equal(values[4 * i + 3], [0, 0, 1, 0][i % 4],
        "deep interfaces must not switch to a column receipt; floor continuation remains");
      const gapAt = 4 * (3075 + i);
      for (const offset of [0, 1]) assert.ok(Math.abs(values[gapAt + offset]! - .2 * (.5 - i / 1024)) < 1e-7,
        "liquid beyond an air gap must not erase a detached sheet");
      assert.ok(Math.abs(values[gapAt + 2]! - .05 * (.5 - i / 1024)) < 1e-7,
        "a separate upper drop must not displace the lower pool interface");
      const fineAt = 4 * (2050 + i);
      assert.ok(Math.abs(values[fineAt]! - i / 1024) < 2e-7,
        "fine sharp waterline retains exact physical height");
      assert.ok(Math.abs(values[fineAt + 1]! - 2 * i / 1024) < 3e-7,
        `conservative refinement retains planar height at fill ${i / 1024}`);
      assert.ok(Math.abs(values[fineAt + 2]! - .2 * (.5 - (.4 + .2 * i / 1024))) < 1e-7,
        "extended fine bracket retains the affine scalar");
      assert.ok(Math.abs(values[fineAt + 3]! - .2 * (.5 - i / 1024)) < 1e-7,
        "extended fine bracket preserves an unbacked detached sheet");
      const at = 4 * (1025 + i);
      assert.ok(Math.abs(values[at]! - values[at + 1]!) < 1e-7,
        "affine diffuse density must retain its original scalar and zero");
      assert.ok(Math.abs(values[at + 3]! - .2 * (.5 - i / 1024)) < 1e-7,
        "an unbacked thin sheet must retain the density scalar");
      if (i > 0) {
        assert.ok(Math.abs(values[at + 2]! - values[at - 2]!) < 1e-5,
          "crossing the former 99% column threshold must not jump");
        const change = values[4 * (i - 1) + 1]! - values[4 * i + 1]!;
        assert.ok(change > 0 && change < .0008,
          "the resolved scalar must vary continuously, including at 1% and 99% fill");
      }
    }
    assert.deepEqual(errors, []);
  } finally {
    device?.destroy(); await releaseWebGPUExclusiveLock(); if (gpu) live.delete(gpu);
  }
});
