import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
const live = new Set<GPU>();
(process.env.WEBGPU_NODE_MODULE ? test : test.skip)("residue cleanup preserves wet, unresolved and nonfinite pages and records deletion once", async () => {
  await acquireWebGPUExclusiveLock("dawn-test", "residue-deletion");
  let device: GPUDevice | undefined;
  try {
    const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href); Object.assign(globalThis, dawn.globals);
    const gpu = dawn.create(["backend=metal"]); live.add(gpu);
    const adapter = await gpu.requestAdapter(); assert.ok(adapter); device = await adapter.requestDevice(); assert.ok(device);
    const source = readFileSync(new URL("../lib/methods/adaptive-volume/resident-volume.wgsl.ts", import.meta.url), "utf8");
    const reduction = source.match(/fn gvAddPhiReduction\([\s\S]*?\n}/)![0];
    const cleanup = source.slice(source.indexOf("var<workgroup> gvResidueReject"), source.indexOf("// One bounded global volume feedback step."));
    const code = `
@group(0)@binding(0)var<storage,read_write>state:array<f32>;
@group(0)@binding(1)var<storage,read_write>conditioning:array<atomic<i32>>;
struct Params {dispatch:vec4u}; const p=Params(vec4u(0u,0u,0u,8u));
const GV_CURRENT=16u;const GV_LOW=32u;const GV_WHOLE_FRAME_CONTROL=0u;
fn brickActive(b:u32)->bool{return true;}
fn gvFailed()->bool{return false;}
fn acceptedBrickResolution(b:u32)->u32{return 2u;}
fn templateBrickCellRange(b:u32,r:u32)->vec2u{return select(vec2u(b,1u),vec2u(2u*b,2u),r==2u);}
fn destinationDensity()->u32{return 0u;}
fn cellVolume(c:u32)->f32{return 8.0;}
fn cellCenter(c:u32)->vec3f{return vec3f(f32(c),0.0,0.0);}
fn cellWidths(c:u32)->vec3f{return vec3f(1.0);}
fn incrementalActivityMarkCellClosure(c:u32){atomicAdd(&conditioning[28u],1);}
const LSV_INVALID=0xffffffffu;const LSV_SUPPORT_DEEP_AIR=2u;
fn lsvFinite(v:f32)->bool{return abs(v)<=3.402823466e38;}
fn lsvAccepted()->bool{return true;}
fn lsvAcceptedSlot()->u32{return 0u;}
fn lsvBrickPhiResolution(slot:u32,brick:u32)->u32{return 1u;}
fn lsvBrickPhiBase(slot:u32,brick:u32)->u32{return brick;}
struct Stencil {resolved:bool,support:array<u32,8>,phi:array<f32,8>}
fn lsvStencilAtOrdinal(slot:u32,page:u32)->Stencil{
 var s:Stencil;s.resolved=page!=2u;
 for(var i=0u;i<8u;i++){s.support[i]=2u;s.phi[i]=0.5;}
 if(page==1u){s.phi[0]=-1.0;}
 if(page==7u){s.support[0]=1u;}
 return s;
}
${reduction}
${cleanup}`;
    const module = device.createShaderModule({ code });
    assert.deepEqual((await module.getCompilationInfo()).messages.filter(m => m.type === "error"), []);
    const values = new Float32Array(48); values.set([1e-5,2e-5, 1e-5,1e-5, 1e-5,1e-5, .01,1e-5, NaN,1e-5, -1e-8,1e-5, 1e-4,1e-4, 1e-5,1e-5]);
    const state = device.createBuffer({ size:192, usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST });
    const counters = device.createBuffer({ size:128, usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC });
    const readback = device.createBuffer({ size:320, usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(state,0,values);
    const pipeline = device.createComputePipeline({ layout:"auto", compute:{module,entryPoint:"deleteTinyVolumeResidues"} });
    const group = device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:state}},{binding:1,resource:{buffer:counters}}]});
    const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,group);
    pass.dispatchWorkgroups(8);pass.dispatchWorkgroups(8);pass.end();
    encoder.copyBufferToBuffer(state,0,readback,0,192);encoder.copyBufferToBuffer(counters,0,readback,192,128);
    device.queue.submit([encoder.finish()]);await readback.mapAsync(GPUMapMode.READ);
    const result=new Float32Array(readback.getMappedRange());const counts=new Uint32Array(result.buffer,192);
    for(let i=0;i<16;i++) if([0,1,12,13].includes(i)) assert.equal(result[i],0);else assert.ok(Object.is(result[i],values[i]));
    const expected=8*(values[0]!+values[1]!+values[12]!+values[13]!);
    assert.ok(Math.abs(result[48+24]!-expected)<1e-9);assert.equal(result[48+25],result[48+24]);
    assert.equal(counts[26],2);assert.equal(counts[27],2);assert.equal(counts[28],4);
    readback.unmap();state.destroy();counters.destroy();readback.destroy();
  } finally {device?.destroy();live.clear();await releaseWebGPUExclusiveLock();}
});

(process.env.WEBGPU_NODE_MODULE ? test : test.skip)("air roundoff cleanup preserves meaningful, wet and unresolved donors and records loss once", async () => {
  await acquireWebGPUExclusiveLock("dawn-test", "air-roundoff-residue");
  let device: GPUDevice | undefined;
  try {
    const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href); Object.assign(globalThis, dawn.globals);
    const gpu = dawn.create(["backend=metal"]); live.add(gpu);
    const adapter = await gpu.requestAdapter(); assert.ok(adapter); device = await adapter.requestDevice(); assert.ok(device);
    const source = readFileSync(new URL("../lib/methods/adaptive-volume/resident-volume.wgsl.ts", import.meta.url), "utf8");
    const reduction = source.match(/fn gvAddPhiReduction\([\s\S]*?\n}/)![0];
    const cleanup = source.match(/fn gvDeleteAirRoundoffResidue\([\s\S]*?\n}/)![0];
    const roundoff = source.match(/fn gvRoundoff\([^\n]+/)![0];
    const module = device.createShaderModule({ code: `
@group(0)@binding(0)var<storage,read_write>state:array<f32>;
@group(0)@binding(1)var<storage,read_write>conditioning:array<atomic<i32>>;
const GV_CURRENT=8u; const GV_WHOLE_FRAME_CONTROL=0u;
fn destinationDensity()->u32{return 0u;}
struct Phase {valid:bool,phi:f32}
fn lsvCellSample(c:u32)->Phase{return Phase(c!=3u,select(1.0,-1.0,c==2u));}
fn incrementalActivityMarkCellClosure(c:u32){atomicAdd(&conditioning[28u],1);}
${roundoff}
${reduction}
${cleanup}
@compute @workgroup_size(1)fn run(@builtin(global_invocation_id)id:vec3u){gvDeleteAirRoundoffResidue(id.x,8.0);}` });
    assert.deepEqual((await module.getCompilationInfo()).messages.filter(m => m.type === "error"), []);
    const values = new Float32Array([3.1114633e-28,1e-6,1e-6,1e-6,1e-3,0,-1e-8,NaN]);
    const initial = new Float32Array(16); initial.set(values); initial.set(values,8);
    const state = device.createBuffer({size:64,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
    const counters = device.createBuffer({size:128,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
    const readback = device.createBuffer({size:192,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
    device.queue.writeBuffer(state,0,initial);
    const pipeline = device.createComputePipeline({layout:"auto",compute:{module,entryPoint:"run"}});
    const group = device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:state}},{binding:1,resource:{buffer:counters}}]});
    const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,group);
    pass.dispatchWorkgroups(8);pass.dispatchWorkgroups(8);pass.end();
    encoder.copyBufferToBuffer(state,0,readback,0,64);encoder.copyBufferToBuffer(counters,0,readback,64,128);
    device.queue.submit([encoder.finish()]);await readback.mapAsync(GPUMapMode.READ);
    const result=new Float32Array(readback.getMappedRange());
    for(let i=0;i<16;i++) assert.ok(Object.is(result[i],i%8<2?0:initial[i]),`state ${i}`);
    assert.equal(result[16+24],values[1]);assert.equal(result[16+25],values[1]);
    assert.equal(new Uint32Array(result.buffer)[16+28],2);
    readback.unmap();state.destroy();counters.destroy();readback.destroy();
  } finally {device?.destroy();live.clear();await releaseWebGPUExclusiveLock();}
});
