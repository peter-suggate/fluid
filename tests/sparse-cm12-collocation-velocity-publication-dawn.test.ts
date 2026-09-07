import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";

const source = readFileSync(new URL("../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.wgsl.ts", import.meta.url), "utf8");
const production = (name: string) => {
  const result = source.match(new RegExp(`fn ${name}\\([\\s\\S]*?\\n}`))?.[0];
  assert.ok(result); return result;
};
const dawnTest = process.env.WEBGPU_NODE_MODULE ? test : test.skip;
dawnTest("collocation diagnostics cannot overwrite the effective velocity plane", async () => {
  await acquireWebGPUExclusiveLock("dawn-test", "collocation-velocity-publication");
  let gpu: GPU | undefined, device: GPUDevice | undefined;
  try {
    const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href);
    Object.assign(globalThis, dawn.globals);
    gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
    const adapter = await gpu!.requestAdapter(); assert.ok(adapter);
    device = await adapter.requestDevice();
    const shader = device.createShaderModule({ code: `
const INVALID=0xffffffffu;const CM12_LIQUID_ISOVALUE=0.5;
struct Params{stateOffsets4:vec4u,stateOffsets5:vec4u,frame:vec4f}
const p=Params(vec4u(0,2600,0,0),vec4u(2304,0,0,0),vec4f(.0333333,.05,0,0));
@group(0)@binding(0)var<storage,read_write>state:array<f32>;
@group(0)@binding(1)var<storage,read_write>partials:array<vec4f>;
@group(0)@binding(2)var<storage,read_write>scalars:array<f32>;
var<workgroup>reduceA:array<f32,64>;var<workgroup>reduceB:array<f32,64>;
fn acceptedTemplateCellInvocation(i:u32)->u32{return select(INVALID,i,i<128u);}
fn acceptedTemplateCellWorkgroups()->u32{return 2u;}
fn cellTransportActive(c:u32)->bool{_=c;return true;}
fn cellActive(c:u32)->bool{_=c;return true;}
fn destinationCellVelocity()->u32{return 0u;}
fn destinationFaceVelocity()->u32{return 1024u;}
fn destinationDensity()->u32{return 2048u;}
fn incrementalActivityMarkCellClosure(c:u32){_=c;}
fn incidenceBegin(c:u32)->u32{return c;}
fn incidenceEnd(c:u32)->u32{return c+1u;}
fn incidenceRow(i:u32)->u32{return i;}
fn incidenceTerm(i:u32)->u32{return i;}
fn rowAccepted(r:u32)->bool{_=r;return true;}
fn rowKind(r:u32)->u32{_=r;return 0u;}
fn rowAxis(r:u32)->u32{_=r;return 0u;}
fn rowDualWeight(r:u32)->f32{_=r;return 1.0;}
fn rowStaticDualWeight(r:u32)->f32{_=r;return 1.0;}
fn hasSolidBoundaries()->bool{return false;}
fn termCoefficient(t:u32)->f32{_=t;return 1.0;}
fn rowSeparatingFromClosedWorld(r:u32)->bool{_=r;return false;}
fn rowOpenFraction(r:u32)->f32{_=r;return 1.0;}
fn rowSolidVelocity(r:u32)->f32{_=r;return 0.0;}
fn pcmCellContains(c:u32)->bool{_=c;return true;}
fn rawPressureDensity(c:u32)->f32{_=c;return 1.0;}
fn cm12VolumeCorrectionDivergence(r:f32,h:f32,dt:f32)->f32{_=r;_=h;_=dt;return 0.0;}
fn cellMinimumWidth(c:u32)->f32{_=c;return 1.0;}
fn cellOpenVolume(c:u32)->f32{_=c;return 1.0;}
fn cm12PublishCollocatedWetEffectiveVelocity(c:u32,v:vec3f,wet:bool){if(wet){partials[c]=vec4f(v,1);}}
@compute @workgroup_size(64)
${production("collocateAndDiagnose")}
@compute @workgroup_size(64)
${production("reduceDivergenceDiagnostics")}
` });
    assert.deepEqual((await shader.getCompilationInfo()).messages.filter(m => m.type === "error"), []);
    const layout = device.createBindGroupLayout({ entries: [0,1,2].map(binding => ({binding,
      visibility: GPUShaderStage.COMPUTE, buffer: {type: "storage" as const}})) });
    const pipelineLayout = device.createPipelineLayout({bindGroupLayouts:[layout]});
    const pipelines = ["collocateAndDiagnose","reduceDivergenceDiagnostics"].map(entryPoint =>
      device!.createComputePipeline({layout:pipelineLayout,compute:{module:shader,entryPoint}}));
    const buffers = [4096*4,128*16,8*4].map(size => device!.createBuffer({size,
      usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC}));
    const initial = new Float32Array(4096);
    for(let i=0;i<128;i++){initial[1024+i]=i+1;initial[2048+i]=1;}
    device.queue.writeBuffer(buffers[0]!,0,initial);
    const group=device.createBindGroup({layout,entries:buffers.map((buffer,binding)=>({binding,resource:{buffer}}))});
    const readback=device.createBuffer({size:128*16+8*4,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
    const encoder=device.createCommandEncoder();
    for(let i=0;i<2;i++){const pass=encoder.beginComputePass();pass.setPipeline(pipelines[i]!);
      pass.setBindGroup(0,group);pass.dispatchWorkgroups(i===0?2:1);pass.end();}
    encoder.copyBufferToBuffer(buffers[1]!,0,readback,0,128*16);
    encoder.copyBufferToBuffer(buffers[2]!,0,readback,128*16,8*4);
    device.queue.submit([encoder.finish()]);await readback.mapAsync(GPUMapMode.READ);
    const values=new Float32Array(readback.getMappedRange());
    for(let i=0;i<128;i++)assert.deepEqual([...values.slice(4*i,4*i+4)],[i+1,0,0,1],`cell ${i}`);
    assert.equal(values[128*4+6],128);assert.equal(values[128*4+7],0);
    readback.unmap();readback.destroy();buffers.forEach(buffer=>buffer.destroy());
  } finally {device?.destroy();await releaseWebGPUExclusiveLock();assert.ok(gpu);}
});
