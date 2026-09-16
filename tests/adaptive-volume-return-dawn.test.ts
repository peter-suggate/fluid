import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { createAdaptiveVolumeReturnWGSL, ADAPTIVE_VOLUME_RETURN_PROPAGATION_PAIRS, ADAPTIVE_VOLUME_RETURN_ROUNDS } from "../lib/methods/adaptive-volume/adaptive-volume-return.wgsl";
import { createGeometricVolumeResidentWGSL, type SparseGeometricVolumeLayout } from "../lib/methods/adaptive-volume/resident-volume.wgsl";
import { createLevelSetThinFeaturesWGSL } from "../lib/methods/adaptive-volume/levelset-volume-thin-features.wgsl";
const dawnTest = process.env.WEBGPU_NODE_MODULE ? test : test.skip;

dawnTest("adaptive GPU return conserves mass across rungs and refuses walls/ambiguous patches", {timeout:60_000}, async()=>{
 await acquireWebGPUExclusiveLock("dawn-test","adaptive 3D distance return");
 let device:GPUDevice|undefined;let gpu:GPU|undefined;
 try {
  const dawn=await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href);Object.assign(globalThis,dawn.globals);
  gpu=createProcessRetainedDawnGPU(dawn,["backend=metal"]);const adapter=await gpu!.requestAdapter();assert.ok(adapter);device=await adapter.requestDevice();
  const production=createGeometricVolumeResidentWGSL({} as SparseGeometricVolumeLayout);
  const budgets=production.slice(production.indexOf("@compute @workgroup_size(64)\nfn gatherWholeFrameVolumeSharpening"),production.indexOf("// Delete only whole dilute pages"));
  const code=`
@group(0) @binding(0) var<storage,read_write> state:array<f32>;
@group(0) @binding(1) var<storage,read_write> conditioning:array<atomic<i32>>;
struct Params{count:u32,wall:u32,two:u32,pad:u32}
@group(0) @binding(2) var<uniform> p:Params;
const INVALID=0xffffffffu;const GV_LOW=352u;const GV_CURRENT=0u;const GV_PLUS=32u;const GV_MINUS=64u;
const GV_EDGE_A=96u;const GV_EDGE_B=128u;const GV_EDGE_META=160u;
const GV_FLUX=256u;const GV_EDGE_CAPACITY=32u;const GV_WHOLE_FRAME_CONTROL=0u;
fn gvFailed()->bool{return atomicLoad(&conditioning[63])!=0;}
fn gvFault(a:u32,b:u32,c:f32,d:f32,e:f32){_=a;_=b;_=c;_=d;_=e;atomicStore(&conditioning[63],1);}
fn surfaceSharpeningEnabled()->bool{return true;}
fn surfaceSharpeningStrength()->f32{return 1.0;}
fn gvRoundoff(c:f32)->f32{return 9.5367431640625e-7*max(1.0,c);}
fn acceptedTemplateCellInvocation(i:u32)->u32{return select(INVALID,i,i<p.count);}
fn cnxCellOrdinalUnchecked(i:u32)->u32{return select(INVALID,i,i<p.count);}
fn acceptedTemplateRowInvocation(i:u32)->u32{return select(INVALID,i,i+1u<p.count);}
fn gvAcceptedPhysicalRow(i:u32)->bool{return i+1u<p.count;}
fn cnxPhysicalFaceRangeUnchecked(i:u32)->vec2u{return vec2u(i,i+1u);}
fn gvCells(i:u32)->vec2u{return vec2u(i,i+1u);}
fn gvRow(i:u32)->u32{return i;}
fn gvArea(i:u32)->f32{_=i;return 1.0;}
fn rowOpenFraction(i:u32)->f32{return select(1.0,0.0,i==p.wall);}
fn gvOtherCell(f:u32,negative:bool)->u32{return select(f,f+1u,negative);}
fn gvCellFaceRange(c:u32)->vec2u{return vec2u(2u*c,2u*c+2u);}
fn gvCellFace(at:u32)->u32{
 let c=at/2u;if(at%2u==0u){return select(0u,2u*(c-1u),c>0u);}
 return select(2u*(p.count-2u),2u*c+1u,c+1u<p.count);
}
fn cellCenter(c:u32)->vec3f{return vec3f(state[384u+c],0.5,0.5);}
fn cellWidths(c:u32)->vec3f{return vec3f(state[416u+c],1.0,1.0);}
fn cellMinimumWidth(c:u32)->f32{_=c;return 1.0;}
fn cellVolume(c:u32)->f32{return state[416u+c];}
fn gvReceiverCapacity(c:u32)->f32{return cellVolume(c);}
fn gvPhiTargetVolume(c:u32)->vec2f{return vec2f(select(0.0,1.0,c==0u||(p.two!=0u&&c+1u==p.count)),1.0);}
struct PhiSample{phi:f32,valid:bool,metric:bool}
fn lsvSampleAt(q:vec3f)->PhiSample{
 let phi=select(q.x-1.0,min(q.x-1.0,state[384u+p.count-1u]-0.5-q.x),p.two!=0u);
 return PhiSample(phi,true,abs(phi)<=4.0);
}
fn destinationDensity()->u32{return 448u;}
fn destinationGamma()->u32{return 480u;}
fn incrementalActivityMarkCellClosure(c:u32){_=c;}
`+createAdaptiveVolumeReturnWGSL()+budgets;
  const module=device.createShaderModule({code});const info=await module.getCompilationInfo();
  assert.deepEqual(info.messages.filter(m=>m.type==="error").map(m=>`${m.lineNum}: ${m.message}`),[]);
  const layout=device.createBindGroupLayout({entries:[0,1,2].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:binding===2?"uniform" as const:"storage" as const}}))});
  const pl=device.createPipelineLayout({bindGroupLayouts:[layout]});
  const names=["beginAdaptiveVolumeReturn","seedAdaptiveVolumeReturn","relaxAdaptiveVolumeReturnA","relaxAdaptiveVolumeReturnB","prepareAdaptiveVolumeReturn","proposeAdaptiveVolumeReturn","gatherWholeFrameVolumeSharpening","commitWholeFrameVolumeSharpening"];
  const pipelines=new Map(names.map(entryPoint=>[entryPoint,device!.createComputePipeline({layout:pl,compute:{module,entryPoint}})]));
  const fixtures: {name:string,n:number,donor:number,mixed?:boolean,wall?:number,two?:boolean,sweeps?:number,passes?:number,unchanged?:boolean}[] = [{name:"far",n:8,donor:6},{name:"mixed widths",n:8,donor:6,mixed:true},{name:"wall",n:8,donor:6,wall:3},{name:"two patches",n:9,donor:4,two:true},{name:"near only",n:8,donor:2},
    {name:"one return pass",n:8,donor:6,passes:1},
    {name:"insufficient sweeps",n:8,donor:6,sweeps:1,unchanged:true},
    {name:"zero return passes",n:8,donor:6,passes:0,unchanged:true}];
  for(const fixture of fixtures){
   const values=new Float32Array(512);let x=0;
   for(let c=0;c<fixture.n;c++){const width=fixture.mixed&&c<2?2:1;values[384+c]=x+width/2;values[416+c]=width;x+=width;}
   values[fixture.donor]=0.75;
   const state=device.createBuffer({size:values.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC});
   const control=device.createBuffer({size:256,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
   const params=device.createBuffer({size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
   const read: GPUBuffer=device.createBuffer({size:values.byteLength+256,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
   device.queue.writeBuffer(state,0,values);device.queue.writeBuffer(params,0,new Uint32Array([fixture.n,fixture.wall??0xffffffff,fixture.two?1:0,0]));
   const group=device.createBindGroup({layout,entries:[state,control,params].map((buffer,binding)=>({binding,resource:{buffer}}))});
   const encoder=device.createCommandEncoder();const run=(name:string)=>{const pass=encoder.beginComputePass();pass.setPipeline(pipelines.get(name)!);pass.setBindGroup(0,group);pass.dispatchWorkgroups(1);pass.end();};
   run(names[0]!);run(names[1]!);for(let i=0;i<(fixture.sweeps??ADAPTIVE_VOLUME_RETURN_PROPAGATION_PAIRS);i++){run(names[2]!);run(names[3]!);}
   for(let i=0;i<(fixture.passes??ADAPTIVE_VOLUME_RETURN_ROUNDS);i++)for(const name of names.slice(4))run(name);
   encoder.copyBufferToBuffer(state,0,read,0,values.byteLength);encoder.copyBufferToBuffer(control,0,read,values.byteLength,256);device.queue.submit([encoder.finish()]);
   await read.mapAsync(GPUMapMode.READ);const mapped=read.getMappedRange();const out=new Float32Array(mapped).slice(0,fixture.n);const receipt=new Uint32Array(mapped).slice(512);
   assert.equal(receipt[63],0,fixture.name);assert.ok(Math.abs(out.reduce((a,b)=>a+b,0)-0.75)<1e-6,fixture.name);assert.ok(out.every((v,i)=>v>=0&&v<=values[416+i]!),fixture.name);
   if(fixture.wall!==undefined||fixture.two||fixture.name==="near only"||fixture.unchanged)assert.deepEqual(out,values.slice(0,fixture.n),fixture.name);
   else {assert.ok(out[fixture.donor]!<0.01,fixture.name);assert.ok(out.slice(0,fixture.donor).reduce((a,b)=>a+b,0)>0.74,fixture.name);assert.ok(receipt[29]!>0);if(fixture.passes===1)assert.equal(out[5],0.75);}
   read.unmap();for(const b of [state,control,params,read])b.destroy();
  }
 }finally{device?.destroy();gpu=undefined;await releaseWebGPUExclusiveLock();}
});

dawnTest("adaptive phi normals detect oblique sheets, air gaps and off-centre drops",{timeout:60_000},async()=>{
 await acquireWebGPUExclusiveLock("dawn-test","3D phi thin-feature detector");let device:GPUDevice|undefined;let gpu:GPU|undefined;
 try{
  const dawn=await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href);Object.assign(globalThis,dawn.globals);gpu=createProcessRetainedDawnGPU(dawn,["backend=metal"]);const adapter=await gpu!.requestAdapter();assert.ok(adapter);device=await adapter.requestDevice();
  const code=`
@group(0) @binding(0) var<storage,read_write> result:array<u32>;
@group(0) @binding(1) var<uniform> mode:vec4u;
struct LsvPhiSample{phi:f32,valid:bool,metric:bool,support:u32}
struct LsvCellStencil{lower:vec3f,widths:vec3f,phi:array<f32,8>,support:array<u32,8>,resolved:bool}
fn raw(p:vec3f)->f32{
 if(mode.x==0u){return p.x-0.5;}
 if(mode.x==1u){return abs((p.x+p.y-1.0)*0.70710678)-0.55;}
 if(mode.x==2u){return 0.55-abs((p.x+p.y-1.0)*0.70710678);}
 if(mode.x==3u){return length(p-vec3f(1.0))-0.4;}
 if(mode.x==4u){return abs(p.x-0.5)-3.0;}
 if(mode.x==6u){return length(vec2f(p.x+13.99,p.y))-14.0;}
 return p.x-0.5;
}
fn stencilAt(lo:vec3f)->LsvCellStencil{
 var s:LsvCellStencil;s.lower=lo;s.widths=vec3f(1.0);s.resolved=true;
 for(var c=0u;c<8u;c++){s.phi[c]=raw(lo+vec3f(f32(c&1u),f32((c>>1u)&1u),f32((c>>2u)&1u)));s.support[c]=select(3u,0u,mode.x==7u);}return s;
}
fn lsvStencilContains(s:LsvCellStencil,p:vec3f)->bool{return all(p>=s.lower)&&all(p<=s.lower+s.widths);}
fn lsvStencilSampleAt(s:LsvCellStencil,p:vec3f)->LsvPhiSample{
 let t=clamp(p-s.lower,vec3f(0.0),vec3f(1.0));var v=0.0;
 for(var c=0u;c<8u;c++){let high=vec3u(c&1u,(c>>1u)&1u,(c>>2u)&1u)!=vec3u(0u);let w=select(vec3f(1.0)-t,t,high);v+=s.phi[c]*w.x*w.y*w.z;}
 return LsvPhiSample(v,mode.x!=5u||p.x<=1.0,true,3u);
}
fn lsvSampleAt(p:vec3f)->LsvPhiSample{return lsvStencilSampleAt(stencilAt(floor(p)),p);}
fn lsvCellStencil(c:u32)->LsvCellStencil{_=c;return stencilAt(vec3f(0.0));}
fn lsvStencilAtPosition(p:vec3f)->LsvCellStencil{return stencilAt(floor(p));}
`+createLevelSetThinFeaturesWGSL()+`
@compute @workgroup_size(1) fn main(){result[0]=lsvThinFeatureCell(0u,2.0);}
`;
  const module=device.createShaderModule({code});const info=await module.getCompilationInfo();assert.deepEqual(info.messages.filter(m=>m.type==="error").map(m=>m.message),[]);
  const pipeline=device.createComputePipeline({layout:"auto",compute:{module,entryPoint:"main"}});
  const out=device.createBuffer({size:4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});const params=device.createBuffer({size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});const read: GPUBuffer=device.createBuffer({size:4,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
  const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[out,params].map((buffer,binding)=>({binding,resource:{buffer}}))});
  for(const [mode,expected] of [[0,0],[1,1],[2,1],[3,1],[4,0],[5,0],[6,0],[7,2]]){
   device.queue.writeBuffer(params,0,new Uint32Array([mode!,0,0,0]));const encoder=device.createCommandEncoder();const pass=encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(1);pass.end();encoder.copyBufferToBuffer(out,0,read,0,4);device.queue.submit([encoder.finish()]);await read.mapAsync(GPUMapMode.READ);assert.equal(new Uint32Array(read.getMappedRange())[0],expected,`mode ${mode}`);read.unmap();
  }
  for(const b of [out,params,read])b.destroy();
 }finally{device?.destroy();gpu=undefined;await releaseWebGPUExclusiveLock();}
});
