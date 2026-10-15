/** Compare packed pressure face payloads with the previous per-cell layout
 * assembled on identical inputs. Diagnostic allocations/timing are not frame benchmarks. */
import assert from "node:assert/strict";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { resolveMethodValues } from "../lib/core/method-contract";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { managedGPUDevice } from "../lib/core/gpu-compilation-manager";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { createProcessRetainedDawnGPU, paceDawnEventPolling } from "../lib/harness/node-dawn-provider";
import { uniformNarrowBandMethod as method } from "../lib/methods/uniform/uniform-narrow-band-method";
import type { WebGPUUniformReferenceSolver } from "../lib/methods/uniform/webgpu-uniform-reference";
const option=(key:string)=>process.argv.find(a=>a.startsWith(`--${key}=`))?.slice(key.length+3);
const frames=(option("frames")??"120,600").split(",").map(Number).sort((a,b)=>a-b);
assert.ok(frames.every(n=>Number.isInteger(n)&&n>0));
paceDawnEventPolling();
const dawn=await import(pathToFileURL(resolve("node_modules/webgpu/index.js")).href);Object.assign(globalThis,dawn.globals);
const gpu=createProcessRetainedDawnGPU(dawn,["backend=metal"]),adapter=await gpu.requestAdapter();assert.ok(adapter);
const raw=await adapter.requestDevice({requiredFeatures:adapter.features.has("subgroups")?["subgroups"]:[],requiredLimits:requiredFluidDeviceLimits(adapter.limits)});
const errors:string[]=[];raw.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);});
const sources=new WeakMap<GPUShaderModule,string>(),references=new WeakMap<GPUComputePipeline,GPUComputePipeline>();
const descriptions=new WeakMap<GPUBindGroup,GPUBindGroupDescriptor>();
let frame=0;const captured=new Set<number>(),pending:{read:GPUBuffer;owned:GPUBuffer[]}[]=[];
const compareModule=raw.createShaderModule({code:`
@group(0) @binding(0) var<storage,read> packed:array<f32>;
@group(0) @binding(1) var<storage,read> original:array<f32>;
@group(0) @binding(2) var<storage,read> index:array<u32>;
@group(0) @binding(3) var<storage,read_write> result:array<atomic<u32>>;
override capacity:u32;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3u){
 let c=id.x+65535u*64u*id.y;if(c>=64u*min(index[0],capacity)){return;}let N=64u*capacity;
 let r=c%64u;let y=(r/2u)%4u;let z=(r/8u)%4u;let x=2u*(r&1u)+((r/32u+y+z)&1u);let l=vec3u(x,y,z);
 for(var k=0u;k<9u;k++){atomicAdd(&result[0],1u);if(bitcast<u32>(packed[k*N+c])!=bitcast<u32>(original[k*N+c])){atomicAdd(&result[1],1u);}}
 for(var f=0u;f<6u;f++){
  let a=f/2u;let face=240u*(c/64u)+80u*a+l[a]+(f&1u)+5u*(l[(a+1u)%3u]+4u*l[(a+2u)%3u]);
  atomicAdd(&result[0],2u);
  if(bitcast<u32>(packed[10u*N+face])!=bitcast<u32>(original[(9u+f)*N+c])){atomicAdd(&result[2],1u);}
  if(bitcast<u32>(packed[10u*N+240u*capacity+face])!=bitcast<u32>(original[(15u+f)*N+c])){atomicAdd(&result[3],1u);}
 }
}`});
const intercepted=new Proxy(raw,{get(target,key){
 if(key==="createShaderModule")return(d:GPUShaderModuleDescriptor)=>{const m=target.createShaderModule(d);sources.set(m,d.code);return m;};
 if(key==="createBindGroup")return(d:GPUBindGroupDescriptor)=>{const g=target.createBindGroup(d);descriptions.set(g,{...d,entries:[...d.entries]});return g;};
 if(key==="createComputePipelineAsync")return async(d:GPUComputePipelineDescriptor)=>{
  const p=await target.createComputePipelineAsync(d);
  if(d.compute.entryPoint==="bandPrep"){
   let code=sources.get(d.compute.module)!;
   const volume="if(sign>0||l[axis]==0u){rows[10u*N+240u*bCap()+bPackedFace(s,l,f)]=volume;}",forced="if(sign>0||l[axis]==0u){rows[10u*N+bPackedFace(s,l,f)]=forced;}";
   assert.ok(code.includes(volume)&&code.includes(forced),"solid packed row assembly");
   code=code.replace(volume,"rows[(15u+f)*N+cell]=volume;").replace(forced,"rows[(9u+f)*N+cell]=forced;");
   const m=target.createShaderModule({code});references.set(p,await target.createComputePipelineAsync({...d,compute:{...d.compute,module:m}}));
  }return p;
 };
 if(key==="createCommandEncoder")return(d?:GPUCommandEncoderDescriptor)=>{
  const encoder=target.createCommandEncoder(d);
  return new Proxy(encoder,{get(e,k){
   if(k==="beginComputePass")return(desc?:GPUComputePassDescriptor)=>{
    let pass=e.beginComputePass(desc),pipeline:GPUComputePipeline|undefined;const groups=new Map<number,unknown[]>();
    const bind=()=>{pass.setPipeline(pipeline!);for(const [i,args] of groups)Reflect.apply(pass.setBindGroup,pass,[i,...args]);};
    return new Proxy(pass,{get(_p,k){
     if(k==="setPipeline")return(p:GPUComputePipeline)=>{pipeline=p;pass.setPipeline(p);};
     if(k==="setBindGroup")return(i:number,...args:unknown[])=>{groups.set(i,args);Reflect.apply(pass.setBindGroup,pass,[i,...args]);};
     if(k==="dispatchWorkgroups")return(x:number,y=1,z=1)=>{
      pass.dispatchWorkgroups(x,y,z);const reference=pipeline&&references.get(pipeline);
      if(!reference||!frames.includes(frame)||captured.has(frame))return;
      captured.add(frame);pass.end();
      const d=descriptions.get(groups.get(1)![0] as GPUBindGroup)!,entries=[...d.entries];
      const rows=entries.find(e=>e.binding===9)!.resource as GPUBufferBinding;
      const index=entries.find(e=>e.binding===2)!.resource as GPUBufferBinding;
      const capacity=rows.size!/(4*1120);assert.ok(Number.isInteger(capacity));
      const old=target.createBuffer({size:4*1408*capacity,usage:GPUBufferUsage.STORAGE});
      const result=target.createBuffer({size:16,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
      const read=target.createBuffer({size:16,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
      pass=e.beginComputePass();bind();pass.setPipeline(reference);
      pass.setBindGroup(1,target.createBindGroup({...d,entries:entries.map(e=>e.binding===9?{binding:9,resource:{buffer:old}}:e)}));pass.dispatchWorkgroups(x,y,z);pass.end();
      const cp=target.createComputePipeline({layout:"auto",compute:{module:compareModule,entryPoint:"main",constants:{capacity}}});
      pass=e.beginComputePass();pass.setPipeline(cp);pass.setBindGroup(0,target.createBindGroup({layout:cp.getBindGroupLayout(0),entries:[{binding:0,resource:rows},{binding:1,resource:{buffer:old}},{binding:2,resource:index},{binding:3,resource:{buffer:result}}]}));
      // Pool-sized launch, one workgroup a tile, spilling into y above 65535.
      pass.dispatchWorkgroups(Math.min(capacity,65535),Math.ceil(capacity/65535));pass.end();e.copyBufferToBuffer(result,0,read,0,16);
      pending.push({read,owned:[old,result]});pass=e.beginComputePass();bind();
     };
     const v=Reflect.get(pass,k,pass);return typeof v==="function"?v.bind(pass):v;
    }});
   };
   const v=Reflect.get(e,k,e);return typeof v==="function"?v.bind(e):v;
  }});
 };
 const v=Reflect.get(target,key,target);return typeof v==="function"?v.bind(target):v;
}});
let solver:WebGPUUniformReferenceSolver|undefined;
try{
 const definition=getSceneDefinition(option("scene")??"nbflip-water-slide-long"),quality=definition.methodProfile!.quality,scene=sceneDocument(definition);
 const values=resolveMethodValues(method,quality,{...method.appDefaults,...definition.methodProfile!.overrides});const dt=method.effectiveStep_s?.(scene,values)??scene.numerics.fixedDt_s;
 solver=await method.createSolverAsync!(managedGPUDevice(intercepted,{requireWorkerRealm:false}),scene,quality,values,undefined,()=>{}) as WebGPUUniformReferenceSolver;
 for(frame=1;frame<=Math.max(...frames);frame++){
  assert.ok(solver.advanceTo(frame*dt,[]));await solver.awaitFrameCompletion();assert.equal(solver.info.simulationPipelineError,undefined);
  for(const p of pending.splice(0)){await p.read.mapAsync(GPUMapMode.READ);const counts=[...new Uint32Array(p.read.getMappedRange())];console.log(JSON.stringify({frame,comparisons:counts[0],cellMismatches:counts[1],velocityMismatches:counts[2],volumeMismatches:counts[3]}));assert.ok(counts[0]!>0);assert.deepEqual(counts.slice(1),[0,0,0]);p.read.unmap();p.read.destroy();for(const b of p.owned)b.destroy();}
 }
 assert.deepEqual([...captured],frames);assert.deepEqual(errors,[]);
}catch(error){console.error(errors);throw error;}finally{solver?.destroy();await raw.queue.onSubmittedWorkDone();raw.destroy();}
