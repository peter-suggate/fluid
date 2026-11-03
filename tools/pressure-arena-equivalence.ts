/** Pressure arena equivalence check: run a scene to a frame, then run a
 * rewritten arena kernel and its reference form (UniformPressureArena.reference)
 * over the same band state and compare their outputs bit for bit.
 *   node --import tsx tools/pressure-arena-equivalence.ts --scene=paddleboats --bodies --frames=30,60,90
 * Run it through tools/gpu/fluid-gpu-run.sh check. Exits 1 on any difference. */
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
import { UniformPressureArena } from "../lib/methods/uniform/uniform-pressure-arena";
import { uniformDetailField } from "../lib/methods/uniform/uniform-detail-fields";
import { mooredMotorBodies } from "./moored-motor-bodies";

const option=(key:string)=>process.argv.find(a=>a.startsWith(`--${key}=`))?.slice(key.length+3);
const frames=(option("frames")??"30,60,90").split(",").map(Number);
paceDawnEventPolling();
const dawn=await import(pathToFileURL(resolve("node_modules/webgpu/index.js")).href);Object.assign(globalThis,dawn.globals);
const gpu=createProcessRetainedDawnGPU(dawn,["backend=metal"]),adapter=await gpu.requestAdapter();assert.ok(adapter);
const raw=await adapter.requestDevice({requiredFeatures:[...(adapter.features.has("subgroups")?["subgroups" as const]:[])],requiredLimits:requiredFluidDeviceLimits(adapter.limits)});
const device=managedGPUDevice(raw,{requireWorkerRealm:false});
UniformPressureArena.reference=true;
const definition=getSceneDefinition(option("scene")??"paddleboats"),quality=definition.methodProfile!.quality;
const scene=structuredClone(sceneDocument(definition));
const values=resolveMethodValues(method,quality,{...method.appDefaults,...definition.methodProfile!.overrides});
const dt=method.effectiveStep_s?.(scene,values)??scene.numerics.fixedDt_s;
const solver=await method.createSolverAsync!(device,scene,quality,values,undefined,()=>{}) as any;
const bodies=process.argv.includes("--bodies")?mooredMotorBodies(scene,dt):undefined;
const band=solver.mixedFrame.band as UniformPressureArena;
const handle=(band as unknown as {fields:{velocity:GPUTexture}}).fields.velocity;
// The output is a detail field: its current h texture and its base block (a 4h store writes both).
const textures=():GPUTexture[]=>{const f=uniformDetailField(handle);if(!f)return [handle];const base=f.storage.baseOf(handle);return [f.storage.physical(handle),...(base?[base]:[])];};
const readModule=raw.createShaderModule({code:`@group(0) @binding(0) var source:texture_3d<f32>;
@group(0) @binding(1) var<storage,read_write> out:array<vec4u>;
override W:u32;override H:u32;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g:vec3u){
 if(g.x>=W*H){return;}let i=g.x+W*H*g.y;
 out[i]=bitcast<vec4u>(textureLoad(source,vec3u(i%W,(i/W)%H,g.y),0));
}`});
// A finite payload (about -3.4e38) no kernel writes: every texel to it before each run, so a
// run's output is exactly its written texels.
const POISON=0xff7fdead;
const poisonModule=raw.createShaderModule({code:`@group(0) @binding(0) var dst:texture_storage_3d<rgba32float,write>;
override W:u32;override H:u32;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g:vec3u){
 if(g.x>=W*H){return;}textureStore(dst,vec3u(g.x%W,g.x/W,g.y),bitcast<vec4f>(vec4u(${POISON}u)));
}`});
const poisoned=(e:GPUCommandEncoder):void=>{
 for(const texture of textures()){
  const [W,H,D]=[texture.width,texture.height,texture.depthOrArrayLayers];
  const pipeline=raw.createComputePipeline({layout:"auto",compute:{module:poisonModule,entryPoint:"main",constants:{W,H}}});
  const pass=e.beginComputePass();pass.setPipeline(pipeline);
  pass.setBindGroup(0,raw.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:texture.createView()}]}));
  pass.dispatchWorkgroups(Math.ceil(W*H/64),D);pass.end();
 }
};
// The texels as they were before the check, written back after it (a run's
// later frames read them).
const restoreModule=raw.createShaderModule({code:`@group(0) @binding(0) var dst:texture_storage_3d<rgba32float,write>;
@group(0) @binding(1) var<storage,read> saved:array<vec4u>;
override W:u32;override H:u32;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g:vec3u){
 if(g.x>=W*H){return;}textureStore(dst,vec3u(g.x%W,g.x/W,g.y),bitcast<vec4f>(saved[g.x+W*H*g.y]));
}`});
const restored=(saved:Uint32Array[]):void=>{
 const e=raw.createCommandEncoder();
 textures().forEach((texture,t)=>{
  const [W,H,D]=[texture.width,texture.height,texture.depthOrArrayLayers];
  const buffer=raw.createBuffer({size:saved[t]!.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});raw.queue.writeBuffer(buffer,0,saved[t]! as Uint32Array<ArrayBuffer>);
  const pipeline=raw.createComputePipeline({layout:"auto",compute:{module:restoreModule,entryPoint:"main",constants:{W,H}}});
  const pass=e.beginComputePass();pass.setPipeline(pipeline);
  pass.setBindGroup(0,raw.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:texture.createView()},{binding:1,resource:{buffer}}]}));
  pass.dispatchWorkgroups(Math.ceil(W*H/64),D);pass.end();
 });
 raw.queue.submit([e.finish()]);
};
const snapshot=async(run:(e:GPUCommandEncoder)=>void):Promise<Uint32Array[]>=>{
 raw.pushErrorScope("validation");
 const e=raw.createCommandEncoder();run(e);const reads:{staging:GPUBuffer;out:GPUBuffer}[]=[];
 for(const texture of textures()){
  const [W,H,D]=[texture.width,texture.height,texture.depthOrArrayLayers],size=16*W*H*D;
  const pipeline=raw.createComputePipeline({layout:"auto",compute:{module:readModule,entryPoint:"main",constants:{W,H}}});
  const out=raw.createBuffer({size,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
  const staging=raw.createBuffer({size,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
  const pass=e.beginComputePass();pass.setPipeline(pipeline);
  pass.setBindGroup(0,raw.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:texture.createView()},{binding:1,resource:{buffer:out}}]}));
  pass.dispatchWorkgroups(Math.ceil(W*H/64),D);pass.end();
  e.copyBufferToBuffer(out,0,staging,0,size);reads.push({staging,out});
 }
 raw.queue.submit([e.finish()]);
 const error=await raw.popErrorScope();if(error)throw new Error(`equivalence snapshot: ${error.message}`);
 return Promise.all(reads.map(async({staging,out})=>{await staging.mapAsync(GPUMapMode.READ);const words=new Uint32Array(staging.getMappedRange().slice(0));staging.destroy();out.destroy();return words;}));
};
let failed=false;
for(let frame=1;frame<=Math.max(...frames);frame++){
 assert.ok(solver.advanceTo(frame*dt,bodies?.(frame*dt)));await solver.awaitFrameCompletion();
 if(!frames.includes(frame))continue;
 const before=await snapshot(()=>{});
 const reference=await snapshot(e=>{poisoned(e);band.encodeProjectEntry(e,"pProjectNativeRef");});
 const rewrite=await snapshot(e=>{poisoned(e);band.encodeProjectEntry(e,"pProjectNative");});
 let written=0,differ=0;
 reference.forEach((words,t)=>{for(let i=0;i<words.length;i++){if(words[i]!==POISON)written++;if(words[i]!==rewrite[t]![i])differ++;}});
 console.log(JSON.stringify({frame,entry:"pProjectNative",wordsWritten:written,wordsDiffering:differ}));
 if(!written)failed=true;
 restored(before);
 if(differ)failed=true;
}
solver.destroy();await raw.queue.onSubmittedWorkDone();raw.destroy();
if(failed){console.error("pressure arena equivalence: outputs differ");process.exit(1);}
