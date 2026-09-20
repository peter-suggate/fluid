import { UNIFORM_VOLUME_PHASE } from "../lib/methods/uniform/uniform-volume-stages";
import assert from "node:assert/strict";
import { pathToFileURL } from "node:url";
import { resolve } from "node:path";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import { managedGPUDevice } from "../lib/core/gpu-compilation-manager";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { uniformVolumeMethod } from "../lib/methods/uniform/uniform-volume-method";
import { uniformReferenceSolverOptions } from "../lib/methods/uniform/method";
import { resolveMethodValues } from "../lib/core/method-contract";
import { WebGPUUniformReferenceSolver } from "../lib/methods/uniform/webgpu-uniform-reference";

const median = (xs: number[]) => {const sorted=[...xs].sort((a,b)=>a-b);return (sorted[Math.floor((sorted.length-1)/2)]!+sorted[Math.floor(sorted.length/2)]!)/2;};
const quick = process.argv.includes("--quick");
const frames = quick ? 6 : 30;
const stages = process.argv.includes("--stages");
const sceneIds = process.argv.includes("--mini-only") ? ["minimal-power-dam-break-64"]
  : ["minimal-power-dam-break-64", "large-power-dam-break"];
async function read(device: GPUDevice, texture: GPUTexture) {
  const components = texture.format === "rgba32float" ? 4 : 1;
  const width = texture.width * components;
  const row = Math.ceil(width * 4 / 256) * 256;
  const buffer = device.createBuffer({size:row*texture.height*texture.depthOrArrayLayers,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  try {
    const e = device.createCommandEncoder();
    e.copyTextureToBuffer({texture},{buffer,bytesPerRow:row,rowsPerImage:texture.height},[texture.width,texture.height,texture.depthOrArrayLayers]);
    device.queue.submit([e.finish()]);await buffer.mapAsync(GPUMapMode.READ);
    const values = new Float32Array(buffer.getMappedRange());
    const result = new Float32Array(width*texture.height*texture.depthOrArrayLayers);
    for(let i=0;i<texture.height*texture.depthOrArrayLayers;i++)result.set(values.subarray(i*row/4,i*row/4+width),i*width);
    return result;
  } finally {buffer.unmap();buffer.destroy();}
}
const compare = (a: Float32Array, b: Float32Array) => {
  let maxAbs = 0, sumAbs = 0, sumA = 0, sumB = 0;
  assert.equal(a.length,b.length);
  for(let i=0;i<a.length;i++) {
    assert.ok(Number.isFinite(a[i]!)&&Number.isFinite(b[i]!));
    const error = Math.abs(a[i]!-b[i]!);maxAbs=Math.max(maxAbs,error);sumAbs+=error;sumA+=a[i]!;sumB+=b[i]!;
  }
  return {maxAbs,meanAbs:sumAbs/a.length,sumA,sumB};
};
await acquireWebGPUExclusiveLock("dawn-probe","uniform geometric 4h work experiment");
let device:GPUDevice|undefined;
try {
  const dawn=await import(pathToFileURL(resolve("node_modules/webgpu/index.js")).href);Object.assign(globalThis,dawn.globals);
  const gpu=createProcessRetainedDawnGPU(dawn,["backend=metal"]);const adapter=await gpu.requestAdapter();assert.ok(adapter);
  device=managedGPUDevice(await adapter.requestDevice({requiredFeatures:["timestamp-query"],requiredLimits:requiredFluidDeviceLimits(adapter.limits)}),{requireWorkerRealm:false});
  const errors:string[]=[];device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);console.error(e.error.message);});
  for(const sceneId of sceneIds) {
    let reference:Float32Array[]|undefined;
    const runs: {tileWork:boolean,wall:number,sharpen:number}[]=[];
    for(const tileWork of stages ? [false] : quick ? [false,true] : [false,true,true,false]) {
      const scene=structuredClone(sceneDocument(getSceneDefinition(sceneId)));
      const values=resolveMethodValues(uniformVolumeMethod,"balanced",{});
      const solver=await WebGPUUniformReferenceSolver.createAsync(device,scene,"balanced",undefined,{
        ...uniformReferenceSolverOptions(values,scene),geometricVolume:true,geometricTileWork:tileWork,
        geometricRedistance:true,activeRegion:false,
        gammaDiffusionIterations:0,densityPostProcessing:false,solidExcessCorrection:false,
      },()=>{});
      const query=device.createQuerySet({type:"timestamp",count:stages?10:2});
      const resolved=device.createBuffer({size:256,usage:GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC});
      const staging=device.createBuffer({size:stages?80:16,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
      const access=solver as unknown as {encodeGeometricVolume(e:GPUCommandEncoder,seam?:unknown):void};
      let stamp:((e:GPUCommandEncoder,index:number)=>void)|undefined;
      let markerBuffer:GPUBuffer|undefined;
      if(stages) {
        markerBuffer=device.createBuffer({size:4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
        const marker=await device.createComputePipelineAsync({layout:"auto",compute:{module:device.createShaderModule({code:"@group(0) @binding(0) var<storage,read_write> count:atomic<u32>; @compute @workgroup_size(1) fn main(){atomicAdd(&count,1u);}"}),entryPoint:"main"}});
        const markerGroup=device.createBindGroup({layout:marker.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:markerBuffer}}]});
        const scratch=(solver as unknown as {conditioningScratch:GPUBuffer}).conditioningScratch;
        stamp=(e:GPUCommandEncoder,index:number)=>{e.copyBufferToBuffer(scratch,0,markerBuffer!,0,4);const p=e.beginComputePass({timestampWrites:{querySet:query,beginningOfPassWriteIndex:index}});p.setPipeline(marker);p.setBindGroup(0,markerGroup);p.dispatchWorkgroups(1);p.end();};
        const extra=solver as unknown as {encodeVelocityExtrapolation(e:GPUCommandEncoder,predicted:boolean,seam?:unknown):void;pressureMultigrid:{encode(e:GPUCommandEncoder,group:GPUBindGroup,seam?:unknown):void}};
        const ext=extra.encodeVelocityExtrapolation.bind(extra),pressure=extra.pressureMultigrid.encode.bind(extra.pressureMultigrid);
        extra.encodeVelocityExtrapolation=(e,predicted,seam)=>{stamp!(e,2);ext(e,predicted,seam);stamp!(e,3);};
        extra.pressureMultigrid.encode=(e,group,seam)=>{stamp!(e,8);pressure(e,group,seam);stamp!(e,9);};
      }
      const original=access.encodeGeometricVolume.bind(access);
      access.encodeGeometricVolume=(encoder,seam)=>{
        if(stages) {
          stamp!(encoder,6);
          original(encoder,(phase:{label:string})=>{
            (seam as ((p:unknown)=>void)|undefined)?.(phase);
            if(phase.label==="Dense vertex phi transport and redistance"){stamp!(encoder,7);stamp!(encoder,4);}
            if(phase.label===UNIFORM_VOLUME_PHASE.gather.label){stamp!(encoder,5);stamp!(encoder,0);}
            if(phase.label==="Dense conservative volume sharpening")stamp!(encoder,1);
          });return;
        }
        let started=false, commits=0;
        const proxy=new Proxy(encoder,{get(target,key){
          if(key==="beginComputePass")return (descriptor:GPUComputePassDescriptor)=>{
            const writes:GPUComputePassTimestampWrites={querySet:query};
            if(!started&&(descriptor?.label==="Classify 4h sharpening work"||descriptor?.label==="uvPrepareSharpen")) {writes.beginningOfPassWriteIndex=0;started=true;}
            if(descriptor?.label==="uvCommitSharpen"&&++commits===8)writes.endOfPassWriteIndex=1;
            return target.beginComputePass(writes.beginningOfPassWriteIndex!==undefined||writes.endOfPassWriteIndex!==undefined ? {...descriptor,timestampWrites:writes} : descriptor);
          };
          const value=Reflect.get(target,key,target);return typeof value==="function"?value.bind(target):value;
        }});
        original(proxy,seam);
      };
      try {
        const samples:{frame:number,wall_ms:number,sharpen_ms:number,extension_ms?:number,transport_ms?:number,phi_ms?:number,pressure_ms?:number}[]=[];
        for(let frame=1;frame<=frames;frame++) {
          const start=performance.now();assert.ok(solver.advanceTo(frame/30));await device.queue.onSubmittedWorkDone();const wall_ms=performance.now()-start;
          const e=device.createCommandEncoder();e.resolveQuerySet(query,0,stages?10:2,resolved,0);e.copyBufferToBuffer(resolved,0,staging,0,stages?80:16);device.queue.submit([e.finish()]);
          await staging.mapAsync(GPUMapMode.READ);const times=new BigUint64Array(staging.getMappedRange());const sharpen_ms=Number(times[1]!-times[0]!)/1e6;
          for(let i=0;i<(stages?10:2);i+=2)assert.ok(times[i]!>0n&&times[i+1]!>=times[i]!,`invalid GPU timestamps ${i}: ${times[i]}, ${times[i+1]}`);
          const extra=stages?{extension_ms:Number(times[3]!-times[2]!)/1e6,transport_ms:Number(times[5]!-times[4]!)/1e6,phi_ms:Number(times[7]!-times[6]!)/1e6,pressure_ms:Number(times[9]!-times[8]!)/1e6}:{};
          staging.unmap();samples.push({frame,wall_ms,sharpen_ms,...extra});
        }
        const fields=[];for(const texture of [solver.volumeTexture,solver.vertexPhiTexture!,solver.velocityTexture])fields.push(await read(device,texture));
        const parity=reference?fields.map((f,i)=>compare(reference![i]!,f)):undefined;reference??=fields;
        // Full-step floating-point donor sums need tolerance, not bitwise equality.
        // Record long-run differences alongside the final dense repeat: donor CAS sums are nondeterministic.
        // Exact scheduling parity is gated separately with identical input in the focused Dawn test.
        const measured=samples.slice(3);const wall=median(measured.map(s=>s.wall_ms)),sharpen=median(measured.map(s=>s.sharpen_ms));
        runs.push({tileWork,wall,sharpen});
        console.log(JSON.stringify({sceneId,tileWork,frames,stageMedians:stages?Object.fromEntries(["extension_ms","transport_ms","phi_ms","pressure_ms"].map(key=>[key,median(measured.map(s=>s[key as "extension_ms"]!))])):undefined,medianWall_ms:wall,medianSharpen_ms:sharpen,parity,stats:await solver.readStats(),samples}));
        assert.deepEqual(errors,[]);
      } finally {solver.destroy();query.destroy();resolved.destroy();staging.destroy();markerBuffer?.destroy();}
    }
    if(stages)continue;
    const dense=runs.filter(r=>!r.tileWork),tiled=runs.filter(r=>r.tileWork);
    const wallRatio=median(tiled.map(r=>r.wall))/median(dense.map(r=>r.wall));
    console.log(JSON.stringify({gate:sceneId,wallRatio,sharpenRatio:median(tiled.map(r=>r.sharpen))/median(dense.map(r=>r.sharpen)),noSlowdown:wallRatio<=1}));
    if(!quick&&sceneId==="minimal-power-dam-break-64")assert.ok(wallRatio<=1,`mini64 must not slow down: ${wallRatio}`);
  }
} finally {device?.destroy();await releaseWebGPUExclusiveLock();}
