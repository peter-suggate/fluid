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
import { resolveMethodValues } from "../lib/core/method-contract";
import type { WebGPUUniformReferenceSolver } from "../lib/methods/uniform/webgpu-uniform-reference";
import type { GPUTimestampPhase } from "../lib/core/performance-trace";
interface Access{volumeEdges:GPUBuffer;encodeGeometricVolume(e:GPUCommandEncoder,seam?:(p:GPUTimestampPhase)=>void):void;}
await acquireWebGPUExclusiveLock("dawn-probe","uniform brick work census");
let device:GPUDevice|undefined,solver:WebGPUUniformReferenceSolver|undefined;
try{
 const dawn=await import(pathToFileURL(resolve("node_modules/webgpu/index.js")).href);Object.assign(globalThis,dawn.globals);
 const gpu=createProcessRetainedDawnGPU(dawn,["backend=metal"]);const adapter=await gpu.requestAdapter();assert.ok(adapter);
 device=managedGPUDevice(await adapter.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter.limits)}),{requireWorkerRealm:false});
 const errors:string[]=[];device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);});
 const scene=sceneDocument(getSceneDefinition("minimal-power-dam-break-64"));
 solver=await uniformVolumeMethod.createSolverAsync!(device,scene,"balanced",resolveMethodValues(uniformVolumeMethod,"balanced",{liquidCapacityBalancing:"on",velocityTransport:"maccormack",liquidCapacityBalancingTolerance:0.1,liquidCapacityBalancingRounds:64}),undefined,()=>{}) as WebGPUUniformReferenceSolver;
 const access=solver as unknown as Access;const n=64,N=n**3;
 const frames=new Set([2,8,22,30]);const rounds=new Set([0,3,15,63]);
 let frame=0;let captures:{round:number,buffer:GPUBuffer}[]=[];
 const fields=device.createBuffer({size:N*8,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
 const original=access.encodeGeometricVolume.bind(access);
 access.encodeGeometricVolume=(e,seam)=>{
  if(!frames.has(frame)){original(e,seam);return;}
  e.copyTextureToBuffer({texture:solver!.volumeTexture},{buffer:fields,bytesPerRow:256,rowsPerImage:n},[n,n,n]);
  e.copyTextureToBuffer({texture:solver!.denseLevelSetVolumeSource!.openFraction},{buffer:fields,offset:N*4,bytesPerRow:256,rowsPerImage:n},[n,n,n]);
  let round=0;
  const proxy=new Proxy(e,{get(target,key){if(key==="beginComputePass")return (desc:GPUComputePassDescriptor)=>{
   if(desc?.label==="uvBalanceLiquidRows"){
    if(rounds.has(round)){const buffer=device!.createBuffer({size:N*80,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});target.copyBufferToBuffer(access.volumeEdges,0,buffer,0,N*80);captures.push({round:round+1,buffer});}round++;
   }return target.beginComputePass(desc);
  };const value=Reflect.get(target,key,target);return typeof value==="function"?value.bind(target):value;}});
  original(proxy,seam);
 };
 for(frame=1;frame<=30;frame++){
  assert.ok(solver.advanceTo(frame/30));await device.queue.onSubmittedWorkDone();if(!frames.has(frame))continue;
  await fields.mapAsync(GPUMapMode.READ);const all=new Float32Array(fields.getMappedRange());const volume=all.slice(0,N),capacity=all.slice(N);fields.unmap();
  for(const capture of captures){
   await capture.buffer.mapAsync(GPUMapMode.READ);const mapped=capture.buffer.getMappedRange();const u=new Uint32Array(mapped),f=new Float32Array(mapped);
   const violating=new Uint8Array(N),donors=new Uint8Array(N),receivers=new Uint8Array(N);let maximumError=0,violations=0,positiveEdges=0,wetReceivers=0;
   for(let i=0;i<N;i++){
    let amount=0;for(let k=0;k<9;k++){const d=u[20*i+k]!,w=f[20*i+9+k]!;amount+=w*volume[d]!;if(w>0&&volume[d]!>0)positiveEdges++;}
    if(amount>1e-12)wetReceivers++;
    const error=Math.max(0,amount-capacity[i]!)/Math.max(capacity[i]!,1e-20);maximumError=Math.max(maximumError,error);
    if(error>0.001){violating[i]=1;violations++;for(let k=0;k<9;k++){const d=u[20*i+k]!;if(f[20*i+9+k]!>0&&volume[d]!>0)donors[d]=1;}}
   }
   for(let i=0;i<N;i++)for(let k=0;k<9;k++)if(f[20*i+9+k]!>0&&donors[u[20*i+k]!]!){receivers[i]=1;break;}
   const reachable=new Float64Array(N);const degree=new Uint32Array(N);
   for(let i=0;i<N;i++){const seen:number[]=[];for(let k=0;k<9;k++){const d=u[20*i+k]!;
    if(f[20*i+9+k]!<=0||volume[d]!<=0||seen.includes(d))continue;
    seen.push(d);reachable[d]=reachable[d]!+capacity[i]!;degree[d]=degree[d]!+1;
   }}
   let infeasibleDonors=0,infeasibleVolume=0,capacityLowerBound=0,singleReceiverDonors=0;
   for(let j=0;j<N;j++){if(volume[j]!>0&&degree[j]===1)singleReceiverDonors++;
    if(volume[j]!>reachable[j]!*(1+0.001)){infeasibleDonors++;infeasibleVolume+=volume[j]!;
     capacityLowerBound=Math.max(capacityLowerBound,volume[j]!/Math.max(reachable[j]!,1e-20)-1);}}
   const bricks=[];
   for(const B of [4,8]){
    const side=n/B,count=side**3;const brick=(i:number)=>Math.floor((i%n)/B)+side*(Math.floor((Math.floor(i/n)%n)/B)+side*Math.floor(Math.floor(i/(n*n))/B));
    const bad=new Uint8Array(count),touched=new Uint8Array(count),affected=new Uint8Array(count),coarseAffected=new Uint8Array(count);
    for(let i=0;i<N;i++){if(violating[i])bad[brick(i)]=1;if(donors[i])touched[brick(i)]=1;if(receivers[i])affected[brick(i)]=1;}
    for(let i=0;i<N;i++)for(let k=0;k<9;k++){const d=u[20*i+k]!;if(f[20*i+9+k]!>0&&volume[d]!>0&&touched[brick(d)]){coarseAffected[brick(i)]=1;break;}}
    const sum=(a:Uint8Array)=>a.reduce((a,b)=>a+b,0);
    bricks.push({B,total:count,violating:sum(bad),donorBricks:sum(touched),exactAffected:sum(affected),coarseAffected:sum(coarseAffected)});
   }
   console.log(JSON.stringify({frame,round:capture.round,cells:N,violations,wetReceivers,positiveEdges,changedDonors:donors.reduce((a,b)=>a+b,0),exactAffectedReceivers:receivers.reduce((a,b)=>a+b,0),maximumError,infeasibleDonors,infeasibleVolume,capacityLowerBound,singleReceiverDonors,bricks}));
   capture.buffer.unmap();capture.buffer.destroy();
  }captures=[];
 }
 fields.destroy();assert.deepEqual(errors,[]);
}finally{solver?.destroy();device?.destroy();await releaseWebGPUExclusiveLock();}
