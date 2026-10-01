import assert from "node:assert/strict";
import test from "node:test";
import {pathToFileURL} from "node:url";
import {managedGPUDevice} from "../lib/core/gpu-compilation-manager";
import {requiredFluidDeviceLimits} from "../lib/core/webgpu-device-limits";
import {createProcessRetainedDawnGPU} from "../lib/harness/node-dawn-provider";
import {acquireWebGPUExclusiveLock,releaseWebGPUExclusiveLock} from "../lib/harness/webgpu-smoke-isolation";
import {sceneDocument} from "../lib/core/scene-definition";
import {getSceneDefinition} from "../lib/core/scenes";
import {uniformVolumeMethod} from "../lib/methods/uniform/uniform-volume-method";
import {UniformMixedForces,type UniformMixedForceFields} from "../lib/methods/uniform/uniform-mixed-forces";
import type {GPUSolverInstance} from "../lib/core/method-contract";
import {readMixedTexture,readMixedTileWords} from "./helpers/uniform-mixed-native-fields";

const modulePath=process.env.WEBGPU_NODE_MODULE;
(modulePath?test:test.skip)("cached curvature preserves forces on the same evolving mixed/solid state",{timeout:240000},async()=>{
 await acquireWebGPUExclusiveLock("dawn-test","Uniform force curvature cache");
 let device:GPUDevice|undefined,solver:GPUSolverInstance|undefined;
 const copies:GPUTexture[]=[];
 const bind=UniformMixedForces.prototype.bind;
 try{
  const dawn=await import(pathToFileURL(modulePath!).href);Object.assign(globalThis,dawn.globals);
  const adapter=await createProcessRetainedDawnGPU(dawn,["backend=metal"]).requestAdapter();assert.ok(adapter);
  device=managedGPUDevice(await adapter.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter.limits)}),{requireWorkerRealm:false});
  const errors:string[]=[];device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);});
  let fields:UniformMixedForceFields|undefined;
  UniformMixedForces.prototype.bind=function(f){fields=f;return bind.call(this,f);};
  const scene=sceneDocument(getSceneDefinition("twin-dam-collision"));
  scene.numerics.fixedDt_s=scene.numerics.maxDt_s=1/60;
  solver=await uniformVolumeMethod.createSolverAsync!(device,scene,"balanced",{timeStep:"scene"},undefined,()=>{});
  UniformMixedForces.prototype.bind=bind;
  assert.ok(fields?.curvature,"Production borrows curvature scratch");
  const frame=(solver as any).mixedFrame;
  const reference=new UniformMixedForces(device,frame.ownership,true,frame.fields.sourceParams,frame.solid);
  await reference.initialize();
  const referenceGroup=reference.bind({...fields,curvature:undefined,normals:undefined});
  for(let i=0;i<3;i++)copies.push(device.createTexture({size:[fields.output.width,fields.output.height,fields.output.depthOrArrayLayers],dimension:"3d",format:"rgba32float",usage:GPUTextureUsage.COPY_SRC|GPUTextureUsage.COPY_DST}));
  let sample=false,samples=0,seams=false;
  const encode=frame.forces.encode.bind(frame.forces);
  frame.forces.encode=(encoder:GPUCommandEncoder,group:GPUBindGroup,capillarity:boolean)=>{
   encode(encoder,group,capillarity);
   if(!sample)return;
   const size:[number,number,number]=[fields!.output.width,fields!.output.height,fields!.output.depthOrArrayLayers];
   encoder.copyTextureToTexture({texture:fields!.output},{texture:copies[0]!},size);
   // Disabled cache specialization must preserve the original inline path.
   encode(encoder,group,capillarity,false);
   encoder.copyTextureToTexture({texture:fields!.output},{texture:copies[2]!},size);
   // Inline stencil, identical frozen inputs. Reuse the same output so stale
   // noncanonical texels also match; projection later consumes this output.
   reference.encode(encoder,referenceGroup);
   encoder.copyTextureToTexture({texture:fields!.output},{texture:copies[1]!},size);
  };
  for(let step=1;step<=180;step++){
   sample=step===1||step%30===0;
   assert.ok(solver.advanceTo(step/60,[]));await solver.awaitFrameCompletion?.();
   if(!sample)continue;
   const a=await readMixedTexture(device,copies[0]!),b=await readMixedTexture(device,copies[1]!),inline=await readMixedTexture(device,copies[2]!);
   assert.deepEqual(inline,b,"Disabled cache must match the original inline force path");
   let delta=0,scale=1;
   for(let i=0;i<a.length;i++){assert.ok(Number.isFinite(a[i]));delta=Math.max(delta,Math.abs(a[i]!-b[i]!));scale=Math.max(scale,Math.abs(b[i]!));}
   assert.ok(delta<=2e-6*scale,`step ${step}: cached/inline force delta ${delta}, scale ${scale}`);
   const words=await readMixedTileWords(device,solver);seams ||= words.some(w=>(w&0x80000000)!==0)&&words.some(w=>(w&0x80000000)===0);
   samples++;
  }
  assert.equal(samples,7);assert.ok(seams,"Exercise h/4h face stencils");assert.deepEqual(errors,[]);
 }finally{UniformMixedForces.prototype.bind=bind;solver?.destroy();for(const t of copies)t.destroy();device?.destroy();await releaseWebGPUExclusiveLock();}
});
