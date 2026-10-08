import assert from "node:assert/strict";
import test from "node:test";
import {sceneDocument} from "../lib/core/scene-definition";
import {getSceneDefinition} from "../lib/core/scenes";
import {uniformNarrowBandMethod} from "../lib/methods/uniform/uniform-narrow-band-method";
import type {WebGPUUniformReferenceSolver} from "../lib/methods/uniform/webgpu-uniform-reference";
import {advanceUniform,withUniformDevice} from "./helpers/uniform-geometric";
import {readMixedBuffer,readMixedTexture} from "./helpers/uniform-mixed-native-fields";

(process.env.WEBGPU_NODE_MODULE?test:test.skip)("FLIP post-pressure extension equals a fresh next-frame rebuild",{timeout:180_000},async()=>{
 await withUniformDevice("FLIP extension reuse",async device=>{
  const scene=structuredClone(sceneDocument(getSceneDefinition("minimal-power-dam-break-32")));
  scene.rigidBodies=[];
  const solver=await uniformNarrowBandMethod.createSolverAsync!(device,scene,"balanced",{timeStep:"paper"},undefined,()=>{}) as WebGPUUniformReferenceSolver;
  try{
   const frame=(solver as unknown as {mixedFrame:{fields:{velocityScratch:GPUTexture;negativeScratch:GPUBuffer};encodeExtension(encoder:GPUCommandEncoder):void}}).mixedFrame;
   for(let step=1;step<=8;step++){
    if(step===5){solver.applyRuntimeValues({detailPolicy:"full"});await solver.pipelinesPrepared();}
    await advanceUniform(solver,step/30);
    if(step!==4&&step!==8)continue;
    const before=[await readMixedTexture(device,frame.fields.velocityScratch),await readMixedBuffer(device,frame.fields.negativeScratch)];
    const encoder=device.createCommandEncoder();frame.encodeExtension(encoder);device.queue.submit([encoder.finish()]);
    const after=[await readMixedTexture(device,frame.fields.velocityScratch),await readMixedBuffer(device,frame.fields.negativeScratch)];
    for(let field=0;field<before.length;field++){
     let difference=0;for(let i=0;i<before[field]!.length;i++)difference=Math.max(difference,Math.abs(before[field]![i]!-after[field]![i]!));
     assert.ok(difference<1e-6,`step ${step}, field ${field}: reused extension differs by ${difference}`);
    }
   }
  }finally{solver.destroy();}
 });
});
