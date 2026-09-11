import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdir, writeFile } from "node:fs/promises";
import { fileURLToPath, pathToFileURL } from "node:url";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { adaptiveMassSolverOptions } from "../lib/methods/adaptive-volume/method";
import { WebGPUAdaptiveMassSolver } from "../lib/methods/adaptive-volume/webgpu-adaptive-mass-solver";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { readPublishedCM12Field } from "./sparse-cm12-published-field";

const output=process.env.BOWL_REGION_OUTPUT ?? "artifacts/bowl-region-edit";
const live=new Set<GPU>();
await acquireWebGPUExclusiveLock("dawn-probe","bowl-region-before-after");
let gpu:GPU|undefined,device:GPUDevice|undefined,solver:WebGPUAdaptiveMassSolver|undefined;
try {
  const dawn=await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE ?? fileURLToPath(new URL('../node_modules/webgpu/index.js',import.meta.url))).href);
  Object.assign(globalThis,dawn.globals);
  gpu=dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? 'metal'}`]);live.add(gpu!);
  const adapter=await gpu!.requestAdapter();assert.ok(adapter);
  device=await adapter.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter.limits)});
  const errors:string[]=[];device.addEventListener('uncapturederror',e=>{e.preventDefault();errors.push(e.error.message);});
  const hashes:Record<string,string>={};const compile=device.createShaderModule.bind(device);
  device.createShaderModule=d=>{hashes[d.label??String(Object.keys(hashes).length)]=createHash('sha256').update(d.code).digest('hex');return compile(d);};
  for(const arm of ['control','additive','split','flat-split']) {
    const scene=sceneDocument(getSceneDefinition('stationary-bowl'));
    if(arm==='flat-split' && scene.fluid.initialHeightField?.kind === 'quadratic') {scene.fluid.initialHeightField!.curvatureX_mInv=0;scene.fluid.initialHeightField!.curvatureZ_mInv=0;}
    const dt=scene.numerics.fixedDt_s=scene.numerics.maxDt_s=1/60;
    // Preserve the original unconditioned before/after diagnostic independently of UI defaults.
    const options=adaptiveMassSolverOptions({...getSceneDefinition('stationary-bowl').methodProfile!.overrides,
      gammaDiffusion:'off',surfaceSharpening:'off'});
    solver=await WebGPUAdaptiveMassSolver.createAsync(device,scene,'balanced',undefined,options,()=>{});
    await solver.waitForSimulationReady();
    const folder=`${output}/${arm}`;await mkdir(folder,{recursive:true});
    await writeFile(`${folder}/scene-before.json`,JSON.stringify(scene,null,2));
    const trace=[];
    const capture=async(label:string)=>{
      const fields=await solver!.readDiagnosticFields(true);
      const published=await readPublishedCM12Field(device!,solver!);
      const activity=await solver!.readGPUActivityPolicy();
      const frame=await solver!.readFrameControlQA();
      const [nx,ny,nz]=[solver!.info.nx,solver!.info.ny,solver!.info.nz];assert.deepEqual([nx,ny,nz],[48,32,40]);
      const heights=new Float32Array(nx*nz).fill(NaN),columns=new Float64Array(nx*nz);
      for(let z=0;z<nz;z++)for(let x=0;x<nx;x++)for(let y=0;y<ny;y++) {
        const at=x+nx*(y+ny*z);columns[x+nx*z]!+=fields.density[at]!*.05;
        if(y<ny-1){const lo=published.values[at]!,hi=published.values[at+nx]!;if(lo<=0&&hi>0)heights[x+nx*z]=(y+.5-lo/(hi-lo))*.05;}
      }
      for(const [name,data]of Object.entries({density:fields.density,velocity:fields.velocity,pressure:fields.pressure,phi:published.values,heights,columns}))
        await writeFile(`${folder}/${label}-${name}.bin`,new Uint8Array(data.buffer,data.byteOffset,data.byteLength));
      await writeFile(`${folder}/${label}-activity.json`,JSON.stringify(activity));
      await writeFile(`${folder}/${label}-frame.json`,JSON.stringify(frame??null));
      const stats=await solver!.readStats();await writeFile(`${folder}/${label}-stats.json`,JSON.stringify(stats));
      const row={label,encodedSteps:solver!.info.encodedSteps,
        widths:Object.fromEntries([1,2,4,8].map(w=>[w,activity.bricks.filter(b=>b.active&&8*b.spanBricks/b.acceptedResolution===w).length])),
        volume_m3:columns.reduce((a,b)=>a+b,0)*.05**2,
        maxSpeed:Math.max(...Array.from({length:fields.density.length},(_,i)=>Math.hypot(fields.velocity[4*i]!,fields.velocity[4*i+1]!,fields.velocity[4*i+2]!))),
        fault:frame?.fault,topologyFault:activity.faultFlags,committed:activity.committedBrickCount};
      assert.equal(activity.faultFlags,0);assert.equal(activity.commitFailed,false);if(frame)assert.equal(frame.fault,0);
      trace.push(row);console.log(JSON.stringify({arm,...row}));
    };
    await capture('reset');
    while(!solver.advanceTo(dt,[]))await new Promise(setImmediate);await solver.waitForTopologyReady();
    await capture('before');
    if(arm!=='control') {
      const original=structuredClone(scene.fluid.refinementRegions![0]!);
      const right={...original,id:'right-width2',minimumCellSize_cells:2,maximumCellSize_cells:2,
        min_m:{x:0,y:0,z:-1},max_m:{x:1.2,y:1.6,z:1}};
      scene.fluid.refinementRegions=arm==='additive'?[original,right]:[
        {...original,max_m:{x:0,y:1.6,z:1}},right];
      solver.applySceneUniforms(structuredClone(scene));
    }
    await writeFile(`${folder}/scene-after.json`,JSON.stringify(scene,null,2));
    await capture('after-uniforms');
    for(let step=2;step<=5;step++) {
      while(!solver.advanceTo(step*dt,[]))await new Promise(setImmediate);
      await solver.waitForTopologyReady();assert.equal(solver.info.encodedSteps,step);
      await capture(`step-${step}`);
    }
    await writeFile(`${folder}/trace.json`,JSON.stringify(trace,null,2));
    assert.deepEqual(errors,[]);solver.destroy();solver=undefined;
  }
  await writeFile(`${output}/shader-hashes.json`,JSON.stringify(hashes,null,2));
}finally{solver?.destroy();device?.destroy();await releaseWebGPUExclusiveLock();if(gpu)live.delete(gpu);}
