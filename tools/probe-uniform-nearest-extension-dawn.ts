/** Figure 3 regression and sequential queue-fenced timing; no renderer running. */
import assert from "node:assert/strict";
import { pathToFileURL } from "node:url";
import { writeFileSync } from "node:fs";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import { managedGPUDevice } from "../lib/core/gpu-compilation-manager";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { uniformGeometricSolverOptions } from "../lib/methods/uniform/uniform-geometric-options";
import { WebGPUUniformReferenceSolver } from "../lib/methods/uniform/webgpu-uniform-reference";
import { readFloatTexture3D } from "../lib/harness/webgpu-smoke-readbacks";
import type { WebGPUUniformVelocityExtrapolator } from "../lib/methods/uniform/webgpu-uniform-velocity-extrapolation";
await acquireWebGPUExclusiveLock("dawn-probe","3D nearest source extension scene and timing");
let device: GPUDevice | undefined;
const results: unknown[] = [];
try {
  const dawn=await import(pathToFileURL(process.cwd()+"/node_modules/webgpu/index.js").href);Object.assign(globalThis,dawn.globals);
  const gpu=createProcessRetainedDawnGPU(dawn,["backend=metal"]);const adapter=await gpu.requestAdapter();assert.ok(adapter);
  device=managedGPUDevice(await adapter.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter.limits)}),{requireWorkerRealm:false});
  const errors:string[]=[];device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);console.error(e.error.message);});
  const timingOnly=process.argv.includes("--timing-only");
  for(const sceneId of timingOnly?["minimal-power-dam-break-64"]:["cm12-figure-3","minimal-power-dam-break-64"]){
    for(const mode of ["prior","nearest-unfused","nearest-fused"]){
      const scene=sceneDocument(getSceneDefinition(sceneId));
      const solver=await WebGPUUniformReferenceSolver.createAsync(device,scene,"balanced",undefined,{
        ...uniformGeometricSolverOptions({},scene),sourceAwareExtension:mode!=="prior",fuseExtensionPack:mode==="nearest-fused",
      },()=>{});
      try {
        const {nx,ny,nz}=solver.info;
        const extension=(solver as unknown as {velocityExtrapolator:WebGPUUniformVelocityExtrapolator}).velocityExtrapolator;
        assert.equal(extension.frontPasses,2);
        const frames:unknown[]=[],times:number[]=[];
        const snapshot=async(frame:number)=>{
          const v=await readFloatTexture3D(device!,solver.volumeTexture,nx,ny,nz);
          const phi=await readFloatTexture3D(device!,solver.vertexPhiTexture!,nx+1,ny+1,nz+1);
          let airborneVolume=0,airborneNegativeVertices=0;
          for(let z=0;z<nz;z++)for(let y=25;y<ny;y++)for(let x=0;x<nx;x++)airborneVolume+=v[x+nx*(y+ny*z)]!;
          for(let z=1;z<nz;z++)for(let y=25;y<=ny;y++)for(let x=0;x<=nx;x++)if(phi[x+(nx+1)*(y+(ny+1)*z)]!<0)airborneNegativeVertices++;
          assert.ok(v.every(v=>Number.isFinite(v)&&v>=-1e-6));assert.ok(phi.every(Number.isFinite));
          frames.push({frame,mass:v.reduce((a,b)=>a+b,0),airborneVolume,airborneNegativeVertices});
        };
        if(sceneId==="cm12-figure-3")await snapshot(0);
        for(let frame=1;frame<=60;frame++){
          const start=performance.now();assert.ok(solver.advanceTo(frame/30));await device.queue.onSubmittedWorkDone();
          if(frame>10)times.push(performance.now()-start);
          if(sceneId==="cm12-figure-3"&&[20,24,30,40,60].includes(frame))await snapshot(frame);
        }
        // Frozen extension isolates packing/transfer cost from diverging scene trajectories.
        const access=solver as unknown as {encodeVelocityExtrapolation(e:GPUCommandEncoder,p:boolean):void};
        const extensionTimes:number[]=[];
        for(let batch=0;batch<12;batch++){
          const start=performance.now(),encoder=device.createCommandEncoder();
          for(let i=0;i<20;i++)access.encodeVelocityExtrapolation(encoder,false);
          device.queue.submit([encoder.finish()]);await device.queue.onSubmittedWorkDone();
          if(batch>1)extensionTimes.push((performance.now()-start)/20);
        }
        times.sort((a,b)=>a-b);extensionTimes.sort((a,b)=>a-b);assert.deepEqual(errors,[]);
        const result={sceneId,mode,dims:[nx,ny,nz],medianWallMs:times[Math.floor(times.length/2)],medianExtensionMs:extensionTimes[Math.floor(extensionTimes.length/2)],extensionPasses:extension.encodedPassCount,scratchBytes:extension.scratchBytes,frames};
        results.push(result);console.log(JSON.stringify(result));
      }finally{solver.destroy();}
    }
  }
  const out=process.argv.find(v=>v.startsWith("--out="))?.slice(6);
  if(out)writeFileSync(out,JSON.stringify(results,null,2)+"\n");
}finally{device?.destroy();await releaseWebGPUExclusiveLock();}
