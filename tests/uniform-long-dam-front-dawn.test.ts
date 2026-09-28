import assert from "node:assert/strict";
import test from "node:test";
import {pathToFileURL} from "node:url";
import {managedGPUDevice} from "../lib/core/gpu-compilation-manager";
import {requiredFluidDeviceLimits} from "../lib/core/webgpu-device-limits";
import {createProcessRetainedDawnGPU} from "../lib/harness/node-dawn-provider";
import {acquireWebGPUExclusiveLock,releaseWebGPUExclusiveLock} from "../lib/harness/webgpu-smoke-isolation";
import {sceneDocument} from "../lib/core/scene-definition";
import {getSceneDefinition} from "../lib/core/scenes";
import {refinementRegionLattice} from "../lib/core/refinement-regions";
import {uniformVolumeMethod} from "../lib/methods/uniform/uniform-volume-method";
import type {GPUSolverInstance} from "../lib/core/method-contract";
import {readMixedTexture} from "./helpers/uniform-mixed-native-fields";

const modulePath=process.env.WEBGPU_NODE_MODULE;
(modulePath?test:test.skip)("long dam retains its advancing surface toe at 0.333 and 0.667 seconds",{timeout:240000},async()=>{
 await acquireWebGPUExclusiveLock("dawn-test","Uniform long dam front progression");
 let device:GPUDevice|undefined,solver:GPUSolverInstance|undefined;
 try{
  const dawn=await import(pathToFileURL(modulePath!).href);Object.assign(globalThis,dawn.globals);
  const gpu=createProcessRetainedDawnGPU(dawn,["backend=metal"]),adapter=await gpu.requestAdapter();assert.ok(adapter);
  device=managedGPUDevice(await adapter.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter.limits)}),{requireWorkerRealm:false});
  const errors:string[]=[];device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);});
  const failures:string[]=[];
  for(const arm of ["fine","coarse corner","dynamic"] as const){
   const scene=structuredClone(sceneDocument(getSceneDefinition("sparse-cm12-long-dam-break")));
   const lattice=refinementRegionLattice(scene),d=lattice.dimensions;assert.deepEqual(d,[192,96,32]);
   const percent=[91.6667,0,75,100,8.3333,100],axes=["x","y","z"] as const;
   const point=(offset:number)=>Object.fromEntries(axes.map((a,i)=>[a,lattice.origin_m[a]+percent[offset+i]!/100*d[i]!*lattice.cellSize_m[i]!])) as {x:number;y:number;z:number};
   scene.fluid.refinementRegions=arm==="coarse corner"?[{id:"far-corner",rule:"minimum-cell-size",minimumCellSize_cells:4,maximumCellSize_cells:4,min_m:point(0),max_m:point(3)}]:[];
   solver=await uniformVolumeMethod.createSolverAsync!(device,scene,"balanced",{coarsening:arm==="dynamic"?"dynamic":"regions"},undefined,()=>{});
   for(let step=1;step<=20;step++){
    assert.ok(solver.advanceTo(step/30,[]));await solver.awaitFrameCompletion?.();assert.equal(solver.info.simulationPipelineError,undefined);
    if(step!==10&&step!==20)continue;
   const fields=solver as unknown as {volumeA:GPUTexture;vertexPhiField:GPUTexture;mixedFrame:{ownership:{layout:{tiles:Uint32Array}}}};
   const volume=await readMixedTexture(device,fields.volumeA),phi=await readMixedTexture(device,fields.vertexPhiField);
   let surfaceFront=-1,toeMass=0;
   for(let x=0;x<=d[0];x++)for(let y=1;y<=d[1];y++)if(phi[x+(d[0]+1)*(y+(d[1]+1)*(d[2]/2))]!<0)surfaceFront=Math.max(surfaceFront,x);
   // Measure actual liquid as well as its rendered level-set toe. Sample
   // each h-volume from its current owner, including dynamically coarse air.
   const toeStart=step===10?88:168;
   for(let z=0;z<d[2];z++)for(let y=0;y<d[1];y++)for(let x=toeStart;x<toeStart+8;x++){
    const tile=Math.floor(x/4)+(d[0]/4)*(Math.floor(y/4)+(d[1]/4)*Math.floor(z/4));
    const word=fields.mixedFrame.ownership.layout.tiles[tile]!,width=word&0x80000000?1:word&0x40000000?2:4;
    const p=[x,y,z].map(v=>Math.floor(v/width)*width);
    toeMass+=volume[p[0]!+d[0]*(p[1]!+d[1]*p[2]!)]!;
   }
   // Frozen original solver, before the rejected velocity and all-owner
   // pressure-graph experiments: fine fronts 90/169, coarse corner 90/168.
   // Keep the same two-cell tolerance and volume limits. The graph-era
   // frame-20 reference (171) was not the original solver's progression.
   const expectedFront=step===10?90:169;
   if(Math.abs(surfaceFront-expectedFront)>2)failures.push(`${arm}, frame ${step}: surface front ${surfaceFront}, expected ${expectedFront} ± 2 h`);
   if(!(toeMass>(step===10?10:200)&&toeMass<(step===10?40:800)))failures.push(`${arm}, frame ${step}: advancing toe mass ${toeMass}`);
   if(arm==="dynamic"){
    assert.ok((solver.info.uniformMixedCoarseTiles??0)>0,"dynamic front check must retain coarse work");
    assert.ok((solver.info.uniformMixedDynamicRelayouts??0)>0,"dynamic front check must change ownership");
   }
   console.log({arm,time_s:step/30,surfaceFront,toeMass});
   }
   solver.destroy();solver=undefined;await device.queue.onSubmittedWorkDone();
  }
  assert.deepEqual(errors,[]);
  assert.deepEqual(failures,[],"Long-dam front and advancing liquid must retain the original reference bounds");
 }finally{solver?.destroy();device?.destroy();await releaseWebGPUExclusiveLock();}
});
