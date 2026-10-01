// Long-dam front probe for the two-stage band pressure (the only mixed pressure solve).
// Usage: WEBGPU_NODE_MODULE=... node --import tsx tools/probe-uniform-pressure-band-dawn.ts [fine|coarse|dynamic] [frames]
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
import {readMixedTexture} from "../tests/helpers/uniform-mixed-native-fields";

const arm=(process.argv[2]??"fine") as "fine"|"coarse"|"dynamic",frames=Number(process.argv[3]??20);
await acquireWebGPUExclusiveLock("dawn-probe","Uniform pressure band long dam");
let device:GPUDevice|undefined,solver:GPUSolverInstance|undefined;
try{
 const dawn=await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href);Object.assign(globalThis,dawn.globals);
 const gpu=createProcessRetainedDawnGPU(dawn,["backend=metal"]),adapter=await gpu.requestAdapter();
 device=managedGPUDevice(await adapter!.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter!.limits)}),{requireWorkerRealm:false});
 device.addEventListener("uncapturederror",e=>{e.preventDefault();console.error("GPU error:",e.error.message);});
 const scene=structuredClone(sceneDocument(getSceneDefinition("sparse-cm12-long-dam-break")));
 const lattice=refinementRegionLattice(scene),d=lattice.dimensions;
 const percent=[91.6667,0,75,100,8.3333,100],axes=["x","y","z"] as const;
 const point=(offset:number)=>Object.fromEntries(axes.map((a,i)=>[a,lattice.origin_m[a]+percent[offset+i]!/100*d[i]!*lattice.cellSize_m[i]!])) as {x:number;y:number;z:number};
 scene.fluid.refinementRegions=arm==="coarse"?[{id:"far-corner",rule:"minimum-cell-size",minimumCellSize_cells:4,maximumCellSize_cells:4,min_m:point(0),max_m:point(3)}]:[];
 solver=await uniformVolumeMethod.createSolverAsync!(device,scene,"balanced",{coarsening:arm==="dynamic"?"dynamic":"regions"},undefined,()=>{});
 // PROBE_PRESSURE_TOLERANCE: the mixed solver's acceptance tolerance (1/s).
 if(process.env.PROBE_PRESSURE_TOLERANCE)(solver as unknown as {applyRuntimeValues(v:Record<string,number>):void}).applyRuntimeValues({pressureResidualTolerance:Number(process.env.PROBE_PRESSURE_TOLERANCE)});
 const rows=[];
 for(let step=1;step<=frames;step++){
  const started=performance.now();
  if(!solver.advanceTo(step/30,[]))throw new Error("advance refused");await solver.awaitFrameCompletion?.();
  const wall=performance.now()-started;
  if(solver.info.simulationPipelineError)throw new Error(String(solver.info.simulationPipelineError));
  const fields=solver as unknown as {volumeA:GPUTexture;vertexPhiField:GPUTexture;mixedFrame:{bandTiles?:number;bandResidual?:number}};
  let surfaceFront=-1,toeMass:number|undefined;
  if(step%5===0||step===frames){
   const phi=await readMixedTexture(device,fields.vertexPhiField);
   for(let x=0;x<=d[0];x++)for(let y=1;y<=d[1];y++)if(phi[x+(d[0]+1)*(y+(d[1]+1)*(d[2]/2))]!<0)surfaceFront=Math.max(surfaceFront,x);
  }
  if(step===10||step===20){
   // Historical long-dam toe mass: bounds 10-40 (frame 10), 200-800 (frame 20).
   const volume=await readMixedTexture(device,fields.volumeA),layout=(solver as unknown as {mixedFrame:{ownership:{layout:{tiles:Uint32Array}}}}).mixedFrame.ownership.layout;
   const start=step===10?88:168;toeMass=0;
   for(let z=0;z<d[2];z++)for(let y=0;y<d[1];y++)for(let x=start;x<start+8;x++){
    const word=layout.tiles[Math.floor(x/4)+(d[0]/4)*(Math.floor(y/4)+(d[1]/4)*Math.floor(z/4))]!,width=word&0x80000000?1:4;
    const p=[x,y,z].map(v=>Math.floor(v/width)*width);toeMass+=volume[p[0]!+d[0]*(p[1]!+d[1]*p[2]!)]!;
   }
   toeMass=+toeMass.toFixed(1);
  }
  const row={step,toeMass,wall_ms:+wall.toFixed(1),bandTiles:fields.mixedFrame.bandTiles,bandResidual:fields.mixedFrame.bandResidual,residual:solver.info.pressureResidual,surfaceFront,cycles:solver.info.pressureIterations};
  rows.push(row);console.log(JSON.stringify(row));
 }
}finally{solver?.destroy();device?.destroy();await releaseWebGPUExclusiveLock();}
