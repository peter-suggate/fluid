// Surface-correctness comparison: dumps vertex phi and total V per frame for one
// scene under one pressure/coarsening arm, for tools/analyze-uniform-surface-compare.mjs.
// Usage: WEBGPU_NODE_MODULE=... node --import tsx tools/probe-uniform-surface-compare-dawn.ts
//   <scene-id> <fine|default> <frames> <every> <out-dir>
//   fine: band pressure, all-h simulation (the reference)
//   default: band pressure, dynamic coarsening (the shipped default)
// PROBE_OVERRIDES: extra method values as JSON, e.g. {"sharpeningStrength":0}.
import {mkdirSync,writeFileSync} from "node:fs";
import {pathToFileURL} from "node:url";
import {managedGPUDevice} from "../lib/core/gpu-compilation-manager";
import {requiredFluidDeviceLimits} from "../lib/core/webgpu-device-limits";
import {createProcessRetainedDawnGPU} from "../lib/harness/node-dawn-provider";
import {sceneDocument} from "../lib/core/scene-definition";
import {getSceneDefinition} from "../lib/core/scenes";
import {refinementRegionLattice} from "../lib/core/refinement-regions";
import {withRefinementRegionsFromQuery} from "../lib/core/editor-refinement-region";
import {uniformVolumeMethod} from "../lib/methods/uniform/uniform-volume-method";
import type {GPUSolverInstance} from "../lib/core/method-contract";
import {readMixedTexture} from "../tests/helpers/uniform-mixed-native-fields";

const [sceneId,arm,framesArg,everyArg,out]=process.argv.slice(2) as [string,"fine"|"default",string,string,string];
const frames=Number(framesArg),every=Number(everyArg);
if(arm!=="fine"&&arm!=="default")throw new Error(`Unknown arm ${arm}`);
const dynamic=arm==="default";
mkdirSync(out,{recursive:true});
let device:GPUDevice|undefined,solver:GPUSolverInstance|undefined;
try{
 const dawn=await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href);Object.assign(globalThis,dawn.globals);
 const gpu=createProcessRetainedDawnGPU(dawn,["backend=metal"]),adapter=await gpu.requestAdapter();
 device=managedGPUDevice(await adapter!.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter!.limits)}),{requireWorkerRealm:false});
 const errors:string[]=[];
 device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);console.error("GPU error:",e.error.message);});
 const scene=structuredClone(sceneDocument(getSceneDefinition(sceneId)));
 // PROBE_REGIONS: the app URL's `regions=` value, e.g. 0_0_0_100_100_100_4_4.
 scene.fluid.refinementRegions=[];if(process.env.PROBE_REGIONS)Object.assign(scene,withRefinementRegionsFromQuery(scene,process.env.PROBE_REGIONS,"uniform-volume"));
 // PROBE_DT: fixed step override (s), e.g. 1/120 for four steps a frame.
 if(process.env.PROBE_DT)scene.numerics.fixedDt_s=scene.numerics.maxDt_s=Number(eval(process.env.PROBE_DT));
 const d=refinementRegionLattice(scene).dimensions;
 solver=await uniformVolumeMethod.createSolverAsync!(device,scene,"balanced",
  {detailPolicy:dynamic?"dynamic":"full",surfaceDeficitBalancing:"off",...JSON.parse(process.env.PROBE_OVERRIDES??"{}")},undefined,()=>{});
 const fields=solver as unknown as {volumeA:GPUTexture;vertexPhiField:GPUTexture;mixedFrame:{bandTiles?:number;bandResidual?:number;ownership:{layout:{tiles:Uint32Array}}}};
 if(process.env.PROBE_PRESSURE_TOLERANCE)(solver as unknown as {applyRuntimeValues(v:Record<string,number>):void}).applyRuntimeValues({pressureResidualTolerance:Number(process.env.PROBE_PRESSURE_TOLERANCE)});
 const rows=[];
 // The lagged plan (last frame's cycles + 1) is fatal when a frame needs
 // more; encode the conservative slot list every frame. Gates still stop at
 // convergence, so every arm gets an unlagged solve. The unused gated slots
 // cost ~15 ms a frame: set PROBE_LAGGED_PLAN=1 for timing.
 const frame=(solver as unknown as {mixedFrame:{pressurePlan:unknown;schedule:{vCycles:number;fullCycles:number}}}).mixedFrame;
 for(let step=1;step<=frames;step++){
  if(process.env.PROBE_LAGGED_PLAN!=="1")frame.pressurePlan={vCycles:frame.schedule.vCycles,fullCycles:frame.schedule.fullCycles};
  const started=performance.now();
  if(!solver.advanceTo(step/Number(process.env.PROBE_RATE??30),[]))throw new Error("advance refused");await solver.awaitFrameCompletion?.();
  const wall=performance.now()-started;
  if(solver.info.simulationPipelineError)throw new Error(String(solver.info.simulationPipelineError));
  const volume=await readMixedTexture(device,fields.volumeA),tiles=fields.mixedFrame.ownership.layout.tiles;
  // V is stored once per owner at its origin texel: weight by the owner's cell count.
  let total=0;
  for(let z=0;z<d[2];z++)for(let y=0;y<d[1];y++)for(let x=0;x<d[0];x++){
   const word=tiles[(x>>2)+(d[0]>>2)*((y>>2)+(d[1]>>2)*(z>>2))]!,width=word&0x80000000?1:4;
   if(x%width===0&&y%width===0&&z%width===0)total+=volume[x+d[0]*(y+d[1]*z)]!*width**3;
  }
  let fine=0;for(const word of tiles)if(word&0x80000000)fine++;
  const row={step,wall_ms:+wall.toFixed(1),volume:+total.toFixed(2),fineTiles:fine,bandTiles:fields.mixedFrame.bandTiles,bandResidual:fields.mixedFrame.bandResidual,residual:(solver.info as unknown as Record<string,number>).uniformPressureAcceptedResidual,cycles:(solver.info as unknown as Record<string,number>).uniformPressureCyclesExecuted};
  rows.push(row);console.log(JSON.stringify(row));
  if(step%every===0){
   const phi=await readMixedTexture(device,fields.vertexPhiField);
   writeFileSync(`${out}/phi-${String(step).padStart(4,"0")}.f32`,Buffer.from(phi.buffer));
   writeFileSync(`${out}/vol-${String(step).padStart(4,"0")}.f32`,Buffer.from(volume.buffer));
   if(process.env.PROBE_VELOCITY==="1"){const v=await readMixedTexture(device,(solver as unknown as {velocityA:GPUTexture}).velocityA);writeFileSync(`${out}/vel-${String(step).padStart(4,"0")}.f32`,Buffer.from(v.buffer));}
   // PROBE_BAND_DUMP=1: the band's linear system after the sweeps (index, rows, iterate + halo, params).
   const bandSolver=(fields.mixedFrame as unknown as {band?:{index:GPUBuffer;rows:GPUBuffer;iterate:GPUBuffer;fields:{params:GPUBuffer}}}).band;
   if(process.env.PROBE_BAND_DUMP==="1"&&bandSolver)for(const [name,buffer] of [["index",bandSolver.index],["rows",bandSolver.rows],["solve",bandSolver.iterate],["params",bandSolver.fields.params]] as const){
    const staging=device.createBuffer({size:buffer.size,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ}),encoder=device.createCommandEncoder();
    encoder.copyBufferToBuffer(buffer,0,staging,0,buffer.size);device.queue.submit([encoder.finish()]);await staging.mapAsync(GPUMapMode.READ);
    writeFileSync(`${out}/band-${name}-${String(step).padStart(4,"0")}.bin`,Buffer.from(staging.getMappedRange().slice(0)));staging.destroy();
   }
   const fine=new Uint8Array(tiles.length);tiles.forEach((w,i)=>{fine[i]=w&0x80000000?1:4;});
   writeFileSync(`${out}/tiles-${String(step).padStart(4,"0")}.u8`,fine);
  }
 }
 writeFileSync(`${out}/run.json`,JSON.stringify({sceneId,arm,dims:d,cellSize:refinementRegionLattice(scene).cellSize_m,rows,errors},null,1));
}finally{solver?.destroy();device?.destroy();}
