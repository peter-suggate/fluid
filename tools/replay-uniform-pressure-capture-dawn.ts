/** Replay a captured pressure equation without fluid advancement or mid-step adoption.
 * Run with node --import tsx; --capture=path.json.gz and --cycles=N are optional.
 * A rejected receipt is reported, never projected or accepted by this diagnostic.
 */
import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {gunzipSync} from "node:zlib";
import {pathToFileURL} from "node:url";
import {managedGPUDevice} from "../lib/core/gpu-compilation-manager";
import {requiredFluidDeviceLimits} from "../lib/core/webgpu-device-limits";
import {createProcessRetainedDawnGPU} from "../lib/harness/node-dawn-provider";
import {sceneDocument} from "../lib/core/scene-definition";
import {getSceneDefinition} from "../lib/core/scenes";
import {uniformVolumeMethod} from "../lib/methods/uniform/uniform-volume-method";
import {uniformMixedLayoutFromTiles} from "../lib/methods/uniform/uniform-mixed-layout";
import type {GPUSolverInstance} from "../lib/core/method-contract";
const arg=(key:string,fallback:string)=>process.argv.find(a=>a.startsWith(`--${key}=`))?.slice(key.length+3)??fallback;
const path=arg("capture","docs/benchmarks/uniform-midstep-pressure-failure-2026-09-28.json.gz");
const capture=JSON.parse(gunzipSync(readFileSync(path)).toString());
const count=Number(arg("cycles","1"));assert.ok(Number.isInteger(count)&&count>0&&count<=100);
let device:GPUDevice|undefined,solver:GPUSolverInstance|undefined;
try {
 const dawn=await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE??`${process.cwd()}/node_modules/webgpu/index.js`).href);
 Object.assign(globalThis,dawn.globals);
 const gpu=createProcessRetainedDawnGPU(dawn,["backend=metal"]),adapter=await gpu.requestAdapter();assert.ok(adapter);
 device=managedGPUDevice(await adapter.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter.limits)}),{requireWorkerRealm:false});
 const errors:string[]=[];device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);});
 const scene=structuredClone(sceneDocument(getSceneDefinition(capture.sceneId)));scene.fluid.refinementRegions=[];
 solver=await uniformVolumeMethod.createSolverAsync!(device,scene,"balanced",{pressureResidualTolerance:capture.parameters.pressureTolerance,detailPolicy:"full"},undefined,()=>{});
 // Deliberate diagnostic access to the production frame and its native continuation.
 const host=solver as any,frame=host.mixedFrame;assert.ok(frame);
 frame.updateLayout(uniformMixedLayoutFromTiles(capture.lattice,new Uint32Array(capture.simulationTiles),[]));
 host.writeParams(capture.parameters.dt,0,0);frame.write(capture.parameters);
 const clear=device.createCommandEncoder();
 for(let i=0;i<frame.levels.length;i++){
  const level=frame.levels[i],saved=capture.levels[i];
  assert.deepEqual(Array.from(level.ownership.layout.tiles),saved.tiles,`Pressure level ${i} ownership differs`);
  assert.equal(level.ownership.layout.cellCount,saved.cellCount);
  for(const view of [level.pressure,level.frozen,level.residual])clear.clearBuffer(view.buffer,view.offset??0,view.size);
 }
 device.queue.submit([clear.finish()]);
 for(let i=0;i<frame.levels.length;i++){
  const level=frame.levels[i],saved=capture.levels[i];
  for(const [view,values] of [[level.rhs[0],saved.rhs],[level.phi,saved.phi],[level.minimum[0],saved.minimum]] as [GPUBufferBinding,number[]][]){
   assert.ok(values.every(Number.isFinite));device.queue.writeBuffer(view.buffer,view.offset??0,Float32Array.from(values));
  }
 }
 const report=async(encoder:GPUCommandEncoder,kind:"initial"|"cycle",cycle:number)=>{
  frame.cycles.encodeMeasure(encoder);frame.acceptance.encode(encoder,frame.acceptanceGroup,frame.state,kind);
  const state:Uint32Array=await frame.receipt(encoder),v=new Float32Array(state.buffer);
  console.log(JSON.stringify({capture:path,cycle,candidate:v[0],accepted:v[1],previous:v[2],rejected:!!state[4],converged:!!state[5]}));
  return state;
 };
 let encoder=device.createCommandEncoder();frame.cycles.encodeSurfaceRestriction(encoder);await report(encoder,"initial",0);
 for(let cycle=1;cycle<=count;cycle++){
  encoder=device.createCommandEncoder();frame.cycles.encodeVCycle(encoder);
  const state=await report(encoder,"cycle",cycle);if(state[4]||state[5])break;
 }
 assert.deepEqual(errors,[]);
}finally{solver?.destroy();device?.destroy();}
