/** Peak probe for Narrow-band FLIP: advances a scene and tracks, every step,
 * the two numbers its GPU memory follows: the page pool's resident pages and
 * the h tiles of the running layout. Browser GPU must be off.
 * node --import tsx tools/probe-nbflip-peaks.ts <scene> <steps> <every> [--dt=0.017] [--page-pool=<json>] [--fine-capacity=<tiles>]
 * --page-pool={"blocks":16}: the scene's page pool record (SceneDescription.fluid.pagePool).
 * --fine-capacity=2000: the layout builder's h-tile bound, set below the
 * storage's (UniformMixedLayoutBuilder.setFineCapacity): the first
 * generation with more h tiles must stop the run with the capacity fatal.
 * The header gives the box, the pool, the seed and the h tiles the
 * owner-indexed storage holds, then every buffer that capacity sizes with
 * its bytes, held and at one h tile a box tile. A row (every `every` steps
 * and the first three): step, resident pages, free blocks (those a commit
 * retired wait one commit before they count), the last commit's distinct
 * requests, h tiles (the solver's last mapped relayout receipt, so up to a
 * step behind), particles, the liquid in cells with its drift against the
 * sources, and the pool's fatal. The last lines give the peak of resident
 * pages, the least free blocks and the peak of h tiles, each with the step
 * that first reached it, then the resident pages and h tiles of every step.
 * The pool's status is copied after each step completes, never on the
 * advance path. A fatal of the pool or the solver prints and exits non-zero. */
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import assert from 'node:assert/strict';
import { createProcessRetainedDawnGPU } from '../lib/harness/node-dawn-provider';
import { managedGPUDevice } from '../lib/core/gpu-compilation-manager';
import { requiredFluidDeviceLimits } from '../lib/core/webgpu-device-limits';
import { sceneDocument } from '../lib/core/scene-definition';
import { findSceneDefinition } from '../lib/core/scenes';
import { uniformNarrowBandMethod } from '../lib/methods/uniform/uniform-narrow-band-method';
import { UNIFORM_PAGE_POOL_STATUS as S, uniformPagePoolFatalMessage, type UniformPagePool } from '../lib/methods/uniform/uniform-page-pool';
import type { WebGPUUniformReferenceSolver } from '../lib/methods/uniform/webgpu-uniform-reference';
const method=uniformNarrowBandMethod;
const [sceneId,stepsText,everyText]=process.argv.slice(2).filter(a=>!a.startsWith("--"));
const steps=Number(stepsText),every=Number(everyText);
assert.ok(sceneId&&Number.isInteger(steps)&&steps>=1&&steps<=2400&&Number.isInteger(every)&&every>=1,"usage: probe-nbflip-peaks.ts <scene> <steps> <every> [--dt=0.017] [--page-pool=<json>] [--fine-capacity=<tiles>]");
const dt=Number(process.argv.find(a=>a.startsWith("--dt="))?.split("=")[1]??0.017);
assert.ok(dt>0&&dt<=0.1);
const definition=findSceneDefinition(sceneId);
assert.ok(definition,`no scene named ${sceneId}`);
const scene=structuredClone(sceneDocument(definition));
scene.numerics={...scene.numerics,fixedDt_s:dt,maxDt_s:dt};
const pagePool=process.argv.find(a=>a.startsWith("--page-pool="))?.slice(12);
if(pagePool)scene.fluid.pagePool=JSON.parse(pagePool);
const fineCapacity=process.argv.find(a=>a.startsWith("--fine-capacity="))?.split("=")[1];
let device:GPUDevice|undefined,solver:WebGPUUniformReferenceSolver|undefined,read:GPUBuffer|undefined;
const errors:string[]=[];
try{
 const dawn=await import(pathToFileURL(resolve('node_modules/webgpu/index.js')).href);Object.assign(globalThis,dawn.globals);
 const gpu=createProcessRetainedDawnGPU(dawn,['backend=metal']);const adapter=await gpu.requestAdapter();assert.ok(adapter);
 device=managedGPUDevice(await adapter.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter.limits)}),{requireWorkerRealm:false});
 device.addEventListener('uncapturederror',e=>{e.preventDefault();errors.push(e.error.message);console.error(e.error.message);});
 solver=await method.createSolverAsync!(device,scene,'balanced',{...method.appDefaults,timeStep:'scene'},undefined,()=>{}) as WebGPUUniformReferenceSolver;
 const frame=(solver as unknown as {mixedFrame:{pagePool:UniformPagePool;fields:{residency:{live:boolean}};ownership:{capacity:{fineTiles:number;owners:number;address:{boxTiles:number}}};stageBandTiles:number;capacityBuffers(fineTiles:number,bandTiles:number):[string,number][]}}).mixedFrame;
 const {pagePool:pool,fields:{residency},ownership}=frame;
 const l=pool.layout,status=new Uint32Array(S.words);read=device.createBuffer({label:"Peak probe pool status",size:4*S.words,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
 console.log(`${sceneId}: box ${l.box.pages.join("x")} = ${l.box.count} pages (${ownership.capacity.address.boxTiles} tiles), pool of ${l.blocks} blocks (${l.slots} pages), ${residency.live?"live":"static"}, seeded ${pool.seeded.state[S.resident]} resident, ${pool.pinned} pinned, h capacity ${ownership.capacity.fineTiles} tiles (${ownership.capacity.owners} owners), allocated ${(solver.info.allocatedBytes/1048576).toFixed(1)} MiB, dt ${dt}`);
 {const held=frame.capacityBuffers(ownership.capacity.fineTiles,frame.stageBandTiles),box=frame.capacityBuffers(ownership.capacity.address.boxTiles,ownership.capacity.address.boxTiles),sum=(list:[string,number][])=>list.reduce((n,[,bytes])=>n+bytes,0);
  console.log(`capacity-sized buffers, bytes held | at every box tile: ${held.map(([label,bytes],i)=>`${label} ${bytes} | ${box[i]![1]}`).join("; ")}; total ${sum(held)} | ${sum(box)}`);}
 if(fineCapacity!==undefined){(solver as unknown as {mixedBuilder:{setFineCapacity(tiles:number):void}}).mixedBuilder.setFineCapacity(Number(fineCapacity));console.log(`builder h-tile bound forced to ${fineCapacity}`);}
 let residentPeak=0,residentStep=0,finePeak=0,fineStep=0,freeLeast=l.slots,freeStep=0;
 const residents:number[]=[],fines:number[]=[];
 for(let step=1;step<=steps;step++){
  let thrown:string|undefined;
  try{solver.advanceTo(step*dt,[]);await solver.awaitFrameCompletion();}catch(e){const cause=(e as Error).cause;thrown=`${(e as Error).message}${cause instanceof Error?`: ${cause.message}`:""}`;}
  const encoder=device.createCommandEncoder();pool.encodeReceipt(encoder,read,0);device.queue.submit([encoder.finish()]);
  await read.mapAsync(GPUMapMode.READ);status.set(new Uint32Array(read.getMappedRange()));read.unmap();
  const fatal=uniformPagePoolFatalMessage(l,status),failure=thrown??solver.info.simulationPipelineError;
  const resident=status[S.resident]!,free=status[S.free]!,fine=solver.info.uniformMixedFineTiles??0;
  residents.push(resident);fines.push(fine);
  if(resident>residentPeak){residentPeak=resident;residentStep=step;}
  if(free<freeLeast){freeLeast=free;freeStep=step;}
  if(fine>finePeak){finePeak=fine;fineStep=step;}
  if(step%every===0||step<=3||fatal||failure){
   const stats=failure?undefined:await solver.readStats();
   console.log(`step ${step} resident ${resident} free ${free} requests ${status[S.requests]} h tiles ${fine} particles ${solver.narrowBandFlipInfo?.particles??0} liquid ${Math.round(stats?.volumeCellSum??NaN)} drift ${(stats?.volumeDrift??NaN).toFixed(4)} fatal ${fatal??"none"}`);
  }
  if(failure||fatal){if(failure)console.log(`solver: ${failure}`);process.exitCode=1;break;}
  assert.ok(Math.abs((solver.info.completedTime_s??0)-step*dt)<1e-8,"complete exactly one requested clock step");
 }
 console.log(`peak resident ${residentPeak} pages at step ${residentStep} of ${l.slots}; least free ${freeLeast} blocks at step ${freeStep}; peak h tiles ${finePeak} at step ${fineStep} (capacity ${ownership.capacity.fineTiles})`);
 console.log(`resident by step: ${residents.join(" ")}`);
 console.log(`h tiles by step: ${fines.join(" ")}`);
 console.log(`errors ${JSON.stringify(errors)}`);
}finally{read?.destroy();solver?.destroy();await device?.queue.onSubmittedWorkDone();device?.destroy();}
