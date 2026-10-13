/** Page pool probe for Narrow-band FLIP: advances a scene and prints, every
 * `every` steps, what the pool's commits did. Browser GPU must be off.
 * node --import tsx tools/probe-nbflip-pages.ts <scene> <steps> <every> [--dt=0.017] [--page-pool=<json>]
 * --page-pool={"blocks":16}: the scene's page pool record (SceneDescription.fluid.pagePool).
 * A row: step, the pool's generation (commits so far: the first step's kick
 * makes two), resident pages, the pages claimed and retired since the last
 * row, free blocks, the last commit's distinct requests, the certificate's
 * closure radius in tiles (the census's, before the pool's cap), the particle count
 * of the last receipt, the pages a sample's support reached at its last move
 * (those resident since the build + those a commit claimed; a failing row
 * lists the claimed ones with the commit that claimed them), the liquid in
 * cells with its drift against the sources (the solver's statistics, read at
 * the row) and the pool's fatal.
 * The status is copied from the pool's state after each step completes,
 * never on the advance path, so a row's claimed and retired miss only the
 * kick's commit of step 1. A fatal of the pool or the solver (FLIP's arrival
 * in a page without storage among them) prints and exits non-zero. */
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import assert from 'node:assert/strict';
import { createProcessRetainedDawnGPU } from '../lib/harness/node-dawn-provider';
import { managedGPUDevice } from '../lib/core/gpu-compilation-manager';
import { requiredFluidDeviceLimits } from '../lib/core/webgpu-device-limits';
import { sceneDocument } from '../lib/core/scene-definition';
import { getSceneDefinition } from '../lib/core/scenes';
import { uniformNarrowBandMethod } from '../lib/methods/uniform/uniform-narrow-band-method';
import { UNIFORM_PAGE_POOL_BLOCK, UNIFORM_PAGE_POOL_RECORD as R, UNIFORM_PAGE_POOL_STATUS as S, uniformPagePoolFatalMessage, type UniformPagePool } from '../lib/methods/uniform/uniform-page-pool';
import type { WebGPUUniformReferenceSolver } from '../lib/methods/uniform/webgpu-uniform-reference';
const method=uniformNarrowBandMethod;
const [sceneId,stepsText,everyText]=process.argv.slice(2).filter(a=>!a.startsWith("--"));
const steps=Number(stepsText),every=Number(everyText);
assert.ok(sceneId&&Number.isInteger(steps)&&steps>=1&&steps<=2400&&Number.isInteger(every)&&every>=1,"usage: probe-nbflip-pages.ts <scene> <steps> <every> [--dt=0.017] [--page-pool=<json>]");
const dt=Number(process.argv.find(a=>a.startsWith("--dt="))?.split("=")[1]??0.017);
assert.ok(dt>0&&dt<=0.1);
const scene=structuredClone(sceneDocument(getSceneDefinition(sceneId)));
scene.numerics={...scene.numerics,fixedDt_s:dt,maxDt_s:dt};
const pagePool=process.argv.find(a=>a.startsWith("--page-pool="))?.slice(12);
if(pagePool)scene.fluid.pagePool=JSON.parse(pagePool);
let device:GPUDevice|undefined,solver:WebGPUUniformReferenceSolver|undefined,read:GPUBuffer|undefined;
const errors:string[]=[];
try{
 const dawn=await import(pathToFileURL(resolve('node_modules/webgpu/index.js')).href);Object.assign(globalThis,dawn.globals);
 const gpu=createProcessRetainedDawnGPU(dawn,['backend=metal']);const adapter=await gpu.requestAdapter();assert.ok(adapter);
 device=managedGPUDevice(await adapter.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter.limits)}),{requireWorkerRealm:false});
 device.addEventListener('uncapturederror',e=>{e.preventDefault();errors.push(e.error.message);console.error(e.error.message);});
 solver=await method.createSolverAsync!(device,scene,'balanced',{...method.appDefaults,timeStep:'scene'},undefined,()=>{}) as WebGPUUniformReferenceSolver;
 const {pagePool:pool,fields:{residency},narrowBandFlip:{particleTiles},ownership}=(solver as unknown as {mixedFrame:{pagePool:UniformPagePool;fields:{residency:{live:boolean}};narrowBandFlip:{particleTiles:{buffer:GPUBuffer;offset:number}};ownership:{support:GPUBuffer;capacity:{tiles:number}}}}).mixedFrame;
 const l=pool.layout,status=new Uint32Array(S.words);read=device.createBuffer({label:"Page pool probe status",size:4*S.words,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
 console.log(`${sceneId}: box ${l.box.pages.join("x")} = ${l.box.count} pages, pool of ${l.blocks} blocks (${l.slots} pages), ${residency.live?"live":"static"}, seeded ${pool.seeded.state[S.resident]} resident, ${pool.pinned} pinned, idle ${l.idleSteps}, dt ${dt}`);
 // The blocks whose tiles a sample's support reached, by the commit that claimed each (0: resident since the build).
 const records=R.words*l.blocks,marks=device.createBuffer({label:"Page pool probe liquid pages",size:4*(records+512*l.blocks)+16,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
 const liquidPages=async()=>{
  const encoder=device!.createCommandEncoder();encoder.copyBufferToBuffer(pool.state,4*l.state.records,marks,0,4*records);encoder.copyBufferToBuffer(particleTiles.buffer,4*particleTiles.offset,marks,4*records,4*512*l.blocks);
  // The residency certificate's header (umSupport[UM_RESIDENCY..]): pages, the closure bit, the radius.
  encoder.copyBufferToBuffer(ownership.support,4*(9*ownership.capacity.tiles+28),marks,4*(records+512*l.blocks),16);device!.queue.submit([encoder.finish()]);
  await marks.mapAsync(GPUMapMode.READ);const words=new Uint32Array(marks.getMappedRange()),pages=new Int32Array(words.buffer,words.byteOffset,records),since:string[]=[];let seeded=0;
  for(let b=1;b<l.blocks-1;b++){
   if(words[R.words*b+R.state]!==UNIFORM_PAGE_POOL_BLOCK.resident)continue;
   let wet=false;for(let t=0;t<512&&!wet;t++)wet=words[records+512*b+t]!==0;
   if(!wet)continue;
   if(words[R.words*b+R.generation]===0)seeded++;else since.push(`${pages[R.words*b]}.${pages[R.words*b+1]}.${pages[R.words*b+2]}@${words[R.words*b+R.generation]}`);
  }
  const reach=words[records+512*l.blocks+2]!;marks.unmap();return {seeded,since,reach};
 };
 let claimed=0,retired=0;
 for(let step=1;step<=steps;step++){
  let thrown:string|undefined;
  try{solver.advanceTo(step*dt,[]);await solver.awaitFrameCompletion();}catch(e){const cause=(e as Error).cause;thrown=`${(e as Error).message}${cause instanceof Error?`: ${cause.message}`:""}`;}
  const encoder=device.createCommandEncoder();pool.encodeReceipt(encoder,read,0);device.queue.submit([encoder.finish()]);
  await read.mapAsync(GPUMapMode.READ);status.set(new Uint32Array(read.getMappedRange()));read.unmap();
  claimed+=status[S.claimed]!;retired+=status[S.pending]!;
  const fatal=uniformPagePoolFatalMessage(l,status),failure=thrown??solver.info.simulationPipelineError;
  if(step%every===0||step===1||fatal||failure){
   const stats=failure?undefined:await solver.readStats(),wet=await liquidPages();
   console.log(`step ${step} generation ${status[S.generation]} resident ${status[S.resident]} claimed ${claimed} retired ${retired} free ${status[S.free]} requests ${status[S.requests]} reach ${wet.reach} particles ${solver.narrowBandFlipInfo?.particles??0} sample pages ${wet.seeded}+${wet.since.length} liquid ${Math.round(stats?.volumeCellSum??NaN)} drift ${(stats?.volumeDrift??NaN).toFixed(4)} fatal ${fatal??"none"}`);
   if(failure||fatal)console.log(`sample pages claimed by a commit (page@commit): ${wet.since.join(" ")||"none"}`);
   claimed=0;retired=0;
  }
  if(failure||fatal){if(failure)console.log(`solver: ${failure}`);process.exitCode=1;break;}
  assert.ok(Math.abs((solver.info.completedTime_s??0)-step*dt)<1e-8,"complete exactly one requested clock step");
 }
 console.log(`errors ${JSON.stringify(errors)}`);
}finally{read?.destroy();solver?.destroy();await device?.queue.onSubmittedWorkDone();device?.destroy();}
