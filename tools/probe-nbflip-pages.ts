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
const auditAllocation=process.argv.includes("--allocation-audit");
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
 const l=pool.layout,status=new Uint32Array(S.words);read=device.createBuffer({label:"Page pool probe status",size:4*(S.words+4),usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
 console.log(`${sceneId}: box ${l.box.pages.join("x")} = ${l.box.count} pages, pool of ${l.blocks} blocks (${l.slots} pages), ${residency.live?"live":"static"}, seeded ${pool.seeded.state[S.resident]} resident, ${pool.pinned} pinned, idle ${l.idleSteps}, dt ${dt}`);
 // The blocks whose tiles a sample's support reached, by the commit that claimed each (0: resident since the build).
 const records=R.words*l.blocks,marks=device.createBuffer({label:"Page pool probe liquid pages",size:4*(records+512*l.blocks)+16,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
 const liquidPages=async()=>{
  const encoder=device!.createCommandEncoder();encoder.copyBufferToBuffer(pool.state,4*l.state.records,marks,0,4*records);encoder.copyBufferToBuffer(particleTiles.buffer,4*particleTiles.offset,marks,4*records,4*512*l.blocks);
  // The residency certificate's header (umSupport[UM_RESIDENCY..]): pages, the closure bit, the radius.
  encoder.copyBufferToBuffer(ownership.support,4*(9*ownership.capacity.tiles+28),marks,4*(records+512*l.blocks),16);device!.queue.submit([encoder.finish()]);
  await marks.mapAsync(GPUMapMode.READ);const words=new Uint32Array(marks.getMappedRange()),pages=new Int32Array(words.buffer,words.byteOffset,records),since:string[]=[];let seeded=0,idle=0;
  for(let b=1;b<l.blocks-1;b++){
   if(words[R.words*b+R.state]!==UNIFORM_PAGE_POOL_BLOCK.resident)continue;
   if(words[R.words*b+R.idle]!>0)idle++;
   let wet=false;for(let t=0;t<512&&!wet;t++)wet=words[records+512*b+t]!==0;
   if(!wet)continue;
   if(words[R.words*b+R.generation]===0)seeded++;else since.push(`${pages[R.words*b]}.${pages[R.words*b+1]}.${pages[R.words*b+2]}@${words[R.words*b+R.generation]}`);
  }
  const reach=words[records+512*l.blocks+2]!;marks.unmap();return {seeded,since,reach,idle};
 };
 // Counterfactuals on the SAME census: global cube / local directional box /
 // local exact occupancy. Numerical evolution therefore cannot explain the difference.
 const allocation=async()=>{
  if(!auditAllocation)return "";
  const dynamic=(solver as unknown as {mixedDynamic:{work:GPUBuffer;localOffset:number}}).mixedDynamic;
  if(!Number.isInteger(dynamic.localOffset))return "";
  const censusRead=device!.createBuffer({size:dynamic.work.size,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  const stateRead=device!.createBuffer({size:pool.state.size,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  const e=device!.createCommandEncoder();e.copyBufferToBuffer(dynamic.work,0,censusRead,0,dynamic.work.size);e.copyBufferToBuffer(pool.state,0,stateRead,0,pool.state.size);device!.queue.submit([e.finish()]);
  await Promise.all([censusRead.mapAsync(GPUMapMode.READ),stateRead.mapAsync(GPUMapMode.READ)]);
  const c=new Uint32Array(censusRead.getMappedRange()),state=new Uint32Array(stateRead.getMappedRange()),signed=new Int32Array(state.buffer);
  const globalRadius=Math.ceil(new Float32Array(c.buffer)[18]!/4)+5,sets=[new Set<string>(),new Set<string>(),new Set<string>()];
  const solid=(q:number[])=>q.some((v,k)=>v<signed[l.state.window+24+k]!||v>=signed[l.state.window+27+k]!);
  const add=(set:Set<string>,lo:number[],hi:number[])=>{for(let z=Math.floor(lo[2]!/8);z<=Math.floor(hi[2]!/8);z++)for(let y=Math.floor(lo[1]!/8);y<=Math.floor(hi[1]!/8);y++)for(let x=Math.floor(lo[0]!/8);x<=Math.floor(hi[0]!/8);x++){if(!solid([x,y,z]))set.add(`${x},${y},${z}`);}};
  for(let page=0;page<8*l.blocks;page++){
   const at=dynamic.localOffset+4*page,m0=c[at]!,m1=c[at+1]!;if((m0|m1)===0)continue;
   const r=l.state.records+R.words*(page>>>3),origin=[0,1,2].map(k=>8*signed[r+k]!+4*((page>>>k)&1));
   const lo=[3,3,3],hi=[0,0,0],negative=[0,1,2].map(k=>(c[at+2]!>>>(8*k))&255),positive=[0,1,2].map(k=>(c[at+3]!>>>(8*k))&255);
   for(let bit=0;bit<64;bit++){
    if((((bit<32?m0:m1)>>>(bit&31))&1)===0)continue;
    const q=[bit&3,(bit>>>2)&3,bit>>>4];for(let k=0;k<3;k++){lo[k]=Math.min(lo[k]!,q[k]!);hi[k]=Math.max(hi[k]!,q[k]!);}
    add(sets[2]!,q.map((v,k)=>origin[k]!+v-negative[k]!),q.map((v,k)=>origin[k]!+v+positive[k]!));
   }
   add(sets[0]!,lo.map((v,k)=>origin[k]!+v-globalRadius),hi.map((v,k)=>origin[k]!+v+globalRadius));
   add(sets[1]!,lo.map((v,k)=>origin[k]!+v-negative[k]!),hi.map((v,k)=>origin[k]!+v+positive[k]!));
  }
  censusRead.unmap();stateRead.unmap();censusRead.destroy();stateRead.destroy();
  return ` candidates global/local-box/local-exact ${sets.map(s=>s.size).join("/")}`;
 };
 let claimed=0,retired=0,residentSum=0,peakResident=0,maxSupport=0;
 let completed=0;

 for(let step=1;step<=steps;step++){
  let thrown:string|undefined;
  try{solver.advanceTo(step*dt,[]);await solver.awaitFrameCompletion();}catch(e){const cause=(e as Error).cause;thrown=`${(e as Error).message}${cause instanceof Error?`: ${cause.message}`:""}`;}
  const encoder=device.createCommandEncoder();pool.encodeReceipt(encoder,read,0);encoder.copyBufferToBuffer(ownership.support,4*(9*ownership.capacity.tiles+28),read,4*S.words,16);device.queue.submit([encoder.finish()]);
  await read.mapAsync(GPUMapMode.READ);const receipt=new Uint32Array(read.getMappedRange());status.set(receipt.subarray(0,S.words));const support=receipt[S.words+1]!;maxSupport|=support;read.unmap();
  residentSum+=status[S.resident]!;peakResident=Math.max(peakResident,status[S.resident]!);completed=step;
  claimed+=status[S.claimed]!;retired+=status[S.pending]!;
  const fatal=uniformPagePoolFatalMessage(l,status),failure=thrown??solver.info.simulationPipelineError??(support?`support violation bits ${support}`:undefined);
  if(step%every===0||step===1||fatal||failure){
   const stats=failure?undefined:await solver.readStats(),wet=await liquidPages();
   console.log(`step ${step} generation ${status[S.generation]} resident ${status[S.resident]} claimed ${claimed} retired ${retired} free ${status[S.free]} requests ${status[S.requests]} reach ${wet.reach} idle ${wet.idle} support ${support} particles ${solver.narrowBandFlipInfo?.particles??0} sample pages ${wet.seeded}+${wet.since.length} liquid ${Math.round(stats?.volumeCellSum??NaN)} drift ${(stats?.volumeDrift??NaN).toFixed(4)} fatal ${fatal??"none"}${await allocation()}`);
   if(failure||fatal)console.log(`sample pages claimed by a commit (page@commit): ${wet.since.join(" ")||"none"}`);
   claimed=0;retired=0;
  }
  if(failure||fatal){if(failure)console.log(`solver: ${failure}`);process.exitCode=1;break;}
  assert.ok(Math.abs((solver.info.completedTime_s??0)-step*dt)<1e-8,"complete exactly one requested clock step");
 }
 console.log(`summary steps ${completed} peak ${peakResident} mean ${(residentSum/completed).toFixed(3)} support ${maxSupport}`);
 console.log(`errors ${JSON.stringify(errors)}`);assert.deepEqual(errors,[]);
}finally{read?.destroy();solver?.destroy();await device?.queue.onSubmittedWorkDone();device?.destroy();}
