/** Page pool probe for Narrow-band FLIP: advances a scene and prints, every
 * `every` steps, what the pool's commits did. Browser GPU must be off.
 * node --import tsx tools/probe-nbflip-pages.ts <scene> <steps> <every> [--dt=0.017] [--page-pool=<json>]
 * --page-pool={"blocks":16}: the scene's page pool record (SceneDescription.fluid.pagePool).
 * --allocation-audit: compare global/local-box/local-exact request sets.
 * --retirement-audit: also report each page's idle age, census reasons,
 * particle occupancy, fine tiles and solid/open owners with negative phi.
 * Census and particle masks describe the frame head; phi describes its end.
 * --values=<json>: method overrides (for example {"particleTransfer":"apic"}).
 * --subgroups: use the subgroup kernels when the adapter supports them.
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
import type { UniformMixedOwnership } from '../lib/methods/uniform/uniform-mixed-ownership';
import { uniformMixedTopologyWGSL } from '../lib/methods/uniform/uniform-mixed-topology.wgsl';
import { uniformMixedSolidWGSL, type UniformMixedSolid } from '../lib/methods/uniform/uniform-mixed-solid.wgsl';
import { uniformDetailBindLayout, uniformDetailModule, uniformDetailPipeline, uniformDetailPick, uniformDetailGroup } from '../lib/methods/uniform/uniform-detail-fields';
const method=uniformNarrowBandMethod;
const [sceneId,stepsText,everyText]=process.argv.slice(2).filter(a=>!a.startsWith("--"));
const steps=Number(stepsText),every=Number(everyText);
assert.ok(sceneId&&Number.isInteger(steps)&&steps>=1&&steps<=2400&&Number.isInteger(every)&&every>=1,"usage: probe-nbflip-pages.ts <scene> <steps> <every> [--dt=0.017] [--page-pool=<json>]");
const auditAllocation=process.argv.includes("--allocation-audit");
const auditRetirement=process.argv.includes("--retirement-audit");
const values=JSON.parse(process.argv.find(a=>a.startsWith("--values="))?.slice(9)??"{}");
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
 device=managedGPUDevice(await adapter.requestDevice({requiredFeatures:process.argv.includes("--subgroups")&&adapter.features.has("subgroups")?["subgroups"]:[],requiredLimits:requiredFluidDeviceLimits(adapter.limits)}),{requireWorkerRealm:false});
 device.addEventListener('uncapturederror',e=>{e.preventDefault();errors.push(e.error.message);console.error(e.error.message);});
 solver=await method.createSolverAsync!(device,scene,'balanced',{...method.appDefaults,...values,timeStep:'scene'},undefined,()=>{}) as WebGPUUniformReferenceSolver;
 const {pagePool:pool,fields:{residency},narrowBandFlip:{particleTiles},ownership}=(solver as unknown as {mixedFrame:{pagePool:UniformPagePool;fields:{residency:{live:boolean}};narrowBandFlip:{particleTiles:{buffer:GPUBuffer;offset:number}};ownership:{support:GPUBuffer;capacity:{tiles:number}}}}).mixedFrame;
 const l=pool.layout,status=new Uint32Array(S.words);read=device.createBuffer({label:"Page pool probe status",size:4*(S.words+4),usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
 console.log(`${sceneId}: box ${l.box.pages.join("x")} = ${l.box.count} pages, pool of ${l.blocks} blocks (${l.slots} pages), ${residency.live?"live":"static"}, seeded ${pool.seeded.state[S.resident]} resident, ${pool.pinned} pinned, idle ${l.idleSteps}, dt ${dt}`);
 // Read phi through the production detail-field bindings; raw texture copies
 // would confuse coarse base values with the fine pages' canonical values.
 const phiProbe=await (async()=>{
  if(!auditRetirement)return undefined;
  const s=solver as unknown as {device:GPUDevice;mixedFrame:{ownership:UniformMixedOwnership;solid:UniformMixedSolid;fields:{phi:GPUTexture}}};
  const d=s.device,o=s.mixedFrame.ownership,n=o.capacity.tiles,solid=s.mixedFrame.solid;
  const output=d.createBuffer({size:32*n,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
  const staging=d.createBuffer({size:output.size,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  const bindings=uniformDetailBindLayout(d,{entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float",viewDimension:"3d"}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}}]});
  const module=uniformDetailModule(d,{code:uniformMixedTopologyWGSL(o.capacity,0)+`
@group(1) @binding(0) var phi:texture_3d<f32>;
@group(1) @binding(1) var<storage,read_write> output:array<vec4f>;
${uniformMixedSolidWGSL(2,solid.coarse?.count)}
@compute @workgroup_size(64) fn measure(@builtin(global_invocation_id) id:vec3u){
 let t=id.x;if(t>=UM_TILES){return;}output[2u*t]=vec4f(0);output[2u*t+1u]=vec4f(0);if(!umTileLive(t)){return;}
 let width=umTileWidth(t);let side=4u/width;let origin=umTileCoord(t)*4u;
 var low=3e38;var high=-3e38;var solidWet=0u;var openWet=0u;var openLow=3e38;
 for(var owner=0u;owner<side*side*side;owner++){
  let cell=vec3i(origin+width*umCorner(owner,side));
  let open=select(umTileOpen(t),umCellOpen(cell),width==1u);var negative=false;
  for(var k=0u;k<8u;k++){
  let p=cell+vec3i(width*umCorner(k,2u));var v=0.0;
  if(width==4u){v=/*4h*/textureLoad(phi,p,0).x;}else{v=textureLoad(phi,p,0).x;}
  low=min(low,v);high=max(high,v);negative=negative||v<0.0;if(open>0.0){openLow=min(openLow,v);}
  }
  if(negative){if(open>0.0){openWet++;}else{solidWet++;}}
 }
 output[2u*t]=vec4f(low,high,f32(width),0);output[2u*t+1u]=vec4f(f32(solidWet),f32(openWet),openLow,0);
}`});
  const pipeline=await uniformDetailPipeline(d,o,{layout:d.createPipelineLayout({bindGroupLayouts:[o.bindLayout,bindings,solid.tileLayout]}),compute:{module,entryPoint:"measure",constants:{umDispatchX:65535}}});
  const group=uniformDetailGroup(d,{layout:bindings,entries:[{binding:0,resource:s.mixedFrame.fields.phi},{binding:1,resource:{buffer:output}}]});
  return {async read(){const e=d.createCommandEncoder(),p=e.beginComputePass();p.setPipeline(uniformDetailPick(pipeline));p.setBindGroup(0,o.bindGroup);p.setBindGroup(1,group.group);p.setBindGroup(2,solid.tileGroup);p.dispatchWorkgroups(Math.ceil(n/64));p.end();e.copyBufferToBuffer(output,0,staging,0,output.size);d.queue.submit([e.finish()]);await staging.mapAsync(GPUMapMode.READ);const values=new Float32Array(staging.getMappedRange().slice(0));staging.unmap();return values;},destroy(){output.destroy();staging.destroy();}};
 })();
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
  if(!auditAllocation&&!auditRetirement)return "";
  const dynamic=(solver as unknown as {mixedDynamic:{work:GPUBuffer;localOffset:number;nearOffset:number;wetOffset:number;bandBits:{wordOffset:number};particles:{allocation:GPUBuffer}}}).mixedDynamic;
  if(!Number.isInteger(dynamic.localOffset))return "";
  const censusRead=device!.createBuffer({size:dynamic.work.size,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  const stateRead=device!.createBuffer({size:pool.state.size,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  const particleRead=device!.createBuffer({size:dynamic.particles.allocation.size,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  const e=device!.createCommandEncoder();e.copyBufferToBuffer(dynamic.work,0,censusRead,0,dynamic.work.size);e.copyBufferToBuffer(pool.state,0,stateRead,0,pool.state.size);e.copyBufferToBuffer(dynamic.particles.allocation,0,particleRead,0,particleRead.size);device!.queue.submit([e.finish()]);
  await Promise.all([censusRead.mapAsync(GPUMapMode.READ),stateRead.mapAsync(GPUMapMode.READ),particleRead.mapAsync(GPUMapMode.READ)]);
  const c=new Uint32Array(censusRead.getMappedRange()),state=new Uint32Array(stateRead.getMappedRange()),signed=new Int32Array(state.buffer);
  if(auditRetirement){
   const phi=await phiProbe!.read();
   const particle=new Uint32Array(particleRead.getMappedRange());
   const bit=(offset:number,t:number)=>(c[offset+(t>>>5)]!>>>(t&31))&1;
   const pages=[];
   for(let b=1;b<l.blocks-1;b++){
    const r=l.state.records+R.words*b;if(state[r+R.state]!==UNIFORM_PAGE_POOL_BLOCK.resident)continue;
    let near=0,wet=0,band=0,sample=0,fine=0,low=Infinity,high=-Infinity,solidWet=0,openWet=0,openLow=Infinity;const regions=[];
    for(let o=0;o<8;o++){
     const page=8*b+o,at=dynamic.localOffset+4*page;
     if(c[at]||c[at+1])regions.push({octant:o,negative:[0,1,2].map(k=>(c[at+2]!>>>(8*k))&255),positive:[0,1,2].map(k=>(c[at+3]!>>>(8*k))&255)});
     for(let lane=0;lane<64;lane++){
      const t=512*b+((4*(o&1)+(lane&3))|((4*((o>>>1)&1)+((lane>>>2)&3))<<3)|((4*((o>>>2)&1)+(lane>>>4))<<6));
      near+=bit(dynamic.nearOffset,t);wet+=bit(dynamic.wetOffset,t);band+=bit(dynamic.bandBits.wordOffset,t);
      sample+=(particle[8*page+(lane>>>5)]!>>>(lane&31))&1;
      if(phi[8*t+2]){fine+=Number(phi[8*t+2]===1);low=Math.min(low,phi[8*t]!);high=Math.max(high,phi[8*t+1]!);solidWet+=phi[8*t+4]!;openWet+=phi[8*t+5]!;openLow=Math.min(openLow,phi[8*t+6]!);}
     }
    }
    pages.push({page:Array.from(signed.subarray(r,r+3)),idle:state[r+R.idle],near_tiles:near,wet_tiles:wet,band_tiles:band,particle_tiles:sample,fine_tiles:fine,phi_min_m:low,phi_max_m:high,negative_phi_solid_owners:solidWet,negative_phi_open_owners:openWet,open_phi_min_m:openLow,regions});
   }
   console.log(JSON.stringify({retirement:pages}));
  }
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
  censusRead.unmap();stateRead.unmap();particleRead.unmap();censusRead.destroy();stateRead.destroy();particleRead.destroy();
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
 phiProbe?.destroy();
}finally{read?.destroy();solver?.destroy();await device?.queue.onSubmittedWorkDone();device?.destroy();}
