/** Ferstl et al. 2016, Figure 8 "Letters" on Uniform Narrow-band FLIP, measured
 * the way the paper's Table 1 reports it: time per step split into the
 * projection and the rest, average particle count and memory. Browser GPU must
 * be off.
 *   node --import tsx tools/benchmark-nbflip-letters.ts [name]
 * --steps=N runs fewer than the take's 193 frames (a smoke run).
 * --track reads the level set after every step and records the top of each
 *   letter as it falls. It is a readback per step, so a tracked run's wall
 *   times are not the benchmark's; pass timestamps are unaffected.
 * --out=DIR writes somewhere other than docs/verification.
 * --set=key=value overrides one method value over the scene's profile, whose
 *   coverage is the method's default (dynamic, surface tiles at h);
 *   --set=detailPolicy=full is the published grid everywhere. Repeatable.
 * --from=N averages the summary over frames N.. only (rows keep every frame).
 * --plain takes no pass timestamps: wall is then the frame time the app pays,
 *   without the per-pass timestamp writes and the encoder proxy that reads them.
 * --transfer-probe replays diagnostic fine-P2G ablations at frames 40/100/180.
 *   Production transfer runs last to restore its output. Wall times include
 *   probes and are not normal scene timings; transferStage holds each replay.
 * --probe-at=120 selects probe frames (default 40,100,180); 120 is 5 s.
 * --sequence keeps each row's passes in order with the gap before each one.
 * --app-trace (with --plain) records the stage trace the app's SIM panel
 *   shows instead of pass timestamps: each row's appPhases is that frame's
 *   boundary-to-boundary time per stage label. Frames are paced to the trace's
 *   cadence, so wall times are not the benchmark's.
 * --pace=MS idles that long before every step (default 110 under --app-trace,
 *   else 0): the GPU a paced step starts on has been idle, as the app's is
 *   whenever a step finishes inside its playback interval. Under --pace=0 the
 *   app trace samples only the steps its cadence admits.
 * --dt=S steps at another interval than the take's 1/24 s (the app's transport
 *   can: 0.0166667 is its 60 Hz), with --steps counted in those steps.
 * --host-work counts what the host asks of the device inside each step's
 *   synchronous advance (objects created, queue writes and their bytes,
 *   submits): a row's hostWork. A step that changes nothing should create
 *   nothing.
 * --layout-at=N[,N] reads the tile ownership after those steps (a readback,
 *   outside the step's timing) into the row's layout: how the h tiles sit
 *   among the 4h ones (interior, seam, isolated, holes, connected pieces, what a
 *   one-tile closing would add), why (the census's flags; how the budget's
 *   admitted set moves between consecutive reads) and the slice holding most
 *   of them: # admitted, c h and still cooling, + h in a hot tile's collar,
 *   - h with no heat, . 4h. Also the h tiles with liquid and no surface
 *   (submerged) or neither (air): what holds each, and how long since a
 *   surface was last in it.
 * --reasons (with --layout-at) records the census's band reasons and tallies
 *   the rule behind each h tile. It turns the layout views on, which scores
 *   every criterion: read the split, not the timings.
 * --marks (with --layout-at on consecutive steps) reads the samples and the
 *   nearest-crossing bank back and runs the refinement kernel on the host: the
 *   tiles the next census joins, by the kind of sample marking each and by
 *   which term of its box reaches it (the surface cell, travel, padding);
 *   whether the tiles held for travel alone then received the surface; and
 *   what tighter boxes would release and miss.
 * --render-load=MS submits an offscreen full-screen fragment pass of about
 *   that cost on the same queue ahead of every step, and does not wait for it:
 *   the step's head then starts behind independent render work, as the app's
 *   does behind its presentation. The row's renderLoad_ms is the pass alone.
 * Every compute pass is timestamped on every step. A pass is projection when
 * its label names the pressure solve; the per-label totals are in the output
 * so that split can be audited. Wall is advance plus completion of one step
 * and excludes the timestamp readback; encode is the host's share of it, the
 * synchronous advance that encodes and submits the step. Between is the GPU
 * time outside every pass, the copies and clears encoded between two of them;
 * a row's `before` attributes it to the pass that follows. */
import { writeFileSync, mkdirSync } from 'node:fs';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import assert from 'node:assert/strict';
import { GPUPassProfile } from './gpu-pass-profile';
import { NarrowBandTransferProbe } from './narrow-band-transfer-probe';
import { NarrowBandDispatchProbe } from './narrow-band-dispatch-probe';
import { createProcessRetainedDawnGPU } from '../lib/harness/node-dawn-provider';
import { managedGPUDevice } from '../lib/core/gpu-compilation-manager';
import { requiredFluidDeviceLimits } from '../lib/core/webgpu-device-limits';
import { resolveMethodValues } from '../lib/core/method-contract';
import { sceneDocument } from '../lib/core/scene-definition';
import { getSceneDefinition } from '../lib/core/scenes';
import type { LiquidExtrusion } from '../lib/core/liquid-extrusion';
import { NBFLIP_LETTERS_FRAMES, NBFLIP_LETTERS_GRID, NBFLIP_LETTERS_SCENE_ID, NBFLIP_LETTERS_TIME_STEP_S } from '../lib/core/nbflip-paper-scenes';
import { uniformNarrowBandMethod } from '../lib/methods/uniform/uniform-narrow-band-method';
import type { WebGPUUniformReferenceSolver } from '../lib/methods/uniform/webgpu-uniform-reference';
import { readUniformFields } from '../tests/helpers/uniform-geometric';
import { GPUStageTimestampRecorder } from '../lib/core/performance-trace';
import { UNIFORM_STAGE_REASON } from '../lib/methods/uniform/uniform-stage-grids';
import { usePerformanceInstrumentationStore } from '../lib/core/stores/performance-instrumentation-store';

const option=(key:string)=>process.argv.find(a=>a.startsWith(`--${key}=`))?.slice(key.length+3);
const steps=Number(option("steps")??NBFLIP_LETTERS_FRAMES),track=process.argv.includes("--track"),plain=process.argv.includes("--plain"),out=option("out")??'docs/verification';
// --transfer-probe replays diagnostic shader ablations at frames 40/100/180.
// It restores production output before continuing; wall times include probes.
const probeTransfer=process.argv.includes("--transfer-probe");
const probeDispatch=process.argv.includes("--dispatch-probe");
const sequence=process.argv.includes("--sequence");
const recordReasons=process.argv.includes("--reasons"),recordMarks=process.argv.includes("--marks"),recordBandReaders=process.argv.includes("--band-readers");
const renderLoad_ms=Number(option("render-load")??0),hostWork=process.argv.includes("--host-work");
const layoutAt=(option("layout-at")??"").split(",").filter(Boolean).map(Number);
const warmAt=(option("warm-at")??"").split(",").filter(Boolean).map(Number);
const appTrace=process.argv.includes("--app-trace"),pace=Number(option("pace")??(appTrace?110:0));
assert.ok(!appTrace||plain,"The app's stage trace and pass timestamps cannot share a pass: --app-trace needs --plain");
// Select simulation frames for focused diagnostics, e.g. frame 120 is 5 s.
const probeFrames=(option("probe-at")??"40,100,180").split(",").map(Number);
assert.ok(probeFrames.every(n=>Number.isInteger(n)&&n>=1&&(!option("probe-at")||n<=steps)),"Probe frames must be within the take");
assert.ok(!probeTransfer||!plain,"Transfer probes require pass timestamps");
assert.ok(!probeDispatch||!plain,"Dispatch probes require pass timestamps");
const stepInterval=Number(option("dt")??NBFLIP_LETTERS_TIME_STEP_S);
assert.ok(stepInterval>0&&stepInterval<=NBFLIP_LETTERS_TIME_STEP_S,"--dt is a step no longer than the take's");
assert.ok(Number.isInteger(steps)&&steps>=1&&steps*stepInterval<=NBFLIP_LETTERS_FRAMES*NBFLIP_LETTERS_TIME_STEP_S+1e-9,`--steps must stay inside the take's ${NBFLIP_LETTERS_FRAMES} frames`);
const name=process.argv.slice(2).find(a=>!a.startsWith("--"))??'current';
assert.match(name,/^[a-z0-9-]+$/);
const PROJECTION=/pressure|projection/i;
const definition=getSceneDefinition(NBFLIP_LETTERS_SCENE_ID),scene=structuredClone(sceneDocument(definition)),dt=stepInterval;
if(option("dt"))scene.numerics={...scene.numerics,fixedDt_s:dt,maxDt_s:dt};
assert.equal(definition.methodProfile?.methodId,uniformNarrowBandMethod.id);
// The values the app resolves for this scene: the method's app defaults under the scene's own overrides.
const sets=Object.fromEntries(process.argv.filter(a=>a.startsWith("--set=")).map(a=>{const [key,...value]=a.slice(6).split("=");const text=value.join("=");return [key!,text!==""&&Number.isFinite(Number(text))?Number(text):text==="true"?true:text==="false"?false:text];}));
const from=Number(option("from")??1);
const values=resolveMethodValues(uniformNarrowBandMethod,definition.methodProfile!.quality,{...uniformNarrowBandMethod.appDefaults,...definition.methodProfile!.overrides,...sets});
const [nx,ny,nz]=NBFLIP_LETTERS_GRID,h=scene.voxelDomain.finestCellSize_m;
// Each letter's highest outline point, as the vertex column under it and the height its liquid starts at.
const letters=scene.fluid.scheduledDrops!.map((drop,index)=>{
 const e=drop.volume as LiquidExtrusion,points=e.contours_m.flatMap(c=>c.flatMap((v,i)=>i%2?[]:[[v,c[i+1]!] as const]));
 const top=points.reduce((a,b)=>b[1]>a[1]?b:a);
 return {letter:"ABCDEFGHI"[index]!,step:Math.round(drop.time_s/dt),x:Math.round(top[0]/h+nx/2),z:Math.round(e.centerZ_m/h+nz/2),top0:(top[1]+e.offset_m)/h};
});
let device:GPUDevice|undefined,profile:GPUPassProfile|undefined,solver:WebGPUUniformReferenceSolver|undefined;
const rows:Record<string,unknown>[]=[],errors:string[]=[],labels=new Map<string,{ms:number;passes:number;dispatches:number}>();
const write=(file:string,body:object)=>{mkdirSync(out,{recursive:true});writeFileSync(`${out}/${file}`,JSON.stringify(body,null,2)+'\n');};
const mean=(values:number[])=>values.reduce((a,b)=>a+b,0)/values.length;
try {
 const dawn=await import(pathToFileURL(resolve('node_modules/webgpu/index.js')).href);Object.assign(globalThis,dawn.globals);
 const gpu=createProcessRetainedDawnGPU(dawn,['backend=metal']);const adapter=await gpu.requestAdapter();assert.ok(adapter);
 const raw=await adapter.requestDevice({requiredFeatures:['timestamp-query',...(adapter.features.has('subgroups')?['subgroups' as const]:[])],requiredLimits:requiredFluidDeviceLimits(adapter.limits)});
 // Counted on the device itself, under every wrapper the run stacks on it.
 let work:Record<string,number>={};
 if(hostWork){
  const count=(target:object,name:string,bytes?:(args:unknown[])=>number)=>{
   const original=(target as Record<string,(...args:unknown[])=>unknown>)[name]!;
   Object.defineProperty(target,name,{configurable:true,value(this:unknown,...args:unknown[]){work[name]=(work[name]??0)+1;if(bytes)work[`${name}Bytes`]=(work[`${name}Bytes`]??0)+bytes(args);return original.apply(this,args);}});
  };
  for(const name of ["createBindGroup","createBuffer","createTexture","createCommandEncoder","createComputePipeline","createShaderModule","createBindGroupLayout","createPipelineLayout","createQuerySet"])count(raw,name);
  const size=(data:unknown,elements?:number)=>ArrayBuffer.isView(data)?(elements===undefined?data.byteLength:elements*((data as unknown as {BYTES_PER_ELEMENT?:number}).BYTES_PER_ELEMENT??1)):(data as ArrayBuffer).byteLength;
  count(raw.queue,"writeBuffer",a=>size(a[2],a[4] as number|undefined));count(raw.queue,"writeTexture",a=>size(a[1]));count(raw.queue,"submit");
 }
 profile=new GPUPassProfile(raw);
 const dispatchProbe=probeDispatch?new NarrowBandDispatchProbe(profile.device):undefined;
 const transferProbe=probeTransfer?new NarrowBandTransferProbe(dispatchProbe?.device??profile.device):undefined;
 const selected=transferProbe?.device??dispatchProbe?.device??(plain?raw:profile.device);
 const spatialDevice=process.argv.includes("--spatial-grid")?new Proxy(selected,{get:(target,key)=>{
  if(key==="createShaderModule")return (d:GPUShaderModuleDescriptor)=>{
   if(d.label!=="Uniform mixed layout builder")return target.createShaderModule(d);
   const helper=`fn spatialTile(i:u32)->u32{
 if(any((T%vec3u(4u))!=vec3u(0u))){return i;}
 let block=i/64u;let b=vec3u(block%(T.x/4u),(block/(T.x/4u))%(T.y/4u),block/((T.x/4u)*(T.y/4u)));
 let k=i%64u;let l=vec3u((k&1u)|((k>>2u)&2u),((k>>1u)&1u)|((k>>3u)&2u),((k>>2u)&1u)|((k>>4u)&2u));
 return key(4u*b+l);
}`;
   let changed=0;const code=d.code.replace(/let t=group.x\*(\d+)u\+lane;/g,(_,block)=>{changed++;return `let t=spatialTile(group.x*${block}u+lane);`;})+helper;
   assert.equal(changed,2);return target.createShaderModule({...d,code});
  };
  const v=Reflect.get(target,key,target);return typeof v==="function"?v.bind(target):v;
 }}):selected;
 device=managedGPUDevice(spatialDevice,{requireWorkerRealm:false});
 device.addEventListener('uncapturederror',e=>{e.preventDefault();errors.push(e.error.message);console.error(e.error.message);});
 const built=performance.now();
 solver=await uniformNarrowBandMethod.createSolverAsync!(device,scene,definition.methodProfile!.quality,values,undefined,()=>{}) as WebGPUUniformReferenceSolver;
 if(recordReasons)solver.setLayoutViewsEnabled(true);
 const construction_ms=performance.now()-built;
 // Host cost of the frame's WebGPU calls: count and time per method, over the measured window.
 const calls:Record<string,{n:number;ms:number}>={};
 const passLabels=new WeakMap<object,string>();
 if(process.argv.includes("--count-calls")){
  const wrap=(proto:object,kind:string,names:string[])=>{for(const name of names){
   const f=(proto as Record<string,unknown>)[name];if(typeof f!=="function")continue;
   (proto as Record<string,unknown>)[name]=function(this:unknown,...a:unknown[]){const t=performance.now(),r=(f as (...a:unknown[])=>unknown).apply(this,a);const c=calls[`${kind}.${name}`]??={n:0,ms:0};c.n++;c.ms+=performance.now()-t;
    if(name==="beginComputePass")passLabels.set(r as object,(a[0] as {label?:string}|undefined)?.label??"?");
    else if(name==="dispatchWorkgroups"||name==="dispatchWorkgroupsIndirect"){const l=`dispatches ${passLabels.get(this as object)}`;(calls[l]??={n:0,ms:0}).n++;}
    return r;};
  }};
  const e=spatialDevice.createCommandEncoder(),pass=e.beginComputePass();
  wrap(Object.getPrototypeOf(pass),"pass",["setPipeline","setBindGroup","dispatchWorkgroups","dispatchWorkgroupsIndirect","end"]);
  wrap(Object.getPrototypeOf(e),"encoder",["beginComputePass","beginRenderPass","copyBufferToBuffer","copyTextureToTexture","copyBufferToTexture","copyTextureToBuffer","clearBuffer","finish","resolveQuerySet"]);
  pass.end();e.finish();
  wrap(Object.getPrototypeOf(spatialDevice.queue),"queue",["writeBuffer","writeTexture","submit","onSubmittedWorkDone"]);
  wrap(Object.getPrototypeOf(spatialDevice),"device",["createBindGroup","createBuffer","createTexture","createCommandEncoder","createComputePipeline","createShaderModule"]);
 }
 let probeFrame=0,tracedSample=0;
 // A stand-in for the app's presentation: nothing in it reads or writes a simulation resource.
 let renderLoad:(()=>void)|undefined,renderLoadAlone_ms=0;
 if(renderLoad_ms>0){
  const target=raw.createTexture({size:[1920,1080],format:"rgba8unorm",usage:GPUTextureUsage.RENDER_ATTACHMENT}).createView();
  const rounds=raw.createBuffer({size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
  const module=raw.createShaderModule({code:`@group(0) @binding(0) var<uniform> rounds:vec4u;
@vertex fn v(@builtin(vertex_index) i:u32)->@builtin(position) vec4f{return vec4f(f32(i32(i&1u)*4-1),f32(i32(i>>1u)*4-1),0,1);}
@fragment fn f(@builtin(position) p:vec4f)->@location(0) vec4f{var a=p.xy*0.001;for(var i=0u;i<rounds.x;i++){a=fract(sin(a.yx*12.9898+f32(i))*43758.5453);}return vec4f(a,0,1);}`});
  const pipeline=await raw.createRenderPipelineAsync({layout:"auto",vertex:{module,entryPoint:"v"},fragment:{module,entryPoint:"f",targets:[{format:"rgba8unorm"}]}});
  const group=raw.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:rounds}}]});
  const submit=()=>{const e=raw.createCommandEncoder();const pass=e.beginRenderPass({colorAttachments:[{view:target,loadOp:"clear",storeOp:"store"}]});
   pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.draw(3);pass.end();raw.queue.submit([e.finish()]);};
  const time=async(n:number)=>{raw.queue.writeBuffer(rounds,0,Uint32Array.of(n,0,0,0));let best=Infinity;
   for(let k=0;k<6;k++){const t=performance.now();submit();await raw.queue.onSubmittedWorkDone();best=Math.min(best,performance.now()-t);}return best;};
  let n=64;while(n<1<<20&&await time(n)<renderLoad_ms)n*=2;
  // Linear between the two bracketing counts.
  const low=await time(n/2),high=await time(n);n=Math.max(1,Math.round(n/2+(n/2)*(renderLoad_ms-low)/Math.max(1e-3,high-low)));
  renderLoadAlone_ms=await time(n);renderLoad=submit;
  console.log(JSON.stringify({renderLoadRounds:n,renderLoadAlone_ms}));
 }
 if(appTrace){await GPUStageTimestampRecorder.prepare(device);usePerformanceInstrumentationStore.getState().setEnabled(true);}
 if(transferProbe){
  const stage=(solver as unknown as {mixedFrame:{narrowBandFlip:{dispatchBatch(e:GPUCommandEncoder,entries:readonly string[],group:string|undefined,label:string,shared?:GPUComputePassEncoder):void}}}).mixedFrame.narrowBandFlip;
  const dispatch=stage.dispatchBatch.bind(stage);
  stage.dispatchBatch=(encoder,entries,group,label,shared)=>{
   if(!entries.includes("transfer")||!probeFrames.includes(probeFrame)){dispatch(encoder,entries,group,label,shared);return;}
   // Split the measured batch only for probes; every replay has identical
   // classified particles and pre-transfer velocity. Production runs last.
   for(const entry of entries){
    if(entry==="transfer"){
     for(let round=0;round<4;round++)for(const name of (round%2?[...transferProbe.names].reverse():transferProbe.names)){
      transferProbe.variant=name;dispatch(encoder,[entry],group,entry);
     }
     transferProbe.variant=undefined;
    }
    dispatch(encoder,[entry],group,entry);
   }
  };
 }
 assert.deepEqual([solver.info.nx,solver.info.ny,solver.info.nz],[nx,ny,nz],"the published grid");
 /** How the h tiles sit among the 4h ones: bit 31 of a tile's owner word is h. */
 const layoutHistory:{frame:number;required:Uint8Array;triggered:Uint8Array;crossing?:Uint8Array;predicted?:{full:Uint8Array;noPad:Uint8Array;noTravel:Uint8Array;bare:Uint8Array;ends:Uint8Array;faces:Uint8Array;point:Uint8Array;point2:Uint8Array;samples:Record<string,number>}}[]=[];
 // Would last frame's h band pressure start this frame's band solve nearer its
 // answer than the 4h start does? Read only: the presented band pressures and
 // the iterate (the correction to the 4h start) of two consecutive frames.
 let warmHeld:{slots:Uint32Array;total:Float32Array;correction:Float32Array}|undefined;
 const readWarm=async(compare:boolean)=>{
  const frame=(solver as unknown as {mixedFrame:{stageBandWord:number;stageBandTiles:number;presentation:{phi:{buffer:GPUBuffer}};band:{iterate:GPUBuffer};ownership:{capacity:{tiles:number;lattice:{dimensions:readonly number[]}}}}}).mixedFrame;
  const tiles=frame.ownership.capacity.tiles,cap=frame.stageBandTiles,[tx,ty,tz]=frame.ownership.capacity.lattice.dimensions.map(n=>n/4) as [number,number,number];
  const a=device!.createBuffer({size:4*(tiles+64*cap),usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ}),b=device!.createBuffer({size:4*64*cap,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  const e=device!.createCommandEncoder();e.copyBufferToBuffer(frame.presentation.phi.buffer,4*frame.stageBandWord,a,0,a.size);e.copyBufferToBuffer(frame.band.iterate,0,b,0,b.size);device!.queue.submit([e.finish()]);
  await a.mapAsync(GPUMapMode.READ);const words=a.getMappedRange().slice(0);a.unmap();a.destroy();
  await b.mapAsync(GPUMapMode.READ);const correction=new Float32Array(b.getMappedRange().slice(0));b.unmap();b.destroy();
  const now={slots:new Uint32Array(words,0,tiles),total:new Float32Array(words,4*tiles),correction},before=warmHeld;warmHeld=now;
  if(!compare||!before)return undefined;
  const row=(x:number,y:number,z:number)=>((x+y+z)&1)*32+(x>>1)+2*y+8*z;
  // Values of a global h cell: [total, correction] of the frame, or undefined outside its band or not liquid.
  const at=(f:NonNullable<typeof warmHeld>,x:number,y:number,z:number):[number,number]|undefined=>{
   if(x<0||y<0||z<0||x>=4*tx||y>=4*ty||z>=4*tz)return undefined;
   const slot=f.slots[(x>>2)+tx*((y>>2)+ty*(z>>2))]!;if(slot===0)return undefined;
   const s=slot-1,lx=x&3,ly=y&3,lz=z&3,total=f.total[s*64+lx+4*ly+16*lz]!,c=f.correction[s*64+row(lx,ly,lz)]!;
   return total===0&&c===0?undefined:[total,c];
  };
  // The start's error against this frame's answer, per start.
  const error=(x:number,y:number,z:number):[number,number,number]|undefined=>{
   const here=at(now,x,y,z);if(!here)return undefined;const old=at(before,x,y,z);
   return [-here[1],old?old[0]-here[0]:-here[1],old?old[1]-here[1]:-here[1]];
  };
  const sum=[0,0,0],residual=[0,0,0],worst=[0,0,0];let cells=0,carried=0,dot=0,n0=0,n1=0;
  for(let t=0;t<tiles;t++){
   if(now.slots[t]===0)continue;const X=4*(t%tx),Y=4*(Math.floor(t/tx)%ty),Z=4*Math.floor(t/(tx*ty));
   for(let l=0;l<64;l++){
    const x=X+(l&3),y=Y+((l>>2)&3),z=Z+(l>>4),own=error(x,y,z);if(!own)continue;cells++;
    const old=at(before,x,y,z),here=at(now,x,y,z)!;if(old){carried++;dot+=old[1]*here[1];n0+=old[1]*old[1];n1+=here[1]*here[1];}
    // Unit-coefficient Laplacian of the error: liquid band neighbours couple, any other open neighbour is taken as air (Dirichlet), a neighbour outside the band as the Neumann seam.
    const r=[0,0,0];
    for(const [dx,dy,dz] of [[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]] as const){
     const q=[x+dx,y+dy,z+dz] as const;if(q[0]<0||q[1]<0||q[2]<0||q[0]>=4*tx||q[1]>=4*ty||q[2]>=4*tz)continue;
     const inBand=now.slots[(q[0]>>2)+tx*((q[1]>>2)+ty*(q[2]>>2))]!==0;if(!inBand)continue;
     const other=error(q[0],q[1],q[2]);for(let k=0;k<3;k++)r[k]+=(other?other[k]!:0)-own[k]!;
    }
    for(let k=0;k<3;k++){sum[k]+=own[k]!*own[k]!;residual[k]+=r[k]!*r[k]!;worst[k]=Math.max(worst[k]!,Math.abs(r[k]!));}
   }
  }
  const rms=(v:number[])=>v.map(x=>Math.sqrt(x/Math.max(1,cells)));
  return {cells,carried,correlation:dot/Math.sqrt(n0*n1),starts:["4h start","last total","4h start + last correction"],errorRms:rms(sum),residualRms:rms(residual),residualMax:worst};
 };
 const readLayout=async(layoutFrame:number)=>{
  const ownership=(solver as unknown as {mixedFrame:{ownership:{presentation:{buffer:GPUBuffer};capacity:{lattice:{dimensions:readonly number[]}}}}}).mixedFrame.ownership;
  const [tx,ty,tz]=ownership.capacity.lattice.dimensions.map(n=>n/4) as [number,number,number],tiles=tx*ty*tz;
  const gpu=device!,read=gpu.createBuffer({size:4*tiles,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  const encoder=gpu.createCommandEncoder();encoder.copyBufferToBuffer(ownership.presentation.buffer,0,read,0,4*tiles);
  // The census's importance words for the same tiles (UNIFORM_STAGE_IMPORTANCE): why a tile is h.
  const importance=(solver as unknown as {mixedDynamic:{importance:{buffer:GPUBuffer;offset:number}}}).mixedDynamic.importance;
  const readWhy=gpu.createBuffer({size:8*tiles,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  encoder.copyBufferToBuffer(importance.buffer,importance.offset,readWhy,0,8*tiles);
  // The activity stage's tile heat (own, then spread one tile): what seeds and keeps samples.
  const flip=(solver as unknown as {mixedFrame:{narrowBandFlip:{bins:GPUBuffer;activityWord:number}}}).mixedFrame.narrowBandFlip;
  const readHeat=gpu.createBuffer({size:8*tiles,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  encoder.copyBufferToBuffer(flip.bins,4*(flip.activityWord+2*tiles),readHeat,0,8*tiles);
  // --reasons: the census rule that put each tile in the band (UNIFORM_STAGE_REASON).
  const reasonWords=(solver as unknown as {mixedDynamic:{reasons:{buffer:GPUBuffer;offset:number}}}).mixedDynamic.reasons;
  const readReason=gpu.createBuffer({size:4*tiles,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  encoder.copyBufferToBuffer(reasonWords.buffer,reasonWords.offset,readReason,0,4*tiles);gpu.queue.submit([encoder.finish()]);
  await readReason.mapAsync(GPUMapMode.READ);const rule=new Uint32Array(readReason.getMappedRange().slice(0));readReason.unmap();readReason.destroy();
  // --marks: the refinement kernel (refine/sweptSurface) on the host, over the
  // samples and nearest-crossing bank this step left: the tiles the next
  // census joins, by the kind of sample that marks each (1 outside the liquid,
  // its own support box; 2 within a cell inside, its nearest crossing cell;
  // 4 inside with no crossing in reach, its own box) and by which term of the
  // box reaches it (the support alone, + travel, + padding, + both).
  let predicted:{full:Uint8Array;noPad:Uint8Array;noTravel:Uint8Array;bare:Uint8Array;ends:Uint8Array;faces:Uint8Array;point:Uint8Array;point2:Uint8Array;samples:Record<string,number>}|undefined;
  if(recordMarks){
   const nb=(solver as unknown as {mixedFrame:{narrowBandFlip:{activeParticles:GPUBuffer;state:GPUBuffer;bins:GPUBuffer}}}).mixedFrame.narrowBandFlip;
   const dims=ownership.capacity.lattice.dimensions as [number,number,number],cells=dims[0]*dims[1]*dims[2];
   const bank=2*cells+2*tiles+(dims[0]+1)*(dims[1]+1)*(dims[2]+1),bankWords=cells+4+3*tiles;
   const readState=gpu.createBuffer({size:16,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
   const readBank=gpu.createBuffer({size:4*bankWords,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
   let e=gpu.createCommandEncoder();e.copyBufferToBuffer(nb.state,0,readState,0,16);e.copyBufferToBuffer(nb.bins,4*bank,readBank,0,4*bankWords);gpu.queue.submit([e.finish()]);
   await readState.mapAsync(GPUMapMode.READ);const live=Math.min(new Uint32Array(readState.getMappedRange())[0]!,nb.activeParticles.size/48);readState.unmap();readState.destroy();
   await readBank.mapAsync(GPUMapMode.READ);const nearestBank=new Uint32Array(readBank.getMappedRange().slice(0));readBank.unmap();readBank.destroy();
   const readSamples=gpu.createBuffer({size:48*live,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
   e=gpu.createCommandEncoder();e.copyBufferToBuffer(nb.activeParticles,0,readSamples,0,48*live);gpu.queue.submit([e.finish()]);
   await readSamples.mapAsync(GPUMapMode.READ);const sample=new Float32Array(readSamples.getMappedRange().slice(0));readSamples.unmap();readSamples.destroy();
   const cell=(solver as unknown as {mixedFrame:{ownership:{capacity:{lattice:{cellSize_m:readonly number[]}}}}}).mixedFrame.ownership.capacity.lattice.cellSize_m;
   const T=[tx,ty,tz],radius=0.875,padding=1;
   predicted={full:new Uint8Array(tiles),noPad:new Uint8Array(tiles),noTravel:new Uint8Array(tiles),bare:new Uint8Array(tiles),ends:new Uint8Array(tiles),faces:new Uint8Array(tiles),point:new Uint8Array(tiles),point2:new Uint8Array(tiles),samples:{live,deep:0,ballistic:0,outside:0,nearCrossing:0,noCrossing:0}};
   const variants:[Uint8Array,number,number][]=[[predicted.full,1,padding],[predicted.noPad,1,0],[predicted.noTravel,0,padding],[predicted.bare,0,0]];
   const lo=[0,0,0],hi=[0,0,0],low=[0,0,0],high=[0,0,0],travel=[0,0,0];
   for(let i=0;i<live;i++){
    const o=12*i,depth=sample[o+7]!,flag=sample[o+11]!;
    if(depth<-1){predicted.samples.deep!++;continue;}if(flag===1){predicted.samples.ballistic!++;continue;}
    const c=[0,1,2].map(a=>Math.min(dims[a]!-1,Math.max(0,Math.floor(sample[o+a]!))));
    const tile=(c[0]!>>2)+tx*((c[1]!>>2)+ty*(c[2]!>>2));
    const reach=nearestBank[cells+4+2*tiles+tile]!;
    const nearest=reach&1?nearestBank[c[0]!+dims[0]*(c[1]!+dims[1]*c[2]!)]!:0xffffffff;
    let kind=1;
    if(depth<=0&&nearest!==0xffffffff){kind=2;low[0]=nearest%dims[0];low[1]=Math.floor(nearest/dims[0])%dims[1];low[2]=Math.floor(nearest/(dims[0]*dims[1]));for(let a=0;a<3;a++)high[a]=low[a]!+1;}
    else{if(depth<=0)kind=4;for(let a=0;a<3;a++){low[a]=sample[o+a]!-radius;high[a]=sample[o+a]!+radius;}}
    predicted.samples[kind===1?"outside":kind===2?"nearCrossing":"noCrossing"]!++;
    for(let a=0;a<3;a++)travel[a]=stepInterval*sample[o+(flag>0.5?4:8)+a]!/cell[a]!;
    for(const [marks,moving,pad] of variants){
     for(let a=0;a<3;a++){
      lo[a]=Math.min(T[a]!-1,Math.max(0,Math.floor((low[a]!+moving*Math.min(travel[a]!,0)-pad)/4)));
      hi[a]=Math.max(lo[a]!,Math.min(T[a]!-1,Math.max(0,Math.floor((high[a]!+moving*Math.max(travel[a]!,0)+pad-1e-5)/4))));
     }
     for(let z=lo[2]!;z<=hi[2]!;z++)for(let y=lo[1]!;y<=hi[1]!;y++)for(let x=lo[0]!;x<=hi[0]!;x++)marks[x+tx*(y+ty*z)]!|=kind;
    }
    // The padded box where the step starts and where it ends, without the hull between.
    for(const end of [0,1]){
     for(let a=0;a<3;a++){
      lo[a]=Math.min(T[a]!-1,Math.max(0,Math.floor((low[a]!+end*travel[a]!-padding)/4)));
      hi[a]=Math.max(lo[a]!,Math.min(T[a]!-1,Math.max(0,Math.floor((high[a]!+end*travel[a]!+padding-1e-5)/4))));
     }
     for(let z=lo[2]!;z<=hi[2]!;z++)for(let y=lo[1]!;y<=hi[1]!;y++)for(let x=lo[0]!;x<=hi[0]!;x++)predicted.ends[x+tx*(y+ty*z)]!|=kind;
    }
    // Travel applied to where in its cell the surface is (within the sample's
    // own distance of it), not to the whole cell: the cell's box at the start,
    // and the cells that part of the surface is carried into at the end.
    // point2 also allows gravity acting unopposed over the step.
    for(const [marks,slack] of [[predicted.point,0],[predicted.point2,9.81*stepInterval*stepInterval/cell[1]!]] as [Uint8Array,number][]){
     for(const end of [0,1]){
      for(let a=0;a<3;a++){
       let from=low[a]!,to=high[a]!;
       if(end&&kind===2){
        const reach=Math.abs(depth),near=Math.max(low[a]!,sample[o+a]!-reach),far=Math.min(high[a]!,sample[o+a]!+reach);
        if(near<=far){from=Math.floor(near+travel[a]!-(a===1?slack:0));to=Math.floor(far+travel[a]!)+1;}
        else{from=low[a]!+Math.min(travel[a]!,0)-(a===1?slack:0);to=high[a]!+Math.max(travel[a]!,0);}
       }else if(end){from+=Math.min(travel[a]!,0);to+=Math.max(travel[a]!,0);}
       lo[a]=Math.min(T[a]!-1,Math.max(0,Math.floor((from-padding)/4)));
       hi[a]=Math.max(lo[a]!,Math.min(T[a]!-1,Math.max(0,Math.floor((to+padding-1e-5)/4))));
      }
      for(let z=lo[2]!;z<=hi[2]!;z++)for(let y=lo[1]!;y<=hi[1]!;y++)for(let x=lo[0]!;x<=hi[0]!;x++)marks[x+tx*(y+ty*z)]!|=kind;
     }
    }
    // The hull padded along one axis at a time (a cell's face neighbours, not its diagonal ones).
    for(let axis=0;axis<3;axis++){
     for(let a=0;a<3;a++){
      const pad=a===axis?padding:0;
      lo[a]=Math.min(T[a]!-1,Math.max(0,Math.floor((low[a]!+Math.min(travel[a]!,0)-pad)/4)));
      hi[a]=Math.max(lo[a]!,Math.min(T[a]!-1,Math.max(0,Math.floor((high[a]!+Math.max(travel[a]!,0)+pad-1e-5)/4))));
     }
     for(let z=lo[2]!;z<=hi[2]!;z++)for(let y=lo[1]!;y<=hi[1]!;y++)for(let x=lo[0]!;x<=hi[0]!;x++)predicted.faces[x+tx*(y+ty*z)]!|=kind;
    }
   }
  }
  await read.mapAsync(GPUMapMode.READ);const words=new Uint32Array(read.getMappedRange().slice(0));read.unmap();read.destroy();
  await readWhy.mapAsync(GPUMapMode.READ);const why=new Uint32Array(readWhy.getMappedRange().slice(0));readWhy.unmap();readWhy.destroy();
  await readHeat.mapAsync(GPUMapMode.READ);const heat=new Float32Array(readHeat.getMappedRange().slice(0));readHeat.unmap();readHeat.destroy();
  const at=(x:number,y:number,z:number)=>x+tx*(y+ty*z),inside=(x:number,y:number,z:number)=>x>=0&&y>=0&&z>=0&&x<tx&&y<ty&&z<tz;
  const fine=Uint8Array.from(words,w=>w>>>31);
  const each=(visit:(x:number,y:number,z:number,t:number)=>void)=>{for(let z=0;z<tz;z++)for(let y=0;y<ty;y++)for(let x=0;x<tx;x++)visit(x,y,z,at(x,y,z));};
  // Seam exposure: an h tile is regular when its whole 3x3x3 tile stencil is h
  // (the extension, sampling and transfer kernels' fast form); a 4h tile is
  // regular when its stencil is all 4h. Everything else takes the seam forms.
  const seams={h:0,hRegular:0,hFaceRegular:0,coarseSeam:0,hByCoarseNeighbours:new Array<number>(27).fill(0)};
  each((x,y,z,t)=>{
   let other=0,face=0;
   for(let dz=-1;dz<=1;dz++)for(let dy=-1;dy<=1;dy++)for(let dx=-1;dx<=1;dx++){
    if(!inside(x+dx,y+dy,z+dz))continue;
    if(fine[at(x+dx,y+dy,z+dz)]!==fine[t]){other++;if(Math.abs(dx)+Math.abs(dy)+Math.abs(dz)===1)face++;}
   }
   if(fine[t]){seams.h++;if(other===0)seams.hRegular++;if(face===0)seams.hFaceRegular++;seams.hByCoarseNeighbours[other]!++;}
   else if(other>0)seams.coarseSeam++;
  });
  // --band-readers: the crossing search's tile lists (search: within two
  // tiles of a crossing; distance: within three) against the tiles anything
  // reads them in: h tiles, tiles holding samples, heated tiles, and one tile
  // around those for the vertices and cells shared across a tile face.
  let bandReaders:Record<string,any>|undefined;
  if(recordBandReaders){
   const dims=ownership.capacity.lattice.dimensions as [number,number,number],cells=dims[0]*dims[1]*dims[2];
   const base=2*cells,bank=2*cells+2*tiles+(dims[0]+1)*(dims[1]+1)*(dims[2]+1)+cells;
   const a=gpu.createBuffer({size:8*tiles,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ}),b=gpu.createBuffer({size:4*(4+3*tiles),usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
   const e=gpu.createCommandEncoder();e.copyBufferToBuffer(flip.bins,4*base,a,0,8*tiles);e.copyBufferToBuffer(flip.bins,4*bank,b,0,4*(4+3*tiles));gpu.queue.submit([e.finish()]);
   await a.mapAsync(GPUMapMode.READ);const held=new Uint32Array(a.getMappedRange().slice(0));a.unmap();a.destroy();
   await b.mapAsync(GPUMapMode.READ);const band=new Uint32Array(b.getMappedRange().slice(0));b.unmap();b.destroy();
   const reader=new Uint8Array(tiles),hot=new Uint8Array(tiles),sampled=new Uint8Array(tiles);
   for(let t=0;t<tiles;t++){sampled[t]=+(held[t]!==0||held[tiles+t]!==0);hot[t]=+(heat[tiles+t]!>0);reader[t]=+(fine[t]!==0||sampled[t]!==0||hot[t]!==0);}
   const near=new Uint8Array(tiles),nearFine=new Uint8Array(tiles),nearSampled=new Uint8Array(tiles);
   each((x,y,z,t)=>{for(let c=-1;c<=1;c++)for(let q=-1;q<=1;q++)for(let p=-1;p<=1;p++){if(!inside(x+p,y+q,z+c))continue;const n=at(x+p,y+q,z+c);if(reader[n])near[t]=1;if(fine[n])nearFine[t]=1;if(fine[n]||sampled[n])nearSampled[t]=1;}});
   bandReaders={candidates:band[0]!,search:band[1]!,distance:band[2]!,crossing:0,h:0,sampled:0,hot:0,readers:0,
    searchRead:0,distanceRead:0,searchNearH:0,distanceNearH:0,searchNearSampled:0,distanceNearSampled:0,searchOwn:0,distanceOwn:0,distanceH:0,searchH:0};
   for(let t=0;t<tiles;t++){
    const reach=band[4+2*tiles+t]!,r=bandReaders;
    r.crossing!+=+(band[4+2*t]!==0||band[4+2*t+1]!==0);r.h!+=fine[t]!;r.sampled!+=sampled[t]!;r.hot!+=hot[t]!;r.readers!+=reader[t]!;
    if(reach&1){r.searchRead!+=near[t]!;r.searchNearH!+=nearFine[t]!;r.searchNearSampled!+=nearSampled[t]!;r.searchOwn!+=reader[t]!;r.searchH!+=fine[t]!;}
    if(reach&2){r.distanceRead!+=near[t]!;r.distanceNearH!+=nearFine[t]!;r.distanceNearSampled!+=nearSampled[t]!;r.distanceOwn!+=reader[t]!;r.distanceH!+=fine[t]!;}
   }
  }
  const FACES=[[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]] as const;
  // Neighbours past the domain count as the tile's own kind.
  const around=(mask:Uint8Array,x:number,y:number,z:number,faces:boolean)=>{let n=0;
   if(faces){for(const [a,b,c] of FACES)n+=inside(x+a,y+b,z+c)?mask[at(x+a,y+b,z+c)]!:mask[at(x,y,z)]!;return n;}
   for(let c=-1;c<=1;c++)for(let b=-1;b<=1;b++)for(let a=-1;a<=1;a++){if(a||b||c)n+=inside(x+a,y+b,z+c)?mask[at(x+a,y+b,z+c)]!:mask[at(x,y,z)]!;}
   return n;};
  const census=(mask:Uint8Array)=>{
   let h=0,interior=0,seam=0,coarseSeam=0,seamFaces=0,holes=0;const faceNeighbours=[0,0,0,0,0,0,0];
   each((x,y,z,t)=>{
    const near=around(mask,x,y,z,false),faces=around(mask,x,y,z,true);
    if(mask[t]){h++;faceNeighbours[faces]!++;if(near===26)interior++;else seam++;
     for(const [a,b,c] of FACES)if(inside(x+a,y+b,z+c)&&!mask[at(x+a,y+b,z+c)])seamFaces++;}
    else{if(near>0)coarseSeam++;if(faces>=5)holes++;}
   });
   return {h,interior,seam,coarseSeam,seamFaces,holes,faceNeighbours};
  };
  // Face-connected pieces of the h set.
  const piece=new Int32Array(tiles).fill(-1),sizes:number[]=[];
  each((x,y,z,t)=>{
   if(!fine[t]||piece[t]!>=0)return;const id=sizes.length,stack=[[x,y,z]];piece[t]=id;let size=0;
   while(stack.length){const [px,py,pz]=stack.pop()!;size++;
    for(const [a,b,c] of FACES){const qx=px!+a,qy=py!+b,qz=pz!+c;if(!inside(qx,qy,qz))continue;const q=at(qx,qy,qz);if(fine[q]&&piece[q]!<0){piece[q]=id;stack.push([qx,qy,qz]);}}}
   sizes.push(size);
  });
  sizes.sort((a,b)=>b-a);
  // A one-tile closing: grow the h set by its 26 neighbours, then shrink it again.
  const grown=new Uint8Array(tiles),closed=new Uint8Array(tiles);
  each((x,y,z,t)=>{grown[t]=fine[t]||around(fine,x,y,z,false)>0?1:0;});
  each((x,y,z,t)=>{closed[t]=grown[t]&&around(grown,x,y,z,false)===26?1:0;if(fine[t])closed[t]=1;});
  // Per h tile: the census flags it carries, and the 4h tiles that hold a crossing.
  const reasons={crossing:0,wet:0,required:0,held:0,dropped:0,triggered:0,shape:0,thin:0,strain:0,rotation:0,impact:0,approach:0,noCrossing:0,dry:0,neither:0,besideCrossing:0},coarse={crossing:0,wet:0,dropped:0,triggered:0};
  const crossing=Uint8Array.from({length:tiles},(_,t)=>(why[2*t+1]!>>>28)&1);
  each((x,y,z,t)=>{
   const w=why[2*t+1]!,flags={crossing:(w>>>28)&1,wet:(w>>>29)&1,required:(w>>>25)&1,held:(w>>>26)&1,dropped:(w>>>27)&1,triggered:(w>>>16)&63?1:0};
   if(!fine[t]){for(const k of ["crossing","wet","dropped","triggered"] as const)coarse[k]+=flags[k];return;}
   for(const k of ["crossing","wet","required","held","dropped","triggered"] as const)reasons[k]+=flags[k];
   (["shape","thin","strain","rotation","impact","approach"] as const).forEach((k,i)=>{reasons[k]+=(w>>>(16+i))&1;});
   if(!flags.crossing){reasons.noCrossing++;if(around(crossing,x,y,z,false)>0)reasons.besideCrossing++;}
   if(!flags.wet)reasons.dry++;if(!flags.crossing&&!flags.required)reasons.neither++;
  });
  // How the budget's admitted set moves between the censuses read: a tile it
  // drops keeps its samples for the retirement time, so what stays h is the
  // union of what was admitted over that time.
  const required=Uint8Array.from({length:tiles},(_,t)=>(why[2*t+1]!>>>25)&1),triggered=Uint8Array.from({length:tiles},(_,t)=>(why[2*t+1]!>>>16)&63?1:0);
  const previous=layoutHistory.at(-1);layoutHistory.push({frame:layoutFrame,required,triggered});
  const count=(mask:Uint8Array)=>mask.reduce((a,b)=>a+b,0);
  const union=(key:"required"|"triggered",back:number)=>{const u=new Uint8Array(tiles);for(const entry of layoutHistory.slice(-back))entry[key].forEach((v,t)=>{if(v)u[t]=1;});return u;};
  const window=(back:number)=>{const r=union("required",back),g=union("triggered",back);let hRequired=0,hTriggered=0;fine.forEach((v,t)=>{if(v){hRequired+=r[t]!;hTriggered+=g[t]!;}});
   return {censuses:Math.min(back,layoutHistory.length),required:count(r),triggered:count(g),hInRequired:hRequired,hInTriggered:hTriggered};};
  const churn=previous&&previous.frame===layoutFrame-1?{keptRequired:required.reduce((n,v,t)=>n+(v&previous.required[t]!),0),requiredWas:count(previous.required),
   droppedWasRequired:required.reduce((n,v,t)=>n+(!v&&previous.required[t]&&triggered[t]?1:0),0),keptTriggered:triggered.reduce((n,v,t)=>n+(v&previous.triggered[t]!),0),triggeredWas:count(previous.triggered)}:undefined;
  // Face-adjacent pairs of triggered tiles by the budget's decision: a
  // selection blind to position splits half of them (admitted beside dropped).
  const pairs={bothAdmitted:0,bothDropped:0,split:0};
  each((x,y,z,t)=>{if(!triggered[t])return;for(const [a,b,c] of [[1,0,0],[0,1,0],[0,0,1]] as const){if(!inside(x+a,y+b,z+c))continue;const q=at(x+a,y+b,z+c);if(!triggered[q])continue;
   if(required[t]&&required[q])pairs.bothAdmitted++;else if(!required[t]&&!required[q])pairs.bothDropped++;else pairs.split++;}});
  const admitted=census(required);
  // What the same number of admitted tiles would hold at h were they chosen
  // by a score pooled over neighbours instead of the tile's own: the h tiles
  // within one tile of an admitted one (its support collar), against today's.
  const top=Float64Array.from({length:tiles},(_,t)=>{if(!triggered[t])return 0;const w0=why[2*t]!,w1=why[2*t+1]!;return Math.max(w0&255,(w0>>>8)&255,(w0>>>16)&255,(w0>>>24)&255,w1&255,(w1>>>8)&255);});
  const allowed=count(required);
  const collar=(mask:Uint8Array)=>{let n=0;each((x,y,z,t)=>{if(fine[t]&&(mask[t]||around(mask,x,y,z,false)>0))n++;});return n;};
  const pooled=(radius:number,mode:"sum"|"max")=>{
   const key=new Float64Array(tiles);
   each((x,y,z,t)=>{if(!triggered[t])return;let v=0;
    for(let c=-radius;c<=radius;c++)for(let b=-radius;b<=radius;b++)for(let a=-radius;a<=radius;a++){if(!inside(x+a,y+b,z+c))continue;const q=top[at(x+a,y+b,z+c)]!;v=mode==="sum"?v+q:Math.max(v,q);}
    key[t]=v+1e-9*t;});
   const order=Array.from({length:tiles},(_,t)=>t).filter(t=>triggered[t]).sort((a,b)=>key[b]!-key[a]!),mask=new Uint8Array(tiles);
   for(const t of order.slice(0,allowed))mask[t]=1;
   const shape=census(mask);return {interior:shape.interior,seamFaces:shape.seamFaces,isolated:shape.faceNeighbours[0],hWithCollar:collar(mask)};
  };
  const whatIf={allowed,today:{interior:admitted.interior,seamFaces:admitted.seamFaces,isolated:admitted.faceNeighbours[0],hWithCollar:collar(required)},triggeredWithCollar:collar(triggered),
   sum1:pooled(1,"sum"),sum2:pooled(2,"sum"),sum3:pooled(3,"sum"),max1:pooled(1,"max")};
  // Every h tile by what holds it: admitted now; its own heat still cooling;
  // only a neighbour's heat (the collar); no heat at all. Each with how many hold liquid.
  const held={admitted:[0,0],cooling:[0,0],collar:[0,0],cold:[0,0],coldCrossing:0,coldBesideHeat:0,hotCoarse:0,collarCoarse:0};
  const warm=Uint8Array.from({length:tiles},(_,t)=>heat[tiles+t]!>0?1:0);
  each((x,y,z,t)=>{
   const own=heat[t]!,target=heat[tiles+t]!,wet=(why[2*t+1]!>>>29)&1;
   if(!fine[t]){if(own>0)held.hotCoarse++;else if(target>0)held.collarCoarse++;return;}
   const kind=required[t]?held.admitted:own>0?held.cooling:target>0?held.collar:held.cold;kind[0]!++;kind[1]!+=wet;
   if(kind===held.cold){held.coldCrossing+=crossing[t]!;if(around(warm,x,y,z,false)>0)held.coldBesideHeat++;}
  });
  // Submerged h tiles (liquid, no crossing): what holds each, whether a
  // crossing tile at h is beside it, and how long since it last held a
  // crossing itself, in consecutive reads (0: not within the reads kept).
  const submerged={count:0,admitted:0,triggered:0,strain:0,cooling:0,collar:0,cold:0,besideCrossingH:0,besideAdmitted:0,sinceCrossing:{within3:0,within12:0,within30:0,never:0}};
  const air={count:0,cooling:0,collar:0,cold:0,besideCrossingH:0};
  const ruleNames=Object.fromEntries(Object.entries(UNIFORM_STAGE_REASON).map(([name,code])=>[code,name]));
  const rules:Record<string,Record<string,number>>={all:{},crossing:{},submerged:{},air:{}};
  const tally=(group:string,t:number)=>{const name=ruleNames[rule[t]!]??String(rule[t]);rules[group]![name]=(rules[group]![name]??0)+1;};
  const crossingH=Uint8Array.from({length:tiles},(_,t)=>crossing[t]!&fine[t]!);
  layoutHistory.at(-1)!.crossing=crossing;
  each((x,y,z,t)=>{
   if(fine[t]&&recordReasons){tally("all",t);tally(crossing[t]?"crossing":(why[2*t+1]!>>>29)&1?"submerged":"air",t);}
   if(!fine[t]||crossing[t])return;const wet=(why[2*t+1]!>>>29)&1,own=heat[t]!,target=heat[tiles+t]!;
   const beside=around(crossingH,x,y,z,false)>0;
   if(!wet){air.count++;if(own>0)air.cooling++;else if(target>0)air.collar++;else air.cold++;if(beside)air.besideCrossingH++;return;}
   submerged.count++;if(required[t])submerged.admitted++;else if(own>0)submerged.cooling++;else if(target>0)submerged.collar++;else submerged.cold++;
   if(triggered[t])submerged.triggered++;if((why[2*t+1]!>>>18)&1)submerged.strain++;
   if(beside)submerged.besideCrossingH++;if(around(required,x,y,z,false)>0)submerged.besideAdmitted++;
   let since=0;for(let back=2;back<=layoutHistory.length;back++){const entry=layoutHistory.at(-back)!;if(entry.frame!==layoutFrame-back+1)break;if(entry.crossing?.[t]){since=back-1;break;}}
   if(since===0)submerged.sinceCrossing.never++;else if(since<=3)submerged.sinceCrossing.within3++;else if(since<=12)submerged.sinceCrossing.within12++;else submerged.sinceCrossing.within30++;
  });
  // Heat with no surface under it: tiles holding their own heat by whether a
  // crossing is in them, and the heat every tile would have were a tile with
  // no crossing to hold none (what its samples and collar would then lose).
  const surfaceless={hot:{crossing:0,submerged:0,air:0},hotH:{crossing:0,submerged:0,air:0},warmBefore:0,warmAfter:0,hWarmBefore:0,hWarmAfter:0,crossingHLosingHeat:0,crossingLosingHeat:0};
  const kept=Float32Array.from({length:tiles},(_,t)=>crossing[t]||required[t]?heat[t]!:0);
  each((x,y,z,t)=>{
   const kind=crossing[t]?"crossing":(why[2*t+1]!>>>29)&1?"submerged":"air";
   if(heat[t]!>0){surfaceless.hot[kind]++;if(fine[t])surfaceless.hotH[kind]++;}
   let spread=kept[t]!;
   for(let dz=-1;dz<=1;dz++)for(let dy=-1;dy<=1;dy++)for(let dx=-1;dx<=1;dx++)if(inside(x+dx,y+dy,z+dz))spread=Math.max(spread,0.5*kept[at(x+dx,y+dy,z+dz)]!);
   const was=heat[tiles+t]!>0,now=spread>0;
   if(was)surfaceless.warmBefore++;if(now)surfaceless.warmAfter++;
   if(fine[t]){if(was)surfaceless.hWarmBefore++;if(now)surfaceless.hWarmAfter++;}
   if(crossing[t]&&was&&!now){surfaceless.crossingLosingHeat++;if(fine[t])surfaceless.crossingHLosingHeat++;}
  });
  // The marks the previous read predicted, against this census's h tiles.
  layoutHistory.at(-1)!.predicted=predicted;
  const before=layoutHistory.at(-2);let marks:Record<string,unknown>|undefined;
  if(before?.predicted&&before.frame===layoutFrame-1){
   const m=before.predicted;
   const group=()=>({count:0,unmarked:0,byKind:{} as Record<string,number>,withoutPadding:0,withoutTravel:0,supportOnly:0,endsOnly:0,facePadding:0});
   const groups={crossing:group(),submerged:group(),air:group()};let markedCoarse=0;
   for(let t=0;t<tiles;t++){
    if(!fine[t]){if(m.full[t])markedCoarse++;continue;}
    const g=groups[crossing[t]?"crossing":(why[2*t+1]!>>>29)&1?"submerged":"air"];g.count++;
    if(!m.full[t]){g.unmarked++;continue;}
    const kinds=[m.full[t]!&1?"outside":"",m.full[t]!&2?"nearCrossing":"",m.full[t]!&4?"noCrossing":""].filter(Boolean).join("+");
    g.byKind[kinds]=(g.byKind[kinds]??0)+1;
    if(m.noPad[t])g.withoutPadding++;if(m.noTravel[t])g.withoutTravel++;if(m.bare[t])g.supportOnly++;if(m.ends[t])g.endsOnly++;if(m.faces[t])g.facePadding++;
   }
   // Travel's precision: the tiles this census held only for where the
   // surface was predicted to go, and those whose vertices a surface cell
   // touches now the step has run (this read's marks without travel).
   const travelOnly={count:0,arrived:0,hull:0,hullArrived:0,missed:0,missedBesideH:0,needed:0};
   // Missed: a tile a surface cell touches after the step that ran it at 4h.
   if(predicted)each((x,y,z,t)=>{if(!predicted!.noTravel[t])return;travelOnly.needed++;if(fine[t])return;travelOnly.missed++;if(around(fine,x,y,z,false)>0)travelOnly.missedBesideH++;});
   if(predicted)for(let t=0;t<tiles;t++){
    if(!fine[t]||!m.full[t]||m.noTravel[t])continue;
    travelOnly.count++;if(predicted.noTravel[t])travelOnly.arrived++;
    if(!m.ends[t]){travelOnly.hull++;if(predicted.noTravel[t])travelOnly.hullArrived++;}
   }
   // Each tighter rule over the same samples: the h tiles it would not have
   // marked, and those of them a surface cell touched once the step had run.
   const tighter:Record<string,{released:number;thenNeeded:number}>={};
   for(const name of ["ends","point","point2"] as const){
    const rule={released:0,thenNeeded:0};tighter[name]=rule;
    for(let t=0;t<tiles;t++)if(fine[t]&&m.full[t]&&!m[name][t]){rule.released++;if(predicted?.noTravel[t])rule.thenNeeded++;}
   }
   marks={samples:m.samples,markedCoarse,travelOnly,tighter,...groups};
  }
  const perSlice=Array.from({length:ty},(_,y)=>{let n=0;for(let z=0;z<tz;z++)for(let x=0;x<tx;x++)n+=fine[at(x,y,z)]!;return n;});
  const y=perSlice.indexOf(Math.max(...perSlice));
  return {tiles:[tx,ty,tz],seams,...(bandReaders?{bandReaders}:{}),reasons,held,submerged,air,surfaceless,...(recordReasons?{rules}:{}),...(marks?{marks}:{}),coarse,pairs,admitted,whatIf,churn,window12:window(12),window30:window(30),now:census(fine),closed:census(closed),pieces:sizes.length,largestPieces:sizes.slice(0,8),piecesUnder8:sizes.filter(n=>n<8).length,
   hPerY:perSlice,slice:{y,rows:Array.from({length:tz},(_,z)=>Array.from({length:tx},(_,x)=>{const t=at(x,y,z);return required[t]?"#":!fine[t]?".":heat[t]!>0?"c":heat[tiles+t]!>0?"+":"-";}).join(""))}};
 };

 for(let frame=1;frame<=steps;frame++){
  probeFrame=frame;
  if(dispatchProbe)dispatchProbe.enabled=probeFrames.includes(frame);
  if(pace>0)await new Promise(r=>setTimeout(r,pace));
  renderLoad?.();
  const start=performance.now();if(!plain)profile.start();
  work={};
  if(frame===from)for(const key of Object.keys(calls))delete calls[key];
  solver.advanceTo(frame*dt,[]);const encode_ms=performance.now()-start,advanceWork=work;work={};await solver.awaitFrameCompletion();
  const wall_ms=performance.now()-start;
  assert.equal(solver.info.simulationPipelineError,undefined);
  assert.ok(Math.abs((solver.info.completedTime_s??0)-frame*dt)<1e-8,"complete exactly one requested clock step");
  assert.equal(solver.info.encodedSteps,frame,"one solver step a frame");
  const passes=plain?[]:await profile.finish();let projection_ms=0,rest_ms=0,between_ms=0;const byLabel:Record<string,number>={},before:Record<string,number>={};
  for(const pass of passes){
   byLabel[pass.label]=(byLabel[pass.label]??0)+pass.ms;between_ms+=pass.idle_ms;
   if(pass.idle_ms>0.05)before[pass.label]=(before[pass.label]??0)+pass.idle_ms;
   if(PROJECTION.test(pass.label))projection_ms+=pass.ms;else rest_ms+=pass.ms;
   const total=labels.get(pass.label)??{ms:0,passes:0,dispatches:0};
   total.ms+=pass.ms;total.passes++;total.dispatches+=pass.dispatches;labels.set(pass.label,total);
  }
  const flip=solver.narrowBandFlipInfo!;
  const row:Record<string,unknown>={frame,time_s:frame*dt,wall_ms,encode_ms,projection_ms,rest_ms,gpu_ms:projection_ms+rest_ms,between_ms,before,passes:passes.length,
   particles:flip.particles,particleCapacity:flip.capacity,reseedClipped:flip.reseedClipped,allocatedBytes:solver.info.allocatedBytes,
   fineTiles:solver.info.uniformMixedFineTiles,bandTiles:solver.info.uniformPressureBandTiles,bandCycles:solver.info.uniformPressureBandCycles,bandResidual:solver.info.uniformPressureBandResidual,labels:byLabel};
  if(renderLoad)row.renderLoad_ms=renderLoadAlone_ms;
  if(hostWork)row.hostWork=advanceWork;
  // The frame's passes in order, each with the gap before it: what a gap follows.
  if(sequence)row.sequence=passes.map(p=>[p.label,+p.ms.toFixed(3),+p.idle_ms.toFixed(3),p.dispatches]);
  if(appTrace){
   // A step inside the trace's cadence carries no sample: wait only for one that was requested.
   const requested=(solver as unknown as {physicsTracePending:boolean}).physicsTracePending||(solver.info.physicsTrace?.sampleId??0)!==tracedSample;
   for(let wait=0;requested&&wait<400&&(solver.info.physicsTrace?.sampleId??0)===tracedSample&&!solver.info.physicsTraceUnavailable;wait++)await new Promise(r=>setTimeout(r,1));
   const trace=solver.info.physicsTrace;
   if(trace&&trace.sampleId!==tracedSample){
    tracedSample=trace.sampleId;
    const phases:Record<string,number>={};for(const phase of trace.phases)phases[phase.label]=(phases[phase.label]??0)+phase.duration_ms;
    row.appTotal_ms=trace.total_ms;row.appPhases=phases;
   }else if(requested)row.appTraceMissing=solver.info.physicsTraceUnavailable??"no trace";
  }
  if(transferProbe&&probeFrames.includes(frame)){
   const begin=passes.findIndex(p=>p.label==="Uniform mixed momentum");
   const end=passes.findIndex((p,i)=>i>=begin&&p.label==="Uniform mixed body forces");
   row.transferStage=passes.slice(begin,end+1);
  }
  if(track){
   // The highest liquid vertex in each falling letter's column, with the crossing above it interpolated.
   const fields=await readUniformFields(device,solver);
   row.tops=Object.fromEntries(letters.filter(l=>frame>=l.step&&frame<l.step+14).map(l=>{
    for(let y=ny-1;y>=0;y--){const below=fields.vertex(l.x,y,l.z),above=fields.vertex(l.x,y+1,l.z);
     if(below<0&&above>=0)return [l.letter,{top:y+below/(below-above),fallen:l.top0-(y+below/(below-above)),sinceDrop:frame-l.step}];}
    return [l.letter,null];
   }));
  }
  if(layoutAt.includes(frame))row.layout=await readLayout(frame);
  if(warmAt.includes(frame)||warmAt.includes(frame-1))row.warm=await readWarm(warmAt.includes(frame-1));
  rows.push(row);
  if(frame%10===0||frame===steps||track)console.log(JSON.stringify({...row,labels:undefined}));
 }
 const final=await solver.readStats();assert.deepEqual(errors,[]);
 const of=(key:string)=>rows.slice(from-1).map(r=>r[key] as number);
 if(Object.keys(calls).length)console.log("CALLS",JSON.stringify(Object.fromEntries(Object.entries(calls).map(([k,c])=>[k,{perFrame:c.n/(steps-from+1),ms:c.ms/(steps-from+1)}]))));
 const summary={steps,from,sets,grid:[nx,ny,nz],cellSize_m:h,dt_s:dt,construction_ms,
  projection_ms:mean(of("projection_ms")),rest_ms:mean(of("rest_ms")),gpu_ms:mean(of("gpu_ms")),wall_ms:mean(of("wall_ms")),encode_ms:mean(of("encode_ms")),between_ms:mean(of("between_ms")),
  wallMedian_ms:of("wall_ms").sort((a,b)=>a-b)[(steps-from+1)>>1],wallMax_ms:Math.max(...of("wall_ms")),
  particles:mean(of("particles")),particlesMax:Math.max(...of("particles")),particleCapacity:rows.at(-1)!.particleCapacity,
  allocatedBytes:mean(of("allocatedBytes")),allocatedBytesMax:Math.max(...of("allocatedBytes")),
  volumeDrift:final.volumeDrift,tracked:track,plain,transferProbe:probeTransfer,dispatchProbe:probeDispatch};
 console.log(JSON.stringify(summary));
 write(`nbflip-letters-${name}.json`,{date:new Date().toISOString(),arguments:process.argv.slice(2),method:uniformNarrowBandMethod.id,values,
  adapter:{vendor:adapter.info.vendor,architecture:adapter.info.architecture,device:adapter.info.device,description:adapter.info.description,subgroupMinSize:adapter.info.subgroupMinSize,subgroupMaxSize:adapter.info.subgroupMaxSize},
  summary,letters,labels:[...labels].map(([label,t])=>({label,projection:PROJECTION.test(label),msPerStep:t.ms/steps,passesPerStep:t.passes/steps,dispatchesPerStep:t.dispatches/steps})).sort((a,b)=>b.msPerStep-a.msPerStep),
  scene,rows,final,errors});
}catch(error){
 write(`nbflip-letters-${name}-failed.json`,{date:new Date().toISOString(),arguments:process.argv.slice(2),method:uniformNarrowBandMethod.id,values,scene,rows,errors,error:String(error)});
 throw error;
}finally{solver?.destroy();await device?.queue.onSubmittedWorkDone();profile?.destroy();device?.destroy();}
