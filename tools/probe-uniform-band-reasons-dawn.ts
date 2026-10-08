/** Diagnostic only: split the dynamic h band by reason on one captured state
 * (docs/plans/uniform-h-band-thickness-2026-09-29.md, evidence step 1).
 *
 * The solver runs normally, one awaited step at a time (so every frame encodes
 * a census). At each capture step the production census encode is wrapped: in
 * the same encoder, right after it, probe-owned classifiers re-run the census
 * with alternative policies on the unchanged state (the frame-tail extension
 * in velocityScratch, the frame's phi, V and ownership). Nothing they produce
 * is adopted; production reads and adopts its own census as usual.
 *
 *   A  production: horizon 3 frames, gravity shift on
 *   B  horizon 1, gravity on (the classifier shifts only older frames: B = C)
 *   C  horizon 1, gravity zero
 *   D  horizon 3, gravity zero
 *   E3 production with sampledFlow's zero-join removed (WGSL patched here only)
 *   E1 horizon 1, gravity zero, zero-join removed
 *   Z  dt = 0: crossing tiles plus the drift margin/closed-box closure only
 *   Z0 dt = 0 and no drift margin: crossings plus the closed-box closure
 *   C0 horizon 1, gravity zero, no drift margin
 * Crossing tiles (X) are the census's required flags (surfaceTolerance 0).
 *
 * WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js FLUID_GPU_COMPILATION_CONCURRENCY=1 \
 *  node --import tsx tools/probe-uniform-band-reasons-dawn.ts --scene=minimal-power-dam-break-64 \
 *  --capture=5,15,30,60,120 --out=/tmp/band-reasons-64.json
 */
import assert from 'node:assert/strict';
import {writeFile} from 'node:fs/promises';
import {resolve} from 'node:path';
import {pathToFileURL} from 'node:url';
import {createProcessRetainedDawnGPU,type NodeDawnProvider} from '../lib/harness/node-dawn-provider';
import {managedGPUDevice} from '../lib/core/gpu-compilation-manager';
import {requiredFluidDeviceLimits} from '../lib/core/webgpu-device-limits';
import {sceneDocument} from '../lib/core/scene-definition';
import {getSceneDefinition} from '../lib/core/scenes';
import {resolveMethodValues} from '../lib/core/method-contract';
import {uniformVolumeMethod} from '../lib/methods/uniform/uniform-volume-method';
import type {WebGPUUniformReferenceSolver} from '../lib/methods/uniform/webgpu-uniform-reference';
import {UniformMixedDynamicClassifier,type UniformMixedDynamicPolicy} from '../lib/methods/uniform/uniform-mixed-dynamic';
import {mixedCellWidth,type UniformMixedLayout} from '../lib/methods/uniform/uniform-mixed-layout';

const arg=(k:string,d:string)=>process.argv.find(a=>a.startsWith(`--${k}=`))?.slice(k.length+3)??d;
const sceneId=arg('scene','minimal-power-dam-break-64');
const captures=arg('capture','5,15,30,60,120').split(',').map(Number).sort((a,b)=>a-b);
const out=arg('out',`/tmp/band-reasons-${sceneId}.json`);
assert.ok(captures.every(n=>Number.isSafeInteger(n)&&n>0));
/** Census work-buffer layout (uniform-mixed-dynamic.ts): header 20 words,
 * band bits, 6 bound words per tile, the (T+1)³ prefix table, gap words per
 * tile, travel words per tile, wet bits, active bits. */
const HEADER=20;
const ZERO_JOIN='f.low=min(vec3f(0),f.low+min(shift,vec3f(0)));f.high=max(vec3f(0),f.high+max(shift,vec3f(0)));';
const SIGNED='f.low=f.low+min(shift,vec3f(0));f.high=f.high+max(shift,vec3f(0));';

let device:GPUDevice|undefined,solver:WebGPUUniformReferenceSolver|undefined;
const owned:{destroy():void}[]=[];
try{
 const dawn=await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE??resolve('node_modules/webgpu/index.js')).href) as NodeDawnProvider;Object.assign(globalThis,dawn.globals);
 const gpu=createProcessRetainedDawnGPU(dawn,[`backend=${process.env.FLUID_WEBGPU_BACKEND??'metal'}`]),adapter=await gpu.requestAdapter();assert.ok(adapter);
 const dev=managedGPUDevice(await adapter.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter.limits)}),{requireWorkerRealm:false});device=dev;
 const errors:string[]=[];dev.addEventListener('uncapturederror',e=>{e.preventDefault();errors.push(e.error.message);console.error(e.error.message);});
 const scene=sceneDocument(getSceneDefinition(sceneId));
 const values=resolveMethodValues(uniformVolumeMethod,'balanced',JSON.parse(arg('values','{}')));
 const started=performance.now();
 solver=await uniformVolumeMethod.createSolverAsync!(dev,scene,'balanced',values,undefined,()=>{}) as WebGPUUniformReferenceSolver;
 // Private access is confined to this diagnostic.
 const s=solver as any;
 assert.equal(s.mixedCoarsening,'dynamic','the probe needs dynamic coarsening');
 const frame=s.mixedFrame,production:UniformMixedDynamicClassifier=s.mixedDynamic;assert.ok(frame&&production);
 const ownership=frame.ownership;
 const layout0:UniformMixedLayout=ownership.layout;
 const T=layout0.tiles.length,W=Math.ceil(T/32),[tx,ty,tz]=layout0.lattice.dimensions.map(n=>n/4) as [number,number,number];
 const P=(tx+1)*(ty+1)*(tz+1),gapBase=HEADER+W+6*T+P,wetBase=gapBase+2*T;
 const coupled:Uint8Array=s.mixedSolidPromotion().coupled;
 console.log(JSON.stringify({phase:'ready',scene:sceneId,ms:Math.round(performance.now()-started),dimensions:layout0.lattice.dimensions,tiles:T,coupledTiles:coupled.reduce((n,c)=>n+c,0),regions:(scene.fluid.refinementRegions??[]).length}));
 // The zero-join variant compiles the classifier with sampledFlow's bounds kept signed.
 let patched=0;
 const signedDevice=new Proxy(dev,{get(target,property){
  if(property==='createShaderModule')return (d:GPUShaderModuleDescriptor)=>{const code=d.code.replace(ZERO_JOIN,()=>{patched++;return SIGNED;});return target.createShaderModule({...d,code});};
  const value=Reflect.get(target,property,target);return typeof value==='function'?value.bind(target):value;
 }});
 type Variant={name:string;signed?:boolean;drift?:number;policy:(p:UniformMixedDynamicPolicy)=>UniformMixedDynamicPolicy};
 const frameDt=(p:UniformMixedDynamicPolicy)=>p.dt/p.steps;
 const zero=[0,0,0] as const;
 const variants:Variant[]=[
  {name:'A',policy:p=>p},
  {name:'B',policy:p=>({...p,dt:frameDt(p),steps:1})},
  {name:'C',policy:p=>({...p,dt:frameDt(p),steps:1,gravity:zero})},
  {name:'D',policy:p=>({...p,gravity:zero})},
  {name:'E3',signed:true,policy:p=>p},
  {name:'E1',signed:true,policy:p=>({...p,dt:frameDt(p),steps:1,gravity:zero})},
  {name:'Z',policy:p=>({...p,dt:0,steps:1,gravity:zero})},
  {name:'Z0',drift:0,policy:p=>({...p,dt:0,steps:1,gravity:zero})},
  {name:'C0',drift:0,policy:p=>({...p,dt:frameDt(p),steps:1,gravity:zero})},
 ];
 const readBytes=(HEADER+W+T+W)*4;
 // Per-tile signed velocity bound keys (6 per tile) of the h1 census, to locate the fastest tiles.
 const boundsReadback=dev.createBuffer({label:'Band reasons bounds',size:6*T*4,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});owned.push(boundsReadback);
 const probes=await Promise.all(variants.map(async v=>{
  const classifier=new UniformMixedDynamicClassifier(v.signed?signedDevice:dev,ownership,s.volumeA,s.vertexPhiField,s.velocityB);
  await classifier.initialize();classifier.setSolid(coupled);
  const readback=dev.createBuffer({label:`Band reasons ${v.name}`,size:readBytes,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  owned.push(classifier,readback);
  const inner=classifier as unknown as {work:GPUBuffer;params:GPUBuffer};
  return {...v,classifier,readback,work:inner.work,params:inner.params};
 }));
 assert.equal(patched,2,'zero-join patch must apply to both signed classifiers');
 let armed=false;
 type Captured={policy:UniformMixedDynamicPolicy;simulationFine:Uint8Array;owners:number};
 const capture:{fired?:Captured}={};
 const encode=production.encode.bind(production);
 production.encode=(encoder:GPUCommandEncoder,policy:UniformMixedDynamicPolicy)=>{
  encode(encoder,policy);
  if(!armed)return;armed=false;
  const l:UniformMixedLayout=ownership.layout;
  capture.fired={policy,simulationFine:Uint8Array.from(l.tiles,w=>mixedCellWidth(w)===1?1:0),owners:l.cellCount};
  for(const v of probes){
   v.classifier.encode(encoder,v.policy(policy));
   // DynamicPolicy.flow.w, the departure drift margin in cells (queued after encode's own write).
   if(v.drift!==undefined)dev.queue.writeBuffer(v.params,60,new Float32Array([v.drift]));
   if(v.name==='C')encoder.copyBufferToBuffer(v.work,(HEADER+W)*4,boundsReadback,0,6*T*4);
   encoder.copyBufferToBuffer(v.work,0,v.readback,0,(HEADER+W)*4);
   encoder.copyBufferToBuffer(v.work,gapBase*4,v.readback,(HEADER+W)*4,T*4);
   encoder.copyBufferToBuffer(v.work,wetBase*4,v.readback,(HEADER+W+T)*4,W*4);
  }
 };
 const count=(m:Uint8Array)=>m.reduce((n,x)=>n+x,0);
 const minus=(a:Uint8Array,b:Uint8Array)=>a.reduce((n,x,t)=>n+(x&&!b[t]?1:0),0);
 const rows:any[]=[];
 const last=captures.at(-1)!;
 for(let step=1;step<=last;step++){
  armed=captures.includes(step);capture.fired=undefined;
  const t0=performance.now();
  assert.ok(solver.advanceTo(step/30,[]));await solver.awaitFrameCompletion();
  const ms=performance.now()-t0;
  if(!captures.includes(step))continue;
  const f=capture.fired as Captured|undefined;
  if(!f){console.log(JSON.stringify({step,note:'no census encoded this step; capture skipped'}));captures.push(step+1);continue;}
  const masks:Record<string,Uint8Array>={},travel:Record<string,number>={},header:Record<string,number[]>={};
  let crossing:Uint8Array|undefined,wet:Uint8Array|undefined;
  for(const v of probes){
   await v.readback.mapAsync(GPUMapMode.READ);
   const words=new Uint32Array(v.readback.getMappedRange()).slice();v.readback.unmap();
   const floats=new Float32Array(words.buffer);
   masks[v.name]=Uint8Array.from({length:T},(_,t)=>(words[HEADER+(t>>5)]!>>>(t&31))&1);
   travel[v.name]=floats[18]!;header[v.name]=[words[0]!,words[2]!,words[19]!];
   const x=Uint8Array.from({length:T},(_,t)=>(words[HEADER+W+t]!)===0xffffffff?0:1);
   const w=Uint8Array.from({length:T},(_,t)=>(words[HEADER+W+T+(t>>5)]!>>>(t&31))&1);
   if(!crossing){crossing=x;wet=w;}else{assert.equal(minus(x,crossing)+minus(crossing,x),0,`${v.name} crossings differ`);}
  }
  const X=crossing!,L=wet!,{A,B,C,D,E3,E1,Z,Z0,C0}=masks as Record<'A'|'B'|'C'|'D'|'E3'|'E1'|'Z'|'Z0'|'C0',Uint8Array>;
  // One-frame travel per tile, in h, from its own face-velocity bounds (classify's speed).
  await boundsReadback.mapAsync(GPUMapMode.READ);
  const keys=new Uint32Array(boundsReadback.getMappedRange()).slice();boundsReadback.unmap();
  const orderValue=(k:number)=>{const b=new Uint32Array([(k&0x80000000)?(k&0x7fffffff):(~k>>>0)]);return new Float32Array(b.buffer)[0]!;};
  const h=layout0.lattice.cellSize_m,frameStep=f.policy.dt/f.policy.steps;
  const speed=Float64Array.from({length:T},(_,t)=>{let m=0;for(let a=0;a<3;a++){const lo=keys[6*t+a]!,hi=keys[6*t+3+a]!;
   if(lo!==0xffffffff)m=Math.max(m,Math.abs(orderValue(lo))*frameStep/h[a]!);if(hi!==0)m=Math.max(m,Math.abs(orderValue(hi))*frameStep/h[a]!);}return m;});
  const order=Array.from({length:T},(_,t)=>t).sort((a,b)=>speed[b]!-speed[a]!);
  const fastest=order.slice(0,5).map(t=>({tile:[t%tx,Math.floor(t/tx)%ty,Math.floor(t/(tx*ty))],travel_h:+speed[t]!.toFixed(2),crossing:X[t],wet:L[t]}));
  const pct=(sel:(t:number)=>boolean,q:number)=>{const v:number[]=[];for(let t=0;t<T;t++)if(sel(t))v.push(speed[t]!);v.sort((a,b)=>a-b);return v.length?+v[Math.min(v.length-1,Math.floor(q*v.length))]!.toFixed(2):0;};
  const S=f.simulationFine;
  const info=solver.info as unknown as Record<string,unknown>;
  const liquidSplit=(m:Uint8Array)=>{let wetOnly=0,dry=0;for(let t=0;t<T;t++)if(m[t]&&!X[t]){if(L[t])wetOnly++;else dry++;}return {wetNonCrossing:wetOnly,dryNonCrossing:dry};};
  const row={step,t:step/30,frameMs:Math.round(ms),
   policy:{dt:f.policy.dt,steps:f.policy.steps,gravity:f.policy.gravity,reach:f.policy.reach,hysteresis:f.policy.hysteresis},
   crossing:count(X),wetTiles:count(L),
   simulationFine:count(S),simulationOwners:f.owners,
   productionCensus:{interface:info.uniformMixedDynamicInterfaceTiles,band:info.uniformMixedDynamicBandTiles,solid:info.uniformMixedDynamicSolidTiles,required:info.uniformMixedDynamicRequiredTiles,boundary:info.uniformMixedDynamicBoundaryTiles,changed:info.uniformMixedDynamicChangedTiles},
   adoptedFine:info.uniformMixedFineTiles,pressureBandTiles:info.uniformPressureBandTiles,
   band:Object.fromEntries(Object.entries(masks).map(([k,m])=>[k,count(m)])),
   headers:header,
   travel_h:{threeFrames:travel.A,oneFrame:travel.C,crossingP50:pct(t=>!!X[t],.5),crossingP90:pct(t=>!!X[t],.9),crossingMax:pct(t=>!!X[t],1),wetP90:pct(t=>!!L[t],.9),wetMax:pct(t=>!!L[t],1),dryMax:pct(t=>!L[t],1),fastest},
   exclusive:{
    driftClosure_Z_minus_X:minus(Z,X),
    closureOnly_Z0_minus_X:minus(Z0,X),
    driftOnly_Z_minus_Z0:minus(Z,Z0),
    driftAtH1_C_minus_C0:minus(C,C0),
    oneStepTravel_C_minus_Z:minus(C,Z),
    gravityAtH1_B_minus_C:minus(B,C),
    multiFrameNoGravity_D_minus_C:minus(D,C),
    gravityAtH3_A_minus_D:minus(A,D),
    multiFrameTotal_A_minus_B:minus(A,B),
    zeroJoinAtH3_A_minus_E3:minus(A,E3),
    zeroJoinAtH1_C_minus_E1:minus(C,E1),
   },
   nesting:{Z0_not_in_Z:minus(Z0,Z),C0_not_in_C:minus(C0,C),X_not_in_Z0:minus(X,Z0),X_not_in_E1:minus(X,E1),Z_not_in_C:minus(Z,C),C_not_in_D:minus(C,D),D_not_in_A:minus(D,A),B_not_in_A:minus(B,A),E3_not_in_A:minus(E3,A),E1_not_in_C:minus(E1,C)},
   liquidSide:{A:liquidSplit(A),C:liquidSplit(C),D:liquidSplit(D)},
   simulationVsCensus:{S_not_in_A:minus(S,A),A_not_in_S:minus(A,S)},
  };
  assert.equal(row.band.A,row.productionCensus.band,'probe production policy must reproduce the production census');
  rows.push(row);console.log(JSON.stringify(row));
  await writeFile(out,JSON.stringify({scene:sceneId,tiles:T,dimensions:layout0.lattice.dimensions,captures,rows},null,2));
 }
 assert.deepEqual(errors,[]);
}finally{for(const o of owned)o.destroy();solver?.destroy();device?.destroy();}
