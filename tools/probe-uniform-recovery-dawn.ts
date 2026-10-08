/** Diagnostic-only local volume-recovery census (docs/plans/uniform-local-volume-recovery.md,
 * step 1). Per tile and step on the unchanged solver: fresh versus inherited
 * transport excess, repair volume, compressive normal strain of u*, wall
 * contact, and candidate activity detectors. Inherited excess is the previous
 * excess moved by this frame's own conservative weights: after the transport
 * gather, the probe gathers max(V_in-1,0) through the same rows and donor sums.
 * node --import tsx tools/probe-uniform-recovery-dawn.ts --frames=600 --out=/tmp/recovery.json
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
import {mixedCellWidth,type UniformMixedLayout} from '../lib/methods/uniform/uniform-mixed-layout';
import {UniformMixedTransportStage} from '../lib/methods/uniform/uniform-mixed-transport';
import {readMixedBuffer,readMixedTexture} from '../tests/helpers/uniform-mixed-native-fields';
const arg=(k:string,d:string)=>process.argv.find(a=>a.startsWith(`--${k}=`))?.slice(k.length+3)??d;
const frames=Number(arg('frames','600')),every=Number(arg('every','3')),out=arg('out','/tmp/recovery.json');
const sceneId=arg('scene','minimal-power-dam-break-64'),dt=Number(arg('dt',String(1/30))),tau=Number(arg('tau','0.2'));
const values=resolveMethodValues(uniformVolumeMethod,'balanced',JSON.parse(arg('values','{}')));
let device:GPUDevice|undefined,solver:WebGPUUniformReferenceSolver|undefined;
const owned:GPUTexture[]=[];
try{
 const dawn=await import(pathToFileURL(resolve('node_modules/webgpu/index.js')).href) as NodeDawnProvider;Object.assign(globalThis,dawn.globals);
 const gpu=createProcessRetainedDawnGPU(dawn,['backend=metal']),adapter=await gpu.requestAdapter();assert.ok(adapter);
 device=managedGPUDevice(await adapter.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter.limits)}),{requireWorkerRealm:false});
 const errors:string[]=[];device.addEventListener('uncapturederror',e=>{e.preventDefault();errors.push(e.error.message);});
 const scene=sceneDocument(getSceneDefinition(sceneId));
 scene.numerics.fixedDt_s=scene.numerics.maxDt_s=dt;
 if(arg('initial','dam')==='rest')scene.fluid.initialCondition='tank-fill';
 solver=await uniformVolumeMethod.createSolverAsync!(device,scene,'balanced',values,undefined,()=>{}) as WebGPUUniformReferenceSolver;
 const frame=(solver as any).mixedFrame,f=frame.fields,stage=frame.transport as any;
 // Local recovery policy (arms): --recovery='{"quietHalfLife":0.4,...}' reaches every advance.
 const recovery=JSON.parse(arg('recovery','null'));
 if(recovery){const advance=frame.advance.bind(frame);frame.advance=(p:any,...rest:any[])=>advance({...p,recovery:{...p.recovery,...recovery}},...rest);}
 // Capacity 1: the probe's scenes carry no static solids (solids are always compiled).
 const size=[f.volume.width,f.volume.height,f.volume.depthOrArrayLayers] as const;
 const texture=(label:string,format:GPUTextureFormat,usage:number)=>{const t=device!.createTexture({label,size,dimension:'3d',format,usage});owned.push(t);return t;};
 const COPY=GPUTextureUsage.COPY_SRC|GPUTextureUsage.COPY_DST;
 // e = max(V_in-1,0) on every texel; gather reads donor origin texels only.
 const excessIn=texture('Recovery excess in','r32float',GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.STORAGE_BINDING|COPY);
 const inherited=texture('Recovery inherited','r32float',GPUTextureUsage.STORAGE_BINDING|COPY);
 const excessModule=device.createShaderModule({code:`@group(0) @binding(0) var v:texture_3d<f32>;@group(0) @binding(1) var e:texture_storage_3d<r32float,write>;
@compute @workgroup_size(4,4,4) fn main(@builtin(global_invocation_id) g:vec3u){if(any(g>=textureDimensions(v))){return;}textureStore(e,vec3i(g),vec4f(max(textureLoad(v,vec3i(g),0).x-1.0,0.0)));}`});
 const excessPipeline=await device.createComputePipelineAsync({layout:'auto',compute:{module:excessModule,entryPoint:'main'}});
 const excessGroup=device.createBindGroup({layout:excessPipeline.getBindGroupLayout(0),entries:[{binding:0,resource:f.volumeScratch.createView()},{binding:1,resource:excessIn.createView()}]});
 // The frame's stage scratch at the ownership's h-tile capacity: edge rows by owner rank, donor sums, decoded sums.
 const scratch=frame.stageScratch as GPUBuffer,capacity=stage.ownership.capacity,ranges=UniformMixedTransportStage.scratchRanges(capacity.tiles,capacity.fineTiles);
 const gatherGroup=device.createBindGroup({layout:stage.resourcesLayout,entries:[
  {binding:0,resource:{buffer:stage.sampling}},
  {binding:1,resource:{buffer:scratch,...ranges.edges}},
  {binding:2,resource:{buffer:scratch,...ranges.donors}},
  {binding:3,resource:{buffer:scratch,...ranges.sums}},
  {binding:4,resource:excessIn.createView()},{binding:5,resource:inherited.createView()},
  {binding:6,resource:f.departure.createView()},
  ...(stage.sourceParams?[{binding:7,resource:{buffer:stage.sourceParams,size:176}}]:[]),
  {binding:8,resource:{buffer:stage.live}},
  // A throwaway evidence buffer: the probe's gather must not add to the frame's.
  ...(stage.recovery?[{binding:9,resource:{buffer:device.createBuffer({label:'Recovery probe evidence sink',size:stage.recovery.size,usage:GPUBufferUsage.STORAGE})}}]:[]),
 ]});
 const snapshots=new Map<string,GPUTexture>();
 const snapshot=(e:GPUCommandEncoder,key:string,source:GPUTexture)=>{
  let target=snapshots.get(key);if(!target){target=texture(`Recovery ${key}`,source.format,COPY);snapshots.set(key,target);}
  e.copyTextureToTexture({texture:source},{texture:target},size);
 };
 const wrap=(object:any,key:string,before:(e:GPUCommandEncoder)=>void,after:(e:GPUCommandEncoder)=>void)=>{
  const original=object[key].bind(object);object[key]=(...args:any[])=>{before(args[0]);const result=original(...args);after(args[0]);return result;};
 };
 wrap(stage,'encodeTransport',()=>{},e=>{
  snapshot(e,'transportIn',f.volume);snapshot(e,'transportOut',f.volumeScratch);
  const pass=e.beginComputePass({label:'Recovery probe excess'});pass.setPipeline(excessPipeline);pass.setBindGroup(0,excessGroup);
  pass.dispatchWorkgroups(Math.ceil(size[0]/4),Math.ceil(size[1]/4),Math.ceil(size[2]/4));pass.end();
  // The same rows, donor sums and live lists the V gather just read.
  const gather=e.beginComputePass({label:'Recovery probe inherited gather'});gather.setBindGroup(0,stage.topologyGroup);gather.setBindGroup(1,gatherGroup);if(stage.solid)gather.setBindGroup(2,stage.solid.bindGroup);
  stage.run(e,'gather',gather);gather.end();
  snapshot(e,'inherited',inherited);
 });
 wrap(frame.momentum,'encode',e=>{snapshot(e,'volume',f.volume);snapshot(e,'centerPhi',f.centerPhi);},()=>{});
 wrap(frame.forces,'encode',()=>{},e=>snapshot(e,'forced',f.velocityScratch));
 wrap(frame.band,'encodePrepare',e=>snapshot(e,'correction',f.correction),()=>{});
 wrap(frame.band,'encodeSolve',()=>{},e=>snapshot(e,'projected',f.velocity));
 const closed=scene.container.top==='open'?0b101111:0b111111;
 // Candidate detectors: trigger on fresh tile excess and compressive strain
 // (0 = fresh alone, 'wall' = fresh at a wall tile); activity decays over tau.
 const detectors:{name:string;fresh:number;strain:number|'wall'|'any'}[]=[];
 for(const fresh of [0.05,0.25])for(const strain of [0,2,5,10,'wall' as const,'any' as const])detectors.push({name:`F${fresh}-S${strain}`,fresh,strain});
 const strainOnly=[2,5,10];
 let activity:Float32Array[]=[],strainActivity:Float32Array[]=[];
 const rows:any[]=[];const h=scene.voxelDomain.finestCellSize_m;
 for(let step=1;step<=frames;step++){
  const layout:UniformMixedLayout=frame.ownership.layout;
  assert.ok(solver.advanceTo(step*dt));await solver.awaitFrameCompletion();
  const d:Record<string,Float32Array>={};for(const [k,t]of snapshots)d[k]=await readMixedTexture(device,t);
  const recoveryActivity:Float32Array|undefined=frame.recovery?(await readMixedBuffer(device,frame.recovery.buffer)).subarray(0,frame.ownership.layout.tiles.length):undefined;
  const [nx,ny]=layout.lattice.dimensions,[tx,ty,tz]=layout.tileDimensions,T=layout.tiles.length;
  if(!activity.length){activity=detectors.map(()=>new Float32Array(T));strainActivity=strainOnly.map(()=>new Float32Array(T));}
  const E=new Float32Array(T),F=new Float32Array(T),I=new Float32Array(T),C=new Float32Array(T),L=new Float32Array(T),S=new Float32Array(T),W=new Uint8Array(T);
  const width=(t:number)=>mixedCellWidth(layout.tiles[t]!);
  // Mean positive-face velocity of tile t's high face on axis a (probe
  // settling rule: a coarse face stores one texel per min(width,neighbour) patch).
  const highFace=(t:number,a:number)=>{
   const c=[t%tx,Math.floor(t/tx)%ty,Math.floor(t/(tx*ty))],dims=[tx,ty,tz],w=width(t);
   const n=[...c];n[a]!+=1;const nw=n[a]!<dims[a]!?width(n[0]!+tx*(n[1]!+ty*n[2]!)):w,patch=Math.min(w,nw),u=(a+1)%3,b=(a+2)%3;
   let sum=0;for(let du=0;du<4;du+=patch)for(let dv=0;dv<4;dv+=patch){const p=[4*c[0]!,4*c[1]!,4*c[2]!];p[a]!+=3;p[u]!+=du;p[b]!+=dv;sum+=d.forced![4*(p[0]!+nx*(p[1]!+ny*p[2]!))+a]!;}
   return sum*patch*patch/16;
  };
  for(let t=0;t<T;t++){
   const w=width(t),c=[t%tx,Math.floor(t/tx)%ty,Math.floor(t/(tx*ty))],dims=[tx,ty,tz];
   for(let dz=0;dz<4;dz+=w)for(let dy=0;dy<4;dy+=w)for(let dx=0;dx<4;dx+=w){
    const i=4*c[0]!+dx+nx*(4*c[1]!+dy+ny*(4*c[2]!+dz)),m=w**3;
    const out=Math.max(d.transportOut![i]!-1,0),inh=d.inherited![i]!;
    E[t]!+=Math.max(d.volume![i]!-1,0)*m;F[t]!+=Math.max(0,out-inh)*m;I[t]!+=Math.min(out,inh)*m;
    C[t]!+=Math.abs(d.correction![i]!)*dt*m;L[t]!+=Math.max(0,Math.min(1,d.volume![i]!))*m;
   }
   if(L[t]!<0.5)continue;
   let worst=0;
   for(let a=0;a<3;a++){
    const lowSide=c[a]===0?((closed>>a)&1?0:NaN):highFace(t-[1,tx,tx*ty][a]!,a),high=c[a]===dims[a]!-1?((closed>>(a+3))&1?0:NaN):highFace(t,a);
    if(c[a]===0||c[a]===dims[a]!-1)W[t]=1;
    if(Number.isFinite(lowSide)&&Number.isFinite(high))worst=Math.max(worst,-(high-lowSide)/(4*h));
   }
   S[t]=worst;
  }
  // Activity: immediate on trigger, exponential release over tau.
  const release=Math.exp(-dt/tau);
  const neighbourhood=(t:number,test:(q:number)=>boolean)=>{
   const c=[t%tx,Math.floor(t/tx)%ty,Math.floor(t/(tx*ty))];
   for(let z=Math.max(0,c[2]!-1);z<=Math.min(tz-1,c[2]!+1);z++)for(let y=Math.max(0,c[1]!-1);y<=Math.min(ty-1,c[1]!+1);y++)for(let x=Math.max(0,c[0]!-1);x<=Math.min(tx-1,c[0]!+1);x++)if(test(x+tx*(y+ty*z)))return true;
   return false;
  };
  const totals=(a:Float32Array)=>{let e=0,cv=0,f=0,n=0;for(let t=0;t<T;t++)if(a[t]!>0.5){e+=E[t]!;cv+=C[t]!;f+=F[t]!;n++;}return {tiles:n,excess:e,correction:cv,fresh:f};};
  const detectorRows:Record<string,any>={};
  detectors.forEach((det,k)=>{
   const a=activity[k]!;const trig=new Uint8Array(T);
   for(let t=0;t<T;t++){const s=det.strain;if(F[t]!>=det.fresh&&(s==='any'||(s==='wall'?W[t]===1:S[t]!>=s)))trig[t]=1;}
   for(let t=0;t<T;t++)a[t]=neighbourhood(t,q=>trig[q]===1)?1:a[t]!*release;
   detectorRows[det.name]=totals(a);
  });
  strainOnly.forEach((s,k)=>{const a=strainActivity[k]!;for(let t=0;t<T;t++)a[t]=neighbourhood(t,q=>S[q]!>=s)?1:a[t]!*release;detectorRows[`S${s}`]=totals(a);});
  if(!(step%every===0||step<=3))continue;
  // Motion and quality: owner-volume-weighted projected speed RMS (positive faces at
  // the owner's far corner, a proxy), total V, largest V, wall climb (highest
  // cell with V>0.5 in the four-cell slabs at the x walls).
  let ke=0,mass=0,maxV=0,climbLow=0,climbHigh=0;
  for(let t=0;t<T;t++){const w=width(t),c=[t%tx,Math.floor(t/tx)%ty,Math.floor(t/(tx*ty))];
   for(let dz=0;dz<4;dz+=w)for(let dy=0;dy<4;dy+=w)for(let dx=0;dx<4;dx+=w){
    const x=4*c[0]!+dx,y=4*c[1]!+dy,z=4*c[2]!+dz,i=x+nx*(y+ny*z),v=d.volume![i]!,m=Math.max(0,v)*w**3;
    mass+=v*w**3;maxV=Math.max(maxV,v);
    const far=(x+w-1)+nx*((y+w-1)+ny*(z+w-1));let u2=0;for(let a=0;a<3;a++)u2+=d.projected![4*far+a]!**2;ke+=Math.min(v,1)>0?Math.min(Math.max(v,0),1)*w**3*u2:0;
    if(v>0.5){if(x<4)climbLow=Math.max(climbLow,(y+w)*h);if(x+w>nx-4)climbHigh=Math.max(climbHigh,(y+w)*h);}
   }}
  const activeTiles=recoveryActivity?(()=>{let n=0,e=0;for(let t=0;t<T;t++)if(recoveryActivity![t]!>0.5){n++;e+=E[t]!;}return {tiles:n,excess:e};})():undefined;
  const sum=(a:ArrayLike<number>)=>{let s=0;for(let i=0;i<a.length;i++)s+=a[i]!;return s;};
  const liquidStrain:number[]=[],excessStrain:number[]=[];for(let t=0;t<T;t++){if(L[t]!>=0.5)liquidStrain.push(S[t]!);if(E[t]!>0.05)excessStrain.push(S[t]!);}
  const q=(a:number[],p:number)=>{if(!a.length)return 0;const s=[...a].sort((x,y)=>x-y);return s[Math.min(s.length-1,Math.floor(p*s.length))]!;};
  let wallExcess=0,freshWall=0;for(let t=0;t<T;t++)if(W[t]){wallExcess+=E[t]!;freshWall+=F[t]!;}
  // Excess age: share of excess in tiles with no fresh excess this step.
  let staleExcess=0;for(let t=0;t<T;t++)if(F[t]!<0.01)staleExcess+=E[t]!;
  const row={step,t:step*dt,rms:Math.sqrt(ke/Math.max(sum(L),1e-9)),mass,maxV,climbLow,climbHigh,activeTiles,liquid:sum(L),excess:sum(E),fresh:sum(F),inherited:sum(I),correction:sum(C),wallExcess,freshWall,staleExcess,
   strain:{liquid:[0.5,0.9,0.99].map(p=>q(liquidStrain,p)),excess:[0.5,0.9,0.99].map(p=>q(excessStrain,p))},detectors:detectorRows};
  rows.push(row);if(step%30===0||step<=3)console.log(JSON.stringify({step,t:row.t.toFixed(2),rms:row.rms.toFixed(3),mass:row.mass.toFixed(1),maxV:row.maxV.toFixed(2),climb:[row.climbLow.toFixed(3),row.climbHigh.toFixed(3)],activeTiles,excess:row.excess.toFixed(1),fresh:row.fresh.toFixed(2),inherited:row.inherited.toFixed(1),correction:row.correction.toFixed(2),stale:row.staleExcess.toFixed(1),strain:row.strain,active:Object.fromEntries(Object.entries(detectorRows).filter(([k])=>/F0.05-S(0|5|wall)$|^S5$/.test(k)).map(([k,v])=>[k,[v.tiles,v.excess.toFixed(1),v.correction.toFixed(2)]]))}));
  assert.deepEqual(errors,[]);
 }
 await writeFile(out,JSON.stringify({sceneId,initial:arg('initial','dam'),dt,tau,values,recovery,detectors,rows},null,1));
}finally{solver?.destroy();for(const t of owned)t.destroy();device?.destroy();}
