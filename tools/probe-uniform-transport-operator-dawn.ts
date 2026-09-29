/** Diagnostic-only frozen transport-operator census (docs/plans/uniform-local-volume-recovery.md,
 * "isolate the transport operator"). At chosen frames, right after the real
 * transport, the probe replays that frame's balancing on the same ownership,
 * departures and incoming V (clear, build, R rounds, gather) for several round
 * counts R, on full support: every tile is listed as a row and a donor, so no
 * compact dependency closure limits R. Each replay gathers the capacity field
 * C (B·C, whose defect B·C−C is the receiver-capacity error), the incoming V
 * (B·V) and its bounded part min(V,C) (new overfill from bounded input).
 * Unsolided scenes only: capacity is one per unit volume.
 * node --import tsx tools/probe-uniform-transport-operator-dawn.ts --steps=18,30,360 --out=/tmp/operator.json
 */
import assert from 'node:assert/strict';
import {writeFile} from 'node:fs/promises';
import {resolve} from 'node:path';
import {pathToFileURL} from 'node:url';
import {createProcessRetainedDawnGPU,type NodeDawnProvider} from '../lib/harness/node-dawn-provider';
import {acquireWebGPUExclusiveLock,releaseWebGPUExclusiveLock,readWebGPUExclusiveLockHolder} from '../lib/harness/webgpu-smoke-isolation';
import {managedGPUDevice,gpuCompilationManagerFor} from '../lib/core/gpu-compilation-manager';
import {requiredFluidDeviceLimits} from '../lib/core/webgpu-device-limits';
import {sceneDocument} from '../lib/core/scene-definition';
import {getSceneDefinition} from '../lib/core/scenes';
import {resolveMethodValues} from '../lib/core/method-contract';
import {uniformVolumeMethod} from '../lib/methods/uniform/uniform-volume-method';
import type {WebGPUUniformReferenceSolver} from '../lib/methods/uniform/webgpu-uniform-reference';
import {mixedCellWidth,type UniformMixedLayout} from '../lib/methods/uniform/uniform-mixed-layout';
import {UNIFORM_MIXED_TRANSPORT_LIVE_HEADER} from '../lib/methods/uniform/uniform-mixed-transport.wgsl';
import {readMixedTexture} from '../tests/helpers/uniform-mixed-native-fields';
const arg=(k:string,d:string)=>process.argv.find(a=>a.startsWith(`--${k}=`))?.slice(k.length+3)??d;
const out=arg('out','/tmp/operator.json'),sceneId=arg('scene','minimal-power-dam-break-64'),dt=Number(arg('dt',String(1/30)));
const steps=arg('steps','18,30,360').split(',').map(Number),rounds=arg('rounds','1,2,3,4,6,8,12,16').split(',').map(Number);
const fields=['capacity','volume','bounded'] as const;
const values=resolveMethodValues(uniformVolumeMethod,'balanced',JSON.parse(arg('values','{}')));
console.log('Waiting for repository WebGPU lease');
while(await readWebGPUExclusiveLockHolder()) await new Promise(r=>setTimeout(r,500));
await acquireWebGPUExclusiveLock('dawn-probe','uniform transport operator');
let device:GPUDevice|undefined,solver:WebGPUUniformReferenceSolver|undefined;
const owned:(GPUTexture|GPUBuffer)[]=[];
try{
 const dawn=await import(pathToFileURL(resolve('node_modules/webgpu/index.js')).href) as NodeDawnProvider;Object.assign(globalThis,dawn.globals);
 const gpu=createProcessRetainedDawnGPU(dawn,['backend=metal']),adapter=await gpu.requestAdapter();assert.ok(adapter);
 device=managedGPUDevice(await adapter.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter.limits)}),{requireWorkerRealm:false});
 const errors:string[]=[];device.addEventListener('uncapturederror',e=>{e.preventDefault();errors.push(e.error.message);});
 // --control=no-correction: local volume repair requests nothing (settling probe's control).
 let replacements=0;const control=arg('control','none');
 if(control==='no-correction'){const compiler=gpuCompilationManagerFor(device),create=compiler.createShaderModule.bind(compiler);
  compiler.createShaderModule=d=>create({...d,code:d.code.replace('return min(uvVolumeCorrectionFractionAt(dt)*max(0.0,v-cap),cap);',()=>{replacements++;return 'return 0.0;';})});}
 const scene=sceneDocument(getSceneDefinition(sceneId));
 scene.numerics.fixedDt_s=scene.numerics.maxDt_s=dt;
 if(arg('initial','dam')==='rest')scene.fluid.initialCondition='tank-fill';
 solver=await uniformVolumeMethod.createSolverAsync!(device,scene,'balanced',values,undefined,()=>{}) as WebGPUUniformReferenceSolver;
 if(control==='no-correction')assert.ok(replacements>0);
 const frame=(solver as any).mixedFrame,f=frame.fields,stage=frame.transport as any;
 const size=[f.volume.width,f.volume.height,f.volume.depthOrArrayLayers] as const;
 const COPY=GPUTextureUsage.COPY_SRC|GPUTextureUsage.COPY_DST;
 const texture=(label:string,usage:number)=>{const t=device!.createTexture({label,size,dimension:'3d',format:'r32float',usage});owned.push(t);return t;};
 const ones=texture('Operator capacity',GPUTextureUsage.TEXTURE_BINDING|COPY);
 device.queue.writeTexture({texture:ones},new Float32Array(size[0]*size[1]*size[2]).fill(1),{bytesPerRow:4*size[0],rowsPerImage:size[1]},size);
 const incoming=texture('Operator incoming V',GPUTextureUsage.TEXTURE_BINDING|COPY);
 const bounded=texture('Operator bounded V',GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.STORAGE_BINDING|COPY);
 const actual=texture('Operator actual transport',COPY);
 const traceVelocity=device.createTexture({label:'Operator trace velocity',size,dimension:'3d',format:f.velocityScratch.format,usage:COPY});owned.push(traceVelocity);
 const departure=device.createTexture({label:'Operator departures',size,dimension:'3d',format:f.departure.format,usage:COPY|GPUTextureUsage.TEXTURE_BINDING});owned.push(departure);
 const outputs=new Map<string,GPUTexture>();
 for(const r of rounds)for(const k of fields)outputs.set(`${r}:${k}`,texture(`Operator ${k} R${r}`,GPUTextureUsage.STORAGE_BINDING|COPY));
 const inputs:Record<typeof fields[number],GPUTexture>={capacity:ones,volume:incoming,bounded};
 const boundModule=device.createShaderModule({code:`@group(0) @binding(0) var v:texture_3d<f32>;@group(0) @binding(1) var b:texture_storage_3d<r32float,write>;
@compute @workgroup_size(4,4,4) fn main(@builtin(global_invocation_id) g:vec3u){if(any(g>=textureDimensions(v))){return;}textureStore(b,vec3i(g),vec4f(clamp(textureLoad(v,vec3i(g),0).x,0.0,1.0)));}`});
 const boundPipeline=await device.createComputePipelineAsync({layout:'auto',compute:{module:boundModule,entryPoint:'main'}});
 const boundGroup=device.createBindGroup({layout:boundPipeline.getBindGroupLayout(0),entries:[{binding:0,resource:incoming.createView()},{binding:1,resource:bounded.createView()}]});
 // Full support: every tile a row and a donor, in its tier's lists.
 const fullLive=device.createBuffer({label:'Operator full-support live set',size:stage.live.size,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});owned.push(fullLive);
 const cells=stage.cells as number,arena=f.arena;
 const sink=stage.recovery?device.createBuffer({label:'Operator evidence sink',size:stage.recovery.size,usage:GPUBufferUsage.STORAGE}):undefined;if(sink)owned.push(sink);
 const group=(input:GPUTexture,output:GPUTexture,dep:GPUTexture=f.departure)=>device!.createBindGroup({layout:stage.resourcesLayout,entries:[
  {binding:0,resource:{buffer:stage.sampling}},
  {binding:1,resource:{buffer:arena.buffer,offset:0,size:cells*40}},
  {binding:2,resource:{buffer:arena.buffer,offset:arena.donorOffset,size:cells*12}},
  {binding:3,resource:{buffer:arena.buffer,offset:arena.donorOffset+cells*12,size:cells*4}},
  {binding:4,resource:input.createView()},{binding:5,resource:output.createView()},
  {binding:6,resource:dep.createView()},
  ...(stage.sourceParams?[{binding:7,resource:{buffer:stage.sourceParams,size:176}}]:[]),
  {binding:8,resource:{buffer:fullLive}},
  ...(sink?[{binding:9,resource:{buffer:sink}}]:[]),
 ]});
 const groups=new Map<string,GPUBindGroup>();for(const r of rounds)for(const k of fields)groups.set(`${r}:${k}`,group(inputs[k],outputs.get(`${r}:${k}`)!));
 // Alternative departure maps, built on the GPU inside the frozen frame's
 // encoder (the stage's scratch state is not reusable after the frame).
 const RV=rounds.includes(3)?3:rounds[0]!;
 const variantSpecs:{name:string;mode?:number;k?:number;from?:string}[]=[{name:'base'},{name:'smooth',mode:0,from:'base'},{name:'smooth2',mode:0,from:'smooth'},{name:'travel×0.5',mode:1,k:0.5,from:'base'},{name:'travel×0.25',mode:1,k:0.25,from:'base'}];
 const widthTex=texture('Operator owner width',GPUTextureUsage.TEXTURE_BINDING|COPY);
 const depTex=new Map<string,GPUTexture>([['base',departure]]);
 for(const v of variantSpecs.slice(1)){const t=device.createTexture({label:`Operator departures ${v.name}`,size,dimension:'3d',format:'rgba32float',usage:GPUTextureUsage.STORAGE_BINDING|GPUTextureUsage.TEXTURE_BINDING|COPY});owned.push(t);depTex.set(v.name,t);}
 const mapModule=device.createShaderModule({code:`@group(0) @binding(0) var src:texture_3d<f32>;@group(0) @binding(1) var widths:texture_3d<f32>;@group(0) @binding(2) var dst:texture_storage_3d<rgba32float,write>;
override mode:u32;override k:f32=1.0;
@compute @workgroup_size(4,4,4) fn main(@builtin(global_invocation_id) g:vec3u){let n=vec3i(textureDimensions(src));let p=vec3i(g);if(any(p>=n)){return;}
 let w=textureLoad(widths,p,0).x;let d=textureLoad(src,p,0);if(w==0.0){textureStore(dst,p,d);return;}
 let centre=vec3f(p)+vec3f(0.5*w);
 if(mode==1u){textureStore(dst,p,vec4f(centre+k*(d.xyz-centre),d.w));return;}
 if(w!=1.0){textureStore(dst,p,d);return;}
 var acc=vec3f(0.0);var count=0.0;
 for(var z=-1;z<=1;z++){for(var y=-1;y<=1;y++){for(var x=-1;x<=1;x++){let q=p+vec3i(x,y,z);if(any(q<vec3i(0))||any(q>=n)){continue;}if(textureLoad(widths,q,0).x!=1.0){continue;}
  acc+=textureLoad(src,q,0).xyz-(vec3f(q)+vec3f(0.5));count+=1.0;}}}
 textureStore(dst,p,vec4f(centre+acc/count,d.w));}`});
 const mapPipes=new Map<string,GPUComputePipeline>();
 for(const v of variantSpecs.slice(1))mapPipes.set(v.name,await device.createComputePipelineAsync({layout:'auto',compute:{module:mapModule,entryPoint:'main',constants:{mode:v.mode!,k:v.k??1}}}));
 const mapGroups=new Map(variantSpecs.slice(1).map(v=>[v.name,device!.createBindGroup({layout:mapPipes.get(v.name)!.getBindGroupLayout(0),entries:[{binding:0,resource:depTex.get(v.from!)!.createView()},{binding:1,resource:widthTex.createView()},{binding:2,resource:depTex.get(v.name)!.createView()}]})]));
 const variantOut=new Map<string,GPUTexture>(),variantGroups=new Map<string,GPUBindGroup>();
 for(const v of variantSpecs)for(const k of fields){const t=texture(`Operator ${k} ${v.name}`,GPUTextureUsage.STORAGE_BINDING|COPY);variantOut.set(`${v.name}:${k}`,t);variantGroups.set(`${v.name}:${k}`,group(inputs[k],t,depTex.get(v.name)!));}
 let freeze=false,frozenLayout:UniformMixedLayout|undefined;
 const original=stage.encodeTransport.bind(stage);
 stage.encodeTransport=(e:GPUCommandEncoder)=>{
  if(freeze){e.copyTextureToTexture({texture:f.volumeScratch},{texture:incoming},size);}
  original(e);
  if(!freeze)return;
  const layout:UniformMixedLayout=frame.ownership.layout;frozenLayout=layout;
  const T=layout.tiles.length,H=UNIFORM_MIXED_TRANSPORT_LIVE_HEADER,live=new Uint32Array(H+7*T),tiers:number[][]=[[],[]];
  for(let t=0;t<T;t++)tiers[mixedCellWidth(layout.tiles[t]!)===1?0:1]!.push(t);
  for(const list of [1,2])for(const tier of [0,1]){live[12+2*(list-1)+tier]=tiers[tier]!.length;live.set(tiers[tier]!,H+(3+2*(list-1)+tier)*T);}
  device!.queue.writeBuffer(fullLive,0,live);
  e.copyTextureToTexture({texture:f.volume},{texture:actual},size);e.copyTextureToTexture({texture:f.departure},{texture:departure},size);e.copyTextureToTexture({texture:f.velocityScratch},{texture:traceVelocity},size);
  const b=e.beginComputePass({label:'Operator bounded input'});b.setPipeline(boundPipeline);b.setBindGroup(0,boundGroup);b.dispatchWorkgroups(Math.ceil(size[0]/4),Math.ceil(size[1]/4),Math.ceil(size[2]/4));b.end();
  for(const r of rounds){
   const pass=e.beginComputePass({label:`Operator replay R${r}`});pass.setBindGroup(0,stage.topologyGroup);pass.setBindGroup(1,groups.get(`${r}:capacity`)!);
   if(stage.solid)pass.setBindGroup(2,stage.solid.bindGroup);
   stage.run(e,'clear',pass);stage.run(e,'build',pass);
   for(let round=0;round<r;round++){stage.run(e,round===0?'rowsFallback':'rowsDivide',pass);stage.run(e,'decode',pass);}
   for(const k of fields){pass.setBindGroup(1,groups.get(`${r}:${k}`)!);stage.run(e,'gather',pass);}
   pass.end();
  }
  const widths=new Float32Array(size[0]*size[1]*size[2]),[lx,ly]=layout.lattice.dimensions,[ttx,tty]=layout.tileDimensions;
  for(let t=0;t<T;t++){const w=mixedCellWidth(layout.tiles[t]!),c=[t%ttx,Math.floor(t/ttx)%tty,Math.floor(t/(ttx*tty))];
   for(let dz=0;dz<4;dz+=w)for(let dy=0;dy<4;dy+=w)for(let dx=0;dx<4;dx+=w)widths[4*c[0]!+dx+lx*(4*c[1]!+dy+ly*(4*c[2]!+dz))]=w;}
  device!.queue.writeTexture({texture:widthTex},widths,{bytesPerRow:4*size[0],rowsPerImage:size[1]},size);
  for(const v of variantSpecs.slice(1)){const m=e.beginComputePass({label:`Operator departure ${v.name}`});m.setPipeline(mapPipes.get(v.name)!);m.setBindGroup(0,mapGroups.get(v.name)!);m.dispatchWorkgroups(Math.ceil(size[0]/4),Math.ceil(size[1]/4),Math.ceil(size[2]/4));m.end();}
  for(const v of variantSpecs){
   const pass=e.beginComputePass({label:`Operator variant ${v.name}`});pass.setBindGroup(0,stage.topologyGroup);pass.setBindGroup(1,variantGroups.get(`${v.name}:capacity`)!);
   if(stage.solid)pass.setBindGroup(2,stage.solid.bindGroup);
   stage.run(e,'clear',pass);stage.run(e,'build',pass);
   for(let round=0;round<RV;round++){stage.run(e,round===0?'rowsFallback':'rowsDivide',pass);stage.run(e,'decode',pass);}
   for(const k of fields){pass.setBindGroup(1,variantGroups.get(`${v.name}:${k}`)!);stage.run(e,'gather',pass);}
   pass.end();
  }
 };
 const results:any[]=[];const last=Math.max(...steps);
 for(let step=1;step<=last;step++){
  freeze=steps.includes(step);
  assert.ok(solver.advanceTo(step*dt));await solver.awaitFrameCompletion();
  assert.deepEqual(errors,[]);
  if(!freeze)continue;
  const layout=frozenLayout!,[nx,ny]=layout.lattice.dimensions,[tx,ty]=layout.tileDimensions;
  const vin=await readMixedTexture(device,incoming),act=await readMixedTexture(device,actual);
  // Owners of the frozen layout: origin texel and width.
  const owners:{i:number;w:number}[]=[];
  for(let t=0;t<layout.tiles.length;t++){const w=mixedCellWidth(layout.tiles[t]!),c=[t%tx,Math.floor(t/tx)%ty,Math.floor(t/(tx*ty))];
   for(let dz=0;dz<4;dz+=w)for(let dy=0;dy<4;dy+=w)for(let dx=0;dx<4;dx+=w)owners.push({i:4*c[0]!+dx+nx*(4*c[1]!+dy+ny*(4*c[2]!+dz)),w});}
  let massIn=0,excessIn=0,capacity=0;for(const {i,w} of owners){massIn+=vin[i]!*w**3;excessIn+=Math.max(vin[i]!-1,0)*w**3;capacity+=w**3;}
  const perRound:any[]=[];
  for(const r of rounds){
   const bc=await readMixedTexture(device,outputs.get(`${r}:capacity`)!),bv=await readMixedTexture(device,outputs.get(`${r}:volume`)!),bb=await readMixedTexture(device,outputs.get(`${r}:bounded`)!);
   const s={rounds:r,capacityOut:0,positiveDefect:0,negativeDefect:0,maxDefect:0,defectOwners:[0,0,0],positiveDefectByTier:[0,0],massOut:0,excessOut:0,boundedExcess:0,maxOut:0,actualDiff:0};
   for(const {i,w} of owners){const m=w**3,d=bc[i]!-1;
    s.capacityOut+=bc[i]!*m;if(d>0){s.positiveDefect+=d*m;s.positiveDefectByTier[w===1?0:1]+=d*m;}else s.negativeDefect+=d*m;
    s.maxDefect=Math.max(s.maxDefect,d);[1e-3,1e-2,1e-1].forEach((x,k)=>{if(d>x)s.defectOwners[k]!++;});
    s.massOut+=bv[i]!*m;s.excessOut+=Math.max(bv[i]!-1,0)*m;s.boundedExcess+=Math.max(bb[i]!-1,0)*m;s.maxOut=Math.max(s.maxOut,bv[i]!);
    if(r===3)s.actualDiff=Math.max(s.actualDiff,Math.abs(bv[i]!-act[i]!));}
   perRound.push(s);
  }
  // Where the defect lives, at R=3 (production) and the largest R: class =
  // phase of V_in (liquid ≥ 0.5, partial, air) × a closed wall within one cell
  // × a tile face shared with a tile of the other width (h/4h seam).
  const dep=await readMixedTexture(device,departure),[nz]=[layout.lattice.dimensions[2]!],dims=[nx,ny,nz],tdims=layout.tileDimensions;
  const closed=scene.container.top==='open'?0b101111:0b111111,R3=rounds.includes(3)?3:rounds[0]!,RL=rounds.at(-1)!;
  const bcOf=new Map<number,Float32Array>(),bbOf=new Map<number,Float32Array>();
  for(const r of new Set([R3,RL])){bcOf.set(r,await readMixedTexture(device,outputs.get(`${r}:capacity`)!));bbOf.set(r,await readMixedTexture(device,outputs.get(`${r}:bounded`)!));}
  const widthAt=(c:number[])=>mixedCellWidth(layout.tiles[c[0]!+tdims[0]!*(c[1]!+tdims[1]!*c[2]!)]!);
  const classes:Record<string,{owners:number;defect3:number;defectL:number;bounded3:number;boundedL:number}>={};
  const top:any[]=[];
  for(const {i,w} of owners){
   const p=[i%nx,Math.floor(i/nx)%ny,Math.floor(i/(nx*ny))],v=vin[i]!,m=w**3;
   let wall=false;for(let a=0;a<3;a++){if(p[a]!<=0&&((closed>>a)&1))wall=true;if(p[a]!+w>=dims[a]!&&((closed>>(a+3))&1))wall=true;}
   const tc=p.map(x=>Math.floor(x/4));let seam=false;
   for(let a=0;a<3;a++)for(const s of [-1,1]){const q=[...tc];q[a]!+=s;if(q[a]!<0||q[a]!>=tdims[a]!)continue;if(widthAt(q)!==w)seam=true;}
   const key=`${v>=0.5?'liquid':v>0?'partial':'air'}${wall?'+wall':''}${seam?'+seam':''}${w===1?'':'+4h'}`;
   const c=classes[key]??={owners:0,defect3:0,defectL:0,bounded3:0,boundedL:0};c.owners++;
   c.defect3+=Math.max(bcOf.get(R3)![i]!-1,0)*m;c.defectL+=Math.max(bcOf.get(RL)![i]!-1,0)*m;
   c.bounded3+=Math.max(bbOf.get(R3)![i]!-1,0)*m;c.boundedL+=Math.max(bbOf.get(RL)![i]!-1,0)*m;
   const b3=Math.max(bbOf.get(R3)![i]!-1,0)*m;
   if(b3>0.05)top.push({p,w,vin:+v.toFixed(3),bounded3:+b3.toFixed(3),bc3:+bcOf.get(R3)![i]!.toFixed(3),bcL:+bcOf.get(RL)![i]!.toFixed(3),travel:[0,1,2].map(a=>+(dep[4*i+a]!-(p[a]!+w/2)).toFixed(2)),wall,seam});
  }
  top.sort((a,b)=>b.bounded3-a.bounded3);
  // Donor coverage: raw column sum Σ_i |box_i ∩ d| / |d| of the receiver
  // departure boxes. A donor touched by a sliver of one box still hands its
  // whole capacity to that box after column normalization.
  const N=nx*ny*nz,ownerWidth=new Uint8Array(N);for(const {i,w} of owners){const p=[i%nx,Math.floor(i/nx)%ny,Math.floor(i/(nx*ny))];for(let z=0;z<w;z++)for(let y=0;y<w;y++)for(let x=0;x<w;x++)ownerWidth[p[0]!+x+nx*(p[1]!+y+ny*(p[2]!+z))]=w;}
  const bins=[0,0.1,0.5,0.9,1.1,2,Infinity];
  const coverage=(d:Float32Array)=>{const cov=new Float32Array(N);
   for(const {i,w} of owners){const lo=[0,1,2].map(a=>d[4*i+a]!-w/2);
    const a0=lo.map(x=>Math.max(0,Math.floor(x))),a1=lo.map((x,a)=>Math.min(dims[a]!-1,Math.ceil(x+w)-1));
    for(let z=a0[2]!;z<=a1[2]!;z++){const oz=Math.min(lo[2]!+w,z+1)-Math.max(lo[2]!,z);if(oz<=0)continue;
     for(let y=a0[1]!;y<=a1[1]!;y++){const oy=Math.min(lo[1]!+w,y+1)-Math.max(lo[1]!,y);if(oy<=0)continue;
      for(let x=a0[0]!;x<=a1[0]!;x++){const ox=Math.min(lo[0]!+w,x+1)-Math.max(lo[0]!,x);if(ox>0)cov[x+nx*(y+ny*z)]+=ox*oy*oz;}}}}
   // Liquid (bounded) mass and owner count per coverage bin; per owner the mean over its texels.
   const mass=bins.slice(1).map(()=>0),count=bins.slice(1).map(()=>0);
   for(const {i,w} of owners){const p=[i%nx,Math.floor(i/nx)%ny,Math.floor(i/(nx*ny))];let c=0;for(let z=0;z<w;z++)for(let y=0;y<w;y++)for(let x=0;x<w;x++)c+=cov[p[0]!+x+nx*(p[1]!+y+ny*(p[2]!+z))]!;c/=w**3;
    const b=bins.findIndex((x,k)=>k>0&&c<x)-1;mass[b]!+=Math.min(Math.max(vin[i]!,0),1)*w**3;count[b]!++;}
   return {mass:mass.map(x=>+x.toFixed(1)),count};};
  const baseCoverage=coverage(dep);
  // Departure-map compression: det J ≈ 1 + div(displacement), and column
  // normalization hands a receiver about det J capacity, so the R3 defect
  // should track div(dep − centre) where the flow (trace velocity) converges.
  // h owners with six h neighbours; deep = self and six neighbours V ≥ 0.5.
  const divergence:Record<string,{n:number;sd:number;sdd:number;sb:number;sbb:number;sdb:number;divPos:number;defectPos:number}>={};
  const divergenceOf=(d:Float32Array,i:number,p:number[])=>{let div=0;for(let a=0;a<3;a++){const e=a===0?1:a===1?nx:nx*ny;div+=((d[4*(i+e)+a]!-(p[a]!+1.5))-(d[4*(i-e)+a]!-(p[a]!-0.5)))/2;}return div;};
  for(const {i,w} of owners){if(w!==1)continue;const p=[i%nx,Math.floor(i/nx)%ny,Math.floor(i/(nx*ny))];
   if(p.some((c,a)=>c<1||c>=dims[a]!-1))continue;const nb=[1,-1,nx,-nx,nx*ny,-nx*ny].map(e=>i+e);if(nb.some(j=>ownerWidth[j]!==1))continue;
   const div=divergenceOf(dep,i,p),defect=bcOf.get(R3)![i]!-1,v=vin[i]!;
   const key=v>=0.5?(nb.every(j=>vin[j]!>=0.5)?'deep':'surface'):v>0?'partial':'air';
   const c=divergence[key]??={n:0,sd:0,sdd:0,sb:0,sbb:0,sdb:0,divPos:0,defectPos:0};c.n++;c.sd+=div;c.sdd+=div*div;c.sb+=defect;c.sbb+=defect*defect;c.sdb+=div*defect;c.divPos+=Math.max(div,0);c.defectPos+=Math.max(defect,0);}
  // The traced field's own MAC divergence (velocityScratch, the extension the
  // RK2 trace samples), in displacement units dt/h: face index c on an axis is
  // the upper face of cell c (convention 'upper'); 'lower' is the other one.
  const vel=await readMixedTexture(device,traceVelocity),hm=layout.lattice.cellSize_m;
  const mac:Record<string,{n:number;upper:number;lower:number;disp:number;su:number;sdu:number;sdd:number}>={};
  for(const {i,w} of owners){if(w!==1)continue;const p=[i%nx,Math.floor(i/nx)%ny,Math.floor(i/(nx*ny))];
   if(p.some((c,a)=>c<1||c>=dims[a]!-1))continue;const nb=[1,-1,nx,-nx,nx*ny,-nx*ny].map(e=>i+e);if(nb.some(j=>ownerWidth[j]!==1))continue;
   let up=0,lo=0;for(let a=0;a<3;a++){const e=a===0?1:a===1?nx:nx*ny,k=dt/hm[a]!;up+=k*(vel[4*i+a]!-vel[4*(i-e)+a]!);lo+=k*(vel[4*(i+e)+a]!-vel[4*i+a]!);}
   const v=vin[i]!,key=v>=0.5?(nb.every(j=>vin[j]!>=0.5)?'deep':'surface'):v>0?'partial':'air',div=divergenceOf(dep,i,p);
   const c=mac[key]??={n:0,upper:0,lower:0,disp:0,su:0,sdu:0,sdd:0};c.n++;c.upper+=up*up;c.lower+=lo*lo;c.disp+=div*div;c.su+=up;c.sdu+=div*up;c.sdd+=div*div;}
  const macRows=Object.fromEntries(Object.entries(mac).map(([k,c])=>[k,{rmsMacUpper:+Math.sqrt(c.upper/c.n).toFixed(4),rmsMacLower:+Math.sqrt(c.lower/c.n).toFixed(4),rmsDisp:+Math.sqrt(c.disp/c.n).toFixed(4),meanMacUpper:+(c.su/c.n).toFixed(4),dispOnMacSlope:+(c.sdu/c.upper).toFixed(3)}]));
  const divergenceRows=Object.fromEntries(Object.entries(divergence).map(([k,c])=>{const md=c.sd/c.n,mb=c.sb/c.n,vd=c.sdd/c.n-md*md,vb=c.sbb/c.n-mb*mb,cv=c.sdb/c.n-md*mb;
   return [k,{n:c.n,rmsDiv:+Math.sqrt(c.sdd/c.n).toFixed(4),divPos:+c.divPos.toFixed(1),defectPos:+c.defectPos.toFixed(1),r:+(cv/Math.sqrt(vd*vb)).toFixed(3),slope:+(cv/vd).toFixed(3)}];}));
  // Alternative departure maps on the same frozen inputs, replayed in the hook
  // at R=RV. base re-gathers through the departure copy (control: equals R3).
  const variantRows:any[]=[];
  for(const {name} of variantSpecs){const d=await readMixedTexture(device,depTex.get(name)!);
   const bc=await readMixedTexture(device,variantOut.get(`${name}:capacity`)!),bv=await readMixedTexture(device,variantOut.get(`${name}:volume`)!),bb=await readMixedTexture(device,variantOut.get(`${name}:bounded`)!);
   const v={name,positiveDefect:0,maxDefect:0,massOut:0,excessOut:0,boundedExcess:0,boundedLiquid:0,boundedAir:0,controlDiff:0,meanShift:0,coverage:coverage(d)};
   for(const {i,w} of owners){const m=w**3,dd=bc[i]!-1;if(dd>0)v.positiveDefect+=dd*m;v.maxDefect=Math.max(v.maxDefect,dd);v.massOut+=bv[i]!*m;v.excessOut+=Math.max(bv[i]!-1,0)*m;
    const be=Math.max(bb[i]!-1,0)*m;v.boundedExcess+=be;if(vin[i]!>=0.5)v.boundedLiquid+=be;else v.boundedAir+=be;
    v.meanShift+=Math.hypot(d[4*i]!-dep[4*i]!,d[4*i+1]!-dep[4*i+1]!,d[4*i+2]!-dep[4*i+2]!)/owners.length;
    if(name==='base')v.controlDiff=Math.max(v.controlDiff,Math.abs(bb[i]!-bbOf.get(R3)![i]!));}
   variantRows.push(v);}
  const row={step,t:step*dt,owners:owners.length,fineTiles:layout.fineTiles.length,capacity,massIn,excessIn,perRound,classes,top:top.slice(0,40),boundedOwners:top.length,coverageBins:bins,baseCoverage,variants:variantRows,divergence:divergenceRows,mac:macRows};results.push(row);
  console.log(`step ${step} t=${row.t.toFixed(2)} owners ${row.owners} fine tiles ${row.fineTiles} massIn ${massIn.toFixed(2)} excessIn ${excessIn.toFixed(1)}`);
  for(const s of perRound)console.log(`  R${String(s.rounds).padStart(2)} defect +${s.positiveDefect.toFixed(2)} (h ${s.positiveDefectByTier[0].toFixed(2)} / 4h ${s.positiveDefectByTier[1].toFixed(2)}) -${(-s.negativeDefect).toFixed(2)} max ${s.maxDefect.toFixed(4)} owners>1e-3/1e-2/1e-1 ${s.defectOwners.join('/')} ΣBC-ΣC ${(s.capacityOut-capacity).toExponential(2)} ΣBV-ΣV ${(s.massOut-massIn).toExponential(2)} excessOut ${s.excessOut.toFixed(1)} boundedExcess ${s.boundedExcess.toFixed(1)} maxV ${s.maxOut.toFixed(2)}${s.rounds===3?` |full−live| ${s.actualDiff.toExponential(2)}`:''}`);
  console.log(`  classes (R${R3} / R${RL}): `+Object.entries(classes).sort((a,b)=>b[1].bounded3-a[1].bounded3).map(([k,c])=>`${k} n${c.owners} bounded ${c.bounded3.toFixed(1)}/${c.boundedL.toFixed(1)} defect ${c.defect3.toFixed(0)}/${c.defectL.toFixed(0)}`).join('; '));
  console.log(`  ${top.length} owners gain >0.05 from bounded input; top: `+top.slice(0,8).map(o=>JSON.stringify(o)).join(' '));
  console.log(`  div(displacement) vs R${R3} defect: `+Object.entries(divergenceRows).map(([k,c])=>`${k} ${JSON.stringify(c)}`).join(' '));
  console.log(`  trace-field MAC div (dt/h units): `+Object.entries(macRows).map(([k,c])=>`${k} ${JSON.stringify(c)}`).join(' '));
  console.log(`  coverage bins ${bins.join('|')} liquid mass ${baseCoverage.mass.join(' / ')} owners ${baseCoverage.count.join(' / ')}`);
  for(const v of variantRows)console.log(`  variant ${v.name.padEnd(11)} R${RV} defect +${v.positiveDefect.toFixed(1)} max ${v.maxDefect.toFixed(2)} ΣBV-ΣV ${(v.massOut-massIn).toExponential(2)} excessOut ${v.excessOut.toFixed(1)} boundedExcess ${v.boundedExcess.toFixed(1)} (liquid ${v.boundedLiquid.toFixed(1)} / air ${v.boundedAir.toFixed(1)}) shift ${v.meanShift.toFixed(3)} coverage mass ${v.coverage.mass.join('/')}${v.name==='base'?` control ${v.controlDiff.toExponential(2)}`:''}`);
  await writeFile(out,JSON.stringify({sceneId,initial:arg('initial','dam'),control,dt,values,rounds,results},null,1));
 }
}finally{solver?.destroy();for(const r of owned)r.destroy();device?.destroy();await releaseWebGPUExclusiveLock();}
