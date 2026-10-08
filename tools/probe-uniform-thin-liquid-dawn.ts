/** Thin-liquid census for Uniform Geometric: where do thin sheets and drops
 * lose their level set (phi) while their conservative volume (V) survives?
 *
 * Per sampled frame it snapshots phi and V between the frame's stages and
 * reports, on h cells (4h owners expanded):
 *  - phiVol: sum of the cell fill of the phi zero set (3^3 subsamples where
 *    the eight corners change sign), i.e. what is rendered and pressurised;
 *  - V: the conservative volume;
 *  - "thin": cells whose 5^3 box mean of start-of-frame V is below 0.25
 *    (a one-cell sheet reads 0.2, a two-cell one 0.4, a flat pool surface 0.5);
 *  - hidden: V in cells phi leaves completely dry (fill 0) -- invisible water;
 *  - V components (6-connected, V>0.05) by size, with the phi fill inside
 *    each, so a drop or sheet that exists in V but not in phi is counted.
 * Stage deltas attribute phi loss in the thin set to advection, redistance
 * (with drain retirement) and the total surface-volume shift, and V moves to
 * transport, dust cleanup and sharpening.
 *
 * --bins=axis:width[,axis:width]: per-body census for the thin-liquid ladders
 * (lib/core/thin-liquid-scenes.ts): V, phi fill, their centroid heights and
 * hidden V per bin of `width` cells along axis 0/1/2 (composite keys are
 * first-axis-major), over cells at or above y = --above (default 10, clear of
 * the 6-cell pool).
 *
 * --jobs=scene:arm[,scene:arm...] --patchdir=D --dir=O: several runs on one
 * device, so a job repeats only the compilation its patches change (Dawn
 * caches modules and pipelines by content; the thin ladders share one grid).
 * Arm "base" is unpatched; "a+b" applies D/patch-a.json then D/patch-b.json;
 * a part "key=value" is a method value instead (phiPreserveSurface=on).
 * Each job writes O/<scene>-<arm>.json and prints build/step/read/analysis s.
 *
 * node --import tsx tools/probe-uniform-thin-liquid-dawn.ts --scene=cm12-figure-9 --frames=120 --every=4 --out=/tmp/thin.json [--values='{"phiDrain":"off"}']
 * node --import tsx tools/probe-uniform-thin-liquid-dawn.ts --jobs=thin-droplet-ladder:base,thin-sheet-ladder:base --frames=30 --every=3 --dir=/tmp/thin
 */
import assert from 'node:assert/strict';
import {writeFile} from 'node:fs/promises';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {pathToFileURL} from 'node:url';
import {createProcessRetainedDawnGPU,type NodeDawnProvider} from '../lib/harness/node-dawn-provider';
import {managedGPUDevice,gpuCompilationManagerFor} from '../lib/core/gpu-compilation-manager';
import {requiredFluidDeviceLimits} from '../lib/core/webgpu-device-limits';
import {sceneDocument} from '../lib/core/scene-definition';
import {getSceneDefinition} from '../lib/core/scenes';
import {resolveMethodValues} from '../lib/core/method-contract';
import {uniformVolumeMethod} from '../lib/methods/uniform/uniform-volume-method';
import type {WebGPUUniformReferenceSolver} from '../lib/methods/uniform/webgpu-uniform-reference';
import {readMixedTexture,readMixedTileWords} from '../tests/helpers/uniform-mixed-native-fields';
import {THIN_DROPLET_BIN_CELLS,THIN_FILM_BIN_CELLS,THIN_SHEET_BIN_CELLS} from '../lib/core/thin-liquid-scenes';

const arg=(k:string,d:string)=>process.argv.find(a=>a.startsWith(`--${k}=`))?.slice(k.length+3)??d;
const sceneId=arg('scene','cm12-figure-9'),frames=Number(arg('frames','120')),every=Number(arg('every','4'));
const out=arg('out',''),dtArg=arg('dt','');
// Composite keys: --bins=0:32,2:14 bins x by 32 and z by 14 (x-major index).
const THIN_BINS:Record<string,string>={'thin-droplet-ladder':`0:${THIN_DROPLET_BIN_CELLS}`,'thin-sheet-ladder':`0:${THIN_SHEET_BIN_CELLS}`,'thin-wall-films':`0:32,2:${THIN_FILM_BIN_CELLS}`};
const binKeysFor=(id:string)=>{const b=arg('bins',THIN_BINS[id]??'');return b?b.split(',').map(k=>k.split(':').map(Number) as [number,number]):[];};
const binAbove=Number(arg('above','10'));
const values=resolveMethodValues(uniformVolumeMethod,'balanced',JSON.parse(arg('values','{}')));
// --patch=<json [[find,replace],...]> or @file.json: WGSL rewrites for A/B arms (each must hit).
const patchArg=arg('patch','[]');
type Patch=[string,string];type Job={sceneId:string;arm:string;patches:Patch[];values:Record<string,string>;out:string};
const patchDir=arg('patchdir','.'),outDir=arg('dir','');
const jobs:Job[]=arg('jobs','')?arg('jobs','').split(',').map(j=>{const [scene,arm='base']=j.split(':') as [string,string?];
 const parts=arm==='base'?[]:arm.split('+');
 const patches=parts.filter(a=>!a.includes('=')).flatMap(a=>JSON.parse(readFileSync(`${patchDir}/patch-${a}.json`,'utf8')) as Patch[]);
 const values=Object.fromEntries(parts.filter(a=>a.includes('=')).map(a=>a.split('=') as [string,string]));
 return {sceneId:scene,arm,patches,values,out:outDir?`${outDir}/${scene}-${arm}.json`:''};})
 :[{sceneId,arm:'cli',patches:JSON.parse(patchArg.startsWith('@')?readFileSync(patchArg.slice(1),'utf8'):patchArg),values:{},out}];
let device:GPUDevice|undefined,solver:WebGPUUniformReferenceSolver|undefined;
const owned:GPUTexture[]=[];const runStart=performance.now();
try{
 const dawn=await import(pathToFileURL(resolve('node_modules/webgpu/index.js')).href) as NodeDawnProvider;Object.assign(globalThis,dawn.globals);
 const gpu=createProcessRetainedDawnGPU(dawn,['backend=metal']),adapter=await gpu.requestAdapter();assert.ok(adapter);
 device=managedGPUDevice(await adapter.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter.limits)}),{requireWorkerRealm:false});
 const errors:string[]=[];device.addEventListener('uncapturederror',e=>{e.preventDefault();errors.push(e.error.message);});
 let active:Patch[]=[],hits:number[]=[];
 {const compiler=gpuCompilationManagerFor(device),create=compiler.createShaderModule.bind(compiler);
  compiler.createShaderModule=d=>{let code=d.code;active.forEach(([find,replace],i)=>{if(code.includes(find)){hits[i]!++;code=code.split(find).join(replace);}});return create({...d,code});};}
 const runJob=async({sceneId,arm,patches,values:armValues,out}:Job)=>{
 const values=resolveMethodValues(uniformVolumeMethod,'balanced',{...JSON.parse(arg('values','{}')),...armValues});
 active=patches;hits=patches.map(()=>0);const binKeys=binKeysFor(sceneId);
 const time={build:0,step:0,read:0,analysis:0};
 try{
 const scene=sceneDocument(getSceneDefinition(sceneId));
 if(dtArg)scene.numerics.fixedDt_s=scene.numerics.maxDt_s=Number(dtArg);
 const dt=scene.numerics.fixedDt_s!;
 const t0=performance.now();
 solver=await uniformVolumeMethod.createSolverAsync!(device!,scene,'balanced',values,undefined,()=>{}) as WebGPUUniformReferenceSolver;
 time.build=(performance.now()-t0)/1000;
 console.log(`${sceneId} ${arm}: built in ${time.build.toFixed(1)} s, dt ${dt.toFixed(5)}`);
 patches.forEach(([find],i)=>assert.ok(hits[i]!>0,`patch ${i} never matched: ${find.slice(0,80)}`));
 const frame=(solver as any).mixedFrame,f=frame.fields;
 const N=[f.volume.width,f.volume.height,f.volume.depthOrArrayLayers] as const;
 const [nx,ny,nz]=N,T=[nx/4,ny/4,nz/4] as const,cells=nx*ny*nz;
 const [vx,vy,vz]=[nx+1,ny+1,nz+1];

 // Snapshots: copies taken between stages on sampled frames.
 type Kind='cell'|'vertex';
 const pool=new Map<string,{kind:Kind;copy:GPUTexture}>();let sampled=false;
 const snap=(e:GPUCommandEncoder,label:string,kind:Kind,source:GPUTexture)=>{
  if(!sampled)return;let s=pool.get(label);const size=[source.width,source.height,source.depthOrArrayLayers];
  if(!s){s={kind,copy:device!.createTexture({label:`Thin ${label}`,size,dimension:'3d',format:source.format,usage:GPUTextureUsage.COPY_SRC|GPUTextureUsage.COPY_DST})};owned.push(s.copy);pool.set(label,s);}
  e.copyTextureToTexture({texture:source},{texture:s.copy},size);
 };
 const wrap=(object:any,key:string,before:(e:GPUCommandEncoder,...a:any[])=>void,after:(e:GPUCommandEncoder,...a:any[])=>void)=>{
  const original=object[key].bind(object);object[key]=(...args:any[])=>{before(args[0],...args.slice(1));const r=original(...args);after(args[0],...args.slice(1));return r;};
 };
 wrap(frame.surface,'encode',(e,kind)=>{if(kind==='advect'){snap(e,'phi.start','vertex',f.phi);snap(e,'V.start','cell',f.volume);}},
  (e,kind)=>{if(kind==='advect')snap(e,'phi.advect','vertex',f.phiScratch);if(kind==='redistance')snap(e,'phi.redistance','vertex',f.phi);});
 wrap(frame.transport,'encodeTransport',()=>{},e=>snap(e,'V.transport','cell',f.volumeScratch));
 wrap(frame.cleanup,'encode',()=>{},e=>snap(e,'V.cleanup','cell',f.volume));
 wrap(frame.surfaceVolume,'encode',()=>{},e=>snap(e,'phi.shift','vertex',f.phi));
 wrap(frame.sharpen,'encodeSweeps',()=>{},e=>snap(e,'V.sharpen','cell',f.volume));

 // Expansion of 4h owners onto h cells / canonical vertices.
 const tileOf=(x:number,y:number,z:number)=>(x>>2)+T[0]*((y>>2)+T[1]*(z>>2));
 const expandCell=(data:Float32Array,widths:Uint8Array)=>{
  const o=new Float32Array(cells);
  for(let z=0;z<nz;z++)for(let y=0;y<ny;y++)for(let x=0;x<nx;x++){const w=widths[tileOf(x,y,z)]!;o[x+nx*(y+ny*z)]=data[(x-x%w)+nx*((y-y%w)+ny*(z-z%w))]!;}
  return o;
 };
 const expandVertex=(data:Float32Array,widths:Uint8Array)=>{
  const o=new Float32Array(data);const at=(x:number,y:number,z:number)=>data[x+vx*(y+vy*z)]!;
  for(let z=0;z<vz;z++)for(let y=0;y<vy;y++)for(let x=0;x<vx;x++){
   if(x%4===0&&y%4===0&&z%4===0)continue;
   let coarse=false;
   for(let k=0;k<8&&!coarse;k++){
    const tx=(x>>2)-((k&1)&&x%4===0?1:0),ty=(y>>2)-((k>>1&1)&&y%4===0?1:0),tz=(z>>2)-((k>>2&1)&&z%4===0?1:0);
    if(tx<0||ty<0||tz<0||tx>=T[0]||ty>=T[1]||tz>=T[2])continue;if(widths[tx+T[0]*(ty+T[1]*tz)]===4)coarse=true;
   }
   if(!coarse)continue;
   const bx=Math.min(x>>2,T[0]-1)*4,by=Math.min(y>>2,T[1]-1)*4,bz=Math.min(z>>2,T[2]-1)*4,fx=(x-bx)/4,fy=(y-by)/4,fz=(z-bz)/4;
   let s=0;for(let k=0;k<8;k++){const cx=k&1,cy=k>>1&1,cz=k>>2&1;const w=(cx?fx:1-fx)*(cy?fy:1-fy)*(cz?fz:1-fz);if(w>0)s+=w*at(bx+4*cx,by+4*cy,bz+4*cz);}
   o[x+vx*(y+vy*z)]=s;
  }
  return o;
 };
 // Fill of each h cell under the trilinear zero set.
 const fillOf=(phi:Float32Array)=>{
  const o=new Float32Array(cells);const c=new Float64Array(8);
  for(let z=0;z<nz;z++)for(let y=0;y<ny;y++)for(let x=0;x<nx;x++){
   let neg=0;for(let k=0;k<8;k++){const v=phi[(x+(k&1))+vx*((y+(k>>1&1))+vy*(z+(k>>2&1)))]!;c[k]=v;if(v<0)neg++;}
   if(neg===0)continue;if(neg===8){o[x+nx*(y+ny*z)]=1;continue;}
   let inside=0;
   for(let s=0;s<27;s++){const u=((s%3)+0.5)/3,v=((((s/3)|0)%3)+0.5)/3,w=(((s/9)|0)+0.5)/3;
    const val=(1-w)*((1-v)*((1-u)*c[0]!+u*c[1]!)+v*((1-u)*c[2]!+u*c[3]!))+w*((1-v)*((1-u)*c[4]!+u*c[5]!)+v*((1-u)*c[6]!+u*c[7]!));
    if(val<0)inside++;}
   o[x+nx*(y+ny*z)]=inside/27;
  }
  return o;
 };
 // Thin mask: 5^3 box mean of V below 0.25 (summed-volume table).
 const thinOf=(V:Float32Array)=>{
  const P=new Float64Array((nx+1)*(ny+1)*(nz+1));const pi=(x:number,y:number,z:number)=>x+(nx+1)*(y+(ny+1)*z);
  for(let z=0;z<nz;z++)for(let y=0;y<ny;y++)for(let x=0;x<nx;x++)
   P[pi(x+1,y+1,z+1)]=Math.max(0,V[x+nx*(y+ny*z)]!)+P[pi(x,y+1,z+1)]!+P[pi(x+1,y,z+1)]!+P[pi(x+1,y+1,z)]!-P[pi(x,y,z+1)]!-P[pi(x,y+1,z)]!-P[pi(x+1,y,z)]!+P[pi(x,y,z)]!;
  const m=new Uint8Array(cells);
  for(let z=0;z<nz;z++)for(let y=0;y<ny;y++)for(let x=0;x<nx;x++){
   const x0=Math.max(0,x-2),x1=Math.min(nx,x+3),y0=Math.max(0,y-2),y1=Math.min(ny,y+3),z0=Math.max(0,z-2),z1=Math.min(nz,z+3);
   const s=P[pi(x1,y1,z1)]!-P[pi(x0,y1,z1)]!-P[pi(x1,y0,z1)]!-P[pi(x1,y1,z0)]!+P[pi(x0,y0,z1)]!+P[pi(x0,y1,z0)]!+P[pi(x1,y0,z0)]!-P[pi(x0,y0,z0)]!;
   m[x+nx*(y+ny*z)]=s<0.25*125?1:0;// the box always counts 125 cells: a wall adds air
  }
  return m;
 };
 const sum=(a:Float32Array,mask?:Uint8Array)=>{let s=0;for(let i=0;i<a.length;i++)if(!mask||mask[i])s+=a[i]!;return s;};
 const hidden=(V:Float32Array,fill:Float32Array,mask?:Uint8Array)=>{let s=0;for(let i=0;i<V.length;i++)if(fill[i]===0&&V[i]!>0&&(!mask||mask[i]))s+=V[i]!;return s;};
 // V components (6-connected over V>0.05) with the phi fill they hold.
 const BUCKETS=[0.5,4,32,256,Infinity] as const;
 const components=(V:Float32Array,fill:Float32Array)=>{
  const label=new Int32Array(cells).fill(-1);const stack:number[]=[];
  const rows=BUCKETS.slice(0,-1).map((lo,i)=>({lo,hi:BUCKETS[i+1]!,count:0,V:0,phi:0,invisible:0,invisibleV:0}));
  for(let s=0;s<cells;s++){
   if(label[s]!==-1||!(V[s]!>0.05))continue;
   let cv=0,cf=0;stack.push(s);label[s]=s;
   while(stack.length){const i=stack.pop()!;cv+=V[i]!;cf+=fill[i]!;
    const x=i%nx,y=((i/nx)|0)%ny,z=(i/(nx*ny))|0;
    for(const [d,ok] of [[-1,x>0],[1,x<nx-1],[-nx,y>0],[nx,y<ny-1],[-nx*ny,z>0],[nx*ny,z<nz-1]] as const){
     const j=i+d;if(ok&&label[j]===-1&&V[j]!>0.05){label[j]=s;stack.push(j);}}}
   const r=rows.find(r=>cv>=r.lo&&cv<r.hi);if(!r)continue;
   r.count++;r.V+=cv;r.phi+=cf;if(cf<0.25*cv){r.invisible++;r.invisibleV+=cv;}
  }
  return rows;
 };

 const binsOf=(V:Float32Array,fill:Float32Array)=>{
  if(!binKeys.length)return undefined;const counts=binKeys.map(([a,w])=>Math.ceil(N[a]!/w));
  const b=Array.from({length:counts.reduce((a,c)=>a*c,1)},()=>({V:0,phi:0,hidden:0,Vy:0,phiy:0}));
  for(let z=0;z<nz;z++)for(let y=binAbove;y<ny;y++)for(let x=0;x<nx;x++){
   let key=0;binKeys.forEach(([a,w],j)=>{key=key*counts[j]!+Math.floor([x,y,z][a]!/w);});
   const i=x+nx*(y+ny*z),r=b[key]!,v=Math.max(0,V[i]!),f=fill[i]!;
   r.V+=v;r.phi+=f;r.Vy+=v*(y+0.5);r.phiy+=f*(y+0.5);if(f===0)r.hidden+=v;
  }
  return b.map(r=>({V:+r.V.toFixed(3),phi:+r.phi.toFixed(3),hidden:+r.hidden.toFixed(3),yV:+(r.Vy/Math.max(r.V,1e-9)).toFixed(2),yPhi:+(r.phiy/Math.max(r.phi,1e-9)).toFixed(2)}));
 };
 const rows:any[]=[];const info=()=>(solver as any).executionInfo??{};
 let dustTotal=0;
 for(let step=1;step<=frames;step++){
  sampled=step%every===0;
  const s0=performance.now();
  assert.ok(solver.advanceTo(step*dt));await solver.awaitFrameCompletion();
  time.step+=(performance.now()-s0)/1000;
  const ex=info();dustTotal+=ex.uniformVolumeDustMass_cells??0;
  if(!sampled)continue;
  const a0=performance.now(),read0=time.read;
  const words=await readMixedTileWords(device!,solver);const widths=new Uint8Array(words.length);for(let i=0;i<words.length;i++)widths[i]=(words[i]!>>>31)?1:4;
  const read=async(label:string)=>{const s=pool.get(label);assert.ok(s,`no snapshot ${label}`);const r0=performance.now();const d=await readMixedTexture(device!,s.copy);time.read+=(performance.now()-r0)/1000;return s.kind==='cell'?expandCell(d,widths):expandVertex(d,widths);};
  const Vs:Record<string,Float32Array>={},fills:Record<string,Float32Array>={};
  for(const k of ['start','transport','cleanup','sharpen'])Vs[k]=await read(`V.${k}`);
  const phis:Record<string,Float32Array>={};
  for(const k of ['start','advect','redistance','shift']){phis[k]=await read(`phi.${k}`);fills[k]=fillOf(phis[k]!);}
  const thin=thinOf(Vs.start!);
  const row:any={step,t:+(step*dt).toFixed(4),hTiles:widths.filter(w=>w===1).length,dustMass:ex.uniformVolumeDustMass_cells??0,dustTotal,V:{},phi:{}};
  for(const [k,v] of Object.entries(Vs))row.V[k]={all:sum(v),thin:sum(v,thin)};
  for(const [k,v] of Object.entries(fills))row.phi[k]={all:sum(v),thin:sum(v,thin)};
  row.hidden={start:hidden(Vs.start!,fills.start!),end:hidden(Vs.sharpen!,fills.shift!),endThin:hidden(Vs.sharpen!,fills.shift!,thin)};
  // Hidden V by the cell's centre phi (mean of its 8 vertices) in cells:
  // beside the surface (<1), in the sharpening band (<2.1) or stranded.
  {const phi=phis.shift!,V=Vs.sharpen!,h=[0,0,0];const hc=scene.voxelDomain.finestCellSize_m;
   for(let z=0;z<nz;z++)for(let y=0;y<ny;y++)for(let x=0;x<nx;x++){const i=x+nx*(y+ny*z);if(!(V[i]!>0)||fills.shift![i]!>0)continue;
    let c=0;for(let k=0;k<8;k++)c+=phi[(x+(k&1))+vx*((y+(k>>1&1))+vy*(z+(k>>2&1)))]!;c/=8*hc;h[c<1?0:c<2.1?1:2]+=V[i]!;}
   row.hiddenBy={beside:h[0],band:h[1],stranded:h[2]};}
  // Paired phi-versus-V deficit on the thin mask: each advected pair first
  // (flux into the mask cancels where phi and V agree), then phi-only stages.
  const deficit=(V:Float32Array,fill:Float32Array)=>{let s=0;for(let i=0;i<cells;i++)if(thin[i])s+=Math.max(0,V[i]!)-fill[i]!;return s;};
  row.deficitThin={start:deficit(Vs.start!,fills.start!),advect:deficit(Vs.transport!,fills.advect!),redistance:deficit(Vs.transport!,fills.redistance!),
   shift:deficit(Vs.cleanup!,fills.shift!),sharpen:deficit(Vs.sharpen!,fills.shift!)};
  row.components=components(Vs.sharpen!,fills.shift!);
  {const core=new Float32Array(cells);for(let i=0;i<cells;i++)core[i]=Vs.sharpen![i]!>0.3?Vs.sharpen![i]!:0;row.coreComponents=components(core,fills.shift!);}
  row.bins=binsOf(Vs.sharpen!,fills.shift!);
  rows.push(row);
  time.analysis+=(performance.now()-a0)/1000-(time.read-read0);
  const d=(a:any,b:any,key:'all'|'thin')=>(b[key]-a[key]).toFixed(2);
  const bulk=row.components.at(-1);
  const small=row.components.slice(0,-1).reduce((a:any,r:any)=>({count:a.count+r.count,V:a.V+r.V,phi:a.phi+r.phi,inv:a.inv+r.invisible,invV:a.invV+r.invisibleV}),{count:0,V:0,phi:0,inv:0,invV:0});
  console.log(`step ${step} t=${row.t} h=${row.hTiles} V=${row.V.sharpen.all.toFixed(1)} phi=${row.phi.shift.all.toFixed(1)} | thin V ${row.V.start.thin.toFixed(1)} phi ${row.phi.start.thin.toFixed(1)} `+
   `| dphi thin adv ${d(row.phi.start,row.phi.advect,'thin')} red ${d(row.phi.advect,row.phi.redistance,'thin')} shift ${d(row.phi.redistance,row.phi.shift,'thin')} `+
   `| dV thin tr ${d(row.V.start,row.V.transport,'thin')} dust ${d(row.V.transport,row.V.cleanup,'thin')} sharp ${d(row.V.cleanup,row.V.sharpen,'thin')} `+
   `| hidden ${row.hidden.end.toFixed(1)} (thin ${row.hidden.endThin.toFixed(1)}) | small comps ${small.count} V ${small.V.toFixed(1)} phi ${small.phi.toFixed(1)} invisible ${small.inv} (${small.invV.toFixed(1)}) bulk ${bulk.count}`);
  console.log(`  deficit thin ${Object.entries(row.deficitThin).map(([k,v])=>`${k} ${(v as number).toFixed(1)}`).join(' ')} | hidden beside ${row.hiddenBy.beside.toFixed(1)} band ${row.hiddenBy.band.toFixed(1)} stranded ${row.hiddenBy.stranded.toFixed(1)}`);
  if(row.bins)console.log('  bins V/phi/hidden yV yPhi: '+row.bins.map((r:any)=>`${r.V.toFixed(1)}/${r.phi.toFixed(1)}/${r.hidden.toFixed(1)} ${r.yV.toFixed(1)} ${r.yPhi.toFixed(1)}`).join(' | '));
  assert.deepEqual(errors,[]);
 }
 // Stage totals over sampled frames.
 const tot=(fn:(r:any)=>number)=>rows.reduce((a,r)=>a+fn(r),0);
 const summary={
  sampled:rows.length,
  phiThin:{advect:tot(r=>r.phi.advect.thin-r.phi.start.thin),redistance:tot(r=>r.phi.redistance.thin-r.phi.advect.thin),shift:tot(r=>r.phi.shift.thin-r.phi.redistance.thin)},
  phiAll:{advect:tot(r=>r.phi.advect.all-r.phi.start.all),redistance:tot(r=>r.phi.redistance.all-r.phi.advect.all),shift:tot(r=>r.phi.shift.all-r.phi.redistance.all)},
  VThin:{transport:tot(r=>r.V.transport.thin-r.V.start.thin),cleanup:tot(r=>r.V.cleanup.thin-r.V.transport.thin),sharpen:tot(r=>r.V.sharpen.thin-r.V.cleanup.thin)},
  VAll:{transport:tot(r=>r.V.transport.all-r.V.start.all),cleanup:tot(r=>r.V.cleanup.all-r.V.transport.all),sharpen:tot(r=>r.V.sharpen.all-r.V.cleanup.all)},
  meanThinV:tot(r=>r.V.start.thin)/rows.length,meanThinPhi:tot(r=>r.phi.start.thin)/rows.length,
  meanHidden:tot(r=>r.hidden.end)/rows.length,dustTotal,
  deficitThin:{advect:tot(r=>r.deficitThin.advect-r.deficitThin.start),redistance:tot(r=>r.deficitThin.redistance-r.deficitThin.advect),
   shiftAndDust:tot(r=>r.deficitThin.shift-r.deficitThin.redistance),sharpen:tot(r=>r.deficitThin.sharpen-r.deficitThin.shift)},
  meanHiddenBy:{beside:tot(r=>r.hiddenBy.beside)/rows.length,band:tot(r=>r.hiddenBy.band)/rows.length,stranded:tot(r=>r.hiddenBy.stranded)/rows.length},
 };
 console.log('summary (sums over sampled frames):',JSON.stringify(summary,(k,v)=>typeof v==='number'?+v.toFixed(3):v));
 console.log(`time ${sceneId} ${arm}: `+Object.entries(time).map(([k,v])=>`${k} ${v.toFixed(1)} s`).join(', '));
 if(out)await writeFile(out,JSON.stringify({sceneId,arm,values,patches,frames,every,dt,time,summary,rows},null,1));
 }finally{solver?.destroy();solver=undefined;for(const t of owned.splice(0))t.destroy();}
 };
 for(const job of jobs)await runJob(job);
 console.log(`all jobs in ${((performance.now()-runStart)/1000).toFixed(1)} s`);
}finally{solver?.destroy();for(const t of owned)t.destroy();device?.destroy();}
