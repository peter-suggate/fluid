/** D4 audit of Uniform Geometric on the symmetric-expansion scene: snapshots
 * every frame stage's field and reports reflect-x / reflect-z / swap-xz error
 * so the first stage that breaks the tank's symmetry is named.
 * Coarse (4h) owners store one texel per owner (scalars) or per face patch
 * (velocity); fields are expanded to per-h-texel owner values before comparing.
 * node --import tsx tools/probe-uniform-symmetry-dawn.ts --frames=30 [--values='{"detailPolicy":"full"}']
 */
import assert from 'node:assert/strict';
import {resolve} from 'node:path';
import {pathToFileURL} from 'node:url';
import {createProcessRetainedDawnGPU,type NodeDawnProvider} from '../lib/harness/node-dawn-provider';
import {acquireWebGPUExclusiveLock,releaseWebGPUExclusiveLock,readWebGPUExclusiveLockHolder} from '../lib/harness/webgpu-smoke-isolation';
import {managedGPUDevice} from '../lib/core/gpu-compilation-manager';
import {requiredFluidDeviceLimits} from '../lib/core/webgpu-device-limits';
import {sceneDocument} from '../lib/core/scene-definition';
import {getSceneDefinition} from '../lib/core/scenes';
import {resolveMethodValues} from '../lib/core/method-contract';
import {uniformVolumeMethod} from '../lib/methods/uniform/uniform-volume-method';
import type {WebGPUUniformReferenceSolver} from '../lib/methods/uniform/webgpu-uniform-reference';
import {readMixedBuffer,readMixedTexture} from '../tests/helpers/uniform-mixed-native-fields';
const arg=(k:string,d:string)=>process.argv.find(a=>a.startsWith(`--${k}=`))?.slice(k.length+3)??d;
const frames=Number(arg('frames','30')),sceneId=arg('scene','symmetric-expansion'),verbose=arg('verbose','0')==='1';
const tolerance=Number(arg('tol','0'));
const values=resolveMethodValues(uniformVolumeMethod,'balanced',JSON.parse(arg('values','{}')));
console.log('Waiting for repository WebGPU lease');
while(await readWebGPUExclusiveLockHolder()) await new Promise(r=>setTimeout(r,500));
await acquireWebGPUExclusiveLock('dawn-probe','uniform symmetry');
let device:GPUDevice|undefined,solver:WebGPUUniformReferenceSolver|undefined;
const owned:GPUTexture[]=[];
type Kind='cell'|'vertex'|'mac'|'coarse';
try{
 const dawn=await import(pathToFileURL(resolve('node_modules/webgpu/index.js')).href) as NodeDawnProvider;Object.assign(globalThis,dawn.globals);
 const gpu=createProcessRetainedDawnGPU(dawn,['backend=metal']),adapter=await gpu.requestAdapter();assert.ok(adapter);
 device=managedGPUDevice(await adapter.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter.limits)}),{requireWorkerRealm:false});
 const errors:string[]=[];device.addEventListener('uncapturederror',e=>{e.preventDefault();errors.push(e.error.message);});
 const scene=sceneDocument(getSceneDefinition(sceneId));
 const dt=scene.numerics.fixedDt_s!;
 solver=await uniformVolumeMethod.createSolverAsync!(device,scene,'balanced',values,undefined,()=>{}) as WebGPUUniformReferenceSolver;
 const frame=(solver as any).mixedFrame,f=frame.fields;
 const N=[f.volume.width,f.volume.height,f.volume.depthOrArrayLayers] as const;
 const T=[N[0]/4,N[1]/4,N[2]/4] as const,tileCount=T[0]*T[1]*T[2];
 // Ordered snapshot list per frame: [label, kind, texture copy].
 const shots:{label:string;kind:Kind;copy:GPUTexture}[]=[];const pool=new Map<string,GPUTexture>();
 const COPY=GPUTextureUsage.COPY_SRC|GPUTextureUsage.COPY_DST;
 const snap=(e:GPUCommandEncoder,label:string,kind:Kind,source:GPUTexture)=>{
  let n=0;while(shots.some(s=>s.label===(n?`${label}#${n}`:label)))n++;const key=n?`${label}#${n}`:label;
  let copy=pool.get(key);const size=[source.width,source.height,source.depthOrArrayLayers];
  if(!copy){copy=device!.createTexture({label:`Symmetry ${key}`,size,dimension:'3d',format:source.format,usage:COPY});owned.push(copy);pool.set(key,copy);}
  e.copyTextureToTexture({texture:source},{texture:copy},size);shots.push({label:key,kind,copy});
  if(kind==='mac'&&wallOf.has(source))snapWall(e,key,wallOf.get(source)!);
 };
 const walls=new Map<string,GPUBuffer>();const wallOf=new Map<GPUTexture,GPUBuffer>([[f.velocity,f.negative],[f.velocityScratch,f.negativeScratch],[f.departure,f.negativeDeparture]]);
 const snapWall=(e:GPUCommandEncoder,label:string,source:GPUBuffer)=>{let b=walls.get(label);if(!b){b=device!.createBuffer({label:`Symmetry wall ${label}`,size:source.size,usage:GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});walls.set(label,b);}e.copyBufferToBuffer(source,0,b,0,source.size);};
 const after=(object:any,key:string,fn:(e:GPUCommandEncoder,...args:any[])=>void)=>{
  const original=object[key].bind(object);object[key]=(...args:any[])=>{const r=original(...args);fn(args[0],...args.slice(1));return r;};
 };
 after(frame.extension,'encode',e=>{snap(e,'extension',"mac",f.velocityScratch);snap(e,'coarseIn','coarse' as Kind,frame.extension.hierarchy.input);snap(e,'coarseOut','coarse' as Kind,frame.extension.hierarchy.output);});
 after(frame.surface,'encode',(e,kind)=>{if(kind==='advect')snap(e,'phiAdvect','vertex',f.phiScratch);if(kind==='redistance')snap(e,'phiRedistance','vertex',f.phi);});
 after(frame.phiResolve,'encode',(e,group)=>{if(group===frame.phiResolveGroups.scratch)snap(e,'phiScratchResolved','vertex',f.phiScratch);else snap(e,'phiResolved','vertex',f.phi);});
 after(frame.transport,'encodeTransport',e=>snap(e,'transport','cell',f.volumeScratch));
 after(frame.cleanup,'encode',e=>snap(e,'cleanup','cell',f.volume));
 after(frame.surfaceVolume,'encode',e=>snap(e,'surfaceVolumePhi','vertex',f.phi));
 after(frame.geometry,'encode',e=>{snap(e,'centerPhi','cell',f.centerPhi);snap(e,'target','cell',f.target);});
 after(frame.sharpen,'encodeSweeps',e=>snap(e,'sharpen','cell',f.volume));
 after(frame.momentum,'encode',e=>snap(e,'momentum','mac',f.departure));
 after(frame.forces,'encode',e=>snap(e,'forces','mac',f.velocityScratch));
 after(frame.authority,'encode',e=>{snap(e,'correction','cell',f.correction);snap(e,'phase','cell',f.phase);});
 after(frame.band,'encodeSolve',e=>snap(e,'projected','mac',f.velocity));

 const widthsOf=async()=>{const words=new Uint32Array((await readMixedBuffer(device!,frame.ownership.topology)).buffer).subarray(0,tileCount);return Array.from(words,w=>(w>>>31)?1:4);};
 const tileOf=(x:number,y:number,z:number)=>(x>>2)+T[0]*((y>>2)+T[1]*(z>>2));
 // Per-h-texel owner values.
 const expand=(kind:Kind,data:Float32Array,widths:number[]):Float32Array=>{
  if(kind==='vertex'){
   // Canonical umVertexValue: a vertex not all-aligned with any incident 4h
   // tile is the trilinear 4h lattice interpolation (the texel is scratch).
   const [vx,vy,vz]=[N[0]+1,N[1]+1,N[2]+1],out=new Float32Array(data);const at=(x:number,y:number,z:number)=>data[x+vx*(y+vy*z)]!;
   for(let z=0;z<vz;z++)for(let y=0;y<vy;y++)for(let x=0;x<vx;x++){const p=[x,y,z];if(p.every(c=>c%4===0))continue;
    let coarse=false;for(let k=0;k<8;k++){const t=p.map((c,a)=>(c>>2)-((k>>a)&1&&c%4===0?1:0));if(t.some((c,a)=>c<0||c>=T[a]!))continue;if(widths[t[0]!+T[0]*(t[1]!+T[1]*t[2]!)]===4)coarse=true;}
    if(!coarse)continue;const b=p.map((c,a)=>Math.min(c>>2,T[a]!-1)*4),f=p.map((c,a)=>(c-b[a]!)/4);let sum=0;
    for(let k=0;k<8;k++){const c=[k&1,(k>>1)&1,(k>>2)&1];let w=1;for(let a=0;a<3;a++)w*=c[a]?f[a]!:1-f[a]!;if(w>0)sum+=w*at(b[0]!+4*c[0]!,b[1]!+4*c[1]!,b[2]!+4*c[2]!);}
    out[x+vx*(y+vy*z)]=sum;}
   return out;
  }
  const [nx,ny,nz]=N;
  if(kind==='cell'){
   const out=new Float32Array(nx*ny*nz);
   for(let z=0;z<nz;z++)for(let y=0;y<ny;y++)for(let x=0;x<nx;x++){const w=widths[tileOf(x,y,z)]!;
    out[x+nx*(y+ny*z)]=data[(x-x%w)+nx*((y-y%w)+ny*(z-z%w))]!;}
   return out;
  }
  const out=new Float32Array(4*nx*ny*nz).fill(NaN);
  for(let z=0;z<nz;z++)for(let y=0;y<ny;y++)for(let x=0;x<nx;x++){
   const p=[x,y,z],w=widths[tileOf(x,y,z)]!;
   for(let a=0;a<3;a++){
    if((p[a]!+1)%w!==0)continue;
    const q=[...p];q[a]!+=1;const nw=q[a]!<N[a]!?widths[tileOf(q[0]!,q[1]!,q[2]!)]!:w,patch=Math.min(w,nw);
    const s=[...p];for(let b=0;b<3;b++)if(b!==a)s[b]=s[b]!-s[b]!%patch;
    out[4*(x+nx*(y+ny*z))+a]=data[4*(s[0]!+nx*(s[1]!+ny*s[2]!))+a]!;
   }
  }
  return out;
 };
 type Metric={max:number;count:number;worst?:string};
 const transforms=['reflect-x','reflect-z','swap-xz'] as const;
 const measure=(kind:Kind,v:Float32Array,dims:readonly number[],components:number):Record<string,Metric>=>{
  const [nx,ny,nz]=dims as [number,number,number];const result:Record<string,Metric>={};
  const at=(x:number,y:number,z:number,c:number)=>v[components*(x+nx*(y+ny*z))+c]!;
  const last=kind==='vertex'?0:1;// vertex mirror x->nx-1-x as well (nx = N+1)
  for(const t of transforms){let max=0,count=0,worst:string|undefined;
   for(let z=0;z<nz;z++)for(let y=0;y<ny;y++)for(let x=0;x<nx;x++)for(let c=0;c<(kind==='mac'?3:1);c++){
    const a=at(x,y,z,c);if(!Number.isFinite(a))continue;
    let tx=x,tz=z,tc=c,sign=1;
    if(t==='reflect-x'){tx=nx-1-x-(kind==='mac'&&c===0?last:0);if(kind==='mac'&&c===0)sign=-1;}
    else if(t==='reflect-z'){tz=nz-1-z-(kind==='mac'&&c===2?last:0);if(kind==='mac'&&c===2)sign=-1;}
    else{tx=z;tz=x;tc=kind==="mac"&&c!==1?2-c:c;}
    if(tx<0||tz<0||tx>=nx||tz>=nz)continue;
    const b=at(tx,y,tz,tc);if(!Number.isFinite(b))continue;
    const err=Math.abs(a-sign*b);
    if(err>tolerance){count++;if(err>max){max=err;worst=`[${x},${y},${z}]c${c}=${a.toPrecision(7)} vs [${tx},${y},${tz}]c${tc}=${b.toPrecision(7)}`;}}
   }
   result[t]={max,count,worst};
  }
  return result;
 };
 const tileMeasure=(widths:number[])=>{const out:Record<string,number>={};
  for(const t of transforms){let n=0;for(let z=0;z<T[2];z++)for(let y=0;y<T[1];y++)for(let x=0;x<T[0];x++){
   const [a,b]=t==='reflect-x'?[T[0]-1-x,z]:t==='reflect-z'?[x,T[2]-1-z]:[z,x];
   if(widths[x+T[0]*(y+T[1]*z)]!==widths[a+T[0]*(y+T[1]*b)])n++;}out[t]=n;}
  return out;
 };
 const firstBroken=new Map<string,number>();
 const compact=arg('compact','0')==='1';let row:string[]=[];
 const report=(step:number,label:string,kind:Kind,m:Record<string,Metric>)=>{
  const broken=Object.values(m).some(x=>x.count>0);
  if(compact){row.push(`${label}=${Math.max(m['reflect-x']!.max,m['reflect-z']!.max).toExponential(1)}`);return;}
  if(broken&&!firstBroken.has(label))firstBroken.set(label,step);
  if(verbose||broken&&(firstBroken.get(label)===step||step%10===0))
   console.log(`step ${step} ${label.padEnd(20)} ${transforms.map(t=>`${t} ${m[t]!.max.toExponential(2)} (${m[t]!.count})`).join('  ')}${broken?`  worst ${Object.values(m).sort((a,b)=>b.max-a.max)[0]!.worst}`:''}`);
 };
 // Initial state (the host layout and fields before frame 1).
 {const widths=await widthsOf();console.log('initial tiles h:',widths.filter(w=>w===1).length,'asym',JSON.stringify(tileMeasure(widths)));
  for(const [label,kind,t] of [['init volume','cell',f.volume],['init phi','vertex',f.phi],['init velocity','mac',f.velocity]] as const){
   const data=await readMixedTexture(device,t),v=expand(kind,data,widths);
   report(0,label,kind,measure(kind,v,kind==='vertex'?[t.width,t.height,t.depthOrArrayLayers]:N,kind==='mac'?4:1));}}
 for(let step=1;step<=frames;step++){
  shots.length=0;
  assert.ok(solver.advanceTo(step*dt));await solver.awaitFrameCompletion();
  const widths=await widthsOf(),tiles=tileMeasure(widths);
  const tileBroken=Object.values(tiles).some(n=>n>0);
  if(tileBroken&&!firstBroken.has('tiles'))firstBroken.set('tiles',step);
  if(verbose||tileBroken&&firstBroken.get('tiles')===step)console.log(`step ${step} tiles h=${widths.filter(w=>w===1).length} asym ${JSON.stringify(tiles)}`);
  for(const s of shots){
   const data=await readMixedTexture(device,s.copy);
   if(s.kind==='coarse'){const c=arg('coarse','');if(c){const [cs,cc,cy]=c.split(':').map(Number);if(cs===step&&s.label.startsWith('extension')===false){
     const [w,hh]=[s.copy.width,s.copy.height];console.log(`  ${s.label} comp ${cc} y ${cy} (rows z, cols x; value/mask)`);
     for(let z=0;z<s.copy.depthOrArrayLayers;z++){const cells:string[]=[];for(let x=0;x<w;x++){const i=4*(x+w*(cy!+hh*z));cells.push(`${data[i+cc!]!.toFixed(4)}/${data[i+3]}`);}console.log(`   z${z}: ${cells.join(' ')}`);}}}continue;}
   // Fields written after the head relayout are in this frame's layout.
   const v=expand(s.kind,data,widths);
   report(step,s.label,s.kind,measure(s.kind,v,s.kind==='vertex'?[s.copy.width,s.copy.height,s.copy.depthOrArrayLayers]:N,s.kind==='mac'?4:1));
   if(s.kind==='mac'&&walls.has(s.label)){
    // +wall texel (D-1) against the mirrored -wall entry (negative buffer).
    const neg=await readMixedBuffer(device,walls.get(s.label)!);const [nx,ny,nz]=N;let max=0,count=0,worst='';
    for(const a of [0,2]){for(let v=0;v<ny;v++)for(let u=0;u<(a===0?nz:nx);u++){
     const hi=a===0?[nx-1,v,u]:[u,v,nz-1];const w=widths[tileOf(hi[0]!,hi[1]!,hi[2]!)]!;
     const s0=hi.map((c,b)=>b===a?c:c-c%w);const high=data[4*(s0[0]!+nx*(s0[1]!+ny*s0[2]!))+a]!;
     const lo=a===0?[0,v,u]:[u,v,0];const wl=widths[tileOf(lo[0]!,lo[1]!,lo[2]!)]!;const l0=lo.map((c,b)=>b===a?c:c-c%wl);
     const low=neg[a===0?l0[1]!+ny*l0[2]!:ny*nz+nx*nz+l0[0]!+nx*l0[1]!]!;
     const err=Math.abs(high+low);if(err>tolerance){count++;if(err>max){max=err;worst=`axis ${a} hi ${hi} ${high} lo ${lo} ${low}`;}}}}
    if(compact)row.push(`${s.label}.wall=${max.toExponential(1)}`);else if(verbose||count)console.log(`step ${step} ${(s.label+' wall').padEnd(20)} max ${max.toExponential(2)} (${count}) ${worst}`);
   }
   // --line=label@step:x,y (z line of component 2 against its z mirror).
   const line=arg('line','');if(line){const [l,rest]=line.split('@');const [st,xy]=rest!.split(':');const [lx,ly,lc=2]=xy!.split(',').map(Number);
    if(l===s.label&&Number(st)===step){const [nx,ny]=N;const cmp=s.kind==='mac'?4:1,off=s.kind==='mac'?lc:0;
     for(let z=0;z<N[2];z++){const i=(zz:number)=>cmp*(lx!+nx*(ly!+ny*zz))+off;const m=N[2]-(s.kind==='mac'&&lc===2?2:1)-z;const sg=s.kind==='mac'&&lc===2?-1:1;
      console.log(`  z=${z} w=${widths[tileOf(lx!,ly!,z)]} raw=${data[i(z)]!.toPrecision(5)} exp=${v[i(z)]!.toPrecision(5)} | mirror z=${m} w=${m>=0?widths[tileOf(lx!,ly!,m)]:'-'} raw=${m>=0?(sg*data[i(m)]!).toPrecision(5):'-'}`);}
     const row=(y:number)=>Array.from({length:T[2]},(_,tz)=>Array.from({length:T[0]},(_,tx)=>widths[tx+T[0]*(y+T[1]*tz)]===1?'h':'4').join('')).join(' ');
     for(let y=0;y<T[1];y++)console.log(`  tiles y${y}: ${row(y)}`);}}
   // --newton=label@step:x,y,z: CPU umWindowSearch (w=1) at p and its z mirror.
   const newton=arg('newton','');if(newton){const [l,rest]=newton.split('@');const [st,xyz]=rest!.split(':');const P=xyz!.split(',').map(Number);
    if(l===s.label&&Number(st)===step&&s.kind==='vertex'){const vx=N[0]+1,vy=N[1]+1,D=N;const hh=0.05;
     const lat=(x:number,y:number,z:number)=>v[x+vx*(y+vy*z)]!;
     const sample=(q:number[])=>{const c=q.map((a,i)=>Math.min(Math.max(a,0),D[i]!));const b=c.map((a,i)=>Math.min(Math.floor(a),D[i]!-1));const t=c.map((a,i)=>a-b[i]!);let sum=0;
      for(let k=0;k<8;k++){const o=[k&1,(k>>1)&1,(k>>2)&1];let w=1;for(let a=0;a<3;a++)w*=o[a]?t[a]!:1-t[a]!;sum+=w*lat(b[0]!+o[0]!,b[1]!+o[1]!,b[2]!+o[2]!);}return sum;};
     const grad=(q:number[])=>[0,1,2].map(a=>{const lo=[...q],hi=[...q];lo[a]=Math.max(q[a]!-0.25,0);hi[a]=Math.min(q[a]!+0.25,D[a]!);return (sample(hi)-sample(lo))/Math.max(hi[a]!-lo[a]!,1e-6);});
     for(const p0 of [P,[P[0]!,P[1]!,D[2]!-P[2]!]]){let q=[...p0];let phiQ=lat(p0[0]!,p0[1]!,p0[2]!);const log=[`p=${p0} initial=${phiQ.toPrecision(7)}`];
      for(let i=0;i<8;i++){const g=grad(q);const norm=g.reduce((a,x)=>a+(x/hh)**2,0);if(norm<1e-16)break;
       const next=q.map((a,i)=>Math.min(Math.max(a-Math.min(Math.max(phiQ*g[i]!/(hh*hh*norm),-2),2),Math.max(0,p0[i]!-4)),Math.min(D[i]!,p0[i]!+4)));
       const phiNext=sample(next);log.push(`  it${i} q=${next.map(x=>x.toFixed(4))} phi=${phiNext.toExponential(4)}`);if(Math.abs(phiNext)>=Math.abs(phiQ))break;q=next;phiQ=phiNext;}
      console.log(log.join('\n'),`\n  found=${Math.abs(phiQ)<0.005*hh} |phiQ|=${Math.abs(phiQ).toExponential(4)} threshold=${(0.005*hh).toExponential(2)}`);}}}
   // --vline=label@step:x,y (vertex z line against its z mirror).
   const vline=arg('vline','');if(vline){const [l,rest]=vline.split('@');const [st,xy]=rest!.split(':');const [lx,ly]=xy!.split(',').map(Number);
    if(l===s.label&&Number(st)===step&&s.kind==='vertex'){const nx=N[0]+1,ny=N[1]+1,nz=N[2]+1;const i=(zz:number)=>lx!+nx*(ly!+ny*zz);
     for(let z=0;z<nz;z++)console.log(`  z=${z} ${v[i(z)]!.toPrecision(6)} | mirror z=${nz-1-z} ${v[i(nz-1-z)]!.toPrecision(6)}`);
     const row=(y:number)=>Array.from({length:T[2]},(_,tz)=>Array.from({length:T[0]},(_,tx)=>widths[tx+T[0]*(y+T[1]*tz)]===1?'h':'4').join('')).join(' ');
     for(let y=0;y<T[1];y++)console.log(`  tiles y${y} (z-major rows of x): ${row(y)}`);}}
  }
  if(compact){console.log(`step ${step} tiles ${JSON.stringify(tiles)} ${row.join(' ')}`);row=[];}
  assert.deepEqual(errors,[]);
 }
 console.log('first broken step per stage:',JSON.stringify(Object.fromEntries(firstBroken)));
}finally{solver?.destroy();for(const t of owned)t.destroy();device?.destroy();await releaseWebGPUExclusiveLock();}
