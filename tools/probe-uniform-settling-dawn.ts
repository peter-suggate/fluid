/** Diagnostic-only owner-weighted velocity/volume census. Energy is a proxy,
 * not the variational MAC-face kinetic energy. No production settings change.
 * node --import tsx tools/probe-uniform-settling-dawn.ts --frames=600 --out=/tmp/settling.json
 */
import assert from 'node:assert/strict';
import {writeFile} from 'node:fs/promises';
import {readFileSync,readdirSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {uniformQualityCensus} from './uniform-quality-census';
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
import {UNIFORM_PRESSURE_BAND_SCHEDULE} from '../lib/methods/uniform/uniform-pressure-band';
import {mixedCellWidth,type UniformMixedLayout} from '../lib/methods/uniform/uniform-mixed-layout';
import {readMixedTexture,readMixedTileWords} from '../tests/helpers/uniform-mixed-native-fields';
import {redistanceEveryStepReference} from "./uniform-redistance-reference";
const arg=(k:string,d:string)=>process.argv.find(a=>a.startsWith(`--${k}=`))?.slice(k.length+3)??d;
const fingerprint=()=>{const files=[...readdirSync('lib/methods/uniform',{recursive:true}).map(String).filter(p=>p.endsWith('.ts')).map(p=>'lib/methods/uniform/'+p),'lib/core/scenes.ts','lib/core/cm12-paper-scenes.ts'].sort();return createHash('sha256').update(files.map(path=>`${path}:${createHash('sha256').update(readFileSync(path)).digest('hex')}`).join('\n')).digest('hex');};
const sourceFingerprint=fingerprint(),qualityEvery=Number(arg('quality-every','0')),qualitySnapshots:unknown[]=[];
const frames=Number(arg('frames','600')),every=Number(arg('every','3')),out=arg('out','/tmp/settling.json');
const control=arg('control','baseline'),switchFrame=Number(arg('switch-frame','0')),dt=Number(arg('dt',String(1/30)));
assert.ok(Number.isFinite(dt)&&dt>0);
const values=resolveMethodValues(uniformVolumeMethod,'balanced',JSON.parse(arg('values','{}')));
UNIFORM_PRESSURE_BAND_SCHEDULE.cycles=Number(arg('band-cycles','4'));
console.log('Waiting for repository WebGPU lease');
while(await readWebGPUExclusiveLockHolder()) await new Promise(r=>setTimeout(r,500));
await acquireWebGPUExclusiveLock('dawn-probe','uniform settling');
let device:GPUDevice|undefined,solver:WebGPUUniformReferenceSolver|undefined;
const snapshots=new Map<string,GPUTexture>();
try{
 const dawn=await import(pathToFileURL(resolve('node_modules/webgpu/index.js')).href) as NodeDawnProvider;Object.assign(globalThis,dawn.globals);
 const gpu=createProcessRetainedDawnGPU(dawn,['backend=metal']),adapter=await gpu.requestAdapter();assert.ok(adapter);
 device=managedGPUDevice(await adapter.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter.limits)}),{requireWorkerRealm:false});
 const errors:string[]=[];device.addEventListener('uncapturederror',e=>{e.preventDefault();errors.push(e.error.message);});
 let replacements=0;
 if(control==='no-correction'){
  const compiler=gpuCompilationManagerFor(device),create=compiler.createShaderModule.bind(compiler);
  compiler.createShaderModule=d=>create({...d,code:d.code.replace('return min(uvVolumeCorrectionFractionAt(dt)*max(0.0,v-cap),cap);',()=>{replacements++;return 'return 0.0;';})});
 }
 if(control==='every-step'||control==='every-step-cached'){
  const compiler=gpuCompilationManagerFor(device),create=compiler.createShaderModule.bind(compiler);
  compiler.createShaderModule=d=>{const code=redistanceEveryStepReference(d.code);if(code!==d.code)replacements++;return create({...d,code});};
 }
 const scene=sceneDocument(getSceneDefinition(arg('scene','minimal-power-dam-break-64')));
 scene.numerics.fixedDt_s=scene.numerics.maxDt_s=dt;
 if(arg('initial','dam')==='rest'){scene.fluid.initialCondition='tank-fill';delete scene.fluid.initialBrickSeeds_m;if(process.argv.some(a=>a.startsWith('--rest-depth=')))scene.container.fillFraction=Number(arg('rest-depth','2'))*scene.voxelDomain.finestCellSize_m/scene.container.height_m;}
 solver=await uniformVolumeMethod.createSolverAsync!(device,scene,'balanced',values,undefined,()=>{}) as WebGPUUniformReferenceSolver;
 if(control==='no-correction'||(control==='every-step'||control==='every-step-cached'))assert.ok(replacements>0);
 // Private access is confined to this diagnostic; copies are encoded before
 // scratch reuse and before the asynchronous dynamic relayout.
 const frame=(solver as any).mixedFrame,f=frame.fields;
 if(control==='every-step'||control==='cadence-inline'){
  const encode=frame.forces.encode.bind(frame.forces);frame.forces.encode=(e:GPUCommandEncoder,g:GPUBindGroup,capillarity:boolean)=>encode(e,g,capillarity,false);
 }
 if(control==='no-redistance'){const advance=frame.advance.bind(frame);frame.advance=(p:any,...rest:any[])=>advance({...p,redistance:false},...rest);}
 if(control==='no-capillarity'){const advance=frame.advance.bind(frame);frame.advance=(p:any,...rest:any[])=>advance({...p,surfaceTension:0},...rest);}
 let sampled=false;
 const snapshot=(e:GPUCommandEncoder,key:string,source:GPUTexture)=>{
  if(!sampled)return;
  let target=snapshots.get(key);if(!target){target=device!.createTexture({label:`Settling ${key}`,size:[source.width,source.height,source.depthOrArrayLayers],dimension:'3d',format:source.format,usage:GPUTextureUsage.COPY_SRC|GPUTextureUsage.COPY_DST});snapshots.set(key,target);}
  e.copyTextureToTexture({texture:source},{texture:target},[source.width,source.height,source.depthOrArrayLayers]);
 };
 const wrap=(object:any,key:string,before:(e:GPUCommandEncoder)=>void,after:(e:GPUCommandEncoder)=>void)=>{
  const original=object[key].bind(object);object[key]=(...args:any[])=>{before(args[0]);const result=original(...args);after(args[0]);return result;};
 };
 wrap(frame.momentum,'encode',e=>{snapshot(e,'volume',f.volume);snapshot(e,'centerPhi',f.centerPhi);snapshot(e,'target',f.target);},e=>snapshot(e,'advected',f.departure));
 wrap(frame.forces,'encode',()=>{},e=>snapshot(e,'forced',f.velocityScratch));
 wrap(frame.band,'encodeSolve',e=>snapshot(e,'bulk',f.velocity),e=>snapshot(e,'projected',f.velocity));
 wrap(frame.transport,'encodeTransport',()=>{},e=>snapshot(e,'transportVolume',f.volume));
 wrap(frame.band,'encodePrepare',e=>snapshot(e,'correction',f.correction),()=>{});
 const rows:any[]=[];
 for(let step=1;step<=frames;step++){
  if(step===switchFrame){
   solver.applyRuntimeValues({...values,...JSON.parse(arg('switch-values','{}'))});
   if(control==='freeze-layout-late')(solver as any).mixedDynamic.encode=()=>{};
   if(control.startsWith('unforced-')){const advance=frame.advance.bind(frame);frame.advance=(p:any,...rest:any[])=>advance({...p,gravity:0,viscosity:0,surfaceTension:0},...rest);}
   if(control==='no-correction-late'||control==='unforced-no-correction-late'){
    const write=frame.write.bind(frame);frame.write=(p:any)=>{write(p);device!.queue.writeBuffer(frame.params.authority,0,new Float32Array([0]));};
   }
  }
  sampled=step%every===0||step<=3;
  const layout:UniformMixedLayout=frame.ownership.capacity;
  assert.ok(solver.advanceTo(step*dt));await solver.awaitFrameCompletion();
  if(!sampled)continue;
  const tiles=await readMixedTileWords(device,solver);
  const data:Record<string,Float32Array>={};for(const [k,t]of snapshots)data[k]=await readMixedTexture(device,t);
  const [nx,ny]=layout.lattice.dimensions,[tx,ty]=layout.tileDimensions,h=scene.voxelDomain.finestCellSize_m;
  let geometryMass=0,geometryError=0;const centerOfMass=[0,0,0];
  let mass=0,pe=0,excess=0,maxV=0,correctionAbs=0,correctionMax=0,transportExcess=0,transportMaxV=0;let maxVAt:number[]=[];let maxVPhi=0,airExcess=0,airMass=0,transportPe=0,transportMass=0;
  const stages:Record<string,{ke:number;max:number;at:number[]}>={};for(const k of ['advected','forced','bulk','projected'])stages[k]={ke:0,max:0,at:[]};
  for(let t=0;t<tiles.length;t++){
   const w=mixedCellWidth(tiles[t]!),ox=4*(t%tx),oy=4*(Math.floor(t/tx)%ty),oz=4*Math.floor(t/(tx*ty));
   for(let dz=0;dz<4;dz+=w)for(let dy=0;dy<4;dy+=w)for(let dx=0;dx<4;dx+=w){
    const x=ox+dx,y=oy+dy,z=oz+dz,i=x+nx*(y+ny*z),v=data.volume![i]!,m=Math.max(0,v)*w**3;
    geometryMass+=data.target![i]!*w**3;geometryError+=Math.abs(v-data.target![i]!)*w**3;centerOfMass[0]!+=m*(x+w/2);centerOfMass[1]!+=m*(y+w/2);centerOfMass[2]!+=m*(z+w/2);
    mass+=m;pe+=m*(y+w/2)*h*9.80665;excess+=Math.max(0,v-1)*w**3;if(v>maxV){maxV=v;maxVAt=[x,y,z];maxVPhi=data.centerPhi![i]!/h;}const transported=data.transportVolume![i]!;transportMaxV=Math.max(transportMaxV,transported);transportExcess+=Math.max(0,transported-1)*w**3;transportMass+=Math.max(0,transported)*w**3;transportPe+=Math.max(0,transported)*w**3*(y+w/2)*h*9.80665;
    if(data.centerPhi![i]!>=0){airExcess+=Math.max(0,v-1)*w**3;airMass+=m;}const c=data.correction?.[i]??0;correctionAbs+=Math.abs(c)*w**3;correctionMax=Math.max(correctionMax,Math.abs(c));
    for(const k of Object.keys(stages)){
     const a=data[k]!;if(!a)continue;const components=[0,0,0];
     for(let axis=0;axis<3;axis++){
      const q=[x,y,z],n=[x,y,z];n[axis]!+=w;
      const dims=layout.lattice.dimensions;
      const neighbor=n[axis]!<dims[axis]! ? mixedCellWidth(tiles[Math.floor(n[0]!/4)+tx*(Math.floor(n[1]!/4)+ty*Math.floor(n[2]!/4))]!) : w;
      const patch=Math.min(w,neighbor),u=(axis+1)%3,b=(axis+2)%3;
      q[axis]!+=w-1;
      for(let du=0;du<w;du+=patch)for(let dv=0;dv<w;dv+=patch){const at=[...q];at[u]!+=du;at[b]!+=dv;components[axis]!+=a[4*(at[0]!+nx*(at[1]!+ny*at[2]!))+axis]!*patch*patch/(w*w);}
     }
     const speed=Math.hypot(...components);const s=stages[k]!;s.ke+=.5*m*speed**2;
     if(v>.01&&speed>s.max){s.max=speed;s.at=[x,y,z];}
    }
   }
  }
  if(qualityEvery>0&&(step===1||step%qualityEvery===0||step===frames)){
   qualitySnapshots.push({frame:step,time_s:step*dt,...uniformQualityCensus(layout.lattice.dimensions,tiles,data.volume!,await readMixedTexture(device,f.phi))});
  }
  const stats=await solver.readStats();
  const row={step,t:step*dt,mass,geometryMass,geometryError,centerOfMass:centerOfMass.map(x=>x/mass),pe,excess,maxV,maxVAt,maxVPhi,airExcess,airMass,transportMass,transportPe,transportMaxV,transportExcess,correctionAbs,correctionMax,stages,fine:tiles.reduce((n,t)=>n+Number(mixedCellWidth(t)===1),0),stats:Object.fromEntries(Object.entries(stats).filter(([k])=>/uniformPressure|uniformMixedDynamicChanged/.test(k)))};
  rows.push(row);if(step%30===0||step<=3)console.log(JSON.stringify(row));
  await writeFile(out,JSON.stringify({sourceFingerprint,sourceFingerprintAfter:fingerprint(),qualitySnapshots,control,values,scene,replacements,bandCycles:UNIFORM_PRESSURE_BAND_SCHEDULE.cycles,energyNote:'Owner-volume-weighted positive-face velocity proxy, omits negative-boundary slabs; positive faces are area-averaged across coarse/fine seams; stage comparisons share the same weights.',rows},null,2));
  assert.deepEqual(errors,[]);
 }
}finally{solver?.destroy();for(const t of snapshots.values())t.destroy();device?.destroy();await releaseWebGPUExclusiveLock();}
