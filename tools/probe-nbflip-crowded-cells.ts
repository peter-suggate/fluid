/** The most crowded NB-FLIP cells after a frame: reads the live samples back
 * (a diagnostic readback after the frame completes) and histograms them by
 * cell. Positions are in cells of the solver's frame (window) coordinates.
 * node_modules/.bin/tsx tools/probe-nbflip-crowded-cells.ts [--scene=ID] [--frames=30,60] [--top=12]
 * Run with the shared GPU lock. */
import assert from 'node:assert/strict';
import {resolve} from 'node:path';
import {pathToFileURL} from 'node:url';
import {resolveMethodValues} from '../lib/core/method-contract';
import {sceneDocument} from '../lib/core/scene-definition';
import {getSceneDefinition} from '../lib/core/scenes';
import {managedGPUDevice} from '../lib/core/gpu-compilation-manager';
import {requiredFluidDeviceLimits} from '../lib/core/webgpu-device-limits';
import {createProcessRetainedDawnGPU,paceDawnEventPolling} from '../lib/harness/node-dawn-provider';
import {uniformNarrowBandMethod as method} from '../lib/methods/uniform/uniform-narrow-band-method';
import type {WebGPUUniformReferenceSolver} from '../lib/methods/uniform/webgpu-uniform-reference';
const option=(key:string)=>process.argv.find(a=>a.startsWith(`--${key}=`))?.slice(key.length+3);
const frames=(option('frames')??'30,60').split(',').map(Number).sort((a,b)=>a-b),top=Number(option('top')??12);
paceDawnEventPolling();
const dawn=await import(pathToFileURL(resolve('node_modules/webgpu/index.js')).href);Object.assign(globalThis,dawn.globals);
const gpu=createProcessRetainedDawnGPU(dawn,['backend=metal']),adapter=await gpu.requestAdapter();assert.ok(adapter);
const raw=await adapter.requestDevice({requiredFeatures:adapter.features.has('subgroups')?['subgroups' as const]:[],requiredLimits:requiredFluidDeviceLimits(adapter.limits)});
const device=managedGPUDevice(raw,{requireWorkerRealm:false});
let solver:WebGPUUniformReferenceSolver|undefined;
try{
 const definition=getSceneDefinition(option('scene')??'sparse-cm12-long-dam-break'),quality=definition.methodProfile?.quality??'balanced',scene=structuredClone(sceneDocument(definition));scene.numerics.fixedDt_s=scene.numerics.maxDt_s=1/60;
 const values=resolveMethodValues(method,quality,{...method.appDefaults,...definition.methodProfile?.overrides}),dt=method.effectiveStep_s?.(scene,values)??scene.numerics.fixedDt_s;
 solver=await method.createSolverAsync!(device,scene,quality,values,undefined,()=>{}) as WebGPUUniformReferenceSolver;
 for(let frame=1;frame<=frames.at(-1)!;frame++){
  assert.ok(solver.advanceTo(frame*dt,[]));await solver.awaitFrameCompletion();
  if(!frames.includes(frame))continue;
  const flip=(solver as unknown as {mixedFrame:{narrowBandFlip:{activeParticles:GPUBuffer;count:number}}}).mixedFrame.narrowBandFlip;
  const buffer=flip.activeParticles,count=flip.count,stride=buffer.size/Math.max(1,count)>=112?112:112;
  const bytes=Math.min(buffer.size,count*stride),read=raw.createBuffer({size:Math.ceil(bytes/4)*4,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  const e=raw.createCommandEncoder();e.copyBufferToBuffer(buffer,0,read,0,Math.ceil(bytes/4)*4);raw.queue.submit([e.finish()]);
  await read.mapAsync(GPUMapMode.READ);const f=new Float32Array(read.getMappedRange().slice(0));read.unmap();read.destroy();
  const cells=new Map<string,{n:number;speed:number;w:number;tie:Set<string>}>();
  for(let i=0;i<count;i++){
   const o=i*stride/4,x=f[o]!,y=f[o+1]!,z=f[o+2]!;if(!(x>=0))continue;
   const key=`${Math.floor(x)},${Math.floor(y)},${Math.floor(z)}`;const c=cells.get(key)??{n:0,speed:0,w:0,tie:new Set<string>()};
   c.n++;c.speed=Math.max(c.speed,Math.hypot(f[o+4]!,f[o+5]!,f[o+6]!));c.w+=f[o+3]!;if(c.tie.size<4096)c.tie.add(`${x},${y},${z}`);cells.set(key,c);
  }
  const sorted=[...cells].sort((a,b)=>b[1].n-a[1].n);let squares=0;for(const [,c] of cells)squares+=c.n*c.n;
  console.log(JSON.stringify({frame,count,cells:cells.size,sumSquares:squares,over64:sorted.filter(([,c])=>c.n>64).length,samplesOver64:sorted.filter(([,c])=>c.n>64).reduce((s,[,c])=>s+c.n,0)}));
  for(const [key,c] of sorted.slice(0,top))console.log(`  cell ${key}: ${c.n} samples, ${c.tie.size} distinct positions, max speed ${c.speed.toFixed(3)}, mean w ${(c.w/c.n).toFixed(3)}`);
 }
}finally{solver?.destroy();await raw.queue.onSubmittedWorkDone();raw.destroy();}
