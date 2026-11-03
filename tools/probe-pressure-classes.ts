/** Pressure convergence by row class: runs a scene (paddleboats bodies driven by default) with
 * UniformPressureArena.diagnostics and optional --<schedule key>= overrides, and prints mean cycles,
 * final residuals and, per measured cycle (0..3, then the final measure) and row class, the mean
 * log10 of the largest residual, the frames, the mean rows past the target and the argmax class.
 * Classes: 0 band deep interior, 1 band seam, 2 band wall, 3 band inside-solid or air, 4 native
 * interior, 5 native seam, 6 native wall, 7 band free-surface (or BULK16). Run from the repository
 * root (or set FLUID_ROOT) through tools/gpu/fluid-gpu-run.sh check. */
import assert from "node:assert/strict";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
const R=process.env.FLUID_ROOT??process.cwd();
const { resolveMethodValues } = await import(R+"/lib/core/method-contract");
const { sceneDocument } = await import(R+"/lib/core/scene-definition");
const { getSceneDefinition } = await import(R+"/lib/core/scenes");
const { managedGPUDevice } = await import(R+"/lib/core/gpu-compilation-manager");
const { requiredFluidDeviceLimits } = await import(R+"/lib/core/webgpu-device-limits");
const { createProcessRetainedDawnGPU, paceDawnEventPolling } = await import(R+"/lib/harness/node-dawn-provider");
const { uniformNarrowBandMethod: method } = await import(R+"/lib/methods/uniform/uniform-narrow-band-method");
const { UNIFORM_PRESSURE_ARENA_SCHEDULE: schedule, UniformPressureArena } = await import(R+"/lib/methods/uniform/uniform-pressure-arena");
UniformPressureArena.diagnostics=true;
const { mooredMotorBodies } = await import(R+"/tools/moored-motor-bodies");
const option=(key:string)=>process.argv.find(a=>a.startsWith(`--${key}=`))?.slice(key.length+3);
for(const key of Object.keys(schedule)){const v=option(key);if(v!==undefined)(schedule as any)[key]=isNaN(+v)?v:+v;}
const frames=Number(option("frames")??90),from=Number(option("from")??31);
paceDawnEventPolling();
const dawn=await import(pathToFileURL(resolve(R+"/node_modules/webgpu/index.js")).href);Object.assign(globalThis,dawn.globals);
const gpu=createProcessRetainedDawnGPU(dawn,["backend=metal"]),adapter=await gpu.requestAdapter();assert.ok(adapter);
const raw=await adapter.requestDevice({requiredFeatures:[...(adapter.features.has("subgroups")?["subgroups" as const]:[])],requiredLimits:requiredFluidDeviceLimits(adapter.limits)});
const device=managedGPUDevice(raw,{requireWorkerRealm:false});
const definition=getSceneDefinition(option("scene")??"paddleboats"),quality=definition.methodProfile!.quality;
const scene=structuredClone(sceneDocument(definition));
const values=resolveMethodValues(method,quality,{...method.appDefaults,...definition.methodProfile!.overrides});
const dt=method.effectiveStep_s?.(scene,values)??scene.numerics.fixedDt_s;
const solver=await method.createSolverAsync!(device,scene,quality,values,undefined,()=>{}) as any;
const bodies=(option("scene")??"paddleboats")==="paddleboats"?mooredMotorBodies(scene,dt):undefined;
const read=async(buffer:GPUBuffer,size:number)=>{const s=raw.createBuffer({size,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});const e=raw.createCommandEncoder();e.copyBufferToBuffer(buffer,0,s,0,size);raw.queue.submit([e.finish()]);await s.mapAsync(GPUMapMode.READ);const o=new Uint32Array(s.getMappedRange().slice(0));s.destroy();return o;};
const f=(u:number)=>new Float32Array(new Uint32Array([u]).buffer)[0]!;
const NC=8,NY=8,HB=63;const dMax:number[][]=Array.from({length:NY},()=>Array(NC).fill(0)),dN:number[][]=Array.from({length:NY},()=>Array(NC).fill(0)),dCnt:number[][]=Array.from({length:NY},()=>Array(NC).fill(0)),dArg:number[][]=Array.from({length:NY},()=>Array(NC).fill(0));
let cycles=0,n=0,logSum=0,worst=0;const finals:number[]=[];const lg=[0,0,0,0,0,0],lc=[0,0,0,0,0,0];const cyc=[0,0,0,0,0,0];
for(let frame=1;frame<=frames;frame++){
 assert.ok(solver.advanceTo(frame*dt,bodies?.(frame*dt)));await solver.awaitFrameCompletion();
 if(frame<from)continue;
 const h=await read(solver.mixedFrame.band.index,4*(HB+2*NC*NY));
 for(let y=0;y<NY;y++){let best=-1,bv=0;for(let k=0;k<NC;k++){const v=f(h[HB+NC*y+k]!);if(h[HB+NC*y+k]!>0){dMax[y]![k]!+=Math.log10(Math.max(v,1e-12));dN[y]![k]!++;if(v>bv){bv=v;best=k;}}dCnt[y]![k]!+=h[HB+NC*NY+NC*y+k]!;}if(best>=0)dArg[y]![best]!++;}const c=h[4]!;const hist=[...h.slice(8,13)].map(f);
 const fin=hist[Math.min(c,4)]||hist[c-1]!;cycles+=c;n++;cyc[Math.min(c,5)]!++;{const init=f(h[26]!);const seq=[init,...hist.slice(0,c+1)];for(let k=0;k<seq.length&&k<6;k++){if(seq[k]!>0){lg[k]!+=Math.log10(seq[k]!);lc[k]!++;}}}finals.push(fin);logSum+=Math.log10(Math.max(fin,1e-12));worst=Math.max(worst,fin);
}
finals.sort((a,b)=>a-b);
console.log(JSON.stringify({schedule,frames:n,meanCycles:+(cycles/n).toFixed(3),geoMeanFinal:+(10**(logSum/n)).toPrecision(4),medianFinal:+finals[n>>1]!.toPrecision(4),worstFinal:+worst.toPrecision(4),drift:solver.info?.uniformVolumeDrift,cycleHistogram:cyc,diag:{log10max:dMax.map((r,y)=>r.map((v,k)=>dN[y]![k]?+(v/dN[y]![k]!).toFixed(2):null)),framesWith:dN,meanRowsOverTarget:dCnt.map(r=>r.map(v=>+(v/n).toFixed(1))),argmax:dArg},meanLog10:lg.map((v,k)=>lc[k]?+(v/lc[k]!).toFixed(3):null),counts:lc}));
solver.destroy();await raw.queue.onSubmittedWorkDone();raw.destroy();
