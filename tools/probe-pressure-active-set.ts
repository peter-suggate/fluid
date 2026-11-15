/** The old pressure arena's active-set and seam-head census, per frame: runs a scene with
 * UniformPressureArena.diagnostics (paddleboats with its bodies driven) and reads the census words
 * (UNIFORM_PRESSURE_CENSUS) and the Krylov layer's bound counters each frame. Prints, per census
 * word, the frames it was nonzero in, its mean and its largest per-frame value, and the means over
 * --windows equal windows of frames. Counts are rows or faces summed over the frame's passes (the MG
 * cycles' words over its cycles). Run from the repository root through tools/gpu/fluid-gpu-run.sh
 * check: --scene=<seep|dam|letters|paddleboats|scene id> --frames=N [--windows=W]. */
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
const { UniformPressureArena, UNIFORM_PRESSURE_CENSUS: CENSUS } = await import(R+"/lib/methods/uniform/uniform-pressure-arena");
UniformPressureArena.diagnostics=true;
const { mooredMotorBodies } = await import(R+"/tools/moored-motor-bodies");
const option=(key:string)=>process.argv.find(a=>a.startsWith(`--${key}=`))?.slice(key.length+3);
const KEYS:Record<string,string>={seep:"reservoir-seep-vast",dam:"high-resolution-dam-break",letters:"nbflip-figure-8-letters",paddleboats:"paddleboats"};
const sceneId=KEYS[option("scene")??"seep"]??option("scene")!;
const frames=Number(option("frames")??120),windows=Number(option("windows")??6);
/** The census words (lib/methods/uniform/uniform-pressure-arena.ts pCensus sites). */
const NAMES:Record<number,string>={
 0:"band.rows",1:"band.K_WALL",2:"band.K_SEAM",3:"band.inside(bounded)",4:"band.K_OPEN",5:"band.K_NEUMANN",6:"band.K_CLOSED",7:"band.rowsWithWall",
 8:"band.wallMaskChangedSinceLastFrame",9:"band.insideFlippedSinceLastFrame",10:"band.frozenAtLastFrameEnd",
 11:"native.rows",12:"native.N_WALL",13:"native.N_HELD",14:"native.N_CLOSED",15:"native.N_SEAM",16:"seam.subfaces",17:"seam.subfacesHeadNonzero",
 19:"native.N_BULKN",20:"native.N_WALL.liftBelowStart(init)",21:"native.closedClamp(init)",22:"native.wallHeldClosedChangedSinceLastFrame",23:"native.rowsWithWall",
 24:"mg.band.K_WALL.binds",25:"mg.native.N_WALL.binds",26:"mg.band.closedClamp",27:"mg.native.closedClamp",28:"mg.band.atBound",29:"mg.native.atBound",
 30:"krylov.start.band.frozen",31:"krylov.start.native.frozen",32:"krylov.start.band.K_WALL.binds",33:"krylov.start.native.N_WALL.binds",
 34:"mg.band.cycle0FrozenVsLastFrameFlips",35:"mg.band.rowsEvaluated",
 37:"final.band.K_WALL.stateVsBuild",38:"final.native.N_WALL.stateVsBuild",39:"build.band.K_WALL.separated",
};
const FLOAT:Record<number,string>={18:"seam.maxAbsHead",36:"seam.maxAbsCoeffTimesHead"};
for(let c=0;c<8;c++){NAMES[40+c]=`mg.band.frozen.cycle${c}`;NAMES[48+c]=`mg.band.correctChanges.cycle${c}`;NAMES[56+c]=`mg.native.frozen.cycle${c}`;NAMES[64+c]=`mg.native.correctChanges.cycle${c}`;}
/** The Krylov layer's header words (KRYLOV_* in the arena) and the cycles word. */
const HEADER_WORDS:Record<number,string>={4:"cycles",54:"final.boundedAtBoundAskingToLeave",55:"final.underBound",56:"krylov.joins",57:"krylov.crosses",59:"krylov.hits",61:"krylov.restarts"};
paceDawnEventPolling();
const dawn=await import(pathToFileURL(resolve(R+"/node_modules/webgpu/index.js")).href);Object.assign(globalThis,dawn.globals);
const gpu=createProcessRetainedDawnGPU(dawn,["backend=metal"]),adapter=await gpu.requestAdapter();assert.ok(adapter);
const raw=await adapter.requestDevice({requiredFeatures:[...(adapter.features.has("subgroups")?["subgroups" as const]:[])],requiredLimits:requiredFluidDeviceLimits(adapter.limits)});
const device=managedGPUDevice(raw,{requireWorkerRealm:false});
const definition=getSceneDefinition(sceneId),profile=definition.methodProfile?.methodId===method.id?definition.methodProfile:undefined,quality=profile?.quality??"balanced";
const scene=structuredClone(sceneDocument(definition));
const values=resolveMethodValues(method,quality,{...method.appDefaults,...profile?.overrides});
const dt=method.effectiveStep_s?.(scene,values)??scene.numerics.fixedDt_s;
const solver=await method.createSolverAsync!(device,scene,quality,values,undefined,()=>{}) as any;
const bodies=sceneId==="paddleboats"?mooredMotorBodies(scene,dt):undefined;
const size=4*(CENSUS.word+CENSUS.words);
const read=async(buffer:GPUBuffer)=>{const s=raw.createBuffer({size,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});const e=raw.createCommandEncoder();e.copyBufferToBuffer(buffer,0,s,0,size);raw.queue.submit([e.finish()]);await s.mapAsync(GPUMapMode.READ);const o=new Uint32Array(s.getMappedRange().slice(0));s.destroy();return o;};
const f32=(u:number)=>new Float32Array(new Uint32Array([u]).buffer)[0]!;
const series:Record<string,number[]>={};
const push=(k:string,v:number)=>{(series[k]??=[]).push(v);};
for(let frame=1;frame<=frames;frame++){
 assert.ok(solver.advanceTo(frame*dt,bodies?.(frame*dt)));await solver.awaitFrameCompletion();
 const h=await read(solver.mixedFrame.band.index);
 for(const [w,k] of Object.entries(HEADER_WORDS))push(k,h[+w]!);
 for(const [w,k] of Object.entries(NAMES))push(k,h[CENSUS.word+ +w]!);
 for(const [w,k] of Object.entries(FLOAT))push(k,f32(h[CENSUS.word+ +w]!));
}
const summary:Record<string,unknown>={};
for(const [k,v] of Object.entries(series)){
 const nz=v.filter(x=>x!==0).length;if(nz===0){summary[k]=0;continue;}
 const isMax=Object.values(FLOAT).includes(k);const mean=v.reduce((a,b)=>a+b,0)/v.length;const w=Math.ceil(v.length/windows);
 summary[k]={framesNonzero:nz,firstNonzero:v.findIndex(x=>x!==0)+1,[isMax?"meanOfFrameMax":"meanPerFrame"]:+mean.toPrecision(4),maxPerFrame:+Math.max(...v).toPrecision(4),
  windows:Array.from({length:Math.ceil(v.length/w)},(_,i)=>{const s=v.slice(i*w,(i+1)*w);return +(s.reduce((a,b)=>a+b,0)/s.length).toPrecision(3);})};
}
console.log(JSON.stringify({scene:sceneId,frames,dt,summary},null,1));
solver.destroy();await raw.queue.onSubmittedWorkDone();raw.destroy();
