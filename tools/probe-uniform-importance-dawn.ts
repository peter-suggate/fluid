/** Detail importance census for Uniform Geometric's Dynamic detail: what each
 * importance criterion (UNIFORM_DETAIL_CRITERIA) scores and triggers over a
 * run, read from the census's own words (the Detail importance layer's data).
 *
 * --scene=sparse-cm12-ladder-long-dam --frames=90 --every=15 --hz=60
 * --arms='[{"name":"app","values":{...}},{"name":"all","values":{...},"views":true}]'
 *   Each arm is a fresh solver on one device (one lease, shared compilation).
 *   values are method overrides over the declared defaults; views records
 *   every criterion's score as the layer does (criteria that are off too).
 *   Without --arms: "declared" (the lanes' census), "app" (what Dynamic
 *   opens on in the app) and "all" (every criterion, bulk, a 30% budget,
 *   views on).
 *
 * Per sampled frame and arm, one JSON line: h/4h tiles, wet and surface
 * tiles, required/held/dropped tiles, and per criterion the tiles it
 * triggers and the mean and largest score over surface tiles (wet tiles for
 * strain and rotation). A pipeline error stops the arm and is reported.
 *
 * WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js node --import tsx tools/probe-uniform-importance-dawn.ts
 */
import assert from "node:assert/strict";
import {pathToFileURL} from "node:url";
import {managedGPUDevice} from "../lib/core/gpu-compilation-manager";
import {requiredFluidDeviceLimits} from "../lib/core/webgpu-device-limits";
import {createProcessRetainedDawnGPU} from "../lib/harness/node-dawn-provider";
import {sceneDocument} from "../lib/core/scene-definition";
import {getSceneDefinition} from "../lib/core/scenes";
import type {MethodParamValues} from "../lib/core/method-contract";
import {WebGPUUniformReferenceSolver} from "../lib/methods/uniform/webgpu-uniform-reference";
import {uniformGeometricSolverOptions} from "../lib/methods/uniform/uniform-geometric-options";
import {UNIFORM_DETAIL_APP_IMPORTANCE} from "../lib/methods/uniform/uniform-detail-policy";
import {UNIFORM_DETAIL_CRITERIA,UNIFORM_STAGE_IMPORTANCE as I} from "../lib/methods/uniform/uniform-stage-grids";

const arg=(key:string,fallback:string)=>process.argv.find(a=>a.startsWith(`--${key}=`))?.slice(key.length+3)??fallback;
const sceneId=arg("scene","sparse-cm12-ladder-long-dam"),frames=Number(arg("frames","90")),every=Number(arg("every","15")),hz=Number(arg("hz","60"));
interface Arm{name:string;values:MethodParamValues;views?:boolean}
const everything={detailThin:"on",detailStrain:"on",detailRotation:"on",detailApproach:"on",detailBulk:"on"};
const arms:Arm[]=JSON.parse(arg("arms","null"))??[
 {name:"declared",values:{}},
 {name:"app",values:{...UNIFORM_DETAIL_APP_IMPORTANCE}},
 {name:"all",values:{...UNIFORM_DETAIL_APP_IMPORTANCE,...everything,detailBudgetPercent:30},views:true},
];
const modulePath=process.env.WEBGPU_NODE_MODULE;assert.ok(modulePath,"WEBGPU_NODE_MODULE is required");

let device:GPUDevice|undefined;
try{
 const dawn=await import(pathToFileURL(modulePath).href);Object.assign(globalThis,dawn.globals);
 const adapter=await createProcessRetainedDawnGPU(dawn,["backend=metal"]).requestAdapter();assert.ok(adapter);
 device=managedGPUDevice(await adapter.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter.limits)}),{requireWorkerRealm:false});
 const errors:string[]=[];device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);});
 for(const arm of arms){
  const scene=sceneDocument(getSceneDefinition(sceneId));
  scene.numerics.fixedDt_s=scene.numerics.maxDt_s=1/hz;
  const started=performance.now();
  const solver=await WebGPUUniformReferenceSolver.createAsync(device,scene,"balanced",undefined,uniformGeometricSolverOptions({timeStep:"scene",detailPolicy:"dynamic",...arm.values},scene),()=>{});
  try{
   if(arm.views)solver.setLayoutViewsEnabled(true);
   // eslint-disable-next-line @typescript-eslint/no-explicit-any
   const census=(solver as any).mixedDynamic,source=census.importance as {buffer:GPUBuffer;offset:number};
   const tiles:number=census.ownership.capacity.tiles;
   const read=device.createBuffer({size:8*tiles,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
   console.log(JSON.stringify({arm:arm.name,scene:sceneId,tiles,build_s:+((performance.now()-started)/1000).toFixed(1),values:arm.values,views:!!arm.views}));
   let step_ms=0;
   for(let step=1;step<=frames;step++){
    const t0=performance.now();
    const advanced=solver.advanceTo(step/hz,[]);await solver.awaitFrameCompletion?.();step_ms+=performance.now()-t0;
    const error=solver.info.simulationPipelineError;
    if(!advanced||error||errors.length){console.log(JSON.stringify({arm:arm.name,step,stopped:error??errors[0]??"advance refused"}));break;}
    if(step%every&&step!==1&&step!==frames)continue;
    const encoder=device.createCommandEncoder();encoder.copyBufferToBuffer(source.buffer,source.offset,read,0,8*tiles);device.queue.submit([encoder.finish()]);
    await read.mapAsync(GPUMapMode.READ);const words=new Uint32Array(read.getMappedRange()).slice();read.unmap();
    let wet=0,surface=0,required=0,held=0,dropped=0,bulk=0;
    const criteria=UNIFORM_DETAIL_CRITERIA.map(()=>({triggered:0,sum:0,max:0,won:0}));
    for(let t=0;t<tiles;t++){
     const w0=words[2*t]!,w1=words[2*t+1]!;
     const isWet=(w1&I.wet)!==0,crossing=(w1&I.crossing)!==0;
     if(!isWet&&!crossing)continue;
     wet+=+isWet;surface+=+crossing;
     if(w1&I.required){required++;if(!crossing)bulk++;}
     held+=+((w1&I.held)!==0);dropped+=+((w1&I.dropped)!==0);
     const winner=(w1>>>I.winnerShift)&7;if(winner)criteria[winner-1]!.won++;
     UNIFORM_DETAIL_CRITERIA.forEach((_,k)=>{
      const score=(k<4?(w0>>>(8*k))&255:(w1>>>(8*(k-4)))&255)/I.scoreOne,c=criteria[k]!;
      c.triggered+=(w1>>>(I.triggeredShift+k))&1;c.sum+=score;c.max=Math.max(c.max,score);
     });
    }
    const info=solver.info;
    console.log(JSON.stringify({arm:arm.name,step,h:info.uniformMixedFineTiles,coarse:info.uniformMixedCoarseTiles,wet,surface,required,bulk,held,dropped,
     step_ms:+(step_ms/step).toFixed(2),
     ...Object.fromEntries(UNIFORM_DETAIL_CRITERIA.map((name,k)=>{const c=criteria[k]!;return [name,{triggered:c.triggered,won:c.won,mean:+(c.sum/Math.max(1,name==="strain"||name==="rotation"?wet:surface)).toFixed(2),max:+c.max.toFixed(2)}];}))}));
   }
   read.destroy();
  }finally{solver.destroy();}
 }
 if(errors.length)console.log(JSON.stringify({uncaptured:errors.slice(0,3)}));
}finally{device?.destroy();}
