/** Serial SPH solver benchmark. Includes publication and completion, excludes rendering.
 * node --import tsx tools/benchmark-sph.ts --out=docs/verification/sph-baseline.json
 */
import assert from "node:assert/strict";
import { writeFileSync, mkdirSync, readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { createHash } from "node:crypto";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import { managedGPUDevice } from "../lib/core/gpu-compilation-manager";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { usePerformanceInstrumentationStore } from "../lib/core/stores/performance-instrumentation-store";
import { SphSolver } from "../lib/methods/sph/solver";
const arg = (key:string, fallback:string) => process.argv.find(a=>a.startsWith(`--${key}=`))?.slice(key.length+3) ?? fallback;
const output=resolve(arg("out","docs/verification/sph-performance.json")), duration=Number(arg("duration","0.1"));
const values=JSON.parse(arg("values","{}"));
const sourceHashes=Object.fromEntries(["solver","shader","parameters"].map(name=>[name,createHash("sha256").update(readFileSync(`lib/methods/sph/${name}.ts`)).digest("hex")]));
let announced=false;
for(;;){try{await acquireWebGPUExclusiveLock("dawn-probe","SPH performance breakdown");break;}catch(error){
  if(!String(error).includes("Refusing concurrent GPU execution"))throw error;
  if(!announced){console.log("Waiting for WebGPU lease");announced=true;}await new Promise(r=>setTimeout(r,250));
}}
let device:GPUDevice|undefined;
try{
  const dawn=await import(pathToFileURL(resolve("node_modules/webgpu/index.js")).href);Object.assign(globalThis,dawn.globals);
  const gpu=createProcessRetainedDawnGPU(dawn,["backend=metal"]),adapter=await gpu.requestAdapter();assert.ok(adapter);
  device=managedGPUDevice(await adapter.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter.limits),requiredFeatures:adapter.features.has("timestamp-query")?["timestamp-query"]:[]}),{requireWorkerRealm:false});
  const errors:string[]=[];device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);});
  const scene=structuredClone(sceneDocument(getSceneDefinition("minimal-power-dam-break-32")));scene.numerics.fixedDt_s=1/60;
  const solver=await SphSolver.createAsync(device,scene,"balanced",values);
  try{
    const advance=async(target:number)=>{let encode=0,advances=0,substeps=0;while(solver.info.submittedTime_s!<target-1e-9){const start=performance.now();solver.advanceTo(target,[]);encode+=performance.now()-start;await solver.awaitFrameCompletion();advances++;substeps+=solver.info.lastSubsteps??0;}return{encode,advances,substeps};};
    await advance(0.02);const samples=[];
    for(let time=0.02;time<duration+0.02-1e-9;){time=Math.min(duration+0.02,time+1/60);const start=performance.now();const result=await advance(time);samples.push({wall:performance.now()-start,...result});}
    usePerformanceInstrumentationStore.setState({enabled:true});await advance(solver.info.submittedTime_s!+1/60);
    const total=samples.reduce((a,b)=>a+b.wall,0),info=await solver.readStats();
    const report={date:new Date().toISOString(),sourceHashes,values,scene,adapter:{vendor:adapter.info.vendor,architecture:adapter.info.architecture,device:adapter.info.device,description:adapter.info.description},methodology:"Short solver-only run: CPU encoding, GPU work, surface publication and completion receipts. Not browser FPS or matched fidelity. Tracing captured separately after timing.",duration_s:duration,wallTotal_ms:total,realtimeFactor:1000*duration/total,meanFrame_ms:total/samples.length,meanEncode_ms:samples.reduce((a,b)=>a+b.encode,0)/samples.length,samples,info};
    mkdirSync(dirname(output),{recursive:true});writeFileSync(output,JSON.stringify(report,null,2)+"\n");
    console.log(JSON.stringify({...report,scene:undefined,info:{...info,physicsTrace:undefined}}));
    assert.deepEqual(errors,[]);
  }finally{solver.destroy();}
}finally{device?.destroy();await releaseWebGPUExclusiveLock();}
