/** Compare the single production kernel with a historical coarse shader, without
 * retaining a second runtime path. Run with --baseline=<git revision>.
 * All other code and options are identical across arms; compilation is excluded.
 */
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { writeFileSync } from "node:fs";
import { pathToFileURL } from "node:url";
import { resolve } from "node:path";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import { managedGPUDevice } from "../lib/core/gpu-compilation-manager";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { uniformGeometricSolverOptions } from "../lib/methods/uniform/uniform-geometric-options";
import { WebGPUUniformReferenceSolver } from "../lib/methods/uniform/webgpu-uniform-reference";
import { uniformCoarseSolverWGSL } from "../lib/methods/uniform/uniform-coarse-solver.wgsl";
import { UNIFORM_CM11A_COARSE_RESIDUAL_TOLERANCE } from "../lib/methods/uniform/pressure-policy";

const arg=(name:string)=>process.argv.find(v=>v.startsWith(`--${name}=`))?.slice(name.length+3);
const baselineRef=arg("baseline");assert.ok(baselineRef,"Pass --baseline=<git revision containing the original shared-memory kernel>");
const revision=execFileSync("git",["rev-parse","--verify",`${baselineRef}^{commit}`],{encoding:"utf8"}).trim();
const historical=execFileSync("git",["show",`${revision}:lib/methods/uniform/webgpu-uniform-pressure-multigrid.wgsl.ts`],{encoding:"utf8"});
const start=historical.indexOf("var<workgroup> mgCoarseP:array<f32,256>;");
const end=historical.indexOf("@compute @workgroup_size(4,4,4)\nfn mgMeasureFineResidual",start);
assert.ok(start>=0&&end>start,"Reference must contain the original coarse shader");
const baseline=historical.slice(start,end)
  .replaceAll("${UNIFORM_CM11A_COARSE_RESIDUAL_TOLERANCE}",String(UNIFORM_CM11A_COARSE_RESIDUAL_TOLERANCE))
  .replaceAll("mgConvergence[","mgState.convergence[")
  .replaceAll("<<16u","<<30u").replaceAll("<<17u","<<31u");
const median=(values:number[])=>{const a=[...values].sort((x,y)=>x-y);return (a[Math.floor((a.length-1)/2)]+a[Math.floor(a.length/2)])/2;};
const frames=Number(arg("frames")??20), warmup=5;
assert.ok(Number.isInteger(frames)&&frames>warmup);
const report: {baseline:string; runs:unknown[]}={baseline:revision,runs:[]};
await acquireWebGPUExclusiveLock("dawn-probe","single-path coarse solver comparison");
try {
  const dawn=await import(pathToFileURL(resolve("node_modules/webgpu/index.js")).href);Object.assign(globalThis,dawn.globals);
  const gpu=createProcessRetainedDawnGPU(dawn,[`backend=${process.env.FLUID_WEBGPU_BACKEND??"metal"}`,"disable-dawn-features=timestamp_quantization"]);
  for(const sceneId of ["minimal-power-dam-break-32","minimal-power-dam-break-64"]) {
    for(const arm of ["original","strided","strided","original"] as const) {
      // Separate devices avoid compilation-cache reuse across injected shaders.
      const adapter=await gpu.requestAdapter();assert.ok(adapter);
      const raw=await adapter.requestDevice({requiredFeatures:["timestamp-query"],requiredLimits:requiredFluidDeviceLimits(adapter.limits)});
      let injected=0;
      const createModule=raw.createShaderModule.bind(raw);
      raw.createShaderModule=descriptor=>{
        if(arm==="original"&&descriptor.code.includes(uniformCoarseSolverWGSL)) {
          injected++;return createModule({...descriptor,code:descriptor.code.replace(uniformCoarseSolverWGSL,baseline)});
        }
        return createModule(descriptor);
      };
      const device=managedGPUDevice(raw,{requireWorkerRealm:false});
      const errors:string[]=[];device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);});
      let solver:WebGPUUniformReferenceSolver|undefined;
      const query=device.createQuerySet({type:"timestamp",count:512});
      const resolved=device.createBuffer({size:4096,usage:GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC});
      const staging=device.createBuffer({size:4096,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
      try {
        const scene=sceneDocument(getSceneDefinition(sceneId));
        solver=await WebGPUUniformReferenceSolver.createAsync(device,scene,"balanced",undefined,uniformGeometricSolverOptions({},scene),()=>{});
        assert.equal(injected,arm==="original"?1:0);
        let queryCount=0;
        // Wrap the hierarchy encoder to timestamp every coarse invocation.
        const pressure=(solver as unknown as {pressureMultigrid:{encode(e:GPUCommandEncoder,...args:unknown[]):void}}).pressureMultigrid;
        const encode=pressure.encode.bind(pressure);
        pressure.encode=(encoder,...args)=>encode(new Proxy(encoder,{get(target,key){
          if(key==="beginComputePass")return (descriptor:GPUComputePassDescriptor)=>{
            if(descriptor.label==="Uniform CM11a mgSolveCoarsest") {
              assert.ok(queryCount+2<=512);
              return target.beginComputePass({...descriptor,timestampWrites:{querySet:query,
                beginningOfPassWriteIndex:queryCount++,endOfPassWriteIndex:queryCount++}});
            }
            return target.beginComputePass(descriptor);
          };
          const value=Reflect.get(target,key,target);return typeof value==="function"?value.bind(target):value;
        }}),...args);
        const samples=[];
        for(let frame=1;frame<=frames;frame++) {
          queryCount=0;const startTime=performance.now();
          assert.ok(solver.advanceTo(frame/30));await device.queue.onSubmittedWorkDone();
          const wall_ms=performance.now()-startTime;assert.ok(queryCount>0,"coarse timestamps must cover every measured frame");
          const e=device.createCommandEncoder();e.resolveQuerySet(query,0,queryCount,resolved,0);
          e.copyBufferToBuffer(resolved,0,staging,0,queryCount*8);device.queue.submit([e.finish()]);
          await staging.mapAsync(GPUMapMode.READ);const times=new BigUint64Array(staging.getMappedRange());let coarse_ms=0;
          for(let i=0;i<queryCount;i+=2) {assert.ok(times[i]>0n&&times[i+1]>=times[i]);coarse_ms+=Number(times[i+1]-times[i])/1e6;}
          staging.unmap();samples.push({frame,wall_ms,coarse_ms});
        }
        const measured=samples.slice(warmup), stats=await solver.readStats();
        const result={sceneId,arm,frames,warmup,medianWall_ms:median(measured.map(s=>s.wall_ms)),
          medianCoarse_ms:median(measured.map(s=>s.coarse_ms)),stats,samples};
        report.runs.push(result);console.log(JSON.stringify(result));assert.deepEqual(errors,[]);
      } finally {solver?.destroy();query.destroy();resolved.destroy();staging.destroy();device.destroy();}
    }
  }
  if(arg("out"))writeFileSync(arg("out")!,JSON.stringify(report,null,2)+"\n");
} finally {await releaseWebGPUExclusiveLock();}
