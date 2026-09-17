/** Serial, reset-state A/B; GPU timestamps exclude readback and construction. */
import assert from "node:assert/strict";
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import { pathToFileURL } from "node:url";
import { getScenePreset } from "../lib/core/scenes";
import { sceneAtContainerExtents } from "../lib/core/scene-scale";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { sparseCM12DawnDefaultOptions } from "../lib/harness/sparse-cm12-dawn-defaults";
import { WebGPUAdaptiveMassSolver } from "../lib/methods/adaptive-volume/webgpu-adaptive-mass-solver";

const arg=(key:string,fallback:string)=>process.argv.find(v=>v.startsWith(`--${key}=`))?.slice(key.length+3)??fallback;
const steps=Number(arg("steps","5"));const thin=arg("thin","0")==="1";
const output=arg("out","artifacts/level-set-volume/air-extension-3d.json");
const report:{thin:boolean;steps:number;arms:unknown[];error?:string}={thin,steps,arms:[]};
await acquireWebGPUExclusiveLock("dawn-probe","air-band reset-state A/B");
let device:GPUDevice|undefined,solver:WebGPUAdaptiveMassSolver|undefined;
try{
  const dawn=await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href);Object.assign(globalThis,dawn.globals);
  const gpu=createProcessRetainedDawnGPU(dawn,[`backend=${process.env.FLUID_WEBGPU_BACKEND??"metal"}`]);
  const adapter=await gpu.requestAdapter();assert.ok(adapter);
  device=await adapter.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter.limits),requiredFeatures:["timestamp-query"]});
  const markerModule=device.createShaderModule({code:`
@group(0)@binding(0)var<storage,read_write>marker:array<atomic<u32>>;
@group(0)@binding(1)var<storage,read_write>velocity:array<atomic<u32>>;
@compute @workgroup_size(1)fn mark(){atomicAdd(&marker[0],0u);atomicAdd(&velocity[0],0u);}`});
  const markerPipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module:markerModule,entryPoint:"mark"}});
  const errors:string[]=[];device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);});
  for(const enabled of (arg("arm","both")==="on"?[true]:arg("arm","both")==="off"?[false]:[false,true])){
    let scene=getScenePreset(arg("scene","coarse-first-pool-impact-half")).create();
    // Same tank cross-section and physical head, with a genuinely 3D drop or
    // a thin extrusion. Both arms begin from independently constructed state.
    if(thin)scene=sceneAtContainerExtents(scene,{width_m:3.2,height_m:2.4,depth_m:.4});
    const dt=1/30;scene.numerics.fixedDt_s=scene.numerics.maxDt_s=dt;
    solver=await WebGPUAdaptiveMassSolver.createCompiledTopologyTransport(device,scene,"balanced",undefined,
      {...sparseCM12DawnDefaultOptions(),airExtensionEnabled:enabled},()=>{});
    await solver.waitForSimulationReady();
    const frames:unknown[]=[];report.arms.push({enabled,frames});
    for(let step=1;step<=steps;step++){
      const query=device.createQuerySet({type:"timestamp",count:6});
      const resolve=device.createBuffer({size:256,usage:GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC});
      const readback=device.createBuffer({size:48,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
      let seen=0;
      solver.setStageCaptureForQA((stage,encoder)=>{
        const index=["transport-velocity-extension","face-preparation","pressure-solve","velocity-projection","conservative-transport","presentation-publication"].indexOf(stage);
        if(index<0)return;
        const source=solver!.fieldSnapshotSourceForQA;
        const markerGroup=device!.createBindGroup({layout:markerPipeline.getBindGroupLayout(0),entries:[
          {binding:0,resource:{buffer:source.state}},{binding:1,resource:{buffer:source.effectiveTransportVelocity!}}]});
        const pass=encoder.beginComputePass({timestampWrites:{querySet:query,beginningOfPassWriteIndex:index}});
        pass.setPipeline(markerPipeline);pass.setBindGroup(0,markerGroup);pass.dispatchWorkgroups(1);pass.end();seen|=1<<index;
      });
      const start=performance.now();while(!solver.advanceTo(step*dt,[]))await new Promise<void>(setImmediate);
      await solver.awaitFrameCompletion();await solver.waitForTopologyReady();solver.setStageCaptureForQA(undefined);
      const wallMs=performance.now()-start;assert.equal(seen,63);
      const encoder=device.createCommandEncoder();encoder.resolveQuerySet(query,0,6,resolve,0);
      encoder.copyBufferToBuffer(resolve,0,readback,0,48);device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);const stamps=Array.from(new BigUint64Array(readback.getMappedRange()));
      const ms=(a:number,b:number)=>Number(stamps[b]!-stamps[a]!)/1e6;
      const [air,volume,transport]=await Promise.all([solver.readAirExtensionReceiptQA(),solver.readAcceptedGeometricVolumeQA(),solver.readGeometricVolumeTransportReceiptQA()]);
      frames.push({step,wallMs,momentumPreparationGpuMs:ms(0,1),projectionAndExtensionGpuMs:ms(2,3),transportGpuMs:ms(3,4),tailGpuMs:ms(4,5),air,volume,coupling:transport.coupling});
      readback.unmap();readback.destroy();resolve.destroy();query.destroy();
      mkdirSync(dirname(output),{recursive:true});writeFileSync(output,JSON.stringify(report,null,2)+"\n");
    }
    solver.destroy();solver=undefined;assert.deepEqual(errors,[]);
  }
}catch(error){report.error=error instanceof Error ? `${error.name}: ${error.message}\n${error.stack??""}` : String(error);process.exitCode=1;}
finally{solver?.destroy();device?.destroy();await releaseWebGPUExclusiveLock();mkdirSync(dirname(output),{recursive:true});writeFileSync(output,JSON.stringify(report,null,2)+"\n");console.log(JSON.stringify({output,error:report.error}));}
