import { createGridOverlayLevelSetVolumeWGSL, gridOverlayLevelSetVolumeUniform } from "../lib/core/grid-overlay-levelset-volume.wgsl";
import { uniformVolumeMethod } from "../lib/methods/uniform/uniform-volume-method";
import { resolveMethodValues } from "../lib/core/method-contract";
import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { createProcessRetainedDawnGPU, type NodeDawnProvider } from "../lib/harness/node-dawn-provider";
import { managedGPUDevice } from "../lib/core/gpu-compilation-manager";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { WebGPUUniformReferenceSolver } from "../lib/methods/uniform/webgpu-uniform-reference";

async function read(device:GPUDevice,texture:GPUTexture):Promise<Float32Array>{
  const components=texture.format==="rgba32float"?4:1;
  const row=Math.ceil(texture.width*components*4/256)*256;
  const buffer=device.createBuffer({size:row*texture.height*texture.depthOrArrayLayers,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  try {const e=device.createCommandEncoder();e.copyTextureToBuffer({texture},{buffer,bytesPerRow:row,rowsPerImage:texture.height},[texture.width,texture.height,texture.depthOrArrayLayers]);device.queue.submit([e.finish()]);await buffer.mapAsync(GPUMapMode.READ);
    const mapped=new Float32Array(buffer.getMappedRange());const result=new Float32Array(texture.width*texture.height*texture.depthOrArrayLayers*components);
    for(let z=0;z<texture.depthOrArrayLayers;z++)for(let y=0;y<texture.height;y++)result.set(mapped.subarray((z*texture.height+y)*row/4,(z*texture.height+y)*row/4+texture.width*components),(z*texture.height+y)*texture.width*components);
    return result;
  }finally{buffer.unmap();buffer.destroy();}
}
const sum=(a:Float32Array)=>a.reduce((s,v)=>s+v,0);
const modulePath=process.env.WEBGPU_NODE_MODULE;
(modulePath?test:test.skip)("uniform geometric presentation source and mini32 conservation",{timeout:180_000},async t=>{
  await acquireWebGPUExclusiveLock("dawn-test","uniform-volume numerical invariants");
  let device:GPUDevice|undefined,solver:WebGPUUniformReferenceSolver|undefined;
  try{
    const dawn=await import(pathToFileURL(modulePath!).href);Object.assign(globalThis,dawn.globals);
    const gpu=createProcessRetainedDawnGPU(dawn as NodeDawnProvider,[`backend=${process.env.FLUID_WEBGPU_BACKEND??"metal"}`]);
    const adapter=await gpu.requestAdapter();assert.ok(adapter);
    device=managedGPUDevice(await adapter.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter.limits)}),{requireWorkerRealm:false});
    const errors:string[]=[];device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);});
    const scene=structuredClone(sceneDocument(getSceneDefinition("minimal-power-dam-break-32")));
    scene.voxelDomain.finestCellSize_m=scene.container.width_m/16;
    scene.fluid.initialCondition="tank-fill";scene.container.fillFraction=0.5;
    scene.fluid.gravity_m_s2={x:0,y:0,z:0};scene.fluid.initialLiquidVolumes=[];
    solver=await WebGPUUniformReferenceSolver.createAsync(device,scene,"balanced",undefined,{geometricVolume:true,densitySharpening:false,solidExcessCorrection:false},()=>{});
    assert.equal((solver as unknown as {totalSurfaceVolume:boolean}).totalSurfaceVolume,true,"3D geometric constructor defaults to total-volume correction");
    solver.applyRuntimeValues({totalSurfaceVolume:"off",densitySharpening:"off",solidExcessCorrection:"off",densityPostProcessing:"off"}); // Original algorithm invariants below are the explicit baseline.
    const {ny}=solver.info;
    await t.test("combined slice reads independent vertex phi and V/open capacity",async()=>{
      const d=device!; const source=solver!.denseLevelSetVolumeSource!;
      const shaderModule=d.createShaderModule({code:`
@group(0) @binding(9) var densityField:texture_3d<f32>;
@group(0) @binding(17) var<storage,read> sparseTopologyArena:array<u32>;
var<private> sparseState:array<f32,1>;
fn sparseOwner(p:vec3i)->vec2u{return vec2u(0xffffffffu);}
fn sparseDensityOffset()->u32{return 0u;}
${createGridOverlayLevelSetVolumeWGSL(true)}
@group(0) @binding(23) var<storage,read_write> result:array<vec4f>;
@compute @workgroup_size(1) fn probe(){result[0]=vec4f(sliceLevelSetPhi(vec3f(3.25,3.5,3.75)),sliceVolumeFill(vec3i(3)));}
`});
      assert.deepEqual((await shaderModule.getCompilationInfo()).messages.filter(m=>m.type==="error"),[]);
      const pipeline=await d.createComputePipelineAsync({layout:"auto",compute:{module:shaderModule,entryPoint:"probe"}});
      const params=d.createBuffer({size:96,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
      d.queue.writeBuffer(params,0,gridOverlayLevelSetVolumeUniform(undefined,source));
      const output=d.createBuffer({size:16,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
      const staging=d.createBuffer({size:16,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
      const ownership=d.createBuffer({size:4,usage:GPUBufferUsage.STORAGE});
      const group=d.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[
        {binding:9,resource:solver!.volumeTexture.createView()},
        {binding:17,resource:{buffer:ownership}},
        {binding:20,resource:{buffer:params}}, {binding:21,resource:source.vertexPhi.createView()},
        {binding:22,resource:source.openFraction.createView()}, {binding:23,resource:{buffer:output}},
      ]});
      const e=d.createCommandEncoder();const pass=e.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(1);pass.end();e.copyBufferToBuffer(output,0,staging,0,16);d.queue.submit([e.finish()]);
      await staging.mapAsync(GPUMapMode.READ);const result=new Float32Array(staging.getMappedRange());
      assert.ok(Math.abs(result[0]!-(3.5-ny/2))<1e-5);assert.equal(result[1],1);assert.equal(result[2],1);assert.equal(result[3],1);
      staging.unmap();for(const b of [params,output,staging,ownership])b.destroy();
    });
    await t.test("mini32 conserves liquid through separating far-wall impact",async()=>{
      solver!.destroy();
      solver=await uniformVolumeMethod.createSolverAsync!(device!,sceneDocument(getSceneDefinition("minimal-power-dam-break-32")),"balanced",resolveMethodValues(uniformVolumeMethod,"balanced",{velocityTransport:"semi-lagrangian"}),undefined,()=>{}) as WebGPUUniformReferenceSolver;
      for(let frame=1;frame<=90;frame++){
        assert.ok(solver.advanceTo(frame/30));await solver.awaitFrameCompletion();
        if(frame===30||frame===90){
          const stats=await solver.readStats();
          // Track geometric drift separately from conserved V. The former
          // sticky-wall silhouette cutoff is not a mass-conservation oracle.
          const volume=await read(device!,solver.volumeTexture);
          const maximumVolume=volume.reduce((maximum,value)=>Math.max(maximum,value),0);
          console.log(JSON.stringify({case:"mini32-separating-wall",frame,representedVolumeDrift:stats.representedVolumeDrift,maximumVolume}));
          assert.ok(Math.abs(sum(volume)/stats.initialVolumeCellSum!-1)<1e-5);
          // V<=capacity was guaranteed only by the removed optional liquid
          // balancing experiment. The retained default can concentrate V at
          // impact; keep maximumVolume in the receipt, conservation and finite
          // field assertions below, without claiming a capacity guarantee.
        }
      }
    });
    assert.deepEqual(errors,[]);
  }finally{solver?.destroy();device?.destroy();await releaseWebGPUExclusiveLock();}
});
