import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { createProcessRetainedDawnGPU, type NodeDawnProvider } from "../lib/harness/node-dawn-provider";
import { managedGPUDevice } from "../lib/core/gpu-compilation-manager";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { uniformGeometricSolverOptions } from "../lib/methods/uniform/uniform-geometric-options";
import { createUniformReferenceComputeShader } from "../lib/methods/uniform/webgpu-uniform-reference.wgsl";
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
function write(device:GPUDevice,texture:GPUTexture,values:Float32Array){const components=texture.format==="rgba32float"?4:1;device.queue.writeTexture({texture},values as Float32Array<ArrayBuffer>,{bytesPerRow:texture.width*components*4,rowsPerImage:texture.height},[texture.width,texture.height,texture.depthOrArrayLayers]);}

const modulePath=process.env.WEBGPU_NODE_MODULE;
(modulePath?test:test.skip)("3D surface-deficit source balance",{timeout:180000},async t=>{
 await acquireWebGPUExclusiveLock("dawn-test","3D surface-deficit source balance");
 let device:GPUDevice|undefined;let solver:WebGPUUniformReferenceSolver|undefined;
 try{
  const dawn=await import(pathToFileURL(modulePath!).href);Object.assign(globalThis,dawn.globals);
  const gpu=createProcessRetainedDawnGPU(dawn as NodeDawnProvider,["backend=metal"]);
  const adapter=await gpu.requestAdapter();assert.ok(adapter);
  device=managedGPUDevice(await adapter.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter.limits)}),{requireWorkerRealm:false});
  const errors:string[]=[];device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);});
  assert.equal(uniformGeometricSolverOptions().surfaceDeficitBalancing,true);
  const scene=structuredClone(sceneDocument(getSceneDefinition("minimal-power-dam-break-32")));
  scene.voxelDomain.finestCellSize_m=scene.container.width_m/16;
  scene.rigidBodies=[];
  solver=await WebGPUUniformReferenceSolver.createAsync(device,scene,"balanced",undefined,{geometricVolume:true},()=>{});
  const a=solver as unknown as {surfaceDeficitBalancing:boolean;conditioningScratch:GPUBuffer;gammaA:GPUTexture;volumeA:GPUTexture;volumeB:GPUTexture;
    mainPipelineLayout:GPUPipelineLayout;sharpenComputeGroup:GPUBindGroup;
    writeParams(dt:number,bodies:number,inflow:number):void;encodeSurfaceDeficitBalance(e:GPUCommandEncoder):void;};
  assert.equal(a.surfaceDeficitBalancing,true);
  const {nx,ny,nz,cellCount:n}=solver.info;const dt=1/30;
  a.writeParams(dt,0,0);
  const module=device.createShaderModule({code:createUniformReferenceComputeShader(true)+`
    @compute @workgroup_size(4,4,4) fn auditBalance(@builtin(global_invocation_id)gid:vec3u){
      let id=vec3i(gid);if(valid(id)){textureStore(volumeOut,id,vec4f(volumeCorrectionDivergence(id)));}}
  `});
  assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==="error"),[]);
  const pipeline=await device.createComputePipelineAsync({layout:a.mainPipelineLayout,compute:{module,entryPoint:"auditBalance"}});
  const phi=new Float32Array((nx+1)*(ny+1)*(nz+1)).fill(-1);
  write(device,solver.vertexPhiTexture!,phi);
  const source=2+nx*(2+ny*2),sink=3+nx*(2+ny*2);
  const run=async(v:Float32Array,target:Float32Array)=>{
    write(device!,a.volumeB,v);write(device!,a.gammaA,target);
    const e=device!.createCommandEncoder();a.encodeSurfaceDeficitBalance(e);
    const pass=e.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,a.sharpenComputeGroup);
    pass.dispatchWorkgroups(Math.ceil(nx/4),Math.ceil(ny/4),Math.ceil(nz/4));pass.end();device!.queue.submit([e.finish()]);
    return read(device!,a.volumeA);
  };
  const close=(x:number,y:number)=>assert.ok(Math.abs(x-y)<1e-4,`${x} vs ${y}`);
  await t.test("overfill preserved; contraction cancels the source",async()=>{
    const v=new Float32Array(n).fill(1);v[source]=2;v[sink]=0;
    const result=await run(v,new Float32Array(n).fill(1));
    close(result[source]!,15);close(result[sink]!,-15);close(result.reduce((s,x)=>s+x,0),0);
    assert.deepEqual(await read(device!,a.volumeB),v,"conservative volume is unchanged");
  });
  await t.test("surface target, not empty capacity, determines the deficit",async()=>{
    const v=new Float32Array(n).fill(1);v[source]=2;v[sink]=0.2;
    const target=new Float32Array(n).fill(1);target[sink]=0.3;
    const result=await run(v,target);close(result[source]!,15);close(result[sink]!,-3);
  });
  await t.test("no overfill or no deficit gives zero contraction",async()=>{
    const v=new Float32Array(n).fill(1);v[sink]=0;
    close((await run(v,new Float32Array(n).fill(1)))[sink]!,0);
    v[sink]=1;v[source]=2;
    const result=await run(v,new Float32Array(n).fill(1));close(result[source]!,15);
    assert.ok(result.every(x=>x>=0));
  });
  await t.test("runtime off clears the previously computed source",async()=>{
    const v=new Float32Array(n).fill(1);v[source]=2;v[sink]=0;
    await run(v,new Float32Array(n).fill(1));
    solver!.applyRuntimeValues({surfaceDeficitBalancing:"off"});
    const result=await run(v,new Float32Array(n).fill(1));close(result[source]!,15);close(result[sink]!,0);
    solver!.applyRuntimeValues({surfaceDeficitBalancing:"on"});
    close((await run(v,new Float32Array(n).fill(1)))[sink]!,-15);
  });
  assert.deepEqual(errors,[]);
 }finally{solver?.destroy();device?.destroy();await releaseWebGPUExclusiveLock();}
});
