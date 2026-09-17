import assert from "node:assert/strict";
import test from "node:test";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { createDawnRenderDevice } from "../tools/svo-dry-frame-harness";
import { renderTerrainProxyWGSL, sparseSceneProxyVoxelizationShaderFor } from "../lib/core/webgpu-sparse-scene-proxies";
import { svoCellContourFitWGSL } from "../lib/svo/features/construction/svo-cell-contour-fit";
import { SVO_GBUFFER_NORMAL_OCT8_WGSL } from "../lib/svo/contracts/svo-gbuffer";

(process.env.WEBGPU_NODE_MODULE?test:test.skip)("terrain contour source shares continuous heights, normals and conservative crossing coverage",async()=>{
 await acquireWebGPUExclusiveLock("dawn-test","terrain-contour");let device:GPUDevice|undefined;
 try{
  device=(await createDawnRenderDevice()).device;
  const terrain={baseWords:0,heightsBaseWords:16,width:8,depth:8,patchBaseWords:80,patchCapacity:1};
  for(const mode of ["dense","occupancy","banded"] as const){
   const module: GPUShaderModule=device.createShaderModule({code:sparseSceneProxyVoxelizationShaderFor("dry","f16-unorm8",mode,undefined,terrain,true)});
   assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==="error"),[]);
   await device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"rebuildDirtyBrickPayload"}});
  }
  const code=`
  @group(0) @binding(0) var<storage,read_write> maintenance:array<atomic<u32>>;
  @group(0) @binding(1) var<storage,read_write> result:array<vec4f>;
  struct SolidWorldSample{fraction:f32,distance:f32,material:u32,normal:vec3f}
  ${renderTerrainProxyWGSL(terrain)}
  ${SVO_GBUFFER_NORMAL_OCT8_WGSL}
  struct ScenePrimitive{centerType:vec4f,extentIdentity:vec4f,rotation:vec4f}
  const primitives=array<ScenePrimitive,1>(ScenePrimitive(vec4f(0),vec4f(1),vec4f(0)));
  fn candidateOffset()->u32{return 0u;}fn candidatesPerBrick()->u32{return 1u;}
  fn scenePrimitiveType(p:ScenePrimitive)->u32{return 0u;}
  fn inverseRotate(p:vec3f,r:vec4f)->vec3f{return p;}
  fn primitiveDistance(p:ScenePrimitive,w:vec3f)->f32{return 1e20;}
  fn primitiveUsesThresholdOccupancy(p:ScenePrimitive)->bool{return false;}
  ${svoCellContourFitWGSL(true,false)}
  @compute @workgroup_size(1) fn check(@builtin(global_invocation_id) id:vec3u){
   let x=1.6+f32(id.x)*.1;let z=3.2;
   let surface=rtSurface(vec2f(x,z));
   // At .6 cells above the centre height, a slope of 2 still crosses the cell.
   let p=vec3f(x,surface.x+.6,z);let sample=sampleRenderTerrain(p,vec3f(1));
   let span=rtSurfaceRange(p.xz-vec2f(.5),p.xz+vec2f(.5));
   let contour=fitSceneContour(p,vec3f(1),sample.normal,sample.fraction,0u,0u);
   result[id.x*3u]=vec4f(surface,sample.fraction);
   result[id.x*3u+1u]=vec4f(sample.normal,sample.distance);
   result[id.x*3u+2u]=vec4f(span,f32(contour),0);
  }`;
  const module=device.createShaderModule({code});const pipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"check"}});
  const storage=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST;
  const input=device.createBuffer({size:88*4,usage:storage}),output=device.createBuffer({size:12*12*4,usage:storage});
  const read=device.createBuffer({size:12*12*4,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
  const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[input,output].map((buffer,binding)=>({binding,resource:{buffer}}))});
  for(const [sx,sz,cross] of [[0,0,0],[.3,.2,0],[2,0,0],[.2,.3,.1]]){
   const height=(x:number,z:number)=>4+sx*x+sz*z+cross*x*z;
   const data=new Float32Array(88);data.set([0,0,1,1]);
   for(let z=0;z<8;z++)for(let x=0;x<8;x++)data[16+x+8*z]=height(x+.5,z+.5);
   let prior:Float32Array|undefined;
   for(const distantPatch of [false,true]){
    new Uint32Array(data.buffer)[7]=Number(distantPatch);data.set([50,50,50,0,51,51,51,0],80);
    device.queue.writeBuffer(input,0,data);const encoder=device.createCommandEncoder();const pass=encoder.beginComputePass();
    pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(12);pass.end();encoder.copyBufferToBuffer(output,0,read,0,12*12*4);device.queue.submit([encoder.finish()]);
    await read.mapAsync(GPUMapMode.READ);const values=new Float32Array(read.getMappedRange().slice(0));read.unmap();
    if(prior)assert.deepEqual(values,prior,"an unrelated edit does not disable terrain contours");prior=values;
    for(let i=0;i<12;i++){
     const x=1.6+i*.1,z=3.2,a=i*12;
     assert.ok(Math.abs(values[a]-height(x,z))<1e-5,"height does not step at column boundaries");
     assert.ok(Math.abs(values[a+1]-(sx+cross*z))<1e-5);
     assert.ok(Math.abs(values[a+2]-(sz+cross*x))<1e-5,"normal is the derivative of the same field");
     if(sx===2)assert.ok(values[a+3]>0&&values[a+3]<1,"steep surface-crossing cells are partial");
     if(sx===0&&sz===0)assert.equal(values[a+3],0,"flat terrain above the surface is empty");
     for(let j=0;j<=8;j++)for(let k=0;k<=8;k++){
      const h=height(x-.5+j/8,z-.5+k/8);
      assert.ok(h>=values[a+8]-1e-5&&h<=values[a+9]+1e-5,"range contains the continuous surface across knots");
     }
    }
   }
  }
  input.destroy();output.destroy();read.destroy();
 }finally{device?.destroy();await releaseWebGPUExclusiveLock();}
});
