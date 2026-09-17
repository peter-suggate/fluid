import {readFile} from "node:fs/promises";
import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { surfaceRasterShader, surfaceWireframeShader } from "../lib/core/webgpu-water-pipeline";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";

const dawnTest=process.env.WEBGPU_NODE_MODULE?test:test.skip;
dawnTest("wireframe depth matches the shaded surface across a hydrostatic pool",async()=>{
  await acquireWebGPUExclusiveLock("dawn-test","water-wireframe-depth");
  let device:GPUDevice|undefined;
  try{
    const {create,globals}=await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href);
    Object.assign(globalThis,globals);const gpu=create(["backend=metal"]);
    const adapter=await gpu.requestAdapter();assert.ok(adapter);device=await adapter.requestDevice();
    assert.ok(device);
    const width=512,height=512;
    const uniform=device.createBuffer({size:112,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
    const data=new Float32Array(28);data.set([width,height,0,0]);data.set([3.2,6,13,0],4);data.set([3.2,1.6,3.2,0],8);data.set([6.4,4.8,6.4,1.6],12);data.set([128,96,128,1],20);device.queue.writeBuffer(uniform,0,data);
    const points:number[]=[];
    const vertex=(x:number,z:number)=>points.push(x,1.6,z,1,0,1,0,0);
    for(let z=0;z<32;z++)for(let x=0;x<32;x++){
      const a=x*.2,b=z*.2;
      vertex(a,b);vertex(a,b+.2);vertex(a+.2,b+.2);
      vertex(a,b);vertex(a+.2,b+.2);vertex(a+.2,b);
    }
    if(process.env.FLUID_POOL_MESH){const bytes=await readFile(process.env.FLUID_POOL_MESH);const mesh=new Float32Array(bytes.buffer,bytes.byteOffset,bytes.byteLength/4);points.length=0;for(let i=0;i<mesh.length;i+=24)if([1,9,17].every(k=>Math.abs(mesh[i+k]!-1.6)<1e-5))points.push(...mesh.subarray(i,i+24));}
    const vertices=device.createBuffer({size:points.length*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});device.queue.writeBuffer(vertices,0,new Float32Array(points));
    const texture=(format:GPUTextureFormat)=>device!.createTexture({size:[width,height],format,usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_SRC});
    const front=texture("rgba32float"),normal=texture("rgba16float"),depth=texture("depth24plus"),dry=texture("rgba32float"),output=texture("rgba8unorm");
    const layout0=device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.VERTEX|GPUShaderStage.FRAGMENT,buffer:{type:"uniform"}},{binding:1,visibility:GPUShaderStage.VERTEX,buffer:{type:"read-only-storage"}}]});
    const layout1=device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.FRAGMENT,texture:{sampleType:"unfilterable-float"}}]});
    const layout=device.createPipelineLayout({bindGroupLayouts:[layout0,layout1]});
    const group=device.createBindGroup({layout:layout0,entries:[{binding:0,resource:{buffer:uniform}},{binding:1,resource:{buffer:vertices}}]});
    const background=device.createBindGroup({layout:layout1,entries:[{binding:0,resource:dry.createView()}]});
    const surface=device.createShaderModule({code:surfaceRasterShader}),wire=device.createShaderModule({code:surfaceWireframeShader});
    const prepass=device.createRenderPipeline({layout,vertex:{module:surface,entryPoint:"surfaceVertex"},fragment:{module:surface,entryPoint:"surfaceFragment",targets:[{format:"rgba32float"},{format:"rgba16float"}]},primitive:{cullMode:"back"},depthStencil:{format:"depth24plus",depthWriteEnabled:true,depthCompare:"less"}});
    const counts:number[]=[];
    for(const compare of ["equal","always"] as const){
      const pipeline=device.createRenderPipeline({layout,vertex:{module:wire,entryPoint:"wireVertex"},fragment:{module:wire,entryPoint:"wireFragment",targets:[{format:"rgba8unorm"}]},primitive:{cullMode:"back"},depthStencil:{format:"depth24plus",depthWriteEnabled:false,depthCompare:compare}});
      const encoder=device.createCommandEncoder();
      const pass=encoder.beginRenderPass({colorAttachments:[front,normal].map(t=>({view:t.createView(),loadOp:"clear" as const,storeOp:"store" as const})),depthStencilAttachment:{view:depth.createView(),depthClearValue:1,depthLoadOp:"clear",depthStoreOp:"store"}});
      pass.setPipeline(prepass);pass.setBindGroup(0,group);pass.setBindGroup(1,background);pass.draw(points.length/8);pass.end();
      const lines=encoder.beginRenderPass({colorAttachments:[{view:output.createView(),loadOp:"clear",storeOp:"store"}],depthStencilAttachment:{view:depth.createView(),depthLoadOp:"load",depthStoreOp:"store"}});
      lines.setPipeline(pipeline);lines.setBindGroup(0,group);lines.setBindGroup(1,background);lines.draw(points.length/8);lines.end();
      const result=device.createBuffer({size:width*height*4,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});encoder.copyTextureToBuffer({texture:output},{buffer:result,bytesPerRow:width*4},[width,height]);device.queue.submit([encoder.finish()]);await result.mapAsync(GPUMapMode.READ);
      const pixels=new Uint8Array(result.getMappedRange());let count=0;for(let i=3;i<pixels.length;i+=4)if(pixels[i])count++;counts.push(count);result.unmap();result.destroy();
    }
    console.log({depthTested:counts[0],reference:counts[1]});assert.ok(counts[1]!>1000);assert.equal(counts[0],counts[1],"depth testing must not erase visible pool edges");
  }finally{device?.destroy();await releaseWebGPUExclusiveLock();}
});
