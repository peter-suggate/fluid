import assert from "node:assert/strict";
import test from "node:test";
import { withUniformDevice } from "./helpers/uniform-geometric";
import { levelSetFillWGSL } from "../lib/core/level-set-fill.wgsl";

test("tetrahedral fill remains finite when the zero face meets roundoff-sized negative corners", {skip:!process.env.WEBGPU_NODE_MODULE}, async()=>{
  await withUniformDevice("level set boundary volume", async device=>{
    const module=device.createShaderModule({code:`${levelSetFillWGSL}
      @group(0) @binding(0) var<storage,read_write> result:array<f32>;
      @group(0) @binding(1) var<storage,read> input:array<f32>;
      @compute @workgroup_size(1) fn main(){
        var v:array<f32,8>;for(var i=0u;i<8u;i++){v[i]=input[i];}result[0]=fill(v);
        result[1]=fill(array<f32,8>(0,0,0,0,-.1,-.1,-.25,-.25));
        result[2]=fill(array<f32,8>(0,0,0,0,.1,1.77635684e-15,.25,1.77635684e-15));
      }`});
    const pipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"main"}});
    const input=device.createBuffer({size:32,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
    device.queue.writeBuffer(input,0,new Float32Array([0,0,0,0,-.1,-1.77635684e-15,-.25,-1.77635684e-15]));
    const output=device.createBuffer({size:12,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
    const read=device.createBuffer({size:12,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
    try{const e=device.createCommandEncoder(),p=e.beginComputePass();p.setPipeline(pipeline);p.setBindGroup(0,device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:output}},{binding:1,resource:{buffer:input}}]}));p.dispatchWorkgroups(1);p.end();e.copyBufferToBuffer(output,0,read,0,12);device.queue.submit([e.finish()]);await read.mapAsync(GPUMapMode.READ);const data=[...new Float32Array(read.getMappedRange())];read.unmap();assert.deepEqual(data,[1,1,0]);}finally{input.destroy();output.destroy();read.destroy();}
  });
});
