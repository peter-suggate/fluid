import assert from "node:assert/strict";
import test from "node:test";
import { verifiedWaterSolidContact, waterSolidContactWGSL } from "../lib/core/water-solid-contact";
const cases:[number,number,boolean,boolean][]=[
  [0,.25,false,true],[.30,.25,false,true],[-.30,.25,false,true],
  [.32,.25,false,false],[-.32,.25,false,false],
  // Actual shifted-exit crack candidates from the .15s breaking-wave capture.
  [-7.083,.25,false,false],[-4.415,.25,false,false],[-1.270,.25,false,false],
  // A true embedded solid intersected by the refracted ray is still opaque.
  [-7.083,.25,true,true],[-1.270,.25,true,true],[1e20,.25,false,false],
];
test("water contact requires bounded geometry proximity or a verified ray hit",()=>{for(const [gap,cell,ray,expected]of cases)assert.equal(verifiedWaterSolidContact(gap,cell,ray),expected);});
test("GPU contact validation rejects distant foreground and preserves embedded solid hits",{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
  const {withUniformDevice}=await import("./helpers/uniform-geometric");
  await withUniformDevice("water solid exit classification",async device=>{
    const module=device.createShaderModule({code:`${waterSolidContactWGSL}
      @group(0) @binding(0) var<storage,read> inputs:array<vec4f>;
      @group(0) @binding(1) var<storage,read_write> results:array<u32>;
      @compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id:vec3u){let v=inputs[id.x];results[id.x]=select(0u,1u,verifiedWaterSolidContact(v.x,v.y,v.z>.5));}`});
    const pipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"main"}});
    const input=device.createBuffer({size:cases.length*16,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),out=device.createBuffer({size:cases.length*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC}),read=device.createBuffer({size:cases.length*4,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});device.queue.writeBuffer(input,0,new Float32Array(cases.flatMap(([gap,cell,ray])=>[gap,cell,+ray,0])));
    try{const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:input}},{binding:1,resource:{buffer:out}}]}));pass.dispatchWorkgroups(cases.length);pass.end();encoder.copyBufferToBuffer(out,0,read,0,cases.length*4);device.queue.submit([encoder.finish()]);await read.mapAsync(GPUMapMode.READ);assert.deepEqual([...new Uint32Array(read.getMappedRange())],cases.map(c=>+c[3]));read.unmap();}finally{input.destroy();out.destroy();read.destroy();}
  });
});
