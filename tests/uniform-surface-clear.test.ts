import assert from "node:assert/strict";
import test from "node:test";
import { narrowBandSplatWGSL } from "../lib/methods/uniform/uniform-narrow-band-splat.wgsl";
import { withUniformDevice } from "./helpers/uniform-geometric";

test("surface scratch clears mapped page halos and recycled pages at any launch width", {skip:!process.env.WEBGPU_NODE_MODULE}, async()=>{
 await withUniformDevice("mapped surface scratch",async device=>{
  const blocks=7,vertices=35937,base=16,hot=base+vertices*blocks+16,words=hot+vertices*blocks+16;
  const clear=narrowBandSplatWGSL.slice(0,narrowBandSplatWGSL.indexOf("// Before classify"));
  const code=/* wgsl */`
const UM_TILES=${blocks*512}u;const NB_DEPTH_A=${base}u;const NB_SURFACE_HOT_NEAREST=${hot}u;
@group(0) @binding(0) var<storage,read_write> bins:array<atomic<u32>>;
@group(0) @binding(1) var<storage,read> live:array<u32>;
fn umTileLive(tile:u32)->bool{return live[tile/512u]!=0u;}
${clear}`;
  const module=device.createShaderModule({code});
  assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==="error").map(m=>m.message),[]);
  const pipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"surfaceClear"}});
  const bins=device.createBuffer({size:4*words,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
  const live=device.createBuffer({size:4*blocks,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
  const read=device.createBuffer({size:bins.size,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
  const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:bins}},{binding:1,resource:{buffer:live}}]});
  try{
   for(const groups of [1,7,1024]){
    for(const active of [[1,4],[2,5],[],[1,2,3,4,5]]){
     // Poison both banks each time: a page returning after retirement must
     // clear the former metric and splat payload, including every high face.
     const expected=new Uint32Array(words).fill(0xdeadbeef),mask=new Uint32Array(blocks);
     for(const b of active){mask[b]=1;}
     device.queue.writeBuffer(bins,0,expected);device.queue.writeBuffer(live,0,mask);
     for(const b of active)for(const offset of [base,hot])expected.fill(0,offset+b*vertices,offset+(b+1)*vertices);
     const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(groups);pass.end();
     encoder.copyBufferToBuffer(bins,0,read,0,read.size);device.queue.submit([encoder.finish()]);await read.mapAsync(GPUMapMode.READ);
     assert.deepEqual(new Uint32Array(read.getMappedRange()),expected,`${groups} groups, pages ${active}`);read.unmap();
    }
   }
  }finally{bins.destroy();live.destroy();read.destroy();}
 });
});
