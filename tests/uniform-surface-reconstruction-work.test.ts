import assert from "node:assert/strict";
import test from "node:test";
import { narrowBandParticleSurfaceWGSL } from "../lib/methods/uniform/uniform-narrow-band-surface.wgsl";
import { withUniformDevice } from "./helpers/uniform-geometric";

test("reconstruction preserves buried/unsupported values without unnecessary solid queries", {skip:!process.env.WEBGPU_NODE_MODULE}, async()=>{
 await withUniformDevice("reconstruction support gate",async device=>{
  const bulk=narrowBandParticleSurfaceWGSL.slice(0,narrowBandParticleSurfaceWGSL.indexOf("// home: the tile q"));
  const code=/* wgsl */`
@group(0) @binding(0) var<storage,read_write> results:array<vec4f>;
@group(0) @binding(1) var<storage,read_write> bins:array<atomic<u32>>;
const NB_SURFACE_TILES=0u;
var<private> sample:u32;
var<private> queries:u32;
fn umVertexBuried(p:vec3i)->bool{queries++;return (sample&1u)!=0u;}
fn nbThetaFrom(home:vec3i,p:vec3f)->f32{return f32((sample/2u)%3u)*0.5;}
fn bandPhi(p:vec3f)->f32{return f32(sample)-4.125;}
fn bulkDepth(c:vec3u,p:vec3f)->f32{return f32(sample)-5.5;}
fn umCellNearFromPage(block:u32,page:vec3i,p:vec3i)->vec3i{return p;}
fn umTileAt(p:vec3u)->u32{return sample;}
${bulk}
fn reference(own:u32,origin:vec3i,q:vec3i)->vec3f{
 if(umVertexBuried(q)){return vec3f(bandPhi(vec3f(q)),0.0,0.0);}
 let theta=nbThetaFrom(origin>>vec3u(2u),vec3f(q));
 if(theta<=0.0){return vec3f(bandPhi(vec3f(q)),0.0,theta);}
 let c=vec3u(umCellNearFromPage(own>>9u,origin>>vec3u(5u),q));
 let value=bulkDepth(c,vec3f(q));let tile=umTileAt(c/4u);
 let gather=atomicLoad(&bins[tile])!=0u&&value>-2.0&&theta>0.0;
 return vec3f(value,select(0.0,1.0,gather),theta);
}
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3u){
 sample=id.x;if(sample>=24u){return;}
 let q=vec3i(i32(4u*sample),4,4);
 queries=0u;let before=reference(0u,vec3i(0),q);results[2u*sample]=vec4f(before,f32(queries));
 queries=0u;let after=particleSurfaceBulk(0u,vec3i(0),q);results[2u*sample+1u]=vec4f(after,f32(queries));
}`;
  const module=device.createShaderModule({code});
  assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==="error").map(m=>m.message),[]);
  const pipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"main"}});
  const output=device.createBuffer({size:24*2*16,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
  const bins=device.createBuffer({size:24*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
  const read=device.createBuffer({size:output.size,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
  device.queue.writeBuffer(bins,0,Uint32Array.from({length:24},(_,i)=>i<12?0:1));
  try{
   const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:output}},{binding:1,resource:{buffer:bins}}]});
   const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(1);pass.end();
   encoder.copyBufferToBuffer(output,0,read,0,read.size);device.queue.submit([encoder.finish()]);await read.mapAsync(GPUMapMode.READ);
   const mapped=read.getMappedRange(),words=new Uint32Array(mapped),values=new Float32Array(mapped);
   for(let i=0;i<24;i++){
    assert.deepEqual(words.slice(8*i,8*i+3),words.slice(8*i+4,8*i+7),`sample ${i}`);
    assert.equal(values[8*i+3],1);
    assert.equal(values[8*i+7],Math.floor(i/2)%3===0?0:1,`solid work at sample ${i}`);
   }
   read.unmap();
  }finally{output.destroy();bins.destroy();read.destroy();}
 });
});
