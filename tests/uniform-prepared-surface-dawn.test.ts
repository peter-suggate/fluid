import assert from "node:assert/strict";
import test from "node:test";
import { withUniformDevice } from "./helpers/uniform-geometric";
import { uniformPreparedSurfaceSamplingWGSL } from "../lib/methods/uniform/uniform-prepared-surface.wgsl";

(process.env.WEBGPU_NODE_MODULE?test:test.skip)("prepared surface gradients retain window arithmetic on random f32 fields",async()=>{
 await withUniformDevice("Prepared surface sample parity",async device=>{
  const field=device.createBuffer({size:4*(4+14**3),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
  const data=new Float32Array(4+14**3);let random=1234567;
  for(let i=4;i<data.length;i++){random=(Math.imul(random,1664525)+1013904223)>>>0;data[i]=(random/2**32-.5)*8;}
  device.queue.writeBuffer(field,0,data);
  const counts=device.createBuffer({size:16,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
  const read=device.createBuffer({size:16,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  try{
   const module=device.createShaderModule({code:/* wgsl */`
const UM_D=vec3u(13);const UM_WINDOW=14u;
@group(0) @binding(0) var<storage,read_write> deferred:UMDeferred;
struct UMDeferred {header:array<atomic<u32>,4>,data:array<u32>}
@group(0) @binding(1) var<storage,read_write> counts:array<atomic<u32>>;
var<workgroup> window:array<f32,2744>;
fn umCorner(k:u32,n:u32)->vec3u{return vec3u(k%n,(k/n)%n,k/(n*n));}
fn umVertexSum8(v:array<f32,8>)->f32{return ((v[0]+v[5])+(v[1]+v[4]))+((v[2]+v[7])+(v[3]+v[6]));}
fn umPreparedIndex(p:vec3u)->u32{return p.x+14u*(p.y+14u*p.z);}
${uniformPreparedSurfaceSamplingWGSL}
// Frozen pre-change sample and centered gradient, reading the staged window.
fn referenceSample(p:vec3f)->f32{
 let q=clamp(p,vec3f(0),vec3f(UM_D));let base=min(vec3u(floor(q)),UM_D-vec3u(1));let t=q-vec3f(base);
 let first=base.x+UM_WINDOW*(base.y+UM_WINDOW*base.z);var values:array<f32,8>;
 for(var k=0u;k<8u;k++){
  let corner=umCorner(k,2u);let w=select(vec3f(1)-t,t,corner!=vec3u(0));
  values[k]=window[first+corner.x+UM_WINDOW*(corner.y+UM_WINDOW*corner.z)]*w.x*w.y*w.z;
 }return umVertexSum8(values);
}
fn referenceGradient(p:vec3f)->vec3f{
 var g=vec3f(0);
 for(var axis=0u;axis<3u;axis++){
  var delta=vec3f(0);delta[axis]=0.25;
  let low=clamp(p-delta,vec3f(0),vec3f(UM_D));let high=clamp(p+delta,vec3f(0),vec3f(UM_D));
  g[axis]=(referenceSample(high)-referenceSample(low))/max(high[axis]-low[axis],1e-6);
 }return g;
}
@compute @workgroup_size(64) fn compare(@builtin(local_invocation_index) lane:u32,@builtin(global_invocation_id) id:vec3u){
 for(var i=lane;i<2744u;i+=64u){window[i]=bitcast<f32>(deferred.data[i]);}workgroupBarrier();
 let grid=umCorner(id.x,32u);let p=vec3f(grid)*13.0/31.0;
 let actual=umPreparedGradient(p);let expected=referenceGradient(p);
 if(any(bitcast<vec3u>(actual)!=bitcast<vec3u>(expected))){atomicAdd(&counts[0],1u);}
 atomicMax(&counts[1],bitcast<u32>(max(abs(actual.x-expected.x),max(abs(actual.y-expected.y),abs(actual.z-expected.z)))));
 atomicAdd(&counts[2],1u);
}`});
   assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==="error").map(m=>m.message),[]);
   const pipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"compare"}});
   const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:field}},{binding:1,resource:{buffer:counts}}]});
   const e=device.createCommandEncoder();const pass=e.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(512);pass.end();e.copyBufferToBuffer(counts,0,read,0,16);device.queue.submit([e.finish()]);await read.mapAsync(GPUMapMode.READ);
   const result=new Uint32Array(read.getMappedRange().slice(0));read.unmap();
   assert.deepEqual([...result.slice(0,3)],[0,0,32768]);
  }finally{read.destroy();counts.destroy();field.destroy();}
 });
});
