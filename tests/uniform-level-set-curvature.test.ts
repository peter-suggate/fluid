import {test} from "node:test";
import assert from "node:assert/strict";
import {UNIFORM_LEVEL_SET_CURVATURE_WGSL} from "../lib/methods/uniform/uniform-level-set-curvature.wgsl";
import {withUniformDevice} from "./helpers/uniform-geometric";

// The census's moving-curve term reads the surface's curvature, not phi's
// laplacian: a sphere's distance field, the same field scaled, and a field
// with the same zero set that is no distance (r^2 - R^2) give one curvature,
// 2/r, and so one 4h error. Its laplacian differs by the scale and by phi_nn.
const N=17,R=5.3,h=0.01;
const fields:Record<string,(r:number)=>number>={
 distance:r=>(r-R)*h,
 scaled:r=>3*(r-R)*h,
 squared:r=>(r*r-R*R)*h,
};
test("level-set curvature is the surface's, whatever phi's magnitude",{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 await withUniformDevice("level-set curvature",async device=>{
  const centre=(N-1)/2+0.37;
  // Vertices within a cell of the sphere, off the stencil's edge.
  const probes:number[]=[];
  for(let z=1;z<N-1;z++)for(let y=1;y<N-1;y++)for(let x=1;x<N-1;x++){const r=Math.hypot(x-centre,y-centre,z-centre);if(Math.abs(r-R)<=1)probes.push(x,y,z,0);}
  const count=probes.length/4;
  const code=/* wgsl */`
const H=vec3f(${h},${h},${h});
@group(0) @binding(0) var<storage,read> field:array<f32>;
@group(0) @binding(1) var<storage,read> probes:array<vec4u>;
@group(0) @binding(2) var<storage,read_write> out:array<vec2f>;
fn umLoadFine(p:vec3u)->f32{return field[p.x+${N}u*(p.y+${N}u*p.z)];}
${UNIFORM_LEVEL_SET_CURVATURE_WGSL}
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3u){
 if(id.x>=${count}u){return;}let p=probes[id.x].xyz;out[id.x]=umLevelSetCurvature(p,umLoadFine(p));
}`;
  const pipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module:device.createShaderModule({code}),entryPoint:"main"}});
  const probeBuffer=device.createBuffer({size:probes.length*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
  device.queue.writeBuffer(probeBuffer,0,new Uint32Array(probes));
  const results:Record<string,Float32Array>={};
  for(const [name,phi] of Object.entries(fields)){
   const values=new Float32Array(N*N*N);
   for(let z=0;z<N;z++)for(let y=0;y<N;y++)for(let x=0;x<N;x++)values[x+N*(y+N*z)]=phi(Math.hypot(x-centre,y-centre,z-centre));
   const fieldBuffer=device.createBuffer({size:values.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});device.queue.writeBuffer(fieldBuffer,0,values);
   const out=device.createBuffer({size:8*count,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
   const read=device.createBuffer({size:8*count,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
   const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:fieldBuffer}},{binding:1,resource:{buffer:probeBuffer}},{binding:2,resource:{buffer:out}}]});
   const encoder=device.createCommandEncoder();const pass=encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(Math.ceil(count/64));pass.end();
   encoder.copyBufferToBuffer(out,0,read,0,8*count);device.queue.submit([encoder.finish()]);
   await read.mapAsync(GPUMapMode.READ);results[name]=new Float32Array(read.getMappedRange().slice(0));read.unmap();
   fieldBuffer.destroy();out.destroy();read.destroy();
  }
  for(let i=0;i<count;i++){
   const r=Math.hypot(probes[4*i]!-centre,probes[4*i+1]!-centre,probes[4*i+2]!-centre),exact=2/(r*h);
   for(const name of Object.keys(fields)){
    const [kappa,valid]=[results[name]![2*i]!,results[name]![2*i+1]!];
    assert.equal(valid,1,`${name} has a normal at r=${r}`);
    // Second-order differences over a 5.3-cell radius: a few percent.
    assert.ok(Math.abs(kappa-exact)<=0.04*exact,`${name}: curvature ${kappa} against 2/r ${exact} at r=${r.toFixed(2)}`);
   }
   // The scaled and squared fields share the distance field's curvature closely, not just the exact one.
   assert.ok(Math.abs(results.scaled![2*i]!-results.distance![2*i]!)<=1e-3*exact);
  }
  probeBuffer.destroy();
 });
});
// A plateau (phi constant) has no normal and reports none, so it adds no curve sample.
test("a plateau reports no curvature sample",{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 await withUniformDevice("level-set plateau",async device=>{
  const code=/* wgsl */`
const H=vec3f(0.01,0.01,0.01);
@group(0) @binding(0) var<storage,read_write> out:vec2f;
fn umLoadFine(p:vec3u)->f32{return 0.02;}
${UNIFORM_LEVEL_SET_CURVATURE_WGSL}
@compute @workgroup_size(1) fn main(){out=umLevelSetCurvature(vec3u(2u),0.02);}`;
  const pipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module:device.createShaderModule({code}),entryPoint:"main"}});
  const out=device.createBuffer({size:8,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC}),read=device.createBuffer({size:8,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
  const encoder=device.createCommandEncoder();const pass=encoder.beginComputePass();pass.setPipeline(pipeline);
  pass.setBindGroup(0,device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:out}}]}));pass.dispatchWorkgroups(1);pass.end();
  encoder.copyBufferToBuffer(out,0,read,0,8);device.queue.submit([encoder.finish()]);await read.mapAsync(GPUMapMode.READ);
  assert.deepEqual([...new Float32Array(read.getMappedRange())],[0,0]);read.unmap();out.destroy();read.destroy();
 });
});
