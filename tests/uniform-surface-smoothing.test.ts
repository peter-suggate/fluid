import assert from "node:assert/strict";
import test from "node:test";
import { uniformSurfaceSmoothingWGSL } from "../lib/methods/uniform/uniform-surface-smoothing.wgsl";
import { withUniformDevice, createUniformSolver, advanceUniform } from "./helpers/uniform-geometric";
import { readMixedBuffer } from "./helpers/uniform-mixed-native-fields";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";

const gpu = { skip: !process.env.WEBGPU_NODE_MODULE };
test("surface smoothing requires an interface without suppressing supported smoothing", gpu, async () => {
 await withUniformDevice("interface surface smoothing", async device => {
  // Captured at Letters frame 94, vertex (156,96,128): the unbounded
  // correction is -0.0014946, creating liquid in an entirely positive stencil.
  const c=0.0005245805368758738,h=9.81/(0.8*24**2);
  const air=[0.006326258182525635,0.005882143974304199,c,c,c,c,0.02205956168472767,0.20298029482364655,c,c];
  const cases=[
   {name:"air at the back wall",c,t:air,h,width:4,eps:1/120,walls:4,closed:0,expected:0},
   {name:"all-liquid mirror",c:-c,t:air.map(v=>-v),h,width:4,eps:1/120,walls:4,closed:0,expected:0},
   // This correction would stay inside the old stencil's bounds, but
   // cross zero after advection returns a smaller positive value.
   {name:"air correction cannot cross the advected value",c:0.1,t:[0.2,0.2,0.1,0.1,0.1,0.1,1,1,0.1,0.1],h:1,width:1,eps:1/32,walls:4,closed:0,expected:0},
   {name:"alternating surface noise",c:1,t:[-1,-1,1,1,1,1,1,1,1,1],h:2,width:1,eps:1/32,walls:4,closed:0,expected:-0.5},
   {name:"linear distance",c:0.2,t:[-0.8,1.2,0.2,0.2,0.2,0.2,-1.8,2.2,0.2,0.2],h:1,width:1,eps:1/32,walls:4,closed:0,expected:0},
   {name:"wall excludes the filter",c:1,t:[-1,-1,1,1,1,1,1,1,1,1],h:2,width:1,eps:1/32,walls:5,closed:0,expected:0},
   {name:"buried tap excludes the filter",c:1,t:[-1,-1,1,1,1,1,1,1,1,1],h:2,width:1,eps:1/32,walls:4,closed:2,expected:0},
   {name:"far marker excludes the filter",c:0.1,t:[0.1,0.1,0.1,0.1,0.1,0.1,0.1,16,0.1,0.1],h:1,width:1,eps:1/32,walls:4,closed:0,expected:0},
   // Rejected z taps cannot make the all-air x stencil eligible.
   {name:"invalid axis cannot admit a new crossing",c,t:air.map((v,i)=>i===4?-10:v),h,width:4,eps:1/120,walls:0,closed:0,expected:0},
   {name:"both axes remain air",c:0.01,t:[0.01,0.01,0.01,0.01,0.01,0.01,1,1,1,1],h:1,width:1,eps:1/32,walls:0,closed:0,expected:0},
  ];
  const data=new Float32Array(16*cases.length);
  cases.forEach((s,i)=>data.set([s.c,s.eps,s.h,s.width,...s.t,s.closed,s.walls],16*i));
  const fixture=device.createBuffer({size:data.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
  const output=device.createBuffer({size:4*cases.length,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
  device.queue.writeBuffer(fixture,0,data);
  const module=device.createShaderModule({code:`
@group(0) @binding(0) var<storage,read> fixture:array<f32>;
@group(0) @binding(1) var<storage,read_write> output:array<f32>;
var<private> walls:u32;
fn umWallWithin(vertex:vec3i,axis:u32,reach:i32)->bool{return (walls&(1u<<axis))!=0u;}
${uniformSurfaceSmoothingWGSL}
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id:vec3u){
 let a=16u*id.x;var taps:array<f32,10>;for(var k=0u;k<10u;k++){taps[k]=fixture[a+4u+k];}
 walls=u32(fixture[a+15u]);
 output[id.x]=umSurfaceSmoothing(vec3i(0),i32(fixture[a+3u]),fixture[a],taps,u32(fixture[a+14u]),vec3f(fixture[a+2u]),fixture[a+1u]);
}`});
  assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==="error"),[]);
  const pipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"main"}});
  const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[fixture,output].map((buffer,binding)=>({binding,resource:{buffer}}))});
  try {
   const e=device.createCommandEncoder(),p=e.beginComputePass();p.setPipeline(pipeline);p.setBindGroup(0,group);p.dispatchWorkgroups(cases.length);p.end();device.queue.submit([e.finish()]);
   const result=await readMixedBuffer(device,output);
   cases.forEach((sample,i)=>assert.ok(Math.abs(result[i]!-sample.expected)<1e-7,`${sample.name}: ${result[i]} vs ${sample.expected}`));
  } finally {fixture.destroy();output.destroy();}
 });
});

test("Letters at 60 Hz completes its take without a residency failure",gpu,async()=>{
 await withUniformDevice("Letters residency regression",async device=>{
  const scene=structuredClone(sceneDocument(getSceneDefinition("nbflip-figure-8-letters")));
  scene.numerics={...scene.numerics,fixedDt_s:1/60,maxDt_s:1/60};
  const solver=await createUniformSolver(device,scene);
  try {for(let frame=1;frame<=482;frame++)await advanceUniform(solver,frame/60);}
  finally {solver.destroy();}
 },["subgroups"]);
});
