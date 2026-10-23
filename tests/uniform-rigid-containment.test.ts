import assert from "node:assert/strict";
import test from "node:test";
import { uniformRigidCoverageWGSL } from "../lib/methods/uniform/uniform-rigid-coverage.wgsl";
import { readMixedBuffer } from "./helpers/uniform-mixed-native-fields";
import { withUniformDevice } from "./helpers/uniform-geometric";

test("compound wall seams use union coverage without double-counting overlaps", { skip: !process.env.WEBGPU_NODE_MODULE }, async () => {
  await withUniformDevice("compound cell coverage", async device => {
    const output = device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    try {
      const module = device.createShaderModule({ code: `
struct Params { cellGravity:vec4f }
const umSolidParams=Params(vec4f(1));
fn umSolidWorldCell(p:vec3i)->vec3f{return vec3f(p)+vec3f(0.5);}
fn umBodyCount()->u32{return 2u;}
fn umBodyNear(i:u32,p:vec3f,margin:f32)->bool{return p.x<3.0;}
fn umInsideBody(i:u32,p:vec3f)->bool{
 if(p.x<1.0){return select((p.y<0.5),(p.y>=0.5),i==1u);}
 if(p.x<2.0){return p.y<0.5;}
 return select(p.y<0.5,p.z<0.5,i==1u);
}
${uniformRigidCoverageWGSL}
@group(0) @binding(0) var<storage,read_write> result:array<f32>;
@compute @workgroup_size(1) fn probe(@builtin(global_invocation_id) id:vec3u){result[id.x]=umCellBodyFraction(vec3i(i32(id.x),0,0));}
` });
      const info = await module.getCompilationInfo();
      assert.deepEqual(info.messages.filter(message => message.type === "error").map(message => message.message), []);
      const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "probe" } });
      const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: output } }] });
      const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
      pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(4); pass.end(); device.queue.submit([encoder.finish()]);
      assert.deepEqual(Array.from(await readMixedBuffer(device, output)), [1, .5, .75, 0]);
    } finally { output.destroy(); }
  });
});

test("surface extraction solid bindings match the renderer resource map", async () => {
  const { uniformSolidContourWGSL } = await import("../lib/methods/uniform/uniform-solid-contour.wgsl");
  assert.doesNotMatch(uniformSolidContourWGSL, /\bundefined\b/);
  const bindings = Array.from(uniformSolidContourWGSL.matchAll(/@binding\((\d+)\)/g), match => Number(match[1]));
  assert.deepEqual(bindings.sort((a,b) => a-b), [18,19,20,21,22]);
});

test("surface seed fitting cannot walk through a solid wall", { skip: !process.env.WEBGPU_NODE_MODULE }, async () => {
  const { narrowBandSeedShellWGSL } = await import("../lib/methods/uniform/uniform-narrow-band-seed.wgsl");
  await withUniformDevice("seed wall containment", async device => {
    const output = device.createBuffer({ size: 32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    try {
      const module = device.createShaderModule({ code: `
const NB_SEED_DEPTH=0.8;
fn particleDepth(p:vec3f)->f32{return -p.x;}
fn nbPointNearIn(home:vec3i,p:vec3f,inset:f32)->vec3f{return p;}
fn nbParticleBlocked(p:vec3f)->bool{return p.y<0.5&&p.x>0.3&&p.x<0.7;}
${narrowBandSeedShellWGSL}
@group(0) @binding(0) var<storage,read_write> result:array<vec4f>;
@compute @workgroup_size(1) fn probe(@builtin(global_invocation_id) id:vec3u){
 result[id.x]=seedShell(vec3f(.1,f32(id.x),0),-.1);
}` });
      const pipeline = await device.createComputePipelineAsync({ layout:"auto",compute:{module,entryPoint:"probe"} });
      const group = device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:output}}]});
      const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(2);pass.end();device.queue.submit([encoder.finish()]);
      const result=await readMixedBuffer(device,output);
      assert.ok(result[0]!<.3,"fitting retains the original wall side");
      assert.ok(Math.abs(result[4]!-.8)<1e-5,"unobstructed fitting still reaches the target depth");
    } finally { output.destroy(); }
  });
});

test("closed fluid domains reverse the obstacle side for geometry, contact and rendering", { skip: !process.env.WEBGPU_NODE_MODULE }, async () => {
  const { fluidRigidBoundaryWGSL } = await import("../lib/core/fluid-rigid-boundary.wgsl");
  const { sceneShapeWgsl, sceneShapeCode } = await import("../lib/core/scene-shape");
  await withUniformDevice("interior fluid boundary", async device => {
    const output=device.createBuffer({size:48,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
    try {
      const code=sceneShapeCode("box");
      const module=device.createShaderModule({code:`
${sceneShapeWgsl()}
${fluidRigidBoundaryWGSL}
@group(0) @binding(0) var<storage,read_write> result:array<vec4f>;
@compute @workgroup_size(1) fn probe(@builtin(global_invocation_id) id:vec3u){
 let p=array<vec3f,3>(vec3f(0),vec3f(1.1,0,0),vec3f(10,10,10))[id.x];
 result[id.x]=vec4f(f32(fluidRigidInside(${code},vec3f(2),p)),f32(fluidRigidInside(${-code-1},vec3f(2),p)),fluidRigidDistance(${code},vec3f(2),p),fluidRigidDistance(${-code-1},vec3f(2),p));
}`});
      const pipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"probe"}});
      const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:output}}]});
      const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(3);pass.end();device.queue.submit([encoder.finish()]);
      const values=await readMixedBuffer(device,output);
      assert.deepEqual(Array.from(values.slice(0,4)),[1,0,-1,1]);
      for(const offset of [4,8]) {assert.equal(values[offset],0);assert.equal(values[offset+1],1);assert.equal(values[offset+2],-values[offset+3]!);}
    } finally {output.destroy();}
  });
});

test("water isosurface extraction compiles with closed-domain solid helpers", { skip: !process.env.WEBGPU_NODE_MODULE }, async () => {
  const { surfaceExtractionShader } = await import("../lib/core/webgpu-water-pipeline");
  await withUniformDevice("water extraction shader", async device => {
    const module=device.createShaderModule({code:surfaceExtractionShader});
    const info=await module.getCompilationInfo();
    assert.deepEqual(info.messages.filter(message=>message.type==="error").map(message=>message.message),[]);
  });
});
