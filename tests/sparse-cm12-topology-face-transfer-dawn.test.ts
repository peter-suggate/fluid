import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";

const source = readFileSync(new URL("../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.wgsl.ts", import.meta.url), "utf8");
const production = (name: string) => {
  const code = source.match(new RegExp(`fn ${name}\\([\\s\\S]*?\\n}`))?.[0];
  assert.ok(code, name); return code;
};
const live = new Set<GPU>();
(process.env.WEBGPU_NODE_MODULE ? test : test.skip)("topology face prolongation preserves coarse flux and normal-linear divergence", { timeout: 120_000 }, async () => {
  await acquireWebGPUExclusiveLock("dawn-test", "topology-face-transfer");
  let device: GPUDevice | undefined;
  try {
    const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href);
    Object.assign(globalThis, dawn.globals);
    const gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]); live.add(gpu);
    const adapter = await gpu.requestAdapter(); assert.ok(adapter);
    device = await adapter.requestDevice();
    const code = `
const INVALID=0xffffffffu;
struct Params{dimensions:vec4u}
const p=Params(vec4u(16));
@group(0)@binding(0)var<storage,read_write>state:array<f32>;
var<private>samplePoint:vec3f;
var<private>targetAxis:u32;
var<private>targetWidth:f32;
const TARGET=100000u;
fn ownerCellAt(q:vec3i)->u32{let c=vec3u(q)/2u;return c.x+8u*(c.y+8u*c.z);}
fn cellCenter(cell:u32)->vec3f{return 2.0*(vec3f(f32(cell%8u),f32((cell/8u)%8u),f32(cell/64u))+.5);}
fn cellWidths(cell:u32)->vec3f{return vec3f(select(2.0,targetWidth,cell==TARGET));}
fn incidenceBegin(cell:u32)->u32{return 6u*cell;}
fn incidenceEnd(cell:u32)->u32{return 6u*cell+6u;}
fn incidenceRow(at:u32)->u32{return at;}
fn incidenceTerm(at:u32)->u32{return at;}
fn termCoefficient(at:u32)->f32{_=at;return .5;}
fn rowStaticDualWeight(row:u32)->f32{_=row;return 8.0;}
fn rowTermOffset(row:u32)->u32{_=row;return 0u;}
fn rowTermCount(row:u32)->u32{_=row;return 1u;}
fn termCell(term:u32)->u32{_=term;return TARGET;}
fn destinationFaceVelocity()->u32{return 0u;}
fn acceptedRowMember(row:u32)->bool{return row<TARGET;}
fn rowAxis(row:u32)->u32{if(row==TARGET){return targetAxis;}return (row%6u)/2u;}
fn rowCenter(row:u32)->vec3f{
 if(row==TARGET){return samplePoint;}var q=cellCenter(row/6u);
 q[rowAxis(row)]+=select(-1.0,1.0,(row&1u)!=0u);return q;
}
fn field(q:vec3f,axis:u32)->f32{
 return sin(.392699081699*q[axis])+0.125*floor(q[(axis+1u)%3u]/2.0);
}
${production("candidateAcceptedFaceSample")}
${production("candidateRemappedFaceVelocity")}
@compute @workgroup_size(1)
fn main(){
 for(var row=0u;row<3072u;row+=1u){state[row]=field(rowCenter(row),rowAxis(row));}
 var output=3072u;
 for(var axis=0u;axis<3u;axis+=1u){
  targetAxis=axis;targetWidth=1.0;
  // All four refined subfaces on an existing coarse face inherit its flux.
  for(var child=0u;child<4u;child+=1u){
   samplePoint=vec3f(6.5);samplePoint[axis]=8.0;
   samplePoint[(axis+1u)%3u]+=f32(child&1u);samplePoint[(axis+2u)%3u]+=f32(child>>1u);
   state[output]=candidateRemappedFaceVelocity(TARGET);output+=1u;
  }
  // The inserted normal face is exactly the normal-linear midpoint, not an
  // average of already averaged cell velocities.
  samplePoint=vec3f(6.5);samplePoint[axis]=7.0;
  state[output]=candidateRemappedFaceVelocity(TARGET);output+=1u;
  // Coarsening integrates four distinct donor patches over a 4x4 face.
  targetWidth=4.0;samplePoint=vec3f(6.0);samplePoint[axis]=8.0;
  state[output]=candidateRemappedFaceVelocity(TARGET);output+=1u;
 }
}`;
    const shader = device!.createShaderModule({ code });
    assert.deepEqual((await shader.getCompilationInfo()).messages.filter(m => m.type === "error"), []);
    const pipeline = device!.createComputePipeline({ layout: "auto", compute: { module: shader, entryPoint: "main" } });
    const buffer = device!.createBuffer({ size: 4 * 3090, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const copy = device!.createBuffer({ size: 72, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = device!.createCommandEncoder(); const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline); pass.setBindGroup(0, device!.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer } }] }));
    pass.dispatchWorkgroups(1); pass.end(); encoder.copyBufferToBuffer(buffer, 4 * 3072, copy, 0, 72);
    device!.queue.submit([encoder.finish()]); await copy.mapAsync(GPUMapMode.READ);
    const values = new Float32Array(copy.getMappedRange()).slice();
    for (let axis = 0; axis < 3; axis++) {
      for (let child = 0; child < 4; child++) assert.ok(Math.abs(values[6 * axis + child]! - .375) < 1e-6);
      assert.ok(Math.abs(values[6 * axis + 4]! - (.5 * Math.sin(3 * Math.PI / 4) + .375)) < 1e-6);
      assert.ok(Math.abs(values[6 * axis + 5]! - .3125) < 1e-6);
    }
    copy.unmap(); copy.destroy(); buffer.destroy();
  } finally { device?.destroy(); live.clear(); await releaseWebGPUExclusiveLock(); }
});
