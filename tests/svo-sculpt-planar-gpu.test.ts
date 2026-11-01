import assert from "node:assert/strict";
import test from "node:test";
import { planarBoundaryWGSL } from "../lib/core/planar-boundary";
import { svoSculptPlanarWGSL } from "../lib/svo/features/scene-publication/svo-sculpt-planar";
import { withUniformDevice } from "./helpers/uniform-geometric";

test("exact planar CSG opens holes, retains cut caps, and preserves untouched material and depth", {
  skip: !process.env.WEBGPU_NODE_MODULE,
}, async () => withUniformDevice("editable exact slabs", async device => {
  const module = device.createShaderModule({ code: `
${planarBoundaryWGSL}
const DRY_MISS=1e30;const SVO_FEATURE_BOX_X=0u;const SVO_FEATURE_SMOOTH=3u;
const DRY_GBUFFER_FIELD_ANALYTIC=0u;const DRY_GBUFFER_MOTION_STATIC=0u;
struct DryHit {t:f32,normal:vec3f,material:u32,owner:u32,feature:u32,field:u32,motion:u32,valid:u32,value:f32,aux:vec3u}
fn missHit()->DryHit{return DryHit(DRY_MISS,vec3f(0),0,0,0,0,0,0,0,vec3u(0));}
struct CutSample {distance:f32,normal:vec3f}
fn svoSculptCutCount()->u32{return 1u;}
// A round 0.5m cut centred on the slab's front face.
fn svoSculptCutDistance(p:vec3f)->f32{return length(p-vec3f(0,0,1))-.5;}
fn svoSculptCutSample(p:vec3f)->CutSample{return CutSample(svoSculptCutDistance(p),normalize(p-vec3f(0,0,1)));}
${svoSculptPlanarWGSL}
@group(0) @binding(0) var<storage,read_write> out:array<vec4f>;
@compute @workgroup_size(1) fn main(){
  let wall=PlanarBoundaryPatch(vec4f(0,0,1.1,.1),vec4f(0,0,1,2),vec4f(1,0,0,2),vec4u(bitcast<vec3u>(vec3f(0,1,0)),17u|(29u<<16u)));
  let thick=PlanarBoundaryPatch(vec4f(0,0,1.5,.5),wall.normalExtentU,wall.tangentUExtentV,wall.tangentVIdentity);
  let hole=dryPlanarBoundaryHit(wall,vec3f(0),vec3f(0,0,1),.0001,10);
  let untouched=dryPlanarBoundaryHit(wall,vec3f(1,0,0),vec3f(0,0,1),.0001,10);
  let cap=dryPlanarBoundaryHit(thick,vec3f(0),vec3f(0,0,1),.0001,10);
  let inside=dryPlanarBoundaryHit(thick,vec3f(0,0,1.1),vec3f(0,0,1),.0001,.6);
  let back=dryPlanarBoundaryHit(thick,vec3f(0,0,1.8),vec3f(0,0,-1),.0001,.6);
  let exact=intersectPlanarBoundary(wall,vec3f(1,0,0),vec3f(0,0,1),.0001,10);
  out[0]=vec4f(hole.t,untouched.t,cap.t,inside.t);
  out[1]=vec4f(cap.normal,back.t);
  out[2]=vec4f(f32(untouched.material),f32(untouched.owner),untouched.t-exact.tHit,distance(untouched.normal,exact.normal));
}
` });
  assert.deepEqual((await module.getCompilationInfo()).messages.filter(m => m.type === "error"), []);
  const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "main" } });
  const output = device.createBuffer({ size: 48, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const read = device.createBuffer({ size: 48, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  try {
    const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: output } }] });
    const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
    pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(1); pass.end();
    encoder.copyBufferToBuffer(output, 0, read, 0, 48); device.queue.submit([encoder.finish()]);
    await read.mapAsync(GPUMapMode.READ); const values = new Float32Array(read.getMappedRange()).slice(); read.unmap();
    assert.ok(values[0]! > 1e20, "a through hole must not leave the original front/back face visible");
    for (const [i, expected] of [[1, 1], [2, 1.5], [3, .4], [6, -1], [7, .3]] as const)
      assert.ok(Math.abs(values[i]! - expected) < 1e-5, `sample ${i}: ${values[i]} != ${expected}`);
    assert.deepEqual([...values.slice(8)], [17, 29, 0, 0], "untouched identity, depth and normal are exact");
  } finally { output.destroy(); read.destroy(); }
}));
