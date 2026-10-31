import assert from "node:assert/strict";
import test from "node:test";
import { importedSurfaceCsgWGSL } from "../lib/svo/features/primary-visibility/imported-surface-csg.wgsl";
import { packSvoSculptCuts, svoSculptCutsWGSL } from "../lib/svo/features/scene-publication/svo-sculpt-cuts";
import { withUniformDevice } from "./helpers/uniform-geometric";

test("live native CSG finds cavity walls, rejects overlapping cut walls and preserves exterior identity", {
  skip: !process.env.WEBGPU_NODE_MODULE,
}, async () => {
  await withUniformDevice("native live CSG", async device => {
    const sphere = { operation: "clear" as const, kind: "ellipsoid" as const, affectsScenery: true,
      minimum: [-5, -5, 5] as const, maximumExclusive: [5, 5, 15] as const };
    const box = { operation: "clear" as const, affectsScenery: true,
      minimum: [-2, -2, 2] as const, maximumExclusive: [2, 2, 8] as const };
    const words = packSvoSculptCuts([sphere, box], { origin_m: [0, 0, 0], cellSize_m: [.1, .1, .1] });
    const input = device.createBuffer({ size: words.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const output = device.createBuffer({ size: 7 * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const read = device.createBuffer({ size: output.size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const module = device.createShaderModule({ code: `
@group(0) @binding(0) var<storage,read> cuts:array<u32>;
@group(0) @binding(1) var<storage,read_write> results:array<vec4f>;
fn drySculptCutWord(i:u32)->u32{return cuts[i];}
${svoSculptCutsWGSL("drySculptCutWord")}
struct Primitive { id:u32 }
struct Dry { metadata:vec4u }
const dry=Dry(vec4u(1,0,0,0));
struct DryHit { t:f32,normal:vec3f,material:u32,owner:u32,feature:u32,field:u32,motion:u32,valid:u32,extra:f32,padding:vec3u }
struct Surface { material:u32,normal:vec3f }
const SVO_KIND_FIELD_PROGRAM=12u;const DRY_SCENE_FIELD_PROGRAM_WORD_OFFSET=0u;const DRY_SCENE_FIELD_PROGRAM_BLOCK_WORDS=4u;
const SVO_FEATURE_SMOOTH=0u;const DRY_GBUFFER_FIELD_ANALYTIC=1u;const DRY_GBUFFER_MOTION_STATIC=0u;
fn missHit()->DryHit{return DryHit(1e20,vec3f(0),0,0,0,0,0,0,0,vec3u(0));}
fn dryPrimitive(i:u32)->Primitive{return Primitive(i);}
fn svoPrimitiveKind(p:Primitive)->u32{return SVO_KIND_FIELD_PROGRAM;}
fn svoPrimitiveFieldProgramReference(p:Primitive)->u32{return 0u;}
fn drySceneWord(i:u32)->u32{return 0xfffffffeu;}
fn dryClusterPacking(p:Primitive)->u32{return 0u;}
fn dryFieldProgramBlockVolumeDistance(base:u32,w:vec3f)->f32{return length(w)-1.0;}
fn svoPrimitiveLocalPoint(p:Primitive,w:vec3f)->vec3f{return w;}
fn dryFieldProgramBlockSurface(i:u32,p:vec3f)->Surface{return Surface(7u,normalize(p));}
fn svoPrimitiveMaterialId(p:Primitive)->u32{return 7u;}
fn svoPrimitiveOwnerId(p:Primitive)->u32{return 9u;}
${importedSurfaceCsgWGSL}
@compute @workgroup_size(1) fn main(){
  let ro=vec3f(0,0,3);let rd=vec3f(0,0,-1);
  let a=importedCutHit(0u,ro,rd,0.0,10.0);results[0]=vec4f(a.t,a.normal.z,f32(a.material),f32(a.owner));
  let b=importedCutHit(1u,ro,rd,0.0,10.0);results[1]=vec4f(b.t,b.normal.z,f32(b.material),f32(b.owner));
  results[2]=vec4f(select(0.0,1.0,importedSurfaceRemoved(vec3f(0,0,1),0xffffffffu)),select(0.0,1.0,importedSurfaceRemoved(vec3f(.8,0,.6),0xffffffffu)),0,0);
  let c=importedCutHit(1u,vec3f(0,0,.7),rd,0.0,10.0);results[3]=vec4f(c.t,c.normal.z,f32(c.material),0);
  results[4]=vec4f(importedCutHit(0u,vec3f(2,0,3),rd,0.0,10.0).t,0,0,0);
  results[5]=vec4f(importedCutHit(1u,ro,rd,0.0,2.7).t,0,0,0);
  results[6]=vec4f(importedCutInterval(0u,vec3f(0,0,3000),rd),0,0);
}` });
    try {
      assert.deepEqual((await module.getCompilationInfo()).messages.filter(m => m.type === "error"), []);
      const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "main" } });
      const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: input } }, { binding: 1, resource: { buffer: output } },
      ] });
      const run = async (count: number) => {
        const image = words.slice(); image[0] = count; device.queue.writeBuffer(input, 0, image);
        const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
        pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(1); pass.end();
        encoder.copyBufferToBuffer(output, 0, read, 0, read.size); device.queue.submit([encoder.finish()]);
        await read.mapAsync(GPUMapMode.READ); const values = new Float32Array(read.getMappedRange().slice(0)); read.unmap(); return values;
      };
      const first = await run(1);
      assert.ok(Math.abs(first[0]! - 2.5) < 1e-5); assert.equal(first[1], 1); assert.equal(first[2], 8199); assert.equal(first[3], 9);
      const overlap = await run(2);
      assert.ok(overlap[0]! > 1e19, "the sphere's interior wall is removed by the overlapping box");
      assert.ok(Math.abs(overlap[4]! - 2.8) < 1e-5); assert.equal(overlap[5], 1); assert.equal(overlap[6], 8199);
      assert.deepEqual(Array.from(overlap.subarray(8, 10)), [1, 0], "only covered original surface is clipped");
      assert.ok(Math.abs(overlap[12]! - .5) < 1e-5, "a camera inside the cavity sees the far wall");
      assert.ok(overlap[16]! > 1e19 && overlap[20]! > 1e19, "off-cut and clipped ray intervals miss");
      assert.deepEqual(Array.from(overlap.subarray(24, 26)), [2998.5, 2999.5], "distant brushes retain their interval instead of losing precision");
      assert.deepEqual(await run(1), first, "undo immediately restores the same GPU result");
    } finally { input.destroy(); output.destroy(); read.destroy(); }
  });
});
