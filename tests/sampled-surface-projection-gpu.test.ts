import assert from "node:assert/strict";
import test from "node:test";
import { sampledShadowSurfaceWGSL, sampledSurfaceProjectionWGSL } from "../lib/svo/features/lighting-visibility/sampled-surface-projection.wgsl";
import { meshVolumeWGSL } from "../lib/core/imported-assets/mesh-volume.wgsl";
import { meshVolumeSurface } from "../lib/core/imported-assets/mesh-volume-surface";
import { sampleMeshVolume, type MeshVolume } from "../lib/core/imported-assets/mesh-volume";
import { svoPrimitiveWGSL } from "../lib/svo/contracts/svo-primitive-abi";
import { svoProceduralNoiseWGSL } from "../lib/svo/features/materials/svo-procedural-material";
import { withUniformDevice } from "./helpers/uniform-geometric";

function fixture(farBlocker: boolean) {
  const distance = new Float32Array(729);
  const words = new Uint32Array(18 + 729 * 2), floats = new Float32Array(words.buffer);
  floats[3] = 1; words.set([8, 8, 8], 4); floats[7] = 10; words.set([1, 1, 1], 8); words[16] = 18;
  for (let i = 0; i < 729; i++) {
    const x = i % 9, y = Math.floor(i / 9) % 9, z = Math.floor(i / 81);
    distance[i] = Math.min(Math.hypot(x - 4, y - 2.5, z - 4) - 1.8,
      farBlocker ? Math.hypot(x - 4, y - 6.5, z - 4) - 1.2 : Infinity);
    floats[18 + i * 2] = distance[i]!; words[19 + i * 2] = 0x18080;
  }
  const volume: MeshVolume = { header: { version: 1, units: "metres", upAxis: "Y", asset: "projection-fixture",
    origin: [0, 0, 0], dimensions: [8, 8, 8], cellSize: 1, bandWidth: 10, brickCount: 1, brickCells: 8,
    nodeStrideBytes: 16, materials: [{ name: "stone", baseColorLinear: [.3, .3, .3], roughness: .6, metallic: 0 }] },
    bricks: new Map([["0,0,0", { origin: [0, 0, 0], distance,
      geometricNormal: new Uint32Array(729).fill(0xffff8000), shadingNormal: new Uint32Array(729).fill(0xffff8000), material: new Uint32Array(729).fill(1) }]]) };
  return { words, volume };
}
function receiver() {
  const { volume } = fixture(false), mesh = meshVolumeSurface(volume);
  for (let i = 0; i < mesh.levels[0]!.count; i += 3) {
    const p = [0, 1, 2].map(axis => [0, 1, 2].reduce((sum, k) => sum + mesh.positions[mesh.indices[i + k]! * 3 + axis]!, 0) / 3 + 4) as [number, number, number];
    if (p[1] > 4 && Math.hypot(p[0] - 4, p[2] - 4) < .7 && sampleMeshVolume(volume, p).distance < -.005) return p;
  }
  throw new Error("Fixture must contain an actual marching-cubes receiver inside the trilinear field");
}

test("FVX projection fixture exercises the extracted-plane/field discrepancy", () => {
  const p = receiver();
  assert.ok(sampleMeshVolume(fixture(false).volume, p).distance < -.005);
});

test("GPU sampled shadow origin leaves its own surface while preserving farther same-owner blockers", { skip: !process.env.WEBGPU_NODE_MODULE }, async () => {
  const p = receiver(), stride = fixture(false).words.length;
  const words = new Uint32Array(stride * 2); words.set(fixture(false).words); words.set(fixture(true).words, stride);
  await withUniformDevice("FVX shadow origin reconciliation", async device => {
    const module = device.createShaderModule({ code: `
@group(0) @binding(0) var<storage,read> data:array<u32>;
@group(0) @binding(1) var<storage,read_write> output:array<vec4f>;
fn loadWord(at:u32)->u32{return data[at];}
${meshVolumeWGSL("fixtureVolume", "loadWord")}
${sampledSurfaceProjectionWGSL("loadWord")}
struct SvoFieldValue{distance_m:f32,lipschitz:f32}
fn svoFieldProgramReferenceSample(reference:u32,p:vec3f)->SvoFieldValue{return SvoFieldValue(fixtureVolumeDistance(reference,p),1.75);}
${svoProceduralNoiseWGSL}
${svoPrimitiveWGSL}
const DRY_OWNER_NONE=65535u;const DRY_SCENE_FIELD_PROGRAM_WORD_OFFSET=0u;
var<private> fixtureBase:u32;var<private> fixtureCuts:u32;
fn svoSculptCutCount()->u32{return fixtureCuts;}
fn drySampledMaterialOwner(owner:u32)->u32{return select(0xffffffffu,0u,owner==1u);}
fn drySceneWord(at:u32)->u32{return select(fixtureBase,0xfffffffeu,at==0u);}
fn dryPrimitive(index:u32)->SvoPrimitiveRecord{return SvoPrimitiveRecord(vec4u(0u,0u,0u,SVO_KIND_FIELD_PROGRAM),vec4u(bitcast<vec3u>(vec3f(8)),65537u),vec4f(0,0,0,1),vec4u(1,0,0,0));}
var<private> fixtureEvaluations:u32;
${sampledShadowSurfaceWGSL.replace("fn drySampledShadowSurfaceUncached(position:vec3f,normal:vec3f,owner:u32)->SampledSurfaceProjection{", "fn drySampledShadowSurfaceUncached(position:vec3f,normal:vec3f,owner:u32)->SampledSurfaceProjection{fixtureEvaluations++;")}
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id:vec3u){
  let base=id.x*${stride}u;fixtureBase=base;let p=vec3f(${p.join(",")});
  let record=SvoPrimitiveRecord(vec4u(0u,0u,0u,SVO_KIND_FIELD_PROGRAM),vec4u(bitcast<vec3u>(vec3f(8)),65537u),vec4f(0,0,0,1),vec4u(1,base,0,0));
  let before=svoIntersectPrimitiveExact(record,p+vec3f(0,.001,0),vec3f(0,1,0),.0001,20.,SvoClusterPacking());
  let projected=sampledSurfaceProjection(base,p);
  let after=svoIntersectPrimitiveExact(record,projected.position+projected.normal*.001,vec3f(0,1,0),.0001,20.,SvoClusterPacking());
  let field=sampledSurfaceField(base,projected.position);
  output[id.x*8u]=vec4f(before.t_m,after.t_m,f32(before.status),f32(after.status));
  output[id.x*8u+1u]=vec4f(projected.position,f32(projected.valid));
  output[id.x*8u+2u]=vec4f(field.gradient,field.distance);
  let invalid=sampledSurfaceProjection(base,vec3f(-1,0,0));let distant=sampledSurfaceProjection(base,vec3f(4,2.5,4));
  output[id.x*8u+3u]=vec4f(f32(invalid.valid),f32(distant.valid),length(projected.position-p),dot(projected.normal,vec3f(0,1,0)));
  let ownerless=drySampledShadowSurface(p,vec3f(0,1,0),DRY_OWNER_NONE);
  let malformed=drySampledShadowSurface(p,vec3f(0,1,0),2u);
  let owned=drySampledShadowSurface(p,vec3f(0,1,0),1u);
  output[id.x*8u+4u]=vec4f(f32(ownerless.valid),length(ownerless.position-p),f32(malformed.valid),f32(owned.valid));
  fixtureCuts=1u;let cut=drySampledShadowSurface(p,vec3f(0,1,0),1u);
  output[id.x*8u+5u]=vec4f(f32(cut.valid),length(cut.position-p),length(owned.position-projected.position),0);
  fixtureCuts=0u;let expected=drySampledShadowSurfaceUncached(p,vec3f(0,1,0),1u);
  fixtureEvaluations=0u;var equal=true;
  for(var sample=0u;sample<64u;sample++){
    let cached=drySampledShadowSurface(p,vec3f(0,1,0),1u);
    equal=equal&&all(bitcast<vec3u>(cached.position)==bitcast<vec3u>(expected.position))
      &&all(bitcast<vec3u>(cached.normal)==bitcast<vec3u>(expected.normal))&&cached.valid==expected.valid;
  }
  output[id.x*8u+6u]=vec4f(f32(fixtureEvaluations),select(0.,1.,equal),0.,0.);
  let changedNormal=drySampledShadowSurface(p,vec3f(1,0,0),1u);
  let changedOwner=drySampledShadowSurface(p,vec3f(1,0,0),2u);
  let repeatedInvalid=drySampledShadowSurface(p,vec3f(1,0,0),2u);
  let changedPosition=drySampledShadowSurface(p+vec3f(.001,0,0),vec3f(1,0,0),2u);
  fixtureCuts=1u;let changedCuts=drySampledShadowSurface(p+vec3f(.001,0,0),vec3f(1,0,0),2u);
  output[id.x*8u+7u]=vec4f(f32(fixtureEvaluations),f32(changedOwner.valid),f32(repeatedInvalid.valid),f32(changedCuts.valid));
}` });
    assert.deepEqual((await module.getCompilationInfo()).messages.filter(m => m.type === "error"), []);
    const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "main" } });
    const data = device.createBuffer({ size: words.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const output = device.createBuffer({ size: 256, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const read = device.createBuffer({ size: 256, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    try {
      device.queue.writeBuffer(data, 0, words);
      const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
      pass.setPipeline(pipeline); pass.setBindGroup(0, device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [data, output].map((buffer, binding) => ({ binding, resource: { buffer } })) }));
      pass.dispatchWorkgroups(2); pass.end(); encoder.copyBufferToBuffer(output, 0, read, 0, 256); device.queue.submit([encoder.finish()]);
      await read.mapAsync(GPUMapMode.READ); const result = new Float32Array(read.getMappedRange()).slice(); read.unmap();
      for (let i = 0; i < 2; i++) {
        const r = result.subarray(i * 32, (i + 1) * 32);
        assert.equal(r[2], 1, "the old origin has a native self-hit"); assert.ok(r[0]! < .2);
        assert.equal(r[7], 1, "bounded projection converges"); assert.ok(Math.abs(r[11]!) < 1e-5);
        assert.equal(r[12], 0, "out-of-volume input is explicitly invalid");
        assert.equal(r[13], 0, "deep interior cannot jump to a different surface");
        assert.ok(r[14]! < .5 && r[15]! > .5);
        assert.equal(r[16], 1, "ownerless merged voxel is valid without source projection"); assert.equal(r[17], 0);
        assert.equal(r[18], 0, "unknown identified owner is an explicit failure"); assert.equal(r[19], 1);
        assert.equal(r[20], 1, "edited cut caps preserve their retained-field origin"); assert.equal(r[21], 0); assert.equal(r[22], 0);
        assert.equal(r[24],1,"64 emitter samples reuse one exact receiver projection");assert.equal(r[25],1,"cached position, normal and status preserve every bit");
        assert.equal(r[28],5,"normal, owner, position and cut changes invalidate; repeated invalid results remain cached");
        assert.equal(r[29],0);assert.equal(r[30],0);assert.equal(r[31],1);
        const cpu = sampleMeshVolume(fixture(Boolean(i)).volume, [r[4]!, r[5]!, r[6]!]);
        for (let axis = 0; axis < 3; axis++) assert.ok(Math.abs(cpu.gradient[axis]! - r[8 + axis]!) < 2e-6, "analytic gradient agrees with CPU");
      }
      assert.equal(result[3], 0, "unobstructed outward ray stays clear");
      assert.equal(result[35], 1, "farther same-owner blocker is retained");
      assert.ok(result[33]! > 1 && result[33]! < 3, `farther blocker distance ${result[33]}`);
    } finally { if (read.mapState === "mapped") read.unmap(); read.destroy(); output.destroy(); data.destroy(); }
  });
});
