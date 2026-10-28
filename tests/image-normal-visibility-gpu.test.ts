import assert from "node:assert/strict";
import test from "node:test";
import { createSvoDrySceneFragmentWGSL } from "../lib/svo/features/shading/program";
import { svoGBufferWGSL } from "../lib/svo/contracts/svo-gbuffer";
import { rasterReceiverPlaneWGSL } from "../lib/svo/features/lighting-visibility/svo-raster-ao";
import { withUniformDevice } from "./helpers/uniform-geometric";

function declaration(source: string, prefix: string): string {
  const start = source.indexOf(prefix);
  assert.ok(start >= 0, prefix);
  let at = source.indexOf("{", start), depth = 1;
  for (at++; depth && at < source.length; at++) {
    if (source[at] === "{") depth++;
    if (source[at] === "}") depth--;
  }
  assert.equal(depth, 0);
  return source.slice(start, at);
}

// Execute the actual publication and metadata decoder, not a restatement of
// their desired behavior. A tilted receiver avoids the axis-only metadata
// special case that previously concealed the lost geometry normal.
test("GPU image normals alter optical detail without altering voxel shadow receiver planes", { skip: !process.env.WEBGPU_NODE_MODULE }, async () => {
  const source = createSvoDrySceneFragmentWGSL(1, "raster-primary", "bounds", "split", 0, false, false, false, { surfaceMesh: true });
  const functions = ["dryRasterPrimarySurface", "dryRasterPrimaryFacedSurface", "dryOpaqueSurfaceMetadata", "dryOpaqueFaceWord", "dryOpaqueFaceNormal", "dryReconstructedReceiver", "dryGeometricNormal"]
    .map(name => declaration(source, `fn ${name}(`)).join("\n");
  await withUniformDevice("image normal visibility separation", async device => {
    const module = device.createShaderModule({ code: `
${svoGBufferWGSL}
${rasterReceiverPlaneWGSL}
${declaration(source, "struct DryHit {")}
${declaration(source, "struct DryRasterPrimaryOut{")}
const DRY_OWNER_NONE=65535u;const DRY_MEDIUM_OPAQUE=3u;const SVO_FEATURE_SMOOTH=0u;
const DRY_GBUFFER_MOTION_RIGID=1u;const DRY_GBUFFER_MOTION_STATIC=0u;const DRY_GBUFFER_HARD_FEATURE=256u;
const DRY_OPAQUE_RECONSTRUCTED=0x80000000u;const DRY_OPAQUE_SAME_NORMAL=0x40000000u;
const DRY_OPAQUE_FACE_VALID=134217728u;const DRY_OPAQUE_FACE_SHIFT=28u;
struct Motion{velocity_m_s:vec3f,generation:u32,valid:u32}
fn dryRigidMotionSurface(hit:DryHit,p:vec3f)->Motion{return Motion(vec3f(0),1u,1u);}
fn dryPublicationGeneration()->u32{return 1u;}
fn dryMediumPair(rd:vec3f,n:vec3f,medium:u32)->vec2u{return vec2u(0u,medium);}
fn dryResolvedMaterialId(hit:DryHit)->u32{return hit.materialId;}
fn dryHardwareDepth(t:f32,rd:vec3f,forward:vec3f)->f32{return .1/t;}
var<private> detail:u32;
fn dryImageSurfaceNormal(hit:DryHit,p:vec3f)->vec3f{
  if(detail==0u){return hit.normal;}
  return normalize(select(vec3f(-.8,.5,1),vec3f(.9,-.5,1),detail==2u));
}
${functions}
${declaration(source, "struct MeshVertexOut {")}
${declaration(source, "struct MeshSurfaceOut {")}
struct Mapping{worldOrigin:vec3f,cellSize:vec3f}
struct DryStub{mapping:Mapping,meshFilter:vec4f,meshFilterNormals:vec4f}
var<private> dry:DryStub;var<private> meshHeader:array<u32,256>;
struct DryShadingNormal{normal:vec3f,featureId:u32}
struct Region{identity:u32}
fn meshDcEnabled()->bool{return true;}
fn meshContoursEnabled()->bool{return true;}
fn meshBoundarySample(p:vec3f)->vec4f{return vec4f(0);}
fn meshBoundaryNormal(p:vec3f,b:vec4f)->vec3f{return normalize(vec3f(-.8,.5,1));}
fn meshAuthoredSurfaceNormal(p:vec3f,m:u32,n:vec3f)->vec3f{return normalize(vec3f(-.8,.5,1));}
fn meshRegionAt(p:vec3f)->Region{return Region(7u);}
fn sceneIdentityMaterial(word:u32)->u32{return word;}
fn sceneIdentitySolid(word:u32)->bool{return true;}
fn sceneIdentityHasNormal(word:u32)->bool{return true;}
fn sceneIdentityNormal(word:u32)->vec3f{return vec3f(0,1,0);}
fn meshFilteredNormal(a:vec3f,b:vec3f,l:u32,c:f32,p:f32)->vec3f{return b;}
fn dryShadingNormal(word:u32,n:vec3f)->DryShadingNormal{return DryShadingNormal(n,0u);}
fn dryRasterPrimaryCamera()->mat4x3f{return mat4x3f(vec3f(0,0,5),vec3f(0,0,-1),vec3f(1,0,0),vec3f(0,1,0));}
const DRY_GBUFFER_FIELD_VOXEL=1u;
${declaration(source, "fn surfaceMeshFragment(")}
@group(0) @binding(0) var<storage,read_write> output:array<vec4f>;
@compute @workgroup_size(1) fn main(){
  let face=normalize(vec3f(.3,.5,1));let rd=vec3f(0,0,-1);
  let original=DryHit(5.0,face,7u,12u,0u,4u,0u,1u,0.0,vec3u(0));
  let actualGradient=rasterSunReceiverGradient(face,10.0,20.0);
  let offset=vec2f(.1,.02);let planeDepth=.5+dot(actualGradient,offset);
  for(var i=0u;i<3u;i++){
    detail=i;let published=dryRasterPrimarySurface(original,vec3f(0,0,5),rd,rd,4u);
    var restored=original;restored.normal=published.geometry.xyz;restored.aux.y=published.opaqueIdentity.y;
    let receiver=dryGeometricNormal(restored);let gradient=rasterSunReceiverGradient(receiver,10.0,20.0);
    let expectedDepth=.5+dot(gradient,offset);
    let shaded=svoGBufferUnpackNormalOct8(published.packedSurface.x>>16u);
    output[i*3u]=vec4f(receiver,published.geometry.w);
    output[i*3u+1u]=vec4f(shaded,f32(published.identityMedia.y));
    // The plane cannot occlude itself. A separate nearer blocker still does.
    output[i*3u+2u]=vec4f(expectedDepth-planeDepth,select(0.0,1.0,planeDepth<expectedDepth-1e-5),select(0.0,1.0,planeDepth-.03<expectedDepth-1e-5),dot(rasterSunReceiverGradient(shaded,10.0,20.0)-actualGradient,offset));
  }
  // Actual structural fragment: continuous authored boundary normal differs
  // from the fitted triangle whose positions camera and shadow maps rasterize.
  detail=0u;dry.mapping.cellSize=vec3f(1);
  for(var i=0u;i<256u;i++){meshHeader[i]=1u;}
  let input=MeshVertexOut(vec4f(0,0,0,1),vec3f(0),7u,face,0u,1.0,1.0,face);
  let published=surfaceMeshFragment(input);
  var restored=original;restored.ownerId=DRY_OWNER_NONE;restored.normal=published.geometry.xyz;restored.aux.y=published.opaqueIdentity.y;
  let receiver=dryGeometricNormal(restored);
  let gradient=rasterSunReceiverGradient(receiver,10.0,20.0);
  output[9]=vec4f(receiver,0);
  output[10]=vec4f(svoGBufferUnpackNormalOct8(published.packedSurface.x>>16u),0);
  output[11]=vec4f(dot(gradient-actualGradient,offset),published.geometry.xyz);
}` });
    assert.deepEqual((await module.getCompilationInfo()).messages.filter(m => m.type === "error"), []);
    const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "main" } });
    const output = device.createBuffer({ size: 12 * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const read = device.createBuffer({ size: output.size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const bind = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: output } }] });
    const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
    pass.setPipeline(pipeline); pass.setBindGroup(0, bind); pass.dispatchWorkgroups(1); pass.end();
    encoder.copyBufferToBuffer(output, 0, read, 0, read.size); device.queue.submit([encoder.finish()]);
    await read.mapAsync(GPUMapMode.READ); const actual = Array.from(new Float32Array(read.getMappedRange()));
    for (let i = 0; i < 3; i++) {
      assert.deepEqual(actual.slice(i * 12, i * 12 + 4), actual.slice(0, 4), "receiver normal and depth remain invariant");
      assert.equal(actual[i * 12 + 7], 12, "sampled owner survives");
      assert.ok(Math.abs(actual[i * 12 + 8]!) < 1e-6, "receiver-plane depth is unchanged");
      assert.equal(actual[i * 12 + 9], 0, "no false planar self-shadow");
      assert.equal(actual[i * 12 + 10], 1, "a nearer blocker remains visible to shadow test");
    }
    assert.notDeepEqual(actual.slice(4, 7), actual.slice(16, 19), "optical image normal remains distinct");
    assert.ok(Math.abs(actual[23]!) > .05 && Math.abs(actual[35]!) > .05, "fixture exposes false receiver gradients if optical normal is used");
    for(let axis=0;axis<3;axis++)assert.ok(Math.abs(actual[36+axis]!-actual[axis]!)<.012,"triangle receiver survives the real structural fragment and metadata quantization");
    assert.ok(actual[40]!<-.4,"smooth boundary optical normal remains");
    assert.ok(Math.abs(actual[44]!)<.003,"triangle receiver depth differs only by bounded oct8 quantization");
    assert.ok(actual[45]!<-.4,"full precision optical normal is retained separately");
    read.unmap(); read.destroy(); output.destroy();
  });
});
