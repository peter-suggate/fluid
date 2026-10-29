import assert from "node:assert/strict";
import test from "node:test";
import { integratedEnvironmentBrdf, resolveUnifiedShadingNormal, unifiedLightingShaderLibrary } from "../lib/core/webgpu-lighting";
import { withUniformDevice } from "./helpers/uniform-geometric";
import { rasterLightVisibilityWGSL } from "../lib/svo/features/lighting-visibility/raster-light-visibility";

const cases = [
  { shading: [0.6, 0, 0.8], geometry: [0, 0, 1], expected: [0.6, 0, 0.8] },
  // Optical normal is in the geometric hemisphere, but behind the viewer.
  { shading: [0.8, 0, -0.6], geometry: [0.8, 0, 0.6], expected: [0.8, 0, 0.6] },
  // Native receivers without independent geometry use their existing two-sided frame.
  { shading: [0.8, 0, -0.6], geometry: [0.8, 0, -0.6], expected: [-0.8, 0, 0.6] },
] as const;

test("a shared optical frame preserves valid detail and rejects behind-view detail", () => {
  for (const c of cases) {
    const result = resolveUnifiedShadingNormal(c.shading, c.geometry, [0, 0, 1]);
    result.forEach((v, i) => assert.ok(Math.abs(v - c.expected[i]!) < 1e-6));
  }
});

test("GPU cached visibility does not discard blockers behind raw optical backfaces", { skip: !process.env.WEBGPU_NODE_MODULE }, async () => {
  const start=rasterLightVisibilityWGSL.indexOf("fn rasterProduceLightVisibility(");
  const end=rasterLightVisibilityWGSL.indexOf("fn rasterLoadVisibilityGuide",start);
  // Execute the entire production producer. Only its storage sink is adapted.
  const producer=rasterLightVisibilityWGSL.slice(start,end).replace("textureStore(rasterLightVisibilityOutput,", "storeResult(");
  await withUniformDevice("optical-independent cached visibility",async device=>{
    const output=device.createBuffer({size:4*64*16,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
    const read=device.createBuffer({size:output.size,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
    const module=device.createShaderModule({code:`
struct Hit{normal:vec3f,materialId:u32,ownerId:u32}
struct Light{identity:vec4u}
struct Lighting{metadata:vec4u,lights:array<Light,32>}
struct Dry{tuningCounts0:vec4u,tuningCounts1:vec4u,materialPublication:vec4u}
struct Uniforms{options:vec4f}
struct Sample{valid:u32,towardLight:vec3f,finiteDistance_m:f32}
var<private> dry:Dry;var<private> dryLighting:Lighting;var<private> uniforms:Uniforms;
var<private> dryVisibilityIgnoredBody:u32;var<private> drySampledReceiver:bool;var<private> rasterResolvedVisibility:vec2f;
const DRY_OWNER_NONE=65535u;const SVO_LIGHT_SPHERE_AREA=2u;const SVO_LIGHT_RECTANGLE_AREA=3u;const SVO_LIGHT_SPOT=4u;const RASTER_VISIBILITY_WATER_SHADOWS=false;
@group(0) @binding(0) var<storage,read_write> output:array<vec4u>;
fn rasterVisibilityHit(c:vec2i,g:vec4f)->Hit{return Hit(vec3f(0,0,-1),0u,12u);}
fn rasterWorldAt(c:vec2f,t:f32)->vec3f{return vec3f(c,0);}
fn dryGeometricNormal(h:Hit)->vec3f{return vec3f(0,0,1);}
fn drySampledMaterial(m:u32)->bool{return true;}
fn rasterSpotSlot(i:u32)->i32{return -1;}
fn dryLightSample(l:Light,i:u32,p:vec3f)->Sample{return Sample(1u,vec3f(0,0,1),3.0);}
fn rasterEvaluateLightVisibility(p:vec3f,n:vec3f,o:u32,l:vec3f,d:f32,i:u32)->vec3f{return vec3f(select(0.0,1.0,p.x>1.0));}
fn rasterAtlasCoordinate(c:vec2i)->vec2i{return c;}
fn rasterPackVisibility(v:vec3f)->u32{return u32(round(v.x*1023.0));}
fn storeResult(c:vec2i,layer:i32,value:vec4u){output[u32(c.x)*64u+u32(layer)]=value;}
${producer}
@compute @workgroup_size(1) fn main(){
  dry.tuningCounts0=vec4u(0,0,1,1);dry.tuningCounts1=vec4u(1);dryLighting.metadata=vec4u(1,1,0,0);dryLighting.lights[0].identity=vec4u(0,0,0,1);
  rasterProduceLightVisibility(vec2i(0,0),vec4f(0,0,-1,1),1.0);
  rasterProduceLightVisibility(vec2i(1,0),vec4f(0,0,-1,1),1.0);
  dry.tuningCounts0.z=32u;dry.tuningCounts0.w=2u;dry.tuningCounts1.x=2u;dryLighting.metadata.x=32u;
  for(var i=0u;i<32u;i++){dryLighting.lights[i].identity=vec4u(SVO_LIGHT_SPHERE_AREA,0,0,1);}
  rasterProduceLightVisibility(vec2i(2,0),vec4f(0,0,-1,1),1.0);
  dry.materialPublication.w=16u;
  rasterProduceLightVisibility(vec2i(3,0),vec4f(0,0,-1,1),1.0);
}`});
    assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==="error"),[]);
    const pipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"main"}});
    const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:output}}]});
    const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(1);pass.end();
    encoder.copyBufferToBuffer(output,0,read,0,read.size);device.queue.submit([encoder.finish()]);await read.mapAsync(GPUMapMode.READ);
    const words=new Uint32Array(read.getMappedRange());assert.equal(words[0],0,"actual blocker is cached despite opposing optical normal");assert.equal(words[64*4],1023,"traced clear space stays clear");
    for(let sample=0;sample<64;sample++)assert.equal(words[(128+sample)*4],1023,`all32 area lights retain both visibility samples, slot${sample}`);
    for(let sample=0;sample<64;sample++)assert.equal(words[(192+sample)*4],1023,`GI preserves both direct visibility samples, slot${sample}`);
    read.unmap();read.destroy();output.destroy();
  });
});

test("GPU direct and environment lobes use the identical visible optical frame", { skip: !process.env.WEBGPU_NODE_MODULE }, async () => {
  await withUniformDevice("optical frame consistency", async device => {
    const output = device.createBuffer({ size: cases.length * 32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const read = device.createBuffer({ size: output.size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const module = device.createShaderModule({ code: `${unifiedLightingShaderLibrary}
@group(0) @binding(0) var<storage,read_write> output:array<vec4f>;
@compute @workgroup_size(1) fn main(){
${cases.map((c, i) => `{
  let source=vec3f(${c.shading.join(",")});let geometry=vec3f(${c.geometry.join(",")});let view=vec3f(0,0,1);
  let resolved=unifiedShadingNormal(source,geometry,view);
  let direct=unifiedLightingInputWithGeometry(resolved,geometry,view,view,vec3f(1));
  let env=unifiedEnvironmentBrdf(max(dot(resolved,view),0.0),.62,vec3f(.04));
  let legacy=unifiedEnvironmentBrdf(max(dot(source,view),0.0),.62,vec3f(.04));
  output[${i * 2}u]=vec4f(resolved,env.x);
  output[${i * 2 + 1}u]=vec4f(direct.normal,legacy.x);
}`).join("\n")}
}` });
    assert.deepEqual((await module.getCompilationInfo()).messages.filter(m => m.type === "error"), []);
    const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "main" } });
    const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: output } }] });
    const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);pass.setBindGroup(0, group);pass.dispatchWorkgroups(1);pass.end();
    encoder.copyBufferToBuffer(output, 0, read, 0, read.size);device.queue.submit([encoder.finish()]);
    await read.mapAsync(GPUMapMode.READ);const values = Array.from(new Float32Array(read.getMappedRange()));
    cases.forEach((c, i) => {
      c.expected.forEach((v, k) => {
        assert.ok(Math.abs(values[i * 8 + k]! - v) < 1e-6);
        assert.ok(Math.abs(values[i * 8 + 4 + k]! - v) < 1e-6, "direct normal agrees with environment");
      });
      const expected = integratedEnvironmentBrdf(c.expected[2], .62, [.04, .04, .04]);
      assert.ok(Math.abs(values[i * 8 + 3]! - expected[0]) < 1e-6);
      if (i > 0) assert.ok(values[i * 8 + 7]! > values[i * 8 + 3]! * 2, "legacy behind-view DFG incorrectly used grazing reflectance");
    });
    read.unmap();output.destroy();read.destroy();
  });
});
