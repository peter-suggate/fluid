import assert from "node:assert/strict";
import test from "node:test";
import { createSvoDrySceneFragmentWGSL } from "../lib/svo/features/shading/program";
import { withUniformDevice } from "./helpers/uniform-geometric";
import { readMixedBuffer } from "./helpers/uniform-mixed-native-fields";

test("GPU cached shadows skip absent bodies but retain finite and directional body blockers",{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
  const source=createSvoDrySceneFragmentWGSL(1,"raster-primary","bounds","split",0,false,false,false,{rasterAo:true,opaqueRasterLighting:true});
  const begin=source.lastIndexOf("fn dryLightVisibilitySolid(");
  const code=source.slice(begin,source.indexOf("fn dryContactVisibilitySolid",begin));
  assert.ok(code.length>1000);
  await withUniformDevice("bodyless mapped shadow",async device=>{
    const out=device.createBuffer({size:8*16,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
    try{
      const module=device.createShaderModule({code:`
struct Dry{materialPublication:vec4u,tuningRays0:vec4f,mapping:Mapping}
struct Mapping{cellSize:vec3f} struct Uniforms{options:vec4f}
struct Light{identity:vec4u} struct Lighting{lights:array<Light,32>}
struct Ray{origin_m:vec3f,tMax_m:f32} struct Hit{t:f32}
const SVO_LIGHT_DIRECTIONAL=1u;
var<private> dry:Dry;var<private> uniforms:Uniforms;var<private> dryLighting:Lighting;
var<private> rasterResolvedVisibility:vec2f;
var<private> rays:u32;var<private> bounds:u32;var<private> queries:u32;var<private> blocker:f32;
fn rasterSpotSlot(i:u32,p:vec3f)->i32{return select(0,-1,i==0u);}
fn rasterSpotVisibility(p:vec3f,n:vec3f,s:u32)->f32{return .25;}
fn directionalLightSceneExitDistance(p:vec3f,d:vec3f)->f32{bounds++;return 10.0;}
fn dryBiasedVisibilityRayUnit(p:vec3f,n:vec3f,d:vec3f,t:f32,c:vec3f,b:f32)->Ray{rays++;return Ray(p,t);}
fn nearestBodyIgnoring(p:vec3f,d:vec3f,o:u32)->Hit{queries++;return Hit(blocker);}
${code}
@group(0) @binding(0) var<storage,read_write> result:array<vec4f>;
@compute @workgroup_size(1) fn main(){
 dry.materialPublication.w=2u;dry.tuningRays0.y=.8;dryLighting.lights[0].identity.x=1u;rasterResolvedVisibility.y=.5;
 for(var i=0u;i<8u;i++){
  rays=0u;bounds=0u;queries=0u;uniforms.options.z=select(0.0,1.0,i>=2u);blocker=select(20.0,2.0,i==3u||i==5u);
  let distance=select(4.0,0.0,i==1u||i==4u||i==5u);let light=select(1u,0u,distance==0.0);
  let normal=select(vec3f(0,1,0),vec3f(0,-1,0),i==6u);dry.materialPublication.w=select(2u,0u,i==7u);
  let v=dryLightVisibilitySolid(vec3f(0),normal,0u,vec3f(0,1,0),distance,light);
  result[i]=vec4f(v.x,f32(rays),f32(bounds),f32(queries));
 }
}`});
      assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==="error"),[]);
      const pipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"main"}});
      const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:out}}]}));pass.dispatchWorkgroups(1);pass.end();device.queue.submit([encoder.finish()]);
      const actual=await readMixedBuffer(device,out);
      const expected=[.4,0,0,0, .6,0,0,0, .4,1,0,1, .2,1,0,1, .6,1,1,1, .2,1,1,1, 0,0,0,0, 1,0,0,0];
      actual.forEach((v,i)=>assert.ok(Math.abs(v-expected[i]!)<1e-6,`word ${i}: ${v} vs ${expected[i]}`));
    }finally{out.destroy();}
  });
});
