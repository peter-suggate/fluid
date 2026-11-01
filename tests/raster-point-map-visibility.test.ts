import assert from "node:assert/strict";
import test from "node:test";
import { drySceneShader } from "../lib/svo/features/shading/program";
import { rasterLightVisibilityWGSL } from "../lib/svo/features/lighting-visibility/raster-light-visibility";
import { svoLightWGSL } from "../lib/svo/contracts/svo-light-abi";
import { withUniformDevice } from "./helpers/uniform-geometric";
import { readMixedBuffer } from "./helpers/uniform-mixed-native-fields";

test("GPU direct point-map queries preserve normalized sampling and fall back for unsupported receivers",{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 const helper=rasterLightVisibilityWGSL.slice(rasterLightVisibilityWGSL.indexOf("fn rasterPointMappedVisibility"),rasterLightVisibilityWGSL.indexOf("fn rasterProduceLightVisibility"));
 const sampler=drySceneShader.slice(drySceneShader.indexOf("struct DryLightSample"),drySceneShader.indexOf("// The static scene is traceStatic alone"));
 await withUniformDevice("point map sampling parity",async device=>{
  const out=device.createBuffer({size:8192*16,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
  try{
   const module=device.createShaderModule({code:`
const UNIFIED_PI=3.141592653589793;
${svoLightWGSL}${sampler}
struct Dry{materialPublication:vec4u,tuningRays0:vec4f} struct Uniforms{options:vec4f}
var<private> dry:Dry;var<private> uniforms:Uniforms;var<private> RASTER_VISIBILITY_WATER_SHADOWS:bool;
fn rasterSpotVisibility(p:vec3f,n:vec3f,s:u32)->f32{return clamp(.5+.03*dot(p,n),0.0,1.0);}
${helper}
@group(0) @binding(0) var<storage,read_write> result:array<vec4f>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3u){
 let i=id.x;let t=f32(i);let p=vec3f(sin(t*.37),cos(t*.19),sin(t*.83))*12.0;
 var n=normalize(vec3f(cos(t*.13),sin(t*.67),cos(t*.53)));
 var light:SvoLightRecord;light.positionRange=vec4f(cos(t*.7),sin(t*.11),cos(t*.29),select(0.0,8.0,i%17u==0u));
 light.colorIntensity=vec4f(1,2,3,select(4.0,0.0,i%19u==0u));light.shape=vec4f(.2);light.identity.x=select(SVO_LIGHT_POINT,SVO_LIGHT_SPOT,i%13u==0u);light.directionCone=vec4f(0,-1,0,.1);
 if(i%4u==0u){n=normalize(cross(light.positionRange.xyz-p,vec3f(1,0,0)));}
 if(i%23u==0u){light.positionRange=vec4f(p,0);}
 uniforms.options.z=select(0.0,1.0,i%29u==0u);RASTER_VISIBILITY_WATER_SHADOWS=i%31u==0u;
 let slot=select(0,-1,i%37u==0u);dry.materialPublication.w=select(2u,0u,i%7u==0u);dry.tuningRays0.y=f32(i%11u)/10.0;
 let fast=rasterPointMappedVisibility(light,p,n,slot);let sample=dryLightSample(light,0u,p);var reference=1.0;
 if(sample.valid!=0u&&dot(n,sample.towardLight)>0.0&&(dry.materialPublication.w&2u)!=0u){reference=mix(1.0,rasterSpotVisibility(p,n,0u),dry.tuningRays0.y);}
 let unsupported=slot<0||uniforms.options.z>=.5||RASTER_VISIBILITY_WATER_SHADOWS||light.positionRange.w!=0.0||light.identity.x!=SVO_LIGHT_POINT;
 result[i]=vec4f(select(0.0,abs(fast.x-reference),fast.y>0.0),fast.y,select(0.0,1.0,unsupported&&fast.y!=0.0),0);
}`});
   assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==="error"),[]);
   const pipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"main"}});
   const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:out}}]}));pass.dispatchWorkgroups(128);pass.end();device.queue.submit([encoder.finish()]);
   const actual=await readMixedBuffer(device,out);let handled=0;
   for(let i=0;i<8192;i++){assert.equal(actual[i*4],0,`visibility ${i}`);assert.equal(actual[i*4+2],0,`unsupported ${i}`);handled+=actual[i*4+1]!;}
   assert.ok(handled>3000,`${handled} mapped samples exercised`);
  }finally{out.destroy();}
 });
});
