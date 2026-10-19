import assert from "node:assert/strict";
import test from "node:test";
import {uniformSolidRefinementWGSL} from "../lib/methods/uniform/uniform-solid-refinement";
import {withUniformDevice} from "./helpers/uniform-geometric";
import {readMixedBuffer} from "./helpers/uniform-mixed-native-fields";

test("coarse solid geometry certifies whole grid cells and faces, never cut cells, bodies or missing evidence",{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 await withUniformDevice("spatial solid accuracy",async device=>{
  const output=device.createBuffer({size:64,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
  try{
   const module=device.createShaderModule({code:`
@group(0) @binding(0) var<storage,read_write> result:array<u32>;
const UM_TILES=27u;
var<private> solidRecord:array<vec4f,594>;
var<private> testCase:u32;
fn umTileCoord(t:u32)->vec3u{return vec3u(t%3u,(t/3u)%3u,t/9u);}
fn umTileKey(p:vec3i)->u32{if(any(p<vec3i(0))||any(p>vec3i(2))){return 27u;}return u32(p.x+3*p.y+9*p.z);}
fn umKeyStored(t:u32)->bool{return t<27u&&!(testCase==8u&&t==4u);}
${uniformSolidRefinementWGSL(0)}
@compute @workgroup_size(1) fn main(){
 for(var c=0u;c<16u;c++){
  testCase=c;for(var t=0u;t<594u;t++){solidRecord[t]=vec4f(0);}
  for(var t=0u;t<27u;t++){solidRecord[t]=vec4f(1);solidRecord[27u+3u*t+1u]=vec4f(f32(0x999999u));solidRecord[27u+3u*t+2u]=vec4f(f32(0x999999u));}
  if(c==1u){solidRecord[13u].x=.5;}
  if(c==2u){solidRecord[13u].y=.5;}
  if(c==3u){solidRecord[12u].y=.5;}
  if(c==4u){solidRecord[20u*27u+2u*13u+1u].z=1.0;}
  if(c==5u){solidRecord[20u*27u+2u*12u+1u].z=1.0;}
  if(c==6u){solidRecord[12u]=vec4f(0);solidRecord[27u+36u+1u]=vec4f(f32(0x111111u));solidRecord[27u+36u+2u]=vec4f(f32(0x111111u));}
  if(c==7u){solidRecord[13u]=vec4f(0);solidRecord[27u+39u+1u]=vec4f(f32(0x111111u));solidRecord[27u+39u+2u]=vec4f(f32(0x111111u));}
  if(c==9u){solidRecord[13u]=vec4f(1,.5,.5,.5);solidRecord[27u+39u+1u]=vec4f(f32(0x555555u));solidRecord[27u+39u+2u]=vec4f(f32(0x555555u));}
  // A single different patch at either end of each axis must invalidate
  // the certificate, including the two axes sharing packed words.
  if(c>=10u){let slots=array<u32,6>(0u,15u,16u,31u,32u,47u);let slot=slots[c-10u];let word=slot/6u;
   solidRecord[27u+39u+1u+word/4u][word%4u]-=f32(8u<<(4u*(slot%6u)));}
  result[c]=u32(umSolidGeometryExact(13u));
 }
}`});
   assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==="error").map(m=>m.message),[]);
   const pipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"main"}}),group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:output}}]});
   const e=device.createCommandEncoder(),p=e.beginComputePass();p.setPipeline(pipeline);p.setBindGroup(0,group);p.dispatchWorkgroups(1);p.end();device.queue.submit([e.finish()]);
   assert.deepEqual(Array.from(new Uint32Array((await readMixedBuffer(device,output)).buffer)),[1,0,0,0,0,0,1,1,0,1,0,0,0,0,0,0]);
  }finally{output.destroy();}
 });
});
