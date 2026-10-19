import {uniformCompiledTopologyWGSL} from "../lib/methods/uniform/uniform-compiled-topology";
import {readFileSync} from "node:fs";
import assert from "node:assert/strict";
import test from "node:test";
import {withUniformDevice} from "./helpers/uniform-geometric";
import {readMixedBuffer} from "./helpers/uniform-mixed-native-fields";

test("production fine listing covers positive seam masks and rectangular low walls",{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 const source=readFileSync(new URL("../lib/methods/uniform/uniform-mixed-surface.ts",import.meta.url),"utf8"),listing=source.slice(source.indexOf("const UM_NO_VERTEX="),source.indexOf("// advectFine's cubic list"));
 await withUniformDevice("surface canonical fine listing",async device=>{
  const output=device.createBuffer({size:16*1024,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
  try{
   const module=device.createShaderModule({code:`
@group(0) @binding(0) var<storage,read_write> out:array<vec4f>;
fn umCorner(k:u32,side:u32)->vec3u{return vec3u(k%side,(k/side)%side,k/(side*side));}
${uniformCompiledTopologyWGSL.slice(uniformCompiledTopologyWGSL.indexOf("const UM_POSITIVE_INCIDENT"),uniformCompiledTopologyWGSL.indexOf("// Window tile offsets"))}
${listing}
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id:vec3u){let wide=(id.x%128u)<<1u;let low=umCorner(id.x/128u,2u)!=vec3u(0);let tile=vec3u(2);var seen:array<u32,125>;var invalid=0u;var returned=0u;var expected=0u;
 for(var round=0u;round<3u;round++){for(var lane=0u;lane<64u;lane++){
  let p=umFineVertex(tile,lane,round,wide,low);if(p.x==UM_NO_VERTEX.x){continue;}let local=p-4u*tile;let key=local.x+5u*local.y+25u*local.z;
  if(any(local>vec3u(4))||seen[key]!=0u){invalid++;}seen[key]++;returned++;
 }}
 for(var k=0u;k<125u;k++){let local=umCorner(k,5u);let p=4u*tile+local;var own=!any((local==vec3u(0))&!low);
  // Independent closed-cell incidence test: a coarse neighboring tile
  // owns every point in its geometric closure, including faces/edges.
  for(var n=1u;n<8u;n++){let origin=4u*(tile+umCorner(n,2u));if((wide&(1u<<n))!=0u&&all(p>=origin)&&all(p<=origin+vec3u(4))){own=false;}}
  if(own){expected++;if(seen[k]!=1u){invalid++;}}else if(seen[k]!=0u){invalid++;}
 }
 out[id.x]=vec4f(f32(invalid),f32(returned),f32(expected),0.0);
}`});
   assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==="error").map(m=>m.message),[]);
   const pipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"main"}}),group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:output}}]});
   const e=device.createCommandEncoder(),p=e.beginComputePass();p.setPipeline(pipeline);p.setBindGroup(0,group);p.dispatchWorkgroups(1024);p.end();device.queue.submit([e.finish()]);const values=await readMixedBuffer(device,output);
   for(let i=0;i<1024;i++){assert.equal(values[4*i],0,`wide/low mask${i}`);assert.equal(values[4*i+1],values[4*i+2],`complete listing${i}`);}
  }finally{output.destroy();}
 });
});
