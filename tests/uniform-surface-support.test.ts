import {readFileSync} from "node:fs";
import assert from "node:assert/strict";
import test from "node:test";
import {withUniformDevice} from "./helpers/uniform-geometric";
import {readMixedBuffer} from "./helpers/uniform-mixed-native-fields";

test("production support table indices cover all64 rectangular masks and page boundaries",{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 const source=readFileSync(new URL("../lib/methods/uniform/uniform-mixed-surface.ts",import.meta.url),"utf8");
 const helpers=source.slice(source.indexOf("fn umSupportIndex"),source.indexOf("fn umRequiredRho"));
 const count=512,data=new Float32Array(8*count),bits=new Uint32Array(data.buffer);
 for(let i=0;i<count;i++){
  const index=i%64,codes=[index%4,Math.floor(index/4)%4,Math.floor(index/16)],ranges=codes.map(c=>c===0?[0]:c===1?[-1,0]:c===2?[0,1]:[-1,0,1]);let mask=0;
  for(const z of ranges[2]!)for(const y of ranges[1]!)for(const x of ranges[0]!)mask|=1<<(x+1+3*(y+1)+9*(z+1));
  const p=[0,1,8,15,16],position=[p[i%5]!,p[Math.floor(i/5)%5]!,p[Math.floor(i/25)%5]!],reach=[1,2.5,4,7,8.5,15][Math.floor(i/64)%6]!;
  const code=position.map(v=>(v-reach<0?1:0)+(v+reach>=16?2:0));
  bits[8*i]=mask;bits[8*i+1]=index;data.set([...position,reach],8*i+2);bits[8*i+6]=code[0]!+4*code[1]!+16*code[2]!;
 }
 await withUniformDevice("surface support indices",async device=>{
  const input=device.createBuffer({size:data.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),output=device.createBuffer({size:16*count,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
  try{
   device.queue.writeBuffer(input,0,data);
   const module=device.createShaderModule({code:`fn umPageCoord(region:u32)->vec3u{return vec3u(0);}\n${helpers}
@group(0) @binding(0) var<storage,read> a:array<u32>;
@group(0) @binding(1) var<storage,read_write> out:array<vec4f>;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id:vec3u){let b=8u*id.x;
 let p=bitcast<vec3f>(vec3u(a[b+2u],a[b+3u],a[b+4u]));
 out[id.x]=vec4f(f32(umSupportMaskIndex(a[b])),f32(a[b+1u]),f32(umSupportIndex(0u,p,bitcast<f32>(a[b+5u]))),f32(a[b+6u]));
}`});
   assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==="error").map(m=>m.message),[]);
   const pipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"main"}}),group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[input,output].map((buffer,binding)=>({binding,resource:{buffer}}))});
   const e=device.createCommandEncoder(),p=e.beginComputePass();p.setPipeline(pipeline);p.setBindGroup(0,group);p.dispatchWorkgroups(count);p.end();device.queue.submit([e.finish()]);const values=await readMixedBuffer(device,output);
   for(let i=0;i<count;i++){assert.equal(values[4*i],values[4*i+1],`rectangular mask ${i}`);assert.equal(values[4*i+2],values[4*i+3],`point/page support ${i}`);}
  }finally{input.destroy();output.destroy();}
 });
});
