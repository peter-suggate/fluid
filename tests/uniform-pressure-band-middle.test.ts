import assert from "node:assert/strict";
import test from "node:test";
import { uniformPressureBandMiddleWGSL } from "../lib/methods/uniform/uniform-pressure-band-middle.wgsl";
import { withUniformDevice } from "./helpers/uniform-geometric";
import { readMixedBuffer } from "./helpers/uniform-mixed-native-fields";

test("fused 2h visits preserve separate red-black updates, including unexpected growth",{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 await withUniformDevice("band middle fusion",async device=>{
  // 64 lanes forces the storage fallback above eight slots even on a
  // device supporting 1024. Rows include inactive cells and missing edges.
  for(const count of [1,8,9,130])for(const prolong of [false,true]){
   const rows=8*count,halves=prolong?4:3,colour=prolong?0:1;
   const common=/* wgsl */`
const N=${rows}u;const T=${count}u;
@group(0) @binding(0) var<storage,read_write> coarse:array<f32>;
fn bLive()->u32{return T;}
fn bM(k:u32,c:u32)->u32{return k*N+c;}
fn bC(k:u32,s:u32)->u32{return 9u*N+s;}
fn bNearSlot(s:u32,f:u32)->u32{
 let axis=f/2u;let stride=select(select(1u,5u,axis==1u),25u,axis==2u);
 let coordinate=(s/stride)%5u;
 if((f&1u)==0u){if(coordinate==0u){return 0u;}return s-stride+1u;}
 if(coordinate==4u||s+stride>=T){return 0u;}return s+stride+1u;
}
`;
   const candidate=device.createShaderModule({code:common+uniformPressureBandMiddleWGSL(64,halves,colour,prolong)});
   const reference=device.createShaderModule({code:common+/* wgsl */`
override colour:u32=0u;override prolong:bool=false;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3u){
 let c=id.x;if(c>=N){return;}
 let local=vec3u(c&1u,(c>>1u)&1u,(c>>2u)&1u);let diagonal=coarse[bM(2u,c)];
 if(((local.x+local.y+local.z)&1u)!=colour||diagonal<=0.0){return;}
 var off=0.0;
 for(var f=0u;f<6u;f++){
  let axis=f/2u;var p=vec3i(local);p[axis]+=select(-1,1,(f&1u)!=0u);var slot=c/8u+1u;
  if(p[axis]<0||p[axis]>1){slot=bNearSlot(c/8u,f);p[axis]=(p[axis]+2)%2;}
  if(slot!=0u){let m=(slot-1u)*8u+u32(p.x+2*p.y+4*p.z);var value=coarse[bM(0u,m)];
   if(prolong&&coarse[bM(2u,m)]>0.0){value+=coarse[bC(0u,m/8u)];}
   off+=coarse[bM(3u+f,c)]*value;
  }
 }
 coarse[bM(0u,c)]=(coarse[bM(1u,c)]+off)/diagonal;
}`});
   for(const module of [candidate,reference])assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==="error"),[]);
   const layout=device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}}]});
   const pipelineLayout=device.createPipelineLayout({bindGroupLayouts:[layout]});
   const fused=await device.createComputePipelineAsync({layout:pipelineLayout,compute:{module:candidate,entryPoint:"main"}});
   const separate=[];for(let half=0;half<halves;half++)separate.push(await device.createComputePipelineAsync({layout:pipelineLayout,
    compute:{module:reference,entryPoint:"main",constants:{colour:(colour+half)%2,prolong:+(prolong&&half===0)}}}));
   const data=new Float32Array(9*rows+count);
   for(let c=0;c<rows;c++){
    data[c]=Math.sin(c)*.02;data[rows+c]=Math.cos(c)*.03;data[2*rows+c]=c%17===0?0:8+.17*Math.sin(c+1);
    for(let f=0;f<6;f++)data[(3+f)*rows+c]=.5+(c%5+f)*.03;
   }
   for(let s=0;s<count;s++)data[9*rows+s]=Math.sin(s+1)*.04;
   const buffers=[0,1].map(()=>device.createBuffer({size:data.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST}));
   try{
    const encoder=device.createCommandEncoder();
    for(let arm=0;arm<2;arm++){
     const buffer=buffers[arm]!;device.queue.writeBuffer(buffer,0,data);
     const group=device.createBindGroup({layout,entries:[{binding:0,resource:{buffer}}]}),pass=encoder.beginComputePass();pass.setBindGroup(0,group);
     if(arm===0){pass.setPipeline(fused);pass.dispatchWorkgroups(1);}
     else for(const pipeline of separate){pass.setPipeline(pipeline);pass.dispatchWorkgroups(Math.ceil(rows/64));}
     pass.end();
    }
    device.queue.submit([encoder.finish()]);
    const a=await readMixedBuffer(device,buffers[0]!),b=await readMixedBuffer(device,buffers[1]!);
    assert.deepEqual(a,b,`${count} slots, prolong ${prolong}`);
   }finally{for(const buffer of buffers)buffer.destroy();}
  }
 });
});
