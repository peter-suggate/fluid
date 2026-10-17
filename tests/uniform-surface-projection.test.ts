import {readFileSync} from "node:fs";
import assert from "node:assert/strict";
import test from "node:test";
import {withUniformDevice} from "./helpers/uniform-geometric";
import {readMixedBuffer} from "./helpers/uniform-mixed-native-fields";
import {UNIFORM_SURFACE_VELOCITY_REACH_H} from "../lib/methods/uniform/uniform-surface-transport.wgsl";
const extract=(text:string,name:string)=>{const start=text.indexOf(`fn ${name}(`),body=text.indexOf("{",start);let level=1,end=body+1;for(;level;end++){if(text[end]==="{")level++;else if(text[end]==="}")level--;}return text.slice(start,end);};

test("stored support proves actual page projection identity at faces, edges and diagonal solids",{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 const lattice=readFileSync(new URL("../lib/methods/uniform/uniform-lattice-address.ts",import.meta.url),"utf8"),surface=readFileSync(new URL("../lib/methods/uniform/uniform-mixed-surface.ts",import.meta.url),"utf8");
 const functions=[extract(lattice,"umBlockHeld"),extract(lattice,"umPointNearFromPage"),extract(lattice,"umPointCell"),extract(surface,"umPointFromPage"),extract(surface,"umPointFrom")].join("\n");
 const count=4096,stride=144,data=new Uint32Array(stride*count),floats=new Float32Array(data.buffer);
 let seed=1194;const random=()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed/4294967296;};
 for(let i=0;i<count;i++){
  const b=stride*i,mode=i%8,width=i%2?1:4,near=(i>>1)%2===1,base=mode===6?3:2,half=[(i>>2)&1,(i>>3)&1,(i>>4)&1],origin=half.map(v=>32*base+16*v);
  const p=origin.map((v,a)=>v+[0,1,8,15,16][(Math.floor(i/32)+a)%5]!);
  if(mode===1||mode===2||mode===5)for(let a=0;a<3;a++){origin[a]=32*base+16;p[a]=origin[a]!+15;}
  if(mode===3)for(let a=0;a<3;a++){origin[a]=32*base+16;p[a]=origin[a]!+16;}
  floats.set([...p,width,near||i%3===0?2*width+.5:width,near?2*width+UNIFORM_SURFACE_VELOCITY_REACH_H:UNIFORM_SURFACE_VELOCITY_REACH_H,...origin],b);data[b+9]=mode===6?1:0;
  for(let z=0;z<5;z++)for(let y=0;y<5;y++)for(let x=0;x<5;x++){
   let kind=1;if(mode===4&&random()<.2)kind=2;if(mode===6&&(x===0||y===0||z===0||x===4||y===4||z===4))kind=2;
   data[b+16+x+5*y+25*z]=kind;
  }
  const home=origin.map(v=>Math.floor(v/32));
  if(mode===1||mode===5)data[b+16+home[0]!+1+5*home[1]!+25*home[2]!]=mode===5?0:2;
  if(mode===2||mode===3)data[b+16+home[0]!+1+5*(home[1]!+1)+25*home[2]!]=2;
 }
 await withUniformDevice("surface stored-page closure",async device=>{
  const input=device.createBuffer({size:data.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),output=device.createBuffer({size:16*count,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
  try{
   device.queue.writeBuffer(input,0,data);
   const module=device.createShaderModule({code:`
@group(0) @binding(0) var<storage,read> a:array<u32>;
@group(0) @binding(1) var<storage,read_write> out:array<vec4f>;
const UM_RESIDENT=1u;const UM_SOLID_WORD=0xffffffffu;var<private> fixtureBase:u32;
fn kind(page:vec3i)->u32{if(any(page<vec3i(0))||any(page>=vec3i(5))){return 2u;}return a[fixtureBase+16u+u32(page.x+5*page.y+25*page.z)];}
fn index(page:vec3i)->u32{return u32(page.x+5*page.y+25*page.z);}
fn umSolidIsBox()->bool{return a[fixtureBase+9u]!=0u;}
fn umBlockStored(home:u32)->bool{return home>0u&&home<=125u;}
fn umFramePage()->vec3i{return vec3i(0);}
fn umWorldSolid(page:vec3i)->bool{return kind(page)==2u;}
fn umBlockOf(page:vec3i)->u32{let k=kind(page);if(k==1u){return index(page)+1u;}return select(0u,UM_SOLID_WORD,k==2u);}
fn umBlockNeighbour(home:u32,d:vec3i)->u32{if(!umBlockStored(home)){return home;}let i=home-1u;let page=vec3i(i32(i%5u),i32((i/5u)%5u),i32(i/25u));return umBlockOf(page+d);}
fn umCellKind(cell:vec3i)->u32{return kind(cell>>vec3u(5u));}
${functions}
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id:vec3u){let b=${stride}u*id.x;fixtureBase=b;
 let p=bitcast<vec3f>(vec3u(a[b],a[b+1u],a[b+2u]));let phiReach=bitcast<f32>(a[b+4u]);let velocityReach=bitcast<f32>(a[b+5u]);let origin=bitcast<vec3f>(vec3u(a[b+6u],a[b+7u],a[b+8u]));
 let proofReach=max(phiReach,velocityReach);let low=vec3i(floor((p-vec3f(proofReach)-origin)/16.0));let high=vec3i(floor((p+vec3f(proofReach)-origin)/16.0));var valid=true;
 for(var z=low.z;z<=high.z;z++){for(var y=low.y;y<=high.y;y++){for(var x=low.x;x<=high.x;x++){let page=(vec3i(origin/16.0)+vec3i(x,y,z))>>vec3u(1u);valid=valid&&kind(page)==UM_RESIDENT;}}}
 var held=0u;for(var k=0u;k<8u;k++){let sign=2.0*vec3f(vec3u(k&1u,(k>>1u)&1u,(k>>2u)&1u))-vec3f(1);let q=p+sign*phiReach;if(any(umPointFrom(p,q)!=q)){held++;}}
 out[id.x]=vec4f(f32(valid),f32(held==0u),f32(held),0.0);
}`});
   assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==="error").map(m=>m.message),[]);
   const pipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"main"}}),group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[input,output].map((buffer,binding)=>({binding,resource:{buffer}}))});
   const e=device.createCommandEncoder(),p=e.beginComputePass();p.setPipeline(pipeline);p.setBindGroup(0,group);p.dispatchWorkgroups(count);p.end();device.queue.submit([e.finish()]);const values=await readMixedBuffer(device,output);let valid=0,excludedHeld=0;
   for(let i=0;i<count;i++){if(values[4*i]===1){valid++;assert.equal(values[4*i+1],1,`stored support clamped case${i}`);}else if(values[4*i+2]!>0)excludedHeld++;}
   assert.ok(valid>100&&excludedHeld>100,`${valid} valid supports and${excludedHeld} excluded clamping cases`);
  }finally{input.destroy();output.destroy();}
 });
});
