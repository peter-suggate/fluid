import assert from "node:assert/strict";
import test from "node:test";
import {uniformSolidRefinementWGSL} from "../lib/methods/uniform/uniform-solid-refinement";
import {withUniformDevice} from "./helpers/uniform-geometric";
import {readMixedBuffer} from "./helpers/uniform-mixed-native-fields";

test("coarse solid geometry certifies whole grid cells and faces, never cut cells, bodies or missing evidence",{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 await withUniformDevice("spatial solid accuracy",async device=>{
  const output=device.createBuffer({size:80,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
  try{
   const module=device.createShaderModule({code:`
@group(0) @binding(0) var<storage,read_write> result:array<u32>;
const UM_TILES=27u;const UM_SOLID=2u;
var<private> solidRecord:array<vec4f,2300>;
var<private> testCase:u32;
fn umTileCoord(t:u32)->vec3u{return vec3u(t%3u,(t/3u)%3u,t/9u);}
fn umTileKey(p:vec3i)->u32{if(any(p<vec3i(0))||any(p>vec3i(2))){return 27u;}return u32(p.x+3*p.y+9*p.z);}
fn umKeyStored(t:u32)->bool{return t<27u&&!((testCase==8u||testCase>=16u)&&t==4u);}
fn umTileKind(p:vec3i)->u32{return select(1u,UM_SOLID,testCase>=16u);}
${uniformSolidRefinementWGSL(0)}
@compute @workgroup_size(1) fn main(){
 for(var c=0u;c<20u;c++){
  testCase=c;for(var t=0u;t<2300u;t++){solidRecord[t]=vec4f(0);}
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
  if(c>=10u&&c<16u){let slots=array<u32,6>(0u,15u,16u,31u,32u,47u);let slot=slots[c-10u];let word=slot/6u;
   solidRecord[27u+39u+1u+word/4u][word%4u]-=f32(8u<<(4u*(slot%6u)));}
  if(c>=16u){let row=23u*27u+1u+61u*13u+49u+8u;
   for(var i=0u;i<4u;i++){solidRecord[row+i]=vec4f(.5);}
   if(c==17u){solidRecord[row+3u].w=1.0;}
   if(c==18u){for(var i=0u;i<4u;i++){solidRecord[row+i]=vec4f(.25);}}
   if(c==19u){solidRecord[20u*27u+2u*13u+1u].z=1.0;}
  }
  result[c]=u32(umSolidGeometryExact(13u));
 }
}`});
   assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==="error").map(m=>m.message),[]);
   const pipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"main"}}),group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:output}}]});
   const e=device.createCommandEncoder(),p=e.beginComputePass();p.setPipeline(pipeline);p.setBindGroup(0,group);p.dispatchWorkgroups(1);p.end();device.queue.submit([e.finish()]);
   assert.deepEqual(Array.from(new Uint32Array((await readMixedBuffer(device,output)).buffer)),[1,0,0,0,0,0,1,1,0,1,0,0,0,0,0,0,1,0,0,0]);
  }finally{output.destroy();}
 });
});

test('impact refinement distinguishes an open wall collar from an obstacle on both sides',{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 const {readFileSync}=await import('node:fs');
 const source=readFileSync(new URL('../lib/methods/uniform/uniform-mixed-dynamic.ts',import.meta.url),'utf8');
 const obstacle=source.slice(source.indexOf('fn umObstacleAt('),source.indexOf('${uniformSolidRefinementWGSL'));
 const boundary=source.slice(source.indexOf('struct Boundary{'),source.indexOf('// Thin features:'));
 await withUniformDevice('symmetric impact classification',async device=>{
  const output=device.createBuffer({size:16,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
  try{
   const module=device.createShaderModule({code:`
@group(0) @binding(0) var<storage,read_write> output:array<u32>;
var<private> solidRecord:array<vec4f,512>;
struct Policy{surface:vec4f,reach:vec4u};var<private> policy:Policy;
struct TileClass{travel:array<f32,6>,reach:array<u32,6>};
const CAP=16u;
fn umTileKey(p:vec3i)->u32{return u32(p.x+8*p.y+64*p.z);}
fn umKeyStored(t:u32)->bool{return t<512u;}
// All these tiles are coupled to a wall: the old obstacle predicate
// therefore mistakes the whole open collar for a barrier.
fn umSolidAt(q:vec3i)->bool{return true;}
fn umDirectionTravel(c:TileClass,k:u32)->f32{return c.travel[k];}
fn scoreFloor()->f32{return .25;}
fn recordReasons()->bool{return false;}
fn umWallBeyond(tile:u32,k:u32)->u32{
 let p=vec3u(tile%8u,(tile/8u)%8u,tile/64u);
 return select(65535u,p[k%3u],k<3u);
}
${obstacle}
${boundary}
@compute @workgroup_size(1) fn main(){
 policy.surface.y=.1;policy.reach.w=0u;
 for(var t=0u;t<512u;t++){let p=vec3u(t%8u,(t/8u)%8u,t/64u);solidRecord[t].x=f32(all(p<vec3u(4)));}
 for(var side=0u;side<2u;side++){
  let p=vec3u(1,1,3u*side);let tile=umTileKey(vec3i(p));var c:TileClass;
  c.travel[3]=.2;output[2u*side]=u32(umBoundary(tile,p,4u,c).triggered);
  c.travel[3]=0.0;c.travel[select(2u,5u,side==1u)]=.2;
  output[2u*side+1u]=u32(umBoundary(tile,p,4u,c).triggered);
 }
}`});
   assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==='error').map(m=>m.message),[]);
   const pipeline=await device.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint:'main'}});
   const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:output}}]});
   const e=device.createCommandEncoder(),p=e.beginComputePass();p.setPipeline(pipeline);p.setBindGroup(0,group);p.dispatchWorkgroups(1);p.end();device.queue.submit([e.finish()]);
   assert.deepEqual(Array.from(new Uint32Array((await readMixedBuffer(device,output)).buffer)),[0,1,0,1]);
  }finally{output.destroy();}
 });
});
