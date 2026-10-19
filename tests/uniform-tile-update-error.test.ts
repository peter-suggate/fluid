import assert from "node:assert/strict";
import test from "node:test";
import {advanceTierHold,uniformTierHoldWGSL,uniformTileFineDefectWGSL} from "../lib/methods/uniform/uniform-tile-update-error";
import {narrowBandActivityTriggerWGSL} from "../lib/methods/uniform/uniform-narrow-band-activity.wgsl";
import {withUniformDevice} from "./helpers/uniform-geometric";
import {readMixedBuffer} from "./helpers/uniform-mixed-native-fields";

// Explicit expected lifetime, rather than comparing two copies of the algorithm.
test("weaker motion cannot extend a hot hold; both tier deadlines remain independent",()=>{
 let packed=advanceTierHold(0,true,false,2).packed;
 const hot=[];
 for(let frame=0;frame<10;frame++){const next=advanceTierHold(packed,false,true,2);hot.push(next.hot);packed=next.packed;}
 assert.deepEqual(hot,[true,true,false,false,false,false,false,false,false,false]);
 assert.equal(packed&65535,0);assert.equal(packed>>>16,2);
 const first=advanceTierHold(packed,true,false,2);
 assert.equal(first.warm,true);assert.equal(first.packed&65535,2);
 assert.deepEqual(advanceTierHold(0,true,true,0),{packed:0,hot:false,warm:false});
});

const gpu={skip:!process.env.WEBGPU_NODE_MODULE};
test("GPU heat holds expire after two frames under continuous WARM triggers",gpu,async()=>{
 await withUniformDevice("independent heat deadlines",async device=>{
  const state=device.createBuffer({size:36,usage:GPUBufferUsage.STORAGE}),output=device.createBuffer({size:40,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
  try{
   const module=device.createShaderModule({code:`
@group(0) @binding(0) var<storage,read_write> state:array<u32>;
@group(0) @binding(1) var<storage,read_write> output:array<u32>;
const RH_HOLD_FRAMES=2u;const RH_HELD_HOT=0x26u;const RH_HELD_WARM=0x1c0u;
@compute @workgroup_size(1) fn main(){let at=0u;
 for(var frame=0u;frame<10u;frame++){var trigger=select(64u,2u,frame==0u);{${uniformTierHoldWGSL}}output[frame]=u32((trigger&0x227u)!=0u);}}
`});
   const pipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"main"}}),group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[state,output].map((buffer,binding)=>({binding,resource:{buffer}}))});
   const e=device.createCommandEncoder(),p=e.beginComputePass();p.setPipeline(pipeline);p.setBindGroup(0,group);p.dispatchWorkgroups(1);p.end();device.queue.submit([e.finish()]);
   assert.deepEqual(Array.from(new Uint32Array((await readMixedBuffer(device,output)).buffer)),[1,1,1,0,0,0,0,0,0,0]);
  }finally{state.destroy();output.destroy();}
 });
});

test("tile reconstruction error ignores affine motion but detects unresolved curvature",gpu,async()=>{
 await withUniformDevice("tile error estimator",async device=>{
  const data=new Float32Array(4*64*3);
  for(let group=0;group<3;group++)for(let i=0;i<64;i++){
   const x=i%4,y=(i>>2)%4,z=i>>4;
   data.set([4+.7*x-.2*y,.5+.3*z,group===2?.5*x*x:group===1?3*x-2*y+z:2],4*(64*group+i));
  }
  const input=device.createBuffer({size:data.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),output=device.createBuffer({size:3*64*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
  try{
   device.queue.writeBuffer(input,0,data);
   const module=device.createShaderModule({code:`
@group(0) @binding(0) var<storage,read> samples:array<vec4f>;
@group(0) @binding(1) var<storage,read_write> output:array<f32>;
var<workgroup> planValues:array<vec3f,64>;
fn umCorner(k:u32,n:u32)->vec3u{return vec3u(k%n,(k/n)%n,k/(n*n));}
${uniformTileFineDefectWGSL}
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3u,@builtin(local_invocation_index) lane:u32){
 planValues[lane]=samples[id.x].xyz;workgroupBarrier();output[id.x]=tuFineDefect(lane);}
`});
   const pipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"main"}}),group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[input,output].map((buffer,binding)=>({binding,resource:{buffer}}))});
   const e=device.createCommandEncoder(),p=e.beginComputePass();p.setPipeline(pipeline);p.setBindGroup(0,group);p.dispatchWorkgroups(3);p.end();device.queue.submit([e.finish()]);const result=await readMixedBuffer(device,output);
   assert.ok(Math.max(...result.slice(0,128))<3e-6,"translation/shear are representable even with a large velocity range");
   for(let i=0;i<64;i++)assert.ok(Math.abs(result[128+i]!-(i%4===1||i%4===2?1:0))<3e-6,"quadratic midpoint defect is one m/s");
  }finally{input.destroy();output.destroy();}
 });
});

test("particle updates distinguish stationary shape from evolving detail",gpu,async()=>{
 await withUniformDevice("shape versus activity",async device=>{
  const output=device.createBuffer({size:32,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
  try{
   const module=device.createShaderModule({code:`
@group(0) @binding(0) var<storage,read_write> output:array<u32>;
struct Params{hDt:vec4f} const params=Params(vec4f(.05,.05,.05,1.0/60.0));
fn tuSpeed(t:u32)->f32{return select(0.0,1.0,t>=4u);}
fn umBodyCount()->u32{return 1u;}fn umBodyPoseNearTile(t:u32,m:u32)->bool{return true;}
${narrowBandActivityTriggerWGSL("need")}
@compute @workgroup_size(1) fn main(){
 for(var i=0u;i<8u;i++){let criterion=array<u32,4>(1u,2u,4u,16u)[i%4u];output[i]=u32(nbActivityTriggered(i,criterion));}}
`});
   const pipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"main"}}),group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:output}}]});
   const e=device.createCommandEncoder(),p=e.beginComputePass();p.setPipeline(pipeline);p.setBindGroup(0,group);p.dispatchWorkgroups(1);p.end();device.queue.submit([e.finish()]);
   assert.deepEqual(Array.from(new Uint32Array((await readMixedBuffer(device,output)).buffer)),[0,0,1,1,1,1,1,1]);
  }finally{output.destroy();}
 });
});

// These are the production shader helpers, with analytic physical cases.
test("freeze error ignores tangential drift, pressure gauge, and unread solid/air; retains normal transport and pressure variation",gpu,async()=>{
 const {uniformTileSurfaceRateWGSL}=await import("../lib/methods/uniform/uniform-tile-update-error");
 const {uniformColdProjectionDefectWGSL}=await import("../lib/methods/uniform/uniform-cold-freeze");
 const {uniformPressureGradientChangeWGSL}=await import("../lib/methods/uniform/uniform-mixed-region-dynamics");
 await withUniformDevice("physical freeze error",async device=>{
  const output=device.createBuffer({size:64,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
  try{
   const module=device.createShaderModule({code:`
@group(0) @binding(0) var<storage,read_write> result:array<f32>;
${uniformTileSurfaceRateWGSL}
${uniformColdProjectionDefectWGSL}
${uniformPressureGradientChangeWGSL}
@compute @workgroup_size(1) fn main(){
 var plane:array<f32,8>;var air:array<f32,8>;var bulk:array<f32,8>;
 for(var k=0u;k<8u;k++){plane[k]=.05*(f32((k>>1u)&1u)-.5);air[k]=plane[k]+2.0;bulk[k]=plane[k]-2.0;}
 result[0]=tuSurfaceRate(plane,vec3f(4,0,2),.05,1u,true).x;
 result[1]=tuSurfaceRate(plane,vec3f(4,.3,2),.05,1u,true).x;
 result[2]=tuSurfaceRate(plane,vec3f(4,.3,2),.05,1u,false).y;
 result[3]=tuSurfaceRate(air,vec3f(4,.3,2),.05,1u,true).y;
 let deep=tuSurfaceRate(bulk,vec3f(4,.3,2),.05,1u,true);result[4]=deep.x;result[5]=deep.y;
 result[6]=umPressureGradientChange(1000.0,1000.0);
 result[7]=umPressureGradientChange(1000.0,1012.0);
 let constant=umColdProjectionDefect(vec3f(.1),vec3f(.1),vec3f(.1));result[8]=dot(constant,vec3f(1.0));
 let variation=umColdProjectionDefect(vec3f(-.2,0,.1),vec3f(.4,0,.1),vec3f(.1,0,0));result[9]=variation.x;result[10]=variation.z;
 let absent=umColdProjectionDefect(vec3f(3e38),vec3f(-3e38),vec3f(0));result[11]=dot(absent,vec3f(1.0));

}`});
   assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==="error"),[]);
   const pipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"main"}}),group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:output}}]});
   const e=device.createCommandEncoder(),p=e.beginComputePass();p.setPipeline(pipeline);p.setBindGroup(0,group);p.dispatchWorkgroups(1);p.end();device.queue.submit([e.finish()]);
   const values=await readMixedBuffer(device,output),expected=[0,.3,0,0,0,1,0,3,0,.3,.1,0];
   for(let i=0;i<expected.length;i++)assert.ok(Math.abs(values[i]!-expected[i]!)<1e-6,`case ${i}: ${values[i]} vs ${expected[i]}`);
  }finally{output.destroy();}
 });
});

test("submerged freeze budget requires depth evidence and charges only omitted pressure response",gpu,async()=>{
 const {uniformTileSurfaceRateWGSL}=await import("../lib/methods/uniform/uniform-tile-update-error");
 const {uniformRegionFreezeBudgetWGSL}=await import("../lib/methods/uniform/uniform-region-heat");
 await withUniformDevice("submerged freeze error",async device=>{
  const output=device.createBuffer({size:72,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
  try{
   const module=device.createShaderModule({code:`
@group(0) @binding(0) var<storage,read_write> result:array<f32>;
${uniformTileSurfaceRateWGSL}
${uniformRegionFreezeBudgetWGSL}
@compute @workgroup_size(1) fn main(){
 var bulk:array<f32,8>;for(var k=0u;k<8u;k++){bulk[k]=-2.0;}
 result[0]=tuReaderClass(bulk,.05,1.0/60.0,0.0,true);
 result[1]=tuReaderClass(bulk,.05,1.0/60.0,40.0,true);
 result[2]=tuReaderClass(bulk,.05,1.0/60.0,0.0,false);
 bulk[7]=-.1;result[3]=tuReaderClass(bulk,.05,1.0/60.0,0.0,true);
 bulk[7]=0.0;result[4]=tuReaderClass(bulk,.05,1.0/60.0,0.0,true);
 bulk[7]=1.0;result[5]=tuReaderClass(bulk,.05,1.0/60.0,0.0,true);
 bulk[7]=3.402823e38;result[6]=tuReaderClass(bulk,.05,1.0/60.0,0.0,true);
 result[7]=rhFreezeBudget(true);result[8]=rhFreezeBudget(false);
 result[9]=rhFrozenErrorSpeed(true,2.0,.03);result[10]=rhFrozenErrorSpeed(false,2.0,.03);
 result[11]=f32(.01<rhFreezeBudget(true)&&.01>rhFreezeBudget(false));
 result[12]=f32(rhHotHaloRequired(true,false));result[13]=f32(rhHotHaloRequired(true,true));
 result[14]=f32(rhHotHaloRequired(false,false));result[15]=f32(rhHotHaloRequired(false,true));
 result[16]=rhBulkAcceleration(.1,0.0,2.0);result[17]=rhBulkAcceleration(0.0,.3,2.0);
}`});
   assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==="error").map(m=>m.message),[]);
   const pipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"main"}});
   const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:output}}]});
   const e=device.createCommandEncoder(),p=e.beginComputePass();p.setPipeline(pipeline);p.setBindGroup(0,group);p.dispatchWorkgroups(1);p.end();device.queue.submit([e.finish()]);
   const values=await readMixedBuffer(device,output),expected=[1,2,0,2,2,2,2,.08,.002,.03,2.03,1,1,0,0,0,.1,.6];
   for(let i=0;i<expected.length;i++)assert.ok(Math.abs(values[i]!-expected[i]!)<1e-6,`case ${i}: ${values[i]} vs ${expected[i]}`);
  }finally{output.destroy();}
 });
});
