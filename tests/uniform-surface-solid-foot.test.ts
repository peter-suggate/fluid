import {uniformSolidSlideStartWGSL} from "../lib/methods/uniform/uniform-solid-point-open.wgsl";
import assert from "node:assert/strict";
import test from "node:test";
import {readFileSync} from "node:fs";
import {uniformSurfaceSmoothingWGSL} from "../lib/methods/uniform/uniform-surface-smoothing.wgsl";
import {withUniformDevice} from "./helpers/uniform-geometric";

test("fine and coarse surface feet keep tangential slopes at both aligned walls",{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 await withUniformDevice("wall surface foot",async device=>{
  const source=readFileSync(new URL("../lib/methods/uniform/uniform-mixed-surface.ts",import.meta.url),"utf8");
  const raw=source.slice(source.indexOf("struct UMFoot {"),source.indexOf("// The departure of a vertex from its foot:"));
  const foot=Function("UNIFORM_MIXED_VERTEX_SMOOTHING_PER_S","return `"+raw+"`;").call({solid:{coarse:{}},workControl:{}},.5) as string;
  const contactRaw=source.slice(source.indexOf("// A vertex on a page wall"),source.indexOf("// The six wall reach words"));
  const contact=Function("return `"+contactRaw+"`;").call({solid:true}) as string;
  const output=device.createBuffer({size:4*16,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
  const module=device.createShaderModule({code:`
struct Params {hDt:vec4f};var<private> params:Params;
@group(0) @binding(0) var<storage,read_write> output:array<vec4f>;
fn umPointCell(p:vec3f)->vec3u{return vec3u(floor(p));}
// Low wall is a page boundary; high wall is embedded padding.
fn umPointFromPage(page:vec3u,p:vec3f)->vec3f{return max(p,vec3f(0));}
fn umPointFrom(p:vec3f,q:vec3f)->vec3f{return max(q,vec3f(0));}
fn umCanonicalSupportAt(p:vec3f,reach:f32)->bool{return false;}
fn umSolidEnabled()->bool{return true;}
fn umSampleClosed(p:vec3f)->bool{return p.z>32.0||p.z<0.0;}
fn umOpenAt(p:vec3f)->bool{return !umSampleClosed(p);}
fn umSmoothOpen(p:vec3i,w:i32)->bool{return umOpenAt(vec3f(p));}
fn umLatticeLoad(p:vec3i,w:u32)->f32{
 if(p.z>32){return 16.0;}
 return .25+.2*(f32(p.x)-16.0)+.3*(f32(p.y)-16.0);
}
fn umLoadCorner(p:vec3u)->f32{return umLatticeLoad(vec3i(p),4u);}
fn umLoadVertex(p:vec3u)->f32{return umLatticeLoad(vec3i(p),1u);}
fn umWallWithin(p:vec3i,axis:u32,reach:i32)->bool{return false;}
${uniformSurfaceSmoothingWGSL}
${uniformSolidSlideStartWGSL}
${foot}
const umFastSurface=false;const UM_WALL_CLIP=true;
var<private> umLastTravel:vec3f;
fn umWallBlock(p:vec3i,k:u32)->u32{return select(0u,1u,k==4u&&p.z==0);}
fn umSampleVelocity(p:vec3f)->vec3f{return vec3f(0,0,.01*(p.z-16.0));}
fn umFastVelocity(p:vec3f)->vec3f{return umSampleVelocity(p);}
struct Walk {q:vec3f}
fn umWalk(p:vec3f,q:vec3f)->Walk{return Walk(q);}
fn umTrace(p:vec3f)->vec3f{return p;}
fn umSampleVertex(p:vec3f)->f32{return -.25+.001*(p.z-16.0)*(p.z-16.0);}
${contact}
@compute @workgroup_size(4) fn main(@builtin(local_invocation_index) i:u32){
 params.hDt=vec4f(1,1,1,0);
 let p=vec3u(16,16,select(0u,32u,(i&1u)!=0u));
 let result=umVertexFoot(p,select(1u,4u,i>=2u));
 params.hDt.w=1.0/60.0;
 output[i]=vec4f(result.foot,umWallContact(vec3f(p),.5,select(1u,4u,i>=2u)));
}`});
  assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==="error").map(m=>m.message),[]);
  const pipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"main"}});
  const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:output}}]});
  const read=device.createBuffer({size:output.size,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  try{
   const e=device.createCommandEncoder(),p=e.beginComputePass();p.setPipeline(pipeline);p.setBindGroup(0,group);p.dispatchWorkgroups(1);p.end();e.copyBufferToBuffer(output,0,read,0,output.size);device.queue.submit([e.finish()]);
   await read.mapAsync(GPUMapMode.READ);const values=new Float32Array(read.getMappedRange());
   for(let i=0;i<4;i++){
    assert.ok(Math.abs(values[4*i]!-(16-.25*.2/.13))<1e-5,`case ${i}: x foot`);
    assert.ok(Math.abs(values[4*i+1]!-(16-.25*.3/.13))<1e-5,`case ${i}: y foot`);
    assert.equal(values[4*i+2],i%2?32:0,"normal foot stays on the wall");
    const inside=16-(i>=2?4:1);
    assert.ok(Math.abs(values[4*i+3]!-(-.25+.001*inside*inside))<1e-6,"contact continuation matches at either wall");
   }
  }finally{if(read.mapState==="mapped")read.unmap();read.destroy();output.destroy();}
 });
});

// Actual source operator comparison without duplicated foot or smoothing code.
test("exact regional foot clearance preserves curved moving and cut-solid feet",{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 await withUniformDevice("exact solid foot",async device=>{
  const source=readFileSync(new URL("../lib/methods/uniform/uniform-mixed-surface.ts",import.meta.url),"utf8");
  const raw=source.slice(source.indexOf("struct UMFoot {"),source.indexOf("// The departure of a vertex from its foot:"));
  const foot=Function("UNIFORM_MIXED_VERTEX_SMOOTHING_PER_S","return `"+raw+"`;").call({solid:{coarse:{}},workControl:{}},.5) as string;
  const output=device.createBuffer({size:128*12*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
  const module=device.createShaderModule({code:`
struct Params {hDt:vec4f};var<private> params:Params;
@group(0) @binding(0) var<storage,read_write> output:array<u32>;
var<private> mode:u32;var<private> moving:f32;var<private> calls:u32;
var<private> umSlow:bool;var<private> umHold:bool;
fn umPointCell(p:vec3f)->vec3u{return vec3u(floor(p));}
fn umPointFromPage(page:vec3u,p:vec3f)->vec3f{umHold=umHold||p.x<1.0;return p;}
fn umPointFrom(p:vec3f,q:vec3f)->vec3f{umSlow=umSlow||q.y<1.0;return q;}
fn umCanonicalSupportAt(p:vec3f,reach:f32)->bool{return mode==0u;}
fn umSolidEnabled()->bool{return true;}
fn umSampleClosed(p:vec3f)->bool{calls++;return mode==2u&&p.x>32.0;}
fn umOpenAt(p:vec3f)->bool{calls++;return !(mode==2u&&p.x>32.0);}
fn umSmoothOpen(p:vec3i,w:i32)->bool{calls++;return !(mode==2u&&p.x>32);}
fn umLatticeLoad(p:vec3i,w:u32)->f32{let q=vec3f(p)-vec3f(32);return .05*q.y+.003*q.x*q.x-.002*q.z*q.z+moving;}
fn umLoadCorner(p:vec3u)->f32{return umLatticeLoad(vec3i(p),4u);}
fn umLoadVertex(p:vec3u)->f32{return umLatticeLoad(vec3i(p),1u);}
fn umWallWithin(p:vec3i,axis:u32,reach:i32)->bool{return false;}
${uniformSurfaceSmoothingWGSL}
${uniformSolidSlideStartWGSL}
${foot}
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3u){
 let i=id.x;if(i>=128u){return;}mode=(i/2u)%3u;moving=(f32(i%17u)-8.0)*.013;
 params.hDt=vec4f(1,1,1,1.0/60.0);let w=select(1u,4u,(i&1u)!=0u);
 let result=umVertexFoot(vec3u(32),w);let a=12u*i;
 output[a]=bitcast<u32>(result.foot.x);output[a+1u]=bitcast<u32>(result.foot.y);output[a+2u]=bitcast<u32>(result.foot.z);
 output[a+3u]=bitcast<u32>(result.shift.x);output[a+4u]=bitcast<u32>(result.shift.y);output[a+5u]=bitcast<u32>(result.shift.z);
 output[a+6u]=bitcast<u32>(result.smoothing);output[a+7u]=bitcast<u32>(result.center);
 output[a+8u]=u32(umSlow);output[a+9u]=u32(umHold);output[a+10u]=calls;
}`});
  assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==="error").map(m=>m.message),[]);
  const run=async(enabled:boolean)=>{
   const pipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"main",constants:{umExactSolidSupport:Number(enabled)}}});
   const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:output}}]});
   const read=device.createBuffer({size:output.size,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
   try{const e=device.createCommandEncoder(),p=e.beginComputePass();p.setPipeline(pipeline);p.setBindGroup(0,group);p.dispatchWorkgroups(2);p.end();e.copyBufferToBuffer(output,0,read,0,output.size);device.queue.submit([e.finish()]);await read.mapAsync(GPUMapMode.READ);return new Uint32Array(read.getMappedRange()).slice();}finally{if(read.mapState==="mapped")read.unmap();read.destroy();}
  };
  try{const reference=await run(false),candidate=await run(true);for(let i=0;i<128;i++){
   assert.deepEqual(candidate.slice(12*i,12*i+10),reference.slice(12*i,12*i+10),`case ${i}: foot, shift, smoothing and sampler flags`);
   if((i/2|0)%3===0)assert.equal(candidate[12*i+10],0,"clear support removes fine predicates");
   else assert.equal(candidate[12*i+10],reference[12*i+10],"near/cut support retains original predicates");
  }}finally{output.destroy();}
 });
});

test("actual regional helper rejects tagged pages and boxes outside its 27-region window",{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 await withUniformDevice("exact solid support helper closure",async device=>{
  const source=readFileSync(new URL("../lib/methods/uniform/uniform-mixed-surface.ts",import.meta.url),"utf8");
  const take=(start:string,end:string)=>source.slice(source.indexOf(start),source.indexOf(end,source.indexOf(start)));
  const helper=take("fn umCanonicalSupportAt(","fn umRequiredLipschitz(");
  const index=take("fn umSupportIndex(","fn umSupportMaskIndex(");
  const rendered=Function("UNIFORM_SURFACE_SUPPORT_WORDS","return `"+helper+"`;").call({supportWord:0},11) as string;
  const lattice=readFileSync(new URL("../lib/methods/uniform/uniform-lattice-address.ts",import.meta.url),"utf8");
  const blockRaw=lattice.split("\n").find(line=>line.startsWith("fn umBlockStored("))!;
  const stored=blockRaw.replace("${blocks}","4");
  const cases=[
   {slot:1,p:8,reach:15,expected:1},{slot:2,p:8,reach:15,expected:1},
   {slot:0,p:8,reach:15,expected:0},{slot:3,p:8,reach:15,expected:0},
   {slot:0xffffffff,p:8,reach:15,expected:0},{slot:0x80000001,p:8,reach:15,expected:0},
   {slot:0x40000001,p:8,reach:15,expected:0},
   {slot:1,p:16,reach:15,expected:1},{slot:1,p:16,reach:16,expected:0},
   {slot:1,p:0,reach:16,expected:1},{slot:1,p:0,reach:17,expected:0},
   {slot:1,p:8,reach:24,expected:0},{slot:1,p:48,reach:1,expected:0},
   {slot:1,p:-17,reach:1,expected:0},
  ];
  const input=device.createBuffer({size:16*cases.length,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
  const payload=new Uint32Array(4*cases.length);cases.forEach((c,i)=>{payload[4*i]=c.slot;const f=new Float32Array([c.p,c.reach]);payload.set(new Uint32Array(f.buffer),4*i+1);});device.queue.writeBuffer(input,0,payload);
  const claims=device.createBuffer({size:4*11*64*24,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});device.queue.writeBuffer(claims,0,new Uint32Array(claims.size/4).fill(3));
  const output=device.createBuffer({size:4*cases.length,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
  const module=device.createShaderModule({code:`
@group(0) @binding(0) var<storage,read> input:array<vec4u>;
@group(0) @binding(1) var<storage,read> umSurfaceEvidence:array<u32>;
@group(0) @binding(2) var<storage,read_write> output:array<u32>;
var<private> slotWord:u32;
fn umVertexSlot(p:vec3i)->vec4u{return vec4u(0,0,0,slotWord);}
fn umPageCoord(region:u32)->vec3u{return vec3u(0);}
${stored}
${index}
${rendered}
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3u){
 if(id.x>=${cases.length}u){return;}let row=input[id.x];slotWord=row.x;
 output[id.x]=u32(umCanonicalSupportAt(vec3f(bitcast<f32>(row.y)),bitcast<f32>(row.z)));
}`});
  assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==="error").map(m=>m.message),[]);
  const pipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"main"}});
  const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[input,claims,output].map((buffer,binding)=>({binding,resource:{buffer}}))});
  const read=device.createBuffer({size:output.size,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  try{const e=device.createCommandEncoder(),p=e.beginComputePass();p.setPipeline(pipeline);p.setBindGroup(0,group);p.dispatchWorkgroups(1);p.end();e.copyBufferToBuffer(output,0,read,0,output.size);device.queue.submit([e.finish()]);await read.mapAsync(GPUMapMode.READ);const values=new Uint32Array(read.getMappedRange());cases.forEach((c,i)=>assert.equal(values[i],c.expected,JSON.stringify(c)));}
  finally{if(read.mapState==="mapped")read.unmap();read.destroy();input.destroy();claims.destroy();output.destroy();}
 });
});
