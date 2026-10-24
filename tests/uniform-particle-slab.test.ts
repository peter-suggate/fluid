import assert from "node:assert/strict";
import test from "node:test";
import { particleSlabWGSL, particleSlabWords } from "../lib/methods/uniform/uniform-particle-slab.wgsl";
import { withUniformDevice } from "./helpers/uniform-geometric";

test("particle slabs conserve overflow and seeds beyond the reserved tail across moves", {skip:!process.env.WEBGPU_NODE_MODULE}, async()=>{
 await withUniformDevice("particle slab overflow",async device=>{
  const capacity=8192,headroom=512,cells=32768;
  const module=device.createShaderModule({code:`
struct Particle {position:vec4f,velocity:vec4f,before:vec4f}
@group(0) @binding(0) var<storage,read_write> state:array<atomic<u32>>;
@group(0) @binding(1) var<storage,read_write> source:array<Particle>;
@group(0) @binding(2) var<storage,read_write> particles:array<Particle>;
@group(0) @binding(3) var<storage,read_write> links:array<u32>;
@group(0) @binding(4) var<storage,read_write> bins:array<atomic<u32>>;
const UB_BRICKS:u32=64u;const UB_NONE:u32=0xffffffffu;const UB_PARTICLES_CORE:u32=1u;
const UM_CELL_KEYS:u32=${cells}u;const UM_TILES:u32=512u;const NB_LINKS:u32=UM_CELL_KEYS;
const NB_OCCUPIED:u32=32768u;const NB_COVERAGE:u32=33300u;
const NB_SURFACE_TILES:u32=33812u;const NB_ACTIVITY_SURFACE_HEAT:u32=34324u;const NB_ACTIVITY_THETA:u32=34836u;
fn ubCount()->u32{return 3u;}
fn ubBrick(slot:u32)->u32{return slot;}
fn ubBrickAt(c:vec3i)->u32{return select(UB_NONE,u32(c.x/8),c.x>=0&&c.x<32);}
fn ubBrickOrigin(id:u32)->vec3u{return vec3u(8u*id,0u,0u);}
fn ubMark(id:u32,kind:u32){atomicOr(&state[24u],1u<<id);}
fn umFramePage()->vec3i{return vec3i(0);}
fn umCorner(k:u32,n:u32)->vec3u{return vec3u(k%n,(k/n)%n,k/(n*n));}
fn umTileLocal(c:vec3u)->u32{return c.x+8u*c.y+64u*c.z;}
fn umTileKey(c:vec3i)->u32{return 0u;}
fn umKeyStored(key:u32)->bool{return true;}
fn nbUnmarked(c:vec3i){}
fn nbCover(tile:u32,n:u32){atomicStore(&bins[NB_COVERAGE+tile],n);}
fn markSurfaceTiles(q:vec3f,heat:f32){}
${particleSlabWGSL(capacity-headroom,headroom)}
fn sample(i:u32,brick:u32)->Particle{
 var p=Particle();p.position=vec4f(vec3f(f32(8u*brick),0.0,0.0)+vec3f(umCorner(i%512u,8u))+0.25,1.0);
 p.velocity=vec4f(f32(i),2.0,3.0,4.0);p.before=vec4f(5.0,6.0,7.0,8.0);return p;
}
@compute @workgroup_size(64) fn moveFront(@builtin(global_invocation_id) gid:vec3u){
 if(gid.x<1024u){nbMoveAppend(sample(gid.x,1u));}
}
@compute @workgroup_size(64) fn seedFront(@builtin(global_invocation_id) gid:vec3u){
 if(gid.x<1200u){_=nbSeedAppend(sample(1024u+gid.x,2u),true);}
 // A brick absent from the table also survives through the current tail.
 if(gid.x==1200u){_=nbSeedAppend(sample(2224u,3u),true);}
}
@compute @workgroup_size(64) fn moveAgain(@builtin(global_invocation_id) gid:vec3u){
 let i=gid.x;
 for(var e=0u;e<atomicLoad(&state[19u]);e++){
  let at=NB_SLAB_IN+4u*e;
  if(i<atomicLoad(&state[at+2u])){nbMoveAppend(particles[atomicLoad(&state[at+1u])+i]);}
 }
 if(i<nbSpillCount(false)){nbMoveAppend(particles[nbSpillBase(false)+i]);}
}
@compute @workgroup_size(64) fn fillTail(@builtin(global_invocation_id) gid:vec3u){
 if(gid.x<1200u){_=nbSpill(sample(gid.x,3u),false,false);}
}
@compute @workgroup_size(1) fn mandatoryOverflow(){_=nbSpill(sample(0u,3u),false,true);}
`});
  assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==="error").map(m=>m.message),[]);
  const layout=device.createBindGroupLayout({entries:Array.from({length:5},(_,binding)=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage" as const}}))});
  const pipelineLayout=device.createPipelineLayout({bindGroupLayouts:[layout]});
  const names=["slabAllocate","moveFront","slabPlace","slabOverflow","slabSort","seedFront","slabMark","moveAgain","fillTail","mandatoryOverflow"];
  const pipelines=new Map(await Promise.all(names.map(async entryPoint=>[entryPoint,await device.createComputePipelineAsync({layout:pipelineLayout,compute:{module,entryPoint}})] as const)));
  const buffers=[4*particleSlabWords(1),48*capacity,48*capacity,4*(cells+7*capacity),4*36000].map(size=>device.createBuffer({size,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST}));
  const group=device.createBindGroup({layout,entries:buffers.map((buffer,binding)=>({binding,resource:{buffer}}))});
  const read=device.createBuffer({size:Math.max(...buffers.map(b=>b.size)),usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  const run=(entries:[string,number][])=>{
   const e=device.createCommandEncoder(),pass=e.beginComputePass();pass.setBindGroup(0,group);
   for(const [name,groups] of entries){pass.setPipeline(pipelines.get(name)!);pass.dispatchWorkgroups(groups);}
   pass.end();device.queue.submit([e.finish()]);
  };
  const words=async(index:number)=>{
   const b=buffers[index]!,e=device.createCommandEncoder();e.copyBufferToBuffer(b,0,read,0,b.size);device.queue.submit([e.finish()]);
   await read.mapAsync(GPUMapMode.READ);const result=new Uint32Array(read.getMappedRange().slice(0,b.size));read.unmap();return result;
  };
  const verify=async(count:number)=>{
   const receipt=await words(0);assert.equal(receipt[3],0,"no allocator fatal");
   const data=await words(2),f=new Float32Array(data.buffer),ids:number[]=[];
   const inspect=(i:number)=>{
    if(f[12*i+3]!<=0)return;
    const id=f[12*i+4]!,brick=id<1024?1:id<2224?2:3,cell=id%512;
    ids.push(id);assert.deepEqual([...f.slice(12*i,12*i+4)],[8*brick+cell%8+0.25,Math.floor(cell/8)%8+0.25,Math.floor(cell/64)+0.25,1]);
    assert.deepEqual([...f.slice(12*i+5,12*i+12)],[2,3,4,5,6,7,8]);
   };
   for(let i=0;i<receipt[0]!;i++)inspect(i);
   for(let k=0;k<receipt[21]!;k++)inspect(receipt[0]!+k);
   assert.deepEqual(ids.sort((a,b)=>a-b),Array.from({length:count},(_,i)=>i),"every full particle payload survives exactly once");
   return receipt;
  };
  try{
   // Previous demand lives in brick 0; this move sends it all to an empty brick.
   device.queue.writeBuffer(buffers[0]!,4*(32+10),new Uint32Array([1024]));
   run([["slabAllocate",1],["moveFront",16],["slabPlace",1],["slabOverflow",32],["slabSort",3]]);
   let receipt=await verify(1024);assert.ok(receipt[16]!>headroom,"move exceeds the old fixed reserve");assert.equal(receipt[1],1024);
   run([["seedFront",19]]);receipt=await verify(2225);assert.ok(receipt[21]!>headroom,"seeds exceed the old fixed reserve");
   run([["slabMark",32],["slabAllocate",1],["moveAgain",32],["slabPlace",1],["slabOverflow",32],["slabSort",3]]);
   receipt=await verify(2225);assert.equal(receipt[1],2225);assert.equal(receipt[21],1,"unlisted brick survives in the tail");assert.equal(receipt[24]!&15,14,"spilled seeds mark their destination bricks");
   // Genuine exhaustion remains bounded: optional reseeds clip; mandatory samples fail loudly.
   const reset=new Uint32Array(particleSlabWords(1));reset[0]=capacity-headroom;device.queue.writeBuffer(buffers[0]!,0,reset);
   run([["fillTail",19]]);receipt=await words(0);assert.equal(receipt[3],0);assert.equal(receipt[2],1200-headroom);
   run([["mandatoryOverflow",1]]);receipt=await words(0);assert.equal(receipt[3],5);
  }finally{read.destroy();for(const b of buffers)b.destroy();}
 });
});
