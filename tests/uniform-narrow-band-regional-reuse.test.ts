import assert from "node:assert/strict";
import test from "node:test";
import {narrowBandNearestWGSL,narrowBandNearestControlEntries,narrowBandNearestEntries} from "../lib/methods/uniform/uniform-narrow-band-nearest.wgsl";
import {narrowBandMembershipWGSL} from "../lib/methods/uniform/uniform-narrow-band-membership.wgsl";
import {narrowBandReuseLayout,narrowBandReuseWGSL} from "../lib/methods/uniform/uniform-narrow-band-reuse.wgsl";
import {withUniformDevice} from "./helpers/uniform-geometric";

test("regional crossing cache restores exact masks before gathers and rejects changed dependencies",{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 await withUniformDevice("regional crossing cache",async device=>{
  const T=4096,R=64,L=narrowBandReuseLayout(30000,R),tile=292;
  const maskWord=8204,reachWord=16400,candidateWord=20500;
  const helper=narrowBandReuseWGSL(L).split("fn nbStoreReconstructed(")[0]!;
  const seeds=narrowBandMembershipWGSL.slice(narrowBandMembershipWGSL.indexOf("var<workgroup> nbBandVertices:"),narrowBandMembershipWGSL.indexOf("@compute @workgroup_size(64) fn depthLists("));
  const bins=device.createBuffer({size:4*(L.base+L.words),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC});
  const topology=device.createBuffer({size:4*T,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});device.queue.writeBuffer(topology,0,new Uint32Array(T).fill(1));
  const module=device.createShaderModule({code:`
const UM_TILES:u32=${T}u;const NB_SURFACE_TILES:u32=0u;const NB_ACTIVITY_TARGET:u32=4096u;
const NB_BAND_CANDIDATES:u32=${candidateWord}u;const NB_BAND_MASK:u32=${maskWord}u;const NB_BAND_REACH:u32=${reachWord}u;
@group(0) @binding(0) var<storage,read_write> bins:array<atomic<u32>>;
@group(0) @binding(1) var<storage,read> umTopology:array<u32>;
struct Params{hDt:vec4f};var<private> params:Params=Params(vec4f(1));
fn umCorner(k:u32,n:u32)->vec3u{return vec3u(k%n,(k/n)%n,k/(n*n));}
fn umTileCoord(t:u32)->vec3u{let b=t>>9u;let q=t&511u;return vec3u(q&7u,(q>>3u)&7u,q>>6u)+8u*vec3u(b&1u,(b>>1u)&1u,b>>2u)+vec3u(atomicLoad(&bins[29996u]),0,0);}
fn umTileKey(q:vec3i)->u32{let p=q-vec3i(i32(atomicLoad(&bins[29996u])),0,0);if(any(p<vec3i(0))||any(p>=vec3i(16))){return 0xffffffffu;}let u=vec3u(p);let b=u>>vec3u(3u);let l=u&vec3u(7u);return 512u*(b.x+2u*b.y+4u*b.z)+l.x+8u*(l.y+8u*l.z);}
fn umTileKind(q:vec3i)->u32{return select(2u,0u,umTileKey(q)<UM_TILES);}
fn umKeyStored(t:u32)->bool{return t<UM_TILES;}
fn umTileLive(t:u32)->bool{return t<UM_TILES;}
fn umSolidRegionClearAtBuild(p:vec3i)->bool{return true;}
fn umSolidEnabled()->bool{return true;}
fn umSolidOpenCells(t:u32)->vec2u{return vec2u(0xffffffffu^atomicLoad(&bins[29995u]),0xffffffffu);}
fn umSolidTraceWorld(p:vec3f)->vec3f{return p;}
fn umBodyCount()->u32{return 0u;}
fn umBodyNear(i:u32,p:vec3f,margin:f32)->bool{return false;}
fn umCellOpen(p:vec3i)->f32{return 1.0;}
fn nbCrossingVertex(t:u32,origin:vec3u,p:vec3u)->f32{atomicAdd(&bins[29998u],1u);if(atomicLoad(&bins[29994u])!=0u){return bitcast<f32>(atomicLoad(&bins[29994u]));}return f32(p.y)-17.5;}
${helper}
${seeds}
@compute @workgroup_size(64) fn prepare(@builtin(workgroup_id) g:vec3u,@builtin(local_invocation_index) lane:u32){let t=nbReuseTile(g.x,lane);nbPrepareReuse(g.x,lane,t);}
@compute @workgroup_size(1) fn markBoundary(){nbMarkPhiChanged(vec3u(16));}
@compute @workgroup_size(1) fn markInterior(){nbMarkPhiChanged(vec3u(17,18,19));}
`});
  const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");assert.deepEqual(errors.map(m=>m.message),[]);
  const entries=["prepare","depthSeeds","markBoundary","markInterior"] as const;
  const layout=device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}}]});
  const pipelineLayout=device.createPipelineLayout({bindGroupLayouts:[layout]});
  const pipelines=await Promise.all(entries.map(entryPoint=>device.createComputePipelineAsync({layout:pipelineLayout,compute:{module,entryPoint}})));
  const group=device.createBindGroup({layout,entries:[{binding:0,resource:{buffer:bins}},{binding:1,resource:{buffer:topology}}]});
  const read=async(offset:number,count:number)=>{const b=device.createBuffer({size:4*count,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});try{const e=device.createCommandEncoder();e.copyBufferToBuffer(bins,4*offset,b,0,b.size);device.queue.submit([e.finish()]);await b.mapAsync(GPUMapMode.READ);return new Uint32Array(b.getMappedRange()).slice();}finally{if(b.mapState==="mapped")b.unmap();b.destroy();}};
  const dispatch=async(index:number,n=1)=>{const e=device.createCommandEncoder(),p=e.beginComputePass();p.setPipeline(pipelines[index]!);p.setBindGroup(0,group);p.dispatchWorkgroups(n);p.end();device.queue.submit([e.finish()]);await device.queue.onSubmittedWorkDone();};
  let parity=0;
  const crossing=async()=>{
   device.queue.writeBuffer(bins,4*L.control,new Uint32Array([parity,1,1]));parity=1-parity;
   device.queue.writeBuffer(bins,4*(maskWord+2*tile),new Uint32Array(2));device.queue.writeBuffer(bins,4*29998,new Uint32Array([0]));
   await dispatch(0,R);await dispatch(1);
   return {calls:(await read(29998,1))[0]!,mask:await read(maskWord+2*tile,2)};
  };
  try{
   device.queue.writeBuffer(bins,4*L.live,new Uint32Array(R).fill(1));device.queue.writeBuffer(bins,4*candidateWord,new Uint32Array([tile]));
   const first=await crossing();assert.equal(first.calls,125);assert.ok(first.mask.some(v=>v!==0));
   const reused=await crossing();assert.equal(reused.calls,0);assert.deepEqual(reused.mask,first.mask);
   // Home closure epochs include changes to every incident boundary vertex.
   device.queue.writeBuffer(bins,4*(L.live+7),new Uint32Array([2]));assert.equal((await crossing()).calls,125);assert.equal((await crossing()).calls,0);
   // Activation requests a previously computed mask but cannot alter it.
   device.queue.writeBuffer(bins,4*(4096+291),new Float32Array([1]));assert.equal((await crossing()).calls,0);
   device.queue.writeBuffer(topology,4*292,new Uint32Array([2]));assert.equal((await crossing()).calls,125);assert.equal((await crossing()).calls,0);
   // Equal epochs cannot alias a remapped physical page.
   device.queue.writeBuffer(bins,4*29996,new Uint32Array([8]));assert.equal((await crossing()).calls,125);assert.equal((await crossing()).calls,0);
   device.queue.writeBuffer(bins,4*29995,new Uint32Array([1]));assert.equal((await crossing()).calls,125);
   device.queue.writeBuffer(bins,4*29995,new Uint32Array([0]));assert.equal((await crossing()).calls,125);assert.equal((await crossing()).calls,0);
   device.queue.writeBuffer(bins,4*29994,new Uint32Array([0x7fc00000]));device.queue.writeBuffer(bins,4*(L.live+7),new Uint32Array([3]));assert.equal((await crossing()).calls,125);assert.equal((await crossing()).calls,125,"nonfinite masks must never become persistent hits");
   device.queue.writeBuffer(bins,4*29994,new Uint32Array([0]));
   device.queue.writeBuffer(bins,4*(L.live+7),new Uint32Array([0xfffffffe]));assert.equal((await crossing()).calls,125);
   device.queue.writeBuffer(bins,4*29996,new Uint32Array([0]));device.queue.writeBuffer(bins,4*L.live,new Uint32Array(R).fill(1));
   await dispatch(2);const boundary=await read(L.live,R);assert.deepEqual([...boundary.slice(0,8)],new Array(8).fill(2));assert.ok(boundary.slice(8).every(v=>v===1));
   await dispatch(3);const interior=await read(L.live,R);assert.equal(interior[7],3);assert.equal(interior[0],2);
  }finally{bins.destroy();topology.destroy();}
 });
});

test("retained XYZ nearest minima match full search and crossing changes invalidate metric payloads",{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 await withUniformDevice("retained nearest minima",async device=>{
  const T=4096,R=64,C=64*T,A=0,B=C/2,band=B+C,mask=band+4,reach=mask+2*T,search=reach+2*T;
  const L=narrowBandReuseLayout(search+T,R),helper=narrowBandReuseWGSL(L).split("fn nbStoreReconstructed(")[0]!;
  const row=narrowBandMembershipWGSL.slice(narrowBandMembershipWGSL.indexOf("fn nbBandRow("),narrowBandMembershipWGSL.indexOf(narrowBandNearestWGSL));
  const bins=device.createBuffer({size:4*(L.base+L.words),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC});
  const module=device.createShaderModule({code:`
const UM_TILES:u32=${T}u;const UM_CELL_KEYS:u32=${C}u;const NB_DEPTH_A:u32=${A}u;const NB_DEPTH_B:u32=${B}u;
const NB_BAND:u32=${band}u;const NB_BAND_MASK:u32=${mask}u;const NB_BAND_REACH:u32=${reach}u;const NB_BAND_SEARCH:u32=${search}u;const NB_NO_SURFACE:u32=0xffffffffu;
@group(0) @binding(0) var<storage,read_write> bins:array<atomic<u32>>;
var<private> umTopology:array<u32,4096>;
struct Params{hDt:vec4f};var<private> params:Params=Params(vec4f(1));
fn umCorner(k:u32,n:u32)->vec3u{return vec3u(k%n,(k/n)%n,k/(n*n));}
fn umTileCoord(t:u32)->vec3u{let b=t>>9u;let q=t&511u;return vec3u(q&7u,(q>>3u)&7u,q>>6u)+8u*vec3u(b&1u,(b>>1u)&1u,b>>2u);}
fn umTileKey(p:vec3i)->u32{if(any(p<vec3i(0))||any(p>=vec3i(16))){return 0xffffffffu;}let u=vec3u(p);let b=u>>vec3u(3u);let l=u&vec3u(7u);return 512u*(b.x+2u*b.y+4u*b.z)+l.x+8u*(l.y+8u*l.z);}
fn umCellPresent(p:vec3i)->bool{return all(p>=vec3i(0))&&all(p<vec3i(64));}
fn umCellAt(p:vec3u)->u32{return 64u*umTileKey(vec3i(p/4u))+(p.x&3u)+4u*(p.y&3u)+16u*(p.z&3u);}
fn umTileKind(q:vec3i)->u32{return select(2u,0u,umTileKey(q)<UM_TILES);}
fn umKeyStored(t:u32)->bool{return t<UM_TILES;}
fn umTileLive(t:u32)->bool{return t<UM_TILES;}
fn umSolidRegionClearAtBuild(p:vec3i)->bool{return true;}
fn umSolidEnabled()->bool{return false;}
fn umSolidOpenCells(t:u32)->vec2u{return vec2u(0xffffffffu);}
fn umSolidTraceWorld(p:vec3f)->vec3f{return p;}
fn umBodyCount()->u32{return 0u;}
fn umBodyNear(i:u32,p:vec3f,margin:f32)->bool{return false;}
${helper}
${row}
fn nbAggregateMetricPlane(region:u32,lane:u32){}
${narrowBandNearestWGSL}
`});
  assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==="error").map(m=>m.message),[]);
  const layout=device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}}]});
  const pipelineLayout=device.createPipelineLayout({bindGroupLayouts:[layout]});
  const names=[...narrowBandNearestControlEntries,...narrowBandNearestEntries];
  const pipelines=await Promise.all(names.map(entryPoint=>device.createComputePipelineAsync({layout:pipelineLayout,compute:{module,entryPoint}})));
  const group=device.createBindGroup({layout,entries:[{binding:0,resource:{buffer:bins}}]});
  const run=()=>{const e=device.createCommandEncoder(),p=e.beginComputePass();p.setBindGroup(0,group);for(const pipeline of pipelines){p.setPipeline(pipeline);p.dispatchWorkgroups(1024);}p.end();device.queue.submit([e.finish()]);};
  const read=async(offset:number,count:number)=>{const b=device.createBuffer({size:4*count,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});try{const e=device.createCommandEncoder();e.copyBufferToBuffer(bins,4*offset,b,0,b.size);device.queue.submit([e.finish()]);await b.mapAsync(GPUMapMode.READ);return new Uint32Array(b.getMappedRange()).slice();}finally{if(b.mapState==="mapped")b.unmap();b.destroy();}};
  const outputs=()=>read(B,C);
  try{
   device.queue.writeBuffer(bins,4*L.control,new Uint32Array([0,1,1]));
   device.queue.writeBuffer(bins,4*(band+1),new Uint32Array([T]));
   device.queue.writeBuffer(bins,4*search,Uint32Array.from({length:T},(_,i)=>i));
   device.queue.writeBuffer(bins,4*reach,new Uint32Array(T).fill(1));
   device.queue.writeBuffer(bins,4*L.allowed,new Uint32Array(R).fill(1));
   const crossing=292;device.queue.writeBuffer(bins,4*(mask+2*crossing),new Uint32Array([0x22222222,0x22222222]));
   run();const full=await outputs();
   device.queue.writeBuffer(bins,4*A,new Uint32Array(C/2+C));run();assert.deepEqual(await outputs(),full);
   // Phi epochs and ownership certificates remain unchanged; a changed
   // crossing selection alone must invalidate both nearest and metric data.
   device.queue.writeBuffer(bins,4*L.metricValid,new Uint32Array(64*R).fill(1));
   device.queue.writeBuffer(bins,4*(mask+2*crossing),new Uint32Array([0x44444444,0x44444444]));
   run();const changed=await outputs();assert.notDeepEqual(changed,full);
   const region=7;assert.equal((await read(L.nearestFlags+region,1))[0],0);
   assert.ok((await read(L.metricValid+64*region,64)).every(v=>v===0));
   device.queue.writeBuffer(bins,4*L.control,new Uint32Array([0,1,0]));run();assert.deepEqual(await outputs(),changed);
  }finally{bins.destroy();}
 });
});
