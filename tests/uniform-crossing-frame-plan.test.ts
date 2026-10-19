import assert from "node:assert/strict";
import test from "node:test";
import { narrowBandMembershipWGSL } from "../lib/methods/uniform/uniform-narrow-band-membership.wgsl";
import { withUniformDevice } from "./helpers/uniform-geometric";

test("frame crossing domain shares exact requests and bounded neighbourhoods through activation and retirement",{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 await withUniformDevice("frame crossing domain",async device=>{
  const regions=24,tiles=1536,plan=20*tiles,words=plan+5+7*regions;
  const module=device.createShaderModule({code:`
@group(0) @binding(0) var<storage,read_write> bins:array<atomic<u32>>;
@group(0) @binding(1) var<storage,read> topology:array<u32>;
const UM_TILES:u32=${tiles}u;const NB_REUSE_REGIONS:u32=${regions}u;const NB_REUSE_CAPACITY:u32=${regions}u;
const NB_SURFACE_TILES:u32=0u;const NB_ACTIVITY_TARGET:u32=UM_TILES;
const NB_BAND_MASK:u32=2u*UM_TILES;const NB_BAND_REACH:u32=4u*UM_TILES;const NB_FRAME_PLAN:u32=${plan}u;
const NB_REUSE_ALLOWED:u32=5u*UM_TILES;const NB_REUSE_NEAREST_VALID:u32=6u*UM_TILES;
const NB_REUSE_NEAREST_FLAGS:u32=7u*UM_TILES;const NB_REUSE_REGION_PLANE:u32=8u*UM_TILES;
const NB_REUSE_VALID:u32=9u*UM_TILES;const NB_REUSE_METRIC_VALID:u32=10u*UM_TILES;const NB_REUSE_PLANE:u32=11u*UM_TILES;
fn umCorner(k:u32,n:u32)->vec3u{return vec3u(k%n,(k/n)%n,k/(n*n));}
fn umTileCoord(t:u32)->vec3u{return vec3u(t%16u,(t/16u)%12u,t/192u);}
fn umTileKey(p:vec3i)->u32{if(any(p<vec3i(0))||any(p>=vec3i(16,12,8))){return UM_TILES;}return u32(p.x+16*p.y+192*p.z);}
fn umKeyStored(t:u32)->bool{return t<UM_TILES&&(topology[t]&1u)!=0u;}
fn umTileLive(t:u32)->bool{return umKeyStored(t);}
fn umTileMinimumWidth(t:u32)->u32{return select(4u,1u,(topology[t]&2u)!=0u);}
fn umPageCoord(r:u32)->vec3u{return vec3u(r%4u,(r/4u)%3u,r/12u);}
fn umPageOfTile(t:u32)->u32{let p=umTileCoord(t)/4u;return p.x+4u*p.y+12u*p.z;}
fn nbReuseTile(r:u32,lane:u32)->u32{return umTileKey(vec3i(4u*umPageCoord(r)+umCorner(lane,4u)));}
${narrowBandMembershipWGSL.slice(narrowBandMembershipWGSL.indexOf("fn nbCrossingRequested("),narrowBandMembershipWGSL.indexOf("fn nbDepthTile("))}
`});
  assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==="error"),[]);
  const layout=device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}}]});
  const pipelineLayout=device.createPipelineLayout({bindGroupLayouts:[layout]});
  const pipelines=await Promise.all(["depthPlanSeed","depthPlanExpand"].map(entryPoint=>device.createComputePipelineAsync({layout:pipelineLayout,compute:{module,entryPoint}})));
  const bins=device.createBuffer({size:4*words,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
  const topology=device.createBuffer({size:4*tiles,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
  const read=device.createBuffer({size:4*words,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  const group=device.createBindGroup({layout,entries:[{binding:0,resource:{buffer:bins}},{binding:1,resource:{buffer:topology}}]});
  const regionOf=(t:number)=>Math.floor(t%16/4)+4*Math.floor(Math.floor(t/16)%12/4)+12*Math.floor(t/768);
  const coord=(r:number)=>[r%4,Math.floor(r/4)%3,Math.floor(r/12)];
  try{
   for(const requests of [[],[1],[1,799,1500],[]]){
    const live=Uint32Array.from({length:tiles},(_,t)=>![5,9].includes(regionOf(t))?1:0);
    const data=new Uint32Array(words);data.fill(0xffffffff,2*tiles,5*tiles);
    // WARM surface work with zero fine tiles/particles must still maintain air.
    const changed=[0,18];for(const r of changed)data[plan+4+4*regions+r]=8;
    requests.forEach((t,i)=>{if(i===0)live[t]!|=2;else if(i===1)data[t]=1;else data[tiles+t]=0x3f800000;});
    device.queue.writeBuffer(bins,0,data);device.queue.writeBuffer(topology,0,live);
    const e=device.createCommandEncoder(),pass=e.beginComputePass();pass.setBindGroup(0,group);
    for(const p of pipelines){pass.setPipeline(p);pass.dispatchWorkgroups(7);}pass.end();e.copyBufferToBuffer(bins,0,read,0,bins.size);
    device.queue.submit([e.finish()]);await read.mapAsync(GPUMapMode.READ);const output=new Uint32Array(read.getMappedRange()).slice();read.unmap();
    const seeds=[...new Set(requests.filter(t=>live[t]).map(regionOf))];
    const expected=(sources:number[],radius:number)=>Array.from({length:regions},(_,r)=>r).filter(r=>![5,9].includes(r)&&sources.some(s=>coord(r).every((v,a)=>Math.abs(v-coord(s)[a]!)<=radius)));
    const list=(count:number,offset:number)=>[...output.slice(offset,offset+output[count]!)].sort((a,b)=>a-b);
    assert.deepEqual(list(plan+1,plan+4+3*regions),expected(seeds,1));
    assert.deepEqual(list(plan+3,plan+4+5*regions),expected([...changed,...seeds],2));
    assert.deepEqual(list(plan+4+6*regions,plan+5+6*regions),[...new Set([...changed,...seeds])].sort((a,b)=>a-b));
    assert.equal(output[plan+2],requests.length);
    assert.ok(output.slice(2*tiles,5*tiles).every(x=>x===0),"old masks and reach survived request retirement");
   }
  }finally{bins.destroy();topology.destroy();read.destroy();}
 });
});
