import assert from "node:assert/strict";
import test from "node:test";
import { narrowBandMembershipWGSL } from "../lib/methods/uniform/uniform-narrow-band-membership.wgsl";
import { narrowBandNearestWGSL } from "../lib/methods/uniform/uniform-narrow-band-nearest.wgsl";
import { withUniformDevice } from "./helpers/uniform-geometric";

test("crossing snapshot emits the exact lists including regions beyond its retained cache",{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 await withUniformDevice("crossing list fusion",async device=>{
  const module=device.createShaderModule({code:`
@group(0) @binding(0) var<storage,read_write> bins:array<atomic<u32>>;
const NB_REUSE_REGIONS:u32=8u;override NB_REUSE_CAPACITY:u32=2u;
const NB_BAND_TILES:u32=512u;const NB_BAND:u32=0u;const NB_BAND_REACH:u32=4u;
const NB_BAND_SEARCH:u32=516u;const NB_BAND_DISTANCE:u32=1028u;const NB_BAND_MASK:u32=1540u;
const NB_REUSE_ALLOWED:u32=2600u;const NB_REUSE_NEAREST_VALID:u32=2610u;const NB_REUSE_NEAREST_FLAGS:u32=2620u;
const NB_REUSE_REGION_PLANE:u32=2640u;const NB_REUSE_METRIC_VALID:u32=2710u;const NB_REUSE_NEAREST_STATE:u32=3300u;
var<workgroup> nbReuseClean:atomic<u32>;
fn umCorner(k:u32,n:u32)->vec3u{return vec3u(k%n,(k/n)%n,k/(n*n));}
fn nbReuseTile(region:u32,lane:u32)->u32{return 64u*region+lane;}
fn umTileLive(tile:u32)->bool{return tile<512u;}
fn umTileCoord(tile:u32)->vec3u{return umCorner(tile,8u);}
fn umTileKey(p:vec3i)->u32{return u32(p.x+8*p.y+64*p.z);}
fn nbReuseCacheEnabled()->bool{return false;}
fn nbPlanRegions(lane:u32)->u32{return NB_REUSE_REGIONS;}
fn nbPlanRegion(slot:u32,lane:u32)->u32{return slot;}
fn nbAggregateMetricPlane(region:u32,lane:u32){}
${narrowBandMembershipWGSL.slice(narrowBandMembershipWGSL.indexOf("fn nbDepthList("),narrowBandMembershipWGSL.indexOf("// The crossings of one x row"))}
${narrowBandNearestWGSL.slice(0,narrowBandNearestWGSL.indexOf("var<workgroup> nbMetricClean"))}
`});
  assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==="error"),[]);
  const layout=device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}}]});
  const pipelineLayout=device.createPipelineLayout({bindGroupLayouts:[layout]});
  const bins=device.createBuffer({size:20000,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC});
  const read=device.createBuffer({size:bins.size,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
  const group=device.createBindGroup({layout,entries:[{binding:0,resource:{buffer:bins}}]});
  try{
   for(const capacity of [0,2,8]){
    const pipeline=await device.createComputePipelineAsync({layout:pipelineLayout,compute:{module,entryPoint:"depthReuseSnapshot",constants:{nbListWithSnapshot:1,NB_REUSE_CAPACITY:capacity}}});
    for(const density of [0,1,4]){
     const flags=Uint32Array.from({length:512},(_,i)=>density===0?0:i%7<density?3:i%11===0?2:0);device.queue.writeBuffer(bins,16,flags);
     const e=device.createCommandEncoder();e.clearBuffer(bins,0,16);const pass=e.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(1024);pass.end();e.copyBufferToBuffer(bins,0,read,0,bins.size);device.queue.submit([e.finish()]);await read.mapAsync(GPUMapMode.READ);const data=new Uint32Array(read.getMappedRange()).slice();read.unmap();
     for(const [count,offset,bit] of [[1,516,1],[2,1028,2]]){
      const expected=[...flags.keys()].filter(i=>(flags[i]!&bit!)!==0);
      assert.equal(data[count!]!,expected.length);assert.deepEqual([...data.slice(offset!,offset!+expected.length)].sort((a,b)=>a-b),expected);
     }
    }
   }
  }finally{bins.destroy();read.destroy();}
 });
});
