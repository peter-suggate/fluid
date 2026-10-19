import assert from "node:assert/strict";
import test from "node:test";
import { narrowBandAirWGSL } from "../lib/methods/uniform/uniform-narrow-band-air.wgsl";
import { withUniformDevice } from "./helpers/uniform-geometric";

// Compare the real propagation/clear code, with dense addressing and a buffer
// standing in for outputPhi. Sparse whole-region holes and canonical writers
// remain explicit, so local protection must agree at region boundaries too.
test("regional air update fuses clearing exactly across holes, cold regions and activation",{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 await withUniformDevice("regional air clear",async device=>{
  const tiles=16*12*8,regions=tiles/64,plan=5*tiles,words=plan+4+7*regions,vertices=65*49*33;
  const body=narrowBandAirWGSL.slice(narrowBandAirWGSL.indexOf("@compute @workgroup_size(64) fn airSpread0"))
   .replaceAll("textureLoad(outputPhi,vec3i(p)).x","values[vertex(p)]")
   .replaceAll("textureStore(outputPhi,vec3i(p),vec4f(next));","values[vertex(p)]=next;")
   .replaceAll("umTopology[UM_TILES+j]","j").replaceAll("umCounts.x","UM_TILES");
  const module=device.createShaderModule({code:`
@group(0) @binding(0) var<storage,read_write> bins:array<atomic<u32>>;
@group(0) @binding(1) var<storage,read> live:array<u32>;
@group(0) @binding(2) var<storage,read_write> values:array<f32>;
@group(0) @binding(3) var<storage,read_write> state:array<atomic<u32>>;
const UM_TILES:u32=${tiles}u;const NB_AIR:u32=0u;const NB_FRAME_PLAN:u32=${plan}u;const NB_REUSE_REGIONS:u32=${regions}u;
override nbAirLanes:u32=64u;override nbAirFused:bool=false;override nbAirListed:bool=false;override width:u32=4u;
struct Params{hDt:vec4f} const params=Params(vec4f(0.02,0.04,0.01,0.016666667));
fn umCorner(k:u32,n:u32)->vec3u{return vec3u(k%n,(k/n)%n,k/(n*n));}
fn umTileCoord(t:u32)->vec3u{return vec3u(t%16u,(t/16u)%12u,t/192u);}
fn umTileKey(p:vec3i)->u32{if(any(p<vec3i(0))||any(p>=vec3i(16,12,8))){return UM_TILES;}return u32(p.x+16*p.y+192*p.z);}
fn umKeyStored(t:u32)->bool{return t<UM_TILES&&live[t]!=0u;}
fn umTileLive(t:u32)->bool{return umKeyStored(t);}
fn umPageOfTile(t:u32)->u32{let p=umTileCoord(t)/4u;return p.x+4u*p.y+12u*p.z;}
fn umPageCoord(r:u32)->vec3u{return vec3u(r%4u,(r/4u)%3u,r/12u);}
fn nbReuseTile(r:u32,lane:u32)->u32{return umTileKey(vec3i(4u*umPageCoord(r)+umCorner(lane,4u)));}
fn umTileWidth(t:u32)->u32{return width;}
fn umTileMinimumWidth(t:u32)->u32{return 1u;}
fn umTileMaximumWidth(t:u32)->u32{return 4u;}
fn umLowOpen(p:vec3i)->vec3<bool>{return p==vec3i(0);}
struct Authority{width:u32,tile:u32}
fn umVertexAuthority(p:vec3u)->Authority{
 var tile=UM_TILES;
 for(var k=0u;k<8u;k++){let t=umTileKey((vec3i(p)-vec3i(umCorner(k,2u)))>>vec3u(2u));if(umKeyStored(t)){tile=min(tile,t);}}
 return Authority(width,tile);
}
fn nbTileCold(t:u32)->bool{return live[t]==2u;}
fn nbVertexCold(p:vec3u)->bool{return false;}
fn nbReuseEnabled()->bool{return false;}
fn nbMarkPhiChanged(p:vec3u){}
fn vertex(p:vec3u)->u32{return p.x+65u*(p.y+49u*p.z);}
${body}`});
  assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==="error"),[]);
  const layout=device.createBindGroupLayout({entries:[0,1,2,3].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:binding===1?"read-only-storage" as const:"storage" as const}}))});
  const pipelineLayout=device.createPipelineLayout({bindGroupLayouts:[layout]});
  const storage=(size:number)=>device.createBuffer({size,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
  const arms=[0,1].map(()=>({bins:storage(4*words),values:storage(4*vertices),state:storage(64)}));
  const stored=storage(4*tiles),read=device.createBuffer({size:8*vertices+128,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  const groups=arms.map(a=>device.createBindGroup({layout,entries:[a.bins,stored,a.values,a.state].map((buffer,binding)=>({binding,resource:{buffer}}))}));
  try{
   for(const width of [1,4]){
    const legacy=await Promise.all(["airSpread0","airSpread1","airSpread2","airGate","airClearFine","airClearCoarse"].map(entryPoint=>device.createComputePipelineAsync({layout:pipelineLayout,compute:{module,entryPoint,constants:{width}}})));
    const fused=await device.createComputePipelineAsync({layout:pipelineLayout,compute:{module,entryPoint:"airPlanPropagate",constants:{width,nbAirFused:1,nbAirLanes:256}}});
    for(const holes of [false,true])for(const density of [0,1,17,0]){
     const included=Array.from({length:regions},(_,r)=>r).filter(r=>!holes||![1,5,9,18].includes(r));
     const live=Uint32Array.from({length:tiles},(_,t)=>{const r=Math.floor((t%16)/4)+4*Math.floor((Math.floor(t/16)%12)/4)+12*Math.floor(Math.floor(t/192)/4);return included.includes(r)?r%7===0?2:1:0;});
     const data=new Uint32Array(words);for(let t=0;t<tiles;t++)data[t]=live[t]&&((t*17+13)%131<density)?1:0;
     data[plan]=1;data[plan+3]=included.length;data.set(included,plan+4+5*regions);
     for(const r of included)data[plan+4+4*regions+r]=8;
     const phi=Float32Array.from({length:vertices},(_,i)=>((i*31)%113-24)*0.01);
     device.queue.writeBuffer(stored,0,live);
     const e=device.createCommandEncoder();
     for(let arm=0;arm<2;arm++){
      const a=arms[arm]!;device.queue.writeBuffer(a.bins,0,data);device.queue.writeBuffer(a.values,0,phi);e.clearBuffer(a.state);
      const pass=e.beginComputePass();pass.setBindGroup(0,groups[arm]!);
      for(const p of arm===0?legacy:[fused]){pass.setPipeline(p);pass.dispatchWorkgroups(arm===0?1024:7);}pass.end();
      e.copyBufferToBuffer(a.values,0,read,4*vertices*arm,4*vertices);e.copyBufferToBuffer(a.state,0,read,8*vertices+64*arm,64);
     }
     device.queue.submit([e.finish()]);await read.mapAsync(GPUMapMode.READ);const output=new Uint32Array(read.getMappedRange()).slice();read.unmap();
     assert.deepEqual(output.slice(vertices,2*vertices),output.slice(0,vertices),`width=${width}, holes=${holes}, density=${density}`);
     assert.deepEqual(output.slice(2*vertices+16),output.slice(2*vertices,2*vertices+16),"changed canonical vertices");
    }
   }
  }finally{for(const a of arms){a.bins.destroy();a.values.destroy();a.state.destroy();}stored.destroy();read.destroy();}
 });
});
