import assert from "node:assert/strict";
import test from "node:test";
import { narrowBandAirWGSL } from "../lib/methods/uniform/uniform-narrow-band-air.wgsl";
import { withUniformDevice } from "./helpers/uniform-geometric";

test("regional air propagation matches separable sparse-hole protection and gate",{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 await withUniformDevice("regional air plan",async device=>{
  const dimensions=[16,12,8],tiles=dimensions.reduce((a,b)=>a*b),regions=tiles/64,plan=5*tiles,words=plan+4+6*regions;
  const module=device.createShaderModule({code:`
@group(0) @binding(0) var<storage,read_write> bins:array<atomic<u32>>;
@group(0) @binding(1) var<storage,read> live:array<u32>;
const UM_TILES:u32=${tiles}u;const NB_AIR:u32=0u;const NB_FRAME_PLAN:u32=${plan}u;const NB_REUSE_REGIONS:u32=${regions}u;
const umDispatchX:u32=65535u;
override nbAirLanes:u32=64u;override nbAirFused:bool=false;
fn nbAirClearRegion(region:u32,world:vec3i,lane:u32){}
fn umCorner(k:u32,n:u32)->vec3u{return vec3u(k%n,(k/n)%n,k/(n*n));}
fn umTileCoord(t:u32)->vec3u{return vec3u(t%16u,(t/16u)%12u,t/192u);}
fn umTileKey(p:vec3i)->u32{if(any(p<vec3i(0))||any(p>=vec3i(16,12,8))){return UM_TILES;}return u32(p.x+16*p.y+192*p.z);}
fn umKeyStored(t:u32)->bool{return t<UM_TILES&&live[t]!=0u;}
fn umTileLive(t:u32)->bool{return umKeyStored(t);}
fn umPageOfTile(t:u32)->u32{let p=umTileCoord(t)/4u;return p.x+4u*p.y+12u*p.z;}
fn umPageCoord(r:u32)->vec3u{return vec3u(r%4u,(r/4u)%3u,r/12u);}
fn nbReuseTile(r:u32,lane:u32)->u32{return umTileKey(vec3i(4u*umPageCoord(r)+umCorner(lane,4u)));}
${narrowBandAirWGSL.slice(narrowBandAirWGSL.indexOf("@compute @workgroup_size(64) fn airSpread0"),narrowBandAirWGSL.indexOf("var<workgroup> nbAirOrigin:"))}
`});
  assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==="error"),[]);
  const layout=device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}}]});
  const pipelineLayout=device.createPipelineLayout({bindGroupLayouts:[layout]});
  const pipelines=await Promise.all(["airSpread0","airSpread1","airSpread2","airGate","airPlanPropagate"].map(entryPoint=>device.createComputePipelineAsync({layout:pipelineLayout,compute:{module,entryPoint}})));
  const buffers=[0,1].map(()=>device.createBuffer({size:4*words,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST}));
  const stored=device.createBuffer({size:4*tiles,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
  const read=device.createBuffer({size:16*tiles,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  const groups=buffers.map(buffer=>device.createBindGroup({layout,entries:[{binding:0,resource:{buffer}},{binding:1,resource:{buffer:stored}}]}));
  try{
   for(const holes of [false,true])for(const density of [0,1,17]){
    const included=Array.from({length:regions},(_,r)=>r).filter(r=>!holes||![1,5,9,18].includes(r));
    const live=Uint32Array.from({length:tiles},(_,t)=>{
     const x=t%16,y=Math.floor(t/16)%12,z=Math.floor(t/192),r=Math.floor(x/4)+4*Math.floor(y/4)+12*Math.floor(z/4);return +included.includes(r);
    });
    const data=new Uint32Array(words);for(let t=0;t<tiles;t++)data[t]=live[t]&&((t*17+13)%131<density)?1:0;
    data[plan]=1;data[plan+3]=included.length;data.set(included,plan+4+5*regions);
    device.queue.writeBuffer(stored,0,live);for(const b of buffers)device.queue.writeBuffer(b,0,data);
    const e=device.createCommandEncoder(),pass=e.beginComputePass();
    pass.setBindGroup(0,groups[0]!);for(const p of pipelines.slice(0,4)){pass.setPipeline(p);pass.dispatchWorkgroups(1024);}
    pass.setBindGroup(0,groups[1]!);pass.setPipeline(pipelines[4]!);pass.dispatchWorkgroups(7);pass.end();
    for(let arm=0;arm<2;arm++)e.copyBufferToBuffer(buffers[arm]!,12*tiles,read,8*tiles*arm,8*tiles);
    device.queue.submit([e.finish()]);await read.mapAsync(GPUMapMode.READ);const output=new Uint32Array(read.getMappedRange()).slice();read.unmap();
    assert.deepEqual(output.slice(2*tiles),output.slice(0,2*tiles),`holes=${holes}, density=${density}`);
   }
  }finally{for(const b of buffers)b.destroy();stored.destroy();read.destroy();}
 });
});
