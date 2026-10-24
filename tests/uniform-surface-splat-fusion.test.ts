import assert from "node:assert/strict";
import test from "node:test";
import { narrowBandSplatWGSL } from "../lib/methods/uniform/uniform-narrow-band-splat.wgsl";
import { withUniformDevice } from "./helpers/uniform-geometric";

test("occupied surface scatter matches cell scan with cold cells and long particle runs",{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 await withUniformDevice("surface scatter fusion",async device=>{
  const module=device.createShaderModule({code:/* wgsl */`
const UM_CELL_KEYS=192u;const UM_TILES=3u;const NB_COVERAGE=256u;const NB_DEPTH_A=1024u;const NB_SURFACE_HOT_NEAREST=1400u;const NB_BAND=2048u;const NB_OCCUPIED=2052u;
const nbSparseBins=true;
const NB_LINKS=UM_CELL_KEYS;fn nbBinTile(t:u32)->u32{return 64u*t;}fn nbBinCell(c:u32)->u32{return c;}
@group(0) @binding(0) var<storage,read_write> bins:array<atomic<u32>>;
@group(0) @binding(1) var<storage,read_write> links:array<u32>;
@group(0) @binding(2) var<storage,read> particles:array<vec4f>;
fn umTileLive(t:u32)->bool{return t<3u;}
fn umLiveTileCount()->u32{return 3u;}fn umLiveTile(j:u32)->u32{return j;}
fn nbTileCold(t:u32)->bool{return t==2u;}
fn umTileCoord(t:u32)->vec3u{return vec3u(t,0,0);}
fn umCorner(j:u32,n:u32)->vec3u{return vec3u(j%n,(j/n)%n,j/(n*n));}
fn umVertexPresent(p:vec3i)->bool{return all(p>=vec3i(0))&&all(p<vec3i(13,5,5));}
fn umVertexAt(p:vec3u)->u32{return p.x+13u*(p.y+5u*p.z);}
fn umVertexSlot(p:vec3i)->vec4u{return vec4u(vec3u(p),u32(umVertexPresent(p)));}
fn umBlockStored(b:u32)->bool{return b!=0u;}
fn umVertexSlotAt(p:vec4u)->u32{return umVertexAt(p.xyz);}
fn umLoadAddressPlan(b:u32,lane:u32){}
fn nbPosition(i:u32)->vec3f{return particles[i].xyz;}
fn nbSurfaceParticleHot(i:u32)->bool{return particles[i].w>1.0;}
`+narrowBandSplatWGSL});
  assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==="error"),[]);
  const layout=device.createBindGroupLayout({entries:[0,1,2].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:binding===2?"read-only-storage" as const:"storage" as const}}))});
  const pipelineLayout=device.createPipelineLayout({bindGroupLayouts:[layout]});
  const pipelines=await Promise.all(["surfaceCells","surfaceSplat","surfaceSplatOccupied","surfaceSplatReference"].map(entryPoint=>device.createComputePipelineAsync({layout:pipelineLayout,compute:{module,entryPoint}})));
  const initial=new Uint32Array(2100),links=new Uint32Array(620),positions=new Float32Array(4*103);
  initial[0]=100;initial[1]=1;initial[128]=1;initial[149]=1;initial[256]=101;initial[258]=2;
  initial[2052]=2;initial[2056]=2;initial[2057]=0;links[1]=100;links[128]=101;links[149]=102;
  for(let i=0;i<100;i++)positions.set([.1+(i%7)*.1,.2+(i%3)*.1,.3,i%2===0?2:1],4*i);
  positions.set([1.2,.3,.2,0,8.2,.2,.2,0,9.2,1.2,1.2,0],400);
  const buffers=[initial,links,positions].map(data=>{const buffer=device.createBuffer({size:data.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});device.queue.writeBuffer(buffer,0,data);return buffer;});
  const read=device.createBuffer({size:4*(1400+325-1024),usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  const group=device.createBindGroup({layout,entries:buffers.map((buffer,binding)=>({binding,resource:{buffer}}))});
  try{
   let reference:Uint32Array|undefined;
   for(const groups of [-1,0,1,3,1024]){
    device.queue.writeBuffer(buffers[0]!,0,initial);device.queue.writeBuffer(buffers[1]!,0,links);
    const e=device.createCommandEncoder(),pass=e.beginComputePass();pass.setBindGroup(0,group);
    if(groups<=0){pass.setPipeline(pipelines[0]!);pass.dispatchWorkgroups(3);pass.setPipeline(pipelines[groups<0?3:1]!);pass.dispatchWorkgroups(3);}
    else{pass.setPipeline(pipelines[2]!);pass.dispatchWorkgroups(groups);}
    pass.end();e.copyBufferToBuffer(buffers[0]!,4096,read,0,read.size);device.queue.submit([e.finish()]);await read.mapAsync(GPUMapMode.READ);
    const actual=new Uint32Array(read.getMappedRange()).slice();read.unmap();
    if(!reference){
     reference=actual;assert.ok(actual.some(v=>v!==0));
     let coldNearest=false;
     for(let i=0;i<325;i++){
      const all=actual[i]!,hot=actual[1400-1024+i]!;
      assert.ok(hot<=all,"a hot-only nearest search cannot be closer than the full search");
      coldNearest ||= all>hot;
     }
     assert.ok(coldNearest,"the fixture must include nearest samples without erasure authority");
    }else assert.deepEqual(actual,reference,`${groups} groups`);
   }
  }finally{for(const buffer of buffers)buffer.destroy();read.destroy();}
 });
});
