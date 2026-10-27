/** Identical-input, bounded surface-scatter comparison. CPU partition checks
 * live in uniform-surface-splat-fusion.test.ts; both GPU outputs must match.
 * Run under the timing lock with --baseline=<exported old splat module>. */
import assert from "node:assert/strict";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { narrowBandSplatWGSL } from "../lib/methods/uniform/uniform-narrow-band-splat.wgsl";
import { withUniformDevice } from "../tests/helpers/uniform-geometric";
import { paceDawnEventPolling } from "../lib/harness/node-dawn-provider";
import { GPUPassProfile } from "./gpu-pass-profile";
const baseline=process.argv.find(a=>a.startsWith("--baseline="))?.slice(11);assert.ok(baseline);
const old=await import(pathToFileURL(resolve(baseline)).href) as {narrowBandSplatWGSL:string};
const prefix=/* wgsl */`
const UM_CELL_KEYS=192u;const UM_TILES=3u;const NB_COVERAGE=256u;const NB_DEPTH_A=1024u;const NB_SURFACE_HOT_NEAREST=1400u;const NB_BAND=2048u;const NB_OCCUPIED=2052u;
const nbSparseBins=true;
const NB_LINKS=UM_CELL_KEYS;fn nbBinTile(t:u32)->u32{return 64u*t;}fn nbBinCell(c:u32)->u32{return c;}
@group(0) @binding(0) var<storage,read_write> bins:array<atomic<u32>>;
@group(0) @binding(1) var<storage,read_write> links:array<u32>;
@group(0) @binding(2) var<storage,read> particles:array<vec4f>;
fn umTileLive(t:u32)->bool{return t<3u;}fn umLiveTileCount()->u32{return 3u;}fn umLiveTile(j:u32)->u32{return j;}
fn nbTileCold(t:u32)->bool{return t==2u;}fn umTileCoord(t:u32)->vec3u{return vec3u(t,0,0);}
fn umCorner(j:u32,n:u32)->vec3u{return vec3u(j%n,(j/n)%n,j/(n*n));}
fn umVertexPresent(p:vec3i)->bool{return all(p>=vec3i(0))&&all(p<vec3i(13,5,5));}
fn umVertexAt(p:vec3u)->u32{return p.x+13u*(p.y+5u*p.z);}
fn umVertexSlot(p:vec3i)->vec4u{return vec4u(vec3u(p),u32(umVertexPresent(p)));}
fn umBlockStored(b:u32)->bool{return b!=0u;}fn umVertexSlotAt(p:vec4u)->u32{return umVertexAt(p.xyz);}
fn umLoadAddressPlan(b:u32,lane:u32){}
fn nbPosition(i:u32)->vec3f{return particles[i].xyz;}fn nbSurfaceParticleHot(i:u32)->bool{return particles[i].w>1.0;}
`;
process.env.WEBGPU_NODE_MODULE??=resolve("node_modules/webgpu/index.js");paceDawnEventPolling();
const watchdog=setTimeout(()=>{console.error("Surface probe exceeded 30s");process.exit(2);},30000);watchdog.unref();
try{await withUniformDevice("surface scatter fixed input",async device=>{
 const profile=new GPUPassProfile(device);
 try{
  const layout=device.createBindGroupLayout({entries:[0,1,2].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:binding===2?"read-only-storage":"storage"}}))});
  const pipelineLayout=device.createPipelineLayout({bindGroupLayouts:[layout]});
  const arms=await Promise.all([false,true].map(async chunked=>{
   const module=device.createShaderModule({code:prefix+(chunked?narrowBandSplatWGSL:old.narrowBandSplatWGSL)});
   assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==="error"),[]);
   return Promise.all((chunked?["surfaceSplatPlan","surfaceSplatOccupied"]:["surfaceSplatOccupied"]).map(entryPoint=>device.createComputePipelineAsync({layout:pipelineLayout,compute:{module,entryPoint}})));
  }));
  for(const dense of [256,65537]){
   const total=dense+3,initial=new Uint32Array(2100),links=new Uint32Array(192+7*total),positions=new Float32Array(4*total);
   initial[0]=dense;initial[1]=1;initial[128]=1;initial[149]=1;initial[256]=dense+1;initial[258]=2;
   initial[2052]=2;initial[2056]=2;initial[2057]=0;
   let first=0;for(let c=0;c<192;c++){links[c]=first;first+=initial[c]!;}
   for(let i=0;i<dense;i++)positions.set([.1+(i%7)*.1,.2+(i%3)*.1,.3,i%2===0?2:1],4*i);
   positions.set([1.2,.3,.2,0,8.2,.2,.2,0,9.2,1.2,1.2,0],4*dense);
   const buffers=[initial,links,positions].map(data=>{const b=device.createBuffer({size:data.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});device.queue.writeBuffer(b,0,data);return b;});
   const read=device.createBuffer({size:4*(1400+325-1024),usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
   const group=device.createBindGroup({layout,entries:buffers.map((buffer,binding)=>({binding,resource:{buffer}}))});
   try{
    let reference:Uint32Array|undefined;const timings:number[][]=[[],[]];
    for(let repeat=0;repeat<7;repeat++)for(let turn=0;turn<2;turn++){
     const arm=(repeat+turn)%2;device.queue.writeBuffer(buffers[0]!,0,initial);
     if(repeat)profile.start();
     const encoder=profile.device.createCommandEncoder(),pass=encoder.beginComputePass({label:arm?"chunked splat":"single-tile splat"});pass.setBindGroup(0,group);
     for(const pipeline of arms[arm]!){pass.setPipeline(pipeline);pass.dispatchWorkgroups(1024);}
     pass.end();encoder.copyBufferToBuffer(buffers[0]!,4096,read,0,read.size);device.queue.submit([encoder.finish()]);
     await read.mapAsync(GPUMapMode.READ);const actual=new Uint32Array(read.getMappedRange()).slice();read.unmap();
     if(!reference){reference=actual;assert.ok(actual.some(x=>x!==0));}else assert.deepEqual(actual,reference);
     if(repeat)timings[arm]!.push((await profile.finish())[0]!.ms);
    }
    const median=timings.map(t=>[...t].sort((a,b)=>a-b).slice(2,4).reduce((a,b)=>a+b)/2);
    console.log(JSON.stringify({dense,everyOutputMatches:true,old_ms:median[0],chunked_ms:median[1],speedup:median[0]!/median[1]!,samples:timings}));
   }finally{read.destroy();for(const b of buffers)b.destroy();}
  }
 }finally{profile.destroy();}
},["timestamp-query"],["disable-dawn-features=timestamp_quantization"]);}finally{clearTimeout(watchdog);}
