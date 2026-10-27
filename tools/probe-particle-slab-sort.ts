/** Bounded, identical-input production slab sort comparison. Baseline is an
 * exported pre-distribution slab module (and its canonical module beside it).
 * CPU truth precedes GPU work. Run under fluid-gpu-run.sh timing. */
import assert from "node:assert/strict";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { particleSlabWGSL, particleSlabSortEntries, particleSlabWords, NB_SLAB_HEADER_WORDS } from "../lib/methods/uniform/uniform-particle-slab.wgsl";
import { narrowBandParticleWGSL } from "../lib/methods/uniform/uniform-narrow-band-apic.wgsl";
import { canonicalDispatchMergeCPU, compareParticleWords } from "./particle-canonical-reference";
import { withUniformDevice } from "../tests/helpers/uniform-geometric";
import { paceDawnEventPolling } from "../lib/harness/node-dawn-provider";
import { GPUPassProfile } from "./gpu-pass-profile";
const baseline=process.argv.find(a=>a.startsWith("--baseline="))?.slice(11);assert.ok(baseline);
const old=await import(pathToFileURL(resolve(baseline)).href) as {particleSlabWGSL:typeof particleSlabWGSL};
const capacity=131072,cells=32768;
const fixtures=([ [256,false],[4096,false],[65537,false],[1025,true] ] as const).map(([count,apic])=>{
 const stride=apic?28:12;
 let seed=1234;
 const records=Array.from({length:count},(_,i)=>{
  seed^=seed<<13;seed^=seed>>>17;seed^=seed<<5;
  const f=new Float32Array(stride);
  f.set(i%2?[.25+(seed&255)/1024,.25+((seed>>>8)&255)/1024,.25+((seed>>>16)&255)/1024,1]:[.25,.25,.25,1]);
  for(let k=4;k<stride;k++)f[k]=(i*3571+k*17)%113;
  return new Uint32Array(f.buffer);
 });
 const expected=[...records].sort(compareParticleWords);
 assert.deepEqual(canonicalDispatchMergeCPU(records,[count]).records,expected);
 return {count,apic,stride,input:Uint32Array.from(records.flatMap(r=>[...r])),expected:Uint32Array.from(expected.flatMap(r=>[...r]))};
});
process.env.WEBGPU_NODE_MODULE??=resolve("node_modules/webgpu/index.js");paceDawnEventPolling();
const watchdog=setTimeout(()=>{console.error("Slab probe exceeded 30s");process.exit(2);},30000);watchdog.unref();
try {await withUniformDevice("dense slab sort fixed input",async device=>{
 const profile=new GPUPassProfile(device);
 try {for(const f of fixtures){
  const buffers=[4*particleSlabWords(1),4*f.stride*capacity,4*f.stride*capacity,4*(cells+7*capacity),4*36000].map(size=>device.createBuffer({size,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST}));
  const read=device.createBuffer({size:f.expected.byteLength,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  const layout=device.createBindGroupLayout({entries:buffers.map((_,binding)=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}}))});
  const pipelineLayout=device.createPipelineLayout({bindGroupLayouts:[layout]});
  const group=device.createBindGroup({layout,entries:buffers.map((buffer,binding)=>({binding,resource:{buffer}}))});
  try {
   const arms=await Promise.all([false,true].map(async distributed=>{
    const module=device.createShaderModule({code:`${narrowBandParticleWGSL(f.apic)}
@group(0) @binding(0) var<storage,read_write> state:array<atomic<u32>>;
@group(0) @binding(1) var<storage,read_write> source:array<Particle>;
@group(0) @binding(2) var<storage,read_write> particles:array<Particle>;
@group(0) @binding(3) var<storage,read_write> links:array<u32>;
@group(0) @binding(4) var<storage,read_write> bins:array<atomic<u32>>;
const UB_BRICKS:u32=64u;const UB_NONE:u32=0xffffffffu;const UB_PARTICLES_CORE:u32=1u;
const UM_CELL_KEYS:u32=${cells}u;const UM_TILES:u32=512u;const NB_LINKS:u32=UM_CELL_KEYS;
const NB_OCCUPIED:u32=32768u;const NB_COVERAGE:u32=33300u;
const NB_SURFACE_TILES:u32=33812u;const NB_ACTIVITY_SURFACE_HEAT:u32=34324u;const NB_ACTIVITY_THETA:u32=34836u;
fn ubCount()->u32{return 1u;}fn ubBrick(slot:u32)->u32{return 0u;}
fn ubBrickAt(c:vec3i)->u32{return 0u;}fn ubBrickOrigin(id:u32)->vec3u{return vec3u(0);}
fn ubMark(id:u32,kind:u32){}fn umFramePage()->vec3i{return vec3i(0);}
fn umCorner(k:u32,n:u32)->vec3u{return vec3u(k%n,(k/n)%n,k/(n*n));}
fn umTileLocal(c:vec3u)->u32{return c.x+8u*c.y+64u*c.z;}
fn umTileKey(c:vec3i)->u32{return 0u;}fn umKeyStored(key:u32)->bool{return true;}
fn nbUnmarked(c:vec3i){}fn nbCover(tile:u32,n:u32){atomicStore(&bins[NB_COVERAGE+tile],n);}
fn markSurfaceTiles(q:vec3f,heat:f32){}
${(distributed?particleSlabWGSL:old.particleSlabWGSL)(capacity-4096,4096,f.apic)}`});
    assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==="error"),[]);
    return Promise.all((distributed?particleSlabSortEntries(capacity):["slabSort"]).map(name=>device.createComputePipelineAsync({layout:pipelineLayout,compute:{module,entryPoint:name.split("@")[0],...(name.includes("@")?{constants:{nbCanonicalWidth:Number(name.split("@")[1])}}:{})}})));
   }));
   device.queue.writeBuffer(buffers[1]!,0,f.input);
   const receipt=new Uint32Array(particleSlabWords(1));
   receipt[0]=f.count+19;receipt[17]=1;receipt[20]=1;
   receipt[NB_SLAB_HEADER_WORDS+2]=f.count;receipt[NB_SLAB_HEADER_WORDS+5]=0;
   receipt[NB_SLAB_HEADER_WORDS+6]=f.count+19;receipt[NB_SLAB_HEADER_WORDS+7]=f.count;
   const timings:number[][]=[[],[]];
   for(let repeat=0;repeat<7;repeat++)for(let turn=0;turn<2;turn++){
    const arm=(repeat+turn)%2;
    device.queue.writeBuffer(buffers[0]!,0,receipt);
    if(repeat)profile.start();
    const encoder=profile.device.createCommandEncoder(),pass=encoder.beginComputePass({label:arm?"distributed slab sort":"single-workgroup slab sort"});pass.setBindGroup(0,group);
    for(const pipeline of arms[arm]!){pass.setPipeline(pipeline);pass.dispatchWorkgroups(1024);}
    pass.end();encoder.copyBufferToBuffer(buffers[2]!,0,read,0,read.size);device.queue.submit([encoder.finish()]);
    await read.mapAsync(GPUMapMode.READ);assert.deepEqual(new Uint32Array(read.getMappedRange()),f.expected,`${f.count}, arm ${arm}`);read.unmap();
    if(repeat)timings[arm]!.push((await profile.finish())[0]!.ms);
   }
   const median=timings.map(t=>[...t].sort((a,b)=>a-b).slice(2,4).reduce((a,b)=>a+b)/2);
   console.log(JSON.stringify({count:f.count,apic:f.apic,everyOutputMatchesCPU:true,old_ms:median[0],distributed_ms:median[1],speedup:median[0]!/median[1]!,samples:timings}));
  } finally {read.destroy();for(const b of buffers)b.destroy();}
 }}finally{profile.destroy();}
},["timestamp-query"],["disable-dawn-features=timestamp_quantization"]);}finally{clearTimeout(watchdog);}
