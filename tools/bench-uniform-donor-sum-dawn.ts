/** Dawn validation and timings for the production donor accumulator.
 * Optional --edges=/path/to/frame-edges.bin replays probe captures; otherwise
 * exercises ordinary fan-in, concentrated fan-in, tiny terms and rounding ties.
 */
import assert from "node:assert/strict";
import { readFile, writeFile } from "node:fs/promises";
import { pathToFileURL } from "node:url";
import { resolve } from "node:path";
import { createProcessRetainedDawnGPU, type NodeDawnProvider } from "../lib/harness/node-dawn-provider";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { uniformVolumeDonorSumWGSL } from "../lib/methods/uniform/uniform-volume-donor-sum.wgsl";
const arg=(key:string)=>process.argv.find(a=>a.startsWith(`--${key}=`))?.slice(key.length+3);
const edgePath=arg("edges");
const code=/* wgsl */ `
struct Edge {donor:array<u32,9>,weight:array<f32,9>,padding:vec2u}
@group(0) @binding(0) var<storage,read> edges:array<Edge>;
@group(0) @binding(1) var<storage,read_write> rigidExchange:array<atomic<i32>>;
@group(0) @binding(2) var<storage,read_write> output:array<f32>;
fn cellCount()->u32{return arrayLength(&rigidExchange)/6u;}
fn dims()->vec3i{return vec3i(64,64,i32(cellCount()/4096u));}
${uniformVolumeDonorSumWGSL}
@compute @workgroup_size(64)
fn deposit(@builtin(global_invocation_id) id:vec3u){
 if(id.x>=cellCount()){return;}
 for(var k=0u;k<9u;k++){uvAddDonor(edges[id.x].donor[k],edges[id.x].weight[k]);}
}
@compute @workgroup_size(64)
fn decode(@builtin(global_invocation_id) id:vec3u){
 if(id.x<cellCount()){output[id.x]=uvDonorSum(id.x);}
}`;
// Independent exact CPU oracle: sum integer multiples of the smallest f32,
// then round using quotient/remainder rather than the GPU's limb extraction.
function roundedBits(sum:bigint):number {
 if(sum<0x800000n)return Number(sum);
 const shift=BigInt(sum.toString(2).length-24);
 let q=sum>>shift;
 if(shift>0n){const r=sum-(q<<shift),half=1n<<(shift-1n);if(r>half||(r===half&&(q&1n)!==0n))q++;}
 let exponent=Number(shift)+1;
 if(q===0x1000000n){q>>=1n;exponent++;}
 return (exponent<<23)|(Number(q)&0x7fffff);
}
await acquireWebGPUExclusiveLock("dawn-probe","bounded donor sum");
let device:GPUDevice|undefined;
try {
 const dawn=await import(pathToFileURL(resolve("node_modules/webgpu/index.js")).href) as NodeDawnProvider;
 Object.assign(globalThis,dawn.globals);
 const gpu=createProcessRetainedDawnGPU(dawn,["backend=metal"]);const adapter=await gpu.requestAdapter();assert.ok(adapter);
 device=await adapter.requestDevice({requiredFeatures:["timestamp-query"],requiredLimits:requiredFluidDeviceLimits(adapter.limits)});
 const errors:string[]=[];device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);});
 const module=device.createShaderModule({code});
 assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==="error"),[]);
 const pipelines=await Promise.all(["deposit","decode"].map(entryPoint=>device!.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint}})));
 const reports=[];
 for(const scenario of edgePath?["capture"]:["ordinary","concentrated","rounding"]){
  let raw:ArrayBuffer;
  if(edgePath){const bytes=await readFile(edgePath);raw=bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength) as ArrayBuffer;}
  else {
   const count=65536;raw=new ArrayBuffer(count*80);const u=new Uint32Array(raw),f=new Float32Array(raw);
   for(let i=0;i<count;i++)for(let k=0;k<9;k++){
    u[20*i+k]=scenario==="concentrated"?0:(i+k)%count;
    f[20*i+9+k]=scenario==="rounding"?0:Math.fround((1+(i*17+k*31)%997)/997);
   }
   if(scenario==="rounding"){
    // Subnormals, smallest normal, cross-limb carries, even/odd rounding ties.
    const cases=[[1,1,0x7ffffe],[0x3f800000,0x33800000],[0x3f800001,0x33800000],[0x3f800000,0x33800000,1],[0x1f800000,1],[0x3f7fffff,0x33800000]];
    cases.forEach((bits,donor)=>bits.forEach((b,k)=>{u[20*donor+k]=donor;u[20*donor+9+k]=b;}));
   }
  }
  const count=raw.byteLength/80;assert.ok(Number.isInteger(count));
  const u=new Uint32Array(raw),exact:bigint[]=Array(count).fill(0n);
  for(let i=0;i<count;i++)for(let k=0;k<9;k++){
   const bits=u[20*i+9+k]!,exponent=bits>>>23;assert.ok(exponent<=127);const donor=u[20*i+k]!;assert.ok(donor<count);
   const mantissa=(bits&0x7fffff)|(exponent?0x800000:0);
   exact[donor]!+=BigInt(mantissa)<<BigInt(Math.max(0,exponent-1));
  }
  const expected=Uint32Array.from(exact,roundedBits);
  const edges=device.createBuffer({size:raw.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});device.queue.writeBuffer(edges,0,raw);
  const sums=device.createBuffer({size:count*24,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
  const output=device.createBuffer({size:count*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
  const read=device.createBuffer({size:count*4,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
  const groups=pipelines.map((p,i)=>device!.createBindGroup({layout:p.getBindGroupLayout(0),entries:[{binding:1,resource:{buffer:sums}},i===0?{binding:0,resource:{buffer:edges}}:{binding:2,resource:{buffer:output}}]}));
  const query=device.createQuerySet({type:"timestamp",count:4});
  const timing=device.createBuffer({size:256,usage:GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC});
  const timingRead=device.createBuffer({size:256,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  for(let repeat=0;repeat<5;repeat++){
   const encoder=device.createCommandEncoder();encoder.clearBuffer(sums);
   for(let i=0;i<2;i++){const pass=encoder.beginComputePass({timestampWrites:{querySet:query,beginningOfPassWriteIndex:2*i,endOfPassWriteIndex:2*i+1}});pass.setPipeline(pipelines[i]!);pass.setBindGroup(0,groups[i]!);pass.dispatchWorkgroups(Math.ceil(count/64));pass.end();}
   encoder.copyBufferToBuffer(output,0,read,0,count*4);encoder.resolveQuerySet(query,0,4,timing,0);encoder.copyBufferToBuffer(timing,0,timingRead,0,32);
   device.queue.submit([encoder.finish()]);await read.mapAsync(GPUMapMode.READ);
   const actual=new Uint32Array(read.getMappedRange());let mismatches=0;
   for(let i=0;i<count;i++)if(actual[i]!==expected[i])mismatches++;
   read.unmap();assert.equal(mismatches,0,`${scenario}: exact rounded donor sums`);
   await timingRead.mapAsync(GPUMapMode.READ);const ts=new BigUint64Array(timingRead.getMappedRange());
   const ms=Number(ts[1]!-ts[0]!+ts[3]!-ts[2]!)/1e6;timingRead.unmap();
   const row={scenario,edgePath,count,repeat,ms,mismatches};reports.push(row);console.log(JSON.stringify(row));
  }
  for(const buffer of [edges,sums,output,read,timing,timingRead])buffer.destroy();query.destroy();
 }
 assert.deepEqual(errors,[]);
 if(arg("out"))await writeFile(arg("out")!,JSON.stringify(reports,null,2));
}finally{device?.destroy();await releaseWebGPUExclusiveLock();}
