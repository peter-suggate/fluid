/**
 * Isolate counting overhead with the production meshAppend helper. This is a
 * synthetic contention benchmark, not a scene-build latency estimate: each
 * brick has 48 face-layer jobs and each job appends a fixed number of faces.
 * Paired ABBA samples use one device, buffers and warmed pipelines.
 *
 * WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js node --import tsx \
 *   tools/benchmark-svo-mesh-count.ts artifacts/svo-build-edit/mesh-count.json
 */
import assert from "node:assert/strict";
import { writeFileSync } from "node:fs";
import { svoSurfaceMeshWGSL } from "../lib/svo/features/primary-visibility/svo-surface-mesh";
import { createDawnRenderDevice } from "./svo-dry-frame-harness";

const source=svoSurfaceMeshWGSL(3,128,true,true,true);
const append=`fn meshAppend(${source.split("fn meshAppend(")[1]!.split("\n}")[0]!}\n}`;
assert.ok(append.includes("meshJobCount+=1u;"),"production count accumulator must be present");
const {device,adapterInfo,validationErrors}=await createDawnRenderDevice({
  modulePath:process.env.WEBGPU_NODE_MODULE??`${process.cwd()}/node_modules/webgpu/index.js`,
  label:"mesh count benchmark",requireTimestampQuery:true,
});
const bricks=8192,jobsPerBrick=48,samples=64,warmups=8;
const work=device.createBuffer({size:bricks*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC});
const state=device.createBuffer({size:1024,usage:GPUBufferUsage.STORAGE});
const arena=device.createBuffer({size:32,usage:GPUBufferUsage.STORAGE});
const queries=device.createQuerySet({type:"timestamp",count:samples*2});
const resolved=device.createBuffer({size:samples*16,usage:GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC});
const read=device.createBuffer({size:samples*16+bricks*4,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
const header=/* wgsl */ `
struct SurfaceQuad { origin:vec3u, identity:u32, extent:vec3u, face:u32 }
@group(0) @binding(0) var<storage,read_write> meshWork:array<atomic<u32>>;
@group(0) @binding(1) var<storage,read_write> meshState:array<atomic<u32>>;
@group(0) @binding(2) var<storage,read_write> meshArena:array<SurfaceQuad>;
var<private> meshPhase:u32;var<private> meshJobCount:u32;var<private> meshEntry:u32;
var<private> meshEmitBase:u32;var<private> meshEmitCount:u32;
fn meshScratchAt(i:u32)->u32{return i;}
fn meshListAt(i:u32)->u32{return 0u;}
fn meshTableAt(i:u32)->u32{return 0u;}
fn sceneIdentityMaterial(i:u32)->u32{return i&65535u;}
struct Material {identity:vec4u}
fn dryMaterial(i:u32)->Material{return Material(vec4u(0u));}
`;
const summary=(values:number[])=>{
  const sorted=values.toSorted((a,b)=>a-b);
  return {median_ms:(sorted[sorted.length/2-1]!+sorted[sorted.length/2]!)/2,
    p95_ms:sorted[Math.ceil(sorted.length*.95)-1],samples_ms:values};
};
const rows=[];
try {
  for (const facesPerJob of [1,8,64]) {
    const pipelines=await Promise.all([false,true].map(async batched=>{
      const body=batched?append:append.replace("meshJobCount+=1u;","atomicAdd(&meshWork[meshScratchAt(meshEntry)],1u);");
      const module=device.createShaderModule({code:`${header}${body}
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3u){
 meshPhase=0u;meshEntry=id.x/${jobsPerBrick}u;meshJobCount=0u;
 for(var i=0u;i<${facesPerJob}u;i++){meshAppend(vec3u(i),vec3u(1u),0u,1u);}
 ${batched?"if(meshJobCount!=0u){atomicAdd(&meshWork[meshScratchAt(meshEntry)],meshJobCount);}":""}
}`});
      assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==="error"),[]);
      return device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"main"}});
    }));
    const groups=pipelines.map(pipeline=>device.createBindGroup({layout:pipeline.getBindGroupLayout(0),
      entries:[work,state,arena].map((buffer,binding)=>({binding,resource:{buffer}}))}));
    const run=(encoder:GPUCommandEncoder,arm:number,sample?:number)=>{
      encoder.clearBuffer(work);
      const pass=encoder.beginComputePass(sample===undefined?{}:{timestampWrites:{querySet:queries,beginningOfPassWriteIndex:sample*2,endOfPassWriteIndex:sample*2+1}});
      pass.setPipeline(pipelines[arm]!);pass.setBindGroup(0,groups[arm]!);pass.dispatchWorkgroups(bricks*jobsPerBrick/64);pass.end();
    };
    const warm=device.createCommandEncoder();for(let i=0;i<warmups;i++)run(warm,i%2);
    device.queue.submit([warm.finish()]);await device.queue.onSubmittedWorkDone();
    const encoder=device.createCommandEncoder();
    for(let i=0;i<samples;i++)run(encoder,i%4===0||i%4===3?0:1,i);
    encoder.resolveQuerySet(queries,0,samples*2,resolved,0);encoder.copyBufferToBuffer(resolved,0,read,0,samples*16);
    encoder.copyBufferToBuffer(work,0,read,samples*16,bricks*4);device.queue.submit([encoder.finish()]);
    await read.mapAsync(GPUMapMode.READ);
    const mapped=read.getMappedRange(),stamps=new BigUint64Array(mapped,0,samples*2);
    const timing:[number[],number[]]=[[],[]];
    for(let i=0;i<samples;i++)timing[i%4===0||i%4===3?0:1].push(Number(stamps[i*2+1]!-stamps[i*2]!)/1e6);
    assert.ok(new Uint32Array(mapped,samples*16,bricks).every(n=>n===jobsPerBrick*facesPerJob));
    read.unmap();
    rows.push({facesPerJob,referenceAtomics:bricks*jobsPerBrick*facesPerJob,candidateAtomics:bricks*jobsPerBrick,
      reference:summary(timing[0]),candidate:summary(timing[1])});
  }
  assert.deepEqual(validationErrors,[]);
  const report={scope:"Synthetic counting overhead only; excludes geometry extraction and whole scene latency",adapterInfo,bricks,jobsPerBrick,rows};
  if(process.argv[2])writeFileSync(process.argv[2],JSON.stringify(report,null,2));
  console.table(rows.map(row=>({facesPerJob:row.facesPerJob,reference_ms:row.reference.median_ms,candidate_ms:row.candidate.median_ms})));
} finally {work.destroy();state.destroy();arena.destroy();queries.destroy();resolved.destroy();read.destroy();device.destroy();}
