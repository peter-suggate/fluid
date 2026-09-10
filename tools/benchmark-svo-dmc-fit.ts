/** Interleaved GPU-only fitting comparison against the frozen pre-optimization fitter.
 * Run exclusively via this tool; it owns the standard WebGPU lease.
 */
import assert from "node:assert/strict";
import { writeFile, mkdir } from "node:fs/promises";
import { dirname } from "node:path";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { createDawnRenderDevice } from "./svo-dry-frame-harness";
import { svoDualMarchingCubesCachedFitWGSL } from "../lib/svo/features/meshing/dual-marching-cubes";
import { referenceDmcFitWGSL } from "../tests/fixtures/svo-dmc-fit-reference";

await acquireWebGPUExclusiveLock("dawn-probe", "benchmark-svo-dmc-fit");
let device: GPUDevice | undefined;
try {
  const dawn=await createDawnRenderDevice({requireTimestampQuery:true});
  device=dawn.device;
  const dimension = Number(process.env.DMC_FIT_DIMENSION ?? 64);
  const cycles = Number(process.env.DMC_FIT_CYCLES ?? 6);
  assert.ok(Number.isInteger(dimension) && dimension >= 8 && dimension <= 128 && dimension%4 === 0);
  assert.ok(Number.isInteger(cycles) && cycles >= 2 && cycles <= 30);
  const count = dimension ** 3;
  const prefix = `
@group(0) @binding(0) var<storage,read_write> output:array<vec4f>;
// A union of curved and sharp features, plus a procedural terrain surface.
// Function evaluations stay identical between the two fitting implementations.
fn dcField(p:vec3f,dirty:u32,candidateCount:u32)->f32{
  let q=p/vec3f(${dimension}.0)*64.;var value=q.y-10.-2.*sin(q.x*.19)*cos(q.z*.17);
  for(var i=0u;i<8u;i+=1u){let centre=vec3f(8.+f32(i%4u)*16.,20.+f32(i/4u)*18.,15.+f32(i/4u)*31.);
    let r=q-centre;
    if((i&1u)==0u){value=min(value,length(r)-9.3);}
    else{let d=abs(r)-vec3f(5.1,8.2,6.3);value=min(value,length(max(d,vec3f(0)))+min(max(d.x,max(d.y,d.z)),0.));}
  }
  return value*${dimension}.0/64.;
}
`;
  const suffix = `
@compute @workgroup_size(256) fn fit(@builtin(global_invocation_id) id:vec3u){
  let i=id.x;if(i>=${count}u){return;}
  let base=vec3f(f32(i%${dimension}u),f32((i/${dimension}u)%${dimension}u),f32(i/${dimension*dimension}u));
  let value=dmcFit(base,vec3f(1),0u,0u);output[i]=vec4f(value.point,value.value);
}
`;
  const output = device.createBuffer({ size: count*16, usage: GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC });
  const read = device.createBuffer({ size: count*16, usage: GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ });
  const tiledSuffix = `
@compute @workgroup_size(64) fn fit(@builtin(global_invocation_id) id:vec3u){
  let tile=id.x/64u;let lane=id.x%64u;let tiles=${dimension/4}u;
  let tileBase=vec3u(tile%tiles,(tile/tiles)%tiles,tile/(tiles*tiles))*4u;
  let base=tileBase+vec3u(lane%4u,(lane/4u)%4u,lane/16u);
  dmcCacheSamples(vec3f(tileBase),vec3f(1),0u,0u,lane,true);
  let value=dmcFit(vec3f(base),vec3f(1),0u,0u);
  let i=base.x+${dimension}u*base.y+${dimension*dimension}u*base.z;output[i]=vec4f(value.point,value.value);
}
`;
  const arms = await Promise.all([referenceDmcFitWGSL,svoDualMarchingCubesCachedFitWGSL].map(async (source,arm) => {
    const module = device!.createShaderModule({ code: prefix+source+(arm===0?suffix:tiledSuffix) });
    assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type === "error"), []);
    const pipeline = await device!.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "fit" } });
    const bind = device!.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: output } }] });
    return { pipeline, bind };
  }));
  const queries=device.createQuerySet({type:"timestamp",count:2});
  const queryResolve=device.createBuffer({size:16,usage:GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC});
  const queryRead=device.createBuffer({size:16,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  const run = async (arm: number, capture = false) => {
    const encoder=device!.createCommandEncoder();const pass=encoder.beginComputePass({timestampWrites:{querySet:queries,beginningOfPassWriteIndex:0,endOfPassWriteIndex:1}});
    pass.setPipeline(arms[arm].pipeline);pass.setBindGroup(0,arms[arm].bind);pass.dispatchWorkgroups(Math.ceil(count/(arm===0?256:64)));pass.end();
    if(capture)encoder.copyBufferToBuffer(output,0,read,0,count*16);
    encoder.resolveQuerySet(queries,0,2,queryResolve,0);encoder.copyBufferToBuffer(queryResolve,0,queryRead,0,16);
    device!.queue.submit([encoder.finish()]);await queryRead.mapAsync(GPUMapMode.READ);
    const stamps=new BigUint64Array(queryRead.getMappedRange().slice(0));queryRead.unmap();return Number(stamps[1]-stamps[0])/1e6;
  };
  const captures: Float32Array[]=[];
  for(let arm=0;arm<2;arm++){
    await run(arm,true);await read.mapAsync(GPUMapMode.READ);captures.push(new Float32Array(read.getMappedRange().slice(0)));read.unmap();
    await run(arm);await run(arm);
  }
  let maxPointDelta=0,maxValueDelta=0,changedSigns=0,nearCells=0;
  for(let i=0;i<count;i++){
    const a=captures[0],b=captures[1],at=i*4;
    assert.ok(Number.isFinite(b[at])&&Number.isFinite(b[at+1])&&Number.isFinite(b[at+2])&&Number.isFinite(b[at+3]));
    if(Math.min(Math.abs(a[at+3]),Math.abs(b[at+3]))<2){nearCells++;}
    {
      maxPointDelta=Math.max(maxPointDelta,Math.hypot(a[at]-b[at],a[at+1]-b[at+1],a[at+2]-b[at+2]));
      maxValueDelta=Math.max(maxValueDelta,Math.abs(a[at+3]-b[at+3]));
      if((a[at+3]<0)!==(b[at+3]<0))changedSigns++;
    }
  }
  const times: number[][]=[[],[]];
  for(let cycle=0;cycle<cycles;cycle++)for(const arm of cycle%2?[1,0]:[0,1])times[arm].push(await run(arm));
  const median=(values:number[])=>[...values].sort((a,b)=>a-b)[Math.floor(values.length/2)];
  const report={dimension,cells:count,cycles,method:"interleaved-GPU-timestamps",reference_ms:median(times[0]),optimized_ms:median(times[1]),speedup:median(times[0])/median(times[1]),samples_ms:times,nearCells,maxPointDelta,maxValueDelta,changedSigns};
  console.log(JSON.stringify(report,null,2));
  const path=process.env.DMC_FIT_OUT??"/tmp/dmc-fit-benchmark.json";await mkdir(dirname(path),{recursive:true});await writeFile(path,JSON.stringify(report,null,2)+"\n");
  // Integer/half-integer sample coordinates are exactly representable here.
  // Sharing those samples must preserve the frozen fitter output exactly.
  assert.equal(maxPointDelta,0,"shared samples preserve fitted positions");
  assert.equal(maxValueDelta,0,"shared samples preserve fitted scalars");
  assert.equal(changedSigns,0,"fitting optimization preserves dual topology signs");
  assert.deepEqual(dawn.validationErrors,[],"GPU validation errors");
  queries.destroy();queryResolve.destroy();queryRead.destroy();read.destroy();output.destroy();
} finally {device?.destroy();await releaseWebGPUExclusiveLock();}
