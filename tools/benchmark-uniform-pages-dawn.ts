/** Fixed-resolution page addressing experiment. No production solver changes.
 * Measures repeated seven-point operator application, not a pressure solve or
 * MiniDam64 frame. All readbacks/compilation are outside GPU timestamp intervals.
 */
import assert from "node:assert/strict";
import { readFileSync, writeFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { pathToFileURL } from "node:url";
import { resolve } from "node:path";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { managedGPUDevice } from "../lib/core/gpu-compilation-manager";
import { planUniformPages, lookupUniformPage, uniformPageCellAddress,
  UNIFORM_PAGE_MISSING, type UniformPageCoordinate, type UniformPageEdge,
  type UniformPageLayout } from "../lib/methods/uniform/uniform-page-layout";
import { uniformDenseStencilWGSL, uniformPageStencilWGSL } from "../lib/methods/uniform/uniform-page-stencil.wgsl";

const arg = (name: string) => process.argv.find(s => s.startsWith(`--${name}=`))?.slice(name.length + 3);
const samples = Number(arg("samples") ?? 20), repeats = 32, warmup = 5;
assert.ok(Number.isInteger(samples) && samples >= 2 && samples <= 200);
const median = (a: number[]) => { const sorted = [...a].sort((x,y) => x-y); return (sorted[Math.floor((a.length-1)/2)]! + sorted[Math.floor(a.length/2)]!) / 2; };
const valueAt = (x: number, y: number, z: number) => ((x * 3 + y * 5 + z * 7) % 17) / 8;
const report = { revision: execFileSync("git", ["rev-parse", "HEAD"], {encoding:"utf8"}).trim(),
  sourceHashes: Object.fromEntries([
    "lib/methods/uniform/uniform-page-layout.ts",
    "lib/methods/uniform/uniform-page-stencil.wgsl.ts",
  ].map(path => [path, createHash("sha256").update(readFileSync(path)).digest("hex")])),
  note: "Experimental seven-point operator only; 64-cubed fixture is not MiniDam64. Same-input repeated dispatches; no pressure convergence or full-frame claim.",
  samples, repeats, warmup, adapter: {}, correctness: [] as unknown[], runs: [] as unknown[] };

function packedLayout(edge: UniformPageEdge): UniformPageLayout {
  const coordinates: UniformPageCoordinate[] = [];
  for (let z=0;z<64/edge;z++) for (let y=0;y<64/edge;y++) for (let x=0;x<64/edge;x++) coordinates.push([x-32/edge,y-32/edge,z-32/edge]);
  // Scramble allocation order so spatial neighbors are not adjacent slot IDs.
  coordinates.reverse();
  return planUniformPages(edge, coordinates.length, coordinates);
}
function fields(layout: UniformPageLayout): Float32Array {
  const b=layout.edge, result=new Float32Array(layout.capacity*b**3);
  for (const slot of layout.activeSlots) {
    const q=layout.coordinates[slot]!;
    for(let z=0;z<b;z++) for(let y=0;y<b;y++) for(let x=0;x<b;x++)
      result[slot*b**3+x+b*(y+b*z)]=valueAt(q[0]*b+x,q[1]*b+y,q[2]*b+z);
  }
  return result;
}
function verify(layout: UniformPageLayout, input: Float32Array, output: Float32Array): void {
  const b=layout.edge;
  const sample=(q:UniformPageCoordinate) => {
    const address=uniformPageCellAddress(q,b), slot=lookupUniformPage(layout,address.coordinate);
    return slot===UNIFORM_PAGE_MISSING?0:input[slot*b**3+address.localIndex]!;
  };
  for(const slot of layout.activeSlots) {
    const page=layout.coordinates[slot]!;
    for(let z=0;z<b;z++) for(let y=0;y<b;y++) for(let x=0;x<b;x++) {
      const q:UniformPageCoordinate=[page[0]*b+x,page[1]*b+y,page[2]*b+z];
      let sum=0;
      for(let axis=0;axis<3;axis++) for(const direction of [-1,1]) {
        const n=[...q] as [number,number,number]; n[axis]+=direction; sum+=sample(n);
      }
      const i=slot*b**3+x+b*(y+b*z);
      assert.equal(output[i],6*input[i]!-sum,`B${b} slot ${slot} local ${x},${y},${z}`);
    }
  }
}

await acquireWebGPUExclusiveLock("dawn-probe", "uniform fixed-resolution page stencil");
let device: GPUDevice | undefined;
try {
  const dawn=await import(pathToFileURL(resolve("node_modules/webgpu/index.js")).href);
  Object.assign(globalThis,dawn.globals);
  const gpu=createProcessRetainedDawnGPU(dawn,[`backend=${process.env.FLUID_WEBGPU_BACKEND??"metal"}`,"disable-dawn-features=timestamp_quantization"]);
  const adapter=await gpu.requestAdapter(); assert.ok(adapter); assert.ok(adapter.features.has("timestamp-query"));
  report.adapter={vendor:adapter.info.vendor,architecture:adapter.info.architecture,device:adapter.info.device,description:adapter.info.description};
  device=managedGPUDevice(await adapter.requestDevice({requiredFeatures:["timestamp-query"]}),{requireWorkerRealm:false});
  const d=device, errors:string[]=[];
  d.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);});
  async function run(arm:"dense"|UniformPageEdge, layout?:UniformPageLayout, timing=true) {
    const buffers:GPUBuffer[]=[];
    const buffer=(data:Float32Array|Uint32Array, extra=0) => {
      const b=d.createBuffer({size:Math.max(4,data.byteLength),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|extra});
      if(data.byteLength)d.queue.writeBuffer(b,0,data as Float32Array<ArrayBuffer>); buffers.push(b);return b;
    };
    const edge=typeof arm==="number"?arm:undefined;
    const input=layout?fields(layout):Float32Array.from({length:64**3},(_,i)=>valueAt(i%64-32,Math.floor(i/64)%64-32,Math.floor(i/4096)-32));
    const source=buffer(input), destination=buffer(new Float32Array(input.length).fill(NaN),GPUBufferUsage.COPY_SRC);
    const bindings:GPUBindGroupEntry[]=[{binding:0,resource:{buffer:source}},{binding:1,resource:{buffer:destination}}];
    if(layout) bindings.push({binding:2,resource:{buffer:buffer(layout.neighbors)}},{binding:3,resource:{buffer:buffer(layout.activeSlots)}});
    const module=d.createShaderModule({code:edge?uniformPageStencilWGSL(edge):uniformDenseStencilWGSL(64)});
    const pipeline=await d.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:edge?"paged":"dense"}});
    const group=d.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:bindings});
    const query=d.createQuerySet({type:"timestamp",count:2});
    const resolved=d.createBuffer({size:16,usage:GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC});buffers.push(resolved);
    const readback=d.createBuffer({size:input.byteLength,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});buffers.push(readback);
    const times=d.createBuffer({size:16,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});buffers.push(times);
    const gpuMs:number[]=[], wallMs:number[]=[];
    try {
      for(let iteration=0;iteration<(timing?samples+warmup:1);iteration++) {
        const start=performance.now(), encoder=d.createCommandEncoder();
        const pass=encoder.beginComputePass({timestampWrites:{querySet:query,beginningOfPassWriteIndex:0,endOfPassWriteIndex:1}});
        pass.setPipeline(pipeline);pass.setBindGroup(0,group);
        for(let repeat=0;repeat<(timing?repeats:1);repeat++) {
          if(edge)pass.dispatchWorkgroups((edge/4)**2,edge/4,layout!.activeSlots.length);
          else pass.dispatchWorkgroups(16,16,16);
        }
        pass.end();encoder.resolveQuerySet(query,0,2,resolved,0);encoder.copyBufferToBuffer(resolved,0,times,0,16);
        d.queue.submit([encoder.finish()]);await d.queue.onSubmittedWorkDone();const wall=performance.now()-start;
        await times.mapAsync(GPUMapMode.READ);const ticks=new BigUint64Array(times.getMappedRange());
        assert.ok(ticks[1]!>ticks[0]!);const elapsed=Number(ticks[1]!-ticks[0]!)/1e6;times.unmap();
        if(iteration>=warmup){gpuMs.push(elapsed/repeats);wallMs.push(wall/repeats);}
      }
      const encoder=d.createCommandEncoder();encoder.copyBufferToBuffer(destination,0,readback,0,input.byteLength);d.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);const output=new Float32Array(readback.getMappedRange()).slice();readback.unmap();
      if(layout)verify(layout,input,output);
      else for(let z=0;z<64;z++)for(let y=0;y<64;y++)for(let x=0;x<64;x++) {
        const i=x+64*(y+64*z);let sum=0;
        if(x>0)sum+=input[i-1]!;if(x<63)sum+=input[i+1]!;
        if(y>0)sum+=input[i-64]!;if(y<63)sum+=input[i+64]!;
        if(z>0)sum+=input[i-4096]!;if(z<63)sum+=input[i+4096]!;
        assert.equal(output[i],6*input[i]!-sum);
      }
      assert.deepEqual(errors,[]);
      return {arm,residentPages:layout?.activeSlots.length,fieldCapacityBytes:input.byteLength,
        metadataBytes:layout?layout.directory.byteLength+layout.neighbors.byteLength+layout.activeSlots.byteLength:0,
        checkedCells:layout?layout.activeSlots.length*layout.edge**3:input.length,
        medianGPU_ms:timing?median(gpuMs):undefined,medianWall_ms:timing?median(wallMs):undefined,gpuMs};
    } finally {query.destroy();for(const b of buffers)b.destroy();}
  }
  for(const edge of [16,32] as const) {
    // Deliberately disconnected and negative-world coverage, then a retired hole
    // and reused slot. Poisoned output ensures every active cell is written.
    const first=planUniformPages(edge,5,[[-1,0,0],[0,0,0],[0,-1,0],[1000000,0,0]]);
    const retired=planUniformPages(edge,5,[[0,0,0],[0,-1,0],[1000000,0,0]],first);
    const reused=planUniformPages(edge,5,[[0,-1,0],[0,0,0],[1000000,0,0],[0,0,-1]],retired);
    report.correctness.push(await run(edge,first,false),await run(edge,reused,false));
    for(const arm of ["dense",edge,edge,"dense"] as const) {
      const result=await run(arm,typeof arm==="number"?packedLayout(arm):undefined);
      report.runs.push(result);const {gpuMs: _, ...summary}=result;console.log(JSON.stringify(summary));
    }
  }
  if(arg("out"))writeFileSync(arg("out")!,JSON.stringify(report,null,2)+"\n");
} finally {device?.destroy();await releaseWebGPUExclusiveLock();}
