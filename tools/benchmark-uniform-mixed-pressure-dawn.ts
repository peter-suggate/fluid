/** Isolated pressure-sweep scaling probe, not an active simulation benchmark.
 * Same operators/work per owner across layouts; includes command encoding and
 * fenced GPU execution, excludes allocation/compilation. No CPU face oracle. */
import assert from "node:assert/strict";
import { pathToFileURL } from "node:url";
import { resolve } from "node:path";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { createUniformMixedLayout, mixedCellWidth, MIXED_CELL_MASK } from "../lib/methods/uniform/uniform-mixed-layout";
import { UniformMixedOwnership } from "../lib/methods/uniform/uniform-mixed-ownership";
import { UniformMixedPressureLevelStage } from "../lib/methods/uniform/uniform-mixed-pressure-stage";

const median = (values: number[]) => [...values].sort((a,b) => a-b)[Math.floor(values.length/2)]!;
const surface = process.argv.includes("--surface");
const size = Number(process.argv.find(a => a.startsWith("--size="))?.slice(7) ?? 64);
assert.ok(Number.isSafeInteger(size) && size >= 16 && size % 16 === 0, "size must be a positive multiple of 16");
await acquireWebGPUExclusiveLock("dawn-benchmark", "Uniform mixed pressure sweep scaling");
let device: GPUDevice | undefined;
try {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE ?? resolve("node_modules/webgpu/index.js")).href);
  Object.assign(globalThis, dawn.globals);
  const gpu = createProcessRetainedDawnGPU(dawn, ["backend=metal"]), adapter = await gpu.requestAdapter(); assert.ok(adapter);
  device = await adapter.requestDevice();
  const errors: string[] = []; device.addEventListener("uncapturederror", e => { e.preventDefault(); errors.push(e.error.message); });
  const runs: { mode: string; cells: number; fineTiles: number; transitionTiles: number; coarseTiles: number; topologyBytes: number; scratchBytes: number; times: number[] }[] = [];
  // ABCCBA controls drift. Both fine and coarse endpoints use the mixed stage:
  // this isolates ownership scaling, not comparison against native fine kernels.
  for (const mode of ["fine", "mixed", "coarse", "coarse", "mixed", "fine"]) {
    const layout = createUniformMixedLayout({ dimensions: [size,size,size], cellSize_m: [1,1,1], origin_m: {x:0,y:0,z:0} },
      mode === "coarse" ? [] : [{id:"fine",rule:"minimum-cell-size",minimumCellSize_cells:1,maximumCellSize_cells:1,
        min_m:{x:0,y:0,z:0},max_m: mode === "fine" ? {x:size,y:size,z:size} : {x:size/2,y:size/2,z:size/2}}], true, 4);
    const ownership = new UniformMixedOwnership(device, layout), n = layout.cellCount;
    const buffers = [4*n,16*n,4*n,4*n,4*n,...(surface?[4*n]:[])].map(bytes => device!.createBuffer({size:bytes,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}));
    try {
      const stage = new UniformMixedPressureLevelStage(device, ownership, false, surface); await stage.initialize();
      const group = stage.bind({pressure:{buffer:buffers[0]!},slopes:{buffer:buffers[1]!},rhs:{buffer:buffers[2]!},frozen:{buffer:buffers[3]!},result:{buffer:buffers[4]!},...(surface?{phi:{buffer:buffers[5]!}}:{})});
      if(surface){
        const phi=new Float32Array(n),tilesPerAxis=size/4;
        layout.tiles.forEach((word,tile)=>{
          const width=mixedCellWidth(word),side=4/width,originY=Math.floor(tile/tilesPerAxis)%tilesPerAxis*4;
          for(let lane=0;lane<side**3;lane++)phi[(word&MIXED_CELL_MASK)+lane]=originY+(Math.floor(lane/side)%side+.5)*width-(size*.5+.25);
        });
        device.queue.writeBuffer(buffers[5]!,0,phi);
      }
      const rhs = Float32Array.from({length:n},(_,i)=>Math.sin(i*13)); device.queue.writeBuffer(buffers[2]!,0,rhs);
      const times: number[] = [], sweeps = 6;
      for (let batch=0;batch<15;batch++) {
        await device.queue.onSubmittedWorkDone();
        const start = performance.now(), encoder = device.createCommandEncoder();
        for (let i=0;i<sweeps;i++) stage.encodeSweep(encoder,group);
        device.queue.submit([encoder.finish()]); await device.queue.onSubmittedWorkDone();
        if(batch>=5)times.push((performance.now()-start)/sweeps);
      }
      const row = {mode,cells:n,fineTiles:layout.fineTiles.length,transitionTiles:layout.transitionTiles.length,coarseTiles:layout.coarseTiles.length,
        topologyBytes:layout.metadataBytes+16,scratchBytes:buffers.reduce((sum,b)=>sum+b.size,0),times};
      runs.push(row); console.log(JSON.stringify({...row,medianSweepMs:median(times)}));
    } finally { buffers.forEach(b=>b.destroy()); ownership.destroy(); }
  }
  const timings = Object.fromEntries(["fine","mixed","coarse"].map(mode=>[mode,median(runs.filter(r=>r.mode===mode).flatMap(r=>r.times))]));
  console.log(JSON.stringify({scope:`isolated ${surface?"free-surface":"closed-liquid"} pressure sweep; mixed kernels at all endpoints`,size,timings,
    mixedToFineRatio:timings.mixed!/timings.fine!,coarseToFineRatio:timings.coarse!/timings.fine!}));
  assert.deepEqual(errors,[]);
} finally {device?.destroy(); await releaseWebGPUExclusiveLock();}
