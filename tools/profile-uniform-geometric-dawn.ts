/** Production Uniform Geometric stage timestamps, without rendering or xctrace.
 * node --import tsx tools/profile-uniform-geometric-dawn.ts
 * Options: --scene=cm12-figure-7-256 --frames=60 --out=/tmp/profile.json
 * --allocation-audit --max-gpu-bytes=3000000000 --scratch-storage=separate
 */
import assert from "node:assert/strict";
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { managedGPUDevice, gpuCompilationManagerFor } from "../lib/core/gpu-compilation-manager";
import { GPUStageTimestampRecorder } from "../lib/core/performance-trace";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { resolveMethodValues } from "../lib/core/method-contract";
import { usePerformanceInstrumentationStore } from "../lib/core/stores/performance-instrumentation-store";
import { createProcessRetainedDawnGPU, type NodeDawnProvider } from "../lib/harness/node-dawn-provider";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { uniformVolumeMethod } from "../lib/methods/uniform/uniform-volume-method";
import { WebGPUUniformReferenceSolver } from "../lib/methods/uniform/webgpu-uniform-reference";
import { uniformGeometricSolverOptions } from "../lib/methods/uniform/uniform-geometric-options";
import { auditUniformGPUAllocations } from "./uniform-gpu-allocation-audit";
const arg = (key: string, fallback: string) => process.argv.find(a => a.startsWith(`--${key}=`))?.slice(key.length + 3) ?? fallback;
const sceneId = arg("scene", "cm12-figure-7-256");
const frames = Number(arg("frames", "60"));
const maxGPUBytes = Number(arg("max-gpu-bytes", "0"));
assert.ok(Number.isSafeInteger(maxGPUBytes) && maxGPUBytes >= 0);
assert.ok(Number.isInteger(frames) && frames > 4);
const out = resolve(arg("out", `/tmp/${sceneId}-stages.json`));
const stats = (values: number[]) => {
  const sorted = [...values].sort((a,b) => a-b);
  const quantile = (q: number) => sorted[Math.max(0, Math.ceil(sorted.length*q)-1)]!;
  return { n: values.length, mean: values.reduce((a,b) => a+b,0)/values.length,
    median: quantile(.5), p10: quantile(.1), p90: quantile(.9) };
};
const rows: { frame: number; time_s: number; wall_ms: number; trace: NonNullable<WebGPUUniformReferenceSolver["info"]["physicsTrace"]>; cpuTrace: unknown; work: Record<string,unknown> }[] = [];
await acquireWebGPUExclusiveLock("dawn-probe", `Uniform Geometric stage profile: ${sceneId}`);
let device: GPUDevice | undefined, solver: WebGPUUniformReferenceSolver | undefined;
let pressureWorkReadback: GPUBuffer | undefined;
let allocationAudit: ReturnType<typeof auditUniformGPUAllocations> | undefined;
try {
  const dawn = await import(pathToFileURL(resolve(process.env.WEBGPU_NODE_MODULE ?? "node_modules/webgpu/index.js")).href) as NodeDawnProvider;
  Object.assign(globalThis, dawn.globals);
  const gpu = createProcessRetainedDawnGPU(dawn, ["backend=metal"]);
  const adapter = await gpu.requestAdapter(); assert.ok(adapter);
  assert.ok(adapter.features.has("timestamp-query"));
  device = managedGPUDevice(await adapter.requestDevice({ requiredFeatures: ["timestamp-query"], requiredLimits: requiredFluidDeviceLimits(adapter.limits) }), { requireWorkerRealm: false });
  if (process.argv.includes("--allocation-audit") || maxGPUBytes > 0) {
    allocationAudit = auditUniformGPUAllocations(device);
    device = allocationAudit.device;
  }
  const errors: string[] = [];
  device.addEventListener("uncapturederror", e => { e.preventDefault(); errors.push(e.error.message); console.error(e.error.message); });
  usePerformanceInstrumentationStore.getState().setMode("timeline");
  await GPUStageTimestampRecorder.prepare(device);
  const scene = sceneDocument(getSceneDefinition(sceneId));
  const values = resolveMethodValues(uniformVolumeMethod, "balanced", {});
  const start = performance.now();
  const unsubscribe = process.argv.includes("--compile-progress")
    ? gpuCompilationManagerFor(device).subscribe(s => { if(s.progress) console.log(JSON.stringify(s.progress)); }) : () => {};
  solver = arg("scratch-storage", "shared") === "separate"
    ? await WebGPUUniformReferenceSolver.createAsync(device, scene, "balanced", undefined,
      { ...uniformGeometricSolverOptions(values, scene), scratchStorageForQA: "separate" }, () => {})
    : await uniformVolumeMethod.createSolverAsync!(device, scene, "balanced", values, undefined, () => {}) as WebGPUUniformReferenceSolver;
  unsubscribe();
  const pressureWork=solver.pressureSmoothingWorkSourceForQA;
  if(pressureWork.length)pressureWorkReadback=device.createBuffer({label:"Pressure tile profile readback",size:4*pressureWork.length,
    usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  await device.queue.onSubmittedWorkDone();
  const setup_ms = performance.now()-start;
  const lattice = { nx: solver.info.nx, ny: solver.info.ny, nz: solver.info.nz, cellSize_m: solver.info.cellSize_m };
  if(sceneId === "cm12-figure-7-256") assert.deepEqual([lattice.nx,lattice.ny,lattice.nz],[256,256,256]);
  console.log(JSON.stringify({ sceneId, lattice, setup_ms, allocatedBytes: solver.info.allocatedBytes }));
  let lastSample = -1;
  mkdirSync(dirname(out), { recursive: true });
  for(let frame=1;frame<=frames;frame++) {
    // Respect the production recorder's 100 ms cadence, outside the timed interval.
    await new Promise(r => setTimeout(r,115));
    const begin = performance.now();
    assert.ok(solver.advanceTo(frame/30));
    await solver.awaitFrameCompletion();
    await device.queue.onSubmittedWorkDone();
    const wall_ms = performance.now()-begin;
    let info: WebGPUUniformReferenceSolver["info"] = await solver.readStats();
    for(let attempt=0;attempt<100 && (!info.physicsTrace || info.physicsTrace.sampleId===lastSample);attempt++) {
      await new Promise(r => setTimeout(r,5)); info=await solver.readStats();
    }
    const trace: WebGPUUniformReferenceSolver["info"]["physicsTrace"] = info.physicsTrace;
    assert.ok(trace && trace.sampleId!==lastSample, `missing fresh trace at frame ${frame}`);
    assert.equal(trace.measurementSource,"gpu-hardware-timestamp");
    lastSample=trace.sampleId;
    const work=Object.fromEntries(Object.entries(info).filter(([key]) => /^(allocatedBytes|lastSubsteps|encodedSteps|volumeCellSum|pressureSolver|maxSpeed_m_s|uniformPressure|uniformCM11a|uniformVolumePages|uniformVolumeTransportWorkgroups|uniformVolumeSharpenWorkgroups)/.test(key)));
    if(pressureWorkReadback){
      const encoder=device.createCommandEncoder();
      pressureWork.forEach(({buffer},i)=>encoder.copyBufferToBuffer(buffer,0,pressureWorkReadback!,4*i,4));
      device.queue.submit([encoder.finish()]);await pressureWorkReadback.mapAsync(GPUMapMode.READ);
      work.uniformPressureSmoothingTiles=Array.from(new Uint32Array(pressureWorkReadback.getMappedRange()),(active,i)=>({level:i,active,capacity:pressureWork[i]!.capacity}));
      pressureWorkReadback.unmap();
    }
    rows.push({frame,time_s:frame/30,wall_ms,trace,cpuTrace:info.physicsCPUTrace,work});
    assert.deepEqual(errors,[]);
    if(frame%10===0) console.log(JSON.stringify({frame,wall_ms,gpu_ms:trace.total_ms,volumeCellSum:info.volumeCellSum}));
    writeFileSync(out,JSON.stringify({sceneId,lattice,setup_ms,values,rows,validationErrors:errors},null,2)+"\n");
  }
  const summarize=(selected:typeof rows) => {
    const labels=[...new Set(selected.flatMap(row=>row.trace.phases.map(p=>p.label)))];
    return {frames:[selected[0]!.frame,selected.at(-1)!.frame],wall_ms:stats(selected.map(r=>r.wall_ms)),gpu_ms:stats(selected.map(r=>r.trace.total_ms)),stages:labels.map(label=>({label,...stats(selected.map(r=>r.trace.phases.filter(p=>p.label===label).reduce((s,p)=>s+p.duration_ms,0)))})).sort((a,b)=>b.mean-a.mean)};
  };
  const windows = Object.fromEntries(Object.entries({all:rows.filter(r=>r.frame>4),freeFall:rows.filter(r=>r.frame>4&&r.frame<=24),impactAndSpread:rows.filter(r=>r.frame>=25)}).filter(([,rs])=>rs.length>0).map(([name,rs])=>[name,summarize(rs)]));
  const report={capturedAt:new Date().toISOString(),sceneId,method:uniformVolumeMethod.id,backend:"Dawn/Metal",adapter:{vendor:adapter.info.vendor,architecture:adapter.info.architecture,device:adapter.info.device,description:adapter.info.description},scope:"Instrumented, queue-fenced simulation. Rendering, 115 ms trace-cadence gaps and stats readbacks excluded from wall timings. First four frames excluded from summaries. GPU stages are seam intervals, not isolated kernel durations.",lattice,setup_ms,values,scene,windows,rows,validationErrors:errors};
  const allocationSnapshot=allocationAudit?.snapshot();
  writeFileSync(out,JSON.stringify({...report,allocationAudit:allocationSnapshot},null,2)+"\n");
  if(maxGPUBytes>0)assert.ok(allocationSnapshot!.peakBytes<=maxGPUBytes,
    `Peak live GPU resources ${allocationSnapshot!.peakBytes} exceed budget ${maxGPUBytes}`);
  console.log(JSON.stringify({out,windows},null,2));
} finally { pressureWorkReadback?.destroy(); solver?.destroy(); device?.destroy(); await releaseWebGPUExclusiveLock(); }
