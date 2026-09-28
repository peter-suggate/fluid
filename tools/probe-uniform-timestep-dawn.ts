/** Timestep cost investigation on the production Uniform Geometric solver.
 * --scene=sparse-cm12-long-dam-break --hz=240 --frames=240 --trace-gap-ms=0
 * --warmup=60 --warmup-hz=30 starts measured advances from a common trajectory.
 * --split-stages separates phi transport, redistance, and surface correction.
 * --after-values='{"totalSurfaceVolume":"off"}' applies only after warmup.
 * --redistance-census instruments [band vertices, preserved vertices, Newton
 * iterations, empty-neighborhood checks]; these runs are NOT timing evidence.
 * --correction-dt-floor is a diagnostic numerical ablation, NOT a proposed fix:
 * clamp only the volume-correction denominator to dt >= 1/30.
 * Acquires the exclusive Dawn/browser GPU lease. No production code is edited.
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
import type { WebGPUUniformReferenceSolver } from "../lib/methods/uniform/webgpu-uniform-reference";
import { auditUniformGPUAllocations } from "./uniform-gpu-allocation-audit";
const arg = (key: string, fallback: string) => process.argv.find(a => a.startsWith(`--${key}=`))?.slice(key.length + 3) ?? fallback;
const sceneId = arg("scene", "cm12-figure-7-256");
const dt = 1 / Number(arg("hz", "30"));
const warmupFrames = Number(arg("warmup", "0"));
const warmupDt = 1 / Number(arg("warmup-hz", "30"));
const frames = Number(arg("frames", "60"));
const traceGapMs = Number(arg("trace-gap-ms", "115"));
assert.ok(Number.isFinite(traceGapMs) && traceGapMs >= 0);
const maxGPUBytes = Number(arg("max-gpu-bytes", "0"));
assert.ok(Number.isSafeInteger(maxGPUBytes) && maxGPUBytes >= 0);
assert.ok(Number.isInteger(frames) && frames > 4);
assert.ok(Number.isFinite(dt) && dt > 0 && Number.isFinite(warmupDt) && warmupDt > 0);
assert.ok(Number.isInteger(warmupFrames) && warmupFrames >= 0);
const out = resolve(arg("out", `/tmp/${sceneId}-stages.json`));
const stats = (values: number[]) => {
  const sorted = [...values].sort((a,b) => a-b);
  const quantile = (q: number) => sorted[Math.max(0, Math.ceil(sorted.length*q)-1)]!;
  return { n: values.length, mean: values.reduce((a,b) => a+b,0)/values.length,
    median: quantile(.5), p10: quantile(.1), p90: quantile(.9) };
};
const rows: { frame: number; time_s: number; wall_ms: number; trace: NonNullable<WebGPUUniformReferenceSolver["info"]["physicsTrace"]>; cpuTrace: unknown; quality: Record<string,unknown>; work: Record<string,unknown> }[] = [];
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
  scene.numerics.fixedDt_s = scene.numerics.maxDt_s = warmupFrames ? warmupDt : dt;
  const values = resolveMethodValues(uniformVolumeMethod, "balanced", {...JSON.parse(arg("values", "{}")),timeStep:"scene"});
  const start = performance.now();
  const unsubscribe = process.argv.includes("--compile-progress")
    ? gpuCompilationManagerFor(device).subscribe(s => { if(s.progress) console.log(JSON.stringify(s.progress)); }) : () => {};
  const compiler = gpuCompilationManagerFor(device);
  const originalModule = compiler.createShaderModule;
  const originalBuffer = device.createBuffer;
  if(process.argv.includes("--redistance-census")||process.argv.includes("--correction-dt-floor")){
    device.createBuffer=function(descriptor){return originalBuffer.call(this,
      process.argv.includes("--redistance-census")&&descriptor.label==="Uniform reference diagnostics and volume control" ? {...descriptor,size:64} : descriptor);};
    compiler.createShaderModule=function(descriptor){
      let code=descriptor.code;
      if(process.argv.includes("--correction-dt-floor"))code=code.replaceAll("/max(params.dimsDt.w,1e-12)","/max(params.dimsDt.w,1.0/30.0)");
      if(process.argv.includes("--redistance-census")){code=code.replace("reductions:array<atomic<u32>,12>","reductions:array<atomic<u32>,16>");
      code=code.replace("if(abs(initial)>1e-8&&abs(initial)<band&&!uvBuried(p)){", "$&atomicAdd(&reductions[12],1u);")
        .replace("if(params.splash.x>0.5&&uvSurfaceVertex(vertex,initial)){", "$&atomicAdd(&reductions[13],1u);")
        .replace("let g=uvGradient(q);let norm=", "atomicAdd(&reductions[14],1u);let g=uvGradient(q);let norm=")
        .replace("fn uvNoNearbySurface(vertex:vec3i,band:f32)->bool{", "$&atomicAdd(&reductions[15],1u);");
      }
      return originalModule.call(this,{...descriptor,code});
    };
  }
  solver = await uniformVolumeMethod.createSolverAsync!(device, scene, "balanced", values, undefined, () => {}) as WebGPUUniformReferenceSolver;
  compiler.createShaderModule=originalModule;device.createBuffer=originalBuffer;
  unsubscribe();
  if(process.argv.includes("--split-stages")){
    const target=solver as any;
    let seam: ((phase:{id:string;label:string})=>void)|undefined;
    const original=target.encodeGeometricVolume;
    target.encodeGeometricVolume=function(encoder:GPUCommandEncoder,boundary:typeof seam){
      seam=boundary;try{return original.call(this,encoder,boundary);}finally{seam=undefined;}
    };
    const vertex=target.runVertex;
    target.runVertex=function(...args:any[]){const result=vertex.apply(this,args);
      if(seam)seam({id:"phi-detail",label:args[1]});return result;};
    if(target.surfaceVolumeCorrection){const correction=target.surfaceVolumeCorrection;
      const encode=correction.encode;
      correction.encode=function(...args:any[]){seam?.({id:"gather-detail",label:"Gather before surface correction"});
        const result=encode.apply(this,args);seam?.({id:"surface-detail",label:"Global surface volume correction"});return result;};
    }
  }
  const pressureWork=solver.pressureSmoothingWorkSourceForQA;
  if(pressureWork.length)pressureWorkReadback=device.createBuffer({label:"Pressure tile profile readback",size:4*pressureWork.length,
    usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  await device.queue.onSubmittedWorkDone();
  const setup_ms = performance.now()-start;
  const lattice = { nx: solver.info.nx, ny: solver.info.ny, nz: solver.info.nz, cellSize_m: solver.info.cellSize_m };
  if(sceneId === "cm12-figure-7-256") assert.deepEqual([lattice.nx,lattice.ny,lattice.nz],[256,256,256]);
  console.log(JSON.stringify({ sceneId, lattice, setup_ms, allocatedBytes: solver.info.allocatedBytes }));
  if(warmupFrames){
    usePerformanceInstrumentationStore.getState().setEnabled(false);
    for(let i=1;i<=warmupFrames;i++){
      assert.ok(solver.advanceTo(i*warmupDt));
      await solver.awaitFrameCompletion();await solver.readStats();
    }
    const nextScene=structuredClone(scene);
    nextScene.numerics.fixedDt_s=nextScene.numerics.maxDt_s=dt;
    solver.applySceneUniforms(nextScene);
    usePerformanceInstrumentationStore.getState().setMode("timeline");
  }
  if(arg("after-values", "")!=="")solver.applyRuntimeValues({...values,...JSON.parse(arg("after-values", "{}"))});
  let lastSample = -1;
  mkdirSync(dirname(out), { recursive: true });
  for(let frame=1;frame<=frames;frame++) {
    // Default follows the UI trace cadence; zero measures sustained execution.
    if(traceGapMs>0)await new Promise(r => setTimeout(r,traceGapMs));
    if(traceGapMs<100)(solver as unknown as {lastPhysicsTraceAt_ms:number}).lastPhysicsTraceAt_ms=-Infinity;
    const begin = performance.now();
    if(process.argv.includes("--reapply-values"))solver.applyRuntimeValues(values);
    assert.ok(solver.advanceTo(warmupFrames*warmupDt+frame*dt));
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
    if(process.argv.includes("--redistance-census")){
      const buffer=device.createBuffer({size:16,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
      const encoder=device.createCommandEncoder();encoder.copyBufferToBuffer((solver as any).reductions,48,buffer,0,16);
      device.queue.submit([encoder.finish()]);await buffer.mapAsync(GPUMapMode.READ);
      work.redistanceCensus=Array.from(new Uint32Array(buffer.getMappedRange()));buffer.unmap();buffer.destroy();
    }
    if(pressureWorkReadback){
      const encoder=device.createCommandEncoder();
      pressureWork.forEach(({buffer},i)=>encoder.copyBufferToBuffer(buffer,0,pressureWorkReadback!,4*i,4));
      device.queue.submit([encoder.finish()]);await pressureWorkReadback.mapAsync(GPUMapMode.READ);
      work.uniformPressureSmoothingTiles=Array.from(new Uint32Array(pressureWorkReadback.getMappedRange()),(active,i)=>({level:pressureWork[i]!.level,list:pressureWork[i]!.list,active,capacity:pressureWork[i]!.capacity}));
      pressureWorkReadback.unmap();
    }
    rows.push({frame,time_s:warmupFrames*warmupDt+frame*dt,wall_ms,trace,cpuTrace:info.physicsCPUTrace,work,
      quality:{volumeCellSum:info.volumeCellSum,volumeDrift:info.volumeDrift,representedVolumeDrift:info.representedVolumeDrift,pressureConverged:info.uniformCM11aConverged}});
    assert.deepEqual(errors,[]);
    if(frame%10===0) console.log(JSON.stringify({frame,wall_ms,gpu_ms:trace.total_ms,volumeCellSum:info.volumeCellSum}));
    writeFileSync(out,JSON.stringify({sceneId,lattice,setup_ms,values,rows,validationErrors:errors},null,2)+"\n");
  }
  const summarize=(selected:typeof rows) => {
    const labels=[...new Set(selected.flatMap(row=>row.trace.phases.map(p=>p.label)))];
    return {frames:[selected[0]!.frame,selected.at(-1)!.frame],wall_ms:stats(selected.map(r=>r.wall_ms)),gpu_ms:stats(selected.map(r=>r.trace.total_ms)),stages:labels.map(label=>({label,...stats(selected.map(r=>r.trace.phases.filter(p=>p.label===label).reduce((s,p)=>s+p.duration_ms,0)))})).sort((a,b)=>b.mean-a.mean)};
  };
  const windows = Object.fromEntries(Object.entries({all:rows.filter(r=>r.frame>4),freeFall:rows.filter(r=>r.frame>4&&r.frame<=24),impactAndSpread:rows.filter(r=>r.frame>=25)}).filter(([,rs])=>rs.length>0).map(([name,rs])=>[name,summarize(rs)]));
  const experiment={dt_s:dt,warmupFrames,warmupDt_s:warmupDt,
    afterValues:JSON.parse(arg("after-values","{}")),
    splitStages:process.argv.includes("--split-stages"),
    redistanceCensus:process.argv.includes("--redistance-census"),
    correctionDtFloor:process.argv.includes("--correction-dt-floor")};
  const report={experiment,abOff:process.env.FLUID_UNIFORM_AB_OFF ?? "",capturedAt:new Date().toISOString(),sceneId,method:uniformVolumeMethod.id,backend:"Dawn/Metal",adapter:{vendor:adapter.info.vendor,architecture:adapter.info.architecture,device:adapter.info.device,description:adapter.info.description},traceGapMs,reapplyValues:process.argv.includes("--reapply-values"),scope:"Instrumented, queue-fenced simulation. Rendering, configured trace-cadence gaps, stats and work-count readbacks excluded from wall timings. First four frames excluded from summaries. GPU stages are seam intervals, not isolated kernel durations.",lattice,setup_ms,values,scene,windows,rows,validationErrors:errors};
  const allocationSnapshot=allocationAudit?.snapshot();
  writeFileSync(out,JSON.stringify({...report,allocationAudit:allocationSnapshot},null,2)+"\n");
  if(maxGPUBytes>0)assert.ok(allocationSnapshot!.peakBytes<=maxGPUBytes,
    `Peak live GPU resources ${allocationSnapshot!.peakBytes} exceed budget ${maxGPUBytes}`);
  console.log(JSON.stringify({out,windows},null,2));
} finally { pressureWorkReadback?.destroy(); solver?.destroy(); device?.destroy(); await releaseWebGPUExclusiveLock(); }
