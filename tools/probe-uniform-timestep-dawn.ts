/** Timestep cost investigation on the production Uniform Geometric solver.
 * --scene=sparse-cm12-long-dam-break --hz=240 --frames=240 --trace-gap-ms=0
 * --warmup=60 --warmup-hz=30 starts measured advances from a common trajectory.
 * --split-stages separates phi transport, redistance, and surface correction.
 * --after-values='{"totalSurfaceVolume":"off"}' applies only after warmup.
 * --band-cycles=2 is a diagnostic override, never a production default.
 * --full-pressure-envelope encodes the existing 4V+3F cap every step; gates
 * and acceptance stay intact. Default uses the production lagged plan.
 * --throughput measures receipt-bounded two-frame batches without timestamps
 * or per-frame stats. --initial=rest uses a still fill.
 * --quality-census reads final canonical GPU owners (unit capacity: use only
 * unsolided scenes). Legacy dense shader ablations are rejected explicitly.
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
import { UNIFORM_PRESSURE_BAND_SCHEDULE } from "../lib/methods/uniform/uniform-pressure-band";
import { initializeRigidBodies } from "../lib/core/rigid-body";
import { readMixedTexture } from "../tests/helpers/uniform-mixed-native-fields";
const arg = (key: string, fallback: string) => process.argv.find(a => a.startsWith(`--${key}=`))?.slice(key.length + 3) ?? fallback;
const sceneId = arg("scene", "cm12-figure-7-256");
const dt = 1 / Number(arg("hz", "30"));
const warmupFrames = Number(arg("warmup", "0"));
const warmupDt = 1 / Number(arg("warmup-hz", "30"));
const frames = Number(arg("frames", "60"));
const bandCycles = Number(arg("band-cycles", "4"));
const afterBandCycles = Number(arg("after-band-cycles", String(bandCycles)));
assert.ok(Number.isInteger(bandCycles) && bandCycles >= 1 && bandCycles <= 12);
assert.ok(Number.isInteger(afterBandCycles) && afterBandCycles >= 1 && afterBandCycles <= bandCycles);
UNIFORM_PRESSURE_BAND_SCHEDULE.cycles = bandCycles;
const discardSeconds = Number(arg("discard-seconds", String(4/30)));
assert.ok(Number.isFinite(discardSeconds) && discardSeconds >= 0 && discardSeconds < frames*dt);
assert.ok(!process.argv.includes("--redistance-census")&&!process.argv.includes("--correction-dt-floor"),
  "Legacy dense shader ablations do not instrument the current mixed solver; use a mixed-stage probe");
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
let reportContext: Record<string,unknown> = {sceneId,experiment:{dt_s:dt,warmupFrames,warmupDt_s:warmupDt,bandCycles,afterBandCycles,
  fullPressureEnvelope:process.argv.includes("--full-pressure-envelope"),afterValues:JSON.parse(arg("after-values","{}"))}};
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
  if (arg("initial", "scene") === "rest") scene.fluid.initialCondition = "tank-fill";
  const roster = process.argv.includes("--no-bodies") ? [] : initializeRigidBodies(scene.rigidBodies);
  scene.numerics.fixedDt_s = scene.numerics.maxDt_s = warmupFrames ? warmupDt : dt;
  const values = resolveMethodValues(uniformVolumeMethod, "balanced", {...JSON.parse(arg("values", "{}")),timeStep:"scene"});
  const start = performance.now();
  const unsubscribe = process.argv.includes("--compile-progress")
    ? gpuCompilationManagerFor(device).subscribe(s => { if(s.progress) console.log(JSON.stringify(s.progress)); }) : () => {};
  solver = await uniformVolumeMethod.createSolverAsync!(device, scene, "balanced", values, undefined, () => {}) as WebGPUUniformReferenceSolver;
  unsubscribe();
  if(process.argv.includes("--full-pressure-envelope")){
    const frame=(solver as any).mixedFrame,get=frame.lagged.get.bind(frame.lagged);
    // Keep the receipt admission check: only replace a plan that exists.
    frame.lagged.get=(key:number)=>get(key)?frame.initialPlan:undefined;
  }
  if(process.argv.includes("--split-stages")){
    const target=solver as any;
    if(target.mixedFrame){
      let active: any;
      const getTrace=target.mixedFrameTrace;
      target.mixedFrameTrace=function(){active=getTrace.call(this);return active;};
      const wrap=(object:any,key:string,label:(args:any[])=>string)=>{
        const original=object[key];object[key]=function(...args:any[]){
          const result=original.apply(this,args);
          active?.phase(args[0],{id:"other",label:label(args)});return result;
        };
      };
      wrap(target.mixedDynamic,"encode",()=>"Dynamic census");
      wrap(target.mixedBuilder,"encode",()=>"Ownership metadata build");
      wrap(target.mixedFrame.surface,"encode",args=>`Surface ${args[1]}`);
      wrap(target.mixedFrame.surfaceVolume,"encode",()=>"Global surface volume correction");
    }else{
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
  }
  const pressureWork=solver.pressureSmoothingWorkSourceForQA;
  if(pressureWork.length)pressureWorkReadback=device.createBuffer({label:"Pressure tile profile readback",size:4*pressureWork.length,
    usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  await device.queue.onSubmittedWorkDone();
  const setup_ms = performance.now()-start;
  const lattice = { nx: solver.info.nx, ny: solver.info.ny, nz: solver.info.nz, cellSize_m: solver.info.cellSize_m };
  reportContext={...reportContext,lattice,setup_ms,values,scene,adapter:{vendor:adapter.info.vendor,device:adapter.info.device,description:adapter.info.description}};
  if(sceneId === "cm12-figure-7-256") assert.deepEqual([lattice.nx,lattice.ny,lattice.nz],[256,256,256]);
  console.log(JSON.stringify({ sceneId, lattice, setup_ms, allocatedBytes: solver.info.allocatedBytes }));
  if(warmupFrames){
    usePerformanceInstrumentationStore.getState().setEnabled(false);
    for(let i=1;i<=warmupFrames;i++){
      assert.ok(solver.advanceTo(i*warmupDt,roster));
      await solver.awaitFrameCompletion();await solver.readStats();
    }
    const nextScene=structuredClone(scene);
    nextScene.numerics.fixedDt_s=nextScene.numerics.maxDt_s=dt;
    solver.applySceneUniforms(nextScene);
    usePerformanceInstrumentationStore.getState().setMode("timeline");
  }
  if(arg("after-values", "")!=="")solver.applyRuntimeValues({...values,...JSON.parse(arg("after-values", "{}"))});
  UNIFORM_PRESSURE_BAND_SCHEDULE.cycles=afterBandCycles;
  const finalQuality=async()=>{
    const frame=(solver as any).mixedFrame,ownership=frame.ownership,n=ownership.capacity.tiles;
    // GPU adoption deliberately does not mirror ownership.layout on the host.
    const read=device!.createBuffer({size:4*n,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
    const encoder=device!.createCommandEncoder();encoder.copyBufferToBuffer(ownership.presentation.buffer,0,read,0,4*n);
    device!.queue.submit([encoder.finish()]);await read.mapAsync(GPUMapMode.READ);
    const words=new Uint32Array(read.getMappedRange()).slice();read.unmap();read.destroy();
    const volume=await readMixedTexture(device!,frame.fields.volume),phi=await readMixedTexture(device!,frame.fields.centerPhi);
    const [nx,ny]=ownership.capacity.lattice.dimensions,[tx,ty]=ownership.capacity.tileDimensions;
    let mass=0,excess=0,maxV=0,airMass=0,fine=0;
    for(let tile=0;tile<n;tile++){
      const w=(words[tile]!&0x80000000)!==0?1:4;if(w===1)fine++;
      const ox=4*(tile%tx),oy=4*(Math.floor(tile/tx)%ty),oz=4*Math.floor(tile/(tx*ty));
      for(let z=oz;z<oz+4;z+=w)for(let y=oy;y<oy+4;y+=w)for(let x=ox;x<ox+4;x+=w){
        const i=x+nx*(y+ny*z),v=volume[i]!;mass+=v*w**3;excess+=Math.max(0,v-1)*w**3;maxV=Math.max(maxV,v);
        if(phi[i]!>=0)airMass+=Math.max(0,v)*w**3;
      }
    }
    return {mass_cells:mass,excess_cells:excess,excessFraction:excess/mass,maxV,airMass_cells:airMass,fineTiles:fine,
      note:"Final canonical GPU ownership; unit capacity (valid for unsolided scenes only). Centre phi classifies air. Extra readbacks excluded from timings."};
  };
  if(process.argv.includes("--throughput")){
    usePerformanceInstrumentationStore.getState().setEnabled(false);
    const advance=async(first:number,count:number)=>{
      for(let i=first;i<first+count;i+=2){
        for(let j=i;j<Math.min(i+2,first+count);j++)assert.ok(solver!.advanceTo(warmupFrames*warmupDt+j*dt,roster));
        await solver!.awaitFrameCompletion();
      }
      await device!.queue.onSubmittedWorkDone();
    };
    const unmeasuredFrames=Math.max(2,2*Math.ceil(discardSeconds/dt/2));
    await advance(1,unmeasuredFrames);
    const began=performance.now();await advance(unmeasuredFrames+1,frames);const elapsed_ms=performance.now()-began;
    const info=await solver.readStats();
    const report={capturedAt:new Date().toISOString(),sceneId,backend:"Dawn/Metal",adapter:adapter.info,
      experiment:{dt_s:dt,warmupFrames,warmupDt_s:warmupDt,bandCycles,afterBandCycles,fullPressureEnvelope:process.argv.includes("--full-pressure-envelope"),initial:arg("initial","scene"),afterValues:JSON.parse(arg("after-values","{}"))},
      values,lattice,setup_ms,frames,unmeasuredFrames,elapsed_ms,msPerStep:elapsed_ms/frames,
      stepsPerWallSecond:frames*1000/elapsed_ms,simulatedSecondsPerWallSecond:frames*dt*1000/elapsed_ms,
      scope:"Simulation only; two frames in flight, production receipt admission. Initial discardSeconds rounded up to an even number of extra steps. No timestamps, per-frame stats, rendering or deliberate gaps.",
      final:info,finalQuality:process.argv.includes("--quality-census")?await finalQuality():undefined,validationErrors:errors};
    assert.deepEqual(errors,[]);mkdirSync(dirname(out),{recursive:true});writeFileSync(out,JSON.stringify(report,null,2)+"\n");
    console.log(JSON.stringify({out,msPerStep:report.msPerStep,stepsPerWallSecond:report.stepsPerWallSecond,simulatedSecondsPerWallSecond:report.simulatedSecondsPerWallSecond}));
    process.exitCode=0;
  }else{
  let lastSample = -1;
  mkdirSync(dirname(out), { recursive: true });
  for(let frame=1;frame<=frames;frame++) {
    // Default follows the UI trace cadence; zero measures sustained execution.
    if(traceGapMs>0)await new Promise(r => setTimeout(r,traceGapMs));
    if(traceGapMs<100)(solver as unknown as {lastPhysicsTraceAt_ms:number}).lastPhysicsTraceAt_ms=-Infinity;
    const begin = performance.now();
    if(process.argv.includes("--reapply-values"))solver.applyRuntimeValues(values);
    assert.ok(solver.advanceTo(warmupFrames*warmupDt+frame*dt,roster));
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
    assert.ok(Math.abs(info.lastDt_s!-dt)<1e-9,`unexpected step ${info.lastDt_s}, requested ${dt}`);
    const work=Object.fromEntries(Object.entries(info).filter(([key]) => /^(allocatedBytes|lastDt_s|lastSubsteps|encodedSteps|volumeCellSum|pressureSolver|maxSpeed_m_s|uniformMixed|uniformPressure|uniformCM11a|uniformVolumePages|uniformVolumeTransportWorkgroups|uniformVolumeSharpenWorkgroups)/.test(key)));
    if(pressureWorkReadback){
      const encoder=device.createCommandEncoder();
      pressureWork.forEach(({buffer},i)=>encoder.copyBufferToBuffer(buffer,0,pressureWorkReadback!,4*i,4));
      device.queue.submit([encoder.finish()]);await pressureWorkReadback.mapAsync(GPUMapMode.READ);
      work.uniformPressureSmoothingTiles=Array.from(new Uint32Array(pressureWorkReadback.getMappedRange()),(active,i)=>({level:pressureWork[i]!.level,list:pressureWork[i]!.list,active,capacity:pressureWork[i]!.capacity}));
      pressureWorkReadback.unmap();
    }
    rows.push({frame,time_s:warmupFrames*warmupDt+frame*dt,wall_ms,trace,cpuTrace:info.physicsCPUTrace,work,
      quality:{volumeCellSum:info.volumeCellSum,volumeDrift:info.volumeDrift,representedVolumeDrift:info.representedVolumeDrift,pressureConverged:info.uniformPressureCyclesConverged}});
    assert.deepEqual(errors,[]);
    if(frame%10===0) console.log(JSON.stringify({frame,wall_ms,gpu_ms:trace.total_ms,volumeCellSum:info.volumeCellSum}));
    writeFileSync(out,JSON.stringify({...reportContext,rows,validationErrors:errors},null,2)+"\n");
  }
  const summarize=(selected:typeof rows) => {
    const labels=[...new Set(selected.flatMap(row=>row.trace.phases.map(p=>p.label)))];
    return {frames:[selected[0]!.frame,selected.at(-1)!.frame],wall_ms:stats(selected.map(r=>r.wall_ms)),gpu_ms:stats(selected.map(r=>r.trace.total_ms)),stages:labels.map(label=>({label,...stats(selected.map(r=>r.trace.phases.filter(p=>p.label===label).reduce((s,p)=>s+p.duration_ms,0)))})).sort((a,b)=>b.mean-a.mean)};
  };
  const windows = {all:summarize(rows.filter(r=>r.frame*dt>discardSeconds+1e-9))};
  const experiment={dt_s:dt,warmupFrames,warmupDt_s:warmupDt,
    bandCycles,afterBandCycles,discardSeconds,fullPressureEnvelope:process.argv.includes("--full-pressure-envelope"),initial:arg("initial","scene"),
    afterValues:JSON.parse(arg("after-values","{}")),
    splitStages:process.argv.includes("--split-stages")};
  const report={experiment,abOff:process.env.FLUID_UNIFORM_AB_OFF ?? "",capturedAt:new Date().toISOString(),sceneId,method:uniformVolumeMethod.id,backend:"Dawn/Metal",adapter:{vendor:adapter.info.vendor,architecture:adapter.info.architecture,device:adapter.info.device,description:adapter.info.description},traceGapMs,reapplyValues:process.argv.includes("--reapply-values"),scope:"Instrumented, queue-fenced simulation. Rendering, configured trace-cadence gaps, stats and work-count readbacks excluded from wall timings. Initial discardSeconds excluded equally in physical time. GPU stages are seam intervals, not isolated kernel durations.",lattice,setup_ms,values,scene,windows,rows,validationErrors:errors};
  const allocationSnapshot=allocationAudit?.snapshot();
  writeFileSync(out,JSON.stringify({...report,finalQuality:process.argv.includes("--quality-census")?await finalQuality():undefined,allocationAudit:allocationSnapshot},null,2)+"\n");
  if(maxGPUBytes>0)assert.ok(allocationSnapshot!.peakBytes<=maxGPUBytes,
    `Peak live GPU resources ${allocationSnapshot!.peakBytes} exceed budget ${maxGPUBytes}`);
  console.log(JSON.stringify({out,windows},null,2));
  }
} catch(error) {
  mkdirSync(dirname(out),{recursive:true});
  writeFileSync(out,JSON.stringify({...reportContext,rows,failure:error instanceof Error?error.message:String(error)},null,2)+"\n");
  throw error;
} finally { pressureWorkReadback?.destroy(); solver?.destroy(); device?.destroy(); await releaseWebGPUExclusiveLock(); }
