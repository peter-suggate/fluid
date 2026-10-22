/** Native Metal simulation benchmark using the same scene/profile resolution as
 * the UI. No method, scene, time-step, or quality overrides are accepted.
 * node --import tsx tools/benchmark-nbflip-ui.ts NAME [--steps=120]
 *   [--scene=nbflip-figure-9-teaser] [--profile] [--dispatch-probe]
 *   [--heap] [--cpu] [--unpaced] [--memory]
 *   [--dispatch-labels="Pass label|Other pass"] (with --dispatch-probe)
 * Plain wall time includes encoding, submissions and completion receipts;
 * excludes construction, renderer and diagnostic timestamp readback.
 * The summary reports the measured window's collections (count and pause
 * time) beside its frame times. --heap samples host allocations over the
 * same window and lists the sites, in bytes a frame. --unpaced restores the
 * Dawn binding's own event polling (paceDawnEventPolling), whose garbage
 * otherwise dominates both. --cpu writes a DevTools CPU profile for the
 * measured frames only (construction and final diagnostics are excluded).
 */
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { Session } from "node:inspector";
import { PerformanceObserver, constants as performanceConstants } from "node:perf_hooks";
import { mkdirSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import { cpus, platform, release } from "node:os";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { resolveMethodValues } from "../lib/core/method-contract";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { managedGPUDevice } from "../lib/core/gpu-compilation-manager";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { createProcessRetainedDawnGPU, paceDawnEventPolling } from "../lib/harness/node-dawn-provider";
import { uniformNarrowBandMethod as method } from "../lib/methods/uniform/uniform-narrow-band-method";
import type { WebGPUUniformReferenceSolver } from "../lib/methods/uniform/webgpu-uniform-reference";
import { GPUMemoryProfile } from "./gpu-memory-profile";
import { GPUPassProfile } from "./gpu-pass-profile";
import { NarrowBandDispatchProbe } from "./narrow-band-dispatch-probe";

const option = (key: string) => process.argv.find(a => a.startsWith(`--${key}=`))?.slice(key.length + 3);
const name = process.argv[2] ?? "current";
assert.match(name, /^[a-z0-9-]+$/);
for (const argument of process.argv.slice(3)) {
  assert.ok(/^(--(?:steps|from|scene|out|dispatch-labels)=.+|--profile|--dispatch-probe|--heap|--cpu|--unpaced|--memory)$/.test(argument), `Unknown option: ${argument}`);
}
const steps = Number(option("steps") ?? 120), from = Number(option("from") ?? 13);
assert.ok(Number.isInteger(steps) && steps > 0 && Number.isInteger(from) && from > 0 && from <= steps);
const definition = getSceneDefinition(option("scene") ?? "nbflip-figure-9-teaser");
assert.equal(definition.methodProfile?.methodId, method.id);
const quality = definition.methodProfile!.quality;
const scene = structuredClone(sceneDocument(definition));
const values = resolveMethodValues(method, quality, { ...method.appDefaults, ...definition.methodProfile!.overrides, ...(process.env.NB_VALUES ? JSON.parse(process.env.NB_VALUES) : {}) });
const dt = method.effectiveStep_s?.(scene, values) ?? scene.numerics.fixedDt_s;
assert.ok(steps * dt <= scene.duration_s + 1e-9, "Stay within the authored take");
const profiled = process.argv.includes("--profile"), split = process.argv.includes("--dispatch-probe");
const dispatchLabels=option("dispatch-labels")?.split("|");
assert.ok(!dispatchLabels || split, "--dispatch-labels requires --dispatch-probe");
const heap = process.argv.includes("--heap"), paced = !process.argv.includes("--unpaced");
const cpu = process.argv.includes("--cpu");
if (paced) paceDawnEventPolling();
/** Collections that end inside the measured window. */
const collections: { kind: string; ms: number; frame: number }[] = [];
let measuring = 0;
const collectionKinds: Record<number, string> = { [performanceConstants.NODE_PERFORMANCE_GC_MAJOR]: "major", [performanceConstants.NODE_PERFORMANCE_GC_MINOR]: "minor",
  [performanceConstants.NODE_PERFORMANCE_GC_INCREMENTAL]: "incremental", [performanceConstants.NODE_PERFORMANCE_GC_WEAKCB]: "weak" };
const collector = new PerformanceObserver(list => {
  if (measuring) for (const e of list.getEntries()) collections.push({ kind: collectionKinds[(e as unknown as { detail: { kind: number } }).detail.kind] ?? "other", ms: e.duration, frame: measuring });
});
collector.observe({ entryTypes: ["gc"] });
const inspector = heap || cpu ? new Session() : undefined; inspector?.connect();
const inspect = <T>(method: string, parameters?: object) => new Promise<T>((done, fail) => inspector!.post(method, parameters, (error, result) => error ? fail(error) : done(result as T)));
interface HeapNode { callFrame: { functionName: string; url: string; lineNumber: number }; selfSize: number; children?: HeapNode[] }
assert.ok(!split || profiled, "Dispatch probes require --profile");
const out = option("out") ?? "artifacts/nbflip-ui-perf";
mkdirSync(out, { recursive: true });
const source = createHash("sha256");
for (const file of readdirSync("lib/methods/uniform").filter(f => f.endsWith(".ts")).sort()) {
  source.update(file); source.update(readFileSync(`lib/methods/uniform/${file}`));
}
const metadata = { paced, date: new Date().toISOString(), arguments: process.argv.slice(2), sourceHash: source.digest("hex"),
  sceneId: definition.id, quality, values, dt, grid: scene.voxelDomain, numerics: scene.numerics,
  host: { cpu: cpus()[0]?.model, platform: platform(), release: release(), node: process.version },
  scope: "simulation advance and completion; excludes renderer and construction", profiled, split };
let memory: GPUMemoryProfile | undefined;
let device: GPUDevice | undefined, profile: GPUPassProfile | undefined, solver: WebGPUUniformReferenceSolver | undefined;
const errors: string[] = [], rows: Record<string, unknown>[] = [];
let activeFrame=0,activePhase="construction";
try {
  const dawn = await import(pathToFileURL(resolve("node_modules/webgpu/index.js")).href);
  Object.assign(globalThis, dawn.globals);
  const gpu = createProcessRetainedDawnGPU(dawn, ["backend=metal"]);
  const adapter = await gpu.requestAdapter(); assert.ok(adapter);
  const raw = await adapter.requestDevice({ requiredFeatures: ["timestamp-query",
    ...(["subgroups", "float32-filterable"] as const).filter(feature => adapter.features.has(feature))],
    requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
  if (profiled) profile = new GPUPassProfile(raw);
  const probe = split ? new NarrowBandDispatchProbe(profile!.device,dispatchLabels?new Set(dispatchLabels):undefined) : undefined;
  if (process.argv.includes("--memory")) memory = new GPUMemoryProfile(probe?.device ?? profile?.device ?? raw);
  device = managedGPUDevice(memory?.device ?? probe?.device ?? profile?.device ?? raw, { requireWorkerRealm: false });
  device.addEventListener("uncapturederror", e => { e.preventDefault(); errors.push(e.error.message); });
  const start = performance.now();
  solver = await method.createSolverAsync!(device, scene, quality, values, undefined, () => {}) as WebGPUUniformReferenceSolver;
  const construction_ms = performance.now() - start;
  const constructionMemory = memory?.snapshot();
  if (constructionMemory) writeFileSync(`${out}/${name}-construction-memory.json`, JSON.stringify(constructionMemory, null, 2) + "\n");
  console.log(JSON.stringify({ phase: "constructed", construction_ms, dt, values, grid: [solver.info.nx, solver.info.ny, solver.info.nz], features: [...raw.features], host: metadata.host }));
  for (let frame = 1; frame <= steps; frame++) {
    activeFrame=frame;activePhase="advance";
    if (probe) probe.enabled = frame >= from;
    if (profile && frame >= from) profile.start();
    if (frame === from && heap) await inspect("HeapProfiler.startSampling", { samplingInterval: 4096, includeObjectsCollectedByMajorGC: true, includeObjectsCollectedByMinorGC: true });
    if (frame === from && cpu) { await inspect("Profiler.enable"); await inspect("Profiler.start"); }
    if (frame >= from) measuring = frame;
    const start = performance.now();
    assert.ok(solver.advanceTo(frame * dt, []));
    const encode_ms = performance.now() - start;
    activePhase="completion";await solver.awaitFrameCompletion();
    const wall_ms = performance.now() - start;
    assert.equal(solver.info.simulationPipelineError, undefined);
    assert.ok(Math.abs((solver.info.completedTime_s ?? 0) - frame * dt) < 1e-8);
    activePhase="profile";
    const passes = profile && frame >= from ? await profile.finish() : undefined;
    const published = solver.info as typeof solver.info & Record<string, unknown>;
    const row = { allocatedBytes: memory?.allocatedBytes, frame, time: frame * dt, wall_ms, encode_ms, ...solver.narrowBandFlipInfo, ...solver.workDomainInfo,
      residentPages: solver.pagePoolInfo?.residentPages, claimedPages: solver.pagePoolInfo?.claimedPages, retiredPages: solver.pagePoolInfo?.retiredPages,
      fineTiles: solver.workDomainInfo?.fineTiles ?? solver.info.uniformMixedFineTiles, fineCapacity: solver.info.uniformMixedFineCapacity,
      changedTiles: published.uniformMixedDynamicChangedTiles, refined: published.uniformMixedDynamicRefined, coarsened: published.uniformMixedDynamicCoarsened,
      bandTiles: solver.info.uniformPressureBandTiles,
      bandCycles: solver.info.uniformPressureBandCycles, bandResidual: solver.info.uniformPressureBandResidual, passes };
    rows.push(row);
    if (frame % 12 === 0 || frame === steps) console.log(JSON.stringify({ ...row, passes: undefined }));
  }
  measuring = 0; collector.disconnect();
  if (cpu) {
    const { profile: sampled } = await inspect<{ profile: unknown }>("Profiler.stop");
    writeFileSync(`${out}/${name}.cpuprofile`, JSON.stringify(sampled) + "\n");
  }
  let allocations: { bytesPerFrame: number; sites: Record<string, number> } | undefined;
  if (heap) {
    const { profile: sampled } = await inspect<{ profile: { head: HeapNode } }>("HeapProfiler.stopSampling");
    const sites: Record<string, number> = {}, frames = steps - from + 1; let bytes = 0;
    const walk = (node: HeapNode) => {
      if (node.selfSize) {
        const f = node.callFrame, site = `${f.functionName || "(anonymous)"} ${f.url.replace(/^.*\/(?=(?:lib|tools|node_modules)\/)/, "")}:${f.lineNumber + 1}`;
        sites[site] = (sites[site] ?? 0) + node.selfSize / frames; bytes += node.selfSize;
      }
      for (const child of node.children ?? []) walk(child);
    };
    walk(sampled.head);
    allocations = { bytesPerFrame: bytes / frames, sites: Object.fromEntries(Object.entries(sites).sort((a, b) => b[1] - a[1]).slice(0, 60)) };
  }
  assert.deepEqual(errors, []);
  const measured = rows.filter(r => Number(r.frame) >= from);
  const walls = measured.map(r => Number(r.wall_ms)).sort((a, b) => a - b);
  const labels: Record<string, number> = {};
  for (const row of measured) for (const p of (row.passes ?? []) as { label: string; ms: number }[]) labels[p.label] = (labels[p.label] ?? 0) + p.ms / measured.length;
  const mean_ms = walls.reduce((a, b) => a + b, 0) / walls.length;
  // A hitch is a frame over the median of the sixteen around it: what a viewer sees, apart from the load's own drift.
  const inOrder = measured.map(r => Number(r.wall_ms)), hitches = inOrder.map((wall, i) => {
    const around = inOrder.slice(Math.max(0, i - 8), i + 9).sort((a, b) => a - b); return wall - around[Math.floor(around.length / 2)]!;
  }).sort((a, b) => a - b);
  const pauses = (kind: string) => { const of = collections.filter(c => c.kind === kind); return { count: of.length, total_ms: of.reduce((n, c) => n + c.ms, 0), longest_ms: of.reduce((n, c) => Math.max(n, c.ms), 0) }; };
  const summary = { frames: measured.length, mean_ms,
    median_ms: walls[Math.floor(walls.length / 2)], p95_ms: walls[Math.ceil(walls.length * .95) - 1],
    p99_ms: walls[Math.ceil(walls.length * .99) - 1], longest_ms: walls[walls.length - 1],
    deviation_ms: Math.sqrt(walls.reduce((n, w) => n + (w - mean_ms) ** 2, 0) / walls.length),
    hitch_p99_ms: hitches[Math.ceil(hitches.length * .99) - 1], hitch_longest_ms: hitches[hitches.length - 1],
    hitchesOver5ms: hitches.filter(h => h > 5).length,
    encode_mean_ms: measured.reduce((n, r) => n + Number(r.encode_ms), 0) / measured.length,
    collections: { minor: pauses("minor"), major: pauses("major") }, allocatedBytesPerFrame: allocations?.bytesPerFrame,
    passMeans_ms: Object.fromEntries(Object.entries(labels).sort((a, b) => b[1] - a[1])) };
  // Equal contiguous thirds expose drift as the take fills. All values use
  // already-published frame data; no diagnostics or readbacks enter timing.
  const scalingWindows = ["early", "middle", "late"].flatMap((name, index) => {
    const window = measured.slice(Math.floor(index * measured.length / 3), Math.floor((index + 1) * measured.length / 3));
    if (!window.length) return [];
    const wall = window.map(row => Number(row.wall_ms)).sort((a, b) => a - b);
    const mean_ms = wall.reduce((sum, value) => sum + value, 0) / wall.length;
    const passMeans_ms: Record<string, number> = {};
    for (const row of window) for (const pass of (row.passes ?? []) as { label: string; ms: number }[])
      passMeans_ms[pass.label] = (passMeans_ms[pass.label] ?? 0) + pass.ms / window.length;
    return [{ name, firstFrame: window[0]!.frame, lastFrame: window[window.length - 1]!.frame,
      frames: window.length, mean_ms, fps: 1000 / mean_ms,
      median_ms: wall[Math.floor(wall.length / 2)], p95_ms: wall[Math.ceil(wall.length * .95) - 1],
      encode_mean_ms: window.reduce((sum, row) => sum + Number(row.encode_ms), 0) / window.length,
      firstDependencyRegions16: window[0]!.dependencyRegions16, lastDependencyRegions16: window[window.length - 1]!.dependencyRegions16,
      firstPressureListedOctants: window[0]!.pressureListedOctants, lastPressureListedOctants: window[window.length - 1]!.pressureListedOctants,
      firstAllOwnerGroups: Number(window[0]!.fineTiles)+Math.ceil(Number(window[0]!.coarseTiles)/64),
      lastAllOwnerGroups: Number(window[window.length - 1]!.fineTiles)+Math.ceil(Number(window[window.length - 1]!.coarseTiles)/64),
      firstResidentPages: window[0]!.residentPages, lastResidentPages: window[window.length - 1]!.residentPages,
      firstClaimedPages: window[0]!.claimedPages, lastClaimedPages: window[window.length - 1]!.claimedPages,
      firstRetiredPages: window[0]!.retiredPages, lastRetiredPages: window[window.length - 1]!.retiredPages,
      firstFineTiles: window[0]!.fineTiles, lastFineTiles: window[window.length - 1]!.fineTiles,
      firstParticles: window[0]!.particles, lastParticles: window[window.length - 1]!.particles,
      passMeans_ms: Object.fromEntries(Object.entries(passMeans_ms).sort((a, b) => b[1] - a[1])) }];
  });
  const scaling = { windows: scalingWindows,
    lateToEarlyWallRatio: scalingWindows.length > 1 ? scalingWindows[scalingWindows.length - 1]!.mean_ms / scalingWindows[0]!.mean_ms : undefined };
  const adapterInfo = { vendor: adapter.info.vendor, architecture: adapter.info.architecture,
    device: adapter.info.device, description: adapter.info.description, features: [...raw.features] };
  // Sample physics only after the timed window: this queues diagnostics and
  // requests an optional volume pass, so it must not perturb later timings.
  activePhase="final physics";
  const stats = await solver.readStats();
  const finalPhysics = { maxSpeed_m_s: stats.maxSpeed_m_s, volumeCellSum: stats.volumeCellSum,
    representedVolumeCellSum: stats.representedVolumeCellSum, volumeDrift: stats.volumeDrift,
    front_m: stats.front_m, pressureResidual: stats.uniformPressureAcceptedResidual,
    bandResidual: stats.uniformPressureBandResidual };
  // This identifies the final aggregate diagnostics exactly; it is not a
  // particle/grid state hash and cannot establish full physical equivalence.
  const finalPhysicsSignature = { kind: "aggregate-diagnostics-sha256", version: 1,
    hash: createHash("sha256").update(JSON.stringify(finalPhysics)).digest("hex") };
  writeFileSync(`${out}/${name}.json`, JSON.stringify({ ...metadata, adapter: adapterInfo, memory: memory?.snapshot(), constructionMemory, construction_ms, summary, scaling, allocations, finalPhysics, finalPhysicsSignature, rows, errors }, null, 2) + "\n");
  console.log(JSON.stringify({ phase: "complete", summary, scaling, finalPhysics, finalPhysicsSignature }));
} catch (error) {
  const measured=rows.filter(row=>Number(row.frame)>=from);
  const partialWindows=["early","middle","late"].flatMap((name,index)=>{
    const window=measured.slice(Math.floor(index*measured.length/3),Math.floor((index+1)*measured.length/3));
    if(!window.length)return [];
    const wall=window.map(row=>Number(row.wall_ms)).sort((a,b)=>a-b),passMeans_ms:Record<string,number>={};
    for(const row of window)for(const pass of (row.passes??[]) as {label:string;ms:number}[])passMeans_ms[pass.label]=(passMeans_ms[pass.label]??0)+pass.ms/window.length;
    return [{name,firstFrame:window[0]!.frame,lastFrame:window[window.length-1]!.frame,frames:window.length,
      mean_ms:wall.reduce((sum,value)=>sum+value,0)/wall.length,p95_ms:wall[Math.ceil(.95*wall.length)-1],passMeans_ms}];
  });
  writeFileSync(`${out}/${name}-failed.json`, JSON.stringify({ ...metadata, completedFrames:rows.length,
    failure:{frame:activeFrame,phase:activePhase,message:String(error),stack:error instanceof Error?error.stack:undefined,
      pagePool:solver?.pagePoolInfo,workDomain:solver?.workDomainInfo,solverInfo:solver?.info},
    scaling:{partial:true,windows:partialWindows},rows,errors,error:String(error) }, null, 2) + "\n");
  throw error;
} finally {
  inspector?.disconnect();
  solver?.destroy(); await device?.queue.onSubmittedWorkDone(); profile?.destroy(); device?.destroy();
}
