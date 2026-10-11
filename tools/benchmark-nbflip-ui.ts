/** Native Metal simulation benchmark using the same scene/profile resolution as
 * the UI. No method, scene, time-step, or quality overrides are accepted.
 * node --import tsx tools/benchmark-nbflip-ui.ts NAME [--steps=120]
 *   [--scene=nbflip-figure-9-teaser] [--profile] [--dispatch-probe]
 *   [--heap] [--unpaced]
 * Plain wall time includes encoding, submissions and completion receipts;
 * excludes construction, renderer and diagnostic timestamp readback.
 * The summary reports the measured window's collections (count and pause
 * time) beside its frame times. --heap samples host allocations over the
 * same window and lists the sites, in bytes a frame. --unpaced restores the
 * Dawn binding's own event polling (paceDawnEventPolling), whose garbage
 * otherwise dominates both.
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
import { GPUPassProfile } from "./gpu-pass-profile";
import { NarrowBandDispatchProbe } from "./narrow-band-dispatch-probe";

const option = (key: string) => process.argv.find(a => a.startsWith(`--${key}=`))?.slice(key.length + 3);
const name = process.argv[2] ?? "current";
assert.match(name, /^[a-z0-9-]+$/);
for (const argument of process.argv.slice(3)) {
  assert.ok(/^(--(?:steps|from|scene|out)=.+|--profile|--dispatch-probe|--heap|--unpaced)$/.test(argument), `Unknown option: ${argument}`);
}
const steps = Number(option("steps") ?? 120), from = Number(option("from") ?? 13);
assert.ok(Number.isInteger(steps) && steps > 0 && Number.isInteger(from) && from > 0 && from <= steps);
const definition = getSceneDefinition(option("scene") ?? "nbflip-figure-9-teaser");
assert.equal(definition.methodProfile?.methodId, method.id);
const quality = definition.methodProfile!.quality;
const scene = structuredClone(sceneDocument(definition));
const values = resolveMethodValues(method, quality, { ...method.appDefaults, ...definition.methodProfile!.overrides });
const dt = method.effectiveStep_s?.(scene, values) ?? scene.numerics.fixedDt_s;
assert.ok(steps * dt <= scene.duration_s + 1e-9, "Stay within the authored take");
const profiled = process.argv.includes("--profile"), split = process.argv.includes("--dispatch-probe");
const heap = process.argv.includes("--heap"), paced = !process.argv.includes("--unpaced");
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
const inspector = heap ? new Session() : undefined; inspector?.connect();
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
let device: GPUDevice | undefined, profile: GPUPassProfile | undefined, solver: WebGPUUniformReferenceSolver | undefined;
const errors: string[] = [], rows: Record<string, unknown>[] = [];
try {
  const dawn = await import(pathToFileURL(resolve("node_modules/webgpu/index.js")).href);
  Object.assign(globalThis, dawn.globals);
  const gpu = createProcessRetainedDawnGPU(dawn, ["backend=metal"]);
  const adapter = await gpu.requestAdapter(); assert.ok(adapter);
  const raw = await adapter.requestDevice({ requiredFeatures: ["timestamp-query", ...(adapter.features.has("subgroups") ? ["subgroups" as const] : [])], requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
  if (profiled) profile = new GPUPassProfile(raw);
  const probe = split ? new NarrowBandDispatchProbe(profile!.device) : undefined;
  device = managedGPUDevice(probe?.device ?? profile?.device ?? raw, { requireWorkerRealm: false });
  device.addEventListener("uncapturederror", e => { e.preventDefault(); errors.push(e.error.message); });
  const start = performance.now();
  solver = await method.createSolverAsync!(device, scene, quality, values, undefined, () => {}) as WebGPUUniformReferenceSolver;
  const construction_ms = performance.now() - start;
  console.log(JSON.stringify({ phase: "constructed", construction_ms, dt, values, grid: [solver.info.nx, solver.info.ny, solver.info.nz], features: [...raw.features], host: metadata.host }));
  for (let frame = 1; frame <= steps; frame++) {
    if (probe) probe.enabled = frame >= from;
    if (profile && frame >= from) profile.start();
    if (frame === from && inspector) await inspect("HeapProfiler.startSampling", { samplingInterval: 4096, includeObjectsCollectedByMajorGC: true, includeObjectsCollectedByMinorGC: true });
    if (frame >= from) measuring = frame;
    const start = performance.now();
    assert.ok(solver.advanceTo(frame * dt, []));
    const encode_ms = performance.now() - start;
    await solver.awaitFrameCompletion();
    const wall_ms = performance.now() - start;
    assert.equal(solver.info.simulationPipelineError, undefined);
    assert.ok(Math.abs((solver.info.completedTime_s ?? 0) - frame * dt) < 1e-8);
    const passes = profile && frame >= from ? await profile.finish() : undefined;
    const published = solver.info as typeof solver.info & Record<string, unknown>;
    const row = { frame, time: frame * dt, wall_ms, encode_ms, ...solver.narrowBandFlipInfo,
      fineTiles: solver.info.uniformMixedFineTiles, fineCapacity: solver.info.uniformMixedFineCapacity,
      changedTiles: published.uniformMixedDynamicChangedTiles, refined: published.uniformMixedDynamicRefined, coarsened: published.uniformMixedDynamicCoarsened,
      bandTiles: solver.info.uniformPressureBandTiles,
      bandCycles: solver.info.uniformPressureBandCycles, bandResidual: solver.info.uniformPressureBandResidual, passes };
    rows.push(row);
    if (frame % 12 === 0 || frame === steps) console.log(JSON.stringify({ ...row, passes: undefined }));
  }
  measuring = 0; collector.disconnect();
  let allocations: { bytesPerFrame: number; sites: Record<string, number> } | undefined;
  if (inspector) {
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
  const adapterInfo = { vendor: adapter.info.vendor, architecture: adapter.info.architecture,
    device: adapter.info.device, description: adapter.info.description, features: [...raw.features] };
  // Sample physics only after the timed window: this queues diagnostics and
  // requests an optional volume pass, so it must not perturb later timings.
  const stats = await solver.readStats();
  const finalPhysics = { maxSpeed_m_s: stats.maxSpeed_m_s, volumeCellSum: stats.volumeCellSum,
    representedVolumeCellSum: stats.representedVolumeCellSum, volumeDrift: stats.volumeDrift,
    front_m: stats.front_m, pressureResidual: stats.uniformPressureAcceptedResidual,
    bandResidual: stats.uniformPressureBandResidual };
  writeFileSync(`${out}/${name}.json`, JSON.stringify({ ...metadata, adapter: adapterInfo, construction_ms, summary, allocations, finalPhysics, rows, errors }, null, 2) + "\n");
  console.log(JSON.stringify({ phase: "complete", summary }));
} catch (error) {
  writeFileSync(`${out}/${name}-failed.json`, JSON.stringify({ ...metadata, rows, errors, error: String(error) }, null, 2) + "\n");
  throw error;
} finally {
  solver?.destroy(); await device?.queue.onSubmittedWorkDone(); profile?.destroy(); device?.destroy();
}
