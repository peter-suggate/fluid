/** Where an NB-FLIP solver's construction goes, and what it computes after.
 *
 * WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js node --import tsx tools/profile-solver-construction.ts
 *   [--scene=reservoir-seep] [--cell=0.005] [--steps=3] [--report=<path.json>]
 *
 * Every native compilation call on the raw device is timed: createShaderModule
 * (WGSL parse and validation; generation is the caller's), and each
 * createCompute/RenderPipelineAsync from issue to settle. The summary splits
 * the construction wall into time with at least one pipeline in flight, time
 * in shader modules, and the rest (host work and buffer setup). Each module's
 * WGSL is hashed, so two runs show whether the source is deterministic. After
 * construction the solver advances --steps fixed steps and the density,
 * pressure and phi fields are hashed: equal hashes between two builds mean
 * the same simulation output. Every host upload (queue.writeBuffer and
 * writeTexture) during construction is hashed too, in order and by label:
 * the gate of a host-side change that must build the same fields. */
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { managedGPUDevice } from "../lib/core/gpu-compilation-manager";
import { resolveMethodValues } from "../lib/core/method-contract";
import { sceneDocument, sceneDocumentAtLattice } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { configureMetalShaderCache } from "../lib/harness/metal-shader-cache";
import { createProcessRetainedDawnGPU, nodeDawnOptions } from "../lib/harness/node-dawn-provider";
import { uniformNarrowBandMethod as method } from "../lib/methods/uniform/uniform-narrow-band-method";
import type { WebGPUUniformReferenceSolver } from "../lib/methods/uniform/webgpu-uniform-reference";
import { readUniformFields } from "../tests/helpers/uniform-geometric";

const options = process.argv.slice(2);
const option = (key: string) => options.find(a => a.startsWith(`--${key}=`))?.slice(key.length + 3);
for (const a of options) assert.ok(/^--(?:scene|cell|steps|report)=.+$/.test(a), `Unknown option: ${a}`);
const sceneId = option("scene") ?? "reservoir-seep", steps = Number(option("steps") ?? 3);
const cell = option("cell") === undefined ? undefined : Number(option("cell"));
assert.ok(Number.isInteger(steps) && steps >= 0);
const definition = getSceneDefinition(sceneId);
assert.equal(definition.methodProfile?.methodId, method.id, `${sceneId} is not an NB-FLIP scene`);
const quality = definition.methodProfile!.quality;
const values = resolveMethodValues(method, quality, { ...method.appDefaults, ...definition.methodProfile!.overrides });
const scene = cell === undefined ? sceneDocument(definition) : sceneDocumentAtLattice(definition, { cellSize_m: cell, detailCellSize_m: cell }).scene;

const t0 = performance.now(), now = () => performance.now() - t0;
const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE ?? resolve("node_modules/webgpu/index.js")).href);
Object.assign(globalThis, dawn.globals);
const dawnOptions = nodeDawnOptions(["backend=metal"]);
const gpu = createProcessRetainedDawnGPU(dawn, dawnOptions);
const adapter = await gpu.requestAdapter(); assert.ok(adapter);
const raw: GPUDevice = await adapter.requestDevice({ requiredFeatures: [...(["subgroups", "float32-filterable"] as const).filter(f => adapter.features.has(f))],
  requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
const device_ms = now();

interface Module { label?: string; bytes: number; sha: string; ms: number }
interface Pipeline { kind: "compute" | "render"; label?: string; module?: string; entryPoint?: string; start: number; end: number }
const modules: Module[] = [], pipelines: Pipeline[] = [];
const moduleLabel = new WeakMap<GPUShaderModule, string>();
const shader = raw.createShaderModule.bind(raw), computeAsync = raw.createComputePipelineAsync.bind(raw), renderAsync = raw.createRenderPipelineAsync.bind(raw);
raw.createShaderModule = descriptor => {
  const start = now(), module = shader(descriptor);
  modules.push({ label: descriptor.label, bytes: descriptor.code.length, sha: createHash("sha256").update(descriptor.code).digest("hex").slice(0, 16), ms: now() - start });
  moduleLabel.set(module, descriptor.label ?? "");
  return module;
};
const timed = <T>(entry: Omit<Pipeline, "start" | "end">, promise: () => Promise<T>) => {
  const record: Pipeline = { ...entry, start: now(), end: Number.NaN }; pipelines.push(record);
  return promise().finally(() => { record.end = now(); });
};
raw.createComputePipelineAsync = descriptor => timed({ kind: "compute", label: descriptor.label, module: moduleLabel.get(descriptor.compute.module), entryPoint: descriptor.compute.entryPoint },
  () => computeAsync(descriptor));
raw.createRenderPipelineAsync = descriptor => timed({ kind: "render", label: descriptor.label, module: moduleLabel.get(descriptor.vertex.module), entryPoint: descriptor.vertex.entryPoint },
  () => renderAsync(descriptor));
/** Every byte the host uploads while constructing, in queue order: equal
 * hashes between two builds mean the host-built fields (initial volume, phi,
 * velocity, solids, layouts, tables) are identical. Per label too, so a
 * mismatch names its resource. */
const uploads = createHash("sha256"), uploadsByLabel = new Map<string, ReturnType<typeof createHash>>();
let uploadBytes = 0, uploading = true;
const bytesOf = (data: BufferSource | SharedArrayBuffer, offset = 0, size?: number) => {
  const view = ArrayBuffer.isView(data) ? new Uint8Array(data.buffer, data.byteOffset, data.byteLength) : new Uint8Array(data as ArrayBuffer);
  return view.subarray(offset, size === undefined ? undefined : offset + size);
};
const upload = (label: string, bytes: Uint8Array) => {
  if (!uploading) return;
  uploads.update(label).update(bytes); uploadBytes += bytes.byteLength;
  let h = uploadsByLabel.get(label); if (!h) uploadsByLabel.set(label, h = createHash("sha256")); h.update(bytes);
};
const writeBuffer = raw.queue.writeBuffer.bind(raw.queue), writeTexture = raw.queue.writeTexture.bind(raw.queue);
raw.queue.writeBuffer = (buffer, bufferOffset, data, dataOffset, size) => {
  // dataOffset and size count elements of a typed array, bytes otherwise.
  const unit = ArrayBuffer.isView(data) && "BYTES_PER_ELEMENT" in data ? (data as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT : 1;
  upload(`${buffer.label}@${bufferOffset}`, bytesOf(data, (dataOffset ?? 0) * unit, size === undefined ? undefined : size * unit));
  return writeBuffer(buffer, bufferOffset, data, dataOffset, size);
};
raw.queue.writeTexture = (destination, data, layout, size) => {
  upload(`${destination.texture.label}@${destination.mipLevel ?? 0}:${JSON.stringify(destination.origin ?? 0)}`, bytesOf(data, layout.offset ?? 0));
  return writeTexture(destination, data, layout, size);
};
const uploadSummary = () => ({ uploadBytes, uploadHash: uploads.copy().digest("hex").slice(0, 16),
  uploadsByLabel: Object.fromEntries([...uploadsByLabel].map(([k, h]) => [k, h.copy().digest("hex").slice(0, 16)]).sort((a, b) => a[0]! < b[0]! ? -1 : 1)) });
const device = managedGPUDevice(raw, { requireWorkerRealm: false });
const errors: string[] = [];
device.addEventListener("uncapturederror", e => { e.preventDefault(); errors.push(e.error.message); console.error(e.error.message); });

/** The initialization runner's reports: each task's span from its first
 * report to the next report of another task (tasks overlap, so this is when
 * the runner moved on, not the task's own work). */
const tasks: { at: number; taskId: string; phase: string; label: string }[] = [];
function taskTimes(end: number) {
  const spans = new Map<string, { taskId: string; phase: string; ms: number }>();
  tasks.forEach((t, i) => {
    const next = tasks.slice(i + 1).find(u => u.taskId !== t.taskId)?.at ?? end;
    const span = spans.get(t.taskId) ?? { taskId: t.taskId, phase: t.phase, ms: 0 };
    if (i === 0 || tasks[i - 1]!.taskId !== t.taskId) span.ms += next - t.at;
    spans.set(t.taskId, span);
  });
  return [...spans.values()].map(s => ({ ...s, ms: Math.round(s.ms) })).sort((a, b) => b.ms - a.ms);
}
/** The construction so far: pipelines still in flight count to `end`. */
function summarize(start: number, end: number) {
  const settled = pipelines.map(p => ({ ...p, end: Number.isNaN(p.end) ? end : p.end, pending: Number.isNaN(p.end) }));
  // Intervals with at least one pipeline in flight, and the widest overlap.
  const events = settled.flatMap(p => [[p.start, 1], [p.end, -1]] as const).sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  let inFlight = 0, busySince = 0, busy_ms = 0, peak = 0;
  for (const [t, d] of events) {
    if (inFlight === 0 && d > 0) busySince = t;
    inFlight += d; peak = Math.max(peak, inFlight);
    if (inFlight === 0) busy_ms += t - busySince;
  }
  const durations = settled.map(p => ({ ...p, ms: p.end - p.start })).sort((a, b) => b.ms - a.ms);
  const module_ms = modules.reduce((n, m) => n + m.ms, 0);
  return { durations, summary: {
    sceneId, cell_m: scene.voxelDomain.finestCellSize_m, dawnOptions, metalShaderCache: configureMetalShaderCache(), device_ms,
    construction_ms: end - start, complete: end === constructEnd,
    shaderModules: modules.length, uniqueModuleSources: new Set(modules.map(m => m.sha)).size, shaderModule_ms: module_ms,
    wgslBytes: modules.reduce((n, m) => n + m.bytes, 0),
    pipelines: pipelines.length, pipelinesPending: settled.filter(p => p.pending).length,
    pipelineBusy_ms: busy_ms, pipelineSum_ms: durations.reduce((n, p) => n + p.ms, 0), peakConcurrentPipelines: peak,
    other_ms: end - start - busy_ms - module_ms,
    slowestPipelines: durations.slice(0, 12).map(p => ({ label: p.label, module: p.module, entryPoint: p.entryPoint, ms: Math.round(p.ms) })),
    sourceHash: createHash("sha256").update(modules.map(m => m.sha).sort().join(",")).digest("hex").slice(0, 16),
    slowestTasks: taskTimes(end).slice(0, 8),
  } };
}
const report = option("report");
const constructStart = now();
let constructEnd = Number.NaN;
// A run cut short (a GPU hold's timeout) still reports how far it got.
process.once("SIGTERM", () => {
  const { summary, durations } = summarize(constructStart, now());
  console.log(JSON.stringify({ phase: "terminated", ...summary }));
  if (report) writeFileSync(report, JSON.stringify({ summary, modules, pipelines: durations }, null, 2) + "\n");
  process.exit(124);
});
const progress = setInterval(() => console.log(JSON.stringify({ phase: "constructing", s: Math.round(now() - constructStart) / 1000,
  modules: modules.length, pipelines: pipelines.length, settled: pipelines.filter(p => !Number.isNaN(p.end)).length })), 30_000);
const solver = await method.createSolverAsync!(device, scene, quality, values, undefined,
  snapshot => { tasks.push({ at: now(), taskId: snapshot.taskId ?? snapshot.label, phase: snapshot.phase, label: snapshot.label }); }) as WebGPUUniformReferenceSolver;
constructEnd = now(); clearInterval(progress); uploading = false;
const { summary, durations: pipelineDurations } = summarize(constructStart, constructEnd);
const host = uploadSummary();
console.log(JSON.stringify({ phase: "constructed", ...summary, uploadBytes: host.uploadBytes, uploadHash: host.uploadHash }));

const dt = method.effectiveStep_s?.(scene, values) ?? scene.numerics.fixedDt_s;
const digest = (a: ArrayBufferView) => createHash("sha256").update(new Uint8Array(a.buffer, a.byteOffset, a.byteLength)).digest("hex").slice(0, 16);
const hashes: Record<string, string>[] = [];
const sample = async (step: number) => {
  const f = await readUniformFields(device, solver);
  hashes.push({ step: String(step), density: digest(f.density), pressure: digest(f.pressure), phi: digest(f.phi), tiles: digest(f.tiles) });
  console.log(JSON.stringify({ phase: "fields", ...hashes.at(-1) }));
};
await sample(0);
const stepStart = now();
for (let step = 1; step <= steps; step++) {
  assert.ok(solver.advanceTo(step * dt, []), `advance to step ${step}`);
  await solver.awaitFrameCompletion();
  assert.equal(solver.info.simulationPipelineError, undefined);
}
const step_ms = steps ? (now() - stepStart) / steps : 0;
if (steps) await sample(steps);
solver.destroy(); await raw.queue.onSubmittedWorkDone();
assert.deepEqual(errors, []);
if (report) writeFileSync(report, JSON.stringify({ summary, step_ms, hashes, uploads: host, modules, pipelines: pipelineDurations }, null, 2) + "\n");
console.log(JSON.stringify({ phase: "done", construction_ms: summary.construction_ms, step_ms, total_ms: now() }));
process.exit(0);
