/** Replay a captured coarse pressure RHS with exact cached cut-face fractions.
 * node --import tsx tools/probe-pressure-rhs-cache.ts [--steps=96] [--rounds=16] [--out=FILE]
 */
import assert from "node:assert/strict";
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { resolveMethodValues } from "../lib/core/method-contract";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { managedGPUDevice } from "../lib/core/gpu-compilation-manager";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import { uniformNarrowBandMethod as method } from "../lib/methods/uniform/uniform-narrow-band-method";
import { uniformDetailField, type UniformDetailGroup } from "../lib/methods/uniform/uniform-detail-fields";
import { UniformMixedPressureVelocity, type UniformMixedPressureRhsFields } from "../lib/methods/uniform/uniform-mixed-pressure-velocity";
import { uniformMixedPressureStorage } from "../lib/methods/uniform/uniform-mixed-pressure-boundary.wgsl";
import type { WebGPUUniformReferenceSolver } from "../lib/methods/uniform/webgpu-uniform-reference";
import { GPUPassProfile } from "./gpu-pass-profile";

export function cachedPressureRhs(code: string): string {
 const needle = "let volume=umPressureFaceV(c,axis);sum+=volume*u;";
 assert.equal(code.split(needle).length, 2, "Expected one cut-face flux site");
 return code.replace(needle, `var volume=0.0;
  if(umSolidEnabled()&&c[axis]>=0){volume=umPatchFaceV(umTileAt(vec3u(c)/4u),c,axis);}
  else{volume=umPressureFaceV(c,axis);}sum+=volume*u;`);
}
class Variants {
 readonly device: GPUDevice;
 candidate = false;
 selected = 0;
 readonly twins = new WeakMap<GPUComputePipeline, GPUComputePipeline>();
 constructor(raw: GPUDevice) {
  const sources = new WeakMap<GPUShaderModule, string>();
  this.device = new Proxy(raw, { get: (target, key) => {
   if (key === "createBuffer") return (d: GPUBufferDescriptor) => target.createBuffer(d.usage & (GPUBufferUsage.MAP_READ | GPUBufferUsage.MAP_WRITE) ? d : { ...d, usage: d.usage | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
   if (key === "createShaderModule") return (d: GPUShaderModuleDescriptor) => { const module = target.createShaderModule(d); sources.set(module, d.code); return module; };
   if (key === "createComputePipelineAsync") return async (d: GPUComputePipelineDescriptor) => {
    const pipeline = await target.createComputePipelineAsync(d), source = sources.get(d.compute.module);
    if (d.compute.entryPoint === "buildRhs" && source?.includes("fn umCutFlux(")) {
     const code = cachedPressureRhs(source), module = target.createShaderModule({ label: "Diagnostic cached pressure face fractions", code });
     const errors = (await module.getCompilationInfo()).messages.filter(m => m.type === "error");
     if (errors.length) { writeFileSync("/tmp/fluid-pressure-rhs-cache-failed.wgsl", code); throw new Error(errors.map(m => `${m.lineNum}:${m.linePos} ${m.message}`).join("\n")); }
     this.twins.set(pipeline, await target.createComputePipelineAsync({ ...d, compute: { ...d.compute, module } }));
    }
    return pipeline;
   };
   if (key === "createCommandEncoder") return (d?: GPUCommandEncoderDescriptor) => {
    const encoder = target.createCommandEncoder(d);
    return new Proxy(encoder, { get: (e, k) => {
     if (k === "beginComputePass") return (desc?: GPUComputePassDescriptor) => {
      const pass = e.beginComputePass(desc);
      return new Proxy(pass, { get: (p, member) => {
       if (member === "setPipeline") return (pipeline: GPUComputePipeline) => { const twin = this.candidate ? this.twins.get(pipeline) : undefined; if (twin) this.selected++; p.setPipeline(twin ?? pipeline); };
       const value = Reflect.get(p, member, p); return typeof value === "function" ? value.bind(p) : value;
      } });
     };
     const value = Reflect.get(e, k, e); return typeof value === "function" ? value.bind(e) : value;
    } });
   };
   const value = Reflect.get(target, key, target); return typeof value === "function" ? value.bind(target) : value;
  } });
 }
}
type RecordCopy = { name: string; size: number; output: boolean; capture(e: GPUCommandEncoder, b: GPUBuffer): void; restore(e: GPUCommandEncoder, b: GPUBuffer): void };
const option = (key: string) => process.argv.find(a => a.startsWith(`--${key}=`))?.slice(key.length + 3);
const steps = Number(option("steps") ?? 96), rounds = Number(option("rounds") ?? 16);
assert.ok(Number.isInteger(steps) && steps > 0 && Number.isInteger(rounds) && rounds >= 4);
const definition = getSceneDefinition(option("scene") ?? "nbflip-figure-9-teaser"), scene = structuredClone(sceneDocument(definition));
assert.equal(definition.methodProfile?.methodId, method.id);
const quality = definition.methodProfile!.quality, values = resolveMethodValues(method, quality, { ...method.appDefaults, ...definition.methodProfile!.overrides });
const dt = method.effectiveStep_s?.(scene, values) ?? scene.numerics.fixedDt_s;
const out = option("out") ?? "artifacts/nbflip-ui-perf/pressure-rhs-cache.json";
const fields = new WeakMap<UniformMixedPressureVelocity, UniformMixedPressureRhsFields>();
const bindRhs = UniformMixedPressureVelocity.prototype.bindRhs;
UniformMixedPressureVelocity.prototype.bindRhs = function(f) { fields.set(this, f); return bindRhs.call(this, f); };
let solver: WebGPUUniformReferenceSolver | undefined, raw: GPUDevice | undefined, profile: GPUPassProfile | undefined;
const buffers = new Set<GPUBuffer>(), errors: string[] = [];
try {
 const dawn = await import(pathToFileURL(resolve("node_modules/webgpu/index.js")).href); Object.assign(globalThis, dawn.globals);
 const gpu = createProcessRetainedDawnGPU(dawn, ["backend=metal"]), adapter = await gpu.requestAdapter(); assert.ok(adapter);
 raw = await adapter.requestDevice({ requiredFeatures: ["timestamp-query", ...(adapter.features.has("subgroups") ? ["subgroups" as const] : [])], requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
 raw.addEventListener("uncapturederror", e => { e.preventDefault(); errors.push(e.error.message); });
 profile = new GPUPassProfile(raw); const variants = new Variants(profile.device), device = managedGPUDevice(variants.device, { requireWorkerRealm: false });
 solver = await method.createSolverAsync!(device, scene, quality, values, undefined, () => {}) as WebGPUUniformReferenceSolver;
 const frame = (solver as unknown as { mixedFrame: { projection: UniformMixedPressureVelocity; split: { rhsGroup: UniformDetailGroup } } }).mixedFrame;
 const projection = frame.projection, f = fields.get(projection); assert.ok(f?.fine);
 const count = uniformMixedPressureStorage(projection.ownership.layout).count, records: RecordCopy[] = [];
 const recordBuffer = (name: string, v: GPUBufferBinding, size: number, output = false) => {
  const offset = v.offset ?? 0; assert.ok(offset + size <= v.buffer.size);
  records.push({ name, size, output, capture: (e, b) => e.copyBufferToBuffer(v.buffer, offset, b, 0, size), restore: (e, b) => e.copyBufferToBuffer(b, 0, v.buffer, offset, size) });
 };
 recordBuffer("rhs", f.rhs, count * 4, true); recordBuffer("minimum", f.minimum, count * 4, true); recordBuffer("pressure", f.pressure, count * 4, true);
 recordBuffer("phi", f.phi, projection.ownership.layout.cellCount * 4);
 recordBuffer("negative", f.negative, f.negative.size ?? f.negative.buffer.size - (f.negative.offset ?? 0));
 recordBuffer("fineNegative", { buffer: f.fine.negative }, f.fine.negative.size);
 recordBuffer("parameters", { buffer: f.params }, 32);
 for (const [name, handle] of [["velocity", f.velocity], ["fineVelocity", f.fine.velocity], ["correction", f.correction]] as const) {
  const field = uniformDetailField(handle), textures = [...new Set(field ? [field.storage.physical(field.handle), field.storage.baseOf(field.handle)!] : [handle])];
  for (const [bank, texture] of textures.entries()) {
   const bytesPerTexel = texture.format === "rgba32float" ? 16 : 4, bytesPerRow = Math.ceil(texture.width * bytesPerTexel / 256) * 256;
   const rowsPerImage = texture.height, extent: [number, number, number] = [texture.width, texture.height, texture.depthOrArrayLayers], size = bytesPerRow * rowsPerImage * extent[2];
   records.push({ name: `${name}-${bank}`, size, output: false, capture: (e, b) => e.copyTextureToBuffer({ texture }, { buffer: b, bytesPerRow, rowsPerImage }, extent), restore: (e, b) => e.copyBufferToTexture({ buffer: b, bytesPerRow, rowsPerImage }, { texture }, extent) });
  }
 }
 const allocate = (size: number, read = false) => { const b = raw!.createBuffer({ size, usage: GPUBufferUsage.COPY_DST | (read ? GPUBufferUsage.MAP_READ : GPUBufferUsage.COPY_SRC) }); buffers.add(b); return b; };
 const release = (b: GPUBuffer) => { buffers.delete(b); b.destroy(); };
 const snapshots = records.map(r => allocate(r.size)), encode = projection.encode.bind(projection); let n = 0, snapshotsTaken = 0;
 projection.encode = (e, entry, group) => { if (n === steps && entry === "rhs") { records.forEach((r, i) => r.capture(e, snapshots[i]!)); snapshotsTaken++; } encode(e, entry, group); };
 for (n = 1; n <= steps; n++) { solver.advanceTo(n * dt, []); await solver.awaitFrameCompletion(); assert.equal(solver.info.simulationPipelineError, undefined); if (n % 12 === 0 || n === steps) console.log(JSON.stringify({ phase: "advance", frame: n })); }
 projection.encode = encode; assert.ok(snapshotsTaken > 0);
 const restore = (e: GPUCommandEncoder) => records.forEach((r, i) => r.restore(e, snapshots[i]!));
 const run = (e: GPUCommandEncoder, candidate: boolean) => { variants.candidate = candidate; encode(e, "rhs", frame.split.rhsGroup); variants.candidate = false; };
 const outputs = records.filter(r => r.output);
 const capture = async (candidate: boolean) => {
  const e = device.createCommandEncoder(), reads = outputs.map(r => allocate(r.size, true)); restore(e); run(e, candidate);
  outputs.forEach((r, i) => r.capture(e, reads[i]!)); raw!.queue.submit([e.finish()]);
  return Promise.all(reads.map(async b => { await b.mapAsync(GPUMapMode.READ); const result = new Uint32Array(b.getMappedRange().slice(0)); b.unmap(); release(b); return result; }));
 };
 const a = await capture(false), b = await capture(true); assert.ok(variants.selected > 0, "Candidate pipeline must actually execute"); let words = 0, mismatches = 0; const first: unknown[] = [];
 a.forEach((data, bank) => { words += data.length; data.forEach((v, i) => { if (v !== b[bank]![i]) { mismatches++; if (first.length < 12) first.push({ bank: outputs[bank]!.name, i, a: v, b: b[bank]![i] }); } }); });
 console.log(JSON.stringify({ phase: "parity", words, mismatches, first })); assert.equal(mismatches, 0);
 const timings: { round: number; candidate: boolean; ms: number }[] = [];
 for (let round = -12; round < rounds; round++) for (const candidate of round % 2 ? [true, false] : [false, true]) {
  let e = raw.createCommandEncoder(); restore(e); raw.queue.submit([e.finish()]); await raw.queue.onSubmittedWorkDone();
  if (round >= 0) profile.start(); e = device.createCommandEncoder(); run(e, candidate); device.queue.submit([e.finish()]); await device.queue.onSubmittedWorkDone();
  if (round >= 0) { const passes = await profile.finish(); timings.push({ round, candidate, ms: passes.reduce((s, p) => s + p.ms, 0) }); }
 }
 const mean = (a: number[]) => a.reduce((s, x) => s + x, 0) / a.length;
 const baseline = timings.filter(t => !t.candidate).map(t => t.ms), candidate = timings.filter(t => t.candidate).map(t => t.ms), saved = baseline.map((v, i) => v - candidate[i]!);
 const result = { date: new Date().toISOString(), sceneId: definition.id, values, dt, steps, rounds, snapshotsTaken, words, mismatches, first, mean_ms: { baseline: mean(baseline), candidate: mean(candidate), saved: mean(saved) }, saved_ms: saved, timings, errors };
 assert.deepEqual(errors, []); mkdirSync(dirname(out), { recursive: true }); writeFileSync(out, JSON.stringify(result, null, 2) + "\n"); console.log(JSON.stringify({ phase: "complete", ...result.mean_ms }));
} finally {
 UniformMixedPressureVelocity.prototype.bindRhs = bindRhs; buffers.forEach(b => b.destroy()); solver?.destroy(); await raw?.queue.onSubmittedWorkDone(); profile?.destroy(); raw?.destroy();
}
