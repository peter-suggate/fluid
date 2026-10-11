/** Frozen real-input proof of sharing extension sweep geometry between two
 * velocity fields. Diagnostic only: production shaders and frame order remain
 * untouched. The companion channel uses one scalar per existing slot; a future
 * frame integration can borrow the inactive particle bank for those words.
 * node --import tsx tools/probe-paired-extension.ts --steps=96 --rounds=8
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
import type { UniformDetailGroup } from "../lib/methods/uniform/uniform-detail-fields";
import type { UniformMixedOwnership } from "../lib/methods/uniform/uniform-mixed-ownership";
import type { WebGPUUniformReferenceSolver } from "../lib/methods/uniform/webgpu-uniform-reference";
import { GPUPassProfile } from "./gpu-pass-profile";

export { pairedSweepSource } from "../lib/methods/uniform/uniform-mixed-extension-pair.wgsl";
import { pairedSweepSource } from "../lib/methods/uniform/uniform-mixed-extension-pair.wgsl";

class PairVariants {
 readonly device: GPUDevice;
 readonly groupDescriptors = new WeakMap<GPUBindGroup, GPUBindGroupDescriptor>();
 private readonly sources = new WeakMap<GPUShaderModule, string>();
 private readonly layoutDescriptors = new WeakMap<GPUPipelineLayout, GPUPipelineLayoutDescriptor>();
 private readonly groupLayouts = new WeakMap<GPUBindGroupLayout, GPUBindGroupLayoutDescriptor>();
 private readonly paired = new WeakMap<GPUComputePipeline, GPUComputePipeline>();
 private readonly pairedLayouts = new WeakMap<GPUBindGroupLayout, GPUBindGroupLayout>();
 private readonly modules = new WeakMap<GPUShaderModule, GPUShaderModule>();
 constructor(private readonly raw: GPUDevice) {
  this.device = new Proxy(raw, { get: (target, key) => {
   if (key === "createShaderModule") return (d: GPUShaderModuleDescriptor) => { const m = target.createShaderModule(d); this.sources.set(m, d.code); return m; };
   if (key === "createBindGroupLayout") return (d: GPUBindGroupLayoutDescriptor) => { const l = target.createBindGroupLayout(d); this.groupLayouts.set(l, { ...d, entries: [...d.entries] }); return l; };
   if (key === "createPipelineLayout") return (d: GPUPipelineLayoutDescriptor) => { const l = target.createPipelineLayout(d); this.layoutDescriptors.set(l, { ...d, bindGroupLayouts: [...d.bindGroupLayouts] }); return l; };
   if (key === "createBindGroup") return (d: GPUBindGroupDescriptor) => { const g = target.createBindGroup(d); this.groupDescriptors.set(g, { ...d, entries: [...d.entries] }); return g; };
   if (key === "createComputePipelineAsync") return async (d: GPUComputePipelineDescriptor) => {
    const p = await target.createComputePipelineAsync(d);
    if (["sweep", "sweepList", "sweepSeams"].includes(d.compute.entryPoint ?? "") && this.sources.get(d.compute.module)?.includes("fn ueCombine(")) {
     let module = this.modules.get(d.compute.module);
     if (!module) {
      const code = pairedSweepSource(this.sources.get(d.compute.module)!);
      module = target.createShaderModule({ label: "Diagnostic paired extension sweeps", code }); this.modules.set(d.compute.module, module);
      const errors = (await module.getCompilationInfo()).messages.filter(m => m.type === "error");
      if (errors.length) { writeFileSync("/tmp/fluid-paired-extension.wgsl", code); throw new Error(errors.map(m => `${m.lineNum}:${m.linePos} ${m.message}`).join("\n")); }
     }
     assert.notEqual(d.layout, "auto"); const layout = this.layoutDescriptors.get(d.layout as GPUPipelineLayout); assert.ok(layout);
     const layouts = [...layout.bindGroupLayouts]; const old = layouts[1]!; let pair = this.pairedLayouts.get(old);
     if (!pair) { const descriptor = this.groupLayouts.get(old); assert.ok(descriptor); pair = target.createBindGroupLayout({ ...descriptor, entries: [...descriptor.entries, ...[2, 3].map(binding => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" as const } }))] }); this.pairedLayouts.set(old, pair); }
     layouts[1] = pair;
     this.paired.set(p, await target.createComputePipelineAsync({ ...d, layout: target.createPipelineLayout({ ...layout, bindGroupLayouts: layouts }), compute: { ...d.compute, module, entryPoint: `pair_${d.compute.entryPoint}` } }));
    }
    return p;
   };
   const value = Reflect.get(target, key, target); return typeof value === "function" ? value.bind(target) : value;
  } });
 }
 pick(p: GPUComputePipeline, pair: boolean): GPUComputePipeline { const result = pair ? this.paired.get(p) : p; assert.ok(result); return result; }
 bind(group: GPUBindGroup, input: GPUBufferBinding, output: GPUBufferBinding): GPUBindGroup {
  const d = this.groupDescriptors.get(group); assert.ok(d); const layout = this.pairedLayouts.get(d.layout); assert.ok(layout);
  return this.raw.createBindGroup({ ...d, layout, entries: [...d.entries, { binding: 2, resource: input }, { binding: 3, resource: output }] });
 }
}

type Frame = {
 ownership: UniformMixedOwnership;
 extension: { regularPipelines: Map<string, GPUComputePipeline>; listPipelines: Map<string, GPUComputePipeline>; seamPipelines: Map<string, GPUComputePipeline>; fineSeamSweep: GPUComputePipeline };
 extensionGroups: readonly [UniformDetailGroup, UniformDetailGroup];
 stageScratch: GPUBuffer;
};

async function run(): Promise<void> {
 const option = (key: string) => process.argv.find(a => a.startsWith(`--${key}=`))?.slice(key.length + 3);
 const steps = Number(option("steps") ?? 96), rounds = Number(option("rounds") ?? 8);
 assert.ok(Number.isInteger(steps) && steps > 0 && Number.isInteger(rounds) && rounds >= 2);
 const definition = getSceneDefinition(option("scene") ?? "nbflip-figure-9-teaser");
 const quality = definition.methodProfile!.quality, scene = structuredClone(sceneDocument(definition));
 const values = resolveMethodValues(method, quality, { ...method.appDefaults, ...definition.methodProfile!.overrides, timeStep: "sixtieth" });
 const dt = method.effectiveStep_s?.(scene, values) ?? scene.numerics.fixedDt_s;
 const out = option("out") ?? "artifacts/nbflip-ui-perf/paired-extension.json";
 let device: GPUDevice | undefined, solver: WebGPUUniformReferenceSolver | undefined, profile: GPUPassProfile | undefined;
 const errors: string[] = [], buffers: GPUBuffer[] = [];
 try {
  const dawn = await import(pathToFileURL(resolve("node_modules/webgpu/index.js")).href); Object.assign(globalThis, dawn.globals);
  const gpu = createProcessRetainedDawnGPU(dawn, ["backend=metal"]), adapter = await gpu.requestAdapter(); assert.ok(adapter);
  const raw = await adapter.requestDevice({ requiredFeatures: ["timestamp-query", ...(adapter.features.has("subgroups") ? ["subgroups" as const] : [])], requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
  profile = new GPUPassProfile(raw); const variants = new PairVariants(profile.device);
  device = managedGPUDevice(variants.device, { requireWorkerRealm: false });
  device.addEventListener("uncapturederror", e => { e.preventDefault(); errors.push(e.error.message); });
  solver = await method.createSolverAsync!(device, scene, quality, values, undefined, () => {}) as WebGPUUniformReferenceSolver;
  for (let n = 1; n <= steps; n++) { solver.advanceTo(n * dt, []); await solver.awaitFrameCompletion(); assert.equal(solver.info.simulationPipelineError, undefined); if (n % 24 === 0 || n === steps) console.log(JSON.stringify({ phase: "advance", frame: n, time: n * dt })); }
  const frame = (solver as unknown as { mixedFrame: Frame }).mixedFrame, o = frame.ownership, ext = frame.extension;
  const sweeps = Number(values.extensionFrontSweeps ?? 2);
  const bindings = frame.extensionGroups.map(g => [...variants.groupDescriptors.get(g.group)!.entries].filter(e => e.binding <= 1).sort((a, b) => a.binding - b.binding).map(e => e.resource as GPUBufferBinding));
  const slots = Number(bindings[0]![0]!.size) / 8, scalarBytes = Math.ceil(slots * 4 / 256) * 256;
  const allocate = (size: number, read = false) => { const b = raw.createBuffer({ size, usage: GPUBufferUsage.COPY_DST | (read ? GPUBufferUsage.MAP_READ : GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC) }); buffers.push(b); return b; };
  const companion = allocate(2 * scalarBytes);
  const pairedGroups = frame.extensionGroups.map((g, parity) => variants.bind(g.group, { buffer: companion, offset: parity * scalarBytes, size: slots * 4 }, { buffer: companion, offset: (parity ^ 1) * scalarBytes, size: slots * 4 }));
  const flags = 3 * o.capacity.tiles + o.capacity.lattice.dimensions[0]! * o.capacity.lattice.dimensions[1]! + o.capacity.lattice.dimensions[0]! * o.capacity.lattice.dimensions[2]! + o.capacity.lattice.dimensions[1]! * o.capacity.lattice.dimensions[2]! + o.capacity.tileDimensions[0]! * o.capacity.tileDimensions[1]! + o.capacity.tileDimensions[0]! * o.capacity.tileDimensions[2]! + o.capacity.tileDimensions[1]! * o.capacity.tileDimensions[2]!;
  const maskEnd = flags + 2 * o.capacity.tiles;
  const initModule = raw.createShaderModule({ code: `
@group(0) @binding(0) var<storage,read_write> a:array<vec2f>;
@group(0) @binding(1) var<storage,read_write> b:array<vec2f>;
@group(0) @binding(2) var<storage,read_write> olderA:array<f32>;
@group(0) @binding(3) var<storage,read_write> olderB:array<f32>;
override replace:bool=false;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid:vec3u){
 for(var i=gid.x;i<arrayLength(&a);i+=65536u){
  var value=a[i].x;
  if(i<${flags}u||i>=${maskEnd}u){value=value*0.375+f32(i%127u)*0.0001-0.007;}
  olderA[i]=value;olderB[i]=value;
  if(replace){a[i].x=value;b[i].x=value;}
 }
}` });
  const initializer = await raw.createComputePipelineAsync({ layout: "auto", compute: { module: initModule, entryPoint: "main" } });
  const oldInitializer = await raw.createComputePipelineAsync({ layout: "auto", compute: { module: initModule, entryPoint: "main", constants: { replace: 1 } } });
  const initGroup = raw.createBindGroup({ layout: initializer.getBindGroupLayout(0), entries: [{ binding: 0, resource: bindings[0]![0]! }, { binding: 1, resource: bindings[0]![1]! }, { binding: 2, resource: { buffer: companion, offset: 0, size: slots * 4 } }, { binding: 3, resource: { buffer: companion, offset: scalarBytes, size: slots * 4 } }] });
  const oldInitGroup = raw.createBindGroup({ layout: oldInitializer.getBindGroupLayout(0), entries: [{ binding: 0, resource: bindings[0]![0]! }, { binding: 1, resource: bindings[0]![1]! }, { binding: 2, resource: { buffer: companion, offset: 0, size: slots * 4 } }, { binding: 3, resource: { buffer: companion, offset: scalarBytes, size: slots * 4 } }] });
  const seed = (e: GPUCommandEncoder, older: boolean) => {
   e.clearBuffer(frame.stageScratch); const p = e.beginComputePass({ label: "fixture seed" }); p.setBindGroup(0, o.bindGroup); p.setBindGroup(1, frame.extensionGroups[1].group);
   o.dispatchTierCounted(p, ext.regularPipelines.get("seed")!, 0); p.setBindGroup(1, frame.extensionGroups[1].base); o.dispatchRegularCoarseCounted(p, ext.listPipelines.get("seed")!);
   p.setBindGroup(1, frame.extensionGroups[1].group); o.dispatchCounted(p, ext.seamPipelines.get("seed")!, o.capacity.tiles, "hanging"); p.end();
   const init = e.beginComputePass({ label: "fixture companion values" }); init.setPipeline(older ? oldInitializer : initializer); init.setBindGroup(0, older ? oldInitGroup : initGroup); init.dispatchWorkgroups(256); init.end();
  };
  const sweep = (e: GPUCommandEncoder, pair: boolean) => {
   const p = e.beginComputePass({ label: pair ? "paired sweeps" : "scalar sweeps" }); p.setBindGroup(0, o.bindGroup);
   for (let step = 0; step < sweeps; step++) {
    p.setBindGroup(1, pair ? pairedGroups[step % 2]! : frame.extensionGroups[step % 2]!.group);
    o.dispatchTierCounted(p, variants.pick(ext.regularPipelines.get("sweep")!, pair), 0);
    if (!o.coarseOnly) { o.dispatchCounted(p, variants.pick(ext.fineSeamSweep, pair), o.capacity.fineTiles, "seamFine"); o.dispatchCounted(p, variants.pick(ext.seamPipelines.get("sweep")!, pair), Math.ceil(o.capacity.tiles / 2) + Math.ceil(o.capacity.tiles / 64), "seams"); }
    else o.dispatchRegularCoarseCounted(p, variants.pick(ext.listPipelines.get("sweep")!, pair));
   }
   p.end();
  };
  const final = bindings[sweeps % 2]![0]!, references = [allocate(slots * 8), allocate(slots * 8)];
  for (const older of [false, true]) { const e = raw.createCommandEncoder(); seed(e, older); sweep(e, false); e.copyBufferToBuffer(final.buffer, final.offset ?? 0, references[+older]!, 0, slots * 8); raw.queue.submit([e.finish()]); await raw.queue.onSubmittedWorkDone(); }
  let e = raw.createCommandEncoder(); seed(e, false); sweep(e, true); raw.queue.submit([e.finish()]); await raw.queue.onSubmittedWorkDone();
  const resultBuffer = allocate(32), read = allocate(32, true);
  const compare = await raw.createComputePipelineAsync({ layout: "auto", compute: { entryPoint: "main", module: raw.createShaderModule({ code: `
@group(0) @binding(0) var<storage,read> current:array<vec2u>;
@group(0) @binding(1) var<storage,read> older:array<u32>;
@group(0) @binding(2) var<storage,read> referenceNew:array<vec2u>;
@group(0) @binding(3) var<storage,read> referenceOld:array<vec2u>;
@group(0) @binding(4) var<storage,read_write> counts:array<atomic<u32>>;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid:vec3u){
 var changed=vec4u(0);var live=0u;
 for(var i=gid.x;i<arrayLength(&current);i+=65536u){
  changed.x+=u32(current[i].x!=referenceNew[i].x);changed.y+=u32(current[i].y!=referenceNew[i].y);
  if(i<${flags}u||i>=${maskEnd}u){changed.z+=u32(older[i]!=referenceOld[i].x);changed.w+=u32(current[i].y!=referenceOld[i].y);live++;}
 }
 for(var k=0u;k<4u;k++){if(changed[k]!=0u){atomicAdd(&counts[k],changed[k]);}}atomicAdd(&counts[4],live);
}` }) } });
  const compareGroup = raw.createBindGroup({ layout: compare.getBindGroupLayout(0), entries: [{ binding: 0, resource: final }, { binding: 1, resource: { buffer: companion, offset: (sweeps % 2) * scalarBytes, size: slots * 4 } }, ...references.map((buffer, index) => ({ binding: index + 2, resource: { buffer } })), { binding: 4, resource: { buffer: resultBuffer } }] });
  e = raw.createCommandEncoder(); e.clearBuffer(resultBuffer); const p = e.beginComputePass(); p.setPipeline(compare); p.setBindGroup(0, compareGroup); p.dispatchWorkgroups(256); p.end(); e.copyBufferToBuffer(resultBuffer, 0, read, 0, 32); raw.queue.submit([e.finish()]); await read.mapAsync(GPUMapMode.READ); const counts = [...new Uint32Array(read.getMappedRange())]; read.unmap();
  console.log(JSON.stringify({ phase: "parity", slots, counts, errors }));
  const passes: Awaited<ReturnType<GPUPassProfile["finish"]>> = [];
  for (let round = 0; round < rounds + 2; round++) for (const pair of round % 2 ? [true, false] : [false, true]) {
   e = raw.createCommandEncoder(); seed(e, false); raw.queue.submit([e.finish()]); await raw.queue.onSubmittedWorkDone();
   profile.start(); e = device.createCommandEncoder(); sweep(e, pair); raw.queue.submit([e.finish()]); await raw.queue.onSubmittedWorkDone(); const row = await profile.finish(); if (round >= 2) passes.push(...row);
  }
  const means = Object.fromEntries(["scalar", "paired"].map(name => { const ms = passes.filter(p => p.label === `${name} sweeps`).map(p => p.ms); return [name, ms.reduce((a, b) => a + b, 0) / ms.length]; }));
  const result = { date: new Date().toISOString(), sceneId: definition.id, values, dt, steps, rounds, slots, pairedStorageBytes: companion.size, mismatches: { newValue: counts[0], distance: counts[1], oldValue: counts[2], oldDistance: counts[3] }, comparedOldValues: counts[4], means_ms: { scalarOnce: means.scalar, scalarTwice: 2 * means.scalar!, pairedOnce: means.paired, saving: 2 * means.scalar! - means.paired! }, scope: "Seed is excluded. One pair of payloads shares each fine/coarse sweep traversal. No hierarchy or publication included.", passes, errors };
  mkdirSync(dirname(out), { recursive: true }); writeFileSync(out, JSON.stringify(result, null, 2) + "\n"); console.log(JSON.stringify(result));
  assert.deepEqual(errors, []); assert.deepEqual(counts.slice(0, 4), [0, 0, 0, 0], "Both fields and all shared distances must match their scalar reference");
 } finally { buffers.forEach(b => b.destroy()); solver?.destroy(); await device?.queue.onSubmittedWorkDone(); profile?.destroy(); device?.destroy(); }
}
if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) await run();
