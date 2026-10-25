/** Which kernels load h texels of slotless blocks under compacted h stores:
 * one solver built with fluid.tiers.fineBlocks and W6_POISON=1 (slot 0, every
 * slotless block's, holds -2^100; the AIR block has its own slot), advanced
 * `steps` frames with every detail field's base and h store (slot 0 aside)
 * scanned after every compute dispatch for poisoned texels (|v| > 1e20 or not
 * finite). Prints, per field, the first dispatch that poisons it, and the
 * dispatches at which any field's poisoned count first grows.
 *   W6_POISON=1 node --import tsx tools/probe-poison-dispatch.ts [--scene=splash] [--length=128] [--pool-blocks=20] [--steps=1] [--fine-blocks=auto] [--gate="RUNNER MODE"] */
import { auditUniformGPUAllocations } from "./uniform-gpu-allocation-audit";
import { readFileSync } from "node:fs";
import { spawn, type ChildProcess } from "node:child_process";
import { resolveMethodValues } from "../lib/core/method-contract";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import type { SceneDescription } from "../lib/core/model";
import { uniformNarrowBandMethod } from "../lib/methods/uniform/uniform-narrow-band-method";
import type { WebGPUUniformReferenceSolver } from "../lib/methods/uniform/webgpu-uniform-reference";
import type { UniformDetailStorage } from "../lib/methods/uniform/uniform-detail-fields";
import { withUniformDevice } from "../tests/helpers/uniform-geometric";
import { gpuCompilationManagerFor } from "../lib/core/gpu-compilation-manager";
import { createRegionTierScene, REGION_TIER_SCENES, type RegionTierScene } from "./region-tier-scenes";

const option = (key: string) => process.argv.find(a => a.startsWith(`--${key}=`))?.slice(key.length + 3);
const sceneOption = option("scene") ?? "splash", steps = Number(option("steps") ?? 1);
const fineOption = option("fine-blocks") ?? "auto";
const method = uniformNarrowBandMethod;
let base: SceneDescription, quality: string, values: unknown;
if ((REGION_TIER_SCENES as readonly string[]).includes(sceneOption)) {
  base = createRegionTierScene(sceneOption as RegionTierScene, Number(option("length") ?? 128), Number(option("depth") ?? 32)); quality = "balanced";
  values = resolveMethodValues(method, quality as never, { ...method.appDefaults, timeStep: "scene" });
} else {
  const d = getSceneDefinition(sceneOption); base = structuredClone(sceneDocument(d)); quality = d.methodProfile!.quality;
  values = resolveMethodValues(method, quality as never, { ...method.appDefaults, ...d.methodProfile!.overrides });
}
if (option("values")) values = { ...(values as object), ...JSON.parse(option("values")!) };
if (option("pool-blocks")) base.fluid = { ...base.fluid, pagePool: { ...base.fluid?.pagePool, blocks: Number(option("pool-blocks")) } } as SceneDescription["fluid"];
if (option("tiers-file")) base.fluid = { ...base.fluid, tiers: { ...base.fluid?.tiers, ...JSON.parse(readFileSync(option("tiers-file")!, "utf8")) } } as SceneDescription["fluid"];
if (option("tiers")) base.fluid = { ...base.fluid, tiers: { ...base.fluid?.tiers, ...JSON.parse(option("tiers")!) } } as SceneDescription["fluid"];
if (fineOption !== "off" && fineOption !== "keep") base.fluid = { ...base.fluid, tiers: { ...base.fluid?.tiers, fineBlocks: fineOption === "auto" ? "auto" : Number(fineOption) } } as SceneDescription["fluid"];
const dt = method.effectiveStep_s?.(base, values as never) ?? base.numerics.fixedDt_s;
base.duration_s = Math.max(base.duration_s, (steps + 1) * dt);

async function acquireGate(gate: string): Promise<ChildProcess> {
  const [command, ...args] = gate.split(" ").filter(Boolean);
  const child = spawn(command!, [...args, "cat"], { stdio: ["pipe", "pipe", "inherit"] });
  await new Promise<void>((done, fail) => { child.stdout!.once("data", () => done()); child.once("exit", code => fail(new Error(`gate exited ${code}`))); child.stdin!.write("held\n"); });
  return child;
}
let lock: ChildProcess | undefined;

const CAPTURES = 1 << 19;
const SCAN = /* wgsl */`
@group(0) @binding(0) var src:texture_3d<f32>;
@group(0) @binding(1) var<storage,read_write> out:array<atomic<u32>>;
struct T{edge:u32,comps:u32,pad0:u32,pad1:u32}
@group(0) @binding(2) var<uniform> t:T;
struct I{index:u32,a:u32,b:u32,c:u32}
@group(0) @binding(3) var<uniform> capture:I;
var<workgroup> count:atomic<u32>;
@compute @workgroup_size(4,4,4) fn main(@builtin(global_invocation_id) g:vec3u,@builtin(local_invocation_index) lane:u32){
 if(lane==0u){atomicStore(&count,0u);}
 workgroupBarrier();
 let d=textureDimensions(src);
 if(all(g<d)&&!(t.edge!=0u&&all(g<vec3u(t.edge)))){
  let v=textureLoad(src,vec3i(g),0);let u=bitcast<vec4u>(v);
  var bad=false;
  for(var c=0u;c<t.comps;c++){if(abs(v[c])>1e20||(u[c]&0x7f800000u)==0x7f800000u){bad=true;}}
  if(bad){atomicAdd(&count,1u);}
 }
 workgroupBarrier();
 if(lane==0u){atomicAdd(&out[capture.index],atomicLoad(&count));}
}`;

await withUniformDevice("poison dispatch", async device => {
  const errors: string[] = [];
  device.addEventListener("uncapturederror", event => { event.preventDefault(); errors.push((event as GPUUncapturedErrorEvent).error.message); });
  // Entry points by pipeline (labels are often empty).
  const entries = new WeakMap<GPUComputePipeline, string>();
  const syncCreate = device.createComputePipeline.bind(device), asyncCreate = device.createComputePipelineAsync.bind(device);
  const name = (d: GPUComputePipelineDescriptor) => `${d.label || d.compute.module.label || "?"}:${d.compute.entryPoint ?? "main"}`;
  (device as { createComputePipeline: unknown }).createComputePipeline = (d: GPUComputePipelineDescriptor) => { const p = syncCreate(d); entries.set(p, name(d)); return p; };
  (device as { createComputePipelineAsync: unknown }).createComputePipelineAsync = async (d: GPUComputePipelineDescriptor) => { const p = await asyncCreate(d); entries.set(p, name(d)); return p; };
  const manager = gpuCompilationManagerFor(device) as unknown as { compileComputePipeline(d: GPUComputePipelineDescriptor): Promise<GPUComputePipeline> };
  const compile = manager.compileComputePipeline.bind(manager);
  manager.compileComputePipeline = async (d: GPUComputePipelineDescriptor) => { const p = await compile(d); entries.set(p, name(d)); return p; };
  const module = device.createShaderModule({ label: "probe scan", code: SCAN });
  const C = GPUShaderStage.COMPUTE;
  const layout = device.createBindGroupLayout({ entries: [
    { binding: 0, visibility: C, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
    { binding: 1, visibility: C, buffer: { type: "storage" } },
    { binding: 2, visibility: C, buffer: { type: "uniform" } }, { binding: 3, visibility: C, buffer: { type: "uniform", hasDynamicOffset: true } }] });
  const pipeline = await asyncCreate({ label: "probe scan", layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }), compute: { module, entryPoint: "main" } });
  const indices = device.createBuffer({ size: 256 * CAPTURES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  { const w = new Uint32Array(64 * CAPTURES); for (let i = 0; i < CAPTURES; i++) w[64 * i] = i; device.queue.writeBuffer(indices, 0, w); }
  const out = device.createBuffer({ size: 4 * CAPTURES, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });

  const probe = device.createCommandEncoder(), p0 = probe.beginComputePass(); const proto = Object.getPrototypeOf(p0); p0.end(); probe.finish();
  const native = { setPipeline: proto.setPipeline, setBindGroup: proto.setBindGroup, dispatch: proto.dispatchWorkgroups };
  const state = new WeakMap<object, { pipeline?: GPUComputePipeline; group0?: GPUBindGroup | null; offsets0?: unknown; start?: number; length?: number }>();
  const of = (pass: object) => { let s = state.get(pass); if (!s) state.set(pass, s = {}); return s; };
  type Target = { texture: () => GPUTexture; name: string; edge: number; comps: number };
  let active: { targets: Target[]; labels: string[]; next: number } | undefined;
  const params = new Map<string, GPUBuffer>(), groups = new Map<GPUTexture, GPUBindGroup>();
  const paramsOf = (edge: number, comps: number) => {
    const key = `${edge},${comps}`; let b = params.get(key);
    if (!b) { b = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }); device.queue.writeBuffer(b, 0, new Uint32Array([edge, comps, 0, 0])); params.set(key, b); }
    return b;
  };
  proto.setPipeline = function (this: object, p: GPUComputePipeline) { of(this).pipeline = p; return native.setPipeline.call(this, p); };
  proto.setBindGroup = function (this: object, index: number, group: GPUBindGroup | null, offsets?: unknown, start?: number, length?: number) {
    if (index === 0) { const s = of(this); s.group0 = group; s.offsets0 = offsets; s.start = start; s.length = length; }
    return start === undefined ? native.setBindGroup.call(this, index, group, offsets) : native.setBindGroup.call(this, index, group, offsets, start, length);
  };
  proto.dispatchWorkgroups = function (this: object, x: number, y?: number, z?: number) {
    native.dispatch.call(this, x, y, z);
    const a = active; if (!a) return;
    const s = of(this);
    a.labels.push(s.pipeline ? entries.get(s.pipeline) ?? (s.pipeline.label || "?") : "?");
    native.setPipeline.call(this, pipeline);
    for (const t of a.targets) {
      const texture = t.texture();
      let g = groups.get(texture);
      if (!g) {
        g = device.createBindGroup({ layout, entries: [{ binding: 0, resource: texture.createView() }, { binding: 1, resource: { buffer: out } },
          { binding: 2, resource: { buffer: paramsOf(t.edge, t.comps) } }, { binding: 3, resource: { buffer: indices, size: 16 } }] });
        groups.set(texture, g);
      }
      if (a.next >= CAPTURES) throw new Error("probe: out of captures");
      native.setBindGroup.call(this, 0, g, [256 * a.next++]);
      native.dispatch.call(this, Math.ceil(texture.width / 4), Math.ceil(texture.height / 4), Math.ceil(texture.depthOrArrayLayers / 4));
    }
    if (s.pipeline) native.setPipeline.call(this, s.pipeline);
    if (s.group0 !== undefined) { if (s.start === undefined) native.setBindGroup.call(this, 0, s.group0, s.offsets0); else native.setBindGroup.call(this, 0, s.group0, s.offsets0, s.start, s.length); }
  };

  const audit = process.argv.includes("--audit") ? auditUniformGPUAllocations(device) : undefined;
  const report = (when: string) => { if (!audit) return; const a = audit.snapshot(); console.log(JSON.stringify({ audit: when, liveMiB: a.liveBytes / 2 ** 20, peakMiB: a.peakBytes / 2 ** 20, resources: a.resources.map(r => `${(r.bytes / 2 ** 20).toFixed(1)} ${r.kind}:${r.label} x${r.count}`) })); };
  const s = await method.createSolverAsync!(audit?.device ?? device, structuredClone(base), quality as never, values as never, undefined, () => {}) as WebGPUUniformReferenceSolver;
  await device.queue.onSubmittedWorkDone();
  const gate = option("gate"); if (gate) lock = await acquireGate(gate);
  try {
    const storage = (s as unknown as { detailStorage: UniformDetailStorage }).detailStorage;
    const targets: Target[] = [];
    for (const h of storage.fieldTextures) {
      const kind = (storage as unknown as { kindOf(t: GPUTexture): string }).kindOf(h), comps = h.format === "rgba32float" ? 4 : 1;
      if (storage.physical(h) !== storage.baseOf(h)) targets.push({ name: h.label, texture: () => storage.physical(h), edge: kind === "vertex" ? 33 : 32, comps });
      if (storage.baseOf(h)) targets.push({ name: `${h.label} base`, texture: () => storage.baseOf(h)!, edge: 0, comps });
    }
    device.queue.writeBuffer(out, 0, new Uint32Array(CAPTURES));
    const labels: string[] = [];
    active = process.argv.includes("--trace") ? undefined : { targets, labels, next: 0 };
    let fatal = "";
    const slotState = (storage as unknown as { slots?: { state: GPUBuffer } }).slots?.state, SB = (storage as unknown as { layout: { address: { blocks: number } } }).layout.address.blocks;
    const inUse = async () => { if (!slotState) return -1; const r = device.createBuffer({ size: 4 * SB, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }); const e2 = device.createCommandEncoder(); e2.copyBufferToBuffer(slotState, 0, r, 0, 4 * SB); device.queue.submit([e2.finish()]);
      await r.mapAsync(GPUMapMode.READ); const w = new Uint32Array(r.getMappedRange().slice(0)); r.unmap(); r.destroy(); return [...w].filter((v, b) => b > 0 && v !== 0).length; };
    const trace = process.argv.includes("--trace"), every = Number(option("every") ?? 1);
    const solidSlots = async () => { const so = (s as unknown as { mixedFrame: { solid?: { slots: number; slotState?: GPUBuffer } } }).mixedFrame.solid; if (!so?.slotState) return null; const r = device.createBuffer({ size: 8, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }); const e2 = device.createCommandEncoder(); e2.copyBufferToBuffer(so.slotState, 0, r, 0, 8); device.queue.submit([e2.finish()]);
      await r.mapAsync(GPUMapMode.READ); const w = new Uint32Array(r.getMappedRange().slice(0)); r.unmap(); r.destroy(); return { cap: so.slots, free: w[0], overflow: w[1] }; };
    const arenaPeak = async () => { const nb = (s as unknown as { mixedFrame: { narrowBandFlip: { state: GPUBuffer; arenaWindows: number } } }).mixedFrame.narrowBandFlip; const r = device.createBuffer({ size: 8, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }); const e2 = device.createCommandEncoder(); e2.copyBufferToBuffer(nb.state, 96, r, 0, 8); device.queue.submit([e2.finish()]);
      await r.mapAsync(GPUMapMode.READ); const w = new Uint32Array(r.getMappedRange().slice(0)); r.unmap(); r.destroy(); return { cap: nb.arenaWindows, peak: w[0], stale: w[1] }; };
    const planCount = async () => { const nb = (s as unknown as { mixedFrame: { narrowBandFlip: { bins: GPUBuffer; reuse: { framePlan: number; regions: number; capacity: number } } } }).mixedFrame.narrowBandFlip; const r = device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }); const e2 = device.createCommandEncoder(); e2.copyBufferToBuffer(nb.bins, 4 * nb.reuse.framePlan, r, 0, 16); device.queue.submit([e2.finish()]);
      await r.mapAsync(GPUMapMode.READ); const w = new Uint32Array(r.getMappedRange().slice(0)); r.unmap(); r.destroy(); return { regions: nb.reuse.regions, capacity: nb.reuse.capacity, on: w[0], included: w[1], requestedTiles: w[2], air: w[3] }; };
    const particles = async () => { const st = (s as unknown as { mixedFrame: { narrowBandFlip: { state: GPUBuffer } } }).mixedFrame.narrowBandFlip.state; const r = device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }); const e2 = device.createCommandEncoder(); e2.copyBufferToBuffer(st, 0, r, 0, 16); device.queue.submit([e2.finish()]);
      await r.mapAsync(GPUMapMode.READ); const n = new Uint32Array(r.getMappedRange())[1]!; r.unmap(); r.destroy(); return n; };
    report("built");
    if (trace) console.log(JSON.stringify({ allocatedMiB: (s.info as { allocatedBytes?: number }).allocatedBytes! / 2 ** 20 }));
    if (trace) { active = undefined; console.log(JSON.stringify({ step: 0, fine: s.info.uniformMixedFineTiles, slots: await inUse() })); }
    try { for (let step = 1; step <= steps; step++) { s.advanceTo(step * dt, []); await s.awaitFrameCompletion(); if (trace && (step % every === 0 || step === steps)) console.log(JSON.stringify({ step, fine: s.info.uniformMixedFineTiles, slots: await inUse(), particles: await particles(), drift: (await s.readStats()).volumeDrift, pool: (s as unknown as { pagePoolInfo?: { residentPages: number } }).pagePoolInfo?.residentPages, solid: await solidSlots(), arena: await arenaPeak(), plan: await planCount() })); } }
    catch (error) { fatal = String(error instanceof Error ? error.message : error).slice(0, 300); }
    active = undefined;
    await device.queue.onSubmittedWorkDone();
    const stats = fatal ? { volumeDrift: NaN } : await s.readStats();
    const read = device.createBuffer({ size: 4 * CAPTURES, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const e = device.createCommandEncoder(); e.copyBufferToBuffer(out, 0, read, 0, 4 * CAPTURES); device.queue.submit([e.finish()]);
    await read.mapAsync(GPUMapMode.READ); const counts = new Uint32Array(read.getMappedRange().slice(0)); read.unmap(); read.destroy();
    const slotWords = await (async () => { const st = (storage as unknown as { slots?: { state: GPUBuffer } }).slots?.state, B = (storage as unknown as { layout: { address: { blocks: number } } }).layout.address.blocks; if (!st) return new Uint32Array(0);
      const r = device.createBuffer({ size: 4 * B, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }); const e2 = device.createCommandEncoder(); e2.copyBufferToBuffer(st, 0, r, 0, 4 * B); device.queue.submit([e2.finish()]);
      await r.mapAsync(GPUMapMode.READ); const w = new Uint32Array(r.getMappedRange().slice(0)); r.unmap(); r.destroy(); return w; })();
    report("end");
    console.log(JSON.stringify({ slotsInUse: [...slotWords].filter((v, b) => b > 0 && v !== 0).length }));
    const T = targets.length;
    console.log(JSON.stringify({ dispatches: labels.length, targets: T, fine: s.info.uniformMixedFineTiles, slots: storage.fineSlots, drift: stats.volumeDrift, fatal, error: s.info.simulationPipelineError, errors: errors.slice(0, 4) }));
    const firsts: Record<string, string> = {};
    for (let t = 0; t < T; t++) for (let d = 0; d < labels.length; d++) if (counts[d * T + t]) { firsts[targets[t]!.name] = `${d}:${labels[d]} (${counts[d * T + t]})`; break; }
    console.log(JSON.stringify({ firsts }, null, 1));
    // Growth events: a dispatch after which some field's poisoned count rose.
    const growth: string[] = [];
    for (let d = 0; d < labels.length && growth.length < 80; d++) {
      const rose: string[] = [];
      for (let t = 0; t < T; t++) { const now = counts[d * T + t]!, was = d ? counts[(d - 1) * T + t]! : 0; if (now > was) rose.push(`${targets[t]!.name}+${now - was}`); }
      if (rose.length) growth.push(`${d}:${labels[d]} -> ${rose.join(", ")}`);
    }
    console.log(growth.join("\n"));
  } finally { s.destroy(); lock?.stdin?.end(); }
}, ["timestamp-query", "subgroups"]);
