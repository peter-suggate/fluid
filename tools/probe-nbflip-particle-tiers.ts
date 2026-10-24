/** Particle memory by h capacity (scene.fluid.tiers): bytes by class, the
 * program and allocation fingerprints, per-frame particle counts and, at the
 * end, a hash of every copyable buffer and texture (the fields).
 * node --import tsx tools/probe-nbflip-particle-tiers.ts NAME --scene=ID
 *   [--steps=N] [--tiers=JSON] [--hash=all|none] [--profile] [--out=DIR]
 * --tiers sets scene.fluid.tiers, e.g. '{"bricks":2000,"particlesPerFineCell":4}'.
 * PROBE_SHADERS=DIR writes every shader module there, in creation order.
 * --profile adds pass timestamps (mean ms by label over the last third).
 * Two arms whose program and allocation fingerprints match run the same GPU
 * work; the field hash then says whether the run itself repeats bit for bit. */
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdirSync, writeFileSync } from "node:fs";
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
import { auditUniformGPUAllocations } from "./uniform-gpu-allocation-audit";
import { createRegionTierScene, type RegionTierScene } from "./region-tier-scenes";

const option = (key: string) => process.argv.find(a => a.startsWith(`--${key}=`))?.slice(key.length + 3);
const name = process.argv[2] ?? "run"; assert.match(name, /^[a-z0-9-]+$/);
const steps = Number(option("steps") ?? 24), hashMode = option("hash") ?? "all", profiled = process.argv.includes("--profile");
const out = option("out") ?? "artifacts/nbflip-particle-tiers"; mkdirSync(out, { recursive: true });
paceDawnEventPolling();
// --scene=region:MODE:LENGTH:BLOCKS is a region-tier scene (tools/region-tier-scenes.ts) on a pool of BLOCKS blocks.
const sceneId = option("scene") ?? "nbflip-figure-8-letters", region = /^region:([a-z]+):(\d+):(\d+)$/.exec(sceneId);
const definition = region ? { id: sceneId, methodProfile: { quality: "balanced" as const, overrides: {} } } : getSceneDefinition(sceneId);
const quality = definition.methodProfile!.quality, scene = region ? createRegionTierScene(region[1] as RegionTierScene, Number(region[2])) : structuredClone(sceneDocument(definition as ReturnType<typeof getSceneDefinition>));
if (region) { scene.fluid.pagePool = { blocks: Number(region[3]) }; scene.duration_s = Math.max(scene.duration_s, steps / 60); }
if (option("tiers")) (scene.fluid as Record<string, unknown>).tiers = JSON.parse(option("tiers")!);
const values = resolveMethodValues(method, quality, { ...method.appDefaults, ...definition.methodProfile!.overrides, ...(process.env.NB_VALUES ? JSON.parse(process.env.NB_VALUES) : {}) });
const dt = method.effectiveStep_s?.(scene, values) ?? scene.numerics.fixedDt_s;
const dawn = await import(pathToFileURL(resolve("node_modules/webgpu/index.js")).href); Object.assign(globalThis, dawn.globals);
const gpu = createProcessRetainedDawnGPU(dawn, ["backend=metal", ...(profiled ? ["disable-dawn-features=timestamp_quantization"] : [])]), adapter = await gpu.requestAdapter(); assert.ok(adapter);
const raw = await adapter.requestDevice({ requiredFeatures: ["timestamp-query", ...(["subgroups", "float32-filterable"] as const).filter(f => adapter.features.has(f))], requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
const profile = profiled ? new GPUPassProfile(raw) : undefined;
let shaderCount = 0;
const programs = createHash("sha256"), allocations = createHash("sha256");
const buffers: GPUBuffer[] = [], textures: GPUTexture[] = [], destroyed = new WeakSet<object>();
const fingerprinted = new Proxy(profile?.device ?? raw, { get(target, key) {
  if (key === "createShaderModule") return (d: GPUShaderModuleDescriptor) => { programs.update(d.code); programs.update("\0"); if (process.env.PROBE_SHADERS) { mkdirSync(process.env.PROBE_SHADERS, { recursive: true }); writeFileSync(`${process.env.PROBE_SHADERS}/${String(shaderCount++).padStart(4, "0")}.wgsl`, `// ${d.label}\n${d.code}`); } return target.createShaderModule(d); };
  if (key === "createBuffer") return (d: GPUBufferDescriptor) => { allocations.update(`b ${d.label} ${d.size} ${d.usage}\n`); const b = target.createBuffer(d); buffers.push(b); const destroy = b.destroy.bind(b); b.destroy = () => { destroyed.add(b); destroy(); }; return b; };
  if (key === "createTexture") return (d: GPUTextureDescriptor) => { allocations.update(`t ${d.label} ${JSON.stringify(d.size)} ${d.format} ${d.usage}\n`); const t = target.createTexture(d); textures.push(t); const destroy = t.destroy.bind(t); t.destroy = () => { destroyed.add(t); destroy(); }; return t; };
  const value = Reflect.get(target, key); return typeof value === "function" ? value.bind(target) : value;
}});
const audit = auditUniformGPUAllocations(fingerprinted);
const device = managedGPUDevice(audit.device, { requireWorkerRealm: false }), errors: string[] = [];
device.addEventListener("uncapturederror", e => { e.preventDefault(); errors.push(e.error.message); });
let solver: WebGPUUniformReferenceSolver | undefined;
const rows: Record<string, unknown>[] = [], history = createHash("sha256");
try {
  solver = await method.createSolverAsync!(device, scene, quality, values, undefined, () => {}) as WebGPUUniformReferenceSolver;
  const constructed = audit.snapshot();
  for (let frame = 1; frame <= steps; frame++) {
    const timed = profile && frame > steps - Math.max(1, Math.floor(steps / 3));
    if (timed) profile!.start();
    const start = performance.now();
    assert.ok(solver.advanceTo(frame * dt, [])); await solver.awaitFrameCompletion();
    const wall_ms = performance.now() - start;
    assert.equal(solver.info.simulationPipelineError, undefined);
    const passes = timed ? await profile!.finish() : undefined;
    const f = solver.narrowBandFlipInfo!;
    const work = { particles: f.particles, reseedClipped: f.reseedClipped, ...solver.workDomainInfo, residentPages: solver.pagePoolInfo?.residentPages, claimedPages: solver.pagePoolInfo?.claimedPages, retiredPages: solver.pagePoolInfo?.retiredPages,
      changedTiles: (solver.info as unknown as Record<string, unknown>).uniformMixedDynamicChangedTiles, bandTiles: solver.info.uniformPressureBandTiles, bandCycles: solver.info.uniformPressureBandCycles, bandResidual: solver.info.uniformPressureBandResidual };
    history.update(JSON.stringify(work));
    rows.push({ frame, wall_ms, capacity: f.capacity, ...work, passes });
  }
  const stats = await solver.readStats();
  const physics = { maxSpeed_m_s: stats.maxSpeed_m_s, volumeCellSum: stats.volumeCellSum, representedVolumeCellSum: stats.representedVolumeCellSum, volumeDrift: stats.volumeDrift, front_m: stats.front_m };
  // Every live copyable resource, in creation order.
  const fields: { label: string; hash: string }[] = [];
  if (hashMode === "all") {
    const chunk = 1 << 26;
    for (const b of buffers) {
      if (destroyed.has(b) || !(b.usage & GPUBufferUsage.COPY_SRC) || (b.usage & GPUBufferUsage.MAP_READ)) continue;
      const h = createHash("sha256");
      for (let at = 0; at < b.size; at += chunk) {
        const n = Math.min(chunk, b.size - at), staging = raw.createBuffer({ size: n, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
        const e = raw.createCommandEncoder(); e.copyBufferToBuffer(b, at, staging, 0, n); raw.queue.submit([e.finish()]);
        try { await staging.mapAsync(GPUMapMode.READ); h.update(new Uint8Array(staging.getMappedRange())); staging.unmap(); } catch { h.update("destroyed"); }
        staging.destroy();
      }
      fields.push({ label: b.label, hash: h.digest("hex").slice(0, 16) });
    }
    const texel: Partial<Record<GPUTextureFormat, number>> = { r32float: 4, r32uint: 4, r32sint: 4, rg32float: 8, rg32uint: 8, rgba32float: 16, rgba32uint: 16, rgba32sint: 16, rgba16float: 8, r16float: 2, rg16float: 4, rgba8unorm: 4, r8unorm: 1, rg8unorm: 2 };
    for (const t of textures) {
      const size = texel[t.format];
      if (destroyed.has(t) || !(t.usage & GPUTextureUsage.COPY_SRC) || size === undefined || t.sampleCount !== 1) continue;
      const h = createHash("sha256"), bytesPerRow = Math.ceil(t.width * size / 256) * 256, slice = bytesPerRow * t.height;
      const layers = Math.max(1, Math.floor(chunk / slice));
      for (let z = 0; z < t.depthOrArrayLayers; z += layers) {
        const n = Math.min(layers, t.depthOrArrayLayers - z), staging = raw.createBuffer({ size: n * slice, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
        const e = raw.createCommandEncoder();
        e.copyTextureToBuffer({ texture: t, origin: { x: 0, y: 0, z } }, { buffer: staging, bytesPerRow, rowsPerImage: t.height }, [t.width, t.height, n]);
        raw.queue.submit([e.finish()]);
        try { await staging.mapAsync(GPUMapMode.READ); const bytes = new Uint8Array(staging.getMappedRange());
          for (let r = 0; r < n * t.height; r++) h.update(bytes.subarray(r * bytesPerRow, r * bytesPerRow + t.width * size)); staging.unmap(); } catch { h.update("destroyed"); }
        staging.destroy();
      }
      fields.push({ label: t.label, hash: h.digest("hex").slice(0, 16) });
    }
  }
  const final = audit.snapshot();
  const classes = (s: typeof final) => s.resources.map(r => ({ label: r.label, kind: r.kind, count: r.count, MiB: +(r.bytes / 2 ** 20).toFixed(3) }));
  const particleBytes = (s: typeof final) => s.resources.filter(r => /Narrow-band FLIP|particle/i.test(r.label)).reduce((n, r) => n + r.bytes, 0);
  const late = rows.slice(-Math.max(1, Math.floor(steps / 3)));
  const passMeans: Record<string, number> = {};
  for (const r of late) for (const p of (r.passes ?? []) as { label: string; ms: number }[]) passMeans[p.label] = (passMeans[p.label] ?? 0) + p.ms / late.length;
  const fieldHash = createHash("sha256").update(JSON.stringify(fields)).digest("hex").slice(0, 16);
  const summary = { name, scene: definition.id, steps, tiers: (scene.fluid as Record<string, unknown>).tiers,
    programHash: programs.digest("hex").slice(0, 16), historyHash: history.digest("hex").slice(0, 16), allocationHash: allocations.digest("hex").slice(0, 16), fieldHash,
    liveMiB: +(final.liveBytes / 2 ** 20).toFixed(2), constructedMiB: +(constructed.liveBytes / 2 ** 20).toFixed(2), particleMiB: +(particleBytes(final) / 2 ** 20).toFixed(2),
    particles: rows.map(r => r.particles), lastParticles: rows.at(-1)!.particles, capacity: rows.at(-1)!.capacity, reseedClipped: rows.at(-1)!.reseedClipped, fineTiles: rows.at(-1)!.fineTiles,
    physics, lateWall_ms: late.reduce((n, r) => n + Number(r.wall_ms), 0) / late.length,
    passMeans_ms: Object.fromEntries(Object.entries(passMeans).sort((a, b) => b[1] - a[1]).slice(0, 40)) };
  writeFileSync(`${out}/${name}.json`, JSON.stringify({ ...summary, classes: classes(final), fields, rows: rows.map(r => ({ ...r, passes: undefined })), errors }, null, 2) + "\n");
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({ ...summary, particles: undefined, passMeans_ms: profiled ? summary.passMeans_ms : undefined }));
} finally { solver?.destroy(); await raw.queue.onSubmittedWorkDone(); profile?.destroy(); raw.destroy(); }
