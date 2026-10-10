/** Same-input, bit-exact A/B of extension publication at unchanged UI values.
 * node --import tsx tools/probe-uniform-extension-publish.ts [--steps=36]
 *   [--scene=nbflip-figure-9-teaser] [--rounds=8] [--out=FILE]
 * Publication reads frozen extension slots/hierarchy and writes disjoint fields.
 * Both variants start from the same output snapshot; all physical/base texels
 * and negative faces are compared as u32 words. Timing excludes copies.
 */
import assert from "node:assert/strict";
import { writeFileSync, mkdirSync } from "node:fs";
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
import type { UniformMixedOwnership } from "../lib/methods/uniform/uniform-mixed-ownership";
import type { WebGPUUniformReferenceSolver } from "../lib/methods/uniform/webgpu-uniform-reference";
import { GPUPassProfile } from "./gpu-pass-profile";

/** Restore the original publication body by moving the unchanged known-value
 * expressions back after cache staging. Works after the detail ABI rewrite,
 * so both variants use exactly the same physical field accessors. */
export function originalPublication(code: string): string {
  const start = code.indexOf("fn uePublishFine(");
  const setup = code.indexOf(" // Resolve seeded and wall faces", start);
  const local = code.indexOf(" let origin=umOrigin(owner);", setup);
  const vote = code.indexOf(" if(needsFar!=0u)", local);
  const staging = code.indexOf(" for(var item=lane;item<81u;item+=64u){", vote);
  const mixed = code.indexOf(" let mixed=atomicLoad(&farMixed);", staging);
  const loop = code.indexOf(" for(var axis=0u;axis<3u;axis++){", mixed);
  const output = code.indexOf(" value.w=", loop);
  assert.ok(start >= 0 && setup > start && local > setup && vote > local && staging > vote && mixed > staging && loop > mixed && output > loop, "Publication probe source must match the candidate");
  const known = code.slice(local, vote).replace(";var needsFar=0u", "").replace(
    "  else{needsFar|=1u<<axis;}",
    "  else if((mixed&(1u<<axis))==0u){value[axis]=umFarValueUniform(face,farState[13u][axis],farLo[39u+axis].w>=0.0);}\n  else{value[axis]=umFarValueStaged(face);}",
  );
  assert.ok(!known.includes("needsFar"));
  const restored = (code.slice(0, setup) + code.slice(staging, loop) + known + code.slice(output)).replace("var<workgroup> farRequired:u32;\n", "");
  // The fixed eight-tap loop also removes dynamically indexed local arrays
  // from coarse publication. Restore both original runtime-bounded loops.
  const far = restored.indexOf("fn umFarValue(");
  const end = restored.indexOf("\n}\n", far) + 3;
  assert.ok(far >= 0 && end > far);
  const bound = restored.includes("umCounts.work.") ? "umCounts.work.w" : "umCounts.w";
  return restored.slice(0, far) + restored.slice(far, end).replaceAll("k<8u", `k<${bound}`) + restored.slice(end);
}

class PublicationVariants {
  readonly device: GPUDevice;
  private readonly sources = new WeakMap<GPUShaderModule, string>();
  private readonly baseline = new WeakMap<GPUComputePipeline, GPUComputePipeline>();
  constructor(raw: GPUDevice) {
    this.device = new Proxy(raw, { get: (target, key) => {
      if (key === "createShaderModule") return (d: GPUShaderModuleDescriptor) => {
        const module = target.createShaderModule(d); this.sources.set(module, d.code); return module;
      };
      if (key === "createComputePipelineAsync") return async (d: GPUComputePipelineDescriptor) => {
        const pipeline = await target.createComputePipelineAsync(d);
        if (["publishFine", "publishSeams", "publishList"].includes(d.compute.entryPoint ?? "")) {
          const source = this.sources.get(d.compute.module); assert.ok(source);
          const module = target.createShaderModule({ label: "Original extension publication", code: originalPublication(source) });
          const errors = (await module.getCompilationInfo()).messages.filter(m => m.type === "error");
          if (errors.length) {
            writeFileSync("/tmp/fluid-publication-baseline.wgsl", originalPublication(source));
            throw new Error(errors.map(m => `${m.lineNum}:${m.linePos} ${m.message}`).join("\n"));
          }
          this.baseline.set(pipeline, await target.createComputePipelineAsync({ ...d, compute: { ...d.compute, module } }));
        }
        return pipeline;
      };
      const value = Reflect.get(target, key, target); return typeof value === "function" ? value.bind(target) : value;
    } });
  }
  pick(pipeline: GPUComputePipeline, baseline: boolean): GPUComputePipeline { return baseline ? this.baseline.get(pipeline) ?? pipeline : pipeline; }
}

type Frame = {
  ownership: UniformMixedOwnership;
  extension: { regularPipelines: Map<string, GPUComputePipeline>; listPipelines: Map<string, GPUComputePipeline>; seamPipelines: Map<string, GPUComputePipeline> };
  extensionGroups: readonly [UniformDetailGroup, UniformDetailGroup];
  fields: { velocityScratch: GPUTexture; negativeScratch: GPUBuffer };
};

async function run(): Promise<void> {
  const option = (key: string) => process.argv.find(a => a.startsWith(`--${key}=`))?.slice(key.length + 3);
  const steps = Number(option("steps") ?? 36), rounds = Number(option("rounds") ?? 8);
  assert.ok(Number.isInteger(steps) && steps > 0 && Number.isInteger(rounds) && rounds >= 2);
  const definition = getSceneDefinition(option("scene") ?? "nbflip-figure-9-teaser");
  assert.equal(definition.methodProfile?.methodId, method.id);
  const quality = definition.methodProfile!.quality, scene = structuredClone(sceneDocument(definition));
  const values = resolveMethodValues(method, quality, { ...method.appDefaults, ...definition.methodProfile!.overrides });
  const dt = method.effectiveStep_s?.(scene, values) ?? scene.numerics.fixedDt_s;
  const out = option("out") ?? "artifacts/nbflip-ui-perf/extension-publication.json";
  let device: GPUDevice | undefined, solver: WebGPUUniformReferenceSolver | undefined, profile: GPUPassProfile | undefined;
  const errors: string[] = [], buffers: GPUBuffer[] = [];
  try {
    const dawn = await import(pathToFileURL(resolve("node_modules/webgpu/index.js")).href); Object.assign(globalThis, dawn.globals);
    const gpu = createProcessRetainedDawnGPU(dawn, ["backend=metal"]), adapter = await gpu.requestAdapter(); assert.ok(adapter);
    const raw = await adapter.requestDevice({ requiredFeatures: ["timestamp-query", ...(adapter.features.has("subgroups") ? ["subgroups" as const] : [])], requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
    profile = new GPUPassProfile(raw); const variants = new PublicationVariants(profile.device);
    device = managedGPUDevice(variants.device, { requireWorkerRealm: false });
    device.addEventListener("uncapturederror", e => { e.preventDefault(); errors.push(e.error.message); });
    solver = await method.createSolverAsync!(device, scene, quality, values, undefined, () => {}) as WebGPUUniformReferenceSolver;
    for (let n = 1; n <= steps; n++) {
      solver.advanceTo(n * dt, []); await solver.awaitFrameCompletion(); assert.equal(solver.info.simulationPipelineError, undefined);
      if (n % 12 === 0 || n === steps) console.log(JSON.stringify({ phase: "advance", frame: n, time: n * dt }));
    }
    const frame = (solver as unknown as { mixedFrame: Frame }).mixedFrame;
    const o = frame.ownership, ext = frame.extension;
    const group = frame.extensionGroups[Number(values.extensionFrontSweeps ?? 2) % 2]!;
    const encode = (e: GPUCommandEncoder, baseline: boolean) => {
      const pass = e.beginComputePass({ label: baseline ? "publish baseline" : "publish candidate" });
      pass.setBindGroup(0, o.bindGroup); pass.setBindGroup(1, group.group);
      o.dispatchTierCounted(pass, variants.pick(ext.regularPipelines.get("publish")!, baseline), 0);
      pass.setBindGroup(1, group.base); o.dispatchRegularCoarseCounted(pass, variants.pick(ext.listPipelines.get("publish")!, baseline));
      pass.setBindGroup(1, group.group); o.dispatchCounted(pass, variants.pick(ext.seamPipelines.get("publish")!, baseline), o.capacity.tiles, "hanging"); pass.end();
    };
    const field = uniformDetailField(frame.fields.velocityScratch);
    const textures = [...new Set(field ? [field.storage.physical(field.handle), field.storage.baseOf(field.handle)!] : [frame.fields.velocityScratch])];
    const allocate = (size: number, read = false) => { const b = raw.createBuffer({ size, usage: GPUBufferUsage.COPY_DST | (read ? GPUBufferUsage.MAP_READ : GPUBufferUsage.COPY_SRC) }); buffers.push(b); return b; };
    const records = textures.map(texture => { const bytesPerRow = Math.ceil(texture.width * 16 / 256) * 256; return { texture, bytesPerRow, rowsPerImage: texture.height, size: bytesPerRow * texture.height * texture.depthOrArrayLayers, extent: [texture.width, texture.height, texture.depthOrArrayLayers] as [number, number, number] }; });
    const original = records.map(r => allocate(r.size)); const originalNegative = allocate(frame.fields.negativeScratch.size);
    let e = raw.createCommandEncoder(); records.forEach((r, i) => e.copyTextureToBuffer({ texture: r.texture }, { buffer: original[i]!, bytesPerRow: r.bytesPerRow, rowsPerImage: r.rowsPerImage }, r.extent));
    e.copyBufferToBuffer(frame.fields.negativeScratch, 0, originalNegative, 0, originalNegative.size); raw.queue.submit([e.finish()]);
    const restore = (e: GPUCommandEncoder) => { records.forEach((r, i) => e.copyBufferToTexture({ buffer: original[i]!, bytesPerRow: r.bytesPerRow, rowsPerImage: r.rowsPerImage }, { texture: r.texture }, r.extent)); e.copyBufferToBuffer(originalNegative, 0, frame.fields.negativeScratch, 0, originalNegative.size); };
    const capture = async (baseline: boolean) => {
      const reads = records.map(r => allocate(r.size, true)); const negative = allocate(originalNegative.size, true);
      const e = raw.createCommandEncoder(); restore(e); encode(e, baseline);
      records.forEach((r, i) => e.copyTextureToBuffer({ texture: r.texture }, { buffer: reads[i]!, bytesPerRow: r.bytesPerRow, rowsPerImage: r.rowsPerImage }, r.extent));
      e.copyBufferToBuffer(frame.fields.negativeScratch, 0, negative, 0, negative.size); raw.queue.submit([e.finish()]);
      return Promise.all([...reads, negative].map(async read => { await read.mapAsync(GPUMapMode.READ); const words = new Uint32Array(read.getMappedRange().slice(0)); read.unmap(); read.destroy(); return words; }));
    };
    const baseline = await capture(true), candidate = await capture(false); let words = 0, mismatches = 0; const first: unknown[] = [];
    baseline.forEach((a, bank) => { const b = candidate[bank]!; words += a.length; for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) { mismatches++; if (first.length < 8) first.push({ bank, word: i, baseline: a[i], candidate: b[i] }); } });
    // Readback disturbs caches; warm both compiled variants before timing.
    e = raw.createCommandEncoder(); for (let n = 0; n < 4; n++) { encode(e, true); encode(e, false); }
    raw.queue.submit([e.finish()]); await raw.queue.onSubmittedWorkDone();
    const passes: Awaited<ReturnType<GPUPassProfile["finish"]>> = [];
    for (let round = 0; round < rounds; round++) for (const baseline of round % 2 ? [false, true] : [true, false]) {
      profile.start(); e = device.createCommandEncoder(); encode(e, baseline);
      device.queue.submit([e.finish()]); await device.queue.onSubmittedWorkDone();
      passes.push(...await profile.finish());
    }
    const means = Object.fromEntries(["baseline", "candidate"].map(name => { const ms = passes.filter(p => p.label === `publish ${name}`).map(p => p.ms); return [name, ms.reduce((a, b) => a + b, 0) / ms.length]; }));
    e = raw.createCommandEncoder(); restore(e); raw.queue.submit([e.finish()]); await raw.queue.onSubmittedWorkDone();
    const result = { date: new Date().toISOString(), sceneId: definition.id, values, dt, steps, rounds, words, mismatches, first, means_ms: means, passes, errors };
    mkdirSync(dirname(out), { recursive: true }); writeFileSync(out, JSON.stringify(result, null, 2) + "\n"); console.log(JSON.stringify(result));
    assert.equal(mismatches, 0, "Every publication output bit must match"); assert.deepEqual(errors, []);
  } finally { buffers.forEach(b => b.destroy()); solver?.destroy(); await device?.queue.onSubmittedWorkDone(); profile?.destroy(); device?.destroy(); }
}
if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) await run();
