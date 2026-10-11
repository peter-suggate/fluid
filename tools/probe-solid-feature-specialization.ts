/** Frozen-input surface A/B of runtime versus statically absent body and
 * terrain features. Rejects scenes using either feature; does not change
 * scene parameters or physics inputs. Diagnostic only: production keeps
 * the general shader until pipeline selection is implemented and measured.
 * node --import tsx tools/probe-solid-feature-specialization.ts [--steps=96]
 * [--scene=nbflip-figure-9-teaser] [--rounds=8] [--out=FILE]
 */
import assert from "node:assert/strict";
import { writeFileSync, mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { resolveMethodValues } from "../lib/core/method-contract";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { sceneHasTerrain } from "../lib/core/terrain";
import { managedGPUDevice } from "../lib/core/gpu-compilation-manager";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import { uniformNarrowBandMethod as method } from "../lib/methods/uniform/uniform-narrow-band-method";
import { uniformDetailField, type UniformDetailGroup } from "../lib/methods/uniform/uniform-detail-fields";
import type { WebGPUUniformReferenceSolver } from "../lib/methods/uniform/webgpu-uniform-reference";
import { GPUPassProfile } from "./gpu-pass-profile";

export function staticSolidSource(code:string):string{
 const body="fn umBodyCount()->u32{if(!umSolidEnabled()){return 0u;}return min(u32(max(round(umSolidParams.boundary.z),0.0)),12u);}";
 const terrain="fn umSolidHasTerrain()->bool{return umSolidParams.container.w>0.5;}";
 assert.ok(code.includes(body)&&code.includes(terrain),"Expected general solid feature queries");
 return code.replace(body,"fn umBodyCount()->u32{return 0u;}").replace(terrain,"fn umSolidHasTerrain()->bool{return false;}");
}

class SolidFeatureVariants {
  readonly device: GPUDevice;
  private readonly sources = new WeakMap<GPUShaderModule, string>();
  private readonly specialized = new WeakMap<GPUComputePipeline, GPUComputePipeline>();
  constructor(raw: GPUDevice) {
    this.device = new Proxy(raw, { get: (target, key) => {
      if (key === "createShaderModule") return (d: GPUShaderModuleDescriptor) => {
        const module = target.createShaderModule(d); this.sources.set(module, d.code); return module;
      };
      if (key === "createComputePipelineAsync") return async (d: GPUComputePipelineDescriptor) => {
        const pipeline = await target.createComputePipelineAsync(d);
        if (["advect", "advectFine", "advectOwners", "advectWalls", "advectDeferred"].includes(d.compute.entryPoint ?? "")) {
          const source = this.sources.get(d.compute.module); assert.ok(source);
          const module = target.createShaderModule({ label: "Diagnostic static solid features", code: staticSolidSource(source) });
          const errors = (await module.getCompilationInfo()).messages.filter(m => m.type === "error");
          if (errors.length) {
            writeFileSync("/tmp/fluid-static-solid-features.wgsl", staticSolidSource(source));
            throw new Error(errors.map(m => `${m.lineNum}:${m.linePos} ${m.message}`).join("\n"));
          }
          this.specialized.set(pipeline, await target.createComputePipelineAsync({ ...d, compute: { ...d.compute, module } }));
        }
        return pipeline;
      };
      const value = Reflect.get(target, key, target); return typeof value === "function" ? value.bind(target) : value;
    } });
  }
  pick(pipeline: GPUComputePipeline, baseline: boolean): GPUComputePipeline { return baseline ? pipeline : this.specialized.get(pipeline) ?? pipeline; }
}

type Frame = {
 surface: { pipelines:Map<string,GPUComputePipeline>; regularPipelines:Map<string,GPUComputePipeline>; finePipelines:Map<string,GPUComputePipeline>; deferredPipelines:GPUComputePipeline[]; claims:GPUBuffer; encode(e:GPUCommandEncoder,entry:"advect",group:UniformDetailGroup):void };
 surfaceGroups:readonly UniformDetailGroup[];
 fields:{phiScratch:GPUTexture};
};

async function run(): Promise<void> {
  const option = (key: string) => process.argv.find(a => a.startsWith(`--${key}=`))?.slice(key.length + 3);
  const steps = Number(option("steps") ?? 96), rounds = Number(option("rounds") ?? 8);
  assert.ok(Number.isInteger(steps) && steps > 0 && Number.isInteger(rounds) && rounds >= 2);
  const definition = getSceneDefinition(option("scene") ?? "nbflip-figure-9-teaser");
  assert.equal(definition.methodProfile?.methodId, method.id);
  const quality = definition.methodProfile!.quality, scene = structuredClone(sceneDocument(definition));
  assert.equal(scene.rigidBodies.length,0,"Diagnostic requires no live bodies");assert.equal(sceneHasTerrain(scene),false,"Diagnostic requires no terrain");
  const values = resolveMethodValues(method, quality, { ...method.appDefaults, ...definition.methodProfile!.overrides });
  const dt = method.effectiveStep_s?.(scene, values) ?? scene.numerics.fixedDt_s;
  const out = option("out") ?? "artifacts/nbflip-ui-perf/static-solid-surface.json";
  let device: GPUDevice | undefined, solver: WebGPUUniformReferenceSolver | undefined, profile: GPUPassProfile | undefined;
  const errors: string[] = [], buffers: GPUBuffer[] = [];
  try {
    const dawn = await import(pathToFileURL(resolve("node_modules/webgpu/index.js")).href); Object.assign(globalThis, dawn.globals);
    const gpu = createProcessRetainedDawnGPU(dawn, ["backend=metal"]), adapter = await gpu.requestAdapter(); assert.ok(adapter);
    const raw = await adapter.requestDevice({ requiredFeatures: ["timestamp-query", ...(adapter.features.has("subgroups") ? ["subgroups" as const] : [])], requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
    profile = new GPUPassProfile(raw); const variants = new SolidFeatureVariants(profile.device);
    device = managedGPUDevice(variants.device, { requireWorkerRealm: false });
    device.addEventListener("uncapturederror", e => { e.preventDefault(); errors.push(e.error.message); });
    solver = await method.createSolverAsync!(device, scene, quality, values, undefined, () => {}) as WebGPUUniformReferenceSolver;
    for (let n = 1; n <= steps; n++) {
      solver.advanceTo(n * dt, []); await solver.awaitFrameCompletion(); assert.equal(solver.info.simulationPipelineError, undefined);
      if (n % 12 === 0 || n === steps) console.log(JSON.stringify({ phase: "advance", frame: n, time: n * dt }));
    }
    const frame = (solver as unknown as { mixedFrame: Frame }).mixedFrame;
    const surface=frame.surface;
    const maps=[surface.pipelines,surface.regularPipelines,surface.finePipelines];
    const originals=maps.map(m=>new Map(m)),deferred=[...surface.deferredPipelines];
    const encode=(e:GPUCommandEncoder,baseline:boolean)=>{
      maps.forEach((m,i)=>originals[i]!.forEach((p,k)=>m.set(k,variants.pick(p,baseline))));
      deferred.forEach((p,i)=>surface.deferredPipelines[i]=variants.pick(p,baseline));
      surface.encode(e,"advect",frame.surfaceGroups[0]!);
    };
    const field=uniformDetailField(frame.fields.phiScratch);assert.ok(field);
    const textures=[...new Set([field.storage.physical(field.handle),field.storage.baseOf(field.handle)!])];
    const allocate = (size: number, read = false) => { const b = raw.createBuffer({ size, usage: GPUBufferUsage.COPY_DST | (read ? GPUBufferUsage.MAP_READ : GPUBufferUsage.COPY_SRC) }); buffers.push(b); return b; };
    const records = textures.map(texture => { const bytesPerRow = Math.ceil(texture.width * 4 / 256) * 256; return { texture, bytesPerRow, rowsPerImage: texture.height, size: bytesPerRow * texture.height * texture.depthOrArrayLayers, extent: [texture.width, texture.height, texture.depthOrArrayLayers] as [number, number, number] }; });
    const original = records.map(r => allocate(r.size)); const originalNegative = allocate(surface.claims.size);
    let e = raw.createCommandEncoder(); records.forEach((r, i) => e.copyTextureToBuffer({ texture: r.texture }, { buffer: original[i]!, bytesPerRow: r.bytesPerRow, rowsPerImage: r.rowsPerImage }, r.extent));
    e.copyBufferToBuffer(surface.claims, 0, originalNegative, 0, originalNegative.size); raw.queue.submit([e.finish()]);
    const restore = (e: GPUCommandEncoder) => { records.forEach((r, i) => e.copyBufferToTexture({ buffer: original[i]!, bytesPerRow: r.bytesPerRow, rowsPerImage: r.rowsPerImage }, { texture: r.texture }, r.extent)); e.copyBufferToBuffer(originalNegative, 0, surface.claims, 0, originalNegative.size); };
    const capture = async (baseline: boolean) => {
      const reads = records.map(r => allocate(r.size, true)); const negative = allocate(originalNegative.size, true);
      const e = raw.createCommandEncoder(); restore(e); encode(e, baseline);
      records.forEach((r, i) => e.copyTextureToBuffer({ texture: r.texture }, { buffer: reads[i]!, bytesPerRow: r.bytesPerRow, rowsPerImage: r.rowsPerImage }, r.extent));
      e.copyBufferToBuffer(surface.claims, 0, negative, 0, negative.size); raw.queue.submit([e.finish()]);
      return Promise.all([...reads, negative].map(async read => { await read.mapAsync(GPUMapMode.READ); const words = new Uint32Array(read.getMappedRange().slice(0)); read.unmap(); read.destroy(); return words; }));
    };
    const baseline = await capture(true), candidate = await capture(false);
    const dims=[solver.info.nx,solver.info.ny,solver.info.nz];
    const travelWords=2*dims.reduce((n,d)=>n*(d/4+1),1);
    baseline[baseline.length-1]=baseline[baseline.length-1]!.slice(-travelWords);
    candidate[candidate.length-1]=candidate[candidate.length-1]!.slice(-travelWords); let words = 0, mismatches = 0; const first: unknown[] = [];
    baseline.forEach((a, bank) => { const b = candidate[bank]!; words += a.length; for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) { mismatches++; if (first.length < 8) first.push({ bank, word: i, baseline: a[i], candidate: b[i] }); } });
    // Readback disturbs caches; warm both compiled variants before timing.
    e = raw.createCommandEncoder(); for (let n = 0; n < 4; n++) { encode(e, true); encode(e, false); }
    raw.queue.submit([e.finish()]); await raw.queue.onSubmittedWorkDone();
    const passes: Awaited<ReturnType<GPUPassProfile["finish"]>> = [];
    for (let round = 0; round < rounds; round++) for (const baseline of round % 2 ? [false, true] : [true, false]) {
      e=raw.createCommandEncoder(); restore(e);raw.queue.submit([e.finish()]);await raw.queue.onSubmittedWorkDone();
      profile.start(); e = device.createCommandEncoder(); encode(e, baseline);
      device.queue.submit([e.finish()]); await device.queue.onSubmittedWorkDone();
      passes.push(...await profile.finish());
    }
    const samplePairs=passes.reduce((all,p,i)=>{if(i%2===0)all.push(p.ms);else all[all.length-1]!+=p.ms;return all;},[] as number[]);
    const baseSamples:number[]=[],candidateSamples:number[]=[];
    for(let r=0;r<rounds;r++)for(let i=0;i<2;i++){const baseline=r%2===0?i===0:i===1;(baseline?baseSamples:candidateSamples).push(samplePairs[2*r+i]!);}
    const mean=(a:number[])=>a.reduce((n,v)=>n+v,0)/a.length;
    const means={baseline:mean(baseSamples),candidate:mean(candidateSamples)};

    e = raw.createCommandEncoder(); restore(e); raw.queue.submit([e.finish()]); await raw.queue.onSubmittedWorkDone();
    const result = { date: new Date().toISOString(), sceneId: definition.id, values, dt, steps, rounds, words, mismatches, first, means_ms: means, passes, errors };
    mkdirSync(dirname(out), { recursive: true }); writeFileSync(out, JSON.stringify(result, null, 2) + "\n"); console.log(JSON.stringify(result));
    assert.equal(mismatches, 0, "Every surface output bit must match"); assert.deepEqual(errors, []);
  } finally { buffers.forEach(b => b.destroy()); solver?.destroy(); await device?.queue.onSubmittedWorkDone(); profile?.destroy(); device?.destroy(); }
}
if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) await run();
