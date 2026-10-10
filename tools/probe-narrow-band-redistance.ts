/** Same-input u32 parity and GPU timings for NB metric/redistance kernels.
 * node --import tsx tools/probe-narrow-band-redistance.ts --steps=24
 * --lattice tests direct fine lattice loads in metric and crossing seeds.
 * --sampler-lattice tests the equivalent guard inside the resolved sampler.
 * --staged-nearest stages the nearest search's 250 mask words per workgroup.
 * --hoist-gradient moves the width branch outside the gradient tap loop.
 * --squared-distance defers the minimum candidate distance's square root.
 * --static-geometry specializes integer scene roots, keeping spacing runtime.
 * --crossing-reuse observes mask equality across adaptive redistance.
 * --prune removes unreachable functions and unused workgroup declarations.
 * Uses unchanged UI values; --scene selects another NB scene.
 */
import assert from "node:assert/strict";
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { createHash } from "node:crypto";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { resolveMethodValues } from "../lib/core/method-contract";
import { managedGPUDevice } from "../lib/core/gpu-compilation-manager";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import { uniformNarrowBandMethod as method } from "../lib/methods/uniform/uniform-narrow-band-method";
import { uniformDetailField } from "../lib/methods/uniform/uniform-detail-fields";
import { narrowBandRedistanceWGSL } from "../lib/methods/uniform/uniform-narrow-band-redistance.wgsl";
import { narrowBandMembershipWGSL } from "../lib/methods/uniform/uniform-narrow-band-membership.wgsl";
import type { WebGPUUniformReferenceSolver } from "../lib/methods/uniform/webgpu-uniform-reference";
import { GPUPassProfile } from "./gpu-pass-profile";
import { pruneWGSLToEntry } from "./prune-wgsl-entry";
import { CrossingMaskObserver } from "./crossing-mask-observer";
import { dynamicIntegerGeometryWGSL, staticIntegerGeometryWGSL } from "./static-integer-geometry";

export function originalRedistance(code: string): string {
  code = code.replace(`  let next=clamp(q-f.w*f.xyz/g2,vec3f(0),vec3f(1));
  // An unchanged iterate repeats identically against these immutable taps.
  if(all(bitcast<vec3u>(next)==bitcast<vec3u>(q))){break;}
  q=next;`, "  q=clamp(q-f.w*f.xyz/g2,vec3f(0),vec3f(1));").replace(`     // The represented field is immutable throughout this trace.
     if(all(bitcast<vec3u>(next)==bitcast<vec3u>(q))){break;}
`, "");
  // The replacement contains no raw texture operations: it remains valid
  // after the storage ABI has rewritten the enclosing shader module.
  const edgeStart = code.indexOf(" // Fixed edge indices keep");
  const edgeEnd = code.indexOf(" // NB_STATIC_EDGES_END", edgeStart);
  if (edgeStart >= 0) {
    assert.ok(edgeEnd > edgeStart);
    code = code.slice(0, edgeStart) + ` for(var a=0u;a<3u;a++){for(var k=0u;k<8u;k++){
  let b=1u<<a;if((k&b)!=0u){continue;}let other=k|b;
  if(v[k]*v[other]>0.0||v[k]==v[other]){continue;}
  let t=clamp(v[k]/(v[k]-v[other]),0.0,1.0);
  let q=origin+mix(vec3f(umCorner(k,2u)),vec3f(umCorner(other,2u)),t);
  distance=min(distance,length(p-q));
 }}` + code.slice(edgeEnd + " // NB_STATIC_EDGES_END".length);
  }
  const intStart = code.indexOf(" let local=p&vec3u(3u);", code.indexOf("fn nbInterfaceVertex("));
  const intEnd = code.indexOf(" for(var k=0u;k<8u;k++){", intStart);
  assert.ok(intStart >= 0 && intEnd > intStart); code = code.slice(0, intStart) + code.slice(intEnd);
  const fn = code.indexOf("fn nbRedistanceVertex(");
  const comment = code.indexOf(" // Before retiring adjacent h owners", fn);
  const normalize = code.indexOf(" let normalize=", comment);
  const end = code.indexOf("@compute @workgroup_size(128) fn redistanceFine", normalize);
  const lastIf = code.indexOf(" if(normalize&&abs(value)>1e-8)", normalize);
  const store = code.indexOf("\n", lastIf);
  assert.ok(fn >= 0 && comment > fn && normalize > comment && lastIf > normalize && end > store);
  const dimensions = /\bfn\s+UM_D\s*\(/.test(code) ? "UM_D()" : "UM_D";
  const depthA = /\bfn\s+NB_DEPTH_A\s*\(/.test(code) ? "NB_DEPTH_A()" : "NB_DEPTH_A";
  // Preserve the active storage route when the in-place experiment supplies
  // an own-vertex reader. This comparison isolates lazy reads and mask tests;
  // in-place versus copied fields need their separate full-stage replay.
  const valueRead = code.slice(normalize, lastIf).match(/var value=([^;]+);/); assert.ok(valueRead);
  return code.slice(0, comment) + ` // Coarse samples normally retain their reconstructed values.
 var value=${valueRead[1]};
` + code.slice(comment, normalize) + ` if((width==1u&&!nbInterfaceVertex(p))||retiring){
  // An h vertex past the reach has no crossing within three tiles: as the
  // metric field holds one with none near, at least six cells away.
  if((nbBandReach(min(p,${dimensions}-1u))&2u)!=0u){value=bitcast<f32>(atomicLoad(&bins[${depthA}+nbVertexIndex(p)]))*h;}
  else if(abs(value)>1e-8){value=sign(value)*max(abs(value),6.0*h);}
 }` + code.slice(store);
}

const entries = ["buildDistance", "redistanceFine", "redistanceCoarse"];
/** Restrict this experiment to resolved samplers: an unresolved fine owner
 * beside coarse owners may still need a reconstructed hanging vertex. */
export function latticeSamplerWGSL(code: string): string {
  const start = code.indexOf("fn umSampleVertex(p:vec3f)->f32 {");
  const end = code.indexOf("\nfn ", start + 1); assert.ok(start >= 0 && end > start);
  let sampler = code.slice(start, end); assert.ok(sampler.includes("// Resolved: the owner's corners"));
  const fine = "var taps:array<f32,8>;", resolved = "let t=(q-vec3f(origin))/f32(width);var values:array<f32,8>;";
  assert.equal(sampler.split(fine).length, 2); assert.equal(sampler.split(resolved).length, 2);
  sampler = sampler.replace(fine, `${fine}
  if(all(f==floor(f))){return umLoadVertex(vec3u(q));}`);
  sampler = sampler.replace(resolved, `${resolved}
  if(all(t==floor(t))){
   if(width==4u){return umLoadCorner(vec3u(q));}
   return umLoadVertex(vec3u(q));
  }`);
  return code.slice(0, start) + sampler + code.slice(end);
}

export function stagedNearestWGSL(code: string): string {
  const row = code.indexOf("fn nbBandRow("), end = code.indexOf("\n}", row) + 2;
  assert.ok(row >= 0 && end > row);
  const body = code.slice(row, end), load = body.match(/  let nibble=.*;/); assert.ok(load);
  assert.ok(load[0].includes("atomicLoad(&bins[NB_BAND_MASK"));
  const staged = body.replace(load[0], `  let at=2u*(u32(o+2)+5u*(u32(y)/4u+2u-coord.y+5u*(u32(z)/4u+2u-coord.z)))+word;
  let nibble=(nbNearestMasks[at]>>shift)&15u;`);
  code = code.slice(0, row) + "var<workgroup> nbNearestMasks:array<u32,250>;\n" + staged + code.slice(end);
  const entry = code.indexOf("fn depthNearest"), init = code.indexOf("\n", code.indexOf("{", entry)) + 1, next = code.indexOf("\n", init);
  assert.ok(entry >= 0 && next > init); assert.ok(code.slice(init, next).includes("let coord=umTileCoord(tile)"));
  const tiles = /\bfn\s+UM_T\s*\(/.test(code) ? "UM_T()" : "UM_T";
  const mask = /\bfn\s+NB_BAND_MASK\s*\(/.test(code) ? "NB_BAND_MASK()" : "NB_BAND_MASK";
  return code.slice(0, next) + `
 for(var k=lane;k<250u;k+=64u){
  let t=vec3i(coord)+vec3i(umCorner(k/2u,5u))-2;var bits=0u;
  if(all(t>=vec3i(0))&&all(t<vec3i(${tiles}))){
   let tile=u32(t.x)+${tiles}.x*(u32(t.y)+${tiles}.y*u32(t.z));
   bits=atomicLoad(&bins[${mask}+2u*tile+(k&1u)]);
  }
  nbNearestMasks[k]=bits;
 }
 workgroupBarrier();` + code.slice(next);
}

export function hoistedGradientWGSL(code: string): string {
  const start = code.indexOf("fn nbPhiGradient("), end = code.indexOf("\n}", start) + 2;
  assert.ok(start >= 0 && end > start);
  const original = code.slice(start, end);
  const loop = ` for(var k=0u;k<8u;k++){
  let corner=umCorner(k,2u);let vertex=origin+corner*width;
  if(width==4u){v[k]=umLoadCorner(vertex);}else{v[k]=umLoadVertex(vertex);}
  let w=select(vec3f(1)-f,f,corner!=vec3u(0));terms[k]=v[k]*w.x*w.y*w.z;
 }`;
  assert.equal(original.split(loop).length, 2);
  const sample = (load: string) => `for(var k=0u;k<8u;k++){
  let corner=umCorner(k,2u);let vertex=origin+corner*width;
  v[k]=${load}(vertex);
  let w=select(vec3f(1)-f,f,corner!=vec3u(0));terms[k]=v[k]*w.x*w.y*w.z;
 }`;
  return code.slice(0, start) + original.replace(loop, ` if(width==4u){${sample("umLoadCorner")}}else{${sample("umLoadVertex")}}`) + code.slice(end);
}

export function squaredDistanceWGSL(code: string): string {
  const start = code.indexOf("fn nbSurfaceDistance("), end = code.indexOf("\n}", start) + 2;
  assert.ok(start >= 0 && end > start);
  let body = code.slice(start, end);
  assert.equal(body.split("length(p-q)").length, 2); assert.equal(body.split("length(p-origin-q)").length, 2);
  body = body.replace("length(p-q)", "dot(p-q,p-q)").replace("length(p-origin-q)", "dot(p-origin-q,p-origin-q)");
  assert.ok(body.includes(" return distance;"));
  body = body.replace(" return distance;", " return select(sqrt(distance),distance,distance==1e20);");
  return code.slice(0, start) + body + code.slice(end);
}
class Variants {
  readonly device: GPUDevice;
  readonly baseline = new WeakMap<GPUComputePipeline, GPUComputePipeline>();
  readonly pruning: unknown[] = [];
  readonly deferred = new Map<string, { descriptor: GPUComputePipelineDescriptor; source: string; pipeline: GPUComputePipeline }>();
  constructor(readonly raw: GPUDevice, prune = false, lattice = false, samplerLattice = false, stagedNearest = false, hoistGradient = false, squaredDistance = false, staticGeometry = false) {
    const metricExperiment = hoistGradient || squaredDistance;
    const targets = staticGeometry ? [...entries, "depthSeeds", "depthNearest"] : stagedNearest ? ["depthNearest"] : metricExperiment ? ["buildDistance"] : lattice && !prune ? ["buildDistance", "depthSeeds"] : lattice || samplerLattice ? [...entries, "depthSeeds"] : entries;
    const sources = new WeakMap<GPUShaderModule, string>(), originals = new WeakMap<GPUShaderModule, GPUShaderModule>();
    this.device = new Proxy(raw, { get: (target, key) => {
      if (key === "createShaderModule") return (d: GPUShaderModuleDescriptor) => { const code = staticGeometry ? dynamicIntegerGeometryWGSL(d.code) : d.code; const module = target.createShaderModule({ ...d, code }); sources.set(module, code); return module; };
      if (key === "createComputePipelineAsync") return async (d: GPUComputePipelineDescriptor) => {
        const pipeline = await target.createComputePipelineAsync(d);
        if (targets.includes(d.compute.entryPoint ?? "")) {
          if (staticGeometry) {
            const source = sources.get(d.compute.module); assert.ok(source);
            this.deferred.set(d.compute.entryPoint!, { descriptor: d, source, pipeline });
            return pipeline;
          }
          if (prune || lattice || samplerLattice || stagedNearest || metricExperiment) {
            const source = sources.get(d.compute.module); assert.ok(source);
            let specializedSource = source;
            if (lattice && d.compute.entryPoint === "buildDistance") {
              const sample = "  let initial=umSampleVertex(vec3f(p));";
              assert.equal(source.split(sample).length, 2);
              specializedSource = source.replace(sample, `  var initial:f32;
  // This job's tile owns p's positive-side cell, including upper domain faces.
  if(umTileWidth(tile)==1u){initial=umLoadVertex(p);}
  else{initial=umSampleVertex(vec3f(p));}`);
            }
            if (lattice && d.compute.entryPoint === "depthSeeds") {
              const sample = " if(lane<125u){nbBandVertices[lane]=umSampleVertex(vec3f(origin+umCorner(lane,5u)));}";
              assert.equal(source.split(sample).length, 2);
              const dimensions = /\bfn\s+UM_D\s*\(/.test(source) ? "UM_D()" : "UM_D";
              specializedSource = source.replace(sample, ` if(lane<125u){
  let p=origin+umCorner(lane,5u);let cell=min(p,${dimensions}-vec3u(1));
  // High-face vertices belong to the positive-side tile except at the domain.
  if(umTileWidth(umTileAt(cell/4u))==1u){nbBandVertices[lane]=umLoadVertex(p);}
  else{nbBandVertices[lane]=umSampleVertex(vec3f(p));}
 }`);
            }
            if (samplerLattice) specializedSource = latticeSamplerWGSL(specializedSource);
            if (stagedNearest) specializedSource = stagedNearestWGSL(specializedSource);
            if (hoistGradient) specializedSource = hoistedGradientWGSL(specializedSource);
            if (squaredDistance) specializedSource = squaredDistanceWGSL(specializedSource);
            const specialized = prune ? pruneWGSLToEntry(specializedSource, d.compute.entryPoint!) : undefined;
            const candidateSource = specialized?.code ?? specializedSource;
            const module = target.createShaderModule({ label: `Isolated ${d.compute.entryPoint}`, code: candidateSource });
            const messages = (await module.getCompilationInfo()).messages.filter(m => m.type === "error");
            if (messages.length) { writeFileSync("/tmp/narrow-band-redistance-specialized.wgsl", candidateSource); throw new Error(messages.map(m => `${m.lineNum}:${m.linePos} ${m.message}`).join("\n")); }
            const candidate = await target.createComputePipelineAsync({ ...d, compute: { ...d.compute, module } });
            this.baseline.set(candidate, pipeline);
            if (specialized) { const { code: _, ...counts } = specialized; this.pruning.push({ entry: d.compute.entryPoint, ...counts, bytesBefore: source.length, bytesAfter: specialized.code.length }); }
            return candidate;
          }
          let module = originals.get(d.compute.module);
          if (!module) {
            const code = sources.get(d.compute.module); assert.ok(code); const original = originalRedistance(code);
            module = target.createShaderModule({ label: "Original NB redistance", code: original });
            const messages = (await module.getCompilationInfo()).messages.filter(m => m.type === "error");
            if (messages.length) { writeFileSync("/tmp/narrow-band-redistance-reference.wgsl", original); throw new Error(messages.map(m => `${m.lineNum}:${m.linePos} ${m.message}`).join("\n")); }
            originals.set(d.compute.module, module);
          }
          this.baseline.set(pipeline, await target.createComputePipelineAsync({ ...d, compute: { ...d.compute, module } }));
        }
        return pipeline;
      };
      const value = Reflect.get(target, key, target); return typeof value === "function" ? value.bind(target) : value;
    } });
  }
  async specializeGeometry(stage: Stage, dimensions: readonly number[]): Promise<void> {
    for (const [entry, { descriptor, source, pipeline }] of this.deferred) {
      const code = staticIntegerGeometryWGSL(source, dimensions);
      const module = this.raw.createShaderModule({ label: `Static integer geometry ${entry}`, code });
      const errors = (await module.getCompilationInfo()).messages.filter(m => m.type === "error");
      if (errors.length) { writeFileSync("/tmp/narrow-band-static-geometry.wgsl", code); throw new Error(errors.map(m => `${m.lineNum}:${m.linePos} ${m.message}`).join("\n")); }
      const candidate = await this.raw.createComputePipelineAsync({ ...descriptor, compute: { ...descriptor.compute, module } });
      assert.equal(stage.pipelines.get(entry), pipeline); this.baseline.set(candidate, pipeline); stage.pipelines.set(entry, candidate);
    }
  }
}
type Stage = { bins: GPUBuffer; groups?: Record<string, unknown>; pipelines: Map<string, GPUComputePipeline>; dispatchBatch(encoder: GPUCommandEncoder, entries: readonly string[], group: string, label: string): void };
type Frame = { narrowBandFlip: Stage; fields: { phi: GPUTexture } };

async function run(): Promise<void> {
  const option = (key: string) => process.argv.find(a => a.startsWith(`--${key}=`))?.slice(key.length + 3);
  const steps = Number(option("steps") ?? 24), rounds = Number(option("rounds") ?? 12), prune = process.argv.includes("--prune"), lattice = process.argv.includes("--lattice"), samplerLattice = process.argv.includes("--sampler-lattice");
  const observeCrossings = process.argv.includes("--crossing-reuse");
  const stagedNearest = process.argv.includes("--staged-nearest");
  const staticGeometry = process.argv.includes("--static-geometry");
  const hoistGradient = process.argv.includes("--hoist-gradient"), squaredDistance = process.argv.includes("--squared-distance"), metricExperiment = hoistGradient || squaredDistance;
  assert.ok(!(lattice && samplerLattice), "Test call-site and shared sampler guards separately");
  assert.ok(!(stagedNearest && (lattice || samplerLattice)), "Test nearest-mask staging separately");
  assert.ok(!(metricExperiment && (stagedNearest || lattice || samplerLattice)), "Test metric arithmetic separately");
  assert.ok(!(staticGeometry && (prune || stagedNearest || lattice || samplerLattice || metricExperiment)), "Test integer geometry separately");
  assert.ok(Number.isInteger(steps) && steps > 0 && Number.isInteger(rounds) && rounds > 1);
  const definition = getSceneDefinition(option("scene") ?? "nbflip-figure-9-teaser"); assert.equal(definition.methodProfile?.methodId, method.id);
  const quality = definition.methodProfile!.quality, scene = structuredClone(sceneDocument(definition));
  const values = resolveMethodValues(method, quality, { ...method.appDefaults, ...definition.methodProfile!.overrides });
  const dt = method.effectiveStep_s?.(scene, values) ?? scene.numerics.fixedDt_s;
  const out = option("out") ?? "artifacts/nbflip-ui-perf/redistance.json";
  let device: GPUDevice | undefined, solver: WebGPUUniformReferenceSolver | undefined, profile: GPUPassProfile | undefined, crossingObserver: CrossingMaskObserver | undefined;
  const errors: string[] = [], buffers: GPUBuffer[] = [];
  try {
    const dawn = await import(pathToFileURL(resolve("node_modules/webgpu/index.js")).href); Object.assign(globalThis, dawn.globals);
    const gpu = createProcessRetainedDawnGPU(dawn, ["backend=metal"]), adapter = await gpu.requestAdapter(); assert.ok(adapter);
    const raw = await adapter.requestDevice({ requiredFeatures: ["timestamp-query", ...(adapter.features.has("subgroups") ? ["subgroups" as const] : [])], requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
    profile = new GPUPassProfile(raw); const variants = new Variants(profile.device, prune, lattice, samplerLattice, stagedNearest, hoistGradient, squaredDistance, staticGeometry); device = managedGPUDevice(variants.device, { requireWorkerRealm: false });
    device.addEventListener("uncapturederror", e => { e.preventDefault(); errors.push(e.error.message); });
    solver = await method.createSolverAsync!(device, scene, quality, values, undefined, () => {}) as WebGPUUniformReferenceSolver;
    const frame = (solver as unknown as { mixedFrame: Frame }).mixedFrame, stage = frame.narrowBandFlip;
    const crossingReuse: unknown[] = [];
    if (observeCrossings) crossingObserver = new CrossingMaskObserver(stage, raw, [solver.info.nx, solver.info.ny, solver.info.nz]);
    for (let n = 1; n <= steps; n++) {
      solver.advanceTo(n * dt, []); await solver.awaitFrameCompletion(); assert.equal(solver.info.simulationPipelineError, undefined);
      const crossings = await crossingObserver?.take(n); if (crossings) crossingReuse.push(crossings);
      if (n % 12 === 0 || n === steps) console.log(JSON.stringify({ phase: "advance", frame: n, time: n * dt, crossings }));
    }
    crossingObserver?.destroy(); crossingObserver = undefined;
    if (staticGeometry) await variants.specializeGeometry(stage, [solver.info.nx, solver.info.ny, solver.info.nz]);
    const encode = (e: GPUCommandEncoder, kind: string, baseline: boolean) => {
      const list = kind === "metric" ? ["buildDistance"] : kind === "seeds" ? ["depthSeeds"] : kind === "nearest" ? ["depthNearest"] : ["redistanceFine", "redistanceCoarse"];
      const originals = list.map(name => stage.pipelines.get(name)!);
      list.forEach((name, i) => { if (baseline) { const p = variants.baseline.get(originals[i]!); assert.ok(p); stage.pipelines.set(name, p); } });
      const group = kind === "redistance" ? stage.groups?.redistanceStore ? "redistanceStore" : "redistance" : "surface";
      try { stage.dispatchBatch(e, list, group, `probe ${kind} ${baseline ? "baseline" : "candidate"}`); }
      finally { list.forEach((name, i) => stage.pipelines.set(name, originals[i]!)); }
    };
    const allocate = (size: number, read = false) => { const b = raw.createBuffer({ size, usage: GPUBufferUsage.COPY_DST | (read ? GPUBufferUsage.MAP_READ : GPUBufferUsage.COPY_SRC) }); buffers.push(b); return b; };
    const dims = [solver.info.nx, solver.info.ny, solver.info.nz], cells = dims.reduce((a, b) => a * b, 1), vertices = dims.reduce((a, b) => a * (b + 1), 1);
    const field = uniformDetailField(frame.fields.phi);
    const textures = [...new Set(field ? [field.storage.physical(field.handle), field.storage.baseOf(field.handle)!] : [frame.fields.phi])];
    const records = textures.map(texture => { const bytesPerRow = Math.ceil(texture.width * 4 / 256) * 256; return { texture, bytesPerRow, rowsPerImage: texture.height, size: bytesPerRow * texture.height * texture.depthOrArrayLayers, extent: [texture.width, texture.height, texture.depthOrArrayLayers] as [number, number, number] }; });
    const tiles = cells / 64, metricOffset = 4 * (2 * cells + 2 * tiles), metricSize = 4 * vertices;
    const crossingOffset = 4 * (3 * cells + vertices + 2 * tiles + 4), crossingSize = 4 * 3 * tiles;
    const nearestOffset = metricOffset + metricSize, nearestSize = 4 * cells;
    const originals = records.map(r => allocate(r.size)), metricOriginal = allocate(metricSize), crossingOriginal = allocate(crossingSize), nearestOriginal = stagedNearest || staticGeometry ? allocate(nearestSize) : undefined;
    let e = raw.createCommandEncoder(); records.forEach((r, i) => e.copyTextureToBuffer({ texture: r.texture }, { buffer: originals[i]!, bytesPerRow: r.bytesPerRow, rowsPerImage: r.rowsPerImage }, r.extent));
    e.copyBufferToBuffer(stage.bins, metricOffset, metricOriginal, 0, metricSize);
    if (nearestOriginal) e.copyBufferToBuffer(stage.bins, nearestOffset, nearestOriginal, 0, nearestSize);
    e.copyBufferToBuffer(stage.bins, crossingOffset, crossingOriginal, 0, crossingSize); raw.queue.submit([e.finish()]);
    const restore = (e: GPUCommandEncoder, kind: string) => {
      if (kind === "metric") e.copyBufferToBuffer(metricOriginal, 0, stage.bins, metricOffset, metricSize);
      else if (kind === "seeds") e.copyBufferToBuffer(crossingOriginal, 0, stage.bins, crossingOffset, crossingSize);
      else if (kind === "nearest") e.copyBufferToBuffer(nearestOriginal!, 0, stage.bins, nearestOffset, nearestSize);
      else records.forEach((r, i) => e.copyBufferToTexture({ buffer: originals[i]!, bytesPerRow: r.bytesPerRow, rowsPerImage: r.rowsPerImage }, { texture: r.texture }, r.extent));
    };
    const results: unknown[] = [];
    for (const kind of staticGeometry ? ["metric", "seeds", "nearest", "redistance"] : stagedNearest ? ["nearest"] : metricExperiment ? ["metric"] : lattice && !prune ? ["metric", "seeds"] : samplerLattice || lattice ? ["metric", "seeds", "redistance"] : ["metric", "redistance"]) {
      const prepare = (e: GPUCommandEncoder) => { if (kind === "seeds") e.clearBuffer(stage.bins, crossingOffset, crossingSize); else restore(e, kind); };
      const storageSize = kind === "metric" ? metricSize : kind === "nearest" ? nearestSize : crossingSize, storageOffset = kind === "metric" ? metricOffset : kind === "nearest" ? nearestOffset : crossingOffset;
      const capture = async (baseline: boolean) => {
        const reads = kind !== "redistance" ? [allocate(storageSize, true)] : records.map(r => allocate(r.size, true));
        const e = raw.createCommandEncoder(); prepare(e); encode(e, kind, baseline);
        if (kind !== "redistance") e.copyBufferToBuffer(stage.bins, storageOffset, reads[0]!, 0, storageSize);
        else records.forEach((r, i) => e.copyTextureToBuffer({ texture: r.texture }, { buffer: reads[i]!, bytesPerRow: r.bytesPerRow, rowsPerImage: r.rowsPerImage }, r.extent));
        raw.queue.submit([e.finish()]); return Promise.all(reads.map(async read => { await read.mapAsync(GPUMapMode.READ); const words = new Uint32Array(read.getMappedRange().slice(0)); read.unmap(); read.destroy(); return words; }));
      };
      const a = await capture(true), b = await capture(false); let words = 0, mismatches = 0, maxUlp = 0, maxAbsolute = 0; const first: unknown[] = [];
      const ordered = (bits: number) => bits & 0x80000000 ? 0x80000000 - (bits & 0x7fffffff) : 0x80000000 + bits;
      a.forEach((reference, bank) => {
        words += reference.length; const af = new Float32Array(reference.buffer), bf = new Float32Array(b[bank]!.buffer);
        for (let i = 0; i < reference.length; i++) if (reference[i] !== b[bank]![i]) {
          mismatches++;
          if (kind === "metric" || kind === "redistance") { maxUlp = Math.max(maxUlp, Math.abs(ordered(reference[i]!) - ordered(b[bank]![i]!))); maxAbsolute = Math.max(maxAbsolute, Math.abs(af[i]! - bf[i]!)); }
          if (first.length < 8) first.push({ bank, word: i, baseline: reference[i], candidate: b[bank]![i] });
        }
      });
      e = raw.createCommandEncoder(); for (let warm = 0; warm < 4; warm++) { prepare(e); encode(e, kind, true); prepare(e); encode(e, kind, false); } raw.queue.submit([e.finish()]); await raw.queue.onSubmittedWorkDone();
      const passes: Awaited<ReturnType<GPUPassProfile["finish"]>> = [];
      for (let round = 0; round < rounds; round++) for (const baseline of round % 2 ? [false, true] : [true, false]) {
        profile.start(); e = device.createCommandEncoder(); prepare(e); encode(e, kind, baseline);
        device.queue.submit([e.finish()]); await device.queue.onSubmittedWorkDone(); passes.push(...await profile.finish());
      }
      const means = Object.fromEntries(["baseline", "candidate"].map(name => { const ms = passes.filter(p => p.label.endsWith(name)).map(p => p.ms); return [name, ms.reduce((a, b) => a + b, 0) / ms.length]; }));
      e = raw.createCommandEncoder(); restore(e, kind); raw.queue.submit([e.finish()]); await raw.queue.onSubmittedWorkDone();
      results.push({ kind, words, mismatches, maxUlp, maxAbsolute, first, means_ms: means, passes }); console.log(JSON.stringify(results.at(-1)));
    }
    const result = { date: new Date().toISOString(), sceneId: definition.id, values, dt, steps, rounds, prune, lattice, samplerLattice, stagedNearest, hoistGradient, squaredDistance, staticGeometry, inPlaceRedistance: Boolean(stage.groups?.redistanceStore), crossingReuse, pruning: variants.pruning, sourceHash: createHash("sha256").update(narrowBandRedistanceWGSL + narrowBandMembershipWGSL).digest("hex"), results, errors };
    mkdirSync(dirname(out), { recursive: true }); writeFileSync(out, JSON.stringify(result, null, 2) + "\n");
    assert.ok(results.every(r => (r as { mismatches: number }).mismatches === 0), "All metric and phi bits must match"); assert.deepEqual(errors, []);
  } finally { crossingObserver?.destroy(); buffers.forEach(b => b.destroy()); solver?.destroy(); await device?.queue.onSubmittedWorkDone(); profile?.destroy(); device?.destroy(); }
}
if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) await run();
