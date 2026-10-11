/** Same-input u32 parity and GPU timings for NB metric/redistance kernels.
 * node --import tsx tools/probe-narrow-band-redistance.ts --steps=24
 * --lattice tests direct fine lattice loads in metric and crossing seeds.
 * --sampler-lattice tests the equivalent guard inside the resolved sampler.
 * --staged-nearest stages the nearest search's 250 mask words per workgroup.
 * --global-nearest shares X/XY minima through expired metric scratch storage.
 * --hoist-gradient moves the width branch outside the gradient tap loop.
 * --squared-distance defers the minimum candidate distance's square root.
 * --static-geometry specializes integer scene roots, keeping spacing runtime.
 * --geometry-cache compares raw crossing-corner caching, including preparation.
 * --newton-cache reuses each requested integer-corner Newton solve once per cell.
 * --newton-fused marks requested corners in the existing nearest-search passes.
 * --newton-packed derives masks directly from packed winning offsets.
 * --newton-parallel compares serial and eight-lane producers on one fixture.
 * --integer-crossings specializes only integer samples in crossing detection.
 * --cached-gradient reuses raw corners while a metric trace stays in one cell.
 * --edge-cache compares compact and raw8 edge-root caches, with producer timings.
 * --vertex-cache replaces that cache with raw gradient-tap lattice values.
 * --crossing-reuse observes mask equality across adaptive redistance.
 * --prune removes unreachable functions and unused workgroup declarations.
 * Uses unchanged UI values; --scene selects another NB scene.
 */
import assert from "node:assert/strict";
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { resolveMethodValues } from "../lib/core/method-contract";
import { managedGPUDevice } from "../lib/core/gpu-compilation-manager";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import { uniformNarrowBandMethod as method } from "../lib/methods/uniform/uniform-narrow-band-method";
import { uniformDetailField } from "../lib/methods/uniform/uniform-detail-fields";
import { narrowBandRedistanceWGSL } from "../lib/methods/uniform/uniform-narrow-band-redistance.wgsl";
import { narrowBandMembershipWGSL, narrowBandMembershipSource } from "../lib/methods/uniform/uniform-narrow-band-membership.wgsl";
import type { WebGPUUniformReferenceSolver } from "../lib/methods/uniform/webgpu-uniform-reference";
import { newtonCornerCacheWGSL, newtonFusedRequestsWGSL, newtonGatedConsumerWGSL, newtonParallelProducerWGSL, newtonRequestEntries } from "./newton-corner-cache";
import { plainMetricStorageWGSL } from "./plain-metric-storage";
import { metricClassificationProbe } from "./metric-classification-probe";
import { integerGradientWGSL, integerSurfaceSamplesWGSL, referenceIntegerCrossingsWGSL } from "./integer-gradient";
import { cachedGradientWGSL } from "./cached-gradient";
import { edgeRootCacheWGSL, type EdgeRootLayout } from "./edge-root-cache";
import { GPUPassProfile } from "./gpu-pass-profile";
import { pruneWGSLToEntry } from "./prune-wgsl-entry";
import { CrossingMaskObserver } from "./crossing-mask-observer";
import { dynamicIntegerGeometryWGSL, staticIntegerGeometryWGSL } from "./static-integer-geometry";
import { checkGlobalNearestStrides, globalNearestEntries, globalNearestFixtures, localNearestSearchWGSL } from "./global-nearest-search";

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
  const entry = code.indexOf("fn depthNearestJob("), init = code.indexOf("\n", code.indexOf("{", entry)) + 1, next = code.indexOf("\n", init);
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
export function vertexCachedGeometryWGSL(code: string): string {
  const root = (name: string) => new RegExp(`\\bfn\\s+${name}\\s*\\(`).test(code) ? `${name}()` : name;
  const cells = root("NB_CELLS"), dimensions = root("UM_D"), tiles = root("NB_BAND_TILES");
  assert.ok(code.includes("override nbCachedGeometry:bool=true;"));
  code = code.replace("override nbCachedGeometry:bool=true;", "override nbCachedGeometry:bool=false;");
  const start = code.indexOf("fn depthGeometryJob("), open = code.indexOf("{", start);
  assert.ok(start >= 0 && open > start); let end = open + 1, depth = 1;
  while (depth && end < code.length) { if (code[end] === "{") depth++; else if (code[end] === "}") depth--; end++; }
  assert.equal(depth, 0);
  code = code.slice(0, open + 1) + `
 let tile=group.x;let fine=umTileMinimumWidth(tile)==1u;
 if(!fine&&lane!=0u){return;}
 let p=4u*umTileCoord(tile)+umCorner(lane,4u);var value:f32;
 if(fine){value=umLoadVertex(p);}else{value=umLoadCorner(p);}
 atomicStore(&bins[${cells}+cellIndex(vec3i(p))],bitcast<u32>(value));
}` + code.slice(end);
  const jobs = /depthGeometryJobs=atomicLoad\(&bins\[NB_BAND(?:\(\))?\+1u\]\);/;
  assert.ok(jobs.test(code)); code = code.replace(jobs, `depthGeometryJobs=${tiles};`);
  const gradient = code.indexOf("fn nbPhiGradient("), gradientEnd = code.indexOf("\n}", gradient) + 2;
  assert.ok(gradient >= 0 && gradientEnd > gradient);
  const before = code.slice(gradient, gradientEnd);
  assert.ok(before.includes("v[k]=umLoadCorner(vertex)") && before.includes("v[k]=umLoadVertex(vertex)"));
  const after = before.replace("v[k]=umLoadCorner(vertex)", "v[k]=nbMetricCorner(vertex)").replace("v[k]=umLoadVertex(vertex)", "v[k]=nbMetricVertex(vertex)");
  const loaders = ["Corner", "Vertex"].map(kind => `fn nbMetric${kind}(p:vec3u)->f32{
 if(any(p>=${dimensions})){return umLoad${kind}(p);}
 return bitcast<f32>(atomicLoad(&bins[${cells}+cellIndex(vec3i(p))]));
}
`).join("");
  return code.slice(0, gradient) + loaders + after + code.slice(gradientEnd);
}
class Variants {
  readonly device: GPUDevice;
  readonly baseline = new WeakMap<GPUComputePipeline, GPUComputePipeline>();
  readonly pruning: unknown[] = [];
  readonly deferred = new Map<string, { descriptor: GPUComputePipelineDescriptor; source: string; pipeline: GPUComputePipeline }>();
  readonly nearestReference = new WeakMap<GPUComputePipeline, GPUComputePipeline>();
  readonly nearestStaged = new WeakMap<GPUComputePipeline, GPUComputePipeline>();
  readonly partialGeometry = new WeakMap<GPUComputePipeline, GPUComputePipeline>();
  readonly newtonGeometry = new WeakMap<GPUComputePipeline, GPUComputePipeline>();
  readonly newtonOverflow = new WeakMap<GPUComputePipeline, GPUComputePipeline>();
  readonly newtonSerialGeometry = new WeakMap<GPUComputePipeline, GPUComputePipeline>();
  newtonRequests?: readonly GPUComputePipeline[];
  newtonStatistics?: GPUComputePipeline;
  newtonSolve?: GPUComputePipeline;
  readonly edgeGeometry = new Map<EdgeRootLayout, WeakMap<GPUComputePipeline, GPUComputePipeline>>([["compact", new WeakMap()], ["raw8", new WeakMap()]]);
  constructor(readonly raw: GPUDevice, prune = false, lattice = false, samplerLattice = false, stagedNearest = false, hoistGradient = false, squaredDistance = false, staticGeometry = false, geometryCache = false, vertexCache = false, globalNearest = false, edgeCache = false, newtonCache = false, newtonGated = false, plainMetric = false, newtonReload = false, integerSamples = false, integerGradient = false, newtonFused = false, integerCrossings = false, cachedGradient = false, newtonPacked = false, cachedGradientSplit = false, newtonParallel = false, newtonCapacity?: number) {
    const metricExperiment = hoistGradient || squaredDistance || integerGradient || cachedGradient || (plainMetric && !newtonCache);
    const targets = [...(globalNearest ? ["depthNearestZ"] : []), ...(newtonFused ? ["depthNearestY", "depthNearestZ"] : []), ...(integerCrossings ? ["depthTiles", "depthSeeds"] : integerSamples ? ["buildDistance", "depthTiles", "depthSeeds"] : vertexCache || geometryCache || edgeCache || newtonCache ? ["buildDistance", "depthGeometry"] : staticGeometry ? [...entries, "depthSeeds", ...globalNearestEntries] : stagedNearest ? ["depthNearestZ"] : metricExperiment ? ["buildDistance"] : lattice && !prune ? ["buildDistance", "depthSeeds"] : lattice || samplerLattice ? [...entries, "depthSeeds"] : globalNearest ? [] : entries)];
    const sources = new WeakMap<GPUShaderModule, string>(), originals = new WeakMap<GPUShaderModule, GPUShaderModule>(), partials = new WeakMap<GPUShaderModule, GPUShaderModule>(), edgeModules = new WeakMap<GPUShaderModule, Map<EdgeRootLayout, GPUShaderModule>>(), newtonModules = new WeakMap<GPUShaderModule, GPUShaderModule>();
    this.device = new Proxy(raw, { get: (target, key) => {
      if (key === "createShaderModule") return (d: GPUShaderModuleDescriptor) => { let code = staticGeometry ? dynamicIntegerGeometryWGSL(d.code) : d.code; if (integerSamples || integerCrossings) code = referenceIntegerCrossingsWGSL(code); const module = target.createShaderModule({ ...d, code }); sources.set(module, code); return module; };
      if (key === "createComputePipelineAsync") return async (d: GPUComputePipelineDescriptor) => {
        const pipeline = await target.createComputePipelineAsync(d);
        if (targets.includes(d.compute.entryPoint ?? "")) {
          if ((globalNearest || stagedNearest) && d.compute.entryPoint === "depthNearestZ") {
            const source = sources.get(d.compute.module); assert.ok(source); const code = localNearestSearchWGSL(source);
            const module = target.createShaderModule({ label: "Original local nearest search", code });
            const messages = (await module.getCompilationInfo()).messages.filter(m => m.type === "error");
            if (messages.length) { writeFileSync("/tmp/narrow-band-local-nearest.wgsl", code); throw new Error(messages.map(m => `${m.lineNum}:${m.linePos} ${m.message}`).join("\n")); }
            const descriptor = { ...d, compute: { ...d.compute, module, entryPoint: "depthNearest" } };
            const reference = await target.createComputePipelineAsync(descriptor); this.nearestReference.set(pipeline, reference);
            if (stagedNearest) {
              const staged = target.createShaderModule({ label: "Staged local nearest search", code: stagedNearestWGSL(code) });
              const candidate = await target.createComputePipelineAsync({ ...descriptor, compute: { ...descriptor.compute, module: staged } });
              this.nearestStaged.set(pipeline, candidate); this.baseline.set(candidate, reference);
            }
            return pipeline;
          }
          if (newtonCache) {
            let module = newtonModules.get(d.compute.module);
            if (!module) {
              const source = sources.get(d.compute.module); assert.ok(source); let code = newtonCornerCacheWGSL(source, newtonReload, newtonCapacity);
              if (newtonFused) code = newtonFusedRequestsWGSL(code, newtonPacked);
              if (newtonParallel) code = newtonParallelProducerWGSL(code);
              module = target.createShaderModule({ label: "Requested corner Newton cache", code });
              const errors = (await module.getCompilationInfo()).messages.filter(m => m.type === "error");
              if (errors.length) { writeFileSync("/tmp/narrow-band-newton.wgsl", code); throw new Error(errors.map(m => `${m.lineNum}:${m.linePos} ${m.message}`).join("\n")); }
              newtonModules.set(d.compute.module, module);
            }
            if (newtonGated && d.compute.entryPoint === "buildDistance") {
              const source = sources.get(d.compute.module); assert.ok(source);
              for (const complete of [true, false]) {
                let cached = newtonCornerCacheWGSL(source, newtonReload, newtonCapacity); if (newtonFused) cached = newtonFusedRequestsWGSL(cached, newtonPacked);
                if (newtonParallel) cached = newtonParallelProducerWGSL(cached);
                let code = newtonGatedConsumerWGSL(cached, complete);
                if (plainMetric) code = plainMetricStorageWGSL(code);
                const consumer = target.createShaderModule({ label: `Newton ${complete ? "complete" : "overflow"} cache ${plainMetric ? "plain" : "atomic"} consumer`, code });
                const errors = (await consumer.getCompilationInfo()).messages.filter(m => m.type === "error");
                if (errors.length) { writeFileSync("/tmp/narrow-band-newton-consumer.wgsl", code); throw new Error(errors.map(m => `${m.lineNum}:${m.linePos} ${m.message}`).join("\n")); }
                const compiled = await target.createComputePipelineAsync({ ...d, compute: { ...d.compute, module: consumer } });
                (complete ? this.newtonGeometry : this.newtonOverflow).set(pipeline, compiled);
              }
            } else this.newtonGeometry.set(pipeline, await target.createComputePipelineAsync({ ...d, compute: { ...d.compute, module } }));
            if (newtonParallel && d.compute.entryPoint === "depthGeometry") {
              this.newtonSolve = await target.createComputePipelineAsync({ ...d, compute: { ...d.compute, module, entryPoint: "depthNewtonSolve" } });
              const source = sources.get(d.compute.module); assert.ok(source);
              let serialSource = newtonCornerCacheWGSL(source, newtonReload, newtonCapacity);
              if (newtonFused) serialSource = newtonFusedRequestsWGSL(serialSource, newtonPacked);
              const serialModule = target.createShaderModule({ label: "Serial requested-corner producer", code: serialSource });
              this.newtonSerialGeometry.set(pipeline, await target.createComputePipelineAsync({ ...d, compute: { ...d.compute, module: serialModule } }));
            }
            if (!newtonFused && d.compute.entryPoint === "depthGeometry") this.newtonRequests = await Promise.all(newtonRequestEntries.map(entryPoint => target.createComputePipelineAsync({ ...d, compute: { ...d.compute, module, entryPoint } })));
            if (d.compute.entryPoint === "buildDistance") this.newtonStatistics = await target.createComputePipelineAsync({ ...d, compute: { ...d.compute, module, constants: { ...d.compute.constants, nbNewtonStatistics: 1 } } });
            return pipeline;
          }
          if (edgeCache) {
            let modules = edgeModules.get(d.compute.module);
            if (!modules) { modules = new Map(); edgeModules.set(d.compute.module, modules); }
            for (const layout of ["compact", "raw8"] as const) {
              let module = modules.get(layout);
              if (!module) {
                const source = sources.get(d.compute.module); assert.ok(source); const code = edgeRootCacheWGSL(source, layout);
                module = target.createShaderModule({ label: `Crossing edge roots ${layout}`, code });
                const errors = (await module.getCompilationInfo()).messages.filter(m => m.type === "error");
                if (errors.length) { writeFileSync(`/tmp/narrow-band-edge-${layout}.wgsl`, code); throw new Error(errors.map(m => `${m.lineNum}:${m.linePos} ${m.message}`).join("\n")); }
                modules.set(layout, module);
              }
              this.edgeGeometry.get(layout)!.set(pipeline, await target.createComputePipelineAsync({ ...d, compute: { ...d.compute, module } }));
            }
            if (!geometryCache) return pipeline;
          }
          if (vertexCache) {
            let module = originals.get(d.compute.module);
            if (!module) {
              const source = sources.get(d.compute.module); assert.ok(source); const code = vertexCachedGeometryWGSL(source);
              module = target.createShaderModule({ label: "Raw metric vertex cache", code });
              const messages = (await module.getCompilationInfo()).messages.filter(m => m.type === "error");
              if (messages.length) { writeFileSync("/tmp/narrow-band-vertex-cache.wgsl", code); throw new Error(messages.map(m => `${m.lineNum}:${m.linePos} ${m.message}`).join("\n")); }
              originals.set(d.compute.module, module);
            }
            const candidate = await target.createComputePipelineAsync({ ...d, compute: { ...d.compute, module } }); this.baseline.set(candidate, pipeline); return candidate;
          }
          if (geometryCache) {
            if (d.compute.entryPoint === "buildDistance") this.baseline.set(pipeline, await target.createComputePipelineAsync({ ...d, compute: { ...d.compute, constants: { ...d.compute.constants, nbCachedGeometry: 0 } } }));
            let partial = partials.get(d.compute.module);
            if (!partial) {
              const source = sources.get(d.compute.module); assert.ok(source);
              const capacity = /fn nbGeometryCapacity\(\)->u32\{return[^}]+\}/g;
              assert.equal([...source.matchAll(capacity)].length, 1);
              partial = target.createShaderModule({ label: "One-cell crossing corner cache", code: source.replace(capacity, "fn nbGeometryCapacity()->u32{return 1u;}") });
              partials.set(d.compute.module, partial);
            }
            this.partialGeometry.set(pipeline, await target.createComputePipelineAsync({ ...d, compute: { ...d.compute, module: partial } }));
            return pipeline;
          }
          if (staticGeometry) {
            const source = sources.get(d.compute.module); assert.ok(source);
            this.deferred.set(d.compute.entryPoint!, { descriptor: d, source, pipeline });
            return pipeline;
          }
          if (prune || lattice || samplerLattice || stagedNearest || metricExperiment || integerSamples || integerCrossings) {
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
            if (integerSamples || integerCrossings) specializedSource = integerSurfaceSamplesWGSL(specializedSource, !integerCrossings);
            if (cachedGradient) specializedSource = cachedGradientWGSL(specializedSource, cachedGradientSplit);
            if (integerGradient) specializedSource = integerGradientWGSL(specializedSource, integerSamples);
            if (plainMetric && !newtonCache) specializedSource = plainMetricStorageWGSL(specializedSource);
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
  const waitIdle = process.argv.includes("--wait-idle");
  const externalProbes = () => execFileSync("ps", ["-axo", "pid=,comm=,args="], { encoding: "utf8" }).split("\n").flatMap(line => {
    const match = /^\s*(\d+)\s+\S*\bnode\s+(.*)$/.exec(line);
    return match && Number(match[1]) !== process.pid && /(?:probe-|benchmark-)/.test(match[2]!) ? [{ pid: Number(match[1]), command: match[2] }] : [];
  });
  const timingExternalProcesses: unknown[] = [];
  const steps = Number(option("steps") ?? 24), rounds = Number(option("rounds") ?? 12), prune = process.argv.includes("--prune"), lattice = process.argv.includes("--lattice"), samplerLattice = process.argv.includes("--sampler-lattice");
  const observeCrossings = process.argv.includes("--crossing-reuse");
  const stagedNearest = process.argv.includes("--staged-nearest");
  const globalNearest = process.argv.includes("--global-nearest");
  const staticGeometry = process.argv.includes("--static-geometry");
  const geometryCache = process.argv.includes("--geometry-cache");
  const vertexCache = process.argv.includes("--vertex-cache");
  const edgeCache = process.argv.includes("--edge-cache");
  const newtonPacked = process.argv.includes("--newton-packed"), newtonParallel = process.argv.includes("--newton-parallel");
  const newtonFused = process.argv.includes("--newton-fused") || newtonPacked;
  const newtonGated = process.argv.includes("--newton-gated") || newtonFused || newtonParallel, plainMetric = process.argv.includes("--plain-metric");
  const integerSamples = process.argv.includes("--integer-samples"), integerGradient = process.argv.includes("--integer-gradient");
  const integerCrossings = process.argv.includes("--integer-crossings"), cachedGradientSplit = process.argv.includes("--cached-gradient-split");
  const cachedGradient = process.argv.includes("--cached-gradient") || cachedGradientSplit;
  assert.ok(!(integerCrossings && (integerSamples || integerGradient || cachedGradient)), "Test crossing integer samples separately");
  assert.ok(!(cachedGradient && (integerSamples || integerGradient)), "Test trace corner reuse separately");
  const newtonReload = process.argv.includes("--newton-reload");
  const newtonCapacity = option("newton-capacity") === undefined ? undefined : Number(option("newton-capacity"));
  assert.ok(newtonCapacity === undefined || (Number.isSafeInteger(newtonCapacity) && newtonCapacity >= 0 && newtonCapacity <= 0xffffffff));
  const newtonCache = process.argv.includes("--newton-cache") || newtonGated;
  assert.ok(!plainMetric || !newtonCache || newtonGated, "Plain metric tests either the original consumer or gated Newton consumers");
  const allowedEdgeUlp = Number(option("max-ulp") ?? 0);
  const allowedAbsolute = Number(option("max-absolute") ?? 0);
  const hoistGradient = process.argv.includes("--hoist-gradient"), squaredDistance = process.argv.includes("--squared-distance"), metricExperiment = hoistGradient || squaredDistance || integerGradient || cachedGradient || (plainMetric && !newtonCache);
  assert.ok(!((integerSamples || integerGradient || integerCrossings || cachedGradient) && (newtonCache || edgeCache || globalNearest || geometryCache || vertexCache || prune || stagedNearest || lattice || samplerLattice || hoistGradient || squaredDistance || staticGeometry || plainMetric)), "Test integer samples and first gradient independently of other transforms");
  assert.ok(!(lattice && samplerLattice), "Test call-site and shared sampler guards separately");
  assert.ok(!(stagedNearest && (lattice || samplerLattice)), "Test nearest-mask staging separately");
  assert.ok(!(metricExperiment && (stagedNearest || lattice || samplerLattice)), "Test metric arithmetic separately");
  assert.ok(!(staticGeometry && (prune || stagedNearest || lattice || samplerLattice || metricExperiment)), "Test integer geometry separately");
  assert.ok(!(geometryCache && (prune || stagedNearest || lattice || samplerLattice || metricExperiment || staticGeometry)), "Test geometry caching separately");
  assert.ok(!(vertexCache && (geometryCache || prune || stagedNearest || lattice || samplerLattice || metricExperiment || staticGeometry)), "Test raw vertex caching separately");
  assert.ok(!(newtonCache && (edgeCache || globalNearest || geometryCache || vertexCache || prune || stagedNearest || lattice || samplerLattice || metricExperiment || staticGeometry)), "Test Newton caching separately");
  assert.ok(!(edgeCache && (vertexCache || prune || stagedNearest || lattice || samplerLattice || metricExperiment || staticGeometry)), "Test edge caching separately from unrelated metric transforms");
  assert.ok(!(globalNearest && (vertexCache || prune || stagedNearest || lattice || samplerLattice || metricExperiment || staticGeometry)), "Test global nearest search separately");
  assert.ok(Number.isSafeInteger(allowedEdgeUlp) && allowedEdgeUlp >= 0);
  assert.ok(Number.isFinite(allowedAbsolute) && allowedAbsolute >= 0);
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
    profile = new GPUPassProfile(raw); const variants = new Variants(profile.device, prune, lattice, samplerLattice, stagedNearest, hoistGradient, squaredDistance, staticGeometry, geometryCache, vertexCache, globalNearest, edgeCache, newtonCache, newtonGated, plainMetric, newtonReload, integerSamples, integerGradient, newtonFused, integerCrossings, cachedGradient, newtonPacked, cachedGradientSplit, newtonParallel, newtonCapacity); device = managedGPUDevice(variants.device, { requireWorkerRealm: false });
    device.addEventListener("uncapturederror", e => { e.preventDefault(); errors.push(e.error.message); });
    solver = await method.createSolverAsync!(device, scene, quality, values, undefined, () => {}) as WebGPUUniformReferenceSolver;
    const frame = (solver as unknown as { mixedFrame: Frame }).mixedFrame, stage = frame.narrowBandFlip;
    const cachedEntries = integerCrossings ? ["depthTiles", "depthSeeds"] : integerSamples ? ["buildDistance", "depthTiles", "depthSeeds"] : integerGradient || cachedGradient || (plainMetric && !newtonCache) ? ["buildDistance"] : vertexCache ? ["depthGeometry", "buildDistance"] : geometryCache ? ["buildDistance"] : [];
    const cachedPipelines = cachedEntries.map(entry => stage.pipelines.get(entry)!);
    cachedEntries.forEach((entry, i) => { const baseline = variants.baseline.get(cachedPipelines[i]!); assert.ok(baseline); stage.pipelines.set(entry, baseline); });
    const crossingReuse: unknown[] = [];
    if (observeCrossings) crossingObserver = new CrossingMaskObserver(stage, raw, [solver.info.nx, solver.info.ny, solver.info.nz]);
    for (let n = 1; n <= steps; n++) {
      solver.advanceTo(n * dt, []); await solver.awaitFrameCompletion(); assert.equal(solver.info.simulationPipelineError, undefined);
      const crossings = await crossingObserver?.take(n); if (crossings) crossingReuse.push(crossings);
      if (n % 12 === 0 || n === steps) console.log(JSON.stringify({ phase: "advance", frame: n, time: n * dt, crossings }));
    }
    crossingObserver?.destroy(); crossingObserver = undefined;
    cachedEntries.forEach((entry, i) => stage.pipelines.set(entry, cachedPipelines[i]!));
    if (staticGeometry) await variants.specializeGeometry(stage, [solver.info.nx, solver.info.ny, solver.info.nz]);
    if (globalNearest || stagedNearest) {
      const key = stage.pipelines.get("depthNearestZ")!;
      const local = stagedNearest ? variants.nearestStaged.get(key) : variants.nearestReference.get(key); assert.ok(local);
      stage.pipelines.set("depthNearest", local);
    }
    if (newtonCache) {
      assert.ok((newtonFused || variants.newtonRequests) && variants.newtonStatistics);
      if (newtonParallel) { assert.ok(variants.newtonSolve); stage.pipelines.set("depthNewtonSolve", variants.newtonSolve); }
      if (variants.newtonRequests) newtonRequestEntries.forEach((entry, i) => stage.pipelines.set(entry, variants.newtonRequests![i]!));
      if (newtonGated) { const overflow = variants.newtonOverflow.get(stage.pipelines.get("buildDistance")!); assert.ok(overflow); stage.pipelines.set("buildDistanceOverflow", overflow); }
    }
    let fallbackMap: GPUBuffer | undefined;
    const encode = (e: GPUCommandEncoder, kind: string, baseline: boolean, split = false) => {
      if (kind.startsWith("metricNewton")) {
        const serial = kind.startsWith("metricNewtonSerial"), statistics = kind.endsWith("Stats");
        const originalEntries = [...(newtonFused ? ["depthNearestY", "depthNearestZ"] : []), "depthGeometry", "buildDistance"], originals = originalEntries.map(name => stage.pipelines.get(name)!);
        const list = [...(newtonFused ? ["depthNearestX"] : baseline ? [] : newtonRequestEntries), ...originalEntries.flatMap(name => name === "depthGeometry" && newtonParallel && !serial && !baseline ? [name, "depthNewtonSolve"] : [name]), ...(!baseline && newtonGated && !statistics ? ["buildDistanceOverflow"] : [])];
        if (!baseline) originalEntries.forEach((name, i) => {
          const variant = name === "buildDistance" && statistics ? variants.newtonStatistics : name === "depthGeometry" && serial ? variants.newtonSerialGeometry.get(originals[i]!) : variants.newtonGeometry.get(originals[i]!);
          assert.ok(variant); stage.pipelines.set(name, variant);
        });
        try {
          e.clearBuffer(stage.bins, 4 * solver!.info.nx * solver!.info.ny * solver!.info.nz, baseline ? 4 : 32);
          if (split) for (const entry of list) stage.dispatchBatch(e, [entry], "surface", `probe ${kind} ${entry} ${baseline ? "baseline" : "candidate"}`);
          else stage.dispatchBatch(e, list, "surface", `probe ${kind} ${baseline ? "baseline" : "candidate"}`);
        } finally { originalEntries.forEach((name, i) => stage.pipelines.set(name, originals[i]!)); }
        return;
      }
      if (kind.startsWith("metricEdge")) {
        const layout: EdgeRootLayout = kind === "metricEdgeCompact" ? "compact" : "raw8";
        const list = ["depthGeometry", "buildDistance"], originals = list.map(name => stage.pipelines.get(name)!);
        if (!baseline) list.forEach((name, i) => { const variant = variants.edgeGeometry.get(layout)!.get(originals[i]!); assert.ok(variant); stage.pipelines.set(name, variant); });
        try {
          e.clearBuffer(stage.bins, 4 * solver!.info.nx * solver!.info.ny * solver!.info.nz, 4);
          if (split) for (const entry of list) stage.dispatchBatch(e, [entry], "surface", `probe ${kind} ${entry} ${baseline ? "baseline" : "candidate"}`);
          else stage.dispatchBatch(e, list, "surface", `probe ${kind} ${baseline ? "baseline" : "candidate"}`);
        } finally { list.forEach((name, i) => stage.pipelines.set(name, originals[i]!)); }
        return;
      }
      if (globalNearest && kind === "nearest") { stage.dispatchBatch(e, baseline ? ["depthNearest"] : globalNearestEntries, "surface", `probe ${kind} ${baseline ? "baseline" : "candidate"}`); return; }
      const list = vertexCache ? ["depthGeometry", "buildDistance"] : kind.startsWith("metric") ? ["buildDistance"] : kind === "seeds" ? integerSamples || integerCrossings ? ["depthTiles", "depthSeeds"] : ["depthSeeds"] : kind === "nearest" ? stagedNearest ? ["depthNearest"] : [...globalNearestEntries] : ["redistanceFine", "redistanceCoarse"];
      const originals = list.map(name => stage.pipelines.get(name)!);
      list.forEach((name, i) => {
        const replacement = baseline ? variants.baseline.get(originals[i]!) : kind === "metricPartial" ? variants.partialGeometry.get(originals[i]!) : undefined;
        if (baseline || kind === "metricPartial") { assert.ok(replacement); stage.pipelines.set(name, replacement); }
      });
      const geometry = stage.pipelines.get("depthGeometry");
      if (kind === "metricPartial" && !baseline) { const partial = variants.partialGeometry.get(geometry!); assert.ok(partial); stage.pipelines.set("depthGeometry", partial); }
      const group = kind === "redistance" ? stage.groups?.redistanceStore ? "redistanceStore" : "redistance" : "surface";
      try {
        let dispatchEntries = list;
        if (vertexCache) e.clearBuffer(stage.bins, 4 * solver!.info.nx * solver!.info.ny * solver!.info.nz, 4);
        if (geometryCache && !baseline) {
          const cells = solver!.info.nx * solver!.info.ny * solver!.info.nz;
          e.clearBuffer(stage.bins, 4 * cells, 4);
          if (kind === "metricFallback") {
            stage.dispatchBatch(e, ["depthGeometry"], group, "probe forced cache overflow");
            e.copyBufferToBuffer(fallbackMap!, 0, stage.bins, crossingOffset + crossingSize, 4 * tiles);
          } else dispatchEntries = ["depthGeometry", ...list];
        }
        stage.dispatchBatch(e, dispatchEntries, group, `probe ${kind} ${baseline ? "baseline" : "candidate"}`);
      }
      finally { list.forEach((name, i) => stage.pipelines.set(name, originals[i]!)); if (geometry) stage.pipelines.set("depthGeometry", geometry); }
    };
    const allocate = (size: number, read = false) => { const b = raw.createBuffer({ size, usage: GPUBufferUsage.COPY_DST | (read ? GPUBufferUsage.MAP_READ : GPUBufferUsage.COPY_SRC) }); buffers.push(b); return b; };
    const dims = [solver.info.nx, solver.info.ny, solver.info.nz], cells = dims.reduce((a, b) => a * b, 1), vertices = dims.reduce((a, b) => a * (b + 1), 1);
    const field = uniformDetailField(frame.fields.phi);
    const textures = [...new Set(field ? [field.storage.physical(field.handle), field.storage.baseOf(field.handle)!] : [frame.fields.phi])];
    const records = textures.map(texture => { const bytesPerRow = Math.ceil(texture.width * 4 / 256) * 256; return { texture, bytesPerRow, rowsPerImage: texture.height, size: bytesPerRow * texture.height * texture.depthOrArrayLayers, extent: [texture.width, texture.height, texture.depthOrArrayLayers] as [number, number, number] }; });
    const tiles = cells / 64, metricOffset = 4 * (2 * cells + 2 * tiles), metricSize = 4 * vertices;
    const crossingOffset = 4 * (3 * cells + vertices + 2 * tiles + 4), crossingSize = 4 * 3 * tiles;
    const nearestOffset = metricOffset + metricSize, nearestSize = 4 * cells;
    if (geometryCache) { fallbackMap = allocate(4 * tiles); raw.queue.writeBuffer(fallbackMap, 0, new Uint32Array(tiles).fill(0xffffffff)); }
    const originals = records.map(r => allocate(r.size)), metricOriginal = allocate(metricSize), crossingOriginal = allocate(crossingSize), bandOriginal = globalNearest ? allocate(4 * (4 + 7 * tiles)) : undefined, nearestOriginal = stagedNearest || staticGeometry || globalNearest ? allocate(nearestSize) : undefined;
    let e = raw.createCommandEncoder(); records.forEach((r, i) => e.copyTextureToBuffer({ texture: r.texture }, { buffer: originals[i]!, bytesPerRow: r.bytesPerRow, rowsPerImage: r.rowsPerImage }, r.extent));
    e.copyBufferToBuffer(stage.bins, metricOffset, metricOriginal, 0, metricSize);
    if (nearestOriginal) e.copyBufferToBuffer(stage.bins, nearestOffset, nearestOriginal, 0, nearestSize);
    if (bandOriginal) e.copyBufferToBuffer(stage.bins, crossingOffset - 16, bandOriginal, 0, bandOriginal.size);
    e.copyBufferToBuffer(stage.bins, crossingOffset, crossingOriginal, 0, crossingSize); raw.queue.submit([e.finish()]);
    // The Newton request pass borrows expired A scratch. Its inactive words
    // are not metric outputs: readers are guarded by the distance/reach bank.
    // Still compare every vertex written by buildDistance, including domain
    // faces, and separately report any differences in the expired scratch.
    let metricReach: Uint32Array | undefined;
    if (newtonCache || integerGradient || cachedGradient) {
      const read = allocate(4 * tiles, true); e = raw.createCommandEncoder();
      e.copyBufferToBuffer(stage.bins, crossingOffset + 8 * tiles, read, 0, read.size); raw.queue.submit([e.finish()]);
      await read.mapAsync(GPUMapMode.READ); metricReach = new Uint32Array(read.getMappedRange()).slice(); read.unmap();
    }
    const restore = (e: GPUCommandEncoder, kind: string) => {
      if (kind.startsWith("metric")) e.copyBufferToBuffer(metricOriginal, 0, stage.bins, metricOffset, metricSize);
      else if (kind === "seeds") e.copyBufferToBuffer(crossingOriginal, 0, stage.bins, crossingOffset, crossingSize);
      else if (kind === "nearest") e.copyBufferToBuffer(nearestOriginal!, 0, stage.bins, nearestOffset, nearestSize);
      else records.forEach((r, i) => e.copyBufferToTexture({ buffer: originals[i]!, bytesPerRow: r.bytesPerRow, rowsPerImage: r.rowsPerImage }, { texture: r.texture }, r.extent));
    };
    const results: unknown[] = [];
    const kinds = newtonCache ? newtonParallel ? ["metricNewtonSerial", "metricNewton"] : ["metricNewton"] : integerCrossings ? ["seeds"] : integerSamples ? ["metric", "seeds"] : edgeCache && !globalNearest && !geometryCache ? [] : vertexCache ? ["metric"] : geometryCache ? [...(globalNearest ? ["nearest"] : []), "metric", "metricFallback", "metricPartial"] : staticGeometry ? ["metric", "seeds", "nearest", "redistance"] : stagedNearest || globalNearest ? ["nearest"] : metricExperiment ? ["metric"] : lattice && !prune ? ["metric", "seeds"] : samplerLattice || lattice ? ["metric", "seeds", "redistance"] : ["metric", "redistance"];
    if (edgeCache) kinds.push("metricEdgeCompact", "metricEdgeRaw8");
    for (const kind of kinds) {
      const prepare = (e: GPUCommandEncoder) => { if (kind === "seeds") { e.clearBuffer(stage.bins, crossingOffset, crossingSize); if (integerSamples || integerCrossings) e.clearBuffer(stage.bins, crossingOffset - 16, 4); } else restore(e, kind); };
      const storageSize = kind.startsWith("metric") ? metricSize : kind === "nearest" ? nearestSize : crossingSize, storageOffset = kind.startsWith("metric") ? metricOffset : kind === "nearest" ? nearestOffset : crossingOffset;
      const capture = async (baseline: boolean) => {
        const reads = kind !== "redistance" ? [allocate(storageSize, true), ...(newtonFused ? [allocate(nearestSize, true)] : [])] : records.map(r => allocate(r.size, true));
        const e = raw.createCommandEncoder(); prepare(e); encode(e, kind, baseline);
        if (kind !== "redistance") { e.copyBufferToBuffer(stage.bins, storageOffset, reads[0]!, 0, storageSize); if (newtonFused) e.copyBufferToBuffer(stage.bins, nearestOffset, reads[1]!, 0, nearestSize); }
        else records.forEach((r, i) => e.copyTextureToBuffer({ texture: r.texture }, { buffer: reads[i]!, bytesPerRow: r.bytesPerRow, rowsPerImage: r.rowsPerImage }, r.extent));
        raw.queue.submit([e.finish()]); return Promise.all(reads.map(async read => { await read.mapAsync(GPUMapMode.READ); const words = new Uint32Array(read.getMappedRange().slice(0)); read.unmap(); read.destroy(); return words; }));
      };
      const a = await capture(true), b = await capture(false); let words = 0, mismatches = 0, integerMismatches = 0, maxUlp = 0, maxAbsolute = 0, maxRelativeAwayZero = 0, inactiveScratchDifferences = 0, signChanges = 0, nonfiniteDifferences = 0; const first: unknown[] = [];
      const absoluteThresholds = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, Infinity], absoluteHistogram = absoluteThresholds.map(() => 0);
      const nodalThresholds = [-4.5, -4, -1, -0.875, -Math.sqrt(0.875 ** 2 - 0.125), 0, 0.9].map(Math.fround), nodalClassificationChanges = nodalThresholds.map(() => 0);
      let largestAbsolute: unknown, largestUlp: unknown;
      const ordered = (bits: number) => bits & 0x80000000 ? 0x80000000 - (bits & 0x7fffffff) : 0x80000000 + bits;
      a.forEach((reference, bank) => {
        words += reference.length; const af = new Float32Array(reference.buffer), bf = new Float32Array(b[bank]!.buffer);
        for (let i = 0; i < reference.length; i++) if (reference[i] !== b[bank]![i]) {
          if (metricReach && kind.startsWith("metricNewton") && bank === 0) {
            const x = Math.min(i % (dims[0]! + 1), dims[0]! - 1), y = Math.min(Math.floor(i / (dims[0]! + 1)) % (dims[1]! + 1), dims[1]! - 1), z = Math.min(Math.floor(i / ((dims[0]! + 1) * (dims[1]! + 1))), dims[2]! - 1);
            const tile = (x >>> 2) + (dims[0]! / 4) * ((y >>> 2) + (dims[1]! / 4) * (z >>> 2));
            if ((metricReach[tile]! & 2) === 0) { inactiveScratchDifferences++; continue; }
          }
          mismatches++;
          if (newtonFused && bank > 0) {
            integerMismatches++;
            if (first.length < 8) first.push({ bank, word: i, baseline: reference[i], candidate: b[bank]![i] });
            continue;
          }
          if (kind.startsWith("metric") || kind === "redistance") {
            const ulp = Math.abs(ordered(reference[i]!) - ordered(b[bank]![i]!)), absolute = Math.abs(af[i]! - bf[i]!);
            const location = { bank, word: i, baseline: af[i], candidate: bf[i], ulp, absolute };
            if (ulp > maxUlp) { maxUlp = ulp; largestUlp = location; }
            if (absolute > maxAbsolute) { maxAbsolute = absolute; largestAbsolute = location; }
            if (Math.abs(af[i]!) >= 1e-3) maxRelativeAwayZero = Math.max(maxRelativeAwayZero, absolute / Math.abs(af[i]!));
            if (!Number.isFinite(af[i]) || !Number.isFinite(bf[i])) nonfiniteDifferences++;
            else absoluteHistogram[absoluteThresholds.findIndex(limit => absolute <= limit)]!++;
            if (af[i] !== 0 && bf[i] !== 0 && Math.sign(af[i]!) !== Math.sign(bf[i]!)) signChanges++;
            nodalThresholds.forEach((threshold, j) => { if ((af[i]! <= threshold) !== (bf[i]! <= threshold)) nodalClassificationChanges[j]!++; });
          }
          if (first.length < 8) first.push({ bank, word: i, baseline: reference[i], candidate: b[bank]![i] });
        }
      });
      if (kind === "metricFallback" || kind === "metricPartial") {
        let partial: { capacity: number; cachedCells: number; straddlingTiles: number } | undefined;
        if (kind === "metricPartial") {
          const read = allocate(4 * tiles, true), masks = allocate(8 * tiles, true), e = raw.createCommandEncoder();
          e.copyBufferToBuffer(stage.bins, crossingOffset + crossingSize, read, 0, 4 * tiles);
          e.copyBufferToBuffer(stage.bins, crossingOffset, masks, 0, 8 * tiles); raw.queue.submit([e.finish()]);
          await Promise.all([read.mapAsync(GPUMapMode.READ), masks.mapAsync(GPUMapMode.READ)]);
          const bases = new Uint32Array(read.getMappedRange()), bits = new Uint32Array(masks.getMappedRange());
          const popcount = (word: number) => { let n = 0; for (let b = word >>> 0; b; b = (b & (b - 1)) >>> 0) n++; return n; };
          let cachedCells = 0, straddlingTiles = 0;
          for (let t = 0; t < tiles; t++) if (bases[t] === 0) { const count = popcount(bits[2 * t]!) + popcount(bits[2 * t + 1]!); if (count) cachedCells++; if (count > 1) straddlingTiles++; }
          read.unmap(); masks.unmap(); partial = { capacity: 1, cachedCells, straddlingTiles };
          assert.equal(cachedCells, 1); assert.equal(straddlingTiles, 1, "One tile must exercise both cached and overflow lanes");
        }
        results.push({ kind, words, mismatches, maxUlp, maxAbsolute, first, partial });
        console.log(JSON.stringify(results.at(-1))); continue;
      }
      if (waitIdle) {
        for (let polls = 0;; polls++) {
          const active = externalProbes(); if (!active.length) break;
          if (polls % 10 === 0) console.log(JSON.stringify({ phase: "wait-external-gpu-probes", active }));
          await new Promise(resolve => setTimeout(resolve, 1000));
        }
      }
      const before = externalProbes(); if (before.length) timingExternalProcesses.push({ kind, when: "before", active: before });
      e = raw.createCommandEncoder(); for (let warm = 0; warm < 4; warm++) { prepare(e); encode(e, kind, true); prepare(e); encode(e, kind, false); } raw.queue.submit([e.finish()]); await raw.queue.onSubmittedWorkDone();
      const passes: Awaited<ReturnType<GPUPassProfile["finish"]>> = [];
      for (let round = 0; round < rounds; round++) for (const baseline of round % 2 ? [false, true] : [true, false]) {
        profile.start(); e = device.createCommandEncoder(); prepare(e); encode(e, kind, baseline);
        device.queue.submit([e.finish()]); await device.queue.onSubmittedWorkDone(); passes.push(...await profile.finish());
      }
      const after = externalProbes(); if (after.length) timingExternalProcesses.push({ kind, when: "after", active: after });
      const means = Object.fromEntries(["baseline", "candidate"].map(name => { const ms = passes.filter(p => p.label.endsWith(name)).map(p => p.ms); return [name, ms.reduce((a, b) => a + b, 0) / ms.length]; }));
      e = raw.createCommandEncoder(); restore(e, kind); raw.queue.submit([e.finish()]); await raw.queue.onSubmittedWorkDone();
      let breakdown: Awaited<ReturnType<GPUPassProfile["finish"]>> | undefined;
      if (kind.startsWith("metricEdge") || kind.startsWith("metricNewton")) {
        breakdown = [];
        for (let n = 0; n < 2; n++) for (const baseline of n ? [false, true] : [true, false]) {
          profile.start(); e = device.createCommandEncoder(); prepare(e); encode(e, kind, baseline, true);
          device.queue.submit([e.finish()]); await device.queue.onSubmittedWorkDone(); breakdown.push(...await profile.finish());
        }
      }
      let newton: unknown;
      let particleDepthClassifications: unknown;
      if (kind.startsWith("metricNewton")) {
        const counts = allocate(32, true); e = raw.createCommandEncoder(); prepare(e); encode(e, `${kind}Stats`, false);
        e.copyBufferToBuffer(stage.bins, 4 * cells, counts, 0, 32); raw.queue.submit([e.finish()]);
        await counts.mapAsync(GPUMapMode.READ); const counter = new Uint32Array(counts.getMappedRange()).slice(); counts.unmap();
        const capacity = Math.min(Math.floor((cells - 8) / 34), newtonCapacity ?? Infinity), cached = Math.min(capacity, counter[0]!), hist = new Array<number>(9).fill(0);
        let starts = 0, valid = 0;
        if (cached) {
          const read = allocate(4 * 34 * cached, true); e = raw.createCommandEncoder(); e.copyBufferToBuffer(stage.bins, 4 * (cells + 8), read, 0, read.size); raw.queue.submit([e.finish()]);
          await read.mapAsync(GPUMapMode.READ); const data = new Uint32Array(read.getMappedRange());
          const popcount = (word: number) => { let n = 0; for (let b = word >>> 0; b; b = (b & (b - 1)) >>> 0) n++; return n; };
          for (let cell = 0; cell < cached; cell++) { const count = popcount(data[34 * cell + 8]!); hist[count]!++; starts += count; valid += popcount(data[34 * cell + 9]!); }
          read.unmap();
        }
        newton = { capacity, crossings: counter[0], cachedCrossings: cached, startsPerCell: hist, preparedStarts: starts, validStarts: valid,
          originalSolveCalls: counter[1], cachedSolveCalls: counter[2], fallbackSolveCalls: counter[3], invalidToValid: counter[4], validToInvalid: counter[5], avoidedRepeatedSolves: counter[2]! - starts };
        console.log(JSON.stringify({ phase: "newton-before-classification", kind, mismatches, maxAbsolute, maxUlp, maxRelativeAwayZero, signChanges, means_ms: means, breakdown, newton }));
        particleDepthClassifications = await metricClassificationProbe(raw, dims, metricReach!, a[0]!, b[0]!);
      }
      if (kind === "metric" && (integerGradient || cachedGradient)) particleDepthClassifications = await metricClassificationProbe(raw, dims, metricReach!, a[0]!, b[0]!);
      results.push({ kind, words, mismatches, integerMismatches, maxUlp, maxAbsolute, maxRelativeAwayZero, relativeMinimumMagnitude: 1e-3, inactiveScratchDifferences, signChanges, nonfiniteDifferences, absoluteThresholds, absoluteHistogram, nodalThresholds, nodalClassificationChanges, largestAbsolute, largestUlp, first, means_ms: means, passes, breakdown, newton, particleDepthClassifications }); console.log(JSON.stringify(results.at(-1)));
      if (globalNearest && kind === "nearest") for (const fixture of globalNearestFixtures(dims)) {
        raw.queue.writeBuffer(stage.bins, crossingOffset - 16, new Uint32Array([0, fixture.search.length, 0, 0]));
        raw.queue.writeBuffer(stage.bins, crossingOffset, Uint32Array.from(fixture.masks));
        raw.queue.writeBuffer(stage.bins, crossingOffset + 8 * tiles, Uint32Array.from(fixture.reach));
        if (fixture.search.length) raw.queue.writeBuffer(stage.bins, crossingOffset + 16 * tiles, Uint32Array.from(fixture.search));
        const [a] = await capture(true), [b] = await capture(false); assert.ok(a && b);
        let mismatches = 0; const first: unknown[] = [];
        for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) { mismatches++; if (first.length < 8) first.push({ word: i, baseline: a[i], candidate: b[i] }); }
        results.push({ kind: `nearest-${fixture.name}`, words: a.length, searchTiles: fixture.search.length, mismatches, first }); console.log(JSON.stringify(results.at(-1)));
      }
      if (globalNearest && kind === "nearest") {
        e = raw.createCommandEncoder();
        e.copyBufferToBuffer(bandOriginal!, 0, stage.bins, crossingOffset - 16, bandOriginal!.size);
        restore(e, "nearest"); raw.queue.submit([e.finish()]); await raw.queue.onSubmittedWorkDone();
      }
    }
    let geometry: { crossings: number; capacity: number; cachedBytes: number } | undefined;
    if (globalNearest) { const strides = await checkGlobalNearestStrides(raw, narrowBandMembershipSource(false)); results.push(...strides); strides.forEach(r => console.log(JSON.stringify(r))); }
    if (geometryCache) {
      const read = allocate(4, true); const e = raw.createCommandEncoder(); e.copyBufferToBuffer(stage.bins, 4 * cells, read, 0, 4); raw.queue.submit([e.finish()]);
      await read.mapAsync(GPUMapMode.READ); const crossings = new Uint32Array(read.getMappedRange())[0]!; read.unmap();
      const capacity = Math.floor((cells - 1) / 8); geometry = { crossings, capacity, cachedBytes: 32 * Math.min(crossings, capacity) };
    }
    const result = { date: new Date().toISOString(), sceneId: definition.id, values, dt, steps, rounds, prune, lattice, samplerLattice, stagedNearest, globalNearest, hoistGradient, squaredDistance, staticGeometry, geometryCache, vertexCache, edgeCache, newtonCache, newtonGated, plainMetric, newtonReload, newtonFused, newtonPacked, newtonParallel, newtonCapacity, integerSamples, integerGradient, integerCrossings, cachedGradient, cachedGradientSplit, allowedEdgeUlp, allowedAbsolute, waitIdle, timingExternalProcesses, geometry, inPlaceRedistance: Boolean(stage.groups?.redistanceStore), crossingReuse, pruning: variants.pruning, sourceHash: createHash("sha256").update(narrowBandRedistanceWGSL + narrowBandMembershipWGSL).digest("hex"), results, errors };
    mkdirSync(dirname(out), { recursive: true }); writeFileSync(out, JSON.stringify(result, null, 2) + "\n");
    assert.ok(results.every(r => {
      const value = r as { kind: string; mismatches: number; integerMismatches?: number; maxUlp: number; maxAbsolute: number; signChanges?: number; nonfiniteDifferences?: number };
      return value.mismatches === 0 || ((value.kind.startsWith("metricEdge") || value.kind.startsWith("metricNewton") || (value.kind === "metric" && (integerGradient || cachedGradient))) && (value.integerMismatches ?? 0) === 0 && (value.maxUlp <= allowedEdgeUlp || value.maxAbsolute <= allowedAbsolute) && value.maxAbsolute <= 1e-4 && value.signChanges === 0 && value.nonfiniteDifferences === 0);
    }), "Outputs must satisfy the explicit cached-solve tolerance, retain signs, and remain finite (at most 1e-4 cells)"); assert.deepEqual(errors, []);
  } finally { crossingObserver?.destroy(); buffers.forEach(b => b.destroy()); solver?.destroy(); await device?.queue.onSubmittedWorkDone(); profile?.destroy(); device?.destroy(); }
}
if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) await run();
