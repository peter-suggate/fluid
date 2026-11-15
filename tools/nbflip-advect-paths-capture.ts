/** Evidence for the M2 level-set advect port (levelset-coarse.wgsl.ts, lane F's lAdvect): whether
 * two old-engine paths of the "uniform-narrow-band-flip" surface advect
 * (lib/methods/uniform/uniform-mixed-surface.ts) ever fire on high-resolution-dam-break.
 *
 * (a) Kept deep air, umDeepAirKept (l.1398), reached through umKeptAir (l.1412) from umAdvectStore
 *     (l.1517, non-wall vertices) and advectDeferred's coarse preflight (l.1580), after umHoldCold
 *     and only where umFarAirKept declines. A vertex keeps its stored value, untraced, iff
 *     - φ (umLatticeLoad at its width) >= (6 + 3 width) min h (9h at h, 18h at 4h), and
 *     - its tile's spread-reach word (claims[evidenceBytes/4 - tiles + tile], encode's copy of
 *       ownership.speeds[tiles + tile]; six 5-bit fields, -x,-y,-z,+x,+y,+z) is <= 2 on every
 *       field (umReach = ceil(travel/4) + 2: no travel either way on any axis), and
 *     - umReleasedMayChange(vertex, φ) is false.
 * (b) Released walls: umReleasedWalls (l.829) raises a value to a released wall face's gap
 *     (dt·away − depth) and umReleasedMayChange (l.1316) defers or rejects a vertex (l.296,
 *     l.1171, l.1207, l.1290, l.1760, and (a) above). Both skip every direction whose
 *     claims[UM_WALL_REACH + k] (wallReach, l.1682: the largest dt·away over the wall faces
 *     of direction k whose owner's projection release bit (wall texel .w) is set and
 *     dt·away > 1e-4 h width) is not positive, so with all six words 0 neither can act.
 *
 * Non-invasive counting: no file of the engine changes. The device is wrapped (as
 * tools/nbflip-census-capture.ts does):
 * - the surface claims buffer ("Uniform mixed surface job claims") is created 64 words longer,
 *   and those words, past every word the engine addresses and never cleared by it, hold counters;
 * - the surface module's WGSL (the one defining umDeepAirKept) gets atomicAdds on those words at
 *   the decision points, each guard evaluated exactly as written (same short-circuit order);
 * - after "Uniform mixed surface advect" the six UM_WALL_REACH words are copied; after
 *   "Uniform mixed surface advect deferred" the counters are copied and cleared.
 * Counters (per frame): deep-air kept by width (h, 4h), evaluations, past the φ test, past the
 * reach test, and kept below 16 max h and below the 0.8 m far-air marker; umReleasedWalls calls
 * that changed a value; umReleasedMayChange true; samples of the kept vertices (any, and below
 * the marker).
 *
 *   node --import tsx tools/nbflip-advect-paths-capture.ts [--steps=300] > out.jsonl
 * Run through tools/gpu/fluid-gpu-run.sh check with WEBGPU_NODE_MODULE set. */
import assert from "node:assert/strict";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import "../lib/methods";
import { getMethod } from "../lib/core/method-registry";
import { resolveMethodValues, type GPUSolverInstance } from "../lib/core/method-contract";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { managedGPUDevice, whenGPUCompilationIdle } from "../lib/core/gpu-compilation-manager";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { createProcessRetainedDawnGPU, paceDawnEventPolling } from "../lib/harness/node-dawn-provider";
import { UNIFORM_MIXED_CLAIM_WORDS } from "../lib/methods/uniform/uniform-mixed-face-dispatch.wgsl";

const METHOD = "uniform-narrow-band-flip", SCENE = "high-resolution-dam-break";
const args = process.argv.slice(2);
const option = (key: string) => args.find(a => a.startsWith(`--${key}=`))?.slice(key.length + 3);
const steps = Number(option("steps") ?? 300);

const method = getMethod(METHOD), definition = getSceneDefinition(SCENE);
const scene = sceneDocument(definition);
const profile = definition.methodProfile?.methodId === METHOD ? definition.methodProfile : undefined;
const quality = profile?.quality ?? "balanced";
const values = resolveMethodValues(method, quality, { ...method.appDefaults, ...profile?.overrides });
const dt = method.effectiveStep_s?.(scene, values) ?? scene.numerics.fixedDt_s;
paceDawnEventPolling();
const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE ?? resolve("node_modules/webgpu/index.js")).href);
Object.assign(globalThis, dawn.globals);
const adapter = await createProcessRetainedDawnGPU(dawn, ["backend=metal"]).requestAdapter();
assert.ok(adapter);
const raw: GPUDevice = await adapter.requestDevice({ requiredFeatures: (["subgroups", "float32-filterable", "timestamp-query"] as const).filter(f => adapter.features.has(f)), requiredLimits: requiredFluidDeviceLimits(adapter.limits) });

// ---- Counters: CW + ...
// KEPT/CALLS/PAST_*/KEPT_SUB/KEPT_UNMARKED: + 0 at h, + 1 at 4h. KEPT_SUB: kept below umFarAirKept's 16 max h;
// KEPT_UNMARKED: kept below 0.75 m (the far-air marker the dam's air holds is 0.8 m).
const C = { KEPT: 0, RELEASED_CHANGED: 2, MAY_CHANGE: 3, CALLS: 4, PAST_PHI: 6, PAST_REACH: 8, SAMPLES: 10, KEPT_SUB: 11, KEPT_UNMARKED: 13, SAMPLES_B: 15, SAMPLE: 16, SAMPLE_B: 40, RELEASED_W: 64, MAY_W: 66, SAMPLES_R: 68, SAMPLE_R: 72, WORDS: 96 } as const;
// RELEASED_W/MAY_W: + 0 or + 1 by the owner width (h, 4h) of the cell below the vertex (p - 1, clamped at 0):
// umReleasedWalls and umReleasedMayChange take no width.
const SAMPLE_MAX = 12;
const CLAIMS_LABEL = "Uniform mixed surface job claims", SURFACE = "Uniform mixed surface advect", DEFERRED = "Uniform mixed surface advect deferred";
let claims: GPUBuffer | undefined, CW = -1, instrumented = 0;
const instrument = (code: string): string => {
  assert.ok(CW >= 0, "the claims buffer precedes the surface module");
  const w = (k: number) => `${CW + k}u`;
  const edits: [string, string][] = [
    ["fn umDeepAirKept(vertex:vec3u,width:u32)->f32{\n", `fn umDeepAirKept(vertex:vec3u,width:u32)->f32{\n let cwW=select(0u,1u,width==4u);atomicAdd(&umClaims[${w(C.CALLS)}+cwW],1u);\n`],
    ["if(!(value>=(6.0+3.0*f32(width))*h)){return -1.0;}\n let tile=umTileAt(", `if(!(value>=(6.0+3.0*f32(width))*h)){return -1.0;}\n atomicAdd(&umClaims[${w(C.PAST_PHI)}+cwW],1u);\n let tile=umTileAt(`],
    [" if(any(reach>vec3u(2u))||umReleasedMayChange(vec3f(vertex),value)){return -1.0;}\n return value;\n}",
      ` if(any(reach>vec3u(2u))){return -1.0;}\n atomicAdd(&umClaims[${w(C.PAST_REACH)}+cwW],1u);\n if(umReleasedMayChange(vec3f(vertex),value)){return -1.0;}\n atomicAdd(&umClaims[${w(C.KEPT)}+cwW],1u);\n if(value<16.0*max(params.hDt.x,max(params.hDt.y,params.hDt.z))){atomicAdd(&umClaims[${w(C.KEPT_SUB)}+cwW],1u);}\n {let s=atomicAdd(&umClaims[${w(C.SAMPLES)}],1u);if(s<${SAMPLE_MAX}u){atomicStore(&umClaims[${w(C.SAMPLE)}+2u*s],vertex.x|(vertex.y<<10u)|(vertex.z<<20u));atomicStore(&umClaims[${w(C.SAMPLE)}+2u*s+1u],width|(bitcast<u32>(value)&0xffffff00u));}}\n if(value<0.75){atomicAdd(&umClaims[${w(C.KEPT_UNMARKED)}+cwW],1u);let s=atomicAdd(&umClaims[${w(C.SAMPLES_B)}],1u);if(s<${SAMPLE_MAX}u){atomicStore(&umClaims[${w(C.SAMPLE_B)}+2u*s],vertex.x|(vertex.y<<10u)|(vertex.z<<20u));atomicStore(&umClaims[${w(C.SAMPLE_B)}+2u*s+1u],width|(bitcast<u32>(value)&0xffffff00u));}}\n return value;\n}`],
    ["  if(probes>0u&&whole>0.0){result=max(result,whole-depth);}\n }return result;\n}", `  if(probes>0u&&whole>0.0){result=max(result,whole-depth);}\n }if(result!=value){atomicAdd(&umClaims[${w(C.RELEASED_CHANGED)}],1u);atomicAdd(&umClaims[${w(C.RELEASED_W)}+select(0u,1u,umOwnerAt(max(at-vec3i(1),vec3i(0))).width==4u)],1u);let s=atomicAdd(&umClaims[${w(C.SAMPLES_R)}],1u);if(s<${SAMPLE_MAX}u){let q=vec3u(at);atomicStore(&umClaims[${w(C.SAMPLE_R)}+2u*s],q.x|(q.y<<10u)|(q.z<<20u)|(umOwnerAt(max(at-vec3i(1),vec3i(0))).width<<30u));atomicStore(&umClaims[${w(C.SAMPLE_R)}+2u*s+1u],bitcast<u32>(result-value));}}return result;\n}`],
    ["  if(tileReach>0.0&&!(tileReach-depth+slack<value)){return true;}", `  if(tileReach>0.0&&!(tileReach-depth+slack<value)){atomicAdd(&umClaims[${w(C.MAY_CHANGE)}],1u);atomicAdd(&umClaims[${w(C.MAY_W)}+select(0u,1u,umOwnerAt(max(at-vec3i(1),vec3i(0))).width==4u)],1u);return true;}`],
  ];
  for (const [from, to] of edits) { assert.equal(code.split(from).length, 2, `instrument: ${JSON.stringify(from.slice(0, 60))} occurs once`); code = code.replace(from, to); }
  instrumented++;
  return code;
};

// ---- The hooks.
type Hook = (e: GPUCommandEncoder) => void;
const after = new Map<string, Hook>();
const intercepted = new Proxy(raw, { get(target, key) {
  if (key === "createBuffer") return (d: GPUBufferDescriptor) => {
    if (d.label !== CLAIMS_LABEL) return target.createBuffer(d);
    assert.ok(!claims, "one surface claims buffer");
    CW = Math.ceil(d.size / 4); claims = target.createBuffer({ ...d, size: 4 * (CW + C.WORDS), usage: d.usage | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    return claims;
  };
  if (key === "createShaderModule") return (d: GPUShaderModuleDescriptor) => target.createShaderModule(d.code.includes("fn umDeepAirKept(") ? { ...d, code: instrument(d.code) } : d);
  if (key === "createCommandEncoder") return (d?: GPUCommandEncoderDescriptor) => {
    const encoder = target.createCommandEncoder(d);
    return new Proxy(encoder, { get(e, key) {
      if (key === "beginComputePass") return (descriptor?: GPUComputePassDescriptor) => {
        const pass = e.beginComputePass(descriptor), hook = descriptor?.label ? after.get(descriptor.label) : undefined;
        if (!hook) return pass;
        return new Proxy(pass, { get(p, key) {
          if (key === "end") return () => { p.end(); hook(e); };
          const v = Reflect.get(p, key, p); return typeof v === "function" ? v.bind(p) : v;
        } });
      };
      const v = Reflect.get(e, key, e); return typeof v === "function" ? v.bind(e) : v;
    } });
  };
  const v = Reflect.get(target, key, target); return typeof v === "function" ? v.bind(target) : v;
} });
const device = managedGPUDevice(intercepted, { requireWorkerRealm: false });
const errors: string[] = [];
device.addEventListener("uncapturederror", e => { e.preventDefault(); errors.push(e.error.message); });
const solver: GPUSolverInstance = await method.createSolverAsync!(device, scene, quality, values, undefined, () => {});
await solver.waitForSimulationReady?.();
await whenGPUCompilationIdle();
assert.ok(claims && instrumented >= 1, `the surface claims buffer (${!!claims}) and an instrumented surface module (${instrumented})`);
raw.queue.writeBuffer(claims, 4 * CW, new Uint32Array(C.WORDS));
console.log(JSON.stringify({ kind: "scene", scene: SCENE, quality, dt, steps, counterWord: CW, instrumentedModules: instrumented }));

// ---- Per frame.
const copy = (e: GPUCommandEncoder, offset: number, words: number) => { const r = raw.createBuffer({ size: 4 * words, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }); e.copyBufferToBuffer(claims!, 4 * offset, r, 0, 4 * words); return r; };
const mapped = async (b: GPUBuffer) => { await b.mapAsync(GPUMapMode.READ); const v = new Uint32Array(b.getMappedRange().slice(0)); b.unmap(); b.destroy(); return v; };
let reachCopies: GPUBuffer[] = [], counterCopies: GPUBuffer[] = [];
after.set(SURFACE, e => { reachCopies.push(copy(e, UNIFORM_MIXED_CLAIM_WORDS, 6)); });
after.set(DEFERRED, e => { counterCopies.push(copy(e, CW, C.WORDS)); e.clearBuffer(claims!, 4 * CW, 4 * C.WORDS); });
const F = new Float32Array(1), U = new Uint32Array(F.buffer);
const asF = (u: number) => { U[0] = u; return F[0]!; };
const samples = (counters: Uint32Array[], n: number, at: number) => counters.flatMap(c => Array.from({ length: Math.min(c[n]!, SAMPLE_MAX) }, (_, s) => {
  const v = c[at + 2 * s]!, wd = c[at + 2 * s + 1]!, q = [v & 1023, (v >>> 10) & 1023, v >>> 20];
  return { vertex: q, tile: q.map(x => x >> 2), width: wd & 255, phi: +asF(wd & 0xffffff00).toFixed(5) };
}));
const total = { releasedH: 0, released4h: 0, mayH: 0, may4h: 0, keptH: 0, kept4h: 0, keptBelowMarkerH: 0, keptBelowMarker4h: 0, framesKeptBelowMarker: 0, releasedChanged: 0, mayChange: 0, framesKept: 0, framesReleased: 0, framesWallReach: 0, maxKept: { frame: 0, h: 0, w4: 0 }, maxCalls: { frame: 0, h: 0, w4: 0 } };
for (let step = 1; step <= steps; step++) {
  assert.ok(solver.advanceTo(step * dt, []), `step ${step} refused`);
  await (solver.awaitFrameCompletion ? solver.awaitFrameCompletion() : device.queue.onSubmittedWorkDone());
  if (solver.info.simulationPipelineError) throw new Error(String(solver.info.simulationPipelineError));
  if (errors.length) throw new Error(`GPU error at step ${step}: ${errors[0]}`);
  const reaches = await Promise.all(reachCopies.map(mapped)), counters = await Promise.all(counterCopies.map(mapped));
  reachCopies = []; counterCopies = [];
  const sum = (k: number) => counters.reduce((s, c) => s + c[k]!, 0);
  const row = {
    kind: "frame", step, advects: counters.length,
    wallReach: reaches.map(r => [...r].map(asF)),
    keptDeepAir: { h: sum(C.KEPT), w4: sum(C.KEPT + 1) },
    deepAirCalls: { h: sum(C.CALLS), w4: sum(C.CALLS + 1) }, pastPhi: { h: sum(C.PAST_PHI), w4: sum(C.PAST_PHI + 1) }, pastReach: { h: sum(C.PAST_REACH), w4: sum(C.PAST_REACH + 1) },
    releasedWallsChanged: sum(C.RELEASED_CHANGED), releasedMayChange: sum(C.MAY_CHANGE),
    releasedByWidth: { h: sum(C.RELEASED_W), w4: sum(C.RELEASED_W + 1) }, mayChangeByWidth: { h: sum(C.MAY_W), w4: sum(C.MAY_W + 1) },
    sampleReleased: counters.flatMap(c => Array.from({ length: Math.min(c[C.SAMPLES_R]!, SAMPLE_MAX) }, (_, s) => { const v = c[C.SAMPLE_R + 2 * s]!, q = [v & 1023, (v >>> 10) & 1023, (v >>> 20) & 1023]; return { vertex: q, tile: q.map(x => x >> 2), ownerWidth: v >>> 30 === 0 ? 4 : v >>> 30, raisedBy: +asF(c[C.SAMPLE_R + 2 * s + 1]!).toExponential(3) }; })).slice(0, 4),
    keptBelow16h: { h: sum(C.KEPT_SUB), w4: sum(C.KEPT_SUB + 1) }, keptBelowMarker: { h: sum(C.KEPT_UNMARKED), w4: sum(C.KEPT_UNMARKED + 1) },
    sample: samples(counters, C.SAMPLES, C.SAMPLE).slice(0, 4), sampleBelowMarker: samples(counters, C.SAMPLES_B, C.SAMPLE_B).slice(0, 8),
  };
  total.keptBelowMarkerH += row.keptBelowMarker.h; total.keptBelowMarker4h += row.keptBelowMarker.w4; if (row.keptBelowMarker.h + row.keptBelowMarker.w4) total.framesKeptBelowMarker++;
  total.releasedH += row.releasedByWidth.h; total.released4h += row.releasedByWidth.w4; total.mayH += row.mayChangeByWidth.h; total.may4h += row.mayChangeByWidth.w4;
  total.keptH += row.keptDeepAir.h; total.kept4h += row.keptDeepAir.w4; total.releasedChanged += row.releasedWallsChanged; total.mayChange += row.releasedMayChange;
  if (row.keptDeepAir.h + row.keptDeepAir.w4) total.framesKept++;
  if (row.releasedWallsChanged + row.releasedMayChange) total.framesReleased++;
  if (row.wallReach.some(r => r.some(v => v > 0))) total.framesWallReach++;
  if (row.keptDeepAir.h + row.keptDeepAir.w4 > total.maxKept.h + total.maxKept.w4) total.maxKept = { frame: step, ...row.keptDeepAir };
  if (row.deepAirCalls.h + row.deepAirCalls.w4 > total.maxCalls.h + total.maxCalls.w4) total.maxCalls = { frame: step, ...row.deepAirCalls };
  console.log(JSON.stringify(row));
}
console.log(JSON.stringify({ kind: "total", steps, ...total }));
solver.destroy();
process.exit(0);
