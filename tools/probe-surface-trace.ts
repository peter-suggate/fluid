/** Frozen surface A/B with integral initial sampling or early deferral.
 * Same arithmetic order, source/solid/wall rules and UI parameters.
 * node --import tsx tools/probe-surface-trace.ts [--steps=96]
 * [--scene=nbflip-figure-9-teaser] [--mode=integral|integral-direct|early-defer] [--rounds=12] [--repeats=8] [--out=FILE]
 */
import assert from "node:assert/strict";
import { writeFileSync, mkdirSync, existsSync } from "node:fs";
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
import type { WebGPUUniformReferenceSolver } from "../lib/methods/uniform/webgpu-uniform-reference";
import { GPUPassProfile } from "./gpu-pass-profile";

type TraceMode="integral"|"integral-direct"|"early-defer";
/** Integral interior vertices share seven vector texels across their three
 * staggered bilinear components. General/fractional/boundary samples retain
 * the original sampler, including every coarse interpolation. */
export function specializeSurfaceTrace(code:string,mode:TraceMode="integral"):string{
 if(mode==="early-defer"){
  const start=code.indexOf("fn umAdvectStore(vertex:vec3u,width:u32){");
  const end=code.indexOf("// The deferred vertices of the dispatch",start);
  assert.ok(start>=0&&end>start);
  const original=code.slice(start,end);
  const before="defer=walk.hit;let q=walk.q;";
  const after="if(!defer){umRecordTravel(p,q,width);}";
  assert.equal(original.split(before).length,2);assert.equal(original.split(after).length,2);
  // The skipped calls only read geometry, phi, sources and released-wall
  // reach. umRecordTravel was already skipped whenever walk.hit is true.
  const replacement=original.replace(before,before+"if(!defer){").replace(after,after+"}");
  return code.slice(0,start)+replacement+code.slice(end);
 }
 const initial="var k1=umSampleVelocity(p)/params.hDt.xyz;";
 assert.equal(code.split(initial).length,2);
 const regular=code.includes("fn udLoadH_velocity(")?"udLoadH_velocity":"";
 const unit=code.includes("fn udLoad_unitVelocity(")?"udLoad_unitVelocity":"";
 assert.ok(regular&&unit&&code.includes("fn umCachedFineSample("),"Expected expanded fine and hanging velocity loaders");
 const taps=(load:string)=>`
 let a=${load}(base).xyz;let b=${load}(base+vec3i(1,0,0)).xyz;
 let c=${load}(base+vec3i(0,1,0)).xyz;let d=${load}(base+vec3i(0,0,1)).xyz;
 let e=${load}(base+vec3i(1,1,0)).xyz;let f=${load}(base+vec3i(1,0,1)).xyz;
 let g=${load}(base+vec3i(0,1,1)).xyz;
 return vec3f(0.5*(0.5*a.x+0.5*c.x)+0.5*(0.5*d.x+0.5*g.x),
  0.5*(0.5*a.y+0.5*b.y)+0.5*(0.5*d.y+0.5*f.y),
  0.5*(0.5*a.z+0.5*b.z)+0.5*(0.5*c.z+0.5*e.z));`;
 let helper=`
fn umIntegralSurfaceVelocity(p:vec3f)->vec3f{
 if(${mode==="integral-direct"?"":"any(p!=floor(p))||"}any(p<=vec3f(0))||any(p>=vec3f(UM_D))){return umSampleVelocity(p);}
 if(umRegularFine){
  if(umVelocitySamplingWeights(p)==0.0){return umSampleVelocity(p);}
  let base=vec3i(p)-vec3i(1);${taps(regular)}
 }
 if(umCachedFineSample(p)){let base=vec3i(p)-vec3i(1);${taps(unit)}}
 return umSampleVelocity(p);
}
`;
 if(code.includes("fn UM_D()"))helper=helper.replace(/\bUM_D\b(?!\()/g,"UM_D()");
 if(mode==="integral-direct"){
  // Only these two callers start at an authored lattice vertex. Fractional
  // contact/centroid traces keep their original sampler and avoid the guard.
  const start=code.indexOf("fn umSurfaceTrace(p:vec3f)->vec3f{");assert.ok(start>=0);
  let end=code.indexOf("{",start)+1,depth=1;
  for(;end<code.length&&depth;end++){if(code[end]==="{")depth++;else if(code[end]==="}")depth--;}
  assert.equal(depth,0);
  const original=code.slice(start,end);
  const integer=original.replace("fn umSurfaceTrace(","fn umIntegerSurfaceTrace(").replace(initial,"var k1=umIntegralSurfaceVelocity(p)/params.hDt.xyz;");
  code=code.slice(0,start)+helper+integer+"\n"+original+code.slice(end);
  code+=`\nfn umIntegerTrace(p:vec3f)->vec3f{return ${code.includes("fn umWalk(")?"umWalk(p,umIntegerSurfaceTrace(p)).q":"umIntegerSurfaceTrace(p)"};}\n`;
  for(const name of ["umAdvected","umAdvectStore"]){
   const begin=code.indexOf(`fn ${name}(`);assert.ok(begin>=0);
   let next=code.indexOf("{",begin)+1,depth=1;
   for(;next<code.length&&depth;next++){if(code[next]==="{")depth++;else if(code[next]==="}")depth--;}
   assert.equal(depth,0);
   const body=code.slice(begin,next).replaceAll("umSurfaceTrace(p)","umIntegerSurfaceTrace(p)").replaceAll("umTrace(p)","umIntegerTrace(p)");
   code=code.slice(0,begin)+body+code.slice(next);
  }
  return code;
 }
 return code.replace("fn umSurfaceTrace(p:vec3f)->vec3f{",helper+"fn umSurfaceTrace(p:vec3f)->vec3f{").replace(initial,"var k1=umIntegralSurfaceVelocity(p)/params.hDt.xyz;");
}

class TraceVariants {
  readonly device: GPUDevice;
  readonly compilations:{entry:string;ms:number}[]=[];
  private readonly sources = new WeakMap<GPUShaderModule, string>();
  private readonly specialized = new WeakMap<GPUComputePipeline, GPUComputePipeline>();
  constructor(raw: GPUDevice,mode:TraceMode) {
    this.device = new Proxy(raw, { get: (target, key) => {
      if (key === "createShaderModule") return (d: GPUShaderModuleDescriptor) => {
        const module = target.createShaderModule(d); this.sources.set(module, d.code); return module;
      };
      if (key === "createComputePipelineAsync") return async (d: GPUComputePipelineDescriptor) => {
        const pipeline = await target.createComputePipelineAsync(d);
        if (["advect", "advectFine", "advectOwners", "advectWalls", "advectDeferred"].includes(d.compute.entryPoint ?? "")) {
          const start=performance.now();
          const source = this.sources.get(d.compute.module); assert.ok(source);
          const module = target.createShaderModule({ label: "Diagnostic surface trace", code: specializeSurfaceTrace(source,mode) });
          const errors = (await module.getCompilationInfo()).messages.filter(m => m.type === "error");
          if (errors.length) {
            writeFileSync("/tmp/fluid-surface-trace.wgsl", specializeSurfaceTrace(source,mode));
            throw new Error(errors.map(m => `${m.lineNum}:${m.linePos} ${m.message}`).join("\n"));
          }
          this.specialized.set(pipeline, await target.createComputePipelineAsync({ ...d, compute: { ...d.compute, module } }));
          const compilation={entry:d.compute.entryPoint!,ms:performance.now()-start};this.compilations.push(compilation);
          console.log(JSON.stringify({phase:"compiled",mode,...compilation}));
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
  const steps = Number(option("steps") ?? 96), rounds = Number(option("rounds") ?? 12), repeats = Number(option("repeats") ?? 8);
  const mode=option("mode")??"integral";assert.ok(mode==="integral"||mode==="integral-direct"||mode==="early-defer");
  assert.ok(Number.isInteger(steps) && steps > 0 && Number.isInteger(rounds) && rounds >= 2 && Number.isInteger(repeats) && repeats > 0);
  const definition = getSceneDefinition(option("scene") ?? "nbflip-figure-9-teaser");
  assert.equal(definition.methodProfile?.methodId, method.id);
  const quality = definition.methodProfile!.quality, scene = structuredClone(sceneDocument(definition));
  const values = resolveMethodValues(method, quality, { ...method.appDefaults, ...definition.methodProfile!.overrides });
  const dt = method.effectiveStep_s?.(scene, values) ?? scene.numerics.fixedDt_s;
  const out = option("out") ?? "artifacts/nbflip-ui-perf/integral-surface-trace.json";
  let device: GPUDevice | undefined, solver: WebGPUUniformReferenceSolver | undefined, profile: GPUPassProfile | undefined;
  const errors: string[] = [], buffers: GPUBuffer[] = [];
  try {
    const dawn = await import(pathToFileURL(resolve("node_modules/webgpu/index.js")).href); Object.assign(globalThis, dawn.globals);
    const gpu = createProcessRetainedDawnGPU(dawn, ["backend=metal"]), adapter = await gpu.requestAdapter(); assert.ok(adapter);
    const raw = await adapter.requestDevice({ requiredFeatures: ["timestamp-query", ...(adapter.features.has("subgroups") ? ["subgroups" as const] : [])], requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
    profile = new GPUPassProfile(raw); const variants = new TraceVariants(profile.device,mode);
    device = managedGPUDevice(variants.device, { requireWorkerRealm: false });
    device.addEventListener("uncapturederror", e => { e.preventDefault(); errors.push(e.error.message); });
    const constructionStart=performance.now();
    solver = await method.createSolverAsync!(device, scene, quality, values, undefined, () => {}) as WebGPUUniformReferenceSolver;
    const construction_ms=performance.now()-constructionStart;
    console.log(JSON.stringify({phase:"constructed",construction_ms}));
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
    candidate[candidate.length-1]=candidate[candidate.length-1]!.slice(-travelWords);
    let words=0,mismatches=0,maxAbsolute=0,maxULP=0,signedZeros=0,signChanges=0,nonfiniteDifferences=0;const first:unknown[]=[];
    const order=(u:number)=>(u&0x80000000)?0x80000000-(u&0x7fffffff):0x80000000+u;
    baseline.forEach((a,bank)=>{
      const b=candidate[bank]!,af=new Float32Array(a.buffer,a.byteOffset,a.length),bf=new Float32Array(b.buffer,b.byteOffset,b.length);words+=a.length;
      for(let i=0;i<a.length;i++)if(a[i]!==b[i]){
        mismatches++;const absolute=Math.abs(af[i]!-bf[i]!),ulp=Math.abs(order(a[i]!)-order(b[i]!));
        if((af[i]!<0)!==(bf[i]!<0))signChanges++;
        if(af[i]===0&&bf[i]===0)signedZeros++;
        else if(Number.isFinite(af[i])&&Number.isFinite(bf[i])){maxAbsolute=Math.max(maxAbsolute,absolute);maxULP=Math.max(maxULP,ulp);}
        else nonfiniteDifferences++;
        if(first.length<8)first.push({bank,word:i,baseline:a[i],candidate:b[i],baselineValue:af[i],candidateValue:bf[i],absolute,ulp});
      }
    });
    const timingReady=option("timing-ready");
    if(timingReady){
      console.log(JSON.stringify({phase:"await-timing",timingReady,words,mismatches,maxAbsolute,maxULP,signChanges}));
      while(!existsSync(timingReady))await new Promise(resolve=>setTimeout(resolve,250));
    }
    // Readback disturbs caches; warm both compiled variants before timing.
    e = raw.createCommandEncoder(); for (let n = 0; n < 4; n++) { encode(e, true); encode(e, false); }
    raw.queue.submit([e.finish()]); await raw.queue.onSubmittedWorkDone();
    const passes: Awaited<ReturnType<GPUPassProfile["finish"]>> = [];
    const baseSamples:number[]=[],candidateSamples:number[]=[];
    for (let round = 0; round < rounds; round++) for (const baseline of round % 2 ? [false, true] : [true, false]) {
      e=raw.createCommandEncoder(); restore(e);raw.queue.submit([e.finish()]);await raw.queue.onSubmittedWorkDone();
      profile.start(); e = device.createCommandEncoder(); for(let n=0;n<repeats;n++)encode(e, baseline);
      device.queue.submit([e.finish()]); await device.queue.onSubmittedWorkDone();
      const sample=await profile.finish();passes.push(...sample);
      (baseline?baseSamples:candidateSamples).push(sample.reduce((sum,p)=>sum+p.ms,0)/repeats);
    }
    const mean=(a:number[])=>a.reduce((n,v)=>n+v,0)/a.length;
    const means={baseline:mean(baseSamples),candidate:mean(candidateSamples)};
    const sorted=(a:number[])=>[...a].sort((x,y)=>x-y);
    const median=(a:number[])=>{const v=sorted(a),m=Math.floor(v.length/2);return v.length%2?v[m]!:(v[m-1]!+v[m]!)/2;};
    const saved=baseSamples.map((v,i)=>v-candidateSamples[i]!);

    e = raw.createCommandEncoder(); restore(e); raw.queue.submit([e.finish()]); await raw.queue.onSubmittedWorkDone();
    const result = { date: new Date().toISOString(), sceneId: definition.id, values, dt, steps, rounds, repeats, mode, construction_ms, compilations:variants.compilations, words, mismatches, maxAbsolute, maxULP, signedZeros, signChanges, nonfiniteDifferences, first, means_ms: means, medians_ms:{baseline:median(baseSamples),candidate:median(candidateSamples),pairedSaved:median(saved)}, samples_ms:{baseline:baseSamples,candidate:candidateSamples,saved}, passes, errors };
    mkdirSync(dirname(out), { recursive: true }); writeFileSync(out, JSON.stringify(result, null, 2) + "\n"); console.log(JSON.stringify(result));
    const allowedULP=Number(option("max-ulp")??0),allowedAbsolute=Number(option("max-absolute")??0);
    assert.equal(nonfiniteDifferences,0);assert.equal(signChanges,0);
    assert.ok(mismatches===0||maxULP<=allowedULP||maxAbsolute<=allowedAbsolute,"Surface differences exceed the explicit diagnostic tolerance");assert.deepEqual(errors, []);
  } finally { buffers.forEach(b => b.destroy()); solver?.destroy(); await device?.queue.onSubmittedWorkDone(); profile?.destroy(); device?.destroy(); }
}
if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) await run();
