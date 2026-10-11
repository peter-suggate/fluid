/** Frozen surface A/B with explicitly expanded 4x4x4 cubic loads.
 * Same arithmetic order, source/solid/wall rules and UI parameters.
 * node --import tsx tools/probe-surface-cubic.ts [--steps=96]
 * [--scene=nbflip-figure-9-teaser] [--unroll=x|xyz|loads|stationary|vector] [--rounds=12] [--repeats=8] [--out=FILE]
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

type CubicMode="x"|"xyz"|"loads"|"stationary"|"vector";
function vectorSurfaceSampler(code:string):string{
 const entry=code.match(/fn umSampleVelocity\(p:vec3f\)->vec3f\s*\{/)?.[0];assert.ok(entry,"Expected MAC velocity sampler");
 assert.equal(code.split(entry).length,2);
 const regular="udLoadH_velocity",unit="udLoad_unitVelocity";
 assert.ok(code.includes(`fn ${regular}(`)&&code.includes(`fn ${unit}(`));
 const dims=code.includes("fn UM_D()")?"UM_D()":"UM_D";
 const body=(load:string)=>{
  const loads=Array.from({length:8},(_,k)=>`var v${k}=${load}(base+vec3i(${k&1},${(k>>1)&1},${k>>2})).xyz;`).join("\n");
  const planes=Array.from({length:3},(_,axis)=>{
   const bit=1<<axis,component=["x","y","z"][axis];
   const lows=Array.from({length:8},(_,k)=>k).filter(k=>!(k&bit));
   return `if(normal.${component}>=0.5){\n`+lows.map(k=>`v[${k|bit}].${component}=v[${k}].${component};`).join("\n")+"\n"+
    lows.map(k=>{const q=[k&1,(k>>1)&1,k>>2];q[axis]=-1;return `v[${k}].${component}=${load}(base+vec3i(${q.join(",")})).${component};`;}).join("\n")+"\n}";
  }).join("\n");
  return `let base=vec3i(floor(p-vec3f(0.5)));let t=fract(p-vec3f(0.5));let normal=fract(p);
 ${loads}\n${planes}
 let fx=vec3f(normal.x,t.x,t.x);let fy=vec3f(t.y,normal.y,t.y);let fz=vec3f(t.z,t.z,normal.z);
 let a=(vec3f(1)-fx)*v[0]+fx*v[1];let b=(vec3f(1)-fx)*v[2]+fx*v[3];
 let c=(vec3f(1)-fx)*v[4]+fx*v[5];let d=(vec3f(1)-fx)*v[6]+fx*v[7];
 return (vec3f(1)-fz)*((vec3f(1)-fy)*a+fy*b)+fz*((vec3f(1)-fy)*c+fy*d);`.replace(/v\[(\d+)\]/g,"v$1");
 };
 const helpers=`fn umVectorRegular(p:vec3f)->vec3f{${body(regular)}}\nfn umVectorUnit(p:vec3f)->vec3f{${body(unit)}}\n`;
 const start=code.indexOf(entry),end=code.indexOf("\n}",start)+2;assert.ok(end>start);
 const original=code.slice(start,end);
 const inside=`all(p>=vec3f(1))&&all(p<=vec3f(${dims})-vec3f(1))`;
 const cached="if(!umRegularFine&&umCachedFineSample(p)){";
 const weight="let weights=umVelocitySamplingWeights(p);";
 assert.equal(original.split(cached).length,2);assert.equal(original.split(weight).length,2);
 const replacement=original.replace(cached,cached+`if(${inside}){return umVectorUnit(p);}`)
  .replace(weight,weight+`if(umRegularFine&&weights>0.0&&${inside}){return umVectorRegular(p);}`);
 return code.slice(0,start)+helpers+replacement+code.slice(end);
}
export function unrollSurfaceCubic(code:string,mode:CubicMode="xyz"):string{
 if(mode==="vector")return vectorSurfaceSampler(code);
 if(mode==="stationary"){
  const original="var k1=umSampleVelocity(p)/params.hDt.xyz;";
  assert.equal(code.split(original).length,2,"Expected one initial surface trace sample");
  const dims=code.includes("fn UM_D()")?"UM_D()":"UM_D";
  return code.replace(original,original+`\n if(all(k1==vec3f(0))&&all(p>=vec3f(0))&&all(p<=vec3f(${dims}))){return p;}`);
 }
 const start=code.indexOf("fn umCubicPhi(q:vec3f,cell:UMVertexCell)->f32{");
 assert.ok(start>=0,"Expected cubic sampler");
 const end=code.indexOf("\nfn umDrain(",start);assert.ok(end>start);
 const original=code.slice(start,end);
 const body=original.indexOf(" var value=0.0;var low=1e30;var high=-1e30;");assert.ok(body>0);
 if(mode==="loads"){
  // Re-reading the eight inner texels avoids an indirect register-array
  // lookup and a branch at every tap. Keep the original accumulation order.
  const replacement=original
   .replace(" var value=0.0;var low=1e30;var high=-1e30;", " var value=0.0;var low=1e30;var high=-1e30;\n for(var k=0u;k<8u;k++){low=min(low,cell.corners[k]);high=max(high,cell.corners[k]);}")
   .replaceAll("let inner=all(offset>=vec3i(0))&&all(offset<=vec3i(1));", "")
   .replace(/var s=0\.0;if\(inner\)\{s=cell\.corners\[x-1u\+2u\*\(y-1u\)\+4u\*\(z-1u\)\];\}else\{s=(umLoad(?:Corner|Vertex)\(vec3u\(vertex\)\));\}/g,"let s=$1;")
   .replaceAll("if(inner){low=min(low,s);high=max(high,s);}", "");
  assert.ok(!replacement.includes("if(inner)"),"Every inner-corner branch must be removed");
  return code.slice(0,start)+replacement+code.slice(end);
 }
 const solid=original.includes("if(!tapsClear&&");
 const fullTaps=(load:string)=>Array.from({length:4},(_,z)=>`{var plane=0.0;\n${Array.from({length:4},(_,y)=>`{var row=0.0;\n${Array.from({length:4},(_,x)=>{
  const offset=[x-1,y-1,z-1],inner=offset.every(k=>k>=0&&k<=1);
  return `{let vertex=clamp(base+vec3i(${offset.join(",")})*width,vec3i(0),vec3i(UM_D));
 let s=${inner?`cell.corners[${x-1+2*(y-1)+4*(z-1)}u]`:`${load}(vec3u(vertex))`};row+=wx[${x}u]*s;
 ${solid?`if(!tapsClear&&wx[${x}u]*wy[${y}u]*wz[${z}u]!=0.0&&umBuried(vec3f(vertex))){return cell.value;}`:""}
 ${inner?"low=min(low,s);high=max(high,s);":""}}`;
 }).join("\n")}plane+=wy[${y}u]*row;}`).join("\n")}value+=wz[${z}u]*plane;}`).join("\n");
 const rowTaps=(load:string)=>`for(var z=0u;z<4u;z++){var plane=0.0;
 for(var y=0u;y<4u;y++){var row=0.0;
 ${Array.from({length:4},(_,x)=>`{
 let vertex=clamp(base+vec3i(${x-1},i32(y)-1,i32(z)-1)*width,vec3i(0),vec3i(UM_D));
 ${x===1||x===2?`let inner=y>0u&&y<3u&&z>0u&&z<3u;
 var s=0.0;if(inner){s=cell.corners[${x-1}u+2u*(y-1u)+4u*(z-1u)];}else{s=${load}(vec3u(vertex));}`:`let s=${load}(vec3u(vertex));`}
 row+=wx[${x}u]*s;
 ${solid?`if(!tapsClear&&wx[${x}u]*wy[y]*wz[z]!=0.0&&umBuried(vec3f(vertex))){return cell.value;}`:""}
 ${x===1||x===2?"if(inner){low=min(low,s);high=max(high,s);}":""}
 }`).join("\n")}
 plane+=wy[y]*row;}value+=wz[z]*plane;}`;
 const taps=mode==="x"?rowTaps:fullTaps;
 let replacement=original.slice(0,body)+` var value=0.0;var low=1e30;var high=-1e30;
 if(!umRegularFine&&cell.width==4u){${taps("umLoadCorner")}}else{${taps("umLoadVertex")}}
 return clamp(value,low,high);
 }`;
 if(code.includes("fn UM_D()"))replacement=replacement.replace(/\bUM_D\b(?!\()/g,"UM_D()");
 return code.slice(0,start)+replacement+code.slice(end);
}

class CubicVariants {
  readonly device: GPUDevice;
  readonly compilations:{entry:string;ms:number}[]=[];
  private readonly sources = new WeakMap<GPUShaderModule, string>();
  private readonly specialized = new WeakMap<GPUComputePipeline, GPUComputePipeline>();
  constructor(raw: GPUDevice,mode:CubicMode) {
    this.device = new Proxy(raw, { get: (target, key) => {
      if (key === "createShaderModule") return (d: GPUShaderModuleDescriptor) => {
        const module = target.createShaderModule(d); this.sources.set(module, d.code); return module;
      };
      if (key === "createComputePipelineAsync") return async (d: GPUComputePipelineDescriptor) => {
        const pipeline = await target.createComputePipelineAsync(d);
        if (["advect", "advectFine", "advectOwners", "advectWalls", "advectDeferred"].includes(d.compute.entryPoint ?? "")) {
          const start=performance.now();
          const source = this.sources.get(d.compute.module); assert.ok(source);
          const module = target.createShaderModule({ label: "Diagnostic unrolled cubic surface", code: unrollSurfaceCubic(source,mode) });
          const errors = (await module.getCompilationInfo()).messages.filter(m => m.type === "error");
          if (errors.length) {
            writeFileSync("/tmp/fluid-unrolled-cubic.wgsl", unrollSurfaceCubic(source,mode));
            throw new Error(errors.map(m => `${m.lineNum}:${m.linePos} ${m.message}`).join("\n"));
          }
          this.specialized.set(pipeline, await target.createComputePipelineAsync({ ...d, compute: { ...d.compute, module } }));
          const compilation={entry:d.compute.entryPoint!,ms:performance.now()-start};this.compilations.push(compilation);
          console.log(JSON.stringify({phase:"compiled",unroll:mode,...compilation}));
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
  const unroll=option("unroll")??"xyz";assert.ok(unroll==="x"||unroll==="xyz"||unroll==="loads"||unroll==="stationary"||unroll==="vector");
  assert.ok(Number.isInteger(steps) && steps > 0 && Number.isInteger(rounds) && rounds >= 2 && Number.isInteger(repeats) && repeats > 0);
  const definition = getSceneDefinition(option("scene") ?? "nbflip-figure-9-teaser");
  assert.equal(definition.methodProfile?.methodId, method.id);
  const quality = definition.methodProfile!.quality, scene = structuredClone(sceneDocument(definition));
  const values = resolveMethodValues(method, quality, { ...method.appDefaults, ...definition.methodProfile!.overrides });
  const dt = method.effectiveStep_s?.(scene, values) ?? scene.numerics.fixedDt_s;
  const out = option("out") ?? "artifacts/nbflip-ui-perf/unrolled-cubic-surface.json";
  let device: GPUDevice | undefined, solver: WebGPUUniformReferenceSolver | undefined, profile: GPUPassProfile | undefined;
  const errors: string[] = [], buffers: GPUBuffer[] = [];
  try {
    const dawn = await import(pathToFileURL(resolve("node_modules/webgpu/index.js")).href); Object.assign(globalThis, dawn.globals);
    const gpu = createProcessRetainedDawnGPU(dawn, ["backend=metal"]), adapter = await gpu.requestAdapter(); assert.ok(adapter);
    const raw = await adapter.requestDevice({ requiredFeatures: ["timestamp-query", ...(adapter.features.has("subgroups") ? ["subgroups" as const] : [])], requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
    profile = new GPUPassProfile(raw); const variants = new CubicVariants(profile.device,unroll);
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
    candidate[candidate.length-1]=candidate[candidate.length-1]!.slice(-travelWords); let words = 0, mismatches = 0,maxULP=0,maxAbsolute=0,nonFiniteDifferences=0,signChanges=0; const first: unknown[] = [];
    const ordered=(bits:number)=>bits&0x80000000?0x80000000-(bits&0x7fffffff):0x80000000+bits;
    baseline.forEach((a, bank) => {
      const b = candidate[bank]!,af=new Float32Array(a.buffer,a.byteOffset,a.length),bf=new Float32Array(b.buffer,b.byteOffset,b.length); words += a.length;
      for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) {
        mismatches++;const absolute=Math.abs(af[i]!-bf[i]!);const ulp=Math.abs(ordered(a[i]!)-ordered(b[i]!));
        if(Number.isFinite(absolute)){maxAbsolute=Math.max(maxAbsolute,absolute);maxULP=Math.max(maxULP,ulp);}else nonFiniteDifferences++;
        if((af[i]!<0)!==(bf[i]!<0))signChanges++;
        if (first.length < 8) first.push({ bank, word: i, baseline: a[i], candidate: b[i],baselineValue:af[i],candidateValue:bf[i],absolute,ulp });
      }
    });
    const timingReady=option("timing-ready");
    if(timingReady){
      console.log(JSON.stringify({phase:"await-timing",timingReady,words,mismatches}));
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
    const result = { date: new Date().toISOString(), sceneId: definition.id, values, dt, steps, rounds, repeats, unroll, construction_ms, compilations:variants.compilations, words, mismatches,maxULP,maxAbsolute,nonFiniteDifferences,signChanges, first, means_ms: means, medians_ms:{baseline:median(baseSamples),candidate:median(candidateSamples),pairedSaved:median(saved)}, samples_ms:{baseline:baseSamples,candidate:candidateSamples,saved}, passes, errors };
    mkdirSync(dirname(out), { recursive: true }); writeFileSync(out, JSON.stringify(result, null, 2) + "\n"); console.log(JSON.stringify(result));
    const allowedULP=Number(option("max-ulp")??0),allowedAbsolute=Number(option("max-absolute")??0);
    assert.equal(nonFiniteDifferences,0);assert.equal(signChanges,0);
    assert.ok(mismatches===0||maxULP<=allowedULP||maxAbsolute<=allowedAbsolute,"Surface differences exceed the explicit diagnostic tolerance"); assert.deepEqual(errors, []);
  } finally { buffers.forEach(b => b.destroy()); solver?.destroy(); await device?.queue.onSubmittedWorkDone(); profile?.destroy(); device?.destroy(); }
}
if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) await run();
