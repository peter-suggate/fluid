/** Diagnostic closed-tank fixed-residency counterfactual. Both arms pin every
 * authored page; candidate replaces only page/block table lookups with the
 * verified initial linear mapping. Not wired into production.
 * Compare this against dynamic residency separately before any adoption. */
/** Compare historical NB-FLIP implementations, including changed defaults and policies.
 * This is a historical comparison, NOT a physics-equivalence test. Records both resolved settings.
 * --transfer=flip matches transfer choice; --profile=true adds diagnostic pass timestamps.
 * --baseline-values=JSON and --candidate-values=JSON override a single arm.
 * Timing is simulation-only; run under the shared GPU lock when possible.
 * Alternate complete simulation frames from two source snapshots.
 * Both solvers share one device; every frame is fenced before the other runs.
 * Construction, renderer and final diagnostics are outside the timed window.
 * node --import tsx tools/benchmark-nbflip-history.ts NAME
 *   --baseline=/absolute/snapshot --candidate=/absolute/snapshot
 *   [--steps=240] [--from=13] [--scene=sparse-cm12-long-dam-break] [--out=DIR]
 */
import { GPUPassProfile } from "./gpu-pass-profile";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import { cpus, release, totalmem } from "node:os";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { createProcessRetainedDawnGPU, paceDawnEventPolling } from "../lib/harness/node-dawn-provider";
import type { uniformNarrowBandMethod } from "../lib/methods/uniform/uniform-narrow-band-method";
import type { WebGPUUniformReferenceSolver } from "../lib/methods/uniform/webgpu-uniform-reference";

const option = (key: string) => process.argv.find(a => a.startsWith(`--${key}=`))?.slice(key.length + 3);
const name = process.argv[2]; assert.ok(name); assert.match(name, /^[a-z0-9-]+$/);
for (const a of process.argv.slice(3)) assert.match(a, /^--(?:baseline|candidate|steps|from|scene|scene-file|out|transfer|profile|baseline-values|candidate-values)=.+$/);
const steps = Number(option("steps") ?? 240), from = Number(option("from") ?? 13);
assert.ok(Number.isInteger(steps) && Number.isInteger(from) && from > 0 && steps >= from);
const roots = [option("baseline"), option("candidate")]; assert.ok(roots.every(Boolean));
assert.ok(!option("transfer") || ["flip", "apic"].includes(option("transfer")!));
assert.ok(!option("profile") || ["true", "false"].includes(option("profile")!));
const profiled = option("profile") === "true";
type PoolView={state:GPUBuffer;layout:{entries:number;blocks:number;state:{table:number;window:number;blocks:number}};pin(p:readonly {x:number;y:number;z:number}[]):void};
let extent:number[]=[];
const replacements:Record<string,number>={};
function directAddress(code:string):string{
 const [x,y,z]=extent;
 code=code.replace(/fn (\w*PageWord)\(page:vec3i\)->u32\{\n let world=page\+[\s\S]*?\n\}/g,(_body,name:string)=>{
  replacements.page=(replacements.page??0)+1;const prefix=name.slice(0,-8);
  return `fn ${name}(page:vec3i)->u32{let world=page+${prefix}FramePage();let p=bitcast<vec3u>(world);let block=1u+p.x+${x}u*(p.y+${y}u*p.z);return select(0xffffffffu,block,all(world>=vec3i(0))&&all(world<vec3i(${x},${y},${z})));}`;
 });
 return code.replace(/fn (\w*BlockWord)\(b:u32\)->u32\{\n let raw=[\s\S]*?\n\}/g,(_body,name:string)=>{
  replacements.block=(replacements.block??0)+1;const prefix=name.slice(0,-9);
  return `fn ${name}(b:u32)->u32{let i=b-1u;let world=vec3u(i%${x}u,(i/${x}u)%${y}u,i/${x!*y!}u);let p=(world-bitcast<vec3u>(${prefix}FramePage()))&vec3u(63u);return select(0xffffffffu,p.x|(p.y<<6u)|(p.z<<12u),b>0u&&b<=${x!*y!*z!}u);}`;
 });
}
async function verifyDense(solver:WebGPUUniformReferenceSolver,device:GPUDevice){
 const pool=(solver as unknown as {mixedFrame:{pagePool:PoolView}}).mixedFrame.pagePool,l=pool.layout;
 const read=device.createBuffer({size:pool.state.size,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
 const e=device.createCommandEncoder();e.copyBufferToBuffer(pool.state,0,read,0,pool.state.size);device.queue.submit([e.finish()]);await read.mapAsync(GPUMapMode.READ);
 const words=new Uint32Array(read.getMappedRange());assert.equal(words[5],0,'pool fatal');
 const [px,py,pz]=extent as [number,number,number];assert.equal(l.blocks,px*py*pz+2);assert.equal(words[l.state.window+12],1);
 for(let z=0;z<pz;z++)for(let y=0;y<py;y++)for(let x=0;x<px;x++){
  const b=1+x+px*(y+py*z),entry=x|(y<<6)|(z<<12);
  assert.equal(words[l.state.table+entry],b,'stable direct page mapping');assert.equal(words[l.state.blocks+b],entry,'stable inverse block mapping');
 }
 read.unmap();read.destroy();
}
let profile: GPUPassProfile | undefined;
const sceneId = option("scene") ?? "sparse-cm12-long-dam-break";
const out = option("out") ?? "artifacts/nbflip-ui-perf"; mkdirSync(out, { recursive: true });
// The Dawn binding's own polling garbage would put a collection in one arm's frame or the other's.
paceDawnEventPolling();
const errors: string[] = [], solvers: WebGPUUniformReferenceSolver[] = [];
const rows: { frame: number; baseline_ms: number; candidate_ms: number; saved_ms: number; first: string; state: unknown[] }[] = [];
const snapshots: Record<string, unknown>[] = [];
let raw: GPUDevice | undefined;
try {
  const dawn = await import(pathToFileURL(resolve("node_modules/webgpu/index.js")).href);
  Object.assign(globalThis, dawn.globals);
  const gpu = createProcessRetainedDawnGPU(dawn, ["backend=metal"]), adapter = await gpu.requestAdapter(); assert.ok(adapter);
  raw = await adapter.requestDevice({ requiredFeatures: ["timestamp-query", ...(adapter.features.has("subgroups") ? ["subgroups" as const] : [])], requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
  raw.addEventListener("uncapturederror", event => { event.preventDefault(); errors.push(event.error.message); });
  if (profiled) profile = new GPUPassProfile(raw);
  let dt = 0;
  for (const [index, root] of roots.entries()) {
    const cwd = resolve(root!);
    const load = (file: string) => import(pathToFileURL(resolve(cwd, file)).href);
    const [{ uniformNarrowBandMethod: method }, { resolveMethodValues }, { sceneDocument }, { getSceneDefinition }, { managedGPUDevice }] = await Promise.all([
      load("lib/methods/uniform/uniform-narrow-band-method.ts"), load("lib/core/method-contract.ts"), load("lib/core/scene-definition.ts"), load("lib/core/scenes.ts"), load("lib/core/gpu-compilation-manager.ts"),
    ]);
    const typed = method as typeof uniformNarrowBandMethod;
    const definition = getSceneDefinition(sceneId), scene = option("scene-file") ? JSON.parse(readFileSync(option("scene-file")!, "utf8")) : structuredClone(sceneDocument(definition));
    assert.equal(scene.fluid.pagePool?.permute,undefined,"linear initial allocation");
    scene.fluid.pagePool={...scene.fluid.pagePool,air:[]}; // Explicit full initial residency on both arms.
    if (definition.methodProfile) assert.equal(definition.methodProfile.methodId, typed.id);
    const quality = definition.methodProfile?.quality ?? "balanced";
    const values = resolveMethodValues(typed, quality, { ...typed.appDefaults, ...definition.methodProfile?.overrides, ...(option("transfer") ? {particleTransfer: option("transfer")} : {}), ...JSON.parse(option(index ? "candidate-values" : "baseline-values") ?? "{}") });
    const step = typed.effectiveStep_s?.(scene, values) ?? scene.numerics.fixedDt_s;
    const adaptivePolicyHash=createHash("sha256").update(readFileSync(resolve(cwd,"lib/methods/uniform/uniform-narrow-band-budget.wgsl.ts"))).digest("hex");
    if (index) {
      assert.equal(step, dt); assert.deepEqual(scene, snapshots[0]!.scene);
    }
    dt = step; assert.ok(steps * dt <= scene.duration_s + 1e-9);
    const hash = createHash("sha256"), directory = resolve(cwd, "lib/methods/uniform");
    for (const f of readdirSync(directory).filter(f => f.endsWith(".ts")).sort()) { hash.update(f); hash.update(readFileSync(resolve(directory, f))); }
    const dimensions=await load("lib/core/scene-lattice-dimensions.ts");
    const pages=dimensions.sceneFluidLatticeDimensions(scene,Number.MAX_SAFE_INTEGER).map((x:number)=>x/32) as number[];
    assert.equal(scene.container.top,"closed");assert.notEqual(scene.container.walls,"none");assert.ok(pages.every(x=>Number.isInteger(x)&&x>0));
    if(index)assert.deepEqual(pages,extent);else extent=pages;
    const baseDevice=profile?.device??raw;
    const variantDevice=index?new Proxy(baseDevice,{get(target,key){
      if(key==='createShaderModule')return (d:GPUShaderModuleDescriptor)=>target.createShaderModule({...d,code:directAddress(d.code)});
      const v=Reflect.get(target,key,target);return typeof v==='function'?v.bind(target):v;
    }}):baseDevice;
    const start = performance.now();
    solvers.push(await typed.createSolverAsync!(managedGPUDevice(variantDevice, { requireWorkerRealm: false }), scene, quality, values, undefined, () => {}) as WebGPUUniformReferenceSolver);
    await raw.queue.onSubmittedWorkDone();
    const pool=(solvers[index] as unknown as {mixedFrame:{pagePool:PoolView}}).mixedFrame.pagePool;
    const pins:{x:number;y:number;z:number}[]=[];
    for(let z=0;z<pages[2]!;z++)for(let y=0;y<pages[1]!;y++)for(let x=0;x<pages[0]!;x++)pins.push({x,y,z});
    pool.pin(pins);await verifyDense(solvers[index]!,raw);
    snapshots.push({ root: cwd, note: existsSync(resolve(cwd,"benchmark-snapshot-note.json")) ? JSON.parse(readFileSync(resolve(cwd,"benchmark-snapshot-note.json"),"utf8")) : undefined, sourceHash: hash.digest("hex"), adaptivePolicyHash, quality, values, scene, dt, construction_ms: performance.now() - start });
    console.log(JSON.stringify({ phase: "constructed", variant: index ? "candidate" : "baseline", ...snapshots[index], scene: undefined }));
  }
  assert.equal(snapshots[0]!.sourceHash,snapshots[1]!.sourceHash,"same source for both arms");
  for (let frame = 1; frame <= steps; frame++) {
    const wall = [0, 0], state: unknown[] = [];
    for (const i of frame % 2 ? [0, 1] : [1, 0]) {
      const solver = solvers[i]!;
      if (profile && frame >= from) profile.start();
      const start = performance.now();
      assert.ok(solver.advanceTo(frame * dt, [])); await solver.awaitFrameCompletion(); wall[i] = performance.now() - start;
      assert.equal(solver.info.simulationPipelineError, undefined);
      assert.ok(Math.abs((solver.info.completedTime_s ?? 0) - frame * dt) < 1e-8);
      state[i] = { passes: profile && frame >= from ? await profile.finish() : undefined, ...solver.workDomainInfo, pages: solver.pagePoolInfo ? {...solver.pagePoolInfo} : undefined, ...solver.narrowBandFlipInfo, fineTiles: solver.workDomainInfo?.fineTiles ?? solver.info.uniformMixedFineTiles, bandTiles: solver.info.uniformPressureBandTiles, bandResidual: solver.info.uniformPressureBandResidual };
    }
    if(frame===12||frame===steps)for(const solver of solvers)await verifyDense(solver,raw);
    rows.push({ frame, baseline_ms: wall[0]!, candidate_ms: wall[1]!, saved_ms: wall[0]! - wall[1]!, first: frame % 2 ? "baseline" : "candidate", state });
    if (frame % 12 === 0 || frame === steps) console.log(JSON.stringify({ ...rows.at(-1), state: undefined }));
  }
  assert.deepEqual(errors, []);
  const measured = rows.filter(r => r.frame >= from), mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / xs.length;
  const saved = measured.map(r => r.saved_ms).sort((a, b) => a - b);
  const median = (xs: number[]) => {
    const sorted = [...xs].sort((a, b) => a - b), middle = Math.floor(sorted.length / 2);
    return sorted.length % 2 ? sorted[middle]! : (sorted[middle - 1]! + sorted[middle]!) / 2;
  };
  const trim = Math.floor(saved.length * .05);
  const summary = {
    frames: measured.length,
    baseline_ms: mean(measured.map(r => r.baseline_ms)), candidate_ms: mean(measured.map(r => r.candidate_ms)), saved_ms: mean(saved),
    baseline_median_ms: median(measured.map(r => r.baseline_ms)), candidate_median_ms: median(measured.map(r => r.candidate_ms)),
    paired_median_saved_ms: median(saved), paired_trimmed_mean_saved_ms: mean(saved.slice(trim, saved.length - trim)),
    baseline_first_saved_ms: mean(measured.filter(r => r.first === "baseline").map(r => r.saved_ms)),
    candidate_first_saved_ms: mean(measured.filter(r => r.first === "candidate").map(r => r.saved_ms)),
  };
  const passMeans = solvers.map((_, i) => {
    const means: Record<string, number> = {};
    for (const row of measured) for (const p of (row.state[i] as {passes?: {label:string;ms:number}[]}).passes ?? []) means[p.label] = (means[p.label] ?? 0) + p.ms / measured.length;
    return Object.fromEntries(Object.entries(means).sort((a,b)=>b[1]-a[1]));
  });
  const certificates = [];
  for (const solver of solvers) {
    const o = (solver as unknown as {mixedFrame:{ownership:{speeds:GPUBuffer;capacity:{tiles:number}}}}).mixedFrame.ownership;
    const n=o.capacity.tiles, read=raw.createBuffer({size:4*n,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
    const e=raw.createCommandEncoder();e.copyBufferToBuffer(o.speeds,4*n,read,0,4*n);raw.queue.submit([e.finish()]);
    await read.mapAsync(GPUMapMode.READ);const histogram=[0,0,0,0];
    for(const w of new Uint32Array(read.getMappedRange()))histogram[w>>>30]!++;
    certificates.push({tileKeys:n,unclassified:histogram[0],regular:histogram[1],coarseInReach:histogram[2],saturated:histogram[3]});read.unmap();read.destroy();
  }
  const physics = [];
  for (const solver of solvers) {
    const s = await solver.readStats(); physics.push({ maxSpeed_m_s: s.maxSpeed_m_s, volumeCellSum: s.volumeCellSum, representedVolumeCellSum: s.representedVolumeCellSum, volumeDrift: s.volumeDrift, front_m: s.front_m, pressureResidual: s.uniformPressureAcceptedResidual, bandResidual: s.uniformPressureBandResidual });
  }
  writeFileSync(`${out}/${name}.json`, JSON.stringify({ date: new Date().toISOString(), host: { cpu: cpus()[0]?.model, memory: totalmem(), release: release(), node: process.version }, sceneId, scope: "alternating simulation advance and completion on one device; excludes construction and renderer", snapshots, profiled, fixedResidency:true, extent, replacements, summary, passMeans, certificates, physics, rows, errors }, null, 2) + "\n");
  console.log(JSON.stringify({ phase: "complete", summary }));
} catch (error) {
  writeFileSync(`${out}/${name}-failed.json`, JSON.stringify({ snapshots, rows, errors, error: String(error) }, null, 2) + "\n"); throw error;
} finally {
  for (const solver of solvers) solver.destroy(); await raw?.queue.onSubmittedWorkDone(); profile?.destroy(); raw?.destroy();
}
