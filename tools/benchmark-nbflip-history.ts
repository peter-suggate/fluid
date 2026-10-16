/** Compare historical NB-FLIP implementations, including changed defaults and policies.
 * This is a historical comparison, NOT a physics-equivalence test. Records both resolved settings.
 * --transfer=flip matches transfer choice; --profile=true adds diagnostic pass timestamps.
 * --baseline-values=JSON and --candidate-values=JSON override a single arm.
 * --sync=N runs both arms from the same GPU state from frame N on: before every frame >= N
 * every copyable buffer and texture of one arm is copied into the other (the source arm
 * alternates by frame), so the arms time the same work and only the code differs. The
 * simulations otherwise diverge chaotically (atomics order), and late in a splash that
 * divergence alone moves a whole frame by several ms. Needs identical resource layouts
 * (same creation sequence of sizes, usages and formats) in both arms; it asserts this.
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
for (const a of process.argv.slice(3)) assert.match(a, /^--(?:baseline|candidate|steps|from|scene|scene-file|out|transfer|profile|baseline-values|candidate-values|sync)=.+$/);
const steps = Number(option("steps") ?? 240), from = Number(option("from") ?? 13);
assert.ok(Number.isInteger(steps) && Number.isInteger(from) && from > 0 && steps >= from);
const roots = [option("baseline"), option("candidate")]; assert.ok(roots.every(Boolean));
assert.ok(!option("transfer") || ["flip", "apic"].includes(option("transfer")!));
assert.ok(!option("profile") || ["true", "false"].includes(option("profile")!));
const profiled = option("profile") === "true";
const syncFrom = option("sync") ? Number(option("sync")) : Infinity;
assert.ok(syncFrom === Infinity || (Number.isInteger(syncFrom) && syncFrom > 1 && syncFrom <= from));
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
  // Resources each arm creates, in creation order, for --sync. Map-usage buffers are host staging: skipped.
  type Tracked = { buffer?: GPUBuffer; texture?: GPUTexture; key: string; dead: boolean };
  const owned: Tracked[][] = [[], []]; let owner = -1;
  const copyable = GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST, mapped = GPUBufferUsage.MAP_READ | GPUBufferUsage.MAP_WRITE;
  const track = (item: Tracked, object: { destroy(): void }) => {
    if (owner < 0) return; owned[owner]!.push(item);
    const destroy = object.destroy.bind(object);
    Object.defineProperty(object, "destroy", { value: () => { item.dead = true; destroy(); } });
  };
  const base = profile?.device ?? raw;
  const device = syncFrom === Infinity ? base : new Proxy(base, { get(target, key) {
    if (key === "createBuffer") return (d: GPUBufferDescriptor) => {
      if (d.usage & mapped) return target.createBuffer(d);
      const buffer = target.createBuffer({ ...d, usage: d.usage | copyable });
      track({ buffer, key: `b ${d.size} ${d.usage}`, dead: false }, buffer); return buffer;
    };
    if (key === "createTexture") return (d: GPUTextureDescriptor) => {
      const texture = target.createTexture({ ...d, usage: d.usage | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST });
      if ((d.sampleCount ?? 1) === 1) track({ texture, key: `t ${d.format} ${JSON.stringify(d.size)} ${d.dimension} ${d.mipLevelCount ?? 1}`, dead: false }, texture);
      return texture;
    };
    const value = Reflect.get(target, key, target); return typeof value === "function" ? value.bind(target) : value;
  } });
  /** Copy every live tracked resource of arm `source` into the other arm's twin. */
  const sync = async (source: number) => {
    const [from_, to] = [owned[source]!.filter(r => !r.dead), owned[1 - source]!.filter(r => !r.dead)];
    assert.equal(from_.length, to.length, "sync: arms hold different resource counts");
    const e = raw!.createCommandEncoder();
    for (const [i, a] of from_.entries()) {
      const b = to[i]!; assert.equal(a.key, b.key, `sync: resource ${i} differs`);
      if (a.buffer) e.copyBufferToBuffer(a.buffer, 0, b.buffer!, 0, a.buffer.size);
      else {
        const t = a.texture!;
        for (let mip = 0; mip < t.mipLevelCount; mip++) e.copyTextureToTexture({ texture: t, mipLevel: mip }, { texture: b.texture!, mipLevel: mip },
          [Math.max(1, t.width >> mip), Math.max(1, t.height >> mip), t.dimension === "3d" ? Math.max(1, t.depthOrArrayLayers >> mip) : t.depthOrArrayLayers]);
      }
    }
    raw!.queue.submit([e.finish()]); await raw!.queue.onSubmittedWorkDone();
  };
  let dt = 0;
  for (const [index, root] of roots.entries()) {
    const cwd = resolve(root!);
    const load = (file: string) => import(pathToFileURL(resolve(cwd, file)).href);
    const [{ uniformNarrowBandMethod: method }, { resolveMethodValues }, { sceneDocument }, { getSceneDefinition }, { managedGPUDevice }] = await Promise.all([
      load("lib/methods/uniform/uniform-narrow-band-method.ts"), load("lib/core/method-contract.ts"), load("lib/core/scene-definition.ts"), load("lib/core/scenes.ts"), load("lib/core/gpu-compilation-manager.ts"),
    ]);
    const typed = method as typeof uniformNarrowBandMethod;
    const definition = getSceneDefinition(sceneId), scene = option("scene-file") ? JSON.parse(readFileSync(option("scene-file")!, "utf8")) : structuredClone(sceneDocument(definition));
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
    const start = performance.now();
    owner = index;
    solvers.push(await typed.createSolverAsync!(managedGPUDevice(device, { requireWorkerRealm: false }), scene, quality, values, undefined, () => {}) as WebGPUUniformReferenceSolver);
    await raw.queue.onSubmittedWorkDone();
    snapshots.push({ root: cwd, note: existsSync(resolve(cwd,"benchmark-snapshot-note.json")) ? JSON.parse(readFileSync(resolve(cwd,"benchmark-snapshot-note.json"),"utf8")) : undefined, sourceHash: hash.digest("hex"), adaptivePolicyHash, quality, values, scene, dt, construction_ms: performance.now() - start });
    console.log(JSON.stringify({ phase: "constructed", variant: index ? "candidate" : "baseline", ...snapshots[index], scene: undefined }));
  }
  for (let frame = 1; frame <= steps; frame++) {
    const wall = [0, 0], state: unknown[] = [];
    if (frame >= syncFrom) await sync(frame % 2);
    for (const i of frame % 2 ? [0, 1] : [1, 0]) {
      const solver = solvers[i]!; owner = i;
      if (profile && frame >= from) profile.start();
      const start = performance.now();
      assert.ok(solver.advanceTo(frame * dt, [])); await solver.awaitFrameCompletion(); wall[i] = performance.now() - start;
      assert.equal(solver.info.simulationPipelineError, undefined);
      assert.ok(Math.abs((solver.info.completedTime_s ?? 0) - frame * dt) < 1e-8);
      owner = -1;
      state[i] = { passes: profile && frame >= from ? await profile.finish() : undefined, ...solver.workDomainInfo, pages: solver.pagePoolInfo ? {...solver.pagePoolInfo} : undefined, ...solver.narrowBandFlipInfo, fineTiles: solver.workDomainInfo?.fineTiles ?? solver.info.uniformMixedFineTiles, bandTiles: solver.info.uniformPressureBandTiles, bandResidual: solver.info.uniformPressureBandResidual };
    }
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
    synced_from: Number.isFinite(syncFrom) ? syncFrom : undefined,
    ...(profiled ? (() => {
      const sum = (row: typeof rows[number], i: number) => ((row.state[i] as { passes?: { ms: number }[] }).passes ?? []).reduce((a, p) => a + p.ms, 0);
      const d = measured.map(r => sum(r, 0) - sum(r, 1)).sort((a, b) => a - b), t = Math.floor(d.length * .05);
      return { baseline_pass_sum_ms: mean(measured.map(r => sum(r, 0))), candidate_pass_sum_ms: mean(measured.map(r => sum(r, 1))),
        pass_sum_saved_ms: mean(d), pass_sum_saved_median_ms: median(d), pass_sum_saved_trimmed_ms: mean(d.slice(t, d.length - t)) };
    })() : {}),
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
  writeFileSync(`${out}/${name}.json`, JSON.stringify({ date: new Date().toISOString(), host: { cpu: cpus()[0]?.model, memory: totalmem(), release: release(), node: process.version }, sceneId, scope: "alternating simulation advance and completion on one device; excludes construction and renderer", snapshots, profiled, summary, passMeans, certificates, physics, rows, errors }, null, 2) + "\n");
  console.log(JSON.stringify({ phase: "complete", summary }));
} catch (error) {
  writeFileSync(`${out}/${name}-failed.json`, JSON.stringify({ snapshots, rows, errors, error: String(error) }, null, 2) + "\n"); throw error;
} finally {
  for (const solver of solvers) solver.destroy(); await raw?.queue.onSubmittedWorkDone(); profile?.destroy(); raw?.destroy();
}
