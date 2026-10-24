/** Alternate fenced simulation frames from two source snapshots on one device,
 * a region-tier fixture (tools/region-tier-scenes.ts) or a registered scene,
 * at an optional pool reservation. Each frame runs both arms, alternating
 * which goes first; it reports paired wall times, final field hashes (read
 * by live page identity: which request gets which block is a race) and
 * every frame's work history: the region tiers read at its end, and its
 * receipt (band, work words, heat evidence) by the frame it describes, not
 * the newest one solver.info shows. Two instances of one source are not
 * bit-identical (atomic slot allocation), so the physics gate is
 * gate-repeat-spread.ts; --identity fails the run on any field or history
 * difference. --profile counts each arm's passes, dispatches, workgroups,
 * clears and copies by pass label (a separate diagnostic run: it adds CPU
 * work).
 * node --import tsx tools/benchmark-region-tier-paired.ts NAME
 *   --baseline=/abs/snapshot --candidate=/abs/snapshot
 *   [--scene=rest|cold|warm|splash|hot|<scene id>] [--length=128] [--depth=32]
 *   [--pool-blocks=N] [--steps=100] [--from=40] [--profile] [--identity] [--keep-going]
 *   [--hash-at=F,F] [--diff] [--out=DIR] [--gate="RUNNER MODE"]
 * --keep-going: with --identity, record work-history differences and run on
 * to the field hashes (the run still fails on either). --hash-at=10,20:
 * compare the field hashes after those frames too; --diff prints where they
 * differ (float index, both values).
 * --gate: construct both arms (mostly CPU: pipeline compiles) first, then
 * hold a GPU slot only over the frames and readback: the gate command (a
 * shared runner and its mode, e.g. "fluid-gpu-run.sh check") runs `cat`
 * once the slot is held, and the slot is released when its stdin closes.
 */
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdirSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import { cpus } from "node:os";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { spawn, type ChildProcess } from "node:child_process";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { createProcessRetainedDawnGPU, paceDawnEventPolling } from "../lib/harness/node-dawn-provider";
import type { SceneDescription } from "../lib/core/model";
import type { WebGPUUniformReferenceSolver } from "../lib/methods/uniform/webgpu-uniform-reference";
import { createRegionTierScene, REGION_TIER_SCENES, type RegionTierScene } from "./region-tier-scenes";

const option = (key: string) => process.argv.find(a => a.startsWith(`--${key}=`))?.slice(key.length + 3);
const name = process.argv[2]; assert.ok(name); assert.match(name, /^[a-z0-9-]+$/);
for (const a of process.argv.slice(3)) assert.match(a, /^--(?:(?:baseline|candidate|steps|from|scene|length|depth|pool-blocks|out|gate|hash-at)=.+|profile|keep-going|diff|identity)$/);
const diffFields = process.argv.includes("--diff"), identity = process.argv.includes("--identity");
const hashAt = new Set((option("hash-at") ?? "").split(",").filter(Boolean).map(Number));
const steps = Number(option("steps") ?? 100), from = Number(option("from") ?? 40), profile = process.argv.includes("--profile"), keepGoing = process.argv.includes("--keep-going");
assert.ok(Number.isInteger(steps) && Number.isInteger(from) && from > 0 && steps >= from);
const roots = [option("baseline"), option("candidate")].map(r => resolve(r!)); assert.ok(roots.every(Boolean));
const sceneOption = option("scene") ?? "rest", region = (REGION_TIER_SCENES as readonly string[]).includes(sceneOption);
const length = Number(option("length") ?? 128), depth = Number(option("depth") ?? 32);
const poolBlocks = option("pool-blocks") === undefined ? undefined : Number(option("pool-blocks"));
assert.ok(poolBlocks === undefined || (Number.isInteger(poolBlocks) && poolBlocks >= 3));
const out = option("out") ?? "artifacts/region-tiers/paired"; mkdirSync(out, { recursive: true });
paceDawnEventPolling();
/** Hold the gate's slot until the returned process's stdin closes. */
async function acquireGate(gate: string): Promise<ChildProcess> {
  const [command, ...args] = gate.split(" ").filter(Boolean);
  const child = spawn(command!, [...args, "cat"], { stdio: ["pipe", "pipe", "inherit"] });
  await new Promise<void>((done, fail) => { child.stdout!.once("data", () => done()); child.once("exit", code => fail(new Error(`gate exited ${code}`))); child.stdin!.write("held\n"); });
  return child;
}
let lock: ChildProcess | undefined;

interface Counts { passes: number; dispatches: number; indirect: number; workgroups: number; clears: number; clearBytes: number; copies: number; copyBytes: number }
type Profile = Record<string, Counts>;
/** Per-label counts of whatever the arm records between begin and end. */
function commandProfile(device: GPUDevice) {
  const encoder = Object.getPrototypeOf(device.createCommandEncoder()) as GPUCommandEncoder;
  const begin = encoder.beginComputePass, clear = encoder.clearBuffer, copy = encoder.copyBufferToBuffer;
  let current: Profile | undefined;
  const row = (label: string) => current![label] ??= { passes: 0, dispatches: 0, indirect: 0, workgroups: 0, clears: 0, clearBytes: 0, copies: 0, copyBytes: 0 };
  encoder.beginComputePass = function (descriptor?: GPUComputePassDescriptor) {
    const pass = begin.call(this, descriptor);
    if (current) {
      const r = row(String(descriptor?.label ?? "unlabelled")); r.passes++;
      const dispatch = pass.dispatchWorkgroups, indirect = pass.dispatchWorkgroupsIndirect;
      pass.dispatchWorkgroups = function (x: number, y = 1, z = 1) { r.dispatches++; r.workgroups += x * y * z; dispatch.call(this, x, y, z); };
      pass.dispatchWorkgroupsIndirect = function (b: GPUBuffer, o: number) { r.dispatches++; r.indirect++; indirect.call(this, b, o); };
    }
    return pass;
  };
  encoder.clearBuffer = function (this: GPUCommandEncoder, buffer: GPUBuffer, offset = 0, size?: number) {
    if (current) { const r = row(`clear: ${buffer.label}`); r.clears++; r.clearBytes += size ?? buffer.size - offset; }
    return clear.call(this, buffer, offset, size);
  } as GPUCommandEncoder["clearBuffer"];
  encoder.copyBufferToBuffer = function (this: GPUCommandEncoder, ...args: unknown[]) {
    if (current) { const r = row(`copy: ${(args[0] as GPUBuffer).label}`); r.copies++; r.copyBytes += Number(args.length >= 5 ? args[4] : (args[0] as GPUBuffer).size); }
    return (copy as (...a: unknown[]) => void).apply(this, args);
  } as GPUCommandEncoder["copyBufferToBuffer"];
  return { begin: () => { current = {}; }, end: () => { const r = current!; current = undefined; return r; },
    destroy: () => { encoder.beginComputePass = begin; encoder.clearBuffer = clear; encoder.copyBufferToBuffer = copy; } };
}
const total = (p: Profile) => Object.values(p).reduce((s, r) => ({ dispatches: s.dispatches + r.dispatches, indirect: s.indirect + r.indirect, workgroups: s.workgroups + r.workgroups, clears: s.clears + r.clears, clearBytes: s.clearBytes + r.clearBytes, copies: s.copies + r.copies, copyBytes: s.copyBytes + r.copyBytes }), { dispatches: 0, indirect: 0, workgroups: 0, clears: 0, clearBytes: 0, copies: 0, copyBytes: 0 });

const errors: string[] = [], solvers: WebGPUUniformReferenceSolver[] = [], helpers: { readMixedTexture: (d: GPUDevice, t: GPUTexture, a?: unknown) => Promise<Float32Array>; readMixedBuffer: (d: GPUDevice, b: GPUBuffer) => Promise<Float32Array>; readUniformLiveAddress: (d: GPUDevice, s: WebGPUUniformReferenceSolver) => Promise<unknown> }[] = [];
const rows: { frame: number; baseline_ms: number; candidate_ms: number; saved_ms: number; first: string; history: unknown[]; profile?: Profile[] }[] = [];
const snapshots: Record<string, unknown>[] = [];
let raw: GPUDevice | undefined, commands: ReturnType<typeof commandProfile> | undefined;
let scene: SceneDescription | undefined;
const read = { buffer: undefined as GPUBuffer | undefined };
try {
  const dawn = await import(pathToFileURL(resolve("node_modules/webgpu/index.js")).href);
  Object.assign(globalThis, dawn.globals);
  const gpu = createProcessRetainedDawnGPU(dawn, ["backend=metal"]), adapter = await gpu.requestAdapter(); assert.ok(adapter);
  const features = (["timestamp-query", "subgroups", "float32-filterable"] as GPUFeatureName[]).filter(f => adapter.features.has(f));
  raw = await adapter.requestDevice({ requiredFeatures: features, requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
  raw.addEventListener("uncapturederror", event => { event.preventDefault(); errors.push(event.error.message); });
  read.buffer = raw.createBuffer({ size: 32, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  let dt = 0;
  for (const [index, cwd] of roots.entries()) {
    const load = (file: string) => import(pathToFileURL(resolve(cwd, file)).href);
    const [{ uniformNarrowBandMethod: method }, { resolveMethodValues }, { sceneDocument }, { getSceneDefinition }, { managedGPUDevice }, fields, geometric] = await Promise.all([
      load("lib/methods/uniform/uniform-narrow-band-method.ts"), load("lib/core/method-contract.ts"), load("lib/core/scene-definition.ts"), load("lib/core/scenes.ts"), load("lib/core/gpu-compilation-manager.ts"),
      load("tests/helpers/uniform-mixed-native-fields.ts"), load("tests/helpers/uniform-geometric.ts"),
    ]);
    helpers.push({ readMixedTexture: fields.readMixedTexture, readMixedBuffer: fields.readMixedBuffer, readUniformLiveAddress: geometric.readUniformLiveAddress });
    let values: unknown, quality: string, built: SceneDescription;
    if (region) {
      built = createRegionTierScene(sceneOption as RegionTierScene, length, depth); quality = "balanced";
      values = resolveMethodValues(method, quality, { ...method.appDefaults, timeStep: "scene" });
    } else {
      const definition = getSceneDefinition(sceneOption); built = structuredClone(sceneDocument(definition));
      quality = definition.methodProfile.quality; values = resolveMethodValues(method, quality, { ...method.appDefaults, ...definition.methodProfile.overrides });
    }
    if (poolBlocks !== undefined) built.fluid = { ...built.fluid, pagePool: { ...built.fluid?.pagePool, blocks: poolBlocks } } as SceneDescription["fluid"];
    const step = method.effectiveStep_s?.(built, values) ?? built.numerics.fixedDt_s;
    built.duration_s = Math.max(built.duration_s, steps * step);
    if (index) { assert.equal(step, dt); assert.deepEqual(values, snapshots[0]!.values); assert.deepEqual(built, scene); }
    scene = built; dt = step;
    const hash = createHash("sha256"), directory = resolve(cwd, "lib/methods/uniform");
    for (const f of readdirSync(directory).filter(f => f.endsWith(".ts")).sort()) { hash.update(f); hash.update(readFileSync(resolve(directory, f))); }
    const start = performance.now();
    solvers.push(await method.createSolverAsync!(managedGPUDevice(raw, { requireWorkerRealm: false }), built, quality, values, undefined, () => {}) as WebGPUUniformReferenceSolver);
    await raw.queue.onSubmittedWorkDone();
    snapshots.push({ root: cwd, sourceHash: hash.digest("hex"), quality, values, dt, construction_ms: performance.now() - start });
    console.log(JSON.stringify({ phase: "constructed", variant: index ? "candidate" : "baseline", root: cwd, construction_ms: snapshots[index]!.construction_ms }));
  }
  const gate = option("gate");
  if (gate) { const t = performance.now(); lock = await acquireGate(gate); console.log(JSON.stringify({ phase: "locked", gate, wait_ms: performance.now() - t })); }
  if (profile) commands = commandProfile(raw);
  const tiersOf = async (solver: WebGPUUniformReferenceSolver) => {
    const heat = (solver as unknown as { mixedFrame: { heat: { tiers: GPUBuffer; regions: number } } }).mixedFrame.heat;
    const encoder = raw!.createCommandEncoder(); encoder.copyBufferToBuffer(heat.tiers, 4 * heat.regions, read.buffer!, 0, 32);
    raw!.queue.submit([encoder.finish()]); await read.buffer!.mapAsync(GPUMapMode.READ);
    const tiers = [...new Uint32Array(read.buffer!.getMappedRange()).slice(0, 5)]; read.buffer!.unmap(); return tiers;
  };
  /** An arm's hashed fields, read by live page identity. A mixed frame's
   * pressure is its presentation buffer: gridPressureTexture is a 1x1x1
   * placeholder there, the same bytes every frame. */
  const fieldData = async (i: number, solver: WebGPUUniformReferenceSolver, address: unknown) => {
    const out: Record<string, Float32Array> = {};
    for (const [key, texture] of Object.entries({ velocity: solver.velocityTexture, volume: solver.volumeTexture, phi: solver.vertexPhiTexture! }))
      out[key] = await helpers[i]!.readMixedTexture(raw!, texture, address);
    const pressure = (solver as unknown as { mixedFrame?: { presentation: { pressure: GPUBufferBinding } } }).mixedFrame?.presentation.pressure;
    out.pressure = pressure ? await helpers[i]!.readMixedBuffer(raw!, pressure.buffer) : await helpers[i]!.readMixedTexture(raw!, solver.gridPressureTexture, address);
    return out;
  };
  /** Each arm's field hashes, read by live page identity. */
  const hashFields = async () => {
    const all: Record<string, string>[] = [], arrays: Record<string, Float32Array>[] = [];
    for (const [i, solver] of solvers.entries()) {
      arrays.push({});
      const address = await helpers[i]!.readUniformLiveAddress(raw!, solver), out: Record<string, string> = {};
      for (const [key, data] of Object.entries(await fieldData(i, solver, address))) {
        out[key] = createHash("sha256").update(new Uint8Array(data.buffer, data.byteOffset, data.byteLength)).digest("hex").slice(0, 16);
        if (diffFields) arrays[i]![key] = data;
      }
      all.push(out);
    }
    // --diff: where two arms' fields differ (float index, both values).
    if (diffFields) for (const key of Object.keys(arrays[0]!)) {
      const a = arrays[0]![key]!, b = arrays[1]![key]!; let count = 0, max = 0; const first: [number, number, number][] = [];
      for (let j = 0; j < Math.min(a.length, b.length); j++) if (!Object.is(a[j], b[j])) { count++; max = Math.max(max, Math.abs(a[j]! - b[j]!)); if (first.length < 8) first.push([j, a[j]!, b[j]!]); }
      if (count || a.length !== b.length) console.log(JSON.stringify({ phase: "diff", key, length: [a.length, b.length], count, max, first }));
    }
    return all;
  };
  const frameHashes: { frame: number; match: boolean; differ: string[] }[] = [];
  let residualDiffers = 0;
  const historyDiffs: { frame: number; baseline: unknown; candidate: unknown }[] = [];
  // Each arm's frame receipts by the frame they describe (the mixed frame's
  // lagged plans: band, work words, remap list, heat evidence). solver.info
  // shows the newest receipt resolved so far, which lags by timing.
  type Lagged = { frame: number; bandTiles: number; bandBricks: number; bandLevel1: number; bandLevel2: number; bandCycles: number; work: Uint32Array; listed: number; heat: Uint32Array };
  const receipts = solvers.map(() => new Map<number, string>()), compared = new Set<number>();
  const harvest = (i: number) => {
    for (const p of (solvers[i] as unknown as { mixedFrame: { lagged: readonly Lagged[] } }).mixedFrame.lagged) if (p.frame >= 0 && !compared.has(p.frame))
      receipts[i]!.set(p.frame, JSON.stringify({ band: p.bandTiles, bricks: p.bandBricks, levels: [p.bandLevel1, p.bandLevel2], cycles: p.bandCycles, work: [...p.work], listed: p.listed, heat: [...p.heat] }));
  };
  for (let frame = 1; frame <= steps; frame++) {
    const wall = [0, 0], history: unknown[] = [], shown: unknown[] = [], profiles: Profile[] = [];
    for (const i of frame % 2 ? [0, 1] : [1, 0]) {
      const solver = solvers[i]!; commands?.begin();
      const start = performance.now();
      assert.ok(solver.advanceTo(frame * dt, [])); await solver.awaitFrameCompletion(); wall[i] = performance.now() - start;
      if (commands) profiles[i] = commands.end();
      assert.equal(solver.info.simulationPipelineError, undefined);
      assert.ok(Math.abs((solver.info.completedTime_s ?? 0) - frame * dt) < 1e-8);
      const nb = solver.narrowBandFlipInfo;
      history[i] = { tiers: await tiersOf(solver) };harvest(i);
      shown[i] = { particles: nb?.particles, fine: solver.info.uniformMixedFineTiles, band: solver.info.uniformPressureBandTiles, bandResidual: solver.info.uniformPressureBandResidual, pool: { ...solver.pagePoolInfo, frame: undefined } };
    }
    // The tiers, read at the frame's end, compare by frame; the receipts by
    // the frame they describe, once both arms hold it.
    const differs = (b: unknown, c: unknown) => { if (keepGoing || !identity) historyDiffs.push({ frame, baseline: b, candidate: c }); else assert.fail(`frame ${frame}: work histories differ: ${JSON.stringify(b)} vs ${JSON.stringify(c)}`); };
    if (JSON.stringify(history[1]) !== JSON.stringify(history[0])) differs(history[0], history[1]);
    for (const [at, b] of receipts[0]!) { const c = receipts[1]!.get(at); if (c === undefined) continue; if (c !== b) differs({ receipt: at, ...JSON.parse(b) }, { receipt: at, ...JSON.parse(c) }); receipts[0]!.delete(at); receipts[1]!.delete(at); compared.add(at); }
    // The band residual is a float reduction whose order differs between two
    // instances of one source (an A/A pair differs at it): counted apart.
    const residuals = shown.map(h => (h as { bandResidual?: number }).bandResidual ?? 0);
    if (residuals[0] !== residuals[1]) residualDiffers++;
    rows.push({ frame, baseline_ms: wall[0]!, candidate_ms: wall[1]!, saved_ms: wall[0]! - wall[1]!, first: frame % 2 ? "baseline" : "candidate", history: [{ ...(history[0] as object), shown: shown[0] }], profile: commands ? profiles : undefined });
    if (hashAt.has(frame)) { const h = await hashFields(); const differ = Object.keys(h[0]!).filter(k => h[0]![k] !== h[1]![k]); frameHashes.push({ frame, match: differ.length === 0, differ }); console.log(JSON.stringify({ phase: "hashes", frame, differ })); }
    if (frame % 20 === 0 || frame === steps) console.log(JSON.stringify({ ...rows.at(-1), profile: undefined, totals: commands ? profiles.map(total) : undefined }));
  }
  const fieldHashes: Record<string, string>[] = [];
  for (const [i, solver] of solvers.entries()) {
    const address = await helpers[i]!.readUniformLiveAddress(raw, solver), out: Record<string, string> = {};
    for (const [key, data] of Object.entries(await fieldData(i, solver, address))) {
      out[key] = createHash("sha256").update(new Uint8Array(data.buffer, data.byteOffset, data.byteLength)).digest("hex");
    }
    fieldHashes.push(out);
  }
  assert.deepEqual(errors, []);
  const measured = rows.filter(r => r.frame >= from), mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / xs.length;
  const median = (xs: number[]) => { const s = [...xs].sort((a, b) => a - b), m = Math.floor(s.length / 2); return s.length % 2 ? s[m]! : (s[m - 1]! + s[m]!) / 2; };
  const summary = {
    frames: measured.length, fieldsMatch: JSON.stringify(fieldHashes[0]) === JSON.stringify(fieldHashes[1]),
    baseline_ms: mean(measured.map(r => r.baseline_ms)), candidate_ms: mean(measured.map(r => r.candidate_ms)),
    baseline_median_ms: median(measured.map(r => r.baseline_ms)), candidate_median_ms: median(measured.map(r => r.candidate_ms)),
    paired_median_saved_ms: median(measured.map(r => r.saved_ms)),
    residualDiffers, frameHashes, historyDiffs: historyDiffs.length, firstHistoryDiffs: historyDiffs.slice(0, 4),
    commands: commands ? [0, 1].map(i => { const t = measured.map(r => total(r.profile![i]!)); return Object.fromEntries(Object.keys(t[0]!).map(k => [k, mean(t.map(x => x[k as keyof typeof x]))])); }) : undefined,
  };
  writeFileSync(`${out}/${name}.json`, JSON.stringify({ date: new Date().toISOString(), cpu: cpus()[0]?.model, node: process.version, scene: scene?.sceneId, poolBlocks, steps, from,
    scope: "alternating fenced simulation advances on one device; excludes construction, renderer and diagnostic reads", snapshots, summary, fieldHashes, rows, errors }, null, 2) + "\n");
  console.log(JSON.stringify({ phase: "complete", summary, fieldHashes }));
  if (identity) {
  assert.ok(summary.fieldsMatch, "final fields differ");
  assert.ok(frameHashes.every(f => f.match), `fields differ at frame ${frameHashes.find(f => !f.match)?.frame}`);
  assert.equal(historyDiffs.length, 0, `work histories differ at ${historyDiffs.length} frames, first ${historyDiffs[0]?.frame}`);
  }
} catch (error) {
  writeFileSync(`${out}/${name}-failed.json`, JSON.stringify({ snapshots, rows, errors, error: String(error) }, null, 2) + "\n"); throw error;
} finally {
  lock?.stdin?.end();
  commands?.destroy(); read.buffer?.destroy();
  for (const solver of solvers) solver.destroy(); await raw?.queue.onSubmittedWorkDone(); raw?.destroy();
}
