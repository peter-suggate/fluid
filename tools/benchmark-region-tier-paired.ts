/** Alternate fenced simulation frames from two source snapshots on one device,
 * a region-tier fixture (tools/region-tier-scenes.ts) or a registered scene,
 * at an optional pool reservation. Each frame runs both arms, alternating
 * which goes first. Fails unless final fields (read by live page identity:
 * which request gets which block is a race) and every frame's work history
 * match. --profile counts each arm's passes, dispatches, workgroups, clears
 * and copies by pass label (a separate diagnostic run: it adds CPU work).
 * node --import tsx tools/benchmark-region-tier-paired.ts NAME
 *   --baseline=/abs/snapshot --candidate=/abs/snapshot
 *   [--scene=rest|cold|warm|splash|hot|<scene id>] [--length=128] [--depth=32]
 *   [--pool-blocks=N] [--steps=100] [--from=40] [--profile] [--out=DIR] [--gate="RUNNER MODE"]
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
for (const a of process.argv.slice(3)) assert.match(a, /^--(?:(?:baseline|candidate|steps|from|scene|length|depth|pool-blocks|out|gate)=.+|profile)$/);
const steps = Number(option("steps") ?? 100), from = Number(option("from") ?? 40), profile = process.argv.includes("--profile");
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

const errors: string[] = [], solvers: WebGPUUniformReferenceSolver[] = [], helpers: { readMixedTexture: (d: GPUDevice, t: GPUTexture, a?: unknown) => Promise<Float32Array>; readUniformLiveAddress: (d: GPUDevice, s: WebGPUUniformReferenceSolver) => Promise<unknown> }[] = [];
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
    helpers.push({ readMixedTexture: fields.readMixedTexture, readUniformLiveAddress: geometric.readUniformLiveAddress });
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
  let residualDiffers = 0;
  for (let frame = 1; frame <= steps; frame++) {
    const wall = [0, 0], history: unknown[] = [], profiles: Profile[] = [];
    for (const i of frame % 2 ? [0, 1] : [1, 0]) {
      const solver = solvers[i]!; commands?.begin();
      const start = performance.now();
      assert.ok(solver.advanceTo(frame * dt, [])); await solver.awaitFrameCompletion(); wall[i] = performance.now() - start;
      if (commands) profiles[i] = commands.end();
      assert.equal(solver.info.simulationPipelineError, undefined);
      assert.ok(Math.abs((solver.info.completedTime_s ?? 0) - frame * dt) < 1e-8);
      const nb = solver.narrowBandFlipInfo;
      history[i] = { tiers: await tiersOf(solver), particles: nb?.particles, fine: solver.info.uniformMixedFineTiles, band: solver.info.uniformPressureBandTiles, bandResidual: solver.info.uniformPressureBandResidual, pool: { ...solver.pagePoolInfo, frame: undefined } };
    }
    // The band residual is a float reduction whose order differs between two
    // instances of one source (an A/A pair differs at it): compared apart.
    const strip = (h: unknown) => ({ ...(h as Record<string, unknown>), bandResidual: undefined });
    assert.deepEqual(strip(history[1]), strip(history[0]), `frame ${frame}: work histories differ`);
    const residuals = history.map(h => (h as { bandResidual?: number }).bandResidual ?? 0);
    if (residuals[0] !== residuals[1]) residualDiffers++;
    rows.push({ frame, baseline_ms: wall[0]!, candidate_ms: wall[1]!, saved_ms: wall[0]! - wall[1]!, first: frame % 2 ? "baseline" : "candidate", history: [history[0]], profile: commands ? profiles : undefined });
    if (frame % 20 === 0 || frame === steps) console.log(JSON.stringify({ ...rows.at(-1), profile: undefined, totals: commands ? profiles.map(total) : undefined }));
  }
  const fieldHashes: Record<string, string>[] = [];
  for (const [i, solver] of solvers.entries()) {
    const address = await helpers[i]!.readUniformLiveAddress(raw, solver), out: Record<string, string> = {};
    for (const [key, texture] of Object.entries({ velocity: solver.velocityTexture, volume: solver.volumeTexture, phi: solver.vertexPhiTexture!, pressure: solver.gridPressureTexture })) {
      const data = await helpers[i]!.readMixedTexture(raw, texture, address);
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
    residualDiffers,
    commands: commands ? [0, 1].map(i => { const t = measured.map(r => total(r.profile![i]!)); return Object.fromEntries(Object.keys(t[0]!).map(k => [k, mean(t.map(x => x[k as keyof typeof x]))])); }) : undefined,
  };
  writeFileSync(`${out}/${name}.json`, JSON.stringify({ date: new Date().toISOString(), cpu: cpus()[0]?.model, node: process.version, scene: scene?.sceneId, poolBlocks, steps, from,
    scope: "alternating fenced simulation advances on one device; excludes construction, renderer and diagnostic reads", snapshots, summary, fieldHashes, rows, errors }, null, 2) + "\n");
  console.log(JSON.stringify({ phase: "complete", summary, fieldHashes }));
  assert.ok(summary.fieldsMatch, "final fields differ");
} catch (error) {
  writeFileSync(`${out}/${name}-failed.json`, JSON.stringify({ snapshots, rows, errors, error: String(error) }, null, 2) + "\n"); throw error;
} finally {
  lock?.stdin?.end();
  commands?.destroy(); read.buffer?.destroy();
  for (const solver of solvers) solver.destroy(); await raw?.queue.onSubmittedWorkDone(); raw?.destroy();
}
