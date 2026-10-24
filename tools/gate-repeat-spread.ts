/** The physics gate of a change that need not be bit-identical: two source
 * snapshots, each run `repeats` times on one device (atomic slot allocation
 * differs between instances, so two repeats of one source already differ),
 * on a region-tier fixture (tools/region-tier-scenes.ts) or a registered
 * scene. Each frame advances every instance, rotating which goes first.
 *
 * Recorded per frame, aligned by the frame each value describes:
 *   receipts (the mixed frame's lagged plans, by the frame they describe,
 *   not the newest one solver.info shows): h tiles, band tiles, band bricks;
 *   read at the frame's end: particles (the FLIP receipt word on the GPU);
 *   every --stats-every frames and the last: volume drift (readStats) and
 *   occupancy (liquid vertices, phi < 0, read by live page identity).
 *
 * For each value and frame: the A/A spread (max - min over the baseline's
 * repeats) and the A/B deviation (max |a - b| over baseline and candidate
 * repeats). It passes when every A/B deviation is within the tolerance:
 * the largest A/A or B/B spread of that value within +-5 frames, or the
 * repeat-noise band (relative to the baseline mean; volume drift, a ratio,
 * absolute): 0.2% to frame 60, 2% after (speed-of-light-plan-2026-10-20.md
 * section 6). A fatal or a GPU error fails the run.
 *
 * node --import tsx tools/gate-repeat-spread.ts NAME
 *   --baseline=/abs/snapshot --candidate=/abs/snapshot
 *   [--scene=rest|cold|warm|splash|hot|<scene id>] [--length=128] [--depth=32]
 *   [--pool-blocks=N] [--steps=70] [--repeats=2] [--stats-every=10] [--out=DIR] [--gate="RUNNER MODE"]
 * --gate: construct every instance (mostly CPU: pipeline compiles) first,
 * then hold a GPU runner slot only over the frames: the gate command runs
 * `cat` once the slot is held, and the slot is released when its stdin
 * closes (on any exit).
 */
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdirSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
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
for (const a of process.argv.slice(3)) assert.match(a, /^--(?:baseline|candidate|steps|repeats|stats-every|scene|length|depth|pool-blocks|out|gate)=.+$/);
const steps = Number(option("steps") ?? 70), repeats = Number(option("repeats") ?? 2), statsEvery = Number(option("stats-every") ?? 10);
assert.ok(Number.isInteger(steps) && steps > 0 && Number.isInteger(repeats) && repeats >= 2 && Number.isInteger(statsEvery) && statsEvery > 0);
const roots = [option("baseline"), option("candidate")].map(r => resolve(r!)); assert.ok(roots.every(Boolean));
const sceneOption = option("scene") ?? "splash", region = (REGION_TIER_SCENES as readonly string[]).includes(sceneOption);
const length = Number(option("length") ?? 128), depth = Number(option("depth") ?? 32);
const poolBlocks = option("pool-blocks") === undefined ? undefined : Number(option("pool-blocks"));
assert.ok(poolBlocks === undefined || (Number.isInteger(poolBlocks) && poolBlocks >= 3));
const out = option("out") ?? "artifacts/region-tiers/gate"; mkdirSync(out, { recursive: true });
paceDawnEventPolling();

let lock: ChildProcess | undefined;
/** Hold the gate's slot until the returned process's stdin closes. */
async function acquireGate(gate: string): Promise<ChildProcess> {
  const [command, ...args] = gate.split(" ").filter(Boolean);
  const child = spawn(command!, [...args, "cat"], { stdio: ["pipe", "pipe", "inherit"] });
  await new Promise<void>((done, fail) => { child.stdout!.once("data", () => done()); child.once("exit", code => fail(new Error(`gate exited ${code}`))); child.stdin!.write("held\n"); });
  return child;
}

type Lagged = { frame: number; bandTiles: number; bandBricks: number; work: Uint32Array };
type Internals = { mixedFrame: { lagged: readonly Lagged[]; narrowBandFlip: { state: GPUBuffer } } };
interface Instance { arm: 0 | 1; solver: WebGPUUniformReferenceSolver; readMixedTexture: (d: GPUDevice, t: GPUTexture, a?: unknown) => Promise<Float32Array>; readUniformLiveAddress: (d: GPUDevice, s: WebGPUUniformReferenceSolver) => Promise<unknown> }
const METRICS = ["fineTiles", "bandTiles", "bandBricks", "particles", "volumeDrift", "occupancy"] as const;
type Metric = typeof METRICS[number];
/** series[metric][instance]: frame -> value. */
const series = Object.fromEntries(METRICS.map(m => [m, [] as Map<number, number>[]])) as Record<Metric, Map<number, number>[]>;
const instances: Instance[] = [], snapshots: Record<string, unknown>[] = [], errors: string[] = [];
let raw: GPUDevice | undefined;
const read = { buffer: undefined as GPUBuffer | undefined };
try {
  const dawn = await import(pathToFileURL(resolve("node_modules/webgpu/index.js")).href);
  Object.assign(globalThis, dawn.globals);
  const adapter = await createProcessRetainedDawnGPU(dawn, ["backend=metal"]).requestAdapter(); assert.ok(adapter);
  const features = (["timestamp-query", "subgroups", "float32-filterable"] as GPUFeatureName[]).filter(f => adapter.features.has(f));
  raw = await adapter.requestDevice({ requiredFeatures: features, requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
  raw.addEventListener("uncapturederror", event => { event.preventDefault(); errors.push(event.error.message); });
  read.buffer = raw.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  let dt = 0, scene: SceneDescription | undefined, values: unknown;
  for (const [arm, cwd] of roots.entries()) {
    const load = (file: string) => import(pathToFileURL(resolve(cwd, file)).href);
    const [{ uniformNarrowBandMethod: method }, { resolveMethodValues }, { sceneDocument }, { getSceneDefinition }, { managedGPUDevice }, fields, geometric] = await Promise.all([
      load("lib/methods/uniform/uniform-narrow-band-method.ts"), load("lib/core/method-contract.ts"), load("lib/core/scene-definition.ts"), load("lib/core/scenes.ts"), load("lib/core/gpu-compilation-manager.ts"),
      load("tests/helpers/uniform-mixed-native-fields.ts"), load("tests/helpers/uniform-geometric.ts"),
    ]);
    let armValues: unknown, quality: string, built: SceneDescription;
    if (region) {
      built = createRegionTierScene(sceneOption as RegionTierScene, length, depth); quality = "balanced";
      armValues = resolveMethodValues(method, quality, { ...method.appDefaults, timeStep: "scene" });
    } else {
      const definition = getSceneDefinition(sceneOption); built = structuredClone(sceneDocument(definition));
      quality = definition.methodProfile.quality; armValues = resolveMethodValues(method, quality, { ...method.appDefaults, ...definition.methodProfile.overrides });
    }
    if (poolBlocks !== undefined) built.fluid = { ...built.fluid, pagePool: { ...built.fluid?.pagePool, blocks: poolBlocks } } as SceneDescription["fluid"];
    const step = method.effectiveStep_s?.(built, armValues) ?? built.numerics.fixedDt_s;
    built.duration_s = Math.max(built.duration_s, steps * step);
    if (arm) { assert.equal(step, dt); assert.deepEqual(armValues, values); assert.deepEqual(built, scene); }
    scene = built; dt = step; values = armValues;
    const hash = createHash("sha256"), directory = resolve(cwd, "lib/methods/uniform");
    for (const f of readdirSync(directory).filter(f => f.endsWith(".ts")).sort()) { hash.update(f); hash.update(readFileSync(resolve(directory, f))); }
    snapshots.push({ root: cwd, sourceHash: hash.digest("hex"), quality, dt });
    for (let r = 0; r < repeats; r++) {
      const start = performance.now();
      const solver = await method.createSolverAsync!(managedGPUDevice(raw, { requireWorkerRealm: false }), built, quality, armValues, undefined, () => {}) as WebGPUUniformReferenceSolver;
      await raw.queue.onSubmittedWorkDone();
      instances.push({ arm: arm as 0 | 1, solver, readMixedTexture: fields.readMixedTexture, readUniformLiveAddress: geometric.readUniformLiveAddress });
      console.log(JSON.stringify({ phase: "constructed", arm: arm ? "candidate" : "baseline", repeat: r, construction_ms: performance.now() - start }));
    }
  }
  for (const m of METRICS) series[m] = instances.map(() => new Map());
  const gate = option("gate");
  if (gate) { const t = performance.now(); lock = await acquireGate(gate); console.log(JSON.stringify({ phase: "locked", gate, wait_ms: performance.now() - t })); }
  const particlesOf = async (solver: WebGPUUniformReferenceSolver) => {
    const encoder = raw!.createCommandEncoder(); encoder.copyBufferToBuffer((solver as unknown as Internals).mixedFrame.narrowBandFlip.state, 0, read.buffer!, 0, 16);
    raw!.queue.submit([encoder.finish()]); await read.buffer!.mapAsync(GPUMapMode.READ);
    const count = new Uint32Array(read.buffer!.getMappedRange())[1]!; read.buffer!.unmap(); return count;
  };
  for (let frame = 1; frame <= steps; frame++) {
    for (let k = 0; k < instances.length; k++) {
      const i = (k + frame) % instances.length, { solver } = instances[i]!;
      assert.ok(solver.advanceTo(frame * dt, []), `instance ${i}: frame ${frame} not accepted`); await solver.awaitFrameCompletion();
      assert.equal(solver.info.simulationPipelineError, undefined, `instance ${i}: frame ${frame}`);
      assert.ok(Math.abs((solver.info.completedTime_s ?? 0) - frame * dt) < 1e-8);
      for (const p of (solver as unknown as Internals).mixedFrame.lagged) if (p.frame >= 0) {
        series.fineTiles[i]!.set(p.frame, p.work[0]!); series.bandTiles[i]!.set(p.frame, p.bandTiles); series.bandBricks[i]!.set(p.frame, p.bandBricks);
      }
      series.particles[i]!.set(frame, await particlesOf(solver));
      if (frame % statsEvery === 0 || frame === steps) {
        series.volumeDrift[i]!.set(frame, (await solver.readStats()).volumeDrift ?? NaN);
        const phi = await instances[i]!.readMixedTexture(raw, solver.vertexPhiTexture!, await instances[i]!.readUniformLiveAddress(raw, solver));
        let liquid = 0; for (const v of phi) if (v < 0) liquid++;
        series.occupancy[i]!.set(frame, liquid);
      }
    }
    assert.deepEqual(errors, [], `frame ${frame}: GPU errors`);
    if (frame % 10 === 0 || frame === steps) console.log(JSON.stringify({ phase: "frame", frame, particles: instances.map((_, i) => series.particles[i]!.get(frame)), occupancy: instances.map((_, i) => series.occupancy[i]!.get(frame)) }));
  }
  lock?.stdin?.end(); lock = undefined;
  // The spreads and the verdict.
  const arms = [0, 1].map(arm => instances.flatMap((x, i) => x.arm === arm ? [i] : []));
  const band = (frame: number) => frame <= 60 ? 0.002 : 0.02;
  const report: Record<string, unknown> = {}; let pass = true;
  for (const m of METRICS) {
    const frames = [...series[m][0]!.keys()].filter(f => instances.every((_, i) => series[m][i]!.has(f))).sort((a, b) => a - b);
    const at = (i: number, f: number) => series[m][i]!.get(f)!;
    const spread = (arm: number[], f: number) => { const v = arm.map(i => at(i, f)); return Math.max(...v) - Math.min(...v); };
    const rows = frames.map(f => {
      let ab = 0; for (const a of arms[0]!) for (const b of arms[1]!) ab = Math.max(ab, Math.abs(at(a, f) - at(b, f)));
      const mean = arms[0]!.reduce((s, i) => s + at(i, f), 0) / arms[0]!.length;
      return { frame: f, aa: spread(arms[0]!, f), bb: spread(arms[1]!, f), ab, noise: m === "volumeDrift" ? band(f) : band(f) * Math.abs(mean) };
    });
    let worst = { frame: 0, ratio: 0, aa: 0, bb: 0, ab: 0, tolerance: 0 }, over = 0;
    for (const r of rows) {
      const near = rows.filter(q => Math.abs(q.frame - r.frame) <= 5), tolerance = Math.max(r.noise, ...near.map(q => Math.max(q.aa, q.bb)));
      const ratio = r.ab === 0 ? 0 : tolerance === 0 ? Infinity : r.ab / tolerance;
      if (ratio > 1) over++;
      if (ratio > worst.ratio) worst = { frame: r.frame, ratio, aa: r.aa, bb: r.bb, ab: r.ab, tolerance };
    }
    pass &&= over === 0;
    report[m] = { frames: rows.length, over, worst, maxAA: Math.max(0, ...rows.map(r => r.aa)), maxAB: Math.max(0, ...rows.map(r => r.ab)) };
  }
  const values_ = Object.fromEntries(METRICS.map(m => [m, instances.map((x, i) => ({ arm: x.arm, values: Object.fromEntries(series[m][i]!) }))]));
  writeFileSync(`${out}/${name}.json`, JSON.stringify({ date: new Date().toISOString(), scene: sceneOption, length, depth, poolBlocks, steps, repeats, statsEvery, snapshots, pass, report, series: values_ }, null, 2) + "\n");
  console.log(JSON.stringify({ phase: "complete", pass, report }));
  assert.ok(pass, "A/B deviates beyond the repeat spread");
} catch (error) {
  writeFileSync(`${out}/${name}-failed.json`, JSON.stringify({ snapshots, errors, error: String(error) }, null, 2) + "\n"); throw error;
} finally {
  lock?.stdin?.end();
  read.buffer?.destroy();
  for (const { solver } of instances) solver.destroy(); await raw?.queue.onSubmittedWorkDone(); raw?.destroy();
}
