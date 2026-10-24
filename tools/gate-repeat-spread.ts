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
 * The verdict is on 10-frame window means (frames 1-10, 11-20, ...), as
 * 60 frames of 6 values each against a spread from the repeats would flag
 * something by chance: every window's |mean A - mean B| within max(noise
 * band, 1.66 x the run's largest A/A or B/B window spread)
 * (gate-repeat-spread-verdict.ts holds the rule and its calibration). A
 * registered scene (seep, letters: its divergence grows through the run)
 * needs 3 repeats at least; a region-tier fixture may run 2. A fatal or a
 * GPU error fails the run.
 *
 * node --import tsx tools/gate-repeat-spread.ts NAME
 *   --baseline=/abs/snapshot --candidate=/abs/snapshot
 *   [--scene=rest|cold|warm|splash|hot|<scene id>] [--length=128] [--depth=32]
 *   [--pool-blocks=N] [--steps=70] [--repeats=3] [--stats-every=10] [--out=DIR] [--gate="RUNNER MODE"]
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
import { METRICS, repeatSpreadVerdict as verdict, type Metric } from "./gate-repeat-spread-verdict";

const option = (key: string) => process.argv.find(a => a.startsWith(`--${key}=`))?.slice(key.length + 3);
const name = process.argv[2]; assert.ok(name); assert.match(name, /^[a-z0-9-]+$/);
for (const a of process.argv.slice(3)) assert.match(a, /^--(?:baseline|candidate|steps|repeats|stats-every|scene|length|depth|pool-blocks|out|gate|rescore)=.+$/);
const steps = Number(option("steps") ?? 70), repeats = Number(option("repeats") ?? 3), statsEvery = Number(option("stats-every") ?? 10);
assert.ok(Number.isInteger(steps) && steps > 0 && Number.isInteger(repeats) && repeats >= 2 && Number.isInteger(statsEvery) && statsEvery > 0);
const roots = [option("baseline"), option("candidate")].map(r => resolve(r ?? "")); assert.ok(option("rescore") || (option("baseline") && option("candidate")));
const sceneOption = option("scene") ?? "splash", region = (REGION_TIER_SCENES as readonly string[]).includes(sceneOption);
assert.ok(region || repeats >= 3 || option("rescore"), `--scene=${sceneOption}: a registered scene needs --repeats=3 at least (gate-repeat-spread-verdict.ts)`);
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
/** series[metric][instance]: frame -> value. */
const series = Object.fromEntries(METRICS.map(m => [m, [] as Map<number, number>[]])) as Record<Metric, Map<number, number>[]>;
// --rescore=NAME.json: the verdict over a saved run's series, without a GPU.
const rescore = option("rescore");
if (rescore) {
  const saved = JSON.parse(readFileSync(rescore, "utf8")) as { series: Record<Metric, { arm: 0 | 1; values: Record<string, number> }[]> };
  const result = verdict(saved.series.fineTiles.map(x => x.arm), Object.fromEntries(METRICS.map(m => [m, saved.series[m].map(x => new Map(Object.entries(x.values).map(([f, v]) => [Number(f), v])))])) as Record<Metric, Map<number, number>[]>);
  console.log(JSON.stringify({ phase: "rescored", file: rescore, ...result }));
  process.exit(result.pass ? 0 : 1);
}
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
  let dt = 0, scene: SceneDescription | undefined, values: Record<string, unknown> | undefined, defaults: Record<string, unknown> = {};
  /** Both arms' values with the keys they share; a key only one arm has (a
   * param the other snapshot predates, such as bulkTier) must sit at that
   * arm's own default. */
  const shared = (mine: Record<string, unknown>, other: Record<string, unknown>, own: Record<string, unknown>) => Object.fromEntries(Object.entries(mine).filter(([key, value]) => {
    if (key in other) return true;
    assert.deepEqual(value, own[key], `${key} exists in one snapshot only and is not at its default`); return false;
  }));
  for (const [arm, cwd] of roots.entries()) {
    const load = (file: string) => import(pathToFileURL(resolve(cwd, file)).href);
    const [{ uniformNarrowBandMethod: method }, { resolveMethodValues }, { sceneDocument }, { getSceneDefinition }, { managedGPUDevice }, fields, geometric] = await Promise.all([
      load("lib/methods/uniform/uniform-narrow-band-method.ts"), load("lib/core/method-contract.ts"), load("lib/core/scene-definition.ts"), load("lib/core/scenes.ts"), load("lib/core/gpu-compilation-manager.ts"),
      load("tests/helpers/uniform-mixed-native-fields.ts"), load("tests/helpers/uniform-geometric.ts"),
    ]);
    let armValues: Record<string, unknown>, quality: string, built: SceneDescription;
    const armDefaults = Object.fromEntries((method.params as { key: string; default: unknown }[]).map(spec => [spec.key, spec.default]));
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
    if (arm) { assert.equal(step, dt); assert.deepEqual(shared(armValues, values!, armDefaults), shared(values!, armValues, defaults)); assert.deepEqual(built, scene); }
    scene = built; dt = step; values = armValues; defaults = armDefaults;
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
  const { pass, report } = verdict(instances.map(x => x.arm), series);
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
