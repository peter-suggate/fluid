/** The composite pressure solve's cycles, residuals and GPU time a frame, on a
 * UI scene or a region-tier fixture, with the final physics the tail gates
 * read: max speed, volume drift and (reservoir seep) the volume past the wall.
 *   node --import tsx tools/probe-pressure-tail.ts NAME --scene=reservoir-seep [--cell=0.005]
 *   node --import tsx tools/probe-pressure-tail.ts NAME --tier=rest [--length=128] [--depth=32]
 *   [--steps=120] [--from=13] [--values={...}] [--pool-blocks=N] [--no-profile] [--out=DIR] [--gate="RUNNER MODE"]
 * --gate: construct first (mostly CPU: pipeline compiles), then hold a slot
 * of the shared GPU runner only over the frames: the gate command runs `cat`
 * once the slot is held, and the slot is released when its stdin closes.
 * Pass times are timestamped passes (GPUPassProfile): the solve pass is the
 * V-cycles, its start and final measure; ms a cycle divides it by the
 * frame's completed cycles. */
import assert from "node:assert/strict";
import { mkdirSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { spawn, type ChildProcess } from "node:child_process";
import { resolveMethodValues, type MethodParamValues } from "../lib/core/method-contract";
import { sceneDocument, sceneDocumentAtLattice } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import type { SceneDescription } from "../lib/core/model";
import { managedGPUDevice } from "../lib/core/gpu-compilation-manager";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { createProcessRetainedDawnGPU, paceDawnEventPolling } from "../lib/harness/node-dawn-provider";
import { uniformNarrowBandMethod as method } from "../lib/methods/uniform/uniform-narrow-band-method";
import type { WebGPUUniformReferenceSolver } from "../lib/methods/uniform/webgpu-uniform-reference";
import { readUniformLiveAddress } from "../tests/helpers/uniform-geometric";
import { readMixedTexture } from "../tests/helpers/uniform-mixed-native-fields";
import { GPUPassProfile } from "./gpu-pass-profile";
import { createRegionTierScene, REGION_TIER_SCENES, type RegionTierScene } from "./region-tier-scenes";

const option = (key: string) => process.argv.find(a => a.startsWith(`--${key}=`))?.slice(key.length + 3);
const name = process.argv[2] ?? "probe";
assert.match(name, /^[a-z0-9.-]+$/);
const steps = Number(option("steps") ?? 120), from = Number(option("from") ?? 13), profiled = !process.argv.includes("--no-profile");
const extra: MethodParamValues = JSON.parse(option("values") ?? "{}");
let scene: SceneDescription, overrides: MethodParamValues = {}, quality = "balanced";
const tier = option("tier") as RegionTierScene | undefined;
if (tier) {
  assert.ok(REGION_TIER_SCENES.includes(tier));
  scene = createRegionTierScene(tier, Number(option("length") ?? 128), Number(option("depth") ?? 32));
} else {
  const definition = getSceneDefinition(option("scene") ?? "reservoir-seep");
  overrides = definition.methodProfile?.overrides ?? {}; quality = definition.methodProfile?.quality ?? quality;
  const cell = option("cell");
  scene = structuredClone(cell ? sceneDocumentAtLattice(definition, { cellSize_m: Number(cell) }).scene : sceneDocument(definition));
}
const poolBlocks = option("pool-blocks");
if (poolBlocks !== undefined) scene.fluid = { ...scene.fluid, pagePool: { ...scene.fluid?.pagePool, blocks: Number(poolBlocks) } } as SceneDescription["fluid"];
scene.duration_s = Math.max(scene.duration_s, steps * scene.numerics.fixedDt_s);
const values = resolveMethodValues(method, quality as never, { ...method.appDefaults, timeStep: "scene", ...overrides, ...extra });
const dt = scene.numerics.fixedDt_s;
paceDawnEventPolling();
const out = option("out") ?? "artifacts/pressure-tail";
mkdirSync(out, { recursive: true });
const quantile = (v: number[], q: number) => { const s = [...v].sort((a, b) => a - b); return s.length ? s[Math.min(s.length - 1, Math.floor(q * s.length))]! : NaN; };
const mean = (v: number[]) => v.reduce((a, b) => a + b, 0) / Math.max(1, v.length);
async function acquireGate(gate: string): Promise<ChildProcess> {
  const [command, ...args] = gate.split(" ").filter(Boolean);
  const child = spawn(command!, [...args, "cat"], { stdio: ["pipe", "pipe", "inherit"] });
  await new Promise<void>((done, fail) => { child.stdout!.once("data", () => done()); child.once("exit", code => fail(new Error(`gate exited ${code}`))); child.stdin!.write("held\n"); });
  return child;
}
let lock: ChildProcess | undefined;
let device: GPUDevice | undefined, profile: GPUPassProfile | undefined, solver: WebGPUUniformReferenceSolver | undefined;
const rows: { frame: number; wall_ms: number; cycles: number; residual: number; bandTiles: number; solve_ms?: number; prepare_ms?: number; frame_gpu_ms?: number }[] = [];
const errors: string[] = [];
try {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE ?? resolve("node_modules/webgpu/index.js")).href);
  Object.assign(globalThis, dawn.globals);
  const adapter = await createProcessRetainedDawnGPU(dawn, ["backend=metal"]).requestAdapter(); assert.ok(adapter);
  const raw = await adapter.requestDevice({ requiredFeatures: ["timestamp-query", ...(["subgroups", "float32-filterable"] as const).filter(f => adapter.features.has(f))],
    requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
  if (profiled) profile = new GPUPassProfile(raw);
  device = managedGPUDevice(profile?.device ?? raw, { requireWorkerRealm: false });
  device.addEventListener("uncapturederror", e => { e.preventDefault(); errors.push(e.error.message); });
  const built = performance.now();
  solver = await method.createSolverAsync!(device, scene, "balanced", values, undefined, () => {}) as WebGPUUniformReferenceSolver;
  console.log(JSON.stringify({ phase: "constructed", ms: performance.now() - built, grid: [solver.info.nx, solver.info.ny, solver.info.nz], dt }));
  const gate = option("gate");
  if (gate) { const t = performance.now(); lock = await acquireGate(gate); console.log(JSON.stringify({ phase: "locked", wait_ms: performance.now() - t })); }
  for (let frame = 1; frame <= steps; frame++) {
    if (profile && frame >= from) profile.start();
    const start = performance.now();
    assert.ok(solver.advanceTo(frame * dt, []));
    await solver.awaitFrameCompletion();
    const wall_ms = performance.now() - start;
    assert.equal(solver.info.simulationPipelineError, undefined);
    const passes = profile && frame >= from ? await profile.finish() as { label: string; ms: number }[] : undefined;
    const sum = (label: string) => passes?.filter(p => p.label === label).reduce((n, p) => n + p.ms, 0);
    rows.push({ frame, wall_ms, cycles: Number(solver.info.uniformPressureBandCycles ?? 0), residual: Number(solver.info.uniformPressureBandResidual ?? NaN),
      bandTiles: Number(solver.info.uniformPressureBandTiles ?? 0), solve_ms: sum("Uniform pressure arena solve"), prepare_ms: sum("Uniform pressure arena marks and rows"),
      frame_gpu_ms: passes?.reduce((n, p) => n + p.ms, 0) });
    if (frame % 30 === 0 || frame === steps) console.log(JSON.stringify(rows[rows.length - 1]));
  }
  const measured = rows.filter(r => r.frame >= from), stats = await solver.readStats();
  let pastWall: number | undefined;
  if (scene.sceneId === "reservoir-seep") {
    const { nx } = solver.info, high = 16 * Math.floor(nx / 32);
    const volume = await readMixedTexture(device, solver.volumeTexture, await readUniformLiveAddress(device, solver));
    pastWall = 0; for (let i = 0; i < volume.length; i++) if (i % nx >= high) pastWall += volume[i]!;
  }
  const solve = measured.map(r => r.solve_ms ?? NaN), cycles = measured.map(r => r.cycles), residual = measured.map(r => r.residual);
  const summary = { frames: measured.length, cycles: { mean: mean(cycles), p50: quantile(cycles, .5), p90: quantile(cycles, .9), max: Math.max(...cycles) },
    residual: { p50: quantile(residual, .5), p90: quantile(residual, .9), max: Math.max(...residual) },
    solve_ms: { mean: mean(solve), p50: quantile(solve, .5) }, msPerCycle: mean(measured.map(r => (r.solve_ms ?? NaN) / Math.max(1, r.cycles))),
    prepare_ms: mean(measured.map(r => r.prepare_ms ?? NaN)), frame_gpu_ms: mean(measured.map(r => r.frame_gpu_ms ?? NaN)), wall_ms: mean(measured.map(r => r.wall_ms)),
    maxSpeed_m_s: stats.maxSpeed_m_s, volumeDrift: stats.volumeDrift, pastWall_cells: pastWall };
  assert.deepEqual(errors, []);
  writeFileSync(`${out}/${name}.json`, JSON.stringify({ name, scene: scene.sceneId, values: extra, steps, from, summary, rows }, null, 2) + "\n");
  console.log(JSON.stringify({ phase: "complete", name, summary }));
} finally {
  lock?.stdin?.end();
  solver?.destroy(); await device?.queue.onSubmittedWorkDone(); profile?.destroy(); device?.destroy();
}
