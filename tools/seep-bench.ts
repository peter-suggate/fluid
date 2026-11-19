/** The seep rewrite's fair benchmark and observation harness
 * (docs/plans/seep-rewrite-2026-11-14.md §3, §5).
 *
 * node --import tsx tools/seep-bench.ts --method=nbflip|seep [--steps=1200] [--every=60]
 *   [--out=artifacts/seep-rewrite/runs] [--name=NAME] [--mode=fenced|pipelined] [--from=13] [--nb-stats]
 *   [--slot=check|timing]   (the fluid-gpu-run.sh mode; seep-compare takes speed from timing runs only)
 *   [--engine=auto|brick|uniform]   (nbflip only: the engine option; the default is the app's, auto)
 *   [--prof=FROM:COUNT]   (attribution only: the kernel timer, tools/gpu-kernel-timer.ts, over steps FROM..FROM+COUNT-1;
 *                          it splits every dispatch into its own pass, so the run is no step time: check slot only)
 *   [--audit=0,120,300,600] (fenced only: at these sample steps also dump every owner at x >= wall.low and log
 *                            an `audit` row: wall zone split open/solid, jet slices; tools/seep-audit-cells.ts)
 *
 * Both solvers are built through the method registry (lib/methods, getMethod)
 * on the same sceneDocument("reservoir-seep"), on the same Dawn device setup
 * (Metal backend, subgroups and float32-filterable where offered, the fluid
 * device limits, a requested-bytes memory profile), and advanced by the same
 * loop. NB-FLIP takes the scene's methodProfile over the method's app
 * defaults (as tools/probe-tier-budget.ts and the app); the Seep solver its
 * own defaults.
 *
 * fenced: each step is advanceTo + awaitFrameCompletion, timed on the wall
 * clock with no readback inside. Every `every` steps (and at 0 and the last
 * step) the observer measures the fenced state (tools/seep-observables.ts);
 * the sample step and the step after it are excluded from the timing
 * statistics and reported apart, as are the warm-up steps before --from.
 * --nb-stats also reads NB-FLIP's own volume (readStats) at each sample; that
 * asks the next step for NB-FLIP's diagnostic volume pass, which lands in the
 * excluded step after the sample.
 * pipelined: every step is submitted as soon as the solver admits it (the
 * app's loop; a full receipt ring yields to Dawn's event loop and retries),
 * one fence at the end; throughput = total wall / steps. Observed at 0 and at
 * the end only.
 *
 * Writes OUT/NAME.jsonl (meta, constructed, samples, summary) incrementally and
 * OUT/NAME-summary.json. Run under tools/gpu/fluid-gpu-run.sh (check for
 * observables, timing for speed). */
import assert from "node:assert/strict";
import { appendFileSync, mkdirSync, writeFileSync } from "node:fs";
import { execSync } from "node:child_process";
import { cpus, platform, release } from "node:os";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import "../lib/methods";
import { getMethod, registeredSimulationMethods } from "../lib/core/method-registry";
import { resolveMethodValues, type GPUSolverInstance } from "../lib/core/method-contract";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { managedGPUDevice, whenGPUCompilationIdle } from "../lib/core/gpu-compilation-manager";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { createProcessRetainedDawnGPU, paceDawnEventPolling } from "../lib/harness/node-dawn-provider";
import type { SeepObservationSource } from "../lib/methods/seep/seep-observation";
import { GPUMemoryProfile } from "./gpu-memory-profile";
import { gpuKernelTimer } from "./gpu-kernel-timer";
import { seepGeometry, seepObservation, type SeepObservation } from "./seep-observables";
import { createNBFlipZoneObserver, type ZoneObserver } from "./seep-observe-nbflip";
import { createSeepZoneObserver } from "./seep-observe-seep";
import { createBrickZoneObserver } from "./seep-observe-brick";
import type { BrickNBFlipSolver } from "../lib/methods/narrow-band/brick-solver";
import { auditCells, createNBFlipCellDump, createSeepCellDump, type CellDump } from "./seep-audit-cells";

export const SEEP_BENCH_METHODS = { nbflip: "uniform-narrow-band-flip", seep: "seep-m1max" } as const;
const SCENE_ID = "reservoir-seep";

const args = process.argv.slice(2);
for (const a of args) assert.ok(/^(--(?:method|steps|every|out|name|mode|from|slot|audit|engine|prof)=.+|--nb-stats)$/.test(a), `Unknown option: ${a}`);
const option = (key: string) => args.find(a => a.startsWith(`--${key}=`))?.slice(key.length + 3);
const methodKey = option("method") as keyof typeof SEEP_BENCH_METHODS | undefined;
assert.ok(methodKey && methodKey in SEEP_BENCH_METHODS, "--method=nbflip|seep");
const mode = option("mode") ?? "fenced";
assert.ok(mode === "fenced" || mode === "pipelined", "--mode=fenced|pipelined");
const steps = Number(option("steps") ?? 1200), every = Number(option("every") ?? 60), from = Number(option("from") ?? 13);
assert.ok(Number.isInteger(steps) && steps > 0 && steps <= 1200, "--steps in 1..1200 (the scene's 20 s take)");
assert.ok(Number.isInteger(every) && every > 0 && Number.isInteger(from) && from >= 1);
const nbStats = args.includes("--nb-stats");
/** The fluid-gpu-run.sh slot the run holds: only `timing` (exclusive GPU) runs count for §5 speed. */
const slot = option("slot") ?? "check";
assert.ok(slot === "check" || slot === "timing", "--slot=check|timing (the fluid-gpu-run.sh mode this run holds)");
const auditSteps = new Set((option("audit") ?? "").split(",").filter(Boolean).map(Number));
for (const a of auditSteps) assert.ok(Number.isInteger(a) && a >= 0 && a <= steps && (a % every === 0 || a === steps), `--audit step ${a} must be a sample step`);
assert.ok(!auditSteps.size || mode === "fenced", "--audit needs --mode=fenced");
const profWindow = option("prof")?.split(":").map(Number);
assert.ok(!profWindow || (profWindow.length === 2 && profWindow.every(v => Number.isInteger(v) && v >= 1) && profWindow[0]! >= 2 && profWindow[0]! + profWindow[1]! - 1 <= steps),
  "--prof=FROM:COUNT, a window of steps inside the run");
assert.ok(!profWindow || (mode === "fenced" && slot === "check"), "--prof splits every dispatch into its own pass: a fenced check-slot run, never a step time");
const out = resolve(option("out") ?? "artifacts/seep-rewrite/runs");
const name = option("name") ?? `${methodKey}-${mode}-${steps}`;
assert.match(name, /^[A-Za-z0-9._-]+$/);
mkdirSync(out, { recursive: true });
const jsonl = `${out}/${name}.jsonl`;
writeFileSync(jsonl, "");
const log = (row: Record<string, unknown>) => { const text = JSON.stringify(row); appendFileSync(jsonl, text + "\n"); console.log(text.length > 600 ? text.slice(0, 600) + "…" : text); };

const methodId = SEEP_BENCH_METHODS[methodKey];
const method = getMethod(methodId);
if (method.id !== methodId) throw new Error(`Method ${methodId} is not registered (registered: ${registeredSimulationMethods().map(m => m.id).join(", ")})`);
const definition = getSceneDefinition(SCENE_ID);
const scene = sceneDocument(definition);
const profile = definition.methodProfile;
const quality = profile?.methodId === methodId ? profile.quality : "balanced";
const engine = option("engine");
assert.ok(!engine || (methodKey === "nbflip" && ["auto", "brick", "uniform"].includes(engine)), "--engine=auto|brick|uniform (nbflip only)");
const values = resolveMethodValues(method, quality, { ...method.appDefaults, ...(profile?.methodId === methodId ? profile.overrides : {}), ...(engine ? { engine } : {}) });
const dt = method.effectiveStep_s?.(scene, values) ?? scene.numerics.fixedDt_s;
assert.ok(steps * dt <= scene.duration_s + 1e-9, "Stay within the authored take");
const geometry = seepGeometry(scene);
let commit = "unknown"; try { commit = execSync("git rev-parse --short HEAD", { encoding: "utf8" }).trim(); } catch { /* not a checkout */ }
let dirty = false; try { dirty = execSync("git status --porcelain -- lib tools", { encoding: "utf8" }).trim().length > 0; } catch { /* not a checkout */ }
log({ kind: "meta", name, method: methodId, methodKey, mode, slot, steps, every, from, nbStats, prof: profWindow, scene: SCENE_ID, quality, dt, values, geometry,
  date: new Date().toISOString(), host: { cpu: cpus()[0]?.model, platform: platform(), release: release(), node: process.version, commit, dirtyLibOrTools: dirty } });

paceDawnEventPolling();
const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE ?? resolve("node_modules/webgpu/index.js")).href);
Object.assign(globalThis, dawn.globals);
const adapter = await createProcessRetainedDawnGPU(dawn, ["backend=metal"]).requestAdapter();
assert.ok(adapter, "No WebGPU adapter");
const raw = await adapter.requestDevice({ requiredFeatures: (["subgroups", "float32-filterable", ...(profWindow ? ["timestamp-query"] as const : [])] as const).filter(f => adapter.features.has(f)), requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
const memory = new GPUMemoryProfile(raw);
assert.ok(!profWindow || raw.features.has("timestamp-query"), "--prof needs the adapter's timestamp-query feature");
const timer = profWindow ? gpuKernelTimer(memory.device) : undefined;
const device = managedGPUDevice(timer?.device ?? memory.device, { requireWorkerRealm: false });
const errors: string[] = [];
device.addEventListener("uncapturederror", e => { e.preventDefault(); errors.push(e.error.message); console.error(e.error.message); });

const median = (v: number[]) => { const s = [...v].sort((a, b) => a - b); return s.length ? s.length % 2 ? s[s.length >> 1]! : (s[s.length / 2 - 1]! + s[s.length / 2]!) / 2 : null; };
const quantile = (v: number[], p: number) => { const s = [...v].sort((a, b) => a - b); return s.length ? s[Math.min(s.length - 1, Math.ceil(s.length * p) - 1)]! : null; };
const mean = (v: number[]) => v.length ? v.reduce((a, b) => a + b, 0) / v.length : null;

let solver: GPUSolverInstance | undefined, observer: ZoneObserver | undefined, cellDump: CellDump | undefined, failure: string | undefined;
const walls: { step: number; ms: number; kind: "timed" | "warmup" | "sample" | "afterSample" }[] = [];
let construction: Record<string, number> = {}, pipelined: Record<string, number> | undefined, stepReached = 0;
const samples: SeepObservation[] = [];
try {
  const began = performance.now();
  solver = await method.createSolverAsync!(device, scene, quality, values, undefined, () => {});
  const construction_ms = performance.now() - began;
  await solver.waitForSimulationReady?.();
  await whenGPUCompilationIdle();
  construction = { construction_ms, compileIdle_ms: performance.now() - began, allocatedBytes: memory.allocatedBytes, peakBytes: memory.snapshot().peakBytes };
  log({ kind: "constructed", ...construction, grid: solver.fluidDomain?.dimensions, nbEngine: (solver.info as { nbEngine?: unknown } | undefined)?.nbEngine });
  // The brick engine (BrickNBFlipSolver: brickEngine) has its own observer over its pools; the old engine's reads the mixed frame.
  const brick = (solver as unknown as { brickEngine?: unknown }).brickEngine ? solver as unknown as BrickNBFlipSolver : undefined;
  assert.ok(!brick || !auditSteps.size, "--audit dumps the old engine's owners; the brick engine has no cell dump");
  observer = brick ? await createBrickZoneObserver(device, brick, geometry) : methodKey === "nbflip" ? await createNBFlipZoneObserver(device, solver, scene) : await (async () => {
    const source = (solver as unknown as { observation?: SeepObservationSource }).observation;
    if (!source) throw new Error("The Seep solver publishes no `observation` (SeepObservationSource)");
    return createSeepZoneObserver(device, source, geometry);
  })();
  const s = solver;
  if (auditSteps.size) cellDump = methodKey === "nbflip" ? await createNBFlipCellDump(device, s, geometry)
    : await createSeepCellDump(device, (s as unknown as { observation: SeepObservationSource }).observation, geometry);
  const sample = async (step: number) => {
    await s.assertSimulationHealthy?.();
    const sums = await observer!.observe();
    let extra: Record<string, unknown> | undefined;
    if (nbStats && methodKey === "nbflip") { const st = await s.readStats(); extra = { nbDrift: st.volumeDrift, nbVolumeCells: st.volumeCellSum }; }
    if (sums.wallAll !== undefined) extra = { ...extra, wallAll_L: sums.wallAll * geometry.cell_m ** 3 * 1000 };
    const row = seepObservation(geometry, step, dt, sums, samples.at(-1), samples[0]?.liquid_L.total, extra);
    samples.push(row); log(row as unknown as Record<string, unknown>);
    if (cellDump && auditSteps.has(step)) log(auditCells(geometry, step, await cellDump.dump(), row.throatU_m_s ?? 0) as unknown as Record<string, unknown>);
  };
  const admit = async (step: number) => {
    // A full receipt ring (framePending) defers admission: yield to Dawn's event loop, as the app's next frame would.
    while (!s.advanceTo(step * dt, [])) {
      if (!s.framePending) { const why = s.info.simulationFailure; throw new Error(`Step ${step} refused with no frame pending${why ? `: ${String((why as { message?: unknown }).message ?? JSON.stringify(why)).slice(0, 600)}` : ""}`); }
      await new Promise<void>(r => setImmediate(r));
    }
  };
  const fence = () => s.awaitFrameCompletion ? s.awaitFrameCompletion() : device.queue.onSubmittedWorkDone();
  await sample(0);
  if (mode === "fenced") {
    for (let step = 1; step <= steps; step++) {
      stepReached = step;
      // The timer wraps the encoders made while it is on. A solver that encodes a step ahead (the brick engine's
      // encodeAhead) makes step k's encoder during step k - 1, so the window opens one step early and closes before its
      // last step: COUNT steps' encoders either way, and none wrapped is still unsubmitted when collect frees its buffers.
      if (timer && profWindow && step === profWindow[0]! - 1) timer.enabled = true;
      if (timer && profWindow && step === profWindow[0]! + profWindow[1]! - 1) timer.enabled = false;
      const start = performance.now();
      await admit(step); await fence();
      const ms = performance.now() - start;
      if (timer && profWindow && step === profWindow[0]! + profWindow[1]! - 1) {
        const { byLabel, dispatches, untimed } = await timer.collect();
        const n = profWindow[1]!, total = [...byLabel.values()].reduce((a, b) => a + b, 0);
        log({ kind: "prof", from: profWindow[0], steps: n, dispatchesPerStep: dispatches / n, untimed, gpuMsPerStep: total / n,
          byLabel: Object.fromEntries([...byLabel].sort((a, b) => b[1] - a[1]).map(([l, v]) => [l, +(v / n).toFixed(4)])) });
      }
      if (s.info.simulationPipelineError) throw new Error(String(s.info.simulationPipelineError));
      const isSample = step % every === 0 || step === steps;
      walls.push({ step, ms, kind: step < from ? "warmup" : isSample ? "sample" : step > 1 && (step - 1) % every === 0 ? "afterSample" : "timed" });
      if (isSample) await sample(step);
      if (errors.length) throw new Error(`GPU error: ${errors[0]}`);
    }
  } else {
    const start = performance.now();
    for (let step = 1; step <= steps; step++) { stepReached = step; await admit(step); }
    const submitted_ms = performance.now() - start;
    await fence();
    const total_ms = performance.now() - start;
    pipelined = { total_ms, submitted_ms, perStep_ms: total_ms / steps };
    if (s.info.simulationPipelineError) throw new Error(String(s.info.simulationPipelineError));
    await sample(steps);
  }
  assert.deepEqual(errors, []);
} catch (error) {
  failure = `${error instanceof Error ? error.message : String(error)}`;
  console.error(`failed at step ${stepReached}: ${failure}`);
}
const timed = walls.filter(w => w.kind === "timed").map(w => w.ms);
const apart = (kind: string) => { const v = walls.filter(w => w.kind === kind).map(w => w.ms); return { n: v.length, median: median(v), mean: mean(v), max: v.length ? Math.max(...v) : null }; };
const thirds = [0, 1, 2].map(i => mean(timed.slice(Math.floor(i * timed.length / 3), Math.floor((i + 1) * timed.length / 3))));
const summary = {
  kind: "summary", name, method: methodId, mode, steps: stepReached, requestedSteps: steps, failure,
  construction, peakGPUBytes: memory.snapshot().peakBytes,
  step_ms: mode === "fenced" ? { n: timed.length, median: median(timed), p95: quantile(timed, 0.95), mean: mean(timed), max: timed.length ? Math.max(...timed) : null, thirds,
    warmup: apart("warmup"), sample: apart("sample"), afterSample: apart("afterSample") } : undefined,
  pipelined,
  // The band solve's cap record (NB-FLIP: frames that ran the cycle cap above target, a reported state).
  pressureCap: solver?.info.uniformPressureCappedFrames === undefined ? undefined : { capped: solver.info.uniformPressureCappedFrames, short: solver.info.uniformPressureShortFrames,
    worstCapped: solver.info.uniformPressureWorstCappedResidual, lastCappedFrame: solver.info.uniformPressureLastCappedFrame },
  final: samples.at(-1),
};
log(summary);
writeFileSync(`${out}/${name}-summary.json`, JSON.stringify({ ...summary, walls }, null, 1) + "\n");
observer?.destroy(); cellDump?.destroy(); solver?.destroy();
await device.queue.onSubmittedWorkDone().catch(() => {});
raw.destroy();
process.exit(failure ? 1 : 0);
