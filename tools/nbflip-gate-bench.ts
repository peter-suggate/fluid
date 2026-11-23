/** The NB-FLIP port's cross-scene bench (docs/plans/nbflip-port-2026-11-15.md §3 lane G, §4):
 * Narrow-band FLIP on the high-resolution dam break, letters or paddleboats,
 * observed by tools/nbflip-observe.ts, timed as tools/seep-bench.ts times the
 * seep (the seep itself runs through tools/seep-bench.ts --method=nbflip).
 *
 * node --import tsx tools/nbflip-gate-bench.ts --scene=dam|letters|paddleboats
 *   [--steps=N] [--every=N] [--dense=N] [--mode=fenced|pipelined] [--slot=check|timing] [--roster=moored|app]
 *   [--out=artifacts/nbflip-port/runs] [--name=NAME] [--from=13] [--no-nb-stats] [--set=key=value ...]
 *
 * Scene defaults (steps, every, dense):
 *  - dam (high-resolution-dam-break, 128^3): 300 steps (5 s of the 20 s take), samples at every step
 *    to 16 (the front reaches the far wall by about step 12) and then every 15 steps;
 *  - letters (nbflip-figure-8-letters): 240 steps (all nine drops land by step 203), every 30;
 *  - paddleboats: 300 steps, every 30, hulls moored and wheels driven at their authored
 *    one-third throttle (tools/moored-motor-bodies.ts, as the paddleboats plan's probes); --roster=app offers the
 *    app's initial roster instead (the GPU owns the hulls, its motors drive the wheels: the realtime number).
 * Every scene runs at its methodProfile over the method's app defaults (dt 1/60), as the app does.
 *
 * fenced: each step is advanceTo + awaitFrameCompletion on the wall clock. Sample steps
 * (0, 1..dense, every k-th, the last) observe the fenced state; a sample step and the step
 * after it are excluded from the timing statistics, as are the warm-up steps before --from.
 * Unless --no-nb-stats, a sample step also encodes NB-FLIP's diagnostic volume pass
 * (requestNarrowBandVolumeProbe before its advance, as tools/probe-reservoir-bulk.ts) and the
 * sample reads it (readStats: nbDrift, that step's own measure; the read asks the excluded next
 * step for another pass). Without the request, readStats returns the last pass taken, one sample old.
 * pipelined: every step submitted as soon as the solver admits it, one fence at the end;
 * observed at 0 and at the end only.
 *
 * Writes OUT/NAME.jsonl (meta, constructed, samples, summary) and OUT/NAME-summary.json.
 * Run under tools/gpu/fluid-gpu-run.sh (check for observables, timing for speed). */
import assert from "node:assert/strict";
import { appendFileSync, mkdirSync, writeFileSync } from "node:fs";
import { execSync } from "node:child_process";
import { cpus, platform, release } from "node:os";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import "../lib/methods";
import { getMethod } from "../lib/core/method-registry";
import { resolveMethodValues, type GPUSolverInstance } from "../lib/core/method-contract";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { managedGPUDevice, whenGPUCompilationIdle } from "../lib/core/gpu-compilation-manager";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { createProcessRetainedDawnGPU, paceDawnEventPolling } from "../lib/harness/node-dawn-provider";
import { initializeRigidBodies, type RigidBodyState } from "../lib/core/rigid-body";
import { GPUMemoryProfile } from "./gpu-memory-profile";
import { mooredMotorBodies } from "./moored-motor-bodies";
import { BOX_SCENES, boxGeometry, boxObservation, damColumn, expectedCells, scheduledDropVolumes, type BoxObservation, type BoxSceneKey } from "./nbflip-observables";
import { createNBFlipBoxObserver, type BoxObserver } from "./nbflip-observe";

const METHOD = "uniform-narrow-band-flip";
export const NBFLIP_GATE_DEFAULTS: Record<BoxSceneKey, { steps: number; every: number; dense: number }> = {
  dam: { steps: 300, every: 15, dense: 16 },
  letters: { steps: 240, every: 30, dense: 0 },
  paddleboats: { steps: 300, every: 30, dense: 0 },
};

const args = process.argv.slice(2);
for (const a of args) assert.ok(/^(--(?:scene|steps|every|dense|out|name|mode|from|slot|set|roster)=.+|--no-nb-stats)$/.test(a), `Unknown option: ${a}`);
const option = (key: string) => args.find(a => a.startsWith(`--${key}=`))?.slice(key.length + 3);
const sceneKey = option("scene") as BoxSceneKey | undefined;
assert.ok(sceneKey && sceneKey in BOX_SCENES, `--scene=${Object.keys(BOX_SCENES).join("|")}`);
const defaults = NBFLIP_GATE_DEFAULTS[sceneKey];
const mode = option("mode") ?? "fenced";
assert.ok(mode === "fenced" || mode === "pipelined", "--mode=fenced|pipelined");
const steps = Number(option("steps") ?? defaults.steps), every = Number(option("every") ?? defaults.every), dense = Number(option("dense") ?? defaults.dense), from = Number(option("from") ?? 13);
assert.ok(Number.isInteger(steps) && steps > 0 && Number.isInteger(every) && every > 0 && Number.isInteger(dense) && dense >= 0 && Number.isInteger(from) && from >= 1);
const nbStats = !args.includes("--no-nb-stats");
const slot = option("slot") ?? "check";
const roster = option("roster") ?? "moored";
assert.ok(roster === "moored" || roster === "app", "--roster=moored|app");
assert.ok(slot === "check" || slot === "timing", "--slot=check|timing (the fluid-gpu-run.sh mode this run holds)");
const sets = Object.fromEntries(args.filter(a => a.startsWith("--set=")).map(a => {
  const [key, ...value] = a.slice(6).split("="); const text = value.join("=");
  return [key!, text !== "" && Number.isFinite(Number(text)) ? Number(text) : text === "true" ? true : text === "false" ? false : text];
}));
const out = resolve(option("out") ?? "artifacts/nbflip-port/runs");
const name = option("name") ?? `${sceneKey}-${mode}-${steps}`;
assert.match(name, /^[A-Za-z0-9._-]+$/);
mkdirSync(out, { recursive: true });
const jsonl = `${out}/${name}.jsonl`;
writeFileSync(jsonl, "");
const log = (row: Record<string, unknown>) => { const text = JSON.stringify(row); appendFileSync(jsonl, text + "\n"); console.log(text.length > 400 ? text.slice(0, 400) + "…" : text); };

const method = getMethod(METHOD);
const sceneId = BOX_SCENES[sceneKey];
const definition = getSceneDefinition(sceneId);
const scene = sceneDocument(definition);
const profile = definition.methodProfile;
const quality = profile?.methodId === METHOD ? profile.quality : "balanced";
const values = resolveMethodValues(method, quality, { ...method.appDefaults, ...(profile?.methodId === METHOD ? profile.overrides : {}), ...sets });
// A --set the method pins (NB-FLIP fixes pressureFullCycles, for one) would otherwise be dropped without a word.
for (const [key, value] of Object.entries(sets)) assert.equal(values[key], value, `--set=${key}=${value}: ${METHOD} resolves ${key} to ${String(values[key])} (a fixed parameter); refusing to run`);
const dt = method.effectiveStep_s?.(scene, values) ?? scene.numerics.fixedDt_s;
assert.ok(steps * dt <= scene.duration_s + 1e-9, "Stay within the authored take");
const geometry = boxGeometry(scene, sceneKey);
const drops = scheduledDropVolumes(scene, dt);
const dam = sceneKey === "dam" ? (() => { const c = damColumn(scene); return { width_cells: c.width_cells, height_cells: c.height_cells, depth_cells: c.depth_cells }; })() : undefined;
const isSample = (step: number) => step === 0 || step <= dense || step % every === 0 || step === steps;
let commit = "unknown"; try { commit = execSync("git rev-parse --short HEAD", { encoding: "utf8", stdio: ["ignore", "pipe", "ignore"] }).trim(); } catch { /* an export */ }
let dirty = false; try { dirty = execSync("git status --porcelain -- lib tools", { encoding: "utf8", stdio: ["ignore", "pipe", "ignore"] }).trim().length > 0; } catch { /* an export */ }
log({ kind: "meta", name, method: METHOD, scene: sceneKey, sceneId, mode, slot, roster, steps, every, dense, from, nbStats, sets, quality, dt, values, geometry, dam, drops,
  date: new Date().toISOString(), cwd: process.cwd(), host: { cpu: cpus()[0]?.model, platform: platform(), release: release(), node: process.version, commit, dirtyLibOrTools: dirty } });

paceDawnEventPolling();
const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE ?? resolve("node_modules/webgpu/index.js")).href);
Object.assign(globalThis, dawn.globals);
const adapter = await createProcessRetainedDawnGPU(dawn, ["backend=metal"]).requestAdapter();
assert.ok(adapter, "No WebGPU adapter");
const raw = await adapter.requestDevice({ requiredFeatures: (["subgroups", "float32-filterable"] as const).filter(f => adapter.features.has(f)), requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
const memory = new GPUMemoryProfile(raw);
const device = managedGPUDevice(memory.device, { requireWorkerRealm: false });
const errors: string[] = [];
device.addEventListener("uncapturederror", e => { e.preventDefault(); errors.push(e.error.message); console.error(e.error.message); });

const median = (v: number[]) => { const s = [...v].sort((a, b) => a - b); return s.length ? s.length % 2 ? s[s.length >> 1]! : (s[s.length / 2 - 1]! + s[s.length / 2]!) / 2 : null; };
const quantile = (v: number[], p: number) => { const s = [...v].sort((a, b) => a - b); return s.length ? s[Math.min(s.length - 1, Math.ceil(s.length * p) - 1)]! : null; };
const mean = (v: number[]) => v.length ? v.reduce((a, b) => a + b, 0) / v.length : null;

let solver: GPUSolverInstance | undefined, observer: BoxObserver | undefined, failure: string | undefined;
const walls: { step: number; ms: number; kind: "timed" | "warmup" | "sample" | "afterSample" }[] = [];
let construction: Record<string, number> = {}, pipelined: Record<string, number> | undefined, stepReached = 0;
const samples: BoxObservation[] = [];
// --roster=moored (default): the probes' roster, the hulls held and the wheels driven from the CPU each step, so the
// rigid system re-signs and re-uploads every body every step (about 0.75 ms CPU for paddleboats' 18). --roster=app:
// the app's, the initial roster offered unchanged every step, so the GPU owns the hulls and its motors drive the wheels.
const appRoster = roster === "app" ? initializeRigidBodies(scene.rigidBodies) : [];
const bodies: (t: number) => RigidBodyState[] = sceneKey !== "paddleboats" ? () => [] : roster === "app" ? () => appRoster : mooredMotorBodies(scene, dt);
try {
  const began = performance.now();
  solver = await method.createSolverAsync!(device, scene, quality, values, undefined, () => {});
  const construction_ms = performance.now() - began;
  await solver.waitForSimulationReady?.();
  await whenGPUCompilationIdle();
  construction = { construction_ms, compileIdle_ms: performance.now() - began, allocatedBytes: memory.allocatedBytes, peakBytes: memory.snapshot().peakBytes };
  log({ kind: "constructed", ...construction, grid: solver.fluidDomain?.dimensions });
  observer = await createNBFlipBoxObserver(device, solver, scene, geometry);
  const s = solver;
  let initial_cells = 0;
  const sample = async (step: number) => {
    await s.assertSimulationHealthy?.();
    const sums = await observer!.observe();
    if (step === 0) initial_cells = sums.total;
    let extra: Record<string, unknown> = { bulkRegions: sums.bulkRegions };
    if (nbStats && mode === "fenced") { const st = await s.readStats(); extra = { ...extra, nbDrift: st.volumeDrift, nbVolumeCells: st.volumeCellSum }; }
    const row = boxObservation(geometry, sceneKey, step, dt, sums, expectedCells(initial_cells, drops, step), { dam }, extra);
    samples.push(row); log(row as unknown as Record<string, unknown>);
  };
  const probe = (solver as unknown as { mixedFrame?: { requestNarrowBandVolumeProbe?(): void } }).mixedFrame;
  const admit = async (step: number) => {
    if (nbStats && mode === "fenced" && isSample(step)) probe?.requestNarrowBandVolumeProbe?.();
    while (!s.advanceTo(step * dt, bodies(step * dt))) {
      if (!s.framePending) {
        // Why the engine refused: the uniform engine's failMixedFrame message (simulationPipelineError), the brick
        // engine's latched failure (simulationFailure: a FAIL word from a lagged receipt, or an advance that threw).
        const info = s.info as { simulationPipelineError?: unknown; simulationFailure?: { code?: unknown; message?: unknown; frame?: unknown; operands?: unknown; rawWords?: unknown } };
        const f = info.simulationFailure;
        const why = info.simulationPipelineError ?? (f ? JSON.stringify({ code: f.code, message: f.message, frame: f.frame, operands: f.operands, rawWords: f.rawWords }) : undefined);
        // The admission state, for a refusal with neither: the clock, a held edit (brick), the uniform engine's own.
        const e = s as unknown as Record<string, unknown>, engine = e.brickEngine as { editHeld?: unknown } | undefined;
        const state = { requested_s: step * dt, submitted_s: (s.info as { submittedTime_s?: unknown }).submittedTime_s, editHeld: engine?.editHeld, lastTime_s: e.lastTime, disposed: e.disposed, admission: e.mixedPipelineAdmission !== undefined, encodedSteps: (s.info as { encodedSteps?: unknown }).encodedSteps, completed_s: (s.info as { completedTime_s?: unknown }).completedTime_s };
        throw new Error(`Step ${step} refused with no frame pending${why ? `: ${String(why).slice(0, 2000)}` : ""} ${JSON.stringify(state)}`);
      }
      await new Promise<void>(r => setImmediate(r));
    }
  };
  const fence = () => s.awaitFrameCompletion ? s.awaitFrameCompletion() : device.queue.onSubmittedWorkDone();
  await sample(0);
  if (mode === "fenced") {
    for (let step = 1; step <= steps; step++) {
      stepReached = step;
      const start = performance.now();
      await admit(step); await fence();
      const ms = performance.now() - start;
      if (s.info.simulationPipelineError) throw new Error(String(s.info.simulationPipelineError));
      const sampled = isSample(step);
      walls.push({ step, ms, kind: step < from ? "warmup" : sampled ? "sample" : isSample(step - 1) ? "afterSample" : "timed" });
      if (sampled) await sample(step);
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
  kind: "summary", name, method: METHOD, scene: sceneKey, mode, steps: stepReached, requestedSteps: steps, failure,
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
observer?.destroy(); solver?.destroy();
await device.queue.onSubmittedWorkDone().catch(() => {});
raw.destroy();
process.exit(failure ? 1 : 0);
