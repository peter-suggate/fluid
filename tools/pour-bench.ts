/** The pour rewrite's benchmark and observation harness (docs/plans/pour-rewrite-2026-11-20.md §3, §5).
 *
 * node --import tsx tools/pour-bench.ts --method=brick|pour [--steps=625] [--every=25] [--from=9]
 *   [--out=artifacts/pour-rewrite/runs] [--name=NAME] [--mode=fenced|pipelined] [--slot=check|timing]
 *   [--prof=FROM:COUNT]   (the kernel timer, tools/gpu-kernel-timer.ts, over steps FROM..FROM+COUNT-1: attribution
 *                          only, it splits every dispatch into its own pass)
 *
 * Both solvers are built through the method registry on the same sceneDocument("nbflip-figure-7-pour"), on the same
 * Dawn device setup, and advanced by the same loop. The brick engine takes the scene's methodProfile with engine=brick.
 * Observables (tools/pour-observables.ts) come from each solver's volumeTexture after a fence, the same reduction for
 * both. fenced: each step is advanceTo + awaitFrameCompletion on the wall clock; a sample step and the one after it
 * are excluded from the timing statistics, as are the warm-up steps before --from. pipelined: every step submitted as
 * the solver admits it, one fence at the end; observed at 0 and the end only.
 *
 * Writes OUT/NAME.jsonl (meta, constructed, samples, summary). Run under tools/gpu/fluid-gpu-run.sh. */
import assert from "node:assert/strict";
import { appendFileSync, mkdirSync, writeFileSync } from "node:fs";
import { execSync } from "node:child_process";
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
import { NBFLIP_POUR_SCENE_ID } from "../lib/core/nbflip-paper-scenes";
import { GPUMemoryProfile } from "./gpu-memory-profile";
import { gpuKernelTimer } from "./gpu-kernel-timer";
import { pourGeometry, pourObservation, readFillTexture, type PourObservation } from "./pour-observables";

export const POUR_BENCH_METHODS = { brick: "uniform-narrow-band-flip", pour: "pour-m1max" } as const;

const args = process.argv.slice(2);
for (const a of args) assert.ok(/^--(?:method|steps|every|out|name|mode|from|slot|prof)=.+$/.test(a), `Unknown option: ${a}`);
const option = (key: string) => args.find(a => a.startsWith(`--${key}=`))?.slice(key.length + 3);
const methodKey = option("method") as keyof typeof POUR_BENCH_METHODS | undefined;
assert.ok(methodKey && methodKey in POUR_BENCH_METHODS, "--method=brick|pour");
const mode = option("mode") ?? "fenced";
assert.ok(mode === "fenced" || mode === "pipelined", "--mode=fenced|pipelined");
const steps = Number(option("steps") ?? 625), every = Number(option("every") ?? 25), from = Number(option("from") ?? 9);
assert.ok(Number.isInteger(steps) && steps > 0 && Number.isInteger(every) && every > 0 && Number.isInteger(from) && from >= 1);
const slot = option("slot") ?? "check";
assert.ok(slot === "check" || slot === "timing", "--slot=check|timing");
const profWindow = option("prof")?.split(":").map(Number);
assert.ok(!profWindow || (profWindow.length === 2 && profWindow[0]! >= 2 && profWindow[0]! + profWindow[1]! - 1 <= steps && mode === "fenced"), "--prof=FROM:COUNT inside a fenced run");
const out = resolve(option("out") ?? "artifacts/pour-rewrite/runs");
const name = option("name") ?? `${methodKey}-${mode}-${steps}`;
assert.match(name, /^[A-Za-z0-9._-]+$/);
mkdirSync(out, { recursive: true });
const jsonl = `${out}/${name}.jsonl`;
writeFileSync(jsonl, "");
const log = (row: Record<string, unknown>) => { const text = JSON.stringify(row); appendFileSync(jsonl, text + "\n"); console.log(text.length > 400 ? text.slice(0, 400) + "…" : text); };

const methodId = POUR_BENCH_METHODS[methodKey];
const method = getMethod(methodId);
assert.equal(method.id, methodId, `Method ${methodId} is not registered`);
const definition = getSceneDefinition(NBFLIP_POUR_SCENE_ID), scene = sceneDocument(definition), profile = definition.methodProfile;
const quality = profile?.methodId === methodId ? profile.quality : "balanced";
const values = resolveMethodValues(method, quality, { ...method.appDefaults, ...(profile?.methodId === methodId ? profile.overrides : {}), ...(methodKey === "brick" ? { engine: "brick" } : {}) });
const dt = method.effectiveStep_s?.(scene, values) ?? scene.numerics.fixedDt_s;
const geometry = pourGeometry(scene);
let commit = "unknown"; try { commit = execSync("git rev-parse --short HEAD", { encoding: "utf8" }).trim(); } catch { /* not a checkout */ }
log({ kind: "meta", name, method: methodId, methodKey, mode, slot, steps, every, from, prof: profWindow, scene: NBFLIP_POUR_SCENE_ID, quality, dt, geometry, date: new Date().toISOString(), commit });

paceDawnEventPolling();
const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE ?? resolve("node_modules/webgpu/index.js")).href);
Object.assign(globalThis, dawn.globals);
// FLUID_WEBGPU_DAWN_FEATURES (e.g. skip_validation) is a diagnostic: the gates and the timing gate run validated, as a browser does.
const adapter = await createProcessRetainedDawnGPU(dawn, ["backend=metal",
  ...(process.env.FLUID_WEBGPU_DAWN_FEATURES ? [`enable-dawn-features=${process.env.FLUID_WEBGPU_DAWN_FEATURES}`] : [])]).requestAdapter();
assert.ok(adapter, "No WebGPU adapter");
const raw = await adapter.requestDevice({ requiredFeatures: (["subgroups", "float32-filterable", ...(profWindow ? ["timestamp-query"] as const : [])] as const).filter(f => adapter.features.has(f)), requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
const memory = new GPUMemoryProfile(raw);
const timer = profWindow ? gpuKernelTimer(memory.device) : undefined;
const device = managedGPUDevice(timer?.device ?? memory.device, { requireWorkerRealm: false });
const errors: string[] = [];
device.addEventListener("uncapturederror", e => { e.preventDefault(); errors.push(e.error.message); console.error(e.error.message); });

const median = (v: number[]) => { const s = [...v].sort((a, b) => a - b); return s.length ? s.length % 2 ? s[s.length >> 1]! : (s[s.length / 2 - 1]! + s[s.length / 2]!) / 2 : null; };
const quantile = (v: number[], p: number) => { const s = [...v].sort((a, b) => a - b); return s.length ? s[Math.min(s.length - 1, Math.ceil(s.length * p) - 1)]! : null; };
const mean = (v: number[]) => v.length ? v.reduce((a, b) => a + b, 0) / v.length : null;

let solver: GPUSolverInstance | undefined, failure: string | undefined, stepReached = 0;
const walls: { step: number; ms: number; kind: "timed" | "warmup" | "sample" | "afterSample" }[] = [];
let construction: Record<string, number> = {}, pipelined: Record<string, number> | undefined;
const samples: PourObservation[] = [];
try {
  const began = performance.now();
  solver = await method.createSolverAsync!(device, scene, quality, values, undefined, () => {});
  const construction_ms = performance.now() - began;
  await solver.waitForSimulationReady?.();
  await whenGPUCompilationIdle();
  construction = { construction_ms, compileIdle_ms: performance.now() - began, allocatedBytes: memory.allocatedBytes, peakBytes: memory.snapshot().peakBytes };
  log({ kind: "constructed", ...construction });
  const s = solver;
  const fence = () => s.awaitFrameCompletion ? s.awaitFrameCompletion() : device.queue.onSubmittedWorkDone();
  const particles = async () => {
    const st = await s.readStats() as unknown as { brickEngine?: { particles?: number }; particles?: number; particleCount?: number };
    return st.brickEngine?.particles ?? st.particles ?? st.particleCount;
  };
  const sample = async (step: number) => {
    await fence(); await s.assertSimulationHealthy?.();
    const fill = await readFillTexture(device, s.volumeTexture, geometry);
    const row = pourObservation(geometry, step, dt, fill, scene, await particles());
    samples.push(row); log(row as unknown as Record<string, unknown>);
  };
  const admit = async (step: number) => {
    while (!s.advanceTo(step * dt + 1e-12, [])) {
      if (!s.framePending) { const why = s.info.simulationFailure; throw new Error(`Step ${step} refused with no frame pending${why ? `: ${String((why as { message?: unknown }).message ?? JSON.stringify(why)).slice(0, 600)}` : ""}`); }
      await new Promise<void>(r => setImmediate(r));
    }
  };
  await sample(0);
  if (mode === "fenced") {
    for (let step = 1; step <= steps; step++) {
      stepReached = step;
      if (timer && profWindow && step === profWindow[0]! - 1) timer.enabled = true;
      if (timer && profWindow && step === profWindow[0]! + profWindow[1]! - 1) timer.enabled = false;
      const start = performance.now();
      await admit(step); await fence();
      const ms = performance.now() - start;
      if (timer && profWindow && step === profWindow[0]! + profWindow[1]! - 1) {
        const { byLabel, dispatches } = await timer.collect();
        const n = profWindow[1]!, total = [...byLabel.values()].reduce((a, b) => a + b, 0);
        log({ kind: "prof", from: profWindow[0], steps: n, dispatchesPerStep: dispatches / n, gpuMsPerStep: total / n,
          // Per-step labels (the brick engine's "stepN.") fold onto one key before averaging.
          byLabel: Object.fromEntries([...[...byLabel].reduce((m, [l, v]) => { const k = l.replace(/step\d+\./g, ""); return m.set(k, (m.get(k) ?? 0) + v); }, new Map<string, number>())]
            .sort((a, b) => b[1] - a[1]).map(([l, v]) => [l, +(v / n).toFixed(4)])) });
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
    await sample(steps);
  }
  assert.deepEqual(errors, []);
} catch (error) {
  failure = error instanceof Error ? error.message : String(error);
  console.error(`failed at step ${stepReached}: ${failure}`);
}
const timed = walls.filter(w => w.kind === "timed").map(w => w.ms);
// The step time by 25-step window (median), as the pour's cost moves with its liquid.
const byWindow = new Map<number, number[]>(); for (const w of walls) if (w.kind === "timed") { const b = Math.floor((w.step - 1) / 25); byWindow.set(b, [...(byWindow.get(b) ?? []), w.ms]); }
const summary = {
  kind: "summary", name, method: methodId, mode, steps: stepReached, requestedSteps: steps, failure, construction, peakGPUBytes: memory.snapshot().peakBytes,
  step_ms: mode === "fenced" ? { n: timed.length, median: median(timed), p95: quantile(timed, 0.95), mean: mean(timed), max: timed.length ? Math.max(...timed) : null,
    byWindow: [...byWindow].map(([b, v]) => [b * 25 + 1, +median(v)!.toFixed(2)]) } : undefined,
  pipelined, final: samples.at(-1),
};
log(summary);
writeFileSync(`${out}/${name}-summary.json`, JSON.stringify({ ...summary, walls }, null, 1) + "\n");
solver?.destroy();
await device.queue.onSubmittedWorkDone().catch(() => {});
raw.destroy();
process.exit(failure ? 1 : 0);
