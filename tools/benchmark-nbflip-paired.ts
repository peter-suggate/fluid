/** Alternate complete UI-default simulation frames from two source snapshots.
 * Both solvers share one device; every frame is fenced before the other runs.
 * Construction, renderer and final diagnostics are outside the timed window.
 * node --import tsx tools/benchmark-nbflip-paired.ts NAME
 *   --baseline=/absolute/snapshot --candidate=/absolute/snapshot
 *   [--steps=620] [--from=13] [--scene=nbflip-figure-9-teaser] [--out=DIR]
 */
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdirSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import { cpus, release, totalmem } from "node:os";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { GPUMemoryProfile } from "./gpu-memory-profile";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { createProcessRetainedDawnGPU, paceDawnEventPolling } from "../lib/harness/node-dawn-provider";
import type { uniformNarrowBandMethod } from "../lib/methods/uniform/uniform-narrow-band-method";
import type { WebGPUUniformReferenceSolver } from "../lib/methods/uniform/webgpu-uniform-reference";

const option = (key: string) => process.argv.find(a => a.startsWith(`--${key}=`))?.slice(key.length + 3);
const name = process.argv[2]; assert.ok(name); assert.match(name, /^[a-z0-9-]+$/);
for (const a of process.argv.slice(3)) assert.match(a, /^--(?:baseline|candidate|steps|from|scene|out)=.+$/);
const steps = Number(option("steps") ?? 620), from = Number(option("from") ?? 13);
assert.ok(Number.isInteger(steps) && Number.isInteger(from) && from > 0 && steps >= from);
const roots = [option("baseline"), option("candidate")]; assert.ok(roots.every(Boolean));
const sceneId = option("scene") ?? "nbflip-figure-9-teaser";
const out = option("out") ?? "artifacts/nbflip-ui-perf"; mkdirSync(out, { recursive: true });
// The Dawn binding's own polling garbage would put a collection in one arm's frame or the other's.
paceDawnEventPolling();
const errors: string[] = [], solvers: WebGPUUniformReferenceSolver[] = [];
const rows: { frame: number; baseline_ms: number; candidate_ms: number; saved_ms: number; first: string; state: unknown[] }[] = [];
const snapshots: Record<string, unknown>[] = [];
const memory: GPUMemoryProfile[] = [];
let raw: GPUDevice | undefined;
try {
  const dawn = await import(pathToFileURL(resolve("node_modules/webgpu/index.js")).href);
  Object.assign(globalThis, dawn.globals);
  const gpu = createProcessRetainedDawnGPU(dawn, ["backend=metal"]), adapter = await gpu.requestAdapter(); assert.ok(adapter);
  raw = await adapter.requestDevice({ requiredFeatures: ["timestamp-query", ...(adapter.features.has("subgroups") ? ["subgroups" as const] : [])], requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
  raw.addEventListener("uncapturederror", event => { event.preventDefault(); errors.push(event.error.message); });
  let dt = 0;
  for (const [index, root] of roots.entries()) {
    const cwd = resolve(root!);
    const load = (file: string) => import(pathToFileURL(resolve(cwd, file)).href);
    const [{ uniformNarrowBandMethod: method }, { resolveMethodValues }, { sceneDocument }, { getSceneDefinition }, { managedGPUDevice }] = await Promise.all([
      load("lib/methods/uniform/uniform-narrow-band-method.ts"), load("lib/core/method-contract.ts"), load("lib/core/scene-definition.ts"), load("lib/core/scenes.ts"), load("lib/core/gpu-compilation-manager.ts"),
    ]);
    const typed = method as typeof uniformNarrowBandMethod;
    const definition = getSceneDefinition(sceneId), scene = structuredClone(sceneDocument(definition));
    assert.equal(definition.methodProfile?.methodId, typed.id);
    const quality = definition.methodProfile.quality;
    const values = resolveMethodValues(typed, quality, { ...typed.appDefaults, ...definition.methodProfile.overrides });
    const step = typed.effectiveStep_s?.(scene, values) ?? scene.numerics.fixedDt_s;
    const adaptivePolicyHash=createHash("sha256").update(readFileSync(resolve(cwd,"lib/methods/uniform/uniform-narrow-band-budget.wgsl.ts"))).digest("hex");
    if (index) {
      assert.equal(step, dt); assert.deepEqual(values, snapshots[0]!.values); assert.deepEqual(scene, snapshots[0]!.scene);
      assert.equal(adaptivePolicyHash,snapshots[0]!.adaptivePolicyHash,"Compare performance changes under the same adaptive physics policy");
    }
    dt = step; assert.ok(steps * dt <= scene.duration_s + 1e-9);
    const hash = createHash("sha256"), directory = resolve(cwd, "lib/methods/uniform");
    for (const f of readdirSync(directory).filter(f => f.endsWith(".ts")).sort()) { hash.update(f); hash.update(readFileSync(resolve(directory, f))); }
    const start = performance.now();
    memory.push(new GPUMemoryProfile(raw));
    solvers.push(await typed.createSolverAsync!(managedGPUDevice(memory[index]!.device, { requireWorkerRealm: false }), scene, quality, values, undefined, () => {}) as WebGPUUniformReferenceSolver);
    await raw.queue.onSubmittedWorkDone();
    snapshots.push({ root: cwd, sourceHash: hash.digest("hex"), adaptivePolicyHash, quality, values, scene, dt, construction_ms: performance.now() - start });
    console.log(JSON.stringify({ phase: "constructed", variant: index ? "candidate" : "baseline", ...snapshots[index], scene: undefined }));
  }
  for (let frame = 1; frame <= steps; frame++) {
    const wall = [0, 0], state: unknown[] = [];
    for (const i of frame % 2 ? [0, 1] : [1, 0]) {
      const solver = solvers[i]!, start = performance.now();
      assert.ok(solver.advanceTo(frame * dt, [])); await solver.awaitFrameCompletion(); wall[i] = performance.now() - start;
      assert.equal(solver.info.simulationPipelineError, undefined);
      assert.ok(Math.abs((solver.info.completedTime_s ?? 0) - frame * dt) < 1e-8);
      state[i] = { allocatedBytes: memory[i]!.allocatedBytes, ...solver.narrowBandFlipInfo, fineTiles: solver.info.uniformMixedFineTiles, bandTiles: solver.info.uniformPressureBandTiles, bandResidual: solver.info.uniformPressureBandResidual };
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
  };
  const physics = [];
  for (const solver of solvers) {
    const s = await solver.readStats(); physics.push({ maxSpeed_m_s: s.maxSpeed_m_s, volumeCellSum: s.volumeCellSum, representedVolumeCellSum: s.representedVolumeCellSum, volumeDrift: s.volumeDrift, front_m: s.front_m, pressureResidual: s.uniformPressureAcceptedResidual, bandResidual: s.uniformPressureBandResidual });
  }
  writeFileSync(`${out}/${name}.json`, JSON.stringify({ date: new Date().toISOString(), host: { cpu: cpus()[0]?.model, memory: totalmem(), release: release(), node: process.version }, sceneId, scope: "alternating simulation advance and completion on one device; excludes construction and renderer", snapshots, summary, memory: memory.map(m => m.snapshot()), physics, rows, errors }, null, 2) + "\n");
  console.log(JSON.stringify({ phase: "complete", summary }));
} catch (error) {
  writeFileSync(`${out}/${name}-failed.json`, JSON.stringify({ snapshots, rows, errors, error: String(error) }, null, 2) + "\n"); throw error;
} finally {
  for (const solver of solvers) solver.destroy(); await raw?.queue.onSubmittedWorkDone(); raw?.destroy();
}
