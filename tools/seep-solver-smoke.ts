/** Lane S development runner for the Seep solver (lib/methods/seep): a short
 * headless run on the default reservoir-seep scene, printing lane H's zone
 * observables (tools/seep-observables.ts) with the solver's own diagnostics,
 * fenced step times and, with --profile, one timestamp-attributed step.
 *
 *   tools/gpu/fluid-gpu-run.sh check env WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js \
 *     node --import tsx tools/seep-solver-smoke.ts --steps=120 --every=30 [--profile]
 *
 * Not the gate harness: tools/seep-bench.ts and tools/seep-compare.ts (lane H) are. */
import assert from "node:assert/strict";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { managedGPUDevice } from "../lib/core/gpu-compilation-manager";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { createProcessRetainedDawnGPU, paceDawnEventPolling } from "../lib/harness/node-dawn-provider";
import { createSeepSolver } from "../lib/methods/seep";
import { seepGeometry, seepObservation, type SeepObservation } from "./seep-observables";
import { createSeepZoneObserver } from "./seep-observe-seep";

const args = process.argv.slice(2);
const option = (key: string) => args.find(a => a.startsWith(`--${key}=`))?.slice(key.length + 3);
const steps = Number(option("steps") ?? 60), every = Number(option("every") ?? 30);
const profile = args.includes("--profile");

paceDawnEventPolling();
const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE ?? resolve("node_modules/webgpu/index.js")).href);
Object.assign(globalThis, dawn.globals);
const adapter = await createProcessRetainedDawnGPU(dawn, ["backend=metal"]).requestAdapter();
assert.ok(adapter, "No WebGPU adapter");
const raw = await adapter.requestDevice({
  requiredFeatures: (["subgroups", "float32-filterable", "timestamp-query"] as const).filter(f => adapter.features.has(f)),
  requiredLimits: requiredFluidDeviceLimits(adapter.limits),
});
const device = managedGPUDevice(raw, { requireWorkerRealm: false });
const errors: string[] = [];
device.addEventListener("uncapturederror", e => { e.preventDefault(); errors.push(e.error.message); console.error("GPU error:", e.error.message); });

const scene = sceneDocument(getSceneDefinition("reservoir-seep"));
const geometry = seepGeometry(scene);
const began = performance.now();
const solver = await createSeepSolver(device, scene);
console.log(JSON.stringify({ kind: "constructed", ms: Math.round(performance.now() - began), allocatedMB: Math.round(solver.info.allocatedBytes / 2 ** 20) }));
const observer = await createSeepZoneObserver(device, solver.observation, geometry);
const rows: SeepObservation[] = [];
const sample = async (step: number) => {
  const d = await solver.readDiagnostics();
  const row = seepObservation(geometry, step, 1 / 60, await observer.observe(), rows.at(-1), rows[0]?.liquid_L.total);
  rows.push(row);
  const r = (x: number | null, n = 3) => x === null ? null : Number(x.toFixed(n));
  console.log(JSON.stringify({
    step, L: { total: r(row.liquid_L.total), left: r(row.liquid_L.left), wall: r(row.liquid_L.wall), right: r(row.liquid_L.right), floor: r(row.liquid_L.rightFloorBand) },
    q: r(row.qLeft_Lps), qR: r(row.qRight_Lps), u: r(row.throatU_m_s), wet: row.wetFloor_cells, cx: r(row.rightCentroid_x, 1), drift: Number(row.drift.toExponential(3)),
    solverDrift: Number(d.volumeDrift.toExponential(3)), particles: d.particles, seeded: d.seeded, deleted: d.deleted,
    cycles: d.cycles, meanCycles: r(d.cyclesTotal / Math.max(1, d.step), 2), capHits: d.capHits,
    res0: Number(d.firstResidual.toExponential(2)), res: Number(d.lastResidual.toExponential(2)), maxSeg: d.maxSegments, lists: d.lists,
    hist: d.residualHistory.map(x => Number(x.toExponential(2))),
    failures: d.failures,
  }));
};
await sample(0);
const walls: number[] = [];
for (let step = 1; step <= steps; step++) {
  const t0 = performance.now();
  solver.advanceTo(step / 60);
  await solver.awaitFrameCompletion();
  walls.push(performance.now() - t0);
  if (errors.length) break;
  if (step % every === 0 || step === steps) await sample(step);
}
const sorted = walls.slice(5).sort((a, b) => a - b);
console.log(JSON.stringify({ kind: "timing", fencedMedian_ms: sorted.length ? Number(sorted[sorted.length >> 1].toFixed(2)) : null, max_ms: Number(Math.max(...walls).toFixed(2)) }));
if (profile && !errors.length) {
  const t = await solver.profileStep();
  const groups = new Map<string, number>();
  for (const { label, ms } of t) { const k = label.replace(/\.\d+$/, ""); groups.set(k, (groups.get(k) ?? 0) + ms); }
  const total = t.reduce((s, x) => s + x.ms, 0);
  console.log(JSON.stringify({ kind: "profile", total_ms: Number(total.toFixed(3)), dispatches: t.length,
    byKernel: Object.fromEntries([...groups].sort((a, b) => b[1] - a[1]).map(([k, v]) => [k, Number(v.toFixed(3))])) }));
}
observer.destroy(); solver.destroy();
await device.queue.onSubmittedWorkDone().catch(() => {});
raw.destroy();
process.exit(errors.length ? 1 : 0);
