// M1 frozen-state A/B (docs/plans/gpu-shape/m1-distinct-once.md), from fluid-scratch/solid-explore fixstate.mts + fixeq.mts:
// warm a scene up (census settings: brick engine, fixed dt 1/60, presentation off, moored bodies on paddleboats), save
// the engine state, then for each variant of the kernel's compiled text (m1/variants.mts):
//   exactness: restore, run the base placements, read BUFS; restore, run the variant's; compare word for word
//              (and base against base first: the replay's own determinism);
//   timing:    restore, R batched copies of the placements untimed then timed (gpu-kernel-timer), base and variants
//              interleaved, S samples each, min and median; the clock probe (clock-probe.ts) after every sample.
//   SCENE=dam|paddleboats|seep|<id> STEPS=100 [TREE=dir] [DUMP=file] [REF=file] KERNEL=lAdvect [ONLY=name,...] BUFS=phi,set,vel,fcell S=5 R=12 OUT=x.jsonl
//   tools/gpu/fluid-gpu-run.sh timing node --import tsx tools/experiments/gpu-shape/m1/m1fix.mts
import { appendFileSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { createClockProbe, TOP_CLOCK_PROBE_US } from "../clock-probe.js";
import { variants } from "./variants.mjs";
// TREE: the checkout or read-only export (tools/nbflip-export.sh) whose lib/ builds the engine (default this checkout).
const root = process.env.TREE ?? resolve(import.meta.dirname, "../../../..");
await import(`${root}/lib/methods/index.ts`);
const { getMethod } = await import(`${root}/lib/core/method-registry.ts`);
const { resolveMethodValues } = await import(`${root}/lib/core/method-contract.ts`);
const { sceneDocument } = await import(`${root}/lib/core/scene-definition.ts`);
const { getSceneDefinition } = await import(`${root}/lib/core/scenes.ts`);
const { managedGPUDevice } = await import(`${root}/lib/core/gpu-compilation-manager.ts`);
const { requiredFluidDeviceLimits } = await import(`${root}/lib/core/webgpu-device-limits.ts`);
const { createProcessRetainedDawnGPU, paceDawnEventPolling } = await import(`${root}/lib/harness/node-dawn-provider.ts`);
const { gpuKernelTimer } = await import(`${root}/tools/gpu-kernel-timer.ts`);
const { mooredMotorBodies } = await import(`${root}/tools/moored-motor-bodies.ts`);
const SCENES: Record<string, string> = { seep: "reservoir-seep", dam: "high-resolution-dam-break", letters: "nbflip-figure-8-letters", paddleboats: "paddleboats", teaser: "nbflip-figure-9-teaser" };
const sceneKey = process.env.SCENE ?? "dam", sceneId = SCENES[sceneKey] ?? sceneKey;
const N = Number(process.env.STEPS ?? 100), K = process.env.KERNEL ?? "lAdvect", R = process.env.ONE ? 1 : Number(process.env.R ?? 12), SAMPLES = Number(process.env.S ?? 5);
// ONE=1 (kernels that change their own inputs, pMove): R = 1 and a restore before the timed copy, so it runs on the frozen state.
const ONE = !!process.env.ONE;
const ONLY = process.env.ONLY?.split(","), BUFS = (process.env.BUFS ?? "phi,set,vel,fcell").split(","), OUT = process.env.OUT;
const emit = (o: Record<string, unknown>) => { if (OUT) appendFileSync(OUT, JSON.stringify({ scene: sceneKey, kernel: K, ...o }) + "\n"); };
const methodId = "uniform-narrow-band-flip", method = getMethod(methodId);
const definition = getSceneDefinition(sceneId), profile = definition.methodProfile, base = sceneDocument(definition);
const scene = { ...base, numerics: { ...base.numerics, fixedDt_s: 1 / 60, maxDt_s: 1 / 60 } };
const quality = profile?.methodId === methodId ? profile.quality : "balanced";
const values = resolveMethodValues(method, quality, { ...method.appDefaults, ...(profile?.methodId === methodId ? profile.overrides : {}), engine: "brick" });
paceDawnEventPolling();
const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE ?? resolve(root, "node_modules/webgpu/index.js")).href);
Object.assign(globalThis, dawn.globals);
const adapter = await createProcessRetainedDawnGPU(dawn, ["backend=metal", "disable-dawn-features=timestamp_quantization"]).requestAdapter();
const raw: GPUDevice = await adapter.requestDevice({ requiredFeatures: ["subgroups", "float32-filterable", "timestamp-query"].filter((f: string) => adapter.features.has(f)), requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
let gpuErrors = 0;
raw.addEventListener?.("uncapturederror", (ev: any) => { gpuErrors++; console.log("GPU ERROR", ev.error?.message); emit({ kind: "gpuError", message: String(ev.error?.message) }); });
const timer = gpuKernelTimer(raw);
const device = managedGPUDevice(timer.device, { requireWorkerRealm: false });
const s: any = await method.createSolverAsync!(device, scene, quality, values, undefined, () => {});
await s.waitForSimulationReady?.();
s.setPresentationEnabled?.(false);
const dt = 1 / 60;
const bodies: (t: number) => unknown[] = sceneKey === "paddleboats" || (scene.rigidBodies?.length ?? 0) > 0 ? mooredMotorBodies(scene, dt) : () => [];
for (let k = 1; k <= N; k++) while (!s.advanceTo(k * dt + 1e-12, bodies(k * dt))) { if (s.info.simulationFailure) throw new Error(s.info.simulationFailure.message); await new Promise<void>(r => setImmediate(r)); }
const e = s.brickEngine; await e.fence();
const all = [...e.scheduleNow.part1, ...e.scheduleNow.part2];
const ps = all.filter((p: any) => p.kernel.label === K); if (!ps.length) throw new Error(`${K} not in schedule`);
// PRE=1: the frozen state is the next step's, run up to the kernel's first placement (a kernel whose inputs the
// step's earlier stages make, pMove: the bins and bounds; replayed off the finished step it latches FAIL.REACH).
if (process.env.PRE) await e.run(all.slice(0, all.indexOf(ps[0])));
const S = await e.save();
const fail = async () => [...(await e.read("set", 0, 2))].map(x => "0x" + x.toString(16)).join("/");
console.log(`fail/info words at the frozen state: ${await fail()}`);
const probe = createClockProbe(raw, { capacity: 2 });
const clock = async () => { const enc = raw.createCommandEncoder(); probe.encode(enc); raw.queue.submit([enc.finish()]); const [us] = await probe.read(); return us!; };
// DUMP=file writes this tree's compiled kernel text; REF=file adds it as variant "ref" (a landing against HEAD's text,
// in process on one frozen state: exactness and timing of the built source).
if (process.env.DUMP) writeFileSync(process.env.DUMP, ps[0].kernel.code);
const ref = process.env.REF ? readFileSync(process.env.REF, "utf8") : undefined;
const vs = [...variants.filter(v => v.kernel === K && (!ONLY || ONLY.includes(v.name))), ...(ref ? [{ kernel: K, name: "ref", apply: () => ref }] : [])];
const sets = [{ name: "base", pl: ps }, ...vs.map(v => { const memo = new Map<any, any>();
  const pl = ps.map((p: any) => { if (!memo.has(p.kernel)) memo.set(p.kernel, { ...p.kernel, label: `${p.kernel.label}V`, code: v.apply(p.kernel.code) }); return { ...p, kernel: memo.get(p.kernel) }; });
  return { name: v.name, pl }; })];
console.log(`${sceneKey} step ${N}: ${K} ${ps.length} placements; variants ${vs.map(v => v.name).join(", ")}`);
// ---- exactness ----
// BUFS=all: every engine buffer (the kernel's written ones among them, whatever they are).
if (BUFS[0] === "all") BUFS.splice(0, 1, ...[...e.buffers.keys()].filter((b: string) => b !== "boot"));
const runFrom = async (pl: any[]) => { await e.restore(S); await e.run(pl); await e.fence(); };
await runFrom(ps); console.log(`fail/info words after the base placements: ${await fail()}`); const A: Record<string, Uint32Array> = {}; for (const b of BUFS) A[b] = new Uint32Array(await e.read(b));
const nanOf = (w: Uint32Array) => w.reduce((n, x) => n + (((x >>> 23) & 0xff) === 0xff && (x & 0x7fffff) !== 0 ? 1 : 0), 0);
const nanA = A.phi ? nanOf(A.phi) : 0;
for (const set of [{ name: "base(again)", pl: ps }, ...sets.slice(1)]) {
  await runFrom(set.pl); const diffs: Record<string, number> = {}; const ex: string[] = [];
  let nan = 0;
  for (const b of BUFS) { let d = 0; const a = A[b]!, c = new Uint32Array(await e.read(b)); for (let i = 0; i < a.length; i++) if (a[i] !== c[i]) { d++; if (ex.length < 6) ex.push(`${b}[${i}] ${a[i]}/${c[i]}`); } diffs[b] = d; if (b === "phi") nan = nanOf(c) - nanA; }
  if (nan) console.log(`  ${set.name}: ${nan} more NaN words in phi than base`);
  console.log(`exact ${set.name}: ${BUFS.map(b => `${b} ${diffs[b]}`).join(", ")} words differ${ex.length ? " e.g. " + ex.join(" ") : ""}`);
  emit({ kind: "exact", variant: set.name, diffs, examples: ex });
}
// ---- timing ----
if (process.env.NOTIME) { console.log(gpuErrors ? `GPU ERRORS ${gpuErrors}` : "ok"); s.destroy(); process.exit(0); }
const time = async (pl: any[]) => { await e.restore(S); const many = Array.from({ length: R }, () => pl).flat(); await e.run(many); if (ONE) await e.restore(S); timer.enabled = true; await e.run(many); timer.enabled = false;
  const { byLabel } = await timer.collect(); return [...byLabel.values()].reduce((a: number, b: number) => a + b, 0) / R; };
const res = new Map<string, { ms: number[]; probe: number[] }>(sets.map(x => [x.name, { ms: [], probe: [] }]));
for (let t = 0; t < SAMPLES; t++) for (const set of sets) { const ms = await time(set.pl); const pr = await clock(); res.get(set.name)!.ms.push(ms); res.get(set.name)!.probe.push(pr); }
const med = (v: number[]) => [...v].sort((x, y) => x - y)[v.length >> 1]!;
const b0 = Math.min(...res.get("base")!.ms);
for (const [name, r] of res) {
  const mn = Math.min(...r.ms);
  console.log(`time ${name}: min ${mn.toFixed(3)} ms, median ${med(r.ms).toFixed(3)} [${r.ms.map(x => x.toFixed(3)).join(" ")}] (${((mn / b0 - 1) * 100).toFixed(1)}% vs base); clock probe ${r.probe.map(x => x.toFixed(1)).join(" ")} us (top ${TOP_CLOCK_PROBE_US})`);
  emit({ kind: "time", variant: name, minMs: mn, medianMs: med(r.ms), ms: r.ms, probeUs: r.probe, vsBase: mn / b0 - 1 });
}
emit({ kind: "done", gpuErrors }); console.log(gpuErrors ? `GPU ERRORS ${gpuErrors}` : "ok");
s.destroy(); process.exit(0);
