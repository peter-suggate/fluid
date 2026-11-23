// Frozen whole-step A/B (docs/plans/gpu-shape/m1-distinct-once.md, "Gates for small exact wins"): the landing gate for
// a bit-identical kernel change. The timing round (tools/nbflip-timing-round.sh) stays the scene-level check.
//
// The engine is built from BASE (census settings, as m1/m1fix.mts: brick engine, fixed dt 1/60, presentation off,
// moored bodies on paddleboats) and warmed to STEPS; the state is saved (S0). The candidate's kernel texts come from
// CAND: a child process builds the same scene's engine from that tree and dumps its schedule (no steps). The two
// schedules must agree placement for placement in everything but the WGSL text (label, entry, bindings, dispatch,
// cfg z/w); anything else is a structural change this tool cannot judge, and it fails explicitly. So the A/B covers
// kernel texts only: a change to the engine's host side is not in it (time those with the timing round).
//
//   exactness, per changed placement i (its text differs): restore S0, run the base step's placements 0..i-1, save
//              (S_i); then from S_i run base i, cand i and base i again, and compare every buffer i binds read-write,
//              word for word. Per placement, not the whole step: the step's head claims slots with atomics, so two
//              replays of a whole step need not agree word for word even with one text. A buffer that base and base
//              again write differently is order-dependent (pSeed's claimed particle slots, pMove's fpart ranks): it is
//              compared as a multiset of records (a particle's NB_PSTRIDE words in parts, a word elsewhere), by two
//              order-free hashes, and passes "up to order" when the candidate's multiset is base's.
//   timing:    the whole step (part1 + part2) from S0, base and candidate interleaved (the order rotating by
//              sample), S samples each. S0 is held on the GPU (a mirror of every buffer) and copied back before each
//              replay in the same submit: a sample is R replays untimed, then R timed with gpu-kernel-timer (one pass
//              a dispatch: the Σ is serialised, above the clean frame; its mean over R), then R GPU walls (the step's
//              one command buffer, encoded first: submit to done after its own S0 copy; the min). A third set, base again, is the
//              A/A control: its Δ against base is the replay's noise. Per family (kernel label) the min of the
//              samples; the gate's figure is the changed families' Σ Δ (the unchanged ones run the same text, and
//              their Δ is the A/A's). The clock probe (clock-probe.ts) after every sample (100-250 us: a low clock,
//              invalid).
//
//   SCENE=seep|dam|letters|paddleboats|<id> STEPS=100 BASE=tree CAND=tree [S=9] [R=4] [OUT=x.jsonl] [NOEXACT=1] [NOTIME=1]
//   tools/gpu/fluid-gpu-run.sh timing node --import tsx tools/experiments/gpu-shape/stepab.mts
// A tree is a checkout or a read-only export (tools/nbflip-export.sh). BASE defaults to this checkout; CANDTEXT=file
// (a DUMPTEXT dump) stands in for CAND. DUMPTEXT=file (internal): build TREE's engine, write its schedule, exit.
import { appendFileSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { spawnSync } from "node:child_process";
import { createClockProbe, TOP_CLOCK_PROBE_US } from "./clock-probe.js";

const here = resolve(import.meta.dirname, "../../..");
const DUMP = process.env.DUMPTEXT;
const root = resolve(DUMP ? process.env.TREE! : process.env.BASE ?? here);
const SCENES: Record<string, string> = { seep: "reservoir-seep", dam: "high-resolution-dam-break", letters: "nbflip-figure-8-letters", paddleboats: "paddleboats", teaser: "nbflip-figure-9-teaser" };
const sceneKey = process.env.SCENE ?? "dam", sceneId = SCENES[sceneKey] ?? sceneKey;
const N = Number(process.env.STEPS ?? 100), SAMPLES = Number(process.env.S ?? 9), R = Number(process.env.R ?? 4), OUT = process.env.OUT;
const emit = (o: Record<string, unknown>) => { if (OUT) appendFileSync(OUT, JSON.stringify({ scene: sceneKey, ...o }) + "\n"); };

type Text = { label: string; entry: string; bindings: unknown; dispatch: unknown; z?: number; w?: number; code: string };
const shape = (t: Text) => JSON.stringify([t.label, t.entry, t.bindings, t.dispatch, t.z ?? 0, t.w ?? 0]);
const textOf = (p: any): Text => ({ label: p.kernel.label, entry: p.kernel.entry, bindings: p.kernel.bindings, dispatch: p.dispatch, z: p.z, w: p.w, code: p.kernel.code });

// The candidate's schedule, before this process takes the GPU (the child builds its own device and exits).
let cand: Text[] | undefined;
if (!DUMP) {
  if (process.env.CANDTEXT) cand = JSON.parse(readFileSync(process.env.CANDTEXT, "utf8"));
  else {
    const tree = process.env.CAND; if (!tree) throw new Error("stepab: CAND=tree (or CANDTEXT=file) is required");
    const file = `${process.env.TMPDIR ?? "/tmp"}/stepab-${process.pid}-${sceneKey}.json`;
    const r = spawnSync(process.execPath, ["--import", "tsx", import.meta.filename], { stdio: ["ignore", "inherit", "inherit"],
      env: { ...process.env, DUMPTEXT: file, TREE: resolve(tree), WEBGPU_NODE_MODULE: process.env.WEBGPU_NODE_MODULE ?? resolve(tree, "node_modules/webgpu/index.js") } });
    if (r.status !== 0) throw new Error(`stepab: the candidate's schedule dump failed (rc ${r.status})`);
    cand = JSON.parse(readFileSync(file, "utf8"));
  }
}

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
const e = s.brickEngine;
const step0 = (): any[] => [...e.schedule.part1, ...e.schedule.part2];
if (DUMP) { writeFileSync(DUMP, JSON.stringify(step0().map(textOf))); s.destroy(); process.exit(gpuErrors ? 1 : 0); }

const dt = 1 / 60;
const bodies: (t: number) => unknown[] = sceneKey === "paddleboats" || (scene.rigidBodies?.length ?? 0) > 0 ? mooredMotorBodies(scene, dt) : () => [];
for (let k = 1; k <= N; k++) while (!s.advanceTo(k * dt + 1e-12, bodies(k * dt))) { if (s.info.simulationFailure) throw new Error(s.info.simulationFailure.message); await new Promise<void>(r => setImmediate(r)); }
await e.fence();
// The schedule at the frozen state (a deferred statics variant may have armed since construction: the child's is the
// constructed one, so both are taken here and the base must match the candidate's shape).
const basePl = step0();
if (!cand || cand.length !== basePl.length) throw new Error(`stepab: the base step has ${basePl.length} placements, the candidate's ${cand?.length}: a structural change (time it with the timing round)`);
const mismatch = basePl.findIndex((p, i) => shape(textOf(p)) !== shape(cand![i]!));
if (mismatch >= 0) throw new Error(`stepab: placement ${mismatch} (${basePl[mismatch].kernel.label}) differs in more than its text: ${shape(textOf(basePl[mismatch]))} vs ${shape(cand[mismatch]!)}`);
// One candidate kernel object per distinct (base kernel, text), so a kernel placed many times compiles once.
const memo = new Map<string, any>();
const candPl = basePl.map((p, i) => {
  const code = cand![i]!.code; if (code === p.kernel.code) return p;
  const key = `${p.kernel.label}\n${code}`; if (!memo.has(key)) memo.set(key, { ...p.kernel, code });
  return { ...p, kernel: memo.get(key) };
});
const changed = basePl.map((p, i) => candPl[i] !== p ? i : -1).filter(i => i >= 0);
const changedLabels = [...new Set(changed.map(i => basePl[i].kernel.label as string))];
console.log(`${sceneKey} step ${N}: ${basePl.length} placements, ${changed.length} with the candidate's text (${changedLabels.join(", ") || "none"}); base ${root}, candidate ${process.env.CAND ?? process.env.CANDTEXT}`);
emit({ kind: "setup", steps: N, placements: basePl.length, changed: changed.length, changedLabels, base: root, candidate: process.env.CAND ?? process.env.CANDTEXT });
const S0 = await e.save();
const fail = async () => [...(await e.read("set", 0, 2))].map((x: number) => "0x" + x.toString(16)).join("/");
console.log(`fail/info words at the frozen state: ${await fail()}`);

// ---- exactness, per changed placement from its own snapshot ----
let exactFail = 0;
const equal = (a: Uint32Array, b: Uint32Array) => { if (a.length !== b.length) return false; for (let k = 0; k < a.length; k++) if (a[k] !== b[k]) return false; return true; };
// A multiset of records by two order-free sums of per-record hashes (FNV-1a and a multiply-xorshift), mod 2^32.
const bag = (w: Uint32Array, stride: number) => {
  let s1 = 0, s2 = 0;
  for (let r = 0; r + stride <= w.length; r += stride) {
    let h1 = 0x811c9dc5, h2 = 0x9e3779b9;
    for (let k = 0; k < stride; k++) { const x = w[r + k]!; h1 = Math.imul(h1 ^ x, 0x01000193); h2 = Math.imul((h2 ^ x) + k, 0x85ebca6b); h2 ^= h2 >>> 13; }
    s1 = (s1 + (h1 >>> 0)) >>> 0; s2 = (s2 + (h2 >>> 0)) >>> 0;
  }
  return [s1, s2, w.length];
};
const sameBag = (x: number[], y: number[]) => x.every((v, k) => v === y[k]);
if (!process.env.NOEXACT) {
  const reads = async (names: string[]) => { const o: Record<string, Uint32Array> = {}; for (const b of names) o[b] = new Uint32Array(await e.read(b)); return o; };
  for (const i of changed) {
    const p = basePl[i], written = (p.kernel.bindings as { name: string; kind: string }[]).filter(b => b.kind === "rw").map(b => b.name);
    await e.restore(S0); await e.run(basePl.slice(0, i)); const Si = await e.save();
    const from = async (pl: any) => { await e.restore(Si); await e.run([pl]); return reads(written); };
    const A = await from(p), B = await from(candPl[i]), A2 = await from(p);
    const verdict: Record<string, string> = {}; const ex: string[] = []; let bad = 0;
    for (const b of written) {
      const a = A[b]!, a2 = A2[b]!, c = B[b]!;
      if (equal(a, a2)) {
        let d = 0; for (let k = 0; k < a.length; k++) if (a[k] !== c[k]) { d++; if (ex.length < 4) ex.push(`${b}[${k}] ${a[k]}/${c[k]}`); }
        verdict[b] = d ? `${d} words differ` : "identical"; if (d) bad++;
      } else {
        const stride = b === "parts" ? e.layout.particleStride : 1, ha = bag(a, stride);
        if (!sameBag(ha, bag(a2, stride))) verdict[b] = "order-dependent, base vs base again differ as a multiset too: NOT JUDGED";
        else if (sameBag(ha, bag(c, stride))) verdict[b] = "identical up to order";
        else { verdict[b] = "DIFFERS as a multiset"; bad++; }
      }
    }
    if (bad) exactFail++;
    console.log(`exact #${i} ${p.kernel.label}${p.z !== undefined ? ` z${p.z}` : ""}: ${bad ? "DIFFERS" : "ok"}: ${written.map(b => `${b} ${verdict[b]}`).join(", ")}${ex.length ? ` e.g. ${ex.join(" ")}` : ""}`);
    emit({ kind: "exact", placement: i, label: p.kernel.label, z: p.z, written, verdict, examples: ex });
  }
  console.log(`exactness: ${changed.length - exactFail} of ${changed.length} changed placements identical${exactFail ? `; ${exactFail} DIFFER` : ""}`);
}

// ---- timing: the whole step from S0, interleaved ----
if (process.env.NOTIME) { console.log(gpuErrors ? `GPU ERRORS ${gpuErrors}` : "ok"); s.destroy(); process.exit(0); }
const probe = createClockProbe(raw, { capacity: 2 });
const clock = async () => { const enc = raw.createCommandEncoder(); probe.encode(enc); raw.queue.submit([enc.finish()]); const [us] = await probe.read(); return us!; };
// A timer label is "<pass> / nb-brick.<kernel label>:<entry>": the family is the kernel label.
const family = (l: string) => /nb-brick\.([^:/]+):/.exec(l.split(" / ").at(-1) ?? l)?.[1] ?? l;
// S0 on the GPU: a mirror of every engine buffer, copied back before each replayed step inside the same submit (a
// host restore uploads GBs between steps, and its per-kernel times then spread 1-3x: 2026-11-24 first rounds).
// e.run compiles the placements' pipelines the engine lacks (the candidate's kernels), then S0 is put back.
await e.run(candPl); await e.restore(S0);
const mirrors = [...e.buffers as Map<string, GPUBuffer>].map(([name, b]) => {
  const m = raw.createBuffer({ label: `stepab S0 ${name}`, size: b.size, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  return [m, b] as const;
});
// Filled from the live buffers, which hold S0 just after the restore (a host write of GBs aborts in Dawn: uploadWords).
{ const enc = raw.createCommandEncoder({ label: "stepab S0 mirror" }); for (const [m, b] of mirrors) enc.copyBufferToBuffer(b, 0, m, 0, b.size); raw.queue.submit([enc.finish()]); }
await raw.queue.onSubmittedWorkDone();
const reset = () => { const enc = raw.createCommandEncoder({ label: "stepab S0" }); for (const [m, b] of mirrors) enc.copyBufferToBuffer(m, 0, b, 0, b.size); return enc.finish(); };
const stepCB = (pl: any[]) => e.encode(pl.map((p: any) => e.bind(p, 0)), "nb-brick.run");
// R replays a sample, each from S0: [S0 copy, step] x R in one submit, once untimed and once timed (gpu-kernel-timer);
// then R walls, each the step's submit to done after its own S0 copy.
const time = async (pl: any[]) => {
  const many = () => Array.from({ length: R }, () => [reset(), stepCB(pl)]).flat();
  raw.queue.submit(many()); await raw.queue.onSubmittedWorkDone();
  timer.enabled = true; const cbs = many(); timer.enabled = false;
  raw.queue.submit(cbs); await raw.queue.onSubmittedWorkDone();
  const { byLabel } = await timer.collect(); const fam = new Map<string, number>();
  for (const [l, ms] of byLabel) fam.set(family(l), (fam.get(family(l)) ?? 0) + ms / R);
  let wall = Infinity;
  for (let r = 0; r < R; r++) { raw.queue.submit([reset()]); const cb = stepCB(pl); await raw.queue.onSubmittedWorkDone(); const t0 = performance.now(); raw.queue.submit([cb]); await raw.queue.onSubmittedWorkDone(); wall = Math.min(wall, performance.now() - t0); }
  return { sum: [...fam.values()].reduce((a, b) => a + b, 0), fam, wall };
};
const sets = [{ name: "base", pl: basePl }, { name: "candidate", pl: candPl }, { name: "base again", pl: basePl }];
const res = new Map(sets.map(x => [x.name, { sum: [] as number[], wall: [] as number[], probe: [] as number[], fam: new Map<string, number[]>() }]));
const dropped: string[] = [];
for (let t = 0; t < SAMPLES; t++) for (const set of [...sets.slice(t % 3), ...sets.slice(0, t % 3)]) {
  const r = await time(set.pl), pr = await clock(), o = res.get(set.name)!;
  // A probe above 1.3x the top clock's (a low clock, clock-probe.ts) drops the sample from every statistic.
  if (pr > 1.3 * TOP_CLOCK_PROBE_US) { dropped.push(`${set.name} sample ${t} (probe ${pr.toFixed(1)} us)`); continue; }
  o.sum.push(r.sum); o.wall.push(r.wall); o.probe.push(pr);
  for (const [f, ms] of r.fam) { if (!o.fam.has(f)) o.fam.set(f, []); o.fam.get(f)!.push(ms); }
}
const min = (v: number[]) => Math.min(...v), med = (v: number[]) => [...v].sort((x, y) => x - y)[v.length >> 1]!;
if (dropped.length) console.log(`dropped at a low clock: ${dropped.join(", ")}`);
const sgn = (x: number, d = 3) => `${x >= 0 ? "+" : ""}${x.toFixed(d)}`;
const B = res.get("base")!, C = res.get("candidate")!, A = res.get("base again")!;
const famMin = (r: typeof B, f: string) => r.fam.has(f) ? min(r.fam.get(f)!) : 0;
const names = [...new Set([...B.fam.keys(), ...C.fam.keys(), ...A.fam.keys()])];
const famSum = (r: typeof B) => names.reduce((a, f) => a + famMin(r, f), 0);
for (const [name, r] of res) {
  console.log(`${name}: Σ placements min ${min(r.sum).toFixed(3)} ms, median ${med(r.sum).toFixed(3)}; Σ of family minima ${famSum(r).toFixed(3)} ms; GPU wall min ${min(r.wall).toFixed(3)} ms, median ${med(r.wall).toFixed(3)}; clock probe ${r.probe.map(x => x.toFixed(1)).join(" ")} us (top ${TOP_CLOCK_PROBE_US})`);
  emit({ kind: "time", variant: name, sumMinMs: min(r.sum), sumMedianMs: med(r.sum), sumMs: r.sum, famSumMs: famSum(r), wallMinMs: min(r.wall), wallMs: r.wall, probeUs: r.probe, fam: Object.fromEntries(r.fam) });
}
const fams = names.map(f => { const b = famMin(B, f), c = famMin(C, f), a = famMin(A, f); return { f, b, c, d: c - b, aa: a - b, changed: changedLabels.includes(f) }; }).sort((x, y) => Math.abs(y.d) - Math.abs(x.d));
const dChanged = fams.filter(x => x.changed).reduce((a, x) => a + x.d, 0), aaChanged = fams.filter(x => x.changed).reduce((a, x) => a + x.aa, 0);
const dOther = fams.filter(x => !x.changed).reduce((a, x) => a + x.d, 0), aaOther = fams.filter(x => !x.changed).reduce((a, x) => a + x.aa, 0);
const bS = famSum(B);
console.log("families (min of samples; A/A = base again - base), the changed ones and the 8 largest other |Δ|:");
for (const x of [...fams.filter(x => x.changed), ...fams.filter(x => !x.changed).slice(0, 8)])
  console.log(`  ${x.changed ? "*" : " "} ${x.f.padEnd(22)} ${x.b.toFixed(3)} -> ${x.c.toFixed(3)} ms  ${sgn(x.d)} (${x.b ? (100 * (x.c / x.b - 1)).toFixed(1) : "-"}%)  A/A ${sgn(x.aa)}`);
console.log(`GATE ${sceneKey}: changed families Σ Δ ${sgn(dChanged)} ms (${sgn(100 * dChanged / bS, 2)}% of the step's ${bS.toFixed(3)} ms Σ), their A/A ${sgn(aaChanged)} ms; unchanged families Σ Δ ${sgn(dOther)} ms, A/A ${sgn(aaOther)} ms; Σ placements min ${sgn(min(C.sum) - min(B.sum))} ms (A/A ${sgn(min(A.sum) - min(B.sum))}); GPU wall min ${sgn(min(C.wall) - min(B.wall))} ms (${sgn(100 * (min(C.wall) / min(B.wall) - 1), 2)}%; A/A ${sgn(min(A.wall) - min(B.wall))})`);
emit({ kind: "gate", families: fams, dChangedMs: dChanged, aaChangedMs: aaChanged, dOtherMs: dOther, aaOtherMs: aaOther, famSumBaseMs: bS,
  dSumMs: min(C.sum) - min(B.sum), aaSumMs: min(A.sum) - min(B.sum), dWallMs: min(C.wall) - min(B.wall), aaWallMs: min(A.wall) - min(B.wall) });
emit({ kind: "done", gpuErrors, exactFail }); console.log(gpuErrors ? `GPU ERRORS ${gpuErrors}` : "ok");
s.destroy(); process.exit(gpuErrors ? 1 : 0);
