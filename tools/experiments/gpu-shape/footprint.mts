// GPU-shape M0c (lane B): clock state inside real steps, the frozen state's slots, and the hardware-counter loop
// (docs/plans/gpu-shape-2026-11-23.md §M0c, docs/plans/gpu-shape/m0c-footprint.md).
//
// MODE=probe (default; a GPU run, exclusive):
//   tools/gpu/fluid-gpu-run.sh timing node --import tsx tools/experiments/gpu-shape/footprint.mts
//   env SCENE=seep|dam|letters|paddleboats|teaser  STEPS=160  OUT=<dir> (default ~/code/me/fluid-scratch/gpu-shape/m0c)
//   Runs STEPS normal back-to-back steps (advanceTo, no fence, no replay) with a clock probe submitted before each
//   step's part 1, between part 1 and part 2, and after part 2 (and the rigid integrator). The probe is lane A's
//   (sweep.mts): one 32-lane workgroup, a 2048-step dependent FMA chain, about 75.5 us at the top clock, alone in a
//   timestamped pass. It binds the engine's `set` buffer read-only, which every part writes, so Metal's hazard
//   tracking orders it after the part before it and the part after it waits for it. Then, fenced, every engine's
//   `set` and `dir` buffers are dumped with the layout, the buffer sizes and the schedule's kernels (hash, bindings),
//   for the offline analysis (footprint-analyze.mts).
// MODE=counters (the child of footprint-ctr.mts, which attaches Instruments' Metal GPU Counters):
//   warm-up STEPS, then for LOOP_MS split-pass steps (tools/gpu-kernel-timer.ts: each dispatch its own encoder,
//   labelled `<pass>|<pipeline>`) each followed by four calibration chases (lane A's kernels.mts chase, L256 D16 M1):
//   random over 8, 32 and 256 MB, and coalesced over 8 MB, each alone in a pass labelled `cal.<name>` and timed.
import { resolve, dirname } from "node:path";
import { pathToFileURL, fileURLToPath } from "node:url";
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import { execFileSync } from "node:child_process";
import { homedir } from "node:os";

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, "../../..");
const scratch = resolve(homedir(), "code/me/fluid-scratch/gpu-shape");
const kcache = resolve(scratch, "kcache");
const SCENES: Record<string, string> = { seep: "reservoir-seep", dam: "high-resolution-dam-break", letters: "nbflip-figure-8-letters", paddleboats: "paddleboats", teaser: "nbflip-figure-9-teaser" };

await import(`${root}/lib/methods/index.ts`);
const { getMethod } = await import(`${root}/lib/core/method-registry.ts`);
const { resolveMethodValues } = await import(`${root}/lib/core/method-contract.ts`);
const { sceneDocument } = await import(`${root}/lib/core/scene-definition.ts`);
const { getSceneDefinition } = await import(`${root}/lib/core/scenes.ts`);
const { managedGPUDevice } = await import(`${root}/lib/core/gpu-compilation-manager.ts`);
const { requiredFluidDeviceLimits } = await import(`${root}/lib/core/webgpu-device-limits.ts`);
const { createProcessRetainedDawnGPU, paceDawnEventPolling } = await import(`${root}/lib/harness/node-dawn-provider.ts`);
const { mooredMotorBodies } = await import(`${root}/tools/moored-motor-bodies.ts`);
const { gpuKernelTimer } = await import(`${root}/tools/gpu-kernel-timer.ts`);

const MODE = process.env.MODE ?? "probe";
const sceneKey = process.env.SCENE ?? "dam";
const sceneId = SCENES[sceneKey] ?? sceneKey;
const N = Number(process.env.STEPS ?? (MODE === "probe" ? 160 : 100));
const detailEnv = process.env.DETAIL ?? (sceneKey === "teaser" ? "2" : "");
const outDir = resolve(process.env.OUT ?? resolve(scratch, "m0c"), sceneKey);
mkdirSync(outDir, { recursive: true }); mkdirSync(kcache, { recursive: true });
const methodId = "uniform-narrow-band-flip", method = getMethod(methodId);
const definition = getSceneDefinition(sceneId), profile = definition.methodProfile, base = sceneDocument(definition);
const [df, db] = detailEnv.split(":").map(Number);
const scene0 = df ? { ...base, fluid: { ...base.fluid, automaticDetail: { factor: df as 2 | 4, ...(db ? { budget: db } : {}) } } } : base;
const scene = { ...scene0, numerics: { ...scene0.numerics, fixedDt_s: 1 / 60, maxDt_s: 1 / 60 } };
const quality = profile?.methodId === methodId ? profile.quality : "balanced";
const values = resolveMethodValues(method, quality, { ...method.appDefaults, ...(profile?.methodId === methodId ? profile.overrides : {}), engine: "brick" });
const log = (...a: unknown[]) => console.log(...a);
log(`footprint ${MODE} scene ${sceneId} (${sceneKey}) steps ${N}, detail ${detailEnv || "off"}, out ${outDir}`);

paceDawnEventPolling();
const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE ?? resolve(root, "node_modules/webgpu/index.js")).href);
Object.assign(globalThis, dawn.globals);
const adapter = await createProcessRetainedDawnGPU(dawn, ["backend=metal", "disable-dawn-features=timestamp_quantization",
  ...(process.env.FLUID_WEBGPU_DAWN_FEATURES ? [`enable-dawn-features=${process.env.FLUID_WEBGPU_DAWN_FEATURES}`] : [])]).requestAdapter();
const raw: GPUDevice = await adapter.requestDevice({ requiredFeatures: ["subgroups", "float32-filterable", "timestamp-query"].filter((f: string) => adapter.features.has(f)), requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
const gpuErrors: string[] = [];
raw.addEventListener?.("uncapturederror", (ev: any) => { gpuErrors.push(String(ev.error?.message)); console.log("GPU ERROR", ev.error?.message); });
const timer = MODE === "counters" ? gpuKernelTimer(raw) : undefined;
const device = managedGPUDevice(timer ? timer.device : raw, { requireWorkerRealm: false });
const s: any = await method.createSolverAsync!(device, scene, quality, values, undefined, () => {});
await s.waitForSimulationReady?.();
s.setPresentationEnabled(false);
const dt = 1 / 60;
const bodies: (t: number) => unknown[] = sceneKey === "paddleboats" || (scene.rigidBodies?.length ?? 0) > 0 ? mooredMotorBodies(scene, dt) : () => [];
const P = s.brickEngine;
const engines: { name: string; e: any }[] = [{ name: "parent", e: P }, ...(s.details ?? []).map((d: any, i: number) => ({ name: i ? `fine${i}` : "fine", e: d.engine }))];
let k = 0;
async function step() {
  k++;
  while (!s.advanceTo(k * dt + 1e-12, bodies(k * dt))) {
    if (s.info.simulationFailure) throw new Error(`step ${k}: ${s.info.simulationFailure.message}`);
    await new Promise<void>(r => setImmediate(r));
  }
}
const U = GPUBufferUsage;
const git = (() => { try { return { head: execFileSync("git", ["rev-parse", "--short", "HEAD"], { cwd: root, encoding: "utf8" }).trim(), status: execFileSync("git", ["status", "--short"], { cwd: root, encoding: "utf8" }) }; } catch { return null; } })();

if (MODE === "counters") await countersLoop();
else await probeRun();
s.destroy?.();
process.exit(gpuErrors.length ? 3 : 0);

// ---------------------------------------------------------------------------------------------------------------
async function probeRun() {
  // Lane A's probe (sweep.mts aluWGSL(32), PROBE_ITERS 2048), plus the read-only `set` binding that orders it.
  const code = `@group(0) @binding(0) var<storage, read_write> o: array<f32>;
@group(0) @binding(1) var<uniform> it: vec4u;
@group(0) @binding(2) var<storage, read> dep: array<u32>;
@compute @workgroup_size(32) fn main(@builtin(global_invocation_id) g: vec3u) {
  var x = f32(g.x) * 1e-6; var y = x + 1.0;
  if (dep[g.x] == 0x7fc00001u) { x = x + 1.0; }
  for (var i = 0u; i < it.x; i++) { x = fma(x, 0.999, 0.25); y = fma(y, x, -0.5); }
  o[g.x] = x + y;
}`;
  const pipe = raw.createComputePipeline({ layout: "auto", compute: { module: raw.createShaderModule({ code }), entryPoint: "main" } });
  const out = raw.createBuffer({ size: 4096, usage: U.STORAGE });
  const it = raw.createBuffer({ size: 16, usage: U.UNIFORM | U.COPY_DST }); raw.queue.writeBuffer(it, 0, new Uint32Array([2048, 0, 0, 0]));
  const groups = new Map<any, GPUBindGroup>();
  for (const { e } of engines) groups.set(e, raw.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: out } }, { binding: 1, resource: { buffer: it } }, { binding: 2, resource: { buffer: e.buffers.get("set"), offset: 0, size: 4096 } }] }));
  const QMAX = 4096, qs = raw.createQuerySet({ type: "timestamp", count: QMAX });
  const probes: { step: number; engine: string; tag: string; q: number }[] = [];
  let q = 0, armed = false;
  // Tag each engine's command buffers by their encode label (the engine's own `encode`, wrapped from outside).
  const tags = new WeakMap<object, { engine: any; name: string; label: string }>();
  for (const { name, e } of engines) {
    const enc0 = e.encode;
    e.encode = function (list: unknown, label: string, receipt?: number) { const cb = enc0.call(this, list, label, receipt); tags.set(cb, { engine: e, name, label }); return cb; };
  }
  // (Each engine encoded its first step ahead before this wrap: that step goes unprobed.)
  const queue: any = raw.queue, submit0 = queue.submit.bind(queue);
  if (device.queue !== raw.queue) throw new Error("the solver's queue is not the raw queue");
  const probe = (e: any, name: string, tag: string) => {
    if (!armed || q + 2 > QMAX) return;
    const enc = raw.createCommandEncoder();
    const pass = enc.beginComputePass({ label: `clock.${tag}`, timestampWrites: { querySet: qs, beginningOfPassWriteIndex: q, endOfPassWriteIndex: q + 1 } });
    pass.setPipeline(pipe); pass.setBindGroup(0, groups.get(e)!); pass.dispatchWorkgroups(1); pass.end();
    submit0([enc.finish()]);
    probes.push({ step: e.steps, engine: name, tag, q }); q += 2;
  };
  queue.submit = (cbs: GPUCommandBuffer[]) => {
    const t = cbs.map(cb => tags.get(cb)).find(x => x);
    if (t && /\.part1$/.test(t.label)) probe(t.engine, t.name, "start");
    submit0(cbs);
    if (t && /\.part1$/.test(t.label)) probe(t.engine, t.name, "mid");
    if (t && /\.part2$/.test(t.label)) probe(t.engine, t.name, "end");
  };
  const t0 = performance.now(); const seen = new Set<string>();
  armed = true;
  for (let j = 1; j <= N; j++) {
    await step();
    if (j % 40 === 0) log(`step ${j} ${((performance.now() - t0) / j).toFixed(1)} ms/step avg (free-running), probes ${probes.length}`);
  }
  armed = false;
  await P.fence();
  queue.submit = submit0;
  for (const p of probes) seen.add(`${p.engine}.${p.tag}`);
  log(`probes: ${probes.length} (${[...seen].join(", ")})`);
  if (s.info.simulationFailure) throw new Error(s.info.simulationFailure.message);
  const resolveBuf = raw.createBuffer({ size: 8 * QMAX, usage: U.QUERY_RESOLVE | U.COPY_SRC }), read = raw.createBuffer({ size: 8 * QMAX, usage: U.MAP_READ | U.COPY_DST });
  const enc = raw.createCommandEncoder(); enc.resolveQuerySet(qs, 0, q, resolveBuf, 0); enc.copyBufferToBuffer(resolveBuf, 0, read, 0, 8 * q); submit0([enc.finish()]);
  await read.mapAsync(GPUMapMode.READ); const ts = new BigUint64Array(read.getMappedRange(0, 8 * q).slice(0)); read.unmap();
  const t00 = ts[0]!;
  const rows = ["step\tengine\ttag\tbegin_us\tend_us\tprobe_us"];
  for (const p of probes) rows.push([p.step, p.engine, p.tag, (Number(ts[p.q]! - t00) / 1e3).toFixed(2), (Number(ts[p.q + 1]! - t00) / 1e3).toFixed(2), (Number(ts[p.q + 1]! - ts[p.q]!) / 1e3).toFixed(2)].join("\t"));
  writeFileSync(resolve(outDir, "probes.tsv"), rows.join("\n") + "\n");
  // The frozen state: set, dir, layout, sizes and the schedule's kernels, a file set an engine.
  const meta: any = { scene: sceneId, key: sceneKey, steps: N, date: new Date().toISOString(), git, gpuErrors, engines: [] };
  for (const { name, e } of engines) {
    const set = await e.read("set"), dir = await e.read("dir");
    writeFileSync(resolve(outDir, `${name}.set.bin`), Buffer.from(set.buffer)); writeFileSync(resolve(outDir, `${name}.dir.bin`), Buffer.from(dir.buffer));
    const L = e.layout;
    const sizes = (m: Map<string, GPUBuffer>) => Object.fromEntries([...m].map(([n, b]) => [n, b.size]));
    const kern = (p: any) => {
      const hash = createHash("sha1").update(p.kernel.entry + "\n" + p.kernel.code).digest("hex").slice(0, 16);
      const f = resolve(kcache, `${hash}.wgsl`); if (!existsSync(f)) { writeFileSync(f, p.kernel.code); writeFileSync(resolve(kcache, `${hash}.entry`), p.kernel.entry); }
      return { label: p.kernel.label, entry: p.kernel.entry, hash, z: p.z ?? 0, w: p.w ?? 0, bindings: p.kernel.bindings.map((b: any) => [b.name, b.kind]), dispatch: "direct" in p.dispatch ? p.dispatch.direct : `indirect@${p.dispatch.indirect}` };
    };
    meta.engines.push({ name, steps: e.steps, lattice: e.options.lattice, cap: L.cap, set: L.set, dir: L.dir, window: L.window, poolWords: L.poolWords, particleStride: L.particleStride, partsWords: L.partsWords, bytes: L.bytes,
      buffers: sizes(e.buffers), external: sizes(e.external), schedule: { part1: e.schedule.part1.map(kern), part2: e.schedule.part2.map(kern) } });
    log(`${name}: dumped set ${set.length} words, dir ${dir.length} words; buffers ${(Object.values(sizes(e.buffers)).reduce((a: number, b: any) => a + b, 0) / 2 ** 20).toFixed(0)} MiB`);
  }
  writeFileSync(resolve(outDir, "meta.json"), JSON.stringify(meta, null, 1));
  log(`wrote ${outDir}: probes.tsv, meta.json, *.set.bin, *.dir.bin; gpu errors ${gpuErrors.length}`);
}

// ---------------------------------------------------------------------------------------------------------------
async function countersLoop() {
  const { chaseWGSL } = await import(`${here}/kernels.mts`);
  const t0 = performance.now();
  for (let j = 1; j <= N; j++) { await step(); if (j % 20 === 0) { await P.fence(); log(`warm-up ${j} ${((performance.now() - t0) / j).toFixed(1)} ms/step`); } }
  await P.fence();
  // Calibration chases: Sattolo single-cycle permutations (random), and a block permutation (coalesced).
  const sattolo = (n: number) => { const p = new Uint32Array(n); for (let i = 0; i < n; i++) p[i] = i; for (let i = n - 1; i > 0; i--) { const j = Math.floor(Math.random() * i); const t = p[i]!; p[i] = p[j]!; p[j] = t; } return p; };
  const cycleOf = (perm: Uint32Array) => { const n = perm.length, nxt = new Uint32Array(n); for (let i = 0; i < n; i++) nxt[perm[i]!] = perm[(i + 1) % n]!; return nxt; };
  const upload = (a: Uint32Array) => { const b = raw.createBuffer({ size: a.byteLength, usage: U.STORAGE | U.COPY_DST }); raw.queue.writeBuffer(b, 0, a.buffer as ArrayBuffer); return b; };
  const block = (n: number) => { const nb = n / 32, pb = cycleOf(sattolo(nb)), a = new Uint32Array(n); for (let b = 0; b < nb; b++) for (let j = 0; j < 32; j++) a[32 * b + j] = 32 * pb[b]! + j; return a; };
  const outb = raw.createBuffer({ size: 16 << 20, usage: U.STORAGE });
  const rnd = raw.createComputePipeline({ layout: "auto", compute: { module: raw.createShaderModule({ code: chaseWGSL({ L: 256, M: 1 }) }), entryPoint: "main" } });
  const coal = raw.createComputePipeline({ layout: "auto", compute: { module: raw.createShaderModule({ code: chaseWGSL({ L: 256, M: 1, C: true }) }), entryPoint: "main" } });
  const cal: { name: string; pipe: GPUComputePipeline; bg: GPUBindGroup; G: number; lanesLoads: number }[] = [];
  const add = (name: string, pipe: GPUComputePipeline, words: number, nxt: GPUBuffer, G: number) => {
    const prm = raw.createBuffer({ size: 16, usage: U.UNIFORM | U.COPY_DST }); raw.queue.writeBuffer(prm, 0, new Uint32Array([words - 1, 16, 12345, 0]));
    cal.push({ name, pipe, G, lanesLoads: 256 * G * 16, bg: raw.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: nxt } }, { binding: 1, resource: { buffer: outb } }, { binding: 2, resource: { buffer: prm } }] }) });
  };
  const W8 = 2 << 20, W32 = 8 << 20, W256 = 64 << 20;
  add("rand8M", rnd, W8, upload(cycleOf(sattolo(W8))), 1024);
  add("rand32M", rnd, W32, upload(cycleOf(sattolo(W32))), 1024);
  add("rand256M", rnd, W256, upload(cycleOf(sattolo(W256))), 192);
  add("coal8M", coal, W8, upload(block(W8)), 8192);
  const qs = raw.createQuerySet({ type: "timestamp", count: 2 * cal.length });
  const rb = raw.createBuffer({ size: 16 * cal.length, usage: U.QUERY_RESOLVE | U.COPY_SRC });
  const calTimes = new Map<string, number[]>(cal.map(c => [c.name, []]));
  const LOOP_MS = Number(process.env.LOOP_MS ?? 9000);
  timer!.enabled = true;
  log("READY"); const tl = performance.now(); let steps = 0;
  while (performance.now() - tl < LOOP_MS) {
    await step(); steps++;
    const enc = raw.createCommandEncoder();
    cal.forEach((c, i) => { const p = enc.beginComputePass({ label: `cal.${c.name}`, timestampWrites: { querySet: qs, beginningOfPassWriteIndex: 2 * i, endOfPassWriteIndex: 2 * i + 1 } }); p.setPipeline(c.pipe); p.setBindGroup(0, c.bg); p.dispatchWorkgroups(c.G); p.end(); });
    const rd = raw.createBuffer({ size: 16 * cal.length, usage: U.MAP_READ | U.COPY_DST });
    enc.resolveQuerySet(qs, 0, 2 * cal.length, rb, 0); enc.copyBufferToBuffer(rb, 0, rd, 0, 16 * cal.length); raw.queue.submit([enc.finish()]);
    await rd.mapAsync(GPUMapMode.READ); const ts = new BigUint64Array(rd.getMappedRange().slice(0)); rd.unmap(); rd.destroy();
    cal.forEach((c, i) => calTimes.get(c.name)!.push(Number(ts[2 * i + 1]! - ts[2 * i]!) / 1e3));
    // Collect every 20 steps: the timer off for one step first, so no step encoded ahead still holds a timed
    // encoder's resolve buffers when collect() destroys them (that step's command buffers would fail validation).
    if (steps % 20 === 0) { timer!.enabled = false; await step(); steps++; await P.fence(); await timer!.collect(); timer!.enabled = true; }
  }
  timer!.enabled = false; await step(); await P.fence();
  const pr = await timer!.collect();
  log("DONE");
  const calOut = cal.map(c => { const v = calTimes.get(c.name)!.sort((a, b) => a - b); const med = v[v.length >> 1]!; return { name: c.name, G: c.G, lanesLoads: c.lanesLoads, min_us: v[0], med_us: med, GlaneLoadsPerS: c.lanesLoads / med / 1e3 }; });
  writeFileSync(resolve(outDir, "counters-loop.json"), JSON.stringify({ scene: sceneId, warm: N, loopSteps: steps, git, cal: calOut, dispatches: pr.dispatches, gpuErrors }, null, 1));
  log(JSON.stringify(calOut));
}
