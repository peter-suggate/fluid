// GPU-shape microbench (M0(a), docs/plans/gpu-shape/m0a-shape.md): time of one dispatch of a dependent-load chain
// kernel vs workgroups G x lanes L x chain depth D x chains per lane M x register cap x workgroup memory, plus the
// latency ladder, packing baseline and dispatch-boundary cost. One JSON line per measurement to <out.jsonl>.
//   tools/gpu/fluid-gpu-run.sh timing node --import tsx tools/experiments/gpu-shape/sweep.mts <out.jsonl> [phase,...]
// Phases: check ladder base reg wgmem pack boundary (default all). Every dispatch is timed in its own compute pass
// (timestampWrites at pass begin/end, Dawn's timestamp quantisation disabled); 1 warm + REPS timed passes per config
// in one command buffer; min and median reported. Work per command buffer is capped (lanes*M*D <= 2^25 loads per
// dispatch, and a combo stops widening once one dispatch exceeds 30 ms).
import { appendFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { chaseWGSL, chainedWGSL, kernelKey, packedWGSL, type ChaseKernel } from "./kernels.mjs";

const root = resolve(import.meta.dirname, "../../..");
const { createProcessRetainedDawnGPU, paceDawnEventPolling } = await import(`${root}/lib/harness/node-dawn-provider.ts`);
const { requiredFluidDeviceLimits } = await import(`${root}/lib/core/webgpu-device-limits.ts`);
const out = process.argv[2] ?? "gpu-shape.jsonl";
const phases = new Set((process.argv[3] ?? "check,dvfs,ladder,foot,base,reg,wgmem,coal,coalreg,coalwg,pack,boundary").split(","));
const REPS = Number(process.env.REPS ?? 4);
writeFileSync(out, "");
const emit = (o: Record<string, unknown>) => appendFileSync(out, JSON.stringify(o) + "\n");

paceDawnEventPolling();
const dawn = await import(pathToFileURL(resolve(root, "node_modules/webgpu/index.js")).href);
Object.assign(globalThis, dawn.globals);
const adapter = await createProcessRetainedDawnGPU(dawn, ["backend=metal", "disable-dawn-features=timestamp_quantization"]).requestAdapter();
const device: GPUDevice = await adapter.requestDevice({ requiredFeatures: ["timestamp-query"], requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
let gpuErrors = 0;
device.addEventListener?.("uncapturederror", (ev: any) => { gpuErrors++; console.log("GPU ERROR", ev.error?.message); emit({ kind: "gpuError", message: String(ev.error?.message) }); });
const info = (adapter as any).info ?? {};
emit({ kind: "setup", vendor: info.vendor, architecture: info.architecture, device: info.device, description: info.description, reps: REPS,
  maxWorkgroupStorage: device.limits.maxComputeWorkgroupStorageSize, maxInvocations: device.limits.maxComputeInvocationsPerWorkgroup });
console.log("adapter", info.vendor, info.architecture, info.description);

// ---- permutations: one random cycle (Sattolo) so every chain walks N distinct addresses ----
const U = GPUBufferUsage;
function permutation(bytes: number): GPUBuffer {
  const n = bytes / 4;
  const buf = device.createBuffer({ size: bytes, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC, mappedAtCreation: true });
  const a = new Uint32Array(buf.getMappedRange());
  for (let i = 0; i < n; i++) a[i] = i;
  // a as a cycle: visit order p, next[p[k]] = p[k+1].
  const p = new Uint32Array(n); for (let i = 0; i < n; i++) p[i] = i;
  for (let i = n - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); const t = p[i]!; p[i] = p[j]!; p[j] = t; }
  for (let k = 0; k < n; k++) a[p[k]!] = p[(k + 1) % n]!;
  buf.unmap();
  return buf;
}
// Block permutation for the coalesced chase: nxt[32b + j] = 32 next(b) + j, one random cycle over 128-byte blocks.
function blockPermutation(bytes: number): GPUBuffer {
  const n = bytes / 4, nb = n / 32;
  const buf = device.createBuffer({ size: bytes, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC, mappedAtCreation: true });
  const a = new Uint32Array(buf.getMappedRange());
  const p = new Uint32Array(nb); for (let i = 0; i < nb; i++) p[i] = i;
  for (let i = nb - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); const t = p[i]!; p[i] = p[j]!; p[j] = t; }
  for (let k = 0; k < nb; k++) { const from = p[k]! * 32, to = p[(k + 1) % nb]! * 32; for (let j = 0; j < 32; j++) a[from + j] = to + j; }
  buf.unmap();
  return buf;
}
const t0 = performance.now();
const SMALL = 8 << 20, BIG = 256 << 20;
const wantRandom = ["check", "ladder", "foot", "base", "reg", "regx", "regp", "regq", "wgmem", "pack", "boundary", "dvfs"].some(p => phases.has(p));
const perm = new Map<number, GPUBuffer>(wantRandom ? [[SMALL, permutation(SMALL)], [BIG, permutation(BIG)]] : []);
const wantBlock = [...phases].some(p => p.startsWith("coal"));
const bperm = new Map<number, GPUBuffer>(wantBlock ? [[SMALL, blockPermutation(SMALL)], [BIG, blockPermutation(BIG)]] : []);
console.log(`permutations ${(performance.now() - t0).toFixed(0)} ms`);
const MAXLANES = 1 << 20;
const outBuf = device.createBuffer({ size: 16 * MAXLANES, usage: U.STORAGE | U.COPY_SRC });
const readBuf = device.createBuffer({ size: 4 * 1024, usage: U.MAP_READ | U.COPY_DST });

// ---- pipelines, params, bind groups ----
const pipes = new Map<string, GPUComputePipeline>();
async function pipeline(key: string, code: string) {
  let p = pipes.get(key); if (p) return p;
  const module = device.createShaderModule({ code });
  for (const m of (await module.getCompilationInfo()).messages) if (m.type === "error") throw new Error(`${key}: ${m.lineNum}:${m.linePos} ${m.message}`);
  p = await device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "main" } });
  pipes.set(key, p); return p;
}
const chasePipe = (k: ChaseKernel) => pipeline(kernelKey(k), chaseWGSL(k));
const prmBufs = new Map<string, GPUBuffer>();
function prm(bytes: number, depth: number, seed = 1, base = 0) {
  const key = `${bytes}/${depth}/${seed}/${base}`; let b = prmBufs.get(key); if (b) return b;
  b = device.createBuffer({ size: 16, usage: U.UNIFORM | U.COPY_DST });
  device.queue.writeBuffer(b, 0, new Uint32Array([bytes / 4 - 1, depth, seed, base])); prmBufs.set(key, b); return b;
}
const bindChase = (p: GPUComputePipeline, nxt: GPUBuffer, bytes: number, depth: number, seed = 1, base = 0, o = outBuf) =>
  device.createBindGroup({ layout: p.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: nxt } }, { binding: 1, resource: { buffer: o } }, { binding: 2, resource: { buffer: prm(bytes, depth, seed, base) } }] });

// ---- timing: each "pass" is a list of dispatches in ONE compute pass with begin/end timestamps ----
type Disp = { pipe: GPUComputePipeline; bg: GPUBindGroup; G: number };
const QMAX = 512;
const qs = device.createQuerySet({ type: "timestamp", count: QMAX });
const qb = device.createBuffer({ size: 8 * QMAX, usage: U.QUERY_RESOLVE | U.COPY_SRC });
const qr = device.createBuffer({ size: 8 * QMAX, usage: U.MAP_READ | U.COPY_DST });
// Clock control. Apple's GPU clock follows load (DVFS), and a microbench that idles between command buffers runs at a
// low clock. Every timed command buffer therefore starts with a HEATER pass (a wide ALU kernel, about HEAT_US of
// work) and ends with a PROBE pass: one 32-lane workgroup running a dependent FMA chain, whose time is inversely
// proportional to the clock. Each record carries its probe time, so clock-state changes are visible in the data.
const aluWGSL = (L: number) => `@group(0) @binding(0) var<storage, read_write> o: array<f32>;
@group(0) @binding(1) var<uniform> it: vec4u;
@compute @workgroup_size(${L}) fn main(@builtin(global_invocation_id) g: vec3u) {
  var x = f32(g.x) * 1e-6; var y = x + 1.0;
  for (var i = 0u; i < it.x; i++) { x = fma(x, 0.999, 0.25); y = fma(y, x, -0.5); }
  o[g.x] = x + y;
}`;
const aluIt = (n: number) => { const b = device.createBuffer({ size: 16, usage: U.UNIFORM | U.COPY_DST }); device.queue.writeBuffer(b, 0, new Uint32Array([n, 0, 0, 0])); return b; };
const heatPipe = await pipeline("heat", aluWGSL(256)), probePipe = await pipeline("probe", aluWGSL(32));
// Heater and probe write outBuf (as every chase does), so Metal's hazard tracking orders them with the timed passes:
// with a buffer of their own, an independent encoder can overlap its neighbours and its timestamps mean nothing.
const aluBG = (p: GPUComputePipeline, n: number) => device.createBindGroup({ layout: p.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: outBuf } }, { binding: 1, resource: { buffer: aluIt(n) } }] });
let HEAT_US = Number(process.env.HEAT_US ?? 1500), heatIters = 2048;
let heatBG = aluBG(heatPipe, heatIters);
let bestProbe = Infinity;
const FILL = process.env.FILL !== "0";
const RAMP = Number(process.env.RAMP ?? 1.5);
const ONLY = process.env.BYTES ? Number(process.env.BYTES) << 20 : 0; // restrict the width sweeps to one footprint (MB)
const foots = (both: number[]) => both.filter(b => !ONLY || b === ONLY);
const PROBE_ITERS = 2048, probeBG = aluBG(probePipe, PROBE_ITERS);
let heatOn = HEAT_US > 0;

// shape: "pass" = all dispatches in one pass; "passes" = one pass per dispatch, timed from the first's begin to the last's end.
async function timeList(list: Disp[], shape: "pass" | "passes" = "pass", reps = REPS, heat = heatOn) {
  const enc = device.createCommandEncoder();
  let q = 0;
  if (heat) { const pass = enc.beginComputePass({ timestampWrites: { querySet: qs, beginningOfPassWriteIndex: q, endOfPassWriteIndex: q + 1 } }); pass.setPipeline(heatPipe); pass.setBindGroup(0, heatBG); pass.dispatchWorkgroups(256); pass.end(); }
  q += 2;
  for (let r = 0; r <= reps; r++) {
    if (shape === "pass") {
      const pass = enc.beginComputePass({ timestampWrites: { querySet: qs, beginningOfPassWriteIndex: q, endOfPassWriteIndex: q + 1 } });
      for (const d of list) { pass.setPipeline(d.pipe); pass.setBindGroup(0, d.bg); if (d.G > 0) pass.dispatchWorkgroups(d.G); }
      pass.end();
    } else {
      list.forEach((d, i) => {
        const tw: GPUComputePassTimestampWrites = { querySet: qs };
        if (i === 0) tw.beginningOfPassWriteIndex = q; if (i === list.length - 1) tw.endOfPassWriteIndex = q + 1;
        const pass = enc.beginComputePass(i === 0 || i === list.length - 1 ? { timestampWrites: tw } : {});
        pass.setPipeline(d.pipe); pass.setBindGroup(0, d.bg); pass.dispatchWorkgroups(d.G); pass.end();
      });
    }
    q += 2;
  }
  { const pass = enc.beginComputePass({ timestampWrites: { querySet: qs, beginningOfPassWriteIndex: q, endOfPassWriteIndex: q + 1 } }); pass.setPipeline(probePipe); pass.setBindGroup(0, probeBG); pass.dispatchWorkgroups(1); pass.end(); }
  q += 2;
  enc.resolveQuerySet(qs, 0, q, qb, 0); enc.copyBufferToBuffer(qb, 0, qr, 0, 8 * q);
  device.queue.submit([enc.finish()]);
  // Filler: a heater-only command buffer queued behind the measurement, so the GPU stays busy while the CPU reads the
  // timestamps and encodes the next measurement (an idle GPU drops its clock within a few ms).
  if (heat && FILL) { const e2 = device.createCommandEncoder(); const p2 = e2.beginComputePass(); p2.setPipeline(heatPipe); p2.setBindGroup(0, heatBG); p2.dispatchWorkgroups(256); p2.end(); device.queue.submit([e2.finish()]); }
  await qr.mapAsync(GPUMapMode.READ); const t = new BigUint64Array(qr.getMappedRange().slice(0, 8 * q)); qr.unmap();
  const span = (i: number) => Number(t[2 * i + 1]! - t[2 * i]!) / 1e3;
  const us: number[] = []; for (let r = 2; r <= reps + 1; r++) us.push(span(r)); // pair 0 heater, pair 1 the warm-up rep
  const s = [...us].sort((a, b) => a - b);
  return { min: s[0]!, med: s[Math.floor(s.length / 2)]!, all: us, heat: heat ? span(0) : 0, probe: span(reps + 2) };
}
// Ramp: sustained ALU load (~10 ms command buffers) until the probe stops improving, i.e. the clock is at its top
// state. Measured 2026-11-23: idle-then-short-bursts run at the probe's 247 us (~3.3x slower than the 76 us top state).
async function ramp(why: string) {
  const big = aluBG(heatPipe, heatIters * 16), keep = heatBG; heatBG = big;
  const t = performance.now(); let last = Infinity, stable = 0, r: Awaited<ReturnType<typeof timeList>> | undefined;
  while (performance.now() - t < 5000) {
    r = await timeList([], "pass", 0, true);
    if (r.probe > 0.97 * last) stable++; else stable = 0;
    last = Math.min(last, r.probe); if (stable >= 4 && performance.now() - t > 300) break;
  }
  heatBG = keep; bestProbe = Math.min(bestProbe, last);
  console.log(`ramp (${why}): ${(performance.now() - t).toFixed(0)} ms, probe ${last.toFixed(1)} us (best ${bestProbe.toFixed(1)})`);
  emit({ kind: "ramp", why, ms: performance.now() - t, probe: last });
}
// Timed measurement with clock control: when the probe says the clock fell (> 1.1x the best probe), ramp and retry.
async function measure(list: Disp[], shape: "pass" | "passes" = "pass", reps = REPS) {
  let r = await timeList(list, shape, reps);
  // Small dispatches pull the clock one or two states down on their own (probe 100-125 us vs 75); only the deep idle
  // state (probe > RAMP x best) is ramped away. Run 1 ramped at 1.1x and spent most of its time ramping.
  for (let tries = 0; heatOn && r.probe > RAMP * bestProbe && tries < 1; tries++) { await ramp(`probe ${r.probe.toFixed(0)}`); r = await timeList(list, shape, reps); }
  bestProbe = Math.min(bestProbe, r.probe);
  return r;
}
// DVFS evidence: the same small dispatch with no heater after idling (the low state), then after a ramp.
if (phases.has("dvfs")) {
  const p = await chasePipe({ L: 256, M: 1 });
  const d = [{ pipe: p, bg: bindChase(p, perm.get(SMALL)!, SMALL, 16), G: 40 }];
  await new Promise(r => setTimeout(r, 500));
  const cold = await timeList(d, "pass", REPS, false);
  await ramp("dvfs");
  const hot = await timeList(d, "pass", REPS, false);
  emit({ kind: "dvfs", coldMin: cold.min, coldProbe: cold.probe, hotMin: hot.min, hotProbe: hot.probe });
  console.log(`dvfs: cold chase ${cold.min.toFixed(1)} us probe ${cold.probe.toFixed(1)}; after ramp chase ${hot.min.toFixed(1)} us probe ${hot.probe.toFixed(1)}`);
}
if (heatOn) {
  await ramp("start");
  // Calibrate the per-measurement heater to about HEAT_US at the top clock.
  for (let i = 0; i < 2; i++) {
    const r = await timeList([], "pass", 0, true);
    heatIters = Math.max(64, Math.round(heatIters * HEAT_US / Math.max(r.heat, 1)));
    heatBG = aluBG(heatPipe, heatIters);
    console.log(`heater ${r.heat.toFixed(0)} us -> iters ${heatIters}; probe ${r.probe.toFixed(1)} us`);
  }
  emit({ kind: "heater", heatIters, HEAT_US });
}

const pow2 = (lo: number, hi: number) => { const a: number[] = []; for (let g = lo; g <= hi; g *= 2) a.push(g); return a; };
const EXTRA = [24, 31, 33, 40, 48, 63, 65, 80, 96, 97, 127, 129, 160, 192, 255, 257];
const LOADCAP = 1 << 25, TSTOP = 30_000;
async function widthSweep(tag: string, k: ChaseKernel, bytes: number, D: number, extras: boolean) {
  const p = await chasePipe(k);
  // Coalesced chases make 1/32 the line requests, so they get 4x the lanes and loads before the cap.
  const gmax = Math.min(k.C ? 4 * MAXLANES : MAXLANES, Math.floor((k.C ? 4 : 1) * LOADCAP / (k.M * Math.max(D, 1)))) / k.L;
  const Gs = [...new Set([...pow2(1, Math.min(gmax, 32768)), ...(extras ? EXTRA.filter(g => g <= gmax) : [])])].sort((a, b) => a - b);
  for (const G of Gs) {
    const r = await measure([{ pipe: p, bg: bindChase(p, (k.C ? bperm : perm).get(bytes)!, bytes, D), G }]);
    emit({ kind: tag, ...k, R: k.R ?? 0, WGB: k.WGB ?? 0, C: !!k.C, X: !!k.X, P: !!k.P, Q: !!k.Q, bytes, D, G, lanes: G * k.L, min: r.min, med: r.med, probe: r.probe });
    if (r.min > TSTOP) break;
  }
}

// ---- check: the chase computes what the CPU says (catches a GPU that executes nothing) ----
if (phases.has("check")) {
  const hsh = (x0: number) => { let x = (Math.imul(x0 >>> 0, 747796405) + 2891336453) >>> 0; x = Math.imul(((x >>> ((x >>> 28) + 4)) ^ x) >>> 0, 277803737) >>> 0; return ((x >>> 22) ^ x) >>> 0; };
  const n = 1 << 12, buf = permutation(4 * n);
  const host = device.createBuffer({ size: 4 * n, usage: U.MAP_READ | U.COPY_DST });
  { const e = device.createCommandEncoder(); e.copyBufferToBuffer(buf, 0, host, 0, 4 * n); device.queue.submit([e.finish()]); }
  await host.mapAsync(GPUMapMode.READ); const nx = new Uint32Array(host.getMappedRange().slice(0)); host.unmap();
  const k = { L: 64, M: 2 }, p = await chasePipe(k), D = 37;
  const chk = device.createBuffer({ size: 4 * 256, usage: U.STORAGE | U.COPY_SRC });
  await measure([{ pipe: p, bg: bindChase(p, buf, 4 * n, D, 5, 0, chk), G: 4 }], "pass", 0);
  { const e = device.createCommandEncoder(); e.copyBufferToBuffer(chk, 0, readBuf, 0, 4 * 256); device.queue.submit([e.finish()]); }
  await readBuf.mapAsync(GPUMapMode.READ); const got = new Uint32Array(readBuf.getMappedRange().slice(0)); readBuf.unmap();
  let bad = 0;
  for (let lane = 0; lane < 256; lane++) {
    let v = 0; for (let m = 0; m < 2; m++) { let i = hsh(lane * 2 + m + 5) & (n - 1); for (let d = 0; d < D; d++) i = nx[i]!; v ^= i; }
    if ((v >>> 0) !== got[lane]) bad++;
  }
  console.log(`check: ${bad} of 256 lanes wrong`); emit({ kind: "check", bad });
  if (bad) process.exit(3);
}

// ---- ladder: one simdgroup, one chain: latency per dependent load vs footprint ----
if (phases.has("ladder")) {
  const p = await chasePipe({ L: 32, M: 1 });
  for (const bytes of [16 << 10, 128 << 10, 1 << 20, 4 << 20, 8 << 20, 32 << 20, 64 << 20, 128 << 20, 256 << 20]) {
    const buf = perm.get(bytes) ?? permutation(bytes);
    const r: Record<number, number> = {};
    for (const D of [256, 1024]) r[D] = (await measure([{ pipe: p, bg: bindChase(p, buf, bytes, D), G: 1 }], "pass", 6)).min;
    const ns = (r[1024]! - r[256]!) * 1e3 / 768;
    console.log(`ladder ${(bytes / 1024).toFixed(0)} KB: ${ns.toFixed(1)} ns/load`); emit({ kind: "ladder", bytes, t256: r[256], t1024: r[1024], nsPerLoad: ns });
    // Saturated random-load throughput at this footprint (wide, several chains a lane).
    if (phases.has("foot")) for (const [M, G] of [[1, 256], [4, 64], [4, 256]] as const) {
      const pf = await chasePipe({ L: 256, M });
      const f = await measure([{ pipe: pf, bg: bindChase(pf, buf, bytes, 16), G }]);
      const loads = 256 * G * M * 16;
      emit({ kind: "foot", bytes, L: 256, M, G, D: 16, min: f.min, probe: f.probe, gLoadsPerS: loads / f.min / 1e3 });
      console.log(`  foot M${M} G${G}: ${f.min.toFixed(0)} us, ${(loads / f.min / 1e3).toFixed(2)} G loads/s`);
    }
    if (!perm.has(bytes)) buf.destroy();
  }
  // single-dispatch floor vs depth: one workgroup of 32 lanes
  for (const bytes of [SMALL, BIG]) for (const D of [0, 1, 4, 16, 64]) {
    const r = await measure([{ pipe: p, bg: bindChase(p, perm.get(bytes)!, bytes, D), G: 1 }], "pass", 6);
    emit({ kind: "floor", bytes, D, min: r.min, med: r.med, probe: r.probe });
  }
  // (An empty pass writes no timestamps on Metal, so it cannot be timed: the D=0 single dispatch is the floor.)
  // A pass of 64 back-to-back empty-chain dispatches (D=0, one 32-lane workgroup): the per-dispatch floor without the pass.
  const z = await measure(Array.from({ length: 64 }, () => ({ pipe: p, bg: bindChase(p, perm.get(SMALL)!, SMALL, 0), G: 1 })), "pass", 4);
  emit({ kind: "floor64", min: z.min / 64, probe: z.probe });
}

const Ls = [32, 64, 128, 256, 512, 1024];
if (phases.has("base")) {
  for (const bytes of foots([SMALL, BIG])) for (const D of [1, 4, 16]) for (const M of [1, 2, 4]) for (const L of Ls) await widthSweep("base", { L, M }, bytes, D, D === 16);
  console.log("base done");
}
if (phases.has("reg")) {
  for (const bytes of [SMALL]) for (const R of [64, 96, 128]) for (const M of [1, 4]) for (const L of [64, 256]) await widthSweep("reg", { L, M, R }, bytes, 16, false);
  console.log("reg done");
}
if (phases.has("wgmem")) {
  for (const bytes of [SMALL]) for (const W of [1024, 8192, 16384, 24576]) for (const M of [1, 4]) for (const L of [64, 256, 1024]) await widthSweep("wgmem", { L, M, WGB: W }, bytes, 16, false);
  console.log("wgmem done");
}

// ---- coalesced chase: a simdgroup's 32 lanes share one line a link (the brick-like access pattern) ----
if (phases.has("coal")) {
  for (const bytes of foots([SMALL, BIG])) for (const D of [4, 16]) for (const M of [1, 4]) for (const L of [64, 256, 1024]) await widthSweep("coal", { L, M, C: true }, bytes, D, D === 16 && M === 1);
  console.log("coal done");
}
if (phases.has("coalreg")) {
  for (const bytes of [SMALL]) for (const R of [64, 96, 128]) for (const M of [1, 4]) for (const L of [64, 256]) await widthSweep("coalreg", { L, M, R, C: true }, bytes, 16, false);
  console.log("coalreg done");
}
// Register classes with cheap (xor-add) updates, caps verified with regcap.sh: R48 896/832, R64 704/640, R96 512/448,
// R128 384/384 (M1/M4, attribute stripped).
// regp/coalregp: pinned registers (loaded before the chain, combined once after it, no per-link work); same caps.
for (const [ph, C, P] of [["regx", false, false], ["coalregx", true, false], ["regp", false, true], ["coalregp", true, true]] as const) if (phases.has(ph)) {
  for (const R of [48, 64, 96, 128]) for (const M of [1, 4]) for (const L of [64, 256]) await widthSweep(ph, { L, M, R, X: !P, P, C }, SMALL, 16, false);
  console.log(`${ph} done`);
}
// regq/coalregq: recurrence registers, no per-link work, no extra loads. regcap.sh caps (M1 and M4 alike):
// R96 832, R128 640, R160 576, R192 448 (R >= 192 stays 448).
for (const [ph, C] of [["regq", false], ["coalregq", true]] as const) if (phases.has(ph)) {
  for (const R of [96, 128, 160, 192]) for (const M of [1, 4]) for (const L of [64, 256]) await widthSweep(ph, { L, M, R, Q: true, C }, SMALL, 16, false);
  console.log(`${ph} done`);
}
if (phases.has("coalwg")) {
  for (const bytes of [SMALL]) for (const W of [1024, 8192, 16384, 24576]) for (const M of [1, 4]) for (const L of [64, 256, 1024]) await widthSweep("coalwg", { L, M, WGB: W, C: true }, bytes, 16, false);
  console.log("coalwg done");
}

// ---- packing: k tiny bodies as k serial dispatches vs one dispatch of k x the workgroups ----
if (phases.has("pack")) {
  for (const bytes of [SMALL, BIG]) for (const [L, gpb, D] of [[64, 8, 16], [256, 8, 16], [64, 8, 4], [256, 40, 16]] as const) {
    const k1 = { L, M: 1 }, p = await chasePipe(k1);
    for (const k of [1, 2, 4, 8, 16, 32]) {
      const serial: Disp[] = []; for (let b = 0; b < k; b++) serial.push({ pipe: p, bg: bindChase(p, perm.get(bytes)!, bytes, D, 1 + b, b * gpb * L, outBuf), G: gpb });
      const s = await measure(serial, "pass");
      const sp = await measure(serial, "passes");
      const one = await measure([{ pipe: p, bg: bindChase(p, perm.get(bytes)!, bytes, D), G: gpb * k }], "pass");
      // k passes whose outputs are distinct buffers: no hazard between them, so Metal may overlap the encoders.
      const outs = Array.from({ length: k }, () => device.createBuffer({ size: 4 * gpb * L, usage: U.STORAGE }));
      const distinct = serial.map((d, b) => ({ ...d, bg: bindChase(p, perm.get(bytes)!, bytes, D, 1 + b, b * gpb * L, outs[b]!) }));
      const pd = await measure(distinct, "passes"), sd = await measure(distinct, "pass");
      outs.forEach(o => o.destroy());
      emit({ kind: "packSame", bytes, L, gpb, D, k, serial: s.min, serialPasses: sp.min, serialDistinct: sd.min, passesDistinct: pd.min, packed: one.min, probe: one.probe });
    }
  }
  // Two different bodies (different buffers and depths) in one module split on the workgroup index.
  for (const [L, gpb, DA, DB] of [[64, 8, 16, 16], [64, 8, 16, 4], [256, 8, 16, 16], [256, 40, 16, 16], [256, 160, 16, 16]] as const) {
    const k1 = { L, M: 1 }, p = await chasePipe(k1);
    const pk = await pipeline(`pack2_${L}_${gpb}`, packedWGSL(k1, 2, gpb));
    const A = { pipe: p, bg: bindChase(p, perm.get(SMALL)!, SMALL, DA, 1), G: gpb };
    const B = { pipe: p, bg: bindChase(p, perm.get(BIG)!, BIG, DB, 2), G: gpb };
    const bg = device.createBindGroup({ layout: pk.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: perm.get(SMALL)! } }, { binding: 1, resource: { buffer: perm.get(BIG)! } }, { binding: 2, resource: { buffer: outBuf } },
      { binding: 3, resource: { buffer: prm(SMALL, DA, 1) } }, { binding: 4, resource: { buffer: prm(BIG, DB, 2) } }] });
    const a = await measure([A]), b = await measure([B]), ab = await measure([A, B]);
    const packed = await measure([{ pipe: pk, bg, G: 2 * gpb }]);
    // The packed module with only one body's workgroups (cost of carrying the second body).
    const packedA = await measure([{ pipe: pk, bg, G: gpb }]);
    emit({ kind: "pack2", L, gpb, DA, DB, a: a.min, b: b.min, serial: ab.min, packed: packed.min, packedOnlyA: packedA.min });
  }
  console.log("pack done");
}

// ---- dispatch boundaries: n back-to-back dispatches, dependent (ping-pong) vs independent, one pass vs n passes ----
if (phases.has("boundary")) {
  for (const [L, G, D] of [[32, 1, 1], [64, 8, 16], [256, 40, 16], [256, 160, 4]] as const) {
    const p = await pipeline(`chained_${L}`, chainedWGSL(L));
    const lanes = L * G;
    const bufs = [0, 1, 2, 3, 4, 5, 6, 7].map(() => device.createBuffer({ size: 4 * lanes, usage: U.STORAGE }));
    const src = device.createBuffer({ size: 4 * lanes, usage: U.STORAGE });
    const bg = (i: GPUBuffer, o: GPUBuffer) => device.createBindGroup({ layout: p.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: perm.get(SMALL)! } }, { binding: 1, resource: { buffer: i } }, { binding: 2, resource: { buffer: o } }, { binding: 3, resource: { buffer: prm(SMALL, D) } }] });
    const dep: Disp[] = [], ind: Disp[] = [];
    const n = 64;
    for (let j = 0; j < n; j++) { dep.push({ pipe: p, bg: bg(bufs[j % 2]!, bufs[(j + 1) % 2]!), G }); ind.push({ pipe: p, bg: bg(src, bufs[j % 8]!), G }); }
    const one = await measure([ind[0]!]);
    const r = { dep: await measure(dep), ind: await measure(ind), depPasses: await measure(dep, "passes"), indPasses: await measure(ind, "passes") };
    emit({ kind: "boundary", L, G, D, n, one: one.min, depPer: r.dep.min / n, indPer: r.ind.min / n, depPassesPer: r.depPasses.min / n, indPassesPer: r.indPasses.min / n });
    bufs.forEach(b => b.destroy()); src.destroy();
  }
  console.log("boundary done");
}
emit({ kind: "end", gpuErrors });
console.log(`done, ${gpuErrors} GPU errors, ${((performance.now() - t0) / 1e3).toFixed(1)} s`);
await device.queue.onSubmittedWorkDone();
process.exit(gpuErrors ? 4 : 0);
