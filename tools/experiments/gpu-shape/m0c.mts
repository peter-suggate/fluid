// GPU-shape M0c (docs/plans/gpu-shape/m0a-shape.md §M0c): where random-gather throughput collapses between 8 MB and
// 1 GB, and whether it follows bytes touched or distinct 16 KB pages touched.
//   tools/gpu/fluid-gpu-run.sh timing node --import tsx tools/experiments/gpu-shape/m0c.mts <out.jsonl> [phase,...]
// Phases: check probe ladder ladder64 geo win clus sparse (default all but ladder64/geo, the geometry controls).
//
// Kernel: a dependent hash chain over a 1 GB pool whose words hold hsh(index). Each link hashes the previously
// loaded value (so every address depends on the previous load) into
//   region r (one of K regions of W bytes, scattered over the pool by an odd-multiplier bijection on W-sized slots)
//   + page   ((h >> 16) % pages-per-region, 16 KB pages)
//   + word   (h & (U/4 - 1): only the first U bytes of each page are ever touched)
// Region choice: mode "clus" picks r from the loaded value every link (gathers spread over all K regions);
// mode "win" gives the whole workgroup one region per link, r = (workgroup + link * G) mod K (a workgroup gathers
// inside one window at a time, like a brick's 27-neighbourhood, and the windows move through all K regions).
// Throughput is the slope between widths G and 2G (loads added / time added), so the dispatch floor drops out.
import { appendFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { createClockProbe, TOP_CLOCK_PROBE_US } from "./clock-probe.js";

const root = resolve(import.meta.dirname, "../../..");
const { createProcessRetainedDawnGPU, paceDawnEventPolling } = await import(`${root}/lib/harness/node-dawn-provider.ts`);
const { requiredFluidDeviceLimits } = await import(`${root}/lib/core/webgpu-device-limits.ts`);
const out = process.argv[2] ?? "m0c.jsonl";
const phases = new Set((process.argv[3] ?? "check,probe,ladder,win,clus,sparse").split(","));
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
emit({ kind: "setup", device: (adapter as any).info?.description, reps: REPS, maxBuffer: device.limits.maxBufferSize, maxBinding: device.limits.maxStorageBufferBindingSize });
const U = GPUBufferUsage;
const t0 = performance.now();

// ---- shaders ----
const HASH = `fn hsh(x0: u32) -> u32 { var x = x0 * 747796405u + 2891336453u; x = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u; return (x >> 22u) ^ x; }`;
const hshJS = (x0: number) => { let x = (Math.imul(x0 >>> 0, 747796405) + 2891336453) >>> 0; x = Math.imul(((x >>> ((x >>> 28) + 4)) ^ x) >>> 0, 277803737) >>> 0; return ((x >>> 22) ^ x) >>> 0; };
const POOL_BYTES = 1 << 30, POOL_WORDS = POOL_BYTES / 4;
const fillWGSL = `@group(0) @binding(0) var<storage, read_write> pool: array<u32>;
${HASH}
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g: vec3u, @builtin(num_workgroups) n: vec3u) {
  let stride = n.x * 256u;
  for (var i = g.x; i < ${POOL_WORDS}u; i += stride) { pool[i] = hsh(i); }
}`;
// P: kmask, slotmask, wsh (log2 words per region slot), pages (16 KB pages per region), umask (words used per page - 1),
// depth, seed, ngroups.
const gathWGSL = (L: number, M: number, win: boolean) => `struct P { kmask: u32, slotmask: u32, wsh: u32, pages: u32, umask: u32, depth: u32, seed: u32, ngroups: u32 }
@group(0) @binding(0) var<storage, read> pool: array<u32>;
@group(0) @binding(1) var<storage, read_write> outb: array<u32>;
@group(0) @binding(2) var<uniform> p: P;
${HASH}
@compute @workgroup_size(${L}) fn main(@builtin(global_invocation_id) g: vec3u, @builtin(workgroup_id) w: vec3u) {
  let lane = g.x;
${[...Array(M).keys()].map(m => `  var v${m} = hsh(lane * ${M}u + ${m}u + p.seed); let s${m} = hsh(v${m} ^ 0x68e31da4u) | 1u;`).join("\n")}
  for (var d = 0u; d < p.depth; d++) {
${win ? `    let r = (w.x + d * p.ngroups) & p.kmask;` : ""}
${[...Array(M).keys()].map(m => `    { let h = hsh(v${m} ^ s${m});
${win ? "" : `      let r = hsh(h ^ 0x9e3779b9u) & p.kmask;`}
      let base = ((r * 0x9e3779b1u) & p.slotmask) << p.wsh;
      v${m} = pool[base + ((((h >> 16u) % p.pages) << 12u) | (h & p.umask))]; }`).join("\n")}
  }
  outb[g.x] = ${[...Array(M).keys()].map(m => `v${m}`).join(" ^ ")};
}`;
// CPU model of one lane (for the check phase): returns the final value and asserts every index is in the pool.
function laneJS(lane: number, M: number, win: boolean, prm: number[], wid: number) {
  const [kmask, slotmask, wsh, pages, umask, depth, seed, ngroups] = prm as [number, number, number, number, number, number, number, number];
  const v: number[] = [], s: number[] = [];
  for (let m = 0; m < M; m++) { v[m] = hshJS(lane * M + m + seed); s[m] = (hshJS((v[m]! ^ 0x68e31da4) >>> 0) | 1) >>> 0; }
  for (let d = 0; d < depth; d++) for (let m = 0; m < M; m++) {
    const h = hshJS((v[m]! ^ s[m]!) >>> 0);
    const r = win ? ((wid + d * ngroups) & kmask) >>> 0 : (hshJS((h ^ 0x9e3779b9) >>> 0) & kmask) >>> 0;
    const base = ((Math.imul(r, 0x9e3779b1) >>> 0) & slotmask) * 2 ** wsh;
    const idx = base + ((((h >>> 16) % pages) * 4096) | (h & umask));
    if (idx >= POOL_WORDS) throw new Error(`index ${idx} out of pool`);
    v[m] = hshJS(idx);
  }
  return v.reduce((a, b) => (a ^ b) >>> 0, 0);
}

const pipes = new Map<string, GPUComputePipeline>();
async function pipeline(key: string, code: string) {
  let pp = pipes.get(key); if (pp) return pp;
  const module = device.createShaderModule({ code });
  for (const m of (await module.getCompilationInfo()).messages) if (m.type === "error") throw new Error(`${key}: ${m.lineNum}:${m.linePos} ${m.message}`);
  pp = await device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "main" } });
  pipes.set(key, pp); return pp;
}

// ---- pool ----
const pool = device.createBuffer({ size: POOL_BYTES, usage: U.STORAGE });
{
  const fp = await pipeline("fill", fillWGSL);
  const enc = device.createCommandEncoder(); const pass = enc.beginComputePass(); pass.setPipeline(fp);
  pass.setBindGroup(0, device.createBindGroup({ layout: fp.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: pool } }] }));
  pass.dispatchWorkgroups(8192); pass.end(); device.queue.submit([enc.finish()]); await device.queue.onSubmittedWorkDone();
}
console.log(`pool filled ${(performance.now() - t0).toFixed(0)} ms`);
const MAXLANES = 1 << 18;
const outBuf = device.createBuffer({ size: 4 * MAXLANES, usage: U.STORAGE | U.COPY_SRC });
const chk = device.createBuffer({ size: 4 * MAXLANES, usage: U.STORAGE | U.COPY_SRC });
const readBuf = device.createBuffer({ size: 4 * 1024, usage: U.MAP_READ | U.COPY_DST });

// ---- clock control: heater (writes outBuf, so it is ordered with the timed passes) + the reusable probe ----
const heatWGSL = `@group(0) @binding(0) var<storage, read_write> o: array<u32>;
@group(0) @binding(1) var<uniform> it: vec4u;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g: vec3u) {
  var x = f32(g.x) * 1e-6; var y = x + 1.0;
  for (var i = 0u; i < it.x; i++) { x = fma(x, 0.999, 0.25); y = fma(y, x, -0.5); }
  o[g.x & ${MAXLANES - 1}u] = bitcast<u32>(x + y);
}`;
const heatPipe = await pipeline("heat", heatWGSL);
const heatBGfor = (n: number) => { const b = device.createBuffer({ size: 16, usage: U.UNIFORM | U.COPY_DST }); device.queue.writeBuffer(b, 0, new Uint32Array([n, 0, 0, 0]));
  return device.createBindGroup({ layout: heatPipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: outBuf } }, { binding: 1, resource: { buffer: b } }] }); };
let heatIters = 7450, heatBG = heatBGfor(heatIters);
const probe = createClockProbe(device, { capacity: 4, after: outBuf });
const qs = device.createQuerySet({ type: "timestamp", count: 64 });
const qb = device.createBuffer({ size: 8 * 64, usage: U.QUERY_RESOLVE | U.COPY_SRC });
const qr = device.createBuffer({ size: 8 * 64, usage: U.MAP_READ | U.COPY_DST });
type Disp = { pipe: GPUComputePipeline; bg: GPUBindGroup; G: number };
let bestProbe = Infinity;

async function timeOne(d: Disp | null, reps = REPS, heat = true) {
  const enc = device.createCommandEncoder();
  let q = 0;
  if (heat) { const pass = enc.beginComputePass({ timestampWrites: { querySet: qs, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } }); pass.setPipeline(heatPipe); pass.setBindGroup(0, heatBG); pass.dispatchWorkgroups(256); pass.end(); }
  q = 2;
  if (d) for (let r = 0; r <= reps; r++) {
    const pass = enc.beginComputePass({ timestampWrites: { querySet: qs, beginningOfPassWriteIndex: q, endOfPassWriteIndex: q + 1 } });
    pass.setPipeline(d.pipe); pass.setBindGroup(0, d.bg); pass.dispatchWorkgroups(d.G); pass.end(); q += 2;
  }
  probe.encode(enc);
  enc.resolveQuerySet(qs, 0, q, qb, 0); enc.copyBufferToBuffer(qb, 0, qr, 0, 8 * q);
  device.queue.submit([enc.finish()]);
  if (heat) { const e2 = device.createCommandEncoder(); const p2 = e2.beginComputePass(); p2.setPipeline(heatPipe); p2.setBindGroup(0, heatBG); p2.dispatchWorkgroups(256); p2.end(); device.queue.submit([e2.finish()]); }
  const [pr] = await probe.read();
  await qr.mapAsync(GPUMapMode.READ); const t = new BigUint64Array(qr.getMappedRange().slice(0, 8 * q)); qr.unmap();
  const span = (i: number) => Number(t[2 * i + 1]! - t[2 * i]!) / 1e3;
  const us: number[] = []; for (let r = 2; r <= reps + 1 && d; r++) us.push(span(r));
  return { min: us.length ? Math.min(...us) : 0, all: us, heat: heat ? span(0) : 0, probe: pr! };
}
async function ramp(why: string) {
  const keep = heatBG; heatBG = heatBGfor(heatIters * 6);
  const t = performance.now(); let last = Infinity, stable = 0;
  while (performance.now() - t < 5000) { const r = await timeOne(null, 0); if (r.probe > 0.97 * last) stable++; else stable = 0; last = Math.min(last, r.probe); if (stable >= 4 && performance.now() - t > 300) break; }
  heatBG = keep; bestProbe = Math.min(bestProbe, last);
  console.log(`ramp (${why}): ${(performance.now() - t).toFixed(0)} ms, probe ${last.toFixed(1)} us`); emit({ kind: "ramp", why, probe: last });
}
async function measure(d: Disp) {
  let r = await timeOne(d);
  if (r.probe > 1.5 * bestProbe) { await ramp(`probe ${r.probe.toFixed(0)}`); r = await timeOne(d); }
  bestProbe = Math.min(bestProbe, r.probe); return r;
}
await ramp("start");
for (let i = 0; i < 2; i++) { const r = await timeOne(null, 0); heatIters = Math.max(64, Math.round(heatIters * 1500 / Math.max(r.heat, 1))); heatBG = heatBGfor(heatIters); console.log(`heater ${r.heat.toFixed(0)} us, probe ${r.probe.toFixed(1)}`); }

// ---- configurations ----
interface Cfg { phase: string; win: boolean; M: number; L: number; G: number; D: number; W: number; K: number; Ub: number }
const log2 = (x: number) => Math.round(Math.log2(x));
function prmOf(c: Cfg, G: number) {
  const slots = Math.max(1, Math.floor(POOL_BYTES / c.W)); // power of two when W is (all but the ladder, where K = 1)
  const pages = Math.max(1, Math.floor(c.W / 16384));
  const umask = Math.min(c.Ub, c.W, 16384) / 4 - 1;
  return [c.K - 1, c.K === 1 ? 0 : slots - 1, c.K === 1 ? 0 : log2(c.W / 4), pages, umask, c.D, 7, G];
}
const prmBuf = (vals: number[]) => { const b = device.createBuffer({ size: 32, usage: U.UNIFORM | U.COPY_DST }); device.queue.writeBuffer(b, 0, new Uint32Array(vals)); return b; };
const bind = (pp: GPUComputePipeline, vals: number[], o = outBuf) => device.createBindGroup({ layout: pp.getBindGroupLayout(0), entries: [
  { binding: 0, resource: { buffer: pool } }, { binding: 1, resource: { buffer: o } }, { binding: 2, resource: { buffer: prmBuf(vals) } }] });
const MB = (b: number) => b / (1 << 20);
async function run(c: Cfg) {
  const pp = await pipeline(`g${c.L}_${c.M}_${c.win}`, gathWGSL(c.L, c.M, c.win));
  const a = await measure({ pipe: pp, bg: bind(pp, prmOf(c, c.G)), G: c.G });
  const b = await measure({ pipe: pp, bg: bind(pp, prmOf(c, 2 * c.G)), G: 2 * c.G });
  const loads = c.L * c.G * c.M * c.D;
  const pages = c.K * Math.max(1, c.W / 16384), used = Math.min(c.Ub, c.W, 16384);
  const touched = c.W >= 16384 ? pages * used : c.K * c.W;
  const rec = { kind: c.phase, win: c.win, M: c.M, L: c.L, G: c.G, D: c.D, W: c.W, K: c.K, U: used, touchedMB: MB(touched), pages16k: c.W >= 16384 ? pages : c.K,
    t1: a.min, t2: b.min, slopeGs: loads / Math.max(1e-3, b.min - a.min) / 1e3, rawGs: 2 * loads / b.min / 1e3, probe1: a.probe, probe2: b.probe };
  emit(rec);
  console.log(`${c.phase} ${c.win ? "win " : "clus"} M${c.M} W ${MB(c.W) >= 1 ? MB(c.W) + "M" : c.W / 1024 + "K"} K ${c.K} U ${used} touched ${rec.touchedMB.toFixed(2)} MB pages ${rec.pages16k}: ${a.min.toFixed(1)} / ${b.min.toFixed(1)} us, slope ${rec.slopeGs.toFixed(2)} G/s (raw ${rec.rawGs.toFixed(2)}), probe ${a.probe.toFixed(1)}/${b.probe.toFixed(1)}`);
}

// ---- check: GPU vs CPU model for a few lanes (also asserts every index lands in the pool) ----
if (phases.has("check")) {
  const cases: Cfg[] = [
    { phase: "check", win: false, M: 1, L: 256, G: 4, D: 16, W: 48 << 20, K: 1, Ub: 16384 },
    { phase: "check", win: false, M: 4, L: 256, G: 4, D: 16, W: 1 << 30, K: 1, Ub: 16384 },
    { phase: "check", win: true, M: 1, L: 64, G: 16, D: 64, W: 16384, K: 65536, Ub: 16384 },
    { phase: "check", win: false, M: 1, L: 64, G: 16, D: 64, W: 16384, K: 4096, Ub: 128 },
    { phase: "check", win: true, M: 1, L: 64, G: 16, D: 64, W: 1 << 20, K: 256, Ub: 16384 },
  ];
  for (const c of cases) {
    const pp = await pipeline(`g${c.L}_${c.M}_${c.win}`, gathWGSL(c.L, c.M, c.win));
    const vals = prmOf(c, c.G);
    const enc = device.createCommandEncoder(); const pass = enc.beginComputePass(); pass.setPipeline(pp); pass.setBindGroup(0, bind(pp, vals, chk)); pass.dispatchWorkgroups(c.G); pass.end();
    enc.copyBufferToBuffer(chk, 0, readBuf, 0, 4 * 256); device.queue.submit([enc.finish()]);
    await readBuf.mapAsync(GPUMapMode.READ); const gpu = new Uint32Array(readBuf.getMappedRange().slice(0)); readBuf.unmap();
    let bad = 0; for (let lane = 0; lane < 256; lane++) if (gpu[lane] !== laneJS(lane, c.M, c.win, vals, Math.floor(lane / c.L))) bad++;
    console.log(`check ${c.win ? "win" : "clus"} M${c.M} W ${c.W} K ${c.K} U ${c.Ub}: ${bad} of 256 lanes wrong`); emit({ kind: "check", ...c, bad });
  }
}

// ---- probe validation: the helper probe ordered after a long memory-bound pass (after: outBuf) vs unordered ----
if (phases.has("probe")) {
  const pp = await pipeline("g256_1_false", gathWGSL(256, 1, false));
  const heavy: Disp = { pipe: pp, bg: bind(pp, prmOf({ phase: "", win: false, M: 1, L: 256, G: 512, D: 16, W: 1 << 30, K: 1, Ub: 16384 }, 512)), G: 512 };
  const loose = createClockProbe(device, { capacity: 4 });
  for (let i = 0; i < 3; i++) {
    const enc = device.createCommandEncoder();
    { const pass = enc.beginComputePass(); pass.setPipeline(heatPipe); pass.setBindGroup(0, heatBG); pass.dispatchWorkgroups(256); pass.end(); }
    probe.encode(enc); // after an ALU pass: clean reading
    { const pass = enc.beginComputePass({ timestampWrites: { querySet: qs, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } }); pass.setPipeline(heavy.pipe); pass.setBindGroup(0, heavy.bg); pass.dispatchWorkgroups(heavy.G); pass.end(); }
    probe.encode(enc); // ordered after the heavy gather (reads outBuf)
    { const pass = enc.beginComputePass(); pass.setPipeline(heavy.pipe); pass.setBindGroup(0, heavy.bg); pass.dispatchWorkgroups(heavy.G); pass.end(); }
    loose.encode(enc); // own buffers only: free to overlap the heavy gather
    { const pass = enc.beginComputePass(); pass.setPipeline(heatPipe); pass.setBindGroup(0, heatBG); pass.dispatchWorkgroups(256); pass.end(); }
    enc.resolveQuerySet(qs, 0, 2, qb, 0); enc.copyBufferToBuffer(qb, 0, qr, 0, 16);
    device.queue.submit([enc.finish()]);
    const [afterAlu, afterGather] = await probe.read(); const [unordered] = await loose.read();
    await qr.mapAsync(GPUMapMode.READ); const t = new BigUint64Array(qr.getMappedRange().slice(0, 16)); qr.unmap();
    const heavyUs = Number(t[1]! - t[0]!) / 1e3;
    console.log(`probe: after ALU ${afterAlu!.toFixed(1)} us, after 1 GB gather (${heavyUs.toFixed(0)} us, ordered) ${afterGather!.toFixed(1)}, unordered ${unordered!.toFixed(1)}`);
    emit({ kind: "probeCheck", afterAlu, afterGather, unordered, heavyUs, top: TOP_CLOCK_PROBE_US });
  }
  loose.destroy();
}

// ---- 1. footprint ladder: contiguous random over F bytes (K = 1), L256 x G256 vs 2G, D16, M1 and M4 ----
if (phases.has("ladder")) {
  const Fs = [1, 4, 8, 16, 24, 32, 40, 48, 56, 64, 96, 128, 192, 256, 384, 512, 768, 1024];
  for (const M of [1, 4]) for (const F of Fs) await run({ phase: "ladder", win: false, M, L: 256, G: 256, D: 16, W: F << 20, K: 1, Ub: 16384 });
}
// Controls: the contiguous ladder at the win/clus geometry (L64 x G1024, D64), and the geometry itself at 8/256 MB.
if (phases.has("ladder64")) for (const F of [1, 4, 8, 16, 24, 32, 40, 48, 64, 128, 256, 1024]) await run({ phase: "ladder64", win: false, M: 1, L: 64, G: 1024, D: 64, W: F << 20, K: 1, Ub: 16384 });
if (phases.has("geo")) for (const F of [8, 256]) for (const [L, D] of [[256, 16], [256, 64], [64, 16], [64, 64], [1024, 16]] as const)
  await run({ phase: "geo", win: false, M: 1, L, G: (64 << 10) / L, D, W: F << 20, K: 1, Ub: 16384 });
// ---- 2. windowed (win) and page/region-clustered (clus): K regions of W bytes scattered over the 1 GB pool ----
const Ws = [16 << 10, 64 << 10, 256 << 10, 1 << 20, 4 << 20, 16 << 20];
const Ts = [8, 16, 32, 64, 128, 256, 512, 1024].map(t => t << 20);
for (const [ph, win] of [["win", true], ["clus", false]] as const) if (phases.has(ph))
  for (const W of Ws) for (const T of Ts) if (T >= W) await run({ phase: ph, win, M: 1, L: 64, G: 1024, D: 64, W, K: T / W, Ub: 16384 });
// ---- 3. sparse pages: K random 16 KB pages, only the first U bytes of each used (bytes = K U, pages = K) ----
if (phases.has("sparse"))
  for (const Ub of [128, 1024, 4096]) for (const T of Ts) await run({ phase: "sparse", win: false, M: 1, L: 64, G: 1024, D: 64, W: 16384, K: T / 16384, Ub });

console.log(`done, ${gpuErrors} GPU errors, ${((performance.now() - t0) / 1e3).toFixed(1)} s`);
emit({ kind: "done", gpuErrors });
process.exit(0);
