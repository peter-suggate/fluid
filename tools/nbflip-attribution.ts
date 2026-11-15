/** NB-FLIP cost attribution at a chosen step of a gate scene (docs/plans/nbflip-port-2026-11-15.md §3, lane A).
 *
 * node --import tsx tools/nbflip-attribution.ts --scene=ID --at=N [--window=10] [--name=NAME]
 *   [--out=artifacts/nbflip-port/attribution] [--set=key:value ...] [--bodies] [--phases=clean,pass,split,pipelined,pipepass,cpuprof]
 *
 * NB-FLIP is built through the method registry on sceneDocument(ID) with the scene's methodProfile over the
 * method's app defaults (as tools/seep-bench.ts), dt = the method's effective step. --bodies drives the scene's
 * bodies as tools/moored-motor-bodies.ts does (paddleboats: hulls held, wheels on their drive). Steps 1..N-1 run
 * fenced and plain (their pressure cycle receipts are kept); then each phase runs --window fenced steps in turn:
 *  - clean:     no instrumentation. Wall per step (advanceTo + awaitFrameCompletion), CPU time inside advanceTo
 *               (the host's encode and submit), CPU time inside queue.submit (Dawn's Metal translation),
 *               submits, command encoders and dispatches a step.
 *  - pass:      a timestamp pair on every compute pass (the passes as the solver encodes them, unsplit): GPU ms
 *               by pass label, the GPU span first-begin to last-end and the idle inside it. Each timestamped pass
 *               adds Metal idle (tools/gpu-pass-profile.ts), so the span runs above the clean step.
 *  - split:     every dispatch in its own timestamped pass (serialises the step; read shares, not the sum), with
 *               each indirect dispatch's arguments copied out just before it runs: per-kernel GPU ms grouped by
 *               pass, direct and indirect dispatches, direct workgroups, and indirect dispatches with zero
 *               workgroups. The last step's ordered dispatch list is saved whole (cycle attribution).
 *  - pipelined: --window steps submitted as the app does (receipt ring), one fence: throughput a step.
 *  - pipepass: pipelined with a timestamp pair on every pass: GPU busy, span and idle gap before each step.
 *  - cpuprof:   --window clean fenced steps under the V8 sampling profiler (100 µs): writes OUT/NAME.cpuprofile and
 *               the self and total CPU ms a step of the hottest functions (where advanceTo's encode time goes).
 * Every buffer created with INDIRECT usage also gets COPY_SRC (for the capture); nothing else differs.
 * Writes OUT/NAME.json. Run under tools/gpu/fluid-gpu-run.sh. */
import assert from "node:assert/strict";
import { execSync } from "node:child_process";
import { mkdirSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { Session } from "node:inspector/promises";
import "../lib/methods";
import { getMethod } from "../lib/core/method-registry";
import { resolveMethodValues, type GPUSolverInstance } from "../lib/core/method-contract";
import { sceneDocument } from "../lib/core/scene-definition";
import { findSceneDefinition } from "../lib/core/scenes";
import { managedGPUDevice, whenGPUCompilationIdle } from "../lib/core/gpu-compilation-manager";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { createProcessRetainedDawnGPU, paceDawnEventPolling } from "../lib/harness/node-dawn-provider";
import { mooredMotorBodies } from "./moored-motor-bodies";

const METHOD = "uniform-narrow-band-flip";
const args = process.argv.slice(2);
for (const a of args) assert.ok(/^(--(?:scene|at|window|name|out|set|phases)=.+|--bodies)$/.test(a), `Unknown option: ${a}`);
const option = (key: string) => args.find(a => a.startsWith(`--${key}=`))?.slice(key.length + 3);
const sceneId = option("scene"); assert.ok(sceneId, "--scene=ID");
const at = Number(option("at")), window = Number(option("window") ?? 10);
assert.ok(Number.isInteger(at) && at >= 1 && Number.isInteger(window) && window >= 1, "--at=N --window=K");
const phases = (option("phases") ?? "clean,pass,split,pipelined").split(",");
for (const p of phases) assert.ok(["clean", "pass", "split", "pipelined", "pipepass", "cpuprof"].includes(p), `phase ${p}`);
const out = resolve(option("out") ?? "artifacts/nbflip-port/attribution");
const name = option("name") ?? `${sceneId}-${at}`;
assert.match(name, /^[A-Za-z0-9._-]+$/);
const sets = Object.fromEntries(args.filter(a => a.startsWith("--set=")).map(a => { const [k, ...v] = a.slice(6).split(":"); const t = v.join(":"); return [k!, t !== "" && Number.isFinite(Number(t)) ? Number(t) : t === "true" ? true : t === "false" ? false : t]; }));

const method = getMethod(METHOD);
// Exact lookup: getSceneDefinition falls back to the catalogue's first scene on an unknown id, which once ran
// lane P2's p1b on water-box-dam-break as "letters" (docs/plans/nbflip-port/progress-lane-f.md, M0(c)).
const definition = findSceneDefinition(sceneId);
assert.ok(definition, `Unknown scene id: ${sceneId}`);
const scene = sceneDocument(definition);
const profile = definition.methodProfile?.methodId === METHOD ? definition.methodProfile : undefined;
const quality = profile?.quality ?? "balanced";
const values = resolveMethodValues(method, quality, { ...method.appDefaults, ...profile?.overrides, ...sets });
const dt = method.effectiveStep_s?.(scene, values) ?? scene.numerics.fixedDt_s;
const poseBodies = args.includes("--bodies") ? mooredMotorBodies(scene, dt) : () => [];
let commit = "unknown"; try { commit = execSync("git rev-parse --short HEAD", { encoding: "utf8" }).trim(); } catch { /* not a checkout */ }

paceDawnEventPolling();
const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE ?? resolve("node_modules/webgpu/index.js")).href);
Object.assign(globalThis, dawn.globals);
const adapter = await createProcessRetainedDawnGPU(dawn, ["backend=metal", "disable-dawn-features=timestamp_quantization"]).requestAdapter();
assert.ok(adapter, "No WebGPU adapter");
const raw: GPUDevice = await adapter.requestDevice({ requiredFeatures: (["subgroups", "float32-filterable", "timestamp-query"] as const).filter(f => adapter.features.has(f)), requiredLimits: requiredFluidDeviceLimits(adapter.limits) });

// ---- Instrumentation ------------------------------------------------------------------------------------------
type Mode = "off" | "count" | "pass" | "split";
let mode: Mode = "off";
const host = { submits: 0, submit_ms: 0, encoders: 0, commandBuffers: 0 };
/** performance.now() at each submit's return (the clean rows' submit offsets from advanceTo's call). */
let submitMarks: number[] = [];
const queue = raw.queue, submit = queue.submit.bind(queue);
Object.defineProperty(queue, "submit", { configurable: true, value: (buffers: Iterable<GPUCommandBuffer>) => {
  const list = [...buffers]; host.submits++; host.commandBuffers += list.length;
  const t = performance.now(); submit(list); const e = performance.now(); host.submit_ms += e - t; submitMarks.push(e);
} });
const names = new WeakMap<GPUComputePipeline, string>();
const named = (p: GPUComputePipeline, d: GPUComputePipelineDescriptor) => { names.set(p, `${d.label ?? d.compute.module.label ?? "?"}:${d.compute.entryPoint ?? "main"}`); return p; };
interface DispatchRow { pass: string; kernel: string; indirect: boolean; groups: number; query?: number; capture?: number }
interface PassRow { pass: string; dispatches: number; query: number }
interface EncoderRecord { step: number; base: number; dispatches: DispatchRow[]; passes: PassRow[]; querySet?: GPUQuerySet; queries: number; capture?: GPUBuffer; captures: number; read?: GPUBuffer; captureRead?: GPUBuffer; resolve?: GPUBuffer }
let records: EncoderRecord[] = [];
/** The step an encoder belongs to (pipepass groups its passes by step). */
let stepTag = 0;
/** pipepass keeps ten steps in flight, and Metal runs out of counter sample buffers at one query set an encoder,
 * so it takes every encoder's queries from one shared set (encoders record one at a time, so each one's range is
 * contiguous from its base). */
let pooled: GPUQuerySet | undefined, pooledNext = 0;
const QUERIES = 4096, CAPTURES = 4096;
const breakSource = raw.createBuffer({ label: "attribution break source", size: 4, usage: GPUBufferUsage.COPY_SRC });
const breakTarget = raw.createBuffer({ label: "attribution break target", size: 4, usage: GPUBufferUsage.COPY_DST });

function wrapEncoder(encoder: GPUCommandEncoder): GPUCommandEncoder {
  const rec: EncoderRecord = { step: stepTag, base: 0, dispatches: [], passes: [], queries: 0, captures: 0 }; records.push(rec);
  const query = () => {
    if (pooled) { if (!rec.querySet) { rec.querySet = pooled; rec.base = pooledNext; } assert.ok(pooledNext + 2 <= QUERIES, "pooled query capacity"); pooledNext += 2; }
    else { rec.querySet ??= raw.createQuerySet({ type: "timestamp", count: QUERIES }); assert.ok(rec.queries + 2 <= QUERIES, "query capacity"); }
    const q = rec.base + rec.queries; rec.queries += 2; return q; };
  const beginComputePass = (descriptor?: GPUComputePassDescriptor): GPUComputePassEncoder => {
    const label = descriptor?.label ?? "(unlabelled pass)";
    assert.ok(!descriptor?.timestampWrites, "a pass with its own timestamps");
    if (mode === "count" || mode === "pass") {
      const row: PassRow | undefined = mode === "pass" ? { pass: label, dispatches: 0, query: query() } : undefined;
      if (row) rec.passes.push(row);
      const pass = encoder.beginComputePass(row ? { ...descriptor, timestampWrites: { querySet: rec.querySet!, beginningOfPassWriteIndex: row.query, endOfPassWriteIndex: row.query + 1 } } : descriptor);
      let kernel = "(no pipeline)";
      return new Proxy(pass, { get(target, key) {
        if (key === "setPipeline") return (p: GPUComputePipeline) => { kernel = names.get(p) ?? "(unnamed)"; target.setPipeline(p); };
        if (key === "dispatchWorkgroups") return (x: number, y = 1, z = 1) => { rec.dispatches.push({ pass: label, kernel, indirect: false, groups: x * y * z }); if (row) row.dispatches++; target.dispatchWorkgroups(x, y, z); };
        if (key === "dispatchWorkgroupsIndirect") return (b: GPUBuffer, o: number) => { rec.dispatches.push({ pass: label, kernel, indirect: true, groups: -1 }); if (row) row.dispatches++; target.dispatchWorkgroupsIndirect(b, o); };
        const v = Reflect.get(target, key); return typeof v === "function" ? v.bind(target) : v;
      } });
    }
    // split: a pass a dispatch, opened at the dispatch with the state set so far.
    let pipeline: GPUComputePipeline | undefined; const groups = new Map<number, unknown[]>();
    const run = (indirect: boolean, go: (p: GPUComputePassEncoder) => void, args3?: [GPUBuffer, number]) => {
      assert.ok(pipeline, "dispatch without a pipeline");
      const row: DispatchRow = { pass: label, kernel: names.get(pipeline) ?? "(unnamed)", indirect, groups: -1 };
      if (indirect) { rec.capture ??= raw.createBuffer({ label: "attribution capture", size: 12 * CAPTURES, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC }); assert.ok(rec.captures < CAPTURES); row.capture = rec.captures++; encoder.copyBufferToBuffer(args3![0], args3![1], rec.capture, 12 * row.capture, 12); }
      else encoder.copyBufferToBuffer(breakSource, 0, breakTarget, 0, 4);
      row.query = query(); rec.dispatches.push(row);
      const p = encoder.beginComputePass({ ...descriptor, timestampWrites: { querySet: rec.querySet!, beginningOfPassWriteIndex: row.query, endOfPassWriteIndex: row.query + 1 } });
      p.setPipeline(pipeline); for (const [, a] of groups) (p.setBindGroup as (...a: unknown[]) => void)(...a);
      go(p); p.end();
      return row;
    };
    const fake = {
      label,
      setPipeline: (p: GPUComputePipeline) => { pipeline = p; },
      setBindGroup: (index: number, group: GPUBindGroup | null, offsets?: Iterable<number> | Uint32Array, start?: number, length?: number) => {
        const values = offsets === undefined ? undefined : Array.from(offsets as Iterable<number>);
        groups.set(index, [index, group, start === undefined ? values : values?.slice(start, start + length!)]);
      },
      dispatchWorkgroups: (x: number, y = 1, z = 1) => { run(false, p => p.dispatchWorkgroups(x, y, z)).groups = x * y * z; },
      dispatchWorkgroupsIndirect: (b: GPUBuffer, o: number) => { run(true, p => p.dispatchWorkgroupsIndirect(b, o), [b, o]); },
      insertDebugMarker() { /* dropped */ }, pushDebugGroup() { /* dropped */ }, popDebugGroup() { /* dropped */ }, end() { /* each dispatch's pass ended */ },
    };
    return new Proxy(fake, { get(target, key) { if (key in target) return Reflect.get(target, key); throw new Error(`split pass: unsupported ${String(key)}`); } }) as unknown as GPUComputePassEncoder;
  };
  return new Proxy(encoder, { get(target, key) {
    if (key === "beginComputePass") return beginComputePass;
    if (key === "finish") return (d?: GPUCommandBufferDescriptor) => {
      if (rec.queries && rec.querySet) {
        rec.resolve = raw.createBuffer({ label: "attribution resolve", size: 8 * rec.queries, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
        rec.read = raw.createBuffer({ label: "attribution read", size: 8 * rec.queries, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
        target.resolveQuerySet(rec.querySet, rec.base, rec.queries, rec.resolve, 0); target.copyBufferToBuffer(rec.resolve, 0, rec.read, 0, 8 * rec.queries);
      }
      if (rec.captures && rec.capture) {
        rec.captureRead = raw.createBuffer({ label: "attribution capture read", size: 12 * rec.captures, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
        target.copyBufferToBuffer(rec.capture, 0, rec.captureRead, 0, 12 * rec.captures);
      }
      return target.finish(d);
    };
    const v = Reflect.get(target, key); return typeof v === "function" ? v.bind(target) : v;
  } });
}
const instrumented = new Proxy(raw, { get(target, key) {
  if (key === "createComputePipeline") return (d: GPUComputePipelineDescriptor) => named(target.createComputePipeline(d), d);
  if (key === "createComputePipelineAsync") return async (d: GPUComputePipelineDescriptor) => named(await target.createComputePipelineAsync(d), d);
  if (key === "createBuffer") return (d: GPUBufferDescriptor) => target.createBuffer(d.usage & GPUBufferUsage.INDIRECT ? { ...d, usage: d.usage | GPUBufferUsage.COPY_SRC } : d);
  if (key === "createCommandEncoder") return (d?: GPUCommandEncoderDescriptor) => { host.encoders++; const e = target.createCommandEncoder(d); return mode === "off" ? e : wrapEncoder(e); };
  const v = Reflect.get(target, key); return typeof v === "function" ? v.bind(target) : v;
} });
/** The records since the last take, their timestamps and captures read back (after the queue drains). */
async function take() {
  const batch = records; records = [];
  await raw.queue.onSubmittedWorkDone();
  const dispatches: (DispatchRow & { ms?: number; begin?: number; end?: number })[] = [], passes: (PassRow & { step: number; ms: number; begin: number; end: number })[] = [];
  for (const r of batch) {
    let t: BigUint64Array | undefined, c: Uint32Array | undefined;
    if (r.read) { await r.read.mapAsync(GPUMapMode.READ); t = new BigUint64Array(r.read.getMappedRange().slice(0)); r.read.unmap(); }
    if (r.captureRead) { await r.captureRead.mapAsync(GPUMapMode.READ); c = new Uint32Array(r.captureRead.getMappedRange().slice(0)); r.captureRead.unmap(); }
    const time = (at: number | undefined) => { if (at === undefined || !t) return undefined; const q = at - r.base; return t[q] === 0n || t[q + 1]! < t[q]! ? undefined : { ms: Number(t[q + 1]! - t[q]!) / 1e6, begin: Number(t[q]!) / 1e6, end: Number(t[q + 1]!) / 1e6 }; };
    for (const d of r.dispatches) dispatches.push({ ...d, ...(d.capture !== undefined && c ? { groups: c[3 * d.capture]! * c[3 * d.capture + 1]! * c[3 * d.capture + 2]! } : {}), ...time(d.query) });
    for (const p of r.passes) { const tm = time(p.query); if (tm) passes.push({ ...p, step: r.step, ...tm }); }
    for (const b of [r.read, r.captureRead, r.resolve, r.capture]) b?.destroy(); if (r.querySet !== pooled) r.querySet?.destroy();
  }
  return { dispatches, passes };
}

const device = managedGPUDevice(instrumented, { requireWorkerRealm: false });
const errors: string[] = [];
device.addEventListener("uncapturederror", e => { e.preventDefault(); errors.push(e.error.message); console.error(e.error.message); });

const median = (v: number[]) => { const s = [...v].sort((a, b) => a - b); return s.length ? s.length % 2 ? s[s.length >> 1]! : (s[s.length / 2 - 1]! + s[s.length / 2]!) / 2 : NaN; };
const mean = (v: number[]) => v.length ? v.reduce((a, b) => a + b, 0) / v.length : NaN;
const r3 = (x: number) => Math.round(x * 1000) / 1000;

const solver: GPUSolverInstance = await method.createSolverAsync!(device, scene, quality, values, undefined, () => {});
await solver.waitForSimulationReady?.();
await whenGPUCompilationIdle();
const info = () => solver.info as unknown as Record<string, number | undefined>;
const particles = () => info().narrowBandFlipParticles;
const band = (solver as unknown as { mixedFrame?: { band?: { schedule: { cycles: number; tailCycles: number; sweeps: number; tail: string }; wideCycles: number; superTail: boolean } } }).mixedFrame?.band;
const fence = () => solver.awaitFrameCompletion ? solver.awaitFrameCompletion() : device.queue.onSubmittedWorkDone();
let step = 0;
const cycles: { step: number; band: number; encoded: number; residual: number; root?: number; rootEncoded?: number }[] = [];
async function fenced() {
  step++; stepTag = step;
  const encoded = band ? Math.min(band.schedule.cycles, band.wideCycles + 1) : NaN;
  const s0 = { ...host }; submitMarks = []; const t0 = performance.now();
  assert.ok(solver.advanceTo(step * dt, poseBodies(step * dt)), `step ${step} refused`);
  const advance_ms = performance.now() - t0;
  await fence();
  const wall_ms = performance.now() - t0;
  if (solver.info.simulationPipelineError) throw new Error(String(solver.info.simulationPipelineError));
  if (errors.length) throw new Error(`GPU error at step ${step}: ${errors[0]}`);
  const i = info();
  cycles.push({ step, band: i.uniformPressureBandCycles ?? -1, encoded, residual: i.uniformPressureBandResidual ?? -1, root: i.uniformPressureCyclesExecuted, rootEncoded: i.uniformPressureCyclesEncoded });
  return { step, wall_ms, advance_ms, submit_ms: host.submit_ms - s0.submit_ms, submits: host.submits - s0.submits, encoders: host.encoders - s0.encoders, commandBuffers: host.commandBuffers - s0.commandBuffers, encoded, submitAt_ms: submitMarks.map(m => r3(m - t0)) };
}

const t0 = performance.now();
const warm: number[] = [];
while (step < at - 1) warm.push((await fenced()).wall_ms);
console.log(JSON.stringify({ kind: "warm", scene: sceneId, step, s: r3((performance.now() - t0) / 1000), medianLast50_ms: r3(median(warm.slice(-50))), particles: particles(), fine: info().uniformMixedFineTiles, bandTiles: info().uniformPressureBandTiles }));
const result: Record<string, unknown> = { kind: "attribution", name, scene: sceneId, at, window, dt, quality, values, sets, commit, date: new Date().toISOString(),
  schedule: band ? { ...band.schedule, superTail: band.superTail } : undefined };
const state = () => ({ step, particles: particles(), fineTiles: info().uniformMixedFineTiles, bandTiles: info().uniformPressureBandTiles });

/** Other GPU jobs on the machine (node harness processes other than this one) and the device utilisation. Before
 * each phase the tool waits (at most 300 s) until the device has idled under 15% for 3 s, since concurrent runs on
 * the other check slots inflate every timing here; each phase records the other jobs alive before and after it. */
const others = () => execSync("ps -axo pid=,command=", { encoding: "utf8" }).split("\n")
  .filter(l => /^\s*\d+ node --import tsx /.test(l) && !l.trim().startsWith(`${process.pid} `)).length;
const utilisation = () => Number(/"Device Utilization %"=(\d+)/.exec(execSync("ioreg -r -c IOAccelerator -d 1", { encoding: "utf8" }))?.[1] ?? 100);
async function quietGate() {
  const t = performance.now(); let quiet = 0;
  while (quiet < 3 && performance.now() - t < 300_000) { quiet = utilisation() < 15 ? quiet + 1 : 0; await new Promise(r => setTimeout(r, 1000)); }
  return { waited_s: r3((performance.now() - t) / 1000), quiet: quiet >= 3, others: others() };
}
const contention: Record<string, unknown> = {};
for (const phase of phases) {
  const gate = await quietGate();
  const begin = state();
  if (phase === "clean") {
    mode = "off";
    const rows: Awaited<ReturnType<typeof fenced>>[] = []; for (let k = 0; k < window; k++) rows.push(await fenced());
    const pick = (f: (r: Awaited<ReturnType<typeof fenced>>) => number) => ({ median: r3(median(rows.map(f))), mean: r3(mean(rows.map(f))) });
    result.clean = { begin, end: state(), wall_ms: pick(r => r.wall_ms), advance_ms: pick(r => r.advance_ms), submit_ms: pick(r => r.submit_ms), submits: pick(r => r.submits), encoders: pick(r => r.encoders), commandBuffers: pick(r => r.commandBuffers), rows };
    // One counted step (dispatch totals the clean steps share; counting adds no GPU work).
    mode = "count"; const counted = await fenced(); mode = "off"; const c = await take();
    result.counted = { ...counted, dispatches: c.dispatches.length, direct: c.dispatches.filter(d => !d.indirect).length, indirect: c.dispatches.filter(d => d.indirect).length,
      kernels: new Set(c.dispatches.map(d => d.kernel)).size, passes: new Set(c.dispatches.map(d => d.pass)).size };
  } else if (phase === "pass") {
    const per: { span: number; busy: number; passes: number }[] = []; const byPass = new Map<string, { ms: number; n: number; dispatches: number }>();
    for (let k = 0; k < window; k++) {
      mode = "pass"; await fenced(); mode = "off"; const { passes } = await take();
      if (!passes.length) continue;
      const first = Math.min(...passes.map(p => p.begin)), last = Math.max(...passes.map(p => p.end));
      per.push({ span: last - first, busy: passes.reduce((a, p) => a + p.ms, 0), passes: passes.length });
      for (const p of passes) { const e = byPass.get(p.pass) ?? { ms: 0, n: 0, dispatches: 0 }; e.ms += p.ms; e.n++; e.dispatches += p.dispatches; byPass.set(p.pass, e); }
    }
    const n = per.length;
    result.pass = { begin, end: state(), steps: n, span_ms: r3(median(per.map(p => p.span))), busy_ms: r3(median(per.map(p => p.busy))), passes: median(per.map(p => p.passes)),
      byPass: [...byPass].map(([pass, e]) => ({ pass, ms: r3(e.ms / n), passes: r3(e.n / n), dispatches: r3(e.dispatches / n) })).sort((a, b) => b.ms - a.ms) };
  } else if (phase === "split") {
    const byKernel = new Map<string, { ms: number; n: number; indirect: number; empty: number; groups: number }>();
    let steps = 0, last: Awaited<ReturnType<typeof take>>["dispatches"] = []; const totals: { direct: number; indirect: number; empty: number; sum: number; groups: number }[] = [];
    for (let k = 0; k < window; k++) {
      mode = "split"; await fenced(); mode = "off"; const { dispatches } = await take(); steps++; last = dispatches;
      const tot = { direct: 0, indirect: 0, empty: 0, sum: 0, groups: 0 };
      for (const d of dispatches) {
        const key = `${d.pass} | ${d.kernel}`, e = byKernel.get(key) ?? { ms: 0, n: 0, indirect: 0, empty: 0, groups: 0 };
        e.ms += d.ms ?? 0; e.n++; if (d.indirect) { e.indirect++; tot.indirect++; } else tot.direct++;
        if (d.groups === 0) { e.empty++; if (d.indirect) tot.empty++; }
        e.groups += Math.max(0, d.groups); tot.groups += Math.max(0, d.groups); tot.sum += d.ms ?? 0; byKernel.set(key, e);
      }
      totals.push(tot);
    }
    result.split = { begin, end: state(), steps, direct: median(totals.map(t => t.direct)), indirect: median(totals.map(t => t.indirect)), emptyIndirect: median(totals.map(t => t.empty)),
      emptyDirectLast: last.filter(d => !d.indirect && d.groups === 0).length,
      workgroups: median(totals.map(t => t.groups)), sum_ms: r3(median(totals.map(t => t.sum))),
      byKernel: [...byKernel].map(([key, e]) => ({ pass: key.split(" | ")[0], kernel: key.split(" | ")[1], ms: r3(e.ms / steps), dispatches: r3(e.n / steps), indirect: r3(e.indirect / steps), empty: r3(e.empty / steps), groups: r3(e.groups / steps) })).sort((a, b) => b.ms - a.ms),
      lastStep: last.map(d => ({ p: d.pass, k: d.kernel, i: d.indirect ? 1 : 0, g: d.groups, ms: d.ms === undefined ? null : r3(d.ms) })) };
  } else if (phase === "cpuprof") {
    mode = "off";
    const session = new Session(); session.connect();
    await session.post("Profiler.enable"); await session.post("Profiler.setSamplingInterval", { interval: 100 }); await session.post("Profiler.start");
    const rows: Awaited<ReturnType<typeof fenced>>[] = []; for (let k = 0; k < window; k++) rows.push(await fenced());
    const { profile: prof } = await session.post("Profiler.stop") as { profile: { nodes: { id: number; callFrame: { functionName: string; url: string; lineNumber: number }; children?: number[] }[]; samples: number[]; timeDeltas: number[] } };
    session.disconnect();
    writeFileSync(`${out}/${name}.cpuprofile`, JSON.stringify(prof));
    // Self and total (inclusive, once per sample) ms a step by function.
    const byId = new Map(prof.nodes.map(n => [n.id, n])), parent = new Map<number, number>();
    for (const n of prof.nodes) for (const c of n.children ?? []) parent.set(c, n.id);
    const key = (id: number) => { const f = byId.get(id)!.callFrame; return `${f.functionName || "(anon)"} ${f.url.replace(/^.*\/fluid\//, "")}:${f.lineNumber + 1}`; };
    const self = new Map<string, number>(), total = new Map<string, number>();
    prof.samples.forEach((id, i) => {
      const dt_ms = (prof.timeDeltas[i + 1] ?? 0) / 1000; self.set(key(id), (self.get(key(id)) ?? 0) + dt_ms);
      const seen = new Set<string>(); for (let n: number | undefined = id; n !== undefined; n = parent.get(n)) { const k = key(n); if (!seen.has(k)) { seen.add(k); total.set(k, (total.get(k) ?? 0) + dt_ms); } }
    });
    const top = (m: Map<string, number>, n: number) => [...m].sort((a, b) => b[1] - a[1]).slice(0, n).map(([fn, ms]) => ({ fn, ms: r3(ms / window) }));
    result.cpuprof = { begin, end: state(), steps: window, advance_ms: r3(median(rows.map(r => r.advance_ms))), wall_ms: r3(median(rows.map(r => r.wall_ms))), self: top(self, 40), total: top(total, 60) };
  } else {
    // pipelined: as the app submits; pipepass: the same with a timestamp pair on every pass, so the GPU's idle
    // time between and inside steps shows directly (busy, span, and the gap before each step's first pass).
    mode = phase === "pipepass" ? "pass" : "off";
    if (phase === "pipepass") { pooled = raw.createQuerySet({ type: "timestamp", count: QUERIES }); pooledNext = 0; }
    const s0 = { ...host }, first = step + 1, start = performance.now();
    for (let k = 0; k < window; k++) {
      step++; stepTag = step;
      while (!solver.advanceTo(step * dt, poseBodies(step * dt))) { if (!solver.framePending) throw new Error(`Step ${step} refused with no frame pending`); await new Promise<void>(r => setImmediate(r)); }
    }
    const submitted_ms = performance.now() - start; await fence(); const total_ms = performance.now() - start; mode = "off";
    if (solver.info.simulationPipelineError) throw new Error(String(solver.info.simulationPipelineError));
    if (phase === "pipepass") {
      const { passes } = await take(); pooled?.destroy(); pooled = undefined; const steps = [...new Set(passes.map(p => p.step))].sort((a, b) => a - b);
      const per = steps.map(n => { const ps = passes.filter(p => p.step === n); return { step: n, begin: Math.min(...ps.map(p => p.begin)), end: Math.max(...ps.map(p => p.end)), busy: ps.reduce((a, p) => a + p.ms, 0) }; });
      const rows = per.map((r, k) => ({ step: r.step, busy_ms: r3(r.busy), span_ms: r3(r.end - r.begin), gapBefore_ms: k ? r3(r.begin - per[k - 1]!.end) : null, period_ms: k ? r3(r.end - per[k - 1]!.end) : null }));
      const later = rows.slice(1);
      result.pipepass = { begin, end: state(), first, steps: window, perStep_ms: r3(total_ms / window), busy_ms: r3(median(later.map(r => r.busy_ms))), span_ms: r3(median(later.map(r => r.span_ms))),
        gapBefore_ms: r3(median(later.map(r => r.gapBefore_ms!))), period_ms: r3(median(later.map(r => r.period_ms!))), rows };
    } else result.pipelined = { begin, end: state(), first, steps: window, perStep_ms: r3(total_ms / window), submitted_ms: r3(submitted_ms), submit_ms: r3((host.submit_ms - s0.submit_ms) / window), submits: (host.submits - s0.submits) / window };
  }
  contention[phase] = { ...gate, othersAfter: others() };
  console.log(JSON.stringify({ kind: "phase", phase, step, contention: contention[phase], ...(phase === "clean" ? { wall: (result.clean as { wall_ms: unknown }).wall_ms, advance: (result.clean as { advance_ms: unknown }).advance_ms, submit: (result.clean as { submit_ms: unknown }).submit_ms, counted: result.counted } : phase === "pass" ? { span: (result.pass as { span_ms: number }).span_ms, busy: (result.pass as { busy_ms: number }).busy_ms } : phase === "split" ? { sum: (result.split as { sum_ms: number }).sum_ms, direct: (result.split as { direct: number }).direct, indirect: (result.split as { indirect: number }).indirect, empty: (result.split as { emptyIndirect: number }).emptyIndirect } : phase === "cpuprof" ? { advance: (result.cpuprof as { advance_ms: number }).advance_ms } : phase === "pipepass" ? { ...(result.pipepass as object), rows: undefined } : result.pipelined as object) }));
}
const hist: Record<string, number> = {}, encHist: Record<string, number> = {};
for (const c of cycles) { hist[c.band] = (hist[c.band] ?? 0) + 1; encHist[`${c.encoded}`] = (encHist[`${c.encoded}`] ?? 0) + 1; }
result.cycles = { histogram: hist, encodedHistogram: encHist, meanBand: r3(mean(cycles.map(c => c.band))), meanEncoded: r3(mean(cycles.filter(c => Number.isFinite(c.encoded)).map(c => c.encoded))), rows: cycles };
result.warmWall_ms = warm;
result.contention = contention;
mkdirSync(out, { recursive: true });
writeFileSync(`${out}/${name}.json`, JSON.stringify(result) + "\n");
console.log(JSON.stringify({ kind: "done", out: `${out}/${name}.json`, cycles: { hist, encHist } }));
solver.destroy();
await raw.queue.onSubmittedWorkDone().catch(() => {});
raw.destroy();
process.exit(0);
