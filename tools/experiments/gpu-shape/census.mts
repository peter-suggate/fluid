// GPU-shape M0(b): the frame fill census (docs/plans/gpu-shape-2026-11-23.md, docs/plans/gpu-shape/m0b-census.md).
//
// Measure (a GPU run, exclusive):
//   tools/gpu/fluid-gpu-run.sh timing node --import tsx tools/experiments/gpu-shape/census.mts
//   env SCENE=seep|dam|letters|paddleboats|teaser|<scene id>  STEPS=100 (warm-up)  R=3 (replays)
//       DETAIL=2[:budget] (automatic detail; teaser defaults to 2)  OUT=<dir> (default ~/code/me/fluid-scratch/gpu-shape/census)
// Caps (compile only, no dispatch; fills the Metal thread cap and threadgroup memory of every kernel, then summarises):
//   tools/gpu/fluid-gpu-run.sh check node --import tsx tools/experiments/gpu-shape/census.mts --caps <OUT>/<scene>
//
// Method. Warm up STEPS steps of the brick engine at a fixed dt of 1/60, fence, and copy every engine buffer (and the
// engine's external body buffers) into GPU shadow buffers. Then, R times: restore the shadows, write the ring words of
// the next step exactly as submitStep does, and replay that one step's recorded placements (the engine's own pipelines,
// bind groups and dispatches, ring slot step % 4) in schedule order, each placement alone in a compute pass with
// timestamps at the pass's start and end, and the indirect args words copied out before each indirect pass. So every
// placement runs once, on its real state, with its real indirect args. A placement's time is the min over the R
// replays. The step's wall is the same restore then the engine's own encoding (one compute pass a part), timestamped
// per part, min over R. The fine (detail) engine: restore the parent, run the parent's step and the allocator (each
// allocator dispatch timed in its own pass), shadow the fine engine, then the same replay on it.
// Caps: each kernel's WGSL (keyed by a hash of its text) through Dawn's dump_shaders to MSL, the
// max_total_threads_per_threadgroup attribute read (the workgroup's threads) and stripped, and psoinfo's
// maxTotalThreadsPerThreadgroup (the register-bound cap, <= 1024) and staticThreadgroupMemoryLength.
import { resolve, dirname } from "node:path";
import { pathToFileURL, fileURLToPath } from "node:url";
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync, openSync, closeSync, unlinkSync } from "node:fs";
import { spawn, execFileSync } from "node:child_process";
import { homedir } from "node:os";

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, "../../..");
const scratch = resolve(homedir(), "code/me/fluid-scratch/gpu-shape");
const kcache = resolve(scratch, "kcache");
const SCENES: Record<string, string> = { seep: "reservoir-seep", dam: "high-resolution-dam-break", letters: "nbflip-figure-8-letters", paddleboats: "paddleboats", teaser: "nbflip-figure-9-teaser" };
const BUCKETS = [1024, 4096, 16384, 65536, 262144, Infinity];
const BUCKET_NAMES = ["<=1k", "1-4k", "4-16k", "16-64k", "64-256k", ">256k"];

interface Row {
  engine: string; part: string; index: number; label: string; entry: string; z: number; w: number; occurrence: number;
  dispatch: string; gx: number; gy: number; gz: number; argsStable: boolean; times_us: number[]; hash: string;
  reads: string[]; writes: string[]; rwUnwritten: string[];
}

// ---------------------------------------------------------------------------------------------------------------
// Caps mode: compile-only, then summary.
if (process.argv[2] === "--caps") { await capsAndSummary(resolve(process.argv[3]!)); process.exit(0); }

// ---------------------------------------------------------------------------------------------------------------
// Measure mode.
await import(`${root}/lib/methods/index.ts`);
const { getMethod } = await import(`${root}/lib/core/method-registry.ts`);
const { resolveMethodValues } = await import(`${root}/lib/core/method-contract.ts`);
const { sceneDocument } = await import(`${root}/lib/core/scene-definition.ts`);
const { getSceneDefinition } = await import(`${root}/lib/core/scenes.ts`);
const { managedGPUDevice } = await import(`${root}/lib/core/gpu-compilation-manager.ts`);
const { requiredFluidDeviceLimits } = await import(`${root}/lib/core/webgpu-device-limits.ts`);
const { createProcessRetainedDawnGPU, paceDawnEventPolling } = await import(`${root}/lib/harness/node-dawn-provider.ts`);
const { RING_SLOTS, RING_WORDS } = await import(`${root}/lib/methods/narrow-band/brick-abi.ts`);
const { mooredMotorBodies } = await import(`${root}/tools/moored-motor-bodies.ts`);

const sceneKey = process.env.SCENE ?? "dam";
const sceneId = SCENES[sceneKey] ?? sceneKey;
const N = Number(process.env.STEPS ?? 100), R = Number(process.env.R ?? 3);
const detailEnv = process.env.DETAIL ?? (sceneKey === "teaser" ? "2" : "");
const outDir = resolve(process.env.OUT ?? resolve(scratch, "census"), sceneKey);
mkdirSync(outDir, { recursive: true }); mkdirSync(kcache, { recursive: true });
const methodId = "uniform-narrow-band-flip", method = getMethod(methodId);
const definition = getSceneDefinition(sceneId), profile = definition.methodProfile, base = sceneDocument(definition);
const [df, db] = detailEnv.split(":").map(Number);
const scene0 = df ? { ...base, fluid: { ...base.fluid, automaticDetail: { factor: df as 2 | 4, ...(db ? { budget: db } : {}) } } } : base;
const scene = { ...scene0, numerics: { ...scene0.numerics, fixedDt_s: 1 / 60, maxDt_s: 1 / 60 } };
const quality = profile?.methodId === methodId ? profile.quality : "balanced";
const values = resolveMethodValues(method, quality, { ...method.appDefaults, ...(profile?.methodId === methodId ? profile.overrides : {}), engine: "brick" });
const log = (...a: unknown[]) => console.log(...a);
log(`census scene ${sceneId} (${sceneKey}) warm-up ${N} steps, R ${R}, detail ${detailEnv || "off"}, out ${outDir}`);

paceDawnEventPolling();
const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE ?? resolve(root, "node_modules/webgpu/index.js")).href);
Object.assign(globalThis, dawn.globals);
const adapter = await createProcessRetainedDawnGPU(dawn, ["backend=metal", "disable-dawn-features=timestamp_quantization"]).requestAdapter();
const raw: GPUDevice = await adapter.requestDevice({ requiredFeatures: ["subgroups", "float32-filterable", "timestamp-query"].filter((f: string) => adapter.features.has(f)), requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
const gpuErrors: string[] = [];
raw.addEventListener?.("uncapturederror", (ev: any) => { gpuErrors.push(String(ev.error?.message)); console.log("GPU ERROR", ev.error?.message); });
const device = managedGPUDevice(raw, { requireWorkerRealm: false });
const s: any = await method.createSolverAsync!(device, scene, quality, values, undefined, () => {});
await s.waitForSimulationReady?.();
s.setPresentationEnabled(false);
const dt = 1 / 60;
const bodies: (t: number) => unknown[] = sceneKey === "paddleboats" || (scene.rigidBodies?.length ?? 0) > 0 ? mooredMotorBodies(scene, dt) : () => [];
const t0 = performance.now();
for (let k = 1; k <= N; k++) {
  while (!s.advanceTo(k * dt + 1e-12, bodies(k * dt))) {
    if (s.info.simulationFailure) throw new Error(`step ${k}: ${s.info.simulationFailure.message}`);
    await new Promise<void>(r => setImmediate(r));
  }
  if (k % 20 === 0) { await s.brickEngine.fence(); log(`warm-up step ${k} ${((performance.now() - t0) / k).toFixed(1)} ms/step avg`); }
}
const P = s.brickEngine;
await P.fence();
const engines: { name: string; e: any; detail?: any }[] = [{ name: "parent", e: P }, ...(s.details ?? []).map((d: any, i: number) => ({ name: i ? `fine${i}` : "fine", e: d.engine, detail: d }))];
if (s.info.simulationFailure) throw new Error(s.info.simulationFailure.message);
if (gpuErrors.length) throw new Error(`GPU errors during warm-up: ${gpuErrors[0]}`);

// ---- shadows ----
const COPY = GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
function shadow(e: any, extra: GPUBuffer[] = []) {
  const list: GPUBuffer[] = [...e.buffers.values(), ...e.external.values(), ...extra].filter((b: GPUBuffer) => (b.usage & COPY) === COPY);
  const uniq = [...new Set(list)];
  const copies = uniq.map(b => raw.createBuffer({ label: `census.shadow.${b.label}`, size: b.size, usage: COPY }));
  const go = (from: GPUBuffer[], to: GPUBuffer[]) => { const enc = raw.createCommandEncoder(); from.forEach((b, i) => enc.copyBufferToBuffer(b, 0, to[i]!, 0, b.size)); raw.queue.submit([enc.finish()]); };
  go(uniq, copies);
  const bytes = uniq.reduce((a, b) => a + b.size, 0);
  return { bytes, restore: () => go(copies, uniq), destroy: () => copies.forEach(c => c.destroy()) };
}
const writeRing = (e: any) => {
  const slot = e.steps % RING_SLOTS;
  raw.queue.writeBuffer(e.buffers.get("set"), 4 * (e.layout.set.ring + RING_WORDS * slot), e.ringWords(e.steps));
  return slot;
};

// ---- timestamps ----
const QMAX = 4096;
const qs = raw.createQuerySet({ type: "timestamp", count: QMAX });
const qResolve = raw.createBuffer({ size: 8 * QMAX, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
const qRead = raw.createBuffer({ size: 8 * QMAX, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
const argStage = raw.createBuffer({ size: 16 * QMAX, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
const argRead = raw.createBuffer({ size: 16 * QMAX, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
async function readTimes(enc: GPUCommandEncoder, n: number, nArgs = 0): Promise<{ t: BigUint64Array; a?: Uint32Array }> {
  enc.resolveQuerySet(qs, 0, n, qResolve, 0); enc.copyBufferToBuffer(qResolve, 0, qRead, 0, 8 * n);
  if (nArgs) enc.copyBufferToBuffer(argStage, 0, argRead, 0, 16 * nArgs);
  raw.queue.submit([enc.finish()]);
  await qRead.mapAsync(GPUMapMode.READ); const t = new BigUint64Array(qRead.getMappedRange(0, 8 * n).slice(0)); qRead.unmap();
  let a: Uint32Array | undefined;
  if (nArgs) { await argRead.mapAsync(GPUMapMode.READ); a = new Uint32Array(argRead.getMappedRange(0, 16 * nArgs).slice(0)); argRead.unmap(); }
  return { t, a };
}
const us = (t: BigUint64Array, i: number) => Number(t[i + 1]! - t[i]!) / 1e3;
const dispatchInto = (pass: GPUComputePassEncoder, r: any, args: GPUBuffer) => {
  pass.setPipeline(r.pipeline); pass.setBindGroup(0, r.group);
  if ("direct" in r.dispatch) pass.dispatchWorkgroups(...(r.dispatch.direct as [number, number, number])); else pass.dispatchWorkgroupsIndirect(args, 4 * r.dispatch.indirect);
};

// The pass floor: a 1-workgroup empty kernel alone in a timestamped pass (what a placement's time cannot go below).
const floorPipe = raw.createComputePipeline({ layout: "auto", compute: { module: raw.createShaderModule({ code: "@compute @workgroup_size(64) fn main(){}" }), entryPoint: "main" } });
async function passFloor() {
  const enc = raw.createCommandEncoder(); const n = 32;
  for (let i = 0; i < n; i++) { const p = enc.beginComputePass({ timestampWrites: { querySet: qs, beginningOfPassWriteIndex: 2 * i, endOfPassWriteIndex: 2 * i + 1 } }); p.setPipeline(floorPipe); p.dispatchWorkgroups(1); p.end(); }
  const { t } = await readTimes(enc, 2 * n); const v = Array.from({ length: n }, (_, i) => us(t, 2 * i)).sort((a, b) => a - b);
  return { min: v[0]!, median: v[n >> 1]! };
}

// ---- text facts (bindings written) ----
function writtenNames(code: string, names: string[]): Set<string> {
  const c = code.replace(/\/\*[\s\S]*?\*\//g, "").replace(/\/\/[^\n]*/g, "");
  const out = new Set<string>();
  for (const n of names) {
    const id = n === "set" ? "nbSet" : n;
    const re = new RegExp(`(&\\s*)?(?<![\\w.])${id}\\s*\\[`, "g"); let m: RegExpExecArray | null;
    while ((m = re.exec(c))) {
      if (m[1]) { out.add(n); break; }
      let j = m.index + m[0].length, d = 1; while (j < c.length && d) { if (c[j] === "[") d++; else if (c[j] === "]") d--; j++; }
      // a member or swizzle, further indices, then an assignment operator
      const rest = /^\s*((\.\w+)|(\[[^\]]*\]))*\s*(=(?!=)|\+=|-=|\*=|\/=|\|=|&=|\^=|<<=|>>=|\+\+|--)/.exec(c.slice(j, j + 200));
      if (rest) { out.add(n); break; }
    }
  }
  return out;
}

const hashOf = (code: string, entry: string) => createHash("sha1").update(entry + "\n" + code).digest("hex").slice(0, 16);
function kernelFacts(k: any) {
  const hash = hashOf(k.code, k.entry);
  const f = resolve(kcache, `${hash}.wgsl`);
  if (!existsSync(f)) { writeFileSync(f, k.code); writeFileSync(resolve(kcache, `${hash}.entry`), k.entry); }
  const storage = k.bindings.filter((b: any) => b.kind !== "uniform");
  const rw = storage.filter((b: any) => b.kind === "rw").map((b: any) => b.name);
  const w = writtenNames(k.code, rw);
  return { hash, reads: storage.map((b: any) => b.name), writes: rw.filter((n: string) => w.has(n)), rwUnwritten: rw.filter((n: string) => !w.has(n)) };
}

// ---- one engine's census ----
async function wall(e: any, restore: () => void) {
  const best = [Infinity, Infinity, Infinity];
  for (let r = 0; r < R; r++) {
    restore(); const slot = writeRing(e); const rec = e.recorded[slot]; const args = e.buffers.get("args");
    const enc = raw.createCommandEncoder();
    const parts = [rec.part1, rec.part2];
    parts.forEach((list: any[], pi: number) => {
      const pass = enc.beginComputePass({ timestampWrites: { querySet: qs, beginningOfPassWriteIndex: 2 * pi, endOfPassWriteIndex: 2 * pi + 1 } });
      for (const x of list) dispatchInto(pass, x, args); pass.end();
    });
    const { t } = await readTimes(enc, 4);
    const p1 = us(t, 0), p2 = us(t, 2);
    best[0] = Math.min(best[0]!, p1); best[1] = Math.min(best[1]!, p2); best[2] = Math.min(best[2]!, p1 + p2);
  }
  return { part1: best[0]!, part2: best[1]!, step: best[2]! };
}

async function replay(name: string, e: any, restore: () => void): Promise<{ rows: Row[]; span: number[] }> {
  const sch = e.schedule;
  const items = [...sch.part1.map((p: any, i: number) => ({ part: "part1", i, p })), ...sch.part2.map((p: any, i: number) => ({ part: "part2", i, p }))];
  if (2 * items.length > QMAX) throw new Error(`${name}: ${items.length} placements over the query set`);
  const occ = new Map<string, number>();
  const rows: Row[] = items.map(({ part, i, p }: any) => {
    const key = `${p.kernel.label}|${p.z ?? 0}|${p.w ?? 0}`; const o = occ.get(key) ?? 0; occ.set(key, o + 1);
    const f = kernelFacts(p.kernel);
    const d = "direct" in p.dispatch ? p.dispatch.direct : undefined;
    return { engine: name, part, index: i, label: p.kernel.label, entry: p.kernel.entry, z: p.z ?? 0, w: p.w ?? 0, occurrence: o,
      dispatch: d ? "direct" : `indirect@${p.dispatch.indirect}`, gx: d ? d[0] : -1, gy: d ? d[1] : -1, gz: d ? d[2] : -1, argsStable: true, times_us: [], ...f };
  });
  const spans: number[] = [];
  for (let r = 0; r < R; r++) {
    restore(); const slot = writeRing(e); const rec = e.recorded[slot]; const args = e.buffers.get("args");
    const all = [...rec.part1, ...rec.part2];
    if (all.length !== items.length) throw new Error(`${name}: recorded ${all.length} vs schedule ${items.length}`);
    // Two command buffers as the engine submits them (part 1, part 2).
    const enc1 = raw.createCommandEncoder(), enc2 = raw.createCommandEncoder(); let nArgs = 0; const argAt: number[] = [];
    all.forEach((x: any, j: number) => {
      const enc = j < rec.part1.length ? enc1 : enc2;
      if (!("direct" in x.dispatch)) { enc.copyBufferToBuffer(args, 4 * x.dispatch.indirect, argStage, 16 * nArgs, 12); argAt[j] = nArgs++; }
      const pass = enc.beginComputePass({ timestampWrites: { querySet: qs, beginningOfPassWriteIndex: 2 * j, endOfPassWriteIndex: 2 * j + 1 } });
      dispatchInto(pass, x, args); pass.end();
    });
    raw.queue.submit([enc1.finish()]);
    const { t, a } = await readTimes(enc2, 2 * all.length, nArgs);
    for (let j = 0; j < all.length; j++) {
      const row = rows[j]!; row.times_us.push(us(t, 2 * j));
      if (argAt[j] !== undefined) {
        const k = argAt[j]!, g = [a![4 * k]!, a![4 * k + 1]!, a![4 * k + 2]!];
        if (r === 0) { row.gx = g[0]!; row.gy = g[1]!; row.gz = g[2]!; }
        else if (row.gx !== g[0] || row.gy !== g[1] || row.gz !== g[2]) row.argsStable = false;
      }
    }
    spans.push(Number(t[2 * all.length - 1]! - t[0]!) / 1e3);
    if (gpuErrors.length) throw new Error(`GPU error during replay: ${gpuErrors[0]}`);
  }
  return { rows, span: spans };
}

const TSV_HEAD = ["engine", "part", "index", "label", "entry", "z", "w", "occurrence", "dispatch", "gx", "gy", "gz", "groups", "argsStable", "t_min_us", "t_med_us", "hash", "reads", "writes", "rwUnwritten"];
const median = (v: number[]) => { const s = [...v].sort((a, b) => a - b); return s.length % 2 ? s[s.length >> 1]! : (s[s.length / 2 - 1]! + s[s.length / 2]!) / 2; };
const results: any = { scene: sceneId, key: sceneKey, warmup: N, R, detail: detailEnv || null, date: new Date().toISOString(),
  git: (() => { try { return { head: execFileSync("git", ["rev-parse", "--short", "HEAD"], { cwd: root, encoding: "utf8" }).trim(), status: execFileSync("git", ["status", "--short"], { cwd: root, encoding: "utf8" }) }; } catch { return null; } })(),
  floor: await passFloor(), engines: [] as any[] };
log(`pass floor (empty 1-workgroup kernel alone in a timestamped pass): min ${results.floor.min.toFixed(2)} us, median ${results.floor.median.toFixed(2)} us`);
const tsv: string[] = [TSV_HEAD.join("\t")];
const pShadow = shadow(P);
let pLive = true; const pDestroy = () => { if (pLive) pShadow.destroy(); pLive = false; };
log(`parent shadow ${(pShadow.bytes / 2 ** 30).toFixed(2)} GiB, steps ${P.steps}, lattice ${JSON.stringify(P.options.lattice)} cap ${JSON.stringify(P.layout.cap)}`);
for (const { name, e, detail } of engines) {
  let restore: () => void, sh: ReturnType<typeof shadow> | undefined, allocRows: Row[] = [];
  if (!detail) restore = pShadow.restore;
  else {
    // The parent's step N+1, then the allocator (timed), then the fine engine's frozen state.
    const al = detail.alloc; const times: number[][] = al.pipelines.map(() => []);
    let aShadow: ReturnType<typeof shadow> | undefined;
    for (let r = 0; r < R; r++) {
      pShadow.restore(); if (r > 0) aShadow!.restore();
      const slot = writeRing(P); const rec = P.recorded[slot]; const args = P.buffers.get("args");
      const enc = raw.createCommandEncoder(); const pass = enc.beginComputePass(); for (const x of [...rec.part1, ...rec.part2]) dispatchInto(pass, x, args); pass.end();
      raw.queue.submit([enc.finish()]);
      if (r === 0) aShadow = shadow({ buffers: new Map([["src", e.buffers.get("src")], ["dir", e.buffers.get("dir")]]), external: new Map() }, [al.state]);
      const enc2 = raw.createCommandEncoder();
      al.pipelines.forEach(([pipe, groups]: [GPUComputePipeline, [number, number, number]], j: number) => {
        const p = enc2.beginComputePass({ timestampWrites: { querySet: qs, beginningOfPassWriteIndex: 2 * j, endOfPassWriteIndex: 2 * j + 1 } });
        p.setPipeline(pipe); p.setBindGroup(0, al.group); p.dispatchWorkgroups(...groups); p.end();
      });
      const { t } = await readTimes(enc2, 2 * al.pipelines.length);
      al.pipelines.forEach((_: unknown, j: number) => times[j]!.push(us(t, 2 * j)));
    }
    aShadow!.destroy();
    // The parent's shadow is no longer needed once the last fine engine's inputs are in place (GPU memory: the teaser).
    if (engines.at(-1)!.e === e) pDestroy();
    allocRows = al.pipelines.map(([pipe, groups]: [GPUComputePipeline, [number, number, number]], j: number) => ({
      engine: name, part: "alloc", index: j, label: `detailAlloc.${String(pipe.label).split(".").pop()}`, entry: String(pipe.label).split(".").pop()!, z: 0, w: 0, occurrence: 0,
      dispatch: "direct", gx: groups[0], gy: groups[1], gz: groups[2], argsStable: true, times_us: times[j]!, hash: "", reads: [], writes: [], rwUnwritten: [] }));
    sh = shadow(e); restore = sh.restore;
    log(`${name} shadow ${(sh.bytes / 2 ** 30).toFixed(2)} GiB, steps ${e.steps}, lattice ${JSON.stringify(e.options.lattice)} cap ${JSON.stringify(e.layout.cap)}`);
  }
  const w = await wall(e, restore);
  const { rows, span } = await replay(name, e, restore);
  const all = [...allocRows, ...rows];
  for (const r of all) tsv.push([r.engine, r.part, r.index, r.label, r.entry, r.z, r.w, r.occurrence, r.dispatch, r.gx, r.gy, r.gz, r.gx * r.gy * r.gz, r.argsStable ? 1 : 0,
    Math.min(...r.times_us).toFixed(2), median(r.times_us).toFixed(2), r.hash, r.reads.join(","), r.writes.join(","), r.rwUnwritten.join(",")].join("\t"));
  const sum = rows.reduce((a, r) => a + Math.min(...r.times_us), 0), sumA = allocRows.reduce((a, r) => a + Math.min(...r.times_us), 0);
  const unstable = rows.filter(r => !r.argsStable).length;
  results.engines.push({ name, wall_us: w, replaySpan_us: Math.min(...span), sumPlacements_us: sum, alloc_us: sumA, placements: rows.length, unstableArgs: unstable,
    lattice: e.options.lattice, cap: e.layout.cap });
  log(`${name}: wall part1 ${w.part1.toFixed(0)} + part2 ${w.part2.toFixed(0)} = ${(w.step / 1000).toFixed(3)} ms | replay span ${(Math.min(...span) / 1000).toFixed(3)} ms | sum of ${rows.length} placements ${(sum / 1000).toFixed(3)} ms${allocRows.length ? ` | allocator ${(sumA / 1000).toFixed(3)} ms` : ""} | args unstable across replays: ${unstable}`);
  sh?.destroy();
}
pDestroy();
writeFileSync(resolve(outDir, "census.tsv"), tsv.join("\n") + "\n");
writeFileSync(resolve(outDir, "census.json"), JSON.stringify(results, null, 1));
log(`wrote ${outDir}/census.tsv (${tsv.length - 1} rows) and census.json; gpu errors ${gpuErrors.length}`);
if (gpuErrors.length) process.exitCode = 3;
s.destroy?.();
process.exit(process.exitCode ?? 0);

// ---------------------------------------------------------------------------------------------------------------
async function capsAndSummary(dir: string) {
  const psoinfo = resolve(scratch, "psoinfo");
  if (!existsSync(psoinfo)) execFileSync("swiftc", ["-O", "-o", psoinfo, resolve(here, "psoinfo.swift")], { stdio: "inherit" });
  const tsvText = readFileSync(resolve(dir, "census.tsv"), "utf8").trim().split("\n");
  const head = tsvText[0]!.split("\t"); const recs = tsvText.slice(1).map(l => Object.fromEntries(l.split("\t").map((v, i) => [head[i]!, v])) as Record<string, string>);
  const hashes = [...new Set(recs.map(r => r.hash).filter(Boolean))];
  const todo = hashes.filter(h => !existsSync(resolve(kcache, `${h}.cap`)));
  console.log(`caps: ${hashes.length} kernels, ${todo.length} to compile`);
  // The child's output goes to a file (a pipe lost the tail of large dumps); a dump without the entry is retried once.
  const run = (h: string, attempt = 0): Promise<void> => new Promise<void>(res => {
    const entry = readFileSync(resolve(kcache, `${h}.entry`), "utf8").trim();
    const logf = resolve(kcache, `${h}.log`), fd = openSync(logf, "w");
    const child = spawn("node", ["--import", "tsx", resolve(here, "msl.mts"), resolve(kcache, `${h}.wgsl`), entry], { cwd: root, stdio: ["ignore", fd, fd] });
    child.on("close", () => {
      closeSync(fd); const out = readFileSync(logf, "utf8");
      const i = out.lastIndexOf("/* Dumped generated MSL */");
      if ((i < 0 || !out.includes("kernel void dawn_entry_point", i)) && attempt === 0) return res(run(h, 1));
      if (i < 0) { writeFileSync(resolve(kcache, `${h}.cap`), JSON.stringify({ error: out.slice(0, 400) })); return res(); }
      let m = out.slice(i + 26); m = m.slice(0, m.lastIndexOf("}") + 1);
      const wg = /\[\[max_total_threads_per_threadgroup\((\d+)\)\]\]/.exec(m);
      unlinkSync(logf);
      writeFileSync(resolve(kcache, `${h}.metal`), m.replace(/\[\[max_total_threads_per_threadgroup\(\d+\)\]\] */g, ""));
      let cap = -1, tg = -1, simd = -1, err = "";
      try {
        const o = execFileSync(psoinfo, [resolve(kcache, `${h}.metal`)], { encoding: "utf8" }).trim().split(/\s+/);
        if (o[1] === "ERR" || o[1] === "nofn") err = o.slice(1).join(" "); else { cap = Number(o[1]); tg = Number(o[2]); simd = Number(o[3]); }
      } catch (e) { err = String(e).slice(0, 200); }
      writeFileSync(resolve(kcache, `${h}.cap`), JSON.stringify({ wg: wg ? Number(wg[1]) : -1, cap, tgmem: tg, simd, ...(err ? { error: err } : {}) }));
      res();
    });
  });
  const pool = Number(process.env.JOBS ?? 6); let next = 0;
  await Promise.all(Array.from({ length: pool }, async () => { while (next < todo.length) { const h = todo[next++]!; await run(h); if (next % 20 === 0) console.log(`  compiled ${next}/${todo.length}`); } }));
  // Tint passes workgroup memory as a [[threadgroup(n)]] argument (Dawn sets its length), so psoinfo's static length is
  // 0: the bytes are the WGSL var<workgroup> declarations the MSL still uses (its module-vars struct names them).
  const caps = new Map(hashes.map(h => {
    const c = JSON.parse(readFileSync(resolve(kcache, `${h}.cap`), "utf8")), mf = resolve(kcache, `${h}.metal`);
    if (existsSync(mf)) { const msl = readFileSync(mf, "utf8"); c.tgmem = Math.max(c.tgmem, 0) + wgslWorkgroupBytes(readFileSync(resolve(kcache, `${h}.wgsl`), "utf8"), n => msl.includes(`.${n}=`)); }
    return [h, c];
  }));
  const json = JSON.parse(readFileSync(resolve(dir, "census.json"), "utf8"));
  // Workgroup threads for kernels whose MSL failed: the WGSL attribute, consts resolved.
  const wgslWG = (h: string) => {
    const code = readFileSync(resolve(kcache, `${h}.wgsl`), "utf8"), entry = readFileSync(resolve(kcache, `${h}.entry`), "utf8").trim();
    const m = new RegExp(`@workgroup_size\\(([^)]*)\\)\\s*fn\\s+${entry}\\b`).exec(code); if (!m) return -1;
    return m[1]!.split(",").map(t => { t = t.trim().replace(/u$/, ""); if (/^\d+$/.test(t)) return Number(t); const c = new RegExp(`const\\s+${t}\\s*(?::\\s*\\w+)?\\s*=\\s*(\\d+)u?\\s*;`).exec(code); return c ? Number(c[1]) : NaN; }).reduce((a, b) => a * b, 1);
  };
  const out: string[] = []; const md: string[] = [];
  const outHead = [...head, "L", "lanes", "cap", "wgmem", "slab32", "fill32"];
  out.push(outHead.join("\t"));
  for (const eng of json.engines) {
    const rows: any[] = recs.filter(r => r.engine === eng.name).map((r): any => {
      const c = r.hash ? caps.get(r.hash) : undefined;
      const L = c && c.wg > 0 ? c.wg : r.hash ? wgslWG(r.hash) : (r.part === "alloc" ? 64 : -1);
      const G = Number(r.groups), t = Number(r.t_min_us);
      const cap = c?.cap ?? -1, lanes = G * L;
      // A resident slab, crudely: 32 cores x the pipeline's thread cap (lane A's measured slab replaces this).
      const slab = cap > 0 ? 32 * cap : -1;
      return { ...r, L, G, t, cap, wgmem: c?.tgmem ?? -1, lanes, slab, fill: slab > 0 ? lanes / slab : -1 };
    });
    for (const r of rows) out.push([...head.map(k => (r as any)[k]), r.L, r.lanes, r.cap, r.wgmem, r.slab, r.fill.toFixed(3)].join("\t"));
    const sum = rows.reduce((a, r) => a + r.t, 0);
    const steps = rows.filter(r => r.part !== "alloc");
    const sumStep = steps.reduce((a, r) => a + r.t, 0);
    const hist = BUCKETS.map(() => ({ n: 0, t: 0 }));
    for (const r of steps) { const b = BUCKETS.findIndex(x => r.lanes <= x); hist[b]!.n++; hist[b]!.t += r.t; }
    const sub = steps.filter(r => r.G > 0 && r.slab > 0 && r.lanes < r.slab), subT = sub.reduce((a, r) => a + r.t, 0);
    const zero = steps.filter(r => r.G === 0), zeroT = zero.reduce((a, r) => a + r.t, 0);
    const floorT = json.floor.min;
    md.push(`### ${json.key} — ${eng.name}`, "");
    md.push(`wall (engine encoding, a pass a part): **${(eng.wall_us.step / 1000).toFixed(2)} ms** (part 1 ${(eng.wall_us.part1 / 1000).toFixed(2)}, part 2 ${(eng.wall_us.part2 / 1000).toFixed(2)}); `
      + `replay span (a pass a placement) ${(eng.replaySpan_us / 1000).toFixed(2)} ms; Σ placements ${(sumStep / 1000).toFixed(2)} ms over ${steps.length} placements`
      + (eng.alloc_us ? `; allocator ${(eng.alloc_us / 1000).toFixed(2)} ms` : "") + `; args unstable across replays: ${eng.unstableArgs}; pass floor ${floorT.toFixed(1)} µs.`, "");
    md.push(`sub-slab (live, lanes < 32 × cap): **${(subT / 1000).toFixed(2)} ms = ${(100 * subT / sumStep).toFixed(0)}% of Σ** in ${sub.length} placements; gated (0 workgroups) ${zero.length} placements, ${(zeroT / 1000).toFixed(2)} ms.`, "");
    md.push("| total lanes | placements | ms | % of Σ |", "|---|---|---|---|");
    hist.forEach((h, i) => md.push(`| ${BUCKET_NAMES[i]} | ${h.n} | ${(h.t / 1000).toFixed(2)} | ${(100 * h.t / sumStep).toFixed(0)}% |`));
    md.push("");
    const FB = [1e-9, 0.25, 1, 4, Infinity], FBN = ["0 (gated)", "< 0.25", "0.25–1", "1–4", "≥ 4"], fh = FB.map(() => ({ n: 0, t: 0, short: 0 }));
    for (const r of steps) if (r.fill >= 0) { const b = FB.findIndex(x => r.fill < x); fh[b]!.n++; fh[b]!.t += r.t; if (r.t < 25) fh[b]!.short++; }
    md.push("| fill32 = lanes / (32 × cap) | placements | ms | % of Σ | placements under 25 µs |", "|---|---|---|---|---|");
    fh.forEach((h, i) => md.push(`| ${FBN[i]} | ${h.n} | ${(h.t / 1000).toFixed(2)} | ${(100 * h.t / sumStep).toFixed(0)}% | ${h.short} |`));
    // Sensitivity of the sub-slab share to the slab (lane A measures the real one): lanes < m x 32 x cap.
    const sens = [0.5, 1, 2, 4].map(m => { const v = steps.filter(r => r.G > 0 && r.slab > 0 && r.lanes < m * r.slab); return `${m}×: ${(100 * v.reduce((a, r) => a + r.t, 0) / sumStep).toFixed(0)}% (${v.length})`; });
    md.push("", `sub-slab share of Σ if the slab is m × 32 × cap: ${sens.join(", ")}.`);
    const short = steps.filter(r => r.t < 25), shortT = short.reduce((a, r) => a + r.t, 0);
    const mid = steps.filter(r => r.t < 50), midT = mid.reduce((a, r) => a + r.t, 0);
    md.push(`placements under 50 µs (any fill): ${mid.length}, ${(midT / 1000).toFixed(2)} ms = ${(100 * midT / sumStep).toFixed(0)}% of Σ.`);
    md.push(`placements under 25 µs (any fill): ${short.length}, ${(shortT / 1000).toFixed(2)} ms = ${(100 * shortT / sumStep).toFixed(0)}% of Σ.`, "");
    // Families: by label.
    const fam = new Map<string, typeof steps>();
    for (const r of steps) { const k = r.label; if (!fam.has(k)) fam.set(k, []); fam.get(k)!.push(r); }
    const famRows = [...fam.entries()].map(([k, v]) => {
      const live = v.filter(r => r.G > 0), Gs = live.map(r => r.G).sort((a, b) => a - b);
      const t = v.reduce((a, r) => a + r.t, 0), tSub = v.filter(r => r.G > 0 && r.slab > 0 && r.lanes < r.slab).reduce((a, r) => a + r.t, 0);
      return { k, n: v.length, live: live.length, G: Gs.length ? (Gs[0] === Gs.at(-1) ? `${Gs[0]}` : `${Gs[0]}–${Gs.at(-1)}`) : "0", Gmax: Gs.at(-1) ?? 0,
        L: v[0]!.L, cap: v[0]!.cap, wgmem: v[0]!.wgmem, t, tSub, fill: Math.max(...v.map(r => r.fill)), perUs: t / v.length, writes: v[0]!.writes };
    }).sort((a, b) => b.t - a.t);
    md.push("| family | placements (live) | G | L | cap | wgmem B | max fill32 | ms | sub-slab ms | µs each |", "|---|---|---|---|---|---|---|---|---|---|");
    const subFam = famRows.filter(f => f.tSub > 0).sort((a, b) => b.tSub - a.tSub);
    for (const f of famRows.slice(0, Number(process.env.TOP ?? 25))) md.push(`| ${f.k} | ${f.n} (${f.live}) | ${f.G} | ${f.L} | ${f.cap} | ${f.wgmem} | ${f.fill.toFixed(2)} | ${(f.t / 1000).toFixed(3)} | ${(f.tSub / 1000).toFixed(3)} | ${f.perUs.toFixed(1)} |`);
    md.push("", "Top sub-slab families (by sub-slab ms):", "", "| family | placements (live) | G | L | cap | wgmem B | max fill32 | sub-slab ms | µs each |", "|---|---|---|---|---|---|---|---|---|");
    for (const f of subFam.slice(0, 12)) md.push(`| ${f.k} | ${f.n} (${f.live}) | ${f.G} | ${f.L} | ${f.cap} | ${f.wgmem} | ${f.fill.toFixed(2)} | ${(f.tSub / 1000).toFixed(3)} | ${f.perUs.toFixed(1)} |`);
    md.push("");
    writeFileSync(resolve(dir, `families-${eng.name}.tsv`), ["family\tplacements\tlive\tG\tL\tcap\twgmem\tmaxFill32\tms\tsubSlabMs\tusEach\twrites", ...famRows.map(f => [f.k, f.n, f.live, f.G, f.L, f.cap, f.wgmem, f.fill.toFixed(3), (f.t / 1000).toFixed(4), (f.tSub / 1000).toFixed(4), f.perUs.toFixed(2), f.writes].join("\t"))].join("\n") + "\n");
    void sum;
  }
  writeFileSync(resolve(dir, "census-caps.tsv"), out.join("\n") + "\n");
  writeFileSync(resolve(dir, "summary.md"), md.join("\n") + "\n");
  const bad = [...caps.entries()].filter(([, c]) => c.error);
  console.log(md.join("\n"));
  if (bad.length) console.log(`caps errors: ${bad.length}: ${bad.slice(0, 5).map(([h, c]) => `${h} ${String(c.error).slice(0, 120)}`).join(" | ")}`);
}

/** Bytes of a module's var<workgroup> declarations that `used` keeps (WGSL host-shareable layout rules). */
function wgslWorkgroupBytes(code: string, used: (name: string) => boolean): number {
  const c = code.replace(/\/\*[\s\S]*?\*\//g, "").replace(/\/\/[^\n]*/g, "");
  const consts = new Map<string, string>();
  for (const m of c.matchAll(/\bconst\s+(\w+)\s*(?::\s*\w+)?\s*=\s*([^;]+);/g)) consts.set(m[1]!, m[2]!);
  const evalN = (e: string, depth = 0): number => {
    const t = e.replace(/\b([A-Za-z_]\w*)\b/g, id => { const v = consts.get(id); if (v === undefined || depth > 8) throw new Error(`const ${id}`); return `(${evalN(v, depth + 1)})`; })
      .replace(/(\d+)[ui]\b/g, "$1");
    if (!/^[\d\s+\-*/()%<>]+$/.test(t)) throw new Error(`expr ${e}`);
    return Math.trunc(Function(`return (${t});`)() as number);
  };
  const structs = new Map<string, string>();
  for (const m of c.matchAll(/\bstruct\s+(\w+)\s*\{([^}]*)\}/g)) structs.set(m[1]!, m[2]!);
  const splitTop = (t: string) => { const out: string[] = []; let d = 0, cur = ""; for (const ch of t) { if (ch === "<") d++; if (ch === ">") d--; if (ch === "," && d === 0) { out.push(cur); cur = ""; } else cur += ch; } if (cur.trim()) out.push(cur); return out; };
  const layout = (t: string): [number, number] => {
    t = t.trim();
    if (/^(f32|u32|i32|bool)$/.test(t) || /^atomic<.*>$/.test(t)) return [4, 4];
    let m = /^vec([234])(?:<\w+>|[fiu])$/.exec(t); if (m) { const n = Number(m[1]); return n === 2 ? [8, 8] : n === 3 ? [12, 16] : [16, 16]; }
    m = /^array<(.+)>$/.exec(t);
    if (m) { const [el, n] = splitTop(m[1]!); const [sz, al] = layout(el!); const stride = Math.ceil(sz / al) * al; return [stride * evalN(n!), al]; }
    const body = structs.get(t);
    if (body !== undefined) {
      let off = 0, al = 1;
      for (const f of splitTop(body)) { const i = f.indexOf(":"); if (i < 0) continue; const [sz, a] = layout(f.slice(i + 1)); off = Math.ceil(off / a) * a + sz; al = Math.max(al, a); }
      return [Math.ceil(off / al) * al, al];
    }
    throw new Error(`type ${t}`);
  };
  let bytes = 0;
  for (const m of c.matchAll(/var<workgroup>\s*(\w+)\s*:\s*([^;]+);/g)) {
    if (!used(m[1]!)) continue;
    try { bytes += layout(m[2]!)[0]; } catch { bytes += 0; }
  }
  return bytes;
}
