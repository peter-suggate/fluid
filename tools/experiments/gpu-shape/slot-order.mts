// GPU-shape M0d (lane B): the slot-order go/no-go (docs/plans/gpu-shape/m0d-slot-order.md).
//
//   tools/gpu/fluid-gpu-run.sh timing node --import tsx tools/experiments/gpu-shape/slot-order.mts <scene> [outdir]
//   env STEPS=100 (warm-up, as the census)  R=5 (interleaved replays)  CTR_SECS=4 (0: no counters)  EQ_ONLY=1 (proof only)
//       PROOF_STEPS=1,4 (steps run from each snapshot)  ID_REPS=2 (identity runs: the noise floor)
//
// Warm up the brick engine STEPS steps (fixed dt 1/60, as census.mts), save() it, and build permuted snapshots of
// that frozen state on the CPU: (a) identity; (b) compact, the live F and C slots renumbered 0..live-1 in their
// current order; (c) compact + Morton, the live slots renumbered in the Morton order of their brick or page key; and
// (s) scatter, a random bijection of all slots, the proof's hardest case (never timed). Each remaps every slot-indexed
// structure the code holds (the region table below, every entry read from its WGSL accessor): positions of every
// slot-strided region (pools, records, children, set marks / bounds / bins / tiles / reach, lane F's fcell, fdep and
// fext4, the census, lane P's pmgF / pmgC / pmgTop, staticNeed), and every word whose value is a slot (directory page
// words, child words, record neighbour words and parent, lists, tlist, the free stacks, pmgF's stage and live lists,
// pmgC's seam faces and page stage, pmgTop's SLOT5, RB, SB and ZC). K and B slots are not renumbered. A buffer the
// table does not know fails explicitly; so does any region whose computed size is not the buffer's.
// Proof: from each snapshot, the engine's own recorded step (1 step, then 4) on the GPU; the observables are read by
// brick and page key, never by slot (key sets, φ and velocity sums, liquid vertex counts, particle counts and
// position sums), and compared with identity's; identity run twice gives the noise floor.
// Timing: census.mts's replay (each placement alone in a timestamped pass, the engine's own pipelines and bind
// groups) and wall (one pass a part), R rounds interleaved across the variants, min over the rounds.
// Counters: the child prints READY and cycles the three variants' split-pass replays (passes labelled
// `m0d.<variant>|<entry>`) while this process attaches Instruments' Metal GPU Counters for CTR_SECS.
import { resolve, dirname } from "node:path";
import { pathToFileURL, fileURLToPath } from "node:url";
import { execFileSync, spawn } from "node:child_process";
import { createWriteStream, existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";

const here = dirname(fileURLToPath(import.meta.url)), root = resolve(here, "../../..");
const SCENES: Record<string, string> = { seep: "reservoir-seep", dam: "high-resolution-dam-break", letters: "nbflip-figure-8-letters", paddleboats: "paddleboats" };
const log = (...a: unknown[]) => console.log(...a);

if (process.env.M0D_CHILD !== "1") await parent();
else await child();

// ---------------------------------------------------------------------------------------------------------------
async function parent() {
  const [scene = "dam", outArg] = process.argv.slice(2);
  const out = resolve(outArg ?? resolve(homedir(), "code/me/fluid-scratch/gpu-shape/m0d", scene)); mkdirSync(out, { recursive: true });
  const secs = Number(process.env.CTR_SECS ?? 4), eqOnly = process.env.EQ_ONLY === "1", ctr = secs > 0 && !eqOnly;
  const trace = `${out}/run.trace`; if (existsSync(trace)) rmSync(trace, { recursive: true });
  const env = { ...process.env, M0D_CHILD: "1", SCENE: scene, OUT: out, LOOP_MS: String(ctr ? 1000 * secs + 3500 : 0),
    ...(ctr ? { FLUID_WEBGPU_DAWN_FEATURES: "use_user_defined_labels_in_backend" } : {}) };
  const kid = spawn(process.execPath, ["--import", "tsx", fileURLToPath(import.meta.url)], { cwd: root, env, stdio: ["ignore", "pipe", "inherit"] });
  let attached = false;
  const recorded = new Promise<void>((done, fail) => {
    kid.stdout.setEncoding("utf8");
    kid.stdout.on("data", (c: string) => {
      process.stdout.write(c);
      if (!ctr || attached || !c.includes("READY")) return;
      attached = true;
      setTimeout(() => {
        log(`attaching to ${kid.pid}`);
        const blank = resolve(execFileSync("xcode-select", ["-p"]).toString().trim(), "../Applications/Instruments.app/Contents/Packages/Base.instrdst/Contents/Templates/Blank.tracetemplate");
        const x = spawn("xcrun", ["xctrace", "record", "--template", blank, "--instrument", "Metal Application", "--instrument", "GPU", "--instrument", "Metal GPU Counters",
          "--output", trace, "--no-prompt", "--time-limit", `${secs}s`, "--attach", String(kid.pid)], { stdio: ["ignore", "pipe", "pipe"] });
        x.stdout.on("data", (d: Buffer) => process.stdout.write(`[xctrace] ${d}`));
        x.stderr.on("data", (d: Buffer) => process.stdout.write(`[xctrace] ${d}`));
        x.once("exit", code => code === 0 ? done() : fail(new Error(`xctrace ${code}`)));
      }, 1000);
    });
  });
  const finished = new Promise<number>(r => kid.once("exit", c => r(c ?? 1)));
  if (ctr) await recorded;
  const rc = await finished; log("child rc", rc);
  if (!ctr || rc !== 0) process.exit(rc);
  const { parseTraceTable } = await import(`${root}/tools/xctrace-trace-tables.ts`);
  const tables = [
    ["metal-gpu-intervals", "gpu-intervals", ["start", "duration", "channel-name", "event-label", "process", "encoder-id"]],
    ["metal-application-encoders-list", "encoders", ["encoder-id", "encoder-label", "process"]],
    ["gpu-counter-info", "counter-info", ["counter-id", "name", "type", "sample-interval"]],
    ["gpu-counter-value", "counter-value", ["timestamp", "counter-id", "value", "ring-buffer-index"]],
  ] as const;
  for (const [schema, file, columns] of tables) {
    const p = spawn("xcrun", ["xctrace", "export", "--input", trace, "--xpath", `/trace-toc/run[@number="1"]/data/table[@schema="${schema}"]`], { stdio: ["ignore", "pipe", "inherit"] });
    p.stdout.setEncoding("utf8");
    const sink = createWriteStream(`${out}/${file}.ndjson`); let n = 0;
    for await (const row of parseTraceTable(p.stdout as AsyncIterable<string>, { columns })) { sink.write(JSON.stringify(row) + "\n"); n++; }
    await new Promise(r => sink.end(r)); log(file, n);
  }
  execFileSync(process.execPath, ["--import", "tsx", `${root}/tools/xctrace-hardware-limiters.ts`, `--artifact=${out}`, `--pid=${kid.pid}`], { cwd: root, stdio: "inherit" });
  const C = ["Compute Occupancy", "ALU Limiter", "Buffer Read Limiter", "GPU Last Level Cache Limiter", "MMU Limiter", "MMU TLB Miss Rate", "GPU Read Bandwidth"];
  const short = ["occ", "aluL", "bufRL", "llcL", "mmuL", "tlbMiss", "rdGBs"];
  const d = JSON.parse(readFileSync(`${out}/hardware-limiters.json`, "utf8"));
  const agg = new Map<string, { us: number; c: Map<string, [number, number]> }>();
  for (const st of d.stages) {
    const m = /m0d\.([a-z]+)\|([A-Za-z0-9_]+)/.exec(String(st.label)); if (!m) continue;
    const lab = `${m[1]}\t${m[2]}`;
    const a = agg.get(lab) ?? { us: 0, c: new Map() }; agg.set(lab, a); a.us += st.exclusiveUs;
    for (const k of C) if (st.counters[k]) { const v = a.c.get(k) ?? [0, 0]; v[0] += st.counters[k].meanAcrossRings * st.exclusiveUs; v[1] += st.exclusiveUs; a.c.set(k, v); }
  }
  const lines = [["variant", "entry", "us", ...short].join("\t")];
  for (const [lab, a] of [...agg].sort((x, y) => y[1].us - x[1].us))
    lines.push([lab, a.us.toFixed(0), ...C.map(k => { const v = a.c.get(k); return v && v[1] ? (v[0] / v[1]).toFixed(1) : "nan"; })].join("\t"));
  writeFileSync(`${out}/counters.tsv`, lines.join("\n") + "\n");
  for (const e of ["nbOperator", "lAdvect", "uTransfer", "uExtend", "uAdvect", "pSeed"])
    log(lines.filter(l => l.split("\t")[1] === e).join("\n"));
}

// ---------------------------------------------------------------------------------------------------------------
async function child() {
  await import(`${root}/lib/methods/index.ts`);
  const { getMethod } = await import(`${root}/lib/core/method-registry.ts`);
  const { resolveMethodValues } = await import(`${root}/lib/core/method-contract.ts`);
  const { sceneDocument } = await import(`${root}/lib/core/scene-definition.ts`);
  const { getSceneDefinition } = await import(`${root}/lib/core/scenes.ts`);
  const { managedGPUDevice } = await import(`${root}/lib/core/gpu-compilation-manager.ts`);
  const { requiredFluidDeviceLimits } = await import(`${root}/lib/core/webgpu-device-limits.ts`);
  const { createProcessRetainedDawnGPU, paceDawnEventPolling } = await import(`${root}/lib/harness/node-dawn-provider.ts`);
  const { mooredMotorBodies } = await import(`${root}/tools/moored-motor-bodies.ts`);
  const ABI = await import(`${root}/lib/methods/narrow-band/brick-abi.ts`);
  const { fluidLayout } = await import(`${root}/lib/methods/narrow-band/fluid-grid.wgsl.ts`);
  const { mgLayout } = await import(`${root}/lib/methods/narrow-band/pressure-mgpcg.ts`);
  const { CENSUS } = await import(`${root}/lib/methods/narrow-band/brick-census.wgsl.ts`);
  const { brickStaticNeedWords } = await import(`${root}/lib/methods/narrow-band/brick-static-need.wgsl.ts`);
  const { BRICK_STRIDE: ST, POOLS, H, RING_SLOTS, RING_WORDS, NONE, SEAM, SLOT_MASK, RECORD_WORDS: RW, CHILD_WORDS, CMASK_WORDS, BOUND_WORDS, PSUPPORT_WORDS, CAND_MASK_WORDS, LIST, LIST_SLOTS } = ABI;

  const sceneKey = process.env.SCENE ?? "dam", sceneId = SCENES[sceneKey] ?? sceneKey;
  const N = Number(process.env.STEPS ?? 100), R = Number(process.env.R ?? 5), LOOP_MS = Number(process.env.LOOP_MS ?? 0);
  const eqOnly = process.env.EQ_ONLY === "1", out = resolve(process.env.OUT!);
  const methodId = "uniform-narrow-band-flip", method = getMethod(methodId);
  const definition = getSceneDefinition(sceneId), profile = definition.methodProfile, base = sceneDocument(definition);
  const scene = { ...base, numerics: { ...base.numerics, fixedDt_s: 1 / 60, maxDt_s: 1 / 60 } };
  const quality = profile?.methodId === methodId ? profile.quality : "balanced";
  const values = resolveMethodValues(method, quality, { ...method.appDefaults, ...(profile?.methodId === methodId ? profile.overrides : {}), engine: "brick" });
  log(`slot-order scene ${sceneId} (${sceneKey}) warm-up ${N}, R ${R}, loop ${LOOP_MS} ms${eqOnly ? ", proof only" : ""}, out ${out}`);
  paceDawnEventPolling();
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE ?? resolve(root, "node_modules/webgpu/index.js")).href);
  Object.assign(globalThis, dawn.globals);
  const adapter = await createProcessRetainedDawnGPU(dawn, ["backend=metal", "disable-dawn-features=timestamp_quantization",
    ...(process.env.FLUID_WEBGPU_DAWN_FEATURES ? [`enable-dawn-features=${process.env.FLUID_WEBGPU_DAWN_FEATURES}`] : [])]).requestAdapter();
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
    if (k % 20 === 0) { await s.brickEngine.fence(); log(`warm-up ${k} ${((performance.now() - t0) / k).toFixed(1)} ms/step`); }
  }
  const P = s.brickEngine;
  await P.fence();
  if ((s.details ?? []).length) throw new Error("slot-order: a detail engine is not supported");
  if (gpuErrors.length) throw new Error(`GPU errors during warm-up: ${gpuErrors[0]}`);

  // ---- the layout and the region table ----
  const L = P.layout, F = L.cap.f, Cc = L.cap.c, tiered = P.stages.tiered === true;
  const FL = fluidLayout(L, { tiered }), M = mgLayout(L), T = M.top;
  const orig: Map<string, Uint32Array> = (await P.save()).buffers;
  const words = (n: string) => orig.get(n)!.length;
  type Pm = { f: Int32Array; c: Int32Array };
  type Region = { buf: string; kind: "F" | "C"; base: number; m: number; nch?: number; vals?: (d: Uint32Array, fl: Float32Array, p: Pm, at: number, m: number) => void };
  // Value maps, each counting the words that look like slots but are not.
  const anomalies = new Map<string, number>();
  const bad = (k: string) => anomalies.set(k, (anomalies.get(k) ?? 0) + 1);
  const mapU = (cap: number, which: "f" | "c", tag: string) => (d: Uint32Array, _f: Float32Array, p: Pm, at: number, m: number) => {
    const q = p[which]; for (let i = at; i < at + m; i++) { const v = d[i]!; if (v === NONE) continue; if (v < cap) d[i] = q[v]!; else bad(tag); } };
  const mapF32 = (cap: number, which: "f" | "c", tag: string) => (_d: Uint32Array, fl: Float32Array, p: Pm, at: number, m: number) => {
    const q = p[which]; for (let i = at; i < at + m; i++) { const v = fl[i]!; if (v < 0) continue; if (Number.isInteger(v) && v < cap) fl[i] = q[v]!; else bad(tag); } };
  const nbr = (v: number, p: Pm, tag: string): number => {
    if (v === NONE) return v;
    const k = v >>> 29;
    if (k < 4) { if (v < F) return p.f[v]!; bad(tag); return v; }
    if (k === 4) { const c = v & SLOT_MASK; if (c < Cc) return (SEAM | p.c[c]!) >>> 0; bad(tag); return v; }
    if (k === 5) return v; // BULK | bslot
    bad(`${tag}.kind${k}`); return v;
  };
  const regions: Region[] = [];
  const add = (buf: string, kind: "F" | "C", base: number, m: number, nch = 1, vals?: Region["vals"]) => regions.push({ buf, kind, base, m, nch, ...(vals ? { vals } : {}) });
  // Pools: channel-major, F channels then C channels, BRICK_STRIDE a slot (brick-abi.md §2.2). cut is K-indexed: kept.
  for (const pool of ["vel", "phi", "pres"] as const) { const n = POOLS[pool].length; add(pool, "F", 0, ST, n); add(pool, "C", n * F * ST, ST, n); }
  // set (brick-abi.ts BrickLayout.set; index forms checked in the WGSL: every region slot-major).
  const S = L.set;
  for (const [o, m] of [[S.markF, 1], [S.boundF, BOUND_WORDS], [S.binStart, 1], [S.binCount, 1], [S.binCursor, 1], [S.pLiquid, 16], [S.tileOcc, 1], [S.trig, 1], [S.psupport, PSUPPORT_WORDS], [S.act, 1]] as const) add("set", "F", o, m);
  for (const [o, m] of [[S.markC, 1], [S.boundC, 1], [S.want, CMASK_WORDS], [S.airGate, CMASK_WORDS], [S.reach0, 512], [S.reach1, 512], [S.reachStep, 1], [S.wallTile, 384], [S.candMask, CAND_MASK_WORDS]] as const) add("set", "C", o, m);
  // dir: records, children, the tile masks (the page window's values below).
  const D = L.dir;
  add("dir", "F", D.frec, RW, 1, (d, _f, p, at) => { for (let w = 0; w < 27; w++) d[at + w] = nbr(d[at + w]!, p, "frec.nbr"); mapU(Cc, "c", "frec.parent")(d, _f, p, at + 27, 1); });
  add("dir", "C", D.crec, RW, 1, (d, _f, p, at) => { for (let w = 0; w < 27; w++) if (d[at + w] !== NONE) bad("crec.nbr"); });
  add("dir", "C", D.child, CHILD_WORDS, 1, mapU(F, "f", "child"));
  add("dir", "C", D.cmask, CMASK_WORDS);
  // Lane E's census and the static need (brick-census.wgsl.ts CENSUS.STRIDE a C slot; brick-static-need.wgsl.ts).
  add("census", "C", 0, CENSUS.STRIDE);
  if (orig.has("staticNeed")) { add("staticNeed", "C", 0, 4); add("staticNeed", "C", 4 * Cc + 4, 32); }
  // Lane F (fluid-grid.wgsl.ts fluidLayout): fcell's per-cell regions (512 a slot), the per-slot words, the C words.
  const R_ = FL.fcell;
  for (const o of [R_.CNT, R_.RUN, R_.NEAR, R_.NEAR2, R_.KEY, R_.HOT, R_.DEPTH]) add("fcell", "F", o, 512);
  add("fcell", "F", R_.MASK, 16); add("fcell", "F", R_.TILE, 96); add("fcell", "F", R_.SLOT, 1); add("fcell", "F", R_.BOUND, 1); add("fcell", "F", R_.LIQ, 16);
  if (tiered) {
    add("fcell", "F", R_.UNIT, 3 * 729); add("fcell", "F", R_.PHIU, 512); add("fcell", "F", R_.HMASK, 8);
    add("fcell", "C", R_.CACHE4, 3 * 512); add("fcell", "C", R_.DIL4, 512); add("fcell", "C", R_.RES, 1); add("fcell", "F", R_.RES + Cc, 1);
    add("fext4", "C", 0, 2 * 512 * 3 * 3);
  }
  add("fdep", "F", 0, 6 * 512);
  // Lane P (pressure-mgpcg.ts mgLayout): pmgF, pmgC, pmgTop.
  add("pmgF", "F", M.W0, ST); add("pmgF", "F", M.W1, ST); add("pmgF", "F", M.L1, 64, 10);
  add("pmgF", "F", M.STG, 81, 1, (d, fl, p, at) => { mapF32(F, "f", "STG.f")(d, fl, p, at, 27); mapF32(Cc, "c", "STG.c")(d, fl, p, at + 27, 27); });
  add("pmgF", "F", M.RKU, 3 * ST); add("pmgF", "F", M.FC, 7 * ST); add("pmgF", "F", M.DRY, 1);
  add("pmgC", "C", M.L2, 512, 15); add("pmgC", "C", M.L3, 64, 10); add("pmgC", "C", M.L4, 8, 10); add("pmgC", "C", M.L5, 1, 10);
  add("pmgC", "C", M.PDELTA, 8); add("pmgC", "C", M.PH, 512, 2); add("pmgC", "C", M.SEAMT, 512); add("pmgC", "C", M.SEAML, 3072);
  add("pmgC", "C", M.SEAMF, 3072, 1, mapF32(F, "f", "SEAMF")); add("pmgC", "C", M.SEAMN, 1); add("pmgC", "C", M.QSTG, 27, 1, mapF32(Cc, "c", "QSTG"));
  add("pmgTop", "C", T.IDX5, 1); add("pmgTop", "F", T.RBF, 1);
  for (const o of [T.ZK, T.ZS, T.FS]) add("pmgTop", "F", o, 1);
  add("pmgTop", "F", T.ZC, 1, 1, mapU(Cc, "c", "ZC"));
  add("pmgTop", "C", T.PHW, 2, 2);
  // Whole-buffer value lists (no slot position): applied after the positions.
  const lists = (d: Map<string, Uint32Array>, p: Pm) => {
    const set = d.get("set")!, dir = d.get("dir")!, f = d.get("pmgF")!, ff = new Float32Array(f.buffer, f.byteOffset, f.length);
    const top = d.get("pmgTop")!;
    for (let j = 0; j < LIST_SLOTS; j++) {
      const kind = j <= 10 ? "F" : j <= 15 ? "C" : j === 16 ? "B" : j === 18 ? "C" : "F";
      if (kind === "B") continue;
      mapU(kind === "F" ? F : Cc, kind === "F" ? "f" : "c", `list${j}`)(set, ff, p, S.lists[j]!, S.listCap[j]!);
    }
    for (let e = 0; e < L.cap.t; e++) { const at = S.tlist + 2 * e, v = set[at]!; if (v === NONE) continue; const c = v >>> 9; if (c < Cc) set[at] = ((p.c[c]! << 9) | (v & 511)) >>> 0; else bad("tlist"); }
    for (let i = D.page; i < D.page + L.window[0] * L.window[1] * L.window[2]; i++) { const v = dir[i]!; if (v === NONE || v >>> 31) { if (v !== NONE && (v >>> 29) !== 5) bad(`page.kind${v >>> 29}`); continue; } if (v < Cc) dir[i] = p.c[v]!; else bad("page"); }
    for (const o of [M.HBL, M.HPL, M.ZLL, M.RLL]) mapF32(F, "f", "pmgF.live")(f, ff, p, o, F);
    mapU(Cc, "c", "SLOT5")(top, ff, p, T.SLOT5, Cc);
    mapU(F, "f", "RB")(top, ff, p, T.RB, F);
    for (let i = T.SB; i < T.SB + 4 * F; i++) { const v = top[i]!; if (v === NONE) continue; const sl = v >>> 2; if (sl < F) top[i] = ((p.f[sl]! << 2) | (v & 3)) >>> 0; else bad("SB"); }
  };
  // Buffers that hold no F or C slot index or value (by their layouts): kept word for word.
  const KEPT = new Set(["args", "cut", "parts", "xfer", "solid", "solidEdit", "src", "boot", "pmgRed", "pmgB", "fpart", "fmoved", "pEnergy"]);
  if (!tiered) KEPT.add("fext4");
  const PERMUTED = new Set(regions.map(r => r.buf));
  for (const n of orig.keys()) if (!KEPT.has(n) && !PERMUTED.has(n)) throw new Error(`slot-order: buffer ${n} is in neither table`);
  // Sizes: every computed region inside its buffer; the layouts' totals equal the buffers'.
  const expect: Record<string, number> = { vel: L.poolWords.vel, phi: L.poolWords.phi, pres: L.poolWords.pres, set: S.words, dir: D.words, fcell: FL.fcell.words, fdep: FL.fdep.words,
    pmgF: M.fWords, pmgC: M.cWords, pmgTop: T.words, census: CENSUS.STRIDE * Cc, ...(tiered ? { fext4: FL.fext4.words } : {}), ...(orig.has("staticNeed") ? { staticNeed: brickStaticNeedWords(L) } : {}) };
  for (const [n, w] of Object.entries(expect)) if (orig.has(n) && Math.max(w, 4) !== words(n)) throw new Error(`slot-order: ${n} is ${words(n)} words, its layout ${w}`);
  for (const r of regions) { const cap = r.kind === "F" ? F : Cc, end = r.base + (r.nch ?? 1) * cap * r.m; if (end > words(r.buf)) throw new Error(`slot-order: region ${r.buf}@${r.base} ends at ${end} past ${words(r.buf)}`); }

  // ---- the frozen state's slots ----
  const set0 = orig.get("set")!, dir0 = orig.get("dir")!;
  const keyF = (d: Uint32Array, sl: number) => d[D.frec + RW * sl + 29]!, keyC = (d: Uint32Array, sl: number) => d[D.crec + RW * sl + 29]!;
  const liveOf = (d: Uint32Array, cap: number, key: (d: Uint32Array, s: number) => number) => { const v: number[] = []; for (let i = 0; i < cap; i++) if (key(d, i) !== NONE) v.push(i); return v; };
  const liveF = liveOf(dir0, F, keyF), liveC = liveOf(dir0, Cc, keyC);
  const check = (what: string, ok: boolean) => { if (!ok) throw new Error(`slot-order: frozen state check failed: ${what}`); };
  check(`LIVE_F ${set0[H.LIVE_F]} = keyed F slots ${liveF.length}`, set0[H.LIVE_F] === liveF.length);
  check(`LIVE_C ${set0[H.LIVE_C]} = keyed C slots ${liveC.length}`, set0[H.LIVE_C] === liveC.length);
  check("TOP_F + live F = cap", set0[H.TOP_F]! + liveF.length === F);
  check("TOP_C + live C = cap", set0[H.TOP_C]! + liveC.length === Cc);
  check(`LIVE_B ${set0[H.LIVE_B]} = 0`, set0[H.LIVE_B] === 0);
  const freeSet = (o: number, top: number) => new Set(Array.from(set0.subarray(o, o + top)));
  { const fr = freeSet(S.freeF, set0[H.TOP_F]!); check("F free stack and live slots partition the pool", fr.size === set0[H.TOP_F] && liveF.every(x => !fr.has(x))); }
  { const fr = freeSet(S.freeC, set0[H.TOP_C]!); check("C free stack and live slots partition the pool", fr.size === set0[H.TOP_C] && liveC.every(x => !fr.has(x))); }
  { const l = new Set(Array.from(set0.subarray(S.lists[LIST.F_ALL]!, S.lists[LIST.F_ALL]! + set0[H.LIST_COUNT + LIST.F_ALL]!))); check("F_ALL = keyed F slots", l.size === liveF.length && liveF.every(x => l.has(x))); }
  { const l = new Set(Array.from(set0.subarray(S.lists[LIST.C_ALL]!, S.lists[LIST.C_ALL]! + set0[H.LIST_COUNT + LIST.C_ALL]!))); check("C_ALL = keyed C slots", l.size === liveC.length && liveC.every(x => l.has(x))); }
  check(`HIGH_F ${set0[H.HIGH_F]} covers the live F slots`, liveF.every(x => x < set0[H.HIGH_F]!));
  check(`HIGH_C ${set0[H.HIGH_C]} covers the live C slots`, liveC.every(x => x < set0[H.HIGH_C]!));
  log(`frozen state: step ${P.steps}, F live ${liveF.length} high ${set0[H.HIGH_F]} cap ${F}; C live ${liveC.length} high ${set0[H.HIGH_C]} cap ${Cc}; K live ${set0[H.LIVE_K]}; particles ${set0[H.PCOUNT0]}/${set0[H.PCOUNT1]}; tiered ${tiered}`);

  // ---- permutations ----
  const spread3 = (x: number) => { x &= 0x3ff; x = (x | (x << 16)) & 0x30000ff; x = (x | (x << 8)) & 0x300f00f; x = (x | (x << 4)) & 0x30c30c3; x = (x | (x << 2)) & 0x9249249; return x; };
  const morton = (k: number) => spread3(k) | (spread3(k >>> 10) << 1) | (spread3(k >>> 20) << 2);
  let seed = 0x9e3779b9;
  const rnd = () => { seed ^= seed << 13; seed >>>= 0; seed ^= seed >>> 17; seed ^= seed << 5; seed >>>= 0; return seed / 2 ** 32; };
  const bijection = (cap: number, order: number[]) => {
    const p = new Int32Array(cap).fill(-1); order.forEach((o, i) => { p[o] = i; });
    let n = order.length; for (let i = 0; i < cap; i++) if (p[i] < 0) p[i] = n++;
    return p;
  };
  const shuffle = (cap: number) => { const a = Array.from({ length: cap }, (_, i) => i); for (let i = cap - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [a[i], a[j]] = [a[j]!, a[i]!]; } const p = new Int32Array(cap); a.forEach((v, i) => { p[i] = v; }); return p; };
  const perms: Record<string, Pm | undefined> = {
    identity: undefined,
    compact: { f: bijection(F, liveF), c: bijection(Cc, liveC) },
    morton: { f: bijection(F, [...liveF].sort((a, b) => morton(keyF(dir0, a)) - morton(keyF(dir0, b)))), c: bijection(Cc, [...liveC].sort((a, b) => morton(keyC(dir0, a)) - morton(keyC(dir0, b)))) },
    scatter: { f: shuffle(F), c: shuffle(Cc) },
  };
  function apply(name: string, p: Pm): Map<string, Uint32Array> {
    const outB = new Map<string, Uint32Array>();
    for (const n of PERMUTED) outB.set(n, orig.get(n)!.slice());
    for (const r of regions) {
      const src = orig.get(r.buf)!, dst = outB.get(r.buf)!, fl = new Float32Array(dst.buffer, dst.byteOffset, dst.length);
      const cap = r.kind === "F" ? F : Cc, q = r.kind === "F" ? p.f : p.c;
      for (let ch = 0; ch < (r.nch ?? 1); ch++) {
        const b = r.base + ch * cap * r.m;
        for (let sl = 0; sl < cap; sl++) {
          const o = b + sl * r.m, nw = b + q[sl]! * r.m;
          dst.set(src.subarray(o, o + r.m), nw);
          if (r.vals) r.vals(dst, fl, p, nw, r.m);
        }
      }
    }
    lists(outB, p);
    // Free stacks, high water: compaction pops live, live + 1, ...; the scatter keeps the stack's order through p.
    const set = outB.get("set")!;
    for (const [kind, top, high, fo, cap, live, q] of [["F", H.TOP_F, H.HIGH_F, S.freeF, F, liveF, p.f], ["C", H.TOP_C, H.HIGH_C, S.freeC, Cc, liveC, p.c]] as const) {
      const n = cap - live.length;
      if (name === "scatter") { for (let i = 0; i < n; i++) set[fo + i] = q[set0[fo + i]!]!; set[high] = Math.max(...live.map(x => q[x]!)) + 1; }
      else { for (let i = 0; i < cap; i++) set[fo + i] = cap - 1 - i; set[high] = live.length; }
      if (set[top] !== n) throw new Error(`slot-order: TOP_${kind}`);
    }
    return outB;
  }

  // ---- locality of a snapshot (footprint-analyze.mts's measures: F_ALL bricks, C_ALL pages) ----
  const quant = (v: number[], q: number) => { const a = [...v].sort((x, y) => x - y); return a.length ? a[Math.min(a.length - 1, Math.floor(q * a.length))]! : NaN; };
  const mean = (v: number[]) => v.reduce((a, b) => a + b, 0) / Math.max(v.length, 1);
  function locality(d: Map<string, Uint32Array>) {
    const dir = d.get("dir")!, set = d.get("set")!;
    const pages = (slots: number[]) => { const pg = new Set<number>(); for (const sl of slots) { const a = 4 * ST * sl; pg.add(Math.floor(a / 16384)); pg.add(Math.floor((a + 4 * ST - 1) / 16384)); } return pg.size; };
    const fl = Array.from(set.subarray(S.lists[LIST.F_ALL]!, S.lists[LIST.F_ALL]! + set[H.LIST_COUNT + LIST.F_ALL]!));
    const fSpread: number[] = [], fPages: number[] = [];
    for (const sl of fl) {
      const n = [sl]; for (let w = 0; w < 27; w++) { const v = dir[D.frec + RW * sl + w]!; if (v < 0x80000000) n.push(v); }
      fSpread.push(Math.max(...n) - Math.min(...n)); fPages.push(pages(n));
    }
    const [wx, wy, wz] = L.window, bx = Math.log2(wx), by = Math.log2(wy);
    const cl = Array.from(set.subarray(S.lists[LIST.C_ALL]!, S.lists[LIST.C_ALL]! + set[H.LIST_COUNT + LIST.C_ALL]!));
    const cSpread: number[] = [], cPages: number[] = [];
    for (const cs of cl) {
      const k = keyC(dir, cs), px = k & 0x3ff, py = (k >>> 10) & 0x3ff, pz = (k >>> 20) & 0x3ff, n = [cs];
      for (let dz = -1; dz <= 1; dz++) for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        if (!dx && !dy && !dz) continue;
        const qx = px + dx, qy = py + dy, qz = pz + dz; if (qx < 0 || qy < 0 || qz < 0) continue;
        const v = dir[D.page + ((qx & (wx - 1)) | ((qy & (wy - 1)) << bx) | ((qz & (wz - 1)) << (bx + by)))]!;
        if (v < 0x80000000 && keyC(dir, v) === ((qx | (qy << 10) | (qz << 20)) >>> 0)) n.push(v);
      }
      cSpread.push(Math.max(...n) - Math.min(...n)); cPages.push(pages(n));
    }
    return { highF: set[H.HIGH_F], highC: set[H.HIGH_C], f: { n: fl.length, spreadP50: quant(fSpread, 0.5), spreadP90: quant(fSpread, 0.9), pagesMean: mean(fPages), pagesP90: quant(fPages, 0.9) },
      c: { n: cl.length, spreadP50: quant(cSpread, 0.5), spreadP90: quant(cSpread, 0.9), pagesMean: mean(cPages), pagesP90: quant(cPages, 0.9) } };
  }

  // ---- GPU state: snapshots as shadows (census.mts) ----
  const COPY = GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
  const engineBufs: [string, GPUBuffer][] = [...P.buffers.entries()], ext: GPUBuffer[] = [...new Set([...P.external.values()])].filter((b: GPUBuffer) => (b.usage & COPY) === COPY);
  const copyAll = (from: GPUBuffer[], to: GPUBuffer[]) => { const enc = raw.createCommandEncoder(); from.forEach((b, i) => enc.copyBufferToBuffer(b, 0, to[i]!, 0, b.size)); raw.queue.submit([enc.finish()]); };
  const live = [...engineBufs.map(([, b]) => b), ...ext];
  const shadowNow = () => { const c = live.map(b => raw.createBuffer({ size: b.size, usage: COPY })); copyAll(live, c); return { restore: () => copyAll(c, live), destroy: () => c.forEach(x => x.destroy()) }; };
  const base0 = shadowNow();
  const upload = (d: Map<string, Uint32Array>) => { for (const [n, w] of d) P.write(n, 0, w); };
  const fence = () => raw.queue.onSubmittedWorkDone();
  const variantState = (name: string) => { base0.restore(); const p = perms[name]; const d = p ? apply(name, p) : undefined; if (d) upload(d); return d; };

  // ---- the engine's recorded steps from a snapshot ----
  const runSteps = async (k: number) => {
    for (let j = 0; j < k; j++) {
      const st = P.steps + j, slot = st % RING_SLOTS;
      raw.queue.writeBuffer(P.buffers.get("set"), 4 * (L.set.ring + RING_WORDS * slot), P.ringWords(st));
      const rec = P.recorded[slot], args = P.buffers.get("args"), enc = raw.createCommandEncoder();
      for (const list of [rec.part1, rec.part2]) { const pass = enc.beginComputePass(); for (const x of list) dispatchInto(pass, x, args); pass.end(); }
      raw.queue.submit([enc.finish()]);
    }
    await fence();
  };
  // Observables by key: per F brick and C page, the φ sum, the liquid vertices (φ < 0) and the U sums; the particles.
  const phiCh = 0, nV = POOLS.vel.length, nP = POOLS.phi.length;
  async function observe() {
    const set = await P.read("set"), dir = await P.read("dir"), phi = new Float32Array((await P.read("phi")).buffer), vel = new Float32Array((await P.read("vel")).buffer);
    const parts = new Float32Array((await P.read("parts")).buffer), pc = L.cap.particles, stride = L.particleStride;
    const per = (kind: "F" | "C") => {
      const cap = kind === "F" ? F : Cc, m = new Map<number, number[]>();
      for (let sl = 0; sl < cap; sl++) {
        const k = kind === "F" ? keyF(dir, sl) : keyC(dir, sl); if (k === NONE) continue;
        const pb = (kind === "F" ? phiCh * F * ST : nP * F * ST + phiCh * Cc * ST) + sl * ST;
        let ps = 0, neg = 0; for (let i = 0; i < 512; i++) { const v = phi[pb + i]!; ps += v; if (v < 0) neg++; }
        const u = [0, 0, 0]; for (let a = 0; a < 3; a++) { const vb = (kind === "F" ? a * F * ST : nV * F * ST + a * Cc * ST) + sl * ST; for (let i = 0; i < 512; i++) u[a] += vel[vb + i]!; }
        m.set(k, [ps, neg, ...u]);
      }
      return m;
    };
    const banks = [0, 1].map(b => { const n = set[H.PCOUNT0 + b]!; let sx = 0, sy = 0, sz = 0; for (let i = 0; i < Math.min(n, pc); i++) { const o = (b * pc + i) * stride; sx += parts[o]!; sy += parts[o + 1]!; sz += parts[o + 2]!; } return { n, sum: [sx, sy, sz] }; });
    return { fail: set[H.FAIL], failInfo: set[H.FAIL_INFO], liveF: set[H.LIVE_F], liveC: set[H.LIVE_C], f: per("F"), c: per("C"), banks };
  }
  type Obs = Awaited<ReturnType<typeof observe>>;
  function compare(a: Obs, b: Obs) {
    const side = (x: Map<number, number[]>, y: Map<number, number[]>) => {
      let only = 0, maxPhi = 0, maxNeg = 0, maxU = 0, negA = 0, negB = 0, phiA = 0, phiB = 0;
      const onlyKeys: string[] = [];
      const kstr = (k: number, v: number[], side: string) => `${side}:${k & 0x3ff},${(k >>> 10) & 0x3ff},${k >>> 20} neg ${v[1]} phi ${v[0]!.toFixed(1)}`;
      for (const [k, v] of x) { const w = y.get(k); negA += v[1]!; phiA += v[0]!; if (!w) { only++; if (onlyKeys.length < 6) onlyKeys.push(kstr(k, v, "a")); continue; } maxPhi = Math.max(maxPhi, Math.abs(v[0]! - w[0]!)); maxNeg = Math.max(maxNeg, Math.abs(v[1]! - w[1]!)); for (let a = 2; a < 5; a++) maxU = Math.max(maxU, Math.abs(v[a]! - w[a]!)); }
      for (const [k, w] of y) { negB += w[1]!; phiB += w[0]!; if (!x.has(k)) { only++; if (onlyKeys.length < 6) onlyKeys.push(kstr(k, w, "b")); } }
      return { keys: x.size, keysOnlyOne: only, onlyKeys, maxBrickPhiSum: maxPhi, maxBrickNegCount: maxNeg, maxBrickUSum: maxU, liquidVertices: [negA, negB], phiTotal: [phiA, phiB] };
    };
    return { fail: [a.fail, b.fail], liveF: [a.liveF, b.liveF], liveC: [a.liveC, b.liveC], f: side(a.f, b.f), c: side(a.c, b.c),
      particles: a.banks.map((x, i) => ({ n: [x.n, b.banks[i]!.n], maxPosSum: Math.max(...x.sum.map((v, j) => Math.abs(v - b.banks[i]!.sum[j]!))) })) };
  }
  const dispatchInto = (pass: GPUComputePassEncoder, r: any, args: GPUBuffer) => {
    pass.setPipeline(r.pipeline); pass.setBindGroup(0, r.group);
    if ("direct" in r.dispatch) pass.dispatchWorkgroups(...(r.dispatch.direct as [number, number, number])); else pass.dispatchWorkgroupsIndirect(args, 4 * r.dispatch.indirect);
  };

  const result: any = { scene: sceneId, key: sceneKey, warmup: N, R, date: new Date().toISOString(), steps: P.steps, cap: L.cap, tiered,
    git: (() => { try { return { head: execFileSync("git", ["rev-parse", "--short", "HEAD"], { cwd: root, encoding: "utf8" }).trim(), status: execFileSync("git", ["status", "--short"], { cwd: root, encoding: "utf8" }) }; } catch { return null; } })(),
    live: { f: liveF.length, c: liveC.length, highF: set0[H.HIGH_F], highC: set0[H.HIGH_C] }, locality: {}, proof: {}, anomalies: {} };
  const write = () => writeFileSync(resolve(out, "slot-order.json"), JSON.stringify(result, null, 1));

  // ---- proof: 1 and 4 recorded steps from each snapshot, by key ----
  const shadows = new Map<string, ReturnType<typeof shadowNow>>();
  const PROOF = (process.env.PROOF_STEPS ?? "1,4").split(",").map(Number), REPS = Number(process.env.ID_REPS ?? 2);
  const obs = new Map<string, Map<number, Obs>>();
  const take = async (label: string, sh: ReturnType<typeof shadowNow>) => { const m = new Map<number, Obs>(); obs.set(label, m); for (const k of PROOF) { sh.restore(); await runSteps(k); m.set(k, await observe()); } };
  for (const name of ["scatter", "identity", "compact", "morton"]) {
    const d = variantState(name);
    result.locality[name] = locality(d ?? orig);
    const sh = shadowNow();
    await take(name, sh);
    if (name === "identity") for (let r = 2; r <= REPS; r++) await take(`identity${r}`, sh);
    if (name === "scatter" || eqOnly) sh.destroy(); else shadows.set(name, sh);
    log(`${name}: locality ${JSON.stringify(result.locality[name])}`);
    if (gpuErrors.length) throw new Error(`GPU error in the proof: ${gpuErrors[0]}`);
  }
  result.anomalies = Object.fromEntries(anomalies);
  for (const name of [...obs.keys()].filter(n => n !== "identity")) {
    result.proof[name] = Object.fromEntries(PROOF.map(k => [`step${k}`, compare(obs.get("identity")!.get(k)!, obs.get(name)!.get(k)!)]));
    log(`proof ${name} vs identity: ${JSON.stringify(result.proof[name])}`);
  }
  log(`anomalies (slot-looking words outside the cap): ${JSON.stringify(result.anomalies)}`);
  write();
  if (eqOnly) { base0.destroy(); s.destroy?.(); process.exit(gpuErrors.length ? 3 : 0); }

  // ---- timing: census replay and wall, interleaved ----
  const V = ["identity", "compact", "morton"];
  const QMAX = 4096, qs = raw.createQuerySet({ type: "timestamp", count: QMAX });
  const qResolve = raw.createBuffer({ size: 8 * QMAX, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
  const qRead = raw.createBuffer({ size: 8 * QMAX, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const readTimes = async (enc: GPUCommandEncoder, n: number) => {
    enc.resolveQuerySet(qs, 0, n, qResolve, 0); enc.copyBufferToBuffer(qResolve, 0, qRead, 0, 8 * n); raw.queue.submit([enc.finish()]);
    await qRead.mapAsync(GPUMapMode.READ); const t = new BigUint64Array(qRead.getMappedRange(0, 8 * n).slice(0)); qRead.unmap(); return t;
  };
  const us = (t: BigUint64Array, i: number) => Number(t[i + 1]! - t[i]!) / 1e3;
  const slot = P.steps % RING_SLOTS, rec = P.recorded[slot], all = [...rec.part1, ...rec.part2], args = P.buffers.get("args");
  const sch = [...P.schedule.part1, ...P.schedule.part2];
  if (sch.length !== all.length) throw new Error("slot-order: recorded placements differ from the schedule");
  const ring = () => raw.queue.writeBuffer(P.buffers.get("set"), 4 * (L.set.ring + RING_WORDS * slot), P.ringWords(P.steps));
  const times = new Map(V.map(v => [v, { wall: [] as number[], part1: [] as number[], part2: [] as number[], place: all.map(() => [] as number[]) }]));
  for (let r = 0; r < R; r++) for (const v of V) {
    const sh = shadows.get(v)!, tv = times.get(v)!;
    sh.restore(); ring();
    { const enc = raw.createCommandEncoder();
      [rec.part1, rec.part2].forEach((list: any[], pi: number) => { const pass = enc.beginComputePass({ timestampWrites: { querySet: qs, beginningOfPassWriteIndex: 2 * pi, endOfPassWriteIndex: 2 * pi + 1 } }); for (const x of list) dispatchInto(pass, x, args); pass.end(); });
      const t = await readTimes(enc, 4); tv.part1.push(us(t, 0)); tv.part2.push(us(t, 2)); tv.wall.push(us(t, 0) + us(t, 2)); }
    sh.restore(); ring();
    { const enc1 = raw.createCommandEncoder(), enc2 = raw.createCommandEncoder();
      all.forEach((x: any, j: number) => { const pass = (j < rec.part1.length ? enc1 : enc2).beginComputePass({ timestampWrites: { querySet: qs, beginningOfPassWriteIndex: 2 * j, endOfPassWriteIndex: 2 * j + 1 } }); dispatchInto(pass, x, args); pass.end(); });
      raw.queue.submit([enc1.finish()]); const t = await readTimes(enc2, 2 * all.length);
      all.forEach((_: unknown, j: number) => tv.place[j]!.push(us(t, 2 * j))); }
    if (gpuErrors.length) throw new Error(`GPU error in the timing: ${gpuErrors[0]}`);
  }
  const min = (v: number[]) => Math.min(...v), med = (v: number[]) => { const a = [...v].sort((x, y) => x - y); return a[a.length >> 1]!; };
  result.timing = {};
  for (const v of V) {
    const tv = times.get(v)!, byEntry: Record<string, number> = {};
    sch.forEach((p: any, j: number) => { byEntry[p.kernel.entry] = (byEntry[p.kernel.entry] ?? 0) + min(tv.place[j]!); });
    result.timing[v] = { wallMin: min(tv.wall), wallMed: med(tv.wall), part1Min: min(tv.part1), part2Min: min(tv.part2), sumPlacements: Object.values(byEntry).reduce((a, b) => a + b, 0), byEntry };
    log(`${v}: wall min ${(min(tv.wall) / 1000).toFixed(3)} ms (med ${(med(tv.wall) / 1000).toFixed(3)}), sum of placements ${(result.timing[v].sumPlacements / 1000).toFixed(3)} ms`);
  }
  const top = Object.entries(result.timing.identity.byEntry as Record<string, number>).sort((a, b) => b[1] - a[1]).slice(0, 12).map(([e]) => e);
  result.heavy = top;
  for (const e of top) log(`${e.padEnd(16)} ${V.map(v => (result.timing[v].byEntry[e] / 1000).toFixed(3)).join("  ")} ms`);
  for (const v of V) { result.timing[v].heavy = top.reduce((a, e) => a + result.timing[v].byEntry[e], 0); }
  log(`heavy-12 ${V.map(v => `${v} ${(result.timing[v].heavy / 1000).toFixed(3)} ms`).join(", ")}`);
  write();

  // ---- counters loop (the parent attaches Instruments) ----
  if (LOOP_MS > 0) {
    log("READY");
    const tEnd = performance.now() + LOOP_MS; let cycles = 0;
    while (performance.now() < tEnd) {
      for (const v of V) {
        shadows.get(v)!.restore(); ring();
        const enc = raw.createCommandEncoder();
        all.forEach((x: any, j: number) => { // timestampWrites keep each pass its own Metal encoder (Dawn merges plain passes: one interval, no per-kernel counters).
        const pass = enc.beginComputePass({ label: `m0d.${v}|${sch[j].kernel.entry}`, timestampWrites: { querySet: qs, beginningOfPassWriteIndex: 2 * j, endOfPassWriteIndex: 2 * j + 1 } }); dispatchInto(pass, x, args); pass.end(); });
        raw.queue.submit([enc.finish()]); await fence();
      }
      cycles++;
    }
    log(`DONE ${cycles} cycles`);
    result.counterCycles = cycles; write();
  }
  for (const sh of shadows.values()) sh.destroy();
  base0.destroy();
  s.destroy?.();
  process.exit(gpuErrors.length ? 3 : 0);
}
