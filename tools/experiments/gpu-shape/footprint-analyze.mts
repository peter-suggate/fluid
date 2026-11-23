// GPU-shape M0c (lane B), offline (no GPU): the frozen state footprint.mts dumped, reduced to
//  - slot-order locality: for each live F brick (F_ALL), the spread (max - min) of the slot indices of itself and its
//    live neighbours (record words 0..26), and the distinct 16 KB pages of one channel those bricks' 2080-byte slot
//    regions cover; the same for C pages (C_ALL, C records). Also the same for a hypothetical slot order (the live
//    bricks renumbered by the Morton order of their keys), the "spatially ordered slots" lever;
//  - per heavy kernel: the pool channels and lane-F regions its entry point can reach (a call graph over the module's
//    WGSL from the entry), the bytes those regions span at the live slots' high water (HIGH_F/HIGH_C/HIGH_K) and at the
//    live counts, the allocated bytes of its bindings, and its static load sites (an estimate, see the doc).
//   node --import tsx tools/experiments/gpu-shape/footprint-analyze.mts <m0c dir> <scene> [census dir]
import { readFileSync, writeFileSync, existsSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { homedir } from "node:os";

const here = dirname(fileURLToPath(import.meta.url)), root = resolve(here, "../../..");
const { H, LIST, REC, POOLS, BRICK_STRIDE, NONE, SEAM } = await import(`${root}/lib/methods/narrow-band/brick-abi.ts`);
const [dirArg, scene, censusArg] = process.argv.slice(2);
const base = resolve(dirArg!, scene!);
const censusDir = resolve(censusArg ?? resolve(root, "docs/plans/gpu-shape/census"), scene!);
const kcache = resolve(homedir(), "code/me/fluid-scratch/gpu-shape/kcache");
const meta = JSON.parse(readFileSync(resolve(base, "meta.json"), "utf8"));
const PAGE = 16384, SLOT_BYTES = 4 * BRICK_STRIDE;
const out: any = { scene, steps: meta.steps, git: meta.git?.head, engines: [] };
const lines: string[] = [];
const say = (s: string) => { lines.push(s); console.log(s); };

const words = (f: string) => { const b = readFileSync(f); return new Uint32Array(b.buffer, b.byteOffset, b.byteLength / 4); };
const morton = (x: number, y: number, z: number) => { let m = 0n; for (let i = 0; i < 10; i++) m |= (BigInt((x >> i) & 1) << BigInt(3 * i)) | (BigInt((y >> i) & 1) << BigInt(3 * i + 1)) | (BigInt((z >> i) & 1) << BigInt(3 * i + 2)); return m; };
const keyXYZ = (k: number) => [k & 1023, (k >>> 10) & 1023, (k >>> 20) & 1023] as const;
const hist = (v: number[], edges: number[]) => { const h = new Array(edges.length + 1).fill(0); for (const x of v) { let i = 0; while (i < edges.length && x > edges[i]!) i++; h[i]++; } return h; };
const q = (v: number[], p: number) => { const s = [...v].sort((a, b) => a - b); return s[Math.min(s.length - 1, Math.floor(p * s.length))] ?? NaN; };
const mean = (v: number[]) => v.reduce((a, b) => a + b, 0) / Math.max(1, v.length);
const PAGE_EDGES = [4, 8, 12, 16, 20, 24, 27, 32, 40];
const SPREAD_EDGES = [27, 64, 256, 1024, 4096, 16384];

for (const E of meta.engines) {
  const set = words(resolve(base, `${E.name}.set.bin`)), dir = words(resolve(base, `${E.name}.dir.bin`));
  const hdr = (k: keyof typeof H) => set[H[k]]!;
  const list = (j: number) => { const n = Math.min(set[H.LIST_COUNT + j]!, E.set.listCap[j]); return Array.from(set.subarray(E.set.lists[j], E.set.lists[j] + n)); };
  const fAll = list(LIST.F_ALL), cAll = list(LIST.C_ALL);
  const head = { LIVE_F: hdr("LIVE_F"), HIGH_F: hdr("HIGH_F"), TOP_F: hdr("TOP_F"), LIVE_C: hdr("LIVE_C"), HIGH_C: hdr("HIGH_C"), LIVE_K: hdr("LIVE_K"), HIGH_K: hdr("HIGH_K"),
    PCOUNT0: hdr("PCOUNT0"), PCOUNT1: hdr("PCOUNT1"), F_ALL: fAll.length, C_ALL: cAll.length, F_ACT: list(LIST.F_ACT).length, F_LIQ: list(LIST.F_LIQ).length, C_ACT: list(LIST.C_ACT).length, cap: E.cap };
  say(`\n## ${scene} ${E.name} (step ${E.steps}): F live ${head.LIVE_F} / high ${head.HIGH_F} / cap ${E.cap.f}; C live ${head.LIVE_C} / high ${head.HIGH_C} / cap ${E.cap.c}; K live ${head.LIVE_K} / high ${head.HIGH_K} / cap ${E.cap.k}; particles ${head.PCOUNT0}/${head.PCOUNT1} of ${E.cap.particles}; F_ALL ${fAll.length} F_ACT ${head.F_ACT} C_ALL ${cAll.length} C_ACT ${head.C_ACT}`);
  // Is the F_ALL list in slot order?
  let inversions = 0; for (let i = 1; i < fAll.length; i++) if (fAll[i]! < fAll[i - 1]!) inversions++;
  // ---- locality ----
  const loc = (recBase: number, slots: number[], chanBase: number, neighbours?: (s: number) => number[]) => {
    const spread: number[] = [], pages: number[] = [], nbs: number[] = [];
    const keyOf = (s: number) => dir[recBase + 32 * s + REC.KEY]!;
    // Morton renumbering of the live slots.
    const order = [...slots].sort((a, b) => { const ka = keyXYZ(keyOf(a)), kb = keyXYZ(keyOf(b)); const ma = morton(...ka), mb = morton(...kb); return ma < mb ? -1 : ma > mb ? 1 : 0; });
    const rank = new Map<number, number>(); order.forEach((s, i) => rank.set(s, i));
    const spreadM: number[] = [], pagesM: number[] = [];
    const pageSet = (ss: number[]) => { const p = new Set<number>(); for (const s of ss) { const a = chanBase + SLOT_BYTES * s; p.add(Math.floor(a / PAGE)); p.add(Math.floor((a + SLOT_BYTES - 1) / PAGE)); } return p.size; };
    for (const s of slots) {
      const r = recBase + 32 * s, ns: number[] = [];
      if (neighbours) ns.push(...neighbours(s)); else for (let j = 0; j < 27; j++) { const v = dir[r + j]!; if (v < SEAM) ns.push(v); }
      if (!ns.includes(s)) ns.push(s);
      nbs.push(ns.length);
      spread.push(Math.max(...ns) - Math.min(...ns)); pages.push(pageSet(ns));
      const m = ns.map(x => rank.get(x) ?? x); spreadM.push(Math.max(...m) - Math.min(...m)); pagesM.push(pageSet(m));
    }
    return { n: slots.length, neighbours: { mean: mean(nbs) }, spread: { p10: q(spread, 0.1), p50: q(spread, 0.5), p90: q(spread, 0.9), hist: hist(spread, SPREAD_EDGES) },
      pages: { mean: mean(pages), p50: q(pages, 0.5), p90: q(pages, 0.9), hist: hist(pages, PAGE_EDGES) }, pagesPerBrick: mean(pages.map((p, i) => p / nbs[i]!)),
      morton: { spreadP50: q(spreadM, 0.5), spreadP90: q(spreadM, 0.9), pagesMean: mean(pagesM), pagesP50: q(pagesM, 0.5), pagesP90: q(pagesM, 0.9), hist: hist(pagesM, PAGE_EDGES) } };
  };
  // Channel 0 of the vel pool: the F region starts at byte 0, the C region after all F channels.
  // C records keep no neighbour words (setClaimPages leaves 0..26 NONE): a page's neighbours come from the page window.
  const W = E.window as number[], WB = W.map(n => Math.log2(n));
  const pageWord = (x: number, y: number, z: number) => dir[E.dir.page + ((x & (W[0]! - 1)) | ((y & (W[1]! - 1)) << WB[0]!) | ((z & (W[2]! - 1)) << (WB[0]! + WB[1]!)))]!;
  const cNeighbours = (cs: number) => {
    const [x, y, z] = keyXYZ(dir[E.dir.crec + 32 * cs + REC.KEY]!), ns: number[] = [];
    for (let dz = -1; dz <= 1; dz++) for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
      if (x + dx < 0 || y + dy < 0 || z + dz < 0) continue;
      const w = pageWord(x + dx, y + dy, z + dz); if (w >= SEAM) continue;
      const k = keyXYZ(dir[E.dir.crec + 32 * w + REC.KEY]!); if (k[0] === x + dx && k[1] === y + dy && k[2] === z + dz) ns.push(w);
    }
    return ns;
  };
  const fLoc = loc(E.dir.frec, fAll, 0), cLoc = loc(E.dir.crec, cAll, 4 * POOLS.vel.length * E.cap.f * BRICK_STRIDE, cNeighbours);
  say(`F_ALL list: ${inversions} descents in ${fAll.length} (0 = ascending slot order)`);
  for (const [name, L] of [["F", fLoc], ["C", cLoc]] as const)
    say(`${name} 27-neighbourhoods (n ${L.n}, mean ${L.neighbours.mean.toFixed(1)} live incl. self): slot spread p10/50/90 ${L.spread.p10}/${L.spread.p50}/${L.spread.p90} (hist ≤${SPREAD_EDGES.join("/≤")}/more: ${L.spread.hist.join(" ")}); ` +
      `16 KB pages of one channel mean ${L.pages.mean.toFixed(1)} p50 ${L.pages.p50} p90 ${L.pages.p90} (hist ≤${PAGE_EDGES.join("/≤")}/more: ${L.pages.hist.join(" ")}), ${L.pagesPerBrick.toFixed(2)} pages a brick | ` +
      `Morton-renumbered: spread p50/p90 ${L.morton.spreadP50}/${L.morton.spreadP90}, pages mean ${L.morton.pagesMean.toFixed(1)} p50 ${L.morton.pagesP50} p90 ${L.morton.pagesP90} (hist ${L.morton.hist.join(" ")})`);
  // Free stack state: the next pops.
  const top = head.TOP_F, nextPops = Array.from(set.subarray(E.set.freeF + Math.max(0, top - 8), E.set.freeF + top)).reverse();
  say(`F free stack: TOP ${top}, next pops ${nextPops.join(",")}; live slots ≥ HIGH: ${fAll.filter(s => s >= head.HIGH_F).length}; holes below HIGH_F: ${head.HIGH_F - fAll.length}`);

  // ---- kernels ----
  const census = existsSync(resolve(censusDir, "census-caps.tsv")) ? readFileSync(resolve(censusDir, "census-caps.tsv"), "utf8").trim().split("\n").map(l => l.split("\t")) : [];
  const ch = census[0] ?? [], ci = (n: string) => ch.indexOf(n);
  const timeBy = new Map<string, { us: number; lanes: number; n: number }>();
  for (const r of census.slice(1)) if (r[ci("engine")] === E.name) { const k = r[ci("entry")]!; const t = timeBy.get(k) ?? { us: 0, lanes: 0, n: 0 }; t.us += Number(r[ci("t_min_us")]); t.lanes += Number(r[ci("lanes")]); t.n++; timeBy.set(k, t); }
  const sched = [...E.schedule.part1, ...E.schedule.part2];
  const byEntry = new Map<string, any>(); for (const p of sched) if (!byEntry.has(p.entry)) byEntry.set(p.entry, p);
  const top12 = [...timeBy].sort((a, b) => b[1].us - a[1].us).slice(0, Number(process.env.TOP ?? 14));
  const fcellSlotBytes: Record<string, [string, number]> = { CNT: ["F", 2048], RUN: ["F", 2048], NEAR: ["F", 2048], NEAR2: ["F", 2048], KEY: ["F", 2048], HOT: ["F", 2048], DEPTH: ["F", 2048], MASK: ["F", 64], TILE: ["F", 384], SLOT: ["F", 4], BOUND: ["F", 4], UNIT: ["F", 8748], PHIU: ["F", 2048], HMASK: ["F", 32], CACHE4: ["C", 6144], LIQ: ["F", 64], DIL4: ["C", 2048], RES: ["F", 4] };
  const P = Math.max(head.PCOUNT0, head.PCOUNT1);
  const kernels: any[] = [];
  for (const [entry, t] of top12) {
    const p = byEntry.get(entry); if (!p) { say(`${entry}: not in the current schedule`); continue; }
    const code = readFileSync(resolve(kcache, `${p.hash}.wgsl`), "utf8").replace(/\/\*[\s\S]*?\*\//g, "").replace(/\/\/[^\n]*/g, "");
    // Functions and the call graph from the entry.
    const fns = new Map<string, string>(); const re = /\bfn\s+(\w+)\s*\(/g; let m: RegExpExecArray | null;
    while ((m = re.exec(code))) { let i = code.indexOf("{", m.index); let d = 1, j = i + 1; while (j < code.length && d) { if (code[j] === "{") d++; else if (code[j] === "}") d--; j++; } fns.set(m[1]!, code.slice(m.index, j)); }
    const reach = new Set<string>([entry]), stack = [entry];
    while (stack.length) { const f = stack.pop()!; const body = fns.get(f) ?? ""; for (const [g] of fns) if (!reach.has(g) && new RegExp(`\\b${g}\\s*\\(`).test(body.slice(body.indexOf("{")))) { reach.add(g); stack.push(g); } }
    const text = [...reach].map(f => fns.get(f) ?? "").join("\n");
    const used = (id: string) => new RegExp(`\\b${id}\\b`).test(text);
    const regions: { buf: string; region: string; kind: string; perSlot: number }[] = [];
    const bound = p.bindings.filter((b: any) => b[1] !== "uniform").map((b: any) => b[0] as string);
    for (const pool of ["vel", "phi", "pres"] as const) if (bound.includes(pool)) for (const c of POOLS[pool]) for (const kind of ["F", "C"]) {
      const id = `NB_${pool.toUpperCase()}_${kind}_${c}`; if (!used(id)) continue;
      // A vector's channel 0 indexed with an axis term (base + STRIDE (slot + a CAP)) reaches its three channels.
      const axes = c.endsWith("0") && new RegExp(`\\b${id}\\s*\\+\\s*NB_STRIDE\\s*\\*\\s*\\(\\s*\\w+\\s*\\+\\s*\\w+\\s*\\*\\s*NB_${kind}CAP`).test(text);
      regions.push({ buf: pool, region: `${kind}.${c}${axes ? "x3" : ""}`, kind, perSlot: (axes ? 3 : 1) * SLOT_BYTES });
    }
    if (bound.includes("cut")) for (const c of POOLS.cut) if (used(`NB_CUT_${c}`)) regions.push({ buf: "cut", region: c, kind: "K", perSlot: SLOT_BYTES });
    if (bound.includes("fcell")) for (const [r, [kind, b]] of Object.entries(fcellSlotBytes)) if (used(`FC_${r}`)) regions.push({ buf: "fcell", region: r, kind, perSlot: b });
    const perSlotBuf: Record<string, [string, number]> = { fdep: ["F", 12288], pmgF: ["F", 27864], pmgC: ["C", 64504], fext4: ["C", 36864] };
    const perParticle: Record<string, number> = { parts: 2 * 4 * E.particleStride, fmoved: 4 * E.particleStride, fpart: 8, pEnergy: 4 };
    const fixed: string[] = [];
    for (const b of bound) {
      if (perSlotBuf[b]) regions.push({ buf: b, region: "*", kind: perSlotBuf[b]![0], perSlot: perSlotBuf[b]![1] });
      else if (perParticle[b]) regions.push({ buf: b, region: "*", kind: "P", perSlot: perParticle[b]! });
      else if (!["vel", "phi", "pres", "cut", "fcell"].includes(b)) fixed.push(b);
    }
    const size = (b: string) => E.buffers[b] ?? E.external[b] ?? 0;
    const span = (kind: string, live: boolean) => kind === "F" ? (live ? head.LIVE_F : head.HIGH_F) : kind === "C" ? (live ? head.LIVE_C : head.HIGH_C) : kind === "K" ? (live ? head.LIVE_K : head.HIGH_K) : P;
    const byBuf = new Map<string, { high: number; live: number; alloc: number; regions: string[] }>();
    for (const r of regions) { const x = byBuf.get(r.buf) ?? { high: 0, live: 0, alloc: size(r.buf), regions: [] as string[] }; x.high += r.perSlot * span(r.kind, false); x.live += r.perSlot * span(r.kind, true); x.regions.push(r.region); byBuf.set(r.buf, x); }
    for (const b of fixed) byBuf.set(b, { high: size(b), live: size(b), alloc: size(b), regions: ["(whole)"] });
    const sum = (k: "high" | "live" | "alloc") => [...byBuf.values()].reduce((a, x) => a + x[k], 0);
    // Static load sites: a binding's indexing in the reachable functions (each site once, loops not unrolled).
    const sites: Record<string, number> = {};
    for (const b of bound) { const id = b === "set" ? "nbSet" : b; const n = (text.match(new RegExp(`(?<![\\w.])${id}\\s*\\[`, "g")) ?? []).length; if (n) sites[b] = n; }
    // F channels of brick-sized (>= 2 KB a slot) regions, and the 16 KB pages one 27-neighbourhood spans over them.
    const fCh = regions.filter(r => r.kind === "F" && r.perSlot >= 2048).reduce((a, r) => a + r.perSlot / (r.buf === "fcell" ? 2048 : SLOT_BYTES), 0);
    const wgPages = fCh * fLoc.pages.mean, wgPagesM = fCh * fLoc.morton.pagesMean;
    const k = { entry, fChannels: fCh, wgPages, wgPagesMorton: wgPagesM, us: t.us, lanes: t.lanes, placements: t.n, hash: p.hash, functions: reach.size, highMB: sum("high") / 1e6, liveMB: sum("live") / 1e6, allocMB: sum("alloc") / 1e6,
      buffers: Object.fromEntries([...byBuf].map(([b, x]) => [b, { highMB: +(x.high / 1e6).toFixed(2), liveMB: +(x.live / 1e6).toFixed(2), allocMB: +(x.alloc / 1e6).toFixed(2), regions: x.regions.join(",") }])), sites };
    kernels.push(k);
    say(`${entry.padEnd(16)} ${(t.us / 1000).toFixed(2)} ms ${t.lanes} lanes, ${fCh.toFixed(0)} F channels, ~${wgPages.toFixed(0)} pages a neighbourhood (Morton ${wgPagesM.toFixed(0)}): reach@high ${k.highMB.toFixed(1)} MB, @live ${k.liveMB.toFixed(1)} MB, bound alloc ${k.allocMB.toFixed(0)} MB | ` +
      [...byBuf].filter(([, x]) => x.high > 1e5).map(([b, x]) => `${b} ${(x.high / 1e6).toFixed(1)}/${(x.alloc / 1e6).toFixed(0)} [${x.regions.join(",")}]`).join("; ") + ` | sites ${JSON.stringify(sites)}`);
  }
  const total = Object.values(E.buffers as Record<string, number>).reduce((a, b) => a + b, 0) + Object.values(E.external as Record<string, number>).reduce((a, b) => a + b, 0);
  say(`engine buffers total ${(total / 1e6).toFixed(0)} MB`);
  out.engines.push({ name: E.name, head, inversions, fLoc, cLoc, nextPops, kernels, totalMB: total / 1e6 });
}
writeFileSync(resolve(base, "analysis.json"), JSON.stringify(out, null, 1));
writeFileSync(resolve(base, "analysis.txt"), lines.join("\n") + "\n");
