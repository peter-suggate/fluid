// setCensus (brick-census.wgsl.ts) on frozen states from the old NB-FLIP engine itself: the
// "Uniform dynamic ownership census" pass of high-resolution-dam-break at frames 14 (the collapse,
// the first frame whose every page is resident) and 43 (the splash), captured before and after the
// pass (parity/dam-census-<step>.{bin.gz,json}, tools/nbflip-census-capture.ts). It pins the tile
// census, keys, crossing cells, gaps, importance words, WANT and TRIG, against the old engine's real
// output rather than a transcription of it (brick-census.test.ts's oracle).
//
// - The window is the whole lattice (128^3), so no stencil reaches a boundary the old lattice does
//   not have, the reach (census[18]) is the same lattice's, and no tile is left out. Both frames
//   hold no AIR, SOLID or BULK16 page (the tool refuses one); frame 8 holds a BULK16 and an AIR
//   page, so it is not captured.
// - The pre state goes in by brick-abi.md §2.5: owners to CMASK (and the F records' word 28), φ to
//   F (h vertices) or C (4h corners) as φ/h, chosen so that the kernel's φ·h is the old metre word
//   exactly; the extension velocity (the old census's velocity texture) to US by the lower-face
//   convention; the join plane to TILE_OCC bits 8..15. Every word that holds no state holds poison.
// - Deep COLD (D = 0, a stated deviation): the old census replays a deep COLD region's records
//   and keeps its tiles' last decision (decidedPrev); setCensus classifies and decides them anew.
//   Their WANT and TRIG are reported, and kept out of the verdict; a non-deep tile whose stencil
//   reads a deep tile is reported with its distance to it. Then the replay (regionDeep, D on): the
//   old replayed regions go in deep COLD with their stamps, records and decidedPrev, every region's
//   mode must be the old one, and WANT the old WANT on every tile.
// - Split vertices: the old engine holds a 4-aligned vertex of an h tile's closure twice, in the h
//   store (umLoadFine reads it) and the base (umLoadCorner); at a few vertices the two disagree
//   (fixture split/splitPhi), and an h tile's scores read both. The brick storage holds one value,
//   so the census runs twice: the h store's value, then the base's at the split vertices. A tile
//   that differs under both, with no split vertex within a tile of it, is a mismatch of the port.
// - Exact: velocity keys, crossing cells, gaps, every flag, trigger and required bit, WANT, TRIG and
//   the counters. Tolerance (as brick-census.test.ts, the GPU may fuse a multiply-add and scores are
//   bytes of f32 products): score bytes within 1; FIN's winner bits follow the bytes. φ words with
//   no exact h pre-image go in as fr(φ/h) and are counted (none on these frames).
// Each capture is one run of a non-deterministic GPU simulation, so the counts belong to these files.
// GPU only.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import test from "node:test";
import { gunzipSync } from "node:zlib";
import { ARGS, BRICK_STRIDE, CMASK_WORDS, H, LIST, REC, RING } from "./brick-abi";
import { CENSUS, CENSUS_H, RG, brickCensusSchedule } from "./brick-census.wgsl";
import { decodeCensusFixture, type CensusFixtureMeta } from "./parity/census-fixture";
import { ownerOf, storedFaces, storedVertices, type V3 } from "./parity/remap-fixture";

const GPU = { skip: !process.env.WEBGPU_NODE_MODULE };
const PARITY = resolve(import.meta.dirname, "parity");
const POISON_PHI = 4321.5, POISON_U = -987.25;
const fr = Math.fround;
const key = (b: readonly number[]) => ((b[0]! & 1023) | ((b[1]! & 1023) << 10) | ((b[2]! & 1023) << 20)) >>> 0;
const local8 = (c: readonly number[]) => (c[0]! & 7) | ((c[1]! & 7) << 3) | ((c[2]! & 7) << 6);
const shr = (c: readonly number[], s: number): V3 => [c[0]! >> s, c[1]! >> s, c[2]! >> s];
const F32 = new Float32Array(1), U32 = new Uint32Array(F32.buffer);
const bitsOf = (x: number) => { F32[0] = x; return U32[0]!; };
const fromBits = (u: number) => { U32[0] = u >>> 0; return F32[0]!; };
const step1 = (x: number, d: number) => { const b = bitsOf(x); return fromBits(x === 0 ? (d > 0 ? 1 : 0x80000001) : (x > 0) === (d > 0) ? b + 1 : b - 1); };
/** A word v with fr(v·h) = x (the kernel's nbVertexPhi·h), or undefined. */
const preimage = (x: number, h: number) => {
  let v = fr(x / h);
  for (let n = 0; n < 8 && fr(v * h) !== x; n++) v = step1(v, fr(v * h) < x === h > 0 ? 1 : -1);
  return fr(v * h) === x ? v : undefined;
};
const CROSSES = 1 << 28, WET = 1 << 29, REQUIRED = 1 << 25;
const FLAGS = (0xffff0000 & ~(7 << 22)) >>> 0;
const byteNear = (a: number, b: number, shift: number) => Math.abs(((a >>> shift) & 255) - ((b >>> shift) & 255)) <= 1;

for (const step of [14, 43]) test(`setCensus (GPU): the old NB-FLIP engine's own dam census, frame ${step}`, GPU, async () => {
  const meta = JSON.parse(readFileSync(resolve(PARITY, `dam-census-${step}.json`), "utf8")) as CensusFixtureMeta;
  const fx = decodeCensusFixture(gunzipSync(readFileSync(resolve(PARITY, `dam-census-${step}.bin.gz`))), meta);
  const W = meta.window, T = W / 4, n = T ** 3;
  assert.deepEqual(meta.lattice, [W, W, W], "the window is the whole lattice");
  const hOld = ownerOf(fx.owner, T), verts = storedVertices(hOld, W), faces = storedFaces(hOld, W);
  assert.equal(verts.length, fx.phi.length); assert.equal(faces.length, fx.US.length);
  const ti = (t: readonly number[]) => t[0]! + T * (t[1]! + T * t[2]!);
  const coord = (i: number): V3 => [i % T, Math.floor(i / T) % T, Math.floor(i / (T * T))];
  // Chebyshev distance (tiles) from each tile to the nearest deep COLD tile (Infinity without one).
  const deepList: V3[] = []; for (let i = 0; i < n; i++) if (fx.deep[i]) deepList.push(coord(i));
  const deepDistance = (t: V3) => deepList.reduce((m, d) => Math.min(m, Math.max(...t.map((v, a) => Math.abs(v - d[a]!)))), Infinity);
  const h = fr(meta.h), dt = fr(meta.dt);
  const { withUniformDevice } = await import("../../../tests/helpers/uniform-geometric");
  const { BrickEngine } = await import("./brick-engine");
  const { bringUpCG } = await import("./brick-bringup-cg");
  const { bringUpStages } = await import("./brick-stages");
  const { snapshot } = await import("./brick-readback");
  await withUniformDevice("brick census dam", async device => {
    const nb = W / 8 + 1, np = Math.ceil((W + 1) / 32);
    const e = await BrickEngine.create(device, { lattice: [W, W, W], h_m: meta.h, dt_s: meta.dt, gravity_mps2: meta.gravity as V3,
      capacities: { f: nb ** 3, c: np ** 3, b: 0, k: 0, particles: 1024 }, liquid: [{ lo: [-1, -1, -1], hi: [W + 1, W + 1, W + 1] }], solver: L => bringUpCG(L), stages: bringUpStages });
    const s = await snapshot(e), L = e.layout;
    assert.equal(s.bricks.size, nb ** 3, "every brick resident"); assert.equal(s.pages.size, np ** 3, "every page resident");
    // ---- Owners: CMASK and word 28.
    const pageOf = new Map<number, V3>();
    for (const [k, cs] of s.pages) {
      const page: V3 = [k & 1023, (k >>> 10) & 1023, (k >>> 20) & 1023], m = new Uint32Array(CMASK_WORDS); pageOf.set(cs, page);
      for (let i = 0; i < 512; i++) if (hOld([8 * page[0] + (i & 7), 8 * page[1] + ((i >> 3) & 7), 8 * page[2] + (i >> 6)])) m[i >> 5] = (m[i >> 5]! | (1 << (i & 31))) >>> 0;
      e.write("dir", L.dir.cmask + CMASK_WORDS * cs, m);
    }
    const recs = await e.read("dir", L.dir.frec, 32 * L.cap.f), occ = new Uint32Array(L.cap.f);
    for (const [k, fs] of s.bricks) {
      const b: V3 = [k & 1023, (k >>> 10) & 1023, (k >>> 20) & 1023];
      let m = 0, j8 = 0;
      for (let j = 0; j < 8; j++) {
        const t: V3 = [2 * b[0] + (j & 1), 2 * b[1] + ((j >> 1) & 1), 2 * b[2] + (j >> 2)];
        if (hOld(t)) m |= 1 << j;
        if (t.every(v => v < T) && fx.joined[ti(t)]) j8 |= 1 << j;
      }
      const r = 32 * fs + REC.MASK; e.write("dir", L.dir.frec + r, new Uint32Array([(recs[r]! & ~0xff) | m]));
      occ[fs] = j8 << 8;
    }
    e.write("set", L.set.tileOcc, occ);
    e.write("set", L.set.ring + RING.DT, new Float32Array([dt])); e.write("set", L.set.ring + RING.H, new Float32Array([h]));
    e.write("set", L.set.ring + RING.GRAVITY, new Float32Array(meta.gravity)); e.write("set", L.set.ring + RING.FLAGS, new Uint32Array([0]));
    // ---- φ (as φ/h) and US at their storage, poison elsewhere.
    const fBuf = new Map<string, Float32Array>(), cBuf = new Map<string, Float32Array>();
    for (const name of ["PHI", "US0", "US1", "US2"]) { const p = name === "PHI" ? POISON_PHI : POISON_U; fBuf.set(name, new Float32Array(BRICK_STRIDE * L.cap.f).fill(p)); cBuf.set(name, new Float32Array(BRICK_STRIDE * L.cap.c).fill(p)); }
    const store = (name: string, kind: "F" | "C", at: V3, value: number) => {
      if (kind === "F") fBuf.get(name)![BRICK_STRIDE * s.bricks.get(key(shr(at, 3)))! + local8(at)] = value;
      else { const T4 = shr(at, 2); cBuf.get(name)![BRICK_STRIDE * s.pages.get(key(shr(T4, 3)))! + local8(T4)] = value; }
    };
    const upload = (names: readonly string[]) => {
      for (const name of names) {
        const pool = name === "PHI" ? "phi" : "vel";
        for (const fs of s.bricks.values()) e.write(pool, L.base(pool, "F", name) + BRICK_STRIDE * fs, fBuf.get(name)!.slice(BRICK_STRIDE * fs, BRICK_STRIDE * fs + 512));
        for (const cs of s.pages.values()) e.write(pool, L.base(pool, "C", name) + BRICK_STRIDE * cs, cBuf.get(name)!.slice(BRICK_STRIDE * cs, BRICK_STRIDE * cs + 512));
      }
    };
    let noPreimage = 0;
    const storePhi = (i: number, x: number) => { const v = preimage(x, h); if (v === undefined) noPreimage++; store("PHI", verts[i]!.kind, verts[i]!.p, v ?? fr(x / h)); };
    fx.phi.forEach((x, i) => storePhi(i, x));
    faces.forEach(({ c, a, kind }, i) => store(`US${a}`, kind, c, fx.US[i]!));
    upload(["PHI", "US0", "US1", "US2"]);
    // ---- localSpeed (uniform-mixed-frame-plan.ts l.269, umSignedFaceExtent l.241) on the old texels: each tile's
    // mean extended face velocity and span (setCensus's DEF words). The old texel of cell q on axis a is the stored
    // lower face of q + e_a (0 outside the lattice, nbFacePlane's); every page is resident, so low-open is the lattice's low face.
    const faceBase = (a: number) => a * (W + 1) * W * W;
    const faceIndex = (c: readonly number[], a: number) => { const d = [W, W, W]; d[a] = W + 1; return faceBase(a) + c[0]! + d[0]! * (c[1]! + d[1]! * c[2]!); };
    const dense = new Float32Array(3 * (W + 1) * W * W).fill(NaN);
    faces.forEach(({ c, a }, i) => { dense[faceIndex(c, a)] = fx.US[i]!; });
    let unstored = 0;
    const texel = (q: readonly number[], a: number) => {
      const c = [...q]; c[a]!++;
      if (!c.every((v, b) => v >= 0 && v < (b === a ? W + 1 : W))) return 0;
      const v = dense[faceIndex(c, a)]!; if (Number.isNaN(v)) unstored++; return v;
    };
    const PLAN_MAX = fr(3.402823e38);
    const orderKey = (x: number) => { const b = bitsOf(x); return ((b & 0x80000000) !== 0 ? ~b : b | 0x80000000) >>> 0; };
    const orderValue = (k: number) => fromBits((k & 0x80000000) !== 0 ? k & 0x7fffffff : ~k);
    const extent = (o: V3, a: number, values: number[]) => {
      let lo = PLAN_MAX, hi = -PLAN_MAX, finite = true;
      if (o[a] === 0) { const q: V3 = [...o]; q[a]!--; values.push(texel(q, a)); }
      for (const v of values) { if (!(Math.abs(v) <= PLAN_MAX)) finite = false; lo = Math.min(lo, v); hi = Math.max(hi, v); }
      return { lo, hi, finite: finite && lo <= hi };
    };
    const planOld: number[][] = [];
    for (let i = 0; i < n; i++) {
      const t = coord(i), mean = [0, 0, 0], hiK = [0, 0, 0], loK = [0, 0, 0];
      const add = (e: { lo: number; hi: number; finite: boolean }, a: number, m: number) => {
        mean[a] = fr(mean[a]! + fr(fr(fr(0.5 * e.lo) + fr(0.5 * e.hi)) / m));
        hiK[a] = Math.max(hiK[a]!, e.finite ? orderKey(e.hi) : 0xffffffff); loK[a] = Math.max(loK[a]!, e.finite ? orderKey(-e.lo) : 0xffffffff);
      };
      if (hOld(t)) for (let lane = 0; lane < 64; lane++) {
        const o: V3 = [4 * t[0] + (lane & 3), 4 * t[1] + ((lane >> 2) & 3), 4 * t[2] + (lane >> 4)];
        for (let a = 0; a < 3; a++) add(extent(o, a, [texel(o, a)]), a, 64);
      } else for (let a = 0; a < 3; a++) {
        const o: V3 = [4 * t[0], 4 * t[1], 4 * t[2]], probe: V3 = [...o]; probe[a]! += 4;
        const anchor: V3 = [...probe]; anchor[a]!--;
        const fine = probe[a]! < W && hOld(shr(probe, 2)), values: number[] = [];
        for (let part = 0; part < (fine ? 16 : 1); part++) { const q: V3 = [...anchor]; q[(a + 1) % 3]! += part % 4; q[(a + 2) % 3]! += Math.floor(part / 4); values.push(texel(q, a)); }
        add(extent(o, a, values), a, 1);
      }
      let span = 0;
      for (let a = 0; a < 3; a++) span = hiK[a] === 0xffffffff || loK[a] === 0xffffffff ? PLAN_MAX : fr(span + Math.max(0, fr(orderValue(hiK[a]!) + orderValue(loK[a]!))));
      planOld.push([...mean, span]);
    }
    assert.equal(unstored, 0, "localSpeed reads stored texels only");
    // The old engine's split vertices (4-aligned h-closure vertices whose base disagrees with the h
    // store): the tiles within a tile of one may read either value.
    const splitTiles = new Set<number>();
    for (const vi of fx.split) { const p = verts[vi]!.p; for (let z = -2; z <= 1; z++) for (let y = -2; y <= 1; y++) for (let x = -2; x <= 1; x++) { const t: V3 = [(p[0] >> 2) + x, (p[1] >> 2) + y, (p[2] >> 2) + z]; if (t.every(v => v >= 0 && v < T)) splitTiles.add(ti(t)); } }
    // ---- Run setCensus: the plain φ (the h store's value at every h-closure vertex), then the
    // canonical one at the split vertices (the base's, as umLoadCorner reads it).
    const runCensus = async () => {
      await e.run(brickCensusSchedule(L, k => ({ indirect: ARGS.PER_LIST * k + ARGS.G1 })));
      const set = await e.read("set"), census = await e.read("census");
      assert.equal(set[H.FAIL], 0, `FAIL ${set[H.FAIL]} info ${set[H.FAIL_INFO]?.toString(16)}`);
      return { set, census };
    };
    const variants = new Map<string, Awaited<ReturnType<typeof runCensus>>>();
    variants.set("plain", await runCensus());
    if (fx.split.length) { fx.split.forEach((vi, j) => storePhi(vi, fx.splitPhi[j]!)); upload(["PHI"]); variants.set("canonical", await runCensus()); }
    const results = [...variants].map(([variant, { set, census }]) => {

    // ---- Compare, tile by tile.
    type Tally = { compared: number; equal: number; deep: number; nonDeep: number; split: number; examples: string[]; hard: number[] };
    const tallies = new Map<string, Tally>();
    const tally = (k: string) => { let t = tallies.get(k); if (!t) tallies.set(k, t = { compared: 0, equal: 0, deep: 0, nonDeep: 0, split: 0, examples: [], hard: [] }); return t; };
    const check = (field: string, i: number, ok: boolean, what: () => string) => {
      const t = tally(field); t.compared++;
      if (ok) { t.equal++; return; }
      const at = coord(i), deep = fx.deep[i] === 1;
      if (deep) t.deep++; else { t.nonDeep++; t.hard.push(i); }
      if (splitTiles.has(i)) t.split++;
      if (t.examples.length < 16 && !deep) t.examples.push(`tile ${at} ${fx.owner[i] === 1 ? "h" : "4h"}${fx.joined[i] ? " joined" : ""}${splitTiles.has(i) ? " split-near" : ""} deep-distance ${deepDistance(at)}: ${what()}`);
    };
    const hex = (x: number) => (x >>> 0).toString(16);
    let wantNew = 0, wantOld = 0, deepBandNew = 0;
    // TRIG is the old need rule (nbActivityTriggered) over the old FIN's criteria, its thin term reading the
    // deformation (tileErrorOf) of the plan words (DEF), which are checked here against localSpeed transcribed on the
    // old engine's own extension texels (planOld). The old engine's own deformation was its last frame's certificate,
    // which the fixture does not hold: so this pins the rule on the frame's state, and counts the thin-alone tiles it heats.
    const planAt = (t: V3) => planOld[ti(t)]!;
    const deformation = (t: V3) => {
      const own = planAt(t); let gradient = 0;
      for (let a = 0; a < 3; a++) {
        const lo: V3 = [...t], hi: V3 = [...t]; lo[a]!--; hi[a]!++;
        if (![lo, hi].every(q => q.every(v => v >= 0 && v < T))) continue;
        const below = planAt(lo), above = planAt(hi);
        let v = 0; for (let c = 0; c < 3; c++) v = Math.max(v, Math.abs(fr(above[c]! - own[c]!)), Math.abs(fr(own[c]! - below[c]!)));
        gradient = fr(gradient + fr(v / fr(4 * h)));
      }
      const d = Math.max(fr(fr(4 * h) * gradient), own[3]!);
      return d < fr(3e38) ? d : fr(3.402823e38);
    };
    const need = (t: V3, criteria: number) => (criteria & 0x3c) !== 0 || (criteria & 1) !== 0 || ((criteria & 2) !== 0 && fr(fr(dt * deformation(t)) / h) > fr(0.02));
    const thinAlone = { tiles: 0, heated: 0 };
    const deepDecided: string[] = [];
    for (let i = 0; i < n; i++) {
      const t = coord(i), cs = s.pages.get(key(shr(t, 3)))!, l = local8(t), o = CENSUS.STRIDE * cs;
      const keys = [...census.subarray(o + CENSUS.KEYS + 6 * l, o + CENSUS.KEYS + 6 * l + 6)], oldKeys = [...fx.keys.subarray(6 * i, 6 * i + 6)];
      check("keys", i, keys.every((k, q) => k === oldKeys[q]), () => `${keys.map(hex)} vs ${oldKeys.map(hex)}`);
      const w0 = census[o + CENSUS.W0 + l]!, fin = census[o + CENSUS.FIN + l]!, ow0 = fx.w0[i]!, ow1 = fx.w1[i]!;
      check("w0 bytes", i, byteNear(w0, ow0, 0) && byteNear(w0, ow0, 8), () => `${hex(w0)} vs ${hex(ow0)}`);
      check("fin flags", i, ((fin & FLAGS) >>> 0) === ((ow1 & FLAGS) >>> 0), () => `${hex(fin)} vs ${hex(ow1)}`);
      check("fin bytes", i, byteNear(fin, ow1, 0) && byteNear(fin, ow1, 8), () => `${hex(fin)} vs ${hex(ow1)}`);
      const cr = [census[o + CENSUS.CROSS + 2 * l]!, census[o + CENSUS.CROSS + 2 * l + 1]!];
      check("crossing cells", i, cr[0] === fx.cross[2 * i] && cr[1] === fx.cross[2 * i + 1], () => `${cr.map(hex)} vs ${[fx.cross[2 * i]!, fx.cross[2 * i + 1]!].map(hex)}`);
      check("gap", i, census[o + CENSUS.GAP + l] === fx.gap[i], () => `${hex(census[o + CENSUS.GAP + l]!)} vs ${hex(fx.gap[i]!)}`);
      const want = (set[L.set.want + 16 * cs + (l >> 5)]! >>> (l & 31)) & 1;
      wantNew += want; wantOld += fx.decided[i]!;
      check("WANT", i, want === fx.decided[i], () => `${want} vs ${fx.decided[i]} (${fx.owner[i] === 1 ? "h" : "4h"} now; fin ${hex(fin)} vs ${hex(ow1)})`);
      const fs = s.bricks.get(key(shr(t, 1)))!, j = (t[0] & 1) | ((t[1] & 1) << 1) | ((t[2] & 1) << 2);
      const criteria = (ow1 >>> 16) & 63, trig = (set[L.set.trig + fs]! >>> j) & 1, oldTrig = need(t, criteria) ? 1 : 0;
      if (criteria === 2) { thinAlone.tiles++; thinAlone.heated += oldTrig; }
      check("TRIG", i, trig === oldTrig, () => `${trig} vs ${oldTrig} (criteria ${criteria})`);
      check("TRIG thin alone", i, ((set[L.set.trig + fs]! >>> (8 + j)) & 1) === ((ow1 >>> 17) & 1), () => `thin bit`);
      const gp = [0, 1, 2, 3].map(k => fromBits(census[o + CENSUS.DEF + 4 * l + k]!)), xp = planOld[i]!;
      check("plan", i, gp.every((g, k) => (k === 3 ? g === xp[k] : Math.abs(g - xp[k]!) <= 1e-6 * (1 + Math.abs(xp[k]!)))), () => `${gp} vs ${xp}`);
      check("band = decided (old)", i, fx.band[i] === fx.decided[i], () => `band ${fx.band[i]} decided ${fx.decided[i]}`);
      if (fx.deep[i]) { if (want) { deepBandNew++; if (deepDecided.length < 8) deepDecided.push(`${t}`); } }
    }
    // Counters against the old header (census words 0, 12, 2, 4, 18).
    const hdr = meta.header, cH = (w: number) => set[w]!;
    const counters = {
      crossing: [cH(CENSUS_H.CENSUS_CROSSING), hdr[0]], required: [cH(CENSUS_H.CENSUS_REQUIRED), hdr[12]], band: [cH(CENSUS_H.CENSUS_BAND), hdr[2]],
      refined: [cH(CENSUS_H.CENSUS_REFINED), hdr[4]], reach: [fromBits(cH(CENSUS_H.CENSUS_REACH)), fromBits(meta.reachBits)], clamped: [cH(CENSUS_H.CENSUS_CLAMPED), 0],
    };
    return { variant, tallies, counters, deepBandNew, deepDecided, wantNew, wantOld, thinAlone };
    });
    const deepTiles = deepList.length, deepFromPrev = deepList.filter(t => !fx.joined[ti(t)]).length, deepPrevBand = deepList.filter(t => !fx.joined[ti(t)] && fx.decidedPrev[ti(t)]).length;
    // A tile that differs under both φ choices and holds no split vertex within a tile: a mismatch of the port.
    const hard = new Map<string, number[]>();
    for (const [k] of results[0]!.tallies) hard.set(k, results[0]!.tallies.get(k)!.hard.filter(i => results.every(r => r.tallies.get(k)!.hard.includes(i)) && !splitTiles.has(i)));
    const report = {
      step, window: W, tiles: n, excluded: 0, phiWithoutPreimage: noPreimage, splitVertices: fx.split.length, splitTiles: splitTiles.size,
      deepCold: { tiles: deepTiles, regions: deepTiles / 64, decidedFromPrev: deepFromPrev, prevBand: deepPrevBand, bandAnew: results.map(r => r.deepBandNew), bandAnewTiles: results[0]!.deepDecided },
      hard: Object.fromEntries([...hard].filter(([, v]) => v.length).map(([k, v]) => [k, v.map(i => coord(i).join(","))])),
      variants: results.map(({ variant, tallies, counters, wantNew, wantOld, thinAlone }) => ({ variant, want: { new: wantNew, old: wantOld }, thinAlone, counters,
        fields: Object.fromEntries([...tallies].map(([k, t]) => [k, { compared: t.compared, equal: t.equal, deep: t.deep, nonDeep: t.nonDeep, splitNear: t.split, examples: t.examples.length ? t.examples : undefined }])) })),
    };
    // ---- Replay (regionDeep, the real D's head): the old engine's replayed regions (mode 1) go in deep COLD with
    // their last classification's stamp and records, and replay. Tiers: COLD on the deep regions and their
    // neighbours (an old deep region's neighbours were COLD, and a region the old engine held not deep had a
    // neighbour that was not, so the deep set is the old one), HOT elsewhere. Records: W0, KEYS, CROSS and GAP are the
    // old engine's (its replay published them; CROSS and GAP after its importance, which a replay's importance edits
    // to the same words); W1 before the importance, NEAR and DEF are not in the fixture and are this census's
    // plain classification. DEC is the old decidedPrev. Every region's mode must be the old one and WANT the old WANT.
    const S = CENSUS.STRIDE, R4 = W / 16, pageKeys = new Map([...s.pages].map(([k, cs]) => [cs, k]));
    const regionOf = (t: V3) => shr(t, 2), rKey = (R: V3) => R[0] + R4 * (R[1] + R4 * R[2]);
    const deepR = new Set<number>(), modeR = new Set<number>();
    for (let i = 0; i < n; i++) { const R = regionOf(coord(i)); if (fx.deep[i]) deepR.add(rKey(R)); if (fx.mode[i] === 1) modeR.add(rKey(R)); }
    const coldR = (R: V3) => { for (let k = 0; k < 27; k++) { const q: V3 = [R[0] + (k % 3) - 1, R[1] + (Math.floor(k / 3) % 3) - 1, R[2] + Math.floor(k / 9) - 1]; if (q.every(v => v >= 0 && v < R4) && deepR.has(rKey(q))) return true; } return false; };
    const cens = new Uint32Array(variants.get("plain")!.census);
    for (const [cs, page] of pageOf) for (let r = 0; r < 8; r++) {
      const at = S * cs + CENSUS.REG + RG.WORDS * r, R: V3 = [2 * page[0] + (r & 1), 2 * page[1] + ((r >> 1) & 1), 2 * page[2] + (r >> 2)];
      cens.fill(0, at, at + RG.WORDS);
      if (!R.every(v => v < R4)) continue;
      cens[at + RG.TIER] = coldR(R) ? 0 : 2;
      cens[at + RG.STAMP] = modeR.has(rKey(R)) ? 1 : 0; cens[at + RG.GEN] = pageKeys.get(cs)! + 1;
      for (let l = 0; l < 64; l++) {
        const t: V3 = [8 * page[0] + (l & 3) + 4 * (r & 1), 8 * page[1] + ((l >> 2) & 3) + 4 * ((r >> 1) & 1), 8 * page[2] + (l >> 4) + 4 * (r >> 2)];
        if (hOld(t)) cens[at + RG.OWN + (l >> 5)] = (cens[at + RG.OWN + (l >> 5)]! | (1 << (l & 31))) >>> 0;
      }
      let pages = 0;
      for (let k = 0; k < 27; k++) { const q: V3 = [page[0] + (k % 3) - 1, page[1] + (Math.floor(k / 3) % 3) - 1, page[2] + Math.floor(k / 9) - 1]; if (q.every(v => v >= 0 && 32 * v < W) && s.pages.has(key(q))) pages |= 1 << k; }
      cens[at + RG.PAGES] = pages >>> 0;
    }
    for (let i = 0; i < n; i++) {
      const t = coord(i), cs = s.pages.get(key(shr(t, 3)))!, l = local8(t), o = S * cs;
      if (fx.decidedPrev[i]) cens[o + CENSUS.DEC + (l >> 5)] = (cens[o + CENSUS.DEC + (l >> 5)]! | (1 << (l & 31))) >>> 0;
      if (fx.mode[i] !== 1) continue;
      cens[o + CENSUS.W0 + l] = fx.w0[i]!;
      for (let k = 0; k < 6; k++) cens[o + CENSUS.KEYS + 6 * l + k] = fx.keys[6 * i + k]!;
      cens[o + CENSUS.RCROSS + 2 * l] = fx.cross[2 * i]!; cens[o + CENSUS.RCROSS + 2 * l + 1] = fx.cross[2 * i + 1]!; cens[o + CENSUS.RGAP + l] = fx.gap[i]!;
    }
    for (const cs of s.pages.values()) cens.fill(0, S * cs + CENSUS.STAT, S * cs + CENSUS.STAT + 16);
    e.write("census", 0, cens);
    await e.run(brickCensusSchedule(L, k => ({ indirect: ARGS.PER_LIST * k + ARGS.G1 }), LIST.C_ALL, LIST.F_ALL, true, true, true));
    const rSet = await e.read("set"), rCensus = await e.read("census");
    assert.equal(rSet[H.FAIL], 0, `replay FAIL ${rSet[H.FAIL]} info ${rSet[H.FAIL_INFO]?.toString(16)}`);
    let modeWrong = 0, replayedRegions = 0, wantWrong = 0, deepWant = 0;
    const wantMiss: string[] = [];
    for (const [cs, page] of pageOf) for (let r = 0; r < 8; r++) {
      const R: V3 = [2 * page[0] + (r & 1), 2 * page[1] + ((r >> 1) & 1), 2 * page[2] + (r >> 2)];
      if (!R.every(v => v < R4)) continue;
      const m = rCensus[S * cs + CENSUS.REG + RG.WORDS * r + RG.MODE]!;
      if (m === 1) replayedRegions++;
      if ((m === 1) !== modeR.has(rKey(R))) modeWrong++;
    }
    for (let i = 0; i < n; i++) {
      const t = coord(i), cs = s.pages.get(key(shr(t, 3)))!, l = local8(t);
      const want = (rSet[L.set.want + 16 * cs + (l >> 5)]! >>> (l & 31)) & 1;
      if (want !== fx.decided[i]) { wantWrong++; if (wantMiss.length < 16) wantMiss.push(`${t}${fx.deep[i] ? " deep" : ""}${splitTiles.has(i) ? " split-near" : ""}: ${want} vs ${fx.decided[i]}`); }
      else if (fx.mode[i] === 1) deepWant++;
    }
    const replay = { oldReplayedRegions: modeR.size, oldReplayedTiles: 64 * modeR.size, replayedRegions, modeWrong, replayedTilesWantEqual: deepWant, wantWrong, wantMiss };
    console.log(JSON.stringify({ ...report, replay }, null, 1));
    e.destroy();
    assert.equal(modeWrong, 0, "every region replays as the old engine's did");
    assert.equal(wantWrong, 0, `WANT with replay is the old WANT: ${wantMiss.join("; ")}`);
    // ---- Verdicts: on the tiles that are not deep COLD, every field exact under one φ choice or the other (or near a split vertex), and the reach.
    for (const [k, v] of hard) assert.equal(v.length, 0, `${k}: ${v.length} non-deep tiles differ under both φ choices, away from a split vertex: ${v.slice(0, 16).map(i => coord(i).join(",")).join("; ")}`);
    for (const r of results) assert.equal(r.counters.reach[0], r.counters.reach[1], `the reach (${r.variant})`);
  });
});
