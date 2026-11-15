// setCensus's C passes on eight workgroups a page (X8) against the frozen G1 passes (parity/census-g1-frozen.ts, the
// census at 5233e246: a workgroup a page, each lane 8 tiles in turn). Both schedules run on the same state, from
// the same set buffer and a zeroed census buffer, and every word must be identical: the whole census buffer, and
// the whole set buffer (the census words, WANT, TRIG, MARK_F, the counters, FAIL), except the candidate list, whose
// order the appending lanes' race sets (compared as a sorted list, so a duplicate counts), and X8's own candidate
// mask (CAND_MASK) and censusImportance's moving-shape bits (census MOV), which the G1 passes do not have. Since
// the need trigger (TRIG bits 0..7 and CENSUS_TRIGGERED, and the census's plan words DEF) the frozen passes are the
// older rule there, so those words are left to brick-census.test.ts's oracle. Two states: brick-census.test.ts's synthetic scene (h and
// 4h tiles, the clamp and its candidates) and the old engine's dam census at frames 14 and 43 (every page
// resident). Each runs tiered at the boot, tiered at a step's head (the ring keep marks) and all-h. GPU only.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import test from "node:test";
import { gunzipSync } from "node:zlib";
import { ARGS, BRICK_STRIDE, CAND_MASK_WORDS, CMASK_WORDS, H, LIST, REC, RING, type BrickLayout } from "./brick-abi";
import { CENSUS, brickCensusSchedule } from "./brick-census.wgsl";
import type { BrickEngine } from "./brick-engine";
import { CENSUS as FROZEN, CENSUS_H, brickCensusSchedule as frozenCensusSchedule } from "./parity/census-g1-frozen";
import { decodeCensusFixture, type CensusFixtureMeta } from "./parity/census-fixture";
import { ownerOf, storedFaces, storedVertices } from "./parity/remap-fixture";

type V3 = [number, number, number];
const GPU = { skip: !process.env.WEBGPU_NODE_MODULE };
const PARITY = resolve(import.meta.dirname, "parity");
const fr = Math.fround;
const key = (b: readonly number[]) => ((b[0]! & 1023) | ((b[1]! & 1023) << 10) | ((b[2]! & 1023) << 20)) >>> 0;
const local8 = (c: readonly number[]) => (c[0]! & 7) | ((c[1]! & 7) << 3) | ((c[2]! & 7) << 6);
const shr = (c: readonly number[], s: number): V3 => [c[0]! >> s, c[1]! >> s, c[2]! >> s];
const add = (a: V3, b: V3): V3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const scale3 = (a: V3, s: number): V3 => [a[0] * s, a[1] * s, a[2] * s];
const g1 = (k: number) => ({ indirect: ARGS.PER_LIST * k + ARGS.G1 });

/** Runs the frozen G1 census and the X8 census from the same state; returns the differing words (at most 20) and the coverage. */
async function compareSchedules(e: BrickEngine, L: BrickLayout, tiered: boolean, keep: boolean) {
  const set0 = await e.read("set"), zero = new Uint32Array((await e.read("census")).length);
  const runOne = async (placements: Parameters<BrickEngine["run"]>[0]) => {
    e.write("set", 0, set0.slice()); e.write("census", 0, zero);
    await e.run(placements);
    return { set: await e.read("set"), census: await e.read("census") };
  };
  const old = await runOne(frozenCensusSchedule(L, g1, LIST.C_ALL, LIST.F_ALL, tiered, keep));
  const neu = await runOne(brickCensusSchedule(L, g1, LIST.C_ALL, LIST.F_ALL, tiered, keep));
  const diff: string[] = [];
  const note = (s: string) => { if (diff.length < 20) diff.push(s); };
  // The frozen census's words at its own stride; the live census adds the plan words (DEF) after them.
  for (let cs = 0; cs < L.cap.c; cs++) for (let k = 0; k < FROZEN.STRIDE; k++) {
    if (k >= CENSUS.MOV && k < CENSUS.MOV + 16) continue;
    const a = old.census[FROZEN.STRIDE * cs + k]!, b = neu.census[CENSUS.STRIDE * cs + k]!;
    if (a !== b) note(`census slot ${cs} +${k}: ${a.toString(16)} vs ${b.toString(16)}`);
  }
  const c0 = set0[H.N_CAND]!, cOld = old.set[H.N_CAND]!, cNew = neu.set[H.N_CAND]!;
  const candEnd = L.set.cand + L.cap.f, maskAt = L.set.candMask, maskEnd = maskAt + CAND_MASK_WORDS * L.cap.c;
  // TRIG bits 0..7 (and their count) are the need rule since the frozen passes (brick-census.test.ts checks them); bits 8..15 are compared.
  const trigAt = L.set.trig, trigEnd = trigAt + L.cap.f;
  for (let w = 0; w < old.set.length; w++) {
    if ((w >= L.set.cand + c0 && w < candEnd) || (w >= maskAt && w < maskEnd) || w === CENSUS_H.CENSUS_TRIGGERED) continue;
    const m = w >= trigAt && w < trigEnd ? 0xffffff00 : 0xffffffff;
    if (((old.set[w]! & m) >>> 0) !== ((neu.set[w]! & m) >>> 0)) note(`set word ${w}: ${old.set[w]!.toString(16)} vs ${neu.set[w]!.toString(16)}`);
  }
  const list = (s: Uint32Array, n: number) => [...s.subarray(L.set.cand + c0, L.set.cand + Math.min(n, L.cap.f))].sort((a, b) => a - b);
  assert.deepEqual(list(neu.set, cNew), list(old.set, cOld), "the candidate list (sorted)");
  const want = old.set.subarray(L.set.want, L.set.want + CMASK_WORDS * L.cap.c).reduce((n, x) => n + popcount(x), 0);
  const trig = old.set.subarray(L.set.trig, L.set.trig + L.cap.f).reduce((n, x) => n + popcount(x & 0xff), 0);
  const marks = old.set.subarray(L.set.markF, L.set.markF + L.cap.f).reduce((n, x) => n + (x !== 0 ? 1 : 0), 0);
  // 4h crossing tiles with a depth score (censusCoarse's depth jobs) and tiles whose MOV bit is set (movingShapeNear's).
  const cmask = await e.read("dir", L.dir.cmask, CMASK_WORDS * L.cap.c);
  let deep4h = 0, moving = 0;
  for (let cs = 0; cs < L.cap.c; cs++) for (let i = 0; i < 512; i++) {
    const o = CENSUS.STRIDE * cs, h = (cmask[CMASK_WORDS * cs + (i >> 5)]! >>> (i & 31)) & 1;
    if (!h && (old.census[o + CENSUS.FIN + i]! & (1 << 28)) && ((old.census[o + CENSUS.W0 + i]! >>> 8) & 255)) deep4h++;
    if ((neu.census[o + CENSUS.MOV + (i >> 5)]! >>> (i & 31)) & 1) moving++;
  }
  return { diff, coverage: { tiered, keep, fail: old.set[H.FAIL]!, want, trig, marks, candidates: cOld - c0, deep4h, moving } };
}
const popcount = (x: number) => { let n = 0; for (let v = x >>> 0; v; v &= v - 1) n++; return n; };

async function checkAll(e: BrickEngine, name: string) {
  const L = e.layout, out = [];
  for (const [tiered, keep] of [[true, false], [true, true], [false, false]] as const) {
    const { diff, coverage } = await compareSchedules(e, L, tiered, keep);
    assert.deepEqual(diff, [], `${name} tiered ${tiered} keep ${keep}: ${diff.length} differing words`);
    out.push(coverage);
  }
  console.log(`setCensus X8 = G1 (${name}):`, JSON.stringify(out));
  return out;
}

// ---- brick-census.test.ts's frozen scene.
const N: V3 = [64, 32, 64];
const H_M = 1 / 32, DT = 1 / 60;
const phiOf = (p: V3) => {
  const pool = p[1] - 9.3 + 0.9 * Math.sin(0.31 * p[0] + 0.2 * p[2]) + 0.4 * Math.cos(0.23 * p[2]);
  const drop = Math.hypot(p[0] - 14.5, p[1] - 17.2, p[2] - 13.8) - 3.4;
  return Math.max(Math.min(pool, drop), -8);
};
const velOf = (c: V3, a: number) => {
  const inDrop = Math.hypot(c[0] - 14.5, c[1] - 17.2, c[2] - 13.8) < 7;
  if (a === 0) return -1.6 * Math.exp(-c[0] / 9) + 0.15 * Math.sin(0.2 * c[2]);
  if (a === 1) return (inDrop ? -2.8 : 0) + 0.25 * Math.sin(0.3 * c[0]) - 0.1 * Math.cos(0.17 * c[2]);
  return 0.35 * Math.cos(0.25 * c[0]) + 0.05 * c[1] / 8;
};
const hTile = (t: V3) => t[0] <= 5 && t[2] <= 5 && !(t[0] === 2 && t[1] === 2 && t[2] === 2);

test("setCensus X8 (GPU): bit-exact with the G1 passes on the synthetic census scene", GPU, async () => {
  const { withUniformDevice } = await import("../../../tests/helpers/uniform-geometric");
  const { BrickEngine } = await import("./brick-engine");
  const { bringUpCG } = await import("./brick-bringup-cg");
  const { bringUpStages } = await import("./brick-stages");
  const { snapshot } = await import("./brick-readback");
  await withUniformDevice("brick census x8", async device => {
    const e = await BrickEngine.create(device, {
      lattice: N, h_m: H_M, dt_s: DT, gravity_mps2: [0, -9.81, 0],
      capacities: { f: 700, c: 16, b: 0, k: 0, particles: 1024 }, liquid: [{ lo: [-1, -1, -1], hi: [40, 12, 22] }], solver: L => bringUpCG(L),
      stages: bringUpStages,
    });
    const L = e.layout, snap = await snapshot(e);
    const resident = (b: V3) => snap.bricks.has(key(b));
    for (const [k, cs] of snap.pages) {
      const pk: V3 = [k & 1023, (k >>> 10) & 1023, (k >>> 20) & 1023], m = new Uint32Array(CMASK_WORDS);
      for (let i = 0; i < 512; i++) { const t = add(scale3(pk, 8), [i & 7, (i >> 3) & 7, i >> 6]); if (hTile(t) && 4 * t[1] < N[1] && resident(shr(t, 1))) m[i >> 5] = (m[i >> 5]! | (1 << (i & 31))) >>> 0; }
      e.write("dir", L.dir.cmask + CMASK_WORDS * cs, m);
      const cw = (fn: (T: V3) => number) => { const w = new Float32Array(512); for (let i = 0; i < 512; i++) w[i] = fn(add(scale3(pk, 8), [i & 7, (i >> 3) & 7, i >> 6])); return w; };
      e.write("phi", L.base("phi", "C", "PHI") + BRICK_STRIDE * cs, cw(T => phiOf(scale3(T, 4)) + 0.15 * Math.sin(T[0] + 2 * T[2])));
      for (let a = 0; a < 3; a++) e.write("vel", L.base("vel", "C", `US${a}`) + BRICK_STRIDE * cs, cw(T => velOf(scale3(T, 4), a) * 0.9));
    }
    const occ = new Uint32Array(L.cap.f);
    for (const [kb, fs] of snap.bricks) {
      const b: V3 = [kb & 1023, (kb >>> 10) & 1023, (kb >>> 20) & 1023];
      const fw = (fn: (c: V3) => number) => { const w = new Float32Array(512); for (let l = 0; l < 512; l++) w[l] = fn(add(scale3(b, 8), [l & 7, (l >> 3) & 7, l >> 6])); return w; };
      e.write("phi", L.base("phi", "F", "PHI") + BRICK_STRIDE * fs, fw(phiOf));
      for (let a = 0; a < 3; a++) e.write("vel", L.base("vel", "F", `US${a}`) + BRICK_STRIDE * fs, fw(c => velOf(c, a)));
      occ[fs] = 0xff | (b[0] === 9 && b[1] === 2 ? 0x8000 : 0);
    }
    e.write("set", L.set.tileOcc, occ);
    e.write("set", L.set.ring + RING.DT, new Float32Array([DT])); e.write("set", L.set.ring + RING.H, new Float32Array([H_M]));
    e.write("set", L.set.ring + RING.GRAVITY, new Float32Array([0, -9.81, 0])); e.write("set", L.set.ring + RING.FLAGS, new Uint32Array([0]));
    const cov = await checkAll(e, "synthetic");
    assert.ok(cov[0]!.candidates > 0 && cov[0]!.want > 0 && cov[0]!.trig > 0 && cov[1]!.marks > 0 && cov.every(c => c.fail === 0), JSON.stringify(cov));
    e.destroy();
  });
});

for (const step of [14, 43]) test(`setCensus X8 (GPU): bit-exact with the G1 passes on the old engine's dam census, frame ${step}`, GPU, async () => {
  const meta = JSON.parse(readFileSync(resolve(PARITY, `dam-census-${step}.json`), "utf8")) as CensusFixtureMeta;
  const fx = decodeCensusFixture(gunzipSync(readFileSync(resolve(PARITY, `dam-census-${step}.bin.gz`))), meta);
  const W = meta.window, T = W / 4, h = fr(meta.h);
  const hOld = ownerOf(fx.owner, T), verts = storedVertices(hOld, W), faces = storedFaces(hOld, W);
  const ti = (t: readonly number[]) => t[0]! + T * (t[1]! + T * t[2]!);
  const { withUniformDevice } = await import("../../../tests/helpers/uniform-geometric");
  const { BrickEngine } = await import("./brick-engine");
  const { bringUpCG } = await import("./brick-bringup-cg");
  const { bringUpStages } = await import("./brick-stages");
  const { snapshot } = await import("./brick-readback");
  await withUniformDevice("brick census x8 dam", async device => {
    const nb = W / 8 + 1, np = Math.ceil((W + 1) / 32);
    const e = await BrickEngine.create(device, { lattice: [W, W, W], h_m: meta.h, dt_s: meta.dt, gravity_mps2: meta.gravity as V3,
      capacities: { f: nb ** 3, c: np ** 3, b: 0, k: 0, particles: 1024 }, liquid: [{ lo: [-1, -1, -1], hi: [W + 1, W + 1, W + 1] }], solver: L => bringUpCG(L), stages: bringUpStages });
    const s = await snapshot(e), L = e.layout;
    for (const [k, cs] of s.pages) {
      const page: V3 = [k & 1023, (k >>> 10) & 1023, (k >>> 20) & 1023], m = new Uint32Array(CMASK_WORDS);
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
    e.write("set", L.set.ring + RING.DT, new Float32Array([fr(meta.dt)])); e.write("set", L.set.ring + RING.H, new Float32Array([h]));
    e.write("set", L.set.ring + RING.GRAVITY, new Float32Array(meta.gravity)); e.write("set", L.set.ring + RING.FLAGS, new Uint32Array([0]));
    // φ as φ/h and US at their storage; the words that hold no state keep the boot's.
    const fBuf = new Map<string, Float32Array>(), cBuf = new Map<string, Float32Array>();
    for (const name of ["PHI", "US0", "US1", "US2"]) {
      const pool = name === "PHI" ? "phi" : "vel";
      fBuf.set(name, new Float32Array((await e.read(pool, L.base(pool, "F", name), BRICK_STRIDE * L.cap.f)).buffer));
      cBuf.set(name, new Float32Array((await e.read(pool, L.base(pool, "C", name), BRICK_STRIDE * L.cap.c)).buffer));
    }
    const store = (name: string, kind: "F" | "C", at: V3, value: number) => {
      if (kind === "F") fBuf.get(name)![BRICK_STRIDE * s.bricks.get(key(shr(at, 3)))! + local8(at)] = value;
      else { const T4 = shr(at, 2); cBuf.get(name)![BRICK_STRIDE * s.pages.get(key(shr(T4, 3)))! + local8(T4)] = value; }
    };
    fx.phi.forEach((x, i) => store("PHI", verts[i]!.kind, verts[i]!.p, fr(x / h)));
    faces.forEach(({ c, a, kind }, i) => store(`US${a}`, kind, c, fx.US[i]!));
    for (const name of ["PHI", "US0", "US1", "US2"]) {
      const pool = name === "PHI" ? "phi" : "vel";
      e.write(pool, L.base(pool, "F", name), fBuf.get(name)! as Float32Array<ArrayBuffer>); e.write(pool, L.base(pool, "C", name), cBuf.get(name)! as Float32Array<ArrayBuffer>);
    }
    const cov = await checkAll(e, `dam frame ${step}`);
    assert.ok(cov[0]!.want > 1000 && cov[0]!.trig > 0 && cov.every(c => c.fail === 0), JSON.stringify(cov));
    e.destroy();
  });
});
