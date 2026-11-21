// Lane E's T4 kernels (levelset-coarse.wgsl.ts) on a tiered boot: each kernel against a Float64 CPU oracle that
// reads the GPU's state before it (the chain lcClear, lcAdvect, lcCouple, lcClear, lcRetire, lcPublish, lcAdvect: the
// advects read the far-air gate the clear before them wrote). The state is the column boot's tiers and residency with
// an analytic φ on every owned vertex, a constant u* on every stored face and hashed lane F tile, cell and vertex
// words; two pages are COLD. Every non-owned φ copy (F copies of C or hanging vertices, C copies no 4h tile owns)
// holds a NaN payload: no kernel may read it into a result or write it. After each kernel the whole φ pool must equal
// the state before it except the C vertices the kernel owns, the set except lcClear's gate words (by the oracle), and
// at the end the velocity pool, fcell and dir must be untouched. A BULK16 page word then latches lcAdvect's
// FAIL.UNSUPPORTED. The second fixture commits h tiles inside 4h pages (brick-set-tiers test 1's block A), so some C
// positions are F's, and runs the same chain.
import assert from "node:assert/strict";
import test from "node:test";
import { ARGS, BRICK_STRIDE, BrickLayout, CMASK_WORDS, brickKey, F_CLASS, FAIL, H, RING, LIST, NONE, REC, SEAM, brickKeyCoord, cellLocal, phiFar, PHI_AIR_C } from "./brick-abi";
import { bringUpCG } from "./brick-bringup-cg";
import { BrickEngine } from "./brick-engine";
import { snapshot } from "./brick-readback";
import { bringUpStages } from "./brick-stages";
import { brickTransferKernels } from "./brick-transfer.wgsl";
import { FT, FTF, fluidLayout } from "./fluid-grid.wgsl";
import { LC_ADVECT_GROUPS, LC_GROUPS, levelSetCoarseKernels } from "./levelset-coarse.wgsl";
import { velocityKernels } from "./velocity-kernels.wgsl";

const GPU = { skip: !process.env.WEBGPU_NODE_MODULE };
type V = [number, number, number];
const N = 64, NL: V = [N, N, N], FAR = phiFar(NL), f32 = Math.fround;
const POISON = 0x7fc0beef;

test("levelset-coarse: every kernel's module defines each function once and takes the tier owners", () => {
  const L = new BrickLayout({ lattice: NL, f: 600, c: 40, b: 0, k: 0, particles: 8192 });
  assert.throws(() => levelSetCoarseKernels(L, fluidLayout(L)), /tiered/);
  const kernels = levelSetCoarseKernels(L);
  for (const [name, k] of Object.entries(kernels)) {
    const fns = [...k.code.matchAll(/\bfn\s+(\w+)\s*\(/g)].map(m => m[1]!);
    const twice = fns.filter((f, i) => fns.indexOf(f) !== i);
    assert.deepEqual(twice, [], `${name}: functions defined twice`);
    assert.ok(fns.includes(name) && k.code.includes("const FG_TIERED:bool=true"), `${name}: entry and FG_TIERED`);
    assert.ok(k.code.includes("NB_LIST_C_ALL"), `${name}: dispatched over C_ALL`);
  }
  assert.ok(kernels.lcAdvect.code.includes("fn fgSampleVel("), "the advect samples lane F's velocity");
});

// ---- deterministic hashing
const hash = (...k: number[]) => { let h = 0x9e3779b9; for (const x of k) { h = Math.imul(h ^ (x | 0), 0x85ebca6b); h ^= h >>> 13; h = Math.imul(h, 0xc2b2ae35); h ^= h >>> 16; } return (h >>> 0) / 4294967296; };
const fbits = (x: number) => { const b = new Float32Array(1); b[0] = x; return new Uint32Array(b.buffer)[0]!; };
const bitsf = (u: number) => { const b = new Uint32Array(1); b[0] = u; return new Float32Array(b.buffer)[0]!; };

// ---- the analytic fields (cells, h units) and u* (m/s)
const phi0 = (v: V) => 0.45 * (v[1] - 21.3) + 2.2 * Math.sin(0.19 * v[0] + 0.4) * Math.cos(0.23 * v[2]) + 0.05 * (v[0] - 30);
const phi1 = (v: V) => phi0(v) + 0.7 * Math.sin(0.31 * v[0] + 0.17 * v[1] + 0.9);
const U: V = [0.37, -1.13, 0.23];

// ---- small vector helpers
const add = (a: V, b: V): V => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const sub = (a: V, b: V): V => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const mul = (a: V, s: number): V => [a[0] * s, a[1] * s, a[2] * s];
const flo = (a: V): V => [Math.floor(a[0]), Math.floor(a[1]), Math.floor(a[2])];
const shr = (a: V, n: number): V => [a[0] >> n, a[1] >> n, a[2] >> n];
const clampV = (a: V, lo: number | V, hi: number | V): V => a.map((x, i) => Math.min(Math.max(x, typeof lo === "number" ? lo : lo[i]!), typeof hi === "number" ? hi : hi[i]!)) as V;
const unit = (a: number): V => { const e: V = [0, 0, 0]; e[a] = 1; return e; };
const corner = (k: number): V => [k & 1, (k >> 1) & 1, k >> 2];
const maxAbs = (a: V) => Math.max(Math.abs(a[0]), Math.abs(a[1]), Math.abs(a[2]));
const clamp = (x: number, lo: number, hi: number) => Math.min(Math.max(x, lo), hi);
const mix = (a: number, b: number, t: number) => a * (1 - t) + b * t;
const sum8 = (v: number[]) => ((v[0]! + v[5]!) + (v[1]! + v[4]!)) + ((v[2]! + v[7]!) + (v[3]! + v[6]!));
const signBit = (x: number) => x < 0 || Object.is(x, -0);
const sign = (x: number) => x > 0 ? 1 : x < 0 ? -1 : 0;
const inLattice = (c: V) => c.every((x, a) => x >= 0 && x < NL[a]!);
const lerp8 = (v: number[], f: V) => {
  const x0 = (1 - f[0]) * v[0]! + f[0] * v[1]!, x1 = (1 - f[0]) * v[2]! + f[0] * v[3]!, x2 = (1 - f[0]) * v[4]! + f[0] * v[5]!, x3 = (1 - f[0]) * v[6]! + f[0] * v[7]!;
  return (1 - f[2]) * ((1 - f[1]) * x0 + f[1] * x1) + f[2] * ((1 - f[1]) * x2 + f[1] * x3);
};
const trilinearWeights = (c: number[], t: V) => sum8(c.map((x, k) => { const b = corner(k); return x * (b[0] ? t[0] : 1 - t[0]) * (b[1] ? t[1] : 1 - t[1]) * (b[2] ? t[2] : 1 - t[2]); }));

/** The brick state as the kernels read it: residency and tiers from dir, φ from the pool words, lane F's words. */
class World {
  readonly F: ReturnType<typeof fluidLayout>;
  readonly dt = f32(1 / 60); readonly hm = f32(1 / N);
  phi!: Float32Array; // the phi pool (f32 view)
  constructor(readonly L: BrickLayout, readonly dir: Uint32Array, public set: Uint32Array, readonly fcell: Uint32Array) { this.F = fluidLayout(L, { tiered: true }); }
  base(kind: "F" | "C", ch: string) { return this.L.base("phi", kind, ch); }
  cslotOfPage(P: V) { const w = this.dir[this.L.dir.page + this.L.pageIndex(P[0], P[1], P[2])]!; return w < SEAM ? w : NONE; }
  fslotOfBrick(b: V) { const cs = this.cslotOfPage(shr(b, 2)); if (cs === NONE) return NONE; return this.dir[this.L.dir.child + 64 * cs + ((b[0] & 3) | ((b[1] & 3) << 2) | ((b[2] & 3) << 4))]!; }
  tileInLattice(t: V) { return t.every((x, a) => x >= 0 && 4 * x < NL[a]!); }
  tileIndex(t: V) { return (t[0] & 7) | ((t[1] & 7) << 3) | ((t[2] & 7) << 6); }
  tileH(cs: number, i: number) { return ((this.dir[this.L.dir.cmask + CMASK_WORDS * cs + (i >> 5)]! >>> (i & 31)) & 1) === 1; }
  tileOwnerH(t: V) { if (!this.tileInLattice(t)) return false; const cs = this.cslotOfPage(shr(t, 3)); return cs !== NONE && this.tileH(cs, this.tileIndex(t)); }
  cellOwnerH(c: V) { return this.tileOwnerH(shr(c, 2)); }
  /** lcGate: last step's AIR_GATE bit of tile t (clear where no page holds it). */
  gate(t: V) {
    if (!this.tileInLattice(t)) return false;
    const cs = this.cslotOfPage(shr(t, 3)); if (cs === NONE) return false;
    const i = this.tileIndex(t); return ((this.set[this.L.set.airGate + CMASK_WORDS * cs + (i >> 5)]! >>> (i & 31)) & 1) === 1;
  }
  vertexCoarse(v: V): V | undefined {
    const tA = shr(v, 2), inside = v.map((x, a) => x - 4 * tA[a]! !== 0);
    for (let k = 0; k < 8; k++) {
      const b = corner(k);
      if (b.some((x, a) => inside[a] && x === 0)) continue;
      const t = sub(add(tA, b), [1, 1, 1]);
      // A tile of a non-resident page is no authority (nbVertexCoarse: no slot stores its corners).
      if (this.tileInLattice(t) && this.cslotOfPage(shr(t, 3)) !== NONE && !this.tileOwnerH(t)) return t;
    }
    return undefined;
  }
  cOwned(v: V) { return v.every((x, a) => x >= 0 && x <= NL[a]!) && this.vertexCoarse(v) !== undefined; }
  cVertex(ch: string, T: V) {
    if (T.some((x, a) => x < 0 || 4 * x > NL[a]!)) return FAR;
    const cs = this.cslotOfPage(shr(T, 3)); if (cs === NONE) return PHI_AIR_C; // absent in the lattice: 16 h (airc)
    return this.phi[this.base("C", ch) + BRICK_STRIDE * cs + this.tileIndex(T)]!;
  }
  fVertex(ch: string, v: V) {
    const s = this.fslotOfBrick(shr(v, 3)); if (s >= SEAM) return PHI_AIR_C;
    return this.phi[this.base("F", ch) + BRICK_STRIDE * s + cellLocal(v[0], v[1], v[2])]!;
  }
  vertexPhi(ch: string, v: V) {
    if (v.some((x, a) => x < 0 || x > NL[a]!)) return FAR;
    const T = this.vertexCoarse(v);
    if (T === undefined) return this.fVertex(ch, v);
    const t = mul(sub(v, mul(T, 4)), 0.25), values: number[] = [];
    for (let k = 0; k < 8; k++) {
      const b = corner(k), w = (b[0] ? t[0] : 1 - t[0]) * (b[1] ? t[1] : 1 - t[1]) * (b[2] ? t[2] : 1 - t[2]);
      values.push(w > 0 ? w * this.cVertex(ch, add(T, b)) : 0);
    }
    return sum8(values);
  }
  // lane F's words
  fgSlot(c: V) { if (c.some(x => x < 0)) return NONE; const b = shr(c, 3); if (!b.every((x, a) => x < Math.floor(NL[a]! / 8) + 1)) return NONE; const s = this.fslotOfBrick(b); return s < SEAM ? s : NONE; }
  fgCell(s: number, c: V) { return 512 * s + cellLocal(c[0], c[1], c[2]); }
  tileLoad(c: V, w: number, otherwise: number) { const s = this.fgSlot(c); if (s === NONE) return otherwise; const t = ((c[0] >> 2) & 1) | (((c[1] >> 2) & 1) << 1) | (((c[2] >> 2) & 1) << 2); return this.fcell[this.F.fcell.TILE + FT.WORDS * (8 * s + t) + w]!; }
  flags(c: V) { return this.tileLoad(c, FT.FLAGS, 0); }
  dist(c: V, k: number) { return (this.tileLoad(c, FT.DIST, 0xffffffff) >>> (4 * k)) & 15; }
  reach1(c: V) { return this.dist(c, 0) <= 2; }
  reach2(c: V) { return this.dist(c, 0) <= 3; }
  // Deviation 10 (lcTileFresh, lcNearKnown): lInit's fresh FT_DIST (its low 13 bits all ones, no dilation's), and
  // lNear's ±5 window about c holding no crossing tile (DIL4's source flags) outside a resident, non-fresh F slot.
  tileFresh(t: V) { return ((this.tileLoad(mul(t, 4), FT.DIST, 0xffffffff) & 0x1fff) >>> 0) === 0x1fff; }
  nearKnown(c: V) {
    const lo = shr(sub(c, [5, 5, 5]), 2), hi = shr(add(c, [5, 5, 5]), 2);
    for (let z = lo[2]; z <= hi[2]; z++) for (let y = lo[1]; y <= hi[1]; y++) for (let x = lo[0]; x <= hi[0]; x++) {
      const t: V = [x, y, z]; if (!this.tileInLattice(t)) continue;
      if (this.fgSlot(mul(t, 4)) !== NONE && !this.tileFresh(t)) continue;
      const cs = this.cslotOfPage(shr(t, 3)); if (cs === NONE) continue;
      if (((this.fcell[this.F.fcell.DIL4 + 512 * cs + this.tileIndex(t)]! >>> 24) & FTF.CROSSING) !== 0) return false;
    }
    return true;
  }
  binCount(c: V) {
    if (!inLattice(c)) return 0;
    const s = this.fgSlot(c); if (s === NONE) return 0;
    const count = this.set[this.L.set.binCount + s]!; if (count === 0) return 0;
    const l = cellLocal(c[0], c[1], c[2]), start = this.fcell[this.F.fcell.RUN + 512 * s + l]!;
    const end = l < 511 ? this.fcell[this.F.fcell.RUN + 512 * s + l + 1]! : this.set[this.L.set.binStart + s]! + count;
    return (end - start) >>> 0;
  }
  // tiers
  tileCold(t: V) { const cs = this.cslotOfPage(shr(t, 3)); return cs !== NONE && (this.dir[this.L.dir.crec + 32 * cs + REC.CLASS]! & F_CLASS.COLD) !== 0; }
  cellStored(c: V) { return inLattice(c) && this.cslotOfPage(shr(c, 5)) !== NONE; }
  /** nbrTileReach: tile t's spread reach word (REACH1), saturated where no page holds t or REACH_STEP is not the run's
   * ring step's (e.run binds ring slot 0). */
  tileReach(t: V) {
    const cs = this.cslotOfPage(shr(t, 3));
    if (cs === NONE || this.set[this.L.set.reachStep + cs] !== this.set[this.L.set.ring + RING.STEP]! + 1) return 0x3fffffff;
    return (this.set[this.L.set.reach1 + 512 * cs + this.tileIndex(t)]! & 0x3fffffff) >>> 0;
  }
  tierCold(v: V) { for (let k = 0; k < 8; k++) { const c = sub(v, corner(k)); if (!this.cellStored(c)) continue; if (!this.tileCold(shr(c, 2))) return false; } return true; }
  vertexCold(v: V) { for (let k = 0; k < 8; k++) { const c = sub(v, corner(k)); if (this.cellStored(c) && this.tileCold(shr(c, 2))) return true; } return false; }
  // sampling
  held(p: V) { return clampV(p, 0, N); }
  cellOf(v: V) { return clampV(v, 0, N - 1); }
  widthAt(c: V) { return this.cellOwnerH(c) ? 1 : 4; }
  sample(ch: string, p: V) {
    const q = this.held(p), cell = this.cellOf(flo(q)), w = this.widthAt(cell), o = mul(flo(mul(cell, 1 / w)), w), t = mul(sub(q, o), 1 / w);
    const zt = t.map(x => x === 0);
    if (zt.filter(Boolean).length >= 2) {
      const a = !zt[0] ? 0 : !zt[1] ? 1 : 2, l0 = this.vertexPhi(ch, o);
      if (t[a] === 0) return l0;
      return l0 * (1 - t[a]!) + this.vertexPhi(ch, add(o, mul(unit(a), w))) * t[a]!;
    }
    const c: number[] = []; for (let k = 0; k < 8; k++) c.push(this.vertexPhi(ch, add(o, mul(corner(k), w))));
    return trilinearWeights(c, t);
  }
  // lane F's sampler (FG_SAMPLE_WGSL) over US: every stored face holds U.
  faceOpen(c: V, a: number) { const t: V = [...c]; t[a] = 0; return inLattice(t) && c[a]! > 0 && c[a]! < N; }
  faceInLattice(c: V, a: number) { const hi: V = [...NL]; hi[a]! += 1; return c.every((x, i) => x >= 0 && x < hi[i]!); }
  ffaceStored(c: V, a: number) { return this.fslotOfBrick(shr(c, 3)) < SEAM ? f32(U[a]!) : 0; }
  cface(T: V, a: number) { if (T.some((x, i) => x < 0 || 4 * x > NL[i]!)) return 0; return this.cslotOfPage(shr(T, 3)) === NONE ? 0 : f32(U[a]!); }
  facePlane(c: V, a: number) { if (!this.faceInLattice(c, a)) return 0; if (this.cellOwnerH(c) || this.cellOwnerH(sub(c, unit(a)))) return this.ffaceStored(c, a); return this.cface(shr(c, 2), a); }
  faceU(c: V, a: number) {
    if (!this.faceInLattice(c, a)) return 0;
    if (this.cellOwnerH(c) || this.cellOwnerH(sub(c, unit(a)))) return this.ffaceStored(c, a);
    const T = shr(c, 2), k = c[a]! - 4 * T[a]!;
    if (k === 0) return this.cface(T, a);
    const cu: V = [...c]; cu[a] = 4 * T[a]! + 4;
    return mix(this.cface(T, a), this.facePlane(cu, a), k / 4);
  }
  fsCache(T: V, a: number) {
    const hi: V = [N / 4 - 1, N / 4 - 1, N / 4 - 1]; hi[a]! += 1;
    const t = clampV(T, 0, hi), c = mul(t, 4);
    if (!this.faceOpen(c, a)) return 0;
    if (!this.cellOwnerH(c) && !this.cellOwnerH(sub(c, unit(a)))) return this.cface(t, a);
    const u = (a + 1) % 3, w = (a + 2) % 3; let s = 0;
    for (let y = 0; y < 4; y++) for (let x = 0; x < 4; x++) { const q: V = [...c]; q[u]! += x; q[w]! += y; s += this.faceU(q, a); }
    return s / 16;
  }
  fsCoarse(p: V, a: number) {
    const off: V = [0.5, 0.5, 0.5]; off[a] = 0;
    const q = sub(mul(p, 0.25), off), b = flo(q), f = sub(q, b), v: number[] = [];
    for (let k = 0; k < 8; k++) v.push(this.fsCache(add(b, corner(k)), a));
    return lerp8(v, f);
  }
  fsTap4(c: V, a: number) {
    const q8 = sub(mul(c, 2), [3, 3, 3]); q8[a] = 2 * c[a]!;
    const b = shr(q8, 3), f = mul(sub(q8, mul(b, 8)), 1 / 8), t: number[] = [];
    for (let k = 0; k < 8; k++) { const cr = corner(k), w = (cr[0] ? f[0] : 1 - f[0]) * (cr[1] ? f[1] : 1 - f[1]) * (cr[2] ? f[2] : 1 - f[2]); t.push(w > 0 ? w * this.fsCache(add(b, cr), a) : 0); }
    return sum8(t);
  }
  fsTapH(c: V, a: number) { if (!this.faceOpen(c, a)) return 0; if (this.cellOwnerH(c) || this.cellOwnerH(sub(c, unit(a)))) return this.faceU(c, a); return this.fsTap4(c, a); }
  fsFine(p: V, a: number) {
    let s = sub(p, [0.5, 0.5, 0.5]); s[a] = p[a]!;
    const hi: V = [N - 1, N - 1, N - 1]; hi[a] = N; s = clampV(s, 0, hi);
    const i0 = flo(s), f = sub(s, i0), low = shr(i0, 2), high = shr(s.map(Math.ceil) as V, 2);
    let fine = true;
    for (let k = 0; k < 8; k++) { const b = corner(k); if (!this.tileOwnerH(b.map((x, i) => x ? high[i]! : low[i]!) as V)) fine = false; }
    const v: number[] = [];
    for (let k = 0; k < 8; k++) { const c = add(i0, corner(k)); v.push(fine ? (this.faceOpen(c, a) ? this.faceU(c, a) : 0) : this.fsTapH(c, a)); }
    return lerp8(v, f);
  }
  fsFineWeight(p: V) {
    const q = this.held(p), t = shr(flo(q).map(x => Math.min(x, N - 1)) as V, 2);
    if (this.dist(mul(t, 4), 2) > 2) return 0;
    if (this.tileOwnerH(t)) return 1;
    let fine = 0;
    for (let k = 0; k < 27; k++) {
      if (k === 13) continue;
      const n = sub(add(t, [k % 3, Math.floor(k / 3) % 3, Math.floor(k / 9)]), [1, 1, 1]);
      if (!this.tileOwnerH(n)) continue;
      const o = mul(n, 4), d = o.map((x, i) => Math.max(0, x - q[i]!, q[i]! - x - 4)) as V;
      fine = Math.max(fine, Math.max(0, 1 - maxAbs(d) / 2));
    }
    return fine;
  }
  vel(p: V, lo = 0, hi = 3): V {
    const v: V = [0, 0, 0], fine = this.fsFineWeight(p), coarse = Math.max(0, 1 - fine);
    for (let a = lo; a < hi; a++) { let x = 0; if (fine > 0) x += fine * this.fsFine(p, a); if (coarse > 0) x += coarse * this.fsCoarse(p, a); v[a] = x; }
    return v;
  }
}

// ---- the formulas (levelset-formulas.wgsl.ts), in Float64
function surfaceTrace(W: World, p: V, oneStep: boolean, fatal: { n: number }): V {
  const dt = W.dt, h = W.hm;
  let q = p, x = p, steps = 1, i = 0, base = 0, stage = 0, k1: V = [0, 0, 0], k2: V = [0, 0, 0], two = false, travel = 1;
  for (let n = 0; n < 387; n++) {
    const k = mul(W.vel(x), 1 / h);
    if (stage === 0) {
      if (k.every(c => c === 0)) { const held = W.held(q); if (held.every((c, a) => c === q[a])) return held; }
      k1 = k;
      if (i === 0) {
        steps = Math.max(1, Math.ceil(dt * maxAbs(k1)) >>> 0);
        if (oneStep) steps = 1;
        if (steps > 128) { fatal.n++; return p; }
        travel = steps; steps = Math.floor((travel + 31) / 32); base = -dt / steps;
      }
      x = add(q, mul(k1, base)); stage = 1; continue;
    }
    if (stage === 1) {
      k2 = k;
      const turn = maxAbs(sub(k2, k1));
      two = steps - i >= 2 && 2 * Math.abs(base) * maxAbs(k1) <= 2.001 && 2 * Math.abs(base) * turn <= 0.125;
      x = two ? add(q, mul(sub(mul(k2, 2), k1), 2 * base)) : add(q, mul(add(k1, k2), 0.25 * base));
      stage = 2; continue;
    }
    const end = two ? add(q, mul(add(add(k1, mul(k2, 4)), k), base / 3)) : add(q, mul(add(add(k1, k2), mul(k, 4)), base / 6));
    i += two ? 2 : 1; q = W.held(end);
    if (i >= steps) break;
    x = q; stage = 0;
  }
  return q;
}
const catmull = (t: number) => { const t2 = t * t, t3 = t2 * t; return [2 * t2 - t3 - t, 3 * t3 - 5 * t2 + 2, 4 * t2 - 3 * t3 + t, t3 - t2].map(x => 0.5 * x); };
const heldWeights = (w: number[], low: boolean, high: boolean) => { const a = low ? w[0]! : 0, b = high ? w[3]! : 0; return [w[0]! - a, w[1]! + 2 * a - b, w[2]! - a + 2 * b, w[3]! - b]; };
function cubicLimited(f: V, lowHeld: boolean[], highHeld: boolean[], tap: (x: number, y: number, z: number) => number) {
  const wx = heldWeights(catmull(f[0]), lowHeld[0]!, highHeld[0]!), wy = heldWeights(catmull(f[1]), lowHeld[1]!, highHeld[1]!), wz = heldWeights(catmull(f[2]), lowHeld[2]!, highHeld[2]!);
  let value = 0, low = 1e30, high = -1e30, zEndLow = 0, zEndHigh = 0;
  const upper = f.map(x => x >= 0.5);
  for (let z = 0; z < 4; z++) {
    let plane = 0, planeLow = 1e30, planeHigh = -1e30, yEndLow = 0, yEndHigh = 0;
    for (let y = 0; y < 4; y++) {
      let row = 0, rowLow = 1e30, rowHigh = -1e30, xEnd = 0;
      for (let x = 0; x < 4; x++) {
        const s = tap(x, y, z); row += wx[x]! * s;
        if (x === 1 || x === 2) { rowLow = Math.min(rowLow, s); rowHigh = Math.max(rowHigh, s); } else if ((x === 3) === upper[0]) xEnd = s;
      }
      plane += wy[y]! * row; rowLow = Math.min(rowLow, xEnd); rowHigh = Math.max(rowHigh, xEnd);
      if (y === 1 || y === 2) { planeLow = Math.min(planeLow, rowLow); planeHigh = Math.max(planeHigh, rowHigh); } else if ((y === 3) === upper[1]) { yEndLow = rowLow; yEndHigh = rowHigh; }
    }
    value += wz[z]! * plane; planeLow = Math.min(planeLow, yEndLow); planeHigh = Math.max(planeHigh, yEndHigh);
    if (z === 1 || z === 2) { low = Math.min(low, planeLow); high = Math.max(high, planeHigh); } else if ((z === 3) === upper[2]) { zEndLow = planeLow; zEndHigh = planeHigh; }
  }
  low = Math.min(low, zEndLow); high = Math.max(high, zEndHigh);
  return clamp(value, low, high);
}
const SURFACE_RADIUS = 0.875, SEED_DEPTH = 0.8003905296791061;
const smoothstep = (e0: number, e1: number, x: number) => { const t = clamp((x - e0) / (e1 - e0), 0, 1); return t * t * (3 - 2 * t); };
function unionParticle(bulk: number, n2: number) {
  const sphere = Math.sqrt(n2) - SURFACE_RADIUS, shell = Math.sqrt(Math.max(n2 - 0.125, 0)) - SEED_DEPTH;
  let excess = bulk - clamp(bulk, Math.min(sphere, shell), Math.max(sphere, shell));
  if (n2 <= 0.25) excess = Math.max(bulk - sphere, 0);
  const ripple = SURFACE_RADIUS - SEED_DEPTH;
  return Math.min(bulk + 1, bulk - excess * smoothstep(0.5 * ripple, ripple, Math.abs(excess)));
}
function unionBound(particle: number, bulk: number, held: number, open: number, heldLow: V, heldHigh: V, openLow: V, openHigh: V) {
  if (open === 8) return Math.max(particle, Math.min(bulk, 1 - held / (4 * open)));
  if (open > 0) {
    let gap = -1, bound = 1;
    for (let a = 0; a < 3; a++) {
      const low = openLow[a]! > 0 ? heldLow[a]! / (8 * openLow[a]!) : 1, high = openHigh[a]! > 0 ? heldHigh[a]! / (8 * openHigh[a]!) : 1;
      if (Math.abs(low - high) <= gap) continue;
      gap = Math.abs(low - high); const lowDense = low >= high;
      const denseOpen = (lowDense ? openLow[a]! : openHigh[a]!) > 0, sparseOpen = (lowDense ? openHigh[a]! : openLow[a]!) > 0;
      const sparse = sparseOpen ? (lowDense ? high : low) : 0;
      bound = denseOpen ? 1 - Math.max(low, high) - sparse : -sparse;
    }
    return Math.max(particle, Math.min(bulk, bound));
  }
  return particle;
}
function bulkDepth(phi: number, p: V, reached: boolean, hasNearest: boolean, low: V, travel: number) {
  if (phi >= 0) return phi;
  let depth = 8;
  if (reached) { depth = 5; if (hasNearest) { const d = low.map((l, a) => Math.max(l - p[a]!, p[a]! - (l + 1), 0)); depth = Math.hypot(...d); } }
  if (travel > 0) depth = Math.max(depth - travel - 2, 0);
  return Math.min(phi, -depth);
}
function trilinear(v: number[], q: V): [number, number, number, number] {
  const x00 = mix(v[0]!, v[1]!, q[0]), x10 = mix(v[2]!, v[3]!, q[0]), x01 = mix(v[4]!, v[5]!, q[0]), x11 = mix(v[6]!, v[7]!, q[0]);
  const y0 = mix(x00, x10, q[1]), y1 = mix(x01, x11, q[1]);
  const dx = mix(mix(v[1]! - v[0]!, v[3]! - v[2]!, q[1]), mix(v[5]! - v[4]!, v[7]! - v[6]!, q[1]), q[2]);
  return [dx, mix(x10 - x00, x11 - x01, q[2]), y1 - y0, mix(y0, y1, q[2])];
}
function cellDistance(q: V, v: number[]) {
  let low = v[0]!, high = v[0]!;
  for (let k = 1; k < 8; k++) { low = Math.min(low, v[k]!); high = Math.max(high, v[k]!); }
  if (low > 0 || high < 0 || low === high) return -1;
  let distance = Math.hypot(...q) + Math.sqrt(3);
  for (let a = 0; a < 3; a++) for (let k = 0; k < 8; k++) {
    const b = 1 << a; if (k & b) continue; const o = k | b;
    if (v[k]! * v[o]! > 0 || v[k] === v[o]) continue;
    const t = clamp(v[k]! / (v[k]! - v[o]!), 0, 1), ck = corner(k), co = corner(o);
    distance = Math.min(distance, Math.hypot(...sub(q, ck.map((x, i) => mix(x, co[i]!, t)) as V)));
  }
  let r = clampV(q, 0, 1);
  for (let i = 0; i < 6; i++) { const f = trilinear(v, r), g2 = f[0] * f[0] + f[1] * f[1] + f[2] * f[2]; if (g2 < 1e-12) break; r = clampV(sub(r, mul([f[0], f[1], f[2]], f[3] / g2)), 0, 1); }
  if (Math.abs(trilinear(v, r)[3]) < 1e-4) distance = Math.min(distance, Math.hypot(...sub(q, r)));
  return distance;
}
const farSentinel = (x: number, h: number) => Math.abs(x) > 1e-8 && Math.abs(x) < 6 * h ? sign(x) * 6 * h : x;

// ---- the kernels' oracles (levelset-coarse.wgsl.ts), per C vertex
interface Lane { cs: number; l: number; v: V }
function lanes(W: World, slots: number[]): Lane[] {
  const out: Lane[] = [];
  for (const cs of slots) {
    const o = mul(brickKeyCoord(W.dir[W.L.dir.crec + 32 * cs + REC.KEY]!) as V, 32);
    for (let l = 0; l < 512; l++) { const v = add(o, mul([l & 7, (l >> 3) & 7, l >> 6], 4)); if (W.cOwned(v)) out.push({ cs, l, v }); }
  }
  return out;
}
/** The page's live tiles and seeds (19^3 about it), the collar (the seeds dilated 4 tiles an axis, x then y then z,
 * each axis held to the live tiles) and the gate words of its own 512 tiles (a live tile with the collar in its 3^3). */
/** nbCrossingTravel (brick-residency): the tile's bulk-depth travel from its spread reach word, 4 (max(reach, 2) - 2)
 * an axis, unbounded where a reach saturates. */
function crossingTravel(w: number) {
  const travel = [0, 0, 0];
  for (let a = 0; a < 3; a++) { const reach = Math.max((w >>> (5 * a)) & 31, (w >>> (15 + 5 * a)) & 31); if (reach >= 31) return 3e38; travel[a] = 4 * (Math.max(reach, 2) - 2); }
  return Math.fround(Math.hypot(...travel));
}
function seedsOf(W: World, cs: number) {
  const P = brickKeyCoord(W.dir[W.L.dir.crec + 32 * cs + REC.KEY]!) as V, T0 = sub(mul(P, 8), [6, 6, 6]);
  const n = 19 ** 3, ix = (r: V) => r[0] + 19 * (r[1] + 19 * r[2]), at = (t: V) => { const r = sub(t, T0); return r.every(x => x >= 0 && x < 19) ? ix(r) : -1; };
  const live = new Uint8Array(n), seed = new Uint8Array(n);
  for (let z = 0; z < 19; z++) for (let y = 0; y < 19; y++) for (let x = 0; x < 19; x++) {
    const t = add(T0, [x, y, z]);
    if (!W.tileInLattice(t)) continue;
    const tcs = W.cslotOfPage(shr(t, 3)); if (tcs === NONE) continue;
    live[ix([x, y, z])] = 1;
    const flags = W.flags(mul(t, 4));
    let s: boolean;
    if (W.tileH(tcs, W.tileIndex(t))) s = (flags & FTF.AIR_SEED) !== 0;
    else if (flags & FTF.SURFACE_TILE) s = true;
    else { s = false; for (let k = 0; k < 8; k++) if (W.cVertex("PHI", add(t, corner(k))) <= 0) s = true; }
    if (s) seed[ix([x, y, z])] = 1;
  }
  const dilate = (src: Uint8Array, a: number) => {
    const out = new Uint8Array(n);
    for (let z = 0; z < 19; z++) for (let y = 0; y < 19; y++) for (let x = 0; x < 19; x++) {
      const r: V = [x, y, z]; if (!live[ix(r)]) continue;
      for (let o = -4; o <= 4; o++) { const q: V = [...r]; q[a] += o; if (q[a]! >= 0 && q[a]! < 19 && src[ix(q)]) { out[ix(r)] = 1; break; } }
    }
    return out;
  };
  const collar = dilate(dilate(dilate(seed, 0), 1), 2);
  const gateWords = () => {
    const w = new Uint32Array(CMASK_WORDS);
    for (let l = 0; l < 512; l++) {
      const r: V = [6 + (l & 7), 6 + ((l >> 3) & 7), 6 + (l >> 6)]; if (!live[ix(r)]) continue;
      let near = false;
      for (let k = 0; k < 27; k++) if (collar[ix(add(r, [(k % 3) - 1, (Math.floor(k / 3) % 3) - 1, Math.floor(k / 9) - 1]))]) near = true;
      if (near) w[l >> 5] = (w[l >> 5]! | (1 << (l & 31))) >>> 0;
    }
    return w;
  };
  return { seedAt: (t: V) => { const i = at(t); return i >= 0 && seed[i] === 1; }, collar: (t: V) => { const i = at(t); return i >= 0 && collar[i] === 1; }, gateWords };
}

type Tally = Record<string, number>;
const bump = (t: Tally, k: string) => { t[k] = (t[k] ?? 0) + 1; };

function advectOracle(W: World, slots: number[], tally: Tally, fatal: { n: number }): Map<string, number> {
  const out = new Map<string, number>(), dt = W.dt, hm: V = [W.hm, W.hm, W.hm], w = 4;
  for (const { cs, l, v } of lanes(W, slots)) {
    const c = W.phi[W.base("C", "PHI") + BRICK_STRIDE * cs + l]!, key = `${cs}:${l}`, wall = v.some((x, a) => x === 0 || x === NL[a]);
    if (W.tierCold(v)) { out.set(key, c); bump(tally, wall ? "wall cold" : "cold hold"); continue; }
    if (c >= 16 && !W.gate(shr(W.cellOf(v), 2))) { out.set(key, c); bump(tally, wall ? "wall far air" : "far air"); continue; }
    bump(tally, wall ? "wall full path" : "full path");
    // lcFullAdvected
    const p: V = [...v];
    let foot = p, shift: V = [0, 0, 0], smoothing = 0;
    if (Math.abs(c) < 2 * w) {
      const at: V[] = [];
      for (let a = 0; a < 3; a++) { const e = mul(unit(a), w); at.push(W.held(sub(p, e)), W.held(add(p, e))); }
      for (let k = 0; k < 4; k++) { const q: V = [...p]; q[k > 1 ? 2 : 0] += (k & 1 ? 2 : -2) * w; at.push(q); }
      const t = at.map(q => W.vertexPhi("PHI", flo(q)));
      const g = [0, 1, 2].map(a => (t[2 * a + 1]! - t[2 * a]!) / Math.max(at[2 * a + 1]![a]! - at[2 * a]![a]!, 1e-6)) as V;
      const g2 = g[0] * g[0] + g[1] * g[1] + g[2] * g[2];
      if (g2 >= 1e-6) { let off = mul(g, -c / g2); const reach = Math.hypot(...off); if (reach > 2 * w) off = mul(off, 2 * w / reach); foot = W.held(add(p, off)); shift = sub(p, foot); bump(tally, "foot"); }
      const eps = Math.min(0.5 * dt, 0.03125);
      if (eps > 0 && [t[0]!, t[1]!, t[4]!, t[5]!, t[6]!, t[7]!, t[8]!, t[9]!].some(s => signBit(s) !== signBit(c))) {
        let s = 0, crossing = false;
        for (const a of [0, 2]) {
          if (v[a]! < 8 || v[a]! > N - 8) continue;
          let part = 6 * c, valid = true, different = false;
          for (let k = 0; k < 4; k++) {
            const x = t[k > 1 ? 6 + a + k - 2 : 2 * a + k]!, span = k > 1 ? 2 * w : w;
            if (Math.abs(x - c) > 1.5 * span) valid = false;
            if (signBit(x) !== signBit(c)) different = true;
            part += (k > 1 ? 1 : -4) * x;
          }
          if (valid) { s += part; crossing ||= different; }
        }
        if (crossing) { smoothing = -eps * s; bump(tally, "smoothing"); }
      }
    }
    const end = surfaceTrace(W, foot, wall, fatal), travel = sub(end, foot);
    let q = end;
    if (shift.some(x => x !== 0)) q = W.held(add(end, shift));
    const cell = W.cellOf(flo(q)), width = W.widthAt(cell), origin = mul(flo(mul(cell, 1 / width)), width), f = mul(sub(q, origin), 1 / width);
    const corners: number[] = []; for (let k = 0; k < 8; k++) corners.push(W.vertexPhi("PHI", add(origin, mul(corner(k), width))));
    let value = trilinearWeights(corners, f);
    const moved = smoothing * clamp(2 - 2 * maxAbs(sub(q, p)) / 4, 0, 1);
    if (Math.abs(value) < 2 * w) {
      bump(tally, "cubic");
      value = cubicLimited(f, origin.map(x => x - width < 0), origin.map((x, a) => x + 2 * width > NL[a]!),
        (x, y, z) => W.vertexPhi("PHI", clampV(add(origin, mul([x - 1, y - 1, z - 1], width)), 0, NL)));
    }
    // lcWallContact
    const tr: V = [...travel]; let contact = false;
    for (let k = 0; k < 6; k++) {
      const a = k >> 1, upper = (k & 1) === 1, inward = upper ? -1 : 1;
      if (!(upper ? p[a] === NL[a] : p[a] === 0)) continue;
      const probe: V = [...p]; probe[a] += inward * 4;
      const speed = W.vel(probe, a, a + 1)[a]!;
      if (inward * speed >= -1e-6) continue;
      contact = true; tr[a] = -dt * speed / hm[a]!;
    }
    if (contact) { const m = W.sample("PHI", W.held(add(p, tr))); if (m < value) bump(tally, "wall contact"); value = Math.min(value, m); }
    out.set(key, value + moved);
  }
  return out;
}

function heatCorner(W: World, base: V, k: number, surface: boolean): [number, number] {
  const t = clampV(add(base, corner(k)), 0, N / 4 - 1);
  const heat = bitsf(W.tileLoad(mul(t, 4), surface ? FT.SURF : FT.THETA, 0));
  if (surface && heat <= 0 && W.sample("PHI1", add(mul(t, 4), [2, 2, 2])) >= 0) return [heat, 0];
  return [heat, 1];
}
function interpolatedHeat(W: World, p: V, surface: boolean) {
  const q = sub(mul(p, 0.25), [0.5, 0.5, 0.5]), base = flo(q), f = sub(q, base);
  let theta = 0, covered = 0;
  for (let k = 0; k < 8; k++) {
    const b = corner(k), cr = heatCorner(W, base, k, surface); if (cr[1] <= 0) continue;
    const weight = (b[0] ? f[0] : 1 - f[0]) * (b[1] ? f[1] : 1 - f[1]) * (b[2] ? f[2] : 1 - f[2]);
    covered += weight; theta += weight * clamp(cr[0] - (surface ? 1 : 0), 0, 1);
  }
  return clamp(theta / Math.max(covered, 1e-20), 0, 1);
}
function coupleOracle(W: World, slots: number[], tally: Tally): Map<string, number> {
  const out = new Map<string, number>(), dt = W.dt, hm = W.hm;
  for (const { cs, l, v } of lanes(W, slots)) {
    if (W.vertexCold(v)) { bump(tally, "cold"); continue; }
    const key = `${cs}:${l}`, q = v, theta = interpolatedHeat(W, q, false), band = W.sample("PHI1", q);
    if (theta <= 0) { out.set(key, band); bump(tally, "no heat"); continue; }
    const c = W.cellOf(v), s = W.fgSlot(c);
    // particleSurfaceBulk's travel: nbCrossingTravel of the tile's spread reach word (617f9291's lcTileReach).
    const travel = crossingTravel(W.tileReach(shr(c, 2)));
    // Deviation 10: the guard's F words only where known, else band.
    let bulk = band;
    if (band < 0 && s !== NONE && !W.tileFresh(shr(c, 2)) && W.nearKnown(c)) {
      const reached = W.reach1(c);
      let nearest = NONE; if (reached) nearest = W.fcell[W.F.fcell.NEAR + W.fgCell(s, c)]!;
      bulk = bulkDepth(band, q, reached, nearest !== NONE, [nearest & 1023, (nearest >>> 10) & 1023, nearest >>> 20], travel);
    } else if (band < 0) bump(tally, "guard unknown");
    if (!((W.flags(c) & FTF.SURFACE_TILE) !== 0 && bulk > -2)) { out.set(key, bulk); bump(tally, "bulk"); continue; }
    const vs = W.fgSlot(v);
    let k = 0, hot = 0; if (vs !== NONE) { k = W.fcell[W.F.fcell.KEY + W.fgCell(vs, v)]!; hot = W.fcell[W.F.fcell.HOT + W.fgCell(vs, v)]!; }
    const n2 = k === 0 ? 4 : bitsf((~k) >>> 0);
    if (n2 >= 4) { out.set(key, bulk); bump(tally, "no sample"); continue; }
    let particle = unionParticle(bulk, n2);
    if (particle < bulk) {
      let held = 0, open = 0; const hl: V = [0, 0, 0], hh: V = [0, 0, 0], ol: V = [0, 0, 0], oh: V = [0, 0, 0];
      for (let j = 0; j < 8; j++) {
        const b = corner(j), cc = sub(add(v, b), [1, 1, 1]);
        if (!inLattice(cc)) continue;
        // R1 (lane E): a 4h-owned cell (fgCellCoarse) counts the 4h authority's fill, its retired records being no
        // evidence of air: clamp(0.5 - φ, 0, 1) of the advected C φ at the cell centre, as eight records' worth.
        const coarse = !W.tileOwnerH(shr(cc, 2));
        if (coarse) bump(tally, "coarse fill");
        const n = coarse ? Math.round(8 * Math.min(1, Math.max(0, f32(0.5 - W.sample("PHI1", add(cc, [0.5, 0.5, 0.5])))))) : Math.min(W.binCount(cc), 8);
        open++; held += n;
        for (let a = 0; a < 3; a++) if (b[a]) { hh[a] += n; oh[a]++; } else { hl[a] += n; ol[a]++; }
      }
      if (held > 0) bump(tally, "bound bins");
      particle = unionBound(particle, bulk, held, open, hl, hh, ol, oh);
      bump(tally, "union adds");
    }
    let blend = theta;
    if (particle > bulk) {
      if (!(hot !== 0 && bitsf((~hot) >>> 0) <= n2)) { out.set(key, bulk); bump(tally, "erasure refused"); continue; }
      blend = interpolatedHeat(W, q, true); bump(tally, "erasure");
    }
    out.set(key, mix(bulk, particle, blend)); bump(tally, "union");
  }
  return out;
}

/** lcClear: the gate words of every C_ALL page (into setWant) and the raised C vertices. */
function clearOracle(W: World, slots: number[], tally: Tally, setWant: Map<number, number>): Map<string, number> {
  const out = new Map<string, number>();
  const bySlot = new Map<number, Lane[]>();
  for (const lane of lanes(W, slots)) { if (!bySlot.has(lane.cs)) bySlot.set(lane.cs, []); bySlot.get(lane.cs)!.push(lane); }
  for (const cs of slots) {
    const S = seedsOf(W, cs), gate = S.gateWords();
    for (let j = 0; j < CMASK_WORDS; j++) setWant.set(W.L.set.airGate + CMASK_WORDS * cs + j, gate[j]!);
    for (const { l, v } of bySlot.get(cs) ?? []) {
      const x = W.phi[W.base("C", "PHI") + BRICK_STRIDE * cs + l]!;
      if (x >= 16) continue;
      let prot = false;
      for (let k = 0; k < 8; k++) { const t = shr(sub(v, corner(k)), 2); if (x <= 0 ? S.seedAt(t) : S.collar(t)) prot = true; }
      if (prot) { bump(tally, x <= 0 ? "seed" : "collar"); continue; }
      if (W.vertexCold(v)) { bump(tally, "cold"); continue; }
      out.set(`${cs}:${l}`, Math.max(x, 16)); bump(tally, "raised");
    }
  }
  return out;
}

function retireOracle(W: World, slots: number[], tally: Tally): Map<string, number> {
  const out = new Map<string, number>();
  const gradient = (c: V, q: V) => {
    const width = W.widthAt(c), o = mul(flo(mul(c, 1 / width)), width), f = mul(sub(q, o), 1 / width), v: number[] = [];
    for (let k = 0; k < 8; k++) v.push(W.vertexPhi("PHI", add(o, mul(corner(k), width))));
    const t = trilinear(v, f);
    return { g: [t[0] / width, t[1] / width, t[2] / width] as V, value: trilinearWeights(v, f) };
  };
  const build = (v: V, c: V) => {
    const initial = W.sample("PHI", v);
    if (Math.abs(initial) <= 1e-8) return initial;
    let nearest = NONE;
    if (W.reach1(c)) { const s = W.fgSlot(c); if (s !== NONE && W.nearKnown(c)) nearest = W.fcell[W.F.fcell.NEAR + W.fgCell(s, c)]!; }
    let distance = Math.max(Math.abs(initial), 5);
    if (nearest !== NONE) {
      const base: V = [nearest & 1023, (nearest >>> 10) & 1023, nearest >>> 20], cv: number[] = [];
      for (let k = 0; k < 8; k++) cv.push(W.vertexPhi("PHI", add(base, corner(k))));
      const cand = cellDistance(sub(v, base), cv); if (cand >= 0) { distance = cand; bump(tally, "crossing cell"); }
    }
    if (Math.abs(initial) < 4.5) {
      let q: V = [...v], cell = W.cellOf(flo(q));
      for (let it = 0; it < 5; it++) {
        const s = gradient(cell, q);
        if (Math.abs(s.value) < 1e-5) { distance = Math.min(distance, Math.hypot(...sub(q, v))); break; }
        const g2 = s.g[0] ** 2 + s.g[1] ** 2 + s.g[2] ** 2; if (g2 < 1e-8) break;
        q = W.held(sub(q, mul(s.g, clamp(s.value / Math.sqrt(g2), -1.5, 1.5) / Math.sqrt(g2)))); cell = W.cellOf(flo(q));
      }
      bump(tally, "newton");
    }
    return sign(initial) * Math.max(distance, 1e-8);
  };
  const closed = (q: V) => { for (let k = 0; k < 8; k++) if (inLattice(sub(q, corner(k)))) return false; return true; };
  const zeroOpen = (p: V) => { const lo = sub(p.map(Math.ceil) as V, [1, 1, 1]), hi = flo(p); for (let k = 0; k < 8; k++) if (inLattice(corner(k).map((b, a) => b ? hi[a]! : lo[a]!) as V)) return true; return false; };
  const bound = (v: V, d: number) => {
    let at = 0, b = Math.abs(d);
    for (let a = 0; a < 7; a++) {
      const q: V = [...v]; if (a > 0) q[(a - 1) >> 1] += ((a - 1) & 1) ? -4 : 4;
      if (q.some(x => x < 0)) continue;
      if (a > 0 && closed(q)) continue;
      const n = W.sample("PHI", q);
      if (a === 0) at = n;
      else if (at * n < 0) {
        const t = Math.abs(at) / (Math.abs(at) + Math.abs(n)), axis = (a - 1) >> 1, zero: V = [...v]; zero[axis] = mix(v[axis]!, q[axis]!, t);
        if (zeroOpen(zero)) b = Math.min(b, 4 * t);
      }
    }
    return d >= 0 ? b : -b;
  };
  for (const { cs, l, v } of lanes(W, slots)) {
    const cur = W.phi[W.base("C", "PHI") + BRICK_STRIDE * cs + l]!, key = `${cs}:${l}`;
    if (W.vertexCold(v)) { out.set(key, cur); bump(tally, "cold"); continue; }
    const c = W.cellOf(v), t = shr(c, 2);
    let fine = false; for (let k = 0; k < 27; k++) if (W.tileOwnerH(sub(add(t, [k % 3, Math.floor(k / 3) % 3, Math.floor(k / 9)]), [1, 1, 1]))) fine = true;
    let hot = false; const base = sub(shr(v, 2), [1, 1, 1]);
    for (let k = 0; k < 8; k++) if (bitsf(W.tileLoad(mul(clampV(add(base, corner(k)), 0, N / 4 - 1), 4), FT.SURF, 0)) > 1) hot = true;
    if (!fine || hot) { out.set(key, cur); bump(tally, fine ? "hot" : "no h stencil"); continue; }
    if (W.reach2(c)) { out.set(key, bound(v, build(v, c))); bump(tally, "rebuilt"); continue; }
    const x = farSentinel(cur, 1); out.set(key, x); bump(tally, x !== cur ? "sentinel" : "far kept");
  }
  return out;
}

interface Fixture {
  /** The boot's liquid box (high corner). */ hi: V;
  /** h tiles to commit over the boot's tiers (WANT then setCommitMask), as brick-set-tiers.test.ts test 1. */ want?: (T: V) => boolean;
  /** Pages made COLD (by key; those resident). */ cold: number[];
  /** Assert every kernel's paths are covered, and force the BULK16 latch at the end. */ full: boolean;
}
async function runFixture(device: GPUDevice, fx: Fixture) {
    const stages = (c: Parameters<typeof bringUpStages>[0]) => ({ ...bringUpStages(c), tiered: true, buffers: { fcell: fluidLayout(c.layout, { tiered: true }).fcell.words } });
    const e = await BrickEngine.create(device, { lattice: NL, h_m: 1 / N, dt_s: 1 / 60, gravity_mps2: [0, -9.81, 0],
      capacities: { f: 600, c: 40, b: 0, k: 0, particles: 8192 }, liquid: [{ lo: [-1, -1, -1], hi: fx.hi }], solver: L => bringUpCG(L), stages });
    const L = e.layout, F = fluidLayout(L, { tiered: true });
    if (fx.want) {
      const boot = await snapshot(e);
      for (const cs of boot.pages.values()) {
        const P = brickKeyCoord(boot.dir[L.dir.crec + 32 * cs + REC.KEY]!) as V, want = new Uint32Array(CMASK_WORDS);
        for (let i = 0; i < 512; i++) { const T = add(mul(P, 8), [i & 7, (i >> 3) & 7, i >> 6]); if (T.every(x => 4 * x < N) && fx.want(T)) want[i >> 5] = (want[i >> 5]! | (1 << (i & 31))) >>> 0; }
        e.write("set", L.set.want + CMASK_WORDS * cs, want);
      }
      await e.run([{ kernel: brickTransferKernels(L).setCommitMask, dispatch: { indirect: 12 * LIST.C_ALL } }]);
    }
    const s0 = await snapshot(e);
    assert.equal(s0.header[H.FAIL], 0);
    const slots = s0.list(LIST.C_ALL);
    assert.equal(slots.length, s0.pages.size, "C_ALL lists every page");
    // ---- lane F's words: hashed tile flags, distances and heats; crossing cells; splat keys; bounds; bins.
    const fcell = new Uint32Array(F.fcell.words), set = new Uint32Array(s0.set);
    for (const [bk, fs] of s0.bricks) {
      const b = brickKeyCoord(bk) as V;
      for (let t = 0; t < 8; t++) {
        const T = add(mul(b, 2), corner(t)), w = (k: number) => F.fcell.TILE + FT.WORDS * (8 * fs + t) + k, r = (k: number) => hash(T[0], T[1], T[2], k);
        let flags = 0;
        if (T[1] <= 6 && r(1) < 0.3) flags |= FTF.SURFACE_TILE;
        if (T[1] <= 6 && r(2) < 0.3) flags |= FTF.AIR_SEED;
        fcell[w(FT.FLAGS)] = flags;
        let dist = 0; for (let k = 0; k < 8; k++) { const x = r(10 + k); dist |= (x < 0.15 ? 15 : Math.floor(x * 5)) << (4 * k); }
        fcell[w(FT.DIST)] = dist >>> 0;
        fcell[w(FT.THETA)] = fbits(r(3) < 0.65 ? f32(1.4 * r(4)) : 0);
        fcell[w(FT.SURF)] = fbits(r(5) < 0.4 ? f32(1.3 * r(6)) : 0);
      }
      fcell[F.fcell.BOUND + fs] = fbits(f32(0.2 + 3 * hash(bk, 7)));
      // A slot's runs stay inside the bank, as pScan's are (fgRun holds a run to its bin, the bank's records): the
      // slots' bins overlap, which no kernel here reads, and end before 2047 + 512 x 10 < 8192.
      const start = (37 * fs) % 2048; let at = start;
      for (let l = 0; l < 512; l++) {
        const c = add(mul(b, 8), [l & 7, (l >> 3) & 7, l >> 6]), r = (k: number) => hash(c[0], c[1], c[2], 100 + k);
        fcell[F.fcell.RUN + 512 * fs + l] = at; at += Math.floor(r(0) * 11);
        // The nearest crossing cell: half the cells name the cell of phi0's zero on their column (as lane F's
        // redistance would), a quarter a hashed cell near them, a quarter none.
        const zero = Math.floor(21.3 - (2.2 * Math.sin(0.19 * c[0] + 0.4) * Math.cos(0.23 * c[2]) + 0.05 * (c[0] - 30)) / 0.45);
        const near: V = r(1) < 0.5 ? [c[0], clamp(zero, 0, N - 1), c[2]] : clampV(add(c, [Math.floor(5 * r(2)) - 2, Math.floor(5 * r(3)) - 2, Math.floor(5 * r(4)) - 2]), 0, N - 1);
        fcell[F.fcell.NEAR + 512 * fs + l] = r(9) < 0.75 ? (near[0] | (near[1] << 10) | (near[2] << 20)) >>> 0 : NONE;
        if (r(5) < 0.6) fcell[F.fcell.KEY + 512 * fs + l] = (~fbits(f32(4.5 * r(6)))) >>> 0;
        if (r(7) < 0.5) fcell[F.fcell.HOT + 512 * fs + l] = (~fbits(f32(4 * r(8)))) >>> 0;
      }
      set[L.set.binStart + fs] = start; set[L.set.binCount + fs] = at - start;
    }
    // DIL4's source flags (deviation 10's lcNearKnown reads the crossing bit of tiles no F slot holds): a hashed tenth.
    for (const [pk, cs] of s0.pages) {
      const P = brickKeyCoord(pk) as V;
      for (let i = 0; i < 512; i++) {
        const T = add(mul(P, 8), [i & 7, (i >> 3) & 7, i >> 6]);
        if (hash(T[0], T[1], T[2], 40) < 0.1) fcell[F.fcell.DIL4 + 512 * cs + i] = (FTF.CROSSING << 24) >>> 0;
      }
    }
    // ---- two COLD pages: (1, 0, 1) on the floor away from the column, (0, 1, 1) in the air above the surface.
    const dir = new Uint32Array(s0.dir), coldSlots = fx.cold.map(k => s0.pages.get(k)).filter(cs => cs !== undefined);
    if (fx.full) assert.equal(coldSlots.length, fx.cold.length, "the COLD pages are resident");
    for (const cs of coldSlots) dir[L.dir.crec + 32 * cs + REC.CLASS]! |= F_CLASS.COLD;
    // ---- φ: analytic on owned vertices, NaN payloads on every other copy; u* = U on every stored face.
    const phiWords = new Uint32Array(await e.read("phi")), W = new World(L, dir, set, fcell);
    W.phi = new Float32Array(phiWords.buffer);
    const put = (kind: "F" | "C", ch: string, slot: number, l: number, x: number | undefined) => { const i = L.base("phi", kind, ch) + BRICK_STRIDE * slot + l; if (x === undefined) phiWords[i] = POISON; else W.phi[i] = x; };
    let poisonF = 0, poisonC = 0, poisonCIn = 0;
    for (const [bk, fs] of s0.bricks) {
      const b = brickKeyCoord(bk) as V;
      for (let l = 0; l < 512; l++) {
        const v = add(mul(b, 8), [l & 7, (l >> 3) & 7, l >> 6]), own = v.every((x, a) => x <= NL[a]!) && W.vertexCoarse(v) === undefined;
        if (!own) poisonF++;
        put("F", "PHI", fs, l, own ? phi0(v) : undefined); put("F", "PHI1", fs, l, own ? phi1(v) : undefined);
      }
    }
    for (const [pk, cs] of s0.pages) {
      const P = brickKeyCoord(pk) as V;
      for (let l = 0; l < 512; l++) {
        const v = add(mul(P, 32), mul([l & 7, (l >> 3) & 7, l >> 6], 4)), own = W.cOwned(v);
        if (!own) { poisonC++; if (v.every((x, a) => x <= NL[a]!)) poisonCIn++; }
        put("C", "PHI", cs, l, own ? phi0(v) : undefined); put("C", "PHI1", cs, l, own ? phi1(v) : undefined);
      }
    }
    const velWords = new Uint32Array(await e.read("vel")), vel = new Float32Array(velWords.buffer);
    for (let a = 0; a < 3; a++) for (const kind of ["F", "C"] as const) {
      const n = (kind === "F" ? L.cap.f : L.cap.c) * BRICK_STRIDE, at = L.base("vel", kind, `US${a}`);
      vel.fill(f32(U[a]!), at, at + n);
    }
    e.write("phi", 0, phiWords); e.write("vel", 0, velWords); e.write("fcell", 0, fcell); e.write("set", 0, set); e.write("dir", 0, dir);
    const kernels = levelSetCoarseKernels(L, F), listC = { indirect: 12 * LIST.C_ALL }, listX8 = { indirect: 12 * LIST.C_ALL + ARGS[LC_ADVECT_GROUPS] };
    // Lane F's uCache (488c4ebf): fgSampleVel reads the caches it builds from the opening velocity, as the engine runs
    // it before lcAdvect every step. The fixture's velocity never changes, so one pass serves the whole chain; the
    // caches are fcell words, so "fcell untouched" below is against fcell after it.
    const uCache = velocityKernels(L, F).uCache;
    assert.ok(uCache, "tiered velocity kernels define uCache");
    await e.run([{ kernel: uCache, dispatch: { indirect: ARGS.FC_G1 } }, { kernel: velocityKernels(L, F).uCache4!, dispatch: { indirect: 12 * LIST.C_ALL } }]);
    const fcellCached = new Uint32Array(await e.read("fcell"));
    console.log(`levelset-coarse: ${s0.bricks.size} F, ${s0.pages.size} C pages, ${lanes(W, slots).length} C vertices, COLD pages ${coldSlots.map(cs => brickKeyCoord(dir[L.dir.crec + 32 * cs + REC.KEY]!).join(",")).join(" ")}; poisoned ${poisonF} F and ${poisonC} C copies (${poisonCIn} of them lattice vertices F owns)`);

    // ---- one kernel: run, then the φ pool differs from before only at the expected C words, by the oracle's values,
    // and the set only at the words the oracle names (lcClear's gate words).
    const words = (a: Uint32Array, b: Uint32Array) => { let n = 0; for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) n++; return n; };
    const step = async (name: keyof typeof kernels, ch: "PHI" | "PHI1", oracle: (W: World, tally: Tally, setWant: Map<number, number>) => Map<string, number>, tol: (x: number) => number) => {
      const before = new Uint32Array(await e.read("phi")), setBefore = new Uint32Array(await e.read("set"));
      W.phi = new Float32Array(before.slice().buffer); W.set = setBefore;
      const tally: Tally = {}, setWant = new Map<number, number>(), want = oracle(W, tally, setWant);
      await e.run([{ kernel: kernels[name], dispatch: LC_GROUPS[name] === "X8" ? listX8 : listC }]);
      const after = await e.read("phi"), af = new Float32Array(after.buffer), setAfter = new Uint32Array(await e.read("set"));
      assert.equal(setAfter[H.FAIL], 0, `${name}: FAIL ${setAfter[H.FAIL]} info ${setAfter[H.FAIL_INFO]}`);
      let setBad = 0, gateBits = 0; const setReport: string[] = [];
      for (let i = 0; i < setAfter.length; i++) {
        const x = setWant.get(i) ?? setBefore[i]!;
        if (setWant.has(i)) for (let b = x; b; b &= b - 1) gateBits++;
        if (setAfter[i] !== x) { setBad++; if (setReport.length < 8) setReport.push(`set word ${i}: ${setBefore[i]} -> gpu ${setAfter[i]} oracle ${x}`); }
      }
      assert.equal(setBad, 0, `${name}: set words off the oracle\n${setReport.join("\n")}`);
      const expected = new Map<number, number>();
      for (const [k, x] of want) { const [cs, l] = k.split(":").map(Number) as [number, number]; expected.set(L.base("phi", "C", ch) + BRICK_STRIDE * cs + l, x); }
      let stray = 0, bad = 0, worst = 0; const report: string[] = [];
      for (let i = 0; i < after.length; i++) {
        const x = expected.get(i);
        if (x === undefined) { if (after[i] !== before[i]) { stray++; if (report.length < 8) report.push(`stray write at ${i}: ${before[i]} -> ${after[i]}`); } continue; }
        const err = Math.abs(af[i]! - x);
        if (!(err <= tol(x))) { bad++; if (report.length < 8) report.push(`word ${i}: gpu ${af[i]} oracle ${x}`); }
        else worst = Math.max(worst, err);
      }
      console.log(`${name}: ${want.size} C vertices${setWant.size ? `, ${gateBits} gate bits` : ""}, worst |err| ${worst.toExponential(2)}; ${JSON.stringify(tally)}`);
      assert.equal(stray, 0, `${name}: writes outside its C vertices\n${report.join("\n")}`);
      assert.equal(bad, 0, `${name}: ${bad} vertices off the oracle\n${report.join("\n")}`);
      return tally;
    };
    const rel = (k: number) => (x: number) => k * Math.max(1, Math.abs(x));
    const fatal = { n: 0 }, sum = (...ts: Tally[]) => { const o: Tally = {}; for (const t of ts) for (const [k, x] of Object.entries(t)) o[k] = (o[k] ?? 0) + x; return o; };
    const cover = (name: string, t: Tally, keys: string[]) => { if (fx.full) for (const k of keys) assert.ok((t[k] ?? 0) > 0, `${name} path ${k} covered`); };
    // The chain: the gate is clear until the first lcClear (zeroed at setClaimPages), which writes it for advect.
    const tL0 = await step("lcClear", "PHI", (W, t, sw) => clearOracle(W, slots, t, sw), rel(0));
    const tA = await step("lcAdvect", "PHI1", (W, t) => advectOracle(W, slots, t, fatal), rel(2e-4));
    const tC = await step("lcCouple", "PHI", (W, t) => coupleOracle(W, slots, t), rel(2e-4));
    const tL = await step("lcClear", "PHI", (W, t, sw) => clearOracle(W, slots, t, sw), rel(0));
    const tR = await step("lcRetire", "PHI1", (W, t) => retireOracle(W, slots, t), rel(2e-4));
    // lcPublish: PHI takes PHI1 at every C vertex.
    await step("lcPublish", "PHI", W => {
      const m = new Map<string, number>();
      for (const { cs, l } of lanes(W, slots)) m.set(`${cs}:${l}`, W.phi[L.base("phi", "C", "PHI1") + BRICK_STRIDE * cs + l]!);
      return m;
    }, rel(0));
    // The next step's advect reads the gate the last lcClear wrote.
    const tA2 = await step("lcAdvect", "PHI1", (W, t) => advectOracle(W, slots, t, fatal), rel(2e-4));
    assert.equal(fatal.n, 0);
    cover("advect", sum(tA, tA2), ["cold hold", "far air", "full path", "wall full path", "foot", "smoothing", "cubic", "wall contact"]);
    cover("couple", tC, ["cold", "no heat", "bulk", "union", "union adds", "erasure", "erasure refused", "guard unknown"]);
    cover("clear", sum(tL0, tL), ["raised", "seed", "collar", "cold"]);
    cover("retire", tR, ["cold", "no h stencil", "hot", "rebuilt", "crossing cell", "newton", "sentinel"]);
    // ---- nothing else moved.
    assert.equal(words(await e.read("vel"), velWords), 0, "vel untouched");
    // Save lcClear's re-resolve stamps (FC_RES's C words: fluid-grid.wgsl.ts), a stamp only on a page it changed.
    { const fc = await e.read("fcell"), RES = F.fcell.RES;
      for (let cs = 0; cs < L.cap.c; cs++) if (fc[RES + cs] !== fcellCached[RES + cs]) { assert.ok(slots.includes(cs), `stamp on C slot ${cs}, no fixture page`); fc[RES + cs] = fcellCached[RES + cs]!; }
      assert.equal(words(fc, fcellCached), 0, "fcell untouched"); }
    assert.equal(words(await e.read("dir"), dir), 0, "dir untouched");
    if (!fx.full) return { poisonCIn };
    // ---- the BULK16 latch (deviation 2): a BULK16 page word beside the pages (the skirt page (2, 0, 0)) fails lcAdvect.
    const skirt = L.dir.page + L.pageIndex(2, 0, 0);
    assert.equal(dir[skirt], NONE, "the skirt page is absent");
    e.write("dir", skirt, new Uint32Array([0xa0000000]));
    await e.run([{ kernel: kernels.lcAdvect, dispatch: listX8 }]);
    const header = new Uint32Array(await e.read("set", 0, 256));
    assert.equal(header[H.FAIL]! & FAIL.UNSUPPORTED, FAIL.UNSUPPORTED, `the BULK16 latch: FAIL ${header[H.FAIL]}`);
    assert.equal(header[H.FAIL_INFO], 0x4c430003, "the BULK16 latch's info");
    console.log(`levelset-coarse: BULK16 latch FAIL ${header[H.FAIL]} info 0x${header[H.FAIL_INFO]!.toString(16)}`);
    return { poisonCIn };
}

test("levelset-coarse (GPU): advect, couple, clear, retire and publish of the 4h corners match the CPU oracle", GPU, async () => {
  const { withUniformDevice } = await import("../../../tests/helpers/uniform-geometric");
  await withUniformDevice("levelset coarse", async device => {
    await runFixture(device, { hi: [30.3, 40.3, 30.3], cold: [brickKey(1, 0, 1), brickKey(0, 1, 1)], full: true });
  });
});

/** brick-set-tiers.test.ts test 1's block A: h tiles x 0..5, y 4..5, z 0..5, inside 4h pages. Its interior corners
 * (and those on the x = 0, z = 0 walls) are C positions F owns. */
const blockA = (T: V) => T[0] <= 5 && T[1] >= 4 && T[1] <= 5 && T[2] <= 5;
test("levelset-coarse (GPU): mixed ownership (h tiles inside 4h pages): F-owned C copies are never read or written", GPU, async () => {
  const { withUniformDevice } = await import("../../../tests/helpers/uniform-geometric");
  await withUniformDevice("levelset coarse mixed", async device => {
    const { poisonCIn } = await runFixture(device, { hi: [N + 1, 20.3, N + 1], want: blockA, cold: [brickKey(1, 0, 1)], full: false });
    // CMASK = WANT: block A's 6 x 6 corners at y = 20 (tile corners x, z 0..5) have no incident 4h tile.
    assert.equal(poisonCIn, 36, "the F-owned C positions are block A's corners");
  });
});
