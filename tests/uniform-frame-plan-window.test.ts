import assert from "node:assert/strict";
import test from "node:test";
import { UniformMixedFramePlan } from "../lib/methods/uniform/uniform-mixed-frame-plan";
import {
  FRAME_PLAN_CERTIFICATE_EXTENT, FRAME_PLAN_REACH_SATURATED, framePlanMaxReach, FRAME_PLAN_SUPPORT_EXTENT, FRAME_PLAN_WINDOW, framePlanPass, framePlanWindowGroups,
} from "../lib/methods/uniform/uniform-frame-plan-window";
import { uniformNearSettled } from "../lib/methods/uniform/uniform-near-settled";
import { UNIFORM_STAGE_CERTIFICATE } from "../lib/methods/uniform/uniform-stage-grids";

// framePlanPass "window" against "chain", without a GPU: a CPU mirror of each
// kernel's WGSL (uniform-mixed-frame-plan.ts dilate/spread/coarseBits/
// coarseBlocks/coarseAny/certify and uniform-frame-plan-window.ts), lane by
// lane in barrier order, over sparse lattices of stored, AIR and SOLID pages
// with mixed h/4h tiles. Then the encoders' launches on a recording pass.

type V3 = [number, number, number];
const SAT = 31, AIR_WORD = 0, SOLID_WORD = 0xffffffff, FREE = 0xffffffff;
const packReach = (p: V3, m: V3) => (p[0] | (p[1] << 5) | (p[2] << 10) | (m[0] << 15) | (m[1] << 20) | (m[2] << 25)) >>> 0;
const plus = (w: number): V3 => [w & 31, (w >>> 5) & 31, (w >>> 10) & 31];
const minus = (w: number): V3 => [(w >>> 15) & 31, (w >>> 20) & 31, (w >>> 25) & 31];
const maxReach = (a: number, b: number) => { const pa = plus(a), pb = plus(b), ma = minus(a), mb = minus(b); return packReach([0, 1, 2].map(i => Math.max(pa[i]!, pb[i]!)) as V3, [0, 1, 2].map(i => Math.max(ma[i]!, mb[i]!)) as V3); };
function rng(seed: number) { let s = seed >>> 0 || 1; return () => { s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0; return s / 4294967296; }; }

/** A sparse lattice of G^3 pages: each stored page its own block (1..n),
 * the rest AIR or SOLID; the resident list the stored blocks, shuffled. */
interface Lattice {
  G: number; last: number; keys: number; blockPage: number[][]; pageWord: (p: V3) => number; resident: number[];
  width: Uint8Array; seed: Uint32Array; reach: Uint32Array;
}
function lattice(seed: number, G: number): Lattice {
  const r = rng(seed), words = new Map<string, number>(), blockPage: number[][] = [[]];
  for (let z = 0; z < G; z++) for (let y = 0; y < G; y++) for (let x = 0; x < G; x++) {
    const u = r(); let w: number;
    if (u < 0.62) { w = blockPage.length; blockPage.push([x, y, z]); } else w = u < 0.72 ? AIR_WORD : SOLID_WORD;
    words.set(`${x},${y},${z}`, w);
  }
  const blocks = blockPage.length + 1, keys = 512 * blocks;
  // Past the lattice: AIR, so a spread that failed to clamp would saturate.
  const pageWord = (p: V3) => words.get(`${p[0]},${p[1]},${p[2]}`) ?? AIR_WORD;
  const resident = Array.from({ length: blockPage.length - 1 }, (_, i) => i + 1);
  for (let i = resident.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [resident[i], resident[j]] = [resident[j]!, resident[i]!]; }
  const width = new Uint8Array(keys), seedFlags = new Uint32Array(keys), reach = new Uint32Array(keys);
  for (const b of resident) {
    const coarsePage = r() < 0.2, quiet = r() < 0.3;
    for (let l = 0; l < 512; l++) {
      const k = 512 * b + l;
      width[k] = coarsePage ? (r() < 0.8 ? 4 : 1) : (r() < 0.003 ? 4 : 1);
      seedFlags[k] = quiet ? 0 : r() < 0.15 ? (r() < 0.5 ? 3 : 1) : 0;
      const f = () => (r() < 0.002 ? SAT : Math.floor(r() * 4));
      reach[k] = packReach([f(), f(), f()], [f(), f(), f()]);
    }
  }
  return { G, last: 8 * G - 1, keys, blockPage, pageWord, resident, width, seed: seedFlags, reach };
}
const pageOf = (p: V3): V3 => [p[0] >> 3, p[1] >> 3, p[2] >> 3];
const stored = (L: Lattice, b: number) => b >= 1 && b <= L.blockPage.length - 1;
const tileKey = (L: Lattice, p: V3) => { const b = L.pageWord(pageOf(p)); return stored(L, b) ? 512 * b + ((p[0] & 7) | ((p[1] & 7) << 3) | ((p[2] & 7) << 6)) : L.keys; };
const keyStored = (L: Lattice, k: number) => k < L.keys;
const tileCoord = (L: Lattice, k: number): V3 => { const P = L.blockPage[k >> 9]!; return [8 * P[0]! + (k & 7), 8 * P[1]! + ((k >> 3) & 7), 8 * P[2]! + ((k >> 6) & 7)]; };
const tileLive = (L: Lattice, k: number) => k < L.keys && stored(L, k >> 9);
const isAir = (L: Lattice, p: V3) => L.pageWord(pageOf(p)) === AIR_WORD;
const radiusOf = (speedTravel: number) => Math.min(Math.ceil(speedTravel * 1.00001 / 4) + 2, 1e8);

interface Policy { fine: number; shell: number; mode: number; surfaceOnly: boolean }
const shellRule = (q: Policy, source: number, d: number) => q.surfaceOnly ? (Math.abs(d) <= q.fine ? source & 3 : (source & 1) << 1) : source & (Math.abs(d) <= q.fine ? 3 : 2);

// ---- chain: dilate0..2 (planes 0 -> 1 -> 2 -> 3), as uniform-mixed-frame-plan's dilate<axis>Tile.
function chainSupport(L: Lattice, q: Policy): Uint32Array {
  const planes = [Uint32Array.from(L.seed), new Uint32Array(L.keys), new Uint32Array(L.keys), new Uint32Array(L.keys)];
  for (let axis = 0; axis < 3; axis++) for (const b of L.resident) for (let l = 0; l < 512; l++) {
    const tile = 512 * b + l; if (!tileLive(L, tile)) continue;
    const p = tileCoord(L, tile); let flags = 0; const reach = q.fine + q.shell;
    for (let d = -reach; d <= reach; d++) { const r = [...p] as V3; r[axis] += d; const k = tileKey(L, r); if (!keyStored(L, k)) continue; flags |= shellRule(q, planes[axis]![k]!, d); }
    planes[axis + 1]![tile] = flags | q.mode;
  }
  return planes[3]!;
}
// ---- window: supportWindow, lane by lane.
function windowSupport(L: Lattice, q: Policy): Uint32Array {
  const out = new Uint32Array(L.keys), [sx, sy, sz] = FRAME_PLAN_SUPPORT_EXTENT, hs = FRAME_PLAN_WINDOW.supportHalo, STORED = 0x80000000;
  const fine = q.fine, reach = q.fine + q.shell; assert.ok(reach <= hs);
  for (let item = 0; item < 2 * L.resident.length; item++) {
    const b = L.resident[item >> 1]!, P = L.blockPage[b]!, low: V3 = [8 * P[0]! - hs, 8 * P[1]! - hs, 8 * P[2]! + 4 * (item & 1) - hs];
    const win = new Uint32Array(sx * sy * sz);
    for (let i = 0; i < win.length; i++) { const k = tileKey(L, [low[0] + (i % sx), low[1] + (Math.floor(i / sx) % sy), low[2] + Math.floor(i / (sx * sy))]); win[i] = keyStored(L, k) ? (STORED | L.seed[k]!) >>> 0 : 0; }
    const dilate = (first: number, stride: number, count: number) => {
      const o: number[] = [];
      for (let c = 0; c < count; c++) { const at = first + c * stride; let flags = 0; for (let d = -reach; d <= reach; d++) { const s = win[at + d * stride]!; if ((s & STORED) === 0) continue; flags |= shellRule({ ...q, fine }, s, d); } o.push(flags | q.mode); }
      for (let c = 0; c < count; c++) { const at = first + c * stride; if ((win[at]! & STORED) !== 0) win[at] = (STORED | o[c]!) >>> 0; }
    };
    for (let lane = 0; lane < sy * sz; lane++) dilate(sx * lane + hs, 1, 8);
    for (let lane = 0; lane < 8 * sz; lane++) dilate(hs + (lane & 7) + sx * (hs + sy * (lane >> 3)), sx, 8);
    for (let lane = 0; lane < 64; lane++) dilate(hs + (lane & 7) + sx * (hs + (lane >> 3)) + sx * sy * hs, sx * sy, 4);
    for (let lane = 0; lane < FRAME_PLAN_WINDOW.lanes; lane++) {
      const tile = 512 * b + 256 * (item & 1) + lane, at = hs + (lane & 7) + sx * (hs + ((lane >> 3) & 7) + sy * (hs + (lane >> 6)));
      if (tileLive(L, tile)) out[tile] = win[at]! & 3;
    }
  }
  return out;
}

// ---- the certificate: tile bits and coarseAny (uniformTileBitsWGSL, air:true), shared by both.
const NEIGHBOUR = (i: number): V3 => [(i % 3) - 1, (Math.floor(i / 3) % 3) - 1, Math.floor(i / 9) - 1];
function coarseAny(L: Lattice, coarse: Uint32Array, home: number, lo: V3, hi: V3): boolean {
  const word = (b: number, i: number) => coarse[18 * b + i]!;
  const coarseIn = (blk: number, low: V3, high: V3) => {
    const slabs = word(blk, 16); if (slabs === 0) return false;
    const a = low.map(v => Math.max(v, 0)), b = high.map(v => Math.min(v, 7));
    const row = (((2 << (b[0]! - a[0]!)) - 1) << a[0]!) >>> 0; const mask = [0, 0];
    for (let y = a[1]!; y <= b[1]!; y++) mask[y >> 2] = (mask[y >> 2]! | (row << (8 * (y & 3)))) >>> 0;
    for (let z = a[2]!; z <= b[2]!; z++) { if (((slabs >>> z) & 1) === 0) continue; if (((word(blk, 2 * z) & mask[0]!) | (word(blk, 2 * z + 1) & mask[1]!)) !== 0) return true; }
    return false;
  };
  const a = lo.map(v => Math.max(v, 0)) as V3, b = hi.map(v => Math.min(v, L.last)) as V3;
  if (b.some((v, i) => v < a[i]!)) return false;
  const pa = pageOf(a), pb = pageOf(b), hp = L.blockPage[home]! as V3;
  const near = pa.every((v, i) => v >= hp[i]! - 1) && pb.every((v, i) => v <= hp[i]! + 1);
  let around = 0; if (near) { around = word(home, 17); if (around === 0) return false; }
  const n = pb.map((v, i) => v - pa[i]! + 1); if (n[0]! * n[1]! * n[2]! > 64) return true;
  for (let z = pa[2]; z <= pb[2]; z++) for (let y = pa[1]; y <= pb[1]; y++) for (let x = pa[0]; x <= pb[0]; x++) {
    const qp: V3 = [x, y, z]; let w: number;
    if (near) { const i = (x - hp[0] + 1) + 3 * (y - hp[1] + 1) + 9 * (z - hp[2] + 1); if (((around >>> i) & 1) === 0) continue; w = L.pageWord(qp); } else w = L.pageWord(qp);
    if (w === 0) return true; if (!stored(L, w)) continue;
    if (coarseIn(w, a.map((v, i) => v - 8 * qp[i]!) as V3, b.map((v, i) => v - 8 * qp[i]!) as V3)) return true;
  }
  return false;
}
function chainCoarse(L: Lattice): Uint32Array {
  const coarse = new Uint32Array(18 * (L.blockPage.length + 1));
  for (const b of L.resident) for (let l = 0; l < 512; l++) if (L.width[512 * b + l] !== 1) coarse[18 * b + (l >> 5)] = (coarse[18 * b + (l >> 5)]! | (1 << (l & 31))) >>> 0;
  for (const b of L.resident) {
    let slabs = 0; for (let z = 0; z < 8; z++) if ((coarse[18 * b + 2 * z]! | coarse[18 * b + 2 * z + 1]!) !== 0) slabs |= 1 << z;
    let around = 0; const P = L.blockPage[b]! as V3;
    for (let i = 0; i < 27; i++) {
      const o = NEIGHBOUR(i), w = L.pageWord([P[0] + o[0], P[1] + o[1], P[2] + o[2]]);
      if (w === 0) { around |= 1 << i; continue; } if (!stored(L, w)) continue;
      for (let j = 0; j < 16; j++) if (coarse[18 * w + j] !== 0) { around |= 1 << i; break; }
    }
    coarse[18 * b + 16] = slabs; coarse[18 * b + 17] = around >>> 0;
  }
  return coarse;
}
interface Certificate { reach: Uint32Array; classed: Uint32Array; regular: Set<number>; general: Set<number>; coarse: Uint32Array }
function certifyTile(L: Lattice, coarse: Uint32Array, tile: number, reach: number, c: Certificate) {
  const p = plus(reach), m = minus(reach), bounded = p.every(v => v < SAT) && m.every(v => v < SAT), at = tileCoord(L, tile);
  const regular = bounded && !coarseAny(L, coarse, tile >> 9, [at[0] - m[0], at[1] - m[1], at[2] - m[2]], [at[0] + p[0], at[1] + p[1], at[2] + p[2]]);
  (regular ? c.regular : c.general).add(tile);
  c.classed[tile] = (reach | ((regular ? UNIFORM_STAGE_CERTIFICATE.regular : bounded ? UNIFORM_STAGE_CERTIFICATE.coarseInReach : UNIFORM_STAGE_CERTIFICATE.saturated) << 30)) >>> 0;
}
// spread0..2 (speeds 0 -> T -> 0 -> T), coarseBits, coarseBlocks, certify.
function chainCertificate(L: Lattice, radius: number): Certificate {
  let source = Uint32Array.from(L.reach);
  for (let axis = 0; axis < 3; axis++) {
    const into = new Uint32Array(L.keys);
    for (const b of L.resident) for (let l = 0; l < 512; l++) {
      const tile = 512 * b + l, p = tileCoord(L, tile); let w = 0;
      for (let qz = Math.max(0, p[axis]! - radius); qz <= Math.min(L.last, p[axis]! + radius); qz++) {
        const r = [...p] as V3; r[axis] = qz; const k = tileKey(L, r);
        if (!keyStored(L, k)) { if (isAir(L, r)) w = FRAME_PLAN_REACH_SATURATED; continue; }
        w = maxReach(w, source[k]!);
      }
      into[tile] = w;
    }
    source = into;
  }
  const coarse = chainCoarse(L), c: Certificate = { reach: source, classed: Uint32Array.from(source), regular: new Set(), general: new Set(), coarse };
  for (const b of L.resident) for (let l = 0; l < 512; l++) if (L.width[512 * b + l] === 1) certifyTile(L, coarse, 512 * b + l, source[512 * b + l]!, c);
  return c;
}
// certifySpread then certifyTiles, lane by lane.
function windowCertificate(L: Lattice, radius: number): Certificate {
  const [cx, cy, cz] = FRAME_PLAN_CERTIFICATE_EXTENT, hc = FRAME_PLAN_WINDOW.certificateHalo, STORED = 0x80000000;
  const reach = Math.min(radius, hc), wide = radius > hc, plane1 = new Uint32Array(L.keys), coarse = new Uint32Array(18 * (L.blockPage.length + 1));
  for (let item = 0; item < 2 * L.resident.length; item++) {
    const b = L.resident[item >> 1]!, P = L.blockPage[b]!, low: V3 = [8 * P[0]! - hc, 8 * P[1]! - hc, 8 * P[2]! + 4 * (item & 1)];
    const win = new Uint32Array(cx * cy * cz), bits = new Uint32Array(8);
    for (let i = 0; i < win.length; i++) {
      const p: V3 = [low[0] + (i % cx), low[1] + (Math.floor(i / cx) % cy), low[2] + Math.floor(i / (cx * cy))]; let w = 0;
      if (p.every(v => v >= 0 && v <= L.last)) { const k = tileKey(L, p); if (keyStored(L, k)) w = (STORED | L.reach[k]!) >>> 0; else if (isAir(L, p)) w = FRAME_PLAN_REACH_SATURATED; }
      win[i] = w;
    }
    for (let lane = 0; lane < 256; lane++) { const tile = 512 * b + 256 * (item & 1) + lane; if (tileLive(L, tile) && L.width[tile] !== 1) bits[lane >> 5] = (bits[lane >> 5]! | (1 << (lane & 31))) >>> 0; }
    const spread = (first: number, stride: number) => {
      const o: number[] = [];
      for (let c = 0; c < 8; c++) { const at = first + c * stride; let w = 0; for (let d = -reach; d <= reach; d++) w = framePlanMaxReach(w, win[at + d * stride]!); o.push(w); }
      for (let c = 0; c < 8; c++) { const at = first + c * stride; if ((win[at]! & STORED) !== 0) win[at] = (STORED | o[c]!) >>> 0; }
    };
    for (let lane = 0; lane < cy * cz; lane++) spread(cx * lane + hc, 1);
    for (let lane = 0; lane < 32; lane++) spread(hc + (lane & 7) + cx * (hc + cy * (lane >> 3)), cx);
    for (let lane = 0; lane < 256; lane++) {
      const tile = 512 * b + 256 * (item & 1) + lane, at = hc + (lane & 7) + cx * (hc + ((lane >> 3) & 7) + cy * (lane >> 6));
      if (tileLive(L, tile)) plane1[tile] = wide ? FRAME_PLAN_REACH_SATURATED : win[at]! & FRAME_PLAN_REACH_SATURATED;
    }
    for (let lane = 0; lane < 8; lane++) coarse[18 * b + 8 * (item & 1) + lane] = bits[lane]!;
    if ((item & 1) === 0) { coarse[18 * b + 16] = 0xff; coarse[18 * b + 17] = 0x7ffffff; }
  }
  const c: Certificate = { reach: new Uint32Array(L.keys), classed: new Uint32Array(L.keys), regular: new Set(), general: new Set(), coarse };
  for (let item = 0; item < 2 * L.resident.length; item++) for (let lane = 0; lane < 256; lane++) {
    const tile = 512 * L.resident[item >> 1]! + 256 * (item & 1) + lane; if (!tileLive(L, tile)) continue;
    const p = tileCoord(L, tile); let w = 0;
    for (let z = Math.max(0, Math.min(L.last, p[2] - radius)); z <= Math.max(0, Math.min(L.last, p[2] + radius)); z++) {
      const r: V3 = [p[0], p[1], z], k = tileKey(L, r);
      if (!keyStored(L, k)) { if (isAir(L, r)) w = FRAME_PLAN_REACH_SATURATED; continue; }
      w = maxReach(w, plane1[k]!);
    }
    c.reach[tile] = w; c.classed[tile] = w;
    if (L.width[tile] === 1) certifyTile(L, coarse, tile, w, c);
  }
  return c;
}

const CASES = [{ seed: 7, G: 4 }, { seed: 19, G: 5 }, { seed: 4242, G: 3 }, { seed: 90001, G: 4 }];

test("supportWindow reproduces dilate0..2's last support plane on every live tile", () => {
  const policies: Policy[] = [];
  for (const surfaceOnly of [true, false]) for (const [fine, shell] of [[2, 1], [1, 1], [3, 0], [0, 3], [1, 2]] as const) for (const mode of [0, 2, 3]) policies.push({ fine, shell, mode, surfaceOnly });
  for (const c of CASES) {
    const L = lattice(c.seed, c.G);
    for (const q of policies) {
      const chain = chainSupport(L, q), window = windowSupport(L, q);
      let live = 0, set = 0;
      for (const b of L.resident) for (let l = 0; l < 512; l++) { const t = 512 * b + l; live++; if (chain[t]) set++; assert.equal(window[t], chain[t], `seed ${c.seed} ${JSON.stringify(q)} tile ${t}`); }
      assert.ok(live > 0 && (q.mode !== 0 || set > 0));
    }
  }
});

test("certifySpread + certifyTiles reproduce the certificate (reach, class, both lists) up to the halo, and saturate past it", () => {
  let regular = 0, general = 0;
  for (const c of CASES) {
    const L = lattice(c.seed, c.G);
    for (const travel of [0, 3, 9, 20, 30, 32, 33, 60, Infinity]) {
      const radius = radiusOf(travel), chain = chainCertificate(L, radius), window = windowCertificate(L, radius);
      for (let w = 0; w < 18 * L.blockPage.length; w++) if (w % 18 < 16) assert.equal(window.coarse[w], chain.coarse[w], `tile bits word ${w}`);
      if (radius <= FRAME_PLAN_WINDOW.certificateHalo) {
        assert.deepEqual(window.reach, chain.reach, `seed ${c.seed} travel ${travel}: reach`);
        assert.deepEqual(window.classed, chain.classed, `seed ${c.seed} travel ${travel}: classes`);
        assert.deepEqual([...window.regular].sort(), [...chain.regular].sort());
        assert.deepEqual([...window.general].sort(), [...chain.general].sort());
        if (travel === 0) { regular += chain.regular.size; general += chain.general.size; }
      } else {
        // Past the halo every reach is saturated: no tile is regular, and a saturated reach bounds the chain's.
        assert.equal(window.regular.size, 0);
        for (const b of L.resident) for (let l = 0; l < 512; l++) { const t = 512 * b + l; assert.equal(window.reach[t], FRAME_PLAN_REACH_SATURATED); assert.ok(plus(chain.reach[t]!).every(v => v <= SAT)); }
      }
    }
  }
  assert.ok(regular > 500 && general > 500, `the fixtures certify both lists (${regular} regular, ${general} general)`);
});

test("coarseAny answers the same from the window's all-set summaries as from coarseBlocks'", () => {
  for (const c of CASES) {
    const L = lattice(c.seed, c.G), chain = chainCoarse(L), window = windowCertificate(L, 3).coarse, r = rng(c.seed + 1);
    let hits = 0;
    for (let n = 0; n < 4000; n++) {
      const home = L.resident[Math.floor(r() * L.resident.length)]!, P = L.blockPage[home]!;
      const lo = [0, 1, 2].map(i => 8 * P[i]! + Math.floor(r() * 8) - Math.floor(r() * 12)) as V3, hi = lo.map(v => v + Math.floor(r() * 18)) as V3;
      const a = coarseAny(L, chain, home, lo, hi); if (a) hits++;
      assert.equal(coarseAny(L, window, home, lo, hi), a);
    }
    assert.ok(hits > 100 && hits < 3900);
  }
});

// ---- the encoders on a recording pass.
interface Recorded { pass: string; entry: string; groups: number | "counted" }
function recorded(pass: "chain" | "window", blocks: number, encode: (plan: UniformMixedFramePlan, encoder: GPUCommandEncoder) => void): Recorded[] {
  const out: Recorded[] = [], named = (name: string) => ({ name }) as unknown as GPUComputePipeline;
  const plan = Object.create(UniformMixedFramePlan.prototype) as UniformMixedFramePlan;
  const entries = ["seed", "seedHeld", "dilate0", "dilate1", "dilate2", "localSpeed", "tileError", "spread0", "spread1", "spread2", "coarseBits", "coarseBlocks", "certify", ...(pass === "window" ? ["supportWindow", "certifySpread", "certifyTiles"] : [])];
  let current = "";
  Object.assign(plan, {
    device: { queue: { writeBuffer: () => {} } }, pass, surfaceOnly: true, supportRevision: -1, supportSettings: [-1, -1, -1, -1],
    pipelines: new Map(entries.map(e => [e, named(e)])), params: {}, group: { group: {} }, extendedGroup: { group: {} }, heldGroup: {},
    ownership: { capacity: { tiles: 512 * blocks, address: { blocks } }, revision: 0, support: {}, regionVelocity: {}, bindGroup: {}, speeds: {},
      dispatchAllCounted: (_: GPUComputePassEncoder, p: { name: string }) => { out.push({ pass: current, entry: p.name, groups: "counted" }); } },
  });
  const encoder = {
    beginComputePass: (d: GPUComputePassDescriptor) => {
      current = d.label!; let pipeline: { name: string } | undefined;
      return { setBindGroup: () => {}, setPipeline: (p: { name: string }) => { pipeline = p; }, end: () => {},
        dispatchWorkgroups: (x: number, y = 1, z = 1) => { assert.equal(y * z, 1); out.push({ pass: current, entry: pipeline!.name, groups: x }); },
        dispatchWorkgroupsIndirect: () => assert.fail("the plan launches no indirect dispatch") };
    },
    clearBuffer: () => {}, copyBufferToBuffer: () => {},
  } as unknown as GPUCommandEncoder;
  encode(plan, encoder);
  return out;
}
const frame = (plan: UniformMixedFramePlan, e: GPUCommandEncoder) => { plan.encode(e, undefined, true); plan.encodeCertificate(e, 1 / 60); plan.encode(e); };

test("framePlanPass window: 5 launches where the chain takes 12 for a plan and its certificate, none of a single workgroup", () => {
  // Paddleboats: a 288-block pool (192 resident at step 30).
  const blocks = 288;
  const chain = recorded("chain", blocks, frame), window = recorded("window", blocks, frame);
  const names = (r: Recorded[]) => r.map(x => x.entry);
  assert.deepEqual(names(chain), ["seedHeld", "dilate0", "dilate1", "dilate2", "localSpeed", "tileError", "spread0", "spread1", "spread2", "coarseBits", "coarseBlocks", "certify", "seed", "dilate0", "dilate1", "dilate2"]);
  assert.deepEqual(names(window), ["seedHeld", "supportWindow", "localSpeed", "certifySpread", "certifyTiles", "seed", "supportWindow"]);
  // The chain's serial tail: coarseBlocks is a lane per block, 5 workgroups for 288 blocks.
  assert.equal(chain.find(x => x.entry === "coarseBlocks")!.groups, 5);
  for (const x of window) if (x.groups !== "counted") assert.equal(x.groups, Math.min(FRAME_PLAN_WINDOW.maxGroups, 2 * blocks));
  // Small pools: a group an item, never fewer than the items.
  assert.equal(framePlanWindowGroups(3), 6); assert.equal(framePlanWindowGroups(4096), FRAME_PLAN_WINDOW.maxGroups);
  // A support policy past the window's halo takes the chain's dilation.
  const wide = recorded("window", blocks, (plan, e) => plan.encode(e, { fineReach: 3, shellReach: 1, twoLevel: true, shellOnly: true }));
  assert.deepEqual(names(wide), ["seed", "dilate0", "dilate1", "dilate2"]);
});

test("pwMaxReach's packed maximum is the field-wise maximum of umReachPlus/Minus", () => {
  const r = rng(5);
  for (let n = 0; n < 200000; n++) {
    const f = () => Math.floor(r() * 32), a = packReach([f(), f(), f()], [f(), f(), f()]), b = packReach([f(), f(), f()], [f(), f(), f()]);
    assert.equal(framePlanMaxReach((a | (r() < 0.5 ? 0x80000000 : 0)) >>> 0, b), maxReach(a, b));
  }
});

test("framePlanPass parses its values, defaults to the chain, and fails on unknown ones", () => {
  assert.equal(framePlanPass(undefined), "chain"); assert.equal(framePlanPass("window"), "window");
  assert.equal(uniformNearSettled().framePlanPass, "chain"); assert.equal(uniformNearSettled({ framePlanPass: "window" }).framePlanPass, "window");
  for (const bad of ["fused", "", 1, null]) {
    assert.throws(() => framePlanPass(bad), /framePlanPass must be one of chain, window/);
    assert.throws(() => uniformNearSettled({ framePlanPass: bad as string }), /framePlanPass/);
  }
});
