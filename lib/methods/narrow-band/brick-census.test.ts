// setCensus (brick-census.wgsl.ts) on a frozen state against a CPU oracle transcribed from the old census
// (fluid-scratch/lane-f/wgsl-dam/063-Uniform_dynamic_ownership_census.wgsl: classify, classifyCoarse,
// boundCubes, importanceTile, publishImportance, interfaceBits, decideTile) on the brick storage
// (brick-abi.md §2.5), with the residency request (WANT keeps the band; the missing ring bricks are candidates). The oracle runs in f64 over the stored f32 words, rounding
// where the kernel's f32 arithmetic decides a comparison (fr). Exact: velocity keys, crossing cells, every
// flag, trigger and required bit, WANT, TRIG and the candidates. Tolerance-checked (the GPU may fuse a multiply
// and add, and scores are bytes of f32 products): score bytes, within 1; the winner bits of FIN follow the
// bytes and are not compared. TRIG's thin term reads the last certificate's packed copy (DEF, certError's, which
// this frozen state does not hold: no certificate ran), so the test supplies DEF words covering every tuDeformation
// branch (a stale key, width 0, a negative width, valid values either side of the budget). The deep COLD test
// (D = 0) is CPU only.
import assert from "node:assert/strict";
import test from "node:test";
import { BRICK_CELLS, ARGS, BRICK_STRIDE, CMASK_WORDS, H, LIST, NONE, RING, SEAM, bricksPerAxis, phiFar } from "./brick-abi";
import { CENSUS, CENSUS_H, CENSUS_POLICY, REGION_H, RG, brickCensusBuffers, brickCensusSchedule, brickTrigCensusSchedule } from "./brick-census.wgsl";

type V3 = [number, number, number];
const GPU = { skip: !process.env.WEBGPU_NODE_MODULE };
const fr = Math.fround;
const F32 = new Float32Array(1), U32 = new Uint32Array(F32.buffer);
const bitsOf = (x: number) => { F32[0] = x; return U32[0]!; };
const fromBits = (u: number) => { U32[0] = u >>> 0; return F32[0]!; };
const orderKey = (x: number) => { const b = bitsOf(x); return ((b & 0x80000000) !== 0 ? ~b : b | 0x80000000) >>> 0; };
const orderValue = (k: number) => fromBits((k & 0x80000000) !== 0 ? k & 0x7fffffff : ~k);
const corner = (k: number, side: number): V3 => [k % side, Math.floor(k / side) % side, Math.floor(k / (side * side))];
const add = (a: V3, b: V3): V3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const scale3 = (a: V3, s: number): V3 => [a[0] * s, a[1] * s, a[2] * s];
const shr = (a: V3, s: number): V3 => [a[0] >> s, a[1] >> s, a[2] >> s];
const unit = (a: number, s = 1): V3 => [a === 0 ? s : 0, a === 1 ? s : 0, a === 2 ? s : 0];
const key = (b: V3) => ((b[0] & 1023) | ((b[1] & 1023) << 10) | ((b[2] & 1023) << 20)) >>> 0;
const tileIndex = (t: V3) => (t[0] & 7) | ((t[1] & 7) << 3) | ((t[2] & 7) << 6);
const local8 = (c: V3) => (c[0] & 7) | ((c[1] & 7) << 3) | ((c[2] & 7) << 6);
const brickLocal = (b: V3) => (b[0] & 3) | ((b[1] & 3) << 2) | ((b[2] & 3) << 4);
const P = CENSUS_POLICY;
const NO_WALL = 0xffff, CROSSES = 1 << 28, WET = 1 << 29, REQUIRED = 1 << 25, HEAT_SHAPE = 1 << 26, HEAT_THIN = 1 << 27, OPEN_CROSS = 1 << 30;
// C'' / C''': FIN's energetic bit (|Du/Dt| >= |g|/2: an h tile's liquid cells' mean, a 4h tile's own) and its class flag.
const ENERGETIC = 0x80000000, CLASS_ENERGETIC = 16;
const trig = (k: number) => (1 << (16 + k)) >>> 0;
const on = (k: number) => ((P.criteria >> k) & 1) !== 0;
const scoreByte = (s: number) => Math.trunc(Math.min(Math.max(fr(s * 64), 0), 255));

/** The census's inputs as stored: the directory, the masks and the f32 words of PHI and US. */
export interface CensusState {
  readonly N: V3; readonly h: number; readonly dt: number; readonly g: V3;
  readonly pages: Map<number, number>; readonly bricks: Map<number, number>;
  readonly cmask: (cs: number) => Uint32Array;
  readonly fPhi: (fs: number) => Float32Array; readonly cPhi: (cs: number) => Float32Array;
  readonly fU: (a: number, fs: number) => Float32Array; readonly cU: (a: number, cs: number) => Float32Array;
  /** C'': the last step's velocity change (UP, the FLIP delta) of F slot fs and C slot cs along a; absent: 0. */
  readonly fUP?: (a: number, fs: number) => Float32Array; readonly cUP?: (a: number, cs: number) => Float32Array;
  readonly tileOcc: (fs: number) => number;
  /** C_ALL in list order. */ readonly cAll: readonly number[];
  /** The census buffer's DEF words of tile i of C slot cs (the certificate's packed copy); absent: 0 (none written). */
  readonly def?: (cs: number, i: number) => readonly number[];
  /** The last certificate's TC record saw liquid in tile i of C slot cs (bit 0 of TC word 8; R2 reads it); absent: dry. */
  readonly wet?: (cs: number, i: number) => boolean;
  /** R2's hysteresis: tile i of C slot cs was calm at the last tiered head (CALM under a valid CALM_KEY); absent: not. */
  readonly calmPrev?: (cs: number, i: number) => boolean;
  /** The last certificate's TC record (16 words) of tile i of C slot cs (R2 reads the mean velocity, words 0..2, and
   * word 15, the second difference of the tile means); absent: zeros. */
  readonly tc?: (cs: number, i: number) => readonly number[];
  /** FLAGS.OPEN clear in cell c's F slot (a mask voxel, a body); absent: every stored cell open. */
  readonly closed?: (c: V3) => boolean;
}
/** The six-tetrahedron fill of a cell's 8 vertex φ (lib/core/level-set-fill.wgsl.ts), Float64. */
function cellFill(v: readonly number[]): number {
  const tetra = (q: number[]) => {
    if (q.every(x => x >= 0)) return 0; if (q.every(x => x <= 0)) return 1;
    const n = q.filter(x => x < 0).length, s = [...q.filter(x => x < 0), ...q.filter(x => !(x < 0))] as [number, number, number, number];
    const [x, y, z, w] = s;
    if (n === 1) return (-x / (y - x)) * (-x / (z - x)) * (-x / (w - x));
    if (n === 3) return 1 - (w / (w - x)) * (w / (w - y)) * (w / (w - z));
    const a = -x / (z - x), b = -x / (w - x), c = -y / (z - y), d = -y / (w - y);
    return Math.min(1, Math.max(0, a * b + b * c * (1 - a) + c * d * (1 - b)));
  };
  return (tetra([v[0]!, v[1]!, v[3]!, v[7]!]) + tetra([v[0]!, v[1]!, v[5]!, v[7]!]) + tetra([v[0]!, v[2]!, v[3]!, v[7]!])
    + tetra([v[0]!, v[2]!, v[6]!, v[7]!]) + tetra([v[0]!, v[4]!, v[5]!, v[7]!]) + tetra([v[0]!, v[4]!, v[6]!, v[7]!])) / 6;
}
interface Class { low: number[]; high: number[]; gap: number[]; reach: number[]; flags: number; error: number; crossing: [number, number]; deep: [number, number] }
const emptyClass = (): Class => ({ low: [0xffffffff, 0xffffffff, 0xffffffff], high: [0, 0, 0], gap: [15, 15, 15, 15, 15, 15], reach: [15, 15, 15, 15, 15, 15], flags: 0, error: 0, crossing: [0, 0], deep: [0, 0] });
const identityKeys = () => [0xffffffff, 0xffffffff, 0xffffffff, 0, 0, 0];
interface Scratch { keys: number[][]; w0: number[]; w1: number[]; fin: number[]; cross: [number, number][]; gap: number[]; req: number[]; cubes: number[][] }
export interface CensusResult {
  scratch: Map<number, Scratch>; want: Map<number, Uint32Array>; trig: Map<number, number>; cand: Set<number>;
  /** Thin-alone tiles of F bricks (FIN's criteria exactly the thin bit), and those the need rule heats. */ thinAlone: { tiles: number; heated: number };
  /** tuDeformation's branches on the thin-alone tiles: stale key, width <= 0, valid and heated, valid and not. */ defBranches: Record<"key" | "width" | "heated" | "cool", number>;
  band: Map<number, boolean>; demoted: Set<number>; calmNow: Set<number>; held: Set<number>;
  /** Calm band tiles R2's shape and thin guard kept (a shape or thin trigger in FIN). */ shaped: Set<number>;
  /** Calm band h tiles R2's curvature test kept (shift > 0.08), the VOL word (the packed mean normal) of every h tile at
   * a head, and the curvature shift of every band h tile tested. */ volumeKept: Set<number>; vol: Map<number, number>; shift: Map<number, number>;
  /** Band tiles calm on velocity and curvature that R2's importance (ncSurfaceQuiet) kept. */ loud: Set<number>;
  counters: Record<keyof typeof CENSUS_H, number>; failed: number;
}

/** The oracle. deep: tiles taking the last census's decision (the old deep COLD branch, l.2852), for the D test.
 * head: a tiered step head with the real D, where R2 (lane E, a stated deviation) makes a calm band tile 4h:
 * dt e / h <= 0.08 on the DEF error (an h tile: and TC word 15) of a fast tile (2 ncKeySpeed >= 4, ncFastTile), on TC word 15 alone of a slow one,
 * under valid DEF words (ncCalm; an unknown copy is never calm, nor a dry one), with no thin trigger and no shape trigger on a fast tile, an h
 * tile only at its second consecutive calm head (calmPrev; the hysteresis), a 4h tile at once; an h tile only where its
 * 4h-scale curvature shift (4/3) h div n of the tile mean normals is <= 0.08 (ncCurveShift; VOL the packed normal); and
 * only where the surface's normal speed about the tile, dt |u.n| / 4h, is <= 0.08 (ncSurfaceQuiet: its own, and each
 * face neighbour's whose surface runs on into it or moves towards it). */
export function censusOracle(s: CensusState, deep?: { has: (t: V3) => boolean; decided: (t: V3) => boolean }, head = false): CensusResult {
  const { N, h, dt } = s;
  const pageKey = (q: V3) => key(q);
  const pageIn = (q: V3) => q.every((v, a) => v >= 0 && 32 * v < N[a]!);
  const pageSlot = (q: V3) => (pageIn(q) ? s.pages.get(pageKey(q)) : undefined);
  const anySlot = (q: V3) => (q.some(v => v < 0) ? undefined : s.pages.get(pageKey(q)));
  const inLattice = (c: V3) => c.every((v, a) => v >= 0 && v < N[a]!);
  const tileInLattice = (t: V3) => t.every((v, a) => v >= 0 && 4 * v < N[a]!);
  const bit = (m: Uint32Array, i: number) => ((m[i >> 5]! >>> (i & 31)) & 1) === 1;
  const tileH = (t: V3) => { if (!tileInLattice(t)) return false; const cs = s.pages.get(pageKey(shr(t, 3))); return cs !== undefined && bit(s.cmask(cs), tileIndex(t)); };
  const cellH = (c: V3) => tileH(shr(c, 2));
  const brickRange = bricksPerAxis(N);
  const fSlot = (b: V3) => { if (b.some(v => v < 0)) return undefined; const cs = s.pages.get(pageKey(shr(b, 2))); if (cs === undefined) return undefined; const fs = s.bricks.get(key(b)); return fs; };
  const fVertex = (v: V3) => { const fs = fSlot(shr(v, 3)); return fs === undefined ? FAR : s.fPhi(fs)[local8(v)]!; };
  const cVertex = (T: V3) => { if (T.some((x, a) => x < 0 || 4 * x > N[a]!)) return FAR; const cs = anySlot(shr(T, 3)); return cs === undefined ? FAR : s.cPhi(cs)[tileIndex(T)]!; };
  // brick-owner's nbVertexCoarse: a tile corner an h tile touches is F's where an F slot stores it (lcMirror keeps
  // its C word F's copy); elsewhere nbVertexRead4, coarse wins.
  const fineCorner = (v: V3) => v.every(x => x % 4 === 0) && fSlot(shr(v, 3)) !== undefined
    && Array.from({ length: 8 }, (_, k) => v.map((x, a) => (x >> 2) - ((k >> a) & 1)) as V3).some(tileH);
  const vertexCoarse = (v: V3): V3 | undefined => {
    if (fineCorner(v)) return undefined;
    const tA = shr(v, 2), inside = v.map((x, a) => x - 4 * tA[a]! !== 0);
    for (let k = 0; k < 8; k++) {
      const c = [k & 1, (k >> 1) & 1, k >> 2];
      if (c.some((x, a) => inside[a] && x === 0)) continue;
      const t = tA.map((x, a) => x + c[a]! - 1) as V3;
      if (tileInLattice(t) && s.pages.get(pageKey(shr(t, 3))) !== undefined && !tileH(t)) return t; // a non-resident page's tile is no authority
    }
    return undefined;
  };
  const vertexPhi = (v: V3) => {
    if (v.some((x, a) => x < 0 || x > N[a]!)) return FAR;
    const T = vertexCoarse(v);
    if (!T) return fVertex(v);
    const t = v.map((x, a) => fr((x - 4 * T[a]!) / 4)), vals = new Array<number>(8).fill(0);
    for (let k = 0; k < 8; k++) {
      const c = [k & 1, (k >> 1) & 1, k >> 2], w = c.map((x, a) => (x ? t[a]! : fr(1 - t[a]!))), weight = fr(fr(w[0]! * w[1]!) * w[2]!);
      if (weight > 0) vals[k] = fr(weight * cVertex(add(T, c as V3)));
    }
    return fr(fr(fr(vals[0]! + vals[5]!) + fr(vals[1]! + vals[4]!)) + fr(fr(vals[2]! + vals[7]!) + fr(vals[3]! + vals[6]!)));
  };
  const load = (p: V3) => fr(vertexPhi(p) * h);
  const faceInLattice = (c: V3, a: number) => c.every((v, i) => v >= 0 && v < (i === a ? N[i]! + 1 : N[i]!));
  const facePlane = (c: V3, a: number) => {
    if (!faceInLattice(c, a)) return 0;
    if (cellH(c) || cellH(add(c, unit(a, -1)))) { const fs = fSlot(shr(c, 3)); return fs === undefined ? 0 : s.fU(a, fs)[local8(c)]!; }
    const T = shr(c, 2), cs = anySlot(shr(T, 3)); return cs === undefined || T.some((x, i) => 4 * x > N[i]!) ? 0 : s.cU(a, cs)[tileIndex(T)]!;
  };
  const face = (q: V3, a: number) => facePlane(add(q, unit(a)), a);
  const cellKind = (c: V3) => (!inLattice(c) ? 2 : pageSlot(shr(c, 5)) === undefined ? 1 : 0);
  const tileState = (q: V3) => (!tileInLattice(q) ? 2 : pageSlot(shr(q, 3)) === undefined ? 1 : 0);
  const widePresent = (v: V3) => {
    const page = shr(add(v, [-1, -1, -1]), 3), plane = v.map(x => (x & 7) === 0);
    for (let k = 0; k < 8; k++) { const st = [k & 1, (k >> 1) & 1, k >> 2]; if (st.some((x, a) => x && !plane[a])) continue; if (pageSlot(add(page, st as V3)) !== undefined) return true; }
    return false;
  };
  const wallBeyond = (p: V3, k: number) => { const a = k % 3, high = k >= 3, page = shr(p, 3); page[a]! += high ? 1 : -1; if (pageIn(page)) return NO_WALL; const l = p[a]! & 7; return high ? 7 - l : l; };
  const g = s.g, ab = g.map(Math.abs) as V3;
  const upAxis = ab[1] >= ab[0] && ab[1] >= ab[2] ? 1 : ab[0] >= ab[2] ? 0 : 2, upSign = Math.sign(-g[upAxis]!), upWord = upSign === 0 ? 0 : (upSign > 0 ? 1 : 2) | (upAxis << 2);
  const perStep = (x: number) => fr(fr(x * dt) / h), scaleDt = fr(dt / h);
  const faceLow = (anchor: V3, axis: number) => (anchor[axis]! & 31) === 31 && cellKind(anchor) >= 2;
  const finite = (u: number) => Math.abs(u) <= fr(3e38);
  // ncCellClosed: outside the box, or OPEN clear in the cell's F slot; a cell no F slot holds is open.
  const cellClosed = (c: V3) => !inLattice(c) || (fSlot(shr(c, 3)) !== undefined && (s.closed?.(c) ?? false));
  // C'' (ncFaceDelta, ncEnergeticCell): the last step's velocity change at c's lower face along a through the owner rule,
  // 0 on a closed face (a closed cell either side) and off the box; energetic where |delta| >= |g| dt / 2.
  const faceDelta = (c: V3, a: number) => {
    const below = add(c, unit(a, -1));
    if (!faceInLattice(c, a) || c[a]! <= 0 || c[a]! >= N[a]! || cellClosed(c) || cellClosed(below)) return 0;
    if (cellH(c) || cellH(below)) { const fs = fSlot(shr(c, 3)); return fs === undefined ? 0 : s.fUP?.(a, fs)[local8(c)] ?? 0; }
    const T = shr(c, 2), cs = anySlot(shr(T, 3)); return cs === undefined ? 0 : s.cUP?.(a, cs)[tileIndex(T)] ?? 0;
  };
  const accelBound = fr(fr(0.5 * fr(Math.hypot(...s.g))) * dt);
  const energeticCell = (c: V3) => {
    const d = [0, 1, 2].map(a => faceDelta(c, a));
    return accelBound > 0 && fr(fr(fr(d[0]! * d[0]!) + fr(d[1]! * d[1]!)) + fr(d[2]! * d[2]!)) >= fr(accelBound * accelBound);
  };

  // umClassifyOwner (surface authority). Every cell classifies, a closed one too (the tier's crossing, HEAD's); an h
  // owner's open cell that crosses also flags OPEN_CROSS, the liquid-air crossing that heats.
  const classifyOwner = (t: V3, width: number, lane: number, c: Class, bounds: boolean) => {
    const side = 4 / width, origin = add(scale3(t, 4), scale3(corner(lane, side), width));
    const closedCell = width === 1 && cellClosed(origin);
    let inside = 0, lo = fr(3e38), hi = fr(-3e38);
    for (let k = 0; k < 8; k++) {
      const v = width === 4 ? load(add(origin, scale3(corner(k, 2), 4))) : load(add(origin, scale3(corner(k, 2), width)));
      lo = Math.min(lo, v); hi = Math.max(hi, v); if (v < 0) inside++;
      if (v < fr(4 * h * width)) c.flags |= 8;
    }
    if (bounds) for (let axis = 0; axis < 3; axis++) {
      let low = fr(3e38), high = fr(-3e38);
      for (const sign of [-1, 1]) {
        const anchor: V3 = [...origin]; if (sign < 0) anchor[axis]! -= 1;
        if (faceLow(anchor, axis)) continue;
        const u = face(anchor, axis); low = Math.min(low, finite(u) ? u : fr(-3e38)); high = Math.max(high, finite(u) ? u : fr(3e38));
      }
      c.low[axis] = Math.min(c.low[axis]!, orderKey(low)); c.high[axis] = Math.max(c.high[axis]!, orderKey(high));
    }
    const local = scale3(corner(lane, side), width), eps = fr(1e-6 * h);
    const crosses = lo <= eps && hi >= -eps && lo < hi;
    if (crosses) {
      if (width === 4) c.crossing = [0xffffffff, 0xffffffff]; else c.crossing[lane >> 5] = (c.crossing[lane >> 5]! | (1 << (lane & 31))) >>> 0;
      if (width === 1 && !closedCell) c.flags |= OPEN_CROSS;
      for (let a = 0; a < 3; a++) { c.gap[a] = Math.min(c.gap[a]!, local[a]!); c.gap[3 + a] = Math.min(c.gap[3 + a]!, 4 - width - local[a]!); }
    }
    const interior = !crosses && inside === 8, air = !crosses && inside === 0;
    if (!air) c.flags |= 4;
    if (!interior && !air) {
      for (let a = 0; a < 3; a++) { c.reach[a] = Math.min(c.reach[a]!, local[a]!); c.reach[3 + a] = Math.min(c.reach[3 + a]!, 4 - width - local[a]!); }
      c.flags |= crosses ? 3 : 1;
    }
  };
  // umSurfaceDisplacement: lane sums as the kernel keeps them (lane i % 64), then the workgroup's.
  // A cut tile (a closed cell of its own): its buried vertices (no incident cell open), its open cells and edges
  // (umCensusCellOpen: a cell outside the tile is closed), as censusFine stages them; an uncut tile's every one open.
  // The tier's pass (cutPass false) takes every cell open, HEAD's; the heat's (a cut tile with an open crossing) the cut,
  // with no buried vertex in the corner curvature.
  // cfCentre (the cell centre phi, m: its 8 vertices' f32 mean) and cfAccelerating.
  const centrePhi = (c: V3) => { let sum = 0; for (let k = 0; k < 8; k++) sum = fr(sum + load(add(c, corner(k, 2)))); return fr(0.125 * sum); };
  const accelerating = (t: V3) => {
    const cut = tileCut(t), lanes: number[][] = [];
    for (let lane = 0; lane < 64; lane++) {
      const l = corner(lane, 4), c = add(scale3(t, 4), l), v = [0, 0, 0, 0, 0, 0], own = centrePhi(c), fs = fSlot(shr(c, 3));
      if (!cellClosed(c) && fs !== undefined && own < 0) for (let a = 0; a < 3; a++) {
        const below = add(c, unit(a, -1));
        if (!faceInLattice(c, a) || c[a]! <= 0 || c[a]! >= N[a]!) continue;
        if ((cut || l[a] === 0) && cellClosed(below)) continue;
        const b = centrePhi(below), theta = b < 0 ? 1 : fr(own / fr(own - b));
        v[a] = fr(theta * (s.fUP?.(a, fs)[local8(c)] ?? 0)); v[3 + a] = theta;
      }
      lanes.push(v);
    }
    for (let lane = 0; lane < 32; lane++) for (let k = 0; k < 6; k++) lanes[lane]![k] = fr(lanes[lane]![k]! + lanes[lane + 32]![k]!);
    const sum = [0, 0, 0, 0, 0, 0]; for (let j = 0; j < 32; j++) for (let k = 0; k < 6; k++) sum[k] = fr(sum[k]! + lanes[j]![k]!);
    const d = [0, 1, 2].map(a => sum[3 + a]! > 0 ? fr(sum[a]! / sum[3 + a]!) : 0);
    return accelBound > 0 && fr(fr(fr(d[0]! * d[0]!) + fr(d[1]! * d[1]!)) + fr(d[2]! * d[2]!)) >= fr(accelBound * accelBound);
  };
  const tileCut = (t: V3) => { for (let i = 0; i < 64; i++) if (cellClosed(add(scale3(t, 4), corner(i, 4)))) return true; return false; };
  const wideBuried = (T: V3) => Array.from({ length: 8 }, (_, q) => add(scale3(T, 4), scale3(corner(q, 2), -1))).every(cellClosed);
  const displacement = (t: V3, cutPass: boolean) => {
    const origin = scale3(t, 4), raw: number[] = [], smooth: number[] = [];
    const cut = cutPass && tileCut(t);
    const buried = (v: V3) => cut && Array.from({ length: 8 }, (_, k) => add(add(origin, v), scale3(corner(k, 2), -1))).every(cellClosed);
    const cellOpen = (l: V3) => l.every(x => x >= 0 && x < 4) && (!cut || !cellClosed(add(origin, l)));
    const edgeOpen = (v: V3, a: number) => !cut || [0, 1, 2, 3].some(i => { const q: V3 = [...v]; q[(a + 1) % 3]! -= i & 1; q[(a + 2) % 3]! -= i >> 1; return cellOpen(q); });
    const vertexOpen = (v: V3) => !cut || Array.from({ length: 8 }, (_, k) => add(v, scale3(corner(k, 2), -1))).every(cellOpen);
    const buriedAt: boolean[] = []; for (let i = 0; i < 125; i++) buriedAt.push(buried(corner(i, 5)));
    const isBuried = (v: V3) => buriedAt[v[0] + 5 * v[1] + 25 * v[2]]!;
    for (let i = 0; i < 125; i++) raw.push(load(add(origin, corner(i, 5))));
    for (let i = 0; i < 125; i++) {
      const v = corner(i, 5), r = v.map(x => (x > 0 && x < 4 ? 1 : 0));
      if (buriedAt[i]) { smooth.push(raw[i]!); continue; }
      let sum = 0, weight = 0;
      for (let z = v[2] - r[2]!; z <= v[2] + r[2]!; z++) for (let y = v[1] - r[1]!; y <= v[1] + r[1]!; y++) for (let x = v[0] - r[0]!; x <= v[0] + r[0]!; x++) {
        if (isBuried([x, y, z]) || isBuried([2 * v[0] - x, 2 * v[1] - y, 2 * v[2] - z])) continue;
        const w = (x === v[0] ? 2 : 1) * (y === v[1] ? 2 : 1) * (z === v[2] ? 2 : 1); sum = fr(sum + fr(w * raw[x + 5 * y + 25 * z]!)); weight += w;
      }
      smooth.push(fr(sum / weight));
    }
    const at = (a: number[], v: V3) => a[v[0] + 5 * v[1] + 25 * v[2]]!;
    const c: number[] = []; for (let k = 0; k < 8; k++) c.push(load(add(origin, scale3(corner(k, 2), 4))));
    // umPlaneBuriedCorners: the buried corners take the least-squares plane through the smoothed vertices of the open
    // cells whose smoothed vertices change sign (a vertex once per such cell); none: they keep their markers.
    const buriedCorners = [0, 1, 2, 3, 4, 5, 6, 7].filter(k => isBuried(scale3(corner(k, 2), 4)));
    if (buriedCorners.length) {
      const A = [0, 0, 0, 0], B = [0, 0, 0, 0], C = [0, 0, 0], D = [0, 0, 0];
      for (let j = 0; j < 64; j++) {
        const cell = corner(j, 4); if (!cellOpen(cell)) continue;
        const vs = Array.from({ length: 8 }, (_, k) => add(cell, corner(k, 2)));
        if (!(vs.some(q => at(smooth, q) < 0) && vs.some(q => !(at(smooth, q) < 0)))) continue;
        for (const q of vs) {
          const v = at(smooth, q);
          for (let a = 0; a < 3; a++) { A[a]! += q[a]!; B[a]! += q[a]! * q[a]!; D[a]! += q[a]! * v; }
          A[3]! += 1; B[3]! += v; C[0]! += q[0] * q[1]; C[1]! += q[0] * q[2]; C[2]! += q[1] * q[2];
        }
      }
      if (A[3]! > 0) {
        const mean = [A[0]! / A[3]!, A[1]! / A[3]!, A[2]! / A[3]!], level = B[3]! / A[3]!;
        const dg = [0, 1, 2].map(a => B[a]! - mean[a]! * A[a]!), cr = [C[0]! - mean[0]! * A[1]!, C[1]! - mean[0]! * A[2]!, C[2]! - mean[1]! * A[2]!];
        const mo = [0, 1, 2].map(a => D[a]! - mean[a]! * B[3]!), shift = 1e-6 * Math.max(dg[0]! + dg[1]! + dg[2]!, 1);
        const M = [[dg[0]! + shift, cr[0]!, cr[1]!], [cr[0]!, dg[1]! + shift, cr[2]!], [cr[1]!, cr[2]!, dg[2]! + shift]];
        const det3 = (m: number[][]) => m[0]![0]! * (m[1]![1]! * m[2]![2]! - m[1]![2]! * m[2]![1]!) - m[0]![1]! * (m[1]![0]! * m[2]![2]! - m[1]![2]! * m[2]![0]!) + m[0]![2]! * (m[1]![0]! * m[2]![1]! - m[1]![1]! * m[2]![0]!);
        const det = det3(M), solve = (rhs: number[]) => [0, 1, 2].map(col => det3(M.map((row, i) => row.map((x, j) => (j === col ? rhs[i]! : x)))) / det);
        const S = [[dg[0]!, cr[0]!, cr[1]!], [cr[0]!, dg[1]!, cr[2]!], [cr[1]!, cr[2]!, dg[2]!]];
        let slope = [0, 0, 0];
        for (let fit = 0; fit < 2; fit++) { const rest = mo.map((m, i) => m - (S[i]![0]! * slope[0]! + S[i]![1]! * slope[1]! + S[i]![2]! * slope[2]!)); const d = solve(rest); slope = slope.map((x, i) => x + d[i]!); }
        for (const k of buriedCorners) { const q = scale3(corner(k, 2), 4); c[k] = fr(level + slope[0]! * (q[0] - mean[0]!) + slope[1]! * (q[1] - mean[1]!) + slope[2]! * (q[2] - mean[2]!)); }
      }
    }
    const held = [...c], metric = new Array<number>(8).fill(Infinity);
    for (let lane = 0; lane < 24; lane++) {
      const k = Math.floor(lane / 3), a = lane % 3, cv = scale3(corner(k, 2), 4), inward = cv.map(x => (x === 4 ? -1 : 1)), step = unit(a, inward[a]!);
      for (let j = 0; j < 4; j++) {
        const v = add(cv, scale3(step, j)), n = add(v, step);
        if (!edgeOpen(v.map((x, i) => Math.min(x, n[i]!)) as V3, a)) break;
        const u = at(raw, v), w = at(raw, n);
        if ((u < 0) === (w < 0)) continue;
        const f = fr(u / (u - w)), gr = [0, 0, 0]; gr[a] = fr(w - u);
        for (let b = 0; b < 3; b++) { if (b === a) continue; const tt = unit(b, inward[b]!); if (isBuried(add(v, tt)) || isBuried(add(n, tt))) continue; gr[b] = inward[b]! * fr((at(raw, add(v, tt)) - u) + f * ((at(raw, add(n, tt)) - w) - (at(raw, add(v, tt)) - u))); }
        const along = Math.abs(gr[a]!) / Math.max(Math.hypot(gr[0]!, gr[1]!, gr[2]!), 1e-20);
        metric[k] = Math.min(metric[k]!, fr((j + f) * along)); break;
      }
    }
    for (let k = 0; k < 8; k++) if (metric[k] !== Infinity && !isBuried(scale3(corner(k, 2), 4)) && c[k] !== 0) c[k] = Math.sign(c[k]!) * fr(metric[k]! * h);
    const low = [c[0]!, c[2]!, c[4]!, c[6]!], high = [c[1]!, c[3]!, c[5]!, c[7]!], dx = low.map((x, i) => high[i]! - x);
    const mix = (a: number, b: number, t: number) => a + t * (b - a);
    const cornerDistance = (v: V3, a: number, f: number) => {
      const t = v.map(x => x * 0.25); t[a]! += 0.25 * f;
      const x0 = low.map((l, i) => mix(l, high[i]!, t[0]!)), y0 = [mix(x0[0]!, x0[1]!, t[1]!), mix(x0[2]!, x0[3]!, t[1]!)];
      const gx = [mix(dx[0]!, dx[1]!, t[1]!), mix(dx[2]!, dx[3]!, t[1]!)];
      const grad = [mix(gx[0]!, gx[1]!, t[2]!), mix(x0[1]! - x0[0]!, x0[3]! - x0[2]!, t[2]!), y0[1]! - y0[0]!];
      return 4 * mix(y0[0]!, y0[1]!, t[2]!) / Math.max(Math.hypot(grad[0]!, grad[1]!, grad[2]!), fr(4e-3 * h));
    };
    let worst = 0, cornerMax = 0, signed = 0, crossings = 0, curveSum = 0, curves = 0;
    const alongL = new Array<number>(64).fill(0), curveL = new Array<number>(64).fill(0), crossL = new Array<number>(64).fill(0), curvesL = new Array<number>(64).fill(0);
    for (let i = 0; i < 125; i++) {
      const lane = i % 64, v = corner(i, 5), value = smooth[i]!;
      if (v.every(x => x >= 1 && x <= 3) && Math.abs(value) <= fr(1.5 * h) && vertexOpen(v)) {
        const gg3 = [0, 0, 0], d = [0, 0, 0], m = [0, 0, 0];
        for (let a = 0; a < 3; a++) {
          const below = at(smooth, add(v, unit(a, -1))), above = at(smooth, add(v, unit(a)));
          gg3[a] = (above - below) / (2 * h); d[a] = (above - 2 * value + below) / (h * h);
          const b = (a + 1) % 3, q = (sa: number, sb: number) => at(smooth, add(add(v, unit(a, sa)), unit(b, sb)));
          m[a] = (q(1, 1) - q(1, -1) - q(-1, 1) + q(-1, -1)) / (4 * h * h);
        }
        const gg = gg3[0]! ** 2 + gg3[1]! ** 2 + gg3[2]! ** 2;
        if (gg >= 1e-4) {
          const ghg = gg3[0]! ** 2 * d[0]! + gg3[1]! ** 2 * d[1]! + gg3[2]! ** 2 * d[2]! + 2 * (gg3[0]! * gg3[1]! * m[0]! + gg3[1]! * gg3[2]! * m[1]! + gg3[2]! * gg3[0]! * m[2]!);
          curveL[lane]! += ((gg * (d[0]! + d[1]! + d[2]!) - ghg) / (gg * Math.sqrt(gg))) * (4 / 3) * h; curvesL[lane]!++;
        }
      }
      if (v.every(x => x % 4 === 0)) {
        const k = shr(v, 2), centre = held[k[0] + 2 * k[1] + 4 * k[2]]!;
        if (Math.abs(centre) <= fr(2 * h)) {
          let lap = 0;
          for (let a = 0; a < 3; a++) {
            if (isBuried(v)) break;
            const lo = add(add(t, k), unit(a, -1)), hi = add(add(t, k), unit(a));
            if (!widePresent(lo) || !widePresent(hi)) continue;
            if (cut && (wideBuried(lo) || wideBuried(hi))) continue;
            const below = load(scale3(lo, 4)), above = load(scale3(hi, 4));
            if (Math.abs(below - centre) > fr(6 * h) || Math.abs(above - centre) > fr(6 * h)) continue;
            lap += above - 2 * centre + below;
          }
          cornerMax = Math.max(cornerMax, Math.abs(lap) / (12 * h));
        }
      }
      for (let a = 0; a < 3; a++) {
        if (v[a]! >= 4 || !edgeOpen(v, a)) continue;
        const n = add(v, unit(a)), next = at(smooth, n);
        if ((value < 0) !== (next < 0)) worst = Math.max(worst, Math.abs(cornerDistance(v, a, fr(value / (value - next)))));
        const r0 = at(raw, v), r1 = at(raw, n);
        if ((r0 < 0) !== (r1 < 0)) { alongL[lane]! += cornerDistance(v, a, fr(r0 / (r0 - r1))); crossL[lane]!++; }
      }
    }
    for (let l = 0; l < 64; l++) {
      if (crossL[l]) { signed += Math.trunc(Math.min(Math.max(alongL[l]!, -1e5), 1e5) * 1024); crossings += crossL[l]!; }
      if (curvesL[l]) { curveSum += Math.trunc(Math.min(Math.max(curveL[l]!, -1e5), 1e5) * 1024); curves += curvesL[l]!; }
    }
    return { worst, volume: crossings ? 4 * Math.abs(signed / (1024 * crossings)) : 0, curve: curves ? 4 * Math.abs(curveSum / (1024 * curves)) : 0, corner: 4 * cornerMax };
  };
  const classMoving = (c: Class) => {
    let speed = 0;
    for (let a = 0; a < 3; a++) { if (c.low[a] !== 0xffffffff) speed = Math.max(speed, perStep(Math.abs(orderValue(c.low[a]!)))); if (c.high[a] !== 0) speed = Math.max(speed, perStep(Math.abs(orderValue(c.high[a]!)))); }
    return speed >= fr(4 * dt / 0.5);
  };
  let coarseCurve = 0;
  const resolutionError4 = (t: V3, lane: number) => {
    let worst = 0; const c = add(scale3(t, 4), scale3(corner(lane, 2), 4)), centre = load(c);
    if (Math.abs(centre) >= fr(8 * h)) return 0;
    let lap = 0;
    for (let a = 0; a < 3; a++) {
      const lo = add(shr(c, 2), unit(a, -1)), hi = add(shr(c, 2), unit(a));
      if (!widePresent(lo) || !widePresent(hi)) continue;
      const below = load(scale3(lo, 4)), above = load(scale3(hi, 4));
      if (Math.abs(below - centre) > fr(6 * h) || Math.abs(above - centre) > fr(6 * h)) continue;
      worst = Math.max(worst, Math.abs(above - 2 * centre + below) / (16 * h)); lap += above - 2 * centre + below;
    }
    if (Math.abs(centre) <= fr(2 * h)) coarseCurve = Math.max(coarseCurve, Math.abs(lap) / (12 * h));
    return worst;
  };
  const depthSample = (t: V3, i: number): [number, number] => {
    const v = add(add(t, corner(i, 4)), [-1, -1, -1]);
    if (!widePresent(v)) return [0, 0];
    const value = load(scale3(v, 4)), low = [0, 0, 0], high = [0, 0, 0];
    for (let k = 0; k < 8; k++) { const b = corner(k, 2), cell = add(add(scale3(v, 4), b), [-1, -1, -1]); if (cellKind(cell) >= 2) continue; for (let a = 0; a < 3; a++) { high[a]! += b[a]!; low[a]! += 1 - b[a]!; } }
    const wall = [0, 1, 2].some(a => (low[a] === 0 && high[a] !== 0) || (high[a] === 0 && low[a] !== 0));
    const thickness = fr(value * (wall ? 1 : 2));
    return [orderKey(-thickness), orderKey(thickness)];
  };
  const directionTravel = (c: Class, k: number) => { const a = k % 3; return k < 3 ? perStep(Math.max(-orderValue(c.low[a]!), 0)) : perStep(Math.max(orderValue(c.high[a]!), 0)); };
  const boundary = (p: V3, width: number, c: Class) => {
    const r = { score: 0, triggered: false };
    const threshold = P.boundaryTravel * (width === 1 ? 0.5 : 1), least = threshold, margin = 4 * (width === 1 ? 2 : 1);
    const travel = [0, 1, 2, 3, 4, 5].map(k => directionTravel(c, k));
    for (let k = 0; k < 6; k++) {
      const a = k % 3, along = Math.max(travel[(a + 1) % 3]!, travel[3 + (a + 1) % 3]!, travel[(a + 2) % 3]!, travel[3 + (a + 2) % 3]!);
      if (travel[k]! < Math.max(least, along)) continue;
      const reach = margin - c.reach[k]!; if (reach < 0) continue;
      if (!(4 * wallBeyond(p, k) <= reach)) continue;
      r.score = Math.max(r.score, fr(travel[k]! / threshold));
      if (travel[k]! >= threshold) { r.triggered = true; return r; }
    }
    if ((upWord & 3) === 0) return r;
    const up = (upWord & 3) === 1 ? travel[3 + upAxis]! : travel[upAxis]!;
    if (up < least) return r;
    for (let k = 0; k < 6; k++) {
      const a = k % 3; if (a === upAxis) continue;
      const b = 3 - a - upAxis, along = Math.max(travel[b]!, travel[3 + b]!);
      if (wallBeyond(p, k) === 0 && up >= along) { r.score = Math.max(r.score, fr(up / threshold)); if (up >= threshold) r.triggered = true; }
    }
    return r;
  };
  const finish = (p: V3, width: number, c: Class) => {
    const crossing = (c.flags & 2) !== 0;
    let speed = 0, reach = 0;
    for (let a = 0; a < 3; a++) speed = Math.max(speed, perStep(Math.max(Math.abs(orderValue(c.low[a]!)), Math.abs(orderValue(c.high[a]!)))));
    for (let a = 0; a < 3; a++) { if (c.low[a] !== 0xffffffff) reach = Math.max(reach, Math.abs(fr(orderValue(c.low[a]!) * scaleDt))); if (c.high[a] !== 0) reach = Math.max(reach, Math.abs(fr(orderValue(c.high[a]!) * scaleDt))); }
    let w0 = 0, w1 = (crossing ? CROSSES : 0) | ((c.flags & 4) !== 0 ? WET : 0), gap = 0xffffffff;
    if (crossing) {
      const error = fromBits(c.error), shaped = on(0) && !(error <= P.tolerance);
      w0 |= scoreByte(fr(error / P.tolerance)); if (shaped) w1 |= trig(0) | HEAT_SHAPE;
      if ((c.flags & CLASS_ENERGETIC) !== 0) w1 |= ENERGETIC;
      if (c.deep[0] !== 0 && c.deep[1] !== 0) {
        const thickness = fr(Math.max(Math.min(orderValue(c.deep[0]), orderValue(c.deep[1])), 0) / h);
        w0 |= scoreByte(fr(P.thin / Math.max(thickness, 0.25 * P.thin))) << 8; if (thickness < P.thin && on(1)) w1 |= trig(1) | HEAT_THIN;
      }
      const b = boundary(p, width, c); w1 |= scoreByte(b.score); if (b.triggered && on(4)) w1 |= trig(4);
      gap = Math.min(Math.ceil(speed), 255) * 2 ** 24; for (let k = 0; k < 6; k++) gap += c.gap[k]! * 2 ** (4 * k);
    }
    return { w0: w0 >>> 0, w1: w1 >>> 0, gap: gap >>> 0, reach };
  };

  // ---- Classification (classify, classifyCoarse).
  const scratch = new Map<number, Scratch>();
  let reachMax = 0, crossingCount = 0, failed = 0;
  const tileOf = (cs: number, i: number, pk: V3): V3 => add(scale3(pk, 8), [i & 7, (i >> 3) & 7, i >> 6]);
  const pageOfSlot = new Map<number, V3>();
  for (const [k, cs] of s.pages) pageOfSlot.set(cs, [k & 1023, (k >>> 10) & 1023, (k >>> 20) & 1023]);
  for (const cs of s.cAll) {
    const pk = pageOfSlot.get(cs)!;
    const sc: Scratch = { keys: [], w0: [], w1: [], fin: [], cross: [], gap: [], req: new Array(16).fill(0), cubes: [] };
    scratch.set(cs, sc);
    for (let i = 0; i < BRICK_CELLS; i++) {
      const t = tileOf(cs, i, pk);
      const store = (c: Class | undefined, r?: { w0: number; w1: number; gap: number }) => {
        sc.keys[i] = c ? [...c.low, ...c.high] : identityKeys(); sc.w0[i] = r?.w0 ?? 0; sc.w1[i] = r?.w1 ?? 0; sc.fin[i] = sc.w1[i]!;
        sc.cross[i] = c ? [c.crossing[0], c.crossing[1]] : [0, 0]; sc.gap[i] = r?.gap ?? 0xffffffff;
      };
      if (!tileInLattice(t)) { store(undefined); continue; }
      if (tileH(t)) {
        if (fSlot(shr(t, 1)) === undefined) { failed++; store(undefined); continue; }
        const c = emptyClass();
        for (let lane = 0; lane < 64; lane++) {
          const l = emptyClass(); classifyOwner(t, 1, lane, l, true);
          for (let a = 0; a < 3; a++) { c.low[a] = Math.min(c.low[a]!, l.low[a]!); c.high[a] = Math.max(c.high[a]!, l.high[a]!); }
          for (let k = 0; k < 6; k++) { c.gap[k] = Math.min(c.gap[k]!, l.gap[k]!); c.reach[k] = Math.min(c.reach[k]!, l.reach[k]!); }
          c.flags |= l.flags; c.crossing = [(c.crossing[0] | l.crossing[0]) >>> 0, (c.crossing[1] | l.crossing[1]) >>> 0];
        }
        // C''' (cfAccelerating): the crossing tile's liquid cells' mean acceleration, per axis the theta-weighted mean of
        // the last step's UP over the open lower faces of its liquid cells (centre phi < 0), at least |g|/2. The lanes'
        // sums in the shader's order: lane + 32 into lane, then lanes 0..31 in turn.
        if ((c.flags & 2) !== 0 && accelerating(t)) c.flags |= CLASS_ENERGETIC;
        const openCross = (c.flags & OPEN_CROSS) !== 0; c.flags &= ~OPEN_CROSS;
        const reheat = (c.flags & 2) !== 0 && openCross && tileCut(t);
        let error = 0, heatError = 0;
        if ((c.flags & 2) !== 0) {
          const d = displacement(t, false);
          error = Math.max(d.worst, Math.max(d.volume, classMoving(c) ? Math.max(d.curve, d.corner) : 0));
          if (reheat) { const e = displacement(t, true); heatError = Math.max(e.worst, Math.max(e.volume, classMoving(c) ? Math.max(e.curve, e.corner) : 0)); }
          for (let lane = 0; lane < 64; lane++) { const q = depthSample(t, lane); c.deep = [Math.max(c.deep[0], q[0]), Math.max(c.deep[1], q[1])]; }
        }
        c.error = bitsOf(error);
        const r = finish(t, 1, c);
        if (!openCross) r.w1 = (r.w1 & ~(HEAT_SHAPE | HEAT_THIN)) >>> 0; else if (reheat && heatError <= P.tolerance) r.w1 = (r.w1 & ~HEAT_SHAPE) >>> 0;
        store(c, r); reachMax = Math.max(reachMax, r.reach); if (c.flags & 2) crossingCount++;
      } else {
        const c = emptyClass(); classifyOwner(t, 4, 0, c, false);
        if ((c.flags & 2) !== 0 && energeticCell(scale3(t, 4))) c.flags |= CLASS_ENERGETIC;
        const lo = [orderKey(fr(3e38)), orderKey(fr(3e38)), orderKey(fr(3e38))], hi = [orderKey(fr(-3e38)), orderKey(fr(-3e38)), orderKey(fr(-3e38))];
        for (let axis = 0; axis < 3; axis++) for (const sign of [-1, 1]) {
          const origin = scale3(t, 4), probe = add(origin, unit(axis, sign > 0 ? 4 : -1));
          const nw = cellKind(probe) === 0 ? (cellH(probe) ? 1 : 4) : 0, width = Math.min(4, nw !== 0 ? nw : 4), side = 4 / width;
          const anchor = add(probe, unit(axis, sign > 0 ? -1 : 0));
          for (let part = 0; part < side * side; part++) {
            const q: V3 = [...anchor]; q[(axis + 1) % 3]! += part % 4; q[(axis + 2) % 3]! += Math.floor(part / 4);
            if (faceLow(q, axis)) continue;
            const u = face(q, axis);
            lo[axis] = Math.min(lo[axis]!, orderKey(finite(u) ? u : fr(-3e38))); hi[axis] = Math.max(hi[axis]!, orderKey(finite(u) ? u : fr(3e38)));
          }
        }
        for (let a = 0; a < 3; a++) { c.low[a] = Math.min(c.low[a]!, lo[a]!); c.high[a] = Math.max(c.high[a]!, hi[a]!); }
        if ((c.flags & 2) !== 0) {
          let error = 0, curve = 0;
          for (let lane = 0; lane < 8; lane++) { coarseCurve = 0; error = Math.max(error, resolutionError4(t, lane)); curve = Math.max(curve, coarseCurve); }
          if (classMoving(c)) error = Math.max(error, 0.5 * 4 * curve);
          c.error = bitsOf(error);
          for (let lane = 0; lane < 64; lane++) { const q = depthSample(t, lane); c.deep = [Math.max(c.deep[0], q[0]), Math.max(c.deep[1], q[1])]; }
        }
        const r = finish(t, 4, c); store(c, r); reachMax = Math.max(reachMax, r.reach); if (c.flags & 2) crossingCount++;
      }
    }
  }
  // tuDeformation (uniform-narrow-band-flip.ts l.938, PLAN's ncDeformation) of the supplied DEF words, and
  // nbActivityTriggered's need rule (ncNeedTriggered).
  const PLAN_MAX = fr(3.402823e38);
  const slotKey = new Map([...s.pages].map(([k, cs]) => [cs, k]));
  let branch: keyof CensusResult["defBranches"] | undefined;
  const deformation = (t: V3) => {
    const cs = tileInLattice(t) ? s.pages.get(pageKey(shr(t, 3))) : undefined;
    if (cs === undefined) { branch = "key"; return PLAN_MAX; }
    const w = s.def ? s.def(cs, tileIndex(t)) : [0, 0, 0, 0];
    if (w[2] !== slotKey.get(cs)! + 1) { branch = "key"; return PLAN_MAX; }
    if (!(fromBits(w[3]!) > 0)) { branch = "width"; return PLAN_MAX; }
    return fromBits(w[0]!);
  };
  // fin: the final word 1, its criteria and their liquid-air heat bits (a wet wall's crossing heats nothing).
  const needTriggered = (t: V3, fin: number, keys: number[]) => {
    branch = undefined;
    const criteria = (fin >>> 16) & 63;
    if ((criteria & 0x3c) !== 0) return true;
    // C'': the shape criterion heats only an energetic tile, fast (ncFastTile) or accelerating (FIN's ENERGETIC).
    if ((criteria & 1) !== 0 && (fin & HEAT_SHAPE) !== 0 && (fast(keys) || (fin & ENERGETIC) !== 0)) return true;
    if ((criteria & 2) === 0 || (fin & HEAT_THIN) === 0) return false;
    const d = deformation(t), heated = fr(fr(dt * d) / h) > fr(0.02);
    if (branch === undefined) branch = heated ? "heated" : "cool";
    return heated;
  };
  // ---- boundCubes.
  for (const sc of scratch.values()) {
    const merge = (into: number[], k: number[]) => { for (let a = 0; a < 3; a++) { into[a] = Math.min(into[a]!, k[a]!); into[3 + a] = Math.max(into[3 + a]!, k[3 + a]!); } };
    for (let lane = 0; lane < 64; lane++) {
      const c = corner(lane, 4), k = identityKeys();
      for (let i = 0; i < 8; i++) merge(k, sc.keys[tileIndex(add(scale3(c, 2), corner(i, 2)))]!);
      sc.cubes[lane] = k;
    }
    for (let lane = 0; lane < 8; lane++) {
      const c = corner(lane, 2), k = identityKeys();
      for (let i = 0; i < 8; i++) { const q = add(scale3(c, 2), corner(i, 2)); merge(k, sc.cubes[q[0] | (q[1] << 2) | (q[2] << 4)]!); }
      sc.cubes[64 + lane] = k;
    }
    const top = identityKeys(); for (let i = 0; i < 8; i++) merge(top, sc.cubes[64 + i]!); sc.cubes[72] = top;
  }
  const slotTile = (q: V3) => { const cs = pageSlot(shr(q, 3)); return cs === undefined ? undefined : { sc: scratch.get(cs)!, i: tileIndex(q) }; };
  const keyTravel = (keys: number[], k: number) => { const a = k % 3; return k < 3 ? perStep(Math.max(-orderValue(keys[a]!), 0)) : perStep(Math.max(orderValue(keys[3 + a]!), 0)); };
  const keyToward = (keys: number[], k: number) => { const a = k % 3; return k < 3 ? perStep(orderValue(keys[3 + a]!)) : -perStep(orderValue(keys[a]!)); };
  const tileSpeed = (keys: number[]) => { let v = 0; for (let a = 0; a < 3; a++) v = Math.max(v, perStep(Math.max(Math.abs(orderValue(keys[a]!)), Math.abs(orderValue(keys[3 + a]!))))); return v; };
  // ---- importanceTile, publishImportance, interfaceBits.
  const approach = (p: V3, keys: number[], word: number) => {
    const horizon = P.approach; let best = 0;
    for (let k = 0; k < 6; k++) {
      const a = k % 3, high = k >= 3, travel = keyTravel(keys, k);
      if (travel <= 0) continue;
      let airSide = true;
      for (let c = 0; c < 4; c++) { const v = scale3(p, 4); v[a]! += high ? 4 : 0; v[(a + 1) % 3]! += 4 * (c & 1); v[(a + 2) % 3]! += 4 * (c >> 1); if (load(v) < 0) airSide = false; }
      if (!airSide) continue;
      const gap = (word >>> (4 * k)) & 15, limit = Math.min(16, Math.ceil(fr(Math.min(fr(horizon * travel), 64) / 4)) + 1);
      let air = false, gapCells = -1, closing = travel;
      for (let n = 1; n <= limit + 1; n++) {
        const q = add(p, unit(a, high ? n : -n)), state = tileState(q);
        if (state === 2) { gapCells = 4 * (n - 1) + gap; break; }
        if (n > limit) break;
        if (state === 1) { air = true; continue; }
        const st = slotTile(q)!;
        if ((st.sc.w1[st.i]! & WET) === 0) { air = true; continue; }
        if (air) { gapCells = 4 * (n - 1) + gap; closing = fr(closing + keyToward(st.sc.keys[st.i]!, k)); }
        break;
      }
      if (gapCells >= 1) best = Math.max(best, fr(fr(horizon * closing) / gapCells));
    }
    return best;
  };
  let required = 0;
  for (const cs of s.cAll) {
    const sc = scratch.get(cs)!, pk = pageOfSlot.get(cs)!;
    for (let i = 0; i < BRICK_CELLS; i++) {
      const t = tileOf(cs, i, pk); if (!tileInLattice(t)) continue;
      const w0 = sc.w0[i]!; let w1 = sc.w1[i]!;
      const crossing = (w1 & CROSSES) !== 0, wet = (w1 & WET) !== 0;
      if (crossing || wet) {
        if (crossing) { const score = approach(t, sc.keys[i]!, sc.gap[i]!); w1 |= scoreByte(score) << 8; if (score >= 1 && on(5)) w1 |= trig(5); }
        let best = 0, winner = 0;
        for (let k = 0; k < 6; k++) { const byte = k < 4 ? (w0 >>> (8 * k)) & 255 : (w1 >>> (8 * (k - 4))) & 255; if (byte > best) { best = byte; winner = k + 1; } }
        w1 |= winner << 22;
        if (((w1 >>> 16) & 63) !== 0) w1 |= REQUIRED;
      }
      w1 >>>= 0; sc.fin[i] = w1;
      if ((w1 & REQUIRED) === 0) { if (crossing) { sc.cross[i] = [0, 0]; sc.gap[i] = 0xffffffff; } continue; }
      required++; sc.req[i >> 5] = (sc.req[i >> 5]! | (1 << (i & 31))) >>> 0;
      if (!crossing) { sc.cross[i] = [0xffffffff, 0xffffffff]; sc.gap[i] = Math.min(Math.ceil(tileSpeed(sc.keys[i]!)), 255) * 2 ** 24; }
    }
  }
  // ---- decideTile (D = 0 unless `deep`).
  const span = N.map(n => n / 4) as V3;
  const listPages = () => s.cAll.map(cs => ({ cs, q: pageOfSlot.get(cs)! }));
  const interfaceIn = (sc: Scratch, low: V3, high: V3) => {
    const a = low.map(x => Math.max(x, 0)), b = high.map(x => Math.min(x, 7));
    const row = (((2 << (b[0]! - a[0]!)) - 1) << a[0]!) >>> 0, mask = [0, 0];
    for (let y = a[1]!; y <= b[1]!; y++) mask[y >> 2] = (mask[y >> 2]! | (row << (8 * (y & 3)))) >>> 0;
    for (let z = a[2]!; z <= b[2]!; z++) if (((sc.req[2 * z]! & mask[0]!) | (sc.req[2 * z + 1]! & mask[1]!)) !== 0) return true;
    return false;
  };
  const clampBox = (low: V3, high: V3) => ({ a: low.map(x => Math.max(x, 0)) as V3, b: high.map(x => Math.min(x, 2047)) as V3 });
  const interfaceAny = (low: V3, high: V3) => {
    const { a, b } = clampBox(low, high); if (b.some((x, i) => x < a[i]!)) return false;
    const pa = shr(a, 3), pb = shr(b, 3), n = pb.map((x, i) => x - pa[i]! + 1);
    const test = (cs: number, q: V3) => interfaceIn(scratch.get(cs)!, add(a, scale3(q, -8)), add(b, scale3(q, -8)));
    if (n[0]! * n[1]! * n[2]! > 64) return listPages().some(({ cs, q }) => q.every((x, i) => x >= pa[i]! && x <= pb[i]!) && test(cs, q));
    for (let z = pa[2]; z <= pb[2]; z++) for (let y = pa[1]; y <= pb[1]; y++) for (let x = pa[0]; x <= pb[0]; x++) { const q: V3 = [x, y, z], cs = pageSlot(q); if (cs !== undefined && test(cs, q)) return true; }
    return false;
  };
  const rangeKeys = (low: V3, high: V3) => {
    const keys = identityKeys(), merge = (k: number[]) => { for (let a = 0; a < 3; a++) { keys[a] = Math.min(keys[a]!, k[a]!); keys[3 + a] = Math.max(keys[3 + a]!, k[3 + a]!); } };
    const { a, b } = clampBox(low, high); if (b.some((x, i) => x < a[i]!)) return keys;
    let j = 0;
    for (;;) { const n = b.map((x, i) => (x >> j) - (a[i]! >> j) + 1); if (j >= 3 || n[0]! * n[1]! * n[2]! <= 64) break; j++; }
    const pa = shr(a, 3), pb = shr(b, 3), n = pb.map((x, i) => x - pa[i]! + 1);
    if (n[0]! * n[1]! * n[2]! > 64) { for (const { cs, q } of listPages()) if (q.every((x, i) => x >= pa[i]! && x <= pb[i]!)) merge(scratch.get(cs)!.cubes[72]!); return keys; }
    for (let z = pa[2]; z <= pb[2]; z++) for (let y = pa[1]; y <= pb[1]; y++) for (let x = pa[0]; x <= pb[0]; x++) {
      const q: V3 = [x, y, z], cs = pageSlot(q); if (cs === undefined) continue;
      const sc = scratch.get(cs)!, la = a.map((v, i) => Math.max(v - 8 * q[i]!, 0) >> j), lb = b.map((v, i) => Math.min(v - 8 * q[i]!, 7) >> j);
      for (let cz = la[2]!; cz <= lb[2]!; cz++) for (let cy = la[1]!; cy <= lb[1]!; cy++) for (let cx = la[0]!; cx <= lb[0]!; cx++) {
        if (j === 0) merge(sc.keys[cx | (cy << 3) | (cz << 6)]!);
        else { const sh = 3 - j; merge(sc.cubes[(j === 1 ? 0 : j === 2 ? 64 : 72) + (cx | (cy << sh) | (cz << (2 * sh)))]!); }
      }
    }
    return keys;
  };
  const sampledFlow = (lo: V3, hi: V3) => {
    const keys = rangeKeys(lo.map(x => Math.floor(fr(fr(x - 2.5) / 4))) as V3, hi.map(x => Math.floor(fr(fr(x + 2.5) / 4))) as V3);
    const low = [0, 1, 2].map(a => Math.min(0, keys[a] !== 0xffffffff ? orderValue(keys[a]!) : 0));
    const high = [0, 1, 2].map(a => Math.max(0, keys[3 + a] !== 0 ? orderValue(keys[3 + a]!) : 0));
    return { low, high };
  };
  const crossingCellsMeet = (q: V3, lo: V3, hi: V3) => {
    const o = scale3(q, 4), a = lo.map((x, i) => Math.max(Math.floor(fr(fr(x - o[i]!) - fr(1e-3))), 0)), b = hi.map((x, i) => Math.min(Math.floor(fr(fr(x - o[i]!) + fr(1e-3))), 3));
    if (b.some((x, i) => x < a[i]!)) return false;
    const st = slotTile(q); if (!st) return false;
    const mask = st.sc.cross[st.i]!; if (mask[0] === 0 && mask[1] === 0) return false;
    const row = ((1 << (b[0]! - a[0]! + 1)) - 1) << a[0]!;
    for (let z = a[2]!; z <= b[2]!; z++) for (let y = a[1]!; y <= b[1]!; y++) { const lane = 4 * y + 16 * z; if (((mask[lane >> 5]! >>> 0) & ((row << (lane & 31)) >>> 0)) !== 0) return true; }
    return false;
  };
  const departureMeetsSurface = (lo: V3, hi: V3) => {
    const first = lo.map(x => Math.floor(fr(fr(x - fr(1e-3)) / 4))) as V3, last = hi.map(x => Math.floor(fr(fr(x + fr(1e-3)) / 4))) as V3;
    if (!interfaceAny(first, last)) return false;
    const inner = lo.map(x => Math.ceil(fr(fr(x - fr(1e-3)) / 4))) as V3, innerLast = last.map(x => x - 1) as V3;
    if (interfaceAny(inner, innerLast)) return true;
    const a = first.map(x => Math.max(Math.min(Math.max(x, 0), 2047), 0)), b = last.map((x, i) => Math.min(Math.min(Math.max(x, 0), 2047), span[i]! - 1));
    for (let z = a[2]!; z <= b[2]!; z++) for (let y = a[1]!; y <= b[1]!; y++) {
      if (!interfaceAny([a[0]!, y, z], [b[0]!, y, z])) continue;
      const core = y >= inner[1] && y <= innerLast[1] && z >= inner[2] && z <= innerLast[2];
      for (let x = a[0]!; x <= b[0]!; x++) { if (core && x >= inner[0] && x <= innerLast[0]) { x = innerLast[0]; continue; } if (crossingCellsMeet([x, y, z], lo, hi)) return true; }
    }
    return false;
  };
  const keySpeed = (keys: number[]) => { let v = 0; for (let a = 0; a < 3; a++) { if (keys[a] !== 0xffffffff) v = Math.max(v, perStep(Math.abs(orderValue(keys[a]!)))); if (keys[3 + a] !== 0) v = Math.max(v, perStep(Math.abs(orderValue(keys[3 + a]!)))); } return v; };
  const movingShapeNear = (p: V3, keys: number[]) => {
    const reach = Math.min(Math.max(Math.ceil(fr(keySpeed(keys) / 4)), 1), 3);
    if (!interfaceAny(add(p, [-reach, -reach, -reach]), add(p, [reach, reach, reach]))) return false;
    const want = (CROSSES | REQUIRED | trig(0)) >>> 0;
    for (let z = Math.max(p[2] - reach, 0); z <= Math.min(p[2] + reach, span[2] - 1); z++) for (let y = Math.max(p[1] - reach, 0); y <= Math.min(p[1] + reach, span[1] - 1); y++) for (let x = Math.max(p[0] - reach, 0); x <= Math.min(p[0] + reach, span[0] - 1); x++) {
      const q: V3 = [x, y, z]; if (q.every((v, i) => v === p[i])) continue;
      const st = slotTile(q); if (!st) continue;
      if (((st.sc.fin[st.i]! & want) >>> 0) === want && keySpeed(st.sc.keys[st.i]!) >= fr(4 * dt / 0.5)) return true;
    }
    return false;
  };
  const joinedOf = (t: V3) => { const fs = fSlot(shr(t, 1)); if (fs === undefined) return false; const j = (t[0] & 1) | ((t[1] & 1) << 1) | ((t[2] & 1) << 2); return ((s.tileOcc(fs) >>> (8 + j)) & 1) === 1; };
  const far = Math.ceil(fr(fr(fr(Math.min(reachMax, 1e6) + 0.5) + fr(1e-3)) / 4)) + 2;
  // R2's speed horizon (ncFastTile): a tile's fastest velocity key crosses a tile within R2's two calm heads, 2 ncKeySpeed >= 4.
  const fast = (keys: number[]) => 2 * keySpeed(keys) >= 4;
  const calm = (p: V3, keys: number[]) => {
    const cs = tileInLattice(p) ? s.pages.get(pageKey(shr(p, 3))) : undefined; if (cs === undefined) return false;
    const w = s.def ? s.def(cs, tileIndex(p)) : [0, 0, 0, 0];
    if (w[2] !== slotKey.get(cs)! + 1 || !(fromBits(w[3]!) > 0)) return false;
    // A dry band tile (no liquid sample) is never calm: no water for a 4h mean to average.
    if (!(s.wet?.(cs, tileIndex(p)) ?? false)) return false;
    // A slow tile reads TC word 15 alone (the tile means); a fast one its DEF error (an h tile: and word 15).
    const tc = fromBits((s.tc ? s.tc(cs, tileIndex(p)) : [])[15] ?? 0);
    const e = !fast(keys) ? tc : tileH(p) ? Math.max(fromBits(w[1]!), tc) : fromBits(w[1]!);
    return e < PLAN_MAX && fr(fr(dt * e) / h) <= fr(0.08);
  };
  const demoted = new Set<number>(), calmNow = new Set<number>(), held = new Set<number>();
  // R2's shape and thin guard: calm means 4h holds the geometry (no shape or thin trigger in FIN).
  const shaped = new Set<number>(), loud = new Set<number>();
  // R2's curvature test: VOL, censusFine's packed mean normal of an h tile (0 without a crossing cell), and the shift.
  const volumeKept = new Set<number>(), vol = new Map<number, number>(), shift = new Map<number, number>();
  const snorm = (x: number) => Math.round(Math.min(Math.max(x, -1), 1) * 127) & 255;
  const packNormal = (n: V3) => (snorm(n[0]) | (snorm(n[1]) << 8) | (snorm(n[2]) << 16) | (127 << 24)) >>> 0;
  const unpackNormal = (w: number): [number, number, number, number] => [0, 8, 16, 24].map(b => Math.max(((((w >>> b) & 255) << 24) >> 24) / 127, -1)) as [number, number, number, number];
  const gradOf = (v: readonly number[]): V3 => [v[1]! + v[3]! + v[5]! + v[7]! - v[0]! - v[2]! - v[4]! - v[6]!, v[2]! + v[3]! + v[6]! + v[7]! - v[0]! - v[1]! - v[4]! - v[5]!, v[4]! + v[5]! + v[6]! + v[7]! - v[0]! - v[1]! - v[2]! - v[3]!];
  const volOf = (p: V3) => {
    const at = (l: V3) => load(add(scale3(p, 4), l)), eps = fr(1e-6 * h);
    let crossing = false, count = 0; const sum = [0, 0, 0];
    for (let i = 0; i < 64; i++) {
      const l: V3 = [i & 3, (i >> 2) & 3, i >> 4];
      const v = Array.from({ length: 8 }, (_, k) => at(add(l, [k & 1, (k >> 1) & 1, k >> 2])));
      const lo = Math.min(...v), hi = Math.max(...v); if (lo <= eps && hi >= -eps && lo < hi) crossing = true;
      if (!(lo < 0 && hi >= 0)) continue;
      const g = gradOf(v), m = Math.hypot(...g); if (!(m > 0)) continue;
      for (let a = 0; a < 3; a++) sum[a] += Math.round(g[a]! / m * 4096); count++;
    }
    if (!crossing || count === 0) return 0;
    const m = Math.hypot(...sum); return m > 0 ? packNormal(sum.map(x => x / m) as V3) : 0;
  };
  const tileNormal = (q: V3): [number, number, number, number] => {
    if (!tileInLattice(q)) return [0, 0, 0, 0];
    const st = slotTile(q); if (!st) return [0, 0, 0, 0];
    if (tileH(q)) { if ((st.sc.fin[st.i]! & CROSSES) === 0) return [0, 0, 0, 0]; const w = volOf(q); return w === 0 ? [0, 0, 0, 0] : unpackNormal(w); }
    const v = Array.from({ length: 8 }, (_, k) => load(add(scale3(q, 4), [4 * (k & 1), 4 * ((k >> 1) & 1), 4 * (k >> 2)])));
    if (!(Math.min(...v) < 0 && Math.max(...v) >= 0)) return [0, 0, 0, 0];
    const g = gradOf(v), m = Math.hypot(...g); return m > 0 ? [g[0] / m, g[1] / m, g[2] / m, 1] : [0, 0, 0, 0];
  };
  const curveShift = (p: V3) => {
    const w = volOf(p); if (w === 0) return 0;
    const n0 = unpackNormal(w); let div = 0;
    for (let a = 0; a < 3; a++) {
      const e: V3 = [0, 0, 0]; e[a] = 1; const lo = tileNormal(add(p, e.map(x => -x) as V3)), hi = tileNormal(add(p, e));
      if (lo[3] > 0.5 && hi[3] > 0.5) div += 0.5 * (hi[a]! - lo[a]!); else if (hi[3] > 0.5) div += hi[a]! - n0[a]!; else if (lo[3] > 0.5) div += n0[a]! - lo[a]!;
    }
    return Math.abs(div) / 3;
  };
  // e: the unit direction from q towards the tile asking (0: q itself); counted where the surface runs on into it or
  // moves towards it (ncNormalRate).
  const normalRate = (q: V3, e: V3 = [0, 0, 0]) => {
    const n = tileNormal(q); if (n[3] < 0.5) return 0;
    const cs = pageSlot(shr(q, 3))!, w = s.tc ? s.tc(cs, tileIndex(q)) : [];
    const v = fromBits(w[0] ?? 0) * n[0] + fromBits(w[1] ?? 0) * n[1] + fromBits(w[2] ?? 0) * n[2], d = n[0] * e[0] + n[1] * e[1] + n[2] * e[2];
    if (!(Math.abs(d) <= 0.75 || v * d > 0)) return 0;
    const r = Math.abs(v); return r < 3e38 ? r : 0;
  };
  const quiet = (p: V3) => {
    let r = normalRate(p);
    for (let a = 0; a < 3; a++) { const e: V3 = [0, 0, 0]; e[a] = 1; r = Math.max(r, normalRate(add(p, e.map(x => -x) as V3), e), normalRate(add(p, e), e.map(x => -x) as V3)); }
    return dt * r / (4 * h) <= 0.08;
  };
  const decide = (p: V3, sc: Scratch, i: number) => {
    const band = decideBand(p, sc, i);
    // censusFine writes VOL for every h tile, band or not: the oracle's packed normal of each, compared word for word.
    if (head && tileH(p)) vol.set(key(p), volOf(p));
    // No thin trigger, and no shape trigger on a fast tile (R2's speed horizon).
    const keys = sc.keys[i]!;
    if (band && head && (((sc.fin[i]! & trig(1)) >>> 0) !== 0 || (((sc.fin[i]! & trig(0)) >>> 0) !== 0 && fast(keys)))) { if (calm(p, keys)) shaped.add(key(p)); return band; }
    if (band && head && tileH(p)) { const d = curveShift(p); shift.set(key(p), d); if (!(d <= 0.08)) { if (calm(p, keys)) volumeKept.add(key(p)); return band; } }
    if (!(band && head && calm(p, keys))) return band;
    if (!quiet(p)) { loud.add(key(p)); return band; }
    calmNow.add(key(p));
    const cs = s.pages.get(pageKey(shr(p, 3)))!;
    if (!tileH(p) || (s.calmPrev?.(cs, tileIndex(p)) ?? false)) { demoted.add(key(p)); return false; }
    held.add(key(p)); return true;
  };
  const decideBand = (p: V3, sc: Scratch, i: number) => {
    const joined = joinedOf(p), fin = sc.fin[i]!, keys = sc.keys[i]!;
    if (deep?.has(p) && !joined) return deep.decided(p);
    if (!joined && !((fin & CROSSES) !== 0 || movingShapeNear(p, keys))) return false;
    if (!joined && !interfaceAny(add(p, [-far, -far, -far]), add(p, [far, far, far]))) return false;
    let fine = joined || (fin & REQUIRED) !== 0;
    if (!fine) {
      let lo = scale3(p, 4), hi = add(lo, [4, 4, 4]);
      const start = sampledFlow(lo, hi);
      const mid = sampledFlow(lo.map((x, a) => fr(x - fr(fr(0.5 * start.high[a]!) * scaleDt))) as V3, hi.map((x, a) => fr(x - fr(fr(0.5 * start.low[a]!) * scaleDt))) as V3);
      lo = lo.map((x, a) => fr(x - fr(mid.high[a]! * scaleDt))) as V3; hi = hi.map((x, a) => fr(x - fr(mid.low[a]! * scaleDt))) as V3;
      fine = departureMeetsSurface(lo.map(x => fr(x - P.drift)) as V3, hi.map(x => fr(x + P.drift)) as V3);
    }
    return fine;
  };
  const want = new Map<number, Uint32Array>(), trigOut = new Map<number, number>(), cand = new Set<number>(), band = new Map<number, boolean>();
  const thinAlone = { tiles: 0, heated: 0 }, defBranches = { key: 0, width: 0, heated: 0, cool: 0 };
  const counters = { CENSUS_CLAMPED: 0, CENSUS_CAND: 0, CENSUS_REACH: bitsOf(reachMax), CENSUS_CROSSING: crossingCount, CENSUS_REQUIRED: required, CENSUS_BAND: 0, CENSUS_REFINED: 0, CENSUS_TRIGGERED: 0 };
  const inRange = (b: V3) => b.every((x, a) => x >= 0 && x < brickRange[a]!);
  for (const cs of s.cAll) {
    const sc = scratch.get(cs)!, pk = pageOfSlot.get(cs)!, w = new Uint32Array(CMASK_WORDS), pageCand = new Set<number>();
    for (let i = 0; i < BRICK_CELLS; i++) {
      const t = tileOf(cs, i, pk); if (!tileInLattice(t)) continue;
      const fine = decide(t, sc, i); band.set(key(t), fine);
      if (!fine) continue;
      counters.CENSUS_BAND++; if (!tileH(t)) counters.CENSUS_REFINED++;
      // The ring (R3): the bricks of the tile's 27 tiles, two a side per axis.
      const b = shr(add(t, [-1, -1, -1]), 1); let clamped = false;
      for (let n = 0; n < 8; n++) { const nb = add(b, [n & 1, (n >> 1) & 1, n >> 2]); if (!inRange(nb) || fSlot(nb) !== undefined) continue; clamped = true; pageCand.add(key(nb)); }
      if (clamped) counters.CENSUS_CLAMPED++;
      w[i >> 5] = (w[i >> 5]! | (1 << (i & 31))) >>> 0;
    }
    want.set(cs, w);
    for (const k of pageCand) cand.add(k);
    counters.CENSUS_CAND += pageCand.size;
    for (let bl = 0; bl < 64; bl++) {
      const fs = s.bricks.get(key(add(scale3(pk, 4), corner(bl, 4)))); if (fs === undefined) continue;
      let bits = 0;
      for (let j = 0; j < 8; j++) {
        const q = add(scale3(corner(bl, 4), 2), corner(j, 2)), w1 = sc.fin[tileIndex(q)]!, criteria = (w1 >>> 16) & 63, heat = needTriggered(add(scale3(pk, 8), q), w1, sc.keys[tileIndex(q)]!);
        if (heat) bits |= 1 << j;
        if ((w1 >>> 17) & 1) bits |= 1 << (8 + j);
        if (criteria === 2) { thinAlone.tiles++; if (heat) thinAlone.heated++; if (branch !== undefined) defBranches[branch]++; }
      }
      trigOut.set(fs, bits >>> 0); counters.CENSUS_TRIGGERED += [...Array(8).keys()].filter(j => (bits >> j) & 1).length;
    }
  }
  return { scratch, want, trig: trigOut, cand, band, demoted, calmNow, held, shaped, volumeKept, vol, shift, loud, counters, failed, thinAlone, defBranches };
}

// ---- The frozen scene: a pool whose surface rolls across h and 4h tiles, a falling drop above it, a run-out
// against the low x wall; the h region is a box of resident bricks, so the band's edge meets the clamp.
const N: V3 = [64, 32, 64];
const FAR = phiFar(N);
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
/** h tiles: tile x 0..5, z 0..5 (all y), and a lone 4h hole at tile (2, 2, 2). */
const hTile = (t: V3) => t[0] <= 5 && t[2] <= 5 && !(t[0] === 2 && t[1] === 2 && t[2] === 2);

test("setCensus (GPU): classification, importance, the band, the clamp, WANT and TRIG match the old census", GPU, async () => {
  const { withUniformDevice } = await import("../../../tests/helpers/uniform-geometric");
  const { BrickEngine } = await import("./brick-engine");
  const { bringUpCG } = await import("./brick-bringup-cg");
  const { bringUpStages } = await import("./brick-stages");
  const { snapshot, readChannel } = await import("./brick-readback");
  await withUniformDevice("brick census", async device => {
    const e = await BrickEngine.create(device, {
      lattice: N, h_m: H_M, dt_s: DT, gravity_mps2: [0, -9.81, 0],
      capacities: { f: 700, c: 16, b: 0, k: 0, particles: 1024 }, liquid: [{ lo: [-1, -1, -1], hi: [40, 12, 22] }], solver: L => bringUpCG(L),
      stages: bringUpStages,
    });
    const L = e.layout, snap = await snapshot(e);
    const cAll = snap.list(LIST.C_ALL);
    assert.ok(cAll.length >= 2 && snap.bricks.size > 8, `pages ${cAll.length}, bricks ${snap.bricks.size}`);
    const pageCoord = new Map([...snap.pages].map(([k, cs]) => [cs, [k & 1023, (k >>> 10) & 1023, (k >>> 20) & 1023] as V3]));
    const resident = (b: V3) => snap.bricks.has(key(b));
    // C'': the last step's velocity change (UP), by tile: free fall (-g dt, energetic) on a third of the tiles, an
    // eighth of g's (calm) elsewhere, with a sideways part on alternate tiles; clear of the |g| dt / 2 bound either way.
    const upOf = (c: V3, a: number) => {
      const T = shr(c, 2), fall = (T[0] * 7 + T[1] * 3 + T[2] * 5) % 3 === 0, g = 9.81 * DT;
      return a === 1 ? -(fall ? 1 : 0.125) * g : a === 0 && (T[2] & 1) === 1 ? 0.125 * g : 0;
    };
    // CMASK: h where hTile holds and the tile's brick is resident; 4h elsewhere.
    for (const cs of cAll) {
      const pk = pageCoord.get(cs)!, m = new Uint32Array(CMASK_WORDS);
      for (let i = 0; i < BRICK_CELLS; i++) { const t = add(scale3(pk, 8), [i & 7, (i >> 3) & 7, i >> 6]); if (hTile(t) && 4 * t[1] < N[1] && resident(shr(t, 1))) m[i >> 5] = (m[i >> 5]! | (1 << (i & 31))) >>> 0; }
      e.write("dir", L.dir.cmask + CMASK_WORDS * cs, m);
      const cw = (fn: (T: V3) => number) => { const w = new Float32Array(BRICK_CELLS); for (let i = 0; i < BRICK_CELLS; i++) w[i] = fn(add(scale3(pk, 8), [i & 7, (i >> 3) & 7, i >> 6])); return w; };
      e.write("phi", L.base("phi", "C", "PHI") + BRICK_STRIDE * cs, cw(T => phiOf(scale3(T, 4)) + 0.15 * Math.sin(T[0] + 2 * T[2])));
      for (let a = 0; a < 3; a++) e.write("vel", L.base("vel", "C", `US${a}`) + BRICK_STRIDE * cs, cw(T => velOf(scale3(T, 4), a) * 0.9));
      for (let a = 0; a < 3; a++) e.write("vel", L.base("vel", "C", `UP${a}`) + BRICK_STRIDE * cs, cw(T => upOf(scale3(T, 4), a)));
    }
    const occ = new Uint32Array(L.cap.f);
    for (const [kb, fs] of snap.bricks) {
      const b: V3 = [kb & 1023, (kb >>> 10) & 1023, (kb >>> 20) & 1023];
      const fw = (fn: (c: V3) => number) => { const w = new Float32Array(BRICK_CELLS); for (let l = 0; l < BRICK_CELLS; l++) w[l] = fn(add(scale3(b, 8), [l & 7, (l >> 3) & 7, l >> 6])); return w; };
      e.write("phi", L.base("phi", "F", "PHI") + BRICK_STRIDE * fs, fw(phiOf));
      for (let a = 0; a < 3; a++) e.write("vel", L.base("vel", "F", `US${a}`) + BRICK_STRIDE * fs, fw(c => velOf(c, a)));
      for (let a = 0; a < 3; a++) e.write("vel", L.base("vel", "F", `UP${a}`) + BRICK_STRIDE * fs, fw(c => upOf(c, a)));
      // Particles on every tile; a join on tile 7 of the bricks at x = 9 (4h there, in the air).
      occ[fs] = 0xff | (b[0] === 9 && b[1] === 2 ? 0x8000 : 0);
    }
    e.write("set", L.set.tileOcc, occ);
    const ring = new Float32Array([DT]), ringH = new Float32Array([H_M]), grav = new Float32Array([0, -9.81, 0]);
    e.write("set", L.set.ring + RING.DT, ring); e.write("set", L.set.ring + RING.H, ringH); e.write("set", L.set.ring + RING.GRAVITY, grav);
    e.write("set", L.set.ring + RING.FLAGS, new Uint32Array([0]));
    // ---- The last certificate's packed copy (DEF: deformation, error, page key + 1, width as f32), which no
    // certificate wrote here: hashed per tile, a stale key (key + 2), width 0, width -4, else a valid deformation a
    // factor 2 either side of the thin budget (dt deformation / h = 0.02), clear of f32 rounding at the threshold, and
    // an error a factor 2 either side of R2's detail budget (dt error / h = 0.08) on an independent hash bit.
    const defWords = new Uint32Array(await e.read("census"));
    {
      const budget = 0.02 * H_M / DT, detail = 0.08 * H_M / DT;
      for (const cs of cAll) {
        const k = [...snap.pages].find(([, v]) => v === cs)![0];
        for (let i = 0; i < BRICK_CELLS; i++) {
          const r = (Math.imul(i + 1, 0x9e3779b1) ^ Math.imul(cs + 7, 0x85ebca6b)) >>> 0, u = (r >>> 8) / 2 ** 24, at = CENSUS.STRIDE * cs + CENSUS.DEF + 4 * i;
          const width = u < 0.08 ? 0 : u < 0.12 ? -4 : (i & 1) ? 1 : 4;
          defWords[at] = bitsOf(fr(budget * (u < 0.56 ? 0.5 : 2) * (1 + 0.3 * u))); defWords[at + 1] = bitsOf(fr(detail * ((r >>> 4) & 1 ? 0.5 : 2) * (1 + 0.3 * u)));
          defWords[at + 2] = (u >= 0.12 && u < 0.2 ? k + 2 : k + 1) >>> 0; defWords[at + 3] = bitsOf(width);
          defWords[CENSUS.STRIDE * cs + CENSUS.TC + 16 * i + 8] = (r >>> 5) & 3 ? 1 : 0; // TC's wet bit: 3/4 wet
          // TC word 15 (R2's velocity test, the second difference of the tile means): a factor 2 either side of the
          // budget on an independent bit; the mean velocity (words 0..2): still in alternate 4-tile blocks along x, else
          // vertical at twice the importance budget (dt |u| / 4h = 0.16), so quiet tiles lie inside still blocks.
          const tc = CENSUS.STRIDE * cs + CENSUS.TC + 16 * i, t = add(scale3(pageCoord.get(cs)!, 8), [i & 7, (i >> 3) & 7, i >> 6]);
          defWords[tc + 15] = bitsOf(fr(detail * ((r >>> 7) & 1 ? 0.5 : 2) * (1 + 0.3 * u)));
          const loud = ((t[0] >> 2) & 1) === 1;
          defWords[tc] = bitsOf(0); defWords[tc + 1] = bitsOf(loud ? fr(0.16 * 4 * H_M / DT) : 0); defWords[tc + 2] = bitsOf(0);
        }
      }
      e.write("census", 0, defWords);
    }
    // ---- Read the state back as stored, and run the oracle.
    const fRead = async (name: string) => (await readChannel(e, name === "PHI" ? "phi" : "vel", name)) as (slot: number) => Float32Array;
    const cRead = async (pool: "phi" | "vel", name: string) => { const m = new Map<number, Float32Array>(); for (const cs of snap.pages.values()) m.set(cs, new Float32Array((await e.read(pool, L.base(pool, "C", name) + BRICK_STRIDE * cs, BRICK_CELLS)).buffer)); return m; };
    const fPhi = await fRead("PHI"), fU = [await fRead("US0"), await fRead("US1"), await fRead("US2")];
    const cPhi = await cRead("phi", "PHI"), cU = [await cRead("vel", "US0"), await cRead("vel", "US1"), await cRead("vel", "US2")];
    const fUP = [await fRead("UP0"), await fRead("UP1"), await fRead("UP2")], cUP = [await cRead("vel", "UP0"), await cRead("vel", "UP1"), await cRead("vel", "UP2")];
    const dir = await e.read("dir");
    const state: CensusState = {
      N, h: fr(H_M), dt: fr(DT), g: [0, fr(-9.81), 0], pages: snap.pages, bricks: snap.bricks,
      cmask: cs => dir.subarray(L.dir.cmask + CMASK_WORDS * cs, L.dir.cmask + CMASK_WORDS * (cs + 1)),
      fPhi, cPhi: cs => cPhi.get(cs)!, fU: (a, fs) => fU[a]!(fs), cU: (a, cs) => cU[a]!.get(cs)!, tileOcc: fs => occ[fs]!, cAll,
      fUP: (a, fs) => fUP[a]!(fs), cUP: (a, cs) => cUP[a]!.get(cs)!,
      def: (cs, i) => [...defWords.subarray(CENSUS.STRIDE * cs + CENSUS.DEF + 4 * i, CENSUS.STRIDE * cs + CENSUS.DEF + 4 * i + 4)],
      wet: (cs, i) => (defWords[CENSUS.STRIDE * cs + CENSUS.TC + 16 * i + 8]! & 1) === 1,
      tc: (cs, i) => [...defWords.subarray(CENSUS.STRIDE * cs + CENSUS.TC + 16 * i, CENSUS.STRIDE * cs + CENSUS.TC + 16 * i + 16)],
    };
    const want = censusOracle(state);
    assert.equal(want.failed, 0);
    // C'': the UP pattern makes some crossing tiles energetic and leaves others calm, each with a shape misfit.
    { let energetic = 0, calmShaped = 0; for (const sc of want.scratch.values()) sc.fin.forEach(f => { if (f & HEAT_SHAPE) { if (f & ENERGETIC) energetic++; else calmShaped++; } });
      console.log(`setCensus C'': shape-heated tiles energetic ${energetic}, not ${calmShaped}`); assert.ok(energetic > 0 && calmShaped > 0, "both kinds of shaped tile"); }
    console.log(`setCensus: thin-alone tiles ${want.thinAlone.tiles}, heated ${want.thinAlone.heated}; tuDeformation branches ${JSON.stringify(want.defBranches)}`);
    // The scene's thin tiles all carry a shape or impact bit too, so its need rule never reads DEF (no thin-alone tile):
    // brick-census-dam.test.ts's frame 43 (874 thin-alone tiles) covers tuDeformation's branches.
    if (want.thinAlone.tiles > 0) for (const [b, n] of Object.entries(want.defBranches)) assert.ok(n > 0, `tuDeformation branch ${b} covered on a thin-alone tile`);
    const candBefore = (await e.read("set", H.N_CAND, 1))[0]!;
    // ---- Run setCensus.
    await e.run(brickCensusSchedule(L, k => ({ indirect: ARGS.PER_LIST * k + ARGS.G1 })));
    const set = await e.read("set"), census = await e.read("census");
    assert.equal(set[H.FAIL], 0, `FAIL ${set[H.FAIL]} info ${set[H.FAIL_INFO]?.toString(16)}`);
    // ---- Compare.
    const mismatch: string[] = [];
    const check = (ok: boolean, what: string) => { if (!ok && mismatch.length < 30) mismatch.push(what); };
    const byteNear = (a: number, b: number, shift: number) => Math.abs(((a >>> shift) & 255) - ((b >>> shift) & 255)) <= 1;
    const FLAG_BITS = 0xffff0000 >>> 0;
    let tiles = 0;
    for (const cs of cAll) {
      const sc = want.scratch.get(cs)!, o = CENSUS.STRIDE * cs, pk = pageCoord.get(cs)!;
      for (let i = 0; i < BRICK_CELLS; i++) {
        const t = add(scale3(pk, 8), [i & 7, (i >> 3) & 7, i >> 6]), at = `page ${cs} tile ${t.join(",")}${state.cmask(cs)[i >> 5]! >>> (i & 31) & 1 ? " h" : " 4h"}`;
        for (let k = 0; k < 6; k++) check(census[o + CENSUS.KEYS + 6 * i + k] === sc.keys[i]![k], `${at} key ${k}: ${census[o + CENSUS.KEYS + 6 * i + k]} vs ${sc.keys[i]![k]}`);
        const gw0 = census[o + CENSUS.W0 + i]!, gw1 = census[o + CENSUS.W1 + i]!, gfin = census[o + CENSUS.FIN + i]!;
        check(byteNear(gw0, sc.w0[i]!, 0) && byteNear(gw0, sc.w0[i]!, 8), `${at} w0 ${gw0.toString(16)} vs ${sc.w0[i]!.toString(16)}`);
        check(((gw1 & FLAG_BITS) >>> 0) === ((sc.w1[i]! & FLAG_BITS) >>> 0) && byteNear(gw1, sc.w1[i]!, 0), `${at} w1 ${gw1.toString(16)} vs ${sc.w1[i]!.toString(16)}`);
        check(((gfin & FLAG_BITS & ~(7 << 22)) >>> 0) === ((sc.fin[i]! & FLAG_BITS & ~(7 << 22)) >>> 0) && byteNear(gfin, sc.fin[i]!, 0) && byteNear(gfin, sc.fin[i]!, 8), `${at} fin ${gfin.toString(16)} vs ${sc.fin[i]!.toString(16)}`);
        check(census[o + CENSUS.CROSS + 2 * i] === sc.cross[i]![0] && census[o + CENSUS.CROSS + 2 * i + 1] === sc.cross[i]![1], `${at} crossing cells`);
        check(census[o + CENSUS.GAP + i] === sc.gap[i], `${at} gap ${census[o + CENSUS.GAP + i]!.toString(16)} vs ${sc.gap[i]!.toString(16)}`);
        for (let k = 0; k < 4; k++) check(census[o + CENSUS.DEF + 4 * i + k] === defWords[CENSUS.STRIDE * cs + CENSUS.DEF + 4 * i + k], `${at} DEF ${k}: the census wrote the certificate's copy`);
        tiles++;
      }
      for (let w = 0; w < 16; w++) check(census[o + CENSUS.REQ + w] === sc.req[w], `page ${cs} required word ${w}`);
      for (let c = 0; c < 73; c++) for (let k = 0; k < 6; k++) check(census[o + CENSUS.CUBES + 6 * c + k] === sc.cubes[c]![k], `page ${cs} cube ${c} key ${k}`);
      const gotWant = set.subarray(L.set.want + CMASK_WORDS * cs, L.set.want + CMASK_WORDS * (cs + 1));
      for (let w = 0; w < 16; w++) check(gotWant[w] === want.want.get(cs)![w], `page ${cs} WANT word ${w}: ${gotWant[w]!.toString(16)} vs ${want.want.get(cs)![w]!.toString(16)}`);
    }
    for (const [fs, bits] of want.trig) check(set[L.set.trig + fs] === bits, `TRIG slot ${fs}: ${set[L.set.trig + fs]} vs ${bits}`);
    const got = new Set<number>(); for (let e2 = candBefore; e2 < set[H.N_CAND]!; e2++) got.add(set[L.set.cand + e2]!);
    check(got.size === want.cand.size && [...want.cand].every(k => got.has(k)), `candidates ${[...got].sort()} vs ${[...want.cand].sort()}`);
    for (const [name, w] of Object.entries(CENSUS_H)) {
      const v = want.counters[name as keyof typeof CENSUS_H];
      if (name === "CENSUS_REACH") check(Math.abs(fromBits(set[w]!) - fromBits(v)) <= 1e-6 * fromBits(v), `${name} ${fromBits(set[w]!)} vs ${fromBits(v)}`);
      else check(set[w] === v, `${name} ${set[w]} vs ${v}`);
    }
    assert.deepEqual(mismatch, [], `${mismatch.length} mismatches`);
    // ---- The boot's TRIG-only census (brickTrigCensusSchedule) on the same state: the same TRIG words, and no WANT,
    // candidate or keep-mark write (WANT and TRIG poisoned first, the candidate count held).
    {
      const wantPoison = new Uint32Array(CMASK_WORDS * L.cap.c).fill(0xa5a5a5a5), trigPoison = new Uint32Array(L.cap.f).fill(0x5a5a5a5a);
      e.write("set", L.set.want, wantPoison); e.write("set", L.set.trig, trigPoison);
      await e.run(brickTrigCensusSchedule(L, k => ({ indirect: ARGS.PER_LIST * k + ARGS.G1 })));
      const after = await e.read("set");
      assert.equal(after[H.FAIL], 0);
      assert.deepEqual([...after.subarray(L.set.want, L.set.want + CMASK_WORDS * L.cap.c)], [...wantPoison], "the TRIG-only census writes no WANT word");
      for (const [fs, bits] of want.trig) assert.equal(after[L.set.trig + fs], bits, `TRIG-only census, slot ${fs}`);
      assert.equal(after[H.N_CAND], set[H.N_CAND], "no candidate");
      assert.deepEqual([...after.subarray(L.set.markF, L.set.markF + L.cap.f)], [...set.subarray(L.set.markF, L.set.markF + L.cap.f)], "no keep mark");
      assert.equal(after[CENSUS_H.CENSUS_TRIGGERED], want.counters.CENSUS_TRIGGERED);
      assert.equal(after[CENSUS_H.CENSUS_BAND], 0, "no band decision");
    }
    // ---- The real D (D != 0): a step head's census (keep, cfg.w = 1) on region tiers. Regions R.x >= 1 COLD, the rest
    // HOT: the head's deep pass (censusCoarse) makes exactly R.x >= 2 deep (every stored neighbour COLD), so the h tiles
    // (x <= 5, regions 0 and 1) take R2's live decision. Each deep
    // tile's carried CC_DECIDED bit is the complement of its D = 0 band (decidedPrev = 1 where D = 0 bands none) and its
    // CC_STATIC bit a pattern: WANT, DEC and STAT must be the oracle's with that D, and the deep count H.HEAT_DEEP.
    {
      const RGW = (cs: number, r: number, w: number) => CENSUS.STRIDE * cs + CENSUS.REG + RG.WORDS * r + w;
      const regionOf = (t: V3): V3 => [t[0] >> 2, t[1] >> 2, t[2] >> 2];
      const cens = await e.read("census");
      const before = new Map<number, Uint32Array>();
      let regions = 0;
      for (const cs of cAll) {
        const pk = pageCoord.get(cs)!;
        for (let r = 0; r < 8; r++) { const R = [2 * pk[0] + (r & 1), 2 * pk[1] + ((r >> 1) & 1), 2 * pk[2] + (r >> 2)]; cens[RGW(cs, r, RG.TIER)] = R[0]! >= 1 ? 0 : 2; regions++; }
        const dec = new Uint32Array(16), stat = new Uint32Array(16);
        for (let i = 0; i < BRICK_CELLS; i++) {
          const t = add(scale3(pk, 8), [i & 7, (i >> 3) & 7, i >> 6]);
          if (!want.band.get(key(t))) dec[i >> 5] = (dec[i >> 5]! | (1 << (i & 31))) >>> 0;
          if ((i * 7) % 5 === 0) stat[i >> 5] = (stat[i >> 5]! | (1 << (i & 31))) >>> 0;
        }
        cens.set(dec, CENSUS.STRIDE * cs + CENSUS.DEC); cens.set(stat, CENSUS.STRIDE * cs + CENSUS.STAT);
        before.set(cs, stat);
        // The hysteresis's CALM bits, hashed half a tile, under the page key + 1 but in one workgroup of five (key + 2: a
        // reused slot's, unknown, not calm).
        const k = [...snap.pages].find(([, v]) => v === cs)![0];
        for (let w = 0; w < 16; w++) cens[CENSUS.STRIDE * cs + CENSUS.CALM + w] = Math.imul(w + 1, 0x9e3779b1) ^ Math.imul(cs + 3, 0x85ebca6b);
        for (let r = 0; r < 8; r++) cens[CENSUS.STRIDE * cs + CENSUS.CALM_KEY + r] = ((cs + r) % 5 === 0 ? k + 2 : k + 1) >>> 0;
      }
      e.write("census", 0, new Uint32Array(cens));
      await e.run(brickCensusSchedule(L, k => ({ indirect: ARGS.PER_LIST * k + ARGS.G1 }), LIST.C_ALL, LIST.F_ALL, true, true, true));
      const after = await e.read("set"), cAfter = await e.read("census");
      assert.equal(after[H.FAIL], 0, `FAIL ${after[H.FAIL]} info ${after[H.FAIL_INFO]?.toString(16)}`);
      const deepT = (t: V3) => regionOf(t)[0] >= 2 && 4 * t[1] < N[1];
      const calmPrev = (cs: number, i: number) => cens[CENSUS.STRIDE * cs + CENSUS.CALM_KEY + (i >> 6)] === [...snap.pages].find(([, v]) => v === cs)![0] + 1 && ((cens[CENSUS.STRIDE * cs + CENSUS.CALM + (i >> 5)]! >>> (i & 31)) & 1) === 1;
      const withDeep = censusOracle({ ...state, calmPrev }, { has: deepT, decided: t => !want.band.get(key(t)) }, true);
      const bad: string[] = [];
      let deepTiles = 0, kept = 0, prevOne = 0, demotedH = 0;
      for (const cs of cAll) {
        const pk = pageCoord.get(cs)!, o = CENSUS.STRIDE * cs;
        for (let r = 0; r < 8; r++) { const deep = 2 * pk[0] + (r & 1) >= 2; if (cAfter[RGW(cs, r, RG.DEEP)] !== +deep && bad.length < 20) bad.push(`page ${cs} region ${r} deep ${cAfter[RGW(cs, r, RG.DEEP)]}`); }
        for (let w = 0; w < 16; w++) {
          const g = after[L.set.want + CMASK_WORDS * cs + w]!, x = withDeep.want.get(cs)![w]!;
          if (g !== x && bad.length < 20) bad.push(`page ${cs} WANT word ${w}: ${g.toString(16)} vs ${x.toString(16)}`);
          if (cAfter[o + CENSUS.DEC + w] !== x && bad.length < 20) bad.push(`page ${cs} DEC word ${w}`);
          let calmX = 0; for (let b = 0; b < 32; b++) { const i = 32 * w + b; if (withDeep.calmNow.has(key(add(scale3(pk, 8), [i & 7, (i >> 3) & 7, i >> 6])))) calmX |= 1 << b; }
          if (cAfter[o + CENSUS.CALM + w] !== calmX >>> 0 && bad.length < 20) bad.push(`page ${cs} CALM word ${w}: ${cAfter[o + CENSUS.CALM + w]!.toString(16)} vs ${(calmX >>> 0).toString(16)}`);
          if (w < 8 && cAfter[o + CENSUS.CALM_KEY + w] !== [...snap.pages].find(([, v]) => v === cs)![0] + 1 && bad.length < 20) bad.push(`page ${cs} CALM_KEY ${w}`);
        }
        for (let i = 0; i < BRICK_CELLS; i++) {
          const t = add(scale3(pk, 8), [i & 7, (i >> 3) & 7, i >> 6]);
          if (withDeep.demoted.has(key(t)) && ((dir[L.dir.cmask + CMASK_WORDS * cs + (i >> 5)]! >>> (i & 31)) & 1) === 1) demotedH++;
          if (4 * t[1] >= N[1]) continue;
          const joined = ((occ[snap.bricks.get(key(shr(t, 1)))!] ?? 0) >>> (8 + ((t[0] & 1) | ((t[1] & 1) << 1) | ((t[2] & 1) << 2)))) & 1;
          const deep = deepT(t) && !joined;
          const statX = deep ? (before.get(cs)![i >> 5]! >>> (i & 31)) & 1 : joined;
          if (((cAfter[o + CENSUS.STAT + (i >> 5)]! >>> (i & 31)) & 1) !== statX && bad.length < 20) bad.push(`tile ${t} STAT`);
          if (deep) { deepTiles++; if (withDeep.band.get(key(t)) !== want.band.get(key(t)) || withDeep.demoted.has(key(t))) kept++; if (!want.band.get(key(t))) prevOne++; }
        }
      }
      assert.equal(after[REGION_H.HEAT_DEEP], regions / 2, "deep regions");
      assert.deepEqual(bad, [], `${bad.length} mismatches with D != 0`);
      // The hysteresis's branches: calm h tiles held (not calm before, or under a stale key), calm h tiles demoted at
      // their second calm head, calm 4h tiles demoted at once; and the counters.
      // R2's curvature test: censusFine's VOL word (the packed mean normal) of every h tile, against the oracle's, each
      // byte within one step of the snorm (f32 against Float64 rounding).
      let volBad = 0, normals = 0;
      const volMiss: string[] = [];
      for (const [k, d] of withDeep.vol) {
        const t: V3 = [k & 1023, (k >>> 10) & 1023, (k >>> 20) & 1023], cs = snap.pages.get(key(shr(t, 3)))!, i = tileIndex(t);
        const g = cAfter[CENSUS.STRIDE * cs + CENSUS.VOL + i]!; if (d !== 0) normals++;
        const close = (g === 0) === (d === 0) && [0, 8, 16, 24].every(b => Math.abs(((((g >>> b) & 255) << 24) >> 24) - ((((d >>> b) & 255) << 24) >> 24)) <= 1);
        if (!close) { volBad++; if (volMiss.length < 20) volMiss.push(`tile ${t} VOL ${g.toString(16)} vs ${d.toString(16)}`); }
      }
      assert.equal(volBad, 0, `VOL: ${volBad} tiles off the oracle; ${volMiss.join("; ")}`);
      const shifts = [...withDeep.shift.values()];
      const hyst = { calm: withDeep.calmNow.size, held: withDeep.held.size, demotedH, demoted4: withDeep.demoted.size - demotedH, thinGuarded: withDeep.shaped.size, loud: withDeep.loud.size, normals, curvatureTested: shifts.length, curvatureKept: withDeep.volumeKept.size, shiftMax: Math.max(...shifts, 0), shiftOver: shifts.filter(d => d > 0.08).length };
      // The curvature test measures the scene's curved surface (a plane's shift is 0): it is exercised on both sides.
      assert.ok(normals > 50 && hyst.shiftOver > 0 && hyst.shiftOver < shifts.length && hyst.loud > 0, JSON.stringify(hyst));
      console.log("setCensus hysteresis:", JSON.stringify(hyst));
      assert.ok(hyst.held > 0 && hyst.demotedH > 0 && hyst.demoted4 > 0, JSON.stringify(hyst));
      // The shape guard's branch is reached on the old dam's census (brick-census-dam.test.ts); here it must agree.
      assert.equal(after[H.CENSUS_CALM_HELD], hyst.held, "CENSUS_CALM_HELD");
      assert.equal(after[H.CENSUS_CALM], withDeep.demoted.size, "CENSUS_CALM");
      assert.ok(deepTiles > 100 && kept === deepTiles && prevOne > 50, `deep tiles ${deepTiles}, kept ${kept}, decidedPrev = 1 ${prevOne}`);
      console.log("setCensus D != 0:", JSON.stringify({ regions, deepTiles, prevOne }));
    }
    // Coverage: the scene reaches every rule the dam's policy runs.
    const fins = [...want.scratch.values()].flatMap(sc => sc.fin);
    const has = (b: number) => fins.filter(x => (x & b) !== 0).length;
    const c = want.counters, bandH = [...want.band.values()].filter(Boolean).length;
    const report = { tiles, thinAlone: want.thinAlone, crossing: c.CENSUS_CROSSING, required: c.CENSUS_REQUIRED, band: c.CENSUS_BAND, refined: c.CENSUS_REFINED, clamped: c.CENSUS_CLAMPED, cand: c.CENSUS_CAND, shape: has(trig(0)), thin: has(trig(1)), impact: has(trig(4)), approach: has(trig(5)), approachScored: fins.filter(x => ((x >>> 8) & 255) !== 0 && (x & CROSSES) !== 0).length, bandH };
    console.log("setCensus:", JSON.stringify(report));
    assert.ok(report.shape > 0 && report.thin > 0 && report.impact > 0 && report.approachScored > 0 && report.clamped > 0 && report.refined > 0 && report.band > report.required, JSON.stringify(report));
    e.destroy();
  });
});

// ---- Deep COLD: the brick census takes D = 0. The old decideTile's deep branch (l.2852) returns a deep tile's
// last decision and reads nothing another tile's decision reads, so on every tile that is not deep the
// decisions are those of D = 0. CPU only, on a synthetic state.
test("setCensus: D = 0 gives the old decisions on every tile that is not deep COLD", () => {
  const pages = new Map<number, number>(), bricks = new Map<number, number>();
  const lattice: V3 = [64, 32, 32];
  pages.set(key([0, 0, 0]), 0); pages.set(key([1, 0, 0]), 1);
  let fsNext = 0;
  for (let bz = 0; bz < 4; bz++) for (let by = 0; by < 3; by++) for (let bx = 0; bx < 7; bx++) bricks.set(key([bx, by, bz]), fsNext++);
  const fPhi = new Map<number, Float32Array>(), fU = [0, 1, 2].map(() => new Map<number, Float32Array>());
  for (const [kb, fs] of bricks) {
    const b: V3 = [kb & 1023, (kb >>> 10) & 1023, (kb >>> 20) & 1023];
    const fw = (fn: (c: V3) => number) => { const w = new Float32Array(BRICK_CELLS); for (let l = 0; l < BRICK_CELLS; l++) w[l] = fn(add(scale3(b, 8), [l & 7, (l >> 3) & 7, l >> 6])); return w; };
    fPhi.set(fs, fw(phiOf)); for (let a = 0; a < 3; a++) fU[a]!.set(fs, fw(c => velOf(c, a)));
  }
  const cw = (pk: V3, fn: (T: V3) => number) => { const w = new Float32Array(BRICK_CELLS); for (let i = 0; i < BRICK_CELLS; i++) w[i] = fn(add(scale3(pk, 8), [i & 7, (i >> 3) & 7, i >> 6])); return w; };
  const masks = new Map<number, Uint32Array>(), cPhi = new Map<number, Float32Array>(), cU = [0, 1, 2].map(() => new Map<number, Float32Array>());
  for (const [k, cs] of pages) {
    const pk: V3 = [k & 1023, (k >>> 10) & 1023, (k >>> 20) & 1023], m = new Uint32Array(16);
    for (let i = 0; i < BRICK_CELLS; i++) { const t = add(scale3(pk, 8), [i & 7, (i >> 3) & 7, i >> 6]); if (t[0] < 12 && t[1] < 6 && bricks.has(key(shr(t, 1)))) m[i >> 5] = (m[i >> 5]! | (1 << (i & 31))) >>> 0; }
    masks.set(cs, m); cPhi.set(cs, cw(pk, T => phiOf(scale3(T, 4)))); for (let a = 0; a < 3; a++) cU[a]!.set(cs, cw(pk, T => velOf(scale3(T, 4), a)));
  }
  const state: CensusState = {
    N: lattice, h: fr(H_M), dt: fr(DT), g: [0, fr(-9.81), 0], pages, bricks, cmask: cs => masks.get(cs)!,
    fPhi: fs => fPhi.get(fs)!, cPhi: cs => cPhi.get(cs)!, fU: (a, fs) => fU[a]!.get(fs)!, cU: (a, cs) => cU[a]!.get(cs)!, tileOcc: () => 0xff, cAll: [0, 1],
  };
  const zero = censusOracle(state);
  // Deep COLD regions (4^3 tiles): every third region, each holding a decision of its own (the complement of D = 0).
  const deepRegion = (t: V3) => ((t[0] >> 2) + 3 * (t[1] >> 2) + 5 * (t[2] >> 2)) % 3 === 0;
  const withDeep = censusOracle(state, { has: deepRegion, decided: t => !zero.band.get(key(t)) });
  let deepTiles = 0, other = 0, bandTiles = 0;
  for (const [k, b] of zero.band) {
    const t: V3 = [k & 1023, (k >>> 10) & 1023, (k >>> 20) & 1023];
    if (deepRegion(t)) { deepTiles++; assert.equal(withDeep.band.get(k), !b, `deep tile ${t} keeps its last decision`); }
    else { other++; if (b) bandTiles++; assert.equal(withDeep.band.get(k), b, `tile ${t} is decided as with D = 0`); }
  }
  assert.ok(deepTiles > 50 && other > 50 && bandTiles > 5, `deep ${deepTiles}, other ${other}, band ${bandTiles}`);
});

// ---- A wetted wall: a glass plate (closed cells x 9..11, y 0..19, through the h tiles at x = 2) standing in a pool
// whose surface (y 9.3) crosses it. Its buried vertices (no incident cell open, x 10..11) hold the air marker +5 h a
// solid's burial leaves them; the plate's open neighbours keep the pool's φ. The tier and the heat apart (coordinator
// 2026-11-26): below the surface the wall's liquid against the markers crosses for the tier, as at HEAD (the wet wall
// stays h), and heats nothing; at the surface the pool meets the wall and crosses as anywhere, and its liquid-air misfit
// (the cut's: open cells, buried corners on the plane, no buried vertex in the corner curvature) is the heat's.
const WALL_N: V3 = [64, 32, 32];
const wallClosed = (c: V3) => c[0] >= 9 && c[0] <= 11 && c[1] <= 19;
const wallCellClosed = (c: V3) => c.some((x, a) => x < 0 || x >= WALL_N[a]!) || wallClosed(c);
const wallBuried = (v: V3) => Array.from({ length: 8 }, (_, k) => add(v, scale3(corner(k, 2), -1))).every(wallCellClosed);
const wallPhi = (v: V3) => (wallBuried(v) ? 5 : Math.max(v[1] - 9.3 + 0.4 * Math.sin(0.3 * v[2]) + 0.2 * Math.cos(0.21 * v[0]), -8));
function wallState(closed: boolean): CensusState {
  const pages = new Map<number, number>(), bricks = new Map<number, number>();
  pages.set(key([0, 0, 0]), 0); pages.set(key([1, 0, 0]), 1);
  let fsNext = 0;
  for (let bz = 0; bz < 4; bz++) for (let by = 0; by < 3; by++) for (let bx = 0; bx < 7; bx++) bricks.set(key([bx, by, bz]), fsNext++);
  const fPhi = new Map<number, Float32Array>(), fU = [0, 1, 2].map(() => new Map<number, Float32Array>());
  for (const [kb, fs] of bricks) {
    const b: V3 = [kb & 1023, (kb >>> 10) & 1023, (kb >>> 20) & 1023];
    const fw = (fn: (c: V3) => number) => { const w = new Float32Array(BRICK_CELLS); for (let l = 0; l < BRICK_CELLS; l++) w[l] = fn(add(scale3(b, 8), [l & 7, (l >> 3) & 7, l >> 6])); return w; };
    fPhi.set(fs, fw(wallPhi)); for (let a = 0; a < 3; a++) fU[a]!.set(fs, fw(c => (wallClosed(c) ? 0 : 0.2 * velOf(c, a))));
  }
  const cw = (pk: V3, fn: (T: V3) => number) => { const w = new Float32Array(BRICK_CELLS); for (let i = 0; i < BRICK_CELLS; i++) w[i] = fn(add(scale3(pk, 8), [i & 7, (i >> 3) & 7, i >> 6])); return w; };
  const masks = new Map<number, Uint32Array>(), cPhi = new Map<number, Float32Array>(), cU = [0, 1, 2].map(() => new Map<number, Float32Array>());
  for (const [k, cs] of pages) {
    const pk: V3 = [k & 1023, (k >>> 10) & 1023, (k >>> 20) & 1023], m = new Uint32Array(16);
    for (let i = 0; i < BRICK_CELLS; i++) { const t = add(scale3(pk, 8), [i & 7, (i >> 3) & 7, i >> 6]); if (t[0] < 12 && t[1] < 6 && bricks.has(key(shr(t, 1)))) m[i >> 5] = (m[i >> 5]! | (1 << (i & 31))) >>> 0; }
    masks.set(cs, m); cPhi.set(cs, cw(pk, T => wallPhi(scale3(T, 4)))); for (let a = 0; a < 3; a++) cU[a]!.set(cs, cw(pk, T => 0.2 * velOf(scale3(T, 4), a)));
  }
  return {
    N: WALL_N, h: fr(H_M), dt: fr(DT), g: [0, fr(-9.81), 0], pages, bricks, cmask: cs => masks.get(cs)!,
    fPhi: fs => fPhi.get(fs)!, cPhi: cs => cPhi.get(cs)!, fU: (a, fs) => fU[a]!.get(fs)!, cU: (a, cs) => cU[a]!.get(cs)!, tileOcc: () => 0xff, cAll: [0, 1],
    ...(closed ? { closed: wallClosed } : {}),
  };
}
/** Every h tile's census words by tile key: W1, the crossing cells and FIN. */
function wallWords(r: CensusResult) {
  const out = new Map<number, { w1: number; cross: [number, number]; fin: number; t: V3 }>();
  for (const [cs, sc] of r.scratch) for (let i = 0; i < BRICK_CELLS; i++) {
    const t = add(scale3([cs, 0, 0], 8), [i & 7, (i >> 3) & 7, i >> 6]);
    if (t[0] < 12 && t[1] < 6) out.set(key(t), { w1: sc.w1[i]!, cross: sc.cross[i]!, fin: sc.fin[i]!, t });
  }
  return out;
}
test("setCensus: a wetted wall keeps HEAD's tier and heats nothing; the free surface at the wall heats by its liquid-air misfit", () => {
  const withRule = censusOracle(wallState(true)), noRule = censusOracle(wallState(false));
  assert.equal(withRule.failed, 0); assert.equal(noRule.failed, 0);
  const a = wallWords(withRule), b = wallWords(noRule);
  const HEAT = (1 << 26) | (1 << 27), SHAPE = 1 << 16, tier = (x: number) => (x & ~HEAT) >>> 0;
  let sleeve = 0, sleeveShape = 0, contact = 0, contactHeatOff = 0, far = 0;
  for (const [k, w] of a) {
    const t = w.t, o = b.get(k)!, crosses = (x: number) => (x & CROSSES) !== 0;
    // The lattice's high z face (z = 32) is beyond the bricks: its vertices read FAR, a domain wall either way.
    if (t[2] === 0 || t[2] === 7) continue;
    if (t[0] === 2 && t[1] <= 1) {
      // The sleeve: the wall's liquid against the markers crosses, the tier's words the all-open mask's, and no heat.
      assert.ok(crosses(w.w1), `tile ${t}: the wetted wall crosses for the tier`);
      assert.deepEqual([tier(w.w1), tier(w.fin), w.cross], [tier(o.w1), tier(o.fin), o.cross], `tile ${t}: the tier's words are HEAD's`);
      assert.equal(w.fin & HEAT, 0, `tile ${t}: a wetted wall heats nothing`);
      if (w.fin & SHAPE) sleeveShape++;
      sleeve++;
    } else if (t[0] === 2 && t[1] === 2) {
      // The contact row: the tier's words are HEAD's; its heat bits are within the tier's criteria, by the cut's misfit.
      assert.ok(crosses(w.w1), `tile ${t}: the free surface at the wall crosses`);
      assert.deepEqual([tier(w.w1), tier(w.fin), w.cross], [tier(o.w1), tier(o.fin), o.cross], `tile ${t}: the tier's words are HEAD's`);
      assert.ok(((w.fin & (1 << 26)) === 0 || (w.fin & SHAPE) !== 0) && ((w.fin & (1 << 27)) === 0 || (w.fin & (1 << 17)) !== 0), `tile ${t}: heat within the tier`);
      if ((w.fin & SHAPE) !== 0 && (w.fin & (1 << 26)) === 0) contactHeatOff++;
      contact++;
    } else if (t[0] >= 5) {
      // Away from the wall (no shell, no depth vertex beside it): the words are the all-open mask's, heat and all.
      assert.deepEqual([w.w1, w.cross], [o.w1, o.cross], `tile ${t}: unchanged away from the wall`);
      far++;
    }
  }
  console.log("setCensus wall (oracle):", JSON.stringify({ sleeve, sleeveShape, contact, contactHeatOff, far }));
  assert.ok(sleeve === 12 && contact === 6 && far > 40 && sleeveShape > 0, `sleeve ${sleeve} (shape ${sleeveShape}), contact ${contact}, far ${far}`);
});

test("setCensus (GPU): the wetted wall's closed cells (FLAGS.OPEN) classify as the oracle's", GPU, async () => {
  const { withUniformDevice } = await import("../../../tests/helpers/uniform-geometric");
  const { BrickEngine } = await import("./brick-engine");
  const { bringUpCG } = await import("./brick-bringup-cg");
  const { bringUpStages } = await import("./brick-stages");
  const { snapshot, readChannel } = await import("./brick-readback");
  const { FLAG } = await import("./brick-abi");
  await withUniformDevice("brick census wall", async device => {
    const e = await BrickEngine.create(device, {
      lattice: WALL_N, h_m: H_M, dt_s: DT, gravity_mps2: [0, -9.81, 0],
      capacities: { f: 400, c: 8, b: 0, k: 0, particles: 1024 }, liquid: [{ lo: [-1, -1, -1], hi: [56, 12, 33] }], solver: L => bringUpCG(L),
      stages: bringUpStages,
    });
    const L = e.layout, snap = await snapshot(e);
    const cAll = snap.list(LIST.C_ALL);
    const pageCoord = new Map([...snap.pages].map(([k, cs]) => [cs, [k & 1023, (k >>> 10) & 1023, (k >>> 20) & 1023] as V3]));
    const wallH = (t: V3) => t[0] < 12 && t[1] < 6;
    for (const cs of cAll) {
      const pk = pageCoord.get(cs)!, m = new Uint32Array(CMASK_WORDS);
      for (let i = 0; i < BRICK_CELLS; i++) { const t = add(scale3(pk, 8), [i & 7, (i >> 3) & 7, i >> 6]); if (wallH(t) && 4 * t[1] < WALL_N[1] && snap.bricks.has(key(shr(t, 1)))) m[i >> 5] = (m[i >> 5]! | (1 << (i & 31))) >>> 0; }
      e.write("dir", L.dir.cmask + CMASK_WORDS * cs, m);
      const cw = (fn: (T: V3) => number) => { const w = new Float32Array(BRICK_CELLS); for (let i = 0; i < BRICK_CELLS; i++) w[i] = fn(add(scale3(pk, 8), [i & 7, (i >> 3) & 7, i >> 6])); return w; };
      e.write("phi", L.base("phi", "C", "PHI") + BRICK_STRIDE * cs, cw(T => wallPhi(scale3(T, 4))));
      for (let a = 0; a < 3; a++) e.write("vel", L.base("vel", "C", `US${a}`) + BRICK_STRIDE * cs, cw(T => 0.2 * velOf(scale3(T, 4), a)));
    }
    const flags = (await readChannel(e, "phi", "FLAGS", true)) as (slot: number) => Uint32Array;
    let closedCells = 0;
    const occ = new Uint32Array(L.cap.f);
    for (const [kb, fs] of snap.bricks) {
      const b: V3 = [kb & 1023, (kb >>> 10) & 1023, (kb >>> 20) & 1023];
      const fw = (fn: (c: V3) => number) => { const w = new Float32Array(BRICK_CELLS); for (let l = 0; l < BRICK_CELLS; l++) w[l] = fn(add(scale3(b, 8), [l & 7, (l >> 3) & 7, l >> 6])); return w; };
      e.write("phi", L.base("phi", "F", "PHI") + BRICK_STRIDE * fs, fw(wallPhi));
      for (let a = 0; a < 3; a++) e.write("vel", L.base("vel", "F", `US${a}`) + BRICK_STRIDE * fs, fw(c => (wallClosed(c) ? 0 : 0.2 * velOf(c, a))));
      const f = new Uint32Array(flags(fs));
      for (let l = 0; l < BRICK_CELLS; l++) { const c = add(scale3(b, 8), [l & 7, (l >> 3) & 7, l >> 6]); if (wallClosed(c)) { f[l] = (f[l]! & ~FLAG.OPEN) >>> 0; closedCells++; } }
      e.write("phi", L.base("phi", "F", "FLAGS") + BRICK_STRIDE * fs, f);
      occ[fs] = 0xff;
    }
    e.write("set", L.set.tileOcc, occ);
    e.write("set", L.set.ring + RING.DT, new Float32Array([DT])); e.write("set", L.set.ring + RING.H, new Float32Array([H_M]));
    e.write("set", L.set.ring + RING.GRAVITY, new Float32Array([0, -9.81, 0])); e.write("set", L.set.ring + RING.FLAGS, new Uint32Array([0]));
    assert.ok(closedCells > 1000, `closed cells ${closedCells}`);
    const fRead = async (name: string) => (await readChannel(e, name === "PHI" ? "phi" : "vel", name)) as (slot: number) => Float32Array;
    const cRead = async (pool: "phi" | "vel", name: string) => { const m = new Map<number, Float32Array>(); for (const cs of snap.pages.values()) m.set(cs, new Float32Array((await e.read(pool, L.base(pool, "C", name) + BRICK_STRIDE * cs, BRICK_CELLS)).buffer)); return m; };
    const fPhi = await fRead("PHI"), fU = [await fRead("US0"), await fRead("US1"), await fRead("US2")];
    const cPhi = await cRead("phi", "PHI"), cU = [await cRead("vel", "US0"), await cRead("vel", "US1"), await cRead("vel", "US2")];
    const dir = await e.read("dir"), words = await e.read("census");
    const state: CensusState = {
      N: WALL_N, h: fr(H_M), dt: fr(DT), g: [0, fr(-9.81), 0], pages: snap.pages, bricks: snap.bricks,
      cmask: cs => dir.subarray(L.dir.cmask + CMASK_WORDS * cs, L.dir.cmask + CMASK_WORDS * (cs + 1)),
      fPhi, cPhi: cs => cPhi.get(cs)!, fU: (a, fs) => fU[a]!(fs), cU: (a, cs) => cU[a]!.get(cs)!, tileOcc: fs => occ[fs]!, cAll,
      def: (cs, i) => [...words.subarray(CENSUS.STRIDE * cs + CENSUS.DEF + 4 * i, CENSUS.STRIDE * cs + CENSUS.DEF + 4 * i + 4)],
      wet: (cs, i) => (words[CENSUS.STRIDE * cs + CENSUS.TC + 16 * i + 8]! & 1) === 1,
      tc: (cs, i) => [...words.subarray(CENSUS.STRIDE * cs + CENSUS.TC + 16 * i, CENSUS.STRIDE * cs + CENSUS.TC + 16 * i + 16)],
      closed: wallClosed,
    };
    const want = censusOracle(state);
    assert.equal(want.failed, 0);
    await e.run(brickCensusSchedule(L, k => ({ indirect: ARGS.PER_LIST * k + ARGS.G1 })));
    const set = await e.read("set"), census = await e.read("census");
    assert.equal(set[H.FAIL], 0, `FAIL ${set[H.FAIL]} info ${set[H.FAIL_INFO]?.toString(16)}`);
    const mismatch: string[] = [];
    const check = (ok: boolean, what: string) => { if (!ok && mismatch.length < 30) mismatch.push(what); };
    const byteNear = (a: number, b: number, shift: number) => Math.abs(((a >>> shift) & 255) - ((b >>> shift) & 255)) <= 1;
    const FLAG_BITS = 0xffff0000 >>> 0;
    let sleeve = 0, contact = 0;
    for (const cs of cAll) {
      const sc = want.scratch.get(cs)!, o = CENSUS.STRIDE * cs, pk = pageCoord.get(cs)!;
      for (let i = 0; i < BRICK_CELLS; i++) {
        const t = add(scale3(pk, 8), [i & 7, (i >> 3) & 7, i >> 6]), at = `tile ${t.join(",")}`;
        for (let k = 0; k < 6; k++) check(census[o + CENSUS.KEYS + 6 * i + k] === sc.keys[i]![k], `${at} key ${k}`);
        const gw0 = census[o + CENSUS.W0 + i]!, gw1 = census[o + CENSUS.W1 + i]!, gfin = census[o + CENSUS.FIN + i]!;
        check(byteNear(gw0, sc.w0[i]!, 0) && byteNear(gw0, sc.w0[i]!, 8), `${at} w0 ${gw0.toString(16)} vs ${sc.w0[i]!.toString(16)}`);
        check(((gw1 & FLAG_BITS) >>> 0) === ((sc.w1[i]! & FLAG_BITS) >>> 0) && byteNear(gw1, sc.w1[i]!, 0), `${at} w1 ${gw1.toString(16)} vs ${sc.w1[i]!.toString(16)}`);
        check(((gfin & FLAG_BITS & ~(7 << 22)) >>> 0) === ((sc.fin[i]! & FLAG_BITS & ~(7 << 22)) >>> 0) && byteNear(gfin, sc.fin[i]!, 0) && byteNear(gfin, sc.fin[i]!, 8), `${at} fin ${gfin.toString(16)} vs ${sc.fin[i]!.toString(16)}`);
        check(census[o + CENSUS.CROSS + 2 * i] === sc.cross[i]![0] && census[o + CENSUS.CROSS + 2 * i + 1] === sc.cross[i]![1], `${at} crossing cells`);
        check(census[o + CENSUS.GAP + i] === sc.gap[i], `${at} gap`);
        if (t[0] === 2 && t[2] >= 1 && t[2] <= 6 && (state.cmask(cs)[i >> 5]! >>> (i & 31) & 1) === 1) {
          if (t[1] <= 1) { check((gw1 & CROSSES) !== 0 && (gfin & ((1 << 26) | (1 << 27))) === 0, `${at}: the wetted wall does not cross for the tier, or heats`); sleeve++; }
          if (t[1] === 2) { check((gw1 & CROSSES) !== 0, `${at}: the free surface at the wall does not cross`); contact++; }
        }
      }
      const gotWant = set.subarray(L.set.want + CMASK_WORDS * cs, L.set.want + CMASK_WORDS * (cs + 1));
      for (let w = 0; w < 16; w++) check(gotWant[w] === want.want.get(cs)![w], `page ${cs} WANT word ${w}`);
    }
    for (const [fs, bits] of want.trig) check(set[L.set.trig + fs] === bits, `TRIG slot ${fs}: ${set[L.set.trig + fs]} vs ${bits}`);
    assert.deepEqual(mismatch, [], `${mismatch.length} mismatches`);
    assert.ok(sleeve === 12 && contact === 6, `sleeve ${sleeve}, contact ${contact}`);
    console.log("setCensus wall:", JSON.stringify({ closedCells, sleeve, contact, crossing: want.counters.CENSUS_CROSSING, band: want.counters.CENSUS_BAND }));
    e.destroy();
  });
});

test("setCensus: the census header words are free in H, or H declares them as the census does", () => {
  for (const [name, w] of Object.entries(CENSUS_H)) {
    assert.ok(w >= 224 && w <= 231, name);
    const declared = (H as Record<string, number>)[name];
    if (declared !== undefined) assert.equal(declared, w, name);
    else assert.ok(!Object.values(H).includes(w as never), `${name}: word ${w} is taken in H`);
  }
  assert.equal(NONE, 0xffffffff); assert.equal(SEAM, 0x80000000);
});
