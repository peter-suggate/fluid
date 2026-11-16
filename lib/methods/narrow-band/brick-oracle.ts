/**
 * CPU oracles for the brick engine (tests only):
 * - `brickSetOracle`: the reach rule's resident set after one setMark and
 *   setUpdate, from the state the GPU held before the step;
 * - `residencyOracle`: the head's residency (brick-residency.wgsl.ts resMark, seep S2j's per-source reach) from the
 *   set before the head (ACT, PSUPPORT, BOUND_F) and, tiered, this census's WANT rings;
 * - `pressureOracle`: the pressure slot contract's system (brick-abi.md §6.1),
 *   assembled in Float64 from the GPU's phi and u*, and solved to round-off by
 *   Jacobi-preconditioned CG.
 */
import { ACT, BRICK_STRIDE, CMASK_WORDS, PHI_AIR, PSUPPORT_WORDS, type BrickLayout, brickKey, brickKeyCoord, bricksPerAxis, cellLocal } from "./brick-abi";
import { reachSet, type Vec3 } from "./brick-engine";
import type { BrickSnapshot } from "./brick-readback";
import { RES_MARGIN } from "./brick-residency.wgsl";

const THETA_MIN = 1e-3;
const liquidPhi = (p: number) => p < -THETA_MIN;
const theta = (l: number, a: number) => Math.min(Math.max(-l / Math.max(a - l, 1e-30), THETA_MIN), 1 / THETA_MIN);

/** A vertex (or cell) field over the live bricks, NONE-bricks reading `outside`. */
export function brickField(s: BrickSnapshot, words: Float32Array, base: number, outside: number) {
  return (c: Vec3): number => {
    if (c.some(v => v < 0)) return outside;
    const key = ((c[0] >> 3) | ((c[1] >> 3) << 10) | ((c[2] >> 3) << 20)) >>> 0, slot = s.bricks.get(key);
    return slot === undefined ? outside : words[base + BRICK_STRIDE * slot + cellLocal(c[0], c[1], c[2])]!;
  };
}

/** The keys the reach rule keeps or enters, from the live set, phi and the bounds the step starts from. */
export function brickSetOracle(s: BrickSnapshot, phi: (v: Vec3) => number, bound: (slot: number) => number, bins: (slot: number) => number,
  lattice: Vec3, dt: number, h: number, gravity: Vec3): Set<number> {
  const slotOf = new Map<string, number>(), coords: Vec3[] = [];
  for (const [key, slot] of s.bricks) { const b = brickKeyCoord(key); slotOf.set(b.join(","), slot); coords.push(b); }
  const liquid = (b: Vec3) => {
    let lo = [8, 8, 8], hi = [-1, -1, -1];
    for (let l = 0; l < 512; l++) {
      const c = [l & 7, (l >> 3) & 7, l >> 6], v = [0, 1, 2].map(a => 8 * b[a]! + c[a]!) as unknown as Vec3;
      if (v.some((x, a) => x > lattice[a]!)) continue;
      if (phi(v) < 0) { lo = lo.map((m, a) => Math.min(m, c[a]!)); hi = hi.map((m, a) => Math.max(m, c[a]!)); }
    }
    return hi[0]! < 0 ? undefined : { lo: lo as unknown as Vec3, hi: hi as unknown as Vec3 };
  };
  return reachSet(lattice, coords, liquid, b => bound(slotOf.get(b.join(","))!), b => bins(slotOf.get(b.join(","))!) > 0, dt, h, gravity);
}

export interface ResidencyOracle {
  /** The F bricks and C pages (keys) the head leaves resident. */
  readonly bricks: Set<number>;
  readonly pages: Set<number>;
  /** H.RES_RADIUS: the largest source reach, in tiles (rounded up). */
  readonly radius: number;
}

/**
 * The head's residency on the CPU (brick-residency.wgsl.ts resMark, seep S2j's per-source per-axis reach). `before`
 * is the set the head starts from (its F_ALL, ACT, PSUPPORT and BOUND_F words, CMASK and pages); `want` the set buffer
 * after the head (this census's WANT and TRIG words, for censusDecide's tiered band rings). A source brick (ACT.RES
 * tiles, PSUPPORT word 0 or this census's TRIG bits 0..7) keeps every brick, and tiered every page, of its tiles' vertex box grown per axis and side by
 * max(travel, (bound + |g| dt) dt / h) + 4 ACT_DILATION cells; every kept brick keeps its page. Tiered, a page source
 * (a page with 4h-owned liquid tiles, a C corner φ < 0, in `c`'s C φ) keeps the pages of those tiles' box grown by
 * (speed + |g| dt) dt / h + 4 ACT_DILATION cells, the speed the largest |u| on their C plane faces (`c`'s C velocity).
 */
export function residencyOracle(L: BrickLayout, before: BrickSnapshot, want: Uint32Array, tiered: boolean, ring: { dt: number; h: number; gravity: Vec3 },
  c?: { readonly phi: Uint32Array; readonly vel: Uint32Array }): ResidencyOracle {
  const nb = bricksPerAxis(L.lattice), np = nb.map(n => Math.ceil(n / 4));
  const bricks = new Set<number>(), pages = new Set<number>();
  const f = Math.fround, f32 = new Float32Array(1), u32 = new Uint32Array(f32.buffer);
  const asF = (w: number) => { u32[0] = w; return f32[0]!; };
  const g = f(Math.sqrt(f(f(f(ring.gravity[0] * ring.gravity[0]) + f(ring.gravity[1] * ring.gravity[1])) + f(ring.gravity[2] * ring.gravity[2]))));
  let radius = 0;
  // Class 4's C reads (tiered): the C faces of the tiles this head refines from 4h (resMark's nbCFace).
  const velC4 = tiered && c ? [0, 1, 2].map(a => L.base("vel", "C", `U${a}`)) : [];
  const tileIndex4 = (T: number[]) => (T[0]! & 7) | ((T[1]! & 7) << 3) | ((T[2]! & 7) << 6);
  const cFace4 = (T: number[], a: number) => {
    if (!c || T.some((v, b) => v < 0 || 4 * v > L.lattice[b]!)) return 0;
    const cs = before.pages.get(brickKey(T[0]! >> 3, T[1]! >> 3, T[2]! >> 3));
    return cs === undefined ? 0 : asF(c.vel[velC4[a]! + BRICK_STRIDE * cs + tileIndex4(T)]!);
  };
  for (const [key, slot] of before.bricks) {
    const at = L.set.psupport + PSUPPORT_WORDS * slot;
    const act = before.set[L.set.act + slot]!, ps = before.set[at]! & 0xff;
    const flow = f(f(f(asF(before.set[L.set.boundF + slot]!) + f(g * ring.dt)) * ring.dt) / ring.h), o = brickKeyCoord(key).map(v => 8 * v);
    // The classes: owners (ACT.RES), support (PSUPPORT word 0), seeds (ACT.SEED | this census's TRIG); each with its
    // brick reach and, tiered, its page reach (the owners' by RES_MARGIN.ownPages, the others' as their bricks').
    const own = tiered ? RES_MARGIN.own.tiered : RES_MARGIN.own.allH;
    // Tiered, class 4: the tiles this head refines from 4h (WANT h, CMASK 4h), the re-entry seed's tiles, the particle
    // class past the larger of the slot's flow and their C face flow.
    let reMask = 0, reSpeed = 0;
    if (tiered) for (let j = 0; j < 8; j++) {
      const T = [0, 1, 2].map(a => (o[a]! >> 2) + ((j >> a) & 1));
      if (T.some((v, a) => v < 0 || 4 * v >= L.lattice[a]!)) continue;
      const cs = before.pages.get(brickKey(T[0]! >> 3, T[1]! >> 3, T[2]! >> 3));
      if (cs === undefined) continue;
      const i = tileIndex4(T);
      if (((want[L.set.want + CMASK_WORDS * cs + (i >> 5)]! >>> (i & 31)) & 1) === 0) continue;
      if ((before.dir[L.dir.cmask + CMASK_WORDS * cs + (i >> 5)]! >>> (i & 31)) & 1) continue;
      reMask |= 1 << j;
      for (let a = 0; a < 3; a++) { const up = [...T]; up[a]! += 1; reSpeed = Math.max(reSpeed, Math.abs(cFace4(T, a)), Math.abs(cFace4(up, a))); }
    }
    const flowR = Math.max(flow, f(f(f(reSpeed + f(g * ring.dt)) * ring.dt) / ring.h));
    // Tiered (R3): the owners' bricks are their tile ring (RES_MARGIN.own.tiered, no move, half-open), and the
    // tiles with particles (TILE_OCC bits 0..7) are a fourth class, RES_MARGIN.particles + flow.
    const classes: [number, (a: number, side: 0 | 1) => number, ((a: number, side: 0 | 1) => number) | null, boolean][] = [
      [(act & ACT.RES) >>> 10, () => tiered ? own : f(own + flow), tiered ? () => f(RES_MARGIN.ownPages + flow) : null, tiered],
      [ps, (a, side) => f(RES_MARGIN.support + Math.max(asF(before.set[at + 1 + 3 * side + a]!), flow)), null, false],
      [((act & ACT.SEED) >>> 18) | (want[L.set.trig + slot]! & 0xff), () => RES_MARGIN.seed, null, false],
      [tiered ? before.set[L.set.tileOcc + slot]! & 0xff : 0, () => f(RES_MARGIN.particles + flow), null, false],
      [reMask, () => f(RES_MARGIN.particles + flowR), null, false],
    ];
    for (const [m, reach, pageReach, halfOpen] of classes) {
      if (m === 0) continue;
      const tlo = [1, 1, 1], thi = [0, 0, 0];
      for (let t = 0; t < 8; t++) if ((m >> t) & 1) for (let a = 0; a < 3; a++) { const v = (t >> a) & 1; tlo[a] = Math.min(tlo[a]!, v); thi[a] = Math.max(thi[a]!, v); }
      const box = (r: (a: number, side: 0 | 1) => number, open = false) => {
        const b0: number[] = [], b1: number[] = [];
        for (let a = 0; a < 3; a++) {
          const rl = r(a, 0), rh = r(a, 1);
          const lo = Math.floor(f(o[a]! + 4 * tlo[a]! - rl)), hi = Math.ceil(f(o[a]! + 4 * thi[a]! + 4 + rh)) - (open ? 1 : 0);
          b0.push(Math.max(lo >> 3, 0)); b1.push(Math.min(hi >> 3, nb[a]! - 1));
          radius = Math.max(radius, Math.ceil(Math.max(rl, rh) / 4));
        }
        return [b0, b1] as const;
      };
      const [b0, b1] = box(reach, halfOpen);
      for (let z = b0[2]!; z <= b1[2]!; z++) for (let y = b0[1]!; y <= b1[1]!; y++) for (let x = b0[0]!; x <= b1[0]!; x++) bricks.add(brickKey(x, y, z));
      if (!tiered) continue;
      const [p0, p1] = pageReach ? box(pageReach) : [b0, b1];
      for (let z = p0[2]! >> 2; z <= Math.min(p1[2]! >> 2, np[2]! - 1); z++) for (let y = p0[1]! >> 2; y <= Math.min(p1[1]! >> 2, np[1]! - 1); y++)
        for (let x = p0[0]! >> 2; x <= Math.min(p1[0]! >> 2, np[0]! - 1); x++) pages.add(brickKey(x, y, z));
    }
  }
  // The page sources (tiered; resMark's pass 0 over C_ALL after F_ALL).
  if (tiered) {
    if (!c) throw new Error("residencyOracle: the tiered oracle needs the C φ and velocity");
    const phiC = L.base("phi", "C", "PHI"), velC = [0, 1, 2].map(a => L.base("vel", "C", `U${a}`));
    const pageOf = (T: number[]) => before.pages.get(brickKey(T[0]! >> 3, T[1]! >> 3, T[2]! >> 3));
    const tileIndex = (T: number[]) => (T[0]! & 7) | ((T[1]! & 7) << 3) | ((T[2]! & 7) << 6);
    const outside = (T: number[]) => T.some((v, a) => v < 0 || 4 * v > L.lattice[a]!);
    const cVertex = (T: number[]) => { if (outside(T)) return Infinity; const cs = pageOf(T); return cs === undefined ? Infinity : asF(c.phi[phiC + BRICK_STRIDE * cs + tileIndex(T)]!); };
    const cFace = (T: number[], a: number) => { if (outside(T)) return 0; const cs = pageOf(T); return cs === undefined ? 0 : asF(c.vel[velC[a]! + BRICK_STRIDE * cs + tileIndex(T)]!); };
    for (const [pk, cs] of before.pages) {
      const P = brickKeyCoord(pk), tlo = [8, 8, 8], thi = [0, 0, 0];
      let speed = 0, any = false;
      for (let i = 0; i < 512; i++) {
        const t = [i & 7, (i >> 3) & 7, i >> 6], T = t.map((v, a) => 8 * P[a]! + v);
        if (T.some((v, a) => v < 0 || 4 * v >= L.lattice[a]!)) continue;
        if ((before.dir[L.dir.cmask + CMASK_WORDS * cs + (i >> 5)]! >>> (i & 31)) & 1) continue;
        let mn = Infinity;
        for (let q = 0; q < 8; q++) mn = Math.min(mn, cVertex([T[0]! + (q & 1), T[1]! + ((q >> 1) & 1), T[2]! + (q >> 2)]));
        if (!(mn < 0)) continue;
        any = true;
        for (let a = 0; a < 3; a++) {
          tlo[a] = Math.min(tlo[a]!, t[a]!); thi[a] = Math.max(thi[a]!, t[a]!);
          const up = [...T]; up[a]! += 1;
          speed = Math.max(speed, Math.abs(cFace(T, a)), Math.abs(cFace(up, a)));
        }
      }
      if (!any) continue;
      const r = f(f(f(f(speed + f(g * ring.dt)) * ring.dt) / ring.h) + RES_MARGIN.page);
      radius = Math.max(radius, Math.ceil(r / 4));
      const p0: number[] = [], p1: number[] = [];
      for (let a = 0; a < 3; a++) {
        p0.push(Math.max(Math.floor(f(32 * P[a]! + 4 * tlo[a]! - r)) >> 5, 0));
        p1.push(Math.min(Math.ceil(f(32 * P[a]! + 4 * thi[a]! + 4 + r)) >> 5, np[a]! - 1));
      }
      for (let z = p0[2]!; z <= p1[2]!; z++) for (let y = p0[1]!; y <= p1[1]!; y++) for (let x = p0[0]!; x <= p1[0]!; x++) pages.add(brickKey(x, y, z));
    }
  }
  // censusDecide's ring (tiered): the bricks of a WANT tile's 27 tiles (R3).
  if (tiered) for (const [pk, cs] of before.pages) {
    const P = brickKeyCoord(pk);
    for (let i = 0; i < 512; i++) {
      if (((want[L.set.want + CMASK_WORDS * cs + (i >> 5)]! >>> (i & 31)) & 1) === 0) continue;
      const T = [i & 7, (i >> 3) & 7, i >> 6].map((v, a) => 8 * P[a]! + v);
      if (T.some((v, a) => 4 * v >= L.lattice[a]!)) continue;
      for (let d = 0; d < 8; d++) {
        const b = [((T[0]! - 1) >> 1) + (d & 1), ((T[1]! - 1) >> 1) + ((d >> 1) & 1), ((T[2]! - 1) >> 1) + (d >> 2)];
        if (b.every((v, a) => v >= 0 && v < nb[a]!)) bricks.add(brickKey(b[0]!, b[1]!, b[2]!));
      }
    }
  }
  for (const k of bricks) { const b = brickKeyCoord(k); pages.add(brickKey(b[0] >> 2, b[1] >> 2, b[2] >> 2)); }
  return { bricks, pages, radius };
}

export interface PressureOracle { readonly X: Map<string, number>; readonly iterations: number; readonly residual: number }

/**
 * The system of brick-abi.md §6.1 over the liquid cells of the lattice that
 * `cells` lists, in Float64: DIAG X - sum K X_nbr = RHS, X = 0 at air.
 * `phi(v)` gives vertex phi (h), `us(a, c)` the lower-face u* along a.
 */
export function pressureOracle(lattice: Vec3, h: number, cells: Iterable<Vec3>, phi: (v: Vec3) => number, us: (a: number, c: Vec3) => number): PressureOracle {
  const inLattice = (c: Vec3) => c.every((v, a) => v >= 0 && v < lattice[a]!);
  const centre = (c: Vec3) => { let s = 0; for (let j = 0; j < 8; j++) s += phi([c[0] + (j & 1), c[1] + ((j >> 1) & 1), c[2] + (j >> 2)]); return s / 8; };
  const open = (c: Vec3, a: number) => { const t = [...c] as [number, number, number]; t[a] = 0; return inLattice(t) && c[a]! > 0 && c[a]! < lattice[a]! ? 1 : 0; };
  const id = (c: Vec3) => c.join(",");
  const rows: { c: Vec3; diag: number; rhs: number; nbr: { j: string; k: number }[] }[] = [], index = new Map<string, number>();
  for (const c of cells) {
    if (!inLattice(c)) continue;
    const pc = centre(c);
    if (!liquidPhi(pc)) continue;
    index.set(id(c), rows.length);
    let diag = 0, div = 0;
    const nbr: { j: string; k: number }[] = [];
    for (let a = 0; a < 3; a++) {
      const lo = [...c] as [number, number, number], up = [...c] as [number, number, number]; lo[a]!--; up[a]!++;
      const al = open(c, a), au = open(up, a);
      if (al) { const pn = centre(lo); if (liquidPhi(pn)) { diag += al; nbr.push({ j: id(lo), k: al }); } else diag += al / theta(pc, pn); }
      if (au) { const pu = centre(up); if (liquidPhi(pu)) { diag += au; nbr.push({ j: id(up), k: au }); } else diag += au / theta(pc, pu); }
      div += au * us(a, up) - al * us(a, c);
    }
    rows.push({ c, diag, rhs: -h * div, nbr });
  }
  const n = rows.length, x = new Float64Array(n), r = new Float64Array(n), z = new Float64Array(n), d = new Float64Array(n), q = new Float64Array(n);
  const apply = (v: Float64Array, out: Float64Array) => rows.forEach((row, i) => { let s = row.diag * v[i]!; for (const e of row.nbr) { const j = index.get(e.j); if (j !== undefined) s -= e.k * v[j]!; } out[i] = s; });
  rows.forEach((row, i) => { r[i] = row.rhs; z[i] = r[i]! / row.diag; d[i] = z[i]!; });
  const dot = (a: Float64Array, b: Float64Array) => a.reduce((s, v, i) => s + v * b[i]!, 0);
  const norm0 = Math.max(...r.map(Math.abs), 1e-300);
  let rz = dot(r, z), it = 0, res = norm0;
  for (; it < 10 * n + 100 && res > 1e-14 * norm0; it++) {
    apply(d, q);
    const alpha = rz / dot(d, q);
    for (let i = 0; i < n; i++) { x[i] += alpha * d[i]!; r[i] -= alpha * q[i]!; z[i] = r[i]! / rows[i]!.diag; }
    const rz2 = dot(r, z), beta = rz2 / rz; rz = rz2;
    for (let i = 0; i < n; i++) d[i] = z[i]! + beta * d[i]!;
    res = Math.max(...r.map(Math.abs));
  }
  const X = new Map<string, number>();
  rows.forEach((row, i) => X.set(id(row.c), x[i]!));
  return { X, iterations: it, residual: res };
}

export const ORACLE_PHI_AIR = PHI_AIR;
