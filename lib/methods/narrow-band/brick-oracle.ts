/**
 * CPU oracles for the brick engine (tests only):
 * - `brickSetOracle`: the reach rule's resident set after one setMark and
 *   setUpdate, from the state the GPU held before the step;
 * - `residencyOracle`: the head's residency (brick-residency.wgsl.ts, the old pageSeed and lrAsk) from this head's
 *   census words, the particle support the completed step left, and the set before the head;
 * - `pressureOracle`: the pressure slot contract's system (brick-abi.md §6.1),
 *   assembled in Float64 from the GPU's phi and u*, and solved to round-off by
 *   Jacobi-preconditioned CG.
 */
import { BRICK_STRIDE, CMASK_WORDS, PHI_AIR, PSUPPORT_WORDS, type BrickLayout, brickKey, brickKeyCoord, bricksPerAxis, cellLocal } from "./brick-abi";
import { reachSet, type Vec3 } from "./brick-engine";
import type { BrickSnapshot } from "./brick-readback";
import { CENSUS } from "./brick-census.wgsl";
import { RES_STAGES } from "./brick-residency.wgsl";
import { residencyStageMeets } from "../uniform/uniform-local-residency";

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

const NC_CROSSES = 0x10000000, NC_WET = 0x20000000;

export interface ResidencyOracle {
  /** Per C slot of the set before the head: its 8 regions' seed masks, 2 words a (region, stage), region-major. */
  readonly masks: Map<number, Uint32Array>;
  /** The F bricks and C pages (keys) the head leaves resident. */
  readonly bricks: Set<number>;
  readonly pages: Set<number>;
}

/**
 * The head's residency on the CPU. `before` is the set the head starts from (its PSUPPORT, CMASK, pages); `census`
 * the lane buffer and `want` the set buffer after the head (this census's FIN, NEAR and WANT words); `radii(cs, r,
 * stage)` the GPU's record radii ([-x,-y,-z,+x,+y,+z] tiles), which the caller checks against the flow on its own.
 * Seeds follow pageSeedOf; a destination is kept when residencyStageMeets says so for a stage requested at its
 * granularity (all-h: every stage at F; tiered: particles and ownership at F, every stage at C), plus, tiered, the
 * 27-brick ring of every WANT tile (censusDecide's keep and candidates).
 */
export function residencyOracle(L: BrickLayout, before: BrickSnapshot, census: Uint32Array, want: Uint32Array, radii: (cs: number, r: number, stage: number) => number[], tiered: boolean): ResidencyOracle {
  const masks = new Map<number, Uint32Array>(), nb = bricksPerAxis(L.lattice), np = nb.map(n => Math.ceil(n / 4));
  const bricks = new Set<number>(), pages = new Set<number>();
  const regions: { R: number[]; cs: number; r: number }[] = [];
  for (const [pk, cs] of before.pages) {
    const P = brickKeyCoord(pk), m = new Uint32Array(2 * 8 * RES_STAGES), o = CENSUS.STRIDE * cs;
    for (let i = 0; i < 512; i++) {
      const t = [i & 7, (i >> 3) & 7, i >> 6], T = t.map((v, a) => 8 * P[a]! + v);
      if (T.some((v, a) => 4 * v >= L.lattice[a]!)) continue;
      const fs = before.dir[L.dir.child + 64 * cs + ((t[0]! >> 1) | ((t[1]! >> 1) << 2) | ((t[2]! >> 1) << 4))]!;
      const sample = fs < L.cap.f && ((before.set[L.set.psupport + PSUPPORT_WORDS * fs]! >>> ((t[0]! & 1) | ((t[1]! & 1) << 1) | ((t[2]! & 1) << 2))) & 1) === 1;
      const near = census[o + CENSUS.NEAR + i] !== 0, band = ((want[L.set.want + CMASK_WORDS * cs + (i >> 5)]! >>> (i & 31)) & 1) === 1;
      if (!near && !band && !sample) continue;
      const fin = census[o + CENSUS.FIN + i]!, wet = (fin & NC_WET) !== 0, crossing = (fin & NC_CROSSES) !== 0;
      const width1 = tiered && ((before.dir[L.dir.cmask + CMASK_WORDS * cs + (i >> 5)]! >>> (i & 31)) & 1) === 1;
      const seeds = [width1 || band, wet || sample, wet || sample, crossing || sample, sample];
      const r = (t[0]! >> 2) | ((t[1]! >> 2) << 1) | ((t[2]! >> 2) << 2), q = (t[0]! & 3) | ((t[1]! & 3) << 2) | ((t[2]! & 3) << 4);
      seeds.forEach((on, s) => { if (on) m[2 * (RES_STAGES * r + s) + (q >> 5)]! |= 1 << (q & 31); });
      // censusDecide's ring (tiered): a WANT tile's brick and its in-range 27-neighbours.
      if (tiered && band) for (let d = 0; d < 27; d++) {
        const b = [(T[0]! >> 1) + (d % 3) - 1, (T[1]! >> 1) + (Math.floor(d / 3) % 3) - 1, (T[2]! >> 1) + Math.floor(d / 9) - 1];
        if (b.every((v, a) => v >= 0 && v < nb[a]!)) bricks.add(brickKey(b[0]!, b[1]!, b[2]!));
      }
    }
    masks.set(cs, m);
    for (let r = 0; r < 8; r++) regions.push({ R: [0, 1, 2].map(a => 8 * P[a]! + 4 * ((r >> a) & 1)), cs, r });
  }
  const fLevel = tiered ? [4, 0] : [0, 1, 2, 3, 4];
  const meets = (stages: number[], lo: number[], size: number) => regions.some(({ R, cs, r }) => stages.some(s => {
    const m = masks.get(cs)!, mask = [m[2 * (RES_STAGES * r + s)]!, m[2 * (RES_STAGES * r + s) + 1]!];
    if ((mask[0]! | mask[1]!) === 0) return false;
    const low = lo.map((v, a) => v - R[a]!);
    return residencyStageMeets(s, mask, radii(cs, r, s), low, low.map(v => v + size - 1));
  }));
  for (let z = 0; z < nb[2]; z++) for (let y = 0; y < nb[1]; y++) for (let x = 0; x < nb[0]; x++)
    if (meets(fLevel, [2 * x, 2 * y, 2 * z], 2)) bricks.add(brickKey(x, y, z));
  for (const k of bricks) { const b = brickKeyCoord(k); pages.add(brickKey(b[0] >> 2, b[1] >> 2, b[2] >> 2)); }
  if (tiered) for (let z = 0; z < np[2]!; z++) for (let y = 0; y < np[1]!; y++) for (let x = 0; x < np[0]!; x++)
    if (meets([0, 1, 2, 3, 4], [8 * x, 8 * y, 8 * z], 8)) pages.add(brickKey(x, y, z));
  return { masks, bricks, pages };
}

/** A residency record's radii, [-x,-y,-z,+x,+y,+z] tiles (lrUnpack of words 2 and 3). */
export function recordRadii(census: Uint32Array, cs: number, r: number, stage: number): number[] {
  const at = CENSUS.STRIDE * cs + CENSUS.RES + 4 * (RES_STAGES * r + stage), a = census[at + 2]!, b = census[at + 3]!;
  return [a & 255, (a >> 8) & 255, (a >> 16) & 255, b & 255, (b >> 8) & 255, (b >> 16) & 255];
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
