/**
 * CPU oracles for the brick engine (tests only):
 * - `brickSetOracle`: the reach rule's resident set after one setMark and
 *   setUpdate, from the state the GPU held before the step;
 * - `pressureOracle`: the pressure slot contract's system (brick-abi.md §6.1),
 *   assembled in Float64 from the GPU's phi and u*, and solved to round-off by
 *   Jacobi-preconditioned CG.
 */
import { BRICK_STRIDE, PHI_AIR, brickKeyCoord, cellLocal } from "./brick-abi";
import { reachSet, type Vec3 } from "./brick-engine";
import type { BrickSnapshot } from "./brick-readback";

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
