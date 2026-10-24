/** The 16h bulk tier's constants and its CPU twin (bulkTier "16h";
 * docs/plans/tiered-adaptivity-huge-grids-2026-10-24.md, W4).
 *
 * A region (16h: 4^3 tiles, 8 to a block) of submerged liquid bulk owns one
 * pressure row and six face velocities. On today's storage its data stays in
 * its 64 tiles' 4h storage, constrained: the 64 native rows of the pressure
 * arena hold one pressure (their sum is the 16h row), and the projection
 * writes every 4h face of the region from the region's six face means
 * (UniformBulkTier). W6 moves it into a 16h slot without changing the rules.
 *
 * The record (one storage buffer, R regions):
 *  [0, R)        flags: bit 0 member
 *  [R, 2R)       hold: consecutive passing frames of a non-member
 *  [2R, 3R)      the last error test, err / threshold (f32 bits)
 *  [3R, 3R+8)    counts (UNIFORM_BULK_COUNT)
 *  [3R+8]        the member list's length, the list after it
 *  [4R+16, ...)  seam heads: 96 a region, 16 a member face (pNative)
 * The faces texture (r32float, UNIFORM_BULK_FACE_WIDTH wide), 128 texels a
 * region: [0] member flag, [1+f] face f's mean velocity, [8+16f+j] subface
 * j's projected value (3e38: the 4h projection's own). */

export const UNIFORM_BULK_HOLD_FRAMES = 30;
/** Restrict-prolong error over local speed below which a region may coarsen (and twice it refines). */
export const UNIFORM_BULK_TOLERANCE = 0.1;
/** The error floor, in h per step. */
export const UNIFORM_BULK_FLOOR = 0.002;
/** Open tiles of the 3^3 region neighbourhood sit at least this deep (min h). */
export const UNIFORM_BULK_DEPTH_CELLS = 5.5;
/** A member's fastest face moves at most this many h a step (velocity-projected reach). */
export const UNIFORM_BULK_REACH_CELLS = 8;
export const UNIFORM_BULK_FACE_TEXELS = 128;
export const UNIFORM_BULK_FACE_WIDTH = 4096;
/** Sentinel face value: no override (the head table's "no seam" too). */
export const UNIFORM_BULK_NONE = 3e38;
export const UNIFORM_BULK_NONE_BITS = 0x7f61b1e6;
export const UNIFORM_BULK_COUNT = { members: 0, eligible: 1, structural: 2, error: 3, entered: 4, left: 5 } as const;
export const UNIFORM_BULK_HOLD = (regions: number) => regions;
export const UNIFORM_BULK_ERROR = (regions: number) => 2 * regions;
export const UNIFORM_BULK_HEADER = (regions: number) => 3 * regions;
export const UNIFORM_BULK_LIST = (regions: number) => 3 * regions + 8;
export const UNIFORM_BULK_HEADS = (regions: number) => 4 * regions + 16;
export const UNIFORM_BULK_WORDS = (regions: number) => 100 * regions + 16;
/** Native coupling per V/h4^2 of a 4h subface: 4h spacing, member to member (16h), member to native (10h). */
export const UNIFORM_BULK_COUPLING = { native: 64, member: 16, seam: 25.6 } as const;

// ---------------------------------------------------------------------------
// The CPU twin. Units: the 4h cell is 1 and dt/density is 1, so a native
// face's velocity is u* - (p_H - p_L); every face V is 1 and the domain's
// faces are walls (no flux). Cells are 4h cells, regions 4^3 of them.

export type BulkVec = [number, number, number];

/** Restrict a region's faces on axis a to its two face means, prolong them
 * back linearly: the selection's error test and the projection's interior. */
export function uniformBulkProlong(lo: number, hi: number, plane: number): number { return lo + 0.25 * plane * (hi - lo); }
/** The selection's error of one axis's 5x16 faces (plane k between region cells k-1 and k). */
export function uniformBulkAxisError(faces: ArrayLike<number>): number {
  let lo = 0, hi = 0;
  for (let j = 0; j < 16; j++) { lo += faces[j]!; hi += faces[64 + j]!; }
  lo /= 16; hi /= 16;
  let error = 0;
  for (let k = 1; k < 4; k++) for (let j = 0; j < 16; j++) error = Math.max(error, Math.abs(faces[16 * k + j]! - uniformBulkProlong(lo, hi, k)));
  return error;
}

/** Hysteresis (UniformBulkTier's lane 0). */
export function uniformBulkStep(state: { member: boolean; hold: number }, structural: boolean, error: number, threshold: number): { member: boolean; hold: number } {
  if (state.member) return structural || error > 2 * threshold ? { member: false, hold: 0 } : state;
  if (!structural && error < threshold) { const hold = Math.min(state.hold + 1, UNIFORM_BULK_HOLD_FRAMES); return { member: hold >= UNIFORM_BULK_HOLD_FRAMES, hold }; }
  return { member: false, hold: 0 };
}

/** A tile of the structural rule. */
export interface UniformBulkTile { kind: "resident" | "air" | "solid"; band?: boolean; hot?: boolean; open?: number; phi?: number }
/** The structural rule of region `r` (tile origin 4r) on a tile grid (tile(x,y,z), undefined outside: SOLID). */
export function uniformBulkStructural(tile: (t: BulkVec) => UniformBulkTile | undefined, region: BulkVec, minH: number): boolean {
  for (let z = -4; z < 8; z++) for (let y = -4; y < 8; y++) for (let x = -4; x < 8; x++) {
    const own = x >= 0 && x < 4 && y >= 0 && y < 4 && z >= 0 && z < 4;
    const t = tile([4 * region[0] + x, 4 * region[1] + y, 4 * region[2] + z]) ?? { kind: "solid" as const };
    if (t.kind !== "resident") { if (own || t.kind === "air") return false; continue; }
    if (t.band || t.hot) return false;
    const open = t.open ?? 1;
    if (open <= 1e-5) { if (own) return false; continue; }
    if (open < 1 - 1e-5 || (t.phi ?? -Infinity) > -UNIFORM_BULK_DEPTH_CELLS * minH) return false;
  }
  return true;
}

/** A composite pressure system of 4h cells (n, multiples of 4) with member
 * regions, as pNative and the bulk faces form it. */
export class UniformBulkComposite {
  readonly regions: BulkVec;
  readonly cells: number;
  private readonly bases: number[];
  constructor(readonly n: BulkVec, readonly members: ReadonlySet<number>) {
    if (n.some(v => v % 4 !== 0)) throw new Error("Whole regions only");
    this.regions = [n[0] / 4, n[1] / 4, n[2] / 4];
    this.cells = n[0] * n[1] * n[2];
    this.bases = [0];
    for (let a = 0; a < 3; a++) { const d: BulkVec = [...n]; d[a] += 1; this.bases.push(this.bases[a]! + d[0] * d[1] * d[2]); }
  }
  get faces(): number { return this.bases[3]!; }
  cell(c: BulkVec): number { return c[0] + this.n[0] * (c[1] + this.n[1] * c[2]); }
  coord(k: number): BulkVec { const [nx, ny] = this.n; return [k % nx, Math.floor(k / nx) % ny, Math.floor(k / (nx * ny))]; }
  region(c: BulkVec): number { const [rx, ry] = this.regions; return (c[0] >> 2) + rx * ((c[1] >> 2) + ry * (c[2] >> 2)); }
  member(c: BulkVec): boolean { return this.members.has(this.region(c)); }
  inside(c: BulkVec): boolean { return c.every((v, a) => v >= 0 && v < this.n[a]!); }
  /** The face on axis a below cell c (c[a] in 0..n[a]). */
  face(a: number, c: BulkVec): number { const d: BulkVec = [...this.n]; d[a] += 1; return this.bases[a]! + c[0] + d[0] * (c[1] + d[1] * c[2]); }
  /** nBulkLink's coupling (per V/h4^2). */
  coupling(k: BulkVec, n: BulkVec): number {
    const mk = this.member(k), mn = this.member(n);
    if (mk && mn) return this.region(k) === this.region(n) ? 0 : UNIFORM_BULK_COUPLING.member;
    return mk || mn ? UNIFORM_BULK_COUPLING.seam : UNIFORM_BULK_COUPLING.native;
  }
  /** nBulkHead: native q's start slope over its native neighbours on the axes tangential to a, times its offset from its region's centre. */
  head(start: Float64Array, q: BulkVec, a: number): number {
    let head = 0;
    for (let b = 0; b < 3; b++) {
      if (b === a) continue;
      let rise = 0, count = 0;
      for (const s of [-1, 1]) {
        const m: BulkVec = [...q]; m[b] += s;
        if (this.inside(m) && !this.member(m)) { rise += s * (start[this.cell(m)]! - start[this.cell(q)]!); count++; }
      }
      if (count) head += rise / count * ((q[b] & 3) - 1.5);
    }
    return head;
  }
  /** The offset o of link k -> n (rows: c (p_n - p_k + o)). */
  offset(start: Float64Array, k: BulkVec, n: BulkVec, a: number): number {
    const mk = this.member(k), mn = this.member(n);
    if (mk === mn) return 0;
    return mk ? -this.head(start, n, a) : this.head(start, k, a);
  }
  private *links(k: BulkVec): Generator<{ n: BulkVec; a: number; s: number }> {
    for (let a = 0; a < 3; a++) for (const s of [-1, 1]) { const n: BulkVec = [...k]; n[a] += s; if (this.inside(n)) yield { n, a, s }; }
  }
  /** Rows (pNative) of pressure p: 64 R + sum c (p_n - p_k + o), R = -sum s u* (the 4h cell's divergence). */
  rows(p: Float64Array, start: Float64Array, ustar: Float64Array): Float64Array {
    const r = new Float64Array(this.cells);
    for (let k = 0; k < this.cells; k++) {
      const c = this.coord(k); let row = 0;
      for (const { n, a, s } of this.links(c)) {
        row -= 64 * s * ustar[this.face(a, s > 0 ? n : c)]!;
        const coupling = this.coupling(c, n);
        if (coupling) row += coupling * (p[this.cell(n)]! - p[k]! + this.offset(start, c, n, a));
      }
      r[k] = row;
    }
    return r;
  }
  /** The constrained residual: native rows, and on each member cell its region's 16h row (the sum) over 64. */
  residual(p: Float64Array, start: Float64Array, ustar: Float64Array): Float64Array {
    const r = this.rows(p, start, ustar), sums = new Map<number, number>();
    for (let k = 0; k < this.cells; k++) { const c = this.coord(k); if (this.member(c)) sums.set(this.region(c), (sums.get(this.region(c)) ?? 0) + r[k]!); }
    for (let k = 0; k < this.cells; k++) { const c = this.coord(k); if (this.member(c)) r[k] = sums.get(this.region(c))! / 64; }
    return r;
  }
  /** Solve the constrained system by CG on its unknowns (native cells, one a member region). */
  solve(start: Float64Array, ustar: Float64Array, iterations = 4000, tolerance = 1e-12): Float64Array {
    const ids = new Int32Array(this.cells), byRegion = new Map<number, number>(); let unknowns = 0;
    for (let k = 0; k < this.cells; k++) {
      const c = this.coord(k);
      if (this.member(c)) { const r = this.region(c); if (!byRegion.has(r)) byRegion.set(r, unknowns++); ids[k] = byRegion.get(r)!; }
      else ids[k] = unknowns++;
    }
    const expand = (x: Float64Array) => { const p = new Float64Array(this.cells); for (let k = 0; k < this.cells; k++) p[k] = x[ids[k]!]!; return p; };
    const gather = (r: Float64Array) => { const g = new Float64Array(unknowns); for (let k = 0; k < this.cells; k++) g[ids[k]!] += r[k]!; return g; };
    const r0 = gather(this.rows(new Float64Array(this.cells), start, ustar));
    const apply = (x: Float64Array) => { const g = gather(this.rows(expand(x), start, ustar)); for (let i = 0; i < unknowns; i++) g[i] = r0[i]! - g[i]!; return g; };
    const x = new Float64Array(unknowns), r = Float64Array.from(r0), d = Float64Array.from(r);
    const dot = (u: Float64Array, v: Float64Array) => { let s = 0; for (let i = 0; i < u.length; i++) s += u[i]! * v[i]!; return s; };
    let rr = dot(r, r); const stop = tolerance * tolerance * Math.max(rr, 1e-300);
    for (let it = 0; it < iterations && rr > stop; it++) {
      const q = apply(d), alpha = rr / dot(d, q);
      for (let i = 0; i < unknowns; i++) { x[i] += alpha * d[i]!; r[i] -= alpha * q[i]!; }
      const next = dot(r, r); for (let i = 0; i < unknowns; i++) d[i] = r[i]! + next / rr * d[i]!; rr = next;
    }
    return expand(x);
  }
  /** The projected faces as the 4h projection writes them: native to native
   * its own; at a member's boundary the bulk faces' subface values (a seam:
   * the 10h gradient on the head; member to member: the 16h gradient on the
   * face's mean u*); inside a member the prolonged face means. */
  project(p: Float64Array, start: Float64Array, ustar: Float64Array): { u: Float64Array; means: Map<number, number[]> } {
    const u = new Float64Array(this.faces), means = new Map<number, number[]>();
    const value = (a: number, low: BulkVec, high: BulkVec): number => {
      const ml = this.member(low), mh = this.member(high), f = this.face(a, high);
      const pl = p[this.cell(low)]!, ph = p[this.cell(high)]!;
      if (!ml && !mh) return ustar[f]! - (ph - pl);
      if (ml && mh) {
        let star = 0; const b = (a + 1) % 3, c = (a + 2) % 3;
        for (let j = 0; j < 16; j++) { const q: BulkVec = [...high]; q[b] = (q[b] & ~3) + (j & 3); q[c] = (q[c] & ~3) + (j >> 2); star += ustar[this.face(a, q)]!; }
        return star / 16 - (ph - pl) / 4;
      }
      const head = this.head(start, ml ? high : low, a);
      return ustar[f]! - ((ph + (mh ? head : 0)) - (pl + (ml ? head : 0))) * UNIFORM_BULK_COUPLING.seam / UNIFORM_BULK_COUPLING.native;
    };
    for (let a = 0; a < 3; a++) for (let k = 0; k < this.cells; k++) {
      const high = this.coord(k), low: BulkVec = [...high]; low[a] -= 1;
      if (!this.inside(low) || (this.member(low) && this.member(high) && this.region(low) === this.region(high))) continue;
      u[this.face(a, high)] = value(a, low, high);
    }
    for (const r of this.members) {
      const o: BulkVec = [4 * (r % this.regions[0]), 4 * (Math.floor(r / this.regions[0]) % this.regions[1]), 4 * Math.floor(r / (this.regions[0] * this.regions[1]))];
      const m: number[] = [];
      for (let a = 0; a < 3; a++) for (const side of [0, 4]) {
        let s = 0; const b = (a + 1) % 3, c = (a + 2) % 3;
        for (let j = 0; j < 16; j++) { const q: BulkVec = [...o]; q[a] += side; q[b] += j & 3; q[c] += j >> 2; if (q[a] > 0 && q[a] < this.n[a]) s += u[this.face(a, q)]!; }
        m.push(s / 16);
      }
      means.set(r, m);
      for (let a = 0; a < 3; a++) for (let plane = 1; plane < 4; plane++) for (let j = 0; j < 16; j++) {
        const b = (a + 1) % 3, c = (a + 2) % 3, q: BulkVec = [...o]; q[a] += plane; q[b] += j & 3; q[c] += j >> 2;
        u[this.face(a, q)] = uniformBulkProlong(m[2 * a]!, m[2 * a + 1]!, plane);
      }
    }
    return { u, means };
  }
  /** Each cell's divergence of faces u (walls 0). */
  divergence(u: Float64Array): Float64Array {
    const d = new Float64Array(this.cells);
    for (let k = 0; k < this.cells; k++) {
      const c = this.coord(k); let s = 0;
      for (let a = 0; a < 3; a++) {
        const up: BulkVec = [...c]; up[a] += 1;
        if (up[a] < this.n[a]) s += u[this.face(a, up)]!;
        if (c[a] > 0) s -= u[this.face(a, c)]!;
      }
      d[k] = s;
    }
    return d;
  }
}
