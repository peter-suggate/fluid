/** Host reference of the pressure root's launches (UniformMixedPressureCycles
 * and the setup kernels of uniformPressureHierarchyWGSL), one launch at a
 * time on the buffers a device holds: tools/probe-root-levels.ts reads the
 * buffers back before and after each launch and holds the device to this.
 *
 * It restates what each launch computes, never how the kernel reaches it:
 * every table word is found through the host's own layout functions (the
 * page table at the mirror's first word, the block records after it), a
 * neighbour through the page table and the block's record (never the block
 * neighbour rows, which the probe checks on their own), a wall slot through
 * uniformFaceLevel, a level's fields through the arena words the plan gives.
 * A smoothing launch is whole-level Jacobi iterations; the tiled kernel's
 * halo, job strides and lanes are the device's to get right. */
import type { UniformPressureLevelWords } from "../lib/methods/uniform/pressure-plan";
import { UNIFORM_PAGE_TABLE_WORDS, uniformFaceLevel } from "../lib/methods/uniform/uniform-lattice-address";
import { UNIFORM_MIXED_ROOT_SMOOTHING } from "../lib/methods/uniform/uniform-mixed-pressure-cycles";
import { UNIFORM_MIXED_THETA_MIN } from "../lib/methods/uniform/uniform-mixed-pressure-surface.wgsl";
import { UNIFORM_MIXED_ROOT_FAR_CELLS } from "../lib/methods/uniform/uniform-mixed-root-owners";
import { uniformMixedResidencyWord } from "../lib/methods/uniform/uniform-mixed-topology.wgsl";

/** What does not change over a solve. pages: the page mirror (the page
 * table, then a record a block). phi: the root's phi, a word a tile key.
 * record: with a solid record, four words a root row. words: the mixed
 * fields, in words of the mixed range. */
export interface RootLevelsFacts {
  blocks: number; h: readonly [number, number, number];
  pages: Uint32Array; support: Uint32Array; phi: Float32Array; record?: Float32Array;
  levels: readonly UniformPressureLevelWords[];
  words: Record<"p" | "b0" | "b1" | "m0" | "m1" | "backup" | "residual", number>;
  listed: boolean; tiled: number;
}
/** What a launch reads and writes: the arena's mixed range, its levels'
 * range (from level 0's first word), the rounding pairs and the job list. */
export interface RootLevelsState { mixed: Float32Array; native: Float32Array; rounding: Float32Array; list: Uint32Array }

const FLT = 3.402823e38, NONE = 0xffffffff, ROW = 256, ODD = 128;
/** The coarsest level's solve: red-black half-sweeps and their relaxation. */
const HALVES = 32, OMEGA = 1.6;
const STEP: readonly (readonly [number, number, number])[] = [[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]];
const sum = (v: ArrayLike<number>): number => { let s = 0; for (let i = 0; i < v.length; i++) s += v[i]!; return s; };

interface Level { l: number; side: number; per: number; cells: number; rows: number; p: [number, number]; b: [number, number]; m: [number, number]; c: number; phi: [number, number]; v: number; sv: number }

export function rootLevelsReference(f: RootLevelsFacts) {
  const N = f.blocks, keys = 512 * N, minH = Math.min(...f.h), far = UNIFORM_MIXED_ROOT_FAR_CELLS * Math.max(...f.h), solid = !!f.record;
  const last = f.levels.length - 1, base = f.levels[0]!.pressure[0], w = f.words;
  const L: Level[] = f.levels.map((x, l) => ({ l, side: x.side, per: x.side ** 3, cells: x.cells, rows: x.cells + x.slots,
    p: [x.pressure[0] - base, x.pressure[1] - base], b: [x.rhs[0] - base, x.rhs[1] - base], m: [x.minimum[0] - base, x.minimum[1] - base],
    c: x.coefficients - base, phi: [x.phi[0] - base, x.phi[1] - base], v: x.topology - base, sv: x.wall - base }));

  // ---- The pool: a page's word, a block's page, the word of a page beside a block's.
  const record = (b: number): number => f.pages[UNIFORM_PAGE_TABLE_WORDS + b]!;
  const stored = (b: number): boolean => b >= 1 && b <= N - 2 && record(b) !== NONE;
  const pageOf = (b: number): [number, number, number] => { const r = record(b); return [r & 63, (r >>> 6) & 63, (r >>> 12) & 63]; };
  const around = (b: number, d: readonly number[]): number => {
    const p = pageOf(b), x = p[0] + d[0]!, y = p[1] + d[1]!, z = p[2] + d[2]!;
    return x < 0 || y < 0 || z < 0 || x > 63 || y > 63 || z > 63 ? NONE : f.pages[x | (y << 6) | (z << 12)]!;
  };
  const across = (b: number, face: number): number => around(b, STEP[face]!);
  const wallWord = (word: number): boolean => word >= N - 1;

  // ---- A level's rows.
  const cellRow = (V: Level, b: number, c: readonly number[]): number => b * V.per + c[0]! + V.side * (c[1]! + V.side * c[2]!);
  const slotRow = (V: Level, b: number, face: number, c: readonly number[]): number => V.cells + uniformFaceLevel(b, face, V.l, c[0]!, c[1]!, c[2]!);
  const blockOf = (V: Level, j: number): number => j < V.cells ? Math.floor(j / V.per) : Math.floor((j - V.cells) / (6 * V.side * V.side));
  const local = (V: Level, j: number): [number, number, number] => [j % V.side, Math.floor(j / V.side) % V.side, Math.floor(j / (V.side * V.side)) % V.side];
  const faceOf = (V: Level, j: number): number => Math.floor((j - V.cells) / (V.side * V.side)) % 6;
  /** A slot's place on its face. */
  const slotUV = (V: Level, j: number): [number, number] => { const s = j - V.cells; return [s % V.side, Math.floor(s / V.side) % V.side]; };
  const slotAt = (V: Level, b: number, face: number, u: number, v: number): number => V.cells + (6 * b + face) * V.side * V.side + u + V.side * v;
  const ownerOf = (V: Level, j: number): number => {
    const face = faceOf(V, j), axis = face >> 1, [u, v] = slotUV(V, j), c = [0, 0, 0];
    c[axis] = face & 1 ? V.side - 1 : 0; c[(axis + 1) % 3] = u; c[(axis + 2) % 3] = v;
    return cellRow(V, blockOf(V, j), c);
  };
  /** The cell at q of block b's page, at most a page out: its row, "air" in an AIR page, "wall" in a SOLID one. */
  const cellAt = (V: Level, b: number, q: readonly number[]): number | "air" | "wall" => {
    const d = q.map(x => (x >= V.side ? 1 : 0) - (x < 0 ? 1 : 0));
    const word = d.some(x => x !== 0) ? around(b, d) : b;
    if (word === 0) return "air";
    if (wallWord(word)) return "wall";
    return cellRow(V, word, q.map((x, i) => x - V.side * d[i]!));
  };
  /** The row across face n of a row; the row itself where there is none. */
  const near = (V: Level, row: number, n: number): number => {
    if (row >= V.cells) return n === (faceOf(V, row) ^ 1) ? ownerOf(V, row) : row;
    const b = blockOf(V, row), c = local(V, row), at = cellAt(V, b, c.map((x, i) => x + STEP[n]![i]!));
    return at === "air" ? row : at === "wall" ? slotRow(V, b, n, c) : at;
  };

  // ---- The root's own rows (umRow, umWall, umTheta) on a state.
  const liquid0 = (k: number): boolean => f.phi[k]! < -UNIFORM_MIXED_THETA_MIN * (4.0 * minH);
  const theta0 = (a: number, b: number): number => {
    const la = liquid0(a); if (la === liquid0(b)) return 1.0;
    const lq = f.phi[la ? a : b]!, air = f.phi[la ? b : a]!;
    return Math.min(Math.max(-lq / Math.max(air - lq, 1e-30), UNIFORM_MIXED_THETA_MIN), 1 / UNIFORM_MIXED_THETA_MIN);
  };
  const topo = (i: number, component: number): number => f.record![4 * i + component]!;
  const tile = (k: number): [number, number, number] => [k & 7, (k >> 3) & 7, (k >> 6) & 7];
  /** The wall slots of tile k: [slot row, axis] for each face of its block it lies on with a SOLID page across. */
  const slotsOf = (k: number): [number, number][] => {
    const out: [number, number][] = [], b = k >>> 9, t = tile(k);
    for (let face = 0; face < 6; face++) if (t[face >> 1] === (face & 1 ? 7 : 0) && wallWord(across(b, face))) out.push([slotRow(L[0]!, b, face, t), face >> 1]);
    return out;
  };
  const absent = (k: number): boolean => f.support[uniformMixedResidencyWord(keys) + 4 + 8 * (k >>> 9) + (((k >> 2) & 1) | ((k >> 4) & 2) | ((k >> 6) & 4))] === 0;

  function run(name: string, S: RootLevelsState): void {
    const M = S.mixed, A = S.native, R = S.rounding;
    const P = (i: number): number => M[w.p + i]!, lo = (i: number): number => R[2 * (i + 1)]!;
    const difference = (a: number, b: number): number => (P(a) - P(b)) + (lo(a) - lo(b));
    const wall = (k: number, s: number, axis: number): number => {
      if (!liquid0(k)) return 0.0;
      const distance = 4.0 * f.h[axis]!, fraction = solid ? topo(s, 0) : 0.5;
      return fraction <= 1e-6 ? 0.0 : fraction / (distance * distance);
    };
    /** Row k's diagonal and A p. */
    const rootRow = (k: number): [number, number] => {
      const b = k >>> 9, t = tile(k); let diagonal = 0, value = 0;
      for (let face = 0; face < 6; face++) {
        const axis = face >> 1, q = t.map((x, i) => x + STEP[face]![i]!);
        let n: number;
        if (q[axis]! < 0 || q[axis]! > 7) {
          const word = across(b, face);
          if (wallWord(word)) { const s = slotRow(L[0]!, b, face, t), weight = wall(k, s, axis); diagonal += weight; value += weight * difference(k, s); continue; }
          // A stored block's tile, or block 0's: far air.
          n = 512 * word + ((q[0]! & 7) | ((q[1]! & 7) << 3) | ((q[2]! & 7) << 6));
        } else n = 512 * b + (q[0]! | (q[1]! << 3) | (q[2]! << 6));
        const distance = 4.0 * f.h[axis]!;
        let coefficient = 1.0 / (distance * distance * theta0(k, n));
        if (solid) { const volume = face & 1 ? topo(k, axis + 1) : topo(n, axis + 1); coefficient = volume <= 1e-6 ? 0.0 : volume / (distance * distance * theta0(k, n)); }
        diagonal += coefficient; value += coefficient * (liquid0(n) ? difference(k, n) : P(k) + lo(k));
      }
      return [diagonal, value];
    };
    const projected = (p: number, r: number, low: number, diagonal: number): number => {
      const gap = Math.max(0.0, p - low), v = Math.max(r < 0.0 && -r >= gap * diagonal ? gap * diagonal : Math.abs(r), Math.max(0.0, low - p) * diagonal);
      return Number.isFinite(p) && Number.isFinite(r) && Number.isFinite(v) && v < 3.4e38 && v >= 0.0 ? v : FLT;
    };
    const addPressure = (i: number, value: number): void => { const total = P(i) + value + lo(i), p = Math.fround(total); M[w.p + i] = p; R[2 * (i + 1)] = total - p; };
    const mask = (V: Level, row: number): number => A[V.c + 7 * row + 6]! >>> 0;
    /** Level 0's jobs: the listed octants, or every octant of a stored block. */
    const jobs0 = (): number[] => {
      if (f.listed) return Array.from(S.list.subarray(4, 4 + S.list[0]!));
      const out: number[] = []; for (let id = 0; id < 8 * N; id++) if (stored(id >>> 3)) out.push(id); return out;
    };
    const jobKeys = (id: number): number[] => {
      const out: number[] = [];
      for (let z = 0; z < 4; z++) for (let y = 0; y < 4; y++) for (let x = 0; x < 4; x++) out.push(512 * (id >>> 3) + ((4 * (id & 1) + x) | ((4 * ((id >> 1) & 1) + y) << 3) | ((4 * ((id >> 2) & 1) + z) << 6)));
      return out;
    };
    /** The rows a launch on level l visits: level 0's jobs' tiles and their wall slots; below it every row of the solve. */
    const visited = (l: number): number[] => {
      const V = L[l]!, out: number[] = [];
      if (l === 0) { for (const id of jobs0()) for (const k of jobKeys(id)) { out.push(k); for (const [s] of slotsOf(k)) out.push(s); } return out; }
      for (let j = 0; j < V.rows; j++) if (mask(V, j) & ROW) out.push(j);
      return out;
    };

    // ---- The setup's geometry.
    const liquidPhi = (l: number, phi: number): boolean => phi < -0.001 * ((4 << l) * minH);
    const surfaceTheta = (lq: number, air: number): number => Math.min(Math.max(-lq / Math.max(air - lq, 1e-30), 0.001), 1000.0);
    const faceCoefficient = (l: number, axis: number, v: number, a: number, b: number): number => {
      if (v <= 1e-6) return 0.0;
      const h = (4 << l) * f.h[axis]!, la = liquidPhi(l, a), lb = liquidPhi(l, b);
      return v / (h * h * (la && !lb ? surfaceTheta(a, b) : !la && lb ? surfaceTheta(b, a) : 1.0));
    };
    const slotPhi = (V: Level, cell: number): number => { const phi = A[V.phi[0] + cell]!; return A[V.v + 4 * cell]! > 1e-5 && phi < 0.0 ? phi : 2.0 * minH; };

    const parts = name.split("."), kind = parts[0]!, arg = (i: number): number => Number(parts[i]);
    if (kind === "setup") {
      const V = L[0]!;
      for (let j = 0; j < V.rows; j++) {
        const b = blockOf(V, j); if (!stored(b)) continue;
        const cell = j < V.cells, k = cell ? j : ownerOf(V, j), row = cell || wallWord(across(b, faceOf(V, j)));
        if (row && absent(k)) { M[w.p + j] = 0.0; M[w.b0 + j] = 0.0; M[w.m0 + j] = !cell || (solid && topo(k, 0) <= 1e-5) ? 0.0 : -FLT; }
        if (cell) {
          const t = tile(j);
          A[V.phi[0] + j] = f.phi[j]!;
          for (let i = 0; i < 4; i++) A[V.v + 4 * j + i] = solid ? topo(j, i) : i > 0 && t[i - 1] === 7 && wallWord(across(b, 2 * (i - 1) + 1)) ? 0.5 : 1.0;
        } else A[V.sv + j - V.cells] = solid ? topo(j, 0) : 0.5;
        if (f.listed) {
          let lower = -FLT; if (row) { lower = M[w.m0 + j]!; M[w.residual + j] = 0.0; }
          A[V.p[0] + j] = 0.0; A[V.p[1] + j] = 0.0; A[V.b[0] + j] = 0.0; A[V.m[0] + j] = lower;
        }
      }
    } else if (kind === "down") {
      const C = L[arg(1)]!, F = L[arg(1) - 1]!, lf = F.l;
      /** Whether the open children under coarse cell q of block b's page average to liquid. */
      const submerged = (b: number, q: readonly number[]): boolean => {
        const at = cellAt(C, b, q); if (at === "wall") return true; if (at === "air") return false;
        const home = blockOf(C, at), c = local(C, at); let total = 0, count = 0;
        for (let o = 0; o < 8; o++) { const r = cellRow(F, home, [2 * c[0] + (o & 1), 2 * c[1] + ((o >> 1) & 1), 2 * c[2] + (o >> 2)]); if (A[F.v + 4 * r]! > 1e-5) { total += A[F.phi[0] + r]!; count++; } }
        return count === 0 || liquidPhi(lf, total / count);
      };
      for (let j = 0; j < C.rows; j++) {
        const b = blockOf(C, j); if (!stored(b)) continue;
        if (j >= C.cells) {
          const face = faceOf(C, j), [u, v] = slotUV(C, j); let total = 0;
          for (let o = 0; o < 4; o++) total += A[F.sv + slotAt(F, b, face, 2 * u + (o & 1), 2 * v + (o >> 1)) - F.cells]!;
          A[C.sv + j - C.cells] = total / 4.0; continue;
        }
        const c = local(C, j); let open = 0, openPhi = 0, allPhi = 0, air = 0, airPhi = 0, liquid = 0;
        const fractions = [0, 0, 0, 0];
        for (let o = 0; o < 8; o++) {
          const bit = [o & 1, (o >> 1) & 1, o >> 2], r = cellRow(F, b, c.map((x, i) => 2 * x + bit[i]!)), phi = A[F.phi[0] + r]!, isOpen = A[F.v + 4 * r]! > 1e-5;
          allPhi += phi; fractions[0]! += A[F.v + 4 * r]! / 8.0;
          for (let i = 0; i < 3; i++) if (bit[i] === 1) fractions[i + 1]! += A[F.v + 4 * r + 1 + i]! / 4.0;
          if (isOpen) { open++; openPhi += phi; if (liquidPhi(lf, phi)) liquid++; else { air++; airPhi += phi; } }
        }
        let phi = open > 0 ? openPhi / open : allPhi / 8.0;
        if (air > 0 && liquid > 0 && liquidPhi(lf, phi)) {
          let vanishes = true;
          for (let k = 0; k < 27 && vanishes; k++) if (k !== 13 && !submerged(b, [c[0] + k % 3 - 1, c[1] + Math.floor(k / 3) % 3 - 1, c[2] + Math.floor(k / 9) - 1])) vanishes = false;
          if (vanishes) phi = airPhi / air;
        }
        A[C.phi[0] + j] = phi; for (let i = 0; i < 4; i++) A[C.v + 4 * j + i] = fractions[i]!;
      }
    } else if (kind === "extend") {
      const V = L[arg(1)]!;
      for (let j = 0; j < V.cells; j++) {
        const b = blockOf(V, j); if (!stored(b)) continue;
        let phi = A[V.phi[0] + j]!;
        if (A[V.v + 4 * j]! <= 1e-5) {
          const c = local(V, j); let total = 0, weight = 0;
          for (let n = 0; n < 6; n++) {
            const at = cellAt(V, b, c.map((x, i) => x + STEP[n]![i]!)); if (typeof at !== "number") continue;
            const v = A[V.v + 4 * at]!, other = A[V.phi[0] + at]!;
            if (v > 1e-5 && other < 0.0) { total += v * other; weight += v; }
          }
          if (weight > 0.0) phi = total / Math.max(weight, 1e-9);
        }
        A[V.phi[1] + j] = phi;
      }
    } else if (kind === "bake") {
      const V = L[arg(1)]!, l = V.l;
      for (let j = 0; j < V.rows; j++) {
        const b = blockOf(V, j), at = V.c + 7 * j;
        if (!stored(b)) { A[at + 6] = 0.0; continue; }
        const a = [0, 0, 0, 0, 0, 0]; let bits = ROW, odd = 0;
        if (l === last) odd = (sum(pageOf(b)) + 1) & 1;
        if (j >= V.cells) {
          const face = faceOf(V, j), axis = face >> 1;
          if (!wallWord(across(b, face))) { A[at + 6] = 0.0; continue; }
          const cell = ownerOf(V, j), phi = slotPhi(V, cell); odd ^= 1;
          if (liquidPhi(l, phi)) {
            const other = A[V.phi[1] + cell]!, v = face & 1 ? A[V.v + 4 * cell + 1 + axis]! : A[V.sv + j - V.cells]!;
            a[face ^ 1] = faceCoefficient(l, axis, v, phi, other); bits |= 1; if (liquidPhi(l, other)) bits |= 2 << (face ^ 1);
          }
        } else {
          const phi = A[V.phi[1] + j]!;
          if (liquidPhi(l, phi)) {
            const c = local(V, j); bits |= 1;
            for (let n = 0; n < 6; n++) {
              const axis = n >> 1, cellNear = cellAt(V, b, c.map((x, i) => x + STEP[n]![i]!));
              let other = far, v = 1.0;
              if (typeof cellNear === "number") { other = A[V.phi[1] + cellNear]!; v = A[V.v + 4 * cellNear + 1 + axis]!; }
              else if (cellNear === "wall") { other = slotPhi(V, j); v = A[V.sv + slotRow(V, b, n, c) - V.cells]!; }
              if (n & 1) v = A[V.v + 4 * j + 1 + axis]!;
              a[n] = faceCoefficient(l, axis, v, phi, other); if (liquidPhi(l, other)) bits |= 2 << n;
            }
          }
        }
        for (let n = 0; n < 6; n++) A[at + n] = a[n]!; A[at + 6] = bits | (odd * ODD);
      }
    } else if (kind === "classify") {
      const V = L[0]!, bounded = (row: number): boolean => A[V.m[0] + row]! > -3.0e38;
      for (let job = 0; job < 8 * N; job++) {
        if (!stored(job >>> 3)) continue; let live = 0, bound = 0;
        for (const k of jobKeys(job)) { if (mask(V, k) & 1) live = 1; if (bounded(k)) bound = 1; for (const [s] of slotsOf(k)) if (bounded(s)) bound = 1; }
        S.list[4 + 8 * N + job] = live | (bound << 1);
      }
    } else if (kind === "list") {
      const classOf = (job: number): number => S.list[4 + 8 * N + job]!;
      for (let job = 0; job < 8 * N; job++) {
        const b = job >>> 3; if (!stored(b)) continue;
        const o = [job & 1, (job >> 1) & 1, (job >> 2) & 1]; let listed = (classOf(job) & 2) !== 0;
        for (let k = 0; k < 27 && !listed; k++) {
          // The octant beside this one: of this block, or of the stored block across the page's face, edge or corner.
          const q = [o[0]! + k % 3 - 1, o[1]! + Math.floor(k / 3) % 3 - 1, o[2]! + Math.floor(k / 9) - 1], d = q.map(x => (x > 1 ? 1 : 0) - (x < 0 ? 1 : 0));
          const word = d.some(x => x !== 0) ? around(b, d) : b;
          if (!(word >= 1 && word <= N - 2)) continue;
          const c = q.map((x, i) => x - 2 * d[i]!); listed = (classOf(8 * word + (c[0]! | (c[1]! << 1) | (c[2]! << 2))) & 1) !== 0;
        }
        if (listed) { S.list[4 + S.list[0]!] = job; S.list[0]!++; S.list[4 + 8 * N + job]! |= 4; }
      }
    } else if (kind === "restrictV" || kind === "restrictInner" || kind === "restrictFull") {
      const rhs = kind === "restrictInner" ? w.b1 : w.b0, minimum = kind === "restrictInner" ? w.m1 : w.m0, full = kind === "restrictFull", V = L[0]!;
      const store = (s: number, b: number): void => {
        const own = P(s), lower = M[minimum + s]! - own;
        if (full) { M[w.backup + s] = own; R[2 * (s + 1) + 1] = lo(s); M[w.m1 + s] = lower; M[w.b1 + s] = b; }
        A[V.p[0] + s] = 0.0; A[V.b[0] + s] = b; A[V.m[0] + s] = lower;
      };
      // Every residual is of the pressure before this launch.
      const out: [number, number][] = [];
      for (const id of jobs0()) for (const k of jobKeys(id)) {
        out.push([k, liquid0(k) ? M[rhs + k]! - rootRow(k)[1] : 0.0]);
        for (const [s, axis] of slotsOf(k)) out.push([s, M[rhs + s]! - wall(k, s, axis) * difference(s, k)]);
      }
      for (const [s, b] of out) store(s, b);
    } else if (kind === "measure") {
      const out: [number, number][] = [];
      for (const id of jobs0()) for (const k of jobKeys(id)) {
        const q = P(k);
        if (!liquid0(k)) out.push([k, Number.isFinite(q) ? 0.0 : FLT]);
        else { const [diagonal, value] = rootRow(k); out.push([k, projected(q, M[w.b0 + k]! - value, M[w.m0 + k]!, diagonal)]); }
        for (const [s, axis] of slotsOf(k)) { const coefficient = wall(k, s, axis); out.push([s, projected(P(s), M[w.b0 + s]! - coefficient * difference(s, k), M[w.m0 + s]!, coefficient)]); }
      }
      for (const [s, v] of out) M[w.residual + s] = v;
    } else {
      // ---- The cycles' launches on the levels.
      /** `count` Jacobi iterations of level V from parity src (weights from `first`): every row's last iterate. */
      const jacobi = (V: Level, src: number, rhs: number, low: number, first: number, count: number): Float64Array => {
        let cur = new Float64Array(V.rows); for (let j = 0; j < V.rows; j++) cur[j] = A[V.p[src] + j]!;
        for (let i = 0; i < count; i++) {
          const weight = UNIFORM_MIXED_ROOT_SMOOTHING[first + i]!, next = new Float64Array(cur);
          for (let j = 0; j < V.rows; j++) {
            const bits = mask(V, j); if (!(bits & ROW)) continue;
            const old = cur[j]!, minimum = A[low + j]!; let value = Math.max(old, minimum);
            if (bits & 1) {
              let diagonal = 0, total = 0;
              for (let n = 0; n < 6; n++) { const a = A[V.c + 7 * j + n]!; diagonal += a; if ((bits >> (n + 1)) & 1) total += a * cur[near(V, j, n)]!; }
              const p = diagonal > 0.0 ? (total + A[rhs + j]!) / diagonal : 0.0;
              value = Math.max(old + (p - old) * weight, minimum);
            }
            next[j] = value;
          }
          cur = next;
        }
        return cur;
      };
      const residual = (V: Level, row: number, par: number, rhs: number): number => {
        const bits = mask(V, row); if (!(bits & 1)) return 0.0;
        const centre = A[V.p[par] + row]!; let total = 0;
        for (let n = 0; n < 6; n++) total += A[V.c + 7 * row + n]! * (centre - ((bits >> (n + 1)) & 1 ? A[V.p[par] + near(V, row, n)]! : 0.0));
        return A[rhs + row]! - total;
      };
      /** The eight rows of F under row j of C: a cell's eight cells, a slot's four slots twice. */
      const children = (F: Level, C: Level, j: number): number[] => {
        const b = blockOf(C, j), out: number[] = [];
        if (j < C.cells) { const c = local(C, j); for (let o = 0; o < 8; o++) out.push(cellRow(F, b, [2 * c[0] + (o & 1), 2 * c[1] + ((o >> 1) & 1), 2 * c[2] + (o >> 2)])); return out; }
        const face = faceOf(C, j), [u, v] = slotUV(C, j);
        for (let o = 0; o < 8; o++) out.push(slotAt(F, b, face, 2 * u + (o & 1), 2 * v + ((o >> 1) & 1)));
        return out;
      };
      /** A V-cycle's descent from parity par of F on (rhs, low), or a Full-Cycle's of (rhs, low) themselves. */
      const descend = (F: Level, C: Level, full: boolean, par: number, rhs: number, low: number): void => {
        const out: [number, number, number][] = [];
        for (let j = 0; j < C.rows; j++) {
          if (!(mask(C, j) & ROW)) continue;
          let total = 0, lower = -FLT;
          for (const at of children(F, C, j)) {
            if (full) { total += A[rhs + at]!; lower = Math.max(lower, A[low + at]!); }
            else { total += residual(F, at, par, rhs); lower = Math.max(lower, A[low + at]! - A[F.p[par] + at]!); }
          }
          out.push([j, total / 8.0, lower]);
        }
        for (const [j, b, lower] of out) {
          if (full) { A[C.b[1] + j] = b; A[C.m[1] + j] = lower; }
          else { A[C.b[0] + j] = b; A[C.p[0] + j] = 0.0; A[C.m[0] + j] = lower; }
        }
      };
      /** Level C's parity `coarse` interpolated onto `rows` of F: cell-centred trilinear, zero in far air, a wall left out and the rest renormalised; a slot on its cell's own layer. */
      const prolong = (F: Level, C: Level, src: number, to: number, coarse: number, add: boolean, rows: readonly number[]): void => {
        const out: number[] = [];
        for (const row of rows) {
          const cell = row < F.cells ? row : ownerOf(F, row), held = row < F.cells ? 3 : faceOf(F, row) >> 1, b = blockOf(F, cell), c = local(F, cell);
          let value = 0, total = 0;
          for (let o = 0; o < 8; o++) {
            const bit = [o & 1, (o >> 1) & 1, o >> 2], q = [0, 0, 0]; let weight = 1.0, skip = false;
            for (let i = 0; i < 3; i++) {
              // The coarse cell below or above the fine cell's centre, a quarter and three quarters of the way.
              if (i === held) { if (bit[i] === 0) skip = true; q[i] = c[i]! >> 1; continue; }
              const lowTap = (c[i]! >> 1) - (c[i]! & 1 ? 0 : 1), fraction = c[i]! & 1 ? 0.25 : 0.75;
              q[i] = lowTap + bit[i]!; weight *= bit[i] === 1 ? fraction : 1.0 - fraction;
            }
            if (skip) continue;
            const at = cellAt(C, b, q); if (at === "wall") continue;
            total += weight; if (at !== "air") value += weight * A[C.p[coarse] + at]!;
          }
          const v = total > 0.0 ? value / total : 0.0;
          out.push(add ? A[F.p[src] + row]! + v : v);
        }
        rows.forEach((row, i) => { A[F.p[to] + row] = out[i]!; });
      };
      const rowsOf = (V: Level): number[] => { const out: number[] = []; for (let j = 0; j < V.rows; j++) if (mask(V, j) & ROW) out.push(j); return out; };
      /** A whole visit in the one-workgroup launch: stored in the other parity. */
      const visit = (V: Level, start: number, rhs: number, low: number): number => {
        const result = jacobi(V, start, rhs, low, 0, UNIFORM_MIXED_ROOT_SMOOTHING.length), end = start ^ (UNIFORM_MIXED_ROOT_SMOOTHING.length & 1);
        for (const j of rowsOf(V)) A[V.p[end] + j] = result[j]!;
        return end;
      };
      const solve = (V: Level, start: number, rhs: number, low: number): void => {
        const x = new Float64Array(V.rows), rows = rowsOf(V);
        for (const j of rows) { const v = A[V.p[start] + j]!; x[j] = mask(V, j) & 1 ? v : Math.max(v, A[low + j]!); }
        for (let half = 0; half < HALVES; half++) for (const j of rows) {
          const bits = mask(V, j); if (!(bits & 1) || (Math.floor(bits / ODD) & 1) !== (half & 1)) continue;
          let diagonal = 0, total = 0;
          for (let n = 0; n < 6; n++) { const a = A[V.c + 7 * j + n]!; diagonal += a; if ((bits >> (n + 1)) & 1) total += a * x[near(V, j, n)]!; }
          const p = diagonal > 0.0 ? (total + A[rhs + j]!) / diagonal : 0.0;
          x[j] = Math.max(x[j]! + (p - x[j]!) * OMEGA, A[low + j]!);
        }
        for (const j of rows) A[V.p[start] + j] = x[j]!;
      };
      /** A V-cycle of level a and below on (rhsA, lowA) from parity 0; the parity of its result. */
      const vCycle = (a: number, rhsA: number, lowA: number): number => {
        const par = new Array<number>(last + 1).fill(0);
        const fields = (l: number): [number, number] => l === a ? [rhsA, lowA] : [L[l]!.b[0], L[l]!.m[0]];
        for (let l = a; l < last; l++) { const [rhs, low] = fields(l); par[l] = visit(L[l]!, par[l]!, rhs, low); descend(L[l]!, L[l + 1]!, false, par[l]!, rhs, low); par[l + 1] = 0; }
        solve(L[last]!, par[last]!, ...fields(last));
        for (let l = last - 1; l >= a; l--) { const [rhs, low] = fields(l); prolong(L[l]!, L[l + 1]!, par[l]!, par[l]! ^ 1, par[l + 1]!, true, rowsOf(L[l]!)); par[l]! ^= 1; par[l] = visit(L[l]!, par[l]!, rhs, low); }
        return par[a]!;
      };
      if (kind === "smooth") {
        const V = L[arg(1)]!, set = arg(2), src = arg(3), result = jacobi(V, src, V.b[set]!, V.m[set]!, arg(4), arg(5)), end = arg(6);
        for (const row of visited(V.l)) {
          const value = result[row]!;
          if (end === 0) A[V.p[src ^ 1]! + row] = value;
          else if (end === 2) { M[w.p + row] = value; R[2 * (row + 1)] = 0.0; }
          else if (end === 3) { addPressure(row, value); addPressure(row, R[2 * (row + 1) + 1]!); addPressure(row, M[w.backup + row]!); }
          else addPressure(row, value);
        }
      } else if (kind === "descend") {
        const F = L[arg(1)]!, C = L[arg(1) + 1]!;
        if (parts[2] === "full") { const set = F.l ? 1 : 0; descend(F, C, true, 0, F.b[set], F.m[set]); }
        else descend(F, C, false, arg(3), F.b[arg(2)]!, F.m[arg(2)]!);
      } else if (kind === "ascend") {
        const F = L[arg(1)]!, C = L[arg(1) + 1]!, rows = visited(F.l);
        if (parts[3] === "assign") prolong(F, C, 0, 0, arg(2), false, rows);
        else prolong(F, C, arg(2), arg(2) ^ 1, arg(3), true, rows);
      } else if (kind === "coarse") {
        const T = L[f.tiled]!;
        if (parts[1] === "full") {
          for (let l = f.tiled; l < last; l++) descend(L[l]!, L[l + 1]!, true, 0, L[l]!.b[1], L[l]!.m[1]);
          const B = L[last]!; for (let j = 0; j < B.rows; j++) A[B.p[0] + j] = 0.0;
          solve(B, 0, B.b[1], B.m[1]);
          let top = 0;
          for (let l = last - 1; l >= f.tiled; l--) { prolong(L[l]!, L[l + 1]!, 0, 0, top, false, rowsOf(L[l]!)); top = vCycle(l, L[l]!.b[1], L[l]!.m[1]); }
        } else vCycle(f.tiled, T.b[arg(1)]!, T.m[arg(1)]!);
      } else throw new Error(`The root levels reference has no launch ${name}`);
    }
  }

  /** A root row's or a level row's place, for a report. */
  const describe = (l: number, j: number): string => {
    const V = L[l]!, b = blockOf(V, j), where = b === 0 ? "block 0 (AIR)" : b === N - 1 ? `block ${b} (SOLID)` : stored(b) ? `block ${b} page ${pageOf(b).join(",")}` : `block ${b} (free)`;
    return j < V.cells ? `level ${l} cell ${local(V, j).join(",")} of ${where}` : `level ${l} slot ${slotUV(V, j).join(",")} on face ${"xyz"[faceOf(V, j) >> 1]}${faceOf(V, j) & 1 ? "+" : "-"} of ${where}`;
  };
  return { run, describe, stored, pageOf, levels: L };
}
