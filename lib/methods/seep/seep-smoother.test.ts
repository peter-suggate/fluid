/**
 * Float64 model of the Seep pressure preconditioner's structure: aggregation
 * multigrid (A_c = P^T A P / OC), brick-block red-black Gauss-Seidel smoothers
 * with a zero halo before and a frozen halo after the coarse correction, and the
 * L0 smoother over two staggered partitions (the bricks, then blocks shifted by
 * half a brick along x and z: seep-kernels.ts D0B / U0B), with post-smoothing in
 * the exact reverse order. Checks that the V-cycle is a symmetric positive
 * definite operator (so PCG applies), and that the staggered partition speeds up
 * a thin film, which converges slowest across brick faces with bricks alone.
 */
import assert from "node:assert/strict";
import test from "node:test";

const NX = 32, NY = 16, NZ = 32, THETA_MIN = 1e-3, OC = 1.6;
interface Level { nx: number; ny: number; nz: number; act: Uint8Array; d: Float64Array; o: Float64Array[] }
const AX = [[1, 0, 0], [0, 1, 0], [0, 0, 1]] as const;
const lid = (L: Level, x: number, y: number, z: number) => x + L.nx * (y + L.ny * z);

function rng(seed: number) { return () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; }; }

/** A film 1-2 cells thick on the floor plus a pool (x < 10) 10 cells deep; ghost-fluid rows at the surface. */
function buildFine(): Level {
  const rnd = rng(7), N = NX * NY * NZ;
  const phi = new Float64Array(N);
  for (let z = 0; z < NZ; z++) for (let y = 0; y < NY; y++) for (let x = 0; x < NX; x++) {
    const h = x < 10 ? 10.3 + 0.2 * Math.sin(z) : 1.0 + rnd();
    phi[x + NX * (y + NY * z)] = y + 0.5 - h;
  }
  const L: Level = { nx: NX, ny: NY, nz: NZ, act: new Uint8Array(N), d: new Float64Array(N), o: [0, 1, 2].map(() => new Float64Array(N)) };
  for (let i = 0; i < N; i++) if (phi[i] < 0) L.act[i] = 1;
  for (let z = 0; z < NZ; z++) for (let y = 0; y < NY; y++) for (let x = 0; x < NX; x++) {
    const i = lid(L, x, y, z); if (!L.act[i]) continue;
    let d = 0;
    for (let a = 0; a < 3; a++) for (const s of [-1, 1]) {
      const q = [x, y, z]; q[a] += s;
      if (q[0] < 0 || q[2] < 0 || q[1] < 0 || q[0] >= NX || q[2] >= NZ) continue; // walls (Neumann)
      const j = q[1] >= NY ? -1 : lid(L, q[0], q[1], q[2]);
      if (j >= 0 && L.act[j]) { d += 1; if (s > 0) L.o[a][i] = 1; }
      else { const pn = j < 0 ? 1 : phi[j]; d += 1 / Math.max(phi[i] / (phi[i] - pn), THETA_MIN); }
    }
    L.d[i] = d;
  }
  return L;
}
function coarsen(F: Level): Level {
  const nx = F.nx >> 1, ny = F.ny >> 1, nz = F.nz >> 1, n = nx * ny * nz;
  const C: Level = { nx, ny, nz, act: new Uint8Array(n), d: new Float64Array(n), o: [0, 1, 2].map(() => new Float64Array(n)) };
  for (let z = 0; z < F.nz; z++) for (let y = 0; y < F.ny; y++) for (let x = 0; x < F.nx; x++) {
    const i = lid(F, x, y, z); if (!F.act[i]) continue;
    const p = lid(C, x >> 1, y >> 1, z >> 1); C.act[p] = 1; C.d[p] += F.d[i];
    for (let a = 0; a < 3; a++) {
      const w = F.o[a][i]; if (!w) continue;
      const q = [x, y, z]; q[a] += 1;
      const pq = lid(C, q[0] >> 1, q[1] >> 1, q[2] >> 1);
      if (pq === p) C.d[p] -= 2 * w; else C.o[a][p] += w;
    }
  }
  for (let i = 0; i < n; i++) { C.d[i] /= OC; for (let a = 0; a < 3; a++) C.o[a][i] /= OC; }
  return C;
}
function apply(L: Level, x: Float64Array): Float64Array {
  const out = new Float64Array(x.length), sy = L.nx, sz = L.nx * L.ny;
  for (let z = 0; z < L.nz; z++) for (let y = 0; y < L.ny; y++) for (let xx = 0; xx < L.nx; xx++) {
    const i = lid(L, xx, y, z); if (!L.act[i]) continue;
    let s = L.d[i] * x[i];
    if (xx + 1 < L.nx) s -= L.o[0][i] * x[i + 1];
    if (y + 1 < L.ny) s -= L.o[1][i] * x[i + sy];
    if (z + 1 < L.nz) s -= L.o[2][i] * x[i + sz];
    if (xx > 0) s -= L.o[0][i - 1] * x[i - 1];
    if (y > 0) s -= L.o[1][i - sy] * x[i - sy];
    if (z > 0) s -= L.o[2][i - sz] * x[i - sz];
    out[i] = s;
  }
  return out;
}
/** e <- e + M^-1 (b - A e) for block red-black GS on blocks of side B offset by off (per axis), the halo frozen at
 *  the iterate on entry; colours: the colour of each pass in order. */
function blockGS(L: Level, e: Float64Array, b: Float64Array, B: number, off: readonly number[], colours: number[]) {
  const frozen = Float64Array.from(e), sy = L.nx, sz = L.nx * L.ny;
  const blk = (v: number, a: number) => Math.floor((v + off[a]) / B);
  for (const col of colours) {
    for (let z = 0; z < L.nz; z++) for (let y = 0; y < L.ny; y++) for (let x = 0; x < L.nx; x++) {
      if (((x + y + z) & 1) !== col) continue;
      const i = lid(L, x, y, z); if (!L.act[i] || !(L.d[i] > 0)) continue;
      let s = b[i];
      const nb = (q: number[], j: number, w: number) => {
        if (!w) return;
        const same = blk(q[0], 0) === blk(x, 0) && blk(q[1], 1) === blk(y, 1) && blk(q[2], 2) === blk(z, 2);
        s += w * (same ? e[j] : frozen[j]);
      };
      if (x + 1 < L.nx) nb([x + 1, y, z], i + 1, L.o[0][i]);
      if (y + 1 < L.ny) nb([x, y + 1, z], i + sy, L.o[1][i]);
      if (z + 1 < L.nz) nb([x, y, z + 1], i + sz, L.o[2][i]);
      if (x > 0) nb([x - 1, y, z], i - 1, L.o[0][i - 1]);
      if (y > 0) nb([x, y - 1, z], i - sy, L.o[1][i - sy]);
      if (z > 0) nb([x, y, z - 1], i - sz, L.o[2][i - sz]);
      e[i] = s / L.d[i];
    }
  }
}
function exactSolve(L: Level, b: Float64Array): Float64Array {
  const ids: number[] = []; for (let i = 0; i < L.act.length; i++) if (L.act[i]) ids.push(i);
  const m = ids.length, A = new Float64Array(m * m), rhs = ids.map(i => b[i]);
  ids.forEach((i, r) => { const u = new Float64Array(L.act.length); u[i] = 1; const Au = apply(L, u); ids.forEach((j, c) => { A[r * m + c] = Au[j]; }); });
  for (let k = 0; k < m; k++) {
    const p = A[k * m + k];
    for (let r = 0; r < m; r++) if (r !== k) { const f = A[r * m + k] / p; if (!f) continue; for (let c = k; c < m; c++) A[r * m + c] -= f * A[k * m + c]; rhs[r] -= f * rhs[k]; }
  }
  const e = new Float64Array(L.act.length); ids.forEach((i, r) => { e[i] = rhs[r] / A[r * m + r]; }); return e;
}

const fine = buildFine();
const levels = [fine]; while (levels.length < 4) levels.push(coarsen(levels.at(-1)!));
const SW = [3, 2, 2], BLK = [8, 4, 2];
const rb = (n: number, rev: boolean) => Array.from({ length: 2 * n }, (_, s) => rev ? 1 - (s & 1) : s & 1);
const ZERO = [0, 0, 0], SHIFT = [4, 0, 4];

function vcycle(l: number, b: Float64Array, staggered: boolean): Float64Array {
  const L = levels[l];
  if (l === levels.length - 1) return exactSolve(L, b);
  const e = new Float64Array(b.length);
  const two = staggered && l === 0;
  blockGS(L, e, b, BLK[l], ZERO, rb(SW[l], false));
  if (two) blockGS(L, e, b, BLK[l], SHIFT, rb(SW[l], false));
  const Ae = apply(L, e), r = b.map((v, i) => v - Ae[i]);
  const C = levels[l + 1], bc = new Float64Array(C.act.length);
  const parent = (x: number, y: number, z: number) => lid(C, x >> 1, y >> 1, z >> 1);
  for (let z = 0; z < L.nz; z++) for (let y = 0; y < L.ny; y++) for (let x = 0; x < L.nx; x++) { const i = lid(L, x, y, z); if (L.act[i]) bc[parent(x, y, z)] += r[i]; }
  const ec = vcycle(l + 1, bc, staggered);
  for (let z = 0; z < L.nz; z++) for (let y = 0; y < L.ny; y++) for (let x = 0; x < L.nx; x++) { const i = lid(L, x, y, z); if (L.act[i]) e[i] += ec[parent(x, y, z)]; }
  if (two) blockGS(L, e, b, BLK[l], SHIFT, rb(SW[l], true));
  blockGS(L, e, b, BLK[l], ZERO, rb(SW[l], true));
  return e;
}
const dot = (a: Float64Array, b: Float64Array) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
function randomField(seed: number, region: (x: number, y: number) => boolean = () => true): Float64Array {
  const rnd = rng(seed), v = new Float64Array(fine.act.length);
  for (let z = 0; z < NZ; z++) for (let y = 0; y < NY; y++) for (let x = 0; x < NX; x++) { const i = lid(fine, x, y, z); if (fine.act[i] && region(x, y)) v[i] = rnd() - 0.5; }
  return v;
}
/** PCG iterations to cut the L-inf residual by `red`. */
function iterations(staggered: boolean, b: Float64Array, red: number): number {
  const linf = (a: Float64Array) => a.reduce((m, v) => Math.max(m, Math.abs(v)), 0);
  const r = Float64Array.from(b), r0 = linf(r);
  let z = vcycle(0, r, staggered), p = Float64Array.from(z), rz = dot(r, z);
  for (let k = 1; k <= 40; k++) {
    const q = apply(fine, p), alpha = rz / dot(p, q);
    for (let i = 0; i < r.length; i++) r[i] -= alpha * q[i];
    if (linf(r) <= red * r0) return k;
    z = vcycle(0, r, staggered); const rz2 = dot(r, z), beta = rz2 / rz; rz = rz2;
    for (let i = 0; i < r.length; i++) p[i] = z[i] + beta * p[i];
  }
  return Infinity;
}

test("staggered-partition V-cycle is symmetric positive definite", () => {
  for (const staggered of [false, true]) {
    const y = randomField(11), w = randomField(23);
    const a = dot(w, vcycle(0, y, staggered)), b = dot(y, vcycle(0, w, staggered));
    assert.ok(Math.abs(a - b) <= 1e-12 * Math.max(Math.abs(a), Math.abs(b)), `symmetry (${staggered}): ${a} vs ${b}`);
    for (const seed of [3, 5, 9]) { const v = randomField(seed); assert.ok(dot(v, vcycle(0, v, staggered)) > 0, `positive (${staggered})`); }
  }
});

test("staggered partition converges a thin film in fewer PCG iterations", () => {
  const film = randomField(31, (x, y) => x >= 10 && y < 2);
  const bricks = iterations(false, film, 1e-4), staggered = iterations(true, film, 1e-4);
  assert.ok(Number.isFinite(staggered) && staggered < bricks, `bricks ${bricks}, staggered ${staggered}`);
});
