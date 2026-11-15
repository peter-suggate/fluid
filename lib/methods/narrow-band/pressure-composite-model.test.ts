// The composite MGPCG's Float64 reference (pressure-composite-model.ts) on a synthetic mixed-tier
// system: h rows around a block of native 4h rows, with seams, band links, native links and Dirichlet
// (surface) terms. The V-cycle is symmetric positive definite, and PCG converges.
import assert from "node:assert/strict";
import test from "node:test";
import { CompositePressureModel, type CompositeRows } from "./pressure-composite-model";

type V3 = [number, number, number];
const unit = (a: number): V3 => [a === 0 ? 1 : 0, a === 1 ? 1 : 0, a === 2 ? 1 : 0];
const add = (u: V3, v: V3, s = 1): V3 => [u[0] + s * v[0], u[1] + s * v[1], u[2] + s * v[2]];
const hash = (c: V3, k: number) => { const s = Math.sin(12.9898 * c[0] + 78.233 * c[1] + 37.719 * c[2] + 4.581 * k) * 43758.5453; return s - Math.floor(s); };

/** 32x16x32 h cells of liquid below y = 13 (a rippled surface), native tiles x 2..5, y 0..2, z 2..5. */
export function syntheticComposite(): CompositeRows {
  const N: V3 = [32, 16, 32];
  const nativeTile = (t: V3) => t[0] >= 2 && t[0] <= 5 && t[1] <= 2 && t[2] >= 2 && t[2] <= 5;
  const inside = (c: V3) => c.every((v, a) => v >= 0 && v < N[a]!);
  const surface = (c: V3) => 13 + 1.5 * Math.sin(0.3 * c[0] + 0.2 * c[2]);
  const liquid = (c: V3) => inside(c) && c[1] + 0.5 < surface(c);
  const tileOf = (c: V3): V3 => [c[0] >> 2, c[1] >> 2, c[2] >> 2];
  const isNativeCell = (c: V3) => inside(c) && nativeTile(tileOf(c));
  const cells: V3[] = [], tiles: V3[] = [];
  for (let z = 0; z < N[2]; z++) for (let y = 0; y < N[1]; y++) for (let x = 0; x < N[0]; x++) { const c: V3 = [x, y, z]; if (liquid(c) && !isNativeCell(c)) cells.push(c); }
  for (let z = 0; z < 8; z++) for (let y = 0; y < 4; y++) for (let x = 0; x < 8; x++) if (nativeTile([x, y, z])) tiles.push([x, y, z]);
  const fd = new Float64Array(cells.length), fk = [0, 1, 2].map(() => new Float64Array(cells.length)), fku = [0, 1, 2].map(() => new Float64Array(cells.length));
  const nd = new Float64Array(tiles.length), nk = [0, 1, 2].map(() => new Float64Array(tiles.length));
  const nIndex = new Map(tiles.map((t, i) => [t.join(","), i]));
  cells.forEach((c, i) => {
    for (let a = 0; a < 3; a++) for (const sign of [-1, 1]) {
      const q = add(c, unit(a), sign);
      if (!inside(q)) { fd[i]! += 0.5; continue; } // a separated wall: its coupling on the diagonal
      if (isNativeCell(q)) {
        const ch = 0.4 * (0.8 + 0.4 * hash(sign < 0 ? c : q, a)); fd[i]! += ch;
        if (sign < 0) fk[a]![i] = ch; else fku[a]![i] = ch;
        nd[nIndex.get(tileOf(q).join(","))!]! += ch;
      } else if (liquid(q)) { fd[i]! += 1; if (sign < 0) fk[a]![i] = 1; }
      else fd[i]! += 1 / Math.max(0.05, Math.min(1, surface(c) - c[1] - 0.5)); // ghost-fluid air face
    }
  });
  tiles.forEach((t, i) => {
    for (let a = 0; a < 3; a++) for (const sign of [-1, 1]) {
      const n = add(t, unit(a), sign), j = nIndex.get(n.join(","));
      if (j !== undefined) { nd[i]! += 4; if (sign < 0) nk[a]![i] = 4; }
      else if (n.some(v => v < 0)) nd[i]! += 2; // a separated native wall
    }
  });
  return { f: { cells, diag: fd, k: fk, ku: fku }, n: { tiles, diag: nd, k: nk } };
}

for (const fineOrder of ["BS", "BSB"] as const) test(`composite model: the operator is symmetric and the V-cycle is SPD (L0 ${fineOrder})`, () => {
  const M = new CompositePressureModel(syntheticComposite(), { levels: 8, fineSweeps: 3, coarseSweeps: 2, nativeSweeps: 2, fineOrder });
  const n = M.fine.keys.length;
  const u = Float64Array.from({ length: n }, (_, i) => Math.sin(1.3 * i) + 0.2), v = Float64Array.from({ length: n }, (_, i) => Math.cos(0.7 * i) - 0.1);
  const dot = (a: Float64Array, b: Float64Array) => a.reduce((s, x, i) => s + x * b[i]!, 0);
  const Au = M.apply(u), Av = M.apply(v);
  assert.ok(Math.abs(dot(Au, v) - dot(u, Av)) <= 1e-9 * Math.abs(dot(Au, v)), "A symmetric");
  const Mu = M.vcycle(u), Mv = M.vcycle(v);
  assert.ok(Math.abs(dot(Mu, v) - dot(u, Mv)) <= 1e-9 * Math.max(Math.abs(dot(Mu, v)), 1e-30), `V-cycle symmetric: ${dot(Mu, v)} against ${dot(u, Mv)}`);
  assert.ok(dot(Mu, u) > 0 && dot(Mv, v) > 0, "V-cycle positive");
  assert.ok(M.levels[2]!.carried.every(c => !c), "every L2 row smoothed");
  assert.ok(M.levels[0]!.carried.some(c => c) && M.levels[1]!.carried.some(c => c), "native rows carried at L0 and L1");
});

test("composite model: PCG converges on the mixed system", () => {
  const rows = syntheticComposite(), M = new CompositePressureModel(rows);
  const n = M.fine.keys.length;
  const b = Float64Array.from({ length: n }, (_, i) => (M.isNative(i) ? 64 : 1) * 1e-3 * Math.sin(0.37 * i));
  const { iterations, residual, x } = M.solve(b, new Float64Array(n), 1e-9, 60);
  const Ax = M.apply(x), r = b.map((v, i) => v - Ax[i]!);
  console.log(`# composite model: ${rows.f.cells.length} h rows, ${rows.n.tiles.length} native rows, ${iterations} iterations, residual ${residual.toExponential(2)}`);
  assert.ok(residual <= 1e-9 && M.norm(r) <= 1e-9, `residual ${residual}`);
  assert.ok(iterations <= 30, `${iterations} iterations`);
});
