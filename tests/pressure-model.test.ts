/**
 * The Float64 oracle of lane P's brick-pool MGPCG
 * (lib/methods/narrow-band/pressure-model.ts).
 *
 * Checks:
 * - the V-cycle is a symmetric positive definite operator;
 * - the coarsest level stays within 64 rows and inverts exactly;
 * - a resting tank solves to the hydrostatic X = g dt h (H - y - 1/2) of
 *   brick-abi.md §6.1's operator, with the ghost-fluid surface;
 * - the staggered Schwarz covering beats the bricks alone on a thin film;
 * - the check before the V-cycle stops a converged start at zero iterations;
 * - a non-positive pivot is refused (FAIL.PIVOT).
 */
import assert from "node:assert/strict";
import test from "node:test";
import { PressureModel, applyLevel, coarsenLevel, denseInverse, packCell, type FineRows, type ModelLevel } from "../lib/methods/narrow-band/pressure-model";

const THETA_MIN = 1e-3;
function rng(seed: number) { return () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; }; }

/** brick-abi.md §6.1's operator over a box [0,n)^3 with walls on every side but an open top, liquid where phi < 0. */
function operator(n: readonly [number, number, number], phi: (x: number, y: number, z: number) => number, us?: (a: number, x: number, y: number, z: number) => number, h = 0.01) {
  const keys: number[] = [], diag: number[] = [], k: [number[], number[], number[]] = [[], [], []], rhs: number[] = [];
  const liquid = (x: number, y: number, z: number) => x >= 0 && z >= 0 && y >= 0 && x < n[0] && z < n[2] && y < n[1] && phi(x, y, z) < 0;
  const open = (x: number, y: number, z: number) => x >= 0 && z >= 0 && y >= 0 && x < n[0] && z < n[2]; // the top is open
  const A = (a: number, x: number, y: number, z: number) => { const d = [0, 0, 0]; d[a] = 1; return open(x, y, z) && open(x - d[0]!, y - d[1]!, z - d[2]!) ? 1 : 0; };
  for (let z = 0; z < n[2]; z++) for (let y = 0; y < n[1]; y++) for (let x = 0; x < n[0]; x++) {
    if (!liquid(x, y, z)) continue;
    keys.push(packCell(x, y, z));
    let d = 0, r = 0;
    for (let a = 0; a < 3; a++) {
      const e = [0, 0, 0]; e[a] = 1;
      const lo = [x - e[0]!, y - e[1]!, z - e[2]!] as const, hi = [x + e[0]!, y + e[1]!, z + e[2]!] as const;
      const aLo = A(a, x, y, z), aHi = A(a, ...hi);
      for (const [q, w] of [[lo, aLo], [hi, aHi]] as const) {
        if (!w) continue;
        if (liquid(...q)) d += w;
        else { const pl = phi(x, y, z), pa = phi(...q); const t = Math.min(Math.max(pl / (pl - pa), THETA_MIN), 1 / THETA_MIN); d += w / t; }
      }
      k[a].push(aLo && liquid(...lo) ? aLo : 0);
      if (us) r += -h * (aHi * us(a, ...hi) - aLo * us(a, x, y, z));
    }
    diag.push(d); rhs.push(r);
  }
  return { rows: { keys, diag, k } as FineRows, rhs: Float64Array.from(rhs) };
}

const dot = (a: Float64Array, b: Float64Array) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i]! * b[i]!; return s; };
const linf = (a: Float64Array) => a.reduce((m, v) => Math.max(m, Math.abs(v)), 0);
const random = (n: number, seed: number, keep: (i: number) => boolean = () => true) => { const r = rng(seed); return Float64Array.from({ length: n }, (_, i) => keep(i) ? r() - 0.5 : 0); };

/** A pool (x < 10, 10 cells deep) beside a film 1-2 cells thick: brick faces cut the film. */
const filmRnd = rng(7);
const filmHeights = new Map<string, number>();
const film = operator([32, 16, 32], (x, y, z) => {
  const key = `${x},${z}`; if (!filmHeights.has(key)) filmHeights.set(key, x < 10 ? 10.3 + 0.2 * Math.sin(z) : 1.0 + filmRnd());
  return y + 0.5 - filmHeights.get(key)!;
});

for (const fineOrder of ["BS", "BSB"] as const) test(`pressure model: the V-cycle is symmetric positive definite (L0 ${fineOrder})`, () => {
  const m = new PressureModel(film.rows, { fineOrder }), n = film.rows.keys.length;
  assert.ok(m.levels.at(-1)!.keys.length <= 64, "coarsest within 64 rows");
  const y = random(n, 11), w = random(n, 23);
  const a = dot(w, m.vcycle(y)), b = dot(y, m.vcycle(w));
  assert.ok(Math.abs(a - b) <= 1e-12 * Math.max(Math.abs(a), Math.abs(b)), `symmetry: ${a} vs ${b}`);
  for (const seed of [3, 5, 9]) { const v = random(n, seed); assert.ok(dot(v, m.vcycle(v)) > 0, "positive"); }
});

test("pressure model: the coarsest inverse is exact", () => {
  const m = new PressureModel(film.rows), top = m.levels.at(-1)!, r = random(top.keys.length, 5);
  const z = m.vcycle(r, m.levels.length - 1), Az = applyLevel(top, z);
  assert.ok(linf(Az.map((v, i) => v - r[i]!)) < 1e-10 * linf(r));
});

test("pressure model: a resting tank solves to hydrostatic pressure", () => {
  const g = 9.81, dt = 1 / 60, h = 0.01, H = 13.3;
  const tank = operator([24, 24, 24], (_x, y) => y + 0.5 - H, (a) => a === 1 ? -g * dt : 0, h);
  const m = new PressureModel(tank.rows), n = tank.rows.keys.length;
  const { x, iterations } = m.solve(tank.rhs, new Float64Array(n), 1e-9 * linf(tank.rhs), 40);
  assert.ok(iterations > 0 && iterations < 20, `${iterations} iterations`);
  let worst = 0;
  tank.rows.keys.forEach((key, i) => {
    const y = Math.floor(key / 16384) % 16384 - 8, exact = g * dt * h * (H - y - 0.5);
    worst = Math.max(worst, Math.abs(x[i]! - exact) / (g * dt * h * H));
  });
  assert.ok(worst < 1e-6, `relative error ${worst}`);
});

test("pressure model: the staggered covering converges a thin film in fewer iterations", () => {
  const n = film.rows.keys.length, b = random(n, 31, i => { const k = film.rows.keys[i]!; const x = k % 16384 - 8, y = Math.floor(k / 16384) % 16384 - 8; return x >= 10 && y < 2; });
  const run = (staggered: boolean) => new PressureModel(film.rows, { staggered }).solve(b, new Float64Array(n), 1e-6 * linf(b), 60).iterations;
  const bricks = run(false), staggered = run(true);
  assert.ok(staggered < bricks, `bricks ${bricks}, staggered ${staggered}`);
});

test("pressure model: the check before the V-cycle stops a converged start", () => {
  const m = new PressureModel(film.rows), n = film.rows.keys.length, x = random(n, 41);
  const b = applyLevel(m.fine, x);
  assert.equal(m.solve(b, x, 1e-6, 10).iterations, 0);
});

test("pressure model: a non-positive pivot is refused", () => {
  const L: ModelLevel = coarsenLevel(new PressureModel(film.rows).fine);
  (L.diag as Float64Array)[0] = -1;
  assert.equal(denseInverse(L), null);
});
