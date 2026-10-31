import assert from "node:assert/strict";
import test from "node:test";
import { krylovGrid, krylovHierarchy, krylovRhs, krylovSolve, krylovThroat } from "./helpers/pressure-krylov-twin";

// The arena's hierarchy: 4 levels, links rediscretised at 0.5 (coarseScale).
const throat = () => { const L = krylovGrid(...krylovThroat); return { L, H: krylovHierarchy(L, 4, 0.5) }; };
const rate = (v: number[], n: number) => (v[n]! / v[0]!) ** (1 / n);

test("GCR on the V-cycle converges faster per cycle than the cycles alone through a 2x2 throat", () => {
  const { L, H } = throat(), b = krylovRhs(L.coord.length, 3), n = 6;
  const cycles = krylovSolve(H, b, n, "cycles"), gcr = krylovSolve(H, b, n, "gcr");
  // The cycles settle near 0.23 a cycle; GCR near 0.1.
  assert.ok(rate(gcr.max, n) < 0.7 * rate(cycles.max, n), `GCR ${rate(gcr.max, n)} against ${rate(cycles.max, n)}`);
  // The max norm the verdict measures follows the 2-norm GCR minimises.
  assert.ok(gcr.max[n]! < cycles.max[n]!);
});

test("GCR's residual 2-norm never increases, step to step", () => {
  const { L, H } = throat(), run = krylovSolve(H, krylovRhs(L.coord.length, 11), 8, "gcr");
  for (let k = 1; k < run.l2.length; k++) assert.ok(run.l2[k]! <= run.l2[k - 1]! * (1 + 1e-9), `step ${k}: ${run.l2[k]} after ${run.l2[k - 1]}`);
});

test("on a sealed full vessel with an inconsistent RHS GCR stays bounded where flexible CG does not", () => {
  // No Dirichlet row: A is singular, its nullspace the constant. The RHS
  // carries a small net source (an f32 or discretisation residue).
  const L = krylovGrid([24, 16, 24], () => 1), H = krylovHierarchy(L, 4, 0.5), n = 12;
  const raw = krylovRhs(L.coord.length, 5), mean = raw.reduce((s, v) => s + v, 0) / raw.length;
  const b = raw.map(v => v - mean + 1e-3) as Float64Array;
  const drift = (x: Float64Array) => Math.abs(x.reduce((s, v) => s + v, 0) / x.length);
  const cycles = krylovSolve(H, b, n, "cycles"), gcr = krylovSolve(H, b, n, "gcr"), fcg = krylovSolve(H, b, n, "fcg");
  // The cycles drift linearly along the constant; GCR no faster.
  assert.ok(drift(gcr.x) <= drift(cycles.x), `GCR drift ${drift(gcr.x)} against the cycles' ${drift(cycles.x)}`);
  // Neither reaches zero on an inconsistent block; GCR holds the cycles' floor.
  assert.ok(gcr.l2[n]! <= 1.01 * cycles.l2[n]!, `GCR ${gcr.l2[n]} against the cycles' ${cycles.l2[n]}`);
  // Flexible CG divides by the constant's vanishing energy.
  assert.ok(drift(fcg.x) > 1e3 * drift(cycles.x), `FCG drift ${drift(fcg.x)}`);
});
