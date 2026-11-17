/**
 * LANE P OWNS THIS FILE. The native rows' stop (pressure-mgpcg.ts; lane P 2026-11-17, the coordinator's ruling: port
 * the old stop verbatim). CPU only (kernel text and f32 arithmetic).
 * - nBound = min(tol, max(0.1 r0_native, nFloor = 1e-4 1/s)), the old root's (uniform-pressure-arena.ts nBound /
 *   nStopScale): the slots count a native row's residual times tol / nBound, and a native start above the floor runs.
 * - The old engine solves the natives for a correction from the start (pNative: PN_X 0, rhs += c (x_n - start)). The
 *   brick keeps the total pressure in X, so its native residual is formed once in face-difference form (pApplyNX) and
 *   the slots carry it by the recurrence r -= alpha A d. b - D x + sum k x_n from the total pressure reads f32 noise
 *   above the floor on a 2.5 m column (the f32 test below), and the solve stalled at 8 slots on bulk-rest-tank.
 */
import assert from "node:assert/strict";
import test from "node:test";
import { BrickLayout } from "./brick-abi";
import { brickPressureSolver } from "./brick-pressure-solver";
import type { BrickPlacement } from "./brick-kernel";

function codes(): string {
  const L = new BrickLayout({ lattice: [64, 32, 32], f: 125, c: 8, b: 16, k: 0, particles: 1024 });
  const s = brickPressureSolver!(L), all: BrickPlacement[] = [...(s.before ?? []), ...s.setup, ...s.finish];
  for (let i = 0; i < s.slots; i++) all.push(...s.slot(i));
  return [...new Set(all.map(p => p.kernel.code))].join("\n");
}

test("pressure native stop: the old root's nBound gates the setup and scales the native rows' residual", () => {
  const code = codes();
  assert.ok(code.includes("atomicMax(&nbSet[PH_NSCALE],bitcast<u32>(rmax));"), "the setup reduces the natives' r0 into NSCALE");
  assert.ok(code.includes("let nb=min(tolR,max(0.1*nr,floorR));atomicStore(&nbSet[PH_NSCALE],bitcast<u32>(select(1.0,tolR/nb,nb>0.0&&nb<tolR)));"), "nBound and nStopScale");
  assert.ok(code.includes("pGate(bitcast<f32>(rmax)>tolR||nr>floorR);"), "a native start above the floor runs");
  const scaled = code.split("abs(r)/64.0*bitcast<f32>(atomicLoad(&nbSet[PH_NSCALE]))").length - 1;
  assert.ok(scaled >= 2, `the update's and the walls' native reductions are scaled (${scaled})`);
});

test("pressure native stop: the native residual is formed in face-difference form and carried by the recurrence", () => {
  const code = codes();
  assert.ok(code.includes("fn pApplyNX(p:vec3i,cs:u32,i:u32,T:vec3i,sm:vec2f)->f32{"), "pApplyNX is the difference form");
  assert.ok(code.includes("return (pP(NB_PRES_C_DIAG,cs,i)-ks)*xi+r-sm.x;"), "(D - sum k) x_i + sum k (x_i - x_n) - seam");
  assert.ok(code.includes("r=pres[NB_PRES_C_R+at]-pAlpha*pres[NB_PRES_C_Q+at];"), "the update's native residual is the recurrence");
  assert.ok(!code.includes("pApplyNR"), "no recomputed native residual from the total pressure");
});

test("pressure native stop: f32 round-off of a hydrostatic native row, D x form against the face differences", () => {
  // A 4h interior row of the 1 cm tank (x = p dt / rho; D 24, six faces k 4), depths 5 cm..2.55 m, x quantised to f32.
  const f = Math.fround, h = 0.01, dt = 1 / 60, g = 9.81, step = f(g * 4 * h * dt), band = (r: number) => r / 64 / (h * h);
  let dx = 0, diff = 0;
  for (let j = 0; j < 20000; j++) {
    const xi = f(g * (0.05 + 2.5 * j / 20000) * dt), n = [f(xi - step), f(xi + step), xi, xi, xi, xi], k = 4, D = 24;
    const exact = D * xi - n.reduce((s, v) => s + k * v, 0);
    let a = f(D * xi); for (const v of n) a = f(a - f(k * v));
    let b = 0, ks = 0; for (const v of n) { b = f(b + f(k * f(xi - v))); ks = f(ks + k); }
    b = f(f(f(D - ks) * xi) + b);
    dx = Math.max(dx, Math.abs(a - exact)); diff = Math.max(diff, Math.abs(b - exact));
  }
  assert.ok(band(dx) > 1e-4, `the D x form's noise is over the floor (${band(dx)})`);
  assert.ok(band(diff) < 1e-6, `the face differences are exact to well under the floor (${band(diff)})`);
});
