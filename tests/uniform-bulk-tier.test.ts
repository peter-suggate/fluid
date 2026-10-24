import assert from "node:assert/strict";
import test from "node:test";
import {
  UNIFORM_BULK_HOLD_FRAMES, UniformBulkComposite, uniformBulkAxisError, uniformBulkProlong, uniformBulkStep, uniformBulkStructural,
  type BulkVec, type UniformBulkTile,
} from "../lib/methods/uniform/uniform-bulk-tier-reference";

const axisFaces = (value: (plane: number, j: number) => number) => Float64Array.from({ length: 80 }, (_, i) => value(Math.floor(i / 16), i % 16));

test("16h remap: restrict then prolong keeps a uniform and an axis-linear field exactly", () => {
  assert.equal(uniformBulkAxisError(axisFaces(() => 0.375)), 0);
  assert.ok(uniformBulkAxisError(axisFaces(() => 0.37)) < 1e-15);
  assert.ok(uniformBulkAxisError(axisFaces(plane => 0.25 - 0.1 * plane)) < 1e-15);
  // A 4h-scale wave is what the high-pass sees.
  assert.ok(uniformBulkAxisError(axisFaces(plane => (plane & 1 ? 1 : -1))) > 0.9);
  // The prolongation's ends are the face means.
  assert.equal(uniformBulkProlong(2, 6, 0), 2);
  assert.equal(uniformBulkProlong(2, 6, 4), 6);
});

test("16h hysteresis: enter after the hold, leave at once on structure or twice the error", () => {
  let s = { member: false, hold: 0 };
  for (let i = 0; i < UNIFORM_BULK_HOLD_FRAMES - 1; i++) { s = uniformBulkStep(s, false, 0.5, 1); assert.equal(s.member, false); }
  s = uniformBulkStep(s, false, 0.5, 1); assert.equal(s.member, true);
  assert.equal(uniformBulkStep(s, false, 1.5, 1).member, true, "between one and two thresholds a member stays");
  assert.equal(uniformBulkStep(s, false, 2.5, 1).member, false);
  assert.equal(uniformBulkStep(s, true, 0, 1).member, false);
  assert.equal(uniformBulkStep({ member: false, hold: 10 }, false, 1.5, 1).hold, 0, "a failing frame restarts the hold");
});

test("16h grading: no region within a 16h cell of an h tile, air, a cut or a shallow tile", () => {
  const n = 28, minH = 1;
  const make = (h: BulkVec[], other: (t: BulkVec) => UniformBulkTile | undefined = () => undefined) => (t: BulkVec): UniformBulkTile | undefined => {
    if (t.some(v => v < 0 || v >= n)) return undefined;
    return other(t) ?? { kind: "resident", band: h.some(q => q.every((v, a) => v === t[a])), phi: -10 };
  };
  const h: BulkVec = [13, 13, 13];
  const tile = make([h]);
  for (let z = 0; z < 7; z++) for (let y = 0; y < 7; y++) for (let x = 0; x < 7; x++) {
    const eligible = uniformBulkStructural(tile, [x, y, z], minH);
    // Chebyshev distance in tiles from the h tile to the region's box.
    const gap = Math.max(...[x, y, z].map((r, a) => Math.max(4 * r - h[a]!, h[a]! - (4 * r + 3), 0)));
    assert.equal(eligible, gap > 4, `region ${x},${y},${z} (gap ${gap})`);
    // Never next to h: a 4h region parts them on faces, edges and corners.
    if (eligible) assert.ok(gap >= 5);
  }
  assert.equal(uniformBulkStructural(make([], t => t[1] > 18 ? { kind: "air" } : undefined), [3, 3, 3], minH), false, "air in the neighbourhood");
  assert.equal(uniformBulkStructural(make([], t => t[0] === 17 && t[1] === 13 ? { kind: "resident", open: 0.5, phi: -10 } : undefined), [3, 3, 3], minH), false, "a cut tile");
  assert.equal(uniformBulkStructural(make([], t => t[0] >= 16 ? { kind: "resident", open: 0, phi: -10 } : undefined), [3, 3, 3], minH), true, "closed solid tiles beside it are walls");
  assert.equal(uniformBulkStructural(make([], t => t[0] >= 12 && t[0] < 16 && t[1] === 13 ? { kind: "resident", open: 0, phi: -10 } : undefined), [3, 3, 3], minH), false, "its own tiles are open");
  assert.equal(uniformBulkStructural(make([], t => t[1] === 19 ? { kind: "resident", phi: -5 } : undefined), [3, 3, 3], minH), false, "a tile above 5.5 h deep");
  assert.equal(uniformBulkStructural(t => t[0] >= 28 ? { kind: "solid" } : make([])(t), [6, 3, 3], minH), true, "a SOLID page beside it");
});

// 3^3 regions of 4h cells; the centre member and its +x neighbour (a member to member face).
const composite = () => new UniformBulkComposite([12, 12, 12], new Set([13, 14]));
const hydrostatic = (c: UniformBulkComposite, g: number) => {
  const p = new Float64Array(c.cells);
  for (let k = 0; k < c.cells; k++) { const q = c.coord(k); p[k] = -g * (c.member(q) ? 4 * (q[1] >> 2) + 1.5 : q[1]); }
  return p;
};
const verticalForce = (c: UniformBulkComposite, g: number) => {
  const u = new Float64Array(c.faces);
  for (let k = 0; k < c.cells; k++) { const q = c.coord(k); if (q[1] > 0) u[c.face(1, q)] = -g; }
  return u;
};

test("16h composite: a hydrostatic column is an exact rest state across seams and member faces", () => {
  const c = composite(), g = 9.81, p = hydrostatic(c, g), ustar = verticalForce(c, g);
  const r = c.residual(p, p, ustar);
  assert.ok(Math.max(...r.map(Math.abs)) < 1e-9, `residual ${Math.max(...r.map(Math.abs))}`);
  const { u } = c.project(p, p, ustar);
  assert.ok(Math.max(...u.map(Math.abs)) < 1e-9, `rest speed ${Math.max(...u.map(Math.abs))}`);
  // Solving from a zero start (no heads) or the rest start returns it.
  for (const start of [p, new Float64Array(c.cells)]) {
    const solved = c.solve(start, ustar), shift = solved[0]! - p[0]!;
    const { u: v } = c.project(solved, start, ustar);
    if (start === p) assert.ok(Math.max(...solved.map((x, k) => Math.abs(x - shift - p[k]!))) < 1e-6);
    assert.ok(Math.max(...v.map(Math.abs)) < (start === p ? 1e-6 : 0.5 * g), `speed ${Math.max(...v.map(Math.abs))}`);
  }
});

test("16h composite: projection is divergence-free on native cells and each 16h cell, with a reflux seam", () => {
  const c = composite();
  let seed = 7; const random = () => { seed = (seed * 1103515245 + 12345) % 2147483648; return seed / 2147483648 - 0.5; };
  const ustar = new Float64Array(c.faces);
  for (let a = 0; a < 3; a++) for (let k = 0; k < c.cells; k++) { const q = c.coord(k); if (q[a] > 0) ustar[c.face(a, q)] = random(); }
  const start = Float64Array.from({ length: c.cells }, (_, k) => { const q = c.coord(k); return c.member(q) ? 0.3 * (q[1] >> 2) : 0.1 * q[1] + 0.05 * q[0]; });
  const p = c.solve(start, ustar, 6000, 1e-13);
  const r = c.residual(p, start, ustar);
  assert.ok(Math.max(...r.map(Math.abs)) < 1e-7, `residual ${Math.max(...r.map(Math.abs))}`);
  const { u, means } = c.project(p, start, ustar), d = c.divergence(u);
  const region = new Map<number, number>();
  let native = 0, inner = 0;
  for (let k = 0; k < c.cells; k++) {
    const q = c.coord(k);
    if (!c.member(q)) { native = Math.max(native, Math.abs(d[k]!)); continue; }
    region.set(c.region(q), (region.get(c.region(q)) ?? 0) + d[k]!);
    if (q.every(v => (v & 3) > 0 && (v & 3) < 3)) inner = Math.max(inner, Math.abs(d[k]!));
  }
  assert.ok(native < 1e-7, `native divergence ${native}`);
  for (const [r16, sum] of region) assert.ok(Math.abs(sum) < 1e-7, `16h divergence of ${r16}: ${sum}`);
  assert.ok(inner < 1e-7, `a member's inner 4h cells ${inner}`);
  // Reflux: a seam face's 16 subface fluxes (what the natives see) sum to the 16h face's flux.
  const m = means.get(13)!;
  let low = 0; for (let j = 0; j < 16; j++) low += u[c.face(0, [4, 4 + (j & 3), 4 + (j >> 2)])]!;
  assert.ok(Math.abs(low - 16 * m[0]!) < 1e-12);
  // The member to member face is one value, the same from both regions.
  assert.equal(m[1], means.get(14)![0]);
  for (let j = 1; j < 16; j++) assert.equal(u[c.face(0, [8, 4 + (j & 3), 4 + (j >> 2)])], u[c.face(0, [8, 4, 4])]);
});
