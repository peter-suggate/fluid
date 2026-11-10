/**
 * The cloth's static solid field and its contact terms in Float64: the brick
 * build (band, Lipschitz bound, refusal of sub-cell features, local rebuild
 * after an edit), the ball bound's certificate against dense sampling of the
 * trilinear field (sharp box edges and corners, a sphere), the truncation that
 * keeps phi >= (1 - gamma_r) phi0 while leaving sliding on flat faces whole,
 * and the vertex-solid energy against finite differences.
 */
import assert from "node:assert/strict";
import test from "node:test";
import { buildClothSolidField, clothSolidClearance, clothSolidSampleEdges, clothSolidSamples, rebuildClothSolidField, sampleClothSolidField, type ClothSolidPart, type ClothSolidSource } from "../lib/cloth/contact/solid-field";
import {
  add, boxSdf, dot, faceInsideSolid, norm, rng, scale, solidBallBound, solidBoundDrop, solidContact, solidKeep, solidTruncate, sphereSdf, sub,
  type ContactParams, type V3,
} from "./cloth-contact-oracle";

const box = boxSdf([0.02, 0.02, 0], [0.08, 0.06, 0.03]), ball = sphereSdf([0.11, 0.05, 0], 0.025);
const scene = (p: readonly number[]) => Math.min(box(p), ball(p));
const source = (sdf = scene): ClothSolidSource => ({ sdf: p => sdf(p), lo: [-0.02, -0.02, -0.03], hi: [0.18, 0.12, 0.1], cell: 0.005, band: 0.06 });
const field = buildClothSolidField(source());
const phiAt = (p: V3) => sampleClothSolidField(field, p).phi;
const unit = (R: () => number): V3 => { const v: V3 = [R() - 0.5, R() - 0.5, R() - 0.5]; return scale(v, 1 / norm(v)); };

/** Points near the box's sharp edges and corners, its faces, and the sphere. */
function nearSolid(R: () => number, count: number): V3[] {
  const out: V3[] = [];
  const anchors: V3[] = [[0.02, 0.02, 0.03], [0.08, 0.06, 0.03], [0.05, 0.02, 0.03], [0.08, 0.04, 0.015], [0.05, 0.04, 0.03], [0.11, 0.05, 0.025], [0.11 + 0.025 * Math.SQRT1_2, 0.05, 0.025 * Math.SQRT1_2]];
  while (out.length < count) {
    const a = anchors[out.length % anchors.length], p = add(a, scale(unit(R), (out.length % 2 ? 0.006 : 0.03) * R()));
    if (scene(p) > 2e-4) out.push(p);
  }
  return out;
}

test("solid field: bricks cover the band, distance sources stay sqrt(3)-Lipschitz, missing bricks lie beyond band", () => {
  assert.ok(field.bricks.size > 0 && field.bricks.size < field.dims[0] * field.dims[1] * field.dims[2], "sparse");
  assert.ok(field.maxLipschitz <= Math.sqrt(3) * (1 + 1e-5) && field.maxLipschitz >= 1 - 1e-5, `L_max ${field.maxLipschitz}`);
  const R = rng(1);
  for (let i = 0; i < 4000; i++) {
    const p: V3 = [-0.02 + 0.2 * R(), -0.02 + 0.14 * R(), -0.03 + 0.13 * R()];
    const s = sampleClothSolidField(field, p), exact = scene(p);
    if (!s.stored) assert.ok(exact > field.band, `missing brick at ${p} holds distance ${exact}`);
    else if (Math.abs(exact) > 0.5 * field.cell) assert.equal(Math.sign(s.phi), Math.sign(exact), `sign at ${p}`);
  }
  // The box's faces lie on node planes: the interpolant is exact on them.
  for (let i = 0; i < 200; i++) {
    const p: V3 = [0.02 + 0.06 * R(), 0.02 + 0.04 * R(), 0.03 + 0.004 * R()];
    assert.ok(Math.abs(phiAt(p) - box(p)) < 1e-6, "flat face exact");
  }
  const words = field.words, f = new Float32Array(words.buffer);
  assert.equal(words[1], field.bricks.size); assert.equal(f[2], Math.fround(field.cell)); assert.equal(f[7], Math.fround(field.maxLipschitz));
});

test("solid field: a feature thinner than a cell fails at authoring", () => {
  const plate = boxSdf([0, 0, 0.0101], [0.1, 0.1, 0.0116]);
  assert.throws(() => buildClothSolidField({ ...source(plate), lo: [-0.02, -0.02, -0.02], hi: [0.12, 0.12, 0.04] }), /thinner than its 0.005 m cell/);
  const block = boxSdf([0, 0, 0], [0.1, 0.1, 0.03]), slot = boxSdf([0.02, -0.01, 0.0101], [0.08, 0.11, 0.0116]);
  assert.throws(() => buildClothSolidField({ ...source(p => Math.max(block(p), -slot(p))), lo: [-0.02, -0.02, -0.02], hi: [0.12, 0.12, 0.04] }), /thinner/, "a slot the lattice closes");
  // Sharp corners off the node lattice are rounded by the interpolant, not refused.
  const resolved = boxSdf([0.0113, 0.0071, 0.0101], [0.0413, 0.0301, 0.0201]);
  assert.doesNotThrow(() => buildClothSolidField({ ...source(resolved), lo: [-0.02, -0.02, -0.02], hi: [0.12, 0.12, 0.04] }));
});

/** A convex wedge of opening theta whose apex line runs along z through an off-lattice point, at a tilt in xy. */
function wedgeSdf(theta: number, phase: number, tilt: number) {
  const h = 0.01, ax = [0.0537 + phase * h, 0.0412 + 0.37 * phase * h], c = Math.cos(tilt), s = Math.sin(tilt);
  const m = [[Math.sin(theta / 2), Math.cos(theta / 2)], [Math.sin(theta / 2), -Math.cos(theta / 2)]].map(([x, y]) => [x * c - y * s, x * s + y * c]);
  return (p: readonly number[]) => {
    const dx = p[0] - ax[0], dy = p[1] - ax[1];
    return Math.max(dx * m[0][0] + dy * m[0][1], dx * m[1][0] + dy * m[1][1]);
  };
}

test("solid field resolution rule: convex edges of 35 degrees pass at every lattice phase, 20 degrees fail; a 67 degree cone rim passes", () => {
  const build = (sdf: (p: readonly number[]) => number) => buildClothSolidField({ sdf, lo: [-0.05, -0.05, -0.02], hi: [0.15, 0.13, 0.05], cell: 0.01, band: 0.03 });
  for (const phase of [0, 0.25, 0.5, 0.75]) for (const tilt of [0, 0.3, 0.7, 1.1]) {
    assert.doesNotThrow(() => build(wedgeSdf(35 * Math.PI / 180, phase, tilt)), `35 degrees at phase ${phase}, tilt ${tilt}`);
    assert.throws(() => build(wedgeSdf(20 * Math.PI / 180, phase, tilt)), /convex edges sharper than 35 degrees/, `20 degrees at phase ${phase}, tilt ${tilt}`);
  }
  // A frustum standing free: base disk radius 0.06 m, 67 degrees between base and side.
  const k = Math.tan(67 * Math.PI / 180), base = 0.003, top = 0.06, R = 0.06, cx = 0.0513, cz = 0.0137;
  const n = [k / Math.hypot(1, k), 1 / Math.hypot(1, k)];
  const frustum = (p: readonly number[]) => {
    const rho = Math.hypot(p[0] - cx, p[2] - cz), y = p[1] - base;
    return Math.max(-y, y - top, (rho - R) * n[0] + y * n[1]);
  };
  assert.doesNotThrow(() => buildClothSolidField({ sdf: frustum, lo: [-0.03, -0.03, -0.07], hi: [0.13, 0.1, 0.1], cell: 0.01, band: 0.03 }));
});

test("solid field: a local rebuild after a cut equals a full build of the cut solid", () => {
  const hole = boxSdf([0.04, 0.03, -0.01], [0.06, 0.05, 0.04]);
  const cut = (p: readonly number[]) => Math.min(Math.max(box(p), -hole(p)), ball(p));
  const local = rebuildClothSolidField(field, source(cut), [0.04, 0.03, -0.01], [0.06, 0.05, 0.04]);
  const full = buildClothSolidField(source(cut));
  assert.deepEqual([...local.bricks.keys()].sort(), [...full.bricks.keys()].sort());
  for (const [k, b] of full.bricks) assert.deepEqual(local.bricks.get(k)!.values, b.values, `brick ${k}`);
  assert.ok(phiAt([0.05, 0.04, 0.03 + 1e-3]) < 2e-3 && sampleClothSolidField(local, [0.05, 0.04, 0.02]).phi > 0, "the cut opens the solid under the point");
});

test("solid clearance table: the field is at least min(band, K_b - |x0 - c_b| - rho) within rho of x0, it certifies long motions far from the solid, and a rebuild keeps it", () => {
  // Every point of a ball, missing bricks (band) and stored ones alike, at the sampled field; the
  // floor and ball scene at the 1 cm lattice of the scenes, where a missing brick reads 6 cm.
  const cell = 0.01, floorLo: V3 = [-1.4, -0.1, -1.4], floorHi: V3 = [1.4, 0, 1.4], sphere = sphereSdf([0.3, 0.3, -0.2], 0.25);
  const parts: ClothSolidPart[] = [{ kind: "lattice-box", lo: floorLo, hi: floorHi, mu: 0.3 }, { kind: "sdf", sdf: p => sphere(p), lo: [0.05, 0.05, -0.45], hi: [0.55, 0.55, 0.05], mu: 1 }];
  const f = buildClothSolidField({ lo: [-1.44, -0.16, -1.44], hi: [1.44, 1.2, 1.44], cell, band: 0.06, parts });
  const R = rng(11);
  let checked = 0, certified = 0;
  for (let i = 0; i < 4000; i++) {
    const x0: V3 = [-1.3 + 2.6 * R(), 0.01 + 1.1 * R(), -1.3 + 2.6 * R()], rho = 0.3 * R() * R(), k = clothSolidClearance(f, x0);
    for (let j = 0; j < 8; j++) {
      const y = add(x0, scale(unit(R), rho * Math.cbrt(R()))), phi = sampleClothSolidField(f, y).phi;
      assert.ok(phi >= Math.min(f.band, k - rho) - 1e-9, `field ${phi} at ${y} below the clearance bound ${k - rho} of x0 ${x0}, rho ${rho}`);
      checked++;
    }
    if (k - rho >= 2e-3) certified++;
  }
  // A vertex 0.5 m above the floor and clear of the ball: a 0.1 m motion is certified, where the
  // bricks' own rule (min(phi, band) - L_max rho >= r) stops at (0.06 - r) / L_max = 3.3 cm.
  const high: V3 = [-0.8, 0.5, 0.8];
  assert.ok(clothSolidClearance(f, high) - 0.1 >= 2e-3, `clearance ${clothSolidClearance(f, high)} at 0.5 m`);
  assert.ok(Math.min(sampleClothSolidField(f, high).phi, f.band) - f.maxLipschitz * 0.1 < 2e-3, "the brick rule alone cannot certify it");
  // At the floor the table certifies nothing it should not.
  assert.ok(clothSolidClearance(f, [-0.8, 0.004, 0.8]) < 2e-3);
  // The edit path rebuilds the table from the edited source.
  const hole = boxSdf([-0.2, -0.1, -0.2], [0.2, 0, 0.2]);
  const cutParts: ClothSolidPart[] = [{ kind: "sdf", sdf: p => Math.max(boxSdf(floorLo, floorHi)(p), -hole(p)), lo: floorLo, hi: floorHi, mu: 0.3 }, parts[1]!];
  const src = { lo: [-1.44, -0.16, -1.44] as V3, hi: [1.44, 1.2, 1.44] as V3, cell, band: 0.06, parts: cutParts };
  const rebuilt = rebuildClothSolidField(f, src, [-0.2, -0.1, -0.2], [0.2, 0, 0.2]), fresh = buildClothSolidField(src);
  assert.deepEqual(rebuilt.clearance, fresh.clearance);
  assert.ok(clothSolidClearance(rebuilt, [0, -0.05, 0]) > clothSolidClearance(f, [0, -0.05, 0]), "the cut opens the clearance inside it");
  console.log(`# clearance: ${checked} points checked, ${certified} of 4000 balls certified by the table`);
});

test("solid ball bound: phi(x0 + d) >= phi0 + g0.d - D.|d| over the ball (per-axis gradient deviation), at sharp edges, corners and the sphere", () => {
  const R = rng(7), r = 2e-3;
  let near = 0, far = 0;
  for (const x0 of nearSolid(R, 300)) {
    const rho = 0.002 + 0.016 * R(), b = solidBallBound(field, x0, rho, r);
    if (b.far) far++; else near++;
    for (let k = 0; k < 120; k++) {
      const d = scale(unit(R), rho * Math.cbrt(R())), phi = phiAt(add(x0, d));
      if (b.far) assert.ok(phi >= r - 1e-9, `far ball reaches ${phi} at ${x0}`);
      else assert.ok(phi >= b.phi0 + dot(b.g0, d) - solidBoundDrop(b, d) - 1e-9, `bound at ${x0} + ${d}: ${phi}`);
    }
  }
  assert.ok(near > 100 && far > 10, `${near} near, ${far} far`);
});

test("solid ball bound at the lattice's edge: cells outside the field's box are no missing bricks, so a ball reaching past a floor on the box's face is cut, not certified far", () => {
  // The scenes' container floor: a lattice-box just below the field's box, its top face on the box's;
  // the scenes' band r + 1.5 L_max rqMax at rqMax 4 cm, and rho <= rqMax / 2.
  const cell = 0.01, r = 2e-3, gammaR = 0.9, rqMax = 0.04;
  const f = buildClothSolidField({ lo: [-0.4, 0, -0.4], hi: [0.4, 0.6, 0.4], cell, band: r + 1.5 * Math.sqrt(3) * rqMax, parts: [{ kind: "lattice-box", lo: [-0.5, -0.02, -0.5], hi: [0.5, 0, 0.5], mu: 0.3 }] });
  const R = rng(13);
  for (let i = 0; i < 400; i++) {
    const x0: V3 = [-0.3 + 0.6 * R(), 0.001 + 0.019 * R(), -0.3 + 0.6 * R()], rho = x0[1] + (rqMax / 2 - x0[1]) * R();
    const b = solidBallBound(f, x0, rho, r);
    assert.ok(!b.far, `ball at height ${x0[1]}, rho ${rho} certified far`);
    const d = solidTruncate(b, [0.3 * rho * (R() - 0.5), -rho, 0.3 * rho * (R() - 0.5)], gammaR, r), y = add(x0, d);
    assert.ok(y[1] >= solidKeep(b.phi0, gammaR, r) - 1e-9, `the motion from ${x0[1]} ends at ${y[1]}`);
  }
  assert.equal(f.exteriorClear, false, "the floor reaches past the box");
  // A moving body's field: its box pads the solid by band, so outside the lattice phi >= band and a
  // ball reaching past the lattice is as clear as one reaching a missing brick (certified far).
  const R0 = 0.1, band = r + 1.5 * Math.sqrt(3) * rqMax, reach = R0 + band + 2 * cell, ball = sphereSdf([0, 0, 0], R0);
  const body = buildClothSolidField({ parts: [{ kind: "sdf", sdf: p => ball(p), lo: [-R0, -R0, -R0], hi: [R0, R0, R0], mu: 0.3 }], lo: [-reach, -reach, -reach], hi: [reach, reach, reach], cell, band });
  assert.equal(body.exteriorClear, true);
  const lat = body.origin[0] + 8 * cell * body.dims[0], edge: V3 = [lat - 0.004, 0, 0], eb = solidBallBound(body, edge, 0.019, 0.1);
  assert.ok(eb.far && sampleClothSolidField(body, edge).stored, `a ball across the body lattice's face (phi ${eb.phi0}) is far`);
});

test("solid truncation keeps phi >= min((1 - gamma_r) phi0, r) along the whole motion and leaves sliding on a flat face whole", () => {
  const R = rng(9), r = 2e-3, gammaR = 0.9;
  let cut = 0;
  for (const x0 of nearSolid(R, 300)) {
    const rho = 0.002 + 0.016 * R(), b = solidBallBound(field, x0, rho, r);
    const raw = scale(unit(R), rho * (0.3 + 0.7 * R()));
    const d = solidTruncate(b, raw, gammaR, r);
    if (norm(sub(d, raw)) > 1e-12) cut++;
    for (let s = 0; s <= 64; s++) {
      const phi = phiAt(add(x0, scale(d, s / 64)));
      assert.ok(phi >= (b.far ? r : solidKeep(b.phi0, gammaR, r)) - 1e-9, `phi ${phi} along the truncated motion from ${x0}`);
    }
  }
  assert.ok(cut > 30, `the motions cut (${cut})`);
  // Resting on the box's top face 0.4 r up: a slide ten times the gap is untouched,
  // an approach of twice the gap keeps (1 - gamma_r) of it.
  const x0: V3 = [0.05, 0.04, 0.03 + 0.4 * r], b = solidBallBound(field, x0, 0.009, r);
  assert.ok(!b.far && Math.max(...b.D) < 1e-4, `flat face D ${b.D}`);
  const slide: V3 = [8e-3, -3e-3, 0];
  assert.deepEqual(solidTruncate(b, slide, gammaR, r), slide);
  const press = solidTruncate(b, [0, 0, -0.8 * r], gammaR, r);
  assert.ok(Math.abs(phiAt(add(x0, press)) - (1 - gammaR) * 0.4 * r) < 1e-8, "approach to the planar limit");
  // An isotropic Lipschitz ball (phi0 / L_max) would have cut the slide to under a tenth.
  assert.ok(0.9 * 0.4 * r / field.maxLipschitz < 0.1 * norm(slide));
});

test("vertex-solid energy: force is -dE/dx, the Hessian is the PSD Gauss-Newton term, phi <= 0 has infinite energy", () => {
  const R = rng(3), p: ContactParams = { r: 2e-3, k: 1e5, mu: 0, epsV: 1e-2, dt: 1 / 60 };
  let active = 0;
  for (const x of nearSolid(R, 200)) {
    const c = solidContact(field, x, x, 0, p);
    if (c.e === 0) continue;
    active++;
    const h = 1e-8;
    for (let a = 0; a < 3; a++) {
      const e = [0, 0, 0] as V3; e[a] = h;
      const fd = -(solidContact(field, add(x, e), x, 0, p).e - solidContact(field, sub(x, e), x, 0, p).e) / (2 * h);
      assert.ok(Math.abs(fd - c.f[a]) <= 1e-4 * Math.max(1, Math.abs(c.f[a])), `force ${a}: ${c.f[a]} vs ${fd}`);
    }
    const v = unit(R);
    assert.ok(dot(v, [0, 1, 2].map(i => dot(c.H[i] as V3, v)) as V3) >= 0);
  }
  assert.ok(active > 40, `${active} active`);
  assert.equal(solidContact(field, [0.05, 0.04, 0.02], [0.05, 0.04, 0.02], 0, p).e, Infinity);
  // Friction (the entry's mu lambda) opposes the slide from the step start and vanishes along the normal.
  const x: V3 = [0.05, 0.04, 0.03 + 1e-3], fr = solidContact(field, add(x, [1e-3, 0, 0]), x, 0.5 * 10, p);
  const nf = solidContact(field, add(x, [1e-3, 0, 0]), x, 0, p);
  assert.ok(fr.f[0] - nf.f[0] < -4.9 && Math.abs(fr.f[2] - nf.f[2]) < 1e-9);
});

test("solid field parts: a lattice-box floor and an sdf ball match the single-sdf build, carry their own friction, and build fast", () => {
  const cell = 0.01, floorLo: V3 = [-1.4, -0.1, -1.4], floorHi: V3 = [1.4, 0, 1.4], ball = sphereSdf([0.3, 0.3, -0.2], 0.25), floor = boxSdf(floorLo, floorHi);
  const parts: ClothSolidPart[] = [
    { kind: "lattice-box", lo: floorLo, hi: floorHi, mu: 0.3 },
    { kind: "sdf", sdf: p => ball(p), lo: [0.05, 0.05, -0.45], hi: [0.55, 0.55, 0.05], mu: 1 },
  ];
  const box = { lo: [-1.44, -0.16, -1.44] as V3, hi: [1.44, 0.6, 1.44] as V3, cell, band: 0.06 };
  const t0 = performance.now(), split = buildClothSolidField({ ...box, parts }), ms = performance.now() - t0;
  const whole = buildClothSolidField({ ...box, sdf: p => Math.min(floor(p), ball(p)), mu: 0 });
  assert.deepEqual([...split.bricks.keys()].sort(), [...whole.bricks.keys()].sort());
  for (const [k, b] of whole.bricks) {
    const v = split.bricks.get(k)!.values;
    for (let i = 0; i < v.length; i++) if (b.values[i]! <= 0.06) assert.equal(v[i], b.values[i], `brick ${k} node ${i}`);
  }
  assert.equal(sampleClothSolidField(split, [-1, 0.004, 1]).mu, 0.3);
  assert.equal(sampleClothSolidField(split, [0.3, 0.554, -0.2]).mu, 1);
  assert.equal(sampleClothSolidField(whole, [0.3, 0.554, -0.2]).mu, 0);
  console.log(`# 2.8 m floor + 0.25 m ball at 1 cm: ${split.bricks.size} bricks in ${ms.toFixed(0)} ms`);
  assert.ok(ms < 3000, `build ${ms} ms`);
  assert.throws(() => buildClothSolidField({ ...box, parts: [{ kind: "lattice-box", lo: [0, 0.003, 0], hi: [0.1, 0.1, 0.1], mu: 0 }] }), /node planes/);
  assert.throws(() => buildClothSolidField({ ...box, parts: [{ kind: "mesh", lo: [0, 0, 0], hi: [1, 1, 1], mu: 0 } as unknown as ClothSolidPart] }), /Unknown cloth solid part kind/);
  assert.throws(() => buildClothSolidField({ ...box, parts: [{ ...parts[1]!, mu: 3 } as ClothSolidPart] }), /friction/);
});

test("solid surface samples: on the zero set wherever it is convex and not flat (box edges, spheres), none on flat faces or along a concave crease", () => {
  const h = 0.005;
  const at = (s: Float32Array) => Array.from({ length: s.length / 4 }, (_, i) => [s[4 * i]!, s[4 * i + 1]!, s[4 * i + 2]!] as V3);
  const pts = at(clothSolidSamples(field));
  // A sample stands out by its holes' measured bulge, at most h / 16 (phi rises up to |grad phi| <= sqrt 3 times that).
  assert.ok(pts.every(p => phiAt(p) > -1e-6 && phiAt(p) < h / 8), "just outside the zero set");
  // Box samples lie on its sharp edges (two faces' planes, where the trilinear field rounds them
  // over a cell): the faces' interiors and the rounding between carry none, and every point of
  // every edge is near one.
  const onBox = pts.filter(p => box(p) < ball(p));
  const lo = [0.02, 0.02, 0], hi = [0.08, 0.06, 0.03];
  const edgeGap = (p: V3) => { const d = [0, 1, 2].map(a => Math.min(Math.abs(p[a]! - lo[a]!), Math.abs(p[a]! - hi[a]!))).sort((x, y) => x - y); return Math.hypot(d[0]!, d[1]!); };
  const offEdge = onBox.filter(p => edgeGap(p) > h / 16);
  assert.ok(onBox.length > 100 && offEdge.length === 0, `${onBox.length} box samples, off the edges: ${offEdge.length} (${offEdge.slice(0, 3).map(p => p.map(v => v.toFixed(4)).join(" ")).join("; ")})`);
  for (let a = 0; a < 3; a++) for (const u of [lo[(a + 1) % 3]!, hi[(a + 1) % 3]!]) for (const v of [lo[(a + 2) % 3]!, hi[(a + 2) % 3]!]) {
    for (let k = 0; k <= 40; k++) {
      const p = [0, 0, 0] as unknown as V3;
      p[a] = lo[a]! + (hi[a]! - lo[a]!) * k / 40; p[(a + 1) % 3] = u; p[(a + 2) % 3] = v;
      assert.ok(onBox.some(q => norm(sub(q, p)) < 0.6 * h), `no box sample within 0.6 h of its edge point ${p.map(x => x.toFixed(4)).join(" ")}`);
    }
  }
  // The ball (it turns about 0.2 rad per cell, near the sharp turn) is covered at h to h / 2: its area over
  // h^2, up to four times that.
  const nBall = pts.length - onBox.length, ideal = 4 * Math.PI * 0.025 ** 2 / h ** 2;
  assert.ok(nBall > 0.5 * ideal && nBall < 8 * ideal, `${nBall} samples on the 2.5 cm ball, ideal ${ideal.toFixed(0)}`);
  // So is a broad sphere (any convex curvature lets a face sag below its vertices), and every point of it is near one.
  const broad = buildClothSolidField({ sdf: sphereSdf([0, 0, 0], 0.1), lo: [-0.13, -0.13, -0.13], hi: [0.13, 0.13, 0.13], cell: h, band: 0.02 });
  const bp = at(clothSolidSamples(broad)), R = rng(5);
  for (let i = 0; i < 200; i++) {
    const u = unit(R), p = scale(u, 0.1);
    assert.ok(bp.some(q => norm(sub(q, p)) < 0.75 * h), `no sample within 0.75 h of ${p}`);
  }
  // A step's inner crease (two boxes) is concave, its faces flat: samples only on the convex edges.
  const step = buildClothSolidField({ parts: [
    { kind: "lattice-box", lo: [-0.05, -0.05, -0.02], hi: [0.05, 0.05, 0], mu: 0.3 },
    { kind: "lattice-box", lo: [-0.05, -0.05, 0], hi: [0, 0.05, 0.02], mu: 0.3 },
  ], lo: [-0.08, -0.08, -0.05], hi: [0.08, 0.08, 0.05], cell: h, band: 0.02 });
  const sp = at(clothSolidSamples(step));
  // The crease ends on the convex edges of the end faces y = +-0.05.
  const inner = sp.filter(p => Math.hypot(p[0], p[2]) <= h && Math.abs(p[1]) < 0.05 - 2 * h);
  const flat = sp.filter(p => Math.abs(p[1]) < 0.05 - h && ((Math.abs(p[2] - 0.02) < 1e-6 && p[0] < -h && p[0] > -0.05 + h) || (Math.abs(p[2]) < 1e-6 && p[0] > h && p[0] < 0.05 - h)));
  assert.ok(sp.length > 100 && inner.length === 0 && flat.length === 0, `${sp.length} step samples, ${inner.length} along the inner crease, ${flat.length} on the flat tops`);
  assert.equal(clothSolidSamples(step), clothSolidSamples(step), "computed once per field");
  assert.throws(() => clothSolidSamples(field, { r: 0, edge: 0.01 }), RangeError);
  // Hull edges: each a pair i < j of neighbouring samples, once; along a convex edge each sample
  // reaches the next, so a straight segment between two samples lies within the hull edges' cover.
  const edges = clothSolidSampleEdges(step), keys = new Set<number>(), S = sp.length, deg = new Uint32Array(S);
  for (let k = 0; k < edges.length; k += 2) {
    const i = edges[k]!, j = edges[k + 1]!;
    assert.ok(i < j && j < S && !keys.has(i * S + j), `hull edge ${i}-${j}`);
    keys.add(i * S + j); deg[i]!++; deg[j]!++;
    assert.ok(norm(sub(sp[i]!, sp[j]!)) < 2 * h, `hull edge ${i}-${j} spans ${norm(sub(sp[i]!, sp[j]!))} m`);
  }
  const lone = sp.filter((_, i) => deg[i] === 0);
  assert.ok(lone.length < 0.05 * S, `${lone.length} of ${S} samples without a neighbour: ${lone.slice(0, 4).map(p => p.map(v => v.toFixed(4)).join(" ")).join("; ")}`);
  // The convex edge x = 0.05, z = -0.02: consecutive samples along it, a cell or less apart, are joined.
  const line = sp.map((p, i) => ({ p, i })).filter(({ p }) => Math.abs(p[0] - 0.05) < 1e-3 && Math.abs(p[2] + 0.02) < 1e-3).sort((a, b) => a.p[1] - b.p[1]);
  let joined = 0;
  for (let k = 1; k < line.length; k++) {
    const a = line[k - 1]!, b = line[k]!;
    if (b.p[1] - a.p[1] > h) continue;
    joined++;
    assert.ok(keys.has(Math.min(a.i, b.i) * S + Math.max(a.i, b.i)), `edge samples at y ${a.p[1]} and ${b.p[1]} are not joined`);
  }
  assert.ok(joined > 10, `${joined} consecutive pairs along the edge x = 0.05, z = -0.02 (${line.length} samples)`);
  assert.equal(clothSolidSampleEdges(step), edges, "computed once per field");
});

test("exact face-in-solid oracle: a face cutting a box edge between outside vertices is inside, one clearing it is not, and it agrees with dense sampling", () => {
  const h = 0.005, f = buildClothSolidField({ parts: [{ kind: "lattice-box", lo: [0, 0, 0], hi: [0.1, 0.05, 0.1], mu: 0.3 }], lo: [-0.05, -0.05, -0.05], hi: [0.15, 0.1, 0.15], cell: h, band: 0.03 });
  // A face in the plane x + y = 0.15 + o over the top edge (x = 0.1, y = 0.05), vertices well outside.
  const tilted = (o: number): V3[] => [[0.08 + o, 0.07, 0.03], [0.12 + o, 0.03, 0.03], [0.11 + o, 0.04, 0.07]];
  const phi = (p: V3) => sampleClothSolidField(f, p).phi;
  for (const o of [-5e-4, -2e-5]) {
    const T = tilted(o);
    assert.ok(T.every(p => phi(p) > 0.005), "vertices outside");
    assert.equal(faceInsideSolid(f, T[0]!, T[1]!, T[2]!).inside, true, `cut by ${o}`);
  }
  for (const o of [2e-5, 5e-4]) {
    const T = tilted(o), r = faceInsideSolid(f, T[0]!, T[1]!, T[2]!);
    assert.equal(r.inside, false, `clear by ${o}`);
    assert.ok(r.lower >= 0 && r.lower <= o, `lower bound ${r.lower} within the clearance ${o}`);
  }
  // Random faces near the box and the ball: inside exactly when a dense lattice finds phi < 0 (or the
  // dip is shallower than the lattice resolves); never outside where the lattice finds phi < 0.
  const R = rng(11);
  let agree = 0, finer = 0;
  for (let i = 0; i < 300; i++) {
    const c = nearSolid(R, 1)[0]!, T = [0, 1, 2].map(() => add(c, scale(unit(R), 0.004 + 0.01 * R()))) as V3[];
    const exact = faceInsideSolid(field, T[0]!, T[1]!, T[2]!), n = 48;
    let lattice = Infinity;
    for (let a = 0; a <= n; a++) for (let b = 0; b <= n - a; b++) {
      const p = [0, 1, 2].map(k => ((n - a - b) * T[0]![k]! + a * T[1]![k]! + b * T[2]![k]!) / n) as unknown as V3;
      lattice = Math.min(lattice, phiAt(p));
    }
    if (lattice < 0) assert.equal(exact.inside, true, `face ${i}: lattice phi ${lattice}`);
    else if (exact.inside) { finer++; assert.ok(lattice < 1e-4, `face ${i} inside, lattice min ${lattice}`); }
    else { agree++; assert.ok(exact.lower <= lattice + 1e-12, `face ${i}: lower ${exact.lower} above lattice ${lattice}`); }
  }
  console.log(`# exact face oracle: ${agree} outside agreeing, ${finer} inside below the lattice's resolution`);
});

// Known soundness defect (coordinator ruling 2026-11-10): the trilinear field rounds the Bridson frustum's slanted
// edge by about 2 mm, so a face can lie inside the true solid while the field reads it outside, and nothing keeps
// it out. The planned fix is C2's (C), cloth-vertex vs sample-hull-facet pairs; this test then checks the contact
// surface (the field or the samples' hull) instead of the field alone.
test("solid soundness: face 360 of the Bridson rotating sphere at tick 49 (vertex 192 1.5 mm inside the true frustum) reads inside the contact surface", {
  todo: "C2 known soundness defect: HEAD's trilinear field rounds the frustum's slanted edge ~2 mm, so face 360 passes 1.79 mm inside the true solid reading phi > 0; fix: (C) cloth-vertex vs sample-hull-facet pairs (progress-lane-c2.md)",
}, async () => {
  const { bridson2002RotatingSphere } = await import("../lib/cloth/scenes/bridson-2002");
  const { bridson2002SceneDescription } = await import("../lib/cloth/scenes/bridson-2002/scene-description");
  const { clothContactBand } = await import("../lib/cloth/scene-contact");
  const { clothSolidSourceForScene } = await import("../lib/cloth/contact/scene-solids");
  const scene = bridson2002SceneDescription(bridson2002RotatingSphere({ nodesAcross: 16 }));
  const src = clothSolidSourceForScene(scene, { band_m: clothContactBand(1e-3), frameOrigin_m: [0, 0.64, 0] }), f = buildClothSolidField(src);
  // The true solid: the union of the source's shaped (sdf) parts, the frustum among them.
  const shaped = (src.parts ?? []).flatMap(p => (p.kind === "sdf" ? [p] : []));
  const truth = (p: V3) => Math.min(...shaped.map(s => s.sdf(p)));
  // The face's scene vertices as dumped; the cloth frame is the rest pose's box centre, (0, 0.64, 0).
  const V = [[-0.49594, 0.18363, 0.49512], [-0.57303, 0.28009, 0.51783], [-0.45847, 0.23810, 0.39021]].map(p => [p[0]!, p[1]! - 0.64, p[2]!] as V3);
  assert.ok(truth(V[0]!) < -1.5e-3, `vertex 192 lies ${truth(V[0]!)} m in the frustum`);
  assert.ok(sampleClothSolidField(f, V[0]!).phi < 0, `vertex 192, ${truth(V[0]!)} m inside the frustum, reads phi ${sampleClothSolidField(f, V[0]!).phi} outside`);
  for (let i = 0; i <= 64; i++) for (let j = 0; j <= 64 - i; j++) {
    const q = [0, 1, 2].map(a => (1 - i / 64 - j / 64) * V[0]![a]! + i / 64 * V[1]![a]! + j / 64 * V[2]![a]!) as unknown as V3;
    if (truth(q) < 0) assert.ok(sampleClothSolidField(f, q).phi < 0, `face point ${q} inside the frustum reads outside the field`);
  }
});
