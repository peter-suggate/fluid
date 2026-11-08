/**
 * The Float64 contact oracle against finite differences, against itself (set
 * semantics of Algorithms 1-2 on configurations with known answers), the
 * Divide-and-Truncate guarantee against continuous brute force, and the
 * setup-time topology the GPU detector reads.
 */
import assert from "node:assert/strict";
import test from "node:test";
import {
  activation, add, bruteForceDetection, closestPointTriangle, closestSegmentSegment, contactEnergy, contactWeights, countIntersections,
  frictionF0, frictionF1OverY, merge, normalGradientHessian, rng, sheet, sub, vertexForceHessian, dot, norm, scale,
  bruteForceDatPairs, isotropicTruncate, motionPenetrates, planarTruncate,
  type ContactParams, type OracleContact, type OracleMesh, type V3,
} from "./cloth-contact-oracle";
import { buildOgcEdges, buildOgcTopology, OGC_NONE } from "../lib/cloth/contact/ogc-topology";

const close = (a: number, b: number, tol: number, msg?: string) => assert.ok(Math.abs(a - b) <= tol * Math.max(1, Math.abs(a), Math.abs(b)), `${msg ?? ""} ${a} vs ${b}`);

test("two-stage activation is C2 at tau = r/2 and vanishes with its derivatives at r", () => {
  const r = 2e-3, k = 1e5, tau = r / 2, h = 1e-9;
  const lo = activation(tau - h, r, k), hi = activation(tau + h, r, k);
  close(lo[0], hi[0], 1e-6); close(lo[1], hi[1], 1e-5); close(lo[2], hi[2], 1e-5);
  const end = activation(r - 1e-12, r, k);
  assert.ok(Math.abs(end[0]) < 1e-12 && Math.abs(end[1]) < 1e-6);
  close(end[2], k, 1e-9, "the quadratic stage keeps k up to r (as OGC eq. 18)");
  for (const d of [0.1 * r, 0.3 * r, 0.6 * r, 0.9 * r]) {
    const e = 1e-8 * r, [g, g1, g2] = activation(d, r, k);
    close(g1, (activation(d + e, r, k)[0] - activation(d - e, r, k)[0]) / (2 * e), 1e-5, "g'");
    close(g2, (activation(d + e, r, k)[1] - activation(d - e, r, k)[1]) / (2 * e), 1e-5, "g''");
    assert.ok(g > 0 && g1 < 0 && g2 > 0);
  }
  // The log stage diverges as the gap closes: no finite force can push through.
  assert.ok(activation(1e-9, r, k)[1] < -1e3 * k * r);
});

test("IPC friction f0 is C1 with f1 = f0'", () => {
  const eps = 1e-4;
  for (const y of [1e-6, 3e-5, 9e-5, 2e-4]) {
    const h = 1e-10;
    close(frictionF1OverY(y, eps) * y, (frictionF0(y + h, eps) - frictionF0(y - h, eps)) / (2 * h), 1e-5);
  }
  close(frictionF0(eps * (1 - 1e-12), eps), eps, 1e-9);
});

test("closest points: triangle features and segment pairs", () => {
  const a: V3 = [0, 0, 0], b: V3 = [1, 0, 0], c: V3 = [0, 1, 0];
  assert.equal(closestPointTriangle([0.2, 0.2, 1], a, b, c).feature, 6);
  assert.equal(closestPointTriangle([-1, -1, 0.1], a, b, c).feature, 0);
  assert.equal(closestPointTriangle([2, -0.1, 0], a, b, c).feature, 1);
  assert.equal(closestPointTriangle([0.5, -1, 0], a, b, c).feature, 3);
  assert.equal(closestPointTriangle([1, 1, 0], a, b, c).feature, 4);
  assert.equal(closestPointTriangle([-1, 0.5, 0], a, b, c).feature, 5);
  const q = closestSegmentSegment([0, 0, 0], [1, 0, 0], [0.25, -1, 0.5], [0.25, 1, 0.5]);
  close(q.s, 0.25, 1e-12); close(q.t, 0.5, 1e-12); close(q.d, 0.5, 1e-12);
});

function fdGradient(f: (X: V3[]) => number, X: V3[], i: number, h: number): V3 {
  const g: V3 = [0, 0, 0];
  for (let k = 0; k < 3; k++) {
    const P = X.map(p => [...p] as V3), M = X.map(p => [...p] as V3);
    P[i][k] += h; M[i][k] -= h;
    g[k] = (f(P) - f(M)) / (2 * h);
  }
  return g;
}

function contactCase(kind: "vf" | "ee", feature: number, seed: number) {
  const R = rng(seed), j = () => (R() - 0.5) * 2e-4;
  const p: ContactParams = { r: 2e-3, k: 1e5, mu: 0.3, epsV: 1e-2, dt: 1 / 60 };
  let X: V3[];
  if (kind === "vf") {
    const tri: V3[] = [[0, 0, 0], [0.01, 0, 0], [0, 0.01, 0]];
    const over: V3 = feature === 6 ? [0.003, 0.002, 1.2e-3] : feature === 3 ? [0.005, -0.4e-3, 1.0e-3] : [-0.5e-3, -0.4e-3, 0.8e-3];
    X = [over, ...tri].map(q => [q[0] + j(), q[1] + j(), q[2] + j()] as V3);
  } else {
    X = [[0, 0, 0], [0.01, 0, 0], [0.004, -0.005, 1.1e-3], [0.0045, 0.005, 1.3e-3]].map(q => [q[0] + j(), q[1] + j(), q[2] + j()] as V3);
  }
  const Xt = X.map(q => [q[0] + 4e-4 * (R() - 0.5), q[1] + 4e-4 * (R() - 0.5), q[2]] as V3);
  const c: OracleContact = { kind, ids: [0, 1, 2, 3], feature, s: 0.42, t: 0.47, lambda: 35 };
  return { X, Xt, c, p };
}

test("pair energy: analytic gradient, Hessian and friction force against finite differences", () => {
  const cases = [["vf", 6], ["vf", 3], ["vf", 0], ["ee", 0]] as const;
  for (const [kind, feature] of cases) for (let seed = 1; seed <= 4; seed++) {
    const { X, Xt, c, p } = contactCase(kind, feature, seed);
    const h = 1e-8;
    const normalOnly = (P: V3[]) => contactEnergy(c, P, Xt, { ...p, mu: 0 });
    for (let role = 0; role < 4; role++) {
      const { grad, hess } = normalGradientHessian(c, X, p, role);
      const fd = fdGradient(normalOnly, X, role, h);
      for (let k = 0; k < 3; k++) close(grad[k], fd[k], 2e-5, `${kind}${feature} role ${role} grad`);
      // The full gradient with friction's basis frozen at this point is minus the accumulated force.
      const w = contactWeights(c, X);
      const g = w.reduce<V3>((s, wi, i) => add(s, scale(X[c.ids[i]], wi)), [0, 0, 0]), n = scale(g, 1 / norm(g));
      const frozen = (P: V3[]) => contactEnergy(c, P, Xt, p, { w, n });
      const fdf = fdGradient(frozen, X, role, h);
      const { f, H } = vertexForceHessian(c, X, Xt, p, role);
      for (let k = 0; k < 3; k++) close(-f[k], fdf[k], 2e-4, `${kind}${feature} role ${role} force`);
      // Hessian: exact where the weights are constant (vertex features, edge-edge); in the
      // normal direction of the contacting vertex it is the Gauss-Newton block VBD uses.
      const fdH = [0, 1, 2].map(k => {
        const P = X.map(q => [...q] as V3), M = X.map(q => [...q] as V3); P[role][k] += 1e-9; M[role][k] -= 1e-9;
        const gp = normalGradientHessian(c, P, p, role).grad, gm = normalGradientHessian(c, M, p, role).grad;
        return [0, 1, 2].map(i => (gp[i] - gm[i]) / 2e-9);
      });
      const constantWeights = kind === "ee" || feature < 3;
      if (constantWeights) for (let i = 0; i < 3; i++) for (let k = 0; k < 3; k++) close(hess[i][k], fdH[k][i], 1e-3, `${kind}${feature} H`);
      if (role === 0 && kind === "vf") {
        const Hn = dot(n, [0, 1, 2].map(i => dot(n, fdH[i] as V3)) as V3);
        const pn = { ...p, mu: 0 }, Hgn = vertexForceHessian(c, X, Xt, pn, 0).H;
        close(dot(n, Hgn.map(row => dot(row as V3, n)) as V3), Hn, 1e-4, "normal curvature");
      }
      // The accumulated block is symmetric positive semidefinite.
      for (let i = 0; i < 3; i++) for (let k = 0; k < 3; k++) close(H[i][k], H[k][i], 1e-12);
      for (const dir of [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1], [1, -2, 0.5]] as V3[])
        assert.ok(dot(dir, H.map(row => dot(row as V3, dir)) as V3) >= -1e-9);
    }
  }
});

test("brute-force detection: offset parallel sheets give each vertex exactly one facet contact", () => {
  const hgrid = 0.01, gap = 1.2e-3, r = 2e-3;
  const bottom = sheet(6, 6, hgrid), top = sheet(6, 6, hgrid, [0.31 * hgrid, 0.27 * hgrid, gap]);
  const mesh = merge(bottom, top);
  const nb = bottom.x.length;
  const rq = mesh.x.map(() => 3e-3);
  const det = bruteForceDetection(mesh, r, rq, 0.45);
  const perVertex = new Map<number, string[]>();
  for (const key of det.vf) { const [v, f] = key.split("|"); perVertex.set(+v, [...(perVertex.get(+v) ?? []), f]); }
  // Every top vertex over the bottom sheet's interior touches exactly one face.
  for (let v = nb; v < mesh.x.length; v++) {
    const p = mesh.x[v];
    if (p[0] < 6 * hgrid && p[1] < 6 * hgrid) assert.deepEqual(perVertex.get(v)?.map(f => f[0]), ["f"], `top vertex ${v}`);
  }
  for (let v = 0; v < nb; v++) {
    const p = mesh.x[v];
    if (p[0] > 0.31 * hgrid && p[1] > 0.27 * hgrid) assert.deepEqual(perVertex.get(v)?.map(f => f[0]), ["f"], `bottom vertex ${v}`);
  }
  // Parallel sheets: no edge pair has both closest points inside both edges at distance < r
  // except crossing diagonals, which are genuine edge-edge contacts.
  for (const key of det.ee) { const [e, f] = key.split("|").map(Number); assert.ok(det.ee.has(`${f}|${e}`), "edge-edge pairs are symmetric"); }
  // Bounds: the gap is the closest distance, so b = gamma_p * gap for vertices over the other sheet.
  for (let v = nb; v < mesh.x.length; v++) assert.ok(det.bound[v] <= 0.45 * gap + 1e-12);
  assert.equal(countIntersections(mesh), 0);
});

test("brute-force detection: a vertex near a shared edge or vertex contacts that feature once", () => {
  const base = sheet(2, 2, 0.01);
  // Fold the sheet: lift the far column so the middle column of edges is a convex ridge.
  const ridge = base.x.map(p => [p[0], p[1], p[0] > 0.015 ? -(p[0] - 0.01) : p[0] > 0.005 ? 0 : -(0.01 - p[0])] as V3);
  const folded = { ...base, x: ridge };
  const probeAboveEdge: V3 = [0.01, 0.004, 1.0e-3];
  const probeAboveVertex: V3 = [0.01, 0.01, 1.0e-3];
  for (const probe of [probeAboveEdge, probeAboveVertex]) {
    const single = { x: [probe, [probe[0] + 0.02, probe[1], probe[2]] as V3, [probe[0], probe[1] + 0.02, probe[2]] as V3], tris: [[0, 1, 2]] as [number, number, number][], edges: [] as [number, number, number, number][] };
    const mesh = merge(folded, single);
    const v = folded.x.length;
    const det = bruteForceDetection(mesh, 2e-3, mesh.x.map(() => 4e-3), 0.45);
    const mine = [...det.vf].filter(k => k.startsWith(`${v}|`));
    assert.equal(mine.length, 1, `probe ${probe}: ${mine}`);
    assert.equal(mine[0][mine[0].indexOf("|") + 1], probe === probeAboveEdge ? "e" : "v");
  }
});

test("intersection oracle: crossing sheets intersect, offset sheets do not", () => {
  const a = sheet(4, 4, 0.01);
  const b = sheet(4, 4, 0.01, [0.013, 0.007, 0]);
  const tilted = { ...b, x: b.x.map(p => [p[0], p[1], (p[0] - 0.03) * 0.5] as V3) };
  assert.ok(countIntersections(merge(a, tilted)) > 0);
  const lifted = { ...b, x: b.x.map(p => [p[0], p[1], 1e-3] as V3) };
  assert.equal(countIntersections(merge(a, lifted)), 0);
});

test("OGC topology: CSR stars, face-edge table and face0 ownership", () => {
  const m = sheet(3, 2, 0.01);
  const tris = new Uint32Array(m.tris.flatMap(t => [...t, 0]));
  const edges = buildOgcEdges(tris);
  const pos = new Float32Array(m.x.flatMap(p => [...p, 0]));
  const { layout: L, words } = buildOgcTopology({ vertexCount: m.x.length, triangles: tris, edges }, pos);
  assert.equal(L.edgeCount, m.edges.length);
  for (let t = 0; t < L.triangleCount; t++) {
    const mask = words[L.triEdgesOffset + 4 * t + 3];
    for (let k = 0; k < 3; k++) {
      const e = words[L.triEdgesOffset + 4 * t + k];
      const a = tris[4 * t + k], b = tris[4 * t + (k + 1) % 3];
      assert.deepEqual([edges[4 * e], edges[4 * e + 1]].sort(), [a, b].sort());
      assert.equal(!!(mask & (1 << k)), edges[4 * e + 2] === t);
    }
  }
  for (let v = 0; v < L.vertexCount; v++) {
    const faces = Array.from(words.subarray(L.vtListOffset + words[L.vtOffOffset + v], L.vtListOffset + words[L.vtOffOffset + v + 1]));
    assert.deepEqual(faces, m.tris.flatMap((t, f) => t.includes(v) ? [f] : []));
    const es = Array.from(words.subarray(L.veListOffset + words[L.veOffOffset + v], L.veListOffset + words[L.veOffOffset + v + 1]));
    for (const e of es) assert.ok(edges[4 * e] === v || edges[4 * e + 1] === v);
  }
  close(L.meanEdgeLength, m.edges.reduce((s, e) => s + norm(sub(m.x[e[0]], m.x[e[1]])), 0) / m.edges.length, 1e-6);
  assert.throws(() => buildOgcTopology({ vertexCount: m.x.length, triangles: tris, edges: edges.slice(0, edges.length - 4) }, pos), /missing from the edge table/);
  const bad = Uint32Array.from([0, 1, 2, 0, 0, 1, 3, 0, 0, 1, 4, 0]);
  assert.throws(() => buildOgcEdges(bad), /more than two faces/);
  assert.equal(edges[4 * 0 + 3] === OGC_NONE || edges[4 * 0 + 3] < L.triangleCount, true);
});

// ---------------------------------------------------------------- Divide and Truncate

/** Two jittered sheets 3 mm apart and a tilted third crossing above them: about 110 vertices. */
function datScene(seed: number): OracleMesh {
  const R = rng(seed), h = 6e-3;
  const a = sheet(4, 4, h, [0, 0, 0]), b = sheet(4, 4, h, [1e-3, 2e-3, 3e-3]), c = sheet(3, 3, h, [4e-3, -2e-3, 7e-3]);
  for (const v of c.x) v[2] += 0.4 * (v[0] - 4e-3);
  const m = merge(a, b, c);
  for (const v of m.x) for (let k = 0; k < 3; k++) v[k] += (R() - 0.5) * 0.6e-3;
  return m;
}

/** Large random displacements, layers driven toward each other, plus tangential noise. */
function datMotion(m: OracleMesh, seed: number, magnitude: number): V3[] {
  const R = rng(seed + 101);
  return m.x.map(([x, y, z]) => {
    const toward = z < 1.5e-3 ? 1 : z < 5e-3 ? (R() < 0.5 ? -1 : 1) : -1;
    return [x + (R() - 0.5) * magnitude, y + (R() - 0.5) * magnitude, z + toward * R() * magnitude] as V3;
  });
}

const envelopes = (m: OracleMesh, Xnew: V3[], r: number, s: number, rqBase: number, rqMax: number) =>
  m.x.map((x, i) => Math.min(rqMax, Math.max(rqBase, r + s * norm(sub(Xnew[i], x)))));

test("DAT condition: every primitive pair left unpaired is further apart than its two isotropic caps", () => {
  const gammaR = 0.9;
  for (const seed of [1, 2, 3]) {
    const m = datScene(seed), Xnew = datMotion(m, seed, 8e-3);
    const rq = envelopes(m, Xnew, 1e-3, 1.5, 1.5e-3, 2e-2);
    const { vt, ee } = bruteForceDatPairs(m, rq, 0);
    for (let v = 0; v < m.x.length; v++) for (let t = 0; t < m.tris.length; t++) {
      const tri = m.tris[t];
      if (tri.includes(v) || vt.has(`${v}|${t}`)) continue;
      const { bary } = closestPointTriangle(m.x[v], m.x[tri[0]], m.x[tri[1]], m.x[tri[2]]);
      const cp = add(add(scale(m.x[tri[0]], bary[0]), scale(m.x[tri[1]], bary[1])), scale(m.x[tri[2]], bary[2]));
      const d = norm(sub(m.x[v], cp)), reach = 0.5 * gammaR * (rq[v] + Math.max(...tri.map(i => rq[i])));
      assert.ok(reach < d, `unpaired vt ${v}|${t}: caps ${reach} vs distance ${d}`);
    }
    for (let e = 0; e < m.edges.length; e++) for (let f = e + 1; f < m.edges.length; f++) {
      const [a0, a1] = m.edges[e], [b0, b1] = m.edges[f];
      if (a0 === b0 || a0 === b1 || a1 === b0 || a1 === b1 || ee.has(`${e}|${f}`)) continue;
      const d = closestSegmentSegment(m.x[a0], m.x[a1], m.x[b0], m.x[b1]).d;
      const reach = 0.5 * gammaR * (Math.max(rq[a0], rq[a1]) + Math.max(rq[b0], rq[b1]));
      assert.ok(reach < d, `unpaired ee ${e}|${f}: caps ${reach} vs distance ${d}`);
    }
  }
});

test("Planar-DAT and isotropic DAT keep large random motions penetration-free; the raw motions penetrate", () => {
  const gammaR = 0.9, gammaP = 0.45, sepEps = 1e-6, r = 1e-3;
  let rawPenetrations = 0;
  for (const seed of [1, 2, 3, 4]) {
    const m = datScene(seed), X = m.x, Xnew = datMotion(m, seed, 8e-3);
    if (motionPenetrates(m, X, Xnew)) rawPenetrations++;
    const rq = envelopes(m, Xnew, r, 1.5, 1.5e-3, 2e-2);
    const { pairs } = bruteForceDatPairs(m, rq, 0);
    const planar = planarTruncate(X, Xnew, pairs, rq.map(q => 0.5 * gammaR * q), gammaR, sepEps);
    assert.equal(motionPenetrates(m, X, planar.x), false, `planar seed ${seed}`);
    assert.equal(countIntersections({ ...m, x: planar.x }), 0);
    const iso = isotropicTruncate(X, Xnew, bruteForceDetection(m, r, rq, gammaP).bound);
    assert.equal(motionPenetrates(m, X, iso), false, `isotropic seed ${seed}`);
    // Planar keeps more of the motion than the sphere bound.
    const kept = (x: V3[]) => x.reduce((s, p, i) => s + norm(sub(p, X[i])), 0);
    assert.ok(kept(planar.x) > kept(iso), `planar keeps ${kept(planar.x)} vs isotropic ${kept(iso)}`);
  }
  assert.ok(rawPenetrations >= 3, `the untruncated motions must penetrate (${rawPenetrations}/4)`);
});

test("Planar-DAT leaves tangential sliding and separation untouched where the sphere bound damps them", () => {
  const gammaR = 0.9, r = 1e-3, sepEps = 1e-6;
  const lower = sheet(4, 4, 6e-3, [0, 0, 0]), upper = sheet(4, 4, 6e-3, [0, 0, 2e-3]);
  const m = merge(lower, upper), X = m.x, n = lower.x.length;
  for (const [label, d] of [["sliding", [3e-3, 1e-3, 0]], ["separating", [1e-3, 0, 2.5e-3]]] as const) {
    const Xnew = X.map((x, i) => (i < n ? x : add(x, d as unknown as V3)));
    // s = 2.5 > 2 / gamma_r: the cap 0.5 gamma_r r_q,v exceeds the vertex's own motion.
    const rq = envelopes(m, Xnew, r, 2.5, 1.5e-3, 2e-2);
    const { pairs } = bruteForceDatPairs(m, rq, 0);
    assert.ok(pairs.length > 0);
    const planar = planarTruncate(X, Xnew, pairs, rq.map(q => 0.5 * gammaR * q), gammaR, sepEps);
    assert.ok(planar.tv.every(t => t === 1), label);
    planar.x.forEach((p, i) => assert.ok(norm(sub(p, Xnew[i])) < 1e-15, `${label}: vertex ${i} moved fully`));
    const iso = isotropicTruncate(X, Xnew, bruteForceDetection(m, r, rq, 0.45).bound);
    assert.ok(norm(sub(iso[n], X[n])) < 0.5 * norm(sub(Xnew[n], X[n])), `${label}: the sphere bound damps`);
  }
  // Approaching layers: each keeps a share of the gap proportional to its own approach.
  const Xnew = X.map((x, i) => add(x, [0, 0, i < n ? 0.5e-3 : -3e-3]));
  const rq = envelopes(m, Xnew, r, 2.5, 1.5e-3, 2e-2);
  const planar = planarTruncate(X, Xnew, bruteForceDatPairs(m, rq, 0).pairs, rq.map(q => 0.5 * gammaR * q), gammaR, sepEps);
  const gapAfter = planar.x[n][2] - planar.x[0][2];
  assert.ok(gapAfter > 0.09 * 2e-3 && gapAfter < 0.5 * 2e-3, `gap after ${gapAfter}`);
});
