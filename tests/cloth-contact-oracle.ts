/**
 * Float64 CPU oracle for the OGC cloth contact (lib/cloth/contact). Test-only:
 * nothing in the step path may import it.
 *
 * It restates every rule the GPU detector and the vertex contact block apply,
 * in double precision and by brute force:
 *  - closest points (vertex-triangle with its Voronoi feature, segment-segment);
 *  - OGC block membership (Chen et al. 2025, eqs. 8-9) and Algorithm 1's set
 *    semantics, so the GPU pair lists can be compared as sets of (vertex, face);
 *  - edge-edge pairs whose closest points lie inside both edges;
 *  - the d_min terms and the conservative bound (eqs. 21-26) with per-vertex
 *    query radii;
 *  - the two-stage C2 activation (eqs. 18-20), the pair energy, its exact
 *    gradient and Hessian, and IPC lagged friction with VBD's stabilised
 *    Hessian;
 *  - exact triangle-triangle intersection between non-adjacent faces.
 */

export type V3 = [number, number, number];

export const sub = (a: V3, b: V3): V3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
export const add = (a: V3, b: V3): V3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
export const scale = (a: V3, s: number): V3 => [a[0] * s, a[1] * s, a[2] * s];
export const dot = (a: V3, b: V3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export const cross = (a: V3, b: V3): V3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
export const norm = (a: V3) => Math.hypot(a[0], a[1], a[2]);

/** Feature codes shared with the WGSL: vertex k (0..2), edge k = (t[k], t[k+1]) (3..5), face (6). */
export const FEATURE_FACE = 6;

/** Ericson's closest point on a triangle, with the Voronoi feature it lies on. */
export function closestPointTriangle(p: V3, a: V3, b: V3, c: V3): { bary: V3; feature: number } {
  const ab = sub(b, a), ac = sub(c, a), ap = sub(p, a);
  const d1 = dot(ab, ap), d2 = dot(ac, ap);
  if (d1 <= 0 && d2 <= 0) return { bary: [1, 0, 0], feature: 0 };
  const bp = sub(p, b), d3 = dot(ab, bp), d4 = dot(ac, bp);
  if (d3 >= 0 && d4 <= d3) return { bary: [0, 1, 0], feature: 1 };
  const vc = d1 * d4 - d3 * d2;
  if (vc <= 0 && d1 >= 0 && d3 <= 0) { const v = d1 / (d1 - d3); return { bary: [1 - v, v, 0], feature: 3 }; }
  const cp = sub(p, c), d5 = dot(ab, cp), d6 = dot(ac, cp);
  if (d6 >= 0 && d5 <= d6) return { bary: [0, 0, 1], feature: 2 };
  const vb = d5 * d2 - d1 * d6;
  if (vb <= 0 && d2 >= 0 && d6 <= 0) { const w = d2 / (d2 - d6); return { bary: [1 - w, 0, w], feature: 5 }; }
  const va = d3 * d6 - d5 * d4;
  if (va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0) { const w = (d4 - d3) / ((d4 - d3) + (d5 - d6)); return { bary: [0, 1 - w, w], feature: 4 }; }
  const denom = 1 / (va + vb + vc);
  const v = vb * denom, w = vc * denom;
  return { bary: [1 - v - w, v, w], feature: FEATURE_FACE };
}

/** Ericson's closest points between segments p0p1 and q0q1: parameters and distance. */
export function closestSegmentSegment(p0: V3, p1: V3, q0: V3, q1: V3): { s: number; t: number; d: number } {
  const d1 = sub(p1, p0), d2 = sub(q1, q0), rr = sub(p0, q0);
  const a = dot(d1, d1), e = dot(d2, d2), f = dot(d2, rr);
  const c = dot(d1, rr), b = dot(d1, d2), denom = a * e - b * b;
  let s = denom > 0 ? Math.min(1, Math.max(0, (b * f - c * e) / denom)) : 0;
  let t = (b * s + f) / e;
  if (t < 0) { t = 0; s = Math.min(1, Math.max(0, -c / a)); }
  else if (t > 1) { t = 1; s = Math.min(1, Math.max(0, (b - c) / a)); }
  const d = norm(sub(add(p0, scale(d1, s)), add(q0, scale(d2, t))));
  return { s, t, d };
}

// ---------------------------------------------------------------- topology

export interface OracleMesh {
  x: V3[];
  tris: [number, number, number][];
  /** v0, v1, face0, face1 (NONE for a boundary edge). */
  edges: [number, number, number, number][];
}
export const NONE = 0xffffffff;

export function buildEdges(tris: [number, number, number][]): [number, number, number, number][] {
  const map = new Map<string, number>(), edges: [number, number, number, number][] = [];
  tris.forEach((t, f) => {
    for (let k = 0; k < 3; k++) {
      const a = t[k], b = t[(k + 1) % 3], key = a < b ? `${a},${b}` : `${b},${a}`;
      const e = map.get(key);
      if (e === undefined) { map.set(key, edges.length); edges.push([Math.min(a, b), Math.max(a, b), f, NONE]); }
      else edges[e][3] = f;
    }
  });
  return edges;
}

function stars(mesh: OracleMesh) {
  const vt: number[][] = mesh.x.map(() => []), ve: number[][] = mesh.x.map(() => []);
  mesh.tris.forEach((t, f) => t.forEach(v => vt[v].push(f)));
  mesh.edges.forEach((e, i) => { ve[e[0]].push(i); ve[e[1]].push(i); });
  return { vt, ve };
}

// ---------------------------------------------------------------- blocks (OGC eqs. 8-9)

/** x in the vertex block of k: (x - x_k).(x_k - x_k') >= 0 for every neighbour k'. */
export function vertexFeasible(mesh: OracleMesh, ve: number[][], x: V3, k: number): boolean {
  for (const e of ve[k]) {
    const o = mesh.edges[e][0] === k ? mesh.edges[e][1] : mesh.edges[e][0];
    if (dot(sub(x, mesh.x[k]), sub(mesh.x[k], mesh.x[o])) < 0) return false;
  }
  return true;
}

/** x in the edge block of edge e: strictly between the end planes and beyond each adjacent face. */
export function edgeFeasible(mesh: OracleMesh, x: V3, e: number): boolean {
  const [i, j, f0, f1] = mesh.edges[e], xi = mesh.x[i], xj = mesh.x[j];
  if (dot(sub(x, xi), sub(xj, xi)) <= 0 || dot(sub(x, xj), sub(xi, xj)) <= 0) return false;
  for (const f of [f0, f1]) {
    if (f === NONE) continue;
    const o = mesh.tris[f].find(v => v !== i && v !== j)!, xo = mesh.x[o];
    const dir = sub(xj, xi), s = dot(sub(xo, xi), dir) / dot(dir, dir), p = add(xi, scale(dir, s));
    if (dot(sub(x, p), sub(p, xo)) < 0) return false;
  }
  return true;
}

export function edgeIdOf(mesh: OracleMesh, a: number, b: number): number {
  const lo = Math.min(a, b), hi = Math.max(a, b);
  const e = mesh.edges.findIndex(q => q[0] === lo && q[1] === hi);
  if (e < 0) throw new Error(`no edge ${a}-${b}`);
  return e;
}

/** A global face id: "v:k", "e:id" or "f:t". */
export function globalFeature(mesh: OracleMesh, t: number, feature: number): string {
  const tri = mesh.tris[t];
  if (feature < 3) return `v:${tri[feature]}`;
  if (feature < 6) return `e:${edgeIdOf(mesh, tri[feature - 3], tri[(feature - 2) % 3])}`;
  return `f:${t}`;
}

// ---------------------------------------------------------------- brute-force detection

export interface OracleDetection {
  /** Vertex-facet pairs as "v|feature" strings. */
  vf: Set<string>;
  /** Edge-edge pairs as "e|e'" strings, both orders. */
  ee: Set<string>;
  dminV: number[];
  dminT: number[];
  dminE: number[];
  /** Conservative bounds b_v. */
  bound: number[];
  /** Pairs within `margin` of a classification threshold (excluded from strict set equality). */
  ambiguousVF: Set<string>;
  ambiguousEE: Set<string>;
}

/**
 * Algorithm 1 and 2 semantics by brute force. `rq[v]` is the vertex's query
 * radius; every d_min term is clipped at the radius of the vertex the bound is
 * for, so the result is independent of how the GPU search culls.
 */
export function bruteForceDetection(mesh: OracleMesh, r: number, rq: number[], gammaP: number, margin = 1e-6): OracleDetection {
  const { vt, ve } = stars(mesh), X = mesh.x;
  const vf = new Set<string>(), ee = new Set<string>(), ambiguousVF = new Set<string>(), ambiguousEE = new Set<string>();
  const dminV = X.map(() => Infinity), dminT = mesh.tris.map(() => Infinity), dminE = mesh.edges.map(() => Infinity);
  for (let v = 0; v < X.length; v++) {
    for (let t = 0; t < mesh.tris.length; t++) {
      const tri = mesh.tris[t];
      if (tri.includes(v)) continue;
      const a = X[tri[0]], b = X[tri[1]], c = X[tri[2]];
      const { bary, feature } = closestPointTriangle(X[v], a, b, c);
      const cp = add(add(scale(a, bary[0]), scale(b, bary[1])), scale(c, bary[2]));
      const d = norm(sub(X[v], cp));
      dminV[v] = Math.min(dminV[v], d); dminT[t] = Math.min(dminT[t], d);
      if (d >= r + margin) continue;
      const key = `${v}|${globalFeature(mesh, t, feature)}`;
      // Near a Voronoi or block boundary the f32 GPU may classify either way.
      const near = Math.abs(d - r) < margin || bary.some(q => q > 0 && q < margin) || bary.some(q => q < 1 && q > 1 - margin)
        || nearBlockBoundary(mesh, ve, X[v], t, feature, margin);
      if (d >= r) { ambiguousVF.add(key); continue; }
      let ok = true;
      if (feature < 3) ok = vertexFeasible(mesh, ve, X[v], tri[feature]);
      else if (feature < 6) ok = edgeFeasible(mesh, X[v], edgeIdOf(mesh, tri[feature - 3], tri[(feature - 2) % 3]));
      if (near) ambiguousVF.add(key);
      if (ok) vf.add(key);
    }
  }
  for (let e = 0; e < mesh.edges.length; e++) {
    const [a0, a1] = mesh.edges[e];
    for (let f = 0; f < mesh.edges.length; f++) {
      const [b0, b1] = mesh.edges[f];
      if (a0 === b0 || a0 === b1 || a1 === b0 || a1 === b1) continue;
      const lo = Math.min(e, f) === e;
      const q = lo ? closestSegmentSegment(X[a0], X[a1], X[b0], X[b1]) : swapST(closestSegmentSegment(X[b0], X[b1], X[a0], X[a1]));
      dminE[e] = Math.min(dminE[e], q.d);
      const interior = q.s > 0 && q.s < 1 && q.t > 0 && q.t < 1;
      const near = Math.abs(q.d - r) < margin || Math.min(q.s, 1 - q.s, q.t, 1 - q.t) < margin;
      if (near) ambiguousEE.add(`${e}|${f}`);
      if (q.d < r && interior) ee.add(`${e}|${f}`);
    }
  }
  const bound = X.map((_, v) => {
    let m = Math.min(rq[v], dminV[v]);
    for (const t of vt[v]) m = Math.min(m, dminT[t]);
    for (const e of ve[v]) m = Math.min(m, dminE[e]);
    return gammaP * m;
  });
  return { vf, ee, dminV, dminT, dminE, bound, ambiguousVF, ambiguousEE };
}

function swapST(q: { s: number; t: number; d: number }) { return { s: q.t, t: q.s, d: q.d }; }

function nearBlockBoundary(mesh: OracleMesh, ve: number[][], x: V3, t: number, feature: number, margin: number): boolean {
  const tri = mesh.tris[t];
  if (feature < 3) {
    const k = tri[feature];
    return ve[k].some(e => {
      const o = mesh.edges[e][0] === k ? mesh.edges[e][1] : mesh.edges[e][0];
      const dir = sub(mesh.x[k], mesh.x[o]);
      return Math.abs(dot(sub(x, mesh.x[k]), dir)) < margin * norm(dir);
    });
  }
  if (feature < 6) {
    const e = edgeIdOf(mesh, tri[feature - 3], tri[(feature - 2) % 3]);
    const [i, j, f0, f1] = mesh.edges[e], xi = mesh.x[i], xj = mesh.x[j], dir = sub(xj, xi);
    if (Math.abs(dot(sub(x, xi), dir)) < margin * norm(dir) || Math.abs(dot(sub(x, xj), dir)) < margin * norm(dir)) return true;
    for (const f of [f0, f1]) {
      if (f === NONE) continue;
      const o = mesh.tris[f].find(v => v !== i && v !== j)!, xo = mesh.x[o];
      const s = dot(sub(xo, xi), dir) / dot(dir, dir), p = add(xi, scale(dir, s)), n = sub(p, xo);
      if (Math.abs(dot(sub(x, p), n)) < margin * norm(n)) return true;
    }
  }
  return false;
}

// ---------------------------------------------------------------- energy (OGC eqs. 18-20, IPC friction)

export interface ContactParams { r: number; k: number; mu: number; epsV: number; dt: number }

/** The two-stage activation g(d) with tau = r/2, and its first two derivatives. */
export function activation(d: number, r: number, k: number): [number, number, number] {
  if (d >= r) return [0, 0, 0];
  const tau = r / 2;
  if (d >= tau) return [0.5 * k * (r - d) ** 2, -k * (r - d), k];
  const k2 = k * tau * (r - tau), b = 0.5 * k * (r - tau) ** 2 + k2 * Math.log(tau);
  return [-k2 * Math.log(d) + b, -k2 / d, k2 / (d * d)];
}

/** IPC's f0 (energy) and f1 = f0' for the static-dynamic friction transition at eps = epsV * dt. */
export function frictionF0(y: number, eps: number) { return y >= eps ? y : -y * y * y / (3 * eps * eps) + y * y / eps + eps / 3; }
export function frictionF1OverY(y: number, eps: number) { return y >= eps ? 1 / y : 2 / eps - y / (eps * eps); }

/**
 * One contact as the vertex block evaluates it: participants and their weights,
 * so the gap is sum_i w_i x_i. Vertex-facet weights come from the projection on
 * the feature's affine hull at the current positions; edge-edge weights are the
 * frozen detection parameters (s, t).
 */
export interface OracleContact {
  kind: "vf" | "ee";
  /** vf: [v', t0, t1, t2]; ee: [e0, e1, e'0, e'1]. */
  ids: [number, number, number, number];
  feature: number;
  s: number; t: number;
  /** Lagged normal force magnitude -g'(d) at detection. */
  lambda: number;
}

export function contactWeights(c: OracleContact, X: V3[]): [number, number, number, number] {
  if (c.kind === "ee") return [1 - c.s, c.s, -(1 - c.t), -c.t];
  const [v, a, b, cc] = c.ids, p = X[v], tri = [X[a], X[b], X[cc]];
  if (c.feature < 3) { const w: [number, number, number, number] = [1, 0, 0, 0]; w[1 + c.feature] = -1; return w; }
  if (c.feature < 6) {
    const i = c.feature - 3, j = (i + 1) % 3, dir = sub(tri[j], tri[i]);
    const s = dot(sub(p, tri[i]), dir) / dot(dir, dir);
    const w: [number, number, number, number] = [1, 0, 0, 0]; w[1 + i] = -(1 - s); w[1 + j] = -s; return w;
  }
  const v0 = sub(tri[1], tri[0]), v1 = sub(tri[2], tri[0]), v2 = sub(p, tri[0]);
  const d00 = dot(v0, v0), d01 = dot(v0, v1), d11 = dot(v1, v1), d20 = dot(v2, v0), d21 = dot(v2, v1);
  const den = d00 * d11 - d01 * d01, bv = (d11 * d20 - d01 * d21) / den, bw = (d00 * d21 - d01 * d20) / den;
  return [1, -(1 - bv - bw), -bv, -bw];
}

function gapOf(w: number[], ids: number[], X: V3[]): V3 {
  let g: V3 = [0, 0, 0];
  for (let i = 0; i < 4; i++) g = add(g, scale(X[ids[i]], w[i]));
  return g;
}

/**
 * Friction's lagged basis (IPC): the weights wl and unit normal nl of the pair at the step
 * start Xt (nl falls back to the current n when the start gap is 0), and the slip
 * u = (I - nl nl^T) sum_i wl_i (X_i - Xt_i), linear in X.
 */
function laggedSlip(c: OracleContact, X: V3[], Xt: V3[], n: V3): { wl: number[]; nl: V3; u: V3 } {
  const wl = contactWeights(c, Xt), gl = gapOf(wl, c.ids, Xt), l = norm(gl), nl = l > 1e-30 ? scale(gl, 1 / l) : n;
  let dx: V3 = [0, 0, 0];
  for (let i = 0; i < 4; i++) dx = add(dx, scale(sub(X[c.ids[i]], Xt[c.ids[i]]), wl[i]));
  return { wl, nl, u: sub(dx, scale(nl, dot(nl, dx))) };
}

/**
 * Pair energy: normal activation of the gap distance plus lagged friction
 * mu*lambda*f0(|u|), u the tangential slip since the step start in the
 * step start's basis (laggedSlip), so the block's force is its exact gradient.
 */
export function contactEnergy(c: OracleContact, X: V3[], Xt: V3[], p: ContactParams): number {
  const w = contactWeights(c, X), ids = c.ids;
  const g = gapOf(w, ids, X), d = norm(g), n = scale(g, 1 / d);
  const { u } = laggedSlip(c, X, Xt, n);
  return activation(d, p.r, p.k)[0] + p.mu * c.lambda * frictionF0(norm(u), p.epsV * p.dt);
}

/** Exact normal-energy gradient on participant `role`, and its exact Hessian block. */
export function normalGradientHessian(c: OracleContact, X: V3[], p: ContactParams, role: number): { grad: V3; hess: number[][] } {
  const w = contactWeights(c, X), g = gapOf(w, c.ids, X), d = norm(g), n = scale(g, 1 / d);
  const [, g1, g2] = activation(d, p.r, p.k), wr = w[role];
  const grad = scale(n, g1 * wr);
  const hess = [0, 1, 2].map(i => [0, 1, 2].map(j => wr * wr * (g2 * n[i] * n[j] + g1 * ((i === j ? 1 : 0) - n[i] * n[j]) / d)));
  return { grad, hess };
}

/**
 * What the vertex block accumulates for participant `role`: force (minus the
 * energy's gradient) and the PSD Hessian
 * g'' w^2 n n^T + mu lambda wl^2 (f1/|u|) (I - nl nl^T).
 */
export function vertexForceHessian(c: OracleContact, X: V3[], Xt: V3[], p: ContactParams, role: number): { f: V3; H: number[][] } {
  const w = contactWeights(c, X), ids = c.ids, g = gapOf(w, ids, X), d = norm(g), n = scale(g, 1 / d);
  const [, g1, g2] = activation(d, p.r, p.k), wr = w[role];
  const { wl, nl, u } = laggedSlip(c, X, Xt, n), wrl = wl[role];
  const fy = frictionF1OverY(norm(u), p.epsV * p.dt), ml = p.mu * c.lambda;
  const f = add(scale(n, -g1 * wr), scale(u, -ml * wrl * fy));
  const H = [0, 1, 2].map(i => [0, 1, 2].map(j => wr * wr * g2 * n[i] * n[j] + wrl * wrl * ml * fy * ((i === j ? 1 : 0) - nl[i] * nl[j])));
  return { f, H };
}

// ---------------------------------------------------------------- triangle-triangle intersection

function orient(a: V3, b: V3, c: V3, d: V3) { return dot(sub(b, a), cross(sub(c, a), sub(d, a))); }

/** Segment pq crosses (or touches) triangle abc. Coplanar configurations report false. */
export function segmentTriangle(p: V3, q: V3, a: V3, b: V3, c: V3): boolean {
  const o1 = orient(a, b, c, p), o2 = orient(a, b, c, q);
  if (o1 === 0 && o2 === 0) return false;
  if (o1 * o2 > 0) return false;
  const s1 = orient(p, q, a, b), s2 = orient(p, q, b, c), s3 = orient(p, q, c, a);
  return (s1 >= 0 && s2 >= 0 && s3 >= 0) || (s1 <= 0 && s2 <= 0 && s3 <= 0);
}

export function trianglesIntersect(A: V3[], B: V3[]): boolean {
  for (let k = 0; k < 3; k++) {
    if (segmentTriangle(A[k], A[(k + 1) % 3], B[0], B[1], B[2])) return true;
    if (segmentTriangle(B[k], B[(k + 1) % 3], A[0], A[1], A[2])) return true;
  }
  return false;
}

/** Brute-force count of intersecting pairs of faces that share no vertex. */
export function countIntersections(mesh: OracleMesh): number {
  let n = 0;
  for (let i = 0; i < mesh.tris.length; i++) for (let j = i + 1; j < mesh.tris.length; j++) {
    const a = mesh.tris[i], b = mesh.tris[j];
    if (a.some(v => b.includes(v))) continue;
    if (trianglesIntersect(a.map(v => mesh.x[v]), b.map(v => mesh.x[v]))) n++;
  }
  return n;
}

// ---------------------------------------------------------------- meshes for tests

/** A regular nx-by-ny sheet in the plane z = z0, alternating diagonals. */
export function sheet(nx: number, ny: number, spacing: number, origin: V3 = [0, 0, 0]): OracleMesh {
  const x: V3[] = [], tris: [number, number, number][] = [];
  for (let j = 0; j <= ny; j++) for (let i = 0; i <= nx; i++) x.push([origin[0] + i * spacing, origin[1] + j * spacing, origin[2]]);
  const id = (i: number, j: number) => j * (nx + 1) + i;
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    const a = id(i, j), b = id(i + 1, j), c = id(i + 1, j + 1), d = id(i, j + 1);
    if ((i + j) % 2) { tris.push([a, b, c], [a, c, d]); } else { tris.push([a, b, d], [b, c, d]); }
  }
  return { x, tris, edges: buildEdges(tris) };
}

/** Concatenate meshes into one (vertex and face ids offset). */
export function merge(...meshes: OracleMesh[]): OracleMesh {
  const x: V3[] = [], tris: [number, number, number][] = [];
  for (const m of meshes) { const o = x.length; x.push(...m.x.map(p => [...p] as V3)); tris.push(...m.tris.map(t => t.map(v => v + o) as [number, number, number])); }
  return { x, tris, edges: buildEdges(tris) };
}

/** A small deterministic generator for jitter. */
export function rng(seed: number) {
  let s = seed >>> 0;
  return () => { s = (s + 0x6d2b79f5) >>> 0; let t = s; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}

// ---------------------------------------------------------------- Divide and Truncate (Chen, Hsu, Ayman, Macklin 2026)

/** A division pair: side A (1 vertex, or an edge's 2) against side B (a face's 3, or an edge's 2). */
export interface DatPair { kind: "vt" | "ee"; ids: [number, number, number, number]; nA: number }

const faceEnvelope = (rq: number[], ids: number[]) => Math.max(...ids.map(i => rq[i]));

/** The DAT test's constants: the truncation ratio gamma_r and the separation sepEps (the GPU's P.gammaR, P.sepEps). */
export interface DatRule { gammaR: number; sepEps: number }

const absMax3 = (p: V3) => Math.max(Math.abs(p[0]), Math.abs(p[1]), Math.abs(p[2]));
/** The f32 error the GPU allows its distances: 128 ulps (2^-16) of the coordinates' scale. */
export const datRound = (s: number) => 2 ** -16 * s;
/** The levels of the path bound: K = ceil(5 L / thr) in [1, 32], so the slack L / 2K is at most a tenth of thr. */
export const datLevels = (L: number, thr: number) => Math.min(32, Math.max(1, Math.ceil(5 * L / Math.max(thr, 1e-30))));

const pointSegment = (p: V3, q0: V3, q1: V3) => {
  const d = sub(q1, q0), e = dot(d, d), t = e > 0 ? Math.min(1, Math.max(0, dot(sub(p, q0), d) / e)) : 0;
  return norm(sub(p, add(q0, scale(d, t))));
};
/** Exact primitive distances; a degenerate segment is its point, a zero-area triangle the union of its edges. */
export function pointTriangleDistance(p: V3, a: V3, b: V3, c: V3): number {
  const n = cross(sub(b, a), sub(c, a));
  if (!(dot(n, n) > 0)) return Math.min(pointSegment(p, a, b), pointSegment(p, b, c), pointSegment(p, c, a));
  const { bary } = closestPointTriangle(p, a, b, c);
  return norm(sub(p, add(add(scale(a, bary[0]), scale(b, bary[1])), scale(c, bary[2]))));
}
export function segmentSegmentDistance(p0: V3, p1: V3, q0: V3, q1: V3): number {
  if (!(dot(sub(p1, p0), sub(p1, p0)) > 0)) return pointSegment(p0, q0, q1);
  if (!(dot(sub(q1, q0), sub(q1, q0)) > 0)) return pointSegment(q0, p0, p1);
  return closestSegmentSegment(p0, p1, q0, q1).d;
}
const primitiveDistance = (kind: "vt" | "ee", P: V3[]) =>
  kind === "vt" ? pointTriangleDistance(P[0], P[1], P[2], P[3]) : segmentSegmentDistance(P[0], P[1], P[2], P[3]);

/**
 * The distance of the predicted primitives, every vertex at X + s w, sampled at K levels: on
 * [k/K, (k+1)/K] the L-Lipschitz g(s) is at least (g_k + g_k+1) / 2 - L / 2K. The minimum of that.
 */
export function datPathBound(kind: "vt" | "ee", P: V3[], W: V3[], L: number, K: number): number {
  let prev = primitiveDistance(kind, P), lo = Infinity;
  for (let k = 1; k <= K; k++) {
    const t = k / K, dk = primitiveDistance(kind, P.map((p, i) => add(p, scale(W[i], t))));
    lo = Math.min(lo, 0.5 * (prev + dk));
    prev = dk;
  }
  return lo - L / (2 * K);
}

/** One pair's DAT test (the GPU's datPairVT / datPairEE): its distance, threshold, spread, levels and path bound. */
export interface DatTest { d: number; thr: number; L: number; K: number; path: number; keep: boolean }

/**
 * Whether a pair needs a division plane. Every vertex moves along the round's one shared parameter
 * s and stays within its cap 0.5 gamma_r r_q of X + s w, so the pair stays more than sepEps apart
 * when its predicted primitives are, at every s, further apart than the two caps (each side's
 * widest) plus the strays: (i) at X, d > thr + L, as the relative motion closes the distance at most
 * L s; (ii) else the path bound exceeds thr. thr adds sepEps and datRound to the caps.
 */
export function datTest(kind: "vt" | "ee", ids: number[], X: V3[], W: V3[], rq: number[], rule: DatRule, strays = 0): DatTest {
  const nA = kind === "vt" ? 1 : 2, P = ids.map(i => X[i]), Wp = ids.map(i => W[i]);
  const ea = faceEnvelope(rq, ids.slice(0, nA)), eb = faceEnvelope(rq, ids.slice(nA));
  const L = predictionSpread(W, ids.slice(0, nA), ids.slice(nA));
  const scaleOf = Math.max(...P.map(absMax3)) + Math.max(...Wp.map(absMax3));
  const thr = 0.5 * rule.gammaR * (ea + eb) + rule.sepEps + strays + datRound(scaleOf);
  const d = primitiveDistance(kind, P), K = datLevels(L, thr);
  if (d > thr + L) return { d, thr, L, K, path: Infinity, keep: false };
  const path = datPathBound(kind, P, Wp, L, K);
  return { d, thr, L, K, path, keep: !(path > thr) };
}

/** Within margin of the test's decision: of (i), of (ii), or of a change in the levels. */
const datAmbiguous = (t: DatTest, margin: number) =>
  Math.abs(t.d - t.thr - t.L) < margin || (t.d <= t.thr + t.L + margin && (Math.abs(t.path - t.thr) < margin
    || datLevels(t.L, t.thr - margin) !== datLevels(t.L, t.thr + margin)));

/**
 * Every non-adjacent vertex-triangle and edge-edge pair the DAT test keeps with every vertex at
 * rest (w = 0): the distance is at most the two caps 0.5 gamma_r (r_q,a + r_q,b) plus sepEps and
 * the rounding (a face's or an edge's envelope is the widest of its vertices'). Keys "v|t" and
 * "e|f" with e < f; pairs within `margin` of the threshold are listed as ambiguous.
 */
export function bruteForceDatPairs(mesh: OracleMesh, rq: number[], margin: number, rule: DatRule) {
  return bruteForcePredictedPairs(mesh, rq, mesh.x.map(() => [0, 0, 0] as V3), margin, rule);
}

/** Support gap of a pair along unit n: min over side A minus max over side B. */
export function supportGap(n: V3, A: V3[], B: V3[]): number {
  return Math.min(...A.map(x => dot(n, x))) - Math.max(...B.map(x => dot(n, x)));
}

const unit = (a: V3): V3 => { const l = norm(a); return l > 1e-30 ? scale(a, 1 / l) : [0, 0, 0]; };

/** The pair's separating axis at X: the closest-point axis, else the better side of the face normal or edge cross product. */
export function datSeparator(p: DatPair, X: V3[], sepEps: number): { n: V3; gap: number } | null {
  const A = p.ids.slice(0, p.nA).map(i => X[i]), B = p.ids.slice(p.nA).map(i => X[i]);
  let n0: V3, alt: V3;
  if (p.kind === "vt") {
    const { bary } = closestPointTriangle(A[0], B[0], B[1], B[2]);
    n0 = unit(sub(A[0], add(add(scale(B[0], bary[0]), scale(B[1], bary[1])), scale(B[2], bary[2]))));
    alt = unit(cross(sub(B[1], B[0]), sub(B[2], B[0])));
  } else {
    const q = closestSegmentSegment(A[0], A[1], B[0], B[1]);
    n0 = unit(sub(add(A[0], scale(sub(A[1], A[0]), q.s)), add(B[0], scale(sub(B[1], B[0]), q.t))));
    alt = unit(cross(sub(A[1], A[0]), sub(B[1], B[0])));
  }
  let best = { n: n0, gap: supportGap(n0, A, B) };
  if (best.gap >= 2 * sepEps) return best;
  for (const n of [alt, scale(alt, -1)]) { const g = supportGap(n, A, B); if (g > best.gap) best = { n, gap: g }; }
  return best.gap > 0 ? best : null;
}

/** Eq. 16 with the endpoint rule: a wrong-side start may approach its side but not retreat further. */
export function datRatio(s0: number, s1: number, gammaR: number): number {
  if (s0 < 0) return s1 >= s0 ? 1 : 0;
  if (s1 >= 0) return 1;
  return Math.min(1, Math.max(0, gammaR * s0 / (s0 - s1)));
}

/**
 * Planar-DAT (Alg. 2): the truncated positions X + t_v dX_v, each then capped at
 * cap[v]. Planes use X; each side's share of the gap is proportional to its own
 * largest approach, at least 5% (and sepEps) per side. A pair with no separator holds its vertices.
 */
export function planarTruncate(X: V3[], Xnew: V3[], pairs: DatPair[], cap: number[], gammaR: number, sepEps: number): { x: V3[]; tv: number[] } {
  const tv = X.map(() => 1);
  const dX = Xnew.map((x, i) => sub(x, X[i]));
  for (const p of pairs) {
    const sep = datSeparator(p, X, sepEps);
    if (!sep) { for (const i of p.ids) tv[i] = 0; continue; }
    const { n, gap } = sep;
    const proj = p.ids.map(i => dot(n, X[i])), dn = p.ids.map(i => dot(n, dX[i]));
    let supA = Infinity, supB = -Infinity, apA = 0, apB = 0;
    p.ids.forEach((_, k) => { if (k < p.nA) { supA = Math.min(supA, proj[k]); apA = Math.max(apA, -dn[k]); } else { supB = Math.max(supB, proj[k]); apB = Math.max(apB, dn[k]); } });
    let lam = 0.5;
    if (gap >= 2 * sepEps) {
      if (apA + apB > 0) lam = apB / (apA + apB);
      const mf = Math.max(0.05, sepEps / gap);
      lam = Math.min(1 - mf, Math.max(mf, lam));
    }
    const plane = supB + lam * gap;
    p.ids.forEach((i, k) => {
      const a = k < p.nA, s0 = (a ? proj[k] - plane : plane - proj[k]) - sepEps;
      tv[i] = Math.min(tv[i], datRatio(s0, s0 + (a ? dn[k] : -dn[k]), gammaR));
    });
  }
  const x = X.map((x0, i) => {
    let d = scale(dX[i], tv[i]);
    const l = norm(d);
    if (l > cap[i]) d = scale(d, cap[i] / l);
    return add(x0, d);
  });
  return { x, tv };
}

// ------------------------------------------------------------------ predicted-motion DAT (option c)

/**
 * Predicted-motion DAT. Each vertex carries a prediction w_v for the round (the
 * inertial v dt + g dt^2, or 0 for a held vertex) and moves X + w_v + delta_v
 * with |delta_v| <= cap_v. Division planes keep their normal and translate:
 * placed at X between the two sides at the start, and between the predicted
 * positions X + w at the end, so a vertex on its side at both ends of its
 * straight path stays on it (both are linear in time).
 */
const sweptBox = (P: V3[], W: V3[], e: number) => {
  const lo: V3 = [Infinity, Infinity, Infinity], hi: V3 = [-Infinity, -Infinity, -Infinity];
  P.forEach((p, i) => { for (const q of [p, add(p, W[i])]) for (let k = 0; k < 3; k++) { lo[k] = Math.min(lo[k], q[k] - e); hi[k] = Math.max(hi[k], q[k] + e); } });
  return { lo, hi };
};
const boxesMeet = (a: { lo: V3; hi: V3 }, b: { lo: V3; hi: V3 }) => [0, 1, 2].every(k => a.lo[k] <= b.hi[k] && b.lo[k] <= a.hi[k]);
/** The largest difference of two primitives' predictions, vertex by vertex. */
export const predictionSpread = (W: V3[], a: number[], b: number[]) => Math.max(...a.flatMap(i => b.map(j => norm(sub(W[i], W[j])))));

/**
 * Every non-adjacent pair whose swept boxes (X to X + w), each grown by its own envelope (a
 * vertex's r_q, an edge's or a face's its vertices' widest; an edge is reached through its first
 * face), meet, and that the DAT test (datTest) keeps. Any other pair's predicted primitives stay
 * further apart than its two caps at every level of the round's shared parameter, which the
 * caps (0.5 gamma_r r_q each) cannot close.
 */
export function bruteForcePredictedPairs(mesh: OracleMesh, rq: number[], W: V3[], margin: number, rule: DatRule) {
  const X = mesh.x, vt = new Set<string>(), ee = new Set<string>(), ambiguous = new Set<string>();
  const pairs: DatPair[] = [];
  for (let v = 0; v < X.length; v++) for (let t = 0; t < mesh.tris.length; t++) {
    const tri = mesh.tris[t];
    if (tri.includes(v)) continue;
    if (!boxesMeet(sweptBox([X[v]], [W[v]], rq[v]), sweptBox(tri.map(i => X[i]), tri.map(i => W[i]), faceEnvelope(rq, tri)))) continue;
    const test = datTest("vt", [v, ...tri], X, W, rq, rule);
    if (datAmbiguous(test, margin)) ambiguous.add(`${v}|${t}`);
    if (test.keep) { vt.add(`${v}|${t}`); pairs.push({ kind: "vt", ids: [v, tri[0], tri[1], tri[2]], nA: 1 }); }
  }
  for (let e = 0; e < mesh.edges.length; e++) for (let f = e + 1; f < mesh.edges.length; f++) {
    const [a0, a1] = mesh.edges[e], [b0, b1] = mesh.edges[f];
    if (a0 === b0 || a0 === b1 || a1 === b0 || a1 === b1) continue;
    const ea = Math.max(rq[a0], rq[a1]), face = mesh.tris[mesh.edges[f][2]];
    if (!boxesMeet(sweptBox([X[a0], X[a1]], [W[a0], W[a1]], ea), sweptBox(face.map(i => X[i]), face.map(i => W[i]), faceEnvelope(rq, face)))) continue;
    const test = datTest("ee", [a0, a1, b0, b1], X, W, rq, rule);
    if (datAmbiguous(test, margin)) ambiguous.add(`${e}|${f}`);
    if (test.keep) { ee.add(`${e}|${f}`); pairs.push({ kind: "ee", ids: [a0, a1, b0, b1], nA: 2 }); }
  }
  return { vt, ee, ambiguous, pairs };
}

/**
 * A pair the round cannot divide: no separating plane at X while some vertex
 * is predicted to move, or predicted positions X + w that cross (or close to
 * within sepEps, half the gap the truncation keeps) along the pair's plane normal. A round with a conflict is
 * redone with every prediction held at 0, which is Planar-DAT itself.
 */
export function predictedConflict(p: DatPair, X: V3[], W: V3[], sepEps: number): boolean {
  if (p.ids.every(i => norm(W[i]) === 0)) return false;
  const sep = datSeparator(p, X, sepEps);
  if (!sep) return true;
  const Y = p.ids.map(i => add(X[i], W[i]));
  const g1 = supportGap(sep.n, Y.slice(0, p.nA), Y.slice(p.nA));
  return g1 < sepEps && g1 < sep.gap;
}

/**
 * Predicted-motion Planar-DAT: each pair's plane is placed among the predicted
 * positions X + w, each side's share of the predicted gap proportional to its
 * own largest residual approach; each vertex's residual delta = x - X - w is
 * scaled by its smallest ratio, then capped at cap[v]. With W = 0 this is
 * planarTruncate.
 */
export function predictedTruncate(X: V3[], Xnew: V3[], W: V3[], pairs: DatPair[], cap: number[], gammaR: number, sepEps: number): { x: V3[]; tv: number[] } {
  const tv = X.map(() => 1);
  const res = Xnew.map((x, i) => sub(sub(x, X[i]), W[i]));
  for (const p of pairs) {
    const sep = datSeparator(p, X, sepEps);
    if (!sep) { for (const i of p.ids) tv[i] = 0; continue; }
    const { n } = sep;
    const proj = p.ids.map(i => dot(n, add(X[i], W[i]))), dn = p.ids.map(i => dot(n, res[i]));
    let supA = Infinity, supB = -Infinity, apA = 0, apB = 0;
    p.ids.forEach((_, k) => { if (k < p.nA) { supA = Math.min(supA, proj[k]); apA = Math.max(apA, -dn[k]); } else { supB = Math.max(supB, proj[k]); apB = Math.max(apB, dn[k]); } });
    const gap = supA - supB;
    let lam = 0.5;
    if (gap >= 2 * sepEps) {
      if (apA + apB > 0) lam = apB / (apA + apB);
      const mf = Math.max(0.05, sepEps / gap);
      lam = Math.min(1 - mf, Math.max(mf, lam));
    }
    const plane = supB + lam * gap;
    p.ids.forEach((i, k) => {
      const a = k < p.nA, s0 = (a ? proj[k] - plane : plane - proj[k]) - sepEps;
      tv[i] = Math.min(tv[i], datRatio(s0, s0 + (a ? dn[k] : -dn[k]), gammaR));
    });
  }
  const x = X.map((x0, i) => {
    let d = scale(res[i], tv[i]);
    const l = norm(d);
    if (l > cap[i]) d = scale(d, cap[i] / l);
    return add(add(x0, W[i]), d);
  });
  return { x, tv };
}

/** Isotropic DAT (OGC): |dX_v| <= b_v. */
export function isotropicTruncate(X: V3[], Xnew: V3[], bound: number[]): V3[] {
  return X.map((x0, i) => { let d = sub(Xnew[i], x0); const l = norm(d); if (l > bound[i]) d = scale(d, bound[i] / l); return add(x0, d); });
}

/**
 * Continuous check of the straight-line motion X0 -> X1 by brute force: some
 * non-incident vertex crosses a face or two non-adjacent edges cross. Roots of
 * the coplanarity cubic are bracketed on a fine grid and bisected in Float64.
 */
export function motionPenetrates(mesh: OracleMesh, X0: V3[], X1: V3[], samples = 64): boolean {
  const at = (i: number, s: number): V3 => add(X0[i], scale(sub(X1[i], X0[i]), s));
  const cop = (ids: number[], s: number) => { const [p, a, b, c] = ids.map(i => at(i, s)); return orient(a, b, c, p); };
  const crossesAt = (ids: number[], s: number, vt: boolean) => {
    const P = ids.map(i => at(i, s));
    if (vt) {
      const { bary } = closestPointTriangle(P[0], P[1], P[2], P[3]);
      const cp = add(add(scale(P[1], bary[0]), scale(P[2], bary[1])), scale(P[3], bary[2]));
      return norm(sub(P[0], cp)) < 1e-12;
    }
    return closestSegmentSegment(P[0], P[1], P[2], P[3]).d < 1e-12;
  };
  const test = (ids: number[], vt: boolean) => {
    // ee coplanarity: orient(p0, p1, q0, q1) shares the cubic form with vt (p, a, b, c).
    const order = vt ? ids : [ids[2], ids[0], ids[1], ids[3]];
    let s0 = 0, f0 = cop(order, 0);
    for (let k = 1; k <= samples; k++) {
      const s1 = k / samples, f1 = cop(order, s1);
      if (f0 === 0 && crossesAt(ids, s0, vt)) return true;
      if (f0 * f1 < 0) {
        let lo = s0, hi = s1, flo = f0;
        for (let j = 0; j < 60; j++) { const m = 0.5 * (lo + hi), fm = cop(order, m); if (fm * flo <= 0) hi = m; else { lo = m; flo = fm; } }
        if (crossesAt(ids, 0.5 * (lo + hi), vt)) return true;
      }
      s0 = s1; f0 = f1;
    }
    return false;
  };
  for (let v = 0; v < X0.length; v++) for (const tri of mesh.tris) if (!tri.includes(v) && test([v, ...tri], true)) return true;
  for (let e = 0; e < mesh.edges.length; e++) for (let f = e + 1; f < mesh.edges.length; f++) {
    const [a0, a1] = mesh.edges[e], [b0, b1] = mesh.edges[f];
    if (a0 === b0 || a0 === b1 || a1 === b0 || a1 === b1) continue;
    if (test([a0, a1, b0, b1], false)) return true;
  }
  return false;
}

// ------------------------------------------------------------------ static solid field (Float64)

import {
  SOLID_BRICK_CELLS, cellGradientRange, sampleClothSolidField, type ClothSolidField,
} from "../lib/cloth/contact/solid-field";

/** Exact signed distances of the test solids. */
export const boxSdf = (lo: V3, hi: V3) => {
  const cx = (lo[0] + hi[0]) / 2, cy = (lo[1] + hi[1]) / 2, cz = (lo[2] + hi[2]) / 2, hx = (hi[0] - lo[0]) / 2, hy = (hi[1] - lo[1]) / 2, hz = (hi[2] - lo[2]) / 2;
  return (p: readonly number[]) => {
    const qx = Math.abs(p[0] - cx) - hx, qy = Math.abs(p[1] - cy) - hy, qz = Math.abs(p[2] - cz) - hz;
    const ox = Math.max(qx, 0), oy = Math.max(qy, 0), oz = Math.max(qz, 0);
    return Math.sqrt(ox * ox + oy * oy + oz * oz) + Math.min(Math.max(qx, qy, qz), 0);
  };
};
export const sphereSdf = (c: V3, R: number) => (p: readonly number[]) => Math.hypot(p[0] - c[0], p[1] - c[1], p[2] - c[2]) - R;

/** phi(x0 + d) >= phi0 + g0.d - sum_a D_a |d_a| over the ball (D: per-axis gradient deviation). */
export interface OracleSolidBound { far: boolean; phi0: number; g0: V3; D: V3; stored: boolean }

/** The bound's drop D.|d|. */
export function solidBoundDrop(b: OracleSolidBound, d: V3): number {
  return b.D[0] * Math.abs(d[0]) + b.D[1] * Math.abs(d[1]) + b.D[2] * Math.abs(d[2]);
}

/** The GPU's ball bound (ogcSolidBallBound) in Float64 on the field's node values. */
export function solidBallBound(field: ClothSolidField, x0: V3, rho: number, r: number): OracleSolidBound {
  const s = sampleClothSolidField(field, x0);
  const out: OracleSolidBound = { far: true, phi0: s.phi, g0: [...s.grad] as V3, D: [0, 0, 0], stored: s.stored };
  if (Math.min(s.phi, field.band) - field.maxLipschitz * rho >= r) return out;
  const cellOf = (p: V3) => p.map((v, a) => Math.floor((v - field.origin[a]) / field.cell));
  const c0 = cellOf(sub(x0, [rho, rho, rho])), c1 = cellOf(add(x0, [rho, rho, rho]));
  const D: V3 = [0, 0, 0];
  for (let cz = c0[2]; cz <= c1[2]; cz++) for (let cy = c0[1]; cy <= c1[1]; cy++) for (let cx = c0[0]; cx <= c1[0]; cx++) {
    const c = [cx, cy, cz], b = c.map(v => Math.floor(v / SOLID_BRICK_CELLS));
    // Outside the lattice: a missing brick when the exterior is clear (a body's padded box),
    // else no information (the container shell lies there).
    if (b.some((v, a) => v < 0 || v >= field.dims[a])) { if (field.exteriorClear) return out; continue; }
    const brick = field.bricks.get(b.join(","));
    if (!brick) return out;
    const { lo, hi } = cellGradientRange(brick.values, c.map((v, a) => v - SOLID_BRICK_CELLS * b[a]) as V3);
    for (let a = 0; a < 3; a++) D[a] = Math.max(D[a], Math.abs(hi[a] / field.cell - s.grad[a]), Math.abs(s.grad[a] - lo[a] / field.cell));
  }
  return { ...out, far: false, D };
}

/** The bound's normalisation: f(d) = (D.|d| - g0.d) s <= 1 keeps phi >= min((1 - gamma_r) phi0, r). */
export function solidTermScale(phi0: number, gammaR: number, r: number, sepEps = 0): number {
  return 1 / Math.max(phi0 - solidKeep(phi0, gammaR, r, sepEps), 1e-30);
}
/** The clearance a cut keeps: min((1 - gamma_r) phi0, r) and at least sepEps (f32 resolution); within sepEps, phi0 (no approach). */
export function solidKeep(phi0: number, gammaR: number, r: number, sepEps = 0): number {
  return phi0 > sepEps ? Math.max(Math.min((1 - gammaR) * phi0, r), sepEps) : phi0;
}

/**
 * Apply's solid factor on the displacement d from x_det: the largest scale either certificate
 * keeps, the affine 1 / (D.|d| s - q.d) or the Lipschitz ball |d| <= rl = (1 / s) / L_max
 * (lmax absent: the affine bound alone).
 */
export function solidTruncate(b: OracleSolidBound, d: V3, gammaR: number, r: number, lmax = Infinity): V3 {
  if (b.far) return d;
  if (!(b.phi0 > 0)) return [0, 0, 0];
  const s = solidTermScale(b.phi0, gammaR, r), den = s * solidBoundDrop(b, d) - s * dot(b.g0, d);
  if (den <= 1) return d;
  const l = norm(d), rl = 1 / (s * lmax);
  return l <= rl ? d : scale(d, Math.max(1 / den, rl / l));
}

/** The solid entry's energy, force and Gauss-Newton Hessian at x (ogcSolidLin); the entry carries mu lambda (the solid's friction times the lagged normal force). */
export function solidContact(field: ClothSolidField, x: V3, xt: V3, lambda: number, p: ContactParams): { e: number; f: V3; H: number[][] } {
  const s = sampleClothSolidField(field, x);
  if (!(s.phi > 0)) return { e: Infinity, f: [0, 0, 0], H: [[0, 0, 0], [0, 0, 0], [0, 0, 0]] };
  const [g0, g1, g2] = activation(s.phi, p.r, p.k), g = s.grad as V3;
  let e = g0, f = scale(g, -g1);
  const H = [0, 1, 2].map(i => [0, 1, 2].map(k => g2 * g[i] * g[k]));
  if (lambda > 0) {
    // Friction's basis lagged at the step start (the field's normal at xt).
    const unit = (v: V3): V3 | undefined => { const l = norm(v); return l > 1e-30 ? scale(v, 1 / l) : undefined; };
    const n: V3 = unit(sampleClothSolidField(field, xt).grad as V3) ?? unit(g) ?? [0, 0, 0];
    const dx = sub(x, xt), u = sub(dx, scale(n, dot(n, dx))), un = norm(u), eps = p.epsV * p.dt;
    const ml = lambda, fy = frictionF1OverY(un, eps);
    f = sub(f, scale(u, ml * fy));
    for (let i = 0; i < 3; i++) for (let k = 0; k < 3; k++) H[i][k] += ml * fy * ((i === k ? 1 : 0) - n[i] * n[k]);
    e += ml * frictionF0(un, eps);
  }
  return { e, f, H };
}

/**
 * Whether the triangle (a, b, c) reaches into the solid, phi < 0, decided
 * exactly on the field's trilinear interpolant (its f32 node values, in double):
 * the triangle is clipped into the cells it crosses; on each piece phi is a
 * cubic in the barycentrics, whose Bernstein coefficients (the blossom of the
 * cell's multi-affine interpolant) bound it. A piece with every coefficient
 * >= -1e-9 h is outside, one with a corner below it inside; any other is split in four,
 * down to `depth` levels, past which the call throws (it never guesses).
 * `lower` is a lower bound on phi over the triangle (the least coefficient
 * where outside).
 */
export function faceInsideSolid(field: ClothSolidField, a: V3, b: V3, c: V3, depth = 16): { inside: boolean; lower: number } {
  const h = field.cell, B = SOLID_BRICK_CELLS, NN = B + 1;
  // Depths under 1e-9 h are rounding (a piece on a node plane where phi = 0): taken as on the surface.
  const eps = 1e-9 * h;
  const toCell = (p: V3): V3 => [(p[0] - field.origin[0]) / h, (p[1] - field.origin[1]) / h, (p[2] - field.origin[2]) / h];
  const P = [toCell(a), toCell(b), toCell(c)];
  // Lipschitz prefilter: phi(p) >= phi(v) - L |p - v| for the triangle's vertices v.
  const edge = Math.max(norm(sub(a, b)), norm(sub(b, c)), norm(sub(c, a)));
  const phiV = [a, b, c].map(p => sampleClothSolidField(field, p as unknown as readonly [number, number, number]).phi);
  if (Math.min(...phiV) - field.maxLipschitz * edge > 0) return { inside: false, lower: Math.min(...phiV) - field.maxLipschitz * edge };
  const lo = [0, 1, 2].map(k => Math.floor(Math.min(P[0]![k]!, P[1]![k]!, P[2]![k]!))), hi = [0, 1, 2].map(k => Math.floor(Math.max(P[0]![k]!, P[1]![k]!, P[2]![k]!)));
  let lower = Infinity;
  for (let z = lo[2]!; z <= hi[2]!; z++) for (let y = lo[1]!; y <= hi[1]!; y++) for (let x = lo[0]!; x <= hi[0]!; x++) {
    const cell = [x, y, z];
    // Clip the triangle to the cell [cell, cell + 1]^3.
    let poly: V3[] = P.map(p => [...p] as V3);
    for (let k = 0; k < 3 && poly.length; k++) for (const [bound, sgn] of [[cell[k]!, 1], [cell[k]! + 1, -1]] as const) {
      const next: V3[] = [];
      for (let i = 0; i < poly.length; i++) {
        const p = poly[i]!, q = poly[(i + 1) % poly.length]!, dp = sgn * (p[k]! - bound), dq = sgn * (q[k]! - bound);
        if (dp >= 0) next.push(p);
        if (dp >= 0 !== dq >= 0) { const t = dp / (dp - dq); next.push([p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1]), p[2] + t * (q[2] - p[2])]); }
      }
      poly = next;
    }
    if (poly.length < 3) continue;
    const bk = cell.map(v => Math.floor(v / B)), brick = field.bricks.get(bk.join(","));
    if (!brick) { lower = Math.min(lower, field.band); continue; }
    const l = cell.map((v, k) => v - bk[k]! * B), vals = Array.from({ length: 8 }, (_, k) =>
      brick.values[(l[0]! + (k & 1)) + NN * ((l[1]! + ((k >> 1) & 1)) + NN * (l[2]! + (k >> 2)))]!);
    // The cell's interpolant at mixed coordinates: x from U, y from V, z from W (cell-local).
    const T = (fx: number, fy: number, fz: number) => {
      let s = 0;
      for (let k = 0; k < 8; k++) s += vals[k]! * ((k & 1) ? fx : 1 - fx) * (((k >> 1) & 1) ? fy : 1 - fy) * ((k >> 2) ? fz : 1 - fz);
      return s;
    };
    const blossom = (U: V3, V: V3, W: V3) => {
      const f = (p: V3, k: number) => p[k]! - cell[k]!;
      return (T(f(U, 0), f(V, 1), f(W, 2)) + T(f(U, 0), f(W, 1), f(V, 2)) + T(f(V, 0), f(U, 1), f(W, 2))
        + T(f(V, 0), f(W, 1), f(U, 2)) + T(f(W, 0), f(U, 1), f(V, 2)) + T(f(W, 0), f(V, 1), f(U, 2))) / 6;
    };
    const mid = (p: V3, q: V3): V3 => [(p[0] + q[0]) / 2, (p[1] + q[1]) / 2, (p[2] + q[2]) / 2];
    const piece = (A: V3, Bp: V3, C: V3, d: number): boolean => {
      const coef = [blossom(A, A, A), blossom(Bp, Bp, Bp), blossom(C, C, C), blossom(A, A, Bp), blossom(A, A, C), blossom(A, Bp, Bp),
        blossom(Bp, Bp, C), blossom(A, C, C), blossom(Bp, C, C), blossom(A, Bp, C)];
      if (coef[0]! < -eps || coef[1]! < -eps || coef[2]! < -eps) return true;
      const m = Math.min(...coef);
      if (m >= -eps) { lower = Math.min(lower, m); return false; }
      if (d >= depth) throw new Error(`faceInsideSolid: undecided at depth ${depth} (coefficients down to ${m.toExponential(2)})`);
      const ab = mid(A, Bp), bc = mid(Bp, C), ca = mid(C, A);
      return piece(A, ab, ca, d + 1) || piece(ab, Bp, bc, d + 1) || piece(ca, bc, C, d + 1) || piece(ab, bc, ca, d + 1);
    };
    for (let i = 1; i + 1 < poly.length; i++) if (piece(poly[0]!, poly[i]!, poly[i + 1]!, 0)) return { inside: true, lower: -Infinity };
  }
  return { inside: false, lower };
}

/** faceInsideSolid over a mesh at positions x (field frame after subtracting `origin`): the faces inside and the least lower bound. */
export function facesInsideSolid(field: ClothSolidField, x: ArrayLike<number>, faces: ArrayLike<number>, origin: readonly number[] = [0, 0, 0]): { inside: number[]; lower: number } {
  const P = (v: number): V3 => [x[3 * v]! - origin[0]!, x[3 * v + 1]! - origin[1]!, x[3 * v + 2]! - origin[2]!];
  const inside: number[] = [];
  let lower = Infinity;
  for (let t = 0; t < faces.length / 3; t++) {
    const r = faceInsideSolid(field, P(faces[3 * t]!), P(faces[3 * t + 1]!), P(faces[3 * t + 2]!));
    if (r.inside) inside.push(t); else lower = Math.min(lower, r.lower);
  }
  return { inside, lower };
}

// Levelled re-detection (C2 item (5)): the certified family S_v(a), a in [0, 1], and the line search over it.
export interface LevelledVertex { X: V3; w: V3; R: number; planes: { m: V3; b: number }[] }
export interface LevelledRound { vertices: LevelledVertex[]; conflict: boolean; records: number }

export function levelledRound(mesh: OracleMesh, W: V3[], pairs: DatPair[], cap: number[], gammaR: number, sepEps: number): LevelledRound {
  const X = mesh.x;
  const vertices: LevelledVertex[] = X.map((x, i) => ({ X: x, w: W[i], R: cap[i], planes: [] }));
  let conflict = false, records = 0;
  for (const p of pairs) {
    const sep = datSeparator(p, X, sepEps);
    const Y = p.ids.map(i => add(X[i], W[i]));
    const moving = p.ids.some(i => norm(W[i]) !== 0);
    let n: V3 | null = sep ? sep.n : null;
    if (moving) {
      const g1 = n ? supportGap(n, Y.slice(0, p.nA), Y.slice(p.nA)) : -Infinity;
      if (!n || (g1 < sepEps && g1 < sep!.gap)) {
        // The predicted primitives' own closest-point axis, when it divides both ends.
        let n1: V3;
        if (p.kind === "vt") {
          const { bary } = closestPointTriangle(Y[0], Y[1], Y[2], Y[3]);
          n1 = unit(sub(Y[0], add(add(scale(Y[1], bary[0]), scale(Y[2], bary[1])), scale(Y[3], bary[2]))));
        } else {
          const q = closestSegmentSegment(Y[0], Y[1], Y[2], Y[3]);
          n1 = unit(sub(add(Y[0], scale(sub(Y[1], Y[0]), q.s)), add(Y[2], scale(sub(Y[3], Y[2]), q.t))));
        }
        const A0 = p.ids.slice(0, p.nA).map(i => X[i]), B0 = p.ids.slice(p.nA).map(i => X[i]);
        const h0 = supportGap(n1, A0, B0), h1 = supportGap(n1, Y.slice(0, p.nA), Y.slice(p.nA));
        if (norm(n1) > 0 && h0 > 0 && (h1 >= sepEps || h1 >= h0)) n = n1;
        else { conflict = true; continue; }
      }
    }
    if (!n) { conflict = true; continue; }
    const nn = n;
    const ends = [0, 1].map(level => {
      const P = p.ids.map((i, k) => (level === 0 ? X[i] : Y[k]));
      const proj = P.map(q => dot(nn, q));
      const supA = Math.min(...proj.slice(0, p.nA)), supB = Math.max(...proj.slice(p.nA));
      const plane = supB + 0.5 * (supA - supB);
      return proj.map((q, k) => -gammaR * Math.max((k < p.nA ? q - plane : plane - q) - sepEps, 0));
    });
    p.ids.forEach((i, k) => {
      const b = Math.max(ends[0][k], ends[1][k]);
      // A half-space the cap keeps whole at every level is redundant.
      if (-b >= cap[i]) return;
      vertices[i].planes.push({ m: k < p.nA ? nn : scale(nn, -1), b });
      records++;
    });
  }
  return { vertices, conflict, records };
}

/** A world half-space m.y >= h (a static solid's convex certificate, holding every anchor). */
export interface WorldHalfSpace { m: V3; h: number }

const ballExit = (z: V3, e: V3, R: number) => {
  // Largest t <= 1 with |z + t e| <= R, for |z| <= R.
  const ee = dot(e, e);
  if (ee === 0) return 1;
  const ze = dot(z, e), zz = dot(z, z);
  const t = (-ze + Math.sqrt(Math.max(ze * ze - ee * (zz - R * R), 0))) / ee;
  return Math.min(1, Math.max(0, t));
};

/** Whether y lies in S_v(a) (and the solids), to within tol. */
export function levelledInside(V: LevelledVertex, solids: WorldHalfSpace[], y: V3, a: number, tol = 0): boolean {
  const c = add(V.X, scale(V.w, a)), z = sub(y, c);
  if (norm(z) > V.R + tol) return false;
  if (V.planes.some(r => dot(r.m, z) < r.b - tol)) return false;
  return solids.every(s => dot(s.m, y) >= s.h - tol);
}

/**
 * A certificate of the exact projection, relative to the anchor: the ball |z - c| <= R, or the
 * half-space n.z <= beta.
 */
export type ProjCon = { ball: true; c: V3; R: number } | { ball: false; n: V3; beta: number };
/** The distance by which z violates k (<= 0 inside); a half-space with no normal holds everywhere. */
export const conViolation = (k: ProjCon, z: V3) => (k.ball ? norm(sub(z, k.c)) - k.R : norm(k.n) === 0 ? -Infinity : (dot(k.n, z) - k.beta) / norm(k.n));

/**
 * The nearest point to y of the intersection of the boundaries of cs (|cs| <= 3), or null when
 * it is empty or degenerate: the first ball is a sphere, each other ball its radical plane with
 * it, and the planes' affine set A meets the sphere in a sphere of A about proj_A(c).
 */
export function nearestOnBoundaries(cs: ProjCon[], y: V3): V3 | null {
  const sph = cs.find(k => k.ball) as { ball: true; c: V3; R: number } | undefined;
  const e: V3[] = [], g: number[] = [];
  for (const k of cs) {
    if (k === sph) continue;
    let n: V3, b: number;
    if (k.ball) { const d = sub(k.c, sph!.c); n = scale(d, 2); b = dot(d, d) + 2 * dot(d, sph!.c) + (sph!.R - k.R) * (sph!.R + k.R); }
    else { n = k.n; b = k.beta; }
    const l0 = norm(n);
    for (let l = 0; l < e.length; l++) { const t = dot(n, e[l]); n = sub(n, scale(e[l], t)); b -= t * g[l]; }
    const L = norm(n);
    if (!(L > 1e-12 * l0)) return null;
    e.push(scale(n, 1 / L)); g.push(b / L);
  }
  const proj = (w: V3) => e.reduce((p, el, l) => sub(p, scale(el, dot(el, p) - g[l])), w);
  const py = proj(y);
  if (!sph) return py;
  const pc = proj(sph.c), r2 = sph.R * sph.R - dot(sub(sph.c, pc), sub(sph.c, pc)), t = sub(py, pc), tl = norm(t);
  if (r2 < 0 || !(tl > 0)) return null;
  return add(pc, scale(t, Math.sqrt(r2) / tl));
}

/**
 * The Euclidean projection of y onto the intersection of cons, by brute force: the nearest of
 * the feasible points nearest(M_W) over every W with |W| <= 3 (Caratheodory: the projection is
 * one of them, and every feasible candidate is in the set). tol: the feasibility slack.
 */
export function bruteForceProjection(cons: ProjCon[], y: V3, tol: number): { p: V3; W: number[] } | null {
  if (cons.every(k => conViolation(k, y) <= tol)) return { p: y, W: [] };
  let best: { p: V3; W: number[]; d: number } | null = null;
  const m = cons.length;
  const tryW = (W: number[]) => {
    const p = nearestOnBoundaries(W.map(i => cons[i]), y);
    if (!p || cons.some(k => conViolation(k, p) > tol)) return;
    const d = norm(sub(p, y));
    if (!best || d < best.d) best = { p, W, d };
  };
  for (let i = 0; i < m; i++) {
    tryW([i]);
    for (let j = i + 1; j < m; j++) { tryW([i, j]); for (let k = j + 1; k < m; k++) tryW([i, j, k]); }
  }
  return best;
}

/**
 * The GPU's algorithm (ogcProjectExact) in Float64: constraint generation from y, the most
 * violated certificate joining the working set S, which becomes the active set of the nearest
 * feasible candidate over S (|S| <= 4). Returns the point and the rounds it took (the GPU
 * bounds them at OGC_PROJECT_ROUNDS = 32).
 */
export function workingSetProjection(cons: ProjCon[], y: V3, tol: number): { p: V3; rounds: number } {
  let z = y, S: number[] = [], rounds = 0;
  for (; rounds < 64; rounds++) {
    let j = -1, jv = tol;
    cons.forEach((k, i) => { const v = conViolation(k, z); if (v > jv) { j = i; jv = v; } });
    if (j < 0 || S.includes(j)) break;
    S = [...S, j].sort((a, b) => a - b);
    let best: { p: V3; W: number[]; d: number; v: number } | null = null;
    for (let mask = 1; mask < 1 << S.length; mask++) {
      const W = S.filter((_, i) => mask & (1 << i));
      if (W.length > 3) continue;
      const q = nearestOnBoundaries(W.map(i => cons[i]), y);
      if (!q) continue;
      const v = Math.max(...S.map(i => conViolation(cons[i], q))), d = norm(sub(q, y)), ok = v <= tol;
      if (!best || (ok && (best.v > tol || d < best.d)) || (!ok && best.v > tol && v < best.v)) best = { p: q, W, d, v };
    }
    if (!best) break;
    z = best.p; S = best.W;
  }
  return { p: z, rounds };
}

/**
 * The KKT conditions of p as the projection of y onto cons: its largest violation, and the
 * residual of y - p in the cone of the normals of the certificates active at p (within
 * activeTol), over |y - p|: the least-squares fit over every subset of at most three active
 * normals with nonnegative multipliers (0 when p = y).
 */
export function projectionKkt(cons: ProjCon[], y: V3, p: V3, activeTol: number): { violation: number; residual: number; active: number } {
  const violation = Math.max(...cons.map(k => conViolation(k, p)));
  const r = sub(y, p), rn = norm(r);
  const normals: V3[] = [];
  for (const k of cons) {
    if (conViolation(k, p) < -activeTol) continue;
    normals.push(k.ball ? unit(sub(p, k.c)) : unit(k.n));
  }
  if (rn === 0) return { violation, residual: 0, active: normals.length };
  let residual = 1;
  const fit = (N: V3[]) => {
    // Normal equations G lam = N^T r, solved by Cramer for |N| <= 3.
    const k = N.length, G = N.map(a => N.map(b => dot(a, b))), h = N.map(a => dot(a, r));
    let lam: number[];
    if (k === 1) lam = [h[0] / G[0][0]];
    else if (k === 2) { const det = G[0][0] * G[1][1] - G[0][1] * G[1][0]; if (Math.abs(det) < 1e-24) return; lam = [(h[0] * G[1][1] - G[0][1] * h[1]) / det, (G[0][0] * h[1] - h[0] * G[1][0]) / det]; }
    else {
      const det3 = (M: number[][]) => M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1]) - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0]) + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]);
      const det = det3(G);
      if (Math.abs(det) < 1e-24) return;
      lam = [0, 1, 2].map(c => det3(G.map((row, i) => row.map((v, j) => (j === c ? h[i] : v)))) / det);
    }
    if (lam.some(l => l < -1e-9 * rn)) return;
    const res = N.reduce((q, a, i) => sub(q, scale(a, lam[i])), r);
    residual = Math.min(residual, norm(res) / rn);
  };
  const n = normals.length;
  for (let i = 0; i < n; i++) {
    fit([normals[i]]);
    for (let j = i + 1; j < n; j++) { fit([normals[i], normals[j]]); for (let k = j + 1; k < n; k++) fit([normals[i], normals[j], normals[k]]); }
  }
  return { violation, residual, active: n };
}

/** A solid piece's certificates relative to the anchor: its ball and its eight orthant faces, (D o s - q).Rb (z - o) <= 1. */
export function solidPieceCons(o: V3, q: V3, R: number, D: V3, Rb: number[][] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]): ProjCon[] {
  const out: ProjCon[] = [{ ball: true, c: o, R }];
  for (let s = 0; s < 8; s++) {
    const sg: V3 = [s & 1 ? 1 : -1, s & 2 ? 1 : -1, s & 4 ? 1 : -1];
    const gb: V3 = [D[0] * sg[0] - q[0], D[1] * sg[1] - q[1], D[2] * sg[2] - q[2]];
    // n = Rb^T gb.
    const n: V3 = [0, 1, 2].map(j => Rb[0][j] * gb[0] + Rb[1][j] * gb[1] + Rb[2][j] * gb[2]) as V3;
    if (norm(n) > 0) out.push({ ball: false, n, beta: 1 + dot(n, o) });
  }
  return out;
}

/** The certificates of S_v(a) (and the solids) relative to its anchor c_v(a). */
export function levelledCons(V: LevelledVertex, solids: WorldHalfSpace[], a: number): { c: V3; cons: ProjCon[] } {
  const c = add(V.X, scale(V.w, a));
  const cons: ProjCon[] = [{ ball: true, c: [0, 0, 0], R: V.R }];
  for (const r of V.planes) cons.push({ ball: false, n: scale(r.m, -1), beta: -r.b });
  for (const s of solids) cons.push({ ball: false, n: scale(s.m, -1), beta: dot(s.m, c) - s.h });
  return { c, cons };
}

/**
 * The projection of a trial y onto S_v(a), as ogcProject computes it: y itself when every
 * certificate keeps it; else its exact Euclidean projection (workingSetProjection), and then the largest t of the
 * segment from the anchor c_v(a) (in every certificate) to that point that every certificate
 * keeps.
 */
export function levelledProject(V: LevelledVertex, solids: WorldHalfSpace[], y: V3, a: number): V3 {
  if (levelledInside(V, solids, y, a)) return y;
  const { c, cons } = levelledCons(V, solids, a);
  const zy = sub(y, c);
  // The GPU's algorithm in Float64 (the brute force's equal on random sets: oracle test); records run to dozens a vertex here.
  // Aimed, as on the GPU, at the set pulled in by 2^-16 (R + |y - c|).
  const sc = V.R + norm(zy);
  const e = workingSetProjection(shrinkCons(cons, 2 ** -16 * sc), zy, 1e-14 * sc).p;
  let t = ballExit([0, 0, 0], e, V.R);
  for (const r of V.planes) { const s = dot(r.m, e); if (s < 0) t = Math.min(t, Math.max(0, r.b / s)); }
  for (const s of solids) { const k = dot(s.m, e); if (k < 0) t = Math.min(t, Math.max(0, (s.h - dot(s.m, c)) / k)); }
  return add(c, scale(e, t));
}

/**
 * One line search of the round: the iterate x at level L moves along d; trial k
 * (alpha = 2^-k) is certified at level a_k = L + alpha (1 - L), each vertex's trial
 * x + alpha d projected onto its S_v(a_k).
 */
export function levelledLadder(round: LevelledRound, solids: WorldHalfSpace[], x: V3[], d: V3[], L: number, trials = 8): { alpha: number; a: number; y: V3[] }[] {
  const out: { alpha: number; a: number; y: V3[] }[] = [];
  for (let k = 0, alpha = 1; k < trials; k++, alpha *= 0.5) {
    const a = L + alpha * (1 - L);
    out.push({ alpha, a, y: x.map((xv, i) => levelledProject(round.vertices[i], solids, add(xv, scale(d[i], alpha)), a)) });
  }
  return out;
}

/**
 * A random vertex certificate set as the filter records it, relative to its anchor (which every
 * certificate holds): the cap radius R, plane records m.z >= b (b <= 0; some near-parallel to the
 * one before, 1-10 degrees off), and optionally a static solid piece and a moving one (origin o,
 * q, its ball Rq, D, and for the moving one its frame Rb: bodyRb, else random), each holding the
 * anchor; and a trial y.
 */
export interface ProjectionPiece { o: V3; q: V3; Rq: number; D: V3; Rb: number[][] }
export interface ProjectionCase { R: number; planes: { m: V3; b: number }[]; solid?: ProjectionPiece; body?: ProjectionPiece; y: V3 }
export function projectionCase(R: () => number, bodyRb?: number[][]): ProjectionCase {
  const unitRand = (): V3 => { for (;;) { const v: V3 = [2 * R() - 1, 2 * R() - 1, 2 * R() - 1]; const l = norm(v); if (l > 0.1 && l <= 1) return scale(v, 1 / l); } };
  const cap = 1e-3 + 19e-3 * R();
  const planes: { m: V3; b: number }[] = [];
  const count = Math.floor(R() * (R() < 0.2 ? 25 : 10));
  for (let i = 0; i < count; i++) {
    let m = unitRand();
    if (i > 0 && R() < 0.4) { const prev = planes[i - 1].m, t = unitRand(), ang = (1 + 9 * R()) * Math.PI / 180; m = unit(add(prev, scale(unit(sub(t, scale(prev, dot(t, prev)))), Math.tan(ang)))); }
    planes.push({ m, b: -0.5 * cap * R() * R() });
  }
  const piece = (rotated: boolean): ProjectionPiece => {
    const o = scale(unitRand(), 0.5 * cap * R()), Rq = norm(o) + (0.2 + 2.8 * R()) * cap;
    let q: V3 = [0, 0, 0], D: V3 = [0, 0, 0];
    if (R() < 0.8) {
      q = scale(unitRand(), 2 * R() / cap); D = [R() / cap, R() / cap, R() < 0.5 ? 0 : R() / cap];
      // Hold the anchor: u0 = Rb (0 - o), f(u0) = D.|u0| - q.u0 < 1.
      const u0 = scale(rb.map(row => dot(row as V3, o)) as V3, -1);
      const f0 = dot(D, u0.map(Math.abs) as V3) - dot(q, u0);
      if (f0 > 0.9) { q = scale(q, 0.9 / f0); D = scale(D, 0.9 / f0); }
    }
    return { o, q, Rq, D, Rb: rb };
  };
  const identity = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  let rb = identity;
  const solid = R() < 0.5 ? piece(false) : undefined;
  rb = bodyRb ?? (() => {
    const a = unitRand(), th = Math.PI * R(), c = Math.cos(th), s = Math.sin(th), K = [[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]];
    return [0, 1, 2].map(i => [0, 1, 2].map(j => (i === j ? c : 0) + s * K[i][j] + (1 - c) * a[i] * a[j]));
  })();
  const body = R() < 0.35 ? piece(true) : undefined;
  const y = scale(unitRand(), (0.3 + 2.7 * R()) * cap);
  return { R: cap, planes, solid, body, y };
}
/** The case's certificates as ProjCon (anchor-relative). */
export function projectionCaseCons(k: ProjectionCase): ProjCon[] {
  const cons: ProjCon[] = [{ ball: true, c: [0, 0, 0], R: k.R }, ...k.planes.map(p => ({ ball: false as const, n: scale(p.m, -1), beta: -p.b }))];
  for (const s of [k.solid, k.body]) if (s) cons.push(...solidPieceCons(s.o, s.q, s.Rq, s.D, s.Rb));
  return cons;
}
/**
 * The trial's projection as ogcProjectAt computes it, in Float64: the metric H = Hinv^-1 (C1's H_vv), trace-normalised
 * (M = 3 Hinv / tr Hinv) and Cholesky-factored (M = K K^T), the exact projection in w = K^-1 z of the half-space
 * certificates pulled in by margin (in z) - the Euclidean brute force there - and then the largest t <= 1 of the segment
 * from o (inside every certificate) to that point that every certificate, the balls included, keeps. Also the KKT of the
 * plane projection in w (its residual over |w_y - w_p|), and t.
 */
export function metricProjection(cons: ProjCon[], zy: V3, Hinv: number[][], margin: number, tol: number, o: V3 = [0, 0, 0]): { p: V3; plane: V3; W: number[]; t: number; kkt: { violation: number; residual: number; active: number }; toW: (z: V3) => V3 } {
  const tr = Hinv[0]![0]! + Hinv[1]![1]! + Hinv[2]![2]!, M = Hinv.map(r => r.map(x => 3 * x / tr));
  const l00 = Math.sqrt(M[0]![0]!), l10 = M[0]![1]! / l00, l20 = M[0]![2]! / l00, l11 = Math.sqrt(M[1]![1]! - l10 * l10);
  const l21 = (M[1]![2]! - l20 * l10) / l11, l22 = Math.sqrt(M[2]![2]! - l20 * l20 - l21 * l21);
  if (!(l00 > 0 && l11 > 0 && l22 > 0)) throw new Error("metricProjection: H_vv^-1 is not positive definite");
  const Kz = (w: V3): V3 => [l00 * w[0], l10 * w[0] + l11 * w[1], l20 * w[0] + l21 * w[1] + l22 * w[2]];
  const KT = (n: V3): V3 => [l00 * n[0] + l10 * n[1] + l20 * n[2], l11 * n[1] + l21 * n[2], l22 * n[2]];
  const solve = (z: V3): V3 => { const w0 = z[0] / l00, w1 = (z[1] - l10 * w0) / l11; return [w0, w1, (z[2] - l20 * w0 - l21 * w1) / l22]; };
  const planes = shrinkCons(cons.filter(k => !k.ball), margin).map(k => { const h = k as { n: V3; beta: number }; return { ball: false as const, n: KT(h.n), beta: h.beta }; });
  const wy = solve(zy), bf = bruteForceProjection(planes, wy, tol);
  if (!bf) throw new Error("metricProjection: no feasible candidate");
  const plane = Kz(bf.p), e = sub(plane, o);
  let t = 1;
  for (const k of cons) {
    if (k.ball) { t = Math.min(t, ballExit(sub(o, k.c), e, k.R)); continue; }
    const sN = dot(k.n, e), room = k.beta - dot(k.n, o);
    if (sN > room) t = Math.min(t, Math.max(0, room / sN));
  }
  return { p: add(o, scale(e, t)), plane, W: bf.W, t, kkt: projectionKkt(planes, wy, bf.p, 1e3 * tol), toW: solve };
}
/** A random symmetric positive definite H_vv^-1 with condition number kappa: Q diag(l) Q^T, l log-spaced over [1, kappa]. */
export function randomMetric(R: () => number, kappa: number): number[][] {
  const unitV = (): V3 => { for (;;) { const v: V3 = [2 * R() - 1, 2 * R() - 1, 2 * R() - 1]; const l = norm(v); if (l > 0.1 && l <= 1) return scale(v, 1 / l); } };
  const a = unitV(), b0 = unitV(), b = unit(sub(b0, scale(a, dot(a, b0)))), c = cross(a, b);
  const l = [1, kappa ** R(), kappa].map(x => x * (0.5 + R()));
  return [0, 1, 2].map(i => [0, 1, 2].map(j => l[0]! * a[i]! * a[j]! + l[1]! * b[i]! * b[j]! + l[2]! * c[i]! * c[j]!));
}
/** cons with every certificate pulled in by margin (the set ogcProjectExact aims at, margin = 2^-16 (R + |y - c|)). */
export const shrinkCons = (cons: ProjCon[], margin: number): ProjCon[] => cons.map(k => (k.ball ? { ...k, R: Math.max(k.R - margin, 0) } : { ...k, beta: k.beta - margin * norm(k.n) }));
