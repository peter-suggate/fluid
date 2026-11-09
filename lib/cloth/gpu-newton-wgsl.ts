/**
 * WGSL of the Newton-PCG cloth step (appended to the solver module, which supplies the element helpers). One implicit
 * step per call, no substeps: Newton iterations on the step's objective G (BDF2 inertia, membrane, Bergou bending,
 * frozen-tangent damping, grab spring), each solving H d = -g by preconditioned conjugate gradients and taking the
 * largest Armijo step of 1, 1/2, ..., 1/128 on G. Pins are Dirichlet DOFs held at their frame targets (their rows and
 * directions are zero). Everything is decided on the GPU: convergence and stalls set control flags that every later
 * kernel of the step reads, so the CPU encodes fixed caps and never reads back.
 *
 * H is matrix-free: per face the Green-strain tangent J D J^T plus the geometric term S (x) I with the 2x2 stress S
 * clamped PSD (nwFaces records f0, f1 and S once per Newton iteration); bending and damping are exact (quadratic). The
 * preconditioner is the inverse of each vertex's 3x3 diagonal block (the VBD block of vertexSystem).
 *
 * The `nw` buffer (vec4 records): per-vertex segments (stride Np) g, r, z, d, p (two, ping-pong), q, M^-1 (two records),
 * then 3 records per face, then the workgroup partials (W = workgroups of a per-vertex dispatch each): G (r.z, g.g at
 * the Newton iterate), A (p.q), B0..B2 (r.z by PCG iteration mod 3), trial (5 per workgroup), then the control record
 * (Newton done, PCG done, accepted step, Newton iteration), then |g|^2 at the first and latest Newton iterate.
 */
import { CLOTH_LINE_SEARCH } from './oracle';

export const CLOTH_NEWTON_SEGMENTS = { g: 0, r: 1, z: 2, d: 3, p0: 4, p1: 5, q: 6, m0: 7, m1: 8 } as const;
export const CLOTH_NEWTON_SEGMENT_COUNT = 9;
/** Line-search trials per Newton iteration: steps 1, 1/2, ..., 2^-(TRIALS-1). */
export const CLOTH_NEWTON_TRIALS = 8;
/** Partial arrays (each W records) before the trial block: G, A, B0, B1, B2. */
export const CLOTH_NEWTON_PARTIALS = 5;
export const CLOTH_NEWTON_TRIAL_RECORDS = 5;

export function clothNewtonWGSL(binding: number, workgroup: number, contact = false): string {
  const S = CLOTH_NEWTON_SEGMENTS;
  return /* wgsl */`
@group(0) @binding(${binding}) var<storage, read_write> nw: array<vec4f>;
const NW_WG = ${workgroup}u;
fn nseg(s: u32, v: u32) -> u32 { return s * P.state0.x + v; }
fn nwFaceRecord(f: u32) -> u32 { return P.newton0.x + 3u * f; }
fn nwPartial(k: u32) -> u32 { return P.newton0.y + k * P.newton0.z; }
fn nwControl() -> u32 { return P.newton0.w; }
fn nwPinned(v: u32) -> bool { return vertexGroup(v) > 0u; }
// The largest step the line search tries: the contact filter's alpha_min (every vertex's certificates hold on [0, alpha_min] d), else 1.
fn nwAlpha0() -> f32 { return ${contact ? 'clamp(ogcAlphaMin(), 0.0, 1.0)' : '1.0'}; }

var<workgroup> nwRed: array<vec4f, ${workgroup}>;
var<workgroup> nwOut: vec4f;
// The sum of partial array \`base\` (W records) by every workgroup, uniform across it.
fn nwSum(lane: u32, base: u32) -> vec4f {
  var a = vec4f(0.0);
  for (var i = lane; i < P.newton0.z; i += NW_WG) { a += nw[base + i]; }
  nwRed[lane] = a;
  workgroupBarrier();
  for (var s = NW_WG / 2u; s > 0u; s >>= 1u) { if (lane < s) { nwRed[lane] += nwRed[lane + s]; } workgroupBarrier(); }
  if (lane == 0u) { nwOut = nwRed[0]; }
  return workgroupUniformLoad(&nwOut);
}
// This workgroup's sum of \`value\` into record \`wgi\` of partial array \`base\`.
fn nwPartialSum(lane: u32, wgi: u32, value: vec4f, base: u32) {
  workgroupBarrier();
  nwRed[lane] = value;
  workgroupBarrier();
  for (var s = NW_WG / 2u; s > 0u; s >>= 1u) { if (lane < s) { nwRed[lane] += nwRed[lane + s]; } workgroupBarrier(); }
  if (lane == 0u) { nw[base + wgi] = nwRed[0]; }
}
var<workgroup> nwCtl: vec4f;
fn nwControlUniform(lane: u32) -> vec4f {
  if (lane == 0u) { nwCtl = nw[nwControl()]; }
  return workgroupUniformLoad(&nwCtl);
}
fn wgIndex(wg: vec3u, nwg: vec3u) -> u32 { return wg.x + wg.y * nwg.x; }

// Step start (one thread): Newton not done.
@compute @workgroup_size(1) fn nwBeginStep() { nw[nwControl()] = vec4f(0.0); }

// Per face at the Newton iterate: f0, f1, the PSD-clamped stress (S11, S22, S12) and the softened tangent diagonal.
@compute @workgroup_size(${workgroup}) fn nwFaces(@builtin(global_invocation_id) id: vec3u) {
  let f = id.x;
  if (f >= P.n.y || nw[nwControl()].x != 0.0) { return; }
  let t = tris(f); let dm = rest[P.rest0.x + f]; let c0 = rest[P.rest1.z + 2u * t.w]; let c1 = rest[P.rest1.z + 2u * t.w + 1u];
  let x0 = pos(t.x); let e1 = pos(t.y) - x0; let e2 = pos(t.z) - x0;
  let f0 = e1 * dm.x + e2 * dm.y; let f1 = e1 * dm.z + e2 * dm.w;
  let e11 = 0.5 * (dot(f0, f0) - 1.0); let e22 = 0.5 * (dot(f1, f1) - 1.0); let gam = dot(f0, f1);
  let w11 = select(0.0, (1.0 - c1.z) * c1.x, e11 < 0.0); let w22 = select(0.0, (1.0 - c1.z) * c1.y, e22 < 0.0);
  let s11 = c0.x * e11 + c0.z * e22 - w11 * e11; let s22 = c0.y * e22 + c0.z * e11 - w22 * e22; let tau = c0.w * gam;
  // The 2x2 stress clamped PSD: S (x) I is the face's geometric stiffness; a compressive eigenvalue drops out.
  let tr = 0.5 * (s11 + s22); let df = length(vec2f(0.5 * (s11 - s22), tau)); let l1 = tr + df; let l2 = tr - df;
  var S = vec3f(s11, s22, tau);
  if (l2 < 0.0) {
    if (l1 <= 0.0) { S = vec3f(0.0); } else {
      var e = vec2f(l1 - s22, tau);
      if (dot(e, e) <= 0.0) { e = vec2f(tau, l1 - s11); }
      if (dot(e, e) <= 0.0) { e = vec2f(1.0, 0.0); }
      e = normalize(e); S = l1 * vec3f(e.x * e.x, e.y * e.y, e.x * e.y);
    }
  }
  let r = nwFaceRecord(f);
  nw[r] = vec4f(f0, S.x); nw[r + 1u] = vec4f(f1, S.y); nw[r + 2u] = vec4f(S.z, c0.x - w11, c0.y - w22, 0.0);
}

// Per vertex: g at the iterate, the inverse diagonal block, and the PCG start r = -g, z = M^-1 r, d = 0, p = 0.
@compute @workgroup_size(${workgroup}) fn nwGradient(@builtin(global_invocation_id) id: vec3u, @builtin(local_invocation_index) lane: u32, @builtin(workgroup_id) wg: vec3u, @builtin(num_workgroups) nwg: vec3u) {
  let ctl = nwControlUniform(lane);
  if (ctl.x != 0.0) { return; }
  let v = wgIndex(wg, nwg) * NW_WG + lane;
  var part = vec4f(0.0);
  if (v < P.n.x) {
    var g = vec3f(0.0); var M0 = vec4f(0.0); var M1 = vec4f(0.0);
    if (!nwPinned(v)) {
      let s = vertexSystem(v, pos(v));
      g = s.g;
      let H = s.H; let det = determinant(H);
      if (det > 0.0 && abs(det) <= 3.0e38) {
        let c0 = cross(H[1], H[2]); let c1 = cross(H[2], H[0]); let c2 = cross(H[0], H[1]);
        M0 = vec4f(c0.x, c0.y, c0.z, c1.y) / det; M1 = vec4f(c1.z, c2.z, 0.0, 0.0) / det;
      } else {
        M0 = vec4f(1.0 / max(H[0][0], 1e-30), 0.0, 0.0, 1.0 / max(H[1][1], 1e-30)); M1 = vec4f(0.0, 1.0 / max(H[2][2], 1e-30), 0.0, 0.0);
      }
    }
    let r = -g; let z = nwPrecondition(M0, M1, r);
    nw[nseg(${S.g}u, v)] = vec4f(g, 0.0); nw[nseg(${S.r}u, v)] = vec4f(r, 0.0); nw[nseg(${S.z}u, v)] = vec4f(z, 0.0);
    nw[nseg(${S.d}u, v)] = vec4f(0.0); nw[nseg(${S.p0}u, v)] = vec4f(0.0);
    nw[nseg(${S.m0}u, v)] = M0; nw[nseg(${S.m1}u, v)] = M1;
    part = vec4f(dot(r, z), dot(g, g), 0.0, 0.0);
  }
  let wgi = wgIndex(wg, nwg);
  nwPartialSum(lane, wgi, part, nwPartial(0u));
  nwPartialSum(lane, wgi, part, nwPartial(2u));
}
// M^-1 r with the symmetric inverse (xx, xy, xz, yy), (yz, zz).
fn nwPrecondition(M0: vec4f, M1: vec4f, r: vec3f) -> vec3f {
  return vec3f(M0.x * r.x + M0.y * r.y + M0.z * r.z, M0.y * r.x + M0.w * r.y + M1.x * r.z, M0.z * r.x + M1.x * r.y + M1.y * r.z);
}

var<private> nwBeta: f32;
var<private> nwPOld: u32;
// p_i at vertex j, formed on the fly from z and p_{i-1}: p_i = z + beta p_{i-1} (pinned vertices have z = p = 0).
fn nwP(j: u32) -> vec3f { return nw[nseg(${S.z}u, j)].xyz + nwBeta * nw[nseg(nwPOld, j)].xyz; }

// (H p)_v: inertia, grab, membrane (tangent + clamped geometric), face damping, bending stencil.
fn nwApplyAt(v: u32, pv: vec3f) -> vec3f {
  let h = hEff(); let m = vertexRest(v).x;
  var out = (m / (h * h) + grabSpring(v, pos(v)).w) * pv;
  let damped = P.state1.z != 0u;
  for (var j = topo[P.topo0.x + v]; j < topo[P.topo0.x + v + 1u]; j++) {
    let entry = topo[P.topo0.y + j]; let f = entry >> 2u; let k = entry & 3u;
    let t = tris(f); let dm = rest[P.rest0.x + f]; let fr = rest[P.rest0.y + f]; let c0 = rest[P.rest1.z + 2u * t.w];
    let a = vec3f(-dm.x - dm.y, dm.x, dm.y); let b = vec3f(-dm.z - dm.w, dm.z, dm.w);
    let p0 = select(nwP(t.x), pv, k == 0u); let p1 = select(nwP(t.y), pv, k == 1u); let p2 = select(nwP(t.z), pv, k == 2u);
    let d0 = a.x * p0 + a.y * p1 + a.z * p2; let d1 = b.x * p0 + b.y * p1 + b.z * p2;
    let ak = select(select(a.x, a.y, k == 1u), a.z, k == 2u); let bk = select(select(b.x, b.y, k == 1u), b.z, k == 2u);
    let r = nwFaceRecord(f); let F0 = nw[r]; let F1 = nw[r + 1u]; let F2 = nw[r + 2u];
    let f0 = F0.xyz; let f1 = F1.xyz;
    let rate = vec3f(dot(f0, d0), dot(f1, d1), dot(f0, d1) + dot(f1, d0));
    let sg = fr.x * vec3f(F2.y * rate.x + c0.z * rate.y, c0.z * rate.x + F2.z * rate.y, c0.w * rate.z);
    out += sg.x * ak * f0 + sg.y * bk * f1 + sg.z * (ak * f1 + bk * f0) + fr.x * (ak * (F0.w * d0 + F2.x * d1) + bk * (F2.x * d0 + F1.w * d1));
    if (damped && fr.y > 0.0) {
      let T0 = state[P.state1.x + 2u * f]; let T1 = state[P.state1.x + 2u * f + 1u];
      let g0 = T0.xyz; let g1 = T1.xyz; let sc = fr.y * fr.x / h;
      let rt = vec3f(dot(g0, d0), dot(g1, d1), dot(g0, d1) + dot(g1, d0));
      let sd = sc * vec3f(T0.w * rt.x + c0.z * rt.y, c0.z * rt.x + T1.w * rt.y, c0.w * rt.z);
      out += sd.x * ak * g0 + sd.y * bk * g1 + sd.z * (ak * g1 + bk * g0);
    }
  }
  // Strain-limit edges (augmented Lagrangian, active only): k n n^T + (f / len) (I - n n^T) on p_v - p_other.
  if (P.flags.z != 0u) {
    let xv = pos(v);
    for (var j = topo[P.topo1.x + v]; j < topo[P.topo1.x + v + 1u]; j++) {
      let e = topo[P.topo1.y + j] >> 2u;
      if (!(rest[P.rest0.w + e].y > 0.0)) { continue; }
      let ev = edges(e); let other = select(ev.x, ev.y, ev.x == v);
      let dd = xv - pos(other); let len = length(dd);
      let dual = edgeDual(e); let f = dual.y * (len - dual.z) + dual.x;
      if (!(len > 0.0) || !(f > 0.0)) { continue; }
      let n = dd / len; let dp = pv - nwP(other); let dn = dot(n, dp);
      out += dual.y * dn * n + (f / len) * (dp - dn * n);
    }
  }
  out += stencilDiagonal(v) * pv;
  // Contact pairs (C2): the sum over v's entries of w_v M (sum_j w_j p_j), linearized at the iterate.
  ${contact ? 'out += ogcVertexContactHvLanes(v, 0u, 1u);' : ''}
  let hd = 1.0 / h;
  for (var j = topo[P.topo2.z + v]; j < topo[P.topo2.z + v + 1u]; j++) {
    let rec = topo4[P.newton1.x + j];
    out += (bitcast<f32>(rec.y) + select(0.0, bitcast<f32>(rec.z) * hd, damped)) * nwP(rec.x);
  }
  return out;
}

// PCG iteration i (R.range.z, from 1): p_i = z + beta p_{i-1}, q = H p_i, and the partial p.q. The first iteration of
// a Newton iteration also tests the residual; any iteration tests the PCG tolerance (both from the partials).
@compute @workgroup_size(${workgroup}) fn pcgApply(@builtin(local_invocation_index) lane: u32, @builtin(workgroup_id) wg: vec3u, @builtin(num_workgroups) nwg: vec3u) {
  let ctl = nwControlUniform(lane);
  if (ctl.x != 0.0 || ctl.y != 0.0) { return; }
  let i = R.range.z;
  let G = nwSum(lane, nwPartial(0u));
  let rz1 = nwSum(lane, nwPartial(2u + (i - 1u) % 3u)).x;
  var beta = 0.0;
  if (i > 1u) { beta = rz1 / nwSum(lane, nwPartial(2u + (i - 2u) % 3u)).x; }
  let wgi = wgIndex(wg, nwg);
  // Residual record (control + 1): |g|^2 over free DOFs at the step's first and latest Newton iterate.
  if (i == 1u && wgi == 0u && lane == 0u) { var rr = nw[nwControl() + 1u]; if (ctl.w == 0.0) { rr.x = G.y; } rr.y = G.y; nw[nwControl() + 1u] = rr; }
  if (i == 1u && G.y <= bitcast<f32>(P.newton1.z)) {
    if (wgi == 0u && lane == 0u) { nw[nwControl()].x = 1.0; }
    return;
  }
  if (!(rz1 > bitcast<f32>(P.newton1.y) * G.x)) {
    if (wgi == 0u && lane == 0u) { nw[nwControl()].y = 1.0; }
    return;
  }
  nwBeta = beta; nwPOld = ${S.p0}u + (i - 1u) % 2u;
  let v = wgi * NW_WG + lane;
  var part = vec4f(0.0);
  if (v < P.n.x) {
    var q = vec3f(0.0); var p = vec3f(0.0);
    if (!nwPinned(v)) { p = nwP(v); q = nwApplyAt(v, p); }
    nw[nseg(${S.p0}u + i % 2u, v)] = vec4f(p, 0.0); nw[nseg(${S.q}u, v)] = vec4f(q, 0.0);
    part = vec4f(dot(p, q), 0.0, 0.0, 0.0);
  }
  nwPartialSum(lane, wgi, part, nwPartial(1u));
}

// PCG iteration i: alpha = r.z / p.q; d += alpha p, r -= alpha q, z = M^-1 r, and the partial r.z of iteration i.
@compute @workgroup_size(${workgroup}) fn pcgUpdate(@builtin(local_invocation_index) lane: u32, @builtin(workgroup_id) wg: vec3u, @builtin(num_workgroups) nwg: vec3u) {
  let ctl = nwControlUniform(lane);
  if (ctl.x != 0.0 || ctl.y != 0.0) { return; }
  let i = R.range.z;
  let rz1 = nwSum(lane, nwPartial(2u + (i - 1u) % 3u)).x;
  let pq = nwSum(lane, nwPartial(1u)).x;
  let wgi = wgIndex(wg, nwg);
  // H is PSD by construction; a non-positive curvature (rounding at convergence) ends this solve with d so far.
  if (!(pq > 0.0)) {
    if (wgi == 0u && lane == 0u) { nw[nwControl()].y = 1.0; }
    return;
  }
  let alpha = rz1 / pq;
  let v = wgi * NW_WG + lane;
  var part = vec4f(0.0);
  if (v < P.n.x) {
    let p = nw[nseg(${S.p0}u + i % 2u, v)].xyz;
    let d = nw[nseg(${S.d}u, v)].xyz + alpha * p;
    let r = nw[nseg(${S.r}u, v)].xyz - alpha * nw[nseg(${S.q}u, v)].xyz;
    let z = nwPrecondition(nw[nseg(${S.m0}u, v)], nw[nseg(${S.m1}u, v)], r);
    nw[nseg(${S.d}u, v)] = vec4f(d, 0.0); nw[nseg(${S.r}u, v)] = vec4f(r, 0.0); nw[nseg(${S.z}u, v)] = vec4f(z, 0.0);
    part = vec4f(dot(r, z), 0.0, 0.0, 0.0);
  }
  nwPartialSum(lane, wgi, part, nwPartial(2u + i % 3u));
}

// Line search: G(x + alpha d) - G(x) in difference form for alpha = 2^-k, k < ${CLOTH_NEWTON_TRIALS}, with its rounding floor, and
// the slope g.d. Each vertex takes its own terms, its bending row and the faces whose first slot it is.
var<private> nwE: array<vec4f, 2>;
var<private> nwF: array<vec4f, 2>;
fn nwAdd(k: u32, e: vec2f) { nwE[k / 4u][k % 4u] += e.x; nwF[k / 4u][k % 4u] += e.y; }
fn nwD(j: u32) -> vec3f { return nw[nseg(${S.d}u, j)].xyz; }
@compute @workgroup_size(${workgroup}) fn nwTrial(@builtin(local_invocation_index) lane: u32, @builtin(workgroup_id) wg: vec3u, @builtin(num_workgroups) nwg: vec3u) {
  let ctl = nwControlUniform(lane);
  if (ctl.x != 0.0) { return; }
  let wgi = wgIndex(wg, nwg); let v = wgi * NW_WG + lane;
  let a0 = nwAlpha0();
  nwE = array<vec4f, 2>(vec4f(0.0), vec4f(0.0)); nwF = array<vec4f, 2>(vec4f(0.0), vec4f(0.0));
  var slope = 0.0;
  if (v < P.n.x) {
    let h = hEff(); let xv = pos(v); let dv = nwD(v);
    slope = dot(nw[nseg(${S.g}u, v)].xyz, dv);
    let w = vertexRest(v).x / (h * h); let rv = xv - state[seg(S_Y, v)].xyz; let gs = grabSpring(v, xv);
    let damped = P.state1.z != 0u;
    var alpha = a0;
    for (var k = 0u; k < ${CLOTH_NEWTON_TRIALS}u; k++) {
      let d = alpha * dv;
      let lin = w * dot(rv, d) + dot(gs.xyz, d); let quad = 0.5 * (w + gs.w) * dot(d, d);
      nwAdd(k, vec2f(lin + quad, abs(lin) + quad + (w * length(rv) + length(gs.xyz)) * length(d)));
      alpha *= 0.5;
    }
    for (var j = topo[P.topo0.x + v]; j < topo[P.topo0.x + v + 1u]; j++) {
      let entry = topo[P.topo0.y + j];
      if ((entry & 3u) != 0u) { continue; }
      let f = entry >> 2u; let t = tris(f); let dm = rest[P.rest0.x + f]; let fr = rest[P.rest0.y + f];
      let c0 = rest[P.rest1.z + 2u * t.w]; let c1 = rest[P.rest1.z + 2u * t.w + 1u];
      let a = vec3f(-dm.x - dm.y, dm.x, dm.y); let b = vec3f(-dm.z - dm.w, dm.z, dm.w);
      let x0 = pos(t.x); let x1 = pos(t.y); let x2 = pos(t.z);
      let f0 = (x1 - x0) * dm.x + (x2 - x0) * dm.y; let f1 = (x1 - x0) * dm.z + (x2 - x0) * dm.w;
      let D0 = nwD(t.x); let D1 = nwD(t.y); let D2 = nwD(t.z);
      let g0 = a.x * D0 + a.y * D1 + a.z * D2; let g1 = b.x * D0 + b.y * D1 + b.z * D2;
      let e11 = 0.5 * (dot(f0, f0) - 1.0); let e22 = 0.5 * (dot(f1, f1) - 1.0); let gam = dot(f0, f1);
      let s11 = (1.0 - c1.z) * c1.x; let s22 = (1.0 - c1.z) * c1.y;
      var T0 = vec4f(0.0); var T1 = vec4f(0.0); var rate = vec3f(0.0); var roundRate = vec3f(0.0); var drate1 = vec3f(0.0);
      let sc = fr.y * fr.x / h;
      if (damped && fr.y > 0.0) {
        T0 = state[P.state1.x + 2u * f]; T1 = state[P.state1.x + 2u * f + 1u];
        let u0 = x0 - state[seg(S_XHAT, t.x)].xyz; let u1 = x1 - state[seg(S_XHAT, t.y)].xyz; let u2 = x2 - state[seg(S_XHAT, t.z)].xyz;
        let q0 = a.x * u0 + a.y * u1 + a.z * u2; let q1 = b.x * u0 + b.y * u1 + b.z * u2;
        rate = vec3f(dot(T0.xyz, q0), dot(T1.xyz, q1), dot(T0.xyz, q1) + dot(T1.xyz, q0));
        let ru = vec2f(abs(a.x) * length(u0) + abs(a.y) * length(u1) + abs(a.z) * length(u2), abs(b.x) * length(u0) + abs(b.y) * length(u1) + abs(b.z) * length(u2));
        roundRate = vec3f(length(T0.xyz) * ru.x, length(T1.xyz) * ru.y, length(T0.xyz) * ru.y + length(T1.xyz) * ru.x);
        drate1 = vec3f(dot(T0.xyz, g0), dot(T1.xyz, g1), dot(T0.xyz, g1) + dot(T1.xyz, g0));
      }
      var alpha = a0;
      for (var k = 0u; k < ${CLOTH_NEWTON_TRIALS}u; k++) {
        let df0 = alpha * g0; let df1 = alpha * g1;
        let d11 = dot(f0, df0) + 0.5 * dot(df0, df0); let d22 = dot(f1, df1) + 0.5 * dot(df1, df1);
        let dg = dot(f0, df1) + dot(df0, f1) + dot(df0, df1);
        let q11 = quadDelta(c0.x - select(0.0, s11, e11 < 0.0), c0.x - select(0.0, s11, e11 + d11 < 0.0), e11, d11);
        let q22 = quadDelta(c0.y - select(0.0, s22, e22 < 0.0), c0.y - select(0.0, s22, e22 + d22 < 0.0), e22, d22);
        let q12 = c0.z * (e11 * d22 + d11 * e22 + d11 * d22); let qs = c0.w * (gam * dg + 0.5 * dg * dg);
        let rounding = (c0.x + c0.z) * abs(d11) * (0.5 + e11) + (c0.y + c0.z) * abs(d22) * (0.5 + e22) + c0.w * abs(dg) * sqrt(max((1.0 + 2.0 * e11) * (1.0 + 2.0 * e22), 0.0));
        var out = fr.x * vec2f(q11 + q22 + q12 + qs, abs(q11) + abs(q22) + abs(q12) + abs(qs) + rounding);
        if (damped && fr.y > 0.0) {
          let dr = alpha * drate1;
          let sigma = vec3f(T0.w * rate.x + c0.z * rate.y, c0.z * rate.x + T1.w * rate.y, c0.w * rate.z);
          let dsigma = vec3f(T0.w * dr.x + c0.z * dr.y, c0.z * dr.x + T1.w * dr.y, c0.w * dr.z);
          let lin = sc * dot(dr, sigma); let quad = 0.5 * sc * dot(dr, dsigma);
          out += vec2f(lin + quad, abs(lin) + quad + sc * dot(abs(dsigma), roundRate));
        }
        nwAdd(k, out);
        alpha *= 0.5;
      }
    }
    // Vertex v's bending rows (and their damping): R d once a row, then each trial exactly.
    for (var r = BEND_ROWS * v; r < BEND_ROWS * (v + 1u); r++) {
      let q = loadBend(r); var ds1 = vec3f(0.0);
      for (var o = topo[P.topo0.z + r]; o < topo[P.topo0.z + r + 1u]; o++) { ds1 += bendW(o) * (nwD(bendJ(o)) - dv); }
      alpha = a0;
      for (var k = 0u; k < ${CLOTH_NEWTON_TRIALS}u; k++) { nwAdd(k, bendDelta(q, alpha * ds1)); alpha *= 0.5; }
    }
    // Strain-limit edges whose first slot v is: exact in the relative move d_v - d_other.
    if (P.flags.z != 0u) {
      for (var j = topo[P.topo1.x + v]; j < topo[P.topo1.x + v + 1u]; j++) {
        let entry = topo[P.topo1.y + j];
        if ((entry & 3u) != 0u) { continue; }
        let e = entry >> 2u; let ev = edges(e); let other = select(ev.x, ev.y, ev.x == v);
        let rel = dv - nwD(other);
        alpha = a0;
        for (var k = 0u; k < ${CLOTH_NEWTON_TRIALS}u; k++) { nwAdd(k, edgeDelta(v, xv, e, alpha * rel)); alpha *= 0.5; }
      }
    }
    ${contact ? `// Contact pairs v owns (each pair once over all vertices), at x and at every trial x + a d.
    let ec0 = ogcContactEnergyOwnedLanes(v, 0u, 1u);
    alpha = a0;
    for (var k = 0u; k < ${CLOTH_NEWTON_TRIALS}u; k++) {
      nwTrialStep = alpha; let ec = ogcContactEnergyOwnedLanes(v, 0u, 1u); nwTrialStep = 0.0;
      nwAdd(k, vec2f(ec - ec0, abs(ec0) + abs(ec)));
      alpha *= 0.5;
    }` : ''}
  }
  let base = nwPartial(${CLOTH_NEWTON_PARTIALS}u);
  nwPartialSum(lane, wgi, nwE[0], base);
  nwPartialSum(lane, wgi, nwE[1], base + P.newton0.z);
  nwPartialSum(lane, wgi, nwF[0], base + 2u * P.newton0.z);
  nwPartialSum(lane, wgi, nwF[1], base + 3u * P.newton0.z);
  nwPartialSum(lane, wgi, vec4f(slope, 0.0, 0.0, 0.0), base + 4u * P.newton0.z);
}

// One workgroup: the largest Armijo step (0 and Newton done if none descends), and the next Newton iteration's PCG reset.
@compute @workgroup_size(${workgroup}) fn nwChoose(@builtin(local_invocation_index) lane: u32) {
  let ctl = nwControlUniform(lane);
  if (ctl.x != 0.0) { return; }
  let base = nwPartial(${CLOTH_NEWTON_PARTIALS}u);
  let E0 = nwSum(lane, base); let E1 = nwSum(lane, base + P.newton0.z);
  let F0 = nwSum(lane, base + 2u * P.newton0.z); let F1 = nwSum(lane, base + 3u * P.newton0.z);
  let slope = nwSum(lane, base + 4u * P.newton0.z).x;
  if (lane != 0u) { return; }
  let a0 = nwAlpha0();
  // A filter that allows no step (alpha_min = 0: a certificate is already tight) moves nothing and leaves Newton running:
  // it arms a re-detection round, whose new certificates the next Newton iteration steps in.
  if (!(a0 > 0.0)) { nw[nwControl()] = vec4f(0.0, 0.0, 0.0, ctl.w + 1.0); return; }
  var chosen = 0.0; var alpha = a0;
  if (slope < 0.0) {
    for (var k = 0u; k < ${CLOTH_NEWTON_TRIALS}u; k++) {
      let e = select(E1[k - 4u], E0[k], k < 4u); let fl = select(F1[k - 4u], F0[k], k < 4u);
      if (e <= ${CLOTH_LINE_SEARCH.armijo} * alpha * slope + 4.0 * EPS32 * fl) { chosen = alpha; break; }
      alpha *= 0.5;
    }
  }
  nw[nwControl()] = vec4f(select(0.0, 1.0, chosen == 0.0), 0.0, chosen, ctl.w + 1.0);
}

@compute @workgroup_size(${workgroup}) fn nwApply(@builtin(global_invocation_id) id: vec3u) {
  let v = id.x;
  let ctl = nw[nwControl()];
  if (v >= P.n.x) { return; }
  // Newton done: d = 0, so the capped iterations left (all early exits) give the contact filter and re-detection nothing to act on.
  if (ctl.x != 0.0) { nw[nseg(${S.d}u, v)] = vec4f(0.0); return; }
  if (!(ctl.z > 0.0)) { return; }
  positions[v] = vec4f(pos(v) + ctl.z * nwD(v), positions[v].w);
}

// After each applied Newton iteration (not once Newton is done): the AVBD dual update of every strain-limit edge, so
// the next iteration's gradient carries the new multipliers and the residual test sees any constraint still violated.
@compute @workgroup_size(${workgroup}) fn nwEdgeDuals(@builtin(global_invocation_id) id: vec3u) {
  if (nw[nwControl()].x != 0.0) { return; }
  updateEdgeDual(id.x);
}
`;
}
