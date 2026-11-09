/**
 * WGSL of the two-level correction (coarse.ts is its Float64 reference), appended to the cloth solver module:
 * it uses the solver's bindings and element helpers and adds the `coarse` work buffer.
 *
 * Per step, at the initial guess: each face's share of A_c as a quadratic form in its corner weights, A_c = P^T H P
 * per coarse pair (one workgroup each, a fixed-order tree sum), then one workgroup inverts A_c in workgroup memory
 * (the symmetric sweep of its packed lower triangle).
 * Per iteration, before the colour sweeps: the gradient of G (contact included, with the vertex's own entries), the restriction r = -P^T g (one
 * workgroup per node), dc = A_c^-1 r, the prolongation d = P dc, the difference-form change of G at the trial
 * steps 1, 1/2, 1/4, 1/8 in one element pass, the choice of the largest step passing Armijo, and the move.
 */
import { CLOTH_COARSE, CLOTH_COARSE_SLOT_BITS } from './coarse';

/** `coarse` buffer: vec4 records; segment offsets are in Params.coarse3..5. */
export const CLOTH_COARSE_WORKGROUP = 64;
export const CLOTH_COARSE_FACTOR_WORKGROUP = 256;
/** The inverse's workgroup: one SIMD-latency-bound workgroup, so as wide as the device allows (8 row pairs a round). */
export const CLOTH_COARSE_SWEEP_WORKGROUP = 1024;
/** Misc record: (slope, chosen step, factor ok, 0); stats record: corrections, accepted at 1, 1/2, 1/4 (1/8 in the next). */
export const CLOTH_COARSE_MISC = { slope: 0, alpha: 1, ok: 2 } as const;
/** Status bit of a step whose A_c had a non-positive pivot (its corrections were skipped). Reported, not failing. */
export const CLOTH_STATUS_COARSE_INDEFINITE = 2;
/** Most a vertex's contact Hessian block may exceed its own diagonal share s (inertia, pin) by in A_c: f32 resolves a soft
 * direction s under a stiff one k to about (k / s) 2^-23 per operation, so 1e4 keeps the sweep's pivots to ~1e-3 (the
 * curtain's factor failed at k / s 4e9 and passed with pivots 4e-5 of their diagonal at 3e7). */
export const CLOTH_COARSE_CONTACT_RANGE = 1e4;

/** Workgroup memory of coarseFactor: two blocks of three columns of A_c and the ok flag. */
export function clothCoarseFactorBytes(nodes: number): number { return 4 * (18 * nodes + 1); }

/** `workgroup` is the solver's per-thread workgroup size (CLOTH_WORKGROUP); `nodes` sizes the factor's workgroup matrix. */
export function clothCoarseWGSL(binding: number, workgroup: number, nodes: number, contact = false): string {
  if (!(Number.isInteger(nodes) && nodes >= 1 && 3 * nodes < 128)) throw new Error(`Cloth coarse factor takes 1 to 42 nodes (a row pair per 128 lanes); got ${nodes}`);
  const rounds = Array.from({ length: Math.ceil(Math.ceil(3 * nodes / 2) / (CLOTH_COARSE_SWEEP_WORKGROUP / 128)) }, (_, r) => r);
  return /* wgsl */`
// Scalar words: concurrent workgroups write neighbouring words of A_c, so no store may touch a whole vec4.
@group(0) @binding(${binding}) var<storage, read_write> coarse: array<f32>;
const CW = ${CLOTH_COARSE_WORKGROUP}u;
const CFW = ${CLOTH_COARSE_FACTOR_WORKGROUP}u;
const C_ENTRIES = ${CLOTH_COARSE.maxEntries}u;
const C_SLOT_BITS = ${CLOTH_COARSE_SLOT_BITS}u;
const C_NONE = 0xffffffffu;
const C_RANGE: f32 = ${CLOTH_COARSE_CONTACT_RANGE.toExponential()};
fn cf(i: u32) -> f32 { return coarse[i]; }
fn cfSet(i: u32, x: f32) { coarse[i] = x; }
fn c4(r: u32) -> vec4f { let i = 4u * r; return vec4f(coarse[i], coarse[i + 1u], coarse[i + 2u], coarse[i + 3u]); }
fn c4Set(r: u32, x: vec4f) { let i = 4u * r; coarse[i] = x.x; coarse[i + 1u] = x.y; coarse[i + 2u] = x.z; coarse[i + 3u] = x.w; }
// Params.coarse0: node count, dofs m, pair count, rows (topo word)
//        coarse1: owner offsets, owner entries, face node offsets, pairs (topo words)
//        coarse2: pair offsets, pair entries, face own (topo words), row weights (rest record)
//        coarse3: face node records (rest), face shares, A, A^-1 (W1) (coarse records)
//        coarse4: gradient, direction, partials, r (coarse records)
//        coarse5: dc, misc (coarse records), partial count (coarseTrial workgroups), 0
//        coarse6: face, hinge, edge corner gradients, per-vertex contact Hessians (coarse records, two a vertex)
fn cA() -> u32 { return 4u * P.coarse3.z; }
fn cW1() -> u32 { return 4u * P.coarse3.w; }
fn cR() -> u32 { return 4u * P.coarse4.w; }
fn cDc() -> u32 { return 4u * P.coarse5.x; }
fn cMisc() -> u32 { return P.coarse5.y; }
fn rowNode(v: u32, s: u32) -> u32 { return topo[P.coarse0.w + C_ENTRIES * v + s]; }
fn rowWeight(v: u32, s: u32) -> f32 { return rest[P.coarse2.w + (C_ENTRIES / 4u) * v + (s >> 2u)][s & 3u]; }
fn foldedGroup(wg: vec3u, nwg: vec3u) -> u32 { return wg.x + wg.y * nwg.x; }

var<workgroup> cRa: array<vec4f, ${CLOTH_COARSE_FACTOR_WORKGROUP}>;
var<workgroup> cRb: array<vec4f, ${CLOTH_COARSE_FACTOR_WORKGROUP}>;
var<workgroup> cRc: array<vec4f, ${CLOTH_COARSE_FACTOR_WORKGROUP}>;
var<workgroup> cOk: u32;
var<workgroup> cGate: u32;
// Fixed-order tree sum of three records over the first n lanes (n a power of two).
fn coarseReduce(lane: u32, n: u32, a: vec4f, b: vec4f, c: vec4f) -> array<vec4f, 3> {
  cRa[lane] = a; cRb[lane] = b; cRc[lane] = c;
  workgroupBarrier();
  for (var k = n / 2u; k > 0u; k >>= 1u) {
    if (lane < k) { cRa[lane] += cRa[lane + k]; cRb[lane] += cRb[lane + k]; cRc[lane] += cRc[lane + k]; }
    workgroupBarrier();
  }
  return array<vec4f, 3>(cRa[0], cRb[0], cRc[0]);
}

// PSD part of the symmetric 2 x 2 [[a, b], [b, c]] as (a, b, c).
fn clampPSD2(a: f32, b: f32, c: f32) -> vec3f {
  let m = 0.5 * (a + c); let r = length(vec2f(0.5 * (a - c), b)); let l1 = m + r; let l2 = m - r;
  if (l2 >= 0.0) { return vec3f(a, b, c); }
  if (l1 <= 0.0) { return vec3f(0.0); }
  var u = vec2f(b, l1 - a); let u2 = vec2f(l1 - c, b);
  if (dot(u, u) < dot(u2, u2)) { u = u2; }
  let q = dot(u, u);
  if (!(q > 0.0)) { return vec3f(max(a, 0.0), 0.0, max(c, 0.0)); }
  return l1 * vec3f(u.x * u.x, u.x * u.y, u.y * u.y) / q;
}

fn tangentParts(f0: vec3f, f1: vec3f, d11: f32, d12: f32, d22: f32, g: f32) -> array<mat3x3f, 3> {
  let f00 = outer(f0, f0); let f11 = outer(f1, f1); let f01 = outer(f0, f1);
  return array<mat3x3f, 3>(d11 * f00 + g * f11, d22 * f11 + g * f00, d12 * f01 + g * transpose(f01));
}

// Per step, at the initial guess, one thread per face: the face's whole share of A_c as a quadratic form in the
// corner weights of the two nodes. With a = w . (-Dm00 - Dm01, Dm00, Dm01) and b = w . (-Dm10 - Dm11, Dm10, Dm11)
// (the node's dF0 and dF1 per unit move), block (c, c') = a a' Kaa + b b' Kbb + a b' Kab + b a' Kab^T
// + (sum_k s_k w_k w'_k) I: the membrane (Gauss-Newton tangent plus the PSD-clamped stress), face damping, the
// strain limits of the edges the face owns (corner differences are linear in a, b) and the inertia and pins of the
// corners it owns (s_k). Records: Kaa (6 words), Kbb (6), then Kab's columns with s_0, s_1, s_2.
@compute @workgroup_size(${workgroup}) fn coarseFactors(@builtin(global_invocation_id) id: vec3u) {
  let f = id.x;
  if (f >= P.n.y) { return; }
  let t = tris(f); let dm = rest[P.rest0.x + f]; let fr = rest[P.rest0.y + f];
  let c0 = rest[P.rest1.z + 2u * t.w]; let c1 = rest[P.rest1.z + 2u * t.w + 1u];
  let x0 = pos(t.x); let e1 = pos(t.y) - x0; let e2 = pos(t.z) - x0;
  let f0 = e1 * dm.x + e2 * dm.y; let f1 = e1 * dm.z + e2 * dm.w;
  let e11 = 0.5 * (dot(f0, f0) - 1.0); let e22 = 0.5 * (dot(f1, f1) - 1.0); let gam = dot(f0, f1);
  let w11 = select(0.0, (1.0 - c1.z) * c1.x, e11 < 0.0); let w22 = select(0.0, (1.0 - c1.z) * c1.y, e22 < 0.0);
  let S = clampPSD2(c0.x * e11 + c0.z * e22 - w11 * e11, c0.w * gam, c0.y * e22 + c0.z * e11 - w22 * e22);
  let m = tangentParts(f0, f1, c0.x - w11, c0.z, c0.y - w22, c0.w);
  var Kaa = fr.x * (m[0] + S.x * IDENTITY); var Kbb = fr.x * (m[1] + S.z * IDENTITY); var Kab = fr.x * (m[2] + S.y * IDENTITY);
  if (P.state1.z != 0u && fr.y > 0.0) {
    let T0 = state[P.state1.x + 2u * f]; let T1 = state[P.state1.x + 2u * f + 1u];
    let q = tangentParts(T0.xyz, T1.xyz, T0.w, c0.z, T1.w, c0.w); let k = fr.y * fr.x / hEff();
    Kaa += k * q[0]; Kbb += k * q[1]; Kab += k * q[2];
  }
  let own = topo[P.coarse2.z + 4u * f];
  let corner = array<u32, 3>(t.x, t.y, t.z);
  var sk = vec3f(0.0);
  for (var k = 0u; k < 3u; k++) {
    if ((own & (1u << k)) == 0u) { continue; }
    let v = corner[k]; let h = hEff();
    sk[k] = vertexRest(v).x / (h * h);
    if (vertexGroup(v) > 0u) { sk[k] += state[seg(S_PIND, v)].w; }
    sk[k] += grabSpring(v, pos(v)).w;${contact ? `
    // The vertex's contact Hessian block at the step's guess, read by every face sharing a node pair with it. A_c holds
    // it in f32 over the soft directions the correction is for, and past C_RANGE times the vertex's own diagonal s_k
    // neither A_c's f32 sums nor its sweep resolve those (the factor fails on a positive definite A_c: a barrier spike
    // reached 4e9). So the coarse model takes that vertex's contact scaled by a = C_RANGE s_k / trace (Hessian and force
    // alike, so a stiff term keeps its minimizer and the coarse step along it stays the Newton one); a is stored for
    // coarseGradientAt. Non-finite enters as the isotropic limit with no force. The objective, the Armijo test's change
    // of G and the fine solve keep the whole contact.
    var fc = vec3f(0.0); var Hc = mat3x3f();
    ogcAccumulateVertexContact(v, pos(v), &fc, &Hc);
    let held = C_RANGE * sk[k]; let tr = Hc[0].x + Hc[1].y + Hc[2].z;
    var a = 1.0;
    if (!(tr <= held)) {
      if (tr <= 3.0e38) { a = held / tr; Hc = a * Hc; } else { a = 0.0; Hc = (held / 3.0) * IDENTITY; }
    }
    c4Set(cContactH(v), vec4f(Hc[0], Hc[1].y)); c4Set(cContactH(v) + 1u, vec4f(Hc[1].z, Hc[2].z, a, 0.0));` : ''}
  }
  if (P.flags.z != 0u) {
    // Corner difference w_k - w_k1 = p a + q b (Dm's inverse maps (a, b) back to (w1 - w0, w2 - w0)).
    let det = dm.x * dm.w - dm.y * dm.z;
    let pq = array<vec2f, 3>(vec2f(-dm.w, dm.y) / det, vec2f(dm.w + dm.z, -dm.y - dm.x) / det, vec2f(-dm.z, dm.x) / det);
    for (var k = 0u; k < 3u; k++) {
      let e = topo[P.coarse2.z + 4u * f + 1u + k];
      if (e == C_NONE) { continue; }
      if (!(rest[P.rest0.w + e].y > 0.0)) { continue; }
      let ev = edges(e); let d = pos(ev.x) - pos(ev.y); let len = length(d);
      if (!(len > 0.0)) { continue; }
      let dual = edgeDual(e); let C = len - dual.z; let force = max(dual.y * C + dual.x, 0.0);
      var kEff = dual.y;
      if (!(dual.y * C + dual.x > 0.0)) { kEff = select(0.0, max(-dual.x / C, 0.0), C != 0.0); }
      let n = d / len;
      // The edge's canonical order may run against the face's; the block is even in the sign.
      let E = (kEff - force / len) * outer(n, n) + (force / len) * IDENTITY;
      let w = pq[k];
      Kaa += (w.x * w.x) * E; Kbb += (w.y * w.y) * E; Kab += (w.x * w.y) * E;
    }
  }
  let b = P.coarse3.y + 6u * f;
  c4Set(b, vec4f(Kaa[0].x, Kaa[1].x, Kaa[2].x, Kaa[1].y)); c4Set(b + 1u, vec4f(Kaa[2].y, Kaa[2].z, Kbb[0].x, Kbb[1].x));
  c4Set(b + 2u, vec4f(Kbb[2].x, Kbb[1].y, Kbb[2].y, Kbb[2].z));
  c4Set(b + 3u, vec4f(Kab[0], sk.x)); c4Set(b + 4u, vec4f(Kab[1], sk.y)); c4Set(b + 5u, vec4f(Kab[2], sk.z));
}

fn sym3(xx: f32, xy: f32, xz: f32, yy: f32, yz: f32, zz: f32) -> mat3x3f { return mat3x3f(xx, xy, xz, xy, yy, yz, xz, yz, zz); }
fn cContactH(v: u32) -> u32 { return P.coarse6.w + 2u * v; }

// One face's share of block (c, c') of A_c; wc, wd are the face's corner weights of nodes c and c'.
fn faceCoarseBlock(f: u32, wc: vec3f, wd: vec3f) -> mat3x3f {
  let dm = rest[P.rest0.x + f];
  let av = vec3f(-dm.x - dm.y, dm.x, dm.y); let bv = vec3f(-dm.z - dm.w, dm.z, dm.w);
  let ac = dot(wc, av); let bc = dot(wc, bv); let ad = dot(wd, av); let bd = dot(wd, bv);
  let b = P.coarse3.y + 6u * f;
  let r0 = c4(b); let r1 = c4(b + 1u); let r2 = c4(b + 2u); let r3 = c4(b + 3u); let r4 = c4(b + 4u); let r5 = c4(b + 5u);
  let Kab = mat3x3f(r3.xyz, r4.xyz, r5.xyz);
  var M = (ac * ad) * sym3(r0.x, r0.y, r0.z, r0.w, r1.x, r1.y) + (bc * bd) * sym3(r1.z, r1.w, r2.x, r2.y, r2.z, r2.w)
    + (ac * bd) * Kab + (bc * ad) * transpose(Kab) + dot(wc * wd, vec3f(r3.w, r4.w, r5.w)) * IDENTITY;${contact ? `
  // Contact: the owned corners' own Hessian blocks (the static solid's exactly; a pair's diagonal blocks).
  let own = topo[P.coarse2.z + 4u * f] & 7u;
  if (own != 0u) {
    let t = tris(f); let corner = array<u32, 3>(t.x, t.y, t.z); let ww = wc * wd;
    for (var k = 0u; k < 3u; k++) {
      if ((own & (1u << k)) == 0u || ww[k] == 0.0) { continue; }
      let h0 = c4(cContactH(corner[k])); let h1 = c4(cContactH(corner[k]) + 1u);
      M += ww[k] * sym3(h0.x, h0.y, h0.z, h0.w, h1.x, h1.y);
    }
  }` : ''}
  return M;
}

// One workgroup per coarse pair (c <= c'): A[c, c'] and its transpose.
@compute @workgroup_size(${CLOTH_COARSE_WORKGROUP}) fn coarseAssemble(@builtin(workgroup_id) wg: vec3u, @builtin(num_workgroups) nwg: vec3u, @builtin(local_invocation_index) lane: u32) {
  let p = foldedGroup(wg, nwg);
  if (p >= P.coarse0.z) { return; }
  let c = topo[P.coarse1.w + 2u * p]; let d = topo[P.coarse1.w + 2u * p + 1u];
  var M = mat3x3f();
  let mask = (1u << C_SLOT_BITS) - 1u;
  for (var j = topo[P.coarse2.x + p] + lane; j < topo[P.coarse2.x + p + 1u]; j += CW) {
    let entry = topo[P.coarse2.y + j]; let f = entry >> (2u * C_SLOT_BITS);
    let base = P.coarse3.x + topo[P.coarse1.z + f];
    let rc = rest[base + ((entry >> C_SLOT_BITS) & mask)]; let rd = rest[base + (entry & mask)];
    M += faceCoarseBlock(f, rc.xyz, rd.xyz);
  }
  let s = coarseReduce(lane, CW, vec4f(M[0], 0.0), vec4f(M[1], 0.0), vec4f(M[2], 0.0));
  if (lane == 0u) {
    let m = P.coarse0.y; let cols = array<vec3f, 3>(s[0].xyz, s[1].xyz, s[2].xyz);
    for (var col = 0u; col < 3u; col++) {
      for (var row = 0u; row < 3u; row++) {
        let x = cols[col][row];
        cfSet(cA() + (3u * c + row) * m + 3u * d + col, x);
        cfSet(cA() + (3u * d + col) * m + 3u * c + row, x);
      }
    }
  }
}

// One workgroup: A^-1 into W1 by the symmetric sweep (Gauss-Jordan without pivoting, SPD), one node's 3 x 3 block
// a pass. Sweeping block K with D = A_KK: A_KK <- -D^-1, A_iK <- A_iK D^-1, A_ij <- A_ij - A_iK D^-1 A_Kj; after every
// block the matrix holds -A^-1. Each lane keeps its packed lower-triangle entries in registers; only the block's three
// columns pass through workgroup memory, written for the next block by the owners of its entries, so a block is one
// pass and one barrier. A block that is not positive definite (only round-off can make the PSD element blocks so;
// coarseFactors holds contact inside the range f32 resolves) skips the step's corrections and raises the status bit.
const CM = ${3 * nodes}u;
const CPAIRS = ${Math.ceil(3 * nodes / 2)}u;
const CSW = ${CLOTH_COARSE_SWEEP_WORKGROUP}u;
// Columns K0 + m of the current matrix (m < 3), then the next block's.
var<workgroup> cCol: array<f32, ${6 * 3 * nodes}>;
fn colAt(buffer: u32, m: u32, row: u32) -> f32 { return cCol[(buffer * 3u + m) * CM + row]; }
// Lane ownership of the packed entries: each 128 lanes take a row pair (p, CM - 1 - p) a round, CM + 1 entries at most
// (CM < 128).
fn sweepEntry(lane: u32, r: u32) -> vec2u {
  let p = (CSW / 128u) * r + lane / 128u; let l = lane % 128u;
  if (p >= CPAIRS) { return vec2u(C_NONE); }
  let a = p; let b = CM - 1u - p;
  if (l <= a) { return vec2u(a, l); }
  let j = l - a - 1u;
  if (b != a && j <= b) { return vec2u(b, j); }
  return vec2u(C_NONE);
}
fn sweepColumns(buffer: u32, row: u32) -> vec3f { return vec3f(colAt(buffer, 0u, row), colAt(buffer, 1u, row), colAt(buffer, 2u, row)); }
fn sweepSave(buffer: u32, i: u32, j: u32, k1: u32, x: f32) {
  if (i >= k1 && i < k1 + 3u) { cCol[(buffer * 3u + i - k1) * CM + j] = x; }
  if (j >= k1 && j < k1 + 3u) { cCol[(buffer * 3u + j - k1) * CM + i] = x; }
}
@compute @workgroup_size(${CLOTH_COARSE_SWEEP_WORKGROUP}) fn coarseFactor(@builtin(local_invocation_index) lane: u32) {
  let A = cA(); let W1 = cW1();
  if (lane == 0u) { cOk = 1u; }
${rounds.map(r => `  let e${r} = sweepEntry(lane, ${r}u); var x${r} = 0.0;
  if (e${r}.x != C_NONE) { x${r} = cf(A + e${r}.x * CM + e${r}.y); sweepSave(0u, e${r}.x, e${r}.y, 0u, x${r}); }`).join('\n')}
  workgroupBarrier();
  for (var c = 0u; c < CM / 3u; c++) {
    let cur = c & 1u; let nxt = 1u - cur; let k0 = 3u * c; let k1 = k0 + 3u;
    // D^-1 = L^-T L^-1 by D's Cholesky factor L, which exists iff D is positive definite. Not by the adjugate: a
    // contact Hessian makes a block one stiff direction k over soft ones s, and its determinant s^2 k cancels from
    // terms of k^3 (f32 loses it at k / s ~ 1e3.5, where the pivots' s survives to k / s ~ 1e7).
    let d0 = sweepColumns(cur, k0); let d1 = sweepColumns(cur, k0 + 1u); let d2 = sweepColumns(cur, k0 + 2u);
    let l00 = sqrt(max(d0.x, 0.0)); let l10 = d0.y / l00; let l20 = d0.z / l00;
    let t11 = d1.y - l10 * l10; let l11 = sqrt(max(t11, 0.0)); let l21 = (d1.z - l20 * l10) / l11;
    let t22 = d2.z - l20 * l20 - l21 * l21; let l22 = sqrt(max(t22, 0.0));
    let good = d0.x > 0.0 && t11 > 0.0 && t22 > 0.0 && d0.x <= 3.0e38 && d1.y <= 3.0e38 && d2.z <= 3.0e38;
    if (!good && lane == 0u) { cOk = 0u; }
    let i00 = 1.0 / select(1.0, l00, good); let i11 = 1.0 / select(1.0, l11, good); let i22 = 1.0 / select(1.0, l22, good);
    let i10 = select(0.0, -l10 * i00 * i11, good); let i21 = select(0.0, -l21 * i11 * i22, good); let i20 = select(0.0, -(l20 * i00 + l21 * i10) * i22, good);
    let Linv = mat3x3f(vec3f(i00, i10, i20), vec3f(0.0, i11, i21), vec3f(0.0, 0.0, i22));
    let Di = transpose(Linv) * Linv;
${rounds.map(r => `    if (e${r}.x != C_NONE) {
      let i = e${r}.x; let j = e${r}.y; let iK = i >= k0 && i < k1; let jK = j >= k0 && j < k1;
      let ci = sweepColumns(cur, i); let cj = sweepColumns(cur, j);
      var y = x${r} - dot(ci, Di * cj);
      if (iK) { y = (Di * cj)[i - k0]; }
      if (jK) { y = (Di * ci)[j - k0]; }
      if (iK && jK) { y = -Di[j - k0][i - k0]; }
      x${r} = y;
      sweepSave(nxt, i, j, k1, y);
    }`).join('\n')}
    workgroupBarrier();
  }
  let ok = workgroupUniformLoad(&cOk);
  if (lane == 0u) {
    cfSet(cMisc() * 4u + ${CLOTH_COARSE_MISC.ok}u, f32(ok));
    if (ok == 0u) { atomicOr(&status[0], ${CLOTH_STATUS_COARSE_INDEFINITE}u); }
  }
${rounds.map(r => `  if (e${r}.x != C_NONE) { cfSet(W1 + e${r}.x * CM + e${r}.y, -x${r}); cfSet(W1 + e${r}.y * CM + e${r}.x, -x${r}); }`).join('\n')}
}

// The gradient of G at the iterate (every term but contact), the coarse step's right-hand side, in two passes:
// each element's corner gradients once (thread i: face i, hinge i, edge i), then each vertex gathers its own.
fn cFaceGrad(f: u32, k: u32) -> u32 { return P.coarse6.x + 3u * f + k; }
fn cHingeGrad(h: u32, k: u32) -> u32 { return P.coarse6.y + 4u * h + k; }
fn cEdgeGrad(e: u32) -> u32 { return P.coarse6.z + e; }
@compute @workgroup_size(${workgroup}) fn coarseElements(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (!coarseLive()) { return; }
  if (i < P.n.y) {
    let t = tris(i); let dm = rest[P.rest0.x + i]; let fr = rest[P.rest0.y + i];
    let c0 = rest[P.rest1.z + 2u * t.w]; let c1 = rest[P.rest1.z + 2u * t.w + 1u];
    let x0 = pos(t.x); let x1 = pos(t.y); let x2 = pos(t.z);
    let e1 = x1 - x0; let e2 = x2 - x0;
    let f0 = e1 * dm.x + e2 * dm.y; let f1 = e1 * dm.z + e2 * dm.w;
    let e11 = 0.5 * (dot(f0, f0) - 1.0); let e22 = 0.5 * (dot(f1, f1) - 1.0); let gam = dot(f0, f1);
    let w11 = select(0.0, (1.0 - c1.z) * c1.x, e11 < 0.0); let w22 = select(0.0, (1.0 - c1.z) * c1.y, e22 < 0.0);
    let s11 = c0.x * e11 + c0.z * e22 - w11 * e11; let s22 = c0.y * e22 + c0.z * e11 - w22 * e22; let tau = c0.w * gam;
    // dE/dF0 and dE/dF1 (area-weighted, membrane plus damping); corner k's gradient is a_k dE/dF0 + b_k dE/dF1.
    var p0 = fr.x * (s11 * f0 + tau * f1); var p1 = fr.x * (s22 * f1 + tau * f0);
    let av = vec3f(-dm.x - dm.y, dm.x, dm.y); let bv = vec3f(-dm.z - dm.w, dm.z, dm.w);
    if (fr.y > 0.0) {
      let T0 = state[P.state1.x + 2u * i]; let T1 = state[P.state1.x + 2u * i + 1u];
      let u0 = x0 - state[seg(S_XHAT, t.x)].xyz; let u1 = x1 - state[seg(S_XHAT, t.y)].xyz; let u2 = x2 - state[seg(S_XHAT, t.z)].xyz;
      let r0 = av.x * u0 + av.y * u1 + av.z * u2; let r1 = bv.x * u0 + bv.y * u1 + bv.z * u2;
      let rate = vec3f(dot(T0.xyz, r0), dot(T1.xyz, r1), dot(T0.xyz, r1) + dot(T1.xyz, r0));
      let sg = (fr.y * fr.x / hEff()) * vec3f(T0.w * rate.x + c0.z * rate.y, c0.z * rate.x + T1.w * rate.y, c0.w * rate.z);
      p0 += sg.x * T0.xyz + sg.z * T1.xyz; p1 += sg.y * T1.xyz + sg.z * T0.xyz;
    }
    for (var k = 0u; k < 3u; k++) { c4Set(cFaceGrad(i, k), vec4f(av[k] * p0 + bv[k] * p1, 0.0)); }
  }
  if (i < P.n.w) {
    // Quadratic bending and its damping: corner k's gradient w_k (k s + (kd k / h~) u).
    let q = loadHinge(i, 0u, pos(hingeVerts(i).x)); let w = hingeW(i);
    let c = q.kd.x * q.s + (q.kd.x * q.kd.y / hEff()) * q.u;
    var g = array<vec3f, 4>(hingeWeight(w, 0u) * c, hingeWeight(w, 1u) * c, hingeWeight(w, 2u) * c, hingeWeight(w, 3u) * c);
    for (var k = 0u; k < 4u; k++) { c4Set(cHingeGrad(i, k), vec4f(g[k], 0.0)); }
  }
  if (P.flags.z != 0u && i < P.n.z) {
    var g = vec3f(0.0);
    if (rest[P.rest0.w + i].y > 0.0) {
      let ev = edges(i); let d = pos(ev.x) - pos(ev.y); let len = length(d);
      if (len > 0.0) { let dual = edgeDual(i); g = max(dual.y * (len - dual.z) + dual.x, 0.0) * d / len; }
    }
    c4Set(cEdgeGrad(i), vec4f(g, 0.0));
  }
}
fn coarseGradientAt(v: u32, xv: vec3f) -> vec3f {
  let h = hEff(); let w = vertexRest(v).x / (h * h);
  var g = w * (xv - state[seg(S_Y, v)].xyz);
  for (var j = topo[P.topo0.x + v]; j < topo[P.topo0.x + v + 1u]; j++) { let e = topo[P.topo0.y + j]; g += c4(cFaceGrad(e >> 2u, e & 3u)).xyz; }
  for (var j = topo[P.topo0.z + v]; j < topo[P.topo0.z + v + 1u]; j++) { let e = topo[P.topo0.w + j]; g += c4(cHingeGrad(e >> 2u, e & 3u)).xyz; }
  if (P.flags.z != 0u) {
    for (var j = topo[P.topo1.x + v]; j < topo[P.topo1.x + v + 1u]; j++) {
      let e = topo[P.topo1.y + j]; let f = c4(cEdgeGrad(e >> 2u)).xyz;
      g += select(f, -f, (e & 3u) == 1u);
    }
  }
  if (vertexGroup(v) > 0u) { let p = state[seg(S_PIND, v)]; g += p.w * (xv - state[seg(S_PINC, v)].xyz) + p.xyz; }
  g += grabSpring(v, xv).xyz;${contact ? `
  var fc = vec3f(0.0); var Hc = mat3x3f();
  ogcAccumulateVertexContact(v, xv, &fc, &Hc);
  // Scaled as coarseFactors scaled the vertex's block into A_c (the coarse model's contact, see there).
  let a = c4(cContactH(v) + 1u).z;
  if (a > 0.0) { g -= a * fc; }` : ''}
  return g;
}
fn coarseLive() -> bool { return cf(cMisc() * 4u + ${CLOTH_COARSE_MISC.ok}u) != 0.0; }

@compute @workgroup_size(${workgroup}) fn coarseGradient(@builtin(global_invocation_id) id: vec3u) {
  let v = id.x;
  if (v >= P.n.x || !coarseLive()) { return; }
  c4Set(P.coarse4.x + v, vec4f(coarseGradientAt(v, pos(v)), 0.0));
}

// One workgroup per node c: r_c = -sum_v P_vc g_v over the node's owner list.
@compute @workgroup_size(${CLOTH_COARSE_WORKGROUP}) fn coarseRestrict(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) lane: u32) {
  let c = wg.x;
  if (c >= P.coarse0.x) { return; }
  var r = vec3f(0.0);
  var end = topo[P.coarse1.x + c + 1u];
  if (!coarseLive()) { end = 0u; }
  for (var j = topo[P.coarse1.x + c] + lane; j < end; j += CW) {
    let e = topo[P.coarse1.y + j]; let v = e >> 3u;
    r -= rowWeight(v, e & 7u) * c4(P.coarse4.x + v).xyz;
  }
  let s = coarseReduce(lane, CW, vec4f(r, 0.0), vec4f(0.0), vec4f(0.0));
  if (lane == 0u) { for (var k = 0u; k < 3u; k++) { cfSet(cR() + 3u * c + k, s[0][k]); } }
}

// One workgroup: dc = A_c^-1 r and the slope g . d = -r . dc.
@compute @workgroup_size(${CLOTH_COARSE_FACTOR_WORKGROUP}) fn coarseSolve(@builtin(local_invocation_index) lane: u32) {
  let m = select(0u, P.coarse0.y, coarseLive());
  var slope = 0.0;
  for (var i = lane; i < m; i += CFW) {
    var s = 0.0;
    for (var j = 0u; j < m; j++) { s += cf(cW1() + i * m + j) * cf(cR() + j); }
    cfSet(cDc() + i, s); slope -= s * cf(cR() + i);
  }
  let t = coarseReduce(lane, CFW, vec4f(slope, 0.0, 0.0, 0.0), vec4f(0.0), vec4f(0.0));
  if (lane == 0u) { cfSet(cMisc() * 4u + ${CLOTH_COARSE_MISC.slope}u, t[0].x); }
}

@compute @workgroup_size(${workgroup}) fn coarseProlong(@builtin(global_invocation_id) id: vec3u) {
  let v = id.x;
  if (v >= P.n.x || !coarseLive()) { return; }
  var d = vec3f(0.0);
  for (var s = 0u; s < C_ENTRIES; s++) {
    let c = rowNode(v, s);
    if (c == C_NONE) { break; }
    d += rowWeight(v, s) * vec3f(cf(cDc() + 3u * c), cf(cDc() + 3u * c + 1u), cf(cDc() + 3u * c + 2u));
  }
  c4Set(P.coarse4.y + v, vec4f(d, 0.0));
}

fn cdir(v: u32) -> vec3f { return c4(P.coarse4.y + v).xyz; }

// Difference-form changes with every corner moving (the vertex versions move one corner).
fn membraneDeltaAll(f: u32, d0: vec3f, d1: vec3f, d2: vec3f) -> vec2f {
  let t = tris(f);
  let x0 = pos(t.x); let x1 = pos(t.y); let x2 = pos(t.z);
  let dm = rest[P.rest0.x + f]; let area = rest[P.rest0.y + f].x;
  let c0 = rest[P.rest1.z + 2u * t.w]; let c1 = rest[P.rest1.z + 2u * t.w + 1u];
  let e1 = x1 - x0; let e2 = x2 - x0;
  let f0 = e1 * dm.x + e2 * dm.y; let f1 = e1 * dm.z + e2 * dm.w;
  let u1 = d1 - d0; let u2 = d2 - d0;
  let df0 = u1 * dm.x + u2 * dm.y; let df1 = u1 * dm.z + u2 * dm.w;
  let e11 = 0.5 * (dot(f0, f0) - 1.0); let e22 = 0.5 * (dot(f1, f1) - 1.0); let gam = dot(f0, f1);
  let d11 = dot(f0, df0) + 0.5 * dot(df0, df0); let d22 = dot(f1, df1) + 0.5 * dot(df1, df1);
  let dg = dot(f0, df1) + dot(df0, f1) + dot(df0, df1);
  let s11 = (1.0 - c1.z) * c1.x; let s22 = (1.0 - c1.z) * c1.y;
  let q11 = quadDelta(c0.x - select(0.0, s11, e11 < 0.0), c0.x - select(0.0, s11, e11 + d11 < 0.0), e11, d11);
  let q22 = quadDelta(c0.y - select(0.0, s22, e22 < 0.0), c0.y - select(0.0, s22, e22 + d22 < 0.0), e22, d22);
  let q12 = c0.z * (e11 * d22 + d11 * e22 + d11 * d22); let qs = c0.w * (gam * dg + 0.5 * dg * dg);
  let rounding = (c0.x + c0.z) * abs(d11) * (0.5 + e11) + (c0.y + c0.z) * abs(d22) * (0.5 + e22) + c0.w * abs(dg) * sqrt((1.0 + 2.0 * e11) * (1.0 + 2.0 * e22));
  return area * vec2f(q11 + q22 + q12 + qs, abs(q11) + abs(q22) + abs(q12) + abs(qs) + rounding);
}
fn faceDampingDeltaAll(f: u32, d0: vec3f, d1: vec3f, d2: vec3f) -> vec2f {
  let fr = rest[P.rest0.y + f];
  if (!(fr.y > 0.0)) { return vec2f(0.0); }
  let t = tris(f); let dm = rest[P.rest0.x + f]; let c0 = rest[P.rest1.z + 2u * t.w];
  let T0 = state[P.state1.x + 2u * f]; let T1 = state[P.state1.x + 2u * f + 1u];
  let f0 = T0.xyz; let f1 = T1.xyz;
  let a = vec3f(-dm.x - dm.y, dm.x, dm.y); let b = vec3f(-dm.z - dm.w, dm.z, dm.w);
  let u0 = pos(t.x) - state[seg(S_XHAT, t.x)].xyz; let u1 = pos(t.y) - state[seg(S_XHAT, t.y)].xyz; let u2 = pos(t.z) - state[seg(S_XHAT, t.z)].xyz;
  let r0 = a.x * u0 + a.y * u1 + a.z * u2; let r1 = b.x * u0 + b.y * u1 + b.z * u2;
  let rate = vec3f(dot(f0, r0), dot(f1, r1), dot(f0, r1) + dot(f1, r0));
  let q0 = a.x * d0 + a.y * d1 + a.z * d2; let q1 = b.x * d0 + b.y * d1 + b.z * d2;
  let dr = vec3f(dot(f0, q0), dot(f1, q1), dot(f0, q1) + dot(f1, q0));
  let sigma = vec3f(T0.w * rate.x + c0.z * rate.y, c0.z * rate.x + T1.w * rate.y, c0.w * rate.z);
  let dsigma = vec3f(T0.w * dr.x + c0.z * dr.y, c0.z * dr.x + T1.w * dr.y, c0.w * dr.z);
  let scale = fr.y * fr.x / hEff();
  let lin = scale * dot(dr, sigma); let quad = 0.5 * scale * dot(dr, dsigma);
  let ru = vec3f(abs(a.x) * length(u0) + abs(a.y) * length(u1) + abs(a.z) * length(u2), abs(b.x) * length(u0) + abs(b.y) * length(u1) + abs(b.z) * length(u2), 0.0);
  let roundRate = vec3f(length(f0) * ru.x, length(f1) * ru.y, length(f0) * ru.y + length(f1) * ru.x);
  return vec2f(lin + quad, abs(lin) + quad + scale * dot(abs(dsigma), roundRate));
}
// Bending and its damping when the hinge's vertices move by d0..d3, exactly (both are quadratic in s and u).
fn hingeDeltaAll(h: u32, d0: vec3f, d1: vec3f, d2: vec3f, d3: vec3f) -> vec2f {
  let q = loadHinge(h, 0u, pos(hingeVerts(h).x)); let ds = hingeS(hingeW(h), d0, d1, d2, d3);
  let scale = q.kd.x * q.kd.y / hEff();
  let lin = q.kd.x * dot(q.s, ds) + scale * dot(q.u, ds); let quad = 0.5 * (q.kd.x + scale) * dot(ds, ds);
  return vec2f(lin + quad, abs(lin) + quad + (q.kd.x * q.sr + scale * q.ur) * length(ds));
}
fn edgeDeltaAll(e: u32, d: vec3f) -> vec2f {
  let er = rest[P.rest0.w + e];
  if (!(er.y > 0.0)) { return vec2f(0.0); }
  let ev = edges(e); let dual = edgeDual(e);
  let d0 = pos(ev.x) - pos(ev.y); let l0 = length(d0); let l1 = length(d0 + d);
  if (!(l0 > 0.0 && l1 > 0.0)) {
    let e0 = select(0.0, augmented(dual.y, dual.x, l0 - dual.z, true), l0 > 0.0); let e1 = select(0.0, augmented(dual.y, dual.x, l1 - dual.z, true), l1 > 0.0);
    return vec2f(e1 - e0, abs(e0) + abs(e1));
  }
  let C = l0 - dual.z; let dC = (2.0 * dot(d0, d) + dot(d, d)) / (l0 + l1);
  let f0 = max(dual.y * C + dual.x, 0.0); let f1 = max(dual.y * (C + dC) + dual.x, 0.0);
  var de = 0.5 * dC * (f0 + f1);
  if (!(f0 > 0.0 && f1 > 0.0)) { de = (f1 - f0) * (f1 + f0) / (2.0 * dual.y); }
  return vec2f(de, abs(de) + dual.y * l0 * abs(dC));
}

${contact ? `// Vertex v's contact entries with v alone moved by d: the static solid's change exactly, a pair's to first order
// (each pair is summed once per vertex holding it). Truncation keeps the move safe; this keeps it descending.
// A trial entering a barrier's domain is infinite: it fails Armijo outright, never through an infinite floor.
fn coarseContactDelta(v: u32, xv: vec3f, ec0: f32, d: vec3f) -> vec2f {
  let ec1 = ogcVertexContactEnergy(v, xv + d);
  if (!(ec1 <= 3.0e38)) { return vec2f(ec1, 0.0); }
  return vec2f(ec1 - ec0, abs(ec0) + abs(ec1));
}
` : ''}// Element passes: (dG, floor) at trial steps; thread i takes vertex i, face i, hinge i and edge i. The full step
// first (nearly every correction takes it), then, only for a correction that rejected it, the shorter trials 1/2,
// 1/4, ... in the Back pass: the energy evaluations are the trial pass's cost, so the first pass evaluates one.
${[['coarseTrial', 0, 1], ['coarseTrialBack', 1, CLOTH_COARSE.trials - 1]].map(([name, first, count]) => `@compute @workgroup_size(${workgroup}) fn ${name}(@builtin(global_invocation_id) id: vec3u, @builtin(local_invocation_index) lane: u32, @builtin(workgroup_id) wg: vec3u) {
  let i = id.x;
  var dE = vec4f(0.0); var fl = vec4f(0.0);
  if (lane == 0u) { cGate = u32(coarseLive() && ${first === 0 ? `cf(cMisc() * 4u + ${CLOTH_COARSE_MISC.slope}u) < 0.0` : `cf(cMisc() * 4u + ${CLOTH_COARSE_MISC.alpha}u) < 0.0`}); }
  let live = workgroupUniformLoad(&cGate) != 0u;
  ${first === 0 ? '' : 'if (!live) { return; }'}
  if (live) {
    var alpha = ${(2 ** -(first as number)).toFixed(1)};${contact ? `
    var ec0 = 0.0; if (i < P.n.x) { ec0 = ogcVertexContactEnergy(i, pos(i)); }` : ''}
    for (var k = 0u; k < ${count}u; k++) {
      var q = vec2f(0.0);
      if (i < P.n.x) { q += ownDelta(i, pos(i), alpha * cdir(i));${contact ? ' q += coarseContactDelta(i, pos(i), ec0, alpha * cdir(i));' : ''} }
      if (i < P.n.y) {
        let t = tris(i); let a0 = alpha * cdir(t.x); let a1 = alpha * cdir(t.y); let a2 = alpha * cdir(t.z);
        q += membraneDeltaAll(i, a0, a1, a2) + faceDampingDeltaAll(i, a0, a1, a2);
      }
      if (i < P.n.w) {
        let hv = hingeVerts(i); let a0 = alpha * cdir(hv.x); let a1 = alpha * cdir(hv.y); let a2 = alpha * cdir(hv.z); let a3 = alpha * cdir(hv.w);
        q += hingeDeltaAll(i, a0, a1, a2, a3);
      }
      if (P.flags.z != 0u && i < P.n.z) { let ev = edges(i); q += edgeDeltaAll(i, alpha * (cdir(ev.x) - cdir(ev.y))); }
      dE[k] = q.x; fl[k] = q.y;
      alpha *= 0.5;
    }
  }
  let s = coarseReduce(lane, ${workgroup}u, dE, fl, vec4f(0.0));
  if (lane == 0u) { c4Set(P.coarse4.z + 2u * wg.x, s[0]); c4Set(P.coarse4.z + 2u * wg.x + 1u, s[1]); }
}
`).join('\n')}
// One workgroup: the largest trial step passing Armijo with the summation error bound, or none. coarseChoose decides
// the full step: alpha 1, or -1 (pending) for coarseTrialBack and coarseChooseBack to try the shorter ones.
${(['coarseChoose', 'coarseChooseBack'] as const).map(name => { const back = name === 'coarseChooseBack'; return `@compute @workgroup_size(${CLOTH_COARSE_FACTOR_WORKGROUP}) fn ${name}(@builtin(local_invocation_index) lane: u32) {
  let slope = cf(cMisc() * 4u + ${CLOTH_COARSE_MISC.slope}u);
  if (lane == 0u) { cGate = u32(coarseLive() && ${back ? `cf(cMisc() * 4u + ${CLOTH_COARSE_MISC.alpha}u) < 0.0` : 'slope < 0.0'}); }
  let live = workgroupUniformLoad(&cGate) != 0u;
  ${back ? 'if (!live) { return; }' : ''}
  var dE = vec4f(0.0); var fl = vec4f(0.0);
  if (live) { for (var j = lane; j < P.coarse5.z; j += CFW) { dE += c4(P.coarse4.z + 2u * j); fl += c4(P.coarse4.z + 2u * j + 1u); } }
  let s = coarseReduce(lane, CFW, dE, fl, vec4f(0.0));
  if (lane == 0u) {
    var chosen = ${back ? '0.0' : '-1.0'}; var alpha = ${back ? '0.5' : '1.0'}; var which = ${back ? '4u' : '0u'};
    if (live) {
      ${back ? `for (var k = 0u; k < ${CLOTH_COARSE.trials - 1}u; k++) {
        if (s[0][k] <= ARMIJO * alpha * slope + 24.0 * EPS32 * s[1][k]) { chosen = alpha; which = k + 1u; break; }
        alpha *= 0.5;
      }` : `if (s[0][0] <= ARMIJO * slope + 24.0 * EPS32 * s[1][0]) { chosen = 1.0; }`}
    } else { chosen = 0.0; }
    cfSet(cMisc() * 4u + ${CLOTH_COARSE_MISC.alpha}u, chosen);
    let st = cMisc() + 1u;
    var a = c4(st); var b = c4(st + 1u);
    ${back ? 'if (which < 3u) { a[which + 1u] += 1.0; } else if (which == 3u) { b.x += 1.0; }' : 'if (coarseLive()) { a.x += 1.0; }\n    if (chosen > 0.0) { a.y += 1.0; }'}
    c4Set(st, a); c4Set(st + 1u, b);
  }
}
`; }).join('\n')}
@compute @workgroup_size(${workgroup}) fn coarseApply(@builtin(global_invocation_id) id: vec3u) {
  let v = id.x;
  if (v >= P.n.x) { return; }
  let alpha = cf(cMisc() * 4u + ${CLOTH_COARSE_MISC.alpha}u);
  if (!(alpha > 0.0) || !coarseLive()) { return; }
  positions[v] = vec4f(pos(v) + alpha * cdir(v), positions[v].w);
}
`;
}
