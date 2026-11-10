/**
 * WGSL of the Newton-PCG cloth step (appended to the solver module, which supplies the element helpers). One implicit
 * step per call, no substeps: Newton iterations on the step's objective G (BDF2 inertia, membrane, Bergou bending,
 * frozen-tangent damping, grab spring), each solving H d = -g by preconditioned conjugate gradients and taking the
 * largest Armijo step of 1, 1/2, ..., 1/128 on G. With contact the trial points are the projected arc x(alpha)_v =
 * Pi_v(x_v + alpha d_v), Pi_v onto vertex v's certified set (nwProject), with the slope g.(x(alpha) - x) of each trial,
 * and the direction is projected Newton's (Bertsekas): each vertex's active set is the outward normals of the
 * certificates binding it at the iterate (up to three, from the latest filter's set: ogcProjectActive at the accepted
 * trial, rebuilt every iteration, never carried across certificate sets) less those whose KKT multipliers are not
 * positive, and H d = -g is solved in the reduced space, P H P d = -P g with P_v the projection off v's active normals
 * (nwGradient, pcgStep, pcgSolve). Pins are Dirichlet DOFs held at their frame targets (their rows and
 * directions are zero). Everything is decided on the GPU: convergence and stalls set control flags that every later
 * kernel of the step reads, so the CPU encodes fixed caps and never reads back.
 *
 * H is assembled once per Newton iteration (nwGradient) over the compiled topology (gpu-solver.ts,
 * compileClothPCGPatches): per face the Green-strain tangent J D J^T plus the geometric term S (x) I with the 2x2
 * stress S clamped PSD (nwFaces records f0, f1 and S), the frozen-tangent damping, the active strain-limit edges and
 * the bending stencil, as 3x3 blocks in sliced-ELL order (patches of 256 rows; per row K1 blocks, slot 0 the diagonal,
 * and K2 scalar bending entries beyond the 1-ring). Contact (C2) stays matrix-free, a separate additive stage of the
 * product. The preconditioner is the inverse of each vertex's 3x3 diagonal block (vertexSystem, contact included).
 *
 * PCG is the single-reduction (Chronopoulos-Gear) form: one dispatch per iteration (pcgStep, one 256-thread
 * workgroup per patch). Dispatch i forms s, r, u = M^-1 r of iteration i for its rows and its halo in workgroup
 * memory, advances d, multiplies w = H u and writes the partials (r.u, w.u) that dispatch i + 1 reduces into alpha and
 * beta. r, w and s ping-pong so a dispatch reads the previous iteration's values while it writes its own.
 *
 * The `nw` buffer (vec4 records): per-vertex segments (stride Np) g, r (two), w (two), s (two), d, p, M^-1 (two
 * records), with contact the trial moves (one per trial), the active normals (three) and the projector P (two records:
 * xx, xy, xz, yy; yz, zz, the changes, the active count) and H_vv^-1 before the projection (two records), then 3 records per face, the assembled blocks and bending
 * entries, the line-search partials (W = workgroups of a per-vertex dispatch each, 10 arrays), the PCG partials (3 arrays of one record per patch), the PCG scalars
 * (gamma_0 and the solve's stop tolerance^2, then alpha by iteration mod 2), then the control record (Newton done, PCG
 * done, accepted step, Newton iteration), then |g|^2 at the first and latest Newton iterate, the latest PCG stop
 * tolerance^2 (z, the forcing's eta^2) and the step's PCG iterations (w), then with contact the latest iteration's
 * active set (control + 2: vertices with an active normal, normals kept, normals released by the KKT sign, and the
 * vertices a travel bound holds at the accepted trial - the cap-aware done rule, read by the contact hook's re-detection),
 * then the trust region (control + 3 to control + 6; nwTrust below).
 *
 * newton.trustRegion 'hybrid' (trustRegion): the line search above until two consecutive iterations of a step accept the
 * full step (with contact, each moving no vertex more than r / 2), then trust-region iterations for the rest of the
 * step (Steihaug-Toint): the model is H with the membrane's stress term unclamped (nwFaces; the exact Hessian of all
 * but contact, ladder 1c, C1 RESULT 41; at BDF2 the oracle's gain over the line search is modest, kc 9668 t15 14 -> 9
 * and t25 43 -> 38, C1 RESULT 41 correcting the backward-Euler counts of RESULT 40), contact's terms and the preconditioner
 * unchanged; PCG stops on the boundary ||d||_M = Delta or at negative curvature (Steihaug's step to the boundary along p),
 * the norms by the CG recurrences (M the preconditioner's inverse, in the reduced space); the trial is the projected arc
 * at alpha 1 (s, nwProject's trial 0), the model's decrease pred = -(g.s + s.H s / 2) with s.H s one more product
 * (nwModel), accepted when the decrease is at least 1e-4 pred less the energy change's rounding floor, and Delta set from
 * rho = ared / pred (rho < 0.25: Delta = ||d||_M / 4; rho > 0.75 on the boundary: 2 Delta; no update while the change is
 * within rounding). Delta starts at ||d||_M of the line search's last (full) step.
 */
import { CLOTH_LINE_SEARCH } from './oracle';

export const CLOTH_NEWTON_SEGMENTS = { g: 0, r0: 1, r1: 2, w0: 3, w1: 4, s0: 5, s1: 6, d: 7, p: 8, m0: 9, m1: 10, moves: 11 } as const;
/** Line-search trials per Newton iteration: steps 1, 1/2, ..., 2^-(TRIALS-1). */
export const CLOTH_NEWTON_TRIALS = 8;
/** With contact, after the trial moves: the active set's normals (3 segments) and projector (2), then the inverse of
 * the vertex's 3x3 diagonal block H_vv before the active-set projection (2: the metric of the contact projection). */
export const CLOTH_NEWTON_ACTIVE = { normals: CLOTH_NEWTON_SEGMENTS.moves + CLOTH_NEWTON_TRIALS, projector: CLOTH_NEWTON_SEGMENTS.moves + CLOTH_NEWTON_TRIALS + 3, inverse: CLOTH_NEWTON_SEGMENTS.moves + CLOTH_NEWTON_TRIALS + 5 } as const;
/** Per-vertex segments: the 11 solver vectors, then with contact one projected move per trial (nwProject) and the active set. */
export function clothNewtonSegmentCount(contact: boolean): number { return CLOTH_NEWTON_SEGMENTS.moves + (contact ? CLOTH_NEWTON_TRIALS + 7 : 0); }
/** Line-search partial arrays (each W records): energy changes (2), rounding floors (2), slopes (2), max |d|^2, the active set,
 * the per-trial counts of vertices a travel bound holds (2). */
export const CLOTH_NEWTON_TRIAL_RECORDS = 10;
/** PCG partial arrays (one record per patch each, by iteration mod 3), then the PCG scalar records. */
export const CLOTH_PCG_PARTIALS = 3;
export const CLOTH_PCG_SCALARS = 3;

/** The compiled PCG layout the shaders are specialised to (gpu-solver.ts, compileClothPCGPatches). */
export interface ClothPCGLayout {
  /** Workgroup-memory vertices a patch stages (its 256 rows, then its halo). */
  readonly shared: number;
  /** Largest blocks per row and bending entries per row over the patches. */
  readonly maxK1: number;
  readonly maxK2: number;
}

export function clothNewtonWGSL(binding: number, workgroup: number, contact: boolean, pcg: ClothPCGLayout, subgroups = false, trustRegion = false, activeSet = true): string {
  const S = CLOTH_NEWTON_SEGMENTS;
  // The PCG kernels' workgroup sum of the vec4 `part` into `total` (valid in lane 0; every lane has passed a barrier
  // after its last read of the staged u): with subgroups, subgroupAdd, the subgroups' partials through pcgRed and a
  // second subgroupAdd (PCG_PATCH / sgn <= sgn: subgroups of at least 16 lanes); else the 8-level tree.
  const sgArgs = subgroups ? ', @builtin(subgroup_invocation_id) sgl: u32, @builtin(subgroup_size) sgn: u32' : '';
  const reduce = (part: string) => subgroups ? `
  let sgPart = subgroupAdd(${part});
  if (sgl == 0u) { pcgRed[lane / sgn] = sgPart; }
  workgroupBarrier();
  let total = subgroupAdd(select(vec4f(0.0), pcgRed[min(sgl, PCG_PATCH / sgn - 1u)], sgl < PCG_PATCH / sgn));` : `
  pcgRed[lane] = ${part};
  workgroupBarrier();
  for (var s = PCG_PATCH / 2u; s > 0u; s >>= 1u) { if (lane < s) { pcgRed[lane] += pcgRed[lane + s]; } workgroupBarrier(); }
  let total = pcgRed[0];`;
  return /* wgsl */`
@group(0) @binding(${binding}) var<storage, read_write> nw: array<vec4f>;
const NW_WG = ${workgroup}u;
const NW_HYBRID = ${trustRegion};
fn nseg(s: u32, v: u32) -> u32 { return s * P.state0.x + v; }
fn nwFaceRecord(f: u32) -> u32 { return P.newton0.x + 3u * f; }
fn nwPartial(k: u32) -> u32 { return P.newton0.y + k * P.newton0.z; }
fn nwControl() -> u32 { return P.newton0.w; }
// The trust region: mode (0 the line search, 1 trust-region iterations) and Delta (the preconditioner's norm), set by
// nwChoose and read-only through the solve; this iteration's direction (written by the solve's lead: ||d||_M^2, 1 where the
// solve ended on the boundary); the step's counts (consecutive full line-search steps, trust-region iterations, of them
// rejected, the last rho; harness reads); the last trust-region iteration's model (g.s, s.H s, contact's share, the energy
// change at s; harness reads, for the fixed-state check of the model against the Float64 Hessian).
fn nwTrust() -> u32 { return nwControl() + 3u; }
fn nwTrustNorm() -> u32 { return nwControl() + 5u; }
fn nwTrustCounts() -> u32 { return nwControl() + 4u; }
fn nwTrustModel() -> u32 { return nwControl() + 6u; }
fn nwPinned(v: u32) -> bool { return vertexGroup(v) > 0u; }
// Trial k's move of vertex j (alpha = 2^-k): alpha d_j, or with contact the projected arc's move Pi_j(x_j + alpha d_j) - x_j
// (nwProject; exactly alpha d_j where the projection keeps the trial point; d_j.w != 0 marks a vertex the arc cuts on some trial).
fn nwMove(j: u32, k: u32, alpha: f32) -> vec3f {
  ${contact ? `let D = nw[nseg(${S.d}u, j)];
  if (D.w == 0.0) { return alpha * D.xyz; }
  return nw[nseg(${S.moves}u + k, j)].xyz;` : 'return alpha * nwD(j);'}
}
fn nwD(j: u32) -> vec3f { return nw[nseg(${S.d}u, j)].xyz; }
// d_j with its cut mark (w != 0: the arc's moves of j are not alpha d_j; always 0 without contact).
fn nwD4(j: u32) -> vec4f { return nw[nseg(${S.d}u, j)]; }
${contact ? `
// ---------------------------------------------------------------- the active set (projected Newton)
const NW_ACTIVE = ${CLOTH_NEWTON_ACTIVE.normals}u;
const NW_PROJ = ${CLOTH_NEWTON_ACTIVE.projector}u;
const NW_HINV = ${CLOTH_NEWTON_ACTIVE.inverse}u;
// H_vv^-1 of vertex v at the current Newton iterate (nwGradient; contact included, before the active-set projection):
// the per-vertex metric for the contact block's projection. Symmetric; the identity's multiple is never substituted.
fn nwVertexInv(v: u32) -> mat3x3f {
  let a = nw[nseg(NW_HINV, v)]; let b = nw[nseg(NW_HINV + 1u, v)];
  return mat3x3f(a.x, a.y, a.z, a.y, a.w, b.x, a.z, b.x, b.y);
}
// P w at vertex v: the projection off its active normals (w itself with none).
fn nwProjectP(v: u32, w: vec3f) -> vec3f {
  let b = nw[nseg(NW_PROJ + 1u, v)];
  if (b.w == 0.0) { return w; }
  let a = nw[nseg(NW_PROJ, v)];
  return vec3f(a.x * w.x + a.y * w.y + a.z * w.z, a.y * w.x + a.w * w.y + b.x * w.z, a.z * w.x + b.x * w.y + b.y * w.z);
}
// Gram-Schmidt of N[0..cnt): orthonormal rows, a (numerically) dependent one zero.
fn nwOrtho(N: array<vec3f, 3>, cnt: u32) -> array<vec3f, 3> {
  var Q: array<vec3f, 3>;
  for (var i = 0u; i < cnt; i++) {
    var r = N[i];
    for (var j = 0u; j < i; j++) { r -= dot(r, Q[j]) * Q[j]; }
    let l = length(r); Q[i] = select(vec3f(0.0), r / l, l > NW_RANK);
  }
  return Q;
}
// The rank floor of the active normals (unit vectors): a normal within asin(1e-2) of the others' span adds nothing.
const NW_RANK = 1e-2;
// An independent set of unit normals (at most three) and the KKT release against the force -g: the multipliers
// mu = (N^T N)^-1 N^T (-g), every normal with mu <= 0 dropped and mu recomputed on the rest (at most three rounds).
struct NwSet { n: array<vec3f, 3>, cnt: u32 }
fn nwAddIndependent(s: ptr<function, NwSet>, a: vec3f) {
  if ((*s).cnt >= 3u) { return; }
  var r = a; let Q = nwOrtho((*s).n, (*s).cnt);
  for (var j = 0u; j < (*s).cnt; j++) { r -= dot(r, Q[j]) * Q[j]; }
  if (length(r) > NW_RANK) { (*s).n[(*s).cnt] = a; (*s).cnt++; }
}
fn nwRelease(s: ptr<function, NwSet>, g: vec3f) {
  for (var round = 0u; round < 3u && (*s).cnt > 0u; round++) {
    let N = (*s).n; let cnt = (*s).cnt;
    var G = IDENTITY; var b = vec3f(0.0);
    for (var i = 0u; i < cnt; i++) { b[i] = -dot(N[i], g); for (var j = 0u; j < cnt; j++) { G[i][j] = dot(N[i], N[j]); } }
    // Cramer on the Gram matrix (identity past cnt): the signs of mu only.
    let det = determinant(G);
    let mu = vec3f(determinant(mat3x3f(b, G[1], G[2])), determinant(mat3x3f(G[0], b, G[2])), determinant(mat3x3f(G[0], G[1], b))) / det;
    var K: array<vec3f, 3>; var k = 0u;
    for (var i = 0u; i < cnt; i++) { if (mu[i] > 0.0) { K[k] = N[i]; k++; } }
    if (k == cnt) { break; }
    (*s).n = K; (*s).cnt = k;
  }
}
// Vertex v's active set at the Newton iterate with gradient g: the normals the last nwApply stored (the certificates
// binding the iterate in the latest filter's set; none at the step's first iteration, whose certificates are the
// step's own), reduced to an independent set, then released by the KKT sign (nwRelease). Writes P = I - Q Q^T
// (projector records: xx, xy, xz, yy; yz, zz, the normals released, the count kept; count 0: P = I).
fn nwActiveSet(v: u32, g: vec3f, first: bool) -> mat3x3f {
  var s: NwSet; var stored = 0.0;
  for (var i = 0u; i < 3u; i++) {
    let a = nw[nseg(NW_ACTIVE + i, v)];
    if (first || a.w == 0.0) { continue; }
    stored += 1.0; nwAddIndependent(&s, a.xyz);
  }
  nwRelease(&s, g);
  let N = s.n; let cnt = s.cnt;
  let Q = nwOrtho(N, cnt);
  var Pm = IDENTITY;
  for (var i = 0u; i < cnt; i++) { Pm -= outer(Q[i], Q[i]); }
  nw[nseg(NW_PROJ, v)] = vec4f(Pm[0][0], Pm[0][1], Pm[0][2], Pm[1][1]);
  nw[nseg(NW_PROJ + 1u, v)] = vec4f(Pm[1][2], Pm[2][2], stored - f32(cnt), f32(cnt));
  return Pm;
}
// Whether a barrier acts on vertex v placed at xv (the projected trial point q the done rule accepts; the pairs' other
// vertices where they are): one of its contact entries within r there (kappa'' > 0, the solids' fields included), or one
// outside the barrier's domain (infinite energy). A held vertex never reads as travel by this test. A certificate binding a vertex no barrier acts on bounds travel, not contact,
// whatever its pair's far bit. The filter's far test is conservative (each side's reach about the iterate, in a
// levelled round up to the farthest anchor plus the cap), so it leaves pairs far apart not-far: at the Bridson sphere's
// step 5 a vertex 8.4 mm over the sphere (r = 1 mm) ended Newton held on such a record.
fn nwBarrierActs(v: u32, xv: vec3f) -> bool {
  let header = ogcContacts[0]; let params = bitcast<vec4f>(ogcContacts[1]);
  let count = min(ogcContacts[header.z + (v >> 2u)][v & 3u], header.y); let base = header.w + v * header.y;
  for (var i = 0u; i < count; i++) {
    let l = ogcEntryLin(v, xv, ogcContacts[base + i], params);
    if (any(l.Mn[0] != vec3f(0.0)) || any(l.Mn[1] != vec3f(0.0)) || any(l.Mn[2] != vec3f(0.0)) || !(l.e < 3.0e38)) { return true; }
  }
  return false;
}
// The cap-aware done rule: whether a travel bound holds vertex v at its trial point Pi(y) (ogcProjectActive.travel: a
// binding certificate there is the cap, a solid piece's ball, a solid face beyond r or a record its pair cannot close
// within r - a bound of the round's certified set, not a contact; or any binding certificate where no barrier acts on v)
// and the binding normals (the cap's included; a
// held vertex: its direction) keep a positive KKT multiplier against the force -g at the iterate: Newton would move
// v further were the bound lifted, so a small step there is no convergence.
fn nwTravelHeld(v: u32, x: vec3f, y: vec3f, alpha: f32, g: vec3f) -> bool {
  let a = ogcProjectActive(v, x, y, alpha);
  if (!a.travel && (a.held || nwBarrierActs(v, a.q))) { return false; }
  var s: NwSet;
  if (a.held) { if (any(y != x)) { nwAddIndependent(&s, normalize(y - x)); } }
  else {
    let N = array<vec3f, 3>(a.n0, a.n1, a.n2);
    for (var i = 0u; i < min(a.count, 3u); i++) { nwAddIndependent(&s, N[i]); }
    if (a.capOn) { nwAddIndependent(&s, a.nc); }
  }
  nwRelease(&s, g);
  return s.cnt > 0u;
}` : ''}

var<workgroup> nwRed: array<vec4f, ${workgroup}>;
var<workgroup> nwOut: vec4f;
// The sum of partial array \`base\` (W records) by every workgroup, uniform across it.
fn nwSum(lane: u32, base: u32) -> vec4f { return nwSumN(lane, base, P.newton0.z); }
// The sum of \`count\` records from \`base\`, uniform across the workgroup.
fn nwSumN(lane: u32, base: u32, count: u32) -> vec4f {
  var a = vec4f(0.0);
  for (var i = lane; i < count; i += NW_WG) { a += nw[base + i]; }
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
// As nwSum / nwPartialSum with x summed and y the maximum (the line search's slope and largest |d_v|^2).
fn nwSumMax(lane: u32, base: u32) -> vec2f {
  var a = vec2f(0.0);
  for (var i = lane; i < P.newton0.z; i += NW_WG) { let q = nw[base + i]; a = vec2f(a.x + q.x, max(a.y, q.y)); }
  nwRed[lane] = vec4f(a, 0.0, 0.0);
  workgroupBarrier();
  for (var s = NW_WG / 2u; s > 0u; s >>= 1u) {
    if (lane < s) { let o = nwRed[lane + s]; nwRed[lane] = vec4f(nwRed[lane].x + o.x, max(nwRed[lane].y, o.y), 0.0, 0.0); }
    workgroupBarrier();
  }
  if (lane == 0u) { nwOut = nwRed[0]; }
  return workgroupUniformLoad(&nwOut).xy;
}
fn nwPartialSumMax(lane: u32, wgi: u32, value: vec2f, base: u32) {
  workgroupBarrier();
  nwRed[lane] = vec4f(value, 0.0, 0.0);
  workgroupBarrier();
  for (var s = NW_WG / 2u; s > 0u; s >>= 1u) {
    if (lane < s) { let o = nwRed[lane + s]; nwRed[lane] = vec4f(nwRed[lane].x + o.x, max(nwRed[lane].y, o.y), 0.0, 0.0); }
    workgroupBarrier();
  }
  if (lane == 0u) { nw[base + wgi] = nwRed[0]; }
}
var<workgroup> nwCtl: vec4f;
fn nwControlUniform(lane: u32) -> vec4f {
  if (lane == 0u) { nwCtl = nw[nwControl()]; }
  return workgroupUniformLoad(&nwCtl);
}
fn wgIndex(wg: vec3u, nwg: vec3u) -> u32 { return wg.x + wg.y * nwg.x; }

// Step start (one thread): Newton not done, the line search's mode (the active-set record, control + 2, is the contact hooks').
@compute @workgroup_size(1) fn nwBeginStep() {
  nw[nwControl()] = vec4f(0.0); nw[nwControl() + 1u] = vec4f(0.0);
  nw[nwTrust()] = vec4f(0.0); nw[nwTrustCounts()] = vec4f(0.0); nw[nwTrustNorm()] = vec4f(0.0); nw[nwTrustModel()] = vec4f(0.0);
}

// Per face at the Newton iterate: f0, f1, the stress (S11, S22, S12) - PSD-clamped, or in trust-region iterations as it
// is (the exact membrane Hessian) - and the softened tangent diagonal.
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
  if (l2 < 0.0 && !(NW_HYBRID && nw[nwTrust()].x != 0.0)) {
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

// ---------------------------------------------------------------- compiled topology (gpu-solver.ts)
const PCG_PATCH = 256u;
const PCG_SHARED = ${pcg.shared}u;
const PCG_MAXK1 = ${pcg.maxK1}u;
const PCG_MAXK2 = ${Math.max(pcg.maxK2, 2)}u;
const PCG_GLOBAL = 0x80000000u;
const PCG_FAR = 0x40000000u;
// Patch p's table words: K1, K2, halo count, halo (topo word), blocks, bending entries (nw records), columns (topo word), rows.
fn pcgTable(p: u32) -> u32 { return P.newton2.x + 8u * p; }
// Patches are tiles of the vertices' Morton order: the vertex of patch slot p PATCH + lane, and the slot of vertex v.
fn pcgRowVertex(slot: u32) -> u32 { return topo[P.newton4.x + slot]; }
fn pcgVertexSlot(v: u32) -> u32 { return topo[P.newton4.y + v]; }
// Block slot k of patch-row lane: row c of H_vj at nw[blocks + (3k + c) PATCH + lane] (row 0's w: the column bits).
fn pcgBlock(blocks: u32, k: u32, c: u32, lane: u32) -> u32 { return blocks + (3u * k + c) * PCG_PATCH + lane; }

// Per vertex: g at the iterate, the inverse diagonal block, the assembled row of H, and the PCG start r = -g, s = 0,
// d = 0, p = 0.
@compute @workgroup_size(${workgroup}) fn nwGradient(@builtin(local_invocation_index) lane: u32, @builtin(workgroup_id) wg: vec3u, @builtin(num_workgroups) nwg: vec3u) {
  let ctl = nwControlUniform(lane);
  if (ctl.x != 0.0) { return; }
  let v = wgIndex(wg, nwg) * NW_WG + lane;
  if (v >= P.n.x) { return; }
  var g = vec3f(0.0); var M0 = vec4f(0.0); var M1 = vec4f(0.0);
  // The row's blocks and bending entries, assembled in function-local arrays (measured fastest: 334 us at 150^2,
  // against 574 for registers selected by a switch and 750 for module-private arrays with direct entry writes).
  var Hb: array<mat3x3f, PCG_MAXK1>;
  var Bw: array<f32, PCG_MAXK2>;
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
    nwAssembleRow(v, &Hb, &Bw);
  }
  var r0 = -g;
  ${contact ? `nw[nseg(NW_HINV, v)] = M0; nw[nseg(NW_HINV + 1u, v)] = M1;
  // The reduced system: r0 = -P g and the preconditioner P M^-1 P (pinned rows: no set, P = I on zero rows).
  if (!nwPinned(v)) {
    let Pm = nwActiveSet(v, g, ctl.w == 0.0);
    if (nw[nseg(NW_PROJ + 1u, v)].w > 0.0) {
      r0 = Pm * r0;
      let Mt = Pm * mat3x3f(M0.x, M0.y, M0.z, M0.y, M0.w, M1.x, M0.z, M1.x, M1.y) * Pm;
      M0 = vec4f(Mt[0][0], Mt[0][1], Mt[0][2], Mt[1][1]); M1 = vec4f(Mt[1][2], Mt[2][2], 0.0, 0.0);
    }
  } else { nw[nseg(NW_PROJ + 1u, v)] = vec4f(0.0); }` : ''}
  nw[nseg(${S.g}u, v)] = vec4f(g, 0.0); nw[nseg(${S.r0}u, v)] = vec4f(r0, 0.0); nw[nseg(${S.s1}u, v)] = vec4f(0.0);
  nw[nseg(${S.d}u, v)] = vec4f(0.0); nw[nseg(${S.p}u, v)] = vec4f(0.0);
  nw[nseg(${S.m0}u, v)] = M0; nw[nseg(${S.m1}u, v)] = M1;
  // The row in sliced-ELL order (a pinned row and the patch's padding slots: zero blocks on a valid column).
  let slot = pcgVertexSlot(v); let tw = pcgTable(slot / PCG_PATCH); let l = slot % PCG_PATCH;
  let K1 = topo[tw]; let K2 = topo[tw + 1u]; let blocks = topo[tw + 4u]; let bends = topo[tw + 5u]; let cols = topo[tw + 6u];
  for (var k = 0u; k < K1; k++) {
    let H = Hb[min(k, PCG_MAXK1 - 1u)]; let col = bitcast<f32>(topo[cols + k * PCG_PATCH + l]);
    nw[pcgBlock(blocks, k, 0u, l)] = vec4f(H[0][0], H[1][0], H[2][0], col);
    nw[pcgBlock(blocks, k, 1u, l)] = vec4f(H[0][1], H[1][1], H[2][1], 0.0);
    nw[pcgBlock(blocks, k, 2u, l)] = vec4f(H[0][2], H[1][2], H[2][2], 0.0);
  }
  for (var m = 0u; 2u * m < K2; m++) {
    let c = cols + (K1 + 2u * m) * PCG_PATCH + l;
    nw[bends + m * PCG_PATCH + l] = vec4f(bitcast<f32>(topo[c]), Bw[min(2u * m, PCG_MAXK2 - 1u)], bitcast<f32>(topo[c + PCG_PATCH]), Bw[min(2u * m + 1u, PCG_MAXK2 - 1u)]);
  }
}
// M^-1 r with the symmetric inverse (xx, xy, xz, yy), (yz, zz).
fn nwPrecondition(M0: vec4f, M1: vec4f, r: vec3f) -> vec3f {
  return vec3f(M0.x * r.x + M0.y * r.y + M0.z * r.z, M0.y * r.x + M0.w * r.y + M1.x * r.z, M0.z * r.x + M1.x * r.y + M1.y * r.z);
}

// Row v of H (all but contact) into its slots: inertia, grab, bending diagonal; per face the membrane tangent (rates
// R_k = (a_k f0, b_k f1, a_k f1 + b_k f0) through the constitutive matrix) plus the clamped geometric term, and the face
// damping; the active strain-limit edges; the bending stencil (1-ring entries into blocks, the rest into Bw).
fn nwAssembleRow(v: u32, Hb: ptr<function, array<mat3x3f, PCG_MAXK1>>, Bw: ptr<function, array<f32, PCG_MAXK2>>) {
  let h = hEff(); let m = vertexRest(v).x;
  (*Hb)[0] = (m / (h * h) + grabSpring(v, pos(v)).w) * IDENTITY + stencilDiagonal(v);
  let damped = P.state1.z != 0u;
  for (var j = topo[P.topo0.x + v]; j < topo[P.topo0.x + v + 1u]; j++) {
    let entry = topo[P.topo0.y + j]; let f = entry >> 2u; let k = entry & 3u; let slots = topo[P.newton2.y + j];
    let t = tris(f); let dm = rest[P.rest0.x + f]; let fr = rest[P.rest0.y + f]; let c0 = rest[P.rest1.z + 2u * t.w];
    let a = vec3f(-dm.x - dm.y, dm.x, dm.y); let b = vec3f(-dm.z - dm.w, dm.z, dm.w);
    let r = nwFaceRecord(f); let F0 = nw[r]; let F1 = nw[r + 1u]; let F2 = nw[r + 2u];
    let f0 = F0.xyz; let f1 = F1.xyz;
    let ak = a[k]; let bk = b[k];
    let rk0 = ak * f0; let rk1 = bk * f1; let rk2 = ak * f1 + bk * f0;
    let wd = damped && fr.y > 0.0;
    var T0 = vec4f(0.0); var T1 = vec4f(0.0); var sc = 0.0;
    if (wd) { T0 = state[P.state1.x + 2u * f]; T1 = state[P.state1.x + 2u * f + 1u]; sc = fr.y * fr.x / h; }
    let gk0 = ak * T0.xyz; let gk1 = bk * T1.xyz; let gk2 = ak * T1.xyz + bk * T0.xyz;
    for (var c = 0u; c < 3u; c++) {
      let al = a[c]; let bl = b[c];
      let rl0 = al * f0; let rl1 = bl * f1; let rl2 = al * f1 + bl * f0;
      let skl = ak * al * F0.w + (ak * bl + bk * al) * F2.x + bk * bl * F1.w;
      var B = fr.x * (F2.y * outer(rk0, rl0) + c0.z * (outer(rk0, rl1) + outer(rk1, rl0)) + F2.z * outer(rk1, rl1) + c0.w * outer(rk2, rl2) + skl * IDENTITY);
      if (wd) {
        let gl0 = al * T0.xyz; let gl1 = bl * T1.xyz; let gl2 = al * T1.xyz + bl * T0.xyz;
        B += sc * (T0.w * outer(gk0, gl0) + c0.z * (outer(gk0, gl1) + outer(gk1, gl0)) + T1.w * outer(gk1, gl1) + c0.w * outer(gk2, gl2));
      }
      let sl = (slots >> (8u * c)) & 255u;
      (*Hb)[sl] += B;
    }
  }
  // Strain-limit edges (augmented Lagrangian, active only): K = k n n^T + (f / len) (I - n n^T) on x_v - x_other.
  if (P.flags.z != 0u) {
    let xv = pos(v);
    for (var j = topo[P.topo1.x + v]; j < topo[P.topo1.x + v + 1u]; j++) {
      let e = topo[P.topo1.y + j] >> 2u;
      if (!(rest[P.rest0.w + e].y > 0.0)) { continue; }
      let ev = edges(e); let other = select(ev.x, ev.y, ev.x == v);
      let dd = xv - pos(other); let len = length(dd);
      let dual = edgeDual(e); let f = dual.y * (len - dual.z) + dual.x;
      if (!(len > 0.0) || !(f > 0.0)) { continue; }
      let n = dd / len; let K = (dual.y - f / len) * outer(n, n) + (f / len) * IDENTITY;
      (*Hb)[0] += K;
      let so = topo[P.newton2.z + j];
      (*Hb)[so] -= K;
    }
  }
  // The bending stencil (and its damping) off the diagonal.
  let hd = 1.0 / h;
  for (var o = topo[P.topo2.z + v]; o < topo[P.topo2.z + v + 1u]; o++) {
    let rec = topo4[P.newton1.x + o];
    let w = bitcast<f32>(rec.y) + select(0.0, bitcast<f32>(rec.z) * hd, damped);
    let dst = topo[P.newton2.w + o];
    if ((dst & PCG_FAR) != 0u) { (*Bw)[dst & (PCG_FAR - 1u)] += w; } else { (*Hb)[dst] += w * IDENTITY; }
  }
}

// ---------------------------------------------------------------- PCG (Chronopoulos-Gear), one dispatch per iteration
var<workgroup> pcgUx: array<f32, PCG_SHARED>;
var<workgroup> pcgUy: array<f32, PCG_SHARED>;
var<workgroup> pcgUz: array<f32, PCG_SHARED>;
var<workgroup> pcgRed: array<vec4f, PCG_PATCH>;
var<workgroup> pcgOut: vec4f;
var<workgroup> pcgScal: vec4f;
var<workgroup> pcgNorm: vec4f;
var<workgroup> pcgTrust: vec4f;
// Where the product's far columns and contact partners read their vector: 0 the PCG's u, 1 the trust-region trial's
// move (nwModel).
var<private> pcgSrc: u32;
var<private> pcgI: u32;
var<private> pcgAlpha: f32;
var<private> pcgBeta: f32;
var<private> pcgPatchIndex: u32;
fn pcgPartial(slot: u32, p: u32) -> u32 { return P.newton3.x + slot * P.newton3.y + p; }
fn pcgScalar(k: u32) -> u32 { return P.newton3.z + k; }
// The PCG stop of Newton iteration n (as tol^2 on r.u / r0.u0): pcgTolerance^2, or with Eisenstat-Walker forcing (eta_max^2
// at P.newton4.z, f32 bits; 0 off) choice 2 at gamma 0.9, eta = 0.9 (|g_n| / |g_n-1|)^2, safeguarded by 0.9 eta_n-1^2 where
// that exceeds 0.1, capped at eta_max and floored at pcgTolerance. prior: the step's residual record (control + 1)
// before iteration n writes it (y |g_n-1|^2, z eta_n-1^2); gg = |g_n|^2.
fn pcgForcing(gg: f32, prior: vec4f, n: f32) -> f32 {
  let floor2 = bitcast<f32>(P.newton1.y); let max2 = bitcast<f32>(P.newton4.z);
  if (!(max2 > 0.0)) { return floor2; }
  var e2 = max2;
  if (n > 0.0 && prior.y > 0.0) {
    let q = gg / prior.y; e2 = 0.81 * q * q;
    let sg = 0.9 * prior.z;
    if (sg > 0.1) { e2 = max(e2, sg * sg); }
    e2 = min(e2, max2);
  }
  return max(e2, floor2);
}
// Steihaug's step to the trust boundary: the tau >= 0 with ||d + tau p||_M = Delta, from ||d||_M^2, <d, p>_M, ||p||_M^2
// (||d||_M < Delta: the solve stops on reaching it, and the test and the recurrence evaluate the same expression).
fn pcgBoundary(dd: f32, dp: f32, pp: f32, delta: f32) -> f32 {
  return (sqrt(max(dp * dp + pp * (delta * delta - dd), 0.0)) - dp) / pp;
}
fn pcgMr(j: u32, r: vec3f) -> vec3f { return nwPrecondition(nw[nseg(${S.m0}u, j)], nw[nseg(${S.m1}u, j)], r); }
// u = M^-1 r of this dispatch's iteration at vertex j from global memory: u_0 = M^-1 r_0; else r_i = r_{i-1} - alpha
// s_{i-1} with s_{i-1} = w_{i-1} + beta s_{i-2}.
fn pcgUGlobal(j: u32) -> vec3f {
  if (pcgSrc == 1u) { return nwMove(j, 0u, 1.0); }
  if (pcgI == 0u) { return pcgMr(j, nw[nseg(${S.r0}u, j)].xyz); }
  let o = (pcgI + 1u) % 2u;
  let sv = nw[nseg(${S.w0}u + o, j)].xyz + pcgBeta * nw[nseg(${S.s0}u + pcgI % 2u, j)].xyz;
  return pcgMr(j, nw[nseg(${S.r0}u + o, j)].xyz - pcgAlpha * sv);
}
// u at a column of a one-patch problem (pcgSolve): no halo, so every column is a staged row.
fn pcgULocal(c: u32) -> vec3f { return vec3f(pcgUx[c], pcgUy[c], pcgUz[c]); }
fn pcgStage(c: u32, u: vec3f) { pcgUx[c] = u.x; pcgUy[c] = u.y; pcgUz[c] = u.z; }
// u at a compiled column: a staged row or halo vertex, else (bit 31) global.
fn pcgU(c: u32) -> vec3f {
  if (c < PCG_SHARED) { return vec3f(pcgUx[c], pcgUy[c], pcgUz[c]); }
  return pcgUGlobal(c & (PCG_GLOBAL - 1u));
}
// u at vertex j (the contact product's partners): the patch's own rows are staged.
fn pcgDir(j: u32) -> vec3f {
  let slot = pcgVertexSlot(j);
  if (slot / PCG_PATCH == pcgPatchIndex) { let l = slot % PCG_PATCH; return vec3f(pcgUx[l], pcgUy[l], pcgUz[l]); }
  return pcgUGlobal(j);
}

// Row lane of the patch's assembled H (all but contact) times the staged vector (pcgU: staged rows and halo, else global).
fn pcgRowProduct(lane: u32, K1: u32, K2: u32, blocks: u32, bends: u32) -> vec3f {
  var w = vec3f(0.0);
  for (var k = 0u; k < K1; k++) {
    let b0 = nw[pcgBlock(blocks, k, 0u, lane)]; let b1 = nw[pcgBlock(blocks, k, 1u, lane)]; let b2 = nw[pcgBlock(blocks, k, 2u, lane)];
    let uj = pcgU(bitcast<u32>(b0.w));
    w += vec3f(dot(b0.xyz, uj), dot(b1.xyz, uj), dot(b2.xyz, uj));
  }
  for (var m = 0u; 2u * m < K2; m++) {
    let e = nw[bends + m * PCG_PATCH + lane];
    w += e.y * pcgU(bitcast<u32>(e.x)) + e.w * pcgU(bitcast<u32>(e.z));
  }
  return w;
}

// PCG dispatch i (R.range.z, from 0). i = 0: u_0 = M^-1 r_0, w_0 = H u_0 and the partials (r.u, w.u, |g|^2). i >= 1:
// gamma = r.u and delta = w.u of iteration i - 1 (with i = 1 the Newton residual test), beta = gamma / gamma_{i-2},
// alpha = gamma / (delta - beta gamma / alpha_{i-2}); then p_{i-1} = u_{i-1} + beta p_{i-2}, s_{i-1} = w_{i-1} + beta
// s_{i-2}, d += alpha p_{i-1}, r_i = r_{i-1} - alpha s_{i-1}, u_i = M^-1 r_i, w_i = H u_i and the partials of iteration i.
@compute @workgroup_size(256) fn pcgStep(@builtin(local_invocation_index) lane: u32, @builtin(workgroup_id) wg: vec3u, @builtin(num_workgroups) nwg: vec3u${sgArgs}) {
  let ctl = nwControlUniform(lane);
  if (ctl.x != 0.0 || ctl.y != 0.0) { return; }
  let i = R.range.z; let p = wgIndex(wg, nwg); let lead = p == 0u && lane == 0u;
  var alpha = 0.0; var beta = 0.0;
  if (i > 0u) {
    // Sums of the partials of iterations i - 1 and i - 2, and the scalars (gamma_0, alpha_{i-2}), uniform.
    var a = vec4f(0.0);
    for (var k = lane; k < P.newton3.y; k += PCG_PATCH) {
      let q1 = nw[pcgPartial((i - 1u) % 3u, k)];
      a += vec4f(q1.xyz, select(0.0, nw[pcgPartial((i + 1u) % 3u, k)].x, i >= 2u));
    }
    ${reduce('a')}
    if (lane == 0u) { let s0 = nw[pcgScalar(0u)]; let q = nw[pcgScalar(1u + i % 2u)]; pcgOut = total; pcgScal = vec4f(s0.x, q.x, s0.y, 0.0); pcgNorm = q; pcgTrust = nw[nwTrust()]; }
    let sums = workgroupUniformLoad(&pcgOut);
    let scal = workgroupUniformLoad(&pcgScal);
    let nrm = workgroupUniformLoad(&pcgNorm);
    let trust = workgroupUniformLoad(&pcgTrust);
    let gam = sums.x;
    var gamma0 = scal.x;
    if (i == 1u) {
      gamma0 = gam;
      // Residual record (control + 1): |g|^2 over free DOFs at the step's first and latest Newton iterate.
      // The lead reads the record before writing it (its forcing needs |g| of the last Newton iterate); at i = 1 the stop
      // below passes for any tolerance < 1, so the other workgroups need the stop's tolerance only from dispatch 2 on.
      if (lead) {
        var rr = nw[nwControl() + 1u]; let e2 = pcgForcing(sums.z, rr, ctl.w);
        if (ctl.w == 0.0) { rr.x = sums.z; } rr.y = sums.z; rr.z = e2; nw[nwControl() + 1u] = rr; nw[pcgScalar(0u)] = vec4f(gam, e2, 0.0, 0.0);
      }
      if (sums.z <= bitcast<f32>(P.newton1.z)) { if (lead) { nw[nwControl()].x = 1.0; } return; }
    }
    // The last dispatch's step reached the trust boundary (its alpha record -1): the solve is done.
    if (i >= 2u && scal.y < 0.0) { if (lead) { nw[nwControl()].y = 1.0; } return; }
    if (!(gam > select(scal.z, 0.0, i == 1u) * gamma0)) { if (lead) { nw[nwControl()].y = 1.0; } return; }
    var denom = sums.y;
    if (i >= 2u) { beta = gam / sums.w; denom = sums.y - beta * gam / scal.y; }
    // ||d||_M^2, <d, p>_M, ||p||_M^2 of d_{i-1} and p_{i-1} (M the preconditioner's inverse), by the Steihaug-Toint
    // recurrences from those of i - 2 (the alpha record's yzw): r_{i-1} is orthogonal to d_{i-1} and p_{i-2}.
    var dd = 0.0; var dp = 0.0; var pp = gam;
    if (i >= 2u) { let a2 = scal.y; dd = nrm.y + a2 * (2.0 * nrm.z + a2 * nrm.w); dp = beta * (nrm.z + a2 * nrm.w); pp = gam + beta * beta * nrm.w; }
    let tr = NW_HYBRID && trust.x != 0.0;
    var edge = false;
    if (!(denom > 0.0)) {
      // The line search's H is PSD by construction: a non-positive curvature (rounding at convergence) ends this solve
      // with d so far. The trust region's model (the exact membrane) can be indefinite: Steihaug's step to the boundary.
      if (!tr) { if (lead) { nw[nwControl()].y = 1.0; } return; }
      edge = true;
    } else {
      alpha = gam / denom;
      edge = tr && dd + alpha * (2.0 * dp + alpha * pp) >= trust.y * trust.y;
    }
    if (edge) { alpha = pcgBoundary(dd, dp, pp, trust.y); }
    // The step's PCG iterations taken (control + 1 .w; harness reads only), and d's norm after this dispatch's update.
    if (lead) {
      nw[pcgScalar(1u + (i - 1u) % 2u)] = vec4f(select(alpha, -1.0, edge), dd, dp, pp); nw[nwControl() + 1u].w += 1.0;
      nw[nwTrustNorm()] = vec4f(dd + alpha * (2.0 * dp + alpha * pp), select(0.0, 1.0, edge), 0.0, 0.0);
    }
  } else if (lead) { nw[nwTrustNorm()] = vec4f(0.0); }
  pcgI = i; pcgAlpha = alpha; pcgBeta = beta;
  let tw = pcgTable(p);
  let K1 = topo[tw]; let K2 = topo[tw + 1u]; let haloN = min(topo[tw + 2u], PCG_SHARED - PCG_PATCH); let halo = topo[tw + 3u];
  let blocks = topo[tw + 4u]; let bends = topo[tw + 5u]; let rows = topo[tw + 7u];
  pcgPatchIndex = p;
  let own = lane < rows; let v = select(0u, pcgRowVertex(p * PCG_PATCH + lane), own);
  var r = vec3f(0.0); var u = vec3f(0.0);
  if (own) {
    let M0 = nw[nseg(${S.m0}u, v)]; let M1 = nw[nseg(${S.m1}u, v)];
    if (i == 0u) { r = nw[nseg(${S.r0}u, v)].xyz; } else {
      let o = (i + 1u) % 2u;
      let rOld = nw[nseg(${S.r0}u + o, v)].xyz;
      let sv = nw[nseg(${S.w0}u + o, v)].xyz + beta * nw[nseg(${S.s0}u + i % 2u, v)].xyz;
      let pv = nwPrecondition(M0, M1, rOld) + beta * nw[nseg(${S.p}u, v)].xyz;
      r = rOld - alpha * sv;
      nw[nseg(${S.p}u, v)] = vec4f(pv, 0.0);
      nw[nseg(${S.d}u, v)] = vec4f(nw[nseg(${S.d}u, v)].xyz + alpha * pv, 0.0);
      nw[nseg(${S.s0}u + o, v)] = vec4f(sv, 0.0);
      nw[nseg(${S.r0}u + i % 2u, v)] = vec4f(r, 0.0);
    }
    u = nwPrecondition(M0, M1, r);
  }
  pcgStage(lane, u);
  for (var h = lane; h < haloN; h += PCG_PATCH) { pcgStage(PCG_PATCH + h, pcgUGlobal(topo[halo + h])); }
  workgroupBarrier();
  var w = vec3f(0.0);
  if (own) {
    w = pcgRowProduct(lane, K1, K2, blocks, bends);
    // Contact pairs (C2), matrix-free: the sum over v's entries of w_v M (sum_j w_j u_j), linearized at the iterate.
    ${contact ? 'if (!nwPinned(v)) { w = nwProjectP(v, w + ogcVertexContactHvLanes(v, 0u, 1u)); }' : ''}
    nw[nseg(${S.w0}u + i % 2u, v)] = vec4f(w, 0.0);
  }
  ${reduce('vec4f(dot(r, u), dot(w, u), select(0.0, dot(r, r), i == 0u), 0.0)')}
  if (lane == 0u) { nw[pcgPartial(i % 3u, p)] = total; }
}

// PCG in one dispatch (one patch: every vertex a row of workgroup 0, no halo): the iterations of pcgStep, each lane
// keeping its row's r, s, p, d, u and w in registers, u shared through the staging arrays and gathered without the
// global fallback (every column is a row lane), the partials reduced in the workgroup, and the stop a uniform break. The same Chronopoulos-Gear recurrence as dispatches 0..cap (R.range.z).
@compute @workgroup_size(256) fn pcgSolve(@builtin(local_invocation_index) lane: u32${sgArgs}) {
  let ctl = nwControlUniform(lane);
  if (ctl.x != 0.0 || ctl.y != 0.0) { return; }
  let cap = R.range.z; let lead = lane == 0u;
  let tw = pcgTable(0u);
  let K1 = topo[tw]; let K2 = topo[tw + 1u]; let blocks = topo[tw + 4u]; let bends = topo[tw + 5u]; let rows = topo[tw + 7u];
  pcgPatchIndex = 0u; pcgI = 0u; pcgAlpha = 0.0; pcgBeta = 0.0;
  let own = lane < rows; let v = select(0u, pcgRowVertex(lane), own);
  var M0 = vec4f(0.0); var M1 = vec4f(0.0); var r = vec3f(0.0);
  if (own) { M0 = nw[nseg(${S.m0}u, v)]; M1 = nw[nseg(${S.m1}u, v)]; r = nw[nseg(${S.r0}u, v)].xyz; }
  var u = nwPrecondition(M0, M1, r);
  var pv = vec3f(0.0); var sv = vec3f(0.0); var d = vec3f(0.0); var w = vec3f(0.0);
  var gamma0 = 0.0; var gamPrev = 0.0; var alphaPrev = 0.0;
  var taken = 0.0; var stop = 0u;
  // The step's residual record before this Newton iteration writes it (the forcing's |g_n-1| and eta_n-1), and the trust region.
  if (lead) { pcgScal = nw[nwControl() + 1u]; pcgTrust = nw[nwTrust()]; }
  let prior = workgroupUniformLoad(&pcgScal);
  let trust = workgroupUniformLoad(&pcgTrust);
  let tr = NW_HYBRID && trust.x != 0.0;
  // ||d||_M^2, <d, p>_M, ||p||_M^2 (the recurrences of pcgStep), d's norm after the last update, the boundary reached.
  var dd = 0.0; var dp = 0.0; var pp = 0.0; var dn = 0.0; var edge = false;
  var tol2 = bitcast<f32>(P.newton1.y);
  for (var i = 0u; i <= cap; i++) {
    if (i > 0u) {
      // gamma = r.u, delta = w.u of iteration i - 1 (pcgRed[0] after the reduction below), uniform.
      let sums = workgroupUniformLoad(&pcgOut);
      let gam = sums.x;
      if (i == 1u) {
        gamma0 = gam;
        tol2 = pcgForcing(sums.z, prior, ctl.w);
        if (lead) { var rr = prior; if (ctl.w == 0.0) { rr.x = sums.z; } rr.y = sums.z; rr.z = tol2; nw[nwControl() + 1u] = rr; nw[pcgScalar(0u)] = vec4f(gam, tol2, 0.0, 0.0); }
        if (sums.z <= bitcast<f32>(P.newton1.z)) { stop = 2u; break; }
      }
      if (!(gam > tol2 * gamma0)) { stop = 1u; break; }
      var beta = 0.0; var denom = sums.y;
      if (i >= 2u) { beta = gam / gamPrev; denom = sums.y - beta * gam / alphaPrev; }
      if (i == 1u) { pp = gam; } else {
        let ddN = dd + alphaPrev * (2.0 * dp + alphaPrev * pp); dp = beta * (dp + alphaPrev * pp); pp = gam + beta * beta * pp; dd = ddN;
      }
      var alpha = 0.0;
      if (!(denom > 0.0)) {
        if (!tr) { stop = 1u; break; }
        edge = true;
      } else {
        alpha = gam / denom;
        edge = tr && dd + alpha * (2.0 * dp + alpha * pp) >= trust.y * trust.y;
      }
      if (edge) { alpha = pcgBoundary(dd, dp, pp, trust.y); }
      taken += 1.0;
      pv = u + beta * pv; sv = w + beta * sv; d += alpha * pv; r -= alpha * sv;
      dn = dd + alpha * (2.0 * dp + alpha * pp);
      if (edge) { stop = 1u; break; }
      u = nwPrecondition(M0, M1, r);
      gamPrev = gam; alphaPrev = alpha;
    }
    pcgStage(lane, u);
    workgroupBarrier();
    w = vec3f(0.0);
    if (own) {
      for (var k = 0u; k < K1; k++) {
        let b0 = nw[pcgBlock(blocks, k, 0u, lane)]; let b1 = nw[pcgBlock(blocks, k, 1u, lane)]; let b2 = nw[pcgBlock(blocks, k, 2u, lane)];
        let uj = pcgULocal(bitcast<u32>(b0.w));
        w += vec3f(dot(b0.xyz, uj), dot(b1.xyz, uj), dot(b2.xyz, uj));
      }
      for (var m = 0u; 2u * m < K2; m++) {
        let e = nw[bends + m * PCG_PATCH + lane];
        w += e.y * pcgULocal(bitcast<u32>(e.x)) + e.w * pcgULocal(bitcast<u32>(e.z));
      }
      ${contact ? 'if (!nwPinned(v)) { w = nwProjectP(v, w + ogcVertexContactHvLanes(v, 0u, 1u)); }' : ''}
    }
    // The reduction's barriers also order this iteration's reads of the staged u before the next one overwrites it;
    // pcgOut is read after the next iteration's uniform load (a barrier).
    ${reduce('vec4f(dot(r, u), dot(w, u), select(0.0, dot(r, r), i == 0u), 0.0)')}
    if (lane == 0u) { pcgOut = total; }
  }
  if (own) { nw[nseg(${S.d}u, v)] = vec4f(d, 0.0); }
  if (lead) {
    if (stop == 2u) { nw[nwControl()].x = 1.0; } else { nw[nwControl()].y = 1.0; }
    nw[nwControl() + 1u].w += taken;
    nw[nwTrustNorm()] = vec4f(dn, select(0.0, 1.0, edge), 0.0, 0.0);
  }
}

${contact ? `// The projected arc (contact): after the filter hook fixes the round's certificates, each trial's move
// delta_k(v) = Pi_v(x_v + 2^-k d_v) - x_v of vertex v, Pi_v onto v's certified set. Where Pi keeps the trial point the
// move is exactly 2^-k d_v; pinned vertices (d = 0) stay put.
@compute @workgroup_size(${workgroup}) fn nwProject(@builtin(global_invocation_id) id: vec3u) {
  let v = id.x;
  if (v >= P.n.x || nw[nwControl()].x != 0.0) { return; }
  let x = pos(v); let d = nwD(v);
  if (nwPinned(v)) { return; }
  let hasActive = nw[nseg(NW_PROJ + 1u, v)].w > 0.0;
  if (all(d == vec3f(0.0)) && !hasActive) { return; }
  var alpha = 1.0; var cut = false;
  // The done rule's travel test (w = 1) runs only on a trial that could end Newton: alpha |d_v| <= tol h~ (nwChoose's
  // small step needs alpha max_v |d_v| <= tol h~), so it is exact wherever done is decided and costs nothing far from it.
  // It runs on a cut trial, and on an uncut one of a vertex with active normals: the reduced direction d = P d runs
  // along the certificates binding v, so the arc no longer cuts it while a travel bound still holds it there (a vertex
  // pinned on a mandatory round's solid face beyond r, its multiplier positive, would otherwise end Newton far from
  // the solid and arm no new round), as on a vertex its active normals hold whole (d = P d = 0). Such a vertex is
  // marked as cut (w = 1); its stored moves are exactly alpha d.
  let tolMove = bitcast<f32>(P.newton3.w) * hEff(); let dl = length(d); let gv = nw[nseg(${S.g}u, v)].xyz;
  for (var k = 0u; k < ${CLOTH_NEWTON_TRIALS}u; k++) {
    var m = alpha * d; var held = 0.0;
    let y = x + m; let q = ogcProjectTrial(v, x, y, alpha);
    if (any(q != y)) {
      m = q - x; cut = true;
      if (alpha * dl <= tolMove && nwTravelHeld(v, x, y, alpha, gv)) { held = 1.0; }
    } else if (hasActive && alpha * dl <= tolMove && nwTravelHeld(v, x, y, alpha, gv)) { held = 1.0; cut = true; }
    nw[nseg(${S.moves}u + k, v)] = vec4f(m, held);
    alpha *= 0.5;
  }
  if (cut) { nw[nseg(${S.d}u, v)] = vec4f(d, 1.0); }
}
` : ''}
// Line search: G(x(alpha)) - G(x) in difference form for alpha = 2^-k, k < ${CLOTH_NEWTON_TRIALS}, with its rounding floor,
// and the slope g.(x(alpha) - x) of each trial; x(alpha) = x + alpha d, or with contact the projected arc (nwMove).
// Each vertex takes its own terms, its bending row and the faces whose first slot it is.
var<private> nwE: array<vec4f, 2>;
var<private> nwF: array<vec4f, 2>;
var<private> nwS: array<vec4f, 2>;
fn nwAdd(k: u32, e: vec2f) { nwE[k / 4u][k % 4u] += e.x; nwF[k / 4u][k % 4u] += e.y; }
@compute @workgroup_size(${workgroup}) fn nwTrial(@builtin(local_invocation_index) lane: u32, @builtin(workgroup_id) wg: vec3u, @builtin(num_workgroups) nwg: vec3u) {
  let ctl = nwControlUniform(lane);
  if (ctl.x != 0.0) { return; }
  let wgi = wgIndex(wg, nwg); let v = wgi * NW_WG + lane;
  nwE = array<vec4f, 2>(vec4f(0.0), vec4f(0.0)); nwF = array<vec4f, 2>(vec4f(0.0), vec4f(0.0)); nwS = array<vec4f, 2>(vec4f(0.0), vec4f(0.0));
  var dmax = 0.0; var cut = 0.0; var act = vec4f(0.0); var trav = array<vec4f, 2>(vec4f(0.0), vec4f(0.0));
  if (v < P.n.x) {
    let h = hEff(); let xv = pos(v); let dv = nwD(v); let gv = nw[nseg(${S.g}u, v)].xyz;
    dmax = dot(dv, dv);
    ${contact ? `let pr = nw[nseg(NW_PROJ + 1u, v)]; act = vec4f(select(0.0, 1.0, pr.w > 0.0), pr.w, pr.z, 0.0);` : ''}
    // The arc cuts v's full step (its certificates bound it): a stall of the whole arc is then no convergence.
    ${contact ? `cut = select(0.0, 1.0, nw[nseg(${S.d}u, v)].w != 0.0);
    // Per trial, whether a travel bound holds v there (nwProject; its moves are this iteration's only when cut).
    if (cut != 0.0) { for (var k = 0u; k < ${CLOTH_NEWTON_TRIALS}u; k++) { trav[k / 4u][k % 4u] = nw[nseg(${S.moves}u + k, v)].w; } }` : ''}
    let w = vertexRest(v).x / (h * h); let rv = xv - state[seg(S_Y, v)].xyz; let gs = grabSpring(v, xv);
    let damped = P.state1.z != 0u;
    var alpha = 1.0;
    for (var k = 0u; k < ${CLOTH_NEWTON_TRIALS}u; k++) {
      let d = nwMove(v, k, alpha);
      nwS[k / 4u][k % 4u] = dot(gv, d);
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
      let e11 = 0.5 * (dot(f0, f0) - 1.0); let e22 = 0.5 * (dot(f1, f1) - 1.0); let gam = dot(f0, f1);
      let s11 = (1.0 - c1.z) * c1.x; let s22 = (1.0 - c1.z) * c1.y;
      var T0 = vec4f(0.0); var T1 = vec4f(0.0); var rate = vec3f(0.0); var roundRate = vec3f(0.0);
      let sc = fr.y * fr.x / h;
      let faceDamped = damped && fr.y > 0.0;
      if (faceDamped) {
        T0 = state[P.state1.x + 2u * f]; T1 = state[P.state1.x + 2u * f + 1u];
        let u0 = x0 - state[seg(S_XHAT, t.x)].xyz; let u1 = x1 - state[seg(S_XHAT, t.y)].xyz; let u2 = x2 - state[seg(S_XHAT, t.z)].xyz;
        let q0 = a.x * u0 + a.y * u1 + a.z * u2; let q1 = b.x * u0 + b.y * u1 + b.z * u2;
        rate = vec3f(dot(T0.xyz, q0), dot(T1.xyz, q1), dot(T0.xyz, q1) + dot(T1.xyz, q0));
        let ru = vec2f(abs(a.x) * length(u0) + abs(a.y) * length(u1) + abs(a.z) * length(u2), abs(b.x) * length(u0) + abs(b.y) * length(u1) + abs(b.z) * length(u2));
        roundRate = vec3f(length(T0.xyz) * ru.x, length(T1.xyz) * ru.y, length(T0.xyz) * ru.y + length(T1.xyz) * ru.x);
      }
      // A face none of whose vertices the arc cuts moves by alpha (a.d, b.d): gathered once.
      let Q0 = nwD4(t.x); let Q1 = nwD4(t.y); let Q2 = nwD4(t.z);
      let linear = Q0.w == 0.0 && Q1.w == 0.0 && Q2.w == 0.0;
      let G0 = a.x * Q0.xyz + a.y * Q1.xyz + a.z * Q2.xyz; let G1 = b.x * Q0.xyz + b.y * Q1.xyz + b.z * Q2.xyz;
      var alpha = 1.0;
      for (var k = 0u; k < ${CLOTH_NEWTON_TRIALS}u; k++) {
        var df0 = alpha * G0; var df1 = alpha * G1;
        if (!linear) {
          let D0 = nwMove(t.x, k, alpha); let D1 = nwMove(t.y, k, alpha); let D2 = nwMove(t.z, k, alpha);
          df0 = a.x * D0 + a.y * D1 + a.z * D2; df1 = b.x * D0 + b.y * D1 + b.z * D2;
        }
        let d11 = dot(f0, df0) + 0.5 * dot(df0, df0); let d22 = dot(f1, df1) + 0.5 * dot(df1, df1);
        let dg = dot(f0, df1) + dot(df0, f1) + dot(df0, df1);
        let q11 = quadDelta(c0.x - select(0.0, s11, e11 < 0.0), c0.x - select(0.0, s11, e11 + d11 < 0.0), e11, d11);
        let q22 = quadDelta(c0.y - select(0.0, s22, e22 < 0.0), c0.y - select(0.0, s22, e22 + d22 < 0.0), e22, d22);
        let q12 = c0.z * (e11 * d22 + d11 * e22 + d11 * d22); let qs = c0.w * (gam * dg + 0.5 * dg * dg);
        let rounding = (c0.x + c0.z) * abs(d11) * (0.5 + e11) + (c0.y + c0.z) * abs(d22) * (0.5 + e22) + c0.w * abs(dg) * sqrt(max((1.0 + 2.0 * e11) * (1.0 + 2.0 * e22), 0.0));
        var out = fr.x * vec2f(q11 + q22 + q12 + qs, abs(q11) + abs(q22) + abs(q12) + abs(qs) + rounding);
        if (faceDamped) {
          let dr = vec3f(dot(T0.xyz, df0), dot(T1.xyz, df1), dot(T0.xyz, df1) + dot(T1.xyz, df0));
          let sigma = vec3f(T0.w * rate.x + c0.z * rate.y, c0.z * rate.x + T1.w * rate.y, c0.w * rate.z);
          let dsigma = vec3f(T0.w * dr.x + c0.z * dr.y, c0.z * dr.x + T1.w * dr.y, c0.w * dr.z);
          let lin = sc * dot(dr, sigma); let quad = 0.5 * sc * dot(dr, dsigma);
          out += vec2f(lin + quad, abs(lin) + quad + sc * dot(abs(dsigma), roundRate));
        }
        nwAdd(k, out);
        alpha *= 0.5;
      }
    }
    // Vertex v's bending rows (and their damping): each trial's stencil move R delta (alpha R d once a row when the
    // arc cuts none of its vertices), then its exact change.
    let cutV = nwD4(v).w != 0.0;
    for (var r = BEND_ROWS * v; r < BEND_ROWS * (v + 1u); r++) {
      let q = loadBend(r); var ds1 = vec3f(0.0); var linear = !cutV;
      for (var o = topo[P.topo0.z + r]; o < topo[P.topo0.z + r + 1u]; o++) { let Q = nwD4(bendJ(o)); ds1 += bendW(o) * (Q.xyz - dv); linear = linear && Q.w == 0.0; }
      alpha = 1.0;
      for (var k = 0u; k < ${CLOTH_NEWTON_TRIALS}u; k++) {
        var ds = alpha * ds1;
        if (!linear) {
          let mv = nwMove(v, k, alpha); ds = vec3f(0.0);
          for (var o = topo[P.topo0.z + r]; o < topo[P.topo0.z + r + 1u]; o++) { ds += bendW(o) * (nwMove(bendJ(o), k, alpha) - mv); }
        }
        nwAdd(k, bendDelta(q, ds));
        alpha *= 0.5;
      }
    }
    // Strain-limit edges whose first slot v is: exact in the relative move.
    if (P.flags.z != 0u) {
      for (var j = topo[P.topo1.x + v]; j < topo[P.topo1.x + v + 1u]; j++) {
        let entry = topo[P.topo1.y + j];
        if ((entry & 3u) != 0u) { continue; }
        let e = entry >> 2u; let ev = edges(e); let other = select(ev.x, ev.y, ev.x == v);
        let Qo = nwD4(other); let linear = !cutV && Qo.w == 0.0; let rel = dv - Qo.xyz;
        alpha = 1.0;
        for (var k = 0u; k < ${CLOTH_NEWTON_TRIALS}u; k++) {
          var m = alpha * rel;
          if (!linear) { m = nwMove(v, k, alpha) - nwMove(other, k, alpha); }
          nwAdd(k, edgeDelta(v, xv, e, m)); alpha *= 0.5;
        }
      }
    }
    ${contact ? `// Contact pairs v owns (each pair once over all vertices), at x and at every trial point x(alpha) (ogcX).
    let ec0 = ogcContactEnergyOwnedLanes(v, 0u, 1u);
    for (var k = 0u; k < ${CLOTH_NEWTON_TRIALS}u; k++) {
      nwTrialMove = k + 1u; let ec = ogcContactEnergyOwnedLanes(v, 0u, 1u); nwTrialMove = 0u;
      nwAdd(k, vec2f(ec - ec0, abs(ec0) + abs(ec)));
    }` : ''}
  }
  let base = nwPartial(0u);
  nwPartialSum(lane, wgi, nwE[0], base);
  nwPartialSum(lane, wgi, nwE[1], base + P.newton0.z);
  nwPartialSum(lane, wgi, nwF[0], base + 2u * P.newton0.z);
  nwPartialSum(lane, wgi, nwF[1], base + 3u * P.newton0.z);
  nwPartialSum(lane, wgi, nwS[0], base + 4u * P.newton0.z);
  nwPartialSum(lane, wgi, nwS[1], base + 5u * P.newton0.z);
  nwPartialSumMax(lane, wgi, vec2f(cut, dmax), base + 6u * P.newton0.z);
  ${contact ? `nwPartialSum(lane, wgi, act, base + 7u * P.newton0.z);
  nwPartialSum(lane, wgi, trav[0], base + 8u * P.newton0.z);
  nwPartialSum(lane, wgi, trav[1], base + 9u * P.newton0.z);` : ''}
}

// A trust-region iteration's model at its step s (the projected arc at alpha 1, nwMove trial 0): s.H s with the model's
// H (the solve's operator before the active-set projection: the assembled rows with the exact membrane, and contact's
// product), per patch into PCG partial array 2 (free once the solve is done: the next solve's dispatch 2 writes it
// before dispatch 3 reads it). nwChoose takes pred = -(g.s + s.H s / 2). One workgroup per patch; a line-search
// iteration returns at once.
@compute @workgroup_size(256) fn nwModel(@builtin(local_invocation_index) lane: u32, @builtin(workgroup_id) wg: vec3u, @builtin(num_workgroups) nwg: vec3u${sgArgs}) {
  let ctl = nwControlUniform(lane);
  if (!NW_HYBRID || ctl.x != 0.0) { return; }
  if (lane == 0u) { pcgTrust = nw[nwTrust()]; }
  if (workgroupUniformLoad(&pcgTrust).x == 0.0) { return; }
  let p = wgIndex(wg, nwg);
  let tw = pcgTable(p);
  let K1 = topo[tw]; let K2 = topo[tw + 1u]; let haloN = min(topo[tw + 2u], PCG_SHARED - PCG_PATCH); let halo = topo[tw + 3u];
  let blocks = topo[tw + 4u]; let bends = topo[tw + 5u]; let rows = topo[tw + 7u];
  pcgPatchIndex = p; pcgSrc = 1u;
  let own = lane < rows; let v = select(0u, pcgRowVertex(p * PCG_PATCH + lane), own);
  var sv = vec3f(0.0);
  if (own) { sv = nwMove(v, 0u, 1.0); }
  pcgStage(lane, sv);
  for (var h = lane; h < haloN; h += PCG_PATCH) { pcgStage(PCG_PATCH + h, nwMove(topo[halo + h], 0u, 1.0)); }
  workgroupBarrier();
  var w = vec3f(0.0); var wc = vec3f(0.0);
  if (own) {
    w = pcgRowProduct(lane, K1, K2, blocks, bends);
    ${contact ? 'if (!nwPinned(v)) { wc = ogcVertexContactHvLanes(v, 0u, 1u); }' : ''}
  }
  // s.H s, and of it contact's share (the model record, control + 6, reads both).
  ${reduce('vec4f(dot(sv, w + wc), dot(sv, wc), 0.0, 0.0)')}
  if (lane == 0u) { nw[pcgPartial(2u, p)] = total; }
}

// One workgroup: the largest trial whose move descends (g.(x(alpha) - x) < 0) and passes Armijo along it, and the next
// Newton iteration's PCG reset. If none does, Newton is done, unless the arc was cut (a vertex's certificates bound its
// full step): then the iterate stays, Newton runs on, and the re-detection round that follows (afterNewtonStep) gives
// the next iteration new certificates to step in.
@compute @workgroup_size(${workgroup}) fn nwChoose(@builtin(local_invocation_index) lane: u32) {
  let ctl = nwControlUniform(lane);
  if (ctl.x != 0.0) { return; }
  let base = nwPartial(0u);
  let E0 = nwSum(lane, base); let E1 = nwSum(lane, base + P.newton0.z);
  let F0 = nwSum(lane, base + 2u * P.newton0.z); let F1 = nwSum(lane, base + 3u * P.newton0.z);
  let S0 = nwSum(lane, base + 4u * P.newton0.z); let S1 = nwSum(lane, base + 5u * P.newton0.z);
  let sm = nwSumMax(lane, base + 6u * P.newton0.z);
  ${contact ? `let act = nwSum(lane, base + 7u * P.newton0.z);
  let T0 = nwSum(lane, base + 8u * P.newton0.z); let T1 = nwSum(lane, base + 9u * P.newton0.z);` : ''}
  var sHs = 0.0;
  var sHv = vec4f(0.0);
  if (NW_HYBRID) { sHv = nwSumN(lane, pcgPartial(2u, 0u), P.newton3.y); sHs = sHv.x; }
  if (lane != 0u) { return; }
  var chosen = 0.0; var alpha = 1.0; var held = 0.0;
  let trust = nw[nwTrust()]; let tn = nw[nwTrustNorm()]; var counts = nw[nwTrustCounts()];
  let region = NW_HYBRID && trust.x != 0.0;
  if (region) {
    // A trust-region iteration: the step s against the model's decrease pred = -(g.s + s.H s / 2), accepted when the
    // decrease is at least 1e-4 pred (Armijo's constant) less the energy change's rounding floor. Delta from rho =
    // ared / pred: rho < 0.25 (or no model decrease) ||d||_M / 4, rho > 0.75 with the solve on the boundary 2 Delta; a
    // change within rounding measures no rho and keeps Delta.
    let e = E0[0]; let fl = F0[0]; let pred = -(S0[0] + 0.5 * sHs);
    if (e <= 4.0 * EPS32 * fl - ${CLOTH_LINE_SEARCH.armijo} * max(pred, 0.0)) { chosen = 1.0; ${contact ? 'held = T0[0]; ' : ''}}
    var radius = trust.y; var rho = -1.0; var measured = true;
    if (pred > 0.0) { rho = -e / pred; measured = abs(e) > 4.0 * EPS32 * fl; }
    if (measured) {
      if (!(rho >= 0.25)) { radius = 0.25 * sqrt(tn.x); } else if (rho > 0.75 && tn.y != 0.0) { radius = 2.0 * radius; }
    }
    nw[nwTrust()] = vec4f(1.0, radius, 0.0, 0.0);
    // The model record (harness reads): g.s, s.H s, contact's share of it, and the energy change at s.
    nw[nwTrustModel()] = vec4f(S0[0], sHs, sHv.y, e);
    counts = vec4f(counts.x, counts.y + 1.0, counts.z + select(1.0, 0.0, chosen > 0.0), rho);
  } else {
    for (var k = 0u; k < ${CLOTH_NEWTON_TRIALS}u; k++) {
      let e = select(E1[k - 4u], E0[k], k < 4u); let fl = select(F1[k - 4u], F0[k], k < 4u); let sl = select(S1[k - 4u], S0[k], k < 4u);
      if (sl < 0.0 && e <= ${CLOTH_LINE_SEARCH.armijo} * sl + 4.0 * EPS32 * fl) { chosen = alpha; ${contact ? 'held = select(T1[k - 4u], T0[k], k < 4u); ' : ''}break; }
      alpha *= 0.5;
    }
    // The hybrid: two consecutive full steps start the trust region, Delta the last full step's ||d||_M. With contact
    // each must also move no vertex more than r / 2 (max |d_v|, which bounds the projected move): the contact scale
    // over which the barrier's model holds (C1 RESULT 40/41: at t25 the full steps of 1.6-2.6 cm inertial transients
    // started the trust region far outside it, and it crawled on Cauchy steps).
    if (NW_HYBRID) {
      let full = chosen == 1.0${contact ? ' && sqrt(sm.y) <= 0.5 * bitcast<vec4f>(ogcContacts[1]).x' : ''};
      counts.x = select(0.0, counts.x + 1.0, full);
      if (counts.x >= 2.0) { nw[nwTrust()] = vec4f(1.0, sqrt(tn.x), 0.0, 0.0); }
    }
  }
  if (NW_HYBRID) { nw[nwTrustCounts()] = counts; }
  ${contact ? `// The active set, and the vertices a travel bound holds at the accepted trial (no step: 0, the stall is its own signal).
  nw[nwControl() + 2u] = vec4f(act.xyz, held);` : ''}
  // Converged: no descending step, or (velocity tolerance, newton3.w) the accepted step moves no vertex faster than
  // tol h~ / h~ (alpha |d_v| bounds the projected move too: each certified set holds x_v). nwApply still applies a converged step's move.
  let tolV = bitcast<f32>(P.newton3.w);
  // A small step is no convergence while a travel bound holds a vertex there (the cap-aware done rule): Newton runs on
  // and the contact hook's re-detection gives those vertices new certificates about the iterate.
  // In a trust-region iteration only an interior step (the solve ended inside the region, tn.y 0) measures the Newton
  // step: a step on the boundary is as small as Delta, not as the iterate's distance to the solution.
  let small = chosen > 0.0 && chosen * sqrt(sm.y) <= tolV * hEff() && held == 0.0 && !(region && tn.y != 0.0);
  // (A rejected trust-region step is no stall: Delta shrinks and Newton runs on.)
  let stalled = chosen == 0.0 && !(sm.x > 0.0) && !region;
  nw[nwControl()] = vec4f(select(0.0, 1.0, stalled || small), 0.0, chosen, ctl.w + 1.0);
}

@compute @workgroup_size(${workgroup}) fn nwApply(@builtin(global_invocation_id) id: vec3u) {
  let v = id.x;
  let ctl = nw[nwControl()];
  if (v >= P.n.x) { return; }
  // The accepted trial's move (a converged step's too; once Newton is done d = 0 and this moves nothing). chosen = 2^-k exactly.
  // Newton done: d = 0 (its w too, so no stored move is read again), so the capped iterations left (all early exits, the
  // control record keeping its last step) move nothing and give the contact filter and re-detection nothing to act on.
  ${contact && activeSet ? `// The active set the next iteration starts from: the outward normals of the certificates binding the vertex's
  // accepted trial point (at a stall, the smallest trial's: the iterate stays) in this iteration's filter's set
  // (ogcProjectActive: up to three independent, the cap apart - the round's trust region bounds travel, not the set),
  // replacing the last iteration's: a normal from an earlier certificate set is not a constraint of the next one
  // (re-detection runs after nwApply and replaces the set). A held vertex (no move) blocks its whole direction d.
  if (ctl.x == 0.0 && !nwPinned(v)) {
    let D = nwD4(v);
    var a = exp2(-f32(${CLOTH_NEWTON_TRIALS - 1}u));
    if (ctl.z > 0.0) { a = ctl.z; }
    let x = pos(v); let act = ogcProjectActive(v, x, x + a * D.xyz, a);
    var N = array<vec3f, 3>(act.n0, act.n1, act.n2); var cnt = act.count;
    if (act.held) { cnt = select(0u, 1u, any(D.xyz != vec3f(0.0))); N[0] = select(vec3f(0.0), normalize(D.xyz), cnt > 0u); }
    for (var i = 0u; i < 3u; i++) { nw[nseg(NW_ACTIVE + i, v)] = select(vec4f(0.0), vec4f(N[i], 1.0), i < cnt); }
  }` : ''}
  if (ctl.z > 0.0) { positions[v] = vec4f(pos(v) + nwMove(v, u32(round(-log2(ctl.z))), ctl.z), positions[v].w); }
  if (ctl.x != 0.0) { nw[nseg(${S.d}u, v)] = vec4f(0.0); }
}

// After each applied Newton iteration (not once Newton is done): the AVBD dual update of every strain-limit edge, so
// the next iteration's gradient carries the new multipliers and the residual test sees any constraint still violated.
// The method of multipliers steps a dual only where the primal could respond: with contact, not in an iteration that
// moved nothing (the stall), and not on an edge an endpoint of which the arc cut (its chosen move is not alpha d) or the
// active set holds (d projected off its normals: the cut the reduced direction no longer makes), since there the
// violation measures the projection, not the multiplier, and ascent would ramp lambda and k without bound.
@compute @workgroup_size(${workgroup}) fn nwEdgeDuals(@builtin(global_invocation_id) id: vec3u) {
  let ctl = nw[nwControl()];
  // Not after an iteration that moved nothing (a stall, or a rejected trust-region step).
  if (ctl.x != 0.0 || ctl.z == 0.0) { return; }${contact ? `
  if (id.x >= P.n.z) { return; }
  let ev = edges(id.x);
  if (nwChosenCut(ev.x, ctl.z) || nwChosenCut(ev.y, ctl.z)) { return; }` : ''}
  updateEdgeDual(id.x);
}${contact ? `
// Whether vertex v's move at the chosen trial alpha (a power of two, as nwApply reads it) is not free: an active normal
// holds it, or the arc cut it.
fn nwChosenCut(v: u32, alpha: f32) -> bool {
  if (nw[nseg(NW_PROJ + 1u, v)].w > 0.0) { return true; }
  let D = nwD4(v);
  if (D.w == 0.0) { return false; }
  return any(nw[nseg(${S.moves}u + u32(round(-log2(alpha))), v)].xyz != alpha * D.xyz);
}` : ''}
`;
}
