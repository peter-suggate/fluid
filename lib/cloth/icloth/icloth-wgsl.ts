/**
 * I-Cloth's collision kernels (Tang et al. 2018, ACM TOG 37(6) 204), and the proximity block the Newton solver
 * compiles in (gpu-solver-wgsl.ts proximityBlock). Buffers and the step's dispatch order are described in icloth.ts.
 *
 * One record format serves proximity pairs, solid planes and impacts: up to four vertices, a fixed unit normal n and
 * signed weights w, C(x) = n . sum_i w_i x_i - d0. Vertex-face: ids (v, a, b, c), w (1, -b_a, -b_b, -b_c); edge-edge:
 * ids (p0, p1, q0, q1), w (-(1 - s), -s, 1 - t, t); a solid plane: ids (v, NONE, NONE, NONE), w (1, 0, 0, 0).
 */
import { OGC_SOLID_WGSL } from '../contact/ogc-wgsl';

/** Words of the control block at the head of the constraint buffer. */
export const ICLOTH_CTL_WORDS = 384;
/** Control words (u32; floats as bits). */
export const ICLOTH_CTL = {
  status: 0, proxCount: 1, impactCount: 2, hashCount: 3, roundHits: 4, roundStart: 5, clean: 6, gdDone: 7, gdFinal: 8,
  /** The round's solve as it ended: 0 converged, 1 at the iteration cap, 2 at the cap with an impact still violated. */
  gdOutcome: 9,
  /** Reference displacement (three f32): the CCD is exact under a common translation of every end position. */
  dref: 12,
  /** Per step: rounds that found impacts, gradient-descent iterations, impacts the verification still finds (pairs,
   * solid), proximity records (pairs, solid), the last solve's largest constraint violation (f32 bits). */
  roundsUsed: 16, gdIterations: 17, remainingPairs: 18, remainingSolid: 19, proxPairs: 20, proxSolid: 21, maxViolation: 22,
  /** Since the state was uploaded: steps, steps that ended with impacts left, solves that hit the iteration cap, degenerate pairs. */
  steps: 23, uncleanSteps: 24, cappedSolves: 25, degenerate: 26, stalledInfeasible: 27,
  /** The halt latch: set at the end of a step that failed. Every later dispatch returns at its first read. */
  halt: 28,
  /** Per round (64 words each): impacts found, moving impact-zone vertices, the largest displacement (f32 bits). */
  impactsRound: 64, zoneVertices: 128, maxDisp: 192,
} as const;
export const ICLOTH_MAX_ROUNDS = 62;
/** Status bits (sticky). The failing ones halt the stage on the GPU and the scene (ICLOTH_FAILING); the rest are notes. */
export const ICLOTH_STATUS = {
  hashOverflow: 1, cellSpan: 2, proxOverflow: 4, proxListOverflow: 8, impactOverflow: 16, impactListOverflow: 32,
  /** Impacts remain after the last round: the step is not penetration-free. Failing: the stage halts at the step's end. */
  roundCap: 64,
  /** An impact-zone solve ran to its iteration cap. */
  iterationCap: 128,
  solidBounds: 256, nonFinite: 512,
  /** A pair whose geometry gave no normal (coincident features, a collapsed triangle): no constraint was written. */
  degenerate: 1024,
} as const;
export const ICLOTH_FAILING = ICLOTH_STATUS.hashOverflow | ICLOTH_STATUS.cellSpan | ICLOTH_STATUS.proxOverflow | ICLOTH_STATUS.proxListOverflow
  | ICLOTH_STATUS.impactOverflow | ICLOTH_STATUS.impactListOverflow | ICLOTH_STATUS.solidBounds | ICLOTH_STATUS.nonFinite | ICLOTH_STATUS.roundCap;
/** Dispatch-record mode flags (above the mode byte): the solve's last gradient; a closing detection that only counts. */
export const ICLOTH_MODE_LAST = 256, ICLOTH_MODE_COUNT_ONLY = 512;
export const ICLOTH_RECORD_WORDS = 16;
export const ICLOTH_MODE = { proximity: 0, ccd: 1, verify: 2 } as const;
export const ICLOTH_KIND = { vertexFace: 0, edgeEdge: 1, solid: 2, body: 3 } as const;
export const ICLOTH_WG = 64;
/** Work segments (vec4 per vertex): the objective's anchor X0, three rotating iterates, two alternating directions (xyz) with their preconditioner (w). */
export const ICLOTH_WORK = { x0: 0, xr: 1, d: 4, segments: 6 } as const;
/** Partial records per workgroup: trial energies (2), their rounding floors (2), (slope, max violation, max |direction|). */
export const ICLOTH_PARTIALS = 5;
export const ICLOTH_TRIALS = 8;
export const ICLOTH_BINDING = { params: 0, record: 1, positions: 2, state: 3, mesh: 4, solid: 5, cons: 6, work: 7, contactState: 8 } as const;
export const ICLOTH_PARAM_WORDS = 48;

/**
 * The impact-zone solver's constants: the penalty mu and the line search's Armijo constant (the paper's, section 5.2),
 * the objective's scale, Chebyshev's rho and first iteration, and the violation (as a fraction of the target
 * separation delta) under which a stalled solve has converged.
 *
 * Three deviations from the paper's Algorithm 3, each measured in Float64 on single pairs with a closed form and on
 * two stacked sheets (fluid-scratch/icloth/zone-explore.mts, zone-explore2.mts):
 *  - The objective is `objective` mu / 2 sum_i (m_i / mean m) |X_i - X0_i|^2, not sum_i m_i |X_i - X0_i|^2 / sum_i m_i.
 *    Scaling the objective does not move the constrained minimum, but at the paper's scale the penalty outweighs it by
 *    mu sum m / m_i (1e5 and up): the Jacobi direction is the penalty's alone (each vertex moves by gap / w_i, the
 *    least-weighted vertex the furthest), the solve stops feasible after 5 iterations and the mass-weighted minimum is
 *    never approached (a vertex-face pair: 1e-4 m off the closed form on a 4e-4 m gap).
 *  - The multipliers are updated when the descent stalls (no accepted step, or one below the larger of the threshold
 *    and `inexact` times the largest violation: the inner minimisation is only worth solving as far as the
 *    constraints are from holding), not every iteration: updated every iteration, an edge-edge pair cycles without
 *    converging at scale 1, and the sheets stay violated at scale 0.25 and below. With the update at a stall every
 *    case converges to within the feasibility tolerance of the closed form (zone-explore3.mts: pairs in 3-20
 *    iterations, sheets 0.3 mm deep in 12-16, 3 mm in 29, 50 mm in 91-109; with `inexact` 0 the 50 mm sheets take 740).
 *  - Chebyshev acceleration is off (rho 0): with the paper's rho = 0.9992 the iteration diverges at this scale, and
 *    with rho = 0.7-0.9 it stalls violated on the sheets.
 */
export const ICLOTH_SOLVER = { mu: 1e3, armijo: 1e-4, objective: 0.25, inexact: 1, rho: 0, chebyshevStart: 10, feasible: 0.1 } as const;

const C = ICLOTH_CTL, ST = ICLOTH_STATUS;

const GEOMETRY = /* wgsl */`
// Barycentric coordinates of the point of triangle (a, b, c) closest to p (Ericson 5.1.5).
fn icClosestTri(p: vec3f, a: vec3f, b: vec3f, c: vec3f) -> vec3f {
  let ab = b - a; let ac = c - a; let ap = p - a;
  let d1 = dot(ab, ap); let d2 = dot(ac, ap);
  if (d1 <= 0.0 && d2 <= 0.0) { return vec3f(1.0, 0.0, 0.0); }
  let bp = p - b; let d3 = dot(ab, bp); let d4 = dot(ac, bp);
  if (d3 >= 0.0 && d4 <= d3) { return vec3f(0.0, 1.0, 0.0); }
  let vc = d1 * d4 - d3 * d2;
  if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) { let v = d1 / max(d1 - d3, 1.0e-30); return vec3f(1.0 - v, v, 0.0); }
  let cp = p - c; let d5 = dot(ab, cp); let d6 = dot(ac, cp);
  if (d6 >= 0.0 && d5 <= d6) { return vec3f(0.0, 0.0, 1.0); }
  let vb = d5 * d2 - d1 * d6;
  if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) { let w = d2 / max(d2 - d6, 1.0e-30); return vec3f(1.0 - w, 0.0, w); }
  let va = d3 * d6 - d5 * d4;
  if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) { let w = (d4 - d3) / max((d4 - d3) + (d5 - d6), 1.0e-30); return vec3f(0.0, 1.0 - w, w); }
  let sum = va + vb + vc;
  if (!(sum > 0.0)) { return vec3f(1.0, 0.0, 0.0); }
  let v = vb / sum; let w = vc / sum;
  return vec3f(1.0 - v - w, v, w);
}
// Parameters (s, t) of the closest points of segments p1 + s (q1 - p1) and p2 + t (q2 - p2) (Ericson 5.1.9).
fn icClosestSeg(p1: vec3f, q1: vec3f, p2: vec3f, q2: vec3f) -> vec2f {
  let d1 = q1 - p1; let d2 = q2 - p2; let r = p1 - p2;
  let a = dot(d1, d1); let e = dot(d2, d2); let f = dot(d2, r);
  var s = 0.0; var t = 0.0;
  if (a <= 1.0e-30 && e <= 1.0e-30) { return vec2f(0.0); }
  if (a <= 1.0e-30) { t = clamp(f / e, 0.0, 1.0); }
  else {
    let c = dot(d1, r);
    if (e <= 1.0e-30) { s = clamp(-c / a, 0.0, 1.0); }
    else {
      let b = dot(d1, d2); let denom = a * e - b * b;
      if (denom > 1.0e-10 * a * e) { s = clamp((b * f - c * e) / denom, 0.0, 1.0); }
      t = (b * s + f) / e;
      if (t < 0.0) { t = 0.0; s = clamp(-c / a, 0.0, 1.0); }
      else if (t > 1.0) { t = 1.0; s = clamp((b - c) / a, 0.0, 1.0); }
    }
  }
  return vec2f(s, t);
}
// The pair's signed weights and d = sum_i w_i x_i (the vector between its closest points).
struct IcFeat { w: vec4f, d: vec3f }
fn icFeat(kind: u32, x0: vec3f, x1: vec3f, x2: vec3f, x3: vec3f) -> IcFeat {
  if (kind == 0u) {
    let b = icClosestTri(x0, x1, x2, x3);
    return IcFeat(vec4f(1.0, -b.x, -b.y, -b.z), x0 - (b.x * x1 + b.y * x2 + b.z * x3));
  }
  let st = icClosestSeg(x0, x1, x2, x3);
  return IcFeat(vec4f(st.x - 1.0, -st.x, 1.0 - st.y, st.y), (x2 + st.y * (x3 - x2)) - (x0 + st.x * (x1 - x0)));
}
// Continuous collision of a vertex-face (kind 0: vertex a0, face a1 a2 a3) or edge-edge (kind 1: a0 a1 against a2 a3)
// pair moving linearly from a* to b*. The coplanarity f(t) = A . (B x C) is a cubic; its roots in [0, 1] are bracketed
// between the stationary points and bisected, and a root is an impact where the features are within eta there. A pair
// with no impact that ends closer than dTol is one too (so a step never ends inside the tolerance of the tests).
// hit 1: an impact at a root, n the features' normal there on the side the pair came from; 2: the end distance, n
// along the closest points; 3: an impact whose geometry gives no normal.
struct IcHit { hit: u32, n: vec3f, w: vec4f }
fn icCubic(c: vec4f, t: f32) -> f32 { return ((c.w * t + c.z) * t + c.y) * t + c.x; }
fn icSign(x: f32) -> f32 { return select(select(0.0, -1.0, x < 0.0), 1.0, x > 0.0); }
fn icCcd(kind: u32, a0: vec3f, a1: vec3f, a2: vec3f, a3: vec3f, b0: vec3f, b1: vec3f, b2: vec3f, b3: vec3f, eta: f32, dTol: f32) -> IcHit {
  var A0: vec3f; var B0: vec3f; var C0: vec3f; var A1: vec3f; var B1: vec3f; var C1: vec3f;
  if (kind == 0u) { A0 = a0 - a1; B0 = a2 - a1; C0 = a3 - a1; A1 = b0 - b1; B1 = b2 - b1; C1 = b3 - b1; }
  else { A0 = a2 - a0; B0 = a1 - a0; C0 = a3 - a2; A1 = b2 - b0; B1 = b1 - b0; C1 = b3 - b2; }
  let dA = A1 - A0; let dB = B1 - B0; let dC = C1 - C0;
  let N0 = cross(B0, C0); let N1 = cross(dB, C0) + cross(B0, dC); let N2 = cross(dB, dC);
  let c = vec4f(dot(A0, N0), dot(A0, N1) + dot(dA, N0), dot(A0, N2) + dot(dA, N1), dot(dA, N2));
  // The stationary points of f in (0, 1), ascending.
  var bp = array<f32, 4>(0.0, 1.0, 1.0, 1.0); var nb = 1u;
  let qa = 3.0 * c.w; let qb = 2.0 * c.z; let qc = c.y;
  var r1 = -1.0; var r2 = -1.0;
  if (qa != 0.0) {
    let disc = qb * qb - 4.0 * qa * qc;
    if (disc > 0.0) {
      let q = -0.5 * (qb + select(-1.0, 1.0, qb >= 0.0) * sqrt(disc));
      r1 = q / qa;
      if (q != 0.0) { r2 = qc / q; }
    }
  } else if (qb != 0.0) { r1 = -qc / qb; }
  if (!(r1 > 0.0 && r1 < 1.0)) { r1 = -1.0; }
  if (!(r2 > 0.0 && r2 < 1.0)) { r2 = -1.0; }
  let lo = min(r1, r2); let hi = max(r1, r2);
  if (lo > 0.0) { bp[nb] = lo; nb++; }
  if (hi > 0.0 && hi != lo) { bp[nb] = hi; nb++; }
  bp[nb] = 1.0;
  var out = IcHit(0u, vec3f(0.0), vec4f(0.0));
  for (var i = 0u; i < nb; i++) {
    var tl = bp[i]; var th = bp[i + 1u];
    let fl = icCubic(c, tl); let fh = icCubic(c, th);
    let sl = icSign(fl); let sh = icSign(fh);
    var t = -1.0; var side = sl;
    if (sl == 0.0) { t = tl; }
    else if (sl * sh < 0.0) {
      for (var k = 0u; k < 26u; k++) {
        let tm = 0.5 * (tl + th);
        if (icSign(icCubic(c, tm)) == sl) { tl = tm; } else { th = tm; }
      }
      t = th;
    }
    if (t < 0.0) { continue; }
    let F = icFeat(kind, mix(a0, b0, t), mix(a1, b1, t), mix(a2, b2, t), mix(a3, b3, t));
    if (dot(F.d, F.d) < eta * eta) {
      let N = N0 + t * (N1 + t * N2); let Bt = B0 + t * dB; let Ct = C0 + t * dC;
      let nn = dot(N, N);
      if (side == 0.0) { side = icSign(dot(icFeat(kind, a0, a1, a2, a3).d, N)); }
      if (nn > 1.0e-10 * dot(Bt, Bt) * dot(Ct, Ct) && side != 0.0) { out.hit = 1u; out.n = side * N * inverseSqrt(nn); out.w = F.w; return out; }
      // No normal at the impact (parallel edges, a collapsed face): the direction the features lay apart at the start.
      let Fs = icFeat(kind, a0, a1, a2, a3); let ds = dot(Fs.d, Fs.d);
      if (ds > 1.0e-20) { out.hit = 1u; out.n = Fs.d * inverseSqrt(ds); out.w = F.w; return out; }
      out.hit = 3u; return out;
    }
  }
  let Fe = icFeat(kind, b0, b1, b2, b3); let de = dot(Fe.d, Fe.d);
  if (de < dTol * dTol) {
    if (de > 1.0e-20) { out.hit = 2u; out.n = Fe.d * inverseSqrt(de); out.w = Fe.w; }
    else { out.hit = 3u; }
  }
  return out;
}
`;

/** The collision module (every kernel of IClothCollision). */
export const ICLOTH_WGSL = /* wgsl */`
struct Params {
  n: vec4u,     // N, T, hash mask, workgroups of N
  cap: vec4u,   // proximity entries per vertex, proximity records, impact entries per vertex, impact records
  off0: vec4u,  // vertex flags, proximity lists, proximity records, impact lists
  off1: vec4u,  // impact records, hash heads, hash pool, hash pool entries
  st: vec4u,    // state segment stride, x_t segment, y segment, rounds
  sol: vec4u,   // static field base (0: none), body field base (0: none), pose table, impact iterations
  wk: vec4u,    // work segment stride, partials, vertex records (mesh), unused
  f0: vec4f,    // r, kc, delta (the impacts' target separation), dTol (the end-distance tolerance)
  f1: vec4f,    // eta (the root test's distance), proximity range, cell base, cloth-cloth friction
  f2: vec4f,    // dt, friction regularisation length, mu (augmented Lagrangian), mean vertex mass
  f3: vec4f,    // Armijo constant, step threshold (m), feasibility tolerance (m), unused
}
struct Record { a: vec4u, b: vec4f } // (round, iteration, pose index, mode | final << 8), (Chebyshev omega, ...)
@group(0) @binding(${ICLOTH_BINDING.params}) var<uniform> P: Params;
@group(0) @binding(${ICLOTH_BINDING.record}) var<uniform> R: Record;
@group(0) @binding(${ICLOTH_BINDING.positions}) var<storage, read_write> positions: array<vec4f>;
@group(0) @binding(${ICLOTH_BINDING.state}) var<storage, read> state: array<vec4f>;
@group(0) @binding(${ICLOTH_BINDING.mesh}) var<storage, read> mesh: array<u32>;
@group(0) @binding(${ICLOTH_BINDING.solid}) var<storage, read> solid: array<u32>;
@group(0) @binding(${ICLOTH_BINDING.cons}) var<storage, read_write> cons: array<atomic<u32>>;
@group(0) @binding(${ICLOTH_BINDING.work}) var<storage, read_write> work: array<vec4f>;
@group(0) @binding(${ICLOTH_BINDING.contactState}) var<storage, read_write> contactState: array<u32>;

const NONE = 0xffffffffu;
const OGC_NONE = 0xffffffffu;
const EPS32: f32 = 1.1920929e-7;
const WG = ${ICLOTH_WG}u;
const TRIALS = ${ICLOTH_TRIALS}u;
const MODE_PROX = ${ICLOTH_MODE.proximity}u; const MODE_CCD = ${ICLOTH_MODE.ccd}u; const MODE_VERIFY = ${ICLOTH_MODE.verify}u;
const CT_STATUS = ${C.status}u; const CT_PROX = ${C.proxCount}u; const CT_IMPACT = ${C.impactCount}u; const CT_HASH = ${C.hashCount}u;
const CT_ROUND_HITS = ${C.roundHits}u; const CT_ROUND_START = ${C.roundStart}u; const CT_CLEAN = ${C.clean}u; const CT_GD_DONE = ${C.gdDone}u;
const CT_GD_FINAL = ${C.gdFinal}u; const CT_GD_OUTCOME = ${C.gdOutcome}u; const CT_DREF = ${C.dref}u; const CT_ROUNDS_USED = ${C.roundsUsed}u; const CT_GD_ITERS = ${C.gdIterations}u;
const CT_REMAIN_PAIRS = ${C.remainingPairs}u; const CT_REMAIN_SOLID = ${C.remainingSolid}u; const CT_PROX_PAIRS = ${C.proxPairs}u;
const CT_PROX_SOLID = ${C.proxSolid}u; const CT_MAX_VIOL = ${C.maxViolation}u; const CT_STEPS = ${C.steps}u; const CT_UNCLEAN = ${C.uncleanSteps}u;
const CT_CAPPED = ${C.cappedSolves}u; const CT_DEGENERATE = ${C.degenerate}u; const CT_STALLED = ${C.stalledInfeasible}u;
const CT_IMPACTS_ROUND = ${C.impactsRound}u; const CT_ZONE = ${C.zoneVertices}u; const CT_MAX_DISP = ${C.maxDisp}u;
const ST_HASH = ${ST.hashOverflow}u; const ST_SPAN = ${ST.cellSpan}u; const ST_PROX = ${ST.proxOverflow}u; const ST_PROX_LIST = ${ST.proxListOverflow}u;
const ST_IMPACT = ${ST.impactOverflow}u; const ST_IMPACT_LIST = ${ST.impactListOverflow}u; const ST_ROUND_CAP = ${ST.roundCap}u;
const ST_ITER_CAP = ${ST.iterationCap}u; const ST_BOUNDS = ${ST.solidBounds}u; const ST_NONFINITE = ${ST.nonFinite}u; const ST_DEGENERATE = ${ST.degenerate}u;
const W_X0 = ${ICLOTH_WORK.x0}u; const W_XR = ${ICLOTH_WORK.xr}u; const W_D = ${ICLOTH_WORK.d}u;
const PARTIALS = ${ICLOTH_PARTIALS}u;
const CT_HALT = ${C.halt}u; const ST_FAILING = ${ICLOTH_FAILING}u; const ST_HARD = ${ICLOTH_FAILING & ~ST.roundCap}u;

fn ld(i: u32) -> u32 { return atomicLoad(&cons[i]); }
fn ldf(i: u32) -> f32 { return bitcast<f32>(atomicLoad(&cons[i])); }
fn sto(i: u32, x: u32) { atomicStore(&cons[i], x); }
fn stf(i: u32, x: f32) { atomicStore(&cons[i], bitcast<u32>(x)); }
fn ld3(i: u32) -> vec3f { return vec3f(ldf(i), ldf(i + 1u), ldf(i + 2u)); }
fn ld4(i: u32) -> vec4f { return vec4f(ldf(i), ldf(i + 1u), ldf(i + 2u), ldf(i + 3u)); }
fn ld4u(i: u32) -> vec4u { return vec4u(ld(i), ld(i + 1u), ld(i + 2u), ld(i + 3u)); }
fn icMode() -> u32 { return R.a.w & 255u; }
// The guard. A pool overflow, a non-finite pose or a vertex outside the solids' lattice stops the stage where it is
// found; a step whose closing detection still finds impacts latches the halt at its end (icContactState). Either way
// every later dispatch returns at its first read: nothing is computed from a state known to be invalid.
fn halted() -> bool { return ld(CT_HALT) != 0u || (ld(CT_STATUS) & ST_HARD) != 0u; }
// Inside a candidate walk: also once the closing detection has counted more impacts than the impact pool holds.
fn overrun() -> bool { return halted() || ld(CT_REMAIN_PAIRS) > P.cap.w; }
fn roundCapped() { if ((R.a.w & ${ICLOTH_MODE_COUNT_ONLY}u) == 0u) { atomicOr(&cons[CT_STATUS], ST_ROUND_CAP); } }
fn xt(i: u32) -> vec3f { return state[P.st.y * P.st.x + i].xyz; }
fn yt(i: u32) -> vec3f { return state[P.st.z * P.st.x + i].xyz; }
fn triOf(t: u32) -> vec4u { return vec4u(mesh[4u * t], mesh[4u * t + 1u], mesh[4u * t + 2u], mesh[4u * t + 3u]); }
fn massOf(v: u32) -> f32 { return bitcast<f32>(mesh[P.wk.z + 2u * v]); }
fn pinned(v: u32) -> bool { return (mesh[P.wk.z + 2u * v + 1u] & 1u) != 0u; }
fn flagOf(v: u32) -> u32 { return ld(P.off0.x + v); }
fn dref() -> vec3f { return ld3(CT_DREF); }
fn finite3(x: vec3f) -> bool { return all(abs(x) <= vec3f(3.0e38)); }
fn wseg(s: u32, v: u32) -> u32 { return s * P.wk.x + v; }
${GEOMETRY}
fn ogcSolidBase() -> u32 { return P.sol.x; }
fn ogcSolidWord(i: u32) -> u32 { return solid[i]; }
${OGC_SOLID_WGSL}
// The moving solid's pose at the start (k = 0) and end (k = 1) of the step in flight (the frame's pose table).
fn icPose(k: u32) -> OgcPose {
  let b = P.sol.z + OGC_POSE_WORDS * (R.a.z + k);
  return OgcPose(ogcWordV(b), mat3x3<f32>(ogcWordV(b + 4u), ogcWordV(b + 8u), ogcWordV(b + 12u)));
}

// ---------------------------------------------------------------- records
fn listPush(base: u32, K: u32, v: u32, rec: u32) -> bool {
  let b = base + v * (K + 1u);
  let k = atomicAdd(&cons[b], 1u);
  if (k >= K) { return false; }
  sto(b + 1u + k, rec + 1u);
  return true;
}
fn writeRecord(b: u32, ids: vec4u, n: vec3f, d0: f32, w: vec4f) {
  sto(b, ids.x); sto(b + 1u, ids.y); sto(b + 2u, ids.z); sto(b + 3u, ids.w);
  stf(b + 4u, n.x); stf(b + 5u, n.y); stf(b + 6u, n.z); stf(b + 7u, d0);
  stf(b + 8u, w.x); stf(b + 9u, w.y); stf(b + 10u, w.z); stf(b + 11u, w.w);
}
// A proximity record: us the solid surface's displacement over the step, muN the friction coefficient times the
// lagged normal force.
fn proxAppend(ids: vec4u, n: vec3f, d0: f32, w: vec4f, us: vec3f, muN: f32) {
  let r = atomicAdd(&cons[CT_PROX], 1u);
  if (r >= P.cap.y) { atomicOr(&cons[CT_STATUS], ST_PROX); return; }
  let b = P.off0.z + ${ICLOTH_RECORD_WORDS}u * r;
  writeRecord(b, ids, n, d0, w);
  stf(b + 12u, us.x); stf(b + 13u, us.y); stf(b + 14u, us.z); stf(b + 15u, muN);
  for (var k = 0u; k < 4u; k++) {
    if (ids[k] != NONE && !listPush(P.off0.y, P.cap.x, ids[k], r)) { atomicOr(&cons[CT_STATUS], ST_PROX_LIST); }
  }
}
// An impact: the pair's record of an earlier round of this step takes the new plane and keeps its multiplier, else a
// new record (its vertices join the impact zone). Each pair is tested by one thread, and only records complete at the
// round's start are matched.
fn impactUpsert(kind: u32, ids: vec4u, n: vec3f, d0: f32, w: vec4f) {
  atomicAdd(&cons[CT_ROUND_HITS], 1u);
  let start = ld(CT_ROUND_START);
  let lb = P.off0.w + ids.x * (P.cap.z + 1u);
  let cnt = min(ld(lb), P.cap.z);
  for (var k = 0u; k < cnt; k++) {
    let e = ld(lb + 1u + k);
    if (e == 0u || e - 1u >= start) { continue; }
    let b = P.off1.x + ${ICLOTH_RECORD_WORDS}u * (e - 1u);
    if (ld(b + 14u) == kind && all(ld4u(b) == ids)) { writeRecord(b, ids, n, d0, w); return; }
  }
  let r = atomicAdd(&cons[CT_IMPACT], 1u);
  if (r >= P.cap.w) { atomicOr(&cons[CT_STATUS], ST_IMPACT); return; }
  let b = P.off1.x + ${ICLOTH_RECORD_WORDS}u * r;
  writeRecord(b, ids, n, d0, w);
  stf(b + 12u, 0.0); stf(b + 13u, 0.0); sto(b + 14u, kind); sto(b + 15u, 0u);
  for (var k = 0u; k < 4u; k++) {
    if (ids[k] == NONE) { continue; }
    if (!listPush(P.off0.w, P.cap.z, ids[k], r)) { atomicOr(&cons[CT_STATUS], ST_IMPACT_LIST); }
    atomicOr(&cons[P.off0.x + ids[k]], 2u);
  }
}

// ---------------------------------------------------------------- step begin
// Before the step's proximity detection: empty lists and flags; vertex 0 resets the step's control words.
@compute @workgroup_size(${ICLOTH_WG}) fn icStepBegin(@builtin(global_invocation_id) id: vec3u) {
  let v = id.x;
  if (v >= P.n.x || halted()) { return; }
  sto(P.off0.x + v, 0u); sto(P.off0.y + v * (P.cap.x + 1u), 0u); sto(P.off0.w + v * (P.cap.z + 1u), 0u);
  if (v == 0u) {
    sto(CT_PROX, 0u); sto(CT_IMPACT, 0u); sto(CT_CLEAN, 0u); sto(CT_GD_DONE, 0u); sto(CT_GD_FINAL, 0u); sto(CT_ROUND_START, 0u);
    sto(CT_ROUNDS_USED, 0u); sto(CT_GD_ITERS, 0u); sto(CT_REMAIN_PAIRS, 0u); sto(CT_REMAIN_SOLID, 0u); sto(CT_PROX_PAIRS, 0u);
    sto(CT_PROX_SOLID, 0u); sto(CT_MAX_VIOL, 0u);
    for (var i = 0u; i < 64u; i++) { sto(CT_IMPACTS_ROUND + i, 0u); sto(CT_ZONE + i, 0u); sto(CT_MAX_DISP + i, 0u); }
    atomicAdd(&cons[CT_STEPS], 1u);
  }
}
// Before the first round: the reference displacement (vertex 0's) and the largest displacement relative to it, which
// sizes the round's hash cells.
@compute @workgroup_size(${ICLOTH_WG}) fn icCcdBegin(@builtin(global_invocation_id) id: vec3u) {
  let v = id.x;
  if (v >= P.n.x || halted()) { return; }
  let dr = positions[0].xyz - xt(0u);
  let x = positions[v].xyz;
  if (!finite3(x)) { atomicOr(&cons[CT_STATUS], ST_NONFINITE); return; }
  let d = length(x - xt(v) - dr);
  if (d < 1.0e30) { atomicMax(&cons[CT_MAX_DISP], bitcast<u32>(d)); }
  if (v == 0u) { stf(CT_DREF, dr.x); stf(CT_DREF + 1u, dr.y); stf(CT_DREF + 2u, dr.z); }
}

// ---------------------------------------------------------------- broad phase: a hash of the triangles' boxes
fn cellSize() -> f32 {
  if (icMode() == MODE_PROX) { return P.f1.z + P.f1.y; }
  return P.f1.z + ldf(CT_MAX_DISP + R.a.x) + 4.0 * P.f0.w;
}
struct Box { lo: vec3f, hi: vec3f }
// Triangle corners: start (x_t) and, for the swept tests, end (the current positions less the reference displacement).
var<private> XA: array<vec3f, 6>;
var<private> XB: array<vec3f, 6>;
fn loadTri(slot: u32, t: vec4u) {
  let prox = icMode() == MODE_PROX; let dr = select(dref(), vec3f(0.0), prox);
  for (var i = 0u; i < 3u; i++) {
    let a = xt(t[i]); XA[slot + i] = a;
    if (prox) { XB[slot + i] = a; } else { XB[slot + i] = positions[t[i]].xyz - dr; }
  }
}
fn boxMargin() -> f32 { return select(P.f0.w, 0.5 * P.f1.y, icMode() == MODE_PROX); }
fn triBox(slot: u32) -> Box {
  var lo = min(XA[slot], XB[slot]); var hi = max(XA[slot], XB[slot]);
  for (var i = 1u; i < 3u; i++) { lo = min(lo, min(XA[slot + i], XB[slot + i])); hi = max(hi, max(XA[slot + i], XB[slot + i])); }
  let m = boxMargin();
  return Box(lo - vec3f(m), hi + vec3f(m));
}
fn cellOf(p: vec3f, cell: f32) -> vec3i { return vec3i(floor(p / cell)); }
fn cellHash(c: vec3i) -> u32 { let u = vec3u(c); return ((u.x * 73856093u) ^ (u.y * 19349663u) ^ (u.z * 83492791u)) & P.n.z; }
fn cellKey(c: vec3i) -> u32 { let u = vec3u(c) & vec3u(1023u); return u.x | (u.y << 10u) | (u.z << 20u); }

@compute @workgroup_size(${ICLOTH_WG}) fn icHashClear(@builtin(global_invocation_id) id: vec3u) {
  if (halted()) { return; }
  if (icMode() != MODE_PROX && ld(CT_CLEAN) != 0u) { return; }
  if (id.x <= P.n.z) { sto(P.off1.y + id.x, 0u); }
  if (id.x == 0u) {
    sto(CT_HASH, 0u);
    if (icMode() == MODE_CCD) { sto(CT_ROUND_HITS, 0u); sto(CT_ROUND_START, min(ld(CT_IMPACT), P.cap.w)); }
  }
}
@compute @workgroup_size(${ICLOTH_WG}) fn icHashInsert(@builtin(global_invocation_id) id: vec3u) {
  let t = id.x;
  if (t >= P.n.y || halted()) { return; }
  if (icMode() != MODE_PROX && ld(CT_CLEAN) != 0u) { return; }
  loadTri(0u, triOf(t));
  let box = triBox(0u); let cell = cellSize();
  if (!(finite3(box.lo) && finite3(box.hi))) { atomicOr(&cons[CT_STATUS], ST_NONFINITE); return; }
  let c0 = cellOf(box.lo, cell); var c1 = cellOf(box.hi, cell);
  if (any(c1 - c0 > vec3i(3))) { atomicOr(&cons[CT_STATUS], ST_SPAN); c1 = min(c1, c0 + vec3i(3)); }
  for (var z = c0.z; z <= c1.z; z++) { for (var y = c0.y; y <= c1.y; y++) { for (var x = c0.x; x <= c1.x; x++) {
    let c = vec3i(x, y, z);
    let e = atomicAdd(&cons[CT_HASH], 1u);
    if (e >= P.off1.w) { atomicOr(&cons[CT_STATUS], ST_HASH); return; }
    let b = P.off1.z + 3u * e;
    sto(b, t); sto(b + 2u, cellKey(c));
    sto(b + 1u, atomicExchange(&cons[P.off1.y + cellHash(c)], e + 1u));
  } } }
}

// ---------------------------------------------------------------- narrow phase
var<private> pairDynamic: array<bool, 6>;
// One elementary pair: corners i0..i3 of XA / XB (kind 0: vertex i0, face i1 i2 i3; kind 1: edges i0 i1 and i2 i3).
fn testFeature(kind: u32, ids: vec4u, i0: u32, i1: u32, i2: u32, i3: u32) {
  let m = icMode();
  let a0 = XA[i0]; let a1 = XA[i1]; let a2 = XA[i2]; let a3 = XA[i3];
  if (m == MODE_PROX) {
    let F = icFeat(kind, a0, a1, a2, a3); let d2 = dot(F.d, F.d);
    if (d2 >= P.f1.y * P.f1.y) { return; }
    if (!(d2 > 1.0e-18)) { atomicOr(&cons[CT_STATUS], ST_DEGENERATE); atomicAdd(&cons[CT_DEGENERATE], 1u); return; }
    let d = sqrt(d2);
    atomicAdd(&cons[CT_PROX_PAIRS], 1u);
    proxAppend(ids, F.d / d, 0.0, F.w, vec3f(0.0), P.f1.w * P.f0.y * max(P.f0.x - d, 0.0));
    return;
  }
  // Later rounds test only pairs with a vertex the impact zones moved (the paper's Algorithm 1).
  if (m == MODE_CCD && R.a.x > 0u && !(pairDynamic[i0] || pairDynamic[i1] || pairDynamic[i2] || pairDynamic[i3])) { return; }
  let b0 = XB[i0]; let b1 = XB[i1]; let b2 = XB[i2]; let b3 = XB[i3];
  // The two features' swept boxes.
  let mg = vec3f(P.f0.w);
  var lo0 = min(a0, b0); var hi0 = max(a0, b0); var lo1 = min(min(a2, b2), min(a3, b3)); var hi1 = max(max(a2, b2), max(a3, b3));
  if (kind == 0u) { lo1 = min(lo1, min(a1, b1)); hi1 = max(hi1, max(a1, b1)); } else { lo0 = min(lo0, min(a1, b1)); hi0 = max(hi0, max(a1, b1)); }
  if (any(lo0 - mg > hi1) || any(lo1 - mg > hi0)) { return; }
  let h = icCcd(kind, a0, a1, a2, a3, b0, b1, b2, b3, P.f1.x, P.f0.w);
  if (h.hit == 0u) { return; }
  if (m == MODE_VERIFY) { atomicAdd(&cons[CT_REMAIN_PAIRS], 1u); roundCapped(); return; }
  if (h.hit == 3u) { atomicOr(&cons[CT_STATUS], ST_DEGENERATE); atomicAdd(&cons[CT_DEGENERATE], 1u); atomicAdd(&cons[CT_ROUND_HITS], 1u); return; }
  impactUpsert(kind, ids, h.n, P.f0.z, h.w);
}
fn testPair(ta: vec4u, tb: vec4u) {
  // Vertex-face: each vertex is tested from the one triangle that owns it (mask bits 0-2), against faces without it.
  for (var i = 0u; i < 3u; i++) {
    if ((ta.w & (1u << i)) != 0u && ta[i] != tb.x && ta[i] != tb.y && ta[i] != tb.z) { testFeature(0u, vec4u(ta[i], tb.x, tb.y, tb.z), i, 3u, 4u, 5u); }
    if ((tb.w & (1u << i)) != 0u && tb[i] != ta.x && tb[i] != ta.y && tb[i] != ta.z) { testFeature(0u, vec4u(tb[i], ta.x, ta.y, ta.z), 3u + i, 0u, 1u, 2u); }
  }
  // Edge-edge: each edge from the one triangle that owns it (mask bits 3-5), against edges sharing no end.
  for (var i = 0u; i < 3u; i++) {
    if ((ta.w & (8u << i)) == 0u) { continue; }
    let i1 = (i + 1u) % 3u; let p = ta[i]; let q = ta[i1];
    for (var j = 0u; j < 3u; j++) {
      if ((tb.w & (8u << j)) == 0u) { continue; }
      let j1 = (j + 1u) % 3u; let u = tb[j]; let w = tb[j1];
      if (p == u || p == w || q == u || q == w) { continue; }
      testFeature(1u, vec4u(p, q, u, w), i, i1, 3u + j, 3u + j1);
    }
  }
}
// Per triangle a: every triangle b > a whose box overlaps a's, found once (in the cell holding the overlap's low corner).
@compute @workgroup_size(${ICLOTH_WG}) fn icPairs(@builtin(global_invocation_id) id: vec3u) {
  let a = id.x;
  if (a >= P.n.y || halted()) { return; }
  let m = icMode();
  if (m != MODE_PROX && ld(CT_CLEAN) != 0u) { return; }
  let ta = triOf(a);
  loadTri(0u, ta);
  let boxA = triBox(0u); let cell = cellSize();
  if (!(finite3(boxA.lo) && finite3(boxA.hi))) { return; }
  let incremental = m == MODE_CCD && R.a.x > 0u;
  var dynA = false;
  for (var i = 0u; i < 3u; i++) { pairDynamic[i] = (flagOf(ta[i]) & 1u) != 0u; dynA = dynA || pairDynamic[i]; }
  let c0 = cellOf(boxA.lo, cell); let c1 = min(cellOf(boxA.hi, cell), c0 + vec3i(3));
  for (var z = c0.z; z <= c1.z; z++) { for (var y = c0.y; y <= c1.y; y++) { for (var x = c0.x; x <= c1.x; x++) {
    let c = vec3i(x, y, z); let key = cellKey(c);
    var e = ld(P.off1.y + cellHash(c));
    var guard = 0u;
    while (e != 0u && guard < P.off1.w) {
      if ((guard & 31u) == 31u && overrun()) { return; }
      guard++;
      let eb = P.off1.z + 3u * (e - 1u);
      let b = ld(eb); let next = ld(eb + 1u); let k = ld(eb + 2u);
      e = next;
      if (k != key || b <= a) { continue; }
      let tb = triOf(b);
      loadTri(3u, tb);
      let boxB = triBox(3u);
      if (any(boxA.lo > boxB.hi) || any(boxB.lo > boxA.hi)) { continue; }
      if (any(cellOf(max(boxA.lo, boxB.lo), cell) != c)) { continue; }
      var dynB = false;
      for (var i = 0u; i < 3u; i++) { pairDynamic[3u + i] = (flagOf(tb[i]) & 1u) != 0u; dynB = dynB || pairDynamic[3u + i]; }
      if (incremental && !(dynA || dynB)) { continue; }
      testPair(ta, tb);
    }
  } } }
}

// ---------------------------------------------------------------- solids
struct IcField { phi: f32, grad: vec3f, inside: bool }
// The field at p: outside the lattice a moving body's field is clear (word 12), a scene field bounds nothing.
fn icField(h: OgcSolidHeader, p: vec3f) -> IcField {
  let b = floor((p - h.origin) / (8.0 * h.cell));
  if (any(b < vec3f(0.0)) || any(b >= vec3f(h.dims))) { return IcField(h.band, vec3f(0.0), ogcSolidWord(h.base + 12u) != 0u); }
  let s = ogcSolidSample(h, p);
  return IcField(s.phi, s.grad, true);
}
// March p0 -> p1 through the field: the first sample below dTol (xyz, w = 1), stepping by the field's own bound on
// how far the path is clear, at least 1/32 of it.
fn icTrace(h: OgcSolidHeader, p0: vec3f, p1: vec3f) -> vec4f {
  let len = max(length(p1 - p0), 1.0e-12); let L = max(h.lmax, 1.0);
  var t = 0.0; var tIn = -1.0;
  for (var i = 0u; i < 40u; i++) {
    let p = mix(p0, p1, t);
    let s = icField(h, p);
    if (!s.inside) {
      // The march stepped over the lattice's edge from a sample inside it (a container's wall is the edge, and the
      // band below dTol in front of it is thinner than the least step): bisect back to the first sample below dTol.
      // A path that leaves a lattice whose edge is clear is the failure.
      var lo = tIn; var hi = t;
      for (var k = 0u; k < 12u && lo >= 0.0; k++) {
        let tm = 0.5 * (lo + hi); let pm = mix(p0, p1, tm); let sm = icField(h, pm);
        if (!sm.inside) { hi = tm; } else if (sm.phi < P.f0.w) { return vec4f(pm, 1.0); } else { lo = tm; }
      }
      atomicOr(&cons[CT_STATUS], ST_BOUNDS); return vec4f(0.0);
    }
    tIn = t;
    if (s.phi < P.f0.w) { return vec4f(p, 1.0); }
    if (t >= 1.0) { break; }
    t = min(1.0, t + max((s.phi - P.f0.w) / (L * len), 1.0 / 32.0));
  }
  return vec4f(0.0);
}
fn solidImpact(kind: u32, v: u32, n: vec3f, d0: f32) {
  if (icMode() == MODE_VERIFY) { atomicAdd(&cons[CT_REMAIN_SOLID], 1u); roundCapped(); return; }
  impactUpsert(kind, vec4u(v, NONE, NONE, NONE), n, d0, vec4f(1.0, 0.0, 0.0, 0.0));
}
@compute @workgroup_size(${ICLOTH_WG}) fn icSolid(@builtin(global_invocation_id) id: vec3u) {
  let v = id.x;
  if (v >= P.n.x || halted()) { return; }
  let m = icMode();
  if (m != MODE_PROX && ld(CT_CLEAN) != 0u) { return; }
  let x0 = xt(v);
  if (m == MODE_PROX) {
    if (pinned(v)) { return; }
    // A plane per solid within reach of the vertex's step: the field linearised at x_t (the moving solid's at its end pose).
    let reach = 2.0 * P.f0.x + length(yt(v) - x0);
    if (P.sol.x != 0u) {
      let h = ogcSolidHeaderAt(P.sol.x); let s = icField(h, x0); let gl = length(s.grad);
      if (!s.inside) { atomicOr(&cons[CT_STATUS], ST_BOUNDS); }
      else if (s.phi < reach && gl > 1.0e-6) {
        let n = s.grad / gl;
        atomicAdd(&cons[CT_PROX_SOLID], 1u);
        proxAppend(vec4u(v, NONE, NONE, NONE), n, dot(n, x0) - s.phi / gl, vec4f(1.0, 0.0, 0.0, 0.0), vec3f(0.0), ogcSolidMu(h, x0) * P.f0.y * max(P.f0.x - s.phi, 0.0));
      }
    }
    if (P.sol.y != 0u) {
      let h = ogcSolidHeaderAt(P.sol.y); let p0 = icPose(0u); let p1 = icPose(1u);
      let q = ogcToBody(p0, x0); let s = icField(h, q); let gl = length(s.grad);
      let us = ogcToWorld(p1, q) - x0;
      if (s.inside && s.phi < reach + length(us) && gl > 1.0e-6) {
        let nb = s.grad / gl; let n = p1.R * nb;
        atomicAdd(&cons[CT_PROX_SOLID], 1u);
        proxAppend(vec4u(v, NONE, NONE, NONE), n, dot(n, p1.c) + dot(nb, q) - s.phi / gl, vec4f(1.0, 0.0, 0.0, 0.0), us, ogcSolidMu(h, q) * P.f0.y * max(P.f0.x - s.phi, 0.0));
      }
    }
    return;
  }
  if (pinned(v)) { return; }
  if (m == MODE_CCD && R.a.x > 0u && (flagOf(v) & 1u) == 0u) { return; }
  // An impact's plane is where the field, linearised at the hit, equals delta: the same measure the trace tests
  // (phi < dTol = delta / 2), so a solve that meets its plane within its tolerance leaves nothing for the next
  // detection to find. A plane delta of distance past the surface is not that where |grad phi| < 1/2 (the lattice's
  // interpolant in a concave corner is phi ~ dx dy / cell: a vertex lying on the floor at a wall has |grad| ~ 0.1).
  let x1 = positions[v].xyz;
  if (!finite3(x1)) { return; }
  if (P.sol.x != 0u) {
    let h = ogcSolidHeaderAt(P.sol.x);
    let hit = icTrace(h, x0, x1);
    if (hit.w != 0.0) {
      let s = icField(h, hit.xyz); let gl = length(s.grad);
      if (gl > 1.0e-6) { let n = s.grad / gl; solidImpact(2u, v, n, dot(n, hit.xyz) + (P.f0.z - s.phi) / gl); }
      else { atomicOr(&cons[CT_STATUS], ST_DEGENERATE); atomicAdd(&cons[CT_DEGENERATE], 1u); }
    }
  }
  if (P.sol.y != 0u) {
    let h = ogcSolidHeaderAt(P.sol.y); let p0 = icPose(0u); let p1 = icPose(1u);
    let hit = icTrace(h, ogcToBody(p0, x0), ogcToBody(p1, x1));
    if (hit.w != 0.0) {
      let s = icField(h, hit.xyz); let gl = length(s.grad);
      if (gl > 1.0e-6) { let nb = s.grad / gl; let n = p1.R * nb; solidImpact(3u, v, n, dot(n, p1.c) + dot(nb, hit.xyz) + (P.f0.z - s.phi) / gl); }
      else { atomicOr(&cons[CT_STATUS], ST_DEGENERATE); atomicAdd(&cons[CT_DEGENERATE], 1u); }
    }
  }
}

// ---------------------------------------------------------------- the impact-zone solve
// Minimise g(X) = q sum_i (m_i / mean m) |X_i - X0_i|^2, q = ${ICLOTH_SOLVER.objective} mu / 2, over the impact zones' vertices subject to every impact's
// C_j(X) >= 0 (the objective's scale: ICLOTH_SOLVER), as the augmented Lagrangian
// f = g + mu / 2 sum_j c~_j^2 - |lam|^2 / (2 mu), c~_j = max(lam_j / mu - C_j, 0),
// by Jacobi-preconditioned gradient descent with a backtracking line search, and the multiplier update
// lam_j <- mu c~_j whenever the descent stalls with an impact still violated (the paper's Algorithm 3 with the
// deviations of ICLOTH_SOLVER). X0 is the pose the time integration left.
//
// Two dispatches an iteration k. icGrad: the line search's decision for iteration k - 1 from icTrial's partial sums
// (every workgroup reduces them itself), X^k from it (each thread forms its own and its partners' from the rotating
// iterates, so nothing is read while written), the multipliers, the gradient and the direction. icTrial: the
// objective's change along the direction at the eight trial steps, in difference form, as per-workgroup partials.

// A round with impacts: the round's iterates start at the current pose, new zone vertices are counted.
@compute @workgroup_size(${ICLOTH_WG}) fn icZoneInit(@builtin(global_invocation_id) id: vec3u) {
  let v = id.x;
  if (v >= P.n.x || halted()) { return; }
  if (ld(CT_CLEAN) != 0u) { return; }
  let hits = ld(CT_ROUND_HITS);
  if (hits == 0u) { sto(CT_CLEAN, 1u); return; }
  if (v == 0u) { sto(CT_GD_DONE, 0u); sto(CT_IMPACTS_ROUND + R.a.x, hits); atomicAdd(&cons[CT_ROUNDS_USED], 1u); }
  let x = positions[v];
  if (R.a.x == 0u) { work[wseg(W_X0, v)] = x; }
  work[wseg(W_XR, v)] = x; work[wseg(W_XR + 1u, v)] = x; work[wseg(W_XR + 2u, v)] = x; work[wseg(W_D, v)] = vec4f(0.0); work[wseg(W_D + 1u, v)] = vec4f(0.0);
  let f = flagOf(v);
  if (f != 0u) {
    if (f != 3u) { sto(P.off0.x + v, 3u); }
    if (!pinned(v)) { atomicAdd(&cons[CT_ZONE + R.a.x], 1u); }
  }
}

var<workgroup> wgCtl: vec2u;
var<workgroup> rd0: array<vec4f, ${ICLOTH_WG}>;
var<workgroup> rd1: array<vec4f, ${ICLOTH_WG}>;
var<workgroup> rd2: array<vec4f, ${ICLOTH_WG}>;
var<workgroup> rd3: array<vec4f, ${ICLOTH_WG}>;
var<workgroup> rd4: array<vec4f, ${ICLOTH_WG}>;
fn trialStep(j: u32) -> f32 { return exp2(-f32(j)); }
// The iteration's step and iterate index, private to the thread.
var<private> gdK: u32;
var<private> gdS: f32;
// X^k of vertex j: X' = X^(k-1) + s d^(k-1), then Chebyshev's omega (X' - X^(k-2)) + X^(k-2).
fn iterate(j: u32) -> vec3f {
  if (gdK == 0u) { return work[wseg(W_XR, j)].xyz; }
  let x1 = work[wseg(W_XR + (gdK + 2u) % 3u, j)].xyz; let x2 = work[wseg(W_XR + (gdK + 1u) % 3u, j)].xyz;
  let xp = x1 + gdS * work[wseg(W_D + (gdK + 1u) % 2u, j)].xyz;
  return R.b.x * (xp - x2) + x2;
}
@compute @workgroup_size(${ICLOTH_WG}) fn icGrad(@builtin(local_invocation_index) lane: u32, @builtin(workgroup_id) wg: vec3u) {
  if (lane == 0u) { wgCtl = vec2u(ld(CT_CLEAN) | select(0u, 1u, halted()), ld(CT_GD_DONE)); }
  let ctl = workgroupUniformLoad(&wgCtl);
  if (ctl.x != 0u || ctl.y != 0u) { return; }
  let k = R.a.y; let last = (R.a.w & ${ICLOTH_MODE_LAST}u) != 0u;
  var s = 0.0; var stalled = false;
  if (k > 0u) {
    var e0 = vec4f(0.0); var e1 = vec4f(0.0); var f0 = vec4f(0.0); var f1 = vec4f(0.0); var mm = vec4f(0.0);
    for (var i = lane; i < P.n.w; i += WG) {
      let b = P.wk.y + PARTIALS * i;
      e0 += work[b]; e1 += work[b + 1u]; f0 += work[b + 2u]; f1 += work[b + 3u];
      let q = work[b + 4u]; mm = vec4f(mm.x + q.x, max(mm.y, q.y), max(mm.z, q.z), 0.0);
    }
    rd0[lane] = e0; rd1[lane] = e1; rd2[lane] = f0; rd3[lane] = f1; rd4[lane] = mm;
    workgroupBarrier();
    for (var h = WG / 2u; h > 0u; h >>= 1u) {
      if (lane < h) {
        rd0[lane] += rd0[lane + h]; rd1[lane] += rd1[lane + h]; rd2[lane] += rd2[lane + h]; rd3[lane] += rd3[lane + h];
        let q = rd4[lane + h]; let p = rd4[lane];
        rd4[lane] = vec4f(p.x + q.x, max(p.y, q.y), max(p.z, q.z), 0.0);
      }
      workgroupBarrier();
    }
    let E0 = rd0[0]; let E1 = rd1[0]; let F0 = rd2[0]; let F1 = rd3[0]; let M = rd4[0];
    // The largest trial step with the Armijo decrease (the sums' f32 rounding allowed for). No step, or one below the
    // threshold, with every impact within the feasibility tolerance: converged at X^(k-1). No step while an impact
    // is violated: the multipliers are updated at X^k (s = 0 without a step).
    var found = false;
    for (var j = 0u; j < TRIALS; j++) {
      var e = E0[j % 4u]; var fl = F0[j % 4u];
      if (j >= 4u) { e = E1[j % 4u]; fl = F1[j % 4u]; }
      let sj = trialStep(j);
      if (M.x < 0.0 && e <= P.f3.x * sj * M.x + 4.0 * EPS32 * fl) { s = sj; found = true; break; }
    }
    if (!found) { s = 0.0; }
    stalled = !found || s * M.z < max(P.f3.y, ${ICLOTH_SOLVER.inexact} * M.y);
    let moving = !stalled || M.y >= P.f3.z;
    if (!moving || last) {
      // Every workgroup reaches the same decision from the same sums (or returned at its start on another's); the
      // stores are of equal values, and icCommit accumulates the outcome once.
      if (lane == 0u) {
        sto(CT_GD_FINAL, k - 1u); stf(CT_MAX_VIOL, M.y);
        sto(CT_GD_OUTCOME, select(0u, select(1u, 2u, M.y >= P.f3.z), moving));
        sto(CT_GD_DONE, 1u + (k + 2u) % 3u);
      }
      return;
    }
  }
  let v = wg.x * WG + lane;
  if (v >= P.n.x) { return; }
  if (flagOf(v) == 0u) { return; }
  gdK = k; gdS = s;
  let xv = iterate(v);
  if (k > 0u) { work[wseg(W_XR + k % 3u, v)] = vec4f(xv, 0.0); }
  let mu = P.f2.z; let held = pinned(v);
  let q = ${0.5 * ICLOTH_SOLVER.objective} * mu * massOf(v) / P.f2.w;
  var g = 2.0 * q * (xv - work[wseg(W_X0, v)].xyz); var pre = 2.0 * q;
  let lb = P.off0.w + v * (P.cap.z + 1u); let cnt = min(ld(lb), P.cap.z);
  for (var e = 0u; e < cnt; e++) {
    let ri = ld(lb + 1u + e);
    if (ri == 0u) { continue; }
    let b = P.off1.x + ${ICLOTH_RECORD_WORDS}u * (ri - 1u);
    let ids = ld4u(b); let nd = ld4(b + 4u); let w = ld4(b + 8u);
    var sum = vec3f(0.0); var wv = 0.0;
    for (var i = 0u; i < 4u; i++) {
      if (ids[i] == NONE) { continue; }
      if (ids[i] == v) { sum += w[i] * xv; wv = w[i]; } else { sum += w[i] * iterate(ids[i]); }
    }
    let Cv = dot(nd.xyz, sum) - nd.w;
    var lam = ldf(b + 12u + k % 2u);
    if (stalled) { lam = max(lam - mu * Cv, 0.0); }
    if (ids.x == v) { stf(b + 12u + (k + 1u) % 2u, lam); }
    let ct = max(lam / mu - Cv, 0.0);
    if (ct > 0.0) { g -= mu * ct * wv * nd.xyz; pre += mu * wv * wv; }
  }
  var d = vec3f(0.0);
  if (!held) { d = -g / pre; }
  work[wseg(W_D + k % 2u, v)] = vec4f(d, pre);
}
@compute @workgroup_size(${ICLOTH_WG}) fn icTrial(@builtin(local_invocation_index) lane: u32, @builtin(workgroup_id) wg: vec3u) {
  if (lane == 0u) { wgCtl = vec2u(ld(CT_CLEAN) | select(0u, 1u, halted()), ld(CT_GD_DONE)); }
  let ctl = workgroupUniformLoad(&wgCtl);
  if (ctl.x != 0u || ctl.y != 0u) { return; }
  let k = R.a.y; let v = wg.x * WG + lane;
  var e0 = vec4f(0.0); var e1 = vec4f(0.0); var f0 = vec4f(0.0); var f1 = vec4f(0.0); var mm = vec4f(0.0);
  if (v < P.n.x && flagOf(v) != 0u) {
    let mu = P.f2.z;
    let xv = work[wseg(W_XR + k % 3u, v)].xyz; let dv = work[wseg(W_D + k % 2u, v)];
    if (!pinned(v)) {
      let q = ${0.5 * ICLOTH_SOLVER.objective} * mu * massOf(v) / P.f2.w;
      let a = xv - work[wseg(W_X0, v)].xyz; let lin = 2.0 * q * dot(dv.xyz, a); let quad = q * dot(dv.xyz, dv.xyz);
      for (var j = 0u; j < TRIALS; j++) {
        let sj = trialStep(j); let de = sj * lin + sj * sj * quad; let fl = abs(sj * lin) + sj * sj * quad;
        if (j < 4u) { e0[j] += de; f0[j] += fl; } else { e1[j - 4u] += de; f1[j - 4u] += fl; }
      }
      mm.x = -dv.w * dot(dv.xyz, dv.xyz); mm.z = length(dv.xyz);
    }
    // The impacts v owns (each once over all vertices).
    let lb = P.off0.w + v * (P.cap.z + 1u); let cnt = min(ld(lb), P.cap.z);
    for (var e = 0u; e < cnt; e++) {
      let ri = ld(lb + 1u + e);
      if (ri == 0u) { continue; }
      let b = P.off1.x + ${ICLOTH_RECORD_WORDS}u * (ri - 1u);
      let ids = ld4u(b);
      if (ids.x != v) { continue; }
      let nd = ld4(b + 4u); let w = ld4(b + 8u);
      var sum = vec3f(0.0); var dsum = vec3f(0.0);
      for (var i = 0u; i < 4u; i++) {
        if (ids[i] == NONE) { continue; }
        sum += w[i] * work[wseg(W_XR + k % 3u, ids[i])].xyz; dsum += w[i] * work[wseg(W_D + k % 2u, ids[i])].xyz;
      }
      let Cv = dot(nd.xyz, sum) - nd.w; let dC = dot(nd.xyz, dsum);
      let lam = ldf(b + 12u + (k + 1u) % 2u);
      let c0 = max(lam / mu - Cv, 0.0);
      mm.y = max(mm.y, -Cv);
      for (var j = 0u; j < TRIALS; j++) {
        let cj = max(lam / mu - (Cv + trialStep(j) * dC), 0.0);
        let de = 0.5 * mu * (cj - c0) * (cj + c0); let fl = abs(de) + mu * (c0 + cj) * EPS32 * (abs(Cv) + abs(nd.w));
        if (j < 4u) { e0[j] += de; f0[j] += fl; } else { e1[j - 4u] += de; f1[j - 4u] += fl; }
      }
    }
  }
  rd0[lane] = e0; rd1[lane] = e1; rd2[lane] = f0; rd3[lane] = f1; rd4[lane] = mm;
  workgroupBarrier();
  for (var h = WG / 2u; h > 0u; h >>= 1u) {
    if (lane < h) {
      rd0[lane] += rd0[lane + h]; rd1[lane] += rd1[lane + h]; rd2[lane] += rd2[lane + h]; rd3[lane] += rd3[lane + h];
      let q = rd4[lane + h]; let p = rd4[lane];
      rd4[lane] = vec4f(p.x + q.x, max(p.y, q.y), max(p.z, q.z), 0.0);
    }
    workgroupBarrier();
  }
  if (lane == 0u) {
    let b = P.wk.y + PARTIALS * wg.x;
    work[b] = rd0[0]; work[b + 1u] = rd1[0]; work[b + 2u] = rd2[0]; work[b + 3u] = rd3[0]; work[b + 4u] = rd4[0];
  }
}
// The round's result into the pose: the final iterate of the zones' vertices; the multipliers settle in slot 0 for
// the next round; every vertex's displacement sizes the next round's cells.
@compute @workgroup_size(${ICLOTH_WG}) fn icCommit(@builtin(global_invocation_id) id: vec3u) {
  let v = id.x;
  if (v >= P.n.x || halted()) { return; }
  if (ld(CT_CLEAN) != 0u) { return; }
  let dn = ld(CT_GD_DONE);
  if (dn == 0u) { return; }
  if (v == 0u) {
    atomicAdd(&cons[CT_GD_ITERS], ld(CT_GD_FINAL) + 1u);
    let outcome = ld(CT_GD_OUTCOME);
    if (outcome != 0u) { atomicOr(&cons[CT_STATUS], ST_ITER_CAP); atomicAdd(&cons[CT_CAPPED], 1u); }
    if (outcome == 2u) { atomicAdd(&cons[CT_STALLED], 1u); }
  }
  var x = positions[v].xyz;
  if (flagOf(v) != 0u) {
    if (!pinned(v)) {
      x = work[wseg(W_XR + (dn - 1u), v)].xyz;
      if (!finite3(x)) { atomicOr(&cons[CT_STATUS], ST_NONFINITE); return; }
      positions[v] = vec4f(x, positions[v].w);
    }
    if ((ld(CT_GD_FINAL) + 1u) % 2u == 1u) {
      let lb = P.off0.w + v * (P.cap.z + 1u); let cnt = min(ld(lb), P.cap.z);
      for (var e = 0u; e < cnt; e++) {
        let ri = ld(lb + 1u + e);
        if (ri == 0u) { continue; }
        let b = P.off1.x + ${ICLOTH_RECORD_WORDS}u * (ri - 1u);
        if (ld(b) == v) { sto(b + 12u, ld(b + 13u)); }
      }
    }
  }
  let d = length(x - xt(v) - dref());
  if (d < 1.0e30) { atomicMax(&cons[CT_MAX_DISP + R.a.x + 1u], bitcast<u32>(d)); }
}

// The step's contact state, under the OGC detector's contract for the solver's BDF2 restart (ClothSolverContact.state):
// bit 0, the vertex is in contact at the step's final pose (a proximity record of its within r); bit 1, an onset - the
// contact switched on in this step, or the collision rounds moved the vertex (its velocity jumps either way).
@compute @workgroup_size(${ICLOTH_WG}) fn icContactState(@builtin(global_invocation_id) id: vec3u) {
  let v = id.x;
  if (v >= P.n.x || ld(CT_HALT) != 0u) { return; }
  if (v == 0u && (ld(CT_STATUS) & ST_FAILING) != 0u) { sto(CT_HALT, 1u); }
  if (pinned(v)) { contactState[v] = 0u; return; }
  let before = contactState[v] & 1u;
  var touching = false;
  let lb = P.off0.y + v * (P.cap.x + 1u); let cnt = min(ld(lb), P.cap.x);
  for (var k = 0u; k < cnt && !touching; k++) {
    let b = P.off0.z + ${ICLOTH_RECORD_WORDS}u * (ld(lb + 1u + k) - 1u);
    let ids = ld4u(b); let nd = ld4(b + 4u); let w = ld4(b + 8u);
    var sum = vec3f(0.0);
    for (var i = 0u; i < 4u; i++) { if (ids[i] != NONE) { sum += w[i] * positions[ids[i]].xyz; } }
    touching = dot(nd.xyz, sum) - nd.w < P.f0.x;
  }
  let onset = (touching && before == 0u) || flagOf(v) != 0u;
  contactState[v] = select(0u, 1u, touching) | select(0u, 2u, onset);
}
`;

export interface IClothProximityBlockOptions {
  readonly binding: number;
  /** Word offsets in the constraint buffer. */
  readonly lists: number;
  readonly records: number;
  readonly entriesPerVertex: number;
  readonly r: number;
  readonly kc: number;
  /** The friction regularisation length eps_v dt (m). */
  readonly epsU: number;
}

/**
 * The Newton solver's proximity block: for every record within r, the penalty kc / 2 (r - C)^2 with its exact
 * (constant, PSD) Hessian, and Coulomb friction lagged at the step's start: mu N f0(|u_T|) with IPC's smoothed f0, u
 * the pair's relative displacement over the step (less the solid's), its Hessian mu N f1(|u_T|) / |u_T| T (the PSD
 * part). The host provides pxX (trial-aware positions), pxXt and pxDir.
 */
export function iclothProximityBlockWGSL(o: IClothProximityBlockOptions): string {
  const f = (x: number) => { const s = Math.fround(x).toString(); return /[.e]/.test(s) ? s : `${s}.0`; };
  return /* wgsl */`
@group(0) @binding(${o.binding}) var<storage, read> pxb: array<u32>;
const PX_NONE = 0xffffffffu;
const PX_K = ${o.entriesPerVertex}u; const PX_LISTS = ${o.lists}u; const PX_RECORDS = ${o.records}u;
const PX_R: f32 = ${f(o.r)}; const PX_KC: f32 = ${f(o.kc)}; const PX_EPS: f32 = ${f(o.epsU)};
fn pxF(i: u32) -> f32 { return bitcast<f32>(pxb[i]); }
struct PxRecord { ids: vec4u, n: vec3f, d0: f32, w: vec4f, us: vec3f, muN: f32 }
fn pxRecord(r: u32) -> PxRecord {
  let b = PX_RECORDS + ${ICLOTH_RECORD_WORDS}u * r;
  return PxRecord(vec4u(pxb[b], pxb[b + 1u], pxb[b + 2u], pxb[b + 3u]), vec3f(pxF(b + 4u), pxF(b + 5u), pxF(b + 6u)), pxF(b + 7u),
    vec4f(pxF(b + 8u), pxF(b + 9u), pxF(b + 10u), pxF(b + 11u)), vec3f(pxF(b + 12u), pxF(b + 13u), pxF(b + 14u)), pxF(b + 15u));
}
fn pxCount(v: u32) -> u32 { return min(pxb[PX_LISTS + v * (PX_K + 1u)], PX_K); }
fn pxEntry(v: u32, k: u32) -> u32 { return pxb[PX_LISTS + v * (PX_K + 1u) + 1u + k] - 1u; }
// f1(y) / y of the smoothed friction (y = |u_T|), and f0(y).
fn pxF1OverY(y: f32) -> f32 { if (y < PX_EPS) { return (2.0 - y / PX_EPS) / PX_EPS; } return 1.0 / y; }
fn pxF0(y: f32) -> f32 { if (y < PX_EPS) { return y * y / PX_EPS - y * y * y / (3.0 * PX_EPS * PX_EPS); } return y - PX_EPS / 3.0; }
// Vertex v's gradient and diagonal block from its records, v at xv and its partners where they are.
fn pxVertexTerms(v: u32, xv: vec3f, g: ptr<function, vec3f>, H: ptr<function, mat3x3f>) {
  let cnt = pxCount(v);
  for (var k = 0u; k < cnt; k++) {
    let rec = pxRecord(pxEntry(v, k));
    var sx = vec3f(0.0); var su = vec3f(0.0); var wv = 0.0;
    for (var i = 0u; i < 4u; i++) {
      let j = rec.ids[i];
      if (j == PX_NONE) { continue; }
      var xj = xv;
      if (j == v) { wv = rec.w[i]; } else { xj = pxX(j); }
      sx += rec.w[i] * xj; su += rec.w[i] * (xj - pxXt(j));
    }
    let Cv = dot(rec.n, sx) - rec.d0;
    if (Cv < PX_R) {
      *g -= PX_KC * (PX_R - Cv) * wv * rec.n;
      *H += PX_KC * wv * wv * mat3x3f(rec.n * rec.n.x, rec.n * rec.n.y, rec.n * rec.n.z);
    }
    if (rec.muN > 0.0) {
      let u = su - rec.us; let ut = u - rec.n * dot(rec.n, u);
      let c = rec.muN * pxF1OverY(length(ut));
      *g += c * wv * ut;
      *H += c * wv * wv * (mat3x3f(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0) - mat3x3f(rec.n * rec.n.x, rec.n * rec.n.y, rec.n * rec.n.z));
    }
  }
}
// Row v of the proximity Hessian times the PCG vector (pxDir), linearised at the iterate.
fn pxVertexHv(v: u32) -> vec3f {
  let cnt = pxCount(v);
  var out = vec3f(0.0);
  for (var k = 0u; k < cnt; k++) {
    let rec = pxRecord(pxEntry(v, k));
    var sx = vec3f(0.0); var su = vec3f(0.0); var sd = vec3f(0.0); var wv = 0.0;
    for (var i = 0u; i < 4u; i++) {
      let j = rec.ids[i];
      if (j == PX_NONE) { continue; }
      if (j == v) { wv = rec.w[i]; }
      let xj = pxX(j);
      sx += rec.w[i] * xj; su += rec.w[i] * (xj - pxXt(j)); sd += rec.w[i] * pxDir(j);
    }
    let nd = dot(rec.n, sd);
    if (dot(rec.n, sx) - rec.d0 < PX_R) { out += PX_KC * wv * nd * rec.n; }
    if (rec.muN > 0.0) {
      let u = su - rec.us; let ut = u - rec.n * dot(rec.n, u);
      out += rec.muN * pxF1OverY(length(ut)) * wv * (sd - rec.n * nd);
    }
  }
  return out;
}
// The energy of the records v owns (its first vertex; each record once over all vertices) at pxX, and their count.
fn pxOwnedEnergy(v: u32) -> vec2f {
  let cnt = pxCount(v);
  var e = 0.0; var owned = 0.0;
  for (var k = 0u; k < cnt; k++) {
    let rec = pxRecord(pxEntry(v, k));
    if (rec.ids.x != v) { continue; }
    owned += 1.0;
    var sx = vec3f(0.0); var su = vec3f(0.0);
    for (var i = 0u; i < 4u; i++) {
      let j = rec.ids[i];
      if (j == PX_NONE) { continue; }
      let xj = pxX(j);
      sx += rec.w[i] * xj; su += rec.w[i] * (xj - pxXt(j));
    }
    let Cv = dot(rec.n, sx) - rec.d0;
    if (Cv < PX_R) { e += 0.5 * PX_KC * (PX_R - Cv) * (PX_R - Cv); }
    if (rec.muN > 0.0) { let u = su - rec.us; let ut = u - rec.n * dot(rec.n, u); e += rec.muN * pxF0(length(ut)); }
  }
  return vec2f(e, owned);
}
`;
}

/** The geometry alone (closest features, the cubic CCD), for the kernel tests' harness module. */
export const ICLOTH_GEOMETRY_WGSL = GEOMETRY;
