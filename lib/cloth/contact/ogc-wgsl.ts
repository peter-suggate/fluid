/**
 * WGSL for OGC cloth self-contact (Chen et al. 2025, "Offset Geometric
 * Contact"): the shared geometry, the activation and friction, and the block
 * the solver's per-vertex VBD kernel includes to gather a vertex's contact
 * force, Hessian and energy. Truncation is a separate pass (ogc-detector-wgsl.ts).
 *
 * Contact entries (vec4<u32>, one list per vertex, written at detection):
 *   vertex-facet  x = contacting vertex v', y = face | feature << 28,
 *                 z = lagged normal force magnitude (f32 bits), w = 0
 *   edge-edge     x = own edge e, y = other edge | 1 << 31,
 *                 z = lagged normal force magnitude, w = pack2x16unorm(s, t)
 *   vertex-solid  x = v, y = OGC_SOLID_Y (feature 7, no face), z = lagged normal force, w = 0
 * Feature: vertex k (0..2), edge (t[k], t[k+1]) (3..5), face interior (6), static solid (7).
 * Every entry is self-describing, so the gather never chases a second list.
 *
 * Contact buffer header (vec4<u32>): [0] (N, list capacity, counts base, entries base),
 * [1] (r, k_c, mu, eps_v dt) as f32, [2] (solid image word offset, 0 when none, 0, 0).
 */
import { SOLID_HEADER_WORDS, SOLID_MU_SCALE, SOLID_MU_WORD } from "./solid-field";


/** Counter word indices (atomic<u32>) shared by the TS host and the WGSL. */
export const OGC_COUNTER = {
  exceed: 0, rounds: 1, status: 2, peakContacts: 3, peakGrid: 4, maxEnvelope: 5, cell: 6, invCell: 7,
  frameDetections: 8, frameSteps: 9, frameDecides: 10, frameTruncated: 11, frameTruncatedMax: 12,
  frameRedetections: 13, framePairEntries: 14, frameCandidates: 15, intersections: 16,
  frameEnvelopeCapHits: 17, framePeakEnvelope: 18, framePairMax: 19, frameRoundCapHits: 20,
  frameVfPairs: 21, frameEePairs: 22,
  /** DAT division pairs: slots reserved this round, their persistent peak, and per-frame stats. */
  pairSlots: 23, peakPairSlots: 24, frameDatPairs: 25, frameCapped: 26, frameUncertified: 27, framePeakPairSlots: 28,
  /** Vertices with a static-solid entry over the frame's rounds, and the truncations the solid bound made. */
  frameSolidContacts: 29, frameSolidCut: 30,
  /**
   * Predicted-motion DAT: the round's conflicting pairs (consumed by the redo
   * decision), the flag that holds every prediction for a redone round, and per
   * frame the conflicts, the redone rounds and the held vertices of mandatory rounds.
   */
  conflicts: 31, forceHeld: 32, frameConflicts: 33, frameRedos: 34, frameHeld: 35,
  /** The round's (r_q / rqMax)^3 summed in units of 1 / 1024 (the cell kernel's cubic mean; consumed by it). */
  envelopeSum: 36,
} as const;
export const OGC_COUNTER_WORDS = 40;
export const OGC_FRAME_COUNTERS: readonly number[] = [8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 25, 26, 27, 28, 29, 30, 33, 34, 35];

/** Status bits: every one is an explicit failure of the step it was raised in. */
export const OGC_STATUS = {
  /** A vertex's contact list overflowed its authored capacity. */
  contactOverflow: 1,
  /** The broad-phase grid overflowed its authored entry capacity. */
  gridOverflow: 2,
  /** Re-detection was needed after the step's authored round cap. */
  roundCap: 4,
  /** A vertex's velocity envelope reached the authored maximum query radius. */
  envelopeCap: 8,
  /** A non-finite position reached detection. */
  nonFinite: 16,
  /** The DAT division-pair list overflowed its authored capacity. */
  pairOverflow: 32,
  /** A pair within the envelope had no strictly separating plane; its four vertices were held. */
  uncertifiedPair: 64,
  /** A vertex was at or inside the static solid's surface at detection; it was held. */
  solidPenetration: 128,
  /** A vertex left the box the solid field covers, where the solid is not seen. */
  solidBounds: 256,
} as const;

/** Truncation schemes: Planar-DAT (Chen, Hsu, Ayman, Macklin 2026) and the isotropic OGC bound. */
export const OGC_TRUNCATION = { planar: 0, isotropic: 1 } as const;
export type OgcTruncationMode = keyof typeof OGC_TRUNCATION;

export const OGC_COUNTERS_WGSL = /* wgsl */`
const OGC_EXCEED = ${OGC_COUNTER.exceed}u;
const OGC_ROUNDS = ${OGC_COUNTER.rounds}u;
const OGC_STATUS_WORD = ${OGC_COUNTER.status}u;
const OGC_PEAK_CONTACTS = ${OGC_COUNTER.peakContacts}u;
const OGC_PEAK_GRID = ${OGC_COUNTER.peakGrid}u;
const OGC_MAX_ENVELOPE = ${OGC_COUNTER.maxEnvelope}u;
const OGC_CELL = ${OGC_COUNTER.cell}u;
const OGC_INV_CELL = ${OGC_COUNTER.invCell}u;
const OGC_F_DETECTIONS = ${OGC_COUNTER.frameDetections}u;
const OGC_F_STEPS = ${OGC_COUNTER.frameSteps}u;
const OGC_F_DECIDES = ${OGC_COUNTER.frameDecides}u;
const OGC_F_TRUNCATED = ${OGC_COUNTER.frameTruncated}u;
const OGC_F_TRUNCATED_MAX = ${OGC_COUNTER.frameTruncatedMax}u;
const OGC_F_REDETECTIONS = ${OGC_COUNTER.frameRedetections}u;
const OGC_F_PAIR_ENTRIES = ${OGC_COUNTER.framePairEntries}u;
const OGC_F_CANDIDATES = ${OGC_COUNTER.frameCandidates}u;
const OGC_INTERSECTIONS = ${OGC_COUNTER.intersections}u;
const OGC_F_ENVELOPE_CAP = ${OGC_COUNTER.frameEnvelopeCapHits}u;
const OGC_F_PEAK_ENVELOPE = ${OGC_COUNTER.framePeakEnvelope}u;
const OGC_F_PAIR_MAX = ${OGC_COUNTER.framePairMax}u;
const OGC_F_ROUND_CAP = ${OGC_COUNTER.frameRoundCapHits}u;
const OGC_F_VF_PAIRS = ${OGC_COUNTER.frameVfPairs}u;
const OGC_F_EE_PAIRS = ${OGC_COUNTER.frameEePairs}u;
const OGC_PAIR_SLOTS = ${OGC_COUNTER.pairSlots}u;
const OGC_PEAK_PAIR_SLOTS = ${OGC_COUNTER.peakPairSlots}u;
const OGC_F_DAT_PAIRS = ${OGC_COUNTER.frameDatPairs}u;
const OGC_F_CAPPED = ${OGC_COUNTER.frameCapped}u;
const OGC_F_UNCERTIFIED = ${OGC_COUNTER.frameUncertified}u;
const OGC_F_PEAK_PAIR_SLOTS = ${OGC_COUNTER.framePeakPairSlots}u;
const OGC_F_SOLID_CONTACTS = ${OGC_COUNTER.frameSolidContacts}u;
const OGC_F_SOLID_CUT = ${OGC_COUNTER.frameSolidCut}u;
const OGC_CONFLICTS = ${OGC_COUNTER.conflicts}u;
const OGC_FORCE_HELD = ${OGC_COUNTER.forceHeld}u;
const OGC_F_CONFLICTS = ${OGC_COUNTER.frameConflicts}u;
const OGC_F_REDOS = ${OGC_COUNTER.frameRedos}u;
const OGC_F_HELD = ${OGC_COUNTER.frameHeld}u;
const OGC_ENVELOPE_SUM = ${OGC_COUNTER.envelopeSum}u;
const OGC_ST_SOLID_PENETRATION = ${OGC_STATUS.solidPenetration}u;
const OGC_ST_SOLID_BOUNDS = ${OGC_STATUS.solidBounds}u;
const OGC_ST_PAIR_OVERFLOW = ${OGC_STATUS.pairOverflow}u;
const OGC_ST_UNCERTIFIED = ${OGC_STATUS.uncertifiedPair}u;
const OGC_PLANAR = ${OGC_TRUNCATION.planar}u;
const OGC_ST_CONTACT_OVERFLOW = ${OGC_STATUS.contactOverflow}u;
const OGC_ST_GRID_OVERFLOW = ${OGC_STATUS.gridOverflow}u;
const OGC_ST_ROUND_CAP = ${OGC_STATUS.roundCap}u;
const OGC_ST_ENVELOPE_CAP = ${OGC_STATUS.envelopeCap}u;
const OGC_ST_NONFINITE = ${OGC_STATUS.nonFinite}u;
`;

/** Geometry, activation and friction shared by detection, the vertex block and the tests. */
export const OGC_COMMON_WGSL = /* wgsl */`
const OGC_FACE = 6u;
const OGC_NONE = 0xffffffffu;
const OGC_EE_BIT = 0x80000000u;
const OGC_ID_MASK = 0x0fffffffu;

struct OgcTriClosest { bary: vec3<f32>, feature: u32 }

// Ericson's closest point on triangle abc to p, with its Voronoi feature.
fn ogcClosestTriangle(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> OgcTriClosest {
  let ab = b - a; let ac = c - a; let ap = p - a;
  let d1 = dot(ab, ap); let d2 = dot(ac, ap);
  if (d1 <= 0.0 && d2 <= 0.0) { return OgcTriClosest(vec3<f32>(1.0, 0.0, 0.0), 0u); }
  let bp = p - b; let d3 = dot(ab, bp); let d4 = dot(ac, bp);
  if (d3 >= 0.0 && d4 <= d3) { return OgcTriClosest(vec3<f32>(0.0, 1.0, 0.0), 1u); }
  let vc = d1 * d4 - d3 * d2;
  if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) { let v = d1 / (d1 - d3); return OgcTriClosest(vec3<f32>(1.0 - v, v, 0.0), 3u); }
  let cp = p - c; let d5 = dot(ab, cp); let d6 = dot(ac, cp);
  if (d6 >= 0.0 && d5 <= d6) { return OgcTriClosest(vec3<f32>(0.0, 0.0, 1.0), 2u); }
  let vb = d5 * d2 - d1 * d6;
  if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) { let w = d2 / (d2 - d6); return OgcTriClosest(vec3<f32>(1.0 - w, 0.0, w), 5u); }
  let va = d3 * d6 - d5 * d4;
  if (va <= 0.0 && d4 - d3 >= 0.0 && d5 - d6 >= 0.0) {
    let w = (d4 - d3) / ((d4 - d3) + (d5 - d6)); return OgcTriClosest(vec3<f32>(0.0, 1.0 - w, w), 4u);
  }
  let denom = 1.0 / (va + vb + vc);
  let v = vb * denom; let w = vc * denom;
  return OgcTriClosest(vec3<f32>(1.0 - v - w, v, w), OGC_FACE);
}

// Ericson's closest points between segments p0p1 and q0q1: (s, t, distance).
fn ogcSegmentSegment(p0: vec3<f32>, p1: vec3<f32>, q0: vec3<f32>, q1: vec3<f32>) -> vec3<f32> {
  let d1 = p1 - p0; let d2 = q1 - q0; let r = p0 - q0;
  let a = dot(d1, d1); let e = dot(d2, d2); let f = dot(d2, r);
  let c = dot(d1, r); let b = dot(d1, d2); let denom = a * e - b * b;
  var s = 0.0;
  if (denom > 0.0) { s = clamp((b * f - c * e) / denom, 0.0, 1.0); }
  var t = (b * s + f) / e;
  if (t < 0.0) { t = 0.0; s = clamp(-c / a, 0.0, 1.0); }
  else if (t > 1.0) { t = 1.0; s = clamp((b - c) / a, 0.0, 1.0); }
  return vec3<f32>(s, t, length(p0 + d1 * s - (q0 + d2 * t)));
}

// OGC's two-stage activation (eqs. 18-20) with tau = r/2: (g, g', g'').
fn ogcActivation(d: f32, r: f32, k: f32) -> vec3<f32> {
  if (d >= r) { return vec3<f32>(0.0); }
  let tau = 0.5 * r;
  if (d >= tau) { return vec3<f32>(0.5 * k * (r - d) * (r - d), -k * (r - d), k); }
  let k2 = k * tau * (r - tau);
  let b = 0.5 * k * (r - tau) * (r - tau) + k2 * log(tau);
  return vec3<f32>(-k2 * log(d) + b, -k2 / d, k2 / (d * d));
}

// IPC friction: f0 (energy) and f1(y)/y, with the transition width eps = eps_v * dt.
fn ogcFrictionF0(y: f32, eps: f32) -> f32 {
  if (y >= eps) { return y; }
  return -y * y * y / (3.0 * eps * eps) + y * y / eps + eps / 3.0;
}
fn ogcFrictionF1OverY(y: f32, eps: f32) -> f32 {
  if (y >= eps) { return 1.0 / y; }
  return 2.0 / eps - y / (eps * eps);
}
`;

/**
 * The static solid field (solid-field.ts) as the GPU reads it. The host defines
 *   fn ogcSolidBase() -> u32          word offset of the image (0: no solid)
 *   fn ogcSolidWord(i: u32) -> u32    word i of the buffer holding it
 */
export const OGC_SOLID_WGSL = /* wgsl */`
const OGC_SOLID_FEATURE = 7u;
const OGC_SOLID_Y = 0x70000000u;
const OGC_SOLID_HEADER_WORDS = ${SOLID_HEADER_WORDS}u;
struct OgcSolidHeader { mask: u32, cell: f32, band: f32, lmax: f32, origin: vec3<f32>, dims: vec3<u32> }
fn ogcSolidHeader() -> OgcSolidHeader {
  let b = ogcSolidBase();
  let w = vec4<u32>(ogcSolidWord(b), ogcSolidWord(b + 2u), ogcSolidWord(b + 3u), ogcSolidWord(b + 7u));
  return OgcSolidHeader(w.x, bitcast<f32>(w.y), bitcast<f32>(w.z), bitcast<f32>(w.w),
    bitcast<vec3<f32>>(vec3<u32>(ogcSolidWord(b + 4u), ogcSolidWord(b + 5u), ogcSolidWord(b + 6u))),
    vec3<u32>(ogcSolidWord(b + 8u), ogcSolidWord(b + 9u), ogcSolidWord(b + 10u)));
}
// Absolute word offset of brick b's nodes, or OGC_NONE. The table is at most half full.
fn ogcSolidBrick(h: OgcSolidHeader, b: vec3<i32>) -> u32 {
  if (any(b < vec3<i32>(0)) || any(vec3<u32>(b) >= h.dims)) { return OGC_NONE; }
  let u = vec3<u32>(b);
  let base = ogcSolidBase();
  var s = ((u.x * 73856093u) ^ (u.y * 19349663u) ^ (u.z * 83492791u)) & h.mask;
  loop {
    let at = base + OGC_SOLID_HEADER_WORDS + 4u * s;
    let rec = ogcSolidWord(at + 3u);
    if (rec == OGC_NONE) { return OGC_NONE; }
    if (ogcSolidWord(at) == u.x && ogcSolidWord(at + 1u) == u.y && ogcSolidWord(at + 2u) == u.z) { return base + rec; }
    s = (s + 1u) & h.mask;
  }
}
fn ogcSolidNode(rec: u32, i: vec3<u32>) -> f32 { return bitcast<f32>(ogcSolidWord(rec + i.x + 9u * (i.y + 9u * i.z))); }
fn ogcSolidCellOf(h: OgcSolidHeader, p: vec3<f32>) -> vec3<i32> { return vec3<i32>(floor((p - h.origin) / h.cell)); }

struct OgcSolidSample { phi: f32, grad: vec3<f32>, stored: bool }
// Trilinear phi (m) and its gradient; a point in no stored brick reads band (the field exceeds it there).
fn ogcSolidSample(h: OgcSolidHeader, p: vec3<f32>) -> OgcSolidSample {
  let q = (p - h.origin) / h.cell;
  let bq = floor(q * 0.125);
  let rec = ogcSolidBrick(h, vec3<i32>(bq));
  if (rec == OGC_NONE) { return OgcSolidSample(h.band, vec3<f32>(0.0), false); }
  let l = q - 8.0 * bq;
  let c = clamp(floor(l), vec3<f32>(0.0), vec3<f32>(7.0));
  let f = l - c;
  let ci = vec3<u32>(c);
  var phi = 0.0;
  var g = vec3<f32>(0.0);
  for (var k = 0u; k < 8u; k++) {
    let b = vec3<u32>(k & 1u, (k >> 1u) & 1u, k >> 2u);
    let bf = vec3<f32>(b);
    let w = mix(1.0 - f, f, bf);
    let v = ogcSolidNode(rec, ci + b);
    phi += v * w.x * w.y * w.z;
    g += v * (2.0 * bf - 1.0) * vec3<f32>(w.y * w.z, w.x * w.z, w.x * w.y);
  }
  return OgcSolidSample(phi, g / h.cell, true);
}

// The friction coefficient of the node nearest p (0 outside the stored bricks).
fn ogcSolidMu(h: OgcSolidHeader, p: vec3<f32>) -> f32 {
  let q = (p - h.origin) / h.cell;
  let bq = floor(q * 0.125);
  let rec = ogcSolidBrick(h, vec3<i32>(bq));
  if (rec == OGC_NONE) { return 0.0; }
  let n = vec3<u32>(clamp(floor(q - 8.0 * bq + 0.5), vec3<f32>(0.0), vec3<f32>(8.0)));
  let i = n.x + 9u * (n.y + 9u * n.z);
  let w = ogcSolidWord(rec + ${SOLID_MU_WORD}u + (i >> 2u));
  return f32((w >> (8u * (i & 3u))) & 255u) * ${(1 / SOLID_MU_SCALE).toFixed(6)};
}

// Over the ball |d| <= rho around x0: phi(x0 + d) >= phi0 + g0.d - J |d|, with g0 the
// gradient at x0 and J the largest deviation of the gradient from g0 over the cells
// the ball's box touches (per axis, d phi / d x_a over a cell spans its four a-edge
// differences). far: the whole ball lies further than r from the solid and is free of
// it: either min(phi0, band) - L_max rho >= r, or the box touches a missing brick, where
// phi > band; the upload requires band >= r + 1.5 L_max rqMax with rho <= rqMax / 2.
struct OgcSolidBound { far: bool, phi0: f32, g0: vec3<f32>, J: f32 }
fn ogcSolidBallBound(h: OgcSolidHeader, x0: vec3<f32>, rho: f32, r: f32) -> OgcSolidBound {
  let s = ogcSolidSample(h, x0);
  var out = OgcSolidBound(true, s.phi, s.grad, 0.0);
  if (min(s.phi, h.band) - h.lmax * rho >= r) { return out; }
  let c0 = ogcSolidCellOf(h, x0 - vec3<f32>(rho));
  let c1 = ogcSolidCellOf(h, x0 + vec3<f32>(rho));
  var lastBrick = vec3<i32>(-2147483647);
  var rec = OGC_NONE;
  var J = 0.0;
  let inv = 1.0 / h.cell;
  for (var cz = c0.z; cz <= c1.z; cz++) { for (var cy = c0.y; cy <= c1.y; cy++) { for (var cx = c0.x; cx <= c1.x; cx++) {
    let c = vec3<i32>(cx, cy, cz);
    let b = c >> vec3<u32>(3u);
    if (any(b != lastBrick)) { lastBrick = b; rec = ogcSolidBrick(h, b); }
    if (rec == OGC_NONE) { return out; }
    let l = vec3<u32>(c - 8 * b);
    var n: array<f32, 8>;
    for (var k = 0u; k < 8u; k++) { n[k] = ogcSolidNode(rec, l + vec3<u32>(k & 1u, (k >> 1u) & 1u, k >> 2u)); }
    // a-edges: corner pairs (k, k | bit a) over the four k without bit a.
    var lo = vec3<f32>(3.4e38); var hi = vec3<f32>(-3.4e38);
    for (var k = 0u; k < 8u; k++) {
      if ((k & 1u) == 0u) { let d = n[k | 1u] - n[k]; lo.x = min(lo.x, d); hi.x = max(hi.x, d); }
      if ((k & 2u) == 0u) { let d = n[k | 2u] - n[k]; lo.y = min(lo.y, d); hi.y = max(hi.y, d); }
      if ((k & 4u) == 0u) { let d = n[k | 4u] - n[k]; lo.z = min(lo.z, d); hi.z = max(hi.z, d); }
    }
    let dev = max(abs(hi * inv - s.grad), abs(s.grad - lo * inv));
    J = max(J, length(dev));
  } } }
  out.far = false;
  out.J = J;
  return out;
}
`;

export interface OgcVertexBlockOptions {
  /** Bind group the solver kernel puts the contact buffer in. */
  group: number;
  /** Binding of the contact buffer (read). */
  binding: number;
}

/**
 * The block a solver's per-vertex kernel includes. The host defines, before
 * this block, the accessors
 *   fn ogcX(i: u32) -> vec3<f32>         current position of vertex i
 *   fn ogcXt(i: u32) -> vec3<f32>        position at the start of the step
 *   fn ogcTri(t: u32) -> vec3<u32>       face vertices
 *   fn ogcEdgeVerts(e: u32) -> vec2<u32> edge vertices
 * and binds OgcContactDetector.contacts at (group, binding).
 * Provided:
 *   ogcAccumulateVertexContact(v, xv, &f, &H)  force and PSD Hessian of v's contacts at xv
 *   ogcVertexContactEnergy(v, xv) -> f32       their energy, for the vertex line search
 */
export function ogcVertexBlockWGSL(options: OgcVertexBlockOptions): string {
  const { group, binding } = options;
  return /* wgsl */`
${OGC_COMMON_WGSL}
@group(${group}) @binding(${binding}) var<storage, read> ogcContacts: array<vec4<u32>>;
fn ogcSolidBase() -> u32 { return ogcContacts[2].x; }
fn ogcSolidWord(i: u32) -> u32 { return ogcContacts[i >> 2u][i & 3u]; }
${OGC_SOLID_WGSL}

struct OgcPairEval { f: vec3<f32>, h: mat3x3<f32>, e: f32 }

// Vertex v against the static solid field at xv: the activation on phi with the
// Gauss-Newton Hessian g'' grad phi grad phi^T, and lagged friction against the
// fixed solid. phi <= 0 lies outside the barrier's domain: infinite energy.
fn ogcSolidEval(v: u32, xv: vec3<f32>, ent: vec4<u32>, params: vec4<f32>) -> OgcPairEval {
  let s = ogcSolidSample(ogcSolidHeader(), xv);
  var out = OgcPairEval(vec3<f32>(0.0), mat3x3<f32>(vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0)), 0.0);
  if (!(s.phi > 0.0)) { out.e = bitcast<f32>(ent.w | 0x7f800000u); return out; }
  let act = ogcActivation(s.phi, params.x, params.y);
  let g = s.grad;
  out.f = -act.y * g;
  out.h = act.z * mat3x3<f32>(g * g.x, g * g.y, g * g.z);
  out.e = act.x;
  // The entry carries mu lambda, mu the solid's own coefficient at x_det.
  let ml = bitcast<f32>(ent.z);
  if (ml > 0.0) {
    let gl = length(g);
    let n = select(vec3<f32>(0.0), g / gl, gl > 1e-30);
    let nn = mat3x3<f32>(n * n.x, n * n.y, n * n.z);
    let dx = xv - ogcXt(v);
    let u = dx - n * dot(n, dx);
    let un = length(u);
    let fy = ogcFrictionF1OverY(un, params.w);
    out.f -= (ml * fy) * u;
    out.h += (ml * fy) * (mat3x3<f32>(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0) - nn);
    out.e += ml * ogcFrictionF0(un, params.w);
  }
  return out;
}

// One contact entry evaluated for vertex v placed at xv.
fn ogcEvalEntry(v: u32, xv: vec3<f32>, ent: vec4<u32>, params: vec4<f32>) -> OgcPairEval {
  if (ent.y == OGC_SOLID_Y) { return ogcSolidEval(v, xv, ent, params); }
  var ids: vec4<u32>;
  var w = vec4<f32>(0.0);
  var X: array<vec3<f32>, 4>;
  if ((ent.y & OGC_EE_BIT) == 0u) {
    let feature = (ent.y >> 28u) & 7u;
    ids = vec4<u32>(ent.x, ogcTri(ent.y & OGC_ID_MASK));
    for (var i = 0u; i < 4u; i++) { X[i] = select(ogcX(ids[i]), xv, ids[i] == v); }
    let p = X[0];
    w.x = 1.0;
    if (feature < 3u) {
      w[1u + feature] = -1.0;
    } else if (feature < OGC_FACE) {
      let i = feature - 3u; let j = (i + 1u) % 3u;
      let dir = X[1u + j] - X[1u + i];
      let s = dot(p - X[1u + i], dir) / dot(dir, dir);
      w[1u + i] = -(1.0 - s); w[1u + j] = -s;
    } else {
      let v0 = X[2] - X[1]; let v1 = X[3] - X[1]; let v2 = p - X[1];
      let d00 = dot(v0, v0); let d01 = dot(v0, v1); let d11 = dot(v1, v1);
      let d20 = dot(v2, v0); let d21 = dot(v2, v1);
      let den = d00 * d11 - d01 * d01;
      let bv = (d11 * d20 - d01 * d21) / den; let bw = (d00 * d21 - d01 * d20) / den;
      w.y = -(1.0 - bv - bw); w.z = -bv; w.w = -bw;
    }
  } else {
    let e0 = ogcEdgeVerts(ent.x); let e1 = ogcEdgeVerts(ent.y & OGC_ID_MASK);
    ids = vec4<u32>(e0, e1);
    for (var i = 0u; i < 4u; i++) { X[i] = select(ogcX(ids[i]), xv, ids[i] == v); }
    let st = unpack2x16unorm(ent.w);
    w = vec4<f32>(1.0 - st.x, st.x, -(1.0 - st.y), -st.y);
  }
  var gap = vec3<f32>(0.0);
  var wr = 0.0;
  for (var i = 0u; i < 4u; i++) { gap += w[i] * X[i]; wr += select(0.0, w[i], ids[i] == v); }
  let d = length(gap);
  let n = gap / d;
  let act = ogcActivation(d, params.x, params.y);
  let nn = mat3x3<f32>(n * n.x, n * n.y, n * n.z);
  var out = OgcPairEval(-act.y * wr * n, (wr * wr * act.z) * nn, act.x);
  let mu = params.z;
  if (mu > 0.0) {
    let ml = mu * bitcast<f32>(ent.z);
    var dx = vec3<f32>(0.0);
    for (var i = 0u; i < 4u; i++) { dx += w[i] * (X[i] - ogcXt(ids[i])); }
    let u = dx - n * dot(n, dx);
    let un = length(u);
    let fy = ogcFrictionF1OverY(un, params.w);
    out.f -= (ml * wr * fy) * u;
    out.h += (ml * wr * wr * fy) * (mat3x3<f32>(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0) - nn);
    out.e += ml * ogcFrictionF0(un, params.w);
  }
  return out;
}

fn ogcContactCount(v: u32) -> u32 {
  let header = ogcContacts[0];
  return min(ogcContacts[header.z + (v >> 2u)][v & 3u], header.y);
}

fn ogcAccumulateVertexContact(v: u32, xv: vec3<f32>, f: ptr<function, vec3<f32>>, H: ptr<function, mat3x3<f32>>) {
  let header = ogcContacts[0];
  let params = bitcast<vec4<f32>>(ogcContacts[1]);
  let count = min(ogcContacts[header.z + (v >> 2u)][v & 3u], header.y);
  let base = header.w + v * header.y;
  for (var i = 0u; i < count; i++) {
    let r = ogcEvalEntry(v, xv, ogcContacts[base + i], params);
    *f += r.f; *H += r.h;
  }
}

fn ogcVertexContactEnergy(v: u32, xv: vec3<f32>) -> f32 {
  let header = ogcContacts[0];
  let params = bitcast<vec4<f32>>(ogcContacts[1]);
  let count = min(ogcContacts[header.z + (v >> 2u)][v & 3u], header.y);
  let base = header.w + v * header.y;
  var e = 0.0;
  for (var i = 0u; i < count; i++) { e += ogcEvalEntry(v, xv, ogcContacts[base + i], params).e; }
  return e;
}
`;
}
