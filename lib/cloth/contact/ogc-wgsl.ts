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
 *   vertex-body   x = v, y = OGC_BODY_Y (feature 7, face 1): the moving solid, likewise
 *   sample-facet  x = N + s for surface sample s, y = face | feature << 28,
 *                 z = mu lambda (the solid's mu), w = 1; held by the face's feature vertices.
 *                 s < S is a static sample; S <= s < S + SB is the moving solid's sample s - S,
 *                 posed by the step's end pose.
 * Feature: vertex k (0..2), edge (t[k], t[k+1]) (3..5), face interior (6), static solid (7).
 * Every entry is self-describing, so the gather never chases a second list.
 *
 * Contact buffer header (vec4<u32>): [0] (N, list capacity, counts base, entries base),
 * [1] (r, k_c, mu, eps_v dt) as f32, [2] (solid image word offset, 0 when none, the surface
 * samples' vec4 offset, their count S, 0); sample s is (x, y, z, mu) as f32.
 * [3] (the moving solid's image word offset, 0 when none, its body-frame samples' vec4
 * offset, their count SB, the pose table's vec4 offset), [4..7] the step's start pose and
 * [8..11] its end pose (OGC_POSE_WORDS each: centre, then the rotation's columns), which the
 * step's begin copies from the table the frame's owner uploads.
 */
import { SOLID_CLEARANCE_CELLS, SOLID_HEADER_WORDS, SOLID_MU_SCALE, SOLID_MU_WORD } from "./solid-field";


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
  /** Sample-face contacts of the frame's rounds (static surface samples against cloth faces). */
  frameSfPairs: 37,
  /** Cloth faces that reach inside the static solid (the intersection count's solid half). */
  solidIntersections: 38,
  /** The round's largest envelope (f32 bits, set by the cell kernel): every face's grid box is grown by it. */
  roundEnvelope: 39,
  /** Vertices with a moving-solid entry over the frame's rounds, and the truncations its bound made. */
  frameBodyContacts: 40, frameBodyCut: 41,
  /** A pair (words 42..45, ids as in the pair buffer: EE bit on an edge-edge pair) that conflicted in the held round. */
  heldConflictPair: 42,
  /**
   * Impact zones with the moving solid over the frame: cloth vertices a conflict carried with
   * it (each once per step), and the largest such zone of one step (all its zones together).
   */
  frameZoneVertices: 46, frameZonePeak: 47,
  /** The current step's zone vertices (each redo decision folds it into frameZonePeak; the step's begin clears it). */
  stepZone: 48,
  redoProgress: 49,
  /** The line-search filter: vertices it held at alpha_v = 0 (consumed by the decision, which then re-detects). */
  filterStalled: 50,
  /** The round's longest vertex prediction |w| (f32 bits; consumed by the cell kernel, which sizes the cell by it). */
  maxPrediction: 51,
  /** The line-search filter over the frame's Newton iterations: vertices whose solid bounds it re-took about the iterate. */
  frameRebound: 52,
  /**
   * The filter's plane records (ogcProject): this filter's total (its allocation counter), their
   * persistent peak against the authored planeRecordCapacity, and the frame's peak.
   */
  planeRecords: 53, peakPlaneRecords: 54, framePeakPlaneRecords: 55,
} as const;
export const OGC_COUNTER_WORDS = 56;
export const OGC_FRAME_COUNTERS: readonly number[] = [8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 25, 26, 27, 28, 29, 30, 33, 34, 35, 37, 40, 41, 46, 47, 52, 55];
/** vec4 words of a vertex's fixed certificate record for ogcProject: the cap, then (origin, q, D) per solid. */
export const OGC_PROJECT_RECORD = 7;

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
  /**
   * A vertex's prediction cannot keep both the moving solid's bound and the static
   * solid's: the static cut takes it back into the moving solid's (cloth pressed
   * between them, nearer than about 2 r).
   */
  bodySqueeze: 512,
  /** The round redone held still had crossing predictions: cloth the moving solid pushes together. */
  heldConflict: 1024,
  /** A step ran past the poses the frame's owner uploaded for the moving solid. */
  bodyPose: 2048,
  /** Two primitives of a contact pair coincided at detection (distance 0): no contact normal exists. */
  coincident: 4096,
  /**
   * Reported, not a failure: the least-shift redos of a step's mandatory round left a conflict,
   * so the fail-safe ran: the round redone with every prediction held (Planar-DAT) and, with a
   * moving solid, the conflicts' cloth carried rigidly with it (Bridson, Fedkiw, Anderson 2002,
   * sec. 7.4). Never the normal path: a scene that reports it is a finding.
   */
  failSafe: 8192,
  /** The filter's plane records (ogcProject) overflowed the authored planeRecordCapacity: the vertices past it were held. */
  planeRecordOverflow: 16384,
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
const OGC_MAX_PREDICTION = ${OGC_COUNTER.maxPrediction}u;
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
const OGC_F_SF_PAIRS = ${OGC_COUNTER.frameSfPairs}u;
const OGC_SOLID_INTERSECTIONS = ${OGC_COUNTER.solidIntersections}u;
const OGC_ROUND_ENVELOPE = ${OGC_COUNTER.roundEnvelope}u;
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
const OGC_ST_BODY_SQUEEZE = ${OGC_STATUS.bodySqueeze}u;
const OGC_ST_HELD_CONFLICT = ${OGC_STATUS.heldConflict}u;
const OGC_ST_BODY_POSE = ${OGC_STATUS.bodyPose}u;
const OGC_ST_COINCIDENT = ${OGC_STATUS.coincident}u;
const OGC_ST_FAIL_SAFE = ${OGC_STATUS.failSafe}u;
const OGC_F_BODY_CONTACTS = ${OGC_COUNTER.frameBodyContacts}u;
const OGC_F_BODY_CUT = ${OGC_COUNTER.frameBodyCut}u;
const OGC_HELD_PAIR = ${OGC_COUNTER.heldConflictPair}u;
const OGC_F_ZONE_VERTICES = ${OGC_COUNTER.frameZoneVertices}u;
const OGC_F_ZONE_PEAK = ${OGC_COUNTER.frameZonePeak}u;
const OGC_STEP_ZONE = ${OGC_COUNTER.stepZone}u;
const OGC_REDO_PROGRESS = ${OGC_COUNTER.redoProgress}u;
const OGC_FILTER_STALLED = ${OGC_COUNTER.filterStalled}u;
const OGC_F_REBOUND = ${OGC_COUNTER.frameRebound}u;
const OGC_PLANE_RECORDS = ${OGC_COUNTER.planeRecords}u;
const OGC_PEAK_PLANE_RECORDS = ${OGC_COUNTER.peakPlaneRecords}u;
const OGC_F_PEAK_PLANE_RECORDS = ${OGC_COUNTER.framePeakPlaneRecords}u;
const OGC_ST_PLANE_RECORD_OVERFLOW = ${OGC_STATUS.planeRecordOverflow}u;
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

// A solid's affine certificate on the motion u from its origin: f(u) = D.|u| - q.u <= 1 within
// the ball |u| <= R (q.w), with f convex, subadditive and positively homogeneous. The largest
// k <= 1 it keeps for base + k delta: room 1 - f(base) over f(delta), and the ball; none from a
// base outside the ball (the room is 0 from a base past f = 1). q = D = 0 is the plain ball.
fn ogcPieceScale(q: vec4<f32>, D: vec3<f32>, base: vec3<f32>, delta: vec3<f32>) -> f32 {
  // The prediction scale leaves a base on a ball's sphere to f32 rounding.
  if (dot(base, base) > q.w * q.w * (1.0 + 1.0e-5)) { return 0.0; }
  let room = max(1.0 - (dot(D, abs(base)) - dot(q.xyz, base)), 0.0);
  let den = dot(D, abs(delta)) - dot(q.xyz, delta);
  return min(select(1.0, room / den, den > room), ogcBallExit(base, delta, q.w));
}
// The largest scale k <= 1 of delta that one of a solid bound's certificates keeps for the motion
// base + k delta (OgcTerms: q, (D, rl)): the affine one (ogcPieceScale), or the Lipschitz ball
// |base + k delta| <= rl when it holds the base. No bound (q.w = 0) keeps all; an inside marker
// (q.w >= 1e38) allows no motion.
fn ogcCutScale(q: vec4<f32>, D4: vec4<f32>, base: vec3<f32>, delta: vec3<f32>) -> f32 {
  if (q.w >= 1.0e38) { return select(1.0, 0.0, any(delta != vec3<f32>(0.0))); }
  if (q.w == 0.0) { return 1.0; }
  var k = ogcPieceScale(q, D4.xyz, base, delta);
  if (k < 1.0 && dot(base, base) <= D4.w * D4.w * (1.0 + 1.0e-5)) { k = max(k, ogcBallExit(base, delta, D4.w)); }
  return k;
}
// The largest t <= 1 with |base + t delta| <= R, for a base within R.
fn ogcBallExit(base: vec3<f32>, delta: vec3<f32>, R: f32) -> f32 {
  let e = base + delta;
  let r2 = R * R;
  if (dot(e, e) <= r2) { return 1.0; }
  let dd = dot(delta, delta);
  let bd = dot(base, delta);
  return clamp((sqrt(max(bd * bd + dd * (r2 - dot(base, base)), 0.0)) - bd) / dd, 0.0, 1.0);
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

/** Words of one pose in the contact buffer: the centre (xyz, 0), then the rotation's three columns. */
export const OGC_POSE_WORDS = 16;
/** Contact-buffer word of the step's start pose; its end pose follows. */
export const OGC_POSE_WORD = 16;

/**
 * Solid fields (solid-field.ts) as the GPU reads them: the static one, and the
 * moving one in its own body frame. The host defines
 *   fn ogcSolidBase() -> u32          word offset of the static image (0: no solid)
 *   fn ogcSolidWord(i: u32) -> u32    word i of the contact buffer holding them
 * The moving solid's image offset is header word 12 (0: none); the step's start
 * and end poses sit at OGC_POSE_WORD.
 */
export const OGC_SOLID_WGSL = /* wgsl */`
const OGC_SOLID_FEATURE = 7u;
const OGC_SOLID_Y = 0x70000000u;
const OGC_BODY_Y = 0x70000001u;
const OGC_SOLID_HEADER_WORDS = ${SOLID_HEADER_WORDS}u;
const OGC_POSE_WORD = ${OGC_POSE_WORD}u;
const OGC_POSE_WORDS = ${OGC_POSE_WORDS}u;
struct OgcSolidHeader { base: u32, mask: u32, cell: f32, band: f32, lmax: f32, origin: vec3<f32>, dims: vec3<u32> }
fn ogcSolidHeaderAt(b: u32) -> OgcSolidHeader {
  let w = vec4<u32>(ogcSolidWord(b), ogcSolidWord(b + 2u), ogcSolidWord(b + 3u), ogcSolidWord(b + 7u));
  return OgcSolidHeader(b, w.x, bitcast<f32>(w.y), bitcast<f32>(w.z), bitcast<f32>(w.w),
    bitcast<vec3<f32>>(vec3<u32>(ogcSolidWord(b + 4u), ogcSolidWord(b + 5u), ogcSolidWord(b + 6u))),
    vec3<u32>(ogcSolidWord(b + 8u), ogcSolidWord(b + 9u), ogcSolidWord(b + 10u)));
}
fn ogcSolidHeader() -> OgcSolidHeader { return ogcSolidHeaderAt(ogcSolidBase()); }
fn ogcBodyBase() -> u32 { return ogcSolidWord(12u); }
// A rigid pose: x_world = R x_body + c.
struct OgcPose { c: vec3<f32>, R: mat3x3<f32> }
fn ogcWordF(i: u32) -> f32 { return bitcast<f32>(ogcSolidWord(i)); }
fn ogcWordV(i: u32) -> vec3<f32> { return vec3<f32>(ogcWordF(i), ogcWordF(i + 1u), ogcWordF(i + 2u)); }
// k = 0: the step's start pose; k = 1: its end pose.
fn ogcPose(k: u32) -> OgcPose {
  let b = OGC_POSE_WORD + OGC_POSE_WORDS * k;
  return OgcPose(ogcWordV(b), mat3x3<f32>(ogcWordV(b + 4u), ogcWordV(b + 8u), ogcWordV(b + 12u)));
}
fn ogcToBody(p: OgcPose, x: vec3<f32>) -> vec3<f32> { return transpose(p.R) * (x - p.c); }
fn ogcToWorld(p: OgcPose, y: vec3<f32>) -> vec3<f32> { return p.R * y + p.c; }
// Absolute word offset of brick b's nodes, or OGC_NONE. The table is at most half full.
fn ogcSolidBrick(h: OgcSolidHeader, b: vec3<i32>) -> u32 {
  if (any(b < vec3<i32>(0)) || any(vec3<u32>(b) >= h.dims)) { return OGC_NONE; }
  let u = vec3<u32>(b);
  let base = h.base;
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

// A certified lower bound of the field at p, min(phi, band). Inside the lattice a missing brick
// reads band, which it exceeds (bricks are left out only when every node lies beyond band). Outside
// it the image holds nothing: a moving body's field is clear there (header word 12), but past a
// scene field's box the scene's shell or the rest of a solid may lie, so it bounds nothing (-3e38).
fn ogcSolidPhiLower(h: OgcSolidHeader, p: vec3<f32>) -> f32 {
  let b = floor((p - h.origin) / (8.0 * h.cell));
  if (any(b < vec3<f32>(0.0)) || any(b >= vec3<f32>(h.dims))) { return select(-3.0e38, h.band, ogcSolidWord(h.base + 12u) != 0u); }
  return min(ogcSolidSample(h, p).phi, h.band);
}

// The field at every point within rho of p is at least min(band, ogcSolidClearance(h, p) - rho)
// (solid-field.ts: K_b - |p - c_b| of p's clearance block, clamped into the table; -3e38
// for an image without the table).
fn ogcSolidClearance(h: OgcSolidHeader, p: vec3<f32>) -> f32 {
  let t = ogcSolidWord(h.base + 11u);
  if (t == 0u) { return -3.0e38; }
  let span = ${SOLID_CLEARANCE_CELLS}.0 * h.cell;
  let n = (8u / ${SOLID_CLEARANCE_CELLS}u) * h.dims;
  let b = vec3<u32>(clamp(floor((p - h.origin) / span), vec3<f32>(0.0), vec3<f32>(n - vec3<u32>(1u))));
  let c = h.origin + (vec3<f32>(b) + 0.5) * span;
  return bitcast<f32>(ogcSolidWord(h.base + t + b.x + n.x * (b.y + n.y * b.z))) - length(p - c);
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

// Over the ball |d| <= rho around x0: phi(x0 + d) >= phi0 + g0.d - sum_a D_a |d_a|, with
// g0 the gradient at x0 and D_a the largest deviation of d phi / d x_a from g0_a over the
// cells the ball's box touches (over a cell it spans its four a-edge differences): the
// path integral of (grad - g0).d, axis by axis. far: the whole ball lies further than r from the solid and is free of
// it: either min(phi0, band) - L_max rho >= r, or the box touches a missing brick, where
// phi > band; the upload requires band >= r + 1.5 L_max rqMax with rho <= rqMax / 2.
// Cells outside the lattice read as a missing brick when the field's exterior is clear (header
// word 12: every part lies band inside the box, a moving body's field). Otherwise they are no
// missing bricks: the scene container's shell lies there, just past the box. They then add
// nothing to D: a path that leaves the lattice crosses its boundary in a stored cell (where the
// bound holds, and phi is 0 on a shell's face) or in a missing one, and a vertex outside the
// lattice fails the next round as OGC_ST_SOLID_BOUNDS.
struct OgcSolidBound { far: bool, phi0: f32, g0: vec3<f32>, D: vec3<f32> }
fn ogcSolidBallBound(h: OgcSolidHeader, x0: vec3<f32>, rho: f32, r: f32) -> OgcSolidBound {
  let s = ogcSolidSample(h, x0);
  var out = OgcSolidBound(true, s.phi, s.grad, vec3<f32>(0.0));
  if (ogcSolidPhiLower(h, x0) - h.lmax * rho >= r) { return out; }
  let c0 = ogcSolidCellOf(h, x0 - vec3<f32>(rho));
  let c1 = ogcSolidCellOf(h, x0 + vec3<f32>(rho));
  var lastBrick = vec3<i32>(-2147483647);
  var rec = OGC_NONE;
  var D = vec3<f32>(0.0);
  let inv = 1.0 / h.cell;
  let clearOutside = ogcSolidWord(h.base + 12u) != 0u;
  for (var cz = c0.z; cz <= c1.z; cz++) { for (var cy = c0.y; cy <= c1.y; cy++) { for (var cx = c0.x; cx <= c1.x; cx++) {
    let c = vec3<i32>(cx, cy, cz);
    let b = c >> vec3<u32>(3u);
    if (any(b < vec3<i32>(0)) || any(vec3<u32>(b) >= h.dims)) { if (clearOutside) { return out; } continue; }
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
    D = max(D, dev);
  } } }
  out.far = false;
  out.D = D;
  return out;
}
`;

export interface OgcVertexBlockOptions {
  /** Bind group the solver kernel puts the contact buffer in. */
  group: number;
  /** Binding of the contact buffer (read). */
  binding: number;
  /**
   * Also the global (Newton) forms, ogcVertexContactHvLanes and ogcContactEnergyOwnedLanes;
   * the host then defines fn ogcDir(i: u32) -> vec3<f32> too, the vector H multiplies.
   */
  newton?: boolean;
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
 * and their lane-split forms for a solver that spreads one vertex over several lanes:
 *   ogcVertexContactLanes(v, xv, first, stride, &f, &H) -> f32
 *       adds the force and Hessian of v's entries first, first + stride, ... and returns
 *       their energy
 *   ogcVertexContactEnergyLanes(v, xv, first, stride) -> f32
 *       the energy of those entries
 * The caller sums the lanes (first = 0 .. stride - 1); stride 1 from 0 is the whole block.
 */
/**
 * The projection of a trial point y of vertex v onto its certified set (the projected arc of a
 * Newton line search, Bertsekas' projected Newton): the intersection of the convex certificates
 * the detector's line-search filter recorded about the iterate x for this Newton iteration
 * (OGC_FILTER_WGSL and its plane records), each of which holds x. All are relative to the cap's
 * centre c = X + w:
 *   the cap        |z| <= R;
 *   each DAT plane  m.z >= b, m = sigma n its side's unit normal and b = -gamma_r max(s0, 0)
 *                   (the slack keeps the truncation's own target); m = 0: no separating plane, held;
 *   each solid     f(u) = D.|u| - q.u <= 1 within |u| <= R (q.w), u = y - O (O its world
 *                   origin), in the moving solid's frame at the step's end pose for the moving
 *                   one (ogcPieceScale).
 * Three sweeps of alternating projections (the cap's radial one, a plane's orthogonal one, a
 * solid's along its active piece's gradient D o sign(u) - q, then its ball), and then,
 * since alternating projections only approach the intersection, the largest t <= 1 of the
 * segment from x to that point that every certificate keeps exactly as the filter does: each
 * is convex and holds x, so x + t (p - x) is certified. A trial the filter's own test keeps
 * whole along y - x (ogcArcScale, the filter's operands in the filter's order) is returned
 * bit-exactly, since a solver reads q != y as a cut; a vertex held (an inside marker, a pair
 * with no plane, records past their capacity) does not move.
 */
const OGC_PROJECT_WGSL = /* wgsl */`
fn ogcProj4(i: u32) -> vec4<f32> { return bitcast<vec4<f32>>(ogcContacts[i]); }
// One solid's projection of p onto f(p - o) <= 1 within |p - o| <= R, with Rt to its frame.
// A point the piece holds comes back bit-exactly (the round trip through the frame would not).
fn ogcProjectPiece(p: vec3<f32>, o: vec3<f32>, q: vec4<f32>, D: vec3<f32>, Rt: mat3x3<f32>) -> vec3<f32> {
  var u = Rt * (p - o);
  var moved = false;
  let f = dot(D, abs(u)) - dot(q.xyz, u);
  if (f > 1.0) {
    let g = D * sign(u) - q.xyz;
    let gg = dot(g, g);
    if (gg > 0.0) { u -= ((f - 1.0) / gg) * g; moved = true; }
  }
  let l = length(u);
  if (l > q.w) { u *= q.w / l; moved = true; }
  if (!moved) { return p; }
  return o + transpose(Rt) * u;
}
// The largest t <= 1 with x + t e inside every certificate of the vertex's records at fb (and its
// n plane records from first), each evaluated exactly as the line-search filter evaluates it along
// d (same operands, same order), so a trial the filter keeps whole is kept here bit for bit.
fn ogcArcScale(fb: u32, first: u32, n: u32, x: vec3<f32>, e: vec3<f32>, Rb: mat3x3<f32>) -> f32 {
  let cap = ogcProj4(fb);
  let zx = x - cap.xyz;
  var t = ogcBallExit(zx, e, cap.w);
  for (var i = 0u; i < n; i++) {
    let r = ogcProj4(first + i);
    let s = dot(r.xyz, e);
    if (s < 0.0) { t = min(t, clamp((dot(r.xyz, zx) - r.w) / -s, 0.0, 1.0)); }
  }
  let so = ogcProj4(fb + 1u);
  if (so.w != 0.0) { t = min(t, ogcPieceScale(ogcProj4(fb + 2u), ogcProj4(fb + 3u).xyz, x - so.xyz, e)); }
  let bo = ogcProj4(fb + 4u);
  if (bo.w != 0.0) { t = min(t, ogcPieceScale(ogcProj4(fb + 5u), ogcProj4(fb + 6u).xyz, Rb * (x - bo.xyz), Rb * e)); }
  return t;
}
fn ogcProject(v: u32, x: vec3<f32>, y: vec3<f32>) -> vec3<f32> {
  let pb = ogcContacts[2].w;
  if (pb == 0u) { return x; }
  let ph = ogcContacts[pb];
  let fb = ph.x + ${OGC_PROJECT_RECORD}u * v;
  let n = ogcSolidWord(ph.y + 2u * v + 1u);
  if (n == OGC_NONE) { return x; }
  let first = ph.z + ogcSolidWord(ph.y + 2u * v);
  let so = ogcProj4(fb + 1u); let bo = ogcProj4(fb + 4u);
  if (so.w == 2.0 || bo.w == 2.0) { return x; }
  for (var i = 0u; i < n; i++) {
    if (all(ogcProj4(first + i).xyz == vec3<f32>(0.0))) { return x; }
  }
  let Rb = transpose(ogcPose(1u).R);
  // A trial every certificate keeps whole (the filter's own test along y - x) is the point itself.
  if (ogcArcScale(fb, first, n, x, y - x, Rb) >= 1.0) { return y; }
  let cap = ogcProj4(fb);
  let sq = ogcProj4(fb + 2u); let sD = ogcProj4(fb + 3u).xyz;
  let bq = ogcProj4(fb + 5u); let bD = ogcProj4(fb + 6u).xyz;
  var p = y;
  for (var sweep = 0u; sweep < 3u; sweep++) {
    let z = p - cap.xyz;
    let l = length(z);
    if (l > cap.w) { p = cap.xyz + z * (cap.w / l); }
    for (var i = 0u; i < n; i++) {
      let r = ogcProj4(first + i);
      let e = dot(r.xyz, p - cap.xyz) - r.w;
      if (e < 0.0) { p -= e * r.xyz; }
    }
    if (so.w != 0.0) { p = ogcProjectPiece(p, so.xyz, sq, sD, OGC_I3); }
    if (bo.w != 0.0) { p = ogcProjectPiece(p, bo.xyz, bq, bD, Rb); }
  }
  let e = p - x;
  return x + ogcArcScale(fb, first, n, x, e, Rb) * e;
}
`;

export function ogcVertexBlockWGSL(options: OgcVertexBlockOptions): string {
  const { group, binding } = options;
  return /* wgsl */`
${OGC_COMMON_WGSL}
@group(${group}) @binding(${binding}) var<storage, read> ogcContacts: array<vec4<u32>>;
fn ogcSolidBase() -> u32 { return ogcContacts[2].x; }
fn ogcSolidWord(i: u32) -> u32 { return ogcContacts[i >> 2u][i & 3u]; }
${OGC_SOLID_WGSL}
${OGC_PROJECT_WGSL}
// Surface sample at index i >= N (sample-facet entries): static below N + S, else the
// moving solid's body-frame sample posed at the step's end, and where it started the step.
fn ogcSampleRel(i: u32) -> u32 { return i - ogcContacts[0].x; }
fn ogcSampleLocal(i: u32) -> vec3<f32> {
  let s = ogcSampleRel(i); let S = ogcContacts[2].z;
  if (s < S) { return bitcast<vec3<f32>>(ogcContacts[ogcContacts[2].y + s].xyz); }
  return bitcast<vec3<f32>>(ogcContacts[ogcContacts[3].y + s - S].xyz);
}
fn ogcSamplePos(i: u32) -> vec3<f32> {
  if (ogcSampleRel(i) < ogcContacts[2].z) { return ogcSampleLocal(i); }
  return ogcToWorld(ogcPose(1u), ogcSampleLocal(i));
}
fn ogcSampleStart(i: u32) -> vec3<f32> {
  if (ogcSampleRel(i) < ogcContacts[2].z) { return ogcSampleLocal(i); }
  return ogcToWorld(ogcPose(0u), ogcSampleLocal(i));
}

struct OgcPairEval { f: vec3<f32>, h: mat3x3<f32>, e: f32 }

// One contact entry linearized for vertex v placed at xv (the others at ogcX). Its energy is
// the barrier kappa(d) on the gap g = sum_i w_i x_i (w at the current positions: the closest
// point, so its gradient is kappa' w n) plus lagged friction mu lambda f0(|u|) on the slip
// u = T sum_i wl_i (x_i - x_t,i), with the weights wl, the normal nl and T = I - nl nl^T all
// taken at the step start (IPC's lagged friction): u is linear in x, so the friction force is
// the exact gradient of its energy and a Newton line search sees one function. The Gauss-Newton
// Hessian is (w w^T) (x) Mn + (wl wl^T) (x) Mf, Mn = kappa'' n n^T and Mf = mu lambda (f1/|u|) T,
// PSD by construction. ids: the pair's vertices (a solid's sample, or none, past the cloth's);
// wc, wlc: the weights on the cloth vertices only (0 at a sample); wr, wrl: v's own weights;
// f: the force on v; e: the pair's energy. A field entry is the one-vertex case (w = (1)).
struct OgcPairLin { ids: vec4<u32>, wc: vec4<f32>, wlc: vec4<f32>, wr: f32, wrl: f32, f: vec3<f32>, Mn: mat3x3<f32>, Mf: mat3x3<f32>, e: f32 }

const OGC_I3 = mat3x3<f32>(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
const OGC_Z3 = mat3x3<f32>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);

fn ogcUnitOr(g: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
  let l = length(g);
  return select(fallback, g / l, l > 1e-30);
}

// Friction on the slip dx (lagged weights already applied) with the lagged unit normal nl:
// adds to the pair its force on v (weight wrl), Mf and energy.
fn ogcFriction(out: ptr<function, OgcPairLin>, ml: f32, dx: vec3<f32>, nl: vec3<f32>, eps: f32) {
  let u = dx - nl * dot(nl, dx);
  let un = length(u);
  let fy = ogcFrictionF1OverY(un, eps);
  (*out).f -= (ml * (*out).wrl * fy) * u;
  (*out).Mf = (ml * fy) * (OGC_I3 - mat3x3<f32>(nl * nl.x, nl * nl.y, nl * nl.z));
  (*out).e += ml * ogcFrictionF0(un, eps);
}

// A vertex against a solid field at a point where it reads phi with world gradient g, having
// slipped dx against the solid's surface over the step, with the field's normal at the step
// start nl: the activation on phi with the Gauss-Newton Hessian kappa'' g g^T, and lagged
// friction on the tangential slip. phi <= 0 lies outside the barrier's domain: infinite energy.
fn ogcFieldEval(v: u32, phi: f32, g: vec3<f32>, dx: vec3<f32>, nl: vec3<f32>, ent: vec4<u32>, params: vec4<f32>) -> OgcPairLin {
  let one = vec4<f32>(1.0, 0.0, 0.0, 0.0);
  var out = OgcPairLin(vec4<u32>(v, OGC_NONE, OGC_NONE, OGC_NONE), one, one, 1.0, 1.0, vec3<f32>(0.0), OGC_Z3, OGC_Z3, 0.0);
  // Infinity from a runtime word (ent.w is 0): a constant one would not fold.
  if (!(phi > 0.0)) { out.e = bitcast<f32>(ent.w | 0x7f800000u); out.wr = 0.0; out.wrl = 0.0; return out; }
  let act = ogcActivation(phi, params.x, params.y);
  out.f = -act.y * g;
  out.Mn = act.z * mat3x3<f32>(g * g.x, g * g.y, g * g.z);
  out.e = act.x;
  // The entry carries mu lambda, mu the solid's own coefficient at x_det.
  let ml = bitcast<f32>(ent.z);
  if (ml > 0.0) { ogcFriction(&out, ml, dx, nl, params.w); }
  return out;
}

// Vertex v against the static solid: the slip is its whole step x - x_t, the lagged normal the field's at x_t.
fn ogcSolidLin(v: u32, xv: vec3<f32>, ent: vec4<u32>, params: vec4<f32>) -> OgcPairLin {
  let h = ogcSolidHeader();
  let s = ogcSolidSample(h, xv);
  let xt = ogcXt(v);
  var nl = vec3<f32>(0.0);
  if (bitcast<f32>(ent.z) > 0.0) { nl = ogcUnitOr(ogcSolidSample(h, xt).grad, ogcUnitOr(s.grad, vec3<f32>(0.0))); }
  return ogcFieldEval(v, s.phi, s.grad, xv - xt, nl, ent, params);
}

// Vertex v against the moving solid at its end pose: phi_body(R1^T (x - c1)) with
// gradient R1 grad phi_body, and the slip measured against the solid's material point
// that sat at x_t when the step began, carried rigidly to R1 R0^T (x_t - c0) + c1, whose
// field normal (turned to the end pose) is the lagged one.
fn ogcBodyLin(v: u32, xv: vec3<f32>, ent: vec4<u32>, params: vec4<f32>) -> OgcPairLin {
  let p1 = ogcPose(1u);
  let h = ogcSolidHeaderAt(ogcBodyBase());
  let s = ogcSolidSample(h, ogcToBody(p1, xv));
  let yt = ogcToBody(ogcPose(0u), ogcXt(v));
  let carried = ogcToWorld(p1, yt);
  var nl = vec3<f32>(0.0);
  if (bitcast<f32>(ent.z) > 0.0) { nl = p1.R * ogcUnitOr(ogcSolidSample(h, yt).grad, ogcUnitOr(s.grad, vec3<f32>(0.0))); }
  return ogcFieldEval(v, s.phi, p1.R * s.grad, xv - carried, nl, ent, params);
}

// The weights of point p against the feature of triangle (a, b, c): (1, -barycentrics).
fn ogcVtWeights(feature: u32, p: vec3<f32>, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> vec4<f32> {
  var w = vec4<f32>(1.0, 0.0, 0.0, 0.0);
  var X = array<vec3<f32>, 3>(a, b, c);
  if (feature < 3u) {
    w[1u + feature] = -1.0;
  } else if (feature < OGC_FACE) {
    let i = feature - 3u; let j = (i + 1u) % 3u;
    let dir = X[j] - X[i];
    let s = dot(p - X[i], dir) / dot(dir, dir);
    w[1u + i] = -(1.0 - s); w[1u + j] = -s;
  } else {
    let v0 = b - a; let v1 = c - a; let v2 = p - a;
    let d00 = dot(v0, v0); let d01 = dot(v0, v1); let d11 = dot(v1, v1);
    let d20 = dot(v2, v0); let d21 = dot(v2, v1);
    let den = d00 * d11 - d01 * d01;
    let bv = (d11 * d20 - d01 * d21) / den; let bw = (d00 * d21 - d01 * d20) / den;
    w.y = -(1.0 - bv - bw); w.z = -bv; w.w = -bw;
  }
  return w;
}

fn ogcEntryLin(v: u32, xv: vec3<f32>, ent: vec4<u32>, params: vec4<f32>) -> OgcPairLin {
  if (ent.y == OGC_SOLID_Y) { return ogcSolidLin(v, xv, ent, params); }
  if (ent.y == OGC_BODY_Y) { return ogcBodyLin(v, xv, ent, params); }
  var ids: vec4<u32>;
  var w: vec4<f32>;
  var wl: vec4<f32>;
  var X: array<vec3<f32>, 4>;
  var Xl: array<vec3<f32>, 4>;
  var cloth = vec4<f32>(1.0);
  let sampled = ent.w == 1u && (ent.y & OGC_EE_BIT) == 0u;
  if ((ent.y & OGC_EE_BIT) == 0u) {
    let feature = (ent.y >> 28u) & 7u;
    ids = vec4<u32>(ent.x, ogcTri(ent.y & OGC_ID_MASK));
    for (var i = 1u; i < 4u; i++) { X[i] = select(ogcX(ids[i]), xv, ids[i] == v); Xl[i] = ogcXt(ids[i]); }
    if (sampled) { X[0] = ogcSamplePos(ent.x); Xl[0] = ogcSampleStart(ent.x); cloth.x = 0.0; }
    else { X[0] = select(ogcX(ids[0]), xv, ids[0] == v); Xl[0] = ogcXt(ids[0]); }
    w = ogcVtWeights(feature, X[0], X[1], X[2], X[3]);
    wl = ogcVtWeights(feature, Xl[0], Xl[1], Xl[2], Xl[3]);
  } else {
    let e0 = ogcEdgeVerts(ent.x); let e1 = ogcEdgeVerts(ent.y & OGC_ID_MASK);
    ids = vec4<u32>(e0, e1);
    for (var i = 0u; i < 4u; i++) { X[i] = select(ogcX(ids[i]), xv, ids[i] == v); Xl[i] = ogcXt(ids[i]); }
    let st = unpack2x16unorm(ent.w);
    w = vec4<f32>(1.0 - st.x, st.x, -(1.0 - st.y), -st.y);
    wl = w;
  }
  var gap = vec3<f32>(0.0);
  var wr = 0.0;
  var wrl = 0.0;
  for (var i = 0u; i < 4u; i++) {
    gap += w[i] * X[i];
    wr += select(0.0, w[i], ids[i] == v);
    wrl += select(0.0, wl[i], ids[i] == v);
  }
  let d = length(gap);
  // Coincident primitives (d = 0) lie outside the barrier's domain, like phi <= 0 above: infinite
  // energy and no direction to push along. Detection fails a coincident pair explicitly (OGC_STATUS.coincident).
  // The infinity is built from a runtime word (ent.w & 0 is 0), as in ogcFieldEval: a constant one would not fold.
  if (!(d > 0.0)) { return OgcPairLin(ids, vec4<f32>(0.0), vec4<f32>(0.0), 0.0, 0.0, vec3<f32>(0.0), OGC_Z3, OGC_Z3, bitcast<f32>(0x7f800000u | (ent.w & 0u))); }
  let n = gap / d;
  let act = ogcActivation(d, params.x, params.y);
  var out = OgcPairLin(ids, cloth * w, cloth * wl, wr, wrl, -act.y * wr * n, act.z * mat3x3<f32>(n * n.x, n * n.y, n * n.z), OGC_Z3, act.x);
  // A sample-facet entry carries mu lambda with the solid's own mu; its sample's own
  // motion over the step is the solid's (none for a static one).
  let ml = select(params.z * bitcast<f32>(ent.z), bitcast<f32>(ent.z), sampled);
  if (ml > 0.0) {
    var dx = vec3<f32>(0.0);
    var gl = vec3<f32>(0.0);
    for (var i = 0u; i < 4u; i++) { dx += wl[i] * (X[i] - Xl[i]); gl += wl[i] * Xl[i]; }
    ogcFriction(&out, ml, dx, ogcUnitOr(gl, n), params.w);
  }
  return out;
}

// One contact entry evaluated for vertex v placed at xv: its force, its diagonal Hessian block
// wr^2 Mn + wrl^2 Mf, its energy.
fn ogcEvalEntry(v: u32, xv: vec3<f32>, ent: vec4<u32>, params: vec4<f32>) -> OgcPairEval {
  let l = ogcEntryLin(v, xv, ent, params);
  return OgcPairEval(l.f, (l.wr * l.wr) * l.Mn + (l.wrl * l.wrl) * l.Mf, l.e);
}

fn ogcContactCount(v: u32) -> u32 {
  let header = ogcContacts[0];
  return min(ogcContacts[header.z + (v >> 2u)][v & 3u], header.y);
}

fn ogcVertexContactLanes(v: u32, xv: vec3<f32>, first: u32, stride: u32, f: ptr<function, vec3<f32>>, H: ptr<function, mat3x3<f32>>) -> f32 {
  let header = ogcContacts[0];
  let params = bitcast<vec4<f32>>(ogcContacts[1]);
  let count = min(ogcContacts[header.z + (v >> 2u)][v & 3u], header.y);
  let base = header.w + v * header.y;
  var e = 0.0;
  for (var i = first; i < count; i += stride) {
    let r = ogcEvalEntry(v, xv, ogcContacts[base + i], params);
    *f += r.f; *H += r.h; e += r.e;
  }
  return e;
}

fn ogcVertexContactEnergyLanes(v: u32, xv: vec3<f32>, first: u32, stride: u32) -> f32 {
  let header = ogcContacts[0];
  let params = bitcast<vec4<f32>>(ogcContacts[1]);
  let count = min(ogcContacts[header.z + (v >> 2u)][v & 3u], header.y);
  let base = header.w + v * header.y;
  var e = 0.0;
  for (var i = first; i < count; i += stride) { e += ogcEvalEntry(v, xv, ogcContacts[base + i], params).e; }
  return e;
}

fn ogcAccumulateVertexContact(v: u32, xv: vec3<f32>, f: ptr<function, vec3<f32>>, H: ptr<function, mat3x3<f32>>) {
  _ = ogcVertexContactLanes(v, xv, 0u, 1u, f, H);
}

fn ogcVertexContactEnergy(v: u32, xv: vec3<f32>) -> f32 { return ogcVertexContactEnergyLanes(v, xv, 0u, 1u); }
${options.newton ? OGC_NEWTON_BLOCK_WGSL : ""}`;
}

/**
 * The global (Newton) forms over the same per-vertex entry lists. Every pair sits in the
 * list of each vertex it weighs on (VT: the vertex and its closest feature's vertices; EE:
 * both edges' vertices, one copy per edge; a field entry: its vertex; sample-face: the
 * face's feature vertices), so each is a gather: no scatter, no atomics.
 *   ogcVertexContactHvLanes(v, first, stride) -> vec3   (H p)_v = sum w_v Mn (sum_j w_j p_j)
 *       + wl_v Mf (sum_j wl_j p_j),
 *       with p = ogcDir (host accessor; a solid's sample reads none), linearized at ogcX
 *   ogcContactEnergyOwnedLanes(v, first, stride) -> f32 the energy of the entries v owns, at
 *       ogcX: summed over all vertices, every pair once
 */
const OGC_NEWTON_BLOCK_WGSL = /* wgsl */`
// The one holder of each pair that counts its energy: a field entry's vertex, a VT pair's
// vertex, a sample-face pair's least feature vertex, an EE pair's lower edge's first vertex.
fn ogcEntryOwned(v: u32, ent: vec4<u32>) -> bool {
  if (ent.y == OGC_SOLID_Y || ent.y == OGC_BODY_Y) { return true; }
  if ((ent.y & OGC_EE_BIT) != 0u) { return ent.x < (ent.y & OGC_ID_MASK) && v == ogcEdgeVerts(ent.x).x; }
  if (ent.w != 1u) { return v == ent.x; }
  let tri = ogcTri(ent.y & OGC_ID_MASK);
  let feature = (ent.y >> 28u) & 7u;
  if (feature < 3u) { return v == tri[feature]; }
  if (feature < OGC_FACE) { let i = feature - 3u; return v == min(tri[i], tri[(i + 1u) % 3u]); }
  return v == min(min(tri.x, tri.y), tri.z);
}

fn ogcVertexContactHvLanes(v: u32, first: u32, stride: u32) -> vec3<f32> {
  let header = ogcContacts[0];
  let params = bitcast<vec4<f32>>(ogcContacts[1]);
  let count = min(ogcContacts[header.z + (v >> 2u)][v & 3u], header.y);
  let base = header.w + v * header.y;
  let xv = ogcX(v);
  var out = vec3<f32>(0.0);
  for (var i = first; i < count; i += stride) {
    let l = ogcEntryLin(v, xv, ogcContacts[base + i], params);
    var q = vec3<f32>(0.0); var ql = vec3<f32>(0.0);
    for (var k = 0u; k < 4u; k++) {
      if (l.wc[k] != 0.0 || l.wlc[k] != 0.0) { let p = ogcDir(l.ids[k]); q += l.wc[k] * p; ql += l.wlc[k] * p; }
    }
    out += l.wr * (l.Mn * q) + l.wrl * (l.Mf * ql);
  }
  return out;
}

fn ogcContactEnergyOwnedLanes(v: u32, first: u32, stride: u32) -> f32 {
  let header = ogcContacts[0];
  let params = bitcast<vec4<f32>>(ogcContacts[1]);
  let count = min(ogcContacts[header.z + (v >> 2u)][v & 3u], header.y);
  let base = header.w + v * header.y;
  let xv = ogcX(v);
  var e = 0.0;
  for (var i = first; i < count; i += stride) {
    let ent = ogcContacts[base + i];
    if (ogcEntryOwned(v, ent)) { e += ogcEvalEntry(v, xv, ent, params).e; }
  }
  return e;
}
`;
