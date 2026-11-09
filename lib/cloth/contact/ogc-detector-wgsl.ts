/**
 * Detection kernels of the OGC detector. Every kernel declares only the
 * bindings it uses, so each pipeline's auto layout is exactly its binding set.
 *
 * Bindings (group 0, one numbering for every kernel):
 *   0 params (uniform)        1 topology (read)          2 solver positions
 *   3 solver velocities       4 detection state x_det|b  5 work: dminV|dminT|dminE|envelope (atomic)
 *   6 grid counts (atomic)    7 grid: scan levels|entries 8 contacts (atomic)
 *   9 counters (atomic)       10 indirect args            11 DAT division pairs
 *   12 planes indirect args (apart from 10: the bound pass is itself dispatched from 10)
 *   13 the step start x_t     14 predictions w | predicted step end and residual estimate
 * work also holds, per vertex at solidOff and solidDevOff, the static solid's truncation
 * terms (q.xyz, marker) and (D.xyz, 0) of the round (OgcTerms), and at heldOff the held flags (bit 0: contact, truncation or conflict
 * this step; bit 1: held this step; bit 2: carried with the moving solid this step, in an impact
 * zone; bit 3: held by a conflict this step; bit 4: touched the moving solid this step (a field
 * bound whose ball reaches r, a cut, or a sample entry); bit 5: held with the moving solid this step (last step it was in
 * contact and touched or moved with the solid); bit 6: moved with the solid this step (w = a_v,
 * set by prep)); the solid image sits in the contact buffer at word solidBase.
 *
 * One detection round is nine indirect dispatches:
 *   prep (N)  cell (1)  insert-count (T)  scan x3  insert (T)  query (N + E + S)  bound (N)
 * with S the static solid's surface samples (solid-field.ts clothSolidSamples), held in
 * x_det after the vertices (index N + s: position, and the solid's mu in w).
 * then the redo decision, and the same nine again (zero workgroups unless the round
 * had a conflict, see OGC_REDO_WGSL). The insert pass consumes the counts back to
 * zero, so no pass ever clears the grid.
 *
 * Truncation (Chen, Hsu, Ayman, Macklin 2026, "Divide and Truncate", Alg. 2) is two
 * dispatches: planes (one thread per division pair, indirect; none in isotropic mode)
 * and apply (N). The positions buffer holds X + dX with X = x_det, so dX accumulates
 * across iterations and the next detection round's prep commits it (Alg. 3, lines 9-12).
 *
 * DAT division pairs (two vec4<u32> each, written at detection):
 *   ids    vertex-triangle (v, a, b, c); edge-edge (p0 | 1 << 31, p1, q0, q1); a hole is x = OGC_NONE
 *   plane  (n.xyz, certified support gap) as f32 bits; n points from the second primitive
 *          to the first; a gap <= 0 marks a pair with no separating plane
 */
import { OGC_COMMON_WGSL, OGC_COUNTERS_WGSL, OGC_POSE_WORD, OGC_POSE_WORDS, OGC_SOLID_WGSL } from "./ogc-wgsl";

export const OGC_PARAM_WORDS = 72;
/**
 * Times a round with a moving solid may be redone (OGC_REDO_WGSL): OGC_REDO_MANDATORY for the
 * step's mandatory round, OGC_REDO_ROUNDS for each later one. Each redo keeps every
 * prediction but its conflicts': a conflict between free cloth holds its vertices, and one
 * with the solid (its samples or hull edges, or cloth moving with it) carries its cloth with
 * the solid, so impact zones moving with the solid grow by a conflict's width a redo. The
 * marks only grow (free, held, carried), so the redos close the zones: a redo whose conflicts
 * mark nothing new would repeat itself and fails at once as heldConflict, as does a conflict
 * in the last. Only the mandatory round, at the start pose, sees the solid's step (a_v); a
 * later round's carry is a hold. Without a moving solid a round is redone once, every
 * prediction held (Planar-DAT, which never conflicts).
 */
export const OGC_REDO_ROUNDS = 4;
export const OGC_REDO_MANDATORY = 32;
/** Indirect-args slots, three u32 each, in dispatch order. */
export const OGC_DETECT_DISPATCHES = ["prep", "cell", "count", "scan1", "scan2", "scan3", "insert", "query", "bound"] as const;
export const OGC_ARGS_WORDS = 3 * OGC_DETECT_DISPATCHES.length;
/** Pair slots a query thread reserves at a time; unused ones become holes. */
const PAIR_BLOCK = 4;

const PARAMS_STRUCT = /* wgsl */`
struct OgcParams {
  N: u32, T: u32, E: u32, H: u32,
  gridCap: u32, contactCap: u32, roundCap: u32, exceedThreshold: u32,
  r: f32, rqBase: f32, rqMax: f32, envScale: f32,
  gammaP: f32, cellFloor: f32, dt: f32, intersectCell: f32,
  gravity: vec4<f32>,
  trisOff: u32, edgesOff: u32, triEdgesOff: u32, vtOffOff: u32,
  vtListOff: u32, veOffOff: u32, veListOff: u32, B1: u32,
  local1Off: u32, sums1Off: u32, local2Off: u32, sums2Off: u32,
  local3Off: u32, sums3Off: u32, entriesOff: u32, B2: u32,
  countsOff: u32, entriesBase: u32, dminTOff: u32, dminEOff: u32,
  envOff: u32, wgN: u32, wgT: u32, wgNE: u32,
  wgS1: u32, wgS2: u32, kc: f32, mode: u32,
  gammaR: f32, pairCap: u32, tvOff: u32, sepEps: f32,
  solidBase: u32, solidOff: u32, heldOff: u32, predict: u32,
  cellRule: u32, S: u32, bodyBase: u32, bodyOff: u32,
  SB: u32, wgPrep: u32, bodyReach: f32, pad2: u32,
  edgeBase: u32, SE: u32, bodyEdgeBase: u32, SEB: u32,
  solidDevOff: u32, bodyDevOff: u32, pad3: u32, pad4: u32,
}
`;

/** Parameter word offsets, matching OgcParams. */
export const OGC_PARAM = {
  N: 0, T: 1, E: 2, H: 3, gridCap: 4, contactCap: 5, roundCap: 6, exceedThreshold: 7,
  r: 8, rqBase: 9, rqMax: 10, envScale: 11, gammaP: 12, cellFloor: 13, dt: 14, intersectCell: 15,
  gravity: 16, trisOff: 20, edgesOff: 21, triEdgesOff: 22, vtOffOff: 23, vtListOff: 24, veOffOff: 25, veListOff: 26, B1: 27,
  local1Off: 28, sums1Off: 29, local2Off: 30, sums2Off: 31, local3Off: 32, sums3Off: 33, entriesOff: 34, B2: 35,
  countsOff: 36, entriesBase: 37, dminTOff: 38, dminEOff: 39, envOff: 40, wgN: 41, wgT: 42, wgNE: 43, wgS1: 44, wgS2: 45, kc: 46,
  mode: 47, gammaR: 48, pairCap: 49, tvOff: 50, sepEps: 51, solidBase: 52, solidOff: 53, heldOff: 54, predict: 55, cellRule: 56, S: 57,
  bodyBase: 58, bodyOff: 59, SB: 60, wgPrep: 61, bodyReach: 62,
  edgeBase: 64, SE: 65, bodyEdgeBase: 66, SEB: 67, solidDevOff: 68, bodyDevOff: 69,
} as const;

const DECL: Record<number, string> = {
  0: "@group(0) @binding(0) var<uniform> P: OgcParams;",
  1: "@group(0) @binding(1) var<storage, read> topo: array<u32>;",
  2: "@group(0) @binding(2) var<storage, read> pos: array<vec4<f32>>;",
  3: "@group(0) @binding(3) var<storage, read> vel: array<vec4<f32>>;",
  4: "@group(0) @binding(4) var<storage, read_write> xdet: array<vec4<f32>>;",
  5: "@group(0) @binding(5) var<storage, read_write> work: array<atomic<u32>>;",
  6: "@group(0) @binding(6) var<storage, read_write> counts: array<atomic<u32>>;",
  7: "@group(0) @binding(7) var<storage, read_write> grid: array<u32>;",
  8: "@group(0) @binding(8) var<storage, read_write> contacts: array<atomic<u32>>;",
  9: "@group(0) @binding(9) var<storage, read_write> counters: array<atomic<u32>>;",
  10: "@group(0) @binding(10) var<storage, read_write> args: array<u32>;",
  11: "@group(0) @binding(11) var<storage, read_write> pairs: array<vec4<u32>>;",
  12: "@group(0) @binding(12) var<storage, read_write> planesArgs: array<u32>;",
  13: "@group(0) @binding(13) var<storage, read> xt: array<vec4<f32>>;",
  14: "@group(0) @binding(14) var<storage, read> pred: array<vec4<f32>>;",
  /** The line-search filter's direction d (the solver's, vec4 per vertex) and its output (N + 1 words, f32 bits: alpha_v, then alpha_min). */
  15: "@group(0) @binding(15) var<storage, read> dir: array<vec4<f32>>;",
  16: "@group(0) @binding(16) var<storage, read_write> alpha: array<atomic<u32>>;",
  /** The solver's inertial target y (vec4 per vertex, prediction "target"; positions otherwise). */
  17: "@group(0) @binding(17) var<storage, read> tgt: array<vec4<f32>>;",
};
/** Truncation writes the solver's positions; prep writes the predictions. */
const DECL_POS_RW = "@group(0) @binding(2) var<storage, read_write> pos: array<vec4<f32>>;";
const DECL_PRED_RW = "@group(0) @binding(14) var<storage, read_write> pred: array<vec4<f32>>;";

/** The kernel's declarations; `rw` names the bindings among 2 and 14 it writes. */
function header(bindings: readonly number[], rw: readonly number[] = []): string {
  return [PARAMS_STRUCT, OGC_COUNTERS_WGSL, OGC_COMMON_WGSL,
    ...bindings.map(b => rw.includes(b) ? (b === 2 ? DECL_POS_RW : DECL_PRED_RW) : DECL[b])].join("\n");
}

/**
 * Predicted-motion DAT (option c). pred[v].xyz is vertex v's prediction w_v for
 * the round: its remaining predicted displacement, or 0 when held (w: the
 * step's whole predicted length, for the held envelope); pred[N + v]
 * holds the step's predicted end x_t + w (xyz) and the residual estimate (w,
 * negative until the first step has measured it). A vertex moves X + w + delta
 * with |delta| <= x_det.w, and every cut scales delta: the division planes keep
 * the normal found at X and translate from X to X + w, so a vertex on its side
 * at both ends of its straight path is on it throughout.
 */
const PRED_FNS = /* wgsl */`
// Indices N .. N + S - 1 are the static solid's surface samples (x_det only): no prediction.
// N + S .. N + S + SB - 1 are the moving solid's, predicted exactly by its poses (pred[N + i - S]).
fn predOf(i: u32) -> vec3<f32> {
  if (i < P.N) { return pred[i].xyz; }
  if (i < P.N + P.S) { return vec3<f32>(0.0); }
  return pred[P.N + i - P.S].xyz;
}
fn anchored(i: u32) -> vec3<f32> { return xdet[i].xyz + predOf(i); }
`;

/**
 * Pairs fixed in the moving solid's frame (query and planes). In the mandatory round,
 * whose detection is at the start pose, a pair with one of the solid's samples (or hull
 * edges), or whose cloth all moves with it (bit 6, w = a_v, prep), keeps its plane fixed in
 * the solid's frame: certified at X and turned to the end pose by R1 R0^T, where the conflict
 * test and the planes pass meet it. Every other pair keeps its plane in the world.
 * Each vertex's round runs straight in one frame: the solid's when it moves with the solid,
 * the world's otherwise. In a pair of the other frame its path strays from straight by up to
 * sigma_v (prep), which its own side keeps from the plane on top of sepEps; on the
 * conflict test's and the predicted axis's gaps the pair needs both sides' strays. A vertex
 * moving with the solid against its samples, and free cloth against free cloth, keep none.
 * After the mandatory round both poses are the end pose and nothing turns.
 */
const REDO_CAP_FN = /* wgsl */`
fn redoCap() -> u32 { return select(${OGC_REDO_ROUNDS}u, ${OGC_REDO_MANDATORY}u, atomicLoad(&counters[OGC_ROUNDS]) == 0u); }
`;
const BODY_PAIR_FNS = REDO_CAP_FN + /* wgsl */`
fn withSolid(i: u32) -> bool { return (atomicLoad(&work[P.heldOff + i]) & 64u) != 0u; }
fn sigmaOf(i: u32) -> f32 { return bitcast<f32>(atomicLoad(&work[P.bodyOff + 8u * i + 7u])); }
fn turnedPair(ids: vec4<u32>) -> bool {
  if (P.bodyBase == 0u || atomicLoad(&counters[OGC_ROUNDS]) != 0u) { return false; }
  var body = false;
  var carried = true;
  for (var k = 0u; k < 4u; k++) {
    let i = ids[k];
    if (i >= P.N + P.S && i < P.N + P.S + P.SB) { body = true; }
    else if (i >= P.N) { carried = false; }
    else if (!withSolid(i)) { carried = false; }
  }
  return body || carried;
}
// Vertex i's stray in a pair of the solid's frame (turned) or the world's.
fn strayOf(i: u32, turned: bool) -> f32 {
  if (P.bodyBase == 0u || i >= P.N || withSolid(i) == turned) { return 0.0; }
  return sigmaOf(i);
}
// The strays of the pair's two sides (its first nA ids, the rest).
fn pairStray(ids: vec4<u32>, nA: u32) -> vec2<f32> {
  let turned = turnedPair(ids);
  var st = vec2<f32>(0.0);
  for (var k = 0u; k < 4u; k++) {
    let sk = strayOf(ids[k], turned);
    if (k < nA) { st.x = max(st.x, sk); } else { st.y = max(st.y, sk); }
  }
  return st;
}
`;

const TOPO_FNS = /* wgsl */`
fn triVerts(t: u32) -> vec3<u32> { let b = P.trisOff + 4u * t; return vec3<u32>(topo[b], topo[b + 1u], topo[b + 2u]); }
fn edgeRec(e: u32) -> vec4<u32> { let b = P.edgesOff + 4u * e; return vec4<u32>(topo[b], topo[b + 1u], topo[b + 2u], topo[b + 3u]); }
fn hasVertex(t: vec3<u32>, v: u32) -> bool { return t.x == v || t.y == v || t.z == v; }
`;

const CELL_FNS = /* wgsl */`
fn cellOf(p: vec3<f32>, inv: f32) -> vec3<i32> { return vec3<i32>(floor(p * inv)); }
fn cellKey(c: vec3<i32>) -> u32 {
  let u = bitcast<vec3<u32>>(c);
  return (u.x & 1023u) | ((u.y & 2047u) << 10u) | ((u.z & 2047u) << 21u);
}
fn cellHash(c: vec3<i32>) -> u32 {
  let u = bitcast<vec3<u32>>(c);
  return ((u.x * 73856093u) ^ (u.y * 19349663u) ^ (u.z * 83492791u)) & (P.H - 1u);
}
struct OgcBox { lo: vec3<f32>, hi: vec3<f32> }
fn boxesOverlap(a: OgcBox, b: OgcBox) -> bool { return all(a.lo <= b.hi) && all(b.lo <= a.hi); }
`;

// A grid entry is 8 words: the face, its cell key and its round box (lo, hi), so a
// query reads each entry's box in one coherent burst instead of re-deriving it.
const GRID_READ_FNS = /* wgsl */`
const OGC_ENTRY_WORDS = 8u;
fn entryFace(j: u32) -> u32 { return grid[P.entriesOff + OGC_ENTRY_WORDS * j]; }
fn entryKey(j: u32) -> u32 { return grid[P.entriesOff + OGC_ENTRY_WORDS * j + 1u]; }
fn entryBox(j: u32) -> OgcBox {
  let b = P.entriesOff + OGC_ENTRY_WORDS * j;
  return OgcBox(bitcast<vec3<f32>>(vec3<u32>(grid[b + 2u], grid[b + 3u], grid[b + 4u])),
                bitcast<vec3<f32>>(vec3<u32>(grid[b + 5u], grid[b + 6u], grid[b + 7u])));
}
fn bucketStart(h: u32) -> u32 {
  if (h >= P.H) { return grid[P.sums3Off]; }
  return grid[P.local1Off + h] + grid[P.local2Off + (h >> 10u)] + grid[P.local3Off + (h >> 20u)];
}
`;

/** Envelope of vertex i (detection) and the face box inflated by its vertices' widest envelope. */
/** The face box helpers; `envelopes` adds the per-vertex envelope reads (the work buffer, binding 5). */
function triBoxFns(intersect: boolean, envelopes = !intersect): string {
  return (intersect ? "" : PRED_FNS) + (envelopes ? /* wgsl */`
fn envOf(i: u32) -> f32 { return bitcast<f32>(atomicLoad(&work[P.envOff + i])); }
fn triEnv(v: vec3<u32>) -> f32 { return max(envOf(v.x), max(envOf(v.y), envOf(v.z))); }
` : "") + /* wgsl */`
// The round's largest envelope: a face's grid box is grown by it, so any primitive's
// unenlarged swept box finds every face within the larger of the two envelopes.
fn roundEnv() -> f32 { ${intersect ? "return 0.0;" : "return bitcast<f32>(atomicLoad(&counters[OGC_ROUND_ENVELOPE]));"} }
// The face's box over the round: from X to the predicted X + w, grown by the round's largest envelope.
fn triBox(t: u32) -> OgcBox {
  let v = triVerts(t);
  let a = xdet[v.x].xyz; let b = xdet[v.y].xyz; let c = xdet[v.z].xyz;
  var lo = min(a, min(b, c)); var hi = max(a, max(b, c));
  ${intersect ? "" : `let a1 = a + predOf(v.x); let b1 = b + predOf(v.y); let c1 = c + predOf(v.z);
  lo = min(lo, min(a1, min(b1, c1))); hi = max(hi, max(a1, max(b1, c1)));`}
  let e = roundEnv();
  return OgcBox(lo - vec3<f32>(e), hi + vec3<f32>(e));
}
fn invCell() -> f32 { ${intersect ? "return 1.0 / P.intersectCell;" : "return bitcast<f32>(atomicLoad(&counters[OGC_INV_CELL]));"} }
`;
}

// ------------------------------------------------------------------ prep, cell, frame and step control

const SOLID_ACCESS = /* wgsl */`
fn ogcSolidBase() -> u32 { return P.solidBase; }
fn ogcSolidWord(i: u32) -> u32 { return atomicLoad(&contacts[i]); }
`;

/**
 * Each vertex's x_det, prediction w, envelope and, with a static solid, its solid
 * terms and entry. The step's mandatory round detects at the step start x_t (the
 * solver has already placed its initial guess, which the round's truncation then
 * cuts); a re-detection round commits the current, truncated iterate.
 *
 * A free vertex is predicted to move its inertial v dt + g dt^2, the solver's initial guess
 * less x_t, or the solver's inertial target y less x_t (prediction "target": exact in free
 * flight, whatever the guess), the rest of it in a re-detection round, and its envelope
 * covers only the residual motion:
 * r + s rho, with rho the last step's deviation from its own prediction. A held
 * vertex (in contact, truncated or in a conflict in this step or the last, or
 * every vertex of a redone round, of isotropic mode, or without predictions) is
 * predicted 0 and its envelope covers the inertial displacement, r + s |v dt + g dt^2|:
 * Planar-DAT itself. A prediction into the static solid stops where the solid's
 * affine bound does, so every cut of the round scales only the residual.
 *
 * The moving solid (one-way: its poses are input) is a static solid in its own frame: its
 * field, samples and hull edges meet the cloth mapped into that frame at both ends of the
 * round. A vertex's detection pose is the step's start pose in the mandatory round and
 * its end pose after (the solver's iterates already face the end pose), and
 * a_v = R1 R_det^T (x_det - c_det) + c1 - x_det is the displacement that keeps it fixed in
 * the solid's frame: there "carried" is "held", which the solid always allows. The field's
 * ball bound is taken around R_det^T (x_det - c_det) on the motion relative to the solid,
 * w - a_v, and a prediction it forbids moves towards a_v only as far as the bound needs,
 * as the static solid scales one towards 0 (a round redone carries it outright). A vertex
 * a conflict marked (bit 2) is carried in its step's rounds redone: an impact zone moving
 * with the solid. A prediction the static solid then has to cut out of the moving one's
 * bound fails as OGC_ST_BODY_SQUEEZE.
 *
 * A vertex's round runs straight in one frame: the solid's when it moves with the solid
 * (bit 6), the world's otherwise. Seen from the other frame the path is a curve. With the
 * solid turning at the rate theta per step about a fixed axis and its centre moving
 * straight, a world-straight path in the solid's frame is y(t) = R(t)^T (x(t) - c(t)), whose
 * second derivative is at most theta^2 |x - c| + 2 theta |d - dc| (d the vertex's motion, dc
 * the centre's), so it leaves its chord by at most sigma_v = (theta^2 (|x_det - c0| + Q)
 * + 2 theta Q) / 8 with Q = |w - dc| + env / 2 >= |d - dc|; a body-straight path in the world,
 * c(t) + R(t) l(t), alike with Q = |w - a_v| + |w| + env / 2 >= |l'|. sigma_v (work,
 * bodyOff + 8 v + 7; 0 after the mandatory round, whose poses agree) is the margin of the
 * other frame's bound (the moving solid's field for a world vertex, the static solid's for
 * one moving with the solid) and of its side of a plane in the other frame (BODY_PAIR_FNS).
 *
 * Threads N .. N + SB - 1 pose the solid's surface samples: x_det at the detection pose
 * and the exact prediction to the end pose.
 */
export const OGC_PREP_WGSL = header([0, 2, 3, 4, 5, 8, 9, 13, 14, 17], [14]) + SOLID_ACCESS + OGC_SOLID_WGSL + /* wgsl */`
var<workgroup> wgMaxEnv: atomic<u32>;
var<workgroup> wgCube: array<f32, 64>;
var<workgroup> wgCap: atomic<u32>;
var<workgroup> wgBad: atomic<u32>;
var<workgroup> wgHeld: atomic<u32>;
var<workgroup> wgSolid: atomic<u32>;
var<workgroup> wgBody: atomic<u32>;
var<workgroup> wgSqueeze: atomic<u32>;
var<workgroup> wgZone: atomic<u32>;
// The envelope r + s rho, clamped to [rqBase, rqMax] (a clamp at rqMax is counted).
fn envelopeOf(rho: f32) -> f32 {
  let env = max(P.rqBase, P.r + P.envScale * rho);
  if (env > P.rqMax) { atomicAdd(&wgCap, 1u); return P.rqMax; }
  return env;
}
fn putWork4(i: u32, q: vec4<f32>) {
  atomicStore(&work[i], bitcast<u32>(q.x)); atomicStore(&work[i + 1u], bitcast<u32>(q.y));
  atomicStore(&work[i + 2u], bitcast<u32>(q.z)); atomicStore(&work[i + 3u], bitcast<u32>(q.w));
}
fn pushSolidEntry(v: u32, y: u32, lambda: f32) {
  let slot = atomicAdd(&contacts[P.countsOff + v], 1u);
  if (slot < P.contactCap) {
    let b = P.entriesBase + 4u * (v * P.contactCap + slot);
    atomicStore(&contacts[b], v); atomicStore(&contacts[b + 1u], y);
    atomicStore(&contacts[b + 2u], bitcast<u32>(lambda)); atomicStore(&contacts[b + 3u], 0u);
  } else {
    atomicOr(&counters[OGC_STATUS_WORD], OGC_ST_CONTACT_OVERFLOW);
  }
}
// The truncation terms of a ball bound, which certify phi(x0 + d) >= min((1 - gamma_r) phi0, r)
// (a vertex may come to the contact distance r, where the barrier holds it, or by gamma_r of
// its clearance once nearer than r / (1 - gamma_r)) for every d in either of two convex sets:
//   affine     f(d) = D.|d| - q.d <= 1 and |d| <= rc, with (q, D) = (g0, D) / max(gamma_r phi0, phi0 - r)
//              from the field over the ball |d| <= rc (q.w = rc; 0: no bound), tight for a short
//              motion along the surface;
//   Lipschitz  |d| <= rl = max(gamma_r phi0, phi0 - r) / L_max, the field's own slope bound,
//              tight for a long motion (the affine bound's D grows with its ball).
// A motion is certified when either holds. The "inside" marker (q.w = 3e38, rl = 0) at
// phi0 <= 0 allows none. Also the entry with mu lambda (the solid's own mu at x0).
struct OgcTerms { q: vec4<f32>, D: vec3<f32>, rl: f32 }
fn boundTerms(v: u32, h: OgcSolidHeader, sb: OgcSolidBound, x0: vec3<f32>, y: u32, rc: f32, margin: f32) -> OgcTerms {
  if (!(sb.phi0 > 0.0)) {
    atomicOr(&counters[OGC_STATUS_WORD], OGC_ST_SOLID_PENETRATION);
    return OgcTerms(vec4<f32>(0.0, 0.0, 0.0, 3.0e38), vec3<f32>(0.0), 0.0);
  }
  // The margin: how far the vertex's true path may leave the straight one the bound is
  // evaluated on, so the field there is at most L_max margin lower (sigma_v, prep).
  // The cut keeps phi >= min((1 - gamma_r) phi0, r), and never below sepEps, the f32 resolution
  // the cloth's pairs keep too: repeated cuts would otherwise close the gap geometrically into
  // rounding. A vertex rounding has left within sepEps may not approach at all (room 0: only
  // D.|d| <= g0.d, motion away), so its barrier moves it out.
  var room = 0.0;
  if (sb.phi0 > P.sepEps) { room = min(max(P.gammaR * sb.phi0, sb.phi0 - P.r), sb.phi0 - P.sepEps); }
  room -= h.lmax * margin;
  if (!(room >= 0.0)) { return OgcTerms(vec4<f32>(0.0, 0.0, 0.0, 3.0e38), vec3<f32>(0.0), 0.0); }
  let s = 1.0 / max(room, 1.0e-30);
  let lambda = select(0.0, -ogcActivation(sb.phi0, P.r, P.kc).y * length(sb.g0), sb.phi0 < P.r) * ogcSolidMu(h, x0);
  pushSolidEntry(v, y, lambda);
  return OgcTerms(vec4<f32>(sb.g0 * s, rc), sb.D * s, room / h.lmax);
}
// A solid's terms for a round that moves the vertex at most rho from x0 (in the field's frame).
// A round the field's slope clears by r needs none, nor one the clearance table clears by r
// (a long motion far from the solid, where the bricks only know the field exceeds band). Otherwise the affine bound is taken over the
// ball rc = min(rho, rqMax / 2), within which an unstored brick still reads clear of r (the band
// holds r + 1.5 L_max rqMax); a ball clear of the solid certifies all of it (q = D = 0). Beyond
// rc only the Lipschitz ball certifies, so a long prediction is never moved for the ball's sake.
fn roundBound(v: u32, h: OgcSolidHeader, x0: vec3<f32>, rho: f32, y: u32, margin: f32) -> OgcTerms {
  if (min(ogcSolidSample(h, x0).phi, h.band) - h.lmax * (rho + margin) >= P.r
    || ogcSolidClearance(h, x0) - (rho + margin) >= P.r) { return OgcTerms(vec4<f32>(0.0), vec3<f32>(0.0), 0.0); }
  let rc = min(rho, 0.5 * P.rqMax);
  var sb = ogcSolidBallBound(h, x0, rc, P.r);
  if (sb.far) { sb.g0 = vec3<f32>(0.0); sb.D = vec3<f32>(0.0); }
  return boundTerms(v, h, sb, x0, y, rc, margin);
}
// f(d) = D.|d| - q.d, convex and positively homogeneous (an inside marker allows only d = 0).
fn cutOf(t: OgcTerms, d: vec3<f32>) -> f32 {
  if (t.q.w >= 1.0e38) { return select(0.0, 3.0e38, any(d != vec3<f32>(0.0))); }
  return dot(t.D, abs(d)) - dot(t.q.xyz, d);
}
// The largest scale k <= 1 of a prediction d that either certificate keeps (1: no bound).
fn predictionScale(t: OgcTerms, d: vec3<f32>) -> f32 {
  if (t.q.w == 0.0) { return 1.0; }
  let l = length(d);
  if (l <= t.rl) { return 1.0; }
  let f = cutOf(t, d);
  if (f <= 1.0 && l <= t.q.w) { return 1.0; }
  var ka = t.q.w / l;
  if (f > 1.0) { ka = min(ka, 1.0 / f); }
  return max(ka, t.rl / l);
}
// The static solid over the ball |d| <= rho around x0: the truncation terms (q, D) =
// (g0, D) / max(gamma_r phi0, phi0 - r), so that phi(x0 + d) >= min((1 - gamma_r) phi0, r) wherever
// D.|d| - q.d <= 1, and a solid entry for the vertex when the ball can reach r.
fn solidRound(v: u32, x0: vec3<f32>, rho: f32, margin: f32) -> OgcTerms {
  let h = ogcSolidHeader();
  let rel = (x0 - h.origin) / (8.0 * h.cell);
  if (any(rel < vec3<f32>(0.0)) || any(rel >= vec3<f32>(h.dims))) { atomicOr(&counters[OGC_STATUS_WORD], OGC_ST_SOLID_BOUNDS); }
  let qj = roundBound(v, h, x0, rho, OGC_SOLID_Y, margin);
  if (qj.q.w != 0.0 && qj.q.w < 1.0e38) { atomicAdd(&wgSolid, 1u); }
  putWork4(P.solidOff + 4u * v, qj.q);
  putWork4(P.solidDevOff + 4u * v, vec4<f32>(qj.D, qj.rl));
  return qj;
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let v = gid.x;
  let mandatory = atomicLoad(&counters[OGC_ROUNDS]) == 0u;
  let forced = atomicLoad(&counters[OGC_FORCE_HELD]) != 0u;
  // A redone round repeats its round's detection with its conflicts' vertices held or carried; only the first measures.
  let fresh = mandatory && !forced;
  if (v == 0u) { atomicStore(&counters[OGC_PAIR_SLOTS], 0u); atomicStore(&counters[OGC_CONFLICTS], 0u); }
  wgCube[li] = 0.0;
  if (v >= P.N && v < P.N + P.SB) {
    // The moving solid's sample: x_det at the detection pose, predicted exactly to the end pose.
    let b = 4u * (atomicLoad(&contacts[13u]) + v - P.N);
    let local = bitcast<vec4<f32>>(vec4<u32>(atomicLoad(&contacts[b]), atomicLoad(&contacts[b + 1u]), atomicLoad(&contacts[b + 2u]), atomicLoad(&contacts[b + 3u])));
    let s = v - P.N;
    let x1 = ogcToWorld(ogcPose(1u), local.xyz);
    let x0 = select(x1, ogcToWorld(ogcPose(0u), local.xyz), mandatory);
    xdet[P.N + P.S + s] = vec4<f32>(x0, local.w);
    pred[2u * P.N + s] = vec4<f32>(x1 - x0, 0.0);
  }
  if (v < P.N) {
    let x = select(pos[v].xyz, xt[v].xyz, mandatory);
    // The step's predicted displacement: v dt + g dt^2, or the solver's own initial
    // guess x_guess - x_t (in positions at the mandatory round); its length is kept
    // in pred[v].w for the held envelope of the step's later rounds.
    var inertial = vel[v].xyz * P.dt + P.gravity.xyz * (P.dt * P.dt);
    if (P.predict == 2u) { inertial = pos[v].xyz - x; }
    if (P.predict == 3u) { inertial = tgt[v].xyz - x; }
    var plen = length(inertial);
    if (!mandatory) { plen = pred[v].w; }
    var flags = atomicLoad(&work[P.heldOff + v]);
    var yref = pred[P.N + v];
    if (fresh) {
      // Held this step when last step saw it in contact, truncated or in a conflict,
      // or when its prediction would leave it more than half of its motion to cover
      // anyway (rho > |w| / 2): Planar-DAT's envelope then costs no more pairs and
      // caps no more often, and a held vertex never conflicts.
      yref = vec4<f32>(x + inertial, select(length(x - yref.xyz), plen, yref.w < 0.0));
      // In contact and touching or moving with the moving solid last step: held in its frame.
      flags = ((flags & 1u) << 1u) | select(0u, 32u, (flags & 1u) != 0u && (flags & 80u) != 0u);
      if (2.0 * yref.w > plen) { flags |= 2u; }
      atomicStore(&work[P.heldOff + v], flags);
    }
    // Held: a vertex the step's fresh round held (bit 1) or a conflict marked (bit 3); after the
    // mandatory round, any vertex with a flag (contact or truncation, bit 0). Without a moving
    // solid a round redone holds every vertex (Planar-DAT); with one, only those its conflicts
    // marked, so the rest keep their predictions.
    let heldBits = select(flags & 15u, flags & 10u, mandatory);
    let held = (forced && P.bodyBase == 0u) || P.predict == 0u || P.mode != OGC_PLANAR || heldBits != 0u;
    // Carried with the moving solid (bit 2): in an impact zone a conflict with it formed this step.
    let carried = P.bodyBase != 0u && (flags & 4u) != 0u;
    var w = select(yref.xyz - x, vec3<f32>(0.0), held);
    var dev = select(yref.w, plen, held);
    atomicStore(&contacts[P.countsOff + v], 0u);
    // The moving solid: its terms (R1 q, j) and the co-moving displacement a_v.
    // The terms are in the body frame at the end pose: the relative motion d is R1^T (w - a_v).
    var bq = OgcTerms(vec4<f32>(0.0), vec3<f32>(0.0), 0.0);
    var bRt = mat3x3<f32>(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0, 0.0, 1.0));
    var a = vec3<f32>(0.0);
    // Pushed: the moving solid moved this vertex's prediction (it is then its reference).
    var pushed = false;
    var env = 0.0;
    var sigma = 0.0;
    var theta = 0.0;
    var withBody = false;
    var pd = OgcPose(vec3<f32>(0.0), bRt);
    if (P.bodyBase != 0u) {
      let p1 = ogcPose(1u);
      bRt = transpose(p1.R);
      pd = p1;
      if (mandatory) { pd = ogcPose(0u); }
      let y0 = ogcToBody(pd, x);
      a = ogcToWorld(p1, y0) - x;
      // Held with the solid (bit 5: last step it was in contact with it, or moved with it in
      // contact), a held vertex is held in the solid's frame: w = a_v, as the solid's frame is
      // where its pairs with the solid hold (Planar-DAT there). Carried or held with the solid,
      // the vertex deviates from its prediction by its inertial motion less a_v.
      if (carried || (held && (flags & 32u) != 0u && length(y0) < P.bodyReach)) {
        w = a; dev = length(inertial - a); pushed = true; withBody = true;
        atomicOr(&work[P.heldOff + v], 64u);
      }
      env = envelopeOf(dev);
      if (mandatory) {
        let Rr = p1.R * transpose(pd.R);
        theta = 2.0 * asin(min(1.0, 0.5 * sqrt(max(3.0 - (Rr[0].x + Rr[1].y + Rr[2].z), 0.0))));
      }
      let hb = ogcSolidHeaderAt(P.bodyBase);
      // Relative to the solid the round moves the vertex at most |w - a_v| + env / 2 along its
      // chord, and sigma_v off it. A ball that clears every stored brick (all within bodyReach
      // of the body origin) is free of it.
      let rho = length(w - a) + 0.5 * env;
      // Q covers every prediction the round may end with: w moved towards a_v, then scaled towards 0.
      let dc = p1.c - pd.c;
      let Q = max(max(length(w - dc), length(a - dc)), length(dc)) + 0.5 * env;
      sigma = (theta * theta * (length(x - pd.c) + Q) + 2.0 * theta * Q) / 8.0;
      // Moving with the solid, the vertex's round is straight in the solid's frame: none there.
      let bm = select(sigma, 0.0, withBody);
      if (length(y0) - rho - bm < P.bodyReach) {
        bq = roundBound(v, hb, y0, rho, OGC_BODY_Y, bm);
        // It touches the solid when its ball can reach r (a far ball's terms are only the
        // Lipschitz length limit, q = D = 0): held with the solid next step only then.
        if (bq.q.w != 0.0 && bq.q.w < 1.0e38) {
          atomicAdd(&wgBody, 1u);
          if (any(bq.q.xyz != vec3<f32>(0.0)) || any(bq.D != vec3<f32>(0.0))) { atomicOr(&work[P.heldOff + v], 16u); }
        }
        // The prediction moves towards a_v only as far as the bound needs; in a round redone the
        // bound carries the vertex outright (it joins the solid's impact zone).
        let kb = predictionScale(bq, bRt * (w - a));
        if (kb < 1.0) {
          if (forced) {
            w = a; withBody = true;
            let old = atomicOr(&work[P.heldOff + v], 69u);
            if ((old & 4u) == 0u) { atomicAdd(&wgZone, 1u); }
          } else { w = a + (w - a) * kb; }
          pushed = true;
        }
      }
      // Moving with the solid (bit 6), the vertex's round is straight in the solid's frame, and in the
      // world it is c(t) + R(t) l(t) with l straight from y0: it leaves its world chord by at most
      // sigma_v = (theta^2 (|x_det - c0| + Q) + 2 theta Q) / 8 with Q = |w - a_v| + |w| + env / 2 >= |l'|
      // (w then scaled towards 0 by the static solid, and the residual within env / 2).
      if (withBody) {
        let Qs = length(w - a) + length(w) + 0.5 * env;
        sigma = (theta * theta * (length(x - pd.c) + Qs) + 2.0 * theta * Qs) / 8.0;
      }
    }
    if (P.bodyBase == 0u) { env = envelopeOf(dev); }
    if (P.solidBase != 0u) {
      // The round moves the vertex at most |w| + env / 2 (the cap is 0.5 gamma_r or gamma_p of env),
      // straight in the world, or within sigma_v of straight when it moves with the solid.
      let qj = solidRound(v, x, length(w) + 0.5 * env, select(0.0, sigma, withBody));
      if (qj.q.w >= 1.0e38) { w = vec3<f32>(0.0); } else { w *= predictionScale(qj, w); }
    }
    if (P.bodyBase != 0u) {
      // Both bounds hold the prediction (f32 rounding aside), or the cloth is squeezed.
      if (predictionScale(bq, (bRt * (w - a)) * (1.0 - 1.0e-5)) < 1.0) { atomicAdd(&wgSqueeze, 1u); }
      putWork4(P.bodyOff + 8u * v, bq.q);
      putWork4(P.bodyOff + 8u * v + 4u, vec4<f32>(a, sigma));
      putWork4(P.bodyDevOff + 4u * v, vec4<f32>(bq.D, bq.rl));
    }
    if (fresh) { pred[P.N + v] = vec4<f32>(select(x + w, yref.xyz, held && !pushed), yref.w); }
    pred[v] = vec4<f32>(w, plen);
    if (!(all(abs(x) < vec3<f32>(3.0e38)))) { atomicOr(&wgBad, 1u); }
    xdet[v] = vec4<f32>(x, 0.0);
    atomicStore(&work[P.envOff + v], bitcast<u32>(env));
    atomicStore(&work[P.tvOff + v], bitcast<u32>(1.0));
    atomicMax(&wgMaxEnv, bitcast<u32>(env));
    let e = env / P.rqMax;
    wgCube[li] = e * e * e;
    if (fresh && held) { atomicAdd(&wgHeld, 1u); }
  }
  workgroupBarrier();
  if (li == 0u) {
    let m = atomicLoad(&wgMaxEnv);
    if (m != 0u) { atomicMax(&counters[OGC_MAX_ENVELOPE], m); }
    var cube = 0.0;
    for (var i = 0u; i < 64u; i++) { cube += wgCube[i]; }
    let es = u32(round(cube * 1024.0));
    if (es != 0u) { atomicAdd(&counters[OGC_ENVELOPE_SUM], es); }
    let caps = atomicLoad(&wgCap);
    if (caps != 0u) { atomicAdd(&counters[OGC_F_ENVELOPE_CAP], caps); atomicOr(&counters[OGC_STATUS_WORD], OGC_ST_ENVELOPE_CAP); }
    if (atomicLoad(&wgBad) != 0u) { atomicOr(&counters[OGC_STATUS_WORD], OGC_ST_NONFINITE); }
    let hc = atomicLoad(&wgHeld);
    if (hc != 0u) { atomicAdd(&counters[OGC_F_HELD], hc); }
    let sc = atomicLoad(&wgSolid);
    if (sc != 0u) { atomicAdd(&counters[OGC_F_SOLID_CONTACTS], sc); }
    let bc = atomicLoad(&wgBody);
    if (bc != 0u) { atomicAdd(&counters[OGC_F_BODY_CONTACTS], bc); }
    if (atomicLoad(&wgSqueeze) != 0u) { atomicOr(&counters[OGC_STATUS_WORD], OGC_ST_BODY_SQUEEZE); }
    let zc = atomicLoad(&wgZone);
    if (zc != 0u) { atomicAdd(&counters[OGC_STEP_ZONE], zc); atomicAdd(&counters[OGC_F_ZONE_VERTICES], zc); }
  }
}
`;

/**
 * The round's grid cell: the larger of the authored floor (default the larger of
 * the mean edge length and rqBase) and either the round's largest envelope
 * (cellRule 0, the default) or its cubic-mean envelope (mean r_q^3)^(1/3)
 * (cellRule 1). Every primitive goes into every cell its swept, enveloped box
 * touches. The cubic mean bounds the grid's entries at about (2 + 2)^3 per
 * primitive whatever the spread of envelopes, but the gates measure the queries'
 * per-cell cost above the candidates a larger cell adds: the twist's round costs
 * 44 ms at the cubic mean (1 cm) and 11 ms at 4 cm. The sum is in units of
 * rqMax^3 / 1024, so N <= 4M.
 */
export const OGC_CELL_WGSL = header([0, 9]) + /* wgsl */`
@compute @workgroup_size(1)
fn main() {
  let m = bitcast<f32>(atomicExchange(&counters[OGC_MAX_ENVELOPE], 0u));
  atomicMax(&counters[OGC_F_PEAK_ENVELOPE], bitcast<u32>(m));
  atomicStore(&counters[OGC_ROUND_ENVELOPE], bitcast<u32>(m));
  let cube = f32(atomicExchange(&counters[OGC_ENVELOPE_SUM], 0u)) / (1024.0 * f32(P.N));
  let h = select(max(P.cellFloor, m), max(P.cellFloor, P.rqMax * pow(cube, 1.0 / 3.0)), P.cellRule == 1u);
  atomicStore(&counters[OGC_CELL], bitcast<u32>(h));
  atomicStore(&counters[OGC_INV_CELL], bitcast<u32>(1.0 / h));
}
`;

const ARGS_FNS = /* wgsl */`
fn writeArgs(on: bool) {
  var n = array<u32, 9>(P.wgPrep, 1u, P.wgT, P.wgS1, P.wgS2, 1u, P.wgT, P.wgNE, P.wgN);
  for (var i = 0u; i < 9u; i++) {
    args[3u * i] = select(0u, n[i], on); args[3u * i + 1u] = 1u; args[3u * i + 2u] = 1u;
  }
}
`;

/**
 * Start of a step: the mandatory detection round, and the step's round count reset.
 * With a moving solid, the step's start and end poses are copied from the frame's
 * pose table (entries k and k + 1 for the frame's step k; the table's first vec4
 * holds its pose count); a step past the table fails as OGC_ST_BODY_POSE.
 */
export const OGC_STEP_BEGIN_WGSL = header([0, 8, 9, 10]) + ARGS_FNS + /* wgsl */`
const OGC_POSE_WORD = ${OGC_POSE_WORD}u;
const OGC_POSE_WORDS = ${OGC_POSE_WORDS}u;
@compute @workgroup_size(1)
fn main() {
  if (P.bodyBase != 0u) {
    let k = atomicLoad(&counters[OGC_F_STEPS]);
    let table = 4u * atomicLoad(&contacts[15u]);
    if (k + 1u < atomicLoad(&contacts[table])) {
      for (var i = 0u; i < 2u * OGC_POSE_WORDS; i++) {
        atomicStore(&contacts[OGC_POSE_WORD + i], atomicLoad(&contacts[table + 4u + OGC_POSE_WORDS * k + i]));
      }
    } else {
      atomicOr(&counters[OGC_STATUS_WORD], OGC_ST_BODY_POSE);
    }
  }
  atomicStore(&counters[OGC_ROUNDS], 0u);
  atomicStore(&counters[OGC_EXCEED], 0u);
  atomicStore(&counters[OGC_FORCE_HELD], 0u);
  atomicStore(&counters[OGC_STEP_ZONE], 0u);
  atomicAdd(&counters[OGC_F_STEPS], 1u);
  atomicAdd(&counters[OGC_F_DETECTIONS], 1u);
  if (P.N > 0u) { writeArgs(true); } else { writeArgs(false); }
}
`;

/**
 * Between iterations (OGC Algorithm 3, lines 23-30): when gamma_e N vertices
 * hit their bound, the next detection round's dispatches get their workgroups;
 * otherwise they get none. The decision never leaves the GPU.
 */
export const OGC_DECIDE_WGSL = header([0, 9, 10]) + ARGS_FNS + /* wgsl */`
@compute @workgroup_size(1)
fn main() {
  atomicStore(&counters[OGC_FORCE_HELD], 0u);
  let ex = atomicExchange(&counters[OGC_EXCEED], 0u);
  // A vertex the line-search filter held at alpha 0 would hold the whole Newton step: re-detect.
  let stalled = atomicExchange(&counters[OGC_FILTER_STALLED], 0u);
  atomicAdd(&counters[OGC_F_DECIDES], 1u);
  atomicAdd(&counters[OGC_F_TRUNCATED], ex);
  atomicMax(&counters[OGC_F_TRUNCATED_MAX], ex);
  var on = false;
  if ((ex > 0u && ex >= P.exceedThreshold) || stalled != 0u) {
    if (atomicLoad(&counters[OGC_ROUNDS]) < P.roundCap) {
      atomicAdd(&counters[OGC_ROUNDS], 1u);
      atomicAdd(&counters[OGC_F_REDETECTIONS], 1u);
      atomicAdd(&counters[OGC_F_DETECTIONS], 1u);
      on = true;
    } else {
      atomicOr(&counters[OGC_STATUS_WORD], OGC_ST_ROUND_CAP);
      atomicAdd(&counters[OGC_F_ROUND_CAP], 1u);
    }
  }
  writeArgs(on);
}
`;

/**
 * After a detection round: a round with conflicting pairs (predictions that cross,
 * or no plane at X while a vertex is predicted to move) is redone at once, before
 * any truncation. Without a moving solid every prediction is held at 0 there, which
 * is Planar-DAT; with one, only the conflicts' vertices are held or carried (impact
 * zones), and a round redone that conflicts is redone again, up to OGC_REDO_ROUNDS.
 * The conflicts' vertices stay held for the rest of the step and the next.
 */
export const OGC_REDO_WGSL = header([0, 9, 10]) + ARGS_FNS + REDO_CAP_FN + /* wgsl */`
@compute @workgroup_size(1)
fn main() {
  atomicMax(&counters[OGC_F_ZONE_PEAK], atomicLoad(&counters[OGC_STEP_ZONE]));
  let c = atomicExchange(&counters[OGC_CONFLICTS], 0u);
  let progress = atomicExchange(&counters[OGC_REDO_PROGRESS], 0u);
  if (c == 0u) { writeArgs(false); return; }
  atomicAdd(&counters[OGC_F_CONFLICTS], c);
  let redone = atomicLoad(&counters[OGC_FORCE_HELD]);
  if (redone == 0u || (P.bodyBase != 0u && progress != 0u && redone < redoCap())) {
    atomicStore(&counters[OGC_FORCE_HELD], redone + 1u);
    atomicAdd(&counters[OGC_F_REDOS], 1u);
    atomicAdd(&counters[OGC_F_DETECTIONS], 1u);
    writeArgs(true);
  } else {
    atomicOr(&counters[OGC_STATUS_WORD], OGC_ST_HELD_CONFLICT);
    writeArgs(false);
  }
}
`;

/**
 * After the last round redone: a conflict left there fails as OGC_ST_HELD_CONFLICT.
 */
export const OGC_REDO_LAST_WGSL = header([0, 9, 10]) + ARGS_FNS + /* wgsl */`
@compute @workgroup_size(1)
fn main() {
  atomicMax(&counters[OGC_F_ZONE_PEAK], atomicLoad(&counters[OGC_STEP_ZONE]));
  let c = atomicExchange(&counters[OGC_CONFLICTS], 0u);
  atomicStore(&counters[OGC_REDO_PROGRESS], 0u);
  if (c > 0u) {
    atomicOr(&counters[OGC_STATUS_WORD], OGC_ST_HELD_CONFLICT);
    atomicAdd(&counters[OGC_F_CONFLICTS], c);
  }
  writeArgs(false);
}
`;

export function ogcFrameBeginWGSL(frameCounters: readonly number[]): string {
  return header([9]) + /* wgsl */`
@compute @workgroup_size(1)
fn main() {
${frameCounters.map(i => `  atomicStore(&counters[${i}u], 0u);`).join("\n")}
}
`;
}

const DECL_PAIRS_READ = "@group(0) @binding(11) var<storage, read> pairs: array<vec4<u32>>;";

/**
 * Planar-DAT division planes (Alg. 2, lines 2-7), one thread per pair. The plane
 * normal and the pair come from detection at X = x_det. With predictions the plane
 * translates: it is placed among the predicted positions X + w at the step's end
 * (detection placed none of a round with crossing predictions), and the placement
 * adapts to the accumulated residual delta = x - X - w: each side's share of the
 * predicted support gap is proportional to its own largest residual approach along
 * n, with at least 5% of the gap (and sepEps) left to each side. Each vertex's
 * truncation ratio (eq. 16) of its residual is reduced into tv by atomic min.
 */
export function ogcPlanesWGSL(fixed: boolean, filter: boolean): string {
  return [PARAMS_STRUCT, OGC_COUNTERS_WGSL, OGC_COMMON_WGSL, DECL[0], DECL[2], DECL[4], DECL[5], DECL[9], DECL_PAIRS_READ, DECL[14], ...(filter ? [DECL[15]] : [])].join("\n") + PRED_FNS + BODY_PAIR_FNS + /* wgsl */`
fn ratio(s0: f32, s1: f32) -> f32 {
  // A wrong-side start may move toward its side but never further across.
  if (s0 < 0.0) { return select(0.0, 1.0, s1 >= s0); }
  if (s1 >= 0.0) { return 1.0; }
  return clamp(P.gammaR * s0 / (s0 - s1), 0.0, 1.0);
}
// A surface sample (index >= N) is static: it neither moves nor is truncated.
fn lower(v: u32, t: f32) { if (t < 1.0 && v < P.N) { atomicMin(&work[P.tvOff + v], bitcast<u32>(t)); } }
var<workgroup> wgUncert: atomic<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) li: u32, @builtin(num_workgroups) nwg: vec3<u32>) {
  // Grid-stride: a round's slots can exceed 65535 workgroups of 64.
  let slots = min(atomicLoad(&counters[OGC_PAIR_SLOTS]), P.pairCap);
  for (var i = gid.x; i < slots; i += 64u * nwg.x) {
    let raw = pairs[2u * i];
    if (raw.x != OGC_NONE) {
      let pl = bitcast<vec4<f32>>(pairs[2u * i + 1u]);
      let nA = select(1u, 2u, (raw.x & OGC_EE_BIT) != 0u);
      var ids = vec4<u32>(raw.x & ~OGC_EE_BIT, raw.y, raw.z, raw.w);
      if (!(pl.w > 0.0)) {
        // No separating plane at X: the pair's vertices hold until the next round.
        for (var k = 0u; k < 4u; k++) { lower(ids[k], 0.0); }
        atomicAdd(&wgUncert, 1u);
      } else {
        let n = pl.xyz;
        let o = xdet[ids[nA]].xyz;
        var proj: array<f32, 4>;
        var dn: array<f32, 4>;
        var supA = 3.4e38; var supB = -3.4e38; var apA = 0.0; var apB = 0.0;
        for (var k = 0u; k < 4u; k++) {
          let Y = anchored(ids[k]);
          proj[k] = dot(n, Y - o);
          dn[k] = select(0.0, dot(n, pos[min(ids[k], P.N - 1u)].xyz - Y), ids[k] < P.N);
          if (k < nA) { supA = min(supA, proj[k]); apA = max(apA, -dn[k]); }
          else { supB = max(supB, proj[k]); apB = max(apB, dn[k]); }
        }
        let gap = supA - supB;
        // Each side keeps sepEps from the plane, and a vertex whose path strays from straight in
        // the pair's frame its stray more (sigma_v); a pair closer than both sides' keep places the
        // plane where both fall equally short.
        let turned = turnedPair(ids);
        let st = pairStray(ids, nA);
        let eA = P.sepEps + st.x; let eB = P.sepEps + st.y;
        var plane = supB + 0.5 * (gap + eB - eA);
        if (${fixed ? "false" : "gap >= eA + eB"}) {
          var lam = 0.5;
          if (apA + apB > 0.0) { lam = apB / (apA + apB); }
          lam = clamp(clamp(lam, 0.05, 0.95), eB / gap, 1.0 - eA / gap);
          plane = supB + lam * gap;
        }
        for (var k = 0u; k < 4u; k++) {
          let a = k < nA;
          let s0 = select(plane - proj[k], proj[k] - plane, a) - P.sepEps - strayOf(ids[k], turned);
          ${filter ? `if (ids[k] < P.N) {
            // Along the direction from x: the slack s(x + alpha d) keeps s0 - gamma_r max(s0, 0), the
            // truncation's own target (a wrong-side anchor: no further across), linear in alpha.
            let sx = s0 + select(-dn[k], dn[k], a);
            let sd = select(-1.0, 1.0, a) * dot(n, dir[ids[k]].xyz);
            if (sd < 0.0) { lower(ids[k], clamp((sx - (s0 - P.gammaR * max(s0, 0.0))) / -sd, 0.0, 1.0)); }
          }` : "lower(ids[k], ratio(s0, s0 + select(-dn[k], dn[k], a)));"}
        }
      }
    }
  }
  workgroupBarrier();
  if (li == 0u) {
    let u = atomicLoad(&wgUncert);
    if (u != 0u) { atomicAdd(&counters[OGC_F_UNCERTIFIED], u); atomicOr(&counters[OGC_STATUS_WORD], OGC_ST_UNCERTIFIED); }
  }
}
`;
}

/** The bounds' cut along a motion (apply and the line-search filter). */
const CUT_FNS = /* wgsl */`
fn work4(b: u32) -> vec4<f32> {
  return bitcast<vec4<f32>>(vec4<u32>(atomicLoad(&work[b]), atomicLoad(&work[b + 1u]), atomicLoad(&work[b + 2u]), atomicLoad(&work[b + 3u])));
}
// The largest scale k <= 1 of delta that one of the bound's certificates keeps for the motion
// base + k delta (prep, OgcTerms): the affine one within its ball |base + k delta| <= rc (q.w),
// f(base + k delta) <= f(base) + k f(delta) <= 1 with f(d) = D.|d| - q.d subadditive and
// positively homogeneous (room 1 - f(base), none from a base it does not hold), or the
// Lipschitz ball |base + k delta| <= rl when it holds the base. An inside marker allows no motion.
fn cutScale(q: vec4<f32>, D4: vec4<f32>, base: vec3<f32>, delta: vec3<f32>) -> f32 {
  if (q.w >= 1.0e38) { return select(1.0, 0.0, any(delta != vec3<f32>(0.0))); }
  if (q.w == 0.0) { return 1.0; }
  let D = D4.xyz;
  let bb = dot(base, base);
  // The prediction scale leaves a base on a ball's sphere to f32 rounding.
  let tol = 1.0 + 1.0e-5;
  var k = 0.0;
  if (bb <= q.w * q.w * tol) {
    let room = max(1.0 - (dot(D, abs(base)) - dot(q.xyz, base)), 0.0);
    let den = dot(D, abs(delta)) - dot(q.xyz, delta);
    k = min(select(1.0, room / den, den > room), ballExit(base, delta, q.w));
  }
  if (k < 1.0 && bb <= D4.w * D4.w * tol) { k = max(k, ballExit(base, delta, D4.w)); }
  return k;
}
// The largest t <= 1 with |base + t delta| <= R, for a base within R.
fn ballExit(base: vec3<f32>, delta: vec3<f32>, R: f32) -> f32 {
  let e = base + delta;
  let r2 = R * R;
  if (dot(e, e) <= r2) { return 1.0; }
  let dd = dot(delta, delta);
  let bd = dot(base, delta);
  return clamp((sqrt(max(bd * bd + dd * (r2 - dot(base, base)), 0.0)) - bd) / dd, 0.0, 1.0);
}
`;

/**
 * Apply the truncation (Alg. 2, lines 8-12) to the residual delta = x - X - w:
 * delta <- t_v delta, then |delta| <= x_det.w, which is 0.5 gamma_r r_q,v in planar
 * mode (a pair left out of the round is further apart than the two caps plus its
 * prediction spread) and OGC's b_v in isotropic mode (w = 0 there). Each truncated
 * vertex counts towards the re-detection decision and is held for the rest of the
 * step and the next; tv is left at 1 for the next pass. A static solid then cuts
 * delta by its affine lower bound of phi (prep), which the prediction already meets:
 * f(w + delta) <= f(w) + j |delta| - q.delta for the convex f(d) = j |d| - q.d.
 * The moving solid cuts delta alike, its bound on the motion relative to the solid
 * w - a_v + delta (its q already turned to the world frame by the end pose).
 */
export const OGC_APPLY_WGSL = header([0, 2, 4, 5, 8, 9, 14], [2]) + PRED_FNS + SOLID_ACCESS + OGC_SOLID_WGSL + CUT_FNS + /* wgsl */`
var<workgroup> wgCount: atomic<u32>;
var<workgroup> wgCapped: atomic<u32>;
var<workgroup> wgSolidCut: atomic<u32>;
var<workgroup> wgBodyCut: atomic<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let v = gid.x;
  if (v < P.N) {
    let tv = bitcast<f32>(atomicExchange(&work[P.tvOff + v], bitcast<u32>(1.0)));
    let p = pos[v];
    let det = xdet[v];
    let w = predOf(v);
    var delta = (p.xyz - det.xyz - w) * tv;
    var cut = tv < 1.0;
    let l = length(delta);
    if (l > det.w) { delta *= det.w / l; cut = true; atomicAdd(&wgCapped, 1u); }
    if (P.solidBase != 0u) {
      // The static solid: phi(x_det + w + t delta) >= min((1 - gamma_r) phi0, r) for t <= (1 - f(w)) / f(delta).
      let k = cutScale(work4(P.solidOff + 4u * v), work4(P.solidDevOff + 4u * v), w, delta);
      if (k < 1.0) { delta *= k; cut = true; atomicAdd(&wgSolidCut, 1u); }
    }
    if (P.bodyBase != 0u) {
      // The moving solid: phi_body >= min((1 - gamma_r) phi0, r) for the motion w - a_v + delta relative to it.
      // Its terms are in the body frame at the end pose.
      let qj = work4(P.bodyOff + 8u * v);
      if (qj.w != 0.0) {
        let Rt = transpose(ogcPose(1u).R);
        let k = cutScale(qj, work4(P.bodyDevOff + 4u * v), Rt * (w - work4(P.bodyOff + 8u * v + 4u).xyz), Rt * delta);
        if (k < 1.0) { delta *= k; cut = true; atomicAdd(&wgBodyCut, 1u); atomicOr(&work[P.heldOff + v], 16u); }
      }
    }
    if (cut) {
      pos[v] = vec4<f32>(det.xyz + w + delta, p.w);
      atomicAdd(&wgCount, 1u);
      atomicOr(&work[P.heldOff + v], 1u);
    }
  }
  workgroupBarrier();
  if (li == 0u) {
    let c = atomicLoad(&wgCount);
    if (c != 0u) { atomicAdd(&counters[OGC_EXCEED], c); }
    let k = atomicLoad(&wgCapped);
    if (k != 0u) { atomicAdd(&counters[OGC_F_CAPPED], k); }
    let sc = atomicLoad(&wgSolidCut);
    if (sc != 0u) { atomicAdd(&counters[OGC_F_SOLID_CUT], sc); }
    let bc = atomicLoad(&wgBodyCut);
    if (bc != 0u) { atomicAdd(&counters[OGC_F_BODY_CUT], bc); }
  }
}
`;

/**
 * The line-search filter for a Newton solver: for the iterate x (positions, untouched) and
 * the solver's direction d, alpha_v = the largest alpha <= 1 with x + alpha d inside every
 * certificate of the vertex's round: the planes (ogcPlanesWGSL(fixed, true), reduced into
 * tv), the |x - X - w| <= x_det.w cap, and the static and moving solids' bounds (cutScale
 * with the base x - X, and x - X - a_v in the body frame). Each is a convex set containing
 * x, so alpha_v bounds an interval [0, alpha_v]. Writes alpha_v and the atomic min alpha_min
 * (alpha[N], set to 1 by OGC_FILTER_BEGIN_WGSL); the vertices it bounds below 1 count towards
 * the re-detection decision, and one held at 0 arms it outright (OGC_FILTER_STALLED).
 */
export const OGC_FILTER_WGSL = header([0, 2, 4, 5, 8, 9, 14, 15, 16]) + PRED_FNS + SOLID_ACCESS + OGC_SOLID_WGSL + CUT_FNS + /* wgsl */`
var<workgroup> wgBound: atomic<u32>;
var<workgroup> wgStalled: atomic<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let v = gid.x;
  if (v < P.N) {
    var al = min(bitcast<f32>(atomicExchange(&work[P.tvOff + v], bitcast<u32>(1.0))), 1.0);
    let x = pos[v].xyz;
    let det = xdet[v];
    let w = predOf(v);
    let d = dir[v].xyz;
    if (any(d != vec3<f32>(0.0))) {
      al = min(al, ballExit(x - det.xyz - w, d, det.w));
      if (P.solidBase != 0u) { al = min(al, cutScale(work4(P.solidOff + 4u * v), work4(P.solidDevOff + 4u * v), x - det.xyz, d)); }
      if (P.bodyBase != 0u) {
        let qj = work4(P.bodyOff + 8u * v);
        if (qj.w != 0.0) {
          let Rt = transpose(ogcPose(1u).R);
          al = min(al, cutScale(qj, work4(P.bodyDevOff + 4u * v), Rt * (x - det.xyz - work4(P.bodyOff + 8u * v + 4u).xyz), Rt * d));
        }
      }
    } else { al = 1.0; }
    al = max(al, 0.0);
    atomicStore(&alpha[v], bitcast<u32>(al));
    atomicMin(&alpha[P.N], bitcast<u32>(al));
    if (al < 1.0) { atomicAdd(&wgBound, 1u); atomicOr(&work[P.heldOff + v], 1u); }
    if (al <= 0.0) { atomicAdd(&wgStalled, 1u); }
  }
  workgroupBarrier();
  if (li == 0u) {
    let c = atomicLoad(&wgBound);
    if (c != 0u) { atomicAdd(&counters[OGC_EXCEED], c); }
    let z = atomicLoad(&wgStalled);
    if (z != 0u) { atomicAdd(&counters[OGC_FILTER_STALLED], z); }
  }
}
`;

/** Before a filter: alpha_min = 1 (it is the atomic min of the alpha_v that follow). */
export const OGC_FILTER_BEGIN_WGSL = header([0, 16]) + /* wgsl */`
@compute @workgroup_size(1)
fn main() { atomicStore(&alpha[P.N], bitcast<u32>(1.0)); }
`;

// ------------------------------------------------------------------ grid build

/** Count each face's (inflated) cell span into the hash buckets; reset its d_min. */
export function ogcGridCountWGSL(intersect: boolean): string {
  const bindings = intersect ? [0, 1, 4, 6, 9] : [0, 1, 4, 5, 6, 9, 14];
  return header(bindings) + TOPO_FNS + CELL_FNS + triBoxFns(intersect) + /* wgsl */`
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let t = gid.x;
  ${intersect ? "if (t == 0u) { atomicStore(&counters[OGC_INTERSECTIONS], 0u); atomicStore(&counters[OGC_SOLID_INTERSECTIONS], 0u); }" : ""}
  if (t >= P.T) { return; }
  ${intersect ? "" : "atomicStore(&work[P.dminTOff + t], 0x7f7fffffu);"}
  let inv = invCell();
  let box = triBox(t);
  let c0 = cellOf(box.lo, inv); let c1 = cellOf(box.hi, inv);
  for (var z = c0.z; z <= c1.z; z++) { for (var y = c0.y; y <= c1.y; y++) { for (var x = c0.x; x <= c1.x; x++) {
    atomicAdd(&counts[cellHash(vec3<i32>(x, y, z))], 1u);
  } } }
}
`;
}

/** Workgroup exclusive scan of 1024 values a block; level 1 reads the counts, levels 2 and 3 the sums below. */
export function ogcScanWGSL(level: 1 | 2 | 3): string {
  const bindings = level === 1 ? [0, 6, 7] : [0, 7];
  const n = level === 1 ? "P.H" : level === 2 ? "P.B1" : "P.B2";
  const input = level === 1 ? "atomicLoad(&counts[i])" : level === 2 ? "grid[P.sums1Off + i]" : "grid[P.sums2Off + i]";
  const out = level === 1 ? "P.local1Off" : level === 2 ? "P.local2Off" : "P.local3Off";
  const sums = level === 1 ? "P.sums1Off" : level === 2 ? "P.sums2Off" : "P.sums3Off";
  return header(bindings) + /* wgsl */`
var<workgroup> sh: array<u32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let n = ${n};
  let base = wg.x * 1024u + li * 4u;
  var a: array<u32, 4>;
  var s = 0u;
  for (var k = 0u; k < 4u; k++) { let i = base + k; var q = 0u; if (i < n) { q = ${input}; } a[k] = q; s += q; }
  sh[li] = s;
  workgroupBarrier();
  for (var off = 1u; off < 256u; off <<= 1u) {
    var t = 0u;
    if (li >= off) { t = sh[li - off]; }
    workgroupBarrier();
    sh[li] += t;
    workgroupBarrier();
  }
  var run = sh[li] - s;
  for (var k = 0u; k < 4u; k++) { let i = base + k; if (i < n) { grid[${out} + i] = run; } run += a[k]; }
  if (li == 255u) { grid[${sums} + wg.x] = sh[255]; }
}
`;
}

/** Scatter each face into its buckets; the count it takes back leaves the counts zero for the next round. */
export function ogcGridInsertWGSL(intersect: boolean): string {
  const bindings = intersect ? [0, 1, 4, 6, 7, 9] : [0, 1, 4, 6, 7, 9, 14];
  return header(bindings) + TOPO_FNS + CELL_FNS + GRID_READ_FNS + triBoxFns(intersect, false) + /* wgsl */`
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let t = gid.x;
  if (t == 0u) {
    let total = grid[P.sums3Off];
    atomicMax(&counters[OGC_PEAK_GRID], total);
    if (total > P.gridCap) { atomicOr(&counters[OGC_STATUS_WORD], OGC_ST_GRID_OVERFLOW); }
  }
  if (t >= P.T) { return; }
  let inv = invCell();
  let box = triBox(t);
  let c0 = cellOf(box.lo, inv); let c1 = cellOf(box.hi, inv);
  for (var z = c0.z; z <= c1.z; z++) { for (var y = c0.y; y <= c1.y; y++) { for (var x = c0.x; x <= c1.x; x++) {
    let c = vec3<i32>(x, y, z);
    let h = cellHash(c);
    let slot = bucketStart(h) + atomicSub(&counts[h], 1u) - 1u;
    if (slot < P.gridCap) {
      let b = P.entriesOff + OGC_ENTRY_WORDS * slot;
      let lo = bitcast<vec3<u32>>(box.lo); let hi = bitcast<vec3<u32>>(box.hi);
      grid[b] = t; grid[b + 1u] = cellKey(c);
      grid[b + 2u] = lo.x; grid[b + 3u] = lo.y; grid[b + 4u] = lo.z;
      grid[b + 5u] = hi.x; grid[b + 6u] = hi.y; grid[b + 7u] = hi.z;
    }
  } } }
}
`;
}

// ------------------------------------------------------------------ query (Algorithms 1 and 2) and bounds

/**
 * The query's roles, one pipeline each over the same dispatch (a workgroup with no
 * thread of its role exits at once): the cloth's vertices, its edges, and the solids'
 * surface samples with their hull edges. One kernel holding all of them carried the
 * largest role's registers into every thread (M1 Max: about 4 ms a round for 2
 * candidates per vertex).
 */
export const OGC_QUERY_ROLES = ["queryVertex", "queryEdge", "querySample"] as const;
export type OgcQueryRole = (typeof OGC_QUERY_ROLES)[number];
export function ogcQueryWGSL(role: OgcQueryRole): string {
  const lo = role === "queryVertex" ? "0u" : role === "queryEdge" ? "P.N" : "P.N + P.E";
  const hi = role === "queryVertex" ? "P.N" : role === "queryEdge" ? "P.N + P.E" : "P.N + P.E + P.S + P.SB + P.SE + P.SEB";
  return header([0, 1, 4, 5, 7, 8, 9, 11, 14]) + TOPO_FNS + CELL_FNS + GRID_READ_FNS + triBoxFns(false) + BODY_PAIR_FNS + SOLID_ACCESS + OGC_SOLID_WGSL + /* wgsl */`
fn xd(i: u32) -> vec3<f32> { return xdet[i].xyz; }

// ---- DAT division pairs: every non-adjacent pair whose swept boxes meet and that is
// within the larger of the two primitives' envelopes (a face's or an edge's is its
// vertices' widest) plus the pair's prediction spread, the largest difference of a
// vertex's prediction on one side from one on the other. Two primitives left out
// close by at most their two caps plus that spread over the round.
const PAIR_BLOCK = ${PAIR_BLOCK}u;
var<private> pairBase: u32 = 0u;
var<private> pairLeft: u32 = 0u;
fn writePair(slot: u32, ids: vec4<u32>, pl: vec4<f32>, real: bool) {
  if (slot < P.pairCap) { pairs[2u * slot] = ids; pairs[2u * slot + 1u] = bitcast<vec4<u32>>(pl); }
  else if (real) { atomicOr(&counters[OGC_STATUS_WORD], OGC_ST_PAIR_OVERFLOW); }
}
fn pushPair(ids: vec4<u32>, pl: vec4<f32>) {
  if (pairLeft == 0u) { pairBase = atomicAdd(&counters[OGC_PAIR_SLOTS], PAIR_BLOCK); pairLeft = PAIR_BLOCK; }
  writePair(pairBase + PAIR_BLOCK - pairLeft, ids, pl, true);
  pairLeft -= 1u;
}
fn closePairs() {
  while (pairLeft > 0u) { writePair(pairBase + PAIR_BLOCK - pairLeft, vec4<u32>(OGC_NONE, 0u, 0u, 0u), vec4<f32>(0.0), false); pairLeft -= 1u; }
}
fn unitOr0(a: vec3<f32>) -> vec3<f32> { let l = length(a); return select(vec3<f32>(0.0), a / l, l > 1e-30); }
// Support gaps along a unit axis n, measured from the second primitive's first vertex.
fn gapVT(n: vec3<f32>, x: vec3<f32>, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> f32 {
  return dot(n, x - a) - max(0.0, max(dot(n, b - a), dot(n, c - a)));
}
fn gapEE(n: vec3<f32>, p0: vec3<f32>, p1: vec3<f32>, q0: vec3<f32>, q1: vec3<f32>) -> f32 {
  return min(dot(n, p0 - q0), dot(n, p1 - q0)) - max(0.0, dot(n, q1 - q0));
}
// The closest-point axis, or else the better orientation of the fallback axis (face
// normal, edge cross product); a non-positive gap means no separating plane.
fn certify(n0: vec3<f32>, g0: f32, alt: vec3<f32>, gp: f32, gm: f32) -> vec4<f32> {
  var best = vec4<f32>(n0, g0);
  if (g0 >= 2.0 * P.sepEps) { return best; }
  if (gp > best.w) { best = vec4<f32>(alt, gp); }
  if (gm > best.w) { best = vec4<f32>(-alt, gm); }
  if (!(best.w > 0.0)) { best = vec4<f32>(0.0); }
  return best;
}
// A conflict: some vertex is predicted to move and either X has no separating plane or
// the predicted positions X + w close below sepEps along it (and below the gap at X).
// The truncation keeps a divided pair 2 sepEps apart (sepEps each side of its plane), so
// a pair pressed together rests there; a threshold at 2 sepEps would make rounding of
// a co-moving pair's predictions (cloth pushed ahead of the moving solid against the
// solid's hull edges) a conflict in every round. Below sepEps a prediction has closed
// the pair by at least sepEps, the rounding scale it is sized for; between sepEps and
// 2 sepEps the planes pass starts both sides wrong side and lets neither cross further.
// The planes pass computes the same predicted gap, bit for bit.
//
// Before it is a conflict the pair tries the predicted primitives' own axis (predictedAxis):
// the axis certified at X need not separate X + w (a sheet the moving solid's extension
// compresses or turns), while the predicted primitives may still be well apart. The pair's
// plane is returned: its own, or that axis.
fn resolvePlane(ids: vec4<u32>, nA: u32, pl: vec4<f32>) -> vec4<f32> {
  var moving = false;
  for (var k = 0u; k < 4u; k++) { if (any(predOf(ids[k]) != vec3<f32>(0.0))) { moving = true; } }
  if (!moving) { return pl; }
  // Both sides' strays at X too (each side's path may stray towards the plane).
  let st = pairStray(ids, nA);
  let sg = st.x + st.y;
  var conflict = !(pl.w > sg);
  if (!conflict) {
    let o = xdet[ids[nA]].xyz;
    var supA = 3.4e38; var supB = -3.4e38;
    for (var k = 0u; k < 4u; k++) {
      let pr = dot(pl.xyz, anchored(ids[k]) - o);
      if (k < nA) { supA = min(supA, pr); } else { supB = max(supB, pr); }
    }
    let g1 = supA - supB;
    conflict = g1 < P.sepEps + sg && g1 < pl.w;
  }
  if (!conflict) { return pl; }
  let alt = predictedAxis(ids, nA, sg);
  if (alt.w > 0.0) { return alt; }
  atomicAdd(&counters[OGC_CONFLICTS], 1u);
  // Its cloth is held, or with a moving solid in the pair (its sample, or cloth already
  // carried with it), carried with the solid: the impact zone the round redone moves with it.
  // A static sample's pair cannot be carried; it conflicts again until the last redo fails.
  var carry = false;
  if (P.bodyBase != 0u) {
    var stat = false;
    for (var k = 0u; k < 4u; k++) {
      let i = ids[k];
      if (i >= P.N + P.S && i < P.N + P.S + P.SB) { carry = true; }
      else if (i >= P.N) { stat = true; }
      else if ((atomicLoad(&work[P.heldOff + i]) & 64u) != 0u) { carry = true; }
    }
    carry = carry && !stat;
  }
  let mark = select(9u, 13u, carry);
  var progress = false;
  for (var k = 0u; k < 4u; k++) {
    if (ids[k] < P.N) {
      let old = atomicOr(&work[P.heldOff + ids[k]], mark);
      if (carry && (old & 4u) == 0u) { atomicAdd(&counters[OGC_STEP_ZONE], 1u); atomicAdd(&counters[OGC_F_ZONE_VERTICES], 1u); }
      if ((old & mark) != mark) { progress = true; }
    }
  }
  if (progress) { atomicAdd(&counters[OGC_REDO_PROGRESS], 1u); }
  // A conflict that marks nothing new, or one in the last round redone, is the step's
  // failure (OGC_REDO_WGSL, OGC_REDO_LAST_WGSL).
  let redone = atomicLoad(&counters[OGC_FORCE_HELD]);
  if (redone != 0u && (!progress || redone == redoCap() || P.bodyBase == 0u)) {
    recordHeldConflict(select(ids, vec4<u32>(ids.x | OGC_EE_BIT, ids.yzw), nA == 2u));
  }
  return pl;
}
// The predicted primitives' closest-point axis n1 (from the second to the first at X + w),
// as (n1, its support gap at X), or 0 when it does not serve. A pair's support gap along a
// fixed axis is concave along the straight paths X + t w (the least of linear functions
// less the largest), so an axis that separates the pair at X and leaves it sepEps apart
// at X + w (or no nearer than at X) separates it on the whole path: the predictions cross
// nothing, and the planes pass divides the pair along it. A pair fixed in the moving solid's
// frame measures X along the axis turned back to the start pose; at both ends the pair keeps
// its sides' strays sg more (a vertex whose path is straight in the other frame strays from
// straight in the pair's by up to sigma_v).
fn predictedAxis(ids: vec4<u32>, nA: u32, sg: f32) -> vec4<f32> {
  var Y: array<vec3<f32>, 4>;
  for (var k = 0u; k < 4u; k++) { Y[k] = anchored(ids[k]); }
  var n1 = vec3<f32>(0.0);
  if (nA == 1u) {
    let cl = ogcClosestTriangle(Y[0], Y[1], Y[2], Y[3]);
    n1 = unitOr0(Y[0] - (cl.bary.x * Y[1] + cl.bary.y * Y[2] + cl.bary.z * Y[3]));
  } else {
    let st = ogcSegmentSegment(Y[0], Y[1], Y[2], Y[3]);
    n1 = unitOr0(Y[0] + st.x * (Y[1] - Y[0]) - (Y[2] + st.y * (Y[3] - Y[2])));
  }
  var n0 = n1;
  if (turnedPair(ids)) { n0 = ogcPose(0u).R * (transpose(ogcPose(1u).R) * n1); }
  let o = xd(ids[nA]);
  var a0 = 3.4e38; var b0 = -3.4e38; var a1 = 3.4e38; var b1 = -3.4e38;
  for (var k = 0u; k < 4u; k++) {
    let p0 = dot(n0, xd(ids[k]) - o);
    let p1 = dot(n1, Y[k] - o);
    if (k < nA) { a0 = min(a0, p0); a1 = min(a1, p1); } else { b0 = max(b0, p0); b1 = max(b1, p1); }
  }
  let g0 = a0 - b0;
  let g1 = a1 - b1;
  if (any(n1 != vec3<f32>(0.0)) && g0 > sg && (g1 >= P.sepEps + sg || g1 >= g0)) { return vec4<f32>(n1, g0); }
  return vec4<f32>(0.0);
}
fn bodyFrame(ids: vec4<u32>, pl: vec4<f32>) -> vec4<f32> {
  if (!turnedPair(ids)) { return pl; }
  return vec4<f32>(ogcPose(1u).R * (transpose(ogcPose(0u).R) * pl.xyz), pl.w);
}
fn recordHeldConflict(ids: vec4<u32>) {
  for (var k = 0u; k < 4u; k++) { atomicStore(&counters[OGC_HELD_PAIR + k], ids[k]); }
}
fn pairVT(v: u32, tri: vec3<u32>, x: vec3<f32>, cp: vec3<f32>) {
  let a = xd(tri.x); let b = xd(tri.y); let c = xd(tri.z);
  let n0 = unitOr0(x - cp);
  let fnorm = unitOr0(cross(b - a, c - a));
  let ids = vec4<u32>(v, tri);
  pushPair(ids, resolvePlane(ids, 1u, bodyFrame(ids, certify(n0, gapVT(n0, x, a, b, c), fnorm, gapVT(fnorm, x, a, b, c), gapVT(-fnorm, x, a, b, c)))));
}
fn pairEE(er: vec4<u32>, r2: vec4<u32>, ca: vec3<f32>, cb: vec3<f32>) {
  let p0 = xd(er.x); let p1 = xd(er.y); let q0 = xd(r2.x); let q1 = xd(r2.y);
  let n0 = unitOr0(ca - cb);
  let ec = unitOr0(cross(p1 - p0, q1 - q0));
  let ids = vec4<u32>(er.x, er.y, r2.x, r2.y);
  let pl = bodyFrame(ids, certify(n0, gapEE(n0, p0, p1, q0, q1), ec, gapEE(ec, p0, p1, q0, q1), gapEE(-ec, p0, p1, q0, q1)));
  pushPair(vec4<u32>(er.x | OGC_EE_BIT, er.y, r2.x, r2.y), resolvePlane(ids, 2u, pl));
}
fn spreadVT(v: u32, tri: vec3<u32>) -> f32 {
  let w = predOf(v);
  return max(length(w - predOf(tri.x)), max(length(w - predOf(tri.y)), length(w - predOf(tri.z))));
}
fn spreadEE(a: vec4<u32>, b: vec4<u32>) -> f32 {
  let wa0 = predOf(a.x); let wa1 = predOf(a.y); let wb0 = predOf(b.x); let wb1 = predOf(b.y);
  return max(max(length(wa0 - wb0), length(wa0 - wb1)), max(length(wa1 - wb0), length(wa1 - wb1)));
}

fn pushEntry(u: u32, ent: vec4<u32>) {
  let slot = atomicAdd(&contacts[P.countsOff + u], 1u);
  if (slot < P.contactCap) {
    let b = P.entriesBase + 4u * (u * P.contactCap + slot);
    atomicStore(&contacts[b], ent.x); atomicStore(&contacts[b + 1u], ent.y);
    atomicStore(&contacts[b + 2u], ent.z); atomicStore(&contacts[b + 3u], ent.w);
  } else {
    atomicOr(&counters[OGC_STATUS_WORD], OGC_ST_CONTACT_OVERFLOW);
  }
}
fn pushSampleEntry(u: u32, ent: vec4<u32>, mark: u32) {
  pushEntry(u, ent);
  if (mark != 0u) { atomicOr(&work[P.heldOff + u], mark); }
}

// Algorithm 1 for a feature of face t closest to v: block membership (eqs. 8-9) and,
// so each (v, feature) is recorded once, ownership by the lowest-index face that
// holds the feature and that v's query does not skip as adjacent.
fn vfAccept(v: u32, x: vec3<f32>, t: u32, tri: vec3<u32>, feature: u32) -> bool {
  if (feature == OGC_FACE) { return true; }
  if (feature < 3u) {
    let k = tri[feature];
    let xk = xd(k);
    for (var i = topo[P.vtOffOff + k]; i < topo[P.vtOffOff + k + 1u]; i++) {
      let f = topo[P.vtListOff + i];
      if (f < t && !hasVertex(triVerts(f), v)) { return false; }
    }
    for (var i = topo[P.veOffOff + k]; i < topo[P.veOffOff + k + 1u]; i++) {
      let er = edgeRec(topo[P.veListOff + i]);
      let o = select(er.x, er.y, er.x == k);
      if (dot(x - xk, xk - xd(o)) < 0.0) { return false; }
    }
    return true;
  }
  let li = feature - 3u;
  let i = tri[li]; let j = tri[(li + 1u) % 3u];
  let er = edgeRec(topo[P.triEdgesOff + 4u * t + li]);
  let other = select(er.z, er.w, er.z == t);
  if (other != OGC_NONE && other < t && !hasVertex(triVerts(other), v)) { return false; }
  let xi = xd(i); let xj = xd(j); let dir = xj - xi;
  if (dot(x - xi, dir) <= 0.0 || dot(x - xj, -dir) <= 0.0) { return false; }
  for (var s = 0u; s < 2u; s++) {
    let f = select(er.z, er.w, s == 1u);
    if (f == OGC_NONE) { continue; }
    let fv = triVerts(f);
    var o = fv.x;
    if (o == i || o == j) { o = fv.y; }
    if (o == i || o == j) { o = fv.z; }
    let xo = xd(o);
    let p = xi + dir * (dot(xo - xi, dir) / dot(dir, dir));
    if (dot(x - p, p - xo) < 0.0) { return false; }
  }
  return true;
}

struct RoleOut { candidates: u32, pairs: u32, dat: u32 }

fn vertexFacet(v: u32) -> RoleOut {
  var out = RoleOut(0u, 0u, 0u);
  let planar = P.mode == OGC_PLANAR;
  let x = xd(v);
  let rv = envOf(v);
  let x1 = x + predOf(v);
  // One-sided: the faces' boxes carry the round's largest envelope, so the query is the bare swept box.
  let q = OgcBox(min(x, x1), max(x, x1));
  let inv = invCell();
  let c0 = cellOf(q.lo, inv); let c1 = cellOf(q.hi, inv);
  var dmin = 3.4e38;
  for (var cz = c0.z; cz <= c1.z; cz++) { for (var cy = c0.y; cy <= c1.y; cy++) { for (var cx = c0.x; cx <= c1.x; cx++) {
    let c = vec3<i32>(cx, cy, cz);
    let key = cellKey(c); let h = cellHash(c);
    let end = min(bucketStart(h + 1u), P.gridCap);
    for (var j = bucketStart(h); j < end; j++) {
      if (entryKey(j) != key) { continue; }
      let t = entryFace(j);
      let tri = triVerts(t);
      if (hasVertex(tri, v)) { continue; }
      let box = entryBox(j);
      if (!boxesOverlap(q, box) || any(cellOf(max(q.lo, box.lo), inv) != c)) { continue; }
      out.candidates++;
      let a = xd(tri.x); let b = xd(tri.y); let cc = xd(tri.z);
      let cl = ogcClosestTriangle(x, a, b, cc);
      let cp = cl.bary.x * a + cl.bary.y * b + cl.bary.z * cc;
      let d = length(x - cp);
      dmin = min(dmin, d);
      if (planar) {
        if (d <= max(rv, triEnv(tri)) + spreadVT(v, tri)) { pairVT(v, tri, x, cp); out.dat++; }
      } else {
        atomicMin(&work[P.dminTOff + t], bitcast<u32>(d));
      }
      if (!(d > 0.0)) { atomicOr(&counters[OGC_STATUS_WORD], OGC_ST_COINCIDENT); }
      else if (d < P.r && vfAccept(v, x, t, tri, cl.feature)) {
        let lambda = -ogcActivation(d, P.r, P.kc).y;
        let ent = vec4<u32>(v, t | (cl.feature << 28u), bitcast<u32>(lambda), 0u);
        pushEntry(v, ent);
        if (cl.feature < 3u) { pushEntry(tri[cl.feature], ent); }
        else if (cl.feature < OGC_FACE) { let li = cl.feature - 3u; pushEntry(tri[li], ent); pushEntry(tri[(li + 1u) % 3u], ent); }
        else { pushEntry(tri.x, ent); pushEntry(tri.y, ent); pushEntry(tri.z, ent); }
        out.pairs++;
      }
    }
  } } }
  atomicStore(&work[v], bitcast<u32>(dmin));
  return out;
}

// A surface sample i (index N + s) against the cloth's faces: the faces' grid boxes
// carry their predictions and the round's largest envelope, so the sample's bare swept
// box (a point for a static sample; to its exact predicted end for the moving solid's)
// finds every face within its envelope of the sample. Entries go to the face's feature
// vertices only (the sample is not solved for) and carry mu lambda with mu the solid's
// at the sample (x_det.w); w = 1 marks them. The DAT pair puts the sample first, as a vertex.
fn sampleFacet(i: u32) -> RoleOut {
  var out = RoleOut(0u, 0u, 0u);
  let planar = P.mode == OGC_PLANAR;
  let xs = xdet[i];
  let x = xs.xyz;
  let x1 = x + predOf(i);
  let q = OgcBox(min(x, x1), max(x, x1));
  let inv = invCell();
  let c0 = cellOf(q.lo, inv); let c1 = cellOf(q.hi, inv);
  for (var cz = c0.z; cz <= c1.z; cz++) { for (var cy = c0.y; cy <= c1.y; cy++) { for (var cx = c0.x; cx <= c1.x; cx++) {
  let c = vec3<i32>(cx, cy, cz);
  let key = cellKey(c); let h = cellHash(c);
  let end = min(bucketStart(h + 1u), P.gridCap);
  for (var j = bucketStart(h); j < end; j++) {
    if (entryKey(j) != key) { continue; }
    let box = entryBox(j);
    if (!boxesOverlap(q, box) || any(cellOf(max(q.lo, box.lo), inv) != c)) { continue; }
    let t = entryFace(j);
    let tri = triVerts(t);
    out.candidates++;
    let a = xd(tri.x); let b = xd(tri.y); let cc = xd(tri.z);
    let cl = ogcClosestTriangle(x, a, b, cc);
    let cp = cl.bary.x * a + cl.bary.y * b + cl.bary.z * cc;
    let d = length(x - cp);
    if (planar) {
      if (d <= triEnv(tri) + spreadVT(i, tri)) { pairVT(i, tri, x, cp); out.dat++; }
    } else {
      atomicMin(&work[P.dminTOff + t], bitcast<u32>(d));
    }
    if (!(d > 0.0)) { atomicOr(&counters[OGC_STATUS_WORD], OGC_ST_COINCIDENT); }
    else if (d < P.r && vfAccept(OGC_NONE, x, t, tri, cl.feature)) {
      let lambda = -ogcActivation(d, P.r, P.kc).y * xs.w;
      let ent = vec4<u32>(i, t | (cl.feature << 28u), bitcast<u32>(lambda), 1u);
      // A moving solid's sample marks its face's vertices as touching it (bit 4).
      let mark = select(0u, 16u, i >= P.N + P.S);
      if (cl.feature < 3u) { pushSampleEntry(tri[cl.feature], ent, mark); }
      else if (cl.feature < OGC_FACE) { let li = cl.feature - 3u; pushSampleEntry(tri[li], ent, mark); pushSampleEntry(tri[(li + 1u) % 3u], ent, mark); }
      else { pushSampleEntry(tri.x, ent, mark); pushSampleEntry(tri.y, ent, mark); pushSampleEntry(tri.z, ent, mark); }
      out.pairs++;
    }
  }
  } } }
  return out;
}

// Algorithm 2 for edge e: every other edge is reached through its face0's grid entries,
// once. A pair whose closest points are inside both edges is an edge-edge contact; the
// endpoint cases are vertex-edge contacts, which Algorithm 1 already holds.
fn edgeEdge(e: u32) -> RoleOut {
  var out = RoleOut(0u, 0u, 0u);
  let planar = P.mode == OGC_PLANAR;
  let er = edgeRec(e);
  let p0 = xd(er.x); let p1 = xd(er.y);
  let re = max(envOf(er.x), envOf(er.y));
  let p01 = p0 + predOf(er.x); let p11 = p1 + predOf(er.y);
  let q = OgcBox(min(min(p0, p1), min(p01, p11)), max(max(p0, p1), max(p01, p11)));
  let inv = invCell();
  let c0 = cellOf(q.lo, inv); let c1 = cellOf(q.hi, inv);
  var dmin = 3.4e38;
  for (var cz = c0.z; cz <= c1.z; cz++) { for (var cy = c0.y; cy <= c1.y; cy++) { for (var cx = c0.x; cx <= c1.x; cx++) {
    let c = vec3<i32>(cx, cy, cz);
    let key = cellKey(c); let h = cellHash(c);
    let end = min(bucketStart(h + 1u), P.gridCap);
    for (var j = bucketStart(h); j < end; j++) {
      if (entryKey(j) != key) { continue; }
      let t = entryFace(j);
      let box = entryBox(j);
      if (!boxesOverlap(q, box) || any(cellOf(max(q.lo, box.lo), inv) != c)) { continue; }
      let te = topo[P.triEdgesOff + 4u * t + 3u];
      for (var k = 0u; k < 3u; k++) {
        if ((te & (1u << k)) == 0u) { continue; }
        let e2 = topo[P.triEdgesOff + 4u * t + k];
        let r2 = edgeRec(e2);
        if (r2.x == er.x || r2.x == er.y || r2.y == er.x || r2.y == er.y) { continue; }
        out.candidates++;
        let q0 = xd(r2.x); let q1 = xd(r2.y);
        // Canonical order, so both edges' threads compute bitwise the same pair.
        var st: vec3<f32>;
        if (e < e2) { st = ogcSegmentSegment(p0, p1, q0, q1); }
        else { let w = ogcSegmentSegment(q0, q1, p0, p1); st = vec3<f32>(w.y, w.x, w.z); }
        dmin = min(dmin, st.z);
        if (planar && e < e2 && st.z <= max(re, max(envOf(r2.x), envOf(r2.y))) + spreadEE(er, r2)) {
          pairEE(er, r2, p0 + st.x * (p1 - p0), q0 + st.y * (q1 - q0));
          out.dat++;
        }
        if (!(st.z > 0.0)) { atomicOr(&counters[OGC_STATUS_WORD], OGC_ST_COINCIDENT); }
        else if (st.z < P.r && st.x > 0.0 && st.x < 1.0 && st.y > 0.0 && st.y < 1.0) {
          let lambda = -ogcActivation(st.z, P.r, P.kc).y;
          let ent = vec4<u32>(e, e2 | OGC_EE_BIT, bitcast<u32>(lambda), pack2x16unorm(st.xy));
          pushEntry(er.x, ent); pushEntry(er.y, ent);
          out.pairs++;
        }
      }
    }
  } } }
  atomicStore(&work[P.dminEOff + e], bitcast<u32>(dmin));
  return out;
}

// A hull edge of the samples (solid-field.ts clothSolidSampleEdges; the static solid's,
// then the moving solid's) against the cloth's edges, reached through their faces' grid
// entries as a sample is: a DAT pair with each cloth edge within its envelope, so no
// cloth edge passes between two samples. (Pairs only: the faces' barrier against the
// samples holds the cloth off; planar truncation only.)
fn hullEdgeAt(k: u32) -> vec2<u32> {
  var w = 4u * P.edgeBase + 2u * k;
  var off = P.N;
  if (k >= P.SE) { w = 4u * P.bodyEdgeBase + 2u * (k - P.SE); off = P.N + P.S; }
  return vec2<u32>(atomicLoad(&contacts[w]), atomicLoad(&contacts[w + 1u])) + vec2<u32>(off);
}
fn hullEdge(k: u32) -> RoleOut {
  var out = RoleOut(0u, 0u, 0u);
  if (P.mode != OGC_PLANAR) { return out; }
  let hv = hullEdgeAt(k);
  let hr = vec4<u32>(hv, 0u, 0u);
  let p0 = xd(hv.x); let p1 = xd(hv.y);
  let p01 = p0 + predOf(hv.x); let p11 = p1 + predOf(hv.y);
  let q = OgcBox(min(min(p0, p1), min(p01, p11)), max(max(p0, p1), max(p01, p11)));
  let inv = invCell();
  let c0 = cellOf(q.lo, inv); let c1 = cellOf(q.hi, inv);
  for (var cz = c0.z; cz <= c1.z; cz++) { for (var cy = c0.y; cy <= c1.y; cy++) { for (var cx = c0.x; cx <= c1.x; cx++) {
    let c = vec3<i32>(cx, cy, cz);
    let key = cellKey(c); let h = cellHash(c);
    let end = min(bucketStart(h + 1u), P.gridCap);
    for (var j = bucketStart(h); j < end; j++) {
      if (entryKey(j) != key) { continue; }
      let t = entryFace(j);
      let box = entryBox(j);
      if (!boxesOverlap(q, box) || any(cellOf(max(q.lo, box.lo), inv) != c)) { continue; }
      let te = topo[P.triEdgesOff + 4u * t + 3u];
      for (var m = 0u; m < 3u; m++) {
        if ((te & (1u << m)) == 0u) { continue; }
        let r2 = edgeRec(topo[P.triEdgesOff + 4u * t + m]);
        out.candidates++;
        let q0 = xd(r2.x); let q1 = xd(r2.y);
        let st = ogcSegmentSegment(q0, q1, p0, p1);
        if (st.z <= max(envOf(r2.x), envOf(r2.y)) + spreadEE(r2, hr)) {
          pairEE(r2, hr, q0 + st.x * (q1 - q0), p0 + st.y * (p1 - p0));
          out.dat++;
        }
      }
    }
  } } }
  return out;
}

var<workgroup> wgCand: atomic<u32>;
var<workgroup> wgVf: atomic<u32>;
var<workgroup> wgEe: atomic<u32>;
var<workgroup> wgDat: atomic<u32>;
var<workgroup> wgSf: atomic<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let roleLo = ${lo}; let roleHi = ${hi};
  if (64u * wg.x + 64u <= roleLo || 64u * wg.x >= roleHi) { return; }
  let i = gid.x;
  if (i >= roleLo && i < roleHi) {
    ${role === "queryVertex" ? `let o = vertexFacet(i);
    atomicAdd(&wgCand, o.candidates); atomicAdd(&wgVf, o.pairs); atomicAdd(&wgDat, o.dat);`
    : role === "queryEdge" ? `let o = edgeEdge(i - P.N);
    atomicAdd(&wgCand, o.candidates); atomicAdd(&wgEe, o.pairs); atomicAdd(&wgDat, o.dat);`
    : `if (i < P.N + P.E + P.S + P.SB) {
      let o = sampleFacet(i - P.E);
      atomicAdd(&wgCand, o.candidates); atomicAdd(&wgSf, o.pairs); atomicAdd(&wgDat, o.dat);
    } else {
      let o = hullEdge(i - P.N - P.E - P.S - P.SB);
      atomicAdd(&wgCand, o.candidates); atomicAdd(&wgDat, o.dat);
    }`}
  }
  closePairs();
  workgroupBarrier();
  if (li == 0u) {
    atomicAdd(&counters[OGC_F_CANDIDATES], atomicLoad(&wgCand));
    atomicAdd(&counters[OGC_F_VF_PAIRS], atomicLoad(&wgVf));
    atomicAdd(&counters[OGC_F_EE_PAIRS], atomicLoad(&wgEe));
    atomicAdd(&counters[OGC_F_DAT_PAIRS], atomicLoad(&wgDat));
    atomicAdd(&counters[OGC_F_SF_PAIRS], atomicLoad(&wgSf));
  }
}
`;
}

/**
 * The per-vertex displacement cap in x_det.w. Isotropic: OGC's
 * b_v = gamma_p min(d_min,v, d^T_min,v, d^E_min,v), each clipped at the vertex's own
 * query radius (eqs. 21-26). Planar: 0.5 gamma_r r_q,v (DAT Alg. 2, line 10), on the
 * residual. Thread 0 sizes the planes dispatch from the round's pair slots. A vertex
 * with any entry (cloth or solid) is held for the rest of the step and the next.
 */
export const OGC_BOUND_WGSL = header([0, 1, 4, 5, 8, 9, 12]) + /* wgsl */`
var<workgroup> wgSum: atomic<u32>;
var<workgroup> wgMax: atomic<u32>;
fn dminAt(i: u32) -> f32 { return bitcast<f32>(atomicLoad(&work[i])); }
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let v = gid.x;
  let planar = P.mode == OGC_PLANAR;
  if (v == 0u) {
    let slots = atomicLoad(&counters[OGC_PAIR_SLOTS]);
    atomicMax(&counters[OGC_PEAK_PAIR_SLOTS], slots);
    atomicMax(&counters[OGC_F_PEAK_PAIR_SLOTS], slots);
    planesArgs[0] = select(0u, min((min(slots, P.pairCap) + 63u) / 64u, 65535u), planar); planesArgs[1] = 1u; planesArgs[2] = 1u;
  }
  if (v < P.N) {
    var cap = 0.5 * P.gammaR * dminAt(P.envOff + v);
    if (!planar) {
      var m = min(dminAt(P.envOff + v), dminAt(v));
      for (var i = topo[P.vtOffOff + v]; i < topo[P.vtOffOff + v + 1u]; i++) { m = min(m, dminAt(P.dminTOff + topo[P.vtListOff + i])); }
      for (var i = topo[P.veOffOff + v]; i < topo[P.veOffOff + v + 1u]; i++) { m = min(m, dminAt(P.dminEOff + topo[P.veListOff + i])); }
      cap = P.gammaP * m;
    }
    let x = xdet[v];
    xdet[v] = vec4<f32>(x.xyz, cap);
    let c = atomicLoad(&contacts[P.countsOff + v]);
    if (c != 0u) { atomicOr(&work[P.heldOff + v], 1u); }
    atomicAdd(&wgSum, min(c, P.contactCap));
    atomicMax(&wgMax, c);
  }
  workgroupBarrier();
  if (li == 0u) {
    atomicAdd(&counters[OGC_F_PAIR_ENTRIES], atomicLoad(&wgSum));
    let m = atomicLoad(&wgMax);
    atomicMax(&counters[OGC_F_PAIR_MAX], m);
    atomicMax(&counters[OGC_PEAK_CONTACTS], m);
  }
}
`;

// ------------------------------------------------------------------ triangle-triangle intersection

/**
 * Pairs of faces that share no vertex and intersect, counted once each. The
 * predicates are exact orientation tests evaluated in f32 on the current
 * positions; offset contact keeps every pair apart by a fraction of r, far
 * above their rounding.
 */
export const OGC_INTERSECT_WGSL = header([0, 1, 4, 7, 8, 9]) + TOPO_FNS + CELL_FNS + GRID_READ_FNS + triBoxFns(true) + SOLID_ACCESS + OGC_SOLID_WGSL + /* wgsl */`
// The face reaches inside a solid: phi < 0 at a point of a barycentric lattice over it
// with spacing at most a quarter cell (the field is trilinear per cell). The face is in
// the field's frame (the moving solid's body frame at its end pose).
fn insideSolid(h: OgcSolidHeader, A: array<vec3<f32>, 3>) -> bool {
  let longest = max(length(A[1] - A[0]), max(length(A[2] - A[1]), length(A[0] - A[2])));
  let n = u32(clamp(ceil(4.0 * longest / h.cell), 1.0, 32.0));
  let inv = 1.0 / f32(n);
  for (var i = 0u; i <= n; i++) { for (var j = 0u; j <= n - i; j++) {
    let u = f32(i) * inv; let w = f32(j) * inv;
    let p = (1.0 - u - w) * A[0] + u * A[1] + w * A[2];
    let s = ogcSolidSample(h, p);
    if (s.stored && s.phi < 0.0) { return true; }
  } }
  return false;
}
// Orientation with its forward error bound: f32 differences of f32 inputs are exact to
// a relative 2^-24, so |error| <= 16 u perm with perm the permanent of the differences.
fn orient(a: vec3<f32>, b: vec3<f32>, c: vec3<f32>, d: vec3<f32>) -> vec2<f32> {
  let ba = b - a; let ca = c - a; let da = d - a;
  let o = dot(ba, cross(ca, da));
  let perm = dot(abs(ba), abs(ca.yzx * da.zxy) + abs(ca.zxy * da.yzx));
  return vec2<f32>(o, 9.5367e-7 * perm);
}
fn certain(o: vec2<f32>) -> bool { return abs(o.x) > o.y; }
// 0: no crossing, 1: the segment crosses the triangle, 2: undecided in f32.
fn segTri(p: vec3<f32>, q: vec3<f32>, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> u32 {
  let o1 = orient(a, b, c, p); let o2 = orient(a, b, c, q);
  if (certain(o1) && certain(o2) && o1.x * o2.x > 0.0) { return 0u; }
  let s1 = orient(p, q, a, b); let s2 = orient(p, q, b, c); let s3 = orient(p, q, c, a);
  let sure = certain(o1) && certain(o2) && certain(s1) && certain(s2) && certain(s3);
  if (sure) { return select(0u, 1u, (s1.x > 0.0 && s2.x > 0.0 && s3.x > 0.0) || (s1.x < 0.0 && s2.x < 0.0 && s3.x < 0.0)); }
  // Undecided signs: only a crossing whose signs could all agree needs the distance test.
  let pos = (s1.x >= -s1.y) && (s2.x >= -s2.y) && (s3.x >= -s3.y);
  let neg = (s1.x <= s1.y) && (s2.x <= s2.y) && (s3.x <= s3.y);
  return select(0u, 2u, pos || neg);
}
fn triDistance(A: array<vec3<f32>, 3>, B: array<vec3<f32>, 3>) -> f32 {
  var d = 3.4e38;
  for (var k = 0u; k < 3u; k++) {
    let ca = ogcClosestTriangle(A[k], B[0], B[1], B[2]).bary;
    d = min(d, length(A[k] - (ca.x * B[0] + ca.y * B[1] + ca.z * B[2])));
    let cb = ogcClosestTriangle(B[k], A[0], A[1], A[2]).bary;
    d = min(d, length(B[k] - (cb.x * A[0] + cb.y * A[1] + cb.z * A[2])));
    for (var l = 0u; l < 3u; l++) { d = min(d, ogcSegmentSegment(A[k], A[(k + 1u) % 3u], B[l], B[(l + 1u) % 3u]).z); }
  }
  return d;
}
// Exact where the f32 signs are certain; an undecided (near-degenerate) pair intersects
// only if the triangles lie within the f32 resolution of their coordinates of each other.
fn triTri(Av: array<vec3<f32>, 3>, Bv: array<vec3<f32>, 3>) -> bool {
  var A = Av; var B = Bv;
  var undecided = false;
  var scale = 0.0;
  for (var k = 0u; k < 3u; k++) {
    let ra = segTri(A[k], A[(k + 1u) % 3u], B[0], B[1], B[2]);
    let rb = segTri(B[k], B[(k + 1u) % 3u], A[0], A[1], A[2]);
    if (ra == 1u || rb == 1u) { return true; }
    undecided = undecided || ra == 2u || rb == 2u;
    scale = max(scale, max(max(abs(A[k].x), abs(A[k].y)), max(max(abs(A[k].z), abs(B[k].x)), max(abs(B[k].y), abs(B[k].z)))));
  }
  return undecided && triDistance(A, B) <= 7.6294e-6 * scale;
}
var<workgroup> wgHits: atomic<u32>;
var<workgroup> wgSolidHits: atomic<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let t = gid.x;
  if (t < P.T) {
    let tv = triVerts(t);
    let A = array<vec3<f32>, 3>(xdet[tv.x].xyz, xdet[tv.y].xyz, xdet[tv.z].xyz);
    if (P.solidBase != 0u && insideSolid(ogcSolidHeader(), A)) { atomicAdd(&wgSolidHits, 1u); }
    else if (P.bodyBase != 0u) {
      let p1 = ogcPose(1u);
      if (insideSolid(ogcSolidHeaderAt(P.bodyBase), array<vec3<f32>, 3>(ogcToBody(p1, A[0]), ogcToBody(p1, A[1]), ogcToBody(p1, A[2])))) { atomicAdd(&wgSolidHits, 1u); }
    }
    let q = triBox(t);
    let inv = invCell();
    let c0 = cellOf(q.lo, inv); let c1 = cellOf(q.hi, inv);
    var hits = 0u;
    for (var cz = c0.z; cz <= c1.z; cz++) { for (var cy = c0.y; cy <= c1.y; cy++) { for (var cx = c0.x; cx <= c1.x; cx++) {
      let c = vec3<i32>(cx, cy, cz);
      let key = cellKey(c); let h = cellHash(c);
      let end = min(bucketStart(h + 1u), P.gridCap);
      for (var j = bucketStart(h); j < end; j++) {
        if (entryKey(j) != key) { continue; }
        let t2 = entryFace(j);
        if (t2 <= t) { continue; }
        let uv = triVerts(t2);
        if (hasVertex(uv, tv.x) || hasVertex(uv, tv.y) || hasVertex(uv, tv.z)) { continue; }
        let box = entryBox(j);
        if (!boxesOverlap(q, box) || any(cellOf(max(q.lo, box.lo), inv) != c)) { continue; }
        let B = array<vec3<f32>, 3>(xdet[uv.x].xyz, xdet[uv.y].xyz, xdet[uv.z].xyz);
        if (triTri(A, B)) { hits++; }
      }
    } } }
    atomicAdd(&wgHits, hits);
  }
  workgroupBarrier();
  if (li == 0u) {
    let h = atomicLoad(&wgHits); if (h != 0u) { atomicAdd(&counters[OGC_INTERSECTIONS], h); }
    let sh = atomicLoad(&wgSolidHits); if (sh != 0u) { atomicAdd(&counters[OGC_SOLID_INTERSECTIONS], sh); }
  }
}
`;
