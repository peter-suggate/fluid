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
 * terms (q.xyz, marker) and (D.xyz, 0) of the round (OgcTerms), and at heldOff the held flags (bit 0: contact, a plane or solid cut, or a
 * fail-safe conflict this step; bit 1: unused; bit 2: carried with the moving solid this step, in a
 * fail-safe impact zone; bit 3: held by a fail-safe conflict this step; bit 4: touched the moving solid this step (a field
 * bound whose ball reaches r, a cut, or a sample entry); bit 5: its least shift anchored at a_v (moving with the solid);
 * bit 6: moved with the solid this step (w = a_v, set by prep); bit 7: a conflict with the solid asked for the anchor a_v;
 * bit 8: touching at the end of the last step; bit 9: its contact onset (OGC_CONTACT_STATE_WGSL)), and at scaleOff
 * the least shift's words of the mandatory round's redos (OGC_SHIFT_WORDS), and at originOff
 * the origin (x - X, 1) about which the line-search filter last re-took the vertex's solid
 * bounds (0: the round's own, about X and a_v; every round clears it), and at recCountOff each
 * vertex's plane-record count of the filter (OGC_FILTER_ALLOC_WGSL; 0 between filters).
 * With a Newton solver the contact buffer holds the filter's records for ogcProject: at vec4
 * header word 11 the sub-header (fixed records' vec4, segments' word, records' vec4, capacity),
 * each vertex's OGC_PROJECT_RECORD vec4s at fixedWord, its segment (first, count) at segWord
 * and the plane records (m, b) at recWord, each vertex's contiguous.
 * The solid image sits in the contact buffer at word solidBase.
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
import { OGC_COMMON_WGSL, OGC_COUNTERS_WGSL, OGC_POSE_WORD, OGC_POSE_WORDS, OGC_PROJECT_RECORD, OGC_SOLID_WGSL, ogcVertexBlockWGSL } from "./ogc-wgsl";

export const OGC_PARAM_WORDS = 88;
/**
 * The redos of a step's mandatory round (OGC_REDO_WGSL). A conflicting pair moves its cloth's
 * predictions only as far as it needs (least shift, resolvePlane): towards X (scale s_v), or with
 * the moving solid in the pair towards moving with it (carry k_v, the relative motion scaled in the
 * solid's frame). Each vertex keeps the least of its pairs' factors, which only fall, so the cloth
 * keeps its own smooth prediction everywhere else. Up to OGC_REDO_ROUNDS such redos; a conflict
 * past them, or one no shift can resolve, takes the fail-safe (reported as failSafe, never the
 * normal path): the round redone with every prediction held (Planar-DAT, which never conflicts
 * without a moving solid), and with one Bridson et al.'s rigid impact zones (2002, sec. 7.4):
 * a conflict with the solid carries its cloth rigidly with it (w = a_v), growing the zones a redo,
 * up to OGC_REDO_MANDATORY redos in all. A zone redo that marks nothing new, or a conflict in the
 * last, fails as heldConflict. A later round's unlevelled form predicts 0 about the iterate,
 * every vertex and every sample exactly (the moving solid is at its end pose, a_v = 0), so it
 * never conflicts: one redo slot, and a conflict left after it fails as heldConflict.
 */
export const OGC_REDO_ROUNDS = 8;
export const OGC_REDO_MANDATORY = 32;
/** Per vertex at scaleOff, the least shift's words: the next redo's scale (atomicMin) and this round's. */
export const OGC_SHIFT_WORDS = 2;
const SHIFT_CONSTS = /* wgsl */`
const OGC_SHIFT_WORDS = ${OGC_SHIFT_WORDS}u;
`;
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
  SB: u32, wgPrep: u32, bodyReach: f32, facetBase: u32,
  edgeBase: u32, SE: u32, bodyEdgeBase: u32, SEB: u32,
  solidDevOff: u32, bodyDevOff: u32, scaleOff: u32, originOff: u32,
  fixedWord: u32, segWord: u32, recWord: u32, recCap: u32,
  recCountOff: u32, newton: u32, levelWord: u32, levelOn: u32,
  targetOff: u32, rqTop: f32, SF: u32, bodyFacetBase: u32,
  SFB: u32, facetCap: u32, pad11: u32, pad12: u32,
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
  bodyBase: 58, bodyOff: 59, SB: 60, wgPrep: 61, bodyReach: 62, facetBase: 63,
  edgeBase: 64, SE: 65, bodyEdgeBase: 66, SEB: 67, solidDevOff: 68, bodyDevOff: 69, scaleOff: 70, originOff: 71,
  fixedWord: 72, segWord: 73, recWord: 74, recCap: 75, recCountOff: 76, newton: 77, levelWord: 78, levelOn: 79, targetOff: 80, rqTop: 81,
  SF: 82, bodyFacetBase: 83, SFB: 84, facetCap: 85,
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
  /** The Newton solver's control record (vec4: done, PCG done, the accepted step alpha, the step's Newton iterations). */
  19: "@group(0) @binding(19) var<storage, read> nwctl: array<vec4<f32>>;",
  /** The end-of-step verification's indirect args (OGC_VERIFY_ARGS_WGSL writes them; the verify pass is dispatched from them). */
  20: "@group(0) @binding(20) var<storage, read_write> vargs: array<u32>;",
};
/** The decision writes the Newton control's done word (the end-of-step verification's verdict). */
const DECL_CTL_RW = "@group(0) @binding(19) var<storage, read_write> nwctl: array<vec4<f32>>;";
/** Truncation writes the solver's positions; prep writes the predictions. */
const DECL_POS_RW = "@group(0) @binding(2) var<storage, read_write> pos: array<vec4<f32>>;";
const DECL_PRED_RW = "@group(0) @binding(14) var<storage, read_write> pred: array<vec4<f32>>;";

/** The kernel's declarations; `rw` names the bindings among 2, 14 and 19 it writes. */
function header(bindings: readonly number[], rw: readonly number[] = []): string {
  return [PARAMS_STRUCT, OGC_COUNTERS_WGSL, OGC_COMMON_WGSL,
    ...bindings.map(b => rw.includes(b) ? (b === 2 ? DECL_POS_RW : b === 19 ? DECL_CTL_RW : DECL_PRED_RW) : DECL[b])].join("\n");
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
const ROUND_LOG_FNS = /* wgsl */`
// The current round's log entry word (0xffffffff before the frame's first round or past the log).
fn roundLogWord() -> u32 {
  let n = atomicLoad(&counters[OGC_ROUND_LOG_COUNT]);
  if (n == 0u || n > OGC_ROUND_LOG_N) { return 0xffffffffu; }
  return OGC_ROUND_LOG_BASE + OGC_ROUND_LOG_W * (n - 1u);
}
`;

// The level header (OGC_LEVEL) in the contact buffer at word P.levelWord (0: no newton config).
const LEVEL_FNS = /* wgsl */`
fn levelFlags() -> u32 { if (P.levelWord == 0u) { return 0u; } return atomicLoad(&contacts[P.levelWord + 1u]); }
fn levelled() -> bool { return (levelFlags() & OGC_LEVELLED) != 0u; }
fn levelU() -> f32 { return select(0.0, bitcast<f32>(atomicLoad(&contacts[P.levelWord])), levelled()); }
`;

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
fn redoCap() -> u32 { return select(${OGC_REDO_ROUNDS + 1}u, ${OGC_REDO_MANDATORY}u, P.bodyBase != 0u); }
// The fail-safe: every prediction held (and, with a moving solid, its impact zones carried).
fn failSafeRound() -> bool { return atomicLoad(&counters[OGC_FORCE_HELD]) > ${OGC_REDO_ROUNDS}u; }
`;
const BODY_ACCESS_FNS = /* wgsl */`
fn withSolid(i: u32) -> bool { return (atomicLoad(&work[P.heldOff + i]) & 64u) != 0u; }
fn sigmaOf(i: u32) -> f32 { return bitcast<f32>(atomicLoad(&work[P.bodyOff + 8u * i + 7u])); }`;
/** `access`: withSolid and sigmaOf (BODY_ACCESS_FNS, or the locality switch's round geometry records). */
const bodyPairFns = (access: string) => REDO_CAP_FN + access + /* wgsl */`
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
const BODY_PAIR_FNS = bodyPairFns(BODY_ACCESS_FNS);

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

// ------------------------------------------------------------------ locality switch (cloth-m1-max-locality, Experiment 2)

/**
 * The query kernels' locality switch (OgcContactConfig.locality; default off, the kernels above unchanged). Each part
 * replaces loads of the query roles and nothing else: the grid, the traversal, the tests and their order are the same,
 * and every value read is the same bits, from another place.
 *   topology  compiled face records (ogc-topology.ts buildOgcFaceRecords) in `loc`, 8 words a face: a box-passing
 *             record's owned edges, their endpoints (in the edge table's order) and its vertices' ownership come from
 *             one 8-word load in place of triEdges -> edge record and vtOff -> vtList.
 *   geometry  round geometry records in `geo`, two vec4<u32> per vertex and solid sample, written after prep by
 *             OGC_LOC_GEOMETRY_WGSL: (X.xyz, envelope), (w.xyz, sigma | moving-with-solid << 31), in place of
 *             x_det, pred and three regions of work.
 * (A third part, the round's final bucket offsets in place of bucketStart's three prefix-table loads, was measured
 * and removed: no gain alone or on top; docs/plans/cloth-m1-max-locality-progress-l2.md.)
 */
export const OGC_LOCALITY_PARTS = ["topology", "geometry"] as const;
/**
 * The locality of a config that names none: both parts (main's ruling on Experiment 2: the geometry records cut the
 * query time 29% at the 150^2 sphere's ticks 35 and 120, the face records about 5% more). "off" stays the A/B baseline.
 */
export const OGC_LOCALITY_DEFAULT = "topology+geometry";
export type OgcLocalityPart = (typeof OGC_LOCALITY_PARTS)[number];
export type OgcLocalityFlags = Readonly<Record<OgcLocalityPart, boolean>>;
/** "off", or parts joined by "+" (e.g. "topology+geometry"). Anything else fails. */
export function parseOgcLocality(value: unknown): OgcLocalityFlags {
  const flags = { topology: false, geometry: false };
  if (value === undefined || value === "off") return flags;
  if (typeof value !== "string" || !value) throw new Error(`OGC: unknown locality "${String(value)}"`);
  for (const part of value.split("+")) {
    if (!(OGC_LOCALITY_PARTS as readonly string[]).includes(part) || flags[part as OgcLocalityPart])
      throw new Error(`OGC: unknown locality "${value}" (off, or ${OGC_LOCALITY_PARTS.join(", ")} joined by +)`);
    flags[part as OgcLocalityPart] = true;
  }
  return flags;
}
/** The bindings a query role uses under the switch (exactly: the pipelines' layouts are automatic). */
export function ogcQueryBindings(L: OgcLocalityFlags): number[] {
  return [0, 1, 4, 5, 7, 8, 9, 11, 14, ...(L.topology ? [24] : []), ...(L.geometry ? [25] : [])];
}
const DECL_LOC = "\n@group(0) @binding(24) var<storage, read> loc: array<u32>;\n";
const DECL_GEO = "\n@group(0) @binding(25) var<storage, read> geo: array<vec4<u32>>;\n";
const DECL_GEO_RW = "\n@group(0) @binding(25) var<storage, read_write> geo: array<vec4<u32>>;\n";
const LOC_TOPO_FNS = /* wgsl */`
const OGC_FACE_RECORD_WORDS = 8u;
struct OgcFaceRec { v: vec3<u32>, flags: u32, e: vec3<u32> }
fn faceRecBase(t: u32) -> u32 { return OGC_FACE_RECORD_WORDS * t; }
fn faceRec(t: u32) -> OgcFaceRec {
  let b = faceRecBase(t);
  return OgcFaceRec(vec3<u32>(loc[b], loc[b + 1u], loc[b + 2u]), loc[b + 3u], vec3<u32>(loc[b + 4u], loc[b + 5u], loc[b + 6u]));
}
fn faceVerts(t: u32) -> vec3<u32> { let b = faceRecBase(t); return vec3<u32>(loc[b], loc[b + 1u], loc[b + 2u]); }
// Edge k of the face as its edge record's (v0, v1), in the edge table's order; the record's faces (zw) are not carried.
fn faceEdge(f: OgcFaceRec, k: u32) -> vec4<u32> {
  let a = f.v[k]; let b = f.v[(k + 1u) % 3u];
  let rev = (f.flags & (8u << k)) != 0u;
  return vec4<u32>(select(a, b, rev), select(b, a, rev), OGC_NONE, OGC_NONE);
}
`;
/** The query's per-vertex reads from the round geometry records (in place of xd, PRED_FNS and triBoxFns' envelopes). */
const LOC_GEO_FNS = /* wgsl */`
fn xd(i: u32) -> vec3<f32> { return bitcast<vec3<f32>>(geo[2u * i].xyz); }
fn envOf(i: u32) -> f32 { return bitcast<f32>(geo[2u * i].w); }
fn triEnv(v: vec3<u32>) -> f32 { return max(envOf(v.x), max(envOf(v.y), envOf(v.z))); }
fn predOf(i: u32) -> vec3<f32> { return bitcast<vec3<f32>>(geo[2u * i + 1u].xyz); }
fn anchored(i: u32) -> vec3<f32> { return xd(i) + predOf(i); }
fn invCell() -> f32 { return bitcast<f32>(atomicLoad(&counters[OGC_INV_CELL])); }
// The least-shift bisection (shiftGap) and resolvePlane's support projection keep PRED_FNS' reads of the live buffers,
// text for text: the compiler contracts x + a + t (w - a), and folds (x + w) - o against the live o, differently when
// x comes from the record, and a last-bit difference there moves a shift or flips a conflict test at a tie.
fn predLive(i: u32) -> vec3<f32> {
  if (i < P.N) { return pred[i].xyz; }
  if (i < P.N + P.S) { return vec3<f32>(0.0); }
  return pred[P.N + i - P.S].xyz;
}
fn anchoredLive(i: u32) -> vec3<f32> { return xdet[i].xyz + predLive(i); }
`;
const LOC_BODY_ACCESS_FNS = /* wgsl */`
fn withSolid(i: u32) -> bool { return (geo[2u * i + 1u].w & 0x80000000u) != 0u; }
fn sigmaOf(i: u32) -> f32 { return bitcast<f32>(geo[2u * i + 1u].w & 0x7fffffffu); }`;

/**
 * locality "geometry": the round's geometry record of every vertex and solid sample, after prep and on prep's
 * indirect slot (its threads stride over the ids, so the static samples are covered whatever the slot's size).
 * Copies, bit for bit: x_det, the envelope, the prediction (0 for a static sample, the exact one of a moving solid's)
 * and, with a moving solid, sigma_v with held bit 6 in its sign bit (sigma_v >= 0; a non-finite one already fails
 * the step as nonFinite). Nothing between prep and the query writes any of them.
 */
export const OGC_LOC_GEOMETRY_WGSL = header([0, 4, 5, 14]) + DECL_GEO_RW + /* wgsl */`
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let total = P.N + P.S + P.SB;
  for (var i = gid.x; i < total; i += 64u * nwg.x) {
    var g0 = vec4<u32>(bitcast<vec3<u32>>(xdet[i].xyz), 0u);
    var g1 = vec4<u32>(0u);
    if (i < P.N) {
      g0.w = atomicLoad(&work[P.envOff + i]);
      var tail = 0u;
      if (P.bodyBase != 0u) {
        tail = (atomicLoad(&work[P.bodyOff + 8u * i + 7u]) & 0x7fffffffu) | ((atomicLoad(&work[P.heldOff + i]) & 64u) << 25u);
      }
      g1 = vec4<u32>(bitcast<vec3<u32>>(pred[i].xyz), tail);
    } else if (i >= P.N + P.S) {
      g1 = vec4<u32>(bitcast<vec3<u32>>(pred[P.N + i - P.S].xyz), 0u);
    }
    geo[2u * i] = g0; geo[2u * i + 1u] = g1;
  }
}
`;

/** Envelope of vertex i (detection) and the face box inflated by its vertices' widest envelope. */
/** The face box helpers; `envelopes` adds the per-vertex envelope reads (the work buffer, binding 5). */
function triBoxFns(intersect: boolean, envelopes = !intersect): string {
  return (intersect ? "" : PRED_FNS) + (envelopes ? /* wgsl */`
fn envOf(i: u32) -> f32 { return bitcast<f32>(atomicLoad(&work[P.envOff + i])); }
fn triEnv(v: vec3<u32>) -> f32 { return max(envOf(v.x), max(envOf(v.y), envOf(v.z))); }
` : "") + /* wgsl */`
// Local swept bounds: a face's grid box is grown by its own envelope (its vertices' widest) and
// a querying primitive's swept box by its own (0 for a solid's sample or hull edge, whose motion
// is exact), so two boxes meet whenever the primitives' swept paths come within the sum of the
// two envelopes. That covers the larger of the two for every pair the round keeps at X, and a pair
// whose boxes do not meet stays apart over the round: each vertex ends within its cap 0.45 r_q of
// its path, and two caps close at most 0.45 of the sum, leaving 0.55 (r_q,a + r_q,b) >= r.
// (Growing every face by the round's largest envelope, a few capped vertices' 4 cm, set every box
// and the cell: 85 candidates per primitive a round on the rotating sphere.)
fn faceEnv(v: vec3<u32>) -> f32 { ${intersect ? "return 0.0;" : "return triEnv(v);"} }
// The face's box over the round: from X to the predicted X + w, grown by its own envelope.
fn triBox(t: u32) -> OgcBox {
  let v = triVerts(t);
  let a = xdet[v.x].xyz; let b = xdet[v.y].xyz; let c = xdet[v.z].xyz;
  var lo = min(a, min(b, c)); var hi = max(a, max(b, c));
  ${intersect ? "" : `let a1 = a + predOf(v.x); let b1 = b + predOf(v.y); let c1 = c + predOf(v.z);
  lo = min(lo, min(a1, min(b1, c1))); hi = max(hi, max(a1, max(b1, c1)));`}
  let e = faceEnv(v);
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

const CUT_FNS = /* wgsl */`
fn work4(b: u32) -> vec4<f32> {
  return bitcast<vec4<f32>>(vec4<u32>(atomicLoad(&work[b]), atomicLoad(&work[b + 1u]), atomicLoad(&work[b + 2u]), atomicLoad(&work[b + 3u])));
}
fn putWork4(i: u32, q: vec4<f32>) {
  atomicStore(&work[i], bitcast<u32>(q.x)); atomicStore(&work[i + 1u], bitcast<u32>(q.y));
  atomicStore(&work[i + 2u], bitcast<u32>(q.z)); atomicStore(&work[i + 3u], bitcast<u32>(q.w));
}
// The truncation terms of a ball bound (boundTerms, below), without its entry.
struct OgcTerms { q: vec4<f32>, D: vec3<f32>, rl: f32 }
fn termsOf(h: OgcSolidHeader, sb: OgcSolidBound, rc: f32, margin: f32) -> OgcTerms {
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
  return OgcTerms(vec4<f32>(sb.g0 * s, rc), sb.D * s, room / h.lmax);
}
`;

/**
 * Each vertex's x_det, prediction w, envelope and, with a static solid, its solid
 * terms and entry. The step's mandatory round detects at the step start x_t (the
 * solver has already placed its initial guess, which the round's truncation then
 * cuts); a re-detection round commits the current, truncated iterate.
 *
 * In the mandatory round a free vertex is predicted to move its inertial v dt + g dt^2, the
 * solver's initial guess less x_t, or the solver's inertial target y less x_t (prediction
 * "target": exact in free flight, whatever the guess), and its envelope covers only the
 * residual motion: r + s rho, with rho the last step's deviation from its own prediction.
 * A re-detection round predicts 0 about the iterate x, held or not, and its envelope covers
 * the motion left from x: at most |yref - x| + rho (yref the step's predicted end, rho the
 * estimate of the end's deviation from it), and with a Newton solver the smaller of that and
 * |d_v|, the solver's last direction, its own estimate of the way left. Neither the step's whole
 * prediction from x_t (|v dt + g dt^2| + rho, the held envelope) nor |yref - x| + rho alone fits
 * an iterate that has already made most of its deviation: on the rotating sphere (frozen
 * 7809cbe1, ticks 4-15) the motion after a step's last round is 1-200 um a vertex on average
 * while those envelopes budget 0.8-6 mm, and the held one alone set every re-detection round's
 * largest envelope (39 mm against 7 mm unheld at tick 5) and with it the cell. An envelope
 * that falls short costs a truncation and a round, never a certificate. A vertex in contact keeps its prediction too: a conflict
 * shifts only its pair's predictions, towards X or towards moving with the moving solid,
 * as far as the pair needs (least shift, resolvePlane), and the envelope grows by the shift.
 * A held vertex (not yet measured, every vertex of the fail-safe round, of isotropic mode,
 * or without predictions) is predicted 0 and its envelope covers the inertial displacement and the residual,
 * r + s (|v dt + g dt^2| + rho): Planar-DAT itself. An envelope without rho caps a held
 * vertex whose solve deviates from its prediction by more than 0.5 gamma_r rqBase, which
 * holds it again (every vertex held and every round at the cap). A prediction into the static solid stops where the solid's
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
 * as the static solid scales one towards 0. In the fail-safe a vertex a conflict marked
 * (bit 2) is carried in its step's rounds redone: an impact zone moving with the solid
 * (Bridson et al. 2002, sec. 7.4). A prediction the static solid then has to cut out of the moving one's
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
export const OGC_PREP_WGSL = header([0, 2, 3, 4, 5, 8, 9, 13, 14, 15, 17], [14]) + LEVEL_FNS + SHIFT_CONSTS + SOLID_ACCESS + OGC_SOLID_WGSL + CUT_FNS + /* wgsl */`
var<workgroup> wgMaxEnv: atomic<u32>;
var<workgroup> wgMaxPred: atomic<u32>;
var<workgroup> wgCube: array<f32, 64>;
var<workgroup> wgCap: atomic<u32>;
var<workgroup> wgBad: atomic<u32>;
var<workgroup> wgHeld: atomic<u32>;
var<workgroup> wgSolid: atomic<u32>;
var<workgroup> wgBody: atomic<u32>;
var<workgroup> wgSqueeze: atomic<u32>;
// The envelope r + s rho, at least rqBase and at most rqTop (a clamp at rqTop is counted): rqMax, or
// gamma_r times the smallest rest altitude when the envelope is capped to the mesh (OgcContactConfig).
fn envelopeOf(rho: f32) -> f32 {
  let env = max(P.rqBase, P.r + P.envScale * rho);
  if (env > P.rqTop) { atomicAdd(&wgCap, 1u); return P.rqTop; }
  return env;
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
// phi0 <= 0 allows none. Also the entry with mu lambda, lagged as its slip's normal is (IPC):
// lambda the barrier's force and mu the solid's own coefficient at the vertex's step start xs
// (in the field's frame), not at x0. A round's x0 is a Newton iterate, or the prediction a
// cut left close to the face; lambda there exceeded the normal force the step balances, and
// a patch on a 35 degree slope (tan > mu = 0.5) stuck.
fn boundTerms(v: u32, h: OgcSolidHeader, sb: OgcSolidBound, x0: vec3<f32>, y: u32, rc: f32, margin: f32, xs: vec3<f32>) -> OgcTerms {
  let t = termsOf(h, sb, rc, margin);
  if (t.q.w >= 1.0e38) { return t; }
  let st = ogcSolidSample(h, xs);
  let lambda = select(0.0, -ogcActivation(st.phi, P.r, P.kc).y * length(st.grad), st.phi > 0.0 && st.phi < P.r) * ogcSolidMu(h, xs);
  pushSolidEntry(v, y, lambda);
  return t;
}
// A solid's terms for a round that moves the vertex at most rho from x0 (in the field's frame).
// A round the field's slope clears by r needs none, nor one the clearance table clears by r
// (a long motion far from the solid, where the bricks only know the field exceeds band). Otherwise the affine bound is taken over the
// ball rc = min(rho, rqMax / 2), within which an unstored brick still reads clear of r (the band
// holds r + 1.5 L_max rqMax); a ball clear of the solid certifies all of it (q = D = 0). Beyond
// rc only the Lipschitz ball certifies, so a long prediction is never moved for the ball's sake.
// The round's reach is the ball |d| <= rho about x0; it lies in the smaller ball about x0 + c of
// radius cr (rho - |c| when the round moves along a chord, c its half: the start and the predicted
// end within env / 2 of it lie there, and the ball is convex). Clear of the solid there, no terms.
fn roundBound(v: u32, h: OgcSolidHeader, x0: vec3<f32>, rho: f32, y: u32, margin: f32, c: vec3<f32>, cr: f32, xs: vec3<f32>) -> OgcTerms {
  let xc = x0 + c; let rr = cr + margin;
  if (ogcSolidPhiLower(h, xc) - h.lmax * rr >= P.r
    || ogcSolidClearance(h, xc) - rr >= P.r) { return OgcTerms(vec4<f32>(0.0), vec3<f32>(0.0), 0.0); }
  let rc = min(rho, 0.5 * P.rqMax);
  var sb = ogcSolidBallBound(h, x0, rc, P.r);
  if (sb.far) { sb.g0 = vec3<f32>(0.0); sb.D = vec3<f32>(0.0); }
  return boundTerms(v, h, sb, x0, y, rc, margin, xs);
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
// A vertex the moving solid bounds (withA) may end anywhere in the hull of w and a_v (the cut
// below moves it towards a_v): the balls about x0 and about the chord's midpoint w / 2 then cover
// a_v too (each is convex, so the hull).
fn solidRound(v: u32, x0: vec3<f32>, w: vec3<f32>, env: f32, margin: f32, a: vec3<f32>, withA: bool) -> OgcTerms {
  let rho = select(length(w), max(length(w), length(a)), withA) + 0.5 * env;
  let cr = select(0.5 * length(w), max(0.5 * length(w), length(a - 0.5 * w)), withA) + 0.5 * env;
  let h = ogcSolidHeader();
  let rel = (x0 - h.origin) / (8.0 * h.cell);
  if (any(rel < vec3<f32>(0.0)) || any(rel >= vec3<f32>(h.dims))) { atomicOr(&counters[OGC_STATUS_WORD], OGC_ST_SOLID_BOUNDS); }
  let qj = roundBound(v, h, x0, rho, OGC_SOLID_Y, margin, 0.5 * w, cr, xt[v].xyz);
  if (qj.q.w != 0.0 && qj.q.w < 1.0e38) { atomicAdd(&wgSolid, 1u); }
  putWork4(P.solidOff + 4u * v, qj.q);
  putWork4(P.solidDevOff + 4u * v, vec4<f32>(qj.D, qj.rl));
  return qj;
}
// How far the round's prediction can lie from its anchor c before the least shift, given the
// prediction w before the moving solid's bound and a_v: that bound moves it towards a_v, so it
// stays in the hull of {w, a_v}. A shift by s moves it (1 - s) of that, more than the inertial
// motion, which the envelope covers (shifting the prediction does not make the vertex follow
// it any better). The cuts after it (the moving solid's about a_v, the static solid's towards 0)
// are pushes, which the vertex follows, as for an unshifted prediction.
fn shiftReach(w: vec3<f32>, a: vec3<f32>, c: vec3<f32>) -> f32 {
  return max(length(w - c), length(a - c));
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let v = gid.x;
  let mandatory = atomicLoad(&counters[OGC_ROUNDS]) == 0u;
  let redone = atomicLoad(&counters[OGC_FORCE_HELD]);
  // A redone round repeats its round's detection with its conflicts' predictions shifted (least
  // shift) or, in the fail-safe, every prediction held; only the first measures.
  let fresh = mandatory && redone == 0u;
  let failSafe = redone > ${OGC_REDO_ROUNDS}u;
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
    // The least-shift scale of the round (1 in the fresh round; its pairs lower the next) and
    // the anchor it shifts the prediction towards (a_v when anchored with the moving solid).
    var shift = 1.0;
    var anc = vec3<f32>(0.0);
    if (fresh) {
      // Every vertex keeps its own prediction, in contact or not: a conflict shifts only its
      // pair's. Before any step has measured rho (w < 0) it is 0. The envelope bounds only how
      // far the vertex strays from its prediction (the filter keeps every trial sound), so an
      // unmeasured vertex predicts as any other: held, two co-moving layers in free fall paired
      // as static ones, the whole inertial motion in their envelopes (33 plane records per
      // vertex on the ladder's first step).
      let unmeasured = yref.w < 0.0;
      yref = vec4<f32>(x + inertial, select(length(x - yref.xyz), 0.0, unmeasured));
      flags = flags & 768u;
      atomicStore(&work[P.heldOff + v], flags);
      atomicStore(&work[P.scaleOff + OGC_SHIFT_WORDS * v], bitcast<u32>(1.0));
    } else if (mandatory && !failSafe) {
      shift = bitcast<f32>(atomicLoad(&work[P.scaleOff + OGC_SHIFT_WORDS * v]));
      // The anchor its scale shifts towards (bit 5): moving with the moving solid when its ball
      // reached the solid this step (bit 4) or a conflict with the solid asked (bit 7); X else.
      if (P.bodyBase != 0u && (flags & 144u) != 0u) { flags |= 32u; atomicOr(&work[P.heldOff + v], 32u); }
    }
    atomicStore(&work[P.scaleOff + OGC_SHIFT_WORDS * v + 1u], bitcast<u32>(shift));
    // Held: every vertex of the fail-safe round; after the mandatory round, any vertex with a
    // flag (contact or truncation, bit 0).
    let heldBits = select(flags & 15u, 0u, mandatory);
    let held = failSafe || P.predict == 0u || P.mode != OGC_PLANAR || heldBits != 0u;
    let anchored = P.bodyBase != 0u && (flags & 32u) != 0u;
    // Carried with the moving solid (bit 2): in an impact zone a conflict with it formed this step.
    let carried = P.bodyBase != 0u && (flags & 4u) != 0u;
    // The envelope covers the round's residual |x - X - w| <= rho + |inertial - w| (triangle
    // inequality): rho, the last step's deviation from its prediction, plus how far the round's
    // prediction lies from the inertial one (0 when free, the whole motion when held). Holding
    // shifts the prediction; it does not make the vertex follow it any better.
    var w = select(yref.xyz - x, vec3<f32>(0.0), held);
    var dev = select(yref.w, plen + yref.w, held);
    // A re-detection round commits the iterate x as X, so its certificates must hold x itself:
    // |x - X - w| = |w| <= the cap only for w = 0. It predicts 0 (Planar-DAT about the iterate),
    // held or not, and its envelope covers the motion left from x: |yref - x| + rho, or with a
    // Newton solver the smaller of that and its last direction |d_v|. A prediction
    // of the rest would leave the iterate |w| from its ball's centre, and the round's
    // truncation would pull it there outside the solve (and hold it, a cascade).
    // A levelled round (ogcDecideWGSL) is the exception: it predicts the rest of the Newton step
    // it re-detects after, w = (1 - alpha_c) d from the iterate x = x_k + alpha_c d to the full
    // step x_k + d, and its certificates are a family over the level a in [0, 1] anchored at
    // X + a w (ogcProjectTrial), so x = X is certified at level 0 and the solve moves the level
    // with it. Its envelope covers how far the solve strays from that chord, measured as how far
    // the iterate strayed from the last round's: |x - c'(L')|, c' = X' + (1 - U') w' at the level
    // the last round reached. Not the step taken, alpha_c |d|: where the projection cut d (a
    // sheet landing flat on the floor, d 50 mm into it) that is motion the vertex never makes,
    // and its 40 mm envelope paired the 16 x 16 sheet's every vertex with its neighbours in the
    // plane (877 plane records per vertex). Nor the step's rho, the inertial prediction's miss.
    // (Its unlevelled redo predicts 0.)
    let lev = !mandatory && levelled() && redone == 0u;
    if (!mandatory) {
      let d = dir[v].xyz;
      if (lev) {
        // The rest of the step to the full Newton point x_k + d (the filter's target): (1 - alpha_c) d
        // where the projection kept the trial, and also what it withheld where it cut it (a vertex
        // stopped by its cap 22 mm above the floor would otherwise predict 0 and close the gap a
        // cap's width a round).
        let Up = bitcast<f32>(atomicLoad(&contacts[P.levelWord + 4u]));
        let wp = pred[v].xyz;
        dev = length(x - (xdet[v].xyz + wp - Up * wp));
        w = work4(P.targetOff + 4u * v).xyz - x;
      } else {
        let left = length(yref.xyz - x) + yref.w;
        dev = select(left, min(left, length(d)), P.newton != 0u);
        w = vec3<f32>(0.0);
      }
    }
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
    var bodyReached = false;
    var pd = OgcPose(vec3<f32>(0.0), bRt);
    if (P.bodyBase != 0u) {
      let p1 = ogcPose(1u);
      bRt = transpose(p1.R);
      pd = p1;
      if (mandatory) { pd = ogcPose(0u); }
      let y0 = ogcToBody(pd, x);
      // After the mandatory round the detection pose is the end pose, where a_v = R1 R1^T (x - c1)
      // + c1 - x is 0 exactly. Evaluated in f32 it is rounding, which gave held and carried cloth
      // a prediction of a few ulps and made a pair nearer than sepEps a conflict of a round that
      // cannot conflict (16^2 rotating sphere, tick 60: a cloth edge against a hull edge).
      if (mandatory) { a = ogcToWorld(p1, y0) - x; }
      // Carried with the solid (the fail-safe's impact zones), the vertex deviates from its
      // prediction by its inertial motion less a_v.
      if (carried) {
        w = a; dev = length(inertial - a) + yref.w; pushed = true; withBody = true;
        atomicOr(&work[P.heldOff + v], 64u);
      }
      if (anchored) { anc = a; }
      dev += (1.0 - shift) * shiftReach(w, a, anc);
      env = envelopeOf(dev);
      if (mandatory) {
        let Rr = p1.R * transpose(pd.R);
        theta = 2.0 * asin(min(1.0, 0.5 * sqrt(max(3.0 - (Rr[0].x + Rr[1].y + Rr[2].z), 0.0))));
      }
      let hb = ogcSolidHeaderAt(P.bodyBase);
      // Relative to the solid the round moves the vertex at most |w - a_v| + env / 2 along its
      // chord, and sigma_v off it. A ball that clears every stored brick (all within bodyReach
      // of the body origin) is free of it.
      // The ball covers the shifted prediction too (in the hull of w, a_v and the anchor).
      let rho = max(length(w - a), select(0.0, length(anc - a), shift < 1.0)) + 0.5 * env;
      // Q covers every prediction the round may end with: w moved towards a_v and the anchor,
      // then scaled towards 0.
      let dc = p1.c - pd.c;
      let Q = max(max(max(length(w - dc), length(a - dc)), length(dc)), select(0.0, length(anc - dc), shift < 1.0)) + 0.5 * env;
      sigma = (theta * theta * (length(x - pd.c) + Q) + 2.0 * theta * Q) / 8.0;
      // Moving with the solid, the vertex's round is straight in the solid's frame: none there.
      let bm = select(sigma, 0.0, withBody);
      if (length(y0) - rho - bm < P.bodyReach) {
        bodyReached = true;
        bq = roundBound(v, hb, y0, rho, OGC_BODY_Y, bm, vec3<f32>(0.0), rho, ogcToBody(ogcPose(0u), xt[v].xyz));
        // It touches the solid when its ball can reach r (a far ball's terms are only the
        // Lipschitz length limit, q = D = 0): held with the solid next step only then.
        if (bq.q.w != 0.0 && bq.q.w < 1.0e38) {
          atomicAdd(&wgBody, 1u);
          if (any(bq.q.xyz != vec3<f32>(0.0)) || any(bq.D != vec3<f32>(0.0))) { atomicOr(&work[P.heldOff + v], 16u); }
        }
        // The prediction moves towards a_v only as far as the bound needs.
        let kb = predictionScale(bq, bRt * (w - a));
        if (kb < 1.0) {
          w = a + (w - a) * kb;
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
    if (P.bodyBase == 0u) { dev += (1.0 - shift) * shiftReach(w, vec3<f32>(0.0), anc); env = envelopeOf(dev); }
    // The least shift, before the static solid's bound (whose ball is taken about the final
    // prediction), and then the moving solid's bound again, about a_v.
    if (shift < 1.0) {
      w = anc + shift * (w - anc);
      if (bodyReached) { let k2 = predictionScale(bq, bRt * (w - a)); if (k2 < 1.0) { w = a + (w - a) * k2; pushed = true; } }
    }
    if (P.solidBase != 0u) {
      // The round moves the vertex at most |w| + env / 2 (the cap is 0.5 gamma_r or gamma_p of env),
      // straight in the world, or within sigma_v of straight when it moves with the solid.
      let anchorA = bodyReached && bq.q.w != 0.0;
      let qj = solidRound(v, x, w, env, select(0.0, sigma, withBody), a, anchorA);
      let D4 = vec4<f32>(qj.D, qj.rl);
      // Bound by the moving solid too, the prediction moves towards a_v when the static solid
      // allows a_v: both sets are star-shaped about it (the moving solid's about a_v, the static
      // solid's affine certificate convex and holding a_v), so the cut keeps both. Scaled
      // towards 0 instead it can leave the moving solid's bound where nothing squeezes it (a
      // vertex on the floor beside the curtain's ball as it starts to roll: w scaled by the
      // floor moved it 0.06 mm further towards the ball). Where the static solid forbids a_v
      // the solids squeeze the vertex, and the check below reports it.
      if (qj.q.w >= 1.0e38) { w = vec3<f32>(0.0); }
      else if (anchorA && ogcCutScale(qj.q, D4, vec3<f32>(0.0), a) >= 1.0) { w = a + (w - a) * ogcCutScale(qj.q, D4, a, w - a); }
      else { w *= predictionScale(qj, w); }
    }
    if (P.bodyBase != 0u) {
      // Both bounds hold the prediction (f32 rounding aside), or the cloth is squeezed.
      if (predictionScale(bq, (bRt * (w - a)) * (1.0 - 1.0e-5)) < 1.0) { atomicAdd(&wgSqueeze, 1u); }
      putWork4(P.bodyOff + 8u * v, bq.q);
      putWork4(P.bodyOff + 8u * v + 4u, vec4<f32>(a, sigma));
      putWork4(P.bodyDevOff + 4u * v, vec4<f32>(bq.D, bq.rl));
    }
    if (fresh) { pred[P.N + v] = vec4<f32>(select(x + w, yref.xyz, held && !pushed), yref.w); }
    // The step's measured residual rho (the next step's fresh round) is how far its final position
    // strays from the last prediction it was certified against: the mandatory round's X + w, or
    // the last levelled round's chord end. (Measured against the mandatory round's alone, a sheet
    // landing flat at 4.7 m/s, its prediction cut 22 mm short of the floor by the solid's
    // Lipschitz bound, carried that shortfall into the next step's envelope: 40 mm while it rests.)
    else if (lev) { pred[P.N + v] = vec4<f32>(x + w, yref.w); }
    pred[v] = vec4<f32>(w, plen);
    if (!(all(abs(x) < vec3<f32>(3.0e38)))) { atomicOr(&wgBad, 1u); }
    xdet[v] = vec4<f32>(x, 0.0);
    putWork4(P.originOff + 4u * v, vec4<f32>(0.0));
    atomicStore(&work[P.envOff + v], bitcast<u32>(env));
    atomicStore(&work[P.tvOff + v], bitcast<u32>(1.0));
    atomicMax(&wgMaxEnv, bitcast<u32>(env));
    atomicMax(&wgMaxPred, bitcast<u32>(length(w)));
    let e = env / P.rqMax;
    wgCube[li] = e * e * e;
    if (fresh && held) { atomicAdd(&wgHeld, 1u); }
  }
  workgroupBarrier();
  if (li == 0u) {
    let m = atomicLoad(&wgMaxEnv);
    if (m != 0u) { atomicMax(&counters[OGC_MAX_ENVELOPE], m); }
    let mw = atomicLoad(&wgMaxPred);
    if (mw != 0u) { atomicMax(&counters[OGC_MAX_PREDICTION], mw); }
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
  let li = atomicAdd(&counters[OGC_ROUND_LOG_COUNT], 1u);
  let m = bitcast<f32>(atomicExchange(&counters[OGC_MAX_ENVELOPE], 0u));
  atomicMax(&counters[OGC_F_PEAK_ENVELOPE], bitcast<u32>(m));
  atomicStore(&counters[OGC_ROUND_ENVELOPE], bitcast<u32>(m));
  let cube = f32(atomicExchange(&counters[OGC_ENVELOPE_SUM], 0u)) / (1024.0 * f32(P.N));
  // A face's box spans its swept prediction as well as the envelope: a cell at least the round's
  // longest prediction keeps every box within a few cells per axis however fast the cloth moves
  // (entries per primitive are bounded by construction, not by the authored grid capacity).
  let wmax = bitcast<f32>(atomicExchange(&counters[OGC_MAX_PREDICTION], 0u));
  atomicMax(&counters[OGC_F_MAX_PREDICTION], bitcast<u32>(wmax));
  if (li < OGC_ROUND_LOG_N) {
    let b = OGC_ROUND_LOG_BASE + OGC_ROUND_LOG_W * li;
    atomicStore(&counters[b], bitcast<u32>(wmax)); atomicStore(&counters[b + 1u], bitcast<u32>(m));
    atomicStore(&counters[b + 2u], atomicLoad(&counters[OGC_F_CANDIDATES]));
    atomicStore(&counters[b + 3u], atomicLoad(&counters[OGC_F_DAT_PAIRS]));
    atomicStore(&counters[b + 4u], atomicLoad(&counters[OGC_F_HELD]));
    for (var k = 5u; k < 8u; k++) { atomicStore(&counters[b + k], 0u); }
    atomicStore(&counters[b + 8u], atomicLoad(&counters[OGC_F_SCANNED]));
    atomicStore(&counters[b + 9u], atomicLoad(&counters[OGC_F_COLLISIONS]));
    atomicStore(&counters[b + 10u], atomicLoad(&counters[OGC_F_BOX_REJECTED]));
    atomicStore(&counters[b + 11u], 0u);
  }
  let h = max(select(max(P.cellFloor, m), max(P.cellFloor, P.rqMax * pow(cube, 1.0 / 3.0)), P.cellRule == 1u), wmax);
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
  // The mandatory round is unlevelled (its anchor X + w, u = 0), and no Newton iteration has run yet.
  if (P.levelWord != 0u) { for (var i = 0u; i < 8u; i++) { atomicStore(&contacts[P.levelWord + i], 0u); } }
  atomicAdd(&counters[OGC_F_STEPS], 1u);
  atomicAdd(&counters[OGC_F_DETECTIONS], 1u);
  if (P.N > 0u) { writeArgs(true); } else { writeArgs(false); }
}
`;

/**
 * Between iterations (OGC Algorithm 3, lines 23-30): when gamma_e N vertices
 * hit their bound, the filter held a vertex at 0, or (levelOn) the Newton iteration's line search
 * took no step on a cut arc, the next detection round's dispatches get their workgroups;
 * otherwise they get none. The decision never leaves the GPU.
 * Levelled re-detection (levelOn: a Newton solver with its control record bound, planar
 * predicted DAT): first the level of the round in flight follows the Newton step just applied,
 * L <- L + alpha_c (1 - L), i.e. u <- u (1 - alpha_c), unless the filter froze it; read once per
 * Newton iteration (the control's count), so the capped iterations after Newton is done change
 * nothing. A round armed here is levelled: it predicts the rest of that step's direction,
 * w = (1 - alpha_c) d (prep), and starts at level 0 (u = 1).
 */
/**
 * freeze (the document's solver.contactFreeze 'mandatory', diagnosis only): the decision never arms a round, so the
 * step's mandatory round (with its redos) is its only one: its pair set and every vertex's entries hold for all the
 * step's Newton iterations (the line-search filter still re-places the projection's records about each iterate from
 * that set). The counters it consumes are consumed as before; a stall, a small step or a travel bound arms nothing.
 */
/**
 * With the Newton solver (level) the decision also takes the end-of-step verification's verdict (OGC_VERIFY_WGSL) and
 * writes the solver's done word: a candidate done (2) that the radius-r query found complete becomes done (1); one with a
 * pair within r and no entry becomes 0 and arms a re-detection round at the iterate, and Newton runs on (as after a
 * stall). At the step's last iteration a missing pair cannot be re-detected: it is accepted as a one-step lag, counted
 * (frameLagPairs, frameLagSteps) and never written as done. With the contact set frozen the same holds at every
 * candidate (the frozen set is the step's by definition).
 */
export function ogcDecideWGSL(level: boolean, freeze = false): string {
  return header(level ? [0, 8, 9, 10, 19] : [0, 9, 10], level ? [19] : []) + ARGS_FNS + /* wgsl */`
@compute @workgroup_size(1)
fn main() {
  // A Newton iteration whose line search took no step on a cut arc (nwChoose: alpha 0, not done)
  // leaves the iterate where the round's set stops it: the next iteration needs new certificates
  // about it (a levelled round's, predicting that direction), as a vertex the filter held at 0.
  var noStep = false;
  ${level ? `if (P.levelOn != 0u) {
    let ctl = nwctl[0];
    let it = u32(ctl.w);
    if (it != atomicLoad(&contacts[P.levelWord + 2u])) {
      // Also, by the solver's cap-aware done rule, an iteration whose step was small while travel
      // bounds bind (certificates at q that are not contacts there: the cap, a piece's ball, a
      // solid face beyond r, a record its pair cannot close within r) with a positive multiplier,
      // their vertices counted in (control + 2).w: Newton is not done, and they need new
      // certificates about the iterate. Read only where the solver binds that word.
      var travel = false;
      if (arrayLength(&nwctl) > 2u) { travel = nwctl[2].w > 0.0; }
      noStep = ctl.x == 0.0 && (ctl.z == 0.0 || travel);
      atomicStore(&contacts[P.levelWord + 2u], it);
      atomicStore(&contacts[P.levelWord + 3u], bitcast<u32>(ctl.z));
      let lf = atomicLoad(&contacts[P.levelWord + 1u]);
      if ((lf & OGC_LEVELLED) != 0u && (lf & OGC_FROZEN) == 0u) {
        atomicStore(&contacts[P.levelWord], bitcast<u32>(bitcast<f32>(atomicLoad(&contacts[P.levelWord])) * (1.0 - ctl.z)));
      }
    }
  }` : ""}
  var verifyArm = false;
  ${level ? `{
    let mode = atomicExchange(&counters[OGC_VERIFY_MODE], 0u);
    let missing = atomicExchange(&counters[OGC_VERIFY_MISSING], 0u);
    if (mode != 0u) {
      atomicAdd(&counters[OGC_F_VERIFIES], 1u);
      if (mode == 1u && missing != 0u && ${freeze ? "false" : "true"}) {
        atomicAdd(&counters[OGC_F_VERIFY_MISSING], 1u);
        atomicAdd(&counters[OGC_F_MISSING_PAIRS], missing);
        verifyArm = true;
        nwctl[0].x = 0.0;
      } else {
        if (missing != 0u) { atomicAdd(&counters[OGC_F_LAG_PAIRS], missing); atomicAdd(&counters[OGC_F_LAG_STEPS], 1u); }
        // A candidate: done when complete (or frozen); at the last iteration a lagged one stays not done.
        if (mode == 1u) { nwctl[0].x = 1.0; }
        else if (mode == 3u) { nwctl[0].x = select(0.0, 1.0, missing == 0u); }
      }
    }
  }` : ""}
  atomicStore(&counters[OGC_FORCE_HELD], 0u);
  let ex = atomicExchange(&counters[OGC_EXCEED], 0u);
  // A vertex the line-search filter held at alpha 0 would hold the whole Newton step: re-detect.
  let stalled = atomicExchange(&counters[OGC_FILTER_STALLED], 0u);
  atomicAdd(&counters[OGC_F_DECIDES], 1u);
  atomicAdd(&counters[OGC_F_TRUNCATED], ex);
  atomicMax(&counters[OGC_F_TRUNCATED_MAX], ex);
  var on = false;
  if (${freeze ? "false" : "(ex > 0u && ex >= P.exceedThreshold) || stalled != 0u || noStep || verifyArm"}) {
    if (atomicLoad(&counters[OGC_ROUNDS]) < P.roundCap) {
      atomicAdd(&counters[OGC_ROUNDS], 1u);
      atomicAdd(&counters[OGC_F_REDETECTIONS], 1u);
      atomicAdd(&counters[OGC_F_DETECTIONS], 1u);
      on = true;
      ${level ? `if (P.levelOn != 0u) {
        // The level the last round reached (its anchors c(a) = X + w - u w), for the new round's envelope.
        atomicStore(&contacts[P.levelWord + 4u], atomicLoad(&contacts[P.levelWord]));
        atomicStore(&contacts[P.levelWord], bitcast<u32>(1.0));
        atomicStore(&contacts[P.levelWord + 1u], OGC_LEVELLED);
        atomicAdd(&counters[OGC_F_LEVELLED], 1u);
      }` : ""}
    } else {
      atomicOr(&counters[OGC_STATUS_WORD], OGC_ST_ROUND_CAP);
      atomicAdd(&counters[OGC_F_ROUND_CAP], 1u);
    }
  }
  writeArgs(on);
}
`;
}

/**
 * After a detection round: a round with conflicting pairs (predictions that cross, or no
 * plane at X while a vertex is predicted to move) is redone at once, before any truncation:
 * with its conflicts' predictions shifted (least shift, resolvePlane) while that makes
 * progress, up to OGC_REDO_ROUNDS times; then the fail-safe (reported as failSafe): every
 * prediction held, which without a moving solid is Planar-DAT and never conflicts; with one,
 * its impact zones carried, redone while they grow, up to redoCap().
 */
export const OGC_REDO_WGSL = header([0, 8, 9, 10]) + LEVEL_FNS + ARGS_FNS + REDO_CAP_FN + /* wgsl */`
@compute @workgroup_size(1)
fn main() {
  atomicMax(&counters[OGC_F_ZONE_PEAK], atomicLoad(&counters[OGC_STEP_ZONE]));
  let c = atomicExchange(&counters[OGC_CONFLICTS], 0u);
  let progress = atomicExchange(&counters[OGC_REDO_PROGRESS], 0u);
  if (c == 0u) { writeArgs(false); return; }
  if (atomicLoad(&counters[OGC_ROUNDS]) != 0u && levelled()) {
    // A levelled re-detection with a conflict (predictions that cross, or no plane at X while a
    // vertex is predicted to move) is redone at once unlevelled: every prediction 0 about the
    // iterate, which never conflicts without a moving solid. Counted apart from the mandatory
    // round's; the redo is unlevelled, so a conflict in it takes the path below, as unlevelled.
    atomicStore(&counters[OGC_FORCE_HELD], 1u);
    atomicStore(&contacts[P.levelWord], 0u);
    atomicStore(&contacts[P.levelWord + 1u], 0u);
    atomicAdd(&counters[OGC_F_LEVEL_FALLBACK], 1u);
    atomicAdd(&counters[OGC_F_DETECTIONS], 1u);
    writeArgs(true);
    return;
  }
  atomicAdd(&counters[OGC_F_CONFLICTS], c);
  let redone = atomicLoad(&counters[OGC_FORCE_HELD]);
  var next = 0u;
  if (redone < ${OGC_REDO_ROUNDS}u && progress != 0u) { next = redone + 1u; }
  else if (redone <= ${OGC_REDO_ROUNDS}u) { next = ${OGC_REDO_ROUNDS + 1}u; }
  else if (P.bodyBase != 0u && progress != 0u && redone < redoCap()) { next = redone + 1u; }
  if (next != 0u) {
    atomicStore(&counters[OGC_FORCE_HELD], next);
    if (next > ${OGC_REDO_ROUNDS}u) { atomicOr(&counters[OGC_STATUS_WORD], OGC_ST_FAIL_SAFE); }
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
export function ogcPlanesWGSL(fixed: boolean, pass: "truncate" | "filter" | "records"): string {
  const filter = pass === "filter", records = pass === "records";
  return [PARAMS_STRUCT, OGC_COUNTERS_WGSL, OGC_COMMON_WGSL, DECL[0], DECL[2], DECL[4], DECL[5], DECL[9], DECL_PAIRS_READ, DECL[14],
    ...(filter || records ? [DECL[15], DECL[8]] : [])].join("\n") + (filter || records ? LEVEL_FNS : "") + PRED_FNS + BODY_PAIR_FNS + /* wgsl */`
fn ratio(s0: f32, s1: f32) -> f32 {
  // A wrong-side start may move toward its side but never further across.
  if (s0 < 0.0) { return select(0.0, 1.0, s1 >= s0); }
  if (s1 >= 0.0) { return 1.0; }
  return clamp(P.gammaR * s0 / (s0 - s1), 0.0, 1.0);
}
// A surface sample (index >= N) is static: it neither moves nor is truncated.
fn lower(v: u32, t: f32) { if (t < 1.0 && v < P.N) { atomicMin(&work[P.tvOff + v], bitcast<u32>(t)); } }
// The filter's plane record (m, b) of cloth vertex v: counted (filter), then written into its segment
// (records) with its far bit (ogcRecordFar: the bits follow the pool, one per slot).
fn record(v: u32, r: vec4<f32>, far: bool) {
  if (v >= P.N) { return; }
  ${filter ? "atomicAdd(&work[P.recCountOff + v], 1u);" : ""}
  ${records ? `let i = atomicAdd(&work[P.recCountOff + v], 1u);
  let slot = atomicLoad(&contacts[P.segWord + 2u * v]) + i;
  if (i < atomicLoad(&contacts[P.segWord + 2u * v + 1u]) && slot < P.recCap) {
    let b = P.recWord + 4u * slot;
    atomicStore(&contacts[b], bitcast<u32>(r.x)); atomicStore(&contacts[b + 1u], bitcast<u32>(r.y));
    atomicStore(&contacts[b + 2u], bitcast<u32>(r.z)); atomicStore(&contacts[b + 3u], bitcast<u32>(r.w));
    let fw = P.recWord + 4u * P.recCap + (slot >> 5u); let bit = 1u << (slot & 31u);
    if (far) { atomicOr(&contacts[fw], bit); } else { atomicAnd(&contacts[fw], ~bit); }
  }` : ""}
}
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
        for (var k = 0u; k < 4u; k++) { ${records ? "record(ids[k], vec4<f32>(0.0, 0.0, 0.0, 1.0), false);" : `lower(ids[k], 0.0);${filter ? " record(ids[k], vec4<f32>(0.0), false);" : ""}`} }
        ${records ? "" : "atomicAdd(&wgUncert, 1u);"}
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
        ${filter || records ? `// A levelled round's records (ogcProjectTrial) hold at every level a: each side's slack is
        // also taken at X, about the plane placed among the supports there, and its record keeps
        // the lesser room of the two ends (b = max(b0, b1)). Relative to c(a) = X + a w the bound
        // on n.y is then linear in a and on its side of the plane interpolated between the two
        // placements, so the pair, divided at both ends, is divided at every level between.
        let lev = levelled();
        let U = levelU();
        var s00 = array<f32, 4>(3.4e38, 3.4e38, 3.4e38, 3.4e38);
        if (lev) {
          var p0: array<f32, 4>;
          var a0 = 3.4e38; var b0 = -3.4e38;
          for (var k = 0u; k < 4u; k++) {
            p0[k] = dot(n, xdet[ids[k]].xyz - o);
            if (k < nA) { a0 = min(a0, p0[k]); } else { b0 = max(b0, p0[k]); }
          }
          let plane0 = b0 + 0.5 * ((a0 - b0) + eB - eA);
          for (var k = 0u; k < 4u; k++) {
            s00[k] = select(plane0 - p0[k], p0[k] - plane0, k < nA) - P.sepEps - strayOf(ids[k], turned);
          }
        }` : ""}
        ${records ? `// Far: the pair stays at least r apart at every trial of this filter, so its records bound
        // travel, not contact (ogcProjectActive's travel). Along n each side's support at a trial is
        // within its reach of the iterate's: |d| (every trial point and its projections stay within
        // |d| of x), or in a levelled round, whose projections start from the anchor, the farthest
        // anchor c(u) (u in [0, U]) plus the cap. A surface sample does not move.
        var far = false;
        {
          let nh = n / length(n);
          var lo = 3.4e38; var hi = -3.4e38;
          for (var k = 0u; k < 4u; k++) {
            var p = dot(nh, anchored(ids[k])); var reach = 0.0;
            if (ids[k] < P.N) {
              let xk = pos[ids[k]].xyz; let dk = xdet[ids[k]]; let c1 = dk.xyz + predOf(ids[k]);
              p = dot(nh, xk);
              reach = select(length(dir[ids[k]].xyz), max(length(c1 - xk), length(c1 - U * predOf(ids[k]) - xk)) + dk.w, lev);
            }
            if (k < nA) { lo = min(lo, p - reach); } else { hi = max(hi, p + reach); }
          }
          far = lo - hi >= P.r;
        }` : ""}
        for (var k = 0u; k < 4u; k++) {
          let a = k < nA;
          let s0 = select(plane - proj[k], proj[k] - plane, a) - P.sepEps - strayOf(ids[k], turned);
          ${filter || records ? `if (ids[k] < P.N) {
            // Along the direction from x: the slack s(x + alpha d) keeps s0 - gamma_r max(s0, 0), the
            // truncation's own target (a wrong-side anchor: no further across), linear in alpha.
            // (Formed as ogcProject forms m.(x - c) - b, so its test along d agrees bit for bit.)
            // -b: the room the record leaves its vertex towards the plane (the lesser of both ends when levelled).
            var nb = P.gammaR * max(s0, 0.0);
            if (lev) { nb = min(nb, P.gammaR * max(s00[k], 0.0)); }
            // The anchor at the iterate's level, c(U) = X + w - U w (U = 0 unlevelled: X + w).
            let wk = predOf(ids[k]);
            let room = select(-dn[k], dn[k], a) + nb + U * (select(-1.0, 1.0, a) * dot(n, wk));
            let d = dir[ids[k]].xyz;
            ${filter ? `// Along the joint path (x + alpha d at level L + alpha (1 - L)) the anchor moves by -U w per unit alpha.
            let sd = select(-1.0, 1.0, a) * dot(n, d - U * wk);
            if (sd < 0.0) { lower(ids[k], clamp(room / -sd, 0.0, 1.0)); }` : ""}
            // As a half-space of y relative to the anchor c: sigma n.(y - c) >= -gamma_r max(s0, 0).
            // Every trial point and its projections stay within |d| of x (each projection is onto a convex
            // set holding x) and within the cap |y - c| <= x_det.w: a plane either keeps whole is
            // redundant and goes unrecorded. A levelled round's projections start from the anchor, not
            // x: there only the cap makes a plane redundant.
            let ln = length(n) * (1.0 + 1.0e-5);
            if ((lev || room < ln * length(d)) && nb < ln * xdet[ids[k]].w) {
              record(ids[k], vec4<f32>(select(-1.0, 1.0, a) * n, -nb), ${records ? "far" : "false"});
            }
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
 * A vertex whose solid bounds the filter re-took about an iterate o (originOff) has them
 * relative to o, which they hold, as they hold the iterate since (the filter's, no round
 * between): the solids then scale its motion from o, not from w (a no-op to rounding).
 */
export const OGC_APPLY_WGSL = header([0, 2, 4, 5, 8, 9, 14], [2]) + LEVEL_FNS + PRED_FNS + SOLID_ACCESS + OGC_SOLID_WGSL + CUT_FNS + /* wgsl */`
var<workgroup> wgCount: atomic<u32>;
var<workgroup> wgCapped: atomic<u32>;
var<workgroup> wgSolidCut: atomic<u32>;
var<workgroup> wgBodyCut: atomic<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let v = gid.x;
  // In a levelled round the iterate is certified at its level by the projected arc itself
  // (x = X at level 0; ogcProjectTrial after), while x - X - w measures the level, not a residual:
  // nothing to truncate.
  let lev = levelled();
  if (v < P.N && lev) { atomicStore(&work[P.tvOff + v], bitcast<u32>(1.0)); }
  if (v < P.N && !lev) {
    let tv = bitcast<f32>(atomicExchange(&work[P.tvOff + v], bitcast<u32>(1.0)));
    let p = pos[v];
    let det = xdet[v];
    let w = predOf(v);
    var delta = (p.xyz - det.xyz - w) * tv;
    var cut = tv < 1.0;
    // A plane (another primitive within the envelope) or a solid cut it: a contact constraint, which
    // holds the vertex next step. The cap alone is no contact: the vertex deviated from its prediction
    // by more than its envelope, and the residual the next step measures widens that envelope.
    var held = cut;
    let l = length(delta);
    if (l > det.w) { delta *= det.w / l; cut = true; atomicAdd(&wgCapped, 1u); }
    // The solids scale the motion m from the anchor: w, or the filter's re-bound origin o.
    let o4 = work4(P.originOff + 4u * v);
    let rebound = o4.w != 0.0;
    let anc = select(w, o4.xyz, rebound);
    var m = w + delta - anc;
    if (P.solidBase != 0u) {
      // The static solid: phi(x_det + w + t delta) >= min((1 - gamma_r) phi0, r) for t <= (1 - f(w)) / f(delta).
      let k = ogcCutScale(work4(P.solidOff + 4u * v), work4(P.solidDevOff + 4u * v), select(w, vec3<f32>(0.0), rebound), m);
      if (k < 1.0) { m *= k; cut = true; held = true; atomicAdd(&wgSolidCut, 1u); }
    }
    if (P.bodyBase != 0u) {
      // The moving solid: phi_body >= min((1 - gamma_r) phi0, r) for the motion w - a_v + delta relative to it.
      // Its terms are in the body frame at the end pose.
      let qj = work4(P.bodyOff + 8u * v);
      if (qj.w != 0.0) {
        let Rt = transpose(ogcPose(1u).R);
        let k = ogcCutScale(qj, work4(P.bodyDevOff + 4u * v), Rt * (anc - select(work4(P.bodyOff + 8u * v + 4u).xyz, o4.xyz, rebound)), Rt * m);
        if (k < 1.0) { m *= k; cut = true; held = true; atomicAdd(&wgBodyCut, 1u); atomicOr(&work[P.heldOff + v], 16u); }
      }
    }
    if (cut) {
      pos[v] = vec4<f32>(det.xyz + anc + m, p.w);
      atomicAdd(&wgCount, 1u);
      if (held) { atomicOr(&work[P.heldOff + v], 1u); }
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
 * The line-search filter for a Newton solver, once per Newton iteration about the iterate x
 * (positions, untouched) and the solver's direction d. Where a solid bound cuts d (k < 1), the
 * vertex's solid bounds are re-taken about the iterate x itself, as OGC re-bounds each iterate:
 * the field's ball bound about x over |d|, with room min(max(gamma_r phi(x), phi(x) - r),
 * phi(x) - sepEps), no margin (the iterate faces the end pose, so its motion is straight in
 * both frames) and no entry (the round's entries stay), stored relative to the origin
 * o = x - X (originOff), which the apply then scales from. Then each vertex's certificate
 * records for ogcProject (the cap ball and both solid bounds, world origins; OGC_PROJECT_RECORD)
 * and alpha_v = the largest alpha <= 1 with x + alpha d inside every certificate: the planes
 * (ogcPlanesWGSL(fixed, "filter"), reduced into tv), the |x - X - w| <= x_det.w cap, and the
 * static and moving solids' bounds, into alpha[v], with the atomic min alpha_min into alpha[N]
 * (set to 1 by OGC_FILTER_BEGIN_WGSL). Each certificate is convex and holds x, so every alpha
 * in [0, alpha_v] is certified. The vertices it bounds below 1 count towards the re-detection
 * decision, and one held at 0 arms it outright (OGC_FILTER_STALLED). A bound holds no vertex:
 * it limits a line search, it is no contact, and the round that follows predicts 0 about the
 * iterate anyway. (Holding them spread the holds a ring per step from a pin.)
 */
export const OGC_FILTER_WGSL = header([0, 2, 4, 5, 8, 9, 14, 15, 16]) + ROUND_LOG_FNS + LEVEL_FNS + PRED_FNS + SOLID_ACCESS + OGC_SOLID_WGSL + CUT_FNS + /* wgsl */`
var<workgroup> wgBound: atomic<u32>;
var<workgroup> wgStalled: atomic<u32>;
var<workgroup> wgRebound: atomic<u32>;
var<workgroup> wgMaxDir: atomic<u32>;
var<workgroup> wgFrozen: atomic<u32>;
// The world origin of v's static (0) or moving (1) solid terms: X plus a_v (0 for the static
// solid), or the re-bound o. ogcProject reads it back from the records, so the filter's cuts and
// its own test take the same operands (x - origin) bit for bit.
fn solidOrigin(v: u32, det: vec4<f32>, body: bool) -> vec3<f32> {
  let o4 = work4(P.originOff + 4u * v);
  if (o4.w != 0.0) { return det.xyz + o4.xyz; }
  if (body) { return det.xyz + work4(P.bodyOff + 8u * v + 4u).xyz; }
  return det.xyz + vec3<f32>(0.0);
}
// The cuts along d of the cap, the static solid and the moving solid (1: none) from the iterate x.
// The cap's along the joint path of a levelled round: its anchor X + w - U w moves by -U w per unit alpha.
fn cutsOf(v: u32, x: vec3<f32>, det: vec4<f32>, w: vec3<f32>, d: vec3<f32>, U: f32) -> vec3<f32> {
  var k = vec3<f32>(ogcBallExit(x - (det.xyz + w - U * w), d - U * w, det.w), 1.0, 1.0);
  if (P.solidBase != 0u) { k.y = ogcCutScale(work4(P.solidOff + 4u * v), work4(P.solidDevOff + 4u * v), x - solidOrigin(v, det, false), d); }
  if (P.bodyBase != 0u) {
    let qj = work4(P.bodyOff + 8u * v);
    if (qj.w != 0.0) {
      let Rt = transpose(ogcPose(1u).R);
      k.z = ogcCutScale(qj, work4(P.bodyDevOff + 4u * v), Rt * (x - solidOrigin(v, det, true)), Rt * d);
    }
  }
  return k;
}
// A ball bound's terms about p over |d| <= rc: a far ball certifies all of it (q = D = 0).
fn reTerms(h: OgcSolidHeader, p: vec3<f32>, rc: f32) -> OgcTerms {
  var sb = ogcSolidBallBound(h, p, rc, P.r);
  if (sb.far) { sb.g0 = vec3<f32>(0.0); sb.D = vec3<f32>(0.0); }
  return termsOf(h, sb, rc, 0.0);
}
// The vertex's solid bounds (those its round has) re-taken about the iterate x = X + rel.
fn rebind(v: u32, x: vec3<f32>, rel: vec3<f32>, d: vec3<f32>) {
  let rc = min(length(d) * (1.0 + 1.0e-5), 0.5 * P.rqMax);
  if (P.solidBase != 0u && work4(P.solidOff + 4u * v).w != 0.0) {
    let t = reTerms(ogcSolidHeader(), x, rc);
    putWork4(P.solidOff + 4u * v, t.q);
    putWork4(P.solidDevOff + 4u * v, vec4<f32>(t.D, t.rl));
  }
  if (P.bodyBase != 0u && work4(P.bodyOff + 8u * v).w != 0.0) {
    let t = reTerms(ogcSolidHeaderAt(P.bodyBase), ogcToBody(ogcPose(1u), x), rc);
    putWork4(P.bodyOff + 8u * v, t.q);
    putWork4(P.bodyDevOff + 4u * v, vec4<f32>(t.D, t.rl));
  }
  putWork4(P.originOff + 4u * v, vec4<f32>(rel, 1.0));
}
fn putContact4(i: u32, q: vec4<f32>) {
  atomicStore(&contacts[i], bitcast<u32>(q.x)); atomicStore(&contacts[i + 1u], bitcast<u32>(q.y));
  atomicStore(&contacts[i + 2u], bitcast<u32>(q.z)); atomicStore(&contacts[i + 3u], bitcast<u32>(q.w));
}
// A solid bound's convex certificate for ogcProject, at word b: (origin, flag), (q, R), (D, 0),
// the world origin and flag 0 none, 1 bound, 2 held (an inside
// marker, or a bound holding neither piece at the iterate). Of its two pieces (ogcCutScale) the
// one that keeps more of d: the affine one, or the Lipschitz ball (q = D = 0, R = rl). In a
// levelled round the piece must also hold the far anchor (X + w, relative to the origin: the
// projection certifies from c(a) on the chord to it, and the round's own origin holds X).
// Returns whether it holds the vertex.
fn putPiece(b: u32, origin: vec3<f32>, q: vec4<f32>, D4: vec4<f32>, base: vec3<f32>, d: vec3<f32>, anchor: vec3<f32>, lev: bool) -> bool {
  var flag = 1.0;
  var pq = q; var pD = D4.xyz;
  if (q.w == 0.0) { flag = 0.0; }
  else if (q.w >= 1.0e38) { flag = 2.0; }
  else {
    let bb = dot(base, base);
    let ka = ogcPieceScale(q, pD, base, d);
    let tol = 1.0 + 1.0e-5;
    if (lev) {
      let aa = dot(anchor, anchor);
      let affOk = bb <= q.w * q.w * tol && aa <= q.w * q.w * tol
        && dot(pD, abs(base)) - dot(q.xyz, base) <= tol && dot(pD, abs(anchor)) - dot(q.xyz, anchor) <= tol;
      let ballOk = bb <= D4.w * D4.w * tol && aa <= D4.w * D4.w * tol;
      if (ballOk && (!affOk || ka <= 0.0 || ogcBallExit(base, d, D4.w) > ka)) { pq = vec4<f32>(0.0, 0.0, 0.0, D4.w); pD = vec3<f32>(0.0); }
      else if (!affOk) { flag = 2.0; }
    } else {
      let ballOk = bb <= D4.w * D4.w * tol;
      if (ballOk && (ka <= 0.0 || ogcBallExit(base, d, D4.w) > ka)) { pq = vec4<f32>(0.0, 0.0, 0.0, D4.w); pD = vec3<f32>(0.0); }
      else if (!(bb <= q.w * q.w * tol)) { flag = 2.0; }
    }
  }
  putContact4(b, vec4<f32>(origin, flag)); putContact4(b + 4u, pq); putContact4(b + 8u, vec4<f32>(pD, 0.0));
  return flag == 2.0;
}
// The vertex's fixed records (OGC_PROJECT_RECORD vec4s): the cap (c1 = X + w, radius), each solid's
// piece, then w (a levelled round's anchor c(a) = c1 - (1 - a) w). Returns whether a piece holds it.
fn writeRecords(v: u32, det: vec4<f32>, w: vec3<f32>, x: vec3<f32>, d: vec3<f32>, lev: bool) -> bool {
  let b = P.fixedWord + 4u * ${OGC_PROJECT_RECORD}u * v;
  putContact4(b, vec4<f32>(det.xyz + w, det.w));
  putContact4(b + 28u, vec4<f32>(w, 0.0));
  var held = false;
  if (P.solidBase != 0u) {
    let o = solidOrigin(v, det, false);
    held = putPiece(b + 4u, o, work4(P.solidOff + 4u * v), work4(P.solidDevOff + 4u * v), x - o, d, det.xyz + w - o, lev);
  } else { putContact4(b + 4u, vec4<f32>(0.0)); }
  var bodyOn = false;
  if (P.bodyBase != 0u) {
    let qj = work4(P.bodyOff + 8u * v);
    if (qj.w != 0.0) {
      let o = solidOrigin(v, det, true);
      let Rt = transpose(ogcPose(1u).R);
      held = putPiece(b + 16u, o, qj, work4(P.bodyDevOff + 4u * v), Rt * (x - o), Rt * d, Rt * (det.xyz + w - o), lev) || held;
      bodyOn = true;
    }
  }
  if (!bodyOn) { putContact4(b + 16u, vec4<f32>(0.0)); }
  return held;
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let v = gid.x;
  if (v < P.N) {
    let tv = min(bitcast<f32>(atomicExchange(&work[P.tvOff + v], bitcast<u32>(1.0))), 1.0);
    let x = pos[v].xyz;
    let det = xdet[v];
    let w = predOf(v);
    let rel = x - det.xyz;
    let d = dir[v].xyz;
    atomicMax(&wgMaxDir, bitcast<u32>(length(d)));
    let lev = levelled();
    let U = levelU();
    var al = 1.0;
    if (any(d != vec3<f32>(0.0))) {
      var k = cutsOf(v, x, det, w, d, U);
      // A re-bound bound holds the iterate, not the levelled round's anchors: none there.
      if (!lev && min(k.y, k.z) < 1.0) { rebind(v, x, rel, d); k = cutsOf(v, x, det, w, d, U); atomicAdd(&wgRebound, 1u); }
      al = max(min(tv, min(k.x, min(k.y, k.z))), 0.0);
    }
    let heldHere = writeRecords(v, det, w, x, d, lev);
    // The full Newton point, which a levelled round armed after this step predicts the rest of (prep).
    putWork4(P.targetOff + 4u * v, vec4<f32>(x + d, 0.0));
    // A vertex that cannot follow the level (held by a piece or by its records' overflow, or no
    // direction) while it is predicted to move would be left behind its anchor: the level freezes
    // for the iteration. One that is predicted not to move has the same set at every level.
    if (lev && any(w != vec3<f32>(0.0))
      && (heldHere || all(d == vec3<f32>(0.0)) || atomicLoad(&contacts[P.segWord + 2u * v + 1u]) == OGC_NONE)) { atomicAdd(&wgFrozen, 1u); }
    atomicStore(&work[P.recCountOff + v], 0u);
    atomicStore(&alpha[v], bitcast<u32>(al));
    atomicMin(&alpha[P.N], bitcast<u32>(al));
    if (al < 1.0) { atomicAdd(&wgBound, 1u); }
    if (al <= 0.0) { atomicAdd(&wgStalled, 1u); }
  }
  workgroupBarrier();
  if (li == 0u) {
    let c = atomicLoad(&wgBound);
    if (c != 0u) { atomicAdd(&counters[OGC_EXCEED], c); }
    let z = atomicLoad(&wgStalled);
    if (z != 0u) { atomicAdd(&counters[OGC_FILTER_STALLED], z); }
    let rb = atomicLoad(&wgRebound);
    if (rb != 0u) { atomicAdd(&counters[OGC_F_REBOUND], rb); }
    if (atomicLoad(&wgFrozen) != 0u) { atomicOr(&contacts[P.levelWord + 1u], OGC_FROZEN); }
    let md = atomicLoad(&wgMaxDir);
    if (md != 0u) {
      atomicMax(&counters[OGC_F_MAX_DIRECTION], md);
      let lw = roundLogWord();
      if (lw != 0xffffffffu) { atomicMax(&counters[lw + 7u], md); }
    }
  }
}
`;

/** Before a filter: alpha_min = 1 (it is the atomic min of the alpha_v that follow). */
export const OGC_FILTER_BEGIN_WGSL = header([0, 8, 9, 16]) + ROUND_LOG_FNS + /* wgsl */`
@compute @workgroup_size(1)
fn main() {
  if (P.levelWord != 0u) { atomicAnd(&contacts[P.levelWord + 1u], ~OGC_FROZEN); }
  atomicStore(&alpha[P.N], bitcast<u32>(1.0)); atomicStore(&counters[OGC_PLANE_RECORDS], 0u);
  atomicStore(&counters[OGC_PLANE_RECORD_HELD], 0u);
  let b = roundLogWord();
  if (b != 0xffffffffu) { atomicAdd(&counters[b + 5u], 1u); }
}
`;

/**
 * The filter's plane-record segments: each vertex's records (counted at recCountOff by the
 * filter's planes pass) get a contiguous segment (first, count) at segWord, a workgroup's
 * vertices in order after one allocation from OGC_PLANE_RECORDS, and its count goes back to 0
 * for the records pass to fill. Records past the authored capacity fail the step
 * (planeRecordOverflow) and hold their vertex (count OGC_NONE), counted (framePlaneRecordHeld, and
 * the most in one filter); the total's peak is kept.
 */
export const OGC_FILTER_ALLOC_WGSL = header([0, 5, 8, 9]) + ROUND_LOG_FNS + /* wgsl */`
var<workgroup> scan: array<u32, 256>;
var<workgroup> base: u32;
var<workgroup> wgOver: atomic<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let v = gid.x;
  var c = 0u;
  if (v < P.N) { c = atomicLoad(&work[P.recCountOff + v]); }
  if (li == 0u) { atomicStore(&wgOver, 0u); }
  scan[li] = c;
  workgroupBarrier();
  for (var s = 1u; s < 256u; s <<= 1u) {
    var t = 0u;
    if (li >= s) { t = scan[li - s]; }
    workgroupBarrier();
    scan[li] += t;
    workgroupBarrier();
  }
  if (li == 255u && scan[255] != 0u) {
    let b = atomicAdd(&counters[OGC_PLANE_RECORDS], scan[255]);
    base = b;
    atomicMax(&counters[OGC_PEAK_PLANE_RECORDS], b + scan[255]);
    atomicMax(&counters[OGC_F_PEAK_PLANE_RECORDS], b + scan[255]);
    atomicAdd(&counters[OGC_F_PLANE_RECORDS_TOTAL], scan[255]);
    let lw = roundLogWord();
    if (lw != 0xffffffffu) { atomicMax(&counters[lw + 6u], b + scan[255]); atomicAdd(&counters[lw + 11u], scan[255]); }
  }
  workgroupBarrier();
  if (v < P.N) {
    let first = base + scan[li] - c;
    var n = c;
    if (c != 0u && first + c > P.recCap) { n = OGC_NONE; atomicOr(&counters[OGC_STATUS_WORD], OGC_ST_PLANE_RECORD_OVERFLOW); atomicAdd(&wgOver, 1u); }
    atomicStore(&contacts[P.segWord + 2u * v], first);
    atomicStore(&contacts[P.segWord + 2u * v + 1u], n);
    atomicStore(&work[P.recCountOff + v], 0u);
  }
  workgroupBarrier();
  // The vertices this filter holds for overflow: the frame's sum, and the filter's own count (the
  // largest of its running totals is its final one).
  let over = atomicLoad(&wgOver);
  if (li == 0u && over != 0u) {
    atomicAdd(&counters[OGC_F_PLANE_RECORD_HELD], over);
    atomicMax(&counters[OGC_F_PLANE_RECORD_HELD_MAX], atomicAdd(&counters[OGC_PLANE_RECORD_HELD], over) + over);
  }
}
`;

// ------------------------------------------------------------------ grid build

/** Count each face's (inflated) cell span into the hash buckets; reset its d_min. */
export function ogcGridCountWGSL(intersect: boolean): string {
  const bindings = intersect ? [0, 1, 4, 6, 9] : [0, 1, 4, 5, 6, 9, 14];
  return header(bindings) + TOPO_FNS + CELL_FNS + triBoxFns(intersect) + /* wgsl */`
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let t = gid.x;
  ${intersect ? "if (t == 0u) { atomicStore(&counters[OGC_INTERSECTIONS], 0u); atomicStore(&counters[OGC_SOLID_INTERSECTIONS], 0u); atomicStore(&counters[OGC_SOLID_HIT_FACE], 0u); }" : ""}
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
  const bindings = intersect ? [0, 1, 4, 6, 7, 9] : [0, 1, 4, 5, 6, 7, 9, 14];
  return header(bindings) + TOPO_FNS + CELL_FNS + GRID_READ_FNS + triBoxFns(intersect) + /* wgsl */`
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
export function ogcQueryWGSL(role: OgcQueryRole, L: OgcLocalityFlags = parseOgcLocality("off")): string {
  // The locality switch (OGC_LOCALITY_PARTS): off, every fragment below is the baseline's own text.
  const vertexFns = L.geometry ? LOC_GEO_FNS : triBoxFns(false) + "\nfn xd(i: u32) -> vec3<f32> { return xdet[i].xyz; }\n";
  const pairFns = L.geometry ? bodyPairFns(LOC_BODY_ACCESS_FNS) : BODY_PAIR_FNS;
  const faceTri = L.topology ? "faceVerts(t)" : "triVerts(t)";
  const faceMask = L.topology ? "let fr = faceRec(t); let te = fr.flags;" : "let te = topo[P.triEdgesOff + 4u * t + 3u];";
  const lo = role === "queryVertex" ? "0u" : role === "queryEdge" ? "P.N" : "P.N + P.E";
  const hi = role === "queryVertex" ? "P.N" : role === "queryEdge" ? "P.N + P.E" : "P.N + P.E + P.S + P.SB + P.SE + P.SEB + P.SF + P.SFB";
  return header(ogcQueryBindings(L).filter(b => b < 24)) + (L.topology ? DECL_LOC : "") + (L.geometry ? DECL_GEO : "") + LEVEL_FNS + SHIFT_CONSTS + TOPO_FNS + (L.topology ? LOC_TOPO_FNS : "") + CELL_FNS + GRID_READ_FNS + vertexFns + pairFns + SOLID_ACCESS + OGC_SOLID_WGSL + /* wgsl */`

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
  var cause = 0u;
  if (!conflict) {
    let o = xdet[ids[nA]].xyz;
    var supA = 3.4e38; var supB = -3.4e38;
    for (var k = 0u; k < 4u; k++) {
      let pr = dot(pl.xyz, ${L.geometry ? "anchoredLive" : "anchored"}(ids[k]) - o);
      if (k < nA) { supA = min(supA, pr); } else { supB = max(supB, pr); }
    }
    let g1 = supA - supB;
    conflict = g1 < P.sepEps + sg && g1 < pl.w;
    cause = 1u;
  }
  if (!conflict) { return pl; }
  let alt = predictedAxis(ids, nA, sg);
  if (alt.w > 0.0) { return alt; }
  atomicAdd(&counters[OGC_CONFLICTS], 1u);
  // A levelled re-detection's conflict redoes its round unlevelled (OGC_REDO_WGSL).
  if (atomicLoad(&counters[OGC_ROUNDS]) != 0u && levelled()) {
    var solid = false;
    for (var k = 0u; k < 4u; k++) { if (ids[k] >= P.N) { solid = true; } }
    atomicAdd(&counters[OGC_F_LEVEL_CONFLICTS + cause], select(1u, 65537u, solid));
    return pl;
  }
  let redone = atomicLoad(&counters[OGC_FORCE_HELD]);
  if (!failSafeRound()) { leastShift(ids, nA, pl, sg); return pl; }
  // The fail-safe (redone past the least-shift redos): its cloth is held, or with a moving
  // solid in the pair (its sample, or cloth already carried with it: bits 2 and 6), carried
  // rigidly with the solid: an impact zone, which the next redo grows. Cloth that only touched
  // the solid (bit 4) is held like any other: a held vertex the solid reaches is still pushed by
  // its bound, and carrying every touched vertex's cloth pairs grew the curtain's zone over the
  // pile on the floor (10k vertices, heldConflict at tick 126). A static sample's pair cannot be
  // carried; it conflicts again until the last redo fails.
  var carry = false;
  if (P.bodyBase != 0u) {
    var stat = false;
    for (var k = 0u; k < 4u; k++) {
      let i = ids[k];
      if (i >= P.N + P.S && i < P.N + P.S + P.SB) { carry = true; }
      else if (i >= P.N) { stat = true; }
      else if ((atomicLoad(&work[P.heldOff + i]) & 68u) != 0u) { carry = true; }
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
  if (!progress || redone == redoCap() || P.bodyBase == 0u) {
    recordHeldConflict(select(ids, vec4<u32>(ids.x | OGC_EE_BIT, ids.yzw), nA == 2u));
  }
  return pl;
}
// The least shift of a conflicting pair: its cloth's predictions move back along straight paths
// towards an anchor c, Y(t) = X + c + t (w - c), only as far as the pair needs: c = a_v (moving
// with the moving solid: the relative motion scaled in the solid's frame, where the pair's plane is
// fixed) for a pair with one of the solid's samples or for cloth anchored with it (bit 5), else
// X (c = 0). Along the pair's plane axis its support gap is concave in t (the least of linear
// functions less the largest), so the largest t keeping it at T, between the conflict threshold
// sepEps + sg and the lesser of the gap at X and 2 sepEps + sg, is a bisection. Each vertex keeps
// the least scale of its pairs (atomicMin of non-negative f32 bits, the next redo's s_v = s t);
// a vertex shifted for one pair can come nearer another, which that redo finds and shifts in
// turn. A pair with the solid asks its cloth to anchor at a_v (bit 7) in the next redo.
//
// (Measured on the Bridson curtain: an anchor at the pair's mean prediction, which keeps a
// sheet's common motion, made 28k conflicts by tick 26 where X made none to tick 30.)
fn bodyStep(i: u32) -> vec3<f32> {
  return bitcast<vec3<f32>>(vec3<u32>(atomicLoad(&work[P.bodyOff + 8u * i + 4u]), atomicLoad(&work[P.bodyOff + 8u * i + 5u]), atomicLoad(&work[P.bodyOff + 8u * i + 6u])));
}
fn shiftGap(ids: vec4<u32>, nA: u32, n: vec3<f32>, anc: array<vec3<f32>, 4>, t: f32) -> f32 {
  let o = xdet[ids[nA]].xyz;
  var sa = 3.4e38; var sb = -3.4e38;
  for (var k = 0u; k < 4u; k++) {
    let i = ids[k];
    var y = ${L.geometry ? "anchoredLive" : "anchored"}(i);
    if (i < P.N) { y = xdet[i].xyz + anc[k] + t * (${L.geometry ? "predLive" : "predOf"}(i) - anc[k]); }
    let pr = dot(n, y - o);
    if (k < nA) { sa = min(sa, pr); } else { sb = max(sb, pr); }
  }
  return sa - sb;
}
fn leastShift(ids: vec4<u32>, nA: u32, pl: vec4<f32>, sg: f32) {
  var body = false;
  for (var k = 0u; k < 4u; k++) { if (ids[k] >= P.N + P.S && ids[k] < P.N + P.S + P.SB) { body = true; } }
  var anc: array<vec3<f32>, 4>;
  for (var k = 0u; k < 4u; k++) {
    let i = ids[k];
    anc[k] = vec3<f32>(0.0);
    if (i < P.N && P.bodyBase != 0u && (body || (atomicLoad(&work[P.heldOff + i]) & 32u) != 0u)) { anc[k] = bodyStep(i); }
  }
  let t0 = P.sepEps + sg;
  var t = 0.0;
  if (pl.w > t0) {
    let goal = 0.5 * (t0 + min(pl.w, 2.0 * P.sepEps + sg));
    if (shiftGap(ids, nA, pl.xyz, anc, 0.0) >= goal) {
      var lo = 0.0; var hi = 1.0;
      for (var it = 0u; it < 24u; it++) {
        let mid = 0.5 * (lo + hi);
        if (shiftGap(ids, nA, pl.xyz, anc, mid) >= goal) { lo = mid; } else { hi = mid; }
      }
      t = lo;
    }
  }
  var progress = false;
  for (var k = 0u; k < 4u; k++) {
    let i = ids[k];
    if (i >= P.N) { continue; }
    let b = P.scaleOff + OGC_SHIFT_WORDS * i;
    if (body && P.bodyBase != 0u && (atomicLoad(&work[P.heldOff + i]) & 32u) == 0u) {
      if ((atomicOr(&work[P.heldOff + i], 128u) & 128u) == 0u) { progress = true; }
    }
    let sn = bitcast<u32>(bitcast<f32>(atomicLoad(&work[b + 1u])) * t);
    if (atomicMin(&work[b], sn) > sn) { progress = true; }
  }
  if (progress) { atomicAdd(&counters[OGC_REDO_PROGRESS], 1u); }
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

// The cloth-cloth DAT test (E12). A round moves every vertex along one shared parameter s in [0, 1]:
// the time of the straight motion X -> x1 in an unlevelled round, the level of the joint path in a
// levelled one (ogcProjectTrial). At s each vertex lies within its cap, 0.5 gamma_r env_v, of its
// chord point X + s w (the cap ball's centre moves affinely with s, and the joint path is convex),
// so a face or edge point within its corners' widest cap of the same point on the predicted
// primitive. Two primitives therefore stay apart over the round when, at every s, their predicted
// primitives are further apart than the two caps (and each vertex's stray from its world chord):
// (i) at X, d > caps + strays + the spread L = max |w_i - w_j| over the pair (the relative motion
// closes the distance at most L s); (ii) on the predicted paths, the bound datPathVT / datPathEE.
// The threshold adds sepEps, the gap paired primitives keep, and datRound, the f32 error of the
// distances (128 ulps of the coordinates' scale), so a dropped pair keeps at least sepEps.
fn datCaps(a: f32, b: f32) -> f32 { return 0.5 * P.gammaR * (a + b) + P.sepEps; }
fn datRound(s: f32) -> f32 { return 1.52587890625e-5 * s; }
fn absMax(p: vec3<f32>) -> f32 { return max(abs(p.x), max(abs(p.y), abs(p.z))); }
fn pathStray(i: u32) -> f32 { return strayOf(i, false); }
// Exact distances for the path bound; a degenerate segment is its point, a zero-area triangle the
// union of its edges.
fn datPointSeg(p: vec3<f32>, q0: vec3<f32>, q1: vec3<f32>) -> f32 {
  let d = q1 - q0; let e = dot(d, d);
  let t = select(0.0, clamp(dot(p - q0, d) / e, 0.0, 1.0), e > 0.0);
  return length(p - (q0 + t * d));
}
fn datSegSeg(p0: vec3<f32>, p1: vec3<f32>, q0: vec3<f32>, q1: vec3<f32>) -> f32 {
  if (!(dot(p1 - p0, p1 - p0) > 0.0)) { return datPointSeg(p0, q0, q1); }
  if (!(dot(q1 - q0, q1 - q0) > 0.0)) { return datPointSeg(q0, p0, p1); }
  return ogcSegmentSegment(p0, p1, q0, q1).z;
}
fn datPointTri(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> f32 {
  let n = cross(b - a, c - a);
  if (!(dot(n, n) > 0.0)) { return min(datPointSeg(p, a, b), min(datPointSeg(p, b, c), datPointSeg(p, c, a))); }
  let cl = ogcClosestTriangle(p, a, b, c);
  return length(p - (cl.bary.x * a + cl.bary.y * b + cl.bary.z * c));
}
// (ii) The distance g(s) of the predicted primitives, every vertex at X + s w, is L-Lipschitz in s
// (for fixed barycentric weights the difference of the two points moves at a convex mix of the
// w_i - w_j), so on [k/K, (k+1)/K] it is at least (g_k + g_k+1) / 2 - L / 2K. K levels make that
// sampling slack at most a tenth of the threshold (K <= 32; the bound stays sound beyond).
fn datLevels(L: f32, thr: f32) -> u32 { return u32(clamp(ceil(5.0 * L / max(thr, 1.0e-30)), 1.0, 32.0)); }
fn datPathVT(x: vec3<f32>, v: u32, tri: vec3<u32>, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>, L: f32, thr: f32) -> f32 {
  let wv = predOf(v); let w0 = predOf(tri.x); let w1 = predOf(tri.y); let w2 = predOf(tri.z);
  let K = datLevels(L, thr);
  var prev = datPointTri(x, a, b, c);
  var lo = 3.4e38;
  for (var k = 1u; k <= K; k++) {
    let t = f32(k) / f32(K);
    let dk = datPointTri(x + t * wv, a + t * w0, b + t * w1, c + t * w2);
    lo = min(lo, 0.5 * (prev + dk));
    prev = dk;
  }
  return lo - L / (2.0 * f32(K));
}
fn datPathEE(ep: vec4<u32>, eq: vec4<u32>, p0: vec3<f32>, p1: vec3<f32>, q0: vec3<f32>, q1: vec3<f32>, L: f32, thr: f32) -> f32 {
  let wp0 = predOf(ep.x); let wp1 = predOf(ep.y); let wq0 = predOf(eq.x); let wq1 = predOf(eq.y);
  let K = datLevels(L, thr);
  var prev = datSegSeg(p0, p1, q0, q1);
  var lo = 3.4e38;
  for (var k = 1u; k <= K; k++) {
    let t = f32(k) / f32(K);
    let dk = datSegSeg(p0 + t * wp0, p1 + t * wp1, q0 + t * wq0, q1 + t * wq1);
    lo = min(lo, 0.5 * (prev + dk));
    prev = dk;
  }
  return lo - L / (2.0 * f32(K));
}
// Whether v and face tri (at X: x; a, b, c) need a DAT pair: (i), then (ii) (a NaN keeps the pair).
fn datPairVT(x: vec3<f32>, d: f32, v: u32, tri: vec3<u32>, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>, rv: f32) -> bool {
  let L = spreadVT(v, tri);
  let wmax = max(absMax(predOf(v)), max(absMax(predOf(tri.x)), max(absMax(predOf(tri.y)), absMax(predOf(tri.z)))));
  let st = pathStray(v) + max(pathStray(tri.x), max(pathStray(tri.y), pathStray(tri.z)));
  let thr = datCaps(rv, triEnv(tri)) + st + datRound(max(absMax(x), max(absMax(a), max(absMax(b), absMax(c)))) + wmax);
  if (d > thr + L) { return false; }
  return !(datPathVT(x, v, tri, a, b, c, L, thr) > thr);
}
fn datPairEE(er: vec4<u32>, r2: vec4<u32>, d: f32, p0: vec3<f32>, p1: vec3<f32>, q0: vec3<f32>, q1: vec3<f32>, re: f32, re2: f32) -> bool {
  let L = spreadEE(er, r2);
  let wmax = max(max(absMax(predOf(er.x)), absMax(predOf(er.y))), max(absMax(predOf(r2.x)), absMax(predOf(r2.y))));
  let st = max(pathStray(er.x), pathStray(er.y)) + max(pathStray(r2.x), pathStray(r2.y));
  let thr = datCaps(re, re2) + st + datRound(max(max(absMax(p0), absMax(p1)), max(absMax(q0), absMax(q1))) + wmax);
  if (d > thr + L) { return false; }
  return !(datPathEE(er, r2, p0, p1, q0, q1, L, thr) > thr);
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

struct RoleOut { candidates: u32, pairs: u32, dat: u32, scanned: u32, collisions: u32, rejected: u32 }

fn vertexFacet(v: u32) -> RoleOut {
  var out = RoleOut(0u, 0u, 0u, 0u, 0u, 0u);
  let planar = P.mode == OGC_PLANAR;
  let x = xd(v);
  let rv = envOf(v);
  let x1 = x + predOf(v);
  // Its swept box grown by its own envelope (local swept bounds, triBox).
  let q = OgcBox(min(x, x1) - vec3<f32>(rv), max(x, x1) + vec3<f32>(rv));
  let inv = invCell();
  let c0 = cellOf(q.lo, inv); let c1 = cellOf(q.hi, inv);
  var dmin = 3.4e38;
  for (var cz = c0.z; cz <= c1.z; cz++) { for (var cy = c0.y; cy <= c1.y; cy++) { for (var cx = c0.x; cx <= c1.x; cx++) {
    let c = vec3<i32>(cx, cy, cz);
    let key = cellKey(c); let h = cellHash(c);
    let end = min(bucketStart(h + 1u), P.gridCap);
    for (var j = bucketStart(h); j < end; j++) {
      out.scanned++;
      if (entryKey(j) != key) { out.collisions++; continue; }
      let t = entryFace(j);
      let tri = ${faceTri};
      if (hasVertex(tri, v)) { continue; }
      let box = entryBox(j);
      if (!boxesOverlap(q, box) || any(cellOf(max(q.lo, box.lo), inv) != c)) { out.rejected++; continue; }
      out.candidates++;
      let a = xd(tri.x); let b = xd(tri.y); let cc = xd(tri.z);
      let cl = ogcClosestTriangle(x, a, b, cc);
      let cp = cl.bary.x * a + cl.bary.y * b + cl.bary.z * cc;
      let d = length(x - cp);
      dmin = min(dmin, d);
      if (planar) {
        if (datPairVT(x, d, v, tri, a, b, cc, rv)) { pairVT(v, tri, x, cp); out.dat++; }
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
// carry their predictions and their own envelopes, so the sample's bare swept box (a
// point for a static sample; to its exact predicted end for the moving solid's; its
// motion is exact, envelope 0) finds every face within its envelope of the sample. Entries go to the face's feature
// vertices only (the sample is not solved for) and carry mu lambda with mu the solid's
// at the sample (x_det.w); w = 1 marks them. The DAT pair puts the sample first, as a vertex.
fn sampleFacet(i: u32) -> RoleOut {
  var out = RoleOut(0u, 0u, 0u, 0u, 0u, 0u);
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
    out.scanned++;
    if (entryKey(j) != key) { out.collisions++; continue; }
    let box = entryBox(j);
    if (!boxesOverlap(q, box) || any(cellOf(max(q.lo, box.lo), inv) != c)) { out.rejected++; continue; }
    let t = entryFace(j);
    let tri = ${faceTri};
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
  var out = RoleOut(0u, 0u, 0u, 0u, 0u, 0u);
  let planar = P.mode == OGC_PLANAR;
  let er = edgeRec(e);
  let p0 = xd(er.x); let p1 = xd(er.y);
  let re = max(envOf(er.x), envOf(er.y));
  let p01 = p0 + predOf(er.x); let p11 = p1 + predOf(er.y);
  let q = OgcBox(min(min(p0, p1), min(p01, p11)) - vec3<f32>(re), max(max(p0, p1), max(p01, p11)) + vec3<f32>(re));
  let inv = invCell();
  let c0 = cellOf(q.lo, inv); let c1 = cellOf(q.hi, inv);
  var dmin = 3.4e38;
  for (var cz = c0.z; cz <= c1.z; cz++) { for (var cy = c0.y; cy <= c1.y; cy++) { for (var cx = c0.x; cx <= c1.x; cx++) {
    let c = vec3<i32>(cx, cy, cz);
    let key = cellKey(c); let h = cellHash(c);
    let end = min(bucketStart(h + 1u), P.gridCap);
    for (var j = bucketStart(h); j < end; j++) {
      out.scanned++;
      if (entryKey(j) != key) { out.collisions++; continue; }
      let t = entryFace(j);
      let box = entryBox(j);
      if (!boxesOverlap(q, box) || any(cellOf(max(q.lo, box.lo), inv) != c)) { out.rejected++; continue; }
      ${faceMask}
      for (var k = 0u; k < 3u; k++) {
        if ((te & (1u << k)) == 0u) { continue; }
        ${L.topology ? "let e2 = fr.e[k]; let r2 = faceEdge(fr, k);" : "let e2 = topo[P.triEdgesOff + 4u * t + k];\n        let r2 = edgeRec(e2);"}
        if (r2.x == er.x || r2.x == er.y || r2.y == er.x || r2.y == er.y) { continue; }
        out.candidates++;
        let q0 = xd(r2.x); let q1 = xd(r2.y);
        // Canonical order, so both edges' threads compute bitwise the same pair.
        var st: vec3<f32>;
        if (e < e2) { st = ogcSegmentSegment(p0, p1, q0, q1); }
        else { let w = ogcSegmentSegment(q0, q1, p0, p1); st = vec3<f32>(w.y, w.x, w.z); }
        dmin = min(dmin, st.z);
        if (planar && e < e2 && datPairEE(er, r2, st.z, p0, p1, q0, q1, re, max(envOf(r2.x), envOf(r2.y)))) {
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
  var out = RoleOut(0u, 0u, 0u, 0u, 0u, 0u);
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
      out.scanned++;
      if (entryKey(j) != key) { out.collisions++; continue; }
      let t = entryFace(j);
      let box = entryBox(j);
      if (!boxesOverlap(q, box) || any(cellOf(max(q.lo, box.lo), inv) != c)) { out.rejected++; continue; }
      ${faceMask}
      for (var m = 0u; m < 3u; m++) {
        if ((te & (1u << m)) == 0u) { continue; }
        let r2 = ${L.topology ? "faceEdge(fr, m)" : "edgeRec(topo[P.triEdgesOff + 4u * t + m])"};
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

// A hull facet of the samples (solid-field.ts clothSolidSampleFacets: the triangles of their local hull, wound outward,
// each on or outside the source's surface; the static solid's, then the moving solid's) against the cloth's vertices.
// The samples' own barrier (sampleFacet) holds a cloth face off each sample and the hull edges' pairs keep cloth edges
// from passing between two (hullEdge); a cloth vertex can still pass into the hole a facet spans (the field rounding
// the solid's convex feature, there by up to the ceiling), and this closes it: with the samples' faces and edges, every
// first contact of the cloth with the samples' hull (vertex-facet, facet corner-cloth face, hull edge-cloth edge) is
// held. The vertices are reached through their faces' grid entries (a face's box holds its vertices' swept boxes and
// envelopes), each once, from its owner face (the lowest-index face holding it, the first of its ascending vtList):
// a DAT pair (the vertex first, the facet's samples as the face) within the vertex's envelope plus the pair's spread
// (the samples' motion is exact), and an entry for the vertex alone within r, the facet's slot (its index, past
// P.facetCap for the moving solid's) and the closest feature as a face's, mu lambda with mu the samples' mixed at the
// closest point, w = 2. The bound pass keeps one static representation per vertex and solid (OGC_BOUND_WGSL).
fn hullFacetAt(k: u32) -> vec4<u32> {
  var b = 4u * (P.facetBase + k);
  var off = P.N; var slot = k;
  if (k >= P.SF) { b = 4u * (P.bodyFacetBase + k - P.SF); off = P.N + P.S; slot = P.facetCap + k - P.SF; }
  return vec4<u32>(atomicLoad(&contacts[b]) + off, atomicLoad(&contacts[b + 1u]) + off, atomicLoad(&contacts[b + 2u]) + off, slot);
}
fn hullFacet(k: u32) -> RoleOut {
  var out = RoleOut(0u, 0u, 0u, 0u, 0u, 0u);
  let planar = P.mode == OGC_PLANAR;
  let hf = hullFacetAt(k);
  let ids = hf.xyz;
  let a = xd(ids.x); let b = xd(ids.y); let cc = xd(ids.z);
  let a1 = a + predOf(ids.x); let b1 = b + predOf(ids.y); let c1 = cc + predOf(ids.z);
  let q = OgcBox(min(min(min(a, b), cc), min(min(a1, b1), c1)), max(max(max(a, b), cc), max(max(a1, b1), c1)));
  let mu = vec3<f32>(xdet[ids.x].w, xdet[ids.y].w, xdet[ids.z].w);
  let mark = select(0u, 16u, k >= P.SF);
  let inv = invCell();
  let c0 = cellOf(q.lo, inv); let cE = cellOf(q.hi, inv);
  for (var cz = c0.z; cz <= cE.z; cz++) { for (var cy = c0.y; cy <= cE.y; cy++) { for (var cx = c0.x; cx <= cE.x; cx++) {
    let c = vec3<i32>(cx, cy, cz);
    let key = cellKey(c); let h = cellHash(c);
    let end = min(bucketStart(h + 1u), P.gridCap);
    for (var j = bucketStart(h); j < end; j++) {
      out.scanned++;
      if (entryKey(j) != key) { out.collisions++; continue; }
      let box = entryBox(j);
      if (!boxesOverlap(q, box) || any(cellOf(max(q.lo, box.lo), inv) != c)) { out.rejected++; continue; }
      let t = entryFace(j);
      ${L.topology ? "let fr = faceRec(t); let tri = fr.v;" : "let tri = triVerts(t);"}
      for (var m = 0u; m < 3u; m++) {
        let v = tri[m];
        if (${L.topology ? "(fr.flags & (64u << m)) == 0u" : "topo[P.vtListOff + topo[P.vtOffOff + v]] != t"}) { continue; }
        out.candidates++;
        let x = xd(v);
        let cl = ogcClosestTriangle(x, a, b, cc);
        let cp = cl.bary.x * a + cl.bary.y * b + cl.bary.z * cc;
        let d = length(x - cp);
        if (planar) {
          if (d <= envOf(v) + spreadVT(v, ids)) { pairVT(v, ids, x, cp); out.dat++; }
        } else {
          atomicMin(&work[v], bitcast<u32>(d));
        }
        if (!(d > 0.0)) { atomicOr(&counters[OGC_STATUS_WORD], OGC_ST_COINCIDENT); }
        else if (d < P.r) {
          let lambda = -ogcActivation(d, P.r, P.kc).y * dot(cl.bary, mu);
          pushSampleEntry(v, vec4<u32>(v, hf.w | (cl.feature << 28u), bitcast<u32>(lambda), 2u), mark);
          out.pairs++;
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
var<workgroup> wgHf: atomic<u32>;
var<workgroup> wgScan: atomic<u32>;
var<workgroup> wgColl: atomic<u32>;
var<workgroup> wgRej: atomic<u32>;
var<workgroup> wgScanMax: atomic<u32>;
fn tally(o: RoleOut) {
  atomicAdd(&wgCand, o.candidates); atomicAdd(&wgDat, o.dat);
  atomicAdd(&wgScan, o.scanned); atomicAdd(&wgColl, o.collisions); atomicAdd(&wgRej, o.rejected); atomicMax(&wgScanMax, o.scanned);
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let roleLo = ${lo}; let roleHi = ${hi};
  if (64u * wg.x + 64u <= roleLo || 64u * wg.x >= roleHi) { return; }
  let i = gid.x;
  if (i >= roleLo && i < roleHi) {
    ${role === "queryVertex" ? `let o = vertexFacet(i);
    tally(o); atomicAdd(&wgVf, o.pairs);`
    : role === "queryEdge" ? `let o = edgeEdge(i - P.N);
    tally(o); atomicAdd(&wgEe, o.pairs);`
    : `if (i < P.N + P.E + P.S + P.SB) {
      let o = sampleFacet(i - P.E);
      tally(o); atomicAdd(&wgSf, o.pairs);
    } else if (i < P.N + P.E + P.S + P.SB + P.SE + P.SEB) {
      tally(hullEdge(i - P.N - P.E - P.S - P.SB));
    } else {
      let o = hullFacet(i - P.N - P.E - P.S - P.SB - P.SE - P.SEB);
      tally(o); atomicAdd(&wgHf, o.pairs);
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
    atomicAdd(&counters[OGC_F_HF_PAIRS], atomicLoad(&wgHf));
    atomicAdd(&counters[OGC_F_SCANNED], atomicLoad(&wgScan));
    atomicAdd(&counters[OGC_F_COLLISIONS], atomicLoad(&wgColl));
    atomicAdd(&counters[OGC_F_BOX_REJECTED], atomicLoad(&wgRej));
    atomicMax(&counters[OGC_F_SCAN_MAX], atomicLoad(&wgScanMax));
  }
}
`;
}

// ------------------------------------------------------------------ end-of-step verification

/**
 * The end-of-step verification's arming (one thread, after each Newton iteration; Claude's ruling B on C2 RESULT 46 (1)).
 * OGC's certificates guarantee no penetration over the round, not that every pair that comes within r at the iterate has
 * an entry: a pair beyond r at the round's x_det can close inside r within the two caps. So whenever Newton would declare
 * done (its done word 2, a candidate) the verify pass runs one radius-r query at the iterate; and at the step's last
 * iteration (`last`: the solver's flag to afterNewtonStep, n = newton.iterations - 1) it runs when Newton is not done, to count the pairs a
 * capped step leaves without an entry. The mode word tells the decision which (OGC_VERIFY_MODE).
 */
export function ogcVerifyArgsWGSL(last: boolean): string {
  return header([0, 9, 19, 20]) + /* wgsl */`
@compute @workgroup_size(1)
fn main() {
  let done = nwctl[0].x;
  var mode = 0u;
  if (done == 2.0) { mode = ${last ? "3u" : "1u"}; }
  ${last ? "else if (done == 0.0) { mode = 2u; }" : ""}
  atomicStore(&counters[OGC_VERIFY_MODE], mode);
  atomicStore(&counters[OGC_VERIFY_MISSING], 0u);
  vargs[0] = select(0u, P.wgNE, mode != 0u); vargs[1] = 1u; vargs[2] = 1u;
}
`;
}

/**
 * The end-of-step verification: one query over the round's grid at radius r, at the iterate (the solver's positions; a
 * sample at its round's predicted end x_det + w, exact: the moving solid's end pose), with no DAT pair, plane or entry.
 * Each pair within r (0 < d < r) that OGC's detection at the iterate would record (Algorithm 1's blocks and ownership
 * at the iterate; Algorithm 2's interior edge-edge pairs, counted once from the lower edge) must have an entry in the
 * round's lists that Newton evaluates at a distance no larger than the pair's own:
 *   vertex-face   an entry (v, t', feature F) of v with the iterate's closest feature f of t a subset of F's vertices
 *                 (an entry is evaluated against its feature's affine hull, so F containing f is never further than f);
 *   sample-face   likewise (the sample, w = 1), in the list of f's first vertex (every feature vertex holds it);
 *   edge-edge     the entry (e, e2) of e's first vertex;
 *   hull facet    any entry of the facet's solid in the vertex's list (one representation per solid: its field entry or
 *                 its nearest facet, OGC_BOUND_WGSL ownSolids).
 * Field entries are taken for every vertex within reach of the solid's band at detection (roundBound), and need none.
 * The grid holds each face's round box, its swept box X -> X + w grown by its envelope; the round's cap keeps every
 * cloth vertex within 0.45 of its envelope of its chord, so a face within r of a primitive is reached by the
 * primitive's box grown by r. A vertex found outside its own swept box grown by its envelope (the hash does not cover
 * it) counts as missing (frameUncovered), so the decision re-detects about it. Hull edges have DAT pairs only: none.
 */
export const OGC_VERIFY_WGSL = header([0, 1, 2, 4, 5, 7, 8, 9, 14]) + TOPO_FNS + CELL_FNS + GRID_READ_FNS + triBoxFns(false) + SOLID_ACCESS + OGC_SOLID_WGSL + /* wgsl */`
// The iterate: a cloth vertex's position, a sample's predicted end.
fn cur(i: u32) -> vec3<f32> { if (i < P.N) { return pos[i].xyz; } return anchored(i); }
fn ldEnt(u: u32, k: u32) -> vec4<u32> {
  let b = P.entriesBase + 4u * (u * P.contactCap + k);
  return vec4<u32>(atomicLoad(&contacts[b]), atomicLoad(&contacts[b + 1u]), atomicLoad(&contacts[b + 2u]), atomicLoad(&contacts[b + 3u]));
}
fn entCount(u: u32) -> u32 { return min(atomicLoad(&contacts[P.countsOff + u]), P.contactCap); }
// A feature of a face as its vertices (unused slots OGC_NONE): vertex k < 3, edge 3 + k (k, k + 1), the face.
fn featVerts(tri: vec3<u32>, f: u32) -> vec3<u32> {
  if (f < 3u) { return vec3<u32>(tri[f], OGC_NONE, OGC_NONE); }
  if (f < OGC_FACE) { let li = f - 3u; return vec3<u32>(tri[li], tri[(li + 1u) % 3u], OGC_NONE); }
  return tri;
}
fn within(a: vec3<u32>, b: vec3<u32>) -> bool {
  for (var k = 0u; k < 3u; k++) { if (a[k] != OGC_NONE && a[k] != b.x && a[k] != b.y && a[k] != b.z) { return false; } }
  return true;
}
// An entry of u's list for the point p (a cloth vertex, w = 0; a sample, w = 1) against the feature f of a cloth face.
fn hasFeatureEntry(u: u32, p: u32, w: u32, f: vec3<u32>) -> bool {
  for (var k = 0u; k < entCount(u); k++) {
    let e = ldEnt(u, k);
    if (e.x != p || e.w != w || (e.y & OGC_EE_BIT) != 0u || e.y == OGC_SOLID_Y || e.y == OGC_BODY_Y) { continue; }
    if (within(f, featVerts(triVerts(e.y & OGC_ID_MASK), (e.y >> 28u) & 7u))) { return true; }
  }
  return false;
}
// Algorithm 1's blocks and ownership (vfAccept) at the iterate.
fn vfAcceptAt(v: u32, x: vec3<f32>, t: u32, tri: vec3<u32>, feature: u32) -> bool {
  if (feature == OGC_FACE) { return true; }
  if (feature < 3u) {
    let k = tri[feature];
    let xk = cur(k);
    for (var i = topo[P.vtOffOff + k]; i < topo[P.vtOffOff + k + 1u]; i++) {
      let f = topo[P.vtListOff + i];
      if (f < t && !hasVertex(triVerts(f), v)) { return false; }
    }
    for (var i = topo[P.veOffOff + k]; i < topo[P.veOffOff + k + 1u]; i++) {
      let er = edgeRec(topo[P.veListOff + i]);
      let o = select(er.x, er.y, er.x == k);
      if (dot(x - xk, xk - cur(o)) < 0.0) { return false; }
    }
    return true;
  }
  let li = feature - 3u;
  let i = tri[li]; let j = tri[(li + 1u) % 3u];
  let er = edgeRec(topo[P.triEdgesOff + 4u * t + li]);
  let other = select(er.z, er.w, er.z == t);
  if (other != OGC_NONE && other < t && !hasVertex(triVerts(other), v)) { return false; }
  let xi = cur(i); let xj = cur(j); let dir = xj - xi;
  if (dot(x - xi, dir) <= 0.0 || dot(x - xj, -dir) <= 0.0) { return false; }
  for (var s = 0u; s < 2u; s++) {
    let f = select(er.z, er.w, s == 1u);
    if (f == OGC_NONE) { continue; }
    let fv = triVerts(f);
    var o = fv.x;
    if (o == i || o == j) { o = fv.y; }
    if (o == i || o == j) { o = fv.z; }
    let xo = cur(o);
    let p = xi + dir * (dot(xo - xi, dir) / dot(dir, dir));
    if (dot(x - p, p - xo) < 0.0) { return false; }
  }
  return true;
}
// A point (cloth vertex v, or sample p with v = OGC_NONE) against the faces its r-box reaches.
fn pointFaces(p: u32, v: u32) -> u32 {
  var missing = 0u;
  let x = cur(p);
  let q = OgcBox(x - vec3<f32>(P.r), x + vec3<f32>(P.r));
  let inv = invCell();
  let c0 = cellOf(q.lo, inv); let c1 = cellOf(q.hi, inv);
  for (var cz = c0.z; cz <= c1.z; cz++) { for (var cy = c0.y; cy <= c1.y; cy++) { for (var cx = c0.x; cx <= c1.x; cx++) {
    let c = vec3<i32>(cx, cy, cz);
    let key = cellKey(c); let h = cellHash(c);
    let end = min(bucketStart(h + 1u), P.gridCap);
    for (var j = bucketStart(h); j < end; j++) {
      if (entryKey(j) != key) { continue; }
      let t = entryFace(j);
      let tri = triVerts(t);
      if (v != OGC_NONE && hasVertex(tri, v)) { continue; }
      let box = entryBox(j);
      if (!boxesOverlap(q, box) || any(cellOf(max(q.lo, box.lo), inv) != c)) { continue; }
      let a = cur(tri.x); let b = cur(tri.y); let cc = cur(tri.z);
      let cl = ogcClosestTriangle(x, a, b, cc);
      let d = length(x - (cl.bary.x * a + cl.bary.y * b + cl.bary.z * cc));
      if (!(d > 0.0) || !(d < P.r) || !vfAcceptAt(v, x, t, tri, cl.feature)) { continue; }
      let f = featVerts(tri, cl.feature);
      var ok = false;
      if (v != OGC_NONE) { ok = hasFeatureEntry(v, v, 0u, f); } else { ok = hasFeatureEntry(f.x, p, 1u, f); }
      if (!ok) { missing++; }
    }
  } } }
  return missing;
}
// Algorithm 2 at the iterate: interior edge-edge pairs within r, each counted from its lower edge.
fn edgeEdges(e: u32) -> u32 {
  var missing = 0u;
  let er = edgeRec(e);
  let p0 = cur(er.x); let p1 = cur(er.y);
  let q = OgcBox(min(p0, p1) - vec3<f32>(P.r), max(p0, p1) + vec3<f32>(P.r));
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
      for (var k = 0u; k < 3u; k++) {
        if ((te & (1u << k)) == 0u) { continue; }
        let e2 = topo[P.triEdgesOff + 4u * t + k];
        if (e2 <= e) { continue; }
        let r2 = edgeRec(e2);
        if (r2.x == er.x || r2.x == er.y || r2.y == er.x || r2.y == er.y) { continue; }
        let st = ogcSegmentSegment(p0, p1, cur(r2.x), cur(r2.y));
        if (!(st.z > 0.0) || !(st.z < P.r) || !(st.x > 0.0 && st.x < 1.0 && st.y > 0.0 && st.y < 1.0)) { continue; }
        var ok = false;
        for (var m = 0u; m < entCount(er.x); m++) {
          let ent = ldEnt(er.x, m);
          if (ent.x == e && ent.y == (e2 | OGC_EE_BIT)) { ok = true; break; }
        }
        if (!ok) { missing++; }
      }
    }
  } } }
  return missing;
}
// A hull facet of the samples (hullFacetAt's slot k) against the cloth vertices its r-box reaches, each from its owner face.
fn facetVertices(k: u32) -> u32 {
  var missing = 0u;
  var b = 4u * (P.facetBase + k); var off = P.N; var g = 0u;
  if (k >= P.SF) { b = 4u * (P.bodyFacetBase + k - P.SF); off = P.N + P.S; g = 1u; }
  let ids = vec3<u32>(atomicLoad(&contacts[b]), atomicLoad(&contacts[b + 1u]), atomicLoad(&contacts[b + 2u])) + vec3<u32>(off);
  let a = cur(ids.x); let bb = cur(ids.y); let cc = cur(ids.z);
  let q = OgcBox(min(min(a, bb), cc) - vec3<f32>(P.r), max(max(a, bb), cc) + vec3<f32>(P.r));
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
      for (var m = 0u; m < 3u; m++) {
        let v = tri[m];
        if (topo[P.vtListOff + topo[P.vtOffOff + v]] != t) { continue; }
        let x = cur(v);
        let cl = ogcClosestTriangle(x, a, bb, cc);
        let d = length(x - (cl.bary.x * a + cl.bary.y * bb + cl.bary.z * cc));
        if (!(d > 0.0) || !(d < P.r)) { continue; }
        var ok = false;
        for (var n = 0u; n < entCount(v); n++) {
          let e = ldEnt(v, n);
          if ((g == 0u && e.y == OGC_SOLID_Y) || (g == 1u && e.y == OGC_BODY_Y)) { ok = true; break; }
          if (e.w == 2u && (e.y & OGC_EE_BIT) == 0u && e.y != OGC_SOLID_Y && e.y != OGC_BODY_Y
              && select(0u, 1u, (e.y & OGC_ID_MASK) >= P.facetCap) == g) { ok = true; break; }
        }
        if (!ok) { missing++; }
      }
    }
  } } }
  return missing;
}
// A cloth vertex outside its round box (its swept box X -> X + w grown by its envelope): the grid does not cover it.
fn uncovered(v: u32) -> bool {
  let x = xdet[v].xyz; let x1 = x + predOf(v); let e = vec3<f32>(envOf(v)); let y = cur(v);
  return any(y < min(x, x1) - e) || any(y > max(x, x1) + e);
}
var<workgroup> wgMissing: atomic<u32>;
var<workgroup> wgUncovered: atomic<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let i = gid.x;
  var missing = 0u;
  if (i < P.N) {
    if (uncovered(i)) { atomicAdd(&wgUncovered, 1u); missing = 1u; }
    else { missing = pointFaces(i, i); }
  } else if (i < P.N + P.E) {
    missing = edgeEdges(i - P.N);
  } else if (i < P.N + P.E + P.S + P.SB) {
    missing = pointFaces(i - P.E, OGC_NONE);
  } else if (i >= P.N + P.E + P.S + P.SB + P.SE + P.SEB && i < P.N + P.E + P.S + P.SB + P.SE + P.SEB + P.SF + P.SFB) {
    missing = facetVertices(i - P.N - P.E - P.S - P.SB - P.SE - P.SEB);
  }
  if (missing != 0u) { atomicAdd(&wgMissing, missing); }
  workgroupBarrier();
  if (li == 0u) {
    let m = atomicLoad(&wgMissing);
    if (m != 0u) { atomicAdd(&counters[OGC_VERIFY_MISSING], m); }
    let u = atomicLoad(&wgUncovered);
    if (u != 0u) { atomicAdd(&counters[OGC_F_UNCOVERED], u); }
  }
}
`;

/**
 * The per-vertex displacement cap in x_det.w. Isotropic: OGC's
 * b_v = gamma_p min(d_min,v, d^T_min,v, d^E_min,v), each clipped at the vertex's own
 * query radius (eqs. 21-26). Planar: 0.5 gamma_r r_q,v (DAT Alg. 2, line 10), on the
 * residual. Thread 0 sizes the planes dispatch from the round's pair slots. A vertex
 * with any entry (cloth or solid) is held for the rest of the step and the next.
 */
export const OGC_BOUND_WGSL = header([0, 1, 4, 5, 8, 9, 12]) + SOLID_ACCESS + OGC_SOLID_WGSL + /* wgsl */`
// A vertex's contact entries in a deterministic order: they are appended atomically, and the
// solver sums their forces in slot order, so the race's append order would otherwise reach the
// cloth through f32 rounding. Its own thread insertion-sorts its c entries (vec4<u32>, at most
// contactCap), lexicographic on (y, x, w, z).
fn ldv4(b: u32) -> vec4<u32> { return vec4<u32>(atomicLoad(&contacts[b]), atomicLoad(&contacts[b + 1u]), atomicLoad(&contacts[b + 2u]), atomicLoad(&contacts[b + 3u])); }
fn stv4(b: u32, e: vec4<u32>) { atomicStore(&contacts[b], e.x); atomicStore(&contacts[b + 1u], e.y); atomicStore(&contacts[b + 2u], e.z); atomicStore(&contacts[b + 3u], e.w); }
fn v4Less(a: vec4<u32>, c: vec4<u32>) -> bool {
  if (a.y != c.y) { return a.y < c.y; }
  if (a.x != c.x) { return a.x < c.x; }
  if (a.w != c.w) { return a.w < c.w; }
  return a.z < c.z;
}
fn sortV4(b: u32, n: u32) {
  for (var i = 1u; i < n; i++) {
    let e = ldv4(b + 4u * i);
    var j = i;
    loop {
      if (j == 0u) { break; }
      let f = ldv4(b + 4u * (j - 1u));
      if (!v4Less(e, f)) { break; }
      stv4(b + 4u * j, f);
      j--;
    }
    stv4(b + 4u * j, e);
  }
}
// The distance at x_det from x to hull facet slot (hullFacetAt's: static below P.facetCap, else the moving solid's).
fn facetDistance(slot: u32, x: vec3<f32>) -> f32 {
  var b = 4u * (P.facetBase + slot); var off = P.N;
  if (slot >= P.facetCap) { b = 4u * (P.bodyFacetBase + slot - P.facetCap); off = P.N + P.S; }
  let a = xdet[atomicLoad(&contacts[b]) + off].xyz; let bb = xdet[atomicLoad(&contacts[b + 1u]) + off].xyz; let c = xdet[atomicLoad(&contacts[b + 2u]) + off].xyz;
  let cl = ogcClosestTriangle(x, a, bb, c);
  return length(x - (cl.bary.x * a + cl.bary.y * bb + cl.bary.z * c));
}
// One representation of each solid per vertex: the solid is the union of the field's zero set and the samples' hull
// (the facets lie on or outside the source, filling the holes where the field rounds its convex features in), so the
// vertex's barrier is on its distance to that union, the nearest of its field entry (phi at x_det, the moving solid's
// at the detection pose: the start pose in the mandatory round, the end pose after) and its facet entries (each the
// distance to its facet; on a hull edge two facets meet at one closest point). The nearest stays, ties to the field
// and then to the lower slot (the entries are sorted); the others of that solid are dropped, so no contact is counted
// twice. The entries keep their order. Returns the count left (n when nothing was dropped).
fn ownSolids(v: u32, b: u32, n: u32) -> u32 {
  let x = xdet[v].xyz;
  var field = array<u32, 2>(OGC_NONE, OGC_NONE);
  var near = array<u32, 2>(OGC_NONE, OGC_NONE);
  var best = array<f32, 2>(3.4e38, 3.4e38);
  for (var i = 0u; i < n; i++) {
    let e = ldv4(b + 4u * i);
    if (e.y == OGC_SOLID_Y) { field[0] = i; }
    else if (e.y == OGC_BODY_Y) { field[1] = i; }
    else if (e.w == 2u && (e.y & OGC_EE_BIT) == 0u) {
      let slot = e.y & OGC_ID_MASK;
      let g = select(0u, 1u, slot >= P.facetCap);
      let d = facetDistance(slot, x);
      if (near[g] == OGC_NONE || d < best[g]) { near[g] = i; best[g] = d; }
    }
  }
  if (near[0] == OGC_NONE && near[1] == OGC_NONE) { return n; }
  var dropField = array<bool, 2>(false, false);
  for (var g = 0u; g < 2u; g++) {
    if (near[g] == OGC_NONE || field[g] == OGC_NONE) { continue; }
    var phi = 3.4e38;
    if (g == 0u) { phi = ogcSolidSample(ogcSolidHeader(), x).phi; }
    else {
      let pd = ogcPose(select(1u, 0u, atomicLoad(&counters[OGC_ROUNDS]) == 0u));
      phi = ogcSolidSample(ogcSolidHeaderAt(P.bodyBase), ogcToBody(pd, x)).phi;
    }
    if (phi <= best[g]) { near[g] = OGC_NONE; } else { dropField[g] = true; }
  }
  var m = 0u;
  for (var i = 0u; i < n; i++) {
    let e = ldv4(b + 4u * i);
    var keep = true;
    if (e.y == OGC_SOLID_Y) { keep = !dropField[0]; }
    else if (e.y == OGC_BODY_Y) { keep = !dropField[1]; }
    else if (e.w == 2u && (e.y & OGC_EE_BIT) == 0u) { keep = i == near[select(0u, 1u, (e.y & OGC_ID_MASK) >= P.facetCap)]; }
    if (keep) { if (m != i) { stv4(b + 4u * m, e); } m++; }
  }
  return m;
}
var<workgroup> wgSum: atomic<u32>;
var<workgroup> wgMax: atomic<u32>;
var<workgroup> wgDropped: atomic<u32>;
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
    var c = atomicLoad(&contacts[P.countsOff + v]);
    if (c > 1u) { sortV4(P.entriesBase + 4u * v * P.contactCap, min(c, P.contactCap)); }
    // A vertex with a facet entry keeps one representation of each solid (an overflowed list has failed already).
    if (c > 1u && c <= P.contactCap && P.SF + P.SFB != 0u) {
      let m = ownSolids(v, P.entriesBase + 4u * v * P.contactCap, c);
      if (m != c) { atomicAdd(&wgDropped, c - m); atomicStore(&contacts[P.countsOff + v], m); c = m; }
    }
    if (c != 0u) { atomicOr(&work[P.heldOff + v], 1u); }
    atomicAdd(&wgSum, min(c, P.contactCap));
    atomicMax(&wgMax, c);
  }
  workgroupBarrier();
  if (li == 0u) {
    atomicAdd(&counters[OGC_F_PAIR_ENTRIES], atomicLoad(&wgSum));
    atomicAdd(&counters[OGC_F_HF_DROPPED], atomicLoad(&wgDropped));
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
var<workgroup> wgHits: atomic<u32>;
var<workgroup> wgSolidHits: atomic<u32>;
// The face reaches inside a solid: phi < 0 at a point of a barycentric lattice over it
// with spacing at most a quarter cell (the field is trilinear per cell). The face is in
// the field's frame (the moving solid's body frame at its end pose).
// Returns the first such phi (negative), or 1 when the face is clear.
fn insideSolid(h: OgcSolidHeader, A: array<vec3<f32>, 3>) -> f32 {
  let longest = max(length(A[1] - A[0]), max(length(A[2] - A[1]), length(A[0] - A[2])));
  let n = u32(clamp(ceil(4.0 * longest / h.cell), 1.0, 32.0));
  let inv = 1.0 / f32(n);
  for (var i = 0u; i <= n; i++) { for (var j = 0u; j <= n - i; j++) {
    let u = f32(i) * inv; let w = f32(j) * inv;
    let p = (1.0 - u - w) * A[0] + u * A[1] + w * A[2];
    let s = ogcSolidSample(h, p);
    if (s.stored && s.phi < 0.0) { return s.phi; }
  } }
  return 1.0;
}
fn solidHit(t: u32, phi: f32) {
  atomicAdd(&wgSolidHits, 1u);
  if (atomicCompareExchangeWeak(&counters[OGC_SOLID_HIT_FACE], 0u, t).exchanged) { atomicStore(&counters[OGC_SOLID_HIT_PHI], bitcast<u32>(phi)); }
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
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let t = gid.x;
  if (t < P.T) {
    let tv = triVerts(t);
    let A = array<vec3<f32>, 3>(xdet[tv.x].xyz, xdet[tv.y].xyz, xdet[tv.z].xyz);
    var phi = 1.0;
    if (P.solidBase != 0u) { phi = insideSolid(ogcSolidHeader(), A); }
    if (phi < 0.0) { solidHit(t + 1u, phi); }
    else if (P.bodyBase != 0u) {
      let p1 = ogcPose(1u);
      phi = insideSolid(ogcSolidHeaderAt(P.bodyBase), array<vec3<f32>, 3>(ogcToBody(p1, A[0]), ogcToBody(p1, A[1]), ogcToBody(p1, A[2])));
      if (phi < 0.0) { solidHit((t + 1u) | 0x80000000u, phi); }
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

/**
 * The per-vertex contact state at the end of a step (once per step, after the solver's last
 * update of its positions and before it forms its velocities and history): a vertex touches
 * when one of its entries of the step's last round (a cloth pair it weighs on, or a static or
 * moving solid entry) puts a normal contact force on it at the step's end x_{n+1}: the gap is
 * below r there (the barrier's kappa'' > 0) and the vertex's own weight in it is nonzero. The
 * last round's entries hold every primitive within r of the final iterate (the round's
 * certificate: OGC's completeness), so this is exact. state[v]: bit 0 touches at the end of
 * this step, bit 1 contact onset, touching now and not at the end of the step before (carried
 * in the held flags, bits 8 and 9, across steps and saves). Writes nothing else.
 */
export const OGC_CONTACT_STATE_WGSL = [PARAMS_STRUCT, DECL[0], DECL[1], DECL[2], DECL[5], DECL[13]].join("\n") + /* wgsl */`
@group(0) @binding(18) var<storage, read_write> state: array<u32>;
fn ogcX(i: u32) -> vec3<f32> { return pos[i].xyz; }
fn ogcXt(i: u32) -> vec3<f32> { return xt[i].xyz; }
fn ogcTri(t: u32) -> vec3<u32> { let b = P.trisOff + 4u * t; return vec3<u32>(topo[b], topo[b + 1u], topo[b + 2u]); }
fn ogcEdgeVerts(e: u32) -> vec2<u32> { let b = P.edgesOff + 4u * e; return vec2<u32>(topo[b], topo[b + 1u]); }
` + ogcVertexBlockWGSL({ group: 0, binding: 8 }) + /* wgsl */`
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let v = gid.x;
  if (v >= P.N) { return; }
  let header = ogcContacts[0];
  let params = bitcast<vec4<f32>>(ogcContacts[1]);
  let count = ogcContactCount(v);
  let base = header.w + v * header.y;
  let xv = pos[v].xyz;
  var touching = false;
  for (var i = 0u; i < count; i++) {
    let l = ogcEntryLin(v, xv, ogcContacts[base + i], params);
    if (l.wr != 0.0 && l.Mn[0][0] + l.Mn[1][1] + l.Mn[2][2] > 0.0) { touching = true; break; }
  }
  let old = atomicLoad(&work[P.heldOff + v]);
  let s = select(0u, 1u, touching) | select(0u, 2u, touching && (old & 256u) == 0u);
  state[v] = s;
  atomicStore(&work[P.heldOff + v], (old & ~768u) | (s << 8u));
}
`;
