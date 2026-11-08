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
 * work also holds, per vertex at solidOff, the static solid's truncation terms (q.xyz, j)
 * of the round, and at heldOff the held flags (bit 0: contact, truncation or conflict
 * this step; bit 1: held this step); the solid image sits in the contact buffer at
 * word solidBase.
 *
 * One detection round is nine indirect dispatches:
 *   prep (N)  cell (1)  insert-count (T)  scan x3  insert (T)  query (N + E)  bound (N)
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
import { OGC_COMMON_WGSL, OGC_COUNTERS_WGSL, OGC_SOLID_WGSL } from "./ogc-wgsl";

export const OGC_PARAM_WORDS = 64;
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
  cellRule: u32, pad0: u32, pad1: u32, pad2: u32,
}
`;

/** Parameter word offsets, matching OgcParams. */
export const OGC_PARAM = {
  N: 0, T: 1, E: 2, H: 3, gridCap: 4, contactCap: 5, roundCap: 6, exceedThreshold: 7,
  r: 8, rqBase: 9, rqMax: 10, envScale: 11, gammaP: 12, cellFloor: 13, dt: 14, intersectCell: 15,
  gravity: 16, trisOff: 20, edgesOff: 21, triEdgesOff: 22, vtOffOff: 23, vtListOff: 24, veOffOff: 25, veListOff: 26, B1: 27,
  local1Off: 28, sums1Off: 29, local2Off: 30, sums2Off: 31, local3Off: 32, sums3Off: 33, entriesOff: 34, B2: 35,
  countsOff: 36, entriesBase: 37, dminTOff: 38, dminEOff: 39, envOff: 40, wgN: 41, wgT: 42, wgNE: 43, wgS1: 44, wgS2: 45, kc: 46,
  mode: 47, gammaR: 48, pairCap: 49, tvOff: 50, sepEps: 51, solidBase: 52, solidOff: 53, heldOff: 54, predict: 55, cellRule: 56,
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
fn predOf(i: u32) -> vec3<f32> { return pred[i].xyz; }
fn anchored(i: u32) -> vec3<f32> { return xdet[i].xyz + pred[i].xyz; }
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
function triBoxFns(intersect: boolean): string {
  return (intersect ? "" : PRED_FNS) + /* wgsl */`
fn envOf(i: u32) -> f32 { ${intersect ? "return 0.0;" : "return bitcast<f32>(atomicLoad(&work[P.envOff + i]));"} }
fn triEnv(v: vec3<u32>) -> f32 { return max(envOf(v.x), max(envOf(v.y), envOf(v.z))); }
// The face's box over the round: from X to the predicted X + w, grown by its widest envelope.
fn triBox(t: u32) -> OgcBox {
  let v = triVerts(t);
  let a = xdet[v.x].xyz; let b = xdet[v.y].xyz; let c = xdet[v.z].xyz;
  var lo = min(a, min(b, c)); var hi = max(a, max(b, c));
  ${intersect ? "" : `let a1 = a + predOf(v.x); let b1 = b + predOf(v.y); let c1 = c + predOf(v.z);
  lo = min(lo, min(a1, min(b1, c1))); hi = max(hi, max(a1, max(b1, c1)));`}
  let e = triEnv(v);
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
 * A free vertex is predicted to move its inertial v dt + g dt^2 (the rest of it
 * in a re-detection round), and its envelope covers only the residual motion:
 * r + s rho, with rho the last step's deviation from its own prediction. A held
 * vertex (in contact, truncated or in a conflict in this step or the last, or
 * every vertex of a redone round, of isotropic mode, or without predictions) is
 * predicted 0 and its envelope covers the inertial displacement, r + s |v dt + g dt^2|:
 * Planar-DAT itself. A prediction into the static solid stops where the solid's
 * affine bound does, so every cut of the round scales only the residual.
 */
export const OGC_PREP_WGSL = header([0, 2, 3, 4, 5, 8, 9, 13, 14], [14]) + SOLID_ACCESS + OGC_SOLID_WGSL + /* wgsl */`
var<workgroup> wgMaxEnv: atomic<u32>;
var<workgroup> wgCube: array<f32, 64>;
var<workgroup> wgCap: atomic<u32>;
var<workgroup> wgBad: atomic<u32>;
var<workgroup> wgHeld: atomic<u32>;
var<workgroup> wgSolid: atomic<u32>;
fn putWork4(i: u32, q: vec4<f32>) {
  atomicStore(&work[i], bitcast<u32>(q.x)); atomicStore(&work[i + 1u], bitcast<u32>(q.y));
  atomicStore(&work[i + 2u], bitcast<u32>(q.z)); atomicStore(&work[i + 3u], bitcast<u32>(q.w));
}
// The static solid over the ball |d| <= rho around x0: the truncation terms (q, j) =
// (g0, J) / (gamma_r phi0), so that phi(x0 + d) >= (1 - gamma_r) phi0 wherever
// j |d| - q.d <= 1, and a solid entry for the vertex when the ball can reach r.
fn solidRound(v: u32, x0: vec3<f32>, rho: f32) -> vec4<f32> {
  let h = ogcSolidHeader();
  let rel = (x0 - h.origin) / (8.0 * h.cell);
  if (any(rel < vec3<f32>(0.0)) || any(rel >= vec3<f32>(h.dims))) { atomicOr(&counters[OGC_STATUS_WORD], OGC_ST_SOLID_BOUNDS); }
  let sb = ogcSolidBallBound(h, x0, rho, P.r);
  var qj = vec4<f32>(0.0);
  if (!sb.far) {
    if (!(sb.phi0 > 0.0)) {
      qj.w = 3.0e38;
      atomicOr(&counters[OGC_STATUS_WORD], OGC_ST_SOLID_PENETRATION);
    } else {
      let s = 1.0 / (P.gammaR * sb.phi0);
      qj = vec4<f32>(sb.g0 * s, sb.J * s);
      // mu lambda: the solid's own friction at x_det times the lagged normal force.
      let lambda = select(0.0, -ogcActivation(sb.phi0, P.r, P.kc).y * length(sb.g0), sb.phi0 < P.r) * ogcSolidMu(h, x0);
      let slot = atomicAdd(&contacts[P.countsOff + v], 1u);
      if (slot < P.contactCap) {
        let b = P.entriesBase + 4u * (v * P.contactCap + slot);
        atomicStore(&contacts[b], v); atomicStore(&contacts[b + 1u], OGC_SOLID_Y);
        atomicStore(&contacts[b + 2u], bitcast<u32>(lambda)); atomicStore(&contacts[b + 3u], 0u);
      } else {
        atomicOr(&counters[OGC_STATUS_WORD], OGC_ST_CONTACT_OVERFLOW);
      }
      atomicAdd(&wgSolid, 1u);
    }
  }
  putWork4(P.solidOff + 4u * v, qj);
  return qj;
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let v = gid.x;
  let mandatory = atomicLoad(&counters[OGC_ROUNDS]) == 0u;
  let forced = atomicLoad(&counters[OGC_FORCE_HELD]) != 0u;
  // A redone round repeats its round's detection with every prediction held; only the first measures.
  let fresh = mandatory && !forced;
  if (v == 0u) { atomicStore(&counters[OGC_PAIR_SLOTS], 0u); atomicStore(&counters[OGC_CONFLICTS], 0u); }
  wgCube[li] = 0.0;
  if (v < P.N) {
    let x = select(pos[v].xyz, xt[v].xyz, mandatory);
    // The step's predicted displacement: v dt + g dt^2, or the solver's own initial
    // guess x_guess - x_t (in positions at the mandatory round); its length is kept
    // in pred[v].w for the held envelope of the step's later rounds.
    var inertial = vel[v].xyz * P.dt + P.gravity.xyz * (P.dt * P.dt);
    if (P.predict == 2u) { inertial = pos[v].xyz - x; }
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
      flags = (flags & 1u) << 1u;
      if (2.0 * yref.w > plen) { flags |= 2u; }
      atomicStore(&work[P.heldOff + v], flags);
    }
    let held = forced || P.predict == 0u || P.mode != OGC_PLANAR || flags != 0u;
    var w = select(yref.xyz - x, vec3<f32>(0.0), held);
    var env = max(P.rqBase, P.r + P.envScale * select(yref.w, plen, held));
    if (env > P.rqMax) { env = P.rqMax; atomicAdd(&wgCap, 1u); }
    atomicStore(&contacts[P.countsOff + v], 0u);
    if (P.solidBase != 0u) {
      // The round moves the vertex at most |w| + env / 2 (the cap is 0.5 gamma_r or gamma_p
      // of env); the field's band covers rqMax / 2.
      let room = max(0.5 * (P.rqMax - env), 0.0);
      let lw = length(w);
      if (lw > room) { w *= room / lw; atomicAdd(&wgCap, 1u); }
      let qj = solidRound(v, x, length(w) + 0.5 * env);
      let f = qj.w * length(w) - dot(qj.xyz, w);
      if (f > 1.0) { w = select(w / f, vec3<f32>(0.0), qj.w >= 1.0e38); }
    }
    if (fresh) { pred[P.N + v] = vec4<f32>(select(x + w, yref.xyz, held), yref.w); }
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
  let cube = f32(atomicExchange(&counters[OGC_ENVELOPE_SUM], 0u)) / (1024.0 * f32(P.N));
  let h = select(max(P.cellFloor, m), max(P.cellFloor, P.rqMax * pow(cube, 1.0 / 3.0)), P.cellRule == 1u);
  atomicStore(&counters[OGC_CELL], bitcast<u32>(h));
  atomicStore(&counters[OGC_INV_CELL], bitcast<u32>(1.0 / h));
}
`;

const ARGS_FNS = /* wgsl */`
fn writeArgs(on: bool) {
  var n = array<u32, 9>(P.wgN, 1u, P.wgT, P.wgS1, P.wgS2, 1u, P.wgT, P.wgNE, P.wgN);
  for (var i = 0u; i < 9u; i++) {
    args[3u * i] = select(0u, n[i], on); args[3u * i + 1u] = 1u; args[3u * i + 2u] = 1u;
  }
}
`;

/** Start of a step: the mandatory detection round, and the step's round count reset. */
export const OGC_STEP_BEGIN_WGSL = header([0, 9, 10]) + ARGS_FNS + /* wgsl */`
@compute @workgroup_size(1)
fn main() {
  atomicStore(&counters[OGC_ROUNDS], 0u);
  atomicStore(&counters[OGC_EXCEED], 0u);
  atomicStore(&counters[OGC_FORCE_HELD], 0u);
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
  atomicAdd(&counters[OGC_F_DECIDES], 1u);
  atomicAdd(&counters[OGC_F_TRUNCATED], ex);
  atomicMax(&counters[OGC_F_TRUNCATED_MAX], ex);
  var on = false;
  if (ex > 0u && ex >= P.exceedThreshold) {
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
 * any truncation, with every prediction held at 0, which is Planar-DAT. The
 * conflict's vertices stay held for the rest of the step and the next.
 */
export const OGC_REDO_WGSL = header([0, 9, 10]) + ARGS_FNS + /* wgsl */`
@compute @workgroup_size(1)
fn main() {
  let c = atomicExchange(&counters[OGC_CONFLICTS], 0u);
  let on = c > 0u;
  atomicStore(&counters[OGC_FORCE_HELD], select(0u, 1u, on));
  if (on) {
    atomicAdd(&counters[OGC_F_CONFLICTS], c);
    atomicAdd(&counters[OGC_F_REDOS], 1u);
    atomicAdd(&counters[OGC_F_DETECTIONS], 1u);
  }
  writeArgs(on);
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
export const OGC_PLANES_WGSL = [PARAMS_STRUCT, OGC_COUNTERS_WGSL, OGC_COMMON_WGSL, DECL[0], DECL[2], DECL[4], DECL[5], DECL[9], DECL_PAIRS_READ, DECL[14]].join("\n") + PRED_FNS + /* wgsl */`
fn ratio(s0: f32, s1: f32) -> f32 {
  // A wrong-side start may move toward its side but never further across.
  if (s0 < 0.0) { return select(0.0, 1.0, s1 >= s0); }
  if (s1 >= 0.0) { return 1.0; }
  return clamp(P.gammaR * s0 / (s0 - s1), 0.0, 1.0);
}
fn lower(v: u32, t: f32) { if (t < 1.0) { atomicMin(&work[P.tvOff + v], bitcast<u32>(t)); } }
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
        var supA = 3.4e38; var supB = 0.0; var apA = 0.0; var apB = 0.0;
        for (var k = 0u; k < 4u; k++) {
          let Y = anchored(ids[k]);
          proj[k] = dot(n, Y - o);
          dn[k] = dot(n, pos[ids[k]].xyz - Y);
          if (k < nA) { supA = min(supA, proj[k]); apA = max(apA, -dn[k]); }
          else { supB = max(supB, proj[k]); apB = max(apB, dn[k]); }
        }
        let gap = supA - supB;
        var lam = 0.5;
        if (gap >= 2.0 * P.sepEps) {
          if (apA + apB > 0.0) { lam = apB / (apA + apB); }
          let mf = max(0.05, P.sepEps / gap);
          lam = clamp(lam, mf, 1.0 - mf);
        }
        let plane = supB + lam * gap;
        for (var k = 0u; k < 4u; k++) {
          let a = k < nA;
          let s0 = select(plane - proj[k], proj[k] - plane, a) - P.sepEps;
          lower(ids[k], ratio(s0, s0 + select(-dn[k], dn[k], a)));
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

/**
 * Apply the truncation (Alg. 2, lines 8-12) to the residual delta = x - X - w:
 * delta <- t_v delta, then |delta| <= x_det.w, which is 0.5 gamma_r r_q,v in planar
 * mode (a pair left out of the round is further apart than the two caps plus its
 * prediction spread) and OGC's b_v in isotropic mode (w = 0 there). Each truncated
 * vertex counts towards the re-detection decision and is held for the rest of the
 * step and the next; tv is left at 1 for the next pass. A static solid then cuts
 * delta by its affine lower bound of phi (prep), which the prediction already meets:
 * f(w + delta) <= f(w) + j |delta| - q.delta for the convex f(d) = j |d| - q.d.
 */
export const OGC_APPLY_WGSL = header([0, 2, 4, 5, 9, 14], [2]) + PRED_FNS + /* wgsl */`
var<workgroup> wgCount: atomic<u32>;
var<workgroup> wgCapped: atomic<u32>;
var<workgroup> wgSolidCut: atomic<u32>;
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
      // The static solid: phi(x_det + w + t delta) >= (1 - gamma_r) phi0 for t <= (1 - f(w)) / (j |delta| - q.delta).
      let b = P.solidOff + 4u * v;
      let qj = bitcast<vec4<f32>>(vec4<u32>(atomicLoad(&work[b]), atomicLoad(&work[b + 1u]), atomicLoad(&work[b + 2u]), atomicLoad(&work[b + 3u])));
      let room = max(1.0 - (qj.w * length(w) - dot(qj.xyz, w)), 0.0);
      let den = qj.w * length(delta) - dot(qj.xyz, delta);
      if (den > room) { delta *= room / den; cut = true; atomicAdd(&wgSolidCut, 1u); }
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
  ${intersect ? "if (t == 0u) { atomicStore(&counters[OGC_INTERSECTIONS], 0u); }" : ""}
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

export const OGC_QUERY_WGSL = header([0, 1, 4, 5, 7, 8, 9, 11, 14]) + TOPO_FNS + CELL_FNS + GRID_READ_FNS + triBoxFns(false) + /* wgsl */`
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
// the predicted positions X + w close below 2 sepEps along it (and below the gap at X).
// The planes pass computes the same predicted gap, bit for bit.
fn checkConflict(ids: vec4<u32>, nA: u32, pl: vec4<f32>) {
  var moving = false;
  for (var k = 0u; k < 4u; k++) { if (any(predOf(ids[k]) != vec3<f32>(0.0))) { moving = true; } }
  if (!moving) { return; }
  var conflict = !(pl.w > 0.0);
  if (!conflict) {
    let o = xdet[ids[nA]].xyz;
    var supA = 3.4e38; var supB = -3.4e38;
    for (var k = 0u; k < 4u; k++) {
      let pr = dot(pl.xyz, anchored(ids[k]) - o);
      if (k < nA) { supA = min(supA, pr); } else { supB = max(supB, pr); }
    }
    let g1 = supA - supB;
    conflict = g1 < 2.0 * P.sepEps && g1 < pl.w;
  }
  if (conflict) {
    atomicAdd(&counters[OGC_CONFLICTS], 1u);
    for (var k = 0u; k < 4u; k++) { atomicOr(&work[P.heldOff + ids[k]], 1u); }
  }
}
fn pairVT(v: u32, tri: vec3<u32>, x: vec3<f32>, cp: vec3<f32>) {
  let a = xd(tri.x); let b = xd(tri.y); let c = xd(tri.z);
  let n0 = unitOr0(x - cp);
  let fnorm = unitOr0(cross(b - a, c - a));
  let ids = vec4<u32>(v, tri);
  let pl = certify(n0, gapVT(n0, x, a, b, c), fnorm, gapVT(fnorm, x, a, b, c), gapVT(-fnorm, x, a, b, c));
  checkConflict(ids, 1u, pl);
  pushPair(ids, pl);
}
fn pairEE(er: vec4<u32>, r2: vec4<u32>, ca: vec3<f32>, cb: vec3<f32>) {
  let p0 = xd(er.x); let p1 = xd(er.y); let q0 = xd(r2.x); let q1 = xd(r2.y);
  let n0 = unitOr0(ca - cb);
  let ec = unitOr0(cross(p1 - p0, q1 - q0));
  let pl = certify(n0, gapEE(n0, p0, p1, q0, q1), ec, gapEE(ec, p0, p1, q0, q1), gapEE(-ec, p0, p1, q0, q1));
  checkConflict(vec4<u32>(er.x, er.y, r2.x, r2.y), 2u, pl);
  pushPair(vec4<u32>(er.x | OGC_EE_BIT, er.y, r2.x, r2.y), pl);
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
  let q = OgcBox(min(x, x1) - vec3<f32>(rv), max(x, x1) + vec3<f32>(rv));
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
      if (d < P.r && vfAccept(v, x, t, tri, cl.feature)) {
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
  let q = OgcBox(min(min(p0, p1), min(p01, p11)) - vec3<f32>(re), max(max(p0, p1), max(p01, p11)) + vec3<f32>(re));
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
        if (st.z < P.r && st.x > 0.0 && st.x < 1.0 && st.y > 0.0 && st.y < 1.0) {
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

var<workgroup> wgCand: atomic<u32>;
var<workgroup> wgVf: atomic<u32>;
var<workgroup> wgEe: atomic<u32>;
var<workgroup> wgDat: atomic<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let i = gid.x;
  if (i < P.N) {
    let o = vertexFacet(i);
    atomicAdd(&wgCand, o.candidates); atomicAdd(&wgVf, o.pairs); atomicAdd(&wgDat, o.dat);
  } else if (i < P.N + P.E) {
    let o = edgeEdge(i - P.N);
    atomicAdd(&wgCand, o.candidates); atomicAdd(&wgEe, o.pairs); atomicAdd(&wgDat, o.dat);
  }
  closePairs();
  workgroupBarrier();
  if (li == 0u) {
    atomicAdd(&counters[OGC_F_CANDIDATES], atomicLoad(&wgCand));
    atomicAdd(&counters[OGC_F_VF_PAIRS], atomicLoad(&wgVf));
    atomicAdd(&counters[OGC_F_EE_PAIRS], atomicLoad(&wgEe));
    atomicAdd(&counters[OGC_F_DAT_PAIRS], atomicLoad(&wgDat));
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
export const OGC_INTERSECT_WGSL = header([0, 1, 4, 7, 9]) + TOPO_FNS + CELL_FNS + GRID_READ_FNS + triBoxFns(true) + /* wgsl */`
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
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let t = gid.x;
  if (t < P.T) {
    let tv = triVerts(t);
    let A = array<vec3<f32>, 3>(xdet[tv.x].xyz, xdet[tv.y].xyz, xdet[tv.z].xyz);
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
  if (li == 0u) { let h = atomicLoad(&wgHits); if (h != 0u) { atomicAdd(&counters[OGC_INTERSECTIONS], h); } }
}
`;
