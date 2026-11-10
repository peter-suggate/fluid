/**
 * GPU OGC self-contact detector for triangle cloth (Chen et al. 2025).
 *
 * Owns the detection positions X = x_det, the per-vertex contact lists, the
 * DAT division pairs, the broad-phase hash grid, the truncation and the step's
 * re-detection decision. Per step, all on the GPU and with no readback:
 *
 *   encodeStepBegin + encodeDetection        mandatory round at x = x_t (stepStart)
 *   (solver initial guess) encodeTruncate
 *   per colour sweep: solver (ogcAccumulateVertexContact)
 *   per iteration:    encodeTruncate, encodeDecide + encodeDetection   indirect: zero workgroups
 *                                                      unless gamma_e N truncated
 *
 * The solver's positions are X + dX: dX accumulates across iterations, every
 * truncation cuts the accumulated dX, and a detection round commits it (DAT
 * Alg. 3). Truncation is Planar-DAT by default, or OGC's isotropic bound.
 *
 * Predicted-motion DAT (the default; `prediction: "none"` is Planar-DAT alone):
 * a free vertex is predicted to move w_v = v dt + g dt^2 over the step (or the
 * solver's initial guess minus x_t, `prediction: "guess"`), and its
 * query radius covers only the residual,
 *   r_q,v = clamp(max(rqBase, r + s rho_v), rqMax),
 * rho_v the last step's deviation from its own prediction. A vertex in contact,
 * truncated or in a conflict (in this step or the last) is held: w_v = 0 and
 * r_q,v = clamp(max(rqBase, r + s |w_v|), rqMax), capped (envelopeMeshCap) at
 * gamma_r h_min, h_min the rest mesh's least altitude, so no envelope pairs a primitive with
 * its own neighbourhood. Every vertex's residual is capped at 0.5 gamma_r r_q,v about its path
 * X + s w (one shared s), so a pair is dropped (DAT test, E12) when d(X) exceeds
 * thr + L, thr = 0.5 gamma_r (r_q,a + r_q,b) + sepEps + strays + rounding and L the
 * predictions' spread, or when a Lipschitz-sampled bound of the distance along the shared path
 * exceeds thr; the kept pairs are recorded once the swept boxes meet. Planes translate with the
 * predictions; a round whose predictions cross is redone with every one held.
 * Capacities are authored per scene; overflow, a round cap or an envelope cap
 * set the status word (OGC_STATUS), which the frame's owner reads off the step path.
 */
import { buildOgcTopology, type OgcMeshTopology, type OgcTopology } from "./ogc-topology";
import {
  OGC_APPLY_WGSL, OGC_ARGS_WORDS, OGC_BOUND_WGSL, OGC_CELL_WGSL, ogcDecideWGSL, OGC_INTERSECT_WGSL, OGC_PARAM,
  OGC_PARAM_WORDS, ogcPlanesWGSL, OGC_FILTER_WGSL, OGC_FILTER_ALLOC_WGSL, OGC_FILTER_BEGIN_WGSL, OGC_CONTACT_STATE_WGSL, OGC_PREP_WGSL, OGC_QUERY_ROLES, OGC_REDO_LAST_WGSL, OGC_REDO_MANDATORY, OGC_REDO_ROUNDS, OGC_SHIFT_WORDS, ogcQueryWGSL, OGC_REDO_WGSL, OGC_STEP_BEGIN_WGSL,
  ogcFrameBeginWGSL, ogcGridCountWGSL, ogcGridInsertWGSL, ogcScanWGSL,
} from "./ogc-detector-wgsl";
import {
  OGC_COUNTER, OGC_COUNTER_WORDS, OGC_FRAME_COUNTERS, OGC_ROUND_LOG, OGC_ROUND_LOG_WORDS, OGC_POSE_WORDS, OGC_LEVEL_VEC4S, OGC_PROJECT_RECORD, OGC_STATUS, OGC_TRUNCATION, ogcVertexBlockWGSL,
  type OgcTruncationMode,
} from "./ogc-wgsl";
import { writeBufferChunked } from "../../methods/uniform/uniform-chunked-upload";
import { SOLID_HEADER_WORDS, SOLID_SAMPLE_WORDS, clothSolidSampleEdges, clothSolidSamples, type ClothSolidField } from "./solid-field";
import { OGC_NONE } from "./ogc-topology";

export interface OgcContactConfig {
  /** Offset radius: the physical cloth thickness (m). */
  r: number;
  /** Contact stiffness k_c of the activation (N/m). */
  kc: number;
  /** Friction coefficient and the static-dynamic transition velocity eps_v (m/s). */
  mu: number;
  epsV: number;
  /** The step (s). */
  dt: number;
  gravity: readonly [number, number, number];
  /** Query radius floor; with envelopeScale 0 it is the fixed r_q. */
  rqBase: number;
  /** Authored envelope cap; reaching it raises OGC_STATUS.envelopeCap. */
  rqMax: number;
  /**
   * Cap the envelope at gamma_r times the mesh's smallest rest triangle altitude h_min too (E12 iii; default off).
   * Two caps then sum to at most gamma_r^2 h_min, below the distance from any vertex to a primitive it is not
   * adjacent to in a sheet at its rest shape (at least its one-ring's altitude), so a resting sheet pairs with
   * nothing and a vertex's plane records are its genuine approaches, not its neighbourhood. Reaching the cap is
   * reported as OGC_STATUS.envelopeCap like rqMax: a longer motion takes more rounds, never a weaker certificate.
   */
  envelopeMeshCap?: boolean;
  /** Safety factor s on the inertial displacement; 0 gives the fixed envelope. */
  envelopeScale: number;
  /** OGC's gamma_p (0 < gamma_p < 0.5) and gamma_e. */
  gammaP?: number;
  gammaE?: number;
  /** Truncation scheme (default planar) and DAT's relaxation gamma_r in (0, 1) (default 0.9). */
  truncation?: OgcTruncationMode;
  gammaR?: number;
  /** Half-width of the empty band DAT keeps on each side of a plane (m); default from the coordinate scale. */
  separationEps?: number;
  /** Authored capacities: entries per vertex list, DAT pair slots, grid entries, hash buckets (power of two). */
  contactCapacity: number;
  pairCapacity: number;
  gridCapacity: number;
  bucketCount: number;
  /** Re-detection rounds allowed per step after the mandatory one. */
  roundCap: number;
  /** Words reserved for the static solid field image (solid-field.ts); 0 or absent: the cloth sees no solid. */
  solidCapacityWords?: number;
  /**
   * Static surface samples reserved for the solid's sharp features (solid-field.ts
   * clothSolidSamples), which meet the cloth's faces; a field with more fails at
   * setSolidField. Absent: 0.
   */
  solidSampleCapacity?: number;
  /**
   * One moving solid (one-way: its poses are prescribed). Words reserved for its
   * body-frame field image (solid-field.ts), its sharp-feature surface samples, and
   * the poses one frame uploads (its steps plus one; setBodyPoses). Planar truncation
   * only. In the solid's frame it is a static solid: its field, samples and hull edges
   * meet the cloth mapped into that frame at both ends of a round, with the curvature of
   * the cloth's straight world motion there added to the margins. A conflict with it moves its
   * cloth's predictions towards moving with it only as far as the pair needs (least shift, the
   * relative motion scaled in the solid's frame; OGC_REDO_WGSL); the fail-safe past the
   * least-shift redos carries the zones rigidly (counted and reported as failSafe).
   * Absent: no moving solid.
   */
  body?: { capacityWords: number; sampleCapacity: number; poseCapacity: number };
  /**
   * Predicted-motion DAT, predicting v dt + g dt^2 ("inertial", the default), the
   * solver's initial guess x_guess - x_t ("guess": needs `stepStart`, with the guess
   * in `positions` when the mandatory round runs), the solver's inertial target
   * y - x_t ("target": needs `stepStart` and `target`; the free-flight solution of
   * any implicit step, so a sheet in free flight is never truncated whatever the
   * solver's guess), or Planar-DAT with every prediction held ("none").
   */
  prediction?: OgcPrediction;
  /** The broad-phase cell's floor (m); default the larger of the mean edge length and rqBase. */
  cellSize?: number;
  /**
   * The round's cell above the floor: its largest envelope ("largest", the default)
   * or its cubic-mean envelope ("cubicMean"). Every primitive goes into every cell
   * its box touches either way; on the M1 Max the per-cell cost of a query outweighs
   * the candidates a larger cell adds (twist gate: 11 ms vs 44 ms a round).
   */
  cellRule?: OgcCellRule;
  /**
   * A Newton solver steps along whole directions (encodeFilter): the division planes are
   * placed once per round, at the clamped midpoint of the anchored support gap, so the
   * round's certificates stay fixed under the iterate; the filter kernels are compiled.
   * Absent: the per-truncation placement that follows the residual (the VBD sweeps).
   */
  newton?: boolean;
  /**
   * Plane records the line-search filter may hold at once for ogcProject (newton config): one per
   * cloth vertex of each DAT pair of the round. Default 16 per vertex; author it from the measured
   * peak (peakPlaneRecords). Overflow fails the step (planeRecordOverflow).
   */
  planeRecordCapacity?: number;
  /**
   * The metric of the solver's trial projection (solverContact; newton only): "off" (default) the exact Euclidean
   * projection, "hvv" the Newton solver's vertex block H_vv (nwVertexInv; ogc-wgsl.ts OGC_PROJECT_METRIC_WGSL).
   */
  projectionMetric?: OgcProjectionMetric;
}
export type OgcProjectionMetric = "off" | "hvv";

export const OGC_PREDICTION = { none: 0, inertial: 1, guess: 2, target: 3 } as const;

/** A rigid pose of the moving solid: x_world = R(orientation) x_body + position. */
/** The per-vertex detector state carried from one step to the next (readCarriedState). */
export interface OgcCarriedState {
  /** Per vertex: the reference predicted position (xyz) and residual rho_v (w; -1 before any step). */
  readonly reference: Float32Array;
  /** Per vertex: the held flags (bit 0: held in the step that wrote it; bit 1: held in it). */
  readonly held: Uint32Array;
}

export interface OgcBodyPose {
  position: readonly [number, number, number];
  /** Unit quaternion (x, y, z, w). */
  orientation: readonly [number, number, number, number];
}
export const OGC_CELL_RULE = { largest: 0, cubicMean: 1 } as const;
export type OgcCellRule = keyof typeof OGC_CELL_RULE;
export type OgcPrediction = keyof typeof OGC_PREDICTION;

export interface OgcSolverInputs {
  /** vec4<f32> per vertex, cloth-local metres (w is the solver's). */
  positions: GPUBuffer | GPUBufferBinding;
  /** vec4<f32> per vertex, m/s: the step's start velocity, for the envelope. */
  velocities: GPUBuffer | GPUBufferBinding;
  /**
   * vec4<f32> per vertex, cloth-local metres: the step start x_t, where the
   * mandatory round detects (the solver's initial guess is already in
   * `positions` by then). Absent: `positions` holds x_t when the round runs.
   */
  stepStart?: GPUBuffer | GPUBufferBinding;
  /** vec4<f32> per vertex, metres: a Newton solver's direction d for encodeFilter (newton config only). */
  direction?: GPUBuffer | GPUBufferBinding;
  /**
   * The Newton solver's control record (one vec4: done, PCG done, the accepted step alpha, the
   * step's Newton iterations; newton config only). Bound, re-detection is levelled: a round
   * predicts the rest of the Newton step it follows and its certificates move with the level
   * (ogcProjectTrial, which the solver's projection must then call with each trial's alpha).
   */
  control?: GPUBufferBinding;
  /**
   * vec4<f32> per vertex, cloth-local metres: the step's inertial target y (the minimizer
   * of the step's inertia term), in place when the mandatory round runs (prediction "target" only).
   */
  target?: GPUBuffer | GPUBufferBinding;
}

export interface OgcFrameCounters {
  status: number;
  steps: number;
  detections: number;
  redetections: number;
  decides: number;
  truncated: number;
  truncatedMax: number;
  pairEntries: number;
  pairMax: number;
  candidates: number;
  vfPairs: number;
  eePairs: number;
  envelopeCapHits: number;
  peakEnvelope: number;
  /** DAT division pairs recorded over the frame's rounds, the widest round's slots, and the persistent peak. */
  datPairs: number;
  framePeakPairSlots: number;
  peakPairSlots: number;
  /** Vertices cut by the 0.5 gamma_r r_q,v cap (or b_v), and pairs held for want of a separating plane. */
  capped: number;
  uncertified: number;
  /** Vertices with a static-solid entry over the frame's rounds, and truncations the solid bound made. */
  solidContacts: number;
  solidCut: number;
  /** Conflicting pairs, rounds redone held, and held vertices summed over the frame's mandatory rounds. */
  conflicts: number;
  redos: number;
  held: number;
  roundCapHits: number;
  peakContacts: number;
  peakGrid: number;
  intersections: number;
  cell: number;
  /** Sample-face contacts (static sharp-feature samples against cloth faces) over the frame's rounds. */
  sfPairs: number;
  /** Cloth faces reaching inside a solid (static or moving) at the last intersection count. */
  solidIntersections: number;
  /** Vertices with a moving-solid entry over the frame's rounds, and truncations its bound made. */
  bodyContacts: number;
  bodyCut: number;
  /**
   * A pair that conflicted in a held round (status heldConflict), ids as in the pair buffer: a
   * report-only example, four plain stores by whichever conflicting thread wrote last, so with
   * several conflicts at once its words may mix two of them. Nothing in a step reads it, and it
   * is the one counter outside the detector's bitwise determinism (the ladder's check skips it).
   */
  heldConflictPair: [number, number, number, number];
  /**
   * Impact zones with the moving solid: cloth vertices a conflict carried rigidly with it, summed
   * over the frame's steps, and the most in one step.
   */
  zoneVertices: number;
  zonePeak: number;
  /** The line-search filter over the frame's Newton iterations (newton config): vertices whose solid bounds it re-took about the iterate. */
  filterRebound: number;
  /** The filter's plane records (ogcProject): the frame's peak and the persistent peak (against planeRecordCapacity). */
  framePeakPlaneRecords: number;
  peakPlaneRecords: number;
  /** Over the frame: the rounds' longest prediction |w| and the longest direction |d| a line-search filter took (m). */
  maxPrediction: number;
  maxDirection: number;
  /** Over the frame: vertices the filters held because their plane records overflowed planeRecordCapacity, summed over
   * the filters, and the most in one filter. */
  planeRecordHeld: number;
  planeRecordHeldMax: number;
  /** Broad phase over the frame's rounds: bucket entries scanned, rejected for another cell's key (hash collisions),
   * rejected by box (or owned by another cell), and the longest single query's scan. */
  scanned: number; collisions: number; boxRejected: number; scanMax: number;
  /** Plane records the frame's line-search filters asked for, summed (each read at every trial of the ladder). */
  planeRecordsTotal: number;
  /** Levelled re-detection over the frame: its rounds, and those a conflict sent back unlevelled (w = 0). */
  levelled: number;
  levelFallback: number;
  /** Levelled rounds' conflicts: no plane at X, predictions closing (high 16 bits: with a solid in the pair). */
  levelConflicts: number[];
}

interface Kernel { pipeline: GPUComputePipeline; bindGroup?: GPUBindGroup; bindings: number[]; override?: Record<number, "positions"> }

const WG = 64;
/** Each round kernel's indirect-argument slot; the query's roles share the query's. */
const DISPATCH = { prep: 0, cell: 1, count: 2, scan1: 3, scan2: 4, scan3: 5, insert: 6, queryVertex: 7, queryEdge: 7, querySample: 7, bound: 8 } as const;
/** The detection round's kernels in dispatch order (for profiling). */
export const OGC_ROUND_KERNELS = Object.keys(DISPATCH) as (keyof typeof DISPATCH)[];
/** Hull edges the contact buffer holds per authored surface sample (a sample's fan holds a few neighbours past it). */
export const OGC_HULL_EDGES_PER_SAMPLE = 8;
const FLOAT32_EPS = 2 ** -23;

export class OgcContactDetector {
  readonly N: number; readonly T: number; readonly E: number;
  readonly topology: OgcTopology;
  readonly params: GPUBuffer; readonly topo: GPUBuffer;
  /** x_det (xyz) and b_v (w), vec4<f32> per vertex. */
  readonly det: GPUBuffer;
  /** Predictions w_v (vec4 per vertex), then each vertex's predicted step end and residual estimate. */
  readonly pred: GPUBuffer;
  readonly work: GPUBuffer; readonly counts: GPUBuffer; readonly grid: GPUBuffer;
  /** Header, per-vertex counts and contact entries (see ogc-wgsl.ts). */
  readonly contacts: GPUBuffer;
  /** DAT division pairs, two vec4<u32> each (see ogc-detector-wgsl.ts). */
  readonly pairs: GPUBuffer;
  readonly mode: OgcTruncationMode;
  readonly prediction: OgcPrediction;
  readonly counters: GPUBuffer; readonly args: GPUBuffer;
  /** The planes dispatch's indirect args, written by the round's bound pass. */
  readonly planesArgs: GPUBuffer;
  /**
   * The line-search filter's output (newton config): N + 1 f32 words, alpha_v of each vertex
   * and then alpha_min = min_v alpha_v. A 4-byte placeholder otherwise.
   */
  readonly alpha: GPUBuffer;
  /**
   * The contact state of every vertex at the end of the last step (encodeContactState, the
   * stepEnd hook): one u32 per vertex, bit 0 touching (a normal contact force on it at x_{n+1}),
   * bit 1 its contact onset (touching, and not at the end of the step before). Read only.
   */
  readonly contactState: GPUBuffer;
  private kernels = new Map<string, Kernel>();
  private readonly paramWords = new Uint32Array(OGC_PARAM_WORDS);
  /** The envelope cap in effect: rqMax, or with envelopeMeshCap the smaller of it and gamma_r h_min. */
  readonly rqTop: number;
  /** The truncation ratio gamma_r and the separation sepEps in effect (the DAT test's caps and margin). */
  readonly gammaR: number; readonly sepEps: number;

  private inputs?: OgcSolverInputs;

  private constructor(private readonly device: GPUDevice, topology: OgcTopology, readonly config: OgcContactConfig, coordinateScale: number) {
    const L = topology.layout;
    this.topology = topology;
    this.N = L.vertexCount; this.T = L.triangleCount; this.E = L.edgeCount;
    const { N, T, E } = this;
    const H = config.bucketCount;
    if (!(H > 0 && (H & (H - 1)) === 0)) throw new Error("OGC: bucketCount must be a power of two");
    const B1 = Math.ceil(H / 1024), B2 = Math.ceil(B1 / 1024);
    if (B2 > 1024) throw new Error("OGC: bucketCount above 2^30 needs a fourth scan level");
    for (const [name, wg] of [["vertices", Math.ceil(N / WG)], ["vertices+edges", Math.ceil((N + E) / WG)], ["faces", Math.ceil(T / WG)], ["buckets", B1]] as const)
      if (wg > 65535) throw new Error(`OGC: ${name} exceed one dispatch dimension`);
    const gammaP = config.gammaP ?? 0.45, gammaE = config.gammaE ?? 0.01;
    if (!(gammaP > 0 && gammaP < 0.5)) throw new Error("OGC: gamma_p must lie in (0, 0.5)");
    if (!(config.r > 0 && config.rqBase >= config.r && config.rqMax >= config.rqBase)) throw new Error("OGC: need 0 < r <= rqBase <= rqMax");
    if (!(config.contactCapacity > 0 && config.pairCapacity > 0 && config.gridCapacity > 0 && config.roundCap >= 0)) throw new Error("OGC: capacities must be authored");
    const mode = config.truncation ?? "planar";
    if (!(mode in OGC_TRUNCATION)) throw new Error(`OGC: unknown truncation "${String(mode)}"`);
    this.mode = mode;
    const prediction = config.prediction ?? "inertial";
    if (!(prediction in OGC_PREDICTION)) throw new Error(`OGC: unknown prediction "${String(prediction)}"`);
    this.prediction = prediction;
    const metric = config.projectionMetric ?? "off";
    if (metric !== "off" && metric !== "hvv") throw new Error(`OGC: unknown projection metric "${String(metric)}"`);
    if (metric !== "off" && !config.newton) throw new Error("OGC: a projection metric needs the Newton filter (newton)");
    const cellRule = config.cellRule ?? "largest";
    if (!(cellRule in OGC_CELL_RULE)) throw new Error(`OGC: unknown cell rule "${String(cellRule)}"`);
    const cell = config.cellSize ?? Math.max(L.meanEdgeLength, config.rqBase);
    if (!(cell > 0)) throw new Error("OGC: cellSize must be positive");
    const gammaR = config.gammaR ?? 0.9;
    if (!(gammaR > 0 && gammaR < 1)) throw new Error("OGC: gamma_r must lie in (0, 1)");
    const rqTop = config.envelopeMeshCap ? Math.min(config.rqMax, gammaR * L.minAltitude) : config.rqMax;
    if (!(rqTop >= config.r)) throw new Error(`OGC: the envelope cap gamma_r h_min = ${rqTop} m is below the contact radius ${config.r} m (a finer mesh needs a smaller r)`);
    this.rqTop = rqTop;
    const sepEps = config.separationEps ?? Math.max(1e-6, 4 * FLOAT32_EPS * coordinateScale);
    if (!(sepEps > 0 && sepEps < 0.5 * config.r)) throw new Error("OGC: separationEps must lie in (0, r/2)");
    this.gammaR = gammaR; this.sepEps = sepEps;
    if (2 * config.pairCapacity * 16 > device.limits.maxStorageBufferBindingSize) throw new Error("OGC: pairCapacity exceeds one storage binding");
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    const mk = (label: string, bytes: number, u = usage) => device.createBuffer({ label: `ogc ${label}`, size: Math.max(16, Math.ceil(bytes / 16) * 16), usage: u });
    this.params = mk("params", 4 * OGC_PARAM_WORDS, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    this.topo = mk("topology", 4 * topology.words.length);
    const sampleCap = config.solidSampleCapacity ?? 0;
    if (!(Number.isSafeInteger(sampleCap) && sampleCap >= 0)) throw new Error("OGC: solidSampleCapacity must be a whole number");
    this.solidSampleCapacity = sampleCap;
    const body = config.body;
    if (body) {
      for (const [k, v] of Object.entries(body)) if (!(Number.isSafeInteger(v) && v >= 0)) throw new Error(`OGC: body.${k} must be a whole number`);
      if (body.capacityWords < SOLID_HEADER_WORDS + 4) throw new Error("OGC: body.capacityWords cannot hold an empty solid image");
      if (body.poseCapacity < 2) throw new Error("OGC: body.poseCapacity must hold a step's start and end poses");
      if ((config.truncation ?? "planar") !== "planar") throw new Error("OGC: a moving solid needs planar truncation");
    }
    const bodySampleCap = body?.sampleCapacity ?? 0;
    this.bodySampleCapacity = bodySampleCap;
    this.det = mk("x_det", 16 * (N + sampleCap + bodySampleCap));
    this.pred = mk("predictions", 32 * N + 16 * bodySampleCap);
    this.work = mk("work", 4 * ((body ? 28 : 16) * N + OGC_SHIFT_WORDS * N + T + E + (config.newton ? 5 * N : 0)));
    this.counts = mk("grid counts", 4 * H);
    const local1Off = 0, sums1Off = H, local2Off = H + B1, sums2Off = H + 2 * B1, local3Off = H + 2 * B1 + B2, sums3Off = H + 2 * B1 + 2 * B2;
    // Entries of 8 words: face, cell key, the face's round box (wgsl GRID_READ_FNS).
    const entriesOff = Math.ceil((sums3Off + 1) / 8) * 8;
    if (4 * (entriesOff + 8 * config.gridCapacity) > device.limits.maxStorageBufferBindingSize) throw new Error("OGC: gridCapacity exceeds one storage binding");
    this.grid = mk("grid", 4 * (entriesOff + 8 * config.gridCapacity));
    const countsBase = 12, entriesBase = 12 + Math.ceil(N / 4);
    const solidWords = config.solidCapacityWords ?? 0;
    if (!(Number.isSafeInteger(solidWords) && solidWords >= 0)) throw new Error("OGC: solidCapacityWords must be a whole number");
    if (solidWords > 0 && solidWords < SOLID_HEADER_WORDS + 4) throw new Error("OGC: solidCapacityWords cannot hold an empty solid image");
    this.solidBase = solidWords > 0 ? 4 * (entriesBase + N * config.contactCapacity) : 0;
    this.solidCapacityWords = solidWords;
    if (sampleCap > 0 && !solidWords) throw new Error("OGC: solidSampleCapacity needs solidCapacityWords");
    this.sampleBase = Math.ceil((4 * (entriesBase + N * config.contactCapacity) + solidWords) / 4);
    // The samples' hull edges, two words each, OGC_HULL_EDGES_PER_SAMPLE per sample authored.
    const edgeVec4s = (cap: number) => Math.ceil(OGC_HULL_EDGES_PER_SAMPLE * cap / 2);
    this.edgeBase = this.sampleBase + sampleCap;
    // The moving solid: its image (word offset), its body-frame samples, their hull edges and its pose table (vec4 offsets).
    this.bodyBase = body ? 4 * (this.edgeBase + edgeVec4s(sampleCap)) : 0;
    this.bodySampleBase = body ? Math.ceil((this.bodyBase + body.capacityWords) / 4) : 0;
    this.bodyEdgeBase = body ? this.bodySampleBase + bodySampleCap : 0;
    this.poseTableBase = body ? this.bodyEdgeBase + edgeVec4s(bodySampleCap) : 0;
    this.poseCapacity = body?.poseCapacity ?? 0;
    const solidVec4s = body ? this.poseTableBase + 1 + (OGC_POSE_WORDS / 4) * body.poseCapacity : this.edgeBase + edgeVec4s(sampleCap);
    // The filter's records for ogcProject (newton config): a sub-header, each vertex's fixed
    // records and segment (first, count), and the plane records.
    const recCap = config.newton ? config.planeRecordCapacity ?? 16 * N : 0;
    if (!(Number.isSafeInteger(recCap) && recCap >= 0)) throw new Error("OGC: planeRecordCapacity must be a whole number");
    this.planeRecordCapacity = recCap;
    this.projBase = config.newton ? solidVec4s : 0;
    // The sub-header, then the level header (OGC_LEVEL), then the fixed records.
    const fixedBase = this.projBase + 1 + OGC_LEVEL_VEC4S, segBase = fixedBase + OGC_PROJECT_RECORD * N, recBase = segBase + Math.ceil(N / 2);
    // The records' far bits (one per slot, ogcRecordFar) follow the pool.
    const contactVec4s = config.newton ? recBase + recCap + Math.ceil(recCap / 128) : solidVec4s;
    this.contacts = mk("contacts", 16 * contactVec4s);
    if (this.contacts.size > device.limits.maxStorageBufferBindingSize) throw new Error("OGC: the contact buffer exceeds one storage binding");
    this.pairs = mk("dat pairs", 32 * config.pairCapacity);
    this.counters = mk("counters", 4 * OGC_COUNTER_WORDS);
    this.args = mk("args", 4 * OGC_ARGS_WORDS, GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
    this.planesArgs = mk("planes args", 12, GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
    this.alpha = mk("filter alpha", config.newton ? 4 * (N + 1) : 4);
    this.contactState = mk("contact state", 4 * N);
    const w = this.paramWords, f = new Float32Array(w.buffer), P = OGC_PARAM;
    w[P.N] = N; w[P.T] = T; w[P.E] = E; w[P.H] = H;
    w[P.gridCap] = config.gridCapacity; w[P.contactCap] = config.contactCapacity; w[P.roundCap] = config.roundCap;
    w[P.exceedThreshold] = Math.max(1, Math.ceil(gammaE * N));
    f[P.r] = config.r; f[P.rqBase] = config.rqBase; f[P.rqMax] = config.rqMax; f[P.rqTop] = this.rqTop; f[P.envScale] = config.envelopeScale;
    f[P.gammaP] = gammaP; f[P.cellFloor] = cell; f[P.dt] = config.dt;
    f[P.intersectCell] = Math.max(L.meanEdgeLength, 1e-6);
    f.set(config.gravity, P.gravity);
    w[P.trisOff] = L.trisOffset; w[P.edgesOff] = L.edgesOffset; w[P.triEdgesOff] = L.triEdgesOffset;
    w[P.vtOffOff] = L.vtOffOffset; w[P.vtListOff] = L.vtListOffset; w[P.veOffOff] = L.veOffOffset; w[P.veListOff] = L.veListOffset;
    w[P.B1] = B1; w[P.B2] = B2;
    w[P.local1Off] = local1Off; w[P.sums1Off] = sums1Off; w[P.local2Off] = local2Off; w[P.sums2Off] = sums2Off;
    w[P.local3Off] = local3Off; w[P.sums3Off] = sums3Off; w[P.entriesOff] = entriesOff;
    w[P.countsOff] = 4 * countsBase; w[P.entriesBase] = 4 * entriesBase;
    w[P.dminTOff] = N; w[P.dminEOff] = N + T; w[P.envOff] = N + T + E;
    w[P.wgN] = Math.ceil(N / WG); w[P.wgT] = Math.ceil(T / WG); w[P.wgNE] = Math.ceil((N + E) / WG);
    w[P.wgS1] = B1; w[P.wgS2] = B2; f[P.kc] = config.kc;
    w[P.mode] = OGC_TRUNCATION[mode]; f[P.gammaR] = gammaR; w[P.pairCap] = config.pairCapacity;
    w[P.tvOff] = 2 * N + T + E; f[P.sepEps] = sepEps;
    w[P.solidBase] = this.solidBase; w[P.solidOff] = 3 * N + T + E;
    w[P.heldOff] = 7 * N + T + E; w[P.predict] = OGC_PREDICTION[prediction]; w[P.cellRule] = OGC_CELL_RULE[cellRule];
    w[P.bodyBase] = this.bodyBase; w[P.bodyOff] = body ? 8 * N + T + E : 0; w[P.SB] = 0; w[P.wgPrep] = Math.ceil(N / WG);
    w[P.edgeBase] = this.edgeBase; w[P.bodyEdgeBase] = this.bodyEdgeBase; w[P.SE] = 0; w[P.SEB] = 0;
    // The per-axis deviations D of the solids' terms (OgcTerms) follow the body's terms.
    w[P.solidDevOff] = (body ? 16 : 8) * N + T + E; w[P.bodyDevOff] = body ? 20 * N + T + E : 0;
    // The least shift's words of the mandatory round's redos (OGC_SHIFT_WORDS).
    w[P.scaleOff] = (body ? 24 : 12) * N + T + E;
    // The filter's re-bound origins (vec4 per vertex) follow the least shift's words.
    w[P.originOff] = (body ? 24 : 12) * N + OGC_SHIFT_WORDS * N + T + E;
    // The filter's per-vertex plane-record counts follow the origins; its records are in the contact buffer.
    w[P.recCountOff] = (body ? 28 : 16) * N + OGC_SHIFT_WORDS * N + T + E;
    w[P.fixedWord] = 4 * fixedBase; w[P.segWord] = 4 * segBase; w[P.recWord] = 4 * recBase; w[P.recCap] = recCap; w[P.newton] = config.newton ? 1 : 0;
    w[P.levelWord] = config.newton ? 4 * (this.projBase + 1) : 0; w[P.levelOn] = 0;
    // The filter's Newton targets x + d (vec4 per vertex) follow the record counts.
    w[P.targetOff] = config.newton ? w[P.recCountOff]! + N : 0;
    device.queue.writeBuffer(this.params, 0, w);
    // No step has measured a residual yet (w < 0): the first uses the inertial displacement.
    const pred0 = new Float32Array(8 * N);
    for (let v = 0; v < N; v++) pred0[4 * (N + v) + 3] = -1;
    writeBufferChunked(device.queue, this.pred, 0, pred0);
    writeBufferChunked(device.queue, this.topo, 0, topology.words);
    const hdr = new Uint32Array(16 + 2 * OGC_POSE_WORDS), hf = new Float32Array(hdr.buffer);
    hdr[0] = N; hdr[1] = config.contactCapacity; hdr[2] = countsBase; hdr[3] = entriesBase;
    hf[4] = config.r; hf[5] = config.kc; hf[6] = config.mu; hf[7] = config.epsV * config.dt;
    hdr[8] = this.solidBase; hdr[9] = this.sampleBase; hdr[10] = 0; hdr[11] = this.projBase;
    hdr[12] = this.bodyBase; hdr[13] = this.bodySampleBase; hdr[14] = 0; hdr[15] = this.poseTableBase;
    // Identity poses until the owner uploads some (no step runs on them: the pose table is empty).
    for (const k of [0, 1]) { hf[16 + OGC_POSE_WORDS * k + 4] = 1; hf[16 + OGC_POSE_WORDS * k + 9] = 1; hf[16 + OGC_POSE_WORDS * k + 14] = 1; }
    device.queue.writeBuffer(this.contacts, 0, hdr);
    if (this.projBase) device.queue.writeBuffer(this.contacts, 16 * this.projBase, new Uint32Array([fixedBase, 4 * segBase, recBase, recCap]));
    // No solid until one is set: one empty slot, no bricks, an unbounded band.
    const empty = new Uint32Array(SOLID_HEADER_WORDS + 4), ef = new Float32Array(empty.buffer);
    ef[2] = 1; ef[3] = 3.0e38; empty[SOLID_HEADER_WORDS + 3] = OGC_NONE;
    if (this.solidBase) device.queue.writeBuffer(this.contacts, 4 * this.solidBase, empty);
    if (this.bodyBase) {
      device.queue.writeBuffer(this.contacts, 4 * this.bodyBase, empty);
      device.queue.writeBuffer(this.contacts, 16 * this.poseTableBase, new Uint32Array(4));
    }
  }

  /** The filter's records for ogcProject: their sub-header's vec4 (0: no newton config) and the plane records' capacity. */
  readonly projBase: number;
  readonly planeRecordCapacity: number;
  /** The moving solid: its image's word offset (0: none), its samples' and pose table's vec4 offsets, and capacities. */
  readonly bodyBase: number;
  readonly bodySampleBase: number;
  readonly poseTableBase: number;
  readonly bodySampleCapacity: number;
  readonly poseCapacity: number;
  bodySamples = 0;
  /** Radius about the body origin that holds every stored brick of the moving solid's field. */
  bodyReach = 0;
  private bodyField?: ClothSolidField;
  /** Whether the frame's poses were uploaded since the last frame began. */
  private posesPending = false;

  /**
   * Upload the moving solid's field, in its body frame (setup and edits of that
   * solid only; never in a step path), with its sharp features' surface samples.
   * Refuses an image or sample set beyond the authored capacity and a band too thin
   * for the envelope, as setSolidField does.
   */
  setBodyField(field: ClothSolidField) {
    const body = this.config.body;
    if (!body) throw new Error("OGC: the detector was built without a moving solid (config.body)");
    if (field.words.length > body.capacityWords)
      throw new Error(`OGC: the moving solid's field needs ${field.words.length} words, the scene authored ${body.capacityWords}`);
    const need = this.config.r + 1.5 * field.maxLipschitz * this.config.rqMax;
    if (!(field.band >= need)) throw new Error(`OGC: the moving solid's band ${field.band} m is below r + 1.5 L_max rqMax = ${need} m`);
    writeBufferChunked(this.device.queue, this.contacts, 4 * this.bodyBase, field.words);
    const samples = clothSolidSamples(field, { r: this.config.r, edge: this.topology.layout.maxEdgeLength });
    const SB = samples.length / SOLID_SAMPLE_WORDS;
    if (SB > this.bodySampleCapacity)
      throw new Error(`OGC: the moving solid has ${SB} surface samples at its sharp features, the scene authored ${this.bodySampleCapacity} (body.sampleCapacity)`);
    if (SB) writeBufferChunked(this.device.queue, this.contacts, 16 * this.bodySampleBase, samples);
    this.bodySamples = SB;
    this.bodySampleEdges = this.uploadHullEdges(field, this.bodyEdgeBase, this.bodySampleCapacity, "the moving solid", "body.sampleCapacity");
    this.paramWords[OGC_PARAM.SEB] = this.bodySampleEdges;
    this.bodyField = field;
    // Every stored brick lies within bodyReach of the body origin: a ball clear of it reads the band.
    const span = 8 * field.cell;
    let reach = 0;
    for (const key of field.bricks.keys()) {
      const b = key.split(",").map(Number);
      let far2 = 0;
      for (let a = 0; a < 3; a++) { const lo = field.origin[a]! + b[a]! * span; far2 += Math.max(lo * lo, (lo + span) ** 2); }
      reach = Math.max(reach, Math.sqrt(far2));
    }
    this.bodyReach = Math.fround(reach * (1 + 2 ** -20));
    new Float32Array(this.paramWords.buffer)[OGC_PARAM.bodyReach] = this.bodyReach;
    this.device.queue.writeBuffer(this.contacts, 56, new Uint32Array([SB]));
    this.paramWords[OGC_PARAM.SB] = SB;
    this.writeDispatchSizes();
  }

  /**
   * The moving solid's poses for the frame about to be encoded: one per step
   * boundary, poses[k] and poses[k + 1] bounding the frame's step k, so a frame of
   * n steps passes n + 1 (an input upload, like an attachment's time: no readback).
   * Every frame with a moving solid must pass them before encodeFrameBegin.
   */
  setBodyPoses(poses: readonly OgcBodyPose[]) {
    if (!this.config.body) throw new Error("OGC: the detector was built without a moving solid (config.body)");
    if (poses.length < 2 || poses.length > this.poseCapacity)
      throw new Error(`OGC: ${poses.length} moving-solid poses for a frame; it takes 2 to ${this.poseCapacity} (body.poseCapacity)`);
    const words = new Uint32Array(4 + OGC_POSE_WORDS * poses.length), f = new Float32Array(words.buffer);
    words[0] = poses.length;
    poses.forEach((p, k) => {
      const [x, y, z, w] = p.orientation, n = Math.hypot(x, y, z, w);
      if (!(n > 0) || Math.abs(n - 1) > 1e-6 || !p.position.every(Number.isFinite)) throw new Error(`OGC: moving-solid pose ${k} needs a finite position and a unit quaternion`);
      const b = 4 + OGC_POSE_WORDS * k;
      f.set(p.position, b);
      // Columns of R(q).
      f.set([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)], b + 4);
      f.set([2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)], b + 8);
      f.set([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], b + 12);
    });
    this.device.queue.writeBuffer(this.contacts, 16 * this.poseTableBase, words);
    this.posesPending = true;
  }

  /** The samples' hull edges (clothSolidSampleEdges) into the contact buffer at vec4 `base`; returns their count. */
  private uploadHullEdges(field: ClothSolidField, base: number, sampleCapacity: number, what: string, knob: string): number {
    const edges = clothSolidSampleEdges(field, { r: this.config.r, edge: this.topology.layout.maxEdgeLength });
    const n = edges.length / 2;
    if (n > OGC_HULL_EDGES_PER_SAMPLE * sampleCapacity)
      throw new Error(`OGC: ${what}'s samples have ${n} hull edges, more than ${OGC_HULL_EDGES_PER_SAMPLE} per authored sample (${knob} ${sampleCapacity})`);
    if (n) writeBufferChunked(this.device.queue, this.contacts, 16 * base, edges);
    return n;
  }

  private writeDispatchSizes() {
    const S = this.solidSamples, SB = this.bodySamples, SE = this.solidSampleEdges + this.bodySampleEdges;
    if (Math.ceil((this.N + this.E + S + SB + SE) / WG) > 65535) throw new Error("OGC: vertices, edges, surface samples and their hull edges exceed one dispatch dimension");
    this.paramWords[OGC_PARAM.wgNE] = Math.ceil((this.N + this.E + S + SB + SE) / WG);
    this.paramWords[OGC_PARAM.wgPrep] = Math.ceil((this.N + SB) / WG);
    this.device.queue.writeBuffer(this.params, 0, this.paramWords);
  }

  /** Word offset of the solid image in the contact buffer (0: no solid) and its authored capacity. */
  readonly solidBase: number;
  readonly solidCapacityWords: number;
  /** The surface samples: their vec4 offset in the contact buffer, authored capacity, and the current field's count. */
  readonly sampleBase: number;
  readonly solidSampleCapacity: number;
  solidSamples = 0;
  /** The samples' hull edges: vec4 offsets in the contact buffer and the current counts (static, moving). */
  readonly edgeBase: number;
  readonly bodyEdgeBase: number;
  solidSampleEdges = 0;
  bodySampleEdges = 0;

  /**
   * Upload the static solid field (setup and solid edits only; never in a step
   * path). Refuses an image beyond the authored capacity, and a band too thin
   * for the envelope: every missing brick must lie further than
   * r + 1.5 L_max rqMax from where a vertex's round can reach.
   */
  setSolidField(field: ClothSolidField) {
    if (!this.solidBase) throw new Error("OGC: the detector was built without solidCapacityWords");
    if (field.words.length > this.solidCapacityWords)
      throw new Error(`OGC: the solid field needs ${field.words.length} words, the scene authored ${this.solidCapacityWords}`);
    const need = this.config.r + 1.5 * field.maxLipschitz * this.config.rqMax;
    if (!(field.band >= need)) throw new Error(`OGC: the solid field's band ${field.band} m is below r + 1.5 L_max rqMax = ${need} m`);
    writeBufferChunked(this.device.queue, this.contacts, 4 * this.solidBase, field.words);
    // The sharp features' surface samples, for this cloth's r and longest edge.
    const samples = clothSolidSamples(field, { r: this.config.r, edge: this.topology.layout.maxEdgeLength });
    const S = samples.length / SOLID_SAMPLE_WORDS;
    if (S > this.solidSampleCapacity)
      throw new Error(`OGC: the solid field has ${S} surface samples at its sharp features, the scene authored ${this.solidSampleCapacity} (solidSampleCapacity)`);
    if (S) {
      writeBufferChunked(this.device.queue, this.det, 16 * this.N, samples);
      writeBufferChunked(this.device.queue, this.contacts, 16 * this.sampleBase, samples);
    }
    this.solidSamples = S;
    this.solidSampleEdges = this.uploadHullEdges(field, this.edgeBase, this.solidSampleCapacity, "the solid field", "solidSampleCapacity");
    this.paramWords[OGC_PARAM.SE] = this.solidSampleEdges;
    this.device.queue.writeBuffer(this.contacts, 40, new Uint32Array([S]));
    this.paramWords[OGC_PARAM.S] = S;
    this.writeDispatchSizes();
  }

  /**
   * Build the detector and compile its kernels (asynchronously, outside any
   * constructor). Without inputs, `bind` attaches the solver's buffers later:
   * the solver itself is built around `solverBuffer`.
   */
  static async create(device: GPUDevice, mesh: OgcMeshTopology, restPositions: Float32Array, config: OgcContactConfig, inputs?: OgcSolverInputs): Promise<OgcContactDetector> {
    let scale = 1;
    for (const c of restPositions) scale = Math.max(scale, Math.abs(c));
    const d = new OgcContactDetector(device, buildOgcTopology(mesh, restPositions), config, scale);
    await d.compile();
    if (inputs) d.bind(inputs);
    return d;
  }

  /** The buffer the solver kernel binds for ogcVertexBlockWGSL. */
  get solverBuffer(): GPUBuffer { return this.contacts; }

  /** metric: the host's fn(v) -> mat3x3<f32> returning H_vv^-1, the projection's metric (OgcVertexBlockOptions.metric). */
  static vertexBlockWGSL(group: number, binding: number, newton = false, metric?: string): string { return ogcVertexBlockWGSL({ group, binding, newton, ...(metric ? { metric } : {}) }); }

  /**
   * What the cloth solver's vertex kernel takes: the block at (group 0, binding) and the buffer it
   * reads; with newton, the global forms too (ogcVertexContactHvLanes, ogcContactEnergyOwnedLanes), and the trial
   * projection in the config's projectionMetric ("hvv": the solver's nwVertexInv).
   */
  solverContact(binding: number, newton = false): { wgsl: string; buffer: GPUBuffer } {
    const metric = newton && this.config.projectionMetric === "hvv" ? { metric: "nwVertexInv" } : {};
    return { wgsl: ogcVertexBlockWGSL({ group: 0, binding, newton, ...metric }), buffer: this.contacts };
  }

  /**
   * The step hooks for one implicit step of the cloth solver (its ClothStepHooks):
   * the mandatory round and a truncation of the initial guess after the step
   * frame, and after every iteration a truncation of the iteration's accumulated
   * moves, the decision and the (indirect, usually empty) re-detection round.
   * Truncation runs once per iteration, not per colour: it cuts the accumulated
   * displacement against every plane and the solid bound before any detection or
   * commit, so it is as safe; colours within an iteration see untruncated
   * neighbours only in their Gauss-Seidel gradient (rotating sphere, 22.5K
   * vertices, K = 64: 622 -> 244 ms a tick, no intersections either way).
   */
  get stepHooks() {
    return {
      afterFrame: (pass: GPUComputePassEncoder) => { this.encodeStepBegin(pass); this.encodeDetection(pass); this.encodeTruncate(pass); },
      afterIteration: (pass: GPUComputePassEncoder) => { this.encodeTruncate(pass); this.encodeDecide(pass); this.encodeDetection(pass, false); },
      // The solver's two-level correction moves every vertex at once: truncate that move too.
      afterCoarse: (pass: GPUComputePassEncoder) => { this.encodeTruncate(pass); },
      // A Newton solver (newton config): filter its direction, then after its step re-detect.
      filter: (pass: GPUComputePassEncoder) => { this.encodeFilter(pass); },
      afterNewtonStep: (pass: GPUComputePassEncoder) => { this.encodeAfterNewtonStep(pass); },
      // Once per step, after the last position update and before velocities and history.
      stepEnd: (pass: GPUComputePassEncoder) => { this.encodeContactState(pass); },
    };
  }

  private async compile() {
    const specs: [string, string, number[], Record<number, "positions">?][] = [
      ["prep", OGC_PREP_WGSL, [0, 2, 3, 4, 5, 8, 9, 13, 14, 15, 17]],
      ["cell", OGC_CELL_WGSL, [0, 9]],
      ["count", ogcGridCountWGSL(false), [0, 1, 4, 5, 6, 9, 14]],
      ["scan1", ogcScanWGSL(1), [0, 6, 7]],
      ["scan2", ogcScanWGSL(2), [0, 7]],
      ["scan3", ogcScanWGSL(3), [0, 7]],
      ["insert", ogcGridInsertWGSL(false), [0, 1, 4, 5, 6, 7, 9, 14]],
      ...OGC_QUERY_ROLES.map(role => [role, ogcQueryWGSL(role), [0, 1, 4, 5, 7, 8, 9, 11, 14]] as [string, string, number[]]),
      ["bound", OGC_BOUND_WGSL, [0, 1, 4, 5, 8, 9, 12]],
      ["stepBegin", OGC_STEP_BEGIN_WGSL, [0, 8, 9, 10]],
      ["decide", ogcDecideWGSL(!!this.config.newton), this.config.newton ? [0, 8, 9, 10, 19] : [0, 9, 10]],
      ["redo", OGC_REDO_WGSL, [0, 8, 9, 10]],
      ["redoLast", OGC_REDO_LAST_WGSL, [0, 9, 10]],
      ["frameBegin", ogcFrameBeginWGSL(OGC_FRAME_COUNTERS), [9]],
      ["planes", ogcPlanesWGSL(!!this.config.newton, "truncate"), [0, 2, 4, 5, 9, 11, 14]],
      ...(this.config.newton ? [
        ["filterBegin", OGC_FILTER_BEGIN_WGSL, [0, 8, 9, 16]],
        ["filterPlanes", ogcPlanesWGSL(true, "filter"), [0, 2, 4, 5, 8, 9, 11, 14, 15]],
        ["filterAlloc", OGC_FILTER_ALLOC_WGSL, [0, 5, 8, 9]],
        ["filterRecords", ogcPlanesWGSL(true, "records"), [0, 2, 4, 5, 8, 9, 11, 14, 15]],
        ["filter", OGC_FILTER_WGSL, [0, 2, 4, 5, 8, 9, 14, 15, 16]],
      ] as [string, string, number[]][] : []),
      ["apply", OGC_APPLY_WGSL, [0, 2, 4, 5, 8, 9, 14]],
      ["contactState", OGC_CONTACT_STATE_WGSL, [0, 1, 2, 5, 8, 13, 18]],
      // Intersection counting reads the solver's current positions where detection reads x_det.
      ["xCount", ogcGridCountWGSL(true), [0, 1, 4, 6, 9], { 4: "positions" }],
      ["xInsert", ogcGridInsertWGSL(true), [0, 1, 4, 6, 7, 9], { 4: "positions" }],
      ["xTest", OGC_INTERSECT_WGSL, [0, 1, 4, 7, 8, 9], { 4: "positions" }],
    ];
    await Promise.all(specs.map(async ([name, code, bindings, override]) => {
      const module = this.device.createShaderModule({ label: `ogc ${name}`, code });
      const pipeline = await this.device.createComputePipelineAsync({ label: `ogc ${name}`, layout: "auto", compute: { module, entryPoint: "main" } });
      this.kernels.set(name, { pipeline, bindings, override });
    }));
  }

  /** Attach the solver's positions and step-start velocities (once, at setup). */
  bind(inputs: OgcSolverInputs) {
    if (this.inputs) throw new Error("OGC: the detector is already bound to a solver");
    if (this.config.newton && !inputs.direction) throw new Error("OGC: a newton detector needs the solver's direction buffer");
    if (!this.config.newton && inputs.direction) throw new Error("OGC: a direction is bound only with the newton config");
    if (!this.config.newton && inputs.control) throw new Error("OGC: a Newton control record is bound only with the newton config");
    if (this.prediction === "guess" && !inputs.stepStart) throw new Error('OGC: prediction "guess" needs the solver\'s stepStart (x_t) with the guess in positions');
    if (this.prediction === "target" && !(inputs.stepStart && inputs.target)) throw new Error('OGC: prediction "target" needs the solver\'s stepStart (x_t) and its inertial target y');
    if (this.prediction !== "target" && inputs.target) throw new Error('OGC: a target is bound only with prediction "target"');
    this.inputs = inputs;
    const res = (b: GPUBuffer | GPUBufferBinding): GPUBufferBinding => ("buffer" in b ? b : { buffer: b });
    const all: Record<number, GPUBufferBinding> = {
      0: res(this.params), 1: res(this.topo), 2: res(inputs.positions), 3: res(inputs.velocities), 4: res(this.det), 5: res(this.work),
      6: res(this.counts), 7: res(this.grid), 8: res(this.contacts), 9: res(this.counters), 10: res(this.args), 11: res(this.pairs), 12: res(this.planesArgs),
      13: res(inputs.stepStart ?? inputs.positions), 14: res(this.pred),
      15: res(inputs.direction ?? inputs.positions), 16: res(this.alpha), 17: res(inputs.target ?? inputs.positions),
      18: res(this.contactState), 19: res(inputs.control ?? this.alpha),
    };
    // Levelled re-detection: with the solver's control, for planar predicted DAT (held or isotropic rounds predict 0).
    const levelOn = !!inputs.control && this.paramWords[OGC_PARAM.mode] === OGC_TRUNCATION.planar && this.paramWords[OGC_PARAM.predict] !== 0;
    this.paramWords[OGC_PARAM.levelOn] = levelOn ? 1 : 0;
    this.device.queue.writeBuffer(this.params, 4 * OGC_PARAM.levelOn, new Uint32Array([this.paramWords[OGC_PARAM.levelOn]!]));
    for (const [name, k] of this.kernels) {
      k.bindGroup = this.device.createBindGroup({
        label: `ogc ${name}`, layout: k.pipeline.getBindGroupLayout(0),
        entries: k.bindings.map(i => ({ binding: i, resource: k.override?.[i] === "positions" ? all[2]! : all[i]! })),
      });
    }
  }

  private kernel(name: string): Kernel & { bindGroup: GPUBindGroup } {
    const k = this.kernels.get(name)!;
    if (!k.bindGroup) throw new Error("OGC: bind the solver's buffers before encoding");
    return k as Kernel & { bindGroup: GPUBindGroup };
  }

  private run(pass: GPUComputePassEncoder, name: string, workgroups: number) {
    const k = this.kernel(name);
    pass.setPipeline(k.pipeline); pass.setBindGroup(0, k.bindGroup); pass.dispatchWorkgroups(workgroups);
  }
  private runIndirect(pass: GPUComputePassEncoder, name: keyof typeof DISPATCH) {
    const k = this.kernel(name);
    pass.setPipeline(k.pipeline); pass.setBindGroup(0, k.bindGroup); pass.dispatchWorkgroupsIndirect(this.args, 12 * DISPATCH[name]);
  }

  /** Reset the per-frame statistics (status and peaks persist). */
  encodeFrameBegin(pass: GPUComputePassEncoder) {
    if (this.bodyField && !this.posesPending) throw new Error("OGC: a frame with a moving solid needs its poses (setBodyPoses) before it begins");
    this.posesPending = false;
    this.run(pass, "frameBegin", 1);
  }
  /** Start of a step: arm the mandatory detection round. */
  encodeStepBegin(pass: GPUComputePassEncoder) { this.run(pass, "stepBegin", 1); }
  /**
   * One detection round, dispatched indirectly: zero workgroups unless armed;
   * then the redo decisions, each with the round again (zero workgroups unless the
   * last had a conflict). The step's mandatory round (after encodeStepBegin) needs the
   * mandatory count of redos; a later round (after encodeDecide) passes false.
   */
  encodeDetection(pass: GPUComputePassEncoder, mandatory = true) {
    for (const name of OGC_ROUND_KERNELS) this.runIndirect(pass, name);
    // The mandatory round's least-shift redos and fail-safe (OGC_REDO_WGSL), with a moving solid its
    // impact zones too (redoCap, as the kernel counts it). A later round predicts 0 about the
    // iterate, the moving solid at rest at its end pose (a_v = 0 exactly): only its levelled form
    // can conflict, and its one redo (unlevelled) cannot; a conflict there fails as heldConflict.
    const slots = !mandatory ? 1 : this.bodyField ? OGC_REDO_MANDATORY : OGC_REDO_ROUNDS + 1;
    for (let k = 0; k < slots; k++) {
      this.run(pass, "redo", 1);
      for (const name of OGC_ROUND_KERNELS) this.runIndirect(pass, name);
    }
    this.run(pass, "redoLast", 1);
  }
  /** One kernel of a detection round, indirectly, or the redo decision (profiling only). */
  encodeRoundKernel(pass: GPUComputePassEncoder, name: keyof typeof DISPATCH | "redo") {
    if (name === "redo") this.run(pass, "redo", 1); else this.runIndirect(pass, name);
  }
  /** The truncation's two kernels, one at a time (profiling only). */
  encodeTruncateKernel(pass: GPUComputePassEncoder, name: "planes" | "apply") {
    if (name === "apply") { this.run(pass, "apply", Math.ceil(this.N / WG)); return; }
    const k = this.kernel("planes");
    pass.setPipeline(k.pipeline); pass.setBindGroup(0, k.bindGroup); pass.dispatchWorkgroupsIndirect(this.planesArgs, 0);
  }
  /** Between iterations: count the iteration's truncations and arm a round when gamma_e N were truncated. */
  encodeDecide(pass: GPUComputePassEncoder) { this.run(pass, "decide", 1); }
  /** Truncate the accumulated displacement x - x_det of every vertex: division planes (indirect), then apply. */
  encodeTruncate(pass: GPUComputePassEncoder) {
    const k = this.kernel("planes");
    pass.setPipeline(k.pipeline); pass.setBindGroup(0, k.bindGroup); pass.dispatchWorkgroupsIndirect(this.planesArgs, 0);
    this.run(pass, "apply", Math.ceil(this.N / WG));
  }
  /**
   * The line-search filter (newton config, OGC_FILTER_WGSL), once per Newton iteration, for the
   * iterate in positions (untouched) and the solver's direction d: the round's certificates of
   * each vertex for ogcProject, which projects the solver's trials onto them (the projected arc).
   * The planes pass counts each vertex's plane records (and reduces each plane's alpha into tv),
   * the allocation gives each vertex a contiguous segment, the records pass writes them, and the
   * filter re-takes the solid bounds a solid cuts d by about the iterate, writes each vertex's
   * fixed records (cap, solid pieces) and alpha_v = the largest alpha <= 1 with x + alpha d
   * inside every certificate into alpha[v], alpha_min into alpha[N]. Vertices bound below 1
   * count towards encodeDecide's re-detection, and each held at 0 arms it.
   */
  encodeFilter(pass: GPUComputePassEncoder) {
    if (!this.config.newton) throw new Error("OGC: encodeFilter needs the newton config");
    this.run(pass, "filterBegin", 1);
    for (const name of ["filterPlanes", "filterAlloc", "filterRecords"] as const) {
      if (name === "filterAlloc") { this.run(pass, name, Math.ceil(this.N / 256)); continue; }
      const k = this.kernel(name);
      pass.setPipeline(k.pipeline); pass.setBindGroup(0, k.bindGroup); pass.dispatchWorkgroupsIndirect(this.planesArgs, 0);
    }
    this.run(pass, "filter", Math.ceil(this.N / WG));
  }
  /**
   * After a Newton step x <- x + alpha d with alpha <= alpha_min: the decision, the (indirect,
   * usually empty) re-detection round and the truncation that re-anchors the iterate in a new
   * round's certificates (a no-op, to rounding, when no round ran: the iterate is certified).
   */
  /** The step's contact state (contactState), once per step after its final positions. */
  encodeContactState(pass: GPUComputePassEncoder) { this.run(pass, "contactState", Math.ceil(this.N / WG)); }
  encodeAfterNewtonStep(pass: GPUComputePassEncoder) {
    this.encodeDecide(pass); this.encodeDetection(pass, false); this.encodeTruncate(pass);
  }
  /** Count intersecting non-adjacent face pairs of the current positions (counter `intersections`). */
  encodeIntersectionCount(pass: GPUComputePassEncoder) {
    const H = this.config.bucketCount, B1 = Math.ceil(H / 1024), B2 = Math.ceil(B1 / 1024), wgT = Math.ceil(this.T / WG);
    this.run(pass, "xCount", Math.max(1, wgT));
    this.run(pass, "scan1", B1); this.run(pass, "scan2", B2); this.run(pass, "scan3", 1);
    this.run(pass, "xInsert", Math.max(1, wgT)); this.run(pass, "xTest", Math.max(1, wgT));
  }

  /**
   * Harness and gate use only (never in a step path): count, at the current positions, the
   * intersecting non-adjacent face pairs and the faces reaching into the static solid (phi < 0 at a
   * lattice of four points per cell along the longest edge, every face), in their own submission.
   */
  async countIntersections(): Promise<{ intersections: number; solidIntersections: number; solidHit?: { face: number; body: boolean; phi: number } }> {
    const enc = this.device.createCommandEncoder(), pass = enc.beginComputePass();
    this.encodeIntersectionCount(pass); pass.end();
    this.device.queue.submit([enc.finish()]);
    const c = await this.readCounters();
    const u = new Uint32Array(await this.readBuffer(this.counters, 4 * OGC_COUNTER.solidHitFace, 8)), w = u[0]!;
    const solidHit = w ? { face: (w & 0x7fffffff) - 1, body: (w & 0x80000000) !== 0, phi: new Float32Array(u.buffer)[1]! } : undefined;
    return { intersections: c.intersections, solidIntersections: c.solidIntersections, solidHit };
  }

  /**
   * The frame's round log (harness and gate use only): per detection round its longest
   * prediction and largest envelope, the candidates and DAT pairs it found and the held vertices
   * its prep counted, then its line-search filters: their count, the largest plane-record demand
   * (records the filter asked for, before the capacity) and the longest direction |d|.
   */
  async readRoundLog(): Promise<{ maxPrediction: number; maxEnvelope: number; candidates: number; datPairs: number; held: number; filters: number; planeRecords: number; maxDirection: number;
    scanned: number; collisions: number; boxRejected: number; planeRecordsTotal: number }[]> {
    const read = this.device.createBuffer({ size: 4 * OGC_COUNTER_WORDS, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(this.counters, 0, read, 0, 4 * OGC_COUNTER_WORDS);
    this.device.queue.submit([enc.finish()]);
    await read.mapAsync(GPUMapMode.READ);
    const u = new Uint32Array(read.getMappedRange().slice(0)), fl = new Float32Array(u.buffer);
    read.destroy();
    const C = OGC_COUNTER, n = Math.min(u[C.roundLogCount]!, OGC_ROUND_LOG), at = (i: number, k: number) => C.roundLog + OGC_ROUND_LOG_WORDS * i + k;
    // Candidates and pairs are found after the round's cell kernel (the next snapshot, or the frame's total, less this one);
    // its prep counted the held vertices before it (this snapshot less the last).
    const next = (i: number, k: number, total: number) => (i + 1 < n ? u[at(i + 1, k)]! : total) - u[at(i, k)]!;
    return Array.from({ length: n }, (_, i) => ({
      maxPrediction: fl[at(i, 0)]!, maxEnvelope: fl[at(i, 1)]!,
      candidates: next(i, 2, u[C.frameCandidates]!), datPairs: next(i, 3, u[C.frameDatPairs]!),
      held: u[at(i, 4)]! - (i ? u[at(i - 1, 4)]! : 0),
      filters: u[at(i, 5)]!, planeRecords: u[at(i, 6)]!, maxDirection: fl[at(i, 7)]!,
      scanned: next(i, 8, u[C.frameScanned]!), collisions: next(i, 9, u[C.frameCollisions]!), boxRejected: next(i, 10, u[C.frameBoxRejected]!),
      planeRecordsTotal: u[at(i, 11)]!,
    }));
  }

  /** Read the counters. Harness and gate use only: never in a step path. */
  async readCounters(): Promise<OgcFrameCounters> {
    const read = this.device.createBuffer({ size: 4 * OGC_COUNTER_WORDS, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(this.counters, 0, read, 0, 4 * OGC_COUNTER_WORDS);
    this.device.queue.submit([enc.finish()]);
    await read.mapAsync(GPUMapMode.READ);
    const u = new Uint32Array(read.getMappedRange().slice(0)), fl = new Float32Array(u.buffer);
    read.destroy();
    const C = OGC_COUNTER;
    return {
      status: u[C.status], steps: u[C.frameSteps], detections: u[C.frameDetections], redetections: u[C.frameRedetections],
      decides: u[C.frameDecides], truncated: u[C.frameTruncated], truncatedMax: u[C.frameTruncatedMax],
      pairEntries: u[C.framePairEntries], pairMax: u[C.framePairMax], candidates: u[C.frameCandidates],
      vfPairs: u[C.frameVfPairs], eePairs: u[C.frameEePairs], envelopeCapHits: u[C.frameEnvelopeCapHits],
      peakEnvelope: fl[C.framePeakEnvelope], roundCapHits: u[C.frameRoundCapHits], peakContacts: u[C.peakContacts],
      datPairs: u[C.frameDatPairs], framePeakPairSlots: u[C.framePeakPairSlots], peakPairSlots: u[C.peakPairSlots],
      capped: u[C.frameCapped], uncertified: u[C.frameUncertified],
      solidContacts: u[C.frameSolidContacts], solidCut: u[C.frameSolidCut],
      conflicts: u[C.frameConflicts], redos: u[C.frameRedos], held: u[C.frameHeld],
      peakGrid: u[C.peakGrid], intersections: u[C.intersections], cell: fl[C.cell],
      sfPairs: u[C.frameSfPairs], solidIntersections: u[C.solidIntersections],
      bodyContacts: u[C.frameBodyContacts], bodyCut: u[C.frameBodyCut],
      heldConflictPair: [u[C.heldConflictPair]!, u[C.heldConflictPair + 1]!, u[C.heldConflictPair + 2]!, u[C.heldConflictPair + 3]!],
      zoneVertices: u[C.frameZoneVertices]!, zonePeak: u[C.frameZonePeak]!,
      filterRebound: u[C.frameRebound]!, framePeakPlaneRecords: u[C.framePeakPlaneRecords]!, peakPlaneRecords: u[C.peakPlaneRecords]!,
      levelConflicts: [u[C.frameLevelConflicts]!, u[C.frameLevelConflicts + 1]!],
      maxPrediction: fl[C.frameMaxPrediction]!, maxDirection: fl[C.frameMaxDirection]!,
      planeRecordHeld: u[C.framePlaneRecordHeld]!, planeRecordHeldMax: u[C.framePlaneRecordHeldMax]!,
      scanned: u[C.frameScanned]!, collisions: u[C.frameCollisions]!, boxRejected: u[C.frameBoxRejected]!, scanMax: u[C.frameScanMax]!,
      planeRecordsTotal: u[C.framePlaneRecordsTotal]!,
      levelled: u[C.frameLevelled]!, levelFallback: u[C.frameLevelFallback]!,
    };
  }

  /**
   * The last round's per-vertex state, for diagnosis (harness only, never in a
   * step path): x_det of the N vertices and S samples, each vertex's prediction w
   * and step length, its residual estimate rho, its envelope r_q and held flags,
   * and the round's broad-phase cell.
   */
  /**
   * The last round's DAT division pairs (harness reads; never a step path): ids as in the pair buffer
   * (a hole has x = OGC_NONE) and planes (n, certified gap), over its slots (pairSlots, at most pairCapacity).
   */
  async readPairs(): Promise<{ ids: Uint32Array; planes: Float32Array; slots: number }> {
    const counters = new Uint32Array(await this.readBuffer(this.counters, 0, 4 * OGC_COUNTER_WORDS));
    const slots = Math.min(counters[OGC_COUNTER.pairSlots]!, this.config.pairCapacity);
    const raw = slots ? await this.readBuffer(this.pairs, 0, 32 * slots) : new ArrayBuffer(0);
    const all = new Uint32Array(raw), ids = new Uint32Array(4 * slots), planes = new Float32Array(4 * slots);
    for (let k = 0; k < slots; k++) { ids.set(all.subarray(8 * k, 8 * k + 4), 4 * k); planes.set(new Float32Array(all.buffer, 32 * k + 16, 4), 4 * k); }
    return { ids, planes, slots };
  }
  private async readBuffer(src: GPUBuffer, offset: number, size: number): Promise<ArrayBuffer> {
    const read = this.device.createBuffer({ size: Math.ceil(size / 16) * 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(src, offset, read, 0, Math.ceil(size / 4) * 4);
    this.device.queue.submit([enc.finish()]);
    await read.mapAsync(GPUMapMode.READ);
    const out = read.getMappedRange(0, Math.ceil(size / 4) * 4).slice(0);
    read.destroy();
    return out;
  }

  async readDiagnosis(): Promise<{ xdet: Float32Array; pred: Float32Array; rho: Float32Array; env: Float32Array; held: Uint32Array; cell: number }> {
    const N = this.N, T = this.T, E = this.E, S = this.solidSamples;
    const copies: [GPUBuffer, number, number][] = [[this.det, 0, 16 * (N + S + this.bodySampleCapacity)], [this.pred, 0, 32 * N + 16 * this.bodySampleCapacity], [this.work, 4 * (N + T + E), 4 * N], [this.work, 4 * (7 * N + T + E), 4 * N], [this.counters, 0, 4 * OGC_COUNTER_WORDS]];
    const reads = copies.map(([, , size]) => this.device.createBuffer({ size: Math.ceil(size / 16) * 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }));
    const enc = this.device.createCommandEncoder();
    copies.forEach(([src, off, size], i) => enc.copyBufferToBuffer(src, off, reads[i]!, 0, Math.ceil(size / 4) * 4));
    this.device.queue.submit([enc.finish()]);
    const out = await Promise.all(reads.map(async (b, i) => { await b.mapAsync(GPUMapMode.READ); const a = b.getMappedRange(0, Math.ceil(copies[i]![2] / 4) * 4).slice(0); b.destroy(); return a; }));
    const pred = new Float32Array(out[1]!), rho = new Float32Array(N);
    for (let v = 0; v < N; v++) rho[v] = pred[4 * (N + v) + 3]!;
    return { xdet: new Float32Array(out[0]!), pred, rho, env: new Float32Array(out[2]!), held: new Uint32Array(out[3]!), cell: new Float32Array(out[4]!)[OGC_COUNTER.cell]! };
  }

  /**
   * The per-vertex state one step hands the next, which is part of the cloth's
   * published state (an undo or load restores it with the positions): each vertex's
   * reference prediction and residual rho_v (pred[N + v]: xyz the predicted end
   * position, w = rho_v, or -1 before any step), and its held flags (bit 0 held in
   * this step, which the next step reads as held in the last). Reading is off the step
   * path (the frame's owner publishes it); everything else the detector holds is
   * rebuilt by every step.
   */
  async readCarriedState(): Promise<OgcCarriedState> {
    const N = this.N, size = [16 * N, 4 * N], src: [GPUBuffer, number][] = [[this.pred, 16 * N], [this.work, 4 * (7 * N + this.T + this.E)]];
    const reads = size.map(s => this.device.createBuffer({ size: s, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }));
    const enc = this.device.createCommandEncoder();
    src.forEach(([b, off], i) => enc.copyBufferToBuffer(b, off, reads[i]!, 0, size[i]!));
    this.device.queue.submit([enc.finish()]);
    const out = await Promise.all(reads.map(async b => { await b.mapAsync(GPUMapMode.READ); const a = b.getMappedRange().slice(0); b.destroy(); return a; }));
    return { reference: new Float32Array(out[0]!), held: new Uint32Array(out[1]!) };
  }

  /** Restore the state readCarriedState returned (for this detector's vertex count), or, with none, the state before any step. */
  writeCarriedState(state?: OgcCarriedState) {
    const N = this.N;
    if (state && (state.reference.length !== 4 * N || state.held.length !== N))
      throw new Error(`OGC: carried state for ${state.held.length} vertices, the detector has ${N}`);
    const reference = state?.reference ?? (() => { const r = new Float32Array(4 * N); for (let v = 0; v < N; v++) r[4 * v + 3] = -1; return r; })();
    writeBufferChunked(this.device.queue, this.pred, 16 * N, reference);
    const held = state?.held ?? new Uint32Array(N);
    writeBufferChunked(this.device.queue, this.work, 4 * (7 * N + this.T + this.E), held);
    writeBufferChunked(this.device.queue, this.contactState, 0, held.map(f => (f >> 8) & 3));
  }

  /** Explain a status word. */
  static describeStatus(status: number): string[] {
    return (Object.entries(OGC_STATUS) as [string, number][]).filter(([, bit]) => status & bit).map(([name]) => name);
  }

  destroy() {
    for (const buffer of [this.params, this.topo, this.det, this.pred, this.work, this.counts, this.grid, this.contacts, this.pairs, this.counters, this.args, this.planesArgs, this.alpha, this.contactState]) buffer.destroy();
  }
}
