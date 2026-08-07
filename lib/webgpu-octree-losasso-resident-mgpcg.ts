/**
 * Resident single-dispatch MGPCG executor for the Losasso coarse tier.
 *
 * The pipelined path (`WebGPUOctreePipelinedMGPCG` + `WebGPUOctreeLosassoVCycle`
 * + `WebGPUOctreeLosassoOperator`) encodes ~15 barrier-separated dispatches per
 * outer iteration to solve a system of ~1-2K unknowns, and its combined-drain
 * schedule cannot retire the encoded suffix after convergence: every encoded
 * iteration launches its full graph. This module runs the identical iteration
 * sequence — warm-seeded state, the closed-form axis-face operator with its
 * signed radix-256 superaccumulator fold, the 2+2 first-order V-cycle with the
 * fused sub-L0 walk, the exact-fixed-point initial reduction, the compensated
 * one-workgroup drains, the CG recurrences and the fail-closed publication —
 * inside ONE dispatch of ONE 256-lane workgroup, with
 * `storageBarrier(); workgroupBarrier();` where dispatch boundaries used to be.
 *
 * Two structural consequences:
 *  - The loop executes exactly as many iterations as the residual gate needs
 *    (temporal coherence is perfect by construction: a warm steady advance
 *    converging in 4 iterations pays for 4, never for an encoded envelope).
 *  - The encoded command graph for the whole solve is one compute pass and a
 *    handful of staging-free bindings, so the per-advance encode cost stops
 *    scaling with the iteration ceiling.
 *
 * Numeric contract: every stage is a transcription of the corresponding
 * dispatched kernel body (see the per-function notes). Cross-row folds keep
 * their exact shapes: the initial reduction deposits into the signed radix-256
 * integer limbs whose decoded f32 is partition-invariant by construction, and
 * the per-iteration drains reproduce the 128-lane compensated tree of
 * `reduceAndFinishMerged` lane-for-lane. The kernel writes the same MGPCG
 * control words to the same buffer, so snapshots and tripwires are unaffected.
 * `control[5]` (zeroed-dispatch accounting) is authored arithmetically, the
 * same convention `webgpu-octree-persistent-mgpcg.wgsl.ts` uses.
 *
 * Scope: this is a ≤4,096-row tier, matching the combined-drain ceiling and
 * the fused sub-L0 envelope. Larger systems keep the wide pipelined path.
 */
import { PassBroker } from "./webgpu-pass-broker";
import { octreeCompensatedF32WGSL } from "./octree-compensated-f32.wgsl";
import type { OctreeLosassoVCycleHierarchySource } from "./webgpu-octree-losasso-vcycle";
import {
  OCTREE_RUNTIME_DIAL_BUILT_BOTTOM_SWEEPS,
  OCTREE_RUNTIME_DIAL_BUILT_SMOOTHING_SWEEPS,
  type OctreeLosassoSolveTuning,
} from "./octree-runtime-dials";

export const OCTREE_LOSASSO_RESIDENT_SOLVE_ENVIRONMENT =
  "FLUID_OCTREE_LOSASSO_RESIDENT_SOLVE";
/** Combined-drain / fused sub-L0 envelope; the resident tier shares it. */
export const OCTREE_LOSASSO_RESIDENT_MAXIMUM_ROWS = 4_096;
/**
 * Lanes in the single resident workgroup. The whole solve is one dispatch of
 * one workgroup, so this is the only occupancy lever WGSL can express. The
 * first-frame capture at 256 lanes measured the dispatch latency-starved
 * (compute occupancy 0.27, ALU 0.30), so the width runs at the M1 Max's
 * 1,024-thread threadgroup ceiling. Value-neutral by construction: row loops
 * are `for (row = lane; row < n; row += LANES)` strides whose per-row
 * arithmetic is independent, and every cross-row fold is pinned to its own
 * association (the 128-lane compensated tree and the integer superaccumulator)
 * regardless of this width.
 */
export const OCTREE_LOSASSO_RESIDENT_LANES = 1_024;

/**
 * Measurement override for the workgroup width. A wider group hides gather
 * latency; a narrower one pays less per threadgroup barrier — the probe
 * decides, production keeps the authored default.
 */
export function octreeLosassoResidentLanes(
  environment?: Readonly<Record<string, string | undefined>>,
): 256 | 512 | 1024 {
  const resolved = environment
    ?? (typeof process !== "undefined" ? process.env : undefined);
  const value = resolved?.FLUID_LOSASSO_RESIDENT_LANES;
  if (value === undefined || value === "") return OCTREE_LOSASSO_RESIDENT_LANES as 1024;
  if (value === "256") return 256;
  if (value === "512") return 512;
  if (value === "1024") return 1024;
  throw new RangeError(`Resident Losasso lane count must be 256, 512 or 1024; received ${value}`);
}
const REDUCTION_LANES = 128;
/**
 * Seven storage bindings: level-0 CSR (control/offsets/incidence/faces), the
 * fused sub-L0 hierarchy arena, one consolidated f32 solver arena, and the
 * atomic solve control. The Metal tier grants ten; everything else is staged
 * into the arena by encode-time copies, the persistent executor's pattern.
 */
const STORAGE_BINDING_COUNT = 7;
/** 128-byte MGPCG control ABI plus the staged accepted-authority prefix. */
const CONTROL_BYTES = 256;
const AUTHORITY_STAGE_WORD = 32;
/**
 * Staged runtime dials: `[iterationCap, bottomSweeps, relativeToleranceBits,
 * reserved]`, each zero-means-compiled. They live past the authority prefix in
 * the same control buffer so the kernel needs no eighth storage binding — the
 * device grants ten and seven are already spoken for.
 */
const TUNING_STAGE_WORD = 40;
const TUNING_WORDS = 4;

/** Phase selected by {@link octreeLosassoResidentPhaseRepeatProbe}. */
export type OctreeLosassoResidentPhaseRepeatPhase = "l0-sweep" | "operator" | "bottom";

/**
 * Timing instrument for the one-dispatch solve, following the persistent
 * executor's precedent: pass-label isolation cannot split a single dispatch,
 * so `FLUID_LOSASSO_RESIDENT_PHASE_REPEAT=operator:4` re-runs a pure phase
 * three extra times per occurrence. Each probed phase is an idempotent gather
 * (its inputs are unchanged between repeats and every item writes only its
 * own row), so the solve is value-neutral and the wall delta divided by the
 * extra repeats is that phase's marginal cost.
 */
export function octreeLosassoResidentPhaseRepeatProbe(
  environment?: Readonly<Record<string, string | undefined>>,
): Readonly<{ phase: OctreeLosassoResidentPhaseRepeatPhase; repeats: number }> | undefined {
  const resolved = environment
    ?? (typeof process !== "undefined" ? process.env : undefined);
  const value = resolved?.FLUID_LOSASSO_RESIDENT_PHASE_REPEAT;
  if (!value) return undefined;
  const [phase, count] = value.split(":");
  if (phase !== "l0-sweep" && phase !== "operator" && phase !== "bottom") {
    throw new RangeError(`Unknown resident Losasso phase repeat probe ${value}`);
  }
  const repeats = count === undefined ? 2 : Number(count);
  if (!Number.isSafeInteger(repeats) || repeats < 1 || repeats > 64) {
    throw new RangeError("Resident Losasso phase repeat count must be an integer in [1,64]");
  }
  return Object.freeze({ phase, repeats });
}

/**
 * Default ON for the coarse-only ≤4K-row tier; `=0` restores the pipelined
 * executor. The symmetry-stage-audit lanes stay on the pipelined path because
 * only it owns the per-stage audit snapshot buffers.
 */
export function octreeLosassoResidentSolveEnabled(
  environment?: Readonly<Record<string, string | undefined>>,
): boolean {
  const resolved = environment
    ?? (typeof process !== "undefined" ? process.env : undefined);
  if (resolved?.FLUID_SYMMETRY_STAGE_AUDIT === "1") return false;
  if (resolved?.FLUID_OCTREE_PRESSURE_COLD_START === "1") return false;
  if (resolved?.FLUID_OCTREE_PRESSURE_TEMPORAL_PREDICTOR) return false;
  return resolved?.[OCTREE_LOSASSO_RESIDENT_SOLVE_ENVIRONMENT] !== "0";
}

export interface ResidentWGSLConfig {
  readonly rowCapacity: number;
  readonly maximumIterations: number;
  readonly relativeTolerance: number;
  readonly absoluteTolerance: number;
  readonly levelCount: number;
  readonly fused: NonNullable<OctreeLosassoVCycleHierarchySource["fusedSubL0"]>;
  readonly phaseRepeat?: Readonly<{
    phase: OctreeLosassoResidentPhaseRepeatPhase; repeats: number;
  }>;
  readonly lanes: 256 | 512 | 1024;
  /**
   * Run the bottom-level smoother entirely in workgroup memory. The phase
   * probe measured ~18 us of fixed cost per storage-fenced stage (a 300-row
   * sweep costs almost as much as a 1,152-row one), so the eight bottom
   * sweeps pay for their device fences, not their arithmetic. With the
   * two-level hierarchy, restrict writes the coarse vectors straight into
   * workgroup storage, the sweeps barrier on workgroupBarrier alone, and
   * prolong reads the result back — zero storage fences inside the phase.
   * Bit-identical: the staged values are bit-copies feeding the identical
   * expressions, and the L1 arena regions simply go unused.
   */
  readonly sharedBottom: boolean;
}

/** Shortest-round-trip f32 literal; parses back to the identical bit pattern. */
function f32Literal(value: number): string {
  const rounded = Math.fround(value);
  if (!Number.isFinite(rounded)) throw new RangeError("Resident MGPCG literal must be finite");
  const text = `${rounded}`;
  return /[.e]/.test(text) ? text : `${text}.0`;
}

/**
 * Word bases of the packed per-level vector arenas. Must match the layout
 * `WebGPUOctreeLosassoVCycle` derives from `fusedSubL0.levelRowCapacities`
 * (`Math.ceil(capacity * 4 / 256) * 64` words per level, finest first).
 */
function vectorBases(levelRowCapacities: readonly number[]): number[] {
  const bases: number[] = [];
  let base = 0;
  for (const capacity of levelRowCapacities) {
    bases.push(base);
    base += Math.ceil(capacity * 4 / 256) * 64;
  }
  return bases;
}

/**
 * Exported for the shader gates. The kernel is generated per hierarchy, so
 * there is no single module-level shader constant a parser test can read the
 * way the pipelined executor offers `octreePipelinedMGPCGShader`.
 */
export function residentLosassoMGPCGWGSL(config: ResidentWGSLConfig): string {
  const capacities = config.fused.levelRowCapacities;
  const bases = vectorBases(capacities);
  const arenaWords = bases[bases.length - 1]!
    + Math.ceil(capacities[capacities.length - 1]! * 4 / 256) * 64;
  const cap = config.rowCapacity;
  return /* wgsl */ `
${octreeCompensatedF32WGSL}
struct MergedScalars {
  gamma: CompensatedF32,
  delta: CompensatedF32,
  rr: CompensatedF32,
  bb: CompensatedF32,
}
struct Face { negativeRow:u32,positiveRow:u32,axis:u32,reserved:u32,
  area:f32,inverseDistance:f32,openFraction:f32,normalVelocity:f32 }

@group(0) @binding(0) var<storage, read> levelZeroControl: array<u32>;
@group(0) @binding(1) var<storage, read> levelZeroRowFaceOffsets: array<u32>;
@group(0) @binding(2) var<storage, read> levelZeroRowFaces: array<u32>;
@group(0) @binding(3) var<storage, read> levelZeroFaces: array<Face>;
@group(0) @binding(4) var<storage, read> hierarchy: array<u32>;
@group(0) @binding(5) var<storage, read_write> arena: array<f32>;
@group(0) @binding(6) var<storage, read_write> control: array<atomic<u32>>;

// Consolidated arena layout (words). Staged inputs are copied in by the
// encoder before the dispatch; the published pressure is copied back out.
const OUT_BASE = ${0 * cap}u;
const SEED_BASE = ${1 * cap}u;
const RHS_BASE = ${2 * cap}u;
const DIAG_BASE = ${3 * cap}u;
const P_BASE = ${4 * cap}u;
const R_BASE = ${5 * cap}u;
const Z_BASE = ${6 * cap}u;
const ZIMG_BASE = ${7 * cap}u;
const D_BASE = ${8 * cap}u;
const DIMG_BASE = ${9 * cap}u;
const V_RHS_BASE = ${10 * cap}u;
const V_XA_BASE = ${10 * cap + arenaWords}u;
const V_XB_BASE = ${10 * cap + 2 * arenaWords}u;
const V_RES_BASE = ${10 * cap + 3 * arenaWords}u;
const AUTHORITY_STAGE = ${AUTHORITY_STAGE_WORD}u;
// Encode-time staged runtime dials, written by the host into a tiny persistent
// buffer and copied in beside the authority. Zero means "use what was
// compiled", so an untouched session reproduces the built constants exactly.
const TUNING_STAGE = ${TUNING_STAGE_WORD}u;

const ERROR_INVALID_AUTHORITY = 1u;
const ERROR_INVALID_ROW = 2u;
const ERROR_NONFINITE = 4u;
const ERROR_NONPOSITIVE_PRECONDITIONER = 8u;
const ERROR_NONPOSITIVE_CURVATURE = 16u;
const ERROR_NONCONVERGENCE = 32u;
const INVALID = 0xffffffffu;
const LANES = ${config.lanes}u;
const REDUCTION_LANES = ${REDUCTION_LANES}u;
// Phase-repeat probe multipliers; 1 everywhere in production.
const REPEAT_L0 = ${config.phaseRepeat?.phase === "l0-sweep" ? config.phaseRepeat.repeats : 1}u;
const REPEAT_OP = ${config.phaseRepeat?.phase === "operator" ? config.phaseRepeat.repeats : 1}u;
const REPEAT_BOTTOM = ${config.phaseRepeat?.phase === "bottom" ? config.phaseRepeat.repeats : 1}u;
const CAP = ${config.rowCapacity}u;
const MAX_ITER_ENVELOPE = ${config.maximumIterations}u;
const BUILT_REL_TOL: f32 = ${f32Literal(config.relativeTolerance)};
const BUILT_BOTTOM_SWEEPS = ${OCTREE_RUNTIME_DIAL_BUILT_BOTTOM_SWEEPS}u;
const BUILT_SMOOTHING_SWEEPS = ${OCTREE_RUNTIME_DIAL_BUILT_SMOOTHING_SWEEPS}u;
const ABS_TOL: f32 = ${f32Literal(config.absoluteTolerance)};
const BB_FLOOR: f32 = ${f32Literal(1e-30)};
const DAMPING: f32 = ${f32Literal(2 / 3)};
const LEVEL_COUNT: u32 = ${config.levelCount}u;
const VECTOR_CAPACITIES: array<u32, ${config.levelCount}> = array<u32, ${config.levelCount}>(
  ${capacities.map((value) => `${value}u`).join(",")});
const VECTOR_BASES: array<u32, ${config.levelCount}> = array<u32, ${config.levelCount}>(
  ${bases.map((value) => `${value}u`).join(",")});
const VECTOR_ARENA_WORDS = ${arenaWords}u;
const TRANSITION_STRIDE: u32 = ${config.fused.transitionStrideWords}u;
const CONTROL_OFFSET: u32 = ${config.fused.controlOffsetWords}u;
const ROW_OFFSETS_OFFSET: u32 = ${config.fused.rowOffsetsOffsetWords}u;
const ROW_FACES_OFFSET: u32 = ${config.fused.rowFacesOffsetWords}u;
const FACES_OFFSET: u32 = ${config.fused.facesOffsetWords}u;
const PARENTS_OFFSET: u32 = ${config.fused.parentsOffsetWords}u;
const CHILD_OFFSETS_OFFSET: u32 = ${config.fused.childOffsetsOffsetWords}u;
const CHILD_LIST_OFFSET: u32 = ${config.fused.childListOffsetWords}u;
const FUSED_FACE_CAPACITY: u32 = ${config.fused.faceCapacity}u;

fn finite(value: f32) -> bool { return value == value && abs(value) <= 3.402823e38; }
// Encode-time staged copy of the seven-word accepted authority. The host
// asserts the source buffer holds at least seven words, so the pipelined
// path's arrayLength guards are constant-true here.
fn auth(word: u32) -> u32 { return atomicLoad(&control[AUTHORITY_STAGE + word]); }
fn tuned(word: u32) -> u32 { return atomicLoad(&control[TUNING_STAGE + word]); }
/**
 * Effective iteration ceiling. The dial may only lower the compiled envelope:
 * the control-word accounting and the fail-closed exhaustion gate were both
 * sized for MAX_ITER_ENVELOPE, so widening it here would let the loop outrun
 * its own bookkeeping.
 */
fn maxIterations() -> u32 {
  let requested = tuned(0u);
  return select(MAX_ITER_ENVELOPE, min(requested, MAX_ITER_ENVELOPE), requested != 0u);
}
/** Even sweeps only: the smoother ping-pongs, and prolong reads bank A. */
fn bottomSweeps() -> u32 {
  let requested = tuned(1u) & 0xfffffffeu;
  return select(BUILT_BOTTOM_SWEEPS, min(requested, 64u), requested >= 2u);
}
fn relativeTolerance() -> f32 {
  let bits = tuned(2u);
  return select(BUILT_REL_TOL, bitcast<f32>(bits), bits != 0u);
}
/**
 * Pre/post smoothing sweeps above the bottom level. Any count is legal here,
 * odd included: the smoother chooses its starting bank from the parity so the
 * pre-chain always ends in xA, which is where formResidual reads and where
 * prolongation accumulates.
 */
fn smoothingSweeps() -> u32 {
  let requested = tuned(3u);
  return select(BUILT_SMOOTHING_SWEEPS, min(requested, 16u), requested >= 1u);
}
fn rows() -> u32 { return min(auth(2u), CAP); }
fn failed() -> bool { return atomicLoad(&control[0]) != 0u; }
fn stopped() -> bool { return failed() || atomicLoad(&control[1]) != 0u; }
fn pairAt(word: u32) -> CompensatedF32 {
  return CompensatedF32(
    bitcast<f32>(atomicLoad(&control[word])),
    bitcast<f32>(atomicLoad(&control[word + 1u])),
  );
}
fn storePair(word: u32, value: CompensatedF32) {
  atomicStore(&control[word], bitcast<u32>(value.hi));
  atomicStore(&control[word + 1u], bitcast<u32>(value.lo));
}
fn reportAt(flag: u32, stage: u32, row: u32) {
  atomicOr(&control[0], flag);
  for (var retry = 0u; retry < 16u; retry += 1u) {
    let claim = atomicCompareExchangeWeak(&control[6], 0u, stage);
    if (claim.exchanged) {
      atomicStore(&control[7], row);
      return;
    }
    if (claim.old_value != 0u) { return; }
  }
}
// GPU-authored zeroed-dispatch accounting words matching what the pipelined
// executor's zeroDispatch loops would have counted for the same event.
fn accountZeroAllOuterDispatches() { atomicAdd(&control[5], 4u * maxIterations()); }
fn accountZeroRemainingAfterUpdate(iteration: u32) {
  let ceiling = maxIterations();
  let future = select(0u, ceiling - 1u - iteration, iteration + 1u < ceiling);
  atomicAdd(&control[5], 2u + 4u * future);
}
fn zeroMergedScalars() -> MergedScalars {
  let zero = CompensatedF32(0.0, 0.0);
  return MergedScalars(zero, zero, zero, zero);
}
fn mergeScalars(a: MergedScalars, b: MergedScalars) -> MergedScalars {
  return MergedScalars(
    mergeCompensatedF32(a.gamma, b.gamma),
    mergeCompensatedF32(a.delta, b.delta),
    mergeCompensatedF32(a.rr, b.rr),
    mergeCompensatedF32(a.bb, b.bb),
  );
}
// Storage-stage boundary. storageBarrier() is itself a control barrier, so
// one call orders both execution and storage memory; the workgroup-memory
// folds carry their own workgroupBarrier() calls where they need them.
fn sync() { storageBarrier(); }

var<workgroup> merged: array<MergedScalars, ${REDUCTION_LANES}>;
var<workgroup> wgFlag: u32;
// Dial values that bound a loop containing a barrier. WGSL requires uniform
// control flow around every barrier, and a storage-buffer read is not provably
// uniform, so the two counts are published once through workgroup memory and
// threaded down as parameters rather than re-read at each use.
var<workgroup> wgMaxIterations: u32;
var<workgroup> wgBottomSweeps: u32;
var<workgroup> wgSmoothingSweeps: u32;

fn publishUniformDials(lane: u32) {
  if (lane == 0u) {
    wgMaxIterations = maxIterations();
    wgBottomSweeps = bottomSweeps();
    wgSmoothingSweeps = smoothingSweeps();
  }
  workgroupBarrier();
}

fn uniformStopped(lane: u32) -> bool {
  if (lane == 0u) { wgFlag = select(0u, 1u, stopped()); }
  workgroupBarrier();
  return workgroupUniformLoad(&wgFlag) == 1u;
}

// ---- Transcription of validateAuthority / initializeControlAndDispatch ----
// (minus the indirect-record table, which the resident schedule does not have)
fn initializeControl() {
  atomicStore(&control[7], INVALID);
  if (auth(2u) == 0u || auth(2u) > CAP) {
    reportAt(ERROR_INVALID_AUTHORITY, 1u, INVALID);
    return;
  }
  if (auth(0u) != 0u
    || auth(1u) != INVALID
    || auth(3u) == 0u
    || auth(4u) == 0u
    || auth(5u) > 1u
    || auth(6u) != auth(4u)) {
    reportAt(ERROR_INVALID_AUTHORITY, 1u, INVALID);
  }
  atomicStore(&control[4], rows());
}

// ---- Workgroup-resident signed radix-256 exact reduction (one partial) ----
// Same digit decomposition, integer accumulation and single-rounding decode as
// webgpu-exact-f32-reduction; the limb totals are integer sums, so holding the
// one partial in workgroup atomics instead of a storage buffer cannot change
// the decoded f32.
var<workgroup> exactPartial: array<atomic<i32>, 144>;

fn clearExactPartial(lane: u32) {
  for (var word = lane; word < 144u; word += LANES) {
    atomicStore(&exactPartial[word], 0);
  }
  workgroupBarrier();
}
fn exactDeposit(scalar: u32, value: f32) {
  let bits = bitcast<u32>(value);
  let magnitude = bits & 0x7fffffffu;
  if (magnitude == 0u) { return; }
  let rawExponent = (magnitude >> 23u) & 0xffu;
  let fraction = magnitude & 0x7fffffu;
  let significand = select(fraction, 0x800000u | fraction, rawExponent != 0u);
  let shift = select(3u, rawExponent + 2u, rawExponent != 0u);
  let firstLimb = shift >> 3u;
  let shifted = significand << (shift & 7u);
  let sign = select(1, -1, (bits & 0x80000000u) != 0u);
  for (var digit = 0u; digit < 4u; digit += 1u) {
    let limb = firstLimb + digit;
    let byte = i32((shifted >> (digit * 8u)) & 0xffu);
    if (byte != 0 && limb < 36u) {
      atomicAdd(&exactPartial[scalar * 36u + limb], sign * byte);
    }
  }
}
fn exactFloorDiv256(value: i32) -> vec2i {
  var carry = value / 256;
  var digit = value - carry * 256;
  if (digit < 0) { digit += 256; carry -= 1; }
  return vec2i(carry, digit);
}
fn exactDecodeLimbs(totals: array<i32, 36>) -> f32 {
  var limbs = totals;
  for (var limb = 0u; limb + 1u < 36u; limb += 1u) {
    let normalized = exactFloorDiv256(limbs[limb]);
    limbs[limb] = normalized.y;
    limbs[limb + 1u] += normalized.x;
  }
  let negative = limbs[35u] < 0;
  if (negative) {
    for (var limb = 0u; limb < 36u; limb += 1u) { limbs[limb] = -limbs[limb]; }
    for (var limb = 0u; limb + 1u < 36u; limb += 1u) {
      let normalized = exactFloorDiv256(limbs[limb]);
      limbs[limb] = normalized.y;
      limbs[limb + 1u] += normalized.x;
    }
  }
  var magnitude = 0.0;
  for (var limb = 0u; limb < 36u; limb += 1u) {
    magnitude += ldexp(f32(limbs[limb]), -152 + i32(limb * 8u));
  }
  return select(magnitude, -magnitude, negative);
}
// Every lane must reach this call (it carries a barrier). Work is selected by
// partialCount, never by branching around the call; a count of zero folds
// nothing and returns exactly +0.0. Only lane 0's value is meaningful.
fn exactScalarValue(scalar: u32, lane: u32, partialCount: u32) -> f32 {
  var value = 0.0;
  if (lane == 0u) {
    var totals: array<i32, 36>;
    if (partialCount > 0u) {
      for (var limb = 0u; limb < 36u; limb += 1u) {
        totals[limb] = atomicLoad(&exactPartial[scalar * 36u + limb]);
      }
    }
    value = exactDecodeLimbs(totals);
  }
  workgroupBarrier();
  return value;
}

// ---- Transcription of initializeState (warm seed; diagonal stride 1, bank 0) ----
fn initializeState(lane: u32) {
  for (var row = lane; row < rows(); row += LANES) {
    if (failed()) { continue; }
    let seed = arena[SEED_BASE + row];
    let d = arena[DIAG_BASE + row];
    if (!finite(d) || d <= 0.0 || !finite(arena[RHS_BASE + row]) || !finite(seed)) {
      reportAt(ERROR_INVALID_ROW, 2u, row);
      continue;
    }
    arena[P_BASE + row] = seed;
    arena[R_BASE + row] = 0.0;
  }
}

fn formInitialResidual(lane: u32) {
  for (var row = lane; row < rows(); row += LANES) {
    if (failed()) { continue; }
    let value = -arena[RHS_BASE + row] - arena[DIMG_BASE + row];
    if (!finite(value)) { reportAt(ERROR_NONFINITE, 3u, row); }
    else { arena[R_BASE + row] = value; }
  }
}

// ---- Transcription of applyLosassoOperator (signed radix-256 row fold) ----
fn opAddExact(limbs: ptr<function, array<i32,36>>, value: f32) {
  let bits = bitcast<u32>(value); let magnitude = bits & 0x7fffffffu;
  if (magnitude == 0u) { return; }
  let rawExponent = (magnitude >> 23u) & 0xffu; let fraction = magnitude & 0x7fffffu;
  let significand = select(fraction, 0x800000u | fraction, rawExponent != 0u);
  let shift = select(3u, rawExponent + 2u, rawExponent != 0u);
  let firstLimb = shift >> 3u; let shifted = significand << (shift & 7u);
  let sign = select(1, -1, (bits & 0x80000000u) != 0u);
  for (var digit = 0u; digit < 4u; digit += 1u) {
    let limb = firstLimb + digit; let byte = i32((shifted >> (digit * 8u)) & 0xffu);
    if (byte != 0 && limb < 36u) { (*limbs)[limb] += sign * byte; }
  }
}
fn opFloorDiv256(value: i32) -> vec2i { let carry = value >> 8; return vec2i(carry, value - carry * 256); }
fn opExactValue(source: ptr<function, array<i32,36>>) -> f32 {
  for (var limb = 0u; limb + 1u < 36u; limb += 1u) {
    let normalized = opFloorDiv256((*source)[limb]);
    (*source)[limb] = normalized.y; (*source)[limb + 1u] += normalized.x;
  }
  let negative = (*source)[35] < 0;
  if (negative) {
    for (var limb = 0u; limb < 36u; limb += 1u) { (*source)[limb] = -(*source)[limb]; }
    for (var limb = 0u; limb + 1u < 36u; limb += 1u) {
      let normalized = opFloorDiv256((*source)[limb]);
      (*source)[limb] = normalized.y; (*source)[limb + 1u] += normalized.x;
    }
  }
  var magnitude = 0.0;
  for (var limb = 0u; limb < 36u; limb += 1u) {
    magnitude += ldexp(f32((*source)[limb]), -152 + i32(limb * 8u));
  }
  return select(magnitude, -magnitude, negative);
}
fn applyOperator(lane: u32, inputBase: u32, outputBase: u32) {
  let bound = levelZeroControl[1];
  for (var row = lane; row < bound; row += LANES) {
    if (levelZeroControl[3] != 1u || stopped()) { continue; }
    let centre = arena[inputBase + row];
    if (row + 1u >= arrayLength(&levelZeroRowFaceOffsets)) {
      arena[outputBase + row] = 3.402823e38;
      continue;
    }
    let begin = levelZeroRowFaceOffsets[row];
    let end = levelZeroRowFaceOffsets[row + 1u];
    if (begin > end || end > arrayLength(&levelZeroRowFaces) || end - begin > 24u) {
      arena[outputBase + row] = 3.402823e38;
      continue;
    }
    var limbs: array<i32,36>; var finiteTerms = true;
    for (var cursor = begin; cursor < end; cursor += 1u) {
      let face = levelZeroFaces[levelZeroRowFaces[cursor]];
      let coefficient = (face.openFraction * face.area) * face.inverseDistance;
      var difference = centre;
      if (face.negativeRow == row) {
        if (face.positiveRow != INVALID) {
          difference = centre - arena[inputBase + face.positiveRow];
        }
      } else {
        difference = centre - arena[inputBase + face.negativeRow];
      }
      let term = coefficient * difference;
      if (term != term || abs(term) >= 3.402823e38) { finiteTerms = false; }
      else { opAddExact(&limbs, term); }
    }
    arena[outputBase + row] = select(3.402823e38, opExactValue(&limbs), finiteTerms);
  }
}

// ---- Transcription of the Losasso V-cycle level-0 kernels ----
const L0_SRC_XA = 0u;
const L0_SRC_XB = 1u;
fn vStopped() -> bool { return stopped() || levelZeroControl[3] != 1u; }
fn l0Value(source: u32, row: u32) -> f32 {
  if (source == L0_SRC_XB) { return arena[V_XB_BASE +row]; }
  return arena[V_XA_BASE +row];
}
fn l0Image(row: u32, source: u32) -> vec2f {
  if (row + 1u >= arrayLength(&levelZeroRowFaceOffsets)) { return vec2f(3.402823e38, -1.); }
  let begin = levelZeroRowFaceOffsets[row]; let end = levelZeroRowFaceOffsets[row + 1u];
  if (begin > end || end > arrayLength(&levelZeroRowFaces)) { return vec2f(3.402823e38, -1.); }
  let centre = l0Value(source, row); var imageSum = CompensatedF32(0., 0.);
  var diagonalSum = CompensatedF32(0., 0.); var valid = true;
  for (var cursor = begin; cursor < end; cursor += 1u) {
    let faceId = levelZeroRowFaces[cursor];
    if (faceId >= arrayLength(&levelZeroFaces)) { valid = false; continue; }
    let face = levelZeroFaces[faceId];
    let coefficient = (face.openFraction * face.area) * face.inverseDistance;
    var difference = centre;
    if (face.negativeRow == row) {
      if (face.positiveRow != INVALID) {
        if (face.positiveRow >= VECTOR_CAPACITIES[0]) { valid = false; continue; }
        difference -= l0Value(source, face.positiveRow);
      }
    } else {
      if (face.negativeRow >= VECTOR_CAPACITIES[0]) { valid = false; continue; }
      difference -= l0Value(source, face.negativeRow);
    }
    let term = coefficient * difference;
    if (!finite(coefficient) || coefficient < 0. || !finite(term)) { valid = false; continue; }
    diagonalSum = addCompensatedF32(diagonalSum, coefficient);
    imageSum = addCompensatedF32(imageSum, term);
  }
  let value = compensatedValue(imageSum); let diag = compensatedValue(diagonalSum);
  return select(vec2f(3.402823e38, -1.), vec2f(value, diag),
    valid && finite(value) && finite(diag) && diag > 0.);
}
fn l0Jacobi(lane: u32, source: u32, publish: bool) {
  let bound = levelZeroControl[1];
  for (var row = lane; row < bound; row += LANES) {
    if (vStopped()) { continue; }
    let pair = l0Image(row, source);
    let value = select(0., l0Value(source, row) + DAMPING * (arena[V_RHS_BASE +row] - pair.x) / pair.y, pair.y > 0.);
    if (source == L0_SRC_XA) { arena[V_XB_BASE +row] = value; } else { arena[V_XA_BASE +row] = value; }
    // The final post-sweep's output has exactly one consumer, the copy into
    // the preconditioned vector; publish it in the same own-row loop instead.
    // Which BANK it landed in is irrelevant to that consumer, so this does not
    // test the source the way it did when the count was fixed at two.
    if (publish) { arena[Z_BASE + row] = value; }
  }
}

// ---- Transcription of the fused sub-L0 walk (single 256-lane workgroup) ----
fn fusedTransitionBase(level: u32) -> u32 { return (level - 1u) * TRANSITION_STRIDE; }
fn fusedLevelControl(level: u32, word: u32) -> u32 {
  return hierarchy[fusedTransitionBase(level) + CONTROL_OFFSET + word];
}
fn fusedVectorIndex(level: u32, row: u32) -> u32 { return VECTOR_BASES[level] + row; }
fn fusedCorrectionValue(fromB: bool, level: u32, row: u32) -> f32 {
  if (fromB) { return arena[V_XB_BASE +fusedVectorIndex(level, row)]; }
  return arena[V_XA_BASE +fusedVectorIndex(level, row)];
}
fn fusedLoadFace(level: u32, faceId: u32) -> Face {
  let base = fusedTransitionBase(level) + FACES_OFFSET + 8u * faceId;
  return Face(hierarchy[base], hierarchy[base + 1u], hierarchy[base + 2u], hierarchy[base + 3u],
    bitcast<f32>(hierarchy[base + 4u]), bitcast<f32>(hierarchy[base + 5u]),
    bitcast<f32>(hierarchy[base + 6u]), bitcast<f32>(hierarchy[base + 7u]));
}
fn fusedImage(level: u32, row: u32, fromB: bool) -> vec2f {
  let base = fusedTransitionBase(level);
  let begin = hierarchy[base + ROW_OFFSETS_OFFSET + row];
  let end = hierarchy[base + ROW_OFFSETS_OFFSET + row + 1u];
  let incidenceCapacity = 2u * FUSED_FACE_CAPACITY;
  if (begin > end || end > incidenceCapacity) { return vec2f(3.402823e38, -1.); }
  let centre = fusedCorrectionValue(fromB, level, row);
  var imageSum = CompensatedF32(0., 0.); var diagonalSum = CompensatedF32(0., 0.); var valid = true;
  for (var cursor = begin; cursor < end; cursor += 1u) {
    let faceId = hierarchy[base + ROW_FACES_OFFSET + cursor];
    if (faceId >= FUSED_FACE_CAPACITY) { valid = false; continue; }
    let face = fusedLoadFace(level, faceId);
    let coefficient = (face.openFraction * face.area) * face.inverseDistance;
    var difference = centre;
    if (face.negativeRow == row) {
      if (face.positiveRow != INVALID) {
        if (face.positiveRow >= VECTOR_CAPACITIES[level]) { valid = false; continue; }
        difference -= fusedCorrectionValue(fromB, level, face.positiveRow);
      }
    } else {
      if (face.negativeRow >= VECTOR_CAPACITIES[level]) { valid = false; continue; }
      difference -= fusedCorrectionValue(fromB, level, face.negativeRow);
    }
    let term = coefficient * difference;
    if (!finite(coefficient) || coefficient < 0. || !finite(term)) { valid = false; continue; }
    diagonalSum = addCompensatedF32(diagonalSum, coefficient);
    imageSum = addCompensatedF32(imageSum, term);
  }
  let value = compensatedValue(imageSum); let diag = compensatedValue(diagonalSum);
  return select(vec2f(3.402823e38, -1.), vec2f(value, diag),
    valid && finite(value) && finite(diag) && diag > 0.);
}
fn fusedRestrictInto(coarseLevel: u32, lane: u32, enabled: bool) {
  let base = fusedTransitionBase(coarseLevel);
  let rowCount = select(0u, fusedLevelControl(coarseLevel, 1u), enabled);
  for (var row = lane; row < rowCount; row += LANES) {
    let begin = hierarchy[base + CHILD_OFFSETS_OFFSET + row];
    let end = hierarchy[base + CHILD_OFFSETS_OFFSET + row + 1u];
    var sum = CompensatedF32(0., 0.);
    for (var cursor = begin; cursor < end; cursor += 1u) {
      let child = hierarchy[base + CHILD_LIST_OFFSET + cursor];
      sum = addCompensatedF32(sum, arena[V_RES_BASE +fusedVectorIndex(coarseLevel - 1u, child)]);
    }
    arena[V_RHS_BASE +fusedVectorIndex(coarseLevel, row)] = compensatedValue(sum);
    // Both banks, not just xA: an odd sweep count starts the smoother from xB
    // so the chain still ends in xA, and that first read must see a zero seed.
    // Value-neutral at even counts, where the first sweep overwrites xB anyway.
    arena[V_XA_BASE +fusedVectorIndex(coarseLevel, row)] = 0.;
    arena[V_XB_BASE +fusedVectorIndex(coarseLevel, row)] = 0.;
  }
  sync();
}
fn fusedRelaxLevel(level: u32, fromB: bool, lane: u32, enabled: bool) {
  let rowCount = select(0u, fusedLevelControl(level, 1u), enabled);
  for (var row = lane; row < rowCount; row += LANES) {
    let pair = fusedImage(level, row, fromB);
    let value = select(0., fusedCorrectionValue(fromB, level, row) + DAMPING
      * (arena[V_RHS_BASE +fusedVectorIndex(level, row)] - pair.x) / pair.y, pair.y > 0.);
    if (fromB) { arena[V_XA_BASE +fusedVectorIndex(level, row)] = value; }
    else { arena[V_XB_BASE +fusedVectorIndex(level, row)] = value; }
  }
  sync();
}
fn fusedFormResidual(level: u32, lane: u32, enabled: bool) {
  let rowCount = select(0u, fusedLevelControl(level, 1u), enabled);
  for (var row = lane; row < rowCount; row += LANES) {
    arena[V_RES_BASE +fusedVectorIndex(level, row)] =
      arena[V_RHS_BASE +fusedVectorIndex(level, row)] - fusedImage(level, row, false).x;
  }
  sync();
}
/**
 * Accumulate the coarse correction onto the fine level's smoothed iterate.
 *
 * The fine target is always xA — pre-smoothing is arranged to end there — but
 * the coarse SOURCE is wherever that level's post-smoothing finished, which is
 * xB after an odd sweep count. coarseFromB carries that.
 */
fn fusedProlongInto(fineLevel: u32, lane: u32, enabled: bool, coarseFromB: bool) {
  let coarseLevel = fineLevel + 1u; let base = fusedTransitionBase(coarseLevel);
  let coarseRows = fusedLevelControl(coarseLevel, 1u);
  let fineRows = select(0u, hierarchy[base + CHILD_OFFSETS_OFFSET + coarseRows], enabled);
  for (var row = lane; row < fineRows; row += LANES) {
    let parent = hierarchy[base + PARENTS_OFFSET + row];
    if (parent < coarseRows) {
      arena[V_XA_BASE +fusedVectorIndex(fineLevel, row)] +=
        fusedCorrectionValue(coarseFromB, coarseLevel, parent);
    }
  }
  sync();
}
/**
 * sweeps relaxations at level, arranged to finish in xA.
 *
 * Starting bank is chosen from the parity, which is what makes an odd count
 * expressible at all: the sweeps ping-pong, and everything downstream —
 * residual formation, restriction, prolongation — reads xA.
 */
fn fusedSmoothToXA(level: u32, lane: u32, enabled: bool, sweeps: u32) {
  var fromB = (sweeps & 1u) == 1u;
  for (var sweep = 0u; sweep < sweeps; sweep += 1u) {
    fusedRelaxLevel(level, fromB, lane, enabled);
    fromB = !fromB;
  }
}
${config.sharedBottom ? /* wgsl */ `
// ---- Workgroup-resident bottom smoother (two-level specialization) ----
// Identical arithmetic to fusedRelaxLevel(1u, ...) with the three level-1
// vectors held in workgroup storage for the whole phase, so the eight sweeps
// pay workgroup fences instead of device fences.
const L1_CAP = ${config.fused.levelRowCapacities[1]}u;
var<workgroup> sharedBottomVectors: array<f32, ${3 * config.fused.levelRowCapacities[1]!}>;
const SH_RHS = 0u;
const SH_XA = L1_CAP;
const SH_XB = 2u * L1_CAP;
fn sharedBottomValue(fromB: bool, row: u32) -> f32 {
  if (fromB) { return sharedBottomVectors[SH_XB + row]; }
  return sharedBottomVectors[SH_XA + row];
}
fn sharedBottomImage(row: u32, fromB: bool) -> vec2f {
  let base = fusedTransitionBase(1u);
  let begin = hierarchy[base + ROW_OFFSETS_OFFSET + row];
  let end = hierarchy[base + ROW_OFFSETS_OFFSET + row + 1u];
  let incidenceCapacity = 2u * FUSED_FACE_CAPACITY;
  if (begin > end || end > incidenceCapacity) { return vec2f(3.402823e38, -1.); }
  let centre = sharedBottomValue(fromB, row);
  var imageSum = CompensatedF32(0., 0.); var diagonalSum = CompensatedF32(0., 0.); var valid = true;
  for (var cursor = begin; cursor < end; cursor += 1u) {
    let faceId = hierarchy[base + ROW_FACES_OFFSET + cursor];
    if (faceId >= FUSED_FACE_CAPACITY) { valid = false; continue; }
    let face = fusedLoadFace(1u, faceId);
    let coefficient = (face.openFraction * face.area) * face.inverseDistance;
    var difference = centre;
    if (face.negativeRow == row) {
      if (face.positiveRow != INVALID) {
        if (face.positiveRow >= VECTOR_CAPACITIES[1]) { valid = false; continue; }
        difference -= sharedBottomValue(fromB, face.positiveRow);
      }
    } else {
      if (face.negativeRow >= VECTOR_CAPACITIES[1]) { valid = false; continue; }
      difference -= sharedBottomValue(fromB, face.negativeRow);
    }
    let term = coefficient * difference;
    if (!finite(coefficient) || coefficient < 0. || !finite(term)) { valid = false; continue; }
    diagonalSum = addCompensatedF32(diagonalSum, coefficient);
    imageSum = addCompensatedF32(imageSum, term);
  }
  let value = compensatedValue(imageSum); let diag = compensatedValue(diagonalSum);
  return select(vec2f(3.402823e38, -1.), vec2f(value, diag),
    valid && finite(value) && finite(diag) && diag > 0.);
}
fn sharedRestrictIntoBottom(lane: u32) {
  let base = fusedTransitionBase(1u);
  let rowCount = fusedLevelControl(1u, 1u);
  for (var row = lane; row < rowCount; row += LANES) {
    let begin = hierarchy[base + CHILD_OFFSETS_OFFSET + row];
    let end = hierarchy[base + CHILD_OFFSETS_OFFSET + row + 1u];
    var sum = CompensatedF32(0., 0.);
    for (var cursor = begin; cursor < end; cursor += 1u) {
      let child = hierarchy[base + CHILD_LIST_OFFSET + cursor];
      sum = addCompensatedF32(sum, arena[V_RES_BASE + child]);
    }
    sharedBottomVectors[SH_RHS + row] = compensatedValue(sum);
    sharedBottomVectors[SH_XA + row] = 0.;
  }
  workgroupBarrier();
}
fn sharedBottomRelax(fromB: bool, lane: u32) {
  let rowCount = fusedLevelControl(1u, 1u);
  for (var row = lane; row < rowCount; row += LANES) {
    let pair = sharedBottomImage(row, fromB);
    let value = select(0., sharedBottomValue(fromB, row) + DAMPING
      * (sharedBottomVectors[SH_RHS + row] - pair.x) / pair.y, pair.y > 0.);
    if (fromB) { sharedBottomVectors[SH_XA + row] = value; }
    else { sharedBottomVectors[SH_XB + row] = value; }
  }
  workgroupBarrier();
}
fn sharedProlongToLevelZero(lane: u32) {
  let base = fusedTransitionBase(1u);
  let coarseRows = fusedLevelControl(1u, 1u);
  let fineRows = hierarchy[base + CHILD_OFFSETS_OFFSET + coarseRows];
  for (var row = lane; row < fineRows; row += LANES) {
    let parent = hierarchy[base + PARENTS_OFFSET + row];
    if (parent < coarseRows) {
      arena[V_XA_BASE + row] += sharedBottomVectors[SH_XA + parent];
    }
  }
  sync();
}
fn fusedSubL0Walk(lane: u32, sweeps: u32, smoothing: u32) {
  if (lane == 0u) {
    var enabled = !stopped();
    enabled = enabled && fusedLevelControl(1u, 3u) == 1u
      && fusedLevelControl(1u, 1u) <= VECTOR_CAPACITIES[1];
    wgFlag = select(0u, 1u, enabled);
  }
  workgroupBarrier();
  if (workgroupUniformLoad(&wgFlag) != 1u) { return; }
  // Two-level hierarchy: level 1 IS the bottom, so the smoothing dial has no
  // intermediate level to act on here and only the bottom count applies.
  sharedRestrictIntoBottom(lane);
  for (var sweep = 0u; sweep < sweeps; sweep += 1u) {
    if ((sweep & 1u) == 0u) {
      for (var probe = 0u; probe < REPEAT_BOTTOM; probe += 1u) { sharedBottomRelax(false, lane); }
    } else {
      for (var probe = 0u; probe < REPEAT_BOTTOM; probe += 1u) { sharedBottomRelax(true, lane); }
    }
  }
  sharedProlongToLevelZero(lane);
}` : /* wgsl */ `
fn fusedSubL0Walk(lane: u32, sweeps: u32, smoothing: u32) {
  if (lane == 0u) {
    var enabled = !stopped();
    for (var level = 1u; level < LEVEL_COUNT; level += 1u) {
      enabled = enabled && fusedLevelControl(level, 3u) == 1u
        && fusedLevelControl(level, 1u) <= VECTOR_CAPACITIES[level];
    }
    wgFlag = select(0u, 1u, enabled);
  }
  workgroupBarrier();
  if (workgroupUniformLoad(&wgFlag) != 1u) { return; }
  fusedRestrictInto(1u, lane, true);
  for (var level = 1u; level + 1u < LEVEL_COUNT; level += 1u) {
    fusedSmoothToXA(level, lane, true, smoothing);
    fusedFormResidual(level, lane, true); fusedRestrictInto(level + 1u, lane, true);
  }
  let bottom = LEVEL_COUNT - 1u;
  for (var sweep = 0u; sweep < sweeps; sweep += 1u) {
    if ((sweep & 1u) == 0u) {
      for (var probe = 0u; probe < REPEAT_BOTTOM; probe += 1u) { fusedRelaxLevel(bottom, false, lane, true); }
    } else {
      for (var probe = 0u; probe < REPEAT_BOTTOM; probe += 1u) { fusedRelaxLevel(bottom, true, lane, true); }
    }
  }
  // The bottom count is even, so it ends in xA. Above it, post-smoothing runs
  // from the prolonged xA and finishes in xB whenever the count is odd, which
  // is exactly what the next prolongation down has to be told.
  var coarseFromB = false;
  for (var level = bottom - 1u; level >= 1u; level -= 1u) {
    fusedProlongInto(level, lane, true, coarseFromB);
    var fromB = false;
    for (var sweep = 0u; sweep < smoothing; sweep += 1u) {
      fusedRelaxLevel(level, fromB, lane, true);
      fromB = !fromB;
    }
    coarseFromB = fromB;
  }
  fusedProlongInto(0u, lane, true, coarseFromB);
}`}

// Prelude of WebGPUOctreeLosassoVCycle.encodeCorrection's fused branch:
// clear xA0 and stage the CG residual as the L0 right-hand side. Both writes
// are own-row, so they share one loop and one stage boundary.
fn stageVCycleInputs(lane: u32) {
  let bound = levelZeroControl[1];
  for (var row = lane; row < bound; row += LANES) {
    if (vStopped()) { continue; }
    arena[V_XA_BASE + row] = 0.;
    // Both banks: an odd smoothing count starts the pre-chain from xB so it
    // still ends in xA. Value-neutral at even counts.
    arena[V_XB_BASE + row] = 0.;
    arena[V_RHS_BASE + row] = arena[R_BASE + row];
  }
}
// Per-iteration fusion of advancePCGState with that prelude. Every write and
// the staged read of R touch only the thread's own row, so the advanced
// residual is read by the same thread that wrote it and no boundary is
// needed between the two halves.
fn advanceAndStageVCycle(lane: u32) {
  let l0Rows = levelZeroControl[1];
  let bound = max(rows(), l0Rows);
  for (var row = lane; row < bound; row += LANES) {
    if (row < rows() && !stopped()) {
      let alpha = compensatedValue(pairAt(16u));
      let nextPressure = arena[P_BASE + row] + alpha * arena[D_BASE + row];
      let nextResidual = arena[R_BASE + row] - alpha * arena[DIMG_BASE + row];
      if (!finite(nextPressure) || !finite(nextResidual)) {
        reportAt(ERROR_NONFINITE, 7u, row);
      } else {
        arena[P_BASE + row] = nextPressure;
        arena[R_BASE + row] = nextResidual;
      }
    }
    if (row < l0Rows && !vStopped()) {
      arena[V_XA_BASE + row] = 0.;
      arena[V_RHS_BASE + row] = arena[R_BASE + row];
    }
  }
}
// Body of the fused-branch correction: pre-smoothing, L0 residual, the fused
// sub-L0 walk, post-smoothing, and the copy into the preconditioned vector.
// Callers stage the inputs (and a boundary) first.
//
// The smoothing count is the pre AND post count; they must match or M stops being
// symmetric and CG loses the property its recurrence is built on. Odd counts
// work because the pre-chain picks its starting bank from the parity: it
// always lands in xA, which is what the residual reads and what the walk's
// final prolongation accumulates into.
fn vcycleCorrection(lane: u32, sweeps: u32, smoothing: u32) {
  let bound = levelZeroControl[1];
  var source = select(L0_SRC_XA, L0_SRC_XB, (smoothing & 1u) == 1u);
  for (var sweep = 0u; sweep < smoothing; sweep += 1u) {
    for (var probe = 0u; probe < REPEAT_L0; probe += 1u) { l0Jacobi(lane, source, false); sync(); }
    source = 1u - source;
  }
  for (var row = lane; row < bound; row += LANES) {
    if (vStopped()) { continue; }
    arena[V_RES_BASE +row] = arena[V_RHS_BASE +row] - l0Image(row, L0_SRC_XA).x;
  }
  sync();
  fusedSubL0Walk(lane, sweeps, smoothing);
  var post = L0_SRC_XA;
  for (var sweep = 0u; sweep < smoothing; sweep += 1u) {
    let last = sweep + 1u == smoothing;
    for (var probe = 0u; probe < REPEAT_L0; probe += 1u) { l0Jacobi(lane, post, last); sync(); }
    post = 1u - post;
  }
}

// ---- Transcription of finishMergedTotal (indirect zeroing -> accounting) ----
fn finishMergedTotal(total: MergedScalars) {
  let initial = atomicLoad(&control[3]) == 0u;
  if (failed()) {
    if (initial) { accountZeroAllOuterDispatches(); }
    else { accountZeroRemainingAfterUpdate(atomicLoad(&control[2])); }
    return;
  }
  if (atomicLoad(&control[1]) != 0u) { return; }
  let gamma = compensatedValue(total.gamma);
  let delta = compensatedValue(total.delta);
  let rr = compensatedValue(total.rr);
  if (!finite(gamma) || !finite(delta) || !finite(rr) || rr < 0.0) {
    reportAt(ERROR_NONFINITE, 5u, INVALID);
    if (initial) { accountZeroAllOuterDispatches(); }
    else { accountZeroRemainingAfterUpdate(atomicLoad(&control[2])); }
    return;
  }
  if (initial) {
    let bb = compensatedValue(total.bb);
    if (!finite(bb) || bb < 0.0) {
      reportAt(ERROR_NONFINITE, 5u, INVALID);
      accountZeroAllOuterDispatches();
      return;
    }
    storePair(8u, total.bb);
    storePair(12u, total.gamma);
    storePair(14u, total.delta);
  }
  storePair(10u, total.rr);
  atomicAdd(&control[3], 1u);
  let bb = max(compensatedValue(pairAt(8u)), BB_FLOOR);
  let relative = relativeTolerance();
  let threshold = max(ABS_TOL * ABS_TOL, relative * relative * bb);
  if (!initial) { atomicAdd(&control[2], 1u); }
  if (rr <= threshold) {
    atomicStore(&control[1], 1u);
    if (initial) { accountZeroAllOuterDispatches(); }
    else { accountZeroRemainingAfterUpdate(atomicLoad(&control[2]) - 1u); }
    return;
  }
  if (!(gamma > 0.0)) {
    reportAt(ERROR_NONPOSITIVE_PRECONDITIONER, 5u, INVALID);
    if (initial) { accountZeroAllOuterDispatches(); }
    else { accountZeroRemainingAfterUpdate(atomicLoad(&control[2]) - 1u); }
    return;
  }
  var alpha = 0.0;
  if (initial) {
    if (!(delta > 0.0)) {
      reportAt(ERROR_NONPOSITIVE_CURVATURE, 5u, INVALID);
      accountZeroAllOuterDispatches();
      return;
    }
    alpha = gamma / delta;
    storePair(18u, CompensatedF32(0.0, 0.0));
  } else {
    let previousGamma = compensatedValue(pairAt(12u));
    let beta = gamma / previousGamma;
    if (!(previousGamma > 0.0) || !finite(beta) || beta < 0.0) {
      reportAt(ERROR_NONPOSITIVE_PRECONDITIONER, 5u, INVALID);
      accountZeroRemainingAfterUpdate(atomicLoad(&control[2]) - 1u);
      return;
    }
    storePair(18u, CompensatedF32(beta, 0.0));
  }
  if (initial && (!finite(alpha) || !(alpha > 0.0))) {
    reportAt(ERROR_NONFINITE, 5u, INVALID);
    accountZeroAllOuterDispatches();
    return;
  }
  storePair(12u, total.gamma);
  storePair(14u, total.delta);
  storePair(16u, CompensatedF32(alpha, 0.0));
  let ceiling = maxIterations();
  if (!initial && atomicLoad(&control[2]) >= ceiling) {
    reportAt(ERROR_NONCONVERGENCE, 6u, INVALID);
    accountZeroRemainingAfterUpdate(ceiling - 1u);
  }
}

// Initial reduction: exact signed radix-256 deposit and cooperative fold,
// matching reduceMergedPartials + finishMergedReduction. Depositing into a
// single partial is bit-identical to the pipelined multi-partial layout: the
// limb totals are integer sums, invariant under any partition, and the decode
// is the same single carry propagation and f32 rounding.
fn initialExactReduction(lane: u32) {
  clearExactPartial(lane);
  let initial = atomicLoad(&control[3]) == 0u;
  for (var row = lane; row < rows(); row += LANES) {
    if (stopped()) { continue; }
    let r = arena[R_BASE + row];
    let u = arena[Z_BASE + row];
    let w = arena[ZIMG_BASE + row];
    let b = -arena[RHS_BASE + row];
    if (!finite(r) || !finite(u) || (initial && !finite(w))) {
      reportAt(ERROR_NONFINITE, 4u, row);
    } else {
      exactDeposit(0u, r * u);
      if (initial) { exactDeposit(1u, u * w); }
      exactDeposit(2u, r * r);
      if (initial) { exactDeposit(3u, b * b); }
    }
  }
  // The deposits above land in workgroup atomics; the fold below reads them.
  workgroupBarrier();
  let retired = failed() || atomicLoad(&control[1]) != 0u;
  let live = select(1u, 0u, retired);
  let initialOnly = select(0u, live, atomicLoad(&control[3]) == 0u);
  let total = MergedScalars(
    CompensatedF32(exactScalarValue(0u, lane, live), 0.0),
    CompensatedF32(exactScalarValue(1u, lane, initialOnly), 0.0),
    CompensatedF32(exactScalarValue(2u, lane, live), 0.0),
    CompensatedF32(exactScalarValue(3u, lane, initialOnly), 0.0),
  );
  if (lane == 0u) { finishMergedTotal(total); }
}

// Per-iteration drain: 128-lane compensated accumulation and fixed merge
// tree, lane-for-lane identical to reduceAndFinishMerged. The upper 128 lanes
// hold zeros and only pass through the barriers.
fn mergedReduceAndFinish(lane: u32) {
  var local = zeroMergedScalars();
  let initial = atomicLoad(&control[3]) == 0u;
  if (lane < REDUCTION_LANES) {
    for (var row = lane; row < rows(); row += REDUCTION_LANES) {
      if (row < rows() && !stopped()) {
        let r = arena[R_BASE + row];
        let u = arena[Z_BASE + row];
        let w = arena[ZIMG_BASE + row];
        let b = -arena[RHS_BASE + row];
        if (!finite(r) || !finite(u) || (initial && !finite(w))) {
          reportAt(ERROR_NONFINITE, 4u, row);
        } else {
          local.gamma = addCompensatedF32(local.gamma, r * u);
          if (initial) { local.delta = addCompensatedF32(local.delta, u * w); }
          local.rr = addCompensatedF32(local.rr, r * r);
          if (initial) { local.bb = addCompensatedF32(local.bb, b * b); }
        }
      }
    }
    merged[lane] = local;
  }
  for (var width = REDUCTION_LANES / 2u; width > 0u; width >>= 1u) {
    workgroupBarrier();
    if (lane < width) {
      merged[lane] = mergeScalars(merged[lane], merged[lane + width]);
    }
  }
  workgroupBarrier();
  if (lane == 0u) { finishMergedTotal(merged[0]); }
}

fn finishDirectionCurvatureTotal(total: MergedScalars) {
  if (stopped()) { return; }
  atomicAdd(&control[3], 1u);
  let curvature = total.delta;
  let direct = compensatedValue(curvature);
  let gamma = compensatedValue(pairAt(12u));
  if (!finite(direct) || !(direct > 0.0)) {
    reportAt(ERROR_NONPOSITIVE_CURVATURE, 15u, INVALID);
    accountZeroRemainingAfterUpdate(atomicLoad(&control[2]) - 1u);
    return;
  }
  let alpha = gamma / direct;
  if (!finite(alpha) || !(alpha > 0.0)) {
    reportAt(ERROR_NONFINITE, 15u, INVALID);
    accountZeroRemainingAfterUpdate(atomicLoad(&control[2]) - 1u);
    return;
  }
  storePair(14u, curvature);
  storePair(16u, CompensatedF32(alpha, 0.0));
}

fn curvatureReduceAndFinish(lane: u32) {
  var local = zeroMergedScalars();
  if (lane < REDUCTION_LANES) {
    for (var row = lane; row < rows(); row += REDUCTION_LANES) {
      if (row < rows() && !stopped()) {
        let d = arena[D_BASE + row];
        let image = arena[DIMG_BASE + row];
        if (!finite(d) || !finite(image)) {
          reportAt(ERROR_NONFINITE, 15u, row);
        } else {
          local.delta = addCompensatedF32(local.delta, d * image);
        }
      }
    }
    merged[lane] = local;
  }
  for (var width = REDUCTION_LANES / 2u; width > 0u; width >>= 1u) {
    workgroupBarrier();
    if (lane < width) {
      merged[lane] = mergeScalars(merged[lane], merged[lane + width]);
    }
  }
  workgroupBarrier();
  if (lane == 0u) { finishDirectionCurvatureTotal(merged[0]); }
}

fn initializeDirections(lane: u32) {
  for (var row = lane; row < rows(); row += LANES) {
    if (stopped()) { continue; }
    arena[D_BASE + row] = arena[Z_BASE + row];
    arena[DIMG_BASE + row] = arena[ZIMG_BASE + row];
  }
}

fn updateDirections(lane: u32) {
  for (var row = lane; row < rows(); row += LANES) {
    if (stopped()) { continue; }
    let beta = compensatedValue(pairAt(18u));
    let nextDirection = arena[Z_BASE + row] + beta * arena[D_BASE + row];
    if (!finite(nextDirection)) {
      reportAt(ERROR_NONFINITE, 8u, row);
      continue;
    }
    arena[D_BASE + row] = nextDirection;
  }
}

fn finalizeAndPublish(lane: u32) {
  if (lane == 0u && !failed() && atomicLoad(&control[1]) == 0u) {
    reportAt(ERROR_NONCONVERGENCE, 9u, INVALID);
  }
  sync();
  let fatal = (atomicLoad(&control[0]) & (ERROR_INVALID_AUTHORITY
    | ERROR_INVALID_ROW | ERROR_NONFINITE)) != 0u;
  let converged = atomicLoad(&control[1]) != 0u;
  let advanced = atomicLoad(&control[2]) > 0u;
  let usable = !fatal && (converged || advanced);
  for (var row = lane; row < rows(); row += LANES) {
    let seed = select(0.0, arena[SEED_BASE + row], finite(arena[SEED_BASE + row]));
    let candidate = arena[P_BASE + row];
    arena[OUT_BASE + row] = select(seed, candidate, usable && finite(candidate));
  }
  if (lane == 0u && converged && !fatal) { atomicStore(&control[20], 1u); }
}

@compute @workgroup_size(${config.lanes})
fn residentLosassoMGPCG(@builtin(local_invocation_index) lane: u32) {
  if (lane == 0u) { initializeControl(); }
  sync();
  publishUniformDials(lane);
  let iterationCeiling = workgroupUniformLoad(&wgMaxIterations);
  let sweeps = workgroupUniformLoad(&wgBottomSweeps);
  let smoothing = workgroupUniformLoad(&wgSmoothingSweeps);
  initializeState(lane);
  sync();
  for (var probe = 0u; probe < REPEAT_OP; probe += 1u) { applyOperator(lane, P_BASE, DIMG_BASE); sync(); }
  formInitialResidual(lane);
  sync();
  stageVCycleInputs(lane);
  sync();
  vcycleCorrection(lane, sweeps, smoothing);
  for (var probe = 0u; probe < REPEAT_OP; probe += 1u) { applyOperator(lane, Z_BASE, ZIMG_BASE); sync(); }
  initialExactReduction(lane);
  sync();
  initializeDirections(lane);
  sync();
  for (var iteration = 0u; iteration < iterationCeiling; iteration += 1u) {
    advanceAndStageVCycle(lane);
    sync();
    vcycleCorrection(lane, sweeps, smoothing);
    mergedReduceAndFinish(lane);
    sync();
    if (uniformStopped(lane)) { break; }
    updateDirections(lane);
    sync();
    for (var probe = 0u; probe < REPEAT_OP; probe += 1u) { applyOperator(lane, D_BASE, DIMG_BASE); sync(); }
    curvatureReduceAndFinish(lane);
    sync();
  }
  finalizeAndPublish(lane);
}
`;
}

export interface WebGPUOctreeLosassoResidentMGPCGSource {
  readonly rowCapacity: number;
  /** Reduced operator diagonal, one f32 per compact row (stride 1, bank 0). */
  readonly diagonal: GPUBuffer;
  /** Current divergence/RHS authority, one f32 per compact row. */
  readonly rightHandSide: GPUBuffer;
  /** Accepted `[flags, firstError, rows, slots, epoch, bank, published]`. */
  readonly acceptedAuthority: GPUBuffer;
  readonly hierarchy: OctreeLosassoVCycleHierarchySource;
}

export interface WebGPUOctreeLosassoResidentMGPCGOptions {
  /** Hard iteration envelope; the loop exits at convergence, so this is a
   * ceiling and never a cost. */
  readonly maximumIterations: number;
  readonly relativeTolerance?: number;
  readonly absoluteTolerance?: number;
}

/** Single-dispatch whole-solve executor for the Losasso ≤4K-row coarse tier. */
export class WebGPUOctreeLosassoResidentMGPCG {
  readonly control: GPUBuffer;
  readonly iterationBudget: number;
  readonly allocatedBytes: number;
  readonly encodedDispatchCount = 1 as const;
  readonly initializationTasks: readonly { label: string; run: () => Promise<void> }[];

  private readonly arena: GPUBuffer;
  /** Host-owned staging for the runtime dials; copied into `control` per solve. */
  private readonly tuning: GPUBuffer;
  private readonly tuningWords = new Uint32Array(TUNING_WORDS);
  private pipeline?: GPUComputePipeline;
  private group?: GPUBindGroup;
  private readonly shaderCode: string;
  private readonly shaderLabel: string;
  private destroyed = false;

  static supports(
    device: GPUDevice,
    hierarchy: OctreeLosassoVCycleHierarchySource,
    rowCapacity: number,
  ): boolean {
    const fused = hierarchy.fusedSubL0;
    if (!fused || hierarchy.levels.length < 2) return false;
    if (fused.levelRowCapacities.length !== hierarchy.levels.length) return false;
    const subLevels = fused.levelRowCapacities.slice(1);
    if (subLevels.length < 1
      || subLevels.some((capacity) => capacity > OCTREE_LOSASSO_RESIDENT_MAXIMUM_ROWS)) {
      return false;
    }
    if (rowCapacity > OCTREE_LOSASSO_RESIDENT_MAXIMUM_ROWS) return false;
    const limits = device.limits;
    if (!limits) return true;
    return OCTREE_LOSASSO_RESIDENT_LANES <= limits.maxComputeInvocationsPerWorkgroup
      && OCTREE_LOSASSO_RESIDENT_LANES <= limits.maxComputeWorkgroupSizeX
      && STORAGE_BINDING_COUNT <= limits.maxStorageBuffersPerShaderStage;
  }

  constructor(
    private readonly device: GPUDevice,
    private readonly source: WebGPUOctreeLosassoResidentMGPCGSource,
    options: WebGPUOctreeLosassoResidentMGPCGOptions,
  ) {
    const rowCapacity = source.rowCapacity;
    if (!Number.isSafeInteger(rowCapacity) || rowCapacity < 1
      || rowCapacity > OCTREE_LOSASSO_RESIDENT_MAXIMUM_ROWS) {
      throw new RangeError("Resident Losasso MGPCG row capacity is outside the ≤4,096-row tier");
    }
    const fused = source.hierarchy.fusedSubL0;
    if (!fused || source.hierarchy.levels.length < 2) {
      throw new Error("Resident Losasso MGPCG requires the fused sub-L0 hierarchy publication");
    }
    if (!WebGPUOctreeLosassoResidentMGPCG.supports(device, source.hierarchy, rowCapacity)) {
      throw new Error("Resident Losasso MGPCG is unsupported for this hierarchy/device");
    }
    if (!Number.isSafeInteger(options.maximumIterations) || options.maximumIterations < 4) {
      throw new RangeError("Resident Losasso MGPCG iteration envelope must be at least four");
    }
    const relativeTolerance = options.relativeTolerance ?? 1e-4;
    const absoluteTolerance = options.absoluteTolerance ?? 0;
    if (!Number.isFinite(relativeTolerance) || relativeTolerance < 0
      || !Number.isFinite(absoluteTolerance) || absoluteTolerance < 0
      || (relativeTolerance === 0 && absoluteTolerance === 0)) {
      throw new RangeError("Resident Losasso MGPCG requires a non-zero stopping tolerance");
    }
    if (fused.levelRowCapacities[0]! < rowCapacity) {
      throw new RangeError("Resident Losasso MGPCG finest vector capacity is below the row capacity");
    }
    if (source.diagonal.size < rowCapacity * 4
      || source.rightHandSide.size < rowCapacity * 4
      || source.acceptedAuthority.size < 7 * 4) {
      throw new RangeError("Resident Losasso MGPCG source capacity is too small");
    }
    this.iterationBudget = options.maximumIterations;

    for (const [label, buffer] of [
      ["accepted authority", source.acceptedAuthority],
      ["diagonal", source.diagonal],
      ["divergence RHS", source.rightHandSide],
    ] as const) {
      if ((buffer.usage & GPUBufferUsage.COPY_SRC) === 0) {
        throw new RangeError(`Resident Losasso MGPCG requires COPY_SRC on the ${label} buffer`);
      }
    }
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    this.control = device.createBuffer({
      label: "Resident Losasso MGPCG solve control",
      size: CONTROL_BYTES,
      usage: storage,
    });
    const bases = vectorBases(fused.levelRowCapacities);
    const vectorArenaWords = bases[bases.length - 1]!
      + Math.ceil(fused.levelRowCapacities[fused.levelRowCapacities.length - 1]! * 4 / 256) * 64;
    // Staged out/seed/rhs/diagonal + six CG vectors + four V-cycle sections.
    this.arena = device.createBuffer({
      label: "Resident Losasso MGPCG consolidated solver arena",
      size: (10 * rowCapacity + 4 * vectorArenaWords) * 4,
      usage: storage,
    });
    this.tuning = device.createBuffer({
      label: "Resident Losasso MGPCG staged runtime dials",
      size: TUNING_WORDS * 4,
      usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(this.tuning, 0, this.tuningWords);
    this.allocatedBytes = this.control.size + this.arena.size + this.tuning.size;

    const phaseRepeat = octreeLosassoResidentPhaseRepeatProbe();
    const lanes = octreeLosassoResidentLanes();
    // merged fold slots + exact-reduction limbs + the uniform flag word.
    const workgroupOverheadBytes = REDUCTION_LANES * 32 + 144 * 4 + 4;
    const sharedBottom = source.hierarchy.levels.length === 2
      && (device.limits === undefined
        || 3 * fused.levelRowCapacities[1]! * 4 + workgroupOverheadBytes
          <= device.limits.maxComputeWorkgroupStorageSize);
    this.shaderCode = residentLosassoMGPCGWGSL({
      rowCapacity,
      maximumIterations: options.maximumIterations,
      relativeTolerance,
      absoluteTolerance,
      levelCount: source.hierarchy.levels.length,
      fused,
      lanes,
      sharedBottom,
      ...(phaseRepeat === undefined ? {} : { phaseRepeat }),
    });
    this.shaderLabel = "Resident Losasso MGPCG · whole solve in one workgroup"
      + ` · ${lanes} lanes`
      + (sharedBottom ? " · workgroup-resident bottom smoother" : "")
      + (phaseRepeat === undefined ? ""
        : ` · ${phaseRepeat.phase} x${phaseRepeat.repeats} repeat probe`);
    this.initializationTasks = [{
      label: "Compile resident Losasso MGPCG",
      run: () => this.initialize(),
    }];
  }

  async initialize(): Promise<void> {
    this.assertLive();
    if (this.pipeline) return;
    const shaderModule = this.device.createShaderModule({
      label: this.shaderLabel,
      code: this.shaderCode,
    });
    this.pipeline = await this.device.createComputePipelineAsync({
      label: "Resident Losasso MGPCG · residentLosassoMGPCG",
      layout: "auto",
      compute: { module: shaderModule, entryPoint: "residentLosassoMGPCG" },
    });
  }

  /**
   * Adopt runtime dials for every subsequent solve.
   *
   * This is a queue write into a four-word staging buffer, not a rebuild: the
   * kernel reads the values through the control buffer it already binds, so a
   * dial can move while the water is running. An omitted field, or a value the
   * clamps reject, restores the compiled constant.
   */
  setSolveTuning(tuning: OctreeLosassoSolveTuning): void {
    this.assertLive();
    const cap = tuning.maximumIterations;
    const sweeps = tuning.bottomSweeps;
    const tolerance = tuning.relativeTolerance;
    const next = new Uint32Array(TUNING_WORDS);
    next[0] = Number.isSafeInteger(cap) && cap! > 0
      ? Math.min(cap!, this.iterationBudget) : 0;
    // Odd counts would leave the smoother's result in the wrong ping-pong bank,
    // so an odd request rounds DOWN to the nearest legal pair count.
    next[1] = Number.isSafeInteger(sweeps) && sweeps! >= 2
      ? Math.min(sweeps!, 64) & ~1 : 0;
    next[2] = Number.isFinite(tolerance) && tolerance! > 0
      ? new Uint32Array(Float32Array.of(Math.fround(tolerance!)).buffer)[0]!
      : 0;
    const smoothing = tuning.smoothingSweeps;
    next[3] = Number.isSafeInteger(smoothing) && smoothing! >= 1
      ? Math.min(smoothing!, 16) : 0;
    if (next.every((word, index) => word === this.tuningWords[index])) return;
    this.tuningWords.set(next);
    this.device.queue.writeBuffer(this.tuning, 0, this.tuningWords);
  }

  /**
   * A control clear, five row-sized staging copies, one dispatch, and the
   * published-pressure copy-back. The staged sources all carry this advance's
   * final values at this call site: authority, diagonal and RHS publication
   * precede the solve in the projection graph, exactly as the pipelined
   * executor relies on when it binds them live.
   */
  encodeSolve(broker: PassBroker, input: {
    readonly pressureSeed: GPUBuffer;
    readonly pressureOut: GPUBuffer;
    readonly rightHandSide: GPUBuffer;
    readonly rowCount: GPUBuffer;
  }): void {
    this.assertLive();
    if (!this.pipeline) throw new Error("Resident Losasso MGPCG pipeline is not initialized");
    const rowBytes = this.source.rowCapacity * 4;
    if (input.pressureSeed.size < rowBytes || input.pressureOut.size < rowBytes) {
      throw new RangeError("Resident Losasso MGPCG solve buffers are smaller than the row capacity");
    }
    if ((input.pressureSeed.usage & GPUBufferUsage.COPY_SRC) === 0
      || (input.pressureOut.usage & GPUBufferUsage.COPY_SRC) === 0
      || (input.pressureOut.usage & GPUBufferUsage.COPY_DST) === 0) {
      throw new RangeError("Resident Losasso MGPCG requires copyable pressure banks");
    }
    broker.clearBuffer(this.control);
    broker.copyBufferToBuffer(this.source.acceptedAuthority, 0,
      this.control, AUTHORITY_STAGE_WORD * 4, 7 * 4);
    broker.copyBufferToBuffer(this.tuning, 0,
      this.control, TUNING_STAGE_WORD * 4, TUNING_WORDS * 4);
    // Rows past the live count must retain their previous published values,
    // so the copy-back below starts from the current bank content.
    broker.copyBufferToBuffer(input.pressureOut, 0, this.arena, 0, rowBytes);
    broker.copyBufferToBuffer(input.pressureSeed, 0, this.arena, rowBytes, rowBytes);
    broker.copyBufferToBuffer(this.source.rightHandSide, 0, this.arena, 2 * rowBytes, rowBytes);
    broker.copyBufferToBuffer(this.source.diagonal, 0, this.arena, 3 * rowBytes, rowBytes);
    const pass = broker.compute({
      label: "Resident Losasso MGPCG - whole solve in one workgroup",
    });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup());
    pass.dispatchWorkgroups(1);
    broker.copyBufferToBuffer(this.arena, 0, input.pressureOut, 0, rowBytes);
    broker.fence("resident Losasso MGPCG pressure publication");
  }

  private bindGroup(): GPUBindGroup {
    if (this.group) return this.group;
    const finest = this.source.hierarchy.levels[0]!;
    const resources: readonly GPUBuffer[] = [
      finest.control,
      finest.rowFaceOffsets,
      finest.rowFaces,
      finest.faces,
      this.source.hierarchy.fusedSubL0!.arena,
      this.arena,
      this.control,
    ];
    this.group = this.device.createBindGroup({
      label: "Resident Losasso MGPCG bindings",
      layout: this.pipeline!.getBindGroupLayout(0),
      entries: resources.map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
    return this.group;
  }

  private assertLive(): void {
    if (this.destroyed) throw new Error("Resident Losasso MGPCG executor is destroyed");
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.group = undefined;
    this.control.destroy();
    this.arena.destroy();
    this.tuning.destroy();
  }
}
