/**
 * GPU orchestration primitive for matrix-free preconditioned CG.
 *
 * The Section 4.3 hybrid/V-cycle storage remains owned by the supplied fixed
 * preconditioner. This primitive owns only scalar control, reduction partials,
 * and indirect outer-iteration records; recurrence vectors are supplied by
 * the integrator. There is no CPU solve or legacy-solver fallback.
 */
import type { OctreeFirstOrderSPDVCycle } from "./webgpu-octree-section43-contract";
import { PassBroker } from "./webgpu-pass-broker";
import {
  FLUID_M1_MAX_REDUCTION_LANES,
  supportsFluidM1MaxReduction,
} from "./webgpu-device-limits";
import {
  createGPULogicalActivityAdoptionContext,
  stableGPULogicalActivityId,
  type GPULogicalActivityAdoptionContext,
} from "./gpu-logical-activity-adoption";
import { performanceShaderVariant } from "./stores/performance-instrumentation-store";
import { gpuCompilationManagerFor } from "./gpu-compilation-manager";

export const OCTREE_PIPELINED_PCG_WORKGROUP_SIZE =
  FLUID_M1_MAX_REDUCTION_LANES;
/**
 * The production caller preserves the established f32 graphics-solve floor
 * of 1e-4. Keep encoded headroom for harder systems; converged solves do not
 * execute their indirect tail.
 */
export const OCTREE_PIPELINED_PCG_DEFAULT_ITERATIONS = 10;
/** Retained as validation metadata; production never emits this many bodies. */
export const OCTREE_PIPELINED_PCG_HARD_ITERATION_CEILING = 16;
export const OCTREE_PIPELINED_PCG_CONTROL_BYTES = 128;
/**
 * Exact radix-256 superaccumulator geometry.
 *
 * Every finite f32 is an integer significand times a power of two.  Using
 * limbs whose least-significant bit is 2^-152 covers every subnormal through
 * every finite normal; the final limb is reserved for signed carry.  A row
 * contributes at most four byte digits to a scalar, so one i32 limb cannot
 * overflow anywhere inside WebGPU's maximum one-dimensional row dispatch.
 */
export const OCTREE_PIPELINED_PCG_FIXED_POINT_LIMBS = 36;
export const OCTREE_PIPELINED_PCG_FIXED_POINT_SCALARS = 4;
export const OCTREE_PIPELINED_PCG_PARTIAL_BYTES =
  OCTREE_PIPELINED_PCG_FIXED_POINT_LIMBS
  * OCTREE_PIPELINED_PCG_FIXED_POINT_SCALARS * 4;
export const OCTREE_PIPELINED_PCG_INDIRECT_STRIDE_BYTES = 16;
export const OCTREE_PIPELINED_PCG_DISPATCHES_PER_ITERATION = 4;
export const OCTREE_PIPELINED_MGPCG_ACTIVITY_MODULE_ID = "octree/pipelined-mgpcg";
/** The solver repeats several dispatches per iteration; keep its shared-recorder load bounded. */

export const OCTREE_PIPELINED_PCG_ERROR = Object.freeze({
  invalidAuthority: 1 << 0,
  invalidRow: 1 << 1,
  nonFinite: 1 << 2,
  nonPositivePreconditioner: 1 << 3,
  nonPositiveCurvature: 1 << 4,
  nonConvergence: 1 << 5,
} as const);

/** Word offsets in the host-readable solve control record. */
export const OCTREE_PIPELINED_PCG_CONTROL = Object.freeze({
  error: 0,
  converged: 1,
  iterations: 2,
  reductions: 3,
  liveRows: 4,
  zeroedDispatches: 5,
  failureStage: 6,
  failureRow: 7,
  rightHandSideSquaredHi: 8,
  rightHandSideSquaredLo: 9,
  residualSquaredHi: 10,
  residualSquaredLo: 11,
  gammaHi: 12,
  gammaLo: 13,
  deltaHi: 14,
  deltaLo: 15,
  alphaHi: 16,
  alphaLo: 17,
  betaHi: 18,
  betaLo: 19,
  published: 20,
  temporalAlpha: 21,
  temporalApplied: 22,
  temporalNumerator: 23,
  temporalCurvature: 24,
  temporalDirectionSquared: 25,
} as const);

export type CompensatedF32 = readonly [hi: number, lo: number];

function twoSumF32(left: number, right: number): CompensatedF32 {
  const hi = Math.fround(Math.fround(left) + Math.fround(right));
  const rightVirtual = Math.fround(hi - Math.fround(left));
  const lo = Math.fround(
    Math.fround(Math.fround(left) - Math.fround(hi - rightVirtual))
      + Math.fround(Math.fround(right) - rightVirtual),
  );
  return [hi, lo];
}

/** CPU mirror of the WGSL `(hi, lo)` f32 accumulation helper. */
export function addCompensatedF32(
  accumulator: CompensatedF32,
  value: number,
): CompensatedF32 {
  if (!Number.isFinite(accumulator[0]) || !Number.isFinite(accumulator[1])
    || !Number.isFinite(value)) return [Number.NaN, Number.NaN];
  const first = twoSumF32(accumulator[0], value);
  const second = twoSumF32(first[1], accumulator[1]);
  return twoSumF32(first[0], Math.fround(second[0] + second[1]));
}

export function mergeCompensatedF32(
  left: CompensatedF32,
  right: CompensatedF32,
): CompensatedF32 {
  const withHigh = addCompensatedF32(left, right[0]);
  return addCompensatedF32(withHigh, right[1]);
}

export function compensatedF32Value(value: CompensatedF32): number {
  return value[0] + value[1];
}

export interface OctreePipelinedMGPCGPlan {
  readonly rowCapacity: number;
  readonly maximumIterations: number;
  readonly hardIterationCeiling: 16;
  readonly reductionLanes: 128;
  readonly rowDispatch: readonly [number, number, number];
  readonly reductionDispatch: readonly [number, number, number];
  readonly reductionPartialCount: number;
  readonly initialReductionCount: 1;
  readonly reductionsPerOuterIteration: 2;
  readonly maximumReductionCount: number;
  readonly mergedScalarsPerReduction: 3;
  readonly dispatchRecordCount: number;
  readonly indirectBytes: number;
  readonly reductionPartialBytes: number;
  readonly ownedBytes: number;
}

export interface OctreePipelinedMGPCGOptions {
  readonly rowCapacity: number;
  readonly maximumIterations?: number;
  readonly hardIterationCeiling?: number;
  readonly relativeTolerance?: number;
  readonly absoluteTolerance?: number;
  /**
   * Measurement-only override of the seeded initial iterate. Omitted (the
   * production default) reads `FLUID_OCTREE_PRESSURE_COLD_START`; see
   * `octreePipelinedMGPCGColdStartEnabled`. Never set this from product code.
   */
  readonly coldStart?: boolean;
  /** Measurement arm for the rank-one previous-pressure secant predictor. */
  readonly temporalPredictor?: boolean;
  readonly temporalPredictorMode?: "fixed" | "current-operator";
  readonly temporalPredictorAlpha?: number;
  /**
   * Factor-1 measurement arm: replace each outer partial/finish reduction
   * pair with one exact-association 128-lane drain. Callers must prove factor
   * one; the executor independently enforces the measured 4,096-row ceiling.
   */
  readonly factorOneCombinedReductionDrains?: boolean;
}

function positiveInteger(value: number, label: string): number {
  if (!Number.isSafeInteger(value) || value < 1) {
    throw new RangeError(`${label} must be a positive integer`);
  }
  return value;
}

function finiteNonNegative(value: number, label: string): number {
  if (!Number.isFinite(value) || value < 0) {
    throw new RangeError(`${label} must be finite and non-negative`);
  }
  return value;
}

export function planOctreePipelinedMGPCG(
  options: Pick<OctreePipelinedMGPCGOptions,
    "rowCapacity" | "maximumIterations" | "hardIterationCeiling">,
): OctreePipelinedMGPCGPlan {
  const rowCapacity = positiveInteger(options.rowCapacity, "Pipelined MGPCG row capacity");
  const maximumIterations = positiveInteger(
    options.maximumIterations ?? OCTREE_PIPELINED_PCG_DEFAULT_ITERATIONS,
    "Pipelined MGPCG maximum iterations",
  );
  const hardIterationCeiling = positiveInteger(
    options.hardIterationCeiling ?? OCTREE_PIPELINED_PCG_HARD_ITERATION_CEILING,
    "Pipelined MGPCG hard iteration ceiling",
  );
  if (hardIterationCeiling !== OCTREE_PIPELINED_PCG_HARD_ITERATION_CEILING) {
    throw new RangeError("Pipelined MGPCG hard iteration ceiling is fixed at 16");
  }
  if (maximumIterations < 4 || maximumIterations > 10) {
    throw new RangeError("Pipelined MGPCG encoded iterations must remain in [4,10]");
  }
  const reductionPartialCount = Math.ceil(
    rowCapacity / OCTREE_PIPELINED_PCG_WORKGROUP_SIZE,
  );
  if (reductionPartialCount > 65_535) {
    throw new RangeError("Pipelined MGPCG row capacity exceeds one-dimensional reduction dispatch");
  }
  const rowGroups = Math.ceil(rowCapacity / 64);
  if (rowGroups > 65_535) {
    throw new RangeError("Pipelined MGPCG row capacity exceeds one-dimensional row dispatch");
  }
  const dispatchRecordCount = maximumIterations
    * OCTREE_PIPELINED_PCG_DISPATCHES_PER_ITERATION;
  const indirectBytes = dispatchRecordCount * OCTREE_PIPELINED_PCG_INDIRECT_STRIDE_BYTES;
  const reductionPartialBytes = reductionPartialCount * OCTREE_PIPELINED_PCG_PARTIAL_BYTES;
  return Object.freeze({
    rowCapacity,
    maximumIterations,
    hardIterationCeiling: OCTREE_PIPELINED_PCG_HARD_ITERATION_CEILING,
    reductionLanes: OCTREE_PIPELINED_PCG_WORKGROUP_SIZE,
    rowDispatch: [rowGroups, 1, 1] as const,
    reductionDispatch: [reductionPartialCount, 1, 1] as const,
    reductionPartialCount,
    initialReductionCount: 1 as const,
    reductionsPerOuterIteration: 2 as const,
    maximumReductionCount: 2 * maximumIterations,
    mergedScalarsPerReduction: 3 as const,
    dispatchRecordCount,
    indirectBytes,
    reductionPartialBytes,
    ownedBytes: OCTREE_PIPELINED_PCG_CONTROL_BYTES + 64
      + indirectBytes + reductionPartialBytes,
  });
}

/**
 * The fixed preconditioner contract is intentionally the existing proof-
 * carrying V-cycle interface. Its kernels already consume the common
 * fail-closed solve control and use a fixed linear schedule.
 */
export type OctreePipelinedFixedPreconditioner = OctreeFirstOrderSPDVCycle & {
  /** Every nested row/page/level launch consumes a solve-control-gated record. */
  readonly convergenceTail: "gpu-zero-indirect";
};

export interface OctreePipelinedMGPCGVectors {
  readonly pressure: GPUBuffer;
  readonly residual: GPUBuffer;
  readonly preconditioned: GPUBuffer;
  readonly preconditionedImage: GPUBuffer;
  readonly direction: GPUBuffer;
  readonly directionImage: GPUBuffer;
}

/** Matrix-free accurate L2 apply. Production supplies the four disjoint
 * resolved-row class pipelines; the Krylov owner never sees CSR entries. */
export interface OctreePipelinedLinearOperator {
  /** Every class/page launch consumes a solve-control-gated indirect record. */
  readonly convergenceTail: "gpu-zero-indirect";
  readonly encodedDispatchCount: number;
  /** Dispatches used by encodeResidualBody when it intentionally retains a
   * different portable binding shape from the ordinary apply. */
  readonly encodedResidualDispatchCount?: number;
  encode(
    broker: PassBroker,
    input: GPUBuffer,
    output: GPUBuffer,
    solverControl: GPUBuffer,
  ): void;
  /** Optional split form used to co-publish independent indirect schedules. */
  encodeGate?(
    pass: GPUComputePassEncoder,
    input: GPUBuffer,
    output: GPUBuffer,
    solverControl: GPUBuffer,
  ): void;
  encodeBody?(
    broker: PassBroker,
    input: GPUBuffer,
    output: GPUBuffer,
    solverControl: GPUBuffer,
  ): void;
  /**
   * Optional body-only specialization for a caller that needs
   * `residualRhs - A * input`. It consumes the same convergence gate as
   * encodeBody and writes the residual directly, avoiding an intermediate
   * operator-image round trip. Implementations must preserve encodeBody's
   * exact per-row operator fold before the final subtraction.
   */
  encodeResidualBody?(
    broker: PassBroker,
    input: GPUBuffer,
    residualRhs: GPUBuffer,
    residual: GPUBuffer,
    solverControl: GPUBuffer,
  ): void;
}

/** Accurate L2 operator with a compact four-class workset override. The
 * Section 4.3 shell requires this contract so its repeated smoothing applies
 * cannot silently fall back to all live pressure rows. */
export interface OctreePipelinedWorksetLinearOperator
extends OctreePipelinedLinearOperator {
  /** Dispatches needed to apply the operator to the compact union workset. */
  readonly encodedMergedBandDispatchCount: number;
  encodeWorksets(
    broker: PassBroker,
    input: GPUBuffer,
    output: GPUBuffer,
    solverControl: GPUBuffer,
    worksets: GPUBuffer,
    classDispatch: GPUBuffer,
    worksetLayout: GPUBuffer,
    classDispatchOffsetBytes?: number,
  ): void;
  encodeMergedBandWorkset(
    broker: PassBroker,
    input: GPUBuffer,
    output: GPUBuffer,
    solverControl: GPUBuffer,
    worksets: GPUBuffer,
    mergedDispatch: GPUBuffer,
    worksetLayout: GPUBuffer,
    mergedDispatchOffsetBytes: number,
  ): void;
}

export interface OctreePipelinedMGPCGSource {
  /** Banked Section 6.3 diagonal + eighteen canonical coefficients. */
  readonly coefficients: GPUBuffer;
  /** Current divergence/RHS authority, one f32 per compact row. */
  readonly rhs: GPUBuffer;
  readonly rowCount: GPUBuffer;
  /** Exact accepted-row indirect record authored by structured publication. */
  readonly rowDispatch: GPUBuffer;
  readonly rowDispatchOffsetBytes?: number;
  /** Accepted dynamic-boundary publication control:
   * [flags, firstError, rows, slots, epoch, bank, published]. */
  readonly acceptedAuthority: GPUBuffer;
  readonly operator: OctreePipelinedLinearOperator;
  readonly preconditioner: OctreePipelinedFixedPreconditioner;
  readonly vectors: OctreePipelinedMGPCGVectors;
}

/**
 * The solve is ALREADY WARM-STARTED, unconditionally, and has been since the
 * structured cutover. This is written down here because the seed path is
 * assembled in three files and reads, from inside this one, like a fail-closed
 * fallback rather than an initial iterate -- which has now cost more than one
 * session an attempt to "add" a warm start that already ships.
 *
 * How `pressureSeed` gets the previous frame's pressure, end to end:
 *
 * 1. Tail of substep N, `encodeInactiveTopologyCandidate` in
 *    `webgpu-octree.ts` builds the row-delta transaction. `mergeFrontierRows`
 *    merges the old and new row sets sorted by `(level, Morton)` and publishes
 *    `frontier[rowDeltaNewToOldBase() + newRow]`, a total map from every new
 *    row to its predecessor, encoded +1 so that 0 means "no predecessor". Row
 *    identity is therefore keyed on the exact `(cell, size)` octree leaf, never
 *    on a row index; `mergeOctreePowerRowIdentities` is the CPU oracle for it.
 * 2. The same pass's `emitLeaves` reads that map and writes the remapped
 *    pressure into the candidate seed buffer:
 *      `previousRow = rowDeltaMapOld(frontier[rowDeltaNewToOldBase() + row])`
 *      `pressureOut[row] = select(0.0, pressureIn[previousRow], warmLane)`
 *    A new row with no predecessor decodes to `INVALID` and seeds exactly 0,
 *    which is the fail-closed case. `warmLane` is `params.pressureCapacity.w & 1u`,
 *    which `webgpu-octree.ts` writes as `flags = 1 | (generation << 2)`: bit 0
 *    is a hard 1, and `tests/octree-coarse-phi-authority.test.ts` pins that the
 *    warm lane "must not remain configurable".
 * 3. Head of substep N+1, `commitCandidateRows`
 *    (`webgpu-octree-topology-epoch.ts`, pass label "Commit accepted topology
 *    row identities and pressure seed") copies the candidate seed into BOTH
 *    pressure banks, gated on the accepted-epoch token. A rejected candidate
 *    copies nothing and the previous accepted seed is retained.
 * 4. `WebGPUOctreeProjection.encode` passes the bank as `pressureSeed`;
 *    `initializeState` sets `pressure[row] = seed`, the operator maps it into
 *    `directionImage`, and `formInitialResidual` forms `r0 = -rhs - A*p_seed`.
 *
 * So CG starts from the previous frame's converged pressure, and because the
 * stopping threshold is relative to `||b||` rather than `||r0||`, a smaller
 * `r0` converts directly into fewer executed iterations. Two consequences for
 * anyone shopping for iteration count:
 *
 * - There is no unclaimed warm-start win here. The measured 4-5 executed
 *   iterations of the encoded 10 are the WARM number.
 * - The seed is dimensional and scales as 1/dt (`rhs = rho*flux/dt`, and A is
 *   dt-independent -- see `encodeForcesAndDivergence`). A lane with a varying
 *   dt seeds a mis-scaled iterate. The Dawn gate lanes are fixed-dt; the
 *   browser's fractional rAF dts are not.
 */
export interface OctreePipelinedMGPCGSolve {
  /**
   * Previous-generation pressure remapped onto this generation's rows. Both
   * the initial iterate and the fail-closed publication fallback.
   */
  readonly pressureSeed: GPUBuffer;
  /** Accepted row-remapped temporal secant p[n-1] - p[n-2]. */
  readonly pressureHistory?: GPUBuffer;
  readonly pressureOut: GPUBuffer;
  /**
   * Encode-only schedule shaping. Allocation and the production envelope stay
   * at `plan.maximumIterations`; this may select only a prefix inside it.
   */
  readonly encodedIterationBudget?: number;
}

/**
 * Measurement instrument for the warm start documented on
 * `OctreePipelinedMGPCGSolve`. **Default off, and off is today.**
 *
 * The flag is named for the change it makes rather than for the feature,
 * because the feature already ships enabled: a `..._WARM_START` flag defaulting
 * off would have to mean "off = warm", which is a trap. Setting
 * `FLUID_OCTREE_PRESSURE_COLD_START=1` makes `initializeState` begin from
 * `p = 0` instead of the published seed, so one binary can measure what the
 * warm start is worth on a lane under the document's interleaved A/B protocol.
 * Everything else -- the row remap, the epoch commit, the encoded dispatch
 * graph, the reduction tree, the fail-closed seed republication in
 * `finalizeAndPublish` -- is untouched, so the only difference between the arms
 * is the initial iterate.
 *
 * Unset, this returns false and the module compiles `octreePipelinedMGPCGShader`
 * byte-for-byte, with the same module cache key and the same shader label.
 * Set, it changes the iterate sequence: Gate B, never Gate A.
 */
export function octreePipelinedMGPCGColdStartEnabled(
  environment?: Readonly<Record<string, string | undefined>>,
): boolean {
  const resolved = environment
    ?? (typeof process !== "undefined" ? process.env : undefined);
  return resolved?.FLUID_OCTREE_PRESSURE_COLD_START === "1";
}

/** Opt-in A/B arm. Production remains on the established warm seed. */
export function octreePipelinedMGPCGTemporalPredictorEnabled(
  environment?: Readonly<Record<string, string | undefined>>,
): boolean {
  const resolved = environment
    ?? (typeof process !== "undefined" ? process.env : undefined);
  return resolved?.FLUID_OCTREE_PRESSURE_TEMPORAL_PREDICTOR === "1"
    || resolved?.FLUID_OCTREE_PRESSURE_TEMPORAL_PREDICTOR === "current-operator";
}

export function octreePipelinedMGPCGTemporalPredictorMode(
  environment?: Readonly<Record<string, string | undefined>>,
): "off" | "fixed" | "current-operator" {
  const resolved = environment
    ?? (typeof process !== "undefined" ? process.env : undefined);
  const value = resolved?.FLUID_OCTREE_PRESSURE_TEMPORAL_PREDICTOR;
  if (value === "current-operator") return "current-operator";
  return value === "1" ? "fixed" : "off";
}

export function octreePipelinedMGPCGTemporalPredictorAlpha(
  environment?: Readonly<Record<string, string | undefined>>,
): number {
  const resolved = environment
    ?? (typeof process !== "undefined" ? process.env : undefined);
  const parsed = Number(resolved?.FLUID_OCTREE_PRESSURE_TEMPORAL_ALPHA ?? "0.25");
  return Number.isFinite(parsed) && parsed >= -1 && parsed <= 2 ? parsed : 0.25;
}

/** Explicit-on until the factor-1 combined reduction A/B has passed. */
export function octreePipelinedMGPCGCombinedReductionDrainsEnabled(
  environment: Record<string, string | undefined> | undefined =
    typeof process !== "undefined" ? process.env : undefined,
): boolean {
  return environment?.FLUID_OCTREE_FACTOR1_COMBINED_REDUCTION_DRAINS === "1";
}

/** QA-only snapshots bracketing the first Section 4.3 preconditioner apply. */
export function octreePipelinedMGPCGSymmetryStageAuditEnabled(
  environment: Record<string, string | undefined> | undefined =
    typeof process !== "undefined" ? process.env : undefined,
): boolean {
  return environment?.FLUID_SYMMETRY_STAGE_AUDIT === "1";
}

export type OctreePipelinedMGPCGPipelineName =
  | "initializeControlAndDispatch"
  | "validateAuthority"
  | "initializeState"
  | "formInitialResidual"
  | "reduceMergedPartials"
  | "finishMergedReduction"
  | "reduceAndFinishMerged"
  | "reduceDirectionCurvaturePartials"
  | "finishDirectionCurvature"
  | "reduceAndFinishDirectionCurvature"
  | "initializeDirections"
  | "advancePCGState"
  | "updateDirections"
  | "finalizeAndPublish";

type PipelineName = OctreePipelinedMGPCGPipelineName;

export interface OctreePipelinedMGPCGActivityTaskDescriptor {
  readonly entryPoint: PipelineName;
  readonly task: string;
  readonly taskId: number;
  readonly id: string;
  readonly label: string;
  readonly checkpoints: Readonly<{ enter: number; exit: number }>;
}

const mgpcgActivityId = (kind: "task" | "checkpoint", task: string, checkpoint?: string) =>
  stableGPULogicalActivityId([
    kind,
    OCTREE_PIPELINED_MGPCG_ACTIVITY_MODULE_ID,
    task,
    ...(checkpoint ? [checkpoint] : []),
  ].join("\0"));

function mgpcgActivityTask(
  entryPoint: PipelineName,
  task: string,
  label: string,
): OctreePipelinedMGPCGActivityTaskDescriptor {
  return Object.freeze({
    entryPoint,
    task,
    taskId: mgpcgActivityId("task", task),
    id: `gpu.physics.pipelined-mgpcg.${task}`,
    label,
    checkpoints: Object.freeze({
      enter: mgpcgActivityId("checkpoint", task, "enter"),
      exit: mgpcgActivityId("checkpoint", task, "exit"),
    }),
  });
}

/** Stable descriptors consumed by shader generation and the frame activity task map. */
export const OCTREE_PIPELINED_MGPCG_ACTIVITY_TASKS: Readonly<
Record<PipelineName, OctreePipelinedMGPCGActivityTaskDescriptor>
> = Object.freeze({
  initializeControlAndDispatch: mgpcgActivityTask("initializeControlAndDispatch", "prepare-control", "MGPCG · prepare solve control"),
  validateAuthority: mgpcgActivityTask("validateAuthority", "validate-authority", "MGPCG · validate structured authority"),
  initializeState: mgpcgActivityTask("initializeState", "initialize-state", "MGPCG · initialize pressure state"),
  formInitialResidual: mgpcgActivityTask("formInitialResidual", "form-initial-residual", "MGPCG · form initial residual"),
  reduceMergedPartials: mgpcgActivityTask("reduceMergedPartials", "reduce-merged-partials", "MGPCG · reduce residual scalars"),
  finishMergedReduction: mgpcgActivityTask("finishMergedReduction", "finish-merged-reduction", "MGPCG · finish residual reduction"),
  reduceAndFinishMerged: mgpcgActivityTask("reduceAndFinishMerged", "combined-merged-reduction", "MGPCG · combined residual reduction drain"),
  reduceDirectionCurvaturePartials: mgpcgActivityTask("reduceDirectionCurvaturePartials", "reduce-direction-curvature", "MGPCG · reduce direction curvature"),
  finishDirectionCurvature: mgpcgActivityTask("finishDirectionCurvature", "finish-direction-curvature", "MGPCG · finish direction curvature"),
  reduceAndFinishDirectionCurvature: mgpcgActivityTask("reduceAndFinishDirectionCurvature", "combined-direction-curvature", "MGPCG · combined direction-curvature drain"),
  initializeDirections: mgpcgActivityTask("initializeDirections", "initialize-directions", "MGPCG · initialize search direction"),
  advancePCGState: mgpcgActivityTask("advancePCGState", "advance-pcg-state", "MGPCG · advance pressure and residual"),
  updateDirections: mgpcgActivityTask("updateDirections", "update-directions", "MGPCG · update search direction"),
  finalizeAndPublish: mgpcgActivityTask("finalizeAndPublish", "finalize-pressure", "MGPCG · publish pressure"),
});

const PIPELINE_BINDINGS: Readonly<Record<PipelineName, readonly number[]>> = Object.freeze({
  initializeControlAndDispatch: [0, 2, 13, 15],
  validateAuthority: [0, 2, 13],
  initializeState: [0, 1, 2, 5, 7, 8, 13, 16],
  formInitialResidual: [0, 2, 8, 12, 13, 16],
  reduceMergedPartials: [0, 2, 8, 9, 10, 13, 14, 16],
  finishMergedReduction: [0, 2, 13, 14, 15],
  reduceAndFinishMerged: [0, 2, 8, 9, 10, 13, 15, 16],
  reduceDirectionCurvaturePartials: [0, 2, 11, 12, 13, 14],
  finishDirectionCurvature: [0, 2, 13, 14, 15],
  reduceAndFinishDirectionCurvature: [0, 2, 11, 12, 13, 15],
  initializeDirections: [0, 2, 9, 10, 11, 12, 13],
  advancePCGState: [0, 2, 7, 8, 11, 12, 13],
  updateDirections: [0, 2, 9, 11, 13],
  finalizeAndPublish: [0, 2, 5, 6, 7, 13],
});

export const OCTREE_PIPELINED_MGPCG_MAX_PRODUCTION_STORAGE_BINDINGS = Math.max(
  ...Object.values(PIPELINE_BINDINGS).map((bindings) => bindings.filter((binding) => binding !== 0).length),
);

interface CachedGroup {
  readonly resources: readonly (GPUBuffer | undefined)[];
  readonly group: GPUBindGroup;
}

type TemporalPredictorPipelineName =
  | "initializeTemporalDirection"
  | "reduceTemporalPartials"
  | "finishTemporalReduction"
  | "applyTemporalPrediction";

const TEMPORAL_PREDICTOR_BINDINGS: Readonly<Record<
TemporalPredictorPipelineName, readonly number[]
>> = Object.freeze({
  initializeTemporalDirection: [0, 2, 11, 13, 17],
  reduceTemporalPartials: [0, 2, 10, 11, 12, 13, 14, 16],
  finishTemporalReduction: [0, 2, 13, 14],
  applyTemporalPrediction: [0, 2, 5, 7, 8, 10, 11, 12, 13, 16],
});

/**
 * Replacement-ready matrix-free PCG owner. It deliberately does not
 * allocate hybrid smoother or multigrid vectors.
 */
export class WebGPUOctreePipelinedMGPCG {
  readonly plan: OctreePipelinedMGPCGPlan;
  readonly iterationBudget: number;
  readonly control: GPUBuffer;
  readonly allocatedBytes: number;
  readonly encodedPassTransitionCountPerIteration: 2 | 4;
  readonly reductionsPerOuterIteration = 2 as const;
  readonly usesCombinedReductionDrains: boolean;
  /**
   * True in production: `initializeState` begins from `solve.pressureSeed`.
   * False only under the `FLUID_OCTREE_PRESSURE_COLD_START` measurement arm.
   */
  readonly startsFromPublishedSeed: boolean;
  readonly usesTemporalPredictor: boolean;
  readonly usesCurrentOperatorTemporalPredictor: boolean;
  readonly temporalPredictorAlpha: number;

  get symmetryStageAuditBuffers(): Readonly<{
    initialResidual: GPUBuffer;
    initialPreconditioned: GPUBuffer;
    initialPreconditionedImage: GPUBuffer;
    preconditionerPreSmoothed: GPUBuffer;
    preconditionerZeroSmoothed: GPUBuffer;
    preconditionerFirstOperatorImage: GPUBuffer;
    preconditionerFirstSmoothed: GPUBuffer;
    preconditionerInnerResidual: GPUBuffer;
    preconditionerInnerCorrection: GPUBuffer;
    preconditionerPostCorrected: GPUBuffer;
  }> | undefined { return this.symmetryStageAudit; }

  /** GPU-authored records suitable for the existing diagnostics readback. */
  get workAccountingBuffers(): Readonly<{ control: GPUBuffer; indirectTail: GPUBuffer }> {
    return Object.freeze({ control: this.control, indirectTail: this.outerDispatch });
  }

  get workAccountingPlan(): Readonly<{ rowCapacity: number; maximumIterations: number;
    combinedReductionDrains: boolean;
    reductionLanes: 128; reductionPartialCount: number }> {
    return Object.freeze({ rowCapacity: this.plan.rowCapacity,
      maximumIterations: this.plan.maximumIterations,
      combinedReductionDrains: this.usesCombinedReductionDrains,
      reductionLanes: this.plan.reductionLanes,
      reductionPartialCount: this.plan.reductionPartialCount });
  }

  private readonly params: GPUBuffer;
  private readonly partials: GPUBuffer;
  private readonly outerDispatch: GPUBuffer;
  private readonly symmetryStageAudit?: {
    initialResidual: GPUBuffer;
    initialPreconditioned: GPUBuffer;
    initialPreconditionedImage: GPUBuffer;
    preconditionerPreSmoothed: GPUBuffer;
    preconditionerZeroSmoothed: GPUBuffer;
    preconditionerFirstOperatorImage: GPUBuffer;
    preconditionerFirstSmoothed: GPUBuffer;
    preconditionerInnerResidual: GPUBuffer;
    preconditionerInnerCorrection: GPUBuffer;
    preconditionerPostCorrected: GPUBuffer;
  };
  private pipelines!: Readonly<Record<PipelineName, GPUComputePipeline>>;
  private temporalPipelines?: Readonly<Record<
    TemporalPredictorPipelineName, GPUComputePipeline
  >>;
  private readonly activity: GPULogicalActivityAdoptionContext;
  private readonly shaderCode: string;
  private readonly shaderLabel: string;
  private pipelineInitialization?: Promise<void>;
  private readonly groups = new Map<PipelineName, CachedGroup[]>();
  private readonly temporalGroups = new Map<TemporalPredictorPipelineName, CachedGroup[]>();
  private readonly relativeTolerance: number;
  private readonly absoluteTolerance: number;
  private parameterIterationBudget: number;
  private destroyed = false;

  constructor(
    private readonly device: GPUDevice,
    private readonly source: OctreePipelinedMGPCGSource,
    options: OctreePipelinedMGPCGOptions,
  ) {
    this.plan = planOctreePipelinedMGPCG(options);
    this.iterationBudget = this.plan.maximumIterations;
    this.parameterIterationBudget = this.plan.maximumIterations;
    this.usesCombinedReductionDrains =
      options.factorOneCombinedReductionDrains === true;
    if (this.usesCombinedReductionDrains && this.plan.rowCapacity > 4_096) {
      throw new RangeError(
        "Factor-1 combined reduction drains require row capacity at most 4,096",
      );
    }
    this.encodedPassTransitionCountPerIteration =
      this.usesCombinedReductionDrains ? 2 : 4;
    const preconditioner = source.preconditioner;
    if (!preconditioner || preconditioner.operatorOrder !== 1
      || preconditioner.isSymmetricPositiveDefinite !== true
      || preconditioner.convergenceTail !== "gpu-zero-indirect"
      || !Number.isSafeInteger(preconditioner.encodedCorrectionDispatchCount)
      || preconditioner.encodedCorrectionDispatchCount < 1) {
      throw new Error("Pipelined MGPCG requires an explicit fixed-schedule SPD first-order V-cycle");
    }
    // The reduction is the pinned shared-memory tree, so no `subgroupAdd`
    // remains -- see targetSubgroupReductionWGSL for why that is a determinism
    // requirement and not a preference. This comment went stale once before,
    // when a revert restored the subgroup form without restoring the comment,
    // so the invariant is pinned by a test on the emitted WGSL rather than by
    // this prose: octreePipelinedMGPCGShader must contain no subgroupAdd.
    // The feature gate, the `enable subgroups;` directive and the
    // `subgroup_invocation_id`/`subgroup_size` parameters stay because the GPU
    // logical-activity variant still emits ballots through those builtins.
    if (!device.features.has("subgroups")) {
      throw new Error("Pipelined MGPCG requires the target subgroup feature");
    }
    if (!supportsFluidM1MaxReduction(device.limits)) {
      throw new Error("Pipelined MGPCG 128-lane target exceeds device limits");
    }
    this.relativeTolerance = finiteNonNegative(
      options.relativeTolerance ?? 1e-4,
      "Pipelined MGPCG relative tolerance",
    );
    this.absoluteTolerance = finiteNonNegative(
      options.absoluteTolerance ?? 0,
      "Pipelined MGPCG absolute tolerance",
    );
    if (this.relativeTolerance === 0 && this.absoluteTolerance === 0) {
      throw new RangeError("Pipelined MGPCG requires a non-zero stopping tolerance");
    }
    if (source.coefficients.size < 2 * this.plan.rowCapacity * 19 * 4
      || source.rhs.size < this.plan.rowCapacity * 4
      || source.rowCount.size < 4
      || source.operator.convergenceTail !== "gpu-zero-indirect"
      || !Number.isSafeInteger(source.operator.encodedDispatchCount)
      || source.operator.encodedDispatchCount < 1) {
      throw new RangeError("Pipelined MGPCG Section 6.3 source is invalid");
    }
    for (const [name, vector] of Object.entries(source.vectors)) {
      if (vector.size < this.plan.rowCapacity * 4) {
        throw new RangeError(`Pipelined MGPCG ${name} vector capacity is too small`);
      }
    }

    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
      | GPUBufferUsage.COPY_SRC;
    this.control = device.createBuffer({
      label: "Pipelined MGPCG compensated solve control",
      size: OCTREE_PIPELINED_PCG_CONTROL_BYTES,
      usage: storage,
    });
    this.params = device.createBuffer({
      label: "Pipelined MGPCG parameters",
      size: 64,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.partials = device.createBuffer({
      label: "Pipelined MGPCG hierarchical merged-reduction partials",
      size: this.plan.reductionPartialBytes,
      usage: GPUBufferUsage.STORAGE,
    });
    this.outerDispatch = device.createBuffer({
      label: "Pipelined MGPCG GPU-zeroed outer tail",
      size: this.plan.indirectBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_SRC,
    });
    if (octreePipelinedMGPCGSymmetryStageAuditEnabled()) {
      const auditBuffer = (label: string) => device.createBuffer({
        label, size: this.plan.rowCapacity * 4,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      });
      this.symmetryStageAudit = {
        initialResidual: auditBuffer("MGPCG symmetry audit · initial residual"),
        initialPreconditioned: auditBuffer("MGPCG symmetry audit · first M r"),
        initialPreconditionedImage: auditBuffer("MGPCG symmetry audit · first A M r"),
        preconditionerPreSmoothed: auditBuffer("MGPCG symmetry audit · first p1"),
        preconditionerZeroSmoothed: auditBuffer("MGPCG symmetry audit · first zero sweep"),
        preconditionerFirstOperatorImage: auditBuffer("MGPCG symmetry audit · first L2 image"),
        preconditionerFirstSmoothed: auditBuffer("MGPCG symmetry audit · first nonzero sweep"),
        preconditionerInnerResidual: auditBuffer("MGPCG symmetry audit · first r1"),
        preconditionerInnerCorrection: auditBuffer("MGPCG symmetry audit · first M1 r1"),
        preconditionerPostCorrected: auditBuffer("MGPCG symmetry audit · first p1 + M1 r1"),
      };
    }
    const words = new Uint32Array(16), floats = new Float32Array(words.buffer);
    words[0] = this.plan.rowCapacity;
    words[1] = this.plan.maximumIterations;
    words[2] = this.plan.rowDispatch[0];
    words[3] = this.plan.reductionPartialCount;
    floats[4] = this.relativeTolerance;
    floats[5] = this.absoluteTolerance;
    floats[6] = 1e-30;
    device.queue.writeBuffer(this.params, 0, words);

    const activityProfile = performanceShaderVariant();
    const activity = createGPULogicalActivityAdoptionContext({
      moduleId: OCTREE_PIPELINED_MGPCG_ACTIVITY_MODULE_ID,
      profile: activityProfile,
      identity: "subgroup",
      subgroupsAlreadyEnabled: true,
    });
    this.activity = activity;
    for (const descriptor of Object.values(OCTREE_PIPELINED_MGPCG_ACTIVITY_TASKS)) {
      activity.describeTask(descriptor.task, {
        id: descriptor.id,
        label: descriptor.label,
        phaseId: "pressure-solve",
        checkpoints: descriptor.checkpoints,
      });
    }
    // Read once, at construction, because it selects a shader variant. Off is
    // today: the base source, the cache key and the label are all unchanged.
    const coldStart = options.coldStart ?? octreePipelinedMGPCGColdStartEnabled();
    this.startsFromPublishedSeed = !coldStart;
    const configuredTemporalMode = options.temporalPredictorMode
      ?? octreePipelinedMGPCGTemporalPredictorMode();
    const temporalMode = options.temporalPredictor === false
      ? "off"
      : options.temporalPredictor === true && configuredTemporalMode === "off"
        ? "fixed"
        : configuredTemporalMode;
    this.usesTemporalPredictor = temporalMode !== "off";
    this.usesCurrentOperatorTemporalPredictor = temporalMode === "current-operator";
    this.temporalPredictorAlpha = options.temporalPredictorAlpha
      ?? octreePipelinedMGPCGTemporalPredictorAlpha();
    if (!Number.isFinite(this.temporalPredictorAlpha)
      || this.temporalPredictorAlpha < -1 || this.temporalPredictorAlpha > 2) {
      throw new RangeError("Temporal pressure predictor alpha must remain in [-1,2]");
    }
    if (coldStart && this.usesTemporalPredictor) {
      throw new Error("Temporal pressure prediction cannot be combined with the cold-start arm");
    }
    const seededShader = octreePipelinedMGPCGSeedVariantShader(coldStart);
    const baseShader = temporalMode === "fixed"
      ? octreePipelinedMGPCGFixedTemporalSeedShader(seededShader, this.temporalPredictorAlpha)
      : seededShader;
    const shaderVariant = activity.module(
      octreePipelinedMGPCGActivityShader(
        activity, baseShader,
      ),
      `${OCTREE_PIPELINED_MGPCG_ACTIVITY_MODULE_ID}/${activityProfile.cacheKey}`
        + (coldStart ? "/cold-start" : "")
        + (temporalMode === "fixed" ? `/temporal-fixed-${this.temporalPredictorAlpha}` : ""),
    );
    this.shaderCode = shaderVariant.code;
    this.shaderLabel = "Octree pipelined MGPCG · row-parallel exact reductions"
      + (coldStart ? " · cold start (measurement)" : "")
      + (temporalMode === "fixed" ? ` · temporal α=${this.temporalPredictorAlpha}` : "");
    this.allocatedBytes = this.plan.ownedBytes;
  }

  async initializePipelines(): Promise<void> {
    if (this.pipelineInitialization) return this.pipelineInitialization;
    this.pipelineInitialization = (async () => {
      const compiler = gpuCompilationManagerFor(this.device);
      const shaderModule = compiler.createShaderModule({
        label: this.shaderLabel, code: this.shaderCode,
      });
      const entries = Object.keys(PIPELINE_BINDINGS) as PipelineName[];
      this.pipelines = Object.freeze(Object.fromEntries(await Promise.all(
        entries.map(async (entryPoint) => [entryPoint,
          this.activity.registerPipeline(await compiler.compileComputePipeline({
            label: `Pipelined MGPCG · ${entryPoint} · exact fixed-point reductions`,
            layout: "auto", compute: { module: shaderModule, entryPoint },
          }))] as const),
      )) as Record<PipelineName, GPUComputePipeline>);
      if (this.usesCurrentOperatorTemporalPredictor) {
        const temporalModule = compiler.createShaderModule({
          label: "Octree temporal pressure predictor · rank-one secant",
          code: octreeTemporalPressurePredictorShader,
        });
        const temporalEntries: TemporalPredictorPipelineName[] =
          Object.keys(TEMPORAL_PREDICTOR_BINDINGS) as TemporalPredictorPipelineName[];
        this.temporalPipelines = Object.freeze(Object.fromEntries(await Promise.all(
          temporalEntries.map(async (entryPoint) => [entryPoint,
            await compiler.compileComputePipeline({
              label: `Temporal pressure predictor · ${entryPoint}`,
              layout: "auto", compute: { module: temporalModule, entryPoint },
            })] as const),
        )) as Record<TemporalPredictorPipelineName, GPUComputePipeline>);
      }
    })();
    return this.pipelineInitialization;
  }

  get encodedDispatchCount(): number {
    return this.encodedDispatchCountFor(this.plan.maximumIterations);
  }

  encodedDispatchCountFor(encodedIterationBudget: number): number {
    if (!Number.isSafeInteger(encodedIterationBudget)
      || encodedIterationBudget < 4
      || encodedIterationBudget > this.plan.maximumIterations) {
      throw new RangeError(
        "Pipelined MGPCG encoded iteration budget must remain inside the planned envelope",
      );
    }
    const setup = this.source.preconditioner.encodedSetupDispatchCount ?? 0;
    const correction = this.source.preconditioner.encodedCorrectionDispatchCount;
    const apply = this.source.operator.encodedDispatchCount;
    // Control/authority/state/residual, setup, initial M, A(Mr), reduction
    // pair, and directions; then advance + M + residual reduction/finish +
    // direction + A(direction) + direct-curvature reduction/finish,
    // and one fail-closed publication.
    const temporalPrelude = this.usesCurrentOperatorTemporalPredictor ? apply + 3 : 0;
    const outerDispatches = this.usesCombinedReductionDrains ? 4 : 6;
    return 8 + 2 * apply + setup + correction + temporalPrelude
      + encodedIterationBudget * (outerDispatches + apply + correction);
  }

  encode(broker: PassBroker, solve: OctreePipelinedMGPCGSolve): void {
    this.assertLive();
    const encodedIterationBudget =
      solve.encodedIterationBudget ?? this.plan.maximumIterations;
    // Keep the GPU exhaustion gate identical to the graph actually encoded.
    // If a prediction ever lacks enough headroom, `finishMergedTotal` marks
    // the solve fatal and final publication retains pressureSeed.
    if (!Number.isSafeInteger(encodedIterationBudget)
      || encodedIterationBudget < 4
      || encodedIterationBudget > this.plan.maximumIterations) {
      throw new RangeError(
        "Pipelined MGPCG encoded iteration budget must remain inside the planned envelope",
      );
    }
    if (encodedIterationBudget !== this.parameterIterationBudget) {
      this.device.queue.writeBuffer(
        this.params, Uint32Array.BYTES_PER_ELEMENT,
        new Uint32Array([encodedIterationBudget]),
      );
      this.parameterIterationBudget = encodedIterationBudget;
    }
    const vectorBytes = this.plan.rowCapacity * 4;
    if (solve.pressureSeed.size < vectorBytes || solve.pressureOut.size < vectorBytes) {
      throw new RangeError("Pipelined MGPCG solve buffers are smaller than the planned capacity");
    }
    if (this.usesTemporalPredictor && (!solve.pressureHistory
      || solve.pressureHistory.size < vectorBytes)) {
      throw new RangeError("Temporal pressure predictor history is smaller than the planned capacity");
    }
    const resources = this.resources(solve);
    broker.clearBuffer(this.control);
    let pass = broker.compute({ label: "Pipelined MGPCG initialization and authority gate" });
    this.runDirect(pass, "initializeControlAndDispatch", [1, 1, 1], resources);
    this.runDirect(pass, "validateAuthority", [1, 1, 1], resources);
    // The singleton has now authored exact row and live-partial records. End
    // the STORAGE usage scope before any record is consumed as INDIRECT.
    broker.fence("pipelined MGPCG live dispatch publication");
    pass = broker.compute({ label: "Pipelined MGPCG state initialization" });
    this.runOuterIndirect(pass, "initializeState", 0, 0, resources);
    if (this.usesCurrentOperatorTemporalPredictor) {
      pass = broker.compute({ label: "Temporal pressure predictor direction" });
      this.runTemporalOuterIndirect(pass, "initializeTemporalDirection", 0, 0, resources);
      this.source.operator.encode(
        broker, this.source.vectors.pressure, this.source.vectors.preconditionedImage,
        this.control,
      );
      this.source.operator.encode(
        broker, this.source.vectors.direction, this.source.vectors.directionImage,
        this.control,
      );
      pass = broker.compute({ label: "Temporal pressure predictor reduction" });
      this.runTemporalOuterIndirect(pass, "reduceTemporalPartials", 0, 2, resources);
      broker.fence("temporal pressure predictor partials complete");
      pass = broker.compute({ label: "Temporal pressure predictor reduction finish" });
      this.runTemporalDirect(pass, "finishTemporalReduction", [1, 1, 1], resources);
      broker.fence("temporal pressure predictor coefficient publication");
      pass = broker.compute({ label: "Temporal pressure predictor apply" });
      this.runTemporalOuterIndirect(pass, "applyTemporalPrediction", 0, 0, resources);
    } else {
      this.source.operator.encode(
        broker, this.source.vectors.pressure, this.source.vectors.directionImage,
        this.control,
      );
      pass = broker.compute({ label: "Pipelined MGPCG initial residual" });
      this.runOuterIndirect(pass, "formInitialResidual", 0, 0, resources);
    }

    const symmetryAudit = this.symmetryStageAudit;
    if (symmetryAudit) broker.copyBufferToBuffer(
      this.source.vectors.residual, 0, symmetryAudit.initialResidual, 0, vectorBytes);

    const preconditioner = this.source.preconditioner;
    preconditioner.encodeSetup(broker, {
      solverControl: this.control,
      rowCount: this.source.rowCount,
    });
    preconditioner.encodeCorrection(broker, {
      rhs: this.source.vectors.residual,
      correction: this.source.vectors.preconditioned,
      solverControl: this.control,
      rowCount: this.source.rowCount,
    });
    if (symmetryAudit) broker.copyBufferToBuffer(
      this.source.vectors.preconditioned, 0, symmetryAudit.initialPreconditioned, 0, vectorBytes);
    const preconditionerAudit = symmetryAudit
      ? (preconditioner as OctreePipelinedFixedPreconditioner & { symmetryStageAuditBuffers?: {
        preSmoothed: GPUBuffer; zeroSmoothed: GPUBuffer;
        firstOperatorImage: GPUBuffer; firstSmoothed: GPUBuffer; innerResidual: GPUBuffer;
        innerCorrection: GPUBuffer; postCorrected: GPUBuffer;
      } }).symmetryStageAuditBuffers : undefined;
    if (preconditionerAudit) {
      broker.copyBufferToBuffer(preconditionerAudit.preSmoothed, 0,
        symmetryAudit!.preconditionerPreSmoothed, 0, vectorBytes);
      broker.copyBufferToBuffer(preconditionerAudit.zeroSmoothed, 0,
        symmetryAudit!.preconditionerZeroSmoothed, 0, vectorBytes);
      broker.copyBufferToBuffer(preconditionerAudit.firstOperatorImage, 0,
        symmetryAudit!.preconditionerFirstOperatorImage, 0, vectorBytes);
      broker.copyBufferToBuffer(preconditionerAudit.firstSmoothed, 0,
        symmetryAudit!.preconditionerFirstSmoothed, 0, vectorBytes);
      broker.copyBufferToBuffer(preconditionerAudit.innerResidual, 0,
        symmetryAudit!.preconditionerInnerResidual, 0, vectorBytes);
      broker.copyBufferToBuffer(preconditionerAudit.innerCorrection, 0,
        symmetryAudit!.preconditionerInnerCorrection, 0, vectorBytes);
      broker.copyBufferToBuffer(preconditionerAudit.postCorrected, 0,
        symmetryAudit!.preconditionerPostCorrected, 0, vectorBytes);
    }
    this.source.operator.encode(
      broker, this.source.vectors.preconditioned, this.source.vectors.preconditionedImage,
      this.control,
    );
    if (symmetryAudit) broker.copyBufferToBuffer(
      this.source.vectors.preconditionedImage, 0, symmetryAudit.initialPreconditionedImage, 0, vectorBytes);
    pass = broker.compute({ label: "Pipelined MGPCG initial merged reduction" });
    this.runOuterIndirect(pass, "reduceMergedPartials", 0, 2, resources);
    broker.fence("pipelined MGPCG initial merged partials complete");
    pass = broker.compute({ label: "Pipelined MGPCG initial merged reduction finish" });
    this.runDirect(pass, "finishMergedReduction", [1, 1, 1], resources);
    // finishMergedReduction writes the indirect tail. A new pass prevents the
    // dispatch buffer from being STORAGE and INDIRECT in one usage scope, and
    // lets an initially converged or failed solve zero the direction copy.
    broker.fence("pipelined MGPCG initial indirect-tail publication");
    pass = broker.compute({ label: "Pipelined MGPCG direction initialization" });
    this.runOuterIndirect(pass, "initializeDirections", 0, 3, resources);

    for (let iteration = 0; iteration < encodedIterationBudget; iteration += 1) {
      pass = broker.compute({ label: `Pipelined MGPCG outer iteration ${iteration}` });
      this.runIndirect(pass, "advancePCGState", iteration, 0, resources);
      preconditioner.encodeCorrection(broker, {
        rhs: this.source.vectors.residual,
        correction: this.source.vectors.preconditioned,
        solverControl: this.control,
        rowCount: this.source.rowCount,
      });
      if (this.usesCombinedReductionDrains) {
        pass = broker.compute({
          label: "Pipelined MGPCG combined merged reduction drain",
        });
        this.runDirect(pass, "reduceAndFinishMerged", [1, 1, 1], resources);
      } else {
        pass = broker.compute({ label: "Pipelined MGPCG merged partial reduction" });
        this.runIndirect(pass, "reduceMergedPartials", iteration, 2, resources);
        // The finishing workgroup reads every partial and mutates future
        // indirect records, so it owns the single global reduction boundary.
        broker.fence("pipelined MGPCG merged partials complete");
        pass = broker.compute({ label: "Pipelined MGPCG merged reduction finish" });
        this.runDirect(pass, "finishMergedReduction", [1, 1, 1], resources);
      }
      broker.fence("pipelined MGPCG convergence-tail publication");
      pass = broker.compute({ label: "Pipelined MGPCG direction update" });
      this.runIndirect(pass, "updateDirections", iteration, 3, resources);
      this.source.operator.encode(
        broker, this.source.vectors.direction, this.source.vectors.directionImage,
        this.control,
      );
      if (this.usesCombinedReductionDrains) {
        pass = broker.compute({
          label: "Pipelined MGPCG combined direct-curvature drain",
        });
        this.runDirect(
          pass, "reduceAndFinishDirectionCurvature", [1, 1, 1], resources,
        );
      } else {
        pass = broker.compute({ label: "Pipelined MGPCG direct direction curvature" });
        this.runIndirect(
          pass, "reduceDirectionCurvaturePartials", iteration, 1, resources,
        );
        broker.fence("pipelined MGPCG direct curvature partials complete");
        pass = broker.compute({ label: "Pipelined MGPCG direct curvature finish" });
        this.runDirect(pass, "finishDirectionCurvature", [1, 1, 1], resources);
      }
      broker.fence("pipelined MGPCG direct curvature publication");
    }
    pass = broker.compute({ label: "Pipelined MGPCG fail-closed publication" });
    this.bind(pass, "finalizeAndPublish", resources);
    pass.dispatchWorkgroupsIndirect(
      this.source.rowDispatch,
      this.source.rowDispatchOffsetBytes ?? 0,
    );
    broker.fence("pipelined MGPCG pressure publication");
  }

  private resources(
    solve: OctreePipelinedMGPCGSolve,
  ): readonly (GPUBuffer | undefined)[] {
    const vectors = this.source.vectors;
    return [
      this.params,
      this.source.coefficients,
      this.source.acceptedAuthority,
      this.source.rowCount,
      undefined,
      solve.pressureSeed,
      solve.pressureOut,
      vectors.pressure,
      vectors.residual,
      vectors.preconditioned,
      vectors.preconditionedImage,
      vectors.direction,
      vectors.directionImage,
      this.control,
      this.partials,
      this.outerDispatch,
      this.source.rhs,
      solve.pressureHistory,
    ];
  }

  private bindTemporal(
    pass: GPUComputePassEncoder,
    name: TemporalPredictorPipelineName,
    resources: readonly (GPUBuffer | undefined)[],
  ): void {
    const pipeline = this.temporalPipelines?.[name];
    if (!pipeline) throw new Error("Temporal pressure predictor pipeline is unavailable");
    const bindings = TEMPORAL_PREDICTOR_BINDINGS[name];
    const variants = this.temporalGroups.get(name);
    const cached = variants?.find((candidate) =>
      bindings.every((binding) => candidate.resources[binding] === resources[binding]));
    const group = cached?.group ?? this.device.createBindGroup({
      label: `Temporal pressure predictor · ${name}`,
      layout: pipeline.getBindGroupLayout(0),
      entries: bindings.map((binding) => ({
        binding,
        resource: { buffer: resources[binding]! },
      })),
    });
    if (!cached) {
      const next = { resources: [...resources], group };
      if (variants) variants.push(next); else this.temporalGroups.set(name, [next]);
    }
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, group);
  }

  private runTemporalDirect(
    pass: GPUComputePassEncoder,
    name: TemporalPredictorPipelineName,
    dispatch: readonly [number, number, number],
    resources: readonly (GPUBuffer | undefined)[],
  ): void {
    this.bindTemporal(pass, name, resources);
    pass.dispatchWorkgroups(...dispatch);
  }

  private runTemporalOuterIndirect(
    pass: GPUComputePassEncoder,
    name: TemporalPredictorPipelineName,
    iteration: number,
    stage: 0 | 1 | 2 | 3,
    resources: readonly (GPUBuffer | undefined)[],
  ): void {
    this.bindTemporal(pass, name, resources);
    const record = iteration * OCTREE_PIPELINED_PCG_DISPATCHES_PER_ITERATION + stage;
    pass.dispatchWorkgroupsIndirect(
      this.outerDispatch,
      record * OCTREE_PIPELINED_PCG_INDIRECT_STRIDE_BYTES,
    );
  }

  private bind(
    pass: GPUComputePassEncoder,
    name: PipelineName,
    resources: readonly (GPUBuffer | undefined)[],
  ): void {
    const pipeline = this.pipelines[name];
    const baseBindings = PIPELINE_BINDINGS[name];
    const bindings = this.usesTemporalPredictor
      && !this.usesCurrentOperatorTemporalPredictor && name === "initializeState"
      ? [...baseBindings, 17]
      : baseBindings;
    const variants = this.groups.get(name);
    const cached = variants?.find((candidate) =>
      bindings.every((binding) => candidate.resources[binding] === resources[binding]));
    const group = cached?.group ?? this.device.createBindGroup({
      label: `Pipelined MGPCG · ${name}`,
      layout: pipeline.getBindGroupLayout(0),
      entries: bindings.map((binding) => ({
        binding,
        resource: { buffer: resources[binding]! },
      })),
    });
    if (!cached) {
      const next = { resources: [...resources], group };
      if (variants) variants.push(next); else this.groups.set(name, [next]);
    }
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, group);
  }

  private runDirect(
    pass: GPUComputePassEncoder,
    name: PipelineName,
    dispatch: readonly [number, number, number],
    resources: readonly (GPUBuffer | undefined)[],
  ): void {
    this.bind(pass, name, resources);
    pass.dispatchWorkgroups(...dispatch);
  }

  private runIndirect(
    pass: GPUComputePassEncoder,
    name: PipelineName,
    iteration: number,
    stage: 0 | 1 | 2 | 3,
    resources: readonly (GPUBuffer | undefined)[],
  ): void {
    this.runOuterIndirect(pass, name, iteration, stage, resources);
  }

  private runOuterIndirect(
    pass: GPUComputePassEncoder,
    name: PipelineName,
    iteration: number,
    stage: 0 | 1 | 2 | 3,
    resources: readonly (GPUBuffer | undefined)[],
  ): void {
    this.bind(pass, name, resources);
    const record = iteration * OCTREE_PIPELINED_PCG_DISPATCHES_PER_ITERATION + stage;
    pass.dispatchWorkgroupsIndirect(
      this.outerDispatch,
      record * OCTREE_PIPELINED_PCG_INDIRECT_STRIDE_BYTES,
    );
  }

  private assertLive(): void {
    if (this.destroyed) throw new Error("Pipelined MGPCG primitive is destroyed");
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.groups.clear();
    this.control.destroy();
    this.params.destroy();
    this.partials.destroy();
    this.outerDispatch.destroy();
    for (const buffer of Object.values(this.symmetryStageAudit ?? {})) buffer.destroy();
  }
}

export const octreeCompensatedF32WGSL = /* wgsl */ `
struct CompensatedF32 { hi: f32, lo: f32 }
fn twoSumF32(a: f32, b: f32) -> CompensatedF32 {
  let hi = a + b;
  let bVirtual = hi - a;
  let lo = (a - (hi - bVirtual)) + (b - bVirtual);
  return CompensatedF32(hi, lo);
}
fn addCompensatedF32(a: CompensatedF32, value: f32) -> CompensatedF32 {
  let first = twoSumF32(a.hi, value);
  let second = twoSumF32(first.lo, a.lo);
  return twoSumF32(first.hi, second.hi + second.lo);
}
fn mergeCompensatedF32(a: CompensatedF32, b: CompensatedF32) -> CompensatedF32 {
  return addCompensatedF32(addCompensatedF32(a, b.hi), b.lo);
}
fn compensatedValue(a: CompensatedF32) -> f32 { return a.hi + a.lo; }
`;

/**
 * The four merged solve scalars, folded over the workgroup by an explicit
 * fixed-shape tree.
 *
 * This deliberately uses NO subgroup reduction, for two independent reasons.
 *
 * 1. Determinism. `subgroupAdd`'s association is implementation-defined, so the
 *    value it returns is not a property of this shader. That made the whole
 *    500-step lane non-reproducible: four captures of one binary disagreed in
 *    the compact-field readback, first at generation 227, in exactly the
 *    `mgpcgControl` residual words this fold publishes -- while two DIFFERENT
 *    binaries could agree exactly. A bit-exact diff against a baseline is only
 *    evidence if the baseline reproduces itself, so every correctness proof on
 *    this branch depended on pinning this.
 *
 * 2. Correctness, independently of determinism. The subgroup form was
 *      CompensatedF32(subgroupAdd(local.gamma.hi), subgroupAdd(local.gamma.lo))
 *    which is NOT a compensated sum. `hi` and `lo` are not independent
 *    accumulators: `lo` is the rounding error of `hi`, an invariant that
 *    `twoSumF32` re-establishes on every merge. Adding the `hi` parts with a
 *    plain f32 reduction throws away every rounding error generated inside that
 *    reduction, and pairing the result with a separately-summed `lo` yields a
 *    pair whose `lo` is not the error term of its `hi`. So the highest-fan-in
 *    stage of the reduction -- 128 lanes collapsing to one partial per subgroup
 *    -- was the one place the compensation was silently dropped, and it was
 *    dropped in favour of roughly plain f32 summation. `mergeScalars` below is
 *    the renormalising merge, and it is now used at every level.
 *
 * The shape is fixed at compile time: REDUCTION_LANES is a constant, so this is
 * always the same seven levels pairing the same lanes, independent of
 * `subgroup_size` and of which lanes happen to be active. `enable subgroups;`
 * and the feature gate stay because the GPU logical-activity variant still
 * emits `subgroupBallot` through those builtins.
 *
 * An earlier pinning attempt was reverted on 2026-07-27 for "reddening" the
 * Gate B values. That justification does not survive this finding: the values
 * it was compared against were not reproducible, and any pinned tree MUST
 * change the association and therefore the values. Movement here is expected
 * and is not evidence of regression.
 */
const targetSubgroupReductionWGSL = /* wgsl */ `
  merged[lane] = local;
  for (var width = REDUCTION_LANES / 2u; width > 0u; width >>= 1u) {
    workgroupBarrier();
    if (lane < width) {
      merged[lane] = mergeScalars(merged[lane], merged[lane + width]);
    }
  }
  workgroupBarrier();`;

/**
 * One-dimensional current-operator projection of the published warm seed onto
 * the temporal secant `p[n-1] - p[n-2]`. It deliberately ends before PCG is
 * initialized: no Krylov direction or beta is carried between solves.
 */
export const octreeTemporalPressurePredictorShader = /* wgsl */ `
struct Params {
  shape: vec4u,
  numerics: vec4f,
  padding0: vec4u,
  padding1: vec4u,
}
struct TemporalScalars {
  numerator: f32,
  curvature: f32,
  directionSquared: f32,
  padding: f32,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(2) var<storage, read> acceptedAuthority: array<u32>;
@group(0) @binding(5) var<storage, read> pressureSeed: array<f32>;
@group(0) @binding(7) var<storage, read_write> pressure: array<f32>;
@group(0) @binding(8) var<storage, read_write> residual: array<f32>;
@group(0) @binding(10) var<storage, read> seedImage: array<f32>;
@group(0) @binding(11) var<storage, read_write> direction: array<f32>;
@group(0) @binding(12) var<storage, read> directionImage: array<f32>;
@group(0) @binding(13) var<storage, read_write> control: array<atomic<u32>>;
@group(0) @binding(14) var<storage, read_write> partials: array<TemporalScalars>;
@group(0) @binding(16) var<storage, read> rhs: array<f32>;
@group(0) @binding(17) var<storage, read> pressureHistory: array<f32>;
const INVALID: u32 = 0xffffffffu;
const ERROR_NONFINITE: u32 = ${OCTREE_PIPELINED_PCG_ERROR.nonFinite}u;
const REDUCTION_LANES: u32 = ${OCTREE_PIPELINED_PCG_WORKGROUP_SIZE}u;
fn rows() -> u32 { return min(acceptedAuthority[2], params.shape.x); }
fn failed() -> bool { return atomicLoad(&control[0]) != 0u; }
fn finite(value: f32) -> bool { return value == value && abs(value) <= 3.402823e38; }
fn reportNonFinite(row: u32) {
  atomicOr(&control[0], ERROR_NONFINITE);
  atomicMin(&control[7], row);
  atomicCompareExchangeWeak(&control[6], 0u, 16u);
}

@compute @workgroup_size(64)
fn initializeTemporalDirection(@builtin(global_invocation_id) global: vec3u) {
  let row = global.x;
  if (row >= rows() || failed()) { return; }
  let history = pressureHistory[row];
  direction[row] = select(0.0, history, finite(history));
}

var<workgroup> temporal: array<TemporalScalars, ${OCTREE_PIPELINED_PCG_WORKGROUP_SIZE}>;

@compute @workgroup_size(${OCTREE_PIPELINED_PCG_WORKGROUP_SIZE})
fn reduceTemporalPartials(
  @builtin(local_invocation_index) lane: u32,
  @builtin(workgroup_id) workgroup: vec3u,
  @builtin(global_invocation_id) global: vec3u,
) {
  var local = TemporalScalars(0.0, 0.0, 0.0, 0.0);
  let row = global.x;
  if (row < rows() && !failed()) {
    let d = direction[row];
    let ad = directionImage[row];
    let r = -rhs[row] - seedImage[row];
    if (!finite(d) || !finite(ad) || !finite(r)) {
      reportNonFinite(row);
    } else {
      local.numerator = d * r;
      local.curvature = d * ad;
      local.directionSquared = d * d;
    }
  }
  temporal[lane] = local;
  for (var width = REDUCTION_LANES / 2u; width > 0u; width >>= 1u) {
    workgroupBarrier();
    if (lane < width) {
      temporal[lane].numerator += temporal[lane + width].numerator;
      temporal[lane].curvature += temporal[lane + width].curvature;
      temporal[lane].directionSquared += temporal[lane + width].directionSquared;
    }
  }
  workgroupBarrier();
  if (lane == 0u) { partials[workgroup.x] = temporal[0]; }
}

@compute @workgroup_size(${OCTREE_PIPELINED_PCG_WORKGROUP_SIZE})
fn finishTemporalReduction(@builtin(local_invocation_index) lane: u32) {
  var local = TemporalScalars(0.0, 0.0, 0.0, 0.0);
  let count = (rows() + REDUCTION_LANES - 1u) / REDUCTION_LANES;
  for (var partial = lane; partial < count; partial += REDUCTION_LANES) {
    local.numerator += partials[partial].numerator;
    local.curvature += partials[partial].curvature;
    local.directionSquared += partials[partial].directionSquared;
  }
  temporal[lane] = local;
  for (var width = REDUCTION_LANES / 2u; width > 0u; width >>= 1u) {
    workgroupBarrier();
    if (lane < width) {
      temporal[lane].numerator += temporal[lane + width].numerator;
      temporal[lane].curvature += temporal[lane + width].curvature;
      temporal[lane].directionSquared += temporal[lane + width].directionSquared;
    }
  }
  workgroupBarrier();
  if (lane != 0u) { return; }
  let total = temporal[0];
  atomicStore(&control[23], bitcast<u32>(total.numerator));
  atomicStore(&control[24], bitcast<u32>(total.curvature));
  atomicStore(&control[25], bitcast<u32>(total.directionSquared));
  var alpha = 0.0;
  let scaleFloor = max(1e-30, total.directionSquared * 1e-12);
  if (!failed() && finite(total.numerator) && finite(total.curvature)
    && finite(total.directionSquared) && total.curvature > scaleFloor) {
    let candidate = total.numerator / total.curvature;
    if (finite(candidate)) { alpha = clamp(candidate, -1.0, 2.0); }
  }
  atomicStore(&control[21], bitcast<u32>(alpha));
  atomicStore(&control[22], select(0u, 1u, alpha != 0.0));
}

@compute @workgroup_size(64)
fn applyTemporalPrediction(@builtin(global_invocation_id) global: vec3u) {
  let row = global.x;
  if (row >= rows() || failed()) { return; }
  let alpha = bitcast<f32>(atomicLoad(&control[21]));
  let baseResidual = -rhs[row] - seedImage[row];
  let predicted = pressureSeed[row] + alpha * direction[row];
  let predictedResidual = baseResidual - alpha * directionImage[row];
  if (!finite(predicted) || !finite(predictedResidual)) {
    pressure[row] = pressureSeed[row];
    residual[row] = baseResidual;
  } else {
    pressure[row] = predicted;
    residual[row] = predictedResidual;
  }
}
`;

export const octreePipelinedMGPCGShader = /* wgsl */ `
enable subgroups;
${octreeCompensatedF32WGSL}
struct Params {
  shape: vec4u,
  numerics: vec4f,
  padding0: vec4u,
  padding1: vec4u,
}
struct MergedScalars {
  gamma: CompensatedF32,
  delta: CompensatedF32,
  rr: CompensatedF32,
  bb: CompensatedF32,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> section63Coefficients: array<f32>;
@group(0) @binding(2) var<storage, read> acceptedAuthority: array<u32>;
@group(0) @binding(3) var<storage, read> rowCount: array<u32>;
@group(0) @binding(5) var<storage, read> pressureSeed: array<f32>;
@group(0) @binding(6) var<storage, read_write> pressureOut: array<f32>;
@group(0) @binding(7) var<storage, read_write> pressure: array<f32>;
@group(0) @binding(8) var<storage, read_write> residual: array<f32>;
@group(0) @binding(9) var<storage, read_write> preconditioned: array<f32>;
@group(0) @binding(10) var<storage, read_write> preconditionedImage: array<f32>;
@group(0) @binding(11) var<storage, read_write> direction: array<f32>;
@group(0) @binding(12) var<storage, read_write> directionImage: array<f32>;
@group(0) @binding(13) var<storage, read_write> control: array<atomic<u32>>;
// One exact signed radix-256 superaccumulator per reduction workgroup.  The
// integer limbs make the result independent of workgroup scheduling, row
// partitioning, and D4's row permutation; no floating-point fold crosses rows.
@group(0) @binding(14) var<storage, read_write> partials: array<atomic<i32>>;
@group(0) @binding(15) var<storage, read_write> outerDispatch: array<u32>;
@group(0) @binding(16) var<storage, read> rhs: array<f32>;

const ERROR_INVALID_AUTHORITY = 1u;
const ERROR_INVALID_ROW = 2u;
const ERROR_NONFINITE = 4u;
const ERROR_NONPOSITIVE_PRECONDITIONER = 8u;
const ERROR_NONPOSITIVE_CURVATURE = 16u;
const ERROR_NONCONVERGENCE = 32u;
const INVALID = 0xffffffffu;
const DISPATCHES_PER_ITERATION = 4u;
const DISPATCH_STRIDE_WORDS = 4u;
const REDUCTION_LANES = ${OCTREE_PIPELINED_PCG_WORKGROUP_SIZE}u;
const FIXED_LIMBS = ${OCTREE_PIPELINED_PCG_FIXED_POINT_LIMBS}u;
const FIXED_SCALARS = ${OCTREE_PIPELINED_PCG_FIXED_POINT_SCALARS}u;
const FIXED_WORDS_PER_PARTIAL = FIXED_LIMBS * FIXED_SCALARS;
const FIXED_MIN_EXPONENT = -152;

fn finite(value: f32) -> bool {
  return value == value && abs(value) <= 3.402823e38;
}
fn capacity() -> u32 { return params.shape.x; }
fn maximumIterations() -> u32 { return params.shape.y; }
fn rowGroups() -> u32 { return params.shape.z; }
fn partialCount() -> u32 { return params.shape.w; }
fn livePartialCount() -> u32 {
  return (rows() + REDUCTION_LANES - 1u) / REDUCTION_LANES;
}
fn rows() -> u32 {
  return min(select(0u, acceptedAuthority[2], arrayLength(&acceptedAuthority) >= 6u), capacity());
}
fn acceptedBank() -> u32 { return acceptedAuthority[5]; }
fn diagonalAt(row: u32) -> f32 {
  return section63Coefficients[(acceptedBank() * capacity() + row) * 19u];
}
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
fn dispatchBase(iteration: u32, stage: u32) -> u32 {
  return (iteration * DISPATCHES_PER_ITERATION + stage) * DISPATCH_STRIDE_WORDS;
}
fn zeroDispatch(iteration: u32, stage: u32) {
  let base = dispatchBase(iteration, stage);
  outerDispatch[base] = 0u;
  outerDispatch[base + 1u] = 0u;
  outerDispatch[base + 2u] = 0u;
  outerDispatch[base + 3u] = 0u;
  atomicAdd(&control[5], 1u);
}
fn zeroAllOuterDispatches() {
  for (var iteration = 0u; iteration < maximumIterations(); iteration += 1u) {
    zeroDispatch(iteration, 0u);
    zeroDispatch(iteration, 1u);
    zeroDispatch(iteration, 2u);
    zeroDispatch(iteration, 3u);
  }
}
fn zeroRemainingAfterUpdate(iteration: u32) {
  zeroDispatch(iteration, 1u);
  zeroDispatch(iteration, 3u);
  for (var future = iteration + 1u; future < maximumIterations(); future += 1u) {
    zeroDispatch(future, 0u);
    zeroDispatch(future, 1u);
    zeroDispatch(future, 2u);
    zeroDispatch(future, 3u);
  }
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

fn fixedAt(partial: u32, scalar: u32, limb: u32) -> u32 {
  return partial * FIXED_WORDS_PER_PARTIAL + scalar * FIXED_LIMBS + limb;
}

fn clearFixedPartial(partial: u32, lane: u32) {
  for (var word = lane; word < FIXED_WORDS_PER_PARTIAL; word += REDUCTION_LANES) {
    atomicStore(&partials[partial * FIXED_WORDS_PER_PARTIAL + word], 0);
  }
  workgroupBarrier();
}

// Deposit the exact finite f32 bit pattern into four signed radix-256 limbs.
// Normal values have value=M*2^(rawExponent-150); subnormals have M*2^-149.
// FIXED_MIN_EXPONENT leaves a non-negative shift for both forms.  Splitting M
// after that shift into byte digits keeps every atomic add far below i32 range:
// 255 * (64*65535 rows) < 2^31.
fn addFixedF32(partial: u32, scalar: u32, value: f32) {
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
    if (byte != 0 && limb < FIXED_LIMBS) {
      atomicAdd(&partials[fixedAt(partial, scalar, limb)], sign * byte);
    }
  }
}

fn floorDiv256(value: i32) -> vec2i {
  var carry = value / 256;
  var digit = value - carry * 256;
  if (digit < 0) { digit += 256; carry -= 1; }
  return vec2i(carry, digit);
}

// The singleton finish performs integer-only merging.  Its order is
// deliberately irrelevant: integer addition is exact, associative and
// commutative.  Only after canonical carry propagation do we round once back
// onto the f32 recurrence surface.
fn fixedScalarValue(scalar: u32) -> f32 {
  var limbs: array<i32, ${OCTREE_PIPELINED_PCG_FIXED_POINT_LIMBS}>;
  for (var partial = 0u; partial < livePartialCount(); partial += 1u) {
    for (var limb = 0u; limb < FIXED_LIMBS; limb += 1u) {
      limbs[limb] += atomicLoad(&partials[fixedAt(partial, scalar, limb)]);
    }
  }
  for (var limb = 0u; limb + 1u < FIXED_LIMBS; limb += 1u) {
    let normalized = floorDiv256(limbs[limb]);
    limbs[limb] = normalized.y;
    limbs[limb + 1u] += normalized.x;
  }

  // A signed canonical radix expansion keeps the sign in its top limb.  Do
  // not Horner-fold that expansion before applying FIXED_MIN_EXPONENT: the
  // unscaled integer can exceed f32 even when the represented physical value
  // is ordinary.  Convert negatives to a positive magnitude first, then add
  // already-scaled non-negative limbs.  Every intermediate is bounded by the
  // final magnitude, while the integer merge above remains exact and
  // partition-independent.
  let negative = limbs[FIXED_LIMBS - 1u] < 0;
  if (negative) {
    for (var limb = 0u; limb < FIXED_LIMBS; limb += 1u) {
      limbs[limb] = -limbs[limb];
    }
    for (var limb = 0u; limb + 1u < FIXED_LIMBS; limb += 1u) {
      let normalized = floorDiv256(limbs[limb]);
      limbs[limb] = normalized.y;
      limbs[limb + 1u] += normalized.x;
    }
  }
  var magnitude = 0.0;
  for (var limb = 0u; limb < FIXED_LIMBS; limb += 1u) {
    magnitude += ldexp(f32(limbs[limb]), FIXED_MIN_EXPONENT + i32(limb * 8u));
  }
  return select(magnitude, -magnitude, negative);
}

fn fixedMergedValue() -> MergedScalars {
  return MergedScalars(
    CompensatedF32(fixedScalarValue(0u), 0.0),
    CompensatedF32(fixedScalarValue(1u), 0.0),
    CompensatedF32(fixedScalarValue(2u), 0.0),
    CompensatedF32(fixedScalarValue(3u), 0.0),
  );
}

@compute @workgroup_size(1)
fn initializeControlAndDispatch() {
  atomicStore(&control[7], INVALID);
  let liveRowGroups = (rows() + 63u) / 64u;
  let livePartials = livePartialCount();
  for (var iteration = 0u; iteration < maximumIterations(); iteration += 1u) {
    for (var stage = 0u; stage < DISPATCHES_PER_ITERATION; stage += 1u) {
      let base = dispatchBase(iteration, stage);
      let reduction = stage == 1u || stage == 2u;
      let groups = select(liveRowGroups, livePartials, reduction);
      outerDispatch[base] = groups;
      outerDispatch[base + 1u] = 1u;
      outerDispatch[base + 2u] = 1u;
      outerDispatch[base + 3u] = 0u;
    }
  }
}

@compute @workgroup_size(1)
fn validateAuthority() {
  if (arrayLength(&acceptedAuthority) < 7u
    || acceptedAuthority[2] == 0u || acceptedAuthority[2] > capacity()) {
    reportAt(ERROR_INVALID_AUTHORITY, 1u, INVALID);
    return;
  }
  if (acceptedAuthority[0] != 0u
    || acceptedAuthority[1] != INVALID
    || acceptedAuthority[3] == 0u
    || acceptedAuthority[4] == 0u
    || acceptedAuthority[5] > 1u
    || acceptedAuthority[6] != acceptedAuthority[4]) {
    reportAt(ERROR_INVALID_AUTHORITY, 1u, INVALID);
  }
  atomicStore(&control[4], rows());
}

@compute @workgroup_size(64)
fn initializeState(@builtin(global_invocation_id) global: vec3u) {
  let row = global.x;
  if (row >= rows() || failed()) { return; }
  let seed = pressureSeed[row];
  let diagonal = diagonalAt(row);
  if (!finite(diagonal) || diagonal <= 0.0
    || !finite(rhs[row]) || !finite(seed)) {
    reportAt(ERROR_INVALID_ROW, 2u, row);
    return;
  }
  pressure[row] = seed;
  residual[row] = 0.0;
}

@compute @workgroup_size(64)
fn formInitialResidual(@builtin(global_invocation_id) global: vec3u) {
  let row = global.x;
  if (row >= rows() || failed()) { return; }
  let value = -rhs[row] - directionImage[row];
  if (!finite(value)) { reportAt(ERROR_NONFINITE, 3u, row); }
  else { residual[row] = value; }
}

var<workgroup> merged: array<MergedScalars, ${OCTREE_PIPELINED_PCG_WORKGROUP_SIZE}>;
var<workgroup> combinedBlocks: array<MergedScalars, 32>;

@compute @workgroup_size(${OCTREE_PIPELINED_PCG_WORKGROUP_SIZE})
fn reduceMergedPartials(
  @builtin(local_invocation_index) lane: u32,
  @builtin(workgroup_id) workgroup: vec3u,
  @builtin(global_invocation_id) global: vec3u,
  @builtin(subgroup_invocation_id) subgroupLane: u32,
  @builtin(subgroup_size) subgroupSize: u32,
) {
  clearFixedPartial(workgroup.x, lane);
  let row = global.x;
  let initial = atomicLoad(&control[3]) == 0u;
  if (row < rows() && !stopped()) {
    let r = residual[row];
    let u = preconditioned[row];
    let w = preconditionedImage[row];
    let b = -rhs[row];
    if (!finite(r) || !finite(u) || (initial && !finite(w))) {
      reportAt(ERROR_NONFINITE, 4u, row);
    } else {
      addFixedF32(workgroup.x, 0u, r * u);
      if (initial) { addFixedF32(workgroup.x, 1u, u * w); }
      addFixedF32(workgroup.x, 2u, r * r);
      if (initial) { addFixedF32(workgroup.x, 3u, b * b); }
    }
  }
}

fn finishMergedTotal(total: MergedScalars) {
  let initial = atomicLoad(&control[3]) == 0u;
  if (failed()) {
    if (initial) { zeroAllOuterDispatches(); }
    else { zeroRemainingAfterUpdate(atomicLoad(&control[2])); }
    return;
  }
  // A converged earlier reduction already zeroed every remaining indirect
  // record. The fixed singleton tail must not recount stale partials.
  if (atomicLoad(&control[1]) != 0u) { return; }
  let gamma = compensatedValue(total.gamma);
  let delta = compensatedValue(total.delta);
  let rr = compensatedValue(total.rr);
  if (!finite(gamma) || !finite(delta) || !finite(rr) || rr < 0.0) {
    reportAt(ERROR_NONFINITE, 5u, INVALID);
    if (initial) { zeroAllOuterDispatches(); }
    else { zeroRemainingAfterUpdate(atomicLoad(&control[2])); }
    return;
  }
  if (initial) {
    let bb = compensatedValue(total.bb);
    if (!finite(bb) || bb < 0.0) {
      reportAt(ERROR_NONFINITE, 5u, INVALID);
      zeroAllOuterDispatches();
      return;
    }
    storePair(8u, total.bb);
    // Preserve the first attempted Section 4.3 scalars even when the fail-closed
    // positivity gate below rejects them. They are diagnostics only on this
    // branch: no recurrence reads previous gamma during the initial reduction.
    storePair(12u, total.gamma);
    storePair(14u, total.delta);
  }
  storePair(10u, total.rr);
  atomicAdd(&control[3], 1u);
  let bb = max(compensatedValue(pairAt(8u)), params.numerics.z);
  let threshold = max(
    params.numerics.y * params.numerics.y,
    params.numerics.x * params.numerics.x * bb,
  );
  if (!initial) { atomicAdd(&control[2], 1u); }
  if (rr <= threshold) {
    atomicStore(&control[1], 1u);
    if (initial) { zeroAllOuterDispatches(); }
    else { zeroRemainingAfterUpdate(atomicLoad(&control[2]) - 1u); }
    return;
  }
  if (!(gamma > 0.0)) {
    reportAt(ERROR_NONPOSITIVE_PRECONDITIONER, 5u, INVALID);
    if (initial) { zeroAllOuterDispatches(); }
    else { zeroRemainingAfterUpdate(atomicLoad(&control[2]) - 1u); }
    return;
  }
  var alpha = 0.0;
  if (initial) {
    if (!(delta > 0.0)) {
      reportAt(ERROR_NONPOSITIVE_CURVATURE, 5u, INVALID);
      zeroAllOuterDispatches();
      return;
    }
    alpha = gamma / delta;
    storePair(18u, CompensatedF32(0.0, 0.0));
  } else {
    let previousGamma = compensatedValue(pairAt(12u));
    let beta = gamma / previousGamma;
    if (!(previousGamma > 0.0) || !finite(beta) || beta < 0.0) {
      reportAt(ERROR_NONPOSITIVE_PRECONDITIONER, 5u, INVALID);
      zeroRemainingAfterUpdate(atomicLoad(&control[2]) - 1u);
      return;
    }
    storePair(18u, CompensatedF32(beta, 0.0));
  }
  if (initial && (!finite(alpha) || !(alpha > 0.0))) {
    reportAt(ERROR_NONFINITE, 5u, INVALID);
    if (initial) { zeroAllOuterDispatches(); }
    else { zeroRemainingAfterUpdate(atomicLoad(&control[2]) - 1u); }
    return;
  }
  storePair(12u, total.gamma);
  storePair(14u, total.delta);
  storePair(16u, CompensatedF32(alpha, 0.0));
  if (!initial && atomicLoad(&control[2]) >= maximumIterations()) {
    reportAt(ERROR_NONCONVERGENCE, 6u, INVALID);
    zeroRemainingAfterUpdate(maximumIterations() - 1u);
  }
}

@compute @workgroup_size(${OCTREE_PIPELINED_PCG_WORKGROUP_SIZE})
fn finishMergedReduction(
  @builtin(local_invocation_index) lane: u32,
  @builtin(subgroup_invocation_id) subgroupLane: u32,
  @builtin(subgroup_size) subgroupSize: u32,
) {
  if (lane == 0u) { finishMergedTotal(fixedMergedValue()); }
}

@compute @workgroup_size(${OCTREE_PIPELINED_PCG_WORKGROUP_SIZE})
fn reduceAndFinishMerged(
  @builtin(local_invocation_index) lane: u32,
  @builtin(subgroup_invocation_id) subgroupLane: u32,
  @builtin(subgroup_size) subgroupSize: u32,
) {
  let blocks = livePartialCount();
  for (var block = 0u; block < blocks; block += 1u) {
    var local = zeroMergedScalars();
    let row = block * REDUCTION_LANES + lane;
    let initial = atomicLoad(&control[3]) == 0u;
    if (row < rows() && !stopped()) {
      let r = residual[row];
      let u = preconditioned[row];
      let w = preconditionedImage[row];
      let b = -rhs[row];
      if (!finite(r) || !finite(u) || (initial && !finite(w))) {
        reportAt(ERROR_NONFINITE, 4u, row);
      } else {
        local.gamma = addCompensatedF32(local.gamma, r * u);
        if (initial) { local.delta = addCompensatedF32(local.delta, u * w); }
        local.rr = addCompensatedF32(local.rr, r * r);
        if (initial) { local.bb = addCompensatedF32(local.bb, b * b); }
      }
    }
${targetSubgroupReductionWGSL}
    if (lane == 0u) { combinedBlocks[block] = merged[0]; }
    workgroupBarrier();
  }
  var local = zeroMergedScalars();
  if (lane < blocks) {
    local = mergeScalars(local, combinedBlocks[lane]);
  }
${targetSubgroupReductionWGSL}
  if (lane == 0u) { finishMergedTotal(merged[0]); }
}

@compute @workgroup_size(64)
fn initializeDirections(@builtin(global_invocation_id) global: vec3u) {
  let row = global.x;
  if (row >= rows() || stopped()) { return; }
  direction[row] = preconditioned[row];
  directionImage[row] = preconditionedImage[row];
}

@compute @workgroup_size(64)
fn advancePCGState(@builtin(global_invocation_id) global: vec3u) {
  let row = global.x;
  if (row >= rows() || stopped()) { return; }
  let alpha = compensatedValue(pairAt(16u));
  let nextPressure = pressure[row] + alpha * direction[row];
  let nextResidual = residual[row] - alpha * directionImage[row];
  if (!finite(nextPressure) || !finite(nextResidual)) {
    reportAt(ERROR_NONFINITE, 7u, row);
    return;
  }
  pressure[row] = nextPressure;
  residual[row] = nextResidual;
}

@compute @workgroup_size(64)
fn updateDirections(@builtin(global_invocation_id) global: vec3u) {
  let row = global.x;
  if (row >= rows() || stopped()) { return; }
  let beta = compensatedValue(pairAt(18u));
  let nextDirection = preconditioned[row] + beta * direction[row];
  if (!finite(nextDirection)) {
    reportAt(ERROR_NONFINITE, 8u, row);
    return;
  }
  direction[row] = nextDirection;
}

@compute @workgroup_size(${OCTREE_PIPELINED_PCG_WORKGROUP_SIZE})
fn reduceDirectionCurvaturePartials(
  @builtin(local_invocation_index) lane: u32,
  @builtin(workgroup_id) workgroup: vec3u,
  @builtin(global_invocation_id) global: vec3u,
  @builtin(subgroup_invocation_id) subgroupLane: u32,
  @builtin(subgroup_size) subgroupSize: u32,
) {
  clearFixedPartial(workgroup.x, lane);
  let row = global.x;
  if (row < rows() && !stopped()) {
    let d = direction[row];
    let image = directionImage[row];
    if (!finite(d) || !finite(image)) {
      reportAt(ERROR_NONFINITE, 15u, row);
    } else {
      addFixedF32(workgroup.x, 1u, d * image);
    }
  }
}

fn finishDirectionCurvatureTotal(total: MergedScalars) {
  if (stopped()) { return; }
  atomicAdd(&control[3], 1u);
  let curvature = total.delta;
  let direct = compensatedValue(curvature);
  let gamma = compensatedValue(pairAt(12u));
  if (!finite(direct) || !(direct > 0.0)) {
    reportAt(ERROR_NONPOSITIVE_CURVATURE, 15u, INVALID);
    zeroRemainingAfterUpdate(atomicLoad(&control[2]) - 1u);
    return;
  }
  let alpha = gamma / direct;
  if (!finite(alpha) || !(alpha > 0.0)) {
    reportAt(ERROR_NONFINITE, 15u, INVALID);
    zeroRemainingAfterUpdate(atomicLoad(&control[2]) - 1u);
    return;
  }
  storePair(14u, curvature);
  storePair(16u, CompensatedF32(alpha, 0.0));
}

@compute @workgroup_size(${OCTREE_PIPELINED_PCG_WORKGROUP_SIZE})
fn finishDirectionCurvature(
  @builtin(local_invocation_index) lane: u32,
  @builtin(subgroup_invocation_id) subgroupLane: u32,
  @builtin(subgroup_size) subgroupSize: u32,
) {
  if (lane == 0u) { finishDirectionCurvatureTotal(fixedMergedValue()); }
}

@compute @workgroup_size(${OCTREE_PIPELINED_PCG_WORKGROUP_SIZE})
fn reduceAndFinishDirectionCurvature(
  @builtin(local_invocation_index) lane: u32,
  @builtin(subgroup_invocation_id) subgroupLane: u32,
  @builtin(subgroup_size) subgroupSize: u32,
) {
  let blocks = livePartialCount();
  for (var block = 0u; block < blocks; block += 1u) {
    var local = zeroMergedScalars();
    let row = block * REDUCTION_LANES + lane;
    if (row < rows() && !stopped()) {
      let d = direction[row];
      let image = directionImage[row];
      if (!finite(d) || !finite(image)) {
        reportAt(ERROR_NONFINITE, 15u, row);
      } else {
        local.delta = addCompensatedF32(local.delta, d * image);
      }
    }
${targetSubgroupReductionWGSL}
    if (lane == 0u) { combinedBlocks[block] = merged[0]; }
    workgroupBarrier();
  }
  var local = zeroMergedScalars();
  if (lane < blocks) {
    local = mergeScalars(local, combinedBlocks[lane]);
  }
${targetSubgroupReductionWGSL}
  if (lane == 0u) { finishDirectionCurvatureTotal(merged[0]); }
}

@compute @workgroup_size(64)
fn finalizeAndPublish(@builtin(global_invocation_id) global: vec3u) {
  let row = global.x;
  if (global.x == 0u && !failed() && atomicLoad(&control[1]) == 0u) {
    reportAt(ERROR_NONCONVERGENCE, 9u, INVALID);
  }
  storageBarrier();
  if (row >= rows()) { return; }
  // Section 4.3 states the gate as "satisfactory convergence within 6-10
  // iterations"; the 1e-4 relative residual is this repository's f32 QA policy,
  // not a paper quantity (see tools/webgpu-smoke-pressure.ts). Discarding a
  // budget-limited iterate and republishing the SEED therefore inverts the
  // paper: a settled tank that reached 2.4e-4 was being handed back completely
  // unprojected, which is far worse than the iterate it threw away. Exhausting
  // the budget, or a preconditioner/curvature breakdown once CG has already
  // advanced, still leaves the best iterate strictly better than the seed, so
  // publish it. The seed remains the fallback for a solve that produced no
  // trustworthy iterate at all: invalid authority, an invalid row, or a
  // non-finite reduction, plus any breakdown before the first update.
  let fatal = (atomicLoad(&control[0]) & (ERROR_INVALID_AUTHORITY
    | ERROR_INVALID_ROW | ERROR_NONFINITE)) != 0u;
  let converged = atomicLoad(&control[1]) != 0u;
  let advanced = atomicLoad(&control[2]) > 0u;
  let usable = !fatal && (converged || advanced);
  let seed = select(0.0, pressureSeed[row], finite(pressureSeed[row]));
  let candidate = pressure[row];
  pressureOut[row] = select(seed, candidate, usable && finite(candidate));
  if (global.x == 0u && converged && !fatal) { atomicStore(&control[20], 1u); }
}
`;

interface MGPCGActivityEntry {
  readonly entryPoint: PipelineName;
  readonly workgroupLaneCount: 1 | 64 | 128;
  readonly workgroupId: string;
  readonly localInvocationIndex: string;
  readonly injectWorkgroupId?: boolean;
  readonly injectLocalInvocationIndex?: boolean;
  /** Uniform logical-work predicate; it does not claim physical execution-unit residency. */
  readonly meaningfulWhen: string;
  /** Per-invocation predicate retained by the activity-only subgroup ballot. */
  readonly activeWhen: string;
}

const MGPCG_ACTIVITY_ENTRIES: readonly MGPCGActivityEntry[] = [
  { entryPoint: "initializeControlAndDispatch", workgroupLaneCount: 1,
    workgroupId: "activityWorkgroupId", localInvocationIndex: "activityLocalInvocationIndex",
    injectWorkgroupId: true, injectLocalInvocationIndex: true,
    meaningfulWhen: "true", activeWhen: "true" },
  { entryPoint: "validateAuthority", workgroupLaneCount: 1,
    workgroupId: "activityWorkgroupId", localInvocationIndex: "activityLocalInvocationIndex",
    injectWorkgroupId: true, injectLocalInvocationIndex: true,
    meaningfulWhen: "true", activeWhen: "true" },
  { entryPoint: "initializeState", workgroupLaneCount: 64,
    workgroupId: "activityWorkgroupId", localInvocationIndex: "activityLocalInvocationIndex",
    injectWorkgroupId: true, injectLocalInvocationIndex: true,
    meaningfulWhen: "activityWorkgroupId.x * 64u < rows() && !failed()",
    activeWhen: "activityWorkgroupId.x * 64u + activityLocalInvocationIndex < rows() && !failed()" },
  { entryPoint: "formInitialResidual", workgroupLaneCount: 64,
    workgroupId: "activityWorkgroupId", localInvocationIndex: "activityLocalInvocationIndex",
    injectWorkgroupId: true, injectLocalInvocationIndex: true,
    meaningfulWhen: "activityWorkgroupId.x * 64u < rows() && !failed()",
    activeWhen: "activityWorkgroupId.x * 64u + activityLocalInvocationIndex < rows() && !failed()" },
  { entryPoint: "reduceMergedPartials", workgroupLaneCount: 128,
    workgroupId: "workgroup", localInvocationIndex: "lane",
    meaningfulWhen: "workgroup.x * REDUCTION_LANES < rows() && !stopped()",
    activeWhen: "workgroup.x * REDUCTION_LANES + lane < rows() && !stopped()" },
  { entryPoint: "finishMergedReduction", workgroupLaneCount: 128,
    workgroupId: "activityWorkgroupId", localInvocationIndex: "lane",
    injectWorkgroupId: true, meaningfulWhen: "true",
    activeWhen: "lane < livePartialCount()" },
  { entryPoint: "reduceAndFinishMerged", workgroupLaneCount: 128,
    workgroupId: "activityWorkgroupId", localInvocationIndex: "lane",
    injectWorkgroupId: true, meaningfulWhen: "!stopped()",
    activeWhen: "lane < min(rows(), REDUCTION_LANES) && !stopped()" },
  { entryPoint: "initializeDirections", workgroupLaneCount: 64,
    workgroupId: "activityWorkgroupId", localInvocationIndex: "activityLocalInvocationIndex",
    injectWorkgroupId: true, injectLocalInvocationIndex: true,
    meaningfulWhen: "activityWorkgroupId.x * 64u < rows() && !stopped()",
    activeWhen: "activityWorkgroupId.x * 64u + activityLocalInvocationIndex < rows() && !stopped()" },
  { entryPoint: "advancePCGState", workgroupLaneCount: 64,
    workgroupId: "activityWorkgroupId", localInvocationIndex: "activityLocalInvocationIndex",
    injectWorkgroupId: true, injectLocalInvocationIndex: true,
    meaningfulWhen: "activityWorkgroupId.x * 64u < rows() && !stopped()",
    activeWhen: "activityWorkgroupId.x * 64u + activityLocalInvocationIndex < rows() && !stopped()" },
  { entryPoint: "updateDirections", workgroupLaneCount: 64,
    workgroupId: "activityWorkgroupId", localInvocationIndex: "activityLocalInvocationIndex",
    injectWorkgroupId: true, injectLocalInvocationIndex: true,
    meaningfulWhen: "activityWorkgroupId.x * 64u < rows() && !stopped()",
    activeWhen: "activityWorkgroupId.x * 64u + activityLocalInvocationIndex < rows() && !stopped()" },
  { entryPoint: "reduceDirectionCurvaturePartials", workgroupLaneCount: 128,
    workgroupId: "workgroup", localInvocationIndex: "lane",
    meaningfulWhen: "workgroup.x * REDUCTION_LANES < rows() && !stopped()",
    activeWhen: "workgroup.x * REDUCTION_LANES + lane < rows() && !stopped()" },
  { entryPoint: "finishDirectionCurvature", workgroupLaneCount: 128,
    workgroupId: "activityWorkgroupId", localInvocationIndex: "lane",
    injectWorkgroupId: true, meaningfulWhen: "!stopped()",
    activeWhen: "lane < livePartialCount() && !stopped()" },
  { entryPoint: "reduceAndFinishDirectionCurvature", workgroupLaneCount: 128,
    workgroupId: "activityWorkgroupId", localInvocationIndex: "lane",
    injectWorkgroupId: true, meaningfulWhen: "!stopped()",
    activeWhen: "lane < min(rows(), REDUCTION_LANES) && !stopped()" },
  { entryPoint: "finalizeAndPublish", workgroupLaneCount: 64,
    workgroupId: "activityWorkgroupId", localInvocationIndex: "activityLocalInvocationIndex",
    injectWorkgroupId: true, injectLocalInvocationIndex: true,
    meaningfulWhen: "activityWorkgroupId.x * 64u < rows()",
    activeWhen: "activityWorkgroupId.x * 64u + activityLocalInvocationIndex < rows()" },
] as const;

function matchingWGSLDelimiter(source: string, start: number, open: string, close: string): number {
  let depth = 0;
  for (let index = start; index < source.length; index += 1) {
    if (source[index] === open) depth += 1;
    else if (source[index] === close && --depth === 0) return index;
  }
  return -1;
}

function instrumentMGPCGActivityEntry(
  source: string,
  activity: GPULogicalActivityAdoptionContext,
  spec: MGPCGActivityEntry,
): string {
  const descriptor = OCTREE_PIPELINED_MGPCG_ACTIVITY_TASKS[spec.entryPoint];
  const signature = `fn ${spec.entryPoint}(`;
  const signatureStart = source.indexOf(signature);
  if (signatureStart < 0) throw new Error(`MGPCG activity entry point ${spec.entryPoint} is missing`);
  const paramsStart = source.indexOf("(", signatureStart + 3);
  const paramsEnd = matchingWGSLDelimiter(source, paramsStart, "(", ")");
  if (paramsStart < 0 || paramsEnd < 0) throw new Error(`MGPCG activity signature ${spec.entryPoint} is malformed`);
  const existing = source.slice(paramsStart + 1, paramsEnd);
  const builtinName = (builtin: string): string | undefined => new RegExp(
    `@builtin\\(${builtin}\\)\\s*([A-Za-z_]\\w*)\\s*:`,
  ).exec(existing)?.[1];
  const existingSubgroupLane = builtinName("subgroup_invocation_id");
  const existingSubgroupSize = builtinName("subgroup_size");
  const subgroupLane = existingSubgroupLane ?? "activitySubgroupLane";
  const subgroupSize = existingSubgroupSize ?? "activitySubgroupSize";
  const meaningful = `let fluidMGPCGActivityMeaningful = ${spec.meaningfulWhen};`;
  const entry = activity.subgroup(descriptor.task, "enter", {
    tick: "atomicLoad(&control[2])",
    workgroupId: spec.workgroupId,
    numWorkgroups: "activityNumWorkgroups",
    // WebGPU exposes subgroup lane and size, but not a portable subgroup ID.
    // Workgroup-local invocation indices are contiguous, so retain this as
    // explicitly reconstructed evidence rather than claiming it was measured.
    subgroupId: `(${spec.localInvocationIndex} / ${subgroupSize})`,
    subgroupIdEvidence: "reconstructed",
    subgroupLane,
    subgroupSize,
    active: spec.activeWhen,
  });
  // Exit checkpoints remain scalar heartbeats. Unlike subgroupBallot, this
  // helper is legal on divergent early-return paths and preserves the existing
  // enter/exit task accounting without pretending to measure a second ballot.
  const exit = activity.workgroup(descriptor.task, "exit", {
    tick: "atomicLoad(&control[2])",
    workgroupId: spec.workgroupId,
    numWorkgroups: "activityNumWorkgroups",
    localInvocationIndex: spec.localInvocationIndex,
    workgroupLaneCount: spec.workgroupLaneCount,
    recordWhen: "fluidMGPCGActivityMeaningful",
  });
  if (!entry && !exit) return source;

  const additions = [
    ...(spec.injectWorkgroupId ? ["@builtin(workgroup_id) activityWorkgroupId: vec3u"] : []),
    ...(spec.injectLocalInvocationIndex ? ["@builtin(local_invocation_index) activityLocalInvocationIndex: u32"] : []),
    ...(existingSubgroupLane ? [] : ["@builtin(subgroup_invocation_id) activitySubgroupLane: u32"]),
    ...(existingSubgroupSize ? [] : ["@builtin(subgroup_size) activitySubgroupSize: u32"]),
    "@builtin(num_workgroups) activityNumWorkgroups: vec3u",
  ];
  const separator = existing.trim().length > 0 && additions.length > 0
    ? existing.trimEnd().endsWith(",") ? "\n  " : ",\n  "
    : "";
  let instrumented = `${source.slice(0, paramsEnd)}${separator}${additions.join(",\n  ")}${source.slice(paramsEnd)}`;
  const bodyStart = instrumented.indexOf("{", paramsEnd + separator.length + additions.join(",\n  ").length);
  const bodyEnd = matchingWGSLDelimiter(instrumented, bodyStart, "{", "}");
  if (bodyStart < 0 || bodyEnd < 0) throw new Error(`MGPCG activity body ${spec.entryPoint} is malformed`);
  const body = instrumented.slice(bodyStart + 1, bodyEnd)
    .replace(/\breturn;/g, `${exit}\n  return;`);
  instrumented = `${instrumented.slice(0, bodyStart + 1)}\n  ${meaningful}\n  ${entry}\n${body}\n  ${exit}\n${instrumented.slice(bodyEnd)}`;
  return instrumented;
}

/** Activity-only solver variant; disabled mode returns the base shader byte-for-byte. */
export function octreePipelinedMGPCGActivityShader(
  activity: GPULogicalActivityAdoptionContext,
  base: string = octreePipelinedMGPCGShader,
): string {
  if (!activity.enabled) return base;
  return MGPCG_ACTIVITY_ENTRIES.reduce(
    (source, spec) => instrumentMGPCGActivityEntry(source, activity, spec),
    base,
  );
}

/**
 * The single statement that seeds the initial iterate, and its cold
 * replacement. The finiteness audit above it is deliberately left alone: a
 * non-finite seed still fails its row closed in both arms, so the two arms
 * differ in the iterate they start from and in nothing else.
 */
const MGPCG_WARM_SEED_STATEMENT = "\n  pressure[row] = seed;\n";
const MGPCG_COLD_SEED_STATEMENT = "\n  pressure[row] = 0.0;\n";

/**
 * Cold-start measurement variant; `false` returns the production shader
 * byte-for-byte (by identity, not by reconstruction). See
 * `octreePipelinedMGPCGColdStartEnabled` for why this exists at all.
 */
export function octreePipelinedMGPCGSeedVariantShader(coldStart: boolean): string {
  if (!coldStart) return octreePipelinedMGPCGShader;
  const occurrences =
    octreePipelinedMGPCGShader.split(MGPCG_WARM_SEED_STATEMENT).length - 1;
  if (occurrences !== 1) {
    throw new Error(
      `Pipelined MGPCG warm-seed statement is not unique (${occurrences} matches)`,
    );
  }
  return octreePipelinedMGPCGShader.replace(
    MGPCG_WARM_SEED_STATEMENT, MGPCG_COLD_SEED_STATEMENT,
  );
}

/** Fixed-coefficient temporal seed with no additional accurate-operator apply. */
export function octreePipelinedMGPCGFixedTemporalSeedShader(
  base: string,
  alpha: number,
): string {
  if (!Number.isFinite(alpha) || alpha < -1 || alpha > 2) {
    throw new RangeError("Fixed temporal pressure alpha must remain in [-1,2]");
  }
  const rhsDeclaration =
    "@group(0) @binding(16) var<storage, read> rhs: array<f32>;";
  const alphaText = Number.isInteger(Math.fround(alpha))
    ? `${Math.fround(alpha)}.0`
    : `${Math.fround(alpha)}`;
  const replacement = `
  let temporal = pressureHistory[row];
  let generation = acceptedAuthority[4];
  let temporalAge = select(0u, generation - 4u, generation > 4u);
  let temporalAlpha = ${alphaText} * clamp(f32(temporalAge) / 64.0, 0.0, 1.0);
  let predicted = seed + temporalAlpha * select(0.0, temporal, finite(temporal));
  pressure[row] = select(seed, predicted, finite(predicted));
`;
  if (!base.includes(rhsDeclaration) || !base.includes(MGPCG_WARM_SEED_STATEMENT)) {
    throw new Error("Pipelined MGPCG fixed temporal seed anchors are unavailable");
  }
  return base
    .replace(rhsDeclaration, `${rhsDeclaration}\n`
      + "@group(0) @binding(17) var<storage, read> pressureHistory: array<f32>;")
    .replace(MGPCG_WARM_SEED_STATEMENT, replacement);
}
