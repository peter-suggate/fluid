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
export const OCTREE_PIPELINED_PCG_PARTIAL_BYTES = 32;
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
  encode(
    broker: PassBroker,
    input: GPUBuffer,
    output: GPUBuffer,
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
  /** Accepted direct structured publication control:
   * [flags, firstError, rows, generation, bank, slots]. */
  readonly acceptedAuthority: GPUBuffer;
  readonly operator: OctreePipelinedLinearOperator;
  readonly preconditioner: OctreePipelinedFixedPreconditioner;
  readonly vectors: OctreePipelinedMGPCGVectors;
}

export interface OctreePipelinedMGPCGSolve {
  readonly pressureSeed: GPUBuffer;
  readonly pressureOut: GPUBuffer;
}

export type OctreePipelinedMGPCGPipelineName =
  | "initializeControlAndDispatch"
  | "validateAuthority"
  | "initializeState"
  | "formInitialResidual"
  | "reduceMergedPartials"
  | "finishMergedReduction"
  | "reduceDirectionCurvaturePartials"
  | "finishDirectionCurvature"
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
  reduceDirectionCurvaturePartials: mgpcgActivityTask("reduceDirectionCurvaturePartials", "reduce-direction-curvature", "MGPCG · reduce direction curvature"),
  finishDirectionCurvature: mgpcgActivityTask("finishDirectionCurvature", "finish-direction-curvature", "MGPCG · finish direction curvature"),
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
  reduceDirectionCurvaturePartials: [0, 2, 11, 12, 13, 14],
  finishDirectionCurvature: [0, 2, 13, 14, 15],
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

/**
 * Replacement-ready matrix-free PCG owner. It deliberately does not
 * allocate hybrid smoother or multigrid vectors.
 */
export class WebGPUOctreePipelinedMGPCG {
  readonly plan: OctreePipelinedMGPCGPlan;
  readonly iterationBudget: number;
  readonly control: GPUBuffer;
  readonly allocatedBytes: number;
  readonly encodedPassTransitionCountPerIteration = 4;
  readonly reductionsPerOuterIteration = 2 as const;

  /** GPU-authored records suitable for the existing diagnostics readback. */
  get workAccountingBuffers(): Readonly<{ control: GPUBuffer; indirectTail: GPUBuffer }> {
    return Object.freeze({ control: this.control, indirectTail: this.outerDispatch });
  }

  get workAccountingPlan(): Readonly<{ rowCapacity: number; maximumIterations: number;
    reductionLanes: 128; reductionPartialCount: number }> {
    return Object.freeze({ rowCapacity: this.plan.rowCapacity,
      maximumIterations: this.plan.maximumIterations,
      reductionLanes: this.plan.reductionLanes,
      reductionPartialCount: this.plan.reductionPartialCount });
  }

  private readonly params: GPUBuffer;
  private readonly partials: GPUBuffer;
  private readonly outerDispatch: GPUBuffer;
  private readonly pipelines: Readonly<Record<PipelineName, GPUComputePipeline>>;
  private readonly groups = new Map<PipelineName, CachedGroup[]>();
  private readonly relativeTolerance: number;
  private readonly absoluteTolerance: number;
  private destroyed = false;

  constructor(
    private readonly device: GPUDevice,
    private readonly source: OctreePipelinedMGPCGSource,
    options: OctreePipelinedMGPCGOptions,
  ) {
    this.plan = planOctreePipelinedMGPCG(options);
    this.iterationBudget = this.plan.maximumIterations;
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
    for (const descriptor of Object.values(OCTREE_PIPELINED_MGPCG_ACTIVITY_TASKS)) {
      activity.describeTask(descriptor.task, {
        id: descriptor.id,
        label: descriptor.label,
        phaseId: "pressure-solve",
        checkpoints: descriptor.checkpoints,
      });
    }
    const shaderVariant = activity.module(
      octreePipelinedMGPCGActivityShader(activity),
      `${OCTREE_PIPELINED_MGPCG_ACTIVITY_MODULE_ID}/${activityProfile.cacheKey}`,
    );
    const shaderModule = device.createShaderModule({
      label: "Octree pipelined MGPCG · M1 Max 128-lane subgroups",
      code: shaderVariant.code,
    });
    this.pipelines = Object.freeze(Object.fromEntries(
      (Object.keys(PIPELINE_BINDINGS) as PipelineName[]).map((entryPoint) => [
        entryPoint,
        activity.registerPipeline(device.createComputePipeline({
          label: `Pipelined MGPCG · ${entryPoint} · target-128-subgroups`,
          layout: "auto",
          compute: { module: shaderModule, entryPoint },
        })),
      ]),
    ) as Record<PipelineName, GPUComputePipeline>);
    this.allocatedBytes = this.plan.ownedBytes;
  }

  get encodedDispatchCount(): number {
    const setup = this.source.preconditioner.encodedSetupDispatchCount ?? 0;
    const correction = this.source.preconditioner.encodedCorrectionDispatchCount;
    const apply = this.source.operator.encodedDispatchCount;
    // Control/authority/state/residual, setup, initial M, A(Mr), reduction
    // pair, and directions; then advance + M + residual reduction/finish +
    // direction + A(direction) + direct-curvature reduction/finish,
    // and one fail-closed publication.
    return 8 + 2 * apply + setup + correction
      + this.plan.maximumIterations * (6 + apply + correction);
  }

  encode(broker: PassBroker, solve: OctreePipelinedMGPCGSolve): void {
    this.assertLive();
    const vectorBytes = this.plan.rowCapacity * 4;
    if (solve.pressureSeed.size < vectorBytes || solve.pressureOut.size < vectorBytes) {
      throw new RangeError("Pipelined MGPCG solve buffers are smaller than the planned capacity");
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
    this.source.operator.encode(
      broker, this.source.vectors.pressure, this.source.vectors.directionImage,
      this.control,
    );
    pass = broker.compute({ label: "Pipelined MGPCG initial residual" });
    this.runOuterIndirect(pass, "formInitialResidual", 0, 0, resources);

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
    this.source.operator.encode(
      broker, this.source.vectors.preconditioned, this.source.vectors.preconditionedImage,
      this.control,
    );
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

    for (let iteration = 0; iteration < this.plan.maximumIterations; iteration += 1) {
      pass = broker.compute({ label: `Pipelined MGPCG outer iteration ${iteration}` });
      this.runIndirect(pass, "advancePCGState", iteration, 0, resources);
      preconditioner.encodeCorrection(broker, {
        rhs: this.source.vectors.residual,
        correction: this.source.vectors.preconditioned,
        solverControl: this.control,
        rowCount: this.source.rowCount,
      });
      pass = broker.compute({ label: "Pipelined MGPCG merged partial reduction" });
      this.runIndirect(pass, "reduceMergedPartials", iteration, 2, resources);
      // The finishing workgroup reads every partial and mutates future
      // indirect records, so it owns the single global reduction boundary.
      broker.fence("pipelined MGPCG merged partials complete");
      pass = broker.compute({ label: "Pipelined MGPCG merged reduction finish" });
      this.runDirect(pass, "finishMergedReduction", [1, 1, 1], resources);
      broker.fence("pipelined MGPCG convergence-tail publication");
      pass = broker.compute({ label: "Pipelined MGPCG direction update" });
      this.runIndirect(pass, "updateDirections", iteration, 3, resources);
      this.source.operator.encode(
        broker, this.source.vectors.direction, this.source.vectors.directionImage,
        this.control,
      );
      pass = broker.compute({ label: "Pipelined MGPCG direct direction curvature" });
      this.runIndirect(pass, "reduceDirectionCurvaturePartials", iteration, 1, resources);
      broker.fence("pipelined MGPCG direct curvature partials complete");
      pass = broker.compute({ label: "Pipelined MGPCG direct curvature finish" });
      this.runDirect(pass, "finishDirectionCurvature", [1, 1, 1], resources);
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
    ];
  }

  private bind(
    pass: GPUComputePassEncoder,
    name: PipelineName,
    resources: readonly (GPUBuffer | undefined)[],
  ): void {
    const pipeline = this.pipelines[name], bindings = PIPELINE_BINDINGS[name];
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
@group(0) @binding(14) var<storage, read_write> partials: array<MergedScalars>;
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
fn acceptedBank() -> u32 { return acceptedAuthority[4]; }
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
  if (arrayLength(&acceptedAuthority) < 6u
    || acceptedAuthority[2] == 0u || acceptedAuthority[2] > capacity()) {
    reportAt(ERROR_INVALID_AUTHORITY, 1u, INVALID);
    return;
  }
  if (acceptedAuthority[0] != 0u
    || acceptedAuthority[1] != INVALID
    || acceptedAuthority[3] == 0u
    || acceptedAuthority[4] > 1u
    || acceptedAuthority[5] == 0u) {
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

@compute @workgroup_size(${OCTREE_PIPELINED_PCG_WORKGROUP_SIZE})
fn reduceMergedPartials(
  @builtin(local_invocation_index) lane: u32,
  @builtin(workgroup_id) workgroup: vec3u,
  @builtin(global_invocation_id) global: vec3u,
  @builtin(subgroup_invocation_id) subgroupLane: u32,
  @builtin(subgroup_size) subgroupSize: u32,
) {
  var local = zeroMergedScalars();
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
      local.gamma = addCompensatedF32(local.gamma, r * u);
      if (initial) { local.delta = addCompensatedF32(local.delta, u * w); }
      local.rr = addCompensatedF32(local.rr, r * r);
      if (initial) { local.bb = addCompensatedF32(local.bb, b * b); }
    }
  }
${targetSubgroupReductionWGSL}
  if (lane == 0u) { partials[workgroup.x] = merged[0]; }
}

@compute @workgroup_size(${OCTREE_PIPELINED_PCG_WORKGROUP_SIZE})
fn finishMergedReduction(
  @builtin(local_invocation_index) lane: u32,
  @builtin(subgroup_invocation_id) subgroupLane: u32,
  @builtin(subgroup_size) subgroupSize: u32,
) {
  var local = zeroMergedScalars();
  for (var partial = lane; partial < livePartialCount(); partial += REDUCTION_LANES) {
    local = mergeScalars(local, partials[partial]);
  }
${targetSubgroupReductionWGSL}
  if (lane != 0u) { return; }
  let initial = atomicLoad(&control[3]) == 0u;
  if (failed()) {
    if (initial) { zeroAllOuterDispatches(); }
    else { zeroRemainingAfterUpdate(atomicLoad(&control[2])); }
    return;
  }
  // A converged earlier reduction already zeroed every remaining indirect
  // record. The fixed singleton tail must not recount stale partials.
  if (atomicLoad(&control[1]) != 0u) { return; }
  let total = merged[0];
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
  var local = zeroMergedScalars();
  let row = global.x;
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
  if (lane == 0u) { partials[workgroup.x] = merged[0]; }
}

@compute @workgroup_size(${OCTREE_PIPELINED_PCG_WORKGROUP_SIZE})
fn finishDirectionCurvature(
  @builtin(local_invocation_index) lane: u32,
  @builtin(subgroup_invocation_id) subgroupLane: u32,
  @builtin(subgroup_size) subgroupSize: u32,
) {
  var local = zeroMergedScalars();
  for (var partial = lane; partial < livePartialCount(); partial += REDUCTION_LANES) {
    local = mergeScalars(local, partials[partial]);
  }
${targetSubgroupReductionWGSL}
  if (lane != 0u || stopped()) { return; }
  atomicAdd(&control[3], 1u);
  let curvature = merged[0].delta;
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

/** Activity-only solver variant; disabled mode returns the production shader byte-for-byte. */
export function octreePipelinedMGPCGActivityShader(
  activity: GPULogicalActivityAdoptionContext,
): string {
  if (!activity.enabled) return octreePipelinedMGPCGShader;
  return MGPCG_ACTIVITY_ENTRIES.reduce(
    (source, spec) => instrumentMGPCGActivityEntry(source, activity, spec),
    octreePipelinedMGPCGShader,
  );
}
