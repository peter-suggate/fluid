/** Matrix-free Section 4.3 pressure solve for compact power-diagram rows. */
import { PassBroker } from "./webgpu-pass-broker";

export const OCTREE_SECTION43_BOUNDARY_SMOOTHING_ITERATIONS = 8;
export const OCTREE_SECTION43_BOUNDARY_BAND_LAYERS = 3;
/** Small power-octree solves are launch-bound long before they are ALU-bound.
 * Exact Dawn audits observe maxima of 8 iterations over 500 generations at
 * 16^3 and 7 iterations over all 62 generations of the browser's 24x18x16
 * mini dam break. Retain 50% headroom over those measured envelopes and the
 * paper's reported 6--10 range. */
export const OCTREE_SECTION43_SMALL_DOMAIN_PCG_ITERATIONS = 12;
/** Immutable execution ceiling for the one-workgroup solve lane. This is a
 * performance bound, not merely a storage-fit bound: profiling a 6,912-row
 * dam operator put 804 ms in this serial workgroup. Keep the persistent lane
 * to one row per lane; normal production capacities use parallel PCG. */
export const OCTREE_PERSISTENT_MGPCG_MAXIMUM_ROW_CAPACITY = 64;
/** x, r, z, d, A*d, four hybrid f32 fields, and two band u32 fields.  A
 * combined persistent shader must pack these channels into one binding to
 * fit beside the live L2 rows and SPGrid topology/state under WebGPU's ten
 * storage-buffer stage limit requested by the fluid renderer. */
export const OCTREE_PERSISTENT_MGPCG_STATE_CHANNELS = 11;
/** Large-domain PCG retains the paper's 6--10 iteration envelope. The local
 * 181-generation Dawn audit converged in at most six iterations, leaving four
 * iterations of headroom without encoding two additional V-cycles that can
 * only early-out. Unlike the retired Chebyshev polynomial, every iteration
 * responds to the actual preconditioned residual and needs no spectral-
 * estimation preamble. */
export const OCTREE_SECTION43_LARGE_DOMAIN_PCG_ITERATIONS =
  Number((globalThis as { process?: { env?: Record<string, string | undefined> } })
    .process?.env?.FLUID_PCG_ITERATIONS ?? 10);

export function normalizeOctreeSection43BoundarySmoothing(value: number | undefined): number {
  const requested = value ?? OCTREE_SECTION43_BOUNDARY_SMOOTHING_ITERATIONS;
  // The fixed ping/pong publication reads A after the pre-sweeps and B after
  // the post-sweeps, so k must remain even as well as symmetry-locked.
  return Math.max(2, Math.min(16, Math.round(requested / 2) * 2));
}

/**
 * The missing middle operation in the paper's Section 4.3 preconditioner.
 * Implementations own their sparse first-order operator, pyramid, and the
 * GhostValuePropagate/GhostValueAccumulate transfers described in Section 4.2.
 * They must encode an SPD linear correction from rhs to correction and report
 * any GPU-side failure through solverControl.
 */
export interface OctreeFirstOrderSPDVCycle {
  readonly operatorOrder: 1;
  readonly isSymmetricPositiveDefinite: true;
  readonly allocatedBytes: number;
  /** Ordered compute dispatches emitted by one correction. */
  readonly encodedCorrectionDispatchCount: number;
  /** Ordered compute dispatches emitted by hierarchy setup. */
  readonly encodedSetupDispatchCount?: number;
  /** Standalone correction pass transitions; optional so alternate hierarchy
   * implementations remain source-compatible with the batching extension. */
  readonly encodedPassTransitionCount?: number;
  /** Exact one-dispatch owner for the complete small-domain outer solve. The
   * capability is proof-carrying; capacities at or below the persistent
   * ceiling are rejected when this proof is absent or incompatible. */
  readonly persistentMGPCG?: OctreePersistentMGPCGExecutor;
  encodeSetup(broker: PassBroker, input: {
    readonly solverControl: GPUBuffer;
    readonly rowCount: GPUBuffer;
  }): void;
  encodeCorrection(broker: PassBroker, input: {
    readonly rhs: GPUBuffer;
    readonly correction: GPUBuffer;
    readonly solverControl: GPUBuffer;
    readonly rowCount: GPUBuffer;
  }): void;
}

export interface OctreePersistentMGPCGExecutor {
  readonly maximumRowCapacity: number;
  readonly encodedDispatchCount: 1;
  readonly dispatchShape: readonly [1, 1, 1];
  readonly invariantProof: Readonly<{
    ghostRows: "spgrid-identical";
    transfers: "validated-adjoint-pair";
    invalidRows: "uniform-fail-closed-before-arithmetic";
  }>;
  /** Encode exactly one workgroup dispatch owning initialization, hybrid
   * preconditioning, PCG reductions/iterations, and final publication. */
  encodeSolve(broker: PassBroker, input: {
    readonly leafHeaders: GPUBuffer;
    readonly leafEntries: GPUBuffer;
    readonly rowCount: GPUBuffer;
    readonly pressureIn: GPUBuffer;
    readonly pressureOut: GPUBuffer;
    readonly solverControl: GPUBuffer;
    readonly boundarySmoothingIterations: number;
    readonly relativeTolerance: number;
  }): void;
}

export type OctreeMGPCGExecutionMode =
  | "persistent-small-domain"
  | "staged-pcg-large-domain";

export function selectOctreeMGPCGExecutionMode(
  rowCapacity: number,
  executor: OctreePersistentMGPCGExecutor | undefined,
): OctreeMGPCGExecutionMode {
  positiveInteger(rowCapacity, "MGPCG row capacity");
  if (rowCapacity > OCTREE_PERSISTENT_MGPCG_MAXIMUM_ROW_CAPACITY) return "staged-pcg-large-domain";
  if (executor === undefined) {
    throw new Error("small-domain MGPCG requires a persistent proof-carrying executor");
  }
  if (executor.maximumRowCapacity < rowCapacity || executor.encodedDispatchCount !== 1
    || executor.dispatchShape[0] !== 1 || executor.dispatchShape[1] !== 1
    || executor.dispatchShape[2] !== 1) {
    throw new Error("small-domain MGPCG persistent executor is incompatible with the immutable solve shape");
  }
  const proof = executor.invariantProof;
  if (proof?.ghostRows === "spgrid-identical"
    && proof.transfers === "validated-adjoint-pair"
    && proof.invalidRows === "uniform-fail-closed-before-arithmetic") return "persistent-small-domain";
  throw new Error("small-domain MGPCG persistent executor is missing the required invariant proof");
}

export const OCTREE_MGPCG_ERROR = Object.freeze({
  invalidRow: 1 << 0,
  nonFinite: 1 << 2,
  nonPositiveOperator: 1 << 3,
  nonConvergence: 1 << 4,
} as const);

export interface OctreeMGPCGPlan {
  readonly rowCapacity: number;
  readonly dispatch: readonly [number, number, number];
  readonly vectorBytes: number;
  readonly hybridBytes: number;
  readonly persistentStateBytes: number;
  readonly persistentRequiredByCapacity: boolean;
  readonly allocatedBytes: number;
}

export interface OctreeMGPCGOptions {
  readonly dimensions: readonly [number, number, number];
  readonly rowCapacity: number;
  /** Matching pre/post L2 sweeps in the Section 4.3 boundary band. A single
   * value deliberately preserves the symmetry required by ordinary PCG. */
  readonly boundarySmoothingIterations?: number;
  readonly relativeTolerance?: number;
}

export interface OctreeMGPCGSource {
  /** LeafHeader array; word zero is the flattened finest-cell origin. */
  readonly leafHeaders: GPUBuffer;
  /** LeafEntry { row:u32, coefficient:f32 } array. */
  readonly leafEntries: GPUBuffer;
  /** Compact publication whose first u32 is the live row count. */
  readonly rowCount: GPUBuffer;
  readonly firstOrderVCycle: OctreeFirstOrderSPDVCycle;
}

function positiveInteger(value: number, label: string): number {
  if (!Number.isSafeInteger(value) || value < 1) throw new RangeError(`${label} must be a positive integer`);
  return value;
}

/** The immutable Section 4.3 command shape is selected only from the planned
 * row capacity. Callers cannot retain or revive a staged iteration budget. */
export function octreeSection43IterationBudget(
  rowCapacity: number,
): typeof OCTREE_SECTION43_SMALL_DOMAIN_PCG_ITERATIONS
  | typeof OCTREE_SECTION43_LARGE_DOMAIN_PCG_ITERATIONS {
  positiveInteger(rowCapacity, "Section 4.3 row capacity");
  return rowCapacity <= OCTREE_PERSISTENT_MGPCG_MAXIMUM_ROW_CAPACITY
    ? OCTREE_SECTION43_SMALL_DOMAIN_PCG_ITERATIONS
    : OCTREE_SECTION43_LARGE_DOMAIN_PCG_ITERATIONS;
}

export function planOctreeMGPCG(options: Pick<OctreeMGPCGOptions, "dimensions" | "rowCapacity">): OctreeMGPCGPlan {
  const rowCapacity = positiveInteger(options.rowCapacity, "MGPCG row capacity");
  options.dimensions.forEach((value) => positiveInteger(value, "MGPCG dimension"));
  const blocks = Math.ceil(rowCapacity / 64);
  const dispatchX = Math.min(65_535, Math.max(1, blocks));
  const dispatchY = Math.max(1, Math.ceil(blocks / dispatchX));
  const vectorBytes = rowCapacity * 4;
  // solution ping/pong, first-order RHS/correction, and two u32 band masks.
  const hybridBytes = vectorBytes * 6;
  return {
    rowCapacity, dispatch: [dispatchX, dispatchY, 1], vectorBytes, hybridBytes,
    persistentStateBytes: vectorBytes * OCTREE_PERSISTENT_MGPCG_STATE_CHANNELS,
    persistentRequiredByCapacity: rowCapacity <= OCTREE_PERSISTENT_MGPCG_MAXIMUM_ROW_CAPACITY,
    // Five PCG vectors, six hybrid fields, params, and solve control. Keep
    // this exact: diagnostics enforce compact-row-bounded pressure storage.
    allocatedBytes: rowCapacity <= OCTREE_PERSISTENT_MGPCG_MAXIMUM_ROW_CAPACITY
      ? 64 : vectorBytes * 5 + hybridBytes + 64 + 64,
  };
}

type Stage = { readonly pipeline: GPUComputePipeline; readonly bindings: readonly number[] };
type CachedStageGroup = {
  readonly resources: readonly (GPUBuffer | undefined)[];
  readonly group: GPUBindGroup;
};

/**
 * Fixed-capacity GPU solver. Tiny domains use one persistent PCG workgroup,
 * normal production domains use a bounded parallel PCG schedule. Every
 * kernel exits from a GPU convergence word, so convergence introduces no CPU
 * scheduling decision or readback.
 */
export class WebGPUOctreeMGPCG {
  readonly plan: OctreeMGPCGPlan;
  readonly control: GPUBuffer;
  private readonly params?: GPUBuffer;
  private readonly x?: GPUBuffer;
  private readonly residual?: GPUBuffer;
  private readonly preconditioned?: GPUBuffer;
  private readonly direction?: GPUBuffer;
  private readonly product?: GPUBuffer;
  private readonly hybridA?: GPUBuffer;
  private readonly hybridB?: GPUBuffer;
  private readonly hybridRhs?: GPUBuffer;
  private readonly hybridCorrection?: GPUBuffer;
  private readonly hybridBandA?: GPUBuffer;
  private readonly hybridBandB?: GPUBuffer;
  private readonly authorityGatePipeline: GPUComputePipeline;
  private authorityGateGroup?: { readonly authorityControl: GPUBuffer; readonly group: GPUBindGroup };
  private readonly stages: Readonly<Record<string, Stage>>;
  /** Immutable descriptors shared by every replay of a stage. */
  private readonly stageGroups = new Map<Stage, CachedStageGroup[]>();
  readonly iterationBudget: typeof OCTREE_SECTION43_SMALL_DOMAIN_PCG_ITERATIONS
    | typeof OCTREE_SECTION43_LARGE_DOMAIN_PCG_ITERATIONS;
  private readonly relativeTolerance: number;
  readonly boundarySmoothingIterations: number;
  private readonly device: GPUDevice;
  private destroyed = false;
  readonly executionMode: OctreeMGPCGExecutionMode;

  /** Fixed host-authored dispatch schedule size. GPU early-out does not remove
   * these commands, so retain the existing observable beside wall timing. */
  get encodedDispatchCount(): number {
    if (this.executionMode === "persistent-small-domain") {
      return 1 + (this.source.firstOrderVCycle.encodedSetupDispatchCount ?? 3) + 1;
    }
    const preconditionerPasses = 4 + 2 * this.boundarySmoothingIterations
      + this.source.firstOrderVCycle.encodedCorrectionDispatchCount;
    const setupPasses = this.source.firstOrderVCycle.encodedSetupDispatchCount ?? 3;
    // initialize + direct residual + band classification/dilation + M*r +
    // initial reduction; then two fused scalar/vector stages per iteration
    // and M*r plus fused rz/direction update between iterations.
    const initialize = 2 + 4 + preconditionerPasses + 1;
    const iterations = 2 * OCTREE_SECTION43_LARGE_DOMAIN_PCG_ITERATIONS
      + (OCTREE_SECTION43_LARGE_DOMAIN_PCG_ITERATIONS - 1) * (preconditionerPasses + 1);
    return 1 + setupPasses + initialize + iterations + 2;
  }

  /** Hierarchy publication must end before its dispatch records can become
   * indirect arguments for the dependency-ordered solve pass. */
  readonly encodedPassTransitionCount = 2;

  constructor(device: GPUDevice, private readonly source: OctreeMGPCGSource, options: OctreeMGPCGOptions) {
    this.device = device;
    this.plan = planOctreeMGPCG(options);
    const cycle = source.firstOrderVCycle;
    if (!cycle || cycle.operatorOrder !== 1 || cycle.isSymmetricPositiveDefinite !== true) {
      throw new Error("Section 4.3 hybrid PCG requires an explicit SPD first-order V-cycle");
    }
    this.iterationBudget = octreeSection43IterationBudget(this.plan.rowCapacity);
    this.executionMode = selectOctreeMGPCGExecutionMode(this.plan.rowCapacity, cycle.persistentMGPCG);
    this.relativeTolerance = Math.max(1e-4, Math.min(0.25, options.relativeTolerance ?? 1e-4));
    this.boundarySmoothingIterations = normalizeOctreeSection43BoundarySmoothing(
      options.boundarySmoothingIterations,
    );
    if (source.leafHeaders.size < this.plan.rowCapacity * 48) throw new RangeError("MGPCG LeafHeader capacity is too small");
    if (source.rowCount.size < 36) {
      throw new RangeError("MGPCG row-count publication must include the exact topology-reuse tail");
    }
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    this.control = device.createBuffer({ label: "Octree MGPCG solve control", size: 64, usage: storage });
    const authorityGateModule = device.createShaderModule({
      label: "Octree MGPCG native L2 authority gate",
      code: octreeMGPCGAuthorityGateShader,
    });
    this.authorityGatePipeline = device.createComputePipeline({
      label: "Reject MGPCG without assembled native L2 rows",
      layout: "auto",
      compute: { module: authorityGateModule, entryPoint: "validateNativeL2Authority" },
    });
    if (this.executionMode === "persistent-small-domain") {
      this.stages = Object.freeze({});
      return;
    }
    const vector = (label: string) => device.createBuffer({ label, size: Math.max(4, this.plan.vectorBytes), usage: storage });
    this.x = vector("Octree MGPCG pressure");
    this.residual = vector("Octree MGPCG residual");
    this.preconditioned = vector("Octree MGPCG preconditioned residual");
    this.direction = vector("Octree MGPCG direction");
    this.product = vector("Octree MGPCG matrix product");
    this.hybridA = vector("Octree Section 4.3 L2 smoother A");
    this.hybridB = vector("Octree Section 4.3 L2 smoother B");
    this.hybridRhs = vector("Octree Section 4.3 L1 residual");
    this.hybridCorrection = vector("Octree Section 4.3 L1 correction");
    this.hybridBandA = vector("Octree Section 4.3 boundary band A");
    this.hybridBandB = vector("Octree Section 4.3 boundary band B");
    this.params = device.createBuffer({ label: "Octree MGPCG parameters", size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

    const words = new Uint32Array(16); const floats = new Float32Array(words.buffer);
    words[0] = options.dimensions[0]; words[1] = options.dimensions[1]; words[2] = options.dimensions[2];
    words[3] = this.plan.rowCapacity; words[7] = this.plan.dispatch[0];
    words[8] = this.iterationBudget;
    // This implementation clamps the relative tolerance to an f32-practical
    // floor. It is an engineering choice, not a tolerance specified by the
    // paper; scene files may inherit much stricter CPU/double settings.
    floats[9] = this.relativeTolerance;
    // This is only a divide-by-zero guard. Operator coefficients and Krylov
    // dot products legitimately become much smaller than 1e-12 when physical
    // cells or open fractions are small.
    floats[10] = 1e-30;
    words[11] = this.boundarySmoothingIterations;
    words[12] = OCTREE_SECTION43_BOUNDARY_BAND_LAYERS;
    floats[13] = 2 / 3;
    device.queue.writeBuffer(this.params, 0, words);

    const shaderModule = device.createShaderModule({
      label: "Octree matrix-free large-domain PCG",
      code: octreeMGPCGShader,
    });
    const pipeline = (entryPoint: string, bindings: readonly number[]): Stage => ({
      pipeline: device.createComputePipeline({ label: `Octree MGPCG · ${entryPoint}`, layout: "auto",
        compute: { module: shaderModule, entryPoint } }),
      bindings,
    });
    this.stages = Object.freeze({
      initialize: pipeline("initializeMGPCG", [0, 1, 2, 3, 4, 5, 6, 7, 11, 12]),
      residual: pipeline("formInitialResidual", [0, 1, 2, 3, 5, 7, 11, 12]),
      clearHybridPreconditioner: pipeline("clearHybridPreconditioner", [0, 3, 11, 13]),
      classifyHybridBand: pipeline("classifyHybridBand", [0, 1, 2, 3, 11, 17]),
      dilateHybridBandAtoB: pipeline("dilateHybridBandAtoB", [0, 1, 2, 3, 11, 17, 18]),
      dilateHybridBandBtoA: pipeline("dilateHybridBandBtoA", [0, 1, 2, 3, 11, 17, 18]),
      smoothHybridAtoB: pipeline("smoothHybridAtoB", [0, 1, 2, 3, 5, 11, 13, 14, 18]),
      smoothHybridBtoA: pipeline("smoothHybridBtoA", [0, 1, 2, 3, 5, 11, 13, 14, 18]),
      formHybridL1Residual: pipeline("formHybridL1Residual", [0, 1, 2, 3, 5, 11, 13, 14, 15]),
      addHybridL1Correction: pipeline("addHybridL1Correction", [0, 3, 11, 13, 14, 16]),
      publishHybridPreconditioner: pipeline("publishHybridPreconditioner", [0, 3, 6, 11, 14]),
      initialReduction: pipeline("initializePCGReduction", [0, 1, 3, 5, 6, 7, 11]),
      preparePCGStep: pipeline("preparePCGStep", [0, 1, 2, 3, 7, 8, 11, 12]),
      updatePCGState: pipeline("updatePCGStateAndReduceResidual", [0, 3, 5, 7, 8, 11, 12]),
      finishPCGIteration: pipeline("finishPCGIteration", [0, 3, 5, 6, 7, 11]),
      finalize: pipeline("finalizeMGPCG", [3, 11]),
      publish: pipeline("publishMGPCG", [0, 3, 4, 10, 11, 12]),
    });
  }

  encode(
    broker: PassBroker,
    pressureIn: GPUBuffer,
    pressureOut: GPUBuffer,
    authorityControl: GPUBuffer,
  ): void {
    this.assertLive();
    if (pressureIn.size < this.plan.vectorBytes || pressureOut.size < this.plan.vectorBytes) {
      throw new RangeError("MGPCG pressure buffer capacity is too small");
    }
    broker.clearBuffer(this.control);
    const authorityGate = broker.compute({ label: "MGPCG native L2 authority gate" });
    authorityGate.setPipeline(this.authorityGatePipeline);
    if (this.authorityGateGroup?.authorityControl !== authorityControl) {
      this.authorityGateGroup = {
        authorityControl,
        group: this.device.createBindGroup({
          layout: this.authorityGatePipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: authorityControl } },
            { binding: 1, resource: { buffer: this.source.rowCount } },
            { binding: 2, resource: { buffer: this.control } },
          ],
        }),
      };
    }
    authorityGate.setBindGroup(0, this.authorityGateGroup.group);
    authorityGate.dispatchWorkgroups(1);
    // The V-cycle publishes live work counts through storage writes. WebGPU
    // requires a pass boundary before those records are copied into an
    // indirect-only buffer, so hierarchy setup deliberately owns its pass.
    this.source.firstOrderVCycle.encodeSetup(broker, {
      solverControl: this.control, rowCount: this.source.rowCount,
    });
    if (this.executionMode === "persistent-small-domain") {
      this.source.firstOrderVCycle.persistentMGPCG!.encodeSolve(broker, {
        leafHeaders: this.source.leafHeaders, leafEntries: this.source.leafEntries,
        rowCount: this.source.rowCount, pressureIn, pressureOut, solverControl: this.control,
        boundarySmoothingIterations: this.boundarySmoothingIterations,
        relativeTolerance: this.relativeTolerance,
      });
      broker.fence("persistent MGPCG pressure publication");
      return;
    }
    const buffers = [this.params, this.source.leafHeaders, this.source.leafEntries, this.source.rowCount,
      pressureIn, this.residual, this.preconditioned, this.direction, this.product, undefined,
      pressureOut, this.control, this.x,
      this.hybridA, this.hybridB, this.hybridRhs, this.hybridCorrection,
      this.hybridBandA, this.hybridBandB] as const;
    const pass = broker.compute({ label: "Octree MGPCG solve" });
    const rows = (stage: Stage) => this.run(pass, stage, this.plan.dispatch, buffers);
    const single = (stage: Stage) => this.run(pass, stage, [1, 1, 1], buffers);

    rows(this.stages.initialize);
    rows(this.stages.residual);
    this.prepareHybridBand(pass, buffers);
    this.applyPreconditioner(broker, buffers);
    single(this.stages.initialReduction);
    for (let iteration = 0;
      iteration < OCTREE_SECTION43_LARGE_DOMAIN_PCG_ITERATIONS;
      iteration += 1) {
      single(this.stages.preparePCGStep);
      single(this.stages.updatePCGState);
      if (iteration + 1 < OCTREE_SECTION43_LARGE_DOMAIN_PCG_ITERATIONS) {
        this.applyPreconditioner(broker, buffers);
        single(this.stages.finishPCGIteration);
      }
    }
    // The final residual gate is authoritative. An unconverged bounded PCG
    // solve is failed closed below; no CPU readback or fallback is involved.
    single(this.stages.finalize);
    rows(this.stages.publish);
    broker.fence("MGPCG pressure publication");
  }

  private applyPreconditioner(broker: PassBroker,
    buffers: readonly (GPUBuffer | undefined)[]): void {
    this.applySection43HybridPreconditioner(broker, buffers);
  }

  private prepareHybridBand(pass: GPUComputePassEncoder, buffers: readonly (GPUBuffer | undefined)[]): void {
    this.run(pass, this.stages.classifyHybridBand, this.plan.dispatch, buffers);
    // Section 4.3 uses a band about three voxels wide. Three deterministic
    // compact-row graph dilations are the first sparse approximation; exact
    // physical-distance classification remains part of the L1 hierarchy work.
    this.run(pass, this.stages.dilateHybridBandAtoB, this.plan.dispatch, buffers);
    this.run(pass, this.stages.dilateHybridBandBtoA, this.plan.dispatch, buffers);
    this.run(pass, this.stages.dilateHybridBandAtoB, this.plan.dispatch, buffers);
  }

  private applySection43HybridPreconditioner(broker: PassBroker,
    buffers: readonly (GPUBuffer | undefined)[]): void {
    const cycle = this.source.firstOrderVCycle;
    const pass = broker.compute();
    this.run(pass, this.stages.clearHybridPreconditioner, this.plan.dispatch, buffers);

    // p0=0; k=8 damped-Jacobi iterations of L2 p=q inside the fixed band.
    for (let i = 0; i < this.boundarySmoothingIterations; i += 1) {
      this.run(pass, i % 2 === 0 ? this.stages.smoothHybridAtoB : this.stages.smoothHybridBtoA,
        this.plan.dispatch, buffers);
    }
    // k is even, so p1 is in A. Form r1=q-L2*p1 and apply the required SPD L1
    // V-cycle. The interface owns its sparse first-order operator/transfers.
    this.run(pass, this.stages.formHybridL1Residual, this.plan.dispatch, buffers);
    const hybridRhs = this.hybridRhs, hybridCorrection = this.hybridCorrection;
    if (!hybridRhs || !hybridCorrection) {
      throw new Error("large-domain MGPCG correction storage was not constructed");
    }
    cycle.encodeCorrection(broker, {
      rhs: hybridRhs, correction: hybridCorrection, solverControl: this.control,
      rowCount: this.source.rowCount,
    });
    // p2=p1+delta starts in B, followed by the matching k post-iterations.
    this.run(pass, this.stages.addHybridL1Correction, this.plan.dispatch, buffers);
    for (let i = 0; i < this.boundarySmoothingIterations; i += 1) {
      this.run(pass, i % 2 === 0 ? this.stages.smoothHybridBtoA : this.stages.smoothHybridAtoB,
        this.plan.dispatch, buffers);
    }
    this.run(pass, this.stages.publishHybridPreconditioner, this.plan.dispatch, buffers);
  }

  private run(pass: GPUComputePassEncoder, stage: Stage, dispatch: readonly [number, number, number],
    buffers: readonly (GPUBuffer | undefined)[]): void {
    // A fixed PCG schedule replays these stages thousands of times. Rebuilding
    // identical bind groups for every dispatch was pure main-thread/driver
    // work; WebGPU bind groups are immutable and safe to retain.
    const variants = this.stageGroups.get(stage);
    const cached = variants?.find((candidate) =>
      stage.bindings.every((binding) => candidate.resources[binding] === buffers[binding]));
    const group = cached?.group ?? this.device.createBindGroup({
      layout: stage.pipeline.getBindGroupLayout(0),
      entries: stage.bindings.map((binding) => ({ binding, resource: { buffer: buffers[binding]! } })),
    });
    if (!cached) {
      const variant = { resources: [...buffers], group };
      if (variants) variants.push(variant); else this.stageGroups.set(stage, [variant]);
    }
    pass.setPipeline(stage.pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(...dispatch);
  }

  private assertLive(): void { if (this.destroyed) throw new Error("Octree MGPCG solver is destroyed"); }
  destroy(): void {
    if (this.destroyed) return; this.destroyed = true;
    this.stageGroups.clear();
    this.authorityGateGroup = undefined;
    this.x?.destroy(); this.residual?.destroy(); this.preconditioned?.destroy(); this.direction?.destroy();
    this.product?.destroy(); this.control.destroy(); this.params?.destroy();
    this.hybridA?.destroy(); this.hybridB?.destroy(); this.hybridRhs?.destroy();
    this.hybridCorrection?.destroy(); this.hybridBandA?.destroy(); this.hybridBandB?.destroy();
  }
}

export const octreeMGPCGAuthorityGateShader = /* wgsl */ `
@group(0) @binding(0) var<storage, read> authorityControl: array<u32>;
@group(0) @binding(1) var<storage, read> rowCounts: array<u32>;
@group(0) @binding(2) var<storage, read_write> solverControl: array<atomic<u32>>;
const ASSEMBLED: u32 = 0x80000000u;
const PROJECTED: u32 = 0x40000000u;
const INVALID_ROW_ERROR: u32 = 1u;
@compute @workgroup_size(1) fn validateNativeL2Authority() {
  if (arrayLength(&authorityControl) < 3u || arrayLength(&rowCounts) == 0u
    || arrayLength(&solverControl) == 0u) {
    if (arrayLength(&solverControl) > 0u) { atomicOr(&solverControl[0], INVALID_ROW_ERROR); }
    return;
  }
  let flags = authorityControl[0];
  let validFlags = (flags & ASSEMBLED) != 0u && (flags & ~(ASSEMBLED | PROJECTED)) == 0u;
  if (!validFlags || authorityControl[2] != rowCounts[0]) {
    atomicOr(&solverControl[0], INVALID_ROW_ERROR);
  }
}
`;

export const octreeMGPCGShader = /* wgsl */ `
struct Params { dimsCapacity:vec4u, hierarchy:vec4u, solve:vec4u, padding:vec4u }
struct LeafHeader { cell:u32,entryStart:u32,entryCount:u32,size:u32,diagonal:f32,rhs:f32,pad0:u32,pad1:u32,gradient:vec4f }
struct LeafEntry { row:u32,coefficient:f32 }
@group(0) @binding(0) var<uniform> params:Params;
@group(0) @binding(1) var<storage,read> headers:array<LeafHeader>;
@group(0) @binding(2) var<storage,read> entries:array<LeafEntry>;
@group(0) @binding(3) var<storage,read_write> counts:array<u32>;
@group(0) @binding(4) var<storage,read> pressureSeed:array<f32>;
@group(0) @binding(5) var<storage,read_write> residual:array<f32>;
@group(0) @binding(6) var<storage,read_write> preconditioned:array<f32>;
@group(0) @binding(7) var<storage,read_write> direction:array<f32>;
@group(0) @binding(8) var<storage,read_write> product:array<f32>;
@group(0) @binding(10) var<storage,read_write> pressureOut:array<f32>;
@group(0) @binding(11) var<storage,read_write> control:array<atomic<u32>>;
@group(0) @binding(12) var<storage,read_write> pressure:array<f32>;
@group(0) @binding(13) var<storage,read_write> hybridA:array<f32>;
@group(0) @binding(14) var<storage,read_write> hybridB:array<f32>;
@group(0) @binding(15) var<storage,read_write> hybridRhs:array<f32>;
@group(0) @binding(16) var<storage,read_write> hybridCorrection:array<f32>;
@group(0) @binding(17) var<storage,read_write> hybridBandA:array<u32>;
@group(0) @binding(18) var<storage,read_write> hybridBandB:array<u32>;
const INVALID_ROW:u32=0xffffffffu;const INVALID_ROW_ERROR:u32=1u;
const NONFINITE:u32=4u;const NONPOSITIVE:u32=8u;const NONCONVERGENCE:u32=16u;
const ROW_BOUNDARY:u32=1u;
fn finite(value:f32)->bool{return value==value&&abs(value)<=3.402823e38;}
fn liveRows()->u32{return min(select(0u,counts[0],arrayLength(&counts)>0u),params.dimsCapacity.w);}
fn rowIndex(gid:vec3u)->u32{return gid.x+gid.y*params.hierarchy.w*64u;}
fn tolerance()->f32{return bitcast<f32>(params.solve.y);}fn epsilon()->f32{return bitcast<f32>(params.solve.z);}
fn hybridOmega()->f32{return bitcast<f32>(params.padding.y);}
fn failed()->bool{return atomicLoad(&control[0])!=0u;}fn stopped()->bool{return failed()||atomicLoad(&control[1])!=0u;}
fn report(flag:u32){atomicOr(&control[0],flag);}
// Words 10-12 are observational diagnostics only: the first failing stage,
// compact row (or INVALID_ROW for a global reduction), and one f32 payload.
// Keeping this beside the existing fail-closed flag lets Dawn identify the
// first bad arithmetic operation without changing solver control flow.
fn reportAt(flag:u32,stage:u32,row:u32,value:f32){
  atomicOr(&control[0],flag);
  for(var retry=0u;retry<16u;retry+=1u){
    let claim=atomicCompareExchangeWeak(&control[10],0u,stage);
    if(claim.exchanged){atomicStore(&control[11],row);atomicStore(&control[12],bitcast<u32>(value));return;}
    if(claim.old_value!=0u){return;}
  }
}
fn fieldValue(row:u32,useDirection:bool)->f32{return select(pressure[row],direction[row],useDirection);}
fn applyA(row:u32,useDirection:bool)->f32{let h=headers[row];if(!finite(h.diagonal)||h.diagonal<=0.0){return 0.0;}
  var value=h.diagonal*fieldValue(row,useDirection);for(var j=0u;j<h.entryCount;j+=1u){let e=entries[h.entryStart+j];if(e.row>=liveRows()||!finite(e.coefficient)){report(INVALID_ROW_ERROR);continue;}value-=e.coefficient*fieldValue(e.row,useDirection);}return value;}
fn hybridValue(row:u32,useB:bool)->f32{return select(hybridA[row],hybridB[row],useB);}
fn applyHybridL2(row:u32,useB:bool)->f32{let h=headers[row];if(!finite(h.diagonal)||h.diagonal<=0.0){report(INVALID_ROW_ERROR);return 0.0;}
  var value=h.diagonal*hybridValue(row,useB);for(var j=0u;j<h.entryCount;j+=1u){let e=entries[h.entryStart+j];
    if(e.row>=liveRows()||!finite(e.coefficient)){report(INVALID_ROW_ERROR);continue;}value-=e.coefficient*hybridValue(e.row,useB);}return value;}
fn smoothHybridValue(row:u32,useB:bool)->f32{let current=hybridValue(row,useB);if(hybridBandB[row]==0u){return current;}
  let diagonal=headers[row].diagonal;let next=current+hybridOmega()*(residual[row]-applyHybridL2(row,useB))/diagonal;
  if(!finite(next)){reportAt(NONFINITE,3u,row,next);return current;}return next;}
@compute @workgroup_size(64) fn initializeMGPCG(@builtin(global_invocation_id) gid:vec3u){let row=rowIndex(gid);if(row>=liveRows()){return;}
  let h=headers[row];if(h.entryStart+h.entryCount>arrayLength(&entries)||!finite(h.diagonal)||h.diagonal<=0.0||!finite(h.rhs)){report(INVALID_ROW_ERROR);return;}
  var seed=pressureSeed[row];if(!finite(seed)){reportAt(NONFINITE,1u,row,seed);seed=0.0;}pressure[row]=seed;residual[row]=0.0;preconditioned[row]=0.0;direction[row]=0.0;}
@compute @workgroup_size(64) fn formInitialResidual(@builtin(global_invocation_id) gid:vec3u){let row=rowIndex(gid);if(row<liveRows()&&!failed()){
  let r=-headers[row].rhs-applyA(row,false);if(!finite(r)){reportAt(NONFINITE,2u,row,r);}else{residual[row]=r;}}}

// Only A must be initialized. The first pre-sweep overwrites every live B
// value, formHybridL1Residual overwrites RHS, and the V-cycle publishes every
// live correction. Clearing four capacity-sized vectors here was dead traffic.
@compute @workgroup_size(64) fn clearHybridPreconditioner(@builtin(global_invocation_id) gid:vec3u){let row=rowIndex(gid);if(row>=liveRows()||stopped()){return;}
  hybridA[row]=0.0;}

@compute @workgroup_size(64) fn classifyHybridBand(@builtin(global_invocation_id) gid:vec3u){let row=rowIndex(gid);if(row>=liveRows()||failed()){return;}
  let h=headers[row];var offDiagonalSum=0.0;var transition=false;
  for(var j=0u;j<h.entryCount;j+=1u){let e=entries[h.entryStart+j];if(e.row>=liveRows()||!finite(e.coefficient)){report(INVALID_ROW_ERROR);continue;}
    offDiagonalSum+=e.coefficient;transition=transition||headers[e.row].size!=h.size;}
  let boundaryGap=h.diagonal-offDiagonalSum;
  let boundary=(h.pad0&ROW_BOUNDARY)!=0u||boundaryGap>1e-5*max(1.0,h.diagonal);
  hybridBandA[row]=select(0u,1u,boundary||transition);}
@compute @workgroup_size(64) fn dilateHybridBandAtoB(@builtin(global_invocation_id) gid:vec3u){let row=rowIndex(gid);if(row>=liveRows()||failed()){return;}
  var bandValue=hybridBandA[row];let h=headers[row];for(var j=0u;j<h.entryCount&&bandValue==0u;j+=1u){let neighbor=entries[h.entryStart+j].row;
    if(neighbor>=liveRows()){report(INVALID_ROW_ERROR);continue;}bandValue=max(bandValue,hybridBandA[neighbor]);}hybridBandB[row]=bandValue;}
@compute @workgroup_size(64) fn dilateHybridBandBtoA(@builtin(global_invocation_id) gid:vec3u){let row=rowIndex(gid);if(row>=liveRows()||failed()){return;}
  var bandValue=hybridBandB[row];let h=headers[row];for(var j=0u;j<h.entryCount&&bandValue==0u;j+=1u){let neighbor=entries[h.entryStart+j].row;
    if(neighbor>=liveRows()){report(INVALID_ROW_ERROR);continue;}bandValue=max(bandValue,hybridBandB[neighbor]);}hybridBandA[row]=bandValue;}
@compute @workgroup_size(64) fn smoothHybridAtoB(@builtin(global_invocation_id) gid:vec3u){let row=rowIndex(gid);if(row<liveRows()&&!stopped()){hybridB[row]=smoothHybridValue(row,false);}}
@compute @workgroup_size(64) fn smoothHybridBtoA(@builtin(global_invocation_id) gid:vec3u){let row=rowIndex(gid);if(row<liveRows()&&!stopped()){hybridA[row]=smoothHybridValue(row,true);}}
@compute @workgroup_size(64) fn formHybridL1Residual(@builtin(global_invocation_id) gid:vec3u){let row=rowIndex(gid);if(row>=liveRows()||stopped()){return;}
  let next=residual[row]-applyHybridL2(row,false);if(!finite(next)){reportAt(NONFINITE,4u,row,next);}else{hybridRhs[row]=next;}}
@compute @workgroup_size(64) fn addHybridL1Correction(@builtin(global_invocation_id) gid:vec3u){let row=rowIndex(gid);if(row>=liveRows()||stopped()){return;}
  let next=hybridA[row]+hybridCorrection[row];if(!finite(next)){reportAt(NONFINITE,5u,row,next);}else{hybridB[row]=next;}}
@compute @workgroup_size(64) fn publishHybridPreconditioner(@builtin(global_invocation_id) gid:vec3u){let row=rowIndex(gid);if(row>=liveRows()||stopped()){return;}
  let value=hybridB[row];if(!finite(value)){reportAt(NONFINITE,6u,row,value);}else{preconditioned[row]=value;}}

var<workgroup> sums:array<vec4f,256>;
@compute @workgroup_size(256) fn initializePCGReduction(@builtin(local_invocation_index) lid:u32){var sum=vec4f(0.0);let n=liveRows();
  if(!failed()){for(var row=lid;row<n;row+=256u){let r=residual[row];let z=preconditioned[row];let b=-headers[row].rhs;
    sum+=vec4f(r*r,b*b,r*z,0.0);direction[row]=z;}}
  sums[lid]=sum;for(var width=128u;width>0u;width>>=1u){workgroupBarrier();if(lid<width){sums[lid]+=sums[lid+width];}}workgroupBarrier();
  if(lid==0u){let v=sums[0];atomicStore(&control[3],n);atomicStore(&control[4],bitcast<u32>(v.x));atomicStore(&control[5],bitcast<u32>(v.y));
    atomicStore(&control[6],bitcast<u32>(v.z));
    let countWords=arrayLength(&counts);if(countWords<8u||counts[countWords-8u]!=0u){report(INVALID_ROW_ERROR);}
    else if(!finite(v.x)||!finite(v.y)||!finite(v.z)||v.z<0.0){reportAt(NONFINITE,7u,INVALID_ROW,v.z);}
    else if(!failed()&&v.x<=tolerance()*tolerance()*max(v.y,epsilon())){atomicStore(&control[1],1u);}}}

// One workgroup owns q=A*d and d·q. This removes the global product/reduction
// dispatch boundary while retaining one deterministic reduction tree.
@compute @workgroup_size(256) fn preparePCGStep(@builtin(local_invocation_index) lid:u32){let solving=!stopped();var dq=0.0;
  if(solving){for(var row=lid;row<liveRows();row+=256u){let q=applyA(row,true);product[row]=q;dq+=direction[row]*q;}}
  sums[lid]=vec4f(dq,0.0,0.0,0.0);for(var width=128u;width>0u;width>>=1u){workgroupBarrier();if(lid<width){sums[lid]+=sums[lid+width];}}
  workgroupBarrier();if(lid==0u&&solving){let total=sums[0].x;let rz=bitcast<f32>(atomicLoad(&control[6]));
    if(!finite(total)||!finite(rz)){reportAt(NONFINITE,8u,INVALID_ROW,total);}
    else if(total<=epsilon()||rz<0.0){report(NONPOSITIVE);}
    else{atomicStore(&control[7],bitcast<u32>(rz/total));atomicStore(&control[9],bitcast<u32>(total));}}}

// The same workgroup updates x/r and immediately reduces the new residual,
// avoiding a row-kernel followed by a singleton reduction.
@compute @workgroup_size(256) fn updatePCGStateAndReduceResidual(@builtin(local_invocation_index) lid:u32){let solving=!stopped();var rr=0.0;
  if(solving){let alpha=bitcast<f32>(atomicLoad(&control[7]));for(var row=lid;row<liveRows();row+=256u){
    let x=pressure[row]+alpha*direction[row];let r=residual[row]-alpha*product[row];
    if(!finite(x)||!finite(r)){reportAt(NONFINITE,9u,row,r);}else{pressure[row]=x;residual[row]=r;rr+=r*r;}}}
  sums[lid]=vec4f(rr,0.0,0.0,0.0);for(var width=128u;width>0u;width>>=1u){workgroupBarrier();if(lid<width){sums[lid]+=sums[lid+width];}}
  workgroupBarrier();if(lid==0u&&solving){let total=sums[0].x;let bb=bitcast<f32>(atomicLoad(&control[5]));
    if(!finite(total)){reportAt(NONFINITE,10u,INVALID_ROW,total);}else{atomicStore(&control[4],bitcast<u32>(total));atomicAdd(&control[2],1u);
      if(total<=tolerance()*tolerance()*max(bb,epsilon())){atomicStore(&control[1],1u);}}}}

// M*r has already been published. Reduce r·z, update the recurrence scalar,
// then update d in the same workgroup after a uniform barrier.
@compute @workgroup_size(256) fn finishPCGIteration(@builtin(local_invocation_index) lid:u32){let solving=!stopped();var rz=0.0;
  if(solving){for(var row=lid;row<liveRows();row+=256u){rz+=residual[row]*preconditioned[row];}}
  sums[lid]=vec4f(rz,0.0,0.0,0.0);for(var width=128u;width>0u;width>>=1u){workgroupBarrier();if(lid<width){sums[lid]+=sums[lid+width];}}
  workgroupBarrier();if(lid==0u&&solving){let next=sums[0].x;let previous=bitcast<f32>(atomicLoad(&control[6]));
    if(!finite(next)||next<0.0||previous<=epsilon()){reportAt(NONFINITE,13u,INVALID_ROW,next);}
    else{atomicStore(&control[8],bitcast<u32>(next/previous));atomicStore(&control[6],bitcast<u32>(next));}}
  workgroupBarrier();if(solving&&!failed()){let beta=bitcast<f32>(atomicLoad(&control[8]));for(var row=lid;row<liveRows();row+=256u){
    let next=preconditioned[row]+beta*direction[row];if(!finite(next)){reportAt(NONFINITE,11u,row,next);}else{direction[row]=next;}}}}
@compute @workgroup_size(1) fn finalizeMGPCG(){if(atomicLoad(&control[1])==0u&&atomicLoad(&control[0])==0u){report(NONCONVERGENCE);}
  if(arrayLength(&counts)>=2u){let tail=arrayLength(&counts);let success=atomicLoad(&control[0])==0u&&atomicLoad(&control[1])!=0u;
    counts[tail-2u]=select(0x7fc00000u,atomicLoad(&control[4]),success);counts[tail-1u]=select(0x7fc00000u,atomicLoad(&control[5]),success);}}
@compute @workgroup_size(64) fn publishMGPCG(@builtin(global_invocation_id) gid:vec3u){let row=rowIndex(gid);let n=liveRows();if(row>=n){return;}
  let success=atomicLoad(&control[0])==0u&&atomicLoad(&control[1])!=0u;let seed=select(0.0,pressureSeed[row],finite(pressureSeed[row]));
  pressureOut[row]=select(seed,pressure[row],success&&finite(pressure[row]));}
`;
