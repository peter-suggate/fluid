/**
 * Matrix-free PCG for compact octree pressure rows with an additive geometric-
 * aggregate diagonal preconditioner.
 *
 * The solver consumes the same LeafHeader/LeafEntry publication used by the
 * power-face projector. Its hierarchy is a sparse collection of independent
 * aggregate diagonal solves built from live row origins on the GPU; no
 * finest-domain field, host row-count readback, or assembled coarse matrix is
 * created.
 *
 * Authoritative power projection injects the sparse first-order V-cycle and
 * selects the Section 4.3 hybrid. The additive aggregate solve remains only as
 * a bounded compatibility/rollback preconditioner for non-authoritative use.
 */

export const OCTREE_MGPCG_PRECONDITIONER_KIND = "additive-geometric-aggregate-diagonal" as const;
export const OCTREE_SECTION43_BOUNDARY_SMOOTHING_ITERATIONS = 8;
export const OCTREE_SECTION43_BOUNDARY_BAND_LAYERS = 3;
// The paper reports convergence in 6-10 PCG iterations for its production
// hierarchy. Our sparse GPU hierarchy is an approximation (notably at the
// graph-ring boundary band), so retain a bounded tail for difficult topology
// generations. Kernels already stop on the GPU as soon as convergence is
// published; this cap bounds the encoded fallback schedule only.
export const OCTREE_SECTION43_MAXIMUM_PCG_ITERATIONS = 128;

export type OctreePCGPreconditionerKind = "aggregate" | "section43-hybrid";

/**
 * The missing middle operation in the paper's Section 4.3 preconditioner.
 * Implementations own their sparse first-order operator, pyramid, and the
 * GhostValuePropagate/GhostValueAccumulate transfers described in Section 4.2.
 * They must encode an SPD linear correction from rhs to correction and report
 * any GPU-side failure through solverControl. No implementation is supplied by
 * the aggregate-PCG path, so the hybrid cannot be selected accidentally.
 */
export interface OctreeFirstOrderSPDVCycle {
  readonly operatorOrder: 1;
  readonly isSymmetricPositiveDefinite: true;
  readonly allocatedBytes: number;
  encodeSetup(encoder: GPUCommandEncoder, input: {
    readonly solverControl: GPUBuffer;
    readonly rowCount: GPUBuffer;
  }): void;
  encodeCorrection(encoder: GPUCommandEncoder, input: {
    readonly rhs: GPUBuffer;
    readonly correction: GPUBuffer;
    readonly solverControl: GPUBuffer;
    readonly rowCount: GPUBuffer;
  }): void;
}

export const OCTREE_MGPCG_ERROR = Object.freeze({
  invalidRow: 1 << 0,
  hierarchyOverflow: 1 << 1,
  nonFinite: 1 << 2,
  nonPositiveOperator: 1 << 3,
  nonConvergence: 1 << 4,
} as const);

export interface OctreeMGPCGPlan {
  readonly preconditionerKind: OctreePCGPreconditionerKind;
  readonly rowCapacity: number;
  readonly hierarchyLevelCount: number;
  readonly hierarchyStride: number;
  readonly baseAggregateSize: number;
  readonly dispatch: readonly [number, number, number];
  readonly vectorBytes: number;
  readonly hierarchyBytes: number;
  readonly hybridBytes: number;
  readonly allocatedBytes: number;
}

export interface OctreeMGPCGOptions {
  readonly dimensions: readonly [number, number, number];
  readonly rowCapacity: number;
  readonly maximumLeafSize: number;
  readonly maximumIterations: number;
  readonly relativeTolerance?: number;
  readonly maximumHierarchyLevels?: number;
  /** Hybrid requires an explicit SPD L1 V-cycle; aggregate is rollback only. */
  readonly preconditionerKind?: OctreePCGPreconditionerKind;
}

export interface OctreeMGPCGSource {
  /** LeafHeader array; word zero is the flattened finest-cell origin. */
  readonly leafHeaders: GPUBuffer;
  /** LeafEntry { row:u32, coefficient:f32 } array. */
  readonly leafEntries: GPUBuffer;
  /** Compact publication whose first u32 is the live row count. */
  readonly rowCount: GPUBuffer;
  readonly firstOrderVCycle?: OctreeFirstOrderSPDVCycle;
}

function positiveInteger(value: number, label: string): number {
  if (!Number.isSafeInteger(value) || value < 1) throw new RangeError(`${label} must be a positive integer`);
  return value;
}

export function planOctreeMGPCG(options: Pick<OctreeMGPCGOptions,
  "dimensions" | "rowCapacity" | "maximumLeafSize" | "maximumHierarchyLevels" | "preconditionerKind">): OctreeMGPCGPlan {
  const rowCapacity = positiveInteger(options.rowCapacity, "MGPCG row capacity");
  const [nx, ny, nz] = options.dimensions.map((value) => positiveInteger(value, "MGPCG dimension")) as [number, number, number];
  const maximumLeafSize = positiveInteger(options.maximumLeafSize, "MGPCG maximum leaf size");
  const baseAggregateSize = Math.min(1 << 30, maximumLeafSize * 2);
  const width = Math.max(nx, ny, nz);
  const requiredLevels = Math.max(1, Math.ceil(Math.log2(Math.max(1, width / baseAggregateSize))) + 1);
  const hierarchyLevelCount = Math.min(Math.max(1, options.maximumHierarchyLevels ?? 12), requiredLevels);
  // One slot per possible row is sufficient because every aggregate owns at
  // least one row.  Level-zero starts above the maximum leaf width, keeping
  // the hash comfortably below full occupancy in ordinary graded trees.
  const hierarchyStride = rowCapacity;
  const blocks = Math.ceil(rowCapacity / 64);
  const dispatchX = Math.min(65_535, Math.max(1, blocks));
  const dispatchY = Math.max(1, Math.ceil(blocks / dispatchX));
  const vectorBytes = rowCapacity * 4;
  const preconditionerKind = options.preconditionerKind ?? "aggregate";
  // maps, hash keys, aggregate diagonals, aggregate RHS, additive correction.
  const hierarchyBytes = preconditionerKind === "aggregate"
    ? hierarchyLevelCount * hierarchyStride * 5 * 4 : 0;
  // solution ping/pong, first-order RHS/correction, and two u32 band masks.
  const hybridBytes = preconditionerKind === "section43-hybrid" ? vectorBytes * 6 : 0;
  return {
    preconditionerKind, rowCapacity, hierarchyLevelCount, hierarchyStride,
    baseAggregateSize, dispatch: [dispatchX, dispatchY, 1], vectorBytes,
    hierarchyBytes, hybridBytes,
    allocatedBytes: vectorBytes * 5 + hierarchyBytes + hybridBytes + 64 + 64,
  };
}

type Stage = { readonly pipeline: GPUComputePipeline; readonly bindings: readonly number[] };
type CachedStageGroup = {
  readonly resources: readonly (GPUBuffer | undefined)[];
  readonly group: GPUBindGroup;
};

/**
 * Fixed-capacity GPU solver.  encode() performs a fixed host-authored maximum
 * iteration schedule, while every kernel exits from a GPU convergence word;
 * convergence never introduces a CPU scheduling decision or readback.
 */
export class WebGPUOctreeMGPCG {
  readonly plan: OctreeMGPCGPlan;
  readonly control: GPUBuffer;
  private readonly params: GPUBuffer;
  private readonly x: GPUBuffer;
  private readonly residual: GPUBuffer;
  private readonly preconditioned: GPUBuffer;
  private readonly direction: GPUBuffer;
  private readonly product: GPUBuffer;
  private readonly hierarchy: GPUBuffer;
  private readonly hybridA?: GPUBuffer;
  private readonly hybridB?: GPUBuffer;
  private readonly hybridRhs?: GPUBuffer;
  private readonly hybridCorrection?: GPUBuffer;
  private readonly hybridBandA?: GPUBuffer;
  private readonly hybridBandB?: GPUBuffer;
  private readonly stages: Readonly<Record<string, Stage>>;
  /** Immutable descriptors shared by every replay of a stage. */
  private readonly stageGroups = new Map<Stage, CachedStageGroup>();
  private readonly maximumIterations: number;
  private readonly device: GPUDevice;
  private destroyed = false;

  constructor(device: GPUDevice, private readonly source: OctreeMGPCGSource, options: OctreeMGPCGOptions) {
    this.device = device;
    this.plan = planOctreeMGPCG(options);
    if (this.plan.preconditionerKind === "section43-hybrid") {
      const cycle = source.firstOrderVCycle;
      if (!cycle || cycle.operatorOrder !== 1 || cycle.isSymmetricPositiveDefinite !== true) {
        throw new Error("Section 4.3 hybrid PCG requires an explicit SPD first-order V-cycle");
      }
    }
    const requestedIterations = Math.max(1, Math.min(1_000, Math.round(options.maximumIterations)));
    this.maximumIterations = this.plan.preconditionerKind === "section43-hybrid"
      ? Math.min(OCTREE_SECTION43_MAXIMUM_PCG_ITERATIONS, requestedIterations)
      : requestedIterations;
    if (source.leafHeaders.size < this.plan.rowCapacity * 48) throw new RangeError("MGPCG LeafHeader capacity is too small");
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    const vector = (label: string) => device.createBuffer({ label, size: Math.max(4, this.plan.vectorBytes), usage: storage });
    this.x = vector("Octree MGPCG pressure");
    this.residual = vector("Octree MGPCG residual");
    this.preconditioned = vector("Octree MGPCG preconditioned residual");
    this.direction = vector("Octree MGPCG direction");
    this.product = vector("Octree MGPCG matrix product");
    this.hierarchy = device.createBuffer({ label: "Octree MGPCG sparse hierarchy", size: Math.max(4, this.plan.hierarchyBytes), usage: storage });
    if (this.plan.preconditionerKind === "section43-hybrid") {
      this.hybridA = vector("Octree Section 4.3 L2 smoother A");
      this.hybridB = vector("Octree Section 4.3 L2 smoother B");
      this.hybridRhs = vector("Octree Section 4.3 L1 residual");
      this.hybridCorrection = vector("Octree Section 4.3 L1 correction");
      this.hybridBandA = vector("Octree Section 4.3 boundary band A");
      this.hybridBandB = vector("Octree Section 4.3 boundary band B");
    }
    this.control = device.createBuffer({ label: "Octree MGPCG solve control", size: 64, usage: storage });
    this.params = device.createBuffer({ label: "Octree MGPCG parameters", size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

    const words = new Uint32Array(16); const floats = new Float32Array(words.buffer);
    words[0] = options.dimensions[0]; words[1] = options.dimensions[1]; words[2] = options.dimensions[2];
    words[3] = this.plan.rowCapacity; words[4] = this.plan.hierarchyLevelCount;
    words[5] = this.plan.hierarchyStride; words[6] = this.plan.baseAggregateSize; words[7] = this.plan.dispatch[0];
    words[8] = this.maximumIterations;
    // This implementation clamps the relative tolerance to an f32-practical
    // floor. It is an engineering choice, not a tolerance specified by the
    // paper; scene files may inherit much stricter CPU/double settings.
    floats[9] = Math.max(1e-4, Math.min(0.25, options.relativeTolerance ?? 1e-4));
    // This is only a divide-by-zero guard. Operator coefficients and Krylov
    // dot products legitimately become much smaller than 1e-12 when physical
    // cells or open fractions are small.
    floats[10] = 1e-30;
    words[11] = OCTREE_SECTION43_BOUNDARY_SMOOTHING_ITERATIONS;
    words[12] = OCTREE_SECTION43_BOUNDARY_BAND_LAYERS;
    floats[13] = 2 / 3;
    device.queue.writeBuffer(this.params, 0, words);

    const module = device.createShaderModule({ label: "Octree matrix-free PCG", code: octreeMGPCGShader });
    const pipeline = (entryPoint: string, bindings: readonly number[]): Stage => ({
      pipeline: device.createComputePipeline({ label: `Octree MGPCG · ${entryPoint}`, layout: "auto", compute: { module, entryPoint } }),
      bindings,
    });
    this.stages = Object.freeze({
      initialize: pipeline("initializeMGPCG", [0, 1, 2, 3, 4, 5, 6, 7, 11, 12]),
      multiplyX: pipeline("multiplyX", [0, 1, 2, 3, 7, 11, 12]),
      residual: pipeline("formInitialResidual", [0, 1, 3, 5, 7, 11]),
      hierarchyMap: pipeline("buildHierarchyMap", [0, 1, 3, 9, 11]),
      hierarchyDiagonal: pipeline("buildHierarchyDiagonal", [0, 1, 2, 3, 9, 11]),
      preconditionFine: pipeline("preconditionFine", [0, 1, 3, 5, 6, 11]),
      restrict: pipeline("restrictResidual", [0, 3, 5, 9, 11]),
      coarse: pipeline("solveCoarseAggregates", [0, 9, 11]),
      prolong: pipeline("prolongateCorrection", [0, 3, 6, 9, 11]),
      classifyHybridBand: pipeline("classifyHybridBand", [0, 1, 2, 3, 11, 17]),
      dilateHybridBandAtoB: pipeline("dilateHybridBandAtoB", [0, 1, 2, 3, 11, 17, 18]),
      dilateHybridBandBtoA: pipeline("dilateHybridBandBtoA", [0, 1, 2, 3, 11, 17, 18]),
      smoothHybridAtoB: pipeline("smoothHybridAtoB", [0, 1, 2, 3, 5, 11, 13, 14, 18]),
      smoothHybridBtoA: pipeline("smoothHybridBtoA", [0, 1, 2, 3, 5, 11, 13, 14, 18]),
      formHybridL1Residual: pipeline("formHybridL1Residual", [0, 1, 2, 3, 5, 11, 13, 14, 15]),
      addHybridL1Correction: pipeline("addHybridL1Correction", [0, 3, 11, 13, 14, 16]),
      publishHybridPreconditioner: pipeline("publishHybridPreconditioner", [0, 3, 6, 11, 14]),
      initialReduction: pipeline("reduceInitialState", [0, 1, 3, 5, 6, 7, 11]),
      multiplyDirection: pipeline("multiplyDirection", [0, 1, 2, 3, 7, 8, 11, 12]),
      directionReduction: pipeline("reduceDirectionProduct", [0, 3, 7, 8, 11]),
      update: pipeline("updatePressureResidual", [0, 3, 5, 7, 8, 11, 12]),
      updatedResidualReduction: pipeline("reduceUpdatedResidual", [0, 3, 5, 11]),
      nextReduction: pipeline("reduceNextState", [0, 3, 5, 6, 11]),
      direction: pipeline("updateDirection", [0, 3, 6, 7, 11]),
      finalize: pipeline("finalizeMGPCG", [3, 11]),
      publish: pipeline("publishMGPCG", [0, 3, 4, 10, 11, 12]),
    });
  }

  encode(encoder: GPUCommandEncoder, pressureIn: GPUBuffer, pressureOut: GPUBuffer): void {
    this.assertLive();
    if (pressureIn.size < this.plan.vectorBytes || pressureOut.size < this.plan.vectorBytes) {
      throw new RangeError("MGPCG pressure buffer capacity is too small");
    }
    encoder.clearBuffer(this.control);
    encoder.clearBuffer(this.hierarchy);
    const buffers = [this.params, this.source.leafHeaders, this.source.leafEntries, this.source.rowCount,
      pressureIn, this.residual, this.preconditioned, this.direction, this.product, this.hierarchy,
      pressureOut, this.control, this.x,
      this.hybridA, this.hybridB, this.hybridRhs, this.hybridCorrection,
      this.hybridBandA, this.hybridBandB] as const;
    const rows = (stage: Stage) => this.run(encoder, stage, this.plan.dispatch, buffers);
    const single = (stage: Stage) => this.run(encoder, stage, [1, 1, 1], buffers);

    rows(this.stages.initialize);
    rows(this.stages.multiplyX);
    rows(this.stages.residual);
    if (this.plan.preconditionerKind === "section43-hybrid") {
      this.source.firstOrderVCycle!.encodeSetup(encoder, {
        solverControl: this.control, rowCount: this.source.rowCount,
      });
      this.prepareHybridBand(encoder, buffers);
    } else {
      rows(this.stages.hierarchyMap);
      rows(this.stages.hierarchyDiagonal);
    }
    this.applyPreconditioner(encoder, buffers);
    single(this.stages.initialReduction);
    for (let iteration = 0; iteration < this.maximumIterations; iteration += 1) {
      rows(this.stages.multiplyDirection);
      single(this.stages.directionReduction);
      rows(this.stages.update);
      // Standard PCG tests the updated residual before applying M again.  In
      // particular, do not let a numerically exhausted r enter the fixed SPD
      // Section 4.3 V-cycle and turn an already-converged solve into failure.
      single(this.stages.updatedResidualReduction);
      this.applyPreconditioner(encoder, buffers);
      single(this.stages.nextReduction);
      rows(this.stages.direction);
    }
    single(this.stages.finalize);
    rows(this.stages.publish);
  }

  private applyPreconditioner(encoder: GPUCommandEncoder, buffers: readonly (GPUBuffer | undefined)[]): void {
    if (this.plan.preconditionerKind === "section43-hybrid") {
      this.applySection43HybridPreconditioner(encoder, buffers);
      return;
    }
    const levelWords = this.plan.hierarchyLevelCount * this.plan.hierarchyStride;
    // Keep maps, keys, and aggregate diagonals; reset only the additive
    // aggregate RHS/corrections for this PCG preconditioner application. This
    // is not restriction/coarse solve/prolongation in a multigrid V-cycle.
    encoder.clearBuffer(this.hierarchy, levelWords * 3 * 4, levelWords * 2 * 4);
    this.run(encoder, this.stages.preconditionFine, this.plan.dispatch, buffers);
    this.run(encoder, this.stages.restrict, this.plan.dispatch, buffers);
    this.run(encoder, this.stages.coarse, this.plan.dispatch, buffers);
    this.run(encoder, this.stages.prolong, this.plan.dispatch, buffers);
  }

  private prepareHybridBand(encoder: GPUCommandEncoder, buffers: readonly (GPUBuffer | undefined)[]): void {
    this.requireHybridBuffers();
    this.run(encoder, this.stages.classifyHybridBand, this.plan.dispatch, buffers);
    // Section 4.3 uses a band about three voxels wide. Three deterministic
    // compact-row graph dilations are the first sparse approximation; exact
    // physical-distance classification remains part of the L1 hierarchy work.
    this.run(encoder, this.stages.dilateHybridBandAtoB, this.plan.dispatch, buffers);
    this.run(encoder, this.stages.dilateHybridBandBtoA, this.plan.dispatch, buffers);
    this.run(encoder, this.stages.dilateHybridBandAtoB, this.plan.dispatch, buffers);
  }

  private applySection43HybridPreconditioner(encoder: GPUCommandEncoder,
    buffers: readonly (GPUBuffer | undefined)[]): void {
    const cycle = this.source.firstOrderVCycle;
    const { hybridA, hybridB, hybridRhs, hybridCorrection } = this.requireHybridBuffers();
    encoder.clearBuffer(hybridA); encoder.clearBuffer(hybridB);
    encoder.clearBuffer(hybridRhs); encoder.clearBuffer(hybridCorrection);

    // p0=0; k=8 damped-Jacobi iterations of L2 p=q inside the fixed band.
    for (let i = 0; i < OCTREE_SECTION43_BOUNDARY_SMOOTHING_ITERATIONS; i += 1) {
      this.run(encoder, i % 2 === 0 ? this.stages.smoothHybridAtoB : this.stages.smoothHybridBtoA,
        this.plan.dispatch, buffers);
    }
    // k is even, so p1 is in A. Form r1=q-L2*p1 and apply the required SPD L1
    // V-cycle. The interface owns its sparse first-order operator/transfers.
    this.run(encoder, this.stages.formHybridL1Residual, this.plan.dispatch, buffers);
    cycle!.encodeCorrection(encoder, {
      rhs: hybridRhs, correction: hybridCorrection, solverControl: this.control,
      rowCount: this.source.rowCount,
    });
    // p2=p1+delta starts in B, followed by the matching k post-iterations.
    this.run(encoder, this.stages.addHybridL1Correction, this.plan.dispatch, buffers);
    for (let i = 0; i < OCTREE_SECTION43_BOUNDARY_SMOOTHING_ITERATIONS; i += 1) {
      this.run(encoder, i % 2 === 0 ? this.stages.smoothHybridBtoA : this.stages.smoothHybridAtoB,
        this.plan.dispatch, buffers);
    }
    this.run(encoder, this.stages.publishHybridPreconditioner, this.plan.dispatch, buffers);
  }

  private requireHybridBuffers(): {
    hybridA: GPUBuffer; hybridB: GPUBuffer; hybridRhs: GPUBuffer;
    hybridCorrection: GPUBuffer; hybridBandA: GPUBuffer; hybridBandB: GPUBuffer;
  } {
    if (!this.hybridA || !this.hybridB || !this.hybridRhs || !this.hybridCorrection
      || !this.hybridBandA || !this.hybridBandB) {
      throw new Error("Section 4.3 hybrid buffers are unavailable");
    }
    return { hybridA: this.hybridA, hybridB: this.hybridB, hybridRhs: this.hybridRhs,
      hybridCorrection: this.hybridCorrection, hybridBandA: this.hybridBandA,
      hybridBandB: this.hybridBandB };
  }

  private run(encoder: GPUCommandEncoder, stage: Stage, dispatch: readonly [number, number, number], buffers: readonly (GPUBuffer | undefined)[]): void {
    // A fixed PCG schedule replays these stages thousands of times. Rebuilding
    // identical bind groups for every dispatch was pure main-thread/driver
    // work; WebGPU bind groups are immutable and safe to retain.
    const cached = this.stageGroups.get(stage);
    const unchanged = cached !== undefined
      && stage.bindings.every((binding) => cached.resources[binding] === buffers[binding]);
    const group = unchanged ? cached.group : this.device.createBindGroup({
      layout: stage.pipeline.getBindGroupLayout(0),
      entries: stage.bindings.map((binding) => ({ binding, resource: { buffer: buffers[binding]! } })),
    });
    if (!unchanged) this.stageGroups.set(stage, { resources: [...buffers], group });
    const pass = encoder.beginComputePass({ label: stage.pipeline.label });
    pass.setPipeline(stage.pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(...dispatch); pass.end();
  }

  private assertLive(): void { if (this.destroyed) throw new Error("Octree MGPCG solver is destroyed"); }
  get iterationBudget(): number { return this.maximumIterations; }
  destroy(): void {
    if (this.destroyed) return; this.destroyed = true;
    this.stageGroups.clear();
    this.x.destroy(); this.residual.destroy(); this.preconditioned.destroy(); this.direction.destroy();
    this.product.destroy(); this.hierarchy.destroy(); this.control.destroy(); this.params.destroy();
    this.hybridA?.destroy(); this.hybridB?.destroy(); this.hybridRhs?.destroy();
    this.hybridCorrection?.destroy(); this.hybridBandA?.destroy(); this.hybridBandB?.destroy();
  }
}

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
@group(0) @binding(9) var<storage,read_write> hierarchy:array<atomic<u32>>;
@group(0) @binding(10) var<storage,read_write> pressureOut:array<f32>;
@group(0) @binding(11) var<storage,read_write> control:array<atomic<u32>>;
@group(0) @binding(12) var<storage,read_write> pressure:array<f32>;
@group(0) @binding(13) var<storage,read_write> hybridA:array<f32>;
@group(0) @binding(14) var<storage,read_write> hybridB:array<f32>;
@group(0) @binding(15) var<storage,read_write> hybridRhs:array<f32>;
@group(0) @binding(16) var<storage,read_write> hybridCorrection:array<f32>;
@group(0) @binding(17) var<storage,read_write> hybridBandA:array<u32>;
@group(0) @binding(18) var<storage,read_write> hybridBandB:array<u32>;

const INVALID_ROW:u32=0xffffffffu;const INVALID_ROW_ERROR:u32=1u;const HASH_OVERFLOW:u32=2u;
const NONFINITE:u32=4u;const NONPOSITIVE:u32=8u;const NONCONVERGENCE:u32=16u;
fn finite(value:f32)->bool{return value==value&&abs(value)<=3.402823e38;}
fn liveRows()->u32{return min(select(0u,counts[0],arrayLength(&counts)>0u),params.dimsCapacity.w);}
fn rowIndex(gid:vec3u)->u32{return gid.x+gid.y*params.hierarchy.w*64u;}
fn levels()->u32{return params.hierarchy.x;}fn stride()->u32{return params.hierarchy.y;}
fn tolerance()->f32{return bitcast<f32>(params.solve.y);}fn epsilon()->f32{return bitcast<f32>(params.solve.z);}
fn hybridOmega()->f32{return bitcast<f32>(params.padding.y);}
fn mapBase()->u32{return 0u;}fn keyBase()->u32{return levels()*stride();}
fn diagonalBase()->u32{return 2u*levels()*stride();}fn rhsBase()->u32{return 3u*levels()*stride();}
fn solutionBase()->u32{return 4u*levels()*stride();}fn offset(base:u32,level:u32,slot:u32)->u32{return base+level*stride()+slot;}
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
fn atomicAddFloat(at:u32,value:f32){if(!finite(value)){report(NONFINITE);return;}var old=atomicLoad(&hierarchy[at]);
  loop{let current=bitcast<f32>(old);if(!finite(current)){report(NONFINITE);return;}
    let exchange=atomicCompareExchangeWeak(&hierarchy[at],old,bitcast<u32>(current+value));if(exchange.exchanged){return;}old=exchange.old_value;}}
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
fn cellCoord(cell:u32)->vec3u{let nx=params.dimsCapacity.x;let ny=params.dimsCapacity.y;return vec3u(cell%nx,(cell/nx)%ny,cell/(nx*ny));}
fn aggregateKey(row:u32,level:u32)->u32{let block=params.hierarchy.z<<level;let q=cellCoord(headers[row].cell)/block;
  let d=(params.dimsCapacity.xyz+vec3u(block-1u))/block;return q.x+d.x*(q.y+d.y*q.z);}
fn hashKey(key:u32)->u32{var h=key*0x9e3779b1u;h=(h^(h>>16u))*0x7feb352du;return (h^(h>>15u))%stride();}
fn findAggregate(row:u32,level:u32)->u32{let key=aggregateKey(row,level)+1u;var slot=hashKey(key);
  for(var probe=0u;probe<64u;probe+=1u){let at=offset(keyBase(),level,slot);var observed=atomicLoad(&hierarchy[at]);
    for(var retry=0u;retry<16u;retry+=1u){if(observed==key){return slot;}if(observed!=0u){break;}let claim=atomicCompareExchangeWeak(&hierarchy[at],0u,key);
      if(claim.exchanged){return slot;}observed=claim.old_value;if(retry==15u&&observed==0u){report(HASH_OVERFLOW);return INVALID_ROW;}}
    slot=(slot+1u)%stride();}report(HASH_OVERFLOW);return INVALID_ROW;}

@compute @workgroup_size(64) fn initializeMGPCG(@builtin(global_invocation_id) gid:vec3u){let row=rowIndex(gid);if(row>=liveRows()){return;}
  let h=headers[row];if(h.entryStart+h.entryCount>arrayLength(&entries)||!finite(h.diagonal)||h.diagonal<=0.0||!finite(h.rhs)){report(INVALID_ROW_ERROR);return;}
  var seed=pressureSeed[row];if(!finite(seed)){reportAt(NONFINITE,1u,row,seed);seed=0.0;}pressure[row]=seed;residual[row]=0.0;preconditioned[row]=0.0;direction[row]=0.0;}
@compute @workgroup_size(64) fn multiplyX(@builtin(global_invocation_id) gid:vec3u){let row=rowIndex(gid);if(row<liveRows()&&!failed()){direction[row]=applyA(row,false);}}
@compute @workgroup_size(64) fn formInitialResidual(@builtin(global_invocation_id) gid:vec3u){let row=rowIndex(gid);if(row<liveRows()&&!failed()){let r=-headers[row].rhs-direction[row];if(!finite(r)){reportAt(NONFINITE,2u,row,r);}else{residual[row]=r;}}}

@compute @workgroup_size(64) fn buildHierarchyMap(@builtin(global_invocation_id) gid:vec3u){let row=rowIndex(gid);if(row>=liveRows()||failed()){return;}
  for(var level=0u;level<levels();level+=1u){let slot=findAggregate(row,level);atomicStore(&hierarchy[offset(mapBase(),level,row)],slot);}}
@compute @workgroup_size(64) fn buildHierarchyDiagonal(@builtin(global_invocation_id) gid:vec3u){let row=rowIndex(gid);if(row>=liveRows()||failed()){return;}let h=headers[row];
  for(var level=0u;level<levels();level+=1u){let slot=atomicLoad(&hierarchy[offset(mapBase(),level,row)]);if(slot==INVALID_ROW){continue;}var value=h.diagonal;
    for(var j=0u;j<h.entryCount;j+=1u){let e=entries[h.entryStart+j];if(e.row>=liveRows()){report(INVALID_ROW_ERROR);continue;}
      if(atomicLoad(&hierarchy[offset(mapBase(),level,e.row)])==slot){value-=e.coefficient;}}
    atomicAddFloat(offset(diagonalBase(),level,slot),value);}}

@compute @workgroup_size(64) fn preconditionFine(@builtin(global_invocation_id) gid:vec3u){let row=rowIndex(gid);if(row>=liveRows()||stopped()){return;}
  let diagonal=headers[row].diagonal;preconditioned[row]=select(0.0,residual[row]/diagonal,diagonal>epsilon());}
@compute @workgroup_size(64) fn restrictResidual(@builtin(global_invocation_id) gid:vec3u){let row=rowIndex(gid);if(row>=liveRows()||stopped()){return;}
  for(var level=0u;level<levels();level+=1u){let slot=atomicLoad(&hierarchy[offset(mapBase(),level,row)]);if(slot!=INVALID_ROW){atomicAddFloat(offset(rhsBase(),level,slot),residual[row]);}}}
@compute @workgroup_size(64) fn solveCoarseAggregates(@builtin(global_invocation_id) gid:vec3u){let slot=rowIndex(gid);if(slot>=stride()||stopped()){return;}
  for(var level=0u;level<levels();level+=1u){let key=atomicLoad(&hierarchy[offset(keyBase(),level,slot)]);if(key==0u){continue;}
    let diagonal=bitcast<f32>(atomicLoad(&hierarchy[offset(diagonalBase(),level,slot)]));let rhs=bitcast<f32>(atomicLoad(&hierarchy[offset(rhsBase(),level,slot)]));
    let correction=select(0.0,rhs/diagonal,finite(diagonal)&&diagonal>epsilon());atomicStore(&hierarchy[offset(solutionBase(),level,slot)],bitcast<u32>(correction));}}
@compute @workgroup_size(64) fn prolongateCorrection(@builtin(global_invocation_id) gid:vec3u){let row=rowIndex(gid);if(row>=liveRows()||stopped()){return;}var value=preconditioned[row];
  let scale=1.0/f32(levels()+1u);value*=scale;for(var level=0u;level<levels();level+=1u){let slot=atomicLoad(&hierarchy[offset(mapBase(),level,row)]);
    if(slot!=INVALID_ROW){value+=scale*bitcast<f32>(atomicLoad(&hierarchy[offset(solutionBase(),level,slot)]));}}
  if(!finite(value)){reportAt(NONFINITE,12u,row,value);}else{preconditioned[row]=value;}}

@compute @workgroup_size(64) fn classifyHybridBand(@builtin(global_invocation_id) gid:vec3u){let row=rowIndex(gid);if(row>=liveRows()||failed()){return;}
  let h=headers[row];var offDiagonalSum=0.0;var transition=false;
  for(var j=0u;j<h.entryCount;j+=1u){let e=entries[h.entryStart+j];if(e.row>=liveRows()||!finite(e.coefficient)){report(INVALID_ROW_ERROR);continue;}
    offDiagonalSum+=e.coefficient;transition=transition||headers[e.row].size!=h.size;}
  let boundaryGap=h.diagonal-offDiagonalSum;let boundary=boundaryGap>1e-5*max(1.0,h.diagonal);
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
@compute @workgroup_size(256) fn reduceInitialState(@builtin(local_invocation_index) lid:u32){var sum=vec4f(0.0);let n=liveRows();
  for(var row=lid;row<n;row+=256u){let r=residual[row];let z=preconditioned[row];let b=-headers[row].rhs;sum+=vec4f(r*r,b*b,r*z,0.0);direction[row]=z;}
  sums[lid]=sum;for(var width=128u;width>0u;width>>=1u){workgroupBarrier();if(lid<width){sums[lid]+=sums[lid+width];}}workgroupBarrier();
  if(lid==0u){let v=sums[0];atomicStore(&control[3],n);atomicStore(&control[4],bitcast<u32>(v.x));atomicStore(&control[5],bitcast<u32>(v.y));atomicStore(&control[6],bitcast<u32>(v.z));
    let countWords=arrayLength(&counts);if(countWords<8u||counts[countWords-8u]!=0u){report(INVALID_ROW_ERROR);}
    else if(!finite(v.x)||!finite(v.y)||!finite(v.z)||v.z<0.0){reportAt(NONFINITE,7u,INVALID_ROW,v.z);}else if(!failed()&&v.x<=tolerance()*tolerance()*max(v.y,epsilon())){atomicStore(&control[1],1u);}}}

@compute @workgroup_size(64) fn multiplyDirection(@builtin(global_invocation_id) gid:vec3u){let row=rowIndex(gid);if(row<liveRows()&&!stopped()){product[row]=applyA(row,true);}}
@compute @workgroup_size(256) fn reduceDirectionProduct(@builtin(local_invocation_index) lid:u32){let solving=!stopped();var sum=0.0;
  if(solving){for(var row=lid;row<liveRows();row+=256u){sum+=direction[row]*product[row];}}sums[lid]=vec4f(sum,0.0,0.0,0.0);
  for(var width=128u;width>0u;width>>=1u){workgroupBarrier();if(lid<width){sums[lid]+=sums[lid+width];}}workgroupBarrier();if(lid==0u){let dq=sums[0].x;let rz=bitcast<f32>(atomicLoad(&control[6]));
    if(!solving){}else if(!finite(dq)||!finite(rz)){reportAt(NONFINITE,8u,INVALID_ROW,dq);}else if(dq<=epsilon()||rz<0.0){report(NONPOSITIVE);}else{atomicStore(&control[7],bitcast<u32>(rz/dq));atomicStore(&control[9],bitcast<u32>(dq));}}}
@compute @workgroup_size(64) fn updatePressureResidual(@builtin(global_invocation_id) gid:vec3u){let row=rowIndex(gid);if(row>=liveRows()||stopped()){return;}let alpha=bitcast<f32>(atomicLoad(&control[7]));
  let nextPressure=pressure[row]+alpha*direction[row];let nextResidual=residual[row]-alpha*product[row];if(!finite(nextPressure)||!finite(nextResidual)){reportAt(NONFINITE,9u,row,nextResidual);return;}pressure[row]=nextPressure;residual[row]=nextResidual;}
@compute @workgroup_size(256) fn reduceUpdatedResidual(@builtin(local_invocation_index) lid:u32){let solving=!stopped();var sum=0.0;
  if(solving){for(var row=lid;row<liveRows();row+=256u){let r=residual[row];sum+=r*r;}}sums[lid]=vec4f(sum,0.0,0.0,0.0);
  for(var width=128u;width>0u;width>>=1u){workgroupBarrier();if(lid<width){sums[lid]+=sums[lid+width];}}workgroupBarrier();if(lid==0u){let rr=sums[0].x;let bb=bitcast<f32>(atomicLoad(&control[5]));
    if(!solving){return;}if(!finite(rr)){reportAt(NONFINITE,10u,INVALID_ROW,rr);return;}atomicStore(&control[4],bitcast<u32>(rr));atomicAdd(&control[2],1u);
    if(rr<=tolerance()*tolerance()*max(bb,epsilon())){atomicStore(&control[1],1u);}}}
@compute @workgroup_size(256) fn reduceNextState(@builtin(local_invocation_index) lid:u32){let solving=!stopped();var sum=0.0;
  if(solving){for(var row=lid;row<liveRows();row+=256u){sum+=residual[row]*preconditioned[row];}}sums[lid]=vec4f(sum,0.0,0.0,0.0);
  for(var width=128u;width>0u;width>>=1u){workgroupBarrier();if(lid<width){sums[lid]+=sums[lid+width];}}workgroupBarrier();if(lid==0u){let nextRz=sums[0].x;let previousRz=bitcast<f32>(atomicLoad(&control[6]));
    if(!solving){return;}if(!finite(nextRz)||nextRz<0.0||previousRz<=epsilon()){reportAt(NONFINITE,13u,INVALID_ROW,nextRz);return;}atomicStore(&control[8],bitcast<u32>(nextRz/previousRz));atomicStore(&control[6],bitcast<u32>(nextRz));}}
@compute @workgroup_size(64) fn updateDirection(@builtin(global_invocation_id) gid:vec3u){let row=rowIndex(gid);if(row>=liveRows()||stopped()){return;}let beta=bitcast<f32>(atomicLoad(&control[8]));let next=preconditioned[row]+beta*direction[row];if(!finite(next)){reportAt(NONFINITE,11u,row,next);}else{direction[row]=next;}}

@compute @workgroup_size(1) fn finalizeMGPCG(){if(atomicLoad(&control[1])==0u&&atomicLoad(&control[0])==0u){report(NONCONVERGENCE);}
  if(arrayLength(&counts)>=2u){let tail=arrayLength(&counts);let success=atomicLoad(&control[0])==0u&&atomicLoad(&control[1])!=0u;
    counts[tail-2u]=select(0x7fc00000u,atomicLoad(&control[4]),success);counts[tail-1u]=select(0x7fc00000u,atomicLoad(&control[5]),success);}}
@compute @workgroup_size(64) fn publishMGPCG(@builtin(global_invocation_id) gid:vec3u){let row=rowIndex(gid);let n=liveRows();if(row>=n){return;}
  let success=atomicLoad(&control[0])==0u&&atomicLoad(&control[1])!=0u;let seed=select(0.0,pressureSeed[row],finite(pressureSeed[row]));
  pressureOut[row]=select(seed,pressure[row],success&&finite(pressure[row]));}
`;
