/**
 * Fixed symmetric Section 4.3 hybrid pressure preconditioner.
 *
 * One application is
 *
 *   J^k(0, q) -> p1
 *   r1 = q - A2 p1
 *   delta = V1(r1)
 *   J^k(p1 + delta, q) -> z
 *
 * where J is damped Jacobi in the three-row boundary/transition band and V1
 * is the supplied proof-carrying SPD first-order V-cycle. The same even k and
 * damping are used on both sides of V1; callers cannot schedule an asymmetric
 * variant.
 */
import {
  normalizeOctreeSection43BoundarySmoothing,
  OCTREE_SECTION43_BOUNDARY_BAND_LAYERS,
  type OctreeFirstOrderSPDVCycle,
} from "./webgpu-octree-section43-contract";
import type { OctreePipelinedWorksetLinearOperator } from "./webgpu-octree-pipelined-mgpcg";
import { PassBroker } from "./webgpu-pass-broker";
import { octreeSection63DirectionChannelWGSL } from "./webgpu-octree-spgrid-vcycle";

type HybridStage =
  | "resetBandWorksets"
  | "classifyBand"
  | "dilateBandAtoB"
  | "dilateBandBtoA"
  | "compactBandIntersections"
  | "finalizeBandWorksets"
  | "prepareCorrectionDispatches"
  | "smoothZeroToB"
  | "smoothBtoA"
  | "smoothAtoB"
  | "formInnerResidual"
  | "addInnerCorrection"
  | "publishCorrection";

const HYBRID_BINDINGS: Readonly<Record<HybridStage, readonly number[]>> = Object.freeze({
  resetBandWorksets: [0, 12, 15, 16, 17],
  classifyBand: [0, 1, 10, 12, 13, 17, 23],
  dilateBandAtoB: [0, 1, 2, 10, 11, 12, 13, 17, 21, 22, 23],
  dilateBandBtoA: [0, 1, 2, 10, 11, 12, 13, 17, 21, 22, 23],
  compactBandIntersections: [0, 1, 11, 12, 13, 15, 17, 23],
  finalizeBandWorksets: [0, 12, 15, 16, 17],
  prepareCorrectionDispatches: [0, 12, 15, 16, 17, 19, 20],
  smoothZeroToB: [0, 1, 4, 6, 7, 11, 12, 13, 17],
  smoothBtoA: [0, 1, 4, 6, 7, 11, 12, 13, 14, 15, 17],
  smoothAtoB: [0, 1, 4, 6, 7, 11, 12, 13, 14, 15, 17],
  formInnerResidual: [0, 4, 8, 12, 14, 17],
  addInnerCorrection: [0, 6, 7, 9, 12, 17],
  publishCorrection: [0, 5, 7, 12, 17],
});

/** Byte offset of the fifth (union) compact class record. */
const BAND_UNION_DISPATCH_OFFSET_BYTES = 4 * 12;

export interface OctreeSection43HybridPreconditionerSource {
  readonly rowCount: GPUBuffer;
  readonly firstOrderVCycle: OctreeFirstOrderSPDVCycle;
  readonly secondOrderOperator: OctreePipelinedWorksetLinearOperator;
  readonly section63: {
    readonly coefficients: GPUBuffer;
    readonly control: GPUBuffer;
    readonly topology: GPUBuffer;
    readonly state: GPUBuffer;
    readonly geometry: GPUBuffer;
    readonly metrics: GPUBuffer;
    readonly layout: GPUBuffer;
  };
}

export interface OctreeSection43HybridPreconditionerOptions {
  readonly rowCapacity: number;
  readonly boundarySmoothingIterations?: number;
  readonly damping?: number;
}

type CachedGroup = {
  readonly resources: readonly (GPUBuffer | undefined)[];
  readonly group: GPUBindGroup;
};

/**
 * Proof-carrying fixed preconditioner consumed by the pipelined outer solve.
 * The class exposes only the ordinary correction interface: the L2 shell is
 * mandatory and cannot be bypassed by the production solver.
 */
export class WebGPUOctreeSection43HybridPreconditioner
implements OctreeFirstOrderSPDVCycle {
  readonly operatorOrder = 1 as const;
  readonly isSymmetricPositiveDefinite = true as const;
  readonly convergenceTail = "gpu-zero-indirect" as const;
  readonly encodedSetupDispatchCount: number;
  readonly encodedCorrectionDispatchCount: number;
  readonly encodedPassTransitionCount: number;
  readonly allocatedBytes: number;
  readonly boundarySmoothingIterations: number;
  readonly workAccountingPlan: Readonly<{ boundarySweeps: number; encodedSetupDispatches: number;
    encodedCorrectionDispatches: number; mergedBandApplies: number; avoidedBandDispatches: number }>;

  private readonly params: GPUBuffer;
  private readonly hybridA: GPUBuffer;
  private readonly hybridB: GPUBuffer;
  private readonly innerRhs: GPUBuffer;
  private readonly innerCorrection: GPUBuffer;
  private readonly bandA: GPUBuffer;
  private readonly bandB: GPUBuffer;
  private readonly operatorImage: GPUBuffer;
  /** Compact four-class intersections of the fixed three-layer L2 shell. */
  private readonly bandWorksets: GPUBuffer;
  private readonly bandDispatch: GPUBuffer;
  private readonly gatedRowDispatch: GPUBuffer;
  private readonly gatedBandDispatch: GPUBuffer;
  private readonly bandOperatorLayout: GPUBuffer;
  private readonly pipelines: Readonly<Record<HybridStage, GPUComputePipeline>>;
  private readonly groups = new Map<HybridStage, CachedGroup[]>();
  private destroyed = false;

  /** Exact GPU-published shell intersections, exposed only for post-submit
   * work accounting. The fifth workset is the union used by smoother passes. */
  get workAccountingBuffers(): Readonly<{
    bandWorksetBanks: readonly (readonly GPUBufferBinding[])[];
    bandDispatch: GPUBuffer;
  }> {
    const strideBytes = this.bandWorksets.size / 2 / 5;
    const bankBytes = this.bandWorksets.size / 2;
    return Object.freeze({
      bandWorksetBanks: Object.freeze([0, 1].map((bank) => Object.freeze(
        [0, 1, 2, 3, 4].map((index) => ({
          buffer: this.bandWorksets,
          offset: bank * bankBytes + index * strideBytes,
          size: 7 * 4,
        })),
      ))),
      bandDispatch: this.bandDispatch,
    });
  }

  constructor(
    private readonly device: GPUDevice,
    private readonly source: OctreeSection43HybridPreconditionerSource,
    options: OctreeSection43HybridPreconditionerOptions,
  ) {
    if (!Number.isSafeInteger(options.rowCapacity) || options.rowCapacity < 1) {
      throw new RangeError("Section 4.3 hybrid row capacity must be a positive integer");
    }
    const rowGroups = Math.ceil(options.rowCapacity / 64);
    if (rowGroups > 65_535) {
      throw new RangeError("Section 4.3 hybrid row capacity exceeds one-dimensional dispatch");
    }
    const inner = source.firstOrderVCycle;
    if (inner.operatorOrder !== 1 || inner.isSymmetricPositiveDefinite !== true
      || (inner as OctreeFirstOrderSPDVCycle & { convergenceTail?: string })
        .convergenceTail !== "gpu-zero-indirect"
      || source.secondOrderOperator.convergenceTail !== "gpu-zero-indirect") {
      throw new Error("Section 4.3 hybrid requires convergence-gated SPD first-order and L2 operators");
    }
    const damping = options.damping ?? 2 / 3;
    if (!Number.isFinite(damping) || damping <= 0 || damping >= 1) {
      throw new RangeError("Section 4.3 hybrid damping must be finite and in (0,1)");
    }
    this.boundarySmoothingIterations =
      normalizeOctreeSection43BoundarySmoothing(options.boundarySmoothingIterations);
    const vectorBytes = options.rowCapacity * 4;
    const worksetStrideWords = options.rowCapacity + 7;
    const worksetBankStrideWords = 5 * worksetStrideWords;
    if (source.section63.coefficients.size < 2 * options.rowCapacity * 19 * 4
      || source.rowCount.size < 4
      || !Number.isSafeInteger(source.secondOrderOperator.encodedDispatchCount)
      || source.secondOrderOperator.encodedDispatchCount < 1
      || !Number.isSafeInteger(source.secondOrderOperator.encodedMergedBandDispatchCount)
      || source.secondOrderOperator.encodedMergedBandDispatchCount < 1) {
      throw new RangeError("Section 4.3 hybrid compact L2 source capacity is too small");
    }
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
      | GPUBufferUsage.COPY_DST;
    const vector = (label: string) => device.createBuffer({
      label,
      size: vectorBytes,
      usage: storage | GPUBufferUsage.COPY_SRC,
    });
    this.hybridA = vector("Section 4.3 hybrid L2 iterate A");
    this.hybridB = vector("Section 4.3 hybrid L2 iterate B");
    this.innerRhs = vector("Section 4.3 hybrid L1 residual");
    this.innerCorrection = vector("Section 4.3 hybrid L1 correction");
    this.bandA = vector("Section 4.3 hybrid boundary band A");
    this.bandB = vector("Section 4.3 hybrid boundary band B");
    this.operatorImage = vector("Section 4.3 hybrid resolved L2 image");
    this.bandWorksets = device.createBuffer({
      label: "Section 4.3 compact A/B class-intersection worksets",
      size: 2 * worksetBankStrideWords * 4,
      usage: storage,
    });
    this.bandDispatch = device.createBuffer({
      label: "Section 4.3 exact compact class dispatches",
      size: 5 * 12,
      usage: storage | GPUBufferUsage.INDIRECT,
    });
    this.gatedRowDispatch = device.createBuffer({
      label: "Section 4.3 convergence-gated row dispatch",
      size: 12,
      usage: storage | GPUBufferUsage.INDIRECT,
    });
    this.gatedBandDispatch = device.createBuffer({
      label: "Section 4.3 convergence-gated compact class dispatches",
      // Five compact class/union records plus direct-term and adjoint records.
      size: 7 * 12,
      usage: storage | GPUBufferUsage.INDIRECT,
    });
    this.bandOperatorLayout = device.createBuffer({
      label: "Section 4.3 compact operator workset layout",
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(this.bandOperatorLayout, 0, new Uint32Array([
      worksetStrideWords, worksetBankStrideWords, 0, 0,
    ]));
    this.params = device.createBuffer({
      label: "Section 4.3 hybrid fixed parameters",
      size: 48,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const words = new Uint32Array(12);
    const floats = new Float32Array(words.buffer);
    words[0] = options.rowCapacity;
    words[1] = this.boundarySmoothingIterations;
    words[2] = OCTREE_SECTION43_BOUNDARY_BAND_LAYERS;
    floats[3] = damping;
    words[4] = worksetStrideWords;
    words[5] = worksetBankStrideWords;
    const smoother = source.firstOrderVCycle.smootherContract;
    if (!smoother || smoother.kind !== "chebyshev"
      || (smoother.degree !== 2 && smoother.degree !== 4)
      || smoother.spectralBounds !== "transactional-scaled-gershgorin") {
      throw new Error("Section 4.3 hybrid correction requires the exact symmetric SPGrid Chebyshev contract");
    }
    words[6] = smoother.degree;
    device.queue.writeBuffer(this.params, 0, words);

    const shaderModule = device.createShaderModule({
      label: "Section 4.3 fixed symmetric hybrid preconditioner",
      code: octreeSection43HybridPreconditionerShader,
    });
    this.pipelines = Object.freeze(Object.fromEntries(
      (Object.keys(HYBRID_BINDINGS) as HybridStage[]).map((entryPoint) => [
        entryPoint,
        device.createComputePipeline({
          label: `Section 4.3 hybrid · ${entryPoint}`,
          layout: "auto",
          compute: { module: shaderModule, entryPoint },
        }),
      ]),
    ) as Record<HybridStage, GPUComputePipeline>);
    this.encodedSetupDispatchCount = (inner.encodedSetupDispatchCount ?? 0)
      + 5 + OCTREE_SECTION43_BOUNDARY_BAND_LAYERS;
    const mergedBandApplies = 2 * this.boundarySmoothingIterations - 1;
    // One convergence-gate publisher, four row-wide stages (zero sweep, inner
    // residual, M1 seeding, publication), one compact band smooth plus the
    // operator's staged merged-band apply per matched sweep after the zero
    // sweep, one exact row-wide L2 apply, and the page-parallel L1 V-cycle.
    this.encodedCorrectionDispatchCount = 5
      + (1 + source.secondOrderOperator.encodedMergedBandDispatchCount) * mergedBandApplies
      + source.secondOrderOperator.encodedDispatchCount
      + inner.encodedCorrectionDispatchCount;
    // Setup keeps its accepted-row and compact-band boundaries. Correction
    // adds its own gate boundary plus the exact L2 apply's gate boundary; the
    // inner V-cycle reports its own.
    this.encodedPassTransitionCount = (inner.encodedPassTransitionCount ?? 0) + 4;
    this.allocatedBytes = this.params.size + 7 * vectorBytes
      + this.bandWorksets.size + this.bandDispatch.size
      + this.gatedRowDispatch.size + this.gatedBandDispatch.size;
    this.workAccountingPlan = Object.freeze({ boundarySweeps: this.boundarySmoothingIterations,
      encodedSetupDispatches: this.encodedSetupDispatchCount,
      encodedCorrectionDispatches: this.encodedCorrectionDispatchCount,
      mergedBandApplies,
      avoidedBandDispatches: mergedBandApplies
        * (source.secondOrderOperator.encodedDispatchCount
          - source.secondOrderOperator.encodedMergedBandDispatchCount) });
  }

  encodeSetup(
    broker: PassBroker,
    input: { readonly solverControl: GPUBuffer; readonly rowCount: GPUBuffer },
  ): void {
    this.assertLive();
    this.assertRowCount(input.rowCount);
    const resources = this.resources(input.solverControl, input.rowCount);
    let pass = broker.compute({ label: "Section 4.3 accepted-row dispatch publication" });
    this.runDirect(pass, "resetBandWorksets", resources);
    this.runDirect(pass, "prepareCorrectionDispatches", resources);
    broker.fence("Section 4.3 accepted-row dispatch publication");
    pass = broker.compute({ label: "Section 4.3 hybrid boundary-band publication" });
    this.runRows(pass, "classifyBand", resources);
    this.runRows(pass, "dilateBandAtoB", resources);
    this.runRows(pass, "dilateBandBtoA", resources);
    this.runRows(pass, "dilateBandAtoB", resources);
    this.runRows(pass, "compactBandIntersections", resources);
    this.runDirect(pass, "finalizeBandWorksets", resources);
    broker.fence("Section 4.3 compact band dispatch publication");
    this.source.firstOrderVCycle.encodeSetup(broker, input);
  }

  encodeCorrection(
    broker: PassBroker,
    input: {
      readonly rhs: GPUBuffer;
      readonly correction: GPUBuffer;
      readonly solverControl: GPUBuffer;
      readonly rowCount: GPUBuffer;
    },
  ): void {
    this.assertLive();
    this.assertRowCount(input.rowCount);
    if (input.rhs.size < this.hybridA.size || input.correction.size < this.hybridA.size) {
      throw new RangeError("Section 4.3 hybrid correction vectors are too small");
    }
    const resources = this.resources(
      input.solverControl, input.rowCount, input.rhs, input.correction,
    );
    // Republish the convergence-gated accepted-row and compact-class records.
    // A converged or failed solve publishes zero groups, so every stage below
    // remains encoded and dispatches no workgroups.
    let pass = broker.compute({ label: "Section 4.3 convergence-gated correction records" });
    this.runDirect(pass, "prepareCorrectionDispatches", resources);
    broker.fence("Section 4.3 correction dispatch publication");

    // §4.3(1): J^k(0, q) -> p1. The first sweep starts from the exact zero
    // vector and needs no operator image; every later sweep consumes a
    // band-restricted L2 apply of the state it reads.
    pass = broker.compute({ label: "Section 4.3 hybrid pre-smoothing shell" });
    this.runRows(pass, "smoothZeroToB", resources);
    for (let iteration = 1; iteration < this.boundarySmoothingIterations; iteration += 1) {
      this.encodeBandSweep(broker, resources, (iteration & 1) === 1, input.solverControl);
    }

    // §4.3(2): r1 = q - L2 p1 over every accepted row, then the page-parallel
    // symmetric first-order V-cycle. p1 now lives in hybridA.
    this.source.secondOrderOperator.encode(
      broker, this.hybridA, this.operatorImage, input.solverControl,
    );
    pass = broker.compute({ label: "Section 4.3 hybrid inner residual" });
    this.runRows(pass, "formInnerResidual", resources);
    this.source.firstOrderVCycle.encodeCorrection(broker, {
      rhs: this.innerRhs,
      correction: this.innerCorrection,
      solverControl: input.solverControl,
      rowCount: input.rowCount,
    });

    // §4.3(3): p2 = p1 + M1 r1 seeds both ping-pong states, then the identical
    // matched k sweeps leave z in hybridB.
    pass = broker.compute({ label: "Section 4.3 hybrid post-smoothing shell" });
    this.runRows(pass, "addInnerCorrection", resources);
    for (let iteration = 0; iteration < this.boundarySmoothingIterations; iteration += 1) {
      this.encodeBandSweep(broker, resources, (iteration & 1) === 0, input.solverControl);
    }
    this.runRows(broker.compute({
      label: "Section 4.3 hybrid correction publication",
    }), "publishCorrection", resources);
  }

  /**
   * One damped-Jacobi shell sweep. The band smoother consumes a precomputed
   * L2 image of the state it reads, so the merged compact-band apply must
   * immediately precede it and target the same source state.
   */
  private encodeBandSweep(
    broker: PassBroker,
    resources: readonly (GPUBuffer | undefined)[],
    fromB: boolean,
    solverControl: GPUBuffer,
  ): void {
    this.source.secondOrderOperator.encodeMergedBandWorkset(
      broker,
      fromB ? this.hybridB : this.hybridA,
      this.operatorImage,
      solverControl,
      this.bandWorksets,
      this.gatedBandDispatch,
      this.bandOperatorLayout,
      BAND_UNION_DISPATCH_OFFSET_BYTES,
    );
    this.runBand(broker.compute({
      label: fromB
        ? "Section 4.3 hybrid band smooth B to A"
        : "Section 4.3 hybrid band smooth A to B",
    }), fromB ? "smoothBtoA" : "smoothAtoB", resources);
  }

  private resources(
    control: GPUBuffer,
    rowCount: GPUBuffer,
    rhs?: GPUBuffer,
    correction?: GPUBuffer,
  ): readonly (GPUBuffer | undefined)[] {
    const section63 = this.source.section63;
    return [
      this.params,
      section63.coefficients,
      section63.topology,
      rowCount,
      rhs,
      correction,
      this.hybridA,
      this.hybridB,
      this.innerRhs,
      this.innerCorrection,
      this.bandA,
      this.bandB,
      control,
      section63.layout,
      this.operatorImage,
      this.bandWorksets,
      this.bandDispatch,
      section63.control,
      undefined,
      this.gatedRowDispatch,
      this.gatedBandDispatch,
      section63.state,
      section63.geometry,
      section63.metrics,
    ];
  }

  /** Accepted-row stage over the convergence-gated row record. */
  private runRows(
    pass: GPUComputePassEncoder,
    name: HybridStage,
    resources: readonly (GPUBuffer | undefined)[],
  ): void {
    this.bindCached(pass, name, resources);
    pass.dispatchWorkgroupsIndirect(
      this.gatedRowDispatch,
      0,
    );
  }

  private bindCached(
    pass: GPUComputePassEncoder,
    name: HybridStage,
    resources: readonly (GPUBuffer | undefined)[],
  ): void {
    const pipeline = this.pipelines[name];
    const bindings = HYBRID_BINDINGS[name];
    const variants = this.groups.get(name);
    const cached = variants?.find((candidate) =>
      bindings.every((binding) => candidate.resources[binding] === resources[binding]));
    const group = cached?.group ?? this.device.createBindGroup({
      label: `Section 4.3 hybrid · ${name} bindings`,
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

  /** Singleton publisher stage. Its own record is what the others consume. */
  private runDirect(
    pass: GPUComputePassEncoder,
    name: HybridStage,
    resources: readonly (GPUBuffer | undefined)[],
  ): void {
    this.bindCached(pass, name, resources);
    pass.dispatchWorkgroups(1);
  }

  /** Compact shell stage over the convergence-gated union class record. */
  private runBand(
    pass: GPUComputePassEncoder,
    name: "smoothBtoA" | "smoothAtoB",
    resources: readonly (GPUBuffer | undefined)[],
  ): void {
    this.bindCached(pass, name, resources);
    pass.dispatchWorkgroupsIndirect(
      this.gatedBandDispatch,
      BAND_UNION_DISPATCH_OFFSET_BYTES,
    );
  }

  private assertRowCount(rowCount: GPUBuffer): void {
    if (rowCount !== this.source.rowCount) {
      throw new Error("Section 4.3 hybrid requires its authoritative compact row count");
    }
  }

  private assertLive(): void {
    if (this.destroyed) throw new Error("Section 4.3 hybrid preconditioner is destroyed");
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.groups.clear();
    this.params.destroy();
    this.hybridA.destroy();
    this.hybridB.destroy();
    this.innerRhs.destroy();
    this.innerCorrection.destroy();
    this.bandA.destroy();
    this.bandB.destroy();
    this.operatorImage.destroy();
    this.bandWorksets.destroy();
    this.bandDispatch.destroy();
    this.gatedRowDispatch.destroy();
    this.gatedBandDispatch.destroy();
    this.bandOperatorLayout.destroy();
  }
}

export const octreeSection43HybridPreconditionerShader = /* wgsl */ `
struct Params {
  rowCapacity: u32,
  smoothingIterations: u32,
  bandLayers: u32,
  damping: f32,
  worksetStrideWords: u32,
  worksetBankStrideWords: u32,
  m1SmoothingDegree: u32,
  padding: u32,
}
struct Section63Layout{workset:vec4u,dimsCapacity:vec4u,hierarchy:vec4u,numerics:vec4f}
struct Metric{caseId:u32,transformAndFlags:u32,volume:f32,error:u32}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> section63Coefficients: array<f32>;
@group(0) @binding(2) var<storage, read> topology: array<u32>;
@group(0) @binding(3) var<storage, read> rowCount: array<u32>;
@group(0) @binding(4) var<storage, read> rhs: array<f32>;
@group(0) @binding(5) var<storage, read_write> correction: array<f32>;
@group(0) @binding(6) var<storage, read_write> hybridA: array<f32>;
@group(0) @binding(7) var<storage, read_write> hybridB: array<f32>;
@group(0) @binding(8) var<storage, read_write> innerRhs: array<f32>;
@group(0) @binding(9) var<storage, read> innerCorrection: array<f32>;
@group(0) @binding(10) var<storage, read_write> bandA: array<u32>;
@group(0) @binding(11) var<storage, read_write> bandB: array<u32>;
@group(0) @binding(12) var<storage, read_write> control: array<atomic<u32>>;
@group(0) @binding(13) var<uniform> section63: Section63Layout;
@group(0) @binding(14) var<storage, read> operatorImage: array<f32>;
@group(0) @binding(15) var<storage, read_write> bandWorksets: array<atomic<u32>>;
@group(0) @binding(16) var<storage, read_write> bandDispatch: array<u32>;
@group(0) @binding(17) var<storage, read> accepted: array<u32>;
@group(0) @binding(19) var<storage, read_write> gatedRowDispatch: array<u32>;
@group(0) @binding(20) var<storage, read_write> gatedBandDispatch: array<u32>;
@group(0) @binding(21) var<storage, read_write> state: array<u32>;
@group(0) @binding(22) var<storage, read> geometry: array<vec4u>;
@group(0) @binding(23) var<storage, read> metrics: array<Metric>;

const INVALID_ROW_ERROR = 2u;
const NONFINITE_ERROR = 4u;
const INVALID = 0xffffffffu;
const ACTIVE=1u;const GHOST=2u;const MG_ONLY=4u;const FLAGS=1u;
const OWNER=24u;const PAGE_RECORD_WORDS=28u;

fn finite(value: f32) -> bool {
  return value == value && abs(value) <= 3.402823e38;
}
fn rows() -> u32 {
  return min(select(0u, accepted[2], arrayLength(&accepted) >= 6u), params.rowCapacity);
}
fn stopped() -> bool {
  return atomicLoad(&control[0]) != 0u || atomicLoad(&control[1]) != 0u;
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
fn diagonalAt(row: u32) -> f32 {
  return section63Coefficients[(accepted[4] & 1u) * section63.hierarchy.w + row * 19u];
}
fn validDiagonal(row: u32) -> bool {
  let diagonal = diagonalAt(row);
  return finite(diagonal) && diagonal > 0.0;
}
fn coefficientBase(row:u32)->u32{return (accepted[4]&1u)*section63.hierarchy.w+row*19u;}
fn validSection63Row(row:u32)->bool{if(row>=rows()||row>=arrayLength(&metrics)){return false;}let base=coefficientBase(row);if(base+19u>arrayLength(&section63Coefficients)){return false;}let m=metrics[row];if(m.error!=0u||(m.transformAndFlags&0x80000000u)==0u){return false;}for(var channel=0u;channel<19u;channel+=1u){let c=section63Coefficients[base+channel];if(!finite(c)||c<0.0){return false;}}return section63Coefficients[base]>0.0;}
fn section63Class(row:u32)->u32{let base=coefficientBase(row);var off=0.0;for(var channel=1u;channel<19u;channel+=1u){off+=section63Coefficients[base+channel];}let physicalBoundary=(metrics[row].transformAndFlags&0x3f00u)!=0u;let boundary=physicalBoundary||section63Coefficients[base]>off+max(1e-6,1e-5*abs(off));let transition=metrics[row].caseId!=0u;return select(select(0u,2u,boundary),select(1u,3u,boundary),transition);}
fn levels()->u32{return section63.hierarchy.x;}fn maxStride()->u32{return section63.hierarchy.y;}fn capacity()->u32{return section63.dimsCapacity.w;}
fn dims(l:u32)->vec3u{let s=1u<<l;return(section63.dimsCapacity.xyz+vec3u(s-1u))/s;}
fn levelCapacity(l:u32)->u32{let d=dims(l);let threshold=maxStride()/2u;var cells=d.x;if(cells>=threshold||d.y>threshold/cells){return maxStride();}cells*=d.y;if(cells>=threshold||d.z>threshold/cells){return maxStride();}cells*=d.z;var result=1u;let limit=2u*cells;while(result<limit){result<<=1u;}return result;}
fn levelBase(l:u32)->u32{var result=0u;for(var k=0u;k<l;k+=1u){result+=levelCapacity(k);}return result;}fn totalLevelSlots()->u32{return section63.hierarchy.z;}fn at(c:u32,l:u32,s:u32)->u32{return c*totalLevelSlots()+levelBase(l)+s;}
fn rowMapBase()->u32{return 16u;}fn workBase()->u32{return rowMapBase()+levels()*capacity();}fn pageWorkBase()->u32{return workBase()+totalLevelSlots();}fn logicalPageDims(l:u32)->vec3u{return(dims(l)+vec3u(7u,7u,3u))/vec3u(8u,8u,4u);}fn logicalPageCount(l:u32)->u32{let d=logicalPageDims(l);return d.x*d.y*d.z;}fn pageLevelOffset(l:u32)->u32{var result=0u;for(var k=0u;k<l;k+=1u){result+=logicalPageCount(k);}return result;}fn pageDirectoryBase()->u32{return pageWorkBase()+PAGE_RECORD_WORDS*totalLevelSlots();}
fn transferCapacity(l:u32)->u32{return min(capacity()*8u,levelCapacity(l)*8u);}fn transferLevelOffset(l:u32)->u32{var result=0u;for(var k=0u;k<l;k+=1u){result+=transferCapacity(k)*4u+4u*levelCapacity(k);}return result;}fn transferBase()->u32{return pageDirectoryBase()+pageLevelOffset(levels());}fn directoryBase()->u32{return transferBase()+transferLevelOffset(levels()-1u);}fn brickDims(l:u32)->vec3u{return(dims(l)+vec3u(3u))/4u;}fn brickCount(l:u32)->u32{let d=brickDims(l);return d.x*d.y*d.z;}fn brickLevelOffset(l:u32)->u32{var result=0u;for(var k=0u;k<l;k+=1u){result+=brickCount(k);}return result;}fn totalBrickCount()->u32{return brickLevelOffset(levels());}fn rankedSlotsBase()->u32{return directoryBase()+16u+totalBrickCount()*4u;}fn brickRecord(l:u32,q:vec3u)->u32{let d=brickDims(l);let b=q/4u;let dense=b.x+d.x*(b.y+d.y*b.z);return directoryBase()+16u+(brickLevelOffset(l)+dense)*4u;}
fn pageRecord(l:u32,page:u32)->u32{return pageWorkBase()+(levelBase(l)+page)*PAGE_RECORD_WORDS;}fn pageNeighbour(l:u32,page:u32,ordinal:u32)->u32{return topology[pageRecord(l,page)+1u+ordinal];}fn decode(key:u32,l:u32)->vec3u{let d=dims(l);let v=key-1u;return vec3u(v%d.x,(v/d.x)%d.y,v/(d.x*d.y));}fn localBit(q:vec3u)->u32{let local=q&vec3u(3u);return local.x+4u*local.y+16u*local.z;}fn pageFor(l:u32,q:vec3u)->u32{let pages=logicalPageDims(l);let v=q/vec3u(8u,8u,4u);return topology[pageDirectoryBase()+pageLevelOffset(l)+v.x+pages.x*(v.y+pages.y*v.z)];}
fn pageSlot(l:u32,page:u32,origin:vec3u,q:vec3u)->u32{let shape=vec3u(8u,8u,4u);let delta=vec3i(q/shape)-vec3i(origin/shape);if(any(delta<vec3i(-1))||any(delta>vec3i(1))){reportAt(INVALID_ROW_ERROR,10u,INVALID);return INVALID;}let ordinal=u32(delta.x+1)+3u*(u32(delta.y+1)+3u*u32(delta.z+1));let physical=pageNeighbour(l,page,ordinal);if(physical==INVALID){return INVALID;}let physicalOrigin=decode(topology[pageRecord(l,physical)],l);if(any(physicalOrigin/shape!=q/shape)){reportAt(INVALID_ROW_ERROR,10u,INVALID);return INVALID;}let record=brickRecord(l,q);let bit=localBit(q);let low=topology[record+1u];let high=topology[record+2u];if(((select(low,high,bit>=32u)>>(bit&31u))&1u)==0u){return INVALID;}let lower=select((1u<<(bit&31u))-1u,0xffffffffu,bit>=32u);var rank=countOneBits(low&lower);if(bit>=32u){rank+=countOneBits(high&((1u<<(bit-32u))-1u));}let slot=topology[rankedSlotsBase()+levelBase(l)+topology[record+3u]+rank];if(slot>=levelCapacity(l)){reportAt(INVALID_ROW_ERROR,10u,INVALID);return INVALID;}return slot;}
fn originOf(h:vec4u)->vec3u{return vec3u(h.x%section63.dimsCapacity.x,(h.x/section63.dimsCapacity.x)%section63.dimsCapacity.y,h.x/(section63.dimsCapacity.x*section63.dimsCapacity.y));}
fn canonicalDirection(channel:u32)->vec3i{let d=array<vec3i,18>(vec3i(1,0,0),vec3i(-1,0,0),vec3i(0,1,0),vec3i(0,-1,0),vec3i(0,0,1),vec3i(0,0,-1),vec3i(1,1,0),vec3i(1,-1,0),vec3i(-1,1,0),vec3i(-1,-1,0),vec3i(1,0,1),vec3i(1,0,-1),vec3i(-1,0,1),vec3i(-1,0,-1),vec3i(0,1,1),vec3i(0,1,-1),vec3i(0,-1,1),vec3i(0,-1,-1));return d[channel];}
fn worldDirection(value:vec3i,code:u32)->vec3i{let signs=vec3i(select(1,-1,(code&1u)!=0u),select(1,-1,(code&2u)!=0u),select(1,-1,(code&4u)!=0u));let q=value*signs;let permutation=(code/8u)%6u;if(permutation==0u){return q.xyz;}if(permutation==1u){return q.xzy;}if(permutation==2u){return q.yxz;}if(permutation==3u){return q.zxy;}if(permutation==4u){return q.yzx;}return q.zyx;}
${octreeSection63DirectionChannelWGSL}
// One indexed load replaces the eighteen-step scan that re-evaluated
// worldDirection per candidate. The table is the memoized scan result.
fn coefficientForDirection(row:u32,direction:vec3i)->f32{
 let channel=section63ChannelForDirection(metrics[row].transformAndFlags&63u,direction);
 if(channel>=18u){return 0.0;}
 return section63Coefficients[coefficientBase(row)+1u+channel];}
fn bandAt(row:u32,useB:bool)->u32{return select(bandA[row],bandB[row],useB);}
fn dilatedBand(row:u32,useB:bool)->u32{var value=bandAt(row,useB);let h=geometry[row];let m=metrics[row];let l=countTrailingZeros(h.y);let q=originOf(h)/(1u<<l);let page=pageFor(l,q);let base=coefficientBase(row);for(var channel=0u;channel<18u&&value==0u;channel+=1u){if(section63Coefficients[base+1u+channel]==0.0){continue;}let targetQ=vec3i(q)+worldDirection(canonicalDirection(channel),m.transformAndFlags&63u);if(any(targetQ<vec3i(0))||any(targetQ>=vec3i(dims(l)))){continue;}let slot=pageSlot(l,page,q,vec3u(targetQ));if(slot==INVALID||(state[at(FLAGS,l,slot)]&MG_ONLY)!=0u){continue;}let encoded=state[at(OWNER,l,slot)];if(encoded>0u&&encoded<=rows()){value=max(value,bandAt(encoded-1u,useB));}}
 if(l>0u&&value==0u){let fine=l-1u;for(var child=0u;child<8u&&value==0u;child+=1u){let ghostQ=2u*q+vec3u(child&1u,(child>>1u)&1u,(child>>2u)&1u);let ghostPage=pageFor(fine,ghostQ);if(ghostPage==INVALID){continue;}let ghost=pageSlot(fine,ghostPage,ghostQ,ghostQ);if(ghost==INVALID||(state[at(FLAGS,fine,ghost)]&GHOST)==0u||state[at(OWNER,fine,ghost)]!=row+1u){continue;}for(var candidate=0u;candidate<18u&&value==0u;candidate+=1u){let delta=canonicalDirection(candidate);let activeQ=vec3i(ghostQ)-delta;if(any(activeQ<vec3i(0))||any(activeQ>=vec3i(dims(fine)))){continue;}let activeSlot=pageSlot(fine,ghostPage,ghostQ,vec3u(activeQ));if(activeSlot==INVALID||(state[at(FLAGS,fine,activeSlot)]&ACTIVE)==0u){continue;}let encoded=state[at(OWNER,fine,activeSlot)];if(encoded>0u&&encoded<=rows()&&coefficientForDirection(encoded-1u,delta)>0.0){value=max(value,bandAt(encoded-1u,useB));}}}}
 return value;}
fn hybridValue(row: u32, useB: bool) -> f32 {
  return select(hybridA[row], hybridB[row], useB);
}
fn bandWorksetBase(cls: u32) -> u32 {
  return (accepted[4] & 1u) * params.worksetBankStrideWords
    + cls * params.worksetStrideWords;
}
fn bandRow(item: u32) -> u32 {
  let base = bandWorksetBase(4u);
  if (atomicLoad(&bandWorksets[base]) != accepted[3]
    || atomicLoad(&bandWorksets[base + 3u]) != 3u
    || item >= atomicLoad(&bandWorksets[base + 1u])) {
    return INVALID;
  }
  return atomicLoad(&bandWorksets[base + 7u + item]);
}

@compute @workgroup_size(1)
fn prepareCorrectionDispatches() {
  let solveLive = !stopped();
  gatedRowDispatch[0] = select(0u, (rows() + 63u) / 64u, solveLive);
  gatedRowDispatch[1] = 1u;
  gatedRowDispatch[2] = 1u;
  for (var word = 0u; word < 15u; word += 1u) {
    var value = bandDispatch[word];
    if (!solveLive && word % 3u == 0u) { value = 0u; }
    gatedBandDispatch[word] = value;
  }
  let unionBase = bandWorksetBase(4u);
  let unionRows = atomicLoad(&bandWorksets[unionBase + 1u]);
  let unionValid = atomicLoad(&bandWorksets[unionBase]) == accepted[3]
    && atomicLoad(&bandWorksets[unionBase + 2u]) >= unionRows
    && atomicLoad(&bandWorksets[unionBase + 3u]) == 3u;
  gatedBandDispatch[15] = select(0u, (unionRows * 18u + 63u) / 64u,
    solveLive && unionValid);
  gatedBandDispatch[16] = 1u;
  gatedBandDispatch[17] = 1u;
  var transitionRows = 0u;
  var transitionValid = true;
  for (var index = 0u; index < 2u; index += 1u) {
    let cls = select(1u, 3u, index == 1u);
    let base = bandWorksetBase(cls);
    let count = atomicLoad(&bandWorksets[base + 1u]);
    transitionValid = transitionValid
      && atomicLoad(&bandWorksets[base]) == accepted[3]
      && atomicLoad(&bandWorksets[base + 2u]) >= count
      && atomicLoad(&bandWorksets[base + 3u]) == 3u;
    transitionRows += count;
  }
  gatedBandDispatch[18] = select(0u, (transitionRows * 8u + 63u) / 64u,
    solveLive && unionValid && transitionValid);
  gatedBandDispatch[19] = 1u;
  gatedBandDispatch[20] = 1u;
}

@compute @workgroup_size(1)
fn resetBandWorksets() {
  if (stopped()) { return; }
  for (var cls = 0u; cls < 5u; cls += 1u) {
    let base = bandWorksetBase(cls);
    atomicStore(&bandWorksets[base], accepted[3]);
    atomicStore(&bandWorksets[base + 1u], 0u);
    atomicStore(&bandWorksets[base + 2u], params.rowCapacity);
    atomicStore(&bandWorksets[base + 3u], 0u);
    atomicStore(&bandWorksets[base + 4u], 0u);
    atomicStore(&bandWorksets[base + 5u], 1u);
    atomicStore(&bandWorksets[base + 6u], 1u);
    bandDispatch[3u * cls] = 0u;
    bandDispatch[3u * cls + 1u] = 1u;
    bandDispatch[3u * cls + 2u] = 1u;
  }
}
fn smoothValue(row: u32, useB: bool) -> f32 {
  let current = hybridValue(row, useB);
  if (bandB[row] == 0u) { return current; }
  let next = current
    + params.damping * (rhs[row] - operatorImage[row]) / diagonalAt(row);
  if (!finite(next)) {
    reportAt(NONFINITE_ERROR, 11u, row);
    return current;
  }
  return next;
}
fn smoothZeroValue(row: u32) -> f32 {
  if (bandB[row] == 0u) { return 0.0; }
  if (!validDiagonal(row)) {
    reportAt(INVALID_ROW_ERROR, 10u, row);
    return 0.0;
  }
  // This is the exact zero-vector Jacobi sweep. It must not read hybridA:
  // rows in one dispatch cannot synchronize storage writes across workgroups.
  let next = params.damping * rhs[row] / diagonalAt(row);
  if (!finite(next)) {
    reportAt(NONFINITE_ERROR, 11u, row);
    return 0.0;
  }
  return next;
}

@compute @workgroup_size(64)
fn classifyBand(@builtin(global_invocation_id) global: vec3u) {
  let row = global.x;
  if (stopped() || row >= rows()) { return; }
  if (!validDiagonal(row)) {
    reportAt(INVALID_ROW_ERROR, 10u, row);
    return;
  }
  if (!validSection63Row(row)) {
    reportAt(INVALID_ROW_ERROR, 10u, row);
    return;
  }
  let rowClass = section63Class(row);
  let transition = rowClass == 1u || rowClass == 3u;
  let boundary = rowClass == 2u || rowClass == 3u;
  bandA[row] = select(0u, 1u, boundary || transition);
}
@compute @workgroup_size(64)
fn dilateBandAtoB(@builtin(global_invocation_id) global: vec3u) {
  let row = global.x;
  if (stopped() || row >= rows()) { return; }
  bandB[row] = dilatedBand(row, false);
}
@compute @workgroup_size(64)
fn dilateBandBtoA(@builtin(global_invocation_id) global: vec3u) {
  let row = global.x;
  if (stopped() || row >= rows()) { return; }
  bandA[row] = dilatedBand(row, true);
}
@compute @workgroup_size(64)
fn compactBandIntersections(@builtin(global_invocation_id) global: vec3u) {
  let row = global.x;
  if (stopped() || row >= rows() || bandB[row] == 0u) { return; }
  let cls = section63Class(row);
  if (cls >= 4u) { reportAt(INVALID_ROW_ERROR, 10u, row); return; }
  let classBase = bandWorksetBase(cls);
  let classRank = atomicAdd(&bandWorksets[classBase + 1u], 1u);
  let allBase = bandWorksetBase(4u);
  let allRank = atomicAdd(&bandWorksets[allBase + 1u], 1u);
  if (classRank >= params.rowCapacity || allRank >= params.rowCapacity) {
    reportAt(INVALID_ROW_ERROR, 10u, row);
    return;
  }
  atomicStore(&bandWorksets[classBase + 7u + classRank], row);
  atomicStore(&bandWorksets[allBase + 7u + allRank], row);
}
@compute @workgroup_size(1)
fn finalizeBandWorksets() {
  if (stopped()) { return; }
  for (var cls = 0u; cls < 5u; cls += 1u) {
    let base = bandWorksetBase(cls);
    let count = atomicLoad(&bandWorksets[base + 1u]);
    let groups = (count + 63u) / 64u;
    atomicStore(&bandWorksets[base + 3u], 3u);
    atomicStore(&bandWorksets[base + 4u], groups);
    bandDispatch[3u * cls] = groups;
  }
}
@compute @workgroup_size(64)
fn smoothZeroToB(@builtin(global_invocation_id) global: vec3u) {
  let row = global.x;
  if (stopped() || row >= rows()) { return; }
  hybridA[row] = 0.0;
  hybridB[row] = smoothZeroValue(row);
}
@compute @workgroup_size(64)
fn smoothBtoA(@builtin(global_invocation_id) global: vec3u) {
  let row = bandRow(global.x);
  if (!stopped() && row != INVALID && row < rows()) { hybridA[row] = smoothValue(row, true); }
}
@compute @workgroup_size(64)
fn smoothAtoB(@builtin(global_invocation_id) global: vec3u) {
  let row = bandRow(global.x);
  if (!stopped() && row != INVALID && row < rows()) { hybridB[row] = smoothValue(row, false); }
}
@compute @workgroup_size(64)
fn formInnerResidual(@builtin(global_invocation_id) global: vec3u) {
  let row = global.x;
  if (stopped() || row >= rows()) { return; }
  let value = rhs[row] - operatorImage[row];
  if (!finite(value)) { reportAt(NONFINITE_ERROR, 12u, row); }
  else { innerRhs[row] = value; }
}
@compute @workgroup_size(64)
fn addInnerCorrection(@builtin(global_invocation_id) global: vec3u) {
  let row = global.x;
  if (stopped() || row >= rows()) { return; }
  let value = hybridA[row] + innerCorrection[row];
  if (!finite(value)) { reportAt(NONFINITE_ERROR, 13u, row); }
  // Paper §4.3 step (3) starts the matching post-sweep from p2 everywhere.
  // Only band rows are subsequently ping-ponged, so both states must carry
  // the domain-wide M1 correction before that compact dispatch begins.
  else { hybridA[row] = value; hybridB[row] = value; }
}
@compute @workgroup_size(64)
fn publishCorrection(@builtin(global_invocation_id) global: vec3u) {
  let row = global.x;
  if (stopped() || row >= rows()) { return; }
  let value = hybridB[row];
  if (!finite(value)) { reportAt(NONFINITE_ERROR, 14u, row); }
  else { correction[row] = value; }
}

`;
