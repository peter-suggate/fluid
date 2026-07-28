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
import {
  octreeSection63DirectionChannelWGSL,
  octreeSection63OperatorLayoutWGSL,
  octreeSection63RowAddressWGSL,
  octreeSection63RowImageWGSL,
} from "./webgpu-octree-spgrid-vcycle";

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
  prepareCorrectionDispatches: [0, 12, 16, 17, 19, 20],
  smoothZeroToB: [0, 1, 4, 6, 11, 12, 13, 17],
  // The fused sweep resolves its own Section 6.3 row image, so it reaches the
  // operator's topology/state/geometry/metric ABI and no longer reads the
  // staged image at binding 14. Ten storage bindings exactly; see the portable
  // ceiling asserted by tests/webgpu-octree-section43-persistent-correction.
  smoothBtoA: [0, 1, 2, 4, 6, 12, 13, 15, 17, 21, 22, 23],
  smoothAtoB: [0, 1, 2, 4, 6, 12, 13, 15, 17, 21, 22, 23],
  formInnerResidual: [0, 4, 8, 12, 14, 17],
  addInnerCorrection: [0, 6, 9, 12, 17],
  publishCorrection: [0, 5, 6, 12, 17],
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
    encodedCorrectionDispatches: number; fusedBandSweeps: number; avoidedBandDispatches: number }>;

  private readonly params: GPUBuffer;
  /** Both §4.3 ping-pong states, A at row 0 and B at row `rowCapacity`. One
   * buffer because the fused band sweep reads one half and writes the other,
   * and because two bindings do not fit its ten-storage operator ABI. */
  private readonly hybrid: GPUBuffer;
  private readonly vectorBytes: number;
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
    this.vectorBytes = vectorBytes;
    const worksetStrideWords = options.rowCapacity + 7;
    const worksetBankStrideWords = 5 * worksetStrideWords;
    if (source.section63.coefficients.size < 2 * options.rowCapacity * 19 * 4
      || source.rowCount.size < 4
      || !Number.isSafeInteger(source.secondOrderOperator.encodedDispatchCount)
      || source.secondOrderOperator.encodedDispatchCount < 1
      || source.secondOrderOperator.encodedMergedBandDispatchCount !== 1) {
      throw new RangeError("Section 4.3 hybrid compact L2 source capacity is too small");
    }
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
      | GPUBufferUsage.COPY_DST;
    const vector = (label: string) => device.createBuffer({
      label,
      size: vectorBytes,
      usage: storage | GPUBufferUsage.COPY_SRC,
    });
    this.hybrid = device.createBuffer({
      label: "Section 4.3 hybrid L2 iterates A|B",
      size: 2 * vectorBytes,
      usage: storage | GPUBufferUsage.COPY_SRC,
    });
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
      size: 5 * 12,
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
    const fusedBandSweeps = 2 * this.boundarySmoothingIterations - 1;
    // One convergence-gate publisher, four row-wide stages (zero sweep, inner
    // residual, M1 seeding, publication), ONE compact band sweep per matched
    // sweep after the zero sweep — the merged-band L2 apply that used to
    // precede each of them is now resolved inside it, see encodeBandSweep —
    // one exact row-wide L2 apply, and the page-parallel first-order V-cycle.
    this.encodedCorrectionDispatchCount = 5 + fusedBandSweeps
      + source.secondOrderOperator.encodedDispatchCount
      + inner.encodedCorrectionDispatchCount;
    // Setup keeps its accepted-row and compact-band boundaries. Correction
    // adds its own gate boundary plus the exact L2 apply's gate boundary; the
    // inner V-cycle reports its own. The fused sweep fences nothing, exactly as
    // the apply/smooth pair it replaces did not.
    this.encodedPassTransitionCount = (inner.encodedPassTransitionCount ?? 0) + 4;
    this.allocatedBytes = this.params.size + this.hybrid.size + 5 * vectorBytes
      + this.bandWorksets.size + this.bandDispatch.size
      + this.gatedRowDispatch.size + this.gatedBandDispatch.size;
    this.workAccountingPlan = Object.freeze({ boundarySweeps: this.boundarySmoothingIterations,
      encodedSetupDispatches: this.encodedSetupDispatchCount,
      encodedCorrectionDispatches: this.encodedCorrectionDispatchCount,
      fusedBandSweeps,
      // Each fused sweep avoids a whole encoded band apply: its gate is the
      // sweep's own, and its row image never reaches device memory.
      avoidedBandDispatches: fusedBandSweeps
        * source.secondOrderOperator.encodedDispatchCount });
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
    if (input.rhs.size < this.vectorBytes || input.correction.size < this.vectorBytes) {
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
      this.encodeBandSweep(broker, resources, (iteration & 1) === 1);
    }

    // §4.3(2): r1 = q - L2 p1 over every accepted row, then the page-parallel
    // symmetric first-order V-cycle. p1 now lives in the arena's A half, which
    // is its row 0, so the operator binds the arena and reads exactly hybridA.
    this.source.secondOrderOperator.encode(
      broker, this.hybrid, this.operatorImage, input.solverControl,
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
      this.encodeBandSweep(broker, resources, (iteration & 1) === 0);
    }
    this.runRows(broker.compute(), "publishCorrection", resources);
  }

  /**
   * One damped-Jacobi shell sweep, over the convergence-gated union class
   * record. Each row resolves its own Section 6.3 operator image inline, from
   * the shared `applyRowImage` the merged-band A2 apply calls; damped Jacobi
   * reads only the source half of the arena and writes only the destination
   * half, so a row taking `A x` of the same source vector it smooths from is
   * not a hazard. This deletes the second indirect dispatch and the device
   * round trip of the image that used to separate the two — see
   * `docs/POWER_LIQUIDS_ULTIMATE_M1MAX.md`, "SPGrid Section 6.3 ·
   * destination-owned merged band", 2.719 ms/advance over 22 brackets and 330
   * dispatches on the mini lane.
   */
  private encodeBandSweep(
    broker: PassBroker,
    resources: readonly (GPUBuffer | undefined)[],
    fromB: boolean,
  ): void {
    this.runBand(broker.compute(), fromB ? "smoothBtoA" : "smoothAtoB", resources);
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
      this.hybrid,
      undefined,
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
    this.hybrid.destroy();
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
${octreeSection63OperatorLayoutWGSL}
struct Metric{caseId:u32,transformAndFlags:u32,volume:f32,error:u32}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> section63Coefficients: array<f32>;
@group(0) @binding(2) var<storage, read> topology: array<u32>;
@group(0) @binding(3) var<storage, read> rowCount: array<u32>;
@group(0) @binding(4) var<storage, read> rhs: array<f32>;
@group(0) @binding(5) var<storage, read_write> correction: array<f32>;
// Both ping-pong states: A occupies rows [0, rowCapacity), B the rows that
// follow. Binding 7 is retired with the split buffers.
@group(0) @binding(6) var<storage, read_write> hybrid: array<f32>;
@group(0) @binding(8) var<storage, read_write> innerRhs: array<f32>;
@group(0) @binding(9) var<storage, read> innerCorrection: array<f32>;
@group(0) @binding(10) var<storage, read_write> bandA: array<u32>;
@group(0) @binding(11) var<storage, read_write> bandB: array<u32>;
@group(0) @binding(12) var<storage, read_write> control: array<atomic<u32>>;
// The SAME buffer the accurate A2 operator binds as its layout uniform, now
// declared with its full shape so the shared Section 6.3 addressing block reads
// the identical memoized level tables the operator does.
@group(0) @binding(13) var<uniform> p: Layout;
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
// The accurate A2 operator's own report policy, so the shared row-image block
// below claims (stage, row) exactly as it does when the operator calls it from
// its own dispatch. The retry loop above is this module's; the two decide only
// which diagnostic pair lands on an already-failed solve.
fn reportRowAt(flag:u32,stage:u32,row:u32){if(arrayLength(&control)>0u){
 atomicOr(&control[0],flag);
 if(arrayLength(&control)>7u){let claim=atomicCompareExchangeWeak(&control[6],0u,stage);
  if(claim.exchanged){atomicStore(&control[7],row);}}
}}
fn acceptedBank()->u32{return accepted[4]&1u;}
// Which half of the ping-pong arena the shared row image is taken of. It is a
// literal written once per entry point, so it is invocation-uniform and no
// control flow depends on storage.
var<private> shellSourceRow: u32 = 0u;
fn operandAt(row:u32)->f32{return hybrid[shellSourceRow+row];}
// Exactly what arrayLength() reported when hybridA and hybridB were separate
// rowCapacity-word buffers, so applyRowImage's bound check is unchanged.
fn operandCount()->u32{return params.rowCapacity;}
fn diagonalAt(row: u32) -> f32 {
  return section63Coefficients[(accepted[4] & 1u) * p.hierarchy.w + row * 19u];
}
fn validDiagonal(row: u32) -> bool {
  let diagonal = diagonalAt(row);
  return finite(diagonal) && diagonal > 0.0;
}
// The Section 6.3 addressing this shell used to transcribe with prefix loops is
// now the operator's own, from the same layout uniform. The band dilation and
// the fused sweep therefore walk one implementation of the hierarchy.
${octreeSection63RowAddressWGSL}
fn validSection63Row(row:u32)->bool{if(row>=rows()||row>=arrayLength(&metrics)){return false;}let base=coefficientBase(row);if(base+19u>arrayLength(&section63Coefficients)){return false;}let m=metrics[row];if(m.error!=0u||(m.transformAndFlags&0x80000000u)==0u){return false;}for(var channel=0u;channel<19u;channel+=1u){let c=section63Coefficients[base+channel];if(!finite(c)||c<0.0){return false;}}return section63Coefficients[base]>0.0;}
fn section63Class(row:u32)->u32{let base=coefficientBase(row);var off=0.0;for(var channel=1u;channel<19u;channel+=1u){off+=section63Coefficients[base+channel];}let physicalBoundary=(metrics[row].transformAndFlags&0x3f00u)!=0u;let boundary=physicalBoundary||section63Coefficients[base]>off+max(1e-6,1e-5*abs(off));let transition=metrics[row].caseId!=0u;return select(select(0u,2u,boundary),select(1u,3u,boundary),transition);}
${octreeSection63DirectionChannelWGSL}
// The operator's own per-row image, verbatim. The shell evaluates it inside its
// band sweep; nothing here is a second transcription of the Section 6.3 stencil.
${octreeSection63RowImageWGSL}
fn bandAt(row:u32,useB:bool)->u32{return select(bandA[row],bandB[row],useB);}
fn dilatedBand(row:u32,useB:bool)->u32{var value=bandAt(row,useB);let h=geometry[row];let m=metrics[row];let l=countTrailingZeros(h.y);let q=originOf(h)/(1u<<l);let page=pageFor(l,q);let base=coefficientBase(row);for(var channel=0u;channel<18u&&value==0u;channel+=1u){if(section63Coefficients[base+1u+channel]==0.0){continue;}let targetQ=vec3i(q)+worldDirection(canonicalDirection(channel),m.transformAndFlags&63u);if(any(targetQ<vec3i(0))||any(targetQ>=vec3i(dims(l)))){continue;}let slot=pageSlot(l,page,q,vec3u(targetQ),row);if(slot==INVALID||(state[at(FLAGS,l,slot)]&MG_ONLY)!=0u){continue;}let encoded=state[at(OWNER,l,slot)];if(encoded>0u&&encoded<=rows()){value=max(value,bandAt(encoded-1u,useB));}}
 if(l>0u&&value==0u){let fine=l-1u;for(var child=0u;child<8u&&value==0u;child+=1u){let ghostQ=2u*q+vec3u(child&1u,(child>>1u)&1u,(child>>2u)&1u);let ghostPage=pageFor(fine,ghostQ);if(ghostPage==INVALID){continue;}let ghost=pageSlot(fine,ghostPage,ghostQ,ghostQ,row);if(ghost==INVALID||(state[at(FLAGS,fine,ghost)]&GHOST)==0u||state[at(OWNER,fine,ghost)]!=row+1u){continue;}for(var candidate=0u;candidate<18u&&value==0u;candidate+=1u){let delta=canonicalDirection(candidate);let activeQ=vec3i(ghostQ)-delta;if(any(activeQ<vec3i(0))||any(activeQ>=vec3i(dims(fine)))){continue;}let activeSlot=pageSlot(fine,ghostPage,ghostQ,vec3u(activeQ),row);if(activeSlot==INVALID||(state[at(FLAGS,fine,activeSlot)]&ACTIVE)==0u){continue;}let encoded=state[at(OWNER,fine,activeSlot)];if(encoded>0u&&encoded<=rows()&&coefficientForDirection(encoded-1u,metrics[encoded-1u],delta)>0.0){value=max(value,bandAt(encoded-1u,useB));}}}}
 return value;}
/** Row index of the A half; B follows one whole row capacity later. */
fn hybridRow(row: u32, useB: bool) -> u32 {
  return select(row, params.rowCapacity + row, useB);
}
fn bandWorksetBase(cls: u32) -> u32 {
  return (accepted[4] & 1u) * params.worksetBankStrideWords
    + cls * params.worksetStrideWords;
}
// The union class record, guarded by BOTH header tests the split encode used:
// this shell's finalized-publication test, and the operator's own workRow
// bounds. bandRow-valid already implied workRow-valid (the count can never
// exceed the capacity word compactBandIntersections fails closed against), so
// the conjunction admits the exact rows that used to receive an image; keeping
// the operator's terms means a fused sweep can never smooth a row the separate
// apply would have skipped.
fn bandRow(item: u32) -> u32 {
  let base = bandWorksetBase(4u);
  if (base + 7u > arrayLength(&bandWorksets)
    || atomicLoad(&bandWorksets[base]) != accepted[3]
    || atomicLoad(&bandWorksets[base + 3u]) != 3u
    || atomicLoad(&bandWorksets[base + 1u]) > atomicLoad(&bandWorksets[base + 2u])
    || item >= atomicLoad(&bandWorksets[base + 1u])
    || base + 7u + item >= arrayLength(&bandWorksets)) {
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
// The image argument is this row's Section 6.3 operator image of the state the
// sweep reads. It used to be staged in a whole extra dispatch and read back
// from operatorImage; the fused sweep hands it over in a register.
//
// The bandB[row]==0u early-out the split form carried is deleted, not dropped:
// the union class record is written only by compactBandIntersections, which
// admits a row only when bandB[row]!=0u, and bandB is written only by
// the setup dilation, which cannot run between the publication and this sweep.
// It was therefore never taken here, and reading it would put an eleventh
// storage binding on the portable ten-buffer ceiling.
fn smoothValue(row: u32, image: f32) -> f32 {
  let current = operandAt(row);
  let next = current
    + params.damping * (rhs[row] - image) / diagonalAt(row);
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
  hybrid[hybridRow(row, false)] = 0.0;
  hybrid[hybridRow(row, true)] = smoothZeroValue(row);
}
// The fused shell sweep. Damped Jacobi reads only the source half of the arena
// and writes only the destination half, so evaluating this row's own operator
// image from that same source vector is not a hazard: no invocation reads a
// word another invocation writes. applyRowImage is the operator's own
// function, so the arithmetic and its order are the merged-band apply's, term
// for term. A row whose image is invalid reported and stored nothing in the
// split form, and the sweep that followed it saw the error flag and did
// nothing; here it returns before writing, which is the same fail-closed
// outcome one dispatch earlier.
@compute @workgroup_size(64)
fn smoothBtoA(@builtin(global_invocation_id) global: vec3u) {
  shellSourceRow = params.rowCapacity;
  let row = bandRow(global.x);
  if (!stopped() && row != INVALID && row < rows()) {
    let image = applyRowImage(row);
    if (image.valid) { hybrid[hybridRow(row, false)] = smoothValue(row, image.value); }
  }
}
@compute @workgroup_size(64)
fn smoothAtoB(@builtin(global_invocation_id) global: vec3u) {
  shellSourceRow = 0u;
  let row = bandRow(global.x);
  if (!stopped() && row != INVALID && row < rows()) {
    let image = applyRowImage(row);
    if (image.valid) { hybrid[hybridRow(row, true)] = smoothValue(row, image.value); }
  }
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
  let value = hybrid[hybridRow(row, false)] + innerCorrection[row];
  if (!finite(value)) { reportAt(NONFINITE_ERROR, 13u, row); }
  // Paper §4.3 step (3) starts the matching post-sweep from p2 everywhere.
  // Only band rows are subsequently ping-ponged, so both states must carry
  // the domain-wide M1 correction before that compact dispatch begins.
  else { hybrid[hybridRow(row, false)] = value; hybrid[hybridRow(row, true)] = value; }
}
@compute @workgroup_size(64)
fn publishCorrection(@builtin(global_invocation_id) global: vec3u) {
  let row = global.x;
  if (stopped() || row >= rows()) { return; }
  let value = hybrid[hybridRow(row, true)];
  if (!finite(value)) { reportAt(NONFINITE_ERROR, 14u, row); }
  else { correction[row] = value; }
}

`;
