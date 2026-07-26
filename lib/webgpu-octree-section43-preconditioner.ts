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
  | "publishCorrection"
  | "persistentCorrection";

const HYBRID_BINDINGS: Readonly<Record<HybridStage, readonly number[]>> = Object.freeze({
  resetBandWorksets: [0, 12, 15, 16, 17],
  classifyBand: [0, 1, 10, 12, 13, 17, 23],
  dilateBandAtoB: [0, 1, 2, 10, 11, 12, 13, 17, 21, 22, 23],
  dilateBandBtoA: [0, 1, 2, 10, 11, 12, 13, 17, 21, 22, 23],
  compactBandIntersections: [0, 1, 11, 12, 13, 15, 17, 23],
  finalizeBandWorksets: [0, 11, 12, 15, 16, 17, 24],
  prepareCorrectionDispatches: [0, 12, 16, 17, 19, 20],
  smoothZeroToB: [0, 1, 4, 6, 7, 11, 12, 13, 17],
  smoothBtoA: [0, 1, 4, 6, 7, 11, 12, 13, 14, 15, 17],
  smoothAtoB: [0, 1, 4, 6, 7, 11, 12, 13, 14, 15, 17],
  formInnerResidual: [0, 4, 8, 12, 14, 17],
  addInnerCorrection: [0, 6, 7, 9, 12, 17],
  publishCorrection: [0, 5, 7, 12, 17],
  persistentCorrection: [0, 1, 2, 4, 5, 12, 13, 17, 21, 22, 24, 26],
});

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
    readonly dispatch: GPUBuffer;
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
  /** Four row-capacity channels: paper p1 ping/pong, r1, and fixed shell mask. */
  private readonly persistentArena: GPUBuffer;
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
    this.hybridA = vector("Section 4.3 hybrid L2 iterate A");
    this.hybridB = vector("Section 4.3 hybrid L2 iterate B");
    this.innerRhs = vector("Section 4.3 hybrid L1 residual");
    this.innerCorrection = vector("Section 4.3 hybrid L1 correction");
    this.bandA = vector("Section 4.3 hybrid boundary band A");
    this.bandB = vector("Section 4.3 hybrid boundary band B");
    this.operatorImage = vector("Section 4.3 hybrid resolved L2 image");
    this.persistentArena = device.createBuffer({
      label: "Section 4.3 persistent correction arena",
      size: 4 * vectorBytes,
      usage: storage,
    });
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
      throw new Error("Section 4.3 persistent correction requires the exact symmetric SPGrid Chebyshev contract");
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
    this.encodedCorrectionDispatchCount = 1;
    this.encodedPassTransitionCount = (inner.encodedPassTransitionCount ?? 0) + 2;
    this.allocatedBytes = this.params.size + 7 * vectorBytes
      + this.bandWorksets.size + this.bandDispatch.size
      + this.gatedRowDispatch.size + this.gatedBandDispatch.size
      + this.persistentArena.size;
    const mergedBandApplies = 2 * this.boundarySmoothingIterations - 1;
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
    this.runAccepted(pass, "classifyBand", resources);
    this.runAccepted(pass, "dilateBandAtoB", resources);
    this.runAccepted(pass, "dilateBandBtoA", resources);
    this.runAccepted(pass, "dilateBandAtoB", resources);
    this.runAccepted(pass, "compactBandIntersections", resources);
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
    const pass = broker.compute({ label: "Section 4.3 paper-exact persistent correction" });
    this.runDirect(pass, "persistentCorrection", resources);
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
      this.persistentArena,
      undefined,
      section63.dispatch,
    ];
  }

  private run(
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
    pass.dispatchWorkgroupsIndirect(
      this.gatedRowDispatch,
      0,
    );
  }

  private runAccepted(
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

  private runDirect(
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
    pass.dispatchWorkgroups(1);
  }

  private runBand(
    pass: GPUComputePassEncoder,
    name: "smoothBtoA" | "smoothAtoB",
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
    pass.dispatchWorkgroupsIndirect(this.gatedBandDispatch, 4 * 12);
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
    this.persistentArena.destroy();
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
@group(0) @binding(24) var<storage, read_write> persistentArena: array<f32>;
@group(0) @binding(26) var<storage, read> spgridDispatch: array<u32>;

const INVALID_ROW_ERROR = 2u;
const NONFINITE_ERROR = 4u;
const INVALID = 0xffffffffu;
const ACTIVE=1u;const GHOST=2u;const MG_ONLY=4u;const KEY=0u;const FLAGS=1u;
const DIAG=2u;const XP=3u;const RHS_CHANNEL=21u;const A_CHANNEL=22u;const B_CHANNEL=23u;
const OWNER=24u;const SPECTRAL=25u;const STATE_CHANNELS=26u;const PAGE_RECORD_WORDS=28u;
const DISPATCH_WORDS=12u;const PAGE_X=8u;const PAGE_Y=8u;const PAGE_Z=4u;
const HALO_X=10u;const HALO_Y=10u;const HALO_Z=6u;const PAGE_ELEMENTS=256u;const HALO_ELEMENTS=600u;
var<workgroup> persistentPageSlots: array<u32,600>;
var<workgroup> persistentPageA: array<f32,600>;
var<workgroup> persistentPageB: array<f32,600>;
var<workgroup> persistentPageRhs: array<f32,600>;
var<workgroup> persistentPageDiagonal: array<f32,600>;

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
fn coefficientForDirection(row:u32,direction:vec3i)->f32{let m=metrics[row];let base=coefficientBase(row);for(var channel=0u;channel<18u;channel+=1u){if(all(worldDirection(canonicalDirection(channel),m.transformAndFlags&63u)==direction)){return section63Coefficients[base+1u+channel];}}return 0.0;}
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
  // Persist the immutable three-layer shell mask in the correction arena so
  // the hot fused kernel does not need an eleventh storage binding.
  for (var row = 0u; row < rows(); row += 1u) {
    persistentArena[3u * params.rowCapacity + row] = select(0.0, 1.0, bandB[row] != 0u);
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

// Persistent paper §4.3 executor. One 128-lane workgroup owns every ordering
// boundary that was formerly expressed as a host dispatch. The arithmetic is
// unchanged: exact destination-owned L2, matched Jacobi halves, and the
// symmetric rediscretized SPGrid M1 with reversed Chebyshev phases.
fn pArena(channel:u32,row:u32)->u32{return channel*params.rowCapacity+row;}
fn pValue(channel:u32,row:u32)->f32{return persistentArena[pArena(channel,row)];}
fn pSet(channel:u32,row:u32,value:f32){persistentArena[pArena(channel,row)]=value;}
fn pBand(row:u32)->bool{return pValue(3u,row)>0.5;}
fn pLoad(channel:u32,l:u32,slot:u32)->f32{return bitcast<f32>(state[at(channel,l,slot)]);}
fn pStore(channel:u32,l:u32,slot:u32,value:f32){state[at(channel,l,slot)]=bitcast<u32>(value);}
fn pCount(l:u32)->u32{return spgridDispatch[l*DISPATCH_WORDS];}
fn pPageCount(l:u32)->u32{return spgridDispatch[l*DISPATCH_WORDS+8u];}
fn pWorkSlot(l:u32,item:u32)->u32{return topology[workBase()+levelBase(l)+item];}
fn pRowMap(l:u32,row:u32)->u32{return topology[rowMapBase()+l*capacity()+row];}
fn pTransferWord(l:u32,item:u32,word:u32)->u32{return transferBase()+transferLevelOffset(l)+4u*item+word;}
fn pParentHeadBase(l:u32)->u32{return transferBase()+transferLevelOffset(l)+4u*transferCapacity(l);}
fn pParentTailBase(l:u32)->u32{return pParentHeadBase(l)+levelCapacity(l);}
fn pFineHeadBase(l:u32)->u32{return pParentTailBase(l)+levelCapacity(l);}
fn pFineCountBase(l:u32)->u32{return pFineHeadBase(l)+levelCapacity(l);}
fn pTransferCount(l:u32)->u32{return spgridDispatch[l*DISPATCH_WORDS+1u];}
fn pSync(){storageBarrier();workgroupBarrier();}
fn pFind(l:u32,q:vec3u)->u32{
  if(l>=levels()||any(q>=dims(l))){return INVALID;}
  let record=brickRecord(l,q);let bit=localBit(q);let low=topology[record+1u];let high=topology[record+2u];
  if(((select(low,high,bit>=32u)>>(bit&31u))&1u)==0u){return INVALID;}
  let lower=select((1u<<(bit&31u))-1u,0xffffffffu,bit>=32u);var rank=countOneBits(low&lower);
  if(bit>=32u){rank+=countOneBits(high&((1u<<(bit-32u))-1u));}
  let slot=topology[rankedSlotsBase()+levelBase(l)+topology[record+3u]+rank];
  if(slot>=levelCapacity(l)||state[at(KEY,l,slot)]!=1u+q.x+dims(l).x*(q.y+dims(l).y*q.z)){
    reportAt(INVALID_ROW_ERROR,40u,slot);return INVALID;
  }
  return slot;
}
fn pFinerAdjoint(row:u32,q:vec3u,l:u32,x:f32,source:u32)->f32{
  if(l==0u){return 0.0;}let fine=l-1u;var result=0.0;
  for(var child=0u;child<8u;child+=1u){let ghostQ=2u*q+vec3u(child&1u,(child>>1u)&1u,(child>>2u)&1u);
    let ghostPage=pageFor(fine,ghostQ);if(ghostPage==INVALID||ghostPage>=levelCapacity(fine)){continue;}
    let ghost=pageSlot(fine,ghostPage,ghostQ,ghostQ);
    if(ghost==INVALID||(state[at(FLAGS,fine,ghost)]&GHOST)==0u||state[at(OWNER,fine,ghost)]!=row+1u){continue;}
    for(var candidate=0u;candidate<18u;candidate+=1u){let delta=canonicalDirection(candidate);let activeQ=vec3i(ghostQ)-delta;
      if(any(activeQ<vec3i(0))||any(activeQ>=vec3i(dims(fine)))){continue;}
      let activeSlot=pageSlot(fine,ghostPage,ghostQ,vec3u(activeQ));
      if(activeSlot==INVALID||(state[at(FLAGS,fine,activeSlot)]&ACTIVE)==0u){continue;}
      let encoded=state[at(OWNER,fine,activeSlot)];if(encoded==0u||encoded>rows()){reportAt(INVALID_ROW_ERROR,41u,row);continue;}
      let other=encoded-1u;var c=0.0;let transform=geometry[other].w&63u;let base=coefficientBase(other);
      for(var channel=0u;channel<18u;channel+=1u){if(all(worldDirection(canonicalDirection(channel),transform)==delta)){c=section63Coefficients[base+1u+channel];}}
      if(c>0.0){result+=c*(x-pValue(source,other));}
    }
  }
  return result;
}
fn pApplyL2(row:u32,source:u32)->f32{
  if(row>=rows()||row>=arrayLength(&geometry)){reportAt(INVALID_ROW_ERROR,42u,row);return 0.0;}
  let h=geometry[row];let base=coefficientBase(row);
  if((h.w&0x80000000u)==0u||base+19u>arrayLength(&section63Coefficients)){reportAt(INVALID_ROW_ERROR,43u,row);return 0.0;}
  let l=countTrailingZeros(h.y);let q=originOf(h)/(1u<<l);let page=pageFor(l,q);
  if(page==INVALID||page>=levelCapacity(l)){reportAt(INVALID_ROW_ERROR,44u,row);return 0.0;}
  let x=pValue(source,row);var sum=0.0;for(var channel=0u;channel<18u;channel+=1u){sum+=section63Coefficients[base+1u+channel];}
  var value=max(0.0,section63Coefficients[base]-sum)*x;
  for(var channel=0u;channel<18u;channel+=1u){let c=section63Coefficients[base+1u+channel];if(c==0.0){continue;}
    let targetQ=vec3i(q)+worldDirection(canonicalDirection(channel),h.w&63u);
    if(any(targetQ<vec3i(0))||any(targetQ>=vec3i(dims(l)))){reportAt(INVALID_ROW_ERROR,45u,row);continue;}
    let slot=pageSlot(l,page,q,vec3u(targetQ));if(slot==INVALID){reportAt(INVALID_ROW_ERROR,46u,row);continue;}
    if((state[at(FLAGS,l,slot)]&MG_ONLY)!=0u){continue;}let encoded=state[at(OWNER,l,slot)];
    if(encoded==0u||encoded>rows()){reportAt(INVALID_ROW_ERROR,47u,row);continue;}
    value+=c*(x-pValue(source,encoded-1u));
  }
  value+=pFinerAdjoint(row,q,l,x,source);
  if(!finite(value)){reportAt(NONFINITE_ERROR,48u,row);return 0.0;}return value;
}
fn pStencilDirection(k:u32)->vec3i{return canonicalDirection(k);}
fn pApplied(l:u32,slot:u32,source:u32)->f32{
  var value=pLoad(DIAG,l,slot)*pLoad(source,l,slot);let q=decode(state[at(KEY,l,slot)],l);
  for(var k=0u;k<18u;k+=1u){let c=pLoad(XP+k,l,slot);if(c==0.0){continue;}
    let targetQ=vec3i(q)+pStencilDirection(k);if(any(targetQ<vec3i(0))||any(targetQ>=vec3i(dims(l)))){reportAt(INVALID_ROW_ERROR,49u,slot);continue;}
    let other=pFind(l,vec3u(targetQ));if(other==INVALID){reportAt(INVALID_ROW_ERROR,50u,slot);continue;}value-=c*pLoad(source,l,other);
  }
  return value;
}
fn pChebyshevWeight(l:u32,phase:u32)->f32{
  let upper=pLoad(SPECTRAL,l,0u);let lower=upper/30.0;
  if(!(lower>0.0)||!(upper>lower)||!finite(upper)){reportAt(INVALID_ROW_ERROR,51u,l);return 0.0;}
  let centre=0.5*(upper+lower);let radius=0.5*(upper-lower);let degree=params.m1SmoothingDegree;
  return 1.0/(centre-radius*cos(3.141592653589793*(2.0*f32(phase)+1.0)/(2.0*f32(degree))));
}
fn pInteriorHalo(local:u32)->u32{let x=local%PAGE_X;let yz=local/PAGE_X;return x+1u+HALO_X*((yz%PAGE_Y)+1u+HALO_Y*((yz/PAGE_Y)+1u));}
fn pPageApplied(l:u32,slot:u32,origin:vec3u,halo:u32,useB:bool)->f32{
  var value=persistentPageDiagonal[halo]*select(persistentPageA[halo],persistentPageB[halo],useB);
  for(var k=0u;k<18u;k+=1u){let c=pLoad(XP+k,l,slot);if(c==0.0){continue;}
    let relative=vec3i(decode(state[at(KEY,l,slot)],l))+pStencilDirection(k)-vec3i(origin)+vec3i(1);
    if(any(relative<vec3i(0))||any(relative>=vec3i(i32(HALO_X),i32(HALO_Y),i32(HALO_Z)))){reportAt(INVALID_ROW_ERROR,52u,slot);continue;}
    let neighbour=u32(relative.x)+HALO_X*(u32(relative.y)+HALO_Y*u32(relative.z));
    if(persistentPageSlots[neighbour]==INVALID){reportAt(INVALID_ROW_ERROR,53u,slot);continue;}
    value-=c*select(persistentPageA[neighbour],persistentPageB[neighbour],useB);
  }
  return value;
}
fn pSmoothLevel(l:u32,reverse:bool,lane:u32){
  // The full-level B snapshot makes page execution true block Jacobi. It also
  // makes the reverse half the exact transpose schedule independent of page order.
  for(var item=lane;item<pCount(l);item+=128u){let slot=pWorkSlot(l,item);pStore(B_CHANNEL,l,slot,pLoad(A_CHANNEL,l,slot));}
  pSync();
  for(var page=0u;page<pPageCount(l);page+=1u){let origin=decode(topology[pageRecord(l,page)],l);let d=dims(l);
    for(var halo=lane;halo<HALO_ELEMENTS;halo+=128u){let x=halo%HALO_X;let yz=halo/HALO_X;
      let q=vec3i(origin)+vec3i(i32(x)-1,i32(yz%HALO_Y)-1,i32(yz/HALO_Y)-1);var slot=INVALID;
      if(all(q>=vec3i(0))&&all(q<vec3i(d))){slot=pageSlot(l,page,origin,vec3u(q));}persistentPageSlots[halo]=slot;
      var value=0.0;var rv=0.0;var diagonal=1.0;if(slot!=INVALID){value=pLoad(B_CHANNEL,l,slot);rv=pLoad(RHS_CHANNEL,l,slot);diagonal=pLoad(DIAG,l,slot);}
      persistentPageA[halo]=value;persistentPageB[halo]=value;persistentPageRhs[halo]=rv;persistentPageDiagonal[halo]=diagonal;
    }
    workgroupBarrier();
    for(var step=0u;step<params.m1SmoothingDegree;step+=1u){let useB=(step&1u)!=0u;
      let phase=select(step,params.m1SmoothingDegree-1u-step,reverse);let weight=pChebyshevWeight(l,phase);
      for(var local=lane;local<PAGE_ELEMENTS;local+=128u){let halo=pInteriorHalo(local);let slot=persistentPageSlots[halo];if(slot==INVALID){continue;}
        let source=select(persistentPageA[halo],persistentPageB[halo],useB);var next=source;
        if((state[at(FLAGS,l,slot)]&GHOST)==0u){let diagonal=persistentPageDiagonal[halo];
          if(!(diagonal>0.0)){reportAt(INVALID_ROW_ERROR,54u,slot);}else{next=source+weight*(persistentPageRhs[halo]-pPageApplied(l,slot,origin,halo,useB))/diagonal;}}
        if(!finite(next)){reportAt(NONFINITE_ERROR,55u,slot);next=source;}
        if(useB){persistentPageA[halo]=next;}else{persistentPageB[halo]=next;}
      }
      workgroupBarrier();
    }
    for(var local=lane;local<PAGE_ELEMENTS;local+=128u){let halo=pInteriorHalo(local);let slot=persistentPageSlots[halo];
      if(slot!=INVALID){pStore(A_CHANNEL,l,slot,persistentPageA[halo]);}}
    pSync();
  }
}
fn pRestrict(l:u32,lane:u32){
  for(var item=lane;item<pCount(l+1u);item+=128u){let coarse=pWorkSlot(l+1u,item);var sum=0.0;var record=topology[pParentHeadBase(l)+coarse];
    for(var visited=0u;record!=INVALID&&visited<pTransferCount(l);visited+=1u){if(record>=pTransferCount(l)){reportAt(INVALID_ROW_ERROR,56u,coarse);break;}
      let fine=topology[pTransferWord(l,record,0u)];let parent=topology[pTransferWord(l,record,1u)];if(parent!=coarse){reportAt(INVALID_ROW_ERROR,57u,coarse);break;}
      let ghost=(state[at(FLAGS,l,fine)]&GHOST)!=0u;let product=pApplied(l,fine,A_CHANNEL);
      let residual=select(-product,pLoad(RHS_CHANNEL,l,fine)-product,!ghost);
      sum+=bitcast<f32>(topology[pTransferWord(l,record,2u)])*residual;record=topology[pTransferWord(l,record,3u)];
    }
    if(record!=INVALID||!finite(sum)){reportAt(INVALID_ROW_ERROR,58u,coarse);}else{pStore(RHS_CHANNEL,l+1u,coarse,sum);}
  }
}
fn pProlong(l:u32,lane:u32){
  for(var item=lane;item<pCount(l);item+=128u){let fine=pWorkSlot(l,item);let ghost=(state[at(FLAGS,l,fine)]&GHOST)!=0u;
    let targets=select(8u,1u,ghost);var value=select(pLoad(A_CHANNEL,l,fine),0.0,ghost);var valid=true;
    let first=topology[pFineHeadBase(l)+fine];let owned=topology[pFineCountBase(l)+fine];
    for(var corner=0u;corner<targets;corner+=1u){if(first==INVALID||corner>=owned||first+corner>=pTransferCount(l)){reportAt(INVALID_ROW_ERROR,59u,fine);valid=false;break;}
      let record=first+corner;if(topology[pTransferWord(l,record,0u)]!=fine){reportAt(INVALID_ROW_ERROR,60u,fine);valid=false;break;}
      let coarse=topology[pTransferWord(l,record,1u)];let weight=bitcast<f32>(topology[pTransferWord(l,record,2u)]);
      if(coarse>=levelCapacity(l+1u)||!finite(weight)){reportAt(INVALID_ROW_ERROR,61u,fine);valid=false;break;}
      value+=weight*pLoad(A_CHANNEL,l+1u,coarse);
    }
    if(valid&&finite(value)){pStore(A_CHANNEL,l,fine,value);}else if(valid){reportAt(NONFINITE_ERROR,62u,fine);}
  }
}
fn pApplyM1(lane:u32){
  for(var l=0u;l<levels();l+=1u){for(var item=lane;item<pCount(l);item+=128u){let slot=pWorkSlot(l,item);
    pStore(RHS_CHANNEL,l,slot,0.0);pStore(A_CHANNEL,l,slot,0.0);pStore(B_CHANNEL,l,slot,0.0);}}
  pSync();
  for(var row=lane;row<rows();row+=128u){let native=countTrailingZeros(geometry[row].y);let slot=pRowMap(native,row);
    if(slot>=levelCapacity(native)){reportAt(INVALID_ROW_ERROR,63u,row);}else{pStore(RHS_CHANNEL,native,slot,pValue(2u,row));}}
  pSync();
  for(var l=0u;l+1u<levels();l+=1u){pSmoothLevel(l,false,lane);pRestrict(l,lane);pSync();}
  if(lane==0u){let bottom=levels()-1u;if(pCount(bottom)!=1u){reportAt(INVALID_ROW_ERROR,64u,bottom);}else{let slot=pWorkSlot(bottom,0u);let d=pLoad(DIAG,bottom,slot);
    if(!(d>0.0)){reportAt(INVALID_ROW_ERROR,65u,slot);}else{let value=pLoad(RHS_CHANNEL,bottom,slot)/d;if(finite(value)){pStore(A_CHANNEL,bottom,slot,value);}else{reportAt(NONFINITE_ERROR,66u,slot);}}}}
  pSync();
  var l=levels()-1u;while(l>0u){l-=1u;pProlong(l,lane);pSync();pSmoothLevel(l,true,lane);}
}

var<workgroup> persistentStopped:atomic<u32>;
@compute @workgroup_size(128)
fn persistentCorrection(@builtin(local_invocation_index) lane:u32){
  if(lane==0u){atomicStore(&persistentStopped,select(0u,1u,stopped()));}
  workgroupBarrier();
  if(workgroupUniformLoad(&persistentStopped)!=0u){return;}
  // §4.3(1): J^k(0,q) -> p1. k is the immutable matching shell count.
  for(var row=lane;row<rows();row+=128u){pSet(0u,row,0.0);var first=0.0;if(pBand(row)){
    let diagonal=diagonalAt(row);if(!(diagonal>0.0)){reportAt(INVALID_ROW_ERROR,67u,row);}else{first=params.damping*rhs[row]/diagonal;}}
    pSet(1u,row,first);}
  pSync();
  for(var sweep=1u;sweep<params.smoothingIterations;sweep+=1u){let source=select(0u,1u,(sweep&1u)==1u);let destination=1u-source;
    for(var row=lane;row<rows();row+=128u){if(pBand(row)){let current=pValue(source,row);let next=current+params.damping*(rhs[row]-pApplyL2(row,source))/diagonalAt(row);
      if(finite(next)){pSet(destination,row,next);}else{reportAt(NONFINITE_ERROR,68u,row);}}}
    pSync();
  }
  // §4.3(2): r1=q-L2p1 and the exact symmetric first-order V-cycle.
  for(var row=lane;row<rows();row+=128u){let value=rhs[row]-pApplyL2(row,0u);if(finite(value)){pSet(2u,row,value);}else{reportAt(NONFINITE_ERROR,69u,row);}}
  pSync();pApplyM1(lane);pSync();
  // §4.3(3): p2=p1+M1r1, followed by the identical k sweeps.
  for(var row=lane;row<rows();row+=128u){let native=countTrailingZeros(geometry[row].y);let slot=pRowMap(native,row);let value=pValue(0u,row)+pLoad(A_CHANNEL,native,slot);
    if(finite(value)){pSet(0u,row,value);pSet(1u,row,value);}else{reportAt(NONFINITE_ERROR,70u,row);}}
  pSync();
  for(var sweep=0u;sweep<params.smoothingIterations;sweep+=1u){let source=select(0u,1u,(sweep&1u)==0u);let destination=1u-source;
    for(var row=lane;row<rows();row+=128u){if(pBand(row)){let current=pValue(source,row);let next=current+params.damping*(rhs[row]-pApplyL2(row,source))/diagonalAt(row);
      if(finite(next)){pSet(destination,row,next);}else{reportAt(NONFINITE_ERROR,71u,row);}}}
    pSync();
  }
  if(!stopped()){for(var row=lane;row<rows();row+=128u){let value=pValue(1u,row);if(finite(value)){correction[row]=value;}else{reportAt(NONFINITE_ERROR,72u,row);}}}
}
`;
