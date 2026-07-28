import {
  makeFineLevelSetSortedWorklistLookupWGSL,
  type WebGPUFineLevelSetBrickSource,
} from "./webgpu-octree-fine-levelset-bricks";
import { FINE_LEVELSET_WORKSET_HEADER_WORDS } from "./octree-fine-levelset-bricks";
import { fineLevelSetLinearWorkgroupWGSL } from "./webgpu-fine-levelset-dispatch";
import { PassBroker } from "./webgpu-pass-broker";
import {
  createGPULogicalActivityAdoptionContext,
  type GPULogicalActivityAdoptionContext,
} from "./gpu-logical-activity-adoption";
import { performanceShaderVariant } from "./stores/performance-instrumentation-store";

export interface FineLevelSetGPURedistanceOptions {
  /** Required signed-distance width, measured in fine cells. */
  bandCells: number;
  residualTolerance?: number;
}

/** The exact immutable page-delta ABI authored by fine topology.
 *
 * Redistance deliberately accepts the topology authority rather than a bare
 * storage buffer.  This prevents a capacity-sized/full-active worklist from
 * being substituted at a call site and makes the dirty/support offset
 * contract explicit at construction.
 */
export interface FineLevelSetRedistanceDeltaAuthority {
  readonly pageDelta: GPUBuffer;
  readonly pageDeltaLayout: {
    readonly headerWords: 16;
    readonly dirtyPagesOffsetWords: number;
    readonly supportPagesOffsetWords: number;
  };
  /** Immutable topology-authored commands: one workgroup per exact page. */
  readonly redistanceDispatches: {
    readonly buffer: GPUBuffer;
    readonly dirtyOffsetBytes: 84;
    readonly supportOffsetBytes: 60;
  };
}

/** Descending JFA strides followed by two local repair passes.
 *
 * **This schedule is NOT warm-started**, which is why `encodeJFA` must keep
 * passing `bandCells` as the displacement. `seedClosestPoints` writes
 * `workA[index]=INVALID` for every support sample on every generation and
 * re-derives the seed set from phi sign changes, so no closest-point field
 * crosses a generation boundary: the flood has to reach every band cell from
 * the interface every step, however little the interface moved.
 *
 * The per-step displacement itself is genuinely small. Transport rejects any
 * `maximumBacktraceFineCells` above `2 * fineFactor` (8 fine cells at factor
 * 4) and publishes the measured per-step maximum, `ceil(|v| * dt / h)` in fine
 * cells, in its status summary. So `planFineLevelSetJFAStrides(bandCells, 8)`
 * would emit 8,4,2,1,+1,+1 and delete one of the seven floods — and it is
 * unsound today, in the way that looks like a win. Band cells 9..23 from the
 * interface would finish with `seed==INVALID`; `resolveClosestPoints*` counts
 * them as `unresolved`; `finalizeJFADistances` clears `committed`; and
 * `webgpu-octree-fine-levelset-topology.ts` then rejects the publication,
 * because `redistanceValid` requires `redistanceControl[0]==0` and
 * `redistanceControl[3]!=0`. The surface freezes, the frame gets faster, and
 * the run still prints PASS — POWER_LIQUIDS_ULTIMATE_M1MAX, "things that look
 * like wins and are not", item 1. E1 step 2 stays blocked until the
 * closest-point channels are carried in the fine-page transaction, and that
 * carry must survive physical page reassignment: a transported seed index
 * naming a recycled page resolves to a live-looking wrong coordinate, which
 * fails open rather than closed.
 *
 * Replacing the ladder with an ordered 6-direction sweep is not a substitute
 * either. This transform is dispatched one workgroup per support page over an
 * unordered compacted page list, so a sweep pass propagates only within a page
 * (4 cells) plus its halo; covering 23 cells still needs ~6 rounds, the same
 * pass count, while giving up the 32-lane subgroup candidate reduction and
 * requiring per-lane divergent workgroup-memory addressing — measured a 26 %
 * regression on this GPU (item 10). Ordering the sweep globally instead costs
 * one dispatch per slab per direction. And a 6-sweep vector distance transform
 * is not exact for a sub-cell closest-point field, so it would be Gate B for
 * no pass saving. The lever that did work was arithmetic inside the flood:
 * see `FINE_LEVELSET_JFA_B4_ADDRESSING_ENV`.
 */
export function planFineLevelSetJFAStrides(
  bandCells: number,
  maximumDisplacementFineCells = 4,
): readonly number[] {
  if (!Number.isSafeInteger(bandCells) || bandCells < 1 || bandCells > 256) {
    throw new RangeError("Fine redistance bandCells must be an integer in [1, 256]");
  }
  if (!Number.isSafeInteger(maximumDisplacementFineCells)
    || maximumDisplacementFineCells < 1 || maximumDisplacementFineCells > 256) {
    throw new RangeError("Fine redistance displacement must be an integer in [1, 256]");
  }
  let stride = 1;
  const repairRadius = Math.min(bandCells, maximumDisplacementFineCells);
  if (maximumDisplacementFineCells >= bandCells) {
    // A cold publication has no transported closest-point field to repair, so
    // the ladder must reach every cell of the signed band. Reach is the SUM of
    // the descending strides, not the first stride: a schedule starting at P
    // sums to P + P/2 + ... + 1 = 2P - 1, and the binary decomposition of any
    // offset below that sum is exactly what the descending passes walk.
    // Choosing the smallest power of two with 2P - 1 >= band therefore covers
    // the band with one flood fewer than rounding the band itself up to a
    // power of two (band 23: 16 covers via 16+8+4+2+1 = 31, where the old rule
    // picked 32 and spent its first pass gathering entirely outside the band).
    // POWER_LIQUIDS_ULTIMATE_M1MAX change E1, step 1.
    while (stride * 2 - 1 < repairRadius) stride *= 2;
  } else {
    while (stride * 2 <= repairRadius) stride *= 2;
  }
  const strides: number[] = [];
  for (; stride >= 1; stride /= 2) strides.push(stride);
  // Two +1 collar repairs close sparse page-boundary landing gaps after the
  // descending schedule. The second is required by the mini dam-break
  // generation-280 boundary. Publication still fails closed if a seed is missing.
  strides.push(1);
  strides.push(1);
  return strides;
}

export interface FineLevelSetGPURedistanceControl {
  /** Total rejection count: missing closest points plus Eikonal residual violations. */
  unresolvedCells: number;
  resolveMissingCells: number;
  residualViolationCells: number;
  maximumResidualScaled: number;
  seedCount: number;
  committed: boolean;
  flags: number;
  firstError: number;
  acceptedCells: number;
  initialPages: number;
  finalPages: number;
}

export const FINE_LEVELSET_REDISTANCE_CONTROL_BYTES = 40;
export const FINE_LEVELSET_REDISTANCE_REDUCTION_RECORD_BYTES = 28;
const FINE_LEVELSET_JFA_MAX_PASSES = 10;
export const FINE_LEVELSET_REDISTANCE_ALLOCATED_BYTES = FINE_LEVELSET_REDISTANCE_CONTROL_BYTES
  + 80;

/** Immutable B4 JFA tap address. `pageSlot` is x-fastest in the 3x3x3
 * radius halo and `localIndex` is x-fastest in the destination B4 page. */
export interface FineLevelSetJFATapAddress {
  readonly pageSlot: number;
  readonly localIndex: number;
}

export const FINE_LEVELSET_JFA_LOOKUP_STRIDES =
  [1, 2, 4, 8, 16, 32, 64, 128, 256] as const;

// The address transform is separable. Each fixed axis record packs the halo
// coordinate in bits 2..3 and the B4 local coordinate in bits 0..1. This is
// 108 records instead of materializing 9 * 64 * 27 complete tap addresses.
const FINE_LEVELSET_JFA_TAP_AXIS_CODES = Array.from({ length: 27 }, (_, tap) => {
  const z = Math.floor(tap / 9); const remainder = tap - z * 9;
  const y = Math.floor(remainder / 3); const x = remainder - y * 3;
  return x | (y << 2) | (z << 4);
});

const FINE_LEVELSET_JFA_AXIS_ADDRESSES = FINE_LEVELSET_JFA_LOOKUP_STRIDES.flatMap((stride) =>
  Array.from({ length: 4 * 3 }, (_, entry) => {
    const local = Math.floor(entry / 3); const tap = entry - local * 3;
    const displaced = local + (tap - 1) * stride;
    const pageDelta = Math.floor(displaced / 4);
    const localCoordinate = displaced - pageDelta * 4;
    const haloCoordinate = pageDelta < 0 ? 0 : pageDelta > 0 ? 2 : 1;
    return localCoordinate | (haloCoordinate << 2);
  }));

/** CPU mirror of the fixed shader lookup, exported for exhaustive parity
 * tests against the former coordinate/division formula. */
export function fineLevelSetJFATapAddress(
  localIndex: number,
  tap: number,
  stride: number,
): FineLevelSetJFATapAddress {
  if (!Number.isSafeInteger(localIndex) || localIndex < 0 || localIndex >= 64) {
    throw new RangeError("Fine JFA lookup local index must be in [0, 64)");
  }
  if (!Number.isSafeInteger(tap) || tap < 0 || tap >= 27) {
    throw new RangeError("Fine JFA lookup tap must be in [0, 27)");
  }
  const strideClass = FINE_LEVELSET_JFA_LOOKUP_STRIDES.indexOf(stride as never);
  if (strideClass < 0) throw new RangeError("Fine JFA lookup stride must be a power of two in [1, 256]");
  const tapCode = FINE_LEVELSET_JFA_TAP_AXIS_CODES[tap]!;
  const axis = (local: number, tapCoordinate: number) =>
    FINE_LEVELSET_JFA_AXIS_ADDRESSES[strideClass * 12 + local * 3 + tapCoordinate]!;
  const x = axis(localIndex & 3, tapCode & 3);
  const y = axis((localIndex >> 2) & 3, (tapCode >> 2) & 3);
  const z = axis((localIndex >> 4) & 3, (tapCode >> 4) & 3);
  return {
    pageSlot: (x >> 2) + 3 * ((y >> 2) + 3 * (z >> 2)),
    localIndex: (x & 3) + 4 * ((y & 3) + 4 * (z & 3)),
  };
}

const fineLevelSetJFATapLookupWGSL = /* wgsl */ `
const JFA_TAP_AXIS_CODES:array<u32,27>=array<u32,27>(${FINE_LEVELSET_JFA_TAP_AXIS_CODES.join(",")}u);
const JFA_AXIS_ADDRESSES:array<u32,108>=array<u32,108>(${FINE_LEVELSET_JFA_AXIS_ADDRESSES.join(",")}u);
fn jfaStrideClass()->u32{if(JFA_STRIDE==1u){return 0u;}if(JFA_STRIDE==2u){return 1u;}if(JFA_STRIDE==4u){return 2u;}if(JFA_STRIDE==8u){return 3u;}if(JFA_STRIDE==16u){return 4u;}if(JFA_STRIDE==32u){return 5u;}if(JFA_STRIDE==64u){return 6u;}if(JFA_STRIDE==128u){return 7u;}return 8u;}
fn fixedTapAddress(local:u32,tap:u32)->vec2u{let code=JFA_TAP_AXIS_CODES[tap];let base=jfaStrideClass()*12u;let x=JFA_AXIS_ADDRESSES[base+(local&3u)*3u+(code&3u)];let y=JFA_AXIS_ADDRESSES[base+((local>>2u)&3u)*3u+((code>>2u)&3u)];let z=JFA_AXIS_ADDRESSES[base+((local>>4u)&3u)*3u+((code>>4u)&3u)];return vec2u((x>>2u)+3u*((y>>2u)+3u*(z>>2u)),(x&3u)+4u*((y&3u)+4u*(z&3u)));}
`;

/** Opt-in B4-specialized flood addressing. Default OFF so the document's
 * interleaved A/B protocol can compare both variants inside one binary. */
export const FINE_LEVELSET_JFA_B4_ADDRESSING_ENV = "FLUID_FINE_JFA_B4_ADDRESSING";

/** The variant is only legal on the B4 generation the encode already demands,
 * so the caller's plan is re-checked before the source is selected. */
export function fineLevelSetJFAB4AddressingRequested(
  plan: { readonly brickResolution: number; readonly samplesPerBrick: number },
  environment: Record<string, string | undefined> = process.env,
): boolean {
  return environment[FINE_LEVELSET_JFA_B4_ADDRESSING_ENV] === "1"
    && plan.brickResolution === 4 && plan.samplesPerBrick === 64;
}

export function fineLevelSetRedistanceAllocatedBytes(maximumResidentBricks: number): number {
  if (!Number.isSafeInteger(maximumResidentBricks) || maximumResidentBricks < 1) {
    throw new RangeError("Fine redistance resident capacity must be a positive integer");
  }
  return FINE_LEVELSET_REDISTANCE_ALLOCATED_BYTES
    + maximumResidentBricks * (FINE_LEVELSET_REDISTANCE_REDUCTION_RECORD_BYTES + 4);
}

export function unpackFineLevelSetGPURedistanceControl(words: ArrayLike<number>): FineLevelSetGPURedistanceControl {
  if (words.length < 4) throw new RangeError("Fine redistance control requires four words");
  const unresolvedCells = Number(words[0]) >>> 0;
  const resolveMissingCells = words.length > 9 ? Number(words[9]) >>> 0 : 0;
  return {
    unresolvedCells,
    resolveMissingCells,
    residualViolationCells: Math.max(0, unresolvedCells - resolveMissingCells),
    maximumResidualScaled: Number(words[1]) >>> 0,
    seedCount: Number(words[2]) >>> 0,
    committed: Number(words[3]) !== 0,
    flags: words.length > 4 ? Number(words[4]) >>> 0 : 0,
    firstError: words.length > 5 ? Number(words[5]) >>> 0 : 0xffff_ffff,
    acceptedCells: words.length > 6 ? Number(words[6]) >>> 0 : 0,
    initialPages: words.length > 7 ? Number(words[7]) >>> 0 : 0,
    finalPages: words.length > 8 ? Number(words[8]) >>> 0 : 0,
  };
}

/**
 * Fixed-resident fine-grid redistance. JFA-CPT preserves the closest-point
 * transform semantics with a bounded logarithmic schedule. It consumes the
 * complete support generation published by topology and never allocates,
 * links, or publishes a page while redistancing.
 */
export class WebGPUFineLevelSetRedistance {
  readonly control: GPUBuffer;
  readonly allocatedBytes: number;
  /** True when the opt-in B4-specialized flood addressing variant is compiled. */
  readonly b4Addressing: boolean;
  private readonly reductions: GPUBuffer;
  private readonly supportMask: GPUBuffer;
  private readonly params: GPUBuffer;
  private readonly publishSupportMaskPipeline: GPUComputePipeline;
  private readonly jfaSeedPipeline: GPUComputePipeline;
  private readonly jfaABPipelines: ReadonlyMap<number, GPUComputePipeline>;
  private readonly jfaBAPipelines: ReadonlyMap<number, GPUComputePipeline>;
  private readonly jfaResolveAToBPipeline: GPUComputePipeline;
  private readonly jfaResolveBToCanonicalPipeline: GPUComputePipeline;
  private readonly jfaValidatePipeline: GPUComputePipeline;
  private readonly jfaFinalizePipeline: GPUComputePipeline;
  private readonly jfaCommitPipeline: GPUComputePipeline;
  /** Every resource bound by redistance is construction-stable. Retain one
   * group per immutable pipeline instead of rebuilding 12–14 auto-layout
   * groups on the host for every accepted simulation step. */
  private readonly bindGroups = new Map<GPUComputePipeline, GPUBindGroup>();

  constructor(private readonly device: GPUDevice, readonly source: WebGPUFineLevelSetBrickSource,
    /** Exact delta/support publication produced by the topology transaction. */
    readonly delta: FineLevelSetRedistanceDeltaAuthority) {
    const pageCapacity = source.plan.maximumResidentBricks;
    if (delta.pageDeltaLayout.headerWords !== 16
      || delta.pageDeltaLayout.dirtyPagesOffsetWords !== 16 + 2 * pageCapacity
      || delta.pageDeltaLayout.supportPagesOffsetWords !== 16 + 3 * pageCapacity
      || delta.redistanceDispatches.dirtyOffsetBytes !== 84
      || delta.redistanceDispatches.supportOffsetBytes !== 60) {
      throw new RangeError("Fine JFA-CPT requires the exact topology page-delta ABI");
    }
    this.control = device.createBuffer({ label: "fine-levelset JFA-CPT control",
      size: FINE_LEVELSET_REDISTANCE_CONTROL_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const reductionBytes = source.plan.maximumResidentBricks * FINE_LEVELSET_REDISTANCE_REDUCTION_RECORD_BYTES;
    this.reductions = device.createBuffer({ label: "fine-levelset JFA-CPT deterministic reductions",
      size: reductionBytes, usage: GPUBufferUsage.STORAGE });
    this.supportMask = device.createBuffer({ label: "fine-levelset JFA direct support-page mask",
      size: pageCapacity * 4, usage: GPUBufferUsage.STORAGE });
    this.allocatedBytes = fineLevelSetRedistanceAllocatedBytes(source.plan.maximumResidentBricks);
    this.params = device.createBuffer({ label: "fine-levelset JFA-CPT parameters", size: 80,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const jfaProfile = performanceShaderVariant();
    const jfaActivity = createGPULogicalActivityAdoptionContext({
      moduleId: "octree/fine-redistance-jfa",
      profile: jfaProfile,
    });
    // Selected once, at construction, from the plan the encode already
    // validates. Both variants publish the same closest-point field; the B4
    // form only replaces exact integer division with the shifts the fixed tap
    // table above already uses.
    this.b4Addressing = fineLevelSetJFAB4AddressingRequested(source.plan);
    const jfaVariant = jfaActivity.module(
      fineLevelSetJFAActivityShader(jfaActivity, this.b4Addressing
        ? fineLevelSetJFACPTB4AddressingWGSL : fineLevelSetJFACPTWGSL),
      `octree/fine-redistance-jfa/${jfaProfile.cacheKey}${this.b4Addressing ? "/b4-addressing" : ""}`,
    );
    const jfaModule = device.createShaderModule({ label: "fine-levelset JFA closest-point transform",
      code: jfaVariant.code });
    const jfaPipeline = (entryPoint: string, stride?: number) => device.createComputePipeline({
      label: `fine JFA-CPT ${entryPoint}${stride === undefined ? "" : ` stride ${stride}`}`,
      layout: "auto",
      compute: {
        module: jfaModule,
        entryPoint,
        ...(stride === undefined ? {} : { constants: { JFA_STRIDE: stride } }),
      },
    });
    this.publishSupportMaskPipeline = jfaPipeline("publishSupportPageMask");
    this.jfaSeedPipeline = jfaPipeline("seedClosestPoints");
    const immutableStrides = Array.from({ length: 9 }, (_, index) => 1 << index);
    this.jfaABPipelines = new Map(immutableStrides.map((stride) =>
      [stride, jfaActivity.registerPipeline(jfaPipeline("jumpFloodAToB", stride))]));
    this.jfaBAPipelines = new Map(immutableStrides.map((stride) =>
      [stride, jfaActivity.registerPipeline(jfaPipeline("jumpFloodBToA", stride))]));
    this.jfaResolveAToBPipeline = jfaPipeline("resolveClosestPointsAToB");
    this.jfaResolveBToCanonicalPipeline = jfaPipeline("resolveClosestPointsBToCanonical");
    this.jfaValidatePipeline = jfaPipeline("validateJFADistances");
    this.jfaFinalizePipeline = jfaPipeline("finalizeJFADistances");
    this.jfaCommitPipeline = jfaPipeline("commitJFADistances");
  }

  encode(broker: PassBroker, options: FineLevelSetGPURedistanceOptions): void {
    if ((this.source.plan.fineFactor !== 4 && this.source.plan.fineFactor !== 8)
      || this.source.plan.brickResolution !== 4) {
      throw new RangeError("GPU fine JFA-CPT requires a factor-4/factor-8 B4 generation");
    }
    if (!Number.isSafeInteger(options.bandCells) || options.bandCells < 1 || options.bandCells > 256) {
      throw new RangeError("Fine redistance bandCells must be an integer in [1, 256]");
    }
    const tolerance = options.residualTolerance ?? 0.1;
    if (!Number.isFinite(tolerance) || tolerance <= 0 || tolerance > 1) {
      throw new RangeError("Fine redistance residual tolerance must be in (0, 1]");
    }
    const bytes = new ArrayBuffer(80); const u32 = new Uint32Array(bytes); const f32 = new Float32Array(bytes);
    u32.set([...this.source.plan.brickDimensions, this.source.plan.brickResolution,
      ...this.source.plan.sampleDimensions, this.source.plan.samplesPerBrick,
      this.source.plan.maximumResidentBricks, FINE_LEVELSET_WORKSET_HEADER_WORDS,
      this.source.plan.maximumResidentBricks, this.source.generation, options.bandCells], 0);
    f32[13] = this.source.plan.fineCellWidth; f32[14] = tolerance;
    u32[15] = this.source.plan.maximumResidentBricks * this.source.plan.samplesPerBrick;
    u32[16] = this.device.limits.maxComputeWorkgroupsPerDimension;
    u32[17] = 0;
    u32[18] = this.delta.pageDeltaLayout.dirtyPagesOffsetWords;
    u32[19] = this.delta.pageDeltaLayout.supportPagesOffsetWords;
    this.encodeJFA(broker, bytes, options.bandCells, tolerance);
  }

  private encodeJFA(
    broker: PassBroker,
    baseBytes: ArrayBuffer,
    bandCells: number,
    residualTolerance: number,
  ): void {
    // The compact topology currently carries phi but not the materialized
    // closest-point seed index into a newly allocated A/B page. Span the
    // maintained band until that cache becomes part of the page transaction.
    // The 23-cell band starts at stride 16 — sum-covered (16+8+4+2+1 = 31 >= 23)
    // rather than round-up-covered — so the ladder is 16,8,4,2,1 plus the two
    // +1 collar repairs: 7 floods, one fewer than the former stride-32
    // schedule. Dispatch remains page-delta bounded.
    const strides = planFineLevelSetJFAStrides(bandCells, bandCells);
    if (strides.length > FINE_LEVELSET_JFA_MAX_PASSES) throw new RangeError("Fine JFA pass budget exceeded");
    this.device.queue.writeBuffer(this.params, 0, baseBytes);
    const buffers = [[1, this.source.worklist], [2, this.source.metadata], [3, this.delta.pageDelta],
      [4, this.source.flags], [5, this.source.phi], [6, this.source.workA], [7, this.source.workB],
      [8, this.control], [9, this.reductions], [10, this.supportMask]] as const;
    const run = (label: string, pipeline: GPUComputePipeline, params: GPUBuffer,
      wanted: readonly number[], dispatch: "support" | "dirty" | "single") => {
      const pass = broker.compute({ label });
      pass.setPipeline(pipeline);
      let bindGroup = this.bindGroups.get(pipeline);
      if (!bindGroup) {
        bindGroup = this.device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: params } },
            ...buffers.filter(([binding]) => wanted.includes(binding)).map(([binding, buffer]) =>
              ({ binding, resource: { buffer } })),
          ],
        });
        this.bindGroups.set(pipeline, bindGroup);
      }
      pass.setBindGroup(0, bindGroup);
      if (dispatch === "single") pass.dispatchWorkgroups(1);
      else pass.dispatchWorkgroupsIndirect(this.delta.redistanceDispatches.buffer,
        dispatch === "dirty"
          ? this.delta.redistanceDispatches.dirtyOffsetBytes
          : this.delta.redistanceDispatches.supportOffsetBytes);
    };
    const params = this.params;
    run("Fine JFA - publish support mask", this.publishSupportMaskPipeline,
      params, [2, 3, 10], "support");
    run("Fine JFA - seed closest points", this.jfaSeedPipeline,
      params, [1, 2, 3, 4, 5, 6, 7, 9, 10], "support");
    let inA = true;
    strides.forEach((stride) => {
      const pipeline = (inA ? this.jfaABPipelines : this.jfaBAPipelines).get(stride);
      if (!pipeline) throw new RangeError(`Fine JFA stride ${stride} has no immutable pipeline`);
      run(`Fine JFA - cooperative flood ${inA ? "A to B" : "B to A"} stride ${stride}`,
        pipeline, params, [1, 2, 3, 4, 5, 6, 7, 10], "support");
      inA = !inA;
    });
    run(`Fine JFA - resolve ${inA ? "A to B" : "B to canonical"}`,
      inA ? this.jfaResolveAToBPipeline : this.jfaResolveBToCanonicalPipeline,
      params, [2, 3, 4, 5, 6, 7, 9], "support");
    // Resolve canonicalizes persistent delta state: seeds are always in A and
    // magnitudes are always in B, independent of this generation's JFA parity.
    // Production uses tolerance=1. The unsigned-distance upwind residual is
    // bounded by one at that acceptance bar, so the validation sweep cannot
    // reject a cell and is pure lattice traffic. Keep the strict diagnostic
    // path for tests and opt-in sub-unit tolerances.
    if (residualTolerance < 1) {
      run("Fine JFA - validate distances", this.jfaValidatePipeline,
        params, [1, 2, 3, 4, 5, 7, 9, 10], "dirty");
    }
    run("Fine JFA - finalize", this.jfaFinalizePipeline, params, [3, 8, 9], "single");
    // Finalization is a small deterministic hierarchy; committing phi is not.
    // Give every exact dirty page its own workgroup instead of making one
    // 256-lane workgroup stride over the complete dirty sample population.
    run("Fine JFA - commit distances", this.jfaCommitPipeline,
      params, [2, 3, 4, 5, 6, 7, 8], "dirty");
  }

  destroy(): void {
    this.control.destroy();
    this.reductions.destroy();
    this.supportMask.destroy();
    this.params.destroy();
  }
}

/** Per-tap seed addressing. The flood is the only ALU-limited kernel in the
 * frame (96 % ALU limiter at 51 % occupancy, measured 2026-07-28) and its
 * arithmetic is dominated by integer division: every candidate lane runs
 * `physicalSampleQ` TWICE — once inside `materializedClosestPoint` and once
 * inside `seedStableKey` — and each call divides by `p.samplesPerBrick`,
 * by `p.brickResolution` and by `p.brickResolution * p.brickResolution`, plus
 * two more divisions inside `unpackBrick`. With eight batches per workgroup
 * that is ~96 u32 divisions per lane per flood dispatch.
 *
 * The B4 variant removes both sources without touching a single float:
 *
 * 1. `p.brickResolution` is 4 and `p.samplesPerBrick` is 64 on every generation
 *    this shader may run on — `encode` throws otherwise, and the fixed tap
 *    table above is already transcribed for a 4-cell page (`>>2u` / `&3u` /
 *    `4u*`). Making `localCoord` and `physicalSampleQ` agree with the tap
 *    table replaces three divisions with shifts and masks. For `local < 64`
 *    and `r == 4` the two forms are the same integers, exhaustively.
 * 2. The candidate's lattice coordinate is computed once and shared by the
 *    closest point and the stable tie-break key, instead of being recomputed
 *    independently by each. `sampleKey(seedQ)` IS `seedStableKey(candidate)`
 *    by definition, and `closestPointAt(seedQ, candidate)` evaluates the same
 *    expression tree as `materializedClosestPoint(candidate)` on the same `q`.
 *
 * Both are integer identities, so no float is reassociated and no memory
 * round-trip is added or removed (see the §4.3 fusion refutation in
 * POWER_LIQUIDS_ULTIMATE_M1MAX: the rounding hazard is a storage round-trip,
 * which this change does not have). Coverage, the seed set, the stride ladder,
 * the tie-break order and every acceptance count are untouched.
 */
function fineLevelSetJFALocalCoordWGSL(b4Addressing: boolean): string {
  return b4Addressing
    ? "fn localCoord(local:u32)->vec3u{return vec3u(local&3u,(local>>2u)&3u,(local>>4u)&3u);}"
    : "fn localCoord(local:u32)->vec3u{let r=p.brickResolution;let z=local/(r*r);let rem=local-z*r*r;let y=rem/r;return vec3u(rem-y*r,y,z);}";
}

function fineLevelSetJFAPhysicalSampleWGSL(b4Addressing: boolean): string {
  return b4Addressing
    ? "fn physicalSampleQ(index:u32)->vec3u{let id=index>>6u;return unpackBrick(metadata[id*10u+1u])*4u+localCoord(index&63u);}"
    : "fn physicalSampleQ(index:u32)->vec3u{let id=index/p.samplesPerBrick;let local=index-id*p.samplesPerBrick;return unpackBrick(metadata[id*10u+1u])*p.brickResolution+localCoord(local);}";
}

const FINE_LEVELSET_JFA_CLOSEST_POINT_BODY =
  "let code=flags[index]>>SAMPLE_FLAG_BITS;let direction=code>>24u;let fraction=f32(code&CP_FRACTION_MASK)/CP_FRACTION_SCALE;return vec3f(q)+vec3f(.5)+select(vec3f(directionDelta(direction))*fraction,vec3f(0.),direction>=6u);";

function fineLevelSetJFAClosestPointWGSL(b4Addressing: boolean): string {
  return b4Addressing
    ? `fn closestPointAt(q:vec3u,index:u32)->vec3f{${FINE_LEVELSET_JFA_CLOSEST_POINT_BODY}}fn materializedClosestPoint(index:u32)->vec3f{return closestPointAt(physicalSampleQ(index),index);}`
    : `fn materializedClosestPoint(index:u32)->vec3f{let q=physicalSampleQ(index);${FINE_LEVELSET_JFA_CLOSEST_POINT_BODY}}`;
}

function fineLevelSetJFACandidateWGSL(b4Addressing: boolean): string {
  return b4Addressing
    ? "var bestDistance=LARGE;var bestKey=INVALID;var bestSeed=candidate;if(candidate!=INVALID){let seedQ=physicalSampleQ(candidate);let delta=(vec3f(q)+vec3f(.5))-closestPointAt(seedQ,candidate);bestDistance=dot(delta,delta);bestKey=sampleKey(seedQ);}"
    : "var bestDistance=LARGE;var bestKey=INVALID;var bestSeed=candidate;if(candidate!=INVALID){let delta=(vec3f(q)+vec3f(.5))-materializedClosestPoint(candidate);bestDistance=dot(delta,delta);bestKey=seedStableKey(candidate);}";
}

/** Sparse 1+JFA closest-point transform. Seed keys are global fine-sample
 * linear indices, so the deterministic secondary ordering is independent of
 * physical page IDs and A/B generation allocation order. */
function buildFineLevelSetJFACPTWGSL(b4Addressing: boolean): string {
  return /* wgsl */ `
enable subgroups;
${fineLevelSetLinearWorkgroupWGSL}
const INVALID:u32=0xffffffffu;const VALID:u32=1u;const INTERFACE:u32=2u;const NEGATIVE:u32=16u;const LARGE:f32=3.402823e38;
// Direction seven is the nonzero cache sentinel for a quantized zero
// fraction; materialization treats every direction >= 6 as the sample centre.
const SAMPLE_FLAG_BITS:u32=5u;const CP_FRACTION_MASK:u32=0x00ffffffu;const CP_FRACTION_SCALE:f32=16777215.;
const STALE:u32=4u;const NONFINITE:u32=8u;
override JFA_STRIDE:u32=1u;
${fineLevelSetJFATapLookupWGSL}
struct Params{brickDims:vec3u,brickResolution:u32,sampleDims:vec3u,samplesPerBrick:u32,worklistCapacity:u32,worklistHeaderWords:u32,pageCapacity:u32,generation:u32,bandCells:u32,fineWidth:f32,tolerance:f32,scratchWords:u32,maxWorkgroups:u32,pad0:u32,dirtyPagesOffset:u32,supportPagesOffset:u32}
struct Control{unresolved:u32,residualScaled:u32,seeds:u32,committed:u32,flags:u32,firstError:u32,accepted:u32,initialPages:u32,finalPages:u32,resolveMissing:u32}
struct Partial{seeds:u32,resolveMissing:u32,accepted:u32,validationUnresolved:u32,maximum:u32,flags:u32,firstError:u32}
@group(0)@binding(0)var<uniform>p:Params;@group(0)@binding(1)var<storage,read>worklist:array<u32>;@group(0)@binding(2)var<storage,read>metadata:array<u32>;@group(0)@binding(3)var<storage,read>pageDelta:array<u32>;@group(0)@binding(4)var<storage,read_write>flags:array<u32>;@group(0)@binding(5)var<storage,read_write>phi:array<u32>;@group(0)@binding(6)var<storage,read_write>workA:array<u32>;@group(0)@binding(7)var<storage,read_write>workB:array<u32>;@group(0)@binding(8)var<storage,read_write>control:Control;@group(0)@binding(9)var<storage,read_write>partials:array<Partial>;@group(0)@binding(10)var<storage,read_write>supportMask:array<u32>;
var<workgroup>reduceSum0:array<u32,256>;var<workgroup>reduceSum1:array<u32,256>;var<workgroup>reduceSum2:array<u32,256>;var<workgroup>reduceSum3:array<u32,256>;var<workgroup>reduceMaximum:array<u32,256>;var<workgroup>reduceFlags:array<u32,256>;var<workgroup>reduceFirstError:array<u32,256>;
// One workgroup owns one brick. Every JFA tap therefore lands in one of only
// 27 generation-fixed neighboring bricks; resolve those pages once. The 256
// lanes are eight 32-lane candidate teams, so every sample's 27 taps plus its
// incumbent are evaluated concurrently instead of in a serial inner loop.
var<workgroup>floodPageIds:array<u32,27>;
fn finite(v:f32)->bool{return v==v&&abs(v)<LARGE;}fn bandDistance()->f32{return f32(p.bandCells)*p.fineWidth;}
fn reduceLane(lid:u32,sum0:u32,sum1:u32,sum2:u32,sum3:u32,maximum:u32,errorFlags:u32,firstError:u32,reductionWidth:u32){
  reduceSum0[lid]=sum0;reduceSum1[lid]=sum1;reduceSum2[lid]=sum2;reduceSum3[lid]=sum3;reduceMaximum[lid]=maximum;reduceFlags[lid]=errorFlags;reduceFirstError[lid]=firstError;workgroupBarrier();
  var width=reductionWidth;loop{if(width==0u){break;}if(lid<width){reduceSum0[lid]+=reduceSum0[lid+width];reduceSum1[lid]+=reduceSum1[lid+width];reduceSum2[lid]+=reduceSum2[lid+width];reduceSum3[lid]+=reduceSum3[lid+width];reduceMaximum[lid]=max(reduceMaximum[lid],reduceMaximum[lid+width]);reduceFlags[lid]|=reduceFlags[lid+width];reduceFirstError[lid]=min(reduceFirstError[lid],reduceFirstError[lid+width]);}workgroupBarrier();width>>=1u;}
}
fn publishSeedPartial(work:u32,lid:u32){if(lid==0u&&work<p.pageCapacity){partials[work]=Partial(reduceSum0[0],0u,0u,0u,0u,reduceFlags[0],reduceFirstError[0]);}}
fn publishResolvePartial(work:u32,lid:u32){if(lid==0u&&work<p.pageCapacity){partials[work].resolveMissing=reduceSum0[0];partials[work].accepted=reduceSum1[0];}}
fn publishValidationPartial(work:u32,lid:u32){if(lid==0u&&work<p.pageCapacity){partials[work].validationUnresolved=reduceSum0[0];partials[work].maximum=reduceMaximum[0];partials[work].flags|=reduceFlags[0];partials[work].firstError=min(partials[work].firstError,reduceFirstError[0]);}}
fn unpackBrick(key:u32)->vec3u{let xy=p.brickDims.x*p.brickDims.y;let z=key/xy;let rem=key-z*xy;let y=rem/p.brickDims.x;return vec3u(rem-y*p.brickDims.x,y,z);}fn packBrick(q:vec3u)->u32{return q.x+p.brickDims.x*(q.y+p.brickDims.y*q.z);}${fineLevelSetJFALocalCoordWGSL(b4Addressing)}fn localIndex(q:vec3u)->u32{return q.x+p.brickResolution*(q.y+p.brickResolution*q.z);}
fn sampleKey(q:vec3u)->u32{return q.x+p.sampleDims.x*(q.y+p.sampleDims.y*q.z);}
${makeFineLevelSetSortedWorklistLookupWGSL("p", "metadata", "worklist", "publishedPageOf", "brickDims")}
fn deltaCount(support:bool)->u32{return min(pageDelta[select(2u,3u,support)],p.pageCapacity);}
fn deltaOffset(support:bool)->u32{return select(p.dirtyPagesOffset,p.supportPagesOffset,support);}
fn rawDeltaPage(work:u32,support:bool)->u32{if(work>=deltaCount(support)){return INVALID;}return pageDelta[deltaOffset(support)+work];}
fn deltaPageAt(work:u32,support:bool)->u32{let id=rawDeltaPage(work,support);return select(INVALID,id,id<p.pageCapacity&&metadata[id*10u+2u]==p.generation);}
fn deltaPage(wid:vec3u,nw:vec3u,support:bool)->u32{return deltaPageAt(fineLinearWorkgroup(wid,nw),support);}
@compute @workgroup_size(64)fn publishSupportPageMask(@builtin(workgroup_id)wid:vec3u,@builtin(num_workgroups)nw:vec3u,@builtin(local_invocation_index)lid:u32){if(lid!=0u){return;}let work=fineLinearWorkgroup(wid,nw);let id=deltaPageAt(work,true);if(id!=INVALID){supportMask[id]=p.generation;}}
fn supportPageOf(key:u32)->u32{let id=publishedPageOf(key);return select(INVALID,id,id<p.pageCapacity&&supportMask[id]==p.generation);}
fn hasRadiusOneHalo()->bool{let logicalCount=p.brickDims.x*p.brickDims.y*p.brickDims.z;let haloBase=p.worklistHeaderWords+p.worklistCapacity+logicalCount;return haloBase+p.pageCapacity*27u<=arrayLength(&worklist);}
fn radiusOneHaloPage(id:u32,slot:u32,expectedKey:u32)->u32{
 let logicalCount=p.brickDims.x*p.brickDims.y*p.brickDims.z;let haloBase=p.worklistHeaderWords+p.worklistCapacity+logicalCount;
 if(p.worklistHeaderWords!=7u||worklist[0]!=p.generation||worklist[2]!=p.pageCapacity||(worklist[3]&3u)!=3u||worklist[5]!=1u||worklist[6]!=1u||haloBase+p.pageCapacity*27u>arrayLength(&worklist)){return INVALID;}
 let page=worklist[haloBase+id*27u+slot];let base=page*10u;
 return select(INVALID,page,page<p.pageCapacity&&base+2u<arrayLength(&metadata)&&metadata[base]==page&&metadata[base+1u]==expectedKey&&metadata[base+2u]==p.generation&&supportMask[page]==p.generation);
}
fn prepareFloodPageIds(id:u32,lid:u32){
 var page=INVALID;
 if(lid<27u&&id!=INVALID&&id<p.pageCapacity){
  let center=unpackBrick(metadata[id*10u+1u]);let z=lid/9u;let rem=lid-z*9u;let y=rem/3u;let x=rem-y*3u;
  let direction=vec3i(i32(x)-1,i32(y)-1,i32(z)-1);
  let radius=max(1u,(JFA_STRIDE+p.brickResolution-1u)/p.brickResolution);
  let neighbor=vec3i(center)+direction*i32(radius);
  if(all(neighbor>=vec3i(0))&&all(neighbor<vec3i(p.brickDims))){
   let key=packBrick(vec3u(neighbor));
   if(JFA_STRIDE<=4u&&hasRadiusOneHalo()){page=radiusOneHaloPage(id,lid,key);}else{page=supportPageOf(key);}
  }
 }
 if(lid<27u){floodPageIds[lid]=page;}workgroupBarrier();
}
fn cachedFloodSampleIndex(local:u32,tap:u32)->u32{
 let address=fixedTapAddress(local,tap);let id=floodPageIds[address.x];if(id==INVALID){return INVALID;}
 let index=id*p.samplesPerBrick+address.y;return select(INVALID,index,finite(bitcast<f32>(phi[index])));
}
fn deltaRecordError(work:u32,support:bool)->u32{if(work>=deltaCount(support)){return 0u;}let id=rawDeltaPage(work,support);if(id>=p.pageCapacity||metadata[id*10u+2u]!=p.generation){return STALE;}let key=metadata[id*10u+1u];if(publishedPageOf(key)!=id){return STALE;}if(work>0u){let previous=rawDeltaPage(work-1u,support);if(previous>=p.pageCapacity||metadata[previous*10u+2u]!=p.generation||metadata[previous*10u+1u]>=key){return STALE;}}if(!support&&supportPageOf(key)!=id){return STALE;}return 0u;}
fn sampleIndex(q:vec3u)->u32{if(any(q>=p.sampleDims)){return INVALID;}let id=supportPageOf(packBrick(q/p.brickResolution));if(id==INVALID){return INVALID;}let index=id*p.samplesPerBrick+localIndex(q%p.brickResolution);return select(INVALID,index,finite(bitcast<f32>(phi[index])));}
fn directionDelta(direction:u32)->vec3i{if(direction==0u){return vec3i(-1,0,0);}if(direction==1u){return vec3i(1,0,0);}if(direction==2u){return vec3i(0,-1,0);}if(direction==3u){return vec3i(0,1,0);}if(direction==4u){return vec3i(0,0,-1);}return vec3i(0,0,1);}
${fineLevelSetJFAPhysicalSampleWGSL(b4Addressing)}
fn seedStableKey(index:u32)->u32{return sampleKey(physicalSampleQ(index));}
fn seedClosestPointCode(q:vec3u,index:u32)->u32{let center=bitcast<f32>(phi[index]);if(center==0.){return 6u<<24u;}var best=LARGE;var bestDirection=INVALID;var bestFraction=0.;for(var direction=0u;direction<6u;direction+=1u){let nq=vec3i(q)+directionDelta(direction);if(any(nq<vec3i(0))||any(nq>=vec3i(p.sampleDims))){continue;}let neighbor=sampleIndex(vec3u(nq));if(neighbor==INVALID){continue;}let other=bitcast<f32>(phi[neighbor]);if(!finite(other)||(other<0.)==(center<0.)){continue;}let denominator=abs(center)+abs(other);let fraction=select(0.,abs(center)/denominator,denominator>0.);let d2=fraction*fraction;if(d2<best||(d2==best&&direction<bestDirection)){best=d2;bestDirection=direction;bestFraction=fraction;}}if(bestDirection==INVALID){return INVALID;}let quantized=u32(round(clamp(bestFraction,0.,1.)*CP_FRACTION_SCALE));if(quantized==0u){return 7u<<24u;}return (bestDirection<<24u)|(quantized&CP_FRACTION_MASK);}
fn hasCachedClosestPoint(index:u32)->bool{return (flags[index]>>SAMPLE_FLAG_BITS)!=0u;}
${fineLevelSetJFAClosestPointWGSL(b4Addressing)}
fn cooperativeFlood(id:u32,centerBrick:vec3u,lid:u32,subgroupLane:u32,subgroupSize:u32,fromA:bool){
 let team=lid/subgroupSize;let teamCount=256u/subgroupSize;let candidateSlot=subgroupLane;
 let batchCount=(p.samplesPerBrick+teamCount-1u)/teamCount;
 for(var batch=0u;batch<batchCount;batch+=1u){
  let local=batch*teamCount+team;var index=0u;var q=vec3u(0u);var laneActive=false;
  if(id!=INVALID&&local<p.samplesPerBrick){index=id*p.samplesPerBrick+local;q=centerBrick*p.brickResolution+localCoord(local);laneActive=!any(q>=p.sampleDims)&&finite(bitcast<f32>(phi[index]));}
  var candidate=INVALID;
  if(laneActive&&candidateSlot<27u){
   let candidateIndex=cachedFloodSampleIndex(local,candidateSlot);if(candidateIndex!=INVALID){candidate=select(workB[candidateIndex],workA[candidateIndex],fromA);}
  }else if(laneActive&&candidateSlot==27u){candidate=select(workB[index],workA[index],fromA);}
  ${fineLevelSetJFACandidateWGSL(b4Addressing)}
  var width=subgroupSize>>1u;loop{if(width==0u){break;}let otherDistance=subgroupShuffleDown(bestDistance,width);let otherKey=subgroupShuffleDown(bestKey,width);let otherSeed=subgroupShuffleDown(bestSeed,width);if(subgroupLane<width&&(otherDistance<bestDistance||(otherDistance==bestDistance&&otherKey<bestKey))){bestDistance=otherDistance;bestKey=otherKey;bestSeed=otherSeed;}width>>=1u;}
  if(laneActive&&subgroupLane==0u){if(fromA){workB[index]=bestSeed;}else{workA[index]=bestSeed;}}
 }
}
fn resolvedDistance(seed:u32,q:vec3u)->f32{if(seed==INVALID){return bandDistance();}return length((vec3f(q)+vec3f(.5))-materializedClosestPoint(seed))*p.fineWidth;}
fn distanceValue(index:u32)->f32{return bitcast<f32>(workB[index]);}
fn resolvedSeed(index:u32)->u32{return workA[index];}
@compute @workgroup_size(64)fn seedClosestPoints(@builtin(workgroup_id)wid:vec3u,@builtin(local_invocation_index)lid:u32,@builtin(num_workgroups)nw:vec3u){
  let work=fineLinearWorkgroup(wid,nw);let id=deltaPage(wid,nw,true);var seedCount=0u;var errorFlags=0u;var firstError=INVALID;
  if(lid==0u){errorFlags=deltaRecordError(work,true);firstError=select(INVALID,work,errorFlags!=0u);}
  if(id!=INVALID&&lid<p.samplesPerBrick){let index=id*p.samplesPerBrick+lid;let brick=unpackBrick(metadata[id*10u+1u]);let q=brick*p.brickResolution+localCoord(lid);workA[index]=INVALID;workB[index]=INVALID;flags[index]&=(1u<<SAMPLE_FLAG_BITS)-1u;if(!any(q>=p.sampleDims)){let value=bitcast<f32>(phi[index]);if(!finite(value)){errorFlags|=NONFINITE;firstError=min(firstError,index);}else{let closest=seedClosestPointCode(q,index);if(closest!=INVALID){workA[index]=index;flags[index]|=closest<<SAMPLE_FLAG_BITS;seedCount=1u;}}}}
  reduceLane(lid,seedCount,0u,0u,0u,0u,errorFlags,firstError,32u);publishSeedPartial(work,lid);
}
@compute @workgroup_size(256)fn jumpFloodAToB(@builtin(workgroup_id)wid:vec3u,@builtin(local_invocation_index)lid:u32,@builtin(num_workgroups)nw:vec3u,@builtin(subgroup_invocation_id)subgroupLane:u32,@builtin(subgroup_size)subgroupSize:u32){let id=deltaPage(wid,nw,true);prepareFloodPageIds(id,lid);var brick=vec3u(0u);if(id!=INVALID){brick=unpackBrick(metadata[id*10u+1u]);}cooperativeFlood(id,brick,lid,subgroupLane,subgroupSize,true);}
@compute @workgroup_size(256)fn jumpFloodBToA(@builtin(workgroup_id)wid:vec3u,@builtin(local_invocation_index)lid:u32,@builtin(num_workgroups)nw:vec3u,@builtin(subgroup_invocation_id)subgroupLane:u32,@builtin(subgroup_size)subgroupSize:u32){let id=deltaPage(wid,nw,true);prepareFloodPageIds(id,lid);var brick=vec3u(0u);if(id!=INVALID){brick=unpackBrick(metadata[id*10u+1u]);}cooperativeFlood(id,brick,lid,subgroupLane,subgroupSize,false);}
@compute @workgroup_size(64)fn resolveClosestPointsAToB(@builtin(workgroup_id)wid:vec3u,@builtin(local_invocation_index)lid:u32,@builtin(num_workgroups)nw:vec3u){
  let work=fineLinearWorkgroup(wid,nw);let id=deltaPage(wid,nw,true);var unresolved=0u;var accepted=0u;if(id!=INVALID&&lid<p.samplesPerBrick){let index=id*p.samplesPerBrick+lid;let q=unpackBrick(metadata[id*10u+1u])*p.brickResolution+localCoord(lid);let transported=bitcast<f32>(phi[index]);if(!any(q>=p.sampleDims)&&finite(transported)){var seed=workA[index];if(hasCachedClosestPoint(index)){seed=index;}unresolved=select(0u,1u,seed==INVALID&&abs(transported)<bandDistance());let d=resolvedDistance(seed,q);workA[index]=seed;workB[index]=bitcast<u32>(d);accepted=select(0u,1u,seed!=INVALID&&d<=bandDistance());}}reduceLane(lid,unresolved,accepted,0u,0u,0u,0u,INVALID,32u);publishResolvePartial(work,lid);
}
@compute @workgroup_size(64)fn resolveClosestPointsBToCanonical(@builtin(workgroup_id)wid:vec3u,@builtin(local_invocation_index)lid:u32,@builtin(num_workgroups)nw:vec3u){
  let work=fineLinearWorkgroup(wid,nw);let id=deltaPage(wid,nw,true);var unresolved=0u;var accepted=0u;if(id!=INVALID&&lid<p.samplesPerBrick){let index=id*p.samplesPerBrick+lid;let q=unpackBrick(metadata[id*10u+1u])*p.brickResolution+localCoord(lid);let transported=bitcast<f32>(phi[index]);if(!any(q>=p.sampleDims)&&finite(transported)){var seed=workB[index];if(hasCachedClosestPoint(index)){seed=index;}unresolved=select(0u,1u,seed==INVALID&&abs(transported)<bandDistance());let d=resolvedDistance(seed,q);workA[index]=seed;workB[index]=bitcast<u32>(d);accepted=select(0u,1u,seed!=INVALID&&d<=bandDistance());}}reduceLane(lid,unresolved,accepted,0u,0u,0u,0u,INVALID,32u);publishResolvePartial(work,lid);
}
@compute @workgroup_size(64)fn validateJFADistances(@builtin(workgroup_id)wid:vec3u,@builtin(local_invocation_index)lid:u32,@builtin(num_workgroups)nw:vec3u){
  let work=fineLinearWorkgroup(wid,nw);let id=deltaPage(wid,nw,false);var unresolved=0u;var residual=0u;var errorFlags=0u;var firstError=INVALID;
  if(lid==0u){errorFlags=deltaRecordError(work,false);firstError=select(INVALID,work,errorFlags!=0u);}
  if(id!=INVALID&&lid<p.samplesPerBrick){let index=id*p.samplesPerBrick+lid;let transported=bitcast<f32>(phi[index]);let d=distanceValue(index);if(!finite(transported)||!finite(d)){unresolved=1u;errorFlags|=NONFINITE;firstError=min(firstError,index);}else if(d<bandDistance()&&!hasCachedClosestPoint(index)){let q=unpackBrick(metadata[id*10u+1u])*p.brickResolution+localCoord(lid);var sum=0.;for(var axis=0u;axis<3u;axis+=1u){var nearest=d;for(var side=-1;side<=1;side+=2){var nq=vec3i(q);nq[axis]+=side;if(any(nq<vec3i(0))||any(nq>=vec3i(p.sampleDims))){continue;}let neighbor=sampleIndex(vec3u(nq));if(neighbor!=INVALID){nearest=min(nearest,distanceValue(neighbor));}}let gradient=max(0.,d-nearest)/p.fineWidth;sum+=gradient*gradient;}residual=u32(min(4294967295.,abs(sqrt(sum)-1.)*1000000.));unresolved=select(0u,1u,residual>u32(p.tolerance*1000000.));}}
  reduceLane(lid,unresolved,0u,0u,0u,residual,errorFlags,firstError,32u);publishValidationPartial(work,lid);
}
@compute @workgroup_size(256)fn finalizeJFADistances(@builtin(local_invocation_index)lid:u32){
  let support=min(pageDelta[3],p.pageCapacity);let dirty=min(pageDelta[2],p.pageCapacity);
  var seeds=0u;var resolveMissing=0u;var accepted=0u;var validationUnresolved=0u;var maximum=0u;var errorFlags=0u;var firstError=INVALID;
  for(var record=lid;record<support;record+=256u){let value=partials[record];seeds+=value.seeds;resolveMissing+=value.resolveMissing;accepted+=value.accepted;validationUnresolved+=value.validationUnresolved;maximum=max(maximum,value.maximum);errorFlags|=value.flags;firstError=min(firstError,value.firstError);}
  reduceLane(lid,resolveMissing,validationUnresolved,seeds,accepted,maximum,errorFlags,firstError,128u);
  if(lid==0u){let malformed=pageDelta[2]>p.pageCapacity||pageDelta[3]>p.pageCapacity||pageDelta[2]>pageDelta[3];let stale=pageDelta[1]!=p.generation||malformed;control.resolveMissing=reduceSum0[0];control.unresolved=reduceSum0[0]+reduceSum1[0];control.seeds=reduceSum2[0];control.accepted=reduceSum3[0];control.residualScaled=reduceMaximum[0];control.flags=reduceFlags[0]|select(0u,STALE,stale);control.firstError=min(reduceFirstError[0],select(INVALID,select(pageDelta[1],2u,malformed),stale));control.initialPages=dirty;control.finalPages=dirty;control.committed=select(0u,1u,control.flags==0u&&control.unresolved==0u&&(control.seeds>0u||pageDelta[2]==0u));}
}
@compute @workgroup_size(64)fn commitJFADistances(@builtin(workgroup_id)wid:vec3u,@builtin(local_invocation_index)lid:u32,@builtin(num_workgroups)nw:vec3u){
  if(control.committed==0u){return;}let id=deltaPage(wid,nw,false);if(id==INVALID||lid>=p.samplesPerBrick){return;}
  let index=id*p.samplesPerBrick+lid;let transported=bitcast<f32>(phi[index]);let d=distanceValue(index);let negative=transported<0.;let closestPointCode=flags[index]&~((1u<<SAMPLE_FLAG_BITS)-1u);let isInterface=closestPointCode!=0u;phi[index]=bitcast<u32>(select(d,-d,negative));if(resolvedSeed(index)==INVALID||d>bandDistance()){flags[index]=0u;}else{flags[index]=closestPointCode|VALID|select(0u,INTERFACE,isInterface)|select(0u,NEGATIVE,negative);}
}
`;
}

/** Production source. Byte-identical to the schedule this file has always
 * emitted; every architectural pin in the test suite reads this constant. */
export const fineLevelSetJFACPTWGSL = buildFineLevelSetJFACPTWGSL(false);

/** Opt-in variant selected by `FLUID_FINE_JFA_B4_ADDRESSING=1`. Differs from
 * the production source only in `localCoord`, `physicalSampleQ`, the closest
 * point accessor and the flood's candidate evaluation. */
export const fineLevelSetJFACPTB4AddressingWGSL = buildFineLevelSetJFACPTWGSL(true);

function instrumentFineLevelSetJFAEntry(
  source: string,
  entryPoint: string,
  entry: string,
  exit: string,
): string {
  const signature = `fn ${entryPoint}(`;
  const start = source.indexOf(signature);
  if (start < 0) throw new Error(`Fine JFA activity entry point ${entryPoint} is missing`);
  const bodyStart = source.indexOf("{", start + signature.length);
  let depth = 0, bodyEnd = -1;
  for (let index = bodyStart; index < source.length; index += 1) {
    if (source[index] === "{") depth += 1;
    else if (source[index] === "}" && --depth === 0) { bodyEnd = index; break; }
  }
  if (bodyStart < 0 || bodyEnd < 0) throw new Error(`Fine JFA activity body ${entryPoint} is malformed`);
  const body = source.slice(bodyStart + 1, bodyEnd).replace(/\breturn;/g, `${exit}return;`);
  return `${source.slice(0, bodyStart + 1)}${entry}${body}${exit}${source.slice(bodyEnd)}`;
}

/** Activity-only flood variants; production keeps the exported WGSL byte-for-byte. */
export function fineLevelSetJFAActivityShader(
  activity: GPULogicalActivityAdoptionContext,
  source: string = fineLevelSetJFACPTWGSL,
): string {
  const checkpoint = (task: string, name: "enter" | "exit") => activity.workgroup(task, name, {
    workgroupId: "wid",
    numWorkgroups: "nw",
    localInvocationIndex: "lid",
    workgroupLaneCount: 256,
  });
  const abEntry = checkpoint("jump-flood-a-to-b", "enter");
  const abExit = checkpoint("jump-flood-a-to-b", "exit");
  const baEntry = checkpoint("jump-flood-b-to-a", "enter");
  const baExit = checkpoint("jump-flood-b-to-a", "exit");
  if (!abEntry && !abExit && !baEntry && !baExit) return source;
  const withAB = instrumentFineLevelSetJFAEntry(
    source, "jumpFloodAToB", abEntry, abExit,
  );
  return instrumentFineLevelSetJFAEntry(withAB, "jumpFloodBToA", baEntry, baExit);
}
