/** Transactional dynamic free-surface/solid coefficients for structured families. */

import type { WebGPUFineLevelSetBrickSource } from "./webgpu-octree-fine-levelset-bricks";
import { fineLevelSetPackedSampleWGSL } from "./fine-levelset-packed-sample";

type StructuredBoundaryFineSource = Pick<WebGPUFineLevelSetBrickSource,
"params" | "metadata" | "worklist" | "samples">;
import {
  makeOctreePowerCoarseLevelSetSampleWGSL,
  type OctreePowerCoarseLevelSetSampleSource,
} from "./webgpu-octree-power-coarse-levelset";
import type { OctreeSolidVertexSdfSource } from "./webgpu-octree-solid-vertex-sdf";
import {
  OCTREE_STRUCTURED_GPU_VALID,
  type DirectStructuredVelocitySource,
} from "./webgpu-octree-structured-velocity-gpu";
import type { PassBroker } from "./webgpu-pass-broker";

type StructuredControlAuthority = "candidate-ready" | "accepted";
interface StructuredBoundaryGroupCache {
  readonly candidateReady?: readonly GPUBindGroup[];
  readonly accepted?: readonly GPUBindGroup[];
}

interface StructuredBoundaryPipelineBundle {
  readonly prepareCandidate: GPUComputePipeline;
  readonly prepareAccepted: GPUComputePipeline;
  readonly classify: GPUComputePipeline;
  readonly resolve: GPUComputePipeline;
  readonly resolveSolid: GPUComputePipeline;
  readonly commit: GPUComputePipeline;
  readonly rebuild: GPUComputePipeline;
  readonly countRowClasses: GPUComputePipeline;
  readonly countFamilyClasses: GPUComputePipeline;
  readonly scanWorksetBlocks: GPUComputePipeline;
  readonly scatterRowWorksets: GPUComputePipeline;
  readonly scatterFamilyWorksets: GPUComputePipeline;
  readonly accept: GPUComputePipeline;
}

const structuredBoundaryPipelineCache = new WeakMap<GPUDevice,
  Promise<StructuredBoundaryPipelineBundle>>();

/** Diagnostic A/B only. Production admits proved dry identity rows by default. */
export function octreePowerHybridDryIdentityEnabled(
  environment?: Readonly<Record<string, string | undefined>>,
): boolean {
  const resolved = environment
    ?? (typeof process !== "undefined" ? process.env : undefined);
  return resolved?.FLUID_POWER_HYBRID_DRY_IDENTITY !== "0";
}

/**
 * Publish zero row and slot dispatches when the whole boundary map provably
 * reproduces the accepted publication bit for bit. Default OFF: this is a Gate
 * A/B arm, not a shipped default.
 *
 * The flag is carried in the uniform (`p.carry.x`), never in the WGSL text, so
 * both arms compile the identical shader with the identical auto bind-group
 * layouts. An A/B therefore differs by one uniform word and nothing else.
 *
 * WHAT THE GATE PROVES (see `boundaryExactRowCarryEligible`): the boundary
 * writes only `a.fractions`, `a.pressureScales`, `solidVelocity`, `liquid`, the
 * 19-channel `rows` coefficients and the nine worksets, all at
 * `control.bank`. The carry retains last epoch's bytes at that same bank, so it
 * is exact exactly when every input to those writes is bitwise unchanged:
 *   - row/slot geometry, handles, areas, inverse distances, normals, centroids:
 *     covered by the structured publication's own exact row-delta identity
 *     carry (`accepted[15]`, i.e. candidate `control.reserved[0]`). That bit is
 *     set only when `beginStructuredPublication` zeroed dispatch records 0 and
 *     3, which is also what guarantees the scatter did NOT reset
 *     fractions/pressureScales to 1.0 underneath the retained values;
 *   - the coarse level set: covered by the coarse publisher's own exact
 *     value/phase delta (`count == 0`), which compares phi, minimumPhi and
 *     maximumPhi as raw bit patterns and reports every insertion and
 *     retirement;
 *   - the fine level set: NOT covered by any receipt, so a live fine band
 *     (`fp.width > 0`) disables the carry outright;
 *   - rigid bodies: no receipt, so a non-zero body count disables the carry;
 *   - the separation mask: `separationFresh` ages a mark against
 *     `control.epoch`, so an unchanged mask still flips an aperture as the
 *     epoch advances. `resolveStructuredBoundarySlots` therefore folds "a
 *     closed world face carried a live mark" into the dirty channel, which
 *     keeps the carry off for as long as any mark exists.
 *
 * TRANSITIVITY, and the one conjunct that lacks it. The coarse delta and the
 * row-delta identity each compare step N against step N-1, so a chain of them
 * proves step N against the last full recompute; the liquid mask is a pure
 * function of those two and inherits the chain. The separation mask does not:
 * a carried step never runs the resolve, so it never observes the mask, and an
 * unbroken chain would stop watching it entirely. Mode 1 therefore refuses to
 * carry twice in a row, which caps mark staleness at exactly one step. Mode 2
 * chains and exists to measure what that alternation costs.
 *
 * RESIDUALS, stated so the A/B knows what it is pricing:
 *   1. The coarse delta's phase comparison covers only PHI_INTERFACE and
 *      PHI_CORRECTED. A PHI_VALID/PHI_FINITE flip with bitwise identical phi
 *      would not be reported. Widening that comparison belongs to the coarse
 *      publisher.
 *   2. Even in mode 1 the separation conjunct is retrospective by one step: it
 *      cannot see the first mark of a run, created by the projection after the
 *      publication whose receipt we are reading. Closing that needs a
 *      mark counter published by `markSeparationRow`.
 *
 * `FLUID_STRUCTURED_BOUNDARY_EXACT_ROW_CARRY`: unset/"0" off, "1" alternating,
 * "2" chained.
 */
export function structuredBoundaryExactRowCarryMode(
  environment?: Readonly<Record<string, string | undefined>>,
): 0 | 1 | 2 {
  const resolved = environment
    ?? (typeof process !== "undefined" ? process.env : undefined);
  const raw = resolved?.FLUID_STRUCTURED_BOUNDARY_EXACT_ROW_CARRY;
  return raw === "1" ? 1 : raw === "2" ? 2 : 0;
}

export function structuredBoundaryExactRowCarryEnabled(
  environment?: Readonly<Record<string, string | undefined>>,
): boolean {
  return structuredBoundaryExactRowCarryMode(environment) !== 0;
}

export interface StructuredBoundaryResources {
  readonly structured: DirectStructuredVelocitySource;
  /** Identity-keyed coarse phi from the accepted pre-adaptation topology.
   * Candidate rows may be inserted or reordered, so row-indexed values are
   * not a valid source for the inactive structured generation. */
  readonly coarse: Pick<OctreePowerCoarseLevelSetSampleSource,
    "directory" | "rowCapacity" | "delta">;
  readonly solid?: OctreeSolidVertexSdfSource;
  /** Per-cell unilateral-contact face bits published by the previous
   * projection's separation stage; fresh marks open closed world faces. */
  readonly separationMask: GPUBuffer;
  readonly rigidBodies: GPUBuffer;
  readonly bodyCount: number;
  readonly dimensions: readonly [number, number, number];
  readonly physicalCellSize: number;
  readonly closedBoundaryMask: number;
  /** Exact authored SDF used for the cold bootstrap and as a narrowly scoped
   * fallback while a recurring coarse-directory generation is unpublished. */
  readonly analyticBootstrap?: Readonly<{
    initialCondition: "dam-break" | "tank-fill";
    fillFraction: number;
    damBreakDimensions?: Readonly<{ x: number; y: number; z: number }>;
  }>;
  /**
   * The imported dense t=0 level set, used by scenes whose surface has no
   * closed form (terrain, rigid bodies, explicitly seeded bricks). It plays
   * exactly the same one-shot role as `analyticBootstrap` and retires through
   * the same hook; without either, coarse phi has no authority at t=0 and the
   * boundary coefficients fail closed on positivity.
   */
  readonly bootstrapLevelSet?: GPUTexture;
}

export class WebGPUStructuredBoundaryCoefficients {
  /** Accepted row liquid mask; one u32 per row. */
  readonly liquidMask: GPUBuffer;
  /** A/B family-handle normal velocity of the closest intersecting rigid. */
  readonly solidNormalVelocities: GPUBuffer;
  readonly control: GPUBuffer;
  readonly candidateControl: GPUBuffer;
  /** Packed A/B current-boundary row/family worksets, matching the structured bank. */
  readonly worksets: GPUBuffer;
  readonly worksetStrideWords: number;
  readonly worksetBankStrideWords: number;
  readonly encodedDispatchCount = 11;
  readonly allocatedBytes: number;
  private readonly worksetClasses: GPUBuffer;
  private readonly worksetBlocks: GPUBuffer;
  /** Inactive logical-row mask exposed only for the X-2 diagnostic census. */
  readonly candidateMask: GPUBuffer;
  /** QA-visible resolved aperture/ghost scale pairs, one vec2f per family. */
  readonly candidates: GPUBuffer;
  private readonly candidateSolidNormalVelocities: GPUBuffer;
  private readonly dispatch: GPUBuffer;
  private readonly params: GPUBuffer;
  private readonly inertSolidStorage?: GPUBuffer;
  /** Bind-layout sentinels for coarse-only tracking. `FineP.width == 0`
   * makes every shader query return compact coarse phi before touching the
   * remaining bindings; these buffers are not a resident surface band. */
  private readonly inertFineParams: GPUBuffer;
  private readonly inertFineStorage: GPUBuffer;
  private prepareCandidate!: GPUComputePipeline;
  private prepareAccepted!: GPUComputePipeline;
  private classify!: GPUComputePipeline;
  private resolve!: GPUComputePipeline;
  private resolveSolid!: GPUComputePipeline;
  private commit!: GPUComputePipeline;
  private rebuild!: GPUComputePipeline;
  private countRowClasses!: GPUComputePipeline;
  private countFamilyClasses!: GPUComputePipeline;
  private scanWorksetBlocks!: GPUComputePipeline;
  private scatterRowWorksets!: GPUComputePipeline;
  private scatterFamilyWorksets!: GPUComputePipeline;
  private accept!: GPUComputePipeline;
  private acceptGroup!: GPUBindGroup;
  private readonly groupsByFineParams = new WeakMap<GPUBuffer,
    WeakMap<GPUBuffer, StructuredBoundaryGroupCache>>();
  private destroyed = false;

  constructor(private readonly device: GPUDevice, private readonly resources: StructuredBoundaryResources) {
    const { structured } = resources;
    const analyticDam = resources.analyticBootstrap?.damBreakDimensions;
    if (!(resources.physicalCellSize > 0) || !Number.isFinite(resources.physicalCellSize)
      || resources.dimensions.some((value) => !Number.isSafeInteger(value) || value < 1)
      || resources.coarse.rowCapacity !== structured.plan.rowCapacity
      || resources.coarse.directory.size < 32 + structured.plan.rowCapacity * 32
      // The exact-carry gate reads only the delta's 16-word publication header.
      || resources.coarse.delta.size < 64
      || !Number.isSafeInteger(resources.bodyCount) || resources.bodyCount < 0 || resources.bodyCount > 12
      || resources.separationMask.size < resources.dimensions[0]! * resources.dimensions[1]! * resources.dimensions[2]! * 4
      || resources.rigidBodies.size < 12 * 8 * 16
      || resources.analyticBootstrap
        && (!Number.isFinite(resources.analyticBootstrap.fillFraction)
          || resources.analyticBootstrap.fillFraction < 0
          || resources.analyticBootstrap.fillFraction > 1)
      || analyticDam && [analyticDam.x, analyticDam.y, analyticDam.z]
        .some((value, axis) => !Number.isFinite(value) || value < 0
          || value > resources.dimensions[axis]! * resources.physicalCellSize)
      || resources.solid && resources.solid.plan.rowCapacity !== structured.plan.rowCapacity) {
      throw new RangeError("Structured boundary resources are invalid or undersized");
    }
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    this.liquidMask = device.createBuffer({ label: "Accepted structured liquid row mask",
      size: 2 * structured.plan.rowCapacity * 4, usage: storage });
    this.candidateMask = device.createBuffer({ label: "Candidate structured liquid row mask",
      size: structured.plan.rowCapacity * 4, usage: storage });
    this.candidates = device.createBuffer({ label: "Candidate structured aperture and pressure scale",
      size: structured.plan.slotCapacity * 8, usage: storage });
    this.solidNormalVelocities = device.createBuffer({ label: "A/B structured solid normal velocities",
      size: 2 * structured.plan.slotCapacity * 4, usage: storage });
    this.candidateSolidNormalVelocities = device.createBuffer({ label: "Candidate structured solid normal velocities",
      size: structured.plan.slotCapacity * 4, usage: storage });
    this.control = device.createBuffer({ label: "Accepted structured boundary coefficient publication", size: 64, usage: storage });
    this.candidateControl = device.createBuffer({ label: "Candidate structured boundary coefficient publication", size: 64, usage: storage });
    this.dispatch = device.createBuffer({ label: "Structured boundary live dispatch", size: 24,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT });
    this.worksetStrideWords = structured.plan.worksetStrideWords;
    this.worksetBankStrideWords = structured.worksetBankStrideWords;
    this.worksets = device.createBuffer({ label: "Packed A/B dynamic structured boundary worksets",
      size: 2 * this.worksetBankStrideWords * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_SRC });
    // Workset compaction scratch. Both are pure per-advance intermediates:
    // the class of every row then every slot, and 9 classes of per-block
    // prefix bases over whichever of the two ranges that class draws from.
    this.worksetClasses = device.createBuffer({ label: "Structured boundary workset element classes",
      size: (structured.plan.rowCapacity + structured.plan.slotCapacity) * 4, usage: storage });
    this.worksetBlocks = device.createBuffer({ label: "Structured boundary workset block prefixes",
      size: 9 * Math.ceil(Math.max(structured.plan.rowCapacity, structured.plan.slotCapacity) / 64) * 4,
      usage: storage });
    // Nine vec4 of published geometry plus the tenth `carry` vec4, which holds
    // only Gate A/B arm selectors. Keeping the arm in the uniform rather than in
    // the WGSL text is what makes both arms compile byte-identical modules.
    this.params = device.createBuffer({ label: "Structured boundary parameters", size: 160,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.inertFineParams = device.createBuffer({
      // FineP is 72 bytes of fields rounded to its 16-byte uniform alignment.
      label: "Coarse-only structured boundary fine-parameter sentinel", size: 80,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.inertFineStorage = device.createBuffer({
      label: "Coarse-only structured boundary fine-storage sentinel", size: 32,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    if (!resources.solid) {
      this.inertSolidStorage = device.createBuffer({ label: "Inert structured solid SDF binding", size: 80,
        usage: storage });
    }
    const words = new Uint32Array(40), floats = new Float32Array(words.buffer);
    words.set([structured.plan.rowCapacity, structured.plan.slotCapacity,
      structured.plan.maximumCaseSlots, structured.plan.authorityWords,
      19, structured.section63.coefficientBankStrideWords,
      resources.closedBoundaryMask >>> 0, resources.solid ? 1 : 0,
      ...resources.dimensions, resources.bodyCount], 0);
    floats[12] = resources.physicalCellSize;
    const o = structured.plan.offsets;
    words.set([o.ownerRows, o.neighborRows, o.areas, o.inverseDistances,
      o.fractions, o.pressureScales, o.normals, o.centroids,
      o.rowNeighbors, o.rowSlotHandles, o.rowSlotSigns, o.rowCatalogSlots], 16);
    words.set([this.worksetStrideWords, this.worksetBankStrideWords,
      resources.analyticBootstrap?.initialCondition === "tank-fill" ? 1
        : resources.analyticBootstrap?.initialCondition === "dam-break" ? 2
          : resources.bootstrapLevelSet ? 3 : 0], 28);
    floats[31] = resources.analyticBootstrap?.fillFraction ?? 0;
    const dam = resources.analyticBootstrap?.damBreakDimensions;
    floats.set([dam?.x ?? 0, dam?.y ?? 0, dam?.z ?? 0,
      resources.analyticBootstrap?.initialCondition === "tank-fill" ? 1
        : resources.analyticBootstrap?.initialCondition === "dam-break" ? 2 : 0], 32);
    words[36] = structuredBoundaryExactRowCarryMode();
    device.queue.writeBuffer(this.params, 0, words.buffer);
    this.allocatedBytes = this.liquidMask.size + this.solidNormalVelocities.size
      + this.candidateMask.size + this.candidates.size + this.candidateSolidNormalVelocities.size
      + this.control.size + this.candidateControl.size + this.dispatch.size + this.params.size + this.worksets.size
      + this.worksetClasses.size + this.worksetBlocks.size
      + this.inertFineParams.size + this.inertFineStorage.size
      + (this.inertSolidStorage?.size ?? 0);
  }

  async initializePipelines(): Promise<void> {
    if (this.acceptGroup) return;
    let compilation = structuredBoundaryPipelineCache.get(this.device);
    if (!compilation) {
      compilation = (async () => {
        const shaderModule = this.device.createShaderModule({
          label: "Structured boundary coefficients", code: structuredBoundaryCoefficientWGSL,
        });
        const make = (entryPoint: string) => this.device.createComputePipelineAsync({
          label: entryPoint, layout: "auto", compute: { module: shaderModule, entryPoint },
        });
        return {
          prepareCandidate: await make("prepareStructuredBoundaryCandidate"),
          prepareAccepted: await make("prepareStructuredBoundaryAccepted"),
          classify: await make("classifyStructuredLiquidRows"),
          resolve: await make("resolveStructuredBoundarySlots"),
          resolveSolid: await make("resolveStructuredSolidSlots"),
          commit: await make("commitStructuredBoundarySlots"),
          rebuild: await make("rebuildStructuredBoundaryRows"),
          countRowClasses: await make("countStructuredRowClasses"),
          countFamilyClasses: await make("countStructuredFamilyClasses"),
          scanWorksetBlocks: await make("scanStructuredWorksetBlocks"),
          scatterRowWorksets: await make("scatterStructuredRowWorksets"),
          scatterFamilyWorksets: await make("scatterStructuredFamilyWorksets"),
          accept: await make("acceptStructuredBoundary"),
        };
      })();
      structuredBoundaryPipelineCache.set(this.device, compilation);
    }
    const pipelines = await compilation;
    Object.assign(this, pipelines);
    this.acceptGroup = this.device.createBindGroup({
      layout: this.accept.getBindGroupLayout(0), entries: [
        { binding: 16, resource: { buffer: this.candidateControl } },
        { binding: 22, resource: { buffer: this.control } },
      ],
    });
  }

  private groups(fine: StructuredBoundaryFineSource | undefined,
    structuredControl: GPUBuffer, authority: StructuredControlAuthority): readonly GPUBindGroup[] {
    const selectedFine: StructuredBoundaryFineSource = fine ?? {
      params: this.inertFineParams,
      metadata: this.inertFineStorage,
      worklist: this.inertFineStorage,
      samples: this.inertFineStorage,
    };
    let byControl = this.groupsByFineParams.get(selectedFine.params);
    if (!byControl) {
      byControl = new WeakMap();
      this.groupsByFineParams.set(selectedFine.params, byControl);
    }
    const cached = byControl.get(structuredControl);
    const cachedGroups = authority === "candidate-ready" ? cached?.candidateReady : cached?.accepted;
    if (cachedGroups) return cachedGroups;
    const { structured, solid } = this.resources;
    const section63 = structured.section63;
    const fparams = selectedFine.params;
    const inertSolid = this.inertSolidStorage;
    const buffers: Record<number, GPUBuffer> = { 0: this.params, 1: structuredControl,
      2: structured.authority, 3: structured.rowGeometry, 4: this.resources.coarse.directory,
      5: fparams, 6: selectedFine.worklist, 7: selectedFine.metadata,
      8: selectedFine.samples, 10: solid?.arena ?? inertSolid!,
      11: this.candidates, 12: this.candidateMask, 13: this.liquidMask,
      14: section63.coefficients, 16: this.candidateControl, 17: this.dispatch,
      18: this.worksets, 19: this.resources.rigidBodies, 20: this.solidNormalVelocities,
      21: this.candidateSolidNormalVelocities, 22: this.control,
      23: section63.topologyMetrics, 24: section63.catalogFaces,
      25: this.resources.separationMask,
      27: this.worksetClasses, 28: this.worksetBlocks,
      // Exact-carry receipt only. Reachable from the candidate prepare and from
      // nowhere else, which keeps every kernel already at Metal's ten-storage
      // ceiling exactly where it was.
      29: this.resources.coarse.delta };
    // Binding 26 is the only texture in this layout; every other entry is a
    // buffer from the map above.
    // `coarseValue` references the texture unconditionally, so the auto layout
    // always contains binding 26 regardless of which bootstrap mode is live.
    if (!this.resources.bootstrapLevelSet) {
      throw new Error("Structured boundary coefficients require a bootstrap level-set binding");
    }
    const bootstrapView = this.resources.bootstrapLevelSet.createView({ dimension: "3d" });
    const group = (pipeline: GPUComputePipeline, bindings: readonly number[]) => this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0), entries: bindings.map((binding) => ({ binding,
        resource: binding === 26 ? bootstrapView! : { buffer: buffers[binding]! } })),
    });
    const prepare = authority === "candidate-ready" ? this.prepareCandidate : this.prepareAccepted;
    const groups = Object.freeze([
      // The candidate prepare owns the whole exact-carry gate, so it is the one
      // kernel that reads the coarse directory header (4), the fine-parameter
      // sentinel (5) and the coarse value/phase delta (29). Six storage
      // bindings on a workgroup_size(1) kernel; the recurring accepted prepare
      // has no carry arm and keeps its original four.
      group(prepare, authority === "candidate-ready" ? [0, 1, 4, 5, 16, 17, 22, 29] : [0, 1, 16, 17]),
      group(this.classify, [0, 3, 4, 5, 6, 7, 8, 12, 16, 26]),
      group(this.resolve, [0, 2, 3, 4, 5, 6, 7, 8, 11, 16, 25, 26]),
      group(this.resolveSolid, [0, 2, 3, 10, 11, 16, 19, 21]),
      group(this.commit, [0, 2, 11, 16, 20, 21]),
      group(this.rebuild, [0, 1, 2, 12, 13, 14, 16, 22, 24, 27]),
      group(this.countRowClasses, [0, 3, 13, 14, 16, 27, 28]),
      group(this.countFamilyClasses, [0, 2, 3, 13, 14, 16, 27, 28]),
      // Binding 22 is the carry arm's revalidation source: the retained workset
      // headers must still be stamped with the ACCEPTED epoch before this scan
      // re-stamps them with the candidate one.
      group(this.scanWorksetBlocks, [0, 16, 18, 22, 28]),
      group(this.scatterRowWorksets, [0, 16, 18, 27, 28]),
      group(this.scatterFamilyWorksets, [0, 16, 18, 27, 28]),
    ]);
    byControl.set(structuredControl, authority === "candidate-ready"
      ? { ...cached, candidateReady: groups }
      : { ...cached, accepted: groups });
    return groups;
  }

  private encodeFromStructuredControl(broker: PassBroker, fine: StructuredBoundaryFineSource | undefined,
    structuredControl: GPUBuffer, authority: StructuredControlAuthority): void {
    if (this.destroyed) throw new Error("Structured boundary coefficients are destroyed");
    const groups = this.groups(fine, structuredControl, authority);
    const prepare = authority === "candidate-ready" ? this.prepareCandidate : this.prepareAccepted;
    const stages = [prepare, this.classify, this.resolve, this.resolveSolid, this.commit,
      this.rebuild] as const;
    const prepareLabel = authority === "candidate-ready"
      ? "Prepare structured boundary candidate-ready transaction"
      : "Prepare structured boundary accepted recurring transaction";
    const labels = [prepareLabel, "Classify fine-over-coarse liquid rows",
      "Resolve canonical free-surface boundary slots", "Resolve canonical solid boundary slots",
      "Commit canonical structured boundary slots",
      "Rebuild symmetric structured boundary rows"] as const;
    const compactBoundaryPass = typeof process === "undefined"
      || process.env.FLUID_STRUCTURED_BOUNDARY_COMPACT_PASS !== "0";
    stages.forEach((pipeline, index) => {
      const pass = broker.compute({ label: labels[index] }); pass.setPipeline(pipeline); pass.setBindGroup(0, groups[index]!);
      if (index === 0) pass.dispatchWorkgroups(1);
      else pass.dispatchWorkgroupsIndirect(this.dispatch, index >= 2 && index <= 4 ? 12 : 0);
      // NOTE: stages 1..5 exchange only plain storage, which WebGPU already
      // orders inside a pass, so in principle only the prepare (which writes
      // `this.dispatch` as storage for later indirect reads) needs a boundary.
      // An old collapse coincided with an unrelated in-app rejection. The
      // current exact tripwires and paired 500-step control proved identical
      // state, while the large lane measured a 7 -> 2 pass reduction and a
      // repeatable wall saving. Zero retains the historical split as an A/B.
      // Prepare writes the indirect records and always owns a boundary.
      // Every later stage exchanges only plain storage at stable bindings;
      // the compact arm prices their native in-pass ordering under the full
      // publication and solve tripwires.
      if (index === 0 || !compactBoundaryPass) broker.fence(labels[index]);
    });
    // Workset publication: count -> block scan -> scatter, all five dispatches
    // in the one pass the serial kernel used to occupy. No fence is needed
    // between them -- they hand off only plain storage (`worksetClasses`,
    // `worksetBlocks`, `dynamicWorksets`), which WebGPU orders within a pass,
    // and nothing here reads a buffer this pass writes as an indirect
    // argument. `this.dispatch` is bound INDIRECT-only for the whole pass.
    // `FLUID_STRUCTURED_WORKSET_SPLIT=1` fences each stage into its own pass.
    // SUPERSEDED by `FLUID_GPU_ISOLATE_PASS_LABELS=1` (`lib/webgpu-pass-broker.ts`),
    // which does this for every labelled stage in the frame; keep this only if
    // you want the split without the global flag.
    //
    // This group is where the instrument's two bugs were caught, so the history
    // is worth keeping: these five dispatches once reported 3.604 ms on
    // whichever one was LAST, and swapping the final two moved that identical
    // number with the position rather than the kernel. The cause was NOT Metal
    // encoder merging (that theory was tested and refuted -- bracket coverage
    // never exceeds 1, so passes do not overlap). It was `PassBroker.compute()`
    // silently dropping the label when a pass is already open, leaving one
    // bracket running across every downstream stage, compounded by Dawn's
    // `timestamp_quantization` toggle rounding every reading to 65536 ns so
    // anything genuinely cheap read as exactly zero. With both fixed these five
    // dispatches total 0.19 ms, matching the ~0.3 ms the wall clock measured.
    const splitPublication = process.env.FLUID_STRUCTURED_WORKSET_SPLIT === "1";
    const publish = (label: string, pipeline: GPUComputePipeline, group: GPUBindGroup,
      indirectOffset?: number) => {
      if (splitPublication) broker.fence(label);
      // Supplying every semantic label costs no production pass: the broker
      // reuses its open encoder unless diagnostic label isolation is enabled.
      // It does let a targeted xctrace split expose count/scan/scatter instead
      // of charging their downstream tail to whichever label opened the pass.
      const publication = broker.compute({ label });
      publication.setPipeline(pipeline); publication.setBindGroup(0, group);
      if (indirectOffset === undefined) publication.dispatchWorkgroups(1);
      else publication.dispatchWorkgroupsIndirect(this.dispatch, indirectOffset);
    };
    publish("Structured boundary worksets - count row classes", this.countRowClasses, groups[6]!, 0);
    publish("Structured boundary worksets - count family classes", this.countFamilyClasses, groups[7]!, 12);
    publish("Structured boundary worksets - scan blocks", this.scanWorksetBlocks, groups[8]!);
    publish("Structured boundary worksets - scatter rows", this.scatterRowWorksets, groups[9]!, 0);
    publish("Structured boundary worksets - scatter families", this.scatterFamilyWorksets, groups[10]!, 12);
  }

  /** Build an inactive boundary publication from a structured candidate that
   * has reached its candidate-ready magic. This path never accepts committed
   * structured-control semantics. */
  encodeCandidate(broker: PassBroker, fine: StructuredBoundaryFineSource | undefined,
    structuredCandidateControl: GPUBuffer): void {
    this.encodeFromStructuredControl(broker, fine, structuredCandidateControl, "candidate-ready");
  }

  private encodeAcceptedCandidate(broker: PassBroker, fine: StructuredBoundaryFineSource | undefined): void {
    this.encodeFromStructuredControl(broker, fine, this.resources.structured.control, "accepted");
  }

  encodeReadyCommit(broker: PassBroker): void {
    if (this.destroyed) throw new Error("Structured boundary coefficients are destroyed");
    const pass = broker.compute({ label: "Accept structured boundary publication" });
    pass.setPipeline(this.accept); pass.setBindGroup(0, this.acceptGroup);
    pass.dispatchWorkgroups(1);
  }

  encode(broker: PassBroker, fine: StructuredBoundaryFineSource | undefined): void {
    this.encodeAcceptedCandidate(broker, fine);
    this.encodeReadyCommit(broker);
  }

  /** End the one-shot authored t=0 authority. The immutable authored-shape
   * marker remains so a recurring transaction can bridge only a missing
   * coarse-directory publication; a valid coarse generation always wins. */
  retireAnalyticBootstrap(): void {
    this.device.queue.writeBuffer(this.params, 120, new Uint32Array([0]));
  }

  destroy(): void {
    if (this.destroyed) return; this.destroyed = true;
    for (const buffer of [this.liquidMask, this.solidNormalVelocities, this.candidateMask, this.candidates,
      this.candidateSolidNormalVelocities, this.worksetClasses, this.worksetBlocks,
      this.control, this.candidateControl, this.dispatch, this.params, this.inertSolidStorage,
      this.inertFineParams, this.inertFineStorage]) buffer?.destroy();
    this.worksets.destroy();
  }
}

export const structuredBoundaryCoefficientWGSL = /* wgsl */ `
// \`carry\` is the tenth vec4 and holds Gate A/B arm selectors only:
// x = FLUID_STRUCTURED_BOUNDARY_EXACT_ROW_CARRY, yzw reserved.
struct P{counts:vec4u,resolved:vec4u,dimensions:vec4u,physical:vec4f,offset0:vec4u,offset1:vec4u,offset2:vec4u,pad:vec4u,damDimensions:vec4f,carry:vec4u}
struct FineP{brickDimensions:vec3u,brickResolution:u32,sampleDimensions:vec3u,samplesPerBrick:u32,origin:vec3f,width:f32,worklistCapacity:u32,worklistHeaderWords:u32,pageCapacity:u32,generation:u32,activeCount:u32,invalid:u32,fineFactor:u32,timestep:f32}
// The 64-byte publication has always carried eight words past the pad. They now
// hold a ghost-fluid theta histogram, which is the only production evidence of
// how stiff a scene's free surface actually is: theta multiplies every surface
// face coefficient by 1/theta, so a surface pinned at the 1e-2 floor raises the
// pressure system's condition number by two orders of magnitude and can exhaust
// the encoded solve budget. Diagnostic only; no kernel reads it back.
struct C{flags:atomic<u32>,firstError:atomic<u32>,rows:u32,slots:u32,epoch:u32,bank:u32,published:u32,pad:atomic<u32>,
 theta:array<atomic<u32>,8>}
struct VertexArena{header:array<u32,16>,values:array<u32>}
struct RigidBody{positionShape:vec4f,dimensions:vec4f,orientation:vec4f,linearVelocity:vec4f,angularVelocity:vec4f,inverseMassInertia:vec4f,angularMomentumRestitution:vec4f,material:vec4f}
struct Metric{caseId:u32,transformAndFlags:u32,volume:f32,error:u32}
struct CatalogSlotGeometry{neighborOffsetSize:vec4f,areaCentroid:vec4f,normalInverseDistance:vec4f}
@group(0)@binding(0)var<uniform>p:P;@group(0)@binding(1)var<storage,read>accepted:array<u32>;@group(0)@binding(2)var<storage,read_write>a:array<u32>;@group(0)@binding(3)var<storage,read>geometry:array<vec4u>;@group(0)@binding(5)var<uniform>fp:FineP;@group(0)@binding(6)var<storage,read>fineWork:array<u32>;@group(0)@binding(7)var<storage,read>fineMeta:array<u32>;@group(0)@binding(8)var<storage,read>fineSamples:array<u32>;@group(0)@binding(10)var<storage,read>solid:VertexArena;@group(0)@binding(11)var<storage,read_write>candidates:array<vec2f>;@group(0)@binding(12)var<storage,read_write>candidateLiquid:array<u32>;@group(0)@binding(13)var<storage,read_write>liquid:array<u32>;@group(0)@binding(14)var<storage,read_write>rows:array<u32>;@group(0)@binding(15)var<storage,read_write>diagonal:array<f32>;@group(0)@binding(16)var<storage,read_write>control:C;@group(0)@binding(17)var<storage,read_write>dispatch:array<u32>;@group(0)@binding(18)var<storage,read_write>dynamicWorksets:array<u32>;@group(0)@binding(19)var<storage,read>rigidBodies:array<RigidBody>;@group(0)@binding(20)var<storage,read_write>solidVelocity:array<f32>;@group(0)@binding(21)var<storage,read_write>candidateSolidVelocity:array<f32>;
${fineLevelSetPackedSampleWGSL("fineSamples")}
// The slot-shaped records published by publishStructuredBoundarySetup pin X at
// exactly 65,535 workgroups when the block count saturates one dimension, so
// both the per-item and per-block indices fold back with that constant stride.
// Below saturation Y is 1 and neither second term contributes.
fn foldedItem(g:vec3u)->u32{return g.x+g.y*65535u*64u;}
fn foldedBlock(wg:vec3u)->u32{return wg.x+wg.y*65535u;}
@group(0)@binding(22)var<storage,read_write>acceptedBoundary:C;@group(0)@binding(23)var<storage,read>metrics:array<Metric>;@group(0)@binding(24)var<storage,read>catalogSlots:array<CatalogSlotGeometry>;
@group(0)@binding(25)var<storage,read>separationMask:array<u32>;
// Workset publication scratch: one class per row then one per slot (offset by
// the row capacity), and a per-class table of per-block prefix bases.
@group(0)@binding(27)var<storage,read_write>worksetClasses:array<u32>;
@group(0)@binding(28)var<storage,read_write>worksetBlocks:array<u32>;
// The coarse publisher's exact value/phase delta. Header words: count,
// generation, flags, valid. \`count == 0\` under a valid publication is a
// bitwise proof that every live coarse sample -- phi, minimumPhi, maximumPhi
// and its interface/corrected phase -- is unchanged, with no row inserted and
// none retired. Read only by the exact-carry gate.
@group(0)@binding(29)var<storage,read>coarsePhiDelta:array<u32>;
// Host-rasterized t=0 level set, authoritative only while pad.z==3. Analytic
// and post-bootstrap lanes bind a placeholder here and never sample it.
@group(0)@binding(26)var bootstrapLevelSetIn:texture_3d<f32>;
${makeOctreePowerCoarseLevelSetSampleWGSL(4)}
const INVALID:u32=0xffffffffu;const SOLID_VALID:u32=0x80000000u;fn finite(v:f32)->bool{return v==v&&abs(v)<=3.402823e38;}fn fail(i:u32,f:u32){atomicOr(&control.flags,f);atomicMin(&control.firstError,i);}
// \`control.pad\` is a bit set, not a boolean. Bit 0 is the historical dirty
// channel every stage ors into; bit 1 records that THIS publication retained
// the accepted rows verbatim, which is how the workset scan distinguishes
// "nothing changed" from "nothing ran".
// The published receipt reuses the same two bit positions: bit 0 becomes "the
// liquid mask was proved identical and no live separation mark was observed",
// bit 1 stays "that publication was itself a carry".
const BOUNDARY_DIRTY:u32=1u;const BOUNDARY_CARRIED:u32=2u;const BOUNDARY_RECEIPT_IDENTITY:u32=1u;
const COARSE_PHI_VALID:u32=0x80000000u;const COARSE_PHI_DENSE:u32=0x40000000u;
const ERROR_CARRY:u32=256u;
fn abase()->u32{return control.bank*p.counts.w;}fn rbase()->u32{return control.bank*p.counts.x;}fn section63Base()->u32{return control.bank*p.resolved.y;}
fn rowCenter(row:u32)->vec3f{let g=geometry[rbase()+row];let q=vec3u(g.x%p.dimensions.x,(g.x/p.dimensions.x)%p.dimensions.y,g.x/(p.dimensions.x*p.dimensions.y));return(vec3f(q)+.5*f32(g.y))*p.physical.x;}
fn finePage(key:u32)->u32{if(fp.worklistHeaderWords!=7u||arrayLength(&fineWork)<7u||fineWork[0]!=fp.generation||fineWork[2]!=fp.pageCapacity||(fineWork[3]&3u)!=3u){return INVALID;}let base=7u+fp.worklistCapacity;if(base+key>=arrayLength(&fineWork)){return INVALID;}let id=fineWork[base+key];let m=id*4u;return select(INVALID,id,id<fp.pageCapacity&&m+2u<arrayLength(&fineMeta)&&fineMeta[m]==id&&fineMeta[m+1u]==key&&fineMeta[m+2u]==fp.generation);}
fn loadFine(q:vec3i)->vec2f{if(any(q<vec3i(0))||any(q>=vec3i(fp.sampleDimensions))){return vec2f(0.);}let u=vec3u(q);let brick=u/fp.brickResolution;let local=u-brick*fp.brickResolution;let key=brick.x+fp.brickDimensions.x*(brick.y+fp.brickDimensions.y*brick.z);let id=finePage(key);if(id==INVALID){return vec2f(0.);}let li=local.x+fp.brickResolution*(local.y+fp.brickResolution*local.z);let at=id*fp.samplesPerBrick+li;if(at>=arrayLength(&fineSamples)||(finePackedFlags(at)&1u)==0u||!finite(finePackedPhi(at))){return vec2f(0.);}return vec2f(finePackedPhi(at),1.);}
fn fineSample(x:vec3f,coarsePhi:f32)->vec2f{if(!(fp.width>0.)||fp.brickResolution==0u){return vec2f(coarsePhi,0.);}var lattice=(x-fp.origin)/fp.width-vec3f(.5);let maximum=vec3f(fp.sampleDimensions)-vec3f(1.);let endpointTolerance=1e-4;if(any(lattice<vec3f(-endpointTolerance))||any(lattice>maximum+vec3f(endpointTolerance))){return vec2f(coarsePhi,0.);}lattice=clamp(lattice,vec3f(0.),maximum);let base=vec3i(floor(lattice));let fraction=fract(lattice);var value=0.;for(var corner=0u;corner<8u;corner+=1u){let o=vec3i(i32(corner&1u),i32((corner>>1u)&1u),i32((corner>>2u)&1u));let weight=select(1.-fraction.x,fraction.x,o.x==1)*select(1.-fraction.y,fraction.y,o.y==1)*select(1.-fraction.z,fraction.z,o.z==1);if(weight==0.){continue;}let sample=loadFine(base+o);if(sample.y==0.){return vec2f(coarsePhi,0.);}value+=weight*sample.x;}return vec2f(value,1.);}
fn authoredAnalyticPhiAvailable()->bool{return p.damDimensions.w==1.||p.damDimensions.w==2.;}
// Evaluate the authored t=0 geometry in finest-cell units. Power row centres
// and their one-cell ghost endpoints are half-lattice points by construction;
// doing the large world-space subtraction first gives reflected points the
// same two magnitudes with their final bits swapped. That changes ghost-fluid
// theta even though the geometry is identical. Snap only topology-sized
// roundoff, perform the SDF in the canonical grid frame, and scale to metres
// once at the end. This path retires after bootstrap.
fn analyticInitialPhi(point:vec3f)->f32{
 let rawGrid=point/p.physical.x;let halfGrid=.5*round(2.*rawGrid);
 let grid=select(rawGrid,halfGrid,abs(rawGrid-halfGrid)<=vec3f(1e-4));
 let dimensions=vec3f(p.dimensions.xyz);
 let world=grid-vec3f(.5*dimensions.x,0.,.5*dimensions.z);
 let fill=bitcast<f32>(p.pad.w);
 if(p.pad.z==1u||p.damDimensions.w==1.){return(world.y-fill*dimensions.y)*p.physical.x;}
 let heightFraction=max(.92,fill);let footprintFraction=sqrt(fill/max(heightFraction,1e-9));
 let fallback=vec3f(footprintFraction*dimensions.x,heightFraction*dimensions.y,footprintFraction*dimensions.z);
 let authored=any(p.damDimensions.xyz>vec3f(0.));let damDimensions=select(fallback,p.damDimensions.xyz/p.physical.x,authored);
 let exposedMaximum=vec3f(-.5*dimensions.x+damDimensions.x,damDimensions.y,-.5*dimensions.z+damDimensions.z);
 let q=world-exposedMaximum;let cells=length(max(q,vec3f(0.)))+min(max(q.x,max(q.y,q.z)),0.);
 return cells*p.physical.x;
}
// Texture values live at cell centres. Structured face centres lie on the
// half-cell boundary between those samples, so flooring the world coordinate
// shifts both opposite faces in the same lattice direction and destroys
// reflection symmetry. Interpolate in the cell-centred lattice, matching the
// fine-seed bootstrap sampler. Snap only roundoff-sized half-cell errors so
// mirrored dyadic face centres publish identical interpolation weights.
fn bootstrapTexturePhi(point:vec3f)->f32{
 let raw=point/p.physical.x-vec3f(.5);let half=.5*round(2.*raw);
 let lattice=select(raw,half,abs(raw-half)<=vec3f(1e-4));let base=vec3i(floor(lattice));let t=fract(lattice);
 var terms:array<f32,8>;for(var corner=0u;corner<8u;corner+=1u){let offset=vec3i(i32(corner&1u),i32((corner>>1u)&1u),i32((corner>>2u)&1u));let high=vec3<bool>((corner&1u)!=0u,(corner&2u)!=0u,(corner&4u)!=0u);let weight=select(1.-t.x,t.x,high.x)*select(1.-t.y,t.y,high.y)*select(1.-t.z,t.z,high.z);terms[corner]=weight*textureLoad(bootstrapLevelSetIn,clamp(base+offset,vec3i(0),vec3i(p.dimensions.xyz)-vec3i(1)),0).x;}
 // Reflections permute the eight corners. Fold the identical weighted multiset
 // in numeric order so interpolation is bitwise permutation-invariant.
 for(var i=1u;i<8u;i+=1u){let value=terms[i];var j=i;loop{if(j==0u||terms[j-1u]<=value){break;}terms[j]=terms[j-1u];j-=1u;}terms[j]=value;}
 var sum=0.;for(var i=0u;i<8u;i+=1u){sum+=terms[i];}return sum;}
fn coarseValue(point:vec3f)->vec2f{if(p.pad.z==3u){return vec2f(bootstrapTexturePhi(point),1.);}
 if(p.pad.z!=0u){return vec2f(analyticInitialPhi(point),1.);}let generation=powerCoarseSamples.generation&0x3fffffffu;let expected=control.epoch&0x3fffffffu;if(powerCoarseSamples.state!=0x80000000u||generation!=expected||any(powerCoarseSamples.dimensions!=p.dimensions.xyz)||abs(powerCoarseSamples.physicalCellSize-p.physical.x)>1e-5*max(powerCoarseSamples.physicalCellSize,p.physical.x)){return select(vec2f(1.,0.),vec2f(analyticInitialPhi(point),1.),authoredAnalyticPhiAvailable());}let value=sampleCoarseOctreePhi(point);let valid=finite(value)&&value<3.402823e38;return vec2f(value,select(0.,1.,valid));}
fn publishStructuredBoundarySetup(rowCount:u32,slotCount:u32,generation:u32,bank:u32,valid:bool){atomicStore(&control.flags,select(1u,0u,valid));atomicStore(&control.firstError,select(0u,INVALID,valid));control.rows=select(0u,rowCount,valid);control.slots=select(0u,slotCount,valid);control.epoch=select(0u,generation,valid);control.bank=select(0u,bank,valid);control.published=0u;atomicStore(&control.pad,0u);for(var b=0u;b<8u;b+=1u){atomicStore(&control.theta[b],0u);}let rowBlocks=(control.rows+63u)/64u;dispatch[0]=rowBlocks;dispatch[1]=1u;dispatch[2]=1u;let slotBlocks=(control.slots+63u)/64u;let slotX=max(1u,min(65535u,slotBlocks));dispatch[3]=slotX;dispatch[4]=(slotBlocks+slotX-1u)/slotX;dispatch[5]=1u;}
// Bucket boundaries are chosen around the 1e-2 floor and the values at which a
// 1/theta face coefficient starts to dominate the row: [floor, .05, .1, .25,
// .5, <1, ==1] with bucket 0 reserved for a decoupled face.
fn recordTheta(scale:f32){
 if(!(scale>0.)){atomicAdd(&control.theta[0],1u);return;}
 let theta=1./scale;
 var bucket=7u;
 if(theta<=1.05e-2){bucket=1u;}
 else if(theta<=5e-2){bucket=2u;}
 else if(theta<=1e-1){bucket=3u;}
 else if(theta<=2.5e-1){bucket=4u;}
 else if(theta<=5e-1){bucket=5u;}
 else if(theta<0.999){bucket=6u;}
 atomicAdd(&control.theta[bucket],1u);}
// \`accepted[15]\` is the structured publication control's \`reserved[0]\`, the
// bit \`beginStructuredPublication\` sets when the GPU row delta proved an exact
// topology identity. It is the same bit that zeroed dispatch records 0 and 3,
// so it certifies both that the row/slot geometry is bitwise unchanged AND that
// the family scatter did not reset fractions/pressureScales underneath the
// coefficients this boundary published last epoch.
fn boundaryLiquidIdentityEligible()->bool{return (atomicLoad(&control.pad)&BOUNDARY_DIRTY)==0u
  &&arrayLength(&accepted)>15u&&accepted[15u]!=0u
  &&atomicLoad(&acceptedBoundary.flags)==0u&&acceptedBoundary.rows==control.rows
  &&acceptedBoundary.slots==control.slots&&acceptedBoundary.epoch!=0u
  &&acceptedBoundary.published==acceptedBoundary.epoch&&acceptedBoundary.bank==control.bank;}
// A genuine (0,1,1). publishStructuredBoundarySetup floors the slot record's X
// at one, so pushing a zero count through its arithmetic still launches a
// workgroup; this is the boundary's analogue of publishExactRowDispatch.
fn publishExactBoundaryDispatch(at:u32,blocks:u32){if(blocks==0u){dispatch[at]=0u;dispatch[at+1u]=1u;dispatch[at+2u]=1u;return;}let x=max(1u,min(65535u,blocks));dispatch[at]=x;dispatch[at+1u]=(blocks+x-1u)/x;dispatch[at+2u]=1u;}
// The phi side of the gate. Everything this checks is a precondition
// \`coarseValue\` would itself have re-established before sampling, plus the
// coarse publisher's exact delta: zero changed records under a valid
// publication is a bitwise proof that every live sample is unchanged. The dense
// complement (generation bit 30) is excluded because the delta walks only the
// compact directory rows and says nothing about the dense tail.
fn boundaryCoarsePhiIdentity()->bool{
 if(powerCoarseSamples.state!=COARSE_PHI_VALID
  ||(powerCoarseSamples.generation&COARSE_PHI_DENSE)!=0u
  ||(powerCoarseSamples.generation&0x3fffffffu)!=(control.epoch&0x3fffffffu)
  ||any(powerCoarseSamples.dimensions!=p.dimensions.xyz)
  ||abs(powerCoarseSamples.physicalCellSize-p.physical.x)>1e-5*max(powerCoarseSamples.physicalCellSize,p.physical.x)){return false;}
 return arrayLength(&coarsePhiDelta)>=4u&&coarsePhiDelta[3]==COARSE_PHI_VALID
  &&coarsePhiDelta[2]==0u&&coarsePhiDelta[0]==0u
  &&(coarsePhiDelta[1]&0x3fffffffu)==(control.epoch&0x3fffffffu);}
// Forward exactness gate. \`eligible\` already carries the retrospective half --
// a complete accepted publication at the same rows/slots/bank over carried
// topology. This adds the forward conjuncts the receipt cannot supply:
//   receipt bit 0         the last full recompute reproduced the accepted
//                         liquid mask AND observed no live separation mark, so
//                         the retained coefficients are a fixed point of the
//                         boundary map under the accepted inputs;
//   receipt bit 1         mode 1 refuses to carry a publication that was itself
//                         a carry. Every other conjunct is transitive across a
//                         chain; the separation mask is not, because a carried
//                         step never runs the resolve that observes it.
//                         Alternating caps that staleness at one step.
//   coarse phi identity   the level set this map reads is bitwise unchanged;
//   fp.width / bodyCount  the two inputs with no receipt at all are absent.
// p.pad.z!=0 means an authored or imported bootstrap still owns phi, which is a
// one-shot authority the carry must not extend.
fn boundaryExactRowCarryEligible(eligible:bool)->bool{
 if(p.carry.x==0u||!eligible){return false;}
 let receipt=atomicLoad(&acceptedBoundary.pad);
 if((receipt&BOUNDARY_RECEIPT_IDENTITY)==0u){return false;}
 if(p.carry.x==1u&&(receipt&BOUNDARY_CARRIED)!=0u){return false;}
 if(p.pad.z!=0u||fp.width>0.||p.dimensions.w!=0u){return false;}
 return boundaryCoarsePhiIdentity();}
@compute @workgroup_size(1)fn prepareStructuredBoundaryCandidate(){if(arrayLength(&accepted)<6u){publishStructuredBoundarySetup(0u,0u,0u,0u,false);return;}let rowCount=accepted[2];let slotCount=accepted[3];let generation=accepted[4];let bank=accepted[5];let valid=accepted[0]==${OCTREE_STRUCTURED_GPU_VALID}u&&rowCount<=p.counts.x&&slotCount<=p.counts.y&&generation!=0u&&bank<=1u;publishStructuredBoundarySetup(rowCount,slotCount,generation,bank,valid);let eligible=valid&&boundaryLiquidIdentityEligible();let carry=boundaryExactRowCarryEligible(eligible);atomicStore(&control.pad,select(select(BOUNDARY_DIRTY,0u,eligible),BOUNDARY_CARRIED,carry));
 // Zero both live records. That removes nine of the eleven encoded dispatches:
 // classify/resolve/resolveSolid/commit/rebuild and the two count and two
 // scatter publications all read one of these two. Only the direct prepare and
 // the direct block scan survive, and the scan takes its own carry arm below.
 // \`rebuild\` normally advances \`published\`, so the carry has to advance it here
 // or worksetPublicationEnabled and acceptStructuredBoundary both fail closed
 // and the boundary never publishes at all. The theta histogram is a pure
 // diagnostic that no kernel reads back; carry it forward so a carried step
 // reports the surface stiffness it actually still has rather than zeros.
 if(carry){publishExactBoundaryDispatch(0u,0u);publishExactBoundaryDispatch(3u,0u);control.published=control.epoch;for(var b=0u;b<8u;b+=1u){atomicStore(&control.theta[b],atomicLoad(&acceptedBoundary.theta[b]));}}}
@compute @workgroup_size(1)fn prepareStructuredBoundaryAccepted(){if(arrayLength(&accepted)<6u){publishStructuredBoundarySetup(0u,0u,0u,0u,false);return;}let rowCount=accepted[2];let generation=accepted[3];let bank=accepted[4];let slotCount=accepted[5];let valid=accepted[0]==0u&&rowCount<=p.counts.x&&slotCount<=p.counts.y&&generation!=0u&&bank<=1u;publishStructuredBoundarySetup(rowCount,slotCount,generation,bank,valid);atomicStore(&control.pad,BOUNDARY_DIRTY);}
fn solidAt(row:u32,x:vec3f)->f32{if(p.resolved.w==0u||solid.header[5]!=SOLID_VALID||row>=solid.header[2]){return 1e20;}let rg=geometry[rbase()+row];let q=vec3u(rg.x%p.dimensions.x,(rg.x/p.dimensions.x)%p.dimensions.y,rg.x/(p.dimensions.x*p.dimensions.y));let t=clamp((x/p.physical.x-vec3f(q))/f32(rg.y),vec3f(0),vec3f(1));var v=0.;for(var corner=0u;corner<8u;corner+=1u){let w=select(1.-t.x,t.x,(corner&1u)!=0u)*select(1.-t.y,t.y,(corner&2u)!=0u)*select(1.-t.z,t.z,(corner&4u)!=0u);v+=w*bitcast<f32>(solid.values[row*8u+corner]);}return v;}
@compute @workgroup_size(64)fn classifyStructuredLiquidRows(@builtin(global_invocation_id)g:vec3u){let row=g.x;if(row>=control.rows){return;}let point=rowCenter(row);let coarseSample=coarseValue(point);let sample=fineSample(point,coarseSample.x);if(coarseSample.y==0.&&sample.y==0.){candidateLiquid[row]=0u;atomicOr(&control.pad,BOUNDARY_DIRTY);fail(row,2u);return;}candidateLiquid[row]=select(0u,1u,sample.x<0.);}
fn handleOwner(h:u32)->u32{return a[abase()+p.offset0.x+h];}fn handleNeighbor(h:u32)->u32{return a[abase()+p.offset0.y+h];}fn handleNormal(h:u32)->vec3f{let at=abase()+p.offset1.z+4u*h;return vec3f(bitcast<f32>(a[at]),bitcast<f32>(a[at+1u]),bitcast<f32>(a[at+2u]));}fn handleCenter(h:u32)->vec3f{let at=abase()+p.offset1.w+4u*h;return vec3f(bitcast<f32>(a[at]),bitcast<f32>(a[at+1u]),bitcast<f32>(a[at+2u]));}fn sbase()->u32{return control.bank*p.counts.y;}fn lbase()->u32{return control.bank*p.counts.x;}fn liquidAt(row:u32)->u32{return liquid[lbase()+row];}
// Lattice-origin face centroids to the centred world frame authored rigid
// poses live in. handleCenter shares the structured centroids region,
// which stores (cellCoord + .5*size)*h and therefore spans [0, dim*h] on
// every axis; worldVertex in the solid vertex-SDF producer centres x and z
// about the container. Sampling a body SDF with the uncentred point looked
// for every body half a domain away in x and z.
fn solidWorld(x:vec3f)->vec3f{return x-vec3f(.5*f32(p.dimensions.x)*p.physical.x,0.,.5*f32(p.dimensions.z)*p.physical.x);}
fn quaternionRotate(q:vec4f,v:vec3f)->vec3f{let uv=cross(q.yzw,v);let uuv=cross(q.yzw,uv);return v+2.*(q.x*uv+uuv);}fn quaternionInverseRotate(q:vec4f,v:vec3f)->vec3f{return quaternionRotate(vec4f(q.x,-q.yzw),v);}fn rigidSdf(body:RigidBody,world:vec3f)->f32{let q=quaternionInverseRotate(body.orientation,world-body.positionShape.xyz);let d=body.dimensions.xyz;let shape=i32(round(body.positionShape.w));if(shape==0){return length(q)-d.x;}if(shape==1){let b=abs(q)-.5*d;return length(max(b,vec3f(0)))+min(max(b.x,max(b.y,b.z)),0.);}if(shape==2){let cy=clamp(q.y,-.5*d.y,.5*d.y);return length(vec3f(q.x,q.y-cy,q.z))-d.x;}let radial=length(q.xz)-d.x;let axial=abs(q.y)-.5*d.y;return length(max(vec2f(radial,axial),vec2f(0)))+min(max(radial,axial),0.);}
fn worldBoundaryBit(h:u32)->u32{if(handleNeighbor(h)!=INVALID){return 0u;}let x=handleCenter(h)/p.physical.x;let n=handleNormal(h);for(var axis=0u;axis<3u;axis+=1u){if(n[axis]<-.5&&x[axis]<=1e-4){return 1u<<(2u*axis);}if(n[axis]>.5&&x[axis]>=f32(p.dimensions[axis])-1e-4){return 1u<<(2u*axis+1u);}}return 0u;}fn closedWorld(h:u32)->bool{let bit=worldBoundaryBit(h);return bit!=0u&&(p.resolved.z&bit)!=0u;}
// Unilateral overhead contact: the previous projection marked rows holding
// tension against gravity-opposed closed world faces, storing the exact
// face bits. A fresh mark opens those faces for this epoch, so the solve
// resolves the separation itself (p = 0 ghost past the face) instead of
// pinning the liquid with suction.
fn separationFresh(row:u32,faceBit:u32)->bool{
  let at=rbase()+row;
  if(at>=arrayLength(&geometry)){return false;}
  let cell=geometry[at].x;
  if(cell>=arrayLength(&separationMask)){return false;}
  let word=separationMask[cell];
  if((word&faceBit)==0u){return false;}
  let marked=word>>6u;
  let epoch=(control.epoch&0x3fffffffu)&0x3ffffffu;
  let age=(epoch-marked)&0x3ffffffu;
  return age<=2u;
}
// The mark WITHOUT the age test. separationFresh ages a mark against
// control.epoch, so a bitwise unchanged mask still flips an aperture from open
// to prescribed-solid as the epoch advances. A carry that skipped the resolve
// would keep the stale open face, so any live mark anywhere must veto the next
// step's carry. This is the only exactness conjunct the gate cannot evaluate
// itself: the mask is dimension-sized and the prepare kernel is one lane.
fn separationMarked(row:u32,faceBit:u32)->bool{
  let at=rbase()+row;
  if(at>=arrayLength(&geometry)){return false;}
  let cell=geometry[at].x;
  if(cell>=arrayLength(&separationMask)){return false;}
  return (separationMask[cell]&faceBit)!=0u;
}
@compute @workgroup_size(64)fn resolveStructuredBoundarySlots(@builtin(global_invocation_id)g:vec3u){let h=foldedItem(g);if(h>=control.slots||atomicLoad(&control.flags)!=0u){return;}let lo=handleOwner(h);let hi=handleNeighbor(h);if(lo>=control.rows){fail(h,4u);return;}let loPoint=rowCenter(lo);let world=worldBoundaryBit(h)!=0u;var hiPoint=handleCenter(h);if(hi!=INVALID){hiPoint=rowCenter(hi);}else if(!world){let inv=bitcast<f32>(a[abase()+p.offset0.w+h]);hiPoint=loPoint+handleNormal(h)/max(inv,1e-20);}let clo=coarseValue(loPoint);let chi=coarseValue(hiPoint);let plo=fineSample(loPoint,clo.x);let phi=fineSample(hiPoint,chi.x);if((clo.y==0.&&plo.y==0.)||(!world&&chi.y==0.&&phi.y==0.)){fail(h,8u|128u);return;}var scale=1.;
// Ghost-fluid crossing handling covers every orientation the moving front
// produces. A band row's centre may legitimately sit on the dry side of its
// own surface face for a step (the frontier admits rows the interface has
// not reached yet); hard-failing that geometry poisoned the whole topology
// epoch, permanently freezing the accepted velocity authority the first time
// an off-cadence step produced it. theta is measured from whichever side is
// wet, with the reference lane's floor so a sliver crossing cannot blow up
// the gradient; a topologically open face with no wet side simply decouples.
if(!world&&(hi==INVALID||((plo.x<0.)!=(phi.x<0.)))){
  if(plo.x<0.&&!(phi.x<0.)){
    let theta=max(-plo.x/(phi.x-plo.x),1e-2);
    if(!finite(theta)||theta>1.){fail(h,8u|1024u);return;}
    scale=1./theta;
  }else if(hi!=INVALID&&phi.x<0.&&!(plo.x<0.)){
    let theta=max(-phi.x/(plo.x-phi.x),1e-2);
    if(!finite(theta)||theta>1.){fail(h,8u|1024u);return;}
    scale=1./theta;
  }else if(plo.x<0.&&phi.x<0.){
    // Open face whose ghost point is still wet: plain liquid gradient.
    scale=1.;
  }else{
    // No wet side: the face carries extended air velocity, no coupling.
    scale=0.;
  }
}else if(hi!=INVALID&&plo.x>=0.&&phi.x>=0.){scale=0.;}else if(world&&plo.x>=0.){scale=0.;}
let boundaryBit=worldBoundaryBit(h);
var aperture=1.;
if(boundaryBit!=0u&&(p.resolved.z&boundaryBit)!=0u){
  // Closed world face. A face carrying a fresh separation mark for exactly
  // this face bit opens for this epoch; every other closed face stays a
  // prescribed solid.
  // Observing a mark of any age publishes the dirty bit, which withdraws the
  // NEXT step's exact row carry. Behind p.carry.x so the disabled arm executes
  // neither the load nor the atomic and stays byte-identical.
  if(p.carry.x!=0u&&separationMarked(lo,boundaryBit)){atomicOr(&control.pad,BOUNDARY_DIRTY);}
  aperture=select(0.,1.,separationFresh(lo,boundaryBit));
}
recordTheta(scale);candidates[h]=vec2f(aperture,scale);}
@compute @workgroup_size(64)fn resolveStructuredSolidSlots(@builtin(global_invocation_id)g:vec3u){let h=foldedItem(g);if(h>=control.slots||atomicLoad(&control.flags)!=0u){return;}let lo=handleOwner(h);let hi=handleNeighbor(h);if(lo>=control.rows){fail(h,4u);return;}let x=handleCenter(h);let n=handleNormal(h);let inv=bitcast<f32>(a[abase()+p.offset0.w+h]);let half=.5/max(inv,1e-20);var aperture=candidates[h].x;if(aperture>0.&&p.resolved.w!=0u){let sdf=min(solidAt(lo,x),select(1e20,solidAt(hi,x),hi!=INVALID));if(!finite(sdf)){fail(h,16u);return;}aperture=clamp(.5+sdf/max(p.physical.x,1e-20),0.,1.);}let world=solidWorld(x);var best=1e20;var normalVelocity=0.;for(var body=0u;body<p.dimensions.w;body+=1u){if(body>=arrayLength(&rigidBodies)){fail(h,32u);return;}let rb=rigidBodies[body];let sdf=rigidSdf(rb,world);if(sdf<half&&sdf<best){best=sdf;normalVelocity=dot(rb.linearVelocity.xyz+cross(rb.angularVelocity.xyz,world-rb.positionShape.xyz),n);}}candidateSolidVelocity[h]=normalVelocity;candidates[h]=vec2f(aperture,candidates[h].y);}
@compute @workgroup_size(64)fn commitStructuredBoundarySlots(@builtin(global_invocation_id)g:vec3u){let h=foldedItem(g);if(h>=control.slots||atomicLoad(&control.flags)!=0u){return;}a[abase()+p.offset1.x+h]=bitcast<u32>(candidates[h].x);a[abase()+p.offset1.y+h]=bitcast<u32>(candidates[h].y);solidVelocity[sbase()+h]=candidateSolidVelocity[h];}
fn canonicalDirection(channel:u32)->vec3i{let d=array<vec3i,18>(vec3i(1,0,0),vec3i(-1,0,0),vec3i(0,1,0),vec3i(0,-1,0),vec3i(0,0,1),vec3i(0,0,-1),vec3i(1,1,0),vec3i(1,-1,0),vec3i(-1,1,0),vec3i(-1,-1,0),vec3i(1,0,1),vec3i(1,0,-1),vec3i(-1,0,1),vec3i(-1,0,-1),vec3i(0,1,1),vec3i(0,1,-1),vec3i(0,-1,1),vec3i(0,-1,-1));return d[channel];}
fn channelForCatalogSlot(global:u32)->u32{if(global>=arrayLength(&catalogSlots)){return INVALID;}let slot=catalogSlots[global];let direction=vec3i(round(slot.areaCentroid.yzw+.5*slot.normalInverseDistance.xyz));for(var channel=0u;channel<18u;channel+=1u){if(all(direction==canonicalDirection(channel))){return channel+1u;}}return INVALID;}
const HYBRID_POWER_BAND:u32=0x80000000u;
@compute @workgroup_size(64)fn rebuildStructuredBoundaryRows(@builtin(global_invocation_id)g:vec3u){let row=g.x;if(row>=control.rows||atomicLoad(&control.flags)!=0u){return;}let value=candidateLiquid[row];if(!boundaryLiquidIdentityEligible()||value!=liquid[acceptedBoundary.bank*p.counts.x+row]){atomicOr(&control.pad,BOUNDARY_DIRTY);}liquid[lbase()+row]=value;let base=section63Base()+row*19u;if(base+19u>arrayLength(&rows)){fail(row,64u);return;}for(var channel=0u;channel<19u;channel+=1u){rows[base+channel]=0u;}var powerBand=candidateLiquid[row]==0u;var diagonalTerms:array<f32,31>;for(var local=0u;local<31u;local+=1u){diagonalTerms[local]=0.;}for(var local=0u;local<p.counts.z;local+=1u){let at=row*p.counts.z+local;let h=a[abase()+p.offset2.y+at];if(h==INVALID||h>=control.slots){continue;}let lo=handleOwner(h);let hi=handleNeighbor(h);let other=select(lo,hi,lo==row);let aperture=bitcast<f32>(a[abase()+p.offset1.x+h]);let scale=bitcast<f32>(a[abase()+p.offset1.y+h]);powerBand=powerBand||hi==INVALID||aperture!=1.||scale!=1.||(other<control.rows&&candidateLiquid[other]!=candidateLiquid[row]);let coefficient=bitcast<f32>(a[abase()+p.offset0.z+h])*bitcast<f32>(a[abase()+p.offset0.w+h])*aperture*scale;if(candidateLiquid[row]==0u||coefficient<=0.){continue;}diagonalTerms[local]=coefficient;let global=a[abase()+p.offset2.w+at];let channel=channelForCatalogSlot(global);if(channel==INVALID){fail(row,64u);continue;}if(other!=INVALID&&other<control.rows&&other!=row&&candidateLiquid[other]!=0u){rows[base+channel]=bitcast<u32>(bitcast<f32>(rows[base+channel])+coefficient);}}for(var i=1u;i<31u;i+=1u){let value=diagonalTerms[i];var j=i;loop{if(j==0u||diagonalTerms[j-1u]<=value){break;}diagonalTerms[j]=diagonalTerms[j-1u];j-=1u;}diagonalTerms[j]=value;}var sum=0.;for(var i=0u;i<31u;i+=1u){sum+=diagonalTerms[i];}let d=select(1.,sum,candidateLiquid[row]!=0u);rows[base]=bitcast<u32>(d);worksetClasses[row]=select(0u,HYBRID_POWER_BAND,powerBand);if(row+1u==control.rows){control.published=control.epoch;}}
fn rowTransition(row:u32)->bool{let at=rbase()+row;return row<control.rows&&at<arrayLength(&geometry)&&geometry[at].z!=0u;}
// Class 4 is a separate dry-identity proof, not regular liquid interior. A
// failed proof falls through to the pre-authored power-band bit, so corrupt or
// non-identity dry rows keep the general operator instead of being optimized.
fn dynamicRowClass(row:u32)->u32{let base=section63Base()+row*19u;var off=0.;var dryIdentity=liquidAt(row)==0u&&rows[base]==bitcast<u32>(1.);for(var channel=1u;channel<19u;channel+=1u){let value=bitcast<f32>(rows[base+channel]);off+=value;dryIdentity=dryIdentity&&rows[base+channel]==0u;}if(${octreePowerHybridDryIdentityEnabled() ? "dryIdentity" : "false"}){return 4u;}let boundary=bitcast<f32>(rows[base])>off+max(1e-6,1e-5*abs(off))||(worksetClasses[row]&HYBRID_POWER_BAND)!=0u;return select(select(0u,2u,boundary),select(1u,3u,boundary),rowTransition(row));}
// A row's class is read once per row and up to twice per incident slot. The
// serial publication recomputed it every time, so the 19-channel offdiagonal
// sum ran up to four times per slot; caching it is the bulk of the win.
fn cachedRowClass(row:u32)->u32{return select(dynamicRowClass(row),worksetClasses[row],row<control.rows);}
fn dynamicFamilyClass(h:u32)->u32{let lo=handleOwner(h);let hi=handleNeighbor(h);if(liquidAt(lo)==0u&&(hi==INVALID||hi>=control.rows||liquidAt(hi)==0u)){return 9u;}let transition=rowTransition(lo)||(hi!=INVALID&&hi<control.rows&&rowTransition(hi));let aperture=bitcast<f32>(a[abase()+p.offset1.x+h]);let scale=bitcast<f32>(a[abase()+p.offset1.y+h]);let boundary=hi==INVALID||cachedRowClass(lo)>=2u||(hi<control.rows&&cachedRowClass(hi)>=2u)||aperture<0.999999||abs(scale-1.)>1e-6||(hi<control.rows&&liquidAt(lo)!=liquidAt(hi));return select(select(5u,7u,boundary),select(6u,8u,boundary),transition);}
fn worksetBase(cls:u32)->u32{return control.bank*p.pad.y+cls*p.pad.x;}
// Wide deterministic workset publication.
//
// This replaces a single 64-lane workgroup that walked every row and every
// slot, ran a 9-class Hillis-Steele scan across its own lanes (108 barriers),
// and then walked both ranges a second time to scatter. At ~12k slots that is
// a latency-bound serial crawl: 64 resident threads of 98,304, with nothing
// to hide a dependent global load behind. It measured 4.00 ms of a 48.7 ms
// advance.
//
// The replacement is the standard count -> block scan -> scatter compaction,
// one lane per element. Destinations are IDENTICAL to the serial form, not
// merely equivalent: a block's base is the exclusive prefix of the per-block
// counts in ascending block order, and a lane's offset within its block is
// the number of lower lanes carrying the same class, so every class's workset
// is still exactly its members in ascending element order. That determinism
// is why the scan is an ordered integer prefix and not an atomic bump; an
// atomic counter is also banned outright in this shader.
//
// Row classes are 0..3 and family classes 5..8, which never overlap, so the
// row and slot phases share the block table without a shared cursor: the row
// count writes classes 0..4 at row-block indices and the slot count writes
// classes 5..8 at slot-block indices.
const WORKSET_SKIP:u32=0xffffffffu;
    var<workgroup> worksetBlockClass:array<u32,64>;
    var<workgroup> worksetScan:array<u32,256>;
    var<workgroup> worksetGate:u32;
    var<workgroup> worksetLiveRowBlocks:u32;
    var<workgroup> worksetLiveSlotBlocks:u32;
fn worksetPublicationEnabled()->bool{return atomicLoad(&control.flags)==0u&&control.rows>0u&&control.published==control.epoch;}
fn boundaryRowCarryActive()->bool{return (atomicLoad(&control.pad)&BOUNDARY_CARRIED)!=0u;}
fn worksetBlockStride()->u32{return (max(p.counts.x,p.counts.y)+63u)/64u;}
fn worksetSlotClassBase()->u32{return p.counts.x;}
fn worksetBlockTotal(cls:u32)->u32{var n=0u;for(var j=0u;j<64u;j+=1u){if(worksetBlockClass[j]==cls){n+=1u;}}return n;}
fn worksetBlockOffset(lane:u32)->u32{let cls=worksetBlockClass[lane];var n=0u;for(var j=0u;j<lane;j+=1u){if(worksetBlockClass[j]==cls){n+=1u;}}return n;}
@compute @workgroup_size(64)fn countStructuredRowClasses(@builtin(global_invocation_id)g:vec3u,@builtin(local_invocation_index)lane:u32,@builtin(workgroup_id)wg:vec3u){
 if(lane==0u){worksetGate=select(0u,1u,worksetPublicationEnabled());}
 if(workgroupUniformLoad(&worksetGate)==0u){return;}
 let row=g.x;
 var cls=WORKSET_SKIP;
 if(row<control.rows){cls=dynamicRowClass(row);worksetClasses[row]=cls;}
 worksetBlockClass[lane]=cls;
 workgroupBarrier();
 if(lane<5u){worksetBlocks[lane*worksetBlockStride()+wg.x]=worksetBlockTotal(lane);}
}
@compute @workgroup_size(64)fn countStructuredFamilyClasses(@builtin(global_invocation_id)g:vec3u,@builtin(local_invocation_index)lane:u32,@builtin(workgroup_id)wg:vec3u){
 if(lane==0u){worksetGate=select(0u,1u,worksetPublicationEnabled());}
 if(workgroupUniformLoad(&worksetGate)==0u){return;}
 let h=foldedItem(g);
 var cls=WORKSET_SKIP;
 if(h<control.slots){let value=dynamicFamilyClass(h);worksetClasses[worksetSlotClassBase()+h]=value;if(value<9u){cls=value;}}
 worksetBlockClass[lane]=cls;
 workgroupBarrier();
 if(lane>=5u&&lane<9u){worksetBlocks[lane*worksetBlockStride()+foldedBlock(wg)]=worksetBlockTotal(lane);}
}
    @compute @workgroup_size(256)fn scanStructuredWorksetBlocks(@builtin(local_invocation_index)lane:u32){
     if(lane==0u){worksetGate=select(0u,select(1u,2u,boundaryRowCarryActive()),worksetPublicationEnabled());worksetLiveRowBlocks=(control.rows+63u)/64u;worksetLiveSlotBlocks=(control.slots+63u)/64u;}
     let worksetMode=workgroupUniformLoad(&worksetGate);
     if(worksetMode==0u){return;}
 // Exact row carry. This kernel is the one direct dispatch in the publication,
 // so it still launches when the two count records are zero -- and it rewrites
 // worksetBlocks IN PLACE from counts to exclusive prefixes. Re-running it over
 // a table that already holds prefixes would prefix them a second time. Instead
 // re-stamp the nine retained workset headers with the candidate epoch and
 // revalidate every field the persistent solver's validateAcceptedRowPartition
 // reads: the accepted epoch stamp, n <= capacity, the 3-bit shape word, the
 // exact (n+63)/64 dispatch triple in words 4..6, the list bound, and
 // total(classes 0..4) == rows. Nothing is stamped until all nine validate, so
 // a rejected carry leaves the accepted publication byte for byte where it was
 // and acceptStructuredBoundary declines to republish. Uniform control flow:
 // the whole workgroup returns here together, before any barrier below.
     if(worksetMode==2u){
      if(lane==0u){
       let acceptedEpoch=acceptedBoundary.epoch;
       var valid=control.epoch!=0u&&acceptedEpoch!=0u;var rowTotal=0u;
       for(var cls=0u;cls<9u;cls+=1u){
        let base=worksetBase(cls);
        if(base+7u>arrayLength(&dynamicWorksets)){valid=false;continue;}
        let n=dynamicWorksets[base+1u];let blocks=(n+63u)/64u;let dx=max(1u,min(65535u,blocks));
        valid=valid&&dynamicWorksets[base]==acceptedEpoch&&n<=dynamicWorksets[base+2u]
         &&(dynamicWorksets[base+3u]&3u)==3u&&dynamicWorksets[base+4u]==dx
         &&dynamicWorksets[base+5u]==(blocks+dx-1u)/dx&&dynamicWorksets[base+6u]==1u
         &&base+7u+n<=arrayLength(&dynamicWorksets);
        if(cls<5u){rowTotal+=n;}
       }
       valid=valid&&rowTotal==control.rows;
       if(valid){for(var cls=0u;cls<9u;cls+=1u){dynamicWorksets[worksetBase(cls)]=control.epoch;}}
       else{fail(0u,ERROR_CARRY);}
      }
      return;
     }
 // Every barrier below is reached under bounds derived from the uniform p, so
 // the loop trip counts are uniform even though the live-element predicate is
     // not; out-of-range blocks contribute a zero and never gate a barrier.
     let stride=worksetBlockStride();
     let rowBlocks=workgroupUniformLoad(&worksetLiveRowBlocks);
     let slotBlocks=workgroupUniformLoad(&worksetLiveSlotBlocks);
 for(var cls=0u;cls<9u;cls+=1u){
  let blocks=select(rowBlocks,slotBlocks,cls>=5u);
  var carry=0u;
     // blocks is uniform and describes the live publication. Iterating to
     // the capacity-derived stride made mini scan 1,920 block slots per
     // class although only ~24 row / ~188 family blocks contained work.
     for(var start=0u;start<blocks;start+=256u){
   let at=start+lane;
   let live=at<blocks;
   var value=0u;if(live){value=worksetBlocks[cls*stride+at];}
   worksetScan[lane]=value;workgroupBarrier();
   for(var width=1u;width<256u;width<<=1u){
    var add=0u;if(lane>=width){add=worksetScan[lane-width];}
    workgroupBarrier();worksetScan[lane]+=add;workgroupBarrier();
   }
   if(live){worksetBlocks[cls*stride+at]=carry+worksetScan[lane]-value;}
   let total=worksetScan[255];workgroupBarrier();
   carry+=total;
  }
  if(lane==0u){let base=worksetBase(cls);dynamicWorksets[base]=control.epoch;dynamicWorksets[base+1u]=carry;dynamicWorksets[base+2u]=select(p.counts.x,p.counts.y,cls>=5u);dynamicWorksets[base+3u]=3u;let carryBlocks=(carry+63u)/64u;let carryX=max(1u,min(65535u,carryBlocks));dynamicWorksets[base+4u]=carryX;dynamicWorksets[base+5u]=(carryBlocks+carryX-1u)/carryX;dynamicWorksets[base+6u]=1u;}
 }
}
@compute @workgroup_size(64)fn scatterStructuredRowWorksets(@builtin(global_invocation_id)g:vec3u,@builtin(local_invocation_index)lane:u32,@builtin(workgroup_id)wg:vec3u){
 if(lane==0u){worksetGate=select(0u,1u,worksetPublicationEnabled());}
 if(workgroupUniformLoad(&worksetGate)==0u){return;}
 let row=g.x;
 var cls=WORKSET_SKIP;
 if(row<control.rows){cls=worksetClasses[row];}
 worksetBlockClass[lane]=cls;
 workgroupBarrier();
 if(cls>=9u){return;}
 dynamicWorksets[worksetBase(cls)+7u+worksetBlocks[cls*worksetBlockStride()+wg.x]+worksetBlockOffset(lane)]=row;
}
@compute @workgroup_size(64)fn scatterStructuredFamilyWorksets(@builtin(global_invocation_id)g:vec3u,@builtin(local_invocation_index)lane:u32,@builtin(workgroup_id)wg:vec3u){
 if(lane==0u){worksetGate=select(0u,1u,worksetPublicationEnabled());}
 if(workgroupUniformLoad(&worksetGate)==0u){return;}
 let h=foldedItem(g);
 var cls=WORKSET_SKIP;
 if(h<control.slots){cls=worksetClasses[worksetSlotClassBase()+h];}
 worksetBlockClass[lane]=cls;
 workgroupBarrier();
 if(cls>=9u){return;}
 dynamicWorksets[worksetBase(cls)+7u+worksetBlocks[cls*worksetBlockStride()+foldedBlock(wg)]+worksetBlockOffset(lane)]=h;
}
@compute @workgroup_size(1)fn acceptStructuredBoundary(){if(atomicLoad(&control.flags)!=0u||control.epoch==0u||control.published!=control.epoch){return;}let pad=atomicLoad(&control.pad);let liquidIdentity=select(0u,BOUNDARY_RECEIPT_IDENTITY,(pad&BOUNDARY_DIRTY)==0u)|(pad&BOUNDARY_CARRIED);atomicStore(&acceptedBoundary.flags,0u);atomicStore(&acceptedBoundary.firstError,INVALID);acceptedBoundary.rows=control.rows;acceptedBoundary.slots=control.slots;acceptedBoundary.epoch=control.epoch;acceptedBoundary.bank=control.bank;acceptedBoundary.published=control.published;atomicStore(&acceptedBoundary.pad,liquidIdentity);for(var b=0u;b<8u;b+=1u){atomicStore(&acceptedBoundary.theta[b],atomicLoad(&control.theta[b]));}}
`;
