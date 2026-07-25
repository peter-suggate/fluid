import type { WebGPUFineLevelSetBrickSource } from "./webgpu-octree-fine-levelset-bricks";
import { fineLevelSetLinearWorkgroupWGSL,
  planFineLevelSetDispatch2D } from "./webgpu-fine-levelset-dispatch";
import { PassBroker } from "./webgpu-pass-broker";
import type { FineLevelSetTransportTopologyDelta } from "./webgpu-octree-fine-levelset-transport";

export const FINE_LEVELSET_TOPOLOGY_ERROR = Object.freeze({
  capacity: 1 << 0,
  hashProbe: 1 << 1,
  nonfiniteCoarsePhi: 1 << 2,
  malformedGeneration: 1 << 3,
  downstreamPublication: 1 << 4,
} as const);

/** `control[7]` reason bits written when downstream publication is rejected. */
export const FINE_LEVELSET_TOPOLOGY_FINALIZE_REASON = Object.freeze({
  topology: 1 << 0,
  redistance: 1 << 1,
  volume: 1 << 2,
  transport: 1 << 3,
} as const);

export interface FineLevelSetGPUTopologyControl {
  flags: number;
  interfaceBricks: number;
  desiredBricks: number;
  /** Exact on success; a strict lower bound (> capacity) on page overflow. */
  requiredDesiredBricks: number;
  requiredDesiredBricksExact: boolean;
  activatedBricks: number;
  published: boolean;
  rolledBack: boolean;
  downstreamFinalizeReason: number;
  dilationBrickRings: number;
  /** Interface/explicit-endpoint prefix captured before support dilation.
   * Section 5 consumers must not treat the wider redistance allocation halo
   * as interface topology. */
  interfaceSeedBricks: number;
}

export interface FineLevelSetTopologyBand {
  /** Conservative complete-trajectory displacement bound, in fine cells. */
  maximumBacktraceFineCells: number;
  /** Fine-cell radius needed by phi/velocity interpolation. */
  interpolationSupportFineCells: number;
  /** Physical signed-distance width that redistance must make valid. */
  redistanceBandFineCells: number;
  /** Whole-brick publication guard required by Section 18.6. */
  safetyBrickRings?: number;
}

/** Bootstrap is the only authority permitted to discover/dilate the complete
 * catalog. Every published generation after it must consume the transport
 * producer's compact phase-mask delta; absence or invalidity rejects the
 * candidate generation instead of selecting the old reconstruction path. */
export type FineLevelSetTopologyPublication =
  | { readonly kind: "bootstrap" }
  | { readonly kind: "delta"; readonly producer: FineLevelSetTransportTopologyDelta };

export interface FineLevelSetTopologyBandPlan extends Required<FineLevelSetTopologyBand> {
  readonly requiredFineCells: number;
  readonly dilationBrickRings: number;
}

export interface FineLevelSetLeafBrickBounds {
  readonly first: readonly [number, number, number];
  readonly last: readonly [number, number, number];
  readonly bricksPerFinestCell: number;
  readonly brickCount: number;
}

/**
 * CPU mirror of the FineSeedLeaf -> global fine-page mapping used by the seed
 * shader.  With B4 pages, factor four maps a finest cell to one page while
 * factor eight maps it to the complete 2 x 2 x 2 page block.  Keeping this
 * mapping explicit prevents octree row IDs or the one-page factor-4 shortcut
 * from leaking into factor-8 topology publication.
 */
export function planFineLevelSetLeafBrickBounds(
  plan: Pick<WebGPUFineLevelSetBrickSource["plan"],
  "fineFactor" | "brickResolution" | "finestCellDimensions" | "brickDimensions">,
  origin: readonly [number, number, number],
  size: number,
): FineLevelSetLeafBrickBounds {
  if (plan.brickResolution !== 4 || (plan.fineFactor !== 4 && plan.fineFactor !== 8)) {
    throw new RangeError("Fine leaf mapping requires a factor-4/factor-8 B4 lattice");
  }
  if (!Number.isSafeInteger(size) || size < 1
    || origin.some((value, axis) => !Number.isSafeInteger(value) || value < 0
      || value + size > plan.finestCellDimensions[axis])) {
    throw new RangeError("Fine leaf mapping is outside the finest-cell domain");
  }
  const bricksPerFinestCell = plan.fineFactor / plan.brickResolution;
  const first = origin.map((value) => value * bricksPerFinestCell) as [number, number, number];
  const last = origin.map((value, axis) => Math.min(plan.brickDimensions[axis] - 1,
    (value + size) * bricksPerFinestCell - 1)) as [number, number, number];
  const brickCount = (last[0] - first[0] + 1) * (last[1] - first[1] + 1) * (last[2] - first[2] + 1);
  return { first, last, bricksPerFinestCell, brickCount };
}

/** Converts the Section 5 physical support requirements to block rings.
 *
 * The transported interface is the common origin of both requirements. A
 * future departure query needs backtrace plus interpolation support, while
 * redistance needs its authored output width. Those are alternative radii,
 * not consecutive legs of one trajectory, so summing all three turns an
 * area-scaled narrow band into a domain-filling volume. The paper's explicit
 * interface-block one-ring is then added in whole blocks. */
export function planFineLevelSetTopologyBand(
  brickResolution: number,
  band: FineLevelSetTopologyBand,
): FineLevelSetTopologyBandPlan {
  if (!Number.isSafeInteger(brickResolution) || brickResolution < 1) {
    throw new RangeError("Fine topology brick resolution must be a positive integer");
  }
  const safetyBrickRings = band.safetyBrickRings ?? 1;
  for (const [label, value] of [
    ["maximum backtrace", band.maximumBacktraceFineCells],
    ["interpolation support", band.interpolationSupportFineCells],
    ["redistance band", band.redistanceBandFineCells],
    ["safety brick rings", safetyBrickRings],
  ] as const) {
    if (!Number.isSafeInteger(value) || value < 0) {
      throw new RangeError(`Fine topology ${label} must be a non-negative integer`);
    }
  }
  if (safetyBrickRings < 1) {
    throw new RangeError("Fine topology requires at least one publication safety ring");
  }
  const requiredFineCells = Math.max(
    band.maximumBacktraceFineCells + band.interpolationSupportFineCells,
    band.redistanceBandFineCells,
  );
  return { ...band, safetyBrickRings, requiredFineCells,
    dilationBrickRings: Math.ceil(requiredFineCells / brickResolution) + safetyBrickRings };
}

export interface FineLevelSetGPUSeedSource { readonly buffer: GPUBuffer; readonly affineValues?: boolean; }

export interface FineLevelSetPowerBoundarySeedSource {
  /** Two vec4f values per face: exact liquid and absent-air power-cell centres. */
  readonly queries: GPUBufferBinding;
  /** Published WebGPUOctreePowerFaces control record. */
  readonly control: GPUBufferBinding;
}

function powerOfTwoCapacity(value: number): number {
  let capacity = 1;
  while (capacity < value) capacity *= 2;
  return capacity;
}

export function fineLevelSetLeafSeedAllocatedBytes(maximumResidentBricks: number): number {
  if (!Number.isSafeInteger(maximumResidentBricks) || maximumResidentBricks < 1) {
    throw new RangeError("Fine seed resident capacity must be positive");
  }
  const sortCapacity = powerOfTwoCapacity(Math.max(256, 2 * maximumResidentBricks));
  // Header + sorted keys/tags/planes, parameters, one four-word record arena,
  // one bounded block-prefix arena, and eight control words. The former
  // ping-pong radix records and 256-bin histograms are deliberately absent.
  return (4 + 10 * maximumResidentBricks) * 4 + 64 + (5 * sortCapacity + 8) * 4;
}

const FINE_LEVELSET_TOPOLOGY_INDIRECT_BYTES = 8 * 12;
export const FINE_LEVELSET_TOPOLOGY_ALLOCATED_BYTES =
  48 + 96 + 8 + 64 + 32 + FINE_LEVELSET_TOPOLOGY_INDIRECT_BYTES;

/** GPU ABI for one exact fine-page topology/phase delta.
 *
 * Header words:
 *  0 exact changed-key count, 1 generation, 2 dirty-output count,
 *  3 JFA-support count, 4..9 reserved, 10 flags, 11 added, 12 retired, 13 reserved,
 *  14 rollback-page count, 15 assignment-valid.
 *
 * The payload contains compact sorted publications plus fixed-cardinality
 * candidate records. Dispatch-ordered stable rank/scan compaction replaces
 * append counters and generation-stamp writes.
 */
export interface FineLevelSetPageDeltaLayout {
  readonly headerWords: 16;
  readonly changedKeysOffsetWords: number;
  readonly dirtyPagesOffsetWords: number;
  readonly supportPagesOffsetWords: number;
  readonly desiredKeysOffsetWords: number;
  readonly addedPagesOffsetWords: number;
  readonly retiredPagesOffsetWords: number;
  readonly rollbackPagesOffsetWords: number;
  readonly changedCandidatesOffsetWords: number;
  readonly dirtyCandidatesOffsetWords: number;
  readonly supportCandidatesOffsetWords: number;
  /** Four streams of hierarchical scan scratch: additions, retirements,
   * free physical pages, and malformed-record validation. */
  readonly identityScanScratchOffsetWords: number;
  readonly identityScanBlockWords: number;
  readonly identityScanSuperBlockWords: number;
  /** Seven xyz records: phase, changed compact, init, affected classify,
   * affected compact, rollback, commit. */
  readonly lifecycleDispatchOffsetWords: number;
  readonly totalWords: number;
  readonly totalBytes: number;
}

export function planFineLevelSetPageDeltaLayout(pageCapacity: number): FineLevelSetPageDeltaLayout {
  if (!Number.isSafeInteger(pageCapacity) || pageCapacity < 1) {
    throw new RangeError("Fine page-delta capacity must be a positive integer");
  }
  const changedKeysOffsetWords = 16;
  const dirtyPagesOffsetWords = changedKeysOffsetWords + 2 * pageCapacity;
  const supportPagesOffsetWords = dirtyPagesOffsetWords + pageCapacity;
  const desiredKeysOffsetWords = supportPagesOffsetWords + pageCapacity;
  const addedPagesOffsetWords = desiredKeysOffsetWords + pageCapacity;
  const retiredPagesOffsetWords = addedPagesOffsetWords + pageCapacity;
  const rollbackPagesOffsetWords = retiredPagesOffsetWords + pageCapacity;
  const changedCandidatesOffsetWords = rollbackPagesOffsetWords + pageCapacity;
  const dirtyCandidatesOffsetWords = changedCandidatesOffsetWords + 2 * pageCapacity;
  const supportCandidatesOffsetWords = dirtyCandidatesOffsetWords + pageCapacity;
  const identityScanScratchOffsetWords = supportCandidatesOffsetWords + pageCapacity;
  const identityScanBlockWords = Math.ceil(pageCapacity / 256);
  const identityScanSuperBlockWords = Math.ceil(identityScanBlockWords / 256);
  const lifecycleDispatchOffsetWords = identityScanScratchOffsetWords
    + 4 * (identityScanBlockWords + identityScanSuperBlockWords);
  const totalWords = lifecycleDispatchOffsetWords + 21;
  return { headerWords: 16, changedKeysOffsetWords, dirtyPagesOffsetWords,
    supportPagesOffsetWords, desiredKeysOffsetWords, addedPagesOffsetWords,
    retiredPagesOffsetWords, rollbackPagesOffsetWords, changedCandidatesOffsetWords,
    dirtyCandidatesOffsetWords, supportCandidatesOffsetWords, identityScanScratchOffsetWords,
    identityScanBlockWords, identityScanSuperBlockWords, lifecycleDispatchOffsetWords,
    totalWords, totalBytes: totalWords * 4 };
}

/** GPU-only bridge from existing compact FineSeedLeaf/core candidates to global brick keys. */
export class WebGPUFineLevelSetLeafSeeds {
  readonly buffer: GPUBuffer;
  readonly allocatedBytes: number;
  private readonly params: GPUBuffer;
  private readonly scratch: GPUBuffer;
  private readonly sortCapacity: number;
  private readonly pipelines: Record<string, GPUComputePipeline>;

  constructor(private readonly device: GPUDevice, readonly target: WebGPUFineLevelSetBrickSource,
    analytic?: { initialCondition: "dam-break" | "tank-fill"; fillFraction: number }) {
    this.sortCapacity = powerOfTwoCapacity(Math.max(256, 2 * target.plan.maximumResidentBricks));
    this.allocatedBytes = fineLevelSetLeafSeedAllocatedBytes(target.plan.maximumResidentBricks);
    this.buffer = device.createBuffer({ label: "global fine brick seed keys",
      size: (4 + 10 * target.plan.maximumResidentBricks) * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const descriptor = new ArrayBuffer(8); const descriptorWords = new Uint32Array(descriptor);
    descriptorWords[0] = analytic?.initialCondition === "tank-fill" ? 1
      : analytic?.initialCondition === "dam-break" ? 2 : 0;
    new Float32Array(descriptor)[1] = analytic?.fillFraction ?? 0;
    device.queue.writeBuffer(this.buffer, 8, descriptor);
    this.params = device.createBuffer({ label: "global fine seed parameters", size: 64,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.scratch = device.createBuffer({ label: "global fine deterministic seed transaction",
      size: (5 * this.sortCapacity + 8) * 4, usage: GPUBufferUsage.STORAGE });
    const shaderModule = device.createShaderModule({ label: "FineSeedLeaf to global fine seeds", code: fineLevelSetLeafSeedWGSL });
    const entryPoints = ["clearSeedState", "classifySourceBlocks", "classifyEndpointBlocks",
      "scanSourceBlocks", "emitSourceRecords", "emitEndpointRecords",
      "sortSeedRecords",
      "classifySeedRuns", "scanSeedRuns", "emitSeedRuns"];
    this.pipelines = Object.fromEntries(entryPoints.map((entryPoint) => [entryPoint,
      device.createComputePipeline({ label: `Global fine seed ${entryPoint}`, layout: "auto",
        compute: { module: shaderModule, entryPoint } })]));
  }

  private bindingBytes(binding: GPUBufferBinding): number {
    return binding.size ?? binding.buffer.size - (binding.offset ?? 0);
  }

  private writeParams(leafCapacity: number, endpointCapacity: number,
    candidateCapacity: number, candidatesOnly: boolean): { sourceGroups: number; endpointGroups: number } {
    const plan = this.target.plan;
    const sourceCapacity = candidatesOnly ? candidateCapacity : leafCapacity;
    const sourceGroups = Math.max(1, Math.ceil(sourceCapacity / 64));
    const endpointGroups = Math.max(1, Math.ceil(endpointCapacity / 64));
    const scannedGroups = sourceGroups + (endpointCapacity > 0 ? endpointGroups : 0);
    if (scannedGroups > this.sortCapacity) {
      throw new RangeError(`Fine seed scan requires ${scannedGroups} block words, capacity is ${this.sortCapacity}`);
    }
    const bytes = new ArrayBuffer(64); const u32 = new Uint32Array(bytes); const f32 = new Float32Array(bytes);
    u32.set([plan.fineFactor, plan.brickResolution, ...plan.brickDimensions,
      plan.maximumResidentBricks, plan.logicalBrickCount, this.sortCapacity]);
    f32.set([...plan.domainOrigin, plan.fineCellWidth], 8);
    u32.set([sourceCapacity, candidatesOnly ? 1 : 0, endpointCapacity, candidateCapacity], 12);
    this.device.queue.writeBuffer(this.params, 0, bytes);
    return { sourceGroups, endpointGroups };
  }

  private group(entryPoint: string, entries: GPUBindGroupEntry[]): GPUBindGroup {
    const bindings: Record<string, readonly number[]> = {
      clearSeedState: [0, 4, 7],
      classifySourceBlocks: [0, 1, 2, 3, 7],
      classifyEndpointBlocks: [0, 5, 6, 7],
      scanSourceBlocks: [0, 4, 7],
      emitSourceRecords: [0, 1, 2, 3, 7],
      emitEndpointRecords: [0, 5, 6, 7],
      sortSeedRecords: [0, 7],
      classifySeedRuns: [0, 7],
      scanSeedRuns: [0, 4, 7],
      emitSeedRuns: [0, 1, 4, 7],
    };
    const used = new Set(bindings[entryPoint]);
    return this.device.createBindGroup({ layout: this.pipelines[entryPoint].getBindGroupLayout(0),
      entries: entries.filter((entry) => used.has(entry.binding)) });
  }

  private run(pass: GPUComputePassEncoder, entryPoint: string, workgroups: number,
    entries: GPUBindGroupEntry[]): void {
    pass.setPipeline(this.pipelines[entryPoint]); pass.setBindGroup(0, this.group(entryPoint, entries));
    pass.dispatchWorkgroups(workgroups);
  }

  private encodeRecords(pass: GPUComputePassEncoder, entries: GPUBindGroupEntry[],
    sourceGroups: number, endpointGroups: number, withEndpoints: boolean): void {
    this.run(pass, "clearSeedState", this.sortCapacity / 64, entries);
    this.run(pass, "classifySourceBlocks", sourceGroups, entries);
    if (withEndpoints) {
      this.run(pass, "classifyEndpointBlocks", endpointGroups, entries);
    }
    this.run(pass, "scanSourceBlocks", 1, entries);
    this.run(pass, "emitSourceRecords", sourceGroups, entries);
    if (withEndpoints) {
      this.run(pass, "emitEndpointRecords", endpointGroups, entries);
    }
    this.run(pass, "sortSeedRecords", 1, entries);
    this.run(pass, "classifySeedRuns", this.sortCapacity / 64, entries);
    this.run(pass, "scanSeedRuns", 1, entries);
    this.run(pass, "emitSeedRuns", this.sortCapacity / 64, entries);
  }

  encode(broker: PassBroker, leaves: GPUBufferBinding, candidates: GPUBufferBinding,
    candidateCountAndDispatch: GPUBufferBinding): FineLevelSetGPUSeedSource {
    const leafCapacity = Math.floor(this.bindingBytes(leaves) / 64);
    const candidateCapacity = Math.floor(this.bindingBytes(candidates) / 8);
    const groups = this.writeParams(leafCapacity, 0, candidateCapacity, true);
    const entries: GPUBindGroupEntry[] = [
      { binding: 0, resource: { buffer: this.params } }, { binding: 1, resource: leaves },
      { binding: 2, resource: candidates }, { binding: 3, resource: candidateCountAndDispatch },
      { binding: 4, resource: { buffer: this.buffer } }, { binding: 7, resource: { buffer: this.scratch } },
    ];
    const pass = broker.compute({ label: "Seed global fine bricks from FineSeedLeaf candidates" });
    this.encodeRecords(pass, entries, groups.sourceGroups, groups.endpointGroups, false);
    return { buffer: this.buffer, affineValues: true };
  }

  encodeFromAllInterfaceLeaves(broker: PassBroker, leaves: GPUBufferBinding,
    rowCount: GPUBufferBinding, powerBoundary?: FineLevelSetPowerBoundarySeedSource): FineLevelSetGPUSeedSource {
    const leafCapacity = Math.floor(this.bindingBytes(leaves) / 64);
    const endpointCapacity = powerBoundary ? Math.floor(this.bindingBytes(powerBoundary.queries) / 32) : 0;
    const groups = this.writeParams(leafCapacity, endpointCapacity, 0, false);
    const entries: GPUBindGroupEntry[] = [
      { binding: 0, resource: { buffer: this.params } }, { binding: 1, resource: leaves },
      { binding: 2, resource: rowCount }, { binding: 3, resource: rowCount },
      { binding: 4, resource: { buffer: this.buffer } },
      { binding: 7, resource: { buffer: this.scratch } },
    ];
    if (powerBoundary) entries.push(
      { binding: 5, resource: powerBoundary.queries },
      { binding: 6, resource: powerBoundary.control },
    );
    const pass = broker.compute({ label: "Seed global fine bricks from every interface leaf" });
    this.encodeRecords(pass, entries, groups.sourceGroups, groups.endpointGroups, Boolean(powerBoundary));
    return { buffer: this.buffer, affineValues: true };
  }

  destroy(): void { this.buffer.destroy(); this.params.destroy(); this.scratch.destroy(); }
}

export function unpackFineLevelSetGPUTopologyControl(words: ArrayLike<number>): FineLevelSetGPUTopologyControl {
  if (words.length < 9) throw new RangeError("Fine topology control requires nine words");
  return { flags: Number(words[0]) >>> 0, interfaceBricks: Number(words[1]) >>> 0,
    desiredBricks: Number(words[2]) >>> 0, activatedBricks: Number(words[3]) >>> 0,
    published: Number(words[4]) !== 0, rolledBack: Number(words[5]) !== 0,
    downstreamFinalizeReason: Number(words[7]) >>> 0,
    requiredDesiredBricks: (Number(words[0]) & FINE_LEVELSET_TOPOLOGY_ERROR.capacity) !== 0
      ? Number(words[6]) >>> 0 : Number(words[2]) >>> 0,
    requiredDesiredBricksExact: (Number(words[0]) & FINE_LEVELSET_TOPOLOGY_ERROR.capacity) === 0,
    dilationBrickRings: Number(words[0]) === 0 ? Number(words[6]) >>> 0 : 0,
    interfaceSeedBricks: Number(words[8]) >>> 0 };
}

/**
 * GPU discovery and deterministic next-generation publication. The injected source
 * must define `fn sampleCoarseOctreePhi(position:vec3f)->f32`; it may use
 * textures/uniforms or additional bindings beginning at binding 8.
 */
export class WebGPUFineLevelSetTopology {
  readonly control: GPUBuffer;
  /** Exact changed keys, dirty output pages, and the required JFA support halo. */
  readonly pageDelta: GPUBuffer;
  /** GPU-authored dispatch records published at explicit storage/indirect boundaries. */
  private readonly indirectDispatch: GPUBuffer;
  /** Exact one-workgroup-per-page commands consumed directly by redistance. */
  readonly redistanceDispatches: {
    readonly buffer: GPUBuffer;
    readonly dirtyOffsetBytes: 84;
    readonly supportOffsetBytes: 60;
  };
  readonly pageDeltaLayout: FineLevelSetPageDeltaLayout;
  /** Rejection-only visibility into the immutable topology invocation parameters. */
  get debugParameterBuffer(): GPUBuffer { return this.params; }
  /** Rejection-only visibility into the sparse recurring candidate records. */
  get debugSparseCandidateBuffer(): GPUBuffer { return this.sparseCandidates; }
  readonly sparseCandidateCapacity: number;
  readonly allocatedBytes: number;
  private readonly params: GPUBuffer;
  private readonly clearPipeline: GPUComputePipeline;
  private readonly discoverPipeline: GPUComputePipeline;
  private readonly externalSeedPipeline: GPUComputePipeline;
  private readonly clearTopologyErrorsPipeline: GPUComputePipeline;
  private readonly reduceTopologyErrorRecordsPipeline: GPUComputePipeline;
  private readonly reduceTopologyErrorGroupsPipeline: GPUComputePipeline;
  private readonly reduceTopologyErrorSuperGroupsPipeline: GPUComputePipeline;
  private readonly validateDesiredSeedsPipeline: GPUComputePipeline;
  private readonly scanSparseSeedRecordsPipeline: GPUComputePipeline;
  private readonly scanSparseCandidateRecordsPipeline: GPUComputePipeline;
  private readonly scanSparseGroupsPipeline: GPUComputePipeline;
  private readonly scanSparseSuperGroupsPipeline: GPUComputePipeline;
  private readonly offsetSparseGroupsPipeline: GPUComputePipeline;
  private readonly offsetSparseRecordsPipeline: GPUComputePipeline;
  private readonly finalizeDesiredSeedCountPipeline: GPUComputePipeline;
  private readonly compactSparseSeedsPipeline: GPUComputePipeline;
  private readonly clearSparseCandidatesPipeline: GPUComputePipeline;
  private readonly sortSparseCandidatesPipeline: GPUComputePipeline;
  private readonly expandSparseDesiredPipelines: readonly GPUComputePipeline[];
  private readonly compactSparseDesiredPipeline: GPUComputePipeline;
  private readonly publishDesiredBricksPipeline: GPUComputePipeline;
  private readonly publishRecurringSparseBandPipeline: GPUComputePipeline;
  private readonly clearPageDirectoryPipeline: GPUComputePipeline;
  private readonly snapshotPipeline: GPUComputePipeline;
  private readonly classifyIdentityPipeline: GPUComputePipeline;
  private readonly scanIdentityRecordsPipeline: GPUComputePipeline;
  private readonly scanIdentityGroupsPipeline: GPUComputePipeline;
  private readonly scanIdentitySuperGroupsPipeline: GPUComputePipeline;
  private readonly offsetIdentityGroupsPipeline: GPUComputePipeline;
  private readonly offsetIdentityRecordsPipeline: GPUComputePipeline;
  private readonly prepareIdentityPipeline: GPUComputePipeline;
  private readonly compactIdentityPipeline: GPUComputePipeline;
  private readonly assignIdentityPipeline: GPUComputePipeline;
  private readonly finalizeIdentityPipeline: GPUComputePipeline;
  private readonly initializePipeline: GPUComputePipeline;
  private readonly linkPipeline: GPUComputePipeline;
  private readonly finalizePipeline: GPUComputePipeline;
  private readonly clearPageDeltaPipeline: GPUComputePipeline;
  private readonly classifyPageDeltaPipeline: GPUComputePipeline;
  private readonly compactChangedKeysPipeline: GPUComputePipeline;
  private readonly preparePageDeltaExpansionPipeline: GPUComputePipeline;
  private readonly classifyAffectedPagesPipeline: GPUComputePipeline;
  private readonly compactAffectedPagesPipeline: GPUComputePipeline;
  private readonly finalizePageDeltaPipeline: GPUComputePipeline;
  private readonly publishSummaryChangedKeysPipeline: GPUComputePipeline;
  private readonly publicationPipeline: GPUComputePipeline;
  private readonly settlePublicationPipeline: GPUComputePipeline;
  private readonly emptySeeds: GPUBuffer;
  private readonly validVolumeControl: GPUBuffer;
  private readonly validTransportControl: GPUBuffer;
  private readonly desiredCandidates: GPUBuffer;
  private readonly sparseCandidates: GPUBuffer;
  private readonly desiredScan: GPUBuffer;
  private readonly topologyErrors: GPUBuffer;
  /** Transported phi must survive physical-page reassignment until the dirty
   * generation has either committed or rolled back. */
  private readonly transportedPhiSnapshot: GPUBuffer;

  constructor(
    private readonly device: GPUDevice,
    readonly current: WebGPUFineLevelSetBrickSource,
    readonly next: WebGPUFineLevelSetBrickSource,
    coarsePhiWGSL: string,
  ) {
    if (current.plan !== next.plan && JSON.stringify(current.plan) !== JSON.stringify(next.plan)) {
      throw new RangeError("Fine topology generations must use the same configured lattice");
    }
    if (next.generation === current.generation) {
      throw new RangeError("Fine topology generations must be distinct");
    }
    if (current.flags !== next.flags || current.phi !== next.phi
      || current.workA !== next.workA || current.workB !== next.workB) {
      throw new RangeError("Fine topology generations must share one A/B payload pool");
    }
    if (current.rollbackPhi !== next.rollbackPhi
      || current.rollbackPhi === current.phi || current.rollbackPhi === current.flags
      || current.rollbackPhi === current.workA || current.rollbackPhi === current.workB) {
      throw new RangeError("Fine topology rollback phi must be one shared, dedicated A/B transaction buffer");
    }
    if (!/fn\s+sampleCoarseOctreePhi\s*\(/.test(coarsePhiWGSL)) {
      throw new RangeError("Fine topology requires sampleCoarseOctreePhi");
    }
    this.control = device.createBuffer({ label: "fine-levelset topology control", size: 48,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    this.pageDeltaLayout = planFineLevelSetPageDeltaLayout(current.plan.maximumResidentBricks);
    this.pageDelta = device.createBuffer({ label: "fine-levelset exact page delta",
      size: this.pageDeltaLayout.totalBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    this.indirectDispatch = device.createBuffer({ label: "fine-levelset immutable indirect dispatch",
      size: FINE_LEVELSET_TOPOLOGY_INDIRECT_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT });
    this.redistanceDispatches = {
      buffer: this.indirectDispatch,
      dirtyOffsetBytes: 84,
      supportOffsetBytes: 60,
    };
    const sampleBytes = current.plan.maximumResidentBricks * current.plan.samplesPerBrick * 4;
    // One axis expansion emits the negative, centre, and positive neighbor of
    // every resident key. A power-of-two arena permits an in-place,
    // synchronization-free bitonic sort before stable run compaction.
    this.sparseCandidateCapacity = powerOfTwoCapacity(
      Math.max(256, 3 * current.plan.maximumResidentBricks),
    );
    const desiredCandidateWords = 2 * current.plan.maximumResidentBricks;
    const desiredScanBlocks = Math.ceil(this.sparseCandidateCapacity / 256);
    const desiredScanSuperBlocks = Math.ceil(desiredScanBlocks / 256);
    const topologyErrorBlocks = Math.ceil(current.plan.maximumResidentBricks / 256);
    const topologyErrorSuperBlocks = Math.ceil(topologyErrorBlocks / 256);
    const topologyScratchWords = Math.max(
      current.plan.maximumResidentBricks,
      current.plan.logicalBrickCount,
    );
    const desiredScanWords = Math.max(
      this.sparseCandidateCapacity + desiredScanBlocks + desiredScanSuperBlocks,
      2 * (topologyErrorBlocks + topologyErrorSuperBlocks),
    );
    this.desiredCandidates = device.createBuffer({ label: "fine-levelset deterministic desired candidates",
      size: desiredCandidateWords * 4, usage: GPUBufferUsage.STORAGE });
    this.sparseCandidates = device.createBuffer({ label: "fine-levelset sorted topology expansion",
      size: (this.sparseCandidateCapacity + current.plan.maximumResidentBricks) * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    this.desiredScan = device.createBuffer({ label: "fine-levelset sparse topology scan",
      size: desiredScanWords * 4, usage: GPUBufferUsage.STORAGE });
    this.topologyErrors = device.createBuffer({
      label: "fine-levelset direct identity mask and fixed topology error records",
      size: topologyScratchWords * 4, usage: GPUBufferUsage.STORAGE,
    });
    this.transportedPhiSnapshot = device.createBuffer({ label: "fine-levelset transported phi transaction",
      size: sampleBytes, usage: GPUBufferUsage.STORAGE });
    this.allocatedBytes = FINE_LEVELSET_TOPOLOGY_ALLOCATED_BYTES
      + this.pageDeltaLayout.totalBytes + sampleBytes
      + (desiredCandidateWords + desiredScanWords + this.sparseCandidateCapacity
        + current.plan.maximumResidentBricks + topologyScratchWords) * 4;
    this.params = device.createBuffer({ label: "fine-levelset topology params", size: 96,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    this.emptySeeds = device.createBuffer({ label: "empty global fine seeds", size: 8,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.validVolumeControl = device.createBuffer({ label: "fine publication valid-volume fallback", size: 64,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(this.validVolumeControl, 0, new Uint32Array([0x8000_0000]));
    this.validTransportControl = device.createBuffer({ label: "fine publication valid-transport fallback", size: 32,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(this.validTransportControl, 12, new Uint32Array([1]));
    const shaderModule = device.createShaderModule({ label: "fine-levelset GPU topology",
      code: makeFineLevelSetTopologyWGSL(coarsePhiWGSL) });
    const pipeline = (label: string, entryPoint: string) => device.createComputePipeline({ label, layout: "auto",
      compute: { module: shaderModule, entryPoint } });
    this.clearPipeline = pipeline("Clear fine topology candidate generation", "clearDesiredGeneration");
    this.discoverPipeline = pipeline("Discover fine interface bricks", "discoverInterfaceBricks");
    this.externalSeedPipeline = pipeline("Insert external fine topology seeds", "insertExternalSeeds");
    this.clearTopologyErrorsPipeline = pipeline("Clear fixed fine topology error records", "clearTopologyErrors");
    this.reduceTopologyErrorRecordsPipeline = pipeline("Reduce fixed fine topology error records", "reduceTopologyErrorRecords");
    this.reduceTopologyErrorGroupsPipeline = pipeline("Reduce fixed fine topology error groups", "reduceTopologyErrorGroups");
    this.reduceTopologyErrorSuperGroupsPipeline = pipeline("Reduce fixed fine topology error super-groups", "reduceTopologyErrorSuperGroups");
    this.validateDesiredSeedsPipeline = pipeline("Validate deterministic fine topology seeds", "validateDesiredSeeds");
    this.scanSparseSeedRecordsPipeline = pipeline("Scan compact fine seed records", "scanSparseSeedRecords");
    this.scanSparseCandidateRecordsPipeline = pipeline("Scan sparse fine topology candidates", "scanSparseCandidateRecords");
    this.scanSparseGroupsPipeline = pipeline("Scan sparse fine topology groups", "scanSparseGroups");
    this.scanSparseSuperGroupsPipeline = pipeline("Scan sparse fine topology super-groups", "scanSparseSuperGroups");
    this.offsetSparseGroupsPipeline = pipeline("Offset sparse fine topology groups", "offsetSparseGroups");
    this.offsetSparseRecordsPipeline = pipeline("Offset sparse fine topology records", "offsetSparseRecords");
    this.finalizeDesiredSeedCountPipeline = pipeline("Finalize deterministic fine seed count", "finalizeDesiredSeedCount");
    this.compactSparseSeedsPipeline = pipeline("Compact deterministic fine seeds", "compactSparseSeeds");
    this.clearSparseCandidatesPipeline = pipeline("Clear sorted fine topology expansion", "clearSparseCandidates");
    this.sortSparseCandidatesPipeline = pipeline("Sort sparse fine topology expansion", "sortSparseCandidates");
    this.expandSparseDesiredPipelines = ["X", "Y", "Z"].map((axis) =>
      pipeline(`Expand sparse fine topology ${axis}`, `expandSparseDesired${axis}`));
    this.compactSparseDesiredPipeline = pipeline("Compact sparse fine topology candidates", "compactSparseDesiredBricks");
    this.publishDesiredBricksPipeline = pipeline("Publish sorted sparse fine topology", "publishDesiredBricks");
    this.publishRecurringSparseBandPipeline = pipeline(
      "Publish compact recurring fine topology band", "publishRecurringSparseBand");
    this.clearPageDirectoryPipeline = pipeline(
      "Clear direct fine page directory", "clearFinePageDirectory");
    this.snapshotPipeline = pipeline("Snapshot exact fine topology rollback pages", "snapshotDeltaPayload");
    this.classifyIdentityPipeline = pipeline("Classify exact fine identity records", "classifyDesiredPageIdentities");
    this.scanIdentityRecordsPipeline = pipeline("Scan exact fine identity records", "scanIdentityRecords");
    this.scanIdentityGroupsPipeline = pipeline("Scan exact fine identity groups", "scanIdentityGroups");
    this.scanIdentitySuperGroupsPipeline = pipeline("Scan exact fine identity super-groups", "scanIdentitySuperGroups");
    this.offsetIdentityGroupsPipeline = pipeline("Offset exact fine identity groups", "offsetIdentityGroups");
    this.offsetIdentityRecordsPipeline = pipeline("Offset exact fine identity records", "offsetIdentityRecords");
    this.prepareIdentityPipeline = pipeline("Prepare exact fine identity assignment", "prepareDesiredPageIdentityAssignment");
    this.compactIdentityPipeline = pipeline("Compact exact fine identity records", "compactDesiredPageIdentities");
    this.assignIdentityPipeline = pipeline("Assign exact fine page identities", "assignDesiredPageIdentities");
    this.finalizeIdentityPipeline = pipeline("Finalize exact fine identity assignment", "finalizeDesiredPageIdentityAssignment");
    this.initializePipeline = pipeline("Initialize next fine topology samples", "initializeDesiredSamples");
    this.linkPipeline = pipeline("Link next fine topology neighbors", "linkDesiredNeighbors");
    this.finalizePipeline = pipeline("Finalize next fine topology generation", "finalizeDesiredGeneration");
    this.clearPageDeltaPipeline = pipeline("Clear exact fine page delta", "clearFinePageDelta");
    this.classifyPageDeltaPipeline = pipeline("Classify exact fine page delta", "classifyFinePageDelta");
    this.compactChangedKeysPipeline = pipeline("Compact sorted fine changed keys", "compactFineChangedKeys");
    this.preparePageDeltaExpansionPipeline = pipeline("Prepare exact fine page expansion", "prepareFinePageDeltaExpansion");
    this.classifyAffectedPagesPipeline = pipeline("Classify exact fine changed-key support",
      "classifyFineAffectedPages");
    this.compactAffectedPagesPipeline = pipeline("Compact exact fine affected pages",
      "compactFineAffectedPages");
    this.finalizePageDeltaPipeline = pipeline("Finalize exact fine page delta", "finalizeFinePageDelta");
    this.publishSummaryChangedKeysPipeline = pipeline(
      "Publish post-redistance fine summary keys", "publishFineSummaryChangedKeys");
    this.publicationPipeline = pipeline("Gate complete fine generation publication", "finalizeFinePublication");
    this.settlePublicationPipeline = pipeline("Settle accepted or rejected fine topology",
      "settleFinePublication");
  }

  encode(broker: PassBroker, seedSource?: FineLevelSetGPUSeedSource,
    extraPublishEntries: readonly GPUBindGroupEntry[] = [], band?: FineLevelSetTopologyBand,
    deferPublication = false,
    publication: FineLevelSetTopologyPublication = { kind: "bootstrap" }): void {
    const plan = this.current.plan;
    const bandPlan = planFineLevelSetTopologyBand(plan.brickResolution, band ?? {
      maximumBacktraceFineCells: 0,
      interpolationSupportFineCells: 0,
      redistanceBandFineCells: 0,
      safetyBrickRings: 1,
    });
    // Redistance consumes an immutable resident generation. Allocate the full
    // transport + interpolation + signed-distance support before JFA-CPT
    // starts; no distance pass is permitted to mutate page tables.
    const dilationBrickRings = bandPlan.dilationBrickRings;
    const bytes = new ArrayBuffer(96); const u32 = new Uint32Array(bytes); const f32 = new Float32Array(bytes);
    u32.set(plan.brickDimensions, 0); u32[3] = plan.brickResolution;
    u32.set(plan.sampleDimensions, 4); u32[7] = plan.samplesPerBrick;
    f32.set(plan.domainOrigin, 8); f32[11] = plan.fineCellWidth;
    u32.set([this.sparseCandidateCapacity, plan.maximumResidentBricks,
      this.current.generation, this.next.generation, plan.fineFactor,
      seedSource?.affineValues ? 1 : 0, dilationBrickRings,
      deferPublication ? 1 : 0], 12);
    // A residency or transported phase-mask change can alter exact signed
    // distance for every page within the authored band. JFA additionally
    // needs one equal-width seed/landing halo around those dirty outputs.
    const dirtyHaloRings = Math.ceil(
      (bandPlan.redistanceBandFineCells + 1) / plan.brickResolution,
    );
    u32[20] = dirtyHaloRings;
    u32[21] = 2 * dirtyHaloRings;
    u32[22] = this.device.limits.maxComputeWorkgroupsPerDimension;
    u32[23] = publication.kind === "delta" ? 1 : 0;
    this.device.queue.writeBuffer(this.params, 0, bytes);
    this.device.queue.writeBuffer(this.control, 0, new Uint32Array(8));
    const resource = (buffer: GPUBuffer) => ({ buffer });
    const discoverEntries: GPUBindGroupEntry[] = [
      { binding: 0, resource: resource(this.params) }, { binding: 1, resource: resource(this.current.metadata) },
      { binding: 2, resource: resource(this.current.worklist) }, { binding: 3, resource: resource(this.current.flags) },
      { binding: 4, resource: resource(this.current.phi) },
      { binding: 6, resource: resource(this.next.worklist) }, { binding: 7, resource: resource(this.control) },
      { binding: 8, resource: resource(seedSource?.buffer ?? this.emptySeeds) },
      { binding: 14, resource: resource(this.current.worklist) },
      { binding: 16, resource: resource(this.current.metadata) },
      { binding: 15, resource: resource(this.pageDelta) },
      { binding: 18, resource: resource(this.desiredCandidates) },
      { binding: 19, resource: resource(this.sparseCandidates) },
      { binding: 20, resource: resource(this.desiredScan) },
      { binding: 21, resource: resource(this.topologyErrors) },
      { binding: 22, resource: resource(this.indirectDispatch) },
      { binding: 23, resource: resource(publication.kind === "delta"
        ? publication.producer.buffer : this.emptySeeds) },
    ];
    const publishEntries: GPUBindGroupEntry[] = [
      { binding: 0, resource: resource(this.params) }, { binding: 3, resource: resource(this.next.metadata) },
      { binding: 4, resource: resource(this.next.worklist) }, { binding: 5, resource: resource(this.next.flags) },
      { binding: 6, resource: resource(this.next.phi) }, { binding: 7, resource: resource(this.control) },
      { binding: 8, resource: resource(seedSource?.buffer ?? this.emptySeeds) },
      { binding: 10, resource: resource(this.transportedPhiSnapshot) },
      { binding: 14, resource: resource(this.current.worklist) },
      { binding: 15, resource: resource(this.pageDelta) },
      { binding: 18, resource: resource(this.desiredCandidates) },
      { binding: 20, resource: resource(this.desiredScan) },
      { binding: 21, resource: resource(this.topologyErrors) },
      { binding: 22, resource: resource(this.indirectDispatch) },
      { binding: 23, resource: resource(publication.kind === "delta"
        ? publication.producer.buffer : this.emptySeeds) },
      ...extraPublishEntries,
    ];
    const deltaEntries: GPUBindGroupEntry[] = [
      { binding: 0, resource: resource(this.params) }, { binding: 3, resource: resource(this.next.metadata) },
      { binding: 4, resource: resource(this.next.worklist) }, { binding: 6, resource: resource(this.next.phi) },
      { binding: 7, resource: resource(this.control) },
      { binding: 10, resource: resource(this.transportedPhiSnapshot) },
      { binding: 14, resource: resource(this.current.worklist) },
      { binding: 15, resource: resource(this.pageDelta) },
      { binding: 16, resource: resource(this.current.metadata) },
      { binding: 17, resource: resource(this.current.rollbackPhi) },
      { binding: 18, resource: resource(this.desiredCandidates) },
      { binding: 19, resource: resource(this.sparseCandidates) },
      { binding: 20, resource: resource(this.desiredScan) },
      { binding: 21, resource: resource(this.topologyErrors) },
      { binding: 22, resource: resource(this.indirectDispatch) },
      { binding: 23, resource: resource(publication.kind === "delta"
        ? publication.producer.buffer : this.emptySeeds) },
    ];
    // Dispatch boundaries provide the storage-buffer ordering required by
    // deterministic seed classification, closure publication, and assignment. Keeping this
    // launch-bound chain in one compute pass avoids a driver pass transition
    // for every one-workgroup control stage on small domains.
    const run = (pipeline: GPUComputePipeline, entries: GPUBindGroupEntry[], _label: string, groups = 1,
      used?: readonly number[]) => {
      const selected = used ? entries.filter((entry) => used.includes(entry.binding)) : entries;
      const group = this.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: selected });
      const pass = broker.compute({ label: _label });
      pass.setPipeline(pipeline); pass.setBindGroup(0, group);
      const dispatch = planFineLevelSetDispatch2D(groups, this.device.limits.maxComputeWorkgroupsPerDimension);
      if (dispatch.workgroups > 0) pass.dispatchWorkgroups(dispatch.x, dispatch.y, dispatch.z);
    };
    const runIndirect = (pipeline: GPUComputePipeline, entries: GPUBindGroupEntry[], _label: string,
      dispatchWord: number, used?: readonly number[]) => {
      const selected = used ? entries.filter((entry) => used.includes(entry.binding)) : entries;
      const group = this.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: selected });
      const pass = broker.compute({ label: _label });
      pass.setPipeline(pipeline); pass.setBindGroup(0, group);
      pass.dispatchWorkgroupsIndirect(this.indirectDispatch, dispatchWord * 12);
    };
    const runIdentity = (pipeline: GPUComputePipeline, entries: GPUBindGroupEntry[],
      x: number, y: number, used: readonly number[]) => {
      const group = this.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: entries.filter((entry) => used.includes(entry.binding)),
      });
      const pass = broker.compute();
      pass.setPipeline(pipeline); pass.setBindGroup(0, group);
      pass.dispatchWorkgroups(x, y);
    };
    const clearItems = Math.max(5 + plan.maximumResidentBricks, 8);
    run(this.clearPageDeltaPipeline, deltaEntries, "Clear exact global fine page delta",
      Math.ceil(24 / 64), [0, 15, 22]);
    const topologyErrorBlocks = Math.ceil(plan.maximumResidentBricks / 256);
    const topologyErrorSuperBlocks = Math.ceil(topologyErrorBlocks / 256);
    if (publication.kind === "bootstrap") {
      run(this.clearPipeline, discoverEntries, "Clear cold global fine topology candidates",
        Math.ceil(clearItems / 64), [0, 6, 7]);
      run(this.clearTopologyErrorsPipeline, discoverEntries, "Clear cold global fine topology seed errors",
        Math.ceil(plan.maximumResidentBricks / 64), [0, 21]);
      run(this.discoverPipeline, discoverEntries, "Discover cold global fine interface bricks",
        Math.ceil(plan.maximumResidentBricks / 64), [0, 1, 2, 3, 4, 18, 21]);
      run(this.externalSeedPipeline, discoverEntries, "Insert cold global fine seed bricks",
        Math.ceil(plan.maximumResidentBricks / 64), [0, 8, 14, 18, 21]);
      runIdentity(this.reduceTopologyErrorRecordsPipeline, discoverEntries,
        topologyErrorBlocks, 1, [0, 20, 21]);
      runIdentity(this.reduceTopologyErrorGroupsPipeline, discoverEntries,
        topologyErrorSuperBlocks, 1, [0, 20]);
      runIdentity(this.reduceTopologyErrorSuperGroupsPipeline, discoverEntries,
        1, 1, [0, 20]);
      run(this.validateDesiredSeedsPipeline, discoverEntries,
        "Validate deterministic cold global fine seeds", 1, [7, 20]);
      const seedBlocks = Math.ceil((2 * plan.maximumResidentBricks) / 256);
      const seedSuperBlocks = Math.ceil(seedBlocks / 256);
      runIdentity(this.scanSparseSeedRecordsPipeline, discoverEntries, seedBlocks, 1, [0, 1, 2, 7, 18, 20]);
      runIdentity(this.scanSparseGroupsPipeline, discoverEntries, seedSuperBlocks, 1, [0, 7, 20]);
      runIdentity(this.scanSparseSuperGroupsPipeline, discoverEntries, 1, 1, [0, 7, 20]);
      runIdentity(this.offsetSparseGroupsPipeline, discoverEntries, seedSuperBlocks, 1, [0, 7, 20]);
      runIdentity(this.offsetSparseRecordsPipeline, discoverEntries, seedBlocks, 1, [0, 7, 20]);
      run(this.finalizeDesiredSeedCountPipeline, discoverEntries, "Finalize deterministic cold fine seed count",
        1, [0, 1, 2, 7, 18, 20]);
      run(this.clearSparseCandidatesPipeline, discoverEntries, "Clear cold fine seed expansion",
        Math.ceil(this.sparseCandidateCapacity / 64), [0, 19]);
      run(this.compactSparseSeedsPipeline, discoverEntries, "Compact deterministic cold fine seeds",
        Math.ceil((2 * plan.maximumResidentBricks) / 64), [0, 1, 2, 7, 18, 19, 20]);
      const desiredBlocks = Math.ceil(this.sparseCandidateCapacity / 256);
      const desiredSuperBlocks = Math.ceil(desiredBlocks / 256);
      const compactSortedExpansion = (label: string) => {
        runIdentity(this.sortSparseCandidatesPipeline, discoverEntries, 1, 1, [0, 19]);
        runIdentity(this.scanSparseCandidateRecordsPipeline, discoverEntries,
          desiredBlocks, 1, [0, 7, 19, 20]);
        runIdentity(this.scanSparseGroupsPipeline, discoverEntries,
          desiredSuperBlocks, 1, [0, 7, 20]);
        runIdentity(this.scanSparseSuperGroupsPipeline, discoverEntries, 1, 1, [0, 7, 20]);
        runIdentity(this.offsetSparseGroupsPipeline, discoverEntries,
          desiredSuperBlocks, 1, [0, 7, 20]);
        runIdentity(this.offsetSparseRecordsPipeline, discoverEntries,
          desiredBlocks, 1, [0, 7, 20]);
        run(this.compactSparseDesiredPipeline, discoverEntries, label,
          Math.ceil(this.sparseCandidateCapacity / 64), [0, 7, 19, 20]);
      };
      compactSortedExpansion("Compact sorted cold global fine seeds");
      for (let ring = 0; ring < dilationBrickRings; ring += 1) {
        for (let axis = 0; axis < this.expandSparseDesiredPipelines.length; axis += 1) {
          run(this.expandSparseDesiredPipelines[axis], discoverEntries,
            `Expand cold global fine topology ring ${ring + 1} axis ${axis}`,
            Math.ceil(this.sparseCandidateCapacity / 64), [0, 7, 19]);
          compactSortedExpansion(`Compact cold global fine topology ring ${ring + 1} axis ${axis}`);
        }
      }
      run(this.publishDesiredBricksPipeline, discoverEntries, "Publish sorted cold global fine topology",
        Math.ceil(plan.maximumResidentBricks / 64), [0, 6, 7, 19]);
    } else {
      if (publication.producer.pageCapacity !== plan.maximumResidentBricks) {
        throw new RangeError("Fine recurring topology producer capacity does not match the destination lattice");
      }
      // One persistent workgroup consumes the compact interface-membership
      // delta and explicit endpoint seeds, then performs deterministic sparse
      // axis closure in-place. This is one dispatch regardless of ring count:
      // the retired scalar resident/cubic search and the tempting
      // multi-dispatch bootstrap reuse are both absent from recurring work.
      runIdentity(this.publishRecurringSparseBandPipeline, discoverEntries,
        1, 1, [0, 6, 7, 8, 14, 16, 19, 21, 23]);
    }
    run(this.clearPageDirectoryPipeline, deltaEntries, "Clear direct global fine page directory",
      Math.ceil(plan.logicalBrickCount / 64), [0, 4]);
    if (publication.kind === "bootstrap") {
      run(this.clearTopologyErrorsPipeline, deltaEntries, "Clear fixed global fine lifecycle errors",
        Math.ceil(plan.maximumResidentBricks / 64), [0, 21]);
    }
    const identityBlocks = this.pageDeltaLayout.identityScanBlockWords;
    const identitySuperBlocks = this.pageDeltaLayout.identityScanSuperBlockWords;
    runIdentity(this.classifyIdentityPipeline, deltaEntries,
      Math.ceil(plan.maximumResidentBricks / 64), 1, [0, 1, 2, 4, 7, 14, 15, 16]);
    runIdentity(this.scanIdentityRecordsPipeline, deltaEntries, identityBlocks, 4, [0, 7, 15, 18]);
    runIdentity(this.scanIdentityGroupsPipeline, deltaEntries, identitySuperBlocks, 4, [0, 15]);
    runIdentity(this.scanIdentitySuperGroupsPipeline, deltaEntries, 1, 4, [0, 15]);
    runIdentity(this.offsetIdentityGroupsPipeline, deltaEntries, identitySuperBlocks, 4, [0, 15]);
    runIdentity(this.offsetIdentityRecordsPipeline, deltaEntries, identityBlocks, 4, [0, 15, 18]);
    runIdentity(this.prepareIdentityPipeline, deltaEntries, 1, 1, [0, 7, 14, 15, 18]);
    runIdentity(this.compactIdentityPipeline, deltaEntries,
      Math.ceil(plan.maximumResidentBricks / 64), 1, [0, 7, 15, 16, 18]);
    runIdentity(this.assignIdentityPipeline, deltaEntries,
      Math.ceil(plan.maximumResidentBricks / 64), 1, [0, 3, 4, 7, 14, 15, 16, 18]);
    runIdentity(this.finalizeIdentityPipeline, deltaEntries, 1, 1, [0, 4, 7, 15, 22]);
    broker.fence("fine identity dispatch publication");
    runIndirect(this.classifyPageDeltaPipeline, deltaEntries, "Classify exact global fine page delta",
      0, [0, 3, 4, 7, 15, 16, 21, 23]);
    runIdentity(this.scanIdentityRecordsPipeline, deltaEntries, identityBlocks, 1, [0, 7, 15, 18]);
    runIdentity(this.scanIdentityGroupsPipeline, deltaEntries, identitySuperBlocks, 1, [0, 15]);
    runIdentity(this.scanIdentitySuperGroupsPipeline, deltaEntries, 1, 1, [0, 15]);
    runIdentity(this.offsetIdentityGroupsPipeline, deltaEntries, identitySuperBlocks, 1, [0, 15]);
    runIdentity(this.offsetIdentityRecordsPipeline, deltaEntries, identityBlocks, 1, [0, 15, 18]);
    runIndirect(this.compactChangedKeysPipeline, deltaEntries, "Compact sorted global fine changed keys",
      1, [0, 7, 15, 16, 18]);
    run(this.preparePageDeltaExpansionPipeline, deltaEntries, "Prepare global fine page delta expansion",
      1, [0, 7, 15, 18, 22]);
    broker.fence("fine changed-key dispatch publication");
    runIndirect(this.classifyAffectedPagesPipeline, deltaEntries,
      "Classify exact changed-key dirty and support pages",
      3, [0, 3, 4, 7, 15, 16, 21]);
    runIdentity(this.scanIdentityRecordsPipeline, deltaEntries, identityBlocks, 3, [0, 7, 15, 18]);
    runIdentity(this.scanIdentityGroupsPipeline, deltaEntries, identitySuperBlocks, 3, [0, 15]);
    runIdentity(this.scanIdentitySuperGroupsPipeline, deltaEntries, 1, 3, [0, 15]);
    runIdentity(this.offsetIdentityGroupsPipeline, deltaEntries, identitySuperBlocks, 3, [0, 15]);
    runIdentity(this.offsetIdentityRecordsPipeline, deltaEntries, identityBlocks, 3, [0, 15, 18]);
    runIndirect(this.compactAffectedPagesPipeline, deltaEntries, "Compact exact dirty and support pages",
      3, [0, 7, 15, 18]);
    run(this.finalizePageDeltaPipeline, deltaEntries, "Finalize global fine page delta",
      1, [0, 7, 15, 18, 22]);
    runIndirect(this.publishSummaryChangedKeysPipeline, deltaEntries,
      "Publish exact post-redistance fine summary keys",
      3, [0, 3, 7, 15, 16]);
    broker.fence("fine exact page-delta dispatch publication");
    const lifecycleEntries: GPUBindGroupEntry[] = [
      { binding: 0, resource: resource(this.params) },
      { binding: 3, resource: resource(this.next.metadata) },
      { binding: 4, resource: resource(this.next.worklist) },
      { binding: 5, resource: resource(this.next.flags) },
      { binding: 6, resource: resource(this.next.phi) },
      { binding: 7, resource: resource(this.control) },
      { binding: 8, resource: resource(seedSource?.buffer ?? this.emptySeeds) },
      { binding: 10, resource: resource(this.transportedPhiSnapshot) },
      { binding: 15, resource: resource(this.pageDelta) },
      { binding: 16, resource: resource(this.current.metadata) },
      { binding: 17, resource: resource(this.current.rollbackPhi) },
      { binding: 14, resource: resource(this.current.worklist) },
      { binding: 18, resource: resource(this.desiredCandidates) },
      { binding: 20, resource: resource(this.desiredScan) },
      { binding: 21, resource: resource(this.topologyErrors) },
      { binding: 22, resource: resource(this.indirectDispatch) },
      ...extraPublishEntries,
    ];
    runIndirect(this.snapshotPipeline, lifecycleEntries, "Snapshot exact global fine rollback pages",
      4, [0, 6, 7, 10, 15, 16, 21]);
    runIndirect(this.initializePipeline, lifecycleEntries, "Initialize added global fine samples",
      2, [0, 3, 5, 6, 7, 8, 14, 15, 21, ...extraPublishEntries.map((entry) => entry.binding)]);
    runIndirect(this.linkPipeline, publishEntries,
      "Link changed global fine neighbors", 6, [0, 3, 4, 7, 15, 21]);
    runIdentity(this.reduceTopologyErrorRecordsPipeline, lifecycleEntries,
      topologyErrorBlocks, 1, [0, 20, 21]);
    runIdentity(this.reduceTopologyErrorGroupsPipeline, lifecycleEntries,
      topologyErrorSuperBlocks, 1, [0, 20]);
    runIdentity(this.reduceTopologyErrorSuperGroupsPipeline, lifecycleEntries,
      1, 1, [0, 20]);
    run(this.finalizePipeline, publishEntries, "Finalize global fine publication", 1, [0, 4, 7, 14, 15, 20, 22]);
    if (!deferPublication) {
      broker.fence("fine immediate settlement dispatch publication");
      runIndirect(this.settlePublicationPipeline, [
        { binding: 0, resource: resource(this.params) },
        { binding: 3, resource: resource(this.next.metadata) },
        { binding: 4, resource: resource(this.next.worklist) },
        { binding: 5, resource: resource(this.next.flags) },
        { binding: 6, resource: resource(this.next.phi) },
        { binding: 7, resource: resource(this.control) },
        { binding: 10, resource: resource(this.transportedPhiSnapshot) },
        { binding: 14, resource: resource(this.current.worklist) },
        { binding: 15, resource: resource(this.pageDelta) },
        { binding: 16, resource: resource(this.current.metadata) },
        { binding: 17, resource: resource(this.current.rollbackPhi) },
      ], "Settle immediate global fine publication", 0,
      [0, 3, 4, 5, 6, 7, 10, 14, 15, 16, 17]);
    }
    broker.fence("global fine topology publication complete");
  }

  /** Commit only after the complete transport/topology/redistance/volume chain
   * is valid. A failed target is replaced GPU-side by the previous valid
   * generation, retagged for the target slot, before any later consumer runs. */
  encodeFinalizePublication(broker: PassBroker, controls: {
    redistance: GPUBuffer; volume?: GPUBuffer; transport?: GPUBuffer;
  }): void {
    const resource = (buffer: GPUBuffer) => ({ buffer });
    const pass = broker.compute({ label: "Finalize global fine publication" });
    pass.setPipeline(this.publicationPipeline);
    pass.setBindGroup(0, this.device.createBindGroup({ layout: this.publicationPipeline.getBindGroupLayout(0), entries: [
      { binding: 0, resource: resource(this.params) },
      { binding: 7, resource: resource(this.control) },
      { binding: 11, resource: resource(controls.redistance) },
      { binding: 12, resource: resource(controls.volume ?? this.validVolumeControl) },
      { binding: 13, resource: resource(controls.transport ?? this.validTransportControl) },
      { binding: 14, resource: resource(this.current.worklist) },
      { binding: 15, resource: resource(this.pageDelta) },
      { binding: 22, resource: resource(this.indirectDispatch) },
    ] }));
    pass.dispatchWorkgroups(1);
    broker.fence("fine deferred settlement dispatch publication");
    const settlePass = broker.compute({ label: "Settle deferred global fine publication" });
    settlePass.setPipeline(this.settlePublicationPipeline);
    settlePass.setBindGroup(0, this.device.createBindGroup({
      layout: this.settlePublicationPipeline.getBindGroupLayout(0), entries: [
      { binding: 0, resource: resource(this.params) },
        { binding: 3, resource: resource(this.next.metadata) },
        { binding: 4, resource: resource(this.next.worklist) },
        { binding: 5, resource: resource(this.next.flags) },
        { binding: 6, resource: resource(this.next.phi) },
        { binding: 7, resource: resource(this.control) },
        { binding: 10, resource: resource(this.transportedPhiSnapshot) },
        { binding: 14, resource: resource(this.current.worklist) },
        { binding: 15, resource: resource(this.pageDelta) },
        { binding: 16, resource: resource(this.current.metadata) },
        { binding: 17, resource: resource(this.current.rollbackPhi) },
      ],
    }));
    settlePass.dispatchWorkgroupsIndirect(this.indirectDispatch, 0);
    broker.fence("global fine topology publication complete");
  }

  destroy(): void { this.control.destroy(); this.params.destroy(); this.emptySeeds.destroy();
    this.pageDelta.destroy(); this.indirectDispatch.destroy(); this.transportedPhiSnapshot.destroy();
    this.desiredCandidates.destroy(); this.sparseCandidates.destroy(); this.desiredScan.destroy();
    this.topologyErrors.destroy();
    this.validVolumeControl.destroy(); this.validTransportControl.destroy(); }
}

export function makeFineLevelSetTopologyWGSL(coarsePhiWGSL: string): string {
  return /* wgsl */ `
	const INVALID:u32=0xffffffffu;const VALID:u32=1u;const CAPACITY:u32=1u;const NONFINITE:u32=4u;const MALFORMED:u32=8u;const RECURRING_SUPPORT:u32=0x80000000u;
struct Params { brickDimensions:vec3u,brickResolution:u32,sampleDimensions:vec3u,samplesPerBrick:u32,
 domainOrigin:vec3f,fineCellWidth:f32,sparseCandidateCapacity:u32,pageCapacity:u32,currentGeneration:u32,nextGeneration:u32,fineFactor:u32,affineSeeds:u32,dilationBrickRings:u32,deferPublication:u32,dirtyHaloRings:u32,supportHaloRings:u32,maxWorkgroups:u32,recurringDelta:u32 }
@group(0) @binding(0) var<uniform> params:Params;
@group(0) @binding(1) var<storage,read_write> sourceA:array<u32>;
@group(0) @binding(2) var<storage,read_write> sourceB:array<u32>;
@group(0) @binding(3) var<storage,read_write> sourceC:array<u32>;
@group(0) @binding(4) var<storage,read_write> sourceD:array<u32>;
@group(0) @binding(5) var<storage,read_write> targetA:array<u32>;
@group(0) @binding(6) var<storage,read_write> targetB:array<u32>;
@group(0) @binding(7) var<storage,read_write> control:array<u32>;
@group(0) @binding(8) var<storage,read> externalSeeds:array<u32>;
@group(0) @binding(10) var<storage,read_write> payloadSnapshot:array<f32>;
@group(0) @binding(11) var<storage,read> redistanceControl:array<u32>;
@group(0) @binding(12) var<storage,read> volumeControl:array<u32>;
@group(0) @binding(13) var<storage,read> transportControl:array<u32>;
@group(0) @binding(14) var<storage,read> currentWorklist:array<u32>;
@group(0) @binding(15) var<storage,read_write> pageDelta:array<u32>;
@group(0) @binding(16) var<storage,read> currentMetadata:array<u32>;
@group(0) @binding(17) var<storage,read_write> committedPhi:array<u32>;
@group(0) @binding(18) var<storage,read_write> desiredCandidates:array<u32>;
		@group(0) @binding(19) var<storage,read_write> sparseCandidates:array<u32>;
		@group(0) @binding(20) var<storage,read_write> desiredScan:array<u32>;
		@group(0) @binding(21) var<storage,read_write> topologyErrors:array<atomic<u32>>;
		@group(0) @binding(22) var<storage,read_write> indirectDispatch:array<u32>;
		@group(0) @binding(23) var<storage,read> transportDelta:array<u32>;
${coarsePhiWGSL}
${fineLevelSetLinearWorkgroupWGSL}
// The compact coarse sampler uses max-finite as an explicit invalid sentinel;
// strict comparison rejects it without asking Dawn to constant-fold a NaN.
fn finite(value:f32)->bool{return value==value&&abs(value)<3.402823e38;}
fn packBrick(coord:vec3u)->u32{return coord.x+params.brickDimensions.x*(coord.y+params.brickDimensions.y*coord.z);}
fn unpackBrick(key:u32)->vec3u{let xy=params.brickDimensions.x*params.brickDimensions.y;let z=key/xy;let rem=key-z*xy;let y=rem/params.brickDimensions.x;return vec3u(rem-y*params.brickDimensions.x,y,z);}
fn localCoord(local:u32)->vec3u{let r=params.brickResolution;let z=local/(r*r);let rem=local-z*r*r;let y=rem/r;return vec3u(rem-y*r,y,z);}
fn localIndex(coord:vec3u)->u32{return coord.x+params.brickResolution*(coord.y+params.brickResolution*coord.z);}
fn currentNeighbor(id:u32,local:u32,direction:u32)->u32{var coord=localCoord(local);var nextId=id;let r=params.brickResolution;
 if(direction==0u){if(coord.x>0u){coord.x-=1u;}else{nextId=sourceA[id*10u+4u];coord.x=r-1u;}}
 else if(direction==1u){if(coord.x+1u<r){coord.x+=1u;}else{nextId=sourceA[id*10u+5u];coord.x=0u;}}
 else if(direction==2u){if(coord.y>0u){coord.y-=1u;}else{nextId=sourceA[id*10u+6u];coord.y=r-1u;}}
 else if(direction==3u){if(coord.y+1u<r){coord.y+=1u;}else{nextId=sourceA[id*10u+7u];coord.y=0u;}}
 else if(direction==4u){if(coord.z>0u){coord.z-=1u;}else{nextId=sourceA[id*10u+8u];coord.z=r-1u;}}
 else{if(coord.z+1u<r){coord.z+=1u;}else{nextId=sourceA[id*10u+9u];coord.z=0u;}}
 if(nextId==INVALID||nextId>=params.pageCapacity||sourceA[nextId*10u+2u]!=params.currentGeneration){return INVALID;}
 return nextId*params.samplesPerBrick+localIndex(coord);}
fn currentLookup(key:u32)->u32{let count=min(currentWorklist[0],params.pageCapacity);var low=0u;var high=count;
 while(low<high){let middle=low+(high-low)/2u;let id=currentWorklist[5u+middle];if(id>=params.pageCapacity){return INVALID;}
  if(currentMetadata[id*10u+1u]<key){low=middle+1u;}else{high=middle;}}
 if(low<count){let id=currentWorklist[5u+low];if(id<params.pageCapacity&&currentMetadata[id*10u+1u]==key
  &&currentMetadata[id*10u+2u]==params.currentGeneration){return id;}}return INVALID;}
fn externalSeedTaggedValue(key:u32)->u32{if(arrayLength(&externalSeeds)<4u){return INVALID;}let count=min(externalSeeds[0],params.pageCapacity);var low=0u;var high=count;while(low<high){let mid=low+(high-low)/2u;let stored=externalSeeds[4u+mid];if(stored<key){low=mid+1u;}else{high=mid;}}if(low<count&&externalSeeds[4u+low]==key){return externalSeeds[4u+params.pageCapacity+low];}return INVALID;}
fn currentFinePublished()->bool{return arrayLength(&currentWorklist)>=5u&&currentWorklist[1]==params.currentGeneration&&currentWorklist[3]==1u&&currentWorklist[4]==1u;}
fn currentFinePopulated()->bool{return currentFinePublished()&&currentWorklist[0]>0u;}
fn exactAnalyticSeedPhi(finestPoint:vec3f)->f32{if(arrayLength(&externalSeeds)<4u){return 3.402823e38;}let mode=externalSeeds[2u];let fill=bitcast<f32>(externalSeeds[3u]);if(mode==0u||!finite(fill)||fill<0.0||fill>1.0){return 3.402823e38;}let extent=vec3f(params.sampleDimensions)*params.fineCellWidth;let point=params.domainOrigin+finestPoint*(params.fineCellWidth*f32(params.fineFactor));if(mode==1u){return point.y-fill*extent.y;}let heightFraction=max(0.92,fill);let footprintFraction=sqrt(fill/max(heightFraction,1e-9));let exposedMaximum=params.domainOrigin+vec3f(footprintFraction*extent.x,heightFraction*extent.y,footprintFraction*extent.z);let q=point-exposedMaximum;return length(max(q,vec3f(0.0)))+min(max(q.x,max(q.y,q.z)),0.0);}
fn externalSeedPhi(key:u32,finestPoint:vec3f)->f32{if(params.affineSeeds==0u||currentFinePopulated()){return 3.402823e38;}let analytic=exactAnalyticSeedPhi(finestPoint);if(finite(analytic)){return analytic;}let tagged=externalSeedTaggedValue(key);if(tagged==INVALID){return 3.402823e38;}let seed=tagged&0x7fffffffu;if(seed>=params.pageCapacity){return 3.402823e38;}let planeBase=4u+2u*params.pageCapacity;let base=planeBase+seed*8u;let leafOrigin=vec3f(vec3u(externalSeeds[base],externalSeeds[base+1u],externalSeeds[base+2u]));let size=f32(externalSeeds[base+3u]);let centre=leafOrigin+vec3f(0.5*size);let value=bitcast<f32>(externalSeeds[base+4u]);let gradient=vec3f(bitcast<f32>(externalSeeds[base+5u]),bitcast<f32>(externalSeeds[base+6u]),bitcast<f32>(externalSeeds[base+7u]));return value+dot(gradient,finestPoint-centre);}
// The initial A/B source is a deliberately published empty generation. It is
// still a cold start: classify ordinary analytic/affine leaf keys by an actual
// zero crossing so only interface blocks precede the paper's one-ring
// allocation. Test the brick's geometric support bounds, not only its first
// and last sample centres. A face can lie exactly between two SPGrid pages;
// centre-only bounds then reject both pages and punch a vertical gap in the
// high-resolution interface band.
fn externalAffineInterfaceBrick(key:u32)->bool{if(params.affineSeeds==0u||currentFinePopulated()){return false;}let brick=unpackBrick(key);let first=vec3f(brick*params.brickResolution)/f32(params.fineFactor);let last=vec3f((brick+vec3u(1u))*params.brickResolution)/f32(params.fineFactor);var minimum=3.402823e38;var maximum=-3.402823e38;for(var corner=0u;corner<8u;corner+=1u){let point=vec3f(select(first.x,last.x,(corner&1u)!=0u),select(first.y,last.y,(corner&2u)!=0u),select(first.z,last.z,(corner&4u)!=0u));let value=externalSeedPhi(key,point);if(!finite(value)){return false;}minimum=min(minimum,value);maximum=max(maximum,value);}return minimum<=0.0&&maximum>=0.0;}
fn linearInvocation(wid:vec3u,nwg:vec3u,local:u32)->u32{return fineLinearWorkgroup(wid,nwg)*64u+local;}
fn changedKeysOffset()->u32{return 16u;}fn dirtyPagesOffset()->u32{return 16u+2u*params.pageCapacity;}
fn supportPagesOffset()->u32{return 16u+3u*params.pageCapacity;}fn desiredKeysOffset()->u32{return 16u+4u*params.pageCapacity;}
fn addedPagesOffset()->u32{return 16u+5u*params.pageCapacity;}fn retiredPagesOffset()->u32{return 16u+6u*params.pageCapacity;}
fn rollbackPagesOffset()->u32{return 16u+7u*params.pageCapacity;}fn changedCandidatesOffset()->u32{return 16u+8u*params.pageCapacity;}
fn dirtyCandidatesOffset()->u32{return 16u+10u*params.pageCapacity;}fn supportCandidatesOffset()->u32{return 16u+11u*params.pageCapacity;}
fn lifecycleDispatchOffset()->u32{let blocks=(params.pageCapacity+255u)/256u;let superBlocks=(blocks+255u)/256u;
 return 16u+12u*params.pageCapacity+4u*(blocks+superBlocks);}
var<workgroup> topologyErrorLanes:array<u32,64>;
fn publishTopologyError(work:u32,local:u32,flag:u32,laneActive:bool){topologyErrorLanes[local]=flag;workgroupBarrier();var width=32u;loop{if(width==0u){break;}if(local<width){topologyErrorLanes[local]|=topologyErrorLanes[local+width];}workgroupBarrier();width>>=1u;}if(local==0u&&laneActive){atomicOr(&topologyErrors[work],topologyErrorLanes[0]);}}
fn producerChangedContains(key:u32)->bool{if(params.recurringDelta==0u||arrayLength(&transportDelta)<8u+2u*params.pageCapacity
 ||transportDelta[1]!=params.currentGeneration||transportDelta[2]!=1u||transportDelta[0]>params.pageCapacity){return false;}
 let count=transportDelta[0];var low=0u;var high=count;while(low<high){let middle=low+(high-low)/2u;
  if(transportDelta[8u+params.pageCapacity+middle]<key){low=middle+1u;}else{high=middle;}}
 return low<count&&transportDelta[8u+params.pageCapacity+low]==key;}
	fn dispatchRecord(count:u32)->vec3u{let x=min(count,params.maxWorkgroups);var y=1u;if(x>0u){y=(count+x-1u)/x;}return vec3u(x,y,1u);}
	fn writeDeltaDispatch(offset:u32,count:u32){let record=dispatchRecord(count);pageDelta[offset]=record.x;pageDelta[offset+1u]=record.y;pageDelta[offset+2u]=record.z;}
	fn writeIndirectDispatch(slot:u32,count:u32){let record=dispatchRecord(count);let base=3u*slot;indirectDispatch[base]=record.x;indirectDispatch[base+1u]=record.y;indirectDispatch[base+2u]=record.z;}
	@compute @workgroup_size(64) fn clearFinePageDelta(@builtin(workgroup_id)wid:vec3u,@builtin(num_workgroups)nwg:vec3u,@builtin(local_invocation_index)local:u32){let item=linearInvocation(wid,nwg,local);if(item<16u){pageDelta[item]=0u;}if(item<21u){pageDelta[lifecycleDispatchOffset()+item]=0u;}if(item<24u){indirectDispatch[item]=0u;}if(item==1u){pageDelta[1]=params.nextGeneration;}}
@compute @workgroup_size(64) fn clearDesiredGeneration(@builtin(workgroup_id)wid:vec3u,@builtin(num_workgroups)nwg:vec3u,@builtin(local_invocation_index)local:u32){let item=linearInvocation(wid,nwg,local);if(item<5u+params.pageCapacity){targetB[item]=0u;}if(item<9u&&item!=6u){control[item]=0u;}if(item==6u){control[6]=params.dilationBrickRings;}}
@compute @workgroup_size(64) fn clearFinePageDirectory(@builtin(workgroup_id)wid:vec3u,@builtin(num_workgroups)nwg:vec3u,@builtin(local_invocation_index)local:u32){let key=linearInvocation(wid,nwg,local);if(key<desiredLogicalCount()){sourceD[5u+params.pageCapacity+key]=INVALID;}}
@compute @workgroup_size(64) fn clearTopologyErrors(@builtin(workgroup_id)wid:vec3u,@builtin(num_workgroups)nwg:vec3u,@builtin(local_invocation_index)local:u32){let work=linearInvocation(wid,nwg,local);if(work<params.pageCapacity){atomicStore(&topologyErrors[work],0u);}}
@compute @workgroup_size(64) fn discoverInterfaceBricks(@builtin(workgroup_id)wid:vec3u,@builtin(num_workgroups)nwg:vec3u,@builtin(local_invocation_index)local:u32){let work=linearInvocation(wid,nwg,local);if(work>=params.pageCapacity){return;}desiredCandidates[work]=INVALID;let activeCount=min(sourceB[0],params.pageCapacity);if(work>=activeCount){return;}let id=sourceB[5u+work];if(id>=params.pageCapacity||sourceA[id*10u+2u]!=params.currentGeneration){atomicOr(&topologyErrors[work],MALFORMED);return;}var interfaceBrick=false;var malformed=false;for(var sample=0u;sample<params.samplesPerBrick&&!interfaceBrick;sample+=1u){let index=id*params.samplesPerBrick+sample;if((sourceC[index]&VALID)==0u){continue;}let center=bitcast<f32>(sourceD[index]);if(!finite(center)){malformed=true;continue;}for(var direction=0u;direction<6u;direction+=1u){let neighbor=currentNeighbor(id,sample,direction);if(neighbor==INVALID||(sourceC[neighbor]&VALID)==0u){continue;}let other=bitcast<f32>(sourceD[neighbor]);if(finite(other)&&(other<0.0)!=(center<0.0)){interfaceBrick=true;break;}}}desiredCandidates[work]=select(INVALID,sourceA[id*10u+1u],interfaceBrick);if(malformed){desiredCandidates[work]=INVALID;atomicOr(&topologyErrors[work],MALFORMED);}}
@compute @workgroup_size(64) fn insertExternalSeeds(@builtin(workgroup_id)wid:vec3u,@builtin(num_workgroups)nwg:vec3u,@builtin(local_invocation_index)local:u32){let recurring=currentFinePopulated();let seed=linearInvocation(wid,nwg,local);if(seed>=params.pageCapacity){return;}let output=params.pageCapacity+seed;desiredCandidates[output]=INVALID;if(arrayLength(&externalSeeds)<4u){return;}let rawCount=externalSeeds[0];let available=arrayLength(&externalSeeds)-4u;if(seed>=min(rawCount,min(params.pageCapacity,available))){return;}if(externalSeeds[1]!=0u||rawCount>params.pageCapacity||rawCount>available){atomicOr(&topologyErrors[seed],MALFORMED);return;}let key=externalSeeds[4u+seed];let tagged=externalSeedTaggedValue(key);let endpoint=tagged!=INVALID&&(tagged&RECURRING_SUPPORT)!=0u;if(recurring&&!endpoint){return;}if(!recurring&&!endpoint&&!externalAffineInterfaceBrick(key)){return;}if(key<params.brickDimensions.x*params.brickDimensions.y*params.brickDimensions.z){desiredCandidates[output]=key;}else{atomicOr(&topologyErrors[seed],MALFORMED);}}
fn desiredLogicalCount()->u32{return params.brickDimensions.x*params.brickDimensions.y*params.brickDimensions.z;}
fn seedRecordPresent(item:u32)->bool{
 let key=desiredCandidates[item];if(key>=desiredLogicalCount()){return false;}
 if(item<params.pageCapacity){return true;}
 let count=min(sourceB[0],params.pageCapacity);var low=0u;var high=count;
 while(low<high){let middle=low+(high-low)/2u;let id=sourceB[5u+middle];
  if(id>=params.pageCapacity||sourceA[id*10u+1u]<key){low=middle+1u;}else{high=middle;}}
 return !(low<count&&desiredCandidates[low]==key);
}
fn sparseScanScratch(index:u32)->u32{return params.sparseCandidateCapacity+index;}
fn sparseScanBlockCount()->u32{return (control[9]+255u)/256u;}
fn sparseScanSuperBlockCount()->u32{return (sparseScanBlockCount()+255u)/256u;}
@compute @workgroup_size(256) fn scanSparseSeedRecords(@builtin(workgroup_id)wid:vec3u,@builtin(local_invocation_index)local:u32){
 let item=wid.x*256u+local;let count=2u*params.pageCapacity;let present=item<count&&seedRecordPresent(item);
 let prefix=scanIdentityBlock(local,select(0u,1u,present));if(item<count){desiredScan[item]=prefix;}
 if(local==255u){desiredScan[sparseScanScratch(wid.x)]=identityScanTotal;}
 if(item==0u){control[9]=count;}}
@compute @workgroup_size(256) fn scanSparseCandidateRecords(@builtin(workgroup_id)wid:vec3u,@builtin(local_invocation_index)local:u32){
 let item=wid.x*256u+local;let count=params.sparseCandidateCapacity;
 let present=item<count&&sparseCandidateRunStart(item);
 let prefix=scanIdentityBlock(local,select(0u,1u,present));if(item<count){desiredScan[item]=prefix;}
 if(local==255u){desiredScan[sparseScanScratch(wid.x)]=identityScanTotal;}
 if(item==0u){control[9]=count;}}
@compute @workgroup_size(256) fn scanSparseGroups(@builtin(workgroup_id)wid:vec3u,@builtin(local_invocation_index)local:u32){
 let item=wid.x*256u+local;let blocks=sparseScanBlockCount();var value=0u;
 if(item<blocks){value=desiredScan[sparseScanScratch(item)];}
 let prefix=scanIdentityBlock(local,value);if(item<blocks){desiredScan[sparseScanScratch(item)]=prefix;}
 if(local==255u){desiredScan[sparseScanScratch(blocks+wid.x)]=identityScanTotal;}}
@compute @workgroup_size(256) fn scanSparseSuperGroups(@builtin(local_invocation_index)local:u32){
 let blocks=sparseScanBlockCount();let count=sparseScanSuperBlockCount();var value=0u;
 if(local<count){value=desiredScan[sparseScanScratch(blocks+local)];}
 let prefix=scanIdentityBlock(local,value);if(local<count){desiredScan[sparseScanScratch(blocks+local)]=prefix;}}
@compute @workgroup_size(256) fn offsetSparseGroups(@builtin(workgroup_id)wid:vec3u,@builtin(local_invocation_index)local:u32){
 let item=wid.x*256u+local;let blocks=sparseScanBlockCount();
 if(item<blocks){desiredScan[sparseScanScratch(item)]+=desiredScan[sparseScanScratch(blocks+wid.x)];}}
@compute @workgroup_size(256) fn offsetSparseRecords(@builtin(workgroup_id)wid:vec3u,@builtin(local_invocation_index)local:u32){
 let item=wid.x*256u+local;if(item<control[9]){desiredScan[item]+=desiredScan[sparseScanScratch(wid.x)];}}
fn sparseSeedTotal()->u32{let count=2u*params.pageCapacity;if(count==0u){return 0u;}let last=count-1u;
 return desiredScan[last]+select(0u,1u,seedRecordPresent(last));}
fn sparseCandidateTotal()->u32{
 let count=params.sparseCandidateCapacity;let last=count-1u;
 return desiredScan[last]+select(0u,1u,sparseCandidateRunStart(last));
}
@compute @workgroup_size(1) fn finalizeDesiredSeedCount(){if(control[0]==0u){control[8]=sparseSeedTotal();control[1]=control[8];}}
@compute @workgroup_size(64) fn clearSparseCandidates(@builtin(workgroup_id)wid:vec3u,@builtin(num_workgroups)nwg:vec3u,@builtin(local_invocation_index)local:u32){
 let item=linearInvocation(wid,nwg,local);if(item<params.sparseCandidateCapacity){sparseCandidates[item]=INVALID;}}
@compute @workgroup_size(64) fn compactSparseSeeds(@builtin(workgroup_id)wid:vec3u,@builtin(num_workgroups)nwg:vec3u,@builtin(local_invocation_index)local:u32){
 let item=linearInvocation(wid,nwg,local);if(item>=2u*params.pageCapacity||control[0]!=0u||!seedRecordPresent(item)){return;}
 sparseCandidates[desiredScan[item]]=desiredCandidates[item];
}
fn sparseCandidateRunStart(item:u32)->bool{let key=sparseCandidates[item];
 return key!=INVALID&&(item==0u||sparseCandidates[item-1u]!=key);}
@compute @workgroup_size(256) fn sortSparseCandidates(@builtin(local_invocation_index)local:u32){
 let count=params.sparseCandidateCapacity;for(var width=2u;width<=count;width<<=1u){
  for(var stride=width>>1u;stride>0u;stride>>=1u){for(var item=local;item<count;item+=256u){
   let partner=item^stride;if(partner>item){let left=sparseCandidates[item];let right=sparseCandidates[partner];
    let descending=(item&width)!=0u;if((left>right)!=descending){sparseCandidates[item]=right;sparseCandidates[partner]=left;}}}
   workgroupBarrier();}}}
fn expandedAxisKey(item:u32,axis:u32)->u32{
 let owner=item/3u;let ordinal=item-owner*3u;let count=min(control[2],params.pageCapacity);
 if(owner>=count){return INVALID;}let key=sparseCandidates[params.sparseCandidateCapacity+owner];
 if(key>=desiredLogicalCount()){return INVALID;}var q=vec3i(unpackBrick(key));
 let offset=i32(ordinal)-1;if(axis==0u){q.x+=offset;}else if(axis==1u){q.y+=offset;}else{q.z+=offset;}
 if(any(q<vec3i(0))||any(q>=vec3i(params.brickDimensions))){return INVALID;}return packBrick(vec3u(q));}
@compute @workgroup_size(64) fn expandSparseDesiredX(@builtin(workgroup_id)wid:vec3u,@builtin(num_workgroups)nwg:vec3u,@builtin(local_invocation_index)local:u32){
 let item=linearInvocation(wid,nwg,local);if(item<params.sparseCandidateCapacity){sparseCandidates[item]=expandedAxisKey(item,0u);}}
@compute @workgroup_size(64) fn expandSparseDesiredY(@builtin(workgroup_id)wid:vec3u,@builtin(num_workgroups)nwg:vec3u,@builtin(local_invocation_index)local:u32){
 let item=linearInvocation(wid,nwg,local);if(item<params.sparseCandidateCapacity){sparseCandidates[item]=expandedAxisKey(item,1u);}}
@compute @workgroup_size(64) fn expandSparseDesiredZ(@builtin(workgroup_id)wid:vec3u,@builtin(num_workgroups)nwg:vec3u,@builtin(local_invocation_index)local:u32){
 let item=linearInvocation(wid,nwg,local);if(item<params.sparseCandidateCapacity){sparseCandidates[item]=expandedAxisKey(item,2u);}}
@compute @workgroup_size(64) fn compactSparseDesiredBricks(@builtin(workgroup_id)wid:vec3u,@builtin(num_workgroups)nwg:vec3u,@builtin(local_invocation_index)local:u32){
 let item=linearInvocation(wid,nwg,local);if(item>=params.sparseCandidateCapacity){return;}
 let count=sparseCandidateTotal();if(item==0u){if(count>params.pageCapacity){control[0]|=CAPACITY;control[6]=count;}else{control[2]=count;}}
 if(count<=params.pageCapacity&&sparseCandidateRunStart(item)){
  sparseCandidates[params.sparseCandidateCapacity+desiredScan[item]]=sparseCandidates[item];}
}
@compute @workgroup_size(64) fn publishDesiredBricks(@builtin(workgroup_id)wid:vec3u,@builtin(num_workgroups)nwg:vec3u,@builtin(local_invocation_index)local:u32){
 let work=linearInvocation(wid,nwg,local);let count=control[2];if(work>=count||control[0]!=0u){return;}
 targetB[5u+work]=sparseCandidates[params.sparseCandidateCapacity+work];
}
fn recurringProducerCount()->u32{return min(transportDelta[0],params.pageCapacity);}
fn recurringProducerChanged(index:u32)->u32{return transportDelta[8u+params.pageCapacity+index];}
fn recurringExternalCount()->u32{
 if(arrayLength(&externalSeeds)<4u){return 0u;}
 return min(externalSeeds[0],params.pageCapacity);
}
var<workgroup> recurringFlags:u32;
var<workgroup> recurringCount:u32;
var<workgroup> recurringSeedCount:u32;
// The logical brick key is already a perfect dense address. Scatter each
// compact seed's bounded Chebyshev halo into that address space: bit zero is
// exact seed membership and bit one is desired-band membership. Atomic OR is
// idempotent, so overlapping halos are deduplicated at the point of insertion
// without a sort, hash probe, or per-output search.
fn recurringScatterMembership(key:u32){
 if(key>=desiredLogicalCount()){return;}
 let center=vec3i(unpackBrick(key));let radius=i32(params.dilationBrickRings);
 for(var z=-radius;z<=radius;z+=1){
  for(var y=-radius;y<=radius;y+=1){
   for(var x=-radius;x<=radius;x+=1){
    let point=center+vec3i(x,y,z);
    if(any(point<vec3i(0))||any(point>=vec3i(params.brickDimensions))){continue;}
    let output=packBrick(vec3u(point));
    let exact=x==0&&y==0&&z==0;
    atomicOr(&topologyErrors[output],2u|select(0u,1u,exact));
   }
  }
 }
}
// Dense key order is canonical row order, so one bounded linear pass is both
// compaction and publication. Physical page identity remains a later delta
// concern; no residency lookup is needed here.
fn recurringPublishScatteredDomain(local:u32){
 if(local==0u){
  var output=0u;var seeds=0u;
  for(var key=0u;key<desiredLogicalCount();key+=1u){
   let membership=atomicLoad(&topologyErrors[key]);
   seeds+=membership&1u;
   if((membership&2u)==0u){continue;}
   if(output>=params.pageCapacity){recurringFlags|=CAPACITY;break;}
   sparseCandidates[params.sparseCandidateCapacity+output]=key;output+=1u;
  }
  recurringSeedCount=seeds;recurringCount=output;
 }
 storageBarrier();workgroupBarrier();
}
fn recurringSort(local:u32){
 let count=params.sparseCandidateCapacity;
 for(var width=2u;width<=count;width<<=1u){
  for(var stride=width>>1u;stride>0u;stride>>=1u){
   for(var item=local;item<count;item+=256u){
    let partner=item^stride;if(partner>item){let left=sparseCandidates[item];let right=sparseCandidates[partner];
     let descending=(item&width)!=0u;if((left>right)!=descending){
      sparseCandidates[item]=right;sparseCandidates[partner]=left;
     }}
   }
   storageBarrier();workgroupBarrier();
  }
 }
}
fn recurringCompact(local:u32){
 if(local==0u){
  var output=0u;var previous=INVALID;
  for(var item=0u;item<params.sparseCandidateCapacity;item+=1u){
   let key=sparseCandidates[item];
   if(key==INVALID){break;}
   if(key!=previous){
    if(output>=params.pageCapacity){recurringFlags|=CAPACITY;break;}
    sparseCandidates[params.sparseCandidateCapacity+output]=key;
    output+=1u;previous=key;
   }
  }
  recurringCount=output;
 }
 storageBarrier();workgroupBarrier();
}
fn recurringExpandedAxisKey(item:u32,axis:u32)->u32{
 let owner=item/3u;let ordinal=item-owner*3u;if(owner>=recurringCount){return INVALID;}
 let key=sparseCandidates[params.sparseCandidateCapacity+owner];
 if(key>=desiredLogicalCount()){return INVALID;}
 var q=vec3i(unpackBrick(key));let offset=i32(ordinal)-1;
 if(axis==0u){q.x+=offset;}else if(axis==1u){q.y+=offset;}else{q.z+=offset;}
 if(any(q<vec3i(0))||any(q>=vec3i(params.brickDimensions))){return INVALID;}
 return packBrick(vec3u(q));
}
@compute @workgroup_size(256) fn publishRecurringSparseBand(
 @builtin(local_invocation_index)local:u32){
 if(local==0u){
  recurringFlags=0u;recurringCount=0u;recurringSeedCount=0u;
  for(var word=0u;word<10u;word+=1u){control[word]=0u;}
  control[6]=params.dilationBrickRings;
  for(var word=0u;word<5u;word+=1u){targetB[word]=0u;}
  if(params.recurringDelta==0u||arrayLength(&transportDelta)<8u+2u*params.pageCapacity
    ||arrayLength(&currentWorklist)<5u||currentWorklist[0]>params.pageCapacity
    ||currentWorklist[1]!=params.currentGeneration||currentWorklist[3]!=1u||currentWorklist[4]!=1u
    ||transportDelta[1]!=params.currentGeneration||transportDelta[2]!=1u
    ||transportDelta[3]!=currentWorklist[0]||transportDelta[0]>params.pageCapacity
    ||(arrayLength(&externalSeeds)>=4u
      &&(externalSeeds[1]!=0u||externalSeeds[0]>params.pageCapacity
        ||externalSeeds[0]>arrayLength(&externalSeeds)-4u))){
   recurringFlags=MALFORMED;
  }
 }
 workgroupBarrier();
 var localError=0u;
 for(var item=local;item<params.sparseCandidateCapacity;item+=256u){
  sparseCandidates[item]=INVALID;
 }
 for(var item=local;item<arrayLength(&topologyErrors);item+=256u){
  atomicStore(&topologyErrors[item],0u);
 }
 storageBarrier();workgroupBarrier();
 for(var item=local;item<params.pageCapacity;item+=256u){
  var interfaceKey=INVALID;var endpointKey=INVALID;
  if(item<recurringProducerCount()){
   let key=recurringProducerChanged(item);
   if(key>=desiredLogicalCount()){localError|=MALFORMED;}
   else{let id=currentLookup(key);
    if(id==INVALID||currentMetadata[id*10u]!=id
      ||currentMetadata[id*10u+2u]!=params.currentGeneration){localError|=MALFORMED;}
    else if((currentMetadata[id*10u+3u]&2u)!=0u){interfaceKey=key;}
   }
  }
  if(item<recurringExternalCount()){
   let key=externalSeeds[4u+item];let tagged=externalSeedTaggedValue(key);
   if(key>=desiredLogicalCount()||tagged==INVALID){localError|=MALFORMED;}
   else if((tagged&RECURRING_SUPPORT)!=0u){endpointKey=key;}
  }
  sparseCandidates[item]=interfaceKey;
  sparseCandidates[params.pageCapacity+item]=endpointKey;
  if(desiredLogicalCount()<=min(params.sparseCandidateCapacity,arrayLength(&topologyErrors))){
   if(interfaceKey!=INVALID){recurringScatterMembership(interfaceKey);}
   if(endpointKey!=INVALID){recurringScatterMembership(endpointKey);}
  }
 }
 errorOrLanes[local]=localError;workgroupBarrier();var width=128u;
 loop{if(width==0u){break;}if(local<width){errorOrLanes[local]|=errorOrLanes[local+width];}
  workgroupBarrier();width>>=1u;}
 if(local==0u){recurringFlags|=errorOrLanes[0];}
 storageBarrier();workgroupBarrier();
 let recurringErrorFlags=workgroupUniformLoad(&recurringFlags);
 if(recurringErrorFlags==0u){
  if(desiredLogicalCount()<=min(params.sparseCandidateCapacity,arrayLength(&topologyErrors))){
   recurringPublishScatteredDomain(local);
  }else{
   recurringSort(local);recurringCompact(local);
   if(local==0u){recurringSeedCount=recurringCount;}
   workgroupBarrier();
   for(var ring=0u;ring<params.dilationBrickRings;ring+=1u){
    for(var axis=0u;axis<3u;axis+=1u){
     for(var item=local;item<params.sparseCandidateCapacity;item+=256u){
      sparseCandidates[item]=recurringExpandedAxisKey(item,axis);
     }
     storageBarrier();workgroupBarrier();recurringSort(local);recurringCompact(local);
    }
   }
  }
 }
 for(var item=local;item<recurringCount;item+=256u){
  targetB[5u+item]=sparseCandidates[params.sparseCandidateCapacity+item];
 }
 storageBarrier();workgroupBarrier();
 if(local==0u){
  control[0]=recurringFlags;control[1]=recurringSeedCount;control[2]=recurringCount;
  control[6]=select(params.dilationBrickRings,params.pageCapacity+1u,(recurringFlags&CAPACITY)!=0u);
  control[8]=recurringSeedCount;
 }
 for(var item=local;item<min(desiredLogicalCount(),arrayLength(&topologyErrors));item+=256u){
  atomicStore(&topologyErrors[item],0u);
 }
}
fn desiredContains(key:u32)->bool{let count=min(control[2],params.pageCapacity);var low=0u;var high=count;
 while(low<high){let middle=low+(high-low)/2u;if(sourceD[5u+middle]<key){low=middle+1u;}else{high=middle;}}
 return low<count&&sourceD[5u+low]==key;}
fn targetLookup(key:u32)->u32{let count=min(control[2],params.pageCapacity);var low=0u;var high=count;
 while(low<high){let middle=low+(high-low)/2u;let id=sourceD[5u+middle];if(id>=params.pageCapacity){return INVALID;}
  if(sourceC[id*10u+1u]<key){low=middle+1u;}else{high=middle;}}
 if(low<count){let id=sourceD[5u+low];if(id<params.pageCapacity&&sourceC[id*10u+1u]==key
  &&sourceC[id*10u+2u]==params.nextGeneration){return id;}}return INVALID;}
fn identityBlockCount()->u32{return (params.pageCapacity+255u)/256u;}
fn identitySuperBlockCount()->u32{return (identityBlockCount()+255u)/256u;}
fn identityScanStride()->u32{return identityBlockCount()+identitySuperBlockCount();}
fn identityScanScratchOffset()->u32{return supportCandidatesOffset()+params.pageCapacity;}
fn identityScanScratch(stream:u32,index:u32)->u32{return identityScanScratchOffset()+stream*identityScanStride()+index;}
fn carriedPage(key:u32)->u32{let old=currentLookup(key);if(old==INVALID||old>=params.pageCapacity){return INVALID;}
 let base=old*10u;return select(INVALID,old,currentMetadata[base+1u]==key&&currentMetadata[base+2u]==params.currentGeneration);}
fn identityCandidate(stream:u32,item:u32)->bool{
 if(pageDelta[10]==1u){
  if(item>=min(control[2],params.pageCapacity)){return false;}
  if(stream==0u){return pageDelta[dirtyCandidatesOffset()+item]!=INVALID;}
  if(stream==1u){return pageDelta[supportCandidatesOffset()+item]!=INVALID;}
  if(stream==2u){return pageDelta[changedCandidatesOffset()+item]!=INVALID;}
  return false;}
 if(stream==0u){return pageDelta[changedCandidatesOffset()+item]!=INVALID;}
 if(stream==1u){return pageDelta[rollbackPagesOffset()+item]!=INVALID;}
 if(stream==2u){return pageDelta[supportCandidatesOffset()+item]!=INVALID;}
 return pageDelta[dirtyPagesOffset()+item]!=0u;
}
fn identityPrefix(stream:u32,item:u32)->u32{
 if(pageDelta[10]==1u){
  if(stream==0u){return pageDelta[changedKeysOffset()+item];}
  if(stream==1u){return pageDelta[changedKeysOffset()+params.pageCapacity+item];}
  return desiredCandidates[item];}
 if(stream==0u){return desiredCandidates[item];}
 if(stream==1u){return desiredCandidates[params.pageCapacity+item];}
 if(stream==2u){return pageDelta[dirtyCandidatesOffset()+item];}
 return pageDelta[changedKeysOffset()+item];
}
fn writeIdentityPrefix(stream:u32,item:u32,value:u32){
 if(pageDelta[10]==1u){
  if(stream==0u){pageDelta[changedKeysOffset()+item]=value;}
  else if(stream==1u){pageDelta[changedKeysOffset()+params.pageCapacity+item]=value;}
  else{desiredCandidates[item]=value;}return;}
 if(stream==0u){desiredCandidates[item]=value;}
 else if(stream==1u){desiredCandidates[params.pageCapacity+item]=value;}
 else if(stream==2u){pageDelta[dirtyCandidatesOffset()+item]=value;}
 else{pageDelta[changedKeysOffset()+item]=value;}
}
@compute @workgroup_size(64) fn classifyDesiredPageIdentities(
 @builtin(workgroup_id)wid:vec3u,@builtin(num_workgroups)nwg:vec3u,@builtin(local_invocation_index)local:u32){
 let item=linearInvocation(wid,nwg,local);if(item>=params.pageCapacity){return;}
 let desiredCount=min(control[2],params.pageCapacity);let currentCount=min(currentWorklist[0],params.pageCapacity);
 pageDelta[rollbackPagesOffset()+item]=INVALID;pageDelta[supportCandidatesOffset()+item]=INVALID;
 pageDelta[dirtyPagesOffset()+item]=0u;pageDelta[changedCandidatesOffset()+item]=INVALID;
 if(item<desiredCount){let key=sourceD[5u+item];pageDelta[desiredKeysOffset()+item]=key;
  if(carriedPage(key)==INVALID){pageDelta[changedCandidatesOffset()+item]=key;}}
 if(item<currentCount){let id=currentWorklist[5u+item];var malformed=id>=params.pageCapacity;
  if(!malformed){let base=id*10u;malformed=currentMetadata[base]!=id||currentMetadata[base+2u]!=params.currentGeneration;}
  pageDelta[dirtyPagesOffset()+item]=select(0u,1u,malformed);}
 let occupied=currentMetadata[item*10u+2u]==params.currentGeneration;
 var available=!occupied;if(occupied){let key=currentMetadata[item*10u+1u];let removed=!desiredContains(key);
  available=removed;pageDelta[rollbackPagesOffset()+item]=select(INVALID,item,removed);}
 pageDelta[supportCandidatesOffset()+item]=select(INVALID,item,available);
}
var<workgroup> identityScanLanes:array<u32,256>;
var<workgroup> identityScanTotal:u32;
fn scanIdentityBlock(local:u32,value:u32)->u32{
 identityScanLanes[local]=value;workgroupBarrier();var step=1u;
 loop{if(step>=256u){break;}let index=(local+1u)*step*2u-1u;
  if(index<256u){identityScanLanes[index]+=identityScanLanes[index-step];}workgroupBarrier();step*=2u;}
 if(local==255u){identityScanTotal=identityScanLanes[255u];identityScanLanes[255u]=0u;}workgroupBarrier();step=128u;
 loop{let index=(local+1u)*step*2u-1u;if(index<256u){let lower=identityScanLanes[index-step];
   identityScanLanes[index-step]=identityScanLanes[index];identityScanLanes[index]+=lower;}workgroupBarrier();
  if(step==1u){break;}step/=2u;}return identityScanLanes[local];
}
var<workgroup> errorOrLanes:array<u32,256>;
var<workgroup> errorFirstLanes:array<u32,256>;
fn reduceErrorBlock(local:u32,value:u32,first:u32){
 errorOrLanes[local]=value;errorFirstLanes[local]=first;workgroupBarrier();var width=128u;
 loop{if(width==0u){break;}if(local<width){errorOrLanes[local]|=errorOrLanes[local+width];
   errorFirstLanes[local]=min(errorFirstLanes[local],errorFirstLanes[local+width]);}
  workgroupBarrier();width>>=1u;}
}
fn topologyErrorBlockCount()->u32{return (params.pageCapacity+255u)/256u;}
fn topologyErrorSuperBlockCount()->u32{return (topologyErrorBlockCount()+255u)/256u;}
fn topologyErrorScratchBlock(block:u32)->u32{return 2u*block;}
fn topologyErrorScratchSuper(block:u32)->u32{return 2u*(topologyErrorBlockCount()+block);}
@compute @workgroup_size(256) fn reduceTopologyErrorRecords(
 @builtin(workgroup_id)wid:vec3u,@builtin(local_invocation_index)local:u32){
 let work=wid.x*256u+local;var value=0u;if(work<params.pageCapacity){value=atomicLoad(&topologyErrors[work]);}
 reduceErrorBlock(local,value,select(INVALID,work,value!=0u));if(local==0u){
  let output=topologyErrorScratchBlock(wid.x);desiredScan[output]=errorOrLanes[0];desiredScan[output+1u]=errorFirstLanes[0];}}
@compute @workgroup_size(256) fn reduceTopologyErrorGroups(
 @builtin(workgroup_id)wid:vec3u,@builtin(local_invocation_index)local:u32){
 let block=wid.x*256u+local;var value=0u;var first=INVALID;if(block<topologyErrorBlockCount()){
  let input=topologyErrorScratchBlock(block);value=desiredScan[input];first=desiredScan[input+1u];}
 reduceErrorBlock(local,value,first);if(local==0u){let output=topologyErrorScratchSuper(wid.x);
  desiredScan[output]=errorOrLanes[0];desiredScan[output+1u]=errorFirstLanes[0];}}
@compute @workgroup_size(256) fn reduceTopologyErrorSuperGroups(@builtin(local_invocation_index)local:u32){
 var value=0u;var first=INVALID;if(local<topologyErrorSuperBlockCount()){
  let input=topologyErrorScratchSuper(local);value=desiredScan[input];first=desiredScan[input+1u];}
 reduceErrorBlock(local,value,first);if(local==0u){desiredScan[0]=errorOrLanes[0];desiredScan[1]=errorFirstLanes[0];}}
@compute @workgroup_size(1) fn validateDesiredSeeds(){let errors=desiredScan[0];if(errors!=0u){control[0]|=errors;control[7]=desiredScan[1];}}
@compute @workgroup_size(256) fn scanIdentityRecords(
 @builtin(workgroup_id)wid:vec3u,@builtin(local_invocation_index)local:u32){
 let stream=wid.y;let item=wid.x*256u+local;let present=item<params.pageCapacity&&identityCandidate(stream,item);
 let prefix=scanIdentityBlock(local,select(0u,1u,present));if(item<params.pageCapacity){writeIdentityPrefix(stream,item,prefix);}
 if(local==255u){pageDelta[identityScanScratch(stream,wid.x)]=identityScanTotal;}
}
@compute @workgroup_size(256) fn scanIdentityGroups(
 @builtin(workgroup_id)wid:vec3u,@builtin(local_invocation_index)local:u32){
 let stream=wid.y;let item=wid.x*256u+local;let blocks=identityBlockCount();
 var value=0u;if(item<blocks){value=pageDelta[identityScanScratch(stream,item)];}
 let prefix=scanIdentityBlock(local,value);if(item<blocks){pageDelta[identityScanScratch(stream,item)]=prefix;}
 if(local==255u){pageDelta[identityScanScratch(stream,blocks+wid.x)]=identityScanTotal;}
}
@compute @workgroup_size(256) fn scanIdentitySuperGroups(
 @builtin(workgroup_id)wid:vec3u,@builtin(local_invocation_index)local:u32){
 let stream=wid.y;let blocks=identityBlockCount();let count=identitySuperBlockCount();
 var value=0u;if(local<count){value=pageDelta[identityScanScratch(stream,blocks+local)];}
 let prefix=scanIdentityBlock(local,value);if(local<count){pageDelta[identityScanScratch(stream,blocks+local)]=prefix;}
}
@compute @workgroup_size(256) fn offsetIdentityGroups(
 @builtin(workgroup_id)wid:vec3u,@builtin(local_invocation_index)local:u32){
 let stream=wid.y;let item=wid.x*256u+local;let blocks=identityBlockCount();if(item>=blocks){return;}
 pageDelta[identityScanScratch(stream,item)]+=pageDelta[identityScanScratch(stream,blocks+wid.x)];
}
@compute @workgroup_size(256) fn offsetIdentityRecords(
 @builtin(workgroup_id)wid:vec3u,@builtin(local_invocation_index)local:u32){
 let stream=wid.y;let item=wid.x*256u+local;if(item>=params.pageCapacity){return;}
 writeIdentityPrefix(stream,item,identityPrefix(stream,item)+pageDelta[identityScanScratch(stream,wid.x)]);
}
fn identityTotal(stream:u32,count:u32)->u32{
 if(count==0u){return 0u;}let last=count-1u;return identityPrefix(stream,last)+select(0u,1u,identityCandidate(stream,last));
}
@compute @workgroup_size(1) fn prepareDesiredPageIdentityAssignment(){
 let desiredCount=control[2];let currentCount=currentWorklist[0];
 if(control[0]!=0u){return;}if(desiredCount>params.pageCapacity||currentCount>params.pageCapacity||!currentFinePublished()){
  control[0]|=MALFORMED;return;}
 let added=identityTotal(0u,desiredCount);let retired=identityTotal(1u,params.pageCapacity);
 let available=identityTotal(2u,params.pageCapacity);let malformed=identityTotal(3u,params.pageCapacity);
 if(malformed!=0u){control[0]|=MALFORMED;return;}if(added>available){control[0]|=CAPACITY;control[6]=desiredCount;return;}
 pageDelta[11]=added;pageDelta[12]=retired;pageDelta[13]=desiredCount+retired;pageDelta[15]=VALID;
}
@compute @workgroup_size(64) fn compactDesiredPageIdentities(
 @builtin(workgroup_id)wid:vec3u,@builtin(num_workgroups)nwg:vec3u,@builtin(local_invocation_index)local:u32){
 let item=linearInvocation(wid,nwg,local);if(item>=params.pageCapacity||control[0]!=0u){return;}
 let retired=pageDelta[rollbackPagesOffset()+item];if(retired!=INVALID){
  let rank=desiredCandidates[params.pageCapacity+item];pageDelta[retiredPagesOffset()+rank]=retired;
  pageDelta[changedCandidatesOffset()+control[2]+rank]=currentMetadata[retired*10u+1u];}
 let available=pageDelta[supportCandidatesOffset()+item];if(available!=INVALID){
  pageDelta[supportPagesOffset()+pageDelta[dirtyCandidatesOffset()+item]]=available;}
 desiredCandidates[params.pageCapacity+item]=0u;pageDelta[changedKeysOffset()+item]=INVALID;
 if(item>=control[2]){desiredCandidates[item]=0u;}
}
@compute @workgroup_size(64) fn assignDesiredPageIdentities(
 @builtin(workgroup_id)wid:vec3u,@builtin(num_workgroups)nwg:vec3u,@builtin(local_invocation_index)local:u32){
 let work=linearInvocation(wid,nwg,local);let desiredCount=min(control[2],params.pageCapacity);
 if(work>=desiredCount||control[0]!=0u||pageDelta[15]!=VALID){return;}let key=pageDelta[desiredKeysOffset()+work];
 let old=carriedPage(key);if(old!=INVALID){let base=old*10u;for(var word=0u;word<10u;word+=1u){sourceC[base+word]=currentMetadata[base+word];}
  sourceC[base+2u]=params.nextGeneration;sourceD[5u+work]=old;sourceD[5u+params.pageCapacity+key]=old;
 }else{let rank=desiredCandidates[work];let id=pageDelta[supportPagesOffset()+rank];let base=id*10u;
  sourceC[base]=id;sourceC[base+1u]=key;sourceC[base+2u]=params.nextGeneration;sourceC[base+3u]=1u;
  for(var direction=0u;direction<6u;direction+=1u){sourceC[base+4u+direction]=INVALID;}
  sourceD[5u+work]=id;sourceD[5u+params.pageCapacity+key]=id;pageDelta[addedPagesOffset()+rank]=id;}
 desiredCandidates[work]=0u;
}
@compute @workgroup_size(1) fn finalizeDesiredPageIdentityAssignment(){
 if(control[0]!=0u||pageDelta[15]!=VALID){return;}let desiredCount=control[2];
 sourceD[0]=desiredCount;sourceD[1]=params.nextGeneration;sourceD[2]=(desiredCount+63u)/64u;sourceD[3]=0u;sourceD[4]=0u;
 let desiredGroups=(desiredCount+63u)/64u;let changedGroups=(pageDelta[13]+63u)/64u;
 writeDeltaDispatch(lifecycleDispatchOffset(),desiredGroups);writeIndirectDispatch(0u,desiredGroups);
 writeDeltaDispatch(lifecycleDispatchOffset()+3u,changedGroups);writeIndirectDispatch(1u,changedGroups);
 writeDeltaDispatch(lifecycleDispatchOffset()+6u,pageDelta[11]);writeIndirectDispatch(2u,pageDelta[11]);
}
@compute @workgroup_size(64) fn classifyFinePageDelta(@builtin(workgroup_id) wid:vec3u,@builtin(num_workgroups) nwg:vec3u,@builtin(local_invocation_index) local:u32){let work=linearInvocation(wid,nwg,local);if(control[0]!=0u||pageDelta[15]!=VALID){return;}
 let nextCount=min(control[2],params.pageCapacity);if(work>=nextCount){return;}let id=sourceD[5u+work];if(id>=params.pageCapacity||sourceC[id*10u+2u]!=params.nextGeneration){atomicOr(&topologyErrors[work],MALFORMED);return;}let key=sourceC[id*10u+1u];
 if(currentMetadata[id*10u+1u]==key&&currentMetadata[id*10u+2u]==params.currentGeneration&&producerChangedContains(key)){pageDelta[changedCandidatesOffset()+work]=key;}}
@compute @workgroup_size(64) fn compactFineChangedKeys(@builtin(workgroup_id)wid:vec3u,@builtin(num_workgroups)nwg:vec3u,@builtin(local_invocation_index)local:u32){
 let item=linearInvocation(wid,nwg,local);if(control[0]!=0u){return;}let desired=min(control[2],params.pageCapacity);
 let retired=min(pageDelta[12],params.pageCapacity);let changedDesired=identityTotal(0u,desired);
 if(item<desired){let key=pageDelta[changedCandidatesOffset()+item];if(key!=INVALID){pageDelta[changedKeysOffset()+desiredCandidates[item]]=key;}}
 if(item<retired){let id=pageDelta[retiredPagesOffset()+item];if(id<params.pageCapacity&&currentMetadata[id*10u+2u]==params.currentGeneration){
  pageDelta[changedKeysOffset()+changedDesired+item]=currentMetadata[id*10u+1u];}}
 if(item<desired){pageDelta[changedCandidatesOffset()+item]=INVALID;}
}
@compute @workgroup_size(1) fn prepareFinePageDeltaExpansion(){if(control[0]!=0u){return;}let desired=min(control[2],params.pageCapacity);
 let changedDesired=identityTotal(0u,desired);let changed=changedDesired+min(pageDelta[12],params.pageCapacity);
 pageDelta[0]=changed;pageDelta[10]=1u;pageDelta[13]=changedDesired;
 writeDeltaDispatch(lifecycleDispatchOffset()+9u,(changed+63u)/64u);
 let affectedGroups=(max(desired,min(pageDelta[12],params.pageCapacity))+63u)/64u;
 writeDeltaDispatch(lifecycleDispatchOffset()+12u,affectedGroups);writeIndirectDispatch(3u,affectedGroups);}
fn sortedChangedStreamContains(key:u32,first:u32,count:u32)->bool{var low=0u;var high=count;
 while(low<high){let middle=low+(high-low)/2u;let stored=pageDelta[changedKeysOffset()+first+middle];
  if(stored<key){low=middle+1u;}else{high=middle;}}
 return low<count&&pageDelta[changedKeysOffset()+first+low]==key;}
fn changedNeighborRadii(key:u32)->vec2u{
 if(key>=desiredLogicalCount()){return vec2u(0u);}let origin=vec3i(unpackBrick(key));
 let firstCount=min(pageDelta[13],params.pageCapacity);
 let total=min(pageDelta[0],2u*params.pageCapacity);var dirty=0u;var support=0u;
 for(var item=0u;item<total&&(dirty==0u||support==0u);item+=1u){
  let changedKey=pageDelta[changedKeysOffset()+item];
  if(changedKey>=desiredLogicalCount()){continue;}
  let delta=abs(origin-vec3i(unpackBrick(changedKey)));
  let distance=max(delta.x,max(delta.y,delta.z));
  dirty|=select(0u,1u,distance<=i32(params.dirtyHaloRings));
  support|=select(0u,1u,distance<=i32(params.supportHaloRings));
 }
 return vec2u(dirty,support);
}
@compute @workgroup_size(64) fn classifyFineAffectedPages(@builtin(workgroup_id)wid:vec3u,@builtin(num_workgroups)nwg:vec3u,@builtin(local_invocation_index)local:u32){
 let work=linearInvocation(wid,nwg,local);if(control[0]!=0u||pageDelta[15]!=VALID){return;}
 let desired=min(control[2],params.pageCapacity);let retired=min(pageDelta[12],params.pageCapacity);
 if(work<desired){let id=sourceD[5u+work];var error=0u;var key=INVALID;
  if(id>=params.pageCapacity||sourceC[id*10u+2u]!=params.nextGeneration){error=MALFORMED;}
  else{key=sourceC[id*10u+1u];if(key>=desiredLogicalCount()){error=MALFORMED;}}
  let affected=select(vec2u(0u),changedNeighborRadii(key),error==0u);
  let dirty=affected.x!=0u;let support=affected.y!=0u;
  pageDelta[dirtyCandidatesOffset()+work]=select(INVALID,id,dirty);
  pageDelta[supportCandidatesOffset()+work]=select(INVALID,id,support);
  let carried=error==0u&&currentMetadata[id*10u+1u]==key&&currentMetadata[id*10u+2u]==params.currentGeneration;
  pageDelta[changedCandidatesOffset()+work]=select(INVALID,id,dirty&&carried);
  if(error!=0u){atomicOr(&topologyErrors[work],error);}}
}
@compute @workgroup_size(64) fn compactFineAffectedPages(@builtin(workgroup_id)wid:vec3u,@builtin(num_workgroups)nwg:vec3u,@builtin(local_invocation_index)local:u32){
 let item=linearInvocation(wid,nwg,local);if(control[0]!=0u||pageDelta[15]!=VALID){return;}
 let desired=min(control[2],params.pageCapacity);let retired=min(pageDelta[12],params.pageCapacity);
 let carriedDirty=identityTotal(2u,desired);
 if(item<desired){let dirty=pageDelta[dirtyCandidatesOffset()+item];if(dirty!=INVALID){
   pageDelta[dirtyPagesOffset()+identityPrefix(0u,item)]=dirty;}
  let support=pageDelta[supportCandidatesOffset()+item];if(support!=INVALID){
   pageDelta[supportPagesOffset()+identityPrefix(1u,item)]=support;}
  let rollback=pageDelta[changedCandidatesOffset()+item];if(rollback!=INVALID){
   pageDelta[rollbackPagesOffset()+identityPrefix(2u,item)]=rollback;}}
 if(item<retired){let id=pageDelta[retiredPagesOffset()+item];
  if(id<params.pageCapacity){pageDelta[rollbackPagesOffset()+carriedDirty+item]=id;}}
}
@compute @workgroup_size(1) fn finalizeFinePageDelta(){if(control[0]!=0u||pageDelta[15]!=VALID){return;}
 let desired=min(control[2],params.pageCapacity);let retired=min(pageDelta[12],params.pageCapacity);
 let dirty=identityTotal(0u,desired);let support=identityTotal(1u,desired);
 let rollback=identityTotal(2u,desired)+retired;
 // Fixed-pass JFA-CPT rewrites every dirty page, not only pages whose phase
 // membership changed during transport. Publish that exact target set to the
 // carried summary transaction so pressure classification observes the same
 // post-redistance phi. Retired keys remain the trailing stream.
 pageDelta[0]=dirty+retired;pageDelta[2]=dirty;pageDelta[3]=support;
 pageDelta[13]=dirty;pageDelta[14]=rollback;
 writeDeltaDispatch(lifecycleDispatchOffset()+15u,rollback);writeDeltaDispatch(lifecycleDispatchOffset()+18u,dirty);
 writeIndirectDispatch(4u,rollback);writeIndirectDispatch(5u,support);writeIndirectDispatch(6u,(support+63u)/64u);writeIndirectDispatch(7u,dirty);
}
@compute @workgroup_size(64) fn publishFineSummaryChangedKeys(
 @builtin(workgroup_id)wid:vec3u,@builtin(num_workgroups)nwg:vec3u,@builtin(local_invocation_index)local:u32){
 let item=linearInvocation(wid,nwg,local);if(control[0]!=0u||pageDelta[15]!=VALID){return;}
 let dirty=min(pageDelta[2],params.pageCapacity);let retired=min(pageDelta[12],params.pageCapacity);
 if(item<dirty){let id=pageDelta[dirtyPagesOffset()+item];if(id<params.pageCapacity&&sourceC[id*10u+2u]==params.nextGeneration){
   pageDelta[changedKeysOffset()+item]=sourceC[id*10u+1u];}}
 if(item<retired){let id=pageDelta[retiredPagesOffset()+item];if(id<params.pageCapacity&&currentMetadata[id*10u+2u]==params.currentGeneration){
   pageDelta[changedKeysOffset()+dirty+item]=currentMetadata[id*10u+1u];}}
}
@compute @workgroup_size(64) fn snapshotDeltaPayload(@builtin(workgroup_id) wid:vec3u,@builtin(num_workgroups) nwg:vec3u,@builtin(local_invocation_index) local:u32){let work=fineLinearWorkgroup(wid,nwg);let laneActive=work<pageDelta[14]&&control[0]==0u;var error=0u;if(laneActive){let id=pageDelta[rollbackPagesOffset()+work];if(id>=params.pageCapacity||currentMetadata[id*10u+2u]!=params.currentGeneration){error=MALFORMED;}else if(local<params.samplesPerBrick){let index=id*params.samplesPerBrick+local;let value=bitcast<f32>(targetB[index]);if(!finite(value)){error=NONFINITE;}else{payloadSnapshot[index]=value;}}}publishTopologyError(work,local,error,laneActive);}
@compute @workgroup_size(64) fn initializeDesiredSamples(@builtin(workgroup_id) wid:vec3u,@builtin(num_workgroups) nwg:vec3u,@builtin(local_invocation_index) local:u32){let work=fineLinearWorkgroup(wid,nwg);let laneActive=work<pageDelta[11]&&control[0]==0u;var error=0u;if(laneActive){let id=pageDelta[addedPagesOffset()+work];if(id>=params.pageCapacity){error=MALFORMED;}else if(local<params.samplesPerBrick){let key=sourceC[id*10u+1u];let index=id*params.samplesPerBrick+local;if(index<arrayLength(&targetA)&&index<arrayLength(&targetB)){let brick=unpackBrick(key);let coord=localCoord(local);let q=brick*params.brickResolution+coord;if(any(q>=params.sampleDimensions)){targetA[index]=0u;targetB[index]=0u;}else{let position=params.domainOrigin+(vec3f(q)+vec3f(0.5))*params.fineCellWidth;var value=sampleCoarseOctreePhi(position);let seeded=externalSeedPhi(key,(vec3f(q)+vec3f(0.5))/f32(params.fineFactor));if(finite(seeded)){value=seeded;}if(!finite(value)){error=NONFINITE;}else{targetA[index]=VALID|select(0u,16u,value<0.0);targetB[index]=bitcast<u32>(value);}}}}}publishTopologyError(work,local,error,laneActive);}
@compute @workgroup_size(64) fn linkDesiredNeighbors(@builtin(workgroup_id) wid:vec3u,@builtin(num_workgroups) nwg:vec3u,@builtin(local_invocation_index) local:u32){let work=fineLinearWorkgroup(wid,nwg)*64u+local;if(work>=pageDelta[3]||control[0]!=0u){return;}let id=pageDelta[supportPagesOffset()+work];if(id>=params.pageCapacity||sourceC[id*10u+2u]!=params.nextGeneration){atomicOr(&topologyErrors[work],MALFORMED);return;}let coord=unpackBrick(sourceC[id*10u+1u]);for(var direction=0u;direction<6u;direction+=1u){var neighbor=INVALID;if(direction==0u&&coord.x>0u){neighbor=targetLookup(packBrick(coord-vec3u(1,0,0)));}else if(direction==1u&&coord.x+1u<params.brickDimensions.x){neighbor=targetLookup(packBrick(coord+vec3u(1,0,0)));}else if(direction==2u&&coord.y>0u){neighbor=targetLookup(packBrick(coord-vec3u(0,1,0)));}else if(direction==3u&&coord.y+1u<params.brickDimensions.y){neighbor=targetLookup(packBrick(coord+vec3u(0,1,0)));}else if(direction==4u&&coord.z>0u){neighbor=targetLookup(packBrick(coord-vec3u(0,0,1)));}else if(direction==5u&&coord.z+1u<params.brickDimensions.z){neighbor=targetLookup(packBrick(coord+vec3u(0,0,1)));}sourceC[id*10u+4u+direction]=neighbor;}}
fn fineSettlementWorkgroups()->u32{
 if(control[0]==0u){return pageDelta[2];}
 let currentCount=min(currentWorklist[0],params.pageCapacity);
 return max(currentCount,max(min(pageDelta[11],params.pageCapacity),min(pageDelta[14],params.pageCapacity)));
}
fn publishFineSettlementDispatch(){writeIndirectDispatch(0u,fineSettlementWorkgroups());}
@compute @workgroup_size(1) fn finalizeDesiredGeneration(){
 if(control[0]==0u){let errors=desiredScan[0];if(errors!=0u){control[0]|=errors;control[7]=desiredScan[1];}}
 if(control[0]==0u){let count=control[2];sourceD[0]=count;sourceD[1]=params.nextGeneration;sourceD[2]=(count+63u)/64u;
  sourceD[3]=1u;sourceD[4]=1u;control[3]=count;control[4]=select(1u,0u,params.deferPublication!=0u);}
 if(params.deferPublication==0u){publishFineSettlementDispatch();}
}
@compute @workgroup_size(1) fn finalizeFinePublication(){
 let topologyValid=control[0]==0u;let redistanceValid=arrayLength(&redistanceControl)>=4u&&redistanceControl[0]==0u&&(redistanceControl[2]>0u||pageDelta[2]==0u)&&redistanceControl[3]!=0u;
 let volumeValid=arrayLength(&volumeControl)>0u&&volumeControl[0]==0x80000000u;let transportValid=arrayLength(&transportControl)>=4u&&transportControl[3]!=0u;
 if(topologyValid&&redistanceValid&&volumeValid&&transportValid){control[4]=1u;}
 else{control[0]|=16u;control[7]=select(0u,1u,!topologyValid)|select(0u,2u,!redistanceValid)|select(0u,4u,!volumeValid)|select(0u,8u,!transportValid);}
 publishFineSettlementDispatch();
}
@compute @workgroup_size(64) fn settleFinePublication(@builtin(workgroup_id)wid:vec3u,@builtin(num_workgroups)nwg:vec3u,@builtin(local_invocation_index)local:u32){
 let work=fineLinearWorkgroup(wid,nwg);if(control[0]==0u){
  if(work>=pageDelta[2]||local>=params.samplesPerBrick){return;}let id=pageDelta[dirtyPagesOffset()+work];
  let index=id*params.samplesPerBrick+local;let value=bitcast<f32>(sourceD[index]);if(finite(value)){committedPhi[index]=bitcast<u32>(value);}return;}
 let currentPublished=arrayLength(&currentWorklist)>=5u&&currentWorklist[1]==params.currentGeneration&&currentWorklist[3]==1u&&currentWorklist[4]==1u;
 let currentCount=select(0u,min(currentWorklist[0],params.pageCapacity),currentPublished);
 if(work<currentCount){let id=currentWorklist[5u+work];if(id<params.pageCapacity&&currentMetadata[id*10u+2u]==params.currentGeneration){
   if(local<10u){let base=id*10u;sourceC[base+local]=currentMetadata[base+local];if(local==2u){sourceC[base+2u]=params.nextGeneration;}}
   if(local==0u){let key=currentMetadata[id*10u+1u];sourceD[5u+work]=id;if(key<desiredLogicalCount()){sourceD[5u+params.pageCapacity+key]=id;}}}}
 if(work<pageDelta[11]){let id=pageDelta[addedPagesOffset()+work];if(id<params.pageCapacity&&currentMetadata[id*10u+2u]!=params.currentGeneration&&local<10u){sourceC[id*10u+local]=INVALID;}}
 if(work<pageDelta[14]&&local<params.samplesPerBrick){let id=pageDelta[rollbackPagesOffset()+work];let index=id*params.samplesPerBrick+local;
  let value=payloadSnapshot[index];if(finite(value)){targetA[index]=VALID|select(0u,16u,value<0.0);targetB[index]=bitcast<u32>(value);}}
 if(work==0u&&local==0u){sourceD[0]=currentCount;sourceD[1]=params.nextGeneration;sourceD[2]=(currentCount+63u)/64u;
  sourceD[3]=select(0u,1u,currentPublished);sourceD[4]=select(0u,1u,currentPublished);
  control[4]=select(0u,1u,currentPublished);control[5]=1u;}
}
`;
}

export const fineLevelSetLeafSeedWGSL = /* wgsl */ `
const CORE:u32=2u;const RECURRING_SUPPORT:u32=0x80000000u;const INVALID:u32=0xffffffffu;
struct Params { header:vec4u,tail:vec4u,fineDomain:vec4f,scan:vec4u }
struct FineSeedLeaf { originX:u32,originY:u32,originZ:u32,size:u32,flags:u32,pad0:u32,pad1:u32,pad2:u32,phiGradient:vec4f,motion:vec4f }
struct Candidate { row:u32,flags:u32 }
struct BoundaryPhiQuery { liquidCenter:vec4f,airCenter:vec4f }
@group(0) @binding(0) var<uniform> params:Params;
@group(0) @binding(1) var<storage,read> leaves:array<FineSeedLeaf>;
@group(0) @binding(2) var<storage,read> candidates:array<Candidate>;
@group(0) @binding(3) var<storage,read> candidateControl:array<u32>;
@group(0) @binding(4) var<storage,read_write> seeds:array<u32>;
@group(0) @binding(5) var<storage,read> boundaryQueries:array<BoundaryPhiQuery>;
@group(0) @binding(6) var<storage,read> powerFaceControl:array<u32>;
@group(0) @binding(7) var<storage,read_write> scratch:array<u32>;
fn brickDimensions()->vec3u{return vec3u(params.header.z,params.header.w,params.tail.x);}
fn packBrick(coord:vec3u)->u32{let dims=brickDimensions();return coord.x+dims.x*(coord.y+dims.y*coord.z);}
fn sortCapacity()->u32{return params.tail.w;}
fn blockBase()->u32{return 4u*sortCapacity();}
fn metaBase()->u32{return 5u*sortCapacity();}fn seedTagBase()->u32{return 4u+params.tail.y;}
fn seedPlaneBase()->u32{return seedTagBase()+params.tail.y;}
fn loadRecord(index:u32)->vec4u{let base=4u*index;return vec4u(scratch[base],scratch[base+1u],scratch[base+2u],scratch[base+3u]);}
fn storeRecord(index:u32,value:vec4u){let base=4u*index;scratch[base]=value.x;scratch[base+1u]=value.y;scratch[base+2u]=value.z;scratch[base+3u]=value.w;}
fn leafOrigin(leaf:FineSeedLeaf)->vec3u{return vec3u(leaf.originX,leaf.originY,leaf.originZ);}
fn leafFirst(leaf:FineSeedLeaf)->vec3u{return leafOrigin(leaf)*params.header.x/params.header.y;}
fn leafLast(leaf:FineSeedLeaf)->vec3u{let high=(leafOrigin(leaf)+vec3u(max(1u,leaf.size)))*params.header.x-vec3u(1u);return min(high/params.header.y,brickDimensions()-vec3u(1u));}
fn leafCount(leaf:FineSeedLeaf)->u32{let first=leafFirst(leaf);let last=leafLast(leaf);if(any(first>=brickDimensions())||any(last<first)){return 0u;}let extent=last-first+vec3u(1u);return extent.x*extent.y*extent.z;}
fn leafKey(leaf:FineSeedLeaf,local:u32)->u32{let first=leafFirst(leaf);let extent=leafLast(leaf)-first+vec3u(1u);let x=local%extent.x;let yz=local/extent.x;let y=yz%extent.y;let z=yz/extent.y;return packBrick(first+vec3u(x,y,z));}
fn sourceCount()->u32{if(params.scan.y!=0u){return min(candidateControl[0],min(params.scan.x,min(params.scan.w,arrayLength(&candidates))));}return min(candidateControl[0],min(params.scan.x,arrayLength(&leaves)));}
fn sourceRow(index:u32)->vec2u{if(index>=sourceCount()){return vec2u(0u);}if(params.scan.y!=0u){let item=candidates[index];return vec2u(item.row,select(0u,1u,(item.flags&CORE)!=0u&&item.row<arrayLength(&leaves)));}return vec2u(index,select(0u,1u,(leaves[index].flags&CORE)!=0u));}
fn sourceRecordCount(index:u32)->u32{let source=sourceRow(index);if(source.y==0u){return 0u;}return leafCount(leaves[source.x]);}
fn endpointValid()->bool{return arrayLength(&powerFaceControl)>=9u&&powerFaceControl[3]==0u&&powerFaceControl[8]==0x80000000u;}
fn endpointKey(face:u32,ordinal:u32)->vec2u{if(!endpointValid()||face>=min(powerFaceControl[1],min(params.scan.z,arrayLength(&boundaryQueries)))){return vec2u(0u);}let query=boundaryQueries[face];if(query.liquidCenter.w==0.0||query.airCenter.w==0.0||!(params.fineDomain.w>0.0)){return vec2u(0u);}let position=select(query.liquidCenter.xyz,query.airCenter.xyz,ordinal>=8u);if(any(position<params.fineDomain.xyz)){return vec2u(0u);}let local=ordinal&7u;let lattice=(position-params.fineDomain.xyz)/params.fineDomain.w-vec3f(0.5);let q=vec3i(floor(lattice))+vec3i(i32(local&1u),i32((local>>1u)&1u),i32((local>>2u)&1u));let sampleDims=brickDimensions()*params.header.y;if(any(q<vec3i(0))||any(q>=vec3i(sampleDims))){return vec2u(0u);}return vec2u(1u,packBrick(vec3u(q)/params.header.y));}
fn endpointCount()->u32{return select(0u,min(powerFaceControl[1],min(params.scan.z,arrayLength(&boundaryQueries))),endpointValid());}
fn sourceBlockCount()->u32{return max(1u,(params.scan.x+63u)/64u);}
fn endpointBlockCount()->u32{return select(0u,max(1u,(params.scan.z+63u)/64u),params.scan.z>0u);}
fn endpointRecordCount(face:u32)->u32{var count=0u;if(face<endpointCount()){for(var ordinal=0u;ordinal<16u;ordinal+=1u){count+=endpointKey(face,ordinal).x;}}return count;}
var<workgroup> laneCounts:array<u32,64>;var<workgroup> scanValues:array<u32,256>;
@compute @workgroup_size(64) fn clearSeedState(@builtin(global_invocation_id)gid:vec3u){let item=gid.x;if(item==0u){seeds[0]=0u;seeds[1]=0u;scratch[metaBase()]=0u;scratch[metaBase()+1u]=0u;}if(item<sortCapacity()){scratch[item*4u]=INVALID;}}
@compute @workgroup_size(64) fn classifySourceBlocks(@builtin(workgroup_id)wid:vec3u,@builtin(local_invocation_index)lid:u32){let item=wid.x*64u+lid;laneCounts[lid]=sourceRecordCount(item);workgroupBarrier();if(lid==0u){var total=0u;for(var lane=0u;lane<64u;lane+=1u){total+=laneCounts[lane];}scratch[blockBase()+wid.x]=total;}}
@compute @workgroup_size(64) fn classifyEndpointBlocks(@builtin(workgroup_id)wid:vec3u,@builtin(local_invocation_index)lid:u32){let face=wid.x*64u+lid;laneCounts[lid]=endpointRecordCount(face);workgroupBarrier();if(lid==0u){var total=0u;for(var lane=0u;lane<64u;lane+=1u){total+=laneCounts[lane];}scratch[blockBase()+sourceBlockCount()+wid.x]=total;}}
@compute @workgroup_size(256) fn scanSourceBlocks(@builtin(local_invocation_index)lid:u32){let blocks=sourceBlockCount()+endpointBlockCount();let chunk=max(1u,(blocks+255u)/256u);let first=lid*chunk;let last=min(first+chunk,blocks);var subtotal=0u;for(var block=first;block<last;block+=1u){subtotal+=scratch[blockBase()+block];}scanValues[lid]=subtotal;workgroupBarrier();for(var offset=1u;offset<256u;offset<<=1u){var add=0u;if(lid>=offset){add=scanValues[lid-offset];}workgroupBarrier();scanValues[lid]+=add;workgroupBarrier();}var cursor=scanValues[lid]-subtotal;for(var block=first;block<last;block+=1u){let count=scratch[blockBase()+block];scratch[blockBase()+block]=cursor;cursor+=count;}let finalLane=(blocks-1u)/chunk;if(lid==finalLane){scratch[metaBase()]=cursor;scratch[metaBase()+1u]=cursor;if(cursor>sortCapacity()){seeds[1]=1u;}}}
@compute @workgroup_size(64) fn emitSourceRecords(@builtin(workgroup_id)wid:vec3u,@builtin(local_invocation_index)lid:u32){let item=wid.x*64u+lid;let source=sourceRow(item);var count=0u;if(source.y!=0u){count=leafCount(leaves[source.x]);}laneCounts[lid]=count;workgroupBarrier();var localOffset=0u;for(var lane=0u;lane<lid;lane+=1u){localOffset+=laneCounts[lane];}if(count==0u){return;}let leaf=leaves[source.x];var output=scratch[blockBase()+wid.x]+localOffset;for(var local=0u;local<count;local+=1u){if(output<sortCapacity()){storeRecord(output,vec4u(leafKey(leaf,local),source.x,0u,item));}output+=1u;}}
@compute @workgroup_size(64) fn emitEndpointRecords(@builtin(workgroup_id)wid:vec3u,@builtin(local_invocation_index)lid:u32){let face=wid.x*64u+lid;let count=endpointRecordCount(face);laneCounts[lid]=count;workgroupBarrier();var localOffset=0u;for(var lane=0u;lane<lid;lane+=1u){localOffset+=laneCounts[lane];}if(count==0u){return;}var output=scratch[blockBase()+sourceBlockCount()+wid.x]+localOffset;for(var ordinal=0u;ordinal<16u;ordinal+=1u){let endpoint=endpointKey(face,ordinal);if(endpoint.x!=0u){if(output<sortCapacity()){storeRecord(output,vec4u(endpoint.y,INVALID,1u,face*16u+ordinal));}output+=1u;}}}
fn recordGreater(left:vec4u,right:vec4u)->bool{return left.x>right.x||(left.x==right.x&&(left.y>right.y||(left.y==right.y&&(left.z>right.z||(left.z==right.z&&left.w>right.w)))));}
@compute @workgroup_size(256) fn sortSeedRecords(@builtin(local_invocation_index)lid:u32){let count=sortCapacity();for(var width=2u;width<=count;width<<=1u){for(var stride=width>>1u;stride>0u;stride>>=1u){for(var index=lid;index<count;index+=256u){let partner=index^stride;if(partner>index){let left=loadRecord(index);let right=loadRecord(partner);let descending=(index&width)!=0u;if(recordGreater(left,right)!=descending){storeRecord(index,right);storeRecord(partner,left);}}}workgroupBarrier();}}}
fn runStart(index:u32)->bool{let key=loadRecord(index).x;return key!=INVALID&&(index==0u||loadRecord(index-1u).x!=key);}
@compute @workgroup_size(64) fn classifySeedRuns(@builtin(workgroup_id)wid:vec3u,@builtin(local_invocation_index)lid:u32){let index=wid.x*64u+lid;laneCounts[lid]=select(0u,1u,runStart(index));workgroupBarrier();if(lid==0u){var total=0u;for(var lane=0u;lane<64u;lane+=1u){total+=laneCounts[lane];}scratch[blockBase()+wid.x]=total;}}
@compute @workgroup_size(256) fn scanSeedRuns(@builtin(local_invocation_index)lid:u32){let blocks=sortCapacity()/64u;let chunk=(blocks+255u)/256u;let first=lid*chunk;let last=min(first+chunk,blocks);var subtotal=0u;for(var block=first;block<last;block+=1u){subtotal+=scratch[blockBase()+block];}scanValues[lid]=subtotal;workgroupBarrier();for(var offset=1u;offset<256u;offset<<=1u){var add=0u;if(lid>=offset){add=scanValues[lid-offset];}workgroupBarrier();scanValues[lid]+=add;workgroupBarrier();}var cursor=scanValues[lid]-subtotal;for(var block=first;block<last;block+=1u){let count=scratch[blockBase()+block];scratch[blockBase()+block]=cursor;cursor+=count;}let finalLane=(blocks-1u)/chunk;if(lid==finalLane){seeds[0]=min(cursor,params.tail.y);if(cursor>params.tail.y){seeds[1]=1u;}}}
fn writePlane(index:u32,owner:u32){let base=seedPlaneBase()+index*8u;if(owner!=INVALID){let leaf=leaves[owner];seeds[base]=leaf.originX;seeds[base+1u]=leaf.originY;seeds[base+2u]=leaf.originZ;seeds[base+3u]=leaf.size;seeds[base+4u]=bitcast<u32>(leaf.phiGradient.x);seeds[base+5u]=bitcast<u32>(leaf.phiGradient.y);seeds[base+6u]=bitcast<u32>(leaf.phiGradient.z);seeds[base+7u]=bitcast<u32>(leaf.phiGradient.w);}else{seeds[base]=0u;seeds[base+1u]=0u;seeds[base+2u]=0u;seeds[base+3u]=1u;seeds[base+4u]=bitcast<u32>(3.402823e38);seeds[base+5u]=0u;seeds[base+6u]=0u;seeds[base+7u]=0u;}}
fn runHasEndpoint(index:u32,key:u32)->bool{for(var cursor=index;cursor<sortCapacity();cursor+=1u){let record=loadRecord(cursor);if(record.x!=key){break;}if(record.z!=0u){return true;}}return false;}
@compute @workgroup_size(64) fn emitSeedRuns(@builtin(workgroup_id)wid:vec3u,@builtin(local_invocation_index)lid:u32){let index=wid.x*64u+lid;let start=runStart(index);laneCounts[lid]=select(0u,1u,start);workgroupBarrier();if(!start){return;}var localOffset=0u;for(var lane=0u;lane<lid;lane+=1u){localOffset+=laneCounts[lane];}let output=scratch[blockBase()+wid.x]+localOffset;if(output>=params.tail.y){return;}let record=loadRecord(index);seeds[4u+output]=record.x;seeds[seedTagBase()+output]=output|select(0u,RECURRING_SUPPORT,runHasEndpoint(index,record.x));writePlane(output,record.y);}
`;
