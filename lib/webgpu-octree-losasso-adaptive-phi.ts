import type { PassBroker } from "./webgpu-pass-broker";
import type { LosassoSurfaceGraphBankSource } from "./webgpu-octree-losasso-surface-graph";
import {
  octreeLosassoAdaptivePhiAcceptedScheduleWGSL,
  octreeLosassoAdaptivePhiBacktraceWGSL,
  octreeLosassoAdaptivePhiCommitWGSL,
  octreeLosassoAdaptivePhiEvidenceWGSL,
  octreeLosassoAdaptivePhiGhostWGSL,
  octreeLosassoAdaptivePhiHandoffWGSL,
  octreeLosassoAdaptivePhiRedistanceAtoBWGSL,
  octreeLosassoAdaptivePhiRedistanceBtoAWGSL,
  octreeLosassoAdaptivePhiRedistanceFinishWGSL,
  octreeLosassoAdaptivePhiRedistanceInitializeWGSL,
  octreeLosassoAdaptivePhiRedistanceProjectAWGSL,
  octreeLosassoAdaptivePhiRedistanceProjectBWGSL,
  octreeLosassoAdaptivePhiRedistancePublishAcceptedWGSL,
  octreeLosassoAdaptivePhiRedistancePublishCandidateWGSL,
  octreeLosassoAdaptivePhiRedistanceResetWGSL,
  octreeLosassoAdaptivePhiScheduleWGSL,
  octreeLosassoAdaptivePhiTransportWGSL,
  octreeLosassoAdaptivePhiVolumeEvidenceWGSL,
  octreeLosassoAdaptivePhiWorklistConstrainedWGSL,
  octreeLosassoAdaptivePhiWorklistConstraintMarkWGSL,
  octreeLosassoAdaptivePhiWorklistFinalizeWGSL,
  octreeLosassoAdaptivePhiWorklistIndependentWGSL,
  octreeLosassoAdaptivePhiWorklistInflowWGSL,
  octreeLosassoAdaptivePhiWorklistPrepareWGSL,
  octreeLosassoAdaptivePhiWorklistProjectWGSL,
  octreeLosassoAdaptivePhiWorklistReachWGSL,
  octreeLosassoAdaptivePhiWorklistReceiptWGSL,
  octreeLosassoAdaptivePhiWGSL,
} from "./webgpu-octree-losasso-adaptive-phi.wgsl";

export const OCTREE_LOSASSO_ADAPTIVE_PHI_MAGIC = 0x4150_4849;
export const OCTREE_LOSASSO_ADAPTIVE_PHI_CONTROL_WORDS = 20;
export const OCTREE_LOSASSO_ADAPTIVE_PHI_RECEIPT_WORDS = 55;
export const OCTREE_LOSASSO_ADAPTIVE_PHI_RENDERER_HEADER_WORDS = 8;
export const OCTREE_LOSASSO_ADAPTIVE_PHI_RENDERER_ROW_WORDS = 8;

/**
 * Compact one-way renderer layout. Primary records are followed immediately
 * by the LIVE auxiliary-corner records; unused capacity is only tail storage.
 */
export const adaptivePhiRendererDirectoryLayout = (rowCount: number, rowCapacity: number) => {
  if (!Number.isInteger(rowCount) || rowCount < 0) throw new Error("adaptive phi renderer row count must be a non-negative integer");
  if (!Number.isInteger(rowCapacity) || rowCapacity <= 0) throw new Error("adaptive phi renderer row capacity must be a positive integer");
  if (rowCount > rowCapacity) throw new Error("adaptive phi renderer row count exceeds capacity");
  const primaryWordOffset = OCTREE_LOSASSO_ADAPTIVE_PHI_RENDERER_HEADER_WORDS;
  const auxiliaryWordOffset = primaryWordOffset
    + OCTREE_LOSASSO_ADAPTIVE_PHI_RENDERER_ROW_WORDS * rowCount;
  return Object.freeze({
    primaryWordOffset,
    auxiliaryWordOffset,
    liveWordCount: auxiliaryWordOffset
      + OCTREE_LOSASSO_ADAPTIVE_PHI_RENDERER_ROW_WORDS * rowCount,
    allocatedWordCount: OCTREE_LOSASSO_ADAPTIVE_PHI_RENDERER_HEADER_WORDS
      + 2 * OCTREE_LOSASSO_ADAPTIVE_PHI_RENDERER_ROW_WORDS * rowCapacity,
  });
};

/**
 * Structural graph-v2 ABI consumed by the scalar operators.  Buffers are
 * compact and bank-local.  No inverse buffer scales with the finest lattice.
 *
 * This is the exact `LosassoSurfaceGraphBankSource` ABI. Its control begins
 * `[epoch,leafCount,nodeCount,published,error,surfaceGen,velocityGen,...]`;
 * publication means `published == epoch != 0`. Leaves embed eight corners,
 * directories are sorted, constraints contain exact rational numerators, and
 * adjacency is the packed 12-word `negative3,positive3,negativeSpan3,
 * positiveSpan3` record.
 */
export type WebGPUOctreeLosassoAdaptivePhiGraphBankSource = LosassoSurfaceGraphBankSource;

export interface WebGPUOctreeLosassoAdaptivePhiGraphSource {
  readonly accepted: WebGPUOctreeLosassoAdaptivePhiGraphBankSource;
  readonly candidate: WebGPUOctreeLosassoAdaptivePhiGraphBankSource;
}

/** Graph control plus interleaved `[accepted,predictor]` `vec4f` records per node. */
export interface WebGPUOctreeLosassoAdaptiveNodalVelocitySource {
  readonly control: GPUBuffer;
  readonly values: GPUBuffer;
  /** `[generation,physicalReachBits,...,ready@7]`, published by adaptive velocity. */
  readonly receipt: GPUBuffer;
}

export interface WebGPUOctreeLosassoAdaptivePhiInflow {
  readonly outletCenter_m: { readonly x: number; readonly y: number; readonly z: number };
  readonly radius_m: number;
  readonly velocity_m_s: { readonly x: number; readonly y: number; readonly z: number };
  readonly apertureScale: number;
  readonly strength: number;
}

/** Existing compact Losasso face layout; rows must match accepted leaf slots. */
export interface WebGPUOctreeLosassoAdaptivePhiFaceSource {
  /** `[epoch,rowCount,faceCount,valid,...]`. */
  readonly control: GPUBuffer;
  readonly faces: GPUBuffer;
}

export type WebGPUOctreeLosassoAdaptivePhiBootstrap =
  | { readonly kind: "nodal-gpu"; readonly values: GPUBuffer }
  | { readonly kind: "nodal-lattice-gpu"; readonly values: GPUBuffer }
  | { readonly kind: "cell-centred-gpu"; readonly values: GPUBuffer }
  | { readonly kind: "nodal-cpu"; readonly values: Float32Array }
  | { readonly kind: "nodal-lattice-cpu"; readonly values: Float32Array }
  | { readonly kind: "cell-centred-cpu"; readonly values: Float32Array };

export interface WebGPUOctreeLosassoAdaptivePhiSource {
  /** Stable state publication; see `adaptivePhiControlLayout`. */
  readonly control: GPUBuffer;
  /** Stable structural/numerical receipt arena; see `adaptivePhiReceiptLayout`. */
  readonly receipts: GPUBuffer;
  /** Two fixed accepted banks; public objects never rotate. */
  readonly acceptedPhiBanks: GPUBuffer;
  readonly candidatePhi: GPUBuffer;
  /** vec4u(bitcast(centre),bitcast(min),bitcast(max),flags), pressure-row indexed. */
  readonly rowPhi: GPUBuffer;
  /** vec4f(world gradient.xyz, valid), pressure-row indexed. */
  readonly rowGradient: GPUBuffer;
  /** vec4u(bitcast(distance),bitcast(theta),bitcast(airPhi),flags), per face. */
  readonly ghostDistances: GPUBuffer;
  /** Liquid volume owned by every accepted adaptive leaf. */
  readonly physicalVolumes: GPUBuffer;
  /** `[valid,epoch,leafCount,generation,totalVolumeBits,error,...]`. */
  readonly volumePublication: GPUBuffer;
  readonly nodeDispatch: GPUBuffer;
  readonly leafDispatch: GPUBuffer;
  readonly rowDispatch: GPUBuffer;
  readonly faceDispatch: GPUBuffer;
  /** Fixed 12-word atomic counters/indirect records. */
  readonly transportControl: GPUBuffer;
  /** Plain-u32 independent and constrained list halves, each node-capacity sized. */
  readonly transportWorklist: GPUBuffer;
  /** Two graph-slot-indexed vec4f traces: backward predictor then forward reverse estimate. */
  readonly transportDepartures: GPUBuffer;
  /** Diagnostic-only views of graph-sized recurring scratch; no consumer may
   * treat these as published scalar authority. */
  readonly transportBandMask: GPUBuffer;
  readonly redistanceDistanceA: GPUBuffer;
  readonly redistanceDistanceB: GPUBuffer;
  /** Canonical compact adaptive row arena (Losasso `LPHI` ABI), never dense. */
  readonly topologyEvidence: GPUBuffer;
  readonly topologyEvidenceRowCapacity: number;
  /** One-way Power coarse-directory ABI used only by renderer/view consumers. */
  readonly rendererDirectory: GPUBuffer;
}

export const adaptivePhiControlLayout = Object.freeze({
  magic: 0, topologyEpoch: 1, surfaceGeneration: 2, velocityGeneration: 3,
  nodeCount: 4, leafCount: 5, acceptedBank: 6, valid: 7,
  candidateEpoch: 8, candidateGeneration: 9, candidateNodeCount: 10,
  candidateValid: 11, errorBits: 12, redistanceConverged: 13,
  compatibilityMaterializations: 14, targetVolumeBits: 16,
});

export const adaptivePhiReceiptLayout = Object.freeze({
  dtBits: 0, maximumTransportDeltaBits: 1, nodeVelocityFailures: 2,
  midpointVelocityFailures: 3, transportedNodes: 4,
  departurePhiFailures: 5, redistanceResidualBits: 6,
  redistanceReachedNodes: 7, redistanceSeedNodes: 8,
  redistanceVolumeValidatedNodes: 9, signedRedistanceVolumeDriftBits: 10,
  measuredVolumeBits: 11, referenceVolumeDeltaBits: 12,
  targetVolumeBits: 13, redistanceConverged: 14, rendererValidRows: 15,
  retainedNodes: 16, prolongedNodes: 17,
  projectedCandidateNodes: 18, pureTransferValid: 19, jointCommit: 20,
  candidateRepairValid: 21, acceptedAdvanceValid: 22,
  rendererGeneration: 23, candidateRedistanceResidualBits: 24,
  candidateMeasuredVolumeBits: 25, candidateRedistanceReachedNodes: 26,
  candidateRedistanceSeedNodes: 27, candidateValidatedNodes: 28,
  candidateTargetVolumeBits: 29, candidateSignedRedistanceVolumeDriftBits: 30,
  candidateRedistanceConverged: 31,
  scheduledIndependentNodes: 32, retainedIndependentNodes: 33,
  scheduledConstrainedNodes: 34, retainedConstrainedNodes: 35,
  volumeTransactionEpoch: 36, volumeTransactionGeneration: 37,
  volumeValidatedNodes: 38, volumeConstrainedNodes: 39,
  volumeCoveredLeaves: 40,
  volumeMaximumAbsoluteLeafDriftBits: 41, volumeTotalAbsoluteLeafDriftBits: 42,
  volumeTransactionValid: 43,
  candidateVolumeTransactionEpoch: 44, candidateVolumeTransactionGeneration: 45,
  candidateVolumeValidatedNodes: 46, candidateVolumeConstrainedNodes: 47,
  candidateVolumeCoveredLeaves: 48,
  candidateVolumeMaximumAbsoluteLeafDriftBits: 49,
  candidateVolumeTotalAbsoluteLeafDriftBits: 50,
  candidateVolumeTransactionValid: 51,
  redistanceActiveIndependentNodes: 52, redistanceActiveConstrainedNodes: 53,
  /** Fail-closed phase/predicate code. High phase bits distinguish transport
   * (0x10000), volume preparation (0x20000), and volume validation (0x30000). */
  publicationFailureCode: 54,
});

const receiptFloat = (word: number): number => {
  const bits = new Uint32Array(1); bits[0] = word >>> 0;
  return new Float32Array(bits.buffer)[0]!;
};

/** CPU-side decoder for read-back of the stable diagnostic receipt. */
export function unpackAdaptivePhiReceipt(words: ArrayLike<number>) {
  if (words.length < OCTREE_LOSASSO_ADAPTIVE_PHI_RECEIPT_WORDS) {
    throw new RangeError(`adaptive phi receipt requires ${OCTREE_LOSASSO_ADAPTIVE_PHI_RECEIPT_WORDS} words`);
  }
  return Object.freeze({
    dt_s: receiptFloat(words[0]!), maximumTransportDelta_m: receiptFloat(words[1]!),
    nodeVelocityFailures: words[2]! >>> 0, midpointVelocityFailures: words[3]! >>> 0,
    transportedNodes: words[4]! >>> 0, departurePhiFailures: words[5]! >>> 0,
    redistanceResidual_m: receiptFloat(words[6]!), redistanceReachedNodes: words[7]! >>> 0,
    redistanceSeedNodes: words[8]! >>> 0, volumeValidatedNodes: words[9]! >>> 0,
    signedRedistanceVolumeDrift_m3: receiptFloat(words[10]!), measuredVolume_m3: receiptFloat(words[11]!),
    referenceVolumeDelta_m3: receiptFloat(words[12]!), targetVolume_m3: receiptFloat(words[13]!),
    redistanceConverged: words[14] === 1, rendererValidRows: words[15]! >>> 0,
    rendererGeneration: words[23]! >>> 0, acceptedAdvanceValid: words[22] === 1,
    transportBand: Object.freeze({ scheduledIndependentNodes: words[32]! >>> 0,
      retainedIndependentNodes: words[33]! >>> 0,
      scheduledConstrainedNodes: words[34]! >>> 0,
      retainedConstrainedNodes: words[35]! >>> 0 }),
    redistanceBand: Object.freeze({ activeIndependentNodes: words[52]! >>> 0,
      activeConstrainedNodes: words[53]! >>> 0 }),
    publicationFailureCode: words[54]! >>> 0,
    volumeTransaction: Object.freeze({ epoch: words[36]! >>> 0,
      generation: words[37]! >>> 0, validatedNodes: words[38]! >>> 0,
      constrainedNodes: words[39]! >>> 0, coveredLeaves: words[40]! >>> 0,
      maximumAbsoluteLeafDrift_m3: receiptFloat(words[41]!),
      totalAbsoluteLeafDrift_m3: receiptFloat(words[42]!),
      valid: words[43] === 1 }),
    candidateVolumeTransaction: Object.freeze({ epoch: words[44]! >>> 0,
      generation: words[45]! >>> 0, validatedNodes: words[46]! >>> 0,
      constrainedNodes: words[47]! >>> 0, coveredLeaves: words[48]! >>> 0,
      maximumAbsoluteLeafDrift_m3: receiptFloat(words[49]!),
      totalAbsoluteLeafDrift_m3: receiptFloat(words[50]!),
      valid: words[51] === 1 }),
    candidate: Object.freeze({ residual_m: receiptFloat(words[24]!), measuredVolume_m3: receiptFloat(words[25]!),
      reachedNodes: words[26]! >>> 0, seedNodes: words[27]! >>> 0,
      validatedNodes: words[28]! >>> 0, targetVolume_m3: receiptFloat(words[29]!),
      signedRedistanceVolumeDrift_m3: receiptFloat(words[30]!), converged: words[31] === 1 }),
  });
}

export interface WebGPUOctreeLosassoAdaptivePhiOptions {
  readonly nodeCapacity: number;
  readonly leafCapacity: number;
  readonly faceCapacity?: number;
  readonly dimensions: readonly [number, number, number];
  readonly maximumLeafSpan: number;
  readonly cellSize: number;
  readonly domainOrigin?: readonly [number, number, number];
  readonly directoryProbeLimit?: number;
  /** Full A→B→A fast-iterative phases. */
  readonly redistanceIterations?: number;
  readonly redistanceBandWorld?: number;
  /** Uniform-fine diagnostic lane: redistance the connected resident graph
   * instead of deriving a compact mask that cannot save any topology work. */
  readonly fullGraphRedistance?: boolean;
  readonly convergenceTolerance?: number;
  readonly constraintTolerance?: number;
  readonly openTop?: boolean;
  readonly exteriorAirPhi?: number;
  readonly faces?: WebGPUOctreeLosassoAdaptivePhiFaceSource;
}

export interface OctreeLosassoAdaptivePhiPlan {
  readonly nodeCapacity: number;
  readonly leafCapacity: number;
  readonly pressureRowCapacity: number;
  readonly faceCapacity: number;
  readonly nodeDispatch: readonly [number, 1, 1];
  readonly leafDispatch: readonly [number, 1, 1];
  readonly pressureRowDispatch: readonly [number, 1, 1];
  readonly faceDispatch: readonly [number, 1, 1];
  readonly redistanceIterations: number;
  readonly allocatedBytes: number;
  /** No finest-lattice-sized allocation is included. */
  readonly physicsAllocationScalesWithGraph: true;
}

type MainPipelineName = "prepareBootstrap" | "bootstrapNodal" | "bootstrapNodalLattice"
  | "bootstrapCellCentred"
  | "applyReferenceVolumeDelta"
  | "projectAccepted" | "finalizeBootstrap" | "captureReferenceVolume"
  | "measureDerivations" | "captureTransportReceipt"
  | "prepareAdvance" | "prepareCandidateRepair"
  | "projectTransported" | "canonicalizeCandidatePhi"
  | "capturePreRedistanceVolumes"
  | "deriveRows" | "deriveLeafVolumes" | "derivePostRedistanceVolumes"
  | "finalizeAccepted" | "finalizeCandidateRepair";
type VolumeEvidencePipelineName = "prepareVolumeEvidence" | "validateVolumeEvidence"
  | "finalizeVolumeEvidence";
type HandoffPipelineName = "prepareCandidateHandoff" | "handoffCandidate"
  | "projectCandidate" | "finalizeCandidateHandoff";
type WorklistPipelineName = "prepareTransportBand" | "prepareRedistanceBand"
  | "markTransportReach" | "markTransportInflow"
  | "publishTransportIndependent" | "markTransportConstraintMasters"
  | "markTransportConstrained" | "publishTransportConstrained"
  | "finalizeTransportDispatch" | "publishTransportPartition" | "projectTransportedBand";
type RedistancePipelineName = "prepareAcceptedRedistance" | "initializeRedistance"
  | "initializeAcceptedRedistanceIndependent" | "initializeAcceptedRedistanceConstrained"
  | "redistanceAtoB" | "redistanceBtoA" | "projectDistanceA" | "projectDistanceB"
  | "finishRedistance" | "finishAcceptedRedistanceIndependent"
  | "finishAcceptedRedistanceConstrained" | "resetRedistanceResidual"
  | "publishAcceptedRedistanceReceipt" | "publishCandidateRedistanceReceipt";
type EvidencePipelineName = "prepareTopologyEvidenceEpoch" | "prepareTopologyEvidence"
  | "publishTopologyEvidenceRows" | "finishTopologyEvidence";

const positiveInteger = (label: string, value: number): number => {
  if (!Number.isSafeInteger(value) || value < 1) throw new RangeError(`${label} must be a positive integer`);
  return value;
};

/**
 * One compact adaptive nodal-phi authority.  Dense data is accepted only by
 * `encodeBootstrap`; recurring transport, redistance, evidence and derivation
 * bind graph-sized buffers exclusively.
 */
export class WebGPUOctreeLosassoAdaptivePhi {
  readonly plan: OctreeLosassoAdaptivePhiPlan;
  readonly source: WebGPUOctreeLosassoAdaptivePhiSource;
  private readonly params: GPUBuffer;
  private readonly candidateParams: GPUBuffer;
  private readonly advanceParams: GPUBuffer;
  private readonly distanceA: GPUBuffer;
  private readonly distanceB: GPUBuffer;
  private readonly preRedistanceVolumes: GPUBuffer;
  private readonly volumePublication: GPUBuffer;
  private readonly candidateSchedule: GPUBuffer;
  private readonly acceptedScheduleLimits: GPUBuffer;
  private readonly acceptedScheduleControl: GPUBuffer;
  private readonly transportControl: GPUBuffer;
  private readonly transportDispatch: GPUBuffer;
  private readonly transportWorklist: GPUBuffer;
  private readonly transportBandMask: GPUBuffer;
  private readonly transportDepartures: GPUBuffer;
  private readonly redistanceReceipt: GPUBuffer;
  private readonly hostRepairGate: GPUBuffer;
  private readonly referenceDelta: GPUBuffer;
  /** Candidate view materialization. The renderer binds only the accepted
   * directory exposed through `source`; a rejected topology transaction must
   * never clear or partially rewrite that live presentation authority. */
  private readonly candidateRendererDirectory: GPUBuffer;
  private main?: Readonly<Record<MainPipelineName, GPUComputePipeline>>;
  private handoff?: Readonly<Record<HandoffPipelineName, GPUComputePipeline>>;
  private commit?: GPUComputePipeline;
  private stampCandidate?: GPUComputePipeline;
  private syncAccepted?: GPUComputePipeline;
  private stampRepair?: GPUComputePipeline;
  private stampAdvance?: GPUComputePipeline;
  private stampTopologyHandoff?: GPUComputePipeline;
  private ghost?: GPUComputePipeline;
  private evidence?: Readonly<Record<EvidencePipelineName, GPUComputePipeline>>;
  private readonly evidencePrepareWorkgroups: number;
  /** Stable graph arenas never rotate buffer identity. Cache the immutable
   * pipeline/buffer tuples instead of asking WebGPU to validate and allocate
   * the same bind group hundreds of times per accepted step. */
  private readonly bindGroupCache = new Map<GPUComputePipeline, Map<string, GPUBindGroup>>();
  private readonly bufferIdentity = new WeakMap<GPUBuffer, number>();
  private nextBufferIdentity = 1;
  private scheduleSource?: GPUComputePipeline;
  private scheduleRepair?: GPUComputePipeline;
  private scheduleAccepted?: GPUComputePipeline;
  private worklist?: Readonly<Record<WorklistPipelineName, GPUComputePipeline>>;
  private redistance?: Readonly<Record<RedistancePipelineName, GPUComputePipeline>>;
  private backtrace?: GPUComputePipeline;
  private transport?: GPUComputePipeline;
  private volumeEvidence?: Readonly<Record<VolumeEvidencePipelineName, GPUComputePipeline>>;
  private retainedBootstrap?: {
    readonly kind: "nodal" | "nodal-lattice" | "cell-centred";
    readonly values: GPUBuffer;
  };
  private pendingReferenceDelta = 0;
  private bootstrapUpload?: GPUBuffer;
  private destroyed = false;

  constructor(
    private readonly device: GPUDevice,
    readonly graph: WebGPUOctreeLosassoAdaptivePhiGraphSource,
    readonly options: WebGPUOctreeLosassoAdaptivePhiOptions,
  ) {
    const nodeCapacity = positiveInteger("adaptive phi node capacity", options.nodeCapacity);
    const leafCapacity = positiveInteger("adaptive phi leaf capacity", options.leafCapacity);
    const pressureRowCapacity = positiveInteger("adaptive phi pressure row capacity",
      graph.accepted.pressureRowCapacity);
    if (graph.candidate.pressureRowCapacity !== pressureRowCapacity) {
      throw new RangeError("adaptive phi graph banks must share pressure row capacity");
    }
    const faceCapacity = positiveInteger("adaptive phi face capacity", options.faceCapacity ?? 1);
    const redistanceIterations = positiveInteger("adaptive phi redistance iterations",
      options.redistanceIterations ?? Math.max(8, Math.ceil(options.maximumLeafSpan * 2)));
    for (const [axis, extent] of options.dimensions.entries()) positiveInteger(`adaptive phi dimension ${axis}`, extent);
    positiveInteger("adaptive phi maximum leaf span", options.maximumLeafSpan);
    if (!(options.cellSize > 0) || !Number.isFinite(options.cellSize)) throw new RangeError("adaptive phi cell size must be finite and positive");
    const maxWorkgroups = device.limits.maxComputeWorkgroupsPerDimension;
    const dispatch = (count: number) => Math.min(Math.ceil(count / 64), maxWorkgroups);
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    const indirect = storage | GPUBufferUsage.INDIRECT;
    this.params = device.createBuffer({ label: "Adaptive nodal phi parameters", size: 128,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.candidateParams = device.createBuffer({ label: "Adaptive candidate phi parameters", size: 128,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.advanceParams = device.createBuffer({ label: "Adaptive accepted-advance phi parameters", size: 128,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const control = device.createBuffer({ label: "Adaptive nodal phi control", size: 80, usage: storage });
    const receipts = device.createBuffer({ label: "Adaptive nodal phi receipts",
      size: 4 * OCTREE_LOSASSO_ADAPTIVE_PHI_RECEIPT_WORDS, usage: storage });
    this.distanceA = device.createBuffer({ label: "Adaptive Eikonal distance A", size: 4 * nodeCapacity, usage: storage });
    this.distanceB = device.createBuffer({ label: "Adaptive Eikonal distance B", size: 4 * nodeCapacity, usage: storage });
    this.preRedistanceVolumes = device.createBuffer({
      label: "Adaptive pre-redistance leaf volumes", size: 4 * leafCapacity, usage: storage });
    const rowPhi = device.createBuffer({ label: "Adaptive pressure-row phi", size: 16 * pressureRowCapacity, usage: storage });
    const rowGradient = device.createBuffer({ label: "Adaptive pressure-row gradient", size: 16 * pressureRowCapacity, usage: storage });
    const ghostDistances = device.createBuffer({ label: "Adaptive nodal-phi ghost distances", size: 16 * faceCapacity, usage: storage });
    const physicalVolumes = device.createBuffer({ label: "Adaptive leaf liquid volumes", size: 4 * leafCapacity, usage: storage });
    this.volumePublication = device.createBuffer({ label: "Adaptive volume publication", size: 32, usage: storage });
    this.candidateSchedule = device.createBuffer({ label: "Adaptive phi GPU candidate schedule", size: 120,
      usage: storage | GPUBufferUsage.INDIRECT });
    this.acceptedScheduleLimits = device.createBuffer({
      label: "Adaptive phi accepted live schedule limits", size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(this.acceptedScheduleLimits, 0,
      Uint32Array.of(nodeCapacity, leafCapacity, pressureRowCapacity, faceCapacity));
    this.acceptedScheduleControl = device.createBuffer({
      label: "Adaptive phi accepted live schedule receipt", size: 32, usage: storage,
    });
    this.transportControl = device.createBuffer({ label: "Adaptive phi transport control",
      size: 52, usage: storage });
    this.transportDispatch = device.createBuffer({ label: "Adaptive phi transport indirect dispatch",
      size: 24, usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST });
    this.transportWorklist = device.createBuffer({ label: "Adaptive phi plain transport worklist",
      size: 8 * nodeCapacity, usage: storage });
    this.transportBandMask = device.createBuffer({ label: "Adaptive phi transport band mask",
      size: 4 * nodeCapacity, usage: storage });
    this.transportDepartures = device.createBuffer({ label: "Adaptive phi compact departures",
      size: 32 * nodeCapacity, usage: storage });
    this.redistanceReceipt = device.createBuffer({ label: "Adaptive phi private redistance receipt",
      size: 8, usage: storage });
    this.hostRepairGate = device.createBuffer({ label: "Adaptive phi host repair fallback gate", size: 24, usage: storage });
    this.referenceDelta = device.createBuffer({ label: "Adaptive phi pending reference-volume delta", size: 4, usage: storage });
    const nodeDispatch = device.createBuffer({ label: "Adaptive phi node dispatch", size: 12, usage: indirect });
    const leafDispatch = device.createBuffer({ label: "Adaptive phi leaf dispatch", size: 12, usage: indirect });
    const rowDispatch = device.createBuffer({ label: "Adaptive phi pressure-row dispatch", size: 12, usage: indirect });
    const faceDispatch = device.createBuffer({ label: "Adaptive phi face dispatch", size: 12, usage: indirect });
    let evidenceDirectoryCapacity = 1; while (evidenceDirectoryCapacity < 2 * leafCapacity) evidenceDirectoryCapacity *= 2;
    const topologyEvidence = device.createBuffer({ label: "Adaptive phi canonical topology evidence",
      // Header + row records + hash directory + eight raw nodal phi values per
      // row. Topology uses the corner suffix to localize a coarse owner's
      // interval when deciding whether one of its children really crosses.
      size: 4 * (20 + 16 * leafCapacity + 4 * evidenceDirectoryCapacity), usage: storage });
    const rendererLayout = adaptivePhiRendererDirectoryLayout(leafCapacity, leafCapacity);
    const rendererDirectory = device.createBuffer({ label: "Adaptive phi one-way renderer directory",
      // Primary Power-directory row followed by one raw eight-corner record per
      // live row.  This remains graph-sized and is a view-only materialization.
      size: 4 * rendererLayout.allocatedWordCount, usage: storage });
    this.candidateRendererDirectory = device.createBuffer({
      label: "Adaptive phi candidate renderer directory",
      size: rendererDirectory.size,
      usage: storage,
    });
    this.evidencePrepareWorkgroups = dispatch(Math.max(
      evidenceDirectoryCapacity, rendererLayout.allocatedWordCount));
    const emptyDispatch = Uint32Array.of(0, 1, 1);
    device.queue.writeBuffer(nodeDispatch, 0, emptyDispatch);
    device.queue.writeBuffer(leafDispatch, 0, emptyDispatch);
    device.queue.writeBuffer(rowDispatch, 0, emptyDispatch);
    device.queue.writeBuffer(faceDispatch, 0, emptyDispatch);
    this.source = Object.freeze({ control, receipts, acceptedPhiBanks: graph.accepted.phi,
      candidatePhi: graph.candidate.phi,
      rowPhi, rowGradient, ghostDistances, physicalVolumes,
      volumePublication: this.volumePublication, nodeDispatch, leafDispatch, rowDispatch, faceDispatch,
      transportControl: this.transportControl, transportWorklist: this.transportWorklist,
      transportDepartures: this.transportDepartures, transportBandMask: this.transportBandMask,
      redistanceDistanceA: this.distanceA, redistanceDistanceB: this.distanceB,
      topologyEvidence, topologyEvidenceRowCapacity: leafCapacity, rendererDirectory });
    const bytes = 3 * 128 + 80 + 4 * OCTREE_LOSASSO_ADAPTIVE_PHI_RECEIPT_WORDS
      + 4 * nodeCapacity * 13 + 4 * leafCapacity * 3 + 52 + 24 + 8
      + 16 * pressureRowCapacity * 2 + 4 * leafCapacity + 16 * faceCapacity + 32 + 48
      + topologyEvidence.size + rendererDirectory.size
      + this.candidateRendererDirectory.size + 196;
    this.plan = Object.freeze({ nodeCapacity, leafCapacity, pressureRowCapacity, faceCapacity,
      nodeDispatch: [dispatch(nodeCapacity), 1, 1] as const,
      leafDispatch: [dispatch(leafCapacity), 1, 1] as const,
      pressureRowDispatch: [dispatch(pressureRowCapacity), 1, 1] as const,
      faceDispatch: [dispatch(faceCapacity), 1, 1] as const,
      redistanceIterations, allocatedBytes: bytes, physicsAllocationScalesWithGraph: true as const });
    this.writeParams(this.params, 0, undefined, false);
    this.writeParams(this.candidateParams, 0, undefined, true);
    this.writeParams(this.advanceParams, 0, undefined, false);
  }

  get initializationTasks(): readonly { readonly label: string; readonly run: () => Promise<void> }[] {
    return [{ label: "Compile fully adaptive Losasso nodal-phi operators", run: () => this.initialize() }];
  }

  async initialize(): Promise<void> {
    this.assertLive();
    if (this.main) return;
    const compile = async <Name extends string>(module: GPUShaderModule, names: readonly Name[]) => {
      const result = {} as Record<Name, GPUComputePipeline>;
      for (const name of names) result[name] = await this.device.createComputePipelineAsync({
          label: `Adaptive phi - ${name}`, layout: "auto", compute: { module, entryPoint: name },
        });
      return Object.freeze(result);
    };
    const mainModule = this.device.createShaderModule({ label: "Adaptive nodal phi operators", code: octreeLosassoAdaptivePhiWGSL });
    const mainNames: readonly MainPipelineName[] = ["prepareBootstrap", "bootstrapNodal", "bootstrapNodalLattice", "bootstrapCellCentred", "applyReferenceVolumeDelta",
      "projectAccepted", "finalizeBootstrap", "captureReferenceVolume", "measureDerivations", "prepareAdvance", "captureTransportReceipt",
      "prepareCandidateRepair", "canonicalizeCandidatePhi",
      "projectTransported", "capturePreRedistanceVolumes",
      "deriveRows", "deriveLeafVolumes", "derivePostRedistanceVolumes",
      "finalizeAccepted", "finalizeCandidateRepair"];
    this.main = await compile(mainModule, mainNames);
    const volumeEvidenceModule = this.device.createShaderModule({
      label: "Adaptive pre/post-redistance volume evidence gate",
      code: octreeLosassoAdaptivePhiVolumeEvidenceWGSL,
    });
    this.volumeEvidence = await compile(volumeEvidenceModule,
      ["prepareVolumeEvidence", "validateVolumeEvidence", "finalizeVolumeEvidence"] as const);
    const prepareModule = this.device.createShaderModule({ label: "Adaptive phi transport-band prepare", code: octreeLosassoAdaptivePhiWorklistPrepareWGSL });
    const reachModule = this.device.createShaderModule({ label: "Adaptive phi transport-band reach", code: octreeLosassoAdaptivePhiWorklistReachWGSL });
    const inflowModule = this.device.createShaderModule({ label: "Adaptive phi transport-band inflow", code: octreeLosassoAdaptivePhiWorklistInflowWGSL });
    const independentModule = this.device.createShaderModule({ label: "Adaptive phi transport-band independent", code: octreeLosassoAdaptivePhiWorklistIndependentWGSL });
    const constrainedModule = this.device.createShaderModule({ label: "Adaptive phi transport-band constrained", code: octreeLosassoAdaptivePhiWorklistConstrainedWGSL });
    const constraintMarkModule = this.device.createShaderModule({ label: "Adaptive phi transport-band constraint mark", code: octreeLosassoAdaptivePhiWorklistConstraintMarkWGSL });
    const finalizeModule = this.device.createShaderModule({ label: "Adaptive phi transport-band finalize", code: octreeLosassoAdaptivePhiWorklistFinalizeWGSL });
    const receiptModule = this.device.createShaderModule({ label: "Adaptive phi transport-band receipt", code: octreeLosassoAdaptivePhiWorklistReceiptWGSL });
    const projectModule = this.device.createShaderModule({ label: "Adaptive phi transport-band project", code: octreeLosassoAdaptivePhiWorklistProjectWGSL });
    this.worklist = Object.freeze({
      prepareTransportBand: (await compile(prepareModule, ["prepareTransportBand"] as const)).prepareTransportBand,
      prepareRedistanceBand: (await compile(prepareModule,
        ["prepareRedistanceBand"] as const)).prepareRedistanceBand,
      markTransportReach: (await compile(reachModule, ["markTransportReach"] as const)).markTransportReach,
      markTransportInflow: (await compile(inflowModule, ["markTransportInflow"] as const)).markTransportInflow,
      publishTransportIndependent: (await compile(independentModule,
        ["publishTransportIndependent"] as const)).publishTransportIndependent,
      markTransportConstraintMasters: (await compile(constraintMarkModule,
        ["markTransportConstraintMasters"] as const)).markTransportConstraintMasters,
      markTransportConstrained: (await compile(constraintMarkModule,
        ["markTransportConstrained"] as const)).markTransportConstrained,
      publishTransportConstrained: (await compile(constrainedModule,
        ["publishTransportConstrained"] as const)).publishTransportConstrained,
      finalizeTransportDispatch: (await compile(finalizeModule,
        ["finalizeTransportDispatch"] as const)).finalizeTransportDispatch,
      publishTransportPartition: (await compile(receiptModule,
        ["publishTransportPartition"] as const)).publishTransportPartition,
      projectTransportedBand: (await compile(projectModule, ["projectTransportedBand"] as const)).projectTransportedBand,
    });
    const initializeRedistanceModule = this.device.createShaderModule({
      code: octreeLosassoAdaptivePhiRedistanceInitializeWGSL });
    const initializeRedistance = await compile(initializeRedistanceModule,
      ["initializeRedistance", "initializeAcceptedRedistanceIndependent",
        "initializeAcceptedRedistanceConstrained"] as const);
    const resetRedistanceModule = this.device.createShaderModule({
      code: octreeLosassoAdaptivePhiRedistanceResetWGSL });
    const resetRedistance = await compile(resetRedistanceModule,
      ["prepareAcceptedRedistance", "resetRedistanceResidual"] as const);
    const finishRedistanceModule = this.device.createShaderModule({
      code: octreeLosassoAdaptivePhiRedistanceFinishWGSL });
    const finishRedistance = await compile(finishRedistanceModule,
      ["finishRedistance", "finishAcceptedRedistanceIndependent",
        "finishAcceptedRedistanceConstrained"] as const);
    this.redistance = Object.freeze({
      ...initializeRedistance, ...resetRedistance, ...finishRedistance,
      redistanceAtoB: (await compile(this.device.createShaderModule({
        code: octreeLosassoAdaptivePhiRedistanceAtoBWGSL }), ["redistanceAtoB"] as const)).redistanceAtoB,
      redistanceBtoA: (await compile(this.device.createShaderModule({
        code: octreeLosassoAdaptivePhiRedistanceBtoAWGSL }), ["redistanceBtoA"] as const)).redistanceBtoA,
      projectDistanceA: (await compile(this.device.createShaderModule({
        code: octreeLosassoAdaptivePhiRedistanceProjectAWGSL }), ["projectDistanceA"] as const)).projectDistanceA,
      projectDistanceB: (await compile(this.device.createShaderModule({
        code: octreeLosassoAdaptivePhiRedistanceProjectBWGSL }), ["projectDistanceB"] as const)).projectDistanceB,
      publishAcceptedRedistanceReceipt: (await compile(this.device.createShaderModule({
        code: octreeLosassoAdaptivePhiRedistancePublishAcceptedWGSL }),
      ["publishAcceptedRedistanceReceipt"] as const)).publishAcceptedRedistanceReceipt,
      publishCandidateRedistanceReceipt: (await compile(this.device.createShaderModule({
        code: octreeLosassoAdaptivePhiRedistancePublishCandidateWGSL }),
      ["publishCandidateRedistanceReceipt"] as const)).publishCandidateRedistanceReceipt,
    } satisfies Record<RedistancePipelineName, GPUComputePipeline>);
    const backtraceModule = this.device.createShaderModule({ label: "Adaptive phi compact backtrace", code: octreeLosassoAdaptivePhiBacktraceWGSL });
    this.backtrace = (await compile(backtraceModule,
      ["backtraceIndependent"] as const)).backtraceIndependent;
    const transportModule = this.device.createShaderModule({ label: "Adaptive phi compact scalar transport", code: octreeLosassoAdaptivePhiTransportWGSL });
    this.transport = (await compile(transportModule, ["transportIndependent"] as const)).transportIndependent;
    const handoffModule = this.device.createShaderModule({ label: "Adaptive nodal phi topology handoff", code: octreeLosassoAdaptivePhiHandoffWGSL });
    this.handoff = await compile(handoffModule, ["prepareCandidateHandoff", "handoffCandidate", "projectCandidate", "finalizeCandidateHandoff"] as const);
    const commitModule = this.device.createShaderModule({ label: "Adaptive nodal phi joint commit", code: octreeLosassoAdaptivePhiCommitWGSL });
    const commitPipelines = await compile(commitModule, ["commitCandidate", "stampCandidateBootstrap", "syncAcceptedCommit", "stampCandidateRepair", "stampAcceptedAdvance", "publishTopologyHandoff"] as const);
    this.commit = commitPipelines.commitCandidate; this.stampCandidate = commitPipelines.stampCandidateBootstrap;
    this.syncAccepted = commitPipelines.syncAcceptedCommit;
    this.stampRepair = commitPipelines.stampCandidateRepair;
    this.stampAdvance = commitPipelines.stampAcceptedAdvance;
    this.stampTopologyHandoff = commitPipelines.publishTopologyHandoff;
    const ghostModule = this.device.createShaderModule({ label: "Adaptive nodal phi pressure ghosts", code: octreeLosassoAdaptivePhiGhostWGSL });
    this.ghost = (await compile(ghostModule, ["deriveGhosts"] as const)).deriveGhosts;
    const evidenceModule = this.device.createShaderModule({ label: "Adaptive phi topology evidence", code: octreeLosassoAdaptivePhiEvidenceWGSL });
    this.evidence = await compile(evidenceModule,
      ["prepareTopologyEvidenceEpoch", "prepareTopologyEvidence", "publishTopologyEvidenceRows",
        "finishTopologyEvidence"] as const);
    const scheduleModule = this.device.createShaderModule({ label: "Adaptive phi GPU candidate schedule", code: octreeLosassoAdaptivePhiScheduleWGSL });
    const schedule = await compile(scheduleModule, ["scheduleCandidateSource", "scheduleCandidateRepair"] as const);
    this.scheduleSource = schedule.scheduleCandidateSource; this.scheduleRepair = schedule.scheduleCandidateRepair;
    const acceptedScheduleModule = this.device.createShaderModule({
      label: "Adaptive phi GPU accepted live schedule",
      code: octreeLosassoAdaptivePhiAcceptedScheduleWGSL,
    });
    this.scheduleAccepted = (await compile(acceptedScheduleModule,
      ["scheduleAcceptedWork"] as const)).scheduleAcceptedWork;
  }

  /** Cold publication. CPU inputs are uploaded once and can be released after submission. */
  encodeBootstrap(broker: PassBroker, input: WebGPUOctreeLosassoAdaptivePhiBootstrap): void {
    this.encodeBootstrapBank(broker, input, this.graph.accepted, false);
  }

  /** Cold-start bootstrap into the candidate graph before its first ready commit. */
  encodeCandidateBootstrap(broker: PassBroker, input: WebGPUOctreeLosassoAdaptivePhiBootstrap): void {
    this.encodeBootstrapBank(broker, input, this.graph.candidate, true);
  }

  private encodeBootstrapBank(broker: PassBroker, input: WebGPUOctreeLosassoAdaptivePhiBootstrap,
    bank: WebGPUOctreeLosassoAdaptivePhiGraphBankSource, candidate: boolean): void {
    this.assertInitialized();
    let values: GPUBuffer;
    if (input.kind.endsWith("-cpu")) {
      const cpu = input.values as Float32Array;
      const expected = input.kind === "nodal-cpu" ? this.plan.nodeCapacity
        : input.kind === "nodal-lattice-cpu"
          ? this.options.dimensions.reduce((product, value) => product * (value + 1), 1)
          : this.options.dimensions.reduce((product, value) => product * value, 1);
      if (cpu.length !== expected) throw new RangeError(`adaptive phi ${input.kind} bootstrap requires ${expected} values`);
      this.bootstrapUpload?.destroy();
      this.bootstrapUpload = this.device.createBuffer({ label: "One-time adaptive phi bootstrap upload", size: Math.max(4, cpu.byteLength),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      this.device.queue.writeBuffer(this.bootstrapUpload, 0, cpu.slice().buffer);
      values = this.bootstrapUpload;
    } else values = input.values as GPUBuffer;
    const bootstrapKind = input.kind.startsWith("nodal-lattice")
      ? "nodal-lattice" as const : input.kind.startsWith("nodal")
        ? "nodal" as const : "cell-centred" as const;
    if (candidate) this.retainedBootstrap = { kind: bootstrapKind, values };
    const parameterBuffer = candidate ? this.candidateParams : this.params;
    this.runMain(broker, "prepareBootstrap", this.mainBuffers(bank, undefined, undefined, parameterBuffer), 1);
    const entryPoint = bootstrapKind === "nodal-lattice" ? "bootstrapNodalLattice"
      : bootstrapKind === "nodal" ? "bootstrapNodal" : "bootstrapCellCentred";
    this.runMain(broker, entryPoint,
      this.mainBuffers(bank, values, undefined, parameterBuffer), this.plan.nodeDispatch[0]);
    this.runMain(broker, "projectAccepted", this.mainBuffers(bank, undefined, undefined, parameterBuffer), this.plan.nodeDispatch[0]);
    this.runMain(broker, "deriveRows", this.mainBuffers(bank, undefined, undefined, parameterBuffer), this.plan.pressureRowDispatch[0]);
    this.runMain(broker, "deriveLeafVolumes", this.mainBuffers(bank, undefined, undefined, parameterBuffer), this.plan.leafDispatch[0]);
    this.runMain(broker, "captureReferenceVolume", this.mainBuffers(bank, undefined, undefined, parameterBuffer), 1);
    this.runMain(broker, "finalizeBootstrap", this.mainBuffers(bank, undefined, undefined, parameterBuffer), 1);
    if (candidate) this.run(broker, this.stampCandidate!, [this.candidateParams, bank.control,
      this.source.receipts, this.source.receipts, this.source.receipts, this.source.control,
      this.source.receipts], 1);
    this.encodeVolumePublication(broker);
    this.encodeTopologyEvidence(broker, bank);
  }

  /** Release a CPU bootstrap upload after the caller has submitted its encoder. */
  releaseBootstrapUpload(): void {
    if (this.retainedBootstrap?.values === this.bootstrapUpload) this.retainedBootstrap = undefined;
    this.bootstrapUpload?.destroy(); this.bootstrapUpload = undefined;
  }

  /** GPU-select warm handoff or retained one-time cold bootstrap without readback. */
  encodeCandidateSelected(broker: PassBroker,
    input?: WebGPUOctreeLosassoAdaptivePhiBootstrap): void {
    this.assertInitialized();
    if (input) {
      let values: GPUBuffer;
      if (input.kind.endsWith("-cpu")) {
        const cpu = input.values as Float32Array;
        const expected = input.kind === "nodal-cpu" ? this.plan.nodeCapacity
          : input.kind === "nodal-lattice-cpu"
            ? this.options.dimensions.reduce((product, value) => product * (value + 1), 1)
            : this.options.dimensions.reduce((product, value) => product * value, 1);
        if (cpu.length !== expected) throw new RangeError(`adaptive phi ${input.kind} bootstrap requires ${expected} values`);
        this.bootstrapUpload?.destroy();
        this.bootstrapUpload = this.device.createBuffer({ label: "Retained one-time adaptive phi bootstrap", size: Math.max(4, cpu.byteLength),
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        this.device.queue.writeBuffer(this.bootstrapUpload, 0, cpu.slice().buffer); values = this.bootstrapUpload;
      } else values = input.values as GPUBuffer;
      this.retainedBootstrap = { kind: input.kind.startsWith("nodal-lattice")
        ? "nodal-lattice" : input.kind.startsWith("nodal") ? "nodal" : "cell-centred", values };
    }
    if (!this.retainedBootstrap) throw new Error("Adaptive phi GPU-selected candidate requires an initial retained bootstrap source");
    const scheduleBuffers = [this.candidateParams, this.graph.accepted.control, this.graph.candidate.control,
      this.hostRepairGate, this.source.control, this.candidateSchedule, this.source.receipts];
    this.run(broker, this.scheduleSource!, scheduleBuffers, 1);
    broker.fence("adaptive phi GPU candidate source schedule ready");
    const a = this.graph.accepted, c = this.graph.candidate;
    const handoffBuffers = [this.candidateParams, this.source.control, a.control, a.leaves, a.nodeDirectory,
      a.leafDirectory, a.phi, c.control, c.nodes, c.constraints, c.phi,
      this.source.control, this.source.receipts, this.source.receipts, this.source.receipts,
      this.source.receipts, this.source.receipts, this.source.receipts];
    this.runIndirect(broker, this.handoff!.handoffCandidate, handoffBuffers, 0);
    this.runIndirect(broker, this.handoff!.projectCandidate, handoffBuffers, 0);
    this.runIndirect(broker, this.handoff!.finalizeCandidateHandoff, handoffBuffers, 12);
    const coldBuffers = this.mainBuffers(c, this.retainedBootstrap.values, undefined, this.candidateParams);
    this.runMainIndirect(broker, this.retainedBootstrap.kind === "nodal-lattice"
      ? "bootstrapNodalLattice" : this.retainedBootstrap.kind === "nodal"
        ? "bootstrapNodal" : "bootstrapCellCentred", coldBuffers, 24);
    this.runMainIndirect(broker, "projectAccepted", coldBuffers, 24);
    this.runMainIndirect(broker, "deriveRows", coldBuffers, 36);
    this.runMainIndirect(broker, "deriveLeafVolumes", coldBuffers, 36);
    this.runMainIndirect(broker, "captureReferenceVolume", coldBuffers, 48);
    this.runMainIndirect(broker, "finalizeBootstrap", coldBuffers, 48);
    this.runIndirect(broker, this.stampCandidate!, [this.candidateParams, c.control,
      this.source.receipts, this.source.receipts, this.source.receipts, this.source.control,
      this.source.receipts], 48);
  }

  /** Pure retained/prolongated/constrained transfer into the candidate bank. */
  encodeCandidateHandoff(broker: PassBroker): void {
    this.assertInitialized(); const a = this.graph.accepted; const c = this.graph.candidate;
    const buffers = [this.candidateParams, this.source.control, a.control, a.leaves, a.nodeDirectory,
      a.leafDirectory, a.phi, c.control, c.nodes, c.constraints, c.phi,
      this.source.control, this.source.receipts, this.source.receipts, this.source.receipts,
      this.source.receipts, this.source.receipts, this.source.receipts];
    this.run(broker, this.handoff!.prepareCandidateHandoff, buffers, 1);
    this.run(broker, this.handoff!.handoffCandidate, buffers, this.plan.nodeDispatch[0]);
    this.run(broker, this.handoff!.projectCandidate, buffers, this.plan.nodeDispatch[0]);
    this.run(broker, this.handoff!.finalizeCandidateHandoff, buffers, 1);
  }

  /** Repair and validate candidate phi after the pure-transfer receipt exists. */
  encodeCandidateFinalize(broker: PassBroker,
    options: { readonly topologyControl?: GPUBuffer; readonly topologyChanged?: boolean } = {}): void {
    this.assertInitialized(); const buffers = this.mainBuffers(this.graph.candidate, undefined, undefined, this.candidateParams);
    const redistanceBuffers = this.redistanceBuffers(this.graph.candidate, this.candidateParams);
    if (!options.topologyControl) {
      const gate = new Uint32Array(6); gate[5] = options.topologyChanged === false ? 1 : 0;
      this.device.queue.writeBuffer(this.hostRepairGate, 0, gate);
    }
    const scheduleBuffers = [this.candidateParams, this.graph.accepted.control, this.graph.candidate.control,
      options.topologyControl ?? this.hostRepairGate, this.source.control, this.candidateSchedule,
      this.source.receipts];
    this.run(broker, this.scheduleRepair!, scheduleBuffers, 1);
    broker.fence("adaptive phi GPU candidate repair schedule ready");
    this.runMainIndirect(broker, "prepareCandidateRepair", buffers, 108);
    this.runMainIndirect(broker, "capturePreRedistanceVolumes", buffers, 96);
    this.runIndirect(broker, this.redistance!.initializeRedistance, redistanceBuffers.initialize, 60);
    for (let iteration = 0; iteration < this.plan.redistanceIterations; iteration += 1) {
      this.runIndirect(broker, this.redistance!.resetRedistanceResidual,
        redistanceBuffers.reset, 72);
      this.runIndirect(broker, this.redistance!.redistanceAtoB, redistanceBuffers.aToB, 60);
      this.runIndirect(broker, this.redistance!.projectDistanceB, redistanceBuffers.projectB, 60);
      this.runIndirect(broker, this.redistance!.redistanceBtoA, redistanceBuffers.bToA, 60);
      this.runIndirect(broker, this.redistance!.projectDistanceA, redistanceBuffers.projectA, 60);
    }
    this.runIndirect(broker, this.redistance!.finishRedistance, redistanceBuffers.finish, 60);
    this.runMainIndirect(broker, "projectTransported", buffers, 60);
    this.runIndirect(broker, this.redistance!.publishCandidateRedistanceReceipt,
      redistanceBuffers.publish, 72);
    this.runMainIndirect(broker, "canonicalizeCandidatePhi", buffers, 84);
    this.runMainIndirect(broker, "projectAccepted", buffers, 84);
    this.runIndirect(broker, this.volumeEvidence!.prepareVolumeEvidence,
      this.volumeEvidenceBuffers(this.graph.candidate, this.candidateParams), 108);
    this.runMainIndirect(broker, "derivePostRedistanceVolumes", buffers, 96);
    this.runIndirect(broker, this.volumeEvidence!.validateVolumeEvidence,
      this.volumeEvidenceBuffers(this.graph.candidate, this.candidateParams), 108);
    this.runIndirect(broker, this.volumeEvidence!.finalizeVolumeEvidence,
      this.volumeEvidenceBuffers(this.graph.candidate, this.candidateParams), 108);
    this.runMainIndirect(broker, "deriveRows", buffers, 96);
    this.runMainIndirect(broker, "finalizeCandidateRepair", buffers, 108);
    this.runIndirect(broker, this.stampRepair!, [this.candidateParams, this.graph.candidate.control,
      this.source.receipts, this.source.receipts, this.source.receipts,
      this.source.control, this.source.receipts], 108);
  }

  /** Midpoint SL transport followed by physical-spacing Eikonal iteration. */
  encodeAcceptedAdvance(broker: PassBroker, dt: number,
    velocity: WebGPUOctreeLosassoAdaptiveNodalVelocitySource,
    inflow?: WebGPUOctreeLosassoAdaptivePhiInflow): void {
    if (!(dt >= 0) || !Number.isFinite(dt)) throw new RangeError("adaptive phi dt must be finite and non-negative");
    this.assertInitialized(); this.writeParams(this.advanceParams, dt, inflow, false);
    const buffers = this.mainBuffers(this.graph.accepted, undefined, velocity, this.advanceParams);
    this.flushReferenceVolumeDelta();
    this.runMain(broker, "prepareAdvance", buffers, 1);
    this.runMain(broker, "applyReferenceVolumeDelta", buffers, 1);
    this.encodeAcceptedLiveSchedule(broker);
    const publishBuffers = [this.advanceParams, this.graph.accepted.control,
      this.graph.accepted.constraints, this.source.control, this.graph.accepted.phi,
      this.transportControl, this.transportWorklist, this.transportBandMask,
      this.graph.accepted.leaves, this.graph.accepted.incidentLeaves, velocity.values];
    this.run(broker, this.worklist!.prepareTransportBand,
      [this.transportControl, velocity.receipt], 1);
    this.runBufferIndirect(broker, this.worklist!.markTransportReach,
      [this.advanceParams, this.graph.accepted.control, this.source.control,
        this.graph.accepted.phi, this.transportControl, this.transportBandMask,
        this.graph.accepted.leaves, this.graph.accepted.incidentLeaves],
      this.source.nodeDispatch, 0);
    this.runBufferIndirect(broker, this.worklist!.markTransportInflow,
      [this.advanceParams, this.graph.accepted.control, this.graph.accepted.nodes,
        this.transportBandMask], this.source.nodeDispatch, 0);
    this.runBufferIndirect(broker, this.worklist!.markTransportConstraintMasters,
      [this.graph.accepted.control, this.graph.accepted.constraints,
        this.source.control, this.transportBandMask], this.source.nodeDispatch, 0);
    this.runBufferIndirect(broker, this.worklist!.markTransportConstrained,
      [this.graph.accepted.control, this.graph.accepted.constraints,
        this.source.control, this.transportBandMask], this.source.nodeDispatch, 0);
    this.runBufferIndirect(broker, this.worklist!.publishTransportIndependent,
      publishBuffers, this.source.nodeDispatch, 0);
    this.runBufferIndirect(broker, this.worklist!.publishTransportConstrained,
      publishBuffers, this.source.nodeDispatch, 0);
    this.run(broker, this.worklist!.finalizeTransportDispatch,
      [this.advanceParams, this.source.control, this.transportControl], 1);
    this.run(broker, this.worklist!.publishTransportPartition,
      [this.transportControl, this.source.receipts], 1);
    broker.copyBufferToBuffer(this.transportControl, 20, this.transportDispatch, 0, 24);
    this.runBufferIndirect(broker, this.backtrace!, [this.advanceParams, this.graph.accepted.control,
      this.graph.accepted.leaves, this.graph.accepted.nodes, this.graph.accepted.incidentLeaves,
      this.source.control, velocity.values, this.transportControl, this.transportWorklist,
      this.transportDepartures], this.transportDispatch, 0);
    this.runBufferIndirect(broker, this.transport!, [this.advanceParams, this.graph.accepted.control,
      this.graph.accepted.leaves, this.graph.accepted.nodes, this.graph.accepted.constraints,
      this.source.control, this.graph.accepted.phi,
      this.transportControl, this.transportWorklist, this.transportDepartures],
      this.transportDispatch, 0);
    this.runBufferIndirect(broker, this.worklist!.projectTransportedBand,
      [this.advanceParams, this.graph.accepted.control, this.graph.accepted.constraints,
        this.source.control, this.graph.accepted.phi, this.transportControl,
      this.transportWorklist], this.transportDispatch, 12);
    // The transport receipt counts the scheduled independent nodes and the
    // constrained projection exactly once.
    this.runMain(broker, "captureTransportReceipt", buffers, 1);
    this.runMainBufferIndirect(broker, "capturePreRedistanceVolumes", buffers,
      this.graph.accepted.control, this.graph.accepted.leafDispatchOffsetBytes);
    // Rebuild the same compact arena against the transported target bank.  A
    // two-finest-cell core plus one immutable incident-leaf ring is sufficient
    // for accurate redistance; transport's wider 7h characteristic support is
    // deliberately not inherited by this local SDF solve.
    this.run(broker, this.worklist!.prepareRedistanceBand,
      [this.transportControl, velocity.receipt, this.advanceParams], 1);
    this.runBufferIndirect(broker, this.worklist!.markTransportReach,
      [this.advanceParams, this.graph.accepted.control, this.source.control,
        this.graph.accepted.phi, this.transportControl, this.transportBandMask,
        this.graph.accepted.leaves, this.graph.accepted.incidentLeaves],
      this.source.nodeDispatch, 0);
    this.runBufferIndirect(broker, this.worklist!.markTransportConstraintMasters,
      [this.graph.accepted.control, this.graph.accepted.constraints,
        this.source.control, this.transportBandMask], this.source.nodeDispatch, 0);
    this.runBufferIndirect(broker, this.worklist!.markTransportConstrained,
      [this.graph.accepted.control, this.graph.accepted.constraints,
        this.source.control, this.transportBandMask], this.source.nodeDispatch, 0);
    this.runBufferIndirect(broker, this.worklist!.publishTransportIndependent,
      publishBuffers, this.source.nodeDispatch, 0);
    this.runBufferIndirect(broker, this.worklist!.publishTransportConstrained,
      publishBuffers, this.source.nodeDispatch, 0);
    this.run(broker, this.worklist!.finalizeTransportDispatch,
      [this.advanceParams, this.source.control, this.transportControl], 1);
    broker.copyBufferToBuffer(this.transportControl, 20, this.transportDispatch, 0, 24);
    const redistanceBuffers = this.redistanceBuffers(this.graph.accepted, this.advanceParams);
    this.run(broker, this.redistance!.prepareAcceptedRedistance,
      redistanceBuffers.prepareAccepted, 1);
    this.runBufferIndirect(broker, this.redistance!.initializeAcceptedRedistanceIndependent,
      redistanceBuffers.initialize, this.transportDispatch, 0);
    this.runBufferIndirect(broker, this.redistance!.initializeAcceptedRedistanceConstrained,
      redistanceBuffers.initialize, this.transportDispatch, 12);
    for (let iteration = 0; iteration < this.plan.redistanceIterations; iteration += 1) {
      this.run(broker, this.redistance!.resetRedistanceResidual, redistanceBuffers.reset, 1);
      this.runBufferIndirect(broker, this.redistance!.redistanceAtoB,
        redistanceBuffers.aToB, this.transportDispatch, 0);
      this.runBufferIndirect(broker, this.redistance!.projectDistanceB,
        redistanceBuffers.projectB, this.transportDispatch, 12);
      this.runBufferIndirect(broker, this.redistance!.redistanceBtoA,
        redistanceBuffers.bToA, this.transportDispatch, 0);
      this.runBufferIndirect(broker, this.redistance!.projectDistanceA,
        redistanceBuffers.projectA, this.transportDispatch, 12);
    }
    this.runBufferIndirect(broker, this.redistance!.finishAcceptedRedistanceIndependent,
      redistanceBuffers.finish, this.transportDispatch, 0);
    this.runBufferIndirect(broker, this.redistance!.finishAcceptedRedistanceConstrained,
      redistanceBuffers.finish, this.transportDispatch, 12);
    this.runBufferIndirect(broker, this.worklist!.projectTransportedBand,
      [this.advanceParams, this.graph.accepted.control, this.graph.accepted.constraints,
        this.source.control, this.graph.accepted.phi, this.transportControl,
        this.transportWorklist], this.transportDispatch, 12);
    this.run(broker, this.redistance!.publishAcceptedRedistanceReceipt,
      redistanceBuffers.publish, 1);
  }

  /** Finalize accepted constraints and derived views without changing volume globally. */
  encodeAcceptedFinalize(broker: PassBroker): WebGPUOctreeLosassoAdaptivePhiSource {
    this.assertInitialized(); const buffers = this.mainBuffers(
      this.graph.accepted, undefined, undefined, this.advanceParams);
    this.flushReferenceVolumeDelta();
    this.runMain(broker, "applyReferenceVolumeDelta", buffers, 1);
    this.encodeAcceptedLiveSchedule(broker);
    this.run(broker, this.volumeEvidence!.prepareVolumeEvidence,
      this.volumeEvidenceBuffers(this.graph.accepted, this.advanceParams), 1);
    this.runMainBufferIndirect(broker, "derivePostRedistanceVolumes", buffers,
      this.graph.accepted.control, this.graph.accepted.leafDispatchOffsetBytes);
    this.run(broker, this.volumeEvidence!.validateVolumeEvidence,
      this.volumeEvidenceBuffers(this.graph.accepted, this.advanceParams), 1);
    this.run(broker, this.volumeEvidence!.finalizeVolumeEvidence,
      this.volumeEvidenceBuffers(this.graph.accepted, this.advanceParams), 1);
    // Commit/reject the nodal bank before touching any pressure- or view-facing
    // derivation.  On rejection currentBank remains the prior accepted field.
    this.runMain(broker, "finalizeAccepted", buffers, 1);
    this.run(broker, this.stampAdvance!, [this.advanceParams, this.graph.accepted.control,
      this.source.receipts, this.source.receipts, this.source.receipts,
      this.source.control, this.source.receipts], 1);
    this.encodeAcceptedLiveSchedule(broker);
    this.runMainBufferIndirect(broker, "deriveRows", buffers, this.source.rowDispatch, 0);
    this.runMainBufferIndirect(broker, "deriveLeafVolumes", buffers,
      this.source.leafDispatch, 0);
    this.runMain(broker, "measureDerivations", buffers, 1);
    if (this.options.faces) {
      const f = this.options.faces;
      this.runBufferIndirect(broker, this.ghost!, [this.advanceParams,
        this.graph.accepted.control, this.graph.accepted.leaves,
        this.source.rowPhi, f.control, f.faces, this.source.ghostDistances,
        this.graph.accepted.pressureRowToGraphLeaf, this.graph.accepted.leafDirectory,
        this.graph.accepted.phi, this.source.control,
        this.graph.accepted.surfaceMass], this.source.faceDispatch, 0);
    }
    this.encodeVolumePublication(broker);
    this.encodeTopologyEvidence(broker, this.graph.accepted);
    return this.source;
  }

  /**
   * Stamp a candidate phi cache derived from authoritative candidate mass.
   * The graph/velocity joint gate runs afterward, once this scalar generation
   * is visible to candidate velocity reconstruction.
   */
  encodeJointCommitGate(broker: PassBroker, jointControl: GPUBuffer): void {
    this.assertInitialized();
    this.run(broker, this.commit!, [this.params, this.graph.candidate.control, jointControl,
      this.source.receipts, this.source.receipts, this.source.control,
      this.source.receipts], 1);
  }

  /** Synchronize scalar clocks after `surfaceGraph.encodeReadyCommit`. */
  encodeAcceptedCommit(broker: PassBroker): void {
    this.assertInitialized();
    this.run(broker, this.syncAccepted!, [this.params, this.graph.accepted.control,
      this.source.receipts, this.source.receipts, this.source.receipts,
      this.source.control, this.source.receipts], 1);
    this.encodeTopologyEvidence(broker, this.graph.accepted);
  }

  encodeAcceptedCommitSync(broker: PassBroker): void { this.encodeAcceptedCommit(broker); }

  /**
   * Adopt the accepted graph's field clocks after a same-topology velocity
   * rebuild. `stampAcceptedAdvance` publishes the scalar generation before
   * nodal velocity is reconstructed, so a cadence-k epoch needs this singleton
   * after the velocity publisher stamps graph word 6. The shared commit kernel
   * recognizes the unchanged epoch and preserves the active phi bank. Unlike a
   * topology commit, this does not rebuild evidence or copy graph-sized data.
   */
  encodeAcceptedFieldClockSync(broker: PassBroker): void {
    this.assertInitialized();
    this.run(broker, this.syncAccepted!, [this.params, this.graph.accepted.control,
      this.source.receipts, this.source.receipts, this.source.receipts,
      this.source.control, this.source.receipts], 1);
  }

  /** Publish the retained/prolongated phi adopted by a ready topology commit. */
  encodeAcceptedTopologyHandoffPublication(broker: PassBroker): void {
    this.assertInitialized();
    this.run(broker, this.stampTopologyHandoff!, [this.params, this.graph.accepted.control,
      this.source.control, this.source.receipts, this.source.receipts, this.source.control,
      this.source.receipts], 1);
  }

  /** Rebuild accepted leaf/face/view derivations without advancing scalar state. */
  encodeAcceptedDerivations(broker: PassBroker): WebGPUOctreeLosassoAdaptivePhiSource {
    this.assertInitialized(); const buffers = this.mainBuffers(this.graph.accepted);
    this.encodeAcceptedLiveSchedule(broker);
    this.runMainBufferIndirect(broker, "deriveRows", buffers, this.source.rowDispatch, 0);
    this.runMainBufferIndirect(broker, "deriveLeafVolumes", buffers,
      this.source.leafDispatch, 0);
    this.runMain(broker, "measureDerivations", buffers, 1);
    if (this.options.faces) {
      const f = this.options.faces;
      this.runBufferIndirect(broker, this.ghost!, [this.params, this.graph.accepted.control,
        this.graph.accepted.leaves, this.source.rowPhi, f.control, f.faces,
        this.source.ghostDistances, this.graph.accepted.pressureRowToGraphLeaf,
        this.graph.accepted.leafDirectory, this.graph.accepted.phi, this.source.control,
        this.graph.accepted.surfaceMass],
      this.source.faceDispatch, 0);
    }
    this.encodeVolumePublication(broker); this.encodeTopologyEvidence(broker, this.graph.accepted);
    return this.source;
  }

  /** This module intentionally has no dense compatibility encoder. */
  readonly denseCompatibilityIsPhysicsInput = false;

  /** Add an authored inflow/reference-volume increment before the next advance. */
  addReferenceVolume(volume_m3: number): void {
    this.assertLive(); if (!Number.isFinite(volume_m3)) throw new RangeError("adaptive phi reference-volume increment must be finite");
    this.pendingReferenceDelta += volume_m3;
  }

  destroy(): void {
    if (this.destroyed) return; this.destroyed = true;
    this.params.destroy(); this.candidateParams.destroy(); this.advanceParams.destroy();
    this.distanceA.destroy(); this.distanceB.destroy(); this.preRedistanceVolumes.destroy();
    this.transportBandMask.destroy(); this.redistanceReceipt.destroy();
    this.transportDispatch.destroy();
    this.candidateSchedule.destroy(); this.acceptedScheduleLimits.destroy();
    this.acceptedScheduleControl.destroy();
    this.hostRepairGate.destroy(); this.referenceDelta.destroy();
    this.candidateRendererDirectory.destroy();
    this.bootstrapUpload?.destroy();
    for (const [name, buffer] of Object.entries(this.source)) {
      if (name !== "acceptedPhiBanks" && name !== "candidatePhi"
        && typeof buffer === "object" && "destroy" in buffer) (buffer as GPUBuffer).destroy();
    }
    this.main = undefined; this.handoff = undefined; this.commit = undefined; this.stampCandidate = undefined; this.syncAccepted = undefined; this.stampRepair = undefined; this.stampAdvance = undefined; this.stampTopologyHandoff = undefined; this.ghost = undefined; this.evidence = undefined; this.scheduleSource = undefined; this.scheduleRepair = undefined; this.scheduleAccepted = undefined; this.worklist = undefined; this.redistance = undefined; this.backtrace = undefined; this.transport = undefined; this.volumeEvidence = undefined;
    this.bindGroupCache.clear();
  }

  private mainBuffers(bank: WebGPUOctreeLosassoAdaptivePhiGraphBankSource,
    distanceInput: GPUBuffer = this.distanceA,
    velocity?: WebGPUOctreeLosassoAdaptiveNodalVelocitySource,
    parameterBuffer: GPUBuffer = this.params): GPUBuffer[] {
    return [parameterBuffer, bank.control, bank.leaves, bank.leaves, bank.nodes, bank.incidentLeaves,
      bank.leafDirectory, bank.constraints, bank.adjacency, this.source.control, this.source.receipts,
      bank.phi, distanceInput, this.distanceB,
      velocity?.control ?? this.source.control, velocity?.values ?? this.source.rowGradient,
      this.source.physicalVolumes, this.source.rowPhi, this.source.rowGradient, this.source.physicalVolumes,
      this.referenceDelta, bank.pressureRowToGraphLeaf,
      this.preRedistanceVolumes, this.preRedistanceVolumes, velocity?.receipt ?? this.source.receipts,
      this.source.physicalVolumes, this.source.physicalVolumes];
  }

  private volumeEvidenceBuffers(bank: WebGPUOctreeLosassoAdaptivePhiGraphBankSource,
    parameterBuffer: GPUBuffer): GPUBuffer[] {
    return [parameterBuffer, bank.control, bank.constraints, this.source.control,
      this.source.receipts, bank.phi, this.source.physicalVolumes, this.preRedistanceVolumes];
  }

  private redistanceBuffers(bank: WebGPUOctreeLosassoAdaptivePhiGraphBankSource,
    parameterBuffer: GPUBuffer) {
    return Object.freeze({
      initialize: [parameterBuffer, bank.control, bank.leaves,
        bank.incidentLeaves, this.source.control, bank.phi, this.distanceA, this.distanceB,
        this.redistanceReceipt, this.transportControl, this.transportWorklist],
      prepareAccepted: [this.redistanceReceipt, this.transportControl],
      reset: [this.redistanceReceipt],
      aToB: [parameterBuffer, bank.control, bank.constraints, bank.adjacency,
        this.source.control, this.redistanceReceipt, this.distanceA, this.distanceB,
        this.transportBandMask, this.transportControl, this.transportWorklist],
      bToA: [parameterBuffer, bank.control, bank.constraints, bank.adjacency,
        this.source.control, this.redistanceReceipt, this.distanceB, this.distanceA,
        this.transportBandMask, this.transportControl, this.transportWorklist],
      projectA: [parameterBuffer, bank.control, bank.constraints,
        this.source.control, this.distanceA, this.transportBandMask,
        this.transportControl, this.transportWorklist],
      projectB: [parameterBuffer, bank.control, bank.constraints,
        this.source.control, this.distanceB, this.transportBandMask,
        this.transportControl, this.transportWorklist],
      finish: [parameterBuffer, bank.control, this.source.control,
        this.redistanceReceipt, bank.phi, this.distanceA, this.transportBandMask,
        this.transportControl, this.transportWorklist],
      publish: [this.redistanceReceipt, this.source.receipts, this.transportControl],
    });
  }


  private runMain(broker: PassBroker, name: MainPipelineName, buffers: readonly GPUBuffer[], workgroups: number): void {
    this.run(broker, this.main![name], buffers, workgroups);
  }

  private runMainIndirect(broker: PassBroker, name: MainPipelineName,
    buffers: readonly GPUBuffer[], offset: number): void {
    this.runIndirect(broker, this.main![name], buffers, offset);
  }

  private runMainBufferIndirect(broker: PassBroker, name: MainPipelineName,
    buffers: readonly GPUBuffer[], dispatch: GPUBuffer, offset: number): void {
    this.runBufferIndirect(broker, this.main![name], buffers, dispatch, offset);
  }

  /** Publish one fail-closed live schedule for the accepted graph tuple. */
  private encodeAcceptedLiveSchedule(broker: PassBroker): void {
    const faceControl = this.options.faces?.control ?? this.referenceDelta;
    this.run(broker, this.scheduleAccepted!, [this.acceptedScheduleLimits,
      this.graph.accepted.control, this.source.control,
      this.graph.accepted.pressureRowToGraphLeaf, faceControl,
      this.source.nodeDispatch, this.source.leafDispatch, this.source.rowDispatch,
      this.source.faceDispatch, this.acceptedScheduleControl], 1);
    // The dispatch buffers were storage outputs in the scheduling pass and are
    // indirect inputs below. End the pass so WebGPU establishes the usage and
    // command-order boundary explicitly.
    broker.fence("adaptive phi accepted live dispatch published");
  }

  private runBufferIndirect(broker: PassBroker, pipeline: GPUComputePipeline,
    buffers: readonly GPUBuffer[], dispatch: GPUBuffer, offset: number): void {
    const group = this.cachedBindGroup(pipeline, buffers);
    const pass = broker.compute({ label: pipeline.label }); pass.setPipeline(pipeline);
    pass.setBindGroup(0, group); pass.dispatchWorkgroupsIndirect(dispatch, offset);
  }

  private run(broker: PassBroker, pipeline: GPUComputePipeline, buffers: readonly GPUBuffer[], workgroups: number): void {
    const group = this.cachedBindGroup(pipeline, buffers);
    const pass = broker.compute({ label: pipeline.label }); pass.setPipeline(pipeline); pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(Math.max(1, workgroups));
  }

  private runIndirect(broker: PassBroker, pipeline: GPUComputePipeline,
    buffers: readonly GPUBuffer[], offset: number): void {
    const group = this.cachedBindGroup(pipeline, buffers);
    const pass = broker.compute({ label: pipeline.label }); pass.setPipeline(pipeline);
    pass.setBindGroup(0, group); pass.dispatchWorkgroupsIndirect(this.candidateSchedule, offset);
  }

  private cachedBindGroup(pipeline: GPUComputePipeline,
    buffers: readonly GPUBuffer[]): GPUBindGroup {
    const used = this.bindingSet(pipeline.label);
    const identity = (buffer: GPUBuffer): number => {
      const existing = this.bufferIdentity.get(buffer);
      if (existing !== undefined) return existing;
      const created = this.nextBufferIdentity++;
      this.bufferIdentity.set(buffer, created);
      return created;
    };
    const key = used.map((binding) => `${binding}:${identity(buffers[binding]!)}`).join(",");
    let pipelineGroups = this.bindGroupCache.get(pipeline);
    if (!pipelineGroups) {
      pipelineGroups = new Map();
      this.bindGroupCache.set(pipeline, pipelineGroups);
    }
    const existing = pipelineGroups.get(key);
    if (existing) return existing;
    const created = this.device.createBindGroup({ label: `${pipeline.label} stable bindings`,
      layout: pipeline.getBindGroupLayout(0), entries: used.map((binding) =>
        ({ binding, resource: { buffer: buffers[binding]! } })) });
    pipelineGroups.set(key, created);
    return created;
  }

  private bindingSet(label: string): readonly number[] {
    const name = label.slice(label.lastIndexOf("-") + 1).trim();
    const exact: Record<string, readonly number[]> = {
      prepareBootstrap: [0, 1, 2, 9, 10],
      bootstrapNodal: [0, 1, 9, 11, 12],
      bootstrapNodalLattice: [0, 1, 4, 9, 11, 12],
      bootstrapCellCentred: [0, 1, 4, 9, 11, 12],
      applyReferenceVolumeDelta: [9, 10, 20],
      projectAccepted: [0, 1, 7, 9, 11], finalizeBootstrap: [0, 1, 9, 10], captureReferenceVolume: [0, 1, 2, 9, 10, 19],
      measureDerivations: [0, 1, 2, 10, 19],
      captureTransportReceipt: [0, 9, 10],
      prepareAdvance: [0, 1, 9, 10, 14, 24],
      prepareTransportBand: [0, 1],
      prepareRedistanceBand: [0, 2],
      markTransportReach: [0, 1, 2, 3, 4, 5, 6, 7],
      markTransportInflow: [0, 1, 2, 3],
      publishTransportIndependent: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
      markTransportConstraintMasters: [0, 1, 2, 3],
      markTransportConstrained: [0, 1, 2, 3],
      publishTransportConstrained: [0, 1, 2, 3, 4, 5, 6, 7],
      finalizeTransportDispatch: [0, 1, 2],
      publishTransportPartition: [0, 1],
      projectTransportedBand: [0, 1, 2, 3, 4, 5, 6],
      backtraceIndependent: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
      transportIndependent: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
      prepareAcceptedRedistance: [0],
      initializeRedistance: [0, 1, 2, 3, 4, 5, 6, 7, 8],
      initializeAcceptedRedistanceIndependent: [0, 1, 2, 3, 4, 5, 6, 7, 9, 10],
      initializeAcceptedRedistanceConstrained: [0, 1, 2, 3, 4, 5, 6, 7, 9, 10],
      resetRedistanceResidual: [0],
      redistanceAtoB: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
      redistanceBtoA: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
      projectDistanceA: [0, 1, 2, 3, 4, 5, 6, 7],
      projectDistanceB: [0, 1, 2, 3, 4, 5, 6, 7],
      finishRedistance: [0, 1, 2, 3, 4, 5, 6],
      finishAcceptedRedistanceIndependent: [0, 1, 2, 3, 4, 5, 6, 7, 8],
      finishAcceptedRedistanceConstrained: [0, 1, 2, 3, 4, 5, 6, 7, 8],
      publishAcceptedRedistanceReceipt: [0, 1, 2],
      publishCandidateRedistanceReceipt: [0, 1],
      prepareCandidateRepair: [1, 9, 10],
      projectTransported: [0, 1, 7, 9, 11],
      canonicalizeCandidatePhi: [0, 1, 7, 9, 11],
      prepareVolumeEvidence: [0, 1, 2, 3, 4, 5, 6, 7],
      capturePreRedistanceVolumes: [0, 1, 2, 7, 9, 11, 23],
      deriveRows: [0, 1, 2, 7, 9, 11, 17, 18, 21], deriveLeafVolumes: [0, 1, 2, 7, 9, 11, 19],
      derivePostRedistanceVolumes: [0, 1, 2, 7, 9, 10, 11, 19],
      validateVolumeEvidence: [0, 1, 2, 3, 4, 5, 6],
      finalizeVolumeEvidence: [0, 1, 3, 4, 6, 7],
      finalizeAccepted: [0, 9, 10], finalizeCandidateRepair: [0, 9, 10],
      prepareCandidateHandoff: [0, 2, 7, 11], handoffCandidate: [0, 3, 4, 5, 6, 8, 10, 11],
      projectCandidate: [0, 7, 9, 10, 11], finalizeCandidateHandoff: [7, 11, 17],
      commitCandidate: [1, 2, 5, 6], deriveGhosts: [0, 1, 2, 3, 4, 5, 6, 7, 8, 11],
      stampCandidateBootstrap: [1, 5, 6],
      syncAcceptedCommit: [1, 5, 6],
      stampCandidateRepair: [0, 1, 5, 6],
      stampAcceptedAdvance: [1, 5, 6],
      publishTopologyHandoff: [1, 5, 6],
      prepareTopologyEvidenceEpoch: [0, 1, 3, 4, 5, 7, 8],
      prepareTopologyEvidence: [0, 1, 4, 5],
      publishTopologyEvidenceRows: [0, 1, 2, 3, 4, 5, 6, 7, 9, 10],
      finishTopologyEvidence: [0, 1, 3, 4, 5, 7, 8],
      scheduleAcceptedWork: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
      scheduleCandidateSource: [0, 1, 2, 4, 5, 6],
      scheduleCandidateRepair: [1, 2, 3, 4, 5],
    };
    const bindings = exact[name]; if (!bindings) throw new Error(`No adaptive phi binding set for ${label}`); return bindings;
  }

  private encodeVolumePublication(broker: PassBroker): void {
    // Publication is deliberately a copy from the canonical control/receipt
    // sources rather than a second numerical authority.
    broker.fence("adaptive phi volume publication");
    broker.copyBufferToBuffer(this.source.control, 28, this.volumePublication, 0, 4);
    broker.copyBufferToBuffer(this.source.control, 4, this.volumePublication, 4, 4);
    broker.copyBufferToBuffer(this.source.control, 20, this.volumePublication, 8, 4);
    broker.copyBufferToBuffer(this.source.control, 8, this.volumePublication, 12, 4);
    broker.copyBufferToBuffer(this.source.receipts, 44, this.volumePublication, 16, 4);
    broker.copyBufferToBuffer(this.source.control, 48, this.volumePublication, 20, 4);
  }

  private encodeTopologyEvidence(broker: PassBroker,
    bank: WebGPUOctreeLosassoAdaptivePhiGraphBankSource): void {
    // Candidate bootstrap/repair may be rejected by the later joint-ready
    // gate. Keep its renderer-only Power-directory materialization isolated;
    // the accepted directory remains byte-for-byte live until the accepted
    // bank is rebuilt after a successful graph commit.
    const rendererDirectory = bank === this.graph.candidate
      ? this.candidateRendererDirectory : this.source.rendererDirectory;
    const buffers = [this.params, bank.control, bank.leaves, bank.leafDirectory,
      this.source.topologyEvidence, rendererDirectory, bank.phi,
      this.source.control, this.source.receipts, bank.constraints, bank.surfaceMass];
    this.run(broker, this.evidence!.prepareTopologyEvidenceEpoch, buffers, 1);
    this.run(broker, this.evidence!.prepareTopologyEvidence, buffers,
      this.evidencePrepareWorkgroups);
    this.runBufferIndirect(broker, this.evidence!.publishTopologyEvidenceRows,
      buffers, bank.control, bank.leafDispatchOffsetBytes);
    this.run(broker, this.evidence!.finishTopologyEvidence, buffers, 1);
  }

  private writeParams(target: GPUBuffer, dt: number,
    inflow: WebGPUOctreeLosassoAdaptivePhiInflow | undefined,
    candidateDiagnostics: boolean): void {
    const buffer = new ArrayBuffer(128); const u = new Uint32Array(buffer); const f = new Float32Array(buffer);
    u.set([...this.options.dimensions, this.options.maximumLeafSpan], 0);
    const origin = this.options.domainOrigin ?? [0, 0, 0]; f.set([...origin, this.options.cellSize], 4);
    u.set([this.plan.nodeCapacity, this.plan.leafCapacity, this.options.directoryProbeLimit ?? 32,
      this.plan.redistanceIterations], 8);
    f.set([dt, this.options.redistanceBandWorld ?? 6 * this.options.maximumLeafSpan * this.options.cellSize,
      this.options.convergenceTolerance ?? 1e-4 * this.options.cellSize,
      this.options.exteriorAirPhi ?? .5 * this.options.cellSize], 12);
    let evidenceCapacity = 1; while (evidenceCapacity < 2 * this.plan.leafCapacity) evidenceCapacity *= 2;
    u.set([this.options.openTop ? 1 : 0, this.options.fullGraphRedistance ? 1 : 0,
      this.plan.faceCapacity, evidenceCapacity], 16);
    f.set([0, this.options.constraintTolerance ?? 1e-5, candidateDiagnostics ? 1 : 0,
      candidateDiagnostics ? 0 : 1], 20);
    if (inflow) {
      const numeric = [inflow.outletCenter_m.x, inflow.outletCenter_m.y, inflow.outletCenter_m.z,
        inflow.radius_m, inflow.velocity_m_s.x, inflow.velocity_m_s.y, inflow.velocity_m_s.z,
        inflow.apertureScale, inflow.strength];
      if (numeric.some((value) => !Number.isFinite(value)) || inflow.radius_m < 0
        || inflow.apertureScale < 0 || inflow.strength < 0) throw new RangeError("adaptive phi inflow must be finite and non-negative");
      const speed = Math.hypot(inflow.velocity_m_s.x, inflow.velocity_m_s.y,
        inflow.velocity_m_s.z);
      const inverseSpeed = speed > 1e-6 ? 1 / speed : 0;
      const centreX = inflow.outletCenter_m.x
        + .5 * this.options.dimensions[0] * this.options.cellSize;
      const centreZ = inflow.outletCenter_m.z
        + .5 * this.options.dimensions[2] * this.options.cellSize;
      f.set([centreX, inflow.outletCenter_m.y, centreZ, inflow.radius_m], 24);
      f.set([inflow.velocity_m_s.x * inverseSpeed,
        inflow.velocity_m_s.y * inverseSpeed, inflow.velocity_m_s.z * inverseSpeed,
        inflow.strength * inflow.apertureScale], 28);
    }
    this.device.queue.writeBuffer(target, 0, buffer);
  }

  private flushReferenceVolumeDelta(): void {
    if (this.pendingReferenceDelta === 0) return;
    this.device.queue.writeBuffer(this.referenceDelta, 0,
      new Float32Array([this.pendingReferenceDelta]).buffer);
    this.pendingReferenceDelta = 0;
  }

  private assertInitialized(): void { this.assertLive(); if (!this.main || !this.handoff || !this.commit || !this.stampCandidate || !this.syncAccepted || !this.stampRepair || !this.stampAdvance || !this.stampTopologyHandoff || !this.ghost || !this.evidence || !this.scheduleSource || !this.scheduleRepair || !this.scheduleAccepted || !this.worklist || !this.redistance || !this.backtrace || !this.transport || !this.volumeEvidence) throw new Error("Adaptive phi pipelines are not initialized"); }
  private assertLive(): void { if (this.destroyed) throw new Error("Adaptive phi authority is destroyed"); }
}
