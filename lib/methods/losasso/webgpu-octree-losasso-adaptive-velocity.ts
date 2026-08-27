import type { GPUInitializationTask } from "../../core/gpu-initialization";
import type { PassBroker } from "../../core/webgpu-pass-broker";
import type { LosassoSurfaceGraphBankSource } from "./webgpu-octree-losasso-surface-graph";
import {
  octreeLosassoAdaptiveVelocitySamplerWGSL,
  octreeLosassoAdaptiveVelocityWGSL,
} from "./webgpu-octree-losasso-adaptive-velocity.wgsl";

export const OCTREE_LOSASSO_ADAPTIVE_VELOCITY_RECEIPT_WORDS = 12;
export const OCTREE_LOSASSO_ADAPTIVE_VELOCITY_RECEIPT = Object.freeze({
  generation: 0, physicalReachBits: 1, invalidBits: 2, seededNodes: 3,
  missingCoverage: 4, unresolvedWithinReach: 5, validNodes: 6, ready: 7,
  renormalizedComponents: 9,
  publicationMode: 11,
} as const);
export const OCTREE_LOSASSO_ADAPTIVE_VELOCITY_PUBLICATION_MODE = Object.freeze({
  reconstructed: 0, coldZero: 1, retainedCommitted: 2,
} as const);
export const OCTREE_LOSASSO_ADAPTIVE_VELOCITY_DIAGNOSTIC_RECORDS = 64;
export const OCTREE_LOSASSO_ADAPTIVE_VELOCITY_DIAGNOSTIC_WORDS = 24;
export const OCTREE_LOSASSO_ADAPTIVE_VELOCITY_DIAGNOSTIC_HEADER_WORDS = 13;
export const OCTREE_LOSASSO_ADAPTIVE_VELOCITY_DIAGNOSTIC_BANK_WORDS =
  OCTREE_LOSASSO_ADAPTIVE_VELOCITY_DIAGNOSTIC_HEADER_WORDS
  + OCTREE_LOSASSO_ADAPTIVE_VELOCITY_DIAGNOSTIC_RECORDS
    * OCTREE_LOSASSO_ADAPTIVE_VELOCITY_DIAGNOSTIC_WORDS;
export const OCTREE_LOSASSO_ADAPTIVE_VELOCITY_RECEIPT_BASE = Object.freeze({
  accepted: 0, predictor: 12, candidate: 24, candidatePredictor: 36,
} as const);
/**
 * One topology-compiled component stencil is exactly two vec4u records:
 * packed metadata + four face slots + four packed u16 area weights.  A node
 * component samples the four incident tangential quadrants, so retaining the
 * former sixteen-slot envelope only multiplied recurring bandwidth.
 */
export const OCTREE_LOSASSO_ADAPTIVE_VELOCITY_STENCIL_WORDS = 8;
export const OCTREE_LOSASSO_ADAPTIVE_VELOCITY_STENCILS_PER_NODE = 3;
export const OCTREE_LOSASSO_ADAPTIVE_VELOCITY_STENCIL_CONTROL_WORDS = 8;
export const OCTREE_LOSASSO_ADAPTIVE_VELOCITY_STENCIL_CONTROL = Object.freeze({
  epoch: 0, nodeCount: 1, faceCount: 2, published: 3, errors: 4,
  weightedSamples: 5, maximumUniqueFaces: 6, compiledNodes: 7,
} as const);

export interface WebGPUOctreeLosassoAdaptiveVelocityFaceSource {
  /** Accepted/candidate Losasso authority prefix: generation,...,published,error. */
  readonly control: GPUBuffer;
  readonly faceCapacity: number;
  /** vec4u(axis | log2(span) << 2, x, y, z). */
  readonly faceGeometry: GPUBuffer;
  readonly axisFaceDirectory: GPUBuffer;
  readonly faceDirectoryCapacity: number;
  /** Topology-committed lagged face field used before the first projection tail. */
  readonly carriedValues?: GPUBuffer;
  readonly projectedValues: GPUBuffer;
  readonly predictorValues: GPUBuffer;
  /** Candidate-only 0/1/2 migration coverage written before nodal rebuild. */
  readonly migrationStatus?: GPUBuffer;
}

export interface WebGPUOctreeLosassoAdaptiveVelocityBuildSource {
  readonly graph: LosassoSurfaceGraphBankSource;
  /** Graph-slot mask compiled by adaptive-phi for the exact recurring
   * transport/redistance band. */
  readonly transportBandMask: GPUBuffer;
  /** Select old/new from the graph-owned vec2f phi record. Defaults to new. */
  readonly phiComponent?: 0 | 1;
  readonly faces: WebGPUOctreeLosassoAdaptiveVelocityFaceSource;
}

export interface WebGPUOctreeLosassoAdaptiveVelocityOptions {
  readonly nodeCapacity: number;
  readonly extensionReach: number;
  readonly minimumCellWidth: number;
  readonly dimensions: readonly [number, number, number];
  readonly maximumLeafSize: number;
  /** Dual-bank owner arena whose accepted direct brick directory identifies
   * the physical page used by the accepted graph leaf locator. */
  readonly ownerArena: GPUBuffer;
  readonly accepted: WebGPUOctreeLosassoAdaptiveVelocityBuildSource;
  readonly candidate: WebGPUOctreeLosassoAdaptiveVelocityBuildSource;
}

export interface WebGPUOctreeLosassoAdaptiveVelocitySamplerSource {
  readonly leaves: GPUBuffer;
  readonly ownerArena: GPUBuffer;
  readonly leafLocator: GPUBuffer;
  readonly velocityArena: GPUBuffer;
  /** Graph ABI is [accepted vec4, predictor vec4] for every node. */
  readonly nodeStrideRecords: 2;
  readonly acceptedFieldRecord: 0;
  readonly predictorFieldRecord: 1;
  readonly recordStrideWords: 4;
  readonly leafRecordStrideWords: 16;
  readonly wgsl: string;
}

/** Coherent accepted nodal tuple consumed by adaptive scalar transport. */
export interface WebGPUOctreeLosassoAdaptiveVelocityTransportSource {
  /** Accepted graph publication control: epoch/count/generation validity tuple. */
  readonly control: GPUBuffer;
  /** Interleaved accepted/predictor vec4 records on graph nodes. */
  readonly values: GPUBuffer;
  /** Stable accepted-field receipt, including physical reach and readiness. */
  readonly receipt: GPUBuffer;
}

export interface OctreeLosassoAdaptiveVelocityPlan {
  readonly nodeCapacity: number;
  /** Jeong-style accurate extrapolation is confined to Tall Cells' two-cell shell. */
  readonly accurateExtensionWaves: number;
  readonly extensionWaves: number;
  readonly scratchBytes: number;
  readonly receiptBytes: number;
  /** Bytes in one accepted/candidate compiled face-stencil bank. */
  readonly stencilBytes: number;
  readonly allocatedBytes: number;
}

export function planOctreeLosassoAdaptiveVelocity(input: {
  readonly nodeCapacity: number; readonly extensionReach: number;
  readonly minimumCellWidth: number;
  readonly dimensions?: readonly [number, number, number];
}): OctreeLosassoAdaptiveVelocityPlan {
  const { nodeCapacity, extensionReach, minimumCellWidth, dimensions } = input;
  if (!Number.isSafeInteger(nodeCapacity) || nodeCapacity < 1) {
    throw new RangeError("Adaptive velocity node capacity must be a positive safe integer");
  }
  if (!Number.isFinite(extensionReach) || extensionReach < 0) {
    throw new RangeError("Adaptive velocity extension reach must be finite and non-negative");
  }
  if (!Number.isFinite(minimumCellWidth) || minimumCellWidth <= 0) {
    throw new RangeError("Adaptive velocity minimum cell width must be finite and positive");
  }
  if (dimensions?.some((value) => !Number.isSafeInteger(value) || value < 1)) {
    throw new RangeError("Adaptive velocity dimensions must be positive integers");
  }
  // A point inside the Euclidean extension band has L1 displacement no larger
  // than sqrt(3) times its physical reach. Pricing three full independent axis
  // reaches encoded 22 waves for the UI dam where 14 are the geometric upper
  // bound. Keep the even ping-pong envelope and the exact unresolved receipt:
  // a graph whose topology needs more than this bound still fails closed.
  const accurateReach = Math.min(extensionReach, 2 * minimumCellWidth);
  const accurateEnvelope = Math.ceil(Math.sqrt(3) * accurateReach / minimumCellWidth) + 1;
  const accurateExtensionWaves = Math.max(2, accurateEnvelope + (accurateEnvelope & 1));
  const reachEnvelope = Math.ceil(Math.sqrt(3) * extensionReach / minimumCellWidth) + 1;
  const required = reachEnvelope;
  const extensionWaves = Math.max(2, required + (required & 1));
  const frontierSlices = accurateExtensionWaves + extensionWaves + 2;
  const scratchBytes = 4 * (16 + 3 * nodeCapacity
    + frontierSlices * (2 + nodeCapacity));
  const receiptBytes = (4 * OCTREE_LOSASSO_ADAPTIVE_VELOCITY_RECEIPT_WORDS + 4) * 4;
  const stencilBytes = 4 * OCTREE_LOSASSO_ADAPTIVE_VELOCITY_STENCIL_WORDS
    * OCTREE_LOSASSO_ADAPTIVE_VELOCITY_STENCILS_PER_NODE * nodeCapacity;
  return Object.freeze({ nodeCapacity, accurateExtensionWaves, extensionWaves,
    scratchBytes, receiptBytes, stencilBytes,
    allocatedBytes: scratchBytes + receiptBytes + 2 * stencilBytes
      + 2 * 4 * OCTREE_LOSASSO_ADAPTIVE_VELOCITY_STENCIL_CONTROL_WORDS + 6 * 64
      + 4 * 4 * OCTREE_LOSASSO_ADAPTIVE_VELOCITY_DIAGNOSTIC_BANK_WORDS });
}

type FieldName = "accepted" | "predictor" | "candidatePair" | "readyPair";

/**
 * Frontier waves consume GPU-authored compact lists.  Keep enough resident
 * lanes to cover the machine while letting each lane stride the actual list;
 * launching the node-capacity rectangle made every sparse wave pay for the
 * maximum graph even when only a few packets were live.
 */
const ADAPTIVE_VELOCITY_FRONTIER_WORKGROUPS = 32;

interface FieldBindings {
  readonly prepare: GPUBindGroup; readonly reconstruct: GPUBindGroup;
  readonly handoff?: GPUBindGroup;
  readonly seed: GPUBindGroup;
  readonly seedLeaves: GPUBindGroup;
  readonly waves: readonly { readonly group: GPUBindGroup; readonly inputSlice: number }[];
  readonly constrainA: GPUBindGroup;
  readonly finalize: GPUBindGroup; readonly finish: GPUBindGroup;
  readonly liveDispatch: GPUBuffer;
  readonly liveDispatchOffsetBytes: number;
  readonly leafDispatchOffsetBytes: number;
}

interface StencilBuildBindings {
  readonly prepare: GPUBindGroup;
  readonly compile: GPUBindGroup;
  readonly finish: GPUBindGroup;
  readonly prepareTopology: GPUBindGroup;
  readonly classifyTopology: GPUBindGroup;
  readonly scanTopology: GPUBindGroup;
  readonly indexTopology: GPUBindGroup;
  readonly emitTopology: GPUBindGroup;
  readonly leafTopology: GPUBindGroup;
  readonly incidentTopology: GPUBindGroup;
  readonly retainedTopology: GPUBindGroup;
  readonly finishTopology: GPUBindGroup;
}

interface StencilCommitBindings {
  readonly commit: GPUBindGroup;
}

const ENTRY_POINTS = ["prepareAdaptiveVelocityStencils", "compileAdaptiveVelocityStencils",
  "finishAdaptiveVelocityStencils", "prepareAdaptiveVelocityTopology",
  "classifyAdaptiveVelocityTopologyBlocks", "scanAdaptiveVelocityTopologyBlocks",
  "indexAdaptiveVelocityTopology", "emitAdaptiveVelocityTopology",
  "emitAdaptiveVelocityTopologyLeaves", "emitAdaptiveVelocityTopologyIncidents",
  "resolveAdaptiveVelocityRetained",
  "finishAdaptiveVelocityTopology",
  "commitAdaptiveVelocityTopology",
  "prepareAdaptiveVelocity", "reconstructAcceptedAdaptiveVelocity",
  "reconstructCandidateAdaptiveVelocity", "handoffAdaptiveVelocity",
  "seedAdaptiveVelocityFrontier", "seedAdaptiveVelocityLiquidLeaves",
  "propagateAdaptiveVelocityFrontier",
  "constrainAdaptiveVelocity", "finalizeAdaptiveVelocity",
  "finishAdaptiveVelocity"] as const;
type EntryPoint = typeof ENTRY_POINTS[number];
const BINDINGS: Readonly<Record<EntryPoint, readonly number[]>> = Object.freeze({
  prepareAdaptiveVelocityStencils: [0, 91, 93, 95, 96, 97],
  compileAdaptiveVelocityStencils: [0, 91, 92, 93, 94, 95, 96, 97],
  finishAdaptiveVelocityStencils: [97],
  prepareAdaptiveVelocityTopology: [0, 101, 102, 103, 104, 105, 106, 107, 108],
  classifyAdaptiveVelocityTopologyBlocks: [103, 108],
  scanAdaptiveVelocityTopologyBlocks: [108],
  indexAdaptiveVelocityTopology: [103, 108],
  emitAdaptiveVelocityTopology: [102, 103, 104, 107, 108],
  emitAdaptiveVelocityTopologyLeaves: [105, 108],
  emitAdaptiveVelocityTopologyIncidents: [106, 108],
  resolveAdaptiveVelocityRetained: [108],
  finishAdaptiveVelocityTopology: [108, 110],
  commitAdaptiveVelocityTopology: [108, 109],
  prepareAdaptiveVelocity: [0, 1, 2, 3, 4, 5],
  reconstructAcceptedAdaptiveVelocity: [0, 1, 2, 3, 4, 5, 6, 7],
  reconstructCandidateAdaptiveVelocity: [0, 1, 2, 3, 4, 5, 6, 7, 8],
  handoffAdaptiveVelocity: [0, 1, 2, 4, 6],
  seedAdaptiveVelocityFrontier: [0, 1, 2, 3, 4],
  seedAdaptiveVelocityLiquidLeaves: [0, 1, 2, 3, 4],
  propagateAdaptiveVelocityFrontier: [0, 1, 2, 3, 6, 7],
  constrainAdaptiveVelocity: [0, 1, 2, 3, 6],
  finalizeAdaptiveVelocity: [0, 1, 2, 3, 4, 6, 7, 8],
  finishAdaptiveVelocity: [0, 3, 51, 52],
});

/**
 * Compact velocity operator. Accepted/candidate buffers remain owned by the
 * graph transaction; this object owns graph-sized compiled face stencils,
 * extension scratch, and receipts, but no second velocity field authority.
 */
export class WebGPUOctreeLosassoAdaptiveVelocity {
  readonly initializationTasks: readonly GPUInitializationTask[];
  readonly velocityArena: GPUBuffer;
  readonly candidateVelocityArena: GPUBuffer;
  readonly receiptBuffer: GPUBuffer;
  /** Four fixed diagnostic banks; first 64 unresolved records per field. */
  readonly diagnosticBuffer: GPUBuffer;
  readonly samplerSource: WebGPUOctreeLosassoAdaptiveVelocitySamplerSource;
  readonly transportSource: WebGPUOctreeLosassoAdaptiveVelocityTransportSource;
  readonly plan: OctreeLosassoAdaptiveVelocityPlan;
  readonly allocatedBytes: number;
  private readonly mutableArena: GPUBuffer;
  private readonly mutableSliceBaseWords: number;
  private readonly mutableSliceStrideWords: number;
  private readonly controlArena: GPUBuffer;
  private readonly topologyArena: GPUBuffer;
  private readonly candidateStencils: GPUBuffer;
  private readonly candidateStencilControl: GPUBuffer;
  private readonly fieldParams = new Map<FieldName, GPUBuffer>();
  private readonly fieldParamWords = new Map<FieldName, Uint32Array>();
  private readonly waveParams: GPUBuffer[] = [];
  private readonly pipelines: Partial<Record<EntryPoint, GPUComputePipeline>> = {};
  private readonly fieldGroups = new Map<FieldName, FieldBindings>();
  private stencilBuildGroups?: StencilBuildBindings;
  private stencilCommitGroups?: StencilCommitBindings;
  private destroyed = false;

  /** Physical air-side reach used by this velocity extension instance. */
  get extensionReach_m(): number { return this.options.extensionReach; }

  /** Diagnostic-only view of the active/inactive compiled topology banks. */
  get acceptedStencilDiagnostics(): Readonly<{
    control: GPUBuffer; records: GPUBuffer;
  }> {
    return Object.freeze({ control: this.topologyArena, records: this.topologyArena });
  }

  constructor(private readonly device: GPUDevice,
    private readonly options: WebGPUOctreeLosassoAdaptiveVelocityOptions) {
    this.plan = planOctreeLosassoAdaptiveVelocity(options);
    if (options.accepted.graph.nodeCapacity < this.plan.nodeCapacity
      || options.candidate.graph.nodeCapacity < this.plan.nodeCapacity) {
      throw new RangeError("Adaptive velocity capacity exceeds a surface-graph bank");
    }
    if (!Number.isSafeInteger(options.maximumLeafSize) || options.maximumLeafSize < 1
      || options.maximumLeafSize > 32
      || (options.maximumLeafSize & (options.maximumLeafSize - 1)) !== 0) {
      throw new RangeError("Adaptive velocity maximum leaf size must be dyadic and no larger than 32");
    }
    if (options.dimensions.some((value) => !Number.isSafeInteger(value) || value < 1)) {
      throw new RangeError("Adaptive velocity dimensions must be positive integers");
    }
    if (this.plan.nodeCapacity >= 0x01000000) {
      throw new RangeError("Adaptive velocity compact frontier supports fewer than 2^24 nodes");
    }
    for (const source of [options.accepted, options.candidate]) {
      if (!Number.isSafeInteger(source.faces.faceCapacity) || source.faces.faceCapacity < 1
        || !Number.isSafeInteger(source.faces.faceDirectoryCapacity)
        || source.faces.faceDirectoryCapacity < 1
        || (source.faces.faceDirectoryCapacity & (source.faces.faceDirectoryCapacity - 1)) !== 0) {
        throw new RangeError("Adaptive velocity face capacities must be positive; directory must be dyadic");
      }
    }
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    this.velocityArena = options.accepted.graph.nodalVelocity;
    this.candidateVelocityArena = options.candidate.graph.nodalVelocity;
    this.candidateStencils = device.createBuffer({
      label: "Losasso candidate topology stencil staging",
      size: this.plan.stencilBytes, usage: storage,
    });
    this.candidateStencilControl = device.createBuffer({
      label: "Losasso candidate topology stencil staging control",
      size: 4 * OCTREE_LOSASSO_ADAPTIVE_VELOCITY_STENCIL_CONTROL_WORDS, usage: storage,
    });
    const packetWords = 38, spillWords = 11;
    const leafCapacity = Math.max(options.accepted.graph.leafCapacity,
      options.candidate.graph.leafCapacity);
    const mapOffset = 32;
    const packetOffset = mapOffset + this.plan.nodeCapacity;
    const spillOffset = packetOffset + packetWords * this.plan.nodeCapacity;
    const leafOffset = spillOffset + spillWords * this.plan.nodeCapacity;
    const incidentOffset = leafOffset + 16 * leafCapacity;
    const retainedOffset = incidentOffset + 8 * this.plan.nodeCapacity;
    const topologyBlockCapacity = Math.ceil(this.plan.nodeCapacity / 256);
    const packetBlockCountOffset = retainedOffset + this.plan.nodeCapacity;
    const spillBlockCountOffset = packetBlockCountOffset + topologyBlockCapacity;
    const packetBlockOffsetOffset = spillBlockCountOffset + topologyBlockCapacity;
    const spillBlockOffsetOffset = packetBlockOffsetOffset + topologyBlockCapacity;
    const topologyBankStrideWords = spillBlockOffsetOffset + topologyBlockCapacity;
    this.topologyArena = device.createBuffer({
      label: "Losasso double-bank compiled velocity topology arena",
      size: 4 * (16 + 2 * topologyBankStrideWords),
      usage: storage,
    });
    device.queue.writeBuffer(this.topologyArena, 0,
      new Uint32Array([0x5654_4f50, topologyBankStrideWords, 0]));
    for (let bank = 0; bank < 2; bank += 1) {
      const header = new Uint32Array(32);
      header.set([mapOffset, packetOffset, spillOffset, leafOffset,
        incidentOffset, retainedOffset, packetWords, spillWords], 7);
      header.set([packetBlockCountOffset, spillBlockCountOffset,
        packetBlockOffsetOffset, spillBlockOffsetOffset, topologyBlockCapacity], 15);
      device.queue.writeBuffer(this.topologyArena,
        4 * (16 + bank * topologyBankStrideWords), header);
    }
    const frontierSlices = this.plan.accurateExtensionWaves + this.plan.extensionWaves + 2;
    this.mutableSliceBaseWords = 16 + 3 * this.plan.nodeCapacity;
    this.mutableSliceStrideWords = 2 + this.plan.nodeCapacity;
    this.mutableArena = device.createBuffer({ label: "Losasso velocity mutable frontier arena",
      size: 4 * (this.mutableSliceBaseWords + frontierSlices * this.mutableSliceStrideWords),
      usage: storage });
    device.queue.writeBuffer(this.mutableArena, 0, new Uint32Array([0, 16,
      16 + 2 * this.plan.nodeCapacity, this.mutableSliceBaseWords,
      this.mutableSliceStrideWords, frontierSlices]));
    this.controlArena = device.createBuffer({ label: "Losasso velocity control arena",
      size: 4 * (4 * OCTREE_LOSASSO_ADAPTIVE_VELOCITY_RECEIPT_WORDS + 4
        + 4 * OCTREE_LOSASSO_ADAPTIVE_VELOCITY_DIAGNOSTIC_BANK_WORDS), usage: storage });
    this.receiptBuffer = device.createBuffer({ label: "Losasso adaptive velocity receipts",
      size: this.plan.receiptBytes, usage: storage });
    this.diagnosticBuffer = device.createBuffer({
      label: "Losasso adaptive velocity unresolved diagnostics",
      size: 4 * 4 * OCTREE_LOSASSO_ADAPTIVE_VELOCITY_DIAGNOSTIC_BANK_WORDS,
      usage: storage,
    });
    const createFieldParams = (name: FieldName, activeMask: 1 | 2 | 3,
      receiptBase: number, predictorReceiptBase: number, mode: 0 | 1 | 2,
      candidate: boolean): GPUBuffer => {
      const buffer = device.createBuffer({ label: `Losasso adaptive velocity ${name} parameters`,
        size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      const words = new Uint32Array(16);
      words.set([this.plan.nodeCapacity, activeMask, receiptBase, predictorReceiptBase], 0);
      new Float32Array(words.buffer)[4] = options.extensionReach;
      words[5] = 2;
      words[6] = (candidate ? options.candidate.phiComponent : options.accepted.phiComponent) ?? 1;
      // 0 = normal accepted, 1 = candidate cold bootstrap, 2 = accepted
      // construction-ready bootstrap. Ready bootstrap has distinct immutable
      // params so a normal accepted rebuild can never silently seed zeros.
      words[7] = mode;
      words.set([...options.dimensions, options.maximumLeafSize], 8);
      const faces = candidate ? options.candidate.faces : options.accepted.faces;
      words[12] = faces.faceCapacity;
      words[13] = faces.faceDirectoryCapacity;
      const terminalWave = 1 + this.plan.accurateExtensionWaves + this.plan.extensionWaves;
      const causalEndWave = 1 + this.plan.accurateExtensionWaves;
      words[14] = (faces.migrationStatus ? 1 : 0) | (terminalWave << 8)
        | (causalEndWave << 16);
      // Tall Cells 3.3: use the accurate causal solver only in a two-finest-cell
      // interface shell. The sparse harmonic hierarchy closes the remaining
      // transport support without allocating or sweeping a dense domain.
      new Float32Array(words.buffer)[15] = Math.min(options.extensionReach,
        2 * options.minimumCellWidth);
      device.queue.writeBuffer(buffer, 0, words);
      this.fieldParams.set(name, buffer);
      this.fieldParamWords.set(name, words);
      return buffer;
    };
    createFieldParams("accepted", 1, 0, 12, 0, false);
    createFieldParams("predictor", 2, 0, 12, 0, false);
    createFieldParams("candidatePair", 3, 24, 36, 1, true);
    createFieldParams("readyPair", 3, 0, 12, 2, false);
    this.allocatedBytes = this.mutableArena.size + this.controlArena.size
      + this.topologyArena.size
      + this.receiptBuffer.size + this.diagnosticBuffer.size
      + this.candidateStencils.size + this.candidateStencilControl.size
      + [...this.fieldParams.values()]
        .reduce((sum, buffer) => sum + buffer.size, 0);
    this.samplerSource = Object.freeze({ leaves: options.accepted.graph.leaves,
      ownerArena: options.ownerArena, leafLocator: options.accepted.graph.leafLocator,
      velocityArena: this.velocityArena, nodeStrideRecords: 2, acceptedFieldRecord: 0,
      predictorFieldRecord: 1, recordStrideWords: 4,
      leafRecordStrideWords: 16, wgsl: octreeLosassoAdaptiveVelocitySamplerWGSL() });
    this.transportSource = Object.freeze({ control: options.accepted.graph.control,
      values: this.velocityArena, receipt: this.receiptBuffer });
    this.initializationTasks = [{ id: "losasso.adaptive-velocity.pipelines",
      phase: "solver-pipelines", label: "Compile adaptive Losasso nodal velocity",
      run: () => this.initialize() }];
  }

  private async initialize(): Promise<void> {
    this.assertLive(); if (this.pipelines.prepareAdaptiveVelocity) return;
    const shaderModule = this.device.createShaderModule({ label: "Losasso adaptive nodal velocity",
      code: octreeLosassoAdaptiveVelocityWGSL });
    const compiled = await Promise.all(ENTRY_POINTS.map(async (entryPoint) => ({ entryPoint,
      pipeline: await this.device.createComputePipelineAsync({ layout: "auto",
        compute: { module: shaderModule, entryPoint } }) })));
    for (const { entryPoint, pipeline } of compiled) this.pipelines[entryPoint] = pipeline;
    const candidateParams = this.fieldParams.get("candidatePair")!;
    const candidate = this.options.candidate;
    this.stencilBuildGroups = {
      prepare: this.createGroup("prepareAdaptiveVelocityStencils", [candidateParams,
        candidate.graph.control, candidate.faces.control, candidate.faces.axisFaceDirectory,
        this.candidateStencils, this.candidateStencilControl]),
      compile: this.createGroup("compileAdaptiveVelocityStencils", [candidateParams,
        candidate.graph.control, candidate.graph.nodes, candidate.faces.control,
        candidate.faces.faceGeometry, candidate.faces.axisFaceDirectory,
        this.candidateStencils, this.candidateStencilControl]),
      finish: this.createGroup("finishAdaptiveVelocityStencils",
        [this.candidateStencilControl]),
      prepareTopology: this.createGroup("prepareAdaptiveVelocityTopology", [candidateParams,
        candidate.graph.control, candidate.graph.nodes, candidate.graph.constraints,
        candidate.graph.adjacency, candidate.graph.leaves, candidate.graph.incidentLeaves,
        this.candidateStencils, this.topologyArena]),
      classifyTopology: this.createGroup("classifyAdaptiveVelocityTopologyBlocks",
        [candidate.graph.constraints, this.topologyArena]),
      scanTopology: this.createGroup("scanAdaptiveVelocityTopologyBlocks", [this.topologyArena]),
      indexTopology: this.createGroup("indexAdaptiveVelocityTopology",
        [candidate.graph.constraints, this.topologyArena]),
      emitTopology: this.createGroup("emitAdaptiveVelocityTopology", [candidate.graph.nodes,
        candidate.graph.constraints, candidate.graph.adjacency, this.candidateStencils,
        this.topologyArena]),
      leafTopology: this.createGroup("emitAdaptiveVelocityTopologyLeaves",
        [candidate.graph.leaves, this.topologyArena]),
      incidentTopology: this.createGroup("emitAdaptiveVelocityTopologyIncidents",
        [candidate.graph.incidentLeaves, this.topologyArena]),
      retainedTopology: this.createGroup("resolveAdaptiveVelocityRetained", [this.topologyArena]),
      finishTopology: this.createGroup("finishAdaptiveVelocityTopology",
        [this.topologyArena, this.candidateStencilControl]),
    };
    const accepted = this.options.accepted;
    this.stencilCommitGroups = {
      commit: this.createGroup("commitAdaptiveVelocityTopology",
        [this.topologyArena, accepted.graph.control]),
    };
    const acceptedCarry = this.options.accepted.faces.carriedValues;
    this.fieldGroups.set("accepted", this.createFieldBindings("accepted", this.options.accepted,
      this.options.accepted.faces.projectedValues, this.options.accepted.faces.predictorValues));
    this.fieldGroups.set("predictor", this.createFieldBindings("predictor", this.options.accepted,
      this.options.accepted.faces.projectedValues, this.options.accepted.faces.predictorValues));
    this.fieldGroups.set("candidatePair", this.createFieldBindings("candidatePair",
      this.options.candidate, this.options.candidate.faces.projectedValues,
      this.options.candidate.faces.predictorValues));
    if (acceptedCarry) this.fieldGroups.set("readyPair", this.createFieldBindings("readyPair",
      this.options.accepted, acceptedCarry, acceptedCarry));
  }

  private createGroup(entryPoint: EntryPoint,
    buffers: readonly (GPUBuffer | GPUBufferBinding)[]): GPUBindGroup {
    const pipeline = this.pipelines[entryPoint]; if (!pipeline) throw new Error("Adaptive velocity pipeline missing");
    if (buffers.length !== BINDINGS[entryPoint].length) {
      throw new Error(`Adaptive velocity ${entryPoint} binding count mismatch`);
    }
    return this.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0),
      entries: buffers.map((buffer, index) => ({ binding: BINDINGS[entryPoint][index]!,
        resource: "buffer" in buffer ? buffer : { buffer } })) });
  }

  private createWaveParams(name: FieldName, inputSlice: number): GPUBuffer {
    const words = new Uint32Array(this.fieldParamWords.get(name)!);
    words[7] = (words[7]! & 0xff) | (inputSlice << 8);
    const buffer = this.device.createBuffer({
      label: `Losasso adaptive velocity ${name} frontier ${inputSlice} parameters`,
      size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(buffer, 0, words); this.waveParams.push(buffer);
    return buffer;
  }

  private createFieldBindings(name: FieldName, source: WebGPUOctreeLosassoAdaptiveVelocityBuildSource,
    faceValues0: GPUBuffer, faceValues1: GPUBuffer): FieldBindings {
    const p = this.fieldParams.get(name)!;
    const candidate = name.startsWith("candidate");
    const topology = this.topologyArena;
    const causalEnd = this.plan.accurateExtensionWaves + 1;
    const terminal = causalEnd + this.plan.extensionWaves;
    const waves: { group: GPUBindGroup; inputSlice: number }[] = [];
    const makeWave = (inputSlice: number) => waves.push({ inputSlice,
      group: this.createGroup("propagateAdaptiveVelocityFrontier",
        [this.createWaveParams(name, inputSlice), topology, this.mutableArena,
          this.controlArena, source.graph.nodalVelocity, source.graph.phi]) });
    for (let input = 0; input < this.plan.accurateExtensionWaves; input += 1) makeWave(input);
    for (let input = causalEnd; input < terminal; input += 1) makeWave(input);
    const reconstructEntry = candidate
      ? "reconstructCandidateAdaptiveVelocity" : "reconstructAcceptedAdaptiveVelocity";
    return {
      prepare: this.createGroup("prepareAdaptiveVelocity", [p, topology, this.mutableArena,
        this.controlArena, source.graph.control, this.options.accepted.graph.control]),
      reconstruct: this.createGroup(reconstructEntry, [p, topology, this.mutableArena,
        this.controlArena, faceValues0, faceValues1, source.graph.nodalVelocity,
        source.graph.phi,
        ...(candidate ? [source.faces.migrationStatus ?? faceValues0] : [])]),
      ...(candidate ? { handoff: this.createGroup("handoffAdaptiveVelocity", [p, topology,
        this.mutableArena, this.options.accepted.graph.nodalVelocity,
        source.graph.nodalVelocity]) } : {}),
      seed: this.createGroup("seedAdaptiveVelocityFrontier", [p, topology, this.mutableArena,
        this.controlArena, source.transportBandMask]),
      seedLeaves: this.createGroup("seedAdaptiveVelocityLiquidLeaves", [p, topology,
        this.mutableArena, this.controlArena, source.transportBandMask]),
      waves,
      constrainA: this.createGroup("constrainAdaptiveVelocity", [p, topology,
        this.mutableArena, this.controlArena, source.graph.nodalVelocity]),
      finalize: this.createGroup("finalizeAdaptiveVelocity", [p, topology, this.mutableArena,
        this.controlArena, source.transportBandMask, source.graph.nodalVelocity,
        source.graph.nodeValidity, source.graph.phi]),
      finish: this.createGroup("finishAdaptiveVelocity", [p, this.controlArena,
        source.graph.control, source.graph.leafLocator]),
      liveDispatch: source.graph.control,
      liveDispatchOffsetBytes: source.graph.nodeDispatchOffsetBytes,
      leafDispatchOffsetBytes: source.graph.leafDispatchOffsetBytes,
    };
  }
  /** Reconstruct only the carried/projected field consumed by forward traces. */
  encodeAcceptedField(broker: PassBroker): void {
    this.encodeField(broker, "accepted");
  }
  /** Reconstruct only the advected predictor consumed by the reverse trace. */
  encodePredictorField(broker: PassBroker): void {
    this.encodeField(broker, "predictor");
  }
  /**
   * Publish the construction/topology-boundary fields from the committed
   * lagged face authority. No projected or MacCormack face field exists yet at
   * that boundary, so both nodal banks intentionally receive the same carried
   * values and the normal projection tail replaces them later.
   */
  encodeAcceptedReadyFields(broker: PassBroker): void {
    this.assertReady();
    this.encodeReadyStencilCommit(broker);
    if (!this.fieldGroups.has("readyPair")) {
      throw new Error("Adaptive velocity has no committed carried face field");
    }
    this.encodeField(broker, "readyPair");
  }
  /**
   * Retain the coherent accepted nodal tuple while refreshing its phi-bound
   * extension support. A ready authority commit publishes only the carried
   * face bank; projected/predictor face arrays still use the previous face
   * ordering and cannot seed reconstruction until S1a/S3 replace them.
   */
  encodeAcceptedRetainedFields(broker: PassBroker): void {
    this.assertReady();
    if (!this.fieldGroups.has("readyPair")) {
      throw new Error("Adaptive velocity has no retained accepted field group");
    }
    this.encodeField(broker, "readyPair");
  }
  /** Compile topology/geometry-only candidate face lookups once per graph publication. */
  encodeCandidateStencils(broker: PassBroker): void {
    const groups = this.stencilBuildGroups;
    if (!groups) throw new Error("Adaptive velocity stencil compiler is not initialized");
    const run = (entryPoint: EntryPoint, group: GPUBindGroup, indirect = false,
      workgroups = 1, indirectOffset = this.options.candidate.graph.nodeDispatchOffsetBytes) => {
      const pass = broker.compute({ label: `Losasso - ${entryPoint}` });
      pass.setPipeline(this.pipelines[entryPoint]!); pass.setBindGroup(0, group);
      if (indirect) pass.dispatchWorkgroupsIndirect(this.options.candidate.graph.control,
        indirectOffset);
      else pass.dispatchWorkgroups(workgroups);
    };
    const runTopologyBlocks = (entryPoint: EntryPoint, group: GPUBindGroup) => {
      const pass = broker.compute({ label: `Losasso - ${entryPoint}` });
      pass.setPipeline(this.pipelines[entryPoint]!); pass.setBindGroup(0, group);
      pass.dispatchWorkgroupsIndirect(this.options.candidate.graph.control,
        this.options.candidate.graph.topologyBlockDispatchOffsetBytes);
    };
    run("prepareAdaptiveVelocityStencils", groups.prepare);
    run("compileAdaptiveVelocityStencils", groups.compile, true);
    run("finishAdaptiveVelocityStencils", groups.finish);
    run("prepareAdaptiveVelocityTopology", groups.prepareTopology);
    runTopologyBlocks("classifyAdaptiveVelocityTopologyBlocks", groups.classifyTopology);
    run("scanAdaptiveVelocityTopologyBlocks", groups.scanTopology);
    runTopologyBlocks("indexAdaptiveVelocityTopology", groups.indexTopology);
    run("emitAdaptiveVelocityTopology", groups.emitTopology, true);
    run("emitAdaptiveVelocityTopologyLeaves", groups.leafTopology, true, 1,
      this.options.candidate.graph.leafDispatchOffsetBytes);
    run("emitAdaptiveVelocityTopologyIncidents", groups.incidentTopology, true);
    run("resolveAdaptiveVelocityRetained", groups.retainedTopology, true);
    run("finishAdaptiveVelocityTopology", groups.finishTopology);
    broker.fence("velocity topology packet publication");
  }
  /** Reconstruct both candidate banks against the already-compiled topology lookups. */
  encodeCandidateFieldRound(broker: PassBroker): void {
    this.encodeField(broker, "candidatePair");
  }
  /** Combined single-round entry point retained for standalone candidate construction. */
  encodeCandidateFields(broker: PassBroker): void {
    this.encodeCandidateStencils(broker);
    this.encodeCandidateFieldRound(broker);
  }
  private encodeReadyStencilCommit(broker: PassBroker): void {
    const groups = this.stencilCommitGroups;
    if (!groups) throw new Error("Adaptive velocity topology commit is not initialized");
    const commit = broker.compute({ label: "Losasso - commit adaptive velocity topology bank" });
    commit.setPipeline(this.pipelines.commitAdaptiveVelocityTopology!);
    commit.setBindGroup(0, groups.commit);
    commit.dispatchWorkgroups(1);
  }
  private encodeField(broker: PassBroker, name: FieldName): void {
    this.assertReady(); const groups = this.fieldGroups.get(name)!;
    const run = (entryPoint: EntryPoint, group: GPUBindGroup, live = false) => {
      const pass = broker.compute({ label: `Losasso - adaptive velocity ${name} ${entryPoint}` });
      pass.setPipeline(this.pipelines[entryPoint]!); pass.setBindGroup(0, group);
      if (live) pass.dispatchWorkgroupsIndirect(groups.liveDispatch,
        groups.liveDispatchOffsetBytes);
      else pass.dispatchWorkgroups(1);
    };
    run("prepareAdaptiveVelocity", groups.prepare);
    run(name.startsWith("candidate") ? "reconstructCandidateAdaptiveVelocity"
      : "reconstructAcceptedAdaptiveVelocity", groups.reconstruct, true);
    if (groups.handoff) run("handoffAdaptiveVelocity", groups.handoff, true);
    for (let wave = 0; wave < this.plan.accurateExtensionWaves; wave += 1) {
      const selected = groups.waves[wave]!;
      const pass = broker.compute({ label: `Losasso - adaptive velocity ${name} causal frontier` });
      pass.setPipeline(this.pipelines.propagateAdaptiveVelocityFrontier!);
      pass.setBindGroup(0, selected.group);
      pass.dispatchWorkgroups(ADAPTIVE_VELOCITY_FRONTIER_WORKGROUPS);
    }
    run("seedAdaptiveVelocityFrontier", groups.seed, true);
    const seedLeaves = broker.compute({
      label: `Losasso - adaptive velocity ${name} seedAdaptiveVelocityLiquidLeaves`,
    });
    seedLeaves.setPipeline(this.pipelines.seedAdaptiveVelocityLiquidLeaves!);
    seedLeaves.setBindGroup(0, groups.seedLeaves);
    seedLeaves.dispatchWorkgroupsIndirect(groups.liveDispatch, groups.leafDispatchOffsetBytes);
    for (let wave = 0; wave < this.plan.extensionWaves; wave += 1) {
      const selected = groups.waves[this.plan.accurateExtensionWaves + wave]!;
      const pass = broker.compute({ label: `Losasso - adaptive velocity ${name} harmonic frontier` });
      pass.setPipeline(this.pipelines.propagateAdaptiveVelocityFrontier!);
      pass.setBindGroup(0, selected.group);
      pass.dispatchWorkgroups(ADAPTIVE_VELOCITY_FRONTIER_WORKGROUPS);
    }
    // Hanging nodes are immutable aliases throughout propagation.  Consumers
    // resolve their independent masters directly; materialize the compact
    // constraint set once into the final even ping-pong bank for publication.
    run("finalizeAdaptiveVelocity", groups.finalize, true);
    run("constrainAdaptiveVelocity", groups.constrainA, true);
    run("finishAdaptiveVelocity", groups.finish);
    broker.copyBufferToBuffer(this.controlArena, 0, this.receiptBuffer, 0,
      this.receiptBuffer.size);
    broker.copyBufferToBuffer(this.controlArena, this.plan.receiptBytes,
      this.diagnosticBuffer, 0, this.diagnosticBuffer.size);
  }

  destroy(): void {
    if (this.destroyed) return; this.destroyed = true;
    this.mutableArena.destroy(); this.controlArena.destroy();
    this.topologyArena.destroy(); this.candidateStencils.destroy();
    this.candidateStencilControl.destroy();
    this.receiptBuffer.destroy(); this.diagnosticBuffer.destroy();
    for (const buffer of this.fieldParams.values()) buffer.destroy();
    for (const buffer of this.waveParams) buffer.destroy();
    this.fieldParams.clear(); this.fieldParamWords.clear(); this.fieldGroups.clear();
    this.stencilBuildGroups = undefined; this.stencilCommitGroups = undefined;
  }
  private assertLive(): void { if (this.destroyed) throw new Error("Adaptive velocity is destroyed"); }
  private assertReady(): void { this.assertLive(); if (!this.pipelines.prepareAdaptiveVelocity) {
    throw new Error("Adaptive velocity is not initialized");
  } }
}
