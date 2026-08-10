import type { GPUInitializationTask } from "./gpu-initialization";
import type { PassBroker } from "./webgpu-pass-broker";
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
export const OCTREE_LOSASSO_ADAPTIVE_VELOCITY_STENCIL_WORDS = 36;
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
  const scratchBytes = 16 * nodeCapacity;
  const receiptBytes = (4 * OCTREE_LOSASSO_ADAPTIVE_VELOCITY_RECEIPT_WORDS + 4) * 4;
  const stencilBytes = 4 * OCTREE_LOSASSO_ADAPTIVE_VELOCITY_STENCIL_WORDS
    * OCTREE_LOSASSO_ADAPTIVE_VELOCITY_STENCILS_PER_NODE * nodeCapacity;
  return Object.freeze({ nodeCapacity, accurateExtensionWaves, extensionWaves,
    scratchBytes, receiptBytes, stencilBytes,
    allocatedBytes: scratchBytes + receiptBytes + 2 * stencilBytes
      + 2 * 4 * OCTREE_LOSASSO_ADAPTIVE_VELOCITY_STENCIL_CONTROL_WORDS + 6 * 64
      + 4 * 4 * OCTREE_LOSASSO_ADAPTIVE_VELOCITY_DIAGNOSTIC_BANK_WORDS });
}

type FieldName = "accepted" | "predictor" | "candidate" | "candidatePredictor";
const FIELD_INDEX: Readonly<Record<FieldName, number>> = Object.freeze({
  accepted: 0, predictor: 1, candidate: 2, candidatePredictor: 3,
});

interface FieldBindings {
  readonly prepare: GPUBindGroup; readonly reconstruct: GPUBindGroup;
  readonly handoff?: GPUBindGroup;
  readonly markSupport: GPUBindGroup;
  readonly extendAB: GPUBindGroup; readonly extendBA: GPUBindGroup;
  readonly copySupportAB: GPUBindGroup; readonly copySupportBA: GPUBindGroup;
  readonly dilateSupportAB: GPUBindGroup; readonly dilateSupportBA: GPUBindGroup;
  readonly closeAB: GPUBindGroup; readonly closeBA: GPUBindGroup;
  readonly constrainA: GPUBindGroup; readonly constrainB: GPUBindGroup;
  readonly finalize: GPUBindGroup; readonly finish: GPUBindGroup;
}

interface StencilBuildBindings {
  readonly prepare: GPUBindGroup;
  readonly compile: GPUBindGroup;
  readonly finish: GPUBindGroup;
}

interface StencilCommitBindings {
  readonly commit: GPUBindGroup;
  readonly finish: GPUBindGroup;
}

const ENTRY_POINTS = ["prepareAdaptiveVelocityStencils", "compileAdaptiveVelocityStencils",
  "finishAdaptiveVelocityStencils", "commitAdaptiveVelocityStencils",
  "finishAdaptiveVelocityStencilCommit",
  "prepareAdaptiveVelocity", "reconstructAdaptiveVelocity", "handoffAdaptiveVelocity",
  "markAdaptiveVelocitySupport", "extendAdaptiveVelocity", "closeAdaptiveVelocity",
  "copyAdaptiveVelocitySupport", "dilateAdaptiveVelocitySupport",
  "constrainAdaptiveVelocity", "finalizeAdaptiveVelocity",
  "finishAdaptiveVelocity"] as const;
type EntryPoint = typeof ENTRY_POINTS[number];
const BINDINGS: Readonly<Record<EntryPoint, readonly number[]>> = Object.freeze({
  prepareAdaptiveVelocityStencils: [0, 91, 93, 95, 96, 97],
  compileAdaptiveVelocityStencils: [0, 91, 92, 93, 94, 95, 96, 97],
  finishAdaptiveVelocityStencils: [97],
  commitAdaptiveVelocityStencils: [101, 102, 103, 104, 106],
  finishAdaptiveVelocityStencilCommit: [101, 103, 104, 105],
  prepareAdaptiveVelocity: [0, 1, 2, 3, 4, 5],
  reconstructAdaptiveVelocity: [0, 11, 12, 13, 14, 15, 16, 17, 18, 19],
  handoffAdaptiveVelocity: [0, 111, 112, 113, 114, 115, 116, 117],
  markAdaptiveVelocitySupport: [0, 61, 62, 63, 64, 65, 66, 67],
  extendAdaptiveVelocity: [0, 21, 22, 23, 24, 25, 26, 27],
  closeAdaptiveVelocity: [0, 21, 22, 23, 24, 25, 26, 27, 28],
  copyAdaptiveVelocitySupport: [0, 81, 82, 83],
  dilateAdaptiveVelocitySupport: [0, 71, 72, 73, 74, 75, 76, 77],
  constrainAdaptiveVelocity: [0, 31, 32, 33, 34, 35, 36],
  finalizeAdaptiveVelocity: [0, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50],
  finishAdaptiveVelocity: [0, 51, 52, 53],
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
  readonly plan: OctreeLosassoAdaptiveVelocityPlan;
  readonly allocatedBytes: number;
  private readonly statusA: GPUBuffer;
  private readonly statusB: GPUBuffer;
  private readonly supportA: GPUBuffer;
  private readonly supportB: GPUBuffer;
  private readonly acceptedStencils: GPUBuffer;
  private readonly candidateStencils: GPUBuffer;
  private readonly acceptedStencilControl: GPUBuffer;
  private readonly candidateStencilControl: GPUBuffer;
  private readonly fieldParams: readonly GPUBuffer[];
  private readonly readyFieldParams: readonly [GPUBuffer, GPUBuffer];
  private readonly pipelines: Partial<Record<EntryPoint, GPUComputePipeline>> = {};
  private readonly fieldGroups = new Map<FieldName, FieldBindings>();
  private stencilBuildGroups?: StencilBuildBindings;
  private stencilCommitGroups?: StencilCommitBindings;
  private readyAcceptedGroups?: readonly [FieldBindings, FieldBindings];
  private destroyed = false;

  /** Physical air-side reach used by this velocity extension instance. */
  get extensionReach_m(): number { return this.options.extensionReach; }

  /** Diagnostic-only accepted reconstruction topology. These buffers remain
   * owned here; callers may copy them but must never bind them for physics. */
  get acceptedStencilDiagnostics(): Readonly<{
    control: GPUBuffer; records: GPUBuffer;
  }> {
    return Object.freeze({ control: this.acceptedStencilControl,
      records: this.acceptedStencils });
  }

  constructor(private readonly device: GPUDevice,
    private readonly options: WebGPUOctreeLosassoAdaptiveVelocityOptions) {
    this.plan = planOctreeLosassoAdaptiveVelocity(options);
    if (options.accepted.graph.nodeCapacity < this.plan.nodeCapacity
      || options.candidate.graph.nodeCapacity < this.plan.nodeCapacity) {
      throw new RangeError("Adaptive velocity capacity exceeds a surface-graph bank");
    }
    if (!Number.isSafeInteger(options.maximumLeafSize) || options.maximumLeafSize < 1
      || (options.maximumLeafSize & (options.maximumLeafSize - 1)) !== 0) {
      throw new RangeError("Adaptive velocity maximum leaf size must be positive and dyadic");
    }
    if (options.dimensions.some((value) => !Number.isSafeInteger(value) || value < 1)) {
      throw new RangeError("Adaptive velocity dimensions must be positive integers");
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
    this.statusA = device.createBuffer({ label: "Losasso adaptive velocity status A",
      size: 4 * this.plan.nodeCapacity, usage: storage });
    this.statusB = device.createBuffer({ label: "Losasso adaptive velocity status B",
      size: 4 * this.plan.nodeCapacity, usage: storage });
    this.supportA = device.createBuffer({ label: "Losasso adaptive velocity support A",
      size: 4 * this.plan.nodeCapacity, usage: storage });
    this.supportB = device.createBuffer({ label: "Losasso adaptive velocity support B",
      size: 4 * this.plan.nodeCapacity, usage: storage });
    this.acceptedStencils = device.createBuffer({
      label: "Losasso accepted compiled nodal face stencils",
      size: this.plan.stencilBytes, usage: storage,
    });
    this.candidateStencils = device.createBuffer({
      label: "Losasso candidate compiled nodal face stencils",
      size: this.plan.stencilBytes, usage: storage,
    });
    this.acceptedStencilControl = device.createBuffer({
      label: "Losasso accepted nodal face-stencil control",
      size: 4 * OCTREE_LOSASSO_ADAPTIVE_VELOCITY_STENCIL_CONTROL_WORDS, usage: storage,
    });
    this.candidateStencilControl = device.createBuffer({
      label: "Losasso candidate nodal face-stencil control",
      size: 4 * OCTREE_LOSASSO_ADAPTIVE_VELOCITY_STENCIL_CONTROL_WORDS, usage: storage,
    });
    this.receiptBuffer = device.createBuffer({ label: "Losasso adaptive velocity receipts",
      size: this.plan.receiptBytes, usage: storage });
    this.diagnosticBuffer = device.createBuffer({
      label: "Losasso adaptive velocity unresolved diagnostics",
      size: 4 * 4 * OCTREE_LOSASSO_ADAPTIVE_VELOCITY_DIAGNOSTIC_BANK_WORDS,
      usage: storage,
    });
    const createFieldParams = (name: FieldName, mode: 0 | 1 | 2): GPUBuffer => {
      const index = FIELD_INDEX[name];
      const buffer = device.createBuffer({ label: `Losasso adaptive velocity ${name} parameters`,
        size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      const words = new Uint32Array(16);
      words.set([this.plan.nodeCapacity, index & 1,
        index * OCTREE_LOSASSO_ADAPTIVE_VELOCITY_RECEIPT_WORDS,
        this.plan.extensionWaves], 0);
      new Float32Array(words.buffer)[4] = options.extensionReach;
      words[5] = 2;
      words[6] = (name.startsWith("candidate") ? options.candidate.phiComponent
        : options.accepted.phiComponent) ?? 1;
      // 0 = normal accepted, 1 = candidate cold bootstrap, 2 = accepted
      // construction-ready bootstrap. Ready bootstrap has distinct immutable
      // params so a normal accepted rebuild can never silently seed zeros.
      words[7] = mode;
      words.set([...options.dimensions, options.maximumLeafSize], 8);
      const faces = name.startsWith("candidate") ? options.candidate.faces : options.accepted.faces;
      words[12] = faces.faceCapacity;
      words[13] = faces.faceDirectoryCapacity;
      words[14] = faces.migrationStatus ? 1 : 0;
      // Tall Cells 3.3: use the accurate causal solver only in a two-finest-cell
      // interface shell. The sparse harmonic hierarchy closes the remaining
      // transport support without allocating or sweeping a dense domain.
      new Float32Array(words.buffer)[15] = Math.min(options.extensionReach,
        2 * options.minimumCellWidth);
      device.queue.writeBuffer(buffer, 0, words);
      return buffer;
    };
    this.fieldParams = (Object.keys(FIELD_INDEX) as FieldName[]).map((name) =>
      createFieldParams(name, name.startsWith("candidate") ? 1 : 0));
    this.readyFieldParams = [createFieldParams("accepted", 2),
      createFieldParams("predictor", 2)];
    this.allocatedBytes = this.statusA.size + this.statusB.size
      + this.supportA.size + this.supportB.size
      + this.receiptBuffer.size + this.diagnosticBuffer.size
      + this.acceptedStencils.size + this.candidateStencils.size
      + this.acceptedStencilControl.size + this.candidateStencilControl.size
      + [...this.fieldParams, ...this.readyFieldParams]
        .reduce((sum, buffer) => sum + buffer.size, 0);
    this.samplerSource = Object.freeze({ leaves: options.accepted.graph.leaves,
      ownerArena: options.ownerArena, leafLocator: options.accepted.graph.leafLocator,
      velocityArena: this.velocityArena, nodeStrideRecords: 2, acceptedFieldRecord: 0,
      predictorFieldRecord: 1, recordStrideWords: 4,
      leafRecordStrideWords: 16, wgsl: octreeLosassoAdaptiveVelocitySamplerWGSL() });
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
    const candidateParams = this.fieldParams[FIELD_INDEX.candidate]!;
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
    };
    const accepted = this.options.accepted;
    this.stencilCommitGroups = {
      commit: this.createGroup("commitAdaptiveVelocityStencils", [this.candidateStencilControl,
        this.candidateStencils, accepted.graph.control, accepted.faces.control,
        this.acceptedStencils]),
      finish: this.createGroup("finishAdaptiveVelocityStencilCommit",
        [this.candidateStencilControl, accepted.graph.control, accepted.faces.control,
          this.acceptedStencilControl]),
    };
    for (const name of Object.keys(FIELD_INDEX) as FieldName[]) {
      const source = name.startsWith("candidate") ? this.options.candidate : this.options.accepted;
      const faceValues = name === "predictor" || name === "candidatePredictor"
        ? source.faces.predictorValues
        : name === "accepted" && source.faces.carriedValues
          ? source.faces.carriedValues : source.faces.projectedValues;
      this.fieldGroups.set(name, this.createFieldBindings(name, source, faceValues));
    }
    const acceptedCarry = this.options.accepted.faces.carriedValues;
    if (acceptedCarry) {
      this.readyAcceptedGroups = [
        this.createFieldBindings("accepted", this.options.accepted, acceptedCarry,
          this.readyFieldParams[0]),
        this.createFieldBindings("predictor", this.options.accepted, acceptedCarry,
          this.readyFieldParams[1]),
      ];
    }
  }

  private createGroup(entryPoint: EntryPoint, buffers: readonly GPUBuffer[]): GPUBindGroup {
    const pipeline = this.pipelines[entryPoint]; if (!pipeline) throw new Error("Adaptive velocity pipeline missing");
    if (buffers.length !== BINDINGS[entryPoint].length) {
      throw new Error(`Adaptive velocity ${entryPoint} binding count mismatch`);
    }
    return this.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0),
      entries: buffers.map((buffer, index) => ({ binding: BINDINGS[entryPoint][index]!,
        resource: { buffer } })) });
  }

  private createFieldBindings(name: FieldName, source: WebGPUOctreeLosassoAdaptiveVelocityBuildSource,
    faceValues: GPUBuffer, params?: GPUBuffer): FieldBindings {
    const p = params ?? this.fieldParams[FIELD_INDEX[name]]!;
    const candidate = name.startsWith("candidate");
    const stencils = candidate ? this.candidateStencils : this.acceptedStencils;
    const stencilControl = candidate
      ? this.candidateStencilControl : this.acceptedStencilControl;
    return {
      prepare: this.createGroup("prepareAdaptiveVelocity",
        [p, source.graph.control, candidate
          ? this.options.accepted.graph.control : source.graph.control,
        stencilControl, this.receiptBuffer, this.diagnosticBuffer]),
      reconstruct: this.createGroup("reconstructAdaptiveVelocity", [p, source.graph.control,
        source.graph.nodes, stencils, faceValues, source.graph.nodalVelocity, this.statusA,
        this.receiptBuffer, this.supportA, source.faces.migrationStatus ?? faceValues]),
      ...(candidate ? { handoff: this.createGroup("handoffAdaptiveVelocity", [p,
        source.graph.control, source.graph.nodes, source.graph.nodalVelocity, this.statusA,
        this.options.accepted.graph.control, this.options.accepted.graph.nodeDirectory,
        this.options.accepted.graph.nodalVelocity]) } : {}),
      markSupport: this.createGroup("markAdaptiveVelocitySupport", [p, source.graph.control,
        source.transportBandMask, source.graph.constraints, this.supportA, this.diagnosticBuffer,
        source.graph.leaves, source.graph.incidentLeaves]),
      extendAB: this.createGroup("extendAdaptiveVelocity", [p, source.graph.control,
        source.graph.adjacency, source.graph.phi, source.graph.nodalVelocity, this.statusA,
        this.statusB, this.receiptBuffer]),
      extendBA: this.createGroup("extendAdaptiveVelocity", [p, source.graph.control,
        source.graph.adjacency, source.graph.phi, source.graph.nodalVelocity, this.statusB,
        this.statusA, this.receiptBuffer]),
      closeAB: this.createGroup("closeAdaptiveVelocity", [p, source.graph.control,
        source.graph.adjacency, source.graph.phi, source.graph.nodalVelocity, this.statusA,
        this.statusB, this.receiptBuffer, this.supportB]),
      closeBA: this.createGroup("closeAdaptiveVelocity", [p, source.graph.control,
        source.graph.adjacency, source.graph.phi, source.graph.nodalVelocity, this.statusB,
        this.statusA, this.receiptBuffer, this.supportA]),
      copySupportAB: this.createGroup("copyAdaptiveVelocitySupport", [p,
        source.graph.control, this.supportA, this.supportB]),
      copySupportBA: this.createGroup("copyAdaptiveVelocitySupport", [p,
        source.graph.control, this.supportB, this.supportA]),
      dilateSupportAB: this.createGroup("dilateAdaptiveVelocitySupport", [p,
        source.graph.control, source.graph.adjacency, source.graph.constraints,
        this.statusA, this.supportA, this.supportB, this.diagnosticBuffer]),
      dilateSupportBA: this.createGroup("dilateAdaptiveVelocitySupport", [p,
        source.graph.control, source.graph.adjacency, source.graph.constraints,
        this.statusB, this.supportB, this.supportA, this.diagnosticBuffer]),
      constrainA: this.createGroup("constrainAdaptiveVelocity", [p, source.graph.control,
        source.graph.constraints, source.graph.constraints, source.graph.nodalVelocity,
        this.statusA, this.receiptBuffer]),
      constrainB: this.createGroup("constrainAdaptiveVelocity", [p, source.graph.control,
        source.graph.constraints, source.graph.constraints, source.graph.nodalVelocity,
        this.statusB, this.receiptBuffer]),
      finalize: this.createGroup("finalizeAdaptiveVelocity", [p, source.graph.control,
        source.graph.phi, source.graph.nodalVelocity, this.statusA, this.receiptBuffer,
        source.graph.nodeValidity, this.supportA, source.graph.constraints,
        source.graph.adjacency, this.diagnosticBuffer]),
      finish: this.createGroup("finishAdaptiveVelocity", [p, this.receiptBuffer,
        source.graph.control, source.graph.leafLocator]),
    };
  }

  encodeAcceptedFields(broker: PassBroker): void {
    this.encodeAcceptedField(broker); this.encodePredictorField(broker);
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
    const groups = this.readyAcceptedGroups;
    if (!groups) throw new Error("Adaptive velocity has no committed carried face field");
    this.encodeField(broker, "accepted", groups[0]);
    this.encodeField(broker, "predictor", groups[1]);
  }
  /** Compile topology/geometry-only candidate face lookups once per graph publication. */
  encodeCandidateStencils(broker: PassBroker): void {
    const groups = this.stencilBuildGroups;
    if (!groups) throw new Error("Adaptive velocity stencil compiler is not initialized");
    const run = (entryPoint: EntryPoint, group: GPUBindGroup, indirect = false) => {
      const pass = broker.compute({ label: `Losasso - ${entryPoint}` });
      pass.setPipeline(this.pipelines[entryPoint]!); pass.setBindGroup(0, group);
      if (indirect) pass.dispatchWorkgroupsIndirect(this.options.candidate.graph.control,
        this.options.candidate.graph.nodeDispatchOffsetBytes);
      else pass.dispatchWorkgroups(1);
    };
    run("prepareAdaptiveVelocityStencils", groups.prepare);
    run("compileAdaptiveVelocityStencils", groups.compile, true);
    run("finishAdaptiveVelocityStencils", groups.finish);
  }
  /** Reconstruct both candidate banks against the already-compiled topology lookups. */
  encodeCandidateFieldRound(broker: PassBroker): void {
    this.encodeField(broker, "candidate"); this.encodeField(broker, "candidatePredictor");
  }
  /** Combined single-round entry point retained for standalone candidate construction. */
  encodeCandidateFields(broker: PassBroker): void {
    this.encodeCandidateStencils(broker);
    this.encodeCandidateFieldRound(broker);
  }
  private encodeReadyStencilCommit(broker: PassBroker): void {
    const groups = this.stencilCommitGroups;
    if (!groups) throw new Error("Adaptive velocity stencil commit is not initialized");
    const commit = broker.compute({ label: "Losasso - commit adaptive velocity stencils" });
    commit.setPipeline(this.pipelines.commitAdaptiveVelocityStencils!);
    commit.setBindGroup(0, groups.commit);
    commit.dispatchWorkgroupsIndirect(this.options.accepted.graph.control,
      this.options.accepted.graph.nodeDispatchOffsetBytes);
    const finish = broker.compute({ label: "Losasso - finish adaptive velocity stencil commit" });
    finish.setPipeline(this.pipelines.finishAdaptiveVelocityStencilCommit!);
    finish.setBindGroup(0, groups.finish); finish.dispatchWorkgroups(1);
  }
  private encodeField(broker: PassBroker, name: FieldName,
    selectedGroups?: FieldBindings): void {
    this.assertReady(); const groups = selectedGroups ?? this.fieldGroups.get(name)!;
    const source = name.startsWith("candidate") ? this.options.candidate : this.options.accepted;
    const run = (entryPoint: EntryPoint, group: GPUBindGroup, indirect = false) => {
      const pass = broker.compute({ label: `Losasso - adaptive velocity ${name} ${entryPoint}` });
      pass.setPipeline(this.pipelines[entryPoint]!); pass.setBindGroup(0, group);
      if (indirect) pass.dispatchWorkgroupsIndirect(source.graph.control,
        source.graph.nodeDispatchOffsetBytes);
      else pass.dispatchWorkgroups(1);
    };
    run("prepareAdaptiveVelocity", groups.prepare);
    run("reconstructAdaptiveVelocity", groups.reconstruct, true);
    if (groups.handoff) run("handoffAdaptiveVelocity", groups.handoff, true);
    run("constrainAdaptiveVelocity", groups.constrainA, true);
    run("markAdaptiveVelocitySupport", groups.markSupport, true);
    for (let wave = 0; wave < this.plan.accurateExtensionWaves; wave += 1) {
      const odd = (wave & 1) !== 0;
      run("extendAdaptiveVelocity", odd ? groups.extendBA : groups.extendAB, true);
      run("constrainAdaptiveVelocity", odd ? groups.constrainA : groups.constrainB, true);
    }
    // The causal pass deliberately rejects strictly farther |phi| donors. A
    // second compact harmonic closure resolves only the discrete basins that
    // therefore remain trial, while accepted causal values stay immutable.
    for (let wave = 0; wave < this.plan.extensionWaves; wave += 1) {
      const odd = (wave & 1) !== 0;
      run("copyAdaptiveVelocitySupport", odd ? groups.copySupportBA : groups.copySupportAB, true);
      run("dilateAdaptiveVelocitySupport",
        odd ? groups.dilateSupportBA : groups.dilateSupportAB, true);
      run("closeAdaptiveVelocity", odd ? groups.closeBA : groups.closeAB, true);
      run("constrainAdaptiveVelocity", odd ? groups.constrainA : groups.constrainB, true);
    }
    run("finalizeAdaptiveVelocity", groups.finalize, true);
    run("finishAdaptiveVelocity", groups.finish);
  }

  destroy(): void {
    if (this.destroyed) return; this.destroyed = true;
    this.statusA.destroy(); this.statusB.destroy();
    this.supportA.destroy(); this.supportB.destroy();
    this.acceptedStencils.destroy(); this.candidateStencils.destroy();
    this.acceptedStencilControl.destroy(); this.candidateStencilControl.destroy();
    this.receiptBuffer.destroy(); this.diagnosticBuffer.destroy();
    for (const buffer of [...this.fieldParams, ...this.readyFieldParams]) buffer.destroy();
    this.fieldGroups.clear(); this.readyAcceptedGroups = undefined;
    this.stencilBuildGroups = undefined; this.stencilCommitGroups = undefined;
  }
  private assertLive(): void { if (this.destroyed) throw new Error("Adaptive velocity is destroyed"); }
  private assertReady(): void { this.assertLive(); if (!this.pipelines.prepareAdaptiveVelocity) {
    throw new Error("Adaptive velocity is not initialized");
  } }
}
