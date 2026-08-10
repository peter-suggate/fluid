import type { PassBroker } from "./webgpu-pass-broker";
import { gpuCompilationManagerFor } from "./gpu-compilation-manager";
import type { LosassoSurfaceGraphBankSource, LosassoSurfaceGraphSources } from "./webgpu-octree-losasso-surface-graph";
import { octreeLosassoAdaptiveMassWGSL } from "./webgpu-octree-losasso-adaptive-mass.wgsl";

export const OCTREE_LOSASSO_ADAPTIVE_MASS_MAGIC = 0x414d_4153;
export const OCTREE_LOSASSO_ADAPTIVE_MASS_CONTROL_WORDS = 32;
export const OCTREE_LOSASSO_ADAPTIVE_MASS_RECEIPT_WORDS = 32;
const ADAPTIVE_MASS_UNITS_PER_FINEST_CELL = 65_536;
const ADAPTIVE_MASS_TRANSFER_RECORDS_PER_LEAF = 20;

export const adaptiveMassControlLayout = Object.freeze({
  magic: 0, topologyEpoch: 1, surfaceGeneration: 2, leafCount: 3,
  donorCount: 4, transferCount: 5, missingRecipients: 6, valid: 7, errorBits: 12,
} as const);

export const adaptiveMassReceiptLayout = Object.freeze({
  acceptedMassBits: 0, transportedMassBits: 1, signedTransportDriftBits: 2,
  maximumDonorWeightDefectBits: 4, donorCount: 5, transferCount: 6,
  missingRecipients: 7, handoffSourceMassBits: 8, handoffTargetMassBits: 9,
  signedHandoffDriftBits: 10, handoffLeafCount: 11, errorBits: 12,
  reconstructionThresholdBits: 13, reconstructionTargetUnits: 14,
  reconstructionMeasuredUnits: 15, reconstructionSignMismatches: 16,
  firstReconstructionSignMismatchItem: 17, handoffGraphErrors: 18,
} as const);

const floatWord = (word: number): number => {
  const bits = new Uint32Array(1); bits[0] = word >>> 0;
  return new Float32Array(bits.buffer)[0]!;
};

export function unpackAdaptiveMassReceipt(words: ArrayLike<number>) {
  if (words.length < OCTREE_LOSASSO_ADAPTIVE_MASS_RECEIPT_WORDS) {
    throw new RangeError(`adaptive mass receipt requires ${OCTREE_LOSASSO_ADAPTIVE_MASS_RECEIPT_WORDS} words`);
  }
  return Object.freeze({
    acceptedMass_m3: floatWord(words[0]!), transportedMass_m3: floatWord(words[1]!),
    signedTransportDrift_m3: floatWord(words[2]!),
    maximumDonorWeightDefect: floatWord(words[4]!), donors: words[5]! >>> 0,
    transfers: words[6]! >>> 0, missingRecipients: words[7]! >>> 0,
    handoffSourceMass_m3: floatWord(words[8]!), handoffTargetMass_m3: floatWord(words[9]!),
    signedHandoffDrift_m3: floatWord(words[10]!), handoffLeafCount: words[11]! >>> 0,
    errors: words[12]! >>> 0,
    reconstructionThreshold: floatWord(words[13]!),
    reconstructionTargetUnits: words[14]! >>> 0,
    reconstructionMeasuredUnits: words[15]! >>> 0,
    reconstructionSignMismatches: words[16]! >>> 0,
    firstReconstructionSignMismatchItem: words[17]! >>> 0,
    handoffGraphErrors: words[18]! >>> 0,
  });
}

export interface WebGPUOctreeLosassoAdaptiveMassVelocitySource {
  /** Interleaved accepted/predictor vec4f records, two records per compact node. */
  readonly values: GPUBuffer;
  readonly record?: 0 | 1;
}

export interface WebGPUOctreeLosassoAdaptiveMassOptions {
  readonly dimensions: readonly [number, number, number];
  readonly maximumLeafSpan: number;
  readonly cellSize: number;
  readonly domainOrigin?: readonly [number, number, number];
  readonly leafCapacity: number;
  readonly nodeCapacity: number;
  readonly pressureRowCapacity?: number;
  /** Stable owner-page arena used by the graph's compiled cell-to-leaf locator. */
  readonly ownerArena: GPUBuffer;
  /** Fixed sparse transfer capacity. Twenty records cover a strict-2:1 tensor fan. */
  readonly transferCapacity?: number;
  readonly massEpsilon?: number;
  /** Disable the current air-side concentration pass while retaining exact
   * conservative transport. The unsharpened path finalizes from nextMass. */
  readonly sharpeningEnabled?: boolean;
}

export interface WebGPUOctreeLosassoAdaptiveMassSource {
  readonly control: GPUBuffer;
  readonly receipts: GPUBuffer;
  /** Graph-owned accepted/candidate integral surface mass. */
  readonly acceptedMass: GPUBuffer;
  readonly acceptedCompression: GPUBuffer;
  readonly candidateMass: GPUBuffer;
  readonly candidateCompression: GPUBuffer;
  /** vec4f(rho, local pseudo-phi, mass, compression), leaf indexed. */
  readonly leafRhoPhi: GPUBuffer;
  /** vec4f(rho,mass,geometric leaf volume,compression), pressure-row indexed. */
  readonly rowRho: GPUBuffer;
  /** vec4u(bitcast(centre),bitcast(min),bitcast(max),flags), pressure-row indexed. */
  readonly rowPhi: GPUBuffer;
  /** Diagnostic sparse donor/recipient transfer records. */
  readonly transferRecords: GPUBuffer;
  /** Fail-only QA view: high bit is face reach, low 31 bits remote tentative units. */
  readonly transportAdmission: GPUBuffer;
}

export interface OctreeLosassoAdaptiveMassPlan {
  readonly leafCapacity: number;
  readonly nodeCapacity: number;
  readonly pressureRowCapacity: number;
  readonly transferCapacity: number;
  readonly leafWorkgroups: number;
  readonly nodeWorkgroups: number;
  readonly rowWorkgroups: number;
  readonly transferWorkgroups: number;
  readonly allocatedBytes: number;
  readonly physicsAllocationScalesWithGraph: true;
}

type PipelineName = "prepareBootstrap" | "bootstrapMassFromPhi" | "finishBootstrap"
  | "prepareTransport" | "clearTransport" | "markTransportSurfaceReach"
  | "buildTransfers" | "scatterPredictedCompression" | "scatterTransfers"
  | "finalizeTransfers" | "returnTransferRemainders" | "clearTransportCompression"
  | "scatterAcceptedCompression" | "finishTransportCompression"
  | "validateTransportLeaves" | "clearSharpen" | "buildSharpenTransfers"
  | "scatterSharpenTransfers" | "validateSharpenLeaves"
  | "finishTransportLeaves" | "finishUnsharpenedTransportLeaves" | "finishTransport"
  | "prepareHandoff" | "handoffMass"
  | "finishHandoffLeaves" | "finishHandoff" | "deriveLeafRhoPhi" | "deriveNodalPseudoPhi"
  | "validateNodalPseudoPhi"
  | "projectNodalPseudoPhi" | "prepareReconstruction" | "clearReconstructionMeasure"
  | "measureReconstructionVolume" | "finishReconstructionIteration" | "deriveRows";

const positive = (value: number, label: string): number => {
  if (!Number.isSafeInteger(value) || value < 1) throw new RangeError(`${label} must be a positive integer`);
  return value;
};

/**
 * Chentanez--Mueller surface-mass authority on compact graph leaves.
 *
 * The graph owns stable mass/compression banks and copies candidate values as
 * part of its topology commit. This operator initializes/advances those banks,
 * supplies conservative handoff, and materializes nodal pseudo-phi one-way for
 * legacy pressure/topology/render consumers.
 */
export class WebGPUOctreeLosassoAdaptiveMass {
  readonly plan: OctreeLosassoAdaptiveMassPlan;
  readonly source: WebGPUOctreeLosassoAdaptiveMassSource;
  private readonly params: GPUBuffer;
  private readonly advanceParams: GPUBuffer;
  private readonly acceptedDerivedParams: GPUBuffer;
  private readonly candidateDerivedParams: GPUBuffer;
  private readonly nextMass: GPUBuffer;
  private readonly nextCompressionMass: GPUBuffer;
  private readonly transportAdmission: GPUBuffer;
  private readonly transferRemainders: GPUBuffer;
  /** Eight atomic words; the fixed rho=.5 midpoint is a one-way phi-cache value. */
  private readonly reconstruction: GPUBuffer;
  private readonly emptyVelocity: GPUBuffer;
  private pipelines?: Readonly<Record<PipelineName, GPUComputePipeline>>;
  private readonly bindGroups = new Map<string, GPUBindGroup>();
  private destroyed = false;

  constructor(private readonly device: GPUDevice, readonly graph: LosassoSurfaceGraphSources,
    readonly options: WebGPUOctreeLosassoAdaptiveMassOptions) {
    const leafCapacity = positive(options.leafCapacity, "adaptive mass leaf capacity");
    const nodeCapacity = positive(options.nodeCapacity, "adaptive mass node capacity");
    const pressureRowCapacity = positive(options.pressureRowCapacity
      ?? graph.accepted.pressureRowCapacity, "adaptive mass pressure row capacity");
    const transferCapacity = positive(options.transferCapacity
      ?? ADAPTIVE_MASS_TRANSFER_RECORDS_PER_LEAF * leafCapacity,
      "adaptive mass transfer capacity");
    if (transferCapacity < ADAPTIVE_MASS_TRANSFER_RECORDS_PER_LEAF * leafCapacity) {
      throw new RangeError("adaptive mass transfer capacity must reserve twenty records per possible donor");
    }
    if (!(options.cellSize > 0) || !Number.isFinite(options.cellSize)) {
      throw new RangeError("adaptive mass cell size must be finite and positive");
    }
    positive(options.maximumLeafSpan, "adaptive mass maximum leaf span");
    options.dimensions.forEach((value, axis) => positive(value, `adaptive mass dimension ${axis}`));
    const finestCellCount = options.dimensions.reduce((product, value) => product * value, 1);
    const fullDomainMassUnits = finestCellCount * ADAPTIVE_MASS_UNITS_PER_FINEST_CELL;
    if (!Number.isSafeInteger(fullDomainMassUnits) || fullDomainMassUnits > 0xffff_ffff) {
      throw new RangeError("adaptive mass fixed-point domain capacity exceeds u32");
    }
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    this.params = device.createBuffer({ label: "Adaptive surface-mass parameters", size: 80,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.advanceParams = device.createBuffer({ label: "Adaptive surface-mass advance parameters", size: 80,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.acceptedDerivedParams = device.createBuffer({
      label: "Adaptive accepted mass-derived cache parameters", size: 80,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.candidateDerivedParams = device.createBuffer({
      label: "Adaptive candidate mass-derived cache parameters", size: 80,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const control = device.createBuffer({ label: "Adaptive surface-mass control",
      size: 4 * OCTREE_LOSASSO_ADAPTIVE_MASS_CONTROL_WORDS, usage: storage });
    const receipts = device.createBuffer({ label: "Adaptive surface-mass receipts",
      size: 4 * OCTREE_LOSASSO_ADAPTIVE_MASS_RECEIPT_WORDS, usage: storage });
    this.nextMass = device.createBuffer({ label: "Adaptive surface-mass transport accumulation",
      size: 4 * leafCapacity, usage: storage });
    this.nextCompressionMass = device.createBuffer({ label: "Adaptive compression transport accumulation",
      size: 4 * leafCapacity, usage: storage });
    this.transportAdmission = device.createBuffer({ label: "Adaptive transport surface admission",
      size: 4 * leafCapacity, usage: storage });
    this.reconstruction = device.createBuffer({ label: "Adaptive nodal reconstruction measurement",
      size: 32, usage: storage });
    const transferRecords = device.createBuffer({ label: "Adaptive conservative transfer records",
      size: 16 * transferCapacity, usage: storage });
    this.transferRemainders = device.createBuffer({ label: "Adaptive conservative transfer remainders",
      size: 4 * transferCapacity, usage: storage });
    const leafRhoPhi = device.createBuffer({ label: "Adaptive leaf rho and pseudo-phi",
      size: 16 * leafCapacity, usage: storage });
    const rowRho = device.createBuffer({ label: "Adaptive pressure-row surface density",
      size: 16 * pressureRowCapacity, usage: storage });
    const rowPhi = device.createBuffer({ label: "Adaptive pressure-row mass pseudo-phi",
      size: 16 * pressureRowCapacity, usage: storage });
    this.emptyVelocity = device.createBuffer({ label: "Adaptive mass inert velocity",
      size: Math.max(16, 32 * nodeCapacity), usage: storage });
    this.source = Object.freeze({ control, receipts,
      acceptedMass: graph.accepted.surfaceMass,
      acceptedCompression: graph.accepted.surfaceCompression,
      candidateMass: graph.candidate.surfaceMass,
      candidateCompression: graph.candidate.surfaceCompression,
      leafRhoPhi, rowRho, rowPhi, transferRecords,
      transportAdmission: this.transportAdmission });
    const dispatch = (n: number) => Math.min(Math.ceil(n / 64), device.limits.maxComputeWorkgroupsPerDimension);
    this.plan = Object.freeze({ leafCapacity, nodeCapacity, pressureRowCapacity, transferCapacity,
      leafWorkgroups: dispatch(leafCapacity), nodeWorkgroups: dispatch(nodeCapacity),
      rowWorkgroups: dispatch(pressureRowCapacity), transferWorkgroups: dispatch(transferCapacity),
      allocatedBytes: 320 + control.size + receipts.size + this.nextMass.size
        + this.nextCompressionMass.size + this.transportAdmission.size
        + transferRecords.size + this.transferRemainders.size + leafRhoPhi.size
        + rowRho.size + rowPhi.size + this.emptyVelocity.size + this.reconstruction.size,
      physicsAllocationScalesWithGraph: true as const });
    this.writeParams(this.params, 0, 0, 0);
    this.writeParams(this.advanceParams, 0, 0, 0);
    this.writeParams(this.acceptedDerivedParams, 0, 0, 0, 2, false);
    this.writeParams(this.candidateDerivedParams, 0, 0, 0, 2, true);
  }

  get initializationTasks(): readonly { readonly label: string; readonly run: () => Promise<void> }[] {
    return [{ label: "Compile adaptive conservative surface-mass operators", run: () => this.initialize() }];
  }

  async initialize(): Promise<void> {
    this.assertLive(); if (this.pipelines) return;
    const compiler = gpuCompilationManagerFor(this.device);
    const shaderModule = compiler.createShaderModule({ label: "Adaptive surface-mass operators",
      code: octreeLosassoAdaptiveMassWGSL });
    const names: readonly PipelineName[] = ["prepareBootstrap", "bootstrapMassFromPhi", "finishBootstrap",
      "prepareTransport", "clearTransport", "markTransportSurfaceReach", "buildTransfers",
      "scatterPredictedCompression", "scatterTransfers", "finalizeTransfers", "returnTransferRemainders",
      "clearTransportCompression", "scatterAcceptedCompression", "finishTransportCompression",
      "validateTransportLeaves",
      "clearSharpen", "buildSharpenTransfers", "scatterSharpenTransfers", "validateSharpenLeaves",
      "finishTransportLeaves", "finishUnsharpenedTransportLeaves", "finishTransport",
      "prepareHandoff", "handoffMass",
      "finishHandoffLeaves", "finishHandoff",
      "deriveLeafRhoPhi", "prepareReconstruction", "clearReconstructionMeasure",
      "deriveNodalPseudoPhi", "validateNodalPseudoPhi", "projectNodalPseudoPhi",
      "measureReconstructionVolume",
      "finishReconstructionIteration", "deriveRows"];
    const pipelines = {} as Record<PipelineName, GPUComputePipeline>;
    for (const name of names) {
      const descriptor: GPUComputePipelineDescriptor = {
        label: `Adaptive mass - ${name}`, layout: "auto",
        compute: { module: shaderModule, entryPoint: name },
      };
      pipelines[name] = await compiler.compileComputePipeline(
        descriptor, { priority: "critical" });
    }
    this.pipelines = Object.freeze(pipelines);
  }

  /** Bootstrap a graph-owned mass bank from one component of its nodal phi cache. */
  encodeBootstrap(broker: PassBroker, bank: "accepted" | "candidate", phiComponent: 0 | 1 = 0): void {
    this.assertInitialized(); this.writeParams(this.params, 0, 0, phiComponent);
    const target = this.graph[bank];
    this.run(broker, "prepareBootstrap", target, this.graph.accepted, this.emptyVelocity, 1);
    this.run(broker, "bootstrapMassFromPhi", target, this.graph.accepted, this.emptyVelocity,
      this.plan.leafWorkgroups);
    this.run(broker, "finishBootstrap", target, this.graph.accepted, this.emptyVelocity, 1);
    this.encodeDerivedOutputs(broker, bank, phiComponent);
  }

  /** Sparse conservative accepted-mass transport; donor columns sum to one. */
  encodeAcceptedAdvance(broker: PassBroker, dt: number,
    velocity: WebGPUOctreeLosassoAdaptiveMassVelocitySource): void {
    if (!(dt >= 0) || !Number.isFinite(dt)) throw new RangeError("adaptive mass dt must be finite and non-negative");
    this.assertInitialized(); this.writeParams(this.advanceParams, dt, velocity.record ?? 0, 0);
    const bank = this.graph.accepted;
    this.run(broker, "prepareTransport", bank, bank, velocity.values, 1, this.advanceParams);
    this.run(broker, "clearTransport", bank, bank, velocity.values, this.plan.leafWorkgroups, this.advanceParams);
    this.run(broker, "buildTransfers", bank, bank, velocity.values, this.plan.leafWorkgroups, this.advanceParams);
    this.run(broker, "scatterPredictedCompression", bank, bank, velocity.values,
      this.plan.transferWorkgroups, this.advanceParams);
    this.run(broker, "scatterTransfers", bank, bank, velocity.values, this.plan.transferWorkgroups, this.advanceParams);
    this.run(broker, "markTransportSurfaceReach", bank, bank, velocity.values,
      this.plan.leafWorkgroups, this.advanceParams);
    this.run(broker, "finalizeTransfers", bank, bank, velocity.values,
      this.plan.transferWorkgroups, this.advanceParams);
    this.run(broker, "returnTransferRemainders", bank, bank, velocity.values,
      this.plan.transferWorkgroups, this.advanceParams);
    this.run(broker, "clearTransportCompression", bank, bank, velocity.values,
      this.plan.leafWorkgroups, this.advanceParams);
    this.run(broker, "scatterAcceptedCompression", bank, bank, velocity.values,
      this.plan.transferWorkgroups, this.advanceParams);
    this.run(broker, "finishTransportCompression", bank, bank, velocity.values,
      this.plan.leafWorkgroups, this.advanceParams);
    this.run(broker, "validateTransportLeaves", bank, bank, velocity.values, this.plan.leafWorkgroups, this.advanceParams);
    // Chentanez--Mueller sharpening removes density only on the air side of
    // rho=.5 and scatters every removed fixed-point unit only where the local
    // gradient trace reaches that existing contour (paper D=1.1h). The donor
    // keeps all units when no contour is reached, preventing diffuse tails from
    // sharpening themselves into remote liquid while remaining exactly local,
    // conservative, and covariant under axis permutations.
    if (this.options.sharpeningEnabled !== false) {
      this.run(broker, "clearSharpen", bank, bank, velocity.values,
        this.plan.leafWorkgroups, this.advanceParams);
      this.run(broker, "buildSharpenTransfers", bank, bank, velocity.values,
        this.plan.leafWorkgroups, this.advanceParams);
      this.run(broker, "scatterSharpenTransfers", bank, bank, velocity.values,
        this.plan.transferWorkgroups, this.advanceParams);
      this.run(broker, "validateSharpenLeaves", bank, bank, velocity.values,
        this.plan.leafWorkgroups, this.advanceParams);
      this.run(broker, "finishTransportLeaves", bank, bank, velocity.values,
        this.plan.leafWorkgroups, this.advanceParams);
    } else {
      this.run(broker, "finishUnsharpenedTransportLeaves", bank, bank, velocity.values,
        this.plan.leafWorkgroups, this.advanceParams);
    }
    this.run(broker, "finishTransport", bank, bank, velocity.values, 1, this.advanceParams);
  }

  /** Exact-volume overlap remap from accepted leaves to every live candidate leaf. */
  encodeCandidateHandoff(broker: PassBroker): void {
    this.assertInitialized(); this.writeParams(this.params, 0, 0, 0);
    const target = this.graph.candidate, source = this.graph.accepted;
    this.run(broker, "prepareHandoff", target, source, this.emptyVelocity, 1);
    this.run(broker, "clearTransport", target, source, this.emptyVelocity, this.plan.leafWorkgroups);
    this.run(broker, "handoffMass", target, source, this.emptyVelocity, this.plan.leafWorkgroups);
    this.run(broker, "finishHandoffLeaves", target, source, this.emptyVelocity, this.plan.leafWorkgroups);
    this.run(broker, "finishHandoff", target, source, this.emptyVelocity, 1);
  }

  /**
   * One-way materialization for existing graph consumers. The selected phi
   * component is a cache; it must never be transported back into surface mass.
   */
  encodeDerivedOutputs(broker: PassBroker, bank: "accepted" | "candidate",
    phiTargetComponent: 0 | 1 | "both" = "both",
    mode: "reconstruct" | "preserve-and-validate" = "reconstruct"): void {
    const derivedParams = bank === "candidate"
      ? this.candidateDerivedParams : this.acceptedDerivedParams;
    this.assertInitialized(); this.writeParams(derivedParams, 0, 0, 0,
      phiTargetComponent === "both" ? 2 : phiTargetComponent, bank === "candidate");
    const target = this.graph[bank];
    // Keep target/source banks distinct so this materialization never aliases
    // a writable graph binding with a read-only handoff binding.
    const source = bank === "candidate" ? this.graph.accepted : this.graph.candidate;
    this.run(broker, "deriveLeafRhoPhi", target, source, this.emptyVelocity,
      this.plan.leafWorkgroups, derivedParams);
    // Reconstruction always uses the physical rho=.5 contour. A global
    // volume-fit threshold is topology-dependent: changing only the leaf
    // partition can move every surface node. Measure once for diagnostics,
    // but never feed that measurement back into the iso-value.
    this.run(broker, "prepareReconstruction", target, source, this.emptyVelocity, 1,
      derivedParams);
    if (mode === "preserve-and-validate") {
      this.run(broker, "validateNodalPseudoPhi", target, source, this.emptyVelocity,
        this.plan.nodeWorkgroups, derivedParams);
    } else {
      this.run(broker, "deriveNodalPseudoPhi", target, source, this.emptyVelocity,
        this.plan.nodeWorkgroups, derivedParams);
      this.run(broker, "projectNodalPseudoPhi", target, source, this.emptyVelocity,
        this.plan.nodeWorkgroups, derivedParams);
    }
    this.run(broker, "clearReconstructionMeasure", target, source,
      this.emptyVelocity, 1, derivedParams);
    this.run(broker, "measureReconstructionVolume", target, source,
      this.emptyVelocity, this.plan.leafWorkgroups, derivedParams);
    this.run(broker, "finishReconstructionIteration", target, source,
      this.emptyVelocity, 1, derivedParams);
    this.run(broker, "deriveRows", target, source, this.emptyVelocity,
      this.plan.rowWorkgroups, derivedParams);
  }

  destroy(): void {
    if (this.destroyed) return; this.destroyed = true;
    this.params.destroy(); this.advanceParams.destroy(); this.acceptedDerivedParams.destroy();
    this.candidateDerivedParams.destroy(); this.nextMass.destroy(); this.nextCompressionMass.destroy();
    this.transportAdmission.destroy(); this.transferRemainders.destroy();
    this.reconstruction.destroy();
    this.emptyVelocity.destroy();
    this.source.control.destroy(); this.source.receipts.destroy(); this.source.leafRhoPhi.destroy();
    this.source.rowRho.destroy(); this.source.rowPhi.destroy(); this.source.transferRecords.destroy();
    this.bindGroups.clear();
  }

  private buffers(target: LosassoSurfaceGraphBankSource, source: LosassoSurfaceGraphBankSource,
    velocity: GPUBuffer, params: GPUBuffer): readonly GPUBuffer[] {
    return [params, target.control, target.leaves, target.leafDirectory, target.nodes,
      target.constraints, target.incidentLeaves, target.pressureRowToGraphLeaf, target.phi, velocity,
      target.surfaceMass, target.surfaceCompression, source.control, source.leaves,
      source.surfaceMass, source.surfaceCompression, this.nextMass, this.nextCompressionMass,
      this.source.transferRecords, this.source.control, this.source.receipts, this.source.leafRhoPhi,
      this.source.rowRho, this.source.rowPhi, this.reconstruction, this.options.ownerArena,
      target.leafLocator, this.transportAdmission, this.transferRemainders];
  }

  private run(broker: PassBroker, name: PipelineName, target: LosassoSurfaceGraphBankSource,
    source: LosassoSurfaceGraphBankSource, velocity: GPUBuffer, workgroups: number,
    params: GPUBuffer = this.params): void {
    const pipeline = this.pipelines![name];
    const parameterBank = params === this.advanceParams ? "d"
      : params === this.acceptedDerivedParams ? "a"
        : params === this.candidateDerivedParams ? "c" : "p";
    const key = `${name}:${target === this.graph.candidate ? "c" : "a"}:${source === this.graph.candidate ? "c" : "a"}:${velocity === this.emptyVelocity ? "e" : "v"}:${parameterBank}`;
    let group = this.bindGroups.get(key);
    if (!group) {
      const buffers = this.buffers(target, source, velocity, params);
      const bindings = this.bindingSet(name);
      group = this.device.createBindGroup({ label: `${pipeline.label} bindings`,
        layout: pipeline.getBindGroupLayout(0), entries: bindings.map((binding) =>
          ({ binding, resource: { buffer: buffers[binding]! } })) });
      this.bindGroups.set(key, group);
    }
    const pass = broker.compute({ label: pipeline.label }); pass.setPipeline(pipeline);
    pass.setBindGroup(0, group); pass.dispatchWorkgroups(Math.max(1, workgroups));
  }

  private bindingSet(name: PipelineName): readonly number[] {
    return ({
      prepareBootstrap: [19, 20],
      bootstrapMassFromPhi: [0, 1, 2, 8, 10, 11, 19],
      finishBootstrap: [0, 1, 2, 10, 19, 20],
      prepareTransport: [0, 1, 2, 10, 19, 20], clearTransport: [0, 1, 2, 16, 17, 27],
      markTransportSurfaceReach: [0, 1, 2, 10, 16, 25, 26, 27],
      buildTransfers: [0, 1, 2, 8, 9, 10, 18, 19, 25, 26, 28],
      scatterPredictedCompression: [0, 1, 2, 10, 11, 17, 18, 19],
      scatterTransfers: [0, 1, 2, 16, 17, 18, 19, 27, 28],
      finalizeTransfers: [2, 16, 18, 19, 27, 28],
      returnTransferRemainders: [0, 1, 2, 16, 18, 19, 28],
      clearTransportCompression: [0, 1, 2, 17],
      scatterAcceptedCompression: [0, 1, 2, 10, 11, 17, 18, 19, 28],
      finishTransportCompression: [0, 1, 2, 11, 17, 19],
      validateTransportLeaves: [0, 1, 2, 16, 19],
      clearSharpen: [0, 1, 2, 17],
      buildSharpenTransfers: [0, 1, 2, 16, 18, 19, 25, 26],
      scatterSharpenTransfers: [0, 1, 2, 17, 18, 19],
      validateSharpenLeaves: [0, 1, 2, 17, 19],
      finishTransportLeaves: [0, 1, 2, 10, 17, 19],
      finishUnsharpenedTransportLeaves: [0, 1, 2, 10, 16, 19],
      finishTransport: [0, 1, 2, 10, 19, 20], prepareHandoff: [19, 20],
      handoffMass: [0, 1, 2, 12, 13, 14, 15, 16, 17, 19, 20],
      finishHandoffLeaves: [0, 1, 2, 10, 11, 16, 17],
      finishHandoff: [0, 1, 2, 12, 13, 14, 16, 19, 20],
      deriveLeafRhoPhi: [0, 1, 2, 10, 11, 21],
      prepareReconstruction: [0, 1, 2, 10, 19, 24],
      clearReconstructionMeasure: [24],
      deriveNodalPseudoPhi: [0, 1, 2, 5, 6, 8, 10, 19, 24],
      validateNodalPseudoPhi: [0, 1, 2, 4, 5, 6, 8, 10, 19, 24],
      projectNodalPseudoPhi: [0, 1, 5, 8, 19],
      measureReconstructionVolume: [0, 1, 2, 8, 19, 24],
      finishReconstructionIteration: [1, 19, 20, 24],
      deriveRows: [0, 1, 2, 7, 10, 11, 19, 21, 22, 23],
    } satisfies Record<PipelineName, readonly number[]>)[name];
  }

  private writeParams(target: GPUBuffer, dt: number, velocityRecord: 0 | 1, phiSource: 0 | 1,
    phiTarget: 0 | 1 | 2 = phiSource, topologyHandoff = false): void {
    const raw = new ArrayBuffer(80), u = new Uint32Array(raw), f = new Float32Array(raw);
    u.set([...this.options.dimensions, this.options.maximumLeafSpan], 0);
    f.set([...(this.options.domainOrigin ?? [0, 0, 0]), this.options.cellSize], 4);
    u.set([this.plan.leafCapacity, this.plan.nodeCapacity, this.plan.pressureRowCapacity,
      this.plan.transferCapacity], 8);
    const massQuantum = this.options.cellSize ** 3 / ADAPTIVE_MASS_UNITS_PER_FINEST_CELL;
    f.set([dt, this.options.massEpsilon ?? 1e-10, massQuantum, 1 / massQuantum], 12);
    u.set([phiSource, velocityRecord, phiTarget, topologyHandoff ? 1 : 0], 16);
    this.device.queue.writeBuffer(target, 0, raw);
  }

  private assertInitialized(): void { this.assertLive(); if (!this.pipelines) throw new Error("Adaptive mass pipelines are not initialized"); }
  private assertLive(): void { if (this.destroyed) throw new Error("Adaptive mass authority is destroyed"); }
}
