import type { WebGPUFineLevelSetBrickSource } from "./webgpu-octree-fine-levelset-bricks";
import { fineLevelSetLinearWorkgroupWGSL,
  planFineLevelSetDispatch2D } from "./webgpu-fine-levelset-dispatch";

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
   * as interface topology. Undefined only for legacy eight-word snapshots. */
  interfaceSeedBricks?: number;
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

/** Number of fixed GPU expansions needed to fill an exact Chebyshev ball.
 * Expansion radii grow 0 -> 1 -> 3 -> 7 ... and clamp at the requested
 * radius, so the dispatch count is logarithmic without over-dilating. */
export function planFineLevelSetChebyshevFloodPasses(dilationBrickRings: number): number {
  if (!Number.isSafeInteger(dilationBrickRings) || dilationBrickRings < 0) {
    throw new RangeError("Fine topology dilation radius must be a non-negative integer");
  }
  return dilationBrickRings === 0 ? 0 : Math.ceil(Math.log2(dilationBrickRings + 1));
}

/** Direct seed dilation has better launch/ALU balance while the entire brick
 * lattice still fits in the mini-domain working set. Larger lattices retain
 * logarithmic frontier flooding so a wide band cannot multiply seed work. */
export const FINE_LEVELSET_DIRECT_DILATION_MAXIMUM_BRICKS = 15 ** 3;

/**
 * CPU mirror of the SurfaceLeaf -> global fine-page mapping used by the seed
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

export function fineLevelSetLeafSeedAllocatedBytes(maximumResidentBricks: number, hashCapacity: number): number {
  if (!Number.isSafeInteger(maximumResidentBricks) || maximumResidentBricks < 1) {
    throw new RangeError("Fine seed resident capacity must be positive");
  }
  if (!Number.isSafeInteger(hashCapacity) || hashCapacity < 1 || (hashCapacity & (hashCapacity - 1)) !== 0) {
    throw new RangeError("Fine seed hash capacity must be a positive power of two");
  }
  // The seed ABI itself is unchanged.  Two words per resident page are a
  // private scan arena: this is enough for one block sum per 64 input rows
  // plus the candidate-row bitset without coupling the scan to a CPU readback.
  return (8 + 11 * maximumResidentBricks + 2 * hashCapacity) * 4 + 64;
}

export const FINE_LEVELSET_TOPOLOGY_ALLOCATED_BYTES = 48 + 96 + 8 + 64 + 32;

/** GPU-only bridge from existing compact SurfaceLeaf/core candidates to global brick keys. */
export class WebGPUFineLevelSetLeafSeeds {
  readonly buffer: GPUBuffer;
  readonly allocatedBytes: number;
  private readonly params: GPUBuffer;
  private readonly scratch: GPUBuffer;
  private readonly pipelines: Record<string, GPUComputePipeline>;

  constructor(private readonly device: GPUDevice, readonly target: WebGPUFineLevelSetBrickSource,
    analytic?: { initialCondition: "dam-break" | "tank-fill"; fillFraction: number }) {
    this.allocatedBytes = fineLevelSetLeafSeedAllocatedBytes(
      target.plan.maximumResidentBricks, target.plan.hashCapacity,
    );
    this.buffer = device.createBuffer({ label: "global fine brick seed keys",
      size: (4 + 9 * target.plan.maximumResidentBricks + 2 * target.plan.hashCapacity) * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const descriptor = new ArrayBuffer(8); const descriptorWords = new Uint32Array(descriptor);
    descriptorWords[0] = analytic?.initialCondition === "tank-fill" ? 1
      : analytic?.initialCondition === "dam-break" ? 2 : 0;
    new Float32Array(descriptor)[1] = analytic?.fillFraction ?? 0;
    device.queue.writeBuffer(this.buffer, 8, descriptor);
    this.params = device.createBuffer({ label: "global fine seed parameters", size: 64,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.scratch = device.createBuffer({ label: "global fine deterministic seed scan",
      size: (2 * target.plan.maximumResidentBricks + 4) * 4, usage: GPUBufferUsage.STORAGE });
    const shaderModule = device.createShaderModule({ label: "SurfaceLeaf to global fine seeds", code: fineLevelSetLeafSeedWGSL });
    const entryPoints = ["clearSeedState", "markCandidateRows", "classifyCandidateLeafBlocks",
      "claimCandidateLeafOwners", "claimAllLeafOwners", "classifyAllLeafBlocks",
      "scanLeafSeedBlocks", "emitCandidateLeafSeeds", "emitAllLeafSeeds", "finalizeLeafHash",
      "claimPowerEndpointOwners", "classifyPowerEndpointBlocks", "scanPowerEndpointBlocks",
      "emitPowerEndpointSeeds"];
    this.pipelines = Object.fromEntries(entryPoints.map((entryPoint) => [entryPoint,
      device.createComputePipeline({ label: `Global fine seed ${entryPoint}`, layout: "auto",
        compute: { module: shaderModule, entryPoint } })]));
  }

  private bindingBytes(binding: GPUBufferBinding): number {
    return binding.size ?? binding.buffer.size - (binding.offset ?? 0);
  }

  private writeParams(leafCapacity: number, endpointCapacity: number): { leafBlocks: number; endpointBlocks: number } {
    const plan = this.target.plan;
    const leafBlocks = Math.max(1, Math.ceil(leafCapacity / 64));
    const endpointBlocks = Math.max(1, Math.ceil(endpointCapacity / 64));
    const eligibilityWords = Math.ceil(leafCapacity / 32);
    const requiredScratchWords = Math.max(leafBlocks, endpointBlocks) + 2 + eligibilityWords;
    if (requiredScratchWords > this.scratch.size / 4) {
      throw new RangeError(`Fine seed scan requires ${requiredScratchWords} words, capacity is ${this.scratch.size / 4}`);
    }
    const bytes = new ArrayBuffer(64); const u32 = new Uint32Array(bytes); const f32 = new Float32Array(bytes);
    u32.set([plan.fineFactor, plan.brickResolution, ...plan.brickDimensions,
      plan.maximumResidentBricks, plan.logicalBrickCount, plan.hashCapacity]);
    f32.set([...plan.domainOrigin, plan.fineCellWidth], 8);
    u32.set([leafCapacity, leafBlocks, endpointCapacity, endpointBlocks], 12);
    this.device.queue.writeBuffer(this.params, 0, bytes);
    return { leafBlocks, endpointBlocks };
  }

  private group(entryPoint: string, entries: GPUBindGroupEntry[]): GPUBindGroup {
    const bindings: Record<string, readonly number[]> = {
      clearSeedState: [0, 4, 7], markCandidateRows: [0, 2, 3, 7],
      claimCandidateLeafOwners: [0, 1, 3, 4, 7], claimAllLeafOwners: [0, 1, 3, 4, 7],
      classifyCandidateLeafBlocks: [0, 1, 3, 4, 7], classifyAllLeafBlocks: [0, 1, 3, 4, 7],
      scanLeafSeedBlocks: [0, 4, 7], emitCandidateLeafSeeds: [0, 1, 3, 4, 7],
      emitAllLeafSeeds: [0, 1, 3, 4, 7], claimPowerEndpointOwners: [0, 4, 5, 6],
      finalizeLeafHash: [0, 4],
      classifyPowerEndpointBlocks: [0, 4, 5, 6, 7], scanPowerEndpointBlocks: [0, 4, 7],
      emitPowerEndpointSeeds: [0, 4, 5, 6, 7],
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

  encode(encoder: GPUCommandEncoder, leaves: GPUBufferBinding, candidates: GPUBufferBinding,
    candidateCountAndDispatch: GPUBufferBinding): FineLevelSetGPUSeedSource {
    const leafCapacity = Math.floor(this.bindingBytes(leaves) / 64);
    const candidateCapacity = Math.floor(this.bindingBytes(candidates) / 8);
    const { leafBlocks } = this.writeParams(leafCapacity, 0);
    const entries: GPUBindGroupEntry[] = [
      { binding: 0, resource: { buffer: this.params } }, { binding: 1, resource: leaves },
      { binding: 2, resource: candidates }, { binding: 3, resource: candidateCountAndDispatch },
      { binding: 4, resource: { buffer: this.buffer } }, { binding: 7, resource: { buffer: this.scratch } },
    ];
    const pass = encoder.beginComputePass({ label: "Seed global fine bricks from SurfaceLeaf candidates" });
    this.run(pass, "clearSeedState", Math.ceil(Math.max(2 * this.target.plan.hashCapacity,
      Math.ceil(leafCapacity / 32), 2) / 64), entries);
    this.run(pass, "markCandidateRows", Math.max(1, Math.ceil(candidateCapacity / 64)), entries);
    this.run(pass, "claimCandidateLeafOwners", leafBlocks, entries);
    this.run(pass, "classifyCandidateLeafBlocks", leafBlocks, entries);
    this.run(pass, "scanLeafSeedBlocks", 1, entries);
    this.run(pass, "emitCandidateLeafSeeds", leafBlocks, entries);
    this.run(pass, "finalizeLeafHash", Math.ceil(this.target.plan.maximumResidentBricks / 64), entries);
    pass.end();
    return { buffer: this.buffer, affineValues: true };
  }

  encodeFromAllInterfaceLeaves(encoder: GPUCommandEncoder, leaves: GPUBufferBinding,
    rowCount: GPUBufferBinding, powerBoundary?: FineLevelSetPowerBoundarySeedSource): FineLevelSetGPUSeedSource {
    const leafCapacity = Math.floor(this.bindingBytes(leaves) / 64);
    const endpointCapacity = powerBoundary ? Math.floor(this.bindingBytes(powerBoundary.queries) / 32) : 0;
    const { leafBlocks, endpointBlocks } = this.writeParams(leafCapacity, endpointCapacity);
    const entries: GPUBindGroupEntry[] = [
      { binding: 0, resource: { buffer: this.params } }, { binding: 1, resource: leaves },
      { binding: 3, resource: rowCount }, { binding: 4, resource: { buffer: this.buffer } },
      { binding: 7, resource: { buffer: this.scratch } },
    ];
    if (powerBoundary) entries.push(
      { binding: 5, resource: powerBoundary.queries },
      { binding: 6, resource: powerBoundary.control },
    );
    const pass = encoder.beginComputePass({ label: "Seed global fine bricks from every interface leaf" });
    this.run(pass, "clearSeedState", Math.ceil(Math.max(2 * this.target.plan.hashCapacity, 2) / 64), entries);
    this.run(pass, "claimAllLeafOwners", leafBlocks, entries);
    this.run(pass, "classifyAllLeafBlocks", leafBlocks, entries);
    this.run(pass, "scanLeafSeedBlocks", 1, entries);
    this.run(pass, "emitAllLeafSeeds", leafBlocks, entries);
    this.run(pass, "finalizeLeafHash", Math.ceil(this.target.plan.maximumResidentBricks / 64), entries);
    if (powerBoundary) {
      this.run(pass, "claimPowerEndpointOwners", endpointBlocks, entries);
      this.run(pass, "classifyPowerEndpointBlocks", endpointBlocks, entries);
      this.run(pass, "scanPowerEndpointBlocks", 1, entries);
      this.run(pass, "emitPowerEndpointSeeds", endpointBlocks, entries);
    }
    pass.end();
    return { buffer: this.buffer, affineValues: true };
  }

  destroy(): void { this.buffer.destroy(); this.params.destroy(); this.scratch.destroy(); }
}

export function unpackFineLevelSetGPUTopologyControl(words: ArrayLike<number>): FineLevelSetGPUTopologyControl {
  if (words.length < 8) throw new RangeError("Fine topology control requires eight words");
  return { flags: Number(words[0]) >>> 0, interfaceBricks: Number(words[1]) >>> 0,
    desiredBricks: Number(words[2]) >>> 0, activatedBricks: Number(words[3]) >>> 0,
    published: Number(words[4]) !== 0, rolledBack: Number(words[5]) !== 0,
    downstreamFinalizeReason: Number(words[7]) >>> 0,
    requiredDesiredBricks: (Number(words[0]) & FINE_LEVELSET_TOPOLOGY_ERROR.capacity) !== 0
      ? Number(words[6]) >>> 0 : Number(words[2]) >>> 0,
    requiredDesiredBricksExact: (Number(words[0]) & FINE_LEVELSET_TOPOLOGY_ERROR.capacity) === 0,
    dilationBrickRings: Number(words[0]) === 0 ? Number(words[6]) >>> 0 : 0,
    ...(words.length > 8 ? { interfaceSeedBricks: Number(words[8]) >>> 0 } : {}) };
}

/**
 * GPU discovery and atomic next-generation publication. The injected source
 * must define `fn sampleCoarseOctreePhi(position:vec3f)->f32`; it may use
 * textures/uniforms or additional bindings beginning at binding 8.
 */
export class WebGPUFineLevelSetTopology {
  readonly control: GPUBuffer;
  readonly allocatedBytes = FINE_LEVELSET_TOPOLOGY_ALLOCATED_BYTES;
  private readonly params: GPUBuffer;
  private readonly clearPipeline: GPUComputePipeline;
  private readonly discoverPipeline: GPUComputePipeline;
  private readonly externalSeedPipeline: GPUComputePipeline;
  private readonly beginDilationPipeline: GPUComputePipeline;
  private readonly directDilationPipeline: GPUComputePipeline;
  private readonly dilatePipeline: GPUComputePipeline;
  private readonly advanceDilationPipeline: GPUComputePipeline;
  private readonly snapshotPipeline: GPUComputePipeline;
  private readonly assignPipeline: GPUComputePipeline;
  private readonly initializePipeline: GPUComputePipeline;
  private readonly linkPipeline: GPUComputePipeline;
  private readonly finalizePipeline: GPUComputePipeline;
  private readonly publicationPipeline: GPUComputePipeline;
  private readonly rollbackPipeline: GPUComputePipeline;
  private readonly restorePipeline: GPUComputePipeline;
  private readonly emptySeeds: GPUBuffer;
  private readonly validVolumeControl: GPUBuffer;
  private readonly validTransportControl: GPUBuffer;

  constructor(
    private readonly device: GPUDevice,
    readonly current: WebGPUFineLevelSetBrickSource,
    readonly next: WebGPUFineLevelSetBrickSource,
    coarsePhiWGSL: string,
    _refreshFromCoarse = false,
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
    this.params = device.createBuffer({ label: "fine-levelset topology params", size: 96,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
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
    this.beginDilationPipeline = pipeline("Begin fine topology ring dilation", "beginDesiredDilation");
    this.directDilationPipeline = pipeline("Directly dilate fine topology seeds", "dilateDesiredFromSeeds");
    this.dilatePipeline = pipeline("Dilate fine topology ring", "dilateDesiredRing");
    this.advanceDilationPipeline = pipeline("Advance fine topology ring", "advanceDesiredDilation");
    this.snapshotPipeline = pipeline("Snapshot current fine topology payload", "snapshotCurrentPayload");
    this.assignPipeline = pipeline("Assign next fine topology pages", "assignDesiredPages");
    this.initializePipeline = pipeline("Initialize next fine topology samples", "initializeDesiredSamples");
    this.linkPipeline = pipeline("Link next fine topology neighbors", "linkDesiredNeighbors");
    this.finalizePipeline = pipeline("Finalize next fine topology generation", "finalizeDesiredGeneration");
    this.publicationPipeline = pipeline("Gate complete fine generation publication", "finalizeFinePublication");
    this.rollbackPipeline = device.createComputePipeline({ label: "Rollback failed fine topology generation", layout: "auto",
      compute: { module: shaderModule, entryPoint: "rollbackFailedGeneration" } });
    this.restorePipeline = pipeline("Restore failed fine topology payload", "restoreFailedPayload");
  }

  encode(encoder: GPUCommandEncoder, seedSource?: FineLevelSetGPUSeedSource,
    extraPublishEntries: readonly GPUBindGroupEntry[] = [], band?: FineLevelSetTopologyBand,
    deferPublication = false): void {
    const plan = this.current.plan;
    const bandPlan = planFineLevelSetTopologyBand(plan.brickResolution, band ?? {
      maximumBacktraceFineCells: 0,
      interpolationSupportFineCells: 0,
      redistanceBandFineCells: 0,
      safetyBrickRings: 1,
    });
    // Redistance consumes an immutable resident generation. Allocate the full
    // transport + interpolation + signed-distance support before either FMM
    // or JFA-CPT starts; no distance pass is permitted to mutate page tables.
    const dilationBrickRings = bandPlan.dilationBrickRings;
    const bytes = new ArrayBuffer(96); const u32 = new Uint32Array(bytes); const f32 = new Float32Array(bytes);
    u32.set(plan.brickDimensions, 0); u32[3] = plan.brickResolution;
    u32.set(plan.sampleDimensions, 4); u32[7] = plan.samplesPerBrick;
    f32.set(plan.domainOrigin, 8); f32[11] = plan.fineCellWidth;
    u32.set([plan.hashCapacity, plan.maximumHashProbes, plan.maximumResidentBricks,
      this.current.generation, this.next.generation, plan.fineFactor], 12);
    u32[18] = seedSource?.affineValues ? 1 : 0; u32[19] = dilationBrickRings;
    u32[20] = deferPublication ? 1 : 0;
    this.device.queue.writeBuffer(this.params, 0, bytes);
    this.device.queue.writeBuffer(this.control, 0, new Uint32Array(8));
    const resource = (buffer: GPUBuffer) => ({ buffer });
    const discoverEntries: GPUBindGroupEntry[] = [
      { binding: 0, resource: resource(this.params) }, { binding: 1, resource: resource(this.current.metadata) },
      { binding: 2, resource: resource(this.current.worklist) }, { binding: 3, resource: resource(this.current.flags) },
      { binding: 4, resource: resource(this.current.phi) }, { binding: 5, resource: resource(this.next.hash) },
      { binding: 6, resource: resource(this.next.worklist) }, { binding: 7, resource: resource(this.control) },
      { binding: 8, resource: resource(seedSource?.buffer ?? this.emptySeeds) },
      { binding: 14, resource: resource(this.current.worklist) },
    ];
    const publishEntries: GPUBindGroupEntry[] = [
      { binding: 0, resource: resource(this.params) }, { binding: 1, resource: resource(this.current.hash) },
      { binding: 2, resource: resource(this.next.hash) }, { binding: 3, resource: resource(this.next.metadata) },
      { binding: 4, resource: resource(this.next.worklist) }, { binding: 5, resource: resource(this.next.flags) },
      { binding: 6, resource: resource(this.next.phi) }, { binding: 7, resource: resource(this.control) },
      { binding: 8, resource: resource(seedSource?.buffer ?? this.emptySeeds) },
      { binding: 10, resource: resource(this.current.rollbackPhi) },
      { binding: 14, resource: resource(this.current.worklist) },
      ...extraPublishEntries,
    ];
    const snapshotEntries: GPUBindGroupEntry[] = [
      { binding: 0, resource: resource(this.params) }, { binding: 1, resource: resource(this.current.metadata) },
      { binding: 2, resource: resource(this.current.worklist) }, { binding: 3, resource: resource(this.current.flags) },
      { binding: 4, resource: resource(this.current.phi) }, { binding: 7, resource: resource(this.control) },
      { binding: 10, resource: resource(this.current.rollbackPhi) },
    ];
    // Dispatch boundaries already provide the storage-buffer ordering required
    // by discovery, dilation, assignment, and initialization. Keeping this
    // launch-bound chain in one compute pass avoids a driver pass transition
    // for every one-workgroup control stage on small domains.
    const pass = encoder.beginComputePass({ label: "Update global fine topology" });
    const run = (pipeline: GPUComputePipeline, entries: GPUBindGroupEntry[], _label: string, groups = 1,
      used?: readonly number[]) => {
      const selected = used ? entries.filter((entry) => used.includes(entry.binding)) : entries;
      const group = this.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: selected });
      pass.setPipeline(pipeline); pass.setBindGroup(0, group);
      const dispatch = planFineLevelSetDispatch2D(groups, this.device.limits.maxComputeWorkgroupsPerDimension);
      if (dispatch.workgroups > 0) pass.dispatchWorkgroups(dispatch.x, dispatch.y, dispatch.z);
    };
    const clearItems = Math.max(plan.hashCapacity * 2, 5 + plan.maximumResidentBricks, 8);
    run(this.clearPipeline, discoverEntries, "Clear global fine topology candidates", Math.ceil(clearItems / 64),
      [0, 5, 6, 7]);
    run(this.discoverPipeline, discoverEntries, "Discover global fine interface bricks",
      Math.ceil(plan.maximumResidentBricks / 64), [0, 1, 2, 3, 4, 5, 6, 7]);
    // Section 5 rebuilds a recurring fine grid from transported interface
    // cells. The discovery pass above preserves those cells; external seeds
    // add only explicitly tagged power-boundary endpoint support. Ordinary
    // affine SurfaceLeaf keys are a cold-start aid and must not turn the fine
    // narrow band into a dense replacement for the background octree.
    run(this.externalSeedPipeline, discoverEntries, "Insert external global fine seed bricks",
      Math.ceil(plan.maximumResidentBricks / 64), [0, 5, 6, 7, 8, 14]);
    run(this.beginDilationPipeline, discoverEntries, "Begin global fine topology dilation", 1, [0, 6, 7]);
    if (plan.logicalBrickCount <= FINE_LEVELSET_DIRECT_DILATION_MAXIMUM_BRICKS) {
      run(this.directDilationPipeline, discoverEntries, "Directly dilate global fine topology support",
        Math.ceil(plan.maximumResidentBricks / 64), [0, 5, 6, 7]);
    } else {
      const floodPasses = planFineLevelSetChebyshevFloodPasses(dilationBrickRings);
      for (let flood = 0; flood < floodPasses; flood += 1) {
        run(this.dilatePipeline, discoverEntries, `Flood global fine topology support ${flood + 1}`,
          Math.ceil(plan.maximumResidentBricks / 64), [0, 5, 6, 7]);
        run(this.advanceDilationPipeline, discoverEntries, `Advance global fine topology support ${flood + 1}`,
          1, [0, 6, 7]);
      }
    }
    run(this.snapshotPipeline, snapshotEntries, "Snapshot current global fine payload", plan.maximumResidentBricks,
      [0, 1, 3, 4, 10]);
    run(this.assignPipeline, publishEntries, "Assign global fine pages", Math.ceil(plan.maximumResidentBricks / 64),
      [0, 2, 3, 4, 7]);
    run(this.initializePipeline, publishEntries, "Initialize global fine samples", plan.maximumResidentBricks,
      [0, 1, 3, 5, 6, 7, 8, 9, 10, 14]);
    run(this.linkPipeline, publishEntries, "Link global fine neighbors", Math.ceil(plan.maximumResidentBricks / 64),
      [0, 2, 3, 7]);
    run(this.finalizePipeline, publishEntries, "Finalize global fine publication", 1, [0, 4, 7]);
    if (!deferPublication) {
      run(this.rollbackPipeline, [
        { binding: 0, resource: resource(this.params) }, { binding: 1, resource: resource(this.current.hash) },
        { binding: 2, resource: resource(this.current.metadata) }, { binding: 3, resource: resource(this.current.worklist) },
        { binding: 4, resource: resource(this.next.hash) }, { binding: 5, resource: resource(this.next.metadata) },
        { binding: 6, resource: resource(this.next.worklist) }, { binding: 7, resource: resource(this.control) },
      ], "Rollback failed global fine generation");
      run(this.restorePipeline, snapshotEntries, "Restore failed global fine payload", plan.maximumResidentBricks,
        [0, 1, 3, 4, 7, 10]);
    }
    pass.end();
  }

  /** Commit only after the complete transport/topology/redistance/volume chain
   * is valid. A failed target is replaced GPU-side by the previous valid
   * generation, retagged for the target slot, before any later consumer runs. */
  encodeFinalizePublication(encoder: GPUCommandEncoder, controls: {
    redistance: GPUBuffer; volume?: GPUBuffer; transport?: GPUBuffer;
  }): void {
    const resource = (buffer: GPUBuffer) => ({ buffer });
    const pass = encoder.beginComputePass({ label: "Finalize global fine publication" });
    pass.setPipeline(this.publicationPipeline);
    pass.setBindGroup(0, this.device.createBindGroup({ layout: this.publicationPipeline.getBindGroupLayout(0), entries: [
      { binding: 7, resource: resource(this.control) },
      { binding: 11, resource: resource(controls.redistance) },
      { binding: 12, resource: resource(controls.volume ?? this.validVolumeControl) },
      { binding: 13, resource: resource(controls.transport ?? this.validTransportControl) },
    ] }));
    pass.dispatchWorkgroups(1);
    pass.setPipeline(this.rollbackPipeline);
    pass.setBindGroup(0, this.device.createBindGroup({ layout: this.rollbackPipeline.getBindGroupLayout(0), entries: [
      { binding: 0, resource: resource(this.params) }, { binding: 1, resource: resource(this.current.hash) },
      { binding: 2, resource: resource(this.current.metadata) }, { binding: 3, resource: resource(this.current.worklist) },
      { binding: 4, resource: resource(this.next.hash) }, { binding: 5, resource: resource(this.next.metadata) },
      { binding: 6, resource: resource(this.next.worklist) }, { binding: 7, resource: resource(this.control) },
    ] }));
    pass.dispatchWorkgroups(1);
    pass.setPipeline(this.restorePipeline);
    pass.setBindGroup(0, this.device.createBindGroup({ layout: this.restorePipeline.getBindGroupLayout(0), entries: [
      { binding: 0, resource: resource(this.params) }, { binding: 1, resource: resource(this.current.metadata) },
      { binding: 3, resource: resource(this.next.flags) }, { binding: 4, resource: resource(this.next.phi) },
      { binding: 7, resource: resource(this.control) }, { binding: 10, resource: resource(this.current.rollbackPhi) },
    ] }));
    const dispatch = planFineLevelSetDispatch2D(this.current.plan.maximumResidentBricks,
      this.device.limits.maxComputeWorkgroupsPerDimension);
    pass.dispatchWorkgroups(dispatch.x, dispatch.y, dispatch.z); pass.end();
  }

  destroy(): void { this.control.destroy(); this.params.destroy(); this.emptySeeds.destroy();
    this.validVolumeControl.destroy(); this.validTransportControl.destroy(); }
}

export function makeFineLevelSetTopologyWGSL(coarsePhiWGSL: string): string {
  return /* wgsl */ `
const INVALID:u32=0xffffffffu;const VALID:u32=1u;const CAPACITY:u32=1u;const HASH:u32=2u;const NONFINITE:u32=4u;const MALFORMED:u32=8u;const RECURRING_SUPPORT:u32=0x80000000u;
struct Params { brickDimensions:vec3u,brickResolution:u32,sampleDimensions:vec3u,samplesPerBrick:u32,
 domainOrigin:vec3f,fineCellWidth:f32,hashCapacity:u32,maximumHashProbes:u32,pageCapacity:u32,currentGeneration:u32,nextGeneration:u32,fineFactor:u32,affineSeeds:u32,dilationBrickRings:u32,deferPublication:u32,p0:u32,p1:u32,p2:u32 }
@group(0) @binding(0) var<uniform> params:Params;
@group(0) @binding(1) var<storage,read_write> sourceA:array<u32>;
@group(0) @binding(2) var<storage,read_write> sourceB:array<u32>;
@group(0) @binding(3) var<storage,read_write> sourceC:array<u32>;
@group(0) @binding(4) var<storage,read_write> sourceD:array<u32>;
@group(0) @binding(5) var<storage,read_write> targetA:array<atomic<u32>>;
@group(0) @binding(6) var<storage,read_write> targetB:array<u32>;
@group(0) @binding(7) var<storage,read_write> control:array<atomic<u32>>;
@group(0) @binding(8) var<storage,read> externalSeeds:array<u32>;
@group(0) @binding(10) var<storage,read_write> payloadSnapshot:array<f32>;
@group(0) @binding(11) var<storage,read> redistanceControl:array<u32>;
@group(0) @binding(12) var<storage,read> volumeControl:array<u32>;
@group(0) @binding(13) var<storage,read> transportControl:array<u32>;
@group(0) @binding(14) var<storage,read> currentWorklist:array<u32>;
${coarsePhiWGSL}
${fineLevelSetLinearWorkgroupWGSL}
// The compact coarse sampler uses max-finite as an explicit invalid sentinel;
// strict comparison rejects it without asking Dawn to constant-fold a NaN.
fn finite(value:f32)->bool{return value==value&&abs(value)<3.402823e38;}
fn hashKey(key:u32)->u32{return ((key^(key>>16u))*0x9e3779b1u)&(params.hashCapacity-1u);}
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
fn desiredLookup(key:u32)->u32{let start=hashKey(key);for(var probe=0u;probe<32u;probe+=1u){if(probe>=params.maximumHashProbes){break;}
 let slot=(start+probe)&(params.hashCapacity-1u);let stored=atomicLoad(&targetA[slot*2u]);if(stored==key){return slot;}if(stored==INVALID){return INVALID;}}return INVALID;}
fn insertDesired(key:u32){let start=hashKey(key);for(var probe=0u;probe<32u;probe+=1u){if(probe>=params.maximumHashProbes){break;}let slot=(start+probe)&(params.hashCapacity-1u);
 loop{let result=atomicCompareExchangeWeak(&targetA[slot*2u],INVALID,key);if(result.old_value==key){return;}if(result.exchanged){atomicStore(&targetA[slot*2u+1u],INVALID);let count=atomicAdd(&control[2],1u);if(count>=params.pageCapacity){atomicOr(&control[0],CAPACITY);atomicMax(&control[6],count+1u);return;}targetB[5u+count]=key;return;}if(result.old_value!=INVALID){break;}}
 }atomicOr(&control[0],HASH);}
fn currentLookup(key:u32)->u32{let start=hashKey(key);for(var probe=0u;probe<32u;probe+=1u){if(probe>=params.maximumHashProbes){break;}
 let slot=(start+probe)&(params.hashCapacity-1u);let stored=sourceA[slot*2u];if(stored==key){return sourceA[slot*2u+1u];}if(stored==INVALID){return INVALID;}}return INVALID;}
fn externalSeedHash(key:u32)->u32{var value=key*0x9e3779b1u;value=(value^(value>>16u))*0x7feb352du;return value^(value>>15u);}
fn externalSeedTaggedValue(key:u32)->u32{let keyBase=4u+params.pageCapacity;let valueBase=keyBase+params.hashCapacity;let start=externalSeedHash(key)&(params.hashCapacity-1u);for(var probe=0u;probe<32u;probe+=1u){let slot=(start+probe)&(params.hashCapacity-1u);let stored=externalSeeds[keyBase+slot];if(stored==key){return externalSeeds[valueBase+slot];}if(stored==INVALID){break;}}return INVALID;}
fn currentFinePublished()->bool{return arrayLength(&currentWorklist)>=5u&&currentWorklist[1]==params.currentGeneration&&currentWorklist[3]==1u&&currentWorklist[4]==1u;}
fn currentFinePopulated()->bool{return currentFinePublished()&&currentWorklist[0]>0u;}
fn exactAnalyticSeedPhi(finestPoint:vec3f)->f32{if(arrayLength(&externalSeeds)<4u){return 3.402823e38;}let mode=externalSeeds[2u];let fill=bitcast<f32>(externalSeeds[3u]);if(mode==0u||!finite(fill)||fill<0.0||fill>1.0){return 3.402823e38;}let extent=vec3f(params.sampleDimensions)*params.fineCellWidth;let point=params.domainOrigin+finestPoint*(params.fineCellWidth*f32(params.fineFactor));if(mode==1u){return point.y-fill*extent.y;}let heightFraction=max(0.92,fill);let footprintFraction=sqrt(fill/max(heightFraction,1e-9));let half=0.5*vec3f(footprintFraction*extent.x,heightFraction*extent.y,footprintFraction*extent.z);let centre=params.domainOrigin+half;let q=abs(point-centre)-half;return length(max(q,vec3f(0.0)))+min(max(q.x,max(q.y,q.z)),0.0);}
fn externalSeedPhi(key:u32,finestPoint:vec3f)->f32{if(params.affineSeeds==0u||currentFinePopulated()){return 3.402823e38;}let analytic=exactAnalyticSeedPhi(finestPoint);if(finite(analytic)){return analytic;}let tagged=externalSeedTaggedValue(key);if(tagged==INVALID){return 3.402823e38;}let seed=tagged&0x7fffffffu;if(seed>=params.pageCapacity){return 3.402823e38;}let planeBase=4u+params.pageCapacity+2u*params.hashCapacity;let base=planeBase+seed*8u;let leafOrigin=vec3f(vec3u(externalSeeds[base],externalSeeds[base+1u],externalSeeds[base+2u]));let size=f32(externalSeeds[base+3u]);let centre=leafOrigin+vec3f(0.5*size);let value=bitcast<f32>(externalSeeds[base+4u]);let gradient=vec3f(bitcast<f32>(externalSeeds[base+5u]),bitcast<f32>(externalSeeds[base+6u]),bitcast<f32>(externalSeeds[base+7u]));return value+dot(gradient,finestPoint-centre);}
// The initial A/B source is a deliberately published empty generation. It is
// still a cold start: classify ordinary analytic/affine leaf keys by an actual
// zero crossing so only interface blocks precede the paper's one-ring
// allocation. Test the brick's geometric support bounds, not only its first
// and last sample centres. A face can lie exactly between two SPGrid pages;
// centre-only bounds then reject both pages and punch a vertical gap in the
// high-resolution interface band.
fn externalAffineInterfaceBrick(key:u32)->bool{if(params.affineSeeds==0u||currentFinePopulated()){return false;}let brick=unpackBrick(key);let first=vec3f(brick*params.brickResolution)/f32(params.fineFactor);let last=vec3f((brick+vec3u(1u))*params.brickResolution)/f32(params.fineFactor);var minimum=3.402823e38;var maximum=-3.402823e38;for(var corner=0u;corner<8u;corner+=1u){let point=vec3f(select(first.x,last.x,(corner&1u)!=0u),select(first.y,last.y,(corner&2u)!=0u),select(first.z,last.z,(corner&4u)!=0u));let value=externalSeedPhi(key,point);if(!finite(value)){return false;}minimum=min(minimum,value);maximum=max(maximum,value);}return minimum<=0.0&&maximum>=0.0;}
fn linearInvocation(wid:vec3u,nwg:vec3u,local:u32)->u32{return fineLinearWorkgroup(wid,nwg)*64u+local;}
@compute @workgroup_size(64) fn clearDesiredGeneration(@builtin(workgroup_id)wid:vec3u,@builtin(num_workgroups)nwg:vec3u,@builtin(local_invocation_index)local:u32){let item=linearInvocation(wid,nwg,local);if(item<params.hashCapacity*2u){atomicStore(&targetA[item],INVALID);}if(item<5u+params.pageCapacity){targetB[item]=0u;}if(item<9u&&item!=6u){atomicStore(&control[item],0u);}if(item==6u){atomicStore(&control[6],params.dilationBrickRings);}}
@compute @workgroup_size(64) fn discoverInterfaceBricks(@builtin(workgroup_id)wid:vec3u,@builtin(num_workgroups)nwg:vec3u,@builtin(local_invocation_index)local:u32){let work=linearInvocation(wid,nwg,local);let rawCount=sourceB[0];if(work==0u&&rawCount>params.pageCapacity){atomicOr(&control[0],MALFORMED);}let activeCount=min(rawCount,params.pageCapacity);if(work>=activeCount){return;}let id=sourceB[5u+work];if(id>=params.pageCapacity||sourceA[id*10u+2u]!=params.currentGeneration){atomicOr(&control[0],MALFORMED);return;}var interfaceBrick=false;for(var sample=0u;sample<params.samplesPerBrick&&!interfaceBrick;sample+=1u){let index=id*params.samplesPerBrick+sample;if((sourceC[index]&VALID)==0u){continue;}let center=bitcast<f32>(sourceD[index]);if(!finite(center)){atomicOr(&control[0],MALFORMED);continue;}for(var direction=0u;direction<6u;direction+=1u){let neighbor=currentNeighbor(id,sample,direction);if(neighbor==INVALID||(sourceC[neighbor]&VALID)==0u){continue;}let other=bitcast<f32>(sourceD[neighbor]);if(finite(other)&&(other<0.0)!=(center<0.0)){interfaceBrick=true;break;}}}if(interfaceBrick){atomicAdd(&control[1],1u);insertDesired(sourceA[id*10u+1u]);}}
@compute @workgroup_size(64) fn insertExternalSeeds(@builtin(workgroup_id)wid:vec3u,@builtin(num_workgroups)nwg:vec3u,@builtin(local_invocation_index)local:u32){let recurring=currentFinePopulated();let seed=linearInvocation(wid,nwg,local);if(arrayLength(&externalSeeds)<4u){return;}let rawCount=externalSeeds[0];let available=arrayLength(&externalSeeds)-4u;if(seed==0u&&(externalSeeds[1]!=0u||rawCount>params.pageCapacity||rawCount>available)){atomicOr(&control[0],CAPACITY);atomicMax(&control[6],max(rawCount,params.pageCapacity+1u));}let count=min(rawCount,min(params.pageCapacity,available));if(seed>=count){return;}let key=externalSeeds[4u+seed];let tagged=externalSeedTaggedValue(key);let endpoint=tagged!=INVALID&&(tagged&RECURRING_SUPPORT)!=0u;if(recurring&&!endpoint){return;}if(!recurring&&!endpoint&&!externalAffineInterfaceBrick(key)){return;}if(key<params.brickDimensions.x*params.brickDimensions.y*params.brickDimensions.z){insertDesired(key);}else{atomicOr(&control[0],MALFORMED);}}
// Preserve the exact pre-dilation prefix. It contains transported interface
// blocks and explicit power-boundary endpoints. Later entries are allocation
// support for backtrace/redistance and must never become Section 5 core rows.
@compute @workgroup_size(1) fn beginDesiredDilation(){targetB[0]=0u;targetB[1]=min(atomicLoad(&control[2]),params.pageCapacity);atomicStore(&control[8],targetB[1]);}
// A 16^3-or-smaller brick lattice is launch-bound. Expand the immutable seed
// prefix directly once instead of redispatching and rehashing its growing
// frontier. Stop early only when capacity covers the complete logical domain;
// capacity-limited configurations must continue and report overflow normally.
@compute @workgroup_size(64) fn dilateDesiredFromSeeds(@builtin(workgroup_id)wid:vec3u,@builtin(num_workgroups)nwg:vec3u,@builtin(local_invocation_index)local:u32){let work=linearInvocation(wid,nwg,local);let seedCount=targetB[1];if(atomicLoad(&control[0])!=0u||work>=seedCount){return;}let center=vec3i(unpackBrick(targetB[5u+work]));let radius=i32(params.dilationBrickRings);let logicalBricks=params.brickDimensions.x*params.brickDimensions.y*params.brickDimensions.z;let completeCapacity=params.pageCapacity>=logicalBricks;for(var dz=-radius;dz<=radius;dz+=1){for(var dy=-radius;dy<=radius;dy+=1){for(var dx=-radius;dx<=radius;dx+=1){if(completeCapacity&&atomicLoad(&control[2])>=logicalBricks){return;}let q=center+vec3i(dx,dy,dz);if(all(q>=vec3i(0))&&all(q<vec3i(params.brickDimensions))){insertDesired(packBrick(vec3u(q)));}}}}}
// Every dispatch expands the already-complete Chebyshev ball by min(r+1,
// remaining) cells. The exact radii are 0,1,3,7... clamped to the requested
// support, hence O(log R) dispatches and no conservative over-dilation.
@compute @workgroup_size(64) fn dilateDesiredRing(@builtin(workgroup_id)wid:vec3u,@builtin(num_workgroups)nwg:vec3u,@builtin(local_invocation_index)local:u32){let work=linearInvocation(wid,nwg,local);let radius=targetB[0];let layerEnd=targetB[1];if(atomicLoad(&control[0])!=0u||radius>=params.dilationBrickRings||work>=layerEnd){return;}let expansion=min(radius+1u,params.dilationBrickRings-radius);let center=vec3i(unpackBrick(targetB[5u+work]));let step=i32(expansion);for(var dz=-step;dz<=step;dz+=1){for(var dy=-step;dy<=step;dy+=1){for(var dx=-step;dx<=step;dx+=1){let q=center+vec3i(dx,dy,dz);if(all(q>=vec3i(0))&&all(q<vec3i(params.brickDimensions))){insertDesired(packBrick(vec3u(q)));}}}}}
@compute @workgroup_size(1) fn advanceDesiredDilation(){let radius=targetB[0];let expansion=min(radius+1u,params.dilationBrickRings-radius);targetB[0]=radius+expansion;targetB[1]=min(atomicLoad(&control[2]),params.pageCapacity);}
fn nextSlot(key:u32)->u32{let start=hashKey(key);for(var probe=0u;probe<32u;probe+=1u){if(probe>=params.maximumHashProbes){break;}
 let slot=(start+probe)&(params.hashCapacity-1u);let stored=sourceB[slot*2u];if(stored==key){return slot;}if(stored==INVALID){return INVALID;}}return INVALID;}
fn updateDesiredId(key:u32,id:u32){let slot=nextSlot(key);if(slot==INVALID){atomicOr(&control[0],HASH);return;}sourceB[slot*2u+1u]=id;}
fn targetLookup(key:u32)->u32{let slot=nextSlot(key);if(slot==INVALID){return INVALID;}return sourceB[slot*2u+1u];}
@compute @workgroup_size(64) fn snapshotCurrentPayload(@builtin(workgroup_id) wid:vec3u,@builtin(num_workgroups) nwg:vec3u,@builtin(local_invocation_index) local:u32){let index=fineLinearWorkgroup(wid,nwg)*64u+local;let capacity=params.pageCapacity*params.samplesPerBrick;if(index>=capacity){return;}let id=index/params.samplesPerBrick;let valid=sourceA[id*10u+2u]==params.currentGeneration&&(sourceC[index]&VALID)!=0u;let value=bitcast<f32>(sourceD[index]);payloadSnapshot[index]=select(3.402823e38,value,valid&&finite(value));}
@compute @workgroup_size(64) fn assignDesiredPages(@builtin(workgroup_id) wid:vec3u,@builtin(num_workgroups) nwg:vec3u,@builtin(local_invocation_index) local:u32){let id=fineLinearWorkgroup(wid,nwg)*64u+local;if(id>=params.pageCapacity){return;}let base=id*10u;for(var word=0u;word<10u;word+=1u){sourceC[base+word]=INVALID;}let count=atomicLoad(&control[2]);if(id>=count||atomicLoad(&control[0])!=0u){return;}let key=sourceD[5u+id];let slot=nextSlot(key);if(slot==INVALID){atomicOr(&control[0],HASH);return;}sourceB[slot*2u+1u]=id;sourceC[base]=id;sourceC[base+1u]=key;sourceC[base+2u]=params.nextGeneration;sourceC[base+3u]=1u;sourceD[5u+id]=id;}
@compute @workgroup_size(64) fn initializeDesiredSamples(@builtin(workgroup_id) wid:vec3u,@builtin(num_workgroups) nwg:vec3u,@builtin(local_invocation_index) local:u32){let id=fineLinearWorkgroup(wid,nwg);if(id>=atomicLoad(&control[2])||id>=params.pageCapacity||atomicLoad(&control[0])!=0u){return;}let key=sourceC[id*10u+1u];let old=currentLookup(key);let index=id*params.samplesPerBrick+local;if(local>=params.samplesPerBrick||index>=arrayLength(&targetA)||index>=arrayLength(&targetB)){return;}if(old!=INVALID&&old<params.pageCapacity){let saved=payloadSnapshot[old*params.samplesPerBrick+local];if(finite(saved)){atomicStore(&targetA[index],VALID|select(0u,16u,saved<0.0));targetB[index]=bitcast<u32>(saved);return;}}
 let brick=unpackBrick(key);let coord=localCoord(local);let q=brick*params.brickResolution+coord;if(any(q>=params.sampleDimensions)){atomicStore(&targetA[index],0u);targetB[index]=0u;return;}let position=params.domainOrigin+(vec3f(q)+vec3f(0.5))*params.fineCellWidth;var value=sampleCoarseOctreePhi(position);let seeded=externalSeedPhi(key,(vec3f(q)+vec3f(0.5))/f32(params.fineFactor));if(finite(seeded)){value=seeded;}if(!finite(value)){atomicOr(&control[0],NONFINITE);atomicMin(&control[6],id);atomicMin(&control[7],key);return;}atomicStore(&targetA[index],VALID|select(0u,16u,value<0.0));targetB[index]=bitcast<u32>(value);}
@compute @workgroup_size(64) fn linkDesiredNeighbors(@builtin(workgroup_id) wid:vec3u,@builtin(num_workgroups) nwg:vec3u,@builtin(local_invocation_index) local:u32){let id=fineLinearWorkgroup(wid,nwg)*64u+local;if(id>=atomicLoad(&control[2])||id>=params.pageCapacity||atomicLoad(&control[0])!=0u){return;}let coord=unpackBrick(sourceC[id*10u+1u]);for(var direction=0u;direction<6u;direction+=1u){var neighbor=INVALID;if(direction==0u&&coord.x>0u){neighbor=targetLookup(packBrick(coord-vec3u(1,0,0)));}else if(direction==1u&&coord.x+1u<params.brickDimensions.x){neighbor=targetLookup(packBrick(coord+vec3u(1,0,0)));}else if(direction==2u&&coord.y>0u){neighbor=targetLookup(packBrick(coord-vec3u(0,1,0)));}else if(direction==3u&&coord.y+1u<params.brickDimensions.y){neighbor=targetLookup(packBrick(coord+vec3u(0,1,0)));}else if(direction==4u&&coord.z>0u){neighbor=targetLookup(packBrick(coord-vec3u(0,0,1)));}else if(direction==5u&&coord.z+1u<params.brickDimensions.z){neighbor=targetLookup(packBrick(coord+vec3u(0,0,1)));}sourceC[id*10u+4u+direction]=neighbor;}}
@compute @workgroup_size(1) fn finalizeDesiredGeneration(){if(atomicLoad(&control[0])!=0u){return;}let count=atomicLoad(&control[2]);sourceD[0]=count;sourceD[1]=params.nextGeneration;sourceD[2]=(count+63u)/64u;sourceD[3]=1u;sourceD[4]=1u;atomicStore(&control[3],count);atomicStore(&control[4],select(1u,0u,params.deferPublication!=0u));}
@compute @workgroup_size(1) fn finalizeFinePublication(){let topologyValid=atomicLoad(&control[0])==0u;let redistanceValid=arrayLength(&redistanceControl)>=4u&&redistanceControl[0]==0u&&redistanceControl[2]>0u&&redistanceControl[3]!=0u;let volumeValid=arrayLength(&volumeControl)>0u&&volumeControl[0]==0x80000000u;let transportValid=arrayLength(&transportControl)>=4u&&transportControl[3]!=0u;if(topologyValid&&redistanceValid&&volumeValid&&transportValid){atomicStore(&control[4],1u);return;}atomicOr(&control[0],16u);atomicStore(&control[7],select(0u,1u,!topologyValid)|select(0u,2u,!redistanceValid)|select(0u,4u,!volumeValid)|select(0u,8u,!transportValid));}
@compute @workgroup_size(1) fn rollbackFailedGeneration(){if(atomicLoad(&control[0])==0u){return;}
 for(var slot=0u;slot<params.hashCapacity;slot+=1u){sourceD[slot*2u]=sourceA[slot*2u];sourceD[slot*2u+1u]=sourceA[slot*2u+1u];}
 for(var word=0u;word<params.pageCapacity*10u;word+=1u){atomicStore(&targetA[word],INVALID);}
 let currentPublished=sourceC[1]==params.currentGeneration&&sourceC[3]==1u&&sourceC[4]==1u;if(!currentPublished){targetB[0]=0u;targetB[1]=params.nextGeneration;targetB[2]=0u;targetB[3]=0u;targetB[4]=0u;atomicStore(&control[4],0u);atomicStore(&control[5],1u);return;}
 let count=min(sourceC[0],params.pageCapacity);for(var work=0u;work<count;work+=1u){let id=sourceC[5u+work];if(id>=params.pageCapacity){continue;}let base=id*10u;for(var word=0u;word<10u;word+=1u){atomicStore(&targetA[base+word],sourceB[base+word]);}atomicStore(&targetA[base+2u],params.nextGeneration);targetB[5u+work]=id;}
 targetB[0]=count;targetB[1]=params.nextGeneration;targetB[2]=(count+63u)/64u;targetB[3]=1u;targetB[4]=1u;atomicStore(&control[4],1u);atomicStore(&control[5],1u);
}
@compute @workgroup_size(64) fn restoreFailedPayload(@builtin(workgroup_id) wid:vec3u,@builtin(num_workgroups) nwg:vec3u,@builtin(local_invocation_index) local:u32){if(atomicLoad(&control[0])==0u){return;}let index=fineLinearWorkgroup(wid,nwg)*64u+local;let capacity=params.pageCapacity*params.samplesPerBrick;if(index>=capacity){return;}let id=index/params.samplesPerBrick;if(sourceA[id*10u+2u]!=params.currentGeneration){return;}let value=payloadSnapshot[index];if(!finite(value)){return;}sourceC[index]=VALID|select(0u,16u,value<0.0);sourceD[index]=bitcast<u32>(value);}
`;
}

export const fineLevelSetLeafSeedWGSL = /* wgsl */ `
const CORE:u32=2u;const RECURRING_SUPPORT:u32=0x80000000u;const INVALID:u32=0xffffffffu;const ENDPOINT_OWNER:u32=0x40000000u;
struct Params { header:vec4u,tail:vec4u,fineDomain:vec4f,scan:vec4u }
struct SurfaceLeaf { originX:u32,originY:u32,originZ:u32,size:u32,flags:u32,pad0:u32,pad1:u32,pad2:u32,phiGradient:vec4f,motion:vec4f }
struct Candidate { row:u32,flags:u32 }
struct BoundaryPhiQuery { liquidCenter:vec4f,airCenter:vec4f }
@group(0) @binding(0) var<uniform> params:Params;
@group(0) @binding(1) var<storage,read> leaves:array<SurfaceLeaf>;
@group(0) @binding(2) var<storage,read> candidates:array<Candidate>;
@group(0) @binding(3) var<storage,read> candidateControl:array<u32>;
@group(0) @binding(4) var<storage,read_write> seeds:array<atomic<u32>>;
@group(0) @binding(5) var<storage,read> boundaryQueries:array<BoundaryPhiQuery>;
@group(0) @binding(6) var<storage,read> powerFaceControl:array<u32>;
@group(0) @binding(7) var<storage,read_write> scratch:array<atomic<u32>>;
fn leafOrigin(leaf:SurfaceLeaf)->vec3u{return vec3u(leaf.originX,leaf.originY,leaf.originZ);}
fn brickDimensions()->vec3u{return vec3u(params.header.z,params.header.w,params.tail.x);}
fn packBrick(coord:vec3u)->u32{let dims=brickDimensions();return coord.x+dims.x*(coord.y+dims.y*coord.z);}
fn seedHash(key:u32)->u32{var value=key*0x9e3779b1u;value=(value^(value>>16u))*0x7feb352du;return value^(value>>15u);}
fn seedKeyBase()->u32{return 4u+params.tail.y;}fn seedValueBase()->u32{return seedKeyBase()+params.tail.w;}fn seedPlaneBase()->u32{return seedValueBase()+params.tail.w;}
// scanBlocks stores both its total at blockCount and its base at
// blockCount+1. Keep the eligibility bitset after that two-word footer so the
// emission dispatch observes the same classified rows as the claim dispatch.
fn scanBlockBase()->u32{return max(params.scan.y,params.scan.w)+2u;}fn eligibleWord(row:u32)->u32{return scanBlockBase()+row/32u;}
fn leafCount(leaf:SurfaceLeaf)->u32{let origin=leafOrigin(leaf);let first=origin*params.header.x/params.header.y;var last=(origin+vec3u(max(1u,leaf.size)))*params.header.x-vec3u(1);last=min(last/params.header.y,brickDimensions()-vec3u(1));let extent=last-first+vec3u(1);return extent.x*extent.y*extent.z;}
fn leafKey(leaf:SurfaceLeaf,local:u32)->u32{let origin=leafOrigin(leaf);let first=origin*params.header.x/params.header.y;var last=(origin+vec3u(max(1u,leaf.size)))*params.header.x-vec3u(1);last=min(last/params.header.y,brickDimensions()-vec3u(1));let extent=last-first+vec3u(1);let x=local%extent.x;let yz=local/extent.x;let y=yz%extent.y;let z=yz/extent.y;return packBrick(first+vec3u(x,y,z));}
fn hashSlot(key:u32)->u32{let start=seedHash(key)&(params.tail.w-1u);for(var probe=0u;probe<32u;probe+=1u){let slot=(start+probe)&(params.tail.w-1u);let stored=atomicLoad(&seeds[seedKeyBase()+slot]);if(stored==key){return slot;}if(stored==INVALID){return INVALID;}}return INVALID;}
fn claimLeaf(key:u32,row:u32){let start=seedHash(key)&(params.tail.w-1u);var probe=0u;loop{if(probe>=32u){atomicMax(&seeds[1],2u);return;}let slot=(start+probe)&(params.tail.w-1u);let result=atomicCompareExchangeWeak(&seeds[seedKeyBase()+slot],INVALID,key);if(result.exchanged||result.old_value==key){atomicMin(&seeds[seedValueBase()+slot],row);return;}if(result.old_value!=INVALID){probe+=1u;}}}
fn writeLeaf(index:u32,key:u32,leaf:SurfaceLeaf){if(index>=params.tail.y){return;}atomicStore(&seeds[4u+index],key);let base=seedPlaneBase()+index*8u;atomicStore(&seeds[base],leaf.originX);atomicStore(&seeds[base+1u],leaf.originY);atomicStore(&seeds[base+2u],leaf.originZ);atomicStore(&seeds[base+3u],leaf.size);atomicStore(&seeds[base+4u],bitcast<u32>(leaf.phiGradient.x));atomicStore(&seeds[base+5u],bitcast<u32>(leaf.phiGradient.y));atomicStore(&seeds[base+6u],bitcast<u32>(leaf.phiGradient.z));atomicStore(&seeds[base+7u],bitcast<u32>(leaf.phiGradient.w));}
@compute @workgroup_size(64) fn clearSeedState(@builtin(global_invocation_id)gid:vec3u){let item=gid.x;if(item==0u){atomicStore(&seeds[0],0u);atomicStore(&seeds[1],0u);}if(item<params.tail.w){atomicStore(&seeds[seedKeyBase()+item],INVALID);atomicStore(&seeds[seedValueBase()+item],INVALID);}let words=(params.scan.x+31u)/32u;if(item<words){atomicStore(&scratch[scanBlockBase()+item],0u);}}
@compute @workgroup_size(64) fn markCandidateRows(@builtin(global_invocation_id)gid:vec3u){let index=gid.x;let count=min(candidateControl[0],arrayLength(&candidates));if(index>=count){return;}let item=candidates[index];if((item.flags&CORE)!=0u&&item.row<params.scan.x){atomicOr(&scratch[eligibleWord(item.row)],1u<<(item.row&31u));}}
fn rowEligible(row:u32,candidatesOnly:bool)->bool{if(row>=params.scan.x||row>=arrayLength(&leaves)){return false;}if(candidatesOnly){return (atomicLoad(&scratch[eligibleWord(row)])&(1u<<(row&31u)))!=0u;}if(row>=candidateControl[0]){return false;}return (leaves[row].flags&CORE)!=0u;}
fn ownsLeafKey(row:u32,key:u32)->bool{let slot=hashSlot(key);return slot!=INVALID&&atomicLoad(&seeds[seedValueBase()+slot])==row;}
fn claimLeafRow(row:u32,candidatesOnly:bool){if(!rowEligible(row,candidatesOnly)){return;}let leaf=leaves[row];let count=leafCount(leaf);for(var local=0u;local<count;local+=1u){claimLeaf(leafKey(leaf,local),row);}}
@compute @workgroup_size(64) fn claimCandidateLeafOwners(@builtin(global_invocation_id)gid:vec3u){claimLeafRow(gid.x,true);}
@compute @workgroup_size(64) fn claimAllLeafOwners(@builtin(global_invocation_id)gid:vec3u){claimLeafRow(gid.x,false);}
var<workgroup> counts:array<u32,64>;var<workgroup> scanValues:array<u32,256>;
fn ownedLeafCount(row:u32,candidatesOnly:bool)->u32{if(!rowEligible(row,candidatesOnly)){return 0u;}let leaf=leaves[row];let candidates=leafCount(leaf);var count=0u;for(var local=0u;local<candidates;local+=1u){count+=select(0u,1u,ownsLeafKey(row,leafKey(leaf,local)));}return count;}
fn classifyLeafBlock(wid:u32,lid:u32,candidatesOnly:bool){let row=wid*64u+lid;counts[lid]=ownedLeafCount(row,candidatesOnly);workgroupBarrier();if(lid==0u){var total=0u;for(var i=0u;i<64u;i+=1u){total+=counts[i];}atomicStore(&scratch[wid],total);}}
@compute @workgroup_size(64) fn classifyCandidateLeafBlocks(@builtin(workgroup_id)wid:vec3u,@builtin(local_invocation_index)lid:u32){classifyLeafBlock(wid.x,lid,true);}
@compute @workgroup_size(64) fn classifyAllLeafBlocks(@builtin(workgroup_id)wid:vec3u,@builtin(local_invocation_index)lid:u32){classifyLeafBlock(wid.x,lid,false);}
fn scanBlocks(lid:u32,blockCount:u32,base:u32){let chunk=(blockCount+255u)/256u;let first=lid*chunk;let last=min(first+chunk,blockCount);var subtotal=0u;for(var block=first;block<last;block+=1u){subtotal+=atomicLoad(&scratch[block]);}scanValues[lid]=subtotal;workgroupBarrier();for(var offset=1u;offset<256u;offset<<=1u){var add=0u;if(lid>=offset){add=scanValues[lid-offset];}workgroupBarrier();scanValues[lid]+=add;workgroupBarrier();}var cursor=base+scanValues[lid]-subtotal;for(var block=first;block<last;block+=1u){let value=atomicLoad(&scratch[block]);atomicStore(&scratch[block],cursor-base);cursor+=value;}if(last==blockCount){atomicStore(&scratch[blockCount],cursor-base);atomicStore(&scratch[blockCount+1u],base);atomicStore(&seeds[0],min(cursor,params.tail.y));if(cursor>params.tail.y){atomicMax(&seeds[1],1u);}}}
@compute @workgroup_size(256) fn scanLeafSeedBlocks(@builtin(local_invocation_index)lid:u32){scanBlocks(lid,params.scan.y,0u);}
fn emitLeafBlock(wid:u32,lid:u32,candidatesOnly:bool){let row=wid*64u+lid;let count=ownedLeafCount(row,candidatesOnly);counts[lid]=count;workgroupBarrier();var localOffset=0u;for(var i=0u;i<lid;i+=1u){localOffset+=counts[i];}if(count==0u){return;}let leaf=leaves[row];let candidates=leafCount(leaf);var output=atomicLoad(&scratch[wid])+localOffset;for(var local=0u;local<candidates;local+=1u){let key=leafKey(leaf,local);if(ownsLeafKey(row,key)){writeLeaf(output,key,leaf);output+=1u;}}}
@compute @workgroup_size(64) fn emitCandidateLeafSeeds(@builtin(workgroup_id)wid:vec3u,@builtin(local_invocation_index)lid:u32){emitLeafBlock(wid.x,lid,true);}
@compute @workgroup_size(64) fn emitAllLeafSeeds(@builtin(workgroup_id)wid:vec3u,@builtin(local_invocation_index)lid:u32){emitLeafBlock(wid.x,lid,false);}
@compute @workgroup_size(64) fn finalizeLeafHash(@builtin(global_invocation_id)gid:vec3u){let index=gid.x;if(index>=atomicLoad(&seeds[0])){return;}let slot=hashSlot(atomicLoad(&seeds[4u+index]));if(slot==INVALID){atomicMax(&seeds[1],2u);}else{atomicStore(&seeds[seedValueBase()+slot],index);}}
fn endpointValid()->bool{return arrayLength(&powerFaceControl)>=9u&&powerFaceControl[3]==0u&&powerFaceControl[8]==0x80000000u;}
fn endpointKey(face:u32,ordinal:u32)->vec2u{if(!endpointValid()||face>=min(powerFaceControl[1],min(params.scan.z,arrayLength(&boundaryQueries)))){return vec2u(0u);}let query=boundaryQueries[face];if(query.liquidCenter.w==0.0||query.airCenter.w==0.0||!(params.fineDomain.w>0.0)){return vec2u(0u);}let position=select(query.liquidCenter.xyz,query.airCenter.xyz,ordinal>=8u);if(any(position<params.fineDomain.xyz)){return vec2u(0u);}let local=ordinal&7u;let lattice=(position-params.fineDomain.xyz)/params.fineDomain.w-vec3f(0.5);let q=vec3i(floor(lattice))+vec3i(i32(local&1u),i32((local>>1u)&1u),i32((local>>2u)&1u));let sampleDims=brickDimensions()*params.header.y;if(any(q<vec3i(0))||any(q>=vec3i(sampleDims))){return vec2u(0u);}return vec2u(1u,packBrick(vec3u(q)/params.header.y));}
fn claimEndpoint(key:u32,token:u32){let start=seedHash(key)&(params.tail.w-1u);var probe=0u;loop{if(probe>=32u){atomicMax(&seeds[1],2u);return;}let slot=(start+probe)&(params.tail.w-1u);let result=atomicCompareExchangeWeak(&seeds[seedKeyBase()+slot],INVALID,key);if(result.exchanged){atomicMin(&seeds[seedValueBase()+slot],ENDPOINT_OWNER|token);return;}if(result.old_value==key){let value=atomicLoad(&seeds[seedValueBase()+slot]);if(value<ENDPOINT_OWNER||(value&RECURRING_SUPPORT)!=0u){atomicOr(&seeds[seedValueBase()+slot],RECURRING_SUPPORT);}else{atomicMin(&seeds[seedValueBase()+slot],ENDPOINT_OWNER|token);}return;}if(result.old_value!=INVALID){probe+=1u;}}}
@compute @workgroup_size(64) fn claimPowerEndpointOwners(@builtin(global_invocation_id)gid:vec3u){let face=gid.x;if(atomicLoad(&seeds[1])!=0u){return;}for(var local=0u;local<16u;local+=1u){let item=endpointKey(face,local);if(item.x!=0u){claimEndpoint(item.y,face*16u+local);}}}
fn endpointOwnerCount(face:u32)->u32{var count=0u;for(var local=0u;local<16u;local+=1u){let item=endpointKey(face,local);if(item.x==0u){continue;}let slot=hashSlot(item.y);if(slot!=INVALID&&atomicLoad(&seeds[seedValueBase()+slot])==(ENDPOINT_OWNER|(face*16u+local))){count+=1u;}}return count;}
@compute @workgroup_size(64) fn classifyPowerEndpointBlocks(@builtin(workgroup_id)wid:vec3u,@builtin(local_invocation_index)lid:u32){let face=wid.x*64u+lid;counts[lid]=select(0u,endpointOwnerCount(face),atomicLoad(&seeds[1])==0u);workgroupBarrier();if(lid==0u){var total=0u;for(var i=0u;i<64u;i+=1u){total+=counts[i];}atomicStore(&scratch[wid.x],total);}}
@compute @workgroup_size(256) fn scanPowerEndpointBlocks(@builtin(local_invocation_index)lid:u32){scanBlocks(lid,params.scan.w,atomicLoad(&seeds[0]));}
@compute @workgroup_size(64) fn emitPowerEndpointSeeds(@builtin(workgroup_id)wid:vec3u,@builtin(local_invocation_index)lid:u32){let face=wid.x*64u+lid;let count=endpointOwnerCount(face);counts[lid]=count;workgroupBarrier();var localOffset=0u;for(var i=0u;i<lid;i+=1u){localOffset+=counts[i];}var cursor=atomicLoad(&scratch[params.scan.w+1u])+atomicLoad(&scratch[wid.x])+localOffset;for(var local=0u;local<16u;local+=1u){let item=endpointKey(face,local);if(item.x==0u){continue;}let slot=hashSlot(item.y);if(slot==INVALID||atomicLoad(&seeds[seedValueBase()+slot])!=(ENDPOINT_OWNER|(face*16u+local))){continue;}if(cursor<params.tail.y){atomicStore(&seeds[4u+cursor],item.y);atomicStore(&seeds[seedValueBase()+slot],cursor|RECURRING_SUPPORT);let base=seedPlaneBase()+cursor*8u;atomicStore(&seeds[base],0u);atomicStore(&seeds[base+1u],0u);atomicStore(&seeds[base+2u],0u);atomicStore(&seeds[base+3u],1u);atomicStore(&seeds[base+4u],bitcast<u32>(3.402823e38));for(var word=5u;word<8u;word+=1u){atomicStore(&seeds[base+word],0u);}}cursor+=1u;}}
`;
