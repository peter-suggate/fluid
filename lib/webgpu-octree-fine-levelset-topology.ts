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

/** Converts the complete physical support formula in plan Section 18.6 to block rings. */
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
  const requiredFineCells = band.maximumBacktraceFineCells
    + band.interpolationSupportFineCells + band.redistanceBandFineCells;
  return { ...band, safetyBrickRings, requiredFineCells,
    dilationBrickRings: Math.ceil(requiredFineCells / brickResolution) + safetyBrickRings };
}

export interface FineLevelSetGPUSeedSource { readonly buffer: GPUBuffer; readonly affineValues?: boolean; }

export function fineLevelSetLeafSeedAllocatedBytes(maximumResidentBricks: number, hashCapacity: number): number {
  if (!Number.isSafeInteger(maximumResidentBricks) || maximumResidentBricks < 1) {
    throw new RangeError("Fine seed resident capacity must be positive");
  }
  if (!Number.isSafeInteger(hashCapacity) || hashCapacity < 1 || (hashCapacity & (hashCapacity - 1)) !== 0) {
    throw new RangeError("Fine seed hash capacity must be a positive power of two");
  }
  return (2 + 7 * maximumResidentBricks + 2 * hashCapacity) * 4 + 32;
}

export const FINE_LEVELSET_TOPOLOGY_ALLOCATED_BYTES = 32 + 96 + 8 + 64 + 32;

/** GPU-only bridge from existing compact SurfaceLeaf/core candidates to global brick keys. */
export class WebGPUFineLevelSetLeafSeeds {
  readonly buffer: GPUBuffer;
  readonly allocatedBytes: number;
  private readonly params: GPUBuffer;
  private readonly pipeline: GPUComputePipeline;
  private readonly allLeavesPipeline: GPUComputePipeline;

  constructor(private readonly device: GPUDevice, readonly target: WebGPUFineLevelSetBrickSource) {
    this.allocatedBytes = fineLevelSetLeafSeedAllocatedBytes(
      target.plan.maximumResidentBricks, target.plan.hashCapacity,
    );
    this.buffer = device.createBuffer({ label: "global fine brick seed keys",
      size: (2 + 7 * target.plan.maximumResidentBricks + 2 * target.plan.hashCapacity) * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    this.params = device.createBuffer({ label: "global fine seed parameters", size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const shaderModule = device.createShaderModule({ label: "SurfaceLeaf to global fine seeds", code: fineLevelSetLeafSeedWGSL });
    this.pipeline = device.createComputePipeline({ label: "Emit global fine seed keys", layout: "auto",
      compute: { module: shaderModule, entryPoint: "emitSeeds" } });
    this.allLeavesPipeline = device.createComputePipeline({ label: "Emit global fine seeds from all interface leaves", layout: "auto",
      compute: { module: shaderModule, entryPoint: "emitAllInterfaceSeeds" } });
  }

  encode(encoder: GPUCommandEncoder, leaves: GPUBufferBinding, candidates: GPUBufferBinding,
    candidateCountAndDispatch: GPUBufferBinding): FineLevelSetGPUSeedSource {
    const plan = this.target.plan;
    const params = new Uint32Array([plan.fineFactor, plan.brickResolution, ...plan.brickDimensions,
      plan.maximumResidentBricks, plan.logicalBrickCount, plan.hashCapacity]);
    this.device.queue.writeBuffer(this.params, 0, params);
    this.device.queue.writeBuffer(this.buffer, 0, new Uint32Array(2));
    const group = this.device.createBindGroup({ layout: this.pipeline.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: this.params } }, { binding: 1, resource: leaves },
      { binding: 2, resource: candidates }, { binding: 3, resource: candidateCountAndDispatch },
      { binding: 4, resource: { buffer: this.buffer } },
    ] });
    const pass = encoder.beginComputePass({ label: "Seed global fine bricks from SurfaceLeaf candidates" });
    pass.setPipeline(this.pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(1); pass.end();
    return { buffer: this.buffer, affineValues: true };
  }

  encodeFromAllInterfaceLeaves(encoder: GPUCommandEncoder, leaves: GPUBufferBinding,
    rowCount: GPUBufferBinding): FineLevelSetGPUSeedSource {
    const plan = this.target.plan;
    const params = new Uint32Array([plan.fineFactor, plan.brickResolution, ...plan.brickDimensions,
      plan.maximumResidentBricks, plan.logicalBrickCount, plan.hashCapacity]);
    this.device.queue.writeBuffer(this.params, 0, params); this.device.queue.writeBuffer(this.buffer, 0, new Uint32Array(2));
    const group = this.device.createBindGroup({ layout: this.allLeavesPipeline.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: this.params } }, { binding: 1, resource: leaves },
      { binding: 3, resource: rowCount }, { binding: 4, resource: { buffer: this.buffer } },
    ] });
    const pass = encoder.beginComputePass({ label: "Seed global fine bricks from every interface leaf" });
    pass.setPipeline(this.allLeavesPipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(1); pass.end();
    return { buffer: this.buffer, affineValues: true };
  }

  destroy(): void { this.buffer.destroy(); this.params.destroy(); }
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
    dilationBrickRings: Number(words[0]) === 0 ? Number(words[6]) >>> 0 : 0 };
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
    this.control = device.createBuffer({ label: "fine-levelset topology control", size: 32,
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
    // Paper Section 5 starts the new SPGrid generation with interface blocks
    // plus exactly one block 1-ring. The distance march, not topology
    // pre-dilation, activates every additional page required by the physical
    // band. Retired pages are absent because the target hash/metadata/worklist
    // are rebuilt from this initial set before marching.
    const initialDilationBrickRings = bandPlan.safetyBrickRings;
    const bytes = new ArrayBuffer(96); const u32 = new Uint32Array(bytes); const f32 = new Float32Array(bytes);
    u32.set(plan.brickDimensions, 0); u32[3] = plan.brickResolution;
    u32.set(plan.sampleDimensions, 4); u32[7] = plan.samplesPerBrick;
    f32.set(plan.domainOrigin, 8); f32[11] = plan.fineCellWidth;
    u32.set([plan.hashCapacity, plan.maximumHashProbes, plan.maximumResidentBricks,
      this.current.generation, this.next.generation, plan.fineFactor], 12);
    u32[18] = seedSource?.affineValues ? 1 : 0; u32[19] = initialDilationBrickRings;
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
    ];
    const publishEntries: GPUBindGroupEntry[] = [
      { binding: 0, resource: resource(this.params) }, { binding: 1, resource: resource(this.current.hash) },
      { binding: 2, resource: resource(this.next.hash) }, { binding: 3, resource: resource(this.next.metadata) },
      { binding: 4, resource: resource(this.next.worklist) }, { binding: 5, resource: resource(this.next.flags) },
      { binding: 6, resource: resource(this.next.phi) }, { binding: 7, resource: resource(this.control) },
      { binding: 8, resource: resource(seedSource?.buffer ?? this.emptySeeds) },
      { binding: 10, resource: resource(this.current.rollbackPhi) },
      ...extraPublishEntries,
    ];
    const snapshotEntries: GPUBindGroupEntry[] = [
      { binding: 0, resource: resource(this.params) }, { binding: 1, resource: resource(this.current.metadata) },
      { binding: 2, resource: resource(this.current.worklist) }, { binding: 3, resource: resource(this.current.flags) },
      { binding: 4, resource: resource(this.current.phi) }, { binding: 7, resource: resource(this.control) },
      { binding: 10, resource: resource(this.current.rollbackPhi) },
    ];
    const run = (pipeline: GPUComputePipeline, entries: GPUBindGroupEntry[], label: string, groups = 1,
      used?: readonly number[]) => {
      const selected = used ? entries.filter((entry) => used.includes(entry.binding)) : entries;
      const group = this.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: selected });
      const pass = encoder.beginComputePass({ label }); pass.setPipeline(pipeline); pass.setBindGroup(0, group);
      const dispatch = planFineLevelSetDispatch2D(groups, this.device.limits.maxComputeWorkgroupsPerDimension);
      if (dispatch.workgroups > 0) pass.dispatchWorkgroups(dispatch.x, dispatch.y, dispatch.z);
      pass.end();
    };
    const clearItems = Math.max(plan.hashCapacity * 2, 5 + plan.maximumResidentBricks, 8);
    run(this.clearPipeline, discoverEntries, "Clear global fine topology candidates", Math.ceil(clearItems / 64),
      [0, 5, 6, 7]);
    run(this.discoverPipeline, discoverEntries, "Discover global fine interface bricks",
      Math.ceil(plan.maximumResidentBricks / 64), [0, 1, 2, 3, 4, 5, 6, 7]);
    run(this.externalSeedPipeline, discoverEntries, "Insert external global fine seed bricks",
      Math.ceil(plan.maximumResidentBricks / 64), [0, 2, 5, 6, 7, 8]);
    run(this.beginDilationPipeline, discoverEntries, "Begin global fine topology dilation", 1, [0, 6, 7]);
    for (let ring = 0; ring < initialDilationBrickRings; ring += 1) {
      run(this.dilatePipeline, discoverEntries, `Dilate global fine topology ring ${ring + 1}`,
        Math.ceil(plan.maximumResidentBricks / 64), [0, 5, 6, 7]);
      run(this.advanceDilationPipeline, discoverEntries, `Advance global fine topology ring ${ring + 1}`,
        1, [0, 6, 7]);
    }
    run(this.snapshotPipeline, snapshotEntries, "Snapshot current global fine payload", plan.maximumResidentBricks,
      [0, 1, 3, 4, 10]);
    run(this.assignPipeline, publishEntries, "Assign global fine pages", Math.ceil(plan.maximumResidentBricks / 64),
      [0, 2, 3, 4, 7]);
    run(this.initializePipeline, publishEntries, "Initialize global fine samples", plan.maximumResidentBricks,
      [0, 1, 3, 5, 6, 7, 8, 9, 10]);
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
  }

  /** Commit only after the complete transport/topology/redistance/volume chain
   * is valid. A failed target is replaced GPU-side by the previous valid
   * generation, retagged for the target slot, before any later consumer runs. */
  encodeFinalizePublication(encoder: GPUCommandEncoder, controls: {
    redistance: GPUBuffer; volume?: GPUBuffer; transport?: GPUBuffer;
  }): void {
    const resource = (buffer: GPUBuffer) => ({ buffer });
    const gate = encoder.beginComputePass({ label: "Gate complete global fine publication" });
    gate.setPipeline(this.publicationPipeline);
    gate.setBindGroup(0, this.device.createBindGroup({ layout: this.publicationPipeline.getBindGroupLayout(0), entries: [
      { binding: 7, resource: resource(this.control) },
      { binding: 11, resource: resource(controls.redistance) },
      { binding: 12, resource: resource(controls.volume ?? this.validVolumeControl) },
      { binding: 13, resource: resource(controls.transport ?? this.validTransportControl) },
    ] }));
    gate.dispatchWorkgroups(1); gate.end();
    const rollback = encoder.beginComputePass({ label: "Rollback rejected complete global fine generation" });
    rollback.setPipeline(this.rollbackPipeline);
    rollback.setBindGroup(0, this.device.createBindGroup({ layout: this.rollbackPipeline.getBindGroupLayout(0), entries: [
      { binding: 0, resource: resource(this.params) }, { binding: 1, resource: resource(this.current.hash) },
      { binding: 2, resource: resource(this.current.metadata) }, { binding: 3, resource: resource(this.current.worklist) },
      { binding: 4, resource: resource(this.next.hash) }, { binding: 5, resource: resource(this.next.metadata) },
      { binding: 6, resource: resource(this.next.worklist) }, { binding: 7, resource: resource(this.control) },
    ] }));
    rollback.dispatchWorkgroups(1); rollback.end();
    const restore = encoder.beginComputePass({ label: "Restore rejected complete global fine payload" });
    restore.setPipeline(this.restorePipeline);
    restore.setBindGroup(0, this.device.createBindGroup({ layout: this.restorePipeline.getBindGroupLayout(0), entries: [
      { binding: 0, resource: resource(this.params) }, { binding: 1, resource: resource(this.current.metadata) },
      { binding: 3, resource: resource(this.next.flags) }, { binding: 4, resource: resource(this.next.phi) },
      { binding: 7, resource: resource(this.control) }, { binding: 10, resource: resource(this.current.rollbackPhi) },
    ] }));
    const dispatch = planFineLevelSetDispatch2D(this.current.plan.maximumResidentBricks,
      this.device.limits.maxComputeWorkgroupsPerDimension);
    restore.dispatchWorkgroups(dispatch.x, dispatch.y, dispatch.z); restore.end();
  }

  destroy(): void { this.control.destroy(); this.params.destroy(); this.emptySeeds.destroy();
    this.validVolumeControl.destroy(); this.validTransportControl.destroy(); }
}

export function makeFineLevelSetTopologyWGSL(coarsePhiWGSL: string): string {
  return /* wgsl */ `
const INVALID:u32=0xffffffffu;const VALID:u32=1u;const CAPACITY:u32=1u;const HASH:u32=2u;const NONFINITE:u32=4u;const MALFORMED:u32=8u;
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
fn externalSeedPhi(key:u32,finestPoint:vec3f)->f32{if(params.affineSeeds==0u){return 3.402823e38;}let keyBase=2u+params.pageCapacity;let valueBase=keyBase+params.hashCapacity;let planeBase=valueBase+params.hashCapacity;let start=externalSeedHash(key)&(params.hashCapacity-1u);
 for(var probe=0u;probe<32u;probe+=1u){let slot=(start+probe)&(params.hashCapacity-1u);let stored=externalSeeds[keyBase+slot];if(stored==key){let seed=externalSeeds[valueBase+slot];if(seed>=params.pageCapacity){return 3.402823e38;}let base=planeBase+seed*6u;let packed=externalSeeds[base];let leafOrigin=vec3f(vec3u(packed&1023u,(packed>>10u)&1023u,(packed>>20u)&1023u));let size=f32(externalSeeds[base+1u]);let centre=leafOrigin+vec3f(0.5*size);let value=bitcast<f32>(externalSeeds[base+2u]);let gradient=vec3f(bitcast<f32>(externalSeeds[base+3u]),bitcast<f32>(externalSeeds[base+4u]),bitcast<f32>(externalSeeds[base+5u]));return value+dot(gradient,finestPoint-centre);}if(stored==INVALID){break;}}
 return 3.402823e38;}
fn linearInvocation(wid:vec3u,nwg:vec3u,local:u32)->u32{return fineLinearWorkgroup(wid,nwg)*64u+local;}
@compute @workgroup_size(64) fn clearDesiredGeneration(@builtin(workgroup_id)wid:vec3u,@builtin(num_workgroups)nwg:vec3u,@builtin(local_invocation_index)local:u32){let item=linearInvocation(wid,nwg,local);if(item<params.hashCapacity*2u){atomicStore(&targetA[item],INVALID);}if(item<5u+params.pageCapacity){targetB[item]=0u;}if(item<8u&&item!=6u){atomicStore(&control[item],0u);}if(item==6u){atomicStore(&control[6],params.dilationBrickRings);}}
@compute @workgroup_size(64) fn discoverInterfaceBricks(@builtin(workgroup_id)wid:vec3u,@builtin(num_workgroups)nwg:vec3u,@builtin(local_invocation_index)local:u32){let work=linearInvocation(wid,nwg,local);let rawCount=sourceB[0];if(work==0u&&rawCount>params.pageCapacity){atomicOr(&control[0],MALFORMED);}let activeCount=min(rawCount,params.pageCapacity);if(work>=activeCount){return;}let id=sourceB[5u+work];if(id>=params.pageCapacity||sourceA[id*10u+2u]!=params.currentGeneration){atomicOr(&control[0],MALFORMED);return;}var interfaceBrick=false;for(var sample=0u;sample<params.samplesPerBrick&&!interfaceBrick;sample+=1u){let index=id*params.samplesPerBrick+sample;if((sourceC[index]&VALID)==0u){continue;}let center=bitcast<f32>(sourceD[index]);if(!finite(center)){atomicOr(&control[0],MALFORMED);continue;}for(var direction=0u;direction<6u;direction+=1u){let neighbor=currentNeighbor(id,sample,direction);if(neighbor==INVALID||(sourceC[neighbor]&VALID)==0u){continue;}let other=bitcast<f32>(sourceD[neighbor]);if(finite(other)&&(other<0.0)!=(center<0.0)){interfaceBrick=true;break;}}}if(interfaceBrick){atomicAdd(&control[1],1u);insertDesired(sourceA[id*10u+1u]);}}
@compute @workgroup_size(64) fn insertExternalSeeds(@builtin(workgroup_id)wid:vec3u,@builtin(num_workgroups)nwg:vec3u,@builtin(local_invocation_index)local:u32){let currentPublished=sourceB[1]==params.currentGeneration&&sourceB[3]==1u&&sourceB[4]==1u;if(currentPublished){return;}let seed=linearInvocation(wid,nwg,local);if(arrayLength(&externalSeeds)<2u){return;}let rawCount=externalSeeds[0];let available=arrayLength(&externalSeeds)-2u;if(seed==0u&&(externalSeeds[1]!=0u||rawCount>params.pageCapacity||rawCount>available)){atomicOr(&control[0],CAPACITY);atomicMax(&control[6],max(rawCount,params.pageCapacity+1u));}let count=min(rawCount,min(params.pageCapacity,available));if(seed>=count){return;}let key=externalSeeds[2u+seed];if(key<params.brickDimensions.x*params.brickDimensions.y*params.brickDimensions.z){insertDesired(key);}else{atomicOr(&control[0],MALFORMED);}}
@compute @workgroup_size(1) fn beginDesiredDilation(){targetB[0]=0u;targetB[1]=min(atomicLoad(&control[2]),params.pageCapacity);}
@compute @workgroup_size(64) fn dilateDesiredRing(@builtin(workgroup_id)wid:vec3u,@builtin(num_workgroups)nwg:vec3u,@builtin(local_invocation_index)local:u32){let work=linearInvocation(wid,nwg,local);let layerStart=targetB[0];let layerEnd=targetB[1];if(atomicLoad(&control[0])!=0u||work>=layerEnd-layerStart){return;}let center=vec3i(unpackBrick(targetB[5u+layerStart+work]));for(var dz=-1;dz<=1;dz+=1){for(var dy=-1;dy<=1;dy+=1){for(var dx=-1;dx<=1;dx+=1){if(dx==0&&dy==0&&dz==0){continue;}let q=center+vec3i(dx,dy,dz);if(all(q>=vec3i(0))&&all(q<vec3i(params.brickDimensions))){insertDesired(packBrick(vec3u(q)));}}}}}
@compute @workgroup_size(1) fn advanceDesiredDilation(){let priorEnd=targetB[1];targetB[0]=priorEnd;targetB[1]=min(atomicLoad(&control[2]),params.pageCapacity);}
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
const CORE:u32=2u;
struct Params { header:vec4u,tail:vec4u }
struct SurfaceLeaf { packedOrigin:u32,size:u32,flags:u32,pad:u32,phiGradient:vec4f,motion:vec4f }
struct Candidate { row:u32,flags:u32 }
@group(0) @binding(0) var<uniform> params:Params;
@group(0) @binding(1) var<storage,read> leaves:array<SurfaceLeaf>;
@group(0) @binding(2) var<storage,read> candidates:array<Candidate>;
@group(0) @binding(3) var<storage,read> candidateControl:array<u32>;
@group(0) @binding(4) var<storage,read_write> seeds:array<u32>;
fn unpackOrigin(word:u32)->vec3u{return vec3u(word&1023u,(word>>10u)&1023u,(word>>20u)&1023u);}
fn brickDimensions()->vec3u{return vec3u(params.header.z,params.header.w,params.tail.x);}
fn packBrick(coord:vec3u)->u32{let dims=brickDimensions();return coord.x+dims.x*(coord.y+dims.y*coord.z);}
fn seedHash(key:u32)->u32{var value=key*0x9e3779b1u;value=(value^(value>>16u))*0x7feb352du;return value^(value>>15u);}
fn seedKeyBase()->u32{return 2u+params.tail.y;}fn seedValueBase()->u32{return seedKeyBase()+params.tail.w;}fn seedPlaneBase()->u32{return seedValueBase()+params.tail.w;}
fn clearSeeds(){seeds[0]=0u;seeds[1]=0u;for(var slot=0u;slot<params.tail.w;slot+=1u){seeds[seedKeyBase()+slot]=0xffffffffu;seeds[seedValueBase()+slot]=0xffffffffu;}}
fn appendSeed(key:u32,leaf:SurfaceLeaf){let hashCapacity=params.tail.w;let start=seedHash(key)&(hashCapacity-1u);for(var probe=0u;probe<32u;probe+=1u){let slot=(start+probe)&(hashCapacity-1u);let at=seedKeyBase()+slot;let stored=seeds[at];if(stored==key){return;}if(stored==0xffffffffu){let count=seeds[0];if(count>=params.tail.y){seeds[1]=1u;return;}seeds[at]=key;seeds[seedValueBase()+slot]=count;seeds[2u+count]=key;let base=seedPlaneBase()+count*6u;seeds[base]=leaf.packedOrigin;seeds[base+1u]=leaf.size;seeds[base+2u]=bitcast<u32>(leaf.phiGradient.x);seeds[base+3u]=bitcast<u32>(leaf.phiGradient.y);seeds[base+4u]=bitcast<u32>(leaf.phiGradient.z);seeds[base+5u]=bitcast<u32>(leaf.phiGradient.w);seeds[0]=count+1u;return;}}seeds[1]=2u;}
@compute @workgroup_size(1) fn emitSeeds(){clearSeeds();let count=min(candidateControl[0],arrayLength(&candidates));
 for(var candidate=0u;candidate<count;candidate+=1u){let item=candidates[candidate];if((item.flags&CORE)==0u||item.row>=arrayLength(&leaves)){continue;}
  let leaf=leaves[item.row];let origin=unpackOrigin(leaf.packedOrigin);let first=origin*params.header.x/params.header.y;
  var last=(origin+vec3u(max(1u,leaf.size)))*params.header.x-vec3u(1);last/=params.header.y;last=min(last,brickDimensions()-vec3u(1));
  for(var z=first.z;z<=last.z;z+=1u){for(var y=first.y;y<=last.y;y+=1u){for(var x=first.x;x<=last.x;x+=1u){
   appendSeed(packBrick(vec3u(x,y,z)),leaf);if(seeds[1]!=0u){return;}
  }}}
 }}
@compute @workgroup_size(1) fn emitAllInterfaceSeeds(){clearSeeds();let count=min(candidateControl[0],arrayLength(&leaves));
 for(var row=0u;row<count;row+=1u){let leaf=leaves[row];if((leaf.flags&CORE)==0u){continue;}let origin=unpackOrigin(leaf.packedOrigin);let first=origin*params.header.x/params.header.y;
  var last=(origin+vec3u(max(1u,leaf.size)))*params.header.x-vec3u(1);last/=params.header.y;last=min(last,brickDimensions()-vec3u(1));
  for(var z=first.z;z<=last.z;z+=1u){for(var y=first.y;y<=last.y;y+=1u){for(var x=first.x;x<=last.x;x+=1u){appendSeed(packBrick(vec3u(x,y,z)),leaf);if(seeds[1]!=0u){return;}}}}
 }}
`;
