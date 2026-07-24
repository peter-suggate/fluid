import {
  FINE_LEVELSET_INVALID,
  type FineLevelSetBrickPlan,
  type FineLevelSetGPUGenerationData,
} from "./octree-fine-levelset-bricks";

/** GPU-resident authority for one domain-global fine level-set generation. */
export interface WebGPUFineLevelSetBrickSource {
  plan: FineLevelSetBrickPlan;
  generation: number;
  generationSlot: 0 | 1;
  params: GPUBuffer;
  metadata: GPUBuffer;
  worklist: GPUBuffer;
  flags: GPUBuffer;
  phi: GPUBuffer;
  workA: GPUBuffer;
  workB: GPUBuffer;
  /**
   * Transactional copy of the last published signed phi. Topology rollback
   * must not alias either work buffer: transport uses workA and the Section 5
   * JFA-CPT uses both work channels for closest-point ping-pong.
   */
  rollbackPhi: GPUBuffer;
  /** GPU-published compact coarse-octree phi directory used outside fine validity. */
  coarsePhiDirectory?: GPUBuffer;
  coarsePhiRowCapacity?: number;
  /** Publication provenance; consumers use topology control only to validate coarse/fine epoch pairing. */
  topologyControl?: GPUBuffer;
  /** Diagnostic-only seed transaction control. */
  seedControl?: GPUBuffer;
}

function storageBuffer(device: GPUDevice, size: number, label: string): GPUBuffer {
  return device.createBuffer({
    label,
    size: Math.max(4, Math.ceil(size / 4) * 4),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });
}

function assertLength(actual: number, expected: number, label: string): void {
  if (actual !== expected) throw new RangeError(`${label} has ${actual} words; expected ${expected}`);
}

function uploadArray(queue: GPUQueue, target: GPUBuffer, data: ArrayBufferView<ArrayBufferLike>): void {
  queue.writeBuffer(target, 0, data.buffer as ArrayBuffer, data.byteOffset, data.byteLength);
}

/**
 * Owns two sorted worklist/metadata generations and one capacity-checked payload
 * pool.  Publishing to a slot is explicit; callers must retire command buffers
 * that reference an old slot before repurposing an old-only physical page.
 */
export class WebGPUFineLevelSetBricks {
  readonly plan: FineLevelSetBrickPlan;
  /** Payload/directory plan plus both persistent 80-byte parameter blocks. */
  readonly allocatedBytes: number;
  readonly flags: GPUBuffer;
  readonly phi: GPUBuffer;
  readonly workA: GPUBuffer;
  readonly workB: GPUBuffer;
  readonly rollbackPhi: GPUBuffer;
  readonly metadata: readonly [GPUBuffer, GPUBuffer];
  readonly worklists: readonly [GPUBuffer, GPUBuffer];
  readonly params: readonly [GPUBuffer, GPUBuffer];
  private readonly generations: [number, number] = [0, 0];

  constructor(private readonly device: GPUDevice, plan: FineLevelSetBrickPlan) {
    this.plan = plan;
    const sampleWords = plan.maximumResidentBricks * plan.samplesPerBrick;
    // `plan.allocatedBytes` contains the four paper-facing payload channels
    // (flags, phi, and two closest-point work channels). Rollback is a separate
    // publication-transaction cost, shared by both A/B page-table slots.
    // Each generation carries a flat logical-key -> physical-page directory
    // after the compact worklist. The domain is only 4,096 words in the paper
    // path and removes a lower_bound from every fine interpolation tap.
    this.allocatedBytes = plan.allocatedBytes + sampleWords * 4 + 2 * 80
      + 2 * plan.logicalBrickCount * 4;
    this.flags = storageBuffer(device, sampleWords * 4, "fine-levelset flags");
    this.phi = storageBuffer(device, sampleWords * 4, "fine-levelset phi");
    this.workA = storageBuffer(device, sampleWords * 4, "fine-levelset work A");
    this.workB = storageBuffer(device, sampleWords * 4, "fine-levelset work B");
    this.rollbackPhi = storageBuffer(device, sampleWords * 4, "fine-levelset signed-phi rollback snapshot");
    this.metadata = [
      storageBuffer(device, plan.maximumResidentBricks * 40, "fine-levelset metadata generation 0"),
      storageBuffer(device, plan.maximumResidentBricks * 40, "fine-levelset metadata generation 1"),
    ];
    this.worklists = [
      storageBuffer(device, (5 + plan.maximumResidentBricks + plan.logicalBrickCount) * 4,
        "fine-levelset worklist and direct page directory generation 0"),
      storageBuffer(device, (5 + plan.maximumResidentBricks + plan.logicalBrickCount) * 4,
        "fine-levelset worklist and direct page directory generation 1"),
    ];
    this.params = [0, 1].map((slot) => device.createBuffer({
      label: `fine-levelset parameters generation ${slot}`,
      size: 80,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })) as [GPUBuffer, GPUBuffer];
  }

  uploadGeneration(data: FineLevelSetGPUGenerationData): WebGPUFineLevelSetBrickSource {
    if (!Number.isSafeInteger(data.generation) || data.generation < 1) {
      throw new RangeError("Fine level-set GPU publication generation must be positive");
    }
    if (data.activeCount > this.plan.maximumResidentBricks || data.worklistWords[0] !== data.activeCount) {
      throw new RangeError("Fine level-set GPU publication exceeds capacity or has a mismatched worklist");
    }
    const sampleWords = this.plan.maximumResidentBricks * this.plan.samplesPerBrick;
    assertLength(data.metadataWords.length, this.plan.maximumResidentBricks * 10, "Fine level-set metadata");
    assertLength(data.worklistWords.length, 5 + this.plan.maximumResidentBricks, "Fine level-set worklist");
    assertLength(data.flags.length, sampleWords, "Fine level-set flags");
    assertLength(data.phi.length, sampleWords, "Fine level-set phi");
    assertLength(data.workA.length, sampleWords, "Fine level-set work A");
    assertLength(data.workB.length, sampleWords, "Fine level-set work B");
    const slot = (data.generation & 1) as 0 | 1;
    const parameterBytes = this.parameterBytes(data.generation, data.activeCount);
    this.device.queue.writeBuffer(this.params[slot], 0, parameterBytes);
    uploadArray(this.device.queue, this.metadata[slot], data.metadataWords);
    uploadArray(this.device.queue, this.worklists[slot], data.worklistWords);
    const direct = new Uint32Array(this.plan.logicalBrickCount).fill(FINE_LEVELSET_INVALID);
    for (let work = 0; work < data.activeCount; work += 1) {
      const physicalId = data.worklistWords[5 + work]!;
      const key = data.metadataWords[physicalId * 10 + 1]!;
      if (physicalId >= this.plan.maximumResidentBricks || key >= direct.length) {
        throw new RangeError("Fine level-set publication contains an invalid direct-page identity");
      }
      direct[key] = physicalId;
    }
    this.device.queue.writeBuffer(this.worklists[slot],
      (5 + this.plan.maximumResidentBricks) * 4, direct);
    uploadArray(this.device.queue, this.flags, data.flags);
    uploadArray(this.device.queue, this.phi, data.phi);
    uploadArray(this.device.queue, this.workA, data.workA);
    uploadArray(this.device.queue, this.workB, data.workB);
    this.generations[slot] = data.generation;
    return this.source(slot);
  }

  /** Reserves the alternate ABI slot for an entirely GPU-authored generation. */
  prepareGPUGeneration(generation: number): WebGPUFineLevelSetBrickSource {
    if (!Number.isSafeInteger(generation) || generation < 1) {
      throw new RangeError("Fine level-set GPU generation must be positive");
    }
    const slot = (generation & 1) as 0 | 1;
    this.device.queue.writeBuffer(this.params[slot], 0, this.parameterBytes(generation, 0));
    this.generations[slot] = generation;
    return this.source(slot);
  }

  /**
   * Retags a retired page-table slot for its next GPU-authored generation.
   *
   * The caller must preserve queue order: commands consuming the slot's old
   * generation have to precede commands publishing the new generation.  No
   * payload allocation or CPU readback is involved; the two page tables keep
   * alternating while the shared, capacity-bounded payload pages are reused.
   */
  repurposeGPUGeneration(source: WebGPUFineLevelSetBrickSource, generation: number): void {
    if (!Number.isSafeInteger(generation) || generation < 1) {
      throw new RangeError("Fine level-set GPU generation must be positive");
    }
    const slot = (generation & 1) as 0 | 1;
    if (slot !== source.generationSlot) {
      throw new RangeError("Fine level-set generation parity must match its page-table slot");
    }
    if (source.plan !== this.plan || source.params !== this.params[slot]
      || source.metadata !== this.metadata[slot]
      || source.worklist !== this.worklists[slot] || source.flags !== this.flags
      || source.phi !== this.phi || source.workA !== this.workA || source.workB !== this.workB
      || source.rollbackPhi !== this.rollbackPhi) {
      throw new RangeError("Fine level-set generation source is not owned by this page pool");
    }
    this.device.queue.writeBuffer(source.params, 0, this.parameterBytes(generation, 0));
    this.generations[slot] = generation;
    source.generation = generation;
  }

  /** Empty valid worklist bootstrap; contains no CPU-authored surface samples. */
  initializeEmptyGPUGeneration(generation: number): WebGPUFineLevelSetBrickSource {
    const source = this.prepareGPUGeneration(generation);
    const emptyMetadata = new Uint32Array(this.plan.maximumResidentBricks * 10).fill(FINE_LEVELSET_INVALID);
    uploadArray(this.device.queue, source.metadata, emptyMetadata);
    const emptyWorklist = new Uint32Array(5 + this.plan.maximumResidentBricks);
    emptyWorklist[1] = generation;
    emptyWorklist[3] = 1;
    emptyWorklist[4] = 1;
    this.device.queue.writeBuffer(source.worklist, 0, emptyWorklist);
    this.device.queue.writeBuffer(source.worklist,
      (5 + this.plan.maximumResidentBricks) * 4,
      new Uint32Array(this.plan.logicalBrickCount).fill(FINE_LEVELSET_INVALID));
    return source;
  }

  private parameterBytes(generation: number, activeCount: number): ArrayBuffer {
    const parameterBytes = new ArrayBuffer(80);
    const u32 = new Uint32Array(parameterBytes);
    const f32 = new Float32Array(parameterBytes);
    u32.set(this.plan.brickDimensions, 0);
    u32[3] = this.plan.brickResolution;
    u32.set(this.plan.sampleDimensions, 4);
    u32[7] = this.plan.samplesPerBrick;
    f32.set(this.plan.domainOrigin, 8);
    f32[11] = this.plan.fineCellWidth;
    u32[12] = this.plan.maximumResidentBricks;
    u32[13] = 5;
    u32[14] = this.plan.maximumResidentBricks;
    u32[15] = generation;
    u32[16] = activeCount;
    u32[17] = FINE_LEVELSET_INVALID;
    u32[18] = this.plan.fineFactor;
    u32[19] = 0;
    return parameterBytes;
  }

  source(slot: 0 | 1): WebGPUFineLevelSetBrickSource {
    const generation = this.generations[slot];
    if (generation === 0) throw new RangeError(`Fine level-set generation slot ${slot} has not been published`);
    return {
      plan: this.plan, generation, generationSlot: slot, params: this.params[slot],
      metadata: this.metadata[slot], worklist: this.worklists[slot], flags: this.flags,
      phi: this.phi, workA: this.workA, workB: this.workB, rollbackPhi: this.rollbackPhi,
    };
  }

  setTransportTimestep(slot: 0 | 1, timestep: number): void {
    if (!Number.isFinite(timestep) || timestep < 0) {
      throw new RangeError("Fine level-set transport timestep must be finite and non-negative");
    }
    this.device.queue.writeBuffer(this.params[slot], 76, new Float32Array([timestep]));
  }

  destroy(): void {
    this.flags.destroy(); this.phi.destroy(); this.workA.destroy(); this.workB.destroy(); this.rollbackPhi.destroy();
    for (const buffer of [...this.metadata, ...this.worklists, ...this.params]) buffer.destroy();
  }
}

/** Exact O(1) lookup shared by every global-fine consumer.
 * The direct table follows the compact sorted worklist in the same binding, so
 * consumers need no extra bind slot. Metadata still validates identity and
 * generation before the physical page becomes visible. */
export function makeFineLevelSetSortedWorklistLookupWGSL(
  params: string,
  metadata: string,
  worklist: string,
  functionName = "lookupFineBrick",
  brickDimensions = "brickDimensions",
): string {
  return /* wgsl */ `
fn ${functionName}(key:u32)->u32 {
  if(${params}.worklistHeaderWords!=5u||arrayLength(&${worklist})<5u||${worklist}[1]!=${params}.generation
    ||${worklist}[3]!=1u||${worklist}[4]!=1u){return INVALID;}
  let count=min(${worklist}[0],min(${params}.worklistCapacity,${params}.pageCapacity));
  let logicalCount=${params}.${brickDimensions}.x*${params}.${brickDimensions}.y*${params}.${brickDimensions}.z;
  let directoryBase=5u+${params}.worklistCapacity;
  if(5u+count>arrayLength(&${worklist})||key>=logicalCount
    ||directoryBase+key>=arrayLength(&${worklist})){return INVALID;}
  let physicalId=${worklist}[directoryBase+key];let base=physicalId*10u;
  return select(INVALID,physicalId,physicalId<${params}.pageCapacity&&base+2u<arrayLength(&${metadata})
    &&${metadata}[base]==physicalId&&${metadata}[base+1u]==key&&${metadata}[base+2u]==${params}.generation);
}`;
}

/**
 * Consumer/reference shader.  A query returns the coarse value supplied by
 * the caller whenever the position is outside the domain, its brick is absent,
 * or its page generation/valid bit is stale.  Missing storage is never zero.
 */
export const fineLevelSetBrickSamplingWGSL = /* wgsl */ `
const INVALID:u32=0xffffffffu;
const VALID:u32=1u;

struct Params {
  brickDimensions:vec3u,
  brickResolution:u32,
  sampleDimensions:vec3u,
  samplesPerBrick:u32,
  domainOrigin:vec3f,
  fineCellWidth:f32,
  worklistCapacity:u32,
  worklistHeaderWords:u32,
  pageCapacity:u32,
  generation:u32,
  activeCount:u32,
  invalid:u32,
  fineFactor:u32,
  timestep:f32,
}
struct Query { position:vec3f, coarsePhi:f32 }
struct Result { phi:f32, found:u32, physicalId:u32, localIndex:u32 }

@group(0) @binding(0) var<uniform> params:Params;
@group(0) @binding(1) var<storage,read> worklist:array<u32>;
@group(0) @binding(2) var<storage,read> metadata:array<u32>;
@group(0) @binding(3) var<storage,read> sampleFlags:array<u32>;
@group(0) @binding(4) var<storage,read> phi:array<f32>;
@group(0) @binding(5) var<storage,read> queries:array<Query>;
@group(0) @binding(6) var<storage,read_write> results:array<Result>;

${makeFineLevelSetSortedWorklistLookupWGSL("params", "metadata", "worklist", "lookupBrick")}
fn packBrick(coord:vec3u)->u32 {
  return coord.x+params.brickDimensions.x*(coord.y+params.brickDimensions.y*coord.z);
}
fn sampleFine(position:vec3f,coarsePhi:f32)->Result {
  let relative=(position-params.domainOrigin)/params.fineCellWidth;
  let q=vec3i(floor(relative));
  if(any(q<vec3i(0))||any(q>=vec3i(params.sampleDimensions))){return Result(coarsePhi,0u,INVALID,INVALID);}
  let uq=vec3u(q);
  let brick=uq/params.brickResolution;
  let local=uq-brick*params.brickResolution;
  let localIndex=local.x+params.brickResolution*(local.y+params.brickResolution*local.z);
  let physicalId=lookupBrick(packBrick(brick));
  if(physicalId==INVALID){return Result(coarsePhi,0u,INVALID,localIndex);}
  let sampleIndex=physicalId*params.samplesPerBrick+localIndex;
  if((sampleFlags[sampleIndex]&VALID)==0u){return Result(coarsePhi,0u,physicalId,localIndex);}
  let value=phi[sampleIndex];
  if(value!=value||abs(value)>3.402823e38){return Result(coarsePhi,0u,physicalId,localIndex);}
  return Result(value,1u,physicalId,localIndex);
}

@compute @workgroup_size(64)
fn sampleQueries(@builtin(global_invocation_id) invocation:vec3u) {
  let index=invocation.x;
  if(index>=arrayLength(&queries)||index>=arrayLength(&results)){return;}
  let query=queries[index];
  results[index]=sampleFine(query.position,query.coarsePhi);
}
`;
