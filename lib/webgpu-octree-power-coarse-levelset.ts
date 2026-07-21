/** GPU-resident WP8 coarse-octree advection, redistance, and fine correction. */

import { OCTREE_COARSE_PHI_BYTES, OCTREE_COARSE_PHI_FLAG } from "./octree-coarse-levelset";
import { OCTREE_FINE_PHI_CONTRIBUTION_BYTES, type OctreeCoarsePhiCorrectionInput,
  type WebGPUOctreeCoarseLevelSet } from "./webgpu-octree-coarse-levelset";
import type { OctreePowerTopologySource } from "./webgpu-octree-power-topology";

export const OCTREE_POWER_COARSE_LEVELSET_CONTROL_BYTES = 64;
export const OCTREE_POWER_COARSE_LEVELSET_VALID = 0x8000_0000;
export const OCTREE_POWER_COARSE_LEVELSET_SAMPLE_HEADER_BYTES = 32;
export const OCTREE_POWER_COARSE_LEVELSET_SAMPLE_ENTRY_BYTES = 32;
/** One cold bootstrap plus the solver's bounded 64 encoded surface substeps. */
export const OCTREE_POWER_COARSE_LEVELSET_ENCODE_SLOTS = 65;
export const OCTREE_POWER_COARSE_LEVELSET_ERROR = Object.freeze({
  capacity: 1, invalidRow: 2, invalidVelocity: 4, invalidCatalog: 8,
  invalidFineOffsets: 16, invalidFineSample: 32, fineContributionBound: 64,
  sampleIndex: 128, invalidSource: 256, noCausalSimplex: 512,
} as const);

export interface OctreePowerCoarseLevelSetPlan {
  readonly rowCapacity: number;
  readonly redistancePasses: number;
  readonly scratchBytes: number;
  readonly sampleHashCapacity: number;
  readonly sampleDirectoryBytes: number;
  readonly allocatedBytes: number;
}

export interface OctreePowerCoarseLevelSetInput {
  readonly headers: GPUBuffer;
  readonly cellVelocities: GPUBuffer;
  readonly siteIndex: GPUBuffer;
  /** CPU count or GPU buffer whose first u32 is the compact live-row count. */
  readonly rowCount: number | GPUBuffer;
  readonly fineCorrection?: OctreeCoarsePhiCorrectionInput & {
    /** Numeric counts for host-authored tests, or a GPU buffer with
     * `(contributionCount, maximumContributionsPerRow)` at byte zero. */
    readonly contributionCount: number | GPUBuffer;
    readonly maximumContributionsPerRow?: number;
    /** One `{nearestPhi,minimumPhi,maximumPhi,valid}` record per row. */
    readonly aggregated?: boolean;
  };
}

export interface OctreePowerCoarseLevelSetOptions {
  readonly dimensions: readonly [number, number, number];
  readonly physicalCellSize: number;
  readonly dt: number;
  readonly hashCapacity: number;
  readonly maximumHashProbes?: number;
  /** Largest power-of-two leaf extent in finest-cell units. */
  readonly maximumLeafSize?: number;
  readonly generation?: number;
}

export interface OctreePowerCoarseLevelSetControl {
  readonly flags: number; readonly firstErrorRow: number; readonly rowCount: number;
  readonly advectedRows: number; readonly uniformUpdates: number; readonly transitionUpdates: number;
  readonly nearestFallbacks: number; readonly redistancePasses: number; readonly correctedRows: number;
  readonly interfaceRows: number; readonly contributionCount: number; readonly generation: number;
  readonly valid: number;
}

export interface OctreePowerCoarseDirectoryHeader {
  readonly state: number; readonly generation: number; readonly hashCapacity: number;
  readonly maximumLeafSize: number; readonly dimensions: readonly [number, number, number];
  readonly physicalCellSize: number; readonly actualHashCapacity: number;
}

/** CPU mirror of the topology shader's all-or-nothing directory gate. */
export function octreePowerCoarseDirectoryIsAuthoritative(
  header: OctreePowerCoarseDirectoryHeader,
  expectedGeneration: number,
  expectedDimensions: readonly [number, number, number],
  expectedPhysicalCellSize: number,
): boolean {
  const capacity = header.hashCapacity;
  return expectedGeneration > 0
    && header.state === OCTREE_POWER_COARSE_LEVELSET_VALID
    && (header.generation & 0x3fff_ffff) === (expectedGeneration & 0x3fff_ffff)
    && header.dimensions.every((value, axis) => value === expectedDimensions[axis])
    && Number.isFinite(header.physicalCellSize) && header.physicalCellSize > 0
    && Math.abs(header.physicalCellSize - expectedPhysicalCellSize)
      <= 1e-5 * Math.max(header.physicalCellSize, expectedPhysicalCellSize)
    && Number.isSafeInteger(capacity) && capacity > 0 && (capacity & (capacity - 1)) === 0
    && capacity === header.actualHashCapacity
    && Number.isSafeInteger(header.maximumLeafSize) && header.maximumLeafSize > 0
    && (header.maximumLeafSize & (header.maximumLeafSize - 1)) === 0;
}

/** One-binding, GPU-published source used to initialize missing fine bricks. */
export interface OctreePowerCoarseLevelSetSampleSource {
  readonly directory: GPUBuffer;
  readonly hashCapacity: number;
  readonly wgsl: (binding?: number) => string;
}

function positiveInteger(value: number, label: string): number {
  if (!Number.isSafeInteger(value) || value < 1) throw new RangeError(`${label} must be positive`);
  return value;
}
function u32(value: number, label: string): number {
  if (!Number.isSafeInteger(value) || value < 0 || value > 0xffff_ffff) throw new RangeError(`${label} must be an unsigned u32`);
  return value;
}

export function planOctreePowerCoarseLevelSet(rowCapacityValue: number, redistancePassesValue = 8): OctreePowerCoarseLevelSetPlan {
  const rowCapacity = positiveInteger(rowCapacityValue, "Power coarse-phi row capacity");
  const redistancePasses = positiveInteger(redistancePassesValue, "Power coarse-phi redistance passes");
  const scratchBytes = rowCapacity * OCTREE_COARSE_PHI_BYTES * 2;
  let sampleHashCapacity = 1; while (sampleHashCapacity < rowCapacity * 2) sampleHashCapacity *= 2;
  const sampleDirectoryBytes = OCTREE_POWER_COARSE_LEVELSET_SAMPLE_HEADER_BYTES
    + sampleHashCapacity * OCTREE_POWER_COARSE_LEVELSET_SAMPLE_ENTRY_BYTES;
  return { rowCapacity, redistancePasses, scratchBytes, sampleHashCapacity, sampleDirectoryBytes,
    allocatedBytes: scratchBytes + sampleDirectoryBytes + OCTREE_POWER_COARSE_LEVELSET_CONTROL_BYTES
      + OCTREE_POWER_COARSE_LEVELSET_ENCODE_SLOTS * (64 * (redistancePasses + 1) + 4) + 32
      + (rowCapacity + 1) * 4 + OCTREE_FINE_PHI_CONTRIBUTION_BYTES };
}

export function unpackOctreePowerCoarseLevelSetControl(words: ArrayLike<number>): OctreePowerCoarseLevelSetControl {
  if (words.length < 16) throw new RangeError("Power coarse-phi control needs sixteen words");
  return { flags: Number(words[0]) >>> 0, firstErrorRow: Number(words[1]) >>> 0, rowCount: Number(words[2]) >>> 0,
    advectedRows: Number(words[3]) >>> 0, uniformUpdates: Number(words[4]) >>> 0, transitionUpdates: Number(words[5]) >>> 0,
    nearestFallbacks: Number(words[6]) >>> 0, redistancePasses: Number(words[7]) >>> 0,
    correctedRows: Number(words[8]) >>> 0, interfaceRows: Number(words[9]) >>> 0,
    contributionCount: Number(words[10]) >>> 0, generation: Number(words[11]) >>> 0, valid: Number(words[12]) >>> 0 };
}

export class WebGPUOctreePowerCoarseLevelSet {
  readonly plan: OctreePowerCoarseLevelSetPlan;
  readonly control: GPUBuffer;
  readonly sampleDirectory: GPUBuffer;
  private readonly scratch: GPUBuffer;
  private readonly params: readonly GPUBuffer[]; private readonly hostRowCount: readonly GPUBuffer[];
  private readonly redistanceParams: readonly (readonly GPUBuffer[])[];
  private readonly emptyOffsets: GPUBuffer; private readonly emptyContributions: GPUBuffer;
  private readonly validFineControl: GPUBuffer;
  private readonly pipelines: Readonly<Record<string, GPUComputePipeline>>;
  private readonly encoderInvocationCounts = new WeakMap<GPUCommandEncoder, number>();
  private activeEncoder?: GPUCommandEncoder;
  private encodeSlot = 0;
  private destroyed = false;

  constructor(private readonly device: GPUDevice, private readonly coarse: WebGPUOctreeCoarseLevelSet,
    private readonly topology: OctreePowerTopologySource, redistancePasses = 8) {
    if (!topology.catalogTetrahedronHeaders || !topology.catalogTetrahedra || !topology.catalogTetrahedronVertices) {
      throw new RangeError("Power coarse level set requires the complete tetrahedron catalog");
    }
    this.plan = planOctreePowerCoarseLevelSet(coarse.plan.rowCapacity, redistancePasses);
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    this.scratch = device.createBuffer({ label: "Power coarse phi advection/redistance", size: this.plan.scratchBytes, usage: storage });
    this.control = device.createBuffer({ label: "Power coarse phi schedule control", size: OCTREE_POWER_COARSE_LEVELSET_CONTROL_BYTES, usage: storage });
    this.sampleDirectory = device.createBuffer({ label: "Power coarse phi sample directory", size: this.plan.sampleDirectoryBytes, usage: storage });
    this.params = Array.from({ length: OCTREE_POWER_COARSE_LEVELSET_ENCODE_SLOTS }, (_, slot) =>
      device.createBuffer({ label: `Power coarse phi schedule params ${slot}`, size: 64,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }));
    this.redistanceParams = Array.from({ length: OCTREE_POWER_COARSE_LEVELSET_ENCODE_SLOTS }, (_, slot) =>
      Array.from({ length: this.plan.redistancePasses }, (_, iteration) =>
        device.createBuffer({ label: `Power coarse phi redistance params ${slot}:${iteration}`, size: 64,
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST })));
    this.hostRowCount = Array.from({ length: OCTREE_POWER_COARSE_LEVELSET_ENCODE_SLOTS }, (_, slot) =>
      device.createBuffer({ label: `Power coarse phi host row count ${slot}`, size: 4, usage: storage }));
    this.emptyOffsets = device.createBuffer({ label: "Empty fine correction offsets", size: (this.plan.rowCapacity + 1) * 4, usage: storage });
    this.emptyContributions = device.createBuffer({ label: "Empty fine correction contribution", size: OCTREE_FINE_PHI_CONTRIBUTION_BYTES, usage: storage });
    this.validFineControl = device.createBuffer({ label: "Valid host fine-correction control", size: 32, usage: storage });
    device.queue.writeBuffer(this.validFineControl, 20, new Uint32Array([OCTREE_POWER_COARSE_LEVELSET_VALID]));
    const shaderModule = device.createShaderModule({ label: "Power coarse phi schedule", code: octreePowerCoarseLevelSetShader });
    const pipeline = (entryPoint: string) => device.createComputePipeline({ label: entryPoint, layout: "auto", compute: { module: shaderModule, entryPoint } });
    this.pipelines = Object.freeze({ prepare: pipeline("preparePowerCoarsePhi"), clearSamples: pipeline("clearPowerCoarsePhiSamples"), advect: pipeline("advectPowerCoarsePhi"),
      redistance: pipeline("redistancePowerCoarsePhi"), validateFine: pipeline("validatePowerCoarseFineCorrection"),
      publish: pipeline("publishPowerCoarsePhi"), finalize: pipeline("finalizePowerCoarsePhi") });
  }

  encode(encoder: GPUCommandEncoder, input: OctreePowerCoarseLevelSetInput, options: OctreePowerCoarseLevelSetOptions): void {
    if (this.destroyed) throw new Error("Power coarse level-set schedule is destroyed");
    if (this.activeEncoder && this.activeEncoder !== encoder) {
      throw new Error("Power coarse level-set encoder must be submitted and retired before encoding another command buffer");
    }
    this.activeEncoder = encoder;
    const encoderInvocation = this.encoderInvocationCounts.get(encoder) ?? 0;
    if (encoderInvocation >= OCTREE_POWER_COARSE_LEVELSET_ENCODE_SLOTS) {
      throw new RangeError("Power coarse level-set encoder exceeds its 65 invocation-stable parameter slots");
    }
    this.encoderInvocationCounts.set(encoder, encoderInvocation + 1);
    // A single solver command encoder owns at most 64 surface substeps; the
    // per-encoder guard above therefore prevents wrap within unsubmitted work.
    // Across solver encoders, advanceTo submits before returning, so a later
    // queue.writeBuffer is ordered after every earlier use of the recycled
    // slot even when that submission has not completed on the device yet.
    const slot = this.encodeSlot; this.encodeSlot = (this.encodeSlot + 1) % OCTREE_POWER_COARSE_LEVELSET_ENCODE_SLOTS;
    const params = this.params[slot], redistanceParams = this.redistanceParams[slot];
    const maximumRows = typeof input.rowCount === "number" ? u32(input.rowCount, "Power coarse-phi row count") : this.plan.rowCapacity;
    if (maximumRows > this.plan.rowCapacity) throw new RangeError("Power coarse-phi row count exceeds capacity");
    if (typeof input.rowCount === "number") this.device.queue.writeBuffer(this.hostRowCount[slot], 0, new Uint32Array([maximumRows]));
    const dimensions = options.dimensions.map((value) => positiveInteger(value, "Power coarse-phi dimension")) as [number, number, number];
    const hashCapacity = positiveInteger(options.hashCapacity, "Power coarse-phi hash capacity");
    if ((hashCapacity & (hashCapacity - 1)) !== 0) throw new RangeError("Power coarse-phi hash capacity must be a power of two");
    const maximumHashProbes = positiveInteger(options.maximumHashProbes ?? 32, "Power coarse-phi hash probe bound");
    const maximumLeafSize = positiveInteger(options.maximumLeafSize ?? Math.max(...dimensions), "Power coarse-phi maximum leaf size");
    if ((maximumLeafSize & (maximumLeafSize - 1)) !== 0) throw new RangeError("Power coarse-phi maximum leaf size must be a power of two");
    if (!(options.physicalCellSize > 0) || !Number.isFinite(options.physicalCellSize)
      || !Number.isFinite(options.dt) || options.dt < 0) throw new RangeError("Power coarse-phi physical parameters are invalid");
    const fine = input.fineCorrection; const gpuFineCounts = fine !== undefined && typeof fine.contributionCount !== "number";
    const contributionCount = fine === undefined ? 0 : typeof fine.contributionCount === "number"
      ? u32(fine.contributionCount, "Fine correction contribution count") : 0;
    const maximumPerRow = positiveInteger(fine?.maximumContributionsPerRow ?? 1, "Fine correction row bound");
    const generation = u32(options.generation ?? 0, "Power coarse-phi generation");
    const data = new ArrayBuffer(64), words = new Uint32Array(data), floats = new Float32Array(data);
    words.set([...dimensions, this.plan.rowCapacity, maximumRows, contributionCount, maximumPerRow, generation]);
    floats.set([options.physicalCellSize, options.dt], 8); words.set([this.plan.redistancePasses, hashCapacity,
      maximumHashProbes, fine ? (fine.aggregated ? 2 : 1) : 0, maximumLeafSize], 10);
    this.device.queue.writeBuffer(params, 0, data);
    redistanceParams.forEach((buffer, iteration) => { const copy = data.slice(0); new Uint32Array(copy)[15] = iteration;
      this.device.queue.writeBuffer(buffer, 0, copy); });
    const rowCountSource = typeof input.rowCount === "number" ? this.hostRowCount[slot] : input.rowCount;
    if (gpuFineCounts) encoder.copyBufferToBuffer(fine!.contributionCount as GPUBuffer, 0, params, 20, 8);
    const offsets = fine?.rowOffsets ?? this.emptyOffsets, contributions = fine?.contributions ?? this.emptyContributions;
    const fineControl = gpuFineCounts ? fine!.contributionCount as GPUBuffer : this.validFineControl;
    const common = new Map<number, GPUBuffer>([[0, params], [1, input.headers], [2, this.topology.metrics],
      [3, this.topology.catalogTetrahedronHeaders!], [4, this.topology.catalogTetrahedra!],
      [5, this.topology.catalogTetrahedronVertices!], [6, input.siteIndex], [7, input.cellVelocities],
      [8, this.coarse.records], [9, this.scratch], [11, offsets], [12, contributions],
      [13, this.control], [14, rowCountSource], [15, this.sampleDirectory], [16, fineControl]]);
    const bindings: Record<string, readonly number[]> = {
      prepare: [0, 13, 14, 15, 16], clearSamples: [0, 15, 16], advect: [0, 1, 2, 5, 6, 7, 8, 9, 13, 16],
      redistance: [0, 1, 2, 3, 4, 5, 6, 9, 13, 16],
      validateFine: [0, 11, 12, 13, 16], publish: [0, 1, 2, 9, 11, 12, 8, 13, 15, 16], finalize: [0, 13, 15, 16],
    };
    const dispatch = (name: keyof typeof bindings, workgroups: number) => {
      const pipeline = this.pipelines[name], group = this.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0),
        entries: bindings[name].map((binding) => ({ binding, resource: { buffer: common.get(binding)! } })) });
      const pass = encoder.beginComputePass({ label: name }); pass.setPipeline(pipeline); pass.setBindGroup(0, group);
      pass.dispatchWorkgroups(workgroups); pass.end();
    };
    dispatch("prepare", 1); dispatch("clearSamples", Math.ceil(this.plan.sampleHashCapacity / 64));
    if (maximumRows > 0) dispatch("advect", Math.ceil(maximumRows / 64));
    for (let iteration = 0; iteration < this.plan.redistancePasses; iteration += 1) {
      common.set(0, redistanceParams[iteration]);
      if (maximumRows > 0) dispatch("redistance", Math.ceil(maximumRows / 64));
    }
    common.set(0, params);
    dispatch("validateFine", Math.max(1, Math.ceil(maximumRows / 64)));
    if (maximumRows > 0) dispatch("publish", Math.ceil(maximumRows / 64));
    dispatch("finalize", 1);
  }

  /** Call immediately after submitting the finished encoder. Queue ordering
   * then makes its parameter slots safe for the next command buffer's writes. */
  retireSubmittedEncoder(encoder: GPUCommandEncoder): void {
    if (!this.activeEncoder) return;
    if (this.activeEncoder !== encoder) {
      throw new Error("Power coarse level-set can retire only its active submitted encoder");
    }
    this.encoderInvocationCounts.delete(encoder); this.activeEncoder = undefined; this.encodeSlot = 0;
  }

  get sampleSource(): OctreePowerCoarseLevelSetSampleSource {
    return { directory: this.sampleDirectory, hashCapacity: this.plan.sampleHashCapacity,
      wgsl: makeOctreePowerCoarseLevelSetSampleWGSL };
  }

  destroy(): void { if (this.destroyed) return; this.destroyed = true;
    this.scratch.destroy(); this.control.destroy(); this.sampleDirectory.destroy();
    this.params.forEach((buffer) => buffer.destroy()); this.hostRowCount.forEach((buffer) => buffer.destroy());
    this.redistanceParams.forEach((buffers) => buffers.forEach((buffer) => buffer.destroy()));
    this.emptyOffsets.destroy(); this.emptyContributions.destroy(); this.validFineControl.destroy(); }
}

/**
 * Injects a containing-leaf sampler into the fine topology shader. A valid
 * sparse coarse publication has an explicit positive-air complement: compact
 * rows cover liquid/interface leaves, while a domain point absent from every
 * containing-leaf key is outside that active set. Invalid publications and
 * out-of-domain queries still fail closed.
 * The directory is rebuilt and atomically marked valid by the coarse schedule;
 * it therefore needs only one storage binding and never consults dense phi.
 */
export function makeOctreePowerCoarseLevelSetSampleWGSL(binding = 9): string {
  if (!Number.isSafeInteger(binding) || binding < 0) throw new RangeError("Coarse-phi sample binding must be non-negative");
  return /* wgsl */ `
struct PowerCoarseSampleEntry { cellPlusOne:u32, size:u32, phi:f32, minimumPhi:f32, maximumPhi:f32, flags:u32, row:u32, physicalVolume:f32 }
struct PowerCoarseSampleDirectory { state:u32, generation:u32, hashCapacity:u32, maximumLeafSize:u32, dimensions:vec3u, physicalCellSize:f32, entries:array<PowerCoarseSampleEntry> }
@group(0) @binding(${binding}) var<storage,read> powerCoarseSamples:PowerCoarseSampleDirectory;
fn powerCoarseHash(cell:u32,size:u32)->u32{var value=cell^(size*0x9e3779b9u);value=(value^(value>>16u))*0x7feb352du;value=(value^(value>>15u))*0x846ca68bu;return value^(value>>16u);}
fn powerCoarseLookup(cell:u32,size:u32)->u32{let capacity=min(powerCoarseSamples.hashCapacity,arrayLength(&powerCoarseSamples.entries));if(capacity==0u||(capacity&(capacity-1u))!=0u){return 0xffffffffu;}let base=powerCoarseHash(cell,size)&(capacity-1u);for(var probe=0u;probe<min(32u,capacity);probe+=1u){let slot=(base+probe)&(capacity-1u);let observed=powerCoarseSamples.entries[slot].cellPlusOne;if(observed==0u){return 0xffffffffu;}if(observed==cell+1u&&powerCoarseSamples.entries[slot].size==size){return slot;}}return 0xffffffffu;}
fn sampleCoarseOctreePhi(position:vec3f)->f32{let invalidPhi=3.402823e38;if(powerCoarseSamples.state!=0x80000000u||!(powerCoarseSamples.physicalCellSize>0.0)){return invalidPhi;}let grid=position/powerCoarseSamples.physicalCellSize;if(any(grid<vec3f(0.0))||any(grid>=vec3f(powerCoarseSamples.dimensions))){return invalidPhi;}let q=vec3u(floor(grid));var size=1u;loop{let origin=(q/vec3u(size))*vec3u(size);let cell=origin.x+powerCoarseSamples.dimensions.x*(origin.y+powerCoarseSamples.dimensions.y*origin.z);let slot=powerCoarseLookup(cell,size);if(slot!=0xffffffffu){let entry=powerCoarseSamples.entries[slot];if((entry.flags&${OCTREE_COARSE_PHI_FLAG.valid}u)!=0u){return entry.phi;}return invalidPhi;}if(size>=powerCoarseSamples.maximumLeafSize){break;}size*=2u;}return powerCoarseSamples.physicalCellSize*f32(max(1u,powerCoarseSamples.maximumLeafSize));}
`;
}

export const octreePowerCoarseLevelSetShader = /* wgsl */ `
struct Params { dimensionsCapacity:vec4u, countsGeneration:vec4u, physical:vec2f, redistancePasses:u32, hashCapacity:u32, maximumHashProbes:u32, hasFine:u32, maximumLeafSize:u32, iteration:u32 }
struct LeafHeader { cell:u32, entryStart:u32, entryCount:u32, size:u32, diagonal:f32, rhs:f32, pad0:u32, pad1:u32, gradient:vec4f }
struct Metric { topologyCode:u32, transformAndFlags:u32, volume:f32, reserved:u32 }
struct TetraHeader { first:u32, count:u32, flags:u32 } struct TetraVertex { offsetSize:vec4f }
struct SiteIndex { cellPlusOne:atomic<u32>, size:u32, row:u32, pad:u32 } struct CoarsePhi { phi:f32, minimumPhi:f32, maximumPhi:f32, flags:u32 }
struct FineContribution { phi:f32, distanceSquared:f32, valid:u32, pad:u32 }
struct SampleEntry { cellPlusOne:atomic<u32>, size:u32, phi:f32, minimumPhi:f32, maximumPhi:f32, flags:u32, row:u32, physicalVolume:f32 }
struct SampleDirectory { state:atomic<u32>, generation:u32, hashCapacity:u32, maximumLeafSize:u32, dimensions:vec3u, physicalCellSize:f32, entries:array<SampleEntry> }
struct Control { flags:atomic<u32>, firstError:atomic<u32>, rowCount:u32, advected:atomic<u32>, uniform:atomic<u32>, transition:atomic<u32>, nearest:atomic<u32>, passes:u32, corrected:atomic<u32>, interfaces:atomic<u32>, contributionCount:u32, generation:u32, valid:u32, pad0:u32, pad1:u32, pad2:u32 }
@group(0) @binding(0) var<uniform> params:Params;@group(0) @binding(1) var<storage,read> headers:array<LeafHeader>;@group(0) @binding(2) var<storage,read> metrics:array<Metric>;
@group(0) @binding(3) var<storage,read> tetraHeaders:array<TetraHeader>;@group(0) @binding(4) var<storage,read> tetrahedra:array<u32>;@group(0) @binding(5) var<storage,read> vertices:array<TetraVertex>;
@group(0) @binding(6) var<storage,read_write> siteIndex:array<SiteIndex>;@group(0) @binding(7) var<storage,read> velocities:array<vec4f>;@group(0) @binding(8) var<storage,read_write> coarse:array<CoarsePhi>;
@group(0) @binding(9) var<storage,read_write> scratchA:array<CoarsePhi>;
@group(0) @binding(11) var<storage,read> fineOffsets:array<u32>;@group(0) @binding(12) var<storage,read> fine:array<FineContribution>;@group(0) @binding(13) var<storage,read_write> control:Control;@group(0) @binding(14) var<storage,read> rowCountSource:array<u32>;
@group(0) @binding(15) var<storage,read_write> sampleDirectory:SampleDirectory;
@group(0) @binding(16) var<storage,read> fineControl:array<u32>;
const INVALID:u32=0xffffffffu;const VALID:u32=0x80000000u;const CAPACITY:u32=1u;const INVALID_ROW:u32=2u;const INVALID_VELOCITY:u32=4u;const INVALID_CATALOG:u32=8u;const INVALID_FINE_OFFSETS:u32=16u;const INVALID_FINE_SAMPLE:u32=32u;const FINE_BOUND:u32=64u;const INVALID_SOURCE:u32=256u;const NO_CAUSAL_SIMPLEX:u32=512u;
const PHI_VALID:u32=${OCTREE_COARSE_PHI_FLAG.valid}u;const PHI_CORRECTED:u32=${OCTREE_COARSE_PHI_FLAG.correctedFromFine}u;const PHI_INTERFACE:u32=${OCTREE_COARSE_PHI_FLAG.containsInterface}u;const PHI_FINITE:u32=${OCTREE_COARSE_PHI_FLAG.finite}u;const UNIFORM:u32=1u;
fn finite(v:f32)->bool{return (bitcast<u32>(v)&0x7f800000u)!=0x7f800000u;}fn sourceRequested()->u32{return select(0u,rowCountSource[0],arrayLength(&rowCountSource)>0u);}fn requested()->u32{return control.rowCount;}fn rejectedFine()->bool{return params.hasFine!=0u&&(arrayLength(&fineControl)<6u||fineControl[0]==INVALID||fineControl[5]!=VALID);}fn dims()->vec3u{return params.dimensionsCapacity.xyz;}
fn coord(cell:u32)->vec3u{let d=dims();return vec3u(cell%d.x,(cell/d.x)%d.y,cell/(d.x*d.y));}fn center(row:u32)->vec3f{return (vec3f(coord(headers[row].cell))+0.5*f32(headers[row].size))*params.physical.x;}fn size(row:u32)->f32{return f32(headers[row].size)*params.physical.x;}
fn inverseTransform(value:vec3f,code:u32)->vec3f{let bits=code&7u;let q=value*vec3f(select(1.0,-1.0,(bits&1u)!=0u),select(1.0,-1.0,(bits&2u)!=0u),select(1.0,-1.0,(bits&4u)!=0u));let p=(code/8u)%6u;if(p==0u){return q.xyz;}if(p==1u){return q.xzy;}if(p==2u){return q.yxz;}if(p==3u){return q.zxy;}if(p==4u){return q.yzx;}return q.zyx;}
fn hashSite(cell:u32,s:u32)->u32{var v=cell^(s*0x9e3779b9u);v=(v^(v>>16u))*0x7feb352du;v=(v^(v>>15u))*0x846ca68bu;return v^(v>>16u);}fn findSite(c:vec3f,s:f32)->u32{let grid=s/params.physical.x;let o=c/params.physical.x-0.5*grid;let rounded=round(o);if(abs(grid-round(grid))>2e-4||any(abs(o-rounded)>vec3f(2e-4))||any(rounded<vec3f(0.0))||any(rounded>=vec3f(dims()))){return INVALID;}let q=vec3u(rounded);let cell=q.x+dims().x*(q.y+dims().y*q.z);let capacity=min(params.hashCapacity,arrayLength(&siteIndex));if(capacity==0u){return INVALID;}let mask=capacity-1u;let base=hashSite(cell,u32(round(grid)))&mask;for(var probe=0u;probe<min(params.maximumHashProbes,capacity);probe+=1u){let slot=(base+probe)&mask;let observed=atomicLoad(&siteIndex[slot].cellPlusOne);if(observed==0u){return INVALID;}if(observed==cell+1u&&siteIndex[slot].size==u32(round(grid))){return siteIndex[slot].row;}}return INVALID;}
fn selectorRow(row:u32,selector:u32)->u32{let metric=metrics[row];let vertex=vertices[selector].offsetSize;let c=center(row)+size(row)*inverseTransform(vertex.xyz,metric.transformAndFlags&63u);return findSite(c,size(row)*vertex.w);}
fn fail(row:u32,flag:u32){atomicOr(&control.flags,flag);atomicMin(&control.firstError,row);}fn solveGradient(m:mat3x3f,b:vec3f)->vec4f{let xx=m[0].x;let xy=m[0].y;let xz=m[0].z;let yy=m[1].y;let yz=m[1].z;let zz=m[2].z;let c00=yy*zz-yz*yz;let c01=xz*yz-xy*zz;let c02=xy*yz-xz*yy;let c11=xx*zz-xz*xz;let c12=xy*xz-xx*yz;let c22=xx*yy-xy*xy;let detValue=xx*c00+xy*c01+xz*c02;if(!finite(detValue)||abs(detValue)<=1e-9){return vec4f(0.0);}return vec4f(vec3f(c00*b.x+c01*b.y+c02*b.z,c01*b.x+c11*b.y+c12*b.z,c02*b.x+c12*b.y+c22*b.z)/detValue,1.0);}
@compute @workgroup_size(1) fn preparePowerCoarsePhi(){if(rejectedFine()){return;}let count=sourceRequested();atomicStore(&control.flags,select(0u,CAPACITY,count>params.dimensionsCapacity.w));atomicStore(&control.firstError,0xffffffffu);control.rowCount=count;atomicStore(&control.advected,0u);atomicStore(&control.uniform,0u);atomicStore(&control.transition,0u);atomicStore(&control.nearest,0u);control.passes=params.redistancePasses;atomicStore(&control.corrected,0u);atomicStore(&control.interfaces,0u);control.contributionCount=params.countsGeneration.y;control.generation=params.countsGeneration.w;control.valid=0u;atomicStore(&sampleDirectory.state,0u);sampleDirectory.generation=params.countsGeneration.w;sampleDirectory.hashCapacity=arrayLength(&sampleDirectory.entries);sampleDirectory.maximumLeafSize=params.maximumLeafSize;sampleDirectory.dimensions=params.dimensionsCapacity.xyz;sampleDirectory.physicalCellSize=params.physical.x;}
@compute @workgroup_size(64) fn clearPowerCoarsePhiSamples(@builtin(global_invocation_id) gid:vec3u){if(rejectedFine()){return;}let slot=gid.x;if(slot>=arrayLength(&sampleDirectory.entries)){return;}atomicStore(&sampleDirectory.entries[slot].cellPlusOne,0u);sampleDirectory.entries[slot].size=0u;sampleDirectory.entries[slot].phi=0.0;sampleDirectory.entries[slot].minimumPhi=0.0;sampleDirectory.entries[slot].maximumPhi=0.0;sampleDirectory.entries[slot].flags=0u;sampleDirectory.entries[slot].row=INVALID;sampleDirectory.entries[slot].physicalVolume=0.0;}
@compute @workgroup_size(64) fn advectPowerCoarsePhi(@builtin(global_invocation_id) gid:vec3u){if(rejectedFine()){return;}let row=gid.x;if(atomicLoad(&control.flags)!=0u||row>=requested()||row>=params.dimensionsCapacity.w){return;}if(row>=arrayLength(&headers)||row>=arrayLength(&metrics)||row>=arrayLength(&velocities)||row>=arrayLength(&coarse)||row>=arrayLength(&scratchA)){fail(row,CAPACITY);return;}let metric=metrics[row];if((metric.transformAndFlags&VALID)==0u){fail(row,INVALID_ROW);return;}let velocity=velocities[row];if(params.physical.y>0.0&&(!finite(velocity.x)||!finite(velocity.y)||!finite(velocity.z)||velocity.w<=0.0)){fail(row,INVALID_VELOCITY);return;}let source=coarse[row];if((source.flags&(PHI_VALID|PHI_FINITE))!=(PHI_VALID|PHI_FINITE)||!finite(source.phi)||!finite(source.minimumPhi)||!finite(source.maximumPhi)||source.minimumPhi>source.maximumPhi||source.phi<source.minimumPhi||source.phi>source.maximumPhi){fail(row,INVALID_SOURCE);return;}var matrix=mat3x3f(vec3f(0.0),vec3f(0.0),vec3f(0.0));var rhs=vec3f(0.0);for(var selector=0u;selector<arrayLength(&vertices);selector+=1u){let neighbor=selectorRow(row,selector);if(neighbor==INVALID||neighbor>=requested()||neighbor>=arrayLength(&headers)||neighbor>=arrayLength(&metrics)||neighbor>=arrayLength(&coarse)){continue;}let delta=center(neighbor)-center(row);let length2=dot(delta,delta);let other=coarse[neighbor];if((other.flags&(PHI_VALID|PHI_FINITE))!=(PHI_VALID|PHI_FINITE)||length2<=1e-12||!finite(other.phi)){continue;}let weight=1.0/length2;matrix+=weight*mat3x3f(delta*delta.x,delta*delta.y,delta*delta.z);rhs+=weight*delta*(other.phi-source.phi);}let gradient=solveGradient(matrix,rhs);var value=source.phi;if(params.physical.y>0.0&&gradient.w>0.0){value-=params.physical.y*dot(velocity.xyz,gradient.xyz);}let shift=value-source.phi;scratchA[row]=CoarsePhi(value,source.minimumPhi+shift,source.maximumPhi+shift,(source.flags&(~PHI_CORRECTED))|PHI_VALID|PHI_FINITE);atomicAdd(&control.advected,1u);}
fn solveTranspose(a:vec3f,b:vec3f,c:vec3f,rhs:vec3f)->vec4f{let d=dot(a,cross(b,c));if(!finite(d)||abs(d)<=1e-10){return vec4f(0.0);}return vec4f((rhs.x*cross(b,c)+rhs.y*cross(c,a)+rhs.z*cross(a,b))/d,1.0);}fn solveColumns(a:vec3f,b:vec3f,c:vec3f,rhs:vec3f)->vec4f{let d=dot(a,cross(b,c));if(!finite(d)||abs(d)<=1e-10){return vec4f(0.0);}return vec4f(dot(rhs,cross(b,c)),dot(a,cross(rhs,c)),dot(a,cross(b,rhs)),d);}fn nonobtuse(a:vec3f,b:vec3f,c:vec3f)->bool{let den=length(a)*length(b)*length(c)+dot(a,b)*length(c)+dot(a,c)*length(b)+dot(b,c)*length(a);let num=abs(dot(a,cross(b,c)));return den+2e-6*max(1.,max(abs(den),num))>=num;}
fn causalTetraCandidate(q:mat3x3f,known:vec3f)->f32{let unavailable=1e30;let av=solveTranspose(q[0],q[1],q[2],known);let bv=solveTranspose(q[0],q[1],q[2],vec3f(1.0));if(av.w==0.0||bv.w==0.0){return unavailable;}let aa=dot(bv.xyz,bv.xyz);let bb=dot(av.xyz,bv.xyz);let cc=dot(av.xyz,av.xyz)-1.0;let disc=bb*bb-aa*cc;if(!finite(aa)||aa<=1e-12||!finite(disc)||disc<0.0){return unavailable;}let candidate=(bb+sqrt(disc))/aa;if(!finite(candidate)||candidate+2e-6<max(known.x,max(known.y,known.z))){return unavailable;}let ray=solveColumns(q[0],q[1],q[2],-(av.xyz-candidate*bv.xyz));if(ray.w==0.0){return unavailable;}let coefficients=ray.xyz/ray.w;let sum=coefficients.x+coefficients.y+coefficients.z;if(!finite(sum)||sum<=2e-6||any(coefficients/sum<vec3f(-2e-6))){return unavailable;}return candidate;}
fn causalTriangleCandidate(a:vec3f,b:vec3f,known:vec2f)->f32{let unavailable=1e30;let g00=dot(a,a);let g01=dot(a,b);let g11=dot(b,b);let determinant=g00*g11-g01*g01;if(!finite(determinant)||determinant<=1e-12||g01+2e-6*max(1.0,sqrt(g00*g11))<0.0){return unavailable;}let av=a*((g11*known.x-g01*known.y)/determinant)+b*((g00*known.y-g01*known.x)/determinant);let bv=a*((g11-g01)/determinant)+b*((g00-g01)/determinant);let aa=dot(bv,bv);let bb=dot(av,bv);let cc=dot(av,av)-1.0;let disc=bb*bb-aa*cc;if(!finite(aa)||aa<=1e-12||!finite(disc)||disc<0.0){return unavailable;}let candidate=(bb+sqrt(disc))/aa;if(!finite(candidate)||candidate+2e-6<max(known.x,known.y)){return unavailable;}let delta=vec2f(candidate)-known;let coefficients=vec2f(g11*delta.x-g01*delta.y,g00*delta.y-g01*delta.x)/determinant;let sum=coefficients.x+coefficients.y;if(!finite(sum)||sum<=2e-6||any(coefficients/sum<vec2f(-2e-6))){return unavailable;}return candidate;}
fn causalEdgeCandidate(offset:vec3f,known:f32)->f32{let distance=length(offset);if(!finite(distance)||distance<=1e-6||!finite(known)){return 1e30;}return known+distance;}
fn eikonal3(values:vec3f,h:f32)->f32{var a=values;if(a.x>a.y){a=vec3f(a.y,a.x,a.z);}if(a.y>a.z){a=vec3f(a.x,a.z,a.y);}if(a.x>a.y){a=vec3f(a.y,a.x,a.z);}var u=a.x+h;if(u>a.y){u=0.5*(a.x+a.y+sqrt(max(0.0,2.0*h*h-(a.x-a.y)*(a.x-a.y))));}if(u>a.z){let disc=3.0*h*h-(a.x-a.y)*(a.x-a.y)-(a.x-a.z)*(a.x-a.z)-(a.y-a.z)*(a.y-a.z);u=(a.x+a.y+a.z+sqrt(max(0.0,disc)))/3.0;}return u;}
@compute @workgroup_size(64) fn redistancePowerCoarsePhi(@builtin(global_invocation_id) gid:vec3u){if(rejectedFine()){return;}let row=gid.x;if(atomicLoad(&control.flags)!=0u||row>=requested()||row>=params.dimensionsCapacity.w){return;}if(row>=arrayLength(&headers)||row>=arrayLength(&metrics)||row>=arrayLength(&scratchA)){fail(row,CAPACITY);return;}let fromA=(params.iteration&1u)==0u;var source=scratchA[params.dimensionsCapacity.w+row];if(fromA){source=scratchA[row];}let metric=metrics[row];if((metric.transformAndFlags&VALID)==0u){fail(row,INVALID_ROW);return;}if(metric.topologyCode>=arrayLength(&tetraHeaders)){fail(row,INVALID_CATALOG);return;}let header=tetraHeaders[metric.topologyCode];var magnitude=1e30;var used=false;if((header.flags&UNIFORM)!=0u){var axes=vec3f(1e30);for(var selector=0u;selector<arrayLength(&vertices);selector+=1u){let v=vertices[selector].offsetSize;if(abs(v.w-1.0)>1e-5){continue;}let world=inverseTransform(v.xyz,metric.transformAndFlags&63u);if(abs(length(world)-1.0)>1e-5){continue;}let neighbor=selectorRow(row,selector);if(neighbor==INVALID||neighbor>=requested()){continue;}let phi=abs(select(scratchA[neighbor].phi,scratchA[params.dimensionsCapacity.w+neighbor].phi,!fromA));let axis=select(select(2u,1u,abs(world.y)>.5),0u,abs(world.x)>.5);axes[axis]=min(axes[axis],phi);}if(all(axes<vec3f(1e29))){magnitude=eikonal3(axes,size(row));used=true;}atomicAdd(&control.uniform,1u);}else{if(header.first>arrayLength(&tetrahedra)||header.count>arrayLength(&tetrahedra)-header.first){fail(row,INVALID_CATALOG);return;}for(var local=0u;local<header.count;local+=1u){let packed=tetrahedra[header.first+local];let s=vec3u(packed&255u,(packed>>8u)&255u,(packed>>16u)&255u);if(any(s>=vec3u(arrayLength(&vertices)))){fail(row,INVALID_CATALOG);return;}let rows=vec3u(selectorRow(row,s.x),selectorRow(row,s.y),selectorRow(row,s.z));if(any(rows==vec3u(INVALID))||any(rows>=vec3u(requested()))){continue;}let q=mat3x3f(size(row)*inverseTransform(vertices[s.x].offsetSize.xyz,metric.transformAndFlags&63u),size(row)*inverseTransform(vertices[s.y].offsetSize.xyz,metric.transformAndFlags&63u),size(row)*inverseTransform(vertices[s.z].offsetSize.xyz,metric.transformAndFlags&63u));if(!nonobtuse(q[0],q[1],q[2])){continue;}let known=vec3f(abs(select(scratchA[rows.x].phi,scratchA[params.dimensionsCapacity.w+rows.x].phi,!fromA)),abs(select(scratchA[rows.y].phi,scratchA[params.dimensionsCapacity.w+rows.y].phi,!fromA)),abs(select(scratchA[rows.z].phi,scratchA[params.dimensionsCapacity.w+rows.z].phi,!fromA)));let full=causalTetraCandidate(q,known);let face0=causalTriangleCandidate(q[0],q[1],known.xy);let face1=causalTriangleCandidate(q[0],q[2],known.xz);let face2=causalTriangleCandidate(q[1],q[2],known.yz);let edge0=causalEdgeCandidate(q[0],known.x);let edge1=causalEdgeCandidate(q[1],known.y);let edge2=causalEdgeCandidate(q[2],known.z);let candidate=min(min(full,min(face0,min(face1,face2))),min(edge0,min(edge1,edge2)));if(candidate<1e29){magnitude=min(magnitude,candidate);used=true;}}atomicAdd(&control.transition,1u);}if(!used){fail(row,NO_CAUSAL_SIMPLEX);return;}let sign=select(1.0,-1.0,source.phi<0.0);let value=sign*min(abs(source.phi),magnitude);let output=CoarsePhi(value,value,value,(source.flags&(~PHI_INTERFACE))|PHI_VALID|PHI_FINITE);if(fromA){scratchA[params.dimensionsCapacity.w+row]=output;}else{scratchA[row]=output;}}
@compute @workgroup_size(64) fn validatePowerCoarseFineCorrection(@builtin(global_invocation_id) gid:vec3u){if(rejectedFine()){return;}let row=gid.x;if(params.hasFine==0u||row>=requested()){return;}if(row+1u>=arrayLength(&fineOffsets)){fail(row,INVALID_FINE_OFFSETS);return;}let begin=fineOffsets[row];let end=fineOffsets[row+1u];if((row==0u&&begin!=0u)||(row+1u==requested()&&end!=params.countsGeneration.y)||end<begin||end>params.countsGeneration.y||end>arrayLength(&fine)){fail(row,INVALID_FINE_OFFSETS);return;}if(end-begin>params.countsGeneration.z){fail(row,FINE_BOUND);return;}for(var cursor=begin;cursor<end;cursor+=1u){let sample=fine[cursor];if(params.hasFine==2u){let maximum=bitcast<f32>(sample.valid);if(sample.pad!=0u&&(!finite(sample.phi)||!finite(sample.distanceSquared)||!finite(maximum)||sample.distanceSquared>maximum)){fail(row,INVALID_FINE_SAMPLE);return;}}else if(sample.valid!=0u&&(!finite(sample.phi)||!finite(sample.distanceSquared)||sample.distanceSquared<0.0)){fail(row,INVALID_FINE_SAMPLE);return;}}}
fn publishSample(row:u32,value:CoarsePhi){let header=headers[row];let metric=metrics[row];let extent=f32(header.size)*params.physical.x;let physicalVolume=metric.volume*extent*extent*extent;if(!finite(physicalVolume)||physicalVolume<=0.0){fail(row,INVALID_ROW);return;}let capacity=arrayLength(&sampleDirectory.entries);if(capacity==0u||(capacity&(capacity-1u))!=0u){fail(row,128u);return;}let base=hashSite(header.cell,header.size)&(capacity-1u);for(var probe=0u;probe<min(params.maximumHashProbes,capacity);probe+=1u){let slot=(base+probe)&(capacity-1u);let result=atomicCompareExchangeWeak(&sampleDirectory.entries[slot].cellPlusOne,0u,header.cell+1u);if(result.exchanged){sampleDirectory.entries[slot].size=header.size;sampleDirectory.entries[slot].phi=value.phi;sampleDirectory.entries[slot].minimumPhi=value.minimumPhi;sampleDirectory.entries[slot].maximumPhi=value.maximumPhi;sampleDirectory.entries[slot].flags=value.flags;sampleDirectory.entries[slot].row=row;sampleDirectory.entries[slot].physicalVolume=physicalVolume;return;}if(result.old_value==0u){probe-=select(0u,1u,probe>0u);continue;}if(result.old_value==header.cell+1u&&sampleDirectory.entries[slot].size==header.size){fail(row,128u);return;}}fail(row,128u);}
@compute @workgroup_size(64) fn publishPowerCoarsePhi(@builtin(global_invocation_id) gid:vec3u){if(rejectedFine()){return;}let row=gid.x;if(row>=requested()||atomicLoad(&control.flags)!=0u){return;}var output=scratchA[row];if((params.redistancePasses&1u)==1u){output=scratchA[params.dimensionsCapacity.w+row];}if(params.hasFine!=0u){let begin=fineOffsets[row];let end=fineOffsets[row+1u];if(params.hasFine==2u){if(end>begin){let aggregate=fine[begin];if(aggregate.pad!=0u){output.phi=aggregate.phi;output.minimumPhi=aggregate.distanceSquared;output.maximumPhi=bitcast<f32>(aggregate.valid);output.flags|=PHI_CORRECTED;atomicAdd(&control.corrected,1u);}}}else{var nearest=1e30;var minimum=1e30;var maximum=-1e30;var count=0u;for(var cursor=begin;cursor<end;cursor+=1u){let sample=fine[cursor];if(sample.valid==0u){continue;}minimum=min(minimum,sample.phi);maximum=max(maximum,sample.phi);if(sample.distanceSquared<nearest||(sample.distanceSquared==nearest&&sample.phi<output.phi)){nearest=sample.distanceSquared;output.phi=sample.phi;}count+=1u;}if(count>0u){output.minimumPhi=minimum;output.maximumPhi=maximum;output.flags|=PHI_CORRECTED;atomicAdd(&control.corrected,1u);}}}if(output.minimumPhi<=0.0&&output.maximumPhi>=0.0){output.flags|=PHI_INTERFACE;atomicAdd(&control.interfaces,1u);}coarse[row]=output;publishSample(row,output);}
@compute @workgroup_size(1) fn finalizePowerCoarsePhi(){if(rejectedFine()){return;}let complete=control.rowCount>0u&&control.rowCount<=params.dimensionsCapacity.w&&atomicLoad(&control.advected)==control.rowCount;if(atomicLoad(&control.flags)==0u&&complete){control.valid=VALID;atomicStore(&sampleDirectory.state,VALID);}else{control.valid=0u;atomicStore(&sampleDirectory.state,0u);}}
`;
