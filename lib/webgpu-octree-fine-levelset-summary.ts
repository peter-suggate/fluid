import type { FineLevelSetBrickPlan } from "./octree-fine-levelset-bricks";
import { fineLevelSetLinearWorkgroupWGSL, planFineLevelSetDispatch2D } from "./webgpu-fine-levelset-dispatch";
import type { WebGPUFineLevelSetBrickSource } from "./webgpu-octree-fine-levelset-bricks";
import type { FineLevelSetPageDeltaLayout } from "./webgpu-octree-fine-levelset-topology";
import { PassBroker } from "./webgpu-pass-broker";

export const FINE_LEVELSET_SUMMARY_VALID = 0x8000_0000;
export const FINE_LEVELSET_SUMMARY_COARSE_AUTHORITY = 0x8000_0000;
/** The eight fine samples required to interpolate the finest coarse-cell
 * centre were all present. This is evidence, not directory authority. */
export const FINE_LEVELSET_SUMMARY_CENTER_COMPLETE = 0x3fc0_0000;
export const FINE_LEVELSET_SUMMARY_ERROR = Object.freeze({ capacity: 1, hashProbe: 2, staleGeneration: 4,
  nonfinite: 8 } as const);

export interface FineLevelSetGPUSummaryPlan {
  readonly maximumResidentBricks: number; readonly maximumLevel: number;
  readonly fineEntryCapacity: number; readonly coarseEntryCapacity: number; readonly entryCapacity: number;
  readonly hierarchyKeyCapacity: number; readonly recordCapacity: number; readonly sortCapacity: number;
  readonly directoryBytes: number; readonly recordBytes: number;
  readonly carryBytes: number; readonly workStateBytes: number; readonly indirectBytes: number;
  readonly parameterBytes: number; readonly allocatedBytes: number;
  readonly levelOffsets: readonly number[]; readonly levelDimensions: readonly (readonly [number, number, number])[];
}

/** Exact topology transaction consumed by the recurring summary update. */
export interface FineLevelSetSummaryPageDeltaSource {
  readonly buffer: GPUBuffer;
  readonly layout: Pick<FineLevelSetPageDeltaLayout, "changedKeysOffsetWords">;
}

export interface FineLevelSetSummaryCoarseSource {
  readonly directory: GPUBuffer;
  readonly control: GPUBuffer;
  readonly delta: GPUBuffer;
  readonly deltaHeaderWords: 16;
  readonly deltaRecordWords: 4;
}

export interface FineLevelSetSummaryLeafLookup {
  readonly level: number;
  readonly key: number;
  readonly brickSide: number;
  readonly expectedBrickCount: number;
  readonly expectedSampleCount: number;
}

/**
 * Map one aligned octree leaf to the single dyadic fine-brick summary node
 * that covers it.  Factor four has one brick per finest octree cell and
 * factor eight has two; consequently every power-of-two octree leaf maps to
 * exactly one node rather than an O(factor^3) sample scan.
 */
export function planFineLevelSetSummaryLeafLookup(
  baseDimensions: readonly [number, number, number],
  finestCellDimensions: readonly [number, number, number],
  origin: readonly [number, number, number],
  size: number,
  samplesPerBrick = 64,
): FineLevelSetSummaryLeafLookup {
  if (!Number.isInteger(size) || size < 1 || (size & (size - 1)) !== 0) {
    throw new RangeError("Fine-summary leaf size must be a positive power of two");
  }
  const ratios = baseDimensions.map((value, axis) => value / finestCellDimensions[axis]);
  const bricksPerCell = ratios[0];
  if (!Number.isInteger(bricksPerCell) || bricksPerCell < 1 || ratios.some((value) => value !== bricksPerCell)) {
    throw new RangeError("Fine-summary lattice must contain an equal integer brick count per finest cell");
  }
  const brickSide = size * bricksPerCell;
  if (!Number.isSafeInteger(brickSide) || (brickSide & (brickSide - 1)) !== 0) {
    throw new RangeError("Fine-summary leaf brick span must be a safe power of two");
  }
  const level = Math.log2(brickSide);
  let levelOffset = 0;
  let dimensions = [...baseDimensions] as [number, number, number];
  for (let current = 0; current < level; current += 1) {
    levelOffset += dimensions[0] * dimensions[1] * dimensions[2];
    dimensions = dimensions.map((value) => Math.ceil(value / 2)) as [number, number, number];
  }
  const coordinate = origin.map((value) => (value * bricksPerCell) / brickSide);
  if (coordinate.some((value) => !Number.isInteger(value))) {
    throw new RangeError("Octree leaf origin is not aligned to its fine-summary node");
  }
  const key = levelOffset + coordinate[0] + dimensions[0]
    * (coordinate[1] + dimensions[1] * coordinate[2]);
  const expectedBrickCount = brickSide ** 3;
  return { level, key, brickSide, expectedBrickCount,
    expectedSampleCount: expectedBrickCount * samplesPerBrick };
}

export type FineLevelSetSummaryRefinementSignal = "refine" | "complete-no-crossing" | "fallback";

/** CPU mirror of the shader's fail-closed summary interpretation. */
export function fineLevelSetSummaryRefinementSignal(summary: {
  readonly published: boolean; readonly directoryFlags: number; readonly found: boolean;
  readonly entryFlags: number; readonly minimumPhi: number; readonly maximumPhi: number;
  readonly minimumAbsolutePhi: number; readonly brickCount: number; readonly sampleCount: number;
}, lookup: Pick<FineLevelSetSummaryLeafLookup, "expectedBrickCount" | "expectedSampleCount">,
bandWidth: number): FineLevelSetSummaryRefinementSignal {
  const coarseAuthority = (summary.entryFlags >>> 31) !== 0;
  if (!summary.published || summary.directoryFlags !== 0 || !summary.found
    || (summary.entryFlags & 0x003f_ffff) !== 0
    || !Number.isFinite(summary.minimumPhi) || !Number.isFinite(summary.maximumPhi)
    || !Number.isFinite(summary.minimumAbsolutePhi)) return "fallback";
  // A crossing observed in even a partial sparse node is sufficient evidence
  // to refine.  Incompleteness is only allowed to remove evidence, never to
  // prove that a leaf is safe to coarsen.
  if (summary.minimumPhi < 0 && summary.maximumPhi >= 0) return "refine";
  if (!coarseAuthority && (summary.brickCount !== lookup.expectedBrickCount
    || summary.sampleCount !== lookup.expectedSampleCount)) {
    return "fallback";
  }
  return summary.minimumAbsolutePhi < bandWidth ? "refine" : "complete-no-crossing";
}

export function planFineLevelSetGPUSummaries(
  plan: FineLevelSetBrickPlan,
  coarseEntryCapacity = 0,
): FineLevelSetGPUSummaryPlan {
  if (!Number.isSafeInteger(coarseEntryCapacity) || coarseEntryCapacity < 0) {
    throw new RangeError("Fine summary coarse-entry capacity must be a non-negative integer");
  }
  const levelOffsets: number[] = []; const levelDimensions: Array<readonly [number, number, number]> = [];
  let dimensions = [...plan.brickDimensions] as [number, number, number]; let hierarchyKeyCapacity = 0;
  let fineEntryCapacity = 0;
  for (;;) {
    const levelKeyCount = dimensions[0] * dimensions[1] * dimensions[2];
    levelOffsets.push(hierarchyKeyCapacity); levelDimensions.push(dimensions);
    hierarchyKeyCapacity += levelKeyCount;
    // A resident base brick contributes at most one distinct key at each
    // ancestor level. Parent sharing and the finite level lattice can only
    // reduce that count. This is the exact sparse upper bound; multiplying
    // resident capacity by the number of levels needlessly charged the base
    // capacity again at every progressively smaller level.
    fineEntryCapacity += Math.min(plan.maximumResidentBricks, levelKeyCount);
    if (dimensions.every((value) => value === 1)) break;
    dimensions = dimensions.map((value) => Math.ceil(value / 2)) as [number, number, number];
  }
  if (!Number.isSafeInteger(hierarchyKeyCapacity) || hierarchyKeyCapacity >= 0xffff_ffff) {
    throw new RangeError("Fine summary hierarchy keys exceed the u32 ABI");
  }
  // Coarse-octree summaries may add at most one hierarchy key per compact
  // row. Ignore overlap for a conservative bound, then clip to the number of
  // representable hierarchy keys. The resulting directory remains
  // O(resident fine surface bricks + compact octree rows), not O(fine-domain
  // volume), while preserving the <= 0.5 load required by bounded probing.
  const entryCapacity = Math.min(hierarchyKeyCapacity, fineEntryCapacity + coarseEntryCapacity);
  // A recurring page transaction can contain one changed record for every
  // desired page and one retirement for every old page. Each changed base key
  // dirties exactly one ancestor at every level. Corrected-coarse rows are
  // exact dirty contributions from the compact value/phase row delta.
  const recordCapacity = 2 * plan.maximumResidentBricks * levelOffsets.length + 2 * coarseEntryCapacity;
  let sortCapacity = 256;
  while (sortCapacity < recordCapacity) sortCapacity *= 2;
  const directoryBytes = 64 + Math.max(1, entryCapacity) * 32;
  const recordBytes = sortCapacity * 32;
  const carryBytes = Math.max(1, entryCapacity) * 32;
  const workStateBytes = 128;
  // The persistent transaction builder publishes one exact recompute extent.
  // The former fine/coarse emitter extents were consumed only by their own
  // recurring dispatches and are deliberately absent.
  const indirectBytes = 12;
  const parameterBytes = levelOffsets.length * 112 + 4 * 16;
  return { maximumResidentBricks: plan.maximumResidentBricks, maximumLevel: levelOffsets.length - 1,
    fineEntryCapacity, coarseEntryCapacity, entryCapacity,
    hierarchyKeyCapacity, recordCapacity, sortCapacity,
    directoryBytes, recordBytes, carryBytes, workStateBytes, indirectBytes, parameterBytes,
    allocatedBytes: 4 * directoryBytes + 2 * recordBytes + carryBytes + workStateBytes
      + indirectBytes
      + parameterBytes, levelOffsets, levelDimensions };
}

/** Sparse immutable hierarchy updated only from the exact topology delta. */
export class WebGPUFineLevelSetSummaries {
  readonly plan: FineLevelSetGPUSummaryPlan;
  /** Stable committed publication consumed by pressure topology. */
  readonly directory: GPUBuffer;
  private readonly candidateDirectory: GPUBuffer;
  private readonly fineDirectory: GPUBuffer;
  private readonly fineCandidateDirectory: GPUBuffer;
  private readonly records: GPUBuffer;
  private readonly recordScratch: GPUBuffer;
  private readonly carryRecords: GPUBuffer;
  private readonly workState: GPUBuffer;
  private readonly indirect: GPUBuffer;
  private readonly params: readonly GPUBuffer[];
  private readonly pipelines: Readonly<Record<string, GPUComputePipeline>>;
  private destroyed = false;

  /** Rejection-only bounded buffers. They never participate in scheduling or
   * publication and exist only to reproduce a UI-visible failure in Dawn. */
  get diagnosticBuffers() {
    return {
      candidateDirectory: this.candidateDirectory,
      fineDirectory: this.fineDirectory,
      fineCandidateDirectory: this.fineCandidateDirectory,
      workState: this.workState,
      records: this.records,
      recordScratch: this.recordScratch,
    };
  }

  constructor(private readonly device: GPUDevice, readonly finePlan: FineLevelSetBrickPlan,
    coarseEntryCapacity = 0) {
    this.plan = planFineLevelSetGPUSummaries(finePlan, coarseEntryCapacity);
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    this.directory = device.createBuffer({ label: "global fine committed sparse summary hierarchy",
      size: this.plan.directoryBytes, usage: storage });
    this.candidateDirectory = device.createBuffer({ label: "global fine candidate sparse summary hierarchy",
      size: this.plan.directoryBytes, usage: storage });
    this.fineDirectory = device.createBuffer({ label: "global fine committed fine-only summary hierarchy",
      size: this.plan.directoryBytes, usage: storage });
    this.fineCandidateDirectory = device.createBuffer({ label: "global fine candidate fine-only summary hierarchy",
      size: this.plan.directoryBytes, usage: storage });
    this.records = device.createBuffer({ label: "global fine dirty summary records",
      size: this.plan.recordBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    this.recordScratch = device.createBuffer({ label: "global fine dirty summary compact scratch",
      size: this.plan.recordBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    this.carryRecords = device.createBuffer({ label: "global fine byte-exact carried summary records",
      size: this.plan.carryBytes, usage: GPUBufferUsage.STORAGE });
    this.workState = device.createBuffer({ label: "global fine delta publication state",
      size: this.plan.workStateBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    this.indirect = device.createBuffer({ label: "global fine immutable delta dispatches",
      size: this.plan.indirectBytes, usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST });
    this.params = this.plan.levelOffsets.map((_, level) => device.createBuffer({
      label: `global fine delta summary parameters level ${level}`, size: 112,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    }));
    const module = device.createShaderModule({ label: "global fine delta summary hierarchy",
      code: fineLevelSetSummaryWGSL });
    const pipeline = (entryPoint: string) => device.createComputePipeline({
      label: entryPoint, layout: "auto", compute: { module, entryPoint },
    });
    this.pipelines = Object.freeze(Object.fromEntries([
      "buildFineSummaryDelta",
      "recomputeFineSummaryBase", "recomputeFineSummaryParents",
      "publishFineSummaryDelta",
    ].map((entryPoint) => [entryPoint, pipeline(entryPoint)])));
  }

  encode(broker: PassBroker, source: WebGPUFineLevelSetBrickSource,
    pageDelta: FineLevelSetSummaryPageDeltaSource,
    coarse: FineLevelSetSummaryCoarseSource): void {
    if (this.destroyed) throw new Error("Fine summary hierarchy is destroyed");
    if (source.plan !== this.finePlan && JSON.stringify(source.plan) !== JSON.stringify(this.finePlan)) {
      throw new RangeError("Fine summary source does not match its configured lattice");
    }
    const changedKeysOffsetWords = pageDelta.layout.changedKeysOffsetWords;
    for (let level = 0; level < this.params.length; level += 1) {
      const data = new Uint32Array(28); const dims = this.plan.levelDimensions[level];
      data.set(this.finePlan.brickDimensions, 0); data[3] = this.finePlan.samplesPerBrick;
      data.set([this.finePlan.maximumResidentBricks, this.plan.entryCapacity,
        this.plan.sortCapacity, source.generation], 4);
      data.set([level, this.plan.levelOffsets[level], dims[0] * dims[1] * dims[2]], 8);
      data.set(dims, 12); data[15] = this.plan.maximumLevel; data[16] = this.plan.hierarchyKeyCapacity;
      data[17] = 5; data[18] = this.plan.coarseEntryCapacity; data[19] = this.plan.recordCapacity;
      data.set(this.finePlan.finestCellDimensions, 20); data[23] = changedKeysOffsetWords;
      data[24] = this.device.limits.maxComputeWorkgroupsPerDimension;
      this.device.queue.writeBuffer(this.params[level], 0, data);
    }
    const delta = pageDelta.buffer;
    const coarseDirectory = coarse.directory;
    const coarseControl = coarse.control;
    if (coarse.deltaHeaderWords !== 16 || coarse.deltaRecordWords !== 4) {
      throw new RangeError("Fine summary coarse-delta ABI is invalid");
    }
    const coarseDelta = coarse.delta;
    const bind = (name: string, params: GPUBuffer | undefined,
      buffers: readonly (readonly [number, GPUBuffer])[]) => this.device.createBindGroup({
      layout: this.pipelines[name].getBindGroupLayout(0), entries: [
        ...(params ? [{ binding: 0, resource: { buffer: params } }] : []),
        ...buffers.map(([binding, buffer]) => ({ binding, resource: { buffer } })),
      ],
    });
    const run = (name: string, group: GPUBindGroup, groups: number, label: string) => {
      const pass = broker.compute({ label }); pass.setPipeline(this.pipelines[name]); pass.setBindGroup(0, group);
      const dispatch = planFineLevelSetDispatch2D(Math.max(1, groups),
        this.device.limits.maxComputeWorkgroupsPerDimension);
      pass.dispatchWorkgroups(dispatch.x, dispatch.y);
    };
    const runIndirect = (name: string, group: GPUBindGroup, offset: number, label: string) => {
      const pass = broker.compute({ label }); pass.setPipeline(this.pipelines[name]); pass.setBindGroup(0, group);
      pass.dispatchWorkgroupsIndirect(this.indirect, offset);
    };
    run("buildFineSummaryDelta", bind("buildFineSummaryDelta", this.params[0], [
      [2, source.worklist], [5, this.directory], [6, coarseDirectory], [12, this.workState],
      [1, source.metadata], [8, this.records], [14, coarseDelta], [16, coarseControl],
      [17, delta], [18, this.candidateDirectory],
    ]), 1, "Build and sort exact global fine summary delta");
    broker.updateIndirectBuffer(this.workState, 16 * 4, this.indirect, 0, this.plan.indirectBytes);
    const summaryInputs = [[1, source.metadata], [2, source.worklist], [3, source.flags],
      [4, source.phi], [8, this.records], [12, this.workState]] as const;
    runIndirect("recomputeFineSummaryBase", bind("recomputeFineSummaryBase", this.params[0], summaryInputs),
      0, "Recompute exact dirty fine-summary bases");
    for (let level = 1; level <= this.plan.maximumLevel; level += 1) {
      runIndirect("recomputeFineSummaryParents",
        bind("recomputeFineSummaryParents", this.params[level],
          [...summaryInputs, [20, this.fineDirectory]]),
        0, `Recompute exact dirty fine-summary parents level ${level}`);
    }
    run("publishFineSummaryDelta", bind("publishFineSummaryDelta", this.params[0],
      [[5, this.directory], [6, coarseDirectory], [8, this.records], [10, this.recordScratch],
        [12, this.workState], [18, this.candidateDirectory], [19, this.carryRecords],
        [20, this.fineDirectory], [21, this.fineCandidateDirectory]]),
    1, "Publish immutable fine-only and corrected-coarse summary delta");
  }

  destroy(): void {
    if (this.destroyed) return; this.destroyed = true;
    this.directory.destroy(); this.candidateDirectory.destroy(); this.fineDirectory.destroy();
    this.fineCandidateDirectory.destroy(); this.records.destroy();
    this.recordScratch.destroy(); this.carryRecords.destroy();
    this.workState.destroy(); this.indirect.destroy();
    this.params.forEach((buffer) => buffer.destroy());
  }
}

/**
 * Each sparse entry is eight words: key, ordered min/max phi, min |phi|,
 * exact sample/brick counts, flags, then the geometric-centre phi. Flags reserve
 * bit 31 for corrected-coarse authority and low bits for fail-closed errors.
 * The final word is written after deterministic compaction with the eight-sample
 * trilinear value at each dyadic node's own geometric centre. This lets every
 * newly rebuilt pressure leaf consume the current fine level set directly.
 */
export const fineLevelSetSummaryWGSL = /* wgsl */ `
${fineLevelSetLinearWorkgroupWGSL}
const INVALID:u32=0xffffffffu;const VALID:u32=1u;const PAGE_DELTA_VALID:u32=1u;const PUBLISHED:u32=0x80000000u;
const CAPACITY:u32=1u;const STALE:u32=4u;const NONFINITE:u32=8u;
const COARSE_AUTHORITY:u32=0x80000000u;const COARSE_DIRTY:u32=0x40000000u;const CENTER_COMPLETE:u32=0x3fc00000u;
struct P{baseDims:vec3u,samplesPerBrick:u32,pageCapacity:u32,entryCapacity:u32,sortCapacity:u32,generation:u32,
 level:u32,levelOffset:u32,levelKeyCount:u32,levelDims:vec3u,maximumLevel:u32,hierarchyKeyCapacity:u32,
 worklistHeaderWords:u32,coarseEntryCapacity:u32,recordCapacity:u32,finestDims:vec3u,changedKeysOffset:u32,
 pad0:u32,pad1:u32,pad2:u32,pad3:u32}
struct Entry{key:u32,minimumPhi:u32,maximumPhi:u32,minimumAbsolutePhi:u32,samples:u32,bricks:u32,flags:u32,centerPhi:u32}
@group(0)@binding(0)var<uniform>p:P;@group(0)@binding(1)var<storage,read>a:array<u32>;
@group(0)@binding(2)var<storage,read>b:array<u32>;@group(0)@binding(3)var<storage,read>c:array<u32>;
@group(0)@binding(4)var<storage,read>d:array<u32>;@group(0)@binding(5)var<storage,read_write>directory:array<u32>;
struct CoarseEntry{cellPlusOne:u32,size:u32,phi:f32,minimumPhi:f32,maximumPhi:f32,flags:u32,row:u32,physicalVolume:f32}
struct CoarseDirectory{state:u32,generation:u32,rowCount:u32,maximumLeafSize:u32,dimensions:vec3u,physicalCellSize:f32,entries:array<CoarseEntry>}
@group(0)@binding(6)var<storage,read>coarse:CoarseDirectory;
@group(0)@binding(8)var<storage,read_write>records:array<Entry>;
@group(0)@binding(10)var<storage,read_write>sortScratch:array<Entry>;
@group(0)@binding(12)var<storage,read_write>workState:array<u32>;
struct CoarseDeltaRecord{cellPlusOne:u32,size:u32,row:u32,flags:u32}
struct CoarseDelta{count:u32,generation:u32,flags:u32,valid:u32,pad:array<u32,12>,items:array<CoarseDeltaRecord>}
@group(0)@binding(14)var<storage,read>coarseDelta:CoarseDelta;
@group(0)@binding(16)var<storage,read>coarseControl:array<u32>;
@group(0)@binding(17)var<storage,read>pageDelta:array<u32>;
@group(0)@binding(18)var<storage,read_write>candidate:array<u32>;
@group(0)@binding(19)var<storage,read_write>carryRecords:array<Entry>;
@group(0)@binding(20)var<storage,read_write>fineDirectory:array<u32>;
@group(0)@binding(21)var<storage,read_write>fineCandidate:array<u32>;
var<workgroup>minimumPhi:array<f32,64>;var<workgroup>maximumPhi:array<f32,64>;
var<workgroup>minimumAbsolutePhi:array<f32,64>;var<workgroup>validSamples:array<u32,64>;var<workgroup>errors:array<u32,64>;
var<workgroup>reductionValues:array<u32,256>;
var<workgroup>summaryBuildState:array<u32,5>;
fn finite(v:f32)->bool{return v==v&&abs(v)<3.402823e38;}
fn ordered(v:f32)->u32{let bits=bitcast<u32>(v);return select(bits^0x80000000u,~bits,(bits&0x80000000u)!=0u);}
fn validFineWorklist()->bool{return p.worklistHeaderWords==5u&&arrayLength(&b)>=5u&&b[1]==p.generation&&b[3]==1u&&b[4]==1u&&b[0]<=p.pageCapacity&&5u+b[0]<=arrayLength(&b);}
fn finePage(key:u32)->u32{if(!validFineWorklist()){return INVALID;}let count=b[0];var low=0u;var high=count;for(var step=0u;step<32u&&low<high;step+=1u){let middle=low+(high-low)/2u;let id=b[5u+middle];let base=id*10u;if(id>=p.pageCapacity||base+2u>=arrayLength(&a)||a[base]!=id||a[base+2u]!=p.generation){return INVALID;}if(a[base+1u]<key){low=middle+1u;}else{high=middle;}}if(low>=count){return INVALID;}let id=b[5u+low];let base=id*10u;return select(INVALID,id,id<p.pageCapacity&&base+2u<arrayLength(&a)&&a[base]==id&&a[base+1u]==key&&a[base+2u]==p.generation);}
fn invalidEntry()->Entry{return Entry(INVALID,0xffffffffu,0u,bitcast<u32>(3.402823e38),0u,0u,0u,0u);}
fn entryBase(index:u32)->u32{return 16u+index*8u;}
fn writeDirectory(index:u32,value:Entry){let base=entryBase(index);directory[base]=value.key;directory[base+1u]=value.minimumPhi;directory[base+2u]=value.maximumPhi;directory[base+3u]=value.minimumAbsolutePhi;directory[base+4u]=value.samples;directory[base+5u]=value.bricks;directory[base+6u]=value.flags;directory[base+7u]=value.centerPhi;}
fn readDirectory(index:u32)->Entry{let base=entryBase(index);return Entry(directory[base],directory[base+1u],directory[base+2u],directory[base+3u],directory[base+4u],directory[base+5u],directory[base+6u],directory[base+7u]);}
fn writeCandidate(index:u32,value:Entry){let base=16u+index*8u;candidate[base]=value.key;candidate[base+1u]=value.minimumPhi;candidate[base+2u]=value.maximumPhi;candidate[base+3u]=value.minimumAbsolutePhi;candidate[base+4u]=value.samples;candidate[base+5u]=value.bricks;candidate[base+6u]=value.flags;candidate[base+7u]=value.centerPhi;}
fn fineEntryBase(index:u32)->u32{return 16u+index*8u;}
fn readFineDirectory(index:u32)->Entry{let base=fineEntryBase(index);return Entry(fineDirectory[base],fineDirectory[base+1u],fineDirectory[base+2u],fineDirectory[base+3u],fineDirectory[base+4u],fineDirectory[base+5u],fineDirectory[base+6u],fineDirectory[base+7u]);}
fn writeFineCandidate(index:u32,value:Entry){let base=fineEntryBase(index);fineCandidate[base]=value.key;fineCandidate[base+1u]=value.minimumPhi;fineCandidate[base+2u]=value.maximumPhi;fineCandidate[base+3u]=value.minimumAbsolutePhi;fineCandidate[base+4u]=value.samples;fineCandidate[base+5u]=value.bricks;fineCandidate[base+6u]=value.flags;fineCandidate[base+7u]=value.centerPhi;}
fn publishRecomputeDispatch(groups:u32){let width=min(groups,p.pad0);workState[16]=width;workState[17]=select(1u,(groups+width-1u)/width,width!=0u);workState[18]=1u;}
fn coarseAuthoritative()->bool{let current=p.generation&0x3fffffffu;return p.coarseEntryCapacity==0u||(arrayLength(&coarseControl)>=12u&&coarse.state==PUBLISHED&&(coarse.generation&0x3fffffffu)==current&&coarse.rowCount==coarseControl[2]&&coarse.rowCount<=arrayLength(&coarse.entries)&&all(coarse.dimensions==p.finestDims)&&coarseControl[0]==0u&&(coarseControl[10]&0x3fffffffu)==current&&coarseControl[11]==PUBLISHED&&coarseDelta.flags==0u&&coarseDelta.valid==PUBLISHED&&(coarseDelta.generation&0x3fffffffu)==current&&coarseDelta.count<=arrayLength(&coarseDelta.items)&&coarseDelta.count<=2u*p.coarseEntryCapacity);}
fn hierarchyKey(baseKey:u32,targetLevel:u32)->u32{let xy=p.baseDims.x*p.baseDims.y;let z=baseKey/xy;let rem=baseKey-z*xy;let y=rem/p.baseDims.x;var coord=vec3u(rem-y*p.baseDims.x,y,z);var dims=p.baseDims;var offset=0u;for(var level=0u;level<targetLevel;level+=1u){offset+=dims.x*dims.y*dims.z;dims=(dims+vec3u(1u))/2u;coord/=2u;}return offset+coord.x+dims.x*(coord.y+dims.y*coord.z);}
fn coarseDirtyIdentity(row:u32)->vec2u{if(workState[0]==1u){if(row>=coarse.rowCount||row>=arrayLength(&coarse.entries)){return vec2u(0u);}let e=coarse.entries[row];return vec2u(e.cellPlusOne,e.size);}if(row>=coarseDelta.count||row>=arrayLength(&coarseDelta.items)){return vec2u(0u);}let e=coarseDelta.items[row];return vec2u(e.cellPlusOne,e.size);}
fn coarseHierarchyKey(identity:vec2u)->u32{let ratio=p.baseDims/p.finestDims;if(identity.x==0u||identity.y==0u||(identity.y&(identity.y-1u))!=0u||ratio.x==0u||any(ratio!=vec3u(ratio.x))){return INVALID;}let cell=identity.x-1u;if(cell>=p.finestDims.x*p.finestDims.y*p.finestDims.z){return INVALID;}let side=identity.y*ratio.x;if(side==0u||(side&(side-1u))!=0u){return INVALID;}let origin=vec3u(cell%p.finestDims.x,(cell/p.finestDims.x)%p.finestDims.y,cell/(p.finestDims.x*p.finestDims.y));let brickOrigin=origin*ratio.x;if(any(brickOrigin%vec3u(side)!=vec3u(0u))){return INVALID;}var level=0u;var remaining=side;loop{if(remaining==1u){break;}level+=1u;remaining>>=1u;}let baseKey=brickOrigin.x+p.baseDims.x*(brickOrigin.y+p.baseDims.y*brickOrigin.z);return hierarchyKey(baseKey,level);}
@compute @workgroup_size(256)fn buildFineSummaryDelta(@builtin(local_invocation_index)lid:u32){
 if(lid==0u){let fineValid=validFineWorklist();let cold=directory[9]!=PUBLISHED;let coarseValid=coarseAuthoritative();let deltaValid=arrayLength(&pageDelta)>p.changedKeysOffset&&pageDelta[1]==p.generation&&pageDelta[15]==PAGE_DELTA_VALID&&pageDelta[0]<=2u*p.pageCapacity&&p.changedKeysOffset+pageDelta[0]<=arrayLength(&pageDelta);let mode=select(0u,select(2u,1u,cold),fineValid&&coarseValid&&(cold||deltaValid));let fineCount=select(0u,select(pageDelta[0],b[0],cold),mode!=0u);let coarseCount=select(0u,select(coarseDelta.count,coarse.rowCount,cold),p.coarseEntryCapacity!=0u&&mode!=0u);let requested=fineCount*(p.maximumLevel+1u)+coarseCount;let error=select(STALE,0u,mode!=0u)|select(0u,CAPACITY,requested>p.recordCapacity||requested>p.sortCapacity||coarseCount>2u*p.coarseEntryCapacity);let count=min(requested,p.sortCapacity);var padded=1u;while(padded<count){padded<<=1u;}padded=min(p.sortCapacity,max(1u,padded));summaryBuildState[0]=mode;summaryBuildState[1]=fineCount;summaryBuildState[2]=coarseCount;summaryBuildState[3]=count;summaryBuildState[4]=padded;workState[0]=mode;workState[1]=fineCount;workState[2]=coarseCount;workState[3]=count;workState[4]=padded;workState[5]=0u;workState[6]=error;workState[7]=0u;workState[8]=0u;workState[9]=0u;workState[10]=0u;workState[11]=0u;workState[12]=INVALID;workState[13]=INVALID;workState[14]=0u;workState[15]=INVALID;workState[19]=arrayLength(&pageDelta);workState[20]=pageDelta[0];workState[21]=select(INVALID,pageDelta[1],arrayLength(&pageDelta)>1u);workState[22]=select(INVALID,pageDelta[15],arrayLength(&pageDelta)>15u);workState[23]=p.changedKeysOffset;workState[24]=p.generation;workState[25]=select(0u,1u,fineValid);workState[26]=select(0u,1u,coarseValid);workState[27]=select(0u,1u,cold);workState[28]=select(0u,1u,deltaValid);workState[29]=select(INVALID,b[1],arrayLength(&b)>1u);workState[30]=select(INVALID,pageDelta[1],arrayLength(&pageDelta)>1u);workState[31]=coarse.generation;publishRecomputeDispatch(count);candidate[0]=error;candidate[1]=p.generation;candidate[2]=0u;candidate[3]=p.entryCapacity;candidate[4]=p.baseDims.x;candidate[5]=p.baseDims.y;candidate[6]=p.baseDims.z;candidate[7]=p.maximumLevel;candidate[8]=0u;candidate[9]=0u;candidate[10]=p.hierarchyKeyCapacity;candidate[11]=p.samplesPerBrick;}
 workgroupBarrier();let mode=workgroupUniformLoad(&summaryBuildState[0]);let fineCount=workgroupUniformLoad(&summaryBuildState[1]);let coarseCount=workgroupUniformLoad(&summaryBuildState[2]);let count=workgroupUniformLoad(&summaryBuildState[3]);let padded=workgroupUniformLoad(&summaryBuildState[4]);let levels=p.maximumLevel+1u;let fineRecords=fineCount*levels;
 for(var index=lid;index<fineRecords;index+=256u){let sourceIndex=index/levels;let level=index%levels;var key=INVALID;if(mode==1u){let id=b[5u+sourceIndex];if(id<p.pageCapacity&&id*10u+2u<arrayLength(&a)&&a[id*10u]==id&&a[id*10u+2u]==p.generation){key=a[id*10u+1u];}}else if(mode==2u){key=pageDelta[p.changedKeysOffset+sourceIndex];}if(key<p.baseDims.x*p.baseDims.y*p.baseDims.z){records[index]=Entry(hierarchyKey(key,level),0xffffffffu,0u,bitcast<u32>(3.402823e38),0u,0u,0u,0u);}else{records[index]=Entry(INVALID,0xffffffffu,0u,bitcast<u32>(3.402823e38),0u,0u,STALE,0u);}}
 for(var row=lid;row<coarseCount;row+=256u){let output=fineRecords+row;let key=coarseHierarchyKey(coarseDirtyIdentity(row));records[output]=Entry(key,0xffffffffu,0u,bitcast<u32>(3.402823e38),0u,0u,select(COARSE_DIRTY,STALE,key==INVALID),0u);}
 for(var index=count+lid;index<padded;index+=256u){records[index]=invalidEntry();}storageBarrier();workgroupBarrier();
 for(var width=2u;width<=padded;width<<=1u){for(var stride=width>>1u;stride>0u;stride>>=1u){for(var index=lid;index<padded;index+=256u){let partner=index^stride;if(partner>index){let left=records[index];let right=records[partner];let descending=(index&width)!=0u;let greater=left.key>right.key;if(greater!=descending){records[index]=right;records[partner]=left;}}}storageBarrier();workgroupBarrier();}}
}
fn combineSummary(left:Entry,right:Entry)->Entry{return Entry(left.key,min(left.minimumPhi,right.minimumPhi),max(left.maximumPhi,right.maximumPhi),min(left.minimumAbsolutePhi,right.minimumAbsolutePhi),left.samples+right.samples,left.bricks+right.bricks,left.flags|right.flags,0u);}
fn recordLowerBound(key:u32)->u32{var lo=0u;var hi=workState[3];while(lo<hi){let mid=lo+(hi-lo)/2u;if(records[mid].key<key){lo=mid+1u;}else{hi=mid;}}return lo;}
fn mortonPart(value:u32)->u32{var x=value&1023u;x=(x|(x<<16u))&0x030000ffu;x=(x|(x<<8u))&0x0300f00fu;x=(x|(x<<4u))&0x030c30c3u;x=(x|(x<<2u))&0x09249249u;return x;}fn coarseMorton(cell:u32)->u32{let q=vec3u(cell%p.finestDims.x,(cell/p.finestDims.x)%p.finestDims.y,cell/(p.finestDims.x*p.finestDims.y));return mortonPart(q.x)|(mortonPart(q.y)<<1u)|(mortonPart(q.z)<<2u);}fn coarseLevel(size:u32)->u32{return 31u-countLeadingZeros(size);}fn coarseLess(al:u32,am:u32,bl:u32,bm:u32)->bool{return al<bl||(al==bl&&am<bm);}
fn coarseIdentityForKey(key:u32)->vec2u{var dims=p.baseDims;var offset=0u;var level=0u;loop{let count=dims.x*dims.y*dims.z;if(key>=offset&&key<offset+count){let local=key-offset;let xy=dims.x*dims.y;let z=local/xy;let rem=local-z*xy;let y=rem/dims.x;let q=vec3u(rem-y*dims.x,y,z);let ratio=p.baseDims/p.finestDims;let side=1u<<level;if(ratio.x==0u||any(ratio!=vec3u(ratio.x))||side%ratio.x!=0u){return vec2u(0u);}let size=side/ratio.x;let origin=q*side/ratio.x;if(size==0u||any(origin>=p.finestDims)){return vec2u(0u);}let cell=origin.x+p.finestDims.x*(origin.y+p.finestDims.y*origin.z);return vec2u(cell+1u,size);}if(level>=p.maximumLevel){break;}offset+=count;dims=(dims+vec3u(1u))/2u;level+=1u;}return vec2u(0u);}
fn coarseSummaryAt(key:u32)->Entry{let identity=coarseIdentityForKey(key);var result=Entry(key,0xffffffffu,0u,bitcast<u32>(3.402823e38),0u,0u,0u,0u);if(identity.x==0u){return result;}let cell=identity.x-1u;let wantedLevel=coarseLevel(identity.y);let wantedMorton=coarseMorton(cell);var low=0u;var high=min(coarse.rowCount,arrayLength(&coarse.entries));while(low<high){let middle=low+(high-low)/2u;let e=coarse.entries[middle];let em=coarseMorton(e.cellPlusOne-1u);if(coarseLess(coarseLevel(e.size),em,wantedLevel,wantedMorton)){low=middle+1u;}else{high=middle;}}if(low>=coarse.rowCount){return result;}let e=coarse.entries[low];if(e.cellPlusOne!=identity.x||e.size!=identity.y||(e.flags&9u)!=9u||!finite(e.phi)||!finite(e.minimumPhi)||!finite(e.maximumPhi)||e.minimumPhi>e.phi||e.phi>e.maximumPhi){return result;}let ma=select(min(abs(e.minimumPhi),abs(e.maximumPhi)),0.0,e.minimumPhi<=0.0&&e.maximumPhi>=0.0);return Entry(key,ordered(e.minimumPhi),ordered(e.maximumPhi),bitcast<u32>(ma),0u,0u,COARSE_AUTHORITY|CENTER_COMPLETE,bitcast<u32>(e.phi));}
fn fineDirtyContains(key:u32)->bool{var at=recordLowerBound(key);while(at<workState[3]&&records[at].key==key){if((records[at].flags&COARSE_DIRTY)==0u){return true;}at+=1u;}return false;}
fn dirtySummaryAt(key:u32)->Entry{var at=recordLowerBound(key);while(at<workState[3]&&records[at].key==key){let value=records[at];if((value.flags&COARSE_DIRTY)==0u){return value;}at+=1u;}var lo=0u;var hi=fineDirectory[2];while(lo<hi){let mid=lo+(hi-lo)/2u;if(readFineDirectory(mid).key<key){lo=mid+1u;}else{hi=mid;}}if(lo<fineDirectory[2]){let value=readFineDirectory(lo);if(value.key==key){return value;}}return invalidEntry();}
fn entryPresent(value:Entry)->bool{return value.samples!=0u||value.bricks!=0u||(value.flags&(COARSE_AUTHORITY|CENTER_COMPLETE))!=0u;}
// Redistance commits a finite signed phi for every current dirty-page sample,
// then uses VALID only to mark membership in the narrow CPT/transport band.
// The exact eight-sample centre phase drives dynamic pressure topology and is
// therefore independent of that band-membership bit. Min/max and completeness
// below remain VALID-gated; this path publishes only the centre value.
fn centerSummary(key:u32,value:Entry)->Entry{var result=value;result.flags&=~CENTER_COMPLETE;result.centerPhi=0u;if(key<p.levelOffset||key>=p.levelOffset+p.levelKeyCount){return result;}let localKey=key-p.levelOffset;let xy=p.levelDims.x*p.levelDims.y;let z=localKey/xy;let rem=localKey-z*xy;let y=rem/p.levelDims.x;let coord=vec3u(rem-y*p.levelDims.x,y,z);let resolution=select(4u,8u,p.samplesPerBrick==512u);let span=(1u<<p.level)*resolution;let centerLow=coord*span+vec3u(span/2u-1u);var center=0.0;var mask=0u;for(var corner=0u;corner<8u;corner+=1u){let delta=vec3u(corner&1u,(corner>>1u)&1u,(corner>>2u)&1u);let q=centerLow+delta;let brick=q/resolution;if(any(brick>=p.baseDims)){continue;}let pageKey=brick.x+p.baseDims.x*(brick.y+p.baseDims.y*brick.z);let id=finePage(pageKey);if(id==INVALID){continue;}let within=q-brick*resolution;let index=id*p.samplesPerBrick+within.x+resolution*(within.y+resolution*within.z);if(index>=arrayLength(&c)||index>=arrayLength(&d)){continue;}let sample=bitcast<f32>(d[index]);if(!finite(sample)){result.flags|=NONFINITE;continue;}center+=0.125*sample;mask|=1u<<corner;}if(mask==0xffu){result.centerPhi=bitcast<u32>(center);result.flags|=CENTER_COMPLETE;}return result;}
@compute @workgroup_size(64)fn recomputeFineSummaryBase(@builtin(workgroup_id)w:vec3u,@builtin(local_invocation_index)lid:u32,@builtin(num_workgroups)n:vec3u){let recordIndex=fineLinearWorkgroup(w,n);let enabled=recordIndex<workState[3];let key=select(INVALID,records[recordIndex].key,enabled);let inLevel=enabled&&key>=p.levelOffset&&key<p.levelOffset+p.levelKeyCount;let fineRecord=inLevel;var lo=3.402823e38;var hi=-3.402823e38;var ma=3.402823e38;var count=0u;var failure=0u;let id=select(INVALID,finePage(key),fineRecord);if(id!=INVALID){for(var local=lid;local<p.samplesPerBrick;local+=64u){let index=id*p.samplesPerBrick+local;if(index>=arrayLength(&c)||index>=arrayLength(&d)){failure|=CAPACITY;continue;}if((c[index]&VALID)==0u){continue;}let sample=bitcast<f32>(d[index]);if(!finite(sample)){failure|=NONFINITE;continue;}lo=min(lo,sample);hi=max(hi,sample);ma=min(ma,abs(sample));count+=1u;}}minimumPhi[lid]=lo;maximumPhi[lid]=hi;minimumAbsolutePhi[lid]=ma;validSamples[lid]=count;errors[lid]=failure;workgroupBarrier();for(var stride=32u;stride>0u;stride>>=1u){if(lid<stride){minimumPhi[lid]=min(minimumPhi[lid],minimumPhi[lid+stride]);maximumPhi[lid]=max(maximumPhi[lid],maximumPhi[lid+stride]);minimumAbsolutePhi[lid]=min(minimumAbsolutePhi[lid],minimumAbsolutePhi[lid+stride]);validSamples[lid]+=validSamples[lid+stride];errors[lid]|=errors[lid+stride];}workgroupBarrier();}if(lid==0u&&fineRecord){var value=Entry(key,0xffffffffu,0u,bitcast<u32>(3.402823e38),0u,0u,errors[0],0u);if(validSamples[0]>0u){value=Entry(key,ordered(minimumPhi[0]),ordered(maximumPhi[0]),bitcast<u32>(minimumAbsolutePhi[0]),validSamples[0],1u,errors[0],0u);}records[recordIndex]=centerSummary(key,value);}}
@compute @workgroup_size(64)fn recomputeFineSummaryParents(@builtin(workgroup_id)w:vec3u,@builtin(local_invocation_index)lid:u32,@builtin(num_workgroups)n:vec3u){if(lid!=0u){return;}let recordIndex=fineLinearWorkgroup(w,n);if(recordIndex>=workState[3]){return;}let key=records[recordIndex].key;if(key<p.levelOffset||key>=p.levelOffset+p.levelKeyCount){return;}let localKey=key-p.levelOffset;let xy=p.levelDims.x*p.levelDims.y;let z=localKey/xy;let rem=localKey-z*xy;let y=rem/p.levelDims.x;let coord=vec3u(rem-y*p.levelDims.x,y,z);var childDims=p.baseDims;var childOffset=0u;for(var level=0u;level+1u<p.level;level+=1u){childOffset+=childDims.x*childDims.y*childDims.z;childDims=(childDims+vec3u(1u))/2u;}var value=Entry(key,0xffffffffu,0u,bitcast<u32>(3.402823e38),0u,0u,0u,0u);for(var child=0u;child<8u;child+=1u){let q=coord*2u+vec3u(child&1u,(child>>1u)&1u,(child>>2u)&1u);if(any(q>=childDims)){continue;}let childKey=childOffset+q.x+childDims.x*(q.y+childDims.y*q.z);let item=dirtySummaryAt(childKey);if(entryPresent(item)){value=combineSummary(value,item);}}records[recordIndex]=centerSummary(key,value);}
fn reductionRange(count:u32,lid:u32)->vec2u{let width=count/256u;let remainder=count%256u;let begin=lid*width+min(lid,remainder);return vec2u(begin,begin+width+select(0u,1u,lid<remainder));}
fn lanePrefix(value:u32,lid:u32)->u32{reductionValues[lid]=value;workgroupBarrier();for(var stride=1u;stride<256u;stride<<=1u){var add=0u;if(lid>=stride){add=reductionValues[lid-stride];}workgroupBarrier();reductionValues[lid]+=add;workgroupBarrier();}return reductionValues[lid]-value;}
fn dirtyContains(key:u32)->bool{let at=recordLowerBound(key);return at<workState[3]&&records[at].key==key;}
fn committedFineAt(key:u32)->Entry{var lo=0u;var hi=fineDirectory[2];while(lo<hi){let mid=lo+(hi-lo)/2u;if(readFineDirectory(mid).key<key){lo=mid+1u;}else{hi=mid;}}if(lo<fineDirectory[2]){let value=readFineDirectory(lo);if(value.key==key){return value;}}return Entry(key,0xffffffffu,0u,bitcast<u32>(3.402823e38),0u,0u,0u,0u);}
fn publicDirtySummary(key:u32)->Entry{var value=committedFineAt(key);let fineCenter=value.centerPhi;let fineCenterComplete=(value.flags&CENTER_COMPLETE)==CENTER_COMPLETE;let coarseValue=coarseSummaryAt(key);if(entryPresent(coarseValue)){value=combineSummary(value,coarseValue);value.centerPhi=select(coarseValue.centerPhi,fineCenter,fineCenterComplete);}return value;}
fn stageFineOnlyCarry(lid:u32){let enabled=workState[0]!=0u&&workState[6]==0u;let count=select(0u,fineDirectory[2],enabled&&fineDirectory[9]==PUBLISHED);let range=reductionRange(count,lid);var localCount=0u;for(var index=range.x;index<range.y;index+=1u){localCount+=select(0u,1u,!fineDirtyContains(readFineDirectory(index).key));}let base=lanePrefix(localCount,lid);var cursor=base;for(var index=range.x;index<range.y;index+=1u){let value=readFineDirectory(index);if(!fineDirtyContains(value.key)){carryRecords[cursor]=value;cursor+=1u;}}if(lid==255u){workState[7]=base+localCount;}}
fn stageFineOnlyDirty(lid:u32){let enabled=workState[0]!=0u&&workState[6]==0u;let range=reductionRange(select(0u,workState[3],enabled),lid);var localCount=0u;for(var index=range.x;index<range.y;index+=1u){let key=records[index].key;let first=index==0u||records[index-1u].key!=key;let fineDirty=first&&key!=INVALID&&fineDirtyContains(key);var value=invalidEntry();if(fineDirty){value=dirtySummaryAt(key);}localCount+=select(0u,1u,fineDirty&&entryPresent(value));}let base=lanePrefix(localCount,lid);var cursor=base;for(var index=range.x;index<range.y;index+=1u){let key=records[index].key;let first=index==0u||records[index-1u].key!=key;let fineDirty=first&&key!=INVALID&&fineDirtyContains(key);var value=invalidEntry();if(fineDirty){value=dirtySummaryAt(key);}if(fineDirty&&entryPresent(value)){sortScratch[cursor]=value;cursor+=1u;}}if(lid==255u){workState[8]=base+localCount;}}
fn stagePrepareFineOnlyMerge(lid:u32){if(lid==0u){let total=workState[7]+workState[8];workState[9]=min(total,p.entryCapacity);workState[6]|=select(0u,CAPACITY,total>p.entryCapacity);fineCandidate[0]=workState[6];fineCandidate[1]=p.generation;fineCandidate[2]=workState[9];fineCandidate[3]=p.entryCapacity;fineCandidate[4]=p.baseDims.x;fineCandidate[5]=p.baseDims.y;fineCandidate[6]=p.baseDims.z;fineCandidate[7]=p.maximumLevel;fineCandidate[8]=0u;fineCandidate[9]=0u;fineCandidate[10]=p.hierarchyKeyCapacity;fineCandidate[11]=p.samplesPerBrick;}}
fn stageMergeFineOnlyCandidate(lid:u32){let range=reductionRange(workState[9],lid);for(var index=range.x;index<range.y;index+=1u){let ai=mergePartition(index,workState[7],workState[8]);let bi=index-ai;if(ai<workState[7]&&(bi>=workState[8]||carryRecords[ai].key<=sortScratch[bi].key)){writeFineCandidate(index,carryRecords[ai]);}else{writeFineCandidate(index,sortScratch[bi]);}}}
fn stageValidateFineOnlyCandidate(lid:u32){let enabled=workState[0]!=0u;let range=reductionRange(select(0u,workState[9],enabled),lid);var error=workState[6];var first=INVALID;for(var index=range.x;index<range.y;index+=1u){let base=fineEntryBase(index);let key=fineCandidate[base];let flags=fineCandidate[base+6u];let bad=(flags&0x003fffffu)!=0u||key==INVALID||key>=fineCandidate[10]||(index>0u&&fineCandidate[base-8u]>=key);error|=flags&0x003fffffu;error|=select(0u,STALE,key==INVALID||key>=fineCandidate[10]||(index>0u&&fineCandidate[base-8u]>=key));first=select(first,min(first,index),bad);}reductionValues[lid]=error;workgroupBarrier();for(var stride=128u;stride>0u;stride>>=1u){if(lid<stride){reductionValues[lid]|=reductionValues[lid+stride];}workgroupBarrier();}let aggregateError=workgroupUniformLoad(&reductionValues[0]);reductionValues[lid]=first;workgroupBarrier();for(var stride=128u;stride>0u;stride>>=1u){if(lid<stride){reductionValues[lid]=min(reductionValues[lid],reductionValues[lid+stride]);}workgroupBarrier();}let valid=enabled&&aggregateError==0u;if(lid==0u){let firstBad=reductionValues[0];if(firstBad!=INVALID){let base=fineEntryBase(firstBad);workState[12]=firstBad;workState[13]=fineCandidate[base];workState[14]=fineCandidate[base+6u];workState[15]=select(INVALID,fineCandidate[base-8u],firstBad>0u);}workState[11]=select(0u,PUBLISHED,valid);workState[6]|=select(STALE,0u,valid);fineCandidate[0]=aggregateError;fineCandidate[9]=select(0u,PUBLISHED,valid);}workgroupBarrier();if(valid){let words=16u+workState[9]*8u;for(var word=lid;word<words;word+=256u){fineDirectory[word]=fineCandidate[word];}}}
fn stageGlobalCarry(lid:u32){let enabled=workState[0]!=0u&&workState[6]==0u;let count=select(0u,directory[2],enabled);let range=reductionRange(count,lid);var localCount=0u;for(var index=range.x;index<range.y;index+=1u){localCount+=select(0u,1u,!dirtyContains(readDirectory(index).key));}let base=lanePrefix(localCount,lid);var cursor=base;for(var index=range.x;index<range.y;index+=1u){let value=readDirectory(index);if(!dirtyContains(value.key)){carryRecords[cursor]=value;cursor+=1u;}}if(lid==255u){workState[7]=base+localCount;}}
fn stageGlobalDirty(lid:u32){let enabled=workState[0]!=0u&&workState[6]==0u;let range=reductionRange(select(0u,workState[3],enabled),lid);var localCount=0u;for(var index=range.x;index<range.y;index+=1u){let key=records[index].key;let last=index+1u>=workState[3]||records[index+1u].key!=key;localCount+=select(0u,1u,last&&key!=INVALID&&entryPresent(publicDirtySummary(key)));}let base=lanePrefix(localCount,lid);var cursor=base;for(var index=range.x;index<range.y;index+=1u){let key=records[index].key;let last=index+1u>=workState[3]||records[index+1u].key!=key;if(last&&key!=INVALID){let value=publicDirtySummary(key);if(entryPresent(value)){sortScratch[cursor]=value;cursor+=1u;}}}if(lid==255u){workState[8]=base+localCount;}}
fn stagePrepareGlobalMerge(lid:u32){if(lid==0u){let total=workState[7]+workState[8];workState[9]=min(total,p.entryCapacity);workState[6]|=select(0u,CAPACITY,total>p.entryCapacity);candidate[0]=workState[6];candidate[2]=workState[9];}}
fn mergePartition(index:u32,aCount:u32,bCount:u32)->u32{var lo=select(0u,index-bCount,index>bCount);var hi=min(index,aCount);while(lo<hi){let ai=lo+(hi-lo)/2u;let bi=index-ai;if(ai<aCount&&bi>0u&&sortScratch[bi-1u].key>=carryRecords[ai].key){lo=ai+1u;}else{hi=ai;}}return lo;}
fn stageMergeGlobalCandidate(lid:u32){let range=reductionRange(workState[9],lid);for(var index=range.x;index<range.y;index+=1u){let ai=mergePartition(index,workState[7],workState[8]);let bi=index-ai;if(ai<workState[7]&&(bi>=workState[8]||carryRecords[ai].key<=sortScratch[bi].key)){writeCandidate(index,carryRecords[ai]);}else{writeCandidate(index,sortScratch[bi]);}}}
fn stageValidateGlobalCandidate(lid:u32){let enabled=workState[0]!=0u;let range=reductionRange(select(0u,workState[9],enabled),lid);var error=workState[6];for(var index=range.x;index<range.y;index+=1u){let base=16u+index*8u;let key=candidate[base];let flags=candidate[base+6u];error|=flags&0x003fffffu;error|=select(0u,STALE,key==INVALID||key>=candidate[10]||(index>0u&&candidate[base-8u]>=key));}reductionValues[lid]=error;workgroupBarrier();for(var stride=128u;stride>0u;stride>>=1u){if(lid<stride){reductionValues[lid]|=reductionValues[lid+stride];}workgroupBarrier();}let valid=enabled&&reductionValues[0]==0u;if(lid==0u){workState[10]=reductionValues[0];candidate[0]=reductionValues[0];candidate[9]=select(0u,PUBLISHED,valid);}workgroupBarrier();if(valid){let words=16u+workState[9]*8u;for(var word=lid;word<words;word+=256u){directory[word]=candidate[word];}}}
@compute @workgroup_size(256)fn publishFineSummaryDelta(@builtin(local_invocation_index)lid:u32){
 stageFineOnlyCarry(lid);stageFineOnlyDirty(lid);storageBarrier();workgroupBarrier();
 stagePrepareFineOnlyMerge(lid);storageBarrier();workgroupBarrier();
 stageMergeFineOnlyCandidate(lid);storageBarrier();workgroupBarrier();
 stageValidateFineOnlyCandidate(lid);storageBarrier();workgroupBarrier();
 stageGlobalCarry(lid);stageGlobalDirty(lid);storageBarrier();workgroupBarrier();
 stagePrepareGlobalMerge(lid);storageBarrier();workgroupBarrier();
 stageMergeGlobalCandidate(lid);storageBarrier();workgroupBarrier();
 stageValidateGlobalCandidate(lid);
}
`;
