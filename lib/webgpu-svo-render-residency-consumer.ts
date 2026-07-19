/**
 * GPU-native consumer for the solver-owned sparse-fluid residency publication.
 *
 * The producer publication, state, and worklist arenas are bound read-only.
 * This class owns only its compact request/release arenas, counters, indirect
 * dispatch metadata, and duplicate-detection stamps. It does not allocate fine
 * pages and it never changes simulation residency or topology.
 */
import { buildSvoRenderResidencyGpuInputs } from "./svo-render-residency-source-adapter";
import {
  SPARSE_VOXEL_FLUID_RESIDENCY_STATE_BITS,
  SPARSE_VOXEL_FLUID_RESIDENCY_WORKLIST_WORDS,
  SPARSE_VOXEL_VALID_FIELDS,
  sparseVoxelFluidResidencyLayout,
  type SparseVoxelStructuralRenderSource,
} from "./webgpu-voxel-debug";

const UINT32_MAX = 0xffff_ffff;
export const SVO_RENDER_RESIDENCY_CONSUMER_STATUS = Object.freeze({
  ready: 1 << 0,
  unchanged: 1 << 1,
  stale: 1 << 2,
  unpublished: 1 << 3,
  missingFields: 1 << 4,
  generationMismatch: 1 << 5,
  counterMismatch: 1 << 6,
  invalidEntry: 1 << 7,
  coarseFallback: 1 << 8,
  sourceOverflow: 1 << 9,
  rendererExhausted: 1 << 10,
} as const);

export const SVO_RENDER_RESIDENCY_CONSUMER_CONTROL_WORDS = Object.freeze({
  status: 0,
  observedGeneration: 1,
  acceptedGeneration: 2,
  sourceActiveCount: 3,
  sourceCoreCount: 4,
  sourceHaloCount: 5,
  sourceRetiredCount: 6,
  dirtyActiveCount: 7,
  dirtyCoreCount: 8,
  dirtyHaloCount: 9,
  dirtyRetiredCount: 10,
  sourceOverflowCount: 11,
  rendererExhaustedCount: 12,
  invalidEntryCount: 13,
  stalePublicationCount: 14,
  coarseFallbackCount: 15,
  attemptToken: 16,
  activeInputDispatch: 20,
  retiredInputDispatch: 24,
  activeOutputDispatch: 28,
  coreOutputDispatch: 32,
  haloOutputDispatch: 36,
  retiredOutputDispatch: 40,
  length: 64,
} as const);

export type SvoRenderResidencyConsumerList = "active" | "core" | "halo" | "retired";

export interface SvoRenderResidencyConsumerLayout {
  capacity: number;
  entryStrideBytes: 8;
  /** Each list starts at WebGPU's portable storage-buffer offset alignment. */
  listStrideBytes: number;
  controlByteLength: number;
  entryByteLength: number;
  entryOffsetsBytes: Readonly<Record<SvoRenderResidencyConsumerList, number>>;
  indirectOffsetsBytes: Readonly<Record<`${SvoRenderResidencyConsumerList}Output` | "activeInput" | "retiredInput", number>>;
}

function positiveUint(value: number, label: string): number {
  if (!Number.isSafeInteger(value) || value < 1 || value > UINT32_MAX) throw new RangeError(`${label} must be a positive uint32`);
  return value >>> 0;
}

export function svoRenderResidencyConsumerLayout(capacity: number): SvoRenderResidencyConsumerLayout {
  positiveUint(capacity, "Renderer residency capacity");
  const listByteLength = capacity * 8;
  const listStrideBytes = Math.ceil(listByteLength / 256) * 256;
  if (!Number.isSafeInteger(listStrideBytes * 4) || listStrideBytes * 4 > UINT32_MAX) {
    throw new RangeError("Renderer residency output arena exceeds the uint32 addressable ABI");
  }
  const output = (word: number) => word * 4;
  return {
    capacity,
    entryStrideBytes: 8,
    listStrideBytes,
    controlByteLength: SVO_RENDER_RESIDENCY_CONSUMER_CONTROL_WORDS.length * 4,
    entryByteLength: listStrideBytes * 4,
    entryOffsetsBytes: {
      active: 0,
      core: listStrideBytes,
      halo: listStrideBytes * 2,
      retired: listStrideBytes * 3,
    },
    indirectOffsetsBytes: {
      activeInput: output(SVO_RENDER_RESIDENCY_CONSUMER_CONTROL_WORDS.activeInputDispatch),
      retiredInput: output(SVO_RENDER_RESIDENCY_CONSUMER_CONTROL_WORDS.retiredInputDispatch),
      activeOutput: output(SVO_RENDER_RESIDENCY_CONSUMER_CONTROL_WORDS.activeOutputDispatch),
      coreOutput: output(SVO_RENDER_RESIDENCY_CONSUMER_CONTROL_WORDS.coreOutputDispatch),
      haloOutput: output(SVO_RENDER_RESIDENCY_CONSUMER_CONTROL_WORDS.haloOutputDispatch),
      retiredOutput: output(SVO_RENDER_RESIDENCY_CONSUMER_CONTROL_WORDS.retiredOutputDispatch),
    },
  };
}

export interface SvoRenderResidencyConsumerReferenceInput {
  publicationWords: Uint32Array;
  worklistWords: Uint32Array;
  stateWords: Uint32Array;
  sourceCapacity: number;
  rendererCapacity: number;
  leafCapacity: number;
  acceptedGeneration?: number;
  /** Optional embedding of the solver-brick list in the structural scene. */
  sourceOriginBricks?: readonly [number, number, number];
  sourceDimensionsBricks?: readonly [number, number, number];
  structuralDimensionsBricks?: readonly [number, number, number];
}

export interface SvoRenderResidencyConsumerReferenceResult {
  status: number;
  acceptedGeneration: number;
  active: readonly (readonly [number, number])[];
  core: readonly (readonly [number, number])[];
  halo: readonly (readonly [number, number])[];
  retired: readonly (readonly [number, number])[];
  telemetry: Readonly<{
    sourceOverflowCount: number;
    rendererExhaustedCount: number;
    invalidEntryCount: number;
    stalePublicationCount: number;
    coarseFallbackCount: number;
  }>;
}

/** Deterministic CPU oracle for the GPU fence and compaction contract. */
export function referenceSvoRenderResidencyConsumption(
  input: SvoRenderResidencyConsumerReferenceInput,
): SvoRenderResidencyConsumerReferenceResult {
  const sourceCapacity = positiveUint(input.sourceCapacity, "Source residency capacity");
  const rendererCapacity = positiveUint(input.rendererCapacity, "Renderer residency capacity");
  const leafCapacity = positiveUint(input.leafCapacity, "Sparse leaf capacity");
  const accepted = input.acceptedGeneration ?? 0;
  if (!Number.isInteger(accepted) || accepted < 0 || accepted > UINT32_MAX) throw new RangeError("Accepted generation must fit uint32");
  const sourceLayout = sparseVoxelFluidResidencyLayout(sourceCapacity);
  const remapSourceBrick = (brick: number) => {
    if (!input.sourceOriginBricks || !input.sourceDimensionsBricks || !input.structuralDimensionsBricks) return brick;
    const source = input.sourceDimensionsBricks, structural = input.structuralDimensionsBricks, origin = input.sourceOriginBricks;
    const x = brick % source[0], y = Math.floor(brick / source[0]) % source[1], z = Math.floor(brick / (source[0] * source[1]));
    const scene = [x + origin[0], y + origin[1], z + origin[2]];
    if (z >= source[2] || scene.some((value, axis) => value >= structural[axis])) return -1;
    return scene[0] + structural[0] * (scene[1] + structural[1] * scene[2]);
  };
  if (input.publicationWords.length < 8 || input.worklistWords.byteLength < sourceLayout.worklistByteLength || input.stateWords.length < sourceCapacity) {
    throw new RangeError("Residency consumer reference snapshot is undersized");
  }
  const empty = (status: number, stalePublicationCount = 0): SvoRenderResidencyConsumerReferenceResult => ({
    status,
    acceptedGeneration: accepted,
    active: [], core: [], halo: [], retired: [],
    telemetry: { sourceOverflowCount: 0, rendererExhaustedCount: 0, invalidEntryCount: 0, stalePublicationCount, coarseFallbackCount: 0 },
  });
  const generation = input.publicationWords[0] >>> 0;
  const validFields = input.publicationWords[1] >>> 0;
  const revision = input.publicationWords[5] >>> 0;
  const listGeneration = input.worklistWords[SPARSE_VOXEL_FLUID_RESIDENCY_WORKLIST_WORDS.generation] >>> 0;
  if (generation === 0) return empty(SVO_RENDER_RESIDENCY_CONSUMER_STATUS.unpublished);
  const required = SPARSE_VOXEL_VALID_FIELDS.topology | SPARSE_VOXEL_VALID_FIELDS.coarseFluid;
  if ((validFields & required) !== required) return empty(SVO_RENDER_RESIDENCY_CONSUMER_STATUS.missingFields);
  if (revision !== generation || listGeneration !== generation) return empty(SVO_RENDER_RESIDENCY_CONSUMER_STATUS.generationMismatch);
  if (generation < accepted) return empty(SVO_RENDER_RESIDENCY_CONSUMER_STATUS.stale, 1);
  if (generation === accepted) return empty(SVO_RENDER_RESIDENCY_CONSUMER_STATUS.unchanged);
  const sourceActiveCount = input.worklistWords[SPARSE_VOXEL_FLUID_RESIDENCY_WORKLIST_WORDS.activeCount] >>> 0;
  const sourceCoreCount = input.worklistWords[SPARSE_VOXEL_FLUID_RESIDENCY_WORKLIST_WORDS.coreCount] >>> 0;
  const sourceHaloCount = input.worklistWords[SPARSE_VOXEL_FLUID_RESIDENCY_WORKLIST_WORDS.haloCount] >>> 0;
  const sourceRetiredCount = input.worklistWords[SPARSE_VOXEL_FLUID_RESIDENCY_WORKLIST_WORDS.retiredCount] >>> 0;
  const retiredStats = input.worklistWords[SPARSE_VOXEL_FLUID_RESIDENCY_WORKLIST_WORDS.retiredStatsCount] >>> 0;
  if (sourceCoreCount + sourceHaloCount !== sourceActiveCount || retiredStats !== sourceRetiredCount) {
    return empty(SVO_RENDER_RESIDENCY_CONSUMER_STATUS.counterMismatch);
  }
  let status = SVO_RENDER_RESIDENCY_CONSUMER_STATUS.ready;
  const sourceActiveOverflow = Math.max(0, sourceActiveCount - sourceCapacity);
  const sourceRetiredOverflow = Math.max(0, sourceRetiredCount - sourceCapacity);
  const sourceOverflowCount = sourceActiveOverflow + sourceRetiredOverflow;
  if (sourceOverflowCount > 0) status |= SVO_RENDER_RESIDENCY_CONSUMER_STATUS.sourceOverflow;
  const active: [number, number][] = [], core: [number, number][] = [], halo: [number, number][] = [], retired: [number, number][] = [];
  const seenBricks = new Set<number>(), seenLeaves = new Set<number>();
  let invalidEntryCount = 0;
  const consume = (offsetBytes: number, count: number, isRetired: boolean) => {
    for (let index = 0; index < Math.min(count, sourceCapacity); index += 1) {
      const word = offsetBytes / 4 + index * 2;
      const brick = input.worklistWords[word] >>> 0, leaf = input.worklistWords[word + 1] >>> 0;
      const state = brick < sourceCapacity ? input.stateWords[brick] >>> 0 : 0;
      const flags = state & 0xff;
      const validState = isRetired
        ? (flags & SPARSE_VOXEL_FLUID_RESIDENCY_STATE_BITS.resident) === 0 && (flags & SPARSE_VOXEL_FLUID_RESIDENCY_STATE_BITS.wasResident) !== 0
        : (flags & SPARSE_VOXEL_FLUID_RESIDENCY_STATE_BITS.resident) !== 0
          && ((flags & SPARSE_VOXEL_FLUID_RESIDENCY_STATE_BITS.core) !== 0) !== ((flags & SPARSE_VOXEL_FLUID_RESIDENCY_STATE_BITS.halo) !== 0);
      if (brick >= sourceCapacity || leaf >= leafCapacity || !validState || seenBricks.has(brick) || seenLeaves.has(leaf)) {
        invalidEntryCount += 1;
        continue;
      }
      seenBricks.add(brick); seenLeaves.add(leaf);
      const structuralBrick = remapSourceBrick(brick);
      if (structuralBrick < 0) { invalidEntryCount += 1; continue; }
      const entry: [number, number] = [structuralBrick, leaf];
      if (isRetired) retired.push(entry);
      else {
        active.push(entry);
        ((flags & SPARSE_VOXEL_FLUID_RESIDENCY_STATE_BITS.core) !== 0 ? core : halo).push(entry);
      }
    }
  };
  consume(sourceLayout.activeEntryOffsetBytes, sourceActiveCount, false);
  consume(sourceLayout.retiredEntryOffsetBytes, sourceRetiredCount, true);
  if (invalidEntryCount > 0 || (sourceActiveCount <= sourceCapacity && (core.length !== sourceCoreCount || halo.length !== sourceHaloCount))) {
    status |= SVO_RENDER_RESIDENCY_CONSUMER_STATUS.invalidEntry | SVO_RENDER_RESIDENCY_CONSUMER_STATUS.coarseFallback;
    return {
      ...empty(status),
      telemetry: { sourceOverflowCount, rendererExhaustedCount: 0, invalidEntryCount, stalePublicationCount: 0, coarseFallbackCount: sourceActiveCount },
    };
  }
  const rendererExhaustedCount = Math.max(0, active.length - rendererCapacity) + Math.max(0, retired.length - rendererCapacity);
  const activeExhausted = Math.max(0, active.length - rendererCapacity);
  const coarseFallbackCount = sourceActiveOverflow + activeExhausted;
  if (rendererExhaustedCount > 0) status |= SVO_RENDER_RESIDENCY_CONSUMER_STATUS.rendererExhausted;
  if (coarseFallbackCount > 0) status |= SVO_RENDER_RESIDENCY_CONSUMER_STATUS.coarseFallback;
  return {
    status,
    acceptedGeneration: generation,
    active: active.slice(0, rendererCapacity),
    core: core.slice(0, rendererCapacity),
    halo: halo.slice(0, rendererCapacity),
    retired: retired.slice(0, rendererCapacity),
    telemetry: { sourceOverflowCount, rendererExhaustedCount, invalidEntryCount: 0, stalePublicationCount: 0, coarseFallbackCount },
  };
}

export const svoRenderResidencyConsumerShader = /* wgsl */ `
struct Params {
  sourceCapacity:u32, rendererCapacity:u32, leafCapacity:u32, activeOffsetWords:u32,
  retiredOffsetWords:u32, requiredFields:u32, outputListStrideWords:u32, _pad0:u32,
  sourceOriginBricks:vec4u, sourceDimensionsBricks:vec4u, structuralDimensionsBricks:vec4u,
}
@group(0) @binding(0) var<storage,read> publication:array<u32>;
@group(0) @binding(1) var<storage,read> states:array<u32>;
@group(0) @binding(2) var<storage,read> sourceWorklist:array<u32>;
@group(0) @binding(3) var<uniform> params:Params;
@group(0) @binding(4) var<storage,read_write> control:array<atomic<u32>>;
@group(0) @binding(5) var<storage,read_write> outputEntries:array<u32>;
@group(0) @binding(6) var<storage,read_write> seenAttempt:array<atomic<u32>>;

const RESIDENT:u32=1u; const CORE:u32=2u; const HALO:u32=4u; const WAS_RESIDENT:u32=32u;
const READY:u32=1u; const UNCHANGED:u32=2u; const STALE:u32=4u; const UNPUBLISHED:u32=8u;
const MISSING_FIELDS:u32=16u; const GENERATION_MISMATCH:u32=32u; const COUNTER_MISMATCH:u32=64u;
const INVALID_ENTRY:u32=128u; const COARSE_FALLBACK:u32=256u; const SOURCE_OVERFLOW:u32=512u; const RENDERER_EXHAUSTED:u32=1024u;
const ACCEPTED:u32=2u; const DIRTY_ACTIVE:u32=7u; const DIRTY_CORE:u32=8u; const DIRTY_HALO:u32=9u; const DIRTY_RETIRED:u32=10u;
const SOURCE_OVERFLOW_COUNT:u32=11u; const EXHAUSTED_COUNT:u32=12u; const INVALID_COUNT:u32=13u; const STALE_COUNT:u32=14u; const COARSE_COUNT:u32=15u; const ATTEMPT:u32=16u;
const ACTIVE_INPUT_DISPATCH:u32=20u; const RETIRED_INPUT_DISPATCH:u32=24u; const ACTIVE_OUTPUT_DISPATCH:u32=28u; const CORE_OUTPUT_DISPATCH:u32=32u; const HALO_OUTPUT_DISPATCH:u32=36u; const RETIRED_OUTPUT_DISPATCH:u32=40u;

fn storeDispatch(base:u32,count:u32) {
  let groups=count/64u+select(0u,1u,(count%64u)!=0u); let x=min(groups,65535u); let y=select(1u,(groups+x-1u)/x,x>0u);
  atomicStore(&control[base],x); atomicStore(&control[base+1u],y); atomicStore(&control[base+2u],1u); atomicStore(&control[base+3u],0u);
}
fn satAdd(a:u32,b:u32)->u32 { return a+min(b,0xffffffffu-a); }
fn clearDispatch(base:u32) { atomicStore(&control[base],0u); atomicStore(&control[base+1u],1u); atomicStore(&control[base+2u],1u); atomicStore(&control[base+3u],0u); }
fn reject(status:u32) { atomicStore(&control[0],status); clearDispatch(ACTIVE_INPUT_DISPATCH); clearDispatch(RETIRED_INPUT_DISPATCH); }

@compute @workgroup_size(1)
fn prepare() {
  for(var i=0u;i<17u;i+=1u) { if(i!=ACCEPTED && i!=ATTEMPT) { atomicStore(&control[i],0u); } }
  clearDispatch(ACTIVE_INPUT_DISPATCH); clearDispatch(RETIRED_INPUT_DISPATCH); clearDispatch(ACTIVE_OUTPUT_DISPATCH); clearDispatch(CORE_OUTPUT_DISPATCH); clearDispatch(HALO_OUTPUT_DISPATCH); clearDispatch(RETIRED_OUTPUT_DISPATCH);
  let generation=publication[0]; atomicStore(&control[1],generation);
  if(generation==0u) { reject(UNPUBLISHED); return; }
  if((publication[1]&params.requiredFields)!=params.requiredFields) { reject(MISSING_FIELDS); return; }
  if(publication[5]!=generation || sourceWorklist[15]!=generation) { reject(GENERATION_MISMATCH); return; }
  let activeCount=sourceWorklist[0]; let retiredCount=sourceWorklist[4]; let coreCount=sourceWorklist[8]; let haloCount=sourceWorklist[9];
  atomicStore(&control[3],activeCount); atomicStore(&control[4],coreCount); atomicStore(&control[5],haloCount); atomicStore(&control[6],retiredCount);
  if(coreCount>activeCount || haloCount!=activeCount-coreCount || sourceWorklist[11]!=retiredCount) { reject(COUNTER_MISMATCH); return; }
  let accepted=atomicLoad(&control[ACCEPTED]);
  if(generation<accepted) { atomicStore(&control[STALE_COUNT],1u); reject(STALE); return; }
  if(generation==accepted) { reject(UNCHANGED); return; }
  let sourceOverflow=satAdd(activeCount-min(activeCount,params.sourceCapacity),retiredCount-min(retiredCount,params.sourceCapacity));
  atomicStore(&control[SOURCE_OVERFLOW_COUNT],sourceOverflow);
  var status=READY;
  if(sourceOverflow>0u) { status|=SOURCE_OVERFLOW; atomicStore(&control[COARSE_COUNT],activeCount-min(activeCount,params.sourceCapacity)); }
  atomicStore(&control[0],status); atomicAdd(&control[ATTEMPT],1u);
  storeDispatch(ACTIVE_INPUT_DISPATCH,min(activeCount,params.sourceCapacity));
  storeDispatch(RETIRED_INPUT_DISPATCH,min(retiredCount,params.sourceCapacity));
}

fn sourceIndex(wid:vec3u,lid:u32,dispatchBase:u32)->u32 { return (wid.y*atomicLoad(&control[dispatchBase])+wid.x)*64u+lid; }
fn claimEntry(brick:u32,leaf:u32,attempt:u32)->bool {
  let leafStamp=params.sourceCapacity+leaf;
  if(brick>=params.sourceCapacity || leaf>=params.leafCapacity || leafStamp>=arrayLength(&seenAttempt)) { return false; }
  let uniqueBrick=atomicExchange(&seenAttempt[brick],attempt)!=attempt;
  let uniqueLeaf=atomicExchange(&seenAttempt[leafStamp],attempt)!=attempt;
  return uniqueBrick&&uniqueLeaf;
}
fn writeEntry(list:u32,slot:u32,brick:u32,leaf:u32) {
  if(slot>=params.rendererCapacity) { return; }
  let base=list*params.outputListStrideWords+slot*2u; outputEntries[base]=brick; outputEntries[base+1u]=leaf;
}
fn structuralBrick(sourceBrick:u32)->u32 {
  let sourceDims=params.sourceDimensionsBricks.xyz;
  if(any(sourceDims==vec3u(0u))||sourceBrick>=sourceDims.x*sourceDims.y*sourceDims.z){return 0xffffffffu;}
  let local=vec3u(sourceBrick%sourceDims.x,(sourceBrick/sourceDims.x)%sourceDims.y,sourceBrick/(sourceDims.x*sourceDims.y));
  let scene=local+params.sourceOriginBricks.xyz;let structural=params.structuralDimensionsBricks.xyz;
  if(any(scene>=structural)){return 0xffffffffu;}
  return scene.x+structural.x*(scene.y+structural.y*scene.z);
}
@compute @workgroup_size(64)
fn consumeActive(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32) {
  if((atomicLoad(&control[0])&READY)==0u) { return; }
  let index=sourceIndex(wid,lid,ACTIVE_INPUT_DISPATCH); if(index>=min(atomicLoad(&control[3]),params.sourceCapacity)) { return; }
  let word=params.activeOffsetWords+index*2u; let brick=sourceWorklist[word]; let leaf=sourceWorklist[word+1u];
  var state=0u; if(brick<params.sourceCapacity) { state=states[brick]; } let flags=state&0xffu;
  let valid=(flags&RESIDENT)!=0u && (((flags&CORE)!=0u)!=((flags&HALO)!=0u));
  let attempt=atomicLoad(&control[ATTEMPT]);
  if(!valid || !claimEntry(brick,leaf,attempt)) { atomicAdd(&control[INVALID_COUNT],1u); atomicOr(&control[0],INVALID_ENTRY); return; }
  let mapped=structuralBrick(brick);if(mapped==0xffffffffu){atomicAdd(&control[INVALID_COUNT],1u);atomicOr(&control[0],INVALID_ENTRY);return;}
  let activeSlot=atomicAdd(&control[DIRTY_ACTIVE],1u); writeEntry(0u,activeSlot,mapped,leaf);
  if((flags&CORE)!=0u) { let slot=atomicAdd(&control[DIRTY_CORE],1u); writeEntry(1u,slot,mapped,leaf); }
  else { let slot=atomicAdd(&control[DIRTY_HALO],1u); writeEntry(2u,slot,mapped,leaf); }
}

@compute @workgroup_size(64)
fn consumeRetired(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32) {
  if((atomicLoad(&control[0])&READY)==0u) { return; }
  let index=sourceIndex(wid,lid,RETIRED_INPUT_DISPATCH); if(index>=min(atomicLoad(&control[6]),params.sourceCapacity)) { return; }
  let word=params.retiredOffsetWords+index*2u; let brick=sourceWorklist[word]; let leaf=sourceWorklist[word+1u];
  var state=0u; if(brick<params.sourceCapacity) { state=states[brick]; } let flags=state&0xffu;
  let valid=(flags&RESIDENT)==0u && (flags&WAS_RESIDENT)!=0u; let attempt=atomicLoad(&control[ATTEMPT]);
  if(!valid || !claimEntry(brick,leaf,attempt)) { atomicAdd(&control[INVALID_COUNT],1u); atomicOr(&control[0],INVALID_ENTRY); return; }
  let mapped=structuralBrick(brick);if(mapped==0xffffffffu){atomicAdd(&control[INVALID_COUNT],1u);atomicOr(&control[0],INVALID_ENTRY);return;}
  let slot=atomicAdd(&control[DIRTY_RETIRED],1u); writeEntry(3u,slot,mapped,leaf);
}

@compute @workgroup_size(1)
fn finalize() {
  var status=atomicLoad(&control[0]); if((status&READY)==0u) { return; }
  let activeWork=atomicLoad(&control[DIRTY_ACTIVE]); let coreWork=atomicLoad(&control[DIRTY_CORE]); let haloWork=atomicLoad(&control[DIRTY_HALO]); let retiredWork=atomicLoad(&control[DIRTY_RETIRED]);
  if((status&INVALID_ENTRY)!=0u || (atomicLoad(&control[3])<=params.sourceCapacity && (coreWork!=atomicLoad(&control[4]) || haloWork!=atomicLoad(&control[5])))) {
    status|=INVALID_ENTRY|COARSE_FALLBACK; atomicStore(&control[0],status); atomicStore(&control[COARSE_COUNT],atomicLoad(&control[3]));
    clearDispatch(ACTIVE_OUTPUT_DISPATCH); clearDispatch(CORE_OUTPUT_DISPATCH); clearDispatch(HALO_OUTPUT_DISPATCH); clearDispatch(RETIRED_OUTPUT_DISPATCH); return;
  }
  let activeExhausted=activeWork-min(activeWork,params.rendererCapacity); let retiredExhausted=retiredWork-min(retiredWork,params.rendererCapacity);
  let exhausted=satAdd(activeExhausted,retiredExhausted); atomicStore(&control[EXHAUSTED_COUNT],exhausted);
  if(exhausted>0u) { status|=RENDERER_EXHAUSTED; }
  let coarse=satAdd(atomicLoad(&control[COARSE_COUNT]),activeExhausted); atomicStore(&control[COARSE_COUNT],coarse); if(coarse>0u) { status|=COARSE_FALLBACK; }
  atomicStore(&control[0],status); atomicStore(&control[ACCEPTED],atomicLoad(&control[1]));
  storeDispatch(ACTIVE_OUTPUT_DISPATCH,min(activeWork,params.rendererCapacity)); storeDispatch(CORE_OUTPUT_DISPATCH,min(coreWork,params.rendererCapacity));
  storeDispatch(HALO_OUTPUT_DISPATCH,min(haloWork,params.rendererCapacity)); storeDispatch(RETIRED_OUTPUT_DISPATCH,min(retiredWork,params.rendererCapacity));
}
`;

export interface WebGPUSvoRenderResidencyConsumerOptions {
  capacity: number;
  /** This consumer intentionally has no incomplete-coverage operating mode. */
  coarseCoverageComplete: true;
}

export class WebGPUSvoRenderResidencyConsumer {
  readonly layout: SvoRenderResidencyConsumerLayout;
  readonly control: GPUBuffer;
  readonly entries: GPUBuffer;
  readonly allocatedBytes: number;
  private readonly params: GPUBuffer;
  private readonly seenAttempt: GPUBuffer;
  private readonly dispatch: GPUBuffer;
  private readonly bindGroup: GPUBindGroup;
  private readonly pipelines: Readonly<Record<"prepare" | "consumeActive" | "consumeRetired" | "finalize", GPUComputePipeline>>;
  private destroyed = false;

  constructor(device: GPUDevice, structural: SparseVoxelStructuralRenderSource, options: WebGPUSvoRenderResidencyConsumerOptions) {
    if (options.coarseCoverageComplete !== true) throw new RangeError("GPU renderer residency requires complete coarse fallback coverage");
    const inputs = buildSvoRenderResidencyGpuInputs(structural);
    const residency = structural.fluidResidency!;
    const sourceCapacity = positiveUint(residency.active.capacity, "Source residency capacity");
    this.layout = svoRenderResidencyConsumerLayout(options.capacity);
    const make = (label: string, size: number, usage: GPUBufferUsageFlags) => device.createBuffer({ label, size, usage });
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    this.control = make("SVO renderer residency counters and indirect dispatches", this.layout.controlByteLength, storage | GPUBufferUsage.INDIRECT);
    this.entries = make("SVO renderer residency compact requests and releases", this.layout.entryByteLength, storage);
    this.seenAttempt = make("SVO renderer residency duplicate stamps", (sourceCapacity + structural.capacities.leaves) * 4, storage);
    this.dispatch = make("SVO renderer residency input dispatch staging", 24, GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST);
    this.params = make("SVO renderer residency consumer parameters", 80, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    const sourceLayout = sparseVoxelFluidResidencyLayout(sourceCapacity);
    const structuralBrickDimensions = structural.domain.dimensionsCells.map((value) => Math.ceil(value / structural.domain.brickSize));
    const sourceOrigin = residency.domain.originBricks, sourceDimensions = residency.domain.dimensionsBricks;
    if (sourceOrigin.some((value, axis) => value + sourceDimensions[axis] > structuralBrickDimensions[axis])) {
      throw new RangeError("Solver residency domain falls outside the structural SVO brick domain");
    }
    device.queue.writeBuffer(this.params, 0, new Uint32Array([
      sourceCapacity, this.layout.capacity, structural.capacities.leaves, sourceLayout.activeEntryOffsetBytes / 4,
      sourceLayout.retiredEntryOffsetBytes / 4, SPARSE_VOXEL_VALID_FIELDS.topology | SPARSE_VOXEL_VALID_FIELDS.coarseFluid, this.layout.listStrideBytes / 4, 0,
      ...sourceOrigin, 0, ...sourceDimensions, 0, ...structuralBrickDimensions, 0,
    ]));
    const shader = device.createShaderModule({ label: "SVO renderer residency consumer", code: svoRenderResidencyConsumerShader });
    const layout = device.createBindGroupLayout({ label: "SVO renderer residency consumer bindings", entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    ] });
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
    const pipeline = (entryPoint: keyof typeof this.pipelines) => device.createComputePipeline({
      label: `SVO renderer residency ${entryPoint}`,
      layout: pipelineLayout,
      compute: { module: shader, entryPoint },
    });
    this.pipelines = {
      prepare: pipeline("prepare"), consumeActive: pipeline("consumeActive"),
      consumeRetired: pipeline("consumeRetired"), finalize: pipeline("finalize"),
    };
    this.bindGroup = device.createBindGroup({ label: "SVO renderer residency consumer bindings", layout, entries: [
      ...inputs.bindGroupEntries,
      { binding: 3, resource: { buffer: this.params } },
      { binding: 4, resource: { buffer: this.control } },
      { binding: 5, resource: { buffer: this.entries } },
      { binding: 6, resource: { buffer: this.seenAttempt } },
    ] });
    this.allocatedBytes = this.layout.controlByteLength + this.layout.entryByteLength + (sourceCapacity + structural.capacities.leaves) * 4 + 80 + 24;
  }

  /** Encode GPU fence validation, bounded compaction, and output dispatch preparation. */
  encode(encoder: GPUCommandEncoder): void {
    if (this.destroyed) return;
    const run = (label: string, callback: (pass: GPUComputePassEncoder) => void) => {
      const pass = encoder.beginComputePass({ label }); callback(pass); pass.end();
    };
    run("SVO renderer residency prepare", (pass) => {
      pass.setPipeline(this.pipelines.prepare); pass.setBindGroup(0, this.bindGroup); pass.dispatchWorkgroups(1);
    });
    encoder.copyBufferToBuffer(this.control, this.layout.indirectOffsetsBytes.activeInput, this.dispatch, 0, 12);
    encoder.copyBufferToBuffer(this.control, this.layout.indirectOffsetsBytes.retiredInput, this.dispatch, 12, 12);
    run("SVO renderer residency compact", (pass) => {
      pass.setBindGroup(0, this.bindGroup);
      pass.setPipeline(this.pipelines.consumeActive); pass.dispatchWorkgroupsIndirect(this.dispatch, 0);
      pass.setPipeline(this.pipelines.consumeRetired); pass.dispatchWorkgroupsIndirect(this.dispatch, 12);
    });
    run("SVO renderer residency finalize", (pass) => {
      pass.setPipeline(this.pipelines.finalize); pass.setBindGroup(0, this.bindGroup); pass.dispatchWorkgroups(1);
    });
  }

  binding(list: SvoRenderResidencyConsumerList): GPUBufferBinding {
    return { buffer: this.entries, offset: this.layout.entryOffsetsBytes[list], size: this.layout.capacity * 8 };
  }

  indirectOffset(list: SvoRenderResidencyConsumerList): number {
    return this.layout.indirectOffsetsBytes[`${list}Output`];
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.params.destroy(); this.dispatch.destroy(); this.control.destroy(); this.entries.destroy(); this.seenAttempt.destroy();
  }
}
