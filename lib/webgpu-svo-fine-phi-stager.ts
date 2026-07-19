/**
 * GPU-only renderer staging for authoritative sparse-surface phi.
 *
 * The solver-owned sparse surface band is immutable here. Renderer residency
 * selects coarse 8^3 owner bricks; each resident owner slot receives a fine
 * `(8 * refinement + 2)^3` tile with a one-sample apron. A page-generation
 * stamp is published only after every source page needed by the tile exists.
 * Missing or stale data therefore remains an explicit coarse-field fallback.
 */
import {
  SVO_OWNER_PAGE_CONTROL_WORDS,
  SVO_OWNER_PAGE_STATUS,
  type OctreeOwnerPagePlan,
  type WebGPUSvoOwnerPageAllocator,
} from "./webgpu-octree-owner-pages";
import {
  SVO_RENDER_RESIDENCY_CONSUMER_CONTROL_WORDS,
  type WebGPUSvoRenderResidencyConsumer,
} from "./webgpu-svo-render-residency-consumer";
import {
  SPARSE_SURFACE_INVALID_PAGE,
  type SparseSurfaceBandGPUSource,
} from "./webgpu-sparse-surface-band";
import { SPARSE_VOXEL_PUBLICATION_STATE } from "./webgpu-voxel-debug";

const UINT32_MAX = 0xffff_ffff;
const FINE_ARENA_ALIGNMENT_WORDS = 64;
const FINE_AIR_PHI_M = 1_000_000;

export const SVO_FINE_PHI_STATUS = Object.freeze({
  ready: 1 << 0,
  unchanged: 1 << 1,
  stale: 1 << 2,
  sourceRejected: 1 << 3,
  partial: 1 << 4,
} as const);

export const SVO_FINE_PHI_CONTROL_WORDS = Object.freeze({
  status: 16,
  observedStructuralGeneration: 17,
  observedFineGeneration: 18,
  acceptedStructuralGeneration: 19,
  acceptedFineGeneration: 20,
  stagedPageCount: 21,
  fallbackPageCount: 22,
  missingSourceSampleCount: 23,
  retiredClearedPageCount: 24,
  sourceRejectedCount: 25,
  stalePublicationCount: 26,
  unchangedPublicationCount: 27,
  length: 16,
} as const);

export interface SvoFinePhiStagingPlan {
  ownerDimensions: readonly [number, number, number];
  ownerBrickDimensions: readonly [number, number, number];
  fineDimensions: readonly [number, number, number];
  sourceFineDimensions: readonly [number, number, number];
  sourceOriginFine: readonly [number, number, number];
  fineCellSize_m: readonly [number, number, number];
  refinementFactor: 1 | 2 | 4;
  ownerBrickSize: 8;
  sourceBrickSize: 4 | 8;
  tileEdge: number;
  tileVoxels: number;
  requestedCapacity: number;
  capacity: number;
  degraded: boolean;
  controlOffsetWords: 16;
  pageGenerationOffsetWords: number;
  ownerPageTableOffsetWords: number;
  payloadOffsetWords: number;
  allocatedWords: number;
  allocatedBytes: number;
}

export interface SvoFinePhiStagingPlanOptions {
  maximumArenaBytes?: number;
  /** Solver-brick origin inside the larger structural SVO domain. */
  sourceOriginBricks?: readonly [number, number, number];
}

function alignWords(words: number): number {
  return Math.ceil(words / FINE_ARENA_ALIGNMENT_WORDS) * FINE_ARENA_ALIGNMENT_WORDS;
}

function positiveFinite(value: number, label: string): number {
  if (!(value > 0) || !Number.isFinite(value)) throw new RangeError(`${label} must be positive and finite`);
  return value;
}

/** Capacity and aligned storage ABI for renderer fine-phi tiles. */
export function planSvoFinePhiStaging(
  owner: OctreeOwnerPagePlan,
  source: Pick<SparseSurfaceBandGPUSource, "fineDimensions" | "brickDimensions" | "brickSize" | "refinementFactor">,
  coarseCellSize_m: readonly [number, number, number],
  options: SvoFinePhiStagingPlanOptions = {},
): SvoFinePhiStagingPlan {
  if (owner.brickSize !== 8) throw new RangeError("Fine-phi staging requires 8-cubed renderer owner bricks");
  coarseCellSize_m.forEach((value, axis) => positiveFinite(value, `Fine-phi coarse cell size axis ${axis}`));
  const factor = source.refinementFactor;
  if (factor !== 1 && factor !== 2 && factor !== 4) throw new RangeError("Fine-phi refinement must be 1, 2, or 4");
  const fineDimensions = owner.dimensions.map((value) => value * factor) as [number, number, number];
  const sourceOriginBricks = options.sourceOriginBricks ?? [0, 0, 0];
  sourceOriginBricks.forEach((value, axis) => {
    if (!Number.isSafeInteger(value) || value < 0) throw new RangeError(`Fine-phi source brick origin axis ${axis} must be a non-negative integer`);
  });
  const sourceCoarseDimensions = source.fineDimensions.map((value, axis) => {
    if (!Number.isSafeInteger(value) || value < 1 || value % factor !== 0) throw new RangeError(`Sparse-surface fine dimension axis ${axis} is not refinement-aligned`);
    return value / factor;
  });
  const sourceOriginCells = sourceOriginBricks.map((value) => value * owner.brickSize);
  if (sourceCoarseDimensions.some((value, axis) => sourceOriginCells[axis] + value > owner.dimensions[axis])) {
    throw new RangeError("Sparse-surface fine domain falls outside the renderer owner domain");
  }
  const sourceOriginFine = sourceOriginCells.map((value) => value * factor) as [number, number, number];
  const expectedSourceBricks = source.fineDimensions.map((value) => Math.ceil(value / source.brickSize));
  if (source.brickDimensions.some((value, axis) => value !== expectedSourceBricks[axis])) {
    throw new RangeError("Sparse-surface brick dimensions are inconsistent with its fine domain");
  }
  const tileEdge = owner.brickSize * factor + 2;
  const tileVoxels = tileEdge ** 3;
  const pageGenerationOffsetWords = FINE_ARENA_ALIGNMENT_WORDS;
  const ownerPageTableOffsetWords = pageGenerationOffsetWords + owner.logicalBrickCount;
  const payloadOffsetWords = alignWords(ownerPageTableOffsetWords + owner.logicalBrickCount);
  const maximumBytes = options.maximumArenaBytes ?? Number.MAX_SAFE_INTEGER;
  if (!Number.isFinite(maximumBytes) || maximumBytes < 0) throw new RangeError("Fine-phi arena byte ceiling must be finite and non-negative");
  const availablePayloadWords = Math.max(0, Math.floor(maximumBytes / 4) - payloadOffsetWords);
  const capacity = Math.min(owner.capacity, Math.floor(availablePayloadWords / tileVoxels));
  if (capacity < 1) throw new RangeError("Fine-phi arena byte ceiling cannot hold one apron tile");
  const allocatedWords = payloadOffsetWords + capacity * tileVoxels;
  return {
    ownerDimensions: [...owner.dimensions], ownerBrickDimensions: [...owner.brickDimensions],
    fineDimensions, sourceFineDimensions: [...source.fineDimensions], sourceOriginFine,
    fineCellSize_m: coarseCellSize_m.map((value) => value / factor) as [number, number, number],
    refinementFactor: factor, ownerBrickSize: 8, sourceBrickSize: source.brickSize,
    tileEdge, tileVoxels, requestedCapacity: owner.capacity, capacity,
    degraded: capacity < owner.capacity, controlOffsetWords: 16,
    pageGenerationOffsetWords, ownerPageTableOffsetWords, payloadOffsetWords, allocatedWords, allocatedBytes: allocatedWords * 4,
  };
}

export interface SvoFinePhiFenceInput {
  structuralGeneration: number;
  fineGeneration: number;
  sourceFineGeneration: number;
  ownerReady: boolean;
}

/** Deterministic CPU oracle for the two-generation fine capability fence. */
export class SvoFinePhiPublicationMirror {
  acceptedStructuralGeneration = 0;
  acceptedFineGeneration = 0;

  publish(input: SvoFinePhiFenceInput): number {
    for (const [label, value] of Object.entries(input).filter(([, value]) => typeof value === "number")) {
      if (!Number.isSafeInteger(value) || (value as number) < 0 || (value as number) > UINT32_MAX) {
        throw new RangeError(`Fine-phi ${label} must fit uint32`);
      }
    }
    if (!input.ownerReady || input.structuralGeneration === 0 || input.fineGeneration === 0
      || input.sourceFineGeneration !== input.fineGeneration) return SVO_FINE_PHI_STATUS.sourceRejected;
    if (input.structuralGeneration < this.acceptedStructuralGeneration || input.fineGeneration < this.acceptedFineGeneration) {
      return SVO_FINE_PHI_STATUS.stale;
    }
    if (input.structuralGeneration === this.acceptedStructuralGeneration && input.fineGeneration === this.acceptedFineGeneration) {
      return SVO_FINE_PHI_STATUS.unchanged;
    }
    this.acceptedStructuralGeneration = input.structuralGeneration;
    this.acceptedFineGeneration = input.fineGeneration;
    return SVO_FINE_PHI_STATUS.ready;
  }
}

/** Dense CPU oracle matching the apron-safe manual trilinear convention. */
export function sampleFinePhiReference(
  values: Float32Array,
  dimensions: readonly [number, number, number],
  position: readonly [number, number, number],
): number {
  if (values.length < dimensions[0] * dimensions[1] * dimensions[2]) throw new RangeError("Fine-phi reference payload is undersized");
  const p = position.map((value, axis) => Math.max(0, Math.min(dimensions[axis] - 1, value)));
  const a = p.map(Math.floor), b = a.map((value, axis) => Math.min(dimensions[axis] - 1, value + 1));
  const t = p.map((value, axis) => value - a[axis]);
  const at = (x: number, y: number, z: number) => values[x + dimensions[0] * (y + dimensions[1] * z)];
  const mix = (x: number, y: number, amount: number) => x + (y - x) * amount;
  const z0 = mix(mix(at(a[0], a[1], a[2]), at(b[0], a[1], a[2]), t[0]), mix(at(a[0], b[1], a[2]), at(b[0], b[1], a[2]), t[0]), t[1]);
  const z1 = mix(mix(at(a[0], a[1], b[2]), at(b[0], a[1], b[2]), t[0]), mix(at(a[0], b[1], b[2]), at(b[0], b[1], b[2]), t[0]), t[1]);
  return mix(z0, z1, t[2]);
}

export function gradientFinePhiReference(
  values: Float32Array,
  dimensions: readonly [number, number, number],
  position: readonly [number, number, number],
  cellSize_m: readonly [number, number, number],
): readonly [number, number, number] {
  return [0, 1, 2].map((axis) => {
    const minus = [...position] as [number, number, number], plus = [...position] as [number, number, number];
    minus[axis] -= 1; plus[axis] += 1;
    return (sampleFinePhiReference(values, dimensions, plus) - sampleFinePhiReference(values, dimensions, minus)) / (2 * cellSize_m[axis]);
  }) as [number, number, number];
}

export const svoFinePhiStagingShader = /* wgsl */ `
struct SurfaceParams { coarseDims:vec4u, fineDims:vec4u, brickDims:vec4u, settings:vec4f, cellAndDt:vec4f, sizing:vec4f, physical:vec4f }
struct FineParams {
  ownerDims:vec4u, fineDims:vec4u, ownerBrickDims:vec4u, sourceBrickDims:vec4u,
  counts:vec4u, offsets:vec4u, source:vec4u, sourceFineDims:vec4u, sourceOriginFine:vec4u, fineCell:vec4f,
}
@group(0) @binding(0) var<storage,read> ownerArena:array<u32>;
@group(0) @binding(1) var<storage,read> retiredSlots:array<u32>;
@group(0) @binding(2) var<storage,read> surfaceControl:array<u32>;
@group(0) @binding(3) var<storage,read> surfacePageTable:array<u32>;
@group(0) @binding(4) var<storage,read> surfacePhi:array<f32>;
@group(0) @binding(5) var<storage,read> residencyControl:array<u32>;
@group(0) @binding(6) var<storage,read> residencyEntries:array<u32>;
@group(0) @binding(7) var<storage,read_write> fineArena:array<atomic<u32>>;
@group(0) @binding(8) var<uniform> surfaceParams:SurfaceParams;
@group(0) @binding(9) var<uniform> params:FineParams;

const READY:u32=${SVO_FINE_PHI_STATUS.ready}u; const UNCHANGED:u32=${SVO_FINE_PHI_STATUS.unchanged}u;
const STALE:u32=${SVO_FINE_PHI_STATUS.stale}u; const SOURCE_REJECTED:u32=${SVO_FINE_PHI_STATUS.sourceRejected}u;
const PARTIAL:u32=${SVO_FINE_PHI_STATUS.partial}u; const INVALID:u32=${SPARSE_SURFACE_INVALID_PAGE}u;
const AIR_PHI:f32=${FINE_AIR_PHI_M}.0;

fn entry(listWords:u32,item:u32)->u32 { return residencyEntries[listWords+item*2u]; }
fn ownerBrick(logical:u32)->vec3u { return vec3u(logical%params.ownerBrickDims.x,(logical/params.ownerBrickDims.x)%params.ownerBrickDims.y,logical/(params.ownerBrickDims.x*params.ownerBrickDims.y)); }
fn sourceSlot(q:vec3i)->vec2u {
  let clamped=vec3u(clamp(q,vec3i(0),vec3i(params.sourceFineDims.xyz)-vec3i(1)));
  let brick=clamped/params.source.x; let logical=brick.x+params.sourceBrickDims.x*(brick.y+params.sourceBrickDims.y*brick.z);
  if(logical>=arrayLength(&surfacePageTable)){return vec2u(INVALID,0u);}
  let slot=surfacePageTable[logical]; if(slot>=params.counts.w){return vec2u(INVALID,0u);}
  let local=clamped-brick*params.source.x; let localIndex=local.x+params.source.x*(local.y+params.source.x*local.z);
  return vec2u(slot,localIndex);
}
fn sourcePhiAt(q:vec3i)->vec2f {
  let address=sourceSlot(q); if(address.x==INVALID){return vec2f(AIR_PHI,0.0);}
  let voxels=params.source.x*params.source.x*params.source.x; let word=address.x*voxels+address.y;
  if(word>=arrayLength(&surfacePhi)){return vec2f(AIR_PHI,0.0);} let value=surfacePhi[word];
  return vec2f(value,select(0.0,1.0,value==value&&abs(value)<3.402823e38));
}
fn tileCoordinate(index:u32)->vec3u { return vec3u(index%params.source.w,(index/params.source.w)%params.source.w,index/(params.source.w*params.source.w)); }

@compute @workgroup_size(1)
fn prepare() {
  for(var i=${SVO_FINE_PHI_CONTROL_WORDS.status}u;i<${SVO_FINE_PHI_CONTROL_WORDS.status + 12}u;i+=1u){if(i!=${SVO_FINE_PHI_CONTROL_WORDS.acceptedStructuralGeneration}u&&i!=${SVO_FINE_PHI_CONTROL_WORDS.acceptedFineGeneration}u&&i!=${SVO_FINE_PHI_CONTROL_WORDS.sourceRejectedCount}u&&i!=${SVO_FINE_PHI_CONTROL_WORDS.stalePublicationCount}u&&i!=${SVO_FINE_PHI_CONTROL_WORDS.unchangedPublicationCount}u){atomicStore(&fineArena[i],0u);}}
  let structural=residencyControl[${SVO_RENDER_RESIDENCY_CONSUMER_CONTROL_WORDS.acceptedGeneration}u];
  let ownerGeneration=ownerArena[${SVO_OWNER_PAGE_CONTROL_WORDS.acceptedGeneration}u]; let fineGeneration=surfaceControl[1u];
  atomicStore(&fineArena[${SVO_FINE_PHI_CONTROL_WORDS.observedStructuralGeneration}],structural); atomicStore(&fineArena[${SVO_FINE_PHI_CONTROL_WORDS.observedFineGeneration}],fineGeneration);
  let ownerStatus=ownerArena[${SVO_OWNER_PAGE_CONTROL_WORDS.status}u];
  let sourceShape=all(surfaceParams.fineDims.xyz==params.sourceFineDims.xyz)&&all(surfaceParams.brickDims.xyz==params.sourceBrickDims.xyz)
    &&surfaceParams.fineDims.w==params.source.x;
  if(structural==0u||ownerGeneration!=structural||(ownerStatus&${SVO_OWNER_PAGE_STATUS.ready | SVO_OWNER_PAGE_STATUS.unchanged}u)==0u
    ||fineGeneration==0u||fineGeneration!=params.source.y||!sourceShape){atomicStore(&fineArena[${SVO_FINE_PHI_CONTROL_WORDS.status}],SOURCE_REJECTED);atomicAdd(&fineArena[${SVO_FINE_PHI_CONTROL_WORDS.sourceRejectedCount}],1u);return;}
  let acceptedStructural=atomicLoad(&fineArena[${SVO_FINE_PHI_CONTROL_WORDS.acceptedStructuralGeneration}]);let acceptedFine=atomicLoad(&fineArena[${SVO_FINE_PHI_CONTROL_WORDS.acceptedFineGeneration}]);
  if(structural<acceptedStructural||fineGeneration<acceptedFine){atomicStore(&fineArena[${SVO_FINE_PHI_CONTROL_WORDS.status}],STALE);atomicAdd(&fineArena[${SVO_FINE_PHI_CONTROL_WORDS.stalePublicationCount}],1u);return;}
  if(structural==acceptedStructural&&fineGeneration==acceptedFine){atomicStore(&fineArena[${SVO_FINE_PHI_CONTROL_WORDS.status}],UNCHANGED);atomicAdd(&fineArena[${SVO_FINE_PHI_CONTROL_WORDS.unchangedPublicationCount}],1u);return;}
  atomicStore(&fineArena[${SVO_FINE_PHI_CONTROL_WORDS.status}],READY);
}

fn dispatchedItem(wid:vec3u,lid:u32,dispatchWord:u32)->u32{return (wid.y*residencyControl[dispatchWord]+wid.x)*64u+lid;}
@compute @workgroup_size(64)
fn clearRetired(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32) {
  let item=dispatchedItem(wid,lid,${SVO_RENDER_RESIDENCY_CONSUMER_CONTROL_WORDS.retiredOutputDispatch}u);let count=min(residencyControl[${SVO_RENDER_RESIDENCY_CONSUMER_CONTROL_WORDS.dirtyRetiredCount}u],params.counts.x);
  if(item>=count||item>=arrayLength(&retiredSlots)){return;} let logical=entry(params.sourceBrickDims.w,item);
  if(logical<params.counts.x){atomicStore(&fineArena[params.offsets.y+logical],0u);} let slot=retiredSlots[item];
  if(slot>=params.counts.z){return;} let base=params.offsets.w+slot*params.source.w*params.source.w*params.source.w;
  let voxels=params.source.w*params.source.w*params.source.w;for(var local=0u;local<voxels;local+=1u){atomicStore(&fineArena[base+local],bitcast<u32>(AIR_PHI));}
  atomicAdd(&fineArena[${SVO_FINE_PHI_CONTROL_WORDS.retiredClearedPageCount}],1u);
}

@compute @workgroup_size(64)
fn stageActive(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32) {
  if((atomicLoad(&fineArena[${SVO_FINE_PHI_CONTROL_WORDS.status}])&READY)==0u){return;} let item=dispatchedItem(wid,lid,${SVO_RENDER_RESIDENCY_CONSUMER_CONTROL_WORDS.activeOutputDispatch}u);
  let count=min(residencyControl[${SVO_RENDER_RESIDENCY_CONSUMER_CONTROL_WORDS.dirtyActiveCount}u],params.counts.x);if(item>=count){return;}
  let logical=entry(params.source.z,item);if(logical>=params.counts.x){atomicAdd(&fineArena[${SVO_FINE_PHI_CONTROL_WORDS.fallbackPageCount}],1u);atomicOr(&fineArena[${SVO_FINE_PHI_CONTROL_WORDS.status}],PARTIAL);return;}
  let encoded=ownerArena[params.offsets.x+logical];atomicStore(&fineArena[params.offsets.z+logical],encoded);if(encoded==0u||encoded>params.counts.y||encoded-1u>=params.counts.z){atomicStore(&fineArena[params.offsets.y+logical],0u);atomicAdd(&fineArena[${SVO_FINE_PHI_CONTROL_WORDS.fallbackPageCount}],1u);atomicOr(&fineArena[${SVO_FINE_PHI_CONTROL_WORDS.status}],PARTIAL);return;}
  let slot=encoded-1u;let brick=ownerBrick(logical);let edge=params.source.w;let voxels=edge*edge*edge;let base=params.offsets.w+slot*voxels;var missing=0u;
  for(var local=0u;local<voxels;local+=1u){let c=tileCoordinate(local);let global=vec3i(brick*params.ownerDims.w*params.fineDims.w+c)-vec3i(1)-vec3i(params.sourceOriginFine.xyz);let sample=sourcePhiAt(global);atomicStore(&fineArena[base+local],bitcast<u32>(sample.x));if(sample.y==0.0){missing+=1u;}}
  if(missing==0u){atomicStore(&fineArena[params.offsets.y+logical],params.source.y);atomicAdd(&fineArena[${SVO_FINE_PHI_CONTROL_WORDS.stagedPageCount}],1u);}
  else{atomicStore(&fineArena[params.offsets.y+logical],0u);atomicAdd(&fineArena[${SVO_FINE_PHI_CONTROL_WORDS.fallbackPageCount}],1u);atomicAdd(&fineArena[${SVO_FINE_PHI_CONTROL_WORDS.missingSourceSampleCount}],missing);atomicOr(&fineArena[${SVO_FINE_PHI_CONTROL_WORDS.status}],PARTIAL);}
}

@compute @workgroup_size(1)
fn finalize() {
  if((atomicLoad(&fineArena[${SVO_FINE_PHI_CONTROL_WORDS.status}])&READY)==0u){return;}
  atomicStore(&fineArena[${SVO_FINE_PHI_CONTROL_WORDS.acceptedStructuralGeneration}],atomicLoad(&fineArena[${SVO_FINE_PHI_CONTROL_WORDS.observedStructuralGeneration}]));atomicStore(&fineArena[${SVO_FINE_PHI_CONTROL_WORDS.acceptedFineGeneration}],atomicLoad(&fineArena[${SVO_FINE_PHI_CONTROL_WORDS.observedFineGeneration}]));
}
`;

/** Composable read-only sampler. Invalid samples deliberately require coarse fallback. */
export function svoFinePhiSamplingWGSL(group: number, ownerBinding: number, fineBinding: number, paramsBinding: number): string {
  return /* wgsl */ `
struct SvoFinePhiParams { ownerDims:vec4u, fineDims:vec4u, ownerBrickDims:vec4u, sourceBrickDims:vec4u, counts:vec4u, offsets:vec4u, source:vec4u, sourceFineDims:vec4u, sourceOriginFine:vec4u, fineCell:vec4f }
struct SvoFinePhiSample { phi_m:f32, valid:u32 }
struct SvoFinePhiGradient { gradient:vec3f, valid:u32 }
@group(${group}) @binding(${ownerBinding}) var<storage,read> svoFineOwnerArena:array<u32>;
@group(${group}) @binding(${fineBinding}) var<storage,read> svoFineArena:array<u32>;
@group(${group}) @binding(${paramsBinding}) var<uniform> svoFineParams:SvoFinePhiParams;
fn svoFineTileCoordinate(position:vec3f,brick:vec3u)->vec3f{return position-vec3f(brick*svoFineParams.ownerDims.w*svoFineParams.fineDims.w)+vec3f(1.0);}
fn svoFineLoad(slot:u32,q:vec3u)->f32{let edge=svoFineParams.source.w;return bitcast<f32>(svoFineArena[svoFineParams.offsets.w+slot*edge*edge*edge+q.x+edge*(q.y+edge*q.z)]);}
fn svoFinePhi(position:vec3f,expectedStructural:u32)->SvoFinePhiSample{
  let status=svoFineArena[${SVO_FINE_PHI_CONTROL_WORDS.status}u];let fineGeneration=svoFineArena[${SVO_FINE_PHI_CONTROL_WORDS.acceptedFineGeneration}u];
  if((status&${SVO_FINE_PHI_STATUS.ready}u)==0u||expectedStructural==0u||svoFineArena[${SVO_FINE_PHI_CONTROL_WORDS.acceptedStructuralGeneration}u]!=expectedStructural){return SvoFinePhiSample(0.0,0u);}
  let p=clamp(position,vec3f(0.0),vec3f(svoFineParams.fineDims.xyz-vec3u(1)));let brick=min(vec3u(floor(p/f32(svoFineParams.ownerDims.w*svoFineParams.fineDims.w))),svoFineParams.ownerBrickDims.xyz-vec3u(1));
  let logical=brick.x+svoFineParams.ownerBrickDims.x*(brick.y+svoFineParams.ownerBrickDims.y*brick.z);if(svoFineArena[svoFineParams.offsets.y+logical]!=fineGeneration){return SvoFinePhiSample(0.0,0u);}
  let encoded=svoFineOwnerArena[svoFineParams.offsets.x+logical];if(encoded==0u||encoded>svoFineParams.counts.z){return SvoFinePhiSample(0.0,0u);}let slot=encoded-1u;
  let local=svoFineTileCoordinate(p,brick);let a=vec3u(floor(local));let b=a+vec3u(1);let t=fract(local);
  let z0=mix(mix(svoFineLoad(slot,vec3u(a.x,a.y,a.z)),svoFineLoad(slot,vec3u(b.x,a.y,a.z)),t.x),mix(svoFineLoad(slot,vec3u(a.x,b.y,a.z)),svoFineLoad(slot,vec3u(b.x,b.y,a.z)),t.x),t.y);
  let z1=mix(mix(svoFineLoad(slot,vec3u(a.x,a.y,b.z)),svoFineLoad(slot,vec3u(b.x,a.y,b.z)),t.x),mix(svoFineLoad(slot,vec3u(a.x,b.y,b.z)),svoFineLoad(slot,vec3u(b.x,b.y,b.z)),t.x),t.y);let value=mix(z0,z1,t.z);
  return SvoFinePhiSample(value,select(0u,1u,value==value&&abs(value)<3.402823e38));
}
fn svoFinePhiGradient(position:vec3f,expectedStructural:u32)->SvoFinePhiGradient{
  let xm=svoFinePhi(position-vec3f(1,0,0),expectedStructural);let xp=svoFinePhi(position+vec3f(1,0,0),expectedStructural);
  let ym=svoFinePhi(position-vec3f(0,1,0),expectedStructural);let yp=svoFinePhi(position+vec3f(0,1,0),expectedStructural);
  let zm=svoFinePhi(position-vec3f(0,0,1),expectedStructural);let zp=svoFinePhi(position+vec3f(0,0,1),expectedStructural);
  if((xm.valid&xp.valid&ym.valid&yp.valid&zm.valid&zp.valid)==0u){return SvoFinePhiGradient(vec3f(0.0),0u);}
  return SvoFinePhiGradient(vec3f((xp.phi_m-xm.phi_m)/(2.0*svoFineParams.fineCell.x),(yp.phi_m-ym.phi_m)/(2.0*svoFineParams.fineCell.y),(zp.phi_m-zm.phi_m)/(2.0*svoFineParams.fineCell.z)),1u);
}
`;
}

/** Types embedded in DryParams without adding a fragment-stage binding. */
export const svoFinePhiTypesWGSL = /* wgsl */ `
struct SvoFinePhiParams { ownerDims:vec4u, fineDims:vec4u, ownerBrickDims:vec4u, sourceBrickDims:vec4u, counts:vec4u, offsets:vec4u, source:vec4u, sourceFineDims:vec4u, sourceOriginFine:vec4u, fineCell:vec4f }
struct SvoFinePhiSample { phi_m:f32, valid:u32 }
struct SvoFinePhiGradient { gradient:vec3f, valid:u32 }
`;

/** Read fine phi from the combined publication/fine arena used by dry rendering. */
export function svoFinePhiPackedSamplingWGSL(arena = "publicationState", params = "dry.finePhi"): string {
  for (const value of [arena, ...params.split(".")]) {
    if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(value)) throw new RangeError(`Invalid fine-phi WGSL identifier: ${value}`);
  }
  return /* wgsl */ `
fn svoFinePackedTileCoordinate(position:vec3f,brick:vec3u)->vec3f{return position-vec3f(brick*${params}.ownerDims.w*${params}.fineDims.w)+vec3f(1.0);}
fn svoFinePackedLoad(slot:u32,q:vec3u)->f32{let edge=${params}.source.w;return bitcast<f32>(${arena}[${params}.offsets.w+slot*edge*edge*edge+q.x+edge*(q.y+edge*q.z)]);}
fn svoFinePackedPhi(position:vec3f,expectedStructural:u32)->SvoFinePhiSample{
  let status=${arena}[${SVO_FINE_PHI_CONTROL_WORDS.status}u];let fineGeneration=${arena}[${SVO_FINE_PHI_CONTROL_WORDS.acceptedFineGeneration}u];
  if((status&${SVO_FINE_PHI_STATUS.ready}u)==0u||expectedStructural==0u||${arena}[${SVO_FINE_PHI_CONTROL_WORDS.acceptedStructuralGeneration}u]!=expectedStructural){return SvoFinePhiSample(0.0,0u);}
  let p=clamp(position,vec3f(0.0),vec3f(${params}.fineDims.xyz-vec3u(1)));let brick=min(vec3u(floor(p/f32(${params}.ownerDims.w*${params}.fineDims.w))),${params}.ownerBrickDims.xyz-vec3u(1));
  let logical=brick.x+${params}.ownerBrickDims.x*(brick.y+${params}.ownerBrickDims.y*brick.z);if(${arena}[${params}.offsets.y+logical]!=fineGeneration){return SvoFinePhiSample(0.0,0u);}
  let encoded=${arena}[${params}.offsets.z+logical];if(encoded==0u||encoded>${params}.counts.z){return SvoFinePhiSample(0.0,0u);}let slot=encoded-1u;
  let local=svoFinePackedTileCoordinate(p,brick);let a=vec3u(floor(local));let b=a+vec3u(1);let t=fract(local);
  let z0=mix(mix(svoFinePackedLoad(slot,vec3u(a.x,a.y,a.z)),svoFinePackedLoad(slot,vec3u(b.x,a.y,a.z)),t.x),mix(svoFinePackedLoad(slot,vec3u(a.x,b.y,a.z)),svoFinePackedLoad(slot,vec3u(b.x,b.y,a.z)),t.x),t.y);
  let z1=mix(mix(svoFinePackedLoad(slot,vec3u(a.x,a.y,b.z)),svoFinePackedLoad(slot,vec3u(b.x,a.y,b.z)),t.x),mix(svoFinePackedLoad(slot,vec3u(a.x,b.y,b.z)),svoFinePackedLoad(slot,vec3u(b.x,b.y,b.z)),t.x),t.y);let value=mix(z0,z1,t.z);
  return SvoFinePhiSample(value,select(0u,1u,value==value&&abs(value)<3.402823e38));
}
fn svoFinePackedGradient(position:vec3f,expectedStructural:u32)->SvoFinePhiGradient{
  let xm=svoFinePackedPhi(position-vec3f(1,0,0),expectedStructural);let xp=svoFinePackedPhi(position+vec3f(1,0,0),expectedStructural);let ym=svoFinePackedPhi(position-vec3f(0,1,0),expectedStructural);let yp=svoFinePackedPhi(position+vec3f(0,1,0),expectedStructural);let zm=svoFinePackedPhi(position-vec3f(0,0,1),expectedStructural);let zp=svoFinePackedPhi(position+vec3f(0,0,1),expectedStructural);
  if((xm.valid&xp.valid&ym.valid&yp.valid&zm.valid&zp.valid)==0u){return SvoFinePhiGradient(vec3f(0.0),0u);}return SvoFinePhiGradient(vec3f((xp.phi_m-xm.phi_m)/(2.0*${params}.fineCell.x),(yp.phi_m-ym.phi_m)/(2.0*${params}.fineCell.y),(zp.phi_m-zm.phi_m)/(2.0*${params}.fineCell.z)),1u);
}
`;
}

export interface SvoFineFluidGpuCapability {
  readonly arena: GPUBufferBinding;
  readonly params: GPUBufferBinding;
  readonly statusWord: typeof SVO_FINE_PHI_CONTROL_WORDS.status;
  readonly acceptedStructuralGenerationWord: typeof SVO_FINE_PHI_CONTROL_WORDS.acceptedStructuralGeneration;
  readonly acceptedFineGenerationWord: typeof SVO_FINE_PHI_CONTROL_WORDS.acceptedFineGeneration;
  readonly pageGenerationOffsetWords: number;
  readonly ownerPageTableOffsetWords: number;
  readonly payloadOffsetWords: number;
  readonly paramsWords: Uint32Array<ArrayBuffer>;
  readonly publicationMirrorWords: 8;
  readonly coarseFallbackRequired: true;
  readonly directWaterOwnership: false;
}

export class WebGPUSvoFinePhiStager {
  readonly plan: SvoFinePhiStagingPlan;
  readonly arena: GPUBuffer;
  readonly allocatedBytes: number;
  private readonly params: GPUBuffer;
  private readonly dispatch: GPUBuffer;
  private readonly residencyControlBuffer: GPUBuffer;
  private readonly bindGroup: GPUBindGroup;
  private readonly pipelines: Readonly<Record<"prepare" | "clearRetired" | "stageActive" | "finalize", GPUComputePipeline>>;
  private readonly paramWords: Uint32Array<ArrayBuffer>;
  private readonly structuralPublication?: GPUBufferBinding;
  private destroyed = false;

  constructor(
    private readonly device: GPUDevice,
    owner: WebGPUSvoOwnerPageAllocator,
    residency: WebGPUSvoRenderResidencyConsumer,
    source: SparseSurfaceBandGPUSource,
    coarseCellSize_m: readonly [number, number, number],
    options: SvoFinePhiStagingPlanOptions & { structuralPublication?: GPUBufferBinding } = {},
  ) {
    if (source.mode !== "authoritative") throw new RangeError("Renderer fine-phi staging requires an authoritative sparse-surface source");
    const maximumArenaBytes = Math.min(options.maximumArenaBytes ?? Number.MAX_SAFE_INTEGER,
      Number(device.limits.maxStorageBufferBindingSize), Number(device.limits.maxBufferSize));
    this.plan = planSvoFinePhiStaging(owner.plan, source, coarseCellSize_m, {
      maximumArenaBytes, sourceOriginBricks: options.sourceOriginBricks,
    });
    this.structuralPublication = options.structuralPublication;
    this.residencyControlBuffer = residency.control;
    const storageCopy = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    this.arena = device.createBuffer({ label: "SVO renderer fine-phi arena", size: this.plan.allocatedBytes, usage: storageCopy });
    this.params = device.createBuffer({ label: "SVO renderer fine-phi parameters", size: 160, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.dispatch = device.createBuffer({ label: "SVO renderer fine-phi dispatch staging", size: 24, usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST });
    this.paramWords = new Uint32Array(new ArrayBuffer(160));
    const p = this.plan;
    this.paramWords.set([...p.ownerDimensions, p.ownerBrickSize], 0);
    this.paramWords.set([...p.fineDimensions, p.refinementFactor], 4);
    this.paramWords.set([...p.ownerBrickDimensions, owner.plan.logicalBrickCount], 8);
    this.paramWords.set([...source.brickDimensions, residency.layout.entryOffsetsBytes.retired / 4], 12);
    this.paramWords.set([owner.plan.logicalBrickCount, owner.plan.capacity, p.capacity, source.pageCapacity], 16);
    this.paramWords.set([owner.plan.pageTableOffsetWords, p.pageGenerationOffsetWords, p.ownerPageTableOffsetWords, p.payloadOffsetWords], 20);
    this.paramWords.set([source.brickSize, source.revision, residency.layout.entryOffsetsBytes.active / 4, p.tileEdge], 24);
    this.paramWords.set([...p.sourceFineDimensions, 0], 28);
    this.paramWords.set([...p.sourceOriginFine, 0], 32);
    new Float32Array(this.paramWords.buffer).set([...p.fineCellSize_m, 0], 36);
    device.queue.writeBuffer(this.arena, 0, new Uint32Array(p.payloadOffsetWords));
    device.queue.writeBuffer(this.params, 0, this.paramWords);
    const layout = device.createBindGroupLayout({ label: "SVO renderer fine-phi staging layout", entries: [
      ...[0,1,2,3,4,5,6].map((binding) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" as const } })),
      { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    ] });
    const shaderModule = device.createShaderModule({ label: "SVO renderer fine-phi staging", code: svoFinePhiStagingShader });
    void shaderModule.getCompilationInfo().then((report) => {
      for (const message of report.messages) if (message.type === "error") console.error(`SVO fine-phi WGSL ${message.lineNum}:${message.linePos} ${message.message}`);
    }).catch(() => { /* Device loss belongs to the owning renderer. */ });
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
    const pipeline = (entryPoint: keyof typeof this.pipelines) => device.createComputePipeline({
      label: `SVO renderer fine-phi ${entryPoint}`, layout: pipelineLayout, compute: { module: shaderModule, entryPoint },
    });
    this.pipelines = { prepare: pipeline("prepare"), clearRetired: pipeline("clearRetired"), stageActive: pipeline("stageActive"), finalize: pipeline("finalize") };
    this.bindGroup = device.createBindGroup({ label: "SVO renderer fine-phi staging bindings", layout, entries: [
      { binding: 0, resource: owner.storageBinding() }, { binding: 1, resource: { buffer: owner.retiredSlots } },
      { binding: 2, resource: source.control }, { binding: 3, resource: source.pageTable }, { binding: 4, resource: source.phi },
      { binding: 5, resource: { buffer: residency.control } }, { binding: 6, resource: { buffer: residency.entries } },
      { binding: 7, resource: { buffer: this.arena } }, { binding: 8, resource: source.params }, { binding: 9, resource: { buffer: this.params } },
    ] });
    this.allocatedBytes = this.arena.size + this.params.size + this.dispatch.size;
  }

  /** Encode after source publication, residency compaction, and owner allocation. */
  encode(encoder: GPUCommandEncoder, expectedFineGeneration: number): void {
    if (this.destroyed) return;
    if (!Number.isSafeInteger(expectedFineGeneration) || expectedFineGeneration < 1 || expectedFineGeneration > UINT32_MAX) {
      throw new RangeError("Expected fine-phi generation must be a positive uint32");
    }
    this.paramWords[25] = expectedFineGeneration >>> 0;
    this.device.queue.writeBuffer(this.params, 0, this.paramWords);
    encoder.copyBufferToBuffer(this.residencyControlBuffer, SVO_RENDER_RESIDENCY_CONSUMER_CONTROL_WORDS.activeOutputDispatch * 4, this.dispatch, 0, 12);
    encoder.copyBufferToBuffer(this.residencyControlBuffer, SVO_RENDER_RESIDENCY_CONSUMER_CONTROL_WORDS.retiredOutputDispatch * 4, this.dispatch, 12, 12);
    const direct = (label: string, pipeline: GPUComputePipeline) => { const pass = encoder.beginComputePass({ label }); pass.setPipeline(pipeline); pass.setBindGroup(0, this.bindGroup); pass.dispatchWorkgroups(1); pass.end(); };
    direct("Prepare SVO renderer fine-phi publication", this.pipelines.prepare);
    const stage = encoder.beginComputePass({ label: "Stage and retire SVO renderer fine-phi pages" });
    stage.setBindGroup(0, this.bindGroup);
    stage.setPipeline(this.pipelines.clearRetired); stage.dispatchWorkgroupsIndirect(this.dispatch, 12);
    stage.setPipeline(this.pipelines.stageActive); stage.dispatchWorkgroupsIndirect(this.dispatch, 0);
    stage.end();
    direct("Publish SVO renderer fine-phi capability", this.pipelines.finalize);
  }

  /** Mirror the producer publication into the first eight arena words every frame. */
  mirrorPublication(encoder: GPUCommandEncoder): void {
    if (this.destroyed || !this.structuralPublication) return;
    encoder.copyBufferToBuffer(
      this.structuralPublication.buffer,
      this.structuralPublication.offset ?? 0,
      this.arena,
      0,
      SPARSE_VOXEL_PUBLICATION_STATE.strideBytes,
    );
  }

  capability(): SvoFineFluidGpuCapability {
    return {
      arena: { buffer: this.arena, offset: 0, size: this.plan.allocatedBytes }, params: { buffer: this.params },
      statusWord: SVO_FINE_PHI_CONTROL_WORDS.status,
      acceptedStructuralGenerationWord: SVO_FINE_PHI_CONTROL_WORDS.acceptedStructuralGeneration,
      acceptedFineGenerationWord: SVO_FINE_PHI_CONTROL_WORDS.acceptedFineGeneration,
      pageGenerationOffsetWords: this.plan.pageGenerationOffsetWords, ownerPageTableOffsetWords: this.plan.ownerPageTableOffsetWords,
      payloadOffsetWords: this.plan.payloadOffsetWords, paramsWords: Uint32Array.from(this.paramWords), publicationMirrorWords: 8,
      coarseFallbackRequired: true, directWaterOwnership: false,
    };
  }

  destroy(): void {
    if (this.destroyed) return; this.destroyed = true;
    this.dispatch.destroy(); this.params.destroy(); this.arena.destroy();
  }
}
