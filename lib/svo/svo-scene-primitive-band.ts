/**
 * The near-field analytic band — which authored SDF records are still marched.
 *
 * ---------------------------------------------------------------------------
 * What this decides, and why it is a decision at all
 * ---------------------------------------------------------------------------
 * After W1 the authored SDF set is drawn through a per-pixel coverage/resolve
 * arena, and after W2 the same records are voxelized ABI-true into the octree's
 * scene lanes. Both paths therefore draw the *same surfaces*: the voxel path
 * resolves a hit to the owning record and re-intersects it exactly over one
 * cell's interval (`traceLeafPayload`), so a voxel hit is a real point on a real
 * authored surface rather than a blocky approximation of one.
 *
 * That makes the analytic raster redundant wherever the voxel lattice is fine
 * enough to *find* the surface in the first place — and it is the analytic
 * raster, not the voxel path, whose cost scales with authored record count. So
 * the policy is: keep a record in the analytic set only while one voxel of the
 * scene lattice still projects to more than an authored number of pixels at that
 * record's nearest point. Everything else stops being drawn analytically at all.
 *
 * The measure is `projectedSvoNodeFootprintPixels`
 * (`lib/svo-screen-space-termination.ts`), evaluated on a *cell-sized* box
 * placed at the record's closest approach to the eye — the same primitive the
 * traversal's screen-space termination uses, with the consumer it was always
 * meant to have.
 *
 * ---------------------------------------------------------------------------
 * Why the budget is load-bearing rather than a safety valve
 * ---------------------------------------------------------------------------
 * "Near-field" suggests a small camera-local set, and in a scene with real depth
 * range it is one. The hero garden is not that scene: it is 1.8 m across, viewed
 * from 1.4 m through a 50 mm lens, so the projected voxel size across the whole
 * set spans about 4x (≈108 px at the near bank, ≈26 px at the far wall). A pure
 * threshold there selects a depth slice whose *size grows with record count*,
 * which is exactly the property the 10x gate forbids.
 *
 * The hard budget is what makes it flat. Records are bucketed by projected voxel
 * size and admitted from the largest bucket down until the budget is reached, so
 * the analytic set is bounded by an authored constant no matter how many records
 * stand behind it, and the ones it keeps are the ones the lattice serves worst.
 * Bucket granularity is deliberate — a strict top-K would need a sort or an
 * order-dependent atomic, and the last partial bucket is worth neither. The
 * nearest occupied bucket is always admitted, so the band is never empty.
 *
 * ---------------------------------------------------------------------------
 * Hysteresis
 * ---------------------------------------------------------------------------
 * Membership flips a record between two exact-but-different resolutions of the
 * same surface, so a record sitting on the threshold would switch every frame a
 * moving camera jitters it across. The exit threshold is a fraction of the entry
 * threshold and the previous frame's *final* membership selects which one
 * applies, which is why the per-record word survives the frame.
 */
import { svoFieldProgramAbsentWGSL, svoPrimitiveWGSL } from "./svo-primitive-abi";
import { SVO_PRIMITIVE_CANDIDATE_MAXIMUM_LEAVES } from "./svo-primitive-candidates";
import { svoScreenSpaceTerminationWGSL } from "./svo-screen-space-termination";
import { cameraApertureShaderLibrary } from "../core/webgpu-camera";

export const SVO_SCENE_PRIMITIVE_BAND_CONTRACT = Object.freeze({
  workgroupSize: 64,
  /**
   * Sixteen buckets per octave of projected voxel size, from 1/16 px to 4096 px.
   *
   * The range is chosen so nothing a real camera can produce lands on an end
   * stop: 4096 px is a voxel filling a 4K viewport, and 1/16 px is a voxel two
   * orders of magnitude beyond the far wall of any authored scene here.
   *
   * The *resolution* is chosen against the budget rather than against the
   * scene. Admission is by whole buckets, so a bucket's population is the
   * amount by which the budget can be overshot; the hero garden spans barely
   * two octaves of projected voxel size, which at four buckets per octave would
   * have put six hundred records in one bucket and made a 256-record budget
   * meaningless. At sixteen a bucket is a 4.4 % slice of projected size.
   */
  buckets: 256,
  bucketsPerOctave: 16,
  minimumLog2Pixels: -4,
  /** cutoff bucket, candidates, admitted, highest occupied bucket, 4 reserved. */
  headerWords: 8,
  /**
   * One state word per addressable record.
   *
   * Sized to the authored record ceiling rather than to the coverage key's
   * 65,536, because a record past `SVO_PRIMITIVE_CANDIDATE_MAXIMUM_LEAVES` never
   * reaches a draw. 64 KB, allocated once, never resized with the viewport.
   */
  maximumRecords: SVO_PRIMITIVE_CANDIDATE_MAXIMUM_LEAVES,
  /** Bits of the per-record state word. */
  bucketMask: 0xff,
  candidateBit: 1 << 8,
  memberBit: 1 << 9,
  /** Group-zero bindings of the standalone classify/resolve module. */
  bindings: Object.freeze({ uniforms: 0, params: 1, scene: 2, state: 3, band: 4 }),
  /**
   * Where the render stages read the finished decision.
   *
   * The voxel-light cache already owns bindings 0-5 of the group above the split
   * bridge; six is the next free slot, and a layout carrying only this binding
   * is what the coverage and overflow pipelines append.
   */
  readGroupBinding: 6,
  /** Sampled arena-pressure counters, written by the resolve. */
  auditGroupBinding: 7,
  entryPoints: Object.freeze({
    classify: "svoScenePrimitiveBandClassifyMain",
    resolve: "svoScenePrimitiveBandResolveMain",
  }),
} as const);

const { buckets, headerWords, maximumRecords } = SVO_SCENE_PRIMITIVE_BAND_CONTRACT;

/** Words before the per-record state array: header, then the histogram. */
export const SVO_SCENE_PRIMITIVE_BAND_RECORD_WORD_OFFSET = headerWords + buckets;
export const SVO_SCENE_PRIMITIVE_BAND_STATE_WORDS =
  SVO_SCENE_PRIMITIVE_BAND_RECORD_WORD_OFFSET + maximumRecords;
export const SVO_SCENE_PRIMITIVE_BAND_STATE_BYTES =
  SVO_SCENE_PRIMITIVE_BAND_STATE_WORDS * Uint32Array.BYTES_PER_ELEMENT;
/** Header plus histogram; cleared before every classify dispatch. */
export const SVO_SCENE_PRIMITIVE_BAND_COUNTER_BYTES =
  SVO_SCENE_PRIMITIVE_BAND_RECORD_WORD_OFFSET * Uint32Array.BYTES_PER_ELEMENT;
/** enter pixels (f32), exit pixels (f32), budget records, flags. */
export const SVO_SCENE_PRIMITIVE_BAND_PARAMS_BYTES = 16;

export interface SvoScenePrimitiveBandParams {
  /** Projected voxel size above which a record stays analytic. Zero admits every record. */
  readonly enterPixels: number;
  /** The same test for a record already in the band; never above `enterPixels`. */
  readonly exitPixels: number;
  /** Hard ceiling on admitted records. Zero means unbounded. */
  readonly budgetRecords: number;
}

export function packSvoScenePrimitiveBandParams(
  params: SvoScenePrimitiveBandParams,
): ArrayBuffer {
  if (!Number.isFinite(params.enterPixels) || params.enterPixels < 0
    || !Number.isFinite(params.exitPixels) || params.exitPixels < 0
    || params.exitPixels > params.enterPixels) {
    throw new RangeError("SVO band thresholds must be finite, non-negative and ordered exit <= enter");
  }
  if (!Number.isSafeInteger(params.budgetRecords) || params.budgetRecords < 0) {
    throw new RangeError("SVO band budget must be a non-negative safe integer");
  }
  const buffer = new ArrayBuffer(SVO_SCENE_PRIMITIVE_BAND_PARAMS_BYTES);
  new Float32Array(buffer, 0, 2).set([params.enterPixels, params.exitPixels]);
  new Uint32Array(buffer, 8, 2).set([Math.min(params.budgetRecords, maximumRecords), 0]);
  return buffer;
}

export interface SvoScenePrimitiveBandReport {
  /** Lowest admitted bucket; 64 means the budget admitted nothing. */
  readonly cutoffBucket: number;
  /** Records over the screen-space threshold before the budget applied. */
  readonly candidates: number;
  /** Records the band actually admitted this frame. */
  readonly admitted: number;
  readonly highestOccupiedBucket: number;
}

export function readSvoScenePrimitiveBandReport(header: Uint32Array): SvoScenePrimitiveBandReport {
  if (header.length < 4) throw new RangeError("SVO band header needs at least four words");
  return {
    cutoffBucket: header[0],
    candidates: header[1],
    admitted: header[2],
    highestOccupiedBucket: header[3],
  };
}

/** Projected pixels at the centre of a bucket, for reporting a band's reach. */
export function svoScenePrimitiveBandBucketPixels(bucket: number): number {
  const { bucketsPerOctave, minimumLog2Pixels } = SVO_SCENE_PRIMITIVE_BAND_CONTRACT;
  return 2 ** ((bucket + 0.5) / bucketsPerOctave + minimumLog2Pixels);
}

export function svoScenePrimitiveBandComputeBindGroupLayoutEntries(): GPUBindGroupLayoutEntry[] {
  const { bindings } = SVO_SCENE_PRIMITIVE_BAND_CONTRACT;
  const visibility = GPUShaderStage.COMPUTE;
  return [
    { binding: bindings.uniforms, visibility, buffer: { type: "uniform" } },
    { binding: bindings.params, visibility, buffer: { type: "uniform" } },
    { binding: bindings.scene, visibility, buffer: { type: "read-only-storage" } },
    { binding: bindings.state, visibility, buffer: { type: "storage" } },
    { binding: bindings.band, visibility, buffer: { type: "uniform" } },
  ];
}

/**
 * What the scene-primitive passes read beside the coverage arena: the band's
 * membership words, and the arena-pressure audit the resolve accumulates.
 *
 * The two are separate buffers rather than two regions of one, because a render
 * pass may not use the same buffer as both a writable and a read-only storage
 * binding — and the membership words have to be readable from the *vertex*
 * stage, where writable storage is not available at all.
 */
export function svoScenePrimitiveBandReadBindGroupLayoutEntries(): GPUBindGroupLayoutEntry[] {
  return [
    {
      binding: SVO_SCENE_PRIMITIVE_BAND_CONTRACT.readGroupBinding,
      visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
      buffer: { type: "read-only-storage" },
    },
    {
      binding: SVO_SCENE_PRIMITIVE_BAND_CONTRACT.auditGroupBinding,
      visibility: GPUShaderStage.FRAGMENT,
      buffer: { type: "storage" },
    },
  ];
}

/**
 * Per-pixel arena pressure, sampled.
 *
 * The arena's capacity is a performance parameter and never an image change, so
 * a catastrophic overflow rate is invisible by construction — which is exactly
 * how the scene-primitive overflow pass came to cost 0.144 ms at 501 records and
 * **335 ms at 5 039** with nothing reporting it. These four words are the
 * missing report.
 *
 * Sampled one pixel in 64 because the alternative is a global atomic per covered
 * pixel — two million of them at the hero resolution, on one cache line. A rate
 * estimate does not need every pixel, and the exact distribution is available
 * off-line through `copyCoverageCounts`.
 */
export const SVO_SCENE_PRIMITIVE_COVERAGE_AUDIT_STRIDE = 64;
export const SVO_SCENE_PRIMITIVE_COVERAGE_AUDIT_BYTES = 16;

export interface SvoScenePrimitiveCoverageAudit {
  /** Sampled pixels carrying at least one candidate. */
  readonly coveredPixels: number;
  /** Of those, the ones whose list exceeded capacity and re-ran the direct pass. */
  readonly overflowedPixels: number;
  readonly maximumCandidates: number;
  readonly meanCandidates: number;
  readonly overflowFraction: number;
  readonly samplingStride: number;
}

export function readSvoScenePrimitiveCoverageAudit(words: Uint32Array): SvoScenePrimitiveCoverageAudit {
  if (words.length < 4) throw new RangeError("SVO coverage audit needs four words");
  const [coveredPixels, overflowedPixels, maximumCandidates, totalCandidates] = words;
  return {
    coveredPixels,
    overflowedPixels,
    maximumCandidates,
    meanCandidates: coveredPixels > 0 ? totalCandidates / coveredPixels : 0,
    overflowFraction: coveredPixels > 0 ? overflowedPixels / coveredPixels : 0,
    samplingStride: SVO_SCENE_PRIMITIVE_COVERAGE_AUDIT_STRIDE,
  };
}

/**
 * The band's read side, spliced into the dry-scene module.
 *
 * A record with no state word is a member: the band can only ever *remove*
 * analytic draws, so every failure of this machinery — an unbound buffer, a
 * record past the state array, a frame before the first classify — has to leave
 * the exact historical set drawing.
 */
export function svoScenePrimitiveBandReadWGSL(group: number, coverageCapacity: number): string {
  const { readGroupBinding, auditGroupBinding, memberBit } = SVO_SCENE_PRIMITIVE_BAND_CONTRACT;
  return /* wgsl */ `
@group(${group}) @binding(${readGroupBinding}) var<storage,read> dryScenePrimitiveBand:array<u32>;
@group(${group}) @binding(${auditGroupBinding}) var<storage,read_write> dryScenePrimitiveCoverageAudit:array<atomic<u32>>;
const DRY_BAND_RECORD_WORD_OFFSET:u32=${SVO_SCENE_PRIMITIVE_BAND_RECORD_WORD_OFFSET}u;
const DRY_BAND_MEMBER_BIT:u32=${memberBit}u;
const DRY_COVERAGE_AUDIT_STRIDE:u32=${SVO_SCENE_PRIMITIVE_COVERAGE_AUDIT_STRIDE}u;
fn dryScenePrimitiveBandMember(record:u32)->bool{
  let index=DRY_BAND_RECORD_WORD_OFFSET+record;
  if(index>=arrayLength(&dryScenePrimitiveBand)){return true;}
  return (dryScenePrimitiveBand[index]&DRY_BAND_MEMBER_BIT)!=0u;
}
// One sampled pixel in sixty-four, from the pass that visits each pixel exactly
// once. Capacity never changes the image, so without this the overflow arm's
// cost is unobservable from anything but a profiler.
fn dryScenePrimitiveAuditCoverage(pixel:u32,candidates:u32){
  if(candidates==0u||(pixel%DRY_COVERAGE_AUDIT_STRIDE)!=0u){return;}
  atomicAdd(&dryScenePrimitiveCoverageAudit[0],1u);
  if(candidates>${coverageCapacity}u){atomicAdd(&dryScenePrimitiveCoverageAudit[1],1u);}
  atomicMax(&dryScenePrimitiveCoverageAudit[2],candidates);
  atomicAdd(&dryScenePrimitiveCoverageAudit[3],candidates);
}
`;
}

export interface SvoScenePrimitiveBandWGSLOptions {
  /** u32 word offset of the primitive region inside the authored scene arena. */
  readonly primitiveWordOffset: number;
}

/**
 * Classify then resolve, as one standalone compute module.
 *
 * It reads the production camera uniform and the `SvoMapping` + metadata prefix
 * of the dry-scene params, so the band is derived from exactly the numbers the
 * frame is drawn with rather than from a CPU mirror of them — which also means
 * the renderer needs no camera of its own to compute it.
 */
export function createSvoScenePrimitiveBandWGSL(options: SvoScenePrimitiveBandWGSLOptions): string {
  const { bindings, entryPoints, workgroupSize, bucketsPerOctave, minimumLog2Pixels,
    bucketMask, candidateBit, memberBit } = SVO_SCENE_PRIMITIVE_BAND_CONTRACT;
  const bucketBias = -minimumLog2Pixels * bucketsPerOctave;
  return /* wgsl */ `
// ---------------------------------------------------------------------------
// Near-field analytic band selection. See lib/svo-scene-primitive-band.ts.
// ---------------------------------------------------------------------------
struct Uniforms{viewport:vec4f,cameraPosition:vec4f}
struct SvoMapping{worldOrigin:vec3f,brickSize:u32,cellSize:vec3f,maximumDepth:u32,nodeCount:u32,leafCount:u32,maxVisits:u32,_padding:u32}
struct SvoBandParams{mapping:SvoMapping,metadata:vec4u}
struct SvoBandControls{thresholds:vec2f,budget:u32,flags:u32}
// Band selection reads a record's extent and its screen footprint; it never
// evaluates a field. No tape arena is bound here, and a field-program record
// therefore resolves to nothing rather than to its conservative box.
${svoFieldProgramAbsentWGSL}
${svoPrimitiveWGSL}
${svoScreenSpaceTerminationWGSL}

@group(0) @binding(${bindings.uniforms}) var<uniform> uniforms:Uniforms;
${cameraApertureShaderLibrary()}
@group(0) @binding(${bindings.params}) var<uniform> params:SvoBandParams;
@group(0) @binding(${bindings.scene}) var<storage,read> bandScene:array<u32>;
@group(0) @binding(${bindings.state}) var<storage,read_write> bandState:array<atomic<u32>>;
@group(0) @binding(${bindings.band}) var<uniform> controls:SvoBandControls;

const BAND_HEADER_WORDS:u32=${headerWords}u;
const BAND_BUCKETS:u32=${buckets}u;
const BAND_RECORD_WORD_OFFSET:u32=${SVO_SCENE_PRIMITIVE_BAND_RECORD_WORD_OFFSET}u;
const BAND_MAXIMUM_RECORDS:u32=${maximumRecords}u;
const BAND_BUCKET_MASK:u32=${bucketMask}u;
const BAND_CANDIDATE_BIT:u32=${candidateBit}u;
const BAND_MEMBER_BIT:u32=${memberBit}u;

fn bandWords4(offset:u32)->vec4u{return vec4u(bandScene[offset],bandScene[offset+1u],bandScene[offset+2u],bandScene[offset+3u]);}
fn bandPrimitive(index:u32)->SvoPrimitiveRecord{
  let base=${options.primitiveWordOffset}u+index*16u;
  return SvoPrimitiveRecord(bandWords4(base),bandWords4(base+4u),bitcast<vec4f>(bandWords4(base+8u)),bandWords4(base+12u));
}
// The record's world-axis-aligned half-extent, identical in form to the proxy
// the raster vertex stage builds; a band that bounded a different box from the
// one being drawn would admit and cull different records.
fn bandWorldExtent(localExtent:vec3f,orientation:vec4f)->vec3f{
  let q=orientation/length(orientation);
  return abs(svoQuaternionRotate(q,vec3f(localExtent.x,0.0,0.0)))
    +abs(svoQuaternionRotate(q,vec3f(0.0,localExtent.y,0.0)))
    +abs(svoQuaternionRotate(q,vec3f(0.0,0.0,localExtent.z)));
}
fn bandRecordCount()->u32{return min(params.metadata.x,BAND_MAXIMUM_RECORDS);}
fn bandBucket(pixels:f32)->u32{
  return u32(clamp(floor(log2(max(pixels,1e-6))*${bucketsPerOctave}.0)+${bucketBias}.0,0.0,${buckets - 1}.0));
}

/**
 * One record: its projected voxel size, its hysteresis-adjusted candidacy, and
 * its contribution to the histogram the budget is resolved from.
 */
@compute @workgroup_size(${workgroupSize})
fn ${entryPoints.classify}(@builtin(global_invocation_id) id:vec3u){
  let record=id.x;
  if(record>=bandRecordCount()){return;}
  let stateIndex=BAND_RECORD_WORD_OFFSET+record;
  let previous=atomicLoad(&bandState[stateIndex]);
  let previouslyMember=(previous&BAND_MEMBER_BIT)!=0u;
  let entry=bandPrimitive(record);
  let localExtent=svoPrimitiveLocalExtent_m(svoPrimitiveKind(entry),svoPrimitiveDimensions_m(entry));
  let orientationLength=length(entry.orientation);
  // A kind with no finite local box (a heightfield) can never be bounded here,
  // and an unnormalizable orientation is a record the raster refuses anyway.
  // Both stay analytic: the band's failure direction is "draw it".
  var footprintPixels=3.402823e38;
  if(all(localExtent>=vec3f(0.0))&&orientationLength>1e-8){
    let extent=bandWorldExtent(localExtent,entry.orientation);
    let centre=svoPrimitiveCenter_m(entry);
    let eye=uniforms.cameraPosition.xyz;
    // Closest approach, so a long record is judged by the end nearest the eye
    // rather than by a centre that may be far behind it.
    let closest=clamp(eye,centre-extent,centre+extent);
    let voxelHalf=0.5*params.mapping.cellSize;
    footprintPixels=svoProjectedNodeFootprintPixels(
      mat2x3f(closest-voxelHalf,closest+voxelHalf),eye,uniforms.viewport.y,cameraTanHalfFov());
  }
  let threshold=select(controls.thresholds.x,controls.thresholds.y,previouslyMember);
  let candidate=controls.thresholds.x<=0.0||footprintPixels>threshold;
  let bucket=bandBucket(footprintPixels);
  var word=bucket&BAND_BUCKET_MASK;
  if(candidate){
    word|=BAND_CANDIDATE_BIT;
    atomicAdd(&bandState[BAND_HEADER_WORDS+bucket],1u);
  }
  // The member bit is deliberately left cleared here: the resolve below is the
  // only writer of it, so a frame that never reached the resolve draws nothing
  // analytic rather than drawing last frame's set.
  atomicStore(&bandState[stateIndex],word);
}

var<workgroup> bandHistogram:array<u32,${buckets}>;
var<workgroup> bandCutoff:u32;

/**
 * One workgroup: turn the histogram into a cutoff bucket, then mark membership.
 *
 * Admission walks from the largest bucket down and stops before the bucket that
 * would cross the budget, so the admitted count never exceeds it — except for
 * the nearest occupied bucket, which is admitted unconditionally so that a
 * budget smaller than the near field still leaves the closest records exact.
 */
@compute @workgroup_size(${workgroupSize})
fn ${entryPoints.resolve}(@builtin(local_invocation_id) local:vec3u){
  let lane=local.x;
  for(var bucket=lane;bucket<BAND_BUCKETS;bucket+=${workgroupSize}u){
    bandHistogram[bucket]=atomicLoad(&bandState[BAND_HEADER_WORDS+bucket]);
  }
  workgroupBarrier();
  if(lane==0u){
    var candidates=0u;var highest=0u;
    for(var bucket=0u;bucket<BAND_BUCKETS;bucket+=1u){
      candidates+=bandHistogram[bucket];
      if(bandHistogram[bucket]>0u){highest=bucket;}
    }
    var cutoff=0u;var admitted=candidates;
    if(controls.budget>0u&&candidates>controls.budget){
      var accumulated=0u;cutoff=BAND_BUCKETS;admitted=0u;
      for(var step=0u;step<BAND_BUCKETS;step+=1u){
        let bucket=BAND_BUCKETS-1u-step;
        let population=bandHistogram[bucket];
        if(population==0u){continue;}
        let next=accumulated+population;
        // Stop *before* the bucket that would cross the budget — unless nothing
        // has been admitted yet, because a band with no near field at all is a
        // worse failure than one bucket of overshoot.
        if(next>controls.budget&&accumulated>0u){break;}
        accumulated=next;cutoff=bucket;admitted=accumulated;
        if(accumulated>=controls.budget){break;}
      }
    }
    bandCutoff=cutoff;
    atomicStore(&bandState[0],cutoff);
    atomicStore(&bandState[1],candidates);
    atomicStore(&bandState[2],admitted);
    atomicStore(&bandState[3],highest);
  }
  workgroupBarrier();
  let cutoff=bandCutoff;
  let count=bandRecordCount();
  for(var record=lane;record<count;record+=${workgroupSize}u){
    let stateIndex=BAND_RECORD_WORD_OFFSET+record;
    let word=atomicLoad(&bandState[stateIndex]);
    let member=(word&BAND_CANDIDATE_BIT)!=0u&&(word&BAND_BUCKET_MASK)>=cutoff;
    atomicStore(&bandState[stateIndex],select(word,word|BAND_MEMBER_BIT,member));
  }
}
`;
}
