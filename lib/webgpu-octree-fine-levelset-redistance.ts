import {
  makeFineLevelSetSortedWorklistLookupWGSL,
  type WebGPUFineLevelSetBrickSource,
} from "./webgpu-octree-fine-levelset-bricks";
import { fineLevelSetLinearWorkgroupWGSL } from "./webgpu-fine-levelset-dispatch";
import { PassBroker } from "./webgpu-pass-broker";

export interface FineLevelSetGPURedistanceOptions {
  /** Required signed-distance width, measured in fine cells. */
  bandCells: number;
  residualTolerance?: number;
}

/** The exact immutable page-delta ABI authored by fine topology.
 *
 * Redistance deliberately accepts the topology authority rather than a bare
 * storage buffer.  This prevents a capacity-sized/full-active worklist from
 * being substituted at a call site and makes the dirty/support offset
 * contract explicit at construction.
 */
export interface FineLevelSetRedistanceDeltaAuthority {
  readonly pageDelta: GPUBuffer;
  readonly pageDeltaLayout: {
    readonly headerWords: 16;
    readonly dirtyPagesOffsetWords: number;
    readonly supportPagesOffsetWords: number;
  };
  /** Immutable topology-authored commands: one workgroup per exact page. */
  readonly redistanceDispatches: {
    readonly buffer: GPUBuffer;
    readonly dirtyOffsetBytes: 84;
    readonly supportOffsetBytes: 60;
  };
}

/** Warm-started descending JFA strides followed by two local repair passes.
 *
 * The recurring sparse field is transported from the preceding accepted
 * generation and the timestep is bounded by `maximumDisplacementFineCells`.
 * New collar bricks are coarse-seeded by topology before this transform.
 * Consequently the flood only has to repair the motion collar; it does not
 * need to rediscover the complete maintained band from a domain-scale jump.
 */
export function planFineLevelSetJFAStrides(
  bandCells: number,
  maximumDisplacementFineCells = 4,
): readonly number[] {
  if (!Number.isSafeInteger(bandCells) || bandCells < 1 || bandCells > 256) {
    throw new RangeError("Fine redistance bandCells must be an integer in [1, 256]");
  }
  if (!Number.isSafeInteger(maximumDisplacementFineCells)
    || maximumDisplacementFineCells < 1 || maximumDisplacementFineCells > 256) {
    throw new RangeError("Fine redistance displacement must be an integer in [1, 256]");
  }
  let stride = 1;
  const repairRadius = Math.min(bandCells, maximumDisplacementFineCells);
  if (maximumDisplacementFineCells >= bandCells) {
    // A cold publication has no transported closest-point field to repair.
    // Round up so its first jump spans the complete signed band.
    while (stride < repairRadius) stride *= 2;
  } else {
    while (stride * 2 <= repairRadius) stride *= 2;
  }
  const strides: number[] = [];
  for (; stride >= 1; stride /= 2) strides.push(stride);
  // Two +1 collar repairs close sparse page-boundary landing gaps after the
  // descending schedule. The second is required by the mini dam-break
  // generation-280 boundary. Publication still fails closed if a seed is missing.
  strides.push(1);
  strides.push(1);
  return strides;
}

export interface FineLevelSetGPURedistanceControl {
  /** Total rejection count: missing closest points plus Eikonal residual violations. */
  unresolvedCells: number;
  resolveMissingCells: number;
  residualViolationCells: number;
  maximumResidualScaled: number;
  seedCount: number;
  committed: boolean;
  flags: number;
  firstError: number;
  acceptedCells: number;
  initialPages: number;
  finalPages: number;
}

export const FINE_LEVELSET_REDISTANCE_CONTROL_BYTES = 40;
export const FINE_LEVELSET_REDISTANCE_REDUCTION_RECORD_BYTES = 28;
const FINE_LEVELSET_JFA_MAX_PASSES = 10;
export const FINE_LEVELSET_REDISTANCE_ALLOCATED_BYTES = FINE_LEVELSET_REDISTANCE_CONTROL_BYTES
  + 80;

export function fineLevelSetRedistanceAllocatedBytes(maximumResidentBricks: number): number {
  if (!Number.isSafeInteger(maximumResidentBricks) || maximumResidentBricks < 1) {
    throw new RangeError("Fine redistance resident capacity must be a positive integer");
  }
  return FINE_LEVELSET_REDISTANCE_ALLOCATED_BYTES
    + maximumResidentBricks * (FINE_LEVELSET_REDISTANCE_REDUCTION_RECORD_BYTES + 4);
}

export function unpackFineLevelSetGPURedistanceControl(words: ArrayLike<number>): FineLevelSetGPURedistanceControl {
  if (words.length < 4) throw new RangeError("Fine redistance control requires four words");
  const unresolvedCells = Number(words[0]) >>> 0;
  const resolveMissingCells = words.length > 9 ? Number(words[9]) >>> 0 : 0;
  return {
    unresolvedCells,
    resolveMissingCells,
    residualViolationCells: Math.max(0, unresolvedCells - resolveMissingCells),
    maximumResidualScaled: Number(words[1]) >>> 0,
    seedCount: Number(words[2]) >>> 0,
    committed: Number(words[3]) !== 0,
    flags: words.length > 4 ? Number(words[4]) >>> 0 : 0,
    firstError: words.length > 5 ? Number(words[5]) >>> 0 : 0xffff_ffff,
    acceptedCells: words.length > 6 ? Number(words[6]) >>> 0 : 0,
    initialPages: words.length > 7 ? Number(words[7]) >>> 0 : 0,
    finalPages: words.length > 8 ? Number(words[8]) >>> 0 : 0,
  };
}

/**
 * Fixed-resident fine-grid redistance. JFA-CPT preserves the closest-point
 * transform semantics with a bounded logarithmic schedule. It consumes the
 * complete support generation published by topology and never allocates,
 * links, or publishes a page while redistancing.
 */
export class WebGPUFineLevelSetRedistance {
  readonly control: GPUBuffer;
  readonly allocatedBytes: number;
  private readonly reductions: GPUBuffer;
  private readonly supportMask: GPUBuffer;
  private readonly params: GPUBuffer;
  private readonly publishSupportMaskPipeline: GPUComputePipeline;
  private readonly jfaSeedPipeline: GPUComputePipeline;
  private readonly jfaABPipelines: ReadonlyMap<number, GPUComputePipeline>;
  private readonly jfaBAPipelines: ReadonlyMap<number, GPUComputePipeline>;
  private readonly jfaResolveAToBPipeline: GPUComputePipeline;
  private readonly jfaResolveBToCanonicalPipeline: GPUComputePipeline;
  private readonly jfaValidatePipeline: GPUComputePipeline;
  private readonly jfaFinalizePipeline: GPUComputePipeline;
  private readonly jfaCommitPipeline: GPUComputePipeline;

  constructor(private readonly device: GPUDevice, readonly source: WebGPUFineLevelSetBrickSource,
    /** Exact delta/support publication produced by the topology transaction. */
    readonly delta: FineLevelSetRedistanceDeltaAuthority) {
    const pageCapacity = source.plan.maximumResidentBricks;
    if (delta.pageDeltaLayout.headerWords !== 16
      || delta.pageDeltaLayout.dirtyPagesOffsetWords !== 16 + 2 * pageCapacity
      || delta.pageDeltaLayout.supportPagesOffsetWords !== 16 + 3 * pageCapacity
      || delta.redistanceDispatches.dirtyOffsetBytes !== 7 * 12
      || delta.redistanceDispatches.supportOffsetBytes !== 5 * 12) {
      throw new RangeError("Fine JFA-CPT requires the exact topology page-delta ABI");
    }
    this.control = device.createBuffer({ label: "fine-levelset JFA-CPT control",
      size: FINE_LEVELSET_REDISTANCE_CONTROL_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const reductionBytes = source.plan.maximumResidentBricks * FINE_LEVELSET_REDISTANCE_REDUCTION_RECORD_BYTES;
    this.reductions = device.createBuffer({ label: "fine-levelset JFA-CPT deterministic reductions",
      size: reductionBytes, usage: GPUBufferUsage.STORAGE });
    this.supportMask = device.createBuffer({ label: "fine-levelset JFA direct support-page mask",
      size: pageCapacity * 4, usage: GPUBufferUsage.STORAGE });
    this.allocatedBytes = fineLevelSetRedistanceAllocatedBytes(source.plan.maximumResidentBricks);
    this.params = device.createBuffer({ label: "fine-levelset JFA-CPT parameters", size: 80,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const jfaModule = device.createShaderModule({ label: "fine-levelset JFA closest-point transform",
      code: fineLevelSetJFACPTWGSL });
    const jfaPipeline = (entryPoint: string, stride?: number) => device.createComputePipeline({
      label: `fine JFA-CPT ${entryPoint}${stride === undefined ? "" : ` stride ${stride}`}`,
      layout: "auto",
      compute: {
        module: jfaModule,
        entryPoint,
        ...(stride === undefined ? {} : { constants: { JFA_STRIDE: stride } }),
      },
    });
    this.publishSupportMaskPipeline = jfaPipeline("publishSupportPageMask");
    this.jfaSeedPipeline = jfaPipeline("seedClosestPoints");
    const immutableStrides = Array.from({ length: 9 }, (_, index) => 1 << index);
    this.jfaABPipelines = new Map(immutableStrides.map((stride) =>
      [stride, jfaPipeline("jumpFloodAToB", stride)]));
    this.jfaBAPipelines = new Map(immutableStrides.map((stride) =>
      [stride, jfaPipeline("jumpFloodBToA", stride)]));
    this.jfaResolveAToBPipeline = jfaPipeline("resolveClosestPointsAToB");
    this.jfaResolveBToCanonicalPipeline = jfaPipeline("resolveClosestPointsBToCanonical");
    this.jfaValidatePipeline = jfaPipeline("validateJFADistances");
    this.jfaFinalizePipeline = jfaPipeline("finalizeJFADistances");
    this.jfaCommitPipeline = jfaPipeline("commitJFADistances");
  }

  encode(broker: PassBroker, options: FineLevelSetGPURedistanceOptions): void {
    if ((this.source.plan.fineFactor !== 4 && this.source.plan.fineFactor !== 8)
      || this.source.plan.brickResolution !== 4) {
      throw new RangeError("GPU fine JFA-CPT requires a factor-4/factor-8 B4 generation");
    }
    if (!Number.isSafeInteger(options.bandCells) || options.bandCells < 1 || options.bandCells > 256) {
      throw new RangeError("Fine redistance bandCells must be an integer in [1, 256]");
    }
    const tolerance = options.residualTolerance ?? 0.1;
    if (!Number.isFinite(tolerance) || tolerance <= 0 || tolerance > 1) {
      throw new RangeError("Fine redistance residual tolerance must be in (0, 1]");
    }
    const bytes = new ArrayBuffer(80); const u32 = new Uint32Array(bytes); const f32 = new Float32Array(bytes);
    u32.set([...this.source.plan.brickDimensions, this.source.plan.brickResolution,
      ...this.source.plan.sampleDimensions, this.source.plan.samplesPerBrick,
      this.source.plan.maximumResidentBricks, 5,
      this.source.plan.maximumResidentBricks, this.source.generation, options.bandCells], 0);
    f32[13] = this.source.plan.fineCellWidth; f32[14] = tolerance;
    u32[15] = this.source.plan.maximumResidentBricks * this.source.plan.samplesPerBrick;
    u32[16] = this.device.limits.maxComputeWorkgroupsPerDimension;
    u32[17] = 0;
    u32[18] = this.delta.pageDeltaLayout.dirtyPagesOffsetWords;
    u32[19] = this.delta.pageDeltaLayout.supportPagesOffsetWords;
    this.encodeJFA(broker, bytes, options.bandCells, tolerance);
  }

  private encodeJFA(
    broker: PassBroker,
    baseBytes: ArrayBuffer,
    bandCells: number,
    residualTolerance: number,
  ): void {
    // The compact topology currently carries phi but not the materialized
    // closest-point seed index into a newly allocated A/B page. Span the
    // maintained band until that cache becomes part of the page transaction.
    // A 16-cell band starts at stride 16 (one pass less than the former
    // 23-cell/stride-32 schedule), and dispatch remains page-delta bounded.
    const strides = planFineLevelSetJFAStrides(bandCells, bandCells);
    if (strides.length > FINE_LEVELSET_JFA_MAX_PASSES) throw new RangeError("Fine JFA pass budget exceeded");
    this.device.queue.writeBuffer(this.params, 0, baseBytes);
    const buffers = [[1, this.source.worklist], [2, this.source.metadata], [3, this.delta.pageDelta],
      [4, this.source.flags], [5, this.source.phi], [6, this.source.workA], [7, this.source.workB],
      [8, this.control], [9, this.reductions], [10, this.supportMask]] as const;
    const pass = broker.compute({ label: "Fine level-set JFA closest-point redistance" });
    const run = (pipeline: GPUComputePipeline, params: GPUBuffer, wanted: readonly number[],
      dispatch: "support" | "dirty" | "single" | "capacity") => {
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, this.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: params } },
        ...buffers.filter(([binding]) => wanted.includes(binding)).map(([binding, buffer]) =>
          ({ binding, resource: { buffer } })),
      ] }));
      if (dispatch === "single") pass.dispatchWorkgroups(1);
      else if (dispatch === "capacity") {
        pass.dispatchWorkgroups(Math.ceil(this.source.plan.maximumResidentBricks / 64));
      }
      else pass.dispatchWorkgroupsIndirect(this.delta.redistanceDispatches.buffer,
        dispatch === "dirty"
          ? this.delta.redistanceDispatches.dirtyOffsetBytes
          : this.delta.redistanceDispatches.supportOffsetBytes);
    };
    const params = this.params;
    run(this.publishSupportMaskPipeline, params, [2, 3, 10], "support");
    run(this.jfaSeedPipeline, params, [1, 2, 3, 4, 5, 6, 7, 9, 10], "support");
    let inA = true;
    strides.forEach((stride) => {
      const pipeline = (inA ? this.jfaABPipelines : this.jfaBAPipelines).get(stride);
      if (!pipeline) throw new RangeError(`Fine JFA stride ${stride} has no immutable pipeline`);
      run(pipeline, params, [1, 2, 3, 4, 5, 6, 7, 10], "support");
      inA = !inA;
    });
    run(inA ? this.jfaResolveAToBPipeline : this.jfaResolveBToCanonicalPipeline, params,
      [2, 3, 4, 5, 6, 7, 9], "support");
    // Resolve canonicalizes persistent delta state: seeds are always in A and
    // magnitudes are always in B, independent of this generation's JFA parity.
    // Production uses tolerance=1. The unsigned-distance upwind residual is
    // bounded by one at that acceptance bar, so the validation sweep cannot
    // reject a cell and is pure lattice traffic. Keep the strict diagnostic
    // path for tests and opt-in sub-unit tolerances.
    if (residualTolerance < 1) {
      run(this.jfaValidatePipeline, params, [1, 2, 3, 4, 5, 7, 9, 10], "dirty");
    }
    run(this.jfaFinalizePipeline, params, [3, 8, 9], "single");
    // Finalization is a small deterministic hierarchy; committing phi is not.
    // Give every exact dirty page its own workgroup instead of making one
    // 256-lane workgroup stride over the complete dirty sample population.
    run(this.jfaCommitPipeline, params, [2, 3, 4, 5, 6, 7, 8], "dirty");
  }

  destroy(): void {
    this.control.destroy();
    this.reductions.destroy();
    this.supportMask.destroy();
    this.params.destroy();
  }
}

/** Sparse 1+JFA closest-point transform. Seed keys are global fine-sample
 * linear indices, so the deterministic secondary ordering is independent of
 * physical page IDs and A/B generation allocation order. */
export const fineLevelSetJFACPTWGSL = /* wgsl */ `
${fineLevelSetLinearWorkgroupWGSL}
const INVALID:u32=0xffffffffu;const VALID:u32=1u;const INTERFACE:u32=2u;const NEGATIVE:u32=16u;const LARGE:f32=3.402823e38;
// Direction seven is the nonzero cache sentinel for a quantized zero
// fraction; materialization treats every direction >= 6 as the sample centre.
const SAMPLE_FLAG_BITS:u32=5u;const CP_FRACTION_MASK:u32=0x00ffffffu;const CP_FRACTION_SCALE:f32=16777215.;
const STALE:u32=4u;const NONFINITE:u32=8u;
override JFA_STRIDE:u32=1u;
struct Params{brickDims:vec3u,brickResolution:u32,sampleDims:vec3u,samplesPerBrick:u32,worklistCapacity:u32,worklistHeaderWords:u32,pageCapacity:u32,generation:u32,bandCells:u32,fineWidth:f32,tolerance:f32,scratchWords:u32,maxWorkgroups:u32,pad0:u32,dirtyPagesOffset:u32,supportPagesOffset:u32}
struct Control{unresolved:u32,residualScaled:u32,seeds:u32,committed:u32,flags:u32,firstError:u32,accepted:u32,initialPages:u32,finalPages:u32,resolveMissing:u32}
struct Partial{seeds:u32,resolveMissing:u32,accepted:u32,validationUnresolved:u32,maximum:u32,flags:u32,firstError:u32}
@group(0)@binding(0)var<uniform>p:Params;@group(0)@binding(1)var<storage,read>worklist:array<u32>;@group(0)@binding(2)var<storage,read>metadata:array<u32>;@group(0)@binding(3)var<storage,read>pageDelta:array<u32>;@group(0)@binding(4)var<storage,read_write>flags:array<u32>;@group(0)@binding(5)var<storage,read_write>phi:array<u32>;@group(0)@binding(6)var<storage,read_write>workA:array<u32>;@group(0)@binding(7)var<storage,read_write>workB:array<u32>;@group(0)@binding(8)var<storage,read_write>control:Control;@group(0)@binding(9)var<storage,read_write>partials:array<Partial>;@group(0)@binding(10)var<storage,read_write>supportMask:array<u32>;
var<workgroup>reduceSum0:array<u32,256>;var<workgroup>reduceSum1:array<u32,256>;var<workgroup>reduceSum2:array<u32,256>;var<workgroup>reduceSum3:array<u32,256>;var<workgroup>reduceMaximum:array<u32,256>;var<workgroup>reduceFlags:array<u32,256>;var<workgroup>reduceFirstError:array<u32,256>;
// One workgroup owns one brick. Every JFA tap therefore lands in one of only
// 27 generation-fixed neighboring bricks; resolve those pages once instead
// of repeating the direct directory validation in all 64 lanes.
var<workgroup>floodPageIds:array<u32,27>;
fn finite(v:f32)->bool{return v==v&&abs(v)<LARGE;}fn bandDistance()->f32{return f32(p.bandCells)*p.fineWidth;}
fn reduceLane(lid:u32,sum0:u32,sum1:u32,sum2:u32,sum3:u32,maximum:u32,errorFlags:u32,firstError:u32,reductionWidth:u32){
  reduceSum0[lid]=sum0;reduceSum1[lid]=sum1;reduceSum2[lid]=sum2;reduceSum3[lid]=sum3;reduceMaximum[lid]=maximum;reduceFlags[lid]=errorFlags;reduceFirstError[lid]=firstError;workgroupBarrier();
  var width=reductionWidth;loop{if(width==0u){break;}if(lid<width){reduceSum0[lid]+=reduceSum0[lid+width];reduceSum1[lid]+=reduceSum1[lid+width];reduceSum2[lid]+=reduceSum2[lid+width];reduceSum3[lid]+=reduceSum3[lid+width];reduceMaximum[lid]=max(reduceMaximum[lid],reduceMaximum[lid+width]);reduceFlags[lid]|=reduceFlags[lid+width];reduceFirstError[lid]=min(reduceFirstError[lid],reduceFirstError[lid+width]);}workgroupBarrier();width>>=1u;}
}
fn publishSeedPartial(work:u32,lid:u32){if(lid==0u&&work<p.pageCapacity){partials[work]=Partial(reduceSum0[0],0u,0u,0u,0u,reduceFlags[0],reduceFirstError[0]);}}
fn publishResolvePartial(work:u32,lid:u32){if(lid==0u&&work<p.pageCapacity){partials[work].resolveMissing=reduceSum0[0];partials[work].accepted=reduceSum1[0];}}
fn publishValidationPartial(work:u32,lid:u32){if(lid==0u&&work<p.pageCapacity){partials[work].validationUnresolved=reduceSum0[0];partials[work].maximum=reduceMaximum[0];partials[work].flags|=reduceFlags[0];partials[work].firstError=min(partials[work].firstError,reduceFirstError[0]);}}
fn unpackBrick(key:u32)->vec3u{let xy=p.brickDims.x*p.brickDims.y;let z=key/xy;let rem=key-z*xy;let y=rem/p.brickDims.x;return vec3u(rem-y*p.brickDims.x,y,z);}fn packBrick(q:vec3u)->u32{return q.x+p.brickDims.x*(q.y+p.brickDims.y*q.z);}fn localCoord(local:u32)->vec3u{let r=p.brickResolution;let z=local/(r*r);let rem=local-z*r*r;let y=rem/r;return vec3u(rem-y*r,y,z);}fn localIndex(q:vec3u)->u32{return q.x+p.brickResolution*(q.y+p.brickResolution*q.z);}
fn sampleKey(q:vec3u)->u32{return q.x+p.sampleDims.x*(q.y+p.sampleDims.y*q.z);}
${makeFineLevelSetSortedWorklistLookupWGSL("p", "metadata", "worklist", "publishedPageOf", "brickDims")}
fn deltaCount(support:bool)->u32{return min(pageDelta[select(2u,3u,support)],p.pageCapacity);}
fn deltaOffset(support:bool)->u32{return select(p.dirtyPagesOffset,p.supportPagesOffset,support);}
fn rawDeltaPage(work:u32,support:bool)->u32{if(work>=deltaCount(support)){return INVALID;}return pageDelta[deltaOffset(support)+work];}
fn deltaPageAt(work:u32,support:bool)->u32{let id=rawDeltaPage(work,support);return select(INVALID,id,id<p.pageCapacity&&metadata[id*10u+2u]==p.generation);}
fn deltaPage(wid:vec3u,nw:vec3u,support:bool)->u32{return deltaPageAt(fineLinearWorkgroup(wid,nw),support);}
@compute @workgroup_size(64)fn publishSupportPageMask(@builtin(workgroup_id)wid:vec3u,@builtin(num_workgroups)nw:vec3u,@builtin(local_invocation_index)lid:u32){if(lid!=0u){return;}let work=fineLinearWorkgroup(wid,nw);let id=deltaPageAt(work,true);if(id!=INVALID){supportMask[id]=p.generation;}}
fn supportPageOf(key:u32)->u32{let id=publishedPageOf(key);return select(INVALID,id,id<p.pageCapacity&&supportMask[id]==p.generation);}
fn prepareFloodPageIds(id:u32,lid:u32){
 var page=INVALID;
 if(lid<27u&&id!=INVALID&&id<p.pageCapacity){
  let center=unpackBrick(metadata[id*10u+1u]);let z=lid/9u;let rem=lid-z*9u;let y=rem/3u;let x=rem-y*3u;
  let direction=vec3i(i32(x)-1,i32(y)-1,i32(z)-1);
  let radius=max(1u,(JFA_STRIDE+p.brickResolution-1u)/p.brickResolution);
  let neighbor=vec3i(center)+direction*i32(radius);
  if(all(neighbor>=vec3i(0))&&all(neighbor<vec3i(p.brickDims))){
   page=supportPageOf(packBrick(vec3u(neighbor)));
  }
 }
 if(lid<27u){floodPageIds[lid]=page;}workgroupBarrier();
}
fn cachedFloodSampleIndex(q:vec3u,center:vec3u)->u32{
 if(any(q>=p.sampleDims)){return INVALID;}let brick=q/p.brickResolution;let delta=vec3i(brick)-vec3i(center);
 let radius=i32(max(1u,(JFA_STRIDE+p.brickResolution-1u)/p.brickResolution));
 if(any(abs(delta)>vec3i(radius))||any((delta!=vec3i(0))&(abs(delta)!=vec3i(radius)))){return INVALID;}
 var slot=vec3u(1u);if(delta.x<0){slot.x=0u;}else if(delta.x>0){slot.x=2u;}if(delta.y<0){slot.y=0u;}else if(delta.y>0){slot.y=2u;}if(delta.z<0){slot.z=0u;}else if(delta.z>0){slot.z=2u;}
 let id=floodPageIds[slot.x+3u*(slot.y+3u*slot.z)];if(id==INVALID){return INVALID;}
 let index=id*p.samplesPerBrick+localIndex(q%p.brickResolution);return select(INVALID,index,finite(bitcast<f32>(phi[index])));
}
fn deltaRecordError(work:u32,support:bool)->u32{if(work>=deltaCount(support)){return 0u;}let id=rawDeltaPage(work,support);if(id>=p.pageCapacity||metadata[id*10u+2u]!=p.generation){return STALE;}let key=metadata[id*10u+1u];if(publishedPageOf(key)!=id){return STALE;}if(work>0u){let previous=rawDeltaPage(work-1u,support);if(previous>=p.pageCapacity||metadata[previous*10u+2u]!=p.generation||metadata[previous*10u+1u]>=key){return STALE;}}if(!support&&supportPageOf(key)!=id){return STALE;}return 0u;}
fn sampleIndex(q:vec3u)->u32{if(any(q>=p.sampleDims)){return INVALID;}let id=supportPageOf(packBrick(q/p.brickResolution));if(id==INVALID){return INVALID;}let index=id*p.samplesPerBrick+localIndex(q%p.brickResolution);return select(INVALID,index,finite(bitcast<f32>(phi[index])));}
fn directionDelta(direction:u32)->vec3i{if(direction==0u){return vec3i(-1,0,0);}if(direction==1u){return vec3i(1,0,0);}if(direction==2u){return vec3i(0,-1,0);}if(direction==3u){return vec3i(0,1,0);}if(direction==4u){return vec3i(0,0,-1);}return vec3i(0,0,1);}
fn physicalSampleQ(index:u32)->vec3u{let id=index/p.samplesPerBrick;let local=index-id*p.samplesPerBrick;return unpackBrick(metadata[id*10u+1u])*p.brickResolution+localCoord(local);}
fn seedStableKey(index:u32)->u32{return sampleKey(physicalSampleQ(index));}
fn seedClosestPointCode(q:vec3u,index:u32)->u32{let center=bitcast<f32>(phi[index]);if(center==0.){return 6u<<24u;}var best=LARGE;var bestDirection=INVALID;var bestFraction=0.;for(var direction=0u;direction<6u;direction+=1u){let nq=vec3i(q)+directionDelta(direction);if(any(nq<vec3i(0))||any(nq>=vec3i(p.sampleDims))){continue;}let neighbor=sampleIndex(vec3u(nq));if(neighbor==INVALID){continue;}let other=bitcast<f32>(phi[neighbor]);if(!finite(other)||(other<0.)==(center<0.)){continue;}let denominator=abs(center)+abs(other);let fraction=select(0.,abs(center)/denominator,denominator>0.);let d2=fraction*fraction;if(d2<best||(d2==best&&direction<bestDirection)){best=d2;bestDirection=direction;bestFraction=fraction;}}if(bestDirection==INVALID){return INVALID;}let quantized=u32(round(clamp(bestFraction,0.,1.)*CP_FRACTION_SCALE));if(quantized==0u){return 7u<<24u;}return (bestDirection<<24u)|(quantized&CP_FRACTION_MASK);}
fn hasCachedClosestPoint(index:u32)->bool{return (flags[index]>>SAMPLE_FLAG_BITS)!=0u;}
fn materializedClosestPoint(index:u32)->vec3f{let q=physicalSampleQ(index);let code=flags[index]>>SAMPLE_FLAG_BITS;let direction=code>>24u;let fraction=f32(code&CP_FRACTION_MASK)/CP_FRACTION_SCALE;return vec3f(q)+vec3f(.5)+select(vec3f(directionDelta(direction))*fraction,vec3f(0.),direction>=6u);}
fn flood(index:u32,q:vec3u,centerBrick:vec3u,fromA:bool)->u32{let point=vec3f(q)+vec3f(.5);var best=select(workB[index],workA[index],fromA);var bestD=LARGE;if(best!=INVALID){let delta=point-materializedClosestPoint(best);bestD=dot(delta,delta);}let stride=i32(JFA_STRIDE);for(var dz=-1;dz<=1;dz+=1){for(var dy=-1;dy<=1;dy+=1){for(var dx=-1;dx<=1;dx+=1){let nq=vec3i(q)+vec3i(dx,dy,dz)*stride;if(any(nq<vec3i(0))||any(nq>=vec3i(p.sampleDims))){continue;}let candidateIndex=cachedFloodSampleIndex(vec3u(nq),centerBrick);if(candidateIndex==INVALID){continue;}let candidate=select(workB[candidateIndex],workA[candidateIndex],fromA);if(candidate==INVALID||candidate==best){continue;}let delta=point-materializedClosestPoint(candidate);let d=dot(delta,delta);if(d<bestD||(d==bestD&&seedStableKey(candidate)<seedStableKey(best))){best=candidate;bestD=d;}}}}return best;}
fn resolvedDistance(seed:u32,q:vec3u)->f32{if(seed==INVALID){return bandDistance();}return length((vec3f(q)+vec3f(.5))-materializedClosestPoint(seed))*p.fineWidth;}
fn distanceValue(index:u32)->f32{return bitcast<f32>(workB[index]);}
fn resolvedSeed(index:u32)->u32{return workA[index];}
@compute @workgroup_size(64)fn seedClosestPoints(@builtin(workgroup_id)wid:vec3u,@builtin(local_invocation_index)lid:u32,@builtin(num_workgroups)nw:vec3u){
  let work=fineLinearWorkgroup(wid,nw);let id=deltaPage(wid,nw,true);var seedCount=0u;var errorFlags=0u;var firstError=INVALID;
  if(lid==0u){errorFlags=deltaRecordError(work,true);firstError=select(INVALID,work,errorFlags!=0u);}
  if(id!=INVALID&&lid<p.samplesPerBrick){let index=id*p.samplesPerBrick+lid;let brick=unpackBrick(metadata[id*10u+1u]);let q=brick*p.brickResolution+localCoord(lid);workA[index]=INVALID;workB[index]=INVALID;flags[index]&=(1u<<SAMPLE_FLAG_BITS)-1u;if(!any(q>=p.sampleDims)){let value=bitcast<f32>(phi[index]);if(!finite(value)){errorFlags|=NONFINITE;firstError=min(firstError,index);}else{let closest=seedClosestPointCode(q,index);if(closest!=INVALID){workA[index]=index;flags[index]|=closest<<SAMPLE_FLAG_BITS;seedCount=1u;}}}}
  reduceLane(lid,seedCount,0u,0u,0u,0u,errorFlags,firstError,32u);publishSeedPartial(work,lid);
}
@compute @workgroup_size(64)fn jumpFloodAToB(@builtin(workgroup_id)wid:vec3u,@builtin(local_invocation_index)lid:u32,@builtin(num_workgroups)nw:vec3u){let id=deltaPage(wid,nw,true);prepareFloodPageIds(id,lid);if(id==INVALID||lid>=p.samplesPerBrick){return;}let brick=unpackBrick(metadata[id*10u+1u]);let index=id*p.samplesPerBrick+lid;let q=brick*p.brickResolution+localCoord(lid);if(any(q>=p.sampleDims)||!finite(bitcast<f32>(phi[index]))){return;}workB[index]=flood(index,q,brick,true);}
@compute @workgroup_size(64)fn jumpFloodBToA(@builtin(workgroup_id)wid:vec3u,@builtin(local_invocation_index)lid:u32,@builtin(num_workgroups)nw:vec3u){let id=deltaPage(wid,nw,true);prepareFloodPageIds(id,lid);if(id==INVALID||lid>=p.samplesPerBrick){return;}let brick=unpackBrick(metadata[id*10u+1u]);let index=id*p.samplesPerBrick+lid;let q=brick*p.brickResolution+localCoord(lid);if(any(q>=p.sampleDims)||!finite(bitcast<f32>(phi[index]))){return;}workA[index]=flood(index,q,brick,false);}
@compute @workgroup_size(64)fn resolveClosestPointsAToB(@builtin(workgroup_id)wid:vec3u,@builtin(local_invocation_index)lid:u32,@builtin(num_workgroups)nw:vec3u){
  let work=fineLinearWorkgroup(wid,nw);let id=deltaPage(wid,nw,true);var unresolved=0u;var accepted=0u;if(id!=INVALID&&lid<p.samplesPerBrick){let index=id*p.samplesPerBrick+lid;let q=unpackBrick(metadata[id*10u+1u])*p.brickResolution+localCoord(lid);let transported=bitcast<f32>(phi[index]);if(!any(q>=p.sampleDims)&&finite(transported)){var seed=workA[index];if(hasCachedClosestPoint(index)){seed=index;}unresolved=select(0u,1u,seed==INVALID&&abs(transported)<bandDistance());let d=resolvedDistance(seed,q);workA[index]=seed;workB[index]=bitcast<u32>(d);accepted=select(0u,1u,seed!=INVALID&&d<=bandDistance());}}reduceLane(lid,unresolved,accepted,0u,0u,0u,0u,INVALID,32u);publishResolvePartial(work,lid);
}
@compute @workgroup_size(64)fn resolveClosestPointsBToCanonical(@builtin(workgroup_id)wid:vec3u,@builtin(local_invocation_index)lid:u32,@builtin(num_workgroups)nw:vec3u){
  let work=fineLinearWorkgroup(wid,nw);let id=deltaPage(wid,nw,true);var unresolved=0u;var accepted=0u;if(id!=INVALID&&lid<p.samplesPerBrick){let index=id*p.samplesPerBrick+lid;let q=unpackBrick(metadata[id*10u+1u])*p.brickResolution+localCoord(lid);let transported=bitcast<f32>(phi[index]);if(!any(q>=p.sampleDims)&&finite(transported)){var seed=workB[index];if(hasCachedClosestPoint(index)){seed=index;}unresolved=select(0u,1u,seed==INVALID&&abs(transported)<bandDistance());let d=resolvedDistance(seed,q);workA[index]=seed;workB[index]=bitcast<u32>(d);accepted=select(0u,1u,seed!=INVALID&&d<=bandDistance());}}reduceLane(lid,unresolved,accepted,0u,0u,0u,0u,INVALID,32u);publishResolvePartial(work,lid);
}
@compute @workgroup_size(64)fn validateJFADistances(@builtin(workgroup_id)wid:vec3u,@builtin(local_invocation_index)lid:u32,@builtin(num_workgroups)nw:vec3u){
  let work=fineLinearWorkgroup(wid,nw);let id=deltaPage(wid,nw,false);var unresolved=0u;var residual=0u;var errorFlags=0u;var firstError=INVALID;
  if(lid==0u){errorFlags=deltaRecordError(work,false);firstError=select(INVALID,work,errorFlags!=0u);}
  if(id!=INVALID&&lid<p.samplesPerBrick){let index=id*p.samplesPerBrick+lid;let transported=bitcast<f32>(phi[index]);let d=distanceValue(index);if(!finite(transported)||!finite(d)){unresolved=1u;errorFlags|=NONFINITE;firstError=min(firstError,index);}else if(d<bandDistance()&&!hasCachedClosestPoint(index)){let q=unpackBrick(metadata[id*10u+1u])*p.brickResolution+localCoord(lid);var sum=0.;for(var axis=0u;axis<3u;axis+=1u){var nearest=d;for(var side=-1;side<=1;side+=2){var nq=vec3i(q);nq[axis]+=side;if(any(nq<vec3i(0))||any(nq>=vec3i(p.sampleDims))){continue;}let neighbor=sampleIndex(vec3u(nq));if(neighbor!=INVALID){nearest=min(nearest,distanceValue(neighbor));}}let gradient=max(0.,d-nearest)/p.fineWidth;sum+=gradient*gradient;}residual=u32(min(4294967295.,abs(sqrt(sum)-1.)*1000000.));unresolved=select(0u,1u,residual>u32(p.tolerance*1000000.));}}
  reduceLane(lid,unresolved,0u,0u,0u,residual,errorFlags,firstError,32u);publishValidationPartial(work,lid);
}
@compute @workgroup_size(256)fn finalizeJFADistances(@builtin(local_invocation_index)lid:u32){
  let support=min(pageDelta[3],p.pageCapacity);let dirty=min(pageDelta[2],p.pageCapacity);
  var seeds=0u;var resolveMissing=0u;var accepted=0u;var validationUnresolved=0u;var maximum=0u;var errorFlags=0u;var firstError=INVALID;
  for(var record=lid;record<support;record+=256u){let value=partials[record];seeds+=value.seeds;resolveMissing+=value.resolveMissing;accepted+=value.accepted;validationUnresolved+=value.validationUnresolved;maximum=max(maximum,value.maximum);errorFlags|=value.flags;firstError=min(firstError,value.firstError);}
  reduceLane(lid,resolveMissing,validationUnresolved,seeds,accepted,maximum,errorFlags,firstError,128u);
  if(lid==0u){let malformed=pageDelta[2]>p.pageCapacity||pageDelta[3]>p.pageCapacity||pageDelta[2]>pageDelta[3];let stale=pageDelta[1]!=p.generation||malformed;control.resolveMissing=reduceSum0[0];control.unresolved=reduceSum0[0]+reduceSum1[0];control.seeds=reduceSum2[0];control.accepted=reduceSum3[0];control.residualScaled=reduceMaximum[0];control.flags=reduceFlags[0]|select(0u,STALE,stale);control.firstError=min(reduceFirstError[0],select(INVALID,select(pageDelta[1],2u,malformed),stale));control.initialPages=dirty;control.finalPages=dirty;control.committed=select(0u,1u,control.flags==0u&&control.unresolved==0u&&(control.seeds>0u||pageDelta[2]==0u));}
}
@compute @workgroup_size(64)fn commitJFADistances(@builtin(workgroup_id)wid:vec3u,@builtin(local_invocation_index)lid:u32,@builtin(num_workgroups)nw:vec3u){
  if(control.committed==0u){return;}let id=deltaPage(wid,nw,false);if(id==INVALID||lid>=p.samplesPerBrick){return;}
  let index=id*p.samplesPerBrick+lid;let transported=bitcast<f32>(phi[index]);let d=distanceValue(index);let negative=transported<0.;let closestPointCode=flags[index]&~((1u<<SAMPLE_FLAG_BITS)-1u);let isInterface=closestPointCode!=0u;phi[index]=bitcast<u32>(select(d,-d,negative));if(resolvedSeed(index)==INVALID||d>bandDistance()){flags[index]=0u;}else{flags[index]=closestPointCode|VALID|select(0u,INTERFACE,isInterface)|select(0u,NEGATIVE,negative);}
}
`;
