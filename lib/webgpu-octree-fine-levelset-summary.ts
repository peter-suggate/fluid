import type { FineLevelSetBrickPlan } from "./octree-fine-levelset-bricks";
import { fineLevelSetLinearWorkgroupWGSL, planFineLevelSetDispatch2D } from "./webgpu-fine-levelset-dispatch";
import type { WebGPUFineLevelSetBrickSource } from "./webgpu-octree-fine-levelset-bricks";

export const FINE_LEVELSET_SUMMARY_VALID = 0x8000_0000;
export const FINE_LEVELSET_SUMMARY_COARSE_AUTHORITY = 0x8000_0000;
export const FINE_LEVELSET_SUMMARY_ERROR = Object.freeze({ capacity: 1, hashProbe: 2, staleGeneration: 4,
  nonfinite: 8 } as const);

export interface FineLevelSetGPUSummaryPlan {
  readonly maximumResidentBricks: number; readonly maximumLevel: number;
  readonly fineEntryCapacity: number; readonly coarseEntryCapacity: number; readonly entryCapacity: number;
  readonly hierarchyKeyCapacity: number; readonly hashCapacity: number;
  readonly directoryBytes: number; readonly parameterBytes: number; readonly allocatedBytes: number;
  readonly levelOffsets: readonly number[]; readonly levelDimensions: readonly (readonly [number, number, number])[];
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
    || (summary.entryFlags & 0x7fff_ffff) !== 0
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
  let hashCapacity = 1; while (hashCapacity < Math.max(1, entryCapacity * 2)) hashCapacity *= 2;
  const directoryBytes = 64 + hashCapacity * 32;
  const parameterBytes = levelOffsets.length * 96;
  return { maximumResidentBricks: plan.maximumResidentBricks, maximumLevel: levelOffsets.length - 1,
    fineEntryCapacity, coarseEntryCapacity, entryCapacity,
    hierarchyKeyCapacity, hashCapacity, directoryBytes, parameterBytes,
    allocatedBytes: directoryBytes + parameterBytes, levelOffsets, levelDimensions };
}

/** Sparse GPU hierarchy of conservative min/max/min-absolute fine-phi summaries. */
export class WebGPUFineLevelSetSummaries {
  readonly plan: FineLevelSetGPUSummaryPlan;
  readonly directory: GPUBuffer;
  private readonly params: readonly GPUBuffer[];
  private readonly resetPipeline: GPUComputePipeline;
  private readonly basePipeline: GPUComputePipeline;
  private readonly coarsePipeline: GPUComputePipeline;
  private readonly parentPipeline: GPUComputePipeline;
  private readonly finalizePipeline: GPUComputePipeline;
  private destroyed = false;

  constructor(private readonly device: GPUDevice, readonly finePlan: FineLevelSetBrickPlan,
    coarseEntryCapacity = 0) {
    this.plan = planFineLevelSetGPUSummaries(finePlan, coarseEntryCapacity);
    this.directory = device.createBuffer({ label: "global fine sparse summary hierarchy", size: this.plan.directoryBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    this.params = this.plan.levelOffsets.map((_, level) => device.createBuffer({
      label: `global fine summary parameters level ${level}`, size: 96,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    }));
    const shaderModule = device.createShaderModule({ label: "global fine sparse summary hierarchy",
      code: fineLevelSetSummaryWGSL });
    const pipeline = (entryPoint: string) => device.createComputePipeline({ label: entryPoint, layout: "auto",
      compute: { module: shaderModule, entryPoint } });
    this.resetPipeline = pipeline("resetFineSummaries"); this.basePipeline = pipeline("summarizeFineBricks");
    this.coarsePipeline = pipeline("mergeCoarsePhiSummaries");
    this.parentPipeline = pipeline("propagateFineSummaries"); this.finalizePipeline = pipeline("finalizeFineSummaries");
  }

  encode(encoder: GPUCommandEncoder, source: WebGPUFineLevelSetBrickSource,
    coarse?: { directory: GPUBuffer; hashCapacity: number }): void {
    if (this.destroyed) throw new Error("Fine summary hierarchy is destroyed");
    if (source.plan !== this.finePlan && JSON.stringify(source.plan) !== JSON.stringify(this.finePlan)) {
      throw new RangeError("Fine summary source does not match its configured lattice");
    }
    if (coarse && this.plan.coarseEntryCapacity === 0) {
      throw new RangeError("Fine summary coarse merge requires a configured compact-row capacity");
    }
    for (let level = 0; level < this.params.length; level += 1) {
      const data = new Uint32Array(24); const dims = this.plan.levelDimensions[level];
      data.set(this.finePlan.brickDimensions, 0); data[3] = this.finePlan.samplesPerBrick;
      data.set([this.finePlan.maximumResidentBricks, this.plan.hashCapacity,
        this.finePlan.maximumHashProbes, source.generation], 4);
      data.set([level, this.plan.levelOffsets[level], dims[0] * dims[1] * dims[2]], 8);
      // WGSL vec3 members are 16-byte aligned: word 11 and words 17..19 are
      // explicit host-side padding in this 96-byte uniform ABI.
      data.set(dims, 12); data[15] = this.plan.maximumLevel; data[16] = this.plan.hierarchyKeyCapacity;
      data.set(this.finePlan.finestCellDimensions, 20);
      this.device.queue.writeBuffer(this.params[level], 0, data);
    }
    const bind = (pipeline: GPUComputePipeline, params: GPUBuffer | undefined,
      buffers: readonly (readonly [number, GPUBuffer])[]) => this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0), entries: [
        ...(params ? [{ binding: 0, resource: { buffer: params } }] : []),
        ...buffers.map(([binding, buffer]) => ({ binding, resource: { buffer } })),
      ],
    });
    const run = (pipeline: GPUComputePipeline, group: GPUBindGroup, groups: number, label: string) => {
      const pass = encoder.beginComputePass({ label }); pass.setPipeline(pipeline); pass.setBindGroup(0, group);
      const dispatch = planFineLevelSetDispatch2D(groups, this.device.limits.maxComputeWorkgroupsPerDimension);
      pass.dispatchWorkgroups(dispatch.x, dispatch.y); pass.end();
    };
    run(this.resetPipeline, bind(this.resetPipeline, this.params[0], [[5, this.directory]]),
      Math.ceil(this.plan.hashCapacity / 64), "Reset global fine summaries");
    run(this.basePipeline, bind(this.basePipeline, this.params[0], [[1, source.metadata], [2, source.worklist],
      [3, source.flags], [4, source.phi], [5, this.directory]]), this.finePlan.maximumResidentBricks, "Summarize resident fine bricks");
    for (let level = 0; level < this.plan.maximumLevel; level += 1) {
      run(this.parentPipeline, bind(this.parentPipeline, this.params[level + 1], [[5, this.directory]]),
        Math.ceil(this.plan.hashCapacity / 64), `Propagate global fine summaries level ${level + 1}`);
    }
    if (coarse) run(this.coarsePipeline,
      bind(this.coarsePipeline, this.params[0], [[5, this.directory], [6, coarse.directory]]),
      Math.ceil(coarse.hashCapacity / 64), "Merge corrected coarse phi into fine summaries");
    run(this.finalizePipeline, bind(this.finalizePipeline, undefined, [[5, this.directory]]), 1,
      "Publish global fine summaries");
  }

  destroy(): void { if (this.destroyed) return; this.destroyed = true;
    this.directory.destroy(); this.params.forEach((buffer) => buffer.destroy()); }
}

/**
 * Each sparse entry is eight words: key, ordered min/max phi, min |phi|,
 * exact sample/brick counts, flags, then an atomic f32 sum.  Flags reserve
 * bit 31 for corrected-coarse authority and low bits for fail-closed errors.
 * The final sum is the eight trilinear fine samples at each finest-cell
 * centre; parents sum brick shares, and consumers only interpret it at the
 * size-1 node where it is a single cell-centre phi.
 */
export const fineLevelSetSummaryWGSL = /* wgsl */ `
${fineLevelSetLinearWorkgroupWGSL}
const INVALID:u32=0xffffffffu;const VALID:u32=1u;const PUBLISHED:u32=0x80000000u;
const CAPACITY:u32=1u;const HASH:u32=2u;const STALE:u32=4u;const NONFINITE:u32=8u;
const COARSE_AUTHORITY:u32=0x80000000u;
struct P{baseDims:vec3u,samplesPerBrick:u32,pageCapacity:u32,hashCapacity:u32,maxProbes:u32,generation:u32,
 level:u32,levelOffset:u32,levelKeyCount:u32,levelDims:vec3u,maximumLevel:u32,hierarchyKeyCapacity:u32,finestDims:vec3u,pad:u32}
@group(0)@binding(0)var<uniform>p:P;@group(0)@binding(1)var<storage,read_write>a:array<u32>;
@group(0)@binding(2)var<storage,read_write>b:array<u32>;@group(0)@binding(3)var<storage,read_write>c:array<u32>;
@group(0)@binding(4)var<storage,read_write>d:array<u32>;@group(0)@binding(5)var<storage,read_write>directory:array<atomic<u32>>;
struct CoarseEntry{cellPlusOne:u32,size:u32,phi:f32,minimumPhi:f32,maximumPhi:f32,flags:u32,row:u32,physicalVolume:f32}
struct CoarseDirectory{state:u32,generation:u32,hashCapacity:u32,maximumLeafSize:u32,dimensions:vec3u,physicalCellSize:f32,entries:array<CoarseEntry>}
@group(0)@binding(6)var<storage,read>coarse:CoarseDirectory;
var<workgroup> minimumPhi:array<f32,64>;var<workgroup> maximumPhi:array<f32,64>;
var<workgroup> minimumAbsolutePhi:array<f32,64>;var<workgroup> centerContribution:array<f32,64>;
var<workgroup> validSamples:array<u32,64>;var<workgroup> errors:array<u32,64>;
fn finite(v:f32)->bool{return v==v&&abs(v)<3.402823e38;}
fn ordered(v:f32)->u32{let bits=bitcast<u32>(v);return select(bits^0x80000000u,~bits,(bits&0x80000000u)!=0u);}
fn hash(key:u32)->u32{var v=key*0x9e3779b1u;v=(v^(v>>16u))*0x7feb352du;return v^(v>>15u);}
fn entryBase(slot:u32)->u32{return 16u+slot*8u;}
fn atomicAddFloat(address:ptr<storage,atomic<u32>,read_write>,value:f32){var old=atomicLoad(address);loop{let next=bitcast<u32>(bitcast<f32>(old)+value);let result=atomicCompareExchangeWeak(address,old,next);if(result.exchanged){return;}old=result.old_value;}}
fn merge(key:u32,minPhi:f32,maxPhi:f32,minAbs:f32,samples:u32,bricks:u32,flags:u32,centerPhi:f32){let start=hash(key)&(p.hashCapacity-1u);
 for(var probe=0u;probe<32u;probe+=1u){if(probe>=p.maxProbes){break;}let slot=(start+probe)&(p.hashCapacity-1u);let base=entryBase(slot);let result=atomicCompareExchangeWeak(&directory[base],INVALID,key);
  if(result.exchanged||result.old_value==key){if(result.exchanged){atomicAdd(&directory[2],1u);}atomicMin(&directory[base+1u],ordered(minPhi));atomicMax(&directory[base+2u],ordered(maxPhi));atomicMin(&directory[base+3u],bitcast<u32>(minAbs));atomicAdd(&directory[base+4u],samples);atomicAdd(&directory[base+5u],bricks);atomicOr(&directory[base+6u],flags);atomicAddFloat(&directory[base+7u],centerPhi);return;}}
 atomicOr(&directory[0],HASH);}
@compute @workgroup_size(64)fn resetFineSummaries(@builtin(workgroup_id)w:vec3u,@builtin(local_invocation_index)lid:u32,@builtin(num_workgroups)n:vec3u){let flat=fineLinearWorkgroup(w,n)*64u+lid;if(flat==0u){atomicStore(&directory[0],0u);atomicStore(&directory[1],p.generation);atomicStore(&directory[2],0u);atomicStore(&directory[3],p.hashCapacity);atomicStore(&directory[4],p.baseDims.x);atomicStore(&directory[5],p.baseDims.y);atomicStore(&directory[6],p.baseDims.z);atomicStore(&directory[7],p.maximumLevel);atomicStore(&directory[8],p.maxProbes);atomicStore(&directory[9],0u);atomicStore(&directory[10],p.hierarchyKeyCapacity);}
 if(flat==0u){atomicStore(&directory[11],p.samplesPerBrick);}
 if(flat<p.hashCapacity){let base=entryBase(flat);atomicStore(&directory[base],INVALID);atomicStore(&directory[base+1u],0xffffffffu);atomicStore(&directory[base+2u],0u);atomicStore(&directory[base+3u],bitcast<u32>(3.402823e38));atomicStore(&directory[base+4u],0u);atomicStore(&directory[base+5u],0u);atomicStore(&directory[base+6u],0u);atomicStore(&directory[base+7u],0u);}}
@compute @workgroup_size(64)fn summarizeFineBricks(@builtin(workgroup_id)group:vec3u,@builtin(local_invocation_id)l:vec3u,@builtin(num_workgroups)n:vec3u){let activeCount=min(b[0],p.pageCapacity);let page=fineLinearWorkgroup(group,n);var lo=3.402823e38;var hi=-3.402823e38;var ma=3.402823e38;var centre=0.0;var count=0u;var failure=0u;
 if(page<activeCount){let id=b[5u+page];if(id>=p.pageCapacity||a[id*10u+2u]!=p.generation){failure=STALE;}else{let key=a[id*10u+1u];let xy=p.baseDims.x*p.baseDims.y;let bz=key/xy;let rem=key-bz*xy;let by=rem/p.baseDims.x;let brick=vec3u(rem-by*p.baseDims.x,by,bz);let ratio=p.baseDims/p.finestDims;let resolution=select(4u,8u,p.samplesPerBrick==512u);let fineFactor=ratio.x*resolution;let cell=brick/ratio.x;let centerLow=cell*fineFactor+vec3u(fineFactor/2u-1u);for(var local=l.x;local<p.samplesPerBrick;local+=64u){let index=id*p.samplesPerBrick+local;if((c[index]&VALID)==0u){continue;}let value=bitcast<f32>(d[index]);if(!finite(value)){failure|=NONFINITE;continue;}lo=min(lo,value);hi=max(hi,value);ma=min(ma,abs(value));count+=1u;let lz=local/(resolution*resolution);let lr=local-lz*resolution*resolution;let ly=lr/resolution;let sample=brick*resolution+vec3u(lr-ly*resolution,ly,lz);let delta=sample-centerLow;if(all(delta<=vec3u(1u))){centre+=0.125*value;}}}}
 minimumPhi[l.x]=lo;maximumPhi[l.x]=hi;minimumAbsolutePhi[l.x]=ma;centerContribution[l.x]=centre;validSamples[l.x]=count;errors[l.x]=failure;workgroupBarrier();for(var stride=32u;stride>0u;stride/=2u){if(l.x<stride){minimumPhi[l.x]=min(minimumPhi[l.x],minimumPhi[l.x+stride]);maximumPhi[l.x]=max(maximumPhi[l.x],maximumPhi[l.x+stride]);minimumAbsolutePhi[l.x]=min(minimumAbsolutePhi[l.x],minimumAbsolutePhi[l.x+stride]);centerContribution[l.x]+=centerContribution[l.x+stride];validSamples[l.x]+=validSamples[l.x+stride];errors[l.x]|=errors[l.x+stride];}workgroupBarrier();}
 if(l.x==0u&&page<activeCount){if(errors[0]!=0u){atomicOr(&directory[0],errors[0]);}else if(validSamples[0]>0u){let id=b[5u+page];let key=a[id*10u+1u];if(key>=p.levelKeyCount){atomicOr(&directory[0],STALE);}else{merge(key,minimumPhi[0],maximumPhi[0],minimumAbsolutePhi[0],validSamples[0],1u,0u,centerContribution[0]);}}}}
@compute @workgroup_size(64)fn mergeCoarsePhiSummaries(@builtin(global_invocation_id)g:vec3u){let sourceSlot=g.x;if(sourceSlot>=coarse.hashCapacity||sourceSlot>=arrayLength(&coarse.entries)||coarse.state!=PUBLISHED||(coarse.generation&0x3fffffffu)!=(p.generation&0x3fffffffu)||any(coarse.dimensions!=p.finestDims)){return;}let e=coarse.entries[sourceSlot];if(e.cellPlusOne==0u||(e.flags&9u)!=9u||e.size==0u||(e.size&(e.size-1u))!=0u||!finite(e.phi)||!finite(e.minimumPhi)||!finite(e.maximumPhi)||e.minimumPhi>e.phi||e.phi>e.maximumPhi){return;}let ratio=p.baseDims/p.finestDims;if(ratio.x==0u||any(ratio!=vec3u(ratio.x))){return;}var side=e.size*ratio.x;if((side&(side-1u))!=0u){return;}let cell=e.cellPlusOne-1u;let origin=vec3u(cell%p.finestDims.x,(cell/p.finestDims.x)%p.finestDims.y,cell/(p.finestDims.x*p.finestDims.y));var offset=0u;var ld=p.baseDims;var remaining=side;loop{if(remaining==1u){break;}offset+=ld.x*ld.y*ld.z;ld=(ld+vec3u(1u))/2u;remaining>>=1u;}let brickOrigin=origin*ratio.x;if(any(brickOrigin%vec3u(side)!=vec3u(0u))){return;}let q=brickOrigin/side;let key=offset+q.x+ld.x*(q.y+ld.y*q.z);let ma=select(min(abs(e.minimumPhi),abs(e.maximumPhi)),0.0,e.minimumPhi<=0.0&&e.maximumPhi>=0.0);let start=hash(key)&(p.hashCapacity-1u);for(var probe=0u;probe<32u;probe+=1u){if(probe>=p.maxProbes){break;}let slot=(start+probe)&(p.hashCapacity-1u);let base=entryBase(slot);let result=atomicCompareExchangeWeak(&directory[base],INVALID,key);if(result.exchanged||result.old_value==key){if(result.exchanged){atomicAdd(&directory[2],1u);}atomicMin(&directory[base+1u],ordered(e.minimumPhi));atomicMax(&directory[base+2u],ordered(e.maximumPhi));atomicMin(&directory[base+3u],bitcast<u32>(ma));atomicOr(&directory[base+6u],COARSE_AUTHORITY);return;}}atomicOr(&directory[0],HASH);}
@compute @workgroup_size(64)fn propagateFineSummaries(@builtin(workgroup_id)w:vec3u,@builtin(local_invocation_index)lid:u32,@builtin(num_workgroups)n:vec3u){let flat=fineLinearWorkgroup(w,n)*64u+lid;if(flat>=p.hashCapacity){return;}let sourceBase=entryBase(flat);let key=atomicLoad(&directory[sourceBase]);let previousLevel=p.level-1u;var previousOffset=0u;var previousDims=p.baseDims;for(var level=0u;level<previousLevel;level+=1u){previousOffset+=previousDims.x*previousDims.y*previousDims.z;previousDims=(previousDims+vec3u(1u))/2u;}let previousCount=previousDims.x*previousDims.y*previousDims.z;if(key<previousOffset||key>=previousOffset+previousCount){return;}let local=key-previousOffset;let xy=previousDims.x*previousDims.y;let z=local/xy;let rem=local-z*xy;let y=rem/previousDims.x;let coord=vec3u(rem-y*previousDims.x,y,z);let parent=coord/2u;let parentKey=p.levelOffset+parent.x+p.levelDims.x*(parent.y+p.levelDims.y*parent.z);let samples=atomicLoad(&directory[sourceBase+4u]);if(samples==0u){return;}merge(parentKey,bitcast<f32>(atomicLoad(&directory[sourceBase+1u])^select(0x80000000u,0xffffffffu,(atomicLoad(&directory[sourceBase+1u])&0x80000000u)==0u)),bitcast<f32>(atomicLoad(&directory[sourceBase+2u])^select(0x80000000u,0xffffffffu,(atomicLoad(&directory[sourceBase+2u])&0x80000000u)==0u)),bitcast<f32>(atomicLoad(&directory[sourceBase+3u])),samples,atomicLoad(&directory[sourceBase+5u]),atomicLoad(&directory[sourceBase+6u]),bitcast<f32>(atomicLoad(&directory[sourceBase+7u])));}
@compute @workgroup_size(1)fn finalizeFineSummaries(){if(atomicLoad(&directory[0])==0u){atomicStore(&directory[9],PUBLISHED);}else{atomicStore(&directory[9],0u);}}
`;
