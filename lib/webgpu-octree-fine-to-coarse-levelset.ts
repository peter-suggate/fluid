/** GPU-only row-owned restriction with O(rows) storage over resident fine samples. */

import { fineLevelSetLinearWorkgroupWGSL } from "./webgpu-fine-levelset-dispatch";
import type { WebGPUFineLevelSetBrickSource } from "./webgpu-octree-fine-levelset-bricks";
import { PassBroker } from "./webgpu-pass-broker";
import {
  createGPULogicalActivityAdoptionContext,
  type GPULogicalActivityAdoptionContext,
} from "./gpu-logical-activity-adoption";
import { performanceShaderVariant } from "./stores/performance-instrumentation-store";

export const FINE_TO_COARSE_LEVELSET_ERROR = Object.freeze({
  capacity: 1, unowned: 2, nonfinite: 4, unpublishedSource: 8,
} as const);

export interface FineToCoarseGPUPlan {
  readonly rowCapacity: number;
  /** Logical diagnostic workload bound only; no O(samples) allocation is made. */
  readonly sampleCapacity: number;
  readonly blockCount: number;
  readonly aggregateScratchBytes: number;
  readonly allocatedBytes: number;
}

export interface FineToCoarseGPUResult {
  readonly rowOffsets: GPUBuffer;
  /** One 16-byte `{centerPhi,minimumPhi,maximumPhi,valid}` record per row. */
  readonly contributions: GPUBuffer;
  /** First two u32s are aggregateCount and maximumAggregatesPerRow. */
  readonly counts: GPUBuffer;
  readonly aggregated: true;
}

export interface FineToCoarseGPUControl {
  readonly contributionCount: number; readonly maximumContributionsPerRow: number;
  readonly flags: number;
  /** Live rows whose eight corner samples were not all resident-and-valid, so no
   * fine correction was published for them. Never an error: a row outside the
   * fine narrow band is legitimately uncorrected. It is the coverage regression
   * signal for a band-width change, which is otherwise silent because an
   * unaccepted row raises no flag and writes no contribution. */
  readonly unacceptedRows: number; readonly rowCount: number; readonly valid: boolean;
  readonly firstUnownedLiquidLogical?: number; readonly maximumUnownedLiquidMagnitude?: number;
}

export function planFineToCoarseLevelSet(rowCapacity: number, sampleCapacity: number): FineToCoarseGPUPlan {
  if (!Number.isSafeInteger(rowCapacity) || rowCapacity < 1
    || !Number.isSafeInteger(sampleCapacity) || sampleCapacity < 1) {
    throw new RangeError("Fine-to-coarse capacities must be positive integers");
  }
  const blockCount = Math.ceil(rowCapacity / 256), aggregateScratchBytes = rowCapacity * 48;
  return { rowCapacity, sampleCapacity, blockCount, aggregateScratchBytes,
    allocatedBytes: 112 + aggregateScratchBytes + (rowCapacity + 1) * 4 + rowCapacity * 16 + 32 + 12 };
}

export function unpackFineToCoarseGPUControl(words: ArrayLike<number>): FineToCoarseGPUControl {
  if (words.length < 6) throw new RangeError("Fine-to-coarse control requires six words");
  const phi = words.length >= 8
    ? new Float32Array(new Uint32Array([Number(words[7]) >>> 0]).buffer)[0] : undefined;
  return { contributionCount: Number(words[0]) >>> 0, maximumContributionsPerRow: Number(words[1]) >>> 0,
    flags: Number(words[2]) >>> 0, unacceptedRows: Number(words[3]) >>> 0,
    rowCount: Number(words[4]) >>> 0, valid: Number(words[5]) !== 0,
    ...(words.length >= 8 ? { firstUnownedLiquidLogical: Number(words[6]) >>> 0,
      maximumUnownedLiquidMagnitude: phi } : {}) };
}

export class WebGPUFineToCoarseLevelSet {
  readonly plan: FineToCoarseGPUPlan;
  readonly result: FineToCoarseGPUResult;
  private readonly params: GPUBuffer;
  private readonly aggregates: GPUBuffer;
  private readonly dispatch: GPUBuffer;
  readonly control: GPUBuffer;
  private readonly pipelines: Record<string, GPUComputePipeline>;
  private cachedBindings?: {
    readonly fineParams: GPUBuffer; readonly headers: GPUBuffer; readonly rowCount: GPUBuffer;
    readonly rowCountOffsetWords: number;
    readonly topologyControl: GPUBuffer; readonly groups: Readonly<Record<string, GPUBindGroup>>;
  };
  private destroyed = false;

  constructor(private readonly device: GPUDevice, rowCapacity: number, sampleCapacity: number) {
    this.plan = planFineToCoarseLevelSet(rowCapacity, sampleCapacity);
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    this.params = device.createBuffer({ label: "Fine-to-coarse restriction params", size: 112,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.aggregates = device.createBuffer({ label: "Fine-to-coarse row-owned restriction records",
      size: this.plan.aggregateScratchBytes, usage: storage });
    const rowOffsets = device.createBuffer({ label: "Fine-to-coarse aggregate offsets",
      size: (rowCapacity + 1) * 4, usage: storage });
    const contributions = device.createBuffer({ label: "Fine-to-coarse row aggregates",
      size: rowCapacity * 16, usage: storage });
    this.control = device.createBuffer({ label: "Fine-to-coarse restriction control", size: 32, usage: storage });
    this.dispatch = device.createBuffer({ label: "Fine-to-coarse exact row dispatch", size: 12,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT });
    this.result = { rowOffsets, contributions, counts: this.control, aggregated: true };
    const activityProfile = performanceShaderVariant();
    const activity = createGPULogicalActivityAdoptionContext({
      moduleId: "octree/fine-to-coarse-levelset",
      profile: activityProfile,
    });
    activity.describeTask("prepare-restriction", {
      id: "gpu.physics.fine-restriction.prepare",
      label: "Fine restriction · prepare rows",
      phaseId: "adaptive-publication",
    });
    activity.describeTask("restrict-coarse-rows", {
      id: "gpu.physics.fine-restriction.rows",
      label: "Fine restriction · restrict coarse rows",
      phaseId: "adaptive-publication",
    });
    activity.describeTask("publish-restriction", {
      id: "gpu.physics.fine-restriction.publish",
      label: "Fine restriction · publish",
      phaseId: "adaptive-publication",
    });
    const variant = activity.module(
      fineToCoarseLevelSetActivityShader(activity),
      `octree/fine-to-coarse-levelset/${activityProfile.cacheKey}`,
    );
    const shaderModule = device.createShaderModule({ label: "Fine-to-coarse row restriction",
      code: variant.code });
    const pipeline = (entryPoint: string) => activity.registerPipeline(device.createComputePipeline({
      label: entryPoint, layout: "auto", compute: { module: shaderModule, entryPoint },
    }));
    this.pipelines = {
      prepare: pipeline("prepareRestriction"),
      restrict: pipeline("restrictCoarseRows"),
      reconstructFactorOne: pipeline("reconstructFactorOneRows"),
      publish: pipeline("publishRestriction"),
    };
  }

  /** Record exact restriction into the caller's publication pass. */
  encode(broker: PassBroker, fine: WebGPUFineLevelSetBrickSource, input: {
    headers: GPUBuffer; rowCount: GPUBuffer;
    /** Word containing the row count in `rowCount`. The accepted structured
     * control publishes it at word two; legacy compact-count buffers use zero. */
    rowCountOffsetWords?: number;
    /** Control for the topology transaction that produced `fine`. */
    topologyControl: GPUBuffer;
    /** Accept a same-command, validated fine candidate whose commit is deliberately
     * deferred until post-projection redistance. Never permits rejected/stale input. */
    allowValidatedProvisional?: boolean;
    dimensions: readonly [number, number, number]; physicalCellSize: number;
    maximumLeafSize: number;
  }): FineToCoarseGPUResult {
    if (this.destroyed) throw new Error("Fine-to-coarse restriction is destroyed");
    const sampleCount = fine.plan.maximumResidentBricks * fine.plan.samplesPerBrick;
    if (sampleCount > this.plan.sampleCapacity) throw new RangeError("Fine-to-coarse sample source exceeds capacity");
    const data = new ArrayBuffer(112), u = new Uint32Array(data), f = new Float32Array(data);
    u.set(fine.plan.brickDimensions, 0); u[3] = fine.plan.brickResolution;
    u.set(fine.plan.sampleDimensions, 4); u[7] = fine.plan.samplesPerBrick;
    f.set(fine.plan.domainOrigin, 8); f[11] = fine.plan.fineCellWidth;
    u.set([fine.plan.maximumResidentBricks, fine.generation, this.plan.rowCapacity, sampleCount,
      this.plan.rowCapacity, input.allowValidatedProvisional ? 1 : 0], 12);
    const rowCountOffsetWords = input.rowCountOffsetWords ?? 0;
    if (!Number.isSafeInteger(rowCountOffsetWords) || rowCountOffsetWords < 0) {
      throw new RangeError("Fine-to-coarse row-count word offset must be non-negative");
    }
    u[18] = rowCountOffsetWords;
    u.set(input.dimensions, 20); u[23] = input.maximumLeafSize; f[24] = input.physicalCellSize;
    this.device.queue.writeBuffer(this.params, 0, data);
    const cached = this.cachedBindings;
    let groups = cached?.fineParams === fine.params && cached.headers === input.headers
      && cached.rowCount === input.rowCount && cached.topologyControl === input.topologyControl
      && cached.rowCountOffsetWords === rowCountOffsetWords
      ? cached.groups : undefined;
    if (!groups) {
      const buffers = new Map<number, GPUBuffer>([[0, this.params], [1, fine.metadata], [2, fine.worklist],
        [3, fine.flags], [4, fine.phi], [5, input.headers], [7, input.rowCount],
        [8, this.aggregates], [9, this.result.rowOffsets], [12, this.result.contributions], [13, this.control],
        [14, input.topologyControl], [15, this.dispatch]]);
      const used: Record<string, number[]> = {
      prepare: [0, 2, 7, 9, 13, 14, 15],
      restrict: [0, 1, 2, 3, 4, 5, 8, 9, 12, 13],
      reconstructFactorOne: [0, 1, 2, 3, 4, 5, 8, 12, 13],
      publish: [8, 13],
      };
      groups = Object.freeze(Object.fromEntries(Object.entries(used).map(([name, bindings]) => {
        const pipeline = this.pipelines[name]!;
        return [name, this.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0),
          entries: bindings.map((binding) => ({ binding, resource: { buffer: buffers.get(binding)! } })) })];
      })));
      this.cachedBindings = { fineParams: fine.params, headers: input.headers,
        rowCount: input.rowCount, rowCountOffsetWords,
        topologyControl: input.topologyControl, groups };
    }
    const run = (name: string) => { const pipeline = this.pipelines[name]!;
      const pass = broker.compute({ label: `Fine-to-coarse restriction · ${name}` }); pass.setPipeline(pipeline);
      pass.setBindGroup(0, groups[name]!); pass.dispatchWorkgroups(1); };
    run("prepare");
    broker.fence("fine-to-coarse exact row dispatch published");
    const restrict = broker.compute({ label: "Fine-to-coarse restriction · restrict" });
    restrict.setPipeline(this.pipelines.restrict!); restrict.setBindGroup(0, groups.restrict!);
    restrict.dispatchWorkgroupsIndirect(this.dispatch, 0);
    if (fine.plan.fineFactor === 1) {
      const reconstruct = broker.compute({
        label: "Fine-to-coarse restriction · factor-1 trilinear cell reconstruction",
      });
      reconstruct.setPipeline(this.pipelines.reconstructFactorOne!);
      reconstruct.setBindGroup(0, groups.reconstructFactorOne!);
      reconstruct.dispatchWorkgroupsIndirect(this.dispatch, 0);
    }
    run("publish");
    broker.fence("fine-to-coarse restriction complete");
    return this.result;
  }

  destroy(): void { if (this.destroyed) return; this.destroyed = true;
    this.params.destroy(); this.aggregates.destroy(); this.result.rowOffsets.destroy();
    this.result.contributions.destroy(); this.control.destroy(); this.dispatch.destroy(); }
}

export const fineToCoarseLevelSetWGSL = /* wgsl */ `
${fineLevelSetLinearWorkgroupWGSL}
struct P{brickDims:vec3u,brickResolution:u32,sampleDims:vec3u,samplesPerBrick:u32,origin:vec3f,fineWidth:f32,
 pageCapacity:u32,generation:u32,rowCapacity:u32,sampleCapacity:u32,directoryCapacity:u32,reserved:u32,
 rowCountOffsetWords:u32,pad0:u32,dimensions:vec3u,maxLeaf:u32,cellWidth:f32}
struct H{cell:u32,a:u32,b:u32,size:u32,x:f32,y:f32,z:u32,w:u32,g:vec4f}
struct C{count:u32,maximumPerRow:u32,flags:u32,unacceptedRows:u32,rowCount:u32,valid:u32,firstUnownedLiquid:u32,maximumUnownedLiquidMagnitude:u32}
struct Aggregate{centerPhi:f32,minimumPhi:f32,maximumPhi:f32,valid:u32,sampleCount:u32,error:u32,pad:array<u32,6>}
struct Contribution{centerPhi:f32,minimumPhi:f32,maximumPhi:f32,valid:u32}struct Sample{positionPhi:vec4f,logical:u32,valid:u32}
@group(0)@binding(0)var<uniform>p:P;@group(0)@binding(1)var<storage,read>metadata:array<u32>;@group(0)@binding(2)var<storage,read>worklist:array<u32>;
@group(0)@binding(3)var<storage,read>flags:array<u32>;@group(0)@binding(4)var<storage,read>phi:array<f32>;@group(0)@binding(5)var<storage,read>headers:array<H>;
@group(0)@binding(7)var<storage,read>rowCountSource:array<u32>;
@group(0)@binding(8)var<storage,read_write>aggregates:array<Aggregate>;@group(0)@binding(9)var<storage,read_write>rowOffsets:array<u32>;
@group(0)@binding(12)var<storage,read_write>out:array<Contribution>;@group(0)@binding(13)var<storage,read_write>control:C;@group(0)@binding(14)var<storage,read>topologyControl:array<u32>;
@group(0)@binding(15)var<storage,read_write>rowDispatch:array<u32>;
const INVALID:u32=0xffffffffu;const VALID:u32=1u;const CAPACITY:u32=1u;const UNOWNED:u32=2u;const NONFINITE:u32=4u;const UNPUBLISHED_SOURCE:u32=8u;
const DOWNSTREAM_ROLLBACK:u32=16u;
fn finite(v:f32)->bool{return (bitcast<u32>(v)&0x7f800000u)!=0x7f800000u;}fn packBrick(q:vec3u)->u32{return q.x+p.brickDims.x*(q.y+p.brickDims.y*q.z);}
fn unpackBrick(key:u32)->vec3u{let xy=p.brickDims.x*p.brickDims.y;let z=key/xy;let r=key-z*xy;let y=r/p.brickDims.x;return vec3u(r-y*p.brickDims.x,y,z);}
fn localCoord(local:u32)->vec3u{let r=p.brickResolution;let z=local/(r*r);let q=local-z*r*r;let y=q/r;return vec3u(q-y*r,y,z);}
fn flatIndex(w:vec3u,lid:u32,n:vec3u)->u32{return fineLinearWorkgroup(w,n)*64u+lid;}
// The fine narrow band is authoritative wherever it is resident. Samples may
// advance beyond the current liquid pressure-row set by less than one CFL
// step; the CPU restriction oracles count and skip those samples as well.
// They do not need a separate coarse-authority entry because consumers sample this
// validated fine publication first. Keep non-positive misses observable, but
// do not invalidate an otherwise complete fine/coarse transaction.
fn finePage(key:u32)->u32{
 if(arrayLength(&worklist)<7u||worklist[0]!=p.generation||worklist[2]!=p.pageCapacity||(worklist[3]&3u)!=3u||worklist[5]!=1u||worklist[6]!=1u){return INVALID;}
 let count=min(worklist[1],p.pageCapacity);let logicalCount=p.brickDims.x*p.brickDims.y*p.brickDims.z;
 let directoryBase=7u+p.pageCapacity;
 if(7u+count>arrayLength(&worklist)||key>=logicalCount||directoryBase+key>=arrayLength(&worklist)){return INVALID;}
 let id=worklist[directoryBase+key];let base=id*10u;
 return select(INVALID,id,id<p.pageCapacity&&base+2u<arrayLength(&metadata)
  &&metadata[base]==id&&metadata[base+1u]==key&&metadata[base+2u]==p.generation);
}
struct FineValue{value:f32,valid:u32,error:u32}
fn fineValueAt(q0:vec3i)->FineValue{
 let q=vec3u(clamp(q0,vec3i(0),vec3i(p.sampleDims)-vec3i(1)));
 let brick=q/p.brickResolution;let id=finePage(packBrick(brick));
 if(id==INVALID){return FineValue(0.,0u,0u);}
 let local=q-brick*p.brickResolution;
 let index=id*p.samplesPerBrick+local.x+p.brickResolution*(local.y+p.brickResolution*local.z);
 if(index>=arrayLength(&flags)||index>=arrayLength(&phi)){return FineValue(0.,0u,CAPACITY);}
 if((flags[index]&VALID)==0u){return FineValue(0.,0u,0u);}
 let value=phi[index];if(!finite(value)){return FineValue(0.,0u,NONFINITE);}
 return FineValue(value,1u,0u);
}
// At factor 1 the single sample owned by a finest row is its cell centre, not
// eight coincident cell corners. Reconstruct the 3x3x3 knot set of the
// piecewise-trilinear field over that physical cell. Its 27 values comprise
// the centre, face/edge midpoints, and corners, and therefore bound every
// trilinear subcell without inventing a finer persistent level-set band.
fn factorOneCellProbe(origin:vec3u,ordinal:u32)->FineValue{
 let mode=vec3u(ordinal%3u,(ordinal/3u)%3u,ordinal/9u);
 let low=vec3i(origin)+vec3i(select(vec3u(0u),vec3u(1u),mode==vec3u(0u)))*vec3i(-1);
 let t=vec3f(select(vec3u(0u),vec3u(1u),mode!=vec3u(1u)))*.5;
 var sum=0.;var weight=0.;var error=0u;
 for(var corner=0u;corner<8u;corner+=1u){
  let offset=vec3i(i32(corner&1u),i32((corner>>1u)&1u),i32((corner>>2u)&1u));
  let choose=vec3<bool>((corner&1u)!=0u,(corner&2u)!=0u,(corner&4u)!=0u);
  let w=select(1.-t.x,t.x,choose.x)*select(1.-t.y,t.y,choose.y)*select(1.-t.z,t.z,choose.z);
  if(w==0.){continue;}let sample=fineValueAt(low+offset);error|=sample.error;
  if(sample.valid==0u){return FineValue(0.,0u,error);}
  sum+=w*sample.value;weight+=w;
 }
 if(weight>.999){return FineValue(sum,1u,error);}
 return FineValue(0.,0u,error);
}
var<workgroup> rowMinimum:array<f32,64>;var<workgroup> rowMaximum:array<f32,64>;
var<workgroup> rowCounts:array<u32,64>;var<workgroup> rowErrors:array<u32,64>;
var<workgroup> rowMasks:array<u32,64>;var<workgroup> rowCombinedMasks:array<u32,64>;var<workgroup> rowCorners:array<f32,512>;
var<workgroup> diagnosticCounts:array<u32,64>;var<workgroup> diagnosticFirst:array<u32,64>;
var<workgroup> diagnosticMagnitude:array<f32,64>;var<workgroup> diagnosticErrors:array<u32,64>;
@compute @workgroup_size(64)fn prepareRestriction(@builtin(global_invocation_id)g:vec3u){let i=g.x;if(i==0u){control.count=0u;control.maximumPerRow=1u;control.flags=0u;control.unacceptedRows=0u;let publishedRows=select(0u,rowCountSource[p.rowCountOffsetWords],p.rowCountOffsetWords<arrayLength(&rowCountSource));control.rowCount=min(publishedRows,p.rowCapacity);if(control.rowCount<arrayLength(&rowOffsets)){rowOffsets[control.rowCount]=control.rowCount;}else{control.flags|=CAPACITY;}control.valid=0u;control.firstUnownedLiquid=INVALID;control.maximumUnownedLiquidMagnitude=0u;var topologyReady=false;if(arrayLength(&topologyControl)>=8u){let committed=topologyControl[4]==1u;let provisional=p.reserved!=0u&&topologyControl[3]>0u&&topologyControl[4]==0u;
 // A downstream-rejected target is replaced in-place by the prior complete
 // fine publication and retagged for this generation. Restricting that exact
 // rollback field is required to republish the paper's separate coarse
 // octree level set; it does not admit any sample from the rejected target.
 // Downstream rollback is a flag, not an exclusive state. The topology may
 // retain the originating rejection bit alongside it; published/rolledBack
 // and the non-zero reason below are the exact retained-field receipt.
 let rollback=(topologyControl[0]&DOWNSTREAM_ROLLBACK)!=0u&&topologyControl[4]==1u
  &&topologyControl[5]==1u&&topologyControl[7]!=0u;
 topologyReady=(topologyControl[0]==0u&&(committed||provisional)&&topologyControl[5]==0u&&topologyControl[7]==0u)||rollback;}if(arrayLength(&worklist)<7u||!topologyReady){control.flags|=UNPUBLISHED_SOURCE;}else if(worklist[0]!=p.generation||worklist[2]!=p.pageCapacity||(worklist[3]&3u)!=3u||worklist[5]!=1u||worklist[6]!=1u){control.flags|=UNPUBLISHED_SOURCE;}let count=select(control.rowCount,0u,control.flags!=0u);let x=min(count,65535u);rowDispatch[0]=x;rowDispatch[1]=select(1u,(count+x-1u)/x,x>0u);rowDispatch[2]=1u;}}
@compute @workgroup_size(64)fn restrictCoarseRows(@builtin(workgroup_id)w:vec3u,@builtin(local_invocation_index)lid:u32,@builtin(num_workgroups)n:vec3u){let r=fineLinearWorkgroup(w,n);if(lid==0u&&r<p.rowCapacity){if(r<arrayLength(&aggregates)){aggregates[r]=Aggregate(0.,0.,0.,0u,0u,0u,array<u32,6>());}if(r<arrayLength(&out)){out[r]=Contribution(0.,0.,0.,0u);}if(r<arrayLength(&rowOffsets)){rowOffsets[r]=r;}}var minimum=3.402823e38;var maximum=-3.402823e38;var count=0u;var error=0u;var mask=0u;for(var corner=0u;corner<8u;corner+=1u){rowCorners[lid*8u+corner]=0.;}if(r<control.rowCount&&r<arrayLength(&headers)){let h=headers[r];let ratioF=p.cellWidth/p.fineWidth;let ratio=u32(round(ratioF));if(h.size==0u||ratio==0u||abs(f32(ratio)-ratioF)>1e-5){error=CAPACITY;}else{let o=vec3u(h.cell%p.dimensions.x,(h.cell/p.dimensions.x)%p.dimensions.y,h.cell/(p.dimensions.x*p.dimensions.y));let first=o*ratio;let last=min((o+vec3u(h.size))*ratio,p.sampleDims);let firstBrick=first/p.brickResolution;let lastBrick=(last+vec3u(p.brickResolution-1u))/p.brickResolution;let extent=lastBrick-firstBrick;let brickCount=extent.x*extent.y*extent.z;let center=(vec3f(o)+.5*f32(h.size))*p.cellWidth;for(var ordinal=0u;ordinal<brickCount;ordinal+=1u){let bx=ordinal%extent.x;let yz=ordinal/extent.x;let brick=firstBrick+vec3u(bx,yz%extent.y,yz/extent.y);let id=finePage(packBrick(brick));if(id==INVALID){continue;}for(var local=lid;local<p.samplesPerBrick;local+=64u){let q=brick*p.brickResolution+localCoord(local);if(any(q<first)||any(q>=last)){continue;}let index=id*p.samplesPerBrick+local;if((flags[index]&VALID)==0u){continue;}let value=phi[index];if(!finite(value)){error|=NONFINITE;continue;}count+=1u;minimum=min(minimum,value);maximum=max(maximum,value);if(ratio==1u&&h.size==1u){mask=255u;for(var corner=0u;corner<8u;corner+=1u){rowCorners[lid*8u+corner]=value;}}else{let x=p.origin+(vec3f(q)+.5)*p.fineWidth;let d=x-center;let centerDelta=abs(abs(d)-vec3f(.5*p.fineWidth));let tolerance=2e-5*max(p.cellWidth,p.fineWidth);if(all(centerDelta<=vec3f(tolerance))){let corner=select(0u,1u,d.x>0.)+select(0u,2u,d.y>0.)+select(0u,4u,d.z>0.);mask|=1u<<corner;rowCorners[lid*8u+corner]=value;}}}}}}rowMinimum[lid]=minimum;rowMaximum[lid]=maximum;rowCounts[lid]=count;rowErrors[lid]=error;rowMasks[lid]=mask;rowCombinedMasks[lid]=mask;workgroupBarrier();var width=32u;loop{if(width==0u){break;}if(lid<width){rowMinimum[lid]=min(rowMinimum[lid],rowMinimum[lid+width]);rowMaximum[lid]=max(rowMaximum[lid],rowMaximum[lid+width]);rowCounts[lid]+=rowCounts[lid+width];rowErrors[lid]|=rowErrors[lid+width];rowCombinedMasks[lid]|=rowCombinedMasks[lid+width];}workgroupBarrier();width>>=1u;}if(lid==0u&&r<control.rowCount&&r<arrayLength(&aggregates)){var centerPhi=0.;if(rowCombinedMasks[0]==255u){for(var corner=0u;corner<8u;corner+=1u){var cornerPhi=0.;for(var lane=0u;lane<64u;lane+=1u){if((rowMasks[lane]&(1u<<corner))!=0u){cornerPhi=rowCorners[lane*8u+corner];}}centerPhi+=.125*cornerPhi;}}let accepted=rowErrors[0]==0u&&rowCounts[0]>0u&&rowCombinedMasks[0]==255u;let aggregate=Aggregate(centerPhi,rowMinimum[0],rowMaximum[0],select(0u,1u,accepted),rowCounts[0],rowErrors[0],array<u32,6>());aggregates[r]=aggregate;if(accepted&&r<arrayLength(&out)){out[r]=Contribution(aggregate.centerPhi,aggregate.minimumPhi,aggregate.maximumPhi,1u);}}}
// Factor 1 has one cell-centred sample per finest pressure row. Replace the
// legacy coincident-corner aggregate with the exact knot envelope of the local
// piecewise-trilinear reconstruction. The centre drives the downstream
// occupancy fraction; the envelope retains sign-crossing/interface evidence.
// An incomplete 3x3x3 knot set publishes no correction, preserving the prior
// coarse value instead of manufacturing interface evidence from a partial
// sparse stencil.
@compute @workgroup_size(64)fn reconstructFactorOneRows(
 @builtin(workgroup_id)w:vec3u,@builtin(local_invocation_index)lid:u32,
 @builtin(num_workgroups)n:vec3u){
 let row=fineLinearWorkgroup(w,n);
 let inBounds=row<control.rowCount&&row<arrayLength(&headers)
  &&row<arrayLength(&aggregates)&&row<arrayLength(&out);
 var h=H(0u,0u,0u,0u,0.,0.,0u,0u,vec4f(0.));if(inBounds){h=headers[row];}
 let ratioF=p.cellWidth/p.fineWidth;let ratio=u32(round(ratioF));
 let enabled=inBounds&&h.size==1u&&ratio==1u&&abs(f32(ratio)-ratioF)<=1e-5;
 if(lid==0u&&enabled){aggregates[row]=Aggregate(0.,0.,0.,0u,0u,0u,array<u32,6>());
  out[row]=Contribution(0.,0.,0.,0u);}
 var minimum=3.402823e38;var maximum=-3.402823e38;
 var count=0u;var error=0u;var mask=0u;
 for(var corner=0u;corner<8u;corner+=1u){rowCorners[lid*8u+corner]=0.;}
 let origin=vec3u(h.cell%p.dimensions.x,(h.cell/p.dimensions.x)%p.dimensions.y,
  h.cell/(p.dimensions.x*p.dimensions.y));
 if(enabled&&lid<27u){
  let probe=factorOneCellProbe(origin,lid);error|=probe.error;
  if(probe.valid!=0u){minimum=probe.value;maximum=probe.value;count=1u;
   if(lid==13u){mask=255u;for(var corner=0u;corner<8u;corner+=1u){
    rowCorners[lid*8u+corner]=probe.value;}}}
 }
 rowMinimum[lid]=minimum;rowMaximum[lid]=maximum;rowCounts[lid]=count;
 rowErrors[lid]=error;rowMasks[lid]=mask;rowCombinedMasks[lid]=mask;
 workgroupBarrier();
 var width=32u;loop{if(width==0u){break;}if(lid<width){
  rowMinimum[lid]=min(rowMinimum[lid],rowMinimum[lid+width]);
  rowMaximum[lid]=max(rowMaximum[lid],rowMaximum[lid+width]);
  rowCounts[lid]+=rowCounts[lid+width];rowErrors[lid]|=rowErrors[lid+width];
  rowCombinedMasks[lid]|=rowCombinedMasks[lid+width];}
  workgroupBarrier();width>>=1u;}
 if(lid==0u&&enabled){
  let accepted=rowErrors[0]==0u&&rowCounts[0]==27u&&rowCombinedMasks[0]==255u;
  let centerPhi=rowCorners[13u*8u];
  let aggregate=Aggregate(centerPhi,rowMinimum[0],rowMaximum[0],
   select(0u,1u,accepted),rowCounts[0],rowErrors[0],array<u32,6>());
  aggregates[row]=aggregate;
  if(accepted){out[row]=Contribution(centerPhi,rowMinimum[0],rowMaximum[0],1u);}
 }
}
// A row is accepted only when all eight corner samples are resident and valid.
// An unaccepted row raises no error, writes no contribution, and leaves the
// coarse level set uncorrected, so narrowing the fine band silently deletes the
// fine correction wherever the band recedes. Publish the unaccepted-row count
// beside the accepted count: coverage loss then shows up as a number rather
// than as an unexplained free surface. It is a diagnostic only and never sets a
// flag, because an uncorrected dry row is legitimate. The count is meaningful
// only when the prepared source was accepted; a rejected source dispatches zero
// row workgroups, so the aggregates it would read belong to a prior command.
@compute @workgroup_size(64)fn publishRestriction(@builtin(local_invocation_index)lid:u32){let sourceRejected=control.flags!=0u;var errors=0u;var unaccepted=0u;for(var r=lid;r<control.rowCount;r+=64u){errors|=aggregates[r].error;unaccepted+=select(1u,0u,aggregates[r].valid!=0u);}diagnosticErrors[lid]=errors;diagnosticCounts[lid]=unaccepted;workgroupBarrier();var width=32u;loop{if(width==0u){break;}if(lid<width){diagnosticErrors[lid]|=diagnosticErrors[lid+width];diagnosticCounts[lid]+=diagnosticCounts[lid+width];}workgroupBarrier();width>>=1u;}if(lid==0u){control.flags|=diagnosticErrors[0];control.unacceptedRows=select(diagnosticCounts[0],0u,sourceRejected);if(control.flags==0u){control.count=control.rowCount;control.maximumPerRow=1u;control.valid=0x80000000u;}else{control.count=0xffffffffu;control.maximumPerRow=1u;}}}
`;

/** Conditional activity variant. Each checkpoint is placed after the entry
 * point has enough state to identify useful work; disabled mode returns the
 * production source byte-for-byte. */
export function fineToCoarseLevelSetActivityShader(
  activity: GPULogicalActivityAdoptionContext,
): string {
  const prepare = activity.workgroup("prepare-restriction", "progress", {
    workgroupId: "vec3u(0u)",
    localInvocationIndex: "g.x",
    workgroupLaneCount: 64,
  });
  const restrict = activity.workgroup("restrict-coarse-rows", "active-row", {
    workgroupId: "w",
    numWorkgroups: "n",
    localInvocationIndex: "lid",
    workgroupLaneCount: 64,
    recordWhen: "r < control.rowCount",
  });
  const publish = activity.workgroup("publish-restriction", "progress", {
    workgroupId: "vec3u(0u)",
    localInvocationIndex: "lid",
    workgroupLaneCount: 64,
  });
  if (!prepare && !restrict && !publish) return fineToCoarseLevelSetWGSL;
  const replacements = [
    ["@compute @workgroup_size(64)fn prepareRestriction(@builtin(global_invocation_id)g:vec3u){",
      `@compute @workgroup_size(64)fn prepareRestriction(@builtin(global_invocation_id)g:vec3u){${prepare}`],
    ["@compute @workgroup_size(64)fn restrictCoarseRows(@builtin(workgroup_id)w:vec3u,@builtin(local_invocation_index)lid:u32,@builtin(num_workgroups)n:vec3u){let r=fineLinearWorkgroup(w,n);",
      `@compute @workgroup_size(64)fn restrictCoarseRows(@builtin(workgroup_id)w:vec3u,@builtin(local_invocation_index)lid:u32,@builtin(num_workgroups)n:vec3u){let r=fineLinearWorkgroup(w,n);${restrict}`],
    ["@compute @workgroup_size(64)fn publishRestriction(@builtin(local_invocation_index)lid:u32){",
      `@compute @workgroup_size(64)fn publishRestriction(@builtin(local_invocation_index)lid:u32){${publish}`],
  ] as const;
  return replacements.reduce((source, [needle, replacement]) => {
    if (!source.includes(needle)) throw new Error(`Fine-restriction activity entry point is missing: ${needle}`);
    return source.replace(needle, replacement);
  }, fineToCoarseLevelSetWGSL);
}
