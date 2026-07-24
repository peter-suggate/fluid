/** GPU-only row-owned restriction with O(rows) storage over resident fine samples. */

import { fineLevelSetLinearWorkgroupWGSL, planFineLevelSetDispatch2D } from "./webgpu-fine-levelset-dispatch";
import type { WebGPUFineLevelSetBrickSource } from "./webgpu-octree-fine-levelset-bricks";
import { PassBroker } from "./webgpu-pass-broker";

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
  readonly flags: number; readonly unownedSamples: number; readonly rowCount: number; readonly valid: boolean;
  readonly firstUnownedLiquidLogical?: number; readonly maximumUnownedLiquidMagnitude?: number;
}

export function planFineToCoarseLevelSet(rowCapacity: number, sampleCapacity: number): FineToCoarseGPUPlan {
  if (!Number.isSafeInteger(rowCapacity) || rowCapacity < 1
    || !Number.isSafeInteger(sampleCapacity) || sampleCapacity < 1) {
    throw new RangeError("Fine-to-coarse capacities must be positive integers");
  }
  const blockCount = Math.ceil(rowCapacity / 256), aggregateScratchBytes = rowCapacity * 48;
  return { rowCapacity, sampleCapacity, blockCount, aggregateScratchBytes,
    allocatedBytes: 112 + aggregateScratchBytes + (rowCapacity + 1) * 4 + rowCapacity * 16 + 32 };
}

export function unpackFineToCoarseGPUControl(words: ArrayLike<number>): FineToCoarseGPUControl {
  if (words.length < 6) throw new RangeError("Fine-to-coarse control requires six words");
  const phi = words.length >= 8
    ? new Float32Array(new Uint32Array([Number(words[7]) >>> 0]).buffer)[0] : undefined;
  return { contributionCount: Number(words[0]) >>> 0, maximumContributionsPerRow: Number(words[1]) >>> 0,
    flags: Number(words[2]) >>> 0, unownedSamples: Number(words[3]) >>> 0,
    rowCount: Number(words[4]) >>> 0, valid: Number(words[5]) !== 0,
    ...(words.length >= 8 ? { firstUnownedLiquidLogical: Number(words[6]) >>> 0,
      maximumUnownedLiquidMagnitude: phi } : {}) };
}

export class WebGPUFineToCoarseLevelSet {
  readonly plan: FineToCoarseGPUPlan;
  readonly result: FineToCoarseGPUResult;
  private readonly params: GPUBuffer;
  private readonly aggregates: GPUBuffer;
  readonly control: GPUBuffer;
  private readonly pipelines: Record<string, GPUComputePipeline>;
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
    this.result = { rowOffsets, contributions, counts: this.control, aggregated: true };
    const shaderModule = device.createShaderModule({ label: "Fine-to-coarse row restriction",
      code: fineToCoarseLevelSetWGSL });
    const pipeline = (entryPoint: string) => device.createComputePipeline({ label: entryPoint, layout: "auto",
      compute: { module: shaderModule, entryPoint } });
    this.pipelines = {
      prepare: pipeline("prepareRestriction"),
      restrict: pipeline("restrictCoarseRows"),
      diagnose: pipeline("diagnoseUnownedFineSamples"),
      finalize: pipeline("finalizeRestrictionRows"),
      publish: pipeline("publishRestriction"),
    };
  }

  /** Record exact restriction into the caller's publication pass. */
  encode(broker: PassBroker, fine: WebGPUFineLevelSetBrickSource, input: {
    headers: GPUBuffer; rowDirectory: GPUBuffer; rowCount: GPUBuffer;
    /** Control for the topology transaction that produced `fine`. */
    topologyControl: GPUBuffer;
    dimensions: readonly [number, number, number]; physicalCellSize: number;
    maximumLeafSize: number; rowDirectoryCapacity: number;
    /** Expensive capacity-wide QA scan. It is observational and never
     * participates in the fine/coarse publication validity predicate. */
    diagnoseUnownedFineSamples?: boolean;
  }): FineToCoarseGPUResult {
    if (this.destroyed) throw new Error("Fine-to-coarse restriction is destroyed");
    const sampleCount = fine.plan.maximumResidentBricks * fine.plan.samplesPerBrick;
    if (sampleCount > this.plan.sampleCapacity) throw new RangeError("Fine-to-coarse sample source exceeds capacity");
    const data = new ArrayBuffer(112), u = new Uint32Array(data), f = new Float32Array(data);
    u.set(fine.plan.brickDimensions, 0); u[3] = fine.plan.brickResolution;
    u.set(fine.plan.sampleDimensions, 4); u[7] = fine.plan.samplesPerBrick;
    f.set(fine.plan.domainOrigin, 8); f[11] = fine.plan.fineCellWidth;
    u.set([fine.plan.maximumResidentBricks, fine.generation, this.plan.rowCapacity, sampleCount,
      input.rowDirectoryCapacity, 0], 12);
    u.set(input.dimensions, 20); u[23] = input.maximumLeafSize; f[24] = input.physicalCellSize;
    this.device.queue.writeBuffer(this.params, 0, data);
    const buffers = new Map<number, GPUBuffer>([[0, this.params], [1, fine.metadata], [2, fine.worklist],
      [3, fine.flags], [4, fine.phi], [5, input.headers], [6, input.rowDirectory], [7, input.rowCount],
      [8, this.aggregates], [9, this.result.rowOffsets], [12, this.result.contributions], [13, this.control],
      [14, input.topologyControl]]);
    const used: Record<string, number[]> = {
      prepare: [0, 2, 7, 8, 9, 12, 13, 14],
      restrict: [0, 1, 2, 3, 4, 5, 8, 13],
      diagnose: [0, 1, 2, 3, 4, 6, 13],
      finalize: [8, 12, 13],
      publish: [8, 13],
    };
    const run = (name: string, x: number, y = 1) => { const pipeline = this.pipelines[name];
      const entries = used[name].map((binding) => ({ binding, resource: { buffer: buffers.get(binding)! } }));
      const pass = broker.compute({ label: `Fine-to-coarse restriction · ${name}` }); pass.setPipeline(pipeline);
      pass.setBindGroup(0, this.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries }));
      pass.dispatchWorkgroups(x, y); };
    run("prepare", Math.ceil((this.plan.rowCapacity + 1) / 64));
    const rows = planFineLevelSetDispatch2D(this.plan.rowCapacity,
      this.device.limits.maxComputeWorkgroupsPerDimension);
    if (rows.workgroups > 0) run("restrict", rows.x, rows.y);
    if (input.diagnoseUnownedFineSamples) run("diagnose", 1);
    run("finalize", Math.ceil(this.plan.rowCapacity / 64)); run("publish", 1);
    return this.result;
  }

  destroy(): void { if (this.destroyed) return; this.destroyed = true;
    this.params.destroy(); this.aggregates.destroy(); this.result.rowOffsets.destroy();
    this.result.contributions.destroy(); this.control.destroy(); }
}

export const fineToCoarseLevelSetWGSL = /* wgsl */ `
${fineLevelSetLinearWorkgroupWGSL}
struct P{brickDims:vec3u,brickResolution:u32,sampleDims:vec3u,samplesPerBrick:u32,origin:vec3f,fineWidth:f32,
 pageCapacity:u32,generation:u32,rowCapacity:u32,sampleCapacity:u32,directoryCapacity:u32,reserved:u32,dimensions:vec3u,maxLeaf:u32,cellWidth:f32}
struct H{cell:u32,a:u32,b:u32,size:u32,x:f32,y:f32,z:u32,w:u32,g:vec4f}struct DirectoryEntry{cellPlusOne:u32,size:u32,row:u32,morton:u32}
struct C{count:u32,maximumPerRow:u32,flags:u32,unowned:u32,rowCount:u32,valid:u32,firstUnownedLiquid:u32,maximumUnownedLiquidMagnitude:u32}
struct Aggregate{centerPhi:f32,minimumPhi:f32,maximumPhi:f32,valid:u32,sampleCount:u32,error:u32,pad:array<u32,6>}
struct Contribution{centerPhi:f32,minimumPhi:f32,maximumPhi:f32,valid:u32}struct Sample{positionPhi:vec4f,logical:u32,valid:u32}
@group(0)@binding(0)var<uniform>p:P;@group(0)@binding(1)var<storage,read>metadata:array<u32>;@group(0)@binding(2)var<storage,read>worklist:array<u32>;
@group(0)@binding(3)var<storage,read>flags:array<u32>;@group(0)@binding(4)var<storage,read>phi:array<f32>;@group(0)@binding(5)var<storage,read>headers:array<H>;
@group(0)@binding(6)var<storage,read>rowDirectory:array<DirectoryEntry>;@group(0)@binding(7)var<storage,read>rowCountSource:array<u32>;
@group(0)@binding(8)var<storage,read_write>aggregates:array<Aggregate>;@group(0)@binding(9)var<storage,read_write>rowOffsets:array<u32>;
@group(0)@binding(12)var<storage,read_write>out:array<Contribution>;@group(0)@binding(13)var<storage,read_write>control:C;@group(0)@binding(14)var<storage,read>topologyControl:array<u32>;
const INVALID:u32=0xffffffffu;const VALID:u32=1u;const CAPACITY:u32=1u;const UNOWNED:u32=2u;const NONFINITE:u32=4u;const UNPUBLISHED_SOURCE:u32=8u;
fn finite(v:f32)->bool{return (bitcast<u32>(v)&0x7f800000u)!=0x7f800000u;}fn packBrick(q:vec3u)->u32{return q.x+p.brickDims.x*(q.y+p.brickDims.y*q.z);}
fn unpackBrick(key:u32)->vec3u{let xy=p.brickDims.x*p.brickDims.y;let z=key/xy;let r=key-z*xy;let y=r/p.brickDims.x;return vec3u(r-y*p.brickDims.x,y,z);}
fn localCoord(local:u32)->vec3u{let r=p.brickResolution;let z=local/(r*r);let q=local-z*r*r;let y=q/r;return vec3u(q-y*r,y,z);}
fn mortonPart10(value:u32)->u32{var x=value&1023u;x=(x|(x<<16u))&0x030000ffu;x=(x|(x<<8u))&0x0300f00fu;x=(x|(x<<4u))&0x030c30c3u;x=(x|(x<<2u))&0x09249249u;return x;}
fn morton(c:u32)->u32{let q=vec3u(c%p.dimensions.x,(c/p.dimensions.x)%p.dimensions.y,c/(p.dimensions.x*p.dimensions.y));return mortonPart10(q.x)|(mortonPart10(q.y)<<1u)|(mortonPart10(q.z)<<2u);}
fn level(size:u32)->u32{return 31u-countLeadingZeros(size);}
fn keyLess(aLevel:u32,aMorton:u32,bLevel:u32,bMorton:u32)->bool{return aLevel<bLevel||(aLevel==bLevel&&aMorton<bMorton);}
fn find(c:u32,s:u32)->u32{let count=min(control.rowCount,min(p.directoryCapacity,arrayLength(&rowDirectory)));let wantedLevel=level(s);let wantedMorton=morton(c);var low=0u;var high=count;while(low<high){let middle=low+(high-low)/2u;let candidate=rowDirectory[middle];if(keyLess(level(candidate.size),candidate.morton,wantedLevel,wantedMorton)){low=middle+1u;}else{high=middle;}}if(low<count){let candidate=rowDirectory[low];if(candidate.cellPlusOne==c+1u&&candidate.size==s){return candidate.row;}}return INVALID;}
fn owner(x:vec3f)->u32{let g=x/p.cellWidth;if(any(g<vec3f(0))||any(g>=vec3f(p.dimensions))){return INVALID;}let q=vec3u(floor(g));var s=1u;loop{let o=(q/s)*s;let c=o.x+p.dimensions.x*(o.y+p.dimensions.y*o.z);let r=find(c,s);if(r!=INVALID){return r;}if(s>=p.maxLeaf){break;}s*=2u;}return INVALID;}
fn sample(flat:u32)->Sample{let residentCount=min(worklist[0],p.pageCapacity);if(flat>=residentCount*p.samplesPerBrick){return Sample(vec4f(0),0u,0u);}let w=flat/p.samplesPerBrick;let local=flat-w*p.samplesPerBrick;let id=worklist[5u+w];if(id>=p.pageCapacity||metadata[id*10u+2u]!=p.generation){return Sample(vec4f(0),0u,0u);}let index=id*p.samplesPerBrick+local;if((flags[index]&VALID)==0u){return Sample(vec4f(0),0u,0u);}let key=metadata[id*10u+1u];let brick=unpackBrick(key);let q=brick*p.brickResolution+localCoord(local);if(any(q>=p.sampleDims)){return Sample(vec4f(0),0u,0u);}let x=p.origin+(vec3f(q)+.5)*p.fineWidth;return Sample(vec4f(x,phi[index]),key*p.samplesPerBrick+local,1u);}
fn flatIndex(w:vec3u,lid:u32,n:vec3u)->u32{return fineLinearWorkgroup(w,n)*64u+lid;}
// The fine narrow band is authoritative wherever it is resident. Samples may
// advance beyond the current liquid pressure-row set by less than one CFL
// step; the CPU restriction oracles count and skip those samples as well.
// They do not need a coarse fallback entry because consumers sample this
// validated fine publication first. Keep non-positive misses observable, but
// do not invalidate an otherwise complete fine/coarse transaction.
fn finePage(key:u32)->u32{let count=min(worklist[0],p.pageCapacity);var low=0u;var high=count;while(low<high){let middle=low+(high-low)/2u;let id=worklist[5u+middle];if(id>=p.pageCapacity){return INVALID;}if(metadata[id*10u+1u]<key){low=middle+1u;}else{high=middle;}}if(low<count){let id=worklist[5u+low];if(id<p.pageCapacity&&metadata[id*10u+1u]==key&&metadata[id*10u+2u]==p.generation){return id;}}return INVALID;}
var<workgroup> rowMinimum:array<f32,64>;var<workgroup> rowMaximum:array<f32,64>;
var<workgroup> rowCounts:array<u32,64>;var<workgroup> rowErrors:array<u32,64>;
var<workgroup> rowMasks:array<u32,64>;var<workgroup> rowCombinedMasks:array<u32,64>;var<workgroup> rowCorners:array<f32,512>;
var<workgroup> diagnosticCounts:array<u32,64>;var<workgroup> diagnosticFirst:array<u32,64>;
var<workgroup> diagnosticMagnitude:array<f32,64>;var<workgroup> diagnosticErrors:array<u32,64>;
@compute @workgroup_size(64)fn prepareRestriction(@builtin(global_invocation_id)g:vec3u){let i=g.x;if(i<arrayLength(&aggregates)){aggregates[i]=Aggregate(0.,0.,0.,0u,0u,0u,array<u32,6>());out[i]=Contribution(0.,0.,0.,0u);}if(i<arrayLength(&rowOffsets)){rowOffsets[i]=i;}if(i==0u){control.count=0u;control.maximumPerRow=1u;control.flags=0u;control.unowned=0u;control.rowCount=min(rowCountSource[0],p.rowCapacity);control.valid=0u;control.firstUnownedLiquid=INVALID;control.maximumUnownedLiquidMagnitude=0u;if(arrayLength(&worklist)<5u||arrayLength(&topologyControl)<8u){control.flags=UNPUBLISHED_SOURCE;}else if(worklist[1]!=p.generation||worklist[3]!=1u||worklist[4]!=1u||topologyControl[0]!=0u||topologyControl[4]!=1u||topologyControl[5]!=0u||topologyControl[7]!=0u){control.flags=UNPUBLISHED_SOURCE;}}}
@compute @workgroup_size(64)fn restrictCoarseRows(@builtin(workgroup_id)w:vec3u,@builtin(local_invocation_index)lid:u32,@builtin(num_workgroups)n:vec3u){let r=fineLinearWorkgroup(w,n);var minimum=3.402823e38;var maximum=-3.402823e38;var count=0u;var error=0u;var mask=0u;for(var corner=0u;corner<8u;corner+=1u){rowCorners[lid*8u+corner]=0.;}if(r<control.rowCount&&r<arrayLength(&headers)){let h=headers[r];let ratioF=p.cellWidth/p.fineWidth;let ratio=u32(round(ratioF));if(h.size==0u||ratio==0u||abs(f32(ratio)-ratioF)>1e-5){error=CAPACITY;}else{let o=vec3u(h.cell%p.dimensions.x,(h.cell/p.dimensions.x)%p.dimensions.y,h.cell/(p.dimensions.x*p.dimensions.y));let first=o*ratio;let last=min((o+vec3u(h.size))*ratio,p.sampleDims);let firstBrick=first/p.brickResolution;let lastBrick=(last+vec3u(p.brickResolution-1u))/p.brickResolution;let extent=lastBrick-firstBrick;let brickCount=extent.x*extent.y*extent.z;let center=(vec3f(o)+.5*f32(h.size))*p.cellWidth;for(var ordinal=0u;ordinal<brickCount;ordinal+=1u){let bx=ordinal%extent.x;let yz=ordinal/extent.x;let brick=firstBrick+vec3u(bx,yz%extent.y,yz/extent.y);let id=finePage(packBrick(brick));if(id==INVALID){continue;}for(var local=lid;local<p.samplesPerBrick;local+=64u){let q=brick*p.brickResolution+localCoord(local);if(any(q<first)||any(q>=last)){continue;}let index=id*p.samplesPerBrick+local;if((flags[index]&VALID)==0u){continue;}let value=phi[index];if(!finite(value)){error|=NONFINITE;continue;}count+=1u;minimum=min(minimum,value);maximum=max(maximum,value);let x=p.origin+(vec3f(q)+.5)*p.fineWidth;let d=x-center;let centerDelta=abs(abs(d)-vec3f(.5*p.fineWidth));let tolerance=2e-5*max(p.cellWidth,p.fineWidth);if(all(centerDelta<=vec3f(tolerance))){let corner=select(0u,1u,d.x>0.)+select(0u,2u,d.y>0.)+select(0u,4u,d.z>0.);mask|=1u<<corner;rowCorners[lid*8u+corner]=value;}}}}}rowMinimum[lid]=minimum;rowMaximum[lid]=maximum;rowCounts[lid]=count;rowErrors[lid]=error;rowMasks[lid]=mask;rowCombinedMasks[lid]=mask;workgroupBarrier();var width=32u;loop{if(width==0u){break;}if(lid<width){rowMinimum[lid]=min(rowMinimum[lid],rowMinimum[lid+width]);rowMaximum[lid]=max(rowMaximum[lid],rowMaximum[lid+width]);rowCounts[lid]+=rowCounts[lid+width];rowErrors[lid]|=rowErrors[lid+width];rowCombinedMasks[lid]|=rowCombinedMasks[lid+width];}workgroupBarrier();width>>=1u;}if(lid==0u&&r<control.rowCount&&r<arrayLength(&aggregates)){var centerPhi=0.;if(rowCombinedMasks[0]==255u){for(var corner=0u;corner<8u;corner+=1u){var cornerPhi=0.;for(var lane=0u;lane<64u;lane+=1u){if((rowMasks[lane]&(1u<<corner))!=0u){cornerPhi=rowCorners[lane*8u+corner];}}centerPhi+=.125*cornerPhi;}}let accepted=rowErrors[0]==0u&&rowCounts[0]>0u&&rowCombinedMasks[0]==255u;aggregates[r]=Aggregate(centerPhi,rowMinimum[0],rowMaximum[0],select(0u,1u,accepted),rowCounts[0],rowErrors[0],array<u32,6>());}}
@compute @workgroup_size(64)fn diagnoseUnownedFineSamples(@builtin(local_invocation_index)lid:u32){var unowned=0u;var first=INVALID;var magnitude=0.;var error=0u;for(var flat=lid;flat<p.sampleCapacity;flat+=64u){let s=sample(flat);if(s.valid==0u){continue;}if(!finite(s.positionPhi.w)){error|=NONFINITE;continue;}let r=owner(s.positionPhi.xyz);if(r==INVALID||r>=control.rowCount){unowned+=1u;if(s.positionPhi.w<=0.){first=min(first,s.logical);magnitude=max(magnitude,abs(s.positionPhi.w));}}}diagnosticCounts[lid]=unowned;diagnosticFirst[lid]=first;diagnosticMagnitude[lid]=magnitude;diagnosticErrors[lid]=error;workgroupBarrier();var width=32u;loop{if(width==0u){break;}if(lid<width){diagnosticCounts[lid]+=diagnosticCounts[lid+width];diagnosticFirst[lid]=min(diagnosticFirst[lid],diagnosticFirst[lid+width]);diagnosticMagnitude[lid]=max(diagnosticMagnitude[lid],diagnosticMagnitude[lid+width]);diagnosticErrors[lid]|=diagnosticErrors[lid+width];}workgroupBarrier();width>>=1u;}if(lid==0u){control.unowned=diagnosticCounts[0];control.firstUnownedLiquid=diagnosticFirst[0];control.maximumUnownedLiquidMagnitude=bitcast<u32>(diagnosticMagnitude[0]);control.flags|=diagnosticErrors[0];}}
@compute @workgroup_size(64)fn finalizeRestrictionRows(@builtin(global_invocation_id)g:vec3u){let r=g.x;if(r>=control.rowCount||r>=arrayLength(&aggregates)||r>=arrayLength(&out)){return;}let a=aggregates[r];if(a.valid==1u){out[r]=Contribution(a.centerPhi,a.minimumPhi,a.maximumPhi,1u);}}
@compute @workgroup_size(64)fn publishRestriction(@builtin(local_invocation_index)lid:u32){var errors=0u;for(var r=lid;r<control.rowCount;r+=64u){errors|=aggregates[r].error;}diagnosticErrors[lid]=errors;workgroupBarrier();var width=32u;loop{if(width==0u){break;}if(lid<width){diagnosticErrors[lid]|=diagnosticErrors[lid+width];}workgroupBarrier();width>>=1u;}if(lid==0u){control.flags|=diagnosticErrors[0];if(control.flags==0u){control.count=control.rowCount;control.maximumPerRow=1u;control.valid=0x80000000u;}else{control.count=0xffffffffu;control.maximumPerRow=1u;}}}
`;
