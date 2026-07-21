/** GPU-only deterministic O(rows) restriction from resident fine bricks to live octree rows. */

import { fineLevelSetLinearWorkgroupWGSL, planFineLevelSetDispatch2D } from "./webgpu-fine-levelset-dispatch";
import type { WebGPUFineLevelSetBrickSource } from "./webgpu-octree-fine-levelset-bricks";

export const FINE_TO_COARSE_LEVELSET_ERROR = Object.freeze({
  capacity: 1, unowned: 2, nonfinite: 4, unpublishedSource: 8,
} as const);

export interface FineToCoarseGPUPlan {
  readonly rowCapacity: number;
  /** Logical workload bound only; no O(samples) allocation is made. */
  readonly sampleCapacity: number;
  readonly blockCount: number;
  readonly aggregateScratchBytes: number;
  readonly allocatedBytes: number;
}

export interface FineToCoarseGPUResult {
  readonly rowOffsets: GPUBuffer;
  /** One 16-byte `{nearestPhi,minimumPhi,maximumPhi,valid}` record per row. */
  readonly contributions: GPUBuffer;
  /** First two u32s are aggregateCount and maximumAggregatesPerRow. */
  readonly counts: GPUBuffer;
  readonly aggregated: true;
}

export interface FineToCoarseGPUControl {
  readonly contributionCount: number; readonly maximumContributionsPerRow: number;
  readonly flags: number; readonly unownedSamples: number; readonly rowCount: number; readonly valid: boolean;
}

export function planFineToCoarseLevelSet(rowCapacity: number, sampleCapacity: number): FineToCoarseGPUPlan {
  if (!Number.isSafeInteger(rowCapacity) || rowCapacity < 1
    || !Number.isSafeInteger(sampleCapacity) || sampleCapacity < 1) {
    throw new RangeError("Fine-to-coarse capacities must be positive integers");
  }
  const blockCount = Math.ceil(rowCapacity / 256), aggregateScratchBytes = rowCapacity * 24;
  return { rowCapacity, sampleCapacity, blockCount, aggregateScratchBytes,
    allocatedBytes: 112 + aggregateScratchBytes + (rowCapacity + 1) * 4 + rowCapacity * 16 + 32 };
}

export function unpackFineToCoarseGPUControl(words: ArrayLike<number>): FineToCoarseGPUControl {
  if (words.length < 6) throw new RangeError("Fine-to-coarse control requires six words");
  return { contributionCount: Number(words[0]) >>> 0, maximumContributionsPerRow: Number(words[1]) >>> 0,
    flags: Number(words[2]) >>> 0, unownedSamples: Number(words[3]) >>> 0,
    rowCount: Number(words[4]) >>> 0, valid: Number(words[5]) !== 0 };
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
    this.aggregates = device.createBuffer({ label: "Fine-to-coarse row aggregate scratch",
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
    this.pipelines = { prepare: pipeline("prepareRestriction"), aggregate: pipeline("aggregateRestriction"),
      select: pipeline("selectRestrictionLogicalId"), emit: pipeline("emitRestrictionNearestPhi"),
      finalize: pipeline("finalizeRestrictionRows"), publish: pipeline("publishRestriction") };
  }

  encode(encoder: GPUCommandEncoder, fine: WebGPUFineLevelSetBrickSource, input: {
    headers: GPUBuffer; siteIndex: GPUBuffer; rowCount: GPUBuffer;
    /** Control for the topology transaction that produced `fine`. */
    topologyControl: GPUBuffer;
    dimensions: readonly [number, number, number]; physicalCellSize: number;
    maximumLeafSize: number; siteHashCapacity: number; maximumHashProbes?: number;
  }): FineToCoarseGPUResult {
    if (this.destroyed) throw new Error("Fine-to-coarse restriction is destroyed");
    const sampleCount = fine.plan.maximumResidentBricks * fine.plan.samplesPerBrick;
    if (sampleCount > this.plan.sampleCapacity) throw new RangeError("Fine-to-coarse sample source exceeds capacity");
    const data = new ArrayBuffer(112), u = new Uint32Array(data), f = new Float32Array(data);
    u.set(fine.plan.brickDimensions, 0); u[3] = fine.plan.brickResolution;
    u.set(fine.plan.sampleDimensions, 4); u[7] = fine.plan.samplesPerBrick;
    f.set(fine.plan.domainOrigin, 8); f[11] = fine.plan.fineCellWidth;
    u.set([fine.plan.maximumResidentBricks, fine.generation, this.plan.rowCapacity, sampleCount,
      input.siteHashCapacity, input.maximumHashProbes ?? 32], 12);
    u.set(input.dimensions, 20); u[23] = input.maximumLeafSize; f[24] = input.physicalCellSize;
    this.device.queue.writeBuffer(this.params, 0, data);
    const buffers = new Map<number, GPUBuffer>([[0, this.params], [1, fine.metadata], [2, fine.worklist],
      [3, fine.flags], [4, fine.phi], [5, input.headers], [6, input.siteIndex], [7, input.rowCount],
      [8, this.aggregates], [9, this.result.rowOffsets], [12, this.result.contributions], [13, this.control],
      [14, input.topologyControl]]);
    const used: Record<string, number[]> = {
      prepare: [0, 2, 7, 8, 9, 12, 13, 14], aggregate: [0, 1, 2, 3, 4, 5, 6, 8, 13],
      select: [0, 1, 2, 3, 4, 5, 6, 8, 13], emit: [0, 1, 2, 3, 4, 6, 8, 13],
      finalize: [8, 12, 13], publish: [13],
    };
    const run = (name: string, x: number, y = 1) => { const pipeline = this.pipelines[name];
      const entries = used[name].map((binding) => ({ binding, resource: { buffer: buffers.get(binding)! } }));
      const pass = encoder.beginComputePass({ label: name }); pass.setPipeline(pipeline);
      pass.setBindGroup(0, this.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries }));
      pass.dispatchWorkgroups(x, y); pass.end(); };
    run("prepare", Math.ceil((this.plan.rowCapacity + 1) / 64));
    const tiled = planFineLevelSetDispatch2D(Math.ceil(sampleCount / 64), this.device.limits.maxComputeWorkgroupsPerDimension);
    for (const name of ["aggregate", "select", "emit"] as const) if (tiled.workgroups > 0) run(name, tiled.x, tiled.y);
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
 pageCapacity:u32,generation:u32,rowCapacity:u32,sampleCapacity:u32,siteCapacity:u32,maxProbes:u32,dimensions:vec3u,maxLeaf:u32,cellWidth:f32}
struct H{cell:u32,a:u32,b:u32,size:u32,x:f32,y:f32,z:u32,w:u32,g:vec4f}struct SI{cellPlusOne:atomic<u32>,size:u32,row:u32,pad:u32}
struct C{count:u32,maximumPerRow:u32,flags:atomic<u32>,unowned:atomic<u32>,rowCount:u32,valid:u32,p0:u32,p1:u32}
struct Aggregate{count:atomic<u32>,minimumOrdered:atomic<u32>,maximumOrdered:atomic<u32>,nearestDistance:atomic<u32>,nearestLogical:atomic<u32>,nearestPhiBits:atomic<u32>}
struct Contribution{nearestPhi:f32,minimumPhi:f32,maximumPhi:f32,valid:u32}struct Sample{positionPhi:vec4f,logical:u32,valid:u32}
@group(0)@binding(0)var<uniform>p:P;@group(0)@binding(1)var<storage,read>metadata:array<u32>;@group(0)@binding(2)var<storage,read>worklist:array<u32>;
@group(0)@binding(3)var<storage,read>flags:array<u32>;@group(0)@binding(4)var<storage,read>phi:array<f32>;@group(0)@binding(5)var<storage,read>headers:array<H>;
@group(0)@binding(6)var<storage,read_write>sites:array<SI>;@group(0)@binding(7)var<storage,read>rowCountSource:array<u32>;
@group(0)@binding(8)var<storage,read_write>aggregates:array<Aggregate>;@group(0)@binding(9)var<storage,read_write>rowOffsets:array<u32>;
@group(0)@binding(12)var<storage,read_write>out:array<Contribution>;@group(0)@binding(13)var<storage,read_write>control:C;@group(0)@binding(14)var<storage,read>topologyControl:array<u32>;
const INVALID:u32=0xffffffffu;const VALID:u32=1u;const CAPACITY:u32=1u;const UNOWNED:u32=2u;const NONFINITE:u32=4u;const UNPUBLISHED_SOURCE:u32=8u;
fn finite(v:f32)->bool{return (bitcast<u32>(v)&0x7f800000u)!=0x7f800000u;}fn packBrick(q:vec3u)->u32{return q.x+p.brickDims.x*(q.y+p.brickDims.y*q.z);}
fn unpackBrick(key:u32)->vec3u{let xy=p.brickDims.x*p.brickDims.y;let z=key/xy;let r=key-z*xy;let y=r/p.brickDims.x;return vec3u(r-y*p.brickDims.x,y,z);}
fn localCoord(local:u32)->vec3u{let r=p.brickResolution;let z=local/(r*r);let q=local-z*r*r;let y=q/r;return vec3u(q-y*r,y,z);}
fn hash(c:u32,s:u32)->u32{var v=c^(s*0x9e3779b9u);v=(v^(v>>16u))*0x7feb352du;v=(v^(v>>15u))*0x846ca68bu;return v^(v>>16u);}
fn find(c:u32,s:u32)->u32{let cap=min(p.siteCapacity,arrayLength(&sites));if(cap==0u){return INVALID;}let start=hash(c,s)&(cap-1u);for(var q=0u;q<min(p.maxProbes,cap);q+=1u){let slot=(start+q)&(cap-1u);let key=atomicLoad(&sites[slot].cellPlusOne);if(key==0u){return INVALID;}if(key==c+1u&&sites[slot].size==s){return sites[slot].row;}}return INVALID;}
fn owner(x:vec3f)->u32{let g=x/p.cellWidth;if(any(g<vec3f(0))||any(g>=vec3f(p.dimensions))){return INVALID;}let q=vec3u(floor(g));var s=1u;loop{let o=(q/s)*s;let c=o.x+p.dimensions.x*(o.y+p.dimensions.y*o.z);let r=find(c,s);if(r!=INVALID){return r;}if(s>=p.maxLeaf){break;}s*=2u;}return INVALID;}
fn sample(flat:u32)->Sample{let residentCount=min(worklist[0],p.pageCapacity);if(flat>=residentCount*p.samplesPerBrick){return Sample(vec4f(0),0u,0u);}let w=flat/p.samplesPerBrick;let local=flat-w*p.samplesPerBrick;let id=worklist[5u+w];if(id>=p.pageCapacity||metadata[id*10u+2u]!=p.generation){return Sample(vec4f(0),0u,0u);}let index=id*p.samplesPerBrick+local;if((flags[index]&VALID)==0u){return Sample(vec4f(0),0u,0u);}let key=metadata[id*10u+1u];let brick=unpackBrick(key);let q=brick*p.brickResolution+localCoord(local);if(any(q>=p.sampleDims)){return Sample(vec4f(0),0u,0u);}let x=p.origin+(vec3f(q)+.5)*p.fineWidth;return Sample(vec4f(x,phi[index]),key*p.samplesPerBrick+local,1u);}
fn ordered(v:f32)->u32{let bits=bitcast<u32>(v);return select(bits^0x80000000u,~bits,(bits&0x80000000u)!=0u);}fn unordered(v:u32)->f32{return bitcast<f32>(select(~v,v^0x80000000u,(v&0x80000000u)!=0u));}
fn flatIndex(w:vec3u,lid:u32,n:vec3u)->u32{return fineLinearWorkgroup(w,n)*64u+lid;}
fn rowAndSample(s:Sample)->vec2u{if(s.valid==0u){return vec2u(INVALID);}if(!finite(s.positionPhi.w)){atomicOr(&control.flags,NONFINITE);return vec2u(INVALID);}let r=owner(s.positionPhi.xyz);if(r==INVALID||r>=control.rowCount){atomicAdd(&control.unowned,1u);if(s.positionPhi.w<=0.0){atomicOr(&control.flags,UNOWNED);}return vec2u(INVALID);}return vec2u(r,s.logical);}
@compute @workgroup_size(64)fn prepareRestriction(@builtin(global_invocation_id)g:vec3u){let i=g.x;if(i<arrayLength(&aggregates)){atomicStore(&aggregates[i].count,0u);atomicStore(&aggregates[i].minimumOrdered,0xffffffffu);atomicStore(&aggregates[i].maximumOrdered,0u);atomicStore(&aggregates[i].nearestDistance,0x7f800000u);atomicStore(&aggregates[i].nearestLogical,INVALID);atomicStore(&aggregates[i].nearestPhiBits,0u);out[i]=Contribution(0.,0.,0.,0u);}if(i<arrayLength(&rowOffsets)){rowOffsets[i]=i;}if(i==0u){control.count=0u;control.maximumPerRow=1u;atomicStore(&control.flags,0u);atomicStore(&control.unowned,0u);control.rowCount=min(rowCountSource[0],p.rowCapacity);control.valid=0u;if(arrayLength(&worklist)<5u||arrayLength(&topologyControl)<8u){atomicOr(&control.flags,UNPUBLISHED_SOURCE);}else if(worklist[1]!=p.generation||worklist[3]!=1u||worklist[4]!=1u||topologyControl[0]!=0u||topologyControl[4]!=1u||topologyControl[5]!=0u||topologyControl[7]!=0u){atomicOr(&control.flags,UNPUBLISHED_SOURCE);}}}
@compute @workgroup_size(64)fn aggregateRestriction(@builtin(workgroup_id)w:vec3u,@builtin(local_invocation_index)lid:u32,@builtin(num_workgroups)n:vec3u){let flat=flatIndex(w,lid,n);if(flat>=p.sampleCapacity){return;}let s=sample(flat);let q=rowAndSample(s);if(q.x==INVALID){return;}let h=headers[q.x];let o=vec3u(h.cell%p.dimensions.x,(h.cell/p.dimensions.x)%p.dimensions.y,h.cell/(p.dimensions.x*p.dimensions.y));let c=(vec3f(o)+.5*f32(h.size))*p.cellWidth;let d=s.positionPhi.xyz-c;atomicAdd(&aggregates[q.x].count,1u);atomicMin(&aggregates[q.x].minimumOrdered,ordered(s.positionPhi.w));atomicMax(&aggregates[q.x].maximumOrdered,ordered(s.positionPhi.w));atomicMin(&aggregates[q.x].nearestDistance,bitcast<u32>(dot(d,d)));}
@compute @workgroup_size(64)fn selectRestrictionLogicalId(@builtin(workgroup_id)w:vec3u,@builtin(local_invocation_index)lid:u32,@builtin(num_workgroups)n:vec3u){let flat=flatIndex(w,lid,n);if(flat>=p.sampleCapacity){return;}let s=sample(flat);if(s.valid==0u||!finite(s.positionPhi.w)){return;}let r=owner(s.positionPhi.xyz);if(r==INVALID||r>=control.rowCount){return;}let h=headers[r];let o=vec3u(h.cell%p.dimensions.x,(h.cell/p.dimensions.x)%p.dimensions.y,h.cell/(p.dimensions.x*p.dimensions.y));let c=(vec3f(o)+.5*f32(h.size))*p.cellWidth;let d=s.positionPhi.xyz-c;if(bitcast<u32>(dot(d,d))==atomicLoad(&aggregates[r].nearestDistance)){atomicMin(&aggregates[r].nearestLogical,s.logical);}}
@compute @workgroup_size(64)fn emitRestrictionNearestPhi(@builtin(workgroup_id)w:vec3u,@builtin(local_invocation_index)lid:u32,@builtin(num_workgroups)n:vec3u){let flat=flatIndex(w,lid,n);if(flat>=p.sampleCapacity){return;}let s=sample(flat);if(s.valid==0u||!finite(s.positionPhi.w)){return;}let r=owner(s.positionPhi.xyz);if(r!=INVALID&&r<control.rowCount&&s.logical==atomicLoad(&aggregates[r].nearestLogical)){atomicStore(&aggregates[r].nearestPhiBits,bitcast<u32>(s.positionPhi.w));}}
@compute @workgroup_size(64)fn finalizeRestrictionRows(@builtin(global_invocation_id)g:vec3u){let r=g.x;if(r>=control.rowCount||r>=arrayLength(&aggregates)||r>=arrayLength(&out)){return;}if(atomicLoad(&aggregates[r].count)>0u){out[r]=Contribution(bitcast<f32>(atomicLoad(&aggregates[r].nearestPhiBits)),unordered(atomicLoad(&aggregates[r].minimumOrdered)),unordered(atomicLoad(&aggregates[r].maximumOrdered)),1u);}}
@compute @workgroup_size(1)fn publishRestriction(){if(atomicLoad(&control.flags)==0u){control.count=control.rowCount;control.maximumPerRow=1u;control.valid=0x80000000u;}else{control.count=0xffffffffu;control.maximumPerRow=1u;}}
`;
