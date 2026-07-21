import type { WebGPUFineLevelSetBrickSource } from "./webgpu-octree-fine-levelset-bricks";
import { fineLevelSetLinearWorkgroupWGSL, planFineLevelSetDispatch2D } from "./webgpu-fine-levelset-dispatch";

export const FINE_LEVELSET_VOLUME_CONTROL_BYTES = 64;
export const FINE_LEVELSET_VOLUME_VALID = 0x8000_0000;

export interface FineLevelSetGPUVolumeControl {
  readonly flags: number; readonly initialized: boolean; readonly samples: number;
  readonly referenceVolume: number; readonly currentVolume: number; readonly interfaceArea: number;
  readonly correction: number; readonly corrected: boolean; readonly coarseVolume: number;
  readonly fineVolume: number; readonly replacedCoarseVolume: number; readonly coarseRows: number;
  readonly expectedAirSamples: number; readonly generation: number;
  readonly lookupFailureSamples: number; readonly staleOwnerSamples: number;
}

export function unpackFineLevelSetGPUVolumeControl(data: ArrayBuffer): FineLevelSetGPUVolumeControl {
  if (data.byteLength < FINE_LEVELSET_VOLUME_CONTROL_BYTES) throw new RangeError("Fine volume control needs 64 bytes");
  const u = new Uint32Array(data, 0, 16), f = new Float32Array(data, 0, 16);
  return { flags: u[0], initialized: u[1] !== 0, samples: u[2], referenceVolume: f[3], currentVolume: f[4],
    interfaceArea: f[5], correction: f[6], corrected: u[7] !== 0, coarseVolume: f[8], fineVolume: f[9],
    replacedCoarseVolume: f[10], coarseRows: u[11], expectedAirSamples: u[12], generation: u[13],
    lookupFailureSamples: u[14], staleOwnerSamples: u[15] };
}

export interface FineLevelSetVolumeCoarseSource {
  readonly headers: GPUBuffer; readonly records: GPUBuffer; readonly physicalVolumes: GPUBuffer;
  /** Current authoritative compact-coarse phi directory. Its valid empty-slot
   * complement is the only proof that a fine sample lies in coarse air. */
  readonly sampleDirectory: GPUBuffer;
  /** Publication control paired with sampleDirectory. The directory is not
   * authoritative unless both publications agree on generation and row count. */
  readonly publicationControl: GPUBuffer; readonly rowCount: GPUBuffer;
  readonly dimensions: readonly [number, number, number]; readonly physicalCellSize: number;
  readonly maximumLeafSize: number; readonly sampleHashCapacity: number; readonly maximumHashProbes?: number;
}

export interface FineLevelSetGPUVolumePlan {
  readonly coarseRowCapacity: number; readonly fineSampleCapacity: number;
  readonly coarsePartialCount: number; readonly finePartialCount: number;
  readonly coarsePartialBytes: number; readonly finePartialBytes: number;
  readonly reductionScratchBytes: number; readonly allocatedBytes: number;
}

export function planFineLevelSetGPUVolume(coarseRowCapacity: number, fineSampleCapacity: number,
  ownsControl = true, coarseDirectoryCapacity = coarseRowCapacity): FineLevelSetGPUVolumePlan {
  if (!Number.isSafeInteger(coarseRowCapacity) || coarseRowCapacity < 1
    || !Number.isSafeInteger(fineSampleCapacity) || fineSampleCapacity < 1
    || !Number.isSafeInteger(coarseDirectoryCapacity) || coarseDirectoryCapacity < coarseRowCapacity) {
    throw new RangeError("Fine volume capacities must be positive integers");
  }
  // Coarse authority is a topology-stable snapshot in the compact sample
  // directory. Scan that publication, not the live pressure-row buffers,
  // which may already describe the next dynamically rebuilt octree.
  const coarsePartialCount = Math.ceil(coarseDirectoryCapacity / 64);
  const finePartialCount = Math.ceil(fineSampleCapacity / 64);
  const coarsePartialBytes = coarsePartialCount * 16;
  const finePartialBytes = finePartialCount * 32;
  const reductionScratchBytes = Math.max(coarsePartialBytes, finePartialBytes);
  return { coarseRowCapacity, fineSampleCapacity, coarsePartialCount, finePartialCount,
    coarsePartialBytes, finePartialBytes, reductionScratchBytes,
    allocatedBytes: 64 + reductionScratchBytes + (ownsControl ? FINE_LEVELSET_VOLUME_CONTROL_BYTES : 0) };
}

/**
 * Project-specific enclosed-volume correction; this is not part of the
 * Section 5 algorithm in Aanjaneya et al.  Coarse rows integrate the complete
 * domain and resident valid fine samples replace their coarse occupancy.  It
 * uses the compact-coarse publication to classify lookup misses: only a valid
 * directory's empty-slot complement is coarse air. Malformed/stale/exhausted
 * lookups remain publication-fatal.
 */
export class WebGPUFineLevelSetVolumeCorrection {
  readonly control: GPUBuffer;
  readonly plan: FineLevelSetGPUVolumePlan;
  get allocatedBytes(): number { return this.plan.allocatedBytes; }
  private readonly coarseParams: GPUBuffer;
  private readonly reductionScratch: GPUBuffer;
  private readonly resetPipeline: GPUComputePipeline;
  private readonly coarsePartialPipeline: GPUComputePipeline;
  private readonly coarseFinalizePipeline: GPUComputePipeline;
  private readonly finePartialPipeline: GPUComputePipeline;
  private readonly fineFinalizePipeline: GPUComputePipeline;
  private readonly applyPipeline: GPUComputePipeline;
  private readonly correctedFinalizePipeline: GPUComputePipeline;
  private readonly measuredFinalizePipeline: GPUComputePipeline;
  private readonly ownsControl: boolean;
  private destroyed = false;

  constructor(private readonly device: GPUDevice, readonly source: WebGPUFineLevelSetBrickSource,
    private readonly coarse: FineLevelSetVolumeCoarseSource, sharedControl?: GPUBuffer) {
    if (!Number.isSafeInteger(coarse.sampleHashCapacity) || coarse.sampleHashCapacity < 1
      || (coarse.sampleHashCapacity & (coarse.sampleHashCapacity - 1)) !== 0
      || !Number.isSafeInteger(coarse.maximumHashProbes ?? 32) || (coarse.maximumHashProbes ?? 32) < 1) {
      throw new RangeError("Fine volume owner hash requires a power-of-two capacity and positive probe bound");
    }
    this.ownsControl = sharedControl === undefined;
    this.plan = planFineLevelSetGPUVolume(Math.floor(coarse.records.size / 16),
      source.plan.maximumResidentBricks * source.plan.samplesPerBrick,
      this.ownsControl, coarse.sampleHashCapacity);
    this.control = sharedControl ?? device.createBuffer({ label: "global fine total-volume control",
      size: FINE_LEVELSET_VOLUME_CONTROL_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    this.coarseParams = device.createBuffer({ label: "global fine total-volume coarse params", size: 64,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.reductionScratch = device.createBuffer({ label: "global fine total-volume partial reductions",
      size: this.plan.reductionScratchBytes, usage: GPUBufferUsage.STORAGE });
    const bytes = new ArrayBuffer(64), u = new Uint32Array(bytes), f = new Float32Array(bytes);
    u.set(coarse.dimensions, 0); u[3] = coarse.maximumLeafSize; u[4] = coarse.sampleHashCapacity;
    u[5] = coarse.maximumHashProbes ?? 32; f[6] = coarse.physicalCellSize;
    u[7] = this.plan.coarsePartialCount; u[8] = this.plan.finePartialCount;
    device.queue.writeBuffer(this.coarseParams, 0, bytes);
    const shaderModule = device.createShaderModule({ label: "global fine total-volume correction", code: fineLevelSetVolumeCorrectionWGSL });
    const pipeline = (entryPoint: string) => device.createComputePipeline({ label: entryPoint, layout: "auto",
      compute: { module: shaderModule, entryPoint } });
    this.resetPipeline = pipeline("resetVolumeControl");
    this.coarsePartialPipeline = pipeline("reduceCoarseVolumePartials");
    this.coarseFinalizePipeline = pipeline("finalizeCoarseVolume");
    this.finePartialPipeline = pipeline("reduceFineOverlapPartials");
    this.fineFinalizePipeline = pipeline("finalizeFineVolume");
    this.applyPipeline = pipeline("applyFineVolumeCorrection");
    this.correctedFinalizePipeline = pipeline("finalizeCorrectedFineVolume");
    this.measuredFinalizePipeline = pipeline("finalizeMeasuredFineVolume");
  }

  encode(encoder: GPUCommandEncoder): void {
    if (this.destroyed) throw new Error("Fine volume correction is destroyed");
    const run = (pipeline: GPUComputePipeline, entries: readonly [number, GPUBuffer][], groups: number, label: string) => {
      const pass = encoder.beginComputePass({ label }); pass.setPipeline(pipeline);
      pass.setBindGroup(0, this.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries:
        entries.map(([binding, buffer]) => ({ binding, resource: { buffer } })) }));
      const dispatch = planFineLevelSetDispatch2D(groups, this.device.limits.maxComputeWorkgroupsPerDimension);
      pass.dispatchWorkgroups(dispatch.x, dispatch.y); pass.end();
    };
    run(this.resetPipeline, [[5, this.control]], 1, "Reset global volume reduction");
    run(this.coarsePartialPipeline, [[0, this.source.params], [6, this.coarseParams],
      [11, this.coarse.sampleDirectory], [12, this.reductionScratch],
      [13, this.coarse.publicationControl]],
    this.plan.coarsePartialCount, "Reduce compact coarse volume partials");
    run(this.coarseFinalizePipeline, [[5, this.control], [6, this.coarseParams],
      [12, this.reductionScratch], [13, this.coarse.publicationControl]], 1,
    "Finalize compact coarse volume");
    run(this.finePartialPipeline, [[0, this.source.params], [1, this.source.metadata],
      [2, this.source.worklist], [3, this.source.flags], [4, this.source.phi], [6, this.coarseParams],
      [11, this.coarse.sampleDirectory],
      [12, this.reductionScratch], [13, this.coarse.publicationControl]],
    this.plan.finePartialCount, "Reduce resident fine overlap partials");
    run(this.fineFinalizePipeline, [[0, this.source.params], [5, this.control],
      [6, this.coarseParams], [12, this.reductionScratch]], 1, "Finalize global fine volume");
    run(this.applyPipeline, [[0, this.source.params], [1, this.source.metadata], [2, this.source.worklist],
      [3, this.source.flags], [4, this.source.phi], [5, this.control]],
    Math.ceil(this.source.plan.maximumResidentBricks * this.source.plan.samplesPerBrick / 64),
    "Apply bounded global fine normal correction");
    // The correction pass mutates the published field. Re-reduce that field
    // before publication so currentVolume describes the same phi consumed by
    // restriction, rendering, and the next transport step.
    run(this.finePartialPipeline, [[0, this.source.params], [1, this.source.metadata],
      [2, this.source.worklist], [3, this.source.flags], [4, this.source.phi], [6, this.coarseParams],
      [11, this.coarse.sampleDirectory], [12, this.reductionScratch],
      [13, this.coarse.publicationControl]],
    this.plan.finePartialCount, "Re-reduce corrected global fine volume");
    run(this.correctedFinalizePipeline, [[0, this.source.params], [5, this.control],
      [6, this.coarseParams], [12, this.reductionScratch]], 1,
    "Finalize first corrected global fine volume");
    // A topology/redistance update can require slightly more than the
    // half-fine-cell bound. Apply one residual bounded shift, matching the
    // convergence behavior used by the standalone conservation oracle.
    run(this.applyPipeline, [[0, this.source.params], [1, this.source.metadata], [2, this.source.worklist],
      [3, this.source.flags], [4, this.source.phi], [5, this.control]],
    Math.ceil(this.source.plan.maximumResidentBricks * this.source.plan.samplesPerBrick / 64),
    "Apply residual bounded global fine normal correction");
    run(this.finePartialPipeline, [[0, this.source.params], [1, this.source.metadata],
      [2, this.source.worklist], [3, this.source.flags], [4, this.source.phi], [6, this.coarseParams],
      [11, this.coarse.sampleDirectory], [12, this.reductionScratch],
      [13, this.coarse.publicationControl]],
    this.plan.finePartialCount, "Measure twice-corrected global fine volume");
    run(this.measuredFinalizePipeline, [[0, this.source.params], [5, this.control],
      [6, this.coarseParams], [12, this.reductionScratch]], 1,
    "Finalize measured global fine volume");
  }

  destroy(): void { if (this.destroyed) return; this.destroyed = true;
    this.coarseParams.destroy(); this.reductionScratch.destroy(); if (this.ownsControl) this.control.destroy(); }
}

export const fineLevelSetVolumeCorrectionWGSL = /* wgsl */ `
${fineLevelSetLinearWorkgroupWGSL}
const INVALID:u32=0xffffffffu;const VALID:u32=1u;const PUBLISHED:u32=0x80000000u;const ERROR_COARSE:u32=1u;const ERROR_FINE:u32=2u;const ERROR_OWNER:u32=4u;const OWNER_FOUND:u32=0u;const OWNER_ABSENT:u32=1u;const OWNER_PROBE_EXHAUSTED:u32=2u;const OWNER_MALFORMED:u32=3u;const OWNER_OUTSIDE:u32=4u;
struct FineParams{brickDimensions:vec3u,brickResolution:u32,sampleDimensions:vec3u,samplesPerBrick:u32,domainOrigin:vec3f,fineCellWidth:f32,hashCapacity:u32,maximumHashProbes:u32,pageCapacity:u32,generation:u32,activeCount:u32,invalid:u32,fineFactor:u32,timestep:f32}
struct CoarseParams{dimensions:vec3u,maximumLeafSize:u32,siteCapacity:u32,maximumHashProbes:u32,physicalCellSize:f32,p0:u32,p1:u32,p2:u32,p3:u32,p4:u32,p5:u32}
struct Header{cell:u32,a:u32,b:u32,size:u32,x:f32,y:f32,z:u32,w:u32,g:vec4f}struct CoarsePhi{phi:f32,minimumPhi:f32,maximumPhi:f32,flags:u32}
struct CoarseSample{cellPlusOne:u32,size:u32,phi:f32,minimumPhi:f32,maximumPhi:f32,flags:u32,row:u32,physicalVolume:f32}struct CoarseDirectory{state:u32,generation:u32,hashCapacity:u32,maximumLeafSize:u32,dimensions:vec3u,physicalCellSize:f32,entries:array<CoarseSample>}
struct Control{flags:u32,initialized:u32,samples:u32,referenceVolume:f32,currentVolume:f32,interfaceArea:f32,correction:f32,corrected:u32,coarseVolume:f32,fineVolume:f32,replacedCoarseVolume:f32,coarseRows:u32,expectedAir:u32,generation:u32,lookupFailures:u32,staleOwners:u32}
@group(0)@binding(0)var<uniform>p:FineParams;@group(0)@binding(1)var<storage,read>metadata:array<u32>;@group(0)@binding(2)var<storage,read>worklist:array<u32>;@group(0)@binding(3)var<storage,read>sampleFlags:array<u32>;@group(0)@binding(4)var<storage,read_write>phi:array<f32>;@group(0)@binding(5)var<storage,read_write>control:Control;
@group(0)@binding(6)var<uniform>c:CoarseParams;@group(0)@binding(7)var<storage,read>headers:array<Header>;@group(0)@binding(8)var<storage,read>coarsePhi:array<CoarsePhi>;@group(0)@binding(9)var<storage,read>physicalVolumes:array<f32>;@group(0)@binding(10)var<storage,read>rowCountSource:array<u32>;@group(0)@binding(11)var<storage,read>coarseDirectory:CoarseDirectory;
@group(0)@binding(12)var<storage,read_write>partials:array<u32>;
@group(0)@binding(13)var<storage,read>coarsePublication:array<u32>;
var<workgroup> sum0:array<f32,64>;var<workgroup> sum1:array<f32,64>;var<workgroup> sum2:array<f32,64>;
var<workgroup> words0:array<u32,64>;var<workgroup> words1:array<u32,64>;var<workgroup> words2:array<u32,64>;var<workgroup> words3:array<u32,64>;var<workgroup> words4:array<u32,64>;
fn finite(v:f32)->bool{return v==v&&abs(v)<3.402823e38;}fn occupancy(value:f32,width:f32)->f32{return clamp(.5-value/width,0.,1.);}
fn hash(cell:u32,size:u32)->u32{var v=cell^(size*0x9e3779b9u);v=(v^(v>>16u))*0x7feb352du;v=(v^(v>>15u))*0x846ca68bu;return v^(v>>16u);}
fn validDirectory()->bool{if(arrayLength(&coarsePublication)<13u){return false;}let generation=coarseDirectory.generation&0x3fffffffu;let fineGeneration=p.generation&0x3fffffffu;let priorFineGeneration=(fineGeneration+0x3fffffffu)&0x3fffffffu;return coarseDirectory.state==PUBLISHED&&coarsePublication[0]==0u&&coarsePublication[2]>0u&&coarsePublication[2]<=arrayLength(&coarseDirectory.entries)&&coarsePublication[11]==coarseDirectory.generation&&coarsePublication[12]==PUBLISHED&&(generation==fineGeneration||generation==priorFineGeneration)&&coarseDirectory.hashCapacity==c.siteCapacity&&coarseDirectory.hashCapacity==arrayLength(&coarseDirectory.entries)&&c.siteCapacity>0u&&(c.siteCapacity&(c.siteCapacity-1u))==0u&&coarseDirectory.maximumLeafSize==c.maximumLeafSize&&all(coarseDirectory.dimensions==c.dimensions)&&finite(coarseDirectory.physicalCellSize)&&abs(coarseDirectory.physicalCellSize-c.physicalCellSize)<=1e-5*max(coarseDirectory.physicalCellSize,c.physicalCellSize);}
fn find(cell:u32,size:u32)->vec2u{let cap=c.siteCapacity;let probes=min(c.maximumHashProbes,cap);if(probes==0u){return vec2u(INVALID,OWNER_MALFORMED);}let start=hash(cell,size)&(cap-1u);for(var q=0u;q<probes;q+=1u){let slot=(start+q)&(cap-1u);let entry=coarseDirectory.entries[slot];if(entry.cellPlusOne==0u){return vec2u(INVALID,OWNER_ABSENT);}if(entry.cellPlusOne==cell+1u&&entry.size==size){return vec2u(slot,OWNER_FOUND);}}return vec2u(INVALID,OWNER_PROBE_EXHAUSTED);}
fn owner(x:vec3f)->vec2u{if(!validDirectory()){return vec2u(INVALID,OWNER_MALFORMED);}let grid=x/c.physicalCellSize;if(any(grid<vec3f(0))||any(grid>=vec3f(c.dimensions))){return vec2u(INVALID,OWNER_OUTSIDE);}let q=vec3u(floor(grid));var size=1u;var unresolved=OWNER_ABSENT;loop{let o=(q/size)*size;let cell=o.x+c.dimensions.x*(o.y+c.dimensions.y*o.z);let found=find(cell,size);if(found.y==OWNER_FOUND){let entry=coarseDirectory.entries[found.x];if(entry.row>=coarsePublication[2]||(entry.flags&9u)!=9u||!finite(entry.phi)||!finite(entry.minimumPhi)||!finite(entry.maximumPhi)||entry.minimumPhi>entry.phi||entry.phi>entry.maximumPhi||!finite(entry.physicalVolume)||entry.physicalVolume<=0.0){return vec2u(INVALID,OWNER_MALFORMED);}return found;}if(found.y!=OWNER_ABSENT){unresolved=found.y;}if(size>=c.maximumLeafSize){break;}size*=2u;}return vec2u(INVALID,unresolved);}
fn activeSample(flat:u32)->vec2u{let count=min(worklist[0],p.pageCapacity);if(flat>=count*p.samplesPerBrick){return vec2u(INVALID);}let w=flat/p.samplesPerBrick;let local=flat-w*p.samplesPerBrick;let id=worklist[5u+w];if(id>=p.pageCapacity||metadata[id*10u+2u]!=p.generation){return vec2u(INVALID);}return vec2u(id,local);}fn unpackBrick(key:u32)->vec3u{let xy=p.brickDimensions.x*p.brickDimensions.y;let z=key/xy;let r=key-z*xy;let y=r/p.brickDimensions.x;return vec3u(r-y*p.brickDimensions.x,y,z);}fn localCoord(local:u32)->vec3u{let r=p.brickResolution;let z=local/(r*r);let q=local-z*r*r;let y=q/r;return vec3u(q-y*r,y,z);}
@compute @workgroup_size(1)fn resetVolumeControl(){let initialized=control.initialized;let reference=control.referenceVolume;control=Control(0u,initialized,0u,reference,0.,0.,0.,0u,0.,0.,0.,0u,0u,0u,0u,0u);}
@compute @workgroup_size(64)fn reduceCoarseVolumePartials(@builtin(local_invocation_id)l:vec3u,@builtin(workgroup_id)group:vec3u,@builtin(num_workgroups)n:vec3u){
 let groupFlat=fineLinearWorkgroup(group,n);let flat=groupFlat*64u+l.x;
 var volume=0.;var validRows=0u;var errors=select(0u,ERROR_COARSE,flat==0u&&!validDirectory());
 if(flat<arrayLength(&coarseDirectory.entries)){let entry=coarseDirectory.entries[flat];if(entry.cellPlusOne!=0u){let width=max(c.physicalCellSize*f32(entry.size),1e-9);if(entry.row>=coarsePublication[2]||entry.size==0u||(entry.flags&9u)!=9u||!finite(entry.phi)||!finite(entry.minimumPhi)||!finite(entry.maximumPhi)||entry.minimumPhi>entry.phi||entry.phi>entry.maximumPhi||!finite(entry.physicalVolume)||entry.physicalVolume<=0.){errors=ERROR_COARSE;}else{volume=occupancy(entry.phi,width)*entry.physicalVolume;validRows=1u;}}}
 sum0[l.x]=volume;words0[l.x]=validRows;words1[l.x]=errors;workgroupBarrier();for(var stride=32u;stride>0u;stride/=2u){if(l.x<stride){sum0[l.x]+=sum0[l.x+stride];words0[l.x]+=words0[l.x+stride];words1[l.x]|=words1[l.x+stride];}workgroupBarrier();}
 if(l.x==0u){let base=groupFlat*4u;partials[base]=bitcast<u32>(sum0[0]);partials[base+1u]=words0[0];partials[base+2u]=words1[0];partials[base+3u]=0u;}}
@compute @workgroup_size(1)fn finalizeCoarseVolume(){var coarseVolume=0.;var coarseRows=0u;var errors=0u;for(var group=0u;group<c.p0;group+=1u){let base=group*4u;coarseVolume+=bitcast<f32>(partials[base]);coarseRows+=partials[base+1u];errors|=partials[base+2u];}control.coarseVolume=coarseVolume;control.coarseRows=coarseRows;control.flags|=errors;if(!finite(coarseVolume)||arrayLength(&coarsePublication)<13u||coarseRows!=coarsePublication[2]){control.flags|=ERROR_COARSE;}control.currentVolume=coarseVolume;}
@compute @workgroup_size(64)fn reduceFineOverlapPartials(@builtin(local_invocation_id)l:vec3u,@builtin(workgroup_id)group:vec3u,@builtin(num_workgroups)n:vec3u){
 let groupFlat=fineLinearWorkgroup(group,n);let flat=groupFlat*64u+l.x;let h=p.fineCellWidth;let cellVolume=h*h*h;var fineVolume=0.;var replaced=0.;var area=0.;var samples=0u;var errors=0u;var expectedAir=0u;var lookupFailure=0u;var staleOwner=0u;let a=activeSample(flat);
 if(flat<min(worklist[0],p.pageCapacity)*p.samplesPerBrick){if(a.x==INVALID){errors|=ERROR_FINE;}else{let index=a.x*p.samplesPerBrick+a.y;if((sampleFlags[index]&VALID)!=0u){let value=phi[index];if(!finite(value)){errors|=ERROR_FINE;}else{let brick=unpackBrick(metadata[a.x*10u+1u]);let q=brick*p.brickResolution+localCoord(a.y);if(all(q<p.sampleDimensions)){let position=p.domainOrigin+(vec3f(q)+.5)*h;let ownership=owner(position);if(ownership.y==OWNER_ABSENT){expectedAir=select(0u,1u,value>=0.0);fineVolume=occupancy(value,h)*cellVolume;area=select(0.,h*h,abs(value)<=.5*h);samples=1u;}else if(ownership.y!=OWNER_FOUND){lookupFailure=1u;errors|=ERROR_OWNER;}else if(ownership.x>=arrayLength(&coarseDirectory.entries)){staleOwner=1u;errors|=ERROR_OWNER;}else{let entry=coarseDirectory.entries[ownership.x];let width=max(c.physicalCellSize*f32(entry.size),1e-9);fineVolume=occupancy(value,h)*cellVolume;replaced=occupancy(entry.phi,width)*cellVolume;area=select(0.,h*h,abs(value)<=.5*h);samples=1u;}}}}}}
 sum0[l.x]=fineVolume;sum1[l.x]=replaced;sum2[l.x]=area;words0[l.x]=samples;words1[l.x]=errors;words2[l.x]=expectedAir;words3[l.x]=lookupFailure;words4[l.x]=staleOwner;workgroupBarrier();for(var stride=32u;stride>0u;stride/=2u){if(l.x<stride){sum0[l.x]+=sum0[l.x+stride];sum1[l.x]+=sum1[l.x+stride];sum2[l.x]+=sum2[l.x+stride];words0[l.x]+=words0[l.x+stride];words1[l.x]|=words1[l.x+stride];words2[l.x]+=words2[l.x+stride];words3[l.x]+=words3[l.x+stride];words4[l.x]+=words4[l.x+stride];}workgroupBarrier();}
 if(l.x==0u){let base=groupFlat*8u;partials[base]=bitcast<u32>(sum0[0]);partials[base+1u]=bitcast<u32>(sum1[0]);partials[base+2u]=bitcast<u32>(sum2[0]);partials[base+3u]=words0[0];partials[base+4u]=words1[0];partials[base+5u]=words2[0];partials[base+6u]=words3[0];partials[base+7u]=words4[0];}}
@compute @workgroup_size(1)fn finalizeFineVolume(){var fineVolume=0.;var replacedVolume=0.;var interfaceArea=0.;var samples=0u;var errors=0u;var expectedAir=0u;var lookupFailures=0u;var staleOwners=0u;for(var group=0u;group<c.p1;group+=1u){let base=group*8u;fineVolume+=bitcast<f32>(partials[base]);replacedVolume+=bitcast<f32>(partials[base+1u]);interfaceArea+=bitcast<f32>(partials[base+2u]);samples+=partials[base+3u];errors|=partials[base+4u];expectedAir+=partials[base+5u];lookupFailures+=partials[base+6u];staleOwners+=partials[base+7u];}control.fineVolume=fineVolume;control.replacedCoarseVolume=replacedVolume;control.interfaceArea=interfaceArea;control.samples=samples;control.flags|=errors;control.expectedAir=expectedAir;control.lookupFailures=lookupFailures;control.staleOwners=staleOwners;control.currentVolume=control.coarseVolume+fineVolume-replacedVolume;control.generation=p.generation;if(!finite(control.coarseVolume)||!finite(fineVolume)||!finite(replacedVolume)||!finite(control.currentVolume)||!finite(interfaceArea)){control.flags|=ERROR_FINE;}if(samples>0u&&(fineVolume<=0.0||interfaceArea<=0.0||control.currentVolume<=0.0)){control.flags|=ERROR_FINE;}if(control.initialized==0u&&control.flags==0u&&control.coarseRows>0u){control.referenceVolume=control.currentVolume;control.initialized=1u;}if(control.initialized!=0u&&control.flags==0u&&interfaceArea>0.){control.correction=clamp((control.currentVolume-control.referenceVolume)/interfaceArea,-.5*p.fineCellWidth,.5*p.fineCellWidth);}if(control.initialized!=0u&&control.flags==0u){control.flags=PUBLISHED;}}
@compute @workgroup_size(64)fn applyFineVolumeCorrection(@builtin(workgroup_id)w:vec3u,@builtin(local_invocation_index)lid:u32,@builtin(num_workgroups)n:vec3u){if(control.flags!=PUBLISHED){return;}let flat=fineLinearWorkgroup(w,n)*64u+lid;if(flat==0u){control.corrected=1u;}let a=activeSample(flat);if(a.x==INVALID){return;}let index=a.x*p.samplesPerBrick+a.y;if((sampleFlags[index]&VALID)==0u){return;}phi[index]+=control.correction;}
fn finalizeCorrectedMeasurement(updateCorrection:bool){if(control.flags!=PUBLISHED){return;}var fineVolume=0.;var replacedVolume=0.;var interfaceArea=0.;var samples=0u;var errors=0u;var expectedAir=0u;var lookupFailures=0u;var staleOwners=0u;for(var group=0u;group<c.p1;group+=1u){let base=group*8u;fineVolume+=bitcast<f32>(partials[base]);replacedVolume+=bitcast<f32>(partials[base+1u]);interfaceArea+=bitcast<f32>(partials[base+2u]);samples+=partials[base+3u];errors|=partials[base+4u];expectedAir+=partials[base+5u];lookupFailures+=partials[base+6u];staleOwners+=partials[base+7u];}control.fineVolume=fineVolume;control.replacedCoarseVolume=replacedVolume;control.interfaceArea=interfaceArea;control.samples=samples;control.expectedAir=expectedAir;control.lookupFailures=lookupFailures;control.staleOwners=staleOwners;control.currentVolume=control.coarseVolume+fineVolume-replacedVolume;control.generation=p.generation;if(errors!=0u||lookupFailures!=0u||staleOwners!=0u||!finite(fineVolume)||!finite(replacedVolume)||!finite(interfaceArea)||!finite(control.currentVolume)||control.currentVolume<=0.0){control.flags=errors|ERROR_FINE;}else{if(updateCorrection&&interfaceArea>0.){control.correction=clamp((control.currentVolume-control.referenceVolume)/interfaceArea,-.5*p.fineCellWidth,.5*p.fineCellWidth);}control.flags=PUBLISHED;control.corrected=1u;}}
@compute @workgroup_size(1)fn finalizeCorrectedFineVolume(){finalizeCorrectedMeasurement(true);}
@compute @workgroup_size(1)fn finalizeMeasuredFineVolume(){finalizeCorrectedMeasurement(false);}
`;
