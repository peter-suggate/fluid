import type { WebGPUFineLevelSetBrickSource } from "./webgpu-octree-fine-levelset-bricks";
import { fineLevelSetLinearWorkgroupWGSL, planFineLevelSetDispatch2D } from "./webgpu-fine-levelset-dispatch";
import { PassBroker } from "./webgpu-pass-broker";
import {
  createGPULogicalActivityAdoptionContext,
  type GPULogicalActivityAdoptionContext,
} from "./gpu-logical-activity-adoption";
import { performanceShaderVariant } from "./stores/performance-instrumentation-store";

export const FINE_LEVELSET_VOLUME_CONTROL_BYTES = 64;
export const FINE_LEVELSET_VOLUME_VALID = 0x8000_0000;
const FINE_LEVELSET_VOLUME_INDIRECT_BYTES = 16;

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
  readonly maximumLeafSize: number; readonly sampleRowCapacity: number;
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
    allocatedBytes: 64 + 16 + reductionScratchBytes + FINE_LEVELSET_VOLUME_INDIRECT_BYTES + 12
      + (ownsControl ? FINE_LEVELSET_VOLUME_CONTROL_BYTES : 0) };
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
  private readonly referenceDeltaParams: GPUBuffer;
  private readonly reductionScratch: GPUBuffer;
  private readonly fineDispatch: GPUBuffer;
  private readonly coarseDispatch: GPUBuffer;
  private readonly resetPipeline: GPUComputePipeline;
  private readonly addReferencePipeline: GPUComputePipeline;
  private readonly coarsePartialPipeline: GPUComputePipeline;
  private readonly prepareCoarseDispatchPipeline: GPUComputePipeline;
  private readonly coarseFinalizePipeline: GPUComputePipeline;
  private readonly prepareFineDispatchPipeline: GPUComputePipeline;
  private readonly finePartialPipeline: GPUComputePipeline;
  private readonly fineFinalizePipeline: GPUComputePipeline;
  private readonly applyPipeline: GPUComputePipeline;
  private readonly correctedFinalizePipeline: GPUComputePipeline;
  private readonly measuredFinalizePipeline: GPUComputePipeline;
  private readonly groups = new Map<GPUComputePipeline, GPUBindGroup>();
  private readonly ownsControl: boolean;
  private pendingReferenceVolume = 0;
  private destroyed = false;

  constructor(private readonly device: GPUDevice, readonly source: WebGPUFineLevelSetBrickSource,
    private readonly coarse: FineLevelSetVolumeCoarseSource, sharedControl?: GPUBuffer) {
    if (!Number.isSafeInteger(coarse.sampleRowCapacity) || coarse.sampleRowCapacity < 1) {
      throw new RangeError("Fine volume coarse row-directory capacity must be positive");
    }
    this.ownsControl = sharedControl === undefined;
    this.plan = planFineLevelSetGPUVolume(Math.floor(coarse.records.size / 16),
      source.plan.maximumResidentBricks * source.plan.samplesPerBrick,
      this.ownsControl, coarse.sampleRowCapacity);
    this.control = sharedControl ?? device.createBuffer({ label: "global fine total-volume control",
      size: FINE_LEVELSET_VOLUME_CONTROL_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    this.coarseParams = device.createBuffer({ label: "global fine total-volume coarse params", size: 64,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.referenceDeltaParams = device.createBuffer({ label: "global fine injected-volume reference delta", size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.reductionScratch = device.createBuffer({ label: "global fine total-volume partial reductions",
      size: this.plan.reductionScratchBytes, usage: GPUBufferUsage.STORAGE });
    this.fineDispatch = device.createBuffer({ label: "global fine total-volume active-sample dispatch",
      size: FINE_LEVELSET_VOLUME_INDIRECT_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT });
    this.coarseDispatch = device.createBuffer({ label: "global fine total-volume exact coarse-row dispatch",
      size: 12, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT });
    const bytes = new ArrayBuffer(64), u = new Uint32Array(bytes), f = new Float32Array(bytes);
    u.set(coarse.dimensions, 0); u[3] = coarse.maximumLeafSize; u[4] = coarse.sampleRowCapacity;
    f[6] = coarse.physicalCellSize;
    u[7] = this.plan.coarsePartialCount; u[8] = this.plan.finePartialCount;
    u[9] = device.limits.maxComputeWorkgroupsPerDimension;
    device.queue.writeBuffer(this.coarseParams, 0, bytes);
    const volumeProfile = performanceShaderVariant();
    const volumeActivity = createGPULogicalActivityAdoptionContext({
      moduleId: "octree/fine-volume-correction",
      profile: volumeProfile,
    });
    const volumeVariant = volumeActivity.module(
      fineLevelSetVolumeActivityShader(volumeActivity),
      `octree/fine-volume-correction/${volumeProfile.cacheKey}`,
    );
    const shaderModule = device.createShaderModule({
      label: "global fine total-volume correction",
      code: volumeVariant.code,
    });
    const pipeline = (entryPoint: string) => device.createComputePipeline({ label: entryPoint, layout: "auto",
      compute: { module: shaderModule, entryPoint } });
    this.resetPipeline = pipeline("resetVolumeControl");
    this.addReferencePipeline = pipeline("addReferenceVolume");
    this.prepareCoarseDispatchPipeline = pipeline("prepareCoarseVolumeDispatch");
    this.coarsePartialPipeline = pipeline("reduceCoarseVolumePartials");
    this.coarseFinalizePipeline = pipeline("finalizeCoarseVolume");
    this.prepareFineDispatchPipeline = pipeline("prepareFineVolumeDispatch");
    this.finePartialPipeline = pipeline("reduceFineOverlapPartials");
    this.fineFinalizePipeline = pipeline("finalizeFineVolume");
    this.applyPipeline = volumeActivity.registerPipeline(pipeline("applyFineVolumeCorrection"));
    this.correctedFinalizePipeline = pipeline("finalizeCorrectedFineVolume");
    this.measuredFinalizePipeline = pipeline("finalizeMeasuredFineVolume");
    const cache = (pipeline: GPUComputePipeline, entries: readonly [number, GPUBuffer][]) => {
      this.groups.set(pipeline, device.createBindGroup({ layout: pipeline.getBindGroupLayout(0),
        entries: entries.map(([binding, buffer]) => ({ binding, resource: { buffer } })) }));
    };
    cache(this.resetPipeline, [[5, this.control]]);
    cache(this.addReferencePipeline, [[5, this.control], [16, this.referenceDeltaParams]]);
    cache(this.prepareCoarseDispatchPipeline, [[0, this.source.params], [6, this.coarseParams], [11, this.coarse.sampleDirectory],
      [12, this.reductionScratch], [13, this.coarse.publicationControl], [15, this.coarseDispatch]]);
    cache(this.coarsePartialPipeline, [[0, this.source.params], [6, this.coarseParams],
      [11, this.coarse.sampleDirectory], [12, this.reductionScratch], [13, this.coarse.publicationControl]]);
    cache(this.coarseFinalizePipeline, [[5, this.control], [6, this.coarseParams],
      [12, this.reductionScratch], [13, this.coarse.publicationControl]]);
    cache(this.prepareFineDispatchPipeline, [[0, this.source.params], [2, this.source.worklist],
      [6, this.coarseParams], [12, this.reductionScratch], [14, this.fineDispatch]]);
    cache(this.finePartialPipeline, [[0, this.source.params], [1, this.source.metadata],
      [2, this.source.worklist], [3, this.source.flags], [4, this.source.phi], [6, this.coarseParams],
      [11, this.coarse.sampleDirectory], [12, this.reductionScratch], [13, this.coarse.publicationControl]]);
    cache(this.fineFinalizePipeline, [[0, this.source.params], [5, this.control],
      [6, this.coarseParams], [12, this.reductionScratch]]);
    cache(this.applyPipeline, [[0, this.source.params], [1, this.source.metadata],
      [2, this.source.worklist], [3, this.source.flags], [4, this.source.phi], [5, this.control]]);
    cache(this.correctedFinalizePipeline, [[0, this.source.params], [5, this.control],
      [6, this.coarseParams], [12, this.reductionScratch]]);
    cache(this.measuredFinalizePipeline, [[0, this.source.params], [5, this.control],
      [6, this.coarseParams], [12, this.reductionScratch]]);
  }

  /**
   * Record the paper-path volume measurement without mutating phi.  The
   * Aanjaneya narrow-band scheme does not apply a global level-set offset;
   * production uses this cheaper telemetry-only path so an unchanged page can
   * remain an exact delta carry.
   */
  encodeMeasurement(broker: PassBroker): void { this.encodePasses(broker, false); }

  /** Standalone project-specific conservation experiment. */
  encode(broker: PassBroker): void { this.encodePasses(broker, true); }

  /** Add analytic source volume before the next resident measurement.
   * This updates the project-specific correction target; it is deliberately
   * not presented as part of Aanjaneya et al.'s Section 5 algorithm. */
  addReferenceVolume(volume_m3: number): void {
    if (!Number.isFinite(volume_m3) || volume_m3 < 0) {
      throw new RangeError("Fine volume reference delta must be finite and non-negative");
    }
    this.pendingReferenceVolume += volume_m3;
  }

  private encodePasses(broker: PassBroker, applyCorrection: boolean): void {
    if (this.destroyed) throw new Error("Fine volume correction is destroyed");
    const run = (pipeline: GPUComputePipeline, _entries: readonly [number, GPUBuffer][], groups: number, label: string) => {
      const pass = broker.compute({ label }); pass.setPipeline(pipeline);
      pass.setBindGroup(0, this.groups.get(pipeline)!);
      const dispatch = planFineLevelSetDispatch2D(groups, this.device.limits.maxComputeWorkgroupsPerDimension);
      pass.dispatchWorkgroups(dispatch.x, dispatch.y);
    };
    const runFine = (pipeline: GPUComputePipeline, _entries: readonly [number, GPUBuffer][], label: string) => {
      const pass = broker.compute({ label }); pass.setPipeline(pipeline);
      pass.setBindGroup(0, this.groups.get(pipeline)!);
      pass.dispatchWorkgroupsIndirect(this.fineDispatch, 0);
    };
    if (this.pendingReferenceVolume > 0) {
      this.device.queue.writeBuffer(this.referenceDeltaParams, 0,
        new Float32Array([this.pendingReferenceVolume, 0, 0, 0]));
      run(this.addReferencePipeline, [[5, this.control], [16, this.referenceDeltaParams]], 1,
        "Advance global fine inflow volume reference");
      this.pendingReferenceVolume = 0;
    }
    run(this.resetPipeline, [[5, this.control]], 1, "Reset global volume reduction");
    run(this.prepareCoarseDispatchPipeline, [[0, this.source.params], [6, this.coarseParams], [11, this.coarse.sampleDirectory],
      [12, this.reductionScratch], [13, this.coarse.publicationControl], [15, this.coarseDispatch]], 1,
    "Prepare exact compact coarse volume dispatch");
    broker.fence("exact coarse volume dispatch published");
    {
      const pass = broker.compute({ label: "Reduce compact coarse volume partials" });
      pass.setPipeline(this.coarsePartialPipeline); pass.setBindGroup(0, this.groups.get(this.coarsePartialPipeline)!);
      pass.dispatchWorkgroupsIndirect(this.coarseDispatch, 0);
    }
    run(this.coarseFinalizePipeline, [[5, this.control], [6, this.coarseParams],
      [12, this.reductionScratch], [13, this.coarse.publicationControl]], 1,
    "Finalize compact coarse volume");
    // Preserve the capacity-shaped reduction tree (and therefore its exact
    // floating-point addition order), but stop launching sample lanes for
    // absent pages. The preparation pass zeroes the same partial records that
    // inactive capacity workgroups used to overwrite with zero.
    run(this.prepareFineDispatchPipeline, [[0, this.source.params], [2, this.source.worklist],
      [6, this.coarseParams], [12, this.reductionScratch], [14, this.fineDispatch]], 1,
    "Prepare active global fine volume dispatch");
    broker.fence("active fine volume dispatch published");
    runFine(this.finePartialPipeline, [[0, this.source.params], [1, this.source.metadata],
      [2, this.source.worklist], [3, this.source.flags], [4, this.source.phi], [6, this.coarseParams],
      [11, this.coarse.sampleDirectory],
      [12, this.reductionScratch], [13, this.coarse.publicationControl]],
    "Reduce resident fine overlap partials");
    run(this.fineFinalizePipeline, [[0, this.source.params], [5, this.control],
      [6, this.coarseParams], [12, this.reductionScratch]], 1, "Finalize global fine volume");
    if (!applyCorrection) {
      broker.fence("global fine volume measurement complete");
      return;
    }
    runFine(this.applyPipeline, [[0, this.source.params], [1, this.source.metadata], [2, this.source.worklist],
      [3, this.source.flags], [4, this.source.phi], [5, this.control]],
    "Apply bounded global fine normal correction");
    // The correction pass mutates the published field. Re-reduce that field
    // before publication so currentVolume describes the same phi consumed by
    // restriction, rendering, and the next transport step.
    runFine(this.finePartialPipeline, [[0, this.source.params], [1, this.source.metadata],
      [2, this.source.worklist], [3, this.source.flags], [4, this.source.phi], [6, this.coarseParams],
      [11, this.coarse.sampleDirectory], [12, this.reductionScratch],
      [13, this.coarse.publicationControl]],
    "Re-reduce corrected global fine volume");
    run(this.correctedFinalizePipeline, [[0, this.source.params], [5, this.control],
      [6, this.coarseParams], [12, this.reductionScratch]], 1,
    "Finalize first corrected global fine volume");
    // A topology/redistance update can require slightly more than the
    // half-fine-cell bound. Apply one residual bounded shift, matching the
    // convergence behavior used by the standalone conservation oracle.
    runFine(this.applyPipeline, [[0, this.source.params], [1, this.source.metadata], [2, this.source.worklist],
      [3, this.source.flags], [4, this.source.phi], [5, this.control]],
    "Apply residual bounded global fine normal correction");
    runFine(this.finePartialPipeline, [[0, this.source.params], [1, this.source.metadata],
      [2, this.source.worklist], [3, this.source.flags], [4, this.source.phi], [6, this.coarseParams],
      [11, this.coarse.sampleDirectory], [12, this.reductionScratch],
      [13, this.coarse.publicationControl]],
    "Measure twice-corrected global fine volume");
    run(this.measuredFinalizePipeline, [[0, this.source.params], [5, this.control],
      [6, this.coarseParams], [12, this.reductionScratch]], 1,
    "Finalize measured global fine volume");
    broker.fence("global fine volume correction complete");
  }

  destroy(): void { if (this.destroyed) return; this.destroyed = true;
    this.coarseParams.destroy(); this.referenceDeltaParams.destroy(); this.reductionScratch.destroy(); this.fineDispatch.destroy(); this.coarseDispatch.destroy();
    if (this.ownsControl) this.control.destroy(); }
}

export const fineLevelSetVolumeCorrectionWGSL = /* wgsl */ `
${fineLevelSetLinearWorkgroupWGSL}
const INVALID:u32=0xffffffffu;const VALID:u32=1u;const PUBLISHED:u32=0x80000000u;const ERROR_COARSE:u32=1u;const ERROR_FINE:u32=2u;const ERROR_OWNER:u32=4u;const OWNER_FOUND:u32=0u;const OWNER_ABSENT:u32=1u;const OWNER_MALFORMED:u32=3u;const OWNER_OUTSIDE:u32=4u;
struct FineParams{brickDimensions:vec3u,brickResolution:u32,sampleDimensions:vec3u,samplesPerBrick:u32,domainOrigin:vec3f,fineCellWidth:f32,worklistCapacity:u32,worklistHeaderWords:u32,pageCapacity:u32,generation:u32,activeCount:u32,invalid:u32,fineFactor:u32,timestep:f32}
struct CoarseParams{dimensions:vec3u,maximumLeafSize:u32,rowCapacity:u32,pad0:u32,physicalCellSize:f32,p0:u32,p1:u32,p2:u32,p3:u32,p4:u32,p5:u32}
struct ReferenceDelta{volume:f32,p0:f32,p1:f32,p2:f32}
struct Header{cell:u32,a:u32,b:u32,size:u32,x:f32,y:f32,z:u32,w:u32,g:vec4f}struct CoarsePhi{phi:f32,minimumPhi:f32,maximumPhi:f32,flags:u32}
struct CoarseSample{cellPlusOne:u32,size:u32,phi:f32,minimumPhi:f32,maximumPhi:f32,flags:u32,row:u32,physicalVolume:f32}struct CoarseDirectory{state:u32,generation:u32,rowCount:u32,maximumLeafSize:u32,dimensions:vec3u,physicalCellSize:f32,entries:array<CoarseSample>}
struct Control{flags:u32,initialized:u32,samples:u32,referenceVolume:f32,currentVolume:f32,interfaceArea:f32,correction:f32,corrected:u32,coarseVolume:f32,fineVolume:f32,replacedCoarseVolume:f32,coarseRows:u32,expectedAir:u32,generation:u32,lookupFailures:u32,staleOwners:u32}
@group(0)@binding(0)var<uniform>p:FineParams;@group(0)@binding(1)var<storage,read>metadata:array<u32>;@group(0)@binding(2)var<storage,read>worklist:array<u32>;@group(0)@binding(3)var<storage,read>sampleFlags:array<u32>;@group(0)@binding(4)var<storage,read_write>phi:array<f32>;@group(0)@binding(5)var<storage,read_write>control:Control;
@group(0)@binding(16)var<uniform>referenceDelta:ReferenceDelta;
@group(0)@binding(6)var<uniform>c:CoarseParams;@group(0)@binding(7)var<storage,read>headers:array<Header>;@group(0)@binding(8)var<storage,read>coarsePhi:array<CoarsePhi>;@group(0)@binding(9)var<storage,read>physicalVolumes:array<f32>;@group(0)@binding(10)var<storage,read>rowCountSource:array<u32>;@group(0)@binding(11)var<storage,read>coarseDirectory:CoarseDirectory;
@group(0)@binding(12)var<storage,read_write>partials:array<u32>;
@group(0)@binding(13)var<storage,read>coarsePublication:array<u32>;
@group(0)@binding(14)var<storage,read_write>fineDispatch:array<u32>;
@group(0)@binding(15)var<storage,read_write>coarseDispatch:array<u32>;
var<workgroup> sum0:array<f32,256>;var<workgroup> sum1:array<f32,256>;var<workgroup> sum2:array<f32,256>;
var<workgroup> words0:array<u32,256>;var<workgroup> words1:array<u32,256>;var<workgroup> words2:array<u32,256>;var<workgroup> words3:array<u32,256>;var<workgroup> words4:array<u32,256>;
struct CoarseVolumeReduction{volume:f32,rows:u32,errors:u32}
struct FineVolumeReduction{fineVolume:f32,replacedVolume:f32,interfaceArea:f32,samples:u32,errors:u32,expectedAir:u32,lookupFailures:u32,staleOwners:u32}
fn finite(v:f32)->bool{return v==v&&abs(v)<3.402823e38;}fn occupancy(value:f32,width:f32)->f32{return clamp(.5-value/width,0.,1.);}
fn balancedReductionRange(count:u32,lid:u32)->vec2u{let width=count/256u;let remainder=count%256u;let begin=lid*width+min(lid,remainder);let length=width+select(0u,1u,lid<remainder);return vec2u(begin,begin+length);}
fn mortonPart(value:u32)->u32{var x=value&1023u;x=(x|(x<<16u))&0x030000ffu;x=(x|(x<<8u))&0x0300f00fu;x=(x|(x<<4u))&0x030c30c3u;x=(x|(x<<2u))&0x09249249u;return x;}fn morton(cell:u32)->u32{let q=vec3u(cell%c.dimensions.x,(cell/c.dimensions.x)%c.dimensions.y,cell/(c.dimensions.x*c.dimensions.y));return mortonPart(q.x)|(mortonPart(q.y)<<1u)|(mortonPart(q.z)<<2u);}fn level(size:u32)->u32{return 31u-countLeadingZeros(size);}fn less(aLevel:u32,aMorton:u32,bLevel:u32,bMorton:u32)->bool{return aLevel<bLevel||(aLevel==bLevel&&aMorton<bMorton);}
fn validDirectory()->bool{if(arrayLength(&coarsePublication)<12u){return false;}let generation=coarseDirectory.generation&0x3fffffffu;let fineGeneration=p.generation&0x3fffffffu;let priorFineGeneration=(fineGeneration+0x3fffffffu)&0x3fffffffu;return coarseDirectory.state==PUBLISHED&&coarsePublication[0]==0u&&coarsePublication[2]>0u&&coarsePublication[2]==coarseDirectory.rowCount&&coarseDirectory.rowCount<=arrayLength(&coarseDirectory.entries)&&coarseDirectory.rowCount<=c.rowCapacity&&coarsePublication[10]==coarseDirectory.generation&&coarsePublication[11]==PUBLISHED&&(generation==fineGeneration||generation==priorFineGeneration)&&c.rowCapacity==arrayLength(&coarseDirectory.entries)&&coarseDirectory.maximumLeafSize==c.maximumLeafSize&&all(coarseDirectory.dimensions==c.dimensions)&&finite(coarseDirectory.physicalCellSize)&&abs(coarseDirectory.physicalCellSize-c.physicalCellSize)<=1e-5*max(coarseDirectory.physicalCellSize,c.physicalCellSize);}
fn find(cell:u32,size:u32)->vec2u{let count=min(coarseDirectory.rowCount,arrayLength(&coarseDirectory.entries));let wantedLevel=level(size);let wantedMorton=morton(cell);var low=0u;var high=count;while(low<high){let middle=low+(high-low)/2u;let entry=coarseDirectory.entries[middle];let entryMorton=morton(entry.cellPlusOne-1u);if(less(level(entry.size),entryMorton,wantedLevel,wantedMorton)){low=middle+1u;}else{high=middle;}}if(low<count){let entry=coarseDirectory.entries[low];if(entry.cellPlusOne==cell+1u&&entry.size==size){return vec2u(low,OWNER_FOUND);}}return vec2u(INVALID,OWNER_ABSENT);}
fn owner(x:vec3f)->vec2u{if(!validDirectory()){return vec2u(INVALID,OWNER_MALFORMED);}let grid=x/c.physicalCellSize;if(any(grid<vec3f(0))||any(grid>=vec3f(c.dimensions))){return vec2u(INVALID,OWNER_OUTSIDE);}let q=vec3u(floor(grid));var size=1u;var unresolved=OWNER_ABSENT;loop{let o=(q/size)*size;let cell=o.x+c.dimensions.x*(o.y+c.dimensions.y*o.z);let found=find(cell,size);if(found.y==OWNER_FOUND){let entry=coarseDirectory.entries[found.x];if(entry.row>=coarsePublication[2]||(entry.flags&9u)!=9u||!finite(entry.phi)||!finite(entry.minimumPhi)||!finite(entry.maximumPhi)||entry.minimumPhi>entry.phi||entry.phi>entry.maximumPhi||!finite(entry.physicalVolume)||entry.physicalVolume<=0.0){return vec2u(INVALID,OWNER_MALFORMED);}return found;}if(found.y!=OWNER_ABSENT){unresolved=found.y;}if(size>=c.maximumLeafSize){break;}size*=2u;}return vec2u(INVALID,unresolved);}
fn activeSample(flat:u32)->vec2u{let count=min(worklist[1],p.pageCapacity);if(flat>=count*p.samplesPerBrick){return vec2u(INVALID);}let w=flat/p.samplesPerBrick;let local=flat-w*p.samplesPerBrick;let id=worklist[7u+w];if(id>=p.pageCapacity||metadata[id*10u+2u]!=p.generation){return vec2u(INVALID);}return vec2u(id,local);}fn unpackBrick(key:u32)->vec3u{let xy=p.brickDimensions.x*p.brickDimensions.y;let z=key/xy;let r=key-z*xy;let y=r/p.brickDimensions.x;return vec3u(r-y*p.brickDimensions.x,y,z);}fn localCoord(local:u32)->vec3u{let r=p.brickResolution;let z=local/(r*r);let q=local-z*r*r;let y=q/r;return vec3u(q-y*r,y,z);}
@compute @workgroup_size(1)fn resetVolumeControl(){let initialized=control.initialized;let reference=control.referenceVolume;control=Control(0u,initialized,0u,reference,0.,0.,0.,0u,0.,0.,0.,0u,0u,0u,0u,0u);}
@compute @workgroup_size(1)fn addReferenceVolume(){if(control.initialized!=0u&&finite(referenceDelta.volume)&&referenceDelta.volume>0.){control.referenceVolume+=referenceDelta.volume;}}
@compute @workgroup_size(256)fn prepareCoarseVolumeDispatch(@builtin(local_invocation_index)lid:u32){
 let count=select(0u,min(coarseDirectory.rowCount,c.rowCapacity),validDirectory());let groups=(count+63u)/64u;
 for(var word=groups*4u+lid;word<c.p0*4u;word+=256u){partials[word]=0u;}
 if(lid==0u){let x=min(groups,c.p2);coarseDispatch[0]=x;coarseDispatch[1]=select(1u,(groups+x-1u)/x,x>0u);coarseDispatch[2]=1u;}}
@compute @workgroup_size(64)fn reduceCoarseVolumePartials(@builtin(local_invocation_id)l:vec3u,@builtin(workgroup_id)group:vec3u,@builtin(num_workgroups)n:vec3u){
 let groupFlat=fineLinearWorkgroup(group,n);let flat=groupFlat*64u+l.x;
 var volume=0.;var validRows=0u;var errors=select(0u,ERROR_COARSE,flat==0u&&!validDirectory());
 if(flat<coarseDirectory.rowCount&&flat<arrayLength(&coarseDirectory.entries)){let entry=coarseDirectory.entries[flat];let width=max(c.physicalCellSize*f32(entry.size),1e-9);if(entry.cellPlusOne==0u||entry.row>=coarsePublication[2]||entry.size==0u||(entry.flags&9u)!=9u||!finite(entry.phi)||!finite(entry.minimumPhi)||!finite(entry.maximumPhi)||entry.minimumPhi>entry.phi||entry.phi>entry.maximumPhi||!finite(entry.physicalVolume)||entry.physicalVolume<=0.){errors=ERROR_COARSE;}else{volume=occupancy(entry.phi,width)*entry.physicalVolume;validRows=1u;}}
 sum0[l.x]=volume;words0[l.x]=validRows;words1[l.x]=errors;workgroupBarrier();for(var stride=32u;stride>0u;stride/=2u){if(l.x<stride){sum0[l.x]+=sum0[l.x+stride];words0[l.x]+=words0[l.x+stride];words1[l.x]|=words1[l.x+stride];}workgroupBarrier();}
 if(l.x==0u){let base=groupFlat*4u;partials[base]=bitcast<u32>(sum0[0]);partials[base+1u]=words0[0];partials[base+2u]=words1[0];partials[base+3u]=0u;}}
fn reduceCoarsePartialHierarchy(lid:u32)->CoarseVolumeReduction{let range=balancedReductionRange(c.p0,lid);var volume=0.;var rows=0u;var errors=0u;for(var group=range.x;group<range.y;group+=1u){let base=group*4u;volume+=bitcast<f32>(partials[base]);rows+=partials[base+1u];errors|=partials[base+2u];}sum0[lid]=volume;words0[lid]=rows;words1[lid]=errors;workgroupBarrier();for(var stride=128u;stride>0u;stride/=2u){if(lid<stride){sum0[lid]+=sum0[lid+stride];words0[lid]+=words0[lid+stride];words1[lid]|=words1[lid+stride];}workgroupBarrier();}return CoarseVolumeReduction(sum0[0],words0[0],words1[0]);}
@compute @workgroup_size(256)fn finalizeCoarseVolume(@builtin(local_invocation_index)lid:u32){let result=reduceCoarsePartialHierarchy(lid);if(lid!=0u){return;}control.coarseVolume=result.volume;control.coarseRows=result.rows;control.flags|=result.errors;if(!finite(result.volume)||arrayLength(&coarsePublication)<13u||result.rows!=coarsePublication[2]){control.flags|=ERROR_COARSE;}control.currentVolume=result.volume;}
@compute @workgroup_size(256)fn prepareFineVolumeDispatch(@builtin(local_invocation_index)lid:u32){
 let pages=min(worklist[1],p.pageCapacity);let samples=pages*p.samplesPerBrick;let groups=(samples+63u)/64u;
 for(var word=groups*8u+lid;word<c.p1*8u;word+=256u){partials[word]=0u;}
 if(lid==0u){if(groups==0u){fineDispatch[0]=0u;fineDispatch[1]=1u;fineDispatch[2]=1u;}else{let x=min(groups,c.p2);fineDispatch[0]=x;fineDispatch[1]=(groups+x-1u)/x;fineDispatch[2]=1u;}fineDispatch[3]=samples;}}
@compute @workgroup_size(64)fn reduceFineOverlapPartials(@builtin(local_invocation_id)l:vec3u,@builtin(workgroup_id)group:vec3u,@builtin(num_workgroups)n:vec3u){
 let groupFlat=fineLinearWorkgroup(group,n);let flat=groupFlat*64u+l.x;let h=p.fineCellWidth;let cellVolume=h*h*h;var fineVolume=0.;var replaced=0.;var area=0.;var samples=0u;var errors=0u;var expectedAir=0u;var lookupFailure=0u;var staleOwner=0u;let a=activeSample(flat);
 if(flat<min(worklist[1],p.pageCapacity)*p.samplesPerBrick){if(a.x==INVALID){errors|=ERROR_FINE;}else{let index=a.x*p.samplesPerBrick+a.y;if((sampleFlags[index]&VALID)!=0u){let value=phi[index];if(!finite(value)){errors|=ERROR_FINE;}else{let brick=unpackBrick(metadata[a.x*10u+1u]);let q=brick*p.brickResolution+localCoord(a.y);if(all(q<p.sampleDimensions)){let position=p.domainOrigin+(vec3f(q)+.5)*h;let ownership=owner(position);if(ownership.y==OWNER_ABSENT){expectedAir=select(0u,1u,value>=0.0);fineVolume=occupancy(value,h)*cellVolume;area=select(0.,h*h,abs(value)<=.5*h);samples=1u;}else if(ownership.y!=OWNER_FOUND){lookupFailure=1u;errors|=ERROR_OWNER;}else if(ownership.x>=arrayLength(&coarseDirectory.entries)){staleOwner=1u;errors|=ERROR_OWNER;}else{let entry=coarseDirectory.entries[ownership.x];let width=max(c.physicalCellSize*f32(entry.size),1e-9);fineVolume=occupancy(value,h)*cellVolume;replaced=occupancy(entry.phi,width)*cellVolume;area=select(0.,h*h,abs(value)<=.5*h);samples=1u;}}}}}}
 sum0[l.x]=fineVolume;sum1[l.x]=replaced;sum2[l.x]=area;words0[l.x]=samples;words1[l.x]=errors;words2[l.x]=expectedAir;words3[l.x]=lookupFailure;words4[l.x]=staleOwner;workgroupBarrier();for(var stride=32u;stride>0u;stride/=2u){if(l.x<stride){sum0[l.x]+=sum0[l.x+stride];sum1[l.x]+=sum1[l.x+stride];sum2[l.x]+=sum2[l.x+stride];words0[l.x]+=words0[l.x+stride];words1[l.x]|=words1[l.x+stride];words2[l.x]+=words2[l.x+stride];words3[l.x]+=words3[l.x+stride];words4[l.x]+=words4[l.x+stride];}workgroupBarrier();}
 if(l.x==0u){let base=groupFlat*8u;partials[base]=bitcast<u32>(sum0[0]);partials[base+1u]=bitcast<u32>(sum1[0]);partials[base+2u]=bitcast<u32>(sum2[0]);partials[base+3u]=words0[0];partials[base+4u]=words1[0];partials[base+5u]=words2[0];partials[base+6u]=words3[0];partials[base+7u]=words4[0];}}
fn reduceFinePartialHierarchy(lid:u32)->FineVolumeReduction{let range=balancedReductionRange(c.p1,lid);var fineVolume=0.;var replacedVolume=0.;var interfaceArea=0.;var samples=0u;var errors=0u;var expectedAir=0u;var lookupFailures=0u;var staleOwners=0u;for(var group=range.x;group<range.y;group+=1u){let base=group*8u;fineVolume+=bitcast<f32>(partials[base]);replacedVolume+=bitcast<f32>(partials[base+1u]);interfaceArea+=bitcast<f32>(partials[base+2u]);samples+=partials[base+3u];errors|=partials[base+4u];expectedAir+=partials[base+5u];lookupFailures+=partials[base+6u];staleOwners+=partials[base+7u];}sum0[lid]=fineVolume;sum1[lid]=replacedVolume;sum2[lid]=interfaceArea;words0[lid]=samples;words1[lid]=errors;words2[lid]=expectedAir;words3[lid]=lookupFailures;words4[lid]=staleOwners;workgroupBarrier();for(var stride=128u;stride>0u;stride/=2u){if(lid<stride){sum0[lid]+=sum0[lid+stride];sum1[lid]+=sum1[lid+stride];sum2[lid]+=sum2[lid+stride];words0[lid]+=words0[lid+stride];words1[lid]|=words1[lid+stride];words2[lid]+=words2[lid+stride];words3[lid]+=words3[lid+stride];words4[lid]+=words4[lid+stride];}workgroupBarrier();}return FineVolumeReduction(sum0[0],sum1[0],sum2[0],words0[0],words1[0],words2[0],words3[0],words4[0]);}
@compute @workgroup_size(256)fn finalizeFineVolume(@builtin(local_invocation_index)lid:u32){let result=reduceFinePartialHierarchy(lid);if(lid!=0u){return;}control.fineVolume=result.fineVolume;control.replacedCoarseVolume=result.replacedVolume;control.interfaceArea=result.interfaceArea;control.samples=result.samples;control.flags|=result.errors;control.expectedAir=result.expectedAir;control.lookupFailures=result.lookupFailures;control.staleOwners=result.staleOwners;control.currentVolume=control.coarseVolume+result.fineVolume-result.replacedVolume;control.generation=p.generation;if(!finite(control.coarseVolume)||!finite(result.fineVolume)||!finite(result.replacedVolume)||!finite(control.currentVolume)||!finite(result.interfaceArea)){control.flags|=ERROR_FINE;}if(result.samples>0u&&(result.fineVolume<=0.0||result.interfaceArea<=0.0||control.currentVolume<=0.0)){control.flags|=ERROR_FINE;}if(control.initialized==0u&&control.flags==0u&&control.coarseRows>0u){control.referenceVolume=control.currentVolume;control.initialized=1u;}if(control.initialized!=0u&&control.flags==0u&&result.interfaceArea>0.){control.correction=clamp((control.currentVolume-control.referenceVolume)/result.interfaceArea,-.5*p.fineCellWidth,.5*p.fineCellWidth);}if(control.initialized!=0u&&control.flags==0u){control.flags=PUBLISHED;}}
@compute @workgroup_size(64)fn applyFineVolumeCorrection(@builtin(workgroup_id)w:vec3u,@builtin(local_invocation_index)lid:u32,@builtin(num_workgroups)n:vec3u){if(control.flags!=PUBLISHED){return;}let flat=fineLinearWorkgroup(w,n)*64u+lid;if(flat==0u){control.corrected=1u;}let a=activeSample(flat);if(a.x==INVALID){return;}let index=a.x*p.samplesPerBrick+a.y;if((sampleFlags[index]&VALID)==0u){return;}phi[index]+=control.correction;}
fn finalizeCorrectedMeasurement(updateCorrection:bool,lid:u32){let result=reduceFinePartialHierarchy(lid);if(lid!=0u||control.flags!=PUBLISHED){return;}control.fineVolume=result.fineVolume;control.replacedCoarseVolume=result.replacedVolume;control.interfaceArea=result.interfaceArea;control.samples=result.samples;control.expectedAir=result.expectedAir;control.lookupFailures=result.lookupFailures;control.staleOwners=result.staleOwners;control.currentVolume=control.coarseVolume+result.fineVolume-result.replacedVolume;control.generation=p.generation;if(result.errors!=0u||result.lookupFailures!=0u||result.staleOwners!=0u||!finite(result.fineVolume)||!finite(result.replacedVolume)||!finite(result.interfaceArea)||!finite(control.currentVolume)||control.currentVolume<=0.0){control.flags=result.errors|ERROR_FINE;}else{if(updateCorrection&&result.interfaceArea>0.){control.correction=clamp((control.currentVolume-control.referenceVolume)/result.interfaceArea,-.5*p.fineCellWidth,.5*p.fineCellWidth);}control.flags=PUBLISHED;control.corrected=1u;}}
@compute @workgroup_size(256)fn finalizeCorrectedFineVolume(@builtin(local_invocation_index)lid:u32){finalizeCorrectedMeasurement(true,lid);}
@compute @workgroup_size(256)fn finalizeMeasuredFineVolume(@builtin(local_invocation_index)lid:u32){finalizeCorrectedMeasurement(false,lid);}
`;

/** Activity-only correction variant; disabled mode returns the production shader itself. */
export function fineLevelSetVolumeActivityShader(activity: GPULogicalActivityAdoptionContext): string {
  const entry = activity.workgroup("apply-fine-volume-correction", "enter", {
    workgroupId: "w",
    numWorkgroups: "n",
    localInvocationIndex: "lid",
    workgroupLaneCount: 64,
  });
  const exit = activity.workgroup("apply-fine-volume-correction", "exit", {
    workgroupId: "w",
    numWorkgroups: "n",
    localInvocationIndex: "lid",
    workgroupLaneCount: 64,
  });
  if (!entry && !exit) return fineLevelSetVolumeCorrectionWGSL;
  const signature = "fn applyFineVolumeCorrection(";
  const start = fineLevelSetVolumeCorrectionWGSL.indexOf(signature);
  if (start < 0) throw new Error("Fine-volume activity entry point is missing");
  const bodyStart = fineLevelSetVolumeCorrectionWGSL.indexOf("{", start + signature.length);
  let depth = 0, bodyEnd = -1;
  for (let index = bodyStart; index < fineLevelSetVolumeCorrectionWGSL.length; index += 1) {
    if (fineLevelSetVolumeCorrectionWGSL[index] === "{") depth += 1;
    else if (fineLevelSetVolumeCorrectionWGSL[index] === "}" && --depth === 0) { bodyEnd = index; break; }
  }
  if (bodyStart < 0 || bodyEnd < 0) throw new Error("Fine-volume activity body is malformed");
  const body = fineLevelSetVolumeCorrectionWGSL.slice(bodyStart + 1, bodyEnd)
    .replace(/\breturn;/g, `${exit}return;`);
  return `${fineLevelSetVolumeCorrectionWGSL.slice(0, bodyStart + 1)}${entry}${body}${exit}${fineLevelSetVolumeCorrectionWGSL.slice(bodyEnd)}`;
}
