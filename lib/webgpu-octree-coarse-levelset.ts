/** GPU-resident compact coarse level set and deterministic fine-band correction. */

import {
  OCTREE_COARSE_PHI_BYTES,
  OCTREE_COARSE_PHI_FLAG,
  type OctreeCoarsePhiRecord,
  packOctreeCoarsePhiRecords,
} from "./octree-coarse-levelset";
import { OCTREE_SURFACE_STATE } from "./webgpu-octree-surface-pages";

export const OCTREE_FINE_PHI_CONTRIBUTION_BYTES = 16;
export const OCTREE_COARSE_PHI_CONTROL_BYTES = 32;

export const OCTREE_COARSE_PHI_ERROR = Object.freeze({
  capacity: 1 << 0,
  invalidOffsets: 1 << 1,
  invalidSample: 1 << 2,
  nonfinite: 1 << 3,
  contributionBound: 1 << 4,
} as const);

export interface OctreeFinePhiContribution {
  readonly phi: number;
  readonly distanceSquared: number;
  readonly valid?: boolean;
}

export interface OctreeCoarsePhiGPUPlan {
  readonly rowCapacity: number;
  readonly recordBytes: number;
  readonly scratchBytes: number;
  readonly allocatedBytes: number;
}

export interface OctreeCoarsePhiCorrectionInput {
  /** CSR offsets of length rowCount + 1 into contributions. */
  readonly rowOffsets: GPUBuffer;
  readonly contributions: GPUBuffer;
}

export interface OctreeCoarsePhiGPUControl {
  readonly flags: number;
  readonly firstErrorRow: number;
  readonly rowCount: number;
  readonly contributionCount: number;
  readonly correctedRows: number;
  readonly interfaceRows: number;
  readonly generation: number;
}

function positiveInteger(value: number, label: string): number {
  if (!Number.isSafeInteger(value) || value < 1) throw new RangeError(`${label} must be a positive integer`);
  return value;
}

function nonNegativeInteger(value: number, label: string): number {
  if (!Number.isSafeInteger(value) || value < 0 || value > 0xffff_ffff) {
    throw new RangeError(`${label} must be a non-negative u32 integer`);
  }
  return value;
}

export function planOctreeCoarsePhi(rowCapacityValue: number): OctreeCoarsePhiGPUPlan {
  const rowCapacity = positiveInteger(rowCapacityValue, "Coarse phi row capacity");
  const recordBytes = rowCapacity * OCTREE_COARSE_PHI_BYTES;
  return {
    rowCapacity,
    recordBytes,
    scratchBytes: recordBytes,
    allocatedBytes: recordBytes * 2 + OCTREE_COARSE_PHI_CONTROL_BYTES + 32,
  };
}

export function packOctreeFinePhiContributions(samples: readonly OctreeFinePhiContribution[]): ArrayBuffer {
  const buffer = new ArrayBuffer(samples.length * OCTREE_FINE_PHI_CONTRIBUTION_BYTES);
  const floats = new Float32Array(buffer);
  const words = new Uint32Array(buffer);
  samples.forEach((sample, index) => {
    if (!Number.isFinite(sample.phi) || !Number.isFinite(sample.distanceSquared) || sample.distanceSquared < 0) {
      throw new RangeError("Fine phi contributions must contain finite phi and non-negative distance");
    }
    const base = index * 4;
    floats[base] = sample.phi;
    floats[base + 1] = sample.distanceSquared;
    words[base + 2] = sample.valid === false ? 0 : 1;
  });
  return buffer;
}

export function unpackOctreeCoarsePhiGPUControl(words: ArrayLike<number>): OctreeCoarsePhiGPUControl {
  if (words.length < 8) throw new RangeError("Coarse phi control requires eight words");
  return {
    flags: Number(words[0]) >>> 0,
    firstErrorRow: Number(words[1]) >>> 0,
    rowCount: Number(words[2]) >>> 0,
    contributionCount: Number(words[3]) >>> 0,
    correctedRows: Number(words[4]) >>> 0,
    interfaceRows: Number(words[5]) >>> 0,
    generation: Number(words[6]) >>> 0,
  };
}

export class WebGPUOctreeCoarseLevelSet {
  readonly plan: OctreeCoarsePhiGPUPlan;
  readonly records: GPUBuffer;
  readonly control: GPUBuffer;
  private readonly scratch: GPUBuffer;
  private readonly params: GPUBuffer;
  private readonly bootstrapPipeline: GPUComputePipeline;
  private readonly preparePipeline: GPUComputePipeline;
  private readonly correctPipeline: GPUComputePipeline;
  private destroyed = false;

  constructor(private readonly device: GPUDevice, rowCapacity: number) {
    this.plan = planOctreeCoarsePhi(rowCapacity);
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    this.records = device.createBuffer({ label: "Octree coarse phi", size: this.plan.recordBytes, usage: storage });
    this.scratch = device.createBuffer({ label: "Octree coarse phi correction", size: this.plan.scratchBytes, usage: storage });
    this.control = device.createBuffer({ label: "Octree coarse phi control", size: OCTREE_COARSE_PHI_CONTROL_BYTES, usage: storage });
    this.params = device.createBuffer({ label: "Octree coarse phi parameters", size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, mappedAtCreation: true });
    new Uint8Array(this.params.getMappedRange()).fill(0);
    this.params.unmap();
    const shaderModule = device.createShaderModule({ label: "Octree coarse phi correction", code: octreeCoarsePhiCorrectionShader });
    this.bootstrapPipeline = device.createComputePipeline({ label: "Bootstrap compact coarse phi", layout: "auto",
      compute: { module: shaderModule, entryPoint: "bootstrapCoarsePhiFromSurfaceLeaves" } });
    this.preparePipeline = device.createComputePipeline({ label: "Prepare coarse phi correction", layout: "auto",
      compute: { module: shaderModule, entryPoint: "prepareCoarsePhiCorrection" } });
    this.correctPipeline = device.createComputePipeline({ label: "Correct coarse phi from fine band", layout: "auto",
      compute: { module: shaderModule, entryPoint: "correctCoarsePhi" } });
  }

  upload(records: ReadonlyMap<number, OctreeCoarsePhiRecord>): void {
    this.assertLive();
    this.device.queue.writeBuffer(this.records, 0, packOctreeCoarsePhiRecords(this.plan.rowCapacity, records));
  }

  /**
   * One-time migration bootstrap from the compact surface-leaf affine state.
   * The source may itself have been initialized from the legacy texture, but
   * every later coarse/fine query reads only this compact row-scaled field.
   */
  encodeBootstrapFromSurfaceLeaves(encoder: GPUCommandEncoder, leaves: GPUBuffer): void {
    this.assertLive();
    const pass = encoder.beginComputePass({ label: "Bootstrap compact coarse phi from surface leaves" });
    pass.setPipeline(this.bootstrapPipeline);
    pass.setBindGroup(0, this.device.createBindGroup({ layout: this.bootstrapPipeline.getBindGroupLayout(0), entries: [
      { binding: 6, resource: { buffer: leaves } }, { binding: 7, resource: { buffer: this.records } },
    ] }));
    pass.dispatchWorkgroups(Math.ceil(this.plan.rowCapacity / 64));
    pass.end();
  }

  encodeFineCorrection(
    encoder: GPUCommandEncoder,
    input: OctreeCoarsePhiCorrectionInput,
    options: { rowCount: number; contributionCount: number; maximumContributionsPerRow: number; generation?: number },
  ): void {
    this.assertLive();
    const rowCount = nonNegativeInteger(options.rowCount, "Coarse phi row count");
    const contributionCount = nonNegativeInteger(options.contributionCount, "Fine phi contribution count");
    const maximumContributionsPerRow = positiveInteger(options.maximumContributionsPerRow, "Fine contribution row bound");
    const generation = nonNegativeInteger(options.generation ?? 0, "Coarse phi generation");
    const words = new Uint32Array(8);
    words.set([rowCount, contributionCount, this.plan.rowCapacity, maximumContributionsPerRow, generation]);
    this.device.queue.writeBuffer(this.params, 0, words);
    const resource = (buffer: GPUBuffer) => ({ buffer });
    let pass = encoder.beginComputePass({ label: "Prepare coarse phi correction" });
    pass.setPipeline(this.preparePipeline);
    pass.setBindGroup(0, this.device.createBindGroup({ layout: this.preparePipeline.getBindGroupLayout(0), entries: [
      { binding: 0, resource: resource(this.params) }, { binding: 5, resource: resource(this.control) },
    ] }));
    pass.dispatchWorkgroups(1);
    pass.end();
    if (rowCount > 0) {
      pass = encoder.beginComputePass({ label: "Correct coarse phi from fine band" });
      pass.setPipeline(this.correctPipeline);
      pass.setBindGroup(0, this.device.createBindGroup({ layout: this.correctPipeline.getBindGroupLayout(0), entries: [
        { binding: 0, resource: resource(this.params) }, { binding: 1, resource: resource(this.records) },
        { binding: 2, resource: resource(input.rowOffsets) }, { binding: 3, resource: resource(input.contributions) },
        { binding: 4, resource: resource(this.scratch) }, { binding: 5, resource: resource(this.control) },
      ] }));
      pass.dispatchWorkgroups(Math.ceil(Math.min(rowCount, this.plan.rowCapacity) / 64));
      pass.end();
      encoder.copyBufferToBuffer(this.scratch, 0, this.records, 0, Math.min(rowCount, this.plan.rowCapacity) * OCTREE_COARSE_PHI_BYTES);
    }
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.records.destroy();
    this.scratch.destroy();
    this.control.destroy();
    this.params.destroy();
  }

  private assertLive(): void {
    if (this.destroyed) throw new Error("Octree coarse level set is destroyed");
  }
}

export const octreeCoarsePhiCorrectionShader = /* wgsl */ `
struct Params { rowCount:u32, contributionCount:u32, rowCapacity:u32, maximumPerRow:u32, generation:u32, pad0:u32, pad1:u32, pad2:u32 }
struct CoarsePhi { phi:f32, minimumPhi:f32, maximumPhi:f32, flags:u32 }
struct SurfaceLeaf { originX:u32,originY:u32,originZ:u32,size:u32,flags:u32,pad0:u32,pad1:u32,pad2:u32,phiGradient:vec4f,motion:vec4f }
struct FineContribution { phi:f32, distanceSquared:f32, valid:u32, pad:u32 }
struct Control { flags:atomic<u32>, firstErrorRow:atomic<u32>, rowCount:u32, contributionCount:u32, correctedRows:atomic<u32>, interfaceRows:atomic<u32>, generation:u32, reserved:u32 }
@group(0) @binding(0) var<uniform> params:Params;
@group(0) @binding(1) var<storage,read> coarse:array<CoarsePhi>;
@group(0) @binding(2) var<storage,read> offsets:array<u32>;
@group(0) @binding(3) var<storage,read> fine:array<FineContribution>;
@group(0) @binding(4) var<storage,read_write> corrected:array<CoarsePhi>;
@group(0) @binding(5) var<storage,read_write> control:Control;
@group(0) @binding(6) var<storage,read> surfaceLeaves:array<SurfaceLeaf>;
@group(0) @binding(7) var<storage,read_write> bootstrapCoarse:array<CoarsePhi>;
const VALID:u32=${OCTREE_COARSE_PHI_FLAG.valid}u;
const CORRECTED:u32=${OCTREE_COARSE_PHI_FLAG.correctedFromFine}u;
const INTERFACE:u32=${OCTREE_COARSE_PHI_FLAG.containsInterface}u;
const FINITE:u32=${OCTREE_COARSE_PHI_FLAG.finite}u;
const CAPACITY:u32=${OCTREE_COARSE_PHI_ERROR.capacity}u;
const INVALID_OFFSETS:u32=${OCTREE_COARSE_PHI_ERROR.invalidOffsets}u;
const INVALID_SAMPLE:u32=${OCTREE_COARSE_PHI_ERROR.invalidSample}u;
const NONFINITE:u32=${OCTREE_COARSE_PHI_ERROR.nonfinite}u;
const CONTRIBUTION_BOUND:u32=${OCTREE_COARSE_PHI_ERROR.contributionBound}u;
fn isFinite(value:f32)->bool{return (bitcast<u32>(value)&0x7f800000u)!=0x7f800000u;}
fn fail(row:u32,code:u32){atomicOr(&control.flags,code);atomicMin(&control.firstErrorRow,row);}
@compute @workgroup_size(64) fn bootstrapCoarsePhiFromSurfaceLeaves(@builtin(global_invocation_id) gid:vec3u){
  let row=gid.x;if(row>=arrayLength(&bootstrapCoarse)){return;}
  if(row>=arrayLength(&surfaceLeaves)){bootstrapCoarse[row]=CoarsePhi(0.0,0.0,0.0,0u);return;}
  let leaf=surfaceLeaves[row];let value=leaf.phiGradient.x;
  if((leaf.flags&${OCTREE_SURFACE_STATE.live}u)==0u||leaf.size==0u||!isFinite(value)||!all(vec3<bool>(isFinite(leaf.phiGradient.y),isFinite(leaf.phiGradient.z),isFinite(leaf.phiGradient.w)))){
    bootstrapCoarse[row]=CoarsePhi(0.0,0.0,0.0,0u);return;
  }
  let extent=0.5*f32(leaf.size)*dot(abs(leaf.phiGradient.yzw),vec3f(1.0));let minimum=value-extent;let maximum=value+extent;
  var flags=VALID|FINITE;if(minimum<=0.0&&maximum>=0.0){flags|=INTERFACE;}
  bootstrapCoarse[row]=CoarsePhi(value,minimum,maximum,flags);
}
@compute @workgroup_size(1) fn prepareCoarsePhiCorrection(){
  atomicStore(&control.flags,select(0u,CAPACITY,params.rowCount>params.rowCapacity));
  atomicStore(&control.firstErrorRow,0xffffffffu);control.rowCount=params.rowCount;control.contributionCount=params.contributionCount;
  atomicStore(&control.correctedRows,0u);atomicStore(&control.interfaceRows,0u);control.generation=params.generation;control.reserved=0u;
}
@compute @workgroup_size(64) fn correctCoarsePhi(@builtin(global_invocation_id) gid:vec3u){
  let row=gid.x;if(row>=params.rowCount||row>=params.rowCapacity){return;}
  if(row>=arrayLength(&coarse)||row>=arrayLength(&corrected)||row+1u>=arrayLength(&offsets)){
    fail(row,CAPACITY);return;
  }
  let begin=offsets[row];let end=offsets[row+1u];
  if(end<begin||end>params.contributionCount||end>arrayLength(&fine)){fail(row,INVALID_OFFSETS);corrected[row]=coarse[row];return;}
  if(end-begin>params.maximumPerRow){fail(row,CONTRIBUTION_BOUND);corrected[row]=coarse[row];return;}
  var output=coarse[row];var count=0u;var nearestDistance=3.402823466e+38;var minimumPhi=3.402823466e+38;var maximumPhi=-3.402823466e+38;
  for(var index=begin;index<end;index+=1u){let sample=fine[index];if(sample.valid==0u){continue;}
    if(!isFinite(sample.phi)||!isFinite(sample.distanceSquared)||sample.distanceSquared<0.0){fail(row,select(NONFINITE,INVALID_SAMPLE,sample.distanceSquared<0.0));continue;}
    minimumPhi=min(minimumPhi,sample.phi);maximumPhi=max(maximumPhi,sample.phi);
    if(sample.distanceSquared<nearestDistance||(sample.distanceSquared==nearestDistance&&sample.phi<output.phi)){
      nearestDistance=sample.distanceSquared;output.phi=sample.phi;
    }count+=1u;
  }
  output.flags=output.flags|VALID|FINITE;
  if(count>0u){output.minimumPhi=minimumPhi;output.maximumPhi=maximumPhi;output.flags=output.flags|CORRECTED;atomicAdd(&control.correctedRows,1u);}
  if(output.minimumPhi<=0.0&&output.maximumPhi>=0.0){output.flags=output.flags|INTERFACE;atomicAdd(&control.interfaceRows,1u);}else{output.flags=output.flags&(~INTERFACE);}
  corrected[row]=output;
}`;
