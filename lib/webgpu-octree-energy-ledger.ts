import type { WebGPUFineLevelSetBrickSource } from "./webgpu-octree-fine-levelset-bricks";

export const OCTREE_ENERGY_LEDGER_RECORD_BYTES = 32;
export const OCTREE_ENERGY_LEDGER_VALID = 0x8000_0000;

export const OCTREE_ENERGY_LEDGER_STAGES = [
  "oldFaceCapture",
  "postRemap",
  "postGravity",
  "postSolidConstraint",
  "postProjection",
  "postFaceBandPublication",
  "preFineTransport",
  "postFineTransport",
  "preFineTopologyCommon",
  "postFineTopology",
  "postFineTopologyCommon",
  "postFineRedistance",
  "postFineRedistanceCommon",
  "postFineVolumeCorrection",
] as const;

export type OctreeEnergyLedgerStage = typeof OCTREE_ENERGY_LEDGER_STAGES[number];

export interface OctreeEnergyLedgerRecord {
  readonly step: number;
  readonly generation: number;
  readonly stage: OctreeEnergyLedgerStage;
  /** `commonFinePotential` records in one step share the immutable logical
   * old/new topology intersection; equal sample counts and zero invalids make
   * their value/volume deltas directly attributable to the named operation. */
  readonly kind: "faceMetricKinetic" | "residentFinePotential" | "commonFinePotential";
  readonly value: number;
  readonly representedVolume_m3: number;
  readonly sampleCount: number;
  readonly invalidCount: number;
}

export interface OctreeEnergyLedgerSnapshot {
  readonly totalSteps: number;
  readonly stepCapacity: number;
  readonly records: readonly OctreeEnergyLedgerRecord[];
}

export interface OctreeEnergyLedgerPlan {
  readonly stepCapacity: number;
  readonly fineSampleCapacity: number;
  readonly recordCount: number;
  readonly recordBytes: number;
  readonly facePartialCount: number;
  readonly finePartialCount: number;
  readonly scratchBytes: number;
  /** Three sample-capacity u32/f32 channels: logical key, local/target index, captured phi. */
  readonly commonSupportBytes: number;
  readonly parameterStride: number;
  readonly parameterBytes: number;
  readonly allocatedBytes: number;
}

function align(value: number, alignment: number): number {
  return Math.ceil(value / alignment) * alignment;
}

export function planOctreeEnergyLedger(
  faceCapacity: number,
  fineSampleCapacity: number,
  stepCapacity = 512,
  minimumUniformAlignment = 256,
): OctreeEnergyLedgerPlan {
  for (const [value, label] of [[faceCapacity, "face"], [fineSampleCapacity, "fine"],
    [stepCapacity, "step"], [minimumUniformAlignment, "uniform alignment"]] as const) {
    if (!Number.isSafeInteger(value) || value < 1) throw new RangeError(`Energy-ledger ${label} capacity must be positive`);
  }
  const recordCount = stepCapacity * OCTREE_ENERGY_LEDGER_STAGES.length;
  const recordBytes = recordCount * OCTREE_ENERGY_LEDGER_RECORD_BYTES;
  const facePartialCount = Math.ceil(faceCapacity / 64);
  const finePartialCount = Math.ceil(fineSampleCapacity / 64);
  const scratchBytes = Math.max(facePartialCount, finePartialCount) * 16;
  const commonSupportBytes = fineSampleCapacity * 12;
  const parameterStride = align(64, minimumUniformAlignment);
  const parameterBytes = parameterStride * OCTREE_ENERGY_LEDGER_STAGES.length;
  return { stepCapacity, fineSampleCapacity, recordCount, recordBytes, facePartialCount, finePartialCount,
    scratchBytes, commonSupportBytes, parameterStride, parameterBytes,
    allocatedBytes: 16 + recordBytes + scratchBytes + commonSupportBytes + parameterBytes };
}

export function decodeOctreeEnergyLedger(
  controlBytes: ArrayBufferLike,
  recordBytes: ArrayBufferLike,
): OctreeEnergyLedgerSnapshot {
  if (controlBytes.byteLength < 16) throw new RangeError("Energy-ledger control needs 16 bytes");
  const control = new Uint32Array(controlBytes, 0, 4);
  const stepCapacity = control[2];
  if (control[0] !== 1 || stepCapacity < 1 || control[3] !== OCTREE_ENERGY_LEDGER_STAGES.length
    || recordBytes.byteLength < stepCapacity * OCTREE_ENERGY_LEDGER_STAGES.length * OCTREE_ENERGY_LEDGER_RECORD_BYTES) {
    throw new RangeError("Energy-ledger publication header is incompatible");
  }
  const words = new Uint32Array(recordBytes);
  const floats = new Float32Array(recordBytes);
  const totalSteps = control[1];
  const firstRetainedStep = Math.max(0, totalSteps - stepCapacity);
  const records: OctreeEnergyLedgerRecord[] = [];
  for (let index = 0; index < stepCapacity * OCTREE_ENERGY_LEDGER_STAGES.length; index += 1) {
    const base = index * 8, step = words[base], stageIndex = words[base + 2], flags = words[base + 3];
    if ((flags & OCTREE_ENERGY_LEDGER_VALID) === 0 || step < firstRetainedStep || step >= totalSteps
      || stageIndex >= OCTREE_ENERGY_LEDGER_STAGES.length) continue;
    records.push({
      step,
      generation: words[base + 1],
      stage: OCTREE_ENERGY_LEDGER_STAGES[stageIndex],
      kind: (flags & 3) === 2 ? "commonFinePotential"
        : (flags & 1) !== 0 ? "residentFinePotential" : "faceMetricKinetic",
      value: floats[base + 4],
      representedVolume_m3: floats[base + 5],
      sampleCount: words[base + 6],
      invalidCount: words[base + 7],
    });
  }
  records.sort((a, b) => a.step - b.step
    || OCTREE_ENERGY_LEDGER_STAGES.indexOf(a.stage) - OCTREE_ENERGY_LEDGER_STAGES.indexOf(b.stage));
  return { totalSteps, stepCapacity, records };
}

interface FaceSource {
  readonly faces: GPUBuffer;
  readonly control: GPUBuffer;
}

/**
 * Opt-in observational reductions for the authoritative compact power path.
 * The ledger never feeds a simulation publication or gate. Its fixed GPU ring
 * is read only on explicit request, so an enabled 500-step run has no recurring
 * host synchronization.
 */
export class WebGPUOctreeEnergyLedger {
  readonly plan: OctreeEnergyLedgerPlan;
  readonly records: GPUBuffer;
  readonly control: GPUBuffer;
  private readonly scratch: GPUBuffer;
  private readonly commonKeys: GPUBuffer;
  private readonly commonLocalOrTarget: GPUBuffer;
  private readonly commonPrePhi: GPUBuffer;
  private readonly parameters: GPUBuffer;
  private readonly facePartialPipeline: GPUComputePipeline;
  private readonly faceFinalizePipeline: GPUComputePipeline;
  private readonly finePartialPipeline: GPUComputePipeline;
  private readonly fineFinalizePipeline: GPUComputePipeline;
  private readonly captureCommonPipeline: GPUComputePipeline;
  private readonly buildCommonPipeline: GPUComputePipeline;
  private readonly commonPartialPipeline: GPUComputePipeline;
  private currentStep = -1;
  private currentSlot = 0;
  private destroyed = false;

  constructor(
    private readonly device: GPUDevice,
    faceCapacity: number,
    fineSampleCapacity: number,
    private readonly gravity: readonly [number, number, number],
    stepCapacity = 512,
  ) {
    this.plan = planOctreeEnergyLedger(faceCapacity, fineSampleCapacity, stepCapacity,
      device.limits.minUniformBufferOffsetAlignment);
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    this.control = device.createBuffer({ label: "Power energy ledger control", size: 16, usage: storage });
    this.records = device.createBuffer({ label: "Power energy ledger records", size: this.plan.recordBytes, usage: storage });
    this.scratch = device.createBuffer({ label: "Power energy ledger reduction scratch", size: this.plan.scratchBytes,
      usage: GPUBufferUsage.STORAGE });
    const commonChannelBytes = fineSampleCapacity * 4;
    this.commonKeys = device.createBuffer({ label: "Power energy ledger common-support keys",
      size: commonChannelBytes, usage: GPUBufferUsage.STORAGE });
    this.commonLocalOrTarget = device.createBuffer({ label: "Power energy ledger common-support indices",
      size: commonChannelBytes, usage: GPUBufferUsage.STORAGE });
    this.commonPrePhi = device.createBuffer({ label: "Power energy ledger common-support pre-topology phi",
      size: commonChannelBytes, usage: GPUBufferUsage.STORAGE });
    this.parameters = device.createBuffer({ label: "Power energy ledger stage parameters", size: this.plan.parameterBytes,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(this.control, 0, new Uint32Array([1, 0, this.plan.stepCapacity,
      OCTREE_ENERGY_LEDGER_STAGES.length]));
    const module = device.createShaderModule({ label: "Power energy ledger reductions", code: octreeEnergyLedgerWGSL });
    const pipeline = (entryPoint: string) => device.createComputePipeline({ label: `Power energy ledger · ${entryPoint}`,
      layout: "auto", compute: { module, entryPoint } });
    this.facePartialPipeline = pipeline("reduceFaceMetricPartials");
    this.faceFinalizePipeline = pipeline("finalizeFaceMetric");
    this.finePartialPipeline = pipeline("reduceFinePotentialPartials");
    this.fineFinalizePipeline = pipeline("finalizeFinePotential");
    this.captureCommonPipeline = pipeline("captureFineCommonCandidates");
    this.buildCommonPipeline = pipeline("buildFineCommonSupport");
    this.commonPartialPipeline = pipeline("reduceFineCommonPotentialPartials");
  }

  beginStep(): void {
    this.currentStep += 1;
    this.currentSlot = this.currentStep % this.plan.stepCapacity;
  }

  private parameterBinding(stage: OctreeEnergyLedgerStage, generation: number, partialCount: number,
    itemCapacity: number, kind: 0 | 1 | 2): GPUBufferBinding {
    if (this.currentStep < 0) throw new Error("Energy-ledger step must begin before a stage is encoded");
    const stageIndex = OCTREE_ENERGY_LEDGER_STAGES.indexOf(stage);
    const bytes = new ArrayBuffer(64), words = new Uint32Array(bytes), floats = new Float32Array(bytes);
    words[0] = this.currentStep; words[1] = this.currentSlot; words[2] = stageIndex; words[3] = generation;
    words[4] = this.plan.stepCapacity; words[5] = partialCount; words[6] = itemCapacity; words[7] = kind;
    floats[8] = this.gravity[0]; floats[9] = this.gravity[1]; floats[10] = this.gravity[2];
    const offset = stageIndex * this.plan.parameterStride;
    this.device.queue.writeBuffer(this.parameters, offset, bytes);
    return { buffer: this.parameters, offset, size: 64 };
  }

  encodeFaceMetric(encoder: GPUCommandEncoder, stage: Extract<OctreeEnergyLedgerStage,
    "oldFaceCapture" | "postRemap" | "postGravity" | "postSolidConstraint" | "postProjection" | "postFaceBandPublication">,
  generation: number, source: FaceSource): void {
    if (this.destroyed) throw new Error("Energy ledger is destroyed");
    const params = this.parameterBinding(stage, generation, this.plan.facePartialCount,
      Math.floor(source.faces.size / 32), 0);
    const partialGroup = this.device.createBindGroup({ layout: this.facePartialPipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: params }, { binding: 3, resource: { buffer: this.scratch } },
        { binding: 4, resource: { buffer: source.faces } }, { binding: 5, resource: { buffer: source.control } }] });
    const finalizeGroup = this.device.createBindGroup({ layout: this.faceFinalizePipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: params }, { binding: 1, resource: { buffer: this.records } },
        { binding: 2, resource: { buffer: this.control } }, { binding: 3, resource: { buffer: this.scratch } }] });
    let pass = encoder.beginComputePass({ label: `Power energy ledger · ${stage} partials` });
    pass.setPipeline(this.facePartialPipeline); pass.setBindGroup(0, partialGroup);
    pass.dispatchWorkgroups(this.plan.facePartialCount); pass.end();
    pass = encoder.beginComputePass({ label: `Power energy ledger · ${stage} finalize` });
    pass.setPipeline(this.faceFinalizePipeline); pass.setBindGroup(0, finalizeGroup);
    pass.dispatchWorkgroups(1); pass.end();
  }

  encodeFinePotential(encoder: GPUCommandEncoder, stage: Extract<OctreeEnergyLedgerStage,
    "preFineTransport" | "postFineTransport" | "postFineTopology" | "postFineRedistance" | "postFineVolumeCorrection">,
  source: WebGPUFineLevelSetBrickSource): void {
    if (this.destroyed) throw new Error("Energy ledger is destroyed");
    const itemCapacity = source.plan.maximumResidentBricks * source.plan.samplesPerBrick;
    const partialCount = Math.ceil(itemCapacity / 64);
    const params = this.parameterBinding(stage, source.generation, partialCount, itemCapacity, 1);
    const partialGroup = this.device.createBindGroup({ layout: this.finePartialPipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: params }, { binding: 3, resource: { buffer: this.scratch } },
        { binding: 6, resource: { buffer: source.params } }, { binding: 7, resource: { buffer: source.metadata } },
        { binding: 8, resource: { buffer: source.worklist } }, { binding: 9, resource: { buffer: source.flags } },
        { binding: 10, resource: { buffer: source.phi } }] });
    const finalizeGroup = this.device.createBindGroup({ layout: this.fineFinalizePipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: params }, { binding: 1, resource: { buffer: this.records } },
        { binding: 2, resource: { buffer: this.control } }, { binding: 3, resource: { buffer: this.scratch } }] });
    let pass = encoder.beginComputePass({ label: `Power energy ledger · ${stage} partials` });
    pass.setPipeline(this.finePartialPipeline); pass.setBindGroup(0, partialGroup);
    pass.dispatchWorkgroups(partialCount); pass.end();
    pass = encoder.beginComputePass({ label: `Power energy ledger · ${stage} finalize` });
    pass.setPipeline(this.fineFinalizePipeline); pass.setBindGroup(0, finalizeGroup);
    pass.dispatchWorkgroups(1); pass.end();
  }

  /** Capture the transported old field before topology can reuse shared payload pages. */
  encodeFineCommonCapture(encoder: GPUCommandEncoder, source: WebGPUFineLevelSetBrickSource): void {
    if (this.destroyed) throw new Error("Energy ledger is destroyed");
    const itemCapacity = source.plan.maximumResidentBricks * source.plan.samplesPerBrick;
    if (itemCapacity !== this.plan.fineSampleCapacity) {
      throw new RangeError("Energy-ledger common support requires the planned fine sample capacity");
    }
    const group = this.device.createBindGroup({ layout: this.captureCommonPipeline.getBindGroupLayout(0), entries: [
      { binding: 6, resource: { buffer: source.params } },
      { binding: 7, resource: { buffer: source.metadata } },
      { binding: 8, resource: { buffer: source.worklist } },
      { binding: 9, resource: { buffer: source.flags } },
      { binding: 10, resource: { buffer: source.phi } },
      { binding: 11, resource: { buffer: this.commonKeys } },
      { binding: 12, resource: { buffer: this.commonLocalOrTarget } },
      { binding: 13, resource: { buffer: this.commonPrePhi } },
    ] });
    const pass = encoder.beginComputePass({ label: "Power energy ledger · capture pre-topology common candidates" });
    pass.setPipeline(this.captureCommonPipeline); pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(this.plan.finePartialCount); pass.end();
  }

  /**
   * Freeze the exact coordinate intersection after topology publication and
   * emit both sides of the topology comparison from that one support.
   */
  encodeFineCommonTopologyPair(encoder: GPUCommandEncoder, previous: WebGPUFineLevelSetBrickSource,
    target: WebGPUFineLevelSetBrickSource): void {
    if (this.destroyed) throw new Error("Energy ledger is destroyed");
    const itemCapacity = target.plan.maximumResidentBricks * target.plan.samplesPerBrick;
    if (itemCapacity !== this.plan.fineSampleCapacity) {
      throw new RangeError("Energy-ledger common support requires the planned fine sample capacity");
    }
    const group = this.device.createBindGroup({ layout: this.buildCommonPipeline.getBindGroupLayout(0), entries: [
      { binding: 6, resource: { buffer: target.params } },
      { binding: 7, resource: { buffer: target.metadata } },
      { binding: 9, resource: { buffer: target.flags } },
      { binding: 10, resource: { buffer: target.phi } },
      { binding: 11, resource: { buffer: this.commonKeys } },
      { binding: 12, resource: { buffer: this.commonLocalOrTarget } },
      { binding: 14, resource: { buffer: target.hash } },
    ] });
    const pass = encoder.beginComputePass({ label: "Power energy ledger · freeze topology common support" });
    pass.setPipeline(this.buildCommonPipeline); pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(this.plan.finePartialCount); pass.end();
    this.encodeFineCommonPotential(encoder, "preFineTopologyCommon", previous.generation, target);
    this.encodeFineCommonPotential(encoder, "postFineTopologyCommon", target.generation, target);
  }

  encodeFineCommonPotential(encoder: GPUCommandEncoder, stage: Extract<OctreeEnergyLedgerStage,
    "preFineTopologyCommon" | "postFineTopologyCommon" | "postFineRedistanceCommon">,
  generation: number, target: WebGPUFineLevelSetBrickSource): void {
    if (this.destroyed) throw new Error("Energy ledger is destroyed");
    const params = this.parameterBinding(stage, generation, this.plan.finePartialCount,
      this.plan.fineSampleCapacity, 2);
    const partialGroup = this.device.createBindGroup({ layout: this.commonPartialPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: params }, { binding: 3, resource: { buffer: this.scratch } },
        { binding: 6, resource: { buffer: target.params } },
        { binding: 7, resource: { buffer: target.metadata } },
        { binding: 9, resource: { buffer: target.flags } },
        { binding: 10, resource: { buffer: target.phi } },
        { binding: 12, resource: { buffer: this.commonLocalOrTarget } },
        { binding: 13, resource: { buffer: this.commonPrePhi } },
      ] });
    const finalizeGroup = this.device.createBindGroup({ layout: this.fineFinalizePipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: params }, { binding: 1, resource: { buffer: this.records } },
        { binding: 2, resource: { buffer: this.control } }, { binding: 3, resource: { buffer: this.scratch } }] });
    let pass = encoder.beginComputePass({ label: `Power energy ledger · ${stage} partials` });
    pass.setPipeline(this.commonPartialPipeline); pass.setBindGroup(0, partialGroup);
    pass.dispatchWorkgroups(this.plan.finePartialCount); pass.end();
    pass = encoder.beginComputePass({ label: `Power energy ledger · ${stage} finalize` });
    pass.setPipeline(this.fineFinalizePipeline); pass.setBindGroup(0, finalizeGroup);
    pass.dispatchWorkgroups(1); pass.end();
  }

  async read(): Promise<OctreeEnergyLedgerSnapshot> {
    if (this.destroyed) throw new Error("Energy ledger is destroyed");
    const readback = this.device.createBuffer({ label: "Read power energy ledger", size: 16 + this.plan.recordBytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = this.device.createCommandEncoder({ label: "Read power energy ledger" });
    encoder.copyBufferToBuffer(this.control, 0, readback, 0, 16);
    encoder.copyBufferToBuffer(this.records, 0, readback, 16, this.plan.recordBytes);
    this.device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const snapshot = readback.getMappedRange().slice(0);
    readback.unmap(); readback.destroy();
    return decodeOctreeEnergyLedger(snapshot.slice(0, 16), snapshot.slice(16));
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.control.destroy(); this.records.destroy(); this.scratch.destroy(); this.parameters.destroy();
    this.commonKeys.destroy(); this.commonLocalOrTarget.destroy(); this.commonPrePhi.destroy();
  }
}

export const octreeEnergyLedgerWGSL = /* wgsl */ `
struct Params{step:u32,slot:u32,stage:u32,generation:u32,stepCapacity:u32,partialCount:u32,itemCapacity:u32,kind:u32,gravity:vec3f,pad:f32}
struct Record{step:u32,generation:u32,stage:u32,flags:u32,value:f32,volume:f32,samples:u32,invalid:u32}
struct PowerFace{negativeRow:u32,positiveRow:u32,geometryCode:u32,flags:u32,normalVelocity:f32,area:f32,inverseDistance:f32,openFraction:f32}
struct FineParams{brickDimensions:vec3u,brickResolution:u32,sampleDimensions:vec3u,samplesPerBrick:u32,domainOrigin:vec3f,fineCellWidth:f32,hashCapacity:u32,maximumHashProbes:u32,pageCapacity:u32,generation:u32,activeCount:u32,invalid:u32,fineFactor:u32,timestep:f32}
@group(0)@binding(0)var<uniform>p:Params;
@group(0)@binding(1)var<storage,read_write>records:array<Record>;
@group(0)@binding(2)var<storage,read_write>control:array<atomic<u32>>;
@group(0)@binding(3)var<storage,read_write>partials:array<vec4u>;
@group(0)@binding(4)var<storage,read>faces:array<PowerFace>;
@group(0)@binding(5)var<storage,read>faceControl:array<u32>;
@group(0)@binding(6)var<uniform>fp:FineParams;
@group(0)@binding(7)var<storage,read>metadata:array<u32>;
@group(0)@binding(8)var<storage,read>worklist:array<u32>;
@group(0)@binding(9)var<storage,read>sampleFlags:array<u32>;
@group(0)@binding(10)var<storage,read>phi:array<f32>;
@group(0)@binding(11)var<storage,read_write>commonKeys:array<u32>;
@group(0)@binding(12)var<storage,read_write>commonLocalOrTarget:array<u32>;
@group(0)@binding(13)var<storage,read_write>commonPrePhi:array<f32>;
@group(0)@binding(14)var<storage,read>targetHash:array<u32>;
var<workgroup>sum0:array<f32,64>;var<workgroup>sum1:array<f32,64>;
var<workgroup>count0:array<u32,64>;var<workgroup>count1:array<u32,64>;
const VALID:u32=0x80000000u;
const INVALID:u32=0xffffffffu;
fn finite(v:f32)->bool{return v==v&&abs(v)<3.402823e38;}
fn recordIndex()->u32{return (p.slot*${OCTREE_ENERGY_LEDGER_STAGES.length}u)+p.stage;}
fn reduceLocal(local:u32){workgroupBarrier();for(var stride=32u;stride>0u;stride/=2u){if(local<stride){sum0[local]+=sum0[local+stride];sum1[local]+=sum1[local+stride];count0[local]+=count0[local+stride];count1[local]+=count1[local+stride];}workgroupBarrier();}}
@compute @workgroup_size(64)fn reduceFaceMetricPartials(@builtin(local_invocation_id)lid:vec3u,@builtin(workgroup_id)wid:vec3u){
 let index=wid.x*64u+lid.x;let faceCount=select(0u,min(faceControl[1],p.itemCapacity),arrayLength(&faceControl)>1u);var energy=0.;var samples=0u;var invalid=0u;
 if(index<faceCount&&index<arrayLength(&faces)){let f=faces[index];if(!finite(f.normalVelocity)||!finite(f.area)||!finite(f.inverseDistance)||!finite(f.openFraction)||f.area<=0.||f.inverseDistance<=0.||f.openFraction<0.||f.openFraction>1.){invalid=1u;}else{if(f.openFraction>0.){energy=.5*f.area/(f.openFraction*f.inverseDistance)*f.normalVelocity*f.normalVelocity;}samples=1u;}}
 sum0[lid.x]=energy;sum1[lid.x]=0.;count0[lid.x]=samples;count1[lid.x]=invalid;reduceLocal(lid.x);if(lid.x==0u){partials[wid.x]=vec4u(bitcast<u32>(sum0[0]),0u,count0[0],count1[0]);}}
@compute @workgroup_size(1)fn finalizeFaceMetric(){var energy=0.;var samples=0u;var invalid=0u;for(var i=0u;i<p.partialCount;i+=1u){let q=partials[i];energy+=bitcast<f32>(q.x);samples+=q.z;invalid+=q.w;}let flags=VALID;records[recordIndex()]=Record(p.step,p.generation,p.stage,flags,energy,0.,samples,invalid);atomicMax(&control[1],p.step+1u);}
fn activeSample(flat:u32)->vec2u{let liveCount=min(worklist[0],fp.pageCapacity);if(flat>=liveCount*fp.samplesPerBrick){return vec2u(0xffffffffu);}let w=flat/fp.samplesPerBrick;let local=flat-w*fp.samplesPerBrick;let id=worklist[5u+w];if(id>=fp.pageCapacity||metadata[id*10u+2u]!=fp.generation){return vec2u(0xffffffffu);}return vec2u(id,local);}
fn unpackBrick(key:u32)->vec3u{let xy=fp.brickDimensions.x*fp.brickDimensions.y;let z=key/xy;let r=key-z*xy;let y=r/fp.brickDimensions.x;return vec3u(r-y*fp.brickDimensions.x,y,z);}
fn localCoord(local:u32)->vec3u{let r=fp.brickResolution;let z=local/(r*r);let q=local-z*r*r;let y=q/r;return vec3u(q-y*r,y,z);}
@compute @workgroup_size(64)fn reduceFinePotentialPartials(@builtin(local_invocation_id)lid:vec3u,@builtin(workgroup_id)wid:vec3u){
 let flat=wid.x*64u+lid.x;var energy=0.;var volume=0.;var samples=0u;var invalid=0u;let liveSampleCount=min(worklist[0],fp.pageCapacity)*fp.samplesPerBrick;
 if(flat<p.itemCapacity&&flat<liveSampleCount){let a=activeSample(flat);if(a.x==0xffffffffu){invalid=1u;}else{let index=a.x*fp.samplesPerBrick+a.y;let brick=unpackBrick(metadata[a.x*10u+1u]);let q=brick*fp.brickResolution+localCoord(a.y);if(all(q<fp.sampleDimensions)){if(index>=arrayLength(&phi)||index>=arrayLength(&sampleFlags)||(sampleFlags[index]&1u)==0u){invalid=1u;}else{let value=phi[index];if(!finite(value)){invalid=1u;}else{let h=fp.fineCellWidth;let alpha=clamp(.5-value/h,0.,1.);let cellVolume=h*h*h;let position=fp.domainOrigin+(vec3f(q)+.5)*h;volume=alpha*cellVolume;energy=-dot(p.gravity,position)*volume;samples=1u;}}}}}
 sum0[lid.x]=energy;sum1[lid.x]=volume;count0[lid.x]=samples;count1[lid.x]=invalid;reduceLocal(lid.x);if(lid.x==0u){partials[wid.x]=vec4u(bitcast<u32>(sum0[0]),bitcast<u32>(sum1[0]),count0[0],count1[0]);}}
@compute @workgroup_size(64)fn captureFineCommonCandidates(@builtin(global_invocation_id)g:vec3u){let flat=g.x;if(flat>=arrayLength(&commonKeys)||flat>=arrayLength(&commonLocalOrTarget)||flat>=arrayLength(&commonPrePhi)){return;}commonKeys[flat]=INVALID;commonLocalOrTarget[flat]=INVALID;commonPrePhi[flat]=0.;let a=activeSample(flat);if(a.x==INVALID){return;}let index=a.x*fp.samplesPerBrick+a.y;if(index>=arrayLength(&phi)||index>=arrayLength(&sampleFlags)||(sampleFlags[index]&1u)==0u){return;}let value=phi[index];if(!finite(value)){return;}let key=metadata[a.x*10u+1u];let q=unpackBrick(key)*fp.brickResolution+localCoord(a.y);if(any(q>=fp.sampleDimensions)){return;}commonKeys[flat]=key;commonLocalOrTarget[flat]=a.y;commonPrePhi[flat]=value;}
fn lookupTargetBrick(key:u32)->u32{if(fp.hashCapacity==0u){return INVALID;}let start=((key^(key>>16u))*0x9e3779b1u)&(fp.hashCapacity-1u);for(var probe=0u;probe<32u;probe+=1u){if(probe>=fp.maximumHashProbes){break;}let slot=(start+probe)&(fp.hashCapacity-1u);if(slot*2u+1u>=arrayLength(&targetHash)){return INVALID;}let stored=targetHash[slot*2u];if(stored==key){let id=targetHash[slot*2u+1u];if(id>=fp.pageCapacity||id*10u+2u>=arrayLength(&metadata)||metadata[id*10u]!=id||metadata[id*10u+1u]!=key||metadata[id*10u+2u]!=fp.generation){return INVALID;}return id;}if(stored==INVALID){return INVALID;}}return INVALID;}
@compute @workgroup_size(64)fn buildFineCommonSupport(@builtin(global_invocation_id)g:vec3u){let flat=g.x;if(flat>=arrayLength(&commonKeys)||flat>=arrayLength(&commonLocalOrTarget)){return;}let key=commonKeys[flat];let local=commonLocalOrTarget[flat];commonLocalOrTarget[flat]=INVALID;if(key==INVALID||local>=fp.samplesPerBrick){return;}let id=lookupTargetBrick(key);if(id==INVALID){return;}let index=id*fp.samplesPerBrick+local;if(index>=arrayLength(&phi)||index>=arrayLength(&sampleFlags)||(sampleFlags[index]&1u)==0u||!finite(phi[index])){return;}commonLocalOrTarget[flat]=index;}
@compute @workgroup_size(64)fn reduceFineCommonPotentialPartials(@builtin(local_invocation_id)lid:vec3u,@builtin(workgroup_id)wid:vec3u){let flat=wid.x*64u+lid.x;var energy=0.;var volume=0.;var samples=0u;var invalid=0u;if(flat<p.itemCapacity&&flat<arrayLength(&commonLocalOrTarget)&&flat<arrayLength(&commonPrePhi)){let index=commonLocalOrTarget[flat];if(index!=INVALID){if(index>=arrayLength(&phi)||index>=arrayLength(&sampleFlags)){invalid=1u;}else{let id=index/fp.samplesPerBrick;let local=index-id*fp.samplesPerBrick;if(id>=fp.pageCapacity||id*10u+2u>=arrayLength(&metadata)||metadata[id*10u+2u]!=fp.generation){invalid=1u;}else{var value=commonPrePhi[flat];if(p.stage!=${OCTREE_ENERGY_LEDGER_STAGES.indexOf("preFineTopologyCommon")}u){if((sampleFlags[index]&1u)==0u){invalid=1u;}else{value=phi[index];}}if(invalid==0u&&!finite(value)){invalid=1u;}if(invalid==0u){let brick=unpackBrick(metadata[id*10u+1u]);let q=brick*fp.brickResolution+localCoord(local);if(any(q>=fp.sampleDimensions)){invalid=1u;}else{let h=fp.fineCellWidth;let alpha=clamp(.5-value/h,0.,1.);let cellVolume=h*h*h;let position=fp.domainOrigin+(vec3f(q)+.5)*h;volume=alpha*cellVolume;energy=-dot(p.gravity,position)*volume;samples=1u;}}}}}}sum0[lid.x]=energy;sum1[lid.x]=volume;count0[lid.x]=samples;count1[lid.x]=invalid;reduceLocal(lid.x);if(lid.x==0u){partials[wid.x]=vec4u(bitcast<u32>(sum0[0]),bitcast<u32>(sum1[0]),count0[0],count1[0]);}}
@compute @workgroup_size(1)fn finalizeFinePotential(){var energy=0.;var volume=0.;var samples=0u;var invalid=0u;for(var i=0u;i<p.partialCount;i+=1u){let q=partials[i];energy+=bitcast<f32>(q.x);volume+=bitcast<f32>(q.y);samples+=q.z;invalid+=q.w;}let flags=VALID|p.kind;records[recordIndex()]=Record(p.step,p.generation,p.stage,flags,energy,volume,samples,invalid);atomicMax(&control[1],p.step+1u);}
`;
