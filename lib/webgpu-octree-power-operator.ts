/** Scalable GPU assembly and projection for generalized octree power faces. */

import { OCTREE_POWER_FACE_RECORD_BYTES, OCTREE_POWER_INVALID_ROW } from "./octree-power-operator";

export const OCTREE_POWER_GPU_ROW_BYTES = 16;
export const OCTREE_POWER_GPU_ENTRY_BYTES = 8;
export const OCTREE_POWER_GPU_CONTROL_BYTES = 64;
export const OCTREE_POWER_GPU_ASSEMBLED = 0x8000_0000;
export const OCTREE_POWER_GPU_PROJECTED = 0x4000_0000;
export const OCTREE_POWER_GPU_ERROR = Object.freeze({
  capacity: 1 << 0,
  invalidFace: 1 << 1,
  nonfiniteFace: 1 << 2,
  incidenceOverflow: 1 << 3,
  entryOverflow: 1 << 4,
  invalidVolume: 1 << 5,
  invalidPressure: 1 << 6,
  invalidState: 1 << 7,
} as const);

export interface OctreePowerGPUOperatorPlan {
  readonly rowCapacity: number;
  readonly faceCapacity: number;
  readonly entryCapacity: number;
  readonly maximumIncidencePerRow: number;
  readonly rowBytes: number;
  readonly faceBytes: number;
  readonly entryOffsetBytes: number;
  readonly entryBytes: number;
  readonly scalarBytes: number;
  readonly rowOffset: number;
  readonly entryOffsetOffset: number;
  readonly entryOffset: number;
  readonly projectedOffset: number;
  readonly divergenceOffset: number;
  readonly arenaBytes: number;
  readonly allocatedBytes: number;
}

export interface OctreePowerGPUOperatorSource {
  readonly plan: OctreePowerGPUOperatorPlan;
  readonly arena: GPUBuffer;
  readonly control: GPUBuffer;
}

export interface OctreePowerGPUIncidenceSource {
  /** WP4 RowWork records: incidence offset is word three of each 16-byte row. */
  readonly incidenceRows: GPUBuffer;
  /** Compact `(face:u32, sign:i32)` records, sorted by face within every row. */
  readonly incidence: GPUBuffer;
}

function positiveInteger(value: number, label: string): number {
  if (!Number.isSafeInteger(value) || value < 1) throw new RangeError(`${label} must be a positive integer`);
  return value;
}

export function planOctreePowerGPUOperator(
  rowCapacityValue: number,
  faceCapacityValue: number,
  entryCapacityValue: number,
  maximumIncidencePerRowValue: number,
): OctreePowerGPUOperatorPlan {
  const rowCapacity = positiveInteger(rowCapacityValue, "Power operator row capacity");
  const faceCapacity = positiveInteger(faceCapacityValue, "Power operator face capacity");
  const entryCapacity = positiveInteger(entryCapacityValue, "Power operator entry capacity");
  const maximumIncidencePerRow = positiveInteger(maximumIncidencePerRowValue, "Power operator incidence bound");
  const rowBytes = rowCapacity * OCTREE_POWER_GPU_ROW_BYTES;
  const faceBytes = faceCapacity * OCTREE_POWER_FACE_RECORD_BYTES;
  const entryOffsetBytes = (rowCapacity + 1) * 4;
  const entryBytes = entryCapacity * OCTREE_POWER_GPU_ENTRY_BYTES;
  const scalarBytes = faceCapacity * 4 + rowCapacity * 4;
  const rowOffset = 0;
  const entryOffsetOffset = rowOffset + rowBytes;
  const entryOffset = entryOffsetOffset + entryOffsetBytes;
  const projectedOffset = entryOffset + entryBytes;
  const divergenceOffset = projectedOffset + faceCapacity * 4;
  const arenaBytes = Math.ceil((divergenceOffset + rowCapacity * 4) / 16) * 16;
  return { rowCapacity, faceCapacity, entryCapacity, maximumIncidencePerRow, rowBytes, faceBytes,
    entryOffsetBytes, entryBytes, scalarBytes, rowOffset, entryOffsetOffset, entryOffset, projectedOffset,
    divergenceOffset, arenaBytes, allocatedBytes: arenaBytes + OCTREE_POWER_GPU_CONTROL_BYTES + 96 };
}

export class WebGPUOctreePowerOperator {
  readonly plan: OctreePowerGPUOperatorPlan;
  readonly arena: GPUBuffer;
  readonly control: GPUBuffer;
  private readonly assembleParams: GPUBuffer;
  private readonly projectParams: GPUBuffer;
  private readonly pipelines: Readonly<Record<string, { pipeline: GPUComputePipeline; bindings: readonly number[] }>>;
  private readonly device: GPUDevice;
  private destroyed = false;

  constructor(device: GPUDevice, rowCapacity: number, faceCapacity: number, entryCapacity: number, maximumIncidencePerRow: number) {
    this.device = device;
    this.plan = planOctreePowerGPUOperator(rowCapacity, faceCapacity, entryCapacity, maximumIncidencePerRow);
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    this.arena = device.createBuffer({ label: "Power compact operator arena", size: this.plan.arenaBytes, usage: storage });
    this.control = device.createBuffer({ label: "Power operator control", size: OCTREE_POWER_GPU_CONTROL_BYTES, usage: storage });
    this.assembleParams = device.createBuffer({ label: "Power assembly params", size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.projectParams = device.createBuffer({ label: "Power projection params", size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const shaderModule = device.createShaderModule({ label: "Octree power compact operator", code: octreePowerOperatorShader });
    const pipeline = (entryPoint: string, bindings: readonly number[]) => ({
      pipeline: device.createComputePipeline({ label: entryPoint, layout: "auto", compute: { module: shaderModule, entryPoint } }), bindings,
    });
    this.pipelines = Object.freeze({
      prepare: pipeline("preparePowerRows", [0, 1, 2, 3, 4, 6]), count: pipeline("countPowerRowEntries", [0, 1, 2, 3, 4, 5, 6]),
      scan: pipeline("scanPowerRowEntries", [0, 5, 6]), emit: pipeline("emitPowerRows", [0, 1, 2, 3, 4, 5, 6]),
      publish: pipeline("publishPowerRows", [6]), prepareProject: pipeline("preparePowerProjection", [0, 6]),
      prepareProjectMGPCG: pipeline("preparePowerProjectionMGPCG", [0, 6, 9]),
      project: pipeline("projectPowerFaces", [0, 1, 4, 5, 6]), divergence: pipeline("computePowerDivergence", [0, 1, 2, 3, 4, 5, 6]),
      publishProject: pipeline("publishPowerProjection", [0, 6]),
      commitProject: pipeline("commitPowerProjection", [0, 1, 5, 6]),
      publishLeafRows: pipeline("publishPowerLeafRows", [0, 5, 6, 7, 8]),
    });
  }

  encodeAssembly(
    encoder: GPUCommandEncoder,
    faces: GPUBuffer,
    csr: OctreePowerGPUIncidenceSource,
    volumes: GPUBuffer,
    rowCount: number,
    faceCount: number,
    incidenceCount: number,
  ): void {
    this.assertLive(); this.validateCounts(rowCount, faceCount, incidenceCount);
    this.device.queue.writeBuffer(this.assembleParams, 0, new Uint32Array([this.plan.maximumIncidencePerRow, rowCount,
      faceCount, incidenceCount, this.plan.entryCapacity, this.plan.rowCapacity, this.plan.faceCapacity, 0, 0, 0, 0, 0]));
    const bindings = [this.assembleParams, faces, csr.incidenceRows, csr.incidence, volumes, this.arena, this.control];
    this.run(encoder, this.pipelines.prepare, 1, bindings);
    if (rowCount > 0) this.run(encoder, this.pipelines.count, Math.ceil(rowCount / 64), bindings);
    this.run(encoder, this.pipelines.scan, 1, bindings);
    if (rowCount > 0) this.run(encoder, this.pipelines.emit, Math.ceil(rowCount / 64), bindings);
    this.run(encoder, this.pipelines.publish, 1, bindings);
  }

  /** No-readback path: source words 0..2 are row, face, and incidence counts (WP4 control ABI). */
  encodeAssemblyFromControl(
    encoder: GPUCommandEncoder,
    faces: GPUBuffer,
    csr: OctreePowerGPUIncidenceSource,
    volumes: GPUBuffer,
    countControl: GPUBuffer,
    velocitySeedControl?: GPUBuffer,
    solidControl?: GPUBuffer,
  ): void {
    this.assertLive();
    this.device.queue.writeBuffer(this.assembleParams, 0, new Uint32Array([this.plan.maximumIncidencePerRow, 0, 0, 0,
      this.plan.entryCapacity, this.plan.rowCapacity, this.plan.faceCapacity, 0,
      velocitySeedControl ? 1 : 0, 0, solidControl ? 1 : 0, 0]));
    encoder.copyBufferToBuffer(countControl, 0, this.assembleParams, 4, 12);
    if (velocitySeedControl) encoder.copyBufferToBuffer(velocitySeedControl, 24, this.assembleParams, 36, 4);
    if (solidControl) encoder.copyBufferToBuffer(solidControl, 28, this.assembleParams, 44, 4);
    const bindings = [this.assembleParams, faces, csr.incidenceRows, csr.incidence, volumes, this.arena, this.control];
    this.run(encoder, this.pipelines.prepare, 1, bindings);
    this.run(encoder, this.pipelines.count, Math.ceil(this.plan.rowCapacity / 64), bindings);
    this.run(encoder, this.pipelines.scan, 1, bindings);
    this.run(encoder, this.pipelines.emit, Math.ceil(this.plan.rowCapacity / 64), bindings);
    this.run(encoder, this.pipelines.publish, 1, bindings);
  }

  encodeProjection(
    encoder: GPUCommandEncoder,
    faces: GPUBuffer,
    csr: OctreePowerGPUIncidenceSource,
    pressure: GPUBuffer,
    rowCount: number,
    faceCount: number,
    incidenceCount: number,
    pressureScale = 1,
  ): void {
    this.assertLive(); this.validateCounts(rowCount, faceCount, incidenceCount);
    if (!Number.isFinite(pressureScale)) throw new RangeError("Power projection scale must be finite");
    const data = new ArrayBuffer(32); const words = new Uint32Array(data); const floats = new Float32Array(data);
    floats[0] = pressureScale; words[1] = rowCount; words[2] = faceCount; words[3] = incidenceCount;
    words[4] = this.plan.entryCapacity; words[5] = this.plan.rowCapacity; words[6] = this.plan.faceCapacity;
    this.device.queue.writeBuffer(this.projectParams, 0, data);
    const bindings = [this.projectParams, faces, csr.incidenceRows, csr.incidence, pressure, this.arena, this.control];
    this.run(encoder, this.pipelines.prepareProject, 1, bindings);
    if (faceCount > 0) this.run(encoder, this.pipelines.project, Math.ceil(faceCount / 64), bindings);
    if (rowCount > 0) this.run(encoder, this.pipelines.divergence, Math.ceil(rowCount / 64), bindings);
    this.run(encoder, this.pipelines.publishProject, 1, bindings);
    if (faceCount > 0) this.run(encoder, this.pipelines.commitProject, Math.ceil(faceCount / 64), bindings);
  }

  /**
   * Atomically selects the assembled power rows for the existing Chebyshev
   * arena.  A failed/incomplete assembly is a no-op, preserving the axis rows
   * that the caller assembled first as the rollback generation.
   */
  encodeLeafRowPublication(
    encoder: GPUCommandEncoder,
    leafHeaders: GPUBuffer,
    leafEntries: GPUBuffer,
  ): void {
    this.assertLive();
    if (leafHeaders.size < this.plan.rowCapacity * 48) {
      throw new RangeError("Power publication LeafHeader capacity is too small");
    }
    if (leafEntries.size < this.plan.entryCapacity * OCTREE_POWER_GPU_ENTRY_BYTES) {
      throw new RangeError("Power publication LeafEntry capacity is too small");
    }
    const bindings = [this.assembleParams, this.arena, this.arena, this.arena, this.arena,
      this.arena, this.control, leafHeaders, leafEntries];
    this.run(encoder, this.pipelines.publishLeafRows, Math.ceil(this.plan.rowCapacity / 64), bindings);
  }

  /** Projects using the first three WP4 control words without a CPU count readback. */
  encodeProjectionFromControl(
    encoder: GPUCommandEncoder,
    faces: GPUBuffer,
    csr: OctreePowerGPUIncidenceSource,
    pressure: GPUBuffer,
    countControl: GPUBuffer,
    pressureScale = 1,
    solverControl?: GPUBuffer,
  ): void {
    this.assertLive();
    if (!Number.isFinite(pressureScale)) throw new RangeError("Power projection scale must be finite");
    const data = new ArrayBuffer(32); const words = new Uint32Array(data); const floats = new Float32Array(data);
    floats[0] = pressureScale; words[4] = this.plan.entryCapacity; words[5] = this.plan.rowCapacity; words[6] = this.plan.faceCapacity;
    this.device.queue.writeBuffer(this.projectParams, 0, data);
    encoder.copyBufferToBuffer(countControl, 0, this.projectParams, 4, 12);
    const bindings = [this.projectParams, faces, csr.incidenceRows, csr.incidence, pressure, this.arena, this.control,
      this.arena, this.arena, solverControl ?? this.control];
    this.run(encoder, solverControl ? this.pipelines.prepareProjectMGPCG : this.pipelines.prepareProject, 1, bindings);
    this.run(encoder, this.pipelines.project, Math.ceil(this.plan.faceCapacity / 64), bindings);
    this.run(encoder, this.pipelines.divergence, Math.ceil(this.plan.rowCapacity / 64), bindings);
    this.run(encoder, this.pipelines.publishProject, 1, bindings);
    this.run(encoder, this.pipelines.commitProject, Math.ceil(this.plan.faceCapacity / 64), bindings);
  }

  get source(): OctreePowerGPUOperatorSource { return { plan: this.plan, arena: this.arena, control: this.control }; }

  private run(encoder: GPUCommandEncoder, stage: { pipeline: GPUComputePipeline; bindings: readonly number[] }, groups: number, buffers: readonly GPUBuffer[]): void {
    const group = this.device.createBindGroup({ layout: stage.pipeline.getBindGroupLayout(0),
      entries: stage.bindings.map((binding) => ({ binding, resource: { buffer: buffers[binding] } })) });
    const pass = encoder.beginComputePass(); pass.setPipeline(stage.pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(groups); pass.end();
  }

  private validateCounts(rowCount: number, faceCount: number, incidenceCount: number): void {
    for (const [label, value, capacity] of [["row", rowCount, this.plan.rowCapacity], ["face", faceCount, this.plan.faceCapacity],
      ["incidence", incidenceCount, this.plan.faceCapacity * 2]] as const) {
      if (!Number.isSafeInteger(value) || value < 0 || value > capacity) throw new RangeError(`Power operator ${label} count exceeds capacity`);
    }
  }

  private assertLive(): void { if (this.destroyed) throw new Error("Octree power operator is destroyed"); }
  destroy(): void {
    if (this.destroyed) return; this.destroyed = true;
    this.arena.destroy(); this.control.destroy(); this.assembleParams.destroy(); this.projectParams.destroy();
  }
}

/**
 * WP4-compatible CSR is authoritative. For ghost-fluid faces, inverseDistance
 * is the caller's bounded effective dual-edge inverse distance; assembly and
 * projection consume exactly the same scalar.
 */
export const octreePowerOperatorShader = /* wgsl */ `
struct Params { primary:vec4u,capacities:vec4u,padding:vec4u }
struct PowerFaceRecord { negativeRow:u32,positiveRow:u32,geometryCode:u32,flags:u32,normalVelocity:f32,area:f32,inverseDistance:f32,openFraction:f32 }
struct RowWork { faceCount:u32,incidenceCount:u32,faceOffset:u32,incidenceOffset:u32 }
struct PowerIncidence { face:u32,sign:i32 }
struct LeafHeader { cell:u32,entryStart:u32,entryCount:u32,size:u32,diagonal:f32,rhs:f32,pad0:u32,pad1:u32,gradient:vec4f }
struct LeafEntry { row:u32,coefficient:f32 }
struct Control { flags:atomic<u32>,firstError:atomic<u32>,rowCount:atomic<u32>,faceCount:atomic<u32>,incidenceCount:atomic<u32>,entryCount:atomic<u32>,projectedCount:atomic<u32>,reserved:atomic<u32>,pad0:atomic<u32>,pad1:atomic<u32>,pad2:atomic<u32>,pad3:atomic<u32>,pad4:atomic<u32>,pad5:atomic<u32>,pad6:atomic<u32>,pad7:atomic<u32> }
@group(0) @binding(0) var<uniform> params:Params;
@group(0) @binding(1) var<storage,read_write> faces:array<PowerFaceRecord>;
@group(0) @binding(2) var<storage,read> incidenceRows:array<RowWork>;
@group(0) @binding(3) var<storage,read> incidences:array<PowerIncidence>;
@group(0) @binding(4) var<storage,read> values:array<f32>;
@group(0) @binding(5) var<storage,read_write> arena:array<u32>;
@group(0) @binding(6) var<storage,read_write> control:Control;
@group(0) @binding(7) var<storage,read_write> leafHeaders:array<LeafHeader>;
@group(0) @binding(8) var<storage,read_write> leafEntries:array<LeafEntry>;
// Optional MGPCG gate. Only preparePowerProjectionMGPCG statically uses this
// binding, so ordinary Chebyshev projection retains its existing layout.
@group(0) @binding(9) var<storage,read_write> solverControl:array<atomic<u32>>;
const INVALID:u32=${OCTREE_POWER_INVALID_ROW}u;const ASSEMBLED:u32=0x80000000u;const PROJECTED:u32=0x40000000u;
const BOUNDARY:u32=1u;const OPEN_BOUNDARY:u32=2u;const ROW_BOUNDARY:u32=1u;
const CAPACITY:u32=1u;const INVALID_FACE:u32=2u;const NONFINITE_FACE:u32=4u;const INCIDENCE_OVERFLOW:u32=8u;
const ENTRY_OVERFLOW:u32=16u;const INVALID_VOLUME:u32=32u;const INVALID_PRESSURE:u32=64u;const INVALID_STATE:u32=128u;
fn finite(v:f32)->bool{return v==v&&abs(v)<=3.402823e38;}
fn rowCount()->u32{return params.primary.y;}fn faceCount()->u32{return params.primary.z;}fn incidenceCount()->u32{return params.primary.w;}
fn entryCapacity()->u32{return params.capacities.x;}fn rowCapacity()->u32{return params.capacities.y;}fn faceCapacity()->u32{return params.capacities.z;}
fn entryOffsetBase()->u32{return rowCapacity()*4u;}fn entryBase()->u32{return entryOffsetBase()+rowCapacity()+1u;}
fn projectedBase()->u32{return entryBase()+entryCapacity()*2u;}fn divergenceBase()->u32{return projectedBase()+faceCapacity();}
fn report(error:u32,index:u32){atomicOr(&control.flags,error);atomicMin(&control.firstError,index);}
fn begin(row:u32)->u32{return incidenceRows[row].incidenceOffset;}fn end(row:u32)->u32{return incidenceRows[row+1u].incidenceOffset;}
fn neighbor(face:PowerFaceRecord,row:u32)->u32{return select(face.negativeRow,face.positiveRow,face.negativeRow==row);}
fn validFace(face:PowerFaceRecord)->bool{return face.negativeRow<rowCount()&&face.positiveRow!=face.negativeRow
  &&(face.positiveRow==INVALID||face.positiveRow<rowCount())&&finite(face.normalVelocity)&&finite(face.area)&&finite(face.inverseDistance)
  &&finite(face.openFraction)&&face.area>0.0&&face.inverseDistance>=0.0&&face.openFraction>=0.0&&face.openFraction<=1.0
  &&(face.positiveRow==INVALID||face.inverseDistance>0.0);}
@compute @workgroup_size(1) fn preparePowerRows(){atomicStore(&control.flags,0u);atomicStore(&control.firstError,INVALID);
  atomicStore(&control.rowCount,rowCount());atomicStore(&control.faceCount,faceCount());atomicStore(&control.incidenceCount,incidenceCount());
  atomicStore(&control.entryCount,0u);atomicStore(&control.projectedCount,0u);
  if((params.padding.x!=0u&&params.padding.y!=0x80000000u)||(params.padding.z!=0u&&params.padding.w!=0x80000000u)){report(INVALID_STATE,0u);}
  if(rowCount()>rowCapacity()||faceCount()>faceCapacity()||rowCount()+1u>arrayLength(&incidenceRows)||incidenceCount()>arrayLength(&incidences)
    ||rowCount()>arrayLength(&values)||faceCount()>arrayLength(&faces)){report(CAPACITY,0u);}}
@compute @workgroup_size(64) fn countPowerRowEntries(@builtin(global_invocation_id) gid:vec3u){let row=gid.x;if(row>=rowCount()){return;}
  if(atomicLoad(&control.flags)!=0u){return;}
  let b=begin(row);let e=end(row);if(b>e||e>incidenceCount()||e-b>params.primary.x){report(INCIDENCE_OVERFLOW,row);return;}
  if(!finite(values[row])||values[row]<=0.0){report(INVALID_VOLUME,row);return;}var unique=0u;var previousFace=0u;
  for(var cursor=b;cursor<e;cursor+=1u){let item=incidences[cursor];if(item.face>=faceCount()||(cursor>b&&item.face<=previousFace)){report(INVALID_FACE,row);return;}
    previousFace=item.face;let face=faces[item.face];let expected=select(-1,1,face.negativeRow==row);
    if(!finite(face.normalVelocity)||!finite(face.area)||!finite(face.inverseDistance)||!finite(face.openFraction)){report(NONFINITE_FACE,item.face);return;}
    if(!validFace(face)||neighbor(face,row)==row||item.sign!=expected||(face.negativeRow!=row&&face.positiveRow!=row)){report(INVALID_FACE,item.face);return;}
    let other=neighbor(face,row);if(other==INVALID){continue;}var first=true;for(var earlier=b;earlier<cursor;earlier+=1u){if(neighbor(faces[incidences[earlier].face],row)==other){first=false;break;}}
    if(first){unique+=1u;}
  }arena[entryOffsetBase()+row]=unique;}
@compute @workgroup_size(1) fn scanPowerRowEntries(){if(atomicLoad(&control.flags)!=0u){return;}var total=0u;
  for(var row=0u;row<rowCount();row+=1u){let count=arena[entryOffsetBase()+row];arena[entryOffsetBase()+row]=total;total+=count;
    if(total>entryCapacity()){report(ENTRY_OVERFLOW,row);return;}}arena[entryOffsetBase()+rowCount()]=total;atomicStore(&control.entryCount,total);}
@compute @workgroup_size(64) fn emitPowerRows(@builtin(global_invocation_id) gid:vec3u){let row=gid.x;if(row>=rowCount()||atomicLoad(&control.flags)!=0u){return;}
  let b=begin(row);let e=end(row);var rhs=0.0;var diagonal=0.0;var local=0u;var rowFlags=0u;
  for(var cursor=b;cursor<e;cursor+=1u){let item=incidences[cursor];let face=faces[item.face];rhs+=f32(item.sign)*face.area*face.normalVelocity;
    if((face.flags&(BOUNDARY|OPEN_BOUNDARY))!=0u||face.openFraction<1.0){rowFlags|=ROW_BOUNDARY;}
    let other=neighbor(face,row);if(other==INVALID){if((face.flags&OPEN_BOUNDARY)!=0u){diagonal+=face.openFraction*face.area*face.inverseDistance;}continue;}var first=true;for(var earlier=b;earlier<cursor;earlier+=1u){if(neighbor(faces[incidences[earlier].face],row)==other){first=false;break;}}
    if(!first){continue;}var coefficient=0.0;for(var merge=cursor;merge<e;merge+=1u){let merged=faces[incidences[merge].face];
      if(neighbor(merged,row)==other){coefficient+=merged.openFraction*merged.area*merged.inverseDistance;}}
    if(!finite(coefficient)||coefficient<0.0){report(NONFINITE_FACE,item.face);return;}let output=arena[entryOffsetBase()+row]+local;
    arena[entryBase()+output*2u]=other;arena[entryBase()+output*2u+1u]=bitcast<u32>(coefficient);diagonal+=coefficient;local+=1u;
  }if(!finite(rhs)||!finite(diagonal)){report(NONFINITE_FACE,row);return;}let base=row*4u;arena[base]=bitcast<u32>(diagonal);
  arena[base+1u]=bitcast<u32>(rhs);arena[base+2u]=bitcast<u32>(values[row]);arena[base+3u]=rowFlags;}
@compute @workgroup_size(1) fn publishPowerRows(){if(atomicLoad(&control.flags)==0u){atomicStore(&control.flags,ASSEMBLED);}}
@compute @workgroup_size(64) fn publishPowerLeafRows(@builtin(global_invocation_id) gid:vec3u){let row=gid.x;
  if(atomicLoad(&control.flags)!=ASSEMBLED||row>=rowCount()||row>=arrayLength(&leafHeaders)){return;}
  let start=arena[entryOffsetBase()+row];let finish=arena[entryOffsetBase()+row+1u];
  if(start>finish||finish>atomicLoad(&control.entryCount)||finish>arrayLength(&leafEntries)){return;}
  var header=leafHeaders[row];let base=row*4u;header.entryStart=start;header.entryCount=finish-start;
  header.diagonal=bitcast<f32>(arena[base]);header.rhs=bitcast<f32>(arena[base+1u]);header.pad0=arena[base+3u];header.pad1=0u;leafHeaders[row]=header;
  for(var index=start;index<finish;index+=1u){leafEntries[index]=LeafEntry(arena[entryBase()+index*2u],bitcast<f32>(arena[entryBase()+index*2u+1u]));}}
fn preparePowerProjectionState(){let old=atomicLoad(&control.flags);atomicStore(&control.pad2,old);atomicStore(&control.pad3,atomicLoad(&control.firstError));atomicStore(&control.projectedCount,0u);
  atomicStore(&control.pad0,0u);atomicStore(&control.pad1,0u);
  if((old&ASSEMBLED)==0u||(old&~(ASSEMBLED|PROJECTED))!=0u||atomicLoad(&control.rowCount)!=rowCount()||atomicLoad(&control.faceCount)!=faceCount()
    ||atomicLoad(&control.incidenceCount)!=incidenceCount()){atomicStore(&control.flags,INVALID_STATE);atomicStore(&control.firstError,0u);return;}
  atomicStore(&control.flags,0u);atomicStore(&control.firstError,INVALID);if(!finite(bitcast<f32>(params.primary.x))){report(INVALID_PRESSURE,0u);}}
@compute @workgroup_size(1) fn preparePowerProjection(){preparePowerProjectionState();}
@compute @workgroup_size(1) fn preparePowerProjectionMGPCG(){preparePowerProjectionState();if(atomicLoad(&control.flags)!=0u){return;}
  if(arrayLength(&solverControl)<2u||atomicLoad(&solverControl[0])!=0u||atomicLoad(&solverControl[1])==0u){report(INVALID_STATE,0u);}}
@compute @workgroup_size(64) fn projectPowerFaces(@builtin(global_invocation_id) gid:vec3u){let index=gid.x;if(index>=faceCount()||atomicLoad(&control.flags)!=0u){return;}
  let face=faces[index];var velocity=face.normalVelocity;atomicMax(&control.pad0,bitcast<u32>(abs(velocity)));if(face.positiveRow!=INVALID||(face.flags&OPEN_BOUNDARY)!=0u){let negative=values[face.negativeRow];var positive=0.0;if(face.positiveRow!=INVALID){positive=values[face.positiveRow];}
    if(!finite(negative)){report(INVALID_PRESSURE,face.negativeRow);return;}if(!finite(positive)){report(INVALID_PRESSURE,face.positiveRow);return;}
    velocity-=bitcast<f32>(params.primary.x)*(positive-negative)*face.inverseDistance*face.openFraction;}
  if(!finite(velocity)){report(INVALID_PRESSURE,index);return;}atomicMax(&control.pad1,bitcast<u32>(abs(velocity)));arena[projectedBase()+index]=bitcast<u32>(velocity);}
@compute @workgroup_size(64) fn computePowerDivergence(@builtin(global_invocation_id) gid:vec3u){let row=gid.x;if(row>=rowCount()||atomicLoad(&control.flags)!=0u){return;}
  if(!finite(values[row])){report(INVALID_PRESSURE,row);return;}
  var integrated=0.0;for(var cursor=begin(row);cursor<end(row);cursor+=1u){let item=incidences[cursor];integrated+=f32(item.sign)*faces[item.face].area*bitcast<f32>(arena[projectedBase()+item.face]);}
  let volume=bitcast<f32>(arena[row*4u+2u]);let value=integrated/volume;if(!finite(value)){report(INVALID_PRESSURE,row);return;}
  arena[divergenceBase()+row]=bitcast<u32>(value);}
@compute @workgroup_size(1) fn publishPowerProjection(){if(atomicLoad(&control.flags)==0u){atomicStore(&control.flags,ASSEMBLED|PROJECTED);atomicStore(&control.projectedCount,faceCount());}}
@compute @workgroup_size(64) fn commitPowerProjection(@builtin(global_invocation_id) gid:vec3u){let index=gid.x;
  if(atomicLoad(&control.flags)!=(ASSEMBLED|PROJECTED)||index>=faceCount()||index>=arrayLength(&faces)){return;}
  faces[index].normalVelocity=bitcast<f32>(arena[projectedBase()+index]);}
`;
