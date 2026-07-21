/**
 * Stage A of the transition-aware velocity interpolant from Aanjaneya et al.
 * (2017), Section 5: reconstruct one full vector at each power-cell center by
 * an area-weighted least-squares fit to all incident face-normal velocities.
 *
 * Input compatibility is deliberately narrow and explicit:
 *
 * - `faces` uses the 32-byte `PowerFaceRecord` ABI shared by
 *   `webgpu-octree-power-faces.ts` and `webgpu-octree-power-operator.ts`;
 * - `faceNormals` is a transient/resolved `vec4f` per face. Its xyz lanes are
 *   the unit normal directed from `negativeRow` to `positiveRow` (or outward
 *   for a boundary face). The w lane is ignored. A future catalog-backed
 *   reconstruction stage can produce the same interface without changing
 *   this solver;
 * - `incidenceRows` is WP4's 16-byte `RowWork` stream; its fourth u32 is the
 *   compact incidence offset and the terminal row stores the final offset;
 * - `incidences` is WP4's 8-byte `(face:u32, sign:i32)` stream.
 *
 * Thus `WebGPUOctreePowerFaces.source` binds directly without an incidence
 * repack. No Cartesian u/v/w face copies are allocated here.
 *
 * Stage B (catalog cube/tetrahedron lookup and point interpolation) is outside
 * this module.
 */

import { OCTREE_POWER_FACE_RECORD_BYTES } from "./octree-power-operator";
import {
  OCTREE_POWER_CATALOG_ENTRY_UNIFORM,
  type OctreePowerCatalogEntry,
} from "./octree-power-catalog";
import type { PowerVec3 } from "./octree-power-geometry";
import type { OctreePowerTopologySource } from "./webgpu-octree-power-topology";

export const OCTREE_POWER_VELOCITY_BYTES = 16;
export const OCTREE_POWER_VELOCITY_CONTROL_BYTES = 32;
export const OCTREE_POWER_VELOCITY_VALID = 0x8000_0000;

export const OCTREE_POWER_VELOCITY_ERROR = Object.freeze({
  capacity: 1 << 0,
  invalidOffsets: 1 << 1,
  invalidIncidence: 1 << 2,
  invalidFace: 1 << 3,
  invalidNormal: 1 << 4,
  nonfiniteSample: 1 << 5,
  nonfiniteSolution: 1 << 6,
  invalidSource: 1 << 7,
  illConditioned: 1 << 8,
} as const);

export interface OctreePowerVelocityPlan {
  readonly rowCapacity: number;
  readonly velocityBytes: number;
  readonly statusBytes: number;
  readonly allocatedBytes: number;
}

export interface OctreePowerVelocityInput {
  readonly faces: GPUBuffer;
  readonly faceNormals: GPUBuffer;
  readonly incidenceRows: GPUBuffer;
  readonly incidences: GPUBuffer;
}

export interface OctreePowerVelocityEncodeOptions {
  readonly rowCount: number;
  readonly faceCount: number;
  readonly incidenceCount: number;
  readonly maximumIncidencePerRow: number;
  readonly determinantTolerance?: number;
  readonly maximumConditionNumber?: number;
  readonly generation?: number;
}

export interface OctreePowerVelocityControl {
  readonly flags: number;
  readonly firstError: number;
  readonly rowCount: number;
  readonly faceCount: number;
  readonly incidenceCount: number;
  readonly reconstructedCount: number;
  readonly fallbackCount: number;
  readonly generation: number;
}

export interface OctreePowerVelocitySource {
  readonly plan: OctreePowerVelocityPlan;
  /** xyz is velocity; w is one for a solved fit and zero for a guarded fallback. */
  readonly velocities: GPUBuffer;
  readonly control: GPUBuffer;
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

export function planOctreePowerVelocity(rowCapacityValue: number): OctreePowerVelocityPlan {
  const rowCapacity = positiveInteger(rowCapacityValue, "Power velocity row capacity");
  const velocityBytes = rowCapacity * OCTREE_POWER_VELOCITY_BYTES;
  const statusBytes = rowCapacity * 4;
  return {
    rowCapacity,
    velocityBytes,
    statusBytes,
    allocatedBytes: velocityBytes + statusBytes + OCTREE_POWER_VELOCITY_CONTROL_BYTES + 32,
  };
}

export function unpackOctreePowerVelocityControl(words: ArrayLike<number>): OctreePowerVelocityControl {
  if (words.length < 8) throw new RangeError("Power velocity control needs at least eight words");
  return {
    flags: Number(words[0]) >>> 0,
    firstError: Number(words[1]) >>> 0,
    rowCount: Number(words[2]) >>> 0,
    faceCount: Number(words[3]) >>> 0,
    incidenceCount: Number(words[4]) >>> 0,
    reconstructedCount: Number(words[5]) >>> 0,
    fallbackCount: Number(words[6]) >>> 0,
    generation: Number(words[7]) >>> 0,
  };
}

export class WebGPUOctreePowerVelocity {
  readonly plan: OctreePowerVelocityPlan;
  readonly velocities: GPUBuffer;
  readonly control: GPUBuffer;
  private readonly params: GPUBuffer;
  private readonly rowStatus: GPUBuffer;
  private readonly preparePipeline: GPUComputePipeline;
  private readonly prepareFromFaceControlPipeline: GPUComputePipeline;
  private readonly prepareFromProjectedFaceControlPipeline: GPUComputePipeline;
  private readonly reconstructPipeline: GPUComputePipeline;
  private readonly publishPipeline: GPUComputePipeline;
  private destroyed = false;

  constructor(private readonly device: GPUDevice, rowCapacity: number) {
    this.plan = planOctreePowerVelocity(rowCapacity);
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    this.velocities = device.createBuffer({ label: "Octree power cell velocities", size: this.plan.velocityBytes, usage: storage });
    this.rowStatus = device.createBuffer({ label: "Octree power velocity row status", size: this.plan.statusBytes, usage: storage });
    this.control = device.createBuffer({ label: "Octree power velocity control", size: OCTREE_POWER_VELOCITY_CONTROL_BYTES, usage: storage });
    this.params = device.createBuffer({ label: "Octree power velocity parameters", size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, mappedAtCreation: true });
    new Uint8Array(this.params.getMappedRange()).fill(0); this.params.unmap();
    const pipeline = (label: string, code: string, entryPoint: string) => device.createComputePipeline({ label,
      layout: "auto", compute: { module: device.createShaderModule({ label, code }), entryPoint } });
    this.preparePipeline = pipeline("Prepare octree power velocity", octreePowerVelocityPrepareShader, "preparePowerVelocity");
    this.prepareFromFaceControlPipeline = pipeline("Prepare octree power velocity from face authority",
      octreePowerVelocityPrepareFromFaceControlShader, "preparePowerVelocityFromFaceControl");
    this.prepareFromProjectedFaceControlPipeline = pipeline("Prepare octree power velocity from projected face authority",
      octreePowerVelocityPrepareFromFaceControlShader, "preparePowerVelocityFromProjectedFaceControl");
    this.reconstructPipeline = pipeline("Reconstruct octree power cell velocities", octreePowerVelocityShader, "reconstructPowerVelocity");
    this.publishPipeline = pipeline("Publish octree power cell velocities", octreePowerVelocityPublishShader, "publishPowerVelocity");
  }

  encode(encoder: GPUCommandEncoder, input: OctreePowerVelocityInput, options: OctreePowerVelocityEncodeOptions): void {
    this.assertLive();
    const rowCount = nonNegativeInteger(options.rowCount, "Power velocity row count");
    const faceCount = nonNegativeInteger(options.faceCount, "Power velocity face count");
    const incidenceCount = nonNegativeInteger(options.incidenceCount, "Power velocity incidence count");
    const maximumIncidencePerRow = positiveInteger(options.maximumIncidencePerRow, "Power velocity incidence bound");
    const generation = nonNegativeInteger(options.generation ?? 0, "Power velocity generation");
    const determinantTolerance = options.determinantTolerance ?? 1e-7;
    const maximumConditionNumber = options.maximumConditionNumber ?? 1e5;
    if (!Number.isFinite(determinantTolerance) || determinantTolerance < 0
      || !Number.isFinite(maximumConditionNumber) || maximumConditionNumber < 1) {
      throw new RangeError("Power velocity conditioning guards must be finite and non-negative");
    }
    const data = new ArrayBuffer(32); const words = new Uint32Array(data); const floats = new Float32Array(data);
    words.set([rowCount, faceCount, incidenceCount, this.plan.rowCapacity, maximumIncidencePerRow, generation]);
    floats[6] = determinantTolerance; floats[7] = maximumConditionNumber;
    this.device.queue.writeBuffer(this.params, 0, words);
    this.encodePasses(encoder, input, Math.min(rowCount, this.plan.rowCapacity));
  }

  /** Reads `(rowCount, faceCount, incidenceCount)` directly from WP4 control on the GPU. */
  encodeFromFaceControl(encoder: GPUCommandEncoder, input: OctreePowerVelocityInput, faceControl: GPUBuffer, options: {
    maximumIncidencePerRow: number; generation?: number; determinantTolerance?: number; maximumConditionNumber?: number;
    projectionControl?: GPUBuffer;
  }): void {
    this.assertLive();
    const maximumIncidencePerRow = positiveInteger(options.maximumIncidencePerRow, "Power velocity incidence bound");
    const generation = nonNegativeInteger(options.generation ?? 0, "Power velocity generation");
    const determinantTolerance = options.determinantTolerance ?? 1e-7;
    const maximumConditionNumber = options.maximumConditionNumber ?? 1e5;
    if (!Number.isFinite(determinantTolerance) || determinantTolerance < 0
      || !Number.isFinite(maximumConditionNumber) || maximumConditionNumber < 1) {
      throw new RangeError("Power velocity conditioning guards must be finite and non-negative");
    }
    const data = new ArrayBuffer(32), words = new Uint32Array(data), floats = new Float32Array(data);
    words.set([0, 0, 0, this.plan.rowCapacity, maximumIncidencePerRow, generation]);
    floats[6] = determinantTolerance; floats[7] = maximumConditionNumber;
    this.device.queue.writeBuffer(this.params, 0, data);
    encoder.copyBufferToBuffer(faceControl, 0, this.params, 0, 12);
    this.encodePasses(encoder, input, this.plan.rowCapacity, faceControl, options.projectionControl);
  }

  private encodePasses(encoder: GPUCommandEncoder, input: OctreePowerVelocityInput, available: number,
    faceControl?: GPUBuffer, projectionControl?: GPUBuffer): void {
    const resource = (buffer: GPUBuffer) => ({ buffer });
    const group = (pipeline: GPUComputePipeline, entries: GPUBindGroupEntry[]) => this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0), entries,
    });
    const params = { binding: 0, resource: resource(this.params) };
    // Every generation starts from explicit invalid vectors/status. An invalid
    // upstream face publication can therefore never expose stale Stage-A data.
    encoder.clearBuffer(this.velocities); encoder.clearBuffer(this.rowStatus);
    let pass = encoder.beginComputePass({ label: "Prepare power velocity" });
    const prepare = projectionControl ? this.prepareFromProjectedFaceControlPipeline
      : faceControl ? this.prepareFromFaceControlPipeline : this.preparePipeline;
    pass.setPipeline(prepare); pass.setBindGroup(0, group(prepare, [params,
      { binding: 1, resource: resource(this.control) },
      ...(faceControl ? [{ binding: 2, resource: resource(faceControl) }] : []),
      ...(projectionControl ? [{ binding: 3, resource: resource(projectionControl) }] : [])]));
    pass.dispatchWorkgroups(1); pass.end();
    if (available > 0) {
      pass = encoder.beginComputePass({ label: "Reconstruct power velocity" });
      pass.setPipeline(this.reconstructPipeline); pass.setBindGroup(0, group(this.reconstructPipeline, [params,
        { binding: 1, resource: resource(input.faces) }, { binding: 2, resource: resource(input.faceNormals) },
        { binding: 3, resource: resource(input.incidenceRows) }, { binding: 4, resource: resource(input.incidences) },
        { binding: 5, resource: resource(this.velocities) }, { binding: 6, resource: resource(this.rowStatus) },
        { binding: 7, resource: resource(this.control) }]));
      pass.dispatchWorkgroups(Math.ceil(available / 64)); pass.end();
    }
    pass = encoder.beginComputePass({ label: "Publish power velocity" });
    pass.setPipeline(this.publishPipeline); pass.setBindGroup(0, group(this.publishPipeline, [params,
      { binding: 1, resource: resource(input.faces) }, { binding: 2, resource: resource(input.faceNormals) },
      { binding: 3, resource: resource(input.incidenceRows) }, { binding: 4, resource: resource(input.incidences) },
      { binding: 5, resource: resource(this.rowStatus) }, { binding: 6, resource: resource(this.control) }]));
    pass.dispatchWorkgroups(1); pass.end();
  }

  get source(): OctreePowerVelocitySource {
    return { plan: this.plan, velocities: this.velocities, control: this.control };
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true; this.velocities.destroy(); this.rowStatus.destroy(); this.control.destroy(); this.params.destroy();
  }

  private assertLive(): void {
    if (this.destroyed) throw new Error("Octree power velocity reconstruction is destroyed");
  }
}

export const octreePowerVelocityShader = /* wgsl */ `
struct PowerFaceRecord {
  negativeRow:u32, positiveRow:u32, geometryCode:u32, flags:u32,
  normalVelocity:f32, area:f32, inverseDistance:f32, openFraction:f32,
}
struct RowWork { faceCount:u32, incidenceCount:u32, faceOffset:u32, incidenceOffset:u32 }
struct PowerIncidence { face:u32, sign:i32 }
struct VelocityParams {
  rowCount:u32, faceCount:u32, incidenceCount:u32, rowCapacity:u32,
  maximumIncidencePerRow:u32, generation:u32, determinantToleranceBits:u32, maximumConditionNumberBits:u32,
}
@group(0) @binding(0) var<uniform> params:VelocityParams;
@group(0) @binding(1) var<storage,read> faces:array<PowerFaceRecord>;
@group(0) @binding(2) var<storage,read> faceNormals:array<vec4f>;
@group(0) @binding(3) var<storage,read> incidenceRows:array<RowWork>;
@group(0) @binding(4) var<storage,read> incidences:array<PowerIncidence>;
@group(0) @binding(5) var<storage,read_write> velocities:array<vec4f>;
@group(0) @binding(6) var<storage,read_write> rowStatus:array<u32>;
@group(0) @binding(7) var<storage,read> velocityControl:array<u32>;
const CAPACITY:u32=1u;
const INVALID_OFFSETS:u32=2u;
const INVALID_INCIDENCE:u32=4u;
const INVALID_FACE:u32=8u;
const INVALID_NORMAL:u32=16u;
const NONFINITE_SAMPLE:u32=32u;
const NONFINITE_SOLUTION:u32=64u;
const FALLBACK:u32=0x80000000u;
fn finite(value:f32)->bool{return (bitcast<u32>(value)&0x7f800000u)!=0x7f800000u;}

@compute @workgroup_size(64) fn reconstructPowerVelocity(@builtin(global_invocation_id) gid:vec3u){
  let row=gid.x;if(velocityControl[0]!=0u||row>=params.rowCount){return;}
  if(params.rowCount>params.rowCapacity||row>=arrayLength(&velocities)||row>=arrayLength(&rowStatus)
    ||row+1u>=arrayLength(&incidenceRows)||params.faceCount>arrayLength(&faces)
    ||params.faceCount>arrayLength(&faceNormals)||params.incidenceCount>arrayLength(&incidences)){
    if(row<arrayLength(&rowStatus)){rowStatus[row]=CAPACITY;}return;
  }
  let begin=incidenceRows[row].incidenceOffset;let end=incidenceRows[row+1u].incidenceOffset;
  if(begin>end||end>params.incidenceCount||end-begin>params.maximumIncidencePerRow){
    rowStatus[row]=INVALID_OFFSETS;return;
  }
  var xx=0.0;var xy=0.0;var xz=0.0;var yy=0.0;var yz=0.0;var zz=0.0;var b=vec3f(0.0);
  for(var cursor=begin;cursor<end;cursor+=1u){
    let item=incidences[cursor];let faceIndex=item.face;let positive=item.sign<0;
    if(faceIndex>=params.faceCount||abs(item.sign)!=1){rowStatus[row]=INVALID_INCIDENCE;return;}
    let face=faces[faceIndex];
    if(select(face.negativeRow,face.positiveRow,positive)!=row||face.negativeRow==face.positiveRow){
      rowStatus[row]=INVALID_FACE;return;
    }
    let rawNormal=faceNormals[faceIndex].xyz;let lengthSquared=dot(rawNormal,rawNormal);
    if(!finite(rawNormal.x)||!finite(rawNormal.y)||!finite(rawNormal.z)||!finite(lengthSquared)
      ||abs(lengthSquared-1.0)>4e-4){rowStatus[row]=INVALID_NORMAL;return;}
    if(!finite(face.area)||face.area<=0.0||!finite(face.normalVelocity)){
      rowStatus[row]=NONFINITE_SAMPLE;return;
    }
    // Positive-row orientation negates both n and u; the normal equation is
    // unchanged, while this explicit form documents the signed-incidence ABI.
    let orientation=select(1.0,-1.0,positive);let n=orientation*rawNormal;let u=orientation*face.normalVelocity;let w=face.area;
    xx+=w*n.x*n.x;xy+=w*n.x*n.y;xz+=w*n.x*n.z;
    yy+=w*n.y*n.y;yz+=w*n.y*n.z;zz+=w*n.z*n.z;b+=w*n*u;
  }
  let c00=yy*zz-yz*yz;let c01=xz*yz-xy*zz;let c02=xy*yz-xz*yy;
  let c11=xx*zz-xz*xz;let c12=xy*xz-xx*yz;let c22=xx*yy-xy*xy;
  let determinant=xx*c00+xy*c01+xz*c02;let trace=xx+yy+zz;
  let matrixNorm2=xx*xx+yy*yy+zz*zz+2.0*(xy*xy+xz*xz+yz*yz);
  let adjugateNorm2=c00*c00+c11*c11+c22*c22+2.0*(c01*c01+c02*c02+c12*c12);
  let conditionEstimate=sqrt(matrixNorm2*adjugateNorm2)/determinant;
  let determinantTolerance=bitcast<f32>(params.determinantToleranceBits);
  let maximumConditionNumber=bitcast<f32>(params.maximumConditionNumberBits);
  let determinantFloor=determinantTolerance*trace*trace*trace;
  if(!finite(determinant)||!finite(trace)||determinant<=determinantFloor
    ||!finite(conditionEstimate)||conditionEstimate>maximumConditionNumber){
    velocities[row]=vec4f(0.0);rowStatus[row]=FALLBACK;return;
  }
  let velocity=vec3f(
    c00*b.x+c01*b.y+c02*b.z,
    c01*b.x+c11*b.y+c12*b.z,
    c02*b.x+c12*b.y+c22*b.z
  )/determinant;
  if(!finite(velocity.x)||!finite(velocity.y)||!finite(velocity.z)){
    rowStatus[row]=NONFINITE_SOLUTION;return;
  }
  velocities[row]=vec4f(velocity,1.0);rowStatus[row]=0u;
}

`;

export const octreePowerVelocityPrepareShader = /* wgsl */ `
struct VelocityParams {
  rowCount:u32, faceCount:u32, incidenceCount:u32, rowCapacity:u32,
  maximumIncidencePerRow:u32, generation:u32, determinantToleranceBits:u32, maximumConditionNumberBits:u32,
}
struct VelocityControl {
  flags:u32, firstError:u32, rowCount:u32, faceCount:u32,
  incidenceCount:u32, reconstructedCount:u32, fallbackCount:u32, generation:u32,
}
@group(0) @binding(0) var<uniform> params:VelocityParams;
@group(0) @binding(1) var<storage,read_write> control:VelocityControl;
const INVALID:u32=0xffffffffu;
const CAPACITY:u32=1u;
@compute @workgroup_size(1) fn preparePowerVelocity(){
  control.flags=0u;control.firstError=INVALID;control.rowCount=params.rowCount;control.faceCount=params.faceCount;
  control.incidenceCount=params.incidenceCount;control.reconstructedCount=0u;control.fallbackCount=0u;control.generation=params.generation;
  if(params.rowCount>params.rowCapacity){control.flags=CAPACITY;control.firstError=0u;}
}
`;

export const octreePowerVelocityPrepareFromFaceControlShader = /* wgsl */ `
struct VelocityParams {
  rowCount:u32, faceCount:u32, incidenceCount:u32, rowCapacity:u32,
  maximumIncidencePerRow:u32, generation:u32, determinantToleranceBits:u32, maximumConditionNumberBits:u32,
}
struct VelocityControl {
  flags:u32, firstError:u32, rowCount:u32, faceCount:u32,
  incidenceCount:u32, reconstructedCount:u32, fallbackCount:u32, generation:u32,
}
@group(0) @binding(0) var<uniform> params:VelocityParams;
@group(0) @binding(1) var<storage,read_write> control:VelocityControl;
@group(0) @binding(2) var<storage,read> faceControl:array<u32>;
@group(0) @binding(3) var<storage,read> projectionControl:array<u32>;
const INVALID:u32=0xffffffffu;
const FACE_VALID:u32=0x80000000u;
const CAPACITY:u32=1u;
const INVALID_SOURCE:u32=128u;
@compute @workgroup_size(1) fn preparePowerVelocityFromFaceControl(){
  control.flags=0u;control.firstError=INVALID;control.rowCount=params.rowCount;control.faceCount=params.faceCount;
  control.incidenceCount=params.incidenceCount;control.reconstructedCount=0u;control.fallbackCount=0u;control.generation=params.generation;
  if(params.rowCount>params.rowCapacity){control.flags=CAPACITY;control.firstError=0u;return;}
  if(arrayLength(&faceControl)<9u||faceControl[8]!=FACE_VALID||faceControl[3]!=0u
    ||faceControl[7]!=params.generation){control.flags=INVALID_SOURCE;control.firstError=0u;}
}
@compute @workgroup_size(1) fn preparePowerVelocityFromProjectedFaceControl(){
  control.flags=0u;control.firstError=INVALID;control.rowCount=params.rowCount;control.faceCount=params.faceCount;
  control.incidenceCount=params.incidenceCount;control.reconstructedCount=0u;control.fallbackCount=0u;control.generation=params.generation;
  if(params.rowCount>params.rowCapacity){control.flags=CAPACITY;control.firstError=0u;return;}
  if(arrayLength(&faceControl)<9u){control.flags=INVALID_SOURCE;control.firstError=1u;return;}
  if(faceControl[8]!=FACE_VALID){control.flags=INVALID_SOURCE;control.firstError=2u;return;}
  if(faceControl[3]!=0u){control.flags=INVALID_SOURCE;control.firstError=3u;return;}
  if(faceControl[7]!=params.generation){control.flags=INVALID_SOURCE;control.firstError=4u;return;}
  if(arrayLength(&projectionControl)<7u){control.flags=INVALID_SOURCE;control.firstError=5u;return;}
  if(projectionControl[0]!=0xc0000000u){control.flags=INVALID_SOURCE;control.firstError=6u;return;}
  if(projectionControl[6]!=params.faceCount){control.flags=INVALID_SOURCE;control.firstError=7u;}
}
`;

export const octreePowerVelocityPublishShader = /* wgsl */ `
struct PowerFaceRecord {
  negativeRow:u32, positiveRow:u32, geometryCode:u32, flags:u32,
  normalVelocity:f32, area:f32, inverseDistance:f32, openFraction:f32,
}
struct RowWork { faceCount:u32, incidenceCount:u32, faceOffset:u32, incidenceOffset:u32 }
struct PowerIncidence { face:u32, sign:i32 }
struct VelocityParams {
  rowCount:u32, faceCount:u32, incidenceCount:u32, rowCapacity:u32,
  maximumIncidencePerRow:u32, generation:u32, determinantToleranceBits:u32, maximumConditionNumberBits:u32,
}
struct VelocityControl {
  flags:u32, firstError:u32, rowCount:u32, faceCount:u32,
  incidenceCount:u32, reconstructedCount:u32, fallbackCount:u32, generation:u32,
}
@group(0) @binding(0) var<uniform> params:VelocityParams;
@group(0) @binding(1) var<storage,read> faces:array<PowerFaceRecord>;
@group(0) @binding(2) var<storage,read> faceNormals:array<vec4f>;
@group(0) @binding(3) var<storage,read> incidenceRows:array<RowWork>;
@group(0) @binding(4) var<storage,read> incidences:array<PowerIncidence>;
@group(0) @binding(5) var<storage,read> rowStatus:array<u32>;
@group(0) @binding(6) var<storage,read_write> control:VelocityControl;
const INVALID:u32=0xffffffffu;
const VALID:u32=0x80000000u;
const CAPACITY:u32=1u;
const FALLBACK:u32=0x80000000u;
const ILL_CONDITIONED:u32=256u;
@compute @workgroup_size(1) fn publishPowerVelocity(){
  if(control.flags!=0u){control.reconstructedCount=0u;return;}
  if(params.rowCount>params.rowCapacity||params.rowCount>arrayLength(&rowStatus)
    ||params.rowCount+1u>arrayLength(&incidenceRows)||params.faceCount>arrayLength(&faces)
    ||params.faceCount>arrayLength(&faceNormals)||params.incidenceCount>arrayLength(&incidences)){
    control.flags=CAPACITY;control.firstError=0u;control.reconstructedCount=0u;return;
  }
  var errors=0u;var firstError=INVALID;var fallbackCount=0u;
  for(var row=0u;row<params.rowCount;row+=1u){let status=rowStatus[row];
    if(status==FALLBACK){fallbackCount+=1u;errors|=ILL_CONDITIONED;firstError=min(firstError,row);}
    else if(status!=0u){errors|=status;firstError=min(firstError,row);}
  }
  control.fallbackCount=fallbackCount;control.firstError=firstError;
  if(errors==0u){control.flags=VALID;control.reconstructedCount=params.rowCount;}
  else{control.flags=errors;control.reconstructedCount=0u;}
}
`;

// Compile-time linkage guard: this module consumes the shared 32-byte face ABI.
void OCTREE_POWER_FACE_RECORD_BYTES;

export type PowerVelocityVector = readonly [number, number, number];

export interface OctreePowerPointSample {
  readonly velocity: PowerVelocityVector;
  readonly mode: "uniform" | "tetrahedron";
  readonly tetrahedron: number;
}

const finiteVelocity = (value: PowerVelocityVector | undefined): value is PowerVelocityVector =>
  value !== undefined && value.length === 3 && value.every(Number.isFinite);

/** Structured path used in ordinary Cartesian regions. Corner bit order is x + 2y + 4z. */
export function trilinearOctreePowerVelocity(
  corners: readonly PowerVelocityVector[],
  coordinates: PowerVec3,
): PowerVelocityVector {
  if (corners.length !== 8 || corners.some((value) => !finiteVelocity(value))
    || coordinates.some((value) => !Number.isFinite(value) || value < 0 || value > 1)) {
    throw new RangeError("Uniform power interpolation needs eight finite corners and coordinates in [0,1]");
  }
  const result = [0, 0, 0];
  for (let corner = 0; corner < 8; corner += 1) {
    const weight = ((corner & 1) ? coordinates[0] : 1 - coordinates[0])
      * ((corner & 2) ? coordinates[1] : 1 - coordinates[1])
      * ((corner & 4) ? coordinates[2] : 1 - coordinates[2]);
    for (let axis = 0; axis < 3; axis += 1) result[axis] += weight * corners[corner][axis];
  }
  return result as [number, number, number];
}

function tetrahedronCoordinates(point: PowerVec3, a: PowerVec3, b: PowerVec3, c: PowerVec3): readonly [number, number, number, number] | undefined {
  const cross = (left: PowerVec3, right: PowerVec3): PowerVec3 => [
    left[1] * right[2] - left[2] * right[1], left[2] * right[0] - left[0] * right[2],
    left[0] * right[1] - left[1] * right[0],
  ];
  const dot = (left: PowerVec3, right: PowerVec3) => left[0] * right[0] + left[1] * right[1] + left[2] * right[2];
  const determinant = dot(a, cross(b, c));
  if (!Number.isFinite(determinant) || Math.abs(determinant) <= 1e-12) return undefined;
  const wa = dot(point, cross(b, c)) / determinant;
  const wb = dot(a, cross(point, c)) / determinant;
  const wc = dot(a, cross(b, point)) / determinant;
  return [1 - wa - wb - wc, wa, wb, wc];
}

/**
 * CPU oracle for the paper's Section 5 transition interpolant. `point` and
 * catalog geometry are in canonical anchor-size coordinates. Catalog order is
 * the deterministic containment tie-break on shared tetrahedron boundaries.
 */
export function sampleOctreePowerCatalogVelocity(
  entry: OctreePowerCatalogEntry,
  tetrahedronVertexData: ArrayLike<number>,
  point: PowerVec3,
  anchorVelocity: PowerVelocityVector,
  neighborVelocities: readonly (PowerVelocityVector | undefined)[],
  containmentTolerance = 2e-6,
): OctreePowerPointSample {
  if (entry.uniform) throw new RangeError("Uniform catalog entries require the structured trilinear path");
  const selectorCount = Math.floor(tetrahedronVertexData.length / 4);
  if (tetrahedronVertexData.length % 4 !== 0 || selectorCount > 256 || !finiteVelocity(anchorVelocity) || neighborVelocities.length < selectorCount
    || point.some((value) => !Number.isFinite(value))) throw new RangeError("Invalid transition interpolation input");
  for (let tetrahedron = 0; tetrahedron < entry.tetrahedra.length; tetrahedron += 1) {
    const selectors = entry.tetrahedra[tetrahedron];
    const velocities = selectors.map((selector) => neighborVelocities[selector]);
    const weights = tetrahedronCoordinates(point, ...selectors.map((selector) => [tetrahedronVertexData[selector * 4],
      tetrahedronVertexData[selector * 4 + 1], tetrahedronVertexData[selector * 4 + 2]] as PowerVec3) as [PowerVec3, PowerVec3, PowerVec3]);
    if (!weights || weights.some((weight) => weight < -containmentTolerance || weight > 1 + containmentTolerance)) continue;
    if (velocities.some((value) => !finiteVelocity(value))) {
      throw new RangeError(`Containing local Delaunay tetrahedron ${tetrahedron} has an unavailable velocity vertex`);
    }
    const velocity = [0, 0, 0];
    const values = [anchorVelocity, ...velocities] as PowerVelocityVector[];
    for (let vertex = 0; vertex < 4; vertex += 1) for (let axis = 0; axis < 3; axis += 1) {
      velocity[axis] += weights[vertex] * values[vertex][axis];
    }
    return { velocity: velocity as [number, number, number], mode: "tetrahedron", tetrahedron };
  }
  throw new RangeError("No containing local Delaunay tetrahedron for Section 5 velocity interpolation");
}

export const OCTREE_POWER_SAMPLE_QUERY_BYTES = 48;
export const OCTREE_POWER_SAMPLE_CONTROL_BYTES = 32;
export const OCTREE_POWER_SAMPLE_VALID = 0x8000_0000;
/** Query flags bits 8..13 carry the resolved world-to-catalog cube transform. */
export const OCTREE_POWER_SAMPLE_TRANSFORM_SHIFT = 8;
export const OCTREE_POWER_SAMPLE_ERROR = Object.freeze({
  capacity: 1, invalidQuery: 2, nonfinite: 4, noContainingSimplex: 8,
} as const);

export interface OctreePowerVelocitySamplePlan {
  readonly queryCapacity: number;
  readonly resultBytes: number;
  readonly statusBytes: number;
  readonly allocatedBytes: number;
}

export function planOctreePowerVelocitySamples(queryCapacityValue: number): OctreePowerVelocitySamplePlan {
  const queryCapacity = positiveInteger(queryCapacityValue, "Power velocity sample capacity");
  const resultBytes = queryCapacity * 16, statusBytes = queryCapacity * 4;
  return { queryCapacity, resultBytes, statusBytes,
    allocatedBytes: resultBytes + statusBytes + OCTREE_POWER_SAMPLE_CONTROL_BYTES + 16 };
}

export interface OctreePowerVelocitySampleControl {
  readonly flags: number;
  readonly firstError: number;
  readonly queryCount: number;
  readonly interpolatedCount: number;
  readonly uniformCount: number;
  readonly tetrahedronCount: number;
  readonly noContainingSimplexCount: number;
  readonly generation: number;
}

export function unpackOctreePowerVelocitySampleControl(words: ArrayLike<number>): OctreePowerVelocitySampleControl {
  if (words.length < 8) throw new RangeError("Power velocity sample control needs eight words");
  return { flags: Number(words[0]) >>> 0, firstError: Number(words[1]) >>> 0, queryCount: Number(words[2]) >>> 0,
    interpolatedCount: Number(words[3]) >>> 0, uniformCount: Number(words[4]) >>> 0,
    tetrahedronCount: Number(words[5]) >>> 0, noContainingSimplexCount: Number(words[6]) >>> 0, generation: Number(words[7]) >>> 0 };
}

/** Indexed Stage-B sampler; query vertex ranges are transient and may be built directly from CSR incidence. */
export class WebGPUOctreePowerVelocitySampler {
  readonly plan: OctreePowerVelocitySamplePlan;
  readonly results: GPUBuffer;
  readonly statuses: GPUBuffer;
  readonly control: GPUBuffer;
  private readonly params: GPUBuffer;
  private readonly preparePipeline: GPUComputePipeline;
  private readonly pipeline: GPUComputePipeline;
  private readonly publishPipeline: GPUComputePipeline;
  private destroyed = false;

  constructor(private readonly device: GPUDevice, queryCapacity: number, private readonly topology: OctreePowerTopologySource) {
    if (!topology.catalogTetrahedra || !topology.catalogTetrahedronVertices) {
      throw new RangeError("Stage-B velocity sampling requires catalog tetrahedron buffers");
    }
    this.plan = planOctreePowerVelocitySamples(queryCapacity);
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    this.results = device.createBuffer({ label: "Octree power sampled velocities", size: this.plan.resultBytes, usage: storage });
    this.statuses = device.createBuffer({ label: "Octree power sample statuses", size: this.plan.statusBytes, usage: storage });
    this.control = device.createBuffer({ label: "Octree power sample control", size: OCTREE_POWER_SAMPLE_CONTROL_BYTES, usage: storage });
    this.params = device.createBuffer({ label: "Octree power sample params", size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const marker = "(query.flags&0x10000000u)";
    const shaderModule = device.createShaderModule({ label: "Octree power Stage-B sampler", code: octreePowerVelocitySampleShader
      .replace("let query=queries[index];",
        "let query=queries[index];if((query.flags&0x20000000u)!=0u){results[query.output]=vec4f(0.0,0.0,0.0,1.0);statuses[query.output]=VALID|0x20000000u;atomicAdd(&control.interpolated,1u);return;}")
      .replace("statuses[query.output]=VALID;", `statuses[query.output]=VALID|${marker};`)
      .replace("statuses[query.output]=VALID|local;", `statuses[query.output]=VALID|local|${marker};`) });
    this.pipeline = device.createComputePipeline({ label: "Sample octree power velocity", layout: "auto",
      compute: { module: shaderModule, entryPoint: "samplePowerVelocity" } });
    this.preparePipeline = device.createComputePipeline({ label: "Prepare octree power velocity samples", layout: "auto",
      compute: { module: shaderModule, entryPoint: "preparePowerVelocitySamples" } });
    this.publishPipeline = device.createComputePipeline({ label: "Publish octree power velocity samples", layout: "auto",
      compute: { module: shaderModule, entryPoint: "publishPowerVelocitySamples" } });
  }

  encode(encoder: GPUCommandEncoder, queries: GPUBuffer | GPUBufferBinding,
    vertexVelocities: GPUBuffer | GPUBufferBinding, queryCountValue: number, generationValue = 0): void {
    if (this.destroyed) throw new Error("Octree power velocity sampler is destroyed");
    const queryCount = nonNegativeInteger(queryCountValue, "Power velocity sample count");
    const generation = nonNegativeInteger(generationValue, "Power velocity sample generation");
    this.device.queue.writeBuffer(this.params, 0, new Uint32Array([queryCount, this.plan.queryCapacity, generation, 0]));
    const resource = (buffer: GPUBuffer | GPUBufferBinding): GPUBufferBinding => "buffer" in buffer ? buffer : ({ buffer });
    const entries: GPUBindGroupEntry[] = [
      { binding: 0, resource: resource(this.params) }, { binding: 1, resource: resource(queries) },
      { binding: 2, resource: resource(vertexVelocities) }, { binding: 3, resource: resource(this.topology.catalogTetrahedronVertices!) },
      { binding: 4, resource: resource(this.topology.catalogTetrahedra!) }, { binding: 5, resource: resource(this.results) },
      { binding: 6, resource: resource(this.statuses) }, { binding: 7, resource: resource(this.control) },
    ];
    { const group = this.device.createBindGroup({ layout: this.preparePipeline.getBindGroupLayout(0), entries:
      entries.filter((entry) => entry.binding === 0 || entry.binding === 7) });
    const pass = encoder.beginComputePass({ label: "Prepare power velocity point samples" });
    pass.setPipeline(this.preparePipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(1); pass.end(); }
    if (queryCount > 0) {
      const group = this.device.createBindGroup({ layout: this.pipeline.getBindGroupLayout(0), entries });
      const pass = encoder.beginComputePass({ label: "Sample power velocity at indexed points" });
      pass.setPipeline(this.pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(Math.ceil(Math.min(queryCount, this.plan.queryCapacity) / 64)); pass.end();
    }
    const publishGroup = this.device.createBindGroup({ layout: this.publishPipeline.getBindGroupLayout(0), entries:
      entries.filter((entry) => [0, 1, 5, 6, 7].includes(entry.binding)) });
    const publish = encoder.beginComputePass({ label: "Publish power velocity samples" });
    publish.setPipeline(this.publishPipeline); publish.setBindGroup(0, publishGroup); publish.dispatchWorkgroups(1); publish.end();
  }

  destroy(): void { if (this.destroyed) return; this.destroyed = true;
    this.results.destroy(); this.statuses.destroy(); this.control.destroy(); this.params.destroy(); }
}

export const octreePowerVelocitySampleShader = /* wgsl */ `
struct Params { queryCount:u32, queryCapacity:u32, generation:u32, pad:u32 }
struct Query { firstFace:u32, faceCount:u32, firstTetrahedron:u32, tetrahedronCount:u32, flags:u32, vertexStart:u32, vertexCount:u32, output:u32, point:vec4f }
struct TetraVertex { offsetSize:vec4f }
struct TetraHeader { first:u32, count:u32, flags:u32 }
struct Control { flags:atomic<u32>, firstError:atomic<u32>, queryCount:u32, interpolated:atomic<u32>, uniform:atomic<u32>, tetrahedron:atomic<u32>, noContainingSimplex:atomic<u32>, generation:u32 }
@group(0) @binding(0) var<uniform> params:Params;
@group(0) @binding(1) var<storage,read> queries:array<Query>;
@group(0) @binding(2) var<storage,read> vertexVelocities:array<vec4f>;
@group(0) @binding(3) var<storage,read> vertices:array<TetraVertex>;
@group(0) @binding(4) var<storage,read> tetrahedra:array<u32>;
@group(0) @binding(5) var<storage,read_write> results:array<vec4f>;
@group(0) @binding(6) var<storage,read_write> statuses:array<u32>;
@group(0) @binding(7) var<storage,read_write> control:Control;
const VALID:u32=0x80000000u;const CAPACITY:u32=1u;const INVALID_QUERY:u32=2u;const NONFINITE:u32=4u;const NO_CONTAINING_SIMPLEX:u32=8u;
const UNIFORM:u32=${OCTREE_POWER_CATALOG_ENTRY_UNIFORM}u;
fn finite(value:f32)->bool{return (bitcast<u32>(value)&0x7f800000u)!=0x7f800000u;}
fn fail(query:u32,output:u32,flag:u32){atomicOr(&control.flags,flag);atomicMin(&control.firstError,query);if(flag==NO_CONTAINING_SIMPLEX){atomicAdd(&control.noContainingSimplex,1u);}if(output<arrayLength(&results)){results[output]=vec4f(0.0);}if(output<arrayLength(&statuses)){statuses[output]=flag;}}
fn velocityValid(value:vec4f)->bool{return value.w>0.0&&finite(value.x)&&finite(value.y)&&finite(value.z);}
fn tetraWeights(point:vec3f,a:vec3f,b:vec3f,c:vec3f)->vec4f{let determinant=dot(a,cross(b,c));if(!finite(determinant)||abs(determinant)<=1e-10){return vec4f(-2.0);}let wa=dot(point,cross(b,c))/determinant;let wb=dot(a,cross(point,c))/determinant;let wc=dot(a,cross(b,point))/determinant;return vec4f(1.0-wa-wb-wc,wa,wb,wc);}
fn contained(weights:vec4f)->bool{return all(weights>=vec4f(-2e-6))&&all(weights<=vec4f(1.000002));}
fn powerTransform(value:vec3f,code:u32)->vec3f{let permutation=(code/8u)%6u;var result=value;if(permutation==1u){result=value.xzy;}else if(permutation==2u){result=value.yxz;}else if(permutation==3u){result=value.yzx;}else if(permutation==4u){result=value.zxy;}else if(permutation==5u){result=value.zyx;}let bits=code&7u;return result*vec3f(select(1.0,-1.0,(bits&1u)!=0u),select(1.0,-1.0,(bits&2u)!=0u),select(1.0,-1.0,(bits&4u)!=0u));}
@compute @workgroup_size(1) fn preparePowerVelocitySamples(){atomicStore(&control.flags,0u);atomicStore(&control.firstError,0xffffffffu);control.queryCount=params.queryCount;atomicStore(&control.interpolated,0u);atomicStore(&control.uniform,0u);atomicStore(&control.tetrahedron,0u);atomicStore(&control.noContainingSimplex,0u);control.generation=params.generation;}
@compute @workgroup_size(64) fn samplePowerVelocity(@builtin(global_invocation_id) gid:vec3u){let index=gid.x;if(index>=params.queryCount||index>=params.queryCapacity){return;}if(params.queryCount>params.queryCapacity||index>=arrayLength(&queries)||index>=arrayLength(&results)||index>=arrayLength(&statuses)){return;}let query=queries[index];if(query.output>=arrayLength(&results)||query.output>=arrayLength(&statuses)){fail(index,query.output,INVALID_QUERY);return;}if(query.firstFace>arrayLength(&vertices)||query.faceCount>arrayLength(&vertices)-query.firstFace||query.firstTetrahedron>arrayLength(&tetrahedra)||query.tetrahedronCount>arrayLength(&tetrahedra)-query.firstTetrahedron||query.vertexStart>arrayLength(&vertexVelocities)||query.vertexCount>arrayLength(&vertexVelocities)-query.vertexStart){fail(index,query.output,INVALID_QUERY);return;}if((query.flags&UNIFORM)!=0u){if(query.vertexCount<8u||any(query.point.xyz<vec3f(0.0))||any(query.point.xyz>vec3f(1.0))){fail(index,query.output,INVALID_QUERY);return;}var value=vec3f(0.0);for(var corner=0u;corner<8u;corner+=1u){let velocity=vertexVelocities[query.vertexStart+corner];if(!velocityValid(velocity)){fail(index,query.output,NONFINITE);return;}let weight=select(1.0-query.point.x,query.point.x,(corner&1u)!=0u)*select(1.0-query.point.y,query.point.y,(corner&2u)!=0u)*select(1.0-query.point.z,query.point.z,(corner&4u)!=0u);value+=weight*velocity.xyz;}results[query.output]=vec4f(value,1.0);statuses[query.output]=VALID;atomicAdd(&control.uniform,1u);atomicAdd(&control.interpolated,1u);return;}if(query.vertexCount<query.faceCount+1u){fail(index,query.output,INVALID_QUERY);return;}let point=powerTransform(query.point.xyz,(query.flags>>8u)&63u);let anchor=vertexVelocities[query.vertexStart];if(!velocityValid(anchor)){fail(index,query.output,NONFINITE);return;}for(var local=0u;local<query.tetrahedronCount;local+=1u){let packed=tetrahedra[query.firstTetrahedron+local];let selectors=vec3u(packed&255u,(packed>>8u)&255u,(packed>>16u)&255u);if(any(selectors>=vec3u(query.faceCount))){fail(index,query.output,INVALID_QUERY);return;}let a=vertices[query.firstFace+selectors.x].offsetSize.xyz;let b=vertices[query.firstFace+selectors.y].offsetSize.xyz;let c=vertices[query.firstFace+selectors.z].offsetSize.xyz;let weights=tetraWeights(point,a,b,c);if(!contained(weights)){continue;}let va=vertexVelocities[query.vertexStart+1u+selectors.x];let vb=vertexVelocities[query.vertexStart+1u+selectors.y];let vc=vertexVelocities[query.vertexStart+1u+selectors.z];if(!velocityValid(va)||!velocityValid(vb)||!velocityValid(vc)){fail(index,query.output,NONFINITE);return;}results[query.output]=vec4f(weights.x*anchor.xyz+weights.y*va.xyz+weights.z*vb.xyz+weights.w*vc.xyz,1.0);statuses[query.output]=VALID|local;atomicAdd(&control.tetrahedron,1u);atomicAdd(&control.interpolated,1u);return;}fail(index,query.output,NO_CONTAINING_SIMPLEX);statuses[query.output]=NO_CONTAINING_SIMPLEX|((u32(query.point.w)&0xffffu)<<8u);}
@compute @workgroup_size(1) fn publishPowerVelocitySamples(){if(params.queryCount>params.queryCapacity||params.queryCount>arrayLength(&queries)||params.queryCount>arrayLength(&results)||params.queryCount>arrayLength(&statuses)){atomicOr(&control.flags,CAPACITY);atomicMin(&control.firstError,min(params.queryCount,params.queryCapacity));return;}if(atomicLoad(&control.flags)==0u&&atomicLoad(&control.interpolated)==params.queryCount){atomicStore(&control.flags,VALID);}}
`;
