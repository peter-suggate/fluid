/**
 * Stage A of the transition-aware velocity interpolant from Aanjaneya et al.
 * (2017), Section 5: reconstruct one full vector at each power-cell center by
 * an area-weighted least-squares fit to all incident face-normal velocities.
 *
 * The native input ABI is deliberately narrow and explicit:
 *
 * - `faces` uses the 32-byte `PowerFaceRecord` ABI shared by
 *   `webgpu-octree-power-faces.ts` and `webgpu-octree-power-operator.ts`;
 * - `faceNormals` is a transient/resolved `vec4f` per face. Its xyz lanes are
 *   the unit normal directed from `negativeRow` to `positiveRow` (or outward
 *   for a boundary face). The w lane is ignored;
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
import type { PassBroker } from "./webgpu-pass-broker";
import { OCTREE_POWER_FACE_RECORD_BYTES } from "./octree-power-operator";
import type { OctreePowerCatalogEntry } from "./octree-power-catalog";
import type { PowerVec3 } from "./octree-power-geometry";

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

export interface OctreePowerVelocityControl {
  readonly flags: number;
  readonly firstError: number;
  readonly rowCount: number;
  readonly faceCount: number;
  readonly incidenceCount: number;
  readonly reconstructedCount: number;
  readonly illConditionedCount: number;
  readonly generation: number;
}

export interface OctreePowerVelocitySource {
  readonly plan: OctreePowerVelocityPlan;
  /** xyz is velocity; w is one for a solved fit and zero for a rejected fit. */
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
  const dispatchBytes = 16 + 12;
  return {
    rowCapacity,
    velocityBytes,
    statusBytes,
    allocatedBytes: velocityBytes + statusBytes + OCTREE_POWER_VELOCITY_CONTROL_BYTES + 32 + dispatchBytes,
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
    illConditionedCount: Number(words[6]) >>> 0,
    generation: Number(words[7]) >>> 0,
  };
}

export class WebGPUOctreePowerVelocity {
  readonly plan: OctreePowerVelocityPlan;
  readonly velocities: GPUBuffer;
  readonly control: GPUBuffer;
  private readonly params: GPUBuffer;
  private readonly rowStatus: GPUBuffer;
  private readonly rowDispatchControl: GPUBuffer;
  private readonly rowDispatch: GPUBuffer;
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
    this.rowDispatchControl = device.createBuffer({ label: "Octree power velocity live-row dispatch publication",
      size: 16, usage: storage });
    this.rowDispatch = device.createBuffer({ label: "Octree power velocity indirect-only live-row dispatch",
      size: 12, usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST });
    this.params = device.createBuffer({ label: "Octree power velocity parameters", size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, mappedAtCreation: true });
    new Uint8Array(this.params.getMappedRange()).fill(0); this.params.unmap();
    const pipeline = (label: string, code: string, entryPoint: string) => device.createComputePipeline({ label,
      layout: "auto", compute: { module: device.createShaderModule({ label, code }), entryPoint } });
    this.prepareFromFaceControlPipeline = pipeline("Prepare octree power velocity from face authority",
      octreePowerVelocityPrepareFromFaceControlShader, "preparePowerVelocityFromFaceControl");
    this.prepareFromProjectedFaceControlPipeline = pipeline("Prepare octree power velocity from projected face authority",
      octreePowerVelocityPrepareFromFaceControlShader, "preparePowerVelocityFromProjectedFaceControl");
    this.reconstructPipeline = pipeline("Reconstruct octree power cell velocities", octreePowerVelocityShader, "reconstructPowerVelocity");
    this.publishPipeline = pipeline("Publish octree power cell velocities", octreePowerVelocityPublishShader, "publishPowerVelocity");
  }

  /** Reads `(rowCount, faceCount, incidenceCount)` directly from WP4 control on the GPU. */
  encodeFromFaceControl(broker: PassBroker, input: OctreePowerVelocityInput, faceControl: GPUBuffer, options: {
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
    broker.copyBufferToBuffer(faceControl, 0, this.params, 0, 12);
    this.encodePasses(broker, input, faceControl, options.projectionControl);
  }

  private encodePasses(broker: PassBroker, input: OctreePowerVelocityInput,
    faceControl: GPUBuffer, projectionControl?: GPUBuffer): void {
    const resource = (buffer: GPUBuffer) => ({ buffer });
    const group = (pipeline: GPUComputePipeline, entries: GPUBindGroupEntry[]) => this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0), entries,
    });
    const params = { binding: 0, resource: resource(this.params) };
    let pass = broker.compute({ label: "Prepare power velocity" });
    const prepare = projectionControl ? this.prepareFromProjectedFaceControlPipeline
      : this.prepareFromFaceControlPipeline;
    pass.setPipeline(prepare); pass.setBindGroup(0, group(prepare, [params,
      { binding: 1, resource: resource(this.control) },
      { binding: 2, resource: resource(faceControl) },
      ...(projectionControl ? [{ binding: 3, resource: resource(projectionControl) }] : []),
      { binding: 4, resource: resource(this.rowDispatchControl) }]));
    pass.dispatchWorkgroups(1);
    // Prepare is the sole device authority for the live row count. Copy its
    // validated dispatch into an INDIRECT-only buffer so reconstruction never
    // launches from immutable capacity and invalid generations schedule zero
    // work without exposing stale suffix bytes.
    broker.updateIndirectBuffer(this.rowDispatchControl, 0, this.rowDispatch, 0, 12);
    pass = broker.compute({ label: "Reconstruct live power velocity prefix" });
    pass.setPipeline(this.reconstructPipeline); pass.setBindGroup(0, group(this.reconstructPipeline, [params,
      { binding: 1, resource: resource(input.faces) }, { binding: 2, resource: resource(input.faceNormals) },
      { binding: 3, resource: resource(input.incidenceRows) }, { binding: 4, resource: resource(input.incidences) },
      { binding: 5, resource: resource(this.velocities) }, { binding: 6, resource: resource(this.rowStatus) },
      { binding: 7, resource: resource(this.control) }]));
    pass.dispatchWorkgroupsIndirect(this.rowDispatch, 0);
    pass = broker.compute({ label: "Publish power velocity" });
    pass.setPipeline(this.publishPipeline); pass.setBindGroup(0, group(this.publishPipeline, [params,
      { binding: 1, resource: resource(input.faces) }, { binding: 2, resource: resource(input.faceNormals) },
      { binding: 3, resource: resource(input.incidenceRows) }, { binding: 4, resource: resource(input.incidences) },
      { binding: 5, resource: resource(this.rowStatus) }, { binding: 6, resource: resource(this.control) }]));
    pass.dispatchWorkgroups(1);
  }

  get source(): OctreePowerVelocitySource {
    return { plan: this.plan, velocities: this.velocities, control: this.control };
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true; this.velocities.destroy(); this.rowStatus.destroy(); this.control.destroy();
    this.rowDispatchControl.destroy(); this.rowDispatch.destroy(); this.params.destroy();
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
const ILL_CONDITIONED_FIT:u32=0x80000000u;
fn finite(value:f32)->bool{return (bitcast<u32>(value)&0x7f800000u)!=0x7f800000u;}

@compute @workgroup_size(64) fn reconstructPowerVelocity(@builtin(global_invocation_id) gid:vec3u){
  let row=gid.x;if(velocityControl[0]!=0u||row>=params.rowCount){return;}
  if(params.rowCount>params.rowCapacity||row>=arrayLength(&velocities)||row>=arrayLength(&rowStatus)
    ||row+1u>=arrayLength(&incidenceRows)||params.faceCount>arrayLength(&faces)
    ||params.faceCount>arrayLength(&faceNormals)||params.incidenceCount>arrayLength(&incidences)){
    if(row<arrayLength(&rowStatus)){rowStatus[row]=CAPACITY;}return;
  }
  // The live compact prefix is a complete transaction. Every scheduled row
  // overwrites both payload and status, so no capacity-wide pre-clear or
  // stale-suffix maintenance is part of recurring publication.
  velocities[row]=vec4f(0.0);rowStatus[row]=0u;
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
    velocities[row]=vec4f(0.0);rowStatus[row]=ILL_CONDITIONED_FIT;return;
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

export const octreePowerVelocityPrepareFromFaceControlShader = /* wgsl */ `
struct VelocityParams {
  rowCount:u32, faceCount:u32, incidenceCount:u32, rowCapacity:u32,
  maximumIncidencePerRow:u32, generation:u32, determinantToleranceBits:u32, maximumConditionNumberBits:u32,
}
struct VelocityControl {
  flags:u32, firstError:u32, rowCount:u32, faceCount:u32,
  incidenceCount:u32, reconstructedCount:u32, illConditionedCount:u32, generation:u32,
}
@group(0) @binding(0) var<uniform> params:VelocityParams;
@group(0) @binding(1) var<storage,read_write> control:VelocityControl;
@group(0) @binding(2) var<storage,read> faceControl:array<u32>;
@group(0) @binding(3) var<storage,read> projectionControl:array<u32>;
@group(0) @binding(4) var<storage,read_write> rowDispatch:array<u32>;
const INVALID:u32=0xffffffffu;
const FACE_VALID:u32=0x80000000u;
const CAPACITY:u32=1u;
const INVALID_SOURCE:u32=128u;
fn beginDispatch(){
  if(arrayLength(&rowDispatch)>=3u){rowDispatch[0]=0u;rowDispatch[1]=1u;rowDispatch[2]=1u;}
}
fn publishDispatch(count:u32){
  if(arrayLength(&rowDispatch)<3u){control.flags=CAPACITY;control.firstError=0u;return;}
  let groups=(count+63u)/64u;let x=min(groups,65535u);
  rowDispatch[0]=x;rowDispatch[1]=select(1u,(groups+x-1u)/max(x,1u),x>0u);rowDispatch[2]=1u;
}
@compute @workgroup_size(1) fn preparePowerVelocityFromFaceControl(){
  beginDispatch();
  control.flags=0u;control.firstError=INVALID;control.rowCount=params.rowCount;control.faceCount=params.faceCount;
  control.incidenceCount=params.incidenceCount;control.reconstructedCount=0u;control.illConditionedCount=0u;control.generation=params.generation;
  if(params.rowCount>params.rowCapacity){control.flags=CAPACITY;control.firstError=0u;return;}
  if(arrayLength(&faceControl)<9u||faceControl[8]!=FACE_VALID||faceControl[3]!=0u
    ||faceControl[7]!=params.generation){control.flags=INVALID_SOURCE;control.firstError=0u;return;}
  publishDispatch(params.rowCount);
}
@compute @workgroup_size(1) fn preparePowerVelocityFromProjectedFaceControl(){
  beginDispatch();
  control.flags=0u;control.firstError=INVALID;control.rowCount=params.rowCount;control.faceCount=params.faceCount;
  control.incidenceCount=params.incidenceCount;control.reconstructedCount=0u;control.illConditionedCount=0u;control.generation=params.generation;
  if(params.rowCount>params.rowCapacity){control.flags=CAPACITY;control.firstError=0u;return;}
  if(arrayLength(&faceControl)<9u){control.flags=INVALID_SOURCE;control.firstError=1u;return;}
  if(faceControl[8]!=FACE_VALID){control.flags=INVALID_SOURCE;control.firstError=2u;return;}
  if(faceControl[3]!=0u){control.flags=INVALID_SOURCE;control.firstError=3u;return;}
  if(faceControl[7]!=params.generation){control.flags=INVALID_SOURCE;control.firstError=4u;return;}
  if(arrayLength(&projectionControl)<7u){control.flags=INVALID_SOURCE;control.firstError=5u;return;}
  if(projectionControl[0]!=0xc0000000u){control.flags=INVALID_SOURCE;control.firstError=6u;return;}
  if(projectionControl[6]!=params.faceCount){control.flags=INVALID_SOURCE;control.firstError=7u;return;}
  publishDispatch(params.rowCount);
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
  incidenceCount:u32, reconstructedCount:u32, illConditionedCount:u32, generation:u32,
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
const ILL_CONDITIONED_FIT:u32=0x80000000u;
const ILL_CONDITIONED:u32=256u;
@compute @workgroup_size(1) fn publishPowerVelocity(){
  if(control.flags!=0u){control.reconstructedCount=0u;return;}
  if(params.rowCount>params.rowCapacity||params.rowCount>arrayLength(&rowStatus)
    ||params.rowCount+1u>arrayLength(&incidenceRows)||params.faceCount>arrayLength(&faces)
    ||params.faceCount>arrayLength(&faceNormals)||params.incidenceCount>arrayLength(&incidences)){
    control.flags=CAPACITY;control.firstError=0u;control.reconstructedCount=0u;return;
  }
  var errors=0u;var firstError=INVALID;var illConditionedCount=0u;
  for(var row=0u;row<params.rowCount;row+=1u){let status=rowStatus[row];
    if(status==ILL_CONDITIONED_FIT){illConditionedCount+=1u;errors|=ILL_CONDITIONED;firstError=min(firstError,row);}
    else if(status!=0u){errors|=status;firstError=min(firstError,row);}
  }
  control.illConditionedCount=illConditionedCount;control.firstError=firstError;
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
