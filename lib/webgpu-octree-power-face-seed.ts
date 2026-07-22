/** GPU-resident bridge from the transferred compact Cartesian face field to power faces. */

import { OCTREE_GPU_FACE_INCIDENCE_PER_ROW, type OctreeFaceMirrorSource } from "./webgpu-octree-face-mirror";
import type { OctreePowerFaceSource } from "./webgpu-octree-power-faces";

export const OCTREE_POWER_FACE_SEED_VALID = 0x8000_0000;
export const OCTREE_POWER_FACE_SEED_CONTROL_BYTES = 64;
export const OCTREE_POWER_FACE_SEED_ERROR = Object.freeze({
  source: 1 << 0,
  capacity: 1 << 1,
  incidence: 1 << 2,
  incompleteVector: 1 << 3,
  normal: 1 << 4,
  nonfinite: 1 << 5,
} as const);

export interface OctreePowerFaceSeedPlan {
  readonly rowCapacity: number;
  readonly faceCapacity: number;
  readonly axisFaceCapacity: number;
  readonly velocityBytes: number;
  readonly rowStatusBytes: number;
  readonly axisVelocityBytes: number;
  readonly allocatedBytes: number;
}

export function planOctreePowerFaceSeed(rowCapacity: number, faceCapacity: number, axisFaceCapacity = faceCapacity): OctreePowerFaceSeedPlan {
  if (!Number.isSafeInteger(rowCapacity) || rowCapacity < 1 || !Number.isSafeInteger(faceCapacity) || faceCapacity < 1
    || !Number.isSafeInteger(axisFaceCapacity) || axisFaceCapacity < 1) {
    throw new RangeError("Power-face seed capacities must be positive integers");
  }
  const velocityBytes = rowCapacity * 16;
  const rowStatusBytes = rowCapacity * 4;
  const axisVelocityBytes = axisFaceCapacity * 4;
  return { rowCapacity, faceCapacity, axisFaceCapacity, velocityBytes, rowStatusBytes, axisVelocityBytes,
    allocatedBytes: velocityBytes + rowStatusBytes + axisVelocityBytes + OCTREE_POWER_FACE_SEED_CONTROL_BYTES + 32 };
}

/**
 * The compact axis source has already performed its exact/restriction/
 * prolongation topology transfer.  This bridge reconstructs one bounded full
 * vector per live row from that canonical field, then projects it onto each
 * generalized face normal.  No dense velocity texture or CPU count readback is
 * involved.  Its VALID word is consumed by power-row assembly as a GPU-side
 * publication prerequisite.
 */
export class WebGPUOctreePowerFaceSeed {
  readonly plan: OctreePowerFaceSeedPlan;
  readonly control: GPUBuffer;
  private readonly rowVelocities: GPUBuffer;
  private readonly rowStatus: GPUBuffer;
  private readonly axisVelocityScratch: GPUBuffer;
  private readonly params: GPUBuffer;
  private readonly accelerationParams: GPUBuffer;
  private readonly preparePipeline: GPUComputePipeline;
  private readonly reconstructPipeline: GPUComputePipeline;
  private readonly seedPipeline: GPUComputePipeline;
  private readonly accelerationPipeline: GPUComputePipeline;
  private readonly publishPipeline: GPUComputePipeline;
  private readonly reversePreparePipeline: GPUComputePipeline;
  private readonly reverseReconstructPipeline: GPUComputePipeline;
  private readonly reversePublishPipeline: GPUComputePipeline;
  private readonly reverseOverridePipeline: GPUComputePipeline;
  private readonly reverseFinalizePipeline: GPUComputePipeline;
  private readonly reverseRejectPipeline: GPUComputePipeline;
  private readonly reverseCommitPipeline: GPUComputePipeline;
  private destroyed = false;

  constructor(
    private readonly device: GPUDevice,
    private readonly axis: OctreeFaceMirrorSource,
    private readonly power: OctreePowerFaceSource,
  ) {
    this.plan = planOctreePowerFaceSeed(power.plan.rowCapacity, power.plan.faceCapacity, axis.plan.faceCapacity);
    if (axis.plan.rowCapacity < power.plan.rowCapacity) throw new RangeError("Axis seed row capacity is too small");
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    this.rowVelocities = device.createBuffer({ label: "Power-face seed row velocities", size: this.plan.velocityBytes, usage: storage });
    this.rowStatus = device.createBuffer({ label: "Power-face seed row status", size: this.plan.rowStatusBytes, usage: storage });
    this.axisVelocityScratch = device.createBuffer({ label: "Power-to-axis velocity publication scratch",
      size: this.plan.axisVelocityBytes, usage: storage });
    this.control = device.createBuffer({ label: "Power-face seed control", size: OCTREE_POWER_FACE_SEED_CONTROL_BYTES, usage: storage });
    this.params = device.createBuffer({ label: "Power-face seed parameters", size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.accelerationParams = device.createBuffer({ label: "Power-face acceleration parameters", size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(this.params, 0, new Uint32Array([
      this.plan.rowCapacity, this.plan.faceCapacity, axis.plan.faceCapacity, OCTREE_GPU_FACE_INCIDENCE_PER_ROW,
    ]));
    const shaderModule = device.createShaderModule({ label: "Power-face canonical velocity seed", code: octreePowerFaceSeedShader });
    const pipeline = (entryPoint: string) => device.createComputePipeline({ label: entryPoint, layout: "auto",
      compute: { module: shaderModule, entryPoint } });
    this.preparePipeline = pipeline("preparePowerFaceSeed");
    this.reconstructPipeline = pipeline("reconstructAxisRowVelocity");
    this.seedPipeline = pipeline("seedPowerFaceVelocity");
    this.accelerationPipeline = pipeline("applyPowerFaceAcceleration");
    this.publishPipeline = pipeline("publishPowerFaceSeed");
    this.reversePreparePipeline = pipeline("preparePowerToAxis");
    this.reverseReconstructPipeline = pipeline("reconstructPowerRowVelocity");
    this.reversePublishPipeline = pipeline("publishAxisFaceVelocity");
    this.reverseOverridePipeline = pipeline("overrideHomologousAxisVelocity");
    this.reverseFinalizePipeline = pipeline("publishPowerToAxis");
    this.reverseRejectPipeline = pipeline("rejectFailedAuthoritativePowerToAxis");
    this.reverseCommitPipeline = pipeline("commitPowerToAxis");
  }

  /** Conservatively republishes the projected power field for compact regular-face consumers. */
  encodePowerToAxis(encoder: GPUCommandEncoder, projectedOperatorControl: GPUBuffer, authoritative = false): void {
    if (this.destroyed) throw new Error("Power-face velocity seed is destroyed");
    const group = (pipeline: GPUComputePipeline, bindings: readonly [number, GPUBuffer][]) => this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0), entries: bindings.map(([binding, buffer]) => ({ binding, resource: { buffer } })),
    });
    const run = (label: string, pipeline: GPUComputePipeline, groups: number, bindings: readonly [number, GPUBuffer][]) => {
      const pass = encoder.beginComputePass({ label }); pass.setPipeline(pipeline); pass.setBindGroup(0, group(pipeline, bindings));
      pass.dispatchWorkgroups(groups); pass.end();
    };
    run("Prepare power-to-axis velocity publication", this.reversePreparePipeline, 1,
      [[1, this.axis.control], [4, this.power.control], [9, this.control], [10, projectedOperatorControl]]);
    run("Reconstruct projected power row velocities", this.reverseReconstructPipeline,
      Math.ceil(this.plan.rowCapacity / 64), [[5, this.power.faces],
        [6, this.power.faceNormals], [7, this.rowVelocities], [8, this.rowStatus], [9, this.control],
        [11, this.power.incidenceRows], [12, this.power.incidence]]);
    run("Publish projected power velocity to compact Cartesian faces", this.reversePublishPipeline,
      Math.ceil(this.axis.plan.faceCapacity / 64), [[1, this.axis.control], [2, this.axis.faces],
        [7, this.rowVelocities], [9, this.control], [13, this.axisVelocityScratch]]);
    run("Copy homologous projected power velocities exactly", this.reverseOverridePipeline,
      Math.ceil(this.axis.plan.faceCapacity / 64), [[1, this.axis.control], [2, this.axis.faces],
        [5, this.power.faces], [6, this.power.faceNormals], [9, this.control],
        [11, this.power.incidenceRows], [12, this.power.incidence], [13, this.axisVelocityScratch]]);
    run("Finalize power-to-axis velocity publication", this.reverseFinalizePipeline, 1,
      [[1, this.axis.control], [9, this.control]]);
    if (authoritative) run("Reject failed authoritative power-to-axis publication", this.reverseRejectPipeline, 1,
      [[1, this.axis.control], [9, this.control]]);
    run("Commit power-to-axis velocity publication", this.reverseCommitPipeline,
      Math.ceil(this.axis.plan.faceCapacity / 64), [[1, this.axis.control], [2, this.axis.faces],
        [9, this.control], [13, this.axisVelocityScratch]]);
  }

  encode(encoder: GPUCommandEncoder): void {
    if (this.destroyed) throw new Error("Power-face velocity seed is destroyed");
    const entries = (pipeline: GPUComputePipeline, bindings: readonly [number, GPUBuffer][]) => this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0), entries: bindings.map(([binding, buffer]) => ({ binding, resource: { buffer } })),
    });
    const run = (label: string, pipeline: GPUComputePipeline, groups: number, bindings: readonly [number, GPUBuffer][]) => {
      const pass = encoder.beginComputePass({ label }); pass.setPipeline(pipeline);
      pass.setBindGroup(0, entries(pipeline, bindings)); pass.dispatchWorkgroups(groups); pass.end();
    };
    run("Prepare canonical power-face velocity seed", this.preparePipeline, 1,
      [[0, this.params], [1, this.axis.control], [2, this.axis.faces], [4, this.power.control], [5, this.power.faces],
        [6, this.power.faceNormals], [9, this.control]]);
    run("Reconstruct transferred Cartesian row velocities", this.reconstructPipeline,
      Math.ceil(this.plan.rowCapacity / 64), [[0, this.params], [1, this.axis.control], [2, this.axis.faces],
        [3, this.axis.incidence], [7, this.rowVelocities], [8, this.rowStatus], [9, this.control]]);
    run("Project transferred velocities onto power faces", this.seedPipeline,
      Math.ceil(this.plan.faceCapacity / 64), [[0, this.params], [1, this.axis.control], [2, this.axis.faces],
        [3, this.axis.incidence], [5, this.power.faces], [6, this.power.faceNormals],
        [7, this.rowVelocities], [9, this.control]]);
    run("Publish canonical power-face velocity seed", this.publishPipeline, 1, [[9, this.control]]);
  }

  /** Applies the split external-force term on the native power-face DOFs. */
  encodeAcceleration(
    encoder: GPUCommandEncoder,
    acceleration: readonly [number, number, number],
    dt: number,
  ): void {
    if (this.destroyed) throw new Error("Power-face velocity seed is destroyed");
    if (![...acceleration, dt].every(Number.isFinite) || dt < 0) {
      throw new RangeError("Power-face acceleration and dt must be finite, with dt non-negative");
    }
    this.device.queue.writeBuffer(this.accelerationParams, 0,
      new Float32Array([acceleration[0], acceleration[1], acceleration[2], dt]));
    const group = this.device.createBindGroup({ layout: this.accelerationPipeline.getBindGroupLayout(0), entries: [
      { binding: 4, resource: { buffer: this.power.control } },
      { binding: 5, resource: { buffer: this.power.faces } },
      { binding: 6, resource: { buffer: this.power.faceNormals } },
      { binding: 9, resource: { buffer: this.control } },
      { binding: 14, resource: { buffer: this.accelerationParams } },
    ] });
    const pass = encoder.beginComputePass({ label: "Apply acceleration on authoritative power faces" });
    pass.setPipeline(this.accelerationPipeline); pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(Math.ceil(this.plan.faceCapacity / 64)); pass.end();
  }

  /** Full projected velocity at every compact power-cell centre. */
  get cellVelocities(): GPUBuffer { return this.rowVelocities; }

  destroy(): void {
    if (this.destroyed) return; this.destroyed = true;
    this.rowVelocities.destroy(); this.rowStatus.destroy(); this.axisVelocityScratch.destroy(); this.control.destroy();
    this.params.destroy(); this.accelerationParams.destroy();
  }
}

export const octreePowerFaceSeedShader = /* wgsl */ `
struct Params { rowCapacity:u32,powerFaceCapacity:u32,axisFaceCapacity:u32,axisIncidencePerRow:u32 }
struct AxisFace { negativeRow:u32,positiveRow:u32,originX:u32,originY:u32,originZ:u32,axisSpan:u32,normalVelocity:f32,area:f32 }
struct PowerFace { negativeRow:u32,positiveRow:u32,geometryCode:u32,flags:u32,normalVelocity:f32,area:f32,inverseDistance:f32,openFraction:f32 }
struct SeedControl { flags:atomic<u32>,firstError:atomic<u32>,rowCount:atomic<u32>,faceCount:atomic<u32>,seededCount:atomic<u32>,generation:atomic<u32>,valid:atomic<u32>,fallbackCount:atomic<u32>,seedMaxBits:atomic<u32>,axisRowMaxBits:atomic<u32>,projectedRowMaxBits:atomic<u32>,axisOutputMaxBits:atomic<u32>,forwardFlags:atomic<u32>,forwardFirstError:atomic<u32>,forwardSeededCount:atomic<u32>,forwardValid:atomic<u32> }
struct AccelerationParams { acceleration:vec3f,dt:f32 }
@group(0) @binding(0) var<uniform> params:Params;
@group(0) @binding(1) var<storage,read_write> axisControl:array<u32>;
@group(0) @binding(2) var<storage,read_write> axisFaces:array<AxisFace>;
@group(0) @binding(3) var<storage,read> axisIncidence:array<u32>;
@group(0) @binding(4) var<storage,read> powerControl:array<u32>;
@group(0) @binding(5) var<storage,read_write> powerFaces:array<PowerFace>;
@group(0) @binding(6) var<storage,read> powerNormals:array<vec4f>;
@group(0) @binding(7) var<storage,read_write> rowVelocities:array<vec4f>;
@group(0) @binding(8) var<storage,read_write> rowStatus:array<u32>;
@group(0) @binding(9) var<storage,read_write> seed:SeedControl;
@group(0) @binding(10) var<storage,read> operatorControl:array<u32>;
@group(0) @binding(11) var<storage,read> powerRows:array<vec4u>;
@group(0) @binding(12) var<storage,read> powerIncidence:array<vec2u>;
@group(0) @binding(13) var<storage,read_write> axisVelocityScratch:array<f32>;
@group(0) @binding(14) var<uniform> accelerationParams:AccelerationParams;
const INVALID:u32=0xffffffffu;const VALID:u32=0x80000000u;const SOURCE:u32=1u;const CAPACITY:u32=2u;
const BOUNDARY:u32=1u;const OPEN_BOUNDARY:u32=2u;
const INCIDENCE:u32=4u;const INCOMPLETE:u32=8u;const NORMAL:u32=16u;const NONFINITE:u32=32u;
fn finite(v:f32)->bool{return v==v&&abs(v)<=3.402823e38;}
fn fail(error:u32,index:u32){atomicOr(&seed.flags,error);atomicMin(&seed.firstError,index);}
fn rowCount()->u32{return atomicLoad(&seed.rowCount);}fn faceCount()->u32{return atomicLoad(&seed.faceCount);}
fn dominantAxis(n:vec3f)->u32{let a=abs(n);var axis=0u;if(a.y>a.x){axis=1u;}if(a.z>a[axis]){axis=2u;}return axis;}
// Paper Sections 4.2/5: a power face that is homologous to an octree face
// carries the SAME normal-velocity degree of freedom. Both bridges below copy
// that scalar exactly for axis-aligned faces instead of resampling through
// cell-centre vector averages: the averaging round trip (axis -> cell -> power
// before the solve, power -> cell -> axis after projection) acts as a
// [1 2 1]/4 smoother on every face every step and visibly damps sloshing.
// Reconstruction remains for genuinely new power-diagram faces (edge
// neighbours and slanted transition faces). Returns (value, found).
fn homologousAxisVelocity(face:PowerFace,n:vec3f)->vec2f{
  let axis=dominantAxis(n);let nAxis=n[axis];if(abs(nAxis)<0.999){return vec2f(0.0);}
  let axisRows=axisControl[3];let row=face.negativeRow;if(row>=axisRows||row>=arrayLength(&axisIncidence)){return vec2f(0.0);}
  let count=min(axisIncidence[row],params.axisIncidencePerRow);
  for(var local=0u;local<count;local+=1u){let at=axisRows+row*params.axisIncidencePerRow+local;if(at>=arrayLength(&axisIncidence)){break;}
    let faceIndex=axisIncidence[at];if(faceIndex>=axisControl[0]||faceIndex>=arrayLength(&axisFaces)){continue;}
    let axisFace=axisFaces[faceIndex];if((axisFace.axisSpan&3u)!=axis||!finite(axisFace.normalVelocity)){continue;}
    let ordered=axisFace.negativeRow==face.negativeRow&&axisFace.positiveRow==face.positiveRow;
    let swapped=axisFace.negativeRow==face.positiveRow&&axisFace.positiveRow==face.negativeRow;
    if(!ordered&&!swapped){continue;}
    if(face.positiveRow==INVALID){
      // The two same-axis wall faces of one cell share this row pair; the
      // live row slot encodes which side the face lies on.
      let outward=select(-1.0,1.0,axisFace.negativeRow!=INVALID);
      if(nAxis*outward<0.0){continue;}
    }
    return vec2f(axisFace.normalVelocity*nAxis,1.0);
  }
  return vec2f(0.0);
}
fn homologousPowerVelocity(face:AxisFace,axis:u32)->vec2f{
  var row=face.negativeRow;if(row==INVALID){row=face.positiveRow;}
  if(row>=rowCount()||row+1u>=arrayLength(&powerRows)){return vec2f(0.0);}
  let begin=powerRows[row].w;let end=powerRows[row+1u].w;
  for(var cursor=begin;cursor<end;cursor+=1u){if(cursor>=arrayLength(&powerIncidence)){break;}
    let faceIndex=powerIncidence[cursor].x;if(faceIndex>=faceCount()||faceIndex>=arrayLength(&powerFaces)||faceIndex>=arrayLength(&powerNormals)){continue;}
    let power=powerFaces[faceIndex];
    let ordered=power.negativeRow==face.negativeRow&&power.positiveRow==face.positiveRow;
    let swapped=power.negativeRow==face.positiveRow&&power.positiveRow==face.negativeRow;
    if(!ordered&&!swapped){continue;}
    let n=powerNormals[faceIndex].xyz;let nAxis=n[axis];if(abs(nAxis)<0.999){continue;}
    if(face.negativeRow==INVALID||face.positiveRow==INVALID){
      let outward=select(-1.0,1.0,face.negativeRow!=INVALID);
      if(nAxis*outward<0.0){continue;}
    }
    if(!finite(power.normalVelocity)){continue;}
    return vec2f(power.normalVelocity*nAxis,1.0);
  }
  return vec2f(0.0);
}
@compute @workgroup_size(1) fn preparePowerFaceSeed(){atomicStore(&seed.flags,0u);atomicStore(&seed.firstError,INVALID);
  atomicStore(&seed.seededCount,0u);atomicStore(&seed.valid,0u);atomicStore(&seed.fallbackCount,0u);
  atomicStore(&seed.forwardFlags,0u);atomicStore(&seed.forwardFirstError,INVALID);atomicStore(&seed.forwardSeededCount,0u);atomicStore(&seed.forwardValid,0u);
  atomicStore(&seed.seedMaxBits,0u);atomicStore(&seed.axisRowMaxBits,0u);atomicStore(&seed.projectedRowMaxBits,0u);atomicStore(&seed.axisOutputMaxBits,0u);
  if(arrayLength(&powerControl)<9u||arrayLength(&axisControl)<6u){atomicStore(&seed.rowCount,0u);atomicStore(&seed.faceCount,0u);fail(CAPACITY,0u);return;}
  atomicStore(&seed.rowCount,powerControl[0]);atomicStore(&seed.faceCount,powerControl[1]);atomicStore(&seed.generation,powerControl[7]);
  if(powerControl[3]!=0u){fail(SOURCE,1u);}
  if(powerControl[8]!=VALID){fail(SOURCE,2u);}
  if(axisControl[1]!=0u){fail(SOURCE,3u);}
  if(powerControl[0]>params.rowCapacity||powerControl[0]>axisControl[3]||powerControl[1]>params.powerFaceCapacity
    ||powerControl[1]>arrayLength(&powerFaces)||powerControl[1]>arrayLength(&powerNormals)
    ||axisControl[0]>params.axisFaceCapacity||axisControl[0]>arrayLength(&axisFaces)){fail(CAPACITY,0u);}}
@compute @workgroup_size(64) fn reconstructAxisRowVelocity(@builtin(global_invocation_id) gid:vec3u){let row=gid.x;
  if(row>=rowCount()||atomicLoad(&seed.flags)!=0u){return;}if(row>=arrayLength(&rowVelocities)||row>=arrayLength(&rowStatus)){fail(CAPACITY,row);return;}
  let axisRows=axisControl[3];let count=min(axisIncidence[row],params.axisIncidencePerRow);var weighted=vec3f(0.0);var area=vec3f(0.0);
  for(var local=0u;local<count;local+=1u){let at=axisRows+row*params.axisIncidencePerRow+local;if(at>=arrayLength(&axisIncidence)){rowStatus[row]=INCIDENCE;fail(INCIDENCE,row);return;}
    let faceIndex=axisIncidence[at];if(faceIndex>=axisControl[0]||faceIndex>=arrayLength(&axisFaces)){rowStatus[row]=INCIDENCE;fail(INCIDENCE,row);return;}
    let face=axisFaces[faceIndex];let axis=face.axisSpan&3u;if(axis>=3u||!finite(face.area)||face.area<=0.0||!finite(face.normalVelocity)
      ||(face.negativeRow!=row&&face.positiveRow!=row)){rowStatus[row]=INCIDENCE;fail(INCIDENCE,row);return;}
    weighted[axis]+=face.area*face.normalVelocity;area[axis]+=face.area;}
  if(any(area<=vec3f(0.0))){rowStatus[row]=INCOMPLETE;fail(INCOMPLETE,row);return;}
  let velocity=weighted/area;if(!all(vec3<bool>(finite(velocity.x),finite(velocity.y),finite(velocity.z)))){rowStatus[row]=NONFINITE;fail(NONFINITE,row);return;}
  rowVelocities[row]=vec4f(velocity,1.0);rowStatus[row]=0u;atomicMax(&seed.axisRowMaxBits,bitcast<u32>(length(velocity)));}
@compute @workgroup_size(64) fn seedPowerFaceVelocity(@builtin(global_invocation_id) gid:vec3u){let index=gid.x;
  if(index>=faceCount()||atomicLoad(&seed.flags)!=0u){return;}let face=powerFaces[index];if(face.negativeRow>=rowCount()){fail(INCIDENCE,index);return;}
  let n=powerNormals[index].xyz;let n2=dot(n,n);if(!all(vec3<bool>(finite(n.x),finite(n.y),finite(n.z)))||!finite(n2)||abs(n2-1.0)>4e-4){fail(NORMAL,index);return;}
  var velocity=rowVelocities[face.negativeRow].xyz;if(face.positiveRow!=INVALID){if(face.positiveRow>=rowCount()){fail(INCIDENCE,index);return;}velocity=0.5*(velocity+rowVelocities[face.positiveRow].xyz);}
  var normalVelocity=dot(velocity,n);
  let homologous=homologousAxisVelocity(face,n);
  if(homologous.y!=0.0){normalVelocity=homologous.x;}else{atomicAdd(&seed.fallbackCount,1u);}
  // Container walls are not part of the terrain/rigid aperture classifier.
  // Enforce their authored no-through-flow condition here; otherwise the
  // cell-centred reconstruction gives a bottom wall half of the gravity kick
  // and the pressure solve sees only half the hydrostatic head. Open world
  // faces and internal free-surface faces retain the predicted velocity.
  if(face.positiveRow==INVALID&&(face.flags&BOUNDARY)!=0u&&(face.flags&OPEN_BOUNDARY)==0u){normalVelocity=0.0;}
  if(!finite(normalVelocity)){fail(NONFINITE,index);return;}powerFaces[index].normalVelocity=normalVelocity;atomicMax(&seed.seedMaxBits,bitcast<u32>(abs(normalVelocity)));atomicAdd(&seed.seededCount,1u);}
@compute @workgroup_size(1) fn publishPowerFaceSeed(){if(atomicLoad(&seed.flags)==0u&&atomicLoad(&seed.seededCount)==faceCount()){
  atomicStore(&seed.valid,VALID);}else{atomicStore(&seed.valid,0u);}atomicStore(&seed.forwardFlags,atomicLoad(&seed.flags));atomicStore(&seed.forwardFirstError,atomicLoad(&seed.firstError));atomicStore(&seed.forwardSeededCount,atomicLoad(&seed.seededCount));atomicStore(&seed.forwardValid,atomicLoad(&seed.valid));}
@compute @workgroup_size(64) fn applyPowerFaceAcceleration(@builtin(global_invocation_id) gid:vec3u){let index=gid.x;
  if(index>=faceCount()||atomicLoad(&seed.valid)!=VALID){return;}
  if(arrayLength(&powerControl)<9u||powerControl[7]!=atomicLoad(&seed.generation)){fail(SOURCE,index);atomicStore(&seed.valid,0u);return;}
  let face=powerFaces[index];let closed=face.positiveRow==INVALID&&(face.flags&BOUNDARY)!=0u&&(face.flags&OPEN_BOUNDARY)==0u;
  if(closed){powerFaces[index].normalVelocity=0.0;return;}
  let n=powerNormals[index].xyz;let force=dot(accelerationParams.acceleration,n)*accelerationParams.dt;
  let normalVelocity=face.normalVelocity+force;
  if(!finite(force)||!finite(normalVelocity)){fail(NONFINITE,index);atomicStore(&seed.valid,0u);return;}
  powerFaces[index].normalVelocity=normalVelocity;atomicMax(&seed.seedMaxBits,bitcast<u32>(abs(normalVelocity)));}
@compute @workgroup_size(1) fn preparePowerToAxis(){atomicStore(&seed.flags,0u);atomicStore(&seed.firstError,INVALID);
  atomicStore(&seed.seededCount,0u);atomicStore(&seed.valid,0u);atomicStore(&seed.fallbackCount,0u);
  atomicStore(&seed.projectedRowMaxBits,0u);atomicStore(&seed.axisOutputMaxBits,0u);
  if(arrayLength(&operatorControl)<7u||arrayLength(&powerControl)<9u||arrayLength(&axisControl)<6u){fail(CAPACITY,0u);return;}
  atomicStore(&seed.rowCount,powerControl[0]);atomicStore(&seed.faceCount,powerControl[1]);atomicStore(&seed.generation,powerControl[7]);
  if(operatorControl[0]!=0xc0000000u||operatorControl[6]!=powerControl[1]||powerControl[8]!=VALID||axisControl[1]!=0u){fail(SOURCE,0u);}}
@compute @workgroup_size(64) fn reconstructPowerRowVelocity(@builtin(global_invocation_id) gid:vec3u){let row=gid.x;
  if(row>=rowCount()||atomicLoad(&seed.flags)!=0u){return;}if(row+1u>=arrayLength(&powerRows)||row>=arrayLength(&rowVelocities)||row>=arrayLength(&rowStatus)){fail(CAPACITY,row);return;}
  let begin=powerRows[row].w;let end=powerRows[row+1u].w;if(begin>end||end>arrayLength(&powerIncidence)){rowStatus[row]=INCIDENCE;fail(INCIDENCE,row);return;}
  var xx=0.0;var xy=0.0;var xz=0.0;var yy=0.0;var yz=0.0;var zz=0.0;var b=vec3f(0.0);var maxNormalSpeed=0.0;
  for(var cursor=begin;cursor<end;cursor+=1u){let item=powerIncidence[cursor];let faceIndex=item.x;if(faceIndex>=faceCount()){rowStatus[row]=INCIDENCE;fail(INCIDENCE,row);return;}
    let face=powerFaces[faceIndex];let positive=bitcast<i32>(item.y)<0;let expected=select(face.negativeRow,face.positiveRow,positive);
    if(expected!=row){rowStatus[row]=INCIDENCE;fail(INCIDENCE,row);return;}let orientation=select(1.0,-1.0,positive);
    let n=orientation*powerNormals[faceIndex].xyz;let u=orientation*face.normalVelocity;let w=face.area;
    if(!all(vec3<bool>(finite(n.x),finite(n.y),finite(n.z)))||!finite(u)||!finite(w)||w<=0.0){rowStatus[row]=NONFINITE;fail(NONFINITE,row);return;}
    xx+=w*n.x*n.x;xy+=w*n.x*n.y;xz+=w*n.x*n.z;yy+=w*n.y*n.y;yz+=w*n.y*n.z;zz+=w*n.z*n.z;b+=w*n*u;maxNormalSpeed=max(maxNormalSpeed,abs(u));}
  let c00=yy*zz-yz*yz;let c01=xz*yz-xy*zz;let c02=xy*yz-xz*yy;let c11=xx*zz-xz*xz;let c12=xy*xz-xx*yz;let c22=xx*yy-xy*xy;
  let determinant=xx*c00+xy*c01+xz*c02;let trace=xx+yy+zz;
  // A finite inverse is not sufficient: a nearly coplanar normal set can
  // amplify harmless face-normal data by orders of magnitude. Fail the whole
  // publication back to the axis rollback unless the area-weighted normal
  // matrix has a useful three-dimensional span.
  if(!finite(determinant)||determinant<=1e-3*trace*trace*trace){rowStatus[row]=INCOMPLETE;fail(INCOMPLETE,row);return;}
  let velocity=vec3f(c00*b.x+c01*b.y+c02*b.z,c01*b.x+c11*b.y+c12*b.z,c02*b.x+c12*b.y+c22*b.z)/determinant;
  if(!all(vec3<bool>(finite(velocity.x),finite(velocity.y),finite(velocity.z)))||length(velocity)>max(1e-4,4.0*maxNormalSpeed)){rowStatus[row]=NONFINITE;fail(NONFINITE,row);return;}
  rowVelocities[row]=vec4f(velocity,1.0);rowStatus[row]=0u;atomicMax(&seed.projectedRowMaxBits,bitcast<u32>(length(velocity)));}
@compute @workgroup_size(64) fn publishAxisFaceVelocity(@builtin(global_invocation_id) gid:vec3u){let index=gid.x;
  if(index>=axisControl[0]||atomicLoad(&seed.flags)!=0u){return;}let face=axisFaces[index];let axis=face.axisSpan&3u;
  if(axis>=3u||(face.negativeRow==INVALID&&face.positiveRow==INVALID)
    ||(face.negativeRow!=INVALID&&face.negativeRow>=rowCount())||(face.positiveRow!=INVALID&&face.positiveRow>=rowCount())){fail(INCIDENCE,index);return;}
  var velocity=vec3f(0.0);if(face.negativeRow!=INVALID){velocity=rowVelocities[face.negativeRow].xyz;}else{velocity=rowVelocities[face.positiveRow].xyz;}
  if(face.negativeRow!=INVALID&&face.positiveRow!=INVALID){velocity=0.5*(velocity+rowVelocities[face.positiveRow].xyz);}
  let value=velocity[axis];if(!finite(value)||index>=arrayLength(&axisVelocityScratch)){fail(NONFINITE,index);return;}axisVelocityScratch[index]=value;atomicMax(&seed.axisOutputMaxBits,bitcast<u32>(abs(value)));atomicAdd(&seed.seededCount,1u);}
// Runs in a dedicated pass (its own auto layout) so the least-squares
// fallback publication above stays within the 8-storage-buffer stage limit.
@compute @workgroup_size(64) fn overrideHomologousAxisVelocity(@builtin(global_invocation_id) gid:vec3u){let index=gid.x;
  if(index>=axisControl[0]||index>=arrayLength(&axisFaces)||index>=arrayLength(&axisVelocityScratch)||atomicLoad(&seed.flags)!=0u){return;}
  let face=axisFaces[index];let axis=face.axisSpan&3u;if(axis>=3u){return;}
  let projected=homologousPowerVelocity(face,axis);
  if(projected.y!=0.0&&finite(projected.x)){axisVelocityScratch[index]=projected.x;}}
@compute @workgroup_size(1) fn publishPowerToAxis(){if(atomicLoad(&seed.flags)==0u&&atomicLoad(&seed.seededCount)==axisControl[0]){
  atomicStore(&seed.valid,VALID);}else{atomicStore(&seed.valid,0u);}}
@compute @workgroup_size(1) fn rejectFailedAuthoritativePowerToAxis(){if(atomicLoad(&seed.valid)!=VALID&&arrayLength(&axisControl)>1u){
  axisControl[1]=1u;}}
@compute @workgroup_size(64) fn commitPowerToAxis(@builtin(global_invocation_id) gid:vec3u){let index=gid.x;
  if(atomicLoad(&seed.valid)!=VALID||index>=axisControl[0]||index>=arrayLength(&axisFaces)||index>=arrayLength(&axisVelocityScratch)){return;}
  axisFaces[index].normalVelocity=axisVelocityScratch[index];}
`;
