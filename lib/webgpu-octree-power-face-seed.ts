/**
 * Native cold-start transaction for generalized power-face velocities.
 *
 * Generation one has no predecessor velocity field: all new generalized
 * face DOFs start at zero, except closed walls which are explicitly locked to
 * zero as well. Recurring generations retain only topology-carried DOFs; the
 * Section 5 old-mesh advection plus mandatory regular-face completion owns
 * every newly-added DOF.
 *
 * This stage deliberately has no Cartesian face source or reverse publication.
 */

import {
  OCTREE_POWER_FACE_DELTA_CARRIED,
  type OctreePowerFaceSource,
} from "./webgpu-octree-power-faces";
import type { PassBroker } from "./webgpu-pass-broker";

export const OCTREE_POWER_FACE_SEED_VALID = 0x8000_0000;
export const OCTREE_POWER_FACE_SEED_CONTROL_BYTES = 64;
export const OCTREE_POWER_FACE_SEED_ERROR = Object.freeze({
  source: 1 << 0,
  capacity: 1 << 1,
  normal: 1 << 4,
  nonfinite: 1 << 5,
} as const);

export interface OctreePowerFaceSeedPlan {
  readonly rowCapacity: number;
  readonly faceCapacity: number;
  readonly velocityScratchBytes: number;
  readonly faceStatusBytes: number;
  readonly allocatedBytes: number;
}

export function planOctreePowerFaceSeed(
  rowCapacity: number,
  faceCapacity: number,
): OctreePowerFaceSeedPlan {
  if (!Number.isSafeInteger(rowCapacity) || rowCapacity < 1
    || !Number.isSafeInteger(faceCapacity) || faceCapacity < 1) {
    throw new RangeError("Power-face seed capacities must be positive integers");
  }
  const velocityScratchBytes = faceCapacity * 4;
  const faceStatusBytes = faceCapacity * 4;
  return {
    rowCapacity,
    faceCapacity,
    velocityScratchBytes,
    faceStatusBytes,
    allocatedBytes: velocityScratchBytes + faceStatusBytes
      + OCTREE_POWER_FACE_SEED_CONTROL_BYTES + 32,
  };
}

export class WebGPUOctreePowerFaceSeed {
  readonly plan: OctreePowerFaceSeedPlan;
  readonly control: GPUBuffer;
  private readonly velocityScratch: GPUBuffer;
  private readonly faceStatus: GPUBuffer;
  private readonly params: GPUBuffer;
  private readonly accelerationParams: GPUBuffer;
  private readonly preparePipeline: GPUComputePipeline;
  private readonly seedPipeline: GPUComputePipeline;
  private readonly publishPipeline: GPUComputePipeline;
  private readonly commitPipeline: GPUComputePipeline;
  private readonly accelerationPreparePipeline: GPUComputePipeline;
  private readonly accelerationPipeline: GPUComputePipeline;
  private readonly accelerationPublishPipeline: GPUComputePipeline;
  private destroyed = false;

  constructor(
    private readonly device: GPUDevice,
    private readonly power: OctreePowerFaceSource,
  ) {
    this.plan = planOctreePowerFaceSeed(power.plan.rowCapacity, power.plan.faceCapacity);
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    this.velocityScratch = device.createBuffer({
      label: "Native power-face velocity transaction",
      size: this.plan.velocityScratchBytes,
      usage: storage,
    });
    this.faceStatus = device.createBuffer({
      label: "Native power-face velocity transaction status",
      size: this.plan.faceStatusBytes,
      usage: storage,
    });
    this.control = device.createBuffer({
      label: "Native power-face seed control",
      size: OCTREE_POWER_FACE_SEED_CONTROL_BYTES,
      usage: storage,
    });
    this.params = device.createBuffer({
      label: "Native power-face seed parameters",
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.accelerationParams = device.createBuffer({
      label: "Native power-face acceleration parameters",
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(this.params, 0, new Uint32Array([
      this.plan.rowCapacity, this.plan.faceCapacity, 0, 0,
    ]));
    const module = device.createShaderModule({
      label: "Native generalized power-face velocity seed",
      code: octreePowerFaceSeedShader,
    });
    const pipeline = (entryPoint: string) => device.createComputePipeline({
      label: entryPoint,
      layout: "auto",
      compute: { module, entryPoint },
    });
    this.preparePipeline = pipeline("preparePowerFaceSeed");
    this.seedPipeline = pipeline("seedPowerFaceVelocity");
    this.publishPipeline = pipeline("publishPowerFaceSeed");
    this.commitPipeline = pipeline("commitPowerFaceVelocity");
    this.accelerationPreparePipeline = pipeline("preparePowerFaceAcceleration");
    this.accelerationPipeline = pipeline("applyPowerFaceAcceleration");
    this.accelerationPublishPipeline = pipeline("publishPowerFaceAcceleration");
  }

  encode(broker: PassBroker): void {
    this.assertLive();
    this.run(broker, "Prepare native power-face velocity seed", this.preparePipeline, 1,
      [[0, this.params], [4, this.power.control], [5, this.power.faces],
        [6, this.power.faceNormals], [9, this.control],
        [15, this.velocityScratch], [16, this.faceStatus]]);
    this.run(broker, "Seed native generalized power-face velocities", this.seedPipeline,
      Math.ceil(this.plan.faceCapacity / 64),
      [[5, this.power.faces], [6, this.power.faceNormals], [9, this.control],
        [15, this.velocityScratch], [16, this.faceStatus]]);
    this.run(broker, "Publish native power-face velocity seed", this.publishPipeline, 1,
      [[9, this.control], [15, this.velocityScratch], [16, this.faceStatus]]);
    this.run(broker, "Commit native power-face velocity seed", this.commitPipeline,
      Math.ceil(this.plan.faceCapacity / 64),
      [[5, this.power.faces], [9, this.control], [15, this.velocityScratch]]);
  }

  /** Applies the split external-force term directly to generalized face DOFs. */
  encodeAcceleration(
    broker: PassBroker,
    acceleration: readonly [number, number, number],
    dt: number,
  ): void {
    this.assertLive();
    if (![...acceleration, dt].every(Number.isFinite) || dt < 0) {
      throw new RangeError("Power-face acceleration and dt must be finite, with dt non-negative");
    }
    this.device.queue.writeBuffer(this.accelerationParams, 0,
      new Float32Array([acceleration[0], acceleration[1], acceleration[2], dt]));
    this.run(broker, "Prepare authoritative power-face acceleration",
      this.accelerationPreparePipeline, 1,
      [[0, this.params], [4, this.power.control], [5, this.power.faces], [6, this.power.faceNormals],
        [9, this.control], [15, this.velocityScratch], [16, this.faceStatus]]);
    this.run(broker, "Apply acceleration to generalized power faces",
      this.accelerationPipeline, Math.ceil(this.plan.faceCapacity / 64),
      [[5, this.power.faces], [6, this.power.faceNormals], [9, this.control],
        [14, this.accelerationParams], [15, this.velocityScratch], [16, this.faceStatus]]);
    this.run(broker, "Publish authoritative power-face acceleration",
      this.accelerationPublishPipeline, 1,
      [[9, this.control], [15, this.velocityScratch], [16, this.faceStatus]]);
    this.run(broker, "Commit authoritative power-face acceleration",
      this.commitPipeline, Math.ceil(this.plan.faceCapacity / 64),
      [[5, this.power.faces], [9, this.control], [15, this.velocityScratch]]);
  }

  private run(
    broker: PassBroker,
    label: string,
    pipeline: GPUComputePipeline,
    groups: number,
    bindings: readonly [number, GPUBuffer][],
  ): void {
    const pass = broker.compute({ label });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: bindings.map(([binding, buffer]) => ({
        binding,
        resource: { buffer },
      })),
    }));
    pass.dispatchWorkgroups(groups);
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.velocityScratch.destroy();
    this.faceStatus.destroy();
    this.control.destroy();
    this.params.destroy();
    this.accelerationParams.destroy();
  }

  private assertLive(): void {
    if (this.destroyed) throw new Error("Power-face velocity seed is destroyed");
  }
}

export const octreePowerFaceSeedShader = /* wgsl */ `
struct Params { rowCapacity:u32,faceCapacity:u32,p0:u32,p1:u32 }
struct PowerFace {
  negativeRow:u32,positiveRow:u32,geometryCode:u32,flags:u32,
  normalVelocity:f32,area:f32,inverseDistance:f32,openFraction:f32
}
struct SeedControl {
  flags:u32,firstError:u32,rowCount:u32,faceCount:u32,seededCount:u32,generation:u32,valid:u32,
  retainedCount:u32,maximumBits:u32,p0:u32,p1:u32,p2:u32,
  forwardFlags:u32,forwardFirstError:u32,forwardSeededCount:u32,forwardValid:u32,
}
struct AccelerationParams { acceleration:vec3f,dt:f32 }
@group(0) @binding(0) var<uniform> params:Params;
@group(0) @binding(4) var<storage,read> powerControl:array<u32>;
@group(0) @binding(5) var<storage,read_write> powerFaces:array<PowerFace>;
@group(0) @binding(6) var<storage,read> powerNormals:array<vec4f>;
@group(0) @binding(9) var<storage,read_write> seed:SeedControl;
@group(0) @binding(14) var<uniform> accelerationParams:AccelerationParams;
@group(0) @binding(15) var<storage,read_write> velocityScratch:array<f32>;
@group(0) @binding(16) var<storage,read_write> faceStatus:array<u32>;
const INVALID=0xffffffffu;const VALID=0x80000000u;
const SOURCE=1u;const CAPACITY=2u;const NORMAL=16u;const NONFINITE=32u;
const BOUNDARY=1u;const OPEN_BOUNDARY=2u;
const DELTA_CARRIED:u32=${OCTREE_POWER_FACE_DELTA_CARRIED}u;
fn finite(v:f32)->bool{return v==v&&abs(v)<=3.402823e38;}
fn fail(error:u32,index:u32){seed.flags|=error;seed.firstError=min(seed.firstError,index);}
fn faceCount()->u32{return seed.faceCount;}

@compute @workgroup_size(1) fn preparePowerFaceSeed(){
  seed.flags=0u;seed.firstError=INVALID;seed.rowCount=0u;seed.faceCount=0u;
  seed.seededCount=0u;seed.generation=0u;seed.valid=0u;seed.retainedCount=0u;
  seed.maximumBits=0u;seed.p0=0u;seed.p1=0u;seed.p2=0u;
  seed.forwardFlags=0u;seed.forwardFirstError=INVALID;
  seed.forwardSeededCount=0u;seed.forwardValid=0u;
  if(arrayLength(&powerControl)<9u){fail(CAPACITY,0u);return;}
  seed.rowCount=powerControl[0];seed.faceCount=powerControl[1];seed.generation=powerControl[7];
  if(powerControl[3]!=0u||powerControl[8]!=VALID){fail(SOURCE,0u);}
  if(seed.rowCount>params.rowCapacity||seed.faceCount>params.faceCapacity
    ||seed.faceCount>arrayLength(&powerFaces)||seed.faceCount>arrayLength(&powerNormals)
    ||seed.faceCount>arrayLength(&velocityScratch)||seed.faceCount>arrayLength(&faceStatus)){
    fail(CAPACITY,0u);
  }
}

@compute @workgroup_size(64) fn seedPowerFaceVelocity(
  @builtin(global_invocation_id) gid:vec3u
){
  let index=gid.x;if(index>=faceCount()||seed.flags!=0u){return;}
  let face=powerFaces[index];let n=powerNormals[index].xyz;let n2=dot(n,n);
  if(face.negativeRow>=seed.rowCount
    ||(face.positiveRow!=INVALID&&face.positiveRow>=seed.rowCount)){
    faceStatus[index]=SOURCE;return;
  }
  if(!all(vec3<bool>(finite(n.x),finite(n.y),finite(n.z)))
    ||!finite(n2)||abs(n2-1.0)>4e-4){
    faceStatus[index]=NORMAL;return;
  }
  let carried=(face.flags&DELTA_CARRIED)!=0u;
  var value=select(0.0,face.normalVelocity,carried);
  if(face.positiveRow==INVALID&&(face.flags&BOUNDARY)!=0u
    &&(face.flags&OPEN_BOUNDARY)==0u){value=0.0;}
  if(!finite(value)){faceStatus[index]=NONFINITE;return;}
  velocityScratch[index]=value;
  faceStatus[index]=VALID|select(0u,1u,carried);
}

@compute @workgroup_size(1) fn publishPowerFaceSeed(){
  var flags=seed.flags;var first=seed.firstError;var seeded=0u;
  var retained=0u;var maximum=0.0;
  if(flags==0u){
    for(var index=0u;index<faceCount();index+=1u){
      let status=faceStatus[index];
      if((status&VALID)==0u){flags|=status;first=min(first,index);}
      else{seeded+=1u;retained+=status&1u;
        maximum=max(maximum,abs(velocityScratch[index]));}
    }
  }
  seed.flags=flags;seed.firstError=first;seed.seededCount=seeded;
  seed.retainedCount=retained;seed.maximumBits=bitcast<u32>(maximum);
  seed.valid=select(0u,VALID,flags==0u&&seeded==faceCount());
  seed.forwardFlags=flags;seed.forwardFirstError=first;
  seed.forwardSeededCount=seeded;seed.forwardValid=seed.valid;
}

@compute @workgroup_size(64) fn commitPowerFaceVelocity(
  @builtin(global_invocation_id) gid:vec3u
){
  let index=gid.x;
  if(seed.valid==VALID&&index<faceCount()&&index<arrayLength(&powerFaces)
    &&index<arrayLength(&velocityScratch)){
    powerFaces[index].normalVelocity=velocityScratch[index];
  }
}

@compute @workgroup_size(1) fn preparePowerFaceAcceleration(){
  let priorValid=seed.valid;let priorGeneration=seed.generation;
  seed.flags=0u;seed.firstError=INVALID;seed.seededCount=0u;
  seed.valid=0u;seed.retainedCount=0u;seed.maximumBits=0u;
  if(arrayLength(&powerControl)<9u){fail(CAPACITY,0u);return;}
  seed.rowCount=powerControl[0];seed.faceCount=powerControl[1];seed.generation=powerControl[7];
  if(priorValid!=VALID||priorGeneration!=powerControl[7]
    ||powerControl[3]!=0u||powerControl[8]!=VALID){fail(SOURCE,0u);}
  if(seed.faceCount>params.faceCapacity||seed.faceCount>arrayLength(&powerFaces)
    ||seed.faceCount>arrayLength(&powerNormals)
    ||seed.faceCount>arrayLength(&velocityScratch)
    ||seed.faceCount>arrayLength(&faceStatus)){fail(CAPACITY,0u);}
}

@compute @workgroup_size(64) fn applyPowerFaceAcceleration(
  @builtin(global_invocation_id) gid:vec3u
){
  let index=gid.x;if(index>=faceCount()||seed.flags!=0u){return;}
  let face=powerFaces[index];
  let closed=face.positiveRow==INVALID&&(face.flags&BOUNDARY)!=0u
    &&(face.flags&OPEN_BOUNDARY)==0u;
  if(closed){velocityScratch[index]=0.0;faceStatus[index]=VALID;return;}
  let n=powerNormals[index].xyz;
  let force=dot(accelerationParams.acceleration,n)*accelerationParams.dt;
  let value=face.normalVelocity+force;
  if(!finite(force)||!finite(value)){faceStatus[index]=NONFINITE;return;}
  velocityScratch[index]=value;faceStatus[index]=VALID;
}

@compute @workgroup_size(1) fn publishPowerFaceAcceleration(){
  var flags=seed.flags;var first=seed.firstError;var seeded=0u;var maximum=0.0;
  if(flags==0u){
    for(var index=0u;index<faceCount();index+=1u){
      let status=faceStatus[index];
      if(status!=VALID){flags|=status;first=min(first,index);}
      else{seeded+=1u;maximum=max(maximum,abs(velocityScratch[index]));}
    }
  }
  seed.flags=flags;seed.firstError=first;seed.seededCount=seeded;
  seed.maximumBits=bitcast<u32>(maximum);
  seed.valid=select(0u,VALID,flags==0u&&seeded==faceCount());
}
`;
