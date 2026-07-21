import { GPU_RIGID_BODY_CAPACITY } from "./webgpu-rigid-body";
import type { OctreeFaceMirrorSource } from "./webgpu-octree-face-mirror";

export const OCTREE_SOLID_FACE_RECORD_BYTES = 16;
export const OCTREE_SOLID_IMPULSE_WORDS_PER_BODY = 8;
export const OCTREE_SOLID_DIAGNOSTIC_BYTES = 48;

export function constrainOctreeFaceVelocity(
  fluidNormalVelocity: number,
  openFraction: number,
  solidNormalVelocity: number,
): number {
  const open = Math.min(1, Math.max(0, openFraction));
  return open * fluidNormalVelocity + (1 - open) * solidNormalVelocity;
}

export interface OctreeSolidFacePlan {
  faceCapacity: number;
  bodyCapacity: number;
  apertureBytes: number;
  impulseBytes: number;
  diagnosticBytes: number;
  allocatedBytes: number;
}

export function planOctreeSolidFaces(
  faceCapacity: number,
  bodyCapacity = GPU_RIGID_BODY_CAPACITY,
): OctreeSolidFacePlan {
  if (!Number.isInteger(faceCapacity) || faceCapacity < 1) throw new RangeError("Octree solid face capacity must be positive");
  if (!Number.isInteger(bodyCapacity) || bodyCapacity < 1 || bodyCapacity > GPU_RIGID_BODY_CAPACITY) {
    throw new RangeError(`Octree solid body capacity must be between 1 and ${GPU_RIGID_BODY_CAPACITY}`);
  }
  const apertureBytes = faceCapacity * OCTREE_SOLID_FACE_RECORD_BYTES;
  const impulseBytes = bodyCapacity * OCTREE_SOLID_IMPULSE_WORDS_PER_BODY * 4;
  return {
    faceCapacity,
    bodyCapacity,
    apertureBytes,
    impulseBytes,
    diagnosticBytes: OCTREE_SOLID_DIAGNOSTIC_BYTES,
    allocatedBytes: apertureBytes + impulseBytes + OCTREE_SOLID_DIAGNOSTIC_BYTES,
  };
}

export interface OctreeSolidFaceResources {
  faces: OctreeFaceMirrorSource;
  rigidBodies: GPUBuffer;
  params: GPUBuffer;
  pressureA: GPUBuffer;
  pressureB: GPUBuffer;
  rigidExchange?: GPUBuffer;
}

/** Compact output consumed by adaptive projection/transport and rigid update. */
export interface OctreeSolidFaceSource {
  plan: OctreeSolidFacePlan;
  /** Face-indexed { openFraction, solidNormalVelocity, dominantOwner, sampleMask }. */
  apertures: GPUBuffer;
  /** Eight fixed-point words/body: linear xyz, angular xyz, two reserved words. */
  bodyImpulses: GPUBuffer;
  /** Flags must be zero before either output is authoritative. */
  diagnostics: GPUBuffer;
}

export interface OctreeSolidFaceDiagnostics {
  flags: number;
  processedFaces: number;
  cutFaces: number;
  occupiedSamples: number;
  bodyImpulseFixed: [number, number, number];
  fluidImpulseFixed: [number, number, number];
}

/**
 * Builds solid apertures and pressure reactions directly on canonical adaptive
 * faces. It deliberately owns its impulse arena: overflow can invalidate a
 * whole batch without contaminating the rigid solver's shared exchange.
 */
export class WebGPUOctreeSolidFaces {
  readonly plan: OctreeSolidFacePlan;
  readonly apertures: GPUBuffer;
  readonly bodyImpulses: GPUBuffer;
  readonly diagnostics: GPUBuffer;
  private readonly classifyPipeline: GPUComputePipeline;
  private readonly impulsePipeline: GPUComputePipeline;
  private readonly constrainPipeline: GPUComputePipeline;
  private readonly lockClosedPipeline: GPUComputePipeline;
  private readonly exchangePipeline?: GPUComputePipeline;
  private readonly classifyGroup: GPUBindGroup;
  private readonly impulseGroups: { pressureA: GPUBindGroup; pressureB: GPUBindGroup };
  private readonly exchangeGroup?: GPUBindGroup;

  constructor(private readonly device: GPUDevice, resources: OctreeSolidFaceResources, bodyCapacity = GPU_RIGID_BODY_CAPACITY) {
    this.plan = planOctreeSolidFaces(resources.faces.plan.faceCapacity, bodyCapacity);
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    this.apertures = device.createBuffer({ label: "Octree adaptive solid face apertures", size: this.plan.apertureBytes, usage: storage });
    this.bodyImpulses = device.createBuffer({ label: "Octree adaptive rigid impulse batch", size: this.plan.impulseBytes, usage: storage });
    this.diagnostics = device.createBuffer({ label: "Octree adaptive solid diagnostics", size: this.plan.diagnosticBytes, usage: storage });
    const shaderModule = device.createShaderModule({ label: "Octree adaptive solid faces", code: octreeSolidFaceShader });
    const layout = device.createBindGroupLayout({ label: "Octree adaptive solid face layout", entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
    ] });
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
    this.classifyPipeline = device.createComputePipeline({ label: "Classify adaptive solid faces", layout: pipelineLayout, compute: { module: shaderModule, entryPoint: "classifyFaces" } });
    this.constrainPipeline = device.createComputePipeline({ label: "Constrain adaptive solid face velocities", layout: pipelineLayout, compute: { module: shaderModule, entryPoint: "constrainFaces" } });
    this.lockClosedPipeline = device.createComputePipeline({ label: "Lock closed adaptive solid faces", layout: pipelineLayout, compute: { module: shaderModule, entryPoint: "lockClosedFaces" } });
    this.impulsePipeline = device.createComputePipeline({ label: "Accumulate adaptive rigid pressure impulses", layout: pipelineLayout, compute: { module: shaderModule, entryPoint: "accumulatePressureImpulses" } });
    const group = (pressure: GPUBuffer) => device.createBindGroup({ label: "Octree adaptive solid face bindings", layout, entries: [
      { binding: 0, resource: { buffer: resources.faces.control } },
      { binding: 1, resource: { buffer: resources.faces.faces } },
      { binding: 2, resource: { buffer: resources.rigidBodies } },
      { binding: 3, resource: { buffer: resources.params } },
      { binding: 4, resource: { buffer: this.apertures } },
      { binding: 5, resource: { buffer: this.bodyImpulses } },
      { binding: 6, resource: { buffer: this.diagnostics } },
      { binding: 7, resource: { buffer: pressure } },
    ] });
    this.classifyGroup = group(resources.pressureA);
    this.impulseGroups = { pressureA: this.classifyGroup, pressureB: group(resources.pressureB) };
    if (resources.rigidExchange) {
      const exchangeLayout = device.createBindGroupLayout({ label: "Octree adaptive rigid impulse exchange layout", entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      ] });
      const exchangeModule = device.createShaderModule({ label: "Octree adaptive rigid impulse exchange", code: octreeSolidExchangeShader });
      this.exchangePipeline = device.createComputePipeline({
        label: "Publish adaptive face impulses to rigid exchange",
        layout: device.createPipelineLayout({ bindGroupLayouts: [exchangeLayout] }),
        compute: { module: exchangeModule, entryPoint: "publishRigidExchange" },
      });
      this.exchangeGroup = device.createBindGroup({ label: "Octree adaptive rigid impulse exchange bindings", layout: exchangeLayout, entries: [
        { binding: 0, resource: { buffer: this.bodyImpulses } },
        { binding: 1, resource: { buffer: resources.rigidExchange } },
        { binding: 2, resource: { buffer: this.diagnostics } },
      ] });
    }
  }

  /** Classify the current face topology and apply the finite-volume no-penetration flux. */
  encodeClassifyAndConstrain(encoder: GPUCommandEncoder): void {
    encoder.clearBuffer(this.bodyImpulses);
    encoder.clearBuffer(this.diagnostics);
    const count = Math.ceil(this.plan.faceCapacity / 256);
    const classify = encoder.beginComputePass({ label: "Classify adaptive solid face apertures" });
    classify.setPipeline(this.classifyPipeline);
    classify.setBindGroup(0, this.classifyGroup);
    classify.dispatchWorkgroups(count);
    classify.end();
    const constrain = encoder.beginComputePass({ label: "Apply adaptive solid finite-volume flux" });
    constrain.setPipeline(this.constrainPipeline);
    constrain.setBindGroup(0, this.classifyGroup);
    constrain.dispatchWorkgroups(count);
    constrain.end();
  }

  /** Re-apply the solid-normal constraint after a pressure gradient update. */
  encodePostProjectionConstraint(encoder: GPUCommandEncoder): void {
    const constrain = encoder.beginComputePass({ label: "Constrain adaptive solid face-normal velocities" });
    constrain.setPipeline(this.lockClosedPipeline);
    constrain.setBindGroup(0, this.classifyGroup);
    constrain.dispatchWorkgroups(Math.ceil(this.plan.faceCapacity / 256));
    constrain.end();
  }

  /** Accumulate paired pressure reactions and publish them to the rigid solver exchange. */
  encodePressureImpulses(encoder: GPUCommandEncoder, pressureInA: boolean): void {
    const count = Math.ceil(this.plan.faceCapacity / 256);
    const couple = encoder.beginComputePass({ label: "Accumulate adaptive solid pressure reactions" });
    couple.setPipeline(this.impulsePipeline);
    couple.setBindGroup(0, pressureInA ? this.impulseGroups.pressureA : this.impulseGroups.pressureB);
    couple.dispatchWorkgroups(count);
    couple.end();
    if (this.exchangePipeline && this.exchangeGroup) {
      const publish = encoder.beginComputePass({ label: "Publish adaptive pressure reactions to rigid exchange" });
      publish.setPipeline(this.exchangePipeline);
      publish.setBindGroup(0, this.exchangeGroup);
      publish.dispatchWorkgroups(Math.ceil(this.plan.bodyCapacity / 64));
      publish.end();
    }
  }

  encode(encoder: GPUCommandEncoder, pressureInA: boolean, accumulatePressureImpulse = true): void {
    this.encodeClassifyAndConstrain(encoder);
    if (accumulatePressureImpulse) this.encodePressureImpulses(encoder, pressureInA);
  }

  get source(): OctreeSolidFaceSource {
    return { plan: this.plan, apertures: this.apertures, bodyImpulses: this.bodyImpulses, diagnostics: this.diagnostics };
  }

  async readDiagnostics(): Promise<OctreeSolidFaceDiagnostics> {
    const readback = this.device.createBuffer({ label: "Read adaptive solid face diagnostics", size: OCTREE_SOLID_DIAGNOSTIC_BYTES, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = this.device.createCommandEncoder({ label: "Copy adaptive solid face diagnostics" });
    encoder.copyBufferToBuffer(this.diagnostics, 0, readback, 0, OCTREE_SOLID_DIAGNOSTIC_BYTES);
    this.device.queue.submit([encoder.finish()]);
    try {
      await readback.mapAsync(GPUMapMode.READ);
      const bytes = readback.getMappedRange();
      const words = new Uint32Array(bytes);
      const signed = new Int32Array(bytes);
      return {
        flags: words[0], processedFaces: words[1], cutFaces: words[2], occupiedSamples: words[3],
        bodyImpulseFixed: [signed[4], signed[5], signed[6]],
        fluidImpulseFixed: [signed[7], signed[8], signed[9]],
      };
    } finally {
      if (readback.mapState === "mapped") readback.unmap();
      readback.destroy();
    }
  }

  destroy(): void {
    this.apertures.destroy();
    this.bodyImpulses.destroy();
    this.diagnostics.destroy();
  }
}

export const octreeSolidFaceShader = /* wgsl */ `
struct FaceRecord { negativeRow:u32, positiveRow:u32, originX:u32, originY:u32, originZ:u32, axisSpan:u32, normalVelocity:f32, area:f32 }
struct ApertureRecord { openFraction:f32, solidNormalVelocity:f32, dominantOwner:i32, sampleMask:u32 }
struct RigidBody { positionShape:vec4f, dimensions:vec4f, orientation:vec4f, linearVelocity:vec4f, angularVelocity:vec4f, inverseMassInertia:vec4f, angularMomentumRestitution:vec4f, material:vec4f }
struct Params { dimsMax:vec4u, cellRelax:vec4f, control:vec4u, solve:vec4f, container:vec4f, inflowPositionRadius:vec4f, inflowDirectionLength:vec4f, physical:vec4f, pressureCapacity:vec4u, hydrostatic:vec4f }
@group(0) @binding(0) var<storage,read> faceControl:array<u32>;
@group(0) @binding(1) var<storage,read_write> faces:array<FaceRecord>;
@group(0) @binding(2) var<storage,read> bodies:array<RigidBody,12>;
@group(0) @binding(3) var<uniform> params:Params;
@group(0) @binding(4) var<storage,read_write> apertures:array<ApertureRecord>;
@group(0) @binding(5) var<storage,read_write> impulses:array<atomic<i32>>;
// flags, processed faces, cut faces, occupied samples, body xyz, fluid xyz, reserved
@group(0) @binding(6) var<storage,read_write> diagnostics:array<atomic<u32>>;
@group(0) @binding(7) var<storage,read> pressure:array<f32>;
const INVALID=0xffffffffu;const SAMPLE_AXIS=4u;const SAMPLE_COUNT=16u;const FIXED_SCALE=1000000.0;const MAX_FIXED=2000000000.0;
fn axisVector(axis:u32)->vec3f{return select(select(vec3f(0,0,1),vec3f(0,1,0),axis==1u),vec3f(1,0,0),axis==0u);}
fn component(v:vec3f,axis:u32)->f32{return select(select(v.z,v.y,axis==1u),v.x,axis==0u);}
fn qConjugate(q:vec4f)->vec4f{return vec4f(q.x,-q.yzw);}
fn qRotate(q:vec4f,v:vec3f)->vec3f{let uv=cross(q.yzw,v);return v+2.0*(q.x*uv+cross(q.yzw,uv));}
fn localPoint(body:RigidBody,world:vec3f)->vec3f{return qRotate(qConjugate(body.orientation),world-body.positionShape.xyz);}
fn bodySdf(body:RigidBody,world:vec3f)->f32{let p=localPoint(body,world);let d=body.dimensions.xyz;let shape=i32(round(body.positionShape.w));
  if(shape==0){return length(p)-d.x;}
  if(shape==1){let q=abs(p)-0.5*d;return length(max(q,vec3f(0)))+min(max(q.x,max(q.y,q.z)),0.0);}
  if(shape==2){let q=vec3f(p.x,p.y-clamp(p.y,-0.5*d.y,0.5*d.y),p.z);return length(q)-d.x;}
  let q=vec2f(length(p.xz)-d.x,abs(p.y)-0.5*d.y);return length(max(q,vec2f(0)))+min(max(q.x,q.y),0.0);
}
fn worldFaceSample(face:FaceRecord,sample:u32)->vec3f{let axis=face.axisSpan&3u;let span=f32(face.axisSpan>>2u);let origin=vec3f(f32(face.originX),f32(face.originY),f32(face.originZ));let a=f32(sample%SAMPLE_AXIS)+0.5;let b=f32(sample/SAMPLE_AXIS)+0.5;var grid=origin;grid[(axis+1u)%3u]+=span*a/f32(SAMPLE_AXIS);grid[(axis+2u)%3u]+=span*b/f32(SAMPLE_AXIS);let h=params.cellRelax.xyz;return vec3f(-0.5*params.container.x+grid.x*h.x,grid.y*h.y,-0.5*params.container.z+grid.z*h.z);}
fn sampleOwner(world:vec3f)->i32{var best=3.402823e38;var owner=-1;for(var i=0u;i<min(params.control.w,12u);i+=1u){let distance=bodySdf(bodies[i],world);if(distance<best){best=distance;owner=i32(i);}}return select(-1,owner,best<0.0);}
fn normalVelocity(owner:i32,world:vec3f,axis:u32)->f32{if(owner<0){return 0.0;}let body=bodies[u32(owner)];let velocity=body.linearVelocity.xyz+cross(body.angularVelocity.xyz,world-body.positionShape.xyz);return component(velocity,axis);}
fn faceValid(index:u32)->bool{if(faceControl[1]!=0u||faceControl[0]>faceControl[2]||params.control.w>12u){atomicOr(&diagnostics[0],1u);return false;}return index<faceControl[0]&&index<faceControl[2]&&index<arrayLength(&faces)&&index<arrayLength(&apertures);}
fn pressureAt(row:u32)->f32{if(row==INVALID||row>=arrayLength(&pressure)){return 0.0;}return pressure[row];}
fn checkedFixed(value:f32)->i32{if(!(value==value)||abs(value*FIXED_SCALE)>MAX_FIXED){atomicOr(&diagnostics[0],2u);return 0;}return i32(round(value*FIXED_SCALE));}
fn addReaction(owner:u32,linear:vec3f,world:vec3f){if(owner>=params.control.w||owner>=12u||owner*8u+5u>=arrayLength(&impulses)){atomicOr(&diagnostics[0],4u);return;}let torque=cross(world-bodies[owner].positionShape.xyz,linear);let base=owner*8u;
  atomicAdd(&impulses[base],checkedFixed(linear.x));atomicAdd(&impulses[base+1u],checkedFixed(linear.y));atomicAdd(&impulses[base+2u],checkedFixed(linear.z));atomicAdd(&impulses[base+3u],checkedFixed(torque.x));atomicAdd(&impulses[base+4u],checkedFixed(torque.y));atomicAdd(&impulses[base+5u],checkedFixed(torque.z));
  atomicAdd(&diagnostics[4],bitcast<u32>(checkedFixed(linear.x)));atomicAdd(&diagnostics[5],bitcast<u32>(checkedFixed(linear.y)));atomicAdd(&diagnostics[6],bitcast<u32>(checkedFixed(linear.z)));atomicAdd(&diagnostics[7],bitcast<u32>(checkedFixed(-linear.x)));atomicAdd(&diagnostics[8],bitcast<u32>(checkedFixed(-linear.y)));atomicAdd(&diagnostics[9],bitcast<u32>(checkedFixed(-linear.z)));
}
@compute @workgroup_size(256) fn classifyFaces(@builtin(global_invocation_id) gid:vec3u){let index=gid.x;if(!faceValid(index)){return;}let face=faces[index];let axis=face.axisSpan&3u;var occupied=0u;var mask=0u;var velocity=0.0;var counts:array<u32,12>;
  for(var sample=0u;sample<SAMPLE_COUNT;sample+=1u){let world=worldFaceSample(face,sample);let owner=sampleOwner(world);if(owner>=0){occupied+=1u;mask|=1u<<sample;counts[u32(owner)]+=1u;velocity+=normalVelocity(owner,world,axis);}}
  var dominant=-1;var largest=0u;for(var i=0u;i<min(params.control.w,12u);i+=1u){if(counts[i]>largest){largest=counts[i];dominant=i32(i);}}
  apertures[index]=ApertureRecord(1.0-f32(occupied)/f32(SAMPLE_COUNT),select(0.0,velocity/f32(occupied),occupied>0u),dominant,mask);atomicAdd(&diagnostics[1],1u);if(occupied>0u&&occupied<SAMPLE_COUNT){atomicAdd(&diagnostics[2],1u);}atomicAdd(&diagnostics[3],occupied);
}
@compute @workgroup_size(256) fn constrainFaces(@builtin(global_invocation_id) gid:vec3u){let index=gid.x;if(!faceValid(index)){return;}var face=faces[index];if(atomicLoad(&diagnostics[0])!=0u){face.normalVelocity=0.0;faces[index]=face;return;}let aperture=apertures[index];face.normalVelocity=aperture.openFraction*face.normalVelocity+(1.0-aperture.openFraction)*aperture.solidNormalVelocity;faces[index]=face;}
@compute @workgroup_size(256) fn lockClosedFaces(@builtin(global_invocation_id) gid:vec3u){let index=gid.x;if(!faceValid(index)){return;}var face=faces[index];if(atomicLoad(&diagnostics[0])!=0u){face.normalVelocity=0.0;}else{let aperture=apertures[index];if(aperture.openFraction<=0.0){face.normalVelocity=aperture.solidNormalVelocity;}}faces[index]=face;}
@compute @workgroup_size(256) fn accumulatePressureImpulses(@builtin(global_invocation_id) gid:vec3u){let index=gid.x;if(!faceValid(index)){return;}let face=faces[index];let axis=face.axisSpan&3u;let scalar=params.physical.x*face.area*(pressureAt(face.negativeRow)-pressureAt(face.positiveRow))/f32(SAMPLE_COUNT);let direction=axisVector(axis);
  for(var sample=0u;sample<SAMPLE_COUNT;sample+=1u){let world=worldFaceSample(face,sample);let owner=sampleOwner(world);if(owner>=0){let bodyReaction=direction*scalar;addReaction(u32(owner),bodyReaction,world);}}
}
`;

export const octreeSolidExchangeShader = /* wgsl */ `
@group(0) @binding(0) var<storage,read> stagedImpulses:array<i32>;
@group(0) @binding(1) var<storage,read_write> rigidExchange:array<atomic<i32>>;
@group(0) @binding(2) var<storage,read> stagedDiagnostics:array<u32>;
@compute @workgroup_size(64) fn publishRigidExchange(@builtin(global_invocation_id) gid:vec3u){let body=gid.x;if(body>=arrayLength(&stagedImpulses)/8u||body*12u+5u>=arrayLength(&rigidExchange)||stagedDiagnostics[0]!=0u){return;}for(var word=0u;word<6u;word+=1u){atomicAdd(&rigidExchange[body*12u+word],stagedImpulses[body*8u+word]);}}
`;
