import { GPU_RIGID_BODY_CAPACITY } from "./webgpu-rigid-body";
import { OCTREE_POWER_FACE_VALID, type OctreePowerFaceSource } from "./webgpu-octree-power-faces";
import {
  OCTREE_SOLID_VERTEX_SDF_VALID,
  type OctreeSolidVertexSdfSource,
} from "./webgpu-octree-solid-vertex-sdf";

export const OCTREE_POWER_SOLID_APERTURE_BYTES = 16;
export const OCTREE_POWER_SOLID_CONTROL_BYTES = 64;
export const OCTREE_POWER_SOLID_VALID = 0x8000_0000;
export const OCTREE_POWER_SOLID_IMPULSE_WORDS = 8;

export interface OctreePowerSolidFacePlan {
  readonly faceCapacity: number;
  readonly bodyCapacity: number;
  readonly apertureBytes: number;
  readonly impulseBytes: number;
  readonly allocatedBytes: number;
}

export function planOctreePowerSolidFaces(
  faceCapacity: number,
  bodyCapacity = GPU_RIGID_BODY_CAPACITY,
): OctreePowerSolidFacePlan {
  if (!Number.isSafeInteger(faceCapacity) || faceCapacity < 1) {
    throw new RangeError("Power solid face capacity must be a positive integer");
  }
  if (!Number.isSafeInteger(bodyCapacity) || bodyCapacity < 1 || bodyCapacity > GPU_RIGID_BODY_CAPACITY) {
    throw new RangeError(`Power solid body capacity must be between 1 and ${GPU_RIGID_BODY_CAPACITY}`);
  }
  const apertureBytes = faceCapacity * OCTREE_POWER_SOLID_APERTURE_BYTES;
  const impulseBytes = bodyCapacity * OCTREE_POWER_SOLID_IMPULSE_WORDS * 4;
  return { faceCapacity, bodyCapacity, apertureBytes, impulseBytes,
    allocatedBytes: apertureBytes + impulseBytes + OCTREE_POWER_SOLID_CONTROL_BYTES + 64 };
}

export interface OctreePowerSolidResources {
  readonly faces: OctreePowerFaceSource;
  readonly rigidBodies: GPUBuffer;
  readonly terrain: GPUTexture;
  readonly pressureA: GPUBuffer;
  readonly pressureB: GPUBuffer;
  readonly rigidExchange?: GPUBuffer;
  /** Paper Section 4.1 cell-vertex solid SDF publication. Required for terrain. */
  readonly solidVertices?: OctreeSolidVertexSdfSource;
}

export interface OctreePowerSolidEncodeOptions {
  readonly dimensions: readonly [number, number, number];
  readonly physicalSpacing: readonly [number, number, number];
  readonly container: readonly [number, number, number];
  readonly rigidBodyCount: number;
  readonly terrainEnabled: boolean;
  /** Multiplier from pressure units to impulse units, normally dt/density. */
  readonly pressureImpulseScale?: number;
}

export interface OctreePowerSolidDiagnostics {
  readonly flags: number;
  readonly processedFaces: number;
  readonly cutFaces: number;
  readonly occupiedSamples: number;
  readonly faceCount: number;
  readonly rigidBodyCount: number;
  readonly terrainEnabled: boolean;
  readonly valid: number;
  readonly bodyImpulseFixed: readonly [number, number, number];
  readonly fluidImpulseFixed: readonly [number, number, number];
  readonly vertexGeneration: number;
  readonly rollbackGeneration: number;
}

/**
 * General power polygons carry their exact area, centroid, normal, and sixteen
 * deterministic equal-area samples of the clipped world-space polygon. Solid
 * aperture integration therefore follows the catalog power face (including
 * reciprocal T-junction clipping) and never substitutes a rectangular face.
 */
export class WebGPUOctreePowerSolidFaces {
  readonly plan: OctreePowerSolidFacePlan;
  readonly apertures: GPUBuffer;
  readonly bodyImpulses: GPUBuffer;
  readonly control: GPUBuffer;
  private readonly params: GPUBuffer;
  private readonly classifyPipeline: GPUComputePipeline;
  private readonly finishPipeline: GPUComputePipeline;
  private readonly constrainPipeline: GPUComputePipeline;
  private readonly lockPipeline: GPUComputePipeline;
  private readonly impulsePipeline: GPUComputePipeline;
  private readonly classifyGroup: GPUBindGroup;
  private readonly impulseGroups: readonly [GPUBindGroup, GPUBindGroup];
  private readonly exchangePipeline?: GPUComputePipeline;
  private readonly exchangeGroup?: GPUBindGroup;
  private readonly fallbackVertexArena?: GPUBuffer;
  private destroyed = false;

  constructor(
    private readonly device: GPUDevice,
    private readonly resources: OctreePowerSolidResources,
    bodyCapacity = GPU_RIGID_BODY_CAPACITY,
  ) {
    this.plan = planOctreePowerSolidFaces(resources.faces.plan.faceCapacity, bodyCapacity);
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    this.apertures = device.createBuffer({ label: "Octree power solid apertures", size: this.plan.apertureBytes, usage: storage });
    this.bodyImpulses = device.createBuffer({ label: "Octree power rigid impulse batch", size: this.plan.impulseBytes, usage: storage });
    this.control = device.createBuffer({ label: "Octree power solid control", size: OCTREE_POWER_SOLID_CONTROL_BYTES, usage: storage });
    this.params = device.createBuffer({ label: "Octree power solid parameters", size: 64,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const fallback = (label: string, size: number) => device.createBuffer({ label, size, usage: storage });
    this.fallbackVertexArena = resources.solidVertices ? undefined : fallback("Invalid solid vertex-SDF publication fallback", 96);
    const shaderModule = device.createShaderModule({ label: "Octree generalized power solid faces", code: octreePowerSolidFaceShader });
    const classifyLayout = device.createBindGroupLayout({ label: "Octree power solid classification layout", entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 8, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "2d" } },
      { binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
    ] });
    const classifyPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [classifyLayout] });
    const makeClassifyPipeline = (label: string, entryPoint: string) => device.createComputePipeline({
      label, layout: classifyPipelineLayout, compute: { module: shaderModule, entryPoint },
    });
    this.classifyPipeline = makeClassifyPipeline("Classify generalized power solid faces", "classifyPowerSolidFaces");
    this.finishPipeline = makeClassifyPipeline("Publish generalized power solid faces", "finishPowerSolidFaces");
    this.constrainPipeline = makeClassifyPipeline("Constrain generalized power solid faces", "constrainPowerSolidFaces");
    this.lockPipeline = makeClassifyPipeline("Lock closed generalized power solid faces", "lockClosedPowerSolidFaces");
    this.classifyGroup = device.createBindGroup({ label: "Octree power solid classification bindings", layout: classifyLayout, entries: [
      { binding: 0, resource: { buffer: resources.faces.control } },
      { binding: 1, resource: { buffer: resources.faces.faces } },
      { binding: 2, resource: { buffer: resources.faces.faceNormals } },
      { binding: 3, resource: { buffer: resources.faces.faceQuadrature } },
      { binding: 4, resource: { buffer: resources.rigidBodies } },
      { binding: 5, resource: { buffer: this.params } },
      { binding: 6, resource: { buffer: this.apertures } },
      { binding: 7, resource: { buffer: this.control } },
      { binding: 8, resource: resources.terrain.createView() },
      { binding: 9, resource: { buffer: resources.solidVertices?.arena ?? this.fallbackVertexArena! } },
    ] });
    const impulseLayout = device.createBindGroupLayout({ label: "Octree power solid impulse layout", entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
    ] });
    const impulsePipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [impulseLayout] });
    const impulseModule = device.createShaderModule({ label: "Octree generalized power solid impulses", code: octreePowerSolidImpulseShader });
    this.impulsePipeline = device.createComputePipeline({ label: "Accumulate generalized power solid pressure reactions",
      layout: impulsePipelineLayout, compute: { module: impulseModule, entryPoint: "accumulatePowerSolidImpulses" } });
    const impulseGroup = (pressure: GPUBuffer) => device.createBindGroup({ label: "Octree power solid impulse bindings", layout: impulseLayout, entries: [
      { binding: 0, resource: { buffer: resources.faces.faces } },
      { binding: 1, resource: { buffer: resources.faces.faceNormals } },
      { binding: 2, resource: { buffer: resources.faces.faceQuadrature } },
      { binding: 3, resource: { buffer: resources.rigidBodies } },
      { binding: 4, resource: { buffer: this.apertures } },
      { binding: 5, resource: { buffer: this.params } },
      { binding: 6, resource: { buffer: this.bodyImpulses } },
      { binding: 7, resource: { buffer: this.control } },
      { binding: 8, resource: { buffer: pressure } },
    ] });
    this.impulseGroups = [impulseGroup(resources.pressureA), impulseGroup(resources.pressureB)];
    if (resources.rigidExchange) {
      const exchangeModule = device.createShaderModule({ label: "Octree power solid exchange", code: octreePowerSolidExchangeShader });
      const exchangeLayout = device.createBindGroupLayout({ entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      ] });
      this.exchangePipeline = device.createComputePipeline({ label: "Publish power solid rigid exchange",
        layout: device.createPipelineLayout({ bindGroupLayouts: [exchangeLayout] }),
        compute: { module: exchangeModule, entryPoint: "publishPowerSolidExchange" } });
      this.exchangeGroup = device.createBindGroup({ layout: exchangeLayout, entries: [
        { binding: 0, resource: { buffer: this.bodyImpulses } },
        { binding: 1, resource: { buffer: resources.rigidExchange } },
        { binding: 2, resource: { buffer: this.control } },
      ] });
    }
  }

  encodeClassifyAndConstrain(encoder: GPUCommandEncoder, options: OctreePowerSolidEncodeOptions): void {
    this.assertLive();
    this.writeOptions(options);
    encoder.clearBuffer(this.bodyImpulses); encoder.clearBuffer(this.control);
    const groups = Math.ceil(this.plan.faceCapacity / 64);
    this.run(encoder, "Classify generalized power solid apertures", this.classifyPipeline, groups, this.classifyGroup);
    this.run(encoder, "Publish generalized power solid apertures", this.finishPipeline, 1, this.classifyGroup);
    this.run(encoder, "Apply generalized power solid flux", this.constrainPipeline, groups, this.classifyGroup);
  }

  encodePostProjectionConstraint(encoder: GPUCommandEncoder): void {
    this.assertLive();
    this.run(encoder, "Lock closed generalized power faces", this.lockPipeline,
      Math.ceil(this.plan.faceCapacity / 64), this.classifyGroup);
  }

  encodePressureImpulses(encoder: GPUCommandEncoder, pressureInA: boolean): void {
    this.assertLive();
    this.run(encoder, "Accumulate generalized power solid pressure reactions", this.impulsePipeline,
      Math.ceil(this.plan.faceCapacity / 64), this.impulseGroups[pressureInA ? 0 : 1]);
    if (this.exchangePipeline && this.exchangeGroup) {
      this.run(encoder, "Publish generalized power rigid exchange", this.exchangePipeline,
        Math.ceil(this.plan.bodyCapacity / 64), this.exchangeGroup);
    }
  }

  async readDiagnostics(): Promise<OctreePowerSolidDiagnostics> {
    this.assertLive();
    const readback = this.device.createBuffer({ size: OCTREE_POWER_SOLID_CONTROL_BYTES,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(this.control, 0, readback, 0, OCTREE_POWER_SOLID_CONTROL_BYTES);
    this.device.queue.submit([encoder.finish()]);
    try {
      await readback.mapAsync(GPUMapMode.READ);
      const bytes = readback.getMappedRange(); const words = new Uint32Array(bytes); const signed = new Int32Array(bytes);
      return { flags: words[0], processedFaces: words[1], cutFaces: words[2], occupiedSamples: words[3],
        faceCount: words[4], rigidBodyCount: words[5], terrainEnabled: words[6] !== 0, valid: words[7],
        bodyImpulseFixed: [signed[8], signed[9], signed[10]], fluidImpulseFixed: [signed[11], signed[12], signed[13]],
        vertexGeneration: words[14], rollbackGeneration: words[15] };
    } finally {
      if (readback.mapState === "mapped") readback.unmap(); readback.destroy();
    }
  }

  destroy(): void {
    if (this.destroyed) return; this.destroyed = true;
    this.apertures.destroy(); this.bodyImpulses.destroy(); this.control.destroy(); this.params.destroy();
    this.fallbackVertexArena?.destroy();
  }

  private writeOptions(options: OctreePowerSolidEncodeOptions): void {
    [...options.dimensions, ...options.physicalSpacing, ...options.container].forEach((value) => {
      if (!Number.isFinite(value) || value <= 0) throw new RangeError("Power solid geometry values must be finite and positive");
    });
    if (!Number.isSafeInteger(options.rigidBodyCount) || options.rigidBodyCount < 0
      || options.rigidBodyCount > this.plan.bodyCapacity) throw new RangeError("Power solid rigid body count exceeds capacity");
    const pressureScale = options.pressureImpulseScale ?? 1;
    if (!Number.isFinite(pressureScale)) throw new RangeError("Power solid pressure impulse scale must be finite");
    const bytes = new ArrayBuffer(64); const words = new Uint32Array(bytes); const floats = new Float32Array(bytes);
    words.set([this.plan.faceCapacity, options.rigidBodyCount, options.terrainEnabled ? 1 : 0, 0], 0);
    words.set([...options.dimensions, 0], 4);
    floats.set([...options.physicalSpacing, 0], 8);
    floats.set([...options.container, pressureScale], 12);
    this.device.queue.writeBuffer(this.params, 0, bytes);
  }

  private run(encoder: GPUCommandEncoder, label: string, pipeline: GPUComputePipeline, groups: number, group: GPUBindGroup): void {
    const pass = encoder.beginComputePass({ label }); pass.setPipeline(pipeline); pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(groups); pass.end();
  }

  private assertLive(): void { if (this.destroyed) throw new Error("Octree power solid faces are destroyed"); }
}

export const octreePowerSolidFaceShader = /* wgsl */ `
struct PowerFace { negativeRow:u32,positiveRow:u32,geometryCode:u32,flags:u32,normalVelocity:f32,area:f32,inverseDistance:f32,openFraction:f32 }
struct Aperture { openFraction:f32,solidNormalVelocity:f32,dominantOwner:i32,sampleMask:u32 }
struct FaceQuadrature { centroidArea:vec4f,sampleUV:array<u32,16> }
struct RigidBody { positionShape:vec4f,dimensions:vec4f,orientation:vec4f,linearVelocity:vec4f,angularVelocity:vec4f,inverseMassInertia:vec4f,angularMomentumRestitution:vec4f,material:vec4f }
struct Params { counts:vec4u,dims:vec4u,spacing:vec4f,container:vec4f }
@group(0) @binding(0) var<storage,read> faceControl:array<u32>;
@group(0) @binding(1) var<storage,read_write> faces:array<PowerFace>;
@group(0) @binding(2) var<storage,read> normals:array<vec4f>;
@group(0) @binding(3) var<storage,read> quadrature:array<FaceQuadrature>;
@group(0) @binding(4) var<storage,read> bodies:array<RigidBody,${GPU_RIGID_BODY_CAPACITY}>;
@group(0) @binding(5) var<uniform> params:Params;
@group(0) @binding(6) var<storage,read_write> apertures:array<Aperture>;
@group(0) @binding(7) var<storage,read_write> control:array<atomic<u32>>;
@group(0) @binding(8) var terrain:texture_2d<f32>;
struct SolidVertexArena { control:array<u32,16>,values:array<u32> }
@group(0) @binding(9) var<storage,read> solidVertices:SolidVertexArena;
const FACE_VALID:u32=${OCTREE_POWER_FACE_VALID}u;const SOLID_VALID:u32=${OCTREE_POWER_SOLID_VALID}u;
const VERTEX_VALID:u32=${OCTREE_SOLID_VERTEX_SDF_VALID}u;
const SAMPLE_COUNT:u32=16u;const INVALID:u32=0xffffffffu;
fn finite(v:f32)->bool{return v==v&&abs(v)<=3.402823e38;}
fn qConjugate(q:vec4f)->vec4f{return vec4f(q.x,-q.yzw);}
fn qRotate(q:vec4f,v:vec3f)->vec3f{let uv=cross(q.yzw,v);return v+2.0*(q.x*uv+cross(q.yzw,uv));}
fn localPoint(body:RigidBody,world:vec3f)->vec3f{return qRotate(qConjugate(body.orientation),world-body.positionShape.xyz);}
fn bodySdf(body:RigidBody,world:vec3f)->f32{let p=localPoint(body,world);let d=body.dimensions.xyz;let shape=i32(round(body.positionShape.w));
  if(shape==0){return length(p)-d.x;}if(shape==1){let q=abs(p)-0.5*d;return length(max(q,vec3f(0)))+min(max(q.x,max(q.y,q.z)),0.0);}
  if(shape==2){let q=vec3f(p.x,p.y-clamp(p.y,-0.5*d.y,0.5*d.y),p.z);return length(q)-d.x;}
  let q=vec2f(length(p.xz)-d.x,abs(p.y)-0.5*d.y);return length(max(q,vec2f(0)))+min(max(q.x,q.y),0.0);}
fn tangentBasis(n:vec3f)->mat2x3f{let helper=select(vec3f(0,1,0),vec3f(1,0,0),abs(n.x)<0.75);let a=normalize(cross(helper,n));return mat2x3f(a,cross(n,a));}
fn samplePoint(index:u32,sample:u32)->vec3f{let n=normals[index].xyz;let basis=tangentBasis(n);let q=quadrature[index];let uv=unpack2x16float(q.sampleUV[sample])*sqrt(q.centroidArea.w);
  return q.centroidArea.xyz+basis[0]*uv.x+basis[1]*uv.y;}
fn rigidWorld(point:vec3f)->vec3f{return point-vec3f(0.5*params.container.x,0.0,0.5*params.container.z);}
fn terrainHeightAt(grid:vec2f)->f32{let extent=vec2i(textureDimensions(terrain));if(any(extent<=vec2i(0))){return 0.0;}
  let p=grid-vec2f(0.5);let base=vec2i(floor(p));let t=clamp(p-vec2f(base),vec2f(0),vec2f(1));let hi=extent-vec2i(1);
  let a=textureLoad(terrain,clamp(base,vec2i(0),hi),0).x;let b=textureLoad(terrain,clamp(base+vec2i(1,0),vec2i(0),hi),0).x;
  let c=textureLoad(terrain,clamp(base+vec2i(0,1),vec2i(0),hi),0).x;let d=textureLoad(terrain,clamp(base+vec2i(1),vec2i(0),hi),0).x;
  return mix(mix(a,b,t.x),mix(c,d,t.x),t.y);}
fn terrainSolid(point:vec3f)->bool{if(params.counts.z==0u){return false;}let grid=point/params.spacing.xyz;return grid.y<terrainHeightAt(grid.xz);}
fn sampleOwner(point:vec3f)->i32{var best=3.402823e38;var owner=-1;let world=rigidWorld(point);for(var i=0u;i<params.counts.y;i+=1u){let d=bodySdf(bodies[i],world);if(d<best){best=d;owner=i32(i);}}
  if(best<0.0){return owner;}return select(-1,-2,terrainSolid(point));}
fn solidVelocity(owner:i32,point:vec3f,n:vec3f)->f32{if(owner<0){return 0.0;}let body=bodies[u32(owner)];let world=rigidWorld(point);return dot(body.linearVelocity.xyz+cross(body.angularVelocity.xyz,world-body.positionShape.xyz),n);}
fn terrainInputsValid()->bool{if(params.counts.z==0u){return true;}let source=solidVertices.control;return source[0]==0u&&source[1]==faceControl[0]&&source[2]==faceControl[0]
  &&source[3]==faceControl[0]*8u&&source[4]==faceControl[7]&&source[5]==VERTEX_VALID&&source[7]==1u&&source[8]==faceControl[7];}
fn sourceValid()->bool{return arrayLength(&faceControl)>=9u&&faceControl[3]==0u&&faceControl[8]==FACE_VALID&&faceControl[1]<=params.counts.x&&terrainInputsValid();}
fn validVertexRow(row:u32)->bool{if(params.counts.z==0u){return true;}if(row>=faceControl[0]||row*8u+7u>=arrayLength(&solidVertices.values)){return false;}
  for(var corner=0u;corner<8u;corner+=1u){if(!finite(bitcast<f32>(solidVertices.values[row*8u+corner]))){return false;}}return true;}
fn validFace(index:u32)->bool{if(index>=faceControl[1]||index>=arrayLength(&faces)||index>=arrayLength(&normals)||index>=arrayLength(&quadrature)||index>=arrayLength(&apertures)){return false;}
  let f=faces[index];let n=normals[index].xyz;let c=quadrature[index].centroidArea;return validVertexRow(f.negativeRow)&&(f.positiveRow==INVALID||validVertexRow(f.positiveRow))
    &&(f.flags&FACE_VALID)!=0u&&finite(f.area)&&f.area>0.0&&finite(f.normalVelocity)&&abs(length(n)-1.0)<2e-3&&all(vec4<bool>(finite(c.x),finite(c.y),finite(c.z),finite(c.w)))&&c.w>0.0&&abs(c.w-f.area)<=max(2e-5,f.area*5e-4);}
@compute @workgroup_size(64) fn classifyPowerSolidFaces(@builtin(global_invocation_id) gid:vec3u){let index=gid.x;if(index>=params.counts.x){return;}if(!sourceValid()){atomicOr(&control[0],1u);return;}if(index>=faceControl[1]){return;}if(!validFace(index)){atomicOr(&control[0],2u);return;}
  var occupied=0u;var mask=0u;var velocity=0.0;var counts:array<u32,${GPU_RIGID_BODY_CAPACITY}>;var terrainCount=0u;let n=normals[index].xyz;
  for(var sample=0u;sample<SAMPLE_COUNT;sample+=1u){let point=samplePoint(index,sample);let owner=sampleOwner(point);if(owner!=-1){occupied+=1u;mask|=1u<<sample;velocity+=solidVelocity(owner,point,n);if(owner>=0){counts[u32(owner)]+=1u;}else{terrainCount+=1u;}}}
  var dominant=select(-1,-2,terrainCount>0u);var largest=terrainCount;for(var i=0u;i<params.counts.y;i+=1u){if(counts[i]>largest){largest=counts[i];dominant=i32(i);}}
  apertures[index]=Aperture(1.0-f32(occupied)/f32(SAMPLE_COUNT),select(0.0,velocity/f32(occupied),occupied>0u),dominant,mask);
  atomicAdd(&control[1],1u);if(occupied>0u&&occupied<SAMPLE_COUNT){atomicAdd(&control[2],1u);}atomicAdd(&control[3],occupied);}
@compute @workgroup_size(1) fn finishPowerSolidFaces(){let count=select(0u,faceControl[1],arrayLength(&faceControl)>=9u);atomicStore(&control[4],count);atomicStore(&control[5],params.counts.y);atomicStore(&control[6],params.counts.z);
  atomicStore(&control[14],solidVertices.control[4]);atomicStore(&control[15],solidVertices.control[8]);
  if(atomicLoad(&control[0])==0u&&sourceValid()&&atomicLoad(&control[1])==count){atomicStore(&control[7],SOLID_VALID);}else{atomicStore(&control[7],0u);}}
@compute @workgroup_size(64) fn constrainPowerSolidFaces(@builtin(global_invocation_id) gid:vec3u){let index=gid.x;if(atomicLoad(&control[7])!=SOLID_VALID||index>=atomicLoad(&control[4])){return;}var f=faces[index];let a=apertures[index];f.openFraction=a.openFraction;f.normalVelocity=a.openFraction*f.normalVelocity+(1.0-a.openFraction)*a.solidNormalVelocity;faces[index]=f;}
@compute @workgroup_size(64) fn lockClosedPowerSolidFaces(@builtin(global_invocation_id) gid:vec3u){let index=gid.x;if(atomicLoad(&control[7])!=SOLID_VALID||index>=atomicLoad(&control[4])){return;}var f=faces[index];let a=apertures[index];if(a.openFraction<=0.0){f.normalVelocity=a.solidNormalVelocity;faces[index]=f;}}
`;

// A separate module keeps both stages within WebGPU's portable eight-storage-buffer limit.
export const octreePowerSolidImpulseShader = /* wgsl */ `
struct PowerFace { negativeRow:u32,positiveRow:u32,geometryCode:u32,flags:u32,normalVelocity:f32,area:f32,inverseDistance:f32,openFraction:f32 }
struct Aperture { openFraction:f32,solidNormalVelocity:f32,dominantOwner:i32,sampleMask:u32 }
struct FaceQuadrature { centroidArea:vec4f,sampleUV:array<u32,16> }
struct RigidBody { positionShape:vec4f,dimensions:vec4f,orientation:vec4f,linearVelocity:vec4f,angularVelocity:vec4f,inverseMassInertia:vec4f,angularMomentumRestitution:vec4f,material:vec4f }
struct Params { counts:vec4u,dims:vec4u,spacing:vec4f,container:vec4f }
@group(0) @binding(0) var<storage,read> faces:array<PowerFace>;
@group(0) @binding(1) var<storage,read> normals:array<vec4f>;
@group(0) @binding(2) var<storage,read> quadrature:array<FaceQuadrature>;
@group(0) @binding(3) var<storage,read> bodies:array<RigidBody,${GPU_RIGID_BODY_CAPACITY}>;
@group(0) @binding(4) var<storage,read> apertures:array<Aperture>;
@group(0) @binding(5) var<uniform> params:Params;
@group(0) @binding(6) var<storage,read_write> impulses:array<atomic<i32>>;
@group(0) @binding(7) var<storage,read_write> control:array<atomic<u32>>;
@group(0) @binding(8) var<storage,read> pressure:array<f32>;
const VALID:u32=${OCTREE_POWER_SOLID_VALID}u;const INVALID:u32=0xffffffffu;const SAMPLE_COUNT:u32=16u;
const FIXED_SCALE:f32=1000000.0;const MAX_FIXED:f32=2000000000.0;
fn qConjugate(q:vec4f)->vec4f{return vec4f(q.x,-q.yzw);}
fn qRotate(q:vec4f,v:vec3f)->vec3f{let uv=cross(q.yzw,v);return v+2.0*(q.x*uv+cross(q.yzw,uv));}
fn localPoint(body:RigidBody,world:vec3f)->vec3f{return qRotate(qConjugate(body.orientation),world-body.positionShape.xyz);}
fn bodySdf(body:RigidBody,world:vec3f)->f32{let p=localPoint(body,world);let d=body.dimensions.xyz;let shape=i32(round(body.positionShape.w));
  if(shape==0){return length(p)-d.x;}if(shape==1){let q=abs(p)-0.5*d;return length(max(q,vec3f(0)))+min(max(q.x,max(q.y,q.z)),0.0);}
  if(shape==2){let q=vec3f(p.x,p.y-clamp(p.y,-0.5*d.y,0.5*d.y),p.z);return length(q)-d.x;}
  let q=vec2f(length(p.xz)-d.x,abs(p.y)-0.5*d.y);return length(max(q,vec2f(0)))+min(max(q.x,q.y),0.0);}
fn tangentBasis(n:vec3f)->mat2x3f{let helper=select(vec3f(0,1,0),vec3f(1,0,0),abs(n.x)<0.75);let a=normalize(cross(helper,n));return mat2x3f(a,cross(n,a));}
fn samplePoint(index:u32,sample:u32)->vec3f{let n=normals[index].xyz;let basis=tangentBasis(n);let q=quadrature[index];let uv=unpack2x16float(q.sampleUV[sample])*sqrt(q.centroidArea.w);
  return q.centroidArea.xyz+basis[0]*uv.x+basis[1]*uv.y;}
fn rigidWorld(point:vec3f)->vec3f{return point-vec3f(0.5*params.container.x,0.0,0.5*params.container.z);}
fn sampleBody(world:vec3f)->i32{var best=3.402823e38;var owner=-1;for(var i=0u;i<params.counts.y;i+=1u){let d=bodySdf(bodies[i],world);if(d<best){best=d;owner=i32(i);}}return select(-1,owner,best<0.0);}
fn pressureAt(row:u32)->f32{if(row==INVALID||row>=arrayLength(&pressure)){return 0.0;}return pressure[row];}
fn checkedFixed(value:f32)->i32{if(!(value==value)||abs(value*FIXED_SCALE)>MAX_FIXED){atomicOr(&control[0],4u);return 0;}return i32(round(value*FIXED_SCALE));}
fn addReaction(owner:u32,force:vec3f,world:vec3f){if(owner>=params.counts.y||owner*8u+5u>=arrayLength(&impulses)){atomicOr(&control[0],8u);return;}let torque=cross(world-bodies[owner].positionShape.xyz,force);let base=owner*8u;
  atomicAdd(&impulses[base],checkedFixed(force.x));atomicAdd(&impulses[base+1u],checkedFixed(force.y));atomicAdd(&impulses[base+2u],checkedFixed(force.z));
  atomicAdd(&impulses[base+3u],checkedFixed(torque.x));atomicAdd(&impulses[base+4u],checkedFixed(torque.y));atomicAdd(&impulses[base+5u],checkedFixed(torque.z));
  atomicAdd(&control[8],bitcast<u32>(checkedFixed(force.x)));atomicAdd(&control[9],bitcast<u32>(checkedFixed(force.y)));atomicAdd(&control[10],bitcast<u32>(checkedFixed(force.z)));
  atomicAdd(&control[11],bitcast<u32>(checkedFixed(-force.x)));atomicAdd(&control[12],bitcast<u32>(checkedFixed(-force.y)));atomicAdd(&control[13],bitcast<u32>(checkedFixed(-force.z)));}
@compute @workgroup_size(64) fn accumulatePowerSolidImpulses(@builtin(global_invocation_id) gid:vec3u){let index=gid.x;if(atomicLoad(&control[7])!=VALID||index>=atomicLoad(&control[4])){return;}let aperture=apertures[index];if(aperture.sampleMask==0u||aperture.dominantOwner<0){return;}let face=faces[index];let scalar=params.container.w*face.area*(pressureAt(face.negativeRow)-pressureAt(face.positiveRow))/f32(SAMPLE_COUNT);let n=normals[index].xyz;
  for(var sample=0u;sample<SAMPLE_COUNT;sample+=1u){if((aperture.sampleMask&(1u<<sample))==0u){continue;}let point=samplePoint(index,sample);let world=rigidWorld(point);let owner=sampleBody(world);if(owner>=0){addReaction(u32(owner),n*scalar,world);}}}
`;

export const octreePowerSolidExchangeShader = /* wgsl */ `
@group(0) @binding(0) var<storage,read> staged:array<i32>;
@group(0) @binding(1) var<storage,read_write> exchange:array<atomic<i32>>;
@group(0) @binding(2) var<storage,read> control:array<u32>;
const VALID:u32=${OCTREE_POWER_SOLID_VALID}u;
@compute @workgroup_size(64) fn publishPowerSolidExchange(@builtin(global_invocation_id) gid:vec3u){let body=gid.x;if(control[0]!=0u||control[7]!=VALID||body>=control[5]||body*8u+5u>=arrayLength(&staged)||body*12u+5u>=arrayLength(&exchange)){return;}for(var word=0u;word<6u;word+=1u){atomicAdd(&exchange[body*12u+word],staged[body*8u+word]);}}
`;
