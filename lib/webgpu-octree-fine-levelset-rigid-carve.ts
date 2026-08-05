import { fineLevelSetPackedSampleWGSL } from "./fine-levelset-packed-sample";
import type { WebGPUFineLevelSetBrickSource } from "./webgpu-octree-fine-levelset-bricks";
import type { PassBroker } from "./webgpu-pass-broker";

export interface FineLevelSetRigidCarveOptions {
  readonly bodyCount: number;
  readonly rigidWorldOrigin: readonly [number, number, number];
}

export const fineLevelSetRigidCarveWGSL = /* wgsl */ `
const VALID:u32=1u;
struct Params{brickDimensions:vec3u,brickResolution:u32,sampleDimensions:vec3u,samplesPerBrick:u32,
 domainOrigin:vec3f,fineCellWidth:f32,pageCapacity:u32,generation:u32,bodyCount:u32,pad:u32,
 rigidWorldOrigin:vec3f,pad2:f32}
struct RigidBody{positionShape:vec4f,dimensions:vec4f,orientation:vec4f,linearVelocity:vec4f,
 angularVelocity:vec4f,inverseMassInertia:vec4f,angularMomentumRestitution:vec4f,material:vec4f}
@group(0)@binding(0)var<uniform>p:Params;
@group(0)@binding(1)var<storage,read>metadata:array<u32>;
@group(0)@binding(2)var<storage,read>worklist:array<u32>;
@group(0)@binding(3)var<storage,read_write>samples:array<u32>;
@group(0)@binding(4)var<storage,read>rigidBodies:array<RigidBody>;
${fineLevelSetPackedSampleWGSL("samples", true)}
fn qRotate(q:vec4f,v:vec3f)->vec3f{let uv=cross(q.yzw,v);return v+2.*(q.x*uv+cross(q.yzw,uv));}
fn qInverseRotate(q:vec4f,v:vec3f)->vec3f{return qRotate(vec4f(q.x,-q.yzw),v);}
fn rigidSdf(body:RigidBody,world:vec3f)->f32{
 let q=qInverseRotate(body.orientation,world-body.positionShape.xyz);let d=body.dimensions.xyz;
 let shape=i32(round(body.positionShape.w));if(shape==0){return length(q)-d.x;}
 if(shape==1){let b=abs(q)-.5*d;return length(max(b,vec3f(0)))+min(max(b.x,max(b.y,b.z)),0.);}
 if(shape==2){let cy=clamp(q.y,-.5*d.y,.5*d.y);return length(vec3f(q.x,q.y-cy,q.z))-d.x;}
 let radial=length(q.xz)-d.x;let axial=abs(q.y)-.5*d.y;
 return length(max(vec2f(radial,axial),vec2f(0)))+min(max(radial,axial),0.);
}
fn unpackBrick(key:u32)->vec3u{let xy=p.brickDimensions.x*p.brickDimensions.y;let z=key/xy;
 let r=key-z*xy;let y=r/p.brickDimensions.x;return vec3u(r-y*p.brickDimensions.x,y,z);}
fn localCoord(local:u32)->vec3u{let r=p.brickResolution;let z=local/(r*r);let q=local-z*r*r;
 let y=q/r;return vec3u(q-y*r,y,z);}
@compute @workgroup_size(64)
fn carveFinePhiAgainstRigids(@builtin(workgroup_id)group:vec3u,@builtin(local_invocation_index)lane:u32){
 if(p.bodyCount==0u||arrayLength(&worklist)<7u||worklist[0]!=p.generation||worklist[2]!=p.pageCapacity
   ||(worklist[3]&3u)!=3u||worklist[5]!=1u||worklist[6]!=1u){return;}
 let work=group.x+group.y*p.pad;if(work>=min(worklist[1],p.pageCapacity)){return;}let page=worklist[7u+work];
 if(page>=p.pageCapacity||4u*page+2u>=arrayLength(&metadata)||metadata[4u*page+2u]!=p.generation){return;}
 let index=page*p.samplesPerBrick+lane;if(lane>=p.samplesPerBrick||index>=arrayLength(&samples)
   ||(finePackedFlags(index)&VALID)==0u){return;}
 let brick=unpackBrick(metadata[4u*page+1u]);let q=brick*p.brickResolution+localCoord(lane);
 if(any(q>=p.sampleDimensions)){return;}let lattice=p.domainOrigin+(vec3f(q)+.5)*p.fineCellWidth;
 let world=p.rigidWorldOrigin+lattice;var solidPhi=3.402823e38;
 for(var body=0u;body<p.bodyCount&&body<arrayLength(&rigidBodies);body+=1u){solidPhi=min(solidPhi,rigidSdf(rigidBodies[body],world));}
 let old=finePackedPhi(index);if(old==old&&solidPhi<0.){fineWritePackedPhi(index,max(old,-solidPhi));}
}
`;

/** Carves only resident fine samples; extrapolated phi remains available on
 * the solid side through the analytic `-phiSolid` value rather than a hard
 * air sentinel. */
export class WebGPUFineLevelSetRigidCarve {
  readonly allocatedBytes = 80;
  private readonly params: GPUBuffer;
  private pipeline?: GPUComputePipeline;
  private group?: GPUBindGroup;

  constructor(private readonly device: GPUDevice, readonly source: WebGPUFineLevelSetBrickSource,
    private readonly rigidBodies: GPUBuffer, private readonly options: FineLevelSetRigidCarveOptions) {
    this.params = device.createBuffer({ label: "Fine rigid carve parameters", size: 80,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  }

  async initializePipelines(): Promise<void> {
    if (this.pipeline) return;
    const module = this.device.createShaderModule({ label: "Fine rigid-body phi carve",
      code: fineLevelSetRigidCarveWGSL });
    this.pipeline = await this.device.createComputePipelineAsync({ label: "Carve fine phi against rigids",
      layout: "auto", compute: { module, entryPoint: "carveFinePhiAgainstRigids" } });
    this.group = this.device.createBindGroup({ layout: this.pipeline.getBindGroupLayout(0), entries: [
      this.params, this.source.metadata, this.source.worklist, this.source.samples, this.rigidBodies,
    ].map((buffer, binding) => ({ binding, resource: { buffer } })) });
  }

  encode(broker: PassBroker): boolean {
    if (!this.pipeline || !this.group) throw new Error("Fine rigid carve pipeline is not initialized");
    const bodyCount = Math.min(this.options.bodyCount, Math.floor(this.rigidBodies.size / 128));
    if (bodyCount === 0) return false;
    const bytes = new ArrayBuffer(80), words = new Uint32Array(bytes), floats = new Float32Array(bytes);
    words.set(this.source.plan.brickDimensions, 0); words[3] = this.source.plan.brickResolution;
    words.set(this.source.plan.sampleDimensions, 4); words[7] = this.source.plan.samplesPerBrick;
    floats.set(this.source.plan.domainOrigin, 8); floats[11] = this.source.plan.fineCellWidth;
    const groups = this.source.plan.maximumResidentBricks;
    const groupsX = Math.min(groups, this.device.limits.maxComputeWorkgroupsPerDimension);
    words.set([this.source.plan.maximumResidentBricks, this.source.generation, bodyCount, groupsX], 12);
    floats.set(this.options.rigidWorldOrigin, 16);
    this.device.queue.writeBuffer(this.params, 0, bytes);
    const pass = broker.compute({ label: "Carve resident fine phi against rigid SDF" });
    pass.setPipeline(this.pipeline); pass.setBindGroup(0, this.group);
    pass.dispatchWorkgroups(groupsX, Math.ceil(groups / groupsX));
    return true;
  }

  destroy(): void { this.params.destroy(); this.pipeline = undefined; this.group = undefined; }
}
