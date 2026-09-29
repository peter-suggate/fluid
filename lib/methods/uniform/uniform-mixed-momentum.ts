import type { UniformMixedOwnership } from "./uniform-mixed-ownership";
import { UNIFORM_MIXED_JOBS, uniformMixedCertifiedEntriesWGSL, uniformMixedTopologyWGSL } from "./uniform-mixed-topology.wgsl";
import { uniformMixedVelocitySamplingSource } from "./uniform-mixed-velocity-sampling.wgsl";
import { uniformMixedFaceAddressWGSL, uniformMixedFaceTileDispatchWGSL } from "./uniform-mixed-face-dispatch.wgsl";
import { uniformVelocityDepartureWGSL } from "./uniform-velocity-departure.wgsl";

export interface UniformMixedMomentumFields {
  extended: GPUTexture;
  coarseExtended: GPUTexture;
  physical: GPUTexture;
  phase: GPUTexture;
  volume: GPUTexture;
  /** Current post-transport geometry, required when pressure-row culling is enabled. */
  centerPhi?: GPUTexture;
  negative: GPUBuffer;
  output: GPUTexture;
  outputNegative: GPUBuffer;
  /** UniformMixedHangingTaps.unitVelocity of `extended`; with hanging taps only. */
  unitVelocity?: GPUTexture;
  /** h.xyz, dt; openTop, 0, 0, fixed limits (u32 flags). Only openTop is
   * read; the layout matches the other mixed stages' parameter blocks, and
   * fixed limits must be UNIFORM_MIXED_MOMENTUM_LIMITS. */
  params: GPUBuffer;
}

export const UNIFORM_MIXED_MOMENTUM_LIMITS = 32 | (16 << 8);

/** Native semi-Lagrangian momentum resampling on canonical MAC faces. Extension
 * and forces are separate stages. All fields are borrowed. */
export class UniformMixedMomentum {
  readonly allocatedBytes = 0;
  private readonly resources: GPUBindGroupLayout;
  private pipeline?: GPUComputePipeline;
  private regularPipeline?:GPUComputePipeline;
  constructor(private readonly device: GPUDevice, readonly ownership: UniformMixedOwnership,
    private readonly cullAir = false, private readonly hanging = false) {
    this.resources = device.createBindGroupLayout({ entries: [
      ...[0, 1, 2, 3].map(binding => ({ binding, visibility: GPUShaderStage.COMPUTE,
        texture: { sampleType: "unfilterable-float" as const, viewDimension: "3d" as const } })),
      { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 9, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "rgba32float", viewDimension: "3d" } },
      { binding: 10, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 11, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      ...[12,15,...(hanging?[16]:[])].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float" as const,viewDimension:"3d" as const}})),
    ] });
  }
  bind(fields: UniformMixedMomentumFields): GPUBindGroup {
    if(this.cullAir&&!fields.centerPhi)throw new Error("Prediction culling requires current center phi");
    if(this.hanging!==(fields.unitVelocity!==undefined))throw new Error("Mixed momentum hanging taps require their unit velocity texture");
    const textures = [fields.extended, fields.physical, fields.phase, fields.volume];
    const d = this.ownership.capacity.lattice.dimensions;
    for (const [i, texture] of [...textures, fields.output].entries()) {
      if ([texture.width, texture.height, texture.depthOrArrayLayers].some((n, a) => n !== d[a])
        || texture.format !== (i === 2 || i === 3 ? "r32float" : "rgba32float")) throw new Error("Mixed momentum requires native-sized canonical fields");
    }
    for(const t of [fields.coarseExtended])
      if(t.format!=="rgba32float"||[t.width,t.height,t.depthOrArrayLayers].some((n,a)=>n!==d[a]!/4+2))throw new Error("Mixed momentum requires a current 4h sampling cache");
    if (textures.includes(fields.output) || fields.negative === fields.outputNegative) throw new Error("Mixed momentum output must be disjoint from its inputs");
    for (const buffer of [fields.negative, fields.outputNegative])
      if (buffer.size < 4 * (d[0] * d[1] + d[0] * d[2] + d[1] * d[2])) throw new Error("Mixed momentum requires complete negative boundary planes");
    return this.device.createBindGroup({ layout: this.resources, entries: [
      ...textures.map((texture, binding) => ({ binding, resource: texture.createView() })),
      { binding: 6, resource: { buffer: fields.negative } },
      { binding: 9, resource: fields.output.createView() }, { binding: 10, resource: { buffer: fields.outputNegative } },
      { binding: 11, resource: { buffer: fields.params, size: 32 } },
      {binding:15,resource:(fields.centerPhi??fields.volume).createView()},
      {binding:12,resource:fields.coarseExtended.createView()},
      ...(fields.unitVelocity?[{binding:16,resource:fields.unitVelocity.createView()}]:[]),
    ] });
  }
  async initialize(): Promise<void> {
    const module = this.device.createShaderModule({ code: uniformMixedCertifiedEntriesWGSL(uniformMixedTopologyWGSL(this.ownership.layout, 0) + /* wgsl */ `
@group(1) @binding(0) var extended:texture_3d<f32>;
@group(1) @binding(1) var physical:texture_3d<f32>;
@group(1) @binding(2) var phase:texture_3d<f32>;
@group(1) @binding(3) var volume:texture_3d<f32>;
@group(1) @binding(6) var<storage,read> negative:array<f32>;
@group(1) @binding(9) var output:texture_storage_3d<rgba32float,write>;
@group(1) @binding(10) var<storage,read_write> boundary:array<f32>;
struct UMMomentumParams {hDt:vec4f,flags:vec4u}
@group(1) @binding(11) var<uniform> momentum:UMMomentumParams;
@group(1) @binding(12) var coarseExtended:texture_3d<f32>;
@group(1) @binding(15) var centerPhi:texture_3d<f32>;
${this.hanging?"@group(1) @binding(16) var unitVelocity:texture_3d<f32>;":""}
override umCullAir:bool=false;
${uniformMixedFaceAddressWGSL}
fn umLoadCoarseFace(index:vec3i,axis:u32)->f32 {
 return textureLoad(coarseExtended,index+vec3i(1),0)[axis];
}
fn umLoadMixedFace(anchor:vec3i,axis:u32)->f32 {
 if(anchor[axis]<0){return negative[umNegativeBoundaryIndex(vec3u(max(anchor,vec3i(0))),axis)];}return textureLoad(extended,anchor,0)[axis];
}
${uniformMixedVelocitySamplingSource(false,true,"extended",this.hanging?2:undefined,this.hanging?"unitVelocity":undefined)}
fn umClampMomentum(p:vec3f)->vec3f {
 var q=p;q.x=clamp(q.x,0.0,f32(UM_D.x));q.z=clamp(q.z,0.0,f32(UM_D.z));q.y=max(q.y,0.0);
 if(momentum.flags.x==0u){q.y=min(q.y,f32(UM_D.y));}return q;
}
fn umMomentumDeparture(position:vec3f,dt:f32,h:vec3f)->vec3f {
${uniformVelocityDepartureWGSL("umSampleVelocity", "umClampMomentum", "select(f32(umOwnerAt(clamp(vec3i(floor(point)),vec3i(0),vec3i(UM_D)-vec3i(1))).width),1.0,umRegularFine)", "32")}
}
fn umOriginalMomentum(face:UMFace)->f32 {
 if(face.anchor[face.axis]<0){return negative[umNegativeBoundaryIndex(vec3u(max(face.anchor,vec3i(0))),face.axis)];}
 return textureLoad(physical,face.anchor,0)[face.axis];
}
fn umClosedPositive(face:UMFace)->bool {
 return face.anchor[face.axis]==i32(UM_D[face.axis])-1 && !(face.axis==1u && momentum.flags.x!=0u);
}
fn umAdvectedMomentum(face:UMFace,dt:f32)->f32 {
 return umSampleVelocityComponent(umMomentumDeparture(umFaceCenter(face),dt,momentum.hDt.xyz),face.axis);
}
// An exterior neighbor is not a pressure row. If the incident interior owner
// is empty too, projection writes zero and no RHS can read this prediction.
fn umPredictionCellLive(owner:UMOwner)->bool {
 if(owner.width==0u){return false;}
 let at=vec3i(umOrigin(owner));
 return textureLoad(volume,at,0).x>0.0||textureLoad(centerPhi,at,0).x<0.0;
}
fn umMomentum(owner:UMOwner,face:UMFace)->f32 {
 if(umCullAir&&!umPredictionCellLive(owner)&&!umPredictionCellLive(face.neighbor)){return 0.0;}
 if(face.anchor[face.axis]<0){return umOriginalMomentum(face);}
 let value=umAdvectedMomentum(face,momentum.hDt.w);
 if(umClosedPositive(face)){return min(value,umOriginalMomentum(face));}return value;
}
// A certified unit stencil has one MAC patch per face. Keep its geometry
// constant through the characteristic sampler; generic neighbour widths would
// otherwise make the face position and wall logic dynamically sized.
fn umRegularMomentumFace(owner:UMOwner,axis:u32,sign:i32)->UMFace {
 let origin=vec3i(umOrigin(owner));var probe=origin;probe[axis]+=sign;
 var anchor=origin;if(sign<0){anchor[axis]-=1;}
 return UMFace(umOwnerAt(probe),anchor,1u,1u,axis,sign);
}
// Each lane traces one component. The tile shares only the final packed
// store; each invocation evaluates only one characteristic. No additional
// persistent field is needed.
var<workgroup> umMomentumComponents:array<f32,192>;
@compute @workgroup_size(192) fn momentumRegularStep(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 let cell=lane%64u;let axis=lane/64u;
 let owner=umOwner(vec3u(group.x*64u+cell,group.y,0));
 var value=0.0;
 if(owner.width!=0u){
  let origin=umOrigin(owner);
  if(origin[axis]==0u){let face=umRegularMomentumFace(owner,axis,-1);boundary[umNegativeBoundaryIndex(origin,axis)]=umMomentum(owner,face);}
  value=umMomentum(owner,umRegularMomentumFace(owner,axis,1));
 }
 umMomentumComponents[lane]=value;workgroupBarrier();
 if(lane<64u&&owner.width!=0u){textureStore(output,vec3i(umOrigin(owner)),vec4f(umMomentumComponents[cell],umMomentumComponents[cell+64u],umMomentumComponents[cell+128u],0));}
}
${uniformMixedFaceTileDispatchWGSL("momentumStep", "umMomentum(owner,face)")}
`, ["momentumRegularStep", "momentumStep"]) });
    const errors = (await module.getCompilationInfo()).messages.filter(m => m.type === "error");
    if (errors.length) throw new Error(errors.map(m => `${m.lineNum}: ${m.message}`).join("\n"));
    const layout = this.device.createPipelineLayout({ bindGroupLayouts: [this.ownership.bindLayout, this.resources, ...(this.hanging ? [this.ownership.hangingLayout] : [])] });
    this.pipeline = await this.ownership.pipeline(layout, module, "momentumStep", UNIFORM_MIXED_JOBS.mergedQuad, { umMergedTiles:1, umCullAir:+this.cullAir });
    this.regularPipeline=await this.ownership.pipeline(layout,module,"momentumRegularStep",UNIFORM_MIXED_JOBS.planned,{umCellWidth:1,umPlannedFine:1,umRegularFine:1,umCullAir:+this.cullAir});
  }
  encode(encoder: GPUCommandEncoder, group: GPUBindGroup): void {
    if (!this.pipeline) throw new Error("Mixed momentum is not initialized");
    const pass = encoder.beginComputePass({ label: "Uniform mixed momentum" });
    pass.setBindGroup(0, this.ownership.bindGroup); pass.setBindGroup(1, group); if (this.hanging) pass.setBindGroup(2, this.ownership.hangingGroup); this.ownership.dispatchCertified(pass, this.pipeline,this.regularPipeline!); pass.end();
  }
}
