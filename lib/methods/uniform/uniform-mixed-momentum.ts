import type { UniformMixedOwnership } from "./uniform-mixed-ownership";
import { uniformMixedTopologyWGSL } from "./uniform-mixed-topology.wgsl";
import { uniformMixedVelocitySamplingSource } from "./uniform-mixed-velocity-sampling.wgsl";
import { uniformMixedFaceAddressWGSL, uniformMixedFaceDispatchWGSL } from "./uniform-mixed-face-dispatch.wgsl";
import { uniformVelocityDepartureWGSL } from "./uniform-velocity-departure.wgsl";

export interface UniformMixedMomentumFields {
  extended: GPUTexture;
  coarseExtended: GPUTexture;
  coarsePhysical: GPUTexture;
  coarseWeight: GPUTexture;
  physical: GPUTexture;
  phase: GPUTexture;
  volume: GPUTexture;
  /** Current post-transport geometry, required when pressure-row culling is enabled. */
  centerPhi?: GPUTexture;
  predicted: GPUTexture;
  reversed: GPUTexture;
  negative: GPUBuffer;
  predictedNegative: GPUBuffer;
  reversedNegative: GPUBuffer;
  output: GPUTexture;
  outputNegative: GPUBuffer;
  /** h.xyz, dt; openTop, liquidOnly, operation, fixed limits (u32 flags).
   * Operation: 0 SL, 1 prediction, 2 reverse, 3 correction.
   * Fixed limits must be UNIFORM_MIXED_MOMENTUM_LIMITS. Keeping native loop
   * bounds in uniforms prevents explosive compiler unrolling of samplers. */
  params: GPUBuffer;
}

export const UNIFORM_MIXED_MOMENTUM_LIMITS = 32 | (16 << 8);

/** Native SL / bounded MacCormack momentum resampling on canonical MAC faces.
 * Extension and forces are separate stages, so MacCormack can extend its
 * prediction and apply forces exactly once after correction. All fields are
 * borrowed; unused predictor/corrector bindings may alias read-only inputs. */
export class UniformMixedMomentum {
  readonly allocatedBytes = 0;
  private readonly resources: GPUBindGroupLayout;
  private pipeline?: GPUComputePipeline[];
  private regularPipeline?:GPUComputePipeline;
  constructor(private readonly device: GPUDevice, readonly ownership: UniformMixedOwnership,
    private readonly defaultOptions = false, private readonly cullAir = false) {
    this.resources = device.createBindGroupLayout({ entries: [
      ...[0, 1, 2, 3, 4, 5].map(binding => ({ binding, visibility: GPUShaderStage.COMPUTE,
        texture: { sampleType: "unfilterable-float" as const, viewDimension: "3d" as const } })),
      ...[6, 7, 8].map(binding => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" as const } })),
      { binding: 9, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "rgba32float", viewDimension: "3d" } },
      { binding: 10, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 11, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      ...[12,13,14,15].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float" as const,viewDimension:"3d" as const}})),
    ] });
  }
  bind(fields: UniformMixedMomentumFields): GPUBindGroup {
    if(this.cullAir&&!fields.centerPhi)throw new Error("Prediction culling requires current center phi");
    const textures = [fields.extended, fields.physical, fields.phase, fields.volume, fields.predicted, fields.reversed];
    const d = this.ownership.layout.lattice.dimensions;
    for (const [i, texture] of [...textures, fields.output].entries()) {
      if ([texture.width, texture.height, texture.depthOrArrayLayers].some((n, a) => n !== d[a])
        || texture.format !== (i === 2 || i === 3 ? "r32float" : "rgba32float")) throw new Error("Mixed momentum requires native-sized canonical fields");
    }
    for(const t of [fields.coarseExtended,fields.coarsePhysical,fields.coarseWeight])
      if(t.format!=="rgba32float"||[t.width,t.height,t.depthOrArrayLayers].some((n,a)=>n!==d[a]!/4+2))throw new Error("Mixed momentum requires a current 4h sampling cache");
    const boundaries = [fields.negative, fields.predictedNegative, fields.reversedNegative];
    if (textures.includes(fields.output) || boundaries.includes(fields.outputNegative)) throw new Error("Mixed momentum output must be disjoint from its inputs");
    for (const buffer of [...boundaries, fields.outputNegative])
      if (buffer.size < 4 * (d[0] * d[1] + d[0] * d[2] + d[1] * d[2])) throw new Error("Mixed momentum requires complete negative boundary planes");
    return this.device.createBindGroup({ layout: this.resources, entries: [
      ...textures.map((texture, binding) => ({ binding, resource: texture.createView() })),
      ...boundaries.map((buffer, i) => ({ binding: 6 + i, resource: { buffer } })),
      { binding: 9, resource: fields.output.createView() }, { binding: 10, resource: { buffer: fields.outputNegative } },
      { binding: 11, resource: { buffer: fields.params, size: 32 } },
      {binding:15,resource:(fields.centerPhi??fields.volume).createView()},
      ...[fields.coarseExtended,fields.coarsePhysical,fields.coarseWeight].map((t,i)=>({binding:12+i,resource:t.createView()})),
    ] });
  }
  async initialize(): Promise<void> {
    const module = this.device.createShaderModule({ code: uniformMixedTopologyWGSL(this.ownership.layout, 0) + /* wgsl */ `
@group(1) @binding(0) var extended:texture_3d<f32>;
@group(1) @binding(1) var physical:texture_3d<f32>;
@group(1) @binding(2) var phase:texture_3d<f32>;
@group(1) @binding(3) var volume:texture_3d<f32>;
@group(1) @binding(4) var predicted:texture_3d<f32>;
@group(1) @binding(5) var reversed:texture_3d<f32>;
@group(1) @binding(6) var<storage,read> negative:array<f32>;
@group(1) @binding(7) var<storage,read> predictedNegative:array<f32>;
@group(1) @binding(8) var<storage,read> reversedNegative:array<f32>;
@group(1) @binding(9) var output:texture_storage_3d<rgba32float,write>;
@group(1) @binding(10) var<storage,read_write> boundary:array<f32>;
struct UMMomentumParams {hDt:vec4f,flags:vec4u}
@group(1) @binding(11) var<uniform> momentum:UMMomentumParams;
@group(1) @binding(12) var coarseExtended:texture_3d<f32>;
@group(1) @binding(13) var coarsePhysical:texture_3d<f32>;
@group(1) @binding(14) var coarseWeight:texture_3d<f32>;
@group(1) @binding(15) var centerPhi:texture_3d<f32>;
override umCullAir:bool=false;
// Numerical-option specialization, independent of ownership. The unified
// frame supports SL without liquid-only filtering; remove unused optional
// sampler payloads at compilation rather than branching per texture tap.
override umDefaultMomentum:bool=false;
${uniformMixedFaceAddressWGSL}
fn umLoadCoarseFace(index:vec3i,axis:u32)->vec3f {
 let p=index+vec3i(1);if(umDefaultMomentum){return vec3f(textureLoad(coarseExtended,p,0)[axis],0,0);}return vec3f(textureLoad(coarseExtended,p,0)[axis],textureLoad(coarsePhysical,p,0)[axis],textureLoad(coarseWeight,p,0)[axis]);
}
fn umFacePhase(anchor:vec3i,axis:u32)->f32 {
 var next=anchor;next[axis]+=1;let left=umOwnerAt(anchor);let right=umOwnerAt(next);
 var known=false;
 if(left.width!=0u){known=textureLoad(phase,vec3i(umOrigin(left)),0).x>0.5;}
 if(right.width!=0u){known=known||textureLoad(phase,vec3i(umOrigin(right)),0).x>0.5;}
 return select(0.0,1.0,known);
}
fn umLoadMixedFace(anchor:vec3i,axis:u32)->vec3f {
 if(umDefaultMomentum){
  if(anchor[axis]<0){return vec3f(negative[umNegativeBoundaryIndex(vec3u(max(anchor,vec3i(0))),axis)],0,0);}
  return vec3f(textureLoad(extended,anchor,0)[axis],0,0);
 }
 let weight=umFacePhase(anchor,axis);var extendedValue=0.0;var physicalValue=0.0;
 if(anchor[axis]<0){extendedValue=negative[umNegativeBoundaryIndex(vec3u(max(anchor,vec3i(0))),axis)];if(weight>0.0){physicalValue=extendedValue;}}
 else {extendedValue=textureLoad(extended,anchor,0)[axis];if(weight>0.0){physicalValue=textureLoad(physical,anchor,0)[axis];}}
 return vec3f(extendedValue,physicalValue,weight);
}
${uniformMixedVelocitySamplingSource(true,true)}
${[1,2,4].map(width => /* wgsl */ `
fn umMomentumBounds${width}(p:vec3f,axis:u32)->vec2f {
 var offset=vec3f(0.5);offset[axis]=1.0;var lower=vec3f(0);lower[axis]=-1.0;
 let q=clamp(p/${width}.0-offset,lower,vec3f(UM_D/${width}u)-vec3f(1));let base=vec3i(floor(q));let fraction=fract(q);
 var bounds=vec2f(1e30,-1e30);
 for(var k=0u;k<select(umCounts.w,8u,umRegularFine);k++){
  let bit=vec3i(umCorner(k,2u));let weights=select(vec3f(1)-fraction,fraction,bit==vec3i(1));
  if(weights.x*weights.y*weights.z>0.0){let at=base+bit;var value=0.0;if(at[axis]>=0){value=umVelocityTap${width}(at,axis).x;}bounds=vec2f(min(bounds.x,value),max(bounds.y,value));}
 }
 return bounds;
}`).join("\n")}
fn umMomentumBounds(p:vec3f,axis:u32)->vec2f {
 let weights=umVelocitySamplingWeights(p);var lower=1e30;var upper=-1e30;
 if(weights.x>0.0){let b=umMomentumBounds1(p,axis);lower=min(lower,b.x);upper=max(upper,b.y);}
 if(weights.y>0.0){let b=umMomentumBounds2(p,axis);lower=min(lower,b.x);upper=max(upper,b.y);}
 if(1.0-weights.x-weights.y>0.0){let b=umMomentumBounds4(p,axis);lower=min(lower,b.x);upper=max(upper,b.y);}
 return vec2f(lower,upper);
}
fn umClampMomentum(p:vec3f)->vec3f {
 var q=p;q.x=clamp(q.x,0.0,f32(UM_D.x));q.z=clamp(q.z,0.0,f32(UM_D.z));q.y=max(q.y,0.0);
 if(momentum.flags.x==0u){q.y=min(q.y,f32(UM_D.y));}return q;
}
fn umMomentumDeparture(position:vec3f,dt:f32,h:vec3f)->vec3f {
${uniformVelocityDepartureWGSL("umSampleVelocity", "umClampMomentum", "select(f32(umOwnerAt(clamp(vec3i(floor(point)),vec3i(0),vec3i(UM_D)-vec3i(1))).width),1.0,umRegularFine)", "i32(momentum.flags.w&255u)")}
}
fn umSamplePhysical(p:vec3f,axis:u32)->vec2f {
 let sample=umSampleVelocityComponent(p,axis);
 return vec2f(select(0.0,sample.y/max(sample.z,1e-30),sample.z>0.0),sample.z);
}
fn umMomentumLiquid(owner:UMOwner,face:UMFace)->bool {
 if(textureLoad(volume,vec3i(umOrigin(owner)),0).x>1e-5){return true;}
 return face.neighbor.width!=0u && textureLoad(volume,vec3i(umOrigin(face.neighbor)),0).x>1e-5;
}
fn umOriginalMomentum(face:UMFace)->f32 {
 if(face.anchor[face.axis]<0){return negative[umNegativeBoundaryIndex(vec3u(max(face.anchor,vec3i(0))),face.axis)];}
 return textureLoad(physical,face.anchor,0)[face.axis];
}
fn umClosedPositive(face:UMFace)->bool {
 return face.anchor[face.axis]==i32(UM_D[face.axis])-1 && !(face.axis==1u && momentum.flags.x!=0u);
}
fn umAdvectedMomentum(owner:UMOwner,face:UMFace,dt:f32)->f32 {
 let position=umFaceCenter(face);
 let departure=umMomentumDeparture(position,dt,momentum.hDt.xyz);
 if(umDefaultMomentum||momentum.flags.y==0u||!umMomentumLiquid(owner,face)){return umSampleVelocityComponent(departure,face.axis).x;}
 let supported=umSamplePhysical(departure,face.axis);if(supported.y>0.0){return supported.x;}
 for(var probe=1u;probe<=(momentum.flags.w>>8u);probe++){
  let recovered=umSamplePhysical(mix(departure,position,f32(probe)/16.0),face.axis);
  if(recovered.y>0.0){return recovered.x;}
 }
 return 0.0;
}
fn umCorrectedMomentum(owner:UMOwner,face:UMFace)->f32 {
 if(face.anchor[face.axis]<0){return umOriginalMomentum(face);}
 let departure=umMomentumDeparture(umFaceCenter(face),momentum.hDt.w,momentum.hDt.xyz);
 let bounds=umMomentumBounds(departure,face.axis);
 var original=0.0;var forward=0.0;var backward=0.0;
 if(face.anchor[face.axis]<0){
  let at=umNegativeBoundaryIndex(vec3u(max(face.anchor,vec3i(0))),face.axis);
  original=negative[at];forward=predictedNegative[at];backward=reversedNegative[at];
 }else{
  original=textureLoad(physical,face.anchor,0)[face.axis];forward=textureLoad(predicted,face.anchor,0)[face.axis];backward=textureLoad(reversed,face.anchor,0)[face.axis];
 }
 let corrected=forward+0.5*(original-backward);
 return select(corrected,forward,corrected<bounds.x||corrected>bounds.y);
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
 if(!umDefaultMomentum&&momentum.flags.z==3u){return umCorrectedMomentum(owner,face);}
 if(face.anchor[face.axis]<0 || (!umDefaultMomentum&&momentum.flags.z!=0u && umClosedPositive(face))){return umOriginalMomentum(face);}
 let dt=select(momentum.hDt.w,-momentum.hDt.w,!umDefaultMomentum&&momentum.flags.z==2u);
 let value=umAdvectedMomentum(owner,face,dt);
 if((umDefaultMomentum||momentum.flags.z==0u) && umClosedPositive(face)){return min(value,umOriginalMomentum(face));}return value;
}
${uniformMixedFaceDispatchWGSL("momentumStep", "umMomentum(owner,face)", false).replace(" let origin=umOrigin(owner);", ` let origin=umOrigin(owner);
 if(umRegularFine){
  for(var axis=0u;axis<3u;axis++){
   if(origin[axis]==0u){let face=umFace(owner,axis,-1,0u);boundary[umNegativeBoundaryIndex(origin,axis)]=umMomentum(owner,face);}
  }
  textureStore(output,vec3i(origin),vec4f(umMomentum(owner,umFace(owner,0u,1,0u)),umMomentum(owner,umFace(owner,1u,1,0u)),umMomentum(owner,umFace(owner,2u,1,0u)),0));
  return;
 }`)}
` });
    const errors = (await module.getCompilationInfo()).messages.filter(m => m.type === "error");
    if (errors.length) throw new Error(errors.map(m => `${m.lineNum}: ${m.message}`).join("\n"));
    const layout = this.device.createPipelineLayout({ bindGroupLayouts: [this.ownership.bindLayout, this.resources] });
    this.pipeline = await Promise.all([1,2,4].map(umCellWidth=>this.device.createComputePipelineAsync({ layout,
      compute: { module, entryPoint: "momentumStep", constants: { umCellWidth,umPlannedFine:2, umDispatchX: this.ownership.dispatchX, umDefaultMomentum: +this.defaultOptions, umCullAir:+this.cullAir } } })));
    this.regularPipeline=await this.device.createComputePipelineAsync({layout,compute:{module,entryPoint:"momentumStep",constants:{umCellWidth:1,umPlannedFine:1,umRegularFine:1,umDispatchX:this.ownership.dispatchX,umDefaultMomentum:+this.defaultOptions,umCullAir:+this.cullAir}}});
  }
  encode(encoder: GPUCommandEncoder, group: GPUBindGroup): void {
    if (!this.pipeline) throw new Error("Mixed momentum is not initialized");
    const pass = encoder.beginComputePass({ label: "Uniform mixed momentum" });
    pass.setBindGroup(0, this.ownership.bindGroup); pass.setBindGroup(1, group); this.ownership.dispatchCertified(pass, this.pipeline,this.regularPipeline!); pass.end();
  }
}
