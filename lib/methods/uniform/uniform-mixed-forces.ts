import {uniformMixedSourceWGSL} from "./uniform-mixed-source.wgsl";
import type { UniformMixedOwnership } from "./uniform-mixed-ownership";
import { uniformMixedTopologyWGSL } from "./uniform-mixed-topology.wgsl";
import { uniformMixedFaceAddressWGSL, uniformMixedFaceDispatchWGSL } from "./uniform-mixed-face-dispatch.wgsl";
import { uniformMixedVelocitySamplingWGSL } from "./uniform-mixed-velocity-sampling.wgsl";
import { uniformMixedVertexSamplingWGSL } from "./uniform-mixed-vertex-sampling.wgsl";
import { uniformMixedPressureReconstructionWGSL } from "./uniform-mixed-pressure-reconstruction.wgsl";

export interface UniformMixedForceFields {
  velocity: GPUTexture;
  advected: GPUTexture;
  phi: GPUTexture;
  volume: GPUTexture;
  centerPhi?: GPUTexture;
  negative: GPUBuffer;
  output: GPUTexture;
  outputNegative: GPUBuffer;
  /** h.xyz,dt; gravity,density,dynamic viscosity,surface tension;
   * noSlip,openTop,airborne,dust threshold. All f32. */
  params: GPUBuffer;
}

/** Apply native force terms once, after SL or MacCormack resampling. The
 * certified domain has no interior solids. Capillarity uses the same seam
 * reconstruction as pressure; viscosity uses physical local MAC spacing. */
export class UniformMixedForces {
  readonly allocatedBytes = 0;
  private pipelines: GPUComputePipeline[]=[];
  private regularPipelines: GPUComputePipeline[]=[];
  private readonly resources: GPUBindGroupLayout;
  constructor(private readonly device: GPUDevice, readonly ownership: UniformMixedOwnership, private readonly cachedGeometry=false,private readonly sourceParams?:GPUBuffer) {
    this.resources = device.createBindGroupLayout({ entries: [
      ...[0,1,2,3].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float" as const,viewDimension:"3d" as const}})),
      {binding:4,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},
      {binding:5,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"write-only",format:"rgba32float",viewDimension:"3d"}},
      {binding:6,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},
      {binding:8,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float",viewDimension:"3d"}},
      {binding:7,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},
      ...(sourceParams?[{binding:9,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform" as const}}]:[]),
    ] });
  }
  bind(f: UniformMixedForceFields): GPUBindGroup {
    if(this.cachedGeometry&&!f.centerPhi)throw new Error("Mixed forces require current center phi for cached geometry");
    const d=this.ownership.layout.lattice.dimensions;
    for(const [i,t] of [f.velocity,f.advected,f.phi,f.volume,f.output].entries()){
      if([t.width,t.height,t.depthOrArrayLayers].some((n,a)=>n!==d[a]!+(i===2?1:0)) || t.format!==(i===2||i===3?"r32float":"rgba32float"))
        throw new Error("Mixed forces require native cell and vertex fields");
    }
    if(f.output===f.velocity||f.output===f.advected||f.negative===f.outputNegative)throw new Error("Mixed force output must be disjoint");
    const bytes=4*(d[0]*d[1]+d[0]*d[2]+d[1]*d[2]);
    if(f.negative.size<bytes||f.outputNegative.size<bytes)throw new Error("Mixed forces require negative boundary planes");
    return this.device.createBindGroup({layout:this.resources,entries:[
      ...[f.velocity,f.advected,f.phi,f.volume].map((t,binding)=>({binding,resource:t.createView()})),
      {binding:4,resource:{buffer:f.negative}},{binding:5,resource:f.output.createView()},
      {binding:6,resource:{buffer:f.outputNegative}},{binding:7,resource:{buffer:f.params,size:48}},
      {binding:8,resource:(f.centerPhi??f.volume).createView()},
      ...(this.sourceParams?[{binding:9,resource:{buffer:this.sourceParams,size:176}}]:[]),
    ]});
  }
  async initialize():Promise<void>{
    const h=this.ownership.layout.lattice.cellSize_m;
    const module=this.device.createShaderModule({code:uniformMixedTopologyWGSL(this.ownership.layout,0)+/* wgsl */`
@group(1) @binding(0) var velocity:texture_3d<f32>;
@group(1) @binding(1) var advected:texture_3d<f32>;
@group(1) @binding(2) var phi:texture_3d<f32>;
@group(1) @binding(3) var volume:texture_3d<f32>;
@group(1) @binding(4) var<storage,read> negative:array<f32>;
@group(1) @binding(5) var output:texture_storage_3d<rgba32float,write>;
@group(1) @binding(6) var<storage,read_write> boundary:array<f32>;
struct UMForceParams {hDt:vec4f,physical:vec4f,controls:vec4f}
@group(1) @binding(7) var<uniform> force:UMForceParams;
@group(1) @binding(8) var centerPhi:texture_3d<f32>;
override umCachedGeometry:bool=false;
const UM_H=vec3f(${h.map(n=>`${n}`).join(",")});
${this.sourceParams?uniformMixedSourceWGSL(9):""}
${uniformMixedFaceAddressWGSL}
fn umLoadVertex(p:vec3u)->f32{return textureLoad(phi,vec3i(p),0).x;}
${uniformMixedVertexSamplingWGSL}
fn umLoadMixedFace(anchor:vec3i,axis:u32)->f32 {
 if(anchor[axis]<0){return negative[umNegativeBoundaryIndex(vec3u(max(anchor,vec3i(0))),axis)];}
 return textureLoad(velocity,anchor,0)[axis];
}
${uniformMixedVelocitySamplingWGSL}
fn umCellPhi(owner:UMOwner)->f32{if(umCachedGeometry){return textureLoad(centerPhi,vec3i(umOrigin(owner)),0).x;}return umSampleVertex(vec3f(umOrigin(owner))+vec3f(0.5*f32(owner.width)));}
fn umOccupancy(owner:UMOwner)->f32 {
 if(owner.width==0u){return 0.0;}
 return clamp(0.5-umCellPhi(owner)/(4.0*force.hDt.y*f32(owner.width)),0.0,1.0);
}
fn umAirborne(owner:UMOwner)->bool {
 if(owner.width==0u||force.controls.z<0.5){return false;}
 let v=textureLoad(volume,vec3i(umOrigin(owner)),0).x;
 if(v>1.0){return true;}
 if(v<=max(force.controls.w,0.05)){return false;}
 let origin=vec3i(umOrigin(owner));let w=i32(owner.width);
 if(any(origin-vec3i(2*w)<vec3i(0))||any(origin+vec3i(3*w)>vec3i(UM_D))){return false;}
 return umCellPhi(owner)>1.5*f32(w)*min(UM_H.x,min(UM_H.y,UM_H.z));
}
fn umNormal(owner:UMOwner)->vec3f {
 let origin=umOrigin(owner);var gradient=vec3f(0);
 for(var k=0u;k<8u;k++){
  let corner=umCorner(k,2u);
  gradient-=(2.0*vec3f(corner)-vec3f(1))*umVertexValue(origin+corner*owner.width)/(4.0*UM_H*f32(owner.width));
 }
 return gradient/max(length(gradient),1e-6);
}
fn umCurvature(owner:UMOwner)->f32 {
 let center=umNormal(owner);var divergence=vec3f(0);
 for(var axis=0u;axis<3u;axis++){
  var delta=0.0;var span=0.0;
  for(var side=0u;side<2u;side++){
   let sign=select(-1,1,side==1u);let first=umFace(owner,axis,sign,0u);
   if(first.neighbor.width==0u){continue;}
   var average=0.0;
   for(var part=0u;part<first.count;part++){average+=umNormal(umFace(owner,axis,sign,part).neighbor)[axis];}
   delta+=f32(sign)*(average/f32(first.count)-center[axis]);
   span+=0.5*f32(owner.width+first.neighbor.width)*UM_H[axis];
  }
  if(span>0.0){divergence[axis]=delta/span;}
 }
 return -((divergence.x+divergence.z)+divergence.y);
}
fn umPressure(owner:UMOwner)->f32{return umOccupancy(owner);}
fn umPressureSlope(owner:UMOwner)->vec3f{return umReconstructPressureSlope(owner);}
${uniformMixedPressureReconstructionWGSL}
fn umDiffusionValue(p:vec3f,axis:u32,width:f32)->f32 {
 var offset=vec3f(0.5*width);offset[axis]=width;
 let q=p-offset;let clamped=clamp(q,vec3f(0),vec3f(UM_D)-vec3f(width))+offset;
 // On a regular unit stencil, viscosity samples exact MAC sites. Use the
 // same canonical load directly instead of evaluating eight trilinear taps
 // whose weights are seven zeros and one one.
 var value=0.0;
 if(umRegularFine){value=umLoadMixedFace(vec3i(clamped-offset),axis);}
 else{value=umSampleVelocityComponent(clamped,axis);}
 return select(value,-value,force.controls.x>0.5&&any(q!=clamped-offset));
}
fn umLaplacian(face:UMFace)->f32 {
 let p=umFaceCenter(face);let width=f32(face.width);let center=umDiffusionValue(p,face.axis,width);var terms=vec3f(0);
 for(var axis=0u;axis<3u;axis++){
  var delta=vec3f(0);delta[axis]=width;
  terms[axis]=(umDiffusionValue(p+delta,face.axis,width)-2.0*center+umDiffusionValue(p-delta,face.axis,width))/(UM_H[axis]*UM_H[axis]*width*width);
 }
 return (terms.x+terms.y)+terms.z;
}
fn umForcedVelocity(owner:UMOwner,face:UMFace)->f32 {
 if(face.anchor[face.axis]<0){return negative[umNegativeBoundaryIndex(vec3u(max(face.anchor,vec3i(0))),face.axis)];}
 var value=textureLoad(advected,face.anchor,0)[face.axis];let dt=force.hDt.w;
 let occupancy=umOccupancy(owner);let neighborOccupancy=umOccupancy(face.neighbor);
 if(occupancy>0.0&&force.physical.z>0.0){value+=dt*(force.physical.z/force.physical.y)*umLaplacian(face);}
 if(face.axis==1u&&(occupancy>1e-5||neighborOccupancy>1e-5||umAirborne(owner)||umAirborne(face.neighbor))){value+=dt*force.physical.x;}
 if(force.physical.w>0.0&&face.neighbor.width!=0u){
  let gradient=umReconstructedPressureGradient(owner,face);
  if(gradient!=0.0){value+=dt*(force.physical.w/force.physical.y)*0.5*(umCurvature(owner)+umCurvature(face.neighbor))*gradient;}
 }
 ${this.sourceParams?`if(umSourceinflowStrength()>0.0){
  var sum=0.0;let u=(face.axis+1u)%3u;let v=(face.axis+2u)%3u;
  for(var j=0u;j<face.width;j++){for(var i=0u;i<face.width;i++){
   var q=face.anchor;q[u]+=i32(i);q[v]+=i32(j);
   sum+=umSourceapplyInflowSweptVelocity(q,vec3f(value))[face.axis];
  }}value=sum/f32(face.width*face.width);
 }`:""}
 return value;
}
${uniformMixedFaceDispatchWGSL("forces","umForcedVelocity(owner,face)").replace(" let origin=umOrigin(owner);",` let origin=umOrigin(owner);
 if(umRegularFine){
  for(var axis=0u;axis<3u;axis++){
   if(origin[axis]==0u){let face=umFace(owner,axis,-1,0u);boundary[umNegativeBoundaryIndex(origin,axis)]=umForcedVelocity(owner,face);}
  }
  textureStore(output,vec3i(origin),vec4f(umForcedVelocity(owner,umFace(owner,0u,1,0u)),umForcedVelocity(owner,umFace(owner,1u,1,0u)),umForcedVelocity(owner,umFace(owner,2u,1,0u)),0));
  return;
 }`)}
`});
    const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");if(errors.length)throw new Error(errors.map(m=>`${m.lineNum}: ${m.message}`).join("\n"));
    const layout=this.device.createPipelineLayout({bindGroupLayouts:[this.ownership.bindLayout,this.resources]});
    const compile=(width:number,regular:boolean)=>this.device.createComputePipelineAsync({layout,compute:{module,entryPoint:"forces",constants:{
      umDispatchX:this.ownership.dispatchX,umCellWidth:width,umCachedGeometry:+this.cachedGeometry,
      umInterfaceTiles:+!regular,umRegularTiles:+regular,umRegularFine:+(regular&&width===1)}}});
    this.pipelines=await Promise.all([1,2,4].map(width=>compile(width,false)));
    this.regularPipelines=await Promise.all([1,2,4].map(width=>compile(width,true)));
  }
  encode(encoder:GPUCommandEncoder,group:GPUBindGroup):void{
    if(this.pipelines.length!==3)throw new Error("Mixed forces are not initialized");
    const pass=encoder.beginComputePass({label:"Uniform mixed body forces"});pass.setBindGroup(0,this.ownership.bindGroup);pass.setBindGroup(1,group);this.ownership.dispatchRegular(pass,this.regularPipelines);this.ownership.dispatchSeams(pass,this.pipelines);pass.end();
  }
}
