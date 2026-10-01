import {uniformMixedSourceWGSL} from "./uniform-mixed-source.wgsl";
import type { UniformMixedOwnership } from "./uniform-mixed-ownership";
import { UNIFORM_MIXED_COUNTED, UNIFORM_MIXED_FUSED_GATE, uniformMixedCountedEntriesWGSL, uniformMixedTopologyWGSL } from "./uniform-mixed-topology.wgsl";
import { uniformMixedFaceAddressWGSL, uniformMixedFaceDispatchWGSL, uniformMixedFaceTileDispatchWGSL, uniformMixedFarAirWGSL } from "./uniform-mixed-face-dispatch.wgsl";
import { uniformMixedVertexSamplingSource } from "./uniform-mixed-vertex-sampling.wgsl";
import { uniformMixedPressureReconstructionWGSL } from "./uniform-mixed-pressure-reconstruction.wgsl";
import { uniformMixedSolidPipeline, uniformMixedSolidWGSL, type UniformMixedSolid } from "./uniform-mixed-solid.wgsl";

export interface UniformMixedForceFields {
  advected: GPUTexture;
  phi: GPUTexture;
  volume: GPUTexture;
  centerPhi?: GPUTexture;
  /** The frame's 4h sampling cache of the extended velocity (UniformMixedMomentumCache). */
  coarseVelocity: GPUTexture;
  /** UniformMixedHangingTaps.unitVelocity of the same extended field. */
  unitVelocity: GPUTexture;
  negative: GPUBuffer;
  output: GPUTexture;
  outputNegative: GPUBuffer;
  /** h.xyz,dt; gravity,density,dynamic viscosity,surface tension;
   * noSlip,openTop,unused,dust threshold. All f32. */
  params: GPUBuffer;
  /** Scratch dead until the following pressure authority writes phase. */
  curvature?: GPUTexture;
  /** Dead transport/sharpening arena, one vec4 per possible owner. */
  normals?: GPUBufferBinding;
}

/** Apply native force terms once, after SL or MacCormack resampling. The
 * certified domain has no interior solids. Capillarity uses the same seam
 * reconstruction as pressure; viscosity uses physical local MAC spacing on
 * the frame's extended-velocity snapshot (umLaplacian).
 * The optional cache evaluates each normal and curvature once per owner,
 * instead of rebuilding their nested stencils at every incident face. Both
 * buffers are borrowed; the caller can skip these passes when tension is zero.
 * Static solids touch unit owners only: capillarity skips faces with a
 * closed cell and curvature is one-sided at closed cells, as natively. */
export class UniformMixedForces {
  readonly allocatedBytes = 0;
  private pipeline?: GPUComputePipeline;
  private normalPipeline?: GPUComputePipeline;
  private curvaturePipeline?: GPUComputePipeline;
  private inlinePipelines?:GPUComputePipeline[];
  private regularFinePipeline?: GPUComputePipeline;
  private readonly resources: GPUBindGroupLayout;
  private regularCoarsePipeline?: GPUComputePipeline;
  constructor(private readonly device: GPUDevice, readonly ownership: UniformMixedOwnership, private readonly cachedGeometry=false,private readonly sourceParams?:GPUBuffer,private readonly solid?:UniformMixedSolid,private readonly cachedCurvature=false) {
    this.resources = device.createBindGroupLayout({ entries: [
      ...[1,2,3,10,11].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float" as const,viewDimension:"3d" as const}})),
      {binding:4,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},
      {binding:5,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"write-only",format:"rgba32float",viewDimension:"3d"}},
      {binding:6,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},
      {binding:8,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float",viewDimension:"3d"}},
      {binding:7,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},
      ...(cachedCurvature?[{binding:12,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"read-write" as const,format:"r32float" as const,viewDimension:"3d" as const}},{binding:13,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage" as const}}]:[]),
      ...(sourceParams?[{binding:9,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform" as const}}]:[]),
    ] });
  }
  bind(f: UniformMixedForceFields): GPUBindGroup {
    if(this.cachedGeometry&&!f.centerPhi)throw new Error("Mixed forces require current center phi for cached geometry");
    if(this.cachedCurvature!==!!f.curvature||this.cachedCurvature!==!!f.normals)throw new Error("Mixed force curvature scratch must match the pipeline");
    const d=this.ownership.capacity.lattice.dimensions,coarse=f.coarseVelocity,unit=f.unitVelocity;
    if(coarse.format!=="rgba32float"||[coarse.width,coarse.height,coarse.depthOrArrayLayers].some((n,a)=>n!==d[a]!/4+2))throw new Error("Mixed force viscosity requires the 4h velocity cache");
    if(f.curvature&&(f.curvature.format!=="r32float"||[f.curvature.width,f.curvature.height,f.curvature.depthOrArrayLayers].some((n,a)=>n!==d[a])||[f.phi,f.volume,f.centerPhi].includes(f.curvature)))throw new Error("Mixed force curvature requires disjoint native scalar scratch");
    const normalBytes=16*d[0]*d[1]*d[2];
    if(f.normals&&(f.normals.size??f.normals.buffer.size-(f.normals.offset??0))<normalBytes)throw new Error("Mixed force normal scratch is too small");
    if(unit.format!=="rgba32float"||[unit.width,unit.height,unit.depthOrArrayLayers].some((n,a)=>n!==d[a]))throw new Error("Mixed force viscosity requires resolved h velocity taps");
    for(const [i,t] of [f.advected,f.phi,f.volume,f.output].entries()){
      if([t.width,t.height,t.depthOrArrayLayers].some((n,a)=>n!==d[a]!+(i===1?1:0)) || t.format!==(i===1||i===2?"r32float":"rgba32float"))
        throw new Error("Mixed forces require native cell and vertex fields");
    }
    if(f.output===f.advected||f.output===unit||f.negative===f.outputNegative)throw new Error("Mixed force output must be disjoint");
    const bytes=4*(d[0]*d[1]+d[0]*d[2]+d[1]*d[2]);
    if(f.negative.size<bytes||f.outputNegative.size<bytes)throw new Error("Mixed forces require negative boundary planes");
    return this.device.createBindGroup({layout:this.resources,entries:[
      ...[f.advected,f.phi,f.volume].map((t,i)=>({binding:i+1,resource:t.createView()})),
      {binding:4,resource:{buffer:f.negative}},{binding:5,resource:f.output.createView()},
      {binding:6,resource:{buffer:f.outputNegative}},{binding:7,resource:{buffer:f.params,size:48}},
      {binding:8,resource:(f.centerPhi??f.volume).createView()},
      {binding:10,resource:coarse.createView()},{binding:11,resource:unit.createView()},
      ...(f.curvature?[{binding:12,resource:f.curvature.createView()},{binding:13,resource:{...f.normals!,size:normalBytes}}]:[]),
      ...(this.sourceParams?[{binding:9,resource:{buffer:this.sourceParams,size:176}}]:[]),
    ]});
  }
  async initialize():Promise<void>{
    const h=this.ownership.capacity.lattice.cellSize_m;
    const module=this.device.createShaderModule({code:uniformMixedCountedEntriesWGSL(uniformMixedTopologyWGSL(this.ownership.capacity,0)+/* wgsl */`
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
override umForceCached:bool=false;
${this.cachedCurvature?"@group(1) @binding(12) var curvature:texture_storage_3d<r32float,read_write>; @group(1) @binding(13) var<storage,read_write> normals:array<vec4f>;":""}
const UM_H=vec3f(${h.map(n=>`${n}`).join(",")});
${this.sourceParams?uniformMixedSourceWGSL(9):""}
${uniformMixedFaceAddressWGSL}
fn umLoadVertex(p:vec3u)->f32{return textureLoad(phi,vec3i(p),0).x;}
${uniformMixedVertexSamplingSource("",false)}
${uniformMixedSolidWGSL(this.solid?2:undefined)}
fn umUnitClosed(owner:UMOwner)->bool{return umSolidEnabled()&&owner.width==1u&&umCellOpen(vec3i(umOrigin(owner)))<=1e-5;}
@group(1) @binding(10) var coarseVelocity:texture_3d<f32>;
@group(1) @binding(11) var unitVelocity:texture_3d<f32>;
fn umCellPhi(owner:UMOwner)->f32{if(umCachedGeometry){return textureLoad(centerPhi,vec3i(umOrigin(owner)),0).x;}return umSampleVertex(vec3f(umOrigin(owner))+vec3f(0.5*f32(owner.width)));}
fn umOccupancy(owner:UMOwner)->f32 {
 if(owner.width==0u){return 0.0;}
 return clamp(0.5-umCellPhi(owner)/(4.0*force.hDt.y*f32(owner.width)),0.0,1.0);
}
// A coarse owner holding more than a twentieth of an h cell: its faces fall
// (detached mass keeps them through projection; see uniform-mixed-detached-mass).
fn umCoarseMass(owner:UMOwner)->bool{return owner.width>1u&&textureLoad(volume,vec3i(umOrigin(owner)),0).x*f32(owner.width*owner.width*owner.width)>max(force.controls.w,0.05);}
fn umNormalRaw(owner:UMOwner)->vec3f {
 let origin=umOrigin(owner);var gradient=vec3f(0);
 for(var k=0u;k<8u;k++){
  let corner=umCorner(k,2u);
  gradient-=(2.0*vec3f(corner)-vec3f(1))*umVertexValue(origin+corner*owner.width)/(4.0*UM_H*f32(owner.width));
 }
 return gradient/max(length(gradient),1e-6);
}
fn umNormal(owner:UMOwner)->vec3f{${this.cachedCurvature?"if(umForceCached){return normals[owner.index].xyz;}":""}return umNormalRaw(owner);}
${this.cachedCurvature?`@compute @workgroup_size(64) fn cacheNormals(@builtin(global_invocation_id) gid:vec3u){
 let owner=umAllOwner(gid);if(owner.width==0u){return;}
 normals[owner.index]=vec4f(umNormalRaw(owner),0);
}`:""}
fn umCurvature(owner:UMOwner)->f32 {
 let center=umNormal(owner);var divergence=vec3f(0);
 for(var axis=0u;axis<3u;axis++){
  var delta=0.0;var span=0.0;
  for(var side=0u;side<2u;side++){
   let sign=select(-1,1,side==1u);let first=umFace(owner,axis,sign,0u);
   if(first.neighbor.width==0u||umUnitClosed(first.neighbor)){continue;}
   var average=0.0;
   for(var part=0u;part<first.count;part++){average+=umNormal(umFace(owner,axis,sign,part).neighbor)[axis];}
   delta+=f32(sign)*(average/f32(first.count)-center[axis]);
   span+=0.5*f32(owner.width+first.neighbor.width)*UM_H[axis];
  }
  if(span>0.0){divergence[axis]=delta/span;}
 }
 return -((divergence.x+divergence.z)+divergence.y);
}
// Geometry is unchanged between this owner pass and the face pass. A seam
// owner can supply many faces: compute its curvature once, using exactly
// the same stencil and reduction order as the inline evaluation.
${this.cachedCurvature?`@compute @workgroup_size(64) fn cacheCurvature(@builtin(global_invocation_id) gid:vec3u){
 let owner=umAllOwner(gid);if(owner.width==0u){return;}
 textureStore(curvature,vec3i(umOrigin(owner)),vec4f(umCurvature(owner)));
}`:""}
fn umForceCurvature(owner:UMOwner)->f32{${this.cachedCurvature?"if(umForceCached){return textureLoad(curvature,vec3i(umOrigin(owner))).x;}":""}return umCurvature(owner);}
fn umPressure(owner:UMOwner)->f32{return umOccupancy(owner);}
fn umPressureSlope(owner:UMOwner)->vec3f{return umReconstructPressureSlope(owner);}
${uniformMixedPressureReconstructionWGSL}
// Viscosity reads the frame's extended-velocity snapshot at exact MAC sites
// of the face's own width (the ungraded layout has widths 1 and 4 only): an
// h site is one UniformMixedHangingTaps.unitVelocity texel, a 4h site one
// UniformMixedMomentumCache texel (halo +1). Every h site lies within one
// cell of a unit owner, so its tile is a unit or slotted seam tile, whose
// texels the hanging fill wrote. A site past the domain clamps to the edge
// site, negated under no-slip.
fn umViscositySite(site:vec3i,axis:u32,width:u32)->f32 {
 let at=clamp(site,vec3i(0),vec3i(UM_D/width)-vec3i(1));
 var value=0.0;
 if(width==1u){value=textureLoad(unitVelocity,at,0)[axis];}else{value=textureLoad(coarseVelocity,at+vec3i(1),0)[axis];}
 return select(value,-value,force.controls.x>0.5&&any(site!=at));
}
fn umLaplacian(face:UMFace)->f32 {
 // A width-w face's anchor is its positive cell's last fine texel along the
 // axis and first transversely: anchor/w is that cell's width-w index.
 let site=face.anchor/vec3i(i32(face.width));let width=f32(face.width);
 let center=umViscositySite(site,face.axis,face.width);var terms=vec3f(0);
 for(var axis=0u;axis<3u;axis++){
  var delta=vec3i(0);delta[axis]=1;
  terms[axis]=(umViscositySite(site+delta,face.axis,face.width)-2.0*center+umViscositySite(site-delta,face.axis,face.width))/(UM_H[axis]*UM_H[axis]*width*width);
 }
 return (terms.x+terms.y)+terms.z;
}
fn umForcedVelocity(owner:UMOwner,face:UMFace)->f32 {
 if(face.anchor[face.axis]<0){return negative[umNegativeBoundaryIndex(vec3u(max(face.anchor,vec3i(0))),face.axis)];}
 var value=textureLoad(advected,face.anchor,0)[face.axis];let dt=force.hDt.w;
 let occupancy=umOccupancy(owner);let neighborOccupancy=umOccupancy(face.neighbor);
 if(occupancy>0.0&&force.physical.z>0.0){value+=dt*(force.physical.z/force.physical.y)*umLaplacian(face);}
 if(face.axis==1u&&(occupancy>1e-5||neighborOccupancy>1e-5||umCoarseMass(owner)||umCoarseMass(face.neighbor))){value+=dt*force.physical.x;}
 if(force.physical.w>0.0&&face.neighbor.width!=0u&&!umUnitClosed(owner)&&!umUnitClosed(face.neighbor)){
  let gradient=umReconstructedPressureGradient(owner,face);
  if(gradient!=0.0){value+=dt*(force.physical.w/force.physical.y)*0.5*(umForceCurvature(owner)+umForceCurvature(face.neighbor))*gradient;}
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
${uniformMixedFaceDispatchWGSL("forcesRegular","umForcedVelocity(owner,face)").replace(" let origin=umOrigin(owner);",` let origin=umOrigin(owner);
 if(umRegularFine){
  for(var axis=0u;axis<3u;axis++){
   if(origin[axis]==0u){let face=umFace(owner,axis,-1,0u);boundary[umNegativeBoundaryIndex(origin,axis)]=umForcedVelocity(owner,face);}
  }
  textureStore(output,vec3i(origin),vec4f(umForcedVelocity(owner,umFace(owner,0u,1,0u)),umForcedVelocity(owner,umFace(owner,1u,1,0u)),umForcedVelocity(owner,umFace(owner,2u,1,0u)),0));
  return;
 }`)}
// Regular 4h owners from the ownership's regular list: one lane per owner,
// no lanes spent rejecting seam tiles.
fn umForcesRegularCoarseOwner(gid:vec3u)->UMOwner{return umRegularCoarseOwner(gid.x+umDispatchX*64u*gid.y);}
// Certified far air (uniformMixedResidencyWord): every tile within two tiles
// of the owner's lies in an absent page, so this owner, every neighbour and
// every owner their pressure slopes read are far air (V=0, corner and so
// centre phi at least 16h, audited by the next census): occupancy zero, no
// coarse mass, no viscosity, gravity or surface-tension term. The forced
// face is the advected one, which the culled prediction makes exactly zero
// (neither side is live); momentum skips these owners with the same
// predicate and never stores them. Inflow may reach any face, so it takes
// the general evaluation.
${uniformMixedFarAirWGSL(this.sourceParams?"umSourceinflowStrength()>0.0":"false")}
fn umForcedVelocityFar(owner:UMOwner,face:UMFace,far:bool)->f32 {
 if(!far){return umForcedVelocity(owner,face);}
 return 0.0;
}
${uniformMixedFaceDispatchWGSL("forcesRegularCoarse","umForcedVelocityFar(owner,face,far)",false,"","umForcesRegularCoarseOwner").replace(" let origin=umOrigin(owner);"," let origin=umOrigin(owner);let far=umFarAirOwner(owner);")}
${uniformMixedFaceTileDispatchWGSL("forces","umForcedVelocity(owner,face)")}
`,["forcesRegular","forcesRegularCoarse","forces",...(this.cachedCurvature?["cacheNormals","cacheCurvature"]:[])])});
    const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");if(errors.length)throw new Error(errors.map(m=>`${m.lineNum}: ${m.message}`).join("\n"));
    const layout=this.device.createPipelineLayout({bindGroupLayouts:[this.ownership.bindLayout,this.resources,...(this.solid?[this.solid.bindLayout]:[])]});
    const compile=(entryPoint:string,constants:Record<string,number>)=>uniformMixedSolidPipeline(this.solid,s=>this.device.createComputePipelineAsync({layout,compute:{module,entryPoint,constants:{
      umDispatchX:this.ownership.dispatchX,umCachedGeometry:+this.cachedGeometry,umForceCached:+this.cachedCurvature,...constants,...s}}}));
    // Seam tiles of every tier and the small regular tiers share one launch,
    // so their serial per-tile latencies overlap; seam 4h tiles pack four
    // per job (fusedQuad). Every launch is GPU-counted: a regular tier that
    // rides the fused launch counts zero jobs in its own (skipFused gate).
    if(this.cachedCurvature)[this.normalPipeline,this.curvaturePipeline]=await Promise.all(["cacheNormals","cacheCurvature"].map(entry=>compile(entry,{umCountedJobs:UNIFORM_MIXED_COUNTED.all})));
    const skipFused=UNIFORM_MIXED_FUSED_GATE.skipFused;
    [this.pipeline,this.regularFinePipeline,this.regularCoarsePipeline]=await Promise.all([
      compile("forces",{umMergedTiles:1,umFusedJobs:1,umCountedJobs:UNIFORM_MIXED_COUNTED.fusedQuad}),
      compile("forcesRegular",{umCellWidth:1,umRegularTiles:1,umRegularFine:1,umCountedJobs:UNIFORM_MIXED_COUNTED.owners,umFusedRegularGate:skipFused}),
      compile("forcesRegularCoarse",{umCellWidth:4,umCountedJobs:UNIFORM_MIXED_COUNTED.regularCoarse,umFusedRegularGate:skipFused})]);
    if(this.cachedCurvature)this.inlinePipelines=await Promise.all([
      compile("forces",{umForceCached:0,umMergedTiles:1,umFusedJobs:1,umCountedJobs:UNIFORM_MIXED_COUNTED.fusedQuad}),
      compile("forcesRegular",{umForceCached:0,umCellWidth:1,umRegularTiles:1,umRegularFine:1,umCountedJobs:UNIFORM_MIXED_COUNTED.owners,umFusedRegularGate:skipFused}),
      compile("forcesRegularCoarse",{umForceCached:0,umCellWidth:4,umCountedJobs:UNIFORM_MIXED_COUNTED.regularCoarse,umFusedRegularGate:skipFused})]);
  }
  encode(encoder:GPUCommandEncoder,group:GPUBindGroup,capillarity=true,useCache=this.cachedCurvature):void{
    if(!this.pipeline)throw new Error("Mixed forces are not initialized");
    const pass=encoder.beginComputePass({label:"Uniform mixed body forces"});
    pass.setBindGroup(0,this.ownership.bindGroup);pass.setBindGroup(1,group);if(this.solid)pass.setBindGroup(2,this.solid.bindGroup);
    const variant=(p:GPUComputePipeline)=>this.solid?.select(p)??p;
    if(useCache&&capillarity&&this.curvaturePipeline){
      this.ownership.dispatchAllCounted(pass,variant(this.normalPipeline!));
      this.ownership.dispatchAllCounted(pass,variant(this.curvaturePipeline));
    }
    const [general,fine,coarse]=!useCache&&this.inlinePipelines?this.inlinePipelines:[this.pipeline,this.regularFinePipeline!,this.regularCoarsePipeline!];
    this.ownership.dispatchTierCounted(pass,variant(fine!),0);
    this.ownership.dispatchRegularCoarseCounted(pass,variant(coarse!));
    this.ownership.dispatchFusedCounted(pass,variant(general!));pass.end();
  }
}
