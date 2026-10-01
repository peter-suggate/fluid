import {uniformMixedSourceWGSL} from "./uniform-mixed-source.wgsl";
import { uniformMixedDetachedMassWGSL } from "./uniform-mixed-detached-mass.wgsl";
import type { UniformMixedOwnership } from "./uniform-mixed-ownership";
import { UNIFORM_MIXED_COUNTED, uniformMixedCountedEntriesWGSL, uniformMixedTopologyWGSL } from "./uniform-mixed-topology.wgsl";
import { uniformMixedFaceAddressWGSL, uniformMixedFaceDispatchWGSL } from "./uniform-mixed-face-dispatch.wgsl";
import { uniformMixedPressureSurfaceWGSL } from "./uniform-mixed-pressure-surface.wgsl";
import { uniformMixedPressureReconstructionSource } from "./uniform-mixed-pressure-reconstruction.wgsl";
import { uniformMixedPressureBoundaryIndexWGSL, uniformMixedPressureBoundaryLoop, uniformMixedPressureStorage } from "./uniform-mixed-pressure-boundary.wgsl";
import { uniformMixedSolidPipeline, uniformMixedSolidWGSL, type UniformMixedSolid } from "./uniform-mixed-solid.wgsl";

interface CommonFields {
 velocity:GPUTexture;
 negative:GPUBufferBinding;
 phi:GPUBufferBinding;
 /** h.xyz, dt; density, openTop, unused, dust threshold. */
 params:GPUBuffer;
}
export interface UniformMixedPressureRhsFields extends CommonFields {
 /** Native geometric excess/deficit divergence, already in inverse seconds. */
 correction:GPUTexture;
 rhs:GPUBufferBinding;
 minimum:GPUBufferBinding;
 pressure:GPUBufferBinding;
 /** Static solids: level-0 CM11a (open, V+x, V+y, V+z) at each owner origin. */
 topology?:GPUTexture;
 /** Coarse (band) mode: the forced field in simulation ownership, whose h
  * faces give each cut 4h face its exact flux sum(V_h u_h). */
 fine?:{velocity:GPUTexture;negative:GPUBuffer};
}
export interface UniformMixedPressureProjectionFields extends CommonFields {
 pressure:GPUBufferBinding;
 /** Geometric centre phi. Unread since airborne momentum was removed; drop with the frame binding. */
 centerPhi:GPUTexture;
 volume:GPUTexture;
 output:GPUTexture;
 outputNegative:GPUBufferBinding;
 topology?:GPUTexture;
}

/** Pressure/velocity coupling on exactly the same canonical patches as the
 * mixed matrix. With static solids, fine owners use the native CM11a face V,
 * p_min=0 solid rows and embedded contact release; the RHS pass publishes the
 * level-0 topology that the smoother and projection read. In coarse mode
 * (band pressure, all-4h ownership) the static all-4h solid record is the
 * topology instead: a cut face (either tile cut) takes the flux of its 16 h
 * faces, sum(V_h u_h), so the 4h divergence is the exact sum of the h ones and
 * the band's pure-Neumann components stay compatible; a closed owner is a
 * p_min=0 row. The caller builds one authoritative pressure phi. No field owns
 * another simulation; all buffers/textures are borrowed from the native host.
 * Both launches stride the owners of the resident pages (residentAll): an
 * absent page is certified far air (V=0, phi at least 16h), whose RHS, bound
 * and pressure the root setup (UniformMixedPressureCycles) writes and whose
 * faces nobody reads. A resident owner reads an absent neighbour's phi and
 * pressure (the setup's far values) and treats its volume as the zero the
 * census audits. */
export class UniformMixedPressureVelocity {
 readonly allocatedBytes=0;
 private readonly rhsLayout:GPUBindGroupLayout;
 private readonly projectLayout:GPUBindGroupLayout;
 private rhsPipeline?:GPUComputePipeline;
 private projectPipeline?:GPUComputePipeline;
 constructor(private readonly device:GPUDevice,readonly ownership:UniformMixedOwnership,private readonly sourceParams?:GPUBuffer,private readonly solid?:UniformMixedSolid,private readonly coarse=false){
  if(coarse&&(!solid?.coarse||ownership.layout.tiles.some(word=>(word&0xc0000000)!==0)))throw new Error("Coarse mixed pressure coupling requires the all-4h solid record and all-4h ownership");
  const texture=(binding:number)=>({binding,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float" as const,viewDimension:"3d" as const}});
  // Read-write buffer views allow disjoint ranges of the shared arena.
  const storage=(binding:number)=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage" as const}});
  const uniform={binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform" as const}};
  const topology=coarse?[texture(8),storage(9)]:solid?[{binding:8,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"write-only" as const,format:"rgba32float" as const,viewDimension:"3d" as const}}]:[];
  this.rhsLayout=device.createBindGroupLayout({entries:[texture(0),storage(1),storage(2),uniform,texture(4),storage(5),storage(6),storage(7),...topology]});
  this.projectLayout=device.createBindGroupLayout({entries:[texture(0),storage(1),storage(2),uniform,storage(4),texture(6),texture(7),
   {binding:8,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"write-only",format:"rgba32float",viewDimension:"3d"}},storage(9),...(sourceParams?[{binding:10,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform" as const}}]:[]),...(solid&&!coarse?[texture(11)]:[])]});
 }
 private scalar(view:GPUBufferBinding,count:number):GPUBufferBinding{
  const offset=view.offset??0,size=4*count;
  if((view.size??view.buffer.size-offset)<size||offset+size>view.buffer.size)throw new Error("Mixed pressure field is too small");
  return {buffer:view.buffer,offset,size};
 }
 private common(f:CommonFields):GPUBindGroupEntry[]{
  const d=this.ownership.layout.lattice.dimensions;
  if(f.velocity.format!=="rgba32float"||[f.velocity.width,f.velocity.height,f.velocity.depthOrArrayLayers].some((v,a)=>v!==d[a]))throw new Error("Mixed pressure requires native MAC extent");
  return [{binding:0,resource:f.velocity.createView()},{binding:1,resource:this.scalar(f.negative,d[0]*d[1]+d[0]*d[2]+d[1]*d[2])},
   {binding:2,resource:this.scalar(f.phi,this.ownership.layout.cellCount)},{binding:3,resource:{buffer:f.params,size:32}}];
 }
 private topology(t:GPUTexture|undefined):GPUTexture|undefined{
  if(!!t!==(!!this.solid&&!this.coarse))throw new Error("Mixed solid topology binding does not match stage mode");
  if(t&&(t.format!=="rgba32float"||[t.width,t.height,t.depthOrArrayLayers].some((v,a)=>v!==this.ownership.layout.lattice.dimensions[a])))throw new Error("Mixed solid topology requires the native lattice extent");
  return t;
 }
 bindRhs(f:UniformMixedPressureRhsFields):GPUBindGroup{
  const count=uniformMixedPressureStorage(this.ownership.layout).count,topology=this.topology(f.topology);
  if(!!f.fine!==this.coarse)throw new Error("Coarse mixed pressure RHS needs the simulation-ownership forced field, and only it");
  const d=this.ownership.layout.lattice.dimensions;
  return this.device.createBindGroup({layout:this.rhsLayout,entries:[...this.common(f),{binding:4,resource:f.correction.createView()},
   ...[f.rhs,f.minimum,f.pressure].map((v,i)=>({binding:5+i,resource:this.scalar(v,count)})),...(topology?[{binding:8,resource:topology.createView()}]:[]),
   ...(f.fine?[{binding:8,resource:f.fine.velocity.createView()},{binding:9,resource:{buffer:f.fine.negative,size:4*(d[0]*d[1]+d[0]*d[2]+d[1]*d[2])}}]:[])]});
 }
 bindProjection(f:UniformMixedPressureProjectionFields):GPUBindGroup{
  if(f.velocity===f.output||f.negative.buffer===f.outputNegative.buffer)throw new Error("Mixed projection requires disjoint velocity output");
  const d=this.ownership.layout.lattice.dimensions;
  return this.device.createBindGroup({layout:this.projectLayout,entries:[...this.common(f),
   {binding:4,resource:this.scalar(f.pressure,uniformMixedPressureStorage(this.ownership.layout).count)},
   {binding:6,resource:f.centerPhi.createView()},{binding:7,resource:f.volume.createView()},{binding:8,resource:f.output.createView()},
   {binding:9,resource:this.scalar(f.outputNegative,d[0]*d[1]+d[0]*d[2]+d[1]*d[2])},
   ...(this.sourceParams?[{binding:10,resource:{buffer:this.sourceParams,size:176}}]:[]),
   ...(this.topology(f.topology)?[{binding:11,resource:f.topology!.createView()}]:[])]});
 }
 async initialize():Promise<void>{
  const ownership=this.ownership,h=ownership.layout.lattice.cellSize_m;
  const common=uniformMixedTopologyWGSL(ownership.capacity,0)+/* wgsl */`
const UM_H=vec3f(${h.join(",")});
@group(1) @binding(0) var velocity:texture_3d<f32>;
@group(1) @binding(1) var<storage,read_write> negative:array<f32>;
@group(1) @binding(2) var<storage,read_write> phi:array<f32>;
struct UMProjectionParams {hDt:vec4f,policy:vec4f}
@group(1) @binding(3) var<uniform> params:UMProjectionParams;
${uniformMixedFaceAddressWGSL}
${uniformMixedPressureBoundaryIndexWGSL(ownership.layout)}
fn umPressurePhi(o:UMOwner)->f32{return phi[o.index];}
${uniformMixedPressureSurfaceWGSL}
${uniformMixedSolidWGSL(this.solid?2:undefined,this.coarse?this.solid!.coarse!.count:undefined)}
fn umOpenTop(face:UMFace)->bool{return params.policy.y>0.5&&face.axis==1u&&face.sign>0;}
// An owner more than its width below the surface: its domain walls hold (no
// p=0 halo bound, no release). Only air can fill a separation gap, and air
// reaches a wall only where the surface meets it. Deeper, a p=0 halo is
// vacuum: the coarse-first pool's collapsing crater released the floor under
// it, the release turned deep floor vertices to air, and the phi-only sheet
// kept its tiles at h. The band keeps its clamped halo (holding it there
// breaks the dam's monotone acceptance at the far-wall impact).
fn umWallHeld(o:UMOwner)->bool{return umPressurePhi(o)< -f32(o.width)*min(UM_H.x,min(UM_H.y,UM_H.z));}
fn umFaceVelocity(face:UMFace)->f32 {
 if(face.anchor[face.axis]<0){return negative[umNegativeBoundaryIndex(vec3u(max(face.anchor,vec3i(0))),face.axis)];}
 return textureLoad(velocity,face.anchor,0)[face.axis];
}
`;
  const rhsSource=common+/* wgsl */`
@group(1) @binding(4) var correction:texture_3d<f32>;
@group(1) @binding(5) var<storage,read_write> rhs:array<f32>;
@group(1) @binding(6) var<storage,read_write> minimum:array<f32>;
@group(1) @binding(7) var<storage,read_write> pressures:array<f32>;
fn umPressure(o:UMOwner)->f32{return pressures[o.index];}
fn umPressureSlope(o:UMOwner)->vec3f{return vec3f(0);}
${this.coarse?`@group(1) @binding(8) var fineVelocity:texture_3d<f32>;
@group(1) @binding(9) var<storage,read_write> fineNegative:array<f32>;
// Either tile of a 4h face cut: its flux is the sum of its 16 h faces'.
fn umCoarseCut(o:UMOwner,axis:u32,sign:i32)->bool{
 let t=vec3i(umOrigin(o)/4u);if(umSolidCut(umTileAt(vec3u(t)))){return true;}
 var n=t;n[axis]+=sign;if(n[axis]<0||n[axis]>=i32(UM_T[axis])){return false;}
 return umSolidCut(umTileAt(vec3u(n)));
}
// sum(V_h u_h) over the h faces of one 4h face, from the h forced field,
// plus each face's moving-wall term.
fn umCutFlux(o:UMOwner,axis:u32,sign:i32)->f32{
 let origin=vec3i(umOrigin(o));let a=(axis+1u)%3u;let b=(axis+2u)%3u;var sum=0.0;
 for(var j=0;j<4;j++){for(var i=0;i<4;i++){
  var c=origin;c[a]+=i;c[b]+=j;c[axis]+=select(-1,3,sign>0);
  var u=0.0;
  if(c[axis]<0){u=fineNegative[umNegativeBoundaryIndex(vec3u(max(c,vec3i(0))),axis)];}else{u=textureLoad(fineVelocity,c,0)[axis];}
  let volume=umPressureFaceV(c,axis);sum+=volume*u;
  // A moving wall's (V_i - V) u_s, V_i the owner-side h cell (divergenceAtWithCapacity).
  if(umBodyCount()>0u){var inner=c;if(sign<0){inner[axis]+=1;}sum+=(umCellOpen(inner)-volume)*umSolidFaceVelocity(c,axis);}
 }}
 return sum;
}
// A 4h face's flux: exact on cut faces, the record-free fraction otherwise
// (an uncut face's record V is exactly 1, or 1/2 at a wall).
fn umCoarseFlux(o:UMOwner,face:UMFace,fraction:f32)->f32{
 if(umCoarseCut(o,face.axis,face.sign)){return umCutFlux(o,face.axis,face.sign);}
 return fraction*umFaceVelocity(face)*f32(face.width*face.width);
}`:this.solid?`@group(1) @binding(8) var topologyOut:texture_storage_3d<rgba32float,write>;
// mgBuildFinestTopology on fine owners. Coarse owners never touch a solid
// (promotion certificate): open 1, interior V 1, walls 1/2, ambient lid 1.
fn umOwnerTopology(o:UMOwner)->vec4f{
 let p=vec3i(umOrigin(o));
 if(o.width==1u){return vec4f(umCellOpen(p),umPressureFaceV(p,0u),umPressureFaceV(p,1u),umPressureFaceV(p,2u));}
 var t=vec4f(1.0);
 for(var axis=0u;axis<3u;axis++){if(umOrigin(o)[axis]+o.width==UM_D[axis]&&!(axis==1u&&params.policy.y>0.5)){t[axis+1u]=0.5;}}
 return t;
}
// Low faces of a fine owner: the neighbour's positive V, or the wall's.
fn umLowFaceV(o:UMOwner,t:vec4f,axis:u32)->f32{
 if(o.width!=1u){return select(1.0,0.5*t.x,umOrigin(o)[axis]==0u);}
 var q=vec3i(umOrigin(o));q[axis]-=1;return umPressureFaceV(q,axis);
}`:""}
@compute @workgroup_size(64) fn buildRhs(@builtin(global_invocation_id) gid:vec3u){
 let o=umResidentAllOwner(gid);if(o.width==0u){return;}var terms:array<f32,6>;
 ${this.coarse?"let topology=umSolidCoarse(o.index);":this.solid?"let topology=umOwnerTopology(o);textureStore(topologyOut,vec3i(umOrigin(o)),topology);":""}
 // Only a liquid owner keeps its divergence: air skips the face loads.
 let liquid=umPressureLiquid(o);
 if(liquid){for(var axis=0u;axis<3u;axis++){for(var side=0u;side<2u;side++){
  let sign=select(-1,1,side==1u);let first=umFace(o,axis,sign,0u);var flux=0.0;
  for(var part=0u;part<first.count;part++){
   let face=umFace(o,axis,sign,part);var fraction=1.0;
   if(face.neighbor.width==0u&&!umOpenTop(face)){fraction=0.5;}
   ${this.solid&&!this.coarse?"if(o.width==1u){fraction=select(umLowFaceV(o,topology,axis),topology[axis+1u],sign>0);}":""}
   ${this.coarse?"flux+=umCoarseFlux(o,face,fraction);":"flux+=fraction*umFaceVelocity(face)*f32(face.width*face.width);"}
  }
  terms[2u*axis+side]=f32(sign)*flux/(f32(o.width*o.width*o.width)*UM_H[axis]);
 }}}
 var value=0.0;
 if(liquid){let divergence=(terms[0]+terms[1])+(terms[4]+terms[5])+(terms[2]+terms[3]);
  value=-params.policy.x*(divergence-textureLoad(correction,vec3i(umOrigin(o)),0).x)/params.hDt.w;}
 rhs[o.index]=value;
 minimum[o.index]=${this.coarse?"select(-3.402823e38,0.0,topology.x<=1e-5)":this.solid?"select(-3.402823e38,0.0,o.width==1u&&umCellInsideSolid(vec3i(umOrigin(o))))":"-3.402823e38"};
 // The coarse root warm-starts from the caller's last pressure; air holds 0.
 pressures[o.index]=${this.coarse?"select(0.0,max(pressures[o.index],minimum[o.index]),liquid)":"0.0"};
 ${uniformMixedPressureBoundaryLoop(`let face=umFace(o,axis,sign,0u);let open=umOpenTop(face);
  let b=f32(sign)*params.policy.x*${this.coarse?"select(0.5*umFaceVelocity(face),umCutFlux(o,axis,sign)/16.0,umCoarseCut(o,axis,sign))":this.solid?"select(umLowFaceV(o,topology,axis),topology[axis+1u],sign>0)*umFaceVelocity(face)":"0.5*umFaceVelocity(face)"}/(f32(o.width)*UM_H[axis]*params.hDt.w);
  rhs[halo]=select(0.0,b,liquid&&!open);minimum[halo]=select(0.0,-3.402823e38,open||umWallHeld(o));pressures[halo]=${this.coarse?"select(0.0,max(pressures[halo],minimum[halo]),liquid&&!open)":"0.0"};`)}
}
`;
  const projectSource=common+/* wgsl */`
@group(1) @binding(4) var<storage,read_write> pressures:array<f32>;
@group(1) @binding(6) var centerPhi:texture_3d<f32>;
@group(1) @binding(7) var volume:texture_3d<f32>;
@group(1) @binding(8) var output:texture_storage_3d<rgba32float,write>;
@group(1) @binding(9) var<storage,read_write> boundary:array<f32>;
fn umPressure(o:UMOwner)->f32{return pressures[o.index];}
${this.sourceParams?uniformMixedSourceWGSL(10):""}
// The all-4h root has no seams: every reconstruction slope is zero.
fn umPressureSlope(o:UMOwner)->vec3f{return vec3f(0);}
${uniformMixedPressureReconstructionSource(true)}
${this.coarse?`// The record V of a 4h face: its low owner's V+, or the wall halo's.
fn umProjectV(o:UMOwner,face:UMFace)->f32{
 if(face.neighbor.width==0u){return umSolidCoarse(umBoundaryIndex(o,face.axis,face.sign)).x;}
 var low=o;if(face.sign<0){low=face.neighbor;}
 return umSolidCoarse(low.index)[face.axis+1u];
}`:this.solid?`@group(1) @binding(11) var topology:texture_3d<f32>;
// Face V of a fine owner's face (seams and coarse faces are solid-free).
fn umProjectV(o:UMOwner,face:UMFace)->f32{
 if(o.width!=1u||(face.neighbor.width!=0u&&face.neighbor.width!=1u)){return 1.0;}
 if(face.anchor[face.axis]<0){return 0.5*textureLoad(topology,vec3i(umOrigin(o)),0).x;}
 return textureLoad(topology,face.anchor,0)[face.axis+1u];
}
fn umSolidPressure(o:UMOwner)->f32{return select(0.0,umPressure(o),umPressureLiquid(o));}`:""}
// An absent page's volume is not transferred: it is the certified V=0.
${uniformMixedDetachedMassWGSL(o=>`umPressureLiquid(${o})`,o=>`select(0.0,textureLoad(volume,vec3i(umOrigin(${o})),0).x,umTileResident(${o}.tile))`,"params.policy.w")}
fn umProject(o:UMOwner,face:UMFace)->f32{
 let liquid=umPressureLiquid(o);let scale=params.hDt.w/params.policy.x;
 // Air on both sides keeps its velocity only beside detached mass. Any
 // other air face is zero whatever its V: skip the V and velocity loads.
 if(!liquid&&(face.neighbor.width==0u||!umPressureLiquid(face.neighbor))
  &&(face.neighbor.width==0u||!(umDetachedMass(o)||umDetachedMass(face.neighbor)))){return 0.0;}
 ${this.solid?"if(umProjectV(o,face)<=1e-6){return 0.0;}":""}
 let v=umFaceVelocity(face);
 if(face.neighbor.width==0u){
  if(!liquid){return v;}
  var other=pressures[umBoundaryIndex(o,face.axis,face.sign)];var theta=1.0;
  if(umOpenTop(face)){other=0.0;theta=umPressureSurfaceTheta(umPressurePhi(o),0.5*f32(o.width)*min(UM_H.x,min(UM_H.y,UM_H.z)),f32(o.width)*min(UM_H.x,min(UM_H.y,UM_H.z)));}
  return v-scale*f32(face.sign)*(other-umPressure(o))/(f32(o.width)*UM_H[face.axis]*theta);
 }
 if(!liquid&&!umPressureLiquid(face.neighbor)){return v;}
 return v-scale*umReconstructedPressureGradient(o,face);
}
fn umProjectWithSource(o:UMOwner,face:UMFace)->f32{
 let value=umProject(o,face);
 ${this.sourceParams?`if(face.anchor[face.axis]>=0&&umSourceinflowStrength()>0.0){
  var sum=0.0;let u=(face.axis+1u)%3u;let v=(face.axis+2u)%3u;
  for(var j=0u;j<face.width;j++){for(var i=0u;i<face.width;i++){
   var q=face.anchor;q[u]+=i32(i);q[v]+=i32(j);
   sum+=umSourceapplyInflowVelocity(q,vec3f(value))[face.axis];
  }}return sum/f32(face.width*face.width);
 }`:""}
 return value;
}
fn umRelease(o:UMOwner,face:UMFace,v:f32)->bool{
 if(face.neighbor.width!=0u||umOpenTop(face)||umWallHeld(o)){return false;}
 ${this.coarse?"if(umSolidCoarse(o.index).x<=1e-5||umProjectV(o,face)<=1e-6){return false;}":this.solid?"if(o.width==1u&&(umCellOpen(vec3i(umOrigin(o)))<=1e-5||umProjectV(o,face)<=1e-6)){return false;}":""}
 let pressure=select(0.0,pressures[umBoundaryIndex(o,face.axis,face.sign)],umPressureLiquid(o));
 return pressure<=0.0&&-f32(face.sign)*v*params.hDt.w>1e-4*f32(o.width)*UM_H[face.axis];
}
// The projection's schedule slot opens only on an accepted solve.
fn umProjectOwner(gid:vec3u)->UMOwner{if(umSlotClosed()){return UMOwner();}return umResidentAllOwner(gid);}
${uniformMixedFaceDispatchWGSL("project","umProjectWithSource(owner,face)",true,`
   var released=0u;
   for(var axis=0u;axis<3u;axis++){
    // Every positive release needs a nonzero component (+-0*dt never
    // exceeds 1e-4 h): a zero one skips its face lookup and contact tests.
    if(value[axis]!=0.0){
    let face=umPositiveFaceAtAnchor(owner,axis,ownedFace.anchor);
    ${this.solid&&!this.coarse?`if(owner.width==1u){
     // Native embedded/wall contact: the solved active set where open and
     // closed cells meet, on the separating side's pressure.
     let p=vec3i(umOrigin(owner));var q=p;q[axis]+=1;let own=umCellOpen(p)>1e-5;
     if(own!=(umCellOpen(q)>1e-5)&&umProjectV(owner,face)>1e-6){
      var solidPressure=0.0;
      if(q[axis]>=i32(UM_D[axis])){solidPressure=select(0.0,pressures[umBoundaryIndex(owner,axis,1)],umPressureLiquid(owner)&&!(axis==1u&&params.policy.y>0.5));}
      else if(own){solidPressure=umSolidPressure(umOwnerAt(q));}else{solidPressure=umSolidPressure(owner);}
      if(solidPressure<=0.0&&select(1.0,-1.0,own)*value[axis]*params.hDt.w>1e-4*UM_H[axis]){released|=1u<<axis;}
     }
    }else `:""}if(face.width!=0u&&umRelease(owner,face,value[axis])){released|=1u<<axis;}
    }
    if(umOrigin(owner)[axis]==0u){let low=umFace(owner,axis,-1,0u);
     if(umRelease(owner,low,umProjectWithSource(owner,low))){released|=1u<<(axis+3u);}}
   }
   value.w=f32(released);`,"umProjectOwner")}
`;
  const compile=async(code:string,entryPoint:string,resources:GPUBindGroupLayout)=>{
   const module=this.device.createShaderModule({code:uniformMixedCountedEntriesWGSL(code,[entryPoint])});const info=await module.getCompilationInfo();
   const errors=info.messages.filter(m=>m.type==="error");if(errors.length)throw new Error(errors.map(m=>`${m.lineNum}: ${m.message}`).join("\n"));
   const layout=this.device.createPipelineLayout({bindGroupLayouts:[ownership.bindLayout,resources,...(this.solid?[this.coarse?this.solid.coarse!.bindLayout:this.solid.bindLayout]:[])]});
   return uniformMixedSolidPipeline(this.solid,s=>this.device.createComputePipelineAsync({layout,compute:{module,entryPoint,constants:{umDispatchX:ownership.dispatchX,umCountedJobs:UNIFORM_MIXED_COUNTED.residentAll,...s}}}));
  };
  this.rhsPipeline=await compile(rhsSource,"buildRhs",this.rhsLayout);this.projectPipeline=await compile(projectSource,"project",this.projectLayout);
 }
 encode(encoder:GPUCommandEncoder,entry:"rhs"|"project",group:GPUBindGroup):void{
  const pipeline=entry==="rhs"?this.rhsPipeline:this.projectPipeline;if(!pipeline)throw new Error("Mixed pressure velocity stage is not initialized");
  const pass=encoder.beginComputePass({label:`Uniform mixed pressure ${entry}`});pass.setBindGroup(0,this.ownership.bindGroup);pass.setBindGroup(1,group);if(this.solid)pass.setBindGroup(2,this.coarse?this.solid.coarse!.bindGroup:this.solid.bindGroup);this.ownership.dispatchCounted(pass,this.solid?.select(pipeline)??pipeline);pass.end();
 }
}
