import {uniformMixedFaceAddressWGSL} from "./uniform-mixed-face-dispatch.wgsl";
import type { UniformMixedOwnership } from "./uniform-mixed-ownership";
import { uniformMixedTopologyWGSL } from "./uniform-mixed-topology.wgsl";
import { uniformMixedVertexSamplingWGSL } from "./uniform-mixed-vertex-sampling.wgsl";
import { uniformSharpenBudgetWGSL } from "./uniform-sharpen-budget.wgsl";
import { uniformMixedSolidWGSL, type UniformMixedSolid } from "./uniform-mixed-solid.wgsl";

/** Native geometric prepare/propose/limit/commit sweeps on physical mixed
 * face patches. Budgets and fluxes use fine-cell mass units, with area shares
 * splitting a coarse cell's offer across its subfaces. Borrows <=40N bytes
 * from the 40N-byte native transport edge slice after transport completes.
 * With static solids a unit owner is admitted only when fully open, and a
 * unit-unit face only when both cells and its aperture are fully open, the
 * native uvOpen/faceOpenFraction >= 0.99999 gates.
 *
 * With a work list, the sweeps visit only tiles holding an owner whose budget
 * can be nonzero (admission reads centre phi, fixed across the sweeps) or
 * whose unchanged volume the commit would clear as dust. Every other owner
 * keeps its volume exactly and has zero budgets, so its faces carry a zero
 * flux: listed owners skip them instead of reading unlisted budgets. Unlisted
 * owners are not rewritten, so the scratch volume holds stale values there
 * after the sweeps; the final (even-sweep) volume is exact. */
export class UniformMixedSharpening {
  readonly allocatedBytes=0;
  /** Bytes of the work list: three indirect dispatches, three tier counts,
   * a flag per tile and one tier-partitioned list of tiles. */
  static workBytes(tiles:number):number{return 4*(12+2*tiles);}
  private readonly resources:GPUBindGroupLayout;
  private readonly pipelines=new Map<string,GPUComputePipeline[]>();
  /** work: a STORAGE|COPY_SRC|COPY_DST buffer of workBytes(tiles) and a
   * separate 36-byte INDIRECT|COPY_DST buffer (Dawn rejects indirect and
   * writable storage use of one buffer within a pass). */
  constructor(private readonly device:GPUDevice,readonly ownership:UniformMixedOwnership,private readonly solid?:UniformMixedSolid,private readonly work?:{list:GPUBuffer;indirect:GPUBuffer}){
    if(work&&(work.list.size<UniformMixedSharpening.workBytes(ownership.layout.tiles.length)||work.indirect.size<36))throw new Error("Mixed sharpening work list is too small");
    this.resources=device.createBindGroupLayout({entries:[
      ...[0,1,2,3].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float" as const,viewDimension:"3d" as const}})),
      {binding:4,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"write-only",format:"r32float",viewDimension:"3d"}},
      {binding:5,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},
      {binding:6,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},
      {binding:7,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},
      ...(work?[{binding:8,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage" as const}}]:[]),
    ]});
  }
  bind(input:GPUTexture,output:GPUTexture,phi:GPUTexture,targetFill:GPUTexture,centerPhi:GPUTexture,scratch:GPUBufferBinding,params:GPUBuffer,reductions:GPUBuffer):GPUBindGroup{
    const d=this.ownership.layout.lattice.dimensions,n=d[0]*d[1]*d[2];
    for(const [i,t] of [input,targetFill,centerPhi,phi,output].entries())
      if(t.format!=="r32float"||[t.width,t.height,t.depthOrArrayLayers].some((v,a)=>v!==d[a]!+(i===3?1:0)))throw new Error("Mixed sharpening requires native cell and vertex fields");
    if(input===output||output===targetFill||output===centerPhi)throw new Error("Mixed sharpening output must be disjoint");
    if((scratch.size??scratch.buffer.size-(scratch.offset??0))<n*16+this.ownership.layout.cellCount*24)throw new Error("Mixed sharpening scratch is too small");
    return this.device.createBindGroup({layout:this.resources,entries:[
      ...[input,targetFill,centerPhi,phi].map((t,binding)=>({binding,resource:t.createView()})),
      {binding:4,resource:output.createView()},{binding:5,resource:scratch},{binding:6,resource:{buffer:params,size:32}},
      {binding:7,resource:{buffer:reductions}},
      ...(this.work?[{binding:8,resource:{buffer:this.work.list}}]:[]),
    ]});
  }
  async initialize():Promise<void>{
    const h=this.ownership.layout.lattice.cellSize_m,tiles=this.ownership.layout.tiles.length,n=tiles*64,listed=!!this.work;
    const module=this.device.createShaderModule({code:uniformMixedTopologyWGSL(this.ownership.layout,0)+/* wgsl */`
@group(1) @binding(0) var volume:texture_3d<f32>;
@group(1) @binding(1) var targetFill:texture_3d<f32>;
@group(1) @binding(2) var centerPhi:texture_3d<f32>;
@group(1) @binding(3) var phi:texture_3d<f32>;
@group(1) @binding(4) var output:texture_storage_3d<r32float,write>;
@group(1) @binding(5) var<storage,read_write> scratch:array<f32>;
struct UMSharpenParams {tuning:vec4f,policy:vec4f}
@group(1) @binding(6) var<uniform> sharpen:UMSharpenParams;
@group(1) @binding(7) var<storage,read_write> reductions:array<atomic<u32>>;
const UM_MIN_H:f32=${Math.min(...h)};const UM_MAX_H:f32=${Math.max(...h)};
fn umLoadVertex(p:vec3u)->f32{return textureLoad(phi,vec3i(p),0).x;}
${uniformMixedVertexSamplingWGSL}
${uniformMixedFaceAddressWGSL}
${uniformSharpenBudgetWGSL}
${uniformMixedSolidWGSL(this.solid?2:undefined)}
fn umSharpenOpen(o:UMOwner)->bool{return o.width!=1u||!umSolidEnabled()||umCellOpen(vec3i(umOrigin(o)))>0.99999;}
fn umSharpenFaceOpen(a:UMOwner,f:UMFace)->bool{
 if(!umSolidEnabled()||a.width!=1u||f.neighbor.width!=1u){return true;}
 return umSharpenOpen(a)&&umSharpenOpen(f.neighbor)&&umFaceOpen(f.anchor,f.axis)>0.99999;
}
${listed?`@group(1) @binding(8) var<storage,read_write> work:array<atomic<u32>>;
const SH_FLAGS:u32=12u;const SH_LIST:u32=${12+tiles}u;
fn shTier(width:u32)->u32{return select(select(2u,1u,width==2u),0u,width==1u);}
fn shListStart(tier:u32)->u32{return SH_LIST+select(0u,umCounts.x,tier>0u)+select(0u,umCounts.y,tier>1u);}
fn shListed(o:UMOwner)->bool{return o.width==0u||atomicLoad(&work[SH_FLAGS+o.tile])!=0u;}
// Owner of a listed tile of this pipeline's tier.
fn shOwner(gid:vec3u)->UMOwner {
 let slot=gid.x+umDispatchX*64u*gid.y;let per=64u/(umCellWidth*umCellWidth*umCellWidth);let tier=shTier(umCellWidth);
 let job=slot/per;if(job>=atomicLoad(&work[9u+tier])){return UMOwner();}
 let tile=atomicLoad(&work[shListStart(tier)+job]);let lane=slot%per;
 return UMOwner(tile,lane,umCellWidth,(umTopology[tile]&0x3fffffffu)+lane);
}`:`fn shListed(o:UMOwner)->bool{return true;}
fn shOwner(gid:vec3u)->UMOwner{return umOwner(gid);}`}
fn umMassScale(o:UMOwner)->f32{return f32(o.width*o.width*o.width);}
fn umV(o:UMOwner)->f32{return textureLoad(volume,vec3i(umOrigin(o)),0).x;}
fn umBudgetAt(o:UMOwner)->u32{return ${3*n}u+6u*o.index;}
fn umRawAt(f:UMFace)->u32{return 3u*(u32(f.anchor.x)+UM_D.x*(u32(f.anchor.y)+UM_D.y*u32(f.anchor.z)))+f.axis;}
fn umCacheAt(anchor:vec3i)->u32{return ${3*n}u+6u*(umCounts.x*64u+umCounts.y*8u+umCounts.z)+u32(anchor.x)+UM_D.x*(u32(anchor.y)+UM_D.y*u32(anchor.z));}
fn umFaceFlags(a:UMOwner,f:UMFace)->u32 {
 if(f.neighbor.width==0u||!umSharpenFaceOpen(a,f)){return 0u;}
 let pa=vec3i(umOrigin(a));let pb=vec3i(umOrigin(f.neighbor));
 let phiA=textureLoad(centerPhi,pa,0).x;let phiB=textureLoad(centerPhi,pb,0).x;
 if(sharpen.policy.x==0.0&&sharpen.policy.y<1.5
  &&abs(phiA)>=sharpen.tuning.y*UM_MIN_H*f32(a.width)
  &&abs(phiB)>=sharpen.tuning.y*UM_MIN_H*f32(f.neighbor.width)){return 0u;}
 let middle=umSampleVertex(umFaceCenter(f));let epsilon=1e-6;
 let inwardA=phiA>=0.0&&phiB<phiA-epsilon&&middle<=phiA+epsilon&&middle>=phiB-epsilon;
 let inwardB=phiB>=0.0&&phiA<phiB-epsilon&&middle<=phiB+epsilon&&middle>=phiA-epsilon;
 let relayA=phiA>0.0&&textureLoad(targetFill,pa,0).x<=1e-6;
 let relayB=phiB>0.0&&textureLoad(targetFill,pb,0).x<=1e-6;
 return select(0u,1u,(middle<=epsilon&&!relayB)||inwardA)|select(0u,2u,(middle<=epsilon&&!relayA)||inwardB);
}
// Geometry is immutable through all eight volume sweeps. Two admission bits
// per component share one scratch word per anchor; no face cache allocation.
@compute @workgroup_size(64) fn cacheGeometry(@builtin(global_invocation_id) gid:vec3u){
 let o=umAllOwner(gid);if(o.width==0u){return;}
 for(var axis=0u;axis<3u;axis++){
  let first=umFace(o,axis,1,0u);
  for(var part=0u;part<first.count;part++){
   let f=umFace(o,axis,1,part);var earlier=false;
   for(var other=0u;other<axis;other++){earlier=earlier||umPositiveFaceAtAnchor(o,other,f.anchor).width!=0u;}
   if(earlier){continue;}var flags=0u;
   for(var other=0u;other<3u;other++){
    let face=umPositiveFaceAtAnchor(o,other,f.anchor);
    if(face.width!=0u){flags|=umFaceFlags(o,face)<<(2u*other);}
   }
   scratch[umCacheAt(f.anchor)]=bitcast<f32>(flags);
  }
 }
}
${listed?`// A tile is listed when an owner can hold a nonzero budget (the admission
// of uvSharpenBudgets) or would be cleared as dust by an unchanged commit.
@compute @workgroup_size(64) fn classify(@builtin(global_invocation_id) id:vec3u){
 let o=umAllOwner(id);if(o.width==0u){return;}
 let phi=textureLoad(centerPhi,vec3i(umOrigin(o)),0).x;let h=UM_MIN_H*f32(o.width);let band=sharpen.tuning.y;let open=umSharpenOpen(o);
 let admitted=open&&select((abs(phi)<band*h),(phi<band*h),(sharpen.policy.x>0.5));
 let orphan=sharpen.policy.y>1.5&&open&&phi>=band*h;
 let value=umV(o);let dust=value!=0.0&&abs(value)<sharpen.tuning.z&&!(value>0.0&&phi<4.0*UM_MAX_H*f32(o.width));
 if(admitted||orphan||dust){atomicStore(&work[SH_FLAGS+o.tile],1u);}
}
@compute @workgroup_size(64) fn compact(@builtin(global_invocation_id) id:vec3u){
 let tile=id.x+umDispatchX*64u*id.y;if(tile>=UM_TILES||atomicLoad(&work[SH_FLAGS+tile])==0u){return;}
 let tier=shTier(umTileWidth(tile));let slot=atomicAdd(&work[9u+tier],1u);atomicStore(&work[shListStart(tier)+slot],tile);
}
@compute @workgroup_size(1) fn publish(){
 for(var tier=0u;tier<3u;tier++){
  let groups=(atomicLoad(&work[9u+tier])*(64u>>(3u*tier))+63u)/64u;
  atomicStore(&work[3u*tier],min(groups,umDispatchX));atomicStore(&work[3u*tier+1u],(groups+umDispatchX-1u)/umDispatchX);atomicStore(&work[3u*tier+2u],1u);
 }
}
`:""}@compute @workgroup_size(64) fn prepare(@builtin(global_invocation_id) gid:vec3u){
 let o=umAllOwner(gid);if(o.width==0u){return;}let p=vec3i(umOrigin(o));let at=umBudgetAt(o);
 let distance=textureLoad(centerPhi,p,0).x;let desired=textureLoad(targetFill,p,0).x;
 let budget=uvSharpenBudgets(umV(o),desired,distance,UM_MIN_H*f32(o.width),clamp(sharpen.tuning.x,0.0,1.0),sharpen.tuning.y,sharpen.policy.x>0.5,sharpen.policy.y,umSharpenOpen(o))*umMassScale(o);
 scratch[at]=budget.x;scratch[at+1u]=budget.y;scratch[at+2u]=distance;scratch[at+3u]=desired;
}
fn umProposal(a:UMOwner,f:UMFace)->f32 {
 let b=f.neighbor;let i=umBudgetAt(a);let j=umBudgetAt(b);
 if(!umSharpenFaceOpen(a,f)){return 0.0;}
 let phiA=scratch[i+2u];let phiB=scratch[j+2u];let dose=clamp(sharpen.tuning.x,0.0,1.0);
 let area=f32(f.width*f.width);let shareA=area/f32(a.width*a.width);let shareB=area/f32(b.width*b.width);
 let giveA=scratch[i]*shareA;let giveB=scratch[j]*shareB;let takeA=scratch[i+1u]*shareA;let takeB=scratch[j+1u]*shareB;
 if(sharpen.policy.y>1.5){
  let orphanA=phiA>=sharpen.tuning.y*UM_MIN_H*f32(a.width);let orphanB=phiB>=sharpen.tuning.y*UM_MIN_H*f32(b.width);
  if(orphanA||orphanB){let va=umV(a);let vb=umV(b);return select(0.0,min(giveA,takeB),orphanA&&vb>va)-select(0.0,min(giveB,takeA),orphanB&&va>vb);}
 }
 // A zero transfer budget cannot contribute in either direction.
 if((giveA==0.0||takeB==0.0)&&(giveB==0.0||takeA==0.0)){return 0.0;}
 let flags=bitcast<u32>(scratch[umCacheAt(f.anchor)])>>(2u*f.axis);let epsilon=1e-6;
 var capA=giveA;var capB=giveB;
 if(sharpen.policy.x>0.5){
  if(!(phiA<0.0&&phiB<phiA-epsilon)){capA=min(capA,dose*max(umV(a)-scratch[i+3u],0.0)*umMassScale(a)*shareA);}
  if(!(phiB<0.0&&phiA<phiB-epsilon)){capB=min(capB,dose*max(umV(b)-scratch[j+3u],0.0)*umMassScale(b)*shareB);}
 }
 return select(0.0,min(capA,takeB),(flags&1u)!=0u)-select(0.0,min(capB,takeA),(flags&2u)!=0u);
}
@compute @workgroup_size(64) fn propose(@builtin(global_invocation_id) gid:vec3u){
 let o=umAllOwner(gid);if(o.width==0u){return;}
 for(var axis=0u;axis<3u;axis++){let first=umFace(o,axis,1,0u);if(first.neighbor.width==0u){continue;}
  for(var part=0u;part<first.count;part++){let face=umFace(o,axis,1,part);if(shListed(face.neighbor)){scratch[umRawAt(face)]=umProposal(o,face);}}
 }
}
@compute @workgroup_size(64) fn limit(@builtin(global_invocation_id) gid:vec3u){
 let o=umAllOwner(gid);if(o.width==0u){return;}var outgoing=0.0;var incoming=0.0;
 let budget=umBudgetAt(o);
 if(scratch[budget]==0.0&&scratch[budget+1u]==0.0){scratch[budget+4u]=1.0;scratch[budget+5u]=1.0;return;}
 for(var axis=0u;axis<3u;axis++){for(var side=0u;side<2u;side++){
  let sign=select(1,-1,side==1u);let first=umFace(o,axis,sign,0u);if(first.neighbor.width==0u){continue;}
  for(var part=0u;part<first.count;part++){let face=umFace(o,axis,sign,part);if(!shListed(face.neighbor)){continue;}let value=f32(sign)*scratch[umRawAt(face)];outgoing+=max(value,0.0);incoming+=max(-value,0.0);}
 }}
 let at=umBudgetAt(o);scratch[at+4u]=min(1.0,scratch[at]/max(outgoing,1e-20));scratch[at+5u]=min(1.0,scratch[at+1u]/max(incoming,1e-20));
}
@compute @workgroup_size(64) fn commit(@builtin(global_invocation_id) gid:vec3u){
 let o=umAllOwner(gid);if(o.width==0u){return;}let at=umBudgetAt(o);var terms:array<f32,6>;
 if(scratch[at]!=0.0||scratch[at+1u]!=0.0){
 for(var axis=0u;axis<3u;axis++){for(var side=0u;side<2u;side++){
  let sign=select(1,-1,side==1u);let first=umFace(o,axis,sign,0u);var sum=0.0;
  if(first.neighbor.width!=0u){for(var part=0u;part<first.count;part++){
   let face=umFace(o,axis,sign,part);if(!shListed(face.neighbor)){continue;}let other=umBudgetAt(face.neighbor);let raw=f32(sign)*scratch[umRawAt(face)];
   let factor=select(min(scratch[at+5u],scratch[other+4u]),min(scratch[at+4u],scratch[other+5u]),raw>=0.0);sum-=raw*factor;
  }}terms[2u*axis+side]=sum;
 }}}
 let delta=((terms[0]+terms[1])+(terms[4]+terms[5]))+(terms[2]+terms[3]);var value=umV(o)+delta/umMassScale(o);
 if(value!=0.0&&abs(value)<sharpen.tuning.z && !(value>0.0&&scratch[at+2u]<4.0*UM_MAX_H*f32(o.width))){
  atomicAdd(&reductions[5],1u);atomicAdd(&reductions[6],min(u32(abs(value)/sharpen.tuning.z*64.0),64u)*o.width*o.width*o.width);value=0.0;
 }
 textureStore(output,vec3i(umOrigin(o)),vec4f(value));
}`.replaceAll('umAllOwner(gid)','shOwner(gid)')});
    const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");if(errors.length)throw new Error(errors.map(m=>`${m.lineNum}: ${m.message}`).join("\n"));
    const layout=this.device.createPipelineLayout({bindGroupLayouts:[this.ownership.bindLayout,this.resources,...(this.solid?[this.solid.bindLayout]:[])]});
    for(const entryPoint of ["cacheGeometry","prepare","propose","limit","commit"])this.pipelines.set(entryPoint,await Promise.all([1,2,4].map(umCellWidth=>this.device.createComputePipelineAsync({layout,compute:{module,entryPoint,constants:{umCellWidth,umDispatchX:this.ownership.dispatchX}}}))));
    if(listed)for(const entryPoint of ["classify","compact","publish"])this.pipelines.set(entryPoint,[await this.device.createComputePipelineAsync({layout,compute:{module,entryPoint,constants:{umDispatchX:this.ownership.dispatchX}}})]);
  }
  encodeGeometry(encoder:GPUCommandEncoder,group:GPUBindGroup):void{
    if(this.pipelines.size!==(this.work?8:5))throw new Error("Mixed sharpening is not initialized");
    const begin=()=>{const pass=encoder.beginComputePass({label:"Uniform mixed sharpening geometry"});pass.setBindGroup(0,this.ownership.bindGroup);pass.setBindGroup(1,group);if(this.solid)pass.setBindGroup(2,this.solid.bindGroup);return pass;};
    if(this.work){
      const tiles=this.ownership.layout.tiles.length,groups=Math.ceil(tiles/64),dx=this.ownership.dispatchX;
      encoder.clearBuffer(this.work.list,0,4*(12+tiles));
      const list=begin();this.ownership.dispatchAll(list,this.pipelines.get("classify")![0]!);
      list.setPipeline(this.pipelines.get("compact")![0]!);list.dispatchWorkgroups(Math.min(groups,dx),Math.ceil(groups/dx));
      list.setPipeline(this.pipelines.get("publish")![0]!);list.dispatchWorkgroups(1);list.end();
      encoder.copyBufferToBuffer(this.work.list,0,this.work.indirect,0,36);
    }
    const pass=begin();this.ownership.dispatch(pass,this.pipelines.get("cacheGeometry")!,this.work?.indirect);pass.end();
  }
  encodeSweep(encoder:GPUCommandEncoder,group:GPUBindGroup,refreshGeometry=true):void{
    if(refreshGeometry)this.encodeGeometry(encoder,group);
    if(this.pipelines.size!==(this.work?8:5))throw new Error("Mixed sharpening is not initialized");
    const pass=encoder.beginComputePass({label:"Uniform mixed geometric sharpening"});pass.setBindGroup(0,this.ownership.bindGroup);pass.setBindGroup(1,group);if(this.solid)pass.setBindGroup(2,this.solid.bindGroup);
    for(const entry of ["prepare","propose","limit","commit"])this.ownership.dispatch(pass,this.pipelines.get(entry)!,this.work?.indirect);pass.end();
  }
}
