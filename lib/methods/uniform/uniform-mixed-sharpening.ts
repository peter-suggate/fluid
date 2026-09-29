import {uniformMixedDustAccountingWGSL} from "./uniform-mixed-dust-accounting.wgsl";
import {uniformMixedFaceAddressWGSL} from "./uniform-mixed-face-dispatch.wgsl";
import type { UniformMixedOwnership } from "./uniform-mixed-ownership";
import { UNIFORM_MIXED_COUNTED, uniformMixedCountedEntriesWGSL, uniformMixedTopologyWGSL } from "./uniform-mixed-topology.wgsl";
import { uniformMixedVertexSamplingSource } from "./uniform-mixed-vertex-sampling.wgsl";
import { uniformSharpenBudgetWGSL } from "./uniform-sharpen-budget.wgsl";
import { uniformMixedSolidPipeline, uniformMixedSolidWGSL, type UniformMixedSolid } from "./uniform-mixed-solid.wgsl";

/** Native geometric propose/limit/commit sweeps on physical mixed
 * face patches. Budgets and fluxes use fine-cell mass units, with area shares
 * splitting a coarse cell's offer across its subfaces. Borrows <=40N bytes
 * from the 40N-byte native transport edge slice after transport completes.
 * With static solids a unit owner is admitted only when fully open, and a
 * unit-unit face only when both cells and its aperture are fully open, the
 * native uvOpen/faceOpenFraction >= 0.99999 gates.
 *
 * The sweeps visit only tiles holding an owner whose budget
 * can be nonzero (admission reads centre phi, fixed across the sweeps) or
 * whose unchanged volume the commit would clear as dust. Every other owner
 * keeps its volume exactly and has zero budgets, so its faces carry a zero
 * flux: listed owners skip them instead of reading unlisted budgets. Unlisted
 * owners are not rewritten, so the scratch volume holds stale values there
 * after the sweeps; the final (even-sweep) volume is exact.
 *
 * Each sweep is three launches: propose, limit, commit. Commit prepares the
 * next sweep's budgets in place from the owner's new volume, the only budget
 * input that changes across sweeps (commit reads only its own budget and its
 * neighbours' limit factors, so the overwrite is race-free). The first
 * sweep's budgets are prepared with the geometry cache. */
/** Workgroups of 192 lanes per merged sharpening launch at most. */
const SHARPEN_GRID=1024;
export class UniformMixedSharpening {
  readonly allocatedBytes=0;
  private readonly grid:number;
  /** Bytes of the work list: three unused words, four list words (three
   * list counts and one unused), one unused word, a flag per tile, one tier-partitioned
   * list of regular tiles and one list of coarse seam tiles (4h tiles with a
   * mixed 3x3x3 stencil). */
  static workBytes(tiles:number):number{return 4*(8+3*tiles);}
  private readonly resources:GPUBindGroupLayout;
  private readonly pipelines=new Map<string,GPUComputePipeline[]>();
  /** work: a STORAGE|COPY_DST buffer of workBytes(tiles). resolved: phi's
   * hanging texels hold umVertexValue (UniformMixedPhiResolve). */
  constructor(private readonly device:GPUDevice,readonly ownership:UniformMixedOwnership,private readonly solid:UniformMixedSolid|undefined,private readonly work:{list:GPUBuffer},private readonly resolved=false){
    const tiles=ownership.capacity.tiles;
    if(work.list.size<UniformMixedSharpening.workBytes(tiles))throw new Error("Mixed sharpening work list is too small");
    // Each merged launch is a fixed grid-stride grid: the capacity's job bound
    // (every tile listed), capped where the GPU is saturated.
    this.grid=Math.max(1,Math.min(SHARPEN_GRID,Math.ceil(64*tiles/192)+tiles));
    this.resources=device.createBindGroupLayout({entries:[
      ...[0,1,2,3].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float" as const,viewDimension:"3d" as const}})),
      {binding:4,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"write-only",format:"r32float",viewDimension:"3d"}},
      {binding:5,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},
      {binding:6,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},
      {binding:7,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},
      {binding:8,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},
    ]});
  }
  bind(input:GPUTexture,output:GPUTexture,phi:GPUTexture,targetFill:GPUTexture,centerPhi:GPUTexture,scratch:GPUBufferBinding,params:GPUBuffer,reductions:GPUBuffer):GPUBindGroup{
    const d=this.ownership.capacity.lattice.dimensions,n=d[0]*d[1]*d[2];
    for(const [i,t] of [input,targetFill,centerPhi,phi,output].entries())
      if(t.format!=="r32float"||[t.width,t.height,t.depthOrArrayLayers].some((v,a)=>v!==d[a]!+(i===3?1:0)))throw new Error("Mixed sharpening requires native cell and vertex fields");
    if(input===output||output===targetFill||output===centerPhi)throw new Error("Mixed sharpening output must be disjoint");
    // Budgets and the face cache index live owners, at most n of them.
    if((scratch.size??scratch.buffer.size-(scratch.offset??0))<n*40)throw new Error("Mixed sharpening scratch is too small");
    return this.device.createBindGroup({layout:this.resources,entries:[
      ...[input,targetFill,centerPhi,phi].map((t,binding)=>({binding,resource:t.createView()})),
      {binding:4,resource:output.createView()},{binding:5,resource:scratch},{binding:6,resource:{buffer:params,size:32}},
      {binding:7,resource:{buffer:reductions}},
      {binding:8,resource:{buffer:this.work.list}},
    ]});
  }
  async initialize():Promise<void>{
    const h=this.ownership.capacity.lattice.cellSize_m,tiles=this.ownership.capacity.tiles,n=tiles*64;
    // classify: a GPU-counted fixed grid over every live owner (all).
    const module=this.device.createShaderModule({code:uniformMixedCountedEntriesWGSL(uniformMixedTopologyWGSL(this.ownership.capacity,0)+/* wgsl */`
@group(1) @binding(0) var volume:texture_3d<f32>;
@group(1) @binding(1) var targetFill:texture_3d<f32>;
@group(1) @binding(2) var centerPhi:texture_3d<f32>;
@group(1) @binding(3) var phi:texture_3d<f32>;
@group(1) @binding(4) var output:texture_storage_3d<r32float,write>;
@group(1) @binding(5) var<storage,read_write> scratch:array<f32>;
struct UMSharpenParams {tuning:vec4f,policy:vec4f}
@group(1) @binding(6) var<uniform> sharpen:UMSharpenParams;
@group(1) @binding(7) var<storage,read_write> reductions:array<atomic<u32>>;
${uniformMixedDustAccountingWGSL(this.ownership.capacity.lattice.dimensions.reduce((n,d)=>n*d,1))}
const UM_MIN_H:f32=${Math.min(...h)};const UM_MAX_H:f32=${Math.max(...h)};
fn umLoadVertex(p:vec3u)->f32{return textureLoad(phi,vec3i(p),0).x;}
${uniformMixedVertexSamplingSource("",this.resolved)}
${uniformMixedFaceAddressWGSL}
${uniformSharpenBudgetWGSL}
${uniformMixedSolidWGSL(this.solid?2:undefined)}
fn umSharpenOpen(o:UMOwner)->bool{return o.width!=1u||!umSolidEnabled()||umCellOpen(vec3i(umOrigin(o)))>0.99999;}
fn umSharpenFaceOpenAB(a:UMOwner,b:UMOwner,f:UMFace)->bool{
 if(!umSolidEnabled()||a.width!=1u||b.width!=1u){return true;}
 return umSharpenOpen(a)&&umSharpenOpen(b)&&umFaceOpen(f.anchor,f.axis)>0.99999;
}
fn umSharpenFaceOpen(a:UMOwner,f:UMFace)->bool{return umSharpenFaceOpenAB(a,f.neighbor,f);}
@group(1) @binding(8) var<storage,read_write> work:array<atomic<u32>>;
// Lists 0 (h) and 1 (regular 4h) are tiers; tier t owners have width 4^t.
const SH_COUNTS:u32=3u;const SH_FLAGS:u32=8u;const SH_LIST:u32=${8+tiles}u;
fn shTier(width:u32)->u32{return select(1u,0u,width==1u);}
// List 2: coarse seam tiles. Their owners meet finer patches, so one lane
// per owner would walk up to sixteen patches per face serially.
fn shListStart(tier:u32)->u32{if(tier==2u){return SH_LIST+UM_TILES;}return SH_LIST+select(0u,umCounts.x,tier>0u);}
fn shListed(o:UMOwner)->bool{return o.width==0u||atomicLoad(&work[SH_FLAGS+o.tile])!=0u;}
// Listed tiers hold h owners and uniform-stencil 4h owners: one patch per
// face. A constant part count lets the six face chains issue together; a
// data-dependent loop serialized them (cold misses at coarse owners).
fn shParts(first:UMFace)->u32{return 1u;}
fn umMassScale(o:UMOwner)->f32{return f32(o.width*o.width*o.width);}
fn umV(o:UMOwner)->f32{return textureLoad(volume,vec3i(umOrigin(o)),0).x;}
// Owner record: give, take, distance, desired, outgoing and incoming
// limit factors.
fn umBudgetAt(o:UMOwner)->u32{return ${3*n}u+6u*o.index;}
fn umRawAt(f:UMFace)->u32{return 3u*(u32(f.anchor.x)+UM_D.x*(u32(f.anchor.y)+UM_D.y*u32(f.anchor.z)))+f.axis;}
fn umCacheAt(anchor:vec3i)->u32{return ${3*n}u+6u*(umCounts.x*64u+umCounts.y)+u32(anchor.x)+UM_D.x*(u32(anchor.y)+UM_D.y*u32(anchor.z));}
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
fn shCacheGeometry(o:UMOwner){
if(o.width==0u){return;}
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
// A tile is listed when an owner can hold a nonzero budget (the admission
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
 let width=umTileWidth(tile);let tier=select(shTier(width),2u,width!=1u&&umTileMaximumWidth(tile)!=umTileMinimumWidth(tile));
 let slot=atomicAdd(&work[SH_COUNTS+tier],1u);atomicStore(&work[shListStart(tier)+slot],tile);
}
// One merged launch per sweep entry: 192 regular owners per job (the tier
// lists in order), then one job per coarse seam tile. Serial tier launches
// paid each tier's latency in every entry of every sweep. (regular, all).
fn shJobs()->vec2u{
 var owners=0u;for(var tier=0u;tier<2u;tier++){owners+=atomicLoad(&work[SH_COUNTS+tier])*(64u>>(6u*tier));}
 let regular=(owners+191u)/192u;return vec2u(regular,regular+atomicLoad(&work[SH_COUNTS+2u]));
}
var<workgroup> shJobCount:vec2u;
fn shMergedOwner(job:u32,lane:u32)->UMOwner{
 var slot=job*192u+lane;
 for(var tier=0u;tier<2u;tier++){
  let per=64u>>(6u*tier);let count=atomicLoad(&work[SH_COUNTS+tier])*per;
  if(slot<count){let tile=atomicLoad(&work[shListStart(tier)+slot/per]);let l=slot%per;return UMOwner(tile,l,1u<<(2u*tier),(umTopology[tile]&0x3fffffffu)+l);}
  slot-=count;
 }
 return UMOwner();
}
// Coarse seam tiles, one 192-lane group each. Face-parallel entries give a
// lane to each (anchor, component) of the tile; owner reductions give each
// of the tile's (4/w)^3 owners 3w^3 lanes, one per (face side, patch), and
// sum the staged terms in one lane in the per-owner kernels' order.
fn shSeamTile(job:u32)->u32{
 if(job>=atomicLoad(&work[SH_COUNTS+2u])){return UM_TILES;}
 return atomicLoad(&work[shListStart(2u)+job]);
}
fn shSeamOwner(tile:u32,local:vec3u)->UMOwner{
 if(tile>=UM_TILES){return UMOwner();}
 let width=umTileWidth(tile);let q=local/width;let side=4u/width;let lane=q.x+side*(q.y+side*q.z);
 return UMOwner(tile,lane,width,(umTopology[tile]&0x3fffffffu)+lane);
}
struct SHSeamLane {owner:UMOwner,first:bool,side:u32,part:u32}
fn shSeamLane(tile:u32,lane:u32)->SHSeamLane{
 if(tile>=UM_TILES){return SHSeamLane(UMOwner(),false,0u,0u);}
 let width=umTileWidth(tile);let per=3u*width*width*width;let side=4u/width;let o=lane/per;let r=lane%per;
 if(o>=side*side*side){return SHSeamLane(UMOwner(),false,0u,0u);}
 let owner=shSeamOwner(tile,umCorner(o,side)*width);return SHSeamLane(owner,r==0u,r/(width*width),r%(width*width));
}
var<workgroup> shSeamTerms:array<vec2f,192>;
fn shCacheGeometrySeam(tile:u32,lane:u32){
 let cell=lane%64u;let axis=lane/64u;let local=umCorner(cell,4u);let o=shSeamOwner(tile,local);
 var term=vec2f(0);var anchor=vec3i(0);
 if(o.width!=0u){anchor=vec3i(umTileCoord(tile)*4u+local);let face=umPositiveFaceAtAnchor(o,axis,anchor);if(face.width!=0u){term=vec2f(bitcast<f32>(umFaceFlags(o,face)<<(2u*axis)),1);}}
 shSeamTerms[lane]=term;workgroupBarrier();
 if(lane<64u){
  let x=shSeamTerms[cell];let y=shSeamTerms[cell+64u];let z=shSeamTerms[cell+128u];
  if(x.y+y.y+z.y>0.0){scratch[umCacheAt(anchor)]=bitcast<f32>(bitcast<u32>(x.x)|bitcast<u32>(y.x)|bitcast<u32>(z.x));}
 }
}
// Budgets from an owner's volume; distance and desired are fixed per frame.
fn shBudgets(o:UMOwner,value:f32,distance:f32,desired:f32)->vec2f{
 return uvSharpenBudgets(value,desired,distance,UM_MIN_H*f32(o.width),clamp(sharpen.tuning.x,0.0,1.0),sharpen.tuning.y,sharpen.policy.x>0.5,sharpen.policy.y,umSharpenOpen(o))*umMassScale(o);
}
fn shProposeSeam(tile:u32,lane:u32){
 let cell=lane%64u;let axis=lane/64u;let local=umCorner(cell,4u);let o=shSeamOwner(tile,local);if(o.width==0u){return;}
 let face=umPositiveFaceAtAnchor(o,axis,vec3i(umTileCoord(tile)*4u+local));
 if(face.width!=0u&&face.neighbor.width!=0u&&shListed(face.neighbor)){scratch[umRawAt(face)]=umProposal(o,face.neighbor,face);}
}
fn shPrepareSeam(tile:u32,lane:u32){
 if(tile>=UM_TILES){return;}let width=umTileWidth(tile);let side=4u/width;if(lane>=side*side*side){return;}
 shPrepare(shSeamOwner(tile,umCorner(lane,side)*width));
}
// The owner's face (axis, sign) patch part, when limit/commit would visit it.
fn shSeamFace(o:UMOwner,side:u32,part:u32)->UMFace{
 let axis=side/2u;let sign=select(1,-1,side%2u==1u);let first=umFace(o,axis,sign,0u);
 if(first.neighbor.width==0u||part>=first.count){return UMFace();}
 let face=umFace(o,axis,sign,part);if(!shListed(face.neighbor)){return UMFace();}return face;
}
fn shLimitSeam(tile:u32,lane:u32){
 let l=shSeamLane(tile,lane);var term=vec2f(0);
 if(l.owner.width!=0u&&l.side<6u){let face=shSeamFace(l.owner,l.side,l.part);if(face.width!=0u){term=vec2f(f32(face.sign)*scratch[umRawAt(face)],1);}}
 shSeamTerms[lane]=term;workgroupBarrier();
 if(!l.first){return;}
 let o=l.owner;let at=umBudgetAt(o);
 if(scratch[at]==0.0&&scratch[at+1u]==0.0){scratch[at+4u]=1.0;scratch[at+5u]=1.0;return;}
 let parts=o.width*o.width;var outgoing=0.0;var incoming=0.0;
 for(var k=0u;k<6u*parts;k++){let t=shSeamTerms[lane+k];if(t.y>0.0){outgoing+=max(t.x,0.0);incoming+=max(-t.x,0.0);}}
 scratch[at+4u]=min(1.0,scratch[at]/max(outgoing,1e-20));scratch[at+5u]=min(1.0,scratch[at+1u]/max(incoming,1e-20));
}
fn shCommitSeam(tile:u32,lane:u32){
 let l=shSeamLane(tile,lane);var term=vec2f(0);
 if(l.owner.width!=0u&&l.side<6u){
  let o=l.owner;let at=umBudgetAt(o);let face=shSeamFace(o,l.side,l.part);
  if(face.width!=0u&&(scratch[at]!=0.0||scratch[at+1u]!=0.0)){
   let other=umBudgetAt(face.neighbor);let raw=f32(face.sign)*scratch[umRawAt(face)];
   let factor=select(min(scratch[at+5u],scratch[other+4u]),min(scratch[at+4u],scratch[other+5u]),raw>=0.0);term=vec2f(raw*factor,1);
  }
 }
 shSeamTerms[lane]=term;workgroupBarrier();
 if(!l.first){return;}
 let o=l.owner;let at=umBudgetAt(o);let parts=o.width*o.width;var terms:array<f32,6>;
 for(var side=0u;side<6u;side++){var sum=0.0;for(var part=0u;part<parts;part++){let t=shSeamTerms[lane+side*parts+part];if(t.y>0.0){sum-=t.x;}}terms[side]=sum;}
 let delta=((terms[0]+terms[1])+(terms[4]+terms[5]))+(terms[2]+terms[3]);shFinish(o,umV(o)+delta/umMassScale(o));
}
// Store the committed volume (after dust) and prepare the next sweep's
// budgets from it in place.
fn shFinish(o:UMOwner,committed:f32){
 let at=umBudgetAt(o);var value=committed;
 if(value!=0.0&&abs(value)<sharpen.tuning.z && !(value>0.0&&scratch[at+2u]<4.0*UM_MAX_H*f32(o.width))){
  umAccountDust(value,o.width*o.width*o.width,sharpen.tuning.z,5u);value=0.0;
 }
 textureStore(output,vec3i(umOrigin(o)),vec4f(value));
 let budget=shBudgets(o,value,scratch[at+2u],scratch[at+3u]);
 scratch[at]=budget.x;scratch[at+1u]=budget.y;
}
fn shPrepare(o:UMOwner){
if(o.width==0u){return;}let p=vec3i(umOrigin(o));let at=umBudgetAt(o);
 let distance=textureLoad(centerPhi,p,0).x;let desired=textureLoad(targetFill,p,0).x;
 let budget=shBudgets(o,umV(o),distance,desired);
 scratch[at]=budget.x;scratch[at+1u]=budget.y;scratch[at+2u]=distance;scratch[at+3u]=desired;
}
// The proposal from lower owner a to upper owner b across patch f (f's
// anchor, axis and width; both incident owners derive the same ones).
fn umProposal(a:UMOwner,b:UMOwner,f:UMFace)->f32 {
 let i=umBudgetAt(a);let j=umBudgetAt(b); if(!umSharpenFaceOpenAB(a,b,f)){return 0.0;}
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
fn shPropose(o:UMOwner){
if(o.width==0u){return;}
 for(var axis=0u;axis<3u;axis++){let first=umFace(o,axis,1,0u);if(first.neighbor.width==0u){continue;}
  for(var part=0u;part<shParts(first);part++){let face=umFace(o,axis,1,part);if(shListed(face.neighbor)){scratch[umRawAt(face)]=umProposal(o,face.neighbor,face);}}
 }
}
fn shLimit(o:UMOwner){
if(o.width==0u){return;}var outgoing=0.0;var incoming=0.0;
 let at=umBudgetAt(o);
 if(scratch[at]==0.0&&scratch[at+1u]==0.0){scratch[at+4u]=1.0;scratch[at+5u]=1.0;return;}
 for(var axis=0u;axis<3u;axis++){for(var side=0u;side<2u;side++){
  let sign=select(1,-1,side==1u);let first=umFace(o,axis,sign,0u);if(first.neighbor.width==0u){continue;}
  for(var part=0u;part<shParts(first);part++){let face=umFace(o,axis,sign,part);if(!shListed(face.neighbor)){continue;}let value=f32(sign)*scratch[umRawAt(face)];outgoing+=max(value,0.0);incoming+=max(-value,0.0);}
 }}
 scratch[at+4u]=min(1.0,scratch[at]/max(outgoing,1e-20));scratch[at+5u]=min(1.0,scratch[at+1u]/max(incoming,1e-20));
}
fn shCommit(o:UMOwner){
if(o.width==0u){return;}let at=umBudgetAt(o);var terms:array<f32,6>;
 if(scratch[at]!=0.0||scratch[at+1u]!=0.0){
 for(var axis=0u;axis<3u;axis++){for(var side=0u;side<2u;side++){
  let sign=select(1,-1,side==1u);let first=umFace(o,axis,sign,0u);var sum=0.0;
  if(first.neighbor.width!=0u){for(var part=0u;part<shParts(first);part++){
   let face=umFace(o,axis,sign,part);if(!shListed(face.neighbor)){continue;}let other=umBudgetAt(face.neighbor);let raw=f32(sign)*scratch[umRawAt(face)];
   let factor=select(min(scratch[at+5u],scratch[other+4u]),min(scratch[at+4u],scratch[other+5u]),raw>=0.0);sum-=raw*factor;
  }}terms[2u*axis+side]=sum;
 }}}
 let delta=((terms[0]+terms[1])+(terms[4]+terms[5]))+(terms[2]+terms[3]);shFinish(o,umV(o)+delta/umMassScale(o));
}
// The geometry launch also prepares the first sweep's budgets.
fn shCacheGeometryPrepare(o:UMOwner){shCacheGeometry(o);shPrepare(o);}
fn shCacheGeometryPrepareSeam(tile:u32,lane:u32){shCacheGeometrySeam(tile,lane);shPrepareSeam(tile,lane);}
${["cacheGeometryPrepare","propose","limit","commit"].map(entry=>{const fn=`sh${entry[0]!.toUpperCase()}${entry.slice(1)}`;return /* wgsl */`
@compute @workgroup_size(192) fn ${entry}(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 if(lane==0u){shJobCount=shJobs();}
 let jobs=workgroupUniformLoad(&shJobCount);
 for(var job=group.x;job<jobs.y;job+=groups.x){
  var tile=UM_TILES;if(job>=jobs.x){tile=shSeamTile(job-jobs.x);}
  ${fn}Seam(tile,lane);
  if(job<jobs.x){${fn}(shMergedOwner(job,lane));}
  // The next job's seam stages its terms in the same workgroup array.
  workgroupBarrier();
 }
}`;}).join("")}`,["classify"])});
    const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");if(errors.length)throw new Error(errors.map(m=>`${m.lineNum}: ${m.message}`).join("\n"));
    const layout=this.device.createPipelineLayout({bindGroupLayouts:[this.ownership.bindLayout,this.resources,...(this.solid?[this.solid.bindLayout]:[])]});
    // One merged pipeline per entry: the listed launch covers every width.
    await Promise.all(["cacheGeometryPrepare","propose","limit","commit"].map(async entryPoint=>{this.pipelines.set(entryPoint,[await this.twin(s=>this.device.createComputePipelineAsync({layout,compute:{module,entryPoint,constants:{umCellWidth:1,umDispatchX:this.ownership.dispatchX,...s}}}))]);}));
    await Promise.all(["classify","compact"].map(async entryPoint=>{this.pipelines.set(entryPoint,[await this.twin(s=>this.device.createComputePipelineAsync({layout,compute:{module,entryPoint,constants:{umDispatchX:this.ownership.dispatchX,...(entryPoint==="classify"?{umCountedJobs:UNIFORM_MIXED_COUNTED.all}:{}),...s}}}))]);}));
  }
  encodeGeometry(encoder:GPUCommandEncoder,group:GPUBindGroup):void{
    if(this.pipelines.size!==6)throw new Error("Mixed sharpening is not initialized");
    const begin=()=>{const pass=encoder.beginComputePass({label:"Uniform mixed sharpening geometry"});pass.setBindGroup(0,this.ownership.bindGroup);pass.setBindGroup(1,group);if(this.solid)pass.setBindGroup(2,this.solid.bindGroup);return pass;};
    {
      const tiles=this.ownership.capacity.tiles,groups=Math.ceil(tiles/64),dx=this.ownership.dispatchX;
      encoder.clearBuffer(this.work.list,0,4*(8+tiles));
      const list=begin();this.ownership.dispatchAllCounted(list,this.variant(this.pipelines.get("classify")![0]!));
      list.setPipeline(this.variant(this.pipelines.get("compact")![0]!));list.dispatchWorkgroups(Math.min(groups,dx),Math.ceil(groups/dx));list.end();
    }
    const pass=begin();this.dispatchEntry(pass,"cacheGeometryPrepare");pass.end();
  }
  /** `sweeps` sweeps over the geometry encodeGeometry cached, alternating
   * the ping-pong groups from groups[0], in one pass. */
  encodeSweeps(encoder:GPUCommandEncoder,groups:readonly [GPUBindGroup,GPUBindGroup],sweeps:number):void{
    if(this.pipelines.size!==6)throw new Error("Mixed sharpening is not initialized");
    const pass=encoder.beginComputePass({label:"Uniform mixed geometric sharpening"});pass.setBindGroup(0,this.ownership.bindGroup);if(this.solid)pass.setBindGroup(2,this.solid.bindGroup);
    for(let sweep=0;sweep<sweeps;sweep++){pass.setBindGroup(1,groups[sweep%2]!);for(const entry of ["propose","limit","commit"])this.dispatchEntry(pass,entry);}
    pass.end();
  }
  /** The merged launch: regular owners, then coarse seam tiles. */
  private twin(create:(solid:Record<string,number>)=>Promise<GPUComputePipeline>):Promise<GPUComputePipeline>{return uniformMixedSolidPipeline(this.solid,create);}
  private variant(pipeline:GPUComputePipeline):GPUComputePipeline{return this.solid?.select(pipeline)??pipeline;}
  private dispatchEntry(pass:GPUComputePassEncoder,entry:string):void{
    pass.setPipeline(this.variant(this.pipelines.get(entry)![0]!));pass.dispatchWorkgroups(this.grid);
  }
}
