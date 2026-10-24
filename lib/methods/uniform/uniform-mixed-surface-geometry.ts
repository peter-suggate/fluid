import { uniformDetailBindLayout, uniformDetailClass, uniformDetailModule, uniformDetailPipeline, uniformDetailPick, uniformDetailGroup, type UniformDetailGroup } from "./uniform-detail-fields";
import {UNIFORM_DETAIL_4H_LOAD} from "../../core/uniform-detail-abi";
import type { UniformMixedOwnership } from "./uniform-mixed-ownership";
import { UNIFORM_MIXED_COUNTED, uniformMixedCountedEntriesWGSL, uniformMixedTopologyWGSL } from "./uniform-mixed-topology.wgsl";
import { uniformMixedVertexSamplingSource } from "./uniform-mixed-vertex-sampling.wgsl";
import { uniformVolumeTargetWGSL } from "./uniform-volume.wgsl";
import { uniformSurfacePlaneWGSL } from "./uniform-surface-plane.wgsl";
import { geometricPlaneBoxWGSL } from "../../core/geometric-plane-box.wgsl";
import { UNIFORM_BODY_GEOMETRY_REACH, uniformMixedSolidPipeline, uniformMixedSolidWGSL, type UniformMixedSolid } from "./uniform-mixed-solid.wgsl";
import { uniformMixedChangedTilesWGSL } from "./uniform-mixed-layout-builder";
import { UNIFORM_LIVE_KEY_GROUPS } from "./uniform-live-lists";
import type { UniformSurfaceWorkControl } from "./uniform-surface-work-control";

/** Workgroups of the changed-tile geometry launch at most (grid-stride). */
const CHANGED_GRID=4096;
/** Workgroups of the regional launch and its mirror (grid-stride over the
 * plan's jobs: a job per h tile, one per 64 4h tiles). The plan's own grid
 * (a workgroup per few regions) left each workgroup tens of jobs in series. */
const REGIONAL_GRID=4096;
/** ceil(jobs/per) in whole steps: a fractional quotient is a heap number until its caller is optimized. */
const ceilDiv=(jobs:number,per:number):number=>{const up=jobs+per-1;return (up-up%per)/per;};
const GEOMETRY_PASS:GPUComputePassDescriptor={label:"Uniform mixed geometric fill"};
/** encode()'s options when none are given: a full launch of the simulation geometry. */
const NO_OPTIONS:{readonly pressure?:boolean;readonly changed?:GPUBuffer;readonly bodies?:boolean}={};
/** Workgroups of the body launch's region grid at most (grid-stride). */
const BODY_REGION_GRID=1024;
/** Retained h-normal slots at most (normalBytesAt): one owner tag each. */
const RETAINED_FINE_TILES=8192;

/** The planar-exact fill rule on mixed owners (corner tetrahedra elsewhere). Target
 * fractions are intensive, phi is physical, and no fine cell is expanded.
 * Subsequent pressure authority and conditioning consume these same values.
 *
 * pressure: the frame's all-4h pressure geometry rides the full launch
 * (encode(..., {pressure:true})): every tile's 4h owner is its tile, whose
 * corners are the tile corners, stored vertices in every layout. A 4h
 * simulation owner is that owner (the same eight texels, the same rule,
 * uncut), so it stores its values twice; an h tile's first lane evaluates the
 * rule once more on its tile corners. The values equal a separate launch over
 * the all-4h ownership.
 *
 * Changed tiles ({changed}): an owner's values are a function of its
 * corner texels (stored or resolved) and, for h owners, the static solid's
 * open fraction. Between two encodes without a phi or solid write other than
 * a relayout's remap and phi resolve, a texel changes only where the width of
 * an incident tile changed (the remap writes only vertices the old layout did
 * not store; tile corners are always stored, so a hanging vertex's resolved
 * value moves only when its incident widths do), and an owner's corners are
 * incident only to tiles in its tile's 3x3x3 neighbourhood. The changed launch
 * takes the GPU-built generation the ownership adopted since the last encode
 * (UniformMixedGenerationBuffers.changes): its dilated list is exactly the
 * 3x3x3 neighbourhoods of the tiles whose width that adoption changed, and
 * only those tiles' owners are evaluated: every other owner keeps the value
 * it already holds, which is the one the full launch would write. The
 * ownership's revision proves that one adoption, and nothing else, came
 * between the two encodes; the caller guarantees the rest of the
 * precondition (no body, solid edit or host phi write since the last encode). */
export class UniformMixedSurfaceGeometry {
  get allocatedBytes():number{return this.changes.size+(this.normalBuffer?.size??0)+(this.curvatureDirty?.size??0);}
  private normalBuffer?:GPUBuffer;
  readonly curvatureDirty?:GPUBuffer;
  get curvatureWorkOffset():number{return 8*(this.ownership.capacity.tiles/64);}
  get curvatureCountersOffset():number{return 16*(this.ownership.capacity.tiles/64);}
  get normalBinding():GPUBufferBinding{if(!this.normalBuffer)throw new Error("Persistent geometry normals require work control");return {buffer:this.normalBuffer};}
  /** Retain at most 8192 fine tiles; coarse records use physical tile keys.
   * Fine overflow is evaluated in the force stage's original borrowed scratch. */
  normalBytesAt(fineTiles:number):number{return 16*(64*Math.min(fineTiles,RETAINED_FINE_TILES)+this.ownership.capacity.tiles);}
  /** Between frames; callers rebuild geometry and force groups when true. */
  reserveNormals(fineTiles:number):boolean{
    if(!this.workControl)return false;const bytes=this.normalBytesAt(fineTiles);if(this.normalBuffer?.size===bytes)return false;
    const next=this.device.createBuffer({label:"Uniform persistent geometry normals and curvature",size:bytes,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
    if(this.normalBuffer){const e=this.device.createCommandEncoder();const coarseBytes=16*this.ownership.capacity.tiles,oldFine=this.normalBuffer.size-coarseBytes,newFine=bytes-coarseBytes;if(Math.min(oldFine,newFine)>0)e.copyBufferToBuffer(this.normalBuffer,0,next,0,Math.min(oldFine,newFine));e.copyBufferToBuffer(this.normalBuffer,oldFine,next,newFine,coarseBytes);this.device.queue.submit([e.finish()]);this.normalBuffer.destroy();}
    this.normalBuffer=next;this.workControl.invalidateGeometry();return true;
  }
  /** Optional last-launch [cheap plane, full target] evaluation counters. */
  get countersBuffer():GPUBuffer{return this.changes;}
  get countersOffset():number{return 8;}
  private pipeline?:GPUComputePipeline;
  private normalCandidatesPipeline?:GPUComputePipeline;
  private normalTagsPipeline?:GPUComputePipeline;
  private pressurePipeline?:GPUComputePipeline;
  private regionalPipeline?:GPUComputePipeline;
  /** Solid-free twins (umSolidFree) of the full launches, launched ahead of
   * them where the stage has a solid library: see umSolidSkip. */
  private freePipelines:GPUComputePipeline[]=[];
  private readonly changedPipelines:GPUComputePipeline[]=[];
  private changedMaintenancePipeline?:GPUComputePipeline;
  /** compactBodies: the swept-body regions' tiles, listed for geometryChanged. */
  private bodiesPipeline?:GPUComputePipeline;
  private readonly resources:GPUBindGroupLayout;
  private readonly mirrorResources:GPUBindGroupLayout;
  private mirrorPipeline?:GPUComputePipeline;
  /** The body launch's mirror: its listed owners only. */
  private mirrorListedPipeline?:GPUComputePipeline;
  private mirrorRegionalPipeline?:GPUComputePipeline;
  private regionalMaintenancePipeline?:GPUComputePipeline;
  private readonly mirrorGroups=new WeakMap<UniformDetailGroup,UniformDetailGroup>();
  /** [h count, 4h count, 2 unused, the marked h tiles, the marked 4h tiles]. */
  private readonly changes:GPUBuffer;
  /** The changed launch's group: a generation's changed tiles. */
  private readonly listResources:GPUBindGroupLayout;
  private listGroup?:{readonly changes:GPUBuffer;readonly group:UniformDetailGroup};
  /** ownership.revision at the last encode. */
  private encodedRevision?:number;
  /** phi's hanging texels hold umVertexValue (UniformMixedPhiResolve). bind()
   * also takes the all-4h pressure target and centre phi, and a second field
   * for the fill (the caller's volume is that measurement): every launch
   * copies each live owner's target into it in the same pass. */
  constructor(private readonly device:GPUDevice,readonly ownership:UniformMixedOwnership,private readonly solid?:UniformMixedSolid,private readonly workControl?:UniformSurfaceWorkControl){
    const tiles=ownership.capacity.tiles;
    this.changes=device.createBuffer({label:"Uniform mixed geometry changed tiles",size:4*(4+2*tiles),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
    this.listResources=uniformDetailBindLayout(device,{entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}}]});
    if(workControl){
      // Behind the force stage's 4*regions+1 words: the retained h-normal
      // slots' owner tags (normalTags).
      this.curvatureDirty=device.createBuffer({label:"Uniform pending 16h curvature changes",size:16*workControl.regions+4+4*RETAINED_FINE_TILES,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
      this.reserveNormals(ownership.capacity.fineTiles);
    }
    this.resources=uniformDetailBindLayout(device,{entries:[
      {binding:0,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float",viewDimension:"3d"}},
      ...[1,2,3,4].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"write-only" as const,format:"r32float" as const,viewDimension:"3d" as const}})),
      {binding:5,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},
      ...(workControl?[{binding:9,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage" as const}}]:[]),
      ...(workControl?[6,7,8].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage" as const}})):[]),
    ]});
    this.mirrorResources=uniformDetailBindLayout(device,{entries:[
      {binding:0,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float",viewDimension:"3d"}},
      {binding:1,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"write-only",format:"r32float",viewDimension:"3d"}},
      {binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},
      ...(workControl?[3,4].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage" as const}})):[]),
    ]});
  }
  bind(phi:GPUTexture,targetFill:GPUTexture,centerPhi:GPUTexture,pressure:{target:GPUTexture;centerPhi:GPUTexture},mirror:GPUTexture):UniformDetailGroup{
    const address=this.ownership.capacity.address;
    const outputs=[targetFill,centerPhi,pressure.target,pressure.centerPhi,mirror];
    for(const [i,t] of [phi,...outputs].entries())
      if(t.format!=="r32float"||uniformDetailClass(t)!==(i===0?"vertex":"cell"))throw new Error("Mixed geometry requires native cell and vertex fields");
    if(new Set(outputs).size!==outputs.length)throw new Error("Mixed geometry outputs must be disjoint");
    const group=uniformDetailGroup(this.device,{layout:this.resources,entries:[
      ...[phi,targetFill,centerPhi,pressure.target,pressure.centerPhi].map((t,binding)=>({binding,resource:t})),
      {binding:5,resource:{buffer:this.changes}},
      ...(this.workControl?[{binding:9,resource:{buffer:this.workControl.framePlan.buffer}},{binding:6,resource:{buffer:this.workControl.geometry}},{binding:7,resource:this.normalBinding},{binding:8,resource:{buffer:this.curvatureDirty!}}]:[]),
    ]});
    this.mirrorGroups.set(group,uniformDetailGroup(this.device,{layout:this.mirrorResources,entries:[{binding:0,resource:targetFill},{binding:1,resource:mirror},{binding:2,resource:{buffer:this.changes}},...(this.workControl?[{binding:3,resource:{buffer:this.workControl.framePlan.buffer}},{binding:4,resource:{buffer:this.workControl.geometry}}]:[])]}));
    return group;
  }
  async initialize():Promise<void>{
    const tiles=this.ownership.capacity.tiles;
    // A GPU-counted fixed grid over every live owner (all).
    const module=uniformDetailModule(this.device,{label:"Uniform mixed surface geometry",code:uniformMixedCountedEntriesWGSL(uniformMixedTopologyWGSL(this.ownership.capacity,0)+/* wgsl */`
@group(1) @binding(0) var phi:texture_3d<f32>;
@group(1) @binding(1) var targetFill:texture_storage_3d<r32float,write>;
@group(1) @binding(2) var centerPhi:texture_storage_3d<r32float,write>;
@group(1) @binding(3) var pressureTarget:texture_storage_3d<r32float,write>;
@group(1) @binding(4) var pressureCenterPhi:texture_storage_3d<r32float,write>;
@group(1) @binding(5) var<storage,read_write> changes:array<atomic<u32>>;
override geometryPressure:bool=false;
${this.workControl?this.workControl.framePlan.wgsl(1,9):""}
${this.workControl?"@group(1) @binding(6) var<storage,read_write> gcWork:array<atomic<u32>>; @group(1) @binding(7) var<storage,read_write> gcNormals:array<vec4f>; @group(1) @binding(8) var<storage,read_write> gcNormalDirty:array<atomic<u32>>;":""}
fn gcPlaneEnabled()->bool{return ${this.workControl?"(atomicLoad(&gcWork[3u*UM_PAGES+2u])&1u)!=0u":"true"};}
fn gcCount(cheap:bool){${this.workControl?"if((atomicLoad(&gcWork[3u*UM_PAGES+2u])&2u)!=0u){atomicAdd(&changes[select(3u,2u,cheap)],1u);}":""}}
fn gcNeeds(owner:UMOwner)->bool{return ${this.workControl?"atomicLoad(&gcWork[umPageOfTile(owner.tile)])!=0u":"true"};}
const GC_FINE:u32=4u;const GC_COARSE:u32=4u+UM_TILES;
fn umLoadVertex(p:vec3u)->f32{return textureLoad(phi,vec3i(p),0).x;}
// A tile corner: the base block (UNIFORM_DETAIL_4H_LOAD).
fn umLoadCorner(p:vec3u)->f32{return ${UNIFORM_DETAIL_4H_LOAD}textureLoad(phi,vec3i(p),0).x;}
// The same loads of the eight vertices p + s*umCorner(k,2u): one page-table read where one block holds them.
fn umLoadVertex8(p:vec3u,s:u32)->array<f32,8>{return udLoad8_phi(vec3i(p),s);}
fn umLoadCorner8(p:vec3u,s:u32)->array<f32,8>{return udLoad48_phi(vec3i(p),s);}
${uniformMixedVertexSamplingSource("",true,undefined,"umLoadCorner",true)}
${geometricPlaneBoxWGSL}
fn uvCorner(k:u32)->vec3i{return vec3i(umCorner(k,2u));}
fn d4Sum8(v:array<f32,8>)->f32{return ((v[0]+v[5])+(v[1]+v[4]))+((v[2]+v[7])+(v[3]+v[6]));}
${uniformVolumeTargetWGSL(true,true)}
${uniformSurfacePlaneWGSL}
${uniformMixedSolidWGSL(this.solid?2:undefined,this.solid?.coarse?.count)}
// The pressure variant's h tile job: each lane's cell's term of the tile's
// all-4h owner's F_Omega (a cut tile's 64 terms on its origin lane alone were
// the launch's serial depth). gcSharedFill adds them in cell order.
var<workgroup> gcTerms:array<f32,64>;
fn gcShare(owner:UMOwner){
 if(owner.width!=1u||umTileOpen(owner.tile)>=0.99999){return;}
 let base=umTileCoord(owner.tile)*4u;var corners:array<f32,8>;
 corners=umLoadCorner8(base,4u);
 // gcEvaluate's far air and deep liquid take no terms.
 var low=corners[0];var high=corners[0];var positive=true;var negative=true;
 for(var j=0u;j<8u;j++){let v=corners[j];low=min(low,v);high=max(high,v);positive=positive&&v>1e-20;negative=negative&&v< -1e-20;}
 if((negative&&4.0*(high-low)<= -high)||(positive&&4.0*(high-low)<=low)){return;}
 gcTerms[owner.lane]=umSurfaceTargetOpenTerm(corners,vec3i(base),owner.lane);
}
fn gcSharedFill()->f32{var fill=0.0;for(var s=0u;s<64u;s++){fill+=gcTerms[s];}return fill/64.0;}
// Whether umSurfaceTarget returns exactly 0 (all corners above 1e-20) or 1
// (all below -1e-20) for these corners. Its tetrahedra then take one sign
// each: 0 or 1 exactly. Its plane, where it takes one, fits the eight
// quarter probes within r <= 1e-4(1 + M) (M the largest |corner|, which
// bounds every probe), and at box corner s is 2 plane(probe s) - centre >=
// 2 sample(s) - centre - 2r. sample(s) weighs corner s 27/64, its three
// neighbours 9/64, the next three 3/64 and the opposite 1/64, and centre is
// the corners' mean, so 2 sample(s) - centre is (46a + 10B - 2C - 6d)/64.
// Where that clears 3e-4(1 + M) at every corner the plane keeps its sign
// over the box by 1e-4(1 + M), far past rounding, and its box fraction is the
// same literal.
fn gcSameSign(v:array<f32,8>,positive:bool,negative:bool)->bool{
 if(!positive&&!negative){return false;}
 var m=0.0;for(var k=0u;k<8u;k++){m=max(m,abs(v[k]));}
 let margin=3e-4*(1.0+m);
 for(var s=0u;s<8u;s++){
  let bound=(46.0*v[s]+10.0*(v[s^1u]+v[s^2u]+v[s^4u])-2.0*(v[s^3u]+v[s^5u]+v[s^6u])-6.0*v[s^7u])/64.0;
  if(positive&&!(bound>margin)){return false;}
  if(negative&&!(bound< -margin)){return false;}
 }
 return true;
}
// (fill, centre phi) from an owner's eight corner values. open: the open
// fraction of a 4h owner's tile; 1 for an h owner, whose caller scales by its
// cell's. The fill is absolute: an uncut owner's fraction times open, a cut
// one's F_Omega over its h cells' open fractions (terms: its cells' are in gcTerms).
fn gcEvaluate(owner:UMOwner,vertices:array<f32,8>,open:f32,terms:bool)->vec2f{
 var centre:array<f32,8>;for(var j=0u;j<8u;j++){centre[j]=vertices[j]*0.125;}
 // Far air and deep liquid: umSurfaceTarget returns exactly 0.0 (1.0) when
 // every corner is above 1e-20 (below -1e-20) and the corner spread is at
 // most a quarter of the smallest magnitude m. Each probe is then a sum of
 // nonnegative (nonpositive) dyadic-weighted terms, strictly one sign, so
 // fill is 0 (8); |gradient| <= 2(M-m) <= m/2 < |centre|/1.9 (relative
 // rounding ~1e-6), so the plane's shifted offset is <= -0.49 (>= total+0.49)
 // and geometricPlaneBoxFraction returns the same literal. Both arms of its
 // select agree, whatever the residual. A NaN corner fails every comparison;
 // an infinite one makes the spread test fail.
 var low=vertices[0];var high=vertices[0];var positive=true;var negative=true;
 for(var j=0u;j<8u;j++){let v=vertices[j];low=min(low,v);high=max(high,v);positive=positive&&v>1e-20;negative=negative&&v< -1e-20;}
 var fraction=0.0;
 if(negative&&4.0*(high-low)<= -high){fraction=open;}
 else if(!(positive&&4.0*(high-low)<=low)){
  if(open>=0.99999){
   var plane=vec2f(-1.0,0.0);if(gcPlaneEnabled()){plane=gcCurrentPlane(vertices);}
   if(plane.x>=0.0){gcCount(true);fraction=plane.x*open;}
   else if(gcSameSign(vertices,positive,negative)){gcCount(true);fraction=select(0.0,open,negative);}
   else{gcCount(false);fraction=umSurfaceTarget(owner,vertices)*open;}
  }else if(terms){gcCount(false);fraction=gcSharedFill();}else{gcCount(false);fraction=umSurfaceTargetOpen(vertices,vec3i(umOrigin(owner)));}
 }
 // umSampleVertex at the owner centre: all eight weights are exactly 1/8 of
 // these same corner values (stored, or reconstructed where hanging).
 return vec2f(fraction,d4Sum8(centre));
}
${this.workControl?`@compute @workgroup_size(64) fn normalCandidates(@builtin(global_invocation_id) id:vec3u){
 let region=id.x;if(region>=UM_PAGES){return;}let origin=16u*umPageCoord(region);var candidate=false;var topology=false;
 for(var z=-1;z<=1;z++){for(var y=-1;y<=1;y++){for(var x=-1;x<=1;x++){
  let tile=umTileKey(vec3i(origin/4u)+4*vec3i(x,y,z));if(!umKeyStored(tile)){continue;}let other=umPageOfTile(tile);
  candidate=candidate||atomicLoad(&gcWork[other])!=0u;topology=topology||atomicLoad(&gcWork[${this.workControl.topologyWordOffset}u+other])!=0u;
 }}}
 if(candidate){atomicStore(&gcNormalDirty[region],1u);}if(topology){atomicStore(&gcNormalDirty[3u*UM_PAGES+region],1u);}
}
// The retained h normals are indexed by owner rank (h tiles in ascending
// key): a relayout adding or removing any h tile shifts every later tile's
// slot, where only the changed tiles' neighbourhoods are marked. Each slot
// is tagged with the tile (key + 1) whose normals it holds; a tile in a slot
// tagged otherwise marks its region's normals topology-changed (the force
// stage then re-evaluates and rewrites them) and takes the slot.
const GC_NORMAL_TAGS:u32=4u*UM_PAGES+1u;
@compute @workgroup_size(64) fn normalTags(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) groups:vec3u){
 let count=min(umCounts.x,${RETAINED_FINE_TILES}u);
 for(var j=gid.x;j<count;j+=64u*groups.x){
  let tile=umTopology[UM_TILES+j];let slot=(umTopology[tile]&0x3fffffffu)/64u;
  if(slot>=${RETAINED_FINE_TILES}u||atomicLoad(&gcNormalDirty[GC_NORMAL_TAGS+slot])==tile+1u){continue;}
  atomicStore(&gcNormalDirty[GC_NORMAL_TAGS+slot],tile+1u);
  let region=umPageOfTile(tile);atomicStore(&gcNormalDirty[region],1u);atomicStore(&gcNormalDirty[3u*UM_PAGES+region],1u);
 }
}`:""}
// mode 0: the owner, and on an h tile's lane 0 of the pressure variant its
// tile's all-4h owner too. 1 (gcSplitTurn): that second turn only where the
// tile is cut (it reads the job's gcTerms); 2: an uncut h tile's all-4h
// owner alone (owner = UMOwner(tile, 0, 4, 0)), its pressure outputs only,
// on a lane of its own instead of lane 0's second turn.
fn gcOwner(owner:UMOwner,terms:bool,mode:u32){
 let origin=umOrigin(owner);
 // select eagerly evaluates its operands: coarse owners must not evaluate
 // discarded fractional h-cell geometry (the tile fraction is in value.x).
 var cellOpen=1.0;if(owner.width==1u){cellOpen=umCellOpen(vec3i(origin));}
 // gcEvaluate has one call site (Metal inlines it whole): turn 0 is this
 // owner, turn 1 the pressure variant's lane-0 tile owner of an h tile.
 let turns=select(1u,2u,geometryPressure&&owner.lane==0u&&owner.width!=4u&&(mode!=1u||umTileOpen(owner.tile)<0.99999));
 var value=vec2f(0.0);
 for(var r=0u;r<turns;r++){
  // An owner's corners are width-aligned vertices of its tile closure. Their
  // tile's stencil holds the owner, so it is unit or mixed (a unit owner) or
  // the corner is 4-aligned (a 4h owner, or the tile's all-4h owner): the
  // resolved field's umVertexValue is this load, without its one or two
  // stencil loads per corner.
  var job=owner;var vertices:array<f32,8>;var open=1.0;
  if(r!=0u){job=UMOwner(owner.tile,0u,4u,0u);}
  if(job.width==4u){vertices=umLoadCorner8(origin,4u);open=umTileOpen(owner.tile);}
  else{vertices=umLoadVertex8(origin,job.width);}
  value=gcEvaluate(job,vertices,open,terms&&r!=0u);
  // Native uvTarget scales by the open fraction; a 4h owner's fill is already
  // over its tile's open cells.
  if(r==0u&&mode!=2u){textureStore(targetFill,vec3i(origin),vec4f(value.x*cellOpen));
   textureStore(centerPhi,vec3i(origin),vec4f(value.y));}
 }
 if(geometryPressure&&owner.lane==0u&&(owner.width==4u||turns==2u)){
  textureStore(pressureTarget,vec3i(origin),vec4f(value.x));textureStore(pressureCenterPhi,vec3i(origin),vec4f(value.y));
 }
}
@compute @workgroup_size(64) fn geometry(@builtin(global_invocation_id) gid:vec3u){
 if(umSolidSkip()){return;}
 let owner=umAllOwner(gid);
 if(geometryPressure){if(owner.width!=0u&&gcNeeds(owner)){gcShare(owner);}workgroupBarrier();}
 if(owner.width==0u||!gcNeeds(owner)){return;}gcOwner(owner,geometryPressure,0u);
}
// Changed tiles: the adopted generation's dilated list (each tile once),
// split by width. A fixed grid strides the list.
${uniformMixedChangedTilesWGSL(tiles,this.solid?3:2,0)}
// A listed tile: its region's controller words dirty and its snapshot
// stale (the next full launch recomputes it), then split by width. Changed
// geometry writes simulation outputs independently of the last full
// certificate; a zero-count list costs no reuse.
fn gcList(tile:u32){
 ${this.workControl?`let region=umPageOfTile(tile);atomicStore(&gcWork[region],3u);atomicStore(&gcWork[${this.workControl.topologyWordOffset}u+region],1u);atomicStore(&gcNormalDirty[region],1u);atomicStore(&gcNormalDirty[3u*UM_PAGES+region],1u);let mapped=atomicLoad(&gcWork[${this.workControl.mappingWordOffset}u+region]);if(mapped!=0u){let local=tile&511u;let lane=(local&3u)|((local>>1u)&12u)|((local>>2u)&48u);atomicStore(&gcWork[${this.workControl.snapshotOffset}u+132u*(64u*(mapped-1u)+lane)+131u],0u);}`:""}
 if(umTileWidth(tile)==1u){atomicStore(&changes[GC_FINE+atomicAdd(&changes[0],1u)],tile);}
 else{atomicStore(&changes[GC_COARSE+atomicAdd(&changes[1],1u)],tile);}
}
@compute @workgroup_size(64) fn compactChanged(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) groups:vec3u){
 for(var i=gid.x;i<umDilatedCount();i+=64u*groups.x){gcList(umDilatedTile(i));}
}
// A job per marked h tile (a lane per owner), then 64 marked 4h tiles per job.
var<workgroup> gcJobs:vec2u;
override geometryChangedMaintenance:bool=false;
fn gcChangedNormals(tile:u32){
 ${this.workControl?`if(geometryChangedMaintenance){
  let region=umPageOfTile(tile);
  // Compacting completed before this dispatch. Exactly one tile consumes
  // each region's pending marker, even when many changed tiles overlap it.
  if(atomicExchange(&gcWork[region],1u)==3u){
   for(var k=0u;k<27u;k++){
    let neighbour=umTileKey(4*(vec3i(umPageCoord(region))+vec3i(umCorner(k,3u))-vec3i(1)));
    if(umKeyStored(neighbour)){let other=umPageOfTile(neighbour);atomicStore(&gcNormalDirty[other],1u);atomicStore(&gcNormalDirty[3u*UM_PAGES+other],1u);}
   }
  }
 }`:""}
}
@compute @workgroup_size(64) fn geometryChanged(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 if(lane==0u){let fine=atomicLoad(&changes[0]);gcJobs=vec2u(fine,fine+(atomicLoad(&changes[1])+63u)/64u);}
 let jobs=workgroupUniformLoad(&gcJobs);
 for(var job=group.x;job<jobs.y;job+=groups.x){
  if(job<jobs.x){let tile=atomicLoad(&changes[GC_FINE+job]);if(lane==0u){gcChangedNormals(tile);}gcOwner(UMOwner(tile,lane,1u,(umTopology[tile]&0x3fffffffu)+lane),false,0u);}
  else{
   let slot=(job-jobs.x)*64u+lane;
   if(slot<atomicLoad(&changes[1])){let tile=atomicLoad(&changes[GC_COARSE+slot]);gcChangedNormals(tile);gcOwner(UMOwner(tile,0u,4u,umTopology[tile]&0x3fffffffu),false,0u);}
  }
 }
}
${this.workControl?/* wgsl */`
override geometryRegionalMaintenance:bool=false;
var<workgroup> gcRegionalJobs:vec3u;
// The frame plan supplies one shared tile domain. Exact producer revisions
// still decide which retained outputs need evaluation inside that domain.
@compute @workgroup_size(64) fn geometryRegional(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 // Publish curvature invalidation from changed regions to their neighbours.
 // Geometry control completed before this dispatch; these atomic ORs have
 // no dependency on the geometry evaluations below. Retained flags stay set
 // until the force stage consumes them.
 if(geometryRegionalMaintenance){
  for(var slot=group.x;slot<regionalPlan[3u];slot+=groups.x){
   let region=regionalPlan[${this.workControl.framePlan.layout.lists}u+3u*UM_PAGES+slot];
   if(lane<27u){
    let dirty=atomicLoad(&gcWork[region])!=0u;let topology=atomicLoad(&gcWork[${this.workControl.topologyWordOffset}u+region])!=0u;
    if(dirty||topology){let tile=umTileKey(4*(vec3i(umPageCoord(region))+vec3i(umCorner(lane,3u))-vec3i(1)));
     if(umKeyStored(tile)){let other=umPageOfTile(tile);if(dirty){atomicStore(&gcNormalDirty[other],1u);}if(topology){atomicStore(&gcNormalDirty[3u*UM_PAGES+other],1u);}}
    }
   }
  }
 }
 // Jobs: an h tile each (a lane per owner), 64 4h tiles each, then 64 h
 // tiles' all-4h owners each (gcOwner mode 2, uncut tiles; a cut tile's
 // takes its job's lane 0, which holds its gcTerms).
 if(lane==0u){let fine=regionalPlan[7u];let coarse=fine+(regionalPlan[8u]+63u)/64u;gcRegionalJobs=vec3u(fine,coarse,coarse+(fine+63u)/64u);}
 let jobs=workgroupUniformLoad(&gcRegionalJobs);
 for(var job=group.x;job<jobs.z;job+=groups.x){
  var owner=UMOwner(0u,0u,0u,0u);var mode=1u;
  if(job<jobs.x){let tile=regionalPlan[${this.workControl.framePlan.layout.fine}u+job];owner=UMOwner(tile,lane,1u,(umTopology[tile]&0x3fffffffu)+lane);}
  else if(job<jobs.y){let slot=(job-jobs.x)*64u+lane;if(slot<regionalPlan[8u]){let tile=regionalPlan[${this.workControl.framePlan.layout.coarse}u+slot];owner=UMOwner(tile,0u,4u,umTopology[tile]&0x3fffffffu);}}
  else{let slot=(job-jobs.y)*64u+lane;mode=2u;if(slot<jobs.x){let tile=regionalPlan[${this.workControl.framePlan.layout.fine}u+slot];if(umTileOpen(tile)>=0.99999){owner=UMOwner(tile,0u,4u,0u);}}}
  let needed=owner.width!=0u&&gcNeeds(owner);
  if(needed&&mode!=2u){gcShare(owner);}workgroupBarrier();
  if(needed){gcOwner(owner,true,mode);}workgroupBarrier();
 }
}`:""}
${this.solid?/* wgsl */`// Moved bodies ({bodies}): every live tile of a region the swept-body mask
// reaches within ${UNIFORM_BODY_GEOMETRY_REACH}h. An owner's values read the solid through its own
// open fractions alone, which a body changes only in tiles its swept support
// reaches (umBodyTileDirty's 2h). A fixed grid strides the regions.
var<workgroup> gcBodyRegion:u32;
@compute @workgroup_size(64) fn compactBodies(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 for(var region=group.x;region<UM_PAGES;region+=groups.x){
  if(lane==0u){gcBodyRegion=u32(umBodySweptReach(region)<=${UNIFORM_BODY_GEOMETRY_REACH}u);}
  if(workgroupUniformLoad(&gcBodyRegion)!=0u){let tile=umPageTile(region,lane);if(umTileLive(tile)){gcList(tile);}}
 }
}`:""}`,["geometry"])});
    const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");if(errors.length)throw new Error(errors.map(m=>`${m.lineNum}: ${m.message}`).join("\n"));
    const groups=[this.ownership.bindLayout,this.resources,...(this.solid?[this.solid.tileLayout]:[])];
    const layout=this.device.createPipelineLayout({bindGroupLayouts:groups}),listed=this.device.createPipelineLayout({bindGroupLayouts:[...groups,this.listResources]});
    const create=(entryPoint:string,constants:Record<string,number>={},pipelineLayout=layout)=>uniformMixedSolidPipeline(this.solid,s=>uniformDetailPipeline(this.device,this.ownership,{layout:pipelineLayout,compute:{module,entryPoint,constants:{umDispatchX:this.ownership.dispatchX,...constants,...s}}}),{entry:entryPoint});
    const all={umCountedJobs:UNIFORM_MIXED_COUNTED.all};
    const twinned:Record<string,number>=this.solid?{umSolidTwinned:1}:{};
    if(this.workControl){
      // Every full controlled launch keeps both output families coherent;
      // the normal variants are unreachable and need no driver compilation.
      this.regionalPipeline=await create("geometryRegional",{geometryPressure:1});
      this.regionalMaintenancePipeline=await create("geometryRegional",{geometryPressure:1,geometryRegionalMaintenance:1});
      this.pressurePipeline=await create("geometry",{...all,geometryPressure:1,...twinned});
      this.pipeline=this.pressurePipeline;
      if(this.solid){const free=await create("geometry",{...all,geometryPressure:1,umSolidFree:1});this.freePipelines=[free,free];}
    }else{
      [this.pipeline,this.pressurePipeline]=await Promise.all([create("geometry",{...all,...twinned}),create("geometry",{...all,geometryPressure:1,...twinned})]);
      if(this.solid)this.freePipelines=await Promise.all([create("geometry",{...all,umSolidFree:1}),create("geometry",{...all,geometryPressure:1,umSolidFree:1})]);
    }
    // Every live owner's texel, as a launch above wrote it.
    const mirror=uniformDetailModule(this.device,{label:"Uniform mixed geometric fill mirror",code:uniformMixedCountedEntriesWGSL(uniformMixedTopologyWGSL(this.ownership.capacity,0)+/* wgsl */`
@group(1) @binding(0) var volume:texture_3d<f32>;
@group(1) @binding(1) var outputVolume:texture_storage_3d<r32float,write>;
@group(1) @binding(2) var<storage,read> listedTiles:array<u32>;
${this.workControl?`@group(1) @binding(3) var<storage,read> mirrorPlan:array<u32>;
@group(1) @binding(4) var<storage,read> mirrorWork:array<u32>;
@compute @workgroup_size(64) fn mirrorRegional(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 let fine=mirrorPlan[7u];let coarse=mirrorPlan[8u];let jobs=fine+(coarse+63u)/64u;
 for(var job=group.x;job<jobs;job+=groups.x){
  var owner=UMOwner(0u,0u,0u,0u);
  if(job<fine){let tile=mirrorPlan[${this.workControl.framePlan.layout.fine}u+job];owner=UMOwner(tile,lane,1u,(umTopology[tile]&0x3fffffffu)+lane);}
  else{let slot=(job-fine)*64u+lane;if(slot<coarse){let tile=mirrorPlan[${this.workControl.framePlan.layout.coarse}u+slot];owner=UMOwner(tile,0u,4u,umTopology[tile]&0x3fffffffu);}}
  if(owner.width!=0u&&mirrorWork[umPageOfTile(owner.tile)]!=0u){let origin=vec3i(umOrigin(owner));textureStore(outputVolume,origin,vec4f(textureLoad(volume,origin,0).x));}
 }
}`:""}
@compute @workgroup_size(64) fn mirror(@builtin(global_invocation_id) gid:vec3u){
 let owner=umAllOwner(gid);if(owner.width==0u){return;}
 let origin=vec3i(umOrigin(owner));textureStore(outputVolume,origin,vec4f(textureLoad(volume,origin,0).x));
}
// The body launch's owners alone (geometryChanged's jobs): nothing else
// wrote a target or the mirror since the last launch's mirror.
@compute @workgroup_size(64) fn mirrorListed(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 let fine=listedTiles[0];let coarse=listedTiles[1];let jobs=fine+(coarse+63u)/64u;
 for(var job=group.x;job<jobs;job+=groups.x){
  var origin=vec3i(-1);
  if(job<fine){let tile=listedTiles[4u+job];origin=vec3i(umOrigin(UMOwner(tile,lane,1u,(umTopology[tile]&0x3fffffffu)+lane)));}
  else{let slot=(job-fine)*64u+lane;if(slot<coarse){let tile=listedTiles[4u+UM_TILES+slot];origin=vec3i(umOrigin(UMOwner(tile,0u,4u,umTopology[tile]&0x3fffffffu)));}}
  if(origin.x>=0){textureStore(outputVolume,origin,vec4f(textureLoad(volume,origin,0).x));}
 }
}`,["mirror"])});
    const mirrorErrors=(await mirror.getCompilationInfo()).messages.filter(m=>m.type==="error");if(mirrorErrors.length)throw new Error(mirrorErrors.map(m=>`${m.lineNum}: ${m.message}`).join("\n"));
    const mirrorLayout=this.device.createPipelineLayout({bindGroupLayouts:[this.ownership.bindLayout,this.mirrorResources]});
    this.mirrorPipeline=await uniformDetailPipeline(this.device,this.ownership,{layout:mirrorLayout,compute:{module:mirror,entryPoint:"mirror",constants:{umDispatchX:this.ownership.dispatchX,...all}}});
    if(this.workControl)this.mirrorRegionalPipeline=await uniformDetailPipeline(this.device,this.ownership,{layout:mirrorLayout,compute:{module:mirror,entryPoint:"mirrorRegional"}});
    if(this.solid)this.mirrorListedPipeline=await uniformDetailPipeline(this.device,this.ownership,{layout:mirrorLayout,compute:{module:mirror,entryPoint:"mirrorListed",constants:{umDispatchX:this.ownership.dispatchX}}});
    if(this.workControl){this.normalCandidatesPipeline=await create("normalCandidates",{},layout);this.normalTagsPipeline=await create("normalTags",{},layout);}
    // The changed launch follows a GPU relayout's changed tiles (Dynamic).
    this.changedPipelines.push(...await Promise.all([create("compactChanged",{},listed),create("geometryChanged",{},layout)]));
    if(this.workControl)this.changedMaintenancePipeline=await create("geometryChanged",{geometryChangedMaintenance:1},layout);
    if(this.solid)this.bodiesPipeline=await create("compactBodies",{},layout);
  }
  /** pressure: also write the all-4h pressure geometry (a full launch).
   * changed: the changed tiles (UniformMixedGenerationBuffers.changes) of the
   * generation the ownership adopted since the last encode: evaluate only
   * their neighbourhoods (see the class comment for the caller's precondition).
   * bodies: only rigid bodies moved since the last encode (no adoption, phi
   * or other solid write): evaluate the tiles of the regions the solid
   * record's swept-body mask reaches (UNIFORM_BODY_GEOMETRY_REACH) through the
   * changed launch, and take the mask's evidence. */
  encode(encoder:GPUCommandEncoder,group:UniformDetailGroup,options:{pressure?:boolean;changed?:GPUBuffer;bodies?:boolean}=NO_OPTIONS):void{
    if(!this.pipeline||!this.pressurePipeline||!this.mirrorPipeline)throw new Error("Mixed surface geometry is not initialized");
    if(options.pressure&&(options.changed||options.bodies))throw new Error("Mixed geometry writes pressure outputs only in a full launch");
    if(options.bodies&&(options.changed||!this.bodiesPipeline))throw new Error(options.changed?"Mixed geometry's body launch takes no changed tiles":"Mixed geometry's body launch needs solids");
    const revision=this.ownership.revision;
    if(options.changed&&this.encodedRevision!==revision-1)throw new Error(`Mixed geometry's changed launch needs exactly one adoption since its last encode (revision ${this.encodedRevision} -> ${revision})`);
    if(options.bodies&&this.encodedRevision!==revision)throw new Error(`Mixed geometry's body launch needs no adoption since its last encode (revision ${this.encodedRevision} -> ${revision})`);
    const sameRevision=this.encodedRevision===revision;
    this.encodedRevision=revision;
    const listed=!!options.changed||!!options.bodies;
    const regional=!listed&&this.workControl?.framePlan.active&&!this.workControl.geometryRefreshRequired;
    const changedMaintenance=listed&&this.workControl?.regionalGeometrySetup;
    const sharedSetup=regional&&sameRevision&&this.workControl!.regionalGeometryReady;
    encoder.clearBuffer(this.changes,0,16);
    if(sharedSetup)encoder.clearBuffer(this.workControl!.geometry,this.workControl!.countersOffset,16);
    else if(!listed)this.workControl?.encode(encoder,"geometry");
    if(options.changed){
      if(this.listGroup?.changes!==options.changed)this.listGroup={changes:options.changed,group:uniformDetailGroup(this.device,{layout:this.listResources,entries:[{binding:0,resource:{buffer:options.changed}}]})};
    }
    const pass=encoder.beginComputePass(GEOMETRY_PASS);if(sharedSetup)this.workControl!.encodeRegionalGeometry(pass);pass.setBindGroup(0,this.ownership.bindGroup);pass.setBindGroup(1,group.group);if(this.solid)pass.setBindGroup(2,this.solid.tileGroup);
    if(options.changed){
      const compact=this.changedPipelines[0]!,changed=changedMaintenance?this.changedMaintenancePipeline!:this.changedPipelines[1]!,tiles=this.ownership.capacity.tiles;
      pass.setBindGroup(this.solid?3:2,this.listGroup!.group.group);
      pass.setPipeline(this.variant(compact));pass.dispatchWorkgroups(Math.min(UNIFORM_LIVE_KEY_GROUPS,ceilDiv(tiles,64)));
      pass.setPipeline(this.variant(changed));pass.dispatchWorkgroups(Math.min(CHANGED_GRID,tiles));
    }
    else if(options.bodies){
      const tiles=this.ownership.capacity.tiles;
      pass.setPipeline(this.variant(this.bodiesPipeline!));pass.dispatchWorkgroups(Math.min(BODY_REGION_GRID,tiles/64));
      pass.setPipeline(this.variant(changedMaintenance?this.changedMaintenancePipeline!:this.changedPipelines[1]!));pass.dispatchWorkgroups(Math.min(CHANGED_GRID,tiles));
    }
    else if(regional){pass.setPipeline(this.variant(sharedSetup?this.regionalMaintenancePipeline!:this.regionalPipeline!));pass.dispatchWorkgroups(Math.min(REGIONAL_GRID,this.ownership.capacity.tiles));}
    else{
      // Both output families share one exact certificate. Keeping them coherent
      // here allows reuse across intermediate full geometry stages.
      const pressure=!!this.workControl||!!options.pressure;
      const free=this.freePipelines[pressure?1:0];if(free)this.ownership.dispatchAllCounted(pass,uniformDetailPick(free));
      this.ownership.dispatchAllCounted(pass,this.variant(pressure?this.pressurePipeline:this.pipeline));
    }
    if(this.normalCandidatesPipeline&&!sharedSetup){if(!changedMaintenance){pass.setPipeline(this.variant(this.normalCandidatesPipeline));pass.dispatchWorkgroups(Math.ceil(this.workControl!.regions/64));}
     pass.setPipeline(this.variant(this.normalTagsPipeline!));pass.dispatchWorkgroups(RETAINED_FINE_TILES/64);}
    // Every live owner, after a changed launch too: the mirror may have been
    // remapped in tiles this launch did not evaluate. The body launch follows
    // a gather with no write since (no adoption): its listed owners alone.
    const mirror=this.mirrorGroups.get(group);if(!mirror)throw new Error("Mixed geometry group was not bound by this stage");
    pass.setBindGroup(1,mirror.group);
    if(options.bodies){pass.setPipeline(uniformDetailPick(this.mirrorListedPipeline!));pass.dispatchWorkgroups(Math.min(CHANGED_GRID,this.ownership.capacity.tiles));}
    else if(sharedSetup){pass.setPipeline(uniformDetailPick(this.mirrorRegionalPipeline!));pass.dispatchWorkgroups(Math.min(REGIONAL_GRID,this.ownership.capacity.tiles));}
    else this.ownership.dispatchAllCounted(pass,uniformDetailPick(this.mirrorPipeline));
    pass.end();
    if(options.bodies||sharedSetup)this.solid?.consumeBodyEvidence(encoder);
  }
  private variant(p:GPUComputePipeline):GPUComputePipeline{return uniformDetailPick(this.solid?.select(p)??p);}
  destroy():void{this.changes.destroy();this.normalBuffer?.destroy();this.curvatureDirty?.destroy();}
}
