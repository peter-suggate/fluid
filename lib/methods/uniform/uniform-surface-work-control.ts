import { uniformDetailBindLayout, uniformDetailGroup, uniformDetailModule, uniformDetailPipeline, uniformDetailPick, type UniformDetailGroup } from "./uniform-detail-fields";
import { UNIFORM_DETAIL_4H_LOAD } from "../../core/uniform-detail-abi";
import type { UniformMixedOwnership } from "./uniform-mixed-ownership";
import { uniformMixedTopologyWGSL } from "./uniform-mixed-topology.wgsl";
import { UNIFORM_BODY_GEOMETRY_REACH, uniformMixedSolidWGSL, type UniformMixedSolid } from "./uniform-mixed-solid.wgsl";
import { uniformPagePoolWGSL, type UniformPagePool } from "./uniform-page-pool";
import { uniformPageInitBlocksWGSL, uniformPageInitGroups } from "./uniform-page-initialise";
import { UNIFORM_SURFACE_SUPPORT_WORDS } from "./uniform-surface-transport.wgsl";
import { UniformRegionalFramePlan } from "./uniform-regional-frame-plan";
import { uniformRegionHeatWGSL, type UniformRegionHeat } from "./uniform-region-heat";
import { UNIFORM_LIVE_GROUP_GROUPS, uniformLiveListsWGSL } from "./uniform-live-lists";

/** Shared current surface models and exact derived-geometry evidence on a
 * 16h lattice, independent of 32h paging. Surface model scans retain no tile
 * snapshots: producer revisions plus compact topology/world identity retain
 * exact affine fits, residuals and slopes before phi gathers. Coverage and
 * velocity support refresh every call. Geometry retains exact tile closures in a
 * bounded 512-region cache; misses conservatively refresh. Topology, page
 * claims, and the solid record's swept-body mask invalidate derived geometry:
 * only regions within GEOMETRY_BODY_REACH h of the support swept by bodies
 * that moved since the last record build, taken once per frame head
 * (UniformMixedSolid.consumeBodyEvidence). A body at rest, or one far away,
 * leaves every retained region on its retained path.
 */
const GEOMETRY_BODY_REACH=UNIFORM_BODY_GEOMETRY_REACH;
/** Workgroups a region in the surface model scan (64/WC_SCAN_PARTS tiles each). */
const WC_SCAN_PARTS=8;
export class UniformSurfaceWorkControl {
  readonly label="Uniform 16h surface work control";
  readonly regions:number;
  readonly cacheRegions:number;
  readonly snapshotOffset:number;
  readonly mappingWordOffset:number;
  /** Eight words/region: gradient.xyz, centre phi, halo residual, 27-bit
   * coverage, local residual, local validity. Centres are origin + 8h. */
  readonly modelWordOffset:number;
  /** Six float words/region: halo minimum.xyz, maximum.xyz. */
  readonly velocityWordOffset:number;
  /** Six exact float extrema/region, low.xyz then high.xyz. Consumers union
   * only their required support, then add interpolation rounding allowance. */
  readonly localVelocityWordOffset:number;
  /** Three current native-edge slope bounds/region, in phi metres/base cell. */
  readonly slopeWordOffset:number;
  /** 64 rectangular supports/region: rho,lo.xyz,hi.xyz,flags,slope.xyz.
   * Flags: bit0 valid; bit1 every current solid cell in the support is open. */
  readonly supportWordOffset:number;
  /** Current geometry topology/solid changes, separate from phi changes. */
  readonly topologyWordOffset:number;
  readonly liveEpochWordOffset:number;
  readonly scratchEpochWordOffset:number;
  readonly topologyEpochWordOffset:number;
  private readonly modelEpochWordOffset:number;
  private readonly modelReuseWordOffset:number;
  private readonly modelMetadataWordOffset:number;
  private readonly modelWorldWordOffset:number;
  private readonly geometryLiveWordOffset:number;
  private readonly geometryEpochWordOffset:number;
  private readonly geometryRemapWordOffset:number;
  private readonly tierWordOffset:number;
  private remapPipeline?:GPUComputePipeline;
  private remapLayout?:GPUBindGroupLayout;
  private remapGroup?:{changes:GPUBuffer;group:GPUBindGroup};
  private revisionGeneration=0;
  get generation():number{return this.revisionGeneration;}
  /** The live and scratch phi revisions, and (with region heat) a word a
   * region its consumers copy the work tiers into (tierOffset, bytes) and
   * one their non-COLD epochs (epochOffset, bytes). */
  get revisions(){return {buffer:this.surface,liveOffset:4*this.liveEpochWordOffset,scratchOffset:4*this.scratchEpochWordOffset,topologyOffset:4*this.topologyEpochWordOffset,regions:this.regions,tiers:this.heat?.tiers,tierOffset:4*this.tierWordOffset,epochOffset:4*(this.tierWordOffset+this.regions)};}
  readonly surface:GPUBuffer;
  readonly geometry:GPUBuffer;
  readonly allocatedBytes:number;
  readonly framePlan:UniformRegionalFramePlan;
  /** The 16h surface models and supports for advect's affine and identity
   * shortcuts. Off by default: on the pour the stage cost 1.25 ms and the
   * shortcuts made advect 0.6 ms slower, not faster. */
  surfaceEvidence=false;
  /** Diagnostic full-work control: same scans/launches, no reuse or plane model. */
  enabled=true;
  /** Diagnostic ablation of the sealed-plan geometry setup. */
  regionalGeometrySetup=true;
  private snapshotGeometry=true;
  /** Diagnostic ablation: retain topology evidence but always evaluate current geometry. */
  get geometrySnapshots():boolean{return this.snapshotGeometry;}
  set geometrySnapshots(value:boolean){if(value!==this.snapshotGeometry){this.snapshotGeometry=value;this.forced[1]=true;}}
  /** Optional per-geometry-launch plane/full counters; no owner atomics otherwise. */
  collectGeometryStats=false;
  private readonly layout:GPUBindGroupLayout;
  private readonly groups:UniformDetailGroup[];
  private readonly pipelines=new Map<string,GPUComputePipeline>();
  private readonly forced=[true,true];
  private readonly grids:readonly [number,number];
  private readonly supportGrids:readonly [number,number];
  private readonly surfaceGrids:readonly [number,number];
  private initPipeline?:GPUComputePipeline;
  private initGroup?:GPUBindGroup;
  constructor(private readonly device:GPUDevice,readonly ownership:UniformMixedOwnership,phi:GPUTexture,private readonly solid?:UniformMixedSolid,private readonly pool?:UniformPagePool,readonly heat?:UniformRegionHeat){
    const tiles=ownership.capacity.tiles;this.regions=tiles/64;
    // Dense closure snapshots are bounded independently of page-pool capacity.
    this.cacheRegions=Math.min(512,this.regions);
    this.mappingWordOffset=3*this.regions+4+this.cacheRegions;
    this.snapshotOffset=this.mappingWordOffset+this.regions;
    this.modelWordOffset=3*this.regions+4;
    this.velocityWordOffset=this.modelWordOffset+8*this.regions;
    this.localVelocityWordOffset=this.velocityWordOffset+6*this.regions;
    this.slopeWordOffset=this.localVelocityWordOffset+6*this.regions;
    this.supportWordOffset=this.slopeWordOffset+3*this.regions;
    this.topologyWordOffset=this.snapshotOffset+132*64*this.cacheRegions;
    this.liveEpochWordOffset=this.supportWordOffset+64*UNIFORM_SURFACE_SUPPORT_WORDS*this.regions;
    this.scratchEpochWordOffset=this.liveEpochWordOffset+this.regions;
    this.modelEpochWordOffset=this.scratchEpochWordOffset+this.regions;
    this.modelReuseWordOffset=this.modelEpochWordOffset+this.regions;
    this.modelWorldWordOffset=this.modelReuseWordOffset+this.regions;
    this.modelMetadataWordOffset=this.modelWorldWordOffset+3*this.regions;
    this.geometryLiveWordOffset=this.topologyWordOffset+this.regions;
    this.geometryEpochWordOffset=this.geometryLiveWordOffset+this.regions;
    this.geometryRemapWordOffset=this.geometryEpochWordOffset+this.regions;
    this.tierWordOffset=this.modelMetadataWordOffset+3*tiles;
    // With heat: a copy slot for the tiers, then one for their non-COLD epochs.
    this.topologyEpochWordOffset=this.tierWordOffset+(heat?2*this.regions:0);
    const surfaceBytes=4*(this.topologyEpochWordOffset+this.regions),geometryBytes=4*(this.geometryRemapWordOffset+this.regions);
    const make=(stage:string,size:number)=>device.createBuffer({label:`Uniform 16h ${stage} work evidence`,size,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
    this.surface=make("surface",surfaceBytes);this.geometry=make("geometry",geometryBytes);
    this.framePlan=new UniformRegionalFramePlan(device,ownership,this.revisions,heat?.tiers,()=>this.generation,{buffer:this.geometry,mapping:this.mappingWordOffset,owners:3*this.regions+4,remap:this.geometryRemapWordOffset,regions:this.cacheRegions},this.pool);
    this.allocatedBytes=surfaceBytes+geometryBytes+this.framePlan.allocatedBytes;
    device.queue.writeBuffer(this.surface,4*this.topologyEpochWordOffset,new Uint32Array(this.regions).fill(1));
    device.queue.writeBuffer(this.surface,4*this.liveEpochWordOffset,new Uint32Array(2*this.regions).fill(1));
    this.layout=uniformDetailBindLayout(device,{entries:[
      {binding:0,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float",viewDimension:"3d"}},
      {binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},
      {binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},
      {binding:4,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},
      {binding:5,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},
      ...(heat?[{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage" as const}}]:[]),
    ]});
    this.groups=[this.surface,this.geometry].map(buffer=>uniformDetailGroup(device,{layout:this.layout,entries:[{binding:5,resource:{buffer:buffer===this.geometry?this.surface:ownership.regionVelocity}},{binding:4,resource:{buffer:this.framePlan.buffer}},{binding:0,resource:phi},{binding:1,resource:{buffer}},{binding:2,resource:{buffer:ownership.regionVelocity}},...(heat?[{binding:3,resource:{buffer:heat.tiers}}]:[])]}));
    const jobs=64*this.cacheRegions,x=Math.min(jobs,ownership.dispatchX);this.grids=[x,Math.ceil(jobs/x)];
    const supportX=Math.min(this.regions,ownership.dispatchX);this.supportGrids=[supportX,Math.ceil(this.regions/supportX)];
    // The model scan: a workgroup per eighth of a region (8 tiles in turn).
    const scanJobs=WC_SCAN_PARTS*this.regions,sx=Math.min(scanJobs,ownership.dispatchX);this.surfaceGrids=[sx,Math.ceil(scanJobs/sx)];
  }
  /** Legacy foot-cache diagnostic ABI; producer revisions use revisions. */
  get epochOffset():undefined{return undefined;}
  /** Header: stored regions, dirty regions, enabled/stats bits, cache misses. */
  get countersOffset():number{return 12*this.regions;}
  invalidate():void{this.forced[0]=this.forced[1]=true;this.revisionGeneration++;}
  /** An intervening writer changed geometry outputs outside this snapshot. */
  get geometryRefreshRequired():boolean{return this.forced[1]||!this.enabled;}
  invalidateGeometry():void{this.forced[1]=true;}
  async initialize():Promise<void>{
    await this.framePlan.initialize();
    const module=uniformDetailModule(this.device,{label:"Uniform 16h surface work control",code:uniformMixedTopologyWGSL(this.ownership.capacity,0)+/* wgsl */`
@group(1) @binding(0) var phi:texture_3d<f32>;
@group(1) @binding(1) var<storage,read_write> wc:array<atomic<u32>>;
@group(1) @binding(2) var<storage,read> wcVelocity:array<u32>;
@group(1) @binding(5) var<storage,read> wcRevisions:array<u32>;
${this.framePlan.wgsl(1,4)}
var<workgroup> wcRegionalCount:u32;
var<workgroup> wcRegionalRegion:u32;
fn wcRegionCount(channel:u32,lane:u32)->u32{if(lane==0u){wcRegionalCount=regionalCount(channel);}return workgroupUniformLoad(&wcRegionalCount);}
fn wcRegionAt(channel:u32,slot:u32,lane:u32)->u32{workgroupBarrier();if(lane==0u){wcRegionalRegion=regionalAt(channel,slot);}return workgroupUniformLoad(&wcRegionalRegion);}
${this.heat?uniformRegionHeatWGSL(1,3,this.heat.regions):"const UM_TIER_COLD:u32=0u;fn umRegionTier(region:u32)->u32{return 2u;}"}
${uniformMixedSolidWGSL(this.solid?2:undefined,this.solid?.coarse?.count)}
override wcForce:bool=false;
override wcMetadataOnly:bool=false;
override wcSurface:bool=false;
override wcEnabled:bool=true;
override wcGeometryStats:bool=false;
const WC_CACHE:u32=${this.cacheRegions}u;
const WC_OWNERS:u32=3u*UM_PAGES+4u;
const WC_MAPPING:u32=${this.mappingWordOffset}u;
const WC_SNAPSHOT:u32=${this.snapshotOffset}u;
const WC_MODEL:u32=${this.modelWordOffset}u;
const WC_VELOCITY:u32=${this.velocityWordOffset}u;
const WC_LOCAL_VELOCITY:u32=${this.localVelocityWordOffset}u;
const WC_SLOPE:u32=${this.slopeWordOffset}u;
const WC_SUPPORT:u32=${this.supportWordOffset}u;
const WC_LIVE:u32=${this.liveEpochWordOffset}u;
const WC_MODEL_EPOCH:u32=${this.modelEpochWordOffset}u;
const WC_MODEL_REUSE:u32=${this.modelReuseWordOffset}u;
const WC_MODEL_WORLD:u32=${this.modelWorldWordOffset}u;
const WC_MODEL_META:u32=${this.modelMetadataWordOffset}u;
const WC_GEOMETRY_LIVE:u32=${this.geometryLiveWordOffset}u;
const WC_GEOMETRY_EPOCH:u32=${this.geometryEpochWordOffset}u;
const WC_GEOMETRY_REMAP:u32=${this.geometryRemapWordOffset}u;
const WC_TOPOLOGY:u32=${this.topologyWordOffset}u;
fn wcFit(region:u32){
 let at=WC_MODEL+8u*region;let origin=16u*umPageCoord(region);
 var c=0.0;var gradient=vec3f(0);var valid=true;
 for(var k=0u;k<8u;k++){
  let v=${UNIFORM_DETAIL_4H_LOAD}textureLoad(phi,vec3i(origin+16u*umCorner(k,2u)),0).x;
  valid=valid&&abs(v)<=3.402823e38;c+=0.125*v;
  gradient+=v*(2.0*vec3f(umCorner(k,2u))-vec3f(1))/64.0;
 }
 valid=valid&&abs(c)<=3.402823e38&&all(abs(gradient)<=vec3f(3.402823e38));
 atomicStore(&wc[at],bitcast<u32>(gradient.x));atomicStore(&wc[at+1u],bitcast<u32>(gradient.y));atomicStore(&wc[at+2u],bitcast<u32>(gradient.z));atomicStore(&wc[at+3u],bitcast<u32>(c));
 atomicStore(&wc[at+6u],0u);atomicStore(&wc[at+7u],u32(valid));
 for(var axis=0u;axis<3u;axis++){atomicStore(&wc[WC_SLOPE+3u*region+axis],0u);}
}
fn wcDecode(word:u32)->f32{return bitcast<f32>(select(~word,word^0x80000000u,(word&0x80000000u)!=0u));}
@compute @workgroup_size(64) fn selectRegions(@builtin(global_invocation_id) gid:vec3u){
 let slot=gid.x;if(slot>=WC_CACHE){return;}
 var owner=0xffffffffu;
 // Stable compact ranks use physical block order, independent of allocator
 // residue/modulo collisions. Fixed slots scan only the small block table.
 var rank=0u;let wanted=slot/8u;
 for(var block=1u;block<UM_TILES/512u-1u;block++){
  if(umBlockWord(block)==0xffffffffu){continue;}
  if(rank==wanted){owner=8u*block+(slot&7u);break;}rank++;
 }
 if(owner!=0xffffffffu){atomicStore(&wc[WC_MAPPING+owner],slot+1u);}

 let previous=atomicLoad(&wc[WC_OWNERS+slot]);
 atomicStore(&wc[WC_OWNERS+slot],owner);
 if(owner!=0xffffffffu&&previous!=owner){atomicStore(&wc[owner],1u);atomicStore(&wc[WC_TOPOLOGY+owner],1u);}
}
// One16h region per workgroup: compact metadata and current solid masks
// are reduced cooperatively before any retained-model phi gather.
var<workgroup> wcSelectLive:u32;
var<workgroup> wcSelectChanged:atomic<u32>;
var<workgroup> wcSelectClosed:atomic<u32>;
@compute @workgroup_size(64) fn selectSurface(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 let count=wcRegionCount(3u,lane);
 for(var job=group.x+groups.x*group.y;job<count*1u;job+=groups.x*groups.y){
 let region=wcRegionAt(3u,job/1u,lane);

 if(lane==0u){
  let block=region>>3u;wcSelectLive=u32(umBlockStored(block)&&umBlockWord(block)!=0xffffffffu);
  atomicStore(&wcSelectChanged,0u);atomicStore(&wcSelectClosed,0u);
 }
 if(workgroupUniformLoad(&wcSelectLive)==0u){
  if(lane==0u){atomicStore(&wc[region],0u);atomicStore(&wc[WC_MODEL_REUSE+region],0u);atomicStore(&wc[WC_MODEL+8u*region+5u],0u);atomicStore(&wc[WC_MODEL+8u*region+7u],0u);atomicStore(&wc[WC_MODEL_EPOCH+region],0u);}continue;
 }
 // A COLD region at the same world with a valid model holds its model,
 // metadata, closed flag and support (summaries skips it too). Its first
 // non-COLD frame compares the metadata stored before it went COLD, so any
 // change since refits it.
 if(lane==0u){
  let world=16u*umPageCoord(region);let m=WC_MODEL_WORLD+3u*region;
  let hold=!wcForce&&umRegionTier(region)==UM_TIER_COLD&&atomicLoad(&wc[WC_MODEL+8u*region+7u])!=0u&&atomicLoad(&wc[WC_MODEL+8u*region+5u])!=0u
   &&atomicLoad(&wc[m])==world.x&&atomicLoad(&wc[m+1u])==world.y&&atomicLoad(&wc[m+2u])==world.z;
  wcSelectLive=select(1u,2u,hold);
 }
 if(workgroupUniformLoad(&wcSelectLive)==2u){if(lane==0u){atomicStore(&wc[WC_MODEL_REUSE+region],1u);}continue;}
 if(lane==0u){
  let epoch=atomicLoad(&wc[WC_LIVE+region]);
  // A COLD region's phi is held (UniformRegionHeat): its model stands
  // whatever its revision says. Elsewhere the exact revision decides.
  let cold=umRegionTier(region)==UM_TIER_COLD;
  var same=!wcForce&&atomicLoad(&wc[WC_MODEL+8u*region+7u])!=0u&&(cold||(epoch>0u&&epoch<0xfffffffeu&&epoch==atomicLoad(&wc[WC_MODEL_EPOCH+region])));
  let world=16u*umPageCoord(region);
  for(var axis=0u;axis<3u;axis++){let at=WC_MODEL_WORLD+3u*region+axis;same=same&&atomicLoad(&wc[at])==world[axis];atomicStore(&wc[at],world[axis]);}
  if(!same){atomicOr(&wcSelectChanged,1u);}
 }
 let tile=umPageTile(region,lane);let at=WC_MODEL_META+3u*tile;
 if(!umTileLive(tile)){atomicOr(&wcSelectChanged,1u);}
 else{
  let stencil=umTileStencil(tile);let values=vec3u(umTopology[tile],stencil);
  for(var k=0u;k<3u;k++){if(atomicLoad(&wc[at+k])!=values[k]){atomicOr(&wcSelectChanged,1u);}atomicStore(&wc[at+k],values[k]);}
  ${this.solid?this.solid.coarse?`if(umSolidPresent()&&any(umSolidOpenCells(tile)!=vec2u(0xffffffffu))){atomicOr(&wcSelectClosed,1u);}`:`if(umSolidPresent()){atomicOr(&wcSelectClosed,1u);}`:""}
 }
 workgroupBarrier();
 if(lane==0u){
  let same=atomicLoad(&wcSelectChanged)==0u;
  atomicStore(&wc[region],atomicLoad(&wcSelectClosed));atomicStore(&wc[WC_MODEL+8u*region+5u],0u);
  atomicStore(&wc[WC_MODEL_REUSE+region],u32(same));if(!same){wcFit(region);}
 }

 workgroupBarrier();
 }
}
var<workgroup> wcChanged:atomic<u32>;
var<workgroup> wcOwner:u32;
var<workgroup> wcSkip:u32;
var<workgroup> wcInputs:array<u32,125>;
var<workgroup> wcPlane:vec4f;
var<workgroup> wcResidual:atomic<u32>;
var<workgroup> wcSlope:array<atomic<u32>,3>;
@compute @workgroup_size(64) fn scan(@builtin(workgroup_id) g:vec3u,@builtin(local_invocation_index) lane:u32){
 let job=g.x+umDispatchX*g.y;
 if(job>=64u*WC_CACHE){return;}
 if(lane==0u){wcOwner=atomicLoad(&wc[WC_OWNERS+job/64u]);}
 let region=workgroupUniformLoad(&wcOwner);if(region==0xffffffffu){return;}
 let tile=umPageTile(region,job%64u);if(!umTileLive(tile)){return;}
 let at=WC_SNAPSHOT+132u*job;
 let base=4u*umTileCoord(tile);
 if(lane==0u){
  let stencil=umTileStencil(tile);
  let metadata=array<u32,7>(umTopology[tile],stencil.x,stencil.y,base.x,base.y,base.z,1u);
  var different=wcForce||atomicLoad(&wc[region])!=0u||atomicLoad(&wc[WC_GEOMETRY_REMAP+region])!=0u;var topology=wcForce||atomicLoad(&wc[WC_GEOMETRY_REMAP+region])!=0u;
  for(var k=0u;k<7u;k++){let value=metadata[k];let changed=atomicLoad(&wc[at+125u+k])!=value;different=different||changed;topology=topology||changed;atomicStore(&wc[at+125u+k],value);}
  if(topology){atomicStore(&wc[WC_TOPOLOGY+region],1u);}atomicStore(&wcChanged,u32(different));
  let epoch=atomicLoad(&wc[WC_GEOMETRY_LIVE+region]);
  wcSkip=u32(!different&&(umRegionTier(region)==UM_TIER_COLD||(epoch>0u&&epoch<0xfffffffeu&&epoch==atomicLoad(&wc[WC_GEOMETRY_EPOCH+region]))));
 }
 if(workgroupUniformLoad(&wcSkip)!=0u){return;}
 let width=umTileMinimumWidth(tile);let dense=width<4u;
 let side=4u/width+1u;let count=select(8u,side*side*side,dense);
 for(var k=lane;k<count;k+=64u){
  var value=0.0;
  if(dense){value=textureLoad(phi,vec3i(base+width*umCorner(k,side)),0).x;}
  else{value=${UNIFORM_DETAIL_4H_LOAD}textureLoad(phi,vec3i(base+4u*umCorner(k,2u)),0).x;}
  wcInputs[k]=bitcast<u32>(value);
  let bits=bitcast<u32>(value);
  let previous=bitcast<f32>(atomicLoad(&wc[at+k]));
  if(bits!=bitcast<u32>(previous)||!(abs(value)<=3.402823e38&&abs(previous)<=3.402823e38)){atomicStore(&wcChanged,1u);}
 }
 workgroupBarrier();
 // Only geometry retains exact tile snapshots, committed collectively.
 if(atomicLoad(&wcChanged)!=0u){
  for(var k=lane;k<count;k+=64u){atomicStore(&wc[at+k],wcInputs[k]);}
  if(lane==0u){atomicStore(&wc[umPageOfTile(tile)],1u);}
 }
}
// Once the frame is sealed the cache mapping cannot change. One group
// owns each selected region: lanes compare its tiles, then publish its
// certificate together. This removes mapping selection, revision copies,
// global clears and a separate region finish dispatch from the active path.
var<workgroup> wcRegionalChanges:atomic<u32>;
@compute @workgroup_size(64) fn geometryRegional(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 let count=wcRegionCount(3u,lane);
 if(group.x==0u&&lane==0u){atomicStore(&wc[3u*UM_PAGES+2u],u32(wcEnabled)+2u*u32(wcGeometryStats));}
 for(var job=group.x;job<count;job+=groups.x){
  let region=wcRegionAt(3u,job,lane);
  let mapped=atomicLoad(&wc[WC_MAPPING+region]);let miss=mapped==0u;
  let remapped=atomicLoad(&wc[WC_GEOMETRY_REMAP+region])!=0u;
  let epoch=wcRevisions[WC_LIVE+region];
  if(lane==0u){atomicStore(&wcRegionalChanges,select(0u,3u,miss||remapped));}workgroupBarrier();
  // Producer topology revisions include page identity and the remap halo.
  // A held region needs no tile metadata scan or texture access at all.
  let topologyPossible=regionalHas(4u,region)||remapped;
  let regionHeld=!topologyPossible&&(umRegionTier(region)==UM_TIER_COLD||(epoch>0u&&epoch<0xfffffffeu&&epoch==atomicLoad(&wc[WC_GEOMETRY_EPOCH+region])));
  if(!miss&&!regionHeld){
   let tile=umPageTile(region,lane);let base=4u*umTileCoord(tile);
   let at=WC_SNAPSHOT+132u*(64u*(mapped-1u)+lane);let stencil=umTileStencil(tile);
   let metadata=array<u32,7>(umTopology[tile],stencil.x,stencil.y,base.x,base.y,base.z,1u);
   var topology=remapped;
   if(topologyPossible){for(var k=0u;k<7u;k++){let value=metadata[k];topology=topology||atomicLoad(&wc[at+125u+k])!=value;atomicStore(&wc[at+125u+k],value);}}
   var dirty=topology;
   let held=!topology&&(umRegionTier(region)==UM_TIER_COLD||(epoch>0u&&epoch<0xfffffffeu&&epoch==atomicLoad(&wc[WC_GEOMETRY_EPOCH+region])));
   if(!held){
    let width=umTileMinimumWidth(tile);let dense=width<4u;let side=4u/width+1u;let vertices=select(8u,side*side*side,dense);
    for(var k=0u;k<vertices;k++){
     var value=0.0;
     if(dense){value=textureLoad(phi,vec3i(base+width*umCorner(k,side)),0).x;}
     else{value=${UNIFORM_DETAIL_4H_LOAD}textureLoad(phi,vec3i(base+4u*umCorner(k,2u)),0).x;}
     let bits=bitcast<u32>(value);let previous=atomicLoad(&wc[at+k]);
     dirty=dirty||bits!=previous||!(abs(value)<=3.402823e38&&abs(bitcast<f32>(previous))<=3.402823e38);
     atomicStore(&wc[at+k],bits);
    }
   }
   atomicOr(&wcRegionalChanges,u32(dirty)|(2u*u32(topology)));
  }
  workgroupBarrier();
  if(lane==0u){
   let body=${this.solid?`umBodyEvidencePending()&&umBodySweptReach(region)<=${GEOMETRY_BODY_REACH}u`:"false"};
   let changes=atomicLoad(&wcRegionalChanges)|select(0u,3u,body);let dirty=(changes&1u)!=0u;
   atomicStore(&wc[region],u32(dirty));atomicStore(&wc[WC_TOPOLOGY+region],u32((changes&2u)!=0u));
   atomicStore(&wc[WC_GEOMETRY_REMAP+region],0u);
   if(umRegionTier(region)!=UM_TIER_COLD){atomicStore(&wc[WC_GEOMETRY_EPOCH+region],epoch);}
   let before=atomicLoad(&wc[UM_PAGES+region]);atomicStore(&wc[UM_PAGES+region],min(0x1fffffffu,before+u32(dirty||before==0u)));
   atomicAdd(&wc[3u*UM_PAGES],1u);if(miss){atomicAdd(&wc[3u*UM_PAGES+3u],1u);}if(dirty){atomicAdd(&wc[3u*UM_PAGES+1u],1u);}
  }
  workgroupBarrier();
 }
}
// An eighth of a region shares one plane. Stage its eight tile lattices
// together, then reduce directly across the whole job. Maxima associate
// exactly; no tile-local reduction or per-tile reset is needed.
var<workgroup> wcSurfaceInputs:array<u32,${125*64/WC_SCAN_PARTS}>;
fn wcSurfacePart(region:u32,first:u32,lane:u32){
 let m=WC_MODEL+8u*region;
 let plane=bitcast<vec4f>(vec4u(atomicLoad(&wc[m]),atomicLoad(&wc[m+1u]),atomicLoad(&wc[m+2u]),atomicLoad(&wc[m+3u])));
 let centre=vec3f(16u*umPageCoord(region)+vec3u(8u));var residual=0.0;
 for(var part=0u;part<${64/WC_SCAN_PARTS}u;part++){
  let tile=umPageTile(region,first+part);if(tile>=UM_TILES||!umTileLive(tile)){continue;}
  let base=4u*umTileCoord(tile);let width=umTileMinimumWidth(tile);let dense=width<4u;
  let side=4u/width+1u;let count=select(8u,side*side*side,dense);
  for(var k=lane;k<count;k+=64u){
   var value=0.0;
   if(dense){value=textureLoad(phi,vec3i(base+width*umCorner(k,side)),0).x;}
   else{value=${UNIFORM_DETAIL_4H_LOAD}textureLoad(phi,vec3i(base+4u*umCorner(k,2u)),0).x;}
   wcSurfaceInputs[125u*part+k]=bitcast<u32>(value);
   let point=base+width*umCorner(k,side);var error=abs(value-(plane.w+dot(plane.xyz,vec3f(point)-centre)));
   if(!(abs(value)<=3.402823e38&&error<=3.402823e38)){error=3.402823e38;}
   residual=max(residual,error);
  }
 }
 workgroupBarrier();var slopes=vec3f(0);
 for(var part=0u;part<${64/WC_SCAN_PARTS}u;part++){
  let tile=umPageTile(region,first+part);if(tile>=UM_TILES||!umTileLive(tile)){continue;}
  let width=umTileMinimumWidth(tile);let dense=width<4u;let side=4u/width+1u;
  let count=select(8u,side*side*side,dense);let n=select(2u,side,dense);let step=select(4.0,f32(width),dense);
  for(var k=lane;k<count;k+=64u){
   let q=umCorner(k,n);let a=bitcast<f32>(wcSurfaceInputs[125u*part+k]);
   for(var axis=0u;axis<3u;axis++){
    if(q[axis]+1u>=n){continue;}
    let stride=select(select(1u,n,axis==1u),n*n,axis==2u);let b=bitcast<f32>(wcSurfaceInputs[125u*part+k+stride]);
    var bound=(abs(b-a)+4.0*1.1920929e-7*(abs(a)+abs(b)))/step;
    if(!(abs(a)<=3.402823e38&&abs(b)<=3.402823e38&&bound<=3.402823e38)){bound=3.402823e38;}
    slopes[axis]=max(slopes[axis],bound);
   }
  }
 }
 atomicMax(&wc[WC_MODEL+8u*region+6u],bitcast<u32>(residual));
 for(var axis=0u;axis<3u;axis++){atomicMax(&wc[WC_SLOPE+3u*region+axis],bitcast<u32>(slopes[axis]));}
}
// The surface model scan: a workgroup per eighth of a region. Its tiles
// share one staged lattice; a reused region's workgroups exit at once.
@compute @workgroup_size(64) fn surfaceScan(@builtin(workgroup_id) g:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 let count=wcRegionCount(3u,lane);
 for(var job=g.x+groups.x*g.y;job<count*8u;job+=groups.x*groups.y){
 let region=wcRegionAt(3u,job/8u,lane);

 if(lane==0u){wcSkip=atomicLoad(&wc[WC_MODEL_REUSE+region]);}if(workgroupUniformLoad(&wcSkip)!=0u){continue;}
 wcSurfacePart(region,${64/WC_SCAN_PARTS}u*(job%${WC_SCAN_PARTS}u),lane);

 workgroupBarrier();
 }
}
// Metadata-only ablation: one lane per tile, no phi reads or collective
// reduction. Region identity/page claims still invalidate retained normals.
@compute @workgroup_size(64) fn metadata(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 let slot=group.x;if(slot>=WC_CACHE){return;}let region=atomicLoad(&wc[WC_OWNERS+slot]);if(region==0xffffffffu){return;}
 let tile=umPageTile(region,lane);if(!umTileLive(tile)){return;}let base=4u*umTileCoord(tile);let stencil=umTileStencil(tile);
 let values=array<u32,7>(umTopology[tile],stencil.x,stencil.y,base.x,base.y,base.z,1u);let at=WC_SNAPSHOT+132u*(64u*slot+lane)+125u;
 var topology=wcForce||atomicLoad(&wc[WC_GEOMETRY_REMAP+region])!=0u;
 for(var k=0u;k<7u;k++){let value=values[k];topology=topology||atomicLoad(&wc[at+k])!=value;atomicStore(&wc[at+k],value);}
 if(topology){atomicStore(&wc[WC_TOPOLOGY+region],1u);}
}
var<workgroup> wcSupportHome:vec4f;
var<workgroup> wcSupportLive:u32;
// xyz local velocity bounds; w transformed rho / validity respectively.
var<workgroup> wcSupportLow:array<vec4f,27>;
var<workgroup> wcSupportHigh:array<vec4f,27>;
var<workgroup> wcSupportSlope:array<vec4f,27>;
fn wcSupportContains(code:u32,offset:i32)->bool{
 return offset==0||(offset<0&&(code&1u)!=0u)||(offset>0&&(code&2u)!=0u);
}
@compute @workgroup_size(64) fn summaries(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 let count=wcRegionCount(6u,lane);
 if(all(group==vec3u(0))&&lane==0u){atomicStore(&wc[3u*UM_PAGES+2u],u32(wcEnabled));}
 for(var job=group.x+groups.x*group.y;job<count*1u;job+=groups.x*groups.y){
 let region=wcRegionAt(6u,job/1u,lane);

 // A region selectSurface held (COLD, model reused) keeps its support.
 if(lane==0u){wcSupportLive=u32(umRegionTier(region)==UM_TIER_COLD&&atomicLoad(&wc[WC_MODEL_REUSE+region])!=0u);if(region==0u){atomicStore(&wc[3u*UM_PAGES+2u],u32(wcEnabled));}}
 if(workgroupUniformLoad(&wcSupportLive)!=0u){if(lane==0u){atomicAdd(&wc[3u*UM_PAGES],1u);}continue;}
 let output=WC_SUPPORT+${UNIFORM_SURFACE_SUPPORT_WORDS}u*(64u*region+lane);atomicStore(&wc[output+7u],0u);
 if(lane==0u){
  // A COLD region's model keeps the revision it was fit at: the first
  // non-COLD frame compares against that.
  if(umRegionTier(region)!=UM_TIER_COLD){atomicStore(&wc[WC_MODEL_EPOCH+region],atomicLoad(&wc[WC_LIVE+region]));}
  if(region==0u){atomicStore(&wc[3u*UM_PAGES+2u],u32(wcEnabled));}
  wcSupportLive=u32(umBlockStored(region>>3u)&&umBlockWord(region>>3u)!=0xffffffffu);
  let home=WC_MODEL+8u*region;wcSupportHome=bitcast<vec4f>(vec4u(atomicLoad(&wc[home]),atomicLoad(&wc[home+1u]),atomicLoad(&wc[home+2u]),atomicLoad(&wc[home+3u])));
 }
 if(workgroupUniformLoad(&wcSupportLive)==0u){continue;}
 let origin=16u*umPageCoord(region);let g=wcSupportHome.xyz;let c=wcSupportHome.w;
 if(lane<27u){
  let delta=vec3i(i32(lane%3u)-1,i32((lane/3u)%3u)-1,i32(lane/9u)-1);
  let tile=umTileKey(vec3i(origin/4u)+4*delta);var valid=wcEnabled&&atomicLoad(&wc[WC_MODEL+8u*region+7u])!=0u&&umKeyStored(tile);
  var rho=0.0;var lo=vec3f(0);var hi=vec3f(0);var slope=vec3f(0);var clear=false;
  if(valid){
   let neighbor=umPageOfTile(tile);let at=WC_MODEL+8u*neighbor;valid=atomicLoad(&wc[at+7u])!=0u;clear=atomicLoad(&wc[neighbor])==0u;
   let ng=bitcast<vec3f>(vec3u(atomicLoad(&wc[at]),atomicLoad(&wc[at+1u]),atomicLoad(&wc[at+2u])));
   let nc=bitcast<f32>(atomicLoad(&wc[at+3u]));let nr=bitcast<f32>(atomicLoad(&wc[at+6u]));let translate=dot(g,16.0*vec3f(delta));
   // Cover both association orders used by the legacy broad/subset paths.
   rho=nr+max(abs(nc-c-translate),abs(nc-(c+translate)))+8.0*dot(abs(ng-g),vec3f(1));
   valid=valid&&nr>=0.0&&nr<3.402823e38&&rho>=0.0&&rho<3.402823e38;
   for(var axis=0u;axis<3u;axis++){
    slope[axis]=bitcast<f32>(atomicLoad(&wc[WC_SLOPE+3u*neighbor+axis]));
    let upper=wcVelocity[6u*neighbor+axis];let lower=wcVelocity[6u*neighbor+3u+axis];
    valid=valid&&upper!=0u&&lower!=0u&&upper!=0xffffffffu&&lower!=0xffffffffu;
    hi[axis]=wcDecode(upper);lo[axis]=-wcDecode(lower);
   }
   valid=valid&&all(abs(lo)<=vec3f(3.402823e38))&&all(abs(hi)<=vec3f(3.402823e38))&&all(lo<=hi)&&all(slope>=vec3f(0))&&all(slope<vec3f(3.402823e38));
  }
  wcSupportLow[lane]=vec4f(lo,rho);wcSupportHigh[lane]=vec4f(hi,select(0.0,1.0,valid));wcSupportSlope[lane]=vec4f(slope,select(0.0,1.0,clear));
 }
 workgroupBarrier();
 var rho=0.0;var lo=vec3f(3.402823e38);var hi=vec3f(-3.402823e38);var slope=vec3f(0);var valid=wcEnabled;var clear=true;var coverage=0u;
 let code=vec3u(lane&3u,(lane>>2u)&3u,(lane>>4u)&3u);
 // Lane63 additionally supplies the broad union of usable neighbors, even
 // when missing boundary neighbors make its strict rectangle invalid.
 for(var k=0u;k<27u;k++){
  let delta=vec3i(i32(k%3u)-1,i32((k/3u)%3u)-1,i32(k/9u)-1);
  if(!wcSupportContains(code.x,delta.x)||!wcSupportContains(code.y,delta.y)||!wcSupportContains(code.z,delta.z)){continue;}
  let usable=wcSupportHigh[k].w!=0.0;valid=valid&&usable;clear=clear&&wcSupportSlope[k].w!=0.0;
  if(usable){coverage|=1u<<k;rho=max(rho,wcSupportLow[k].w);lo=min(lo,wcSupportLow[k].xyz);hi=max(hi,wcSupportHigh[k].xyz);slope=max(slope,wcSupportSlope[k].xyz);}
 }
 rho+=32.0*1.1920929e-7*(abs(c)+32.0*dot(abs(g),vec3f(1))+rho);
 let padding=64.0*1.1920929e-7*max(abs(lo),abs(hi));lo-=padding;hi+=padding;
 valid=valid&&rho<=3.402823e38&&all(abs(lo)<=vec3f(3.402823e38))&&all(abs(hi)<=vec3f(3.402823e38))&&all(lo<=hi);
 atomicStore(&wc[output],bitcast<u32>(rho));for(var axis=0u;axis<3u;axis++){atomicStore(&wc[output+1u+axis],bitcast<u32>(lo[axis]));atomicStore(&wc[output+4u+axis],bitcast<u32>(hi[axis]));}atomicStore(&wc[output+7u],select(0u,1u+2u*u32(clear),valid));
 for(var axis=0u;axis<3u;axis++){atomicStore(&wc[output+8u+axis],bitcast<u32>(slope[axis]));}
 if(lane==63u){
  let home=WC_MODEL+8u*region;if(!(rho<=3.402823e38)||(coverage&(1u<<13u))==0u){coverage=0u;}
  atomicStore(&wc[home+4u],bitcast<u32>(rho));atomicStore(&wc[home+5u],coverage);
  for(var axis=0u;axis<3u;axis++){atomicStore(&wc[WC_VELOCITY+6u*region+axis],bitcast<u32>(lo[axis]));atomicStore(&wc[WC_VELOCITY+6u*region+3u+axis],bitcast<u32>(hi[axis]));
   let upper=wcVelocity[6u*region+axis];let lower=wcVelocity[6u*region+3u+axis];let finite=upper!=0u&&lower!=0u&&upper!=0xffffffffu&&lower!=0xffffffffu;
   atomicStore(&wc[WC_LOCAL_VELOCITY+6u*region+axis],bitcast<u32>(select(-3.402823e38,-wcDecode(lower),finite)));atomicStore(&wc[WC_LOCAL_VELOCITY+6u*region+3u+axis],bitcast<u32>(select(3.402823e38,wcDecode(upper),finite)));
  }
  atomicAdd(&wc[3u*UM_PAGES],1u);
 }

 workgroupBarrier();
 }
}
// Separate launch: every scan's dirty word is complete before it is read.
@compute @workgroup_size(64) fn finish(@builtin(global_invocation_id) gid:vec3u){
 let region=gid.x;if(region>=UM_PAGES){return;}
 if(region==0u){atomicStore(&wc[3u*UM_PAGES+2u],u32(wcEnabled)+2u*u32(wcGeometryStats));}
 let block=region>>3u;if(!umBlockStored(block)||umBlockWord(block)==0xffffffffu){return;}
 let miss=atomicLoad(&wc[WC_MAPPING+region])==0u;
 // A COLD region's snapshot keeps the revision it was taken at.
 if(umRegionTier(region)!=UM_TIER_COLD){atomicStore(&wc[WC_GEOMETRY_EPOCH+region],atomicLoad(&wc[WC_GEOMETRY_LIVE+region]));}
 let remapped=atomicLoad(&wc[WC_GEOMETRY_REMAP+region])!=0u;
 var dirty=wcMetadataOnly||miss||atomicLoad(&wc[region])!=0u||remapped;
 // Swept-body evidence: current until the first geometry launch after the
 // head takes it; this region's retained geometry then holds the current pose.
 let body=${this.solid?`umBodyEvidencePending()&&umBodySweptReach(region)<=${GEOMETRY_BODY_REACH}u`:"false"};
 dirty=dirty||body;
 let epoch=atomicLoad(&wc[UM_PAGES+region]);
 // Saturation disables cache reads rather than ever reusing an old epoch.
 atomicStore(&wc[UM_PAGES+region],min(0x1fffffffu,epoch+u32(dirty||epoch==0u)));
 atomicAdd(&wc[3u*UM_PAGES],1u);if(miss){atomicAdd(&wc[3u*UM_PAGES+3u],1u);}if(dirty){atomicAdd(&wc[3u*UM_PAGES+1u],1u);}
 // Geometry reads this plane. Surface must not overwrite it here: neighbors
 // in this dispatch still read the scan's undilated dirty evidence.
 if(!wcSurface){atomicStore(&wc[region],u32(dirty));if(miss||body||remapped){atomicStore(&wc[WC_TOPOLOGY+region],1u);}atomicStore(&wc[WC_GEOMETRY_REMAP+region],0u);}
}
`});
    const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");if(errors.length)throw new Error(errors.map(m=>`${m.lineNum}: ${m.message}`).join("\n"));
    const layout=this.device.createPipelineLayout({bindGroupLayouts:[this.ownership.bindLayout,this.layout,...(this.solid?[this.solid.tileLayout]:[])]});
    for(const [name,entryPoint,constants] of [
      ["metadata","metadata",{}],["metadataForce","metadata",{wcForce:1}],["geometryCurrent","finish",{wcMetadataOnly:1}],["geometryCurrentStats","finish",{wcMetadataOnly:1,wcGeometryStats:1}],["geometryCurrentDisabled","finish",{wcMetadataOnly:1,wcEnabled:0}],
      ["select","selectRegions",{}],["selectSurface","selectSurface",{wcSurface:1}],["selectSurfaceForce","selectSurface",{wcSurface:1,wcForce:1}],["scan","scan",{}],["force","scan",{wcForce:1}],["modelScan","surfaceScan",{wcSurface:1}],["surface","summaries",{wcSurface:1}],["surfaceDisabled","summaries",{wcSurface:1,wcEnabled:0}],["geometry","finish",{}],["geometryStats","finish",{wcGeometryStats:1}],["geometryDisabled","finish",{wcEnabled:0}],
    ] as const)this.pipelines.set(name,await uniformDetailPipeline(this.device,this.ownership,{layout,compute:{module,entryPoint,constants:{umDispatchX:entryPoint==="summaries"||entryPoint==="selectSurface"?this.supportGrids[0]:name==="modelScan"?this.surfaceGrids[0]:this.grids[0],...constants}}}));
    for(const [name,entryPoint] of [["selectSurfaceListed","selectSurface"],["modelScanListed","surfaceScan"],["surfaceListed","summaries"]]){
      this.pipelines.set(name!,await uniformDetailPipeline(this.device,this.ownership,{layout,compute:{module,entryPoint:entryPoint!,constants:{wcSurface:1,regionalListed:1}}}));
    }
    for(const stats of [false,true])this.pipelines.set(stats?"geometryRegionalStats":"geometryRegional",await uniformDetailPipeline(this.device,this.ownership,{layout,compute:{module,entryPoint:"geometryRegional",constants:{regionalListed:1,wcGeometryStats:+stats}}}));
    const remap=this.device.createShaderModule({label:"Uniform16h remap producer revisions",code:uniformMixedTopologyWGSL(this.ownership.capacity,0)+`
@group(1) @binding(0) var<storage,read> changes:array<u32>;
@group(1) @binding(1) var<storage,read_write> surface:array<atomic<u32>>;
@group(1) @binding(2) var<storage,read_write> geometry:array<atomic<u32>>;
${this.heat?this.heat.promoteWGSL(1,3,4):"fn rhPromoteWarm(region:u32){}"}
${this.pool?uniformLiveListsWGSL(1,5,this.pool.layout):""}
var<workgroup> changed:atomic<u32>;
// A region only a stored tile marks: with a pool, a fixed grid of
// workgroups strides the resident blocks' regions (uniform-live-lists.ts).
@compute @workgroup_size(64) fn publish(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 ${this.pool?"for(var visit=group.x;visit<8u*ulResidentCount();visit+=groups.x){let region=8u*ulResidentBlock(visit>>3u)+(visit&7u);":"let region=group.x+umDispatchX*group.y;if(region>=UM_PAGES){return;}{"}
 if(lane==0u){atomicStore(&changed,0u);}workgroupBarrier();
 let tile=umPageTile(region,lane);
 if(umTileLive(tile)&&(changes[76u+tile]&16u)!=0u){atomicOr(&changed,1u);}workgroupBarrier();
 if(lane==0u&&atomicLoad(&changed)!=0u){
  let at=${this.liveEpochWordOffset}u+region;let before=atomicLoad(&surface[at]);atomicStore(&surface[at],select(min(before,0xfffffffeu)+1u,0xfffffffeu,before>=0xfffffffdu));
  let topology=${this.topologyEpochWordOffset}u+region;atomicStore(&surface[topology],min(atomicLoad(&surface[topology]),0xfffffffdu)+1u);
  atomicStore(&surface[${this.modelEpochWordOffset}u+region],0u);atomicStore(&geometry[${this.geometryRemapWordOffset}u+region],1u);
  // A width change re-resolves the region's phi: it runs at least WARM
  // from here (UniformRegionHeat). The listed tiles include their halo, so
  // the neighbours a changed tile reaches are listed regions themselves.
  rhPromoteWarm(region);
 }
 }
}`});
    this.remapLayout=this.device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},...(this.heat?[1,2,3,4]:[1,2]).map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage" as const}})),...(this.pool?[{binding:5,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage" as const}}]:[])]});
    this.remapPipeline=await this.device.createComputePipelineAsync({layout:this.device.createPipelineLayout({bindGroupLayouts:[this.ownership.bindLayout,this.remapLayout]}),compute:{module:remap,entryPoint:"publish",constants:{umDispatchX:this.supportGrids[0]}}});
    if(this.pool){
      const init=this.device.createShaderModule({label:"Uniform 16h work evidence page claims",code:`
${uniformPagePoolWGSL(this.pool.layout,0,0)}
@group(0) @binding(1) var<storage,read_write> geometry:array<atomic<u32>>;
@group(0) @binding(2) var<storage,read_write> surface:array<atomic<u32>>;
${this.heat?this.heat.promoteWGSL(0,3,4):"fn rhPromote(region:u32){}"}
${uniformPageInitBlocksWGSL(this.pool.layout,"initialize",512,`let region=8u*block+((local>>2u)&1u)+2u*((local>>5u)&1u)+4u*((local>>8u)&1u);atomicStore(&surface[${this.modelEpochWordOffset}u+region],0u);let mapped=atomicLoad(&geometry[${this.mappingWordOffset}u+region]);if(mapped!=0u){let lane=(local&3u)|((local>>1u)&12u)|((local>>2u)&48u);let at=${this.snapshotOffset}u+132u*(64u*(mapped-1u)+lane)+131u;atomicStore(&geometry[at],0u);}
if(local<8u){let home=2*upPageOf(block)+vec3i(i32(local&1u),i32((local>>1u)&1u),i32(local>>2u));
 for(var z=-1;z<=1;z++){for(var y=-1;y<=1;y++){for(var x=-1;x<=1;x++){
  let q=home+vec3i(x,y,z);let neighbor=upBlockOf(q>>vec3u(1u));if(!upStored(neighbor)){continue;}
  let bits=vec3u(q)&vec3u(1u);let other=8u*neighbor+bits.x+2u*bits.y+4u*bits.z;
  atomicStore(&surface[${this.modelEpochWordOffset}u+other],0u);atomicStore(&geometry[${this.geometryRemapWordOffset}u+other],1u);rhPromote(other);
  // Epoch increments saturate without wrapping, even for overlapping claims.
  for(var plane=0u;plane<3u;plane++){let ep=select(${this.liveEpochWordOffset}u+plane*${this.regions}u,${this.topologyEpochWordOffset}u,plane==2u)+other;
   var before=atomicLoad(&surface[ep]);loop{let next=select(before+1u,0xfffffffeu,before>=0xfffffffdu);let result=atomicCompareExchangeWeak(&surface[ep],before,next);if(result.exchanged){break;}before=result.old_value;}
  }
 }}}
}`)}`});
      this.initPipeline=await this.device.createComputePipelineAsync({layout:"auto",compute:{module:init,entryPoint:"initialize"}});
      this.initGroup=this.device.createBindGroup({layout:this.initPipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.pool.state}},{binding:1,resource:{buffer:this.geometry}},{binding:2,resource:{buffer:this.surface}},...(this.heat?[{binding:3,resource:{buffer:this.heat.tiers}},{binding:4,resource:{buffer:this.heat.state}}]:[])]});
    }
  }
  /** Publish a builder remap immediately after remap/resolve, before any
   * geometry consumer. FLAGS bit16 already includes its affected tile halo.
   * This fixed metadata-only pass cannot lose changes if the builder is reused. */
  encodeRemapRevisions(encoder:GPUCommandEncoder,changes:GPUBuffer):void{
    if(!this.remapPipeline||!this.remapLayout)throw new Error("Work-control remap revisions are not initialized");
    if(this.remapGroup?.changes!==changes)this.remapGroup={changes,group:this.device.createBindGroup({layout:this.remapLayout,entries:[{binding:0,resource:{buffer:changes}},{binding:1,resource:{buffer:this.surface}},{binding:2,resource:{buffer:this.geometry}},...(this.heat?[{binding:3,resource:{buffer:this.heat.tiers}},{binding:4,resource:{buffer:this.heat.state}}]:[]),...(this.pool?[{binding:5,resource:{buffer:this.pool.state}}]:[])]})};
    const pass=encoder.beginComputePass({label:"Uniform16h remap producer revisions"});pass.setPipeline(this.remapPipeline);pass.setBindGroup(0,this.ownership.bindGroup);pass.setBindGroup(1,this.remapGroup.group);
    if(this.pool)pass.dispatchWorkgroups(Math.min(UNIFORM_LIVE_GROUP_GROUPS,this.regions));else pass.dispatchWorkgroups(...this.supportGrids);pass.end();
  }
  encodeInitBlocks(encoder:GPUCommandEncoder):void{
    if(!this.pool||!this.initPipeline||!this.initGroup)throw new Error("Work control has no initialized page pool");
    const pass=encoder.beginComputePass({label:"Uniform 16h work evidence page claims"});pass.setPipeline(this.initPipeline);pass.setBindGroup(0,this.initGroup);pass.dispatchWorkgroups(uniformPageInitGroups(this.pool.layout));pass.end();
  }
  /** The shared head has maintained the cache mapping on the current pages. */
  get regionalGeometryReady():boolean{return this.regionalGeometrySetup&&this.framePlan.retainGeometryCache&&this.framePlan.active&&!this.geometryRefreshRequired&&this.geometrySnapshots;}
  /** Same dispatch in a consumer's compute pass. Dispatch ordering preserves
   * the snapshot-to-geometry dependency without another pass boundary. The
   * caller clears the 16-byte counters before the pass and consumes body
   * evidence after it. */
  encodeRegionalGeometry(pass:GPUComputePassEncoder):void{
    if(!this.regionalGeometryReady)throw new Error("Regional geometry needs a sealed frame plan and retained snapshots");
    pass.setBindGroup(0,this.ownership.bindGroup);pass.setBindGroup(1,this.groups[1]!.group);if(this.solid)pass.setBindGroup(2,this.solid.tileGroup);
    pass.setPipeline(uniformDetailPick(this.pipelines.get(this.collectGeometryStats?"geometryRegionalStats":"geometryRegional")!));pass.dispatchWorkgroups(this.framePlan.groups);
  }
  encode(encoder:GPUCommandEncoder,stage:"surface"|"geometry"):void{
    if(stage==="geometry"&&this.regionalGeometryReady){
      encoder.clearBuffer(this.geometry,this.countersOffset,16);
      const pass=encoder.beginComputePass({label:"Uniform regional geometry work control"});
      this.encodeRegionalGeometry(pass);pass.end();
      this.solid?.consumeBodyEvidence(encoder);return;
    }
    // The surface model evidence is off: its words stay zero (no coverage, no valid support), so advect takes its full paths.
    if(stage==="surface"&&!this.surfaceEvidence)return;
    const bank=stage==="surface"?0:1,buffer=bank===0?this.surface:this.geometry;
    if(stage==="geometry"){encoder.clearBuffer(buffer,0,4*this.regions);encoder.clearBuffer(buffer,4*this.topologyWordOffset,4*this.regions);encoder.clearBuffer(buffer,4*this.mappingWordOffset,4*this.regions);}encoder.clearBuffer(buffer,this.countersOffset,16);
    if(stage==="geometry")encoder.copyBufferToBuffer(this.surface,4*this.liveEpochWordOffset,this.geometry,4*this.geometryLiveWordOffset,4*this.regions);
    const force=this.forced[bank]||!this.enabled;
    const listed=stage==="surface"&&this.framePlan.active&&!force;

    const pass=encoder.beginComputePass({label:`Uniform 16h ${stage} work control`});
    pass.setBindGroup(0,this.ownership.bindGroup);pass.setBindGroup(1,this.groups[bank]!.group);if(this.solid)pass.setBindGroup(2,this.solid.tileGroup);
    pass.setPipeline(uniformDetailPick(this.pipelines.get(stage==="surface"?(listed?"selectSurfaceListed":force?"selectSurfaceForce":"selectSurface"):"select")!));if(stage==="surface"){if(listed)pass.dispatchWorkgroups(this.framePlan.groups);else pass.dispatchWorkgroups(...this.supportGrids);}else pass.dispatchWorkgroups(Math.ceil(this.cacheRegions/64));
    const metadataOnly=stage==="geometry"&&!this.geometrySnapshots;
    pass.setPipeline(uniformDetailPick(this.pipelines.get(stage==="surface"?(listed?"modelScanListed":"modelScan"):metadataOnly?(force?"metadataForce":"metadata"):(force?"force":"scan"))!));
    if(listed)pass.dispatchWorkgroups(this.framePlan.groups);else if(metadataOnly)pass.dispatchWorkgroups(this.cacheRegions);else pass.dispatchWorkgroups(...(stage==="surface"?this.surfaceGrids:this.grids));
    const finish=metadataOnly?(!this.enabled?"geometryCurrentDisabled":this.collectGeometryStats?"geometryCurrentStats":"geometryCurrent"):!this.enabled?`${stage}Disabled`:stage==="geometry"&&this.collectGeometryStats?"geometryStats":stage;
    pass.setPipeline(uniformDetailPick(this.pipelines.get(listed?"surfaceListed":finish)!));if(stage==="surface"){if(listed)pass.dispatchWorkgroups(this.framePlan.groups);else pass.dispatchWorkgroups(...this.supportGrids);}else pass.dispatchWorkgroups(Math.ceil(this.regions/64));pass.end();this.forced[bank]=false;
    if(stage==="geometry"){this.solid?.consumeBodyEvidence(encoder);}
  }
  destroy():void{this.framePlan.destroy();this.surface.destroy();this.geometry.destroy();}
}
