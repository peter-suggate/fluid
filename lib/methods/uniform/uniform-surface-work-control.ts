import { uniformDetailBindLayout, uniformDetailGroup, uniformDetailModule, uniformDetailPipeline, uniformDetailPick, type UniformDetailGroup } from "./uniform-detail-fields";
import { UNIFORM_DETAIL_4H_LOAD } from "../../core/uniform-detail-abi";
import type { UniformMixedOwnership } from "./uniform-mixed-ownership";
import { uniformMixedTopologyWGSL } from "./uniform-mixed-topology.wgsl";
import { uniformMixedSolidWGSL, type UniformMixedSolid } from "./uniform-mixed-solid.wgsl";
import { uniformPagePoolWGSL, type UniformPagePool } from "./uniform-page-pool";
import { uniformPageInitBlocksWGSL, uniformPageInitGroups } from "./uniform-page-initialise";
import { UNIFORM_SURFACE_SUPPORT_WORDS } from "./uniform-surface-transport.wgsl";

/** Shared current surface models and exact derived-geometry evidence on a
 * 16h lattice, independent of 32h paging. Surface model scans retain no tile
 * snapshots: producer revisions plus compact topology/world identity retain
 * exact affine fits, residuals and slopes before phi gathers. Coverage and
 * velocity support refresh every call. Geometry retains exact tile closures in a
 * bounded 512-region cache; misses conservatively refresh. Topology, page
 * claims, and current/previous body support invalidate derived geometry.
 */
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
  private readonly modelEpochWordOffset:number;
  private readonly modelReuseWordOffset:number;
  private readonly modelMetadataWordOffset:number;
  private readonly modelWorldWordOffset:number;
  private readonly geometryLiveWordOffset:number;
  private readonly geometryEpochWordOffset:number;
  private readonly geometryRemapWordOffset:number;
  private remapPipeline?:GPUComputePipeline;
  private remapLayout?:GPUBindGroupLayout;
  private remapGroup?:{changes:GPUBuffer;group:GPUBindGroup};
  private revisionGeneration=0;
  get generation():number{return this.revisionGeneration;}
  get revisions(){return {buffer:this.surface,liveOffset:4*this.liveEpochWordOffset,scratchOffset:4*this.scratchEpochWordOffset,regions:this.regions};}
  readonly surface:GPUBuffer;
  readonly geometry:GPUBuffer;
  readonly allocatedBytes:number;
  /** Diagnostic full-work control: same scans/launches, no reuse or plane model. */
  enabled=true;
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
  constructor(private readonly device:GPUDevice,readonly ownership:UniformMixedOwnership,phi:GPUTexture,private readonly solid?:UniformMixedSolid,private readonly pool?:UniformPagePool){
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
    const surfaceBytes=4*(this.modelMetadataWordOffset+3*tiles),geometryBytes=4*(this.geometryRemapWordOffset+this.regions);
    const make=(stage:string,size:number)=>device.createBuffer({label:`Uniform 16h ${stage} work evidence`,size,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
    this.surface=make("surface",surfaceBytes);this.geometry=make("geometry",geometryBytes);this.allocatedBytes=surfaceBytes+geometryBytes;
    device.queue.writeBuffer(this.surface,4*this.liveEpochWordOffset,new Uint32Array(2*this.regions).fill(1));
    this.layout=uniformDetailBindLayout(device,{entries:[
      {binding:0,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float",viewDimension:"3d"}},
      {binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},
      {binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},
    ]});
    this.groups=[this.surface,this.geometry].map(buffer=>uniformDetailGroup(device,{layout:this.layout,entries:[{binding:0,resource:phi},{binding:1,resource:{buffer}},{binding:2,resource:{buffer:ownership.regionVelocity}}]}));
    const jobs=64*this.cacheRegions,x=Math.min(jobs,ownership.dispatchX);this.grids=[x,Math.ceil(jobs/x)];
    const supportX=Math.min(this.regions,ownership.dispatchX);this.supportGrids=[supportX,Math.ceil(this.regions/supportX)];
    const sx=Math.min(tiles,ownership.dispatchX);this.surfaceGrids=[sx,Math.ceil(tiles/sx)];
  }
  /** Legacy foot-cache diagnostic ABI; producer revisions use revisions. */
  get epochOffset():undefined{return undefined;}
  /** Header: stored regions, dirty regions, enabled/stats bits, cache misses. */
  get countersOffset():number{return 12*this.regions;}
  invalidate():void{this.forced[0]=this.forced[1]=true;this.revisionGeneration++;}
  /** An intervening writer changed geometry outputs outside this snapshot. */
  invalidateGeometry():void{this.forced[1]=true;}
  async initialize():Promise<void>{
    const module=uniformDetailModule(this.device,{label:"Uniform 16h surface work control",code:uniformMixedTopologyWGSL(this.ownership.capacity,0)+/* wgsl */`
@group(1) @binding(0) var phi:texture_3d<f32>;
@group(1) @binding(1) var<storage,read_write> wc:array<atomic<u32>>;
@group(1) @binding(2) var<storage,read> wcVelocity:array<u32>;
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
@compute @workgroup_size(64) fn selectSurface(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 let region=group.x+umDispatchX*group.y;if(region>=UM_PAGES){return;}
 if(lane==0u){
  let block=region>>3u;wcSelectLive=u32(umBlockStored(block)&&umBlockWord(block)!=0xffffffffu);
  atomicStore(&wcSelectChanged,0u);atomicStore(&wcSelectClosed,0u);
 }
 if(workgroupUniformLoad(&wcSelectLive)==0u){
  if(lane==0u){atomicStore(&wc[region],0u);atomicStore(&wc[WC_MODEL_REUSE+region],0u);atomicStore(&wc[WC_MODEL+8u*region+5u],0u);atomicStore(&wc[WC_MODEL+8u*region+7u],0u);atomicStore(&wc[WC_MODEL_EPOCH+region],0u);}return;
 }
 if(lane==0u){
  let epoch=atomicLoad(&wc[WC_LIVE+region]);
  var same=!wcForce&&epoch>0u&&epoch<0xfffffffeu&&epoch==atomicLoad(&wc[WC_MODEL_EPOCH+region])&&atomicLoad(&wc[WC_MODEL+8u*region+7u])!=0u;
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
}
var<workgroup> wcChanged:atomic<u32>;
var<workgroup> wcOwner:u32;
var<workgroup> wcSkip:u32;
var<workgroup> wcInputs:array<u32,125>;
var<workgroup> wcPlane:vec4f;
var<workgroup> wcResidual:atomic<u32>;
var<workgroup> wcSlope:array<atomic<u32>,3>;
@compute @workgroup_size(64) fn scan(@builtin(workgroup_id) g:vec3u,@builtin(local_invocation_index) lane:u32){
 let job=g.x+umDispatchX*g.y;var tile=job;var region=0u;var at=0u;
 if(wcSurface){
  if(tile>=UM_TILES||!umTileLive(tile)){return;}region=umPageOfTile(tile);
 }else{
  if(job>=64u*WC_CACHE){return;}
  if(lane==0u){wcOwner=atomicLoad(&wc[WC_OWNERS+job/64u]);}
  let owner=workgroupUniformLoad(&wcOwner);if(owner==0xffffffffu){return;}
  region=owner;tile=umPageTile(region,job%64u);if(!umTileLive(tile)){return;}
  at=WC_SNAPSHOT+132u*job;
 }
 if(wcSurface){if(lane==0u){wcSkip=atomicLoad(&wc[WC_MODEL_REUSE+region]);}if(workgroupUniformLoad(&wcSkip)!=0u){return;}}
 let base=4u*umTileCoord(tile);
 if(lane==0u){
  if(!wcSurface){
   let stencil=umTileStencil(tile);
   let metadata=array<u32,7>(umTopology[tile],stencil.x,stencil.y,base.x,base.y,base.z,1u);
   var different=wcForce||atomicLoad(&wc[region])!=0u||atomicLoad(&wc[WC_GEOMETRY_REMAP+region])!=0u;var topology=wcForce||atomicLoad(&wc[WC_GEOMETRY_REMAP+region])!=0u;
   for(var k=0u;k<7u;k++){let value=metadata[k];let changed=atomicLoad(&wc[at+125u+k])!=value;different=different||changed;topology=topology||changed;atomicStore(&wc[at+125u+k],value);}
   if(topology){atomicStore(&wc[WC_TOPOLOGY+region],1u);}atomicStore(&wcChanged,u32(different));
   let epoch=atomicLoad(&wc[WC_GEOMETRY_LIVE+region]);
   wcSkip=u32(!different&&epoch>0u&&epoch<0xfffffffeu&&epoch==atomicLoad(&wc[WC_GEOMETRY_EPOCH+region]));
  }
  if(wcSurface){
   let m=WC_MODEL+8u*region;wcPlane=bitcast<vec4f>(vec4u(atomicLoad(&wc[m]),atomicLoad(&wc[m+1u]),atomicLoad(&wc[m+2u]),atomicLoad(&wc[m+3u])));atomicStore(&wcResidual,0u);
   for(var axis=0u;axis<3u;axis++){atomicStore(&wcSlope[axis],0u);}

  }
 }
 if(!wcSurface&&workgroupUniformLoad(&wcSkip)!=0u){return;}
 workgroupBarrier();
 let width=umTileMinimumWidth(tile);let dense=width<4u;
 let side=4u/width+1u;let count=select(8u,side*side*side,dense);
 var residual=0.0;
 for(var k=lane;k<count;k+=64u){
  var value=0.0;
  if(dense){value=textureLoad(phi,vec3i(base+width*umCorner(k,side)),0).x;}
  else{value=${UNIFORM_DETAIL_4H_LOAD}textureLoad(phi,vec3i(base+4u*umCorner(k,2u)),0).x;}
  wcInputs[k]=bitcast<u32>(value);
  if(!wcSurface){
   let bits=bitcast<u32>(value);
   let previous=bitcast<f32>(atomicLoad(&wc[at+k]));
   if(bits!=bitcast<u32>(previous)||!(abs(value)<=3.402823e38&&abs(previous)<=3.402823e38)){atomicStore(&wcChanged,1u);}
  }
  if(wcSurface){
   let point=base+width*umCorner(k,side);var error=abs(value-(wcPlane.w+dot(wcPlane.xyz,vec3f(point)-vec3f(16u*umPageCoord(region)+vec3u(8u)))));
   if(!(abs(value)<=3.402823e38&&error<=3.402823e38)){error=3.402823e38;}
   residual=max(residual,error);
  }
 }
 if(wcSurface){atomicMax(&wcResidual,bitcast<u32>(residual));}
 workgroupBarrier();
 if(wcSurface){
  // Every derivative of a trilinear cell is a convex combination of its
  // four parallel edge slopes. Include resolved hanging vertices: native
  // coarse edges are sums of these shorter edges at mixed boundaries.
  let n=select(2u,side,dense);let step=select(4.0,f32(width),dense);var slopes=vec3f(0);
  for(var k=lane;k<count;k+=64u){
   let q=umCorner(k,n);let a=bitcast<f32>(wcInputs[k]);
   for(var axis=0u;axis<3u;axis++){
    if(q[axis]+1u>=n){continue;}
    let stride=select(select(1u,n,axis==1u),n*n,axis==2u);let b=bitcast<f32>(wcInputs[k+stride]);
    var bound=(abs(b-a)+4.0*1.1920929e-7*(abs(a)+abs(b)))/step;
    if(!(abs(a)<=3.402823e38&&abs(b)<=3.402823e38&&bound<=3.402823e38)){bound=3.402823e38;}
    slopes[axis]=max(slopes[axis],bound);
   }
  }
  for(var axis=0u;axis<3u;axis++){atomicMax(&wcSlope[axis],bitcast<u32>(slopes[axis]));}
  workgroupBarrier();
  if(lane==0u){atomicMax(&wc[WC_MODEL+8u*region+6u],atomicLoad(&wcResidual));for(var axis=0u;axis<3u;axis++){atomicMax(&wc[WC_SLOPE+3u*region+axis],atomicLoad(&wcSlope[axis]));}}
 }
 // Only geometry retains exact tile snapshots, committed collectively.
 if(!wcSurface&&atomicLoad(&wcChanged)!=0u){
  for(var k=lane;k<count;k+=64u){atomicStore(&wc[at+k],wcInputs[k]);}
  if(lane==0u){atomicStore(&wc[umPageOfTile(tile)],1u);}
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
@compute @workgroup_size(64) fn summaries(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 let region=group.x+umDispatchX*group.y;if(region>=UM_PAGES){return;}
 let output=WC_SUPPORT+${UNIFORM_SURFACE_SUPPORT_WORDS}u*(64u*region+lane);atomicStore(&wc[output+7u],0u);
 if(lane==0u){
  atomicStore(&wc[WC_MODEL_EPOCH+region],atomicLoad(&wc[WC_LIVE+region]));
  if(region==0u){atomicStore(&wc[3u*UM_PAGES+2u],u32(wcEnabled));}
  wcSupportLive=u32(umBlockStored(region>>3u)&&umBlockWord(region>>3u)!=0xffffffffu);
  let home=WC_MODEL+8u*region;wcSupportHome=bitcast<vec4f>(vec4u(atomicLoad(&wc[home]),atomicLoad(&wc[home+1u]),atomicLoad(&wc[home+2u]),atomicLoad(&wc[home+3u])));
 }
 if(workgroupUniformLoad(&wcSupportLive)==0u){return;}
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
}
// Separate launch: every scan's dirty word is complete before it is read.
@compute @workgroup_size(64) fn finish(@builtin(global_invocation_id) gid:vec3u){
 let region=gid.x;if(region>=UM_PAGES){return;}
 if(region==0u){atomicStore(&wc[3u*UM_PAGES+2u],u32(wcEnabled)+2u*u32(wcGeometryStats));}
 let block=region>>3u;if(!umBlockStored(block)||umBlockWord(block)==0xffffffffu){return;}
 let origin=16u*umPageCoord(region);
 let miss=atomicLoad(&wc[WC_MAPPING+region])==0u;
 atomicStore(&wc[WC_GEOMETRY_EPOCH+region],atomicLoad(&wc[WC_GEOMETRY_LIVE+region]));
 let remapped=atomicLoad(&wc[WC_GEOMETRY_REMAP+region])!=0u;
 var dirty=wcMetadataOnly||miss||atomicLoad(&wc[region])!=0u||remapped;
 var body=false;
 ${this.solid?/* wgsl */`let lo=umSolidWorldCell(vec3i(origin));let hi=umSolidWorldCell(vec3i(origin+vec3u(16u)));
 let margin=8.0*umSceneMaximumSpacing();
 for(var i=0u;i<umBodyCount();i++){
  let p=umSolidBodies[i].positionShape.xyz;
  body=body||umBodyNear(i,clamp(p,lo,hi),margin);
 }`:""}
 let previousBody=atomicLoad(&wc[2u*UM_PAGES+region])!=0u;
 dirty=dirty||body||previousBody;
 atomicStore(&wc[2u*UM_PAGES+region],u32(body));
 let epoch=atomicLoad(&wc[UM_PAGES+region]);
 // Saturation disables cache reads rather than ever reusing an old epoch.
 atomicStore(&wc[UM_PAGES+region],min(0x1fffffffu,epoch+u32(dirty||epoch==0u)));
 atomicAdd(&wc[3u*UM_PAGES],1u);if(miss){atomicAdd(&wc[3u*UM_PAGES+3u],1u);}if(dirty){atomicAdd(&wc[3u*UM_PAGES+1u],1u);}
 // Geometry reads this plane. Surface must not overwrite it here: neighbors
 // in this dispatch still read the scan's undilated dirty evidence.
 if(!wcSurface){atomicStore(&wc[region],u32(dirty));if(miss||body||previousBody||remapped){atomicStore(&wc[WC_TOPOLOGY+region],1u);}atomicStore(&wc[WC_GEOMETRY_REMAP+region],0u);}
}
`});
    const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");if(errors.length)throw new Error(errors.map(m=>`${m.lineNum}: ${m.message}`).join("\n"));
    const layout=this.device.createPipelineLayout({bindGroupLayouts:[this.ownership.bindLayout,this.layout,...(this.solid?[this.solid.tileLayout]:[])]});
    for(const [name,entryPoint,constants] of [
      ["metadata","metadata",{}],["metadataForce","metadata",{wcForce:1}],["geometryCurrent","finish",{wcMetadataOnly:1}],["geometryCurrentStats","finish",{wcMetadataOnly:1,wcGeometryStats:1}],["geometryCurrentDisabled","finish",{wcMetadataOnly:1,wcEnabled:0}],
      ["select","selectRegions",{}],["selectSurface","selectSurface",{wcSurface:1}],["selectSurfaceForce","selectSurface",{wcSurface:1,wcForce:1}],["scan","scan",{}],["force","scan",{wcForce:1}],["modelScan","scan",{wcSurface:1}],["surface","summaries",{wcSurface:1}],["surfaceDisabled","summaries",{wcSurface:1,wcEnabled:0}],["geometry","finish",{}],["geometryStats","finish",{wcGeometryStats:1}],["geometryDisabled","finish",{wcEnabled:0}],
    ] as const)this.pipelines.set(name,await uniformDetailPipeline(this.device,this.ownership,{layout,compute:{module,entryPoint,constants:{umDispatchX:entryPoint==="summaries"||entryPoint==="selectSurface"?this.supportGrids[0]:name==="modelScan"?this.surfaceGrids[0]:this.grids[0],...constants}}}));
    const remap=this.device.createShaderModule({label:"Uniform16h remap producer revisions",code:uniformMixedTopologyWGSL(this.ownership.capacity,0)+`
@group(1) @binding(0) var<storage,read> changes:array<u32>;
@group(1) @binding(1) var<storage,read_write> surface:array<atomic<u32>>;
@group(1) @binding(2) var<storage,read_write> geometry:array<atomic<u32>>;
var<workgroup> changed:atomic<u32>;
@compute @workgroup_size(64) fn publish(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 let region=group.x+umDispatchX*group.y;if(region>=UM_PAGES){return;}
 if(lane==0u){atomicStore(&changed,0u);}workgroupBarrier();
 let tile=umPageTile(region,lane);
 if(umTileLive(tile)&&(changes[76u+tile]&16u)!=0u){atomicOr(&changed,1u);}workgroupBarrier();
 if(lane==0u&&atomicLoad(&changed)!=0u){
  let at=${this.liveEpochWordOffset}u+region;let before=atomicLoad(&surface[at]);atomicStore(&surface[at],select(min(before,0xfffffffeu)+1u,0xfffffffeu,before>=0xfffffffdu));
  atomicStore(&surface[${this.modelEpochWordOffset}u+region],0u);atomicStore(&geometry[${this.geometryRemapWordOffset}u+region],1u);
 }
}`});
    this.remapLayout=this.device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},...[1,2].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage" as const}}))]});
    this.remapPipeline=await this.device.createComputePipelineAsync({layout:this.device.createPipelineLayout({bindGroupLayouts:[this.ownership.bindLayout,this.remapLayout]}),compute:{module:remap,entryPoint:"publish",constants:{umDispatchX:this.supportGrids[0]}}});
    if(this.pool){
      const init=this.device.createShaderModule({label:"Uniform 16h work evidence page claims",code:`
${uniformPagePoolWGSL(this.pool.layout,0,0)}
@group(0) @binding(1) var<storage,read_write> geometry:array<atomic<u32>>;
@group(0) @binding(2) var<storage,read_write> surface:array<atomic<u32>>;
${uniformPageInitBlocksWGSL(this.pool.layout,"initialize",512,`let region=8u*block+((local>>2u)&1u)+2u*((local>>5u)&1u)+4u*((local>>8u)&1u);atomicStore(&surface[${this.modelEpochWordOffset}u+region],0u);let mapped=atomicLoad(&geometry[${this.mappingWordOffset}u+region]);if(mapped!=0u){let lane=(local&3u)|((local>>1u)&12u)|((local>>2u)&48u);let at=${this.snapshotOffset}u+132u*(64u*(mapped-1u)+lane)+131u;atomicStore(&geometry[at],0u);}
if(local<8u){let home=2*upPageOf(block)+vec3i(i32(local&1u),i32((local>>1u)&1u),i32(local>>2u));
 for(var z=-1;z<=1;z++){for(var y=-1;y<=1;y++){for(var x=-1;x<=1;x++){
  let q=home+vec3i(x,y,z);let neighbor=upBlockOf(q>>vec3u(1u));if(!upStored(neighbor)){continue;}
  let bits=vec3u(q)&vec3u(1u);let other=8u*neighbor+bits.x+2u*bits.y+4u*bits.z;
  atomicStore(&surface[${this.modelEpochWordOffset}u+other],0u);atomicStore(&geometry[${this.geometryRemapWordOffset}u+other],1u);
  // Epoch increments saturate without wrapping, even for overlapping claims.
  for(var plane=0u;plane<2u;plane++){let ep=${this.liveEpochWordOffset}u+plane*${this.regions}u+other;
   var before=atomicLoad(&surface[ep]);loop{let next=select(before+1u,0xfffffffeu,before>=0xfffffffdu);let result=atomicCompareExchangeWeak(&surface[ep],before,next);if(result.exchanged){break;}before=result.old_value;}
  }
 }}}
}`)}`});
      this.initPipeline=await this.device.createComputePipelineAsync({layout:"auto",compute:{module:init,entryPoint:"initialize"}});
      this.initGroup=this.device.createBindGroup({layout:this.initPipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.pool.state}},{binding:1,resource:{buffer:this.geometry}},{binding:2,resource:{buffer:this.surface}}]});
    }
  }
  /** Publish a builder remap immediately after remap/resolve, before any
   * geometry consumer. FLAGS bit16 already includes its affected tile halo.
   * This fixed metadata-only pass cannot lose changes if the builder is reused. */
  encodeRemapRevisions(encoder:GPUCommandEncoder,changes:GPUBuffer):void{
    if(!this.remapPipeline||!this.remapLayout)throw new Error("Work-control remap revisions are not initialized");
    if(this.remapGroup?.changes!==changes)this.remapGroup={changes,group:this.device.createBindGroup({layout:this.remapLayout,entries:[{binding:0,resource:{buffer:changes}},{binding:1,resource:{buffer:this.surface}},{binding:2,resource:{buffer:this.geometry}}]})};
    const pass=encoder.beginComputePass({label:"Uniform16h remap producer revisions"});pass.setPipeline(this.remapPipeline);pass.setBindGroup(0,this.ownership.bindGroup);pass.setBindGroup(1,this.remapGroup.group);pass.dispatchWorkgroups(...this.supportGrids);pass.end();
  }
  encodeInitBlocks(encoder:GPUCommandEncoder):void{
    if(!this.pool||!this.initPipeline||!this.initGroup)throw new Error("Work control has no initialized page pool");
    const pass=encoder.beginComputePass({label:"Uniform 16h work evidence page claims"});pass.setPipeline(this.initPipeline);pass.setBindGroup(0,this.initGroup);pass.dispatchWorkgroups(uniformPageInitGroups(this.pool.layout));pass.end();
  }
  encode(encoder:GPUCommandEncoder,stage:"surface"|"geometry"):void{
    const bank=stage==="surface"?0:1,buffer=bank===0?this.surface:this.geometry;
    if(stage==="geometry"){encoder.clearBuffer(buffer,0,4*this.regions);encoder.clearBuffer(buffer,4*this.topologyWordOffset,4*this.regions);encoder.clearBuffer(buffer,4*this.mappingWordOffset,4*this.regions);}encoder.clearBuffer(buffer,this.countersOffset,16);
    if(stage==="geometry")encoder.copyBufferToBuffer(this.surface,4*this.liveEpochWordOffset,this.geometry,4*this.geometryLiveWordOffset,4*this.regions);
    const force=this.forced[bank]||!this.enabled;
    const pass=encoder.beginComputePass({label:`Uniform 16h ${stage} work control`});
    pass.setBindGroup(0,this.ownership.bindGroup);pass.setBindGroup(1,this.groups[bank]!.group);if(this.solid)pass.setBindGroup(2,this.solid.tileGroup);
    pass.setPipeline(uniformDetailPick(this.pipelines.get(stage==="surface"?(force?"selectSurfaceForce":"selectSurface"):"select")!));if(stage==="surface")pass.dispatchWorkgroups(...this.supportGrids);else pass.dispatchWorkgroups(Math.ceil(this.cacheRegions/64));
    const metadataOnly=stage==="geometry"&&!this.geometrySnapshots;
    pass.setPipeline(uniformDetailPick(this.pipelines.get(stage==="surface"?"modelScan":metadataOnly?(force?"metadataForce":"metadata"):(force?"force":"scan"))!));
    if(metadataOnly)pass.dispatchWorkgroups(this.cacheRegions);else pass.dispatchWorkgroups(...(stage==="surface"?this.surfaceGrids:this.grids));
    const finish=metadataOnly?(!this.enabled?"geometryCurrentDisabled":this.collectGeometryStats?"geometryCurrentStats":"geometryCurrent"):!this.enabled?`${stage}Disabled`:stage==="geometry"&&this.collectGeometryStats?"geometryStats":stage;
    pass.setPipeline(uniformDetailPick(this.pipelines.get(finish)!));if(stage==="surface")pass.dispatchWorkgroups(...this.supportGrids);else pass.dispatchWorkgroups(Math.ceil(this.regions/64));pass.end();this.forced[bank]=false;
  }
  destroy():void{this.surface.destroy();this.geometry.destroy();}
}
