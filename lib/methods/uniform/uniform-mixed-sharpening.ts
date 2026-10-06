import { uniformBufferedWork } from "./uniform-buffered-work";
import { uniformSeamSideOrders } from "./uniform-compiled-topology";
import { uniformDetailBindLayout, uniformDetailExtent, uniformDetailModule, uniformDetailPipeline, uniformDetailPick, uniformDetailGroup, type UniformDetailGroup } from "./uniform-detail-fields";
import {UNIFORM_DETAIL_4H_LOAD} from "../../core/uniform-detail-abi";
import {uniformMixedDustAccountingWGSL} from "./uniform-mixed-dust-accounting.wgsl";
import {uniformMixedFaceAddressWGSL} from "./uniform-mixed-face-dispatch.wgsl";
import type { UniformMixedOwnership } from "./uniform-mixed-ownership";
import { UNIFORM_MIXED_COUNTED, uniformMixedCountedEntriesWGSL, uniformMixedTopologyWGSL } from "./uniform-mixed-topology.wgsl";
import { uniformMixedVertexSamplingSource } from "./uniform-mixed-vertex-sampling.wgsl";
import { uniformSharpenBudgetWGSL } from "./uniform-sharpen-budget.wgsl";
import { uniformMixedSolidPipeline, uniformMixedSolidWGSL, type UniformMixedSolid } from "./uniform-mixed-solid.wgsl";

/** Native geometric propose/limit/commit sweeps on physical mixed
 * face patches. Budgets and fluxes use fine-cell mass units, with area shares
 * splitting a coarse cell's offer across its subfaces. Its scratch is keyed
 * by owner index and fine-tile rank (scratchBytes): nothing in it is sized
 * by the lattice, and its bases follow the live owner count.
 * With static solids an owner is admitted only when fully open (a unit cell,
 * or a 4h owner's whole tile), and a unit-unit face only when both cells and
 * its aperture are fully open, the native uvOpen/faceOpenFraction >= 0.99999
 * gates.
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
 * sweep's budgets are prepared with the geometry cache.
 *
 * Quiet owners leave the sweeps. A proposal is exactly zero when either
 * owner has zero give and take budgets (every branch of umProposal takes a
 * min with one of them, or returns early), so such an owner's faces carry no
 * flux, its commit adds nothing, and its volume and budgets never change: an
 * owner quiet after the prepare (zero budgets, and a volume the dust rule
 * keeps) is quiet in all eight sweeps. The prepare writes once what its
 * sweeps would rewrite unchanged: its volume into the first sweep's output
 * (both ping-pong textures then hold it), unit limit factors, and a zero
 * proposal on the faces it would propose. The sweeps then visit only the
 * owners the prepare listed as active, regular or seam; each value equals
 * the dense sweep's.
 *
 * A seam tile is a 4h tile with an h tile across a face: sixteen unit
 * patches on that side, so its owner takes a lane per patch. Its flag word
 * names those sides, and a seam lane learns its side's patch count from it
 * with no neighbour lookup. A 4h tile that meets h tiles only across edges
 * and corners has one patch per face: a regular owner.
 *
 * The sweeps pack the seam owners that stay in them by their number of h
 * sides (shPackClass): an owner with one h side has 21 patches, with two 36,
 * with three 51, so 8, 4 or 3 of them share a job's 192 lanes (2 with more
 * sides). Each owner's lanes and the order its first lane reduces them in
 * are its own, so the packing changes no value. */
/** Workgroups of 192 lanes per merged sharpening launch at most. */
const SHARPEN_GRID=1024;
export class UniformMixedSharpening {
  readonly allocatedBytes=0;
  private readonly grid:number;
  private sweepWork:number;
  /** The packed seam owners' weight (24ths of a job, SH_SEAM_WEIGHT) and the
   * active-owner count from a completed frame. Each of the four pack
   * classes rounds its own jobs up: three spare jobs cover that.
   * The current shader still strides every job when detail grows abruptly.
   * reach: the reach of request edits in builds the counts do not report
   * yet (UniformWorkEdit): each of its tiles can add a seam owner (half a
   * job at most) or an h tile's 64 regular owners. */
  observeWork(counts:ArrayLike<number>,reserve=0,reach=0):void{
    if(counts.length!==2)throw new Error("Invalid sharpening work receipt");
    this.sweepWork=uniformBufferedWork(this.sweepWork,Math.ceil((counts[1]!+64*reach)/192)+Math.ceil((counts[0]!+12*reach)/24)+3,this.grid,reserve);
  }
  /** No evidence of the layout to come: the sweeps launch at the grid. */
  forgetWork():void{this.sweepWork=this.grid;}
  encodeWorkReceipt(encoder:GPUCommandEncoder,target:GPUBuffer,offset:number):void{
    encoder.copyBufferToBuffer(this.work.list,20,target,offset,8);
  }
  /** Bytes of the work list: three unused words, nine list words (the two
   * regular list counts, the packed seam owners' weight, the active regular
   * owner count, the seam tile count and the four pack counts), a flag per
   * tile (a seam tile's holds its h sides above the flag), one
   * tier-partitioned list of regular tiles with the seam tiles (4h tiles
   * with an h face neighbour) filling it from its end, two regions of sweep
   * packs (two classes each, from either end), then the active regular
   * six prepared neighbor indices per tile, then the active regular
   * owners (tile*64+lane): at most the ownership's owner capacity, every
   * tile h when none is given. */
  static workBytes(tiles:number,owners=64*tiles):number{return 4*(12+10*tiles+owners);}
  /** Between frames, after the ownership's owner capacity changed: a new
   * list of workBytes at it. Each encode rebuilds the list, so nothing is
   * copied; groups from bind() name the old one and must be bound again. */
  setWork(list:GPUBuffer):void{
    if(list.size<UniformMixedSharpening.workBytes(this.ownership.capacity.tiles,this.ownership.capacity.owners))throw new Error("Mixed sharpening work list is too small");
    this.work.list=list;
  }
  /** Bytes of bind()'s scratch for that many owners and h tiles: six budget
   * words an owner, then a raw flux and an admission word per face patch.
   * A patch is its lower owner's (three an owner), except the unit patches
   * of a 4h owner under an h tile, which are that tile's (48 an h tile). */
  static scratchBytes(owners:number,fineTiles:number):number{return 4*(12*owners+96*fineTiles);}
  get scratchBytes():number{const c=this.ownership.capacity;return UniformMixedSharpening.scratchBytes(c.owners,c.fineTiles);}
  private readonly resources:GPUBindGroupLayout;
  /** resources with the work list read-only: the sweeps never write it. */
  private readonly sweepResources:GPUBindGroupLayout;
  /** bind()'s group -> its twin on sweepResources (the same resources). */
  private readonly sweepGroups=new WeakMap<UniformDetailGroup,UniformDetailGroup>();
  private readonly pipelines=new Map<string,GPUComputePipeline[]>();
  /** work: a STORAGE|COPY_DST buffer of workBytes(tiles, owners). resolved: phi's
   * hanging texels hold umVertexValue (UniformMixedPhiResolve). */
  constructor(private readonly device:GPUDevice,readonly ownership:UniformMixedOwnership,private readonly solid:UniformMixedSolid|undefined,private readonly work:{list:GPUBuffer},private readonly resolved=false){
    const tiles=ownership.capacity.tiles;
    if(work.list.size<UniformMixedSharpening.workBytes(tiles,ownership.capacity.owners))throw new Error("Mixed sharpening work list is too small");
    // A sweep pack word holds its tile below its sides and class.
    if(tiles>=1<<24)throw new Error("Mixed sharpening packs at most 2^24 tiles");
    // Each merged launch is a fixed grid-stride grid: the capacity's job bound
    // (every tile listed), capped where the GPU is saturated.
    this.grid=Math.max(1,Math.min(SHARPEN_GRID,Math.ceil(64*tiles/192)+tiles));
    this.sweepWork=this.grid;
    const resources=(work:GPUBufferBindingType)=>uniformDetailBindLayout(device,{entries:[
      ...[0,1,2,3].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float" as const,viewDimension:"3d" as const}})),
      {binding:4,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"write-only",format:"r32float",viewDimension:"3d"}},
      {binding:5,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},
      {binding:6,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},
      {binding:7,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},
      {binding:8,visibility:GPUShaderStage.COMPUTE,buffer:{type:work}},
    ]});
    this.resources=resources("storage");this.sweepResources=resources("read-only-storage");
  }
  bind(input:GPUTexture,output:GPUTexture,phi:GPUTexture,targetFill:GPUTexture,centerPhi:GPUTexture,scratch:GPUBufferBinding,params:GPUBuffer,reductions:GPUBuffer):UniformDetailGroup{
    const d=this.ownership.capacity.lattice.dimensions;
    for(const [i,t] of [input,targetFill,centerPhi,phi,output].entries())
      if(t.format!=="r32float"||uniformDetailExtent(t).some((v,a)=>v!==d[a]!+(i===3?1:0)))throw new Error("Mixed sharpening requires native cell and vertex fields");
    if(input===output||output===targetFill||output===centerPhi)throw new Error("Mixed sharpening output must be disjoint");
    if((scratch.size??scratch.buffer.size-(scratch.offset??0))<this.scratchBytes)throw new Error("Mixed sharpening scratch is too small");
    const group=(layout:GPUBindGroupLayout)=>uniformDetailGroup(this.device,{layout,entries:[
      ...[input,targetFill,centerPhi,phi].map((t,binding)=>({binding,resource:t})),
      {binding:4,resource:output},{binding:5,resource:scratch},{binding:6,resource:{buffer:params,size:32}},
      {binding:7,resource:{buffer:reductions}},
      {binding:8,resource:{buffer:this.work.list}},
    ]});
    const geometry=group(this.resources);this.sweepGroups.set(geometry,group(this.sweepResources));return geometry;
  }
  async initialize():Promise<void>{
    const h=this.ownership.capacity.lattice.cellSize_m,tiles=this.ownership.capacity.tiles;
    // classify: a GPU-counted fixed grid over every live owner (all).
    // The sweeps (propose, limit, commit) never write the work list: their
    // module binds it read-only, so list and flag reads are plain loads.
    const source=(sweep:boolean)=>uniformMixedCountedEntriesWGSL(uniformMixedTopologyWGSL(this.ownership.capacity,0)+/* wgsl */`
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
// A tile corner, and below a 4h owner's own texel (its tile origin): the
// base blocks (UNIFORM_DETAIL_4H_LOAD).
fn umLoadCorner(p:vec3u)->f32{return ${UNIFORM_DETAIL_4H_LOAD}textureLoad(phi,vec3i(p),0).x;}
${uniformMixedVertexSamplingSource("",this.resolved,undefined,"umLoadCorner")}
${uniformMixedFaceAddressWGSL}
${uniformSharpenBudgetWGSL}
${uniformMixedSolidWGSL(this.solid?2:undefined,this.solid?.coarse?.count)}
// Sharpening moves V between whole owners: a cut owner, at either width, and
// every face of one stay out (its V is not a fraction of a whole cell).
fn umSharpenOpen(o:UMOwner)->bool{
 if(!umSolidEnabled()){return true;}
 if(o.width==1u){return umCellOpen(vec3i(umOrigin(o)))>0.99999;}
 return umTileOpen(o.tile)>0.99999;
}
fn umSharpenFaceOpenAB(a:UMOwner,b:UMOwner,f:UMFace)->bool{
 if(!umSolidEnabled()){return true;}
 if(!umSharpenOpen(a)||!umSharpenOpen(b)){return false;}
 // Two whole owners with a 4h one among them share a whole face.
 return a.width!=1u||b.width!=1u||umFaceOpen(f.anchor,f.axis)>0.99999;
}
fn umSharpenFaceOpen(a:UMOwner,f:UMFace)->bool{return umSharpenFaceOpenAB(a,f.neighbor,f);}
${sweep?"@group(1) @binding(8) var<storage,read> work:array<u32>;\nfn shWord(i:u32)->u32{return work[i];}":"@group(1) @binding(8) var<storage,read_write> work:array<atomic<u32>>;\nfn shWord(i:u32)->u32{return atomicLoad(&work[i]);}"}
// Lists 0 (h) and 1 (regular 4h) are tiers; tier t owners have width 4^t.
const SH_COUNTS:u32=3u;const SH_FLAGS:u32=12u;const SH_LIST:u32=${12+tiles}u;
const SH_ACTIVE_COUNT:u32=6u;const SH_LINKS:u32=${12+4*tiles}u;const SH_ACTIVE:u32=${12+10*tiles}u;
fn shTier(width:u32)->u32{return select(1u,0u,width==1u);}
// Seam tiles: 4h tiles with an h tile across a face. Their owners meet
// sixteen unit patches there, so one lane per owner would walk them
// serially. SH_SEAMS counts them; they fill the tier lists' region from its
// end (4h tiles are regular or seam, so the two never meet).
const SH_SEAMS:u32=7u;
fn shListStart(tier:u32)->u32{return SH_LIST+select(0u,umCounts.x,tier>0u);}
// Sweep packs: the seam tiles whose owner stays in the sweeps, in four
// classes by their number of h sides (1, 2, 3, 4 or more). A class's owners
// each take the same block of a job's 192 lanes, 24, 48, 64 or 96 of them:
// sixteen patches on each h side and one on each other side come to 21, 36,
// 51 and at most 96, so 8, 4, 3 or 2 owners share a job and a lane finds its
// owner by division. A pack word is the tile, its h sides (bits 24-29) and
// its class (bits 30-31); zero is no owner. SH_PACK counts each class.
// Classes 0 and 1 fill one region of UM_TILES words from its two ends and
// classes 2 and 3 the next (seam tiles never outnumber the tiles).
// SH_SEAM_WEIGHT sums 24/owners-per-job over the packed owners: the host's
// receipt of the seam jobs (observeWork).
const SH_PACK:u32=8u;const SH_SEAM_WEIGHT:u32=5u;
fn shPackClass(sides:u32)->u32{return min(countOneBits(sides),4u)-1u;}
fn shPackOwners(c:u32)->u32{return select(select(select(2u,3u,c==2u),4u,c==1u),8u,c==0u);}
fn shPackAt(c:u32,slot:u32)->u32{let base=SH_LIST+UM_TILES*(1u+c/2u);return select(base+slot,base+UM_TILES-1u-slot,(c&1u)!=0u);}
fn shSeamJobTile(job:u32)->u32{if(job>=shWord(SH_SEAMS)){return UM_TILES;}return shWord(SH_LIST+UM_TILES-1u-job);}
// A seam tile's side (2*axis, +1 the low one) holds sixteen patches when
// its bit above the tile's flag is set (an h tile there), one otherwise.
fn shSides(tile:u32)->u32{return shWord(SH_FLAGS+tile)>>1u;}
fn shSideParts(sides:u32,side:u32)->u32{return select(1u,16u,((sides>>side)&1u)!=0u);}
fn shListed(o:UMOwner)->bool{return o.width==0u||shWord(SH_FLAGS+o.tile)!=0u;}
// Listed tiers hold h owners and uniform-stencil 4h owners: one patch per
// face. A constant part count lets the six face chains issue together; a
// data-dependent loop serialized them (cold misses at coarse owners).
fn shParts(first:UMFace)->u32{return 1u;}
fn umMassScale(o:UMOwner)->f32{return f32(o.width*o.width*o.width);}
fn umV(o:UMOwner)->f32{let p=vec3i(umOrigin(o));if(o.width==4u){return ${UNIFORM_DETAIL_4H_LOAD}textureLoad(volume,p,0).x;}return textureLoad(volume,p,0).x;}
fn shPhi(o:UMOwner)->f32{let p=vec3i(umOrigin(o));if(o.width==4u){return ${UNIFORM_DETAIL_4H_LOAD}textureLoad(centerPhi,p,0).x;}return textureLoad(centerPhi,p,0).x;}
fn shFill(o:UMOwner)->f32{let p=vec3i(umOrigin(o));if(o.width==4u){return ${UNIFORM_DETAIL_4H_LOAD}textureLoad(targetFill,p,0).x;}return textureLoad(targetFill,p,0).x;}
// Owner record: give, take, distance, desired, outgoing and incoming
// limit factors.
fn umBudgetAt(o:UMOwner)->u32{return 6u*o.index;}
// A face patch's record among the live owners' (shLive): its lower owner a's
// (index, component), an h owner's unit patch or a 4h owner's width-4 one.
// The sixteen unit patches a 4h owner has under an h tile are that tile's
// instead: 48 a fine-tile rank (the upper owner b's), by component and
// column. Both incident owners hold a, b and f: no load.
fn shLive()->u32{return 64u*umCounts.x+umCounts.y;}
fn shFaceAt(a:UMOwner,b:UMOwner,f:UMFace)->u32{
 if(a.width!=1u&&b.width==1u){
  return 3u*shLive()+48u*((b.index-b.lane)>>6u)+16u*f.axis+(u32(f.anchor[(f.axis+1u)%3u])&3u)+4u*(u32(f.anchor[(f.axis+2u)%3u])&3u);
 }
 return 3u*a.index+f.axis;
}
// The raw flux across patch f from its lower owner a to its upper owner b,
// then its two admission bits and the immutable open-face bit.
fn umRawAt(a:UMOwner,b:UMOwner,f:UMFace)->u32{return 6u*shLive()+shFaceAt(a,b,f);}
fn umCacheAt(a:UMOwner,b:UMOwner,f:UMFace)->u32{return 9u*shLive()+48u*umCounts.x+shFaceAt(a,b,f);}
// The raw flux of owner o's face f, on either side.
fn umRawOf(o:UMOwner,f:UMFace)->u32{if(f.sign<0){return umRawAt(f.neighbor,o,f);}return umRawAt(o,f.neighbor,f);}
// Regular fine interiors use lane strides. Only crossing a tile boundary
// reads a prepared link; classification/listed checks were done once.
fn shFaceFirst(o:UMOwner,axis:u32,sign:i32)->UMFace{
 let stride=vec3u(1u,4u,16u);let local=umCorner(o.lane,4u);
 var anchor=vec3i(umOrigin(o));anchor[axis]+=select(-1,i32(o.width)-1,sign>0);
 if(o.width==1u&&((sign>0&&local[axis]<3u)||(sign<0&&local[axis]>0u))){
  let delta=sign*i32(stride[axis]);
  return UMFace(UMOwner(o.tile,u32(i32(o.lane)+delta),1u,u32(i32(o.index)+delta)),anchor,1u,1u,axis,sign);
 }
 let side=2u*axis+select(0u,1u,sign<0);let link=shWord(SH_LINKS+6u*o.tile+side);
 if(link==0u){return UMFace();}
 let width=select(4u,1u,(link&0x80000000u)!=0u);var lane=0u;var offset=0u;
 if(width==1u){
  let u=(axis+1u)%3u;let v=(axis+2u)%3u;
  offset=local[u]*stride[u]+local[v]*stride[v];lane=offset+select(0u,3u*stride[axis],sign<0);
 }
 let tiles=vec3u(1u,UM_T.x,UM_T.x*UM_T.y);let tile=u32(i32(o.tile)+sign*i32(tiles[axis]));
 let neighbor=UMOwner(tile,lane,width,(link&0x7fffffffu)-1u+offset);
 let faceWidth=min(o.width,width);let parts=o.width/faceWidth;
 return UMFace(neighbor,anchor,faceWidth,parts*parts,axis,sign);
}
fn umFaceFlags(a:UMOwner,f:UMFace)->u32 {
 if(f.neighbor.width==0u||!umSharpenFaceOpen(a,f)){return 0u;}
 let phiA=shPhi(a);let phiB=shPhi(f.neighbor);
 if(sharpen.policy.x==0.0&&sharpen.policy.y<1.5
  &&abs(phiA)>=sharpen.tuning.y*UM_MIN_H*f32(a.width)
  &&abs(phiB)>=sharpen.tuning.y*UM_MIN_H*f32(f.neighbor.width)){return 4u;}
 let middle=umSampleVertex(umFaceCenter(f));let epsilon=1e-6;
 let inwardA=phiA>=0.0&&phiB<phiA-epsilon&&middle<=phiA+epsilon&&middle>=phiB-epsilon;
 let inwardB=phiB>=0.0&&phiA<phiB-epsilon&&middle<=phiB+epsilon&&middle>=phiA-epsilon;
 let relayA=phiA>0.0&&shFill(a)<=1e-6;
 let relayB=phiB>0.0&&shFill(f.neighbor)<=1e-6;
 return 4u|select(0u,1u,(middle<=epsilon&&!relayB)||inwardA)|select(0u,2u,(middle<=epsilon&&!relayA)||inwardB);
}
// Geometry is immutable through all eight volume sweeps: admission and
// open-face bits per positive patch, in the existing word. Reusing the open
// bit avoids repeating owner-solid and face-aperture checks in every sweep.
// A listed regular owner has one
// patch per positive face (shParts).
fn shCacheGeometry(o:UMOwner){
if(o.width==0u){return;}
 for(var axis=0u;axis<3u;axis++){
  let f=shFaceFirst(o,axis,1);
  if(f.width!=0u){scratch[umCacheAt(o,f.neighbor,f)]=bitcast<f32>(umFaceFlags(o,f));}
 }
}
// A tile is listed when an owner can hold a nonzero budget (the admission
// of uvSharpenBudgets) or would be cleared as dust by an unchanged commit.
// An absent page's 4h owners are far air (V=0, corner and so centre phi at
// least 16h > the band's 2x4h reach): never admitted, never dust, and an
// orphan there holds zero budgets (V=0 gives nothing and is not receptive).
${sweep?"":/* wgsl */`@compute @workgroup_size(64) fn classify(@builtin(global_invocation_id) id:vec3u){
 let o=umResidentAllOwner(id);if(o.width==0u){return;}
 let phi=shPhi(o);let h=UM_MIN_H*f32(o.width);let band=sharpen.tuning.y;let open=umSharpenOpen(o);
 let admitted=open&&select((abs(phi)<band*h),(phi<band*h),(sharpen.policy.x>0.5));
 let orphan=sharpen.policy.y>1.5&&open&&phi>=band*h;
 let value=umV(o);let dust=value!=0.0&&abs(value)<sharpen.tuning.z&&!(value>0.0&&phi<4.0*UM_MAX_H*f32(o.width));
 if(admitted||orphan||dust){atomicStore(&work[SH_FLAGS+o.tile],1u);}
}
@compute @workgroup_size(64) fn compact(@builtin(global_invocation_id) id:vec3u){
 let tile=id.x+umDispatchX*64u*id.y;if(tile>=UM_TILES||shWord(SH_FLAGS+tile)==0u){return;}
 // One packed cross-tile link per side: low 31 bits are the first
 // neighbor index plus one, high bit marks h. Zero means unlisted/boundary.
 // The same six words serve regular tiles and coarse seam tiles.
 let tileOwner=UMOwner(tile,0u,4u,umTopology[tile]&0x3fffffffu);
 for(var side=0u;side<6u;side++){
  let face=umFaceFirst(tileOwner,side/2u,select(1,-1,(side&1u)!=0u));
  var link=0u;
  if(face.neighbor.width!=0u&&shListed(face.neighbor)){link=(face.neighbor.index+1u)|select(0u,0x80000000u,face.neighbor.width==1u);}
  atomicStore(&work[SH_LINKS+6u*tile+side],link);
 }
 let width=umTileWidth(tile);
 if(width!=1u&&umTileMaximumWidth(tile)!=umTileMinimumWidth(tile)){
  let sides=umFineFaceSides(tile);
  if(sides!=0u){
   atomicStore(&work[SH_FLAGS+tile],1u|(sides<<1u));
   let slot=atomicAdd(&work[SH_SEAMS],1u);atomicStore(&work[SH_LIST+UM_TILES-1u-slot],tile);return;
  }
 }
 let tier=shTier(width);let slot=atomicAdd(&work[SH_COUNTS+tier],1u);atomicStore(&work[shListStart(tier)+slot],tile);
}`}
// One merged launch per sweep entry: 192 regular owners per job (the tier
// lists in order), then one job per seam tile. Serial tier launches
// paid each tier's latency in every entry of every sweep. (regular, all).
fn shJobs()->vec2u{
 var owners=0u;for(var tier=0u;tier<2u;tier++){owners+=shWord(SH_COUNTS+tier)*(64u>>(6u*tier));}
 let regular=(owners+191u)/192u;return vec2u(regular,regular+shWord(SH_SEAMS));
}
var<workgroup> shJobCount:vec2u;
// Sweep jobs: 192 active regular owners, or one pack class's share of seam
// owners (shPackOwners). Geometry still uses one whole tile a seam job.
// shPackEnds: each class's last seam job (exclusive), from the first seam job.
fn shPackEnds()->vec4u{
 var ends=vec4u(0u);var total=0u;
 for(var c=0u;c<4u;c++){let owners=shPackOwners(c);total+=(shWord(SH_PACK+c)+owners-1u)/owners;ends[c]=total;}
 return ends;
}
var<workgroup> shPackEnd:vec4u;
fn shSweepJobs()->vec2u{
 let regular=(shWord(SH_ACTIVE_COUNT)+191u)/192u;return vec2u(regular,regular+shPackEnds().w);
}
// The pack word of a seam job's lane (the job counted from the first seam
// job): its class by the job, its owner by the lane's block.
fn shPackWord(job:u32,ends:vec4u,lane:u32)->u32{
 var c=0u;var first=0u;
 if(job>=ends.x){c=1u;first=ends.x;}
 if(job>=ends.y){c=2u;first=ends.y;}
 if(job>=ends.z){c=3u;first=ends.z;}
 let owners=shPackOwners(c);let slot=(job-first)*owners+lane/(192u/owners);
 if(slot>=shWord(SH_PACK+c)){return 0u;}
 return shWord(shPackAt(c,slot));
}
fn shActiveOwner(job:u32,lane:u32)->UMOwner{
 let slot=job*192u+lane;if(slot>=shWord(SH_ACTIVE_COUNT)){return UMOwner();}
 let word=shWord(SH_ACTIVE+slot);let tile=word/64u;let l=word%64u;
 return UMOwner(tile,l,umTileWidth(tile),(umTopology[tile]&0x3fffffffu)+l);
}
fn shMergedOwner(job:u32,lane:u32)->UMOwner{
 var slot=job*192u+lane;
 for(var tier=0u;tier<2u;tier++){
  let per=64u>>(6u*tier);let count=shWord(SH_COUNTS+tier)*per;
  if(slot<count){let tile=shWord(shListStart(tier)+slot/per);let l=slot%per;return UMOwner(tile,l,1u<<(2u*tier),(umTopology[tile]&0x3fffffffu)+l);}
  slot-=count;
 }
 return UMOwner();
}
// Geometry visits all 192 (anchor, component) lanes of a tile. Sweeps pack
// several coarse seam owners into the same group (shPackWord), a block of
// lanes per owner. Each owner reduces its six sides in serial patch order.
fn shSeamOwner(tile:u32,local:vec3u)->UMOwner{
 if(tile>=UM_TILES){return UMOwner();}
 let width=umTileWidth(tile);let q=local/width;let side=4u/width;let lane=q.x+side*(q.y+side*q.z);
 return UMOwner(tile,lane,width,(umTopology[tile]&0x3fffffffu)+lane);
}
// sides: the owner's h sides. base: its block's first lane.
struct SHSeamLane {owner:UMOwner,first:bool,side:u32,part:u32,sides:u32,base:u32}
const SH_SIDE_ORDER:array<u32,64> = array<u32,64>(${uniformSeamSideOrders.map(n=>`${n}u`).join(",")});
// A pack word's lane: one patch of the owner. Its block holds the h sides
// in order, sixteen lanes each, then a lane for each other side. A lane past
// them has no owner; the block's first lane also reduces. The word names the
// sides, so no lane reads the tile's flags.
fn shSeamLane(word:u32,lane:u32)->SHSeamLane{
 let sides=(word>>24u)&63u;
 if(sides==0u){return SHSeamLane(UMOwner(),false,0u,0u,0u,0u);}
 let q=lane%(192u/shPackOwners(word>>30u));let fine=countOneBits(sides);
 // The (q/16)-th h side, or the (q-16*fine)-th other side.
 let onFine=q<16u*fine;let ordinal=select(q-15u*fine,q/16u,onFine);
 if(ordinal>=6u){return SHSeamLane(UMOwner(),false,0u,0u,0u,0u);}
 let side=(SH_SIDE_ORDER[sides]>>(3u*ordinal))&7u;
 let tile=word&0xffffffu;
 return SHSeamLane(UMOwner(tile,0u,4u,umTopology[tile]&0x3fffffffu),q==0u,side,select(0u,q%16u,onFine),sides,lane-q);
}
// Where a side's first term sits in its owner's block (shSeamLane's order).
fn shSeamTermAt(sides:u32,side:u32)->u32{
 let below=(1u<<side)-1u;
 if(((sides>>side)&1u)!=0u){return 16u*countOneBits(sides&below);}
 return 16u*countOneBits(sides)+countOneBits(~sides&below);
}
// umFace's patch part of seam owner o's side. The geometry pass compiled
// its neighbor index and listed status; every sweep derives local lanes
// directly without an ownership lookup or a second tile-flag read.
fn shSeamPatch(o:UMOwner,sides:u32,side:u32,part:u32)->UMFace{
 let parts=shSideParts(sides,side);if(part>=parts){return UMFace();}
 let axis=side/2u;let low=side%2u==1u;var probe=vec3i(umOrigin(o));
 probe[axis]+=select(4,-1,low);probe[(axis+1u)%3u]+=i32(part%4u);probe[(axis+2u)%3u]+=i32(part/4u);
 var anchor=probe;anchor[axis]-=select(1,0,low);
 let width=select(4u,1u,parts==16u);let link=shWord(SH_LINKS+6u*o.tile+side);var neighbor=UMOwner();
 if(link!=0u){
  let q=vec3u(probe);let local=(q%4u)/width;let n=4u/width;let stride=vec3u(1u,4u,16u);
  let offset=select(0u,(part%4u)*stride[(axis+1u)%3u]+(part/4u)*stride[(axis+2u)%3u],parts==16u);
  neighbor=UMOwner(umTileAt(q/4u),local.x+n*(local.y+n*local.z),width,(link&0x7fffffffu)-1u+offset);
 }
 return UMFace(neighbor,anchor,width,parts,axis,select(1,-1,low));
}
// An absent patch contributes zero. Keep only its full-precision value,
// not a second float presence flag. Serial side/patch summation is unchanged.
var<workgroup> shSeamTerms:array<f32,192>;
// Each (anchor, component) lane holds the patch anchored there, if any. An
// owner that stays in the sweeps caches the patch's admission word; a quiet
// one writes the zero proposal its sweeps would (shPrepareActive).
fn shCacheGeometrySeam(tile:u32,lane:u32,stays:bool){
 if(tile>=UM_TILES){return;}
 let axis=lane/64u;let local=umCorner(lane%64u,4u);let u=(axis+1u)%3u;let v=(axis+2u)%3u;
 if(local[axis]!=3u){return;}
 let part=local[u]+4u*local[v];if(part>=shSideParts(shSides(tile),2u*axis)){return;}
 let o=shSeamOwner(tile,vec3u(0));let face=shSeamPatch(o,shSides(tile),2u*axis,part);
 if(face.neighbor.width==0u){return;}
 if(stays){scratch[umCacheAt(o,face.neighbor,face)]=bitcast<f32>(umFaceFlags(o,face));}
 else{scratch[umRawAt(o,face.neighbor,face)]=0.0;}
}
// Budgets from an owner's volume; distance and desired are fixed per frame.
fn shBudgets(o:UMOwner,value:f32,distance:f32,desired:f32)->vec2f{
 return uvSharpenBudgets(value,desired,distance,UM_MIN_H*f32(o.width),clamp(sharpen.tuning.x,0.0,1.0),sharpen.tuning.y,sharpen.policy.x>0.5,sharpen.policy.y,umSharpenOpen(o))*umMassScale(o);
}
// The sweep entries' seam functions take a pack word (shPackWord) as tile.
// A patch is proposed by its lower owner: the seam owner's high sides.
fn shProposeSeam(tile:u32,lane:u32){
 let l=shSeamLane(tile,lane);if(l.owner.width==0u||(l.side&1u)!=0u){return;}
 let o=l.owner;let face=shSeamPatch(o,l.sides,l.side,l.part);
 if(face.neighbor.width!=0u){scratch[umRawAt(o,face.neighbor,face)]=umProposal(o,face.neighbor,face);}
}
// The owner's side patch part, when limit/commit would visit it.
fn shSeamFace(o:UMOwner,sides:u32,side:u32,part:u32)->UMFace{
 let face=shSeamPatch(o,sides,side,part);
 if(face.width==0u||face.neighbor.width==0u){return UMFace();}return face;
}
fn shLimitSeam(tile:u32,lane:u32){
 let l=shSeamLane(tile,lane);var term=0.0;
 if(l.owner.width!=0u){let face=shSeamFace(l.owner,l.sides,l.side,l.part);if(face.width!=0u){term=f32(face.sign)*scratch[umRawOf(l.owner,face)];}}
 shSeamTerms[lane]=term;workgroupBarrier();
 if(!l.first){return;}
 let o=l.owner;let at=umBudgetAt(o);
 if(scratch[at]==0.0&&scratch[at+1u]==0.0){scratch[at+4u]=1.0;scratch[at+5u]=1.0;return;}
 let sides=l.sides;var outgoing=0.0;var incoming=0.0;
 for(var side=0u;side<6u;side++){let head=l.base+shSeamTermAt(sides,side);for(var part=0u;part<shSideParts(sides,side);part++){let t=shSeamTerms[head+part];outgoing+=max(t,0.0);incoming+=max(-t,0.0);}}
 scratch[at+4u]=min(1.0,scratch[at]/max(outgoing,1e-20));scratch[at+5u]=min(1.0,scratch[at+1u]/max(incoming,1e-20));
}
fn shCommitSeam(tile:u32,lane:u32){
 let l=shSeamLane(tile,lane);var term=0.0;
 if(l.owner.width!=0u){
  let o=l.owner;let at=umBudgetAt(o);let face=shSeamFace(o,l.sides,l.side,l.part);
  if(face.width!=0u&&(scratch[at]!=0.0||scratch[at+1u]!=0.0)){
   let other=umBudgetAt(face.neighbor);let raw=f32(face.sign)*scratch[umRawOf(o,face)];
   let factor=select(min(scratch[at+5u],scratch[other+4u]),min(scratch[at+4u],scratch[other+5u]),raw>=0.0);term=raw*factor;
  }
 }
 shSeamTerms[lane]=term;workgroupBarrier();
 if(!l.first){return;}
 let o=l.owner;let at=umBudgetAt(o);var terms:array<f32,6>;
 let sides=l.sides;
 for(var side=0u;side<6u;side++){var sum=0.0;let head=l.base+shSeamTermAt(sides,side);for(var part=0u;part<shSideParts(sides,side);part++){sum-=shSeamTerms[head+part];}terms[side]=sum;}
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
// An owner's prepare: whether it stays in the sweeps. A quiet owner (zero
// budgets, a volume shFinish's dust rule keeps) writes what its eight
// sweeps would: its volume and unit limit factors here, zero proposals on
// its positive patches by its caller.
fn shPrepareOwner(o:UMOwner)->bool{
 let p=vec3i(umOrigin(o));let at=umBudgetAt(o);
 let distance=shPhi(o);let desired=shFill(o);let value=umV(o);
 let budget=shBudgets(o,value,distance,desired);
 scratch[at]=budget.x;scratch[at+1u]=budget.y;scratch[at+2u]=distance;scratch[at+3u]=desired;
 let dust=value!=0.0&&abs(value)<sharpen.tuning.z&&!(value>0.0&&distance<4.0*UM_MAX_H*f32(o.width));
 if(budget.x!=0.0||budget.y!=0.0||dust){return true;}
 scratch[at+4u]=1.0;scratch[at+5u]=1.0;textureStore(output,p,vec4f(value));
 return false;
}
// A regular owner's (shPropose's faces).
fn shPrepareActive(o:UMOwner)->bool{
 if(o.width==0u){return false;}
 if(shPrepareOwner(o)){return true;}
 for(var axis=0u;axis<3u;axis++){let first=shFaceFirst(o,axis,1);if(first.neighbor.width==0u){continue;}
  for(var part=0u;part<shParts(first);part++){let face=umFacePatch(first,part);scratch[umRawAt(o,face.neighbor,face)]=0.0;}
 }
 return false;
}
// The proposal from lower owner a to upper owner b across patch f (f's
// anchor, axis and width; both incident owners derive the same ones).
fn umProposal(a:UMOwner,b:UMOwner,f:UMFace)->f32 {
 let i=umBudgetAt(a);let j=umBudgetAt(b);
 let phiA=scratch[i+2u];let phiB=scratch[j+2u];let dose=clamp(sharpen.tuning.x,0.0,1.0);
 let area=f32(f.width*f.width);let shareA=area/f32(a.width*a.width);let shareB=area/f32(b.width*b.width);
 let giveA=scratch[i]*shareA;let giveB=scratch[j]*shareB;let takeA=scratch[i+1u]*shareA;let takeB=scratch[j+1u]*shareB;
 // A zero transfer budget cannot contribute in either direction. Before the
 // orphan branch: a negative sliver's give is negative there, and min()
 // with a quiet neighbour's zero take would propose it.
 if((giveA==0.0||takeB==0.0)&&(giveB==0.0||takeA==0.0)){return 0.0;}
 // Quiet owners have no cached geometry. Only read after the zero-budget
 // rejection; every face capable of moving mass has a prepared lower owner.
 let flags=bitcast<u32>(scratch[umCacheAt(a,b,f)]);
 if((flags&4u)==0u){return 0.0;}
 if(sharpen.policy.y>1.5){
  let orphanA=phiA>=sharpen.tuning.y*UM_MIN_H*f32(a.width);let orphanB=phiB>=sharpen.tuning.y*UM_MIN_H*f32(b.width);
  if(orphanA||orphanB){let va=umV(a);let vb=umV(b);return select(0.0,min(giveA,takeB),orphanA&&vb>va)-select(0.0,min(giveB,takeA),orphanB&&va>vb);}
 }
 let epsilon=1e-6;
 var capA=giveA;var capB=giveB;
 if(sharpen.policy.x>0.5){
  if(!(phiA<0.0&&phiB<phiA-epsilon)){capA=min(capA,dose*max(umV(a)-scratch[i+3u],0.0)*umMassScale(a)*shareA);}
  if(!(phiB<0.0&&phiA<phiB-epsilon)){capB=min(capB,dose*max(umV(b)-scratch[j+3u],0.0)*umMassScale(b)*shareB);}
 }
 return select(0.0,min(capA,takeB),(flags&1u)!=0u)-select(0.0,min(capB,takeA),(flags&2u)!=0u);
}
fn shPropose(o:UMOwner){
if(o.width==0u){return;}
 for(var axis=0u;axis<3u;axis++){let first=shFaceFirst(o,axis,1);if(first.neighbor.width==0u){continue;}
  for(var part=0u;part<shParts(first);part++){let face=umFacePatch(first,part);scratch[umRawAt(o,face.neighbor,face)]=umProposal(o,face.neighbor,face);}
 }
}
fn shLimit(o:UMOwner){
if(o.width==0u){return;}var outgoing=0.0;var incoming=0.0;
 let at=umBudgetAt(o);
 if(scratch[at]==0.0&&scratch[at+1u]==0.0){scratch[at+4u]=1.0;scratch[at+5u]=1.0;return;}
 for(var axis=0u;axis<3u;axis++){for(var side=0u;side<2u;side++){
  let sign=select(1,-1,side==1u);let first=shFaceFirst(o,axis,sign);if(first.neighbor.width==0u){continue;}
  for(var part=0u;part<shParts(first);part++){let face=umFacePatch(first,part);let value=f32(sign)*scratch[umRawOf(o,face)];outgoing+=max(value,0.0);incoming+=max(-value,0.0);}
 }}
 scratch[at+4u]=min(1.0,scratch[at]/max(outgoing,1e-20));scratch[at+5u]=min(1.0,scratch[at+1u]/max(incoming,1e-20));
}
fn shCommit(o:UMOwner){
if(o.width==0u){return;}let at=umBudgetAt(o);var terms:array<f32,6>;
 if(scratch[at]!=0.0||scratch[at+1u]!=0.0){
 for(var axis=0u;axis<3u;axis++){for(var side=0u;side<2u;side++){
  let sign=select(1,-1,side==1u);let first=shFaceFirst(o,axis,sign);var sum=0.0;
  if(first.neighbor.width!=0u){for(var part=0u;part<shParts(first);part++){
   let face=umFacePatch(first,part);let other=umBudgetAt(face.neighbor);let raw=f32(sign)*scratch[umRawOf(o,face)];
   let factor=select(min(scratch[at+5u],scratch[other+4u]),min(scratch[at+4u],scratch[other+5u]),raw>=0.0);sum-=raw*factor;
  }}terms[2u*axis+side]=sum;
 }}}
 let delta=((terms[0]+terms[1])+(terms[4]+terms[5]))+(terms[2]+terms[3]);shFinish(o,umV(o)+delta/umMassScale(o));
}
// The geometry launch also prepares the first sweep's budgets, over every
// listed owner, and lists the active regular owners (one global add per job)
// and packs the active seam owners.
${sweep?"":/* wgsl */`var<workgroup> shActiveLocal:atomic<u32>;
var<workgroup> shActiveBase:u32;var<workgroup> shSeamStays:u32;
@compute @workgroup_size(192) fn cacheGeometryPrepare(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 if(lane==0u){shJobCount=shJobs();}
 let jobs=workgroupUniformLoad(&shJobCount);
 for(var job=group.x;job<jobs.y;job+=groups.x){
  if(job>=jobs.x){
   // A seam tile: its first lane prepares the owner, then a lane per patch.
   let tile=shSeamJobTile(job-jobs.x);
   if(lane==0u){
    var stays=0u;
    if(tile<UM_TILES&&shPrepareOwner(shSeamOwner(tile,vec3u(0)))){
     stays=1u;let sides=shSides(tile);let c=shPackClass(sides);
     let slot=atomicAdd(&work[SH_PACK+c],1u);atomicStore(&work[shPackAt(c,slot)],tile|(sides<<24u)|(c<<30u));
     atomicAdd(&work[SH_SEAM_WEIGHT],24u/shPackOwners(c));
    }
    shSeamStays=stays;
   }
   shCacheGeometrySeam(tile,lane,workgroupUniformLoad(&shSeamStays)!=0u);
  }
  if(lane==0u){atomicStore(&shActiveLocal,0u);}
  var o=UMOwner();var live=false;
  // A quiet owner's admission words (its positive patches') are never
  // read: umProposal returns before the flags when either owner
  // has zero budgets, which a quiet owner has in every sweep.
  if(job<jobs.x){o=shMergedOwner(job,lane);live=shPrepareActive(o);if(live){shCacheGeometry(o);}}
  workgroupBarrier();
  var slot=0u;if(live){slot=atomicAdd(&shActiveLocal,1u);}
  workgroupBarrier();
  if(lane==0u){let n=atomicLoad(&shActiveLocal);var base=0u;if(n!=0u){base=atomicAdd(&work[SH_ACTIVE_COUNT],n);}shActiveBase=base;}
  let base=workgroupUniformLoad(&shActiveBase);
  if(live){atomicStore(&work[SH_ACTIVE+base+slot],o.tile*64u+o.lane);}
  // The next job counts its active owners in the same workgroup words.
  workgroupBarrier();
 }
}`}
${(sweep?["propose","limit","commit"]:[]).map(entry=>{const fn=`sh${entry[0]!.toUpperCase()}${entry.slice(1)}`;return /* wgsl */`
@compute @workgroup_size(192) fn ${entry}(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 if(lane==0u){shJobCount=shSweepJobs();shPackEnd=shPackEnds();}
 let jobs=workgroupUniformLoad(&shJobCount);let ends=workgroupUniformLoad(&shPackEnd);
 for(var job=group.x;job<jobs.y;job+=groups.x){
  // A seam job's lane holds a pack word (zero: none), not a bare tile.
  var tile=0u;if(job>=jobs.x){tile=shPackWord(job-jobs.x,ends,lane);}
  // The GPU count partitions regular and seam jobs uniformly for the group.
  // Regular jobs need no empty seam staging or seam reduction barrier.
  if(job<jobs.x){${fn}(shActiveOwner(job,lane));}
  else{${fn}Seam(tile,lane);}
  // The next job's seam stages its terms in the same workgroup array.
  workgroupBarrier();
 }
}`;}).join("")}`,sweep?[]:["classify"]);
    const module=uniformDetailModule(this.device,{label:"Uniform mixed sharpening",code:source(false)}),sweepModule=uniformDetailModule(this.device,{label:"Uniform mixed sharpening sweep",code:source(true)});
    const errors=[...(await module.getCompilationInfo()).messages,...(await sweepModule.getCompilationInfo()).messages].filter(m=>m.type==="error");if(errors.length)throw new Error(errors.map(m=>`${m.lineNum}: ${m.message}`).join("\n"));
    const solid=this.solid?[this.solid.tileLayout]:[];
    const layout=this.device.createPipelineLayout({bindGroupLayouts:[this.ownership.bindLayout,this.resources,...solid]}),sweepLayout=this.device.createPipelineLayout({bindGroupLayouts:[this.ownership.bindLayout,this.sweepResources,...solid]});
    // One merged pipeline per entry: the listed launch covers every width.
    await Promise.all(["cacheGeometryPrepare","propose","limit","commit"].map(async entryPoint=>{const sweep=entryPoint!=="cacheGeometryPrepare";this.pipelines.set(entryPoint,[await this.twin(s=>uniformDetailPipeline(this.device,this.ownership,{layout:sweep?sweepLayout:layout,compute:{module:sweep?sweepModule:module,entryPoint,constants:{umCellWidth:1,umDispatchX:this.ownership.dispatchX,...s}}}))]);}));
    await Promise.all(["classify","compact"].map(async entryPoint=>{this.pipelines.set(entryPoint,[await this.twin(s=>uniformDetailPipeline(this.device,this.ownership,{layout,compute:{module,entryPoint,constants:{umDispatchX:this.ownership.dispatchX,...(entryPoint==="classify"?{umCountedJobs:UNIFORM_MIXED_COUNTED.residentAll}:{}),...s}}}))]);}));
  }
  encodeGeometry(encoder:GPUCommandEncoder,group:UniformDetailGroup):void{
    if(this.pipelines.size!==6)throw new Error("Mixed sharpening is not initialized");
    const begin=()=>{const pass=encoder.beginComputePass({label:"Uniform mixed sharpening geometry"});pass.setBindGroup(0,this.ownership.bindGroup);pass.setBindGroup(1,group.group);if(this.solid)pass.setBindGroup(2,this.solid.tileGroup);return pass;};
    {
      const tiles=this.ownership.capacity.tiles,groups=Math.ceil(tiles/64),dx=this.ownership.dispatchX;
      encoder.clearBuffer(this.work.list,0,4*(12+tiles));
      const list=begin();this.ownership.dispatchAllCounted(list,this.variant(this.pipelines.get("classify")![0]!));
      list.setPipeline(this.variant(this.pipelines.get("compact")![0]!));list.dispatchWorkgroups(Math.min(groups,dx),Math.ceil(groups/dx));list.end();
    }
    const pass=begin();this.dispatchEntry(pass,"cacheGeometryPrepare");pass.end();
  }
  /** `sweeps` sweeps over the geometry encodeGeometry cached, alternating
   * the ping-pong groups from groups[0], in one pass. */
  encodeSweeps(encoder:GPUCommandEncoder,groups:readonly [UniformDetailGroup,UniformDetailGroup],sweeps:number):void{
    if(this.pipelines.size!==6)throw new Error("Mixed sharpening is not initialized");
    // An odd count would leave the result in groups[0]'s output, the scratch.
    if(!Number.isInteger(sweeps)||sweeps<2||sweeps%2!==0)throw new Error(`Mixed sharpening needs an even sweep count, not ${sweeps}`);
    const pass=encoder.beginComputePass({label:"Uniform mixed geometric sharpening"});pass.setBindGroup(0,this.ownership.bindGroup);if(this.solid)pass.setBindGroup(2,this.solid.tileGroup);
    const sweepGroups=groups.map(g=>{const twin=this.sweepGroups.get(g);if(!twin)throw new Error("Mixed sharpening sweeps need groups from bind()");return twin;});
    for(let sweep=0;sweep<sweeps;sweep++){pass.setBindGroup(1,sweepGroups[sweep%2]!.group);for(const entry of ["propose","limit","commit"])this.dispatchEntry(pass,entry);}
    pass.end();
  }
  /** The merged launch: regular owners, then coarse seam tiles. */
  private twin(create:(solid:Record<string,number>)=>Promise<GPUComputePipeline>):Promise<GPUComputePipeline>{return uniformMixedSolidPipeline(this.solid,create);}
  private variant(pipeline:GPUComputePipeline):GPUComputePipeline{return uniformDetailPick(this.solid?.select(pipeline)??pipeline);}
  private dispatchEntry(pass:GPUComputePassEncoder,entry:string):void{
    this.ownership.dispatchBuffered(pass,this.variant(this.pipelines.get(entry)![0]!),"sharpen",entry==="cacheGeometryPrepare"?this.grid:this.sweepWork);
  }
}
