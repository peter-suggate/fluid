import { uniformDetailModule, uniformDetailPick, uniformDetailPipeline } from "./uniform-detail-fields";
import type { UniformMixedOwnership } from "./uniform-mixed-ownership";
import { uniformMixedTopologyWGSL } from "./uniform-mixed-topology.wgsl";
import { UNIFORM_REGION_TIER, uniformRegionHeatWGSL } from "./uniform-region-heat";

/** Per-16h-region dynamics record (region = umPageOfTile(tile), UM_PAGES of
 * them): what a frame's pressure did to each region, for the next
 * consumers to wake on, and the force tier the frame's forces ran.
 *
 * Words per region (UNIFORM_REGION_DYNAMICS_WORDS), u32 unless noted:
 *   0 tier       this frame's force tier (UNIFORM_FORCE_TIER).
 *   1 key        own geometry key (topology width bits, block word), low
 *                word. Liquid motion is not keyed: only the viscous term
 *                is retained, and its budget bounds it from the speeds.
 *   2 hood       neighbourhood key (27 regions' keys, force parameters,
 *                host generation) at the last classify, low word.
 *   3 budget     f32: viscous velocity dropped since the last full forces
 *                evaluation (m/s), bounded by UNIFORM_FORCE_RETAIN_BUDGET.
 *   4 lastFull   the frame stamp of the last full forces evaluation.
 *   5 vmax       f32: the 27-region extended speed bound the viscous budget used.
 *   6 dp         f32: h times the change in root pressure gradient (Pa),
 *                bounded over the region's 64 tiles, published after the
 *                band's projection (+Inf where unknown: a tile not published
 *                the frame before, a free region). Current from the publish
 *                to the next frame's publish: a frame-head consumer reads
 *                the previous frame's change.
 *                A distant impact reaches here the frame it happens, before
 *                any velocity changes.
 *   7 residual   f32: the largest h band row residual (1/s) of this frame's
 *                predictor (the band's initial measure: the prolonged 4h
 *                pressure before any band smoothing) over the region's
 *                liquid band rows; 0 where the band holds none, +Inf for a
 *                free region. Cleared at classify, current from the band
 *                solve to the next frame's classify.
 *   8 hood high word, 9 key high word.
 *
 * Binding for readers: uniformMixedRegionDynamicsWGSL. The record buffer is
 * fixed for the frame's lifetime (never replaced). */
export const UNIFORM_REGION_DYNAMICS_WORDS=10;
export const UNIFORM_REGION_DYNAMICS={tier:0,key:1,hood:2,budget:3,lastFull:4,vmax:5,pressureDelta:6,pressureResidual:7,hoodHigh:8,keyHigh:9} as const;
/** Retain: the forces drop the viscous term, within the region's budget.
 * Bounded: every term exact, the budget kept. Full: every term exact, the
 * budget reset (a geometry, parameter or body change in the region's
 * neighbourhood). The region work tier (UniformRegionHeat) decides drive
 * reuse (UNIFORM_FORCE_MODE); a full tier ends it. */
export const UNIFORM_FORCE_TIER={retain:0,bounded:1,full:2} as const;
/** Dropped viscous velocity a region may accumulate between full
 * evaluations, m/s: a bound, from |lap u| <= 4 vmax sum(1/h^2). */
export const UNIFORM_FORCE_RETAIN_BUDGET=1e-6;
/** Reach of the forces' dependency footprint, in h: the swept-body margin a
 * region's retained forces take (stencil 16h, record rebuild 2h). */
export const UNIFORM_FORCE_BODY_MARGIN=18;
/** Body pose reach (UniformMixedSolid.bodyRegions bits 8-15, h) past which
 * a region's faces carry no moving-wall flux: umSolidFaceVelocity is zero
 * beyond 2 hmax of a body's bounding sphere, and a cell's faces lie within
 * 1h of its region's box. */
export const UNIFORM_BODY_FLUX_REACH=3;
/** The forces' store (UniformMixedRegionDynamics.forceStore), u32 words:
 * a mirror of UNIFORM_REGION_MIRROR_WORDS per region, then a 4-word entry
 * per force slot (UNIFORM_FORCE_STORE_FINE_TILES*64 h owner indices, then
 * one per tile key for its 4h owner): the drive kept for the owner's three
 * +faces (f32 velocity increments of viscosity, bed and capillarity; gravity
 * and inflow are applied fresh) and a tag of the owner's tile and the
 * region's keeping stamp; then UNIFORM_FORCE_STORE_SEAM_TILES seam 4h
 * entries by seam list position, 16 patches by 3 axes and the tag.
 * A tag that does not match (a relayout moved the owner) evaluates the
 * owner without keeping and unprimes its region, which keeps anew the
 * next frame: no lane writes an entry another lane may be reading.
 * Mirror words:
 *   tier        the dynamics tier (UNIFORM_FORCE_TIER), classify
 *   stamp       classify's frame stamp
 *   heat        the region's work tier (UNIFORM_REGION_TIER), classify
 *   mode        this frame's force mode (UNIFORM_FORCE_MODE), classify
 *   primed      1 while the region's kept drives are current
 *   storeStamp  the stamp of the frame that kept them (in every tag). */
export const UNIFORM_REGION_MIRROR={tier:0,stamp:1,heat:2,mode:3,primed:4,storeStamp:5} as const;
export const UNIFORM_REGION_MIRROR_WORDS=6;
/** A region's force mode bits. reuse: a COLD region whose drives are
 * current applies them (gravity and inflow fresh) and skips its retained
 * normals and curvature. store: a COLD region evaluates and keeps them.
 * viscous: the viscous term is evaluated (WARM regions retaining within
 * their budget drop it). normalsExact: every retained normal and curvature
 * is evaluated (leaving reuse: revisions were not tracked meanwhile). */
export const UNIFORM_FORCE_MODE={reuse:1,store:2,viscous:4,normalsExact:16} as const;
/** h tiles whose owners hold force slots (the retained normals' bound). */
export const UNIFORM_FORCE_STORE_FINE_TILES=8192;
/** Seam 4h tiles (by seam list position) whose patches hold force slots. */
export const UNIFORM_FORCE_STORE_SEAM_TILES=8192;
const SEAM_WORDS=49;
/** The force store's bytes (UniformMixedRegionDynamics.forceStore). */
export function uniformRegionForceStoreBytes(regions:number,fineSlots:number,tiles:number):number{
  return 4*(UNIFORM_REGION_MIRROR_WORDS*regions+4*(fineSlots+tiles)+SEAM_WORDS*UNIFORM_FORCE_STORE_SEAM_TILES);
}

/** The forces' store at (group, binding), read-write (needs the topology WGSL):
 *   fn umForceMode(region)->u32          UNIFORM_FORCE_MODE bits
 *   fn umViscosityExact(owner)->bool
 *   fn umForceRetain(owner)->vec3u       (0 evaluate, 1 evaluate and keep, 2 reuse;
 *                                         the +x drive's word; the tag's word)
 *   fn umForceRetainSeam(owner,seam,part)->vec3u  a seam 4h owner's patch
 *   fn umForceKept(retain)->vec3f, umForceKeep(owner,retain,axis,drive)
 * Classify sets each region's mode (umRdForce) before the forces run. */
export function uniformRegionForceWGSL(group:number,binding:number,regions:number,fineSlots:number,tiles:number):string{
  const M=UNIFORM_REGION_MIRROR_WORDS,W=UNIFORM_REGION_MIRROR,F=UNIFORM_FORCE_MODE,T=UNIFORM_FORCE_TIER,H=UNIFORM_REGION_TIER;
  return /* wgsl */`
@group(${group}) @binding(${binding}) var<storage,read_write> umForceStore:array<u32>;
const UM_FORCE_REGIONS:u32=${regions}u;
const UM_FORCE_FINE_SLOTS:u32=${fineSlots}u;
const UM_FORCE_ENTRIES:u32=${M*regions}u;
const UM_FORCE_NO_SLOT:u32=0xffffffffu;
fn umForceMode(region:u32)->u32{if(region>=UM_FORCE_REGIONS){return ${F.viscous|F.normalsExact}u;}return umForceStore[${M}u*region+${W.mode}u];}
fn umViscosityExact(owner:UMOwner)->bool{return (umForceMode(umPageOfTile(owner.tile))&${F.viscous}u)!=0u;}
fn umForceTag(owner:UMOwner)->u32{
 let stamp=umForceStore[${M}u*umPageOfTile(owner.tile)+${W.storeStamp}u];
 var v=(owner.tile^select(0u,0x80000000u,owner.width!=1u)^(stamp*0x9e3779b1u))*0x7feb352du;
 v=(v^(v>>15u))*0x846ca68bu;return v^(v>>16u);
}
fn umForceRetainAt(owner:UMOwner,drive:u32,tag:u32)->vec3u{
 let region=umPageOfTile(owner.tile);let mode=umForceMode(region);
 if(drive==UM_FORCE_NO_SLOT||(mode&${F.reuse|F.store}u)==0u){return vec3u(0u);}
 if((mode&${F.reuse}u)==0u){return vec3u(1u,drive,tag);}
 if(umForceStore[tag]==umForceTag(owner)){return vec3u(2u,drive,tag);}
 umForceStore[${M}u*region+${W.primed}u]=0u;return vec3u(0u);
}
fn umForceRetain(owner:UMOwner)->vec3u{
 var slot=UM_FORCE_FINE_SLOTS+owner.tile;
 if(owner.width==1u){slot=select(UM_FORCE_NO_SLOT,owner.index,owner.index<UM_FORCE_FINE_SLOTS);}
 if(slot==UM_FORCE_NO_SLOT){return vec3u(0u);}
 let at=UM_FORCE_ENTRIES+4u*slot;return umForceRetainAt(owner,at,at+3u);
}
fn umForceRetainSeam(owner:UMOwner,seam:u32,part:u32)->vec3u{
 if(seam>=${UNIFORM_FORCE_STORE_SEAM_TILES}u){return vec3u(0u);}
 let at=UM_FORCE_ENTRIES+${4*(fineSlots+tiles)}u+${SEAM_WORDS}u*seam;return umForceRetainAt(owner,at+3u*part,at+48u);
}
fn umForceKept(retain:vec3u)->vec3f{let at=retain.y;return bitcast<vec3f>(vec3u(umForceStore[at],umForceStore[at+1u],umForceStore[at+2u]));}
fn umForceKeep(owner:UMOwner,retain:vec3u,axis:u32,drive:f32){umForceStore[retain.y+axis]=bitcast<u32>(drive);umForceStore[retain.z]=umForceTag(owner);}
`;}

export interface UniformRegionDynamicsSources {
  /** The forces' params uniform (UniformMixedForceFields.params), 48 bytes. */
  forceParams:GPUBuffer;
  /** UniformMixedSolid.bodyRegions, when solids are compiled in. */
  bodyRegions?:{buffer:GPUBuffer;regions:number};
  /** The accepted all-4h root pressure, indexed by simulation tile key
   * (the band's coarsePressure). Fixed for the frame's lifetime. */
  pressure:GPUBufferBinding;
  /** The region work tiers (UniformRegionHeat.tiers), written before
   * classify: classify copies each region's into the force mirror. */
  heat:GPUBuffer;
  /** UniformMixedSurfaceGeometry.curvatureDirty (the forces' normalDirty):
   * its topology plane, written before classify. */
  normalDirty?:GPUBuffer;
}

/** The pressure publish's fixed grid: 64-tile groups striding the tiles. */
const PRESSURE_GROUPS=4096;
const CLASSIFY_PASS:GPUComputePassDescriptor={label:"Uniform region dynamics classify"};
const PUBLISH_PASS:GPUComputePassDescriptor={label:"Uniform region dynamics publish"};

/** The per-region dynamics record and the forces' tiers. */
export class UniformMixedRegionDynamics {
  readonly regions:number;
  /** Fixed for the frame's lifetime: readers bind it once. */
  readonly record:GPUBuffer;
  /** The forces' mirror and kept drives (fixed: the forces bind it once). */
  readonly forceStore:GPUBuffer;
  /** h owner indices below this hold a force slot. */
  readonly forceFineSlots:number;
  private readonly params:GPUBuffer;
  /** Per tile: the last published root pressure and its frame stamp. */
  private readonly pressureHistory:GPUBuffer;
  private readonly paramWords=new Uint32Array(4);
  private stamp=0;
  private generation=0;
  private readonly classifyLayout:GPUBindGroupLayout;
  private readonly publishLayout:GPUBindGroupLayout;
  private classifyGroup?:GPUBindGroup;
  private publishGroup?:GPUBindGroup;
  private keyPipeline?:GPUComputePipeline;
  private decidePipeline?:GPUComputePipeline;
  private beginPipeline?:GPUComputePipeline;
  private pressurePipeline?:GPUComputePipeline;
  private rememberPressurePipeline?:GPUComputePipeline;
  constructor(private readonly device:GPUDevice,readonly ownership:UniformMixedOwnership,private readonly sources:UniformRegionDynamicsSources){
    this.regions=ownership.capacity.tiles/64;
    if(!Number.isInteger(this.regions))throw new Error(`Region dynamics: ${ownership.capacity.tiles} tiles are not whole regions`);
    if(sources.bodyRegions&&sources.bodyRegions.regions!==this.regions)throw new Error("Region dynamics: body mask region count differs");
    if(this.regions>device.limits.maxComputeWorkgroupsPerDimension)throw new Error(`Region dynamics: regionKey launches a workgroup a region; ${this.regions} regions exceed ${device.limits.maxComputeWorkgroupsPerDimension}`);
    const words=UNIFORM_REGION_DYNAMICS_WORDS*this.regions;
    this.record=device.createBuffer({label:"Uniform region dynamics record",size:4*words,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST,mappedAtCreation:true});
    {const r=new Uint32Array(this.record.getMappedRange());const W=UNIFORM_REGION_DYNAMICS;
     for(let i=0;i<this.regions;i++){const at=UNIFORM_REGION_DYNAMICS_WORDS*i;r[at+W.pressureDelta]=0x7f800000;r[at+W.tier]=UNIFORM_FORCE_TIER.full;r[at+W.hood]=r[at+W.hoodHigh]=0xffffffff;}
     this.record.unmap();}
    this.params=device.createBuffer({label:"Uniform region dynamics params",size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
    const tiles=ownership.capacity.tiles;
    if((sources.pressure.size??sources.pressure.buffer.size-(sources.pressure.offset??0))<4*tiles)throw new Error("Region dynamics: the root pressure view holds fewer words than tiles");
    // Zero stamps: no tile has a previous pressure.
    this.pressureHistory=device.createBuffer({label:"Uniform region dynamics pressure history",size:8*tiles,usage:GPUBufferUsage.STORAGE});
    // All ones: no tag matches, no region is primed, tiers read exact until classify.
    this.forceFineSlots=64*Math.min(ownership.capacity.address.poolTiles,UNIFORM_FORCE_STORE_FINE_TILES);
    const mirror=UNIFORM_REGION_MIRROR_WORDS*this.regions;
    this.forceStore=device.createBuffer({label:"Uniform region force store",size:uniformRegionForceStoreBytes(this.regions,this.forceFineSlots,tiles),usage:GPUBufferUsage.STORAGE,mappedAtCreation:true});
    {const w=new Uint32Array(this.forceStore.getMappedRange());w.fill(0xffffffff);for(let i=0;i<this.regions;i++)w[UNIFORM_REGION_MIRROR_WORDS*i+UNIFORM_REGION_MIRROR.primed]=0;this.forceStore.unmap();}
    const storage=(binding:number,type:GPUBufferBindingType)=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type}});
    this.classifyLayout=device.createBindGroupLayout({label:"Uniform region dynamics classify",entries:[
      storage(0,"storage"),storage(1,"storage"),storage(3,"read-only-storage"),storage(7,"read-only-storage"),
      {binding:4,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},{binding:5,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},
      ...(sources.bodyRegions?[storage(6,"read-only-storage")]:[]),...(sources.normalDirty?[storage(8,"read-only-storage")]:[])]});
    this.publishLayout=device.createBindGroupLayout({label:"Uniform region dynamics publish",entries:[
      storage(0,"storage"),storage(2,"read-only-storage"),storage(3,"storage"),{binding:4,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}}]});
  }
  /** Bytes held: the record, the force store, the params and the pressure history. */
  get allocatedBytes():number{return this.record.size+this.forceStore.size+this.params.size+this.pressureHistory.size;}

  /** Host-side change the keys cannot see (a static solid edit, a scene
   * edit): every region evaluates its forces in full next frame. */
  invalidate():void{this.generation=(this.generation+1)>>>0;}
  async initialize():Promise<void>{
    const o=this.ownership,regions=this.regions,R=UNIFORM_REGION_DYNAMICS,W=UNIFORM_REGION_DYNAMICS_WORDS,M=UNIFORM_REGION_MIRROR_WORDS;
    const h=o.capacity.lattice.cellSize_m,inverse=h.reduce((s,n)=>s+1/(n*n),0);
    const common=uniformMixedTopologyWGSL(o.capacity,0)+/* wgsl */`
const UM_RD_REGIONS:u32=${regions}u;
struct UMRDParams{stamp:u32,generation:u32,unused0:u32,unused1:u32}
@group(1) @binding(4) var<uniform> rd:UMRDParams;
fn umRdMix(h:u32,x:u32)->u32{var v=(h^x)*0x9e3779b1u;v=(v^(v>>16u))*0x7feb352du;v=(v^(v>>15u))*0x846ca68bu;return v^(v>>16u);}
fn umRdStored(region:u32)->bool{let b=region>>3u;return umBlockStored(b)&&umBlockWord(b)!=0xffffffffu;}
`;
    const classify=common+/* wgsl */`
@group(1) @binding(0) var<storage,read_write> record:array<u32>;
@group(1) @binding(1) var<storage,read_write> mirror:array<u32>;
@group(1) @binding(3) var<storage,read> regionVelocity:array<u32>;
${uniformRegionHeatWGSL(1,7,regions)}
${this.sources.normalDirty?`@group(1) @binding(8) var<storage,read> normalDirty:array<u32>;
// A surface topology change under capillarity (UniformMixedSurfaceGeometry's
// topology plane; without capillarity it accumulates for the normals).
fn umRdTopology(region:u32)->bool{return force.physical.w>0.0&&normalDirty[3u*UM_PAGES+region]!=0u;}`:"fn umRdTopology(region:u32)->bool{return false;}"}
// A COLD region whose neighbourhood is unchanged (dynamics tier not full,
// no topology change) keeps its drives the first frame and reuses them
// after; anything else (HOT, WARM, a changed COLD region) evaluates as
// before the tiers: revision-filtered retained normals against their fixed
// anchors, the viscous budget. Leaving reuse evaluates every retained
// normal and curvature once.
fn umRdForce(region:u32,stored:bool){
 let topology=umRdTopology(region);
 let at=${UNIFORM_REGION_MIRROR_WORDS}u*region;let dynamics=mirror[at+${UNIFORM_REGION_MIRROR.tier}u];let heat=mirror[at+${UNIFORM_REGION_MIRROR.heat}u];let last=mirror[at+${UNIFORM_REGION_MIRROR.mode}u];
 let changed=!stored||dynamics==${UNIFORM_FORCE_TIER.full}u||topology;
 var mode=0u;
 if(heat==${UNIFORM_REGION_TIER.cold}u&&!changed){
  if(mirror[at+${UNIFORM_REGION_MIRROR.primed}u]==1u){mode=${UNIFORM_FORCE_MODE.reuse|UNIFORM_FORCE_MODE.viscous}u;}
  else{mode=${UNIFORM_FORCE_MODE.store|UNIFORM_FORCE_MODE.viscous}u;mirror[at+${UNIFORM_REGION_MIRROR.primed}u]=1u;mirror[at+${UNIFORM_REGION_MIRROR.storeStamp}u]=mirror[at+${UNIFORM_REGION_MIRROR.stamp}u];}
 }else{
  mirror[at+${UNIFORM_REGION_MIRROR.primed}u]=0u;
  if(dynamics!=${UNIFORM_FORCE_TIER.retain}u){mode|=${UNIFORM_FORCE_MODE.viscous}u;}
  if((last&${UNIFORM_FORCE_MODE.reuse}u)!=0u){mode|=${UNIFORM_FORCE_MODE.normalsExact}u;}
 }
 mirror[at+${UNIFORM_REGION_MIRROR.mode}u]=mode;
}
struct UMForceParams {hDt:vec4f,physical:vec4f,controls:vec4f,gravity:vec4f}
@group(1) @binding(5) var<uniform> force:UMForceParams;
${this.sources.bodyRegions?`@group(1) @binding(6) var<storage,read> bodyRegions:array<u32>;
fn umRdBodySwept(region:u32)->u32{return bodyRegions[region]&0xffu;}`:"fn umRdBodySwept(region:u32)->u32{return 255u;}"}
// The region's own geometry key: block word (its page), every tile's width
// bits. One workgroup per region, one lane per tile: each lane ORs its two
// width bits into the region's four 16-tile words (tile i at bit
// 2*(15-(i&15)) of word i>>4, the order a serial walk shifts them in), and
// lane 0 mixes them. An unstored region exits the whole workgroup.
var<workgroup> umRdWidths:array<atomic<u32>,4>;
@compute @workgroup_size(64) fn regionKey(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) i:u32){
 let region=wid.x;
 if(!umRdStored(region)){
  if(i==0u){record[${W}u*region+${R.key}u]=0x01234567u;record[${W}u*region+${R.keyHigh}u]=0x89abcdefu;record[${W}u*region+${R.pressureResidual}u]=0x7f800000u;}
  return;
 }
 if(i<4u){atomicStore(&umRdWidths[i],0u);}
 workgroupBarrier();
 let block=region>>3u;let o=vec3u(region&1u,(region>>1u)&1u,(region>>2u)&1u)*4u;
 let l=o+vec3u(i&3u,(i>>2u)&3u,i>>4u);let t=512u*block+(l.x|(l.y<<3u)|(l.z<<6u));
 atomicOr(&umRdWidths[i>>4u],(umTopology[t]>>30u)<<(2u*(15u-(i&15u))));
 workgroupBarrier();
 if(i!=0u){return;}
 let word=umBlockWord(block);var a=umRdMix(0x01234567u,word);var b=umRdMix(0x89abcdefu^0x5bd1e995u,word);
 for(var k=0u;k<4u;k++){let widths=atomicLoad(&umRdWidths[k]);a=umRdMix(a,widths);b=umRdMix(b,widths^0x165667b1u);}
 record[${W}u*region+${R.key}u]=a;record[${W}u*region+${R.keyHigh}u]=b;
 // The band's initial measure fills it this frame (UniformPressureBand).
 record[${W}u*region+${R.pressureResidual}u]=0u;
}
fn umRdDecode(word:u32)->f32{
 if(word==0u){return 0.0;}if(word==0xffffffffu){return 3.402823e38;}
 return abs(bitcast<f32>(select(~word,word^0x80000000u,(word&0x80000000u)!=0u)));
}
// The neighbourhood key over 27 regions, the forces' parameters and the
// host generation; a change or a moved body within the forces' footprint
// takes the full tier. Otherwise the viscous budget picks retain or bounded.
@compute @workgroup_size(64) fn regionDecide(@builtin(global_invocation_id) gid:vec3u){
 let region=gid.x;if(region>=UM_RD_REGIONS){return;}
 let at=${W}u*region;
 if(!umRdStored(region)){
  record[at+${R.tier}u]=${UNIFORM_FORCE_TIER.full}u;record[at+${R.hood}u]=0xffffffffu;record[at+${R.hoodHigh}u]=0xffffffffu;
  mirror[${M}u*region]=${UNIFORM_FORCE_TIER.full}u;mirror[${M}u*region+1u]=rd.stamp;mirror[${M}u*region+2u]=umRegionTier(region);umRdForce(region,false);return;
 }
 let origin=vec3i(4u*umPageCoord(region));
 var a=umRdMix(0x31415926u,rd.generation);var b=umRdMix(0x27182818u,rd.generation);var vmax=0.0;
 for(var d=0u;d<27u;d++){
  let offset=vec3i(i32(d%3u),i32((d/3u)%3u),i32(d/9u))-vec3i(1);
  let tile=umTileKey(origin+4*offset);
  if(umKeyStored(tile)){
   let n=umPageOfTile(tile);a=umRdMix(a,record[${W}u*n+${R.key}u]);b=umRdMix(b,record[${W}u*n+${R.keyHigh}u]);
   for(var k=0u;k<6u;k++){vmax=max(vmax,umRdDecode(regionVelocity[6u*n+k]));}
  }else{let page=umPageWord((origin+4*offset)>>vec3u(3u));a=umRdMix(a,page^d);b=umRdMix(b,page+d);}
 }
 let p=array<vec4f,4>(force.hDt,force.physical,force.controls,force.gravity);
 for(var i=0u;i<4u;i++){for(var c=0u;c<4u;c++){let w=bitcast<u32>(p[i][c]);a=umRdMix(a,w);b=umRdMix(b,w^0x9e3779b9u);}}
 let changed=a!=record[at+${R.hood}u]||b!=record[at+${R.hoodHigh}u]||umRdBodySwept(region)<=${UNIFORM_FORCE_BODY_MARGIN}u;
 record[at+${R.hood}u]=a;record[at+${R.hoodHigh}u]=b;
 var tier=${UNIFORM_FORCE_TIER.full}u;var budget=0.0;var last=record[at+${R.lastFull}u];
 if(changed){last=rd.stamp&0x0fffffffu;}
 else{
  budget=bitcast<f32>(record[at+${R.budget}u]);
  let nu=select(0.0,force.physical.z/force.physical.y,force.physical.z>0.0);
  let bound=force.hDt.w*nu*4.0*vmax*${inverse};
  if(budget+bound<=${UNIFORM_FORCE_RETAIN_BUDGET}){tier=${UNIFORM_FORCE_TIER.retain}u;budget+=bound;}
  else{tier=${UNIFORM_FORCE_TIER.bounded}u;}
 }
 record[at+${R.tier}u]=tier;record[at+${R.budget}u]=bitcast<u32>(budget);record[at+${R.lastFull}u]=last;record[at+${R.vmax}u]=bitcast<u32>(vmax);
 mirror[${M}u*region]=tier;mirror[${M}u*region+1u]=rd.stamp;mirror[${M}u*region+2u]=umRegionTier(region);
 umRdForce(region,true);
}
`;
    const publish=common+uniformPressureGradientChangeWGSL+/* wgsl */`
@group(1) @binding(0) var<storage,read_write> record:array<atomic<u32>>;
@compute @workgroup_size(64) fn publishBegin(@builtin(global_invocation_id) gid:vec3u){
 let region=gid.x;if(region>=UM_RD_REGIONS){return;}
 let unknown=select(0x7f800000u,0u,umRdStored(region));
 atomicStore(&record[${W}u*region+${R.pressureDelta}u],unknown);
}
@group(1) @binding(2) var<storage,read> pressure:array<f32>;
@group(1) @binding(3) var<storage,read_write> pressureHistory:array<u32>;
// One lane per tile key: the root pressure's change since the last publish.
// A fixed grid strides the tiles; a free region's 64 tiles exit together.
@compute @workgroup_size(64) fn publishPressure(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 for(var base=64u*group.x;base<UM_TILES;base+=64u*groups.x){
  let t=base+lane;let region=umPageOfTile(t);
  if(t<UM_TILES&&umRdStored(region)){umRdPressure(t,region);}
 }
}
fn umRdPressure(t:u32,region:u32){
 let now=pressure[t];let previous=select(rd.stamp-1u,0x0fffffffu,rd.stamp<=1u);
 var known=pressureHistory[2u*t+1u]==previous;var delta=0.0;
 let change=now-bitcast<f32>(pressureHistory[2u*t]);let c=vec3i(umTileCoord(t));
 // Pressure offsets are a gauge, not acceleration. Bound the change of
 // face pressure gradients; root neighbours are four base cells apart.
 for(var axis=0u;axis<3u;axis++){
  var edge=0.0;
  for(var side=0u;side<2u;side++){
   var d=vec3i(0);d[axis]=select(-1,1,side!=0u);let n=umTileKey(c+d);
   if(!umKeyStored(n)){continue;}
   known=known&&pressureHistory[2u*n+1u]==previous;
   let neighbour=pressure[n]-bitcast<f32>(pressureHistory[2u*n]);
   edge=max(edge,umPressureGradientChange(change,neighbour));
  }
  delta+=edge;
 }
 let bits=select(0x7f800000u,bitcast<u32>(delta),known&&delta<=3.402823e38);
 let at=${W}u*region+${R.pressureDelta}u;
 if(bits>atomicLoad(&record[at])){atomicMax(&record[at],bits);}
}
// A separate dispatch is essential: every gradient above must see the same
// old pressure, including across workgroups and region boundaries.
@compute @workgroup_size(64) fn rememberPressure(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 for(var t=64u*group.x+lane;t<UM_TILES;t+=64u*groups.x){
  if(!umRdStored(umPageOfTile(t))){continue;}
  pressureHistory[2u*t]=bitcast<u32>(pressure[t]);pressureHistory[2u*t+1u]=rd.stamp;
 }
}
`;
    const classifyModule=uniformDetailModule(this.device,{label:"Uniform region dynamics classify",code:classify});
    const publishModule=uniformDetailModule(this.device,{label:"Uniform region dynamics publish",code:publish});
    for(const m of [classifyModule,publishModule]){const errors=(await m.getCompilationInfo()).messages.filter(x=>x.type==="error");if(errors.length)throw new Error(errors.map(x=>`${x.lineNum}: ${x.message}`).join("\n"));}
    const classifyLayout=this.device.createPipelineLayout({bindGroupLayouts:[o.bindLayout,this.classifyLayout]});
    const publishLayout=this.device.createPipelineLayout({bindGroupLayouts:[o.bindLayout,this.publishLayout]});
    const compile=(module:GPUShaderModule,layout:GPUPipelineLayout,entryPoint:string,constants:Record<string,number>={})=>uniformDetailPipeline(this.device,o,{layout,compute:{module,entryPoint,constants:{umDispatchX:o.dispatchX,...constants}}});
    [this.keyPipeline,this.decidePipeline,this.beginPipeline,this.pressurePipeline,this.rememberPressurePipeline]=await Promise.all([
      compile(classifyModule,classifyLayout,"regionKey"),compile(classifyModule,classifyLayout,"regionDecide"),
      compile(publishModule,publishLayout,"publishBegin"),compile(publishModule,publishLayout,"publishPressure"),compile(publishModule,publishLayout,"rememberPressure")]);
  }
  private groups():{classify:GPUBindGroup;publish:GPUBindGroup}{
    const s=this.sources;
    this.classifyGroup??=this.device.createBindGroup({label:"Uniform region dynamics classify",layout:this.classifyLayout,entries:[
      {binding:0,resource:{buffer:this.record}},{binding:1,resource:{buffer:this.forceStore}},{binding:7,resource:{buffer:s.heat}},
      {binding:3,resource:{buffer:this.ownership.regionVelocity}},
      {binding:4,resource:{buffer:this.params}},{binding:5,resource:{buffer:s.forceParams,size:64}},
      ...(s.bodyRegions?[{binding:6,resource:{buffer:s.bodyRegions.buffer}}]:[]),...(s.normalDirty?[{binding:8,resource:{buffer:s.normalDirty}}]:[])]});
    this.publishGroup??=this.device.createBindGroup({label:"Uniform region dynamics publish",layout:this.publishLayout,entries:[
      {binding:0,resource:{buffer:this.record}},{binding:2,resource:s.pressure},{binding:3,resource:{buffer:this.pressureHistory}},
      {binding:4,resource:{buffer:this.params}}]});
    return {classify:this.classifyGroup,publish:this.publishGroup};
  }
  /** Frame head, after the region velocity certificate, the surface
   * body mask, before the forces: each region's keys and
   * its force tier for this frame. */
  encodeClassify(encoder:GPUCommandEncoder):void{
    if(!this.decidePipeline)throw new Error("Region dynamics are not initialized");
    this.stamp=(this.stamp+1)&0x0fffffff;if(this.stamp===0)this.stamp=1;
    this.paramWords[0]=this.stamp;this.paramWords[1]=this.generation;
    this.device.queue.writeBuffer(this.params,0,this.paramWords);
    const pass=encoder.beginComputePass(CLASSIFY_PASS),groups=Math.ceil(this.regions/64);
    pass.setBindGroup(0,this.ownership.bindGroup);pass.setBindGroup(1,this.groups().classify);
    pass.setPipeline(uniformDetailPick(this.keyPipeline!));pass.dispatchWorkgroups(this.regions);
    pass.setPipeline(uniformDetailPick(this.decidePipeline));pass.dispatchWorkgroups(groups);
    pass.end();
  }
  /** After the band's projection: each region's root pressure change. */
  encodePublish(encoder:GPUCommandEncoder):void{
    if(!this.pressurePipeline)throw new Error("Region dynamics are not initialized");
    const pass=encoder.beginComputePass(PUBLISH_PASS);
    pass.setBindGroup(0,this.ownership.bindGroup);pass.setBindGroup(1,this.groups().publish);
    pass.setPipeline(uniformDetailPick(this.beginPipeline!));pass.dispatchWorkgroups(Math.ceil(this.regions/64));
    pass.setPipeline(uniformDetailPick(this.pressurePipeline));pass.dispatchWorkgroups(Math.min(PRESSURE_GROUPS,this.ownership.capacity.tiles/64));
    pass.setPipeline(uniformDetailPick(this.rememberPressurePipeline!));pass.dispatchWorkgroups(Math.min(PRESSURE_GROUPS,this.ownership.capacity.tiles/64));
    pass.end();
  }
}

/** Base-cell pressure equivalent of a change across a four-cell root face. */
export const uniformPressureGradientChangeWGSL=/* wgsl */`
fn umPressureGradientChange(change:f32,neighbour:f32)->f32{return abs(change-neighbour)*0.25;}
`;
