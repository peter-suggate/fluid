import { uniformDetailModule, uniformDetailPick, uniformDetailPipeline } from "./uniform-detail-fields";
import type { UniformMixedOwnership } from "./uniform-mixed-ownership";
import { uniformMixedTopologyWGSL } from "./uniform-mixed-topology.wgsl";
import { UNIFORM_BODY_GEOMETRY_REACH } from "./uniform-mixed-solid.wgsl";
import { UNIFORM_PARAMS_BYTES, uniformMixedSourceWGSL } from "./uniform-mixed-source.wgsl";
import { UNIFORM_SURFACE_TRANSPORT_ERROR_H } from "./uniform-surface-transport.wgsl";
import { UNIFORM_REGION_DYNAMICS, UNIFORM_REGION_DYNAMICS_WORDS } from "./uniform-mixed-region-dynamics";
import { UNIFORM_COLD_FREEZE, UniformColdFreeze } from "./uniform-cold-freeze";

/** Per-16h-region work heat (region = umPageOfTile(tile), UM_PAGES slots):
 * where a frame's non-pressure work goes. EXNB-style (the particle activity
 * heat of uniform-narrow-band-activity): dimensionless, set to 2 by a
 * trigger, cooling 2/UNIFORM_REGION_HEAT_FADE_SECONDS per second. HOT is
 * kept for detail the cheap tiers do not certify; motion alone refreshes a
 * region WARM; the neighbours of a non-COLD region are at least WARM, so
 * the tiers fall off a ring a level from a trigger.
 *
 * One-frame HOT triggers (this frame only, no heat):
 * - a host invalidation (work-control generation) or the first frame;
 * - region identity: its block word or window coordinate changed (a page
 *   claim, a recycled key, a moved window); a free region is HOT;
 * - a page claim (the claim hook promotes the region HOT for the current
 *   frame, and pending HOT for the next);
 * - unknown evidence: a saturated velocity bound, an unknown pressure change.
 * The frame after one of these the pressure change is not evidence (its
 *   baseline was unknown: the first solve's change is the whole hydrostatic
 *   field), and the accumulated displacement restarts.
 *
 * Every one-frame trigger but the host's own (force) is held
 * UNIFORM_REGION_HEAT_HOLD_FRAMES frames more at its tier.
 *
 * One-frame WARM triggers (level 0.75 this frame, no heat):
 * - the accumulated displacement bound exceeding UNIFORM_REGION_HEAT_BUDGET_H:
 *   sum over frames since the region last ran non-COLD of
 *   dt*|u|max/h (the frame plan's current extended velocity extrema of the
 *   region, plus a frozen region's coarse term |T|max: UniformColdFreeze) + dt^2*|dp|/(rho*h^2) (stream 1's dynamics record word 6: the
 *   velocity a pressure change across a cell drives, over one step). Exact
 *   zero motion adds exactly zero;
 * - a remap (width change) of one of the region's tiles: the remap hook
 *   promotes it WARM now and pending WARM for the next frame.
 *
 * Heat triggers (heat := 2):
 * - a body whose swept support since the last record build lies within
 *   UNIFORM_BODY_GEOMETRY_REACH h of the region;
 * - a source within reach of the region;
 * - velocity detail beyond the cheap transport's certificate: the span
 *   0.5*dt*sum_axis(max-min)/h over the region's velocity extrema above
 *   UNIFORM_SURFACE_TRANSPORT_ERROR_H, or a step's travel dt*|u|max/h > 0.5.
 * The particle activity heat (max over the region's tiles) raises the
 * region's heat directly.
 *
 * Tier, final for the frame from the frame head (encode) on, except that a
 * remap or page claim later in the frame only raises regions:
 *   a region's level: its heat, at least 2 on a one-frame HOT trigger and
 *   0.75 on a one-frame WARM one;
 *   HOT (2): own level >= 1;
 *   WARM (1): otherwise max(own level, 0.5 * max neighbour level) > 0;
 *   COLD (0): otherwise.
 * COLD: the region's non-pressure producers hold their values (no write, no
 * revision bump) and its consumers retain their outputs. A vertex shared by
 * several regions is written only when every stored region holding it is
 * non-COLD (umVertexTier), so a COLD region's closure never changes.
 *
 * Buffer `tiers` (bind read-only through uniformRegionHeatWGSL):
 *   words [0, regions): the region's tier this frame (UNIFORM_REGION_TIER);
 *   words [regions, regions+4): diagnostics, regions COLD, WARM, HOT, free;
 *   word regions+4 (UNIFORM_REGION_HEAT_PROMOTED_WORD): regions a claim or
 *     remap hook raised after the head this frame (rhPromoteTo: counted
 *     when the hook raised the region's tier); words to regions+8 spare;
 *   words [R+8, 2R+8) (UNIFORM_REGION_HEAT_DEEP_WORD): deep COLD, 1 when the
 *     region is live and it and every stored neighbour (26) are COLD;
 *   words [2R+8, 3R+8) (UNIFORM_REGION_HEAT_EPOCH_WORD): the region's
 *     non-COLD epoch, advanced by every deep pass that finds it non-COLD (a
 *     consumer that saw the same epoch and COLD then has seen no non-COLD
 *     frame since).
 *   words [3R+8, 4R+8) (UNIFORM_REGION_HEAT_DEEP_EPOCH_WORD): the region's
 *     not-deep-COLD epoch, advanced by every deep pass that finds the region
 *     or one of its stored neighbours non-COLD (or the region free): a
 *     consumer that cached a result while the region was deep COLD at this
 *     epoch has seen nothing change in the region's 27-region neighbourhood.
 * The deep words and epochs follow the tiers as encode leaves them; after a
 * relayout's claims and remaps raise tiers, encodeDeep recomputes them.
 * The buffer is fixed for the frame's lifetime (never replaced). */
export const UNIFORM_REGION_TIER={cold:0,warm:1,hot:2} as const;
export const UNIFORM_REGION_HEAT_HEADER_WORDS=8;
/** The header word that counts this frame's mid-frame promotions (claims, remaps). */
export const UNIFORM_REGION_HEAT_PROMOTED_WORD=(regions:number):number=>regions+4;
/** Word offsets in tiers of the deep COLD flags and the non-COLD epochs. */
export const UNIFORM_REGION_HEAT_DEEP_WORD=(regions:number):number=>regions+UNIFORM_REGION_HEAT_HEADER_WORDS;
export const UNIFORM_REGION_HEAT_EPOCH_WORD=(regions:number):number=>2*regions+UNIFORM_REGION_HEAT_HEADER_WORDS;
export const UNIFORM_REGION_HEAT_DEEP_EPOCH_WORD=(regions:number):number=>3*regions+UNIFORM_REGION_HEAT_HEADER_WORDS;
/** Retirement time of a trigger's heat, s (the NB-FLIP default). */
export const UNIFORM_REGION_HEAT_FADE_SECONDS=0.5;
/** Frames a GPU-raised one-frame tier (identity, pending claim or remap,
 * unknown evidence, displacement over budget) is held after its own: the
 * host's cold-frame evidence (uniform-cold-frame) lags
 * UNIFORM_MIXED_RECEIPT_RING (2) frames, so a region a host skipper left
 * out at the rise is still non-COLD when the host sees it. */
export const UNIFORM_REGION_HEAT_HOLD_FRAMES=2;
/** Displacement a COLD region may hold back, in h (the transport budget). */
export const UNIFORM_REGION_HEAT_BUDGET_H=0.02;
/** State words per region: heat, displacement (h), block word, window
 * coordinate xyz, pending trigger (claims, remaps), trigger bits. */
const STATE_WORDS=9;
const HEAT_GRID=1024;
const PASS:GPUComputePassDescriptor={label:"Uniform region work heat"};

/** The tiers bound read-only at (group, binding); needs the topology WGSL.
 *   UM_TIER_COLD/WARM/HOT
 *   fn umRegionTier(region:u32)->u32     out of range: HOT
 *   fn umRegionTierAtTile(tile:u32)->u32 an unstored key: HOT
 *   fn umVertexTier(p:vec3u)->u32        lowest tier of the stored regions
 *                                        holding lattice vertex p (HOT if none)
 *   fn umRegionDeepCold(region:u32)->bool the region and its stored
 *                                        neighbours COLD (false out of range) */
export function uniformRegionHeatWGSL(group:number,binding:number,regions:number):string{return /* wgsl */`
@group(${group}) @binding(${binding}) var<storage,read> umWorkTier:array<u32>;
${uniformRegionTierWGSL(regions,"umWorkTier[region]",`umWorkTier[${UNIFORM_REGION_HEAT_DEEP_WORD(regions)}u+region]`)}`;}
/** The same accessors over a copy of the tiers the module already binds:
 * load reads region's tier (a WGSL expression in region); deepLoad, when
 * given, its deep COLD word (umRegionDeepCold). */
export function uniformRegionTierWGSL(regions:number,load:string,deepLoad?:string):string{return /* wgsl */`
${deepLoad?`fn umRegionDeepCold(region:u32)->bool{if(region>=${regions}u){return false;}return ${deepLoad}!=0u;}`:""}
const UM_TIER_COLD:u32=0u;const UM_TIER_WARM:u32=1u;const UM_TIER_HOT:u32=2u;
const UM_TIER_REGIONS:u32=${regions}u;
fn umRegionTier(region:u32)->u32{if(region>=UM_TIER_REGIONS){return UM_TIER_HOT;}return ${load};}
fn umRegionTierAtTile(tile:u32)->u32{if(!umKeyStored(tile)){return UM_TIER_HOT;}return umRegionTier(umPageOfTile(tile));}
fn umVertexTier(p:vec3u)->u32{
 var tier=UM_TIER_HOT+1u;
 for(var k=0u;k<8u;k++){
  let side=umCorner(k,2u);if(any((side!=vec3u(0u))&((p&vec3u(15u))!=vec3u(0u)))){continue;}
  let tile=umTileKey((vec3i(p)-vec3i(side))>>vec3u(2u));if(!umKeyStored(tile)){continue;}
  tier=min(tier,umRegionTier(umPageOfTile(tile)));
 }
 return select(tier,UM_TIER_HOT,tier>UM_TIER_HOT);
}
`;}

export interface UniformRegionHeatSources {
  /** UniformMixedSolid.bodyRegions, when bodies are compiled in. */
  bodyRegions?:{buffer:GPUBuffer;regions:number};
  /** UniformMixedRegionDynamics.record. */
  dynamics:GPUBuffer;
  /** The NB-FLIP particle activity heat: f32 a tile key from word. */
  particleHeat?:{buffer:GPUBuffer;word:number};
  /** The scene's source uniforms (UNIFORM_PARAMS_BYTES). */
  sourceParams?:GPUBuffer;
  /** The COLD freeze (uniform-cold-freeze): a frozen region's coarse term
   * T (its region snapshot, as of the frame before's head) adds
   * dt*max|T|/h to the region's displacement, so a coarse response over
   * the budget thaws it. */
  coldFreeze?:UniformColdFreeze;
  /** Hold GPU-raised one-frame tiers UNIFORM_REGION_HEAT_HOLD_FRAMES more
   * frames (the NB-FLIP coldFrameSkip "on" setting; default true): host
   * skippers then see every rise while it stands. False holds none. */
  holdRises?:boolean;
}

export class UniformRegionHeat {
  readonly regions:number;
  /** Tiers and diagnostics (see the file comment); fixed for life. */
  readonly tiers:GPUBuffer;
  /** Private: heat, displacement, identity and pending words a region. */
  readonly state:GPUBuffer;
  readonly allocatedBytes:number;
  private readonly params:GPUBuffer;
  private readonly words=new Float32Array(8);
  private readonly bits=new Uint32Array(this.words.buffer);
  private layout?:GPUBindGroupLayout;
  private sources?:UniformRegionHeatSources;
  private group?:GPUBindGroup;
  private heatPipeline?:GPUComputePipeline;
  private spreadPipeline?:GPUComputePipeline;
  private deepPipeline?:GPUComputePipeline;
  private forced=true;
  private seenGeneration=-1;
  /** The tiers exist from construction, for every stage to bind; the
   * sources attach before initialize. */
  constructor(private readonly device:GPUDevice,readonly ownership:UniformMixedOwnership,private readonly generation:()=>number){
    this.regions=ownership.capacity.tiles/64;
    if(!Number.isInteger(this.regions))throw new Error(`Region heat: ${ownership.capacity.tiles} tiles are not whole regions`);
    const usage=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST;
    // Every region reads HOT (and not deep COLD) until the first encode.
    this.tiers=device.createBuffer({label:"Uniform region work tiers",size:4*UNIFORM_REGION_HEAT_DEEP_EPOCH_WORD(this.regions)+4*this.regions,usage,mappedAtCreation:true});
    new Uint32Array(this.tiers.getMappedRange()).fill(UNIFORM_REGION_TIER.hot,0,this.regions);this.tiers.unmap();
    this.state=device.createBuffer({label:"Uniform region work heat state",size:4*STATE_WORDS*this.regions,usage});
    this.params=device.createBuffer({label:"Uniform region work heat params",size:32,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
    this.allocatedBytes=this.tiers.size+this.state.size+this.params.size;
  }
  attach(sources:UniformRegionHeatSources):void{
    if(this.sources)throw new Error("Region heat sources are attached once");
    if(sources.bodyRegions&&sources.bodyRegions.regions!==this.regions)throw new Error("Region heat: body mask region count differs");
    if(sources.dynamics.size<4*UNIFORM_REGION_DYNAMICS_WORDS*this.regions)throw new Error("Region heat: the dynamics record holds fewer regions");
    this.sources=sources;const device=this.device;
    const entry=(binding:number,buffer:GPUBufferBindingLayout)=>({binding,visibility:GPUShaderStage.COMPUTE,buffer});
    this.layout=device.createBindGroupLayout({label:"Uniform region work heat",entries:[
      entry(0,{type:"storage"}),entry(1,{type:"storage"}),entry(2,{type:"uniform"}),entry(3,{type:"read-only-storage"}),entry(4,{type:"read-only-storage"}),
      ...(sources.bodyRegions?[entry(5,{type:"read-only-storage"})]:[]),...(sources.particleHeat?[entry(6,{type:"read-only-storage"})]:[]),...(sources.sourceParams?[entry(7,{type:"uniform"})]:[]),
      ...(this.freezeOf(sources)?[{binding:8,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float" as const,viewDimension:"2d" as const}}]:[]),
    ]});
  }
  /** The freeze the heat binds: none while UNIFORM_COLD_FREEZE is off. */
  private freezeOf(sources:UniformRegionHeatSources):UniformColdFreeze|undefined{return UNIFORM_COLD_FREEZE?sources.coldFreeze:undefined;}
  /** Diagnostics: COLD, WARM, HOT and free region counts of the last encode. */
  get counts():{readonly buffer:GPUBuffer;readonly offset:number}{return {buffer:this.tiers,offset:4*this.regions};}
  /** Every region HOT at the next encode (any host state change). */
  invalidate():void{this.forced=true;}
  async initialize():Promise<void>{
    const s=this.sources;if(!s||!this.layout)throw new Error("Region heat initializes after its sources attach");
    const h=Math.min(...this.ownership.capacity.lattice.cellSize_m);
    const module=uniformDetailModule(this.device,{label:"Uniform region work heat",code:uniformMixedTopologyWGSL(this.ownership.capacity,0)+/* wgsl */`
@group(1) @binding(0) var<storage,read_write> tiers:array<atomic<u32>>;
@group(1) @binding(1) var<storage,read_write> state:array<u32>;
struct RHParams{dt:f32,cooling:f32,pressure:f32,hInverse:f32,force:u32,pad0:u32,pad1:u32,pad2:u32}
@group(1) @binding(2) var<uniform> rh:RHParams;
@group(1) @binding(3) var<storage,read> regionVelocity:array<u32>;
@group(1) @binding(4) var<storage,read> dynamics:array<u32>;
${s.bodyRegions?"@group(1) @binding(5) var<storage,read> bodyMask:array<u32>;":""}
${s.particleHeat?`@group(1) @binding(6) var<storage,read> particleHeat:array<u32>;const RH_PARTICLE:u32=${s.particleHeat.word}u;`:""}
${s.sourceParams?uniformMixedSourceWGSL(7):""}
${UniformColdFreeze.wgsl(this.freezeOf(s),1,8)}
// A frozen region's coarse term: its speed bound, m/s (0 when not frozen).
fn rhColdSpeed(region:u32)->f32{if(!umFrozenRegion(region)){return 0.0;}return max(abs(umColdTermOf(region,0u)),max(abs(umColdTermOf(region,1u)),abs(umColdTermOf(region,2u))));}
const RH_REGIONS:u32=${this.regions}u;
const RH_STATE:u32=${STATE_WORDS}u;
const RH_BUDGET:f32=${UNIFORM_REGION_HEAT_BUDGET_H};
const RH_UNKNOWN:f32=3.0e38;
// Force, identity, pending, unknown and a held HOT rise: HOT this frame, no heat.
const RH_ONE_FRAME:u32=0x227u;
// Displacement over budget, a WARM promotion, a held WARM rise: WARM this frame, no heat.
const RH_WARM_FRAME:u32=0x540u;
// The one-frame rises the host does not cause (all but force): held.
const RH_HELD_HOT:u32=0x26u;
const RH_HELD_WARM:u32=0x140u;
const RH_HOLD_FRAMES:u32=${s.holdRises===false?0:UNIFORM_REGION_HEAT_HOLD_FRAMES}u;
const RH_SPAN_H:f32=${UNIFORM_SURFACE_TRANSPORT_ERROR_H};
// A region's level: its heat, at least 2 on a one-frame HOT trigger and
// 0.75 on a one-frame WARM one.
fn rhLevel(region:u32)->f32{
 let at=RH_STATE*region;let bits=state[at+7u];var level=bitcast<f32>(state[at]);
 if((bits&RH_ONE_FRAME)!=0u){level=2.0;}else if((bits&RH_WARM_FRAME)!=0u){level=max(level,0.75);}
 return level;
}
// A signed-order velocity word's value (0: no owner; all ones: saturated).
fn rhSigned(word:u32)->f32{return bitcast<f32>(select(~word,word^0x80000000u,(word&0x80000000u)!=0u));}
// Signed-order velocity words (umVelocityOrder): their magnitude, m/s.
fn rhSpeed(word:u32)->f32{
 if(word==0u){return 0.0;}if(word==0xffffffffu){return 3.402823e38;}
 return abs(bitcast<f32>(select(~word,word^0x80000000u,(word&0x80000000u)!=0u)));
}
var<workgroup> rhLive:u32;
var<workgroup> rhParticle:atomic<u32>;
// One workgroup a region (a fixed grid striding them): a lane a tile reads
// the particle heat, lane 0 updates the region's heat and displacement.
@compute @workgroup_size(64) fn heat(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) l:u32){
 for(var region=group.x;region<RH_REGIONS;region+=groups.x){
  if(l==0u){rhLive=u32(umTileLive(umPageTile(region,0u)));atomicStore(&rhParticle,0u);}
  let live=workgroupUniformLoad(&rhLive);
  let at=RH_STATE*region;
  if(live==0u){
   // A free region keeps no identity: whatever it later holds triggers.
   if(l==0u){state[at]=0u;state[at+1u]=0u;state[at+2u]=0xffffffffu;state[at+6u]=0u;state[at+7u]=0u;state[at+8u]=0u;}
   continue;
  }
  ${s.particleHeat?`{let t=umPageTile(region,l);if(umTileLive(t)){atomicMax(&rhParticle,particleHeat[RH_PARTICLE+t]);}}`:""}
  workgroupBarrier();
  if(l==0u){
   var trigger=0u;
   if(rh.force!=0u){trigger|=1u;}
   let block=umPageTile(region,0u)>>9u;let word=umBlockWord(block);let c=umPageCoord(region);
   if(state[at+2u]!=word||state[at+3u]!=c.x||state[at+4u]!=c.y||state[at+5u]!=c.z){trigger|=2u;}
   state[at+2u]=word;state[at+3u]=c.x;state[at+4u]=c.y;state[at+5u]=c.z;
   {let pending=state[at+6u];if(pending>=${UNIFORM_REGION_TIER.hot}u){trigger|=4u;}else if(pending!=0u){trigger|=256u;}state[at+6u]=0u;}
   ${s.bodyRegions?`if((bodyMask[region]&0x10000u)!=0u||(bodyMask[region]&0xffu)<=${UNIFORM_BODY_GEOMETRY_REACH}u){trigger|=8u;}`:""}
   ${s.sourceParams?`{let centre=16.0*vec3f(c)+vec3f(8.0);let reach=(8.0*1.7320508+4.0)/rh.hInverse;if(umSourceuvSourcePhi(centre,1e30)<reach){trigger|=16u;}}`:""}
   // A region that ran non-COLD last frame was refreshed then.
   var displacement=bitcast<f32>(state[at+1u]);if(atomicLoad(&tiers[region])!=0u){displacement=0.0;}
   var speed=0.0;for(var k=0u;k<6u;k++){speed=max(speed,rhSpeed(regionVelocity[6u*region+k]));}
   // Detail the cheap transport tiers do not certify: the velocity span
   // (umSurfaceTransport's velocityError against its error budget) or a
   // step's travel beyond half a cell.
   {var span=0.0;
    for(var axis=0u;axis<3u;axis++){let hi=regionVelocity[6u*region+axis];let lo=regionVelocity[6u*region+3u+axis];
     if(hi!=0u&&lo!=0u&&hi!=0xffffffffu&&lo!=0xffffffffu){span+=max(0.0,rhSigned(hi)+rhSigned(lo));}}
    if(0.5*rh.dt*span*rh.hInverse>RH_SPAN_H||rh.dt*speed*rh.hInverse>0.5){trigger|=128u;}}
   var dp=bitcast<f32>(dynamics[${UNIFORM_REGION_DYNAMICS_WORDS}u*region+${UNIFORM_REGION_DYNAMICS.pressureDelta}u]);
   // After a one-frame trigger the pressure change has no baseline.
   if((state[at+7u]&RH_ONE_FRAME)!=0u){dp=0.0;}
   if(!(speed<=RH_UNKNOWN)||!(dp<=RH_UNKNOWN)){trigger|=32u;}
   else{
    displacement+=rh.dt*(speed+rhColdSpeed(region))*rh.hInverse+rh.pressure*dp;
    if(displacement>RH_BUDGET){trigger|=64u;}
   }
   // A rise the host does not cause holds RH_HOLD_FRAMES frames more
   // (bit 9 a held HOT, bit 10 a held WARM): the host's evidence lags
   // UNIFORM_MIXED_RECEIPT_RING frames, and sees it while it still stands.
   {let held=state[at+8u];var left=held>>8u;var tier=select(0u,held&255u,left>0u);
    if(left>0u){trigger|=select(1024u,512u,tier>=${UNIFORM_REGION_TIER.hot}u);left--;}
    if((trigger&RH_HELD_HOT)!=0u){tier=${UNIFORM_REGION_TIER.hot}u;left=RH_HOLD_FRAMES;}
    else if((trigger&RH_HELD_WARM)!=0u){tier=max(tier,${UNIFORM_REGION_TIER.warm}u);left=RH_HOLD_FRAMES;}
    state[at+8u]=select(0u,tier|(left<<8u),left>0u);}
   if(trigger!=0u){displacement=0.0;}
   var value=max(0.0,bitcast<f32>(state[at])-rh.cooling*rh.dt);
   if((trigger&~(RH_ONE_FRAME|RH_WARM_FRAME))!=0u){value=2.0;}
   ${s.particleHeat?"value=max(value,bitcast<f32>(atomicLoad(&rhParticle)));":""}
   state[at]=bitcast<u32>(value);state[at+1u]=bitcast<u32>(displacement);state[at+7u]=trigger;
  }
 }
}
// A thread a region: HOT from its own level only, WARM from its own or a
// neighbour's at half weight. Every region beside a non-COLD one is
// non-COLD, so the tiers fall off a ring a level.
@compute @workgroup_size(64) fn spread(@builtin(global_invocation_id) gid:vec3u){
 let region=gid.x;if(region>=RH_REGIONS){return;}
 if(!umTileLive(umPageTile(region,0u))){atomicStore(&tiers[region],${UNIFORM_REGION_TIER.hot}u);atomicAdd(&tiers[RH_REGIONS+3u],1u);return;}
 let c=vec3i(umPageCoord(region));let own=rhLevel(region);var level=own;
 for(var k=0u;k<27u;k++){
  if(k==13u){continue;}
  let key=umTileKey(4*(c+vec3i(umCorner(k,3u))-vec3i(1)));if(!umKeyStored(key)){continue;}
  level=max(level,0.5*rhLevel(umPageOfTile(key)));
 }
 let tier=select(select(${UNIFORM_REGION_TIER.cold}u,${UNIFORM_REGION_TIER.warm}u,level>0.0),${UNIFORM_REGION_TIER.hot}u,own>=1.0);
 atomicStore(&tiers[region],tier);atomicAdd(&tiers[RH_REGIONS+tier],1u);
}
// A thread a region, on the frame's tiers as they stand: its deep COLD flag
// and its non-COLD epoch.
@compute @workgroup_size(64) fn deep(@builtin(global_invocation_id) gid:vec3u){
 let region=gid.x;if(region>=RH_REGIONS){return;}
 let tier=atomicLoad(&tiers[region]);
 if(tier!=${UNIFORM_REGION_TIER.cold}u){atomicAdd(&tiers[${UNIFORM_REGION_HEAT_EPOCH_WORD(this.regions)}u+region],1u);}
 var cold=tier==${UNIFORM_REGION_TIER.cold}u&&umTileLive(umPageTile(region,0u));
 let c=vec3i(umPageCoord(region));
 for(var k=0u;k<27u&&cold;k++){
  let key=umTileKey(4*(c+vec3i(umCorner(k,3u))-vec3i(1)));if(!umKeyStored(key)){continue;}
  cold=atomicLoad(&tiers[umPageOfTile(key)])==${UNIFORM_REGION_TIER.cold}u;
 }
 atomicStore(&tiers[${UNIFORM_REGION_HEAT_DEEP_WORD(this.regions)}u+region],u32(cold));
 if(!cold){atomicAdd(&tiers[${UNIFORM_REGION_HEAT_DEEP_EPOCH_WORD(this.regions)}u+region],1u);}
}
`});
    const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");if(errors.length)throw new Error(errors.map(m=>`${m.lineNum}: ${m.message}`).join("\n"));
    const layout=this.device.createPipelineLayout({bindGroupLayouts:[this.ownership.bindLayout,this.layout]});
    [this.heatPipeline,this.spreadPipeline,this.deepPipeline]=await Promise.all(["heat","spread","deep"].map(entryPoint=>uniformDetailPipeline(this.device,this.ownership,{layout,compute:{module,entryPoint,constants:{umDispatchX:this.ownership.dispatchX}}})));
    const entries:GPUBindGroupEntry[]=[{binding:0,resource:{buffer:this.tiers}},{binding:1,resource:{buffer:this.state}},{binding:2,resource:{buffer:this.params}},
      {binding:3,resource:{buffer:this.ownership.regionVelocity}},{binding:4,resource:{buffer:s.dynamics}}];
    if(s.bodyRegions)entries.push({binding:5,resource:{buffer:s.bodyRegions.buffer,size:4*this.regions}});
    if(s.particleHeat)entries.push({binding:6,resource:{buffer:s.particleHeat.buffer}});
    if(s.sourceParams)entries.push({binding:7,resource:{buffer:s.sourceParams,size:UNIFORM_PARAMS_BYTES}});
    {const freeze=this.freezeOf(s);if(freeze)entries.push({binding:8,resource:freeze.regions.createView()});}
    this.group=this.device.createBindGroup({label:"Uniform region work heat",layout:this.layout,entries});
    this.hInverse=1/h;
  }
  private hInverse=0;
  /** The frame head's tiers, before any stage that reads them. dt: this
   * frame's step; density: the liquid's, kg/m^3. */
  encode(encoder:GPUCommandEncoder,dt:number,density:number):void{
    if(!this.heatPipeline||!this.spreadPipeline||!this.group)throw new Error("Region heat is not initialized");
    const generation=this.generation();const force=this.forced||generation!==this.seenGeneration;this.forced=false;this.seenGeneration=generation;
    const w=this.words;w[0]=dt;w[1]=2/UNIFORM_REGION_HEAT_FADE_SECONDS;w[2]=dt*dt*this.hInverse*this.hInverse/Math.max(density,1e-30);w[3]=this.hInverse;this.bits[4]=+force;
    this.device.queue.writeBuffer(this.params,0,this.words);
    encoder.clearBuffer(this.tiers,4*this.regions,4*UNIFORM_REGION_HEAT_HEADER_WORDS);
    const pass=encoder.beginComputePass(PASS);pass.setBindGroup(0,this.ownership.bindGroup);pass.setBindGroup(1,this.group);
    pass.setPipeline(uniformDetailPick(this.heatPipeline));pass.dispatchWorkgroups(Math.min(HEAT_GRID,this.regions));
    pass.setPipeline(uniformDetailPick(this.spreadPipeline));pass.dispatchWorkgroups(Math.ceil(this.regions/64));
    pass.setPipeline(uniformDetailPick(this.deepPipeline!));pass.dispatchWorkgroups(Math.ceil(this.regions/64));
    pass.end();
  }
  /** The deep COLD flags and epochs again, after a relayout's claims and
   * remaps raised tiers (promoteWGSL): before any stage that reads them. */
  encodeDeep(encoder:GPUCommandEncoder):void{
    if(!this.deepPipeline||!this.group)throw new Error("Region heat is not initialized");
    const pass=encoder.beginComputePass(PASS);pass.setBindGroup(0,this.ownership.bindGroup);pass.setBindGroup(1,this.group);
    pass.setPipeline(uniformDetailPick(this.deepPipeline));pass.dispatchWorkgroups(Math.ceil(this.regions/64));pass.end();
  }
  /** WGSL for hooks that raise a region's tier now and promote it again at
   * the next encode (a one-frame trigger): tiers and state bound read-write
   * at (group, tiers binding) and (group, state binding).
   * fn rhPromote(region:u32) to HOT; fn rhPromoteWarm(region:u32) to WARM. */
  promoteWGSL(group:number,tiersBinding:number,stateBinding:number):string{return /* wgsl */`
@group(${group}) @binding(${tiersBinding}) var<storage,read_write> rhTiers:array<atomic<u32>>;
@group(${group}) @binding(${stateBinding}) var<storage,read_write> rhState:array<atomic<u32>>;
fn rhPromoteTo(region:u32,tier:u32){if(region>=${this.regions}u){return;}if(atomicMax(&rhTiers[region],tier)<tier){atomicAdd(&rhTiers[${UNIFORM_REGION_HEAT_PROMOTED_WORD(this.regions)}u],1u);}atomicMax(&rhState[${STATE_WORDS}u*region+6u],tier);}
fn rhPromote(region:u32){rhPromoteTo(region,${UNIFORM_REGION_TIER.hot}u);}
fn rhPromoteWarm(region:u32){rhPromoteTo(region,${UNIFORM_REGION_TIER.warm}u);}
`;}
  destroy():void{this.tiers.destroy();this.state.destroy();this.params.destroy();}
}
