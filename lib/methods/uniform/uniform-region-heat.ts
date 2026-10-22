import {uniformTileUpdateWGSL,uniformTierHoldWGSL,UNIFORM_TILE_UPDATE_BUDGET_H} from "./uniform-tile-update-error";
import { uniformDetailModule, uniformDetailPick, uniformDetailPipeline } from "./uniform-detail-fields";
import type { UniformMixedOwnership } from "./uniform-mixed-ownership";
import { uniformMixedTopologyWGSL } from "./uniform-mixed-topology.wgsl";
import { UNIFORM_BODY_GEOMETRY_REACH } from "./uniform-mixed-solid.wgsl";
import { UNIFORM_PARAMS_BYTES, uniformMixedSourceWGSL } from "./uniform-mixed-source.wgsl";
import { UNIFORM_SURFACE_TRANSPORT_ERROR_H } from "./uniform-surface-transport.wgsl";
import { UNIFORM_REGION_DYNAMICS, UNIFORM_REGION_DYNAMICS_WORDS } from "./uniform-mixed-region-dynamics";
import { UNIFORM_COLD_FREEZE, UniformColdFreeze } from "./uniform-cold-freeze";

/** Per-16h-region update activity, independent of the spatial owner widths.
 * In error mode, COLD retains non-pressure fields only while the accumulated
 * omitted-physics estimate fits its depth-dependent budget (in h). Per-tile evidence bounds
 * normal surface transport and momentum advection; the root pressure
 * gradient change and the frozen pressure response's discarded variation
 * also consume this budget. Surface regions use the transport tolerance of
 * their finest physical owner (0.02h for h, 0.08h for 4h). Fully submerged
 * regions allow 0.08h and do not charge already represented
 * uniform pressure motion as omitted transport. Pressure still solves globally.
 *
 * HOT means body/source contact, live particle detail, a tile reconstruction
 * error above the detailed-update budget, excessive characteristic travel,
 * or invalid evidence. Spatial curvature alone never makes a region HOT.
 * WARM means the freeze budget was exhausted, a spatial remap needs refreshing,
 * or a fine surface reader shares the dependency closure of a HOT region. WARM
 * uses the spatial resolution selected by the geometry census by default.
 * Empty extrapolated air and solid interiors are not physical readers.
 *
 * Host/topology invalidations are immediate. GPU rises have independent
 * two-frame HOT/WARM holds, matching the host receipt latency: a WARM rise
 * cannot renew a HOT deadline. Body/source/detail heat decays over 0.5 s;
 * non-reader regions discard heat once the source/body trigger is gone.
 * In motion mode the older absolute-speed and velocity-span policy remains
 * available for comparison.
 *
 * Shared vertices/pressure faces incident to COLD regions remain frozen.
 * A one-region closure around HOT stays WARM for fine surface readers.
 * Coarse and deep readers supply F+T at these interfaces instead; their
 * local omission budgets decide when to thaw, not neighbour activity alone.
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
/** Accumulated freeze error in h, scaled to the actual transport owner.
 * The old 0.002h threshold was ten times smaller than even an h transport
 * step's approximation tolerance, making almost every ripple stay WARM. */
export const UNIFORM_REGION_HEAT_FINE_BUDGET_H=UNIFORM_SURFACE_TRANSPORT_ERROR_H;
export const UNIFORM_REGION_HEAT_BUDGET_H=4*UNIFORM_REGION_HEAT_FINE_BUDGET_H;
/** Deep bulk water has no nearby interface to transport. Permit 0.08h of
 * accumulated omission error (0.5% of the 16h update cell), rather than the
 * fine surface's 0.02h. Pressure continues to solve across the whole domain. */
export const UNIFORM_REGION_HEAT_SUBMERGED_BUDGET_H=0.08;
export const uniformRegionFreezeBudgetWGSL=/* wgsl */`
fn rhFreezeBudget(submerged:bool,fine:bool)->f32{
 return select(select(${UNIFORM_REGION_HEAT_BUDGET_H},${UNIFORM_REGION_HEAT_FINE_BUDGET_H},fine),${UNIFORM_REGION_HEAT_SUBMERGED_BUDGET_H},submerged);
}
fn rhFrozenErrorSpeed(submerged:bool,representedSpeed:f32,discardedSpeed:f32)->f32{
 // T is already applied by the frozen momentum/pressure model. Its motion
 // matters for interface transport, but is not omitted bulk momentum.
 return discardedSpeed+select(representedSpeed,0.0,submerged);
}
fn rhHotHaloRequired(reader:bool,submerged:bool,fine:bool)->bool{return reader&&!submerged&&fine;}
fn rhBulkAcceleration(acceleration:f32,gradient:f32,representedSpeed:f32)->f32{
 // The pressure translation itself is represented, but its advection of
 // nonuniform frozen velocity is not. Uniform flow (gradient=0) stays free.
 return acceleration+representedSpeed*gradient;
}
`;
/** State words per region: heat, displacement (h), block word, window
 * coordinate xyz, pending trigger (claims, remaps), trigger bits, independent
 * HOT/WARM hold counters (low/high 16 bits). Words 9..15 publish the decision
 * evidence: surface, momentum, pressure-gradient and frozen-response errors
 * (h), detailed-update error (h), physical tile count (u32), incoming debt (h).
 * Trigger bit 12 records a physical reader; bit 13 certifies all physical
 * readers submerged; bit 14 records a physical h owner. */
export const UNIFORM_REGION_HEAT_STATE_WORDS=16;
const STATE_WORDS=UNIFORM_REGION_HEAT_STATE_WORDS;
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
export function uniformRegionTierWGSL(regions:number,load:string,deepLoad?:string,vertexCapacity?:string):string{return /* wgsl */`
${deepLoad?`fn umRegionDeepCold(region:u32)->bool{if(region>=${regions}u){return false;}return ${deepLoad}!=0u;}`:""}
const UM_TIER_COLD:u32=0u;const UM_TIER_WARM:u32=1u;const UM_TIER_HOT:u32=2u;
const UM_TIER_REGIONS:u32=${regions}u;
fn umRegionTier(region:u32)->u32{if(region>=UM_TIER_REGIONS){return UM_TIER_HOT;}return ${load};}
fn umRegionTierAtTile(tile:u32)->u32{if(!umKeyStored(tile)){return UM_TIER_HOT;}return umRegionTier(umPageOfTile(tile));}
fn umVertexTier(p:vec3u)->u32{
 var tier=UM_TIER_HOT+1u;
 let boundary=${vertexCapacity?"vec3<bool>(true)":"(p&vec3u(15u))==vec3u(0u)"};
 let sides=u32(boundary.x)|(2u*u32(boundary.y))|(4u*u32(boundary.z));
 // Without geometry, enumerate incident regions. With geometry, visit each
 // incident cell: a cold solid interior must not freeze its liquid wall.
 ${vertexCapacity?"var open=false;":""}
 for(var k=0u;k<=sides;k=select((k-sides)&sides,8u,k==sides)){
  let side=umCorner(k,2u);
  let cell=vec3i(p)-vec3i(side);let tile=umTileKey(cell>>vec3u(2u));
  // An unstored cell holds no state (as nbVertexCold), and its capacity
  // would cost a solid evaluation per h cell (umTileOpenAt).
  if(!umKeyStored(tile)){continue;}
  ${vertexCapacity?`if(!(${vertexCapacity})){continue;}open=true;`:""}
  tier=min(tier,umRegionTier(umPageOfTile(tile)));
 }
 ${vertexCapacity?`// Buried (no stored incident cell has capacity, as umVertexBuried): phi
 // there is not state, so the vertex holds it in every tier. Without this it
 // read as HOT, and every buried wall and floor vertex took the full transport.
 if(!open){return UM_TIER_COLD;}`:""}
 return select(tier,UM_TIER_HOT,tier>UM_TIER_HOT);
}
`;}

export interface UniformRegionHeatSources {
  /** UniformMixedSolid.bodyRegions, when bodies are compiled in. */
  bodyRegions?:{buffer:GPUBuffer;regions:number};
  /** UniformMixedRegionDynamics.record. */
  dynamics:GPUBuffer;
  /** Per-tile reconstruction errors from the preceding velocity certificate. */
  updateErrors?:GPUBuffer;
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
  /** "error" (default): operator error estimates determine activity; spatial
   * refinement remains independent. "motion" retains absolute-motion tiers
   * and their wider neighbour activation for comparison. */
  tiering?:"error"|"motion";
}

export class UniformRegionHeat {
  readonly regions:number;
  /** Tiers and diagnostics (see the file comment); fixed for life. */
  readonly tiers:GPUBuffer;
  /** Heat, identity, independent holds and the numerical wake-up evidence. */
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
      entry(0,{type:"storage"}),entry(1,{type:"storage"}),entry(2,{type:"uniform"}),entry(3,{type:"read-only-storage"}),entry(4,{type:"read-only-storage"}),...(sources.updateErrors?[entry(9,{type:"read-only-storage"})]:[]),
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
    const h=Math.min(...this.ownership.capacity.lattice.cellSize_m);const error=s.tiering!=="motion";
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
${s.updateErrors?uniformTileUpdateWGSL(1,9):""}
// A frozen region's coarse term: its speed bound, m/s (0 when not frozen).
fn rhColdSpeed(region:u32)->f32{if(!umFrozenRegion(region)){return 0.0;}return max(abs(umColdTermOf(region,0u)),max(abs(umColdTermOf(region,1u)),abs(umColdTermOf(region,2u))));}
const RH_REGIONS:u32=${this.regions}u;
const RH_STATE:u32=${STATE_WORDS}u;
${uniformRegionFreezeBudgetWGSL}
const RH_UNKNOWN:f32=3.0e38;
// Force, identity, pending, unknown and a held HOT rise: HOT this frame, no heat.
const RH_ONE_FRAME:u32=0x227u;
// Displacement over budget, a WARM promotion, a held WARM rise: WARM this frame, no heat.
const RH_WARM_FRAME:u32=${error?"0x5c0u":"0x540u"};
// The one-frame rises the host does not cause (all but force): held.
const RH_HELD_HOT:u32=0x26u;
const RH_HELD_WARM:u32=${error?"0x1c0u":"0x140u"};
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
var<workgroup> rhError:atomic<u32>;
var<workgroup> rhNormal:atomic<u32>;
var<workgroup> rhAcceleration:atomic<u32>;
var<workgroup> rhRelevant:atomic<u32>;
var<workgroup> rhReaderClass:atomic<u32>;
var<workgroup> rhFine:atomic<u32>;
// One workgroup a region (a fixed grid striding them): a lane a tile reads
// the particle heat, lane 0 updates the region's heat and displacement.
@compute @workgroup_size(64) fn heat(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) l:u32){
 for(var region=group.x;region<RH_REGIONS;region+=groups.x){
  if(l==0u){rhLive=u32(umTileLive(umPageTile(region,0u)));atomicStore(&rhParticle,0u);atomicStore(&rhError,0u);atomicStore(&rhNormal,0u);atomicStore(&rhAcceleration,0u);atomicStore(&rhRelevant,0u);atomicStore(&rhReaderClass,0u);atomicStore(&rhFine,0u);}
  let live=workgroupUniformLoad(&rhLive);
  let at=RH_STATE*region;
  if(live==0u){
   // A free region keeps no identity: whatever it later holds triggers.
   if(l==0u){state[at]=0u;state[at+1u]=0u;state[at+2u]=0xffffffffu;state[at+6u]=0u;state[at+7u]=0u;state[at+8u]=0u;}
   continue;
  }
  ${s.particleHeat?`{let t=umPageTile(region,l);if(umTileLive(t)){atomicMax(&rhParticle,particleHeat[RH_PARTICLE+t]);}}`:""}
  ${s.updateErrors?`{let t=umPageTile(region,l);if(umTileLive(t)){atomicMax(&rhError,bitcast<u32>(tuError(t)));let physical=tuPhysical(t);atomicMax(&rhNormal,bitcast<u32>(physical.x));let acceleration=rhBulkAcceleration(physical.y,physical.w,select(0.0,rhColdSpeed(region),physical.z==1.0));atomicMax(&rhAcceleration,bitcast<u32>(acceleration));atomicAdd(&rhRelevant,u32(physical.z>0.0));atomicMax(&rhReaderClass,u32(physical.z));if(physical.z>0.0&&umTileWidth(t)==1u){atomicStore(&rhFine,1u);}}}`:""}
  workgroupBarrier();
  if(l==0u){
   let submerged=${error&&s.updateErrors?"atomicLoad(&rhReaderClass)==1u":"false"};
   let fine=atomicLoad(&rhFine)!=0u;
   var trigger=0u;for(var k=9u;k<RH_STATE;k++){state[at+k]=0u;}
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
    ${error&&s.updateErrors?"":"if(0.5*rh.dt*span*rh.hInverse>RH_SPAN_H||rh.dt*speed*rh.hInverse>0.5){trigger|=128u;}"}
    ${error?(s.updateErrors?`let error=bitcast<f32>(atomicLoad(&rhError));state[at+13u]=bitcast<u32>(rh.dt*error*rh.hInverse);if(!(error<RH_UNKNOWN)||rh.dt*error*rh.hInverse>${UNIFORM_TILE_UPDATE_BUDGET_H}||rh.dt*speed*rh.hInverse>2.0){trigger|=2048u;}`:"if(0.5*rh.dt*span*rh.hInverse>4.0*RH_SPAN_H||rh.dt*speed*rh.hInverse>2.0){trigger|=2048u;}"):""}}
   var dp=bitcast<f32>(dynamics[${UNIFORM_REGION_DYNAMICS_WORDS}u*region+${UNIFORM_REGION_DYNAMICS.pressureDelta}u]);
   // After a one-frame trigger the pressure change has no baseline.
   if((state[at+7u]&RH_ONE_FRAME)!=0u){dp=0.0;}
   if(!(speed<=RH_UNKNOWN)||!(dp<=RH_UNKNOWN)){${error&&s.updateErrors?"if(atomicLoad(&rhRelevant)!=0u)":""}{trigger|=32u;}}
   else{
    ${error&&s.updateErrors?`
    let normal=bitcast<f32>(atomicLoad(&rhNormal));let acceleration=bitcast<f32>(atomicLoad(&rhAcceleration));
    state[at+14u]=atomicLoad(&rhRelevant);state[at+15u]=bitcast<u32>(displacement);
    if(atomicLoad(&rhRelevant)!=0u){
     // Integrate the omitted normal transport and the accumulated momentum
     // change, not distance travelled tangentially or through deep water.
     let surface= rh.dt*normal*rh.hInverse;
     let momentum=rh.dt*sqrt(max(0.0,2.0*acceleration*displacement*rh.hInverse))+rh.dt*rh.dt*acceleration*rh.hInverse;
     let pressure=rh.pressure*dp;let frozen=rh.dt*rhFrozenErrorSpeed(submerged,rhColdSpeed(region),umColdError(region))*rh.hInverse;
     state[at+9u]=bitcast<u32>(surface);state[at+10u]=bitcast<u32>(momentum);state[at+11u]=bitcast<u32>(pressure);state[at+12u]=bitcast<u32>(frozen);
     displacement+=surface+momentum+pressure+frozen;
    }`:"displacement+=rh.dt*(speed+rhColdSpeed(region))*rh.hInverse+rh.pressure*dp;"}
    if(displacement>${error?"rhFreezeBudget(submerged,fine)":"0.002"}){trigger|=64u;}
   }
   // A rise the host does not cause holds RH_HOLD_FRAMES frames more
   // (bit 9 a held HOT, bit 10 a held WARM): the host's evidence lags
   // UNIFORM_MIXED_RECEIPT_RING frames, and sees it while it still stands.
   {${uniformTierHoldWGSL}}
   if(trigger!=0u){displacement=0.0;}
   var value=max(0.0,bitcast<f32>(state[at])-rh.cooling*rh.dt);
   if((trigger&~(RH_ONE_FRAME|RH_WARM_FRAME))!=0u){value=2.0;}
   ${s.particleHeat?"value=max(value,bitcast<f32>(atomicLoad(&rhParticle)));":""}
   ${error&&s.updateErrors?"if(atomicLoad(&rhRelevant)==0u&&(trigger&24u)==0u){value=0.0;}":""}
   state[at]=bitcast<u32>(value);state[at+1u]=bitcast<u32>(displacement);state[at+7u]=trigger|select(0u,4096u,atomicLoad(&rhRelevant)!=0u)|select(0u,8192u,submerged)|select(0u,16384u,fine);
  }
 }
}
// HOT from own evidence; fine surface readers retain a WARM collar.
// Coarse and deep readers keep their own error budget at the interface.
@compute @workgroup_size(64) fn spread(@builtin(global_invocation_id) gid:vec3u){
 let region=gid.x;if(region>=RH_REGIONS){return;}
 if(!umTileLive(umPageTile(region,0u))){atomicStore(&tiers[region],${UNIFORM_REGION_TIER.hot}u);atomicAdd(&tiers[RH_REGIONS+3u],1u);return;}
 let c=vec3i(umPageCoord(region));let own=rhLevel(region);var level=own;
 for(var k=0u;k<27u;k++){
  if(k==13u){continue;}
  let key=umTileKey(4*(c+vec3i(umCorner(k,3u))-vec3i(1)));if(!umKeyStored(key)){continue;}
  ${error?`{let n=rhLevel(umPageOfTile(key));if(n>=1.0${s.updateErrors?"&&rhHotHaloRequired((state[RH_STATE*region+7u]&4096u)!=0u,(state[RH_STATE*region+7u]&8192u)!=0u,(state[RH_STATE*region+7u]&16384u)!=0u)":""}){level=max(level,0.5*n);}}`:"level=max(level,0.5*rhLevel(umPageOfTile(key)));"}
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
    if(s.updateErrors)entries.push({binding:9,resource:{buffer:s.updateErrors}});
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
