import { UNIFORM_MOMENTUM_TRACE_SEGMENTS, uniformCachedMomentumSamplingWGSL, type UniformMomentumSampler, type UniformMomentumTrace } from "./uniform-cached-momentum.wgsl";
import { uniformCoarseCacheExtent } from "./uniform-coarse-velocity-cache";
import { uniformDetailBindLayout, uniformDetailClass, uniformDetailExtent, uniformDetailModule, uniformDetailPipeline, uniformDetailGroup, type UniformDetailGroup } from "./uniform-detail-fields";
import {UNIFORM_DETAIL_4H_LOAD,UNIFORM_DETAIL_RING_4H_LOAD} from "../../core/uniform-detail-abi";
import { uniformRegionHeatWGSL } from "./uniform-region-heat";
import type { UniformMixedOwnership } from "./uniform-mixed-ownership";
import { uniformMixedTopologyWGSL } from "./uniform-mixed-topology.wgsl";
import { uniformMixedVelocitySamplingSource } from "./uniform-mixed-velocity-sampling.wgsl";
import { UNIFORM_MIXED_CLAIMED_GRID, UNIFORM_MIXED_CLAIM_WORDS, uniformMixedClaimedEntriesWGSL, uniformMixedFaceAddressWGSL, uniformMixedFaceTileDispatchWGSL, uniformMixedFarAirWGSL } from "./uniform-mixed-face-dispatch.wgsl";
import { UNIFORM_PARAMS_BYTES, uniformMixedSourceWGSL } from "./uniform-mixed-source.wgsl";
import { uniformVelocityDepartureWGSL, uniformVelocitySegmentedDepartureWGSL } from "./uniform-velocity-departure.wgsl";
import { UNIFORM_COLD_FREEZE, UniformColdFreeze } from "./uniform-cold-freeze";
import { uniformTileContextSidesWGSL } from "./uniform-tile-context";

export interface UniformMixedMomentumFields {
  extended: GPUTexture;
  coarseExtended: GPUTexture;
  physical: GPUTexture;
  phase: GPUTexture;
  volume: GPUTexture;
  /** Current post-transport geometry, required when pressure-row culling is enabled. */
  centerPhi?: GPUTexture;
  negative: GPUBuffer;
  output: GPUTexture;
  outputNegative: GPUBuffer;
  /** UniformMixedHangingTaps.unitVelocity of `extended`. */
  unitVelocity?: GPUTexture;
  /** Its halo (UniformMixedHangingTaps.haloOf), filled with it. */
  unitHalo?: GPUTexture;
  /** With the texture16 sampler: the halo's rgba16float copy
   * (UniformMixedHangingTaps.halfHaloOf). */
  unitHaloHalf?: GPUTexture;
  /** h.xyz, dt; four u32 flags, the last the fixed limits. None is read; the
   * layout matches the other mixed stages' parameter blocks, and fixed
   * limits must be UNIFORM_MIXED_MOMENTUM_LIMITS. */
  params: GPUBuffer;
}

const MOMENTUM_PASS: GPUComputePassDescriptor = { label: "Uniform mixed momentum" };

/** Claim word counting, this frame, the substep traces cut at
 * UNIFORM_MOMENTUM_TRACE_SEGMENTS (umTraceCapped): a diagnostic that should
 * stay zero, since so long a travel is past the residency closure's cap. */
export const UNIFORM_MOMENTUM_TRACE_CAPPED_WORD=4;
/** Momentum through the unit sampler (uniformCachedMomentumSamplingWGSL).
 * An h tile job, a face a lane, resolves the tile's context once (its six
 * sides, cold states and the liveness of its cells and the cells across its
 * positive faces) and makes umUnitMomentum's tests from it; a face that
 * needs a trace takes it at once, from one umUnitTrace call site, and the
 * job stores its cells' texels. The merged seam and regular 4h jobs
 * evaluate umUnitMomentum. Every sample stays in the sampler (the unit halo
 * holds the low-wall taps), so no face is deferred. */
const uniformMixedMomentumUnitWGSL = (sampler: UniformMomentumSampler, trace: UniformMomentumTrace) => uniformCachedMomentumSamplingWGSL(sampler, trace) + uniformTileContextSidesWGSL + /* wgsl */ `
// umMomentum through the unit sampler.
// A frozen face keeps its frozen value; a thawing one folds its region's term in.
fn umUnitMomentum(owner:UMOwner,face:UMFace)->f32 {
 let freeze=umFaceFreeze(owner,face);if(freeze.x!=0.0){return umOriginalMomentum(face);}
 return umUnitMomentumValue(owner,face)+freeze.y;
}
fn umUnitMomentumValue(owner:UMOwner,face:UMFace)->f32 {
 if(umMomentumRetained(owner)){return umRetainedFace(face);}
 if(umCullAir&&!umPredictionCellLive(owner)&&!umPredictionCellLive(face.neighbor)){return 0.0;}
 if(umFaceLow(face.anchor,face.axis)||umClosedPositive(face)){return umOriginalMomentum(face);}
 if(umDropFace(face)){return 0.0;}
 return umUnitTrace(umFaceCenter(face),face.axis,momentum.hDt.w,momentum.hDt.xyz);
}
var<workgroup> umUnitComponents:array<f32,192>;
// The tile context: each side's region cold state (6: the tile's own) and
// the liveness (umPredictionCellLive) of the tile's 64 cells, then of the 16
// cells across each positive side (axis a at 64 + 16 a + l[a+1] + 4 l[a+2]).
var<workgroup> umFineCold:array<u32,7>;
// The tile's region is COLD (umMomentumRetained).
var<workgroup> umFineRetained:u32;
var<workgroup> umFineLive:array<u32,112>;
fn umLoadFineTile(t:u32,lane:u32){
 if(lane<6u){
  utcLoadSide(t,lane);let side=utcSides[lane];
  umFineCold[lane]=select(0u,umColdState(umPageOfTile(side.key)),side.width!=0u);
 }
 if(lane==6u){let region=umPageOfTile(t);umFineCold[6]=umColdState(region);umFineRetained=u32(umRegionTier(region)==UM_TIER_COLD);}
 workgroupBarrier();
 if(umCullAir&&lane<112u){
  // Lanes 0..63 the tile's cells; 64.. the cells across the positive sides.
  var at=utcOrigin+vec3i(umCorner(lane,4u));var width=1u;
  if(lane>=64u){
   let j=lane-64u;let a=j/16u;var l=vec3i(0);l[a]=4;l[(a+1u)%3u]=i32(j%4u);l[(a+2u)%3u]=i32((j/4u)%4u);
   width=utcSides[2u*a+1u].width;at=utcOrigin+l;
   // A 4h owner's own texel is its tile origin: the base blocks.
   if(width==4u){at&=vec3i(-4);}
  }
  var live=false;
  if(width==4u){live=${UNIFORM_DETAIL_4H_LOAD}textureLoad(volume,at,0).x>0.0||${UNIFORM_DETAIL_4H_LOAD}textureLoad(centerPhi,at,0).x<0.0;}
  else if(width==1u){live=textureLoad(volume,at,0).x>0.0||textureLoad(centerPhi,at,0).x<0.0;}
  umFineLive[lane]=u32(live);
 }
 workgroupBarrier();
}
fn umFineOriginal(face:UMFace,low:bool)->f32 {
 if(low){return negative[umLowFaceIndex(umLowFaceAbove(face.anchor,face.axis),face.axis)];}
 return textureLoad(physical,face.anchor,0)[face.axis];
}
// umUnitMomentum for a face of the h owner cell (first cell origin) from
// the tile context. Only an open low face is a negative one.
fn umFineMomentum(owner:UMOwner,origin:vec3i,cell:u32,axis:u32,sign:i32)->f32 {
 var probe=origin;probe[axis]+=sign;
 let inside=all((probe>>vec3u(2u))==(origin>>vec3u(2u)));
 let s=2u*axis+u32(sign>0);
 // umFaceLow (negative) or umClosedPositive (positive): a SOLID page across.
 let wall=!inside&&utcSides[s].kind>=UM_SOLID;let low=wall&&sign<0;
 var anchor=origin;if(sign<0){anchor[axis]-=1;}
 let face=UMFace(UMOwner(),anchor,1u,1u,axis,sign);
 // umFaceFreeze: frozen, else the thawing term (the owner's, else the neighbour's).
 var thaw=0.0;
 if(UM_COLD_FREEZE){
  let own=umFineCold[6];var other=own;var region=umPageOfTile(owner.tile);
  if(!inside&&utcSides[s].width!=0u){other=umFineCold[s];}
  if(own==1u||own==2u||other==1u||other==2u){return umFineOriginal(face,low);}
  if(own==3u||other==3u){
   if(own!=3u){region=umPageOfTile(utcSides[s].key);}
   thaw=umColdTermOf(region,axis);
  }
 }
 // umRetainedFace.
 if(umFineRetained!=0u){
  if(low){return umFineOriginal(face,true)+thaw;}
  if(umDropFace(face)&&!wall){return thaw;}
  return textureLoad(extended,anchor,0)[axis]+thaw;
 }
 if(umCullAir){
  let l=umCorner(cell,4u);var near=0u;
  if(inside){near=umFineLive[u32(i32(cell)+sign*i32(vec3u(1u,4u,16u)[axis]))];}
  else if(sign>0){near=umFineLive[64u+16u*axis+l[(axis+1u)%3u]+4u*l[(axis+2u)%3u]];}
  if(umFineLive[cell]==0u&&near==0u){return thaw;}
 }
 if(wall){return umFineOriginal(face,low)+thaw;}
 if(umDropFace(face)){return thaw;}
 var center=vec3f(origin)+vec3f(0.5);center[axis]=f32(origin[axis]+select(0,1,sign>0));
 return umUnitTrace(center,axis,momentum.hDt.w,momentum.hDt.xyz)+thaw;
}
// h tile job: the tile at h list index job, one face a lane.
fn umFineJob(job:u32,lane:u32){
 let cell=lane%64u;let axis=lane/64u;
 let t=umTopology[UM_TILES+job];let owner=UMOwner(t,cell,1u,(umTopology[t]&0x3fffffffu)+cell);
 umLoadFineTile(t,lane);
 let origin=utcOrigin+vec3i(umCorner(cell,4u));
 var value=0.0;
 // A frozen owner's texels and low walls hold (UniformColdFreeze).
 let live=umFineCold[6]!=1u&&umFineCold[6]!=2u;
 if(live){
  // The low boundary face where it is open (nothing stored below), then the
  // positive one: one umFineMomentum site.
  let low=(origin[axis]&31)==0&&utcSides[2u*axis].width==0u;
  for(var sign=select(1,-1,low);sign<=1;sign+=2){
   let v=umFineMomentum(owner,origin,cell,axis,sign);
   if(sign<0){boundary[umLowFaceIndex(vec3u(origin),axis)]=v;}else{value=v;}
  }
 }
 umUnitComponents[lane]=value;workgroupBarrier();
 if(lane<64u&&live){textureStore(output,origin,vec4f(umUnitComponents[cell],umUnitComponents[cell+64u],umUnitComponents[cell+128u],0));}
}
// One launch: a fixed grid whose workgroups claim jobs from umClaims[0]
// (zeroed each frame): every h tile (the h list), then the merged seam and
// regular 4h jobs (momentumStep's, umCertifiedJobCount).
var<workgroup> umMomentumJobs:u32;
var<workgroup> umMomentumFine:u32;
var<workgroup> umMomentumClaim:u32;
@compute @workgroup_size(192) fn advectMomentum(@builtin(local_invocation_index) lane:u32){
 if(lane==0u){umMomentumFine=umCounts.x;umMomentumJobs=umCounts.x+umCertifiedJobCount();}
 let fine=workgroupUniformLoad(&umMomentumFine);let jobs=workgroupUniformLoad(&umMomentumJobs);
 loop{
  if(lane==0u){umMomentumClaim=atomicAdd(&umClaims[0],1u);}
  let job=workgroupUniformLoad(&umMomentumClaim);
  if(job>=jobs){break;}
  if(job<fine){umFineJob(job,lane);}else{momentumStepJob(vec3u(job-fine,0u,0u),lane);}
  workgroupBarrier();
 }
}
`;

export const UNIFORM_MIXED_MOMENTUM_LIMITS = 32 | (16 << 8);

/** Native semi-Lagrangian momentum resampling on canonical MAC faces. Extension
 * and forces are separate stages. All fields are borrowed. */
export class UniformMixedMomentum {
  readonly allocatedBytes: number;
  private readonly resources: GPUBindGroupLayout;
  /** The launch's job counter (word 0), then the capped-trace count
   * (UNIFORM_MOMENTUM_TRACE_CAPPED_WORD). */
  readonly claims: GPUBuffer;
  /** The texture samplers' linear filter (texture32, texture16). */
  private readonly filter?: GPUSampler;
  private pipeline?: GPUComputePipeline;
  constructor(private readonly device: GPUDevice, readonly ownership: UniformMixedOwnership,
    /** heat: the frame's region work tiers (UniformRegionHeat.tiers). A COLD region's faces retain. */
    private readonly heat: GPUBuffer,
    private readonly cullAir = false, private readonly sourceParams?: GPUBuffer,
    /** COLD freeze: frozen owners hold, frozen faces keep their value. */
    private readonly freeze?: UniformColdFreeze,
    /** The unit sampler's interpolation: texture32 requires float32-filterable. */
    readonly sampler: UniformMomentumSampler = "software",
    /** The unit trace's characteristic (UniformMomentumTrace). */
    readonly trace: UniformMomentumTrace = "midpoint") {
    // Far-air regular 4h owners (uniformMixedFarAirWGSL) are culled to zero,
    // so the merged jobs skip them; forces supplies that zero itself.
    if(sourceParams&&!cullAir)throw new Error("Mixed momentum far-air skipping requires prediction culling");
    if(sampler==="texture32"&&!device.features.has("float32-filterable"))throw new Error("Mixed momentum texture32 sampling requires float32-filterable");
    const filtered=sampler==="software"?[]:[22,...(sampler==="texture32"?[24]:[])];
    this.resources = uniformDetailBindLayout(device,{ entries: [
      ...[0, 1, 2, 3].map(binding => ({ binding, visibility: GPUShaderStage.COMPUTE,
        texture: { sampleType: "unfilterable-float" as const, viewDimension: "3d" as const } })),
      { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 19, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 9, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "rgba32float", viewDimension: "3d" } },
      { binding: 10, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 11, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      ...[12,15,16,18].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float" as const,viewDimension:"3d" as const}})),
      { binding: 13, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      ...(sourceParams?[{binding:17,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform" as const}}]:[]),
      ...(freeze?[{binding:21,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float" as const,viewDimension:"2d" as const}}]:[]),
      ...filtered.map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"float" as const,viewDimension:"3d" as const}})),
      ...(filtered.length?[{binding:23,visibility:GPUShaderStage.COMPUTE,sampler:{type:"filtering" as const}}]:[]),
    ] },filtered);
    if(filtered.length)this.filter=device.createSampler({label:"Uniform mixed momentum filter",magFilter:"linear",minFilter:"linear",addressModeU:"clamp-to-edge",addressModeV:"clamp-to-edge",addressModeW:"clamp-to-edge"});
    this.claims=device.createBuffer({label:"Uniform mixed momentum job claims",size:4*UNIFORM_MIXED_CLAIM_WORDS,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC});
    this.allocatedBytes=4*UNIFORM_MIXED_CLAIM_WORDS;
  }
  bind(fields: UniformMixedMomentumFields): UniformDetailGroup {
    if(this.cullAir&&!fields.centerPhi)throw new Error("Prediction culling requires current center phi");
    if(!fields.unitVelocity||!fields.unitHalo)throw new Error("Mixed momentum traces through the hanging taps: it requires their unit velocity texture and its halo");
    if((this.sampler==="texture16")!==(fields.unitHaloHalf!==undefined)||(fields.unitHaloHalf&&fields.unitHaloHalf.format!=="rgba16float"))throw new Error("Mixed momentum texture16 sampling requires the halo's rgba16float copy");
    const filtered=this.sampler==="texture32"?[{binding:22,resource:fields.unitHalo},{binding:24,resource:fields.coarseExtended}]
      :this.sampler==="texture16"?[{binding:22,resource:fields.unitHaloHalf!}]:[];
    const textures = [fields.extended, fields.physical, fields.phase, fields.volume];
    const address = this.ownership.capacity.address;
    for (const [i, texture] of [...textures, fields.output].entries()) {
      if (uniformDetailClass(texture) !== (i === 2 || i === 3 ? "cell" : "face")
        || texture.format !== (i === 2 || i === 3 ? "r32float" : "rgba32float")) throw new Error("Mixed momentum requires native-sized canonical fields");
    }
    for(const t of [fields.coarseExtended])
      if(t.format!=="rgba32float"||uniformDetailExtent(t).some((n,a)=>n!==uniformCoarseCacheExtent(address)[a]))throw new Error("Mixed momentum requires a current 4h sampling cache");
    if (textures.includes(fields.output) || fields.negative === fields.outputNegative) throw new Error("Mixed momentum output must be disjoint from its inputs");
    for (const buffer of [fields.negative, fields.outputNegative])
      if (buffer.size < 4 * address.lowFaceKeys) throw new Error("Mixed momentum requires complete negative boundary planes");
    return uniformDetailGroup(this.device,{ layout: this.resources, entries: [
      ...textures.map((texture, binding) => ({ binding, resource: texture })),
      { binding: 6, resource: { buffer: fields.negative } },
      { binding: 9, resource: fields.output }, { binding: 10, resource: { buffer: fields.outputNegative } },
      { binding: 11, resource: { buffer: fields.params, size: 32 } },
      {binding:15,resource:fields.centerPhi??fields.volume},
      {binding:12,resource:fields.coarseExtended},
      {binding:16,resource:fields.unitVelocity},{binding:18,resource:fields.unitHalo},
      {binding:13,resource:{buffer:this.claims}},{binding:19,resource:{buffer:this.heat}},
      ...(this.sourceParams?[{binding:17,resource:{buffer:this.sourceParams,size:UNIFORM_PARAMS_BYTES}}]:[]),
      ...(this.freeze?[{binding:21,resource:this.freeze.regions.createView()}]:[]),
      ...filtered,...(this.filter?[{binding:23,resource:this.filter}]:[]),
    ] });
  }
  async initialize(): Promise<void> {
    const module = uniformDetailModule(this.device,{specializeWorldGeometry:true, label: "Uniform mixed momentum", code: uniformMixedClaimedEntriesWGSL(uniformMixedTopologyWGSL(this.ownership.capacity, 0) + /* wgsl */ `
@group(1) @binding(0) var extended:texture_3d<f32>;
@group(1) @binding(1) var physical:texture_3d<f32>;
@group(1) @binding(2) var phase:texture_3d<f32>;
@group(1) @binding(3) var volume:texture_3d<f32>;
@group(1) @binding(6) var<storage,read> negative:array<f32>;
@group(1) @binding(9) var output:texture_storage_3d<rgba32float,write>;
@group(1) @binding(10) var<storage,read_write> boundary:array<f32>;
struct UMMomentumParams {hDt:vec4f,flags:vec4u}
@group(1) @binding(11) var<uniform> momentum:UMMomentumParams;
@group(1) @binding(12) var coarseExtended:texture_3d<f32>;
@group(1) @binding(15) var centerPhi:texture_3d<f32>;
@group(1) @binding(13) var<storage,read_write> umClaims:array<atomic<u32>>;
@group(1) @binding(16) var unitVelocity:texture_3d<f32>;
@group(1) @binding(18) var unitHalo:texture_3d<f32>;
${this.sampler!=="software"?"@group(1) @binding(22) var unitHaloFiltered:texture_3d<f32>;\n@group(1) @binding(23) var umMomentumSampler:sampler;":""}
${this.sampler==="texture32"?"@group(1) @binding(24) var coarseFiltered:texture_3d<f32>;":""}
${this.trace==="substep"?`const UM_TRACE_CAPPED:u32=${UNIFORM_MOMENTUM_TRACE_CAPPED_WORD}u;
fn umTraceCapped(){atomicAdd(&umClaims[UM_TRACE_CAPPED],1u);}`:""}
${UniformColdFreeze.wgsl(this.freeze,1,21)}
${uniformRegionHeatWGSL(1,19,this.ownership.capacity.tiles/64)}
// A frozen owner (UniformColdFreeze): its texels and low walls hold.
fn umOwnerFrozen(owner:UMOwner)->bool{return owner.width!=0u&&umFrozenRegion(umPageOfTile(owner.tile));}
// x: 1 for a frozen face (an incident cell frozen), y: a thawing region's
// term (the owner's, else the neighbour's).
fn umFaceFreeze(owner:UMOwner,face:UMFace)->vec2f{
 if(!UM_COLD_FREEZE){return vec2f(0);}
 let own=umColdState(umPageOfTile(owner.tile));var other=own;var region=umPageOfTile(owner.tile);
 if(face.neighbor.width!=0u&&face.neighbor.tile!=owner.tile){other=umColdState(umPageOfTile(face.neighbor.tile));}
 if(own==1u||own==2u||other==1u||other==2u){return vec2f(1.0,0.0);}
 if(own!=3u){if(other!=3u){return vec2f(0);}region=umPageOfTile(face.neighbor.tile);}
 return vec2f(0.0,umColdTermOf(region,face.axis));
}
override umCullAir:bool=false;
${this.sourceParams?uniformMixedSourceWGSL(17)+uniformMixedFarAirWGSL("umSourceinflowStrength()>0.0"):""}
${uniformMixedFaceAddressWGSL}
fn umLoadMixedFace(anchor:vec3i,axis:u32)->f32 {
 if(umFaceLow(anchor,axis)){return negative[umLowFaceIndex(umLowFaceAbove(anchor,axis),axis)];}return textureLoad(extended,anchor,0)[axis];
}
${uniformMixedVelocitySamplingSource(false,"coarseExtended","extended",2,"unitVelocity",undefined,"unitHalo")}
// A departure stops at a wall, a face with a SOLID page across it, and
// nowhere else: it runs on into an AIR page.
fn umClampMomentum(p:vec3f)->vec3f {return umPointHeld(p);}
fn umOriginalMomentum(face:UMFace)->f32 {
 if(umFaceLow(face.anchor,face.axis)){return negative[umLowFaceIndex(umLowFaceAbove(face.anchor,face.axis),face.axis)];}
 // A 4h-wide patch's anchor is canonical: the base block where the h store
 // keeps only the ring (UNIFORM_DETAIL_RING_4H_LOAD).
 if(face.width==4u){return ${UNIFORM_DETAIL_RING_4H_LOAD}textureLoad(physical,face.anchor,0)[face.axis];}
 return textureLoad(physical,face.anchor,0)[face.axis];
}
// A COLD region retains (UNIFORM_REGION_HEAT_BUDGET_H): the heat holds its
// displacement within budget over the frames it stays COLD, and triggers on
// bodies, sources, identity and unknown evidence. Its faces take this frame's
// extended value, which the last projection corrected, and skip the trace.
fn umMomentumRetained(owner:UMOwner)->bool{return umRegionTier(umPageOfTile(owner.tile))==UM_TIER_COLD;}
fn umRetainedMomentum(face:UMFace)->f32 {
 if(face.width==4u){return ${UNIFORM_DETAIL_RING_4H_LOAD}textureLoad(extended,face.anchor,0)[face.axis];}
 return textureLoad(extended,face.anchor,0)[face.axis];
}
// A retained region's face ahead of the cull and the closed-wall test: a
// low face still reads its table (its anchor's page is not stored); a closed
// positive face's extended value is its physical one (the extension publishes
// closed walls from physical); a culled face (both cells dead) is no pressure
// row and the projection zeroes it, so its retained value is never read. The
// drop keeps its zero below an open face.
fn umRetainedFace(face:UMFace)->f32 {
 if(umFaceLow(face.anchor,face.axis)){return umOriginalMomentum(face);}
 if(umDropFace(face)&&!umClosedPositive(face)){return 0.0;}
 return umRetainedMomentum(face);
}
fn umClosedPositive(face:UMFace)->bool {
 return umAcrossKind(face.anchor,face.axis,1)>=UM_SOLID;
}
// An exterior neighbor is not a pressure row. If the incident interior owner
// is empty too, projection writes zero and no RHS can read this prediction.
fn umPredictionCellLive(owner:UMOwner)->bool {
 if(owner.width==0u){return false;}
 let at=vec3i(umOrigin(owner));
 // A 4h owner's own texel is its tile origin: the base blocks.
 if(!umRegularFine&&owner.width==4u){return ${UNIFORM_DETAIL_4H_LOAD}textureLoad(volume,at,0).x>0.0||${UNIFORM_DETAIL_4H_LOAD}textureLoad(centerPhi,at,0).x<0.0;}
 return textureLoad(volume,at,0).x>0.0||textureLoad(centerPhi,at,0).x<0.0;
}
// A face within a patch width of this step's drop borders liquid the source
// made. That liquid arrives at rest: the characteristic would instead hand it
// whatever the air it replaces was extrapolated to from the liquid around it,
// and a body born with that field is torn apart by its own projection.
fn umDropFace(face:UMFace)->bool {
 ${this.sourceParams?/* wgsl */`if(umSourceParams.drop.w<=0.0){return false;}
 let h=umSourceParams.cellGravity.xyz;let reach=f32(face.width)*max(h.x,max(h.y,h.z));
 return umSourcedropDistance(umSourcetraceWorld(umFaceCenter(face)),reach)<reach;`:"return false;"}
}
${uniformMixedFaceTileDispatchWGSL("momentumStep","umUnitMomentum(owner,face)",undefined,this.sourceParams?"umFarAirOwner":undefined,"umOwnerFrozen")}
${uniformMixedMomentumUnitWGSL(this.sampler,this.trace)}
`, ["momentumStep"], "umClaims") });
    const errors = (await module.getCompilationInfo()).messages.filter(m => m.type === "error");
    if (errors.length) throw new Error(errors.map(m => `${m.lineNum}: ${m.message}`).join("\n"));
    const layout = this.device.createPipelineLayout({ bindGroupLayouts: [this.ownership.bindLayout, this.resources, this.ownership.hangingLayout] });
    // One launch: the h list's tiles, then the merged seam and regular 4h
    // jobs. Deep-COLD owners are frozen (no writes): the 4h jobs enumerate
    // the active regions only.
    this.pipeline=await uniformDetailPipeline(this.device,this.ownership,{layout,compute:{module,entryPoint:"advectMomentum",constants:{umDispatchX:this.ownership.dispatchX,umCullAir:+this.cullAir,
      umMergedTiles:1,umMergedCoarse:1,umCertifiedJobs:3,umActiveCoarse:+(UNIFORM_COLD_FREEZE&&!!this.freeze)}}});
  }
  destroy():void{this.claims.destroy();}
  encode(encoder: GPUCommandEncoder, group: UniformDetailGroup): void {
    if (!this.pipeline) throw new Error("Mixed momentum is not initialized");
    encoder.clearBuffer(this.claims);
    const pass = encoder.beginComputePass(MOMENTUM_PASS);
    pass.setBindGroup(0, this.ownership.bindGroup); pass.setBindGroup(1,group.group); pass.setBindGroup(2, this.ownership.hangingGroup);
    // A fixed grid: its claim loop consumes the complete current queues.
    this.ownership.dispatchBuffered(pass,this.pipeline,"merged",Math.min(UNIFORM_MIXED_CLAIMED_GRID,this.ownership.capacity.tiles));
    pass.end();
  }
}
