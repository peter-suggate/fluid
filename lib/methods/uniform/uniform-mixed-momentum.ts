import { uniformCachedMomentumSamplingWGSL } from "./uniform-cached-momentum.wgsl";
import { uniformDetailBindLayout, uniformDetailExtent, uniformDetailModule, uniformDetailPipeline, uniformDetailGroup, type UniformDetailGroup } from "./uniform-detail-fields";
import {UNIFORM_DETAIL_4H_LOAD,UNIFORM_DETAIL_RING_4H_LOAD} from "../../core/uniform-detail-abi";
import type { UniformMixedOwnership } from "./uniform-mixed-ownership";
import { uniformMixedTopologyWGSL } from "./uniform-mixed-topology.wgsl";
import { uniformMixedVelocitySamplingSource } from "./uniform-mixed-velocity-sampling.wgsl";
import { UNIFORM_MIXED_CLAIMED_GRID, UNIFORM_MIXED_CLAIM_WORDS, uniformMixedClaimedEntriesWGSL, uniformMixedFaceAddressWGSL, uniformMixedFaceTileDispatchWGSL, uniformMixedFarAirWGSL } from "./uniform-mixed-face-dispatch.wgsl";
import { uniformMixedSourceWGSL } from "./uniform-mixed-source.wgsl";
import { uniformVelocityDepartureWGSL } from "./uniform-velocity-departure.wgsl";

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
  /** UniformMixedHangingTaps.unitVelocity of `extended`; with hanging taps only. */
  unitVelocity?: GPUTexture;
  /** h.xyz, dt; openTop, 0, 0, fixed limits (u32 flags). Only openTop is
   * read; the layout matches the other mixed stages' parameter blocks, and
   * fixed limits must be UNIFORM_MIXED_MOMENTUM_LIMITS. */
  params: GPUBuffer;
}

/** Momentum on general h tiles reads prepared h and 4h velocity fields.
 * Classification and blend weight are shared by the three components. The
 * characteristic retains its actual local owner width even when a sample
 * crosses into a coarse tile. Only fine samples touching negative boundary
 * planes escape to the general sampler; seam blends no longer discard and
 * repeat a partially evaluated characteristic. */
const uniformMixedMomentumUnitWGSL = uniformCachedMomentumSamplingWGSL + /* wgsl */ `
// umMomentum through the unit sampler (umUnitEscaped when it cannot).
fn umUnitMomentum(owner:UMOwner,face:UMFace)->f32 {
 if(umCullAir&&!umPredictionCellLive(owner)&&!umPredictionCellLive(face.neighbor)){return 0.0;}
 if(face.anchor[face.axis]<0||umClosedPositive(face)){return umOriginalMomentum(face);}
 let departure=umUnitDeparture(umFaceCenter(face),momentum.hDt.w,momentum.hDt.xyz);
 if(umUnitEscaped){return 0.0;}
 return umUnitSampleComponent(departure,face.axis);
}
var<workgroup> umUnitComponents:array<f32,192>;
var<workgroup> umUnitEscapes:array<atomic<u32>,2>;
// One h tile of the general list per job, one face per lane (as
// uniformMixedFaceTileDispatchWGSL's width-1 jobs). A tile with escaped cells
// is listed once with their mask; their texels are left to momentumDeferred.
@compute @workgroup_size(192) fn momentumUnitStep(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 let cell=lane%64u;let axis=lane/64u;
 if(lane<2u){atomicStore(&umUnitEscapes[lane],0u);}
 workgroupBarrier();
 let owner=umOwner(vec3u(group.x*64u+cell,group.y,0));
 var value=0.0;umUnitEscaped=false;
 if(owner.width!=0u){
  let origin=umOrigin(owner);
  if(origin[axis]==0u){boundary[umNegativeBoundaryIndex(origin,axis)]=umUnitMomentum(owner,umFace(owner,axis,-1,0u));}
  value=umUnitMomentum(owner,umFace(owner,axis,1,0u));
  if(umUnitEscaped){atomicOr(&umUnitEscapes[cell/32u],1u<<(cell%32u));}
 }
 umUnitComponents[lane]=value;workgroupBarrier();
 let escapes=vec2u(atomicLoad(&umUnitEscapes[0]),atomicLoad(&umUnitEscapes[1]));
 if(lane<64u&&owner.width!=0u&&((escapes[cell/32u]>>(cell%32u))&1u)==0u){
  textureStore(output,vec3i(umOrigin(owner)),vec4f(umUnitComponents[cell],umUnitComponents[cell+64u],umUnitComponents[cell+128u],0));
 }
 if(lane==0u&&owner.width!=0u&&any(escapes!=vec2u(0))){
  let entry=atomicAdd(&umDeferred[0],1u);
  atomicStore(&umDeferred[4u+3u*entry],owner.tile);atomicStore(&umDeferred[5u+3u*entry],escapes.x);atomicStore(&umDeferred[6u+3u*entry],escapes.y);
 }
}
// The escaped cells: every positive face of each with the general sampler.
// A fixed grid claims entries from umClaims[umClaimWord]. A listed tile holds
// about a quarter of its cells (the layer against a 4h face), so a round
// takes as many whole entries as fit its 64 cells (lane 0: a few loads an
// entry) and each lane finds its own cell among them: the entry by the
// running cell counts, the cell as that entry's n-th listed bit. An entry
// that does not fit waits with the workgroup that claimed it for its next
// round. Values and stores are per (cell, axis); the packing only moves their
// lanes. Cells stay whole: tracing the escaped faces alone and recomputing
// the others with the unit sampler measured slower (see the storage design).
var<workgroup> umDeferredCount:u32;
var<workgroup> umDeferredFill:u32;
// tile, mask.xy, cells of this and the earlier entries of the round.
var<workgroup> umDeferredEntries:array<vec4u,8>;
var<workgroup> umDeferredComponents:array<f32,192>;
// The n-th set bit of a 64-cell mask (n below its population).
fn umDeferredListedCell(mask:vec2u,n:u32)->u32 {
 let low=countOneBits(mask.x);let high=n>=low;
 let word=select(mask.x,mask.y,high);var rank=select(n,n-low,high);var bit=0u;
 for(var width=16u;width>0u;width>>=1u){
  let below=countOneBits((word>>bit)&((1u<<width)-1u));
  if(rank>=below){rank-=below;bit+=width;}
 }
 return select(0u,32u,high)+bit;
}
@compute @workgroup_size(192) fn momentumDeferred(@builtin(local_invocation_index) lane:u32){
 let slot=lane%64u;let axis=lane/64u;
 if(lane==0u){umDeferredCount=atomicLoad(&umDeferred[0]);}
 let count=workgroupUniformLoad(&umDeferredCount);
 var pending=vec4u(0);
 loop{
  if(lane==0u){
   var fill=0u;var held=0u;
   loop{
    if(held>=8u){break;}
    if(pending.w==0u){
     let entry=atomicAdd(&umClaims[umClaimWord],1u);if(entry>=count){break;}
     pending=vec4u(atomicLoad(&umDeferred[4u+3u*entry]),atomicLoad(&umDeferred[5u+3u*entry]),atomicLoad(&umDeferred[6u+3u*entry]),0u);
     pending.w=countOneBits(pending.y)+countOneBits(pending.z);
     continue;
    }
    if(fill+pending.w>64u){break;}
    fill+=pending.w;umDeferredEntries[held]=vec4u(pending.xyz,fill);held++;pending.w=0u;
   }
   umDeferredFill=fill;
  }
  let fill=workgroupUniformLoad(&umDeferredFill);if(fill==0u){break;}
  let listed=slot<fill;
  var owner=UMOwner();var value=0.0;
  if(listed){
   // Entries past the round's last keep an earlier round's counts; a listed
   // slot is below the last count and stops before them.
   var held=0u;var before=0u;
   for(var i=0u;i<7u;i++){let end=umDeferredEntries[i].w;if(slot<end){break;}before=end;held=i+1u;}
   let entry=umDeferredEntries[held];let cell=umDeferredListedCell(entry.yz,slot-before);
   owner=UMOwner(entry.x,cell,1u,(umTopology[entry.x]&0x3fffffffu)+cell);
   value=umMomentum(owner,umFace(owner,axis,1,0u));
  }
  umDeferredComponents[lane]=value;workgroupBarrier();
  if(lane<64u&&listed){
   textureStore(output,vec3i(umOrigin(owner)),vec4f(umDeferredComponents[slot],umDeferredComponents[slot+64u],umDeferredComponents[slot+128u],0));
  }
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
  /** Job counters of the claimed launches (regular 0, unit 1, merged 2, deferred 3). */
  readonly claims: GPUBuffer;
  /** Tiles of the general h list whose unit launch left cells to the general
   * sampler: a count word, three reserved, then (tile, 64-bit cell mask) per
   * entry. At most every tile of the list: 3 words per capacity tile. */
  readonly deferred: GPUBuffer;
  private pipeline?: GPUComputePipeline;
  private regularPipeline?:GPUComputePipeline;
  private generalPipeline?:GPUComputePipeline;
  private deferredPipeline?:GPUComputePipeline;
  constructor(private readonly device: GPUDevice, readonly ownership: UniformMixedOwnership,
    private readonly cullAir = false, private readonly hanging = false, private readonly sourceParams?: GPUBuffer) {
    // Far-air regular 4h owners (uniformMixedFarAirWGSL) are culled to zero,
    // so the merged launch skips them; forces supplies that zero itself.
    if(sourceParams&&!cullAir)throw new Error("Mixed momentum far-air skipping requires prediction culling");
    this.resources = uniformDetailBindLayout(device,{ entries: [
      ...[0, 1, 2, 3].map(binding => ({ binding, visibility: GPUShaderStage.COMPUTE,
        texture: { sampleType: "unfilterable-float" as const, viewDimension: "3d" as const } })),
      { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 9, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "rgba32float", viewDimension: "3d" } },
      { binding: 10, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 11, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      ...[12,15,...(hanging?[16]:[])].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float" as const,viewDimension:"3d" as const}})),
      { binding: 13, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 14, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      ...(sourceParams?[{binding:17,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform" as const}}]:[]),
    ] });
    this.claims=device.createBuffer({label:"Uniform mixed momentum job claims",size:4*UNIFORM_MIXED_CLAIM_WORDS,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC});
    const deferredBytes=4*(4+3*ownership.capacity.tiles);
    this.deferred=device.createBuffer({label:"Uniform mixed momentum deferred tiles",size:deferredBytes,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
    this.allocatedBytes=4*UNIFORM_MIXED_CLAIM_WORDS+deferredBytes;
  }
  bind(fields: UniformMixedMomentumFields): UniformDetailGroup {
    if(this.cullAir&&!fields.centerPhi)throw new Error("Prediction culling requires current center phi");
    if(this.hanging!==(fields.unitVelocity!==undefined))throw new Error("Mixed momentum hanging taps require their unit velocity texture");
    const textures = [fields.extended, fields.physical, fields.phase, fields.volume];
    const d = this.ownership.capacity.lattice.dimensions;
    for (const [i, texture] of [...textures, fields.output].entries()) {
      if (uniformDetailExtent(texture).some((n, a) => n !== d[a])
        || texture.format !== (i === 2 || i === 3 ? "r32float" : "rgba32float")) throw new Error("Mixed momentum requires native-sized canonical fields");
    }
    for(const t of [fields.coarseExtended])
      if(t.format!=="rgba32float"||uniformDetailExtent(t).some((n,a)=>n!==d[a]!/4+2))throw new Error("Mixed momentum requires a current 4h sampling cache");
    if (textures.includes(fields.output) || fields.negative === fields.outputNegative) throw new Error("Mixed momentum output must be disjoint from its inputs");
    for (const buffer of [fields.negative, fields.outputNegative])
      if (buffer.size < 4 * (d[0] * d[1] + d[0] * d[2] + d[1] * d[2])) throw new Error("Mixed momentum requires complete negative boundary planes");
    return uniformDetailGroup(this.device,{ layout: this.resources, entries: [
      ...textures.map((texture, binding) => ({ binding, resource: texture })),
      { binding: 6, resource: { buffer: fields.negative } },
      { binding: 9, resource: fields.output }, { binding: 10, resource: { buffer: fields.outputNegative } },
      { binding: 11, resource: { buffer: fields.params, size: 32 } },
      {binding:15,resource:fields.centerPhi??fields.volume},
      {binding:12,resource:fields.coarseExtended},
      ...(fields.unitVelocity?[{binding:16,resource:fields.unitVelocity}]:[]),
      {binding:13,resource:{buffer:this.claims}},{binding:14,resource:{buffer:this.deferred}},
      ...(this.sourceParams?[{binding:17,resource:{buffer:this.sourceParams,size:176}}]:[]),
    ] });
  }
  async initialize(): Promise<void> {
    const module = uniformDetailModule(this.device,{ label: "Uniform mixed momentum", code: uniformMixedClaimedEntriesWGSL(uniformMixedTopologyWGSL(this.ownership.capacity, 0) + /* wgsl */ `
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
@group(1) @binding(14) var<storage,read_write> umDeferred:array<atomic<u32>>;
${this.hanging?"@group(1) @binding(16) var unitVelocity:texture_3d<f32>;":""}
override umCullAir:bool=false;
${this.sourceParams?uniformMixedSourceWGSL(17)+uniformMixedFarAirWGSL("umSourceinflowStrength()>0.0"):""}
${uniformMixedFaceAddressWGSL}
fn umLoadCoarseFace(index:vec3i,axis:u32)->f32 {
 return textureLoad(coarseExtended,index+vec3i(1),0)[axis];
}
fn umLoadMixedFace(anchor:vec3i,axis:u32)->f32 {
 if(anchor[axis]<0){return negative[umNegativeBoundaryIndex(vec3u(max(anchor,vec3i(0))),axis)];}return textureLoad(extended,anchor,0)[axis];
}
${uniformMixedVelocitySamplingSource(false,true,"extended",this.hanging?2:undefined,this.hanging?"unitVelocity":undefined)}
fn umClampMomentum(p:vec3f)->vec3f {
 var q=p;q.x=clamp(q.x,0.0,f32(UM_D.x));q.z=clamp(q.z,0.0,f32(UM_D.z));q.y=max(q.y,0.0);
 if(momentum.flags.x==0u){q.y=min(q.y,f32(UM_D.y));}return q;
}
fn umMomentumDeparture(position:vec3f,dt:f32,h:vec3f)->vec3f {
${uniformVelocityDepartureWGSL("umSampleVelocity", "umClampMomentum", "select(f32(umOwnerAt(clamp(vec3i(floor(point)),vec3i(0),vec3i(UM_D)-vec3i(1))).width),1.0,umRegularFine)", "32")}
}
fn umOriginalMomentum(face:UMFace)->f32 {
 if(face.anchor[face.axis]<0){return negative[umNegativeBoundaryIndex(vec3u(max(face.anchor,vec3i(0))),face.axis)];}
 // A 4h-wide patch's anchor is canonical: the base block where the h store
 // keeps only the ring (UNIFORM_DETAIL_RING_4H_LOAD).
 if(face.width==4u){return ${UNIFORM_DETAIL_RING_4H_LOAD}textureLoad(physical,face.anchor,0)[face.axis];}
 return textureLoad(physical,face.anchor,0)[face.axis];
}
fn umClosedPositive(face:UMFace)->bool {
 return face.anchor[face.axis]==i32(UM_D[face.axis])-1 && !(face.axis==1u && momentum.flags.x!=0u);
}
fn umAdvectedMomentum(face:UMFace,dt:f32)->f32 {
 return umSampleVelocityComponent(umMomentumDeparture(umFaceCenter(face),dt,momentum.hDt.xyz),face.axis);
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
fn umMomentum(owner:UMOwner,face:UMFace)->f32 {
 if(umCullAir&&!umPredictionCellLive(owner)&&!umPredictionCellLive(face.neighbor)){return 0.0;}
 // Every closed wall carries its projected velocity; the projection's
 // contact inequality alone releases it. A positive wall that also took an
 // advected away-velocity (min) released where its negative mirror could
 // not, so a D4-symmetric flow drifted off its symmetry at the walls.
 if(face.anchor[face.axis]<0||umClosedPositive(face)){return umOriginalMomentum(face);}
 return umAdvectedMomentum(face,momentum.hDt.w);
}
// A certified unit stencil has one MAC patch per face. Keep its geometry
// constant through the characteristic sampler; generic neighbour widths would
// otherwise make the face position and wall logic dynamically sized.
fn umRegularMomentumFace(owner:UMOwner,axis:u32,sign:i32)->UMFace {
 let origin=vec3i(umOrigin(owner));var probe=origin;probe[axis]+=sign;
 var anchor=origin;if(sign<0){anchor[axis]-=1;}
 return UMFace(umOwnerAt(probe),anchor,1u,1u,axis,sign);
}
// Each lane traces one component. The tile shares only the final packed
// store; each invocation evaluates only one characteristic. No additional
// persistent field is needed.
var<workgroup> umMomentumComponents:array<f32,192>;
@compute @workgroup_size(192) fn momentumRegularStep(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 let cell=lane%64u;let axis=lane/64u;
 let owner=umOwner(vec3u(group.x*64u+cell,group.y,0));
 var value=0.0;
 if(owner.width!=0u){
  let origin=umOrigin(owner);
  if(origin[axis]==0u){boundary[umNegativeBoundaryIndex(origin,axis)]=umMomentum(owner,umRegularMomentumFace(owner,axis,-1));}
  value=umMomentum(owner,umRegularMomentumFace(owner,axis,1));
 }
 umMomentumComponents[lane]=value;workgroupBarrier();
 if(lane<64u&&owner.width!=0u){textureStore(output,vec3i(umOrigin(owner)),vec4f(umMomentumComponents[cell],umMomentumComponents[cell+64u],umMomentumComponents[cell+128u],0));}
}
${uniformMixedFaceTileDispatchWGSL("momentumStep", "umMomentum(owner,face)",undefined,this.sourceParams?"umFarAirOwner":undefined)}
${this.hanging?uniformMixedMomentumUnitWGSL:""}
`, ["momentumRegularStep", "momentumStep",...(this.hanging?["momentumUnitStep"]:[])], "umClaims") });
    const errors = (await module.getCompilationInfo()).messages.filter(m => m.type === "error");
    if (errors.length) throw new Error(errors.map(m => `${m.lineNum}: ${m.message}`).join("\n"));
    const layout = this.device.createPipelineLayout({ bindGroupLayouts: [this.ownership.bindLayout, this.resources, ...(this.hanging ? [this.ownership.hangingLayout] : [])] });
    const compile=(entryPoint:string,constants:Record<string,number>)=>uniformDetailPipeline(this.device,this.ownership,{layout,compute:{module,entryPoint,constants:{umDispatchX:this.ownership.dispatchX,umCullAir:+this.cullAir,...constants}}});
    // The general h list has launches of its own (as the surface stages):
    // with the hanging tap cache, the unit kernel and its deferred cells;
    // merged jobs hold seam and regular 4h only.
    [this.pipeline,this.regularPipeline,this.generalPipeline,this.deferredPipeline]=await Promise.all([
      compile("momentumStep",{umMergedTiles:1,umMergedCoarse:1,umCertifiedJobs:3,umClaimWord:2}),
      compile("momentumRegularStep",{umCellWidth:1,umPlannedFine:1,umRegularFine:1,umCertifiedJobs:1,umClaimWord:0}),
      this.hanging?compile("momentumUnitStep",{umCellWidth:1,umPlannedFine:2,umCertifiedJobs:1,umClaimWord:1})
        :compile("momentumStep",{umCellWidth:1,umPlannedFine:2,umCertifiedJobs:1,umClaimWord:1}),
      ...(this.hanging?[compile("momentumDeferred",{umClaimWord:3})]:[]),
    ]);
  }
  destroy():void{this.claims.destroy();this.deferred.destroy();}
  encode(encoder: GPUCommandEncoder, group: UniformDetailGroup): void {
    if (!this.pipeline) throw new Error("Mixed momentum is not initialized");
    encoder.clearBuffer(this.claims);encoder.clearBuffer(this.deferred,0,16);
    const pass = encoder.beginComputePass({ label: "Uniform mixed momentum" });
    pass.setBindGroup(0, this.ownership.bindGroup); pass.setBindGroup(1,group.group); if (this.hanging) pass.setBindGroup(2, this.ownership.hangingGroup);
    // Buffered direct grids: GPU claims consume the complete current queues.
    const grid=Math.min(UNIFORM_MIXED_CLAIMED_GRID,this.ownership.capacity.tiles);
    for(const pipeline of [this.regularPipeline!,this.generalPipeline!])this.ownership.dispatchBuffered(pass,pipeline,"fine",grid);
    // The deferred cells follow the unit launch that lists them. Their tiles
    // are seam h tiles (a faster characteristic can list others; the claims
    // consume the whole list at any grid) and a round packs several, so the
    // grid follows the seam h count (down to its reserve where no tile is 4h).
    if(this.deferredPipeline)this.ownership.dispatchBuffered(pass,this.deferredPipeline,"seamFine",grid,2);
    this.ownership.dispatchBuffered(pass,this.pipeline,"coarseMerged",grid);
    pass.end();
  }
}
