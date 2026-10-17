import { uniformSurfaceFarCellWGSL } from "./uniform-surface-far-cell.wgsl";
import { UNIFORM_SURFACE_SUPPORT_WORDS, UNIFORM_SURFACE_VELOCITY_REACH_H, uniformSurfaceTransportWGSL } from "./uniform-surface-transport.wgsl";
import { uniformSurfaceSmoothingWGSL } from "./uniform-surface-smoothing.wgsl";
import type { UniformSurfaceWorkControl } from "./uniform-surface-work-control";
import { NARROW_BAND_TRACE_LIMIT, narrowBandTraceWGSL } from "./uniform-narrow-band-advection.wgsl";
import { UNIFORM_DETAIL_H_STORE, UNIFORM_DETAIL_MERGED_SLOTS, uniformDetailBindLayout, uniformDetailClass, uniformDetailExtent, uniformDetailModule, uniformDetailPipeline, uniformDetailPick, uniformDetailGroup, type UniformDetailGroup } from "./uniform-detail-fields";
import {UNIFORM_DETAIL_4H_LOAD,UNIFORM_DETAIL_RING_4H_LOAD} from "../../core/uniform-detail-abi";
import {UNIFORM_PARAMS_BYTES,uniformMixedSourceWGSL} from "./uniform-mixed-source.wgsl";
import type { UniformMixedOwnership } from "./uniform-mixed-ownership";
import { uniformMixedTopologyNeighboursWGSL, uniformMixedTopologyWGSL } from "./uniform-mixed-topology.wgsl";
import { uniformMixedWallAheadWGSL, uniformMixedWallFaceWGSL, uniformMixedWallVertexWGSL } from "./uniform-mixed-walls.wgsl";
import { uniformMixedVertexSamplingSource } from "./uniform-mixed-vertex-sampling.wgsl";
import { uniformMixedVelocitySamplingSource } from "./uniform-mixed-velocity-sampling.wgsl";
import { uniformCoarseCacheExtent } from "./uniform-coarse-velocity-cache";
import { UNIFORM_MIXED_CLAIMED_GRID, UNIFORM_MIXED_CLAIM_WORDS, uniformMixedClaimedEntriesWGSL, uniformMixedFaceAddressWGSL } from "./uniform-mixed-face-dispatch.wgsl";
import { uniformMixedSolidPipeline, uniformMixedSolidWGSL, uniformMixedVertexBuriedWGSL, type UniformMixedSolid } from "./uniform-mixed-solid.wgsl";
import { uniformFaceTileSlots, uniformKeyGroups, type UniformLatticeAddress } from "./uniform-lattice-address";

/** Head of the surface claim buffer: job counters, six wall reach words (a
 * face direction each) and two pad words; the reach of every tile of every
 * block face (umFaceTile) follows it. */
const SURFACE_CLAIM_WORDS=UNIFORM_MIXED_CLAIM_WORDS+8;
const surfaceClaimWords=(a:UniformLatticeAddress)=>SURFACE_CLAIM_WORDS+uniformFaceTileSlots(a.blocks);
/** The wall faces a pool can list: six a stored block. */
const wallFaces=(a:UniformLatticeAddress)=>6*(a.blocks-2);
/** The largest launches of wallReach and advectWalls: past them a workgroup strides the wall-face list. */
const WALL_REACH_GRID=1024,WALL_VERTEX_GRID=4096;
/** Per second: the NB-FLIP surface vertices' fourth-difference weight (umVertexFoot). */
export const UNIFORM_MIXED_VERTEX_SMOOTHING_PER_S=0.5;
/** encode()'s pass descriptors. */
const SURFACE_PASS:GPUComputePassDescriptor={label:"Uniform mixed surface advect"};
const DEFERRED_PASS:GPUComputePassDescriptor={label:"Uniform mixed surface advect deferred"};

export interface UniformMixedSurfaceFields {
  /** NB-FLIP sticky trajectory failure receipt. */
  narrowBandState:GPUBuffer;
  phi: GPUTexture;
  outputPhi: GPUTexture;
  velocity: GPUTexture;
  coarseVelocity: GPUTexture;
  negative: GPUBuffer;
  /** h.xyz, dt; flags: x is not read, y cubic (1: every vertex, 2: 4h vertices only). */
  params: GPUBuffer;
  /** Scratch, dead before transport: with solids two words per tile (closed,
   * clear), then from a 256-aligned offset the deferred advect list. */
  evidence: GPUBufferBinding;
  /** UniformMixedHangingTaps.unitVelocity of `velocity`. */
  unitVelocity: GPUTexture;
  /** Its halo (UniformMixedHangingTaps.haloOf). */
  unitHalo: GPUTexture;
}

/** Narrow-band phi advection on canonical vertices of a resolved phi field
 * (UniformMixedPhiResolve): a vertex beside the surface is carried by its foot
 * on the zero set, every other along its own characteristic. Hanging values
 * are sampled from their authority, never expanded. Static solids take the
 * walked trace, the buried-vertex freeze and a released embedded wall's air
 * continuation; they live near fine owners only. */
export class UniformMixedSurface {
  readonly allocatedBytes:number;
  /** Bounded compact owner cache. Higher fine owner indices use full work. */
  readonly predictorFineCapacity:number;
  readonly predictorCacheBytes:number;
  private readonly claimBytes:number;
  private readonly resources: GPUBindGroupLayout;
  /** Job counters and wall reach, cleared per frame. */
  readonly claims: GPUBuffer;
  private readonly evidenceBytes:number;
  private readonly modelWord:number;
  private readonly velocityWord:number;
  private readonly supportWord:number;
  private readonly counterWord:number;
  private readonly regionCounterWord:number;
  private readonly revisionWord:number;
  get regionCounters(){return {buffer:this.claims,offset:4*this.regionCounterWord,regions:this.workControl?.regions??0,wordsPerRegion:5,labels:["identity","affine","middle","fullLinear","fullCubic"] as const};}
  readonly workCounterWords=22;
  /** Trial diagnostics: admitted fast-fine attempts and direct/cubic-listed results. */
  get localSolidAdmissionCountersOffset():number{return 4*(this.counterWord+20);}
  /** Two u32 counters: predicted vertices, fully evaluated vertices. */
  get workCountersOffset():number{return 4*this.counterWord;}
  /** wallReach, solidClosed and solidClear, then advect by launch: advect
   * (merged), advectFine (the general h list, a lane per vertex) and
   * advectOwners (regular h). */
  private readonly pipelines = new Map<string, GPUComputePipeline>();
  /** advectWalls and advectDeferred for the merged and regular dispatches, and the deferred grid. */
  private readonly deferredPipelines:GPUComputePipeline[]=[];
  /** advectDeferred's solid-free twins (merged, regular), launched ahead of
   * the general ones where the stage has a solid library: see umSolidSkip. */
  private readonly freeDeferredPipelines:GPUComputePipeline[]=[];
  /** advectDeferred's fast twin (umFastSurface, solid-free) of the merged
   * dispatch, launched first: it retires the entries it evaluates. The
   * regular dispatch has no halo homes (umUnitHaloHome), so no fast twin. */
  private fastDeferredPipeline?:GPUComputePipeline;
  /** Workgroups of advectCubic: its list's bound (cubicEntries), capped
   * where the GPU is saturated. */
  private readonly cubicGrid:number;
  /** The cubic list's bound: every vertex advectFine can evaluate, 125 a fine tile. */
  private readonly cubicEntries:number;
  /** Workgroups of the deferred advect launches: the list's bound (one word
   * per lattice vertex), capped where the GPU is saturated. */
  private readonly deferredGrid:number;
  /** Launch widths the pool fixes: wallReach (a workgroup a wall face) and
   * advectWalls (a lane a wall-face vertex), each striding the wall-face list. */
  private readonly wallGroups:number;
  private readonly deferredWallGroups:number;
  private readonly address:UniformLatticeAddress;
  /** Dense per-tile launches: a lane a tile (64 a group), and a group a tile. */
  private readonly tileLaneGroups:[number,number]=[0,0];
  private readonly tileGroups:[number,number]=[0,0];
  /** farGate: a word a tile key from `word` of `buffer` (NB-FLIP's air
   * gate, NARROW_BAND_AIR_GATE_PLANE), zero where the last clear certified a
   * tile far air beyond its collar's edge: its far-marker vertices keep
   * their value (umFarAir). encode() copies it, and the frame plan's speed
   * certificate, behind the claim words (the stage's storage bindings are at
   * the per-stage limit). */
  constructor(private readonly device: GPUDevice, readonly ownership: UniformMixedOwnership,private readonly sourceParams?:GPUBuffer,private readonly solid?:UniformMixedSolid,private readonly farGate?:{readonly buffer:GPUBuffer;readonly word:number},private readonly workControl?:UniformSurfaceWorkControl) {
    this.resources = uniformDetailBindLayout(device,{entries:[
      ...[0,2,3,11].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float" as const,viewDimension:"3d" as const}})),
      {binding:1,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"write-only",format:"r32float",viewDimension:"3d"}},
      {binding:5,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},
      {binding:6,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},
      ...(solid?[{binding:8,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage" as const}}]:[]),
      ...(sourceParams?[{binding:9,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform" as const}}]:[]),
      ...[10,12,13].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage" as const}})),
      // The unit taps' halo: no detail field.
      {binding:14,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float",viewDimension:"3d"}},
    ]},[14]);
    const address=this.address=ownership.capacity.address;
    // A deferred advect word holds a vertex key in 30 bits (all ones is no word): 29,878 pages of 33³ vertices.
    if(address.vertexKeys>0x3fffffff)throw new Error(`Mixed surface deferred advect packs a vertex key in 30 bits: ${address.vertexKeys} vertex keys exceed 1073741823 (29,878 pages of 35,937)`);
    this.claimBytes=4*surfaceClaimWords(address);
    // Behind the claims: the far gate's words, then the certificate's (umDeepAir).
    this.evidenceBytes=this.claimBytes+4*ownership.capacity.tiles*(farGate?2:1);
    this.modelWord=this.evidenceBytes/4+(workControl?.regions??0);
    this.velocityWord=this.modelWord+8*(workControl?.regions??0);
    this.supportWord=this.velocityWord+6*(workControl?.regions??0);
    this.counterWord=this.supportWord+64*UNIFORM_SURFACE_SUPPORT_WORDS*(workControl?.regions??0);
    this.regionCounterWord=this.counterWord+this.workCounterWords;
    this.revisionWord=this.regionCounterWord+5*(workControl?.regions??0);
    this.predictorFineCapacity=0;
    this.predictorCacheBytes=0;
    this.allocatedBytes=this.evidenceBytes+(workControl?4*((21+64*UNIFORM_SURFACE_SUPPORT_WORDS)*workControl.regions+this.workCounterWords)+this.predictorCacheBytes:0);
    const claimsLimit=Math.min(device.limits.maxStorageBufferBindingSize,device.limits.maxBufferSize);
    if(this.allocatedBytes>claimsLimit)throw new Error(`Uniform mixed surface claims and 16h support tables require ${this.allocatedBytes} bytes, exceeding the device buffer limit ${claimsLimit}; reduce scene capacity or resolution`);
    this.claims=device.createBuffer({label:"Uniform mixed surface job claims",size:this.allocatedBytes,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC});
    this.deferredGrid=Math.min(1024,Math.ceil(address.vertexKeys/64));
    this.cubicEntries=125*ownership.capacity.fineTiles;
    this.cubicGrid=Math.max(1,Math.min(1024,Math.ceil(this.cubicEntries/64)));
    this.wallGroups=Math.min(WALL_REACH_GRID,wallFaces(address));
    this.deferredWallGroups=Math.min(WALL_VERTEX_GRID,Math.ceil(1089*wallFaces(address)/64));
    uniformKeyGroups(address.tileKeys,64,ownership.dispatchX,this.tileLaneGroups);uniformKeyGroups(address.tileKeys,1,ownership.dispatchX,this.tileGroups);
  }
  /** bind()'s evidence scratch from a 256-aligned offset: the tile words,
   * then the deferred advect list and the cubic list. */
  get scratchBytes():number{
    const c=this.ownership.capacity;
    return Math.ceil((this.solid?8*c.tiles:0)/256)*256+this.deferredBytes;
  }
  /** The deferred binding: count words, a word per lattice vertex (the
   * deferred list), then five words per cubic entry (advectCubic). */
  private get deferredBytes():number{return 16+4*this.address.vertexKeys+20*this.cubicEntries;}
  bind(f: UniformMixedSurfaceFields): UniformDetailGroup {
    const a=this.address;
    for(const [i,t] of [f.phi,f.outputPhi,f.velocity,f.coarseVelocity].entries()){
      // The 4h velocity cache is keyed by block.
      if(t.format!==(i>=2?"rgba32float":"r32float") || (i===3?uniformDetailExtent(t).some((n,k)=>n!==uniformCoarseCacheExtent(a)[k]):uniformDetailClass(t)!==(i<2?"vertex":"face")))
        throw new Error("Mixed surface requires native vertex/cell fields and a 4h velocity cache");
    }
    if(f.phi===f.outputPhi)throw new Error("Mixed surface outputs overlap inputs");
    // The solid closed and clear tile words.
    const evidenceBytes=this.solid?8*this.ownership.capacity.tiles:0;
    // The deferred advect list follows, a separate binding: its count, the
    // cubic list's count, two unused words, then at most one word per
    // lattice vertex, then the cubic list (deferredBytes).
    const deferredOffset=Math.ceil(((f.evidence.offset??0)+evidenceBytes)/256)*256,deferredBytes=this.deferredBytes;
    if((f.evidence.size??f.evidence.buffer.size-(f.evidence.offset??0))<deferredOffset-(f.evidence.offset??0)+deferredBytes)throw new Error("Mixed surface evidence scratch is too small");
    const group=uniformDetailGroup(this.device,{layout:this.resources,entries:[
      ...[f.phi,f.outputPhi,f.velocity,f.coarseVelocity].map((t,binding)=>({binding,resource:t})),
      {binding:5,resource:{buffer:f.negative}},{binding:6,resource:{buffer:f.params,size:32}},
      ...(this.solid?[{binding:8,resource:{...f.evidence,size:evidenceBytes}}]:[]),
      {binding:10,resource:{buffer:f.evidence.buffer,offset:deferredOffset,size:deferredBytes}},
      ...(this.sourceParams?[{binding:9,resource:{buffer:this.sourceParams,size:UNIFORM_PARAMS_BYTES}}]:[]),
      {binding:11,resource:f.unitVelocity},{binding:14,resource:f.unitHalo},
      {binding:12,resource:{buffer:this.claims}},
      {binding:13,resource:{buffer:f.narrowBandState}},
    ]});
    return group;
  }
  async initialize(): Promise<void> {
    // regularTexture: interior unit-stencil samples load their eight taps
    // directly (no per-tap boundary branch, transverse clamp or select); the
    // skipped taps are exactly the zero-weight ones.
    const velocitySampling=uniformMixedVelocitySamplingSource(false,"coarseVelocity","velocity",this.solid?3:2,"unitVelocity",undefined,"unitHalo");
    const module=uniformDetailModule(this.device,{label:"Uniform mixed surface",code:uniformMixedClaimedEntriesWGSL(uniformMixedTopologyWGSL(this.ownership.capacity,0)+uniformMixedTopologyNeighboursWGSL()+/* wgsl */`
@group(1) @binding(0) var phi:texture_3d<f32>;
@group(1) @binding(1) var outputPhi:texture_storage_3d<r32float,write>;
@group(1) @binding(2) var velocity:texture_3d<f32>;
@group(1) @binding(3) var coarseVelocity:texture_3d<f32>;
@group(1) @binding(5) var<storage,read> negative:array<f32>;
struct Params {hDt:vec4f,flags:vec4u}
@group(1) @binding(6) var<uniform> params:Params;
${this.solid?/* wgsl */`// A closed word a tile (solidClosed), then a clear word a tile (solidClear).
@group(1) @binding(8) var<storage,read_write> evidence:array<u32>;`:""}
// Advect vertices a released wall or solid can change: count, grid, list.
@group(1) @binding(10) var<storage,read_write> deferred:UMDeferred;
struct UMDeferred { header:array<atomic<u32>,4>, data:array<u32> }
@group(1) @binding(12) var<storage,read_write> umClaims:array<atomic<u32>>;
@group(1) @binding(13) var<storage,read_write> nbState:array<atomic<u32>>;
${this.sourceParams?uniformMixedSourceWGSL(9):""}
${uniformMixedFaceAddressWGSL}
fn umLoadVertex(p:vec3u)->f32{return textureLoad(phi,vec3i(p),0).x;}
// A tile corner: the base block (UNIFORM_DETAIL_4H_LOAD).
fn umLoadCorner(p:vec3u)->f32{return ${UNIFORM_DETAIL_4H_LOAD}textureLoad(phi,vec3i(p),0).x;}
// The same loads of the eight vertices p + s*umCorner(k,2u): one page-table read where one block holds them.
fn umLoadVertex8(p:vec3u,s:u32)->array<f32,8>{return udLoad8_phi(vec3i(p),s);}
fn umLoadCorner8(p:vec3u,s:u32)->array<f32,8>{return udLoad48_phi(vec3i(p),s);}
// A wall owner's face texel (speed and release bits): a 4h owner's anchor is canonical.
fn umLoadWallTexel(anchor:vec3i,width:u32)->vec4f{if(width==4u){return ${UNIFORM_DETAIL_RING_4H_LOAD}textureLoad(velocity,anchor,0);}return textureLoad(velocity,anchor,0);}
${uniformMixedVertexSamplingSource("",true,undefined,"umLoadCorner",true)}
fn umLoadMixedFace(anchor:vec3i,axis:u32)->f32{
 if(umFaceLow(anchor,axis)){return negative[umLowFaceIndex(umLowFaceAbove(anchor,axis),axis)];}
 return textureLoad(velocity,anchor,0)[axis];
}
@group(1) @binding(11) var unitVelocity:texture_3d<f32>;
@group(1) @binding(14) var unitHalo:texture_3d<f32>;
${velocitySampling}
${narrowBandTraceWGSL("umSampleVelocity")}
// p as a tap reaches it from the page: held inside it on an axis only where
// a SOLID page lies across that face (umPointNearFromPage). An AIR page
// never holds. p lies in the page or one of the 26 around it.
fn umPointFromPage(page:vec3i,p:vec3f)->vec3f{
 let low=vec3f(page<<vec3u(5u));if(all(p>=low)&&all(p<=low+vec3f(32.0))){return p;}
 return umPointNearFromPage(umBlockOf(page),page,p);
}
fn umPointFrom(home:vec3f,p:vec3f)->vec3f{return umPointFromPage(umPointCell(home)>>vec3u(5u),p);}
// advectFine and advectDeferred's fast twin are compiled with umFastSurface:
// their samples take umSampleVelocity's weights and its two interpolants in
// their cached forms only, the unit one at plain offsets in the sample
// page's halo block and the 4h one from the cache (the same terms, order
// and sums: the same value). The general sampler's resolution paths are
// then not in those pipelines. A sample with fine weight and no halo home
// (an unstored block) ends the trace (umSlow), and the vertex is left for
// the general launch, which traces it from the start.
override umFastSurface:bool=false;
// Publish exact input changes while the producer already owns the value.
// Every incident 16h closure includes a shared boundary vertex. Consumers
// can retain whole outputs by revision without scanning fine phi again.
fn umStoreSurface(vertex:vec3u,width:u32,value:f32){
 ${this.workControl?`if(bitcast<u32>(value)!=bitcast<u32>(umLatticeLoad(vec3i(vertex),width))){
  let boundary=(vertex&vec3u(15u))==vec3u(0u);
  for(var k=0u;k<8u;k++){
   let bit=umCorner(k,2u);if(any((bit!=vec3u(0u))&!boundary)){continue;}
   let tile=umTileKey((vec3i(vertex)-vec3i(bit))>>vec3u(2u));if(!umKeyStored(tile)){continue;}
   let at=${this.revisionWord}u+umPageOfTile(tile);let epoch=atomicLoad(&umClaims[at]);
   atomicMax(&umClaims[at],min(epoch,0xfffffffdu)+1u);
  }
 }`:""}
 textureStore(outputPhi,vec3i(vertex),vec4f(value));
}
override umExactLocalSolidTransport:bool=true;
var<private> umLocalSolidAdmission:bool=false;
var<private> umSlow:bool=false;
fn umFastVelocity(p:vec3f)->vec3f{
 let s=umSampleTile(p);
 var f=1.0;if(!umCachedFineTile(s.key)){f=umVelocitySamplingWeightsAt(s);}
 var home=vec4i(0);if(f>0.0){home=umUnitHaloHome(s);if(home.w<0){umSlow=true;return vec3f(0.0);}}
 let coarseHome=umCoarseHome(p,s);let coarse=max(0.0,1.0-f);var v=vec3f(0.0);
 for(var a=0u;a<3u;a++){
  var value=0.0;
  if(f>0.0){
   var offset=vec3f(0.5);offset[a]=1.0;
   let b=umHold;var lower=b.lo;lower[a]-=1.0;
   let q=clamp(p-offset,lower,b.hi-vec3f(1.0));
   let base=vec3i(floor(q));var terms:array<f32,8>;
   for(var k=0u;k<8u;k++){terms[k]=textureLoad(unitHalo,base+vec3i(i32(k&1u),i32((k>>1u)&1u),i32(k>>2u))+home.xyz,0)[a];}
   value+=f*umVelocityLerp8(terms,fract(q));
  }
  if(coarse>0.0){value+=coarse*umSampleVelocity4In(coarseHome,a);}
  v[a]=value;
 }
 return v;
}
// nbTraceSpan's steps, every stage's sample at one call site: stage 0 the
// field at q (k1; the first sets the substeps), 1 one base substep ahead
// (k2), 2 the third point, after which the step ends.
override umSurfaceStationaryTrace:bool=true;
// Fast candidates and full fallback read the same frozen velocity field.
// Only a successful exact fast sample may seed the next trace's first stage.
override umSurfaceTraceSeed:bool=true;
var<private> umTraceSeedValid:bool=false;
var<private> umTraceSeedPoint:vec3f;
var<private> umTraceSeedVelocity:vec3f;
fn umSurfaceTrace(p:vec3f)->vec3f{
 let h=params.hDt.xyz;
 let useSeed=umSurfaceTraceSeed&&umFastSurface&&umTraceSeedValid&&all(umTraceSeedPoint==p);
 let seededVelocity=umTraceSeedVelocity;umTraceSeedValid=false;
 // Sampling initializes this private wall context even when RK terminates
 // on its stationary first stage. Preserve that sampler side effect.
 if(useSeed){umHold=umPointBounds(p,8.0);}
 var q=p;var x=p;var steps=1u;var i=0u;var base=0.0;var stage=0u;var k1=vec3f(0.0);var k2=vec3f(0.0);var two=false;
 for(var n=0u;n<${3*NARROW_BAND_TRACE_LIMIT+3}u;n++){
  var k=vec3f(0.0);
  if(useSeed&&n==0u){k=seededVelocity/h;}
  else if(umFastSurface){k=umFastVelocity(x)/h;if(umSlow){return p;}}else{k=umSampleVelocity(x)/h;}
  if(stage==0u){
   // Exactly stationary in the frozen field: every later RK stage is at q
   // and samples the same zero velocity. No tolerance or spatial mask.
   if(umSurfaceStationaryTrace&&all(k==vec3f(0.0))){
    let held=umPointFrom(q,q);if(all(held==q)){return held;}
   }
   k1=k;
   if(i==0u){
    steps=nbTraceSteps(k1,params.hDt.w);
    if(steps>${NARROW_BAND_TRACE_LIMIT}u){atomicStore(&nbState[3],1u);return p;}
    base=-params.hDt.w/f32(steps);
   }
   x=q+base*k1;stage=1u;continue;
  }
  if(stage==1u){k2=k;two=nbTraceTwo(base,k1,k2,steps-i);x=nbTraceThird(q,base,k1,k2,two);stage=2u;continue;}
  let span=nbTraceEnd(q,base,k1,k2,k,two);
  i+=u32(span.w);q=umPointFrom(q,span.xyz);
  if(i>=steps){break;}
  x=q;stage=0u;
 }
 return q;
}
${uniformMixedSolidWGSL(this.solid?2:undefined,this.solid?.coarse?.count)}
${this.solid?/* wgsl */`${uniformMixedVertexBuriedWGSL}
// Every caller needs only the open predicate; the coarse record caches it.
fn umOpenAt(q:vec3f)->bool{return umCellIsOpen(umPointCell(q));}
// uvTrace: walk every crossed half-cell so no characteristic tunnels a wall.
struct UMWalk{q:vec3f,hit:bool}
fn umWalk(p:vec3f,end:vec3f)->UMWalk{
 if(!umSolidEnabled()){return UMWalk(end,false);}
 let steps=max(1u,u32(ceil(2.0*max(abs(end.x-p.x),max(abs(end.y-p.y),abs(end.z-p.z))))));
 var previous=p;
 for(var s=1u;s<=steps;s++){let q=mix(p,end,f32(s)/f32(steps));if(!umOpenAt(q)){return UMWalk(previous,true);}previous=q;}
 return UMWalk(end,false);
}
fn umTrace(p:vec3f)->vec3f{return umWalk(p,umSurfaceTrace(p)).q;}
// solidClosed/solidClear: no in-domain closed cell within four cells of the
// tile holding vertex p's cell (umPointCell; 27 tiles), a superset of the
// cells umBuried reads at a vertex or cubic tap of any vertex of that tile.
fn umSolidClear(p:vec3i)->bool{
 if(!umSolidEnabled()){return true;}
 let tile=umTileAt(vec3u(umPointCell(vec3f(p)))/4u);
 return ${this.solid.coarse?"umSolidSurfaceClear(tile)":"evidence[UM_TILES+tile]!=0u"};
}
// uvBuried: phi at a vertex no incident owner has capacity at is not state.
fn umBuried(p:vec3f)->bool{
 let vertex=vec3i(p);
 ${this.solid.coarse?/* wgsl */`let local=vec3u(vertex)&vec3u(3u);
 // Interior vertices' eight incident cells share one tile. Read its open
 // mask once instead of resolving the same page, tile and record eight times.
 if(umSolidEnabled()&&all(local!=vec3u(0u))){
  let base=vertex-vec3i(1);let tile=umTileKey(base>>vec3u(2u));
  if(umKeyStored(tile)){
   if(!umSolidValid(base)){return true;}
   if(umTileWidth(tile)==4u){return !(umTileOpen(tile)>1e-5);}
   let cell=vec3u(base)&vec3u(3u);let shift=cell.x+4u*(cell.y+4u*cell.z);
   // A 2^3 cell stencil: offsets 0,1,4,5,16,17,20,21 in the 64-cell mask.
   var stencil=vec2u(0u);
   if(shift<32u){stencil.x=0x00330033u<<shift;if(shift!=0u){stencil.y=0x00330033u>>(32u-shift);}}
   else{stencil.y=0x00330033u<<(shift-32u);}
   return all((umSolidOpenCells(tile)&stencil)==vec2u(0u));
  }
 }
`:""}
 return umVertexBuried(vertex);
}
fn umFluidFace(face:vec3i,axis:u32)->f32{
 if(umSolidValid(face)){return textureLoad(velocity,face,0)[axis];}
 var n=face;n[axis]+=1;
 if(umSolidValid(n)&&umFaceLow(face,axis)){return negative[umLowFaceIndex(vec3u(n),axis)];}
 return 0.0;
}
fn umContactReleased(face:vec3i,axis:u32)->bool{
 var cell=face;var bit=axis;if(umFaceLow(face,axis)){cell[axis]+=1;bit+=3u;}
 if(!umSolidValid(cell)){return false;}
 return (u32(round(textureLoad(velocity,cell,0).w))&(1u<<bit))!=0u;
}
// uvEmbeddedAir: a released embedded wall supplies air at its first hit.
// end is umSurfaceTrace(p). Its walk tests the same half-cells as umWalk and
// a subset of their closed tests, so a walk without a hit leaves advected.
fn umEmbeddedAir(p:vec3f,end:vec3f,advected:f32)->f32{
 let h=params.hDt.xyz;let dt=params.hDt.w;
 let steps=max(1u,u32(ceil(2.0*max(abs(end.x-p.x),max(abs(end.y-p.y),abs(end.z-p.z))))));
 var previous=p;var result=advected;
 for(var step=1u;step<=steps;step++){
  let q=mix(p,end,f32(step)/f32(steps));let solid=vec3i(floor(q));
  if(umSolidValid(solid)&&!umCellIsOpen(solid)){
   for(var axis=0u;axis<3u;axis++){for(var side=-1;side<=1;side+=2){
    var fluid=solid;fluid[axis]+=side;if(!umCellIsOpen(fluid)){continue;}
    let inward=f32(side);let plane=f32(solid[axis])+select(0.0,1.0,side>0);
    let distance=inward*(p[axis]-plane);if(distance< -1e-5||inward*(end[axis]-p[axis])>=0.0){continue;}
    let a=inward*(previous[axis]-plane);let b=inward*(q[axis]-plane);
    if(a< -1e-5||b>1e-5){continue;}
    let face=select(fluid,solid,side>0);let away=inward*umFluidFace(face,axis);
    if(umContactReleased(face,axis)&&dt*away>1e-4*h[axis]){result=max(result,dt*away-distance*h[axis]);}
   }}
   return result;
  }
  previous=q;
 }
 return result;
}`:/* wgsl */`fn umTrace(p:vec3f)->vec3f{return umSurfaceTrace(p);}`}
fn umCatmull(t:f32)->vec4f{
 let t2=t*t;let t3=t2*t;return 0.5*vec4f(2.0*t2-t3-t,3.0*t3-5.0*t2+2.0,4.0*t2-3.0*t3+t,t3-t2);
}
// Catmull-Rom weights w (taps -1..2) with a held end tap read as the line
// through the two inner taps: s(-1)=2s(0)-s(1), s(2)=2s(1)-s(0).
fn umHeldWeights(w:vec4f,low:bool,high:bool)->vec4f{
 let a=select(0.0,w.x,low);let b=select(0.0,w.w,high);
 return vec4f(w.x-a,w.y+2.0*a-b,w.z-a+2.0*b,w.w-b);
}
// umSampleVertex(p) on the resolved field, keeping what umCubicPhi reuses:
// the owner cell's origin and width (its tile's width), and its eight corner
// values in umCorner order (the same loads, weights and D4 sum).
struct UMVertexCell {origin:vec3u,width:u32,corners:array<f32,8>,value:f32}
fn umVertexCell(p:vec3f)->UMVertexCell{
 var c:UMVertexCell;var t=vec3f(0);let cell=vec3u(umPointCell(p));
 if(umRegularFine){c.origin=cell;c.width=1u;t=p-vec3f(c.origin);}
 else{c.width=umTileWidth(umTileAt(cell/4u));c.origin=(cell/c.width)*c.width;t=(p-vec3f(c.origin))/f32(c.width);}
 var values:array<f32,8>;
 // One branch on the cell's width, outside its eight taps: a 4h cell's
 // corners are tile corners, read from the base block. A branch inside the
 // tap (one per load) doubled the merged pass on Metal.
 if(!umRegularFine&&c.width==4u){c.corners=umLoadCorner8(c.origin,4u);}else{c.corners=umLoadVertex8(c.origin,c.width);}
 for(var k=0u;k<8u;k++){
  let corner=umCorner(k,2u);let w=select(vec3f(1)-t,t,corner!=vec3u(0));
  values[k]=c.corners[k]*w.x*w.y*w.z;
 }
 c.value=umVertexSum8(values);return c;
}
// The cubic at q over cell = umVertexCell(q): its owner (origin, width) and
// the eight inner taps (offsets 0..1) come from the cell, not reloaded.
fn umCubicPhi(q:vec3f,cell:UMVertexCell)->f32{
 let width=i32(cell.width);let base=vec3i(cell.origin);let fraction=(q-vec3f(base))/f32(width);
 // Resolved: every tap is a width-aligned vertex within one tile of the
 // owner's tile (4h taps are 4-aligned, stored everywhere), so it is stored
 // or resolved. The 64 taps then unroll with a constant bound.
 // A tap is held in the owner's page where a SOLID page lies across a face
 // (umVertexNearFromPage), so every tap has an incident cell no wall parts
 // from the owner. With no embedded solids it cannot be buried, at any owner
 // width. Fold that case
 // before the 64 taps: the fine-tile clearance certificate alone excludes
 // coarse owners and leaves their per-tap domain tests in the solid-free twin.
 ${this.solid?`let tapsClear=!umSolidEnabled()||((umRegularFine||cell.width==1u)&&umSolidClear(base))${this.workControl&&this.solid?.coarse?"||(umExactSolidSupport&&umCanonicalSupportAt(vec3f(cell.origin),2.0*f32(cell.width)+4.0))":""};`:""}
 var value=0.0;var low=1e30;var high=-1e30;let page=base>>vec3u(5u);let home=umBlockOf(page);
 // A tap held at a SOLID face repeats the inner tap beside it, and the
 // cubic read that as a level set flat across the face: between the face
 // and the next vertex it dipped up to 2/27 width below the line. The held
 // tap stands for the line through the two inner taps on its axis: its
 // weight moves onto them. An axis has at most one end tap past the page.
 let at=base&vec3i(31);let past=vec3i(at+vec3i(2*width)>vec3i(32))-vec3i(at<vec3i(width));
 let held=umBlockHeld(home,page,past);let lowHeld=held&(past<vec3i(0));let highHeld=held&(past>vec3i(0));
 let wx=umHeldWeights(umCatmull(fraction.x),lowHeld.x,highHeld.x);let wy=umHeldWeights(umCatmull(fraction.y),lowHeld.y,highHeld.y);let wz=umHeldWeights(umCatmull(fraction.z),lowHeld.z,highHeld.z);
 // A weighted buried tap returns the cell's value, whichever tap it is:
 // test them ahead of the taps at one umBuried site (inlined per site, a
 // test inside the unrolled taps compiled 64 of them per width).
 ${this.solid?`for(var k=0u;k<select(64u,0u,tapsClear);k++){
  let x=k&3u;let y=(k>>2u)&3u;let z=k>>4u;if(wx[x]*wy[y]*wz[z]==0.0){continue;}
  if(umBuried(vec3f(umVertexNearFromPage(home,page,base+(vec3i(vec3u(x,y,z))-vec3i(1))*width)))){return cell.value;}
 }`:""}
 // As umVertexCell: the width branch is outside the 64 taps.
 ${(taps=>`if(!umRegularFine&&cell.width==4u){${taps("umLoadCorner")}}else{${taps("umLoadVertex")}}`)((load:string)=>/* wgsl */`
 for(var z=0u;z<4u;z++){var plane=0.0;
  for(var y=0u;y<4u;y++){var row=0.0;
   for(var x=0u;x<4u;x++){
    let offset=vec3i(vec3u(x,y,z))-vec3i(1);let vertex=umVertexNearFromPage(home,page,base+offset*width);
    // Cubic taps lie on lattice vertices, stored or resolved: direct loads.
    // The inner taps are the cell's corners.
    let inner=all(offset>=vec3i(0))&&all(offset<=vec3i(1));
    var s=0.0;if(inner){s=cell.corners[x-1u+2u*(y-1u)+4u*(z-1u)];}else{s=${load}(vec3u(vertex));}row+=wx[x]*s;
    if(inner){low=min(low,s);high=max(high,s);}
   }plane+=wy[y]*row;
  }value+=wz[z]*plane;
 }`)}
 return clamp(value,low,high);
}
${uniformMixedWallVertexWGSL}
${uniformMixedWallAheadWGSL}
// A vertex on a wall (a face of a stored page with a SOLID page across it;
// an AIR page across is none) whose liquid one width inside moves onto the
// wall takes the phi carried to that inside point, where it is liquid.
fn umWallContact(p:vec3f,value:f32,width:u32)->f32{
 if(params.hDt.w<=0.0){return value;}var interior=p;var contact=false;
 for(var k=0u;k<6u;k++){
  if(umWallBlock(vec3i(p),k)==0u){continue;}
  let axis=k>>1u;let inward=select(1.0,-1.0,(k&1u)!=0u);
  var probe=p;probe[axis]+=inward*f32(width);
  var speed=0.0;if(umFastSurface){speed=umFastVelocity(probe)[axis];}else{speed=umSampleVelocity(probe)[axis];}
  if(inward*speed>=-1e-6){continue;}
  interior[axis]+=inward*f32(width);contact=true;
 }
 if(!contact${this.solid?"||!umOpenAt(interior)":""}){return value;}let continued=umSampleVertex(umTrace(interior));
 return select(value,min(value,continued),continued<0.0);
}
// The six wall reach words (UM_WALL_REACH, a face direction each) are final
// once wallReach, an earlier dispatch, is done. Every advect entry stages
// them per job, so its vertex lanes read workgroup memory instead of six
// atomics per vertex.
var<workgroup> umPlaneReach:array<f32,6>;
fn umStagePlaneReach(lane:u32){
 if(lane<6u){umPlaneReach[lane]=bitcast<f32>(atomicLoad(&umClaims[UM_WALL_REACH+lane]));}
 workgroupBarrier();
}
fn umReleasedWalls(p:vec3f,value:f32)->f32{
 var result=value;let h=params.hDt.xyz;let dt=params.hDt.w;let at=vec3i(p);
 for(var k=0u;k<6u;k++){
  let axis=k>>1u;let upper=(k&1u)!=0u;let inward=select(1.0,-1.0,upper);
  // wallReach stored this direction's largest contributing dt*away, zero
  // when no wall face of it is released. umWallAhead finds the wall that can
  // reach p, and none where even that reach cannot raise phi here: then none
  // of the four probes can. The rounding margin keeps this a work exclusion,
  // never a modification of the wall continuation.
  let reach=umPlaneReach[k];
  if(reach<=0.0){continue;}
  let wall=umWallAhead(at,k,reach,result,h);
  let depth=inward*(p[axis]-f32(wall.plane))*h[axis];
  var whole=1e30;var probes=0u;
  for(var c=0u;c<4u;c++){
   // The probe cells: the wall's cells around p's foot on it. A column with
   // no wall at the plane has no owner on it.
   if(wall.blocks[c]==0u){continue;}
   var cell=at;cell[axis]=wall.plane-select(0,1,upper);cell[(axis+1u)%3u]-=i32(1u-(c&1u));cell[(axis+2u)%3u]-=i32(1u-(c>>1u));
   let owner=umOwnerAt(cell);
   let origin=umOrigin(owner);var anchor=vec3i(origin);anchor[axis]+=i32(owner.width)-1;
   let wallTexel=umLoadWallTexel(anchor,owner.width);var speed=wallTexel[axis];var bit=axis;
   if(!upper){speed=negative[umLowFaceIndex(origin,axis)];bit+=3u;}
   let released=(u32(round(wallTexel.w))&(1u<<bit))!=0u;
   let away=inward*speed;
   // A wide owner's release is its centre pressure, and liquid it cannot
   // resolve above it (V under an air centre) lowers that pressure. One
   // such owner separates the wall only within half its width of the
   // surface; deeper, its lift alone would snap a submerged wall column to
   // air.
   let resolved=owner.width==1u||value>-0.5*f32(owner.width)*min(h.x,min(h.y,h.z));
   let gap=select(0.0,dt*away,released&&dt*away>1e-4*h[axis]*f32(owner.width));
   if(resolved&&gap>0.0){result=max(result,gap-depth);}
   whole=min(whole,gap);probes++;
  }
  // Deeper, the wall separates where every owner around the vertex has
  // released, by the least of their gaps. A slab parallel to the wall and
  // thicker than half a wide owner has no vertex the rule above reaches:
  // the pressure released its owners and drained their V while phi stayed
  // on the wall, and each such owner (a pressure row with a half-open wall
  // face and no V) doubled its wall face speed against its opposite face
  // every step (the all-4h Figure 9 lid: 30 m/s wall faces at -70 kPa).
  if(probes>0u&&whole>0.0){result=max(result,whole-depth);}
 }return result;
}
${this.solid?`// Every cell at vertex q is open (an h vertex), or its tile is clear of solids.
fn umSmoothOpen(q:vec3i,w:i32)->bool{
 if(umSolidClear(q)){return true;}
 if(w!=1){return false;}
 for(var k=0u;k<8u;k++){let cell=q-vec3i(1)+vec3i(umCorner(k,2u));if(!umSolidValid(cell)||!umCellIsOpen(cell)){return false;}}
 return true;
}`:""}
// A width-aligned lattice vertex, loaded directly: an h vertex from the h
// store (an h vertex's neighbours are stored or, in a seam tile, resolved),
// a tile corner from the base block (stored everywhere).
fn umLatticeLoad(q:vec3i,width:u32)->f32{if(width==4u){return umLoadCorner(vec3u(q));}return umLoadVertex(vec3u(q));}
// A traced vertex's foot and smoothing, from one stencil of its own lattice
// (umLatticeLoad) read before the trace: the full sampler's code in the
// advect kernels cost more than its taps, and only the foot, its shift and
// the smoothing stay live across the trace. foot: where umSurfaceTrace
// starts; shift: p - foot, applied at the trace's end (zero unfooted);
// smoothing: added to the advected value.
//
// The departure of a level-set vertex: a vertex within two of its own cells
// of the surface is carried by the surface, displaced as its foot on the zero
// set is. The vertices beside the surface keep their advected values, so
// nothing restores a distance there: carried by the velocity at their own
// depth, any steady shear under a still surface moved their values and not
// the zero between them, and the pressure rows, which read the surface off
// those values, saw a rise as a dip and filled it
// (uniform-trough-settled-tank: the wet vertex 0.93 h under the surface
// 2e-2 h off its distance in 240 steps). The foot's gradient is the central
// difference a width either side (taps 0..5: -x +x -y +y -z +z), held at a
// wall (the tap is p) and, at a 4h vertex, at a closed quarter-width point.
// Off a seam that is the trilinear difference a quarter width either side,
// in exact arithmetic; a 4h vertex at a seam takes its 4h neighbours.
//
// Phi is vertex state and both its couplings are means: a centre's phi is the
// mean of its eight vertices, and a vertex moves with the faces around its
// foot. A vertex pattern alternating along x or z has no pressure and no
// advection, and nothing removed it: the outline of a disturbance stood in a
// 4h surface for good, and stripes grew in an h one. The fourth difference
// along the two horizontal axes, at the vertex's own width, damps that
// pattern and leaves a cubic alone: a step takes 16 eps of an alternating
// pattern and eps (2 pi/n)^4 of a wave n vertices long. An axis is skipped
// where a wall lies before its far tap, where the vertex or a tap is not
// open liquid state, and where a tap is not a distance from the vertex. Its
// near taps are the gradient's x and z taps (with no wall within two widths
// none is held); taps 6..9 are the far ones (-2x +2x -2z +2z).
${uniformSurfaceSmoothingWGSL}
${uniformSurfaceTransportWGSL}
${uniformSurfaceFarCellWGSL}
struct UMFoot {foot:vec3f,shift:vec3f,smoothing:f32,center:f32}
override umExactSolidSupport:bool=true;
fn umVertexFoot(vertex:vec3u,width:u32)->UMFoot{
 let p=vec3f(vertex);let w=f32(width);let wi=i32(width);let h=min(params.hDt.x,min(params.hDt.y,params.hDt.z));
 var result=UMFoot(p,vec3f(0.0),0.0,0.0);
 let c=umLatticeLoad(vec3i(vertex),width);result.center=c;if(!(abs(c)<2.0*w*h)){return result;}
 ${this.workControl&&this.solid?.coarse?"let solidFootClear=!umSolidEnabled()||(umExactSolidSupport&&umCanonicalSupportAt(p,2.0*w+7.0));":"let solidFootClear=false;"}
 let page=umPointCell(p)>>vec3u(5u);
 var at:array<vec3f,10>;
 for(var axis=0u;axis<3u;axis++){
  var delta=vec3f(0);delta[axis]=w;
  at[2u*axis]=umPointFromPage(page,p-delta);at[2u*axis+1u]=umPointFromPage(page,p+delta);
  ${this.solid?/* wgsl */`if(width>1u&&!solidFootClear){
   var quarter=vec3f(0);quarter[axis]=0.25*w;
   if(umSampleClosed(umPointFromPage(page,p-quarter))){at[2u*axis]=p;}
   if(umSampleClosed(umPointFromPage(page,p+quarter))){at[2u*axis+1u]=p;}
  }`:""}
 }
 for(var k=0u;k<4u;k++){var q=p;q[select(0u,2u,k>1u)]+=select(-2.0,2.0,(k&1u)!=0u)*w;at[6u+k]=q;}
 // The width branch is outside the taps.
 var t:array<f32,10>;
 if(width==4u){for(var k=0u;k<10u;k++){t[k]=umLoadCorner(vec3u(at[k]));}}
 else{for(var k=0u;k<10u;k++){t[k]=umLoadVertex(vec3u(at[k]));}}
 var g=vec3f(0);
 for(var axis=0u;axis<3u;axis++){g[axis]=(t[2u*axis+1u]-t[2u*axis])/max(at[2u*axis+1u][axis]-at[2u*axis][axis],1e-6);}
 let g2=dot(g,g);
 if(g2>=1e-6*h*h){
  var offset=-c*g/g2;let reach=length(offset);if(reach>2.0*w){offset*=2.0*w/reach;}
  let foot=umPointFrom(p,p+offset);
  if(${this.solid?"solidFootClear||umOpenAt(foot)":"true"}){result.foot=foot;result.shift=p-foot;}
 }
 let eps=min(${UNIFORM_MIXED_VERTEX_SMOOTHING_PER_S.toExponential(8)}*params.hDt.w,0.03125);if(!(eps>0.0)){return result;}
 // Closed bits: 0 the vertex, 1+2*axis+k tap k of axis (x, z: near low,
 // near high, far low, far high). The nine tests share one call site.
 var closed=0u;
 ${this.solid?/* wgsl */`if(umSolidEnabled()&&!solidFootClear){
  for(var j=0u;j<9u;j++){
   var q=vec3i(vertex);if(j>0u){let k=(j-1u)&3u;q[select(0u,2u,j>4u)]+=select(wi,2*wi,(k&2u)!=0u)*select(-1,1,(k&1u)!=0u);}
   if(!umSmoothOpen(q,wi)){closed|=1u<<j;}
  }
  if((closed&1u)!=0u){return result;}
 }`:""}
 result.smoothing=umSurfaceSmoothing(vec3i(vertex),wi,c,t,closed,params.hDt.xyz,eps);
 return result;
}
// The departure of a vertex from its foot: the foot's trace, displaced back
// by its shift (within two widths of where the trace ended, reached from
// there). One trace (one call site of umSurfaceTrace).
fn umVertexTrace(foot:UMFoot)->vec3f{
 let end=umSurfaceTrace(foot.foot);if(all(foot.shift==vec3f(0.0))){return end;}
 return umPointFrom(end,end+foot.shift);
}
var<private> umFullCubic:bool=false;
var<private> umFullIdentity:bool=false;
fn umAdvected(p:vec3f,width:u32,foot:UMFoot)->f32{
 ${this.solid?/* wgsl */`let clear=umSolidClear(vec3i(p));if(!clear&&umBuried(p)){umFullIdentity=true;return umLoadVertex(vec3u(p));}
 let end=umVertexTrace(foot);let walk=umWalk(p,end);let q=walk.q;`:"let q=umVertexTrace(foot);"}let cell=umVertexCell(q);var value=cell.value;
 let h=f32(width)*min(params.hDt.x,min(params.hDt.y,params.hDt.z));
 if(params.flags.y!=0u&&abs(value)<2.0*h&&(params.flags.y==1u||width>1u)){umFullCubic=true;value=umCubicPhi(q,cell);}
 ${this.solid?"value=umWallContact(p,value,width);if(walk.hit){value=umEmbeddedAir(p,end,value);}value=umReleasedWalls(p,value);":"value=umReleasedWalls(p,umWallContact(p,value,width));"}
 return ${this.sourceParams?"umSourceuvSourcePhi(p,value)":"value"};
}
${this.workControl?/* wgsl */`
fn umRecordRegion(vertex:vec3u,tier:u32){
 if(params.flags.z==0u){return;}let slot=umVertexSlot(vec3i(vertex));let local=min(slot.xyz,vec3u(31u))>>vec3u(4u);
 let region=8u*slot.w+local.x+2u*local.y+4u*local.z;
 atomicAdd(&umClaims[${this.regionCounterWord}u+5u*region+tier],1u);
}
// The controller amortizes each of the64 rectangular support proofs once
// per16h region. Axis codes select home,low+home,home+high,or all three.
fn umSupportIndex(region:u32,p:vec3f,reach:f32)->u32{
 let origin=vec3f(16u*umPageCoord(region));
 let low=vec3i(floor((p-vec3f(reach)-origin)/16.0));let high=vec3i(floor((p+vec3f(reach)-origin)/16.0));
 let code=vec3u(low<vec3i(0))+2u*vec3u(high>vec3i(0));return code.x+4u*code.y+16u*code.z;
}
fn umSupportMaskIndex(required:u32)->u32{
 let x=u32((required&0x1249249u)!=0u)+2u*u32((required&0x4924924u)!=0u);
 let y=u32((required&0x01c0e07u)!=0u)+2u*u32((required&0x70381c0u)!=0u);
 let z=u32((required&0x00001ffu)!=0u)+2u*u32((required&0x7fc0000u)!=0u);
 return x+4u*y+16u*z;
}
fn umRequiredRho(region:u32,g:vec3f,required:u32,broad:f32)->f32{
 let at=${this.supportWord}u+${UNIFORM_SURFACE_SUPPORT_WORDS}u*(64u*region+umSupportMaskIndex(required));
 if(atomicLoad(&umClaims[at+7u])==0u){return broad;}
 return min(broad,bitcast<f32>(atomicLoad(&umClaims[at])));
}
struct UMRegionalVelocity {lo:vec3f,hi:vec3f,valid:bool}
fn umVelocityCovered(region:u32,p:vec3f,reach:f32,coverage:u32)->bool{
 let at=${this.supportWord}u+${UNIFORM_SURFACE_SUPPORT_WORDS}u*(64u*region+umSupportIndex(region,p,reach));return atomicLoad(&umClaims[at+7u])!=0u;
}
fn umHaloVelocity(region:u32)->UMRegionalVelocity{
 let at=${this.velocityWord}u+6u*region;
 return UMRegionalVelocity(bitcast<vec3f>(vec3u(atomicLoad(&umClaims[at]),atomicLoad(&umClaims[at+1u]),atomicLoad(&umClaims[at+2u]))),bitcast<vec3f>(vec3u(atomicLoad(&umClaims[at+3u]),atomicLoad(&umClaims[at+4u]),atomicLoad(&umClaims[at+5u]))),true);
}
fn umRequiredVelocity(region:u32,p:vec3f,reach:f32,coverage:u32)->UMRegionalVelocity{
 let at=${this.supportWord}u+${UNIFORM_SURFACE_SUPPORT_WORDS}u*(64u*region+umSupportIndex(region,p,reach));
 return UMRegionalVelocity(bitcast<vec3f>(vec3u(atomicLoad(&umClaims[at+1u]),atomicLoad(&umClaims[at+2u]),atomicLoad(&umClaims[at+3u]))),bitcast<vec3f>(vec3u(atomicLoad(&umClaims[at+4u]),atomicLoad(&umClaims[at+5u]),atomicLoad(&umClaims[at+6u]))),atomicLoad(&umClaims[at+7u])!=0u);
}
// Canonical openness is sufficient only for predicate-based surface rules.
// Missing/out-of-window support stays on the original exact evaluator.
fn umCanonicalSupportAt(p:vec3f,reach:f32)->bool{
 let vertex=vec3i(p);let slot=umVertexSlot(vertex);let local=min(slot.xyz,vec3u(31u))>>vec3u(4u);
 if(!umBlockStored(slot.w)){return false;}
 let region=8u*slot.w+local.x+2u*local.y+4u*local.z;
 let origin=vec3f(16u*umPageCoord(region));
 let low=vec3i(floor((p-vec3f(reach)-origin)/16.0));
 let high=vec3i(floor((p+vec3f(reach)-origin)/16.0));
 if(any(low<vec3i(-1))||any(high>vec3i(1))){return false;}
 return umSupportSolidClear(region,p,reach);
}
fn umSupportSolidClear(region:u32,p:vec3f,reach:f32)->bool{
 let at=${this.supportWord}u+${UNIFORM_SURFACE_SUPPORT_WORDS}u*(64u*region+umSupportIndex(region,p,reach));
 return (atomicLoad(&umClaims[at+7u])&2u)!=0u;
}
fn umRequiredLipschitz(region:u32,required:u32)->vec3f{
 let at=${this.supportWord}u+${UNIFORM_SURFACE_SUPPORT_WORDS}u*(64u*region+umSupportMaskIndex(required));
 if(atomicLoad(&umClaims[at+7u])==0u){return vec3f(3.402823e38);}
 return bitcast<vec3f>(vec3u(atomicLoad(&umClaims[at+8u]),atomicLoad(&umClaims[at+9u]),atomicLoad(&umClaims[at+10u])));
}
fn umTransportRejected(reason:u32)->vec2f{
 if(params.flags.z!=0u){atomicAdd(&umClaims[${this.counterWord+3}u],1u);atomicAdd(&umClaims[${this.counterWord+5}u+reason],1u);}
 return vec2f(0.0);
}
// Current-state Euler transport on a regional affine model. Full RK's
// foot shift cancels: departure stays within .5 cell of this vertex, while
// its velocity samples stay within 2 widths+.5. The model's 27-region halo
// includes those samples and the cubic stencil for h/4h owners.
fn umCheapAdvected(vertex:vec3u,width:u32)->vec2f{
 if(width!=1u&&width!=4u){return umTransportRejected(1u);}
 ${this.solid&&!this.solid.coarse?"if(umSolidFree&&umSolidPresent()){return umTransportRejected(2u);}":""}
 let p=vec3f(vertex);let slot=umVertexSlot(vec3i(vertex));let local=min(slot.xyz,vec3u(31u))>>vec3u(4u);
 let region=8u*slot.w+local.x+2u*local.y+4u*local.z;let model=${this.modelWord}u+8u*region;
 let coverage=atomicLoad(&umClaims[model+5u]);if(coverage==0u){return umTransportRejected(0u);}
 let c=umLatticeLoad(vec3i(vertex),width);let h=params.hDt.xyz;let dt=params.hDt.w;
 let far=abs(c)>=2.0*f32(width)*min(h.x,min(h.y,h.z));
 ${this.solid?.coarse?/* wgsl */`let clearReach=max(2.0*f32(width)+0.5,select(2.0*f32(width)+${UNIFORM_SURFACE_VELOCITY_REACH_H.toFixed(1)},${UNIFORM_SURFACE_VELOCITY_REACH_H.toFixed(1)},far));
 let supportSolidClear=!umSolidPresent()||umSupportSolidClear(region,p,clearReach);
 if(umSolidFree&&umSolidPresent()&&!supportSolidClear){return umTransportRejected(2u);}`:"let supportSolidClear=false;"}
 let origin=vec3f(16u*umPageCoord(region));let reach=vec3f(2.0*f32(width)+0.5);
 let low=vec3i(floor((p-reach-origin)/16.0));let high=vec3i(floor((p+reach-origin)/16.0));
 if(any(low<vec3i(-1))||any(high>vec3i(1))){return umTransportRejected(1u);}
 var required=0u;
 for(var z=low.z;z<=high.z;z++){for(var y=low.y;y<=high.y;y++){for(var x=low.x;x<=high.x;x++){
  required|=1u<<u32((x+1)+3*(y+1)+9*(z+1));
 }}}
 if((coverage&required)!=required){return umTransportRejected(0u);}

 var velocities=UMRegionalVelocity(vec3f(0),vec3f(0),false);
 ${this.solid?.coarse?/* wgsl */`let tryLocalSolid=umExactLocalSolidTransport&&umSolidFree&&umFastSurface&&umSolidPresent()&&(width==1u||width==4u);
 if(tryLocalSolid){
  velocities=umRequiredVelocity(region,p,select(2.0*f32(width)+${UNIFORM_SURFACE_VELOCITY_REACH_H.toFixed(1)},${UNIFORM_SURFACE_VELOCITY_REACH_H.toFixed(1)},far),coverage);
  // Stored canonical-open support includes the near-foot stencil even
  // for a far marker. At4h, radius15 also encloses native cubic taps and
  // every incident coarse owner; the bound does not assume fine corners. Slot/window checks precede summary use.
  if(velocities.valid&&umCanonicalSupportAt(p,2.0*f32(width)+7.0)&&dt>=0.0&&dt<3.402823e38&&all(h>vec3f(0))&&all(h<vec3f(3.402823e38))){
   let travel=dt*max(abs(velocities.lo),abs(velocities.hi))/h;
   // Extra outward arithmetic allowance: stage coordinate and multiply/
   // divide rounding; generous enough to keep the actual RK span below1h.
   let allowance=64.0*1.1920929e-7*(vec3f(1)+abs(p)+travel);
   umLocalSolidAdmission=all(travel>=vec3f(0))&&all(travel+allowance<=vec3f(0.5));
  }
 }`:'let tryLocalSolid=false;'}
 let g=bitcast<vec3f>(vec3u(atomicLoad(&umClaims[model]),atomicLoad(&umClaims[model+1u]),atomicLoad(&umClaims[model+2u])));
 var rho=bitcast<f32>(atomicLoad(&umClaims[model+4u]));
 let eps=select(min(${UNIFORM_MIXED_VERTEX_SMOOTHING_PER_S.toExponential(8)}*dt,0.03125),0.0,far);
 if(!(rho>=0.0)){return umTransportRejected(0u);}
 if((2.953125+32.0*eps)*rho>0.02*min(h.x,min(h.y,h.z))){
  let localRho=bitcast<f32>(atomicLoad(&umClaims[model+6u]));
  if(32.0*eps*localRho>0.02*min(h.x,min(h.y,h.z))){return umTransportRejected(0u);}
  rho=umRequiredRho(region,g,required,rho);
 }
 if(!(32.0*eps*rho<=0.02*min(h.x,min(h.y,h.z)))){return umTransportRejected(0u);}
 if(!tryLocalSolid){velocities=umRequiredVelocity(region,p,select(2.0*f32(width)+${UNIFORM_SURFACE_VELOCITY_REACH_H.toFixed(1)},${UNIFORM_SURFACE_VELOCITY_REACH_H.toFixed(1)},far),coverage);}
 if(!velocities.valid){return umTransportRejected(0u);}let lo=velocities.lo;let hi=velocities.hi;
 if(any(dt*max(abs(lo),abs(hi))/h>vec3f(0.5))){return umTransportRejected(1u);}

 if(umWallPlane(vec3i(vertex))<6u){return umTransportRejected(2u);}
 // The validated phi support rectangle consists only of stored pages.
 // Every crossed face/diagonal umBlockHeld probe is therefore stored too:
 // page projection is identity.
 // Embedded solids remain covered by the separate clearance evidence.
 if(!supportSolidClear){for(var k=0u;k<8u;k++){
  let sign=2.0*vec3f(umCorner(k,2u))-vec3f(1.0);
  ${this.solid?/* wgsl */`// At w=4, p is 4-aligned: tile-clear cubes at p±5 span
  // [-12,-1] and [0,11] per axis, covering the full ±8.5 support.
  // At w=1, p±2 cubes overlap and cover at least ±6, beyond ±2.5.
  let clearPoint=vec3i(vertex)+vec3i(sign)*i32(width+1u);
  if(!umSolidClear(clearPoint)){return umTransportRejected(2u);}`:""}
 }}
 let result=umSurfaceTransport(c,g,rho,lo,hi,h,dt,eps,p,f32(width));
 if(result.w==0.0){return umTransportRejected(0u);}
 if(umReleasedMayChange(p,result.x-result.y)){return umTransportRejected(2u);}
 var value=result.x;
 ${this.sourceParams?"value=umSourceuvSourcePhi(p,value);":""}
 if(params.flags.z!=0u){atomicAdd(&umClaims[${this.counterWord+2}u],1u);umRecordRegion(vertex,1u);atomicMax(&umClaims[${this.counterWord+4}u],bitcast<u32>(result.y));}
 return vec2f(value,1.0);
}
// Far-band identity foot has no smoothing. A signed current velocity
// enclosure can prove every rounded RK endpoint equals p, without sampling
// velocity or loading the regional phi fit. Keep the full velocity halo
// certificate: nested fine/coarse MAC taps reach <5 base cells; RK third
// stages reach1.5 under the .5-cell step cap. Radius7 includes rounding and
// negative-owner assignment of faces exactly on a regional boundary.
fn umEarlyFarAdvected(vertex:vec3u,width:u32)->vec2f{
 if(width!=1u&&width!=4u){return vec2f(0.0);}
 // Positive fine vertices exclude every coarse incident tile upstream.
 // Low-wall rounds require a missing page and cannot pass stored support;
 // merged/owner/wall lists choose their authority upstream.
 let h=params.hDt.xyz;let dt=params.hDt.w;let c=umLatticeLoad(vec3i(vertex),width);
 if(!(abs(c)>=2.0*f32(width)*min(h.x,min(h.y,h.z)))||umWallPlane(vec3i(vertex))<6u){return vec2f(0.0);}
 let p=vec3f(vertex);let slot=umVertexSlot(vec3i(vertex));let local=min(slot.xyz,vec3u(31u))>>vec3u(4u);
 let region=8u*slot.w+local.x+2u*local.y+4u*local.z;
 let coverage=atomicLoad(&umClaims[${this.modelWord}u+8u*region+5u]);
 if(!umVelocityCovered(region,p,${UNIFORM_SURFACE_VELOCITY_REACH_H.toFixed(1)},coverage)){return vec2f(0.0);}
 var velocities=umHaloVelocity(region);
 var interval=umTraceEndpointInterval(p,vec3f(0.0),velocities.lo,velocities.hi,vec3f(0.0),h,dt);
 if(!all(interval.lo==p)||!all(interval.hi==p)){
  velocities=umRequiredVelocity(region,p,${UNIFORM_SURFACE_VELOCITY_REACH_H.toFixed(1)},coverage);
  interval=umTraceEndpointInterval(p,vec3f(0.0),velocities.lo,velocities.hi,vec3f(0.0),h,dt);
 }
 if(!velocities.valid||!all(velocities.lo<=velocities.hi)||!all(dt*max(abs(velocities.lo),abs(velocities.hi))/h<=vec3f(0.5))||!(dt>=0.0)){return vec2f(0.0);}
 if(!all(interval.lo==p)||!all(interval.hi==p)){return vec2f(0.0);}
 var value=c;
 ${this.solid?/* wgsl */`if(umSolidFree&&umSolidPresent()){
  // Both the buried shortcut and the open-cell path return c when no
  // continuation/source can change it. This certificate needs no solid
  // sampler, so owner launches may retire the work before enqueueing it.
  if(umReleasedMayChange(p,c)){return vec2f(0.0);}
  ${this.sourceParams?"if(umSourceuvSourcePhi(p,c)!=c){return vec2f(0.0);}":""}
 }else{
  let buried=!umSolidClear(vec3i(vertex))&&umBuried(p);if(!buried){value=umReleasedWalls(p,value);
  ${this.sourceParams?"value=umSourceuvSourcePhi(p,value);":""}
  }
 }`:/* wgsl */`value=umReleasedWalls(p,value);
 ${this.sourceParams?"value=umSourceuvSourcePhi(p,value);":""}` }
 if(params.flags.z!=0u){atomicAdd(&umClaims[${this.counterWord}u],1u);atomicAdd(&umClaims[${this.counterWord+8}u],1u);atomicAdd(&umClaims[${this.counterWord+11}u],1u);umRecordRegion(vertex,2u);}
 return vec2f(value,1.0);
}
fn umEulerRejected(reason:u32)->vec2f{if(params.flags.z!=0u){atomicAdd(&umClaims[${this.counterWord+9}u],1u);atomicAdd(&umClaims[${this.counterWord+12}u+reason],1u);}return vec2f(0.0);}
fn umEulerAdvected(vertex:vec3u,width:u32,foot:UMFoot)->vec2f{
 if(width!=1u&&width!=4u){return umEulerRejected(0u);}
 ${this.solid?"if(umSolidFree&&umSolidPresent()&&!umLocalSolidAdmission){return umEulerRejected(0u);}":""}
 let p=vec3f(vertex);let slot=umVertexSlot(vec3i(vertex));let local=min(slot.xyz,vec3u(31u))>>vec3u(4u);
 let region=8u*slot.w+local.x+2u*local.y+4u*local.z;let model=${this.modelWord}u+8u*region;
 let coverage=atomicLoad(&umClaims[model+5u]);if(coverage==0u){return umEulerRejected(1u);}
 let far=abs(foot.center)>=2.0*f32(width)*min(params.hDt.x,min(params.hDt.y,params.hDt.z));
 let origin=vec3f(16u*umPageCoord(region));let reach=vec3f(select(2.0*f32(width)+0.5,f32(width),far));
 let low=vec3i(floor((p-reach-origin)/16.0));let high=vec3i(floor((p+reach-origin)/16.0));
 if(any(low<vec3i(-1))||any(high>vec3i(1))){return umEulerRejected(1u);}
 var required=0u;for(var z=low.z;z<=high.z;z++){for(var y=low.y;y<=high.y;y++){for(var x=low.x;x<=high.x;x++){required|=1u<<u32((x+1)+3*(y+1)+9*(z+1));}}}
 if((coverage&required)!=required||umWallPlane(vec3i(vertex))<6u){return umEulerRejected(1u);}
 var g=vec3f(0.0);var rho=0.0;var velocities=umHaloVelocity(region);var lo=velocities.lo;var hi=velocities.hi;var tightVelocity=false;
 if(!umVelocityCovered(region,p,select(2.0*f32(width)+${UNIFORM_SURFACE_VELOCITY_REACH_H.toFixed(1)},${UNIFORM_SURFACE_VELOCITY_REACH_H.toFixed(1)},far),coverage)){return umEulerRejected(1u);}
 let h=params.hDt.xyz;let dt=params.hDt.w;
 if(any(dt*max(abs(lo),abs(hi))/h>vec3f(0.5))){
  velocities=umRequiredVelocity(region,p,select(2.0*f32(width)+${UNIFORM_SURFACE_VELOCITY_REACH_H.toFixed(1)},${UNIFORM_SURFACE_VELOCITY_REACH_H.toFixed(1)},far),coverage);lo=velocities.lo;hi=velocities.hi;tightVelocity=true;
  if(!velocities.valid){return umEulerRejected(1u);}if(any(dt*max(abs(lo),abs(hi))/h>vec3f(0.5))){return umEulerRejected(4u);}
 }
 // Valid stored velocity support contains the phi box, including all
 // face/diagonal pages umPointFrom could test. Its projection is identity.
 umSlow=false;let u=umFastVelocity(foot.foot);if(umSlow){return umEulerRejected(3u);}
 if(umSurfaceTraceSeed&&umFastSurface){umTraceSeedPoint=foot.foot;umTraceSeedVelocity=u;umTraceSeedValid=true;}
 if(any(lo>u)||any(u>hi)){return umEulerRejected(4u);}
 var interval=umTraceEndpointInterval(foot.foot,foot.shift,lo,hi,u,h,dt);
 var localCell:UMVertexCell;var useLocalCell=false;
 var bound=vec4f(0.0);
 if(far){
  let L=umRequiredLipschitz(region,required);
  bound=umSurfaceNativeFarBound(foot.center,L,h,dt,p,f32(width),interval);
  if(bound.w==0.0&&!tightVelocity){
   velocities=umRequiredVelocity(region,p,${UNIFORM_SURFACE_VELOCITY_REACH_H.toFixed(1)},coverage);lo=velocities.lo;hi=velocities.hi;tightVelocity=true;
   if(!velocities.valid||any(lo>u)||any(u>hi)){return umEulerRejected(4u);}
   interval=umTraceEndpointInterval(foot.foot,foot.shift,lo,hi,u,h,dt);
   bound=umSurfaceNativeFarBound(foot.center,L,h,dt,p,f32(width),interval);
  }
 }
 if(bound.w==0.0){
  g=bitcast<vec3f>(vec3u(atomicLoad(&umClaims[model]),atomicLoad(&umClaims[model+1u]),atomicLoad(&umClaims[model+2u])));
  rho=bitcast<f32>(atomicLoad(&umClaims[model+4u]));
  bound=umSurfaceEndpointBound(foot.center,g,rho,h,dt,p,f32(width),foot.smoothing,interval,far);
 }
 if(bound.w==0.0){
  rho=umRequiredRho(region,g,required,rho);
  bound=umSurfaceEndpointBound(foot.center,g,rho,h,dt,p,f32(width),foot.smoothing,interval,far);
  if(bound.w==0.0&&!tightVelocity){
   velocities=umRequiredVelocity(region,p,select(2.0*f32(width)+${UNIFORM_SURFACE_VELOCITY_REACH_H.toFixed(1)},${UNIFORM_SURFACE_VELOCITY_REACH_H.toFixed(1)},far),coverage);lo=velocities.lo;hi=velocities.hi;
   if(!velocities.valid){return umEulerRejected(1u);}if(any(lo>u)||any(u>hi)){return umEulerRejected(4u);}
   interval=umTraceEndpointInterval(foot.foot,foot.shift,lo,hi,u,h,dt);
   bound=umSurfaceEndpointBound(foot.center,g,rho,h,dt,p,f32(width),foot.smoothing,interval,far);
  }
  if(bound.w==0.0&&far){
   // A local eight-corner certificate is independent of regional phi
   // outliers. Tighten velocity first, then reuse this exact Euler cell.
   localCell=umVertexCell(interval.euler);
   let local=umSurfaceFarCellBound(foot.center,width,h,localCell.origin,localCell.width,localCell.corners,interval);
   if(local.y>0.0){useLocalCell=true;bound=vec4f(local.x,0.0,1.0,1.0);}
  }
  if(bound.w==0.0){return umEulerRejected(6u);}
 }
 let endpointIdentity=all(interval.lo==p)&&all(interval.hi==p)&&all(interval.euler==p);
 ${this.solid?/* wgsl */`if(endpointIdentity){if(!umSolidClear(vec3i(vertex))&&umBuried(p)){return umEulerRejected(2u);}}
 else if(!umSolidClear(vec3i(vertex))){
  let lower=umPointCell(min(p,interval.lo));let upper=umPointCell(max(p,interval.hi));
  for(var z=lower.z;z<=upper.z;z++){for(var y=lower.y;y<=upper.y;y++){for(var x=lower.x;x<=upper.x;x++){
   if(!umCellIsOpen(vec3i(x,y,z))){return umEulerRejected(2u);}
  }}}
 }`:""}
 let end=umPointFrom(p,interval.euler);
 ${this.solid?"let walk=umWalk(p,end);if(walk.hit&&!endpointIdentity){return umEulerRejected(7u);}let q=walk.q;":"let q=end;"}
 var value=foot.center;if(!endpointIdentity){if(useLocalCell){value=localCell.value;}else{value=umVertexCell(q).value;}}
 if(!endpointIdentity&&umReleasedMayChange(p,value-bound.x)){return umEulerRejected(7u);}
 ${this.solid?"value=umWallContact(p,value,width);":"value=umWallContact(p,value,width);"}
 value=umReleasedWalls(p,value);
 ${this.sourceParams?"value=umSourceuvSourcePhi(p,value);":""}
 if(params.flags.z!=0u){if(useLocalCell){atomicAdd(&umClaims[${this.counterWord+1}u],1u);}if(far){atomicAdd(&umClaims[${this.counterWord+11}u],1u);}atomicAdd(&umClaims[${this.counterWord+8}u],1u);umRecordRegion(vertex,2u);atomicMax(&umClaims[${this.counterWord+10}u],bitcast<u32>(bound.x));}
 return vec2f(value+foot.smoothing,1.0);
}
`:"fn umEarlyFarAdvected(vertex:vec3u,width:u32)->vec2f{return vec2f(0.0);} fn umRecordRegion(vertex:vec3u,tier:u32){} fn umCheapAdvected(vertex:vec3u,width:u32)->vec2f{return vec2f(0.0); } fn umEulerAdvected(vertex:vec3u,width:u32,foot:UMFoot)->vec2f{return vec2f(0.0);}"}
// Whether umReleasedWalls can change value at p: a direction's reach clears
// its own skip test at the wall umWallAhead finds. Accepted continuations
// only raise result, so testing the entry value probes a superset of its
// walls. A wall that passes on its direction's reach is retested on the
// reach of the face tiles holding p's probe cells on it (umFaceTile of each
// column's block): a probe's accepted dt*away is at most its tile's reach,
// so a vertex failing both skips keeps its value. Only vertices above
// released faces moving off the wall defer, not every vertex any released
// face can reach.
fn umReleasedMayChange(p:vec3f,value:f32)->bool{
 let h=params.hDt.xyz;let slack=1e-5*max(h.x,max(h.y,h.z));let at=vec3i(p);
 for(var k=0u;k<6u;k++){
  let reach=umPlaneReach[k];
  if(!(reach>0.0)){continue;}
  let axis=k>>1u;let inward=select(1.0,-1.0,(k&1u)!=0u);
  let wall=umWallAhead(at,k,reach,value,h);
  let depth=inward*(p[axis]-f32(wall.plane))*h[axis];
  var local=0u;
  for(var c=0u;c<4u;c++){
   if(wall.blocks[c]==0u){continue;}
   var cell=at;cell[(axis+1u)%3u]-=i32(1u-(c&1u));cell[(axis+2u)%3u]-=i32(1u-(c>>1u));
   local=max(local,atomicLoad(&umClaims[UM_WALL_TILE_REACH+umFaceTile(wall.blocks[c],k,vec3u(cell)>>vec3u(2u))]));
  }
  let tileReach=bitcast<f32>(local);
  if(tileReach>0.0&&!(tileReach-depth+slack<value)){return true;}
 }return false;
}
// umAdvected off the walls, without the wall and solid continuations. A
// vertex any continuation can change goes to advectDeferred, which evaluates
// umAdvected itself; a vertex on a wall (umWallPlane: the only vertices
// umWallContact can change) is advectWalls'. Keeping the continuations'
// sampler call sites out of this kernel is its cost.
${this.farGate?/* wgsl */`// A far-marker vertex (at least clearAir's 16 max h) of a tile the gate
// leaves out keeps its value, the source's plug included: this step's clear
// lifts whatever its advection gives there to the marker, so tracing it is
// no work. Negative where the vertex is traced.
fn umFarAir(vertex:vec3u)->f32{
 let tile=umTileAt(vec3u(umPointCell(vec3f(vertex)))/4u);
 if(atomicLoad(&umClaims[${this.claimBytes/4}u+tile])!=0u){return -1.0;}
 let value=umLoadVertex(vertex);
 return select(-1.0,${this.sourceParams?"umSourceuvSourcePhi(vec3f(vertex),value)":"value"},value>=16.0*max(params.hDt.x,max(params.hDt.y,params.hDt.z)));
}`:"fn umFarAir(vertex:vec3u)->f32{return -1.0;}"}
// A deep air vertex of a still tile keeps its value (the source's plug
// included): the tile's spread speed certificate (umReach: ceil(travel/4)+2
// tiles a direction) is 2 on every axis only where no velocity a departure
// from it reads moves, so its trace ends on the vertex itself and returns
// the stored value. It is that deep (no foot, smoothing or cubic, which
// start within 2 widths) to stay the plain lattice read. A moving tile is
// traced: a kept value there is only within the departure's length of the
// traced one, and the census' thin and shape criteria read those vertices
// (a free-falling ball refined from frame 19 instead of at impact, frame 49,
// cm12-figure-7-256). Negative where the vertex is traced.
fn umDeepAir(vertex:vec3u,width:u32)->f32{
 let h=min(params.hDt.x,min(params.hDt.y,params.hDt.z));
 let value=umLoadVertex(vertex);if(!(value>=(6.0+3.0*f32(width))*h)){return -1.0;}
 let tile=umTileAt(vec3u(umPointCell(vec3f(vertex)))/4u);
 let word=atomicLoad(&umClaims[${this.evidenceBytes/4-this.ownership.capacity.tiles}u+tile])&0x3fffffffu;
 let reach=max(vec3u(word&31u,(word>>5u)&31u,(word>>10u)&31u),vec3u((word>>15u)&31u,(word>>20u)&31u,(word>>25u)&31u));
 if(any(reach>vec3u(2u))||umReleasedMayChange(vec3f(vertex),value)){return -1.0;}
 return ${this.sourceParams?"umSourceuvSourcePhi(vec3f(vertex),value)":"value"};
}
override umDeferTraced:bool=false;
override umCubicListed:bool=false;
// The cubic list's first word in deferred.data: past a word per lattice vertex.
const UM_CUBIC_BASE=${this.address.vertexKeys}u;
fn umAdvectStore(vertex:vec3u,width:u32){
 umLocalSolidAdmission=false;umTraceSeedValid=false;
 umFullCubic=false;umFullIdentity=false;
 if(params.hDt.w>0.0&&umWallPlane(vec3i(vertex))<6u){return;}
 {let far=umFarAir(vertex);if(far>=0.0){umRecordRegion(vertex,0u);umStoreSurface(vertex,width,far);return;}}
 {let deep=umDeepAir(vertex,width);if(deep>=0.0){umRecordRegion(vertex,0u);umStoreSurface(vertex,width,deep);return;}}
 {let early=umEarlyFarAdvected(vertex,width);if(early.y>0.0){umStoreSurface(vertex,width,early.x);return;}}
 {let cheap=umCheapAdvected(vertex,width);if(cheap.y>0.0){umStoreSurface(vertex,width,cheap.x);return;}}
 // A launch compiled with umDeferTraced lists every vertex it would trace:
 // advectDeferred's grid stride evaluates them densely, where this launch's
 // lanes (most of them far or deep air) would idle behind the few traces.
 // A solid-free fine launch keeps only locally certified current-open
 // full transport; every other globally solid-present vertex is listed.
 let p=vec3f(vertex);var defer=umDeferTraced${this.solid?"||(umSolidFree&&umSolidPresent()&&!umLocalSolidAdmission)":""};var value=0.0;
 ${this.workControl?`if(umLocalSolidAdmission&&params.flags.z!=0u){atomicAdd(&umClaims[${this.counterWord+20}u],1u);}`:""}
 ${this.solid?/* wgsl */`// Off a clear tile a vertex can be buried: it keeps its value, umAdvected's
 // first exit.
 if(!umSolidClear(vec3i(p))&&umBuried(p)){umRecordRegion(vertex,0u);umStoreSurface(vertex,width,umLoadVertex(vertex));return;}
 // A hit is evaluated by advectDeferred. Its first-pass phi and source
 // samples are discarded.
 var smoothing=0.0;
 {if(!defer){umSlow=false;let foot=umVertexFoot(vertex,width);
 let euler=umEulerAdvected(vertex,width,foot);if(euler.y>0.0){${this.workControl?`if(umLocalSolidAdmission&&params.flags.z!=0u){atomicAdd(&umClaims[${this.counterWord+21}u],1u);}`:""}umStoreSurface(vertex,width,euler.x);return;}smoothing=foot.smoothing;let walk=umWalk(p,umVertexTrace(foot));defer=walk.hit||umSlow;let q=walk.q;if(!defer){`:`var smoothing=0.0;{if(!defer){umSlow=false;let foot=umVertexFoot(vertex,width);
 let euler=umEulerAdvected(vertex,width,foot);if(euler.y>0.0){umStoreSurface(vertex,width,euler.x);return;}smoothing=foot.smoothing;let q=umVertexTrace(foot);defer=umSlow;if(!defer){`}
  let cell=umVertexCell(q);value=cell.value;
  let h=f32(width)*min(params.hDt.x,min(params.hDt.y,params.hDt.z));
  if(params.flags.y!=0u&&abs(value)<2.0*h&&(params.flags.y==1u||width>1u)){
   // A launch compiled umCubicListed (advectFine) lists the vertex, its end
   // and smoothing for advectCubic: the cubic's lanes then run densely, not
   // idle behind the trace or hold the trace's lanes behind 64 taps.
   if(umCubicListed){
    ${this.workControl?`if(umLocalSolidAdmission&&params.flags.z!=0u){atomicAdd(&umClaims[${this.counterWord+21}u],1u);}`:""}
    let j=atomicAdd(&deferred.header[1],1u);let b=UM_CUBIC_BASE+5u*j;
    deferred.data[b]=umVertexAt(vertex);deferred.data[b+1u]=bitcast<u32>(q.x);deferred.data[b+2u]=bitcast<u32>(q.y);deferred.data[b+3u]=bitcast<u32>(q.z);deferred.data[b+4u]=bitcast<u32>(smoothing);
    return;
   }
   umFullCubic=true;value=umCubicPhi(q,cell);
  }
  defer=defer||umReleasedMayChange(p,value);
  value=${this.sourceParams?"umSourceuvSourcePhi(p,value)":"value"};
 }
 }}
 ${this.workControl?`if(!defer&&umLocalSolidAdmission&&params.flags.z!=0u){atomicAdd(&umClaims[${this.counterWord+21}u],1u);}`:""}
 if(!defer){umRecordRegion(vertex,select(3u,4u,umFullCubic));umStoreSurface(vertex,width,value+smoothing);return;}
 // Each canonical vertex is stored once: the list holds at most every vertex.
 // A word is the vertex key (umVertexAt) below a two-bit code: log2 of the
 // width (0, 2) off the regular-h dispatch, 3 on it. All ones is no word.
 let i=atomicAdd(&deferred.header[0],1u);
 deferred.data[i]=umVertexAt(vertex)|(select(firstTrailingBit(width),3u,umRegularFine)<<30u);
}
// The deferred vertices of the dispatch compiled with the same umRegularFine.
// A fixed grid (deferredGrid workgroups) strides over the list.
// With solids bound, ordinary free/general twins retain their whole-scene
// gate. The existing merged fast twin also evaluates locally certified
// current-open entries while a remote solid is present, leaving every
// rejected/cache-miss word to the general pipeline. It writes all
// ones over its word; the later launches skip such words. An all-ones word
// reads as code 3 (regular), so the regular launches test it.
@compute @workgroup_size(64) fn advectDeferred(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 let localSolidTwin=umExactLocalSolidTransport&&umFastSurface&&umSolidFree&&umSolidPresent();
 if(umSolidSkip()&&!localSolidTwin){return;}
 umStagePlaneReach(lane);
 let count=atomicLoad(&deferred.header[0]);
 for(var i=gid.x;i<count;i+=64u*groups.x){
  let word=deferred.data[i];if(word==0xffffffffu){continue;}let code=word>>30u;
  if((code==3u)!=umRegularFine){continue;}
  let vertex=umVertexCoord(word&0x3fffffffu);
  let width=select(1u<<code,1u,code==3u);
  umLocalSolidAdmission=false;umTraceSeedValid=false;
  let early=umEarlyFarAdvected(vertex,width);
  if(early.y>0.0){if(umFastSurface){deferred.data[i]=0xffffffffu;}umStoreSurface(vertex,width,early.x);continue;}
  let cheap=umCheapAdvected(vertex,width);
  if(cheap.y>0.0){if(umFastSurface){deferred.data[i]=0xffffffffu;}umStoreSurface(vertex,width,cheap.x);continue;}
  // A globally solid-free fast pipeline may evaluate only a currently
  // certified open footprint. Uncertified words remain for the general twin.
  if(localSolidTwin&&!umLocalSolidAdmission){continue;}
  ${this.workControl?`if(localSolidTwin&&params.flags.z!=0u){atomicAdd(&umClaims[${this.counterWord+20}u],1u);}`:""}
  umFullCubic=false;umFullIdentity=false;umSlow=false;let foot=umVertexFoot(vertex,width);
  let euler=umEulerAdvected(vertex,width,foot);if(euler.y>0.0){if(umFastSurface){deferred.data[i]=0xffffffffu;}${this.workControl?`if(localSolidTwin&&params.flags.z!=0u){atomicAdd(&umClaims[${this.counterWord+21}u],1u);}`:""}umStoreSurface(vertex,width,euler.x);continue;}
  umSlow=false;let value=umAdvected(vec3f(vertex),width,foot)+foot.smoothing;
  // The fast twin leaves a vertex with a slow sample to the general launch
  // and retires the rest from the list.
  if(umFastSurface){if(umSlow){continue;}deferred.data[i]=0xffffffffu;}
  ${this.workControl?`if(localSolidTwin&&params.flags.z!=0u){atomicAdd(&umClaims[${this.counterWord+21}u],1u);}`:""}
  umRecordRegion(vertex,select(select(3u,4u,umFullCubic),0u,umFullIdentity));umStoreSurface(vertex,width,value);
 }
}
// Canonical vertices of the walls, each once, with its authority's width. A
// lane per vertex of each listed wall face (umWallFace, 33 x 33), the launch
// striding the list. A vertex several wall faces share is listed by the one
// on its first wall plane (umWallPlane) whose block umWallBlock names, and
// its width is read at its cell in that block's page. With dt>0
// umAdvectStore leaves them all to this entry, which lists them for
// advectDeferred (its umAdvected is this vertex's value): a plane of 4h
// tiles keeps one vertex in sixteen, so evaluating here idled most lanes of
// every group behind the few characteristics. With dt<=0 the owner passes
// store them (the wall continuations are then the identity).
${uniformMixedWallFaceWGSL}
var<workgroup> wallListed:atomic<u32>;
var<workgroup> wallBase:u32;
@compute @workgroup_size(64) fn advectWalls(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 let lanes=select(0u,1089u*umWallFaceCount(),params.hDt.w>0.0);
 for(var first=0u;first<lanes;first+=64u*groups.x){
  if(lane==0u){atomicStore(&wallListed,0u);}
  workgroupBarrier();
  let i=first+gid.x;var word=0xffffffffu;var slot=0u;
  if(i<lanes){
   let face=umWallFace(i/1089u);let vertex=umWallFaceVertex(face,i%1089u);
   if(umWallPlane(vertex)==face%6u&&umWallBlock(vertex,face%6u)==face/6u){
    var width=1u;
    if(!umRegularFine){
     let low=vec3i(umBlockPage(face/6u)<<vec3u(5u));let cell=vec3u(clamp(vertex,low,low+vec3i(31)));
     width=select(1u,4u,umCoarseIncident(umTileAt(cell/4u),vec3u(vertex)-(cell/4u)*4u)!=0u);
    }
    let far=umFarAir(vec3u(vertex));
    if(far>=0.0&&all(vec3u(vertex)%width==vec3u(0))){umRecordRegion(vec3u(vertex),0u);umStoreSurface(vec3u(vertex),width,far);}
    else if(all(vec3u(vertex)%width==vec3u(0))){
     slot=atomicAdd(&wallListed,1u);
     word=umVertexAt(vec3u(vertex))|(select(firstTrailingBit(width),3u,umRegularFine)<<30u);
    }
   }
  }
  workgroupBarrier();
  // One list atomic per group and round.
  if(lane==0u){let n=atomicLoad(&wallListed);if(n>0u){wallBase=atomicAdd(&deferred.header[0],n);}}
  workgroupBarrier();
  if(word!=0xffffffffu){deferred.data[wallBase+slot]=word;}
 }
}
// Six samples through one sampler call site: low then high per axis. The
// resolved sampler is eight loads, so its six samples unroll.
${this.solid?/* wgsl */`// A closed owner's corners are not all state (umVertexBuried): a closed 4h
// owner's, and a closed h cell's beside an h seam, whose buried corners hold
// the air sentinel. A search from the seam's 4h corner found that sentinel's
// zero a tenth of a cell away, sixteen cells under the surface.
fn umSampleClosed(q:vec3f)->bool{
 if(!umSolidEnabled()){return false;}
 let c=umPointCell(q);let tile=c>>vec3u(2u);let t=umTileKey(tile);
 if(!umKeyStored(t)){return umTileOpenAt(tile)<=1e-5;}
 if(umTileWidth(t)==4u){return umTileOpen(t)<=1e-5;}
 return !umCellIsOpen(c);
}`:""}
// Each face direction's (2*axis+side) largest dt*away over the wall faces
// umReleasedWalls can probe that pass its release test, as f32 bits
// (positive floats order as u32), in umClaims[UM_WALL_REACH+direction]
// (zeroed with the claims). Every probe resolves to the owner of one cell on
// a wall face, so scanning those cells covers each probe exactly: the launch
// strides the wall-face list (umWallFace), a workgroup a face and a lane a
// tile of its plane, which takes the maximum over the tile's plane cells
// aligned to its width (a 4h tile's sixteen all resolve to its single
// owner). The maximum does not depend on the order the lanes combine in.
// The same maximum per tile (UM_WALL_TILE_REACH+umFaceTile(block,face,tile),
// 64 words a wall face, a lane each) bounds the probes whose owners lie in
// that tile.
const UM_WALL_REACH:u32=${UNIFORM_MIXED_CLAIM_WORDS}u;
const UM_WALL_TILE_REACH:u32=${SURFACE_CLAIM_WORDS}u;
@compute @workgroup_size(64) fn wallReach(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 if(group.x==0u&&lane==0u){atomicStore(&deferred.header[0],0u);atomicStore(&deferred.header[1],0u);}
 let h=params.hDt.xyz;let dt=params.hDt.w;let faces=umWallFaceCount();
 for(var i=group.x;i<faces;i+=groups.x){
  let face=umWallFace(i);let f=face%6u;let axis=f>>1u;let upper=(f&1u)!=0u;let inward=select(1.0,-1.0,upper);
  let tile=umWallFaceTile(face,lane);let width=umTileWidth(umTileAt(tile));var bits=0u;
  for(var j=0u;j<16u;j+=width*width){
   var cell=vec3i(tile<<vec3u(2u));cell[axis]+=select(0,3,upper);cell[(axis+1u)%3u]+=i32(j&3u);cell[(axis+2u)%3u]+=i32(j>>2u);
   let owner=umOwnerAt(cell);
   let origin=umOrigin(owner);var anchor=vec3i(origin);anchor[axis]+=i32(owner.width)-1;
   let wallTexel=umLoadWallTexel(anchor,owner.width);var speed=wallTexel[axis];var bit=axis;
   if(!upper){speed=negative[umLowFaceIndex(origin,axis)];bit+=3u;}
   let released=(u32(round(wallTexel.w))&(1u<<bit))!=0u;
   let away=inward*speed;
   if(released&&dt*away>1e-4*h[axis]*f32(owner.width)){bits=max(bits,bitcast<u32>(dt*away));}
  }
  // A face tile has one lane: its word has no other writer.
  if(bits!=0u){atomicStore(&umClaims[UM_WALL_TILE_REACH+umFaceTile(face/6u,f,tile)],bits);atomicMax(&umClaims[UM_WALL_REACH+f],bits);}
 }
}
${this.solid?/* wgsl */`
${this.solid.coarse?/* wgsl */`
// The frame's solid record already classifies all 64 cells of each tile.
// One lane reads its mask instead of re-evaluating geometry in 64 lanes.
@compute @workgroup_size(64) fn solidClosed(@builtin(global_invocation_id) gid:vec3u){
 let tile=gid.x+64u*umDispatchX*gid.y;if(!umTileLive(tile)){return;}
 var closed=false;if(umSolidEnabled()){closed=any(umSolidOpenCells(tile)!=vec2u(0xffffffffu));}
 evidence[tile]=u32(closed);
}
`:/* wgsl */`var<workgroup> solidClosedFlag:atomic<u32>;
@compute @workgroup_size(64) fn solidClosed(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 let tile=group.x+umDispatchX*group.y;if(!umTileLive(tile)){return;}
 if(lane==0u){atomicStore(&solidClosedFlag,0u);}workgroupBarrier();
 if(umCellOpen(vec3i(umTileCoord(tile)*4u+umCorner(lane,4u)))<=1e-5){atomicStore(&solidClosedFlag,1u);}
 workgroupBarrier();if(lane==0u){evidence[tile]=atomicLoad(&solidClosedFlag);}
}
`}
@compute @workgroup_size(64) fn solidClear(@builtin(global_invocation_id) gid:vec3u){
 let tile=gid.x+64u*umDispatchX*gid.y;if(!umTileLive(tile)){return;}
 let t=vec3i(umTileCoord(tile));var clear=1u;
 for(var k=0u;k<27u;k++){
  let n=t+vec3i(umCorner(k,3u))-vec3i(1);
  if(umTilePresent(n)&&evidence[umTileAt(vec3u(n))]!=0u){clear=0u;}
 }
 evidence[UM_TILES+tile]=clear;
}`:""}
// An h tile job's vertices (advectFine): a lane per positive
// vertex (local 1..4, its cells' upper corners); a tile on a negative domain
// wall also owns its wall vertices, in two more rounds. The tile owns a
// vertex exactly when no incident tile is 4h: a 4h owner is wider, and among
// h owners the lowest index is clamp(p-1)'s, in this tile. wide: the 4h
// tiles among tile+{0,1}^3, the only tiles incident to its vertices. low:
// the axes with no tile stored below this one (umTileLowOpen).
const UM_NO_VERTEX=vec3u(0xffffffffu);
fn umFineVertex(tile:vec3u,lane:u32,round:u32,wide:u32,low:vec3<bool>)->vec3u{
 var local=umCorner(lane,4u)+vec3u(1);
 if(round>0u){let j=lane+64u*(round-1u);if(j>=125u){return UM_NO_VERTEX;}local=umCorner(j,5u);if(all(local!=vec3u(0))){return UM_NO_VERTEX;}}
 let vertex=tile*4u+local;
 if(any((local==vec3u(0))&!low)){return UM_NO_VERTEX;}
 for(var k=1u;k<8u;k++){if((wide&(1u<<k))!=0u&&all((umCorner(k,2u)==vec3u(0))|(local==vec3u(4u)))){return UM_NO_VERTEX;}}
 return vertex;
}
// advectFine's cubic list (umCubicListed): each entry's vertex, trace end
// and smoothing, finished as umAdvectStore would: the cubic at the end, the
// released-wall test (a vertex it may change goes to the deferred list) and
// the source. A fixed grid (cubicGrid workgroups) strides over the list.
@compute @workgroup_size(64) fn advectCubic(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 umStagePlaneReach(lane);
 let count=atomicLoad(&deferred.header[1]);
 for(var i=gid.x;i<count;i+=64u*groups.x){
  let b=UM_CUBIC_BASE+5u*i;let key=deferred.data[b];let vertex=umVertexCoord(key);let p=vec3f(vertex);
  let q=bitcast<vec3f>(vec3u(deferred.data[b+1u],deferred.data[b+2u],deferred.data[b+3u]));let smoothing=bitcast<f32>(deferred.data[b+4u]);
  var value=umCubicPhi(q,umVertexCell(q));
  if(umReleasedMayChange(p,value)){let k=atomicAdd(&deferred.header[0],1u);deferred.data[k]=key;continue;}
  value=${this.sourceParams?"umSourceuvSourcePhi(p,value)":"value"};
  umRecordRegion(vertex,4u);umStoreSurface(vertex,1u,value+smoothing);
 }
}
@compute @workgroup_size(64) fn advectFine(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 umStagePlaneReach(lane);
 let owner=umTileJobOwner(group);
 if(owner.width==0u){return;}
 let tile=umTileCoord(owner.tile);let wide=umFineWideMask(umTileAt(tile));
 let low=umTileLowOpen(vec3i(tile));let rounds=select(1u,3u,any(low));
 for(var round=0u;round<rounds;round++){
  let vertex=umFineVertex(tile,lane,round,wide,low);
  if(vertex.x!=UM_NO_VERTEX.x){umAdvectStore(vertex,1u);}
 }
}
// Merged jobs without the general h list (umMergedCoarse): a seam 4h tile's
// candidate vertices are its closure's eight 4-aligned ones, a lane each,
// umMergedPack tiles per job.
// Packed merged jobs run 64 regular coarse owners, one lane each, with the
// vertices advectOwners gives a regular owner. One evaluation call site.
@compute @workgroup_size(64) fn advect(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 umStagePlaneReach(lane);
 let job=group.x+umDispatchX*group.y;let tiles=umMergedTileJobs();
 let packed=umMergedTiles&&!umFusedJobs&&job>=tiles;
 var owner=UMOwner();var first=7u;var tileVertex=vec3u(0);var low=vec3<bool>(false);
 if(!packed){
  let header=7u*UM_TILES+16u;let slot=lane/8u;let index=job*umMergedPack+slot;
  if(slot>=umMergedPack||index>=umSupport[header+1u]){return;}
  let tile=umSupport[header+4u+umSupport[header]+index];let tileOwner=UMOwner(tile,0u,4u,umTopology[tile]&0x3fffffffu);
  tileVertex=umTileCoord(tile)*4u+umCorner(lane%8u,2u)*4u;
  if((umCoarseCornerMask(tile)&(1u<<(lane%8u)))==0u){return;}
  owner=tileOwner;
 }else{
  owner=umRegularCoarseOwner((job-tiles)*64u+lane);if(owner.width==0u){return;}
  // Owners on a negative domain wall also own their corners on that wall.
  low=umLowOpen(vec3i(umOrigin(owner)));if(any(low)){first=0u;}
 }
 let origin=umOrigin(owner);
 for(var k=first;k<8u;k++){
  let corner=umCorner(k,2u);
  if(packed&&!all((corner!=vec3u(0))|low)){continue;}
  let vertex=select(tileVertex,origin+corner*owner.width,packed);
  umAdvectStore(vertex,owner.width);
 }
}
@compute @workgroup_size(64) fn advectOwners(@builtin(global_invocation_id) gid:vec3u,@builtin(local_invocation_index) lane:u32){
 umStagePlaneReach(lane);
 let owner=umOwner(gid);
 if(owner.width==0u){return;}
 let origin=umOrigin(owner);let regular=umRegularFine||(umTileMaximumWidth(owner.tile)==owner.width&&umTileMinimumWidth(owner.tile)==owner.width);
 let low=umLowOpen(vec3i(origin));
 // Every regular interior owner writes exactly its positive corner, first;
 // only owners touching a negative domain wall own additional vertices. One
 // umAdvectStore call site (Metal inlines every call): j==0 is the regular
 // positive corner, j>0 corner j-1.
 for(var j=select(1u,0u,regular);j<=umCounts.w;j++){
  if(j==1u&&regular&&!any(low)){return;}
  var vertex=origin+vec3u(owner.width);var owned=j==0u;
  if(j>0u){
   let k=j-1u;if(regular&&k==7u){continue;}
   let corner=umCorner(k,2u);vertex=origin+corner*owner.width;
   if(regular){owned=all((corner!=vec3u(0))|low);}
   else{owned=umVertexAuthority(vertex).index==owner.index;}
  }
  if(owned){umAdvectStore(vertex,owner.width);}
 }
}
`,["advect","advectFine","advectOwners"],"umClaims")});
    const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");
    if(errors.length)throw new Error(errors.map(m=>`${m.lineNum}: ${m.message}`).join("\n"));
    const layout=this.device.createPipelineLayout({bindGroupLayouts:[this.ownership.bindLayout,this.resources,...(this.solid?[this.solid.tileLayout]:[]),this.ownership.hangingLayout]});
    const compile=(entryPoint:string,constants:Record<string,number>,solidsOnly=false)=>this.twin(s=>uniformDetailPipeline(this.device,this.ownership,{layout,compute:{module,entryPoint,constants:{umDispatchX:this.ownership.dispatchX,...constants,...s}}}),{solidsOnly,entry:entryPoint});
    const keyed=async(entryPoint:string,constants:Record<string,number>={},solidsOnly=false)=>{this.pipelines.set(entryPoint,await compile(entryPoint,constants,solidsOnly));};
    const regular={umCellWidth:1,umPlannedFine:1,umRegularFine:1};
    const free:Record<string,number>=this.solid?{umSolidFree:1}:{},twinned:Record<string,number>=this.solid?{umSolidTwinned:1}:{};
    await Promise.all([
      // Dispatched only under solids (encode: solid.present): no solid-free twin.
      ...(this.solid?["solidClosed","solidClear"]:[]).map(entryPoint=>keyed(entryPoint,{},true)),
      keyed("wallReach"),
      keyed("advect",{umMergedTiles:1,umMergedCoarse:1,umMergedPack:8,umCertifiedJobs:2,umClaimWord:2,umDeferTraced:1}),
      // The fast trace (umFastSurface), solid-free: with a solid present it lists every traced vertex.
      keyed("advectFine",{umCellWidth:1,umPlannedFine:2,umCertifiedJobs:1,umClaimWord:1,...UNIFORM_DETAIL_H_STORE,umFastSurface:1,umCubicListed:1,...free}),
      keyed("advectCubic",{umCellWidth:1,...UNIFORM_DETAIL_H_STORE,...free}),
      Promise.all([compile("advectWalls",{}),compile("advectDeferred",{umMergedTiles:1,...UNIFORM_DETAIL_MERGED_SLOTS,...twinned}),compile("advectDeferred",{...regular,...UNIFORM_DETAIL_H_STORE,...twinned})]).then(deferred=>{this.deferredPipelines.push(...deferred);}),
      ...(this.solid?[Promise.all([compile("advectDeferred",{umMergedTiles:1,...UNIFORM_DETAIL_MERGED_SLOTS,...free}),compile("advectDeferred",{...regular,...UNIFORM_DETAIL_H_STORE,...free})]).then(twins=>{this.freeDeferredPipelines.push(...twins);})]:[]),
      compile("advectDeferred",{umMergedTiles:1,...UNIFORM_DETAIL_MERGED_SLOTS,umFastSurface:1,...free}).then(fast=>{this.fastDeferredPipeline=fast;}),
      keyed("advectOwners",{...regular,umCertifiedJobs:1,umClaimWord:0,...UNIFORM_DETAIL_H_STORE}),
    ]);
  }
  private twin(create:(solid:Record<string,number>)=>Promise<GPUComputePipeline>,options?:{solidsOnly?:boolean;entry?:string}):Promise<GPUComputePipeline>{return uniformMixedSolidPipeline(this.solid,create,options);}
  private variant(pipeline:GPUComputePipeline):GPUComputePipeline{return uniformDetailPick(this.solid?.select(pipeline)??pipeline);}
  encode(encoder:GPUCommandEncoder,group:UniformDetailGroup):void{
    const pipeline=this.pipelines.get("advect");if(!pipeline)throw new Error("Mixed surface stage is not initialized");
    encoder.clearBuffer(this.claims,0,this.claimBytes);
    const tiles=4*this.ownership.capacity.tiles;
    if(this.farGate)encoder.copyBufferToBuffer(this.farGate.buffer,4*this.farGate.word,this.claims,this.claimBytes,tiles);
    encoder.copyBufferToBuffer(this.ownership.speeds,tiles,this.claims,this.evidenceBytes-tiles,tiles);
    if(this.workControl){encoder.clearBuffer(this.claims,4*this.counterWord,4*(this.workCounterWords+5*this.workControl.regions));this.workControl.encode(encoder,"surface");
     encoder.copyBufferToBuffer(this.workControl.revisions.buffer,this.workControl.revisions.liveOffset,this.claims,4*this.revisionWord,4*this.workControl.regions);
     encoder.copyBufferToBuffer(this.workControl.surface,4*this.workControl.modelWordOffset,this.claims,4*this.modelWord,32*this.workControl.regions);
     encoder.copyBufferToBuffer(this.workControl.surface,4*this.workControl.velocityWordOffset,this.claims,4*this.velocityWord,24*this.workControl.regions);
     encoder.copyBufferToBuffer(this.workControl.surface,4*this.workControl.supportWordOffset,this.claims,4*this.supportWord,256*UNIFORM_SURFACE_SUPPORT_WORDS*this.workControl.regions);}
    const pass=encoder.beginComputePass(SURFACE_PASS);pass.setBindGroup(0,this.ownership.bindGroup);pass.setBindGroup(1,group.group);if(this.solid)pass.setBindGroup(2,this.solid.tileGroup);pass.setBindGroup(this.solid?3:2,this.ownership.hangingGroup);
    pass.setPipeline(this.variant(this.pipelines.get("wallReach")!));pass.dispatchWorkgroups(this.wallGroups);
    // The closed/clear evidence only exists for a scene holding solids.
    if(this.solid?.present&&!this.solid.coarse){
      const closed=this.solid.coarse?this.tileLaneGroups:this.tileGroups,clear=this.tileLaneGroups;
      pass.setPipeline(this.variant(this.pipelines.get("solidClosed")!));pass.dispatchWorkgroups(closed[0],closed[1]);
      pass.setPipeline(this.variant(this.pipelines.get("solidClear")!));pass.dispatchWorkgroups(clear[0],clear[1]);
    }
    // Buffered direct grids; the claim loops consume the full current lists.
    const grid=Math.min(UNIFORM_MIXED_CLAIMED_GRID,this.ownership.capacity.tiles);
    this.ownership.dispatchBuffered(pass,this.variant(this.pipelines.get("advectOwners")!),"fine",grid);
    this.ownership.dispatchBuffered(pass,this.variant(this.pipelines.get("advectFine")!),"fine",grid);
    pass.setPipeline(this.variant(this.pipelines.get("advectCubic")!));pass.dispatchWorkgroups(this.cubicGrid);
    this.ownership.dispatchBuffered(pass,this.variant(pipeline),"coarseMerged",grid);
    pass.setPipeline(this.variant(this.deferredPipelines[0]!));pass.dispatchWorkgroups(this.deferredWallGroups);
    pass.end();
    const deferred=encoder.beginComputePass(DEFERRED_PASS);deferred.setBindGroup(0,this.ownership.bindGroup);deferred.setBindGroup(1,group.group);
    if(this.solid)deferred.setBindGroup(2,this.solid.tileGroup);deferred.setBindGroup(this.solid?3:2,this.ownership.hangingGroup);
    // [merged, regular h]: each strides the entries its own launch listed,
    // and no h launch ran without h-tile capacity.
    for(let i=1,n=this.ownership.coarseOnly?2:3;i<n&&i<this.deferredPipelines.length;i++){
      if(i===1&&this.fastDeferredPipeline){deferred.setPipeline(this.variant(this.fastDeferredPipeline));deferred.dispatchWorkgroups(this.deferredGrid);}
      const twin=this.freeDeferredPipelines[i-1];if(twin){deferred.setPipeline(this.variant(twin));deferred.dispatchWorkgroups(this.deferredGrid);}
      deferred.setPipeline(this.variant(this.deferredPipelines[i]!));deferred.dispatchWorkgroups(this.deferredGrid);
    }
    deferred.end();
    if(this.workControl)encoder.copyBufferToBuffer(this.claims,4*this.revisionWord,this.workControl.revisions.buffer,this.workControl.revisions.scratchOffset,4*this.workControl.regions);
  }
  destroy():void{this.claims.destroy();}
}
