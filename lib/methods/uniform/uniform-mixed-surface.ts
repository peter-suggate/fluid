import { NARROW_BAND_TRACE_LIMIT, narrowBandTraceWGSL } from "./uniform-narrow-band-advection.wgsl";
import { uniformPreparedSurfaceSamplingWGSL } from "./uniform-prepared-surface.wgsl";
import { uniformDetailBindLayout, uniformDetailExtent, uniformDetailModule, uniformDetailPipeline, uniformDetailPick, uniformDetailGroup, type UniformDetailGroup } from "./uniform-detail-fields";
import {UNIFORM_DETAIL_4H_LOAD,UNIFORM_DETAIL_RING_4H_LOAD} from "../../core/uniform-detail-abi";
import {UNIFORM_PARAMS_BYTES,uniformMixedSourceWGSL} from "./uniform-mixed-source.wgsl";
import type { UniformMixedOwnership } from "./uniform-mixed-ownership";
import { UNIFORM_MIXED_COUNTED, uniformMixedCertifiedEntriesWGSL, uniformMixedTopologyWGSL } from "./uniform-mixed-topology.wgsl";
import { uniformMixedVertexSamplingSource } from "./uniform-mixed-vertex-sampling.wgsl";
import { uniformMixedVelocitySamplingSource } from "./uniform-mixed-velocity-sampling.wgsl";
import { UNIFORM_MIXED_CLAIMED_GRID, UNIFORM_MIXED_CLAIM_WORDS, uniformMixedClaimedEntriesWGSL, uniformMixedFaceAddressWGSL } from "./uniform-mixed-face-dispatch.wgsl";
import { uniformMixedSolidPipeline, uniformMixedSolidWGSL, uniformMixedVertexBuriedWGSL, type UniformMixedSolid } from "./uniform-mixed-solid.wgsl";

/** Scratch prefix of the surface claim buffer: job counters, six wall-plane
 * reach words, redistance's band count (UM_BAND_COUNT) and a pad word, then
 * per-tile wall reach. Persistent coarse
 * vertex travel follows this prefix and must survive its per-frame clear. */
const SURFACE_CLAIM_WORDS=UNIFORM_MIXED_CLAIM_WORDS+8;
/** Tiles of the largest domain plane: the stride of the per-tile reach words. */
const wallReachTiles=(d:readonly number[])=>{const t=d.map(n=>n/4);return Math.max(t[0]!*t[1]!,t[1]!*t[2]!,t[2]!*t[0]!);};
const surfaceClaimWords=(d:readonly number[])=>SURFACE_CLAIM_WORDS+6*wallReachTiles(d);
const wideVertices=(d:readonly number[])=>d.reduce((n,v)=>n*(v/4+1),1);
/** Word of the held-vertex corrections in the surface claims buffer: one per
 * 4h lattice vertex (x fastest), after the travel words (umHeldIndex). */
export const uniformMixedSurfaceHeldWord=(d:readonly number[])=>surfaceClaimWords(d)+wideVertices(d);
/** Workgroups of 256 lanes wallReach spends on one plane: its cells. */
const wallReachGroups=(d:readonly number[])=>Math.ceil(Math.max(d[0]!*d[1]!,d[1]!*d[2]!,d[2]!*d[0]!)/256);

export interface UniformMixedSurfaceFields {
  /** NB-FLIP sticky trajectory failure receipt; absent in Uniform Geometric. */
  narrowBandState?:GPUBuffer;
  phi: GPUTexture;
  outputPhi: GPUTexture;
  velocity: GPUTexture;
  coarseVelocity: GPUTexture;
  volume: GPUTexture;
  negative: GPUBuffer;
  departures: GPUTexture;
  /** h.xyz, dt; flags (bit 0 openTop, bit 1 preserve, bit 2 coarse travel cadence), cubic, drain, loop bound (=4). */
  params: GPUBuffer;
  /** One temporary surface-evidence word per tile, six wall words (plus two
   * pad), then with solids two words per tile (closed, clear); dead before transport. */
  evidence: GPUBufferBinding;
  /** UniformMixedHangingTaps.unitVelocity of `velocity`; with hanging taps only. */
  unitVelocity?: GPUTexture;
}

/** Native RK2 vertex/volume characteristics and surface rebuilding on canonical
 * vertices. Hanging values are sampled from their authority, never expanded.
 * Static solids port the native walked trace, buried-vertex freeze and the
 * embedded contact/air continuations; they live near fine owners only.
 * Optional phi experiments are rejected by the host. */
export class UniformMixedSurface {
  readonly allocatedBytes:number;
  private readonly resources: GPUBindGroupLayout;
  /** Job counters and wall reach, followed by accumulated characteristic
   * travel at 4h vertices, then the held vertices' corrections (umHeldIndex).
   * Only the scratch prefix is cleared per frame. */
  readonly claims: GPUBuffer;
  /** The held-vertex corrections, for the Dynamic census (umHeldIndex). */
  get held():{readonly buffer:GPUBuffer;readonly word:number}{return {buffer:this.claims,word:uniformMixedSurfaceHeldWord(this.ownership.capacity.lattice.dimensions)};}
  private readonly pipelines = new Map<string, GPUComputePipeline>();
  private readonly regularPipelines=new Map<string,GPUComputePipeline>();
  /** advect and redistance over the general h list, a lane per vertex. */
  private readonly finePipelines=new Map<string,GPUComputePipeline>();
  /** advectWalls and advectDeferred for the merged and regular dispatches, and the deferred grid. */
  private readonly deferredPipelines:GPUComputePipeline[]=[];
  /** Workgroups of the deferred advect launches: the list's bound (one word
   * per lattice vertex), capped where the GPU is saturated. */
  private readonly deferredGrid:number;
  constructor(private readonly device: GPUDevice, readonly ownership: UniformMixedOwnership,private readonly sourceParams?:GPUBuffer,private readonly solid?:UniformMixedSolid,private readonly hanging=false,resolved=false,private readonly narrowBand=false) {
    // Every sampler, cubic and evidence read assumes UniformMixedPhiResolve.
    if(!resolved)throw new Error("Mixed surface reads a resolved phi field (UniformMixedPhiResolve)");
    this.resources = uniformDetailBindLayout(device,{entries:[
      ...[0,2,3,4].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float" as const,viewDimension:"3d" as const}})),
      {binding:1,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"write-only",format:"r32float",viewDimension:"3d"}},
      {binding:5,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},
      {binding:6,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},
      {binding:7,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"write-only",format:"rgba32float",viewDimension:"3d"}},
      {binding:8,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},
      ...(sourceParams?[{binding:9,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform" as const}}]:[]),
      {binding:10,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},
      ...(hanging?[{binding:11,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float" as const,viewDimension:"3d" as const}}]:[]),
      {binding:12,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},
      ...(this.narrowBand?[{binding:13,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage" as const}}]:[]),
    ]});
    this.allocatedBytes=4*(surfaceClaimWords(ownership.capacity.lattice.dimensions)+2*wideVertices(ownership.capacity.lattice.dimensions));
    this.claims=device.createBuffer({label:"Uniform mixed surface job claims",size:this.allocatedBytes,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC});
    this.deferredGrid=Math.min(1024,Math.ceil(ownership.capacity.lattice.dimensions.reduce((a,n)=>a*(n+1),1)/64));
  }
  /** bind()'s evidence scratch from a 256-aligned offset: the tile words, then the deferred advect list. */
  get scratchBytes():number{
    const c=this.ownership.capacity;
    return Math.ceil((c.tiles*(this.solid?12:4)+32)/256)*256+16+4*c.lattice.dimensions.reduce((a,n)=>a*(n+1),1);
  }
  bind(f: UniformMixedSurfaceFields): UniformDetailGroup {
    const d=this.ownership.capacity.lattice.dimensions;
    for(const [i,t] of [f.phi,f.outputPhi,f.velocity,f.coarseVelocity,f.volume,f.departures].entries()){
      const size=d.map(n=>i<2?n+1:i===3?n/4+2:n);
      if(t.format!==([2,3,5].includes(i)?"rgba32float":"r32float") || uniformDetailExtent(t).some((n,a)=>n!==size[a]))
        throw new Error("Mixed surface requires native vertex/cell fields and a 4h velocity cache");
    }
    if(f.phi===f.outputPhi||f.departures===f.velocity||f.departures===f.coarseVelocity)throw new Error("Mixed surface outputs overlap inputs");
    if(this.hanging!==(f.unitVelocity!==undefined))throw new Error("Mixed surface hanging taps require their unit velocity texture");
    // Per-tile retirement evidence, the six wallReach plane words, then the
    // solid closed/clear tile words.
    const evidenceBytes=this.ownership.capacity.tiles*(this.solid?12:4)+32;
    // The deferred advect list follows, a separate binding: count, three
    // unused words, then at most one word per lattice vertex.
    const deferredOffset=Math.ceil(((f.evidence.offset??0)+evidenceBytes)/256)*256,deferredBytes=16+4*d.reduce((a,n)=>a*(n+1),1);
    if((f.evidence.size??f.evidence.buffer.size-(f.evidence.offset??0))<deferredOffset-(f.evidence.offset??0)+deferredBytes)throw new Error("Mixed surface evidence scratch is too small");
    if(d.some(n=>n>1023))throw new Error("Mixed surface deferred advect packs vertices in ten bits");
    const group=uniformDetailGroup(this.device,{layout:this.resources,entries:[
      ...[f.phi,f.outputPhi,f.velocity,f.coarseVelocity,f.volume].map((t,binding)=>({binding,resource:t})),
      {binding:5,resource:{buffer:f.negative}},{binding:6,resource:{buffer:f.params,size:32}},
      {binding:7,resource:f.departures},
      {binding:8,resource:{...f.evidence,size:evidenceBytes}},
      {binding:10,resource:{buffer:f.evidence.buffer,offset:deferredOffset,size:deferredBytes}},
      ...(this.sourceParams?[{binding:9,resource:{buffer:this.sourceParams,size:UNIFORM_PARAMS_BYTES}}]:[]),
      ...(f.unitVelocity?[{binding:11,resource:f.unitVelocity}]:[]),
      {binding:12,resource:{buffer:this.claims}},
      ...(this.narrowBand?[{binding:13,resource:{buffer:f.narrowBandState!}}]:[]),
    ]});
    return group;
  }
  async initialize(): Promise<void> {
    // regularTexture: interior unit-stencil samples load their eight taps
    // directly (no per-tap boundary branch, transverse clamp or select); the
    // skipped taps are exactly the zero-weight ones.
    const velocitySampling=uniformMixedVelocitySamplingSource(false,true,"velocity",this.hanging?(this.solid?3:2):undefined,this.hanging?"unitVelocity":undefined);
    const module=uniformDetailModule(this.device,{label:"Uniform mixed surface",code:uniformMixedCertifiedEntriesWGSL(uniformMixedClaimedEntriesWGSL(uniformMixedTopologyWGSL(this.ownership.capacity,0)+/* wgsl */`
@group(1) @binding(0) var phi:texture_3d<f32>;
@group(1) @binding(1) var outputPhi:texture_storage_3d<r32float,write>;
@group(1) @binding(2) var velocity:texture_3d<f32>;
@group(1) @binding(3) var coarseVelocity:texture_3d<f32>;
@group(1) @binding(4) var volume:texture_3d<f32>;
@group(1) @binding(5) var<storage,read> negative:array<f32>;
struct Params {hDt:vec4f,flags:vec4u}
@group(1) @binding(6) var<uniform> params:Params;
@group(1) @binding(7) var departures:texture_storage_3d<rgba32float,write>;
@group(1) @binding(8) var<storage,read_write> evidence:array<u32>;
// Advect vertices a released wall or solid can change: count, grid, list.
@group(1) @binding(10) var<storage,read_write> deferred:UMDeferred;
struct UMDeferred { header:array<atomic<u32>,4>, data:array<u32> }
@group(1) @binding(12) var<storage,read_write> umClaims:array<atomic<u32>>;
${this.narrowBand?"@group(1) @binding(13) var<storage,read_write> nbState:array<atomic<u32>>;":""}
${this.sourceParams?uniformMixedSourceWGSL(9):""}
${uniformMixedFaceAddressWGSL}
fn umLoadVertex(p:vec3u)->f32{return textureLoad(phi,vec3i(p),0).x;}
// A tile corner, and a 4h owner's origin cell: the base blocks
// (UNIFORM_DETAIL_4H_LOAD). The W loaders take the width that aligns p.
fn umLoadCorner(p:vec3u)->f32{return ${UNIFORM_DETAIL_4H_LOAD}textureLoad(phi,vec3i(p),0).x;}
fn umLoadVertexW(p:vec3u,width:u32)->f32{if(!umRegularFine&&width==4u){return umLoadCorner(p);}return umLoadVertex(p);}
fn umLoadVolumeW(p:vec3i,width:i32)->f32{if(width==4){return ${UNIFORM_DETAIL_4H_LOAD}textureLoad(volume,p,0).x;}return textureLoad(volume,p,0).x;}
// A wall owner's face texel (speed and release bits): a 4h owner's anchor is canonical.
fn umLoadWallTexel(anchor:vec3i,width:u32)->vec4f{if(width==4u){return ${UNIFORM_DETAIL_RING_4H_LOAD}textureLoad(velocity,anchor,0);}return textureLoad(velocity,anchor,0);}
${uniformMixedVertexSamplingSource("",true,undefined,"umLoadCorner")}
fn umLoadMixedFace(anchor:vec3i,axis:u32)->f32{
 if(anchor[axis]<0){return negative[umNegativeBoundaryIndex(vec3u(max(anchor,vec3i(0))),axis)];}
 return textureLoad(velocity,anchor,0)[axis];
}
fn umLoadCoarseFace(index:vec3i,axis:u32)->f32{return textureLoad(coarseVelocity,index+vec3i(1),0)[axis];}
${this.hanging?"@group(1) @binding(11) var unitVelocity:texture_3d<f32>;":""}
${velocitySampling}
${this.narrowBand?narrowBandTraceWGSL("umSampleVelocity"):""}
fn umSurfaceTrace(p:vec3f)->vec3f{
 ${this.narrowBand?`var k1=umSampleVelocity(p)/params.hDt.xyz;
 let steps=nbTraceSteps(k1,params.hDt.w);
 if(steps>${NARROW_BAND_TRACE_LIMIT}u){atomicStore(&nbState[3],1u);return p;}
 var q=p;let base=-params.hDt.w/f32(steps);var units=1u;
 for(var i=0u;i<steps;i+=units){
  if(i>0u){k1=umSampleVelocity(q)/params.hDt.xyz;}let span=nbTraceSpan(q,base,k1,steps-i);
  units=u32(span.w);q=clamp(span.xyz,vec3f(0),vec3f(UM_D));
 }
 return q;`:`let h=params.hDt.xyz;let dt=params.hDt.w;
 let mid=clamp(p-0.5*dt*umSampleVelocity(p)/h,vec3f(0),vec3f(UM_D));
 return clamp(p-dt*umSampleVelocity(mid)/h,vec3f(0),vec3f(UM_D));`}
}
${uniformMixedSolidWGSL(this.solid?2:undefined,this.solid?.coarse?.count)}
${this.solid?/* wgsl */`${uniformMixedVertexBuriedWGSL}
fn umOpenAt(q:vec3f)->f32{return umCellOpen(clamp(vec3i(floor(q)),vec3i(0),vec3i(UM_D)-vec3i(1)));}
// uvTrace: walk every crossed half-cell so no characteristic tunnels a wall.
struct UMWalk{q:vec3f,hit:bool}
fn umWalk(p:vec3f,end:vec3f)->UMWalk{
 let steps=max(1u,u32(ceil(2.0*max(abs(end.x-p.x),max(abs(end.y-p.y),abs(end.z-p.z))))));
 var previous=p;
 for(var s=1u;s<=steps;s++){let q=mix(p,end,f32(s)/f32(steps));if(umOpenAt(q)<=1e-5){return UMWalk(previous,true);}previous=q;}
 return UMWalk(end,false);
}
fn umTrace(p:vec3f)->vec3f{return umWalk(p,umSurfaceTrace(p)).q;}
// solidClosed/solidClear: no in-domain closed cell within four cells of the
// tile holding clamp(p) (27 tiles), a superset of the cells umBuried at a
// vertex or cubic tap and umEmbeddedContact read from any vertex of that tile.
const UM_SOLID_WORDS:u32=UM_TILES+8u;
fn umSolidClear(p:vec3i)->bool{return !umSolidEnabled()||evidence[UM_SOLID_WORDS+UM_TILES+umTileAt(vec3u(clamp(p,vec3i(0),vec3i(UM_D)-vec3i(1)))/4u)]!=0u;}
// uvBuried: phi at a vertex no incident owner has capacity at is not state.
fn umBuried(p:vec3f)->bool{return umVertexBuried(vec3i(p));}
fn umFluidFace(face:vec3i,axis:u32)->f32{
 if(umSolidValid(face)){return textureLoad(velocity,face,0)[axis];}
 var n=face;n[axis]+=1;
 if(umSolidValid(n)&&face[axis]==-1){return negative[umNegativeBoundaryIndex(vec3u(n),axis)];}
 return 0.0;
}
fn umContactReleased(face:vec3i,axis:u32)->bool{
 var cell=face;var bit=axis;if(face[axis]<0){cell[axis]=0;bit+=3u;}
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
  if(umSolidValid(solid)&&umCellOpen(solid)<=1e-5){
   for(var axis=0u;axis<3u;axis++){for(var side=-1;side<=1;side+=2){
    var fluid=solid;fluid[axis]+=side;if(umCellOpen(fluid)<=1e-5){continue;}
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
}
// uvEmbeddedContact: arriving/continued liquid and released air at a vertex
// incident to an embedded wall.
// Every (fluid, solid) pair is two incident cells across one of the vertex's
// planes, and its interior probe depends only on that direction (2*axis+side):
// the probes, a velocity sample and at most one trace each, run once per
// direction rather than once per pair.
fn umEmbeddedContact(p:vec3f,advected:f32)->f32{
 var arriving=1e20;var continued=1e20;var air=-1e20;var supported=advected>=0.0;
 let base=vec3i(p)-vec3i(1);var open=0u;var valid=0u;
 for(var k=0u;k<8u;k++){
  let cell=base+vec3i(umCorner(k,2u));
  if(umSolidValid(cell)){valid|=1u<<k;}if(umCellOpen(cell)>1e-5){open|=1u<<k;}
 }
 var directions=0u;
 for(var k=0u;k<8u;k++){
  if((open&(1u<<k))==0u){continue;}let fluid=base+vec3i(umCorner(k,2u));
  {let o=umOwnerAt(fluid);if(umLoadVolumeW(select(fluid,vec3i(umOrigin(o)),o.width!=0u),i32(o.width))>0.05){supported=true;}}
  for(var axis=0u;axis<3u;axis++){
   // The solid is the incident cell across the vertex on this axis.
   let across=k^(1u<<axis);if((valid&(1u<<across))==0u||(open&(1u<<across))!=0u){continue;}
   let upper=((k>>axis)&1u)==0u;let side=select(-1,1,upper);var solid=fluid;solid[axis]+=side;
   directions|=1u<<(2u*axis+select(0u,1u,upper));
   let face=select(solid,fluid,upper);
   let travel=params.hDt.w*(-f32(side)*umFluidFace(face,axis));
   if(umContactReleased(face,axis)&&travel>1e-4*params.hDt[axis]){air=max(air,travel);}
  }
 }
 for(var d=0u;d<6u;d++){
  if((directions&(1u<<d))==0u){continue;}
  let axis=d/2u;let side=select(-1.0,1.0,(d&1u)!=0u);var interior=p;interior[axis]-=side;
  let into=side*umSampleVelocity(interior)[axis]>1e-6;
  if(advected<0.0||into){
   let value=umSampleVertex(umTrace(interior));
   continued=min(continued,value);if(into){arriving=min(arriving,value);}
  }
 }
 var result=advected;
 if(arriving<1e20){result=select(arriving,min(result,arriving),supported);}else if(continued<1e20){result=select(continued,min(result,continued),supported);}
 return max(result,air);
}`:/* wgsl */`fn umTrace(p:vec3f)->vec3f{return umSurfaceTrace(p);}`}
fn umCatmull(t:f32)->vec4f{
 let t2=t*t;let t3=t2*t;return 0.5*vec4f(2.0*t2-t3-t,3.0*t3-5.0*t2+2.0,4.0*t2-3.0*t3+t,t3-t2);
}
// umSampleVertex(p) on the resolved field, keeping what umCubicPhi reuses:
// the owner cell's origin and width (its tile's width), and its eight corner
// values in umCorner order (the same loads, weights and D4 sum).
struct UMVertexCell {origin:vec3u,width:u32,corners:array<f32,8>,value:f32}
fn umVertexCell(p:vec3f)->UMVertexCell{
 let q=clamp(p,vec3f(0),vec3f(UM_D));var c:UMVertexCell;var t=vec3f(0);
 if(umRegularFine){c.origin=min(vec3u(floor(q)),UM_D-vec3u(1));c.width=1u;t=q-vec3f(c.origin);}
 else{let cell=vec3u(min(vec3i(floor(q)),vec3i(UM_D)-vec3i(1)));c.width=umTileWidth(umTileAt(cell/4u));c.origin=(cell/c.width)*c.width;t=(q-vec3f(c.origin))/f32(c.width);}
 var values:array<f32,8>;
 // One branch on the cell's width, outside its eight taps: a 4h cell's
 // corners are tile corners, read from the base block. A branch inside the
 // tap (one per load) doubled the merged redistance pass on Metal.
 if(!umRegularFine&&c.width==4u){
  for(var k=0u;k<8u;k++){
   let corner=umCorner(k,2u);let w=select(vec3f(1)-t,t,corner!=vec3u(0));
   c.corners[k]=umLoadCorner(c.origin+corner*4u);values[k]=c.corners[k]*w.x*w.y*w.z;
  }
 }else{
  for(var k=0u;k<8u;k++){
   let corner=umCorner(k,2u);let w=select(vec3f(1)-t,t,corner!=vec3u(0));
   c.corners[k]=umLoadVertex(c.origin+corner*c.width);values[k]=c.corners[k]*w.x*w.y*w.z;
  }
 }
 c.value=umVertexSum8(values);return c;
}
// The cubic at q over cell = umVertexCell(q): its owner (origin, width) and
// the eight inner taps (offsets 0..1) come from the cell, not reloaded.
fn umCubicPhi(q:vec3f,cell:UMVertexCell)->f32{
 let width=i32(cell.width);let base=vec3i(cell.origin);let fraction=(q-vec3f(base))/f32(width);
 let wx=umCatmull(fraction.x);let wy=umCatmull(fraction.y);let wz=umCatmull(fraction.z);
 // Resolved: every tap is a width-aligned vertex within one tile of the
 // owner's tile (4h taps are 4-aligned, stored everywhere), so it is stored
 // or resolved. The 64 taps then unroll with a constant bound.
 // Every clamped lattice vertex has an incident in-domain cell. With no
 // embedded solids it cannot be buried, at any owner width. Fold that case
 // before the 64 taps: the fine-tile clearance certificate alone excludes
 // coarse owners and leaves their per-tap domain tests in the solid-free twin.
 ${this.solid?"let tapsClear=!umSolidEnabled()||((umRegularFine||cell.width==1u)&&umSolidClear(base));":""}
 var value=0.0;var low=1e30;var high=-1e30;
 // As umVertexCell: the width branch is outside the 64 taps.
 ${(taps=>`if(!umRegularFine&&cell.width==4u){${taps("umLoadCorner")}}else{${taps("umLoadVertex")}}`)((load:string)=>/* wgsl */`
 for(var z=0u;z<4u;z++){var plane=0.0;
  for(var y=0u;y<4u;y++){var row=0.0;
   for(var x=0u;x<4u;x++){
    let offset=vec3i(vec3u(x,y,z))-vec3i(1);let vertex=clamp(base+offset*width,vec3i(0),vec3i(UM_D));
    // Cubic taps lie on lattice vertices, stored or resolved: direct loads.
    // The inner taps are the cell's corners.
    let inner=all(offset>=vec3i(0))&&all(offset<=vec3i(1));
    var s=0.0;if(inner){s=cell.corners[x-1u+2u*(y-1u)+4u*(z-1u)];}else{s=${load}(vec3u(vertex));}row+=wx[x]*s;
    ${this.solid?"if(!tapsClear&&wx[x]*wy[y]*wz[z]!=0.0&&umBuried(vec3f(vertex))){return cell.value;}":""}
    if(inner){low=min(low,s);high=max(high,s);}
   }plane+=wy[y]*row;
  }value+=wz[z]*plane;
 }`)}
 return clamp(value,low,high);
}
fn umDrain(q:vec3f,value:f32,width:u32)->f32{
 let h=f32(width)*min(params.hDt.x,min(params.hDt.y,params.hDt.z));if(value>=0.5*h){return value;}
 let centre=vec3i(floor(q/f32(width)+vec3f(0.5)))*i32(width);
 let low=max(vec3i(0),centre-vec3i(2*i32(width)));let high=min(vec3i(UM_D),centre+vec3i(2*i32(width)));
 // Deep liquid: the owner of the departure cell, when that cell is in the
 // box, intersects it and is one of the loop's own probes (its origin is a
 // width step of its tile), so test it before walking the box.
 let at=clamp(vec3i(floor(q)),vec3i(0),vec3i(UM_D)-vec3i(1));
 if(all(at>=low)&&all(at<high)){let w=i32(umTileWidth(umTileAt(vec3u(at)/4u)));if(umLoadVolumeW((at/w)*w,w)>0.05){return value;}}
 // Visit intersecting owners once per tile. This is exactly the native 4^3
 // evidence box at either uniform endpoint, and never skips a fine droplet.
 let first=low/4;let last=(high+vec3i(3))/4;
 for(var z=first.z;z<last.z;z++){for(var y=first.y;y<last.y;y++){for(var x=first.x;x<last.x;x++){
  let origin=vec3i(x,y,z)*4;let step=i32(umTileWidth(umTileAt(vec3u(vec3i(x,y,z)))));
  for(var dz=0;dz<4;dz+=step){for(var dy=0;dy<4;dy+=step){for(var dx=0;dx<4;dx+=step){
   let p=origin+vec3i(dx,dy,dz);
   if(all(p<high)&&all(p+vec3i(step)>low)&&umLoadVolumeW(p,step)>0.05){return value;}
  }}}
 }}}
 // The retreat is 0.5h per 1/30 s of simulated time, whatever the step.
 return min(value+15.0*params.hDt.w*h,0.5*h);
}
fn umWallContact(p:vec3f,value:f32,width:u32)->f32{
 if(params.hDt.w<=0.0){return value;}var interior=p;var contact=false;
 for(var axis=0u;axis<3u;axis++){for(var side=0u;side<2u;side++){
  let upper=side==1u;let inward=select(1.0,-1.0,upper);let plane=select(0.0,f32(UM_D[axis]),upper);
  if(abs(p[axis]-plane)>1e-5||(axis==1u&&upper&&(params.flags.x&1u)!=0u)){continue;}
  var probe=p;probe[axis]+=inward*f32(width);
  ${this.narrowBand?"if(inward*umSampleVelocity(probe)[axis]>=-1e-6){continue;}":""}
  if(value>=0.0&&inward*umSampleVelocity(probe)[axis]>=-1e-6){continue;}
  interior[axis]+=inward*f32(width);contact=true;
 }}
 if(!contact${this.solid?"||umOpenAt(interior)<=1e-5":""}){return value;}let continued=umSampleVertex(umTrace(interior));
 return select(value,min(value,continued),continued<0.0);
}
// The six plane reach words (UM_WALL_REACH) are final once wallReach, an
// earlier dispatch, is done. Every advect entry stages them per job, so its
// vertex lanes read workgroup memory instead of six atomics per vertex.
var<workgroup> umPlaneReach:array<f32,6>;
fn umStagePlaneReach(lane:u32){
 if(lane<6u){umPlaneReach[lane]=bitcast<f32>(atomicLoad(&umClaims[UM_WALL_REACH+lane]));}
 workgroupBarrier();
}
fn umReleasedWalls(p:vec3f,value:f32)->f32{
 var result=value;let h=params.hDt.xyz;let dt=params.hDt.w;
 for(var axis=0u;axis<3u;axis++){for(var side=0u;side<2u;side++){
  let upper=side==1u;let inward=select(1.0,-1.0,upper);let plane=select(0.0,f32(UM_D[axis]),upper);
  let ambient=axis==1u&&upper&&(params.flags.x&1u)!=0u;
  // wallReach stored this plane's largest contributing dt*away, zero when
  // no face on it is released or ambient. If even that cannot raise phi
  // here, none of the four probes can. The rounding margin keeps this a work
  // exclusion, never a modification of the wall continuation.
  let reach=umPlaneReach[2u*axis+side];
  if(reach<=0.0||reach-inward*(p[axis]-plane)*h[axis]+1e-5*max(h.x,max(h.y,h.z))<result){continue;}
  var whole=1e30;
  for(var k=0u;k<4u;k++){
   var probe=p;probe[(axis+1u)%3u]+=select(-1e-4,1e-4,(k&1u)!=0u);probe[(axis+2u)%3u]+=select(-1e-4,1e-4,(k&2u)!=0u);
   probe[axis]=plane+inward*1e-4;
   let owner=umOwnerAt(clamp(vec3i(floor(probe)),vec3i(0),vec3i(UM_D)-vec3i(1)));
   let origin=umOrigin(owner);var anchor=vec3i(origin);anchor[axis]+=i32(owner.width)-1;
   let wallTexel=umLoadWallTexel(anchor,owner.width);var speed=wallTexel[axis];var bit=axis;
   if(!upper){speed=negative[umNegativeBoundaryIndex(origin,axis)];bit+=3u;}
   let released=(u32(round(wallTexel.w))&(1u<<bit))!=0u;
   let away=inward*speed;
   // A wide owner's release is its centre pressure, and liquid it cannot
   // resolve above it (V under an air centre) lowers that pressure. One
   // such owner separates the wall only within half its width of the
   // surface; deeper, its lift alone would snap a submerged wall column to
   // air.
   let resolved=ambient||owner.width==1u||value>-0.5*f32(owner.width)*min(h.x,min(h.y,h.z));
   let gap=select(0.0,dt*away,(ambient||released)&&dt*away>1e-4*h[axis]*f32(owner.width));
   if(resolved&&gap>0.0){result=max(result,gap-inward*(p[axis]-plane)*h[axis]);}
   whole=min(whole,gap);
  }
  // Deeper, the wall separates where every owner around the vertex has
  // released, by the least of their gaps. A slab parallel to the wall and
  // thicker than half a wide owner has no vertex the rule above reaches:
  // the pressure released its owners and drained their V while phi stayed
  // on the wall, and each such owner (a pressure row with a half-open wall
  // face and no V) doubled its wall face speed against its opposite face
  // every step (the all-4h Figure 9 lid: 30 m/s wall faces at -70 kPa).
  if(whole>0.0){result=max(result,whole-inward*(p[axis]-plane)*h[axis]);}
 }}return result;
}
const UM_TRAVEL_BASE:u32=${surfaceClaimWords(this.ownership.capacity.lattice.dimensions)}u;
// A coarse rebuild has a spatial interpolation error even at u=0. Applying
// it every small timestep makes that error a persistent artificial motion.
// Accumulate characteristic travel in owner-cell units in the existing
// advection kernels; rebuild after a cell traversal, retaining the remainder.
// Fine vertices reset their 4h-aligned slots, so a later coarsening starts
// from the fine path's freshly maintained distance field.
fn umTravelIndex(p:vec3u)->u32{let q=p/4u;let d=UM_D/4u+vec3u(1);return UM_TRAVEL_BASE+q.x+d.x*(q.y+d.y*q.z);}
// A held wide vertex keeps a value that preserves its cells' zero set and is
// no longer a distance. For the Dynamic census' shape criterion, which
// measures distances, redistance stores here what a rebuild would have added
// to the vertex (umBandCandidate, redistanceBand); 0: the stored value is
// current. An h tile's own 4-aligned vertices are rebuilt every step.
fn umHeldIndex(p:vec3u)->u32{let d=UM_D/4u+vec3u(1);return umTravelIndex(p)+d.x*d.y*d.z;}
fn umRecordTravel(p:vec3f,q:vec3f,width:u32){
 if((params.flags.x&4u)==0u){return;}
 if(width>1u){let i=umTravelIndex(vec3u(p));atomicStore(&umClaims[i],bitcast<u32>(bitcast<f32>(atomicLoad(&umClaims[i]))+length(p-q)/f32(width)));}
 else if(all(vec3u(p)%4u==vec3u(0))){atomicStore(&umClaims[umTravelIndex(vec3u(p))],0u);atomicStore(&umClaims[umHeldIndex(vec3u(p))],0u);}
}
fn umAdvected(p:vec3f,width:u32)->f32{
 ${this.solid?/* wgsl */`let clear=umSolidClear(vec3i(p));if(!clear&&umBuried(p)){return umLoadVertex(vec3u(p));}
 let end=umSurfaceTrace(p);let walk=umWalk(p,end);let q=walk.q;`:"let q=umTrace(p);"}let cell=umVertexCell(q);var value=cell.value;umRecordTravel(p,q,width);
 let h=f32(width)*min(params.hDt.x,min(params.hDt.y,params.hDt.z));
 if(params.flags.y!=0u&&abs(value)<2.0*h){value=umCubicPhi(q,cell);}
 ${this.narrowBand?(this.solid?"value=umWallContact(p,value,width);if(walk.hit){value=umEmbeddedAir(p,end,value);}value=umReleasedWalls(p,value);":"value=umReleasedWalls(p,umWallContact(p,value,width));"):this.solid?/* wgsl */`value=umWallContact(p,value,width);
 if(!clear){value=umEmbeddedContact(p,value);}if(walk.hit){value=umEmbeddedAir(p,end,value);}
 value=umReleasedWalls(p,value);`:"value=umReleasedWalls(p,umWallContact(p,value,width));"}
 ${this.narrowBand?"":"if(params.flags.z!=0u){value=umDrain(q,value,width);}"}return ${this.sourceParams?"umSourceuvSourcePhi(p,value)":"value"};
}
// The first non-ambient domain plane through p (2*axis+side), or 6: the only
// vertices umWallContact can change. advectWalls owns them.
fn umWallPlane(p:vec3u)->u32{
 for(var k=0u;k<6u;k++){
  let axis=k/2u;let upper=(k&1u)==1u;
  if(p[axis]==select(0u,UM_D[axis],upper)&&!(axis==1u&&upper&&(params.flags.x&1u)!=0u)){return k;}
 }return 6u;
}
// Whether umReleasedWalls can change value at p: a plane's reach clears its
// own skip test. Accepted continuations only raise result, so testing the
// entry value probes a superset of its planes. A plane that passes on its
// whole-plane reach is retested on the reach of the tiles holding p's four
// probe cells (floor(p+-1e-4) on the in-plane axes, clamped as the probes
// are): a probe's accepted dt*away is at most its tile's reach, so a vertex
// failing both skips keeps its value. Only vertices above released faces
// moving off the wall defer, not every vertex any released face can reach.
fn umReleasedMayChange(p:vec3f,value:f32)->bool{
 let h=params.hDt.xyz;let slack=1e-5*max(h.x,max(h.y,h.z));
 for(var k=0u;k<6u;k++){
  let axis=k/2u;let upper=(k&1u)==1u;let inward=select(1.0,-1.0,upper);let plane=select(0.0,f32(UM_D[axis]),upper);
  let depth=inward*(p[axis]-plane)*h[axis];
  let reach=umPlaneReach[k];
  if(!(reach>0.0&&!(reach-depth+slack<value))){continue;}
  let u=(axis+1u)%3u;let v=(axis+2u)%3u;
  let lowU=u32(clamp(i32(floor(p[u]-1e-4)),0,i32(UM_D[u])-1))/4u;let highU=u32(clamp(i32(floor(p[u]+1e-4)),0,i32(UM_D[u])-1))/4u;
  let lowV=u32(clamp(i32(floor(p[v]-1e-4)),0,i32(UM_D[v])-1))/4u;let highV=u32(clamp(i32(floor(p[v]+1e-4)),0,i32(UM_D[v])-1))/4u;
  let base=UM_WALL_TILE_REACH+k*UM_WALL_PLANE_TILES;
  let local=max(max(atomicLoad(&umClaims[base+lowU+UM_T[u]*lowV]),atomicLoad(&umClaims[base+highU+UM_T[u]*lowV])),
   max(atomicLoad(&umClaims[base+lowU+UM_T[u]*highV]),atomicLoad(&umClaims[base+highU+UM_T[u]*highV])));
  let tileReach=bitcast<f32>(local);
  if(tileReach>0.0&&!(tileReach-depth+slack<value)){return true;}
 }return false;
}
// umAdvected off the domain walls, without the wall and solid continuations.
// A vertex any continuation can change goes to advectDeferred, which
// evaluates umAdvected itself; domain-wall vertices are advectWalls'. Keeping
// the continuations' sampler call sites out of this kernel is its cost.
fn umAdvectStore(vertex:vec3u,width:u32){
 if(params.hDt.w>0.0&&umWallPlane(vertex)<6u){return;}
 let p=vec3f(vertex);var defer=false;var value=0.0;
 ${this.solid?/* wgsl */`// Off a clear tile a vertex can be buried or embedded-contact. Both read
 // only its eight incident cells (a cut 4h owner's corner also its tiles'
 // capacity): a buried vertex keeps its value, umAdvected's first exit, and
 // umEmbeddedContact is the identity unless an in-domain incident cell is closed.
 ${this.narrowBand?"if(!umSolidClear(vec3i(p))&&umBuried(p)){textureStore(outputPhi,vec3i(vertex),vec4f(umLoadVertex(vertex)));return;}":` if(!umSolidClear(vec3i(p))){
  var open=false;
  for(var k=0u;k<8u;k++){
   let cell=vec3i(p)-vec3i(1)+vec3i(umCorner(k,2u));let cellOpen=umCellOpen(cell)>1e-5;
   open=open||cellOpen;defer=defer||(umSolidValid(cell)&&!cellOpen);
  }
  if(!open&&umBuried(p)){textureStore(outputPhi,vec3i(vertex),vec4f(umLoadVertex(vertex)));return;}
 }
`}
 if(!defer){let walk=umWalk(p,umSurfaceTrace(p));defer=walk.hit;let q=walk.q;`:"{let q=umTrace(p);"}
  let cell=umVertexCell(q);value=cell.value;
  let h=f32(width)*min(params.hDt.x,min(params.hDt.y,params.hDt.z));
  if(params.flags.y!=0u&&abs(value)<2.0*h){value=umCubicPhi(q,cell);}
  defer=defer||umReleasedMayChange(p,value);
  ${this.narrowBand?"":"if(params.flags.z!=0u){value=umDrain(q,value,width);}"}value=${this.sourceParams?"umSourceuvSourcePhi(p,value)":"value"};
  if(!defer){umRecordTravel(p,q,width);}
 }
 if(!defer){textureStore(outputPhi,vec3i(vertex),vec4f(value));return;}
 // Each canonical vertex is stored once: the list holds at most every vertex.
 let i=atomicAdd(&deferred.header[0],1u);
 deferred.data[i]=vertex.x|(vertex.y<<10u)|(vertex.z<<20u)|(select(firstTrailingBit(width),3u,umRegularFine)<<30u);
}
// The deferred vertices of the dispatch compiled with the same umRegularFine.
// A fixed grid (deferredGrid workgroups) strides over the list.
@compute @workgroup_size(64) fn advectDeferred(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 umStagePlaneReach(lane);
 let count=atomicLoad(&deferred.header[0]);
 for(var i=gid.x;i<count;i+=64u*groups.x){
  let word=deferred.data[i];let code=word>>30u;
  if((code==3u)!=umRegularFine){continue;}
  let vertex=vec3u(word&1023u,(word>>10u)&1023u,(word>>20u)&1023u);
  textureStore(outputPhi,vec3i(vertex),vec4f(umAdvected(vec3f(vertex),select(1u<<code,1u,code==3u))));
 }
}
// Canonical vertices of the non-ambient domain walls, each on the first
// plane through it, with its authority's width. One lane per plane vertex:
// the six planes' vertex grids concatenated (2*axis+side order). With dt>0
// umAdvectStore leaves them all to this entry, which lists them for
// advectDeferred (its umAdvected is this vertex's value): a plane of 4h
// tiles keeps one vertex in sixteen, so evaluating here idled most lanes of
// every group behind the few characteristics. With dt<=0 the owner passes
// store them (the wall continuations are then the identity).
var<workgroup> wallListed:atomic<u32>;
var<workgroup> wallBase:u32;
@compute @workgroup_size(64) fn advectWalls(@builtin(global_invocation_id) gid:vec3u,@builtin(local_invocation_index) lane:u32){
 if(lane==0u){atomicStore(&wallListed,0u);}
 workgroupBarrier();
 var i=gid.x+umDispatchX*64u*gid.y;var k=0u;
 for(;k<6u;k++){let a=k/2u;let area=(UM_D[(a+1u)%3u]+1u)*(UM_D[(a+2u)%3u]+1u);if(i<area){break;}i-=area;}
 var word=0xffffffffu;var slot=0u;
 if(k<6u&&params.hDt.w>0.0){
  let axis=k/2u;let u=(axis+1u)%3u;let v=(axis+2u)%3u;
  var vertex=vec3u(0);vertex[axis]=select(0u,UM_D[axis],(k&1u)==1u);vertex[u]=i%(UM_D[u]+1u);vertex[v]=i/(UM_D[u]+1u);
  if(umWallPlane(vertex)==k){
   var width=1u;
   if(!umRegularFine){let cell=min(vertex,UM_D-vec3u(1));width=select(1u,4u,umCoarseIncident(umTileAt(cell/4u),vertex-(cell/4u)*4u)!=0u);}
   if(all(vertex%width==vec3u(0))){
    slot=atomicAdd(&wallListed,1u);
    word=vertex.x|(vertex.y<<10u)|(vertex.z<<20u)|(select(firstTrailingBit(width),3u,umRegularFine)<<30u);
   }
  }
 }
 workgroupBarrier();
 // One list atomic per group.
 if(lane==0u){let n=atomicLoad(&wallListed);if(n>0u){wallBase=atomicAdd(&deferred.header[0],n);}}
 workgroupBarrier();
 if(word!=0xffffffffu){deferred.data[wallBase+slot]=word;}
}
// Six samples through one sampler call site: low then high per axis. The
// resolved sampler is eight loads, so its six samples unroll.
${this.solid?/* wgsl */`// A closed owner's corners are not all state (umVertexBuried): a closed 4h
// owner's, and a closed h cell's beside an h seam, whose buried corners hold
// the air sentinel. A search from the seam's 4h corner found that sentinel's
// zero a tenth of a cell away, sixteen cells under the surface.
fn umSampleClosed(q:vec3f)->bool{
 if(!umSolidEnabled()){return false;}
 let c=clamp(vec3i(floor(q)),vec3i(0),vec3i(UM_D)-vec3i(1));let t=umTileAt(vec3u(c)/4u);
 if(umTileWidth(t)==4u){return umTileOpen(t)<=1e-5;}
 return umCellOpen(c)<=1e-5;
}`:""}
fn umSurfaceGradient(p:vec3f,width:f32)->vec3f{
 var g=vec3f(0);var lowValue=0.0;
 for(var k=0u;k<6u;k++){
  let axis=k/2u;var delta=vec3f(0);delta[axis]=0.25*width;
  var low=clamp(p-delta,vec3f(0),vec3f(UM_D));var high=clamp(p+delta,vec3f(0),vec3f(UM_D));
  // A wide slope is one-sided beside a closed owner.
  ${this.solid?"if(width>1.0){if(umSampleClosed(low)){low=p;}if(umSampleClosed(high)){high=p;}}":""}
  let value=umSampleVertex(select(low,high,(k&1u)!=0u));
  if((k&1u)==0u){lowValue=value;}else{g[axis]=(value-lowValue)/max(high[axis]-low[axis],1e-6);}
 }return g;
}
// Each domain plane's (2*axis+side) largest dt*away over the faces
// umReleasedWalls can probe on it that pass its release test, as f32 bits
// (positive floats order as u32), in umClaims[UM_WALL_REACH+plane] (zeroed
// with the claims). Every probe resolves to the owner of one plane cell, so
// scanning every plane cell covers each probe exactly: a lane per plane cell
// aligned to its tile's width (a 4h tile's sixteen plane cells all resolve
// to its single owner), umWallReachGroups workgroups per plane. The maximum
// does not depend on the order the lanes combine in. The same maximum per
// plane tile (UM_WALL_TILE_REACH+plane*UM_WALL_PLANE_TILES+tu+UM_T[u]*tv)
// bounds the probes whose owners lie in that tile.
const UM_WALL_REACH:u32=${UNIFORM_MIXED_CLAIM_WORDS}u;
const UM_WALL_TILE_REACH:u32=${SURFACE_CLAIM_WORDS}u;
const UM_WALL_PLANE_TILES:u32=${wallReachTiles(this.ownership.capacity.lattice.dimensions)}u;
override umWallReachGroups:u32=1u;
var<workgroup> wallReachBits:atomic<u32>;
@compute @workgroup_size(256) fn wallReach(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 let plane=group.x/umWallReachGroups;let chunk=group.x%umWallReachGroups;
 let axis=plane/2u;let side=plane%2u;let upper=side==1u;let inward=select(1.0,-1.0,upper);
 let u=(axis+1u)%3u;let v=(axis+2u)%3u;let h=params.hDt.xyz;let dt=params.hDt.w;
 let ambient=axis==1u&&upper&&(params.flags.x&1u)!=0u;
 if(lane==0u){atomicStore(&wallReachBits,0u);}
 if(group.x==0u&&lane==0u){atomicStore(&deferred.header[0],0u);}
 workgroupBarrier();
 let i=chunk*256u+lane;
 if(plane<6u&&i<UM_D[u]*UM_D[v]){
  var cell=vec3i(0);cell[axis]=select(0,i32(UM_D[axis])-1,upper);cell[u]=i32(i%UM_D[u]);cell[v]=i32(i/UM_D[u]);
  let width=i32(umTileWidth(umTileAt(vec3u(cell)/4u)));
  if(cell[u]%width==0&&cell[v]%width==0){
   let owner=umOwnerAt(cell);
   let origin=umOrigin(owner);var anchor=vec3i(origin);anchor[axis]+=i32(owner.width)-1;
   let wallTexel=umLoadWallTexel(anchor,owner.width);var speed=wallTexel[axis];var bit=axis;
   if(!upper){speed=negative[umNegativeBoundaryIndex(origin,axis)];bit+=3u;}
   let released=(u32(round(wallTexel.w))&(1u<<bit))!=0u;
   let away=inward*speed;
   if((ambient||released)&&dt*away>1e-4*h[axis]*f32(owner.width)){
    let bits=bitcast<u32>(dt*away);atomicMax(&wallReachBits,bits);
    atomicMax(&umClaims[UM_WALL_TILE_REACH+plane*UM_WALL_PLANE_TILES+u32(cell[u])/4u+UM_T[u]*(u32(cell[v])/4u)],bits);
   }
  }
 }
 workgroupBarrier();
 if(lane==0u&&plane<6u){let bits=atomicLoad(&wallReachBits);if(bits!=0u){atomicMax(&umClaims[UM_WALL_REACH+plane],bits);}}
}
${this.solid?/* wgsl */`
var<workgroup> solidClosedFlag:atomic<u32>;
@compute @workgroup_size(64) fn solidClosed(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 let tile=group.x+umDispatchX*group.y;if(tile>=UM_TILES){return;}
 if(lane==0u){atomicStore(&solidClosedFlag,0u);}workgroupBarrier();
 if(umCellOpen(vec3i(umTileCoord(tile)*4u+umCorner(lane,4u)))<=1e-5){atomicStore(&solidClosedFlag,1u);}
 workgroupBarrier();if(lane==0u){evidence[UM_SOLID_WORDS+tile]=atomicLoad(&solidClosedFlag);}
}
@compute @workgroup_size(64) fn solidClear(@builtin(global_invocation_id) gid:vec3u){
 let tile=gid.x+64u*umDispatchX*gid.y;if(tile>=UM_TILES){return;}
 let t=vec3i(umTileCoord(tile));var clear=1u;
 for(var k=0u;k<27u;k++){
  let n=t+vec3i(umCorner(k,3u))-vec3i(1);
  if(all(n>=vec3i(0))&&all(n<vec3i(UM_T))&&evidence[UM_SOLID_WORDS+umTileAt(vec3u(n))]!=0u){clear=0u;}
 }
 evidence[UM_SOLID_WORDS+UM_TILES+tile]=clear;
}`:""}
// Each tile's evidence: every stored vertex of its 5^3 closure is positive.
// retirementEvidence: one workgroup per h tile, a lane per closure vertex;
// retirementEvidenceCoarse: one lane per 4h tile, whose closure stores only
// its 8 corners (local%4==0). Together they cover the h/4h partition once.
fn umEvidenceVertex(tile:u32,local:vec3u)->bool{
 let vertex=umTileCoord(tile)*4u+local;
 // Resolved: an h tile's closure vertices (their tile's stencil holds a
 // unit tile) and a 4h tile's corners are umVertexValue's direct load.
 return umLoadVertex(vertex)>0.0;
}
var<workgroup> evidencePositive:atomic<u32>;
@compute @workgroup_size(128) fn retirementEvidence(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 let job=group.x+umDispatchX*group.y;if(job>=umCounts.x){return;}
 let tile=umTopology[UM_TILES+job];
 if(params.flags.z==0u){if(lane==0u){evidence[tile]=UM_EVIDENCE_FAR;}return;}
 if(lane==0u){atomicStore(&evidencePositive,1u);}workgroupBarrier();
 if(lane<125u&&!umEvidenceVertex(tile,umCorner(lane,5u))){atomicStore(&evidencePositive,0u);}
 workgroupBarrier();if(lane==0u){evidence[tile]=select(0u,UM_EVIDENCE_FAR,atomicLoad(&evidencePositive)!=0u);}
}
@compute @workgroup_size(64) fn retirementEvidenceCoarse(@builtin(global_invocation_id) gid:vec3u){
 let job=gid.x+umDispatchX*64u*gid.y;if(job>=umCounts.y){return;}
 let tile=umTopology[UM_TILES+umCounts.x+job];
 if(params.flags.z==0u){evidence[tile]=UM_EVIDENCE_FAR;return;}
 var positive=true;
 for(var k=0u;k<8u;k++){positive=positive&&umEvidenceVertex(tile,umCorner(k,2u)*4u);}
 evidence[tile]=select(0u,UM_EVIDENCE_FAR,positive);
}
// evidenceDistance turns the positive flags into the Chebyshev distance in
// tiles to the nearest tile that may hold a zero (0: this one), capped at
// UM_EVIDENCE_FAR: d=min_j max(|i-j|,d_j) along x, then y, then z, one
// workgroup per line in place.
const UM_EVIDENCE_FAR=15u;
override umEvidenceAxis:u32=0u;
var<workgroup> evidenceLine:array<u32,256>;
@compute @workgroup_size(64) fn evidenceDistance(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 let axis=umEvidenceAxis;let u=(axis+1u)%3u;let v=(axis+2u)%3u;
 let line=group.x+umDispatchX*group.y;if(line>=UM_T[u]*UM_T[v]){return;}
 var start=vec3u(0);start[u]=line%UM_T[u];start[v]=line/UM_T[u];let n=UM_T[axis];
 for(var i=lane;i<n;i+=64u){var q=start;q[axis]=i;evidenceLine[i]=evidence[umTileAt(q)];}
 workgroupBarrier();
 for(var i=lane;i<n;i+=64u){
  var best=evidenceLine[i];
  for(var j=max(i,UM_EVIDENCE_FAR)-UM_EVIDENCE_FAR;j<min(n,i+UM_EVIDENCE_FAR+1u);j++){best=min(best,max(select(j-i,i-j,i>j),evidenceLine[j]));}
  var q=start;q[axis]=i;evidence[umTileAt(q)]=best;
 }
}
// A failed Newton search cannot certify distance. Retire a drained positive
// plateau only after checking all owner polynomials meeting the physical band.
// Checking their corners is conservative at a clipped coarse-cell boundary.
// Owner corners and unit-tile closure vertices are umVertexValue's direct
// load on the resolved field (as umEvidenceVertex).
fn umNoNearbySurface(p:vec3f,band:f32)->bool{
 let reach=vec3i(ceil(vec3f(band)/params.hDt.xyz));
 let low=max(vec3i(0),vec3i(p)-reach);let high=min(vec3i(UM_D),vec3i(p)+reach);
 let first=max(vec3i(0),(low-vec3i(1))/4);let last=min(vec3i(UM_T)-1,high/4);
 // Every tile of the box lies within its Chebyshev radius of p's tile, and
 // every in-domain tile within inner of it lies wholly inside the box, where
 // an evidence-free tile answers false below.
 let own=clamp(vec3i(p)/4,vec3i(0),vec3i(UM_T)-1);let radius=max(own-first,last-own);
 let reachLow=select((4*own-low)/4,vec3i(64),low==vec3i(0));let reachHigh=select((high-4*own-vec3i(4))/4,vec3i(64),high==vec3i(UM_D));
 let inner=min(reachLow,reachHigh);let nearest=i32(evidence[umTileAt(vec3u(own))]);
 if(nearest>max(radius.x,max(radius.y,radius.z))){return true;}
 if(nearest<=min(inner.x,min(inner.y,inner.z))){return false;}
 for(var z=first.z;z<=last.z;z++){for(var y=first.y;y<=last.y;y++){for(var x=first.x;x<=last.x;x++){
  let tile=vec3i(x,y,z);let width=i32(umTileWidth(umTileAt(vec3u(tile))));
  // An all-positive tile cannot contain a zero of any of its reconstructed
  // owner polynomials. Only clipped tiles with evidence need the exact scan.
  // A tile d from the nearest evidence-free one has d-1 positive successors.
  let distance=i32(evidence[umTileAt(vec3u(tile))]);if(distance!=0){x+=distance-1;continue;}
  if(all(tile*4>=low)&&all(tile*4+vec3i(4)<=high)){return false;}
  if(width==1){
   // Unit owners' in-box corners are the tile closure's in-box vertices.
   let lo=max(tile*4,low);let hi=min(tile*4+vec3i(4),high);
   for(var vz=lo.z;vz<=hi.z;vz++){for(var vy=lo.y;vy<=hi.y;vy++){for(var vx=lo.x;vx<=hi.x;vx++){
    if(!(umLoadVertex(vec3u(vec3i(vx,vy,vz)))>0.0)){return false;}
   }}}
   continue;
  }
  for(var dz=0;dz<4;dz+=width){for(var dy=0;dy<4;dy+=width){for(var dx=0;dx<4;dx+=width){
   let origin=tile*4+vec3i(dx,dy,dz);
   if(any(origin>high)||any(origin+vec3i(width)<low)){continue;}
   for(var k=0u;k<select(umCounts.w,8u,umRegularFine);k++){
    if(!(umLoadVertexW(vec3u(origin+vec3i(umCorner(k,2u))*width),u32(width))>0.0)){return false;}
   }
  }}}
 }}}return true;
}
// Whether a rebuild searches from p: a band vertex off zero, not buried.
// Preserve (CM11b sec. 3.4: "do not modify phi values of grid points next to
// the surface in order to avoid moving it"): an h vertex with a face
// neighbour of opposite sign keeps its advected value. Redistancing
// projects onto the trilinear zero set, which lies inside a convex body, so
// rebuilding these vertices shrinks every drop and sheet rim a little on
// every step, at rest or in flight; the thin-liquid ladders lose r = 2 drops
// to half their volume in 15 steps. Wide vertices keep the native rule.
fn umPreserved(p:vec3f,initial:f32,width:u32)->bool{
 if((params.flags.x&2u)==0u||width!=1u){return false;}
 for(var k=0u;k<6u;k++){let axis=k/2u;var q=vec3i(p);q[axis]+=select(-1,1,(k&1u)!=0u);
  if(q[axis]<0||q[axis]>i32(UM_D[axis])){continue;}
  if(umLoadVertex(vec3u(q))*initial<0.0){return true;}}
 return false;
}
// A wide vertex that defines the zero set: a corner of a cut wide cell, one of
// the eight cells at its own stride it touches holding a corner of the other
// sign (or one on zero). The same-sign corners of a cut cell shape the
// trilinear zero set inside it as the crossing ones do: held by its edge
// crossings alone, a still sphere's far corners were rebuilt against that
// zero set and moved it (a corner phi by 2 % at the first rebuild). The
// travel gate holds these corners alone.
// Every other wide band vertex is a distance to the surface they define and
// is rebuilt each step, as an h vertex off the surface is: held, a still
// seam's wide vertices were never rebuilt at all, and the hanging vertices
// of the seam plane interpolated values that only advection had ever moved.
// A buried neighbour is not state (umBuried) and certifies nothing: beside
// one (a face neighbour) the vertex is held whatever sign the sentinel there
// has, and a buried diagonal corner cuts no cell. Read by its
// sign, the air sentinel held the liquid vertices of a shoreline and let
// the air ones search a field with that sentinel in it (5 cells of error
// one tile above a 1:1 shore, and a churned pond there left its rest).
fn umWideCrosses(q:vec3i,face:bool,initial:f32,width:u32)->bool{
 if(any(q<vec3i(0))||any(q>vec3i(UM_D))){return false;}
 ${this.solid?"if(umBuried(vec3f(q))){return face;}":""}
 return umLoadVertexW(vec3u(q),width)*initial<=0.0;
}
// The six face neighbours first: most held vertices end a crossing edge, and
// the order of an any() is free.
fn umWideAdjacent(p:vec3f,initial:f32,width:u32)->bool{
 for(var k=0u;k<6u;k++){var d=vec3i(0);d[k/2u]=select(-1,1,(k&1u)!=0u);
  if(umWideCrosses(vec3i(p)+d*i32(width),true,initial,width)){return true;}}
 for(var k=0u;k<27u;k++){let d=vec3i(vec3u(k%3u,(k/3u)%3u,k/9u))-vec3i(1);
  if(abs(d.x)+abs(d.y)+abs(d.z)<2){continue;}
  if(umWideCrosses(vec3i(p)+d*i32(width),false,initial,width)){return true;}}
 return false;
}
// 0: the vertex keeps its value; 1: it is rebuilt; 2: the travel gate holds it.
fn umBandState(p:vec3f,initial:f32,width:u32)->u32{
 // The band first: most vertices are off it, and the gate and the preserve
 // rule below load the vertex's neighbours.
 let h=params.hDt.xyz;let band=4.0*f32(width)*max(h.x,max(h.y,h.z));
 if(!(abs(initial)>1e-8&&abs(initial)<band)){return 0u;}
 if((params.flags.x&4u)!=0u&&width>1u&&bitcast<f32>(atomicLoad(&umClaims[umTravelIndex(vec3u(p))]))<1.0){
  // The existing evidence distance is in tile units. Every tile incident
  // to p lies within one tile of clamp(p-1)'s tile. Keep far-air rebuilding
  // and retirement unconditional: residency relies on those certificates.
  // The evidence before the neighbours: it is one load, and two tiles into
  // the air it spares the 26.
  let tile=umTileAt(vec3u(clamp(vec3i(p)-vec3i(1),vec3i(0),vec3i(UM_D)-1))/4u);
  if((params.flags.z==0u||evidence[tile]<=1u)&&umWideAdjacent(p,initial,width)){return 2u;}
 }
 if(umPreserved(p,initial,width)){return 0u;}
 return ${this.solid?"select(1u,0u,umBuried(p))":"1u"};
}
fn umRebuildBand(p:vec3f,initial:f32,width:u32)->bool{return umBandState(p,initial,width)==1u;}
// The census reads a wide vertex's distance only as a corner of an h tile.
fn umHeldRead(p:vec3u)->bool{
 for(var k=0u;k<8u;k++){
  let t=vec3i(p/4u)-vec3i(umCorner(k,2u));
  if(all(t>=vec3i(0))&&all(t<vec3i(UM_T))&&umTileWidth(umTileAt(vec3u(t)))==1u){return true;}
 }
 return false;
}
// A wide vertex past the band beside a band vertex holds a stale distance:
// advection copies far air (or deep liquid) values toward a surface that
// arrives under them, and the band test above never rebuilds them. One 4h
// cell from the surface, that kink sits inside the cubic taps, the Newton
// gradient and the volume shift's slope (long dam: 42h beside 4h, detaching
// phi-only wall sheets). It is searched from the band value; a miss keeps
// the band value. Unit vertices keep the native band.
fn umStaleWide(p:vec3u,initial:f32,width:u32)->bool{
 let h=params.hDt.xyz;let band=4.0*f32(width)*max(h.x,max(h.y,h.z));
 if(width==1u||!(abs(initial)>=band)${this.solid?"||umBuried(vec3f(p))":""}){return false;}
 for(var k=0u;k<6u;k++){
  let axis=k/2u;var q=vec3i(p);q[axis]+=select(-1,1,(k&1u)!=0u)*i32(width);
  if(q[axis]<0||q[axis]>i32(UM_D[axis])){continue;}
  // Width-aligned taps are stored or resolved (umCubicPhi).
  if(abs(umLoadVertexW(vec3u(q),width))<band){return true;}
 }
 return false;
}
${[["umRebuildSearch","",true],["umPreparedSearch","",false]].map(([name,param,global])=>/* wgsl */`
fn ${name}(p:vec3f,initial:f32,width:u32${param})->f32{
 let h=params.hDt.xyz;let w=f32(width);let band=4.0*w*max(h.x,max(h.y,h.z));
 // umStaleWide vertices start (and on a miss stay) at the band value.
 let start=select(initial,sign(initial)*min(abs(initial),band),width>1u);
 var value=start;
 ${global?/* wgsl */`// phi(q) is carried across iterations: the sampler is a pure function of
 // q, so each accepted step reuses the value that justified it.
 // A trilinear zero set chords inside a curved surface by O(w^2 kappa), so
 // projecting onto it erodes a 4h rim every frame (about 0.1-0.2 cells at
 // 4h, 1/16 of that at h). Wide vertices project onto the cubic interface
 // that the advection carries, but only where the point's whole tile is
 // that width: across an h seam the cubic kept a phi-only one-cell sheet
 // running ahead of the long-dam toe.
 let cubic=!umRegularFine&&params.flags.y!=0u&&width>1u;`:""}
 var q=p;var phiQ=start;
 for(var i=0u;i<umCounts.w;i++){
  let g=${global?"umSurfaceGradient(q,w)":"umPreparedGradient(q)"};let norm=dot(g/h,g/h);if(norm<1e-16){break;}
  let next=clamp(q-clamp(phiQ*g/(h*h*norm),vec3f(-2.0*w),vec3f(2.0*w)),max(vec3f(0),p-vec3f(4.0*w)),min(vec3f(UM_D),p+vec3f(4.0*w)));
  var phiNext=0.0;
  ${global?/* wgsl */`// The cell's width is next's owner width (its tile's).
  let cell=umVertexCell(next);var wide=cubic&&cell.width==width;
  if(wide){wide=umTileMinimumWidth(umTileAt(cell.origin/4u))==width;}
  if(wide){phiNext=umCubicPhi(next,cell);}else{phiNext=cell.value;}`:"phiNext=umPreparedSample(next);"}
  if(abs(phiNext)>=abs(phiQ)){break;}q=next;phiQ=phiNext;
  // A tenth of the acceptance tolerance below: one more step moves q by less.
  if(abs(phiQ)<=0.0005*w*min(h.x,min(h.y,h.z))){break;}
 }
 // Acceptance is continuous in the residual: past tol it blends toward the
 // unsearched value, reached at 2 tol. A hard cut let mirror-image searches
 // that end an ulp either side of tol (Newton alternating across a trilinear
 // kink) keep and replace a vertex respectively: 0.035 cells of asymmetry.
 let miss=clamp(abs(phiQ)/(0.005*w*min(h.x,min(h.y,h.z)))-1.0,0.0,1.0);
 // q stops within the tolerance of the surface, not on it: the distance is
 // |p-q| plus what phi still reads at q. Without it every rebuilt vertex
 // carries its own stop residual (up to 5e-4 w h), and a resting h surface
 // grows a 2h sawtooth from the rows that stop on different iterates.
 if(miss<1.0){value=mix(sign(initial)*length((p-q)*h)+phiQ,start,miss);}
 else if(params.flags.z!=0u&&initial>0.0&&value<band&&umNoNearbySurface(p,band)){value=band;}
 return value;
}`).join("\n")}
// An h tile job's vertices (advectFine, redistanceFine): a lane per positive
// vertex (local 1..4, its cells' upper corners); a tile on a negative domain
// wall also owns its wall vertices, in two more rounds. The tile owns a
// vertex exactly when no incident tile is 4h: a 4h owner is wider, and among
// h owners the lowest index is clamp(p-1)'s, in this tile. wide: the 4h
// tiles among tile+{0,1}^3, the only tiles incident to its vertices.
const UM_NO_VERTEX=vec3u(0xffffffffu);

fn umFineVertex(tile:vec3u,lane:u32,round:u32,wide:u32)->vec3u{
 var local=umCorner(lane,4u)+vec3u(1);
 if(round>0u){let j=lane+64u*(round-1u);if(j>=125u){return UM_NO_VERTEX;}local=umCorner(j,5u);if(all(local!=vec3u(0))){return UM_NO_VERTEX;}}
 let vertex=tile*4u+local;
 if(any((local==vec3u(0))&(vertex!=vec3u(0)))){return UM_NO_VERTEX;}
 for(var k=1u;k<8u;k++){if((wide&(1u<<k))!=0u&&all((umCorner(k,2u)==vec3u(0))|(local==vec3u(4u)))){return UM_NO_VERTEX;}}
 return vertex;
}
fn umFineRounds(tile:vec3u)->u32{return select(1u,3u,any(tile==vec3u(0)));}
@compute @workgroup_size(64) fn advectFine(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 umStagePlaneReach(lane);
 let owner=umTileJobOwner(group);if(owner.width==0u){return;}
 let tile=umTileCoord(owner.tile);let wide=umFineWideMask(umTileAt(tile));
 for(var round=0u;round<umFineRounds(tile);round++){
  let vertex=umFineVertex(tile,lane,round,wide);
  if(vertex.x!=UM_NO_VERTEX.x){umAdvectStore(vertex,1u);}
 }
}
// Every fine Newton search reads the same prepared phi epoch. Prepare fine
// values and interpolated coarse samples once, instead of materializing an
// overlapping 14-cubed window in each searching workgroup. The dead deferred
// list owns this storage until all fine readers finish; coarse redistance
// may then reuse it as a list. Eight producer corners are the only shared
// field storage. Data words have a unique writer and need no atomics.
var<workgroup> umPrepareCorners:array<f32,8>;
fn umPreparedIndex(p:vec3u)->u32{return p.x+(UM_D.x+1u)*(p.y+(UM_D.y+1u)*p.z);}
@compute @workgroup_size(64) fn prepareSurface(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 let tile=group.x+umDispatchX*group.y;
 if(tile>=UM_TILES){return;}
 let fine=umTileWidth(tile)==1u;if(!fine&&(umTileStencil(tile).y&1u)==0u){return;}
 let origin=umTileCoord(tile)*4u;
 if(!fine&&lane<8u){umPrepareCorners[lane]=umLoadCorner(origin+umCorner(lane,2u)*4u);}
 workgroupBarrier();
 for(var i=lane;i<125u;i+=64u){
  let local=umCorner(i,5u);let p=origin+local;
  // The positive cell owns a shared vertex, except at the upper wall.
  if(any((local==vec3u(4))&(p!=UM_D))){continue;}
  if(fine){deferred.data[umPreparedIndex(p)]=bitcast<u32>(umLoadVertex(p));continue;}
  let t=vec3f(local)/4.0;var values:array<f32,8>;
  for(var k=0u;k<8u;k++){
   let corner=umCorner(k,2u);let w=select(vec3f(1)-t,t,corner!=vec3u(0));
   values[k]=umPrepareCorners[k]*w.x*w.y*w.z;
  }
  deferred.data[umPreparedIndex(p)]=bitcast<u32>(umVertexSum8(values));
 }
}
${uniformPreparedSurfaceSamplingWGSL}
@compute @workgroup_size(64) fn redistanceFine(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 let owner=umTileJobOwner(group);if(owner.width==0u){return;}
 let tile=umTileCoord(owner.tile);let wide=umFineWideMask(owner.tile);
 for(var round=0u;round<umFineRounds(tile);round++){
  let vertex=umFineVertex(tile,lane,round,wide);if(vertex.x==UM_NO_VERTEX.x){continue;}
  let initial=umLoadVertex(vertex);var value=initial;
  if(umRebuildBand(vec3f(vertex),initial,1u)){value=umPreparedSearch(vec3f(vertex),initial,1u);}
  textureStore(outputPhi,vec3i(vertex),vec4f(value));
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
 var owner=UMOwner();var first=7u;var tileVertex=vec3u(0);
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
  if(any(umOrigin(owner)==vec3u(0))){first=0u;}
 }
 let origin=umOrigin(owner);
 for(var k=first;k<8u;k++){
  let corner=umCorner(k,2u);
  if(packed&&!all((corner!=vec3u(0))|(origin==vec3u(0)))){continue;}
  let vertex=select(tileVertex,origin+corner*owner.width,packed);
  umAdvectStore(vertex,owner.width);
 }
}
// redistance's merged jobs: advect's vertices, eight seam 4h tiles or 64
// regular coarse owners per job. Off-band vertices keep their value. The
// band vertices are listed for redistanceBand, which searches them: a job's
// band vertices are few and uneven (a seam job holds eight tiles' corners,
// most of them another tile's; a pack's owners are mostly off the band), so
// searching them here idled the rest of the group's lanes behind each
// Newton chain. The list is advect's deferred list, dead since
// advectDeferred; its count is a claim word, zeroed with the claims.
const UM_BAND_COUNT:u32=UM_WALL_REACH+6u;
var<workgroup> umBandCount:atomic<u32>;
var<workgroup> umBandTotal:u32;
var<workgroup> umBandBase:u32;
var<workgroup> umBandList:array<u32,512>;
fn umBandCandidate(vertex:vec3u,width:u32){
 let initial=umLoadVertexW(vertex,width);
 let state=umBandState(vec3f(vertex),initial,width);
 // A held vertex the census reads (flags.x bit 3) is searched as well: its
 // value stays, and redistanceBand stores the difference. 1: pending.
 let publish=state==2u&&(params.flags.x&8u)!=0u&&umHeldRead(vertex)${this.solid?"&&!umBuried(vec3f(vertex))":""};
 if((params.flags.x&4u)!=0u){atomicStore(&umClaims[umHeldIndex(vertex)],select(0u,1u,publish));}
 if(state!=1u&&!umStaleWide(vertex,initial,width)){textureStore(outputPhi,vec3i(vertex),vec4f(initial));if(!publish){return;}}
 umBandList[atomicAdd(&umBandCount,1u)]=vertex.x|(vertex.y<<10u)|(vertex.z<<20u)|(firstTrailingBit(width)<<30u);
}
@compute @workgroup_size(64) fn redistance(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 if(lane==0u){atomicStore(&umBandCount,0u);}
 workgroupBarrier();
 let job=group.x+umDispatchX*group.y;let tiles=umMergedTileJobs();
 if(job<tiles){
  let header=7u*UM_TILES+16u;let slot=lane/8u;let index=job*umMergedPack+slot;
  if(slot<umMergedPack&&index<umSupport[header+1u]){
   let tile=umSupport[header+4u+umSupport[header]+index];
   let vertex=umTileCoord(tile)*4u+umCorner(lane%8u,2u)*4u;
   if((umCoarseCornerMask(tile)&(1u<<(lane%8u)))!=0u){umBandCandidate(vertex,4u);}
  }
 }else{
  let owner=umRegularCoarseOwner((job-tiles)*64u+lane);
  if(owner.width!=0u){
   // Owners on a negative domain wall also own their corners on that wall.
   let origin=umOrigin(owner);
   for(var k=select(7u,0u,any(origin==vec3u(0)));k<8u;k++){
    let corner=umCorner(k,2u);
    if(all((corner!=vec3u(0))|(origin==vec3u(0)))){umBandCandidate(origin+corner*owner.width,owner.width);}
   }
  }
 }
 workgroupBarrier();
 // One list atomic per group.
 if(lane==0u){let n=atomicLoad(&umBandCount);umBandTotal=n;if(n>0u){umBandBase=atomicAdd(&umClaims[UM_BAND_COUNT],n);}}
 let total=workgroupUniformLoad(&umBandTotal);
 for(var i=lane;i<total;i+=64u){deferred.data[umBandBase+i]=umBandList[i];}
}
// The listed band vertices, a lane each: a fixed grid strides over the list,
// so every lane of a searching group but the last holds a search.
@compute @workgroup_size(64) fn redistanceBand(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) groups:vec3u){
 let count=atomicLoad(&umClaims[UM_BAND_COUNT]);
 for(var i=gid.x;i<count;i+=64u*groups.x){
  let word=deferred.data[i];let vertex=vec3u(word&1023u,(word>>10u)&1023u,(word>>20u)&1023u);let width=1u<<(word>>30u);
  let initial=umLoadVertexW(vertex,width);let found=umRebuildSearch(vec3f(vertex),initial,width);
  if((params.flags.x&4u)!=0u&&width>1u){
   // A held vertex (umBandCandidate stored its value) publishes the difference.
   let held=umHeldIndex(vertex);
   if(atomicLoad(&umClaims[held])!=0u){atomicStore(&umClaims[held],bitcast<u32>(found-initial));continue;}
   let t=umTravelIndex(vertex);atomicStore(&umClaims[t],bitcast<u32>(fract(bitcast<f32>(atomicLoad(&umClaims[t])))));
  }
  textureStore(outputPhi,vec3i(vertex),vec4f(found));
 }
}
@compute @workgroup_size(64) fn advectOwners(@builtin(global_invocation_id) gid:vec3u,@builtin(local_invocation_index) lane:u32){
 umStagePlaneReach(lane);
 let owner=umOwner(gid);if(owner.width==0u){return;}
 let origin=umOrigin(owner);let regular=umRegularFine||(umTileMaximumWidth(owner.tile)==owner.width&&umTileMinimumWidth(owner.tile)==owner.width);
 // Every regular interior owner writes exactly its positive corner. Keep the
 // expensive characteristic/Newton evaluation out of the eight-corner loop;
 // only owners touching a negative domain wall own additional vertices.
 if(regular){
  let vertex=origin+vec3u(owner.width);
  umAdvectStore(vertex,owner.width);
  if(all(origin!=vec3u(0))){return;}
 }
 for(var k=0u;k<umCounts.w;k++){
  if(regular&&k==7u){continue;}
  let corner=umCorner(k,2u);let vertex=origin+corner*owner.width;
  var owned=false;
  if(regular){owned=all((corner!=vec3u(0))|(origin==vec3u(0)));}
  else{owned=umVertexAuthority(vertex).index==owner.index;}
  if(owned){umAdvectStore(vertex,owner.width);}
 }
}
${this.solid?/* wgsl */`// A cut 4h owner's centre can lie in the solid, where the walk stops at once.
// Its characteristic leaves the centroid of its open h cells; where that
// point is itself closed, the open h cells nearest it, as one mean (no cell
// of a tie is preferred). The owner's box moves by the displacement: the
// identity at rest.
fn umCutDeparture(owner:UMOwner)->vec3f{
 let origin=vec3f(umOrigin(owner));var sum=vec3f(0);var total=0.0;var mask:array<u32,2>;
 for(var k=0u;k<64u;k++){
  let local=umCorner(k,4u);let open=umCellOpen(vec3i(umOrigin(owner)+local));
  if(open>1e-5){sum+=open*(vec3f(local)+0.5);total+=open;mask[k>>5u]|=1u<<(k&31u);}
 }
 let centroid=origin+sum/total;
 if(umOpenAt(centroid)>1e-5){return umTrace(centroid)-centroid;}
 var best=3.402823e38;
 for(var k=0u;k<64u;k++){if((mask[k>>5u]&(1u<<(k&31u)))!=0u){
  let r=origin+vec3f(umCorner(k,4u))+0.5-centroid;best=min(best,dot(r,r));
 }}
 var moved=vec3f(0);var count=0.0;
 for(var k=0u;k<64u;k++){if((mask[k>>5u]&(1u<<(k&31u)))!=0u){
  let start=origin+vec3f(umCorner(k,4u))+0.5;let r=start-centroid;
  if(dot(r,r)<=best+1e-3){moved+=umTrace(start)-start;count+=1.0;}
 }}
 return moved/count;
}`:""}
fn umTraceCell(owner:UMOwner){
 let origin=umOrigin(owner);let centre=vec3f(origin)+vec3f(0.5*f32(owner.width));
 ${this.solid?/* wgsl */`if(owner.width==4u&&umSolidEnabled()){
  let open=umTileOpen(owner.tile);
  if(open>1e-5&&open<0.99999){textureStore(departures,vec3i(origin),vec4f(centre+umCutDeparture(owner),0));return;}
 }`:""}
 textureStore(departures,vec3i(origin),vec4f(umTrace(centre),0));
}
@compute @workgroup_size(64) fn traceCells(@builtin(global_invocation_id) gid:vec3u){
 let owner=umOwner(gid);if(owner.width!=0u){umTraceCell(owner);}
}
// The merged launch's jobs: a general h tile each (a lane per owner), then
// the seam 4h tiles 64 per job (a lane per owner: a merged tile job left 63
// lanes idle behind each one's blend-zone trace), then 64 regular coarse
// owners per job. Claimed as uniformMixedClaimedEntriesWGSL claims, with
// this job count.
fn umTraceMergedJobs()->vec3u{
 let header=7u*UM_TILES+16u;let general=umSupport[4u*UM_TILES+2u];let seams=general+(umSupport[header+1u]+63u)/64u;
 return vec3u(general,seams,seams+(umSupport[8u*UM_TILES+20u]+63u)/64u);
}
fn umTraceMergedOwner(job:u32,lane:u32,jobs:vec3u)->UMOwner{
 if(job<jobs.x){let tile=umSupport[6u*UM_TILES+16u+job];return UMOwner(tile,lane,1u,(umTopology[tile]&0x3fffffffu)+lane);}
 if(job>=jobs.y){return umRegularCoarseOwner((job-jobs.y)*64u+lane);}
 let header=7u*UM_TILES+16u;let slot=(job-jobs.x)*64u+lane;if(slot>=umSupport[header+1u]){return UMOwner();}
 let tile=umSupport[header+4u+umSupport[header]+slot];return UMOwner(tile,0u,4u,umTopology[tile]&0x3fffffffu);
}
var<workgroup> traceMergedJobs:vec3u;
var<workgroup> traceMergedClaim:u32;
@compute @workgroup_size(64) fn traceCellsMerged(@builtin(local_invocation_index) lane:u32){
 if(lane==0u){traceMergedJobs=umTraceMergedJobs();}
 let jobs=workgroupUniformLoad(&traceMergedJobs);
 loop{
  if(lane==0u){traceMergedClaim=atomicAdd(&umClaims[umClaimWord],1u);}
  let job=workgroupUniformLoad(&traceMergedClaim);
  if(job>=jobs.z){break;}
  let owner=umTraceMergedOwner(job,lane,jobs);if(owner.width!=0u){umTraceCell(owner);}
  workgroupBarrier();
 }
}
`,["advect","redistance","advectFine","redistanceFine","advectOwners","traceCells"],"umClaims"),["retirementEvidence","retirementEvidenceCoarse"])});
    const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");
    if(errors.length)throw new Error(errors.map(m=>`${m.lineNum}: ${m.message}`).join("\n"));
    const layout=this.device.createPipelineLayout({bindGroupLayouts:[this.ownership.bindLayout,this.resources,...(this.solid?[this.solid.tileLayout]:[]),...(this.hanging?[this.ownership.hangingLayout]:[])]});
    const compile=(entryPoint:string,constants:Record<string,number>,solidsOnly=false)=>this.twin(s=>uniformDetailPipeline(this.device,this.ownership,{layout,compute:{module,entryPoint,constants:{umDispatchX:this.ownership.dispatchX,...constants,...s}}}),{solidsOnly,entry:entryPoint});
    // NB shares its own crossing search with membership and never dispatches
    // the inherited distance/evidence or volume-characteristic kernels.
    const unusedInNarrowBand=new Set(["evidenceDistance","retirementEvidence","retirementEvidenceCoarse","redistance","redistanceFine","redistanceBand","prepareSurface","traceCells","traceCellsMerged"]);
    const keyed=async<K>(into:Map<K,GPUComputePipeline>,key:K,entryPoint:string,constants:Record<string,number>={},solidsOnly=false)=>{
      if(this.narrowBand&&unusedInNarrowBand.has(entryPoint))return;
      into.set(key,await compile(entryPoint,constants,solidsOnly));
    };
    const regular={umCellWidth:1,umPlannedFine:1,umRegularFine:1};
    await Promise.all([
      // Dispatched only under solids (encode: solid.present): no solid-free twin.
      ...(this.solid?["solidClosed","solidClear"]:[]).map(entryPoint=>keyed(this.pipelines,entryPoint,entryPoint,{},true)),
      keyed(this.pipelines,"wallReach","wallReach",{umWallReachGroups:wallReachGroups(this.ownership.capacity.lattice.dimensions)}),
      ...[0,1,2].map(axis=>keyed(this.pipelines,`evidenceDistance${axis}`,"evidenceDistance",{umEvidenceAxis:axis})),
      ...([["retirementEvidence",UNIFORM_MIXED_COUNTED.fineTiles],["retirementEvidenceCoarse",UNIFORM_MIXED_COUNTED.coarseTiles]] as const).map(([entryPoint,umCountedJobs])=>keyed(this.pipelines,entryPoint,entryPoint,{umCountedJobs})),
      ...(["advect","redistance"] as const).map(entry=>keyed(this.pipelines,entry,entry,{umMergedTiles:1,umMergedCoarse:1,umMergedPack:8,umCertifiedJobs:2,umClaimWord:2})),
      keyed(this.pipelines,"redistanceBand","redistanceBand"),
      keyed(this.pipelines,"prepareSurface","prepareSurface"),
      keyed(this.pipelines,"traceCells","traceCellsMerged",{umClaimWord:2}),
      ...(["advect","redistance"] as const).map(entry=>keyed(this.finePipelines,entry,`${entry}Fine`,{umCellWidth:1,umPlannedFine:2,umCertifiedJobs:1,umClaimWord:1})),
      Promise.all([compile("advectWalls",{}),compile("advectDeferred",{umMergedTiles:1}),compile("advectDeferred",regular)]).then(deferred=>{this.deferredPipelines.push(...deferred);}),
      ...([["advect","advectOwners"],["redistance","redistanceFine"],["traceCells","traceCells"]] as const).map(([entry,entryPoint])=>keyed(this.regularPipelines,entry,entryPoint,{...regular,umCertifiedJobs:1,umClaimWord:0})),
    ]);
  }
  private twin(create:(solid:Record<string,number>)=>Promise<GPUComputePipeline>,options?:{solidsOnly?:boolean;entry?:string}):Promise<GPUComputePipeline>{return uniformMixedSolidPipeline(this.solid,create,options);}
  private variant(pipeline:GPUComputePipeline):GPUComputePipeline{return uniformDetailPick(this.solid?.select(pipeline)??pipeline);}
  resetTravel(encoder:GPUCommandEncoder):void{
    encoder.clearBuffer(this.claims,4*surfaceClaimWords(this.ownership.capacity.lattice.dimensions));
  }
  encode(encoder:GPUCommandEncoder,entry:"advect"|"redistance"|"traceCells",group:UniformDetailGroup):void{
    const pipeline=this.pipelines.get(entry);if(!pipeline)throw new Error("Mixed surface stage is not initialized");
    // Only advect reads wall reach. Travel beyond the scratch prefix is
    // persistent, including across the intervening redistance/trace stages.
    encoder.clearBuffer(this.claims,0,entry==="advect"?4*surfaceClaimWords(this.ownership.capacity.lattice.dimensions):4*SURFACE_CLAIM_WORDS);
    const pass=encoder.beginComputePass({label:`Uniform mixed surface ${entry}`});pass.setBindGroup(0,this.ownership.bindGroup);pass.setBindGroup(1,group.group);if(this.solid)pass.setBindGroup(2,this.solid.tileGroup);if(this.hanging)pass.setBindGroup(this.solid?3:2,this.ownership.hangingGroup);
    if(entry==="advect"){
      pass.setPipeline(this.variant(this.pipelines.get("wallReach")!));pass.dispatchWorkgroups(6*wallReachGroups(this.ownership.capacity.lattice.dimensions));
      // The closed/clear evidence only exists for a scene holding solids.
      if(this.solid?.present){
        const tiles=this.ownership.capacity.tiles,x=this.ownership.dispatchX,groups=Math.ceil(tiles/64);
        pass.setPipeline(this.variant(this.pipelines.get("solidClosed")!));pass.dispatchWorkgroups(Math.min(tiles,x),Math.ceil(tiles/x));
        pass.setPipeline(this.variant(this.pipelines.get("solidClear")!));pass.dispatchWorkgroups(Math.min(groups,x),Math.ceil(groups/x));
      }
    }
    if(entry==="redistance"){
      // Evidence covers the h/4h partition: a job per h tile, 64 4h tiles per job.
      this.ownership.dispatchTierCounted(pass,this.variant(this.pipelines.get("retirementEvidence")!),0);
      // A 4h tile's evidence reads its eight corners: the base block.
      pass.setBindGroup(1,group.base);
      this.ownership.dispatchTierCounted(pass,this.variant(this.pipelines.get("retirementEvidenceCoarse")!),1);
      pass.setBindGroup(1,group.group);
      const t=this.ownership.capacity.lattice.dimensions.map(n=>n/4);
      for(const axis of [0,1,2]){
        const lines=t[(axis+1)%3]!*t[(axis+2)%3]!;
        pass.setPipeline(this.variant(this.pipelines.get(`evidenceDistance${axis}`)!));
        pass.dispatchWorkgroups(Math.min(lines,this.ownership.dispatchX),Math.ceil(lines/this.ownership.dispatchX));
      }
    }
    if(entry==="redistance"&&!this.ownership.coarseOnly){
      const tiles=this.ownership.capacity.tiles,x=this.ownership.dispatchX;
      pass.setPipeline(this.variant(this.pipelines.get("prepareSurface")!));
      pass.dispatchWorkgroups(Math.min(tiles,x),Math.ceil(tiles/x));
    }
    // Buffered direct grids; the claim loops consume the full current lists.
    const grid=Math.min(UNIFORM_MIXED_CLAIMED_GRID,this.ownership.capacity.tiles);
    this.ownership.dispatchBuffered(pass,this.variant(this.regularPipelines.get(entry)!),"fine",grid);
    if(entry!=="traceCells")this.ownership.dispatchBuffered(pass,this.variant(this.finePipelines.get(entry)!),"fine",grid);
    this.ownership.dispatchBuffered(pass,this.variant(pipeline),entry==="traceCells"?"merged":"coarseMerged",grid);
    // The band vertices the merged launch listed, 64 a group: at most one a
    // regular owner and a few a seam 4h tile, so its job budget covers them.
    if(entry==="redistance")this.ownership.dispatchBuffered(pass,this.variant(this.pipelines.get("redistanceBand")!),"coarseMerged",grid);
    if(entry==="advect"){
      const d=this.ownership.capacity.lattice.dimensions.map(n=>n+1),x=this.ownership.dispatchX;
      const groups=Math.ceil(2*(d[1]!*d[2]!+d[2]!*d[0]!+d[0]!*d[1]!)/64);
      pass.setPipeline(this.variant(this.deferredPipelines[0]!));pass.dispatchWorkgroups(Math.min(groups,x),Math.ceil(groups/x));
    }
    pass.end();
    if(entry==="advect"){
      const deferred=encoder.beginComputePass({label:"Uniform mixed surface advect deferred"});deferred.setBindGroup(0,this.ownership.bindGroup);deferred.setBindGroup(1,group.group);
      if(this.solid)deferred.setBindGroup(2,this.solid.tileGroup);if(this.hanging)deferred.setBindGroup(this.solid?3:2,this.ownership.hangingGroup);
      // [merged, regular h]: each strides the entries its own launch listed,
      // and no h launch ran without h-tile capacity.
      for(const pipeline of this.deferredPipelines.slice(1,this.ownership.coarseOnly?2:3)){deferred.setPipeline(this.variant(pipeline));deferred.dispatchWorkgroups(this.deferredGrid);}
      deferred.end();
    }
  }
  destroy():void{this.claims.destroy();}
}
