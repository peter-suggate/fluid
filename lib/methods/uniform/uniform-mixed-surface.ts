import { NARROW_BAND_TRACE_LIMIT, narrowBandTraceWGSL } from "./uniform-narrow-band-advection.wgsl";
import { uniformDetailBindLayout, uniformDetailExtent, uniformDetailModule, uniformDetailPipeline, uniformDetailPick, uniformDetailGroup, type UniformDetailGroup } from "./uniform-detail-fields";
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
/** Per second: the NB-FLIP surface vertices' fourth-difference weight (umVertexSmoothing). */
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
}

/** Narrow-band phi advection on canonical vertices of a resolved phi field
 * (UniformMixedPhiResolve): a vertex beside the surface is carried by its foot
 * on the zero set, every other along its own characteristic. Hanging values
 * are sampled from their authority, never expanded. Static solids take the
 * walked trace, the buried-vertex freeze and a released embedded wall's air
 * continuation; they live near fine owners only. */
export class UniformMixedSurface {
  readonly allocatedBytes:number;
  private readonly resources: GPUBindGroupLayout;
  /** Job counters and wall reach, cleared per frame. */
  readonly claims: GPUBuffer;
  /** wallReach, solidClosed and solidClear, then advect by launch: advect
   * (merged), advectFine (the general h list, a lane per vertex) and
   * advectOwners (regular h). */
  private readonly pipelines = new Map<string, GPUComputePipeline>();
  /** advectWalls and advectDeferred for the merged and regular dispatches, and the deferred grid. */
  private readonly deferredPipelines:GPUComputePipeline[]=[];
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
  constructor(private readonly device: GPUDevice, readonly ownership: UniformMixedOwnership,private readonly sourceParams?:GPUBuffer,private readonly solid?:UniformMixedSolid) {
    this.resources = uniformDetailBindLayout(device,{entries:[
      ...[0,2,3,11].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float" as const,viewDimension:"3d" as const}})),
      {binding:1,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"write-only",format:"r32float",viewDimension:"3d"}},
      {binding:5,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},
      {binding:6,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},
      ...(solid?[{binding:8,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage" as const}}]:[]),
      ...(sourceParams?[{binding:9,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform" as const}}]:[]),
      ...[10,12,13].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage" as const}})),
    ]});
    const address=this.address=ownership.capacity.address;
    // A deferred advect word holds a vertex key in 30 bits (all ones is no word): 29,878 pages of 33³ vertices.
    if(address.vertexKeys>0x3fffffff)throw new Error(`Mixed surface deferred advect packs a vertex key in 30 bits: ${address.vertexKeys} vertex keys exceed 1073741823 (29,878 pages of 35,937)`);
    this.allocatedBytes=4*surfaceClaimWords(address);
    this.claims=device.createBuffer({label:"Uniform mixed surface job claims",size:this.allocatedBytes,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC});
    this.deferredGrid=Math.min(1024,Math.ceil(address.vertexKeys/64));
    this.wallGroups=Math.min(WALL_REACH_GRID,wallFaces(address));
    this.deferredWallGroups=Math.min(WALL_VERTEX_GRID,Math.ceil(1089*wallFaces(address)/64));
    uniformKeyGroups(address.tileKeys,64,ownership.dispatchX,this.tileLaneGroups);uniformKeyGroups(address.tileKeys,1,ownership.dispatchX,this.tileGroups);
  }
  /** bind()'s evidence scratch from a 256-aligned offset: the tile words, then the deferred advect list. */
  get scratchBytes():number{
    const c=this.ownership.capacity;
    return Math.ceil((this.solid?8*c.tiles:0)/256)*256+16+4*this.address.vertexKeys;
  }
  bind(f: UniformMixedSurfaceFields): UniformDetailGroup {
    const a=this.address;
    for(const [i,t] of [f.phi,f.outputPhi,f.velocity,f.coarseVelocity].entries()){
      // The 4h velocity cache is keyed by block.
      const size=i<2?a.vertexBox:i===3?uniformCoarseCacheExtent(a):a.cells;
      if(t.format!==(i>=2?"rgba32float":"r32float") || uniformDetailExtent(t).some((n,a)=>n!==size[a]))
        throw new Error("Mixed surface requires native vertex/cell fields and a 4h velocity cache");
    }
    if(f.phi===f.outputPhi)throw new Error("Mixed surface outputs overlap inputs");
    // The solid closed and clear tile words.
    const evidenceBytes=this.solid?8*this.ownership.capacity.tiles:0;
    // The deferred advect list follows, a separate binding: count, three
    // unused words, then at most one word per lattice vertex.
    const deferredOffset=Math.ceil(((f.evidence.offset??0)+evidenceBytes)/256)*256,deferredBytes=16+4*a.vertexKeys;
    if((f.evidence.size??f.evidence.buffer.size-(f.evidence.offset??0))<deferredOffset-(f.evidence.offset??0)+deferredBytes)throw new Error("Mixed surface evidence scratch is too small");
    const group=uniformDetailGroup(this.device,{layout:this.resources,entries:[
      ...[f.phi,f.outputPhi,f.velocity,f.coarseVelocity].map((t,binding)=>({binding,resource:t})),
      {binding:5,resource:{buffer:f.negative}},{binding:6,resource:{buffer:f.params,size:32}},
      ...(this.solid?[{binding:8,resource:{...f.evidence,size:evidenceBytes}}]:[]),
      {binding:10,resource:{buffer:f.evidence.buffer,offset:deferredOffset,size:deferredBytes}},
      ...(this.sourceParams?[{binding:9,resource:{buffer:this.sourceParams,size:UNIFORM_PARAMS_BYTES}}]:[]),
      {binding:11,resource:f.unitVelocity},
      {binding:12,resource:{buffer:this.claims}},
      {binding:13,resource:{buffer:f.narrowBandState}},
    ]});
    return group;
  }
  async initialize(): Promise<void> {
    // regularTexture: interior unit-stencil samples load their eight taps
    // directly (no per-tap boundary branch, transverse clamp or select); the
    // skipped taps are exactly the zero-weight ones.
    const velocitySampling=uniformMixedVelocitySamplingSource(false,"coarseVelocity","velocity",this.solid?3:2,"unitVelocity");
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
 if(umFaceLow(anchor,axis)){return negative[umLowFaceIndex(vec3u(max(anchor,vec3i(0))),axis)];}
 return textureLoad(velocity,anchor,0)[axis];
}
@group(1) @binding(11) var unitVelocity:texture_3d<f32>;
${velocitySampling}
${narrowBandTraceWGSL("umSampleVelocity")}
// The cell a point (in cell units) is read from: the one it lies in. A point
// on a page face, edge or corner lies in every page that meets there: where
// the high side's is not stored the cell is that of the first stored one of
// the others (the low side on x first), so a point on a wall, or on the face
// of an AIR page, is read from the stored page it bounds. The page table is
// read only for a point on a page face.
fn umPointCell(p:vec3f)->vec3i{
 let c=vec3i(floor(p));let on=((c&vec3i(31))==vec3i(0))&(p==vec3f(c));
 if(!any(on)||umCellKind(c)==UM_RESIDENT){return c;}
 let faces=u32(on.x)|(u32(on.y)<<1u)|(u32(on.z)<<2u);
 for(var low=1u;low<8u;low++){
  if((low&faces)!=low){continue;}
  let q=c-vec3i(vec3u(low&1u,(low>>1u)&1u,low>>2u));
  if(umCellKind(q)==UM_RESIDENT){return q;}
 }
 return c;
}
// p as a tap reaches it from the page: held inside it on an axis only where
// a SOLID page lies across that face (umPointNearFromPage). An AIR page
// never holds. p lies in the page or one of the 26 around it.
fn umPointFromPage(page:vec3i,p:vec3f)->vec3f{
 let low=vec3f(page<<vec3u(5u));if(all(p>=low)&&all(p<=low+vec3f(32.0))){return p;}
 return umPointNearFromPage(umBlockOf(page),page,p);
}
fn umPointFrom(home:vec3f,p:vec3f)->vec3f{return umPointFromPage(umPointCell(home)>>vec3u(5u),p);}
fn umSurfaceTrace(p:vec3f)->vec3f{
 var k1=umSampleVelocity(p)/params.hDt.xyz;
 let steps=nbTraceSteps(k1,params.hDt.w);
 if(steps>${NARROW_BAND_TRACE_LIMIT}u){atomicStore(&nbState[3],1u);return p;}
 var q=p;let base=-params.hDt.w/f32(steps);var units=1u;
 for(var i=0u;i<steps;i+=units){
  if(i>0u){k1=umSampleVelocity(q)/params.hDt.xyz;}let span=nbTraceSpan(q,base,k1,steps-i);
  units=u32(span.w);q=umPointFrom(q,span.xyz);
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
fn umSolidClear(p:vec3i)->bool{return !umSolidEnabled()||evidence[UM_TILES+umTileAt(vec3u(umPointCell(vec3f(p)))/4u)]!=0u;}
// uvBuried: phi at a vertex no incident owner has capacity at is not state.
fn umBuried(p:vec3f)->bool{return umVertexBuried(vec3i(p));}
fn umFluidFace(face:vec3i,axis:u32)->f32{
 if(umSolidValid(face)){return textureLoad(velocity,face,0)[axis];}
 var n=face;n[axis]+=1;
 if(umSolidValid(n)&&umFaceLow(face,axis)){return negative[umLowFaceIndex(vec3u(n),axis)];}
 return 0.0;
}
fn umContactReleased(face:vec3i,axis:u32)->bool{
 var cell=face;var bit=axis;if(umFaceLow(face,axis)){cell[axis]=0;bit+=3u;}
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
 let wx=umCatmull(fraction.x);let wy=umCatmull(fraction.y);let wz=umCatmull(fraction.z);
 // Resolved: every tap is a width-aligned vertex within one tile of the
 // owner's tile (4h taps are 4-aligned, stored everywhere), so it is stored
 // or resolved. The 64 taps then unroll with a constant bound.
 // A tap is held in the owner's page where a SOLID page lies across a face
 // (umVertexNearFromPage), so every tap has an incident cell no wall parts
 // from the owner. With no embedded solids it cannot be buried, at any owner
 // width. Fold that case
 // before the 64 taps: the fine-tile clearance certificate alone excludes
 // coarse owners and leaves their per-tap domain tests in the solid-free twin.
 ${this.solid?"let tapsClear=!umSolidEnabled()||((umRegularFine||cell.width==1u)&&umSolidClear(base));":""}
 var value=0.0;var low=1e30;var high=-1e30;let page=base>>vec3u(5u);let home=umBlockOf(page);
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
    ${this.solid?"if(!tapsClear&&wx[x]*wy[y]*wz[z]!=0.0&&umBuried(vec3f(vertex))){return cell.value;}":""}
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
  if(inward*umSampleVelocity(probe)[axis]>=-1e-6){continue;}
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
// Phi is vertex state and both its couplings are means: a centre's phi is the
// mean of its eight vertices, and a vertex moves with the faces around its
// foot. A vertex pattern alternating along x or z has no pressure and no
// advection, and nothing removed it: the outline of a disturbance stood in a
// 4h surface for good, and stripes grew in an h one. The fourth difference
// along the two horizontal axes, at the vertex's own width, damps that
// pattern and leaves a cubic alone: a step takes 16 eps of an alternating
// pattern and eps (2 pi/n)^4 of a wave n vertices long. An axis is skipped
// where a wall lies before its far tap and where the vertex or a tap is not
// open liquid state.
fn umVertexSmoothing(vertex:vec3u,width:u32)->f32{
 let eps=min(${UNIFORM_MIXED_VERTEX_SMOOTHING_PER_S.toExponential(8)}*params.hDt.w,0.03125);if(!(eps>0.0)){return 0.0;}
 let w=i32(width);let c=umSampleVertex(vec3f(vertex));if(!(abs(c)<2.0*f32(width)*min(params.hDt.x,min(params.hDt.y,params.hDt.z)))){return 0.0;}
 ${this.solid?"if(!umSmoothOpen(vec3i(vertex),w)){return 0.0;}":""}
 var sum=0.0;
 for(var axis=0u;axis<3u;axis+=2u){
  if(umWallWithin(vec3i(vertex),axis,2*w)){continue;}
  var part=6.0*c;var valid=true;
  for(var k=0u;k<4u;k++){
   var q=vec3i(vertex);q[axis]+=select(w,2*w,(k&2u)!=0u)*select(-1,1,(k&1u)!=0u);
   ${this.solid?"if(!umSmoothOpen(q,w)){valid=false;}":""}
   part+=select(-4.0,1.0,(k&2u)!=0u)*umSampleVertex(vec3f(q));
  }
  if(valid){sum+=part;}
 }
 return -eps*sum;
}
fn umAdvected(p:vec3f,width:u32)->f32{
 ${this.solid?/* wgsl */`let clear=umSolidClear(vec3i(p));if(!clear&&umBuried(p)){return umLoadVertex(vec3u(p));}
 let end=umVertexTrace(p,width);let walk=umWalk(p,end);let q=walk.q;`:"let q=umVertexTrace(p,width);"}let cell=umVertexCell(q);var value=cell.value;
 let h=f32(width)*min(params.hDt.x,min(params.hDt.y,params.hDt.z));
 if(params.flags.y!=0u&&abs(value)<2.0*h&&(params.flags.y==1u||width>1u)){value=umCubicPhi(q,cell);}
 ${this.solid?"value=umWallContact(p,value,width);if(walk.hit){value=umEmbeddedAir(p,end,value);}value=umReleasedWalls(p,value);":"value=umReleasedWalls(p,umWallContact(p,value,width));"}
 return ${this.sourceParams?"umSourceuvSourcePhi(p,value)":"value"};
}
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
fn umAdvectStore(vertex:vec3u,width:u32){
 if(params.hDt.w>0.0&&umWallPlane(vec3i(vertex))<6u){return;}
 let p=vec3f(vertex);var defer=false;var value=0.0;
 ${this.solid?/* wgsl */`// Off a clear tile a vertex can be buried: it keeps its value, umAdvected's
 // first exit.
 if(!umSolidClear(vec3i(p))&&umBuried(p)){textureStore(outputPhi,vec3i(vertex),vec4f(umLoadVertex(vertex)));return;}
 // A hit is evaluated by advectDeferred. Its first-pass phi and source
 // samples are discarded.
 {let walk=umWalk(p,umVertexTrace(p,width));defer=walk.hit;let q=walk.q;if(!defer){`:"{let q=umVertexTrace(p,width);"}
  let cell=umVertexCell(q);value=cell.value;
  let h=f32(width)*min(params.hDt.x,min(params.hDt.y,params.hDt.z));
  if(params.flags.y!=0u&&abs(value)<2.0*h&&(params.flags.y==1u||width>1u)){value=umCubicPhi(q,cell);}
  defer=defer||umReleasedMayChange(p,value);
  value=${this.sourceParams?"umSourceuvSourcePhi(p,value)":"value"};
 ${this.solid?"}":""}
 }
 if(!defer){textureStore(outputPhi,vec3i(vertex),vec4f(value+umVertexSmoothing(vertex,width)));return;}
 // Each canonical vertex is stored once: the list holds at most every vertex.
 // A word is the vertex key (umVertexAt) below a two-bit code: log2 of the
 // width (0, 2) off the regular-h dispatch, 3 on it. All ones is no word.
 let i=atomicAdd(&deferred.header[0],1u);
 deferred.data[i]=umVertexAt(vertex)|(select(firstTrailingBit(width),3u,umRegularFine)<<30u);
}
// The deferred vertices of the dispatch compiled with the same umRegularFine.
// A fixed grid (deferredGrid workgroups) strides over the list.
@compute @workgroup_size(64) fn advectDeferred(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 umStagePlaneReach(lane);
 let count=atomicLoad(&deferred.header[0]);
 for(var i=gid.x;i<count;i+=64u*groups.x){
  let word=deferred.data[i];let code=word>>30u;
  if((code==3u)!=umRegularFine){continue;}
  let vertex=umVertexCoord(word&0x3fffffffu);
  let width=select(1u<<code,1u,code==3u);
  textureStore(outputPhi,vec3i(vertex),vec4f(umAdvected(vec3f(vertex),width)+umVertexSmoothing(vertex,width)));
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
    if(all(vec3u(vertex)%width==vec3u(0))){
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
// The departure of a level-set vertex: a vertex within two of its own cells
// of the surface is carried by the surface, displaced as its foot on the zero
// set is. The vertices beside the surface keep their advected values, so
// nothing restores a distance there: carried by the velocity at their own
// depth, any steady shear under a still surface moved their values and not
// the zero between them, and the pressure rows, which read the surface off
// those values, saw a rise as a dip and filled it
// (uniform-trough-settled-tank: the wet vertex 0.93 h under the surface
// 2e-2 h off its distance in 240 steps).
fn umVertexTrace(p:vec3f,width:u32)->vec3f{
 let w=f32(width);let h=min(params.hDt.x,min(params.hDt.y,params.hDt.z));
 let value=umSampleVertex(p)/h;
 if(!(abs(value)<2.0*w)){return umSurfaceTrace(p);}
 let g=umSurfaceGradient(p,w)/h;let g2=dot(g,g);
 if(g2<1e-6){return umSurfaceTrace(p);}
 var offset=-value*g/g2;let reach=length(offset);if(reach>2.0*w){offset*=2.0*w/reach;}
 let foot=umPointFrom(p,p+offset);
 ${this.solid?"if(!umOpenAt(foot)){return umSurfaceTrace(p);}":""}
 // The foot's displacement, applied at p: within two widths of where the foot's trace ended, and reached from there.
 let end=umSurfaceTrace(foot);return umPointFrom(end,p+(end-foot));
}
fn umSurfaceGradient(p:vec3f,width:f32)->vec3f{
 var g=vec3f(0);var lowValue=0.0;let page=umPointCell(p)>>vec3u(5u);
 for(var k=0u;k<6u;k++){
  let axis=k/2u;var delta=vec3f(0);delta[axis]=0.25*width;
  var low=umPointFromPage(page,p-delta);var high=umPointFromPage(page,p+delta);
  // A wide slope is one-sided beside a closed owner.
  ${this.solid?"if(width>1.0){if(umSampleClosed(low)){low=p;}if(umSampleClosed(high)){high=p;}}":""}
  let value=umSampleVertex(select(low,high,(k&1u)!=0u));
  if((k&1u)==0u){lowValue=value;}else{g[axis]=(value-lowValue)/max(high[axis]-low[axis],1e-6);}
 }return g;
}
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
 if(group.x==0u&&lane==0u){atomicStore(&deferred.header[0],0u);}
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
@compute @workgroup_size(64) fn advectFine(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 umStagePlaneReach(lane);
 let owner=umTileJobOwner(group);if(owner.width==0u){return;}
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
 let owner=umOwner(gid);if(owner.width==0u){return;}
 let origin=umOrigin(owner);let regular=umRegularFine||(umTileMaximumWidth(owner.tile)==owner.width&&umTileMinimumWidth(owner.tile)==owner.width);
 let low=umLowOpen(vec3i(origin));
 // Every regular interior owner writes exactly its positive corner. Keep the
 // expensive characteristic evaluation out of the eight-corner loop;
 // only owners touching a negative domain wall own additional vertices.
 if(regular){
  let vertex=origin+vec3u(owner.width);
  umAdvectStore(vertex,owner.width);
  if(!any(low)){return;}
 }
 for(var k=0u;k<umCounts.w;k++){
  if(regular&&k==7u){continue;}
  let corner=umCorner(k,2u);let vertex=origin+corner*owner.width;
  var owned=false;
  if(regular){owned=all((corner!=vec3u(0))|low);}
  else{owned=umVertexAuthority(vertex).index==owner.index;}
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
    await Promise.all([
      // Dispatched only under solids (encode: solid.present): no solid-free twin.
      ...(this.solid?["solidClosed","solidClear"]:[]).map(entryPoint=>keyed(entryPoint,{},true)),
      keyed("wallReach"),
      keyed("advect",{umMergedTiles:1,umMergedCoarse:1,umMergedPack:8,umCertifiedJobs:2,umClaimWord:2}),
      keyed("advectFine",{umCellWidth:1,umPlannedFine:2,umCertifiedJobs:1,umClaimWord:1}),
      Promise.all([compile("advectWalls",{}),compile("advectDeferred",{umMergedTiles:1}),compile("advectDeferred",regular)]).then(deferred=>{this.deferredPipelines.push(...deferred);}),
      keyed("advectOwners",{...regular,umCertifiedJobs:1,umClaimWord:0}),
    ]);
  }
  private twin(create:(solid:Record<string,number>)=>Promise<GPUComputePipeline>,options?:{solidsOnly?:boolean;entry?:string}):Promise<GPUComputePipeline>{return uniformMixedSolidPipeline(this.solid,create,options);}
  private variant(pipeline:GPUComputePipeline):GPUComputePipeline{return uniformDetailPick(this.solid?.select(pipeline)??pipeline);}
  encode(encoder:GPUCommandEncoder,group:UniformDetailGroup):void{
    const pipeline=this.pipelines.get("advect");if(!pipeline)throw new Error("Mixed surface stage is not initialized");
    encoder.clearBuffer(this.claims);
    const pass=encoder.beginComputePass(SURFACE_PASS);pass.setBindGroup(0,this.ownership.bindGroup);pass.setBindGroup(1,group.group);if(this.solid)pass.setBindGroup(2,this.solid.tileGroup);pass.setBindGroup(this.solid?3:2,this.ownership.hangingGroup);
    pass.setPipeline(this.variant(this.pipelines.get("wallReach")!));pass.dispatchWorkgroups(this.wallGroups);
    // The closed/clear evidence only exists for a scene holding solids.
    if(this.solid?.present){
      const closed=this.solid.coarse?this.tileLaneGroups:this.tileGroups,clear=this.tileLaneGroups;
      pass.setPipeline(this.variant(this.pipelines.get("solidClosed")!));pass.dispatchWorkgroups(closed[0],closed[1]);
      pass.setPipeline(this.variant(this.pipelines.get("solidClear")!));pass.dispatchWorkgroups(clear[0],clear[1]);
    }
    // Buffered direct grids; the claim loops consume the full current lists.
    const grid=Math.min(UNIFORM_MIXED_CLAIMED_GRID,this.ownership.capacity.tiles);
    this.ownership.dispatchBuffered(pass,this.variant(this.pipelines.get("advectOwners")!),"fine",grid);
    this.ownership.dispatchBuffered(pass,this.variant(this.pipelines.get("advectFine")!),"fine",grid);
    this.ownership.dispatchBuffered(pass,this.variant(pipeline),"coarseMerged",grid);
    pass.setPipeline(this.variant(this.deferredPipelines[0]!));pass.dispatchWorkgroups(this.deferredWallGroups);
    pass.end();
    const deferred=encoder.beginComputePass(DEFERRED_PASS);deferred.setBindGroup(0,this.ownership.bindGroup);deferred.setBindGroup(1,group.group);
    if(this.solid)deferred.setBindGroup(2,this.solid.tileGroup);deferred.setBindGroup(this.solid?3:2,this.ownership.hangingGroup);
    // [merged, regular h]: each strides the entries its own launch listed,
    // and no h launch ran without h-tile capacity.
    for(let i=1,n=this.ownership.coarseOnly?2:3;i<n&&i<this.deferredPipelines.length;i++){deferred.setPipeline(this.variant(this.deferredPipelines[i]!));deferred.dispatchWorkgroups(this.deferredGrid);}
    deferred.end();
  }
  destroy():void{this.claims.destroy();}
}
