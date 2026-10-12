import { uniformDetailBindLayout, uniformDetailExtent, uniformDetailModule, uniformDetailPipeline, uniformDetailPick, uniformDetailGroup, type UniformDetailGroup } from "./uniform-detail-fields";
import { uniformMixedPressureCutWGSL } from "./uniform-mixed-pressure-cut.wgsl";
import {UNIFORM_DETAIL_4H_LOAD} from "../../core/uniform-detail-abi";
import type { UniformMixedOwnership } from "./uniform-mixed-ownership";
import { UNIFORM_MIXED_COUNTED, uniformMixedCountedEntriesWGSL, uniformMixedPageCount, uniformMixedTopologyWGSL } from "./uniform-mixed-topology.wgsl";
import { uniformVolumeCorrectionWGSL } from "./uniform-volume-correction.wgsl";
import { uniformMixedDetachedMassWGSL } from "./uniform-mixed-detached-mass.wgsl";
import { uniformMixedPressureLiquidWGSL } from "./uniform-mixed-pressure-surface.wgsl";
import { uniformMixedPressureBoundaryIndexWGSL, uniformMixedPressureBoundaryLoop, uniformMixedPressureStorage } from "./uniform-mixed-pressure-boundary.wgsl";
import type { UniformMixedLayout } from "./uniform-mixed-layout";
import { uniformMixedSolidPipeline, uniformMixedSolidWGSL, type UniformMixedSolid } from "./uniform-mixed-solid.wgsl";

export interface UniformMixedPressureAuthorityFields {
 centerPhi:GPUTexture;volume:GPUTexture;targetFill:GPUTexture;
 /** Each owner's phi; coarse mode appends uniformMixedRootMaskWords. */
 phi:GPUBufferBinding;
 /** Phase flag for momentum's supported physical donor gather. */
 phase:GPUTexture;
 correction:GPUTexture;
 scratch:GPUBufferBinding;
 /** dt, deficit balance (0 on, -1 off), unused, dust threshold. */
 params:GPUBuffer;
 /** Coarse mode only: the vertex phi in simulation ownership (the cut tiles
  * the simulation holds at h), and the root topology this stage writes
  * (uniformMixedRootTopologyBytes). */
 fine?:{phi:GPUTexture;topology:GPUBufferBinding};
 /** Resident mode only: the simulation ownership's tile words (the tiles it
  * holds at h, where V claims no row). */
 simulation?:GPUBufferBinding;
}
/** The all-4h root's topology of one pressure solve: a vec4 per owner and halo
 * slot, in the static record's order (open, V+x, V+y, V+z). */
export function uniformMixedRootTopologyBytes(layout:UniformMixedLayout):number{return 16*uniformMixedPressureStorage(layout).count;}
/** What follows the owners' phi in the root's phi words with coarse solids,
 * three floats a tile: its +x, +y and +z face masks (bit u+4v of an h face the
 * root's topology keeps). The RHS has no storage binding to spare. */
export const UNIFORM_MIXED_ROOT_TILE_WORDS=3;
export function uniformMixedRootMaskWords(layout:UniformMixedLayout):number{return UNIFORM_MIXED_ROOT_TILE_WORDS*layout.tiles.length;}
/** One pressure interface for RHS, projection, momentum support and extension.
 * Excess/deficit balance uses physical cell mass; coarse cells are not counted
 * as one fine cell. All fields and reduction scratch are caller-owned.
 * With static solids, fine owners follow native pressurePhi exactly: rho'=V/open
 * claims rows, closed cells continue the open liquid interface (CM11a's one
 * layer of solid unknowns), and balance counts open liquid capacity only.
 * Coarse mode (band pressure's all-4h owners) reads the static all-4h solid
 * record the same way: capacity is the owner's mean open fraction, a closed
 * owner continues its open liquid neighbours, and deficits are of target*cap.
 * A wide owner's centre phi cannot see V the owner does not resolve, so
 * over capacity V claims its row (umOwnerSurfacePhi): material piled into an
 * air centre has no row, the liquid beside it sees a vacuum face inside that
 * material, and its excess correction jets into it (native
 * pressureSurfacePhi's rule; the all-4h Figure 9 piled 40 capacities into
 * one air owner and projected 30 m/s into it). V claims only where the
 * simulation holds the owner at 4h: an h owner keeps its centre phi, and so
 * does the all-4h root's owner of a tile the simulation holds at h, whose
 * row is a coarse model of the band's h rows (claimed there too, the 64-cube
 * dam's accepted residual rose under Dynamic: mean of the runs' maxima 2.8
 * to 3.6 of the tolerance's 5).
 * The rows are written first (rows), and build reads them for the owner and
 * for detached mass's neighbours: mass is detached when it touches no row,
 * which is the velocity stage's test (umPressureLiquid on these words). The
 * extension takes a face of a phase owner as physical, so an owner detached
 * here but beside a row there kept the air faces the projection had zeroed
 * (centre phi against the rows V claims: the all-4h Figure 9's first impact
 * peaked at 25 m/s, 21 with the rows). Rebuilding a neighbour's row in
 * place costs its V and capacity on every owner's lane (+0.25 ms on that
 * scene and on the pool); the launch costs a dispatch.
 * A cut tile the simulation holds at h has no 4h centre phi: its corner
 * vertices may be buried in the solid, where phi is not state. Its owner's
 * phi is the least-squares plane through the tile's live h vertices (those
 * of an open h cell of the tile), read at the tile centre: the plane's own
 * value there, which is what a cut owner simulated at 4h reads from its eight
 * corners, so a resting plane surface is one hydrostatic root across an h/4h
 * seam (native's open-child vote prefers air and broke it at shorelines).
 * The fit runs first, sharing each workgroup across cut tiles, into scratch
 * after the balance words; build reads it for the owner and for a closed owner's cut
 * neighbours.
 * Coarse mode then writes the root's topology (wet, after the rows it
 * reads). The static record's V of a 4h face is the mean of its 16 h faces,
 * whatever lies behind them, and a wall no thicker than a tile puts the liquid
 * and the air outside it in one owner: the bath's resting liquid met p=0
 * through its own wall, the root settled 15% under hydrostatic, and the wall
 * faces took 1.4 to 4 times the step's gravity. A face of a cut tile the
 * simulation holds at h keeps only the h faces the liquid's rows hold: an
 * open liquid h cell on one side, and across a closed h cell an owner that is
 * itself a row (an air owner's wall is the band's contact, not a surface).
 * The RHS sums the same h faces, so a resting liquid is one hydrostatic root
 * whatever its vessel.
 * Resident mode (the split's all-4h pressure ownership, whose support carries
 * the simulation's residency certificate) strides the resident pages' owners
 * only (residentAll): an absent page is certified far air (V=0, corner phi at
 * least 16h), so its owners add nothing to the balance and no reader needs
 * their phase or correction. Resident mode writes no phase at all: phase
 * is simulation state (momentum's and the extension's donor flag), which the
 * simulation authority of the same frame wrote and nothing reads in pressure
 * ownership. Their phi is still read (neighbour faces, the
 * native continuation lattice), and the pressure root's phi words are shared
 * scratch, so an "absent" pass writes umAuthority(o,0) for each owner of an
 * absent page: the same value the dense build writes there. */
/** Owner entries: GPU-counted launches over every tier (umAllOwner). */
const counted:readonly string[]=["rows","build","resolve","surface"];
/** The stage's passes, described once. */
const SURFACE_PASS:GPUComputePassDescriptor={label:"NB-FLIP surface pressure authority"};
const BALANCE_PASS:GPUComputePassDescriptor={label:"Uniform mixed pressure authority and volume correction"};
const PHASE_PASS:GPUComputePassDescriptor={label:"Uniform mixed pressure authority phase"};

/** surfaceOnly is the NB-FLIP contract: pressure and donor phase follow phi,
 * with solid continuation but no independently conserved mass. Its one owner
 * pass writes phi, phase and a zero source; no balance kernels are compiled.
 * The shared cut-cell fit is geometry, not volume recovery. */
export class UniformMixedPressureAuthority {
 readonly allocatedBytes=0;
 readonly scratchBytes:number;
 /** Owners this ownership can hold: the construction cellCount. */
 private readonly owners:number;
 /** Balance partials (one per counted job) and their chunk sums. */
 private readonly groups:number;
 private readonly chunks:number;
 /** Resident mode's coarse jobs: the lattice's residency pages. */
 private readonly pages:number;
 /** The cut launch's groups: 64 tiles of the lattice each. */
 private readonly cutGroups:number;
 private readonly resources:GPUBindGroupLayout;
 private readonly pipelines=new Map<string,GPUComputePipeline>();
 constructor(private readonly device:GPUDevice,readonly ownership:UniformMixedOwnership,private readonly solid?:UniformMixedSolid,private readonly coarse=false,private readonly resident=false,readonly surfaceOnly=false){
  if(coarse&&(!solid?.coarse||ownership.layout.tiles.some(word=>(word&0xc0000000)!==0)))throw new Error("Coarse mixed pressure authority requires the all-4h solid record and all-4h ownership");
  // A capacity bound, read once: the unified frame reserves its simulation
  // ownership all-h (cellCount = 64 per tile) and pressure ownership is the
  // fixed all-4h hierarchy, so no later generation has more owners. build's
  // counted jobs (umCounts.x + ceil(umCounts.y/64), which chunks reads back
  // from the GPU) therefore never exceed groups. Resident mode's coarse jobs
  // are its resident pages instead (umCounts.x + umResidentPageCount()): a
  // page is 4^3 tiles of a lattice it may overhang, so there are more pages
  // than ceil(tiles/64) unless every dimension is a multiple of 16 (27 pages
  // for the 16 groups of 40^3, whose last 11 partials were dropped).
  this.owners=ownership.layout.cellCount;
  this.pages=uniformMixedPageCount(ownership.capacity.lattice);this.cutGroups=Math.max(1,Math.min(1024,Math.ceil(ownership.capacity.tiles/64)));
  this.groups=surfaceOnly?0:Math.ceil(this.owners/64)+(resident?this.pages:0);this.chunks=Math.ceil(this.groups/1024);
  // Two vec2f words per partial; coarse: one vec2f cut-phi slot per owner
  // after the balance words.
  this.scratchBytes=8*(1+2*(this.groups+this.chunks)+(coarse?this.owners:0));
  this.resources=uniformDetailBindLayout(device,{entries:[
   ...[0,1,2].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float" as const,viewDimension:"3d" as const}})),
   ...[3,6].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage" as const}})),
   ...[4,5].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"write-only" as const,format:"r32float" as const,viewDimension:"3d" as const}})),
   {binding:7,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},
   ...(coarse?[8].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float" as const,viewDimension:"3d" as const}})):[]),
   ...(coarse?[{binding:10,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage" as const}}]:[]),
   ...(resident?[{binding:9,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage" as const}}]:[]),
  ]});
 }
 bind(f:UniformMixedPressureAuthorityFields):UniformDetailGroup{
  if((f.scratch.size??f.scratch.buffer.size-(f.scratch.offset??0))<this.scratchBytes)throw new Error("Mixed pressure authority scratch is too small");
  const d=this.ownership.capacity.lattice.dimensions;
  for(const t of [f.centerPhi,f.volume,f.targetFill,f.phase,f.correction])if(t.format!=="r32float"||uniformDetailExtent(t).some((v,a)=>v!==d[a]))throw new Error("Mixed pressure authority requires native scalar fields");
  if([f.centerPhi,f.volume,f.targetFill].some(t=>t===f.phase||t===f.correction)||f.phase===f.correction)throw new Error("Mixed pressure authority outputs must be disjoint");
  if(!!f.fine!==this.coarse)throw new Error("Coarse mixed pressure authority needs the h simulation vertex phi, and only it");
  if(!!f.simulation!==this.resident)throw new Error("Resident mixed pressure authority needs the simulation's tile words, and only it");
  if(f.fine&&(f.fine.phi.format!=="r32float"||uniformDetailExtent(f.fine.phi).some((v,a)=>v!==d[a]!+1)))throw new Error("Coarse mixed pressure authority requires the native h vertex phi");
  const topologyBytes=uniformMixedRootTopologyBytes(this.ownership.layout);
  if(f.fine&&(f.fine.topology.size??f.fine.topology.buffer.size-(f.fine.topology.offset??0))<topologyBytes)throw new Error("Coarse mixed pressure authority's root topology is too small");
  return uniformDetailGroup(this.device,{layout:this.resources,entries:[
   ...[f.centerPhi,f.volume,f.targetFill].map((t,binding)=>({binding,resource:t})),
   {binding:3,resource:{...f.phi,size:4*(this.ownership.capacity.owners+(this.coarse?uniformMixedRootMaskWords(this.ownership.layout):0))}},
   {binding:4,resource:f.phase},{binding:5,resource:f.correction},
   {binding:6,resource:{...f.scratch,size:this.scratchBytes}},{binding:7,resource:{buffer:f.params,size:16}},
   ...(f.fine?[{binding:8,resource:f.fine.phi},{binding:10,resource:{...f.fine.topology,size:topologyBytes}}]:[]),
   ...(f.simulation?[{binding:9,resource:f.simulation}]:[]),
  ]});
 }
 async initialize():Promise<void>{
  // rows, build (and its phase variant) and resolve stride the GPU-counted
  // owners of every tier (umAllOwner); build writes one balance partial per job.
  const module=uniformDetailModule(this.device,{label:"Uniform mixed pressure authority",code:uniformMixedCountedEntriesWGSL(uniformMixedTopologyWGSL(this.ownership.capacity,0)+/* wgsl */`
@group(1) @binding(0) var centerPhi:texture_3d<f32>;
@group(1) @binding(1) var volume:texture_3d<f32>;
@group(1) @binding(2) var targetFill:texture_3d<f32>;
@group(1) @binding(3) var<storage,read_write> phi:array<f32>;
@group(1) @binding(4) var phase:texture_storage_3d<r32float,write>;
@group(1) @binding(5) var correction:texture_storage_3d<r32float,write>;
@group(1) @binding(6) var<storage,read_write> balance:array<vec2f>;
@group(1) @binding(7) var<uniform> params:vec4f;
${uniformVolumeCorrectionWGSL}
${uniformMixedSolidWGSL(this.solid?2:undefined,this.solid?.coarse?.count)}
const UM_HMIN=${Math.min(...this.ownership.capacity.lattice.cellSize_m)};
// An owner's own texel. A 4h owner's is its tile origin: the base blocks.
fn umOwnerCenterPhi(o:UMOwner)->f32{let at=vec3i(umOrigin(o));if(o.width==4u){return ${UNIFORM_DETAIL_4H_LOAD}textureLoad(centerPhi,at,0).x;}return textureLoad(centerPhi,at,0).x;}
fn umOwnerVolume(o:UMOwner)->f32{let at=vec3i(umOrigin(o));if(o.width==4u){return ${UNIFORM_DETAIL_4H_LOAD}textureLoad(volume,at,0).x;}return textureLoad(volume,at,0).x;}
fn umFill(o:UMOwner)->f32{let at=vec3i(umOrigin(o));if(o.width==4u){return ${UNIFORM_DETAIL_4H_LOAD}textureLoad(targetFill,at,0).x;}return textureLoad(targetFill,at,0).x;}
// pressureSurfacePhi for one fine cell (solid scenes only): centre phi.
// A cell of a 4h owner reads the owner's (its origin texel): its other texels are not state.
fn umSurfacePhiCell(p:vec3i)->f32{let o=umOwnerAt(p);return textureLoad(centerPhi,select(p,vec3i(umOrigin(o)),o.width!=0u),0).x;}
// pressurePhi of an h owner's cell: a closed cell continues its open liquid
// neighbours, an open one reads its own texel.
fn umPressurePhiCell(p:vec3i)->f32{
 if(umCellIsOpen(p)){return textureLoad(centerPhi,p,0).x;}
 var terms:array<f32,6>;var weights:array<f32,6>;
 for(var i=0u;i<6u;i++){var q=p;q[i/2u]+=select(-1,1,(i&1u)!=0u);let open=umCellOpen(q);
  terms[i]=0.0;weights[i]=0.0;if(open<=1e-5){continue;}
  let phi=umSurfacePhiCell(q);if(phi<0.0){terms[i]=open*phi;weights[i]=open;}}
 let weight=((weights[0]+weights[1])+(weights[4]+weights[5]))+(weights[2]+weights[3]);
 let sum=((terms[0]+terms[1])+(terms[4]+terms[5]))+(terms[2]+terms[3]);
 return select(0.5*UM_HMIN,sum/max(weight,1e-9),weight>0.0);
}
// What an owner can hold: its open fraction (an h cell's, a 4h owner's tile's).
fn umCapacity(o:UMOwner)->f32{${this.coarse?"return umSolidCoarse(o.index).x;":"if(!umSolidEnabled()){return 1.0;}if(o.width==1u){return umCellOpen(vec3i(umOrigin(o)));}return umTileOpen(o.tile);"}}
// umPressureLiquid on this authority's phi: the rows' liquid owners.
fn umAuthorityLiquid(o:UMOwner,distance:f32)->bool{return ${uniformMixedPressureLiquidWGSL("distance","f32(o.width)*UM_HMIN")};}
// A closed owner continues its neighbours' phi in the pressure rows but owns
// no phase. The extension takes a phase owner's faces as physical, and a face
// between two closed owners holds no velocity of its own: as a source it is
// a zero that every sample within half a cell of the wall blends in. Unowned,
// it continues the liquid's tangential velocity (free slip).
fn umPhase(o:UMOwner,liquid:bool)->f32{return select(0.0,1.0,liquid&&umCapacity(o)>1e-5);}
${this.resident?"@group(1) @binding(9) var<storage,read> umSimulationTiles:array<u32>;":""}
// V claims a row only where the simulation holds the owner at 4h.
fn umClaims(o:UMOwner)->bool{return ${this.resident?"(umSimulationTiles[o.tile]&0x80000000u)==0u":"o.width>=2u"};}
// A neighbour's V: an absent page's volume is not transferred (certified V=0).
fn umNeighbourVolume(n:UMOwner)->f32{return ${this.resident?"select(0.0,umOwnerVolume(n),umTileResident(n.tile))":"umOwnerVolume(n)"};}
// Surface phi of an owner: its centre phi, and for a wide owner holding V
// the distance its fill implies where that is nearer. Over capacity the
// distance continues on both sides (native pressureSurfacePhi: a branch at
// V=cap would jump from the raw positive phi to zero).
fn umOwnerSurfacePhi(o:UMOwner,v:f32,cap:f32)->f32{
 let centre=umOwnerCenterPhi(o);
 ${this.surfaceOnly?"return centre;":`if(v<=0.0||cap<=1e-5||!umClaims(o)){return centre;}
 let width=f32(o.width)*UM_HMIN;
 return min(centre,max(width*(1.0-v/cap),-0.5*width));`}
}
// Detached mass touches no row: rows has written every owner's phi.
${uniformMixedDetachedMassWGSL(o=>`umAuthorityLiquid(${o},phi[${o}.index])`,o=>`umOwnerVolume(${o})`,"params.w")}
${this.coarse?`@group(1) @binding(8) var vertexPhi:texture_3d<f32>;
const UM_CUT_BASE=${1+2*(this.groups+this.chunks)}u;
const pcHeld=${this.surfaceOnly};
${uniformMixedPressureCutWGSL}
// Vertex v of the tile's 5^3: live when one of the tile's h cells at it is open.
fn cutLive(open:vec2u,v:u32)->bool{
 let g=vec3i(vec3u(v%5u,(v/5u)%5u,v/25u));
 for(var k=0u;k<8u;k++){let c=g-vec3i(umCorner(k,2u));
  if(all(c>=vec3i(0))&&all(c<vec3i(4))){let i=u32(c.x+4*c.y+16*c.z);if(((open[i>>5u]>>(i&31u))&1u)!=0u){return true;}}}
 return false;
}
// One face-connected set of the tile's open h cells (bit x+4y+16z), grown from
// its lowest: a wall through the tile leaves the rest on its other side.
fn cutSide(open:vec2u)->vec2u{
 var side=vec2u(0u);
 if(open.x!=0u){side.x=open.x&(~open.x+1u);}else{side.y=open.y&(~open.y+1u);}
 for(var k=0u;k<64u;k++){
  let x=((side<<vec2u(1u))&vec2u(0xeeeeeeeeu))|((side>>vec2u(1u))&vec2u(0x77777777u));
  let y=((side<<vec2u(4u))&vec2u(0xfff0fff0u))|((side>>vec2u(4u))&vec2u(0x0fff0fffu));
  let z=vec2u(side.x<<16u,(side.y<<16u)|(side.x>>16u))|vec2u((side.x>>16u)|(side.y<<16u),side.y>>16u);
  let grown=(side|x|y|z)&open;if(all(grown==side)){break;}side=grown;
 }
 return side;
}
// Ordered reference for frozen shader replays; unreachable from production
// entry points, so the shader compiler removes its private vertex array.
// Keep physics edits in sync with uniform-mixed-pressure-cut.wgsl.ts.
fn cutTile(t:u32){
 let origin=vec3i(4u*umTileCoord(t));let open=umSolidOpenCells(t);
 // Empty/full masks preserve the ordered fit: no live vertices use 2h;
 // all 125 vertices have exactly zero centred mean, so only their mean
 // phi contributes at the tile centre. Partial masks keep the full fit.
 if(all(open==vec2u(0u))){balance[UM_CUT_BASE+(umTopology[t]&0x3fffffffu)]=vec2f(2.0*UM_HMIN,0.0);return;}
 if(all(open==vec2u(0xffffffffu))){
  var count=0.0;var total=0.0;
  for(var v=0u;v<125u;v++){count+=1.0;total+=textureLoad(vertexPhi,origin+vec3i(vec3u(v%5u,(v/5u)%5u,v/25u)),0).x;}
  balance[UM_CUT_BASE+(umTopology[t]&0x3fffffffu)]=vec2f(total/count,0.0);return;
 }
 var cutVertex:array<f32,125>;var live=array<u32,4>(0u,0u,0u,0u);var rest=array<u32,4>(0u,0u,0u,0u);
 // Liquid on one side of a wall alone: the fit is that side's.
 let side=cutSide(open);let other=open&~side;var wet=false;var wetRest=false;
 for(var v=0u;v<125u;v++){if(cutLive(open,v)){cutVertex[v]=textureLoad(vertexPhi,origin+vec3i(vec3u(v%5u,(v/5u)%5u,v/25u)),0).x;
  if(cutLive(side,v)){live[v>>5u]|=1u<<(v&31u);wet=wet||cutVertex[v]<0.0;}
  else{rest[v>>5u]|=1u<<(v&31u);wetRest=wetRest||cutVertex[v]<0.0;}}}
 for(var w=0u;w<4u;w++){if(wetRest||!wet){live[w]=select(live[w]|rest[w],rest[w],wetRest&&!wet);}}
 // Centred normal equations in cell units about the tile centre: a live
 // vertex set holds a whole h cell, so the moment matrix is definite.
 var count=0.0;var first=vec3f(0);var total=0.0;
 for(var v=0u;v<125u;v++){if(((live[v>>5u]>>(v&31u))&1u)==0u){continue;}
  count+=1.0;first+=vec3f(vec3u(v%5u,(v/5u)%5u,v/25u))-2.0;total+=cutVertex[v];}
 var value=2.0*UM_HMIN;
 if(count>0.0){
  let mean=first/count;let level=total/count;var diagonal=vec3f(0);var cross3=vec3f(0);var moment=vec3f(0);
  for(var v=0u;v<125u;v++){if(((live[v>>5u]>>(v&31u))&1u)==0u){continue;}
   let r=vec3f(vec3u(v%5u,(v/5u)%5u,v/25u))-2.0-mean;
   diagonal+=r*r;cross3+=r.xxy*r.yzz;moment+=r*(cutVertex[v]-level);}
  let c0=vec3f(diagonal.x,cross3.x,cross3.y);let c1=vec3f(cross3.x,diagonal.y,cross3.z);let c2=vec3f(cross3.y,cross3.z,diagonal.z);
  let slope=vec3f(dot(moment,cross(c1,c2)),dot(c0,cross(moment,c2)),dot(c0,cross(c1,moment)))/dot(c0,cross(c1,c2));
  value=level-dot(slope,mean);
 }
 balance[UM_CUT_BASE+(umTopology[t]&0x3fffffffu)]=vec2f(value,0.0);
}
fn umCutPhi(o:UMOwner)->f32{return balance[UM_CUT_BASE+o.index].x;}
// An open owner's phi: a cut h tile's is its fitted plane; where phi is the
// one liquid authority, an uncut h tile's is the mean of its h cells (cut).
fn umOpenOwnerPhi(o:UMOwner,v:f32,cap:f32)->f32{
 let t=umTileAt(umOrigin(o)/4u);
 if(umSolidCut(t)||(pcHeld&&umSolidFineTile(t))){return umCutPhi(o);}
 return umOwnerSurfacePhi(o,v,cap);
}
@group(1) @binding(10) var<storage,read_write> rootTopology:array<vec4f>;
const UM_ROOT_MASKS=${this.ownership.capacity.owners}u;
${uniformMixedPressureBoundaryIndexWGSL(this.ownership.layout)}
// The tile's owner is a row: umPressureLiquid on the words just written.
fn umWetRow(t:u32)->bool{return ${uniformMixedPressureLiquidWGSL("phi[umTopology[t]&0x3fffffffu]","4.0*UM_HMIN")};}
fn umWetOpen(open:vec2u,l:vec3u)->bool{let i=l.x+4u*l.y+16u*l.z;return ((open[i>>5u]>>(i&31u))&1u)!=0u;}
// The +axis face of tile t, toward tile n: (V, mask) of the h faces the
// liquid's rows hold. An h cell of a tile the simulation holds at h is liquid
// by its centre phi (the mean of its corners, summed here); a 4h tile's cells
// follow their owner.
fn umWetFace(t:u32,n:u32,origin:vec3i,axis:u32)->vec2f{
 let a=(axis+1u)%3u;let b=(axis+2u)%3u;
 let fineLow=umSolidFineTile(t);let fineHigh=umSolidFineTile(n);let rowLow=umWetRow(t);let rowHigh=umWetRow(n);
 let openLow=umSolidOpenCells(t);let openHigh=umSolidOpenCells(n);
 // The vertex planes of the two cell layers at the face; a 4h tile's own
 // plane is not state.
 var vertex:array<f32,75>;
 for(var k=0u;k<75u;k++){
  let plane=k/25u;if((plane==0u&&!fineLow)||(plane==2u&&!fineHigh)){continue;}
  var p=origin;p[a]+=i32(k%5u);p[b]+=i32((k/5u)%5u);p[axis]+=3+i32(plane);
  vertex[k]=textureLoad(vertexPhi,p,0).x;
 }
 var kept=0.0;var mask=0u;
 for(var j=0u;j<4u;j++){for(var i=0u;i<4u;i++){
  var id=origin;id[a]+=i32(i);id[b]+=i32(j);id[axis]+=3;
  let at=i+5u*j;let middle=(vertex[25u+at]+vertex[26u+at])+(vertex[30u+at]+vertex[31u+at]);
  let v=umPatchFaceV(t,id,axis);if(v<=0.0){continue;}
  var l=vec3u(0u);l[a]=i;l[b]=j;l[axis]=3u;let low=umWetOpen(openLow,l);l[axis]=0u;let high=umWetOpen(openHigh,l);
  var wetLow=low&&rowLow;var wetHigh=high&&rowHigh;
  if(low&&fineLow){wetLow=middle+((vertex[at]+vertex[1u+at])+(vertex[5u+at]+vertex[6u+at]))<0.0;}
  if(high&&fineHigh){wetHigh=middle+((vertex[50u+at]+vertex[51u+at])+(vertex[55u+at]+vertex[56u+at]))<0.0;}
  if((wetLow||wetHigh)&&(low||rowLow)&&(high||rowHigh)){kept+=v;mask|=1u<<(i+4u*j);}
 }}
 return vec2f(kept/16.0,f32(mask));
}
fn umWetTile(t:u32){
 let o=UMOwner(t,0u,4u,umTopology[t]&0x3fffffffu);let c=umTileCoord(t);
 var topology=umSolidCoarse(o.index);var masks=vec3f(65535.0);
 for(var axis=0u;axis<3u;axis++){
  if(c[axis]+1u>=UM_T[axis]){continue;}
  var q=c;q[axis]+=1u;let n=umTileAt(q);
  if(!umSolidCut(t)&&!umSolidCut(n)){continue;}
  let face=umWetFace(t,n,vec3i(4u*c),axis);topology[axis+1u]=face.x;masks[axis]=face.y;
 }
 rootTopology[o.index]=topology;let words=UM_ROOT_MASKS+${UNIFORM_MIXED_ROOT_TILE_WORDS}u*t;
 for(var axis=0u;axis<3u;axis++){phi[words+axis]=masks[axis];}
 // A domain wall's slot keeps the record's V.
 ${uniformMixedPressureBoundaryLoop("rootTopology[halo]=umSolidCoarse(halo);")}
}
@compute @workgroup_size(64) fn wet(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) l:u32){
 for(var t=64u*group.x+l;t<UM_TILES;t+=64u*groups.x){umWetTile(t);}
}`:""}
fn umAuthority(o:UMOwner,v:f32)->f32{
 ${this.coarse?`let cap=umCapacity(o);if(cap>1e-5){return umOpenOwnerPhi(o,v,cap);}
 // A closed owner continues its open liquid neighbours, capacity-weighted.
 var terms:array<f32,6>;var weights:array<f32,6>;
 for(var i=0u;i<6u;i++){
  terms[i]=0.0;weights[i]=0.0;var q=vec3i(umOrigin(o));q[i/2u]+=select(-i32(o.width),i32(o.width),(i&1u)!=0u);
  if(any(q<vec3i(0))||any(q>=vec3i(UM_D))){continue;}
  let n=umOwnerAt(q);let open=umCapacity(n);if(n.width==0u||open<=1e-5){continue;}
  let phi=umOpenOwnerPhi(n,${this.surfaceOnly?"0.0":"umNeighbourVolume(n)"},open);if(phi<0.0){terms[i]=open*phi;weights[i]=open;}
 }
 let weight=((weights[0]+weights[1])+(weights[4]+weights[5]))+(weights[2]+weights[3]);
 let sum=((terms[0]+terms[1])+(terms[4]+terms[5]))+(terms[2]+terms[3]);
 return select(0.5*UM_HMIN*f32(o.width),sum/max(weight,1e-9),weight>0.0);`:`if(umSolidEnabled()&&o.width==1u){return umPressurePhiCell(vec3i(umOrigin(o)));}
 return umOwnerSurfacePhi(o,v,umCapacity(o));`}
}
fn umDeficit(o:UMOwner,v:f32,distance:f32)->f32{
 let cap=umCapacity(o);
 if(cap<=1e-5||v>cap||!umAuthorityLiquid(o,distance)){return 0.0;}
 // The fill target is absolute (at most the capacity) at every width.
 return max(0.0,umFill(o)-v);
}
// Diverging transport dilutes bulk liquid and leaves its V in a skin of air
// owners above phi. That stranded V (uncut air owners) is a second budget,
// spent only on deep bulk deficits (uncut, more than a 4h width below the
// surface), so the contraction refilling the diluted liquid draws the skin
// back down. It relaxes with a 0.25 s half-life, not the excess's 1/30 s:
// a resting pool's skin persists for seconds, while a moving body's V lags
// its phi for a few frames. Spent at the excess rate, or on cut and shallow
// owners, the lag contracted transient flow (cm12-figure-9's falling ball
// stretched and its dam front frayed). Cut owners keep the excess-only rate.
fn umStranded(o:UMOwner,v:f32,distance:f32,detached:bool)->f32{
 let cap=umCapacity(o);
 if(cap<=1e-5||umAuthorityLiquid(o,distance)||detached||umFill(o)>0.0){return 0.0;}
 return uvVolumeCorrectionFractionAt(params.x/7.5)*min(v,cap);
}
fn umBulkDeficit(o:UMOwner,v:f32,distance:f32)->f32{return select(0.0,umDeficit(o,v,distance),umFill(o)>=umCapacity(o)&&distance< -4.0*UM_HMIN);}
// false: the phase-only build (phi and phase; no balance reduction).
override umAuthorityBalance:bool=true;
var<workgroup> sums:array<vec4f,64>;
// Every owner's row phi, before build reads it and its face neighbours'.
@compute @workgroup_size(64) fn rows(@builtin(global_invocation_id) gid:vec3u){
 let o=${this.resident?"umResidentAllOwner":"umAllOwner"}(gid);if(o.width!=0u){phi[o.index]=umAuthority(o,umOwnerVolume(o));}
}
// NB-FLIP has one liquid authority: reconstructed phi. No mass claims,
// detached-mass phase flags, balance reductions or recovery source. Its one
// correction is volume control: the divergence the budget's gauge asks of
// every liquid owner alike (params.z, 1/s), over the owner's open share.
@compute @workgroup_size(64) fn surface(@builtin(global_invocation_id) gid:vec3u){
 let o=${this.resident?"umResidentAllOwner":"umAllOwner"}(gid);if(o.width==0u){return;}
 let distance=umAuthority(o,0.0);let origin=vec3i(umOrigin(o));
 phi[o.index]=distance;
 ${this.resident?"":"textureStore(phase,origin,vec4f(umPhase(o,umAuthorityLiquid(o,distance))));"}
 var control=0.0;if(params.z!=0.0&&umAuthorityLiquid(o,distance)){control=params.z*umCapacity(o);}
 textureStore(correction,origin,vec4f(control));
}
fn umReduce(l:u32){workgroupBarrier();for(var stride=32u;stride>0u;stride/=2u){if(l<stride){sums[l]+=sums[l+stride];}workgroupBarrier();}}
@compute @workgroup_size(64) fn build(@builtin(global_invocation_id) gid:vec3u,@builtin(local_invocation_index) l:u32,@builtin(workgroup_id) group:vec3u){
 let o=${this.resident?"umResidentAllOwner":"umAllOwner"}(gid);var values=vec4f(0);
 if(o.width!=0u){let origin=vec3i(umOrigin(o));let v=umOwnerVolume(o);let distance=phi[o.index];
  let liquid=umAuthorityLiquid(o,distance);let detached=!liquid&&umDetachedMass(o);
  ${this.resident?"":"textureStore(phase,origin,vec4f(umPhase(o,liquid||detached)));"}
  if(umAuthorityBalance){
  let cap=umCapacity(o);
  values=vec4f(uvVolumeCorrectionAmountAt(v,cap,params.x),umDeficit(o,v,distance),umStranded(o,v,distance,detached),umBulkDeficit(o,v,distance))*f32(o.width*o.width*o.width);
  // Native balance counts open liquid rows only.
  if(umSolidEnabled()&&(cap<=1e-5||!umAuthorityLiquid(o,distance))){values=vec4f(0,0,values.z,0);}
  }
 }
 if(!umAuthorityBalance){return;}
 sums[l]=values;umReduce(l);let index=group.x+umDispatchX*group.y;
 if(l==0u&&index<${this.groups}u){balance[1u+2u*index]=sums[0].xy;balance[2u+2u*index]=sums[0].zw;}
}
@compute @workgroup_size(64) fn chunks(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) l:u32){
 var value=vec4f(0);for(var i=group.x*1024u+l;i<min(${this.resident?"umCounts.x+umResidentPageCount()":"(umCounts.x*64u+umCounts.y+63u)/64u"},(group.x+1u)*1024u);i+=64u){value+=vec4f(balance[1u+2u*i],balance[2u+2u*i]);}
 sums[l]=value;umReduce(l);if(l==0u){balance[${1+2*this.groups}u+2u*group.x]=sums[0].xy;balance[${2+2*this.groups}u+2u*group.x]=sums[0].zw;}
}
@compute @workgroup_size(64) fn reduce(@builtin(local_invocation_index) l:u32){
 var value=vec4f(0);for(var i=l;i<${this.chunks}u;i+=64u){value+=vec4f(balance[${1+2*this.groups}u+2u*i],balance[${2+2*this.groups}u+2u*i]);}
 sums[l]=value;umReduce(l);if(l==0u){
  // rate: every deficit, from excess. bulk: what remains of uncut deficits,
  // from stranded V.
  var rate=0.0;var bulk=0.0;let s=sums[0];
  if(params.y>=0.0&&s.y>0.0){rate=min(1.0,s.x/s.y);}
  if(params.y>=0.0&&s.w>0.0){bulk=min(1.0-rate,s.z/s.w);}
  balance[0]=vec2f(rate,bulk);}
}
@compute @workgroup_size(64) fn resolve(@builtin(global_invocation_id) gid:vec3u){
 let o=${this.resident?"umResidentAllOwner":"umAllOwner"}(gid);if(o.width==0u){return;}let origin=vec3i(umOrigin(o));let v=umOwnerVolume(o);
 let distance=phi[o.index];
 let amount=uvVolumeCorrectionAmountAt(v,umCapacity(o),params.x)-balance[0].x*umDeficit(o,v,distance)-balance[0].y*umBulkDeficit(o,v,distance);
 textureStore(correction,origin,vec4f(amount/max(params.x,1e-12)));
}
${this.resident?`// One group per page: the 4h owners of an absent page take the dense
// build's phi (V=0 there).
@compute @workgroup_size(64) fn absent(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) l:u32){
 let page=group.x;if(page>=UM_PAGES||umPageResident(page)){return;}
 let tile=umPageTile(page,l);if(tile>=UM_TILES||umTileWidth(tile)!=4u){return;}
 let o=UMOwner(tile,0u,4u,umTopology[tile]&0x3fffffffu);phi[o.index]=umAuthority(o,0.0);
}`:""}
`,counted)});
  const info=await module.getCompilationInfo(),errors=info.messages.filter(m=>m.type==="error");if(errors.length)throw new Error(errors.map(m=>`${m.lineNum}: ${m.message}`).join("\n"));
  const layout=this.device.createPipelineLayout({bindGroupLayouts:[this.ownership.bindLayout,this.resources,...(this.solid?[this.solid.tileLayout]:[])]});
  const mode=this.resident?UNIFORM_MIXED_COUNTED.residentAll:UNIFORM_MIXED_COUNTED.all;
  await Promise.all([...(this.surfaceOnly?["surface"]:["rows","build","chunks","reduce","resolve"]),...(this.coarse?["cut","wet"]:[]),...(this.resident?["absent"]:[])].map(async entryPoint=>{this.pipelines.set(entryPoint,await uniformMixedSolidPipeline(this.solid,s=>uniformDetailPipeline(this.device,this.ownership,{layout,compute:{module,entryPoint,constants:{umDispatchX:this.ownership.dispatchX,...(counted.includes(entryPoint)?{umCountedJobs:mode}:{}),...s}}})));}));
  if(!this.surfaceOnly)this.pipelines.set("phase",await uniformMixedSolidPipeline(this.solid,s=>uniformDetailPipeline(this.device,this.ownership,{layout,compute:{module,entryPoint:"build",constants:{umDispatchX:this.ownership.dispatchX,umCountedJobs:mode,umAuthorityBalance:0,...s}}})));
 }
 /** balance=false writes phi and phase only: the extension's authority,
  * when this same stage (same ownership, same origin texels) rebuilds the
  * correction and its balance scratch before their readers (band rows, RHS). */
 private variant(pipeline:GPUComputePipeline):GPUComputePipeline{return uniformDetailPick(this.solid?.select(pipeline)??pipeline);}
 /** The owner launches: resident mode's all-4h ownership has no h jobs, so
  * its jobs are its resident pages (any grid is correct; this prices it). */
 private dispatchOwners(pass:GPUComputePassEncoder,pipeline:GPUComputePipeline):void{
  if(this.resident)this.ownership.dispatchCounted(pass,pipeline,this.pages);else this.ownership.dispatchAllCounted(pass,pipeline);
 }
 /** Coarse mode: the root's topology, from the rows the pass has written
  * and the h vertices (the full group, as cut). */
 private encodeTopology(pass:GPUComputePassEncoder,group:UniformDetailGroup):void{
  if(!this.coarse)return;
  pass.setBindGroup(1,group.group);pass.setPipeline(this.variant(this.pipelines.get("wet")!));pass.dispatchWorkgroups(this.cutGroups);
 }
 encode(encoder:GPUCommandEncoder,group:UniformDetailGroup,balance=true):void{
  if(this.pipelines.size!==(this.surfaceOnly?1:6)+(this.coarse?2:0)+(this.resident?1:0))throw new Error("Mixed pressure authority is not initialized");
  const pass=encoder.beginComputePass(this.surfaceOnly?SURFACE_PASS:balance?BALANCE_PASS:PHASE_PASS);pass.setBindGroup(0,this.ownership.bindGroup);pass.setBindGroup(1,group.group);if(this.solid)pass.setBindGroup(2,this.solid.tileGroup);
  if(this.coarse){pass.setPipeline(this.variant(this.pipelines.get("cut")!));pass.dispatchWorkgroups(this.cutGroups);}
  // The all-4h ownership's owners are tiles: every load and store below is
  // a tile origin, of the owner or of a face neighbour (cut alone reads h
  // vertices): the base blocks.
  if(this.resident)pass.setBindGroup(1,group.base);
  if(this.resident){pass.setPipeline(this.variant(this.pipelines.get("absent")!));pass.dispatchWorkgroups(this.pages);}
  if(this.surfaceOnly){this.dispatchOwners(pass,this.variant(this.pipelines.get("surface")!));this.encodeTopology(pass,group);pass.end();return;}
  this.dispatchOwners(pass,this.variant(this.pipelines.get("rows")!));
  if(!balance){this.dispatchOwners(pass,this.variant(this.pipelines.get("phase")!));this.encodeTopology(pass,group);pass.end();return;}
  const build=this.variant(this.pipelines.get("build")!);pass.setPipeline(build);this.dispatchOwners(pass,build);
  pass.setPipeline(this.variant(this.pipelines.get("chunks")!));pass.dispatchWorkgroups(this.chunks);
  pass.setPipeline(this.variant(this.pipelines.get("reduce")!));pass.dispatchWorkgroups(1);
  const resolve=this.variant(this.pipelines.get("resolve")!);pass.setPipeline(resolve);this.dispatchOwners(pass,resolve);
  this.encodeTopology(pass,group);pass.end();
 }
}
