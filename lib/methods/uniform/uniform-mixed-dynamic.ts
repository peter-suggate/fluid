import type {UniformMixedOwnership} from "./uniform-mixed-ownership";
import type {UniformMixedBandBits} from "./uniform-mixed-layout-builder";
import {uniformMixedTopologyWGSL} from "./uniform-mixed-topology.wgsl";
import {uniformMixedVertexSamplingSource} from "./uniform-mixed-vertex-sampling.wgsl";

/** Largest reach/hysteresis, in tiles, and the boundary rules' tile scan. */
export const UNIFORM_MIXED_DYNAMIC_DISTANCE_CAP=16;
/** Cells a departure box grows by for surface motion no frozen-velocity trace
 * predicts: the volume correction's phi shift, redistance and drain move a
 * surface by a fraction of a cell per frame. On cm12-figure-9 a half cell
 * missed fewer next-frame surface tiles than the radius-cube census did. */
export const UNIFORM_MIXED_DYNAMIC_SURFACE_DRIFT=0.5;
/** Interior V deficit that still counts as bulk liquid. At pressure
 * tolerance 5 submerged V drifts ~10% within six dam-break frames in fine
 * and 4h owners alike (same cells, both layouts); only a hole larger than a
 * quarter owner is V/phi disagreement that needs h resolution. */
export const UNIFORM_MIXED_DYNAMIC_FULL_TOLERANCE=0.25;
/** Default UniformMixedDynamicPolicy.surfaceTolerance, in h. */
export const UNIFORM_MIXED_DYNAMIC_SURFACE_TOLERANCE=0.5;
/** Fast moving bulk uses 4h; independent h phi and pressure retain surface samples. */
export const UNIFORM_MIXED_DYNAMIC_FAST_TRAVEL=4;
/** Default UniformMixedDynamicPolicy.boundaryTravel, in h per step. */
export const UNIFORM_MIXED_DYNAMIC_BOUNDARY_TRAVEL=1;
/** Census header words: counters 0-17, 18 = largest tile travel (f32 bits),
 * 19 = tiles liquid-conditional solid promotion added to the band. */
const HEADER=20;
/** Velocity bounds over any tile box: level 0 (the tile) lives in the atomic
 * census; levels 1..CUBE_LEVELS are cubes of side 2^j at every tile start, in
 * a plain buffer (atomic loads and stores do not coalesce). */
const CUBE_LEVELS=5;
/** A box query reads at most this many cubes before it takes coarser ones
 * (which may overhang the box on its short axes: looser, never tighter). */
const BOX_CUBES=64;

export interface UniformMixedDynamicPolicy {
 /** The time the census plans for: every frame until its layout's successor
  * is adopted. With the tile speed it bounds travel (boundary rules). */
 dt:number;
 /** Frames dt spans: departure boxes trace them one step at a time. */
 steps:number;
 /** Gravity, m/s²: the later frames trace a velocity it has changed. */
 gravity:readonly [number,number,number];
 /** Tiles added around each tile's predicted departure box. */
 reach:number;
 /** Extra tiles a fine tile keeps before it may coarsen. */
 hysteresis:number;
 /** Interior requires V >= 1 - fullTolerance; air requires V <= emptyTolerance. */
 fullTolerance:number;
 emptyTolerance:number;
 /** Surface error, in h, a crossing tile may take at 4h. An h tile coarsens
  * when trilinear phi from its 4h corners stays within it at every vertex
  * near the surface; a 4h tile refines when its 4h-lattice second difference
  * predicts twice that. 0 keeps every crossing tile at h. */
 surfaceTolerance:number;
 /** Travel, in h per step, from which a surface tile runs at 4h whatever its
  * shape: a fast front is better resolved in time at 4h (Courant/4) than at
  * h. Its tile's own velocity bounds decide. 0 disables. */
 fastTravel:number;
 /** Travel, in h per step, from which surface liquid a boundary redirects
  * is h whatever its speed. At 4h a sheet thinner than half an owner has no
  * pressure row and no momentum of its own: liquid driven into a closed
  * wall or solid piles up instead of turning, and a sheet climbing a wall
  * stalls (docs/plans/uniform-dynamic-coarsening.md, far-wall run-up).
  * Two rules: impact (surface within one 4h cell of a closed wall or solid
  * tile, moving toward it faster than along it) and lift (a tile on a closed vertical wall moving up
  * faster than along the wall). A free fast front meets neither. 0 disables. */
 boundaryTravel:number;
 /** Closed domain faces, bit axis+3*side (side 0 = low, 1 = high). */
 closedWalls:number;
 /** Axis-1 direction of "up": +1, -1, or 0 without gravity (no lift rule). */
 up:number;
}

export interface UniformMixedDynamicCensus {
 /** One byte per tile: the band, or liquid-conditional solid promotion,
  * requires h ownership. */
 readonly fine:Uint8Array;
 readonly fineTiles:number;
 /** Tiles holding a phi sign change. */
 readonly interfaceTiles:number;
 /** Interface tiles 4h cannot resolve (surfaceTolerance): they and their departure boxes are h. */
 readonly requiredTiles:number;
 /** 4h interface tiles found unresolvable, refined by this census. */
 readonly unresolvedCoarse:number;
 /** Up to three of those tiles. */
 readonly unresolvedTiles:readonly number[];
 /** Surface tiles required h by a boundary (impact or lift), not by shape. */
 readonly boundaryTiles:number;
 /** Coarse interface owners by cause: partial V, phi sign change, phi liquid without V. */
 readonly coarsePartialVolume:number;
 readonly coarsePhiCrossing:number;
 readonly coarseDryLiquidPhi:number;
 /** max(1 - V) over owners two widths inside phi, and max V over phi-air owners, [fine, coarse]. */
 readonly interiorDeficit:readonly [number,number];
 readonly airVolume:readonly [number,number];
 readonly refined:number;
 readonly coarsened:number;
 /** Tiles in `fine` only for solid promotion. */
 readonly solidTiles:number;
}

/** State-driven ownership census (docs/plans/uniform-dynamic-coarsening.md).
 * Classifies each tile against the live ownership after a completed frame and
 * predicts the next frame's surface tiles. The next frame advects phi and V
 * semi-Lagrangian (RK2 from each destination point), so tile t can hold the
 * surface only if the departure box of its points holds a current interface
 * owner. The box is traced back one frame at a time: each RK2 stage takes
 * signed velocity bounds over just the tiles its sample points can read
 * (the box so far, or its midpoint box), of the extended field the next
 * frame traces (the host re-runs that extension first); a 3D prefix sum
 * answers the interface query. Still liquid keeps exactly its interface
 * tiles fine; moving liquid extends the set upstream by its local travel
 * only. Reads back one bit per tile. */
export class UniformMixedDynamicClassifier {
 readonly allocatedBytes:number;
 private readonly work:GPUBuffer;
 /** Cube table levels 1..CUBE_LEVELS, component-major over start tiles. */
 private readonly bounds:GPUBuffer;
 private readonly readback:GPUBuffer;
 private readonly params:GPUBuffer;
 /** One bit per tile: solid-coupled (uniformMixedSolidTiles().coupled),
  * plus the tiles moving bodies sweep (setBodies), rewritten each census. */
 private readonly solidTiles:GPUBuffer;
 /** The static coupled bits setSolid wrote, restored under the body bits. */
 private readonly staticSolidTiles:GPUBuffer;
 /** Ors the tiles moving bodies sweep into solidTiles before a census. */
 private bodies?:(encoder:GPUCommandEncoder,tiles:GPUBuffer)=>void;
 /** solidTiles holds body bits the static mask lacks. */
 private bodyBits=false;
 private readonly resources:GPUBindGroupLayout;
 private readonly group:GPUBindGroup;
 private readonly pipelines=new Map<string,GPUComputePipeline>();
 private readonly words:number;
 /** setSolid marked a coupled tile: the promotion passes run. */
 private solid=false;
 private encoded=false;
 /** The band decided by the last encode, for the GPU layout builder. */
 get bandBits():UniformMixedBandBits{return {buffer:this.work,wordOffset:HEADER};}
 constructor(private readonly device:GPUDevice,readonly ownership:UniformMixedOwnership,volume:GPUTexture,phi:GPUTexture,extended:GPUTexture){
  const tiles=ownership.layout.tiles.length;
  this.words=Math.ceil(tiles/32);
  const t=ownership.layout.lattice.dimensions.map(n=>n/4);
  const workBytes=(HEADER+this.words+6*tiles+(t[0]!+1)*(t[1]!+1)*(t[2]!+1)+2*tiles+2*this.words)*4,readBytes=(HEADER+this.words)*4;
  this.work=device.createBuffer({label:"Uniform dynamic ownership census",size:workBytes,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
  this.readback=device.createBuffer({label:"Uniform dynamic ownership readback",size:readBytes,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  this.params=device.createBuffer({label:"Uniform dynamic ownership policy",size:64,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
  this.solidTiles=device.createBuffer({label:"Uniform dynamic ownership solid tiles",size:this.words*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
  this.staticSolidTiles=device.createBuffer({label:"Uniform dynamic ownership static solid tiles",size:this.words*4,usage:GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
  const boundsBytes=6*CUBE_LEVELS*tiles*4;
  this.bounds=device.createBuffer({label:"Uniform dynamic ownership bound cubes",size:boundsBytes,usage:GPUBufferUsage.STORAGE});
  this.allocatedBytes=workBytes+boundsBytes+readBytes+64+this.words*8;
  this.resources=device.createBindGroupLayout({entries:[
   ...[0,1,2].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float" as const,viewDimension:"3d" as const}})),
   {binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},
   {binding:4,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},
   {binding:5,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},
   {binding:6,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},
  ]});
  this.group=device.createBindGroup({layout:this.resources,entries:[
   {binding:0,resource:volume.createView()},{binding:1,resource:phi.createView()},
   {binding:2,resource:extended.createView()},{binding:3,resource:{buffer:this.work}},{binding:4,resource:{buffer:this.params}},
   {binding:5,resource:{buffer:this.solidTiles}},{binding:6,resource:{buffer:this.bounds}},
  ]});
 }
 async initialize():Promise<void>{
  const h=this.ownership.layout.lattice.cellSize_m,cap=UNIFORM_MIXED_DYNAMIC_DISTANCE_CAP;
  const module=this.device.createShaderModule({label:"Uniform dynamic ownership census",code:uniformMixedTopologyWGSL(this.ownership.layout,0)+/* wgsl */`
@group(1) @binding(0) var volume:texture_3d<f32>;
@group(1) @binding(1) var phi:texture_3d<f32>;
@group(1) @binding(2) var velocity:texture_3d<f32>;
@group(1) @binding(3) var<storage,read_write> census:array<atomic<u32>>;
// step: horizon dt, surface tolerance, full and empty tolerance; surface:
// fast travel, boundary travel, frames in the horizon; flow: the velocity
// gravity adds per frame (m/s), and the departure margin in cells.
struct DynamicPolicy {step:vec4f,reach:vec4u,surface:vec4f,flow:vec4f}
@group(1) @binding(4) var<uniform> policy:DynamicPolicy;
@group(1) @binding(5) var<storage,read> solidTiles:array<u32>;
@group(1) @binding(6) var<storage,read_write> bounds:array<u32>;
// The census tail follows the frame's last phi resolve: hanging texels hold
// umVertexValue, so vertex reads are direct loads.
fn umLoadVertex(p:vec3u)->f32{return textureLoad(phi,vec3i(p),0).x;}
${uniformMixedVertexSamplingSource("",true)}
const CAP:u32=${cap}u;
// Largest travel, in h per step, boundary tiles dilate by (a 4-tile radius).
const BOUNDARY_TRAVEL_CAP:u32=16u;
const WORDS:u32=${this.words}u;
const H=vec3f(${h.join(",")});
// Per tile: ordered keys of the signed minimum and maximum face velocity per axis.
fn boundIndex(t:u32,k:u32)->u32{return ${HEADER}u+WORDS+6u*t+k;}
// Cube table: level j in 1..CUBE_LEVELS holds, per start tile s, the keys of
// the cube [s, s+2^j)³ clipped to the domain.
const CUBE_LEVELS:u32=${CUBE_LEVELS}u;
fn cubeIndex(j:u32,k:u32,s:vec3i)->u32{return ((j-1u)*6u+k)*UM_TILES+umTileAt(vec3u(s));}
// Inclusive prefix sum of interface flags over tiles, with a zero border plane.
const PX:u32=UM_T.x+1u;const PY:u32=UM_T.y+1u;const PZ:u32=UM_T.z+1u;
fn prefixIndex(p:vec3u)->u32{return ${HEADER}u+WORDS+6u*UM_TILES+p.x+PX*(p.y+PY*p.z);}
// Per required tile: cells between each face and the nearest crossing owner,
// one nibble per face (-x,-y,-z,+x,+y,+z), and its own travel in whole h
// cells per step in the top byte; all ones for a tile that is not required.
fn gapIndex(t:u32)->u32{return prefixIndex(vec3u(0u))+PX*PY*PZ+t;}
// Per boundary tile: travel in h per step along each direction (-x,-y,-z,
// +x,+y,+z), five bits each; all ones for any other tile. Its dilation
// follows the flow, not a sphere: a climbing sheet refines the tiles above.
fn travelIndex(t:u32)->u32{return gapIndex(UM_TILES)+t;}
// One bit per tile: liquid, a non-air owner (wet); then solid-coupled tiles
// within one tile of liquid or of the decided band (active).
fn wetIndex(w:u32)->u32{return travelIndex(UM_TILES)+w;}
fn activeIndex(w:u32)->u32{return wetIndex(WORDS)+w;}
fn orderKey(x:f32)->u32{let b=bitcast<u32>(x);return select(b|0x80000000u,~b,(b&0x80000000u)!=0u);}
fn orderValue(k:u32)->f32{return bitcast<f32>(select(~k,k&0x7fffffffu,(k&0x80000000u)!=0u));}
// One tile's classification: ordered velocity keys, nibble distances from
// each face to the nearest crossing owner (gap) and surface owner (reach),
// flags (bit 0 surface owner, bit 1 phi sign change) and 4h surface error.
struct TileClass{low:array<u32,3>,high:array<u32,3>,gap:array<u32,6>,reach:array<u32,6>,flags:u32,error:u32}
fn umEmptyClass()->TileClass{return TileClass(array<u32,3>(0xffffffffu,0xffffffffu,0xffffffffu),array<u32,3>(0u,0u,0u),array<u32,6>(15u,15u,15u,15u,15u,15u),array<u32,6>(15u,15u,15u,15u,15u,15u),0u,0u);}
var<workgroup> mixedTile:atomic<u32>;
var<workgroup> tileLow:array<atomic<u32>,3>;
var<workgroup> tileHigh:array<atomic<u32>,3>;
var<workgroup> tileGap:array<atomic<u32>,6>;
var<workgroup> tileReach:array<atomic<u32>,6>;
var<workgroup> tileError:atomic<u32>;
const UM_H:f32=${Math.min(...h)};
const MAX_H:f32=${Math.max(...h)};
// Surface error of the tile at 4h, in h (non-negative, ordered as bits).
fn umResolutionError(tile:u32,lane:u32,width:u32)->f32{
 let origin=umTileCoord(tile)*4u;var worst=0.0;
 if(width==1u){
  var corners:array<f32,8>;
  for(var k=0u;k<8u;k++){corners[k]=umVertexValue(origin+umCorner(k,2u)*4u);}
  for(var i=lane;i<125u;i+=64u){
   let v=umCorner(i,5u);let t=vec3f(v)*0.25;
   let x0=mix(vec4f(corners[0],corners[2],corners[4],corners[6]),vec4f(corners[1],corners[3],corners[5],corners[7]),t.x);
   let y0=mix(vec2f(x0.x,x0.z),vec2f(x0.y,x0.w),t.y);
   let interpolated=mix(y0.x,y0.y,t.z);let value=umVertexValue(origin+v);
   if(min(abs(value),abs(interpolated))<2.0*UM_H){worst=max(worst,abs(value-interpolated)/UM_H);}
  }
  return worst;
 }
 if(width!=4u){return 3.0e38;}
 if(lane>=8u){return 0.0;}
 // Quadratic interpolation error at the midpoint is |second difference|/8;
 // halved so one tolerance gives refine-at-2x hysteresis.
 let c=origin+umCorner(lane,2u)*4u;let centre=umVertexValue(c);
 if(abs(centre)>=8.0*UM_H){return 0.0;}
 for(var a=0u;a<3u;a++){
  if(c[a]<4u||c[a]+4u>UM_D[a]){continue;}
  var lo=c;lo[a]-=4u;var hi=c;hi[a]+=4u;
  worst=max(worst,abs(umVertexValue(hi)-2.0*centre+umVertexValue(lo))/(16.0*UM_H));
 }
 return worst;
}
fn umSolidTile(p:vec3u)->bool{let t=umTileAt(p);return ((solidTiles[t/32u]>>(t%32u))&1u)!=0u;}
// Boundary rules (UniformMixedDynamicPolicy.boundaryTravel), from the tile's
// own velocity bounds and the cells between its faces and its surface owners.
// Travel thresholds halve for an h tile (hysteresis) and its margin doubles.
fn umDirectionTravel(c:TileClass,k:u32)->f32{
 let a=k%3u;
 if(k<3u){return max(-orderValue(c.low[a]),0.0)*policy.step.x/H[a];}
 return max(orderValue(c.high[a]),0.0)*policy.step.x/H[a];
}
fn umBoundaryRequired(p:vec3u,width:u32,c:TileClass)->bool{
 let threshold=policy.surface.y*select(1.0,0.5,width==1u);
 if(threshold<=0.0){return false;}
 let margin=4.0*select(1.0,2.0,width==1u);
 var travel:array<f32,6>;
 for(var k=0u;k<6u;k++){travel[k]=umDirectionTravel(c,k);}
 // Impact: surface liquid within the margin of a closed wall or solid tile,
 // moving toward it faster than along it. A run-out skimming the floor, or a
 // front passing a wall, is not redirected by it. Contact, not predicted
 // arrival: refining a fast front a step or two early (reach + travel) held
 // the long-dam toe short of the wall for three steps.
 for(var k=0u;k<6u;k++){
  let a=k%3u;let high=k>=3u;
  let along=max(max(travel[(a+1u)%3u],travel[3u+(a+1u)%3u]),max(travel[(a+2u)%3u],travel[3u+(a+2u)%3u]));
  if(travel[k]<max(threshold,along)){continue;}
  let reach=margin-f32(c.reach[k]);
  if(reach<0.0){continue;}
  let beyond=select(p[a],UM_T[a]-1u-p[a],high);
  if(((policy.reach.z>>k)&1u)!=0u&&4.0*f32(beyond)<=reach){return true;}
  for(var n=1u;n<=min(beyond,CAP)&&4.0*f32(n-1u)<=reach;n++){
   var q=p;q[a]=select(p[a]-n,p[a]+n,high);
   if(umSolidTile(q)){return true;}
  }
 }
 // Lift: on a closed vertical wall, moving up faster than along the wall.
 if(policy.reach.w==0u){return false;}
 let up=select(travel[1],travel[4],policy.reach.w==1u);
 if(up<threshold){return false;}
 for(var k=0u;k<6u;k++){
  let a=k%3u;if(a==1u){continue;}
  let beyond=select(p[a],UM_T[a]-1u-p[a],k>=3u);
  let along=max(travel[2u-a],travel[5u-a]);
  if(((policy.reach.z>>k)&1u)!=0u&&beyond==0u&&up>=along){return true;}
 }
 return false;
}
// Interface: an owner whose corner phi changes sign. V/phi disagreement
// without one is counted, not refined: 4h V error is at or below fine on the
// same cells (plan, same-cell A/B).
fn umClassifyOwner(tile:u32,width:u32,lane:u32,c:ptr<function,TileClass>){
 let side=4u/width;
 let origin=umTileCoord(tile)*4u+umCorner(lane,side)*width;
 let owner=umOwnerAt(vec3i(origin));
 let v=textureLoad(volume,vec3i(origin),0).x;
 var inside=0u;var deep=0u;
 for(var k=0u;k<8u;k++){
  let value=umVertexValue(origin+umCorner(k,2u)*width);
  if(value<0.0){inside++;}
  if(value< -2.0*MAX_H*f32(width)){deep++;}
 }
 // Signed bounds of the extended face velocities the next trace samples.
 for(var axis=0u;axis<3u;axis++){
  var low=3.0e38;var high=-3.0e38;
  for(var sign=-1;sign<=1;sign+=2){
   let first=umFace(owner,axis,sign,0u);
   for(var part=0u;part<first.count;part++){
    let face=umFace(owner,axis,sign,part);if(face.anchor[axis]<0){continue;}
    let u=textureLoad(velocity,face.anchor,0)[axis];
    // A non-finite speed bounds nothing: saturate both ends.
    let finite=abs(u)<=3.0e38;
    low=min(low,select(-3.0e38,u,finite));high=max(high,select(3.0e38,u,finite));
   }
  }
  (*c).low[axis]=min((*c).low[axis],orderKey(low));(*c).high[axis]=max((*c).high[axis],orderKey(high));
 }
 // Largest interior deficit and largest air volume, per fine/coarse width.
 let coarse=select(0u,1u,width!=1u);
 if(deep==8u){atomicMax(&census[8u+coarse],bitcast<u32>(max(1.0-v,0.0)));}
 if(inside==0u){atomicMax(&census[10u+coarse],bitcast<u32>(max(v,0.0)));}
 let local=umCorner(lane,side)*width;
 if(inside!=0u&&inside!=8u){
  for(var a=0u;a<3u;a++){(*c).gap[a]=min((*c).gap[a],local[a]);(*c).gap[3u+a]=min((*c).gap[3u+a],4u-width-local[a]);}
 }
 let interior=inside==8u&&v>=1.0-policy.step.z;
 let air=inside==0u&&v<=policy.step.w;
 // Flag 4: liquid (any owner that is not air).
 if(!air){(*c).flags|=4u;}
 if(!interior&&!air){
  for(var a=0u;a<3u;a++){(*c).reach[a]=min((*c).reach[a],local[a]);(*c).reach[3u+a]=min((*c).reach[3u+a],4u-width-local[a]);}
  // Bit 2: the geometric surface itself (a phi sign change) is in this owner.
  (*c).flags|=select(1u,3u,inside!=0u&&inside!=8u);
  // Why a coarse owner is interface: V between the tolerances, or a phi sign change.
  if(width!=1u){
   if(v>policy.step.w&&v<1.0-policy.step.z){atomicAdd(&census[6],1u);}
   if(inside!=0u&&inside!=8u){atomicAdd(&census[7],1u);}
   if(inside==8u&&v<=policy.step.w){atomicAdd(&census[3],1u);}
  }
 }
}
fn umFinishTile(tile:u32,width:u32,c:TileClass){
 let crossing=(c.flags&2u)!=0u;
 // Travel of the tile's own extended faces over one step, in h.
 var speed=0.0;
 for(var a=0u;a<3u;a++){speed=max(speed,max(abs(orderValue(c.low[a])),abs(orderValue(c.high[a])))*policy.step.x/H[a]);}
 // Largest travel any bound box can hold, in decide's own arithmetic, so
 // the cube table builds only the levels decide can reach.
 var reach=0.0;let scale=policy.step.x/H;
 for(var a=0u;a<3u;a++){
  if(c.low[a]!=0xffffffffu){reach=max(reach,abs(orderValue(c.low[a])*scale[a]));}
  if(c.high[a]!=0u){reach=max(reach,abs(orderValue(c.high[a])*scale[a]));}
 }
 atomicMax(&census[18],bitcast<u32>(reach));
 // Hysteresis: a 4h tile returns to h only below half the travel.
 let fast=policy.surface.x>0.0&&speed>=policy.surface.x*select(0.5,1.0,width==1u);
 let shaped=crossing&&!fast&&!(policy.step.y>0.0&&bitcast<f32>(c.error)<=policy.step.y);
 // A boundary rule holds a phi surface whatever its speed. Partial V alone
 // does not qualify: airborne spray refined at the wall falls under the
 // dust threshold and is discarded (128³ dam: 9 cells in three steps).
 let bounded=!shaped&&crossing&&umBoundaryRequired(umTileCoord(tile),width,c);
 let required=shaped||bounded;
 atomicStore(&census[prefixIndex(umTileCoord(tile)+vec3u(1u))],select(0u,1u,required));
 if(required){atomicAdd(&census[12],1u);}
 if(bounded){atomicAdd(&census[16],1u);atomicMax(&census[17],u32(ceil(speed)));}
 var gap=0xffffffffu;
 if(required){gap=min(u32(ceil(speed)),255u)<<24u;for(var k=0u;k<6u;k++){gap|=c.gap[k]<<(4u*k);}}
 atomicStore(&census[gapIndex(tile)],gap);
 var directed=0xffffffffu;
 if(bounded){directed=0u;for(var k=0u;k<6u;k++){directed|=min(u32(ceil(umDirectionTravel(c,k))),31u)<<(5u*k);}}
 atomicStore(&census[travelIndex(tile)],directed);
 for(var a=0u;a<3u;a++){atomicStore(&census[boundIndex(tile,a)],c.low[a]);atomicStore(&census[boundIndex(tile,3u+a)],c.high[a]);}
 if(crossing){atomicAdd(&census[0],1u);}
 if((c.flags&4u)!=0u){atomicOr(&census[wetIndex(tile/32u)],1u<<(tile%32u));}
 // A 4h surface the lattice cannot resolve: refined by this census.
 if(required&&width!=1u){let slot=atomicAdd(&census[1],1u);if(slot<3u){atomicStore(&census[13u+slot],tile);}}
}
// h tiles: one workgroup per tile, one lane per owner.
@compute @workgroup_size(64) fn classify(@builtin(workgroup_id) gid:vec3u,@builtin(local_invocation_index) lane:u32){
 let job=gid.x+umDispatchX*gid.y;if(job>=umCounts.x){return;}
 let tile=umTopology[UM_TILES+job];
 if(lane==0u){atomicStore(&mixedTile,0u);for(var a=0u;a<3u;a++){atomicStore(&tileLow[a],0xffffffffu);atomicStore(&tileHigh[a],0u);atomicStore(&tileGap[a],15u);atomicStore(&tileGap[3u+a],15u);atomicStore(&tileReach[a],15u);atomicStore(&tileReach[3u+a],15u);}atomicStore(&tileError,0u);}workgroupBarrier();
 let width=umTileWidth(tile);let side=4u/width;
 if(lane<side*side*side){
  var c=umEmptyClass();umClassifyOwner(tile,width,lane,&c);
  for(var a=0u;a<3u;a++){atomicMin(&tileLow[a],c.low[a]);atomicMax(&tileHigh[a],c.high[a]);}
  for(var k=0u;k<6u;k++){atomicMin(&tileGap[k],c.gap[k]);atomicMin(&tileReach[k],c.reach[k]);}
  atomicOr(&mixedTile,c.flags);
 }
 workgroupBarrier();
 if((atomicLoad(&mixedTile)&2u)!=0u&&policy.step.y>0.0){atomicMax(&tileError,bitcast<u32>(umResolutionError(tile,lane,width)));}
 workgroupBarrier();
 if(lane==0u){
  var c=umEmptyClass();
  for(var a=0u;a<3u;a++){c.low[a]=atomicLoad(&tileLow[a]);c.high[a]=atomicLoad(&tileHigh[a]);}
  for(var k=0u;k<6u;k++){c.gap[k]=atomicLoad(&tileGap[k]);c.reach[k]=atomicLoad(&tileReach[k]);}
  c.flags=atomicLoad(&mixedTile);c.error=atomicLoad(&tileError);
  umFinishTile(tile,width,c);
 }
}
// Single-owner (4h) tiles: one lane per tile. A workgroup per tile left 63
// of its 64 lanes idle over most of the domain.
@compute @workgroup_size(64) fn classifyCoarse(@builtin(global_invocation_id) gid:vec3u){
 let index=gid.x+umDispatchX*64u*gid.y;if(index>=umCounts.y){return;}
 let tile=umTopology[UM_TILES+umCounts.x+index];
 var c=umEmptyClass();umClassifyOwner(tile,4u,0u,&c);
 if((c.flags&2u)!=0u&&policy.step.y>0.0){var error=0.0;for(var l=0u;l<8u;l++){error=max(error,umResolutionError(tile,l,4u));}c.error=bitcast<u32>(error);}
 umFinishTile(tile,4u,c);
}
// Separable inclusive prefix sum: one lane per line of the (T+1)³ table.
${[0,1,2].map(axis=>{const [a,b]=[0,1,2].filter(k=>k!==axis);return /* wgsl */`
@compute @workgroup_size(64) fn prefix${axis}(@builtin(global_invocation_id) gid:vec3u){
 let line=gid.x+umDispatchX*64u*gid.y;let extent=vec3u(PX,PY,PZ);
 if(line>=extent[${a}]*extent[${b}]){return;}
 var p=vec3u(0u);p[${a}]=line%extent[${a}];p[${b}]=line/extent[${a}];var sum=0u;
 for(var i=1u;i<extent[${axis}];i++){p[${axis}]=i;sum+=atomicLoad(&census[prefixIndex(p)]);atomicStore(&census[prefixIndex(p)],sum);}
}`;}).join("\n")}
// Cube levels 1..topCube() are built: no box query needs a larger side.
override cubeLevel:u32=1u;
// The largest box side, in tiles, a departure query can take: the tile, the
// sampling reach and drift margin on each side, the travel of the fastest
// tile over the horizon (census[18]) and gravity's.
fn topCube()->u32{
 let steps=max(1.0,policy.surface.z);
 let gravity=0.5*steps*(steps-1.0)*length(policy.flow.xyz)*policy.step.x/(steps*UM_H);
 let travel=min(bitcast<f32>(atomicLoad(&census[18]))+gravity+SAMPLE_REACH+policy.flow.w,1.0e6);
 let side=2u+u32(ceil((4.0+2.0*travel)/4.0));
 return clamp(firstLeadingBit(side),1u,CUBE_LEVELS);
}
fn identityKeys()->array<u32,6>{return array<u32,6>(0xffffffffu,0xffffffffu,0xffffffffu,0u,0u,0u);}
// Cube level j from eight cubes of level j-1 (level 0: the tiles), for the
// levels a query can reach. One lane per start tile.
@compute @workgroup_size(64) fn boundCube(@builtin(global_invocation_id) gid:vec3u){
 let index=gid.x+umDispatchX*64u*gid.y;if(index>=UM_TILES){return;}
 if(cubeLevel>topCube()){return;}
 let s=vec3i(umTileCoord(index));
 let half=i32(1u<<(cubeLevel-1u));var keys=identityKeys();
 for(var o=0u;o<8u;o++){
  let q=s+half*vec3i(vec3u(o&1u,(o>>1u)&1u,o>>2u));
  if(any(q>=vec3i(UM_T))){continue;}
  if(cubeLevel==1u){
   let t=umTileAt(vec3u(q));
   for(var k=0u;k<3u;k++){keys[k]=min(keys[k],atomicLoad(&census[boundIndex(t,k)]));keys[3u+k]=max(keys[3u+k],atomicLoad(&census[boundIndex(t,3u+k)]));}
  }else{
   for(var k=0u;k<3u;k++){keys[k]=min(keys[k],bounds[cubeIndex(cubeLevel-1u,k,q)]);keys[3u+k]=max(keys[3u+k],bounds[cubeIndex(cubeLevel-1u,3u+k,q)]);}
  }
 }
 for(var k=0u;k<6u;k++){bounds[cubeIndex(cubeLevel,k,s)]=keys[k];}
}
// Keys of the inclusive tile box [a, b], clipped to the domain: cubes of one
// level tile it, per axis from its low end with the last one flush with its
// high end, or (an axis shorter than the side) one cube overhanging it.
fn rangeKeys(a:vec3i,b:vec3i,top:u32)->array<u32,6>{
 var keys=identityKeys();
 let lo=max(a,vec3i(0));let hi=min(b,vec3i(UM_T)-vec3i(1));
 if(any(hi<lo)){return keys;}
 let e=vec3u(hi-lo)+vec3u(1u);
 var j=min(firstLeadingBit(min(e.x,min(e.y,e.z))),top);
 loop{
  let n=(e+vec3u((1u<<j)-1u))>>vec3u(j);
  if(n.x*n.y*n.z<=${BOX_CUBES}u||j>=top){break;}
  j++;
 }
 let side=i32(1u<<j);let n=vec3i((e+vec3u(u32(side)-1u))>>vec3u(j));let last=max(hi-vec3i(side-1),lo);
 for(var z=0;z<n.z;z++){for(var y=0;y<n.y;y++){for(var x=0;x<n.x;x++){
  let q=min(lo+side*vec3i(x,y,z),last);
  if(j==0u){
   let t=umTileAt(vec3u(q));
   for(var k=0u;k<3u;k++){keys[k]=min(keys[k],atomicLoad(&census[boundIndex(t,k)]));keys[3u+k]=max(keys[3u+k],atomicLoad(&census[boundIndex(t,3u+k)]));}
  }else{
   for(var k=0u;k<3u;k++){keys[k]=min(keys[k],bounds[cubeIndex(j,k,q)]);keys[3u+k]=max(keys[3u+k],bounds[cubeIndex(j,3u+k,q)]);}
  }
 }}}
 return keys;
}
// Cells past a point whose face velocities a sample there reads: one at h,
// two across 4h owners (their centres are two cells in).
const SAMPLE_REACH:f32=2.0;
// Signed bounds of every velocity sampled in the cell box [lo, hi] after
// gravity adds shift, joined with zero: sampling near a wall or solid blends
// in its zero face, and the trace stops short at a solid or the domain clamp.
struct Flow{low:vec3f,high:vec3f}
fn sampledFlow(lo:vec3f,hi:vec3f,shift:vec3f,top:u32)->Flow{
 let keys=rangeKeys(vec3i(floor((lo-SAMPLE_REACH)/4.0)),vec3i(floor((hi+SAMPLE_REACH)/4.0)),top);
 var f=Flow(vec3f(0),vec3f(0));
 for(var a=0u;a<3u;a++){
  if(keys[a]!=0xffffffffu){f.low[a]=orderValue(keys[a]);}
  if(keys[3u+a]!=0u){f.high[a]=orderValue(keys[3u+a]);}
 }
 f.low=min(vec3f(0),f.low+min(shift,vec3f(0)));f.high=max(vec3f(0),f.high+max(shift,vec3f(0)));
 return f;
}
fn interfaceTilesIn(low:vec3i,high:vec3i)->u32 {
 let a=vec3u(clamp(low,vec3i(0),vec3i(UM_T)));let b=vec3u(clamp(high+vec3i(1),vec3i(0),vec3i(UM_T)));
 if(any(b<=a)){return 0u;}
 let s=atomicLoad(&census[prefixIndex(b)])-atomicLoad(&census[prefixIndex(vec3u(a.x,b.y,b.z))])-atomicLoad(&census[prefixIndex(vec3u(b.x,a.y,b.z))])-atomicLoad(&census[prefixIndex(vec3u(b.x,b.y,a.z))]);
 return s+atomicLoad(&census[prefixIndex(vec3u(a.x,a.y,b.z))])+atomicLoad(&census[prefixIndex(vec3u(a.x,b.y,a.z))])+atomicLoad(&census[prefixIndex(vec3u(b.x,a.y,a.z))])-atomicLoad(&census[prefixIndex(a)]);
}
// Predicted surface tile: the departure box of t's points, traced back over
// the horizon's frames, meets an interface owner (see decide).
// Forward reach (fastTravel > 0): a required tile's surface moves by at most
// its own travel (below fastTravel, except for boundary tiles). Tile p is fine if a required tile q
// within that travel (from q's nearest crossing owner) plus the margin
// reaches it; a still surface keeps just the tiles whose vertices it touches.
fn forwardFine(p:vec3i,margin:i32)->bool{
 // Boundary tiles are fast by design: their radius is their own travel
 // (census[17], capped), or a climbing sheet outruns its h tiles.
 let fastest=max(policy.surface.x,f32(min(atomicLoad(&census[17]),BOUNDARY_TRAVEL_CAP)));
 let radius=i32(ceil(fastest/4.0))+margin;
 let a=max(p-vec3i(radius),vec3i(0));let b=min(p+vec3i(radius),vec3i(UM_T)-vec3i(1));
 for(var z=a.z;z<=b.z;z++){for(var y=a.y;y<=b.y;y++){for(var x=a.x;x<=b.x;x++){
  let q=vec3i(x,y,z);let word=atomicLoad(&census[gapIndex(umTileAt(vec3u(q)))]);
  if(word==0xffffffffu){continue;}
  let directed=atomicLoad(&census[travelIndex(umTileAt(vec3u(q)))]);
  let slack=4.0*f32(margin)+1e-3;let isotropic=f32(word>>24u);var reaches=true;
  for(var axis=0u;axis<3u;axis++){
   // Travel toward p along this axis: the tile's own speed, or its directed travel.
   let toward=select(axis,3u+axis,q[axis]<p[axis]);
   let travel=select(f32((directed>>(5u*toward))&31u),isotropic,directed==0xffffffffu)+slack;
   if(q[axis]<p[axis]){reaches=reaches&&f32((word>>(4u*(3u+axis)))&15u)+4.0*f32(p[axis]-q[axis]-1)<=travel;}
   if(q[axis]>p[axis]){reaches=reaches&&f32((word>>(4u*axis))&15u)+4.0*f32(q[axis]-p[axis]-1)<=travel;}
  }
  if(reaches){return true;}
 }}}
 return false;
}
// Whether the cell box [lo, hi] (closed) holds a current interface owner. A
// crossing tile q counts only if its nearest crossing owner lies within the
// depth the box reaches into q on every axis: a still surface keeps just the
// tiles whose vertices it touches, not a 26-tile shell. Tiles inside the box
// on every axis reach it outright; only the shell is tested owner-wise.
fn departureMeetsSurface(p:vec3i,lo:vec3f,hi:vec3f)->bool{
 let first=vec3i(floor((lo-1e-3)/4.0));let last=vec3i(floor((hi+1e-3)/4.0));
 if(interfaceTilesIn(first,last)==0u){return false;}
 let inner=vec3i(ceil((lo-1e-3)/4.0));let innerLast=last-vec3i(1);
 if(interfaceTilesIn(inner,innerLast)!=0u){return true;}
 let a=max(first,vec3i(0));let b=min(last,vec3i(UM_T)-vec3i(1));
 for(var z=a.z;z<=b.z;z++){for(var y=a.y;y<=b.y;y++){
  if(interfaceTilesIn(vec3i(a.x,y,z),vec3i(b.x,y,z))==0u){continue;}
  let core=y>=inner.y&&y<=innerLast.y&&z>=inner.z&&z<=innerLast.z;
  for(var x=a.x;x<=b.x;x++){
   if(core&&x>=inner.x&&x<=innerLast.x){x=innerLast.x;continue;}
   let q=vec3i(x,y,z);let gap=atomicLoad(&census[gapIndex(umTileAt(vec3u(q)))]);
   if(gap==0xffffffffu){continue;}
   var reaches=true;
   for(var axis=0u;axis<3u;axis++){
    if(q[axis]<p[axis]){reaches=reaches&&f32((gap>>(4u*(3u+axis)))&15u)<=4.0*f32(q[axis])+4.0-lo[axis]+1e-3;}
    if(q[axis]>p[axis]){reaches=reaches&&f32((gap>>(4u*axis))&15u)<=hi[axis]-4.0*f32(q[axis])+1e-3;}
   }
   if(reaches){return true;}
  }
 }}
 return false;
}
@compute @workgroup_size(64) fn decide(@builtin(global_invocation_id) gid:vec3u){
 let tile=gid.x+umDispatchX*64u*gid.y;if(tile>=UM_TILES){return;}
 let p=vec3i(umTileCoord(tile));let scale=policy.step.x/H;
 if(policy.surface.x>0.0){
  let width=umTileWidth(tile);
  let fine=forwardFine(p,i32(policy.reach.x)+select(0,i32(policy.reach.y),width==1u));
  if(!fine){if(width==1u){atomicAdd(&census[5],1u);}return;}
  atomicAdd(&census[2],1u);if(width!=1u){atomicAdd(&census[4],1u);}
  atomicOr(&census[${HEADER}u+tile/32u],1u<<(tile%32u));
  return;
 }
 // A current interface tile stays fine: phi does not only move with the
 // flow (residual sheets left behind a falling surface persist in place).
 let width=umTileWidth(tile);
 var fine=interfaceTilesIn(p,p)!=0u;
 if(!fine){
  // Trace the tile's points back one frame at a time, newest frame first, as
  // RK2 does (x - dt·u(x - dt/2·u(x)), clamped to the domain): each stage
  // bounds only the velocities its own sample box reads, so a tile's box
  // follows its flow instead of every speed within a radius of it. The
  // newer frames trace a velocity gravity has changed since this census.
  let steps=max(1u,u32(policy.surface.z));let s=scale/f32(steps);let top=topCube();let D=vec3f(UM_D);
  var lo=4.0*vec3f(p);var hi=lo+vec3f(4.0);
  for(var i=0u;i<steps;i++){
   let shift=policy.flow.xyz*f32(steps-1u-i);
   let start=sampledFlow(lo,hi,shift,top);
   let mid=sampledFlow(clamp(lo-0.5*start.high*s,vec3f(0),D),clamp(hi-0.5*start.low*s,vec3f(0),D),shift,top);
   lo=clamp(lo-mid.high*s,vec3f(0),D);hi=clamp(hi-mid.low*s,vec3f(0),D);
  }
  // Closed: a boundary vertex belongs to both tiles it separates.
  let margin=policy.flow.w+4.0*f32(i32(policy.reach.x)+select(0,i32(policy.reach.y),width==1u));
  fine=departureMeetsSurface(p,lo-vec3f(margin),hi+vec3f(margin));
 }
 if(!fine){if(width==1u){atomicAdd(&census[5],1u);}return;}
 atomicAdd(&census[2],1u);if(width!=1u){atomicAdd(&census[4],1u);}
 atomicOr(&census[${HEADER}u+tile/32u],1u<<(tile%32u));
}
// Liquid-conditional solid promotion. Fine-owner solid terms need a cut
// tile, and each neighbour of it, at h wherever liquid can meet it; a dry cut
// tile far from liquid runs 4h without them. Liquid reaches a coupled tile
// only from a wet tile or a band tile (the surface this census predicts over
// its horizon) within one tile of it: that tile is active, and it and its
// 26 neighbours join the band. The band certificate fails a liquid row in
// a cut tile the simulation holds at 4h.
fn umBit(base:u32,t:u32)->bool{return (atomicLoad(&census[base+t/32u])&(1u<<(t%32u)))!=0u;}
@compute @workgroup_size(64) fn solidActive(@builtin(global_invocation_id) gid:vec3u){
 let tile=gid.x+umDispatchX*64u*gid.y;if(tile>=UM_TILES){return;}
 if(((solidTiles[tile/32u]>>(tile%32u))&1u)==0u){return;}
 let p=vec3i(umTileCoord(tile));
 let a=max(p-vec3i(1),vec3i(0));let b=min(p+vec3i(1),vec3i(UM_T)-vec3i(1));
 for(var z=a.z;z<=b.z;z++){for(var y=a.y;y<=b.y;y++){for(var x=a.x;x<=b.x;x++){
  let q=umTileAt(vec3u(vec3i(x,y,z)));
  if(umBit(wetIndex(0u),q)||umBit(${HEADER}u,q)){atomicOr(&census[activeIndex(tile/32u)],1u<<(tile%32u));return;}
 }}}
}
@compute @workgroup_size(64) fn solidPromote(@builtin(global_invocation_id) gid:vec3u){
 let tile=gid.x+umDispatchX*64u*gid.y;if(tile>=UM_TILES){return;}
 if(umBit(${HEADER}u,tile)){return;}
 let p=vec3i(umTileCoord(tile));
 let a=max(p-vec3i(1),vec3i(0));let b=min(p+vec3i(1),vec3i(UM_T)-vec3i(1));
 for(var z=a.z;z<=b.z;z++){for(var y=a.y;y<=b.y;y++){for(var x=a.x;x<=b.x;x++){
  if(!umBit(activeIndex(0u),umTileAt(vec3u(vec3i(x,y,z))))){continue;}
  atomicOr(&census[${HEADER}u+tile/32u],1u<<(tile%32u));
  atomicAdd(&census[2],1u);atomicAdd(&census[19],1u);
  if(umTileWidth(tile)==1u){atomicSub(&census[5],1u);}else{atomicAdd(&census[4],1u);}
  return;
 }}}
}
`});
  const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");
  if(errors.length)throw new Error(errors.map(m=>`${m.lineNum}: ${m.message}`).join("\n"));
  const layout=this.device.createPipelineLayout({bindGroupLayouts:[this.ownership.bindLayout,this.resources]});
  await Promise.all(["classify","classifyCoarse","prefix0","prefix1","prefix2","decide","solidActive","solidPromote"].map(async entryPoint=>{this.pipelines.set(entryPoint,await this.device.createComputePipelineAsync({layout,compute:{module,entryPoint,constants:{umDispatchX:this.ownership.dispatchX}}}));}));
  await Promise.all(Array.from({length:CUBE_LEVELS},async(_,i)=>{const level=i+1;
   this.pipelines.set(`cube${level}`,await this.device.createComputePipelineAsync({layout,compute:{module,entryPoint:"boundCube",constants:{umDispatchX:this.ownership.dispatchX,cubeLevel:level}}}));}));
 }
 /** Solid-coupled tiles (uniformMixedSolidTiles().coupled), for the impact rule. */
 setSolid(coupled:Uint8Array):void{
  const tiles=this.ownership.layout.tiles.length;
  if(coupled.length!==tiles)throw new Error(`Dynamic ownership solid mask has ${coupled.length} tiles, expected ${tiles}`);
  const bits=new Uint32Array(this.words);
  for(let t=0;t<tiles;t++)if(coupled[t])bits[t>>5]!|=1<<(t&31);
  this.device.queue.writeBuffer(this.solidTiles,0,bits);this.device.queue.writeBuffer(this.staticSolidTiles,0,bits);
  this.solid=coupled.some(c=>c!==0);this.bodyBits=false;
 }
 /** Moving bodies: before each census, `mark` ors the tiles they may cut
  * over the next frame into the coupled mask (on the GPU, from the body
  * state), and the liquid-conditional promotion passes run. undefined
  * restores the static mask. */
 setBodies(mark?:(encoder:GPUCommandEncoder,tiles:GPUBuffer)=>void):void{this.bodies=mark;}
 /** Encode after a completed frame, while its ownership and the local speed
  * velocity (the one the next frame advects with) are still in place. */
 encode(encoder:GPUCommandEncoder,policy:UniformMixedDynamicPolicy):void{
  if(this.pipelines.size!==8+CUBE_LEVELS)throw new Error("Dynamic ownership census is not initialized");
  for(const [name,value] of Object.entries({dt:policy.dt,fullTolerance:policy.fullTolerance,emptyTolerance:policy.emptyTolerance}))
   if(!Number.isFinite(value)||value<0)throw new Error(`Dynamic ownership ${name} must be finite and non-negative: ${value}`);
  for(const [name,value] of Object.entries({reach:policy.reach,hysteresis:policy.hysteresis}))
   if(!Number.isSafeInteger(value)||value<0||value>UNIFORM_MIXED_DYNAMIC_DISTANCE_CAP)throw new Error(`Dynamic ownership ${name} must be an integer in 0..${UNIFORM_MIXED_DYNAMIC_DISTANCE_CAP}: ${value}`);
  if(!Number.isFinite(policy.surfaceTolerance)||policy.surfaceTolerance<0)throw new Error(`Dynamic ownership surfaceTolerance must be finite and non-negative: ${policy.surfaceTolerance}`);
  this.device.queue.writeBuffer(this.params,0,new Float32Array([policy.dt,policy.surfaceTolerance,policy.fullTolerance,policy.emptyTolerance]));
  if(!Number.isSafeInteger(policy.closedWalls)||policy.closedWalls<0||policy.closedWalls>63)throw new Error(`Dynamic ownership closedWalls must be a 6-bit mask: ${policy.closedWalls}`);
  if(policy.up!==1&&policy.up!==-1&&policy.up!==0)throw new Error(`Dynamic ownership up must be 1, -1 or 0: ${policy.up}`);
  this.device.queue.writeBuffer(this.params,16,new Uint32Array([policy.reach,policy.hysteresis,policy.closedWalls,policy.up===1?1:policy.up===-1?2:0]));
  for(const [name,value] of Object.entries({fastTravel:policy.fastTravel,boundaryTravel:policy.boundaryTravel}))
   if(!Number.isFinite(value)||value<0)throw new Error(`Dynamic ownership ${name} must be finite and non-negative: ${value}`);
  if(!Number.isSafeInteger(policy.steps)||policy.steps<1)throw new Error(`Dynamic ownership steps must be a positive integer: ${policy.steps}`);
  if(policy.gravity.length!==3||!policy.gravity.every(Number.isFinite))throw new Error(`Dynamic ownership gravity must be three finite components: ${policy.gravity}`);
  const frame=policy.dt/policy.steps;
  this.device.queue.writeBuffer(this.params,32,new Float32Array([policy.fastTravel,policy.boundaryTravel,policy.steps,0,...policy.gravity.map(g=>g*frame),UNIFORM_MIXED_DYNAMIC_SURFACE_DRIFT]));
  // Cube levels up to the one decide can reach are written in full before
  // decide reads them; the census is cleared whole.
  encoder.clearBuffer(this.work);
  if(this.bodies||this.bodyBits){
   encoder.copyBufferToBuffer(this.staticSolidTiles,0,this.solidTiles,0,this.words*4);
   this.bodies?.(encoder,this.solidTiles);this.bodyBits=!!this.bodies;
  }
  const tiles=this.ownership.layout.tiles.length,x=this.ownership.dispatchX,t=this.ownership.layout.lattice.dimensions.map(n=>n/4+1);
  for(const [label,entries] of [["classify",["classify","classifyCoarse"]],["prefix",["prefix0","prefix1","prefix2",...Array.from({length:CUBE_LEVELS},(_,j)=>`cube${j+1}`)]],["decide",this.solid||this.bodies?["decide","solidActive","solidPromote"]:["decide"]]] as const){
   const pass=encoder.beginComputePass({label:`Uniform dynamic ownership census ${label}`});
   pass.setBindGroup(0,this.ownership.bindGroup);pass.setBindGroup(1,this.group);
   for(const entry of entries){
    const lines=entry.startsWith("prefix")?[0,1,2].filter(k=>k!==Number(entry.at(-1))).reduce((n,k)=>n*t[k]!,1):tiles;
    const l=this.ownership.layout,groups=entry==="classify"?l.fineTiles.length:entry==="classifyCoarse"?Math.ceil(l.coarseTiles.length/64):entry.startsWith("cube")?Math.ceil(tiles/64):Math.ceil(lines/64);
    if(!groups)continue;
    pass.setPipeline(this.pipelines.get(entry)!);pass.dispatchWorkgroups(Math.min(groups,x),Math.ceil(groups/x));
   }
   pass.end();
  }
  encoder.copyBufferToBuffer(this.work,0,this.readback,0,(HEADER+this.words)*4);
  this.encoded=true;
 }
 /** Map the census encoded by the last submitted encode(). */
 async read():Promise<UniformMixedDynamicCensus>{
  if(!this.encoded)throw new Error("Dynamic ownership census was not encoded");
  this.encoded=false;
  await this.readback.mapAsync(GPUMapMode.READ);
  const words=new Uint32Array(this.readback.getMappedRange()).slice();this.readback.unmap();
  const f=new Float32Array(words.buffer);
  const tiles=this.ownership.layout.tiles.length,fine=new Uint8Array(tiles);
  for(let t=0;t<tiles;t++)fine[t]=(words[HEADER+(t>>5)]!>>>(t&31))&1;
  return {fine,interfaceTiles:words[0]!,requiredTiles:words[12]!,unresolvedCoarse:words[1]!,unresolvedTiles:Array.from(words.subarray(13,13+Math.min(3,words[1]!))),boundaryTiles:words[16]!,fineTiles:words[2]!,refined:words[4]!,coarsened:words[5]!,solidTiles:words[19]!,
  coarsePartialVolume:words[6]!,coarsePhiCrossing:words[7]!,coarseDryLiquidPhi:words[3]!,
  interiorDeficit:[f[8]!,f[9]!],airVolume:[f[10]!,f[11]!]};
 }
 destroy():void{this.solidTiles.destroy();this.staticSolidTiles.destroy();this.work.destroy();this.bounds.destroy();this.readback.destroy();this.params.destroy();}
}
