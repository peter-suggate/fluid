import type {UniformMixedOwnership} from "./uniform-mixed-ownership";
import type {UniformMixedBandBits} from "./uniform-mixed-layout-builder";
import {UNIFORM_MIXED_JOBS,uniformMixedCertifiedEntriesWGSL,uniformMixedTopologyWGSL} from "./uniform-mixed-topology.wgsl";
import {uniformMixedVertexSamplingSource} from "./uniform-mixed-vertex-sampling.wgsl";

/** Largest tile radius the census gathers velocity bounds over, and the
 * largest reach/hysteresis. A tile whose RK2 midpoints may leave that radius
 * is kept fine rather than trusting a truncated bound. */
export const UNIFORM_MIXED_DYNAMIC_DISTANCE_CAP=16;
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
/** Velocity-bound boxes: min/max over radius 0 (the tile) up to 16 tiles.
 * Small radii are dense: a radius rounded up widens every box the census
 * tests. Level 0 lives in the atomic census; the boxes come from a cube
 * table in a plain buffer (atomic loads and stores do not coalesce). */
const BOUND_RADII=[0,1,2,3,4,5,6,8,10,12,16] as const;
/** Per radius r: the cube side 2^j with 2^j <= 2r+1 < 2^(j+1); eight such
 * cubes cover the box exactly. */
const BOUND_CUBES=BOUND_RADII.map(r=>Math.floor(Math.log2(2*r+1)));
const CUBE_LEVELS=BOUND_CUBES[BOUND_CUBES.length-1]!;
/** Cube starts run from -CUBE_PAD (the largest radius) per axis. */
const CUBE_PAD=BOUND_RADII[BOUND_RADII.length-1]!;

export interface UniformMixedDynamicPolicy {
 /** The step the next frame will take; with the tile speed it bounds travel. */
 dt:number;
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

/** Lagged diagnostics readback: a copy lands in a free ring slot and maps
 * after submit; while every slot is still mapping, a frame skips its copy.
 * Nothing on the advance path waits for it. */
export class UniformMixedReadbackRing {
 private readonly all:readonly GPUBuffer[];
 private readonly free:GPUBuffer[];
 constructor(device:GPUDevice,label:string,bytes:number,slots=3){
  this.all=Array.from({length:slots},(_,i)=>device.createBuffer({label:`${label} ${i}`,size:bytes,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ}));
  this.free=[...this.all];
 }
 get allocatedBytes():number{return this.all.reduce((n,b)=>n+b.size,0);}
 /** copy fills the slot; the returned read maps it once the encoder is submitted. */
 encode(encoder:GPUCommandEncoder,copy:(buffer:GPUBuffer)=>void):(()=>Promise<Uint32Array>)|undefined{
  const buffer=this.free.pop();if(!buffer)return undefined;
  copy(buffer);
  return async()=>{
   try{await buffer.mapAsync(GPUMapMode.READ);const words=new Uint32Array(buffer.getMappedRange()).slice();buffer.unmap();return words;}
   finally{this.free.push(buffer);}
  };
 }
 destroy():void{for(const b of this.all)b.destroy();}
}

/** One census's counters (lagged diagnostics). */
export interface UniformMixedDynamicCensus {
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
 /** Band tiles only for solid promotion. */
 readonly solidTiles:number;
}

/** State-driven ownership census (docs/plans/uniform-dynamic-coarsening.md).
 * Classifies each tile against the live ownership after a completed frame and
 * predicts the next frame's surface tiles. The next frame advects phi and V
 * semi-Lagrangian (RK2 from each destination point), so tile t can hold the
 * surface only if the departure box of its points holds a current interface
 * tile. The box comes from signed velocity bounds gathered over the tiles
 * its midpoints can reach. The next frame extends once, on the adopted
 * layout, so no extended field exists here: a tile bounds the extension's
 * values instead, the physical faces of the liquid owners within the sweep
 * reach and the far states of the tiles around it (the frame's tail builds
 * the nearest-source hierarchy alone for this). A 3D prefix sum answers the
 * box query. Still
 * liquid keeps exactly its interface tiles fine; moving liquid extends the
 * set upstream by its local travel only. The layout builder reads the band
 * bits on the GPU; only counters read back (lagged diagnostics). */
export class UniformMixedDynamicClassifier {
 readonly allocatedBytes:number;
 private readonly work:GPUBuffer;
 /** Cube table levels 1..CUBE_LEVELS, component-major over padded starts. */
 private readonly bounds:GPUBuffer;
 private readonly readbacks:UniformMixedReadbackRing;
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
 /** The band decided by the last encode, for the GPU layout builder. */
 get bandBits():UniformMixedBandBits{return {buffer:this.work,wordOffset:HEADER};}
 constructor(private readonly device:GPUDevice,readonly ownership:UniformMixedOwnership,volume:GPUTexture,phi:GPUTexture,velocity:GPUTexture,far:GPUTexture){
  const tiles=ownership.capacity.tileCount;
  this.words=Math.ceil(tiles/32);
  const t=ownership.capacity.lattice.dimensions.map(n=>n/4);
  const workBytes=(HEADER+this.words+6*tiles+(t[0]!+1)*(t[1]!+1)*(t[2]!+1)+2*tiles+2*this.words+6*tiles)*4;
  this.work=device.createBuffer({label:"Uniform dynamic ownership census",size:workBytes,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
  this.readbacks=new UniformMixedReadbackRing(device,"Uniform dynamic ownership census counters",HEADER*4);
  this.params=device.createBuffer({label:"Uniform dynamic ownership policy",size:48,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
  this.solidTiles=device.createBuffer({label:"Uniform dynamic ownership solid tiles",size:this.words*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
  this.staticSolidTiles=device.createBuffer({label:"Uniform dynamic ownership static solid tiles",size:this.words*4,usage:GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
  const boundsBytes=6*CUBE_LEVELS*t.reduce((n,k)=>n*(k+CUBE_PAD),1)*4;
  this.bounds=device.createBuffer({label:"Uniform dynamic ownership bound cubes",size:boundsBytes,usage:GPUBufferUsage.STORAGE});
  this.allocatedBytes=workBytes+boundsBytes+this.readbacks.allocatedBytes+48+this.words*8;
  this.resources=device.createBindGroupLayout({entries:[
   ...[0,1,2,7].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float" as const,viewDimension:"3d" as const}})),
   {binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},
   {binding:4,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},
   {binding:5,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},
   {binding:6,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},
  ]});
  this.group=device.createBindGroup({layout:this.resources,entries:[
   {binding:0,resource:volume.createView()},{binding:1,resource:phi.createView()},
   {binding:2,resource:velocity.createView()},{binding:3,resource:{buffer:this.work}},{binding:4,resource:{buffer:this.params}},
   {binding:5,resource:{buffer:this.solidTiles}},{binding:6,resource:{buffer:this.bounds}},{binding:7,resource:far.createView()},
  ]});
 }
 async initialize():Promise<void>{
  const h=this.ownership.capacity.lattice.cellSize_m,cap=UNIFORM_MIXED_DYNAMIC_DISTANCE_CAP;
  const module=this.device.createShaderModule({label:"Uniform dynamic ownership census",code:uniformMixedCertifiedEntriesWGSL(uniformMixedTopologyWGSL(this.ownership.layout,0)+/* wgsl */`
@group(1) @binding(0) var volume:texture_3d<f32>;
@group(1) @binding(1) var phi:texture_3d<f32>;
@group(1) @binding(2) var velocity:texture_3d<f32>;
@group(1) @binding(3) var<storage,read_write> census:array<atomic<u32>>;
struct DynamicPolicy {step:vec4f,reach:vec4u,surface:vec4f}
@group(1) @binding(4) var<uniform> policy:DynamicPolicy;
@group(1) @binding(5) var<storage,read> solidTiles:array<u32>;
@group(1) @binding(6) var<storage,read_write> bounds:array<u32>;
// The frame's extension hierarchy: per tile, the nearest-source state its
// far faces blend (xyz per component, w the valid-component mask).
@group(1) @binding(7) var far:texture_3d<f32>;
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
const RADII=array<i32,${BOUND_RADII.length}>(${BOUND_RADII.join(",")});
const CUBES=array<u32,${BOUND_CUBES.length}>(${BOUND_CUBES.join(",")});
// Cube table: level j >= 1 holds, per start s (from -CUBE_PAD per axis), the
// keys of the cube [s, s+2^j)³ clipped to the domain; identity if empty.
const CUBE_PAD:i32=${CUBE_PAD};
const PT=UM_T+vec3u(u32(CUBE_PAD));const CUBE:u32=PT.x*PT.y*PT.z;
fn cubeIndex(j:u32,k:u32,s:vec3i)->u32{let q=vec3u(s+vec3i(CUBE_PAD));return ((j-1u)*6u+k)*CUBE+q.x+PT.x*(q.y+PT.y*q.z);}
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
// Per tile: keys of its liquid owners' physical face velocities, the minima
// inverted so the cleared census is the identity.
fn sourceIndex(t:u32,k:u32)->u32{return activeIndex(WORDS)+6u*t+k;}
fn orderKey(x:f32)->u32{let b=bitcast<u32>(x);return select(b|0x80000000u,~b,(b&0x80000000u)!=0u);}
fn orderValue(k:u32)->f32{return bitcast<f32>(select(~k,k&0x7fffffffu,(k&0x80000000u)!=0u));}
// One tile's classification: ordered velocity keys, nibble distances from
// each face to the nearest crossing owner (gap) and surface owner (reach),
// flags (bit 0 surface owner, bit 1 phi sign change) and 4h surface error.
struct TileClass{low:array<u32,3>,high:array<u32,3>,gap:array<u32,6>,reach:array<u32,6>,flags:u32,error:u32}
fn umEmptyClass()->TileClass{return TileClass(array<u32,3>(0xffffffffu,0xffffffffu,0xffffffffu),array<u32,3>(0u,0u,0u),array<u32,6>(15u,15u,15u,15u,15u,15u),array<u32,6>(15u,15u,15u,15u,15u,15u),0u,0u);}
var<workgroup> mixedTile:atomic<u32>;
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
// Extension sources: the physical faces of every liquid (not air) owner.
// Every extended face the next frame traces blends the sources within its
// reach or takes its nearest source's value.
fn umOwnerSources(tile:u32,width:u32,lane:u32){
 let side=4u/width;
 let origin=umTileCoord(tile)*4u+umCorner(lane,side)*width;
 let owner=umOwnerAt(vec3i(origin));
 let v=textureLoad(volume,vec3i(origin),0).x;
 var inside=0u;for(var k=0u;k<8u;k++){if(umVertexValue(origin+umCorner(k,2u)*width)<0.0){inside++;}}
 if(inside==0u&&v<=policy.step.w){return;}
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
  atomicMax(&census[sourceIndex(tile,axis)],~orderKey(low));atomicMax(&census[sourceIndex(tile,3u+axis)],orderKey(high));
 }
}
@compute @workgroup_size(64) fn sources(@builtin(workgroup_id) gid:vec3u,@builtin(local_invocation_index) lane:u32){
 let job=gid.x+umDispatchX*gid.y;if(job>=umCounts.x){return;}
 let tile=umTopology[UM_TILES+job];let width=umTileWidth(tile);
 if(lane<(64u/(width*width*width))){umOwnerSources(tile,width,lane);}
}
@compute @workgroup_size(64) fn sourcesCoarse(@builtin(global_invocation_id) gid:vec3u){
 let index=gid.x+umDispatchX*64u*gid.y;if(index>=umCounts.y){return;}
 umOwnerSources(umTopology[UM_TILES+umCounts.x+index],4u,0u);
}
// A tile's velocity bounds, the values the extension can give its faces:
// the sources within the sweep reach (two face widths: one tile at h, two
// at 4h) and the far states beyond it.
fn umReachBounds(tile:u32,width:u32,c:ptr<function,TileClass>){
 let p=vec3i(umTileCoord(tile));let r=select(2,1,width==1u);
 let a=max(p-vec3i(r),vec3i(0));let b=min(p+vec3i(r),vec3i(UM_T)-vec3i(1));
 for(var z=a.z;z<=b.z;z++){for(var y=a.y;y<=b.y;y++){for(var x=a.x;x<=b.x;x++){
  let q=umTileAt(vec3u(vec3i(x,y,z)));
  for(var k=0u;k<3u;k++){(*c).low[k]=min((*c).low[k],~atomicLoad(&census[sourceIndex(q,k)]));(*c).high[k]=max((*c).high[k],atomicLoad(&census[sourceIndex(q,3u+k)]));}
 }}}
 // A face beyond the sweep reach takes the far state of one of the tiles
 // around it (the trilinear taps of its point: this tile and its 26).
 let fa=max(p-vec3i(1),vec3i(0));let fb=min(p+vec3i(1),vec3i(UM_T)-vec3i(1));
 for(var z=fa.z;z<=fb.z;z++){for(var y=fa.y;y<=fb.y;y++){for(var x=fa.x;x<=fb.x;x++){
  let state=textureLoad(far,vec3i(x,y,z),0);let mask=u32(round(state.w));
  for(var k=0u;k<3u;k++){if((mask&(1u<<k))!=0u){let key=orderKey(state[k]);(*c).low[k]=min((*c).low[k],key);(*c).high[k]=max((*c).high[k],key);}}
 }}}
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
 let interior=inside==8u&&v>=1.0-policy.step.z;
 let air=inside==0u&&v<=policy.step.w;
 // Largest interior deficit and largest air volume, per fine/coarse width.
 let coarse=select(0u,1u,width!=1u);
 if(deep==8u){atomicMax(&census[8u+coarse],bitcast<u32>(max(1.0-v,0.0)));}
 if(inside==0u){atomicMax(&census[10u+coarse],bitcast<u32>(max(v,0.0)));}
 let local=umCorner(lane,side)*width;
 if(inside!=0u&&inside!=8u){
  for(var a=0u;a<3u;a++){(*c).gap[a]=min((*c).gap[a],local[a]);(*c).gap[3u+a]=min((*c).gap[3u+a],4u-width-local[a]);}
 }
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
fn umFinishTile(tile:u32,width:u32,classified:TileClass){
 var c=classified;umReachBounds(tile,width,&c);
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
 if(lane==0u){atomicStore(&mixedTile,0u);for(var a=0u;a<3u;a++){atomicStore(&tileGap[a],15u);atomicStore(&tileGap[3u+a],15u);atomicStore(&tileReach[a],15u);atomicStore(&tileReach[3u+a],15u);}atomicStore(&tileError,0u);}workgroupBarrier();
 let width=umTileWidth(tile);let side=4u/width;
 if(lane<side*side*side){
  var c=umEmptyClass();umClassifyOwner(tile,width,lane,&c);
  for(var k=0u;k<6u;k++){atomicMin(&tileGap[k],c.gap[k]);atomicMin(&tileReach[k],c.reach[k]);}
  atomicOr(&mixedTile,c.flags);
 }
 workgroupBarrier();
 if((atomicLoad(&mixedTile)&2u)!=0u&&policy.step.y>0.0){atomicMax(&tileError,bitcast<u32>(umResolutionError(tile,lane,width)));}
 workgroupBarrier();
 if(lane==0u){
  var c=umEmptyClass();
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
// Box min (low keys) and max (high keys) of radius r_k, clamped to the
// domain. Dry tiles hold the identity keys and drop out.
override cubeLevel:u32=1u;
const LEVELS:u32=${BOUND_RADII.length}u;
// decide climbs from level 1 until the radius covers its tile's travel; no
// box exceeds the largest tile travel (census[18]), so no tile climbs past
// the level that covers it.
fn topLevel()->u32{
 let need=1+i32(ceil(0.5*bitcast<f32>(atomicLoad(&census[18]))/4.0));
 var level=1u;loop{if(need<=RADII[level]||level+1u>=LEVELS){break;}level++;}
 return level;
}
fn identityKeys()->array<u32,6>{return array<u32,6>(0xffffffffu,0xffffffffu,0xffffffffu,0u,0u,0u);}
// Cube level j from eight cubes of level j-1 (level 0: the tiles), for the
// levels decide can reach. One lane per padded start.
@compute @workgroup_size(64) fn boundCube(@builtin(global_invocation_id) gid:vec3u){
 let index=gid.x+umDispatchX*64u*gid.y;if(index>=CUBE){return;}
 if(cubeLevel>CUBES[topLevel()]){return;}
 let s=vec3i(vec3u(index%PT.x,(index/PT.x)%PT.y,index/(PT.x*PT.y)))-vec3i(CUBE_PAD);
 let half=i32(1u<<(cubeLevel-1u));var keys=identityKeys();
 for(var o=0u;o<8u;o++){
  let q=s+half*vec3i(vec3u(o&1u,(o>>1u)&1u,o>>2u));
  if(any(q>=vec3i(UM_T))){continue;}
  if(cubeLevel==1u){
   if(any(q<vec3i(0))){continue;}
   let t=umTileAt(vec3u(q));
   for(var k=0u;k<3u;k++){keys[k]=min(keys[k],atomicLoad(&census[boundIndex(t,k)]));keys[3u+k]=max(keys[3u+k],atomicLoad(&census[boundIndex(t,3u+k)]));}
  }else{
   for(var k=0u;k<3u;k++){keys[k]=min(keys[k],bounds[cubeIndex(cubeLevel-1u,k,q)]);keys[3u+k]=max(keys[3u+k],bounds[cubeIndex(cubeLevel-1u,3u+k,q)]);}
  }
 }
 for(var k=0u;k<6u;k++){bounds[cubeIndex(cubeLevel,k,s)]=keys[k];}
}
// Radius-r box of tile p: the eight cubes of side 2^j starting at p-r or at
// p+r-2^j+1 per axis cover [p-r, p+r]³ exactly (2^(j+1) > 2r+1).
fn boxKeys(p:vec3i,level:u32)->array<u32,6>{
 let r=RADII[level];let j=CUBES[level];
 let a=p-vec3i(r);let b=p+vec3i(r+1-i32(1u<<j));var keys=identityKeys();
 for(var o=0u;o<8u;o++){
  let q=select(a,b,vec3<bool>((o&1u)!=0u,(o&2u)!=0u,(o&4u)!=0u));
  if(any(q>=vec3i(UM_T))){continue;}
  for(var k=0u;k<3u;k++){keys[k]=min(keys[k],bounds[cubeIndex(j,k,q)]);keys[3u+k]=max(keys[3u+k],bounds[cubeIndex(j,3u+k,q)]);}
 }
 return keys;
}
fn interfaceTilesIn(low:vec3i,high:vec3i)->u32 {
 let a=vec3u(clamp(low,vec3i(0),vec3i(UM_T)));let b=vec3u(clamp(high+vec3i(1),vec3i(0),vec3i(UM_T)));
 if(any(b<=a)){return 0u;}
 let s=atomicLoad(&census[prefixIndex(b)])-atomicLoad(&census[prefixIndex(vec3u(a.x,b.y,b.z))])-atomicLoad(&census[prefixIndex(vec3u(b.x,a.y,b.z))])-atomicLoad(&census[prefixIndex(vec3u(b.x,b.y,a.z))]);
 return s+atomicLoad(&census[prefixIndex(vec3u(a.x,a.y,b.z))])+atomicLoad(&census[prefixIndex(vec3u(a.x,b.y,a.z))])+atomicLoad(&census[prefixIndex(vec3u(b.x,a.y,a.z))])-atomicLoad(&census[prefixIndex(a)]);
}
// Predicted surface tile: the RK2 departure box of t's points (x - dt·u with
// u sampled at x and at the midpoint) holds an interface tile. The velocity
// bounds come from the smallest box level that covers the midpoints.
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
 // The smallest box level whose bounds cover its own midpoints.
 var level=1u;var low=vec3f(0);var high=vec3f(0);var unbounded=false;
 loop{
  // Zero joins the bounds: sampling near a wall or solid blends in its
  // zero face, and the trace stops short at a solid or the domain clamp.
  let keys=boxKeys(p,level);
  for(var a=0u;a<3u;a++){low[a]=min(0.0,orderValue(keys[a]));high[a]=max(0.0,orderValue(keys[3u+a]));}
  let travel=max(abs(low*scale),abs(high*scale));
  let need=1+i32(ceil(0.5*max(travel.x,max(travel.y,travel.z))/4.0));
  if(need<=RADII[level]){break;}
  if(level+1u>=${BOUND_RADII.length}u){unbounded=true;break;}
  level++;
 }
 let width=umTileWidth(tile);
 let margin=i32(policy.reach.x)+select(0,i32(policy.reach.y),width==1u);
 // Departure points span [4p - hi·s, 4p + 4 - lo·s] in cells (closed: a
 // boundary vertex belongs to both tiles it separates). A neighbouring
 // crossing tile q counts only if its nearest crossing owner lies within the
 // depth the span reaches into q on every axis; a still surface keeps just
 // the tiles whose vertices it touches, not a 26-tile shell.
 // Two h margins beyond the trace that are not phi reach (without either,
 // no crossing lands in a 4h owner: 0 unresolved at 128³ and the long dam)
 // but that the long-dam toe needs: one cell everywhere (without it a
 // 3-cell sheet runs ahead of the fine front by frame 10, toe window 430 vs
 // fine 1.7; not the cubic taps, same with cubic off, nor the margin on wet
 // or dry tiles alone), and a closed-wall tile one tile inward (without it
 // the frame-20 tip runs to 187 vs fine 169). Mechanism open.
 var lo=4.0*vec3f(p)-high*scale-4.0*f32(margin)-1.0;
 var hi=4.0*vec3f(p)+4.0-low*scale+4.0*f32(margin)+1.0;
 // Beyond the trace, phi moves by more than a cell only where
 // umReleasedWalls lifts it: a vertex within dt·away of a released plane
 // (away: the velocity leaving it, bounded by the box) rises to air if its
 // advected value lies within half its authority width of the surface (2
 // cells at 4h) at a closed wall, at any depth under the ambient top. A tile
 // within that reach of a closed wall counts crossing owners 2 cells beyond
 // its span; of the ambient top, it is fine wherever its span holds liquid.
 var walls=false;var ambient=false;
 for(var a=0u;a<3u;a++){for(var side=0u;side<2u;side++){
  let distance=select(4.0*f32(p[a]),f32(UM_D[a])-4.0*f32(p[a]+1),side==1u);
  let away=select(high[a],-low[a],side==1u)*scale[a];
  if(!(away>0.0)||distance>=away){continue;}
  if(((policy.reach.z>>(a+3u*side))&1u)!=0u){walls=true;}else if(a==1u&&side==1u){ambient=true;}
 }}
 if(walls){lo-=vec3f(2.0);hi+=vec3f(2.0);}
 for(var a=0u;a<3u;a++){
  if(p[a]==0&&((policy.reach.z>>a)&1u)!=0u){hi[a]+=4.0;}
  if(p[a]==i32(UM_T[a])-1&&((policy.reach.z>>(3u+a))&1u)!=0u){lo[a]-=4.0;}
 }
 let first=vec3i(floor((lo-1e-3)/4.0));
 let last=vec3i(floor((hi+1e-3)/4.0));
 // A current interface tile stays fine: phi does not only move with the
 // flow (residual sheets left behind a falling surface persist in place).
 var fine=unbounded||interfaceTilesIn(p,p)!=0u;
 if(!fine&&ambient){
  let a=max(first,vec3i(0));let b=min(last,vec3i(UM_T)-vec3i(1));
  for(var z=a.z;z<=b.z&&!fine;z++){for(var y=a.y;y<=b.y&&!fine;y++){for(var x=a.x;x<=b.x&&!fine;x++){fine=umBit(wetIndex(0u),umTileAt(vec3u(vec3i(x,y,z))));}}}
 }
 if(!fine&&interfaceTilesIn(first,last)!=0u){
  if(any(last-first>vec3i(4))){fine=true;}
  else{
   let a=max(first,vec3i(0));let b=min(last,vec3i(UM_T)-vec3i(1));
   for(var z=a.z;z<=b.z&&!fine;z++){for(var y=a.y;y<=b.y&&!fine;y++){for(var x=a.x;x<=b.x&&!fine;x++){
    let q=vec3i(x,y,z);let gap=atomicLoad(&census[gapIndex(umTileAt(vec3u(q)))]);
    if(gap==0xffffffffu){continue;}
    var reaches=true;
    for(var axis=0u;axis<3u;axis++){
     if(q[axis]<p[axis]){reaches=reaches&&f32((gap>>(4u*(3u+axis)))&15u)<=4.0*f32(q[axis])+4.0-lo[axis]+1e-3;}
     if(q[axis]>p[axis]){reaches=reaches&&f32((gap>>(4u*axis))&15u)<=hi[axis]-4.0*f32(q[axis])+1e-3;}
    }
    fine=reaches;
   }}}
  }
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
`,["sources","sourcesCoarse","classify","classifyCoarse"])});
  const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");
  if(errors.length)throw new Error(errors.map(m=>`${m.lineNum}: ${m.message}`).join("\n"));
  const layout=this.device.createPipelineLayout({bindGroupLayouts:[this.ownership.bindLayout,this.resources]});
  for(const [entryPoint,umCellWidth] of [["sources",1],["sourcesCoarse",4],["classify",1],["classifyCoarse",4]] as const)this.pipelines.set(entryPoint,await this.ownership.pipeline(layout,module,entryPoint,UNIFORM_MIXED_JOBS.tier,{umCellWidth}));
  for(const entryPoint of ["prefix0","prefix1","prefix2","decide","solidActive","solidPromote"])
   this.pipelines.set(entryPoint,await this.device.createComputePipelineAsync({layout,compute:{module,entryPoint,constants:{umDispatchX:this.ownership.dispatchX}}}));
  for(let level=1;level<=CUBE_LEVELS;level++)
   this.pipelines.set(`cube${level}`,await this.device.createComputePipelineAsync({layout,compute:{module,entryPoint:"boundCube",constants:{umDispatchX:this.ownership.dispatchX,cubeLevel:level}}}));
 }
 /** Solid-coupled tiles (uniformMixedSolidTiles().coupled), for the impact rule. */
 setSolid(coupled:Uint8Array):void{
  const tiles=this.ownership.capacity.tileCount;
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
  if(this.pipelines.size!==10+CUBE_LEVELS)throw new Error("Dynamic ownership census is not initialized");
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
  this.device.queue.writeBuffer(this.params,32,new Float32Array([policy.fastTravel,policy.boundaryTravel,0,0]));
  // Cube levels up to the one decide can reach are written in full before
  // decide reads them; the census is cleared whole.
  encoder.clearBuffer(this.work);
  if(this.bodies||this.bodyBits){
   encoder.copyBufferToBuffer(this.staticSolidTiles,0,this.solidTiles,0,this.words*4);
   this.bodies?.(encoder,this.solidTiles);this.bodyBits=!!this.bodies;
  }
  const tiles=this.ownership.capacity.tileCount,x=this.ownership.dispatchX,t=this.ownership.capacity.lattice.dimensions.map(n=>n/4+1);
  const cubeStarts=t.reduce((n,k)=>n*(k-1+CUBE_PAD),1);
  for(const [label,entries] of [["sources",["sources","sourcesCoarse"]],["classify",["classify","classifyCoarse"]],["prefix",["prefix0","prefix1","prefix2",...Array.from({length:CUBE_LEVELS},(_,j)=>`cube${j+1}`)]],["decide",this.solid||this.bodies?["decide","solidActive","solidPromote"]:["decide"]]] as const){
   const pass=encoder.beginComputePass({label:`Uniform dynamic ownership census ${label}`});
   pass.setBindGroup(0,this.ownership.bindGroup);pass.setBindGroup(1,this.group);
   for(const entry of entries){
    const lines=entry.startsWith("prefix")?[0,1,2].filter(k=>k!==Number(entry.at(-1))).reduce((n,k)=>n*t[k]!,1):tiles;
    if(entry.startsWith("classify")||entry.startsWith("sources")){this.ownership.dispatchTier(pass,this.pipelines.get(entry)!,entry.endsWith("Coarse")?1:0);continue;}
    const groups=entry.startsWith("cube")?Math.ceil(cubeStarts/64):Math.ceil(lines/64);
    pass.setPipeline(this.pipelines.get(entry)!);pass.dispatchWorkgroups(Math.min(groups,x),Math.ceil(groups/x));
   }
   pass.end();
  }
 }
 /** Copy the last encode's counters into a free ring slot; the returned
  * start maps them after submit. undefined while every slot is mapping. */
 encodeReadback(encoder:GPUCommandEncoder):(()=>Promise<UniformMixedDynamicCensus>)|undefined{
  const read=this.readbacks.encode(encoder,buffer=>encoder.copyBufferToBuffer(this.work,0,buffer,0,HEADER*4));
  if(!read)return undefined;
  return async()=>{
   const words=await read();const f=new Float32Array(words.buffer);
   return {interfaceTiles:words[0]!,requiredTiles:words[12]!,unresolvedCoarse:words[1]!,unresolvedTiles:Array.from(words.subarray(13,13+Math.min(3,words[1]!))),boundaryTiles:words[16]!,fineTiles:words[2]!,refined:words[4]!,coarsened:words[5]!,solidTiles:words[19]!,
    coarsePartialVolume:words[6]!,coarsePhiCrossing:words[7]!,coarseDryLiquidPhi:words[3]!,
    interiorDeficit:[f[8]!,f[9]!],airVolume:[f[10]!,f[11]!]};
  };
 }
 destroy():void{this.solidTiles.destroy();this.staticSolidTiles.destroy();this.work.destroy();this.bounds.destroy();this.readbacks.destroy();this.params.destroy();}
}
