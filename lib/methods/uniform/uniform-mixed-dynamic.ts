import { uniformDetailBindLayout, uniformDetailModule, uniformDetailPipeline, uniformDetailPick, uniformDetailGroup, type UniformDetailGroup } from "./uniform-detail-fields";
import type {UniformMixedOwnership} from "./uniform-mixed-ownership";
import type {UniformMixedBandBits} from "./uniform-mixed-layout-builder";
import {uniformMixedPageCount,uniformMixedTopologyWGSL} from "./uniform-mixed-topology.wgsl";
import {uniformMixedSourceWGSL} from "./uniform-mixed-source.wgsl";
import {UNIFORM_DETAIL_4H_LOAD} from "../../core/uniform-detail-abi";
import {UNIFORM_STAGE_REASON as REASON,UNIFORM_STAGE_IMPORTANCE as IMPORTANCE,UNIFORM_DETAIL_CRITERIA,type UniformDetailCriterion} from "./uniform-stage-grids";

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
/** Workgroups of the h-tile classify launch at most: a fixed grid that
 * strides over the live h list (umCounts.x), so no host tile count sizes it.
 * The ownership's counted-launch grid (COUNTED_GRID): enough to fill the
 * GPU, and an h tile's classify costs about the same as any other's. */
const CENSUS_TILE_GRID=4096;
/** Bins of the budget's histogram of triggering tiles, by top score byte. */
const IMPORTANCE_BINS=64;
/** Persistent words after the importance words: a hold per tile, the
 * budget's score cutoff, its histogram. */
const importanceStateWords=(tiles:number)=>tiles+1+IMPORTANCE_BINS;
/** Bytes of the scene uniform prefix the source mark reads (UMSourceParams). */
const SOURCE_PARAMS_BYTES=176;
/** Residency closure radius beyond the travel, in tiles
 * (uniformMixedResidencyWord). The census certifies the state a frame
 * starts from; the frame's readers of the certificate run on that state
 * advanced by at most departureTravel() cells (the ceil of it over 4, plus
 * one tile for the sub-tile start), and read around the tiles the census
 * marks near (an owner with V != 0 or a corner phi under 4h times its
 * width: 16h on a 4h owner). The widest reader is the surface volume:
 *  - its band seeds are owners whose corners straddle phi = 0, so they lie
 *    in near tiles;
 *  - the band grows four owners past them. An owner k face steps from a
 *    seed has a corner within 4k cells of the seed's non-positive corner, so
 *    with phi a distance there (the redistanced band) owners up to three
 *    steps out hold a corner under 12h < 16h and are near; only the fourth
 *    step leaves the near set, by one tile;
 *  - visited is the band plus one tile, measured the visited set plus one
 *    tile, and the measured tiles read one tile further.
 * 1 (sub-tile) + 1 (band past near) + 1 + 1 + 1 = 5. The frame plan's
 * certificate spread (ceil(travel/4)+2) is inside it. Page rounding adds
 * up to three tiles of slack that nothing relies on. A reader that would
 * still leave the closure (phi not a distance on the band's fourth step,
 * say) sets the sticky support bit the next census turns into a builder
 * fatal: never a silent read of an absent page. */
const RESIDENCY_MARGIN_TILES=5;
/** Page reach the mark pass scans (pages): a larger radius makes every page
 * resident (always correct, no work saved). */
const RESIDENCY_PAGE_REACH=4;
/** Audit bits: a tile of an absent page held liquid or near-surface phi
 * (or was h); an in-frame reader left the closure (support violation bits). */
export const UNIFORM_MIXED_RESIDENCY_AUDIT={absentNear:1,closure:2} as const;

/** Detail importance (UNIFORM_DETAIL_CRITERIA): what requires a tile at h.
 * Every criterion scores a tile as its measure over its threshold; a
 * criterion that is on requires the tile at a score of 1. Shape's tolerance
 * is the policy's surfaceTolerance and impact's travel its boundaryTravel.
 * The required tiles seed the band: decide adds their closure and travel. */
export interface UniformMixedDynamicImportance {
 /** Filter automatic band admission by distance to a phi-crossing tile. Does
  * not seed a tile, change the scores, or shrink storage support closure. */
 surfaceOnly?:boolean;
 /** Allowed neighbouring tile layers, including diagonals (0..3). */
 surfaceDistance?:number;
 /** NB has surface-derived occupancy; include zero boundaries on both sides. */
 surfaceAuthority?:boolean;
 criteria:Readonly<Record<UniformDetailCriterion,boolean>>;
 /** Liquid or air thinner than this across a surface tile, in h. Twice the
  * deepest phi on either side, over the 4-aligned vertices one tile around. */
 thinThickness:number;
 /** dt·‖sym ∇u‖ and dt·‖curl u‖, from the tile velocity bounds' midpoints
  * across the wet face neighbours. */
 strainThreshold:number;
 rotationThreshold:number;
 /** Steps within which a surface tile's travel closes the air ahead of it:
  * to a solid tile, a closed wall or liquid beyond air (closing speed). */
 approachSteps:number;
 /** Strain and rotation also judge liquid tiles without a surface. */
 bulk:boolean;
 /** Censuses a tile stays required after its last trigger; a score at or
  * above retireRatio keeps the count from running down. 0: no hold. */
 holdSteps:number;
 retireRatio:number;
 /** Band tiles the criteria may hold, or undefined for no limit. The seeds
  * are cut, lowest top score first, to the share of the last band they were
  * (closure and travel ride on them): a one-census lag. A saturated score
  * (shape at tolerance 0) is never cut. */
 budgetTiles?:number;
 /** A tile this frame's drop or inflow plug can fill is band (absent: it
  * is). Off under a census of requests only: a source is not a request. */
 sources?:boolean;
 /** What shape measures on an h tile (UniformDetailShapeMetric; absent:
  * "value"). "displacement": umSurfaceDisplacement. */
 shapeMetric?:"value"|"displacement";
}
/** No importance policy: shape and impact as the policy's tolerances say. */
const LEGACY_IMPORTANCE:UniformMixedDynamicImportance={criteria:{shape:true,thin:false,strain:false,rotation:false,impact:true,approach:false},
 thinThickness:0,strainThreshold:0,rotationThreshold:0,approachSteps:0,bulk:false,holdSteps:0,retireRatio:1};

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
 /** Record why each tile is band (UNIFORM_STAGE_REASON, the `reasons`
  * words) for the tiles layer, and every criterion's score (on or off) in
  * the importance words. Off records nothing and adds no work. */
 reasons?:boolean;
 /** Absent: shape and impact only (surfaceTolerance, boundaryTravel). */
 importance?:UniformMixedDynamicImportance;
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
 /** Cube table levels 1..CUBE_LEVELS, component-major over start tiles,
  * then the (T+1)³ interface prefix table. */
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
 private readonly group:UniformDetailGroup;
 private readonly pipelines=new Map<string,GPUComputePipeline>();
 private readonly words:number;
 /** Residency pages (uniformMixedPageCount). */
 private readonly pages:number;
 /** First wet word (wetIndex(0)): wet then active bits, cleared per census. */
 private readonly wetOffset:number;
 /** First near word (nearIndex(0)), after the band reasons: near bits, the audit word, page boxes. */
 private readonly nearOffset:number;
 /** One word per tile after the active bits: the band reason
  * (UNIFORM_STAGE_REASON), written by every decide of a census encoded with
  * policy.reasons, and by solid promotion. Undefined content otherwise. */
 get reasons():{readonly buffer:GPUBuffer;readonly offset:number}{return {buffer:this.work,offset:(this.wetOffset+2*this.words)*4};}
 /** Two words per tile (UNIFORM_STAGE_IMPORTANCE), written by every census.
  * Criteria that are off, and tiles shape alone decides, are scored only by
  * a census encoded with policy.reasons. */
 get importance():{readonly buffer:GPUBuffer;readonly offset:number}{return {buffer:this.work,offset:this.importanceOffset*4};}
 /** First importance word, after the crossing-cell words. */
 private readonly importanceOffset:number;
 /** setSolid marked a coupled tile: the promotion passes run. */
 private solid=false;
 /** Bound without the surface stage's held corrections: zeros. */
 private readonly noHeld?:GPUBuffer;
 /** First held-correction word of the bound buffer. */
 private readonly heldWord:number;
 private encoded=false;
 /** Tiles forced into the band by the next census: GPU producer words
  * [0, words) (joinTarget), then host words [words, 2·words) (join). decide
  * ors both into the band; the census clears them once it has decided.
  * Words [2·words, 3·words) are the builder's static h tiles (setStatic),
  * kept: the build makes them h whatever the band holds, so the residency
  * certificate seeds them (pageSeed). */
 private readonly joinTiles:GPUBuffer;
 /** Host joins accumulated since the last encode, uploaded by it. */
 private hostJoin?:Uint32Array<ArrayBuffer>;
 /** Join words may be non-zero: every census clears them after decide. */
 private joinLive=false;
 /** Zero source uniform when the host binds none: no source, no mark. */
 private readonly noSource?:GPUBuffer;
 /** The band decided by the last encode, for the GPU layout builder. Words
  * [0, HEADER) of the buffer are the census header its receipt carries. */
 get bandBits():UniformMixedBandBits{return {buffer:this.work,wordOffset:HEADER,headerWords:HEADER,auditWord:this.nearOffset+this.words};}
 /** source: the host's scene uniform (UMSourceParams prefix ABI, as the
  * mixed frame's sourceParams). decide marks every tile this frame's drop or
  * inflow plug can fill (uvSourcePhi, frame dt dimsDt.w) as band, so the
  * source lands on h owners. Its values are the ones bound when the census
  * runs: encode the census after the frame's params upload. Without it
  * nothing is marked.
  * held: the surface stage's held-vertex corrections (UniformMixedSurface.held);
  * without it every 4h vertex reads as stored. */
 constructor(private readonly device:GPUDevice,readonly ownership:UniformMixedOwnership,volume:GPUTexture,phi:GPUTexture,extended:GPUTexture,source?:GPUBuffer,held?:{readonly buffer:GPUBuffer;readonly word:number}){
  const tiles=ownership.capacity.tiles;
  this.words=Math.ceil(tiles/32);
  const t=ownership.capacity.lattice.dimensions.map(n=>n/4);
  this.wetOffset=HEADER+this.words+8*tiles;
  this.nearOffset=this.wetOffset+2*this.words+tiles;
  this.pages=uniformMixedPageCount(ownership.capacity.lattice);
  // After wet and active: band reasons (a word per tile), then near bits
  // and the audit word (cleared per census), then a packed near-tile box
  // per page.
  // Append two crossing-cell words per tile; existing diagnostic offsets stay stable.
  // Then two importance words per tile and the importance state.
  this.importanceOffset=this.nearOffset+this.words+1+this.pages+2*tiles;
  const workBytes=(this.importanceOffset+2*tiles+importanceStateWords(tiles))*4,readBytes=(HEADER+this.words)*4;
  this.work=device.createBuffer({label:"Uniform dynamic ownership census",size:workBytes,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
  this.readback=device.createBuffer({label:"Uniform dynamic ownership readback",size:readBytes,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  this.params=device.createBuffer({label:"Uniform dynamic ownership policy",size:96,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
  this.solidTiles=device.createBuffer({label:"Uniform dynamic ownership solid tiles",size:this.words*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
  this.staticSolidTiles=device.createBuffer({label:"Uniform dynamic ownership static solid tiles",size:this.words*4,usage:GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
  this.joinTiles=device.createBuffer({label:"Uniform dynamic ownership join tiles",size:3*this.words*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
  if(source&&source.size<SOURCE_PARAMS_BYTES)throw new Error(`Dynamic ownership source params hold ${source.size} bytes, expected at least ${SOURCE_PARAMS_BYTES}`);
  if(!source)this.noSource=device.createBuffer({label:"Uniform dynamic ownership no source",size:SOURCE_PARAMS_BYTES,usage:GPUBufferUsage.UNIFORM});
  const heldBytes=4*(t[0]!+1)*(t[1]!+1)*(t[2]!+1);
  if(held&&held.buffer.size<4*held.word+heldBytes)throw new Error(`Dynamic ownership held corrections hold ${held.buffer.size} bytes, expected at least ${4*held.word+heldBytes}`);
  if(!held)this.noHeld=device.createBuffer({label:"Uniform dynamic ownership no held corrections",size:heldBytes,usage:GPUBufferUsage.STORAGE});
  this.heldWord=held?.word??0;
  const boundsBytes=(6*CUBE_LEVELS*tiles+(t[0]!+1)*(t[1]!+1)*(t[2]!+1))*4;
  this.bounds=device.createBuffer({label:"Uniform dynamic ownership bound cubes",size:boundsBytes,usage:GPUBufferUsage.STORAGE});
  this.allocatedBytes=workBytes+boundsBytes+readBytes+96+this.words*16+(this.noSource?SOURCE_PARAMS_BYTES:0)+(this.noHeld?heldBytes:0);
  this.resources=uniformDetailBindLayout(device,{entries:[
   ...[0,1,2].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float" as const,viewDimension:"3d" as const}})),
   {binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},
   {binding:4,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},
   {binding:5,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},
   {binding:6,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},
   {binding:7,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},
   {binding:8,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},
   {binding:9,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},
  ]});
  this.group=uniformDetailGroup(device,{layout:this.resources,entries:[
   {binding:0,resource:volume},{binding:1,resource:phi},
   {binding:2,resource:extended},{binding:3,resource:{buffer:this.work}},{binding:4,resource:{buffer:this.params}},
   {binding:5,resource:{buffer:this.solidTiles}},{binding:6,resource:{buffer:this.bounds}},{binding:7,resource:{buffer:this.joinTiles}},
   {binding:8,resource:{buffer:source??this.noSource!,size:SOURCE_PARAMS_BYTES}},
   {binding:9,resource:{buffer:held?.buffer??this.noHeld!}},
  ]});
 }
 async initialize():Promise<void>{
  const h=this.ownership.capacity.lattice.cellSize_m,cap=UNIFORM_MIXED_DYNAMIC_DISTANCE_CAP;
  const module=uniformDetailModule(this.device,{label:"Uniform dynamic ownership census",code:uniformMixedTopologyWGSL(this.ownership.capacity,0)+/* wgsl */`
@group(1) @binding(0) var volume:texture_3d<f32>;
@group(1) @binding(1) var phi:texture_3d<f32>;
@group(1) @binding(2) var velocity:texture_3d<f32>;
@group(1) @binding(3) var<storage,read_write> census:array<atomic<u32>>;
// step: horizon dt, surface tolerance, full and empty tolerance; surface:
// fast travel, boundary travel, frames in the horizon; flow: the velocity
// gravity adds per frame (m/s), and the departure margin in cells;
// importance: thin thickness (h), strain and rotation thresholds (per step),
// approach horizon (steps); shaping: criteria that are on (bits 0..5,
// UNIFORM_DETAIL_CRITERIA), bit 8 bulk, bit 9 sources and bit 11 the
// displacement shape metric, bit 12 surface admission, bits 13..14 its tile distance, hold steps, the seed budget's band
// tiles (all ones: none), the score byte that keeps a hold.
struct DynamicPolicy {step:vec4f,reach:vec4u,surface:vec4f,flow:vec4f,importance:vec4f,shaping:vec4u}
@group(1) @binding(4) var<uniform> policy:DynamicPolicy;
@group(1) @binding(5) var<storage,read> solidTiles:array<u32>;
@group(1) @binding(6) var<storage,read_write> bounds:array<u32>;
// Joined tiles: GPU producer words, then host words (joinTarget, join); then
// the builder's static h tiles (setStatic).
@group(1) @binding(7) var<storage,read> joinTiles:array<u32>;
${uniformMixedSourceWGSL(8)}
// What a rebuild would have added to a 4h vertex the surface stage's travel
// gate holds (umHeldIndex; 0 elsewhere): a held value keeps its cells' zero
// set and stops being a distance, and the shape criterion measures distances.
@group(1) @binding(9) var<storage,read> heldDistance:array<u32>;
fn umHeldCorrection(p:vec3u)->f32{let q=p/4u;let d=UM_D/4u+vec3u(1);return bitcast<f32>(heldDistance[${this.heldWord}u+q.x+d.x*(q.y+d.y*q.z)]);}
// The census tail follows the frame's last phi resolve: every texel of a
// mixed-stencil tile holds umVertexValue, and the census reads only vertices
// of an h tile's closure or 4-aligned ones, so umVertexValue is this load.
// A reader knows what it walks, so no load tests its address: a tile corner
// is canonical (the base block, at every capacity), an h tile's vertex is in
// the h texture (the tile's own, or a neighbour's inside the detail ring).
fn umLoadCorner(p:vec3u)->f32{return ${UNIFORM_DETAIL_4H_LOAD}textureLoad(phi,vec3i(p),0).x;}
fn umLoadFine(p:vec3u)->f32{return textureLoad(phi,vec3i(p),0).x;}
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
// Inclusive prefix sum of interface flags over tiles, with a zero border
// plane, after the cube table. Each pass writes disjoint entries and reads
// only earlier passes': plain words, not census atomics.
const PX:u32=UM_T.x+1u;const PY:u32=UM_T.y+1u;const PZ:u32=UM_T.z+1u;
fn prefixIndex(p:vec3u)->u32{return 6u*CUBE_LEVELS*UM_TILES+p.x+PX*(p.y+PY*p.z);}
// The zero border planes are implicit: never cleared, written or loaded.
fn prefixLoad(p:vec3u)->u32{if(any(p==vec3u(0u))){return 0u;}return bounds[prefixIndex(p)];}
// Per required tile: cells between each face and the nearest crossing owner,
// one nibble per face (-x,-y,-z,+x,+y,+z), and its own travel in whole h
// cells per step in the top byte; all ones for a tile that is not required.
fn gapIndex(t:u32)->u32{return ${HEADER}u+WORDS+6u*UM_TILES+t;}
// Per boundary tile: travel in h per step along each direction (-x,-y,-z,
// +x,+y,+z), five bits each; all ones for any other tile. Its dilation
// follows the flow, not a sphere: a climbing sheet refines the tiles above.
fn travelIndex(t:u32)->u32{return gapIndex(UM_TILES)+t;}
// One bit per tile: liquid, a non-air owner (wet); then solid-coupled tiles
// that are wet or in the decided band (active).
fn wetIndex(w:u32)->u32{return travelIndex(UM_TILES)+w;}
fn activeIndex(w:u32)->u32{return wetIndex(WORDS)+w;}
fn reasonIndex(t:u32)->u32{return activeIndex(WORDS)+t;}
// Band reasons for the tiles layer (policy.surface.w, UNIFORM_STAGE_REASON).
fn recordReasons()->bool{return policy.surface.w>0.0;}
fn recordReason(tile:u32,reason:u32){if(recordReasons()){atomicStore(&census[reasonIndex(tile)],reason);}}
// Residency: one bit per near tile (an h tile, or a 4h tile with V != 0 or a
// corner phi below the frame plan seed's 4h-per-width threshold: a tile the
// seed can mark), the audit word (UNIFORM_MIXED_RESIDENCY_AUDIT), and per
// page the packed box of its near and band tiles (pageSeed).
fn nearIndex(w:u32)->u32{return reasonIndex(UM_TILES)+w;}
fn auditIndex()->u32{return nearIndex(WORDS);}
fn pageBoxIndex(page:u32)->u32{return auditIndex()+1u+page;}
// Exact union of crossing owners, in unit cells (x + 4*y + 16*z).
// A crossing 4h owner covers all 64 cells: do not invent sub-cell detail.
fn crossingIndex(t:u32,k:u32)->u32{return pageBoxIndex(UM_PAGES)+2u*t+k;}
// Detail importance (UNIFORM_STAGE_IMPORTANCE): two words per tile. classify
// writes the scores a tile's own samples decide (shape, thin, impact) with
// its crossing and wet flags; importance adds the neighbour criteria and the
// decision. Kept across censuses after them: the steps each tile's hold has
// left, the budget's score cutoff and its histogram of triggering tiles.
fn importanceIndex(t:u32,k:u32)->u32{return crossingIndex(UM_TILES,0u)+2u*t+k;}
fn holdIndex(t:u32)->u32{return importanceIndex(UM_TILES,0u)+t;}
fn cutoffIndex()->u32{return holdIndex(UM_TILES);}
fn binIndex(b:u32)->u32{return cutoffIndex()+1u+b;}
const SHAPE:u32=0u;const THIN:u32=1u;const STRAIN:u32=2u;const ROTATION:u32=3u;const IMPACT:u32=4u;const APPROACH:u32=5u;
fn criterionOn(k:u32)->bool{return ((policy.shaping.x>>k)&1u)!=0u;}
fn triggerBit(k:u32)->u32{return 1u<<(${IMPORTANCE.triggeredShift}u+k);}
// A criterion that is off is scored only for a layer.
fn scored(k:u32)->bool{return criterionOn(k)||recordReasons();}
fn scoreByte(score:f32)->u32{return u32(clamp(score*${IMPORTANCE.scoreOne}.0,0.0,255.0));}
fn budgetOn()->bool{return policy.shaping.z!=0xffffffffu;}
// No criterion, hold or layer can require a tile (encode): the interface
// table is all zeros, and importance, the prefix and the cubes that would
// build it are not launched.
fn unseeded()->bool{return (policy.shaping.x&1024u)!=0u;}
// Shape measures an h tile by umSurfaceDisplacement, not by value error.
fn shapeDisplacement()->bool{return (policy.shaping.x&2048u)!=0u;}
// A tile shape requires needs no other score, unless a layer shows them or
// the budget ranks it by its top one.
fn scoreAll()->bool{return recordReasons()||budgetOn();}
// The lowest score, over its threshold, that can matter: a trigger, or with
// a hold the score that keeps it; a layer shows every one.
fn scoreFloor()->f32{
 if(recordReasons()){return 0.0;}
 return select(1.0,f32(policy.shaping.w)/${IMPORTANCE.scoreOne}.0,policy.shaping.y>0u);
}
// Thin is measured for crossing tiles shape does not already require: with
// shape on at tolerance 0 that is none, and classify samples nothing.
fn thinWanted()->bool{
 if(policy.importance.x<=0.0||!scored(THIN)){return false;}
 return scoreAll()||!criterionOn(SHAPE)||policy.step.y>0.0;
}
fn orderKey(x:f32)->u32{let b=bitcast<u32>(x);return select(b|0x80000000u,~b,(b&0x80000000u)!=0u);}
fn orderValue(k:u32)->f32{return bitcast<f32>(select(~k,k&0x7fffffffu,(k&0x80000000u)!=0u));}
// One tile's classification: ordered velocity keys, nibble distances from
// each face to the nearest crossing owner (gap) and surface owner (reach),
// flags (bit 0 surface owner, bit 1 phi sign change) and 4h surface error.
// deep: ordered keys of the deepest liquid (-phi) and air (+phi) vertex
// around the tile (umDepthSample), zero when not sampled.
struct TileClass{low:array<u32,3>,high:array<u32,3>,gap:array<u32,6>,reach:array<u32,6>,flags:u32,error:u32,crossing:vec2u,deep:vec2u}
fn umEmptyClass()->TileClass{return TileClass(array<u32,3>(0xffffffffu,0xffffffffu,0xffffffffu),array<u32,3>(0u,0u,0u),array<u32,6>(15u,15u,15u,15u,15u,15u),array<u32,6>(15u,15u,15u,15u,15u,15u),0u,0u,vec2u(0u),vec2u(0u));}
var<workgroup> mixedTile:atomic<u32>;
var<workgroup> tileCrossing:array<atomic<u32>,2>;
var<workgroup> tileLow:array<atomic<u32>,3>;
var<workgroup> tileHigh:array<atomic<u32>,3>;
var<workgroup> tileGap:array<atomic<u32>,6>;
var<workgroup> tileReach:array<atomic<u32>,6>;
var<workgroup> tileError:atomic<u32>;
var<workgroup> tileWide:atomic<u32>;
var<workgroup> tileDeep:array<atomic<u32>,2>;
// Largest travel of the workgroup's tiles (census[18]), flushed once per group.
var<workgroup> groupReach:atomic<u32>;
// census[8..11] (deficit and air extremes) gathered per workgroup: nearly
// every coarse owner is air, and one global atomicMax each on one word
// serialised the container sweep. Non-negative: ordered as bits, and a zero
// is max's identity, so only non-zero extremes are flushed.
var<workgroup> groupExtreme:array<atomic<u32>,4>;
fn resetGroupCensus(){atomicStore(&groupReach,0u);for(var k=0u;k<4u;k++){atomicStore(&groupExtreme[k],0u);}}
fn flushGroupCensus(){
 atomicMax(&census[18],atomicLoad(&groupReach));
 for(var k=0u;k<4u;k++){let e=atomicLoad(&groupExtreme[k]);if(e!=0u){atomicMax(&census[8u+k],e);}}
}
const UM_H:f32=${Math.min(...h)};
const MAX_H:f32=${Math.max(...h)};
// A vertex of an h tile that a 4h tile shares: wide holds the 4h tiles of
// the tile's 3^3 neighbourhood, a bit each (umCorner(k,3u)-1).
fn umWideShared(wide:u32,local:vec3u)->bool{
 if(wide==0u){return false;}
 let lo=select(vec3i(0),vec3i(-1),local==vec3u(0u));let hi=select(vec3i(0),vec3i(1),local==vec3u(4u));
 for(var z=lo.z;z<=hi.z;z++){for(var y=lo.y;y<=hi.y;y++){for(var x=lo.x;x<=hi.x;x++){
  if((wide&(1u<<u32(x+1+3*(y+1+3*(z+1)))))!=0u){return true;}
 }}}
 return false;
}
// How far an h tile's surface moves when its eight 4h corners carry it, in
// h: at every zero crossing of an h edge, the corners' trilinear phi over
// its gradient (the distance to their zero set, to first order). The stored
// corners, with no held correction: their zero set is what 4h keeps.
fn umSurfaceDisplacement(origin:vec3u,lane:u32)->f32{
 var c:array<f32,8>;var worst=0.0;
 for(var k=0u;k<8u;k++){c[k]=umLoadCorner(origin+umCorner(k,2u)*4u);}
 let low=vec4f(c[0],c[2],c[4],c[6]);let high=vec4f(c[1],c[3],c[5],c[7]);let dx=high-low;
 for(var i=lane;i<125u;i+=64u){
  let v=umCorner(i,5u);let value=umLoadFine(origin+v);
  for(var a=0u;a<3u;a++){
   if(v[a]>=4u){continue;}
   var n=v;n[a]+=1u;let next=umLoadFine(origin+n);
   if((value<0.0)==(next<0.0)){continue;}
   var t=vec3f(v)*0.25;t[a]+=0.25*value/(value-next);
   let x0=mix(low,high,t.x);
   let y0=mix(vec2f(x0.x,x0.z),vec2f(x0.y,x0.w),t.y);
   let gx=mix(vec2f(dx.x,dx.z),vec2f(dx.y,dx.w),t.y);
   // Per tile width (4h), so the distance in h is 4 |phi| / |gradient|.
   let gradient=vec3f(mix(gx.x,gx.y,t.z),mix(x0.y-x0.x,x0.w-x0.z,t.z),y0.y-y0.x);
   worst=max(worst,4.0*abs(mix(y0.x,y0.y,t.z))/max(length(gradient),4.0e-3*UM_H));
  }
 }
 return worst;
}
// Surface error of the tile at 4h, in h (non-negative, ordered as bits).
// wide: an h tile's 4h neighbours (umWideShared).
fn umResolutionError(tile:u32,lane:u32,width:u32,wide:u32)->f32{
 let origin=umTileCoord(tile)*4u;var worst=0.0;
 if(width==1u){
  if(shapeDisplacement()){return umSurfaceDisplacement(origin,lane);}
  // The corners as distances: one a 4h tile shares may be held.
  var corners:array<f32,8>;
  for(var k=0u;k<8u;k++){let c=origin+umCorner(k,2u)*4u;corners[k]=umLoadCorner(c);if(wide!=0u){corners[k]+=umHeldCorrection(c);}}
  for(var i=lane;i<125u;i+=64u){
   let v=umCorner(i,5u);let t=vec3f(v)*0.25;
   // A vertex a 4h tile shares is a corner or the corners' own trilinear
   // value: 4h loses nothing there, and a held corner's is not a distance.
   if(umWideShared(wide,v)){continue;}
   let x0=mix(vec4f(corners[0],corners[2],corners[4],corners[6]),vec4f(corners[1],corners[3],corners[5],corners[7]),t.x);
   let y0=mix(vec2f(x0.x,x0.z),vec2f(x0.y,x0.w),t.y);
   let interpolated=mix(y0.x,y0.y,t.z);let value=umLoadFine(origin+v);
   if(min(abs(value),abs(interpolated))<2.0*UM_H){worst=max(worst,abs(value-interpolated)/UM_H);}
  }
  return worst;
 }
 if(width!=4u){return 3.0e38;}
 if(lane>=8u){return 0.0;}
 // Quadratic interpolation error at the midpoint is |second difference|/8;
 // halved so one tolerance gives refine-at-2x hysteresis.
 let c=origin+umCorner(lane,2u)*4u;let centre=umLoadCorner(c);
 if(abs(centre)>=8.0*UM_H){return 0.0;}
 for(var a=0u;a<3u;a++){
  if(c[a]<4u||c[a]+4u>UM_D[a]){continue;}
  var lo=c;lo[a]-=4u;var hi=c;hi[a]+=4u;
  worst=max(worst,abs(umLoadCorner(hi)-2.0*centre+umLoadCorner(lo))/(16.0*UM_H));
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
struct Boundary{score:f32,triggered:bool}
fn umBoundary(p:vec3u,width:u32,c:TileClass)->Boundary{
 var result=Boundary(0.0,false);
 let threshold=policy.surface.y*select(1.0,0.5,width==1u);
 if(threshold<=0.0){return result;}
 let least=scoreFloor()*threshold;
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
  if(travel[k]<max(least,along)){continue;}
  let reach=margin-f32(c.reach[k]);
  if(reach<0.0){continue;}
  let beyond=select(p[a],UM_T[a]-1u-p[a],high);
  var met=((policy.reach.z>>k)&1u)!=0u&&4.0*f32(beyond)<=reach;
  for(var n=1u;!met&&n<=min(beyond,CAP)&&4.0*f32(n-1u)<=reach;n++){
   var q=p;q[a]=select(p[a]-n,p[a]+n,high);
   met=umSolidTile(q);
  }
  if(!met){continue;}
  result.score=max(result.score,travel[k]/threshold);
  if(travel[k]>=threshold){result.triggered=true;if(!recordReasons()){return result;}}
 }
 // Lift: on a closed vertical wall, moving up faster than along the wall.
 if(policy.reach.w==0u){return result;}
 let up=select(travel[1],travel[4],policy.reach.w==1u);
 if(up<least){return result;}
 for(var k=0u;k<6u;k++){
  let a=k%3u;if(a==1u){continue;}
  let beyond=select(p[a],UM_T[a]-1u-p[a],k>=3u);
  let along=max(travel[2u-a],travel[5u-a]);
  if(((policy.reach.z>>k)&1u)!=0u&&beyond==0u&&up>=along){result.score=max(result.score,up/threshold);if(up>=threshold){result.triggered=true;}}
 }
 return result;
}
// Thin features: the deepest liquid and the deepest air among the 4-aligned
// vertices one tile around the tile (its own corners and the next ring). A
// sheet, film or neck holds no vertex deeper than half its thickness, and
// neither does a closing air gap. Lane i of 64 samples one vertex.
fn umDepthSample(tile:u32,i:u32)->vec2u{
 let v=vec3i(umTileCoord(tile))+vec3i(umCorner(i,4u))-vec3i(1);
 if(any(v<vec3i(0))||any(v>vec3i(UM_T))){return vec2u(0u);}
 let value=umLoadCorner(vec3u(v)*4u);
 return vec2u(orderKey(-value),orderKey(value));
}
// Interface: an owner whose corner phi changes sign. V/phi disagreement
// without one is counted, not refined: 4h V error is at or below fine on the
// same cells (plan, same-cell A/B).
fn umClassifyOwner(tile:u32,width:u32,lane:u32,c:ptr<function,TileClass>){
 let surfaceAuthority=(policy.shaping.x&32768u)!=0u;
 let side=4u/width;
 let origin=umTileCoord(tile)*4u+umCorner(lane,side)*width;
 // By the owner's width: a 4h owner's texels are canonical, an h owner's the h texture's.
 let coarseOwner=width==4u;
 var v=0.0;if(!surfaceAuthority){if(coarseOwner){v=${UNIFORM_DETAIL_4H_LOAD}textureLoad(volume,vec3i(origin),0).x;}else{v=textureLoad(volume,vec3i(origin),0).x;}}
 var inside=0u;var deep=0u;var lowPhi=3.0e38;var highPhi=-3.0e38;
 for(var k=0u;k<8u;k++){
  var value=0.0;if(coarseOwner){value=umLoadCorner(origin+umCorner(k,2u)*4u);}else{value=umLoadFine(origin+umCorner(k,2u)*width);}
  lowPhi=min(lowPhi,value);highPhi=max(highPhi,value);
  if(value<0.0){inside++;}
  if(value< -2.0*MAX_H*f32(width)){deep++;}
  // Flag 8: the frame plan seed would mark this owner (residency near).
  if(value<4.0*MAX_H*f32(width)){(*c).flags|=8u;}
 }
 if(!surfaceAuthority&&v!=0.0){(*c).flags|=8u;}
 // Signed bounds of the extended face velocities the next trace samples.
 // An h owner's faces are single patches anchored at its origin (+) and one
 // cell below it (-), whatever the neighbour (umFace).
 var owner=UMOwner();if(width!=1u){owner=umOwnerAt(vec3i(origin));}
 for(var axis=0u;axis<3u;axis++){
  var low=3.0e38;var high=-3.0e38;
  for(var sign=-1;sign<=1;sign+=2){
   var count=1u;var anchor=vec3i(origin);if(sign<0){anchor[axis]-=1;}
   if(width!=1u){count=umFace(owner,axis,sign,0u).count;}
   for(var part=0u;part<count;part++){
    // A 4h-wide patch is anchored on a canonical face; a seam part is an h face.
    var widePatch=false;
    if(width!=1u){let face=umFace(owner,axis,sign,part);anchor=face.anchor;widePatch=face.width==4u;}
    if(anchor[axis]<0){continue;}
    var u=0.0;if(widePatch){u=${UNIFORM_DETAIL_4H_LOAD}textureLoad(velocity,anchor,0)[axis];}else{u=textureLoad(velocity,anchor,0)[axis];}
    // A non-finite speed bounds nothing: saturate both ends.
    let finite=abs(u)<=3.0e38;
    low=min(low,select(-3.0e38,u,finite));high=max(high,select(3.0e38,u,finite));
   }
  }
  (*c).low[axis]=min((*c).low[axis],orderKey(low));(*c).high[axis]=max((*c).high[axis],orderKey(high));
 }
 // Largest interior deficit and largest air volume, per fine/coarse width.
 let coarse=select(0u,1u,width!=1u);
 if(!surfaceAuthority&&deep==8u){atomicMax(&groupExtreme[coarse],bitcast<u32>(max(1.0-v,0.0)));}
 if(!surfaceAuthority&&inside==0u){atomicMax(&groupExtreme[2u+coarse],bitcast<u32>(max(v,0.0)));}
 let local=umCorner(lane,side)*width;
 // Include both sides of an interface on a tile plane despite roundoff.
 let epsilon=1e-6*MAX_H;
 let crosses=select(inside!=0u&&inside!=8u,lowPhi<=epsilon&&highPhi>=-epsilon&&lowPhi<highPhi,surfaceAuthority);
 if(crosses){
  if(width==4u){(*c).crossing=vec2u(0xffffffffu);}
  else{(*c).crossing[lane/32u]|=1u<<(lane%32u);}
  for(var a=0u;a<3u;a++){(*c).gap[a]=min((*c).gap[a],local[a]);(*c).gap[3u+a]=min((*c).gap[3u+a],4u-width-local[a]);}
 }
 let interior=!crosses&&inside==8u&&(surfaceAuthority||v>=1.0-policy.step.z);
 let air=!crosses&&inside==0u&&(surfaceAuthority||v<=policy.step.w);
 // Flag 4: liquid (any owner that is not air).
 if(!air){(*c).flags|=4u;}
 if(!interior&&!air){
  for(var a=0u;a<3u;a++){(*c).reach[a]=min((*c).reach[a],local[a]);(*c).reach[3u+a]=min((*c).reach[3u+a],4u-width-local[a]);}
  // Bit 2: the geometric surface itself (a phi sign change) is in this owner.
  (*c).flags|=select(1u,3u,crosses);
  // Why a coarse owner is interface: V between the tolerances, or a phi sign change.
  if(width!=1u){
   if(!surfaceAuthority&&v>policy.step.w&&v<1.0-policy.step.z){atomicAdd(&census[6],1u);}
   if(crosses){atomicAdd(&census[7],1u);}
   if(!surfaceAuthority&&inside==8u&&v<=policy.step.w){atomicAdd(&census[3],1u);}
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
 // Non-negative: ordered as bits. One global atomic per workgroup (callers).
 atomicMax(&groupReach,bitcast<u32>(reach));
 // Hysteresis: a 4h tile returns to h only below half the travel.
 let fast=policy.surface.x>0.0&&speed>=policy.surface.x*select(0.5,1.0,width==1u);
 // The scores the tile's own samples decide. importance (the pass) adds the
 // neighbour criteria and decides: until then the tile's crossing cells, gap
 // and directed travel are stored as if it were required.
 var w0=0u;var w1=select(0u,${IMPORTANCE.crossing}u,crossing)|select(0u,${IMPORTANCE.wet}u,(c.flags&4u)!=0u);
 var gap=0xffffffffu;var directed=0xffffffffu;
 if(crossing){
  // Shape: the surface error 4h would make over the tolerance; at tolerance 0
  // every surface tile (a fast one aside) saturates.
  let tolerant=policy.step.y>0.0;let error=bitcast<f32>(c.error);
  let shaped=!fast&&criterionOn(SHAPE)&&!(tolerant&&error<=policy.step.y);
  if(!fast&&scored(SHAPE)){w0|=select(255u,scoreByte(error/max(policy.step.y,1e-30)),tolerant);}
  if(shaped){w1|=triggerBit(SHAPE);}
  let more=!shaped||scoreAll();
  if(more&&thinWanted()&&all(c.deep!=vec2u(0u))){
   let thickness=2.0*max(min(orderValue(c.deep.x),orderValue(c.deep.y)),0.0)/UM_H;let thin=policy.importance.x;
   w0|=scoreByte(thin/max(thickness,0.25*thin))<<8u;
   if(thickness<thin&&criterionOn(THIN)){w1|=triggerBit(THIN);}
  }
  // A boundary rule holds a phi surface whatever its speed. Partial V alone
  // does not qualify: airborne spray refined at the wall falls under the
  // dust threshold and is discarded (128³ dam: 9 cells in three steps).
  if(more&&scored(IMPACT)){
   let boundary=umBoundary(umTileCoord(tile),width,c);
   w1|=scoreByte(boundary.score);
   if(boundary.triggered&&criterionOn(IMPACT)){
    w1|=triggerBit(IMPACT);directed=0u;
    for(var k=0u;k<6u;k++){directed|=min(u32(ceil(umDirectionTravel(c,k))),31u)<<(5u*k);}
   }
  }
  gap=min(u32(ceil(speed)),255u)<<24u;for(var k=0u;k<6u;k++){gap|=c.gap[k]<<(4u*k);}
 }
 for(var k=0u;k<2u;k++){atomicStore(&census[crossingIndex(tile,k)],c.crossing[k]);atomicStore(&census[importanceIndex(tile,k)],select(w0,w1,k==1u));}
 atomicStore(&census[gapIndex(tile)],gap);
 atomicStore(&census[travelIndex(tile)],directed);
 for(var a=0u;a<3u;a++){atomicStore(&census[boundIndex(tile,a)],c.low[a]);atomicStore(&census[boundIndex(tile,3u+a)],c.high[a]);}
 if(crossing){atomicAdd(&census[0],1u);}
 if((c.flags&4u)!=0u){atomicOr(&census[wetIndex(tile/32u)],1u<<(tile%32u));}
 // Residency: every h tile is near. A near tile of a page the last census
 // left absent is a closure violation: that frame skipped it as far air.
 let near=width==1u||(c.flags&8u)!=0u;
 if(near){atomicOr(&census[nearIndex(tile/32u)],1u<<(tile%32u));if(!umTileResident(tile)){atomicOr(&census[auditIndex()],1u);}}
}
// h tiles: one workgroup per tile, one lane per owner. A fixed grid strides
// over the live h list; umCounts is uniform, so the loop keeps its barriers.
@compute @workgroup_size(64) fn classify(@builtin(workgroup_id) gid:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 for(var job=gid.x;job<umCounts.x;job+=groups.x){
 let tile=umTopology[UM_TILES+job];
 if(lane==0u){resetGroupCensus();atomicStore(&mixedTile,0u);for(var k=0u;k<2u;k++){atomicStore(&tileCrossing[k],0u);}for(var a=0u;a<3u;a++){atomicStore(&tileLow[a],0xffffffffu);atomicStore(&tileHigh[a],0u);atomicStore(&tileGap[a],15u);atomicStore(&tileGap[3u+a],15u);atomicStore(&tileReach[a],15u);atomicStore(&tileReach[3u+a],15u);}atomicStore(&tileError,0u);atomicStore(&tileWide,0u);for(var k=0u;k<2u;k++){atomicStore(&tileDeep[k],0u);}}workgroupBarrier();
 // The h list holds h tiles only: one lane per owner.
 const width=1u;
 // The tile's 4h neighbours, for the shape criterion (umWideShared).
 if(lane<27u&&policy.step.y>0.0&&!shapeDisplacement()){
  let n=vec3i(umTileCoord(tile))+vec3i(umCorner(lane,3u))-vec3i(1);
  if(all(n>=vec3i(0))&&all(n<vec3i(UM_T))&&umTileWidth(umTileAt(vec3u(n)))!=1u){atomicOr(&tileWide,1u<<lane);}
 }
 {
  var c=umEmptyClass();umClassifyOwner(tile,width,lane,&c);
  for(var a=0u;a<3u;a++){atomicMin(&tileLow[a],c.low[a]);atomicMax(&tileHigh[a],c.high[a]);}
  for(var k=0u;k<6u;k++){atomicMin(&tileGap[k],c.gap[k]);atomicMin(&tileReach[k],c.reach[k]);}
  atomicOr(&mixedTile,c.flags);
  for(var k=0u;k<2u;k++){if(c.crossing[k]!=0u){atomicOr(&tileCrossing[k],c.crossing[k]);}}
 }
 workgroupBarrier();
 if((atomicLoad(&mixedTile)&2u)!=0u&&policy.step.y>0.0){atomicMax(&tileError,bitcast<u32>(umResolutionError(tile,lane,width,atomicLoad(&tileWide))));}
 if((atomicLoad(&mixedTile)&2u)!=0u&&thinWanted()){let deep=umDepthSample(tile,lane);for(var k=0u;k<2u;k++){atomicMax(&tileDeep[k],deep[k]);}}
 workgroupBarrier();
 if(lane==0u){
  var c=umEmptyClass();
  for(var a=0u;a<3u;a++){c.low[a]=atomicLoad(&tileLow[a]);c.high[a]=atomicLoad(&tileHigh[a]);}
  for(var k=0u;k<6u;k++){c.gap[k]=atomicLoad(&tileGap[k]);c.reach[k]=atomicLoad(&tileReach[k]);}
  c.flags=atomicLoad(&mixedTile);c.error=atomicLoad(&tileError);
  c.crossing=vec2u(atomicLoad(&tileCrossing[0]),atomicLoad(&tileCrossing[1]));
  c.deep=vec2u(atomicLoad(&tileDeep[0]),atomicLoad(&tileDeep[1]));
  umFinishTile(tile,width,c);
  flushGroupCensus();
 }
 }
}
// Single-owner (4h) tiles: one lane per tile. A workgroup per tile left 63
// of its 64 lanes idle over most of the domain.
@compute @workgroup_size(64) fn classifyCoarse(@builtin(global_invocation_id) gid:vec3u,@builtin(local_invocation_index) lane:u32){
 if(lane==0u){resetGroupCensus();}
 workgroupBarrier();
 let index=gid.x+umDispatchX*64u*gid.y;
 if(index<umCounts.y){
  let tile=umTopology[UM_TILES+umCounts.x+index];
  var c=umEmptyClass();umClassifyOwner(tile,4u,0u,&c);
  if((c.flags&2u)!=0u&&policy.step.y>0.0){var error=0.0;for(var l=0u;l<8u;l++){error=max(error,umResolutionError(tile,l,4u,0u));}c.error=bitcast<u32>(error);}
  if((c.flags&2u)!=0u&&thinWanted()){for(var i=0u;i<64u;i++){c.deep=max(c.deep,umDepthSample(tile,i));}}
  umFinishTile(tile,4u,c);
 }
 workgroupBarrier();
 if(lane==0u){flushGroupCensus();}
}
// The neighbour criteria and the decision, one lane per tile, after every
// tile is classified. It reads other tiles' velocity keys, wet bits and the
// solid mask only: nothing this pass writes.
fn umTileWet(t:u32)->bool{return (atomicLoad(&census[wetIndex(t/32u)])&(1u<<(t%32u)))!=0u;}
// Travel of tile t's fastest face along direction k (-x,-y,-z,+x,+y,+z), in h per step.
fn umTileTravel(t:u32,k:u32)->f32{
 let a=k%3u;
 if(k<3u){return max(-orderValue(atomicLoad(&census[boundIndex(t,a)])),0.0)*policy.step.x/H[a];}
 return max(orderValue(atomicLoad(&census[boundIndex(t,3u+a)])),0.0)*policy.step.x/H[a];
}
fn umTileSpeed(t:u32)->f32{
 var speed=0.0;
 for(var a=0u;a<3u;a++){speed=max(speed,max(abs(orderValue(atomicLoad(&census[boundIndex(t,a)]))),abs(orderValue(atomicLoad(&census[boundIndex(t,3u+a)]))))*policy.step.x/H[a]);}
 return speed;
}
// The tile's velocity: the middle of its face velocity bounds.
fn umTileFlow(t:u32)->vec3f{
 var u=vec3f(0.0);
 for(var a=0u;a<3u;a++){u[a]=0.5*(orderValue(atomicLoad(&census[boundIndex(t,a)]))+orderValue(atomicLoad(&census[boundIndex(t,3u+a)])));}
 return u;
}
// Deformation and rotation per step: dt·‖sym ∇u‖ and dt·‖curl u‖, with ∇u
// from the tile velocities of the wet face neighbours (centred where both
// are wet, one-sided at a surface or wall, zero with neither). Air tiles
// hold an extension, not the flow.
fn umFlowActivity(p:vec3u,tile:u32)->vec2f{
 let centre=umTileFlow(tile);var g:array<vec3f,3>;
 for(var b=0u;b<3u;b++){
  var lo=centre;var hi=centre;var span=0.0;
  if(p[b]>0u){var q=p;q[b]-=1u;let t=umTileAt(q);if(umTileWet(t)){lo=umTileFlow(t);span+=4.0*H[b];}}
  if(p[b]+1u<UM_T[b]){var q=p;q[b]+=1u;let t=umTileAt(q);if(umTileWet(t)){hi=umTileFlow(t);span+=4.0*H[b];}}
  g[b]=vec3f(0.0);if(span>0.0){g[b]=(hi-lo)/span;}
 }
 var strain=0.0;
 for(var a=0u;a<3u;a++){for(var b=0u;b<3u;b++){let e=0.5*(g[b][a]+g[a][b]);strain+=e*e;}}
 let curl=vec3f(g[1].z-g[2].y,g[2].x-g[0].z,g[0].y-g[1].x);
 return policy.step.x*vec2f(sqrt(strain),length(curl));
}
// Approaching contact: the horizon over the steps until the surface in tile
// p closes the air ahead of it. Per direction it moves in, with air on that
// face of the tile (liquid there is already across it): march the tiles
// ahead to a solid tile, liquid beyond air (closing at both travels) or a
// closed wall. The distance is whole tiles plus the cells from the face to
// the tile's nearest crossing owner; under a cell is contact, the impact
// rule's.
fn umApproach(p:vec3u,tile:u32)->f32{
 let horizon=policy.importance.w;if(horizon<=0.0){return 0.0;}
 let word=atomicLoad(&census[gapIndex(tile)]);var best=0.0;
 for(var k=0u;k<6u;k++){
  let a=k%3u;let high=k>=3u;let travel=umTileTravel(tile,k);
  if(travel<=0.0){continue;}
  var airSide=true;
  for(var c=0u;c<4u;c++){
   var v=4u*p;v[a]+=select(0u,4u,high);v[(a+1u)%3u]+=4u*(c&1u);v[(a+2u)%3u]+=4u*(c>>1u);
   if(umLoadCorner(v)<0.0){airSide=false;}
  }
  if(!airSide){continue;}
  let beyond=select(p[a],UM_T[a]-1u-p[a],high);
  let gap=f32((word>>(4u*k))&15u);
  let limit=min(beyond,min(CAP,u32(ceil(min(horizon*travel,64.0)/4.0))+1u));
  var air=false;var unblocked=true;var gapCells=-1.0;var closing=travel;
  for(var n=1u;n<=limit;n++){
   var q=p;q[a]=select(p[a]-n,p[a]+n,high);let t=umTileAt(q);
   if(umSolidTile(q)){gapCells=4.0*f32(n-1u)+gap;unblocked=false;break;}
   if(!umTileWet(t)){air=true;continue;}
   // Liquid with no air before it is this surface's own.
   if(air){gapCells=4.0*f32(n-1u)+gap;closing+=umTileTravel(t,(k+3u)%6u);}
   unblocked=false;break;
  }
  if(unblocked&&limit==beyond&&((policy.reach.z>>k)&1u)!=0u){gapCells=4.0*f32(beyond)+gap;}
  if(gapCells>=1.0){best=max(best,horizon*closing/gapCells);}
 }
 return best;
}
fn umScoreByte(w0:u32,w1:u32,k:u32)->u32{
 if(k<4u){return (w0>>(8u*k))&255u;}
 return (w1>>(8u*(k-4u)))&255u;
}
@compute @workgroup_size(64) fn importance(@builtin(global_invocation_id) gid:vec3u){
 let tile=gid.x+umDispatchX*64u*gid.y;if(tile>=UM_TILES){return;}
 let own=vec2u(atomicLoad(&census[importanceIndex(tile,0u)]),atomicLoad(&census[importanceIndex(tile,1u)]));
 var w0=own.x;var w1=own.y;
 let crossing=(w1&${IMPORTANCE.crossing}u)!=0u;let wet=(w1&${IMPORTANCE.wet}u)!=0u;
 let p=umTileCoord(tile);let holding=policy.shaping.y>0u;
 let shaped=(w1&triggerBit(SHAPE))!=0u;
 var required=false;var remaining=0u;
 if(crossing||wet){
  if(!shaped||scoreAll()){
   if(wet&&(crossing||(policy.shaping.x&256u)!=0u)&&(scored(STRAIN)||scored(ROTATION))){
    let activity=umFlowActivity(p,tile);
    if(scored(STRAIN)&&policy.importance.y>0.0){
     let score=activity.x/policy.importance.y;w0|=scoreByte(score)<<16u;
     if(score>=1.0&&criterionOn(STRAIN)){w1|=triggerBit(STRAIN);}
    }
    if(scored(ROTATION)&&policy.importance.z>0.0){
     let score=activity.y/policy.importance.z;w0|=scoreByte(score)<<24u;
     if(score>=1.0&&criterionOn(ROTATION)){w1|=triggerBit(ROTATION);}
    }
   }
   if(crossing&&scored(APPROACH)){
    let score=umApproach(p,tile);w1|=scoreByte(score)<<8u;
    if(score>=1.0&&criterionOn(APPROACH)){w1|=triggerBit(APPROACH);}
   }
  }
  // top: the highest score of a criterion that is on (the budget's rank and
  // the hold's keep); the winner is the highest of any, for the layer.
  var top=0u;var best=0u;var winner=0u;
  for(var k=0u;k<6u;k++){
   let byte=umScoreByte(w0,w1,k);
   if(byte>best){best=byte;winner=k+1u;}
   if(criterionOn(k)){top=max(top,byte);}
  }
  w1|=winner<<${IMPORTANCE.winnerShift}u;
  let triggered=((w1>>${IMPORTANCE.triggeredShift}u)&63u)!=0u;
  var kept=triggered;
  if(triggered&&budgetOn()){
   atomicAdd(&census[binIndex(top>>2u)],1u);
   if(top<atomicLoad(&census[cutoffIndex()])){kept=false;w1|=${IMPORTANCE.dropped}u;}
  }
  var held=false;
  if(holding){
   remaining=atomicLoad(&census[holdIndex(tile)]);
   if(kept){remaining=policy.shaping.y;}
   else if(remaining>0u){
    // A dropped tile's hold runs down whatever it scores: the budget wins.
    held=true;if(triggered||top<policy.shaping.w){remaining--;}
   }
  }
  required=kept||held;
  w1|=select(0u,${IMPORTANCE.required}u,required)|select(0u,${IMPORTANCE.held}u,held);
  // Bulk liquid nothing scores keeps classify's words.
  if(w0!=own.x){atomicStore(&census[importanceIndex(tile,0u)],w0);}
  if(w1!=own.y){atomicStore(&census[importanceIndex(tile,1u)],w1);}
 }
 if(holding){atomicStore(&census[holdIndex(tile)],remaining);}
 // Required tiles seed decide: their prefix flag, crossing cells (the whole
 // tile without a surface of its own), gap and, for a boundary tile, its
 // directed travel. classify stored a crossing tile's as if required.
 bounds[prefixIndex(p+vec3u(1u))]=select(0u,1u,required);
 let impacted=(w1&triggerBit(IMPACT))!=0u;
 let bounded=required&&!shaped&&impacted;
 if(impacted&&!bounded){atomicStore(&census[travelIndex(tile)],0xffffffffu);}
 if(!required){
  if(crossing){for(var k=0u;k<2u;k++){atomicStore(&census[crossingIndex(tile,k)],0u);}atomicStore(&census[gapIndex(tile)],0xffffffffu);}
  return;
 }
 atomicAdd(&census[12],1u);
 let speed=umTileSpeed(tile);
 if(!crossing){for(var k=0u;k<2u;k++){atomicStore(&census[crossingIndex(tile,k)],0xffffffffu);}atomicStore(&census[gapIndex(tile)],min(u32(ceil(speed)),255u)<<24u);}
 if(bounded){atomicAdd(&census[16],1u);atomicMax(&census[17],u32(ceil(speed)));}
 // A 4h tile the criteria require: refined by this census.
 if(umTileWidth(tile)!=1u){let slot=atomicAdd(&census[1],1u);if(slot<3u){atomicStore(&census[13u+slot],tile);}}
}
// The budget, after the band is final: the seeds allowed are the budget's
// share of the band that this census's seeds were of it (closure, travel and
// promotion ride on the seeds). The cutoff keeps whole score bins from the
// top while they fit, and always the saturated one; the next census applies it.
@compute @workgroup_size(1) fn budget(){
 let band=max(atomicLoad(&census[2]),1u);let seeds=min(atomicLoad(&census[12]),band);
 let allowed=u32(f32(policy.shaping.z)*f32(seeds)/f32(band));
 var sum=0u;var keep=${IMPORTANCE_BINS-1}u;var fits=true;
 for(var i=0u;i<${IMPORTANCE_BINS}u;i++){
  let bin=${IMPORTANCE_BINS-1}u-i;sum+=atomicExchange(&census[binIndex(bin)],0u);
  fits=fits&&(i==0u||sum<=allowed);
  if(fits){keep=bin;}
 }
 atomicStore(&census[cutoffIndex()],keep*${256/IMPORTANCE_BINS}u);
}
// Separable inclusive prefix sum: one lane per line of the (T+1)³ table.
${[0,1,2].map(axis=>{const [a,b]=[0,1,2].filter(k=>k!==axis);return /* wgsl */`
@compute @workgroup_size(64) fn prefix${axis}(@builtin(global_invocation_id) gid:vec3u){
 let line=gid.x+umDispatchX*64u*gid.y;let extent=vec3u(PX,PY,PZ);
 if(line>=extent[${a}]*extent[${b}]){return;}
 var p=vec3u(0u);p[${a}]=line%extent[${a}];p[${b}]=line/extent[${a}];var sum=0u;
 // A line on a border plane is all zeros (prefixLoad).
 if(p[${a}]==0u||p[${b}]==0u){return;}
 for(var i=1u;i<extent[${axis}];i++){p[${axis}]=i;let i=prefixIndex(p);sum+=bounds[i];bounds[i]=sum;}
}`;}).join("\n")}
// Cube levels 1..topCube() are built: no box query needs a larger side.
override cubeLevel:u32=1u;
// The largest box side, in tiles, a departure query can take: the tile, the
// sampling reach and drift margin on each side, the travel of the fastest
// tile over the horizon (census[18]) and gravity's.
// Largest distance, in cells, any departure box edge travels over the
// horizon: the fastest tile's (census[18]) plus gravity's, per axis.
fn departureTravel()->f32{
 let steps=max(1.0,policy.surface.z);
 let gravity=0.5*steps*(steps-1.0)*length(policy.flow.xyz)*policy.step.x/(steps*UM_H);
 return min(bitcast<f32>(atomicLoad(&census[18]))+gravity,1.0e6);
}
// Tiles from p within which decide's closed, margined departure box of any
// tile p lies (either width's margin): no interface tile there, no h.
fn departureReachTiles()->i32{
 let margin=policy.flow.w+4.0*f32(policy.reach.x+policy.reach.y);
 return i32(ceil((departureTravel()+margin+1e-3)/4.0))+2;
}
// Tiles from p within which every cube or tile its trace's velocity queries
// read starts (the unmargined boxes, grown by SAMPLE_REACH).
fn sampleReachTiles()->i32{return i32(ceil((departureTravel()+SAMPLE_REACH+1e-3)/4.0))+2;}
fn topCube()->u32{
 let travel=min(departureTravel()+SAMPLE_REACH+policy.flow.w,1.0e6);
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
 // Only decide's queries read cubes, only for tiles within departureReach of
 // an interface tile, and only at starts within sampleReach of the querying
 // tile, at levels up to topCube(). A cube at s reads level cubeLevel-1 at
 // s + [0, half]: the radius grows by half per level down from the top one,
 // so every cube a built one reads is built.
 let radius=departureReachTiles()+sampleReachTiles()+i32((1u<<topCube())-(1u<<cubeLevel));
 if(interfaceTilesIn(s-vec3i(radius),s+vec3i(radius))==0u){return;}
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
 let s=prefixLoad(b)-prefixLoad(vec3u(a.x,b.y,b.z))-prefixLoad(vec3u(b.x,a.y,b.z))-prefixLoad(vec3u(b.x,b.y,a.z));
 return s+prefixLoad(vec3u(a.x,a.y,b.z))+prefixLoad(vec3u(a.x,b.y,a.z))+prefixLoad(vec3u(b.x,a.y,a.z))-prefixLoad(a);
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
// Intersect the closed box with the actual crossing cells, not their six
// independent extrema (which can come from different owners). Closed bounds
// retain shared faces, edges and corners, including the existing drift margin.
fn crossingCellsMeet(q:vec3i,lo:vec3f,hi:vec3f)->bool{
 let origin=4.0*vec3f(q);
 let a=max(vec3i(floor(lo-origin-vec3f(1e-3))),vec3i(0));
 let b=min(vec3i(floor(hi-origin+vec3f(1e-3))),vec3i(3));
 if(any(b<a)){return false;}
 let tile=umTileAt(vec3u(q));
 let mask=vec2u(atomicLoad(&census[crossingIndex(tile,0u)]),atomicLoad(&census[crossingIndex(tile,1u)]));
 if(all(mask==vec2u(0u))){return false;}
 let row=((1u<<u32(b.x-a.x+1))-1u)<<u32(a.x);
 for(var z=a.z;z<=b.z;z++){for(var y=a.y;y<=b.y;y++){
  let lane=4u*u32(y)+16u*u32(z);
  if((mask[lane/32u]&(row<<(lane%32u)))!=0u){return true;}
 }}
 return false;
}
// Whole tiles inside the box use the prefix sum; only its shell needs the
// crossing-cell mask. This preserves the cheap rejection of distant tiles.
fn departureMeetsSurface(lo:vec3f,hi:vec3f)->bool{
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
   if(crossingCellsMeet(vec3i(x,y,z),lo,hi)){return true;}
  }
 }}
 return false;
}
// A joined tile (joinTarget, join) is band whatever the trace finds; so is
// a tile this frame's drop or inflow plug can fill: the source phi (a lower
// bound on the distance to its ball or plug) at the tile centre within half
// the tile diagonal, plus two cells for the aperture edge and the donor
// layer upstream of the nozzle face. Liquid-conditional solid promotion then
// treats either as band.
fn umJoined(t:u32)->bool{return (((joinTiles[t/32u]|joinTiles[WORDS+t/32u])>>(t%32u))&1u)!=0u;}
fn umSourceTile(p:vec3i)->bool{
 if((policy.shaping.x&512u)==0u){return false;}
 if(umSourceParams.drop.w<=0.0&&umSourceinflowStrength()<=0.0){return false;}
 let centre=4.0*vec3f(p)+vec3f(2.0);
 return umSourceuvSourcePhi(centre,3.0e38)<=0.5*length(4.0*H)+2.0*MAX_H;
}
// Chebyshev tile distance: each layer includes face, edge and corner neighbours.
// Read actual crossings, not the required-seed prefix (criteria may omit seeds).
fn surfaceAllowed(tile:u32,p:vec3i)->bool{
 if((atomicLoad(&census[importanceIndex(tile,1u)])&${IMPORTANCE.crossing}u)!=0u){return true;}
 let radius=i32((policy.shaping.x>>13u)&3u);if(radius==0){return false;}
 let lo=max(p-vec3i(radius),vec3i(0));let hi=min(p+vec3i(radius),vec3i(UM_T)-vec3i(1));
 for(var z=lo.z;z<=hi.z;z++){for(var y=lo.y;y<=hi.y;y++){for(var x=lo.x;x<=hi.x;x++){
  let t=umTileAt(vec3u(vec3i(x,y,z)));
  if((atomicLoad(&census[importanceIndex(t,1u)])&${IMPORTANCE.crossing}u)!=0u){return true;}
 }}}
 return false;
}
@compute @workgroup_size(64) fn decide(@builtin(global_invocation_id) gid:vec3u){
 let tile=gid.x+umDispatchX*64u*gid.y;if(tile>=UM_TILES){return;}
 let p=vec3i(umTileCoord(tile));let scale=policy.step.x/H;
 let hostJoined=umJoined(tile);let source=!hostJoined&&umSourceTile(p);
 // Intersect the existing selection, including hold, closure and travel,
 // with the allowed distance from actual geometric crossings.
 // Explicit joins, new liquid sources and mandatory solid promotion retain
 // priority. A source has no existing surface crossing yet; excluding it here
 // also excludes its pages from the residency certificate below.
 // Keep the census and residency closure intact: this changes ownership only.
 if((policy.shaping.x&4096u)!=0u&&!hostJoined&&!source&&!surfaceAllowed(tile,p)){
  if(umTileWidth(tile)==1u){atomicAdd(&census[5],1u);}recordReason(tile,${REASON.skipped}u);return;
 }
 let joined=hostJoined||source;
 let joinReason=select(${REASON.source}u,${REASON.join}u,hostJoined);
 if(policy.surface.x>0.0){
  let width=umTileWidth(tile);
  let fine=joined||forwardFine(p,i32(policy.reach.x)+select(0,i32(policy.reach.y),width==1u));
  if(!fine){if(width==1u){atomicAdd(&census[5],1u);}recordReason(tile,${REASON.skipped}u);return;}
  atomicAdd(&census[2],1u);if(width!=1u){atomicAdd(&census[4],1u);}
  atomicOr(&census[${HEADER}u+tile/32u],1u<<(tile%32u));
  recordReason(tile,select(${REASON.travel}u,joinReason,joined));
  return;
 }
 // A current interface tile stays fine: phi does not only move with the
 // flow (residual sheets left behind a falling surface persist in place).
 let width=umTileWidth(tile);
 // No interface tile within the farthest any departure box can reach: the
 // trace below cannot meet one (an O(1) prefix query).
 let far=departureReachTiles();
 if(!joined&&(unseeded()||interfaceTilesIn(p-vec3i(far),p+vec3i(far))==0u)){if(width==1u){atomicAdd(&census[5],1u);}recordReason(tile,${REASON.skipped}u);return;}
 let crossing=!unseeded()&&interfaceTilesIn(p,p)!=0u;
 var fine=joined||crossing;
 var reason=select(${REASON.crossing}u,joinReason,joined);
 // Views only: a required tile with no surface of its own.
 if(crossing&&!joined&&recordReasons()&&(atomicLoad(&census[importanceIndex(tile,1u)])&${IMPORTANCE.crossing}u)==0u){reason=${REASON.bulk}u;}
 if(!fine){
  // Closed: a boundary vertex belongs to both tiles it separates.
  let margin=policy.flow.w+4.0*f32(i32(policy.reach.x)+select(0,i32(policy.reach.y),width==1u));
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
  fine=departureMeetsSurface(lo-vec3f(margin),hi+vec3f(margin));
  // Views only: closure is the part a still surface (dt = 0) already needs.
  if(fine&&recordReasons()){
   let box=4.0*vec3f(p);
   reason=select(${REASON.travel}u,${REASON.closure}u,departureMeetsSurface(box-vec3f(margin),box+vec3f(4.0+margin)));
  }
 }
 if(!fine){if(width==1u){atomicAdd(&census[5],1u);}recordReason(tile,${REASON.traced}u);return;}
 atomicAdd(&census[2],1u);if(width!=1u){atomicAdd(&census[4],1u);}
 atomicOr(&census[${HEADER}u+tile/32u],1u<<(tile%32u));
 recordReason(tile,reason);
}
// Liquid-conditional solid promotion. Fine-owner solid terms need a cut
// tile, and each neighbour of it, at h wherever liquid can meet it; a dry cut
// tile far from liquid runs 4h without them. Liquid reaches a coupled tile
// this frame only if it is wet or band: the band is already the surface this
// census predicts over its horizon, closed and margined, so no further
// dilation is needed. That tile is active, and it and its 26 neighbours join
// the band. The band certificate fails a liquid row in a cut tile the
// simulation holds at 4h.
fn umBit(base:u32,t:u32)->bool{return (atomicLoad(&census[base+t/32u])&(1u<<(t%32u)))!=0u;}
@compute @workgroup_size(64) fn solidActive(@builtin(global_invocation_id) gid:vec3u){
 let tile=gid.x+umDispatchX*64u*gid.y;if(tile>=UM_TILES){return;}
 if(((solidTiles[tile/32u]>>(tile%32u))&1u)==0u){return;}
 if(umBit(wetIndex(0u),tile)||umBit(${HEADER}u,tile)){atomicOr(&census[activeIndex(tile/32u)],1u<<(tile%32u));}
}
@compute @workgroup_size(64) fn solidPromote(@builtin(global_invocation_id) gid:vec3u){
 let tile=gid.x+umDispatchX*64u*gid.y;if(tile>=UM_TILES){return;}
 if(umBit(${HEADER}u,tile)){return;}
 let p=vec3i(umTileCoord(tile));
 let a=max(p-vec3i(1),vec3i(0));let b=min(p+vec3i(1),vec3i(UM_T)-vec3i(1));
 for(var z=a.z;z<=b.z;z++){for(var y=a.y;y<=b.y;y++){for(var x=a.x;x<=b.x;x++){
  if(!umBit(activeIndex(0u),umTileAt(vec3u(vec3i(x,y,z))))){continue;}
  atomicOr(&census[${HEADER}u+tile/32u],1u<<(tile%32u));
  atomicAdd(&census[2],1u);atomicAdd(&census[19],1u);recordReason(tile,${REASON.solid}u);
  if(umTileWidth(tile)==1u){atomicSub(&census[5],1u);}else{atomicAdd(&census[4],1u);}
  return;
 }}}
}
// Residency certificate (uniformMixedResidencyWord), after the band is
// final. pageSeed: per page, the box of its near tiles, the band's new h
// tiles and the builder's static h tiles (local tile coordinates, two bits
// each; bit 12 set when non-empty). The static tiles are h after this build
// without the band: a Fine region drawn in far air would otherwise be h in a
// page this certificate leaves absent, which the next census audits as fatal.
fn umStaticFine(t:u32)->bool{return ((joinTiles[2u*WORDS+t/32u]>>(t%32u))&1u)!=0u;}
@compute @workgroup_size(64) fn pageSeed(@builtin(global_invocation_id) gid:vec3u){
 let page=gid.x+umDispatchX*64u*gid.y;
 // The in-frame readers' sticky closure bits join this census's audit.
 if(page==0u&&umSupport[UM_RESIDENCY+1u]!=0u){atomicOr(&census[auditIndex()],${UNIFORM_MIXED_RESIDENCY_AUDIT.closure}u);}
 if(page>=UM_PAGES){return;}
 var lo=vec3u(3u);var hi=vec3u(0u);var found=false;
 for(var lane=0u;lane<64u;lane++){
  let t=umPageTile(page,lane);if(t>=UM_TILES){continue;}
  if(!umBit(nearIndex(0u),t)&&!umBit(${HEADER}u,t)&&!umStaticFine(t)){continue;}
  let c=umCorner(lane,4u);lo=min(lo,c);hi=max(hi,c);found=true;
 }
 atomicStore(&census[pageBoxIndex(page)],select(0u,lo.x|(lo.y<<2u)|(lo.z<<4u)|(hi.x<<6u)|(hi.y<<8u)|(hi.z<<10u)|(1u<<12u),found));
}
// Closure radius in tiles: the travel over the horizon plus the margin.
fn residencyRadius()->u32{return u32(ceil(departureTravel()/4.0))+${RESIDENCY_MARGIN_TILES}u;}
// pageMark: a page is resident when a seeded tile lies within the radius
// (Chebyshev, in tiles) of any of its tiles.
@compute @workgroup_size(64) fn pageMark(@builtin(global_invocation_id) gid:vec3u){
 let page=gid.x+umDispatchX*64u*gid.y;if(page>=UM_PAGES){return;}
 let r=i32(residencyRadius());let reach=(r+3)/4;
 var resident=reach>${RESIDENCY_PAGE_REACH};
 let q=vec3i(umPageCoord(page));let qa=4*q;let qb=qa+vec3i(3);
 if(!resident){
  let a=max(q-vec3i(reach),vec3i(0));let b=min(q+vec3i(reach),vec3i(UM_PD)-vec3i(1));
  for(var z=a.z;z<=b.z&&!resident;z++){for(var y=a.y;y<=b.y&&!resident;y++){for(var x=a.x;x<=b.x;x++){
   let s=vec3i(x,y,z);let box=atomicLoad(&census[pageBoxIndex(u32(s.x)+UM_PD.x*(u32(s.y)+UM_PD.y*u32(s.z)))]);
   if(box==0u){continue;}
   let lo=4*s+vec3i(i32(box&3u),i32((box>>2u)&3u),i32((box>>4u)&3u));
   let hi=4*s+vec3i(i32((box>>6u)&3u),i32((box>>8u)&3u),i32((box>>10u)&3u));
   let gap=max(max(lo-qb,qa-hi),vec3i(0));
   if(max(gap.x,max(gap.y,gap.z))<=r){resident=true;break;}
  }}}
 }
 umSupport[UM_RESIDENCY+4u+page]=select(0u,1u,resident);
}
// pageCompact: one workgroup lists the resident pages in ascending order.
const COMPACT_LANES:u32=256u;
var<workgroup> compactCounts:array<u32,256>;
@compute @workgroup_size(256) fn pageCompact(@builtin(local_invocation_index) lane:u32){
 let per=(UM_PAGES+COMPACT_LANES-1u)/COMPACT_LANES;let first=lane*per;let last=min(first+per,UM_PAGES);
 var count=0u;
 for(var p=first;p<last;p++){if(umPageResident(p)){count++;}}
 compactCounts[lane]=count;workgroupBarrier();
 // Inclusive Hillis-Steele scan over the lanes.
 for(var stride=1u;stride<COMPACT_LANES;stride*=2u){
  var add=0u;if(lane>=stride){add=compactCounts[lane-stride];}
  workgroupBarrier();compactCounts[lane]+=add;workgroupBarrier();
 }
 var at=compactCounts[lane]-count;
 for(var p=first;p<last;p++){if(umPageResident(p)){umSupport[UM_RESIDENCY+4u+UM_PAGES+at]=p;at++;}}
 if(lane==COMPACT_LANES-1u){umSupport[UM_RESIDENCY]=compactCounts[lane];umSupport[UM_RESIDENCY+2u]=residencyRadius();}
}
`});
  const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");
  if(errors.length)throw new Error(errors.map(m=>`${m.lineNum}: ${m.message}`).join("\n"));
  const layout=this.device.createPipelineLayout({bindGroupLayouts:[this.ownership.bindLayout,this.resources]});
  await Promise.all(["classify","classifyCoarse","importance","prefix0","prefix1","prefix2","decide","solidActive","solidPromote","budget","pageSeed","pageMark","pageCompact"].map(async entryPoint=>{this.pipelines.set(entryPoint,await uniformDetailPipeline(this.device,this.ownership,{layout,compute:{module,entryPoint,constants:{umDispatchX:this.ownership.dispatchX}}}));}));
  await Promise.all(Array.from({length:CUBE_LEVELS},async(_,i)=>{const level=i+1;
   this.pipelines.set(`cube${level}`,await uniformDetailPipeline(this.device,this.ownership,{layout,compute:{module,entryPoint:"boundCube",constants:{umDispatchX:this.ownership.dispatchX,cubeLevel:level}}}));}));
 }
 /** Solid-coupled tiles (uniformMixedSolidTiles().coupled), for the impact rule. */
 setSolid(coupled:Uint8Array):void{
  const tiles=this.ownership.capacity.tiles;
  if(coupled.length!==tiles)throw new Error(`Dynamic ownership solid mask has ${coupled.length} tiles, expected ${tiles}`);
  const bits=new Uint32Array(this.words);
  for(let t=0;t<tiles;t++)if(coupled[t])bits[t>>5]!|=1<<(t&31);
  this.device.queue.writeBuffer(this.solidTiles,0,bits);this.device.queue.writeBuffer(this.staticSolidTiles,0,bits);
  this.solid=coupled.some(c=>c!==0);this.bodyBits=false;
 }
 /** The builder's static h tiles (UniformMixedLayoutBuilder.setStatic's fine
  * mask), set with it: every census seeds their pages resident. */
 setStatic(fine:Uint8Array):void{
  const tiles=this.ownership.capacity.tiles;
  if(fine.length!==tiles)throw new Error(`Dynamic ownership static mask has ${fine.length} tiles, expected ${tiles}`);
  const bits=new Uint32Array(this.words);
  for(let t=0;t<tiles;t++)if(fine[t])bits[t>>5]!|=1<<(t&31);
  this.device.queue.writeBuffer(this.joinTiles,2*this.words*4,bits);this.staticBits=bits;
 }
 private staticBits?:Uint32Array<ArrayBuffer>;
 /** setStatic for a mask that differs from the one held at `tiles` at most
  * (UniformMixedLayoutBuilder.editStatic, set with it): the words holding
  * them are written, in runs, so the census reads what setStatic(fine) writes. */
 editStatic(tiles:readonly number[],fine:Uint8Array):void{
  const bits=this.staticBits;
  if(!bits||fine.length!==this.ownership.capacity.tiles)throw new Error("A static mask edit needs a static mask of the tile lattice");
  const dirty:number[]=[];
  for(const t of tiles){const w=t>>5,bit=1<<(t&31),before=bits[w]!,after=(fine[t]?before|bit:before&~bit)>>>0;if(after!==before){bits[w]=after;dirty.push(w);}}
  dirty.sort((a,b)=>a-b);
  for(let i=0;i<dirty.length;){
   let j=i;while(j+1<dirty.length&&dirty[j+1]!-dirty[j]!<=64)j++;
   this.device.queue.writeBuffer(this.joinTiles,4*(2*this.words+dirty[i]!),bits,dirty[i]!,dirty[j]!-dirty[i]!+1);i=j+1;
  }
 }
 /** Moving bodies: before each census, `mark` ors the tiles they may cut
  * over the next frame into the coupled mask (on the GPU, from the body
  * state), and the liquid-conditional promotion passes run. undefined
  * restores the static mask. */
 setBodies(mark?:(encoder:GPUCommandEncoder,tiles:GPUBuffer)=>void):void{this.bodies=mark;}
 /** Host join: `tiles` (one byte per tile) are band in the next census,
  * ored with any earlier join it has not consumed. Uploaded by encode. */
 join(tiles:Uint8Array):void{
  const n=this.ownership.capacity.tiles;
  if(tiles.length!==n)throw new Error(`Dynamic ownership join has ${tiles.length} tiles, expected ${n}`);
  const bits=this.hostJoin??=new Uint32Array(this.words);
  for(let t=0;t<n;t++)if(tiles[t])bits[t>>5]!|=1<<(t&31);
 }
 /** The GPU join words: a producer ors one bit per tile (atomicOr) in work
  * submitted before the next census, or encoded before it in its encoder;
  * that census consumes and clears them. */
 joinTarget():UniformMixedBandBits{this.joinLive=true;return {buffer:this.joinTiles,wordOffset:0};}
 /** Encode after a completed frame, while its ownership and the local speed
  * velocity (the one the next frame advects with) are still in place. Every
  * launch is a fixed grid bounded on the GPU (umCounts): no host tile count.
  * readback: copy the header and band bits for read() (tools, tests and the
  * host relayout path); without it the census reads nothing back. */
 encode(encoder:GPUCommandEncoder,policy:UniformMixedDynamicPolicy,readback=true):void{
  if(this.pipelines.size!==13+CUBE_LEVELS)throw new Error("Dynamic ownership census is not initialized");
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
  this.device.queue.writeBuffer(this.params,32,new Float32Array([policy.fastTravel,policy.boundaryTravel,policy.steps,policy.reasons?1:0,...policy.gravity.map(g=>g*frame),UNIFORM_MIXED_DYNAMIC_SURFACE_DRIFT]));
  const importance=policy.importance??LEGACY_IMPORTANCE;
  for(const [name,value] of Object.entries({thinThickness:importance.thinThickness,strainThreshold:importance.strainThreshold,rotationThreshold:importance.rotationThreshold,approachSteps:importance.approachSteps,retireRatio:importance.retireRatio}))
   if(!Number.isFinite(value)||value<0)throw new Error(`Dynamic ownership importance ${name} must be finite and non-negative: ${value}`);
  // Thin reads vertices one tile around: phi is a distance there only so far.
  if(importance.thinThickness>8)throw new Error(`Dynamic ownership importance thinThickness must be at most 8 h: ${importance.thinThickness}`);
  if(!Number.isSafeInteger(importance.holdSteps)||importance.holdSteps<0)throw new Error(`Dynamic ownership importance holdSteps must be a non-negative integer: ${importance.holdSteps}`);
  const surfaceDistance=importance.surfaceDistance??0;
  if(!Number.isSafeInteger(surfaceDistance)||surfaceDistance<0||surfaceDistance>3)throw new Error(`Dynamic ownership surfaceDistance must be an integer in 0..3: ${surfaceDistance}`);
  const budget=importance.budgetTiles;
  const shapeMetric=importance.shapeMetric??"value";
  if(shapeMetric!=="value"&&shapeMetric!=="displacement")throw new Error(`Dynamic ownership importance shapeMetric must be "value" or "displacement": ${String(shapeMetric)}`);
  if(budget!==undefined&&(!Number.isSafeInteger(budget)||budget<0))throw new Error(`Dynamic ownership importance budgetTiles must be a non-negative integer: ${budget}`);
  this.device.queue.writeBuffer(this.params,64,new Float32Array([importance.thinThickness,importance.strainThreshold,importance.rotationThreshold,importance.approachSteps]));
  // A census of requests only (no criterion, hold or layer; joins, sources,
  // solids and bodies remain) requires no tile: decide then reads no table.
  const seeded=!!policy.reasons||policy.fastTravel>0||importance.holdSteps>0||UNIFORM_DETAIL_CRITERIA.some(name=>importance.criteria[name]);
  this.device.queue.writeBuffer(this.params,80,new Uint32Array([
   UNIFORM_DETAIL_CRITERIA.reduce((bits,name,k)=>bits|(importance.criteria[name]?1<<k:0),(importance.bulk?256:0)|(importance.sources===false?0:512)|(seeded?0:1024)|(shapeMetric==="displacement"?2048:0)|(importance.surfaceOnly?4096:0)|(surfaceDistance<<13)|(importance.surfaceAuthority?32768:0)),
   importance.holdSteps,budget??0xffffffff,Math.min(255,Math.floor(importance.retireRatio*IMPORTANCE.scoreOne))]));
  // Cube levels are written wherever decide can read them before it does,
  // classify writes every tile's keys, gap, travel and importance words and
  // importance its prefix entry; only the counters, band bits and wet/active
  // bits accumulate. The holds and the budget's cutoff persist by design.
  encoder.clearBuffer(this.work,0,(HEADER+this.words)*4);
  encoder.clearBuffer(this.work,this.wetOffset*4,2*this.words*4);
  encoder.clearBuffer(this.work,this.nearOffset*4,(this.words+1)*4);
  if(this.bodies||this.bodyBits){
   encoder.copyBufferToBuffer(this.staticSolidTiles,0,this.solidTiles,0,this.words*4);
   this.bodies?.(encoder,this.solidTiles);this.bodyBits=!!this.bodies;
  }
  // Host joins upload now, ordered after every earlier submission (and
  // after the census that consumed the last ones); GPU producers or theirs.
  if(this.hostJoin){this.device.queue.writeBuffer(this.joinTiles,this.words*4,this.hostJoin);this.hostJoin=undefined;this.joinLive=true;}
  const tiles=this.ownership.capacity.tiles,x=this.ownership.dispatchX,t=this.ownership.capacity.lattice.dimensions.map(n=>n/4+1);
  // One pass: every dispatch is its own usage scope, so each reads what
  // the earlier ones wrote.
  const pass=encoder.beginComputePass({label:"Uniform dynamic ownership census"});
  pass.setBindGroup(0,this.ownership.bindGroup);pass.setBindGroup(1,this.group.group);
  for(const entry of ["classify","classifyCoarse",...(seeded?["importance","prefix0","prefix1","prefix2",...Array.from({length:CUBE_LEVELS},(_,j)=>`cube${j+1}`)]:[]),...(this.solid||this.bodies?["decide","solidActive","solidPromote"]:["decide"]),...(budget===undefined?[]:["budget"]),"pageSeed","pageMark","pageCompact"]){
   // Fixed grids: classify strides over the h list; classifyCoarse has a
   // lane per tile, bounded by the 4h count; the rest are lattice-sized.
   const lines=entry.startsWith("prefix")?[0,1,2].filter(k=>k!==Number(entry.at(-1))).reduce((n,k)=>n*t[k]!,1):entry.startsWith("page")?this.pages:tiles;
   pass.setPipeline(uniformDetailPick(this.pipelines.get(entry)!));
   if(entry==="classify")pass.dispatchWorkgroups(Math.min(CENSUS_TILE_GRID,tiles));
   else if(entry==="pageCompact"||entry==="budget")pass.dispatchWorkgroups(1);
   else{const groups=Math.ceil(lines/64);pass.dispatchWorkgroups(Math.min(groups,x),Math.ceil(groups/x));}
  }
  pass.end();
  // The band now holds the joins: clear them for the next producers.
  if(this.joinLive)encoder.clearBuffer(this.joinTiles,0,2*this.words*4);
  if(readback){encoder.copyBufferToBuffer(this.work,0,this.readback,0,(HEADER+this.words)*4);this.encoded=true;}
 }
 /** Map the census encoded by the last submitted encode(readback). An
  * opt-in diagnostic: nothing on the advance path may wait for it. */
 async read():Promise<UniformMixedDynamicCensus>{
  if(!this.encoded)throw new Error("Dynamic ownership census was not encoded with a readback");
  this.encoded=false;
  await this.readback.mapAsync(GPUMapMode.READ);
  const words=new Uint32Array(this.readback.getMappedRange()).slice();this.readback.unmap();
  const f=new Float32Array(words.buffer);
  const tiles=this.ownership.capacity.tiles;let fine:Uint8Array|undefined;
  // Expanded from the band bits on first use (the adopt path needs none).
  const expand=()=>{const bytes=new Uint8Array(tiles);for(let t=0;t<tiles;t++)bytes[t]=(words[HEADER+(t>>5)]!>>>(t&31))&1;return bytes;};
  return {get fine(){return fine??=expand();},interfaceTiles:words[0]!,requiredTiles:words[12]!,unresolvedCoarse:words[1]!,unresolvedTiles:Array.from(words.subarray(13,13+Math.min(3,words[1]!))),boundaryTiles:words[16]!,fineTiles:words[2]!,refined:words[4]!,coarsened:words[5]!,solidTiles:words[19]!,
  coarsePartialVolume:words[6]!,coarsePhiCrossing:words[7]!,coarseDryLiquidPhi:words[3]!,
  interiorDeficit:[f[8]!,f[9]!],airVolume:[f[10]!,f[11]!]};
 }
 destroy():void{this.solidTiles.destroy();this.staticSolidTiles.destroy();this.joinTiles.destroy();this.noSource?.destroy();this.noHeld?.destroy();this.work.destroy();this.bounds.destroy();this.readback.destroy();this.params.destroy();}
}
