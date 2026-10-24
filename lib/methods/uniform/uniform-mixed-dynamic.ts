import {uniformSolidRefinementWGSL} from "./uniform-solid-refinement";
import { STAGE_RESIDENCY_WORDS, RESIDENCY_STAGES, RESIDENCY_STENCIL_TILES, localResidencyWGSL, localResidencyRequestWGSL, localResidencyMarkWGSL } from "./uniform-local-residency";
import { narrowBandBudgetWGSL, narrowBandBudgetWords, NARROW_BAND_BUDGET_ENTRIES } from "./uniform-narrow-band-budget.wgsl";
import { uniformDetailBindLayout, uniformDetailModule, uniformDetailPipeline, uniformDetailPick, uniformDetailGroup, type UniformDetailGroup } from "./uniform-detail-fields";
import {uniformKeyGroups} from "./uniform-lattice-address";
import type {UniformMixedSolidCoarse} from "./uniform-mixed-solid.wgsl";
import type {UniformMixedOwnership} from "./uniform-mixed-ownership";
import {UNIFORM_MIXED_REGION_LIMIT,UNIFORM_MIXED_REGION_TABLE_WORDS,uniformMixedRegionWGSL,type UniformMixedBandBits} from "./uniform-mixed-layout-builder";
import {UNIFORM_BOUND_CUBE_WORDS,UNIFORM_TILE_BITS_WORDS,uniformBoundCubesWGSL,uniformTileBitsWGSL} from "./uniform-block-tables";
import {uniformMixedPageCount,uniformMixedTopologyNeighboursWGSL,uniformMixedTopologyWGSL} from "./uniform-mixed-topology.wgsl";
import {UNIFORM_PARAMS_BYTES,uniformMixedSourceWGSL} from "./uniform-mixed-source.wgsl";
import {UNIFORM_REGION_HEAT_DEEP_WORD,UNIFORM_REGION_TIER} from "./uniform-region-heat";
import {UNIFORM_RESIDENCY_PAGE_REACH,UNIFORM_RESIDENCY_LIQUID as PAGE_LIQUID,UNIFORM_RESIDENCY_SEEDED as PAGE_SEEDED} from "./uniform-mixed-residency";
import {uniformPagePoolWGSL,type UniformPagePool} from "./uniform-page-pool";
import {uniformPageInitBlocksWGSL,uniformPageInitGroups,type UniformPageInitBlocks} from "./uniform-page-initialise";
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
/** A tile of the 512-tile window as one word: x | y<<9 | z<<18. */
export function uniformMixedTileWord(x:number,y:number,z:number):number{
 if(![x,y,z].every(v=>Number.isInteger(v)&&v>=0&&v<512))throw new RangeError(`Tile ${x},${y},${z} is not a tile of the 512-tile window`);
 return x|(y<<9)|(z<<18);
}
/** Words of a host tile plane of `pages` pages: word 0 the count of pages
 * holding a set tile (0: the plane answers nothing), a 16,384-word directory
 * over the 64³ page window (16 pages a word: the low half a presence bit a
 * page, the high half the record of the word's first present page), then a
 * 512-bit record a page. */
const hostPlaneWords=(pages:number)=>1+16384+16*pages;
/** The plane of `tiles` (uniformMixedTileWord each, any order, repeats
 * allowed), in `pages` records at most. */
function hostTilePlane(tiles:ArrayLike<number>,pages:number,what:string):Uint32Array<ArrayBuffer>{
 const plane=new Uint32Array(hostPlaneWords(pages)),present=new Set<number>();
 const pageOf=(w:number)=>((w>>3)&63)|(((w>>12)&63)<<6)|(((w>>21)&63)<<12);
 for(let i=0;i<tiles.length;i++)present.add(pageOf(tiles[i]!));
 if(present.size>pages)throw new RangeError(`Dynamic ownership ${what} lie in ${present.size} pages, above the ${pages} the census was built for`);
 const order=[...present].sort((a,b)=>a-b),record=new Map<number,number>();
 order.forEach((e,r)=>{record.set(e,r);const d=1+(e>>4);if((plane[d]!&0xffff)===0)plane[d]=(r<<16)>>>0;plane[d]=(plane[d]!|(1<<(e&15)))>>>0;});
 plane[0]=order.length;
 for(let i=0;i<tiles.length;i++){
  const w=tiles[i]!,l=(w&7)|(((w>>9)&7)<<3)|(((w>>18)&7)<<6);
  const at=1+16384+16*record.get(pageOf(w))!+(l>>5);plane[at]=(plane[at]!|(1<<(l&31)))>>>0;
 }
 return plane;
}
/** WGSL: fn <name>(c:vec3i)->bool, tile c's bit in the host tile plane at
 * word `offset` of `buffer`; a tile outside the window has none. */
const hostPlaneWGSL=(name:string,buffer:string,offset:string)=>/* wgsl */`fn ${name}(local:vec3i)->bool{
 let c=local+8*umFramePage();
 if(any(c<vec3i(0))||any(c>vec3i(511))||${buffer}[${offset}]==0u){return false;}
 let page=vec3u(c)>>vec3u(3u);let e=page.x|(page.y<<6u)|(page.z<<12u);
 let d=${buffer}[${offset}+1u+(e>>4u)];let bit=e&15u;
 if(((d>>bit)&1u)==0u){return false;}
 let r=(d>>16u)+countOneBits(d&((1u<<bit)-1u));let l=vec3u(c)&vec3u(7u);let i=l.x|(l.y<<3u)|(l.z<<6u);
 return ((${buffer}[${offset}+16385u+16u*r+(i>>5u)]>>(i&31u))&1u)!=0u;
}`;
/** A velocity-bound box query reads at most this many tiles, or aligned cubes
 * of one side, before it takes the next side up (uniformBoundCubesWGSL: a
 * cover may overhang the box by under its side, looser and never tighter). */
const BOX_CUBES=64;
/** Workgroups of the h-tile classify launch at most: a fixed grid that
 * strides over the live h list (umCounts.x), so no host tile count sizes it.
 * The ownership's counted-launch grid (COUNTED_GRID): enough to fill the
 * GPU, and an h tile's classify costs about the same as any other's. */
const CENSUS_TILE_GRID=4096;
/** A tile's deep COLD replay record (classify, bounds words): crossing cells
 * (2), importance words (2), gap, directed travel, flags (bit 0 crossing,
 * 1 wet, 2 near, 3 required, 4 a crossing 4h owner), reach (f32 bits), stamp
 * (the region's deep epoch + 1 when written; 0 none) and the policy
 * generation (policy.reach.z: encode advances it at any policy change). A record replays while
 * its region is deep COLD at the epoch that wrote it: nothing in the
 * region's 27-region neighbourhood has run non-COLD since, so phi, velocity,
 * widths and the open-cell masks classify read are the ones it saw. */
const REPLAY_WORDS=10;
/** pageMark's fixed grid: a workgroup a page, strided. */
const PAGE_MARK_GRID=4096;
/** Persistent words after the importance words: a hold per tile, the
 * budget's score cutoff, its histogram. */
const importanceStateWords=(tiles:number)=>tiles;
/** Bytes of the scene uniform prefix the source mark reads (UMSourceParams). */
const SOURCE_PARAMS_BYTES=UNIFORM_PARAMS_BYTES;
/** Bound for the local velocity search and the legacy VOF path. NBFLIP
 * expands each stage's seeds independently (see RESIDENCY_STAGES). */
const RESIDENCY_MARGIN_TILES=RESIDENCY_STENCIL_TILES;
/** Workgroups of the page request's fixed grid, 64 lanes each: more pages
 * or tiles are strided over it. */
const PAGE_REQUEST_GROUPS=64;
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
 /** NB-only quota over requesting tiles; absent/100 admits every candidate. */
 activityBudget?:{percent:number};
 /** A tile this frame's drop or inflow plug can fill is band (absent: it
  * is). Off under a census of requests only: a source is not a request; and
  * "unresolved" for NB-FLIP: only plugs too thin for 4h require h before
  * the surface census can see them. Resolved streams stay 4h. The page seed keeps a source's
  * tiles resident either way. */
 sources?:boolean|"unresolved";
 /** What shape measures on an h tile (UniformDetailShapeMetric; absent:
  * "value"). "displacement": umSurfaceDisplacement. */
 shapeMetric?:"value"|"displacement";
}
/** No importance policy: shape and impact as the policy's tolerances say. */
const LEGACY_IMPORTANCE:UniformMixedDynamicImportance={criteria:{shape:true,thin:false,strain:false,rotation:false,impact:true,approach:false},
 thinThickness:0,strainThreshold:0,rotationThreshold:0,approachSteps:0,bulk:false,holdSteps:0,retireRatio:1};

/** The census's up for gravity g (m/s^2): the axis g pulls along most (the
 * first of equals) and the sign of -g on it; up 0 without gravity. A tilted
 * gravity's lift rule takes the walls along that axis. */
export function uniformMixedUp(g:readonly [number,number,number]):{up:-1|0|1;upAxis:0|1|2}{
 const a=Math.abs(g[0]),b=Math.abs(g[1]),c=Math.abs(g[2]);
 const upAxis=b>=a&&b>=c?1:a>=c?0:2,up=Math.sign(-g[upAxis]);
 return {up:up>0?1:up<0?-1:0,upAxis};
}
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
  * tile, moving toward it faster than along it) and lift (a tile on a closed wall along `upAxis` moving up
  * faster than along the wall). A free fast front meets neither. 0 disables. */
 boundaryTravel:number;
 /** The direction of "up" along upAxis: +1, -1, or 0 without gravity (no lift rule). */
 up:number;
 /** The axis gravity pulls along most (uniformMixedUpAxis): the lift rule's
  * walls run along it. Absent: 1. */
 upAxis?:0|1|2;
 /** Record why each tile is band (UNIFORM_STAGE_REASON, the `reasons`
  * words) for the tiles layer, and every criterion's score (on or off) in
  * the importance words. Off records nothing and adds no work. */
 reasons?:boolean;
 /** Absent: shape and impact only (surfaceTolerance, boundaryTravel). */
 importance?:UniformMixedDynamicImportance;
}

export interface UniformMixedDynamicCensus {
 /** One byte per tile key: the band, or liquid-conditional solid promotion,
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
 * frame traces (the host re-runs that extension first); a bit a tile,
 * kept by block, answers the interface query. Still liquid keeps exactly its interface
 * tiles fine; moving liquid extends the set upstream by its local travel
 * only. Reads back one bit per tile. */
export class UniformMixedDynamicClassifier implements UniformPageInitBlocks {
 readonly label="Uniform dynamic ownership page words";
 readonly allocatedBytes:number;
 private readonly work:GPUBuffer;
 /** By block: every block's velocity-bound cubes (UNIFORM_BOUND_CUBE_WORDS a
  * block), then every block's interface tile bits (UNIFORM_TILE_BITS_WORDS). */
 private readonly bounds:GPUBuffer;
 /** bounds words: the deep COLD section (the heat's deep flags, then its
  * non-COLD epochs, then its deep epochs, a word a region each), then the
  * replay records from replayRecords. */
 private readonly replayOffset:number;
 private readonly replayRecords:number;
 /** census words: decide's band bits, a bit a tile key, kept across censuses. */
 private readonly decidedOffset:number;
 /** The region heat tiers (attachHeat): the census freezes and replays deep COLD regions. */
 private heatTiers?:GPUBuffer;
 /** attachHeat: deep COLD replay (replay) and 4h solids in WARM regions (warmSolids). */
 private heatReplay=false;private warmSolids=false;
 /** The last encode's params words: a change advances the policy
  * generation (policy.reach.z), which every replay record must match. */
 private readonly policyLast=new Uint32Array(28);
 private policyGeneration=1;
 private readonly readback:GPUBuffer;
 private readonly params:GPUBuffer;
 /** The tiles moving bodies sweep (setBodies), a bit a tile key, cleared and
  * rewritten each census; then the host tile plane of the solid-coupled
  * tiles (setSolid). */
 private readonly solidTiles:GPUBuffer;
 /** Ors the tiles moving bodies sweep into solidTiles before a census. */
 private bodies?:(encoder:GPUCommandEncoder,tiles:GPUBuffer)=>void;
 /** solidTiles holds body bits the static mask lacks. */
 private bodyBits=false;
 private readonly resources:GPUBindGroupLayout;
 private readonly group:UniformDetailGroup;
 private readonly pipelines=new Map<string,GPUComputePipeline>();
 /** The page request (pageRequest): the pages the certificate's seeds
  * reach asked of the page pool, in the census's pass after pageSeed. */
 private request?:{readonly pipeline:GPUComputePipeline;readonly group:GPUBindGroup;readonly groups:number};
 /** The page initialise's hook (encodeInitBlocks): the hold and budget flight words of a claimed block's keys. */
 private initBlocks?:{readonly pipeline:GPUComputePipeline;readonly group:GPUBindGroup};
 private static readonly INIT_PASS:GPUComputePassDescriptor={label:"Uniform dynamic ownership page words"};
 private readonly words:number;
 /** Words of a host tile plane (hostTilePlane): solidTiles' and joinTiles' after their key words. */
 private readonly planeWords:number;
 /** Words of the GPU join plane: a bit per tile key (joinTarget). */
 private readonly joinWords:number;
 /** Residency pages (uniformMixedPageCount). */
 private readonly pages:number;
 /** First wet word (wetIndex(0)): wet then active bits, cleared per census. */
 private readonly wetOffset:number;
 /** First near word (nearIndex(0)), after the band reasons: near bits, the audit word, page boxes. */
 private readonly nearOffset:number;
 /** One word per tile after the active bits: the band reason
  * (UNIFORM_STAGE_REASON), written by every decide of a census encoded with
  * policy.reasons, and by solid promotion. Undefined content otherwise. */
 get reasons():{readonly buffer:GPUBuffer;readonly offset:number}{return this.reasonsView??={buffer:this.work,offset:(this.wetOffset+2*this.words)*4};}
 /** The views below name fixed buffers and offsets: each is built once. */
 private reasonsView?:{readonly buffer:GPUBuffer;readonly offset:number};
 private importanceView?:{readonly buffer:GPUBuffer;readonly offset:number};
 private bandView?:UniformMixedBandBits;
 private joinView?:UniformMixedBandBits&{readonly words:number};
 /** Two words per tile (UNIFORM_STAGE_IMPORTANCE), written by every census.
  * Criteria that are off, and tiles shape alone decides, are scored only by
  * a census encoded with policy.reasons. */
 get importance():{readonly buffer:GPUBuffer;readonly offset:number}{return this.importanceView??={buffer:this.work,offset:this.importanceOffset*4};}
 /** First importance word, after the crossing-cell words. */
 private readonly importanceOffset:number;
 /** Stage-major records, four words per region: occupancy and signed radii.
  * A final maximum radius bounds the post-commit mark search only. */
 private readonly localOffset:number;
 /** setSolid marked a coupled tile: the promotion passes run. */
 private solid=false;
 private encoded=false;
 /** Tiles forced into the band by the next census: GPU producer words [0, joinWords), a bit
  * a tile key (joinTarget), then the host tile plane of the host's joins (join). decide
  * ors both into the band; the census clears them once it has decided.
  * Then the region table (setRegions, the builder's): a Fine region's tiles
  * are h whatever the band holds, so the residency certificate seeds them
  * (pageSeed), and decide gives a Coarse one's no band bit. */
 private readonly joinTiles:GPUBuffer;
 /** Host joins (tile words) accumulated since the last encode, uploaded by it. */
 private hostJoin?:number[];
 /** Join words may be non-zero: every census clears them after decide. */
 private joinLive=false;
 /** The band decided by the last encode, for the GPU layout builder. Words
  * [0, HEADER) of the buffer are the census header its receipt carries. */
 get bandBits():UniformMixedBandBits{return this.bandView??={buffer:this.work,wordOffset:HEADER,headerWords:HEADER,auditWord:this.nearOffset+this.words};}
 /** source: the host's scene uniform (UMSourceParams prefix ABI, as the
  * mixed frame's sourceParams). decide marks every tile this frame's drop or
  * inflow plug can fill (uvSourcePhi, frame dt dimsDt.w) as band, so the
  * source lands on h owners. Its values are the ones bound when the census
  * runs: encode the census after the frame's params upload.
  * pool: the frame's page pool. Every census asks it for the pages of its
  * residency certificate; the frame's commit follows the census.
  * particles.allocation: exact current particle tile masks and six signed
  * travel bounds per 16-cell region, produced by FLIP refinement before
  * the census. This includes newly seeded and ballistic particles.
  * solidGeometry: the frame's cached open-cell masks, rebuilt before the
  * census and initialized for new pages before their first classification. */
 constructor(private readonly device:GPUDevice,readonly ownership:UniformMixedOwnership,volume:GPUTexture,phi:GPUTexture,extended:GPUTexture,source:GPUBuffer,private readonly pool:UniformPagePool,private readonly particles:{readonly buffer:GPUBuffer;readonly offset:number;readonly allocation:GPUBuffer},private readonly hostPages:number,private readonly solidGeometry:UniformMixedSolidCoarse,
  /** solidContact "surface": a cut tile is active (solidActive) only where
   * the band already holds it at h, not wherever it is wet; a submerged cut
   * tile the surface census leaves at 4h runs as a cut 4h owner. Its ring
   * (solidPromote) takes only wet tiles: a dry one off the band stays 4h. */
  private readonly solidSurfaceOnly=false){
  const tiles=ownership.capacity.tiles;
  this.words=Math.ceil(tiles/32);
  const address=ownership.capacity.address;
  if(!Number.isInteger(hostPages)||hostPages<1||hostPages>65535)throw new RangeError(`Dynamic ownership host pages ${hostPages} is not an integer in 1..65535`);
  this.planeWords=hostPlaneWords(hostPages);this.joinWords=Math.ceil(address.tileKeys/32);
  this.wetOffset=HEADER+this.words+8*tiles;
  this.nearOffset=this.wetOffset+2*this.words+tiles;
  this.pages=uniformMixedPageCount(tiles);
  // After wet and active: band reasons (a word per tile), then near bits
  // and the audit word (cleared per census), then a seed word per page.
  // Append two crossing-cell words per tile; existing diagnostic offsets stay stable.
  // Then two importance words per tile and the importance state.
  this.importanceOffset=this.nearOffset+this.words+1+this.pages+2*tiles;
  this.localOffset=this.importanceOffset+2*tiles+importanceStateWords(tiles)+narrowBandBudgetWords(tiles);
  // Then decide's own band bits (CC_DECIDED), never cleared.
  this.decidedOffset=this.localOffset+STAGE_RESIDENCY_WORDS*this.pages+1;
  const workBytes=(this.decidedOffset+this.words)*4,readBytes=(HEADER+this.words)*4;
  this.work=device.createBuffer({label:"Uniform dynamic ownership census",size:workBytes,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
  this.readback=device.createBuffer({label:"Uniform dynamic ownership readback",size:readBytes,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  this.params=device.createBuffer({label:"Uniform dynamic ownership policy",size:112,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
  this.solidTiles=device.createBuffer({label:"Uniform dynamic ownership solid tiles",size:(this.joinWords+this.planeWords)*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
  this.joinTiles=device.createBuffer({label:"Uniform dynamic ownership join tiles",size:(this.joinWords+this.planeWords+UNIFORM_MIXED_REGION_TABLE_WORDS)*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
  if(source.size<SOURCE_PARAMS_BYTES)throw new Error(`Dynamic ownership source params hold ${source.size} bytes, expected at least ${SOURCE_PARAMS_BYTES}`);
  // Every block's bound cubes, then every block's interface bits.
  // Then the deep COLD replay (attachHeat): the region heat's deep flags,
  // non-COLD and deep epochs (3 words a region, one copy per census), and
  // REPLAY_WORDS a tile key of classify's results. A fourth region word
  // after the three: the frame's region tier (attachHeat warmSolids).
  this.replayOffset=(UNIFORM_BOUND_CUBE_WORDS+UNIFORM_TILE_BITS_WORDS)*address.blocks;
  this.replayRecords=this.replayOffset+4*this.pages;
  const boundsBytes=(this.replayRecords+REPLAY_WORDS*tiles)*4;
  this.bounds=device.createBuffer({label:"Uniform dynamic ownership bound cubes",size:boundsBytes,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
  this.allocatedBytes=workBytes+boundsBytes+readBytes+112+(2*(this.joinWords+this.planeWords)+UNIFORM_MIXED_REGION_TABLE_WORDS)*4;
  this.resources=uniformDetailBindLayout(device,{entries:[
   ...[0,1,2].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float" as const,viewDimension:"3d" as const}})),
   {binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},
   {binding:4,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},
   {binding:5,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},
   {binding:6,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},
   {binding:7,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},
   {binding:8,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},
   {binding:9,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},
   {binding:10,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},
  ]});
  this.group=uniformDetailGroup(device,{layout:this.resources,entries:[
   {binding:0,resource:volume},{binding:1,resource:phi},
   {binding:2,resource:extended},{binding:3,resource:{buffer:this.work}},{binding:4,resource:{buffer:this.params}},
   {binding:5,resource:{buffer:this.solidTiles}},{binding:6,resource:{buffer:this.bounds}},{binding:7,resource:{buffer:this.joinTiles}},
   {binding:8,resource:{buffer:source,size:SOURCE_PARAMS_BYTES}},
   {binding:9,resource:{buffer:particles.allocation}},
   {binding:10,resource:{buffer:solidGeometry.record}},
  ]});
 }
 async initialize():Promise<void>{
  const h=this.ownership.capacity.lattice.cellSize_m,cap=UNIFORM_MIXED_DYNAMIC_DISTANCE_CAP;
  const module=uniformDetailModule(this.device,{label:"Uniform dynamic ownership census",code:uniformMixedTopologyWGSL(this.ownership.capacity,0)+uniformMixedTopologyNeighboursWGSL()+/* wgsl */`
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
struct DynamicPolicy {step:vec4f,reach:vec4u,surface:vec4f,flow:vec4f,importance:vec4f,shaping:vec4u,activity:vec4u}
@group(1) @binding(4) var<uniform> policy:DynamicPolicy;
@group(1) @binding(5) var<storage,read> solidTiles:array<u32>;
@group(1) @binding(6) var<storage,read_write> bounds:array<u32>;
// Joined tiles: GPU producer words (joinTarget), then the host's tile plane
// (join); then the region table (setRegions).
@group(1) @binding(7) var<storage,read> joinTiles:array<u32>;
${uniformMixedSourceWGSL(8)}
@group(1) @binding(9) var<storage,read> particleAllocation:array<u32>;
@group(1) @binding(10) var<storage,read> solidRecord:array<vec4f>;
// The frame rebuilds this cache before the census. Buried phi vertices hold
// positive air markers, not a fluid interface: only open owners can cross it.
fn umCensusOpenCells(tile:u32)->vec2u{
 let r=vec4u(solidRecord[${this.solidGeometry.count}u+UM_TILES+3u*tile]);
 return vec2u(r.x|(r.y<<16u),r.z|(r.w<<16u));
}
fn umCensusCellOpen(mask:vec2u,c:vec3i)->bool{
 if(any(c<vec3i(0))||any(c>=vec3i(4))){return false;}
 let i=u32(c.x+4*c.y+16*c.z);return (mask[i/32u]&(1u<<(i%32u)))!=0u;
}
// An edge belongs to the fluid domain only if an incident cell is open.
fn umCensusEdgeOpen(mask:vec2u,v:vec3u,a:u32)->bool{
 if(all(mask==vec2u(0xffffffffu))){return true;}
 for(var i=0u;i<4u;i++){
  var c=vec3i(v);c[(a+1u)%3u]-=i32(i&1u);c[(a+2u)%3u]-=i32(i>>1u);
  if(umCensusCellOpen(mask,c)){return true;}
 }
 return false;
}
${localResidencyWGSL}
fn lrRecord(page:u32,stage:u32)->vec4u{let i=${this.localOffset}u+4u*(stage*UM_PAGES+page);return vec4u(atomicLoad(&census[i]),atomicLoad(&census[i+1u]),atomicLoad(&census[i+2u]),atomicLoad(&census[i+3u]));}
// The census tail follows the frame's last phi resolve: every texel of a
// mixed-stencil tile holds umVertexValue, and the census reads only vertices
// of an h tile's closure or 4-aligned ones, so umVertexValue is this load.
// A reader knows what it walks, so no load tests its address: a tile corner
// is canonical (the base block, at every capacity), an h tile's vertex is in
// the h texture (the tile's own, or a neighbour's inside the detail ring).
fn umLoadCorner(p:vec3u)->f32{return ${UNIFORM_DETAIL_4H_LOAD}textureLoad(phi,vec3i(p),0).x;}
fn umLoadFine(p:vec3u)->f32{return textureLoad(phi,vec3i(p),0).x;}
// The same loads of the eight vertices p + s*umCorner(k,2u): one page-table read where one block holds them.
fn umLoadCorner8(p:vec3u,s:u32)->array<f32,8>{return udLoad48_phi(vec3i(p),s);}
fn umLoadFine8(p:vec3u,s:u32)->array<f32,8>{return udLoad8_phi(vec3i(p),s);}
const CAP:u32=${cap}u;
// Largest travel, in h per step, boundary tiles dilate by (a 4-tile radius).
const BOUNDARY_TRAVEL_CAP:u32=16u;
fn umCensusWords()->u32{return (UM_TILES+31u)/32u;}
// Words of the GPU join plane and the body plane, a bit per tile key; the
// host tile planes follow them.
fn umJoinWords()->u32{return (UM_TILES+31u)/32u;}
${hostPlaneWGSL("umHostSolid","solidTiles","umJoinWords()")}
${hostPlaneWGSL("umHostJoined","joinTiles","umJoinWords()")}
${uniformMixedRegionWGSL("umRegionsAt",i=>`joinTiles[umJoinWords()+${this.planeWords}u+${i}]`)}
const H=vec3f(${h.join(",")});

// Per tile: ordered keys of the signed minimum and maximum face velocity per axis.
fn boundIndex(t:u32,k:u32)->u32{return ${HEADER}u+umCensusWords()+6u*t+k;}
// The interface bits: a bit per tile key the criteria require, by block, after
// every block's bound cubes. Plain words, not census atomics: each build
// launch writes its own words and reads only an earlier launch's.
fn interfaceIndex(b:u32,i:u32)->u32{return ${UNIFORM_BOUND_CUBE_WORDS}u*UM_BLOCKS+${UNIFORM_TILE_BITS_WORDS}u*b+i;}
// Deep COLD (UniformRegionHeat, copied in by encode; zero without a heat):
// a region whose 27-region neighbourhood holds its values. decide keeps its
// tiles' widths, holds do not run down and the budget does not drop them;
// under surface authority classify replays their records (REPLAY_WORDS).
const CC_DEEP:u32=${this.replayOffset}u;
const CC_EPOCH:u32=CC_DEEP+${2*this.pages}u;
// The frame's region tiers (attachHeat warmSolids; unread otherwise).
const CC_TIER:u32=CC_DEEP+${3*this.pages}u;
const CC_RECORD:u32=${this.replayRecords}u;
const CC_WORDS:u32=${REPLAY_WORDS}u;
// decide's own band bit per tile key (before solid promotion), kept across
// censuses: a deep COLD tile keeps it, every other live tile rewrites it.
const CC_DECIDED:u32=${this.decidedOffset}u;
fn ccDeep(tile:u32)->bool{return bounds[CC_DEEP+umPageOfTile(tile)]!=0u;}
// The stamp a record written now carries: 0 (write none) unless the region is deep COLD.
fn ccStamp(tile:u32)->u32{
 if((policy.shaping.x&32768u)==0u){return 0u;}
 let region=umPageOfTile(tile);
 if(bounds[CC_DEEP+region]==0u){return 0u;}
 return bounds[CC_EPOCH+region]+1u;
}
fn ccValid(tile:u32)->bool{let stamp=ccStamp(tile);let r=CC_RECORD+CC_WORDS*tile;return stamp!=0u&&bounds[r+8u]==stamp&&bounds[r+9u]==policy.reach.z;}
// A tile of the window: a box of tiles is held to its 64 pages, never to a box
// of the scene's, and a tile nothing stores answers nothing.
fn umWindowTile(p:vec3i)->vec3i{return clamp(p,vec3i(0),vec3i(511));}
// Per required tile: cells between each face and the nearest crossing owner,
// one nibble per face (-x,-y,-z,+x,+y,+z), and its own travel in whole h
// cells per step in the top byte; all ones for a tile that is not required.
fn gapIndex(t:u32)->u32{return ${HEADER}u+umCensusWords()+6u*UM_TILES+t;}
// Per boundary tile: travel in h per step along each direction (-x,-y,-z,
// +x,+y,+z), five bits each; all ones for any other tile. Its dilation
// follows the flow, not a sphere: a climbing sheet refines the tiles above.
fn travelIndex(t:u32)->u32{return gapIndex(UM_TILES)+t;}
// One bit per tile: liquid, a non-air owner (wet); then solid-coupled tiles
// that are wet or in the decided band (active).
fn wetIndex(w:u32)->u32{return travelIndex(UM_TILES)+w;}
fn activeIndex(w:u32)->u32{return wetIndex(umCensusWords())+w;}
fn reasonIndex(t:u32)->u32{return activeIndex(umCensusWords())+t;}
// Band reasons for the tiles layer (policy.surface.w, UNIFORM_STAGE_REASON).
fn recordReasons()->bool{return policy.surface.w>0.0;}
fn recordReason(tile:u32,reason:u32){if(recordReasons()){atomicStore(&census[reasonIndex(tile)],reason);}}
// Residency: one bit per near tile (an h tile, or a 4h tile with V != 0 or a
// corner phi below the frame plan seed's 4h-per-width threshold: a tile the
// seed can mark), the audit word (UNIFORM_MIXED_RESIDENCY_AUDIT), and per
// page the seed word: the packed box of its near and band tiles (pageSeed).
fn nearIndex(w:u32)->u32{return reasonIndex(UM_TILES)+w;}
fn auditIndex()->u32{return nearIndex(umCensusWords());}
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
const SHAPE:u32=0u;const THIN:u32=1u;const STRAIN:u32=2u;const ROTATION:u32=3u;const IMPACT:u32=4u;const APPROACH:u32=5u;
fn criterionOn(k:u32)->bool{return ((policy.shaping.x>>k)&1u)!=0u;}
fn triggerBit(k:u32)->u32{return 1u<<(${IMPORTANCE.triggeredShift}u+k);}
// A criterion that is off is scored only for a layer.
fn scored(k:u32)->bool{return criterionOn(k)||recordReasons();}
fn scoreByte(score:f32)->u32{return u32(clamp(score*${IMPORTANCE.scoreOne}.0,0.0,255.0));}
// No criterion, hold or layer can require a tile (encode): the interface
// table is all zeros, and importance, the prefix and the cubes that would
// build it are not launched.
fn unseeded()->bool{return (policy.shaping.x&1024u)!=0u;}
// Shape measures an h tile by umSurfaceDisplacement, not by value error.
fn shapeDisplacement()->bool{return (policy.shaping.x&2048u)!=0u;}
// A tile shape requires needs no other score, unless a layer shows them or
// the budget ranks it by its top one.
fn scoreAll()->bool{return recordReasons()||nbBudgetOn();}
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
var<workgroup> replayTile:u32;
var<workgroup> tileCrossing:array<atomic<u32>,2>;
var<workgroup> tileLow:array<atomic<u32>,3>;
var<workgroup> tileHigh:array<atomic<u32>,3>;
var<workgroup> tileGap:array<atomic<u32>,6>;
var<workgroup> tileReach:array<atomic<u32>,6>;
var<workgroup> tileError:atomic<u32>;
// The h tile's signed 4h displacement summed over its crossings (1/1024 h)
// and their count: umSurfaceDisplacement's volume term.
var<workgroup> tileSigned:atomic<i32>;
var<workgroup> tileCrossings:atomic<u32>;
// The h tile's (4h)^2/12 laplacian(phi) summed over its interior vertices
// near the surface (1/1024 h) and their count: umCurveError.
var<workgroup> tileCurve:atomic<i32>;
var<workgroup> tileCurves:atomic<u32>;
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
// its gradient (the distance to their zero set, to first order). Signed
// (positive: the 4h surface lies on the liquid side), the crossings' mean is
// the volume 4h takes per unit surface (tileSigned, tileCrossings): a curved
// surface's is one-signed, (4h)^2/12 of its curvature, however small the
// largest is, and every closed body loses 2 pi (4h)^2/12 of area per section
// when coarsened (Gauss-Bonnet), whatever its radius.
fn umSurfaceDisplacement(tile:u32,lane:u32)->f32{
 let origin=umTileCoord(tile)*4u;let open=umCensusOpenCells(tile);
 var c:array<f32,8>;var worst=0.0;var along=0.0;var crossings=0u;var curve=0.0;var curves=0u;
 c=umLoadCorner8(origin,4u);
 let low=vec4f(c[0],c[2],c[4],c[6]);let high=vec4f(c[1],c[3],c[5],c[7]);let dx=high-low;
 for(var i=lane;i<125u;i+=64u){
  let v=umCorner(i,5u);let value=umLoadFine(origin+v);
  // An interior vertex within 1.5 h of the surface: the mean displacement a
  // 4h lattice makes of it anywhere, (L^2/12) phi_aa per axis with L = 4
  // cells, 4/3 of the h second differences (umCurveError).
  if(all(v>=vec3u(1u))&&all(v<=vec3u(3u))&&abs(value)<=1.5*UM_H){
   var sum=0.0;var closed=false;
   for(var a=0u;a<3u;a++){
    var below=v;below[a]-=1u;var above=v;above[a]+=1u;
    if(!umCensusEdgeOpen(open,below,a)||!umCensusEdgeOpen(open,v,a)){closed=true;break;}
    sum+=umLoadFine(origin+below)+umLoadFine(origin+above)-2.0*value;
   }
   if(!closed){curve+=sum*(4.0/3.0)/UM_H;curves++;}
  }
  for(var a=0u;a<3u;a++){
   if(v[a]>=4u||!umCensusEdgeOpen(open,v,a)){continue;}
   var n=v;n[a]+=1u;let next=umLoadFine(origin+n);
   if((value<0.0)==(next<0.0)){continue;}
   var t=vec3f(v)*0.25;t[a]+=0.25*value/(value-next);
   let x0=mix(low,high,t.x);
   let y0=mix(vec2f(x0.x,x0.z),vec2f(x0.y,x0.w),t.y);
   let gx=mix(vec2f(dx.x,dx.z),vec2f(dx.y,dx.w),t.y);
   // Per tile width (4h), so the distance in h is 4 |phi| / |gradient|.
   let gradient=vec3f(mix(gx.x,gx.y,t.z),mix(x0.y-x0.x,x0.w-x0.z,t.z),y0.y-y0.x);
   let d=4.0*mix(y0.x,y0.y,t.z)/max(length(gradient),4.0e-3*UM_H);
   worst=max(worst,abs(d));along+=d;crossings++;
  }
 }
 if(crossings!=0u){atomicAdd(&tileSigned,i32(clamp(along,-1.0e5,1.0e5)*1024.0));atomicAdd(&tileCrossings,crossings);}
 if(curves!=0u){atomicAdd(&tileCurve,i32(clamp(curve,-1.0e5,1.0e5)*1024.0));atomicAdd(&tileCurves,curves);}
 return worst;
}
// The volume term of an h tile's 4h error: VOLUME_SHAPE times the mean
// signed displacement, so a tile coarsens only where 4h keeps the surface
// within the tolerance everywhere and within a quarter of it on average.
// At 0.5 h that holds a body of radius under ~10 h at h (Sato 2018's
// geometric heat |phi - G*phi| / dx > T_g / A_g = 0.2, taken at 4h), and
// lets a flat or gently curved surface coarsen as before.
const VOLUME_SHAPE:f32=4.0;
fn umVolumeError()->f32{
 let n=atomicLoad(&tileCrossings);if(n==0u){return 0.0;}
 return VOLUME_SHAPE*abs(f32(atomicLoad(&tileSigned))/(1024.0*f32(n)));
}
// A moving surface's 4h error wherever the lattice lies: VOLUME_SHAPE
// (4h)^2/12 |laplacian(phi)| over the h tile's interior vertices near it. The
// displacement above is zero for a corner or curve the 4h lattice happens to
// carry (a block aligned to it at rest) and stays small until the surface
// has moved off the lattice, by when a census has coarsened it and the 4h
// resample has rounded it. Sato 2018's geometric heat (|phi - G*phi| / dx,
// a laplacian) "acts to retain sharp corners during the liquid's
// free-fall". A 90 degree corner scores several h; a cylinder of radius
// 10.7 h the 0.5 h tolerance; a flat or gently curved surface nothing.
fn umCurveError()->f32{
 let n=atomicLoad(&tileCurves);if(n==0u){return 0.0;}
 return VOLUME_SHAPE*abs(f32(atomicLoad(&tileCurve))/(1024.0*f32(n)));
}
// A 4h tile's (umResolutionError, one lane per tile): the largest
// (4h)^2/12 |laplacian(phi)| at its corners within 2 h of the surface.
var<private> coarseCurve:f32;
// A surface moves when it crosses a 4h tile within MOVING_SECONDS: its
// tile's fastest extended face, in h per step, against 4 dt / MOVING_SECONDS.
// A still surface keeps the displacement test alone.
const MOVING_SECONDS:f32=0.5;
fn umMovingTravel()->f32{return 4.0*policy.step.x/MOVING_SECONDS;}
fn umClassMoving(c:TileClass)->bool{
 var speed=0.0;
 for(var a=0u;a<3u;a++){
  if(c.low[a]!=0xffffffffu){speed=max(speed,abs(orderValue(c.low[a]))*policy.step.x/H[a]);}
  if(c.high[a]!=0u){speed=max(speed,abs(orderValue(c.high[a]))*policy.step.x/H[a]);}
 }
 return speed>=umMovingTravel();
}
// Surface error of the tile at 4h, in h (non-negative, ordered as bits).
// wide: an h tile's 4h neighbours (umWideShared).
fn umResolutionError(tile:u32,lane:u32,width:u32,wide:u32)->f32{
 let origin=umTileCoord(tile)*4u;var worst=0.0;
 if(width==1u){
  if(shapeDisplacement()){return umSurfaceDisplacement(tile,lane);}
  var corners:array<f32,8>;
  corners=umLoadCorner8(origin,4u);
  for(var i=lane;i<125u;i+=64u){
   let v=umCorner(i,5u);let t=vec3f(v)*0.25;
   // A vertex a 4h tile shares is a corner or the corners' own trilinear
   // value: 4h loses nothing there.
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
 var lap=0.0;
 for(var a=0u;a<3u;a++){
  var lo=vec3i(c/4u);lo[a]-=1;var hi=vec3i(c/4u);hi[a]+=1;
  if(!umWideVertexPresent(lo)||!umWideVertexPresent(hi)){continue;}
  let below=umLoadCorner(4u*vec3u(lo));let above=umLoadCorner(4u*vec3u(hi));
  // A neighbour further from the centre than its span allows is no
  // distance: far air outside the box holds a marker (16h and more), and its
  // second difference refined every surface tile beside the box's low faces.
  if(abs(below-centre)>6.0*H[a]||abs(above-centre)>6.0*H[a]){continue;}
  worst=max(worst,abs(above-2.0*centre+below)/(16.0*UM_H));
  lap+=above-2.0*centre+below;
 }
 // (L^2/12) phi_aa is a twelfth of the 4h second difference. Within 2 h of
 // the surface: a deeper corner's stencil reaches the medial axis of a body
 // three tiles across, which no 4h zero crossing sees (and h would drop).
 if(abs(centre)<=2.0*UM_H){coarseCurve=max(coarseCurve,abs(lap)/(12.0*UM_H));}
 return worst;
}
// The solid mask at any tile of the window: a body's bit (by the tile's key,
// where its page is stored) or the host's coupled plane.
fn umSolidAt(q:vec3i)->bool{
 if(any(q!=umWindowTile(q))){return false;}
 let t=umTileKey(q);if(umKeyStored(t)&&((solidTiles[t/32u]>>(t%32u))&1u)!=0u){return true;}
 return umHostSolid(q);
}
fn umSolidTile(p:vec3u)->bool{return umSolidAt(vec3i(p));}
// Coupled tiles include fully open neighbours of a wall. They need solid
// geometry support, but are not obstacles to tangential flow. Impact and
// approach use the baked capacity instead of that support mask.
fn umObstacleAt(q:vec3i)->bool{
 let t=umTileKey(q);if(umKeyStored(t)){return solidRecord[t].x<1.0;}
 return umSolidAt(q);
}
${uniformSolidRefinementWGSL(this.solidGeometry.count)}
// Tiles between tile t and the closed wall of its page across direction k
// (-x,-y,-z,+x,+y,+z): the SOLID page there. NO_WALL where that page is
// stored or AIR: past every tile scan.
const NO_WALL:u32=0xffffu;
fn umWallBeyond(t:u32,k:u32)->u32{
 let a=k%3u;let high=k>=3u;var d=vec3i(0);d[a]=select(-1,1,high);
 if(umBlockNeighbour(t>>9u,d)!=UM_SOLID_WORD){return NO_WALL;}
 let l=umLocalTile(t);return select(l[a],7u-l[a],high);
}
// Boundary rules (UniformMixedDynamicPolicy.boundaryTravel), from the tile's
// own velocity bounds and the cells between its faces and its surface owners.
// Travel thresholds halve for an h tile (hysteresis) and its margin doubles.
fn umDirectionTravel(c:TileClass,k:u32)->f32{
 let a=k%3u;
 if(k<3u){return max(-orderValue(c.low[a]),0.0)*policy.step.x/H[a];}
 return max(orderValue(c.high[a]),0.0)*policy.step.x/H[a];
}
struct Boundary{score:f32,triggered:bool}
fn umBoundary(tile:u32,p:vec3u,width:u32,c:TileClass)->Boundary{
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
  let beyond=umWallBeyond(tile,k);
  var met=4.0*f32(beyond)<=reach;
  for(var n=1u;!met&&n<=min(beyond,CAP)&&4.0*f32(n-1u)<=reach;n++){
   var q=vec3i(p);q[a]+=select(-i32(n),i32(n),high);
   met=umObstacleAt(q);
  }
  if(!met){continue;}
  result.score=max(result.score,travel[k]/threshold);
  if(travel[k]>=threshold){result.triggered=true;if(!recordReasons()){return result;}}
 }
 // Lift: on a closed wall along the up axis (reach.w bits 2-3; bits 0-1 its
 // sign, 1 +, 2 -), moving up faster than along the wall.
 if((policy.reach.w&3u)==0u){return result;}
 let upAxis=policy.reach.w>>2u;
 let up=select(travel[upAxis],travel[3u+upAxis],(policy.reach.w&3u)==1u);
 if(up<least){return result;}
 for(var k=0u;k<6u;k++){
  let a=k%3u;if(a==upAxis){continue;}
  let b=3u-a-upAxis;let along=max(travel[b],travel[3u+b]);
  if(umWallBeyond(tile,k)==0u&&up>=along){result.score=max(result.score,up/threshold);if(up>=threshold){result.triggered=true;}}
 }
 return result;
}
// Thin features: the deepest liquid and the deepest air among the 4-aligned
// vertices one tile around the tile (its own corners and the next ring). A
// sheet, film or neck holds no vertex deeper than half its thickness, and
// neither does a closing air gap. Lane i of 64 samples one vertex.
fn umDepthSample(tile:u32,i:u32)->vec2u{
 let v=vec3i(umTileCoord(tile))+vec3i(umCorner(i,4u))-vec3i(1);
 if(!umWideVertexPresent(v)){return vec2u(0u);}
 let value=umLoadCorner(vec3u(v)*4u);
 return vec2u(orderKey(-value),orderKey(value));
}
// Interface: an owner whose corner phi changes sign. V/phi disagreement
// without one is counted, not refined: 4h V error is at or below fine on the
// same cells (plan, same-cell A/B).
fn umClassifyOwner(tile:u32,width:u32,lane:u32,c:ptr<function,TileClass>){
 let open=umCensusOpenCells(tile);
 if(all(open==vec2u(0u))||(width==1u&&(open[lane/32u]&(1u<<(lane%32u)))==0u)){return;}
 let surfaceAuthority=(policy.shaping.x&32768u)!=0u;
 let side=4u/width;
 let origin=umTileCoord(tile)*4u+umCorner(lane,side)*width;
 // By the owner's width: a 4h owner's texels are canonical, an h owner's the h texture's.
 let coarseOwner=width==4u;
 var v=0.0;if(!surfaceAuthority){if(coarseOwner){v=${UNIFORM_DETAIL_4H_LOAD}textureLoad(volume,vec3i(origin),0).x;}else{v=textureLoad(volume,vec3i(origin),0).x;}}
 var inside=0u;var deep=0u;var lowPhi=3.0e38;var highPhi=-3.0e38;
 var c8:array<f32,8>;if(coarseOwner){c8=umLoadCorner8(origin,4u);}else{c8=umLoadFine8(origin,width);}
 for(var k=0u;k<8u;k++){
  let value=c8[k];
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
    if(umFaceLow(anchor,axis)){continue;}
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
   let boundary=umBoundary(tile,umTileCoord(tile),width,c);
   w1|=scoreByte(boundary.score);
   if(boundary.triggered&&criterionOn(IMPACT)){
    w1|=triggerBit(IMPACT);directed=0u;
    for(var k=0u;k<6u;k++){directed|=min(u32(ceil(umDirectionTravel(c,k))),31u)<<(5u*k);}
   }
  }
  gap=min(u32(ceil(speed)),255u)<<24u;for(var k=0u;k<6u;k++){gap|=c.gap[k]<<(4u*k);}
 }
 // The velocity keys are classify's alone: a replay keeps them in place.
 for(var a=0u;a<3u;a++){atomicStore(&census[boundIndex(tile,a)],c.low[a]);atomicStore(&census[boundIndex(tile,3u+a)],c.high[a]);}
 // A positive coarse distance is not a physical reader. NB needs owned h
 // tiles and liquid here;
 // gradient and particle readers separately audit their actual footprints.
 let near=width==1u||(c.flags&8u)!=0u;
 let required=select(near,width==1u||(c.flags&4u)!=0u,(policy.shaping.x&32768u)!=0u);
 let flags=select(0u,1u,crossing)|select(0u,2u,(c.flags&4u)!=0u)|select(0u,4u,near)|select(0u,8u,required)|select(0u,16u,crossing&&width!=1u);
 let reachBits=bitcast<u32>(reach);
 umPublishTile(tile,c.crossing,w0,w1,gap,directed,flags,reachBits);
 let stamp=ccStamp(tile);
 if(stamp!=0u){
  let r=CC_RECORD+CC_WORDS*tile;
  bounds[r]=c.crossing.x;bounds[r+1u]=c.crossing.y;bounds[r+2u]=w0;bounds[r+3u]=w1;bounds[r+4u]=gap;bounds[r+5u]=directed;bounds[r+6u]=flags;bounds[r+7u]=reachBits;bounds[r+8u]=stamp;bounds[r+9u]=policy.reach.z;
 }
}
// classify's per-tile results (umFinishTile, or a replay of its record).
fn umPublishTile(tile:u32,cells:vec2u,w0:u32,w1:u32,gap:u32,directed:u32,flags:u32,reach:u32){
 // Non-negative: ordered as bits. One global atomic per workgroup (callers).
 atomicMax(&groupReach,reach);
 for(var k=0u;k<2u;k++){atomicStore(&census[crossingIndex(tile,k)],cells[k]);atomicStore(&census[importanceIndex(tile,k)],select(w0,w1,k==1u));}
 atomicStore(&census[gapIndex(tile)],gap);
 atomicStore(&census[travelIndex(tile)],directed);
 if((flags&1u)!=0u){atomicAdd(&census[0],1u);}
 if((flags&2u)!=0u){atomicOr(&census[wetIndex(tile/32u)],1u<<(tile%32u));}
 if((flags&4u)!=0u){atomicOr(&census[nearIndex(tile/32u)],1u<<(tile%32u));}
 if((flags&8u)!=0u&&!umTileResident(tile)){atomicOr(&census[auditIndex()],1u);}
}
// A deep COLD tile's record (ccValid): its owners' census counter too.
fn umReplayTile(tile:u32){
 let r=CC_RECORD+CC_WORDS*tile;let flags=bounds[r+6u];
 if((flags&16u)!=0u){atomicAdd(&census[7],1u);}
 umPublishTile(tile,vec2u(bounds[r],bounds[r+1u]),bounds[r+2u],bounds[r+3u],bounds[r+4u],bounds[r+5u],flags,bounds[r+7u]);
}
// h tiles: one workgroup per tile, one lane per owner. A fixed grid strides
// over the live h list; umCounts is uniform, so the loop keeps its barriers.
@compute @workgroup_size(64) fn classify(@builtin(workgroup_id) gid:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 for(var job=gid.x;job<umCounts.x;job+=groups.x){
 let tile=umTopology[UM_TILES+job];
 // A deep COLD tile replays its record on one lane.
 if(lane==0u){replayTile=u32(ccValid(tile));}
 if(workgroupUniformLoad(&replayTile)!=0u){
  if(lane==0u){resetGroupCensus();umReplayTile(tile);flushGroupCensus();}
  workgroupBarrier();continue;
 }
 if(lane==0u){resetGroupCensus();atomicStore(&mixedTile,0u);for(var k=0u;k<2u;k++){atomicStore(&tileCrossing[k],0u);}for(var a=0u;a<3u;a++){atomicStore(&tileLow[a],0xffffffffu);atomicStore(&tileHigh[a],0u);atomicStore(&tileGap[a],15u);atomicStore(&tileGap[3u+a],15u);atomicStore(&tileReach[a],15u);atomicStore(&tileReach[3u+a],15u);}atomicStore(&tileError,0u);atomicStore(&tileSigned,0);atomicStore(&tileCrossings,0u);atomicStore(&tileCurve,0);atomicStore(&tileCurves,0u);atomicStore(&tileWide,0u);for(var k=0u;k<2u;k++){atomicStore(&tileDeep[k],0u);}}workgroupBarrier();
 // The h list holds h tiles only: one lane per owner.
 const width=1u;
 // The tile's 4h neighbours, for the shape criterion (umWideShared).
 if(lane<27u&&policy.step.y>0.0&&!shapeDisplacement()){
  let n=vec3i(umTileCoord(tile))+vec3i(umCorner(lane,3u))-vec3i(1);
  if(umTilePresent(n)&&umTileWidth(umTileAt(vec3u(n)))!=1u){atomicOr(&tileWide,1u<<lane);}
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
  if(shapeDisplacement()){c.error=bitcast<u32>(max(bitcast<f32>(c.error),max(umVolumeError(),select(0.0,umCurveError(),umClassMoving(c)))));}
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
  if(ccValid(tile)){umReplayTile(tile);}
  else{
   var c=umEmptyClass();umClassifyOwner(tile,4u,0u,&c);
   if((c.flags&2u)!=0u&&policy.step.y>0.0){
    var error=0.0;coarseCurve=0.0;for(var l=0u;l<8u;l++){error=max(error,umResolutionError(tile,l,4u,0u));}
    // A moving curve or corner (umCurveError), halved like the corners'
    // second differences: refined at twice the error an h tile keeps.
    if(shapeDisplacement()&&umClassMoving(c)){error=max(error,0.5*VOLUME_SHAPE*coarseCurve);}
    c.error=bitcast<u32>(error);
   }
   if((c.flags&2u)!=0u&&thinWanted()){for(var i=0u;i<64u;i++){c.deep=max(c.deep,umDepthSample(tile,i));}}
   umFinishTile(tile,4u,c);
  }
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
// Signed speed toward a tile moving along k of tile t beyond it on that side,
// in h per step: its fastest face that way. Negative recedes: the edge of a
// falling stream reads its own liquid a tile below across a sliver of air.
fn umTileToward(t:u32,k:u32)->f32{
 let a=k%3u;
 if(k<3u){return orderValue(atomicLoad(&census[boundIndex(t,3u+a)]))*policy.step.x/H[a];}
 return -orderValue(atomicLoad(&census[boundIndex(t,a)]))*policy.step.x/H[a];
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
  var below=vec3i(p);below[b]-=1;var above=vec3i(p);above[b]+=1;
  if(umTilePresent(below)){let t=umTileAt(vec3u(below));if(umTileWet(t)){lo=umTileFlow(t);span+=4.0*H[b];}}
  if(umTilePresent(above)){let t=umTileAt(vec3u(above));if(umTileWet(t)){hi=umTileFlow(t);span+=4.0*H[b];}}
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
// ahead to a solid tile, liquid beyond air (closing at the relative speed:
// liquid falling away as fast closes nothing) or a
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
  let gap=f32((word>>(4u*k))&15u);
  let limit=min(CAP,u32(ceil(min(horizon*travel,64.0)/4.0))+1u);
  var air=false;var gapCells=-1.0;var closing=travel;
  // One tile past the limit: a closed wall there is still within it.
  for(var n=1u;n<=limit+1u;n++){
   var q=vec3i(p);q[a]+=select(-i32(n),i32(n),high);
   // A closed wall: a SOLID page, or the window's edge.
   let w=select(UM_SOLID_WORD,umPageWord(q>>vec3u(3u)),all(q==umWindowTile(q)));
   if(w!=0u&&!umBlockStored(w)){gapCells=4.0*f32(n-1u)+gap;break;}
   if(n>limit){break;}
   // An AIR page is air.
   if(w==0u){air=true;continue;}
   if(umObstacleAt(q)){gapCells=4.0*f32(n-1u)+gap;break;}
   let t=512u*w+umTileLocal(vec3u(q));
   if(!umTileWet(t)){air=true;continue;}
   // Liquid with no air before it is this surface's own.
   if(air){gapCells=4.0*f32(n-1u)+gap;closing+=umTileToward(t,k);}
   break;
  }
  if(gapCells>=1.0){best=max(best,horizon*closing/gapCells);}
 }
 return best;
}
fn umScoreByte(w0:u32,w1:u32,k:u32)->u32{
 if(k<4u){return (w0>>(8u*k))&255u;}
 return (w1>>(8u*(k-4u)))&255u;
}
@compute @workgroup_size(64) fn importance(@builtin(global_invocation_id) gid:vec3u){
 let tile=gid.x+umDispatchX*64u*gid.y;if(!umTileLive(tile)){return;}
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
  // A resolved source authors its tiles again each step: its back rim
  // need not request h. Unresolved inflows retain their criteria so thin
  // moving liquid can also heat and seed particles.
  if(((w1>>${IMPORTANCE.triggeredShift}u)&63u)!=0u&&!umSourceinflowNeedsFine()&&umSourceTile(vec3i(p))){w1&=~(63u<<${IMPORTANCE.triggeredShift}u);}
  let triggered=((w1>>${IMPORTANCE.triggeredShift}u)&63u)!=0u;
  var kept=triggered;

  var held=false;
  if(holding){
   remaining=atomicLoad(&census[holdIndex(tile)]);
   if(kept){remaining=policy.shaping.y;}
   else if(remaining>0u){
    // A dropped tile's hold runs down whatever it scores: the budget wins.
    // A deep COLD tile's hold is frozen with its width (decide).
    held=true;if((triggered||top<policy.shaping.w)&&!ccDeep(tile)){remaining--;}
   }
  }
  required=kept||held;
  w1|=select(0u,${IMPORTANCE.required}u,required)|select(0u,${IMPORTANCE.held}u,held);
  // Bulk liquid nothing scores keeps classify's words.
  if(w0!=own.x){atomicStore(&census[importanceIndex(tile,0u)],w0);}
  if(w1!=own.y){atomicStore(&census[importanceIndex(tile,1u)],w1);}
 }
 if(holding){atomicStore(&census[holdIndex(tile)],remaining);}
 // The budget ranks and decides after this pass (activityRank .. activityAdmit).
 if(nbBudgetOn()){return;}
 publishImportance(tile,w1);
}
fn publishImportance(tile:u32,w1:u32){
 let required=(w1&${IMPORTANCE.required}u)!=0u;
 let crossing=(w1&${IMPORTANCE.crossing}u)!=0u;let shaped=(w1&triggerBit(SHAPE))!=0u;
 // Required tiles seed decide: their interface bit (interfaceBits reads the
 // required flag of the stored word), crossing cells (the whole tile without
 // a surface of its own), gap and, for a boundary tile, its directed travel.
 // classify stored a crossing tile's as if required.
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
${narrowBandBudgetWGSL}
// The interface bits, once the criteria (and the budget) have decided: the
// required flag of every live tile's importance word.
${uniformTileBitsWGSL({name:"interface",word:(b,i)=>`bounds[interfaceIndex(${b},${i})]`,store:(b,i,v)=>`bounds[interfaceIndex(${b},${i})]=${v};`,flag:t=>`(atomicLoad(&census[importanceIndex(${t},1u)])&${IMPORTANCE.required}u)!=0u`,air:false})}
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
// Velocity bounds over any tile box: a tile's keys live in the atomic census,
// a block's cubes in bounds (atomic loads and stores do not coalesce).
${uniformBoundCubesWGSL({tile:(t,k)=>`atomicLoad(&census[boundIndex(${t},${k})])`,word:i=>`bounds[${i}]`,store:(i,v)=>`bounds[${i}]=${v};`,budget:BOX_CUBES})}
// Cells past a point to every tile whose faces bound a sample there: two
// across 4h owners (their centres are two cells in). At h a sample reads the
// faces within a cell of it, and the remap hands a promoted tile's 4h face
// over tilted toward the faces beside it: a patch is a convex combination of
// its tile's faces and those of the nearer tile on each tangential axis
// (remapSlope), a cell and a half from the patch's centre at most.
const SAMPLE_REACH:f32=2.5;
// Signed bounds of every velocity sampled in the cell box [lo, hi] after
// gravity adds shift, joined with zero: sampling near a wall or solid blends
// in its zero face, and the trace stops short at a solid or the domain clamp.
struct Flow{low:vec3f,high:vec3f}
fn sampledFlow(lo:vec3f,hi:vec3f,shift:vec3f)->Flow{
 let keys=rangeKeys(vec3i(floor((lo-SAMPLE_REACH)/4.0)),vec3i(floor((hi+SAMPLE_REACH)/4.0)));
 var f=Flow(vec3f(0),vec3f(0));
 for(var a=0u;a<3u;a++){
  if(keys[a]!=0xffffffffu){f.low[a]=orderValue(keys[a]);}
  if(keys[3u+a]!=0u){f.high[a]=orderValue(keys[3u+a]);}
 }
 f.low=min(vec3f(0),f.low+min(shift,vec3f(0)));f.high=max(vec3f(0),f.high+max(shift,vec3f(0)));
 return f;
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
 let a=umWindowTile(p-vec3i(radius));let b=umWindowTile(p+vec3i(radius));
 for(var z=a.z;z<=b.z;z++){for(var y=a.y;y<=b.y;y++){for(var x=a.x;x<=b.x;x++){
  let q=vec3i(x,y,z);let t=umTileKey(q);if(!umKeyStored(t)){continue;}
  let word=atomicLoad(&census[gapIndex(t)]);
  if(word==0xffffffffu){continue;}
  let directed=atomicLoad(&census[travelIndex(t)]);
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
 let tile=umTileKey(q);if(!umKeyStored(tile)){return false;}
 let mask=vec2u(atomicLoad(&census[crossingIndex(tile,0u)]),atomicLoad(&census[crossingIndex(tile,1u)]));
 if(all(mask==vec2u(0u))){return false;}
 let row=((1u<<u32(b.x-a.x+1))-1u)<<u32(a.x);
 for(var z=a.z;z<=b.z;z++){for(var y=a.y;y<=b.y;y++){
  let lane=4u*u32(y)+16u*u32(z);
  if((mask[lane/32u]&(row<<(lane%32u)))!=0u){return true;}
 }}
 return false;
}
// Whole tiles inside the box use the interface bits; only its shell needs the
// crossing-cell mask. This preserves the cheap rejection of distant tiles.
// home: the block of the tile that asks.
fn departureMeetsSurface(home:u32,lo:vec3f,hi:vec3f)->bool{
 let first=vec3i(floor((lo-1e-3)/4.0));let last=vec3i(floor((hi+1e-3)/4.0));
 if(!interfaceAny(home,first,last)){return false;}
 let inner=vec3i(ceil((lo-1e-3)/4.0));let innerLast=last-vec3i(1);
 if(interfaceAny(home,inner,innerLast)){return true;}
 let a=umWindowTile(first);let b=umWindowTile(last);
 for(var z=a.z;z<=b.z;z++){for(var y=a.y;y<=b.y;y++){
  if(!interfaceAny(home,vec3i(a.x,y,z),vec3i(b.x,y,z))){continue;}
  let core=y>=inner.y&&y<=innerLast.y&&z>=inner.z&&z<=innerLast.z;
  for(var x=a.x;x<=b.x;x++){
   if(core&&x>=inner.x&&x<=innerLast.x){x=innerLast.x;continue;}
   if(crossingCellsMeet(vec3i(x,y,z),lo,hi)){return true;}
  }
 }}
 return false;
}
// A joined tile (joinTarget, join) is band whatever the trace finds; so,
// with sources on (umSourceBand), is a tile this frame's drop or inflow plug
// can fill (umSourceTile): the source phi (a lower bound on the distance to
// its ball or plug) at the tile centre within half the tile diagonal, plus
// two cells for the aperture edge and the donor layer upstream of the
// nozzle face. Liquid-conditional solid promotion then treats either as
// band. Off, a source tile is decided like any other, and the page seed
// still keeps it resident.
fn umJoined(t:u32)->bool{
 return ((joinTiles[t/32u]>>(t%32u))&1u)!=0u||umHostJoined(vec3i(umTileCoord(t)));
}
fn umSourceBand()->bool{return (policy.shaping.x&512u)!=0u||((policy.shaping.x&65536u)!=0u&&umSourceinflowNeedsFine());}
fn umSourceTile(p:vec3i)->bool{
 if(umSourceParams.drop.w<=0.0&&umSourceinflowStrength()<=0.0){return false;}
 let centre=4.0*vec3f(p)+vec3f(2.0);
 return umSourceuvSourcePhi(centre,3.0e38)<=0.5*length(4.0*H)+2.0*MAX_H;
}
// Chebyshev tile distance: each layer includes face, edge and corner neighbours.
// Read actual crossings, not the required-seed prefix (criteria may omit seeds).
fn surfaceAllowed(tile:u32,p:vec3i)->bool{
 if((atomicLoad(&census[importanceIndex(tile,1u)])&${IMPORTANCE.crossing}u)!=0u){return true;}
 let radius=i32((policy.shaping.x>>13u)&3u);
 if(radius!=0){
  let lo=umWindowTile(p-vec3i(radius));let hi=umWindowTile(p+vec3i(radius));
  for(var z=lo.z;z<=hi.z;z++){for(var y=lo.y;y<=hi.y;y++){for(var x=lo.x;x<=hi.x;x++){
   let t=umTileKey(vec3i(x,y,z));if(!umKeyStored(t)){continue;}
   if((atomicLoad(&census[importanceIndex(t,1u)])&${IMPORTANCE.crossing}u)!=0u){return true;}
  }}}
 }
 return movingShapeNear(tile,p);
}
// Tile speed in h per step from its velocity keys; a tile with no faces has none.
fn umKeySpeed(t:u32)->f32{
 var speed=0.0;
 for(var a=0u;a<3u;a++){
  let low=atomicLoad(&census[boundIndex(t,a)]);let high=atomicLoad(&census[boundIndex(t,3u+a)]);
  if(low!=0xffffffffu){speed=max(speed,abs(orderValue(low))*policy.step.x/H[a]);}
  if(high!=0u){speed=max(speed,abs(orderValue(high))*policy.step.x/H[a]);}
 }
 return speed;
}
// A moving surface shape requires (a corner or curve 4h would round:
// umCurveError) carries its h tiles with it: decide's departure trace may
// refine any tile within the step's travel of one (Chebyshev tiles, from the
// tile's own speed, one to three), not only crossing ones. Elsewhere a
// moving h surface enters a 4h tile and is regenerated from the 4h
// interpolant; a flat or still one keeps the crossing filter.
fn movingShapeNear(tile:u32,p:vec3i)->bool{
 let reach=clamp(i32(ceil(umKeySpeed(tile)/4.0)),1,3);
 if(!interfaceAny(tile>>9u,p-vec3i(reach),p+vec3i(reach))){return false;}
 let want=${IMPORTANCE.crossing}u|${IMPORTANCE.required}u|triggerBit(SHAPE);
 let lo=umWindowTile(p-vec3i(reach));let hi=umWindowTile(p+vec3i(reach));
 for(var z=lo.z;z<=hi.z;z++){for(var y=lo.y;y<=hi.y;y++){for(var x=lo.x;x<=hi.x;x++){
  let t=umTileKey(vec3i(x,y,z));if(!umKeyStored(t)||t==tile){continue;}
  if((atomicLoad(&census[importanceIndex(t,1u)])&want)==want&&umKeySpeed(t)>=umMovingTravel()){return true;}
 }}}
 return false;
}
@compute @workgroup_size(64) fn decide(@builtin(global_invocation_id) gid:vec3u){
 let tile=gid.x+umDispatchX*64u*gid.y;if(!umTileLive(tile)){return;}
 let p=vec3i(umTileCoord(tile));let scale=policy.step.x/H;
 // A Coarse region that no Fine one overlaps holds no band tile, whatever
 // asks; solidPromote may still add one.
 let decided=CC_DECIDED+tile/32u;let bit=1u<<(tile%32u);
 if((umRegionsAt(vec3u(p))&3u)==2u){atomicAnd(&census[decided],~bit);if(umTileWidth(tile)==1u){atomicAdd(&census[5],1u);}recordReason(tile,${REASON.skipped}u);return;}
 let hostJoined=umJoined(tile);let source=!hostJoined&&umSourceBand()&&umSourceTile(p);
 // A deep COLD tile keeps the last census's decision: nothing around it
 // moved since (joins and sources still band it). Solid promotion may still
 // refine it: the band certificate needs that.
 if(ccDeep(tile)&&!hostJoined&&!source){
  let width=umTileWidth(tile);
  if((atomicLoad(&census[decided])&bit)!=0u){
   atomicAdd(&census[2],1u);if(width!=1u){atomicAdd(&census[4],1u);}
   atomicOr(&census[${HEADER}u+tile/32u],1u<<(tile%32u));recordReason(tile,${REASON.closure}u);
  }else{if(width==1u){atomicAdd(&census[5],1u);}recordReason(tile,${REASON.skipped}u);}
  return;
 }
 atomicAnd(&census[decided],~bit);
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
  atomicOr(&census[${HEADER}u+tile/32u],1u<<(tile%32u));atomicOr(&census[decided],bit);
  recordReason(tile,select(${REASON.travel}u,joinReason,joined));
  return;
 }
 // A current interface tile stays fine: phi does not only move with the
 // flow (residual sheets left behind a falling surface persist in place).
 let width=umTileWidth(tile);
 // No interface tile within the farthest any departure box can reach: the
 // trace below cannot meet one (one word where no page around holds one).
 let far=departureReachTiles();
 if(!joined&&(unseeded()||!interfaceAny(tile>>9u,p-vec3i(far),p+vec3i(far)))){if(width==1u){atomicAdd(&census[5],1u);}recordReason(tile,${REASON.skipped}u);return;}
 let crossing=!unseeded()&&(atomicLoad(&census[importanceIndex(tile,1u)])&${IMPORTANCE.required}u)!=0u;
 var fine=joined||crossing;
 var reason=select(${REASON.crossing}u,joinReason,joined);
 // Views only: a required tile with no surface of its own.
 if(crossing&&!joined&&recordReasons()&&(atomicLoad(&census[importanceIndex(tile,1u)])&${IMPORTANCE.crossing}u)==0u){reason=${REASON.bulk}u;}
 if(!fine){
  // Closed: a boundary vertex belongs to both tiles it separates.
  let margin=policy.flow.w+4.0*f32(i32(policy.reach.x)+select(0,i32(policy.reach.y),width==1u));
  // Trace the tile's points back one frame at a time, newest frame first, as
  // RK2 does (x - dt·u(x - dt/2·u(x))): each stage
  // bounds only the velocities its own sample box reads, so a tile's box
  // follows its flow instead of every speed within a radius of it. The
  // newer frames trace a velocity gravity has changed since this census.
  // The box is held to nothing: where no page is stored it bounds no
  // velocity and meets no surface.
  let steps=max(1u,u32(policy.surface.z));let s=scale/f32(steps);
  var lo=4.0*vec3f(p);var hi=lo+vec3f(4.0);
  for(var i=0u;i<steps;i++){
   let shift=policy.flow.xyz*f32(steps-1u-i);
   let start=sampledFlow(lo,hi,shift);
   let mid=sampledFlow(lo-0.5*start.high*s,hi-0.5*start.low*s,shift);
   lo=lo-mid.high*s;hi=hi-mid.low*s;
  }
  fine=departureMeetsSurface(tile>>9u,lo-vec3f(margin),hi+vec3f(margin));
  // Views only: closure is the part a still surface (dt = 0) already needs.
  if(fine&&recordReasons()){
   let box=4.0*vec3f(p);
   reason=select(${REASON.travel}u,${REASON.closure}u,departureMeetsSurface(tile>>9u,box-vec3f(margin),box+vec3f(4.0+margin)));
  }
 }
 if(!fine){if(width==1u){atomicAdd(&census[5],1u);}recordReason(tile,${REASON.traced}u);return;}
 atomicAdd(&census[2],1u);if(width!=1u){atomicAdd(&census[4],1u);}
 atomicOr(&census[${HEADER}u+tile/32u],1u<<(tile%32u));atomicOr(&census[decided],bit);
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
 let tile=gid.x+umDispatchX*64u*gid.y;if(!umTileLive(tile)){return;}
 if(!umSolidTile(umTileCoord(tile))){return;}
 ${this.warmSolids?`let tier=bounds[CC_TIER+umPageOfTile(tile)];
 let coarseHeat=tier==${UNIFORM_REGION_TIER.warm}u||(tier==${UNIFORM_REGION_TIER.cold}u&&umTileWidth(tile)!=1u);`:"let coarseHeat=false;"}
 if(umSolidContactActive(tile,umBit(wetIndex(0u),tile),umBit(${HEADER}u,tile),${this.solidSurfaceOnly},coarseHeat)){
  atomicOr(&census[activeIndex(tile/32u)],1u<<(tile%32u));
 }
}
@compute @workgroup_size(64) fn solidPromote(@builtin(global_invocation_id) gid:vec3u){
 let tile=gid.x+umDispatchX*64u*gid.y;if(!umTileLive(tile)){return;}
 if(umBit(${HEADER}u,tile)){return;}
 let p=vec3i(umTileCoord(tile));
 let a=umWindowTile(p-vec3i(1));let b=umWindowTile(p+vec3i(1));
 // Mirror the nearest open tile's resolution into closed padding. A
 // diagonal fine neighbour must not widen the fine band along a flat wall
 // when the tile directly across that wall is coarse.
 var wallDistance=4;
 if(solidRecord[tile].x==0.0){
  for(var z=a.z;z<=b.z;z++){for(var y=a.y;y<=b.y;y++){for(var x=a.x;x<=b.x;x++){
   let q=vec3i(x,y,z);let t=umTileKey(q);
   if(umKeyStored(t)&&solidRecord[t].x>1e-5){let d=abs(q-p);wallDistance=min(wallDistance,d.x+d.y+d.z);}
  }}}
 }
 ${this.warmSolids?`// warmResolution "4h": a COLD region keeps its widths. Its h tile stays h
 // (with or without a cut of its own), and its tiles promote no neighbour:
 // that relayout of the frozen pool warmed the region through the remap's
 // mid-frame raise, whose 4h solids demoted the tiles again at the next
 // census, a two-frame flicker along every COLD/WARM boundary.
 let keep=bounds[CC_TIER+umPageOfTile(tile)]==${UNIFORM_REGION_TIER.cold}u&&umTileWidth(tile)==1u;`:"let keep=false;"}
 // Static surface-only rings need wet neighbours; a wet body's active
 // tile always retains its complete traction/interpolation stencil.
 let wet=umBit(wetIndex(0u),tile);
 for(var z=a.z;z<=b.z;z++){for(var y=a.y;y<=b.y;y++){for(var x=a.x;x<=b.x;x++){
  let t=umTileKey(vec3i(x,y,z));
  if(!keep){
   if(!umKeyStored(t)){continue;}
   // A closed coarse tile must not impose 4h hanging vertices on an h
   // liquid wall. Exterior SOLID pages have no such vertex authority;
   // give embedded padding the open neighbour's boundary resolution.
   // Read decide's frozen bits, not this pass's band writes: promotion
   // cannot race or recursively spread through the padding.
   let d=abs(vec3i(x,y,z)-p);
   let wallSupport=solidRecord[tile].x==0.0&&d.x+d.y+d.z==wallDistance&&solidRecord[t].x>1e-5&&(umBit(CC_DECIDED,t)||umStaticFine(t));
   ${this.warmSolids?`let coldNeighbour=t!=tile&&bounds[CC_TIER+umPageOfTile(t)]==${UNIFORM_REGION_TIER.cold}u;`:"let coldNeighbour=false;"}
   if(!wallSupport&&(!umBit(activeIndex(0u),t)||!umSolidContactRing(t,wet,${this.solidSurfaceOnly},coldNeighbour))){continue;}
  }
  atomicOr(&census[${HEADER}u+tile/32u],1u<<(tile%32u));
  atomicAdd(&census[2],1u);atomicAdd(&census[19],1u);recordReason(tile,${REASON.solid}u);
  if(umTileWidth(tile)==1u){atomicSub(&census[5],1u);}else{atomicAdd(&census[4],1u);}
  return;
 }}}
}
// Residency certificate (uniformMixedResidencyWord, uniform-mixed-residency.ts),
// after the band is final. pageSeed: per page, the box of its near tiles, the
// band's new h tiles and the builder's static h tiles (tile coordinates within
// the page, three bits each; bit 18 set when non-empty); 0 for a page of a
// block that holds none. The static tiles are h after this build
// without the band: a Fine region drawn in far air would otherwise be h in a
// page this certificate leaves absent, which the next census audits as fatal.
// Bit 19: a seed is the liquid's (a band tile, or a near tile that is not a
// static h tile), which is what the page request audits (pageRequest).
fn umStaticFine(t:u32)->bool{return (umRegionsAt(umTileCoord(t))&1u)!=0u;}
@compute @workgroup_size(64) fn pageSeed(@builtin(global_invocation_id) gid:vec3u){
 let page=gid.x+umDispatchX*64u*gid.y;
 // The in-frame readers' sticky closure bits join this census's audit, and
 // its closure radius is the certificate's from here on: the request reads
 // it before the commit and the mark after.
 if(page==0u){
  if(umSupport[UM_RESIDENCY+1u]!=0u){atomicOr(&census[auditIndex()],${UNIFORM_MIXED_RESIDENCY_AUDIT.closure}u);}
  umSupport[UM_RESIDENCY+2u]=residencyRadius();
 }
 if(page>=UM_PAGES){return;}
 var lo=vec3u(3u);var hi=vec3u(0u);var found=false;var liquid=false;
 var masks:array<vec2u,${RESIDENCY_STAGES.length}>;
 let staged=(policy.shaping.x&32768u)!=0u;
 let held=umBlockStored(page>>3u)&&umBlockWord(page>>3u)!=UM_FREE;
 let particle=select(vec2u(0u),vec2u(particleAllocation[8u*page],particleAllocation[8u*page+1u]),held);
 for(var lane=0u;lane<64u;lane++){
  let t=umPageTile(page,lane);if(!umTileLive(t)){continue;}
  let near=umBit(nearIndex(0u),t);let authored=umStaticFine(t);let own=umBit(${HEADER}u,t)||(near&&!authored);
  let sample=((particle[lane>>5u]>>(lane&31u))&1u)!=0u;
  // A source tile seeds whether or not decide made it band (umSourceBand).
  let source=umSourceTile(vec3i(umTileCoord(t)));
  if(!near&&!own&&!authored&&!sample&&!source){continue;}
  let c=umCorner(lane,4u);lo=min(lo,c);hi=max(hi,c);found=true;liquid=liquid||own||sample||source;
  let band=umBit(${HEADER}u,t);
  let wet=umBit(wetIndex(0u),t);
  let crossing=(atomicLoad(&census[importanceIndex(t,1u)])&${IMPORTANCE.crossing}u)!=0u;
  // Sources and particle reconstruction can create a new surface this frame.
  // Budget admission may omit a crossing: read the physical census flag,
  // never the refinement-only crossing mask that importance can clear.
  let seeds=array<bool,${RESIDENCY_STAGES.length}>(umTileWidth(t)==1u||authored||band,wet||source||sample||authored,wet||source||sample,crossing||source||sample,sample,false);
  for(var stage=0u;stage<${RESIDENCY_STAGES.length}u;stage++){
   if(select(stage==5u,seeds[stage],staged)){masks[stage][lane>>5u]|=1u<<(lane&31u);}
  }
 }
 atomicStore(&census[pageBoxIndex(page)],select(0u,lo.x|(lo.y<<3u)|(lo.z<<6u)|(hi.x<<9u)|(hi.y<<12u)|(hi.z<<15u)|${PAGE_SEEDED}u|select(0u,${PAGE_LIQUID}u,liquid),found));
 // Bound every velocity a swept seed or its numerical halo can sample.
 // The global envelope only locates local extrema; it never dilates seeds.
 let origin=16.0*vec3f(umPageCoord(page));
 let envelope=departureTravel()+4.0*f32(${RESIDENCY_MARGIN_TILES}u);
 let shift=max(policy.surface.z-1.0,0.0)*policy.flow.xyz;
 var flow:Flow;
 if(found){flow=sampledFlow(origin-vec3f(envelope),origin+vec3f(16.0+envelope),shift);}
 let pl=bitcast<vec3f>(vec3u(particleAllocation[8u*page+2u],particleAllocation[8u*page+3u],particleAllocation[8u*page+4u]));
 let ph=bitcast<vec3f>(vec3u(particleAllocation[8u*page+5u],particleAllocation[8u*page+6u],particleAllocation[8u*page+7u]));
 let margins=array<f32,${RESIDENCY_STAGES.length}>(${RESIDENCY_STAGES.map(s=>s.margin+".0").join(",")});
 for(var stage=0u;stage<${RESIDENCY_STAGES.length}u;stage++){
  let at=${this.localOffset}u+4u*(stage*UM_PAGES+page);
  atomicStore(&census[at],masks[stage].x);atomicStore(&census[at+1u],masks[stage].y);
  var radii=vec2u(0u);
  if((masks[stage].x|masks[stage].y)!=0u&&stage!=0u){
   // Ballistic travel belongs to particle-created fluid, not every air tile
   // in the same region. Authored/new ownership has no advection footprint.
   let samples=held&&stage>=2u;
   radii=lrRadii(flow.low,flow.high,policy.step.x/H,select(vec3f(0),pl,samples),select(vec3f(0),ph,samples),margins[stage]);
   let r=max(lrUnpack(radii.x),lrUnpack(radii.y));
   atomicMax(&census[${this.localOffset+STAGE_RESIDENCY_WORDS*this.pages}u],u32(max(r.x,max(r.y,r.z))));
  }
  atomicStore(&census[at+2u],radii.x);atomicStore(&census[at+3u],radii.y);
 }
}
// Legacy global envelope, retained for the bounded local velocity search.
fn residencyRadius()->u32{return u32(ceil(departureTravel()/4.0))+${RESIDENCY_MARGIN_TILES}u;}
${localResidencyMarkWGSL}
// pageMark, after the commit: the same exact occupied-tile dilation as
// the request. The largest local radius only bounds the seed search; it
// never expands an individual region. Claimed blocks had zero seed masks.
// A workgroup a page, strided over a fixed grid (lrResidentLane).
@compute @workgroup_size(64) fn pageMark(@builtin(workgroup_id) g:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 let radius=atomicLoad(&census[${this.localOffset+STAGE_RESIDENCY_WORDS*this.pages}u]);
 if(g.x==0u&&lane==0u){umSupport[UM_RESIDENCY+2u]=radius;}
 for(var page=g.x;page<UM_PAGES;page+=groups.x){
  if(lane==0u){atomicStore(&lrMarks,0u);}
  workgroupBarrier();
  lrResidentLane(page,min(radius,${4*UNIFORM_RESIDENCY_PAGE_REACH}u),lane);
  workgroupBarrier();
  if(lane==0u){umSupport[UM_RESIDENCY+4u+page]=atomicLoad(&lrMarks);}
 }
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
 if(lane==COMPACT_LANES-1u){umSupport[UM_RESIDENCY]=compactCounts[lane];}
}
`});
  const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");
  if(errors.length)throw new Error(errors.map(m=>`${m.lineNum}: ${m.message}`).join("\n"));
  const layout=this.device.createPipelineLayout({bindGroupLayouts:[this.ownership.bindLayout,this.resources]});
  await Promise.all(["classify","classifyCoarse","importance","interfaceBits","interfaceBlocks","boundCubes","decide","solidActive","solidPromote","pageSeed","pageMark","pageCompact",...NARROW_BAND_BUDGET_ENTRIES].map(async entryPoint=>{this.pipelines.set(entryPoint,await uniformDetailPipeline(this.device,this.ownership,{layout,compute:{module,entryPoint,constants:{umDispatchX:this.ownership.dispatchX}}}));}));
  await this.initializeRequest();
 }
 /** The page request: a module of its own beside the census's, which binds
  * the census words read-only and the page pool's state and work. */
 private async initializeRequest():Promise<void>{
  const device=this.device,groups=Math.min(Math.ceil(RESIDENCY_STAGES.length*this.pages/64),PAGE_REQUEST_GROUPS),tiles=this.ownership.capacity.tiles;
  const module=uniformDetailModule(device,{label:"Uniform dynamic ownership page request",code:uniformMixedTopologyWGSL(this.ownership.capacity,0)+/* wgsl */`
@group(1) @binding(0) var<storage,read> census:array<u32>;
${uniformPagePoolWGSL(this.pool.layout,1,1,2)}
${localResidencyWGSL}
fn lrRecord(page:u32,stage:u32)->vec4u{let i=${this.localOffset}u+4u*(stage*UM_PAGES+page);return vec4u(census[i],census[i+1u],census[i+2u],census[i+3u]);}
${localResidencyRequestWGSL}
// Union only after each stage dilates its own exact occupancy and signed
// travel. A lane a (page, stage): a lane a page held the six stages' boxes
// in series (0.16 ms over 43 workgroups in an all-4h frame).
@compute @workgroup_size(64) fn pageRequest(@builtin(global_invocation_id) gid:vec3u){
 for(var item=gid.x;item<${RESIDENCY_STAGES.length}u*UM_PAGES;item+=${64*groups}u){lrAsk(item%UM_PAGES,item/UM_PAGES);}
}
`});
  const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");
  if(errors.length)throw new Error(errors.map(m=>`${m.lineNum}: ${m.message}`).join("\n"));
  const resources=device.createBindGroupLayout({label:"Uniform dynamic ownership page request",entries:[
   {binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},
   {binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},
   {binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},
  ]});
  const layout=device.createPipelineLayout({bindGroupLayouts:[this.ownership.bindLayout,resources]});
  const pipeline=await uniformDetailPipeline(device,this.ownership,{layout,compute:{module,entryPoint:"pageRequest"}});
  this.request={pipeline,groups,group:device.createBindGroup({layout:resources,entries:[{binding:0,resource:{buffer:this.work}},{binding:1,resource:{buffer:this.pool.state}},{binding:2,resource:{buffer:this.pool.work}}]})};
  // The words a tile key carries from census to census: its hold's steps and its budget flight word. A claimed block's keys take 0, a tile no census has seen.
  const hold=this.importanceOffset+2*tiles,flight=hold+narrowBandBudgetWords(tiles),l=this.pool.layout;
  const words=device.createShaderModule({label:this.label,code:/* wgsl */`${uniformPagePoolWGSL(l,0,0)}
@group(0) @binding(1) var<storage,read_write> census:array<u32>;
${uniformPageInitBlocksWGSL(l,"censusInitBlocks",512,`census[${hold}u+key]=0u;census[${flight}u+key]=0u;`)}`});
  const wordErrors=(await words.getCompilationInfo()).messages.filter(m=>m.type==="error");
  if(wordErrors.length)throw new Error(wordErrors.map(m=>`${m.lineNum}: ${m.message}`).join("\n"));
  const wordLayout=device.createBindGroupLayout({label:this.label,entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}}]});
  this.initBlocks={pipeline:await device.createComputePipelineAsync({label:this.label,layout:device.createPipelineLayout({bindGroupLayouts:[wordLayout]}),compute:{module:words,entryPoint:"censusInitBlocks"}}),
   group:device.createBindGroup({label:this.label,layout:wordLayout,entries:[{binding:0,resource:{buffer:this.pool.state}},{binding:1,resource:{buffer:this.work}}]})};
 }
 /** After the pool's commit, before the next census: one fixed launch that strides the pool's new list. */
 encodeInitBlocks(encoder:GPUCommandEncoder):void{
  const init=this.initBlocks;if(!init)throw new Error("Dynamic ownership census is not initialized");
  const pass=encoder.beginComputePass(UniformMixedDynamicClassifier.INIT_PASS);pass.setPipeline(init.pipeline);pass.setBindGroup(0,init.group);pass.dispatchWorkgroups(uniformPageInitGroups(this.pool.layout));pass.end();
 }
 /** Solid-coupled tiles, for the impact rule: one uniformMixedTileWord a
  * tile, any order, repeats allowed; empty, none. */
 setSolid(tiles:ArrayLike<number>):void{
  this.device.queue.writeBuffer(this.solidTiles,this.joinWords*4,hostTilePlane(tiles,this.hostPages,"solid tiles"));
  this.solid=tiles.length>0;
 }
 /** The region table (uniformMixedRegionTable), the builder's too: every
  * census seeds a Fine region's pages resident and bands no tile of a Coarse
  * one that no Fine one holds. */
 setRegions(table:Uint32Array<ArrayBuffer>):void{
  if(table.length!==UNIFORM_MIXED_REGION_TABLE_WORDS||table[0]!>UNIFORM_MIXED_REGION_LIMIT)throw new Error(`Dynamic ownership region table has ${table.length} words and ${table[0]} regions, expected ${UNIFORM_MIXED_REGION_TABLE_WORDS} words and at most ${UNIFORM_MIXED_REGION_LIMIT} regions`);
  this.device.queue.writeBuffer(this.joinTiles,(this.joinWords+this.planeWords)*4,table);
 }
 /** Moving bodies: before each census the body words are cleared and `mark`
  * ors the tiles they may cut over the next frame (on the GPU, from the body
  * state, a bit a tile key), and the liquid-conditional promotion passes
  * run. undefined leaves the coupled plane alone. */
 setBodies(mark?:(encoder:GPUCommandEncoder,tiles:GPUBuffer)=>void):void{this.bodies=mark;}
 /** The frame's region heat (UniformRegionHeat.tiers, regions = the census's
  * pages), read after its head encode: decide keeps a deep COLD region's
  * widths and classify replays its tiles (REPLAY_WORDS). Without it every
  * tile is classified and decided each census. */
 attachHeat(tiers:GPUBuffer,replay=true,warmSolids=false):void{
  if(tiers.size<4*(UNIFORM_REGION_HEAT_DEEP_WORD(this.pages)+3*this.pages))throw new Error(`Dynamic ownership: region heat tiers of ${tiers.size} bytes do not hold ${this.pages} regions`);
  if(this.pipelines.size)throw new Error("Dynamic ownership: attachHeat after initialize");
  if(!replay&&!warmSolids)throw new Error("Dynamic ownership: attachHeat with neither replay nor warmSolids");
  this.heatTiers=tiers;this.heatReplay=replay;this.warmSolids=warmSolids;
 }
 /** Host join: `tiles` (one uniformMixedTileWord a tile, repeats allowed)
  * are band in the next census, ored with any earlier join it has not
  * consumed. Uploaded by encode. */
 join(tiles:ArrayLike<number>):void{
  const held=this.hostJoin??=[];
  for(let i=0;i<tiles.length;i++)held.push(tiles[i]!);
 }
 /** The GPU join words: a producer ors one bit per tile (atomicOr) in work
  * submitted before the next census, or encoded before it in its encoder;
  * that census consumes and clears them. */
 joinTarget():UniformMixedBandBits&{readonly words:number}{this.joinLive=true;return this.joinView??={buffer:this.joinTiles,wordOffset:0,words:this.joinWords};}
 /** encode's params words, for the queue (which copies them at each write). */
 private readonly policyBytes=new ArrayBuffer(112);
 private readonly policyF=new Float32Array(this.policyBytes);
 private readonly policyU=new Uint32Array(this.policyBytes);
 /** encode's launches in order, with their fixed grids (y 0: one argument).
  * when: 0 always, 1 a seeded census, 2 one under an activity budget, 3 solids or bodies. */
 private launches?:readonly {readonly pipeline:GPUComputePipeline;readonly when:0|1|2|3;readonly x:number;readonly y:number}[];
 /** encodeResidency's launches: pageMark's fixed grid over the pages, then pageCompact's one workgroup. */
 private marks?:{readonly mark:GPUComputePipeline;readonly x:number;readonly y:number;readonly compact:GPUComputePipeline};
 private static readonly RESIDENCY_PASS:GPUComputePassDescriptor={label:"Uniform dynamic ownership residency"};
 /** The certificate's flags and list, in the encoder of the page commit
  * that follows encode and straight after it: the mark reads the page table
  * the commit left, so a page it claimed for this census's request is
  * resident before any stage of the generation reads a flag. */
 encodeResidency(encoder:GPUCommandEncoder):void{
  if(!this.request)throw new Error("Dynamic ownership is not initialized");
  const m=this.marks??=this.residencyLaunches();
  const pass=encoder.beginComputePass(UniformMixedDynamicClassifier.RESIDENCY_PASS);
  pass.setBindGroup(0,this.ownership.bindGroup);pass.setBindGroup(1,this.group.group);
  pass.setPipeline(uniformDetailPick(m.mark));pass.dispatchWorkgroups(m.x,m.y);
  pass.setPipeline(uniformDetailPick(m.compact));pass.dispatchWorkgroups(1);
  pass.end();
 }
 private residencyLaunches(){
  return {mark:this.pipelines.get("pageMark")!,x:Math.min(this.pages,PAGE_MARK_GRID),y:1,compact:this.pipelines.get("pageCompact")!};
 }
 private censusLaunches(){
  const tiles=this.ownership.capacity.tiles,x=this.ownership.dispatchX,blocks=this.ownership.capacity.address.blocks;
  const launch=(entry:string,when:0|1|2|3)=>{
   // Fixed grids: classify strides over the h list; classifyCoarse has a
   // lane per tile, bounded by the 4h count; the bound cubes a workgroup
   // per block and the interface summary a lane per block; the rest are
   // sized by the tile keys.
   const lines=entry.startsWith("page")?this.pages:tiles;
   const pipeline=this.pipelines.get(entry)!,[gx,gy]=uniformKeyGroups(lines,64,x,[0,0]);
   if(entry==="classify")return {pipeline,when,x:Math.min(CENSUS_TILE_GRID,tiles),y:0};
   if(entry==="boundCubes")return {pipeline,when,x:blocks,y:0};
   if(entry==="interfaceBlocks")return {pipeline,when,x:Math.ceil(blocks/64),y:0};
   if(entry==="activityCutoff")return {pipeline,when,x:1,y:0};
   return {pipeline,when,x:gx,y:gy};
  };
  return [launch("classify",0),launch("classifyCoarse",0),launch("importance",1),...NARROW_BAND_BUDGET_ENTRIES.map(entry=>launch(entry,2)),
   launch("interfaceBits",1),launch("interfaceBlocks",1),launch("boundCubes",0),
   launch("decide",0),launch("solidActive",3),launch("solidPromote",3),launch("pageSeed",0)];
 }
 private static readonly PASS:GPUComputePassDescriptor={label:"Uniform dynamic ownership census"};
 /** encode's range checks: the message is built by the failure alone. */
 private finite(name:string,value:number):void{if(!Number.isFinite(value)||value<0)throw new Error(`Dynamic ownership ${name} must be finite and non-negative: ${value}`);}
 private distance(name:string,value:number):void{if(!Number.isSafeInteger(value)||value<0||value>UNIFORM_MIXED_DYNAMIC_DISTANCE_CAP)throw new Error(`Dynamic ownership ${name} must be an integer in 0..${UNIFORM_MIXED_DYNAMIC_DISTANCE_CAP}: ${value}`);}
 /** Encode after a completed frame, while its ownership and the local speed
  * velocity (the one the next frame advects with) are still in place. Every
  * launch is a fixed grid bounded on the GPU (umCounts): no host tile count.
  * readback: copy the header and band bits for read() (tools, tests and the
  * host relayout path); without it the census reads nothing back. */
 encode(encoder:GPUCommandEncoder,policy:UniformMixedDynamicPolicy,readback=true):void{
  const request=this.request;
  if(this.pipelines.size!==12+NARROW_BAND_BUDGET_ENTRIES.length||!request)throw new Error("Dynamic ownership census is not initialized");
  const queue=this.device.queue,F=this.policyF,U=this.policyU;
  // Each of the policy's numbers is read once: a read of one is a heap number until this function is optimised.
  const dt=policy.dt,surfaceTolerance=policy.surfaceTolerance,fullTolerance=policy.fullTolerance,emptyTolerance=policy.emptyTolerance,reach=policy.reach,hysteresis=policy.hysteresis;
  this.finite("dt",dt);this.finite("fullTolerance",fullTolerance);this.finite("emptyTolerance",emptyTolerance);
  this.distance("reach",reach);this.distance("hysteresis",hysteresis);
  if(!Number.isFinite(surfaceTolerance)||surfaceTolerance<0)throw new Error(`Dynamic ownership surfaceTolerance must be finite and non-negative: ${surfaceTolerance}`);
  F[0]=dt;F[1]=surfaceTolerance;F[2]=fullTolerance;F[3]=emptyTolerance;
  const up=policy.up,upAxis=policy.upAxis??1;
  if(up!==1&&up!==-1&&up!==0)throw new Error(`Dynamic ownership up must be 1, -1 or 0: ${up}`);
  if(upAxis!==0&&upAxis!==1&&upAxis!==2)throw new Error(`Dynamic ownership upAxis must be 0, 1 or 2: ${upAxis}`);
  U[4]=reach;U[5]=hysteresis;U[7]=up===0?0:(up===1?1:2)|upAxis<<2;
  const fastTravel=policy.fastTravel,boundaryTravel=policy.boundaryTravel,steps=policy.steps;
  this.finite("fastTravel",fastTravel);this.finite("boundaryTravel",boundaryTravel);
  if(!Number.isSafeInteger(steps)||steps<1)throw new Error(`Dynamic ownership steps must be a positive integer: ${steps}`);
  const gravity=policy.gravity,gx=gravity[0],gy=gravity[1],gz=gravity[2];
  if(gravity.length!==3||!Number.isFinite(gx)||!Number.isFinite(gy)||!Number.isFinite(gz))throw new Error(`Dynamic ownership gravity must be three finite components: ${policy.gravity}`);
  // (One step: the frame is the step, and no quotient is taken.)
  const frame=steps===1?dt:dt/steps;
  F[8]=fastTravel;F[9]=boundaryTravel;F[10]=steps;F[11]=policy.reasons?1:0;F[12]=gx*frame;F[13]=gy*frame;F[14]=gz*frame;F[15]=UNIFORM_MIXED_DYNAMIC_SURFACE_DRIFT;
  const importance=policy.importance??LEGACY_IMPORTANCE;
  const thinThickness=importance.thinThickness,strainThreshold=importance.strainThreshold,rotationThreshold=importance.rotationThreshold,approachSteps=importance.approachSteps,retireRatio=importance.retireRatio,holdSteps=importance.holdSteps;
  this.finite("importance thinThickness",thinThickness);this.finite("importance strainThreshold",strainThreshold);this.finite("importance rotationThreshold",rotationThreshold);
  this.finite("importance approachSteps",approachSteps);this.finite("importance retireRatio",retireRatio);
  // Thin reads vertices one tile around: phi is a distance there only so far.
  if(thinThickness>8)throw new Error(`Dynamic ownership importance thinThickness must be at most 8 h: ${thinThickness}`);
  if(!Number.isSafeInteger(holdSteps)||holdSteps<0)throw new Error(`Dynamic ownership importance holdSteps must be a non-negative integer: ${holdSteps}`);
  const surfaceDistance=importance.surfaceDistance??0;
  if(!Number.isSafeInteger(surfaceDistance)||surfaceDistance<0||surfaceDistance>3)throw new Error(`Dynamic ownership surfaceDistance must be an integer in 0..3: ${surfaceDistance}`);
  const activityBudget=importance.activityBudget;
  const budgetPercent=activityBudget?activityBudget.percent:100;
  if(!Number.isFinite(budgetPercent)||budgetPercent<0||budgetPercent>100)throw new Error("Activity budget values must be in 0..100");
  const activityPercent=Math.round(budgetPercent);
  const activityBudgetOn=!!activityBudget&&activityPercent<100;
  U[24]=activityPercent;U[25]=0;U[26]=0;U[27]=0;
  const shapeMetric=importance.shapeMetric??"value";
  if(shapeMetric!=="value"&&shapeMetric!=="displacement")throw new Error(`Dynamic ownership importance shapeMetric must be "value" or "displacement": ${String(shapeMetric)}`);
  F[16]=thinThickness;F[17]=strainThreshold;F[18]=rotationThreshold;F[19]=approachSteps;
  // A census of requests only (no criterion, hold or layer; joins, sources,
  // solids and bodies remain) requires no tile: decide then reads no table.
  let criteria=0;for(let k=0;k<UNIFORM_DETAIL_CRITERIA.length;k++)if(importance.criteria[UNIFORM_DETAIL_CRITERIA[k]!])criteria|=1<<k;
  const seeded=!!policy.reasons||fastTravel>0||holdSteps>0||criteria!==0;
  U[20]=(importance.bulk?256:0)|(importance.sources==="unresolved"?65536:importance.sources===false?0:512)|(seeded?0:1024)|(shapeMetric==="displacement"?2048:0)|(importance.surfaceOnly?4096:0)|(surfaceDistance<<13)|(importance.surfaceAuthority?32768:0)|criteria;
  U[21]=holdSteps;U[22]=0xffffffff;U[23]=Math.min(255,Math.floor(retireRatio*IMPORTANCE.scoreOne));
  // Any change of the policy (word 6 aside) retires every replay record.
  {const last=this.policyLast;let changed=false;for(let i=0;i<28;i++)if(i!==6&&last[i]!==U[i]){last[i]=U[i]!;changed=true;}if(changed)this.policyGeneration=(this.policyGeneration%0xfffffffe)+1;}
  U[6]=this.policyGeneration;queue.writeBuffer(this.params,0,U);
  // The interface bits and the bound cubes are written whole before decide
  // reads them, classify writes every tile's keys, gap, travel and importance
  // words; only the counters, band bits and wet/active
  // bits accumulate. The holds and the budget's cutoff persist by design.
  encoder.clearBuffer(this.work,0,(HEADER+this.words)*4);
  encoder.clearBuffer(this.work,this.wetOffset*4,2*this.words*4);
  encoder.clearBuffer(this.work,this.nearOffset*4,(this.words+1)*4);
  encoder.clearBuffer(this.work,4*(this.localOffset+STAGE_RESIDENCY_WORDS*this.pages),4);
  if(this.bodies||this.bodyBits){
   encoder.clearBuffer(this.solidTiles,0,this.joinWords*4);
   this.bodies?.(encoder,this.solidTiles);this.bodyBits=!!this.bodies;
  }
  // Host joins upload now, ordered after every earlier submission (and
  // after the census that consumed the last ones); GPU producers or theirs.
  if(this.hostJoin){this.device.queue.writeBuffer(this.joinTiles,this.joinWords*4,hostTilePlane(this.hostJoin,this.hostPages,"join tiles"));this.hostJoin=undefined;this.joinLive=true;}
  // The frame head's deep COLD flags and epochs (one copy: the heat's three
  // region sections are contiguous), before any launch reads them.
  if(this.heatTiers&&this.heatReplay)encoder.copyBufferToBuffer(this.heatTiers,4*UNIFORM_REGION_HEAT_DEEP_WORD(this.pages),this.bounds,4*this.replayOffset,4*3*this.pages);
  if(this.heatTiers&&this.warmSolids)encoder.copyBufferToBuffer(this.heatTiers,0,this.bounds,4*(this.replayOffset+3*this.pages),4*this.pages);
  // One pass: every dispatch is its own usage scope, so each reads what
  // the earlier ones wrote.
  const launches=this.launches??=this.censusLaunches(),solids=this.solid||!!this.bodies;
  const pass=encoder.beginComputePass(UniformMixedDynamicClassifier.PASS);
  pass.setBindGroup(0,this.ownership.bindGroup);pass.setBindGroup(1,this.group.group);
  for(let i=0;i<launches.length;i++){
   const l=launches[i]!;
   if(l.when===1?!seeded:l.when===2?!(seeded&&activityBudgetOn):l.when===3&&!solids)continue;
   pass.setPipeline(uniformDetailPick(l.pipeline));
   if(l.y===0)pass.dispatchWorkgroups(l.x);else pass.dispatchWorkgroups(l.x,l.y);
  }
  // The certificate is seeded: the pages its seeds reach ask the pool, ahead of the frame's commit and of encodeResidency.
  pass.setPipeline(uniformDetailPick(request.pipeline));pass.setBindGroup(1,request.group);pass.dispatchWorkgroups(request.groups);
  pass.end();
  // The band now holds the joins: clear them for the next producers.
  if(this.joinLive)encoder.clearBuffer(this.joinTiles,0,(this.joinWords+1)*4);
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
 destroy():void{this.solidTiles.destroy();this.joinTiles.destroy();this.work.destroy();this.bounds.destroy();this.readback.destroy();this.params.destroy();}
}
