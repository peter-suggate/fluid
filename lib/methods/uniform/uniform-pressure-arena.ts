import { uniformComputePassEpoch } from "./uniform-compute-blit";
import { uniformDetailBindLayout, uniformDetailModule, uniformDetailPipeline, uniformDetailPick, uniformDetailGroup, type UniformDetailGroup } from "./uniform-detail-fields";
import { uniformBufferedWork } from "./uniform-buffered-work";
import type {UniformMixedOwnership} from "./uniform-mixed-ownership";
import {uniformMixedTopologyNeighboursWGSL,uniformMixedTopologyWGSL} from "./uniform-mixed-topology.wgsl";
import {uniformMixedFaceAddressWGSL} from "./uniform-mixed-face-dispatch.wgsl";
import {UNIFORM_MIXED_THETA_MIN,uniformMixedPressureLiquidWGSL,uniformMixedSurfaceThetaWGSL} from "./uniform-mixed-pressure-surface.wgsl";
import {uniformMixedSolidPipeline,uniformMixedSolidWGSL,type UniformMixedSolid} from "./uniform-mixed-solid.wgsl";
import {UNIFORM_BODY_FLUX_REACH,UNIFORM_REGION_DYNAMICS,UNIFORM_REGION_DYNAMICS_WORDS} from "./uniform-mixed-region-dynamics";
import {UNIFORM_COLD_FREEZE,UniformColdFreeze} from "./uniform-cold-freeze";
import type {UniformColdFrame} from "./uniform-cold-frame";
import {UNIFORM_REGION_TIER,uniformRegionHeatWGSL} from "./uniform-region-heat";
import {UNIFORM_ARENA_CLASS,UNIFORM_ARENA_RELEASE_ENTRY,UNIFORM_ARENA_TABLE_ENTRIES,UBA_STRIDE,UniformBrickArena,type UniformArenaChannel} from "./uniform-brick-arena";
import {UNIFORM_MIXED_FAILURE,uniformMixedFrameStatusWGSL} from "./uniform-mixed-frame-status";
import {UNIFORM_DETAIL_4H_LOAD} from "../../core/uniform-detail-abi";
import {UNIFORM_MIXED_ROOT_TILE_WORDS} from "./uniform-mixed-pressure-authority";
import {UNIFORM_BULK_FLOOR,UNIFORM_BULK_HEADS,UNIFORM_BULK_MEAN,UNIFORM_BULK_TOLERANCE,UNIFORM_BULK_WORDS} from "./uniform-bulk-tier-reference";
import {UNIFORM_PRESSURE_SUPER6_SIDE,planUniformPressureSuper,type UniformPressureSuperWords} from "./uniform-pressure-super";
import {UNIFORM_LIVE_GROUP_GROUPS,UNIFORM_LIVE_KEY_GROUPS,UNIFORM_LIVE_LANE_GROUPS} from "./uniform-live-lists";
import {uniformStageListsWGSL} from "./uniform-stage-lists";
import {BULK16_PAGE,BULK16_PAGE_WORDS,bulk16FaceIndexWGSL} from "./uniform-bulk-pages-reference";
import {BULK16_STATUS,type UniformBulkPages} from "./uniform-bulk-pages";

/** The composite pressure solve on the brick arena
 * (docs/plans/pressure-rfe-speed-of-light-2026-10-20.md §1.6, §1.9): one
 * system of the band's h rows and the native 4h rows of every other liquid
 * tile, V-cycles over the arena's levels. Level 0 is the band's h rows in
 * 8^3 bricks; level 1 (2h) is their MG level; level 2 (4h, a page a brick,
 * the root's own rows: key 512 block + cell) holds the native rows and the
 * band's aggregates; levels 3-5 (8h, 16h, 32h) are per page, and the 32h
 * rows of every page are solved by one workgroup. A band face whose tile
 * across is a native liquid row is a seam (K_SEAM): coupled to that 4h row
 * over 2.5h, V/(2.5 h^2), the native row taking the sum of its 16 h faces.
 * Level 0 smooths in place, red-black sweeps over the band's tiles (a
 * workgroup two tiles, a lane a red cell then a black one; a sweep reads
 * one iterate plane and writes the other, so neighbours in other tiles are
 * the last sweep's; neighbours across
 * a brick face through the brick's record). Levels 1 and 2 smooth with an
 * overlapping box: a brick stages its cells two deep around it, red-black
 * sweeps all but the outer ring and keeps its own 8^3. Each level's
 * residual is measured against the neighbour bricks' kept cores, never the
 * box's own view of them (which no up leg sees). At level 1 (pDownR) the box
 * view left paddleboats at 3.22 cycles, geo-mean final 0.028, against 3.13 and
 * 0.019, and letters' capped final at 0.040 against 0.034. Native rows take the
 * composite residual each cycle and their level 2 correction; the band's
 * levels sum their children's residuals. Prolongation is trilinear over the
 * parents that exist, renormalised. The aggregated operators are the
 * children's inter-aggregate links scaled by 0.5, with their Dirichlet row
 * sums preserved (7-point at every level), baked once
 * a frame by the level below. */
export interface UniformPressureArenaSchedule {
 /** V-cycles at most (NB-FLIP bandCycles). */
 cycles:number;
 /** Red-black sweeps of each level, before and after each coarse correction. */
 sweeps:number;
 /** Red-black sweeps of each of level 2's (the coarsest's) two box passes. */
 coarsestSweeps:number;
 /** The aggregated operator's scale (0.5: the rediscretised interior operator). */
 coarseScale:number;
 /** The 16h and 32h aggregates' scale: 0.5 there makes the coarse correction
  * overshoot (a CPU twin of the native levels diverges); 0.75 converges. */
 tailScale:number;
 /** V-cycles of the 8h level (8h, 16h, 32h) in each cycle, odd: the 8h
  * iterate alternates channels and level 2's up leg reads Y. One leaves the
  * natives at a twin's 0.5 a cycle; three at 0.15. */
 tailCycles:number;
 /** NB-FLIP pressureTail: the 32h rows' correction. "pages" solves them in
  * one workgroup (at most TAIL_ROWS pages); "superblocks" takes them through
  * 64h (2^3 pages) and 128h (4^3 pages, a row a super-block) and solves the
  * 128h rows in that workgroup; "auto" is "pages" while the pool fits it. */
 tail:"auto"|"pages"|"superblocks";
 /** The 64h and 128h aggregates' scale, as tailScale: 0.5 (rediscretised)
  * suits full groups but overshoots where a pool's faces leave part groups
  * (a CPU twin diverges on ragged pools); 0.75 contracts on every pool the
  * twin tries (tests/pressure-tail.test.ts). */
 superScale:number;
 /** NB-FLIP nativeSmoothing: red-black sweeps of the native rows against
  * the band's iterate after each cycle's pre-smoothing (0: level 2's
  * correction alone). Level 2 sees the band as its 4h aggregates, so a
  * native row coupled mostly through seams (a solid's pocket beside a
  * throat) keeps the residual of the band's real profile, and the band's
  * own sweeps answer it: a period-2 exchange no cycle count removes. */
 nativeSweeps:number;
 /** Flexible GCR(1) (Orthomin(1) with a variable preconditioner; Vuik & van
  * der Vorst 1994; Saad, Iterative Methods, 9.4) on the band solve: the
  * first cycle is the projected V-cycle, which settles the active set; the
  * set then freezes (Kornhuber's truncation, the cycle's own), and each
  * later cycle is the preconditioner of one step. The step takes the
  * cycle's correction z and A z (one operator apply), orthogonalises A z
  * against the last direction's image and moves the iterate by the
  * residual-minimising length (q.r)/(q.q). The coarse correction's length
  * was the band's limit: a fixed 1.25x cut the final residual about 3x on
  * seep and letters (BULK16 plan, band cap item). GCR rather than CG: on a
  * block with no Dirichlet row (a sealed full vessel) the operator is
  * singular, and CG's (p.r)/(p.Ap) divides by the block constant's
  * vanishing energy (tests/pressure-krylov.test.ts). */
 krylov:boolean;
}
/** A pressure at its lower bound leaves the coarse correction space until
 * its residual asks it to increase. Recomputed after each pre-smoothing. */
export const uniformPressureCorrectionWGSL=/* wgsl */`
fn pCorrectionFree(bounded:bool,total:f32,residual:f32)->bool{
 return !(bounded&&total<=0.0&&residual<=0.0);
}
`;
/** The correction prolongation preserves constants. Keep the aggregate's
 * Dirichlet row sum while rediscretising only its links to other unknowns.
 * Scaling the entire diagonal weakens free-surface constraints at every level
 * and overcorrects near-constant errors in weakly open liquid regions. */
export const uniformPressureAggregateWGSL=/* wgsl */`
fn pAggregateDiagonal(part:f32,outward:f32,scale:f32)->f32{
 return (part-outward)+scale*outward;
}
`;
export const UNIFORM_PRESSURE_ARENA_SCHEDULE:UniformPressureArenaSchedule={cycles:4,sweeps:3,coarsestSweeps:6,coarseScale:0.5,tailScale:0.75,tailCycles:3,tail:"auto",superScale:0.75,nativeSweeps:0,krylov:true};
/** bandExit "returns": the band's residual target (1/s, h-equivalent). The
 * band stops at it or at the cycle cap; the strict exit keeps the frame's
 * 1e-4 tolerance. 0.05 1/s is a 2.5 mm/s worst-cell velocity error at h 5 cm;
 * the 4h root accepts up to 5. */
export const UNIFORM_PRESSURE_ARENA_RETURNS_RESIDUAL=0.05;
/** The native rows form their own RHS from the simulation's u* (pRhs, ahead
 * of pNative) in place of the root buildRhs's: the all-4h mirror's
 * retirement, step 1 (plan §1.10). Off: the root's rows. */
export const UNIFORM_PRESSURE_ARENA_NATIVE_RHS:boolean=false;
/** Startup Picard iterations for the pressure-dependent seam heads. Three
 * re-solves left a depth-dependent hydrostatic transient; 23 keep the full
 * 1x..2.75x closed-rest ladder stationary to the volume readback precision.
 * They run once at t=0, with no position or clock advance (encodeSolve). */
export const UNIFORM_PRESSURE_KICK_RESOLVES=23;

/** Workgroups of the band-tile mark at most (the simulation's buffered h tiles). */
const LIST_GROUPS=4096;
/** Brick, tile and page launches stride their lists from at most this many groups. */
const BRICK_GROUPS=2048,TILE_GROUPS=4096,PAGE_GROUPS=1024,TABLE_GROUPS=1024;
/** The BULK16 solve's indirect launches (pLaunch, a workgroup an item, at
 * most PAGE_GROUPS; every reader strides its count): the live slots; the
 * level 2 pages or the live slots, the larger (a loop over both lists); the
 * two lists' sum (the tail's items); the level 2 pages; the live slots eight
 * a workgroup; the items 64 a workgroup. Byte offsets in the launch buffer. */
const LAUNCH={live:0,either:16,items:32,level2:48,live8:64,items64:80,bytes:96} as const;
type UniformArenaLaunch=number|"tiles"|{indirect:number};
/** Fine slots: a share of the lattice's bricks (at least ARENA_SLOTS_MIN). The
 * band's bricks hug the surface: the pour at 1/60 s peaks at 2310 of 8320
 * (0.28, frame 410), letters at 1228 of 12416; 0.45 holds the pour with 1.6x
 * headroom. A frame needing more is the fatal overflow (header word 1 =
 * SLOT_OVERFLOW), never growth. */
const ARENA_SLOT_SHARE=0.45,ARENA_SLOTS_MIN=2048,SLOT_OVERFLOW=3;
/** Header word 1 when a row read a stale solid aperture record (umPressureFaceVRecorded's -1): fatal. */
const STALE_APERTURES=4;
/** Brick kernels: a lane a cell of the brick (and of each colour of its box). */
const WG=512;
/** Level 0's tile kernels: a lane a cell of a band tile (half sweeps: of two tiles' cells of one colour). */
const TILE_WG=64;
const BAND_CYCLE_WINDOW=4;
const groupsOf=(count:number,per:number):number=>{const padded=count+per-1;return (padded-padded%per)/per;};
const PREPARE_PASS:GPUComputePassDescriptor={label:"Uniform pressure arena marks and rows"};
const UNMARK_PASS:GPUComputePassDescriptor={label:"Uniform pressure arena unmark"};
const SOLVE_PASS:GPUComputePassDescriptor={label:"Uniform pressure arena solve"};
const VERDICT_PASS:GPUComputePassDescriptor={label:"Uniform pressure arena verdict"};
const PROJECT_PASS:GPUComputePassDescriptor={label:"Uniform pressure arena projection"};
/** Header words: band tiles, overflow, final residual, live bricks, completed
 * cycles, reserved, level 1's and level 2's bricks
 * (the receipt's first 8 words end there), the residual after each cycle's
 * pre-smoothing (and after the last cycle) in HISTORY words, converged,
 * valid residual bits (history in bits 0..15, initial in bit 16), then the
 * initial residual. The tile slot+1 map (presentation slots), the
 * per-brick band-tile masks, the band tile list (by presentation slot:
 * brick slot << 3 | tile bit, from the rows) and the arena table follow. */
/** Diagnostics (UniformPressureArena.diagnostics): per measured cycle and row class, the
 * largest residual and the rows past the target, in DIAG_WORDS header words after HEADER. */
const DIAG_CLASSES=8,DIAG_CYCLES=8,DIAG_WORDS=2*DIAG_CLASSES*DIAG_CYCLES;
const HEADER=63,HISTORY_WORD=8,HISTORY=16,INITIAL_WORD=26,VALID_WORD=25,DONE_WORD=24,BRICKS_WORD=3,LEVEL1_WORD=6,LEVEL2_WORD=7;
/** Header words 27 and 28: the native rows' largest initial and final residual (diagnostics). */
const NATIVE_INITIAL_WORD=27,NATIVE_FINAL_WORD=28,NATIVE_HISTORY_WORD=29;
/** Header words 37..57, the Krylov steps' (schedule.krylov): the step's
 * alpha and beta, the residual 2-norm before and after each step (KRYLOV_L2
 * + cycle). Then the complementarity the final measure finds (every
 * schedule): bounded rows at their bound whose residual asks to leave it
 * (positive) past the target (KRYLOV_SIGN), and rows under their bound
 * (KRYLOV_BOUND). Then the solve's frozen rows that joined the free set
 * (KRYLOV_JOIN) and the steps that took a free bounded row under its bound
 * (KRYLOV_CROSS). Then the feasible step's: the step's first bound hit
 * along its direction (KRYLOV_LIMIT, a ratio's f32 bits), the rows a hit
 * put on their bound (KRYLOV_HIT), the restart a hit asks of the next step
 * (KRYLOV_RESTART) and the restarts taken (KRYLOV_RESTARTS). */
const KRYLOV_ALPHA_WORD=37,KRYLOV_BETA_WORD=38,KRYLOV_L2_WORD=39,KRYLOV_SIGN_WORD=54,KRYLOV_BOUND_WORD=55,KRYLOV_JOIN_WORD=56,KRYLOV_CROSS_WORD=57;
const KRYLOV_LIMIT_WORD=58,KRYLOV_HIT_WORD=59,KRYLOV_RESTART_WORD=60,KRYLOV_RESTARTS_WORD=61,KRYLOV_SPREAD_WORD=62;
/** The Krylov dots' per-workgroup partials: band tiles, native pages and BULK16 slots each a segment. */
const KRYLOV_PLANES=["x","q","z","p","r","d"] as const;
const KRYLOV_DOTS=6,KRYLOV_SEGMENTS=[TILE_GROUPS,PAGE_GROUPS,PAGE_GROUPS] as const,KRYLOV_PARTIALS=KRYLOV_SEGMENTS[0]+KRYLOV_SEGMENTS[1]+KRYLOV_SEGMENTS[2];
/** Header word 33: the tiles pRows routed to its geometry variant. */
const GEOMETRY_WORD=33;
/** Kinds bits beyond the band's: an active row (liquid, positive diagonal) and a warm one (its slot held a liquid row last frame). */
const ACTIVE_BIT=1<<27,WARM_BIT=1<<28;
/** The coarsest rows are one workgroup's: the 32h rows (a page each) of
 * the "pages" tail, the 128h rows (a super-block each) of "superblocks". */
const TAIL_ROWS=4096,TAIL_LANES=256,TAIL_SWEEPS=32;
/** pTail's lanes: enough that every page's row stays in a lane's registers
 * (a reloaded row walks its six neighbour records each half sweep). */
const tailLanes=(blocks:number,limit:number):number=>{let n=TAIL_LANES;while(n<blocks&&2*n<=limit)n*=2;return n;};

export interface UniformPressureArenaFields {
 phi:GPUBufferBinding;
 authorityParams:GPUBuffer;
 vertexPhi:GPUTexture;
 forced:{velocity:GPUTexture;negative:GPUBuffer};
 velocity:GPUTexture;negative:GPUBuffer;copy:GPUTexture;
 /** The all-4h root's rows (UniformMixedPressureVelocity.buildRhs: RHS,
  * bound and warm pressure at key 512 block + cell, the wall slots after
  * them), its phi and (solids) its topology record. The native rows read
  * them; the solve writes the native rows' (and the band tiles' mean)
  * pressure back for the 4h projection. */
 root:{rhs:GPUBufferBinding;minimum:GPUBufferBinding;pressure:GPUBufferBinding;phi:GPUBufferBinding;topology?:GPUBufferBinding;
  /** The 4h projection's gate word (umSlotClosed): the verdict opens it on a healthy frame. */
  support:GPUBuffer;supportWord:number};
 /** The frame status record (uniform-mixed-frame-status) and the acceptance state (8 words) the verdict writes. */
 status:GPUBuffer;acceptance:GPUBuffer;
 /** As the band's (80 bytes): h.xyz, dt; density, unused, dt/density, min
  * h; the residual target, then 1/h^2 per axis; four u32 the arena writes:
  * the presentation capacity, the native residual scale, spare, the
  * presentation word; rho g.xyz (the hydrostatic gradient, Pa/m), unused. */
 params:GPUBuffer;
 presentation:{buffer:GPUBuffer;word:number};
 /** Band tiles the presentation holds (more is the fatal overflow). */
 capacity:number;
 dynamics?:GPUBuffer;
 freeze?:UniformColdFreeze;
 coldFrame?:UniformColdFrame;
 warmTiers?:GPUBuffer;
 /** The 16h bulk tier's regions: the arena holds its record (UniformBulkTier:
  * member flags; the seam heads pNative writes for its face pass) after pool
  * B's words (bulkRecord), so pNative reads it through pool B and stays at
  * ten storage bindings. Absent with bulkTier off, which leaves every
  * module's text as it was. */
 bulk?:{regions:number};
 /** The BULK16 slot store (bulkTier "16h"). It moves into pool B after the
  * bulk record (UniformBulkPages.adopt: bulkStore), so every solve kernel
  * reads it through pool B. A demoted page's eight 16h rows are coarse-only
  * rows of the composite: the 4h rows across its faces couple to them
  * (N_BULKN), and they take part in levels 3 to 5 (16h cells; the 32h row
  * their page's), their corrections the coarse levels' (composite FAC,
  * docs/plans/bulk16-plan-2026-10-28.md M2 step 3, A1). */
 bulkPages?:UniformBulkPages;
 /** The most h tiles the simulation ever holds (the frame's ceiling): pMark
  * gives a fine slot only to a brick holding a band h tile, and frames
  * hold 4.4 to 22 h tiles for each such brick (splash L128, letters, seep
  * at 767fc07f), so a quarter of this many fine slots holds the band (the
  * overflow stays fatal). Absent: the lattice share alone. */
 fineTiles?:number;
}

/** The two-stage solve's h band (the band's contract: tiles, rows, faces,
 * start and projection are UniformPressureBand's, now on persistent arena
 * bricks). Band tiles mark their 8^3 bricks with the fine bit; a brick keeps
 * its slot while it stays marked, so the last frame's pressure of a row
 * that was liquid in the same slot starts it (warm), and the 4h trilinear
 * start does elsewhere. */
export class UniformPressureArena {
 readonly arena:UniformBrickArena;
 /** Header, tile slot map, brick masks and the arena table (one storage binding). */
 readonly index:GPUBuffer;
 /** The channels in two storage bindings, each under the binding limit: A
  * (9) level 0's iterates, rows and coefficients; B (10) level 0's forced
  * velocities and apertures, levels 1 and 2 and the super-block tail. A
  * channel's pool is fixed, so every access names its binding. */
 private readonly pool:GPUBuffer;
 private readonly poolB:GPUBuffer;
 /** The 16h bulk tier's record, pool B's last words (fields.bulk). */
 readonly bulkRecord?:{buffer:GPUBuffer;offset:number;size:number};
 private readonly bulkWord:number=0;
 /** The BULK16 store's range of pool B (fields.bulkPages), for its adopt. */
 readonly bulkStore?:GPUBufferBinding;
 private readonly storeWord:number=0;
 /** Pool B's words of the Krylov dots' partials and of the BULK16 rows' Krylov vectors (schedule.krylov). */
 private readonly krylovDotWord:number=0;
 private readonly krylovBulkWord:number=0;
 /** The BULK16 solve's launches (LAUNCH), written by pLaunch from the
  * counts: its own buffer, as an indirect buffer is no storage binding of
  * the dispatch that reads it. */
 private readonly launch?:GPUBuffer;
 private launcher?:{pipeline:GPUComputePipeline;group:GPUBindGroup};
 get allocatedBytes():number{return this.index.size+this.pool.size+this.poolB.size+(this.launch?.size??0);}
 private readonly tiles:number;
 private readonly maskWord:number;
 private readonly tileListWord:number;
 /** The tiles pRows leaves to its geometry variant (their tile list indices; the count in header word GEOMETRY_WORD). */
 private readonly geometryListWord:number;
 private readonly tableWord:number;
 /** The tile slot map and brick masks hold only the last frame's band
  * (pUnmark clears them through its list): false until the first frame's
  * whole clear, and again after a resize (a new band capacity bounds the
  * list differently). */
 private indexClean=false;
 /** Level 2's band channels (pool words first.., a level 2 slot a block):
  * pClear2 zeroes the held blocks' slots each frame, a workgroup a (block,
  * channel). A free block's slot is no live neighbour's (its records' words
  * are UB_NONE or a seam), so no stage reads it. */
 private readonly level2Clear:{first:number;channels:number};
 private slots=0;
 get capacity():number{return this.slots;}
 /** The 32h rows' correction goes through the super-block levels (schedule.tail). */
 readonly superTail:boolean;
 private readonly super?:UniformPressureSuperWords;
 get slotMapOffset():number{return 4*this.header;}
 /** Header words: HEADER, and with diagnostics their DIAG_WORDS. */
 readonly header:number;
 private workBricks=1;private workLevel1=1;private level1Groups=1;private workLevel2=1;private workHeld=1;private pageGroups=PAGE_GROUPS;
 private workTiles=1;private tileGroups=1;private relaxGroups=1;
 private readonly recentCycles=new Uint32Array(BAND_CYCLE_WINDOW);
 private recentCount=0;
 private wideCycles=Number.POSITIVE_INFINITY;
 private readonly layouts=new Map<string,GPUBindGroupLayout>();
 private readonly groups=new Map<string,UniformDetailGroup>();
 private readonly pipelines=new Map<string,GPUComputePipeline>();
 private readonly launches=new Map<string,{pipeline:GPUComputePipeline;group:UniformDetailGroup}>();
 private readonly restricts:readonly string[];
 private readonly residuals:readonly string[];
 private boundPass?:GPUComputePassEncoder;
 private boundEpoch=-1;
 private boundGroup?:UniformDetailGroup;
 /** encodePrepare's verdict, which encodeSolve keeps. */
 /** The root's one binding (26) and its fields' word offsets in it. */
 private readonly rootSpan:{binding:GPUBufferBinding;pressure:number;rhs:number;minimum:number};
 private get cold():UniformColdFreeze|undefined{return UNIFORM_COLD_FREEZE?this.fields.freeze:undefined;}
 constructor(private readonly device:GPUDevice,private readonly simulation:UniformMixedOwnership,private readonly pressure:UniformMixedOwnership,private fields:UniformPressureArenaFields,
  readonly schedule:UniformPressureArenaSchedule=UNIFORM_PRESSURE_ARENA_SCHEDULE,private readonly solid?:UniformMixedSolid){
  this.header=HEADER+(UniformPressureArena.diagnostics?DIAG_WORDS:0);
  if(pressure.layout.tiles.some(word=>(word&0xc0000000)!==0))throw new Error("The pressure arena's global stage must be all-4h");
  if(solid&&!solid.coarse)throw new Error("The solid pressure arena needs the all-4h solid record");
  if(!Number.isInteger(schedule.cycles)||schedule.cycles<1||schedule.cycles>=HISTORY)throw new Error(`Pressure arena cycles must be 1..${HISTORY-1}`);
  const address=simulation.capacity.address,blocks=address.blocks;
  // The coarsest rows: a 32h row a page, a pool block's or a BULK16 slot's.
  // Up to TAIL_ROWS of them the page tail (pTail, one workgroup) is the
  // cheaper; past it auto takes the super-block tail, which holds BULK16
  // rows too.
  if(fields.bulkPages&&!fields.bulk)throw new Error("Pressure arena: BULK16 pages need the 16h bulk tier");
  const tailRows=blocks+(fields.bulkPages?.layout.capacity??0);
  this.superTail=schedule.tail==="superblocks"||(schedule.tail==="auto"&&tailRows>TAIL_ROWS);
  if(!this.superTail&&tailRows>TAIL_ROWS)throw new Error(`Pressure arena: ${blocks} pages and ${fields.bulkPages?.layout.capacity??0} BULK16 slots, over the page tail's ${TAIL_ROWS} rows; use the super-block tail`);
  // The super-block tail keys a group by its page coordinate from the box's
  // low page mod the window's 64 pages (sRel): a box past 64 pages an axis
  // would fold two groups onto one key. 64 pages a side is 1024 h (5.12 m
  // at h = 5 mm, 10.24 m at 1 cm).
  if(this.superTail){
   const box=address.pool.box.pages,side=2*UNIFORM_PRESSURE_SUPER6_SIDE;
   if(box.some(n=>n>side))throw new Error(`Pressure arena: the super-block tail's key window holds ${side} pages an axis (${side*16} h), the box ${box.join("x")} pages`);
  }
  if(!(schedule.superScale>0))throw new Error("Pressure arena: the super-block tail's scale must be positive");
  if(!Number.isInteger(schedule.nativeSweeps)||schedule.nativeSweeps<0)throw new Error(`Pressure arena: ${schedule.nativeSweeps} native sweeps`);
  if(schedule.tailCycles%2!==1)throw new Error(`Pressure arena: ${schedule.tailCycles} 8h cycles; level 2's up leg reads the odd cycles' channel`);
  if(pressure.capacity.address.blocks!==blocks)throw new Error("Pressure arena: the root's blocks are the simulation's");
  const rootWords=(view:GPUBufferBinding)=>(view.size??view.buffer.size-(view.offset??0))/4;
  if(rootWords(fields.root.rhs)<896*blocks||rootWords(fields.root.minimum)<896*blocks||rootWords(fields.root.pressure)<896*blocks)throw new Error("Pressure arena: the root's rows and wall slots do not fit its views");
  if(solid&&!fields.root.topology)throw new Error("Pressure arena: solids need the root's topology record");
  // The root's pressure, RHS and bound share one buffer: one binding holds
  // them (one writable range; separate bindings of one buffer conflict).
  {const r=fields.root,views=[r.pressure,r.rhs,r.minimum];
   if(views.some(v=>v.buffer!==r.pressure.buffer))throw new Error("Pressure arena: the root's pressure, RHS and bound must share one buffer");
   const align=device.limits.minStorageBufferOffsetAlignment,start=Math.min(...views.map(v=>v.offset??0)),base=start-start%align;
   const end=Math.max(...views.map(v=>(v.offset??0)+(v.size??v.buffer.size-(v.offset??0))));
   const word=(v:GPUBufferBinding)=>((v.offset??0)-base)/4;
   this.rootSpan={binding:{buffer:r.pressure.buffer,offset:base,size:end-base},pressure:word(r.pressure),rhs:word(r.rhs),minimum:word(r.minimum)};}
  this.tiles=address.tileKeys;
  const cell=(name:string,level:0|1|2,pool=level?1:0):UniformArenaChannel=>({name,levels:[level],layout:"cell",pool});
  const six=(prefix:string,level:0|1|2,pool=level?1:0)=>Array.from({length:6},(_,f)=>cell(`${prefix}${f}`,level,pool));
  // Level 2's band channels contiguous: one clear empties them each frame.
  // The native rows' channels (n*) are written whole by pNative; the tail's
  // (8h at word 0, 16h at 64, 32h at 72 of a page) by pDownB's bake.
  const channels=[...["x","x1","diag","base","warm","rhs","kinds"].map(name=>cell(name,0)),...six("c",0),...six("u",0,1),...(solid?six("v",0,1):[]),
   ...["xa","ya","ra","da"].map(name=>cell(name,1)),...six("ka",1),
   ...["xb","yb","rb","db"].map(name=>cell(name,2)),...six("kb",2),
   ...["nx","nb","nr","nd","nm"].map(name=>cell(name,2)),...six("nk",2),...six("nl",2),
   ...["xt","yt","rt","dt"].map(name=>cell(name,2)),...six("kt",2),
   // The Krylov step's vectors (schedule.krylov): the step's start, the
   // direction's image, A z, the direction, the residual, the frozen
   // diagonal (0 off the free set), for the band's rows and the native rows.
   ...(schedule.krylov?[...KRYLOV_PLANES.map(name=>cell(`k${name}`,0)),...KRYLOV_PLANES.map(name=>cell(`nk${name}`,2))]:[])];
  const bricks=64*blocks,fineCapacity=Math.max(1,Math.min(bricks,Math.max(ARENA_SLOTS_MIN,Math.ceil(bricks*ARENA_SLOT_SHARE)),Math.ceil((fields.fineTiles??4*bricks)/4)));
  this.arena=new UniformBrickArena({blocks,channels,lists:[{name:"fine",any:UNIFORM_ARENA_CLASS.fine}],level1From:"fine",fineCapacity});
  const a=this.arena;
  for(const [prefix,level] of [["c",0],["u",0],...(solid?[["v",0]]:[]),["ka",1],["kb",2],["nk",2],["nl",2],["kt",2]] as [string,0|1|2][]){
   for(let f=0;f<6;f++)if(a.base(`${prefix}${f}`,level)!==a.base(`${prefix}0`,level)+f*UBA_STRIDE*a.capacity(level))throw new Error(`Pressure arena: channel ${prefix} is not contiguous`);
  }
  const first=a.base("xb",2),last=a.base("kb5",2)+UBA_STRIDE*blocks;
  const level2=(last-first)/(UBA_STRIDE*blocks);
  if(!Number.isInteger(level2))throw new Error("Pressure arena: level 2's band channels are not whole slots");
  this.level2Clear={first,channels:level2};
  this.maskWord=this.header+this.tiles;this.tileListWord=this.maskWord+a.bricks;this.geometryListWord=this.tileListWord+this.tiles;this.tableWord=this.geometryListWord+this.tiles;
  const limit=device.limits.maxStorageBufferBindingSize;
  // The super-block tail's lists and maps follow the arena table; its rows pool B's channels.
  if(a.poolWords.length!==2)throw new Error("Pressure arena: the channels fill pools A and B");
  // Its 64h and 128h rows' capacities: a row a group holding a tail item
  // (a page block or a BULK16 slot), at most one a key of the window.
  this.super=this.superTail?planUniformPressureSuper(blocks+(fields.bulkPages?.layout.capacity??0),this.tableWord+a.tableWords,a.poolWords[1]!):undefined;
  const indexWords=this.super?.index.end??this.tableWord+a.tableWords,poolWords=[a.poolWords[0]!,this.super?.pool.end??a.poolWords[1]!];
  // The bulk record follows pool B's words at a storage offset.
  const align=device.limits.minStorageBufferOffsetAlignment/4;
  if(fields.bulk){this.bulkWord=Math.ceil(poolWords[1]!/align)*align;poolWords[1]=this.bulkWord+UNIFORM_BULK_WORDS(fields.bulk.regions);}
  if(fields.bulkPages){
   this.storeWord=Math.ceil(poolWords[1]!/align)*align;poolWords[1]=this.storeWord+fields.bulkPages.layout.words;
   // The BULK16 rows' Krylov vectors: a slot's eight regions, a plane each.
   if(schedule.krylov){this.krylovBulkWord=poolWords[1]!;poolWords[1]=this.krylovBulkWord+8*KRYLOV_PLANES.length*fields.bulkPages.layout.capacity;}
   // A BULK16 address names a tail channel by its plane (baWord).
   const xt=a.base("xt",2);
   for(const [j,name] of ["xt","yt","rt","dt","kt0","kt1","kt2","kt3","kt4","kt5"].entries())if(a.base(name,2)!==xt+j*UBA_STRIDE*blocks)throw new Error("Pressure arena: the tail channels are not contiguous");
  }
  if(schedule.krylov){this.krylovDotWord=poolWords[1]!;poolWords[1]=this.krylovDotWord+KRYLOV_DOTS*KRYLOV_PARTIALS;}
  for(const [k,words] of poolWords.entries())if(4*words>Math.min(limit,device.limits.maxBufferSize))throw new Error(`Pressure arena pool ${"AB"[k]} (${4*words} bytes) exceeds the storage binding limit ${Math.min(limit,device.limits.maxBufferSize)}`);
  this.index=device.createBuffer({label:"Uniform pressure arena state",size:4*indexWords,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC});
  this.pool=device.createBuffer({label:"Uniform pressure arena pool",size:4*poolWords[0]!,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC});
  this.poolB=device.createBuffer({label:"Uniform pressure arena pool B",size:4*poolWords[1]!,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC});
  if(fields.bulk)this.bulkRecord={buffer:this.poolB,offset:4*this.bulkWord,size:4*UNIFORM_BULK_WORDS(fields.bulk.regions)};
  if(fields.bulkPages)this.bulkStore={buffer:this.poolB,offset:4*this.storeWord,size:fields.bulkPages.layout.bytes};
  if(fields.bulkPages)this.launch=device.createBuffer({label:"Uniform pressure arena BULK16 launches",size:LAUNCH.bytes,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.INDIRECT});
  for(const w of a.initialWrites(this.tableWord))device.queue.writeBuffer(this.index,w.offset,w.data as Uint32Array<ArrayBuffer>);
  this.restricts=Array.from({length:schedule.cycles},(_,k)=>`pRestrict0@${k}`);
  this.residuals=Array.from({length:schedule.cycles},(_,k)=>`pResidual2@${k}`);
  const texture=(binding:number)=>({binding,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float" as const,viewDimension:"3d" as const}});
  const buffer=(binding:number,type:GPUBufferBindingType="storage")=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type}});
  const write=(binding:number)=>({binding,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"write-only" as const,format:"rgba32float" as const,viewDimension:"3d" as const}});
  const params=buffer(0,"uniform"),cold=UNIFORM_COLD_FREEZE&&fields.freeze?[{binding:14,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float" as const,viewDimension:"2d" as const}}]:[];
  const entries:Record<string,GPUBindGroupLayoutEntry[]>={
   table:[buffer(2)],
   mark:[params,buffer(2),buffer(3,"read-only-storage"),texture(6),...cold,...(fields.warmTiers?[buffer(25,"read-only-storage")]:[])],
   rows:[params,buffer(2),buffer(9),buffer(10),buffer(3,"read-only-storage"),buffer(4,"uniform"),texture(6),texture(7),buffer(8,"read-only-storage"),buffer(29,"read-only-storage"),...cold],
   init:[params,buffer(2),buffer(9),buffer(10),buffer(12,"read-only-storage"),buffer(13,"read-only-storage"),...cold],
   copy:[params,buffer(2),buffer(9),buffer(10),texture(11),write(23)],
   native:[params,buffer(2),buffer(9),buffer(10),buffer(26),buffer(29,"read-only-storage"),...(solid?[buffer(30,"read-only-storage")]:[])],
   ...(UNIFORM_PRESSURE_ARENA_NATIVE_RHS?{nrhs:[params,buffer(2),buffer(4,"uniform"),texture(7),buffer(8,"read-only-storage"),buffer(26),buffer(29,"read-only-storage"),...cold]}:{}),
   solver:[params,buffer(2),buffer(9),buffer(10),...(fields.dynamics?[buffer(24)]:[])],
   project:[params,buffer(2),buffer(9),buffer(10),texture(7),texture(15),write(16),buffer(17),buffer(21),...cold],
   // The BULK16 rows (pBulkRows: the 4h wall slots' flux and the volume
   // control's rate) and their projection.
   ...(fields.bulkPages?{bulk:[params,buffer(2),buffer(9),buffer(10),buffer(26),buffer(4,"uniform")]}:{}),
  };
  for(const [name,list] of Object.entries(entries))this.layouts.set(name,uniformDetailBindLayout(device,{label:`Uniform pressure arena ${name}`,entries:list}));
  this.bind();
 }
 /** Band tiles the presentation holds: the params' capacity word. */
 private bind():void{
  const f=this.fields,c=this.simulation.capacity,device=this.device;
  if(!Number.isInteger(f.capacity)||f.capacity<1||f.capacity>c.tiles)throw new Error(`The pressure arena cannot present ${f.capacity} of ${c.tiles} tiles`);
  this.slots=f.capacity;
  if(f.presentation.buffer.size<4*(f.presentation.word+64*f.capacity))throw new Error("The stage grids cannot hold the pressure band");
  device.queue.writeBuffer(f.params,48,new Uint32Array([f.capacity,0,0,f.presentation.word]));
  const cells=c.owners,faces=c.address.lowFaceKeys,coarse=this.pressure.capacity.owners;
  const scalars=(view:GPUBufferBinding,count:number):GPUBufferBinding=>{
   const offset=view.offset??0;if((view.size??view.buffer.size-offset)<4*count)throw new Error("Pressure arena field view is too small");return {buffer:view.buffer,offset,size:4*count};
  };
  const P={0:{buffer:f.params,size:80}},S={2:{buffer:this.index}},Q={9:{buffer:this.pool},10:{buffer:this.poolB}};
  const resources:Record<string,Record<number,GPUBindingResource|GPUTexture>>={
   table:{...S},
   mark:{...P,...S,3:scalars(f.phi,cells),6:f.vertexPhi},
   rows:{...P,...S,...Q,3:scalars(f.phi,cells),4:{buffer:f.authorityParams,size:16},6:f.vertexPhi,7:f.forced.velocity,8:{buffer:f.forced.negative,size:4*faces},29:scalars(f.root.phi,coarse)},
   init:{...P,...S,...Q,12:scalars(f.root.pressure,coarse),13:scalars(f.root.phi,coarse)},
   copy:{...P,...S,...Q,11:f.velocity,23:f.copy},
   native:{...P,...S,...Q,26:this.rootSpan.binding,29:scalars(f.root.phi,coarse)},
   // The root's cut face masks follow its owners' phi words.
   ...(UNIFORM_PRESSURE_ARENA_NATIVE_RHS?{nrhs:{...P,...S,4:{buffer:f.authorityParams,size:16},7:f.forced.velocity,8:{buffer:f.forced.negative,size:4*faces},26:this.rootSpan.binding,29:f.root.phi}}:{}),
   solver:{...P,...S,...Q},
   project:{...P,...S,...Q,7:f.forced.velocity,15:f.copy,16:f.velocity,17:{buffer:f.negative,size:4*faces},21:{buffer:f.presentation.buffer}},
   ...(f.bulkPages?{bulk:{...P,...S,...Q,26:this.rootSpan.binding,4:{buffer:f.authorityParams,size:16}}}:{}),
  };
  if(f.dynamics)resources.solver![24]={buffer:f.dynamics};
  if(f.root.topology)resources.native![30]=f.root.topology;
  const freeze=this.cold;if(freeze)for(const name of ["mark","rows","init","project",...(UNIFORM_PRESSURE_ARENA_NATIVE_RHS?["nrhs"]:[])])resources[name]![14]=freeze.regions;
  if(f.warmTiers)resources.mark![25]={buffer:f.warmTiers};
  for(const [name,map] of Object.entries(resources))this.groups.set(name,uniformDetailGroup(device,{label:`Uniform pressure arena ${name}`,layout:this.layouts.get(name)!,
   entries:Object.entries(map).map(([binding,resource])=>({binding:+binding,resource}))}));
  this.bindLaunches();this.boundPass=undefined;this.boundGroup=undefined;
 }
 /** Between frames, after the h-tile capacity (the phi view, the stage grids) changed: rebind. The arena's own buffers are lattice-sized. */
 resize(fields:Pick<UniformPressureArenaFields,"phi"|"presentation"|"capacity">):void{this.fields={...this.fields,...fields};this.bind();this.indexClean=false;}
 /** Fixed-lag evidence: band tiles, live bricks, level 1's and 2's bricks
  * and the held blocks (the composite level 2's pages: every held block) of
  * a receipted frame. Launch width only; no evidence of the held blocks
  * takes them all. reserve counts h tiles (UNIFORM_WORK_RELAYOUT_RESERVE):
  * the block widths take the blocks those tiles fill, 512 a block, not that
  * many blocks (which held every launch at the pool). Bricks and level 1
  * likewise: 8 and 64 h tiles a slot (a reserve of 1024 h tiles held
  * level 1's launch at 1,024 workgroups of 512 lanes for ~140 bricks). */
 observeWork(tiles:number,reserve=0,bricks=tiles,level1=bricks,level2=level1,held=this.simulation.capacity.address.blocks):void{
  const cap=this.arena.fineCapacity,blocks=this.simulation.capacity.address.blocks,blockReserve=Math.ceil(reserve/512);
  this.workBricks=uniformBufferedWork(this.workBricks,Math.min(bricks,tiles),cap,Math.ceil(reserve/8));
  this.workLevel1=uniformBufferedWork(this.workLevel1,Math.min(level1,this.workBricks),cap,Math.ceil(reserve/64));
  this.workLevel2=uniformBufferedWork(this.workLevel2,Math.min(level2,this.workLevel1,blocks),blocks,blockReserve);
  this.level1Groups=Math.min(this.workLevel1,BRICK_GROUPS);
  this.workHeld=uniformBufferedWork(this.workHeld,Math.min(held,blocks),blocks,blockReserve);this.pageGroups=Math.min(this.workHeld,PAGE_GROUPS);
  this.workTiles=uniformBufferedWork(this.workTiles,Math.min(tiles,this.slots),this.slots,reserve);
  this.tileGroups=Math.min(this.workTiles,TILE_GROUPS);this.relaxGroups=Math.min(groupsOf(this.workTiles,2),TILE_GROUPS);
 }
 observeCycles(cycles:number):void{
  this.recentCycles[this.recentCount%BAND_CYCLE_WINDOW]=cycles;this.recentCount++;
  if(this.recentCount<BAND_CYCLE_WINDOW){this.wideCycles=Number.POSITIVE_INFINITY;return;}
  let most=0;for(let i=0;i<BAND_CYCLE_WINDOW;i++)most=Math.max(most,this.recentCycles[i]!);
  this.wideCycles=most;
 }
 forgetCycles():void{this.recentCount=0;this.wideCycles=Number.POSITIVE_INFINITY;}
 /** Bytes of the arena's buffers (lattice-sized: no capacity changes them). */
 get bytes():{state:number;pool:number}{return {state:this.index.size,pool:this.pool.size+this.poolB.size};}

 async initialize():Promise<void>{
  const layout=this.simulation.layout,S=!!this.solid,schedule=this.schedule,a=this.arena,B=this.fields.bulk,BP=this.fields.bulkPages;
  const cap=a.fineCapacity,blocks=this.simulation.capacity.address.blocks;
  // The tail's rows are the page list's and the BULK16 pages' (tItem): its lanes are sized for both.
  const plane0=520*cap,plane2=520*blocks,lanesTail=tailLanes(blocks+(BP?.layout.capacity??0),this.device.limits.maxComputeInvocationsPerWorkgroup);
  if(lanesTail>this.device.limits.maxComputeInvocationsPerWorkgroup)throw new Error(`Pressure arena: the tail's ${lanesTail} lanes are past the device's ${this.device.limits.maxComputeInvocationsPerWorkgroup} invocations a workgroup`);
  const C=(name:string,level:0|1|2=0)=>a.base(name,level);
  const tableWGSL=uniformStageListsWGSL+a.tableWGSL("state",`${this.tableWord}u`,`atomicLoad(&state[MASK+id])`);
  const constants=/* wgsl */`
const SLOTS:u32=${this.header}u;const MASK:u32=${this.maskWord}u;const TLIST:u32=${this.tileListWord}u;const GLIST:u32=${this.geometryListWord}u;const GCOUNT:u32=${GEOMETRY_WORD}u;const HISTORY:u32=${HISTORY_WORD}u;const CYCLE:u32=4u;const DONE:u32=${DONE_WORD}u;const BRICKS:u32=${BRICKS_WORD}u;
const INITIAL:u32=${INITIAL_WORD}u;const VALID:u32=${VALID_WORD}u;const INITIAL_BIT:u32=1u<<${HISTORY}u;const CYCLES:u32=${schedule.cycles}u;
const K_GHOST:u32=0u;const K_BAND:u32=1u;const K_WALL:u32=2u;const K_OPEN:u32=3u;const K_NEUMANN:u32=4u;const K_CLOSED:u32=5u;
const P_ACTIVE:u32=${ACTIVE_BIT}u;const P_WARM:u32=${WARM_BIT}u;const P_FINE:u32=${UNIFORM_ARENA_CLASS.fine}u;
const P_X:u32=${C("x")}u;const P_X1:u32=${C("x1")}u;const P_DIAG:u32=${C("diag")}u;const P_BASE:u32=${C("base")}u;const P_WARMP:u32=${C("warm")}u;const P_RHS:u32=${C("rhs")}u;const P_KINDS:u32=${C("kinds")}u;
const P_C:u32=${C("c0")}u;const P_U:u32=${C("u0")}u;${S?`const P_V:u32=${C("v0")}u;`:""}const PLANE:u32=${plane0}u;
const PA_X:u32=${C("xa",1)}u;const PA_Y:u32=${C("ya",1)}u;const PA_R:u32=${C("ra",1)}u;const PA_D:u32=${C("da",1)}u;const PA_K:u32=${C("ka0",1)}u;const PLANE1:u32=${520*a.capacity(1)}u;
const PB_X:u32=${C("xb",2)}u;const PB_Y:u32=${C("yb",2)}u;const PB_R:u32=${C("rb",2)}u;const PB_D:u32=${C("db",2)}u;const PB_K:u32=${C("kb0",2)}u;const PLANE2:u32=${plane2}u;
const P_SCALE:f32=${schedule.coarseScale};const T_SCALE:f32=${schedule.tailScale};
${uniformPressureAggregateWGSL}
${uniformPressureCorrectionWGSL}
const P_CORRECT:u32=1u<<24u;const N_CORRECT:u32=1u<<23u;
const K_SEAM:u32=6u;
// The composite's seam coupling on the coarse levels, its one definition. A
// seam's native row stays 2.5h away at every level, so its coupling k never
// coarsens: (the diagonal part, the cross coupling) a child adds before the
// P_SCALE sum leaves k unscaled in both.
fn seamCoarse(k:f32)->vec2f{return vec2f(k*(1.0/P_SCALE-1.0),k/P_SCALE);}
const PN_X:u32=${C("nx",2)}u;const PN_B:u32=${C("nb",2)}u;const PN_R:u32=${C("nr",2)}u;const PN_D:u32=${C("nd",2)}u;const PN_M:u32=${C("nm",2)}u;const PN_K:u32=${C("nk0",2)}u;const PN_L:u32=${C("nl0",2)}u;
const PT_X:u32=${C("xt",2)}u;const PT_Y:u32=${C("yt",2)}u;const PT_R:u32=${C("rt",2)}u;const PT_D:u32=${C("dt",2)}u;const PT_K:u32=${C("kt0",2)}u;
${schedule.krylov?`const P_KX:u32=${C("kx")}u;const P_KQ:u32=${C("kq")}u;const P_KZ:u32=${C("kz")}u;const P_KP:u32=${C("kp")}u;const P_KR:u32=${C("kr")}u;const P_KD:u32=${C("kd")}u;
const PN_KX:u32=${C("nkx",2)}u;const PN_KQ:u32=${C("nkq",2)}u;const PN_KZ:u32=${C("nkz",2)}u;const PN_KP:u32=${C("nkp",2)}u;const PN_KR:u32=${C("nkr",2)}u;const PN_KD:u32=${C("nkd",2)}u;
`:""}const N_KEYS:u32=${512*blocks}u;const N_THETA:u32=${896*blocks}u;const N_SENTINEL:u32=${blocks-1}u;
const N_NONE:u32=0u;const N_NATIVE:u32=1u;const N_SEAM:u32=2u;const N_GHOST:u32=3u;const N_WALL:u32=4u;const N_HELD:u32=5u;
const N_ACTIVE:u32=${1<<20}u;const N_CLOSED:u32=${1<<21}u;
const NATIVE_INITIAL:u32=${NATIVE_INITIAL_WORD}u;const NATIVE_FINAL:u32=${NATIVE_FINAL_WORD}u;const NATIVE_HISTORY:u32=${NATIVE_HISTORY_WORD}u;
// Native and band rows are one coupled system. A native-only early exit
// removes rows from the aggregate while their neighbours retain links to
// them, breaking reciprocity. Only the composite residual may stop it.
// The root's acceptance, band-scaled: its target, and a tenth of the native
// rows' initial residual down to an absolute floor (the old root's bound).
fn nFloor()->f32{return 1e-4*bitcast<f32>(params.capacity.y);}
fn nBound()->f32{return min(params.solve.x,max(0.1*bitcast<f32>(pIndex(NATIVE_INITIAL)),nFloor()));}
// A native residual on the band's threshold: met there exactly when it meets nBound.
fn nStopScale()->f32{let b=nBound();return select(1.0,params.solve.x/b,b>0.0&&b<params.solve.x);}
`;
  const params=/* wgsl */`
struct BandParams {hDt:vec4f,policy:vec4f,solve:vec4f,capacity:vec4u,gravity:vec4f}
@group(1) @binding(0) var<uniform> params:BandParams;
`;
  const helpers=/* wgsl */`
fn pIndex(i:u32)->u32{return atomicLoad(&state[i]);}
// Diagnostics: a measured row's class and band-equivalent magnitude (UniformPressureArena.diagnostics).
fn pDiag(cycle:u32,cls:u32,mag:u32){${UniformPressureArena.diagnostics?`let w=${HEADER}u+${DIAG_CLASSES}u*min(cycle,${DIAG_CYCLES-1}u)+cls;atomicMax(&state[w],mag);if(bitcast<f32>(mag)>params.solve.x){atomicAdd(&state[w+${DIAG_CLASSES*DIAG_CYCLES}u],1u);}`:""}}
fn pDiagClass(kinds:u32)->u32{var c=select(0u,3u,bInside(kinds));for(var f=0u;f<6u;f++){let k=bFaceKind(kinds,f);if(k==K_WALL){return 2u;}if(k==K_SEAM){c=1u;}if((k==K_OPEN||k==K_GHOST||(k==K_BAND&&!bUnit(kinds,f)))&&c==0u){c=7u;}}if(!bLiquid(kinds)&&c==0u){c=3u;}return c;}
fn bLocal(lane:u32)->vec3u{return vec3u(lane%4u,(lane/4u)%4u,lane/16u);}
fn bUnit(kinds:u32,f:u32)->bool{return (kinds&(1u<<(18u+f)))!=0u;}
fn bFaceKind(kinds:u32,f:u32)->u32{return (kinds>>(3u*f))&7u;}
fn bLiquid(kinds:u32)->bool{return (kinds&0x80000000u)!=0u;}
fn bInside(kinds:u32)->bool{return (kinds&0x40000000u)!=0u;}
fn bShut(kinds:u32)->bool{return (kinds&0x20000000u)!=0u;}
fn pActive(kinds:u32)->bool{return (kinds&P_ACTIVE)!=0u;}
fn bSign(f:u32)->i32{return select(-1,1,(f&1u)==1u);}
// Positive float bits have integer order; every nonfinite residual is infinity.
fn bResidualMagnitude(r:f32)->u32{return min(bitcast<u32>(r)&0x7fffffffu,0x7f800000u);}
// The tile of brick-local cell l (its bit in the brick's mask).
fn pTileBit(l:vec3u)->u32{let t=(l>>vec3u(2u))&vec3u(1u);return t.x|(t.y<<1u)|(t.z<<2u);}
// A band tile: its presentation slot + 1, 0 outside the band.
fn pBandCell(q:vec3i)->bool{if(!umCellPresent(q)){return false;}return pIndex(SLOTS+umTileAt(vec3u(q)/4u))!=0u;}
// Only completed measurements are read (a dispatch never tests the word it reduces).
fn bMet(word:u32,mask:u32)->bool{return (pIndex(VALID)&mask)!=0u&&params.solve.x>0.0&&bitcast<f32>(pIndex(word))<=params.solve.x;}
fn bInitialMet()->bool{return bMet(INITIAL,INITIAL_BIT);}
fn bStop(j:u32)->bool{return bMet(HISTORY+j,1u<<j);}
fn bConvergedBefore(k:u32)->bool{if(bInitialMet()){return true;}for(var j=0u;j<min(k,${HISTORY-1}u);j++){if(bStop(j)){return true;}}return false;}
fn bConverged()->bool{return bConvergedBefore(CYCLES);}
// The fine list (the rows precede the verdict), and the solve's: none when closed.
fn pFine()->u32{return ubaListCount(UBA_LIST_FINE);}
fn pCount()->u32{return pFine();}
fn pLive()->u32{return select(pCount(),0u,bConverged());}
// The band tiles (the tile list's length), and the solve's.
// A native row's residual, band-equivalent: 1/64 of the row (the root's
// operator) at the step's scale, times the band's over the root's target
// (capacity.y), so one threshold stops both.
fn nScale()->f32{return params.policy.z/64.0*bitcast<f32>(params.capacity.y);}
fn pTileCount()->u32{return min(pIndex(0u),params.capacity.x);}
fn pTileLive()->u32{return select(pTileCount(),0u,bConverged());}
// Tile j of the list: its brick slot and the cell of lane-local l (0..3 per axis) in the brick.
fn pTileSlot(w:u32)->u32{return w>>3u;}
fn pTileCell(w:u32,l:vec3u)->vec3u{let t=w&7u;return 4u*vec3u(t&1u,(t>>1u)&1u,t>>2u)+l;}
`;
  const poolWGSL=/* wgsl */`
@group(1) @binding(9) var<storage,read_write> pool:array<u32>;
@group(1) @binding(10) var<storage,read_write> poolB:array<u32>;
${a.poolWGSL(["pool","poolB"])}
fn pF(base:u32,at:u32)->f32{return bitcast<f32>(pool[base+at]);}
fn pSet(base:u32,at:u32,value:f32){pool[base+at]=bitcast<u32>(value);}
fn qF(base:u32,at:u32)->f32{return bitcast<f32>(poolB[base+at]);}
fn qSet(base:u32,at:u32,value:f32){poolB[base+at]=bitcast<u32>(value);}
fn pAddr(slot:u32,l:vec3u)->u32{return UBA_STRIDE*slot+l.x+8u*(l.y+8u*l.z);}
// The h face and its native 4h row differ tangentially by up to 1.5h.
// Carry the row's start to the h face's tangential position along the start
// pressure's own tangential gradient, leaving the dynamic pressure as the
// solved unknown. Otherwise an h/4h interface along gravity turns a resting
// pressure column into flow across it. The gradient is the start's (last
// frame's pressure; the kick's own, re-solved: encodeSolve), measured at the
// h cell from its band neighbours, not rho g: grad p = rho (g - a), so a
// falling or accelerating liquid has none of it. rho g there forced a jump
// of up to 1.5 rho g h across every seam subface: a free-falling block grew
// seam shear of 0.1 m/s from frame 1, its wall faces read suction and
// separated, and the wall-plane phi opened into an air film (5-11% of each
// depth wall's volume in 16 frames).
// Native and band RHS use opposite offsets, and projection uses the same
// offset: one pressure difference for both incident cells and their flux.
fn bSeamHead(at:u32,f:u32)->f32{
 // An air-side row is the zero-pressure free-surface boundary, not a
 // pressure sample at the fine cell centre. Its ghost-fluid coefficient
 // already locates that boundary. A hydrostatic offset here imposes a
 // different nonzero air pressure on each subface; a tiny theta amplifies
 // that incompatible boundary data into opposing jets.
 let kinds=pool[P_KINDS+at];
 if(!bLiquid(kinds)){return 0.0;}
 // The h cell's offset from its 4h row's centre on each tangential axis (the
 // normal offset is the discretisation's) times the start's slope there: a
 // central difference over liquid band neighbours, one-sided at the band's
 // or the liquid's edge, none without a neighbour. Exact for a linear start.
 let slot=at/UBA_STRIDE;let local=at%UBA_STRIDE;let lb=vec3u(local&7u,(local>>3u)&7u,local>>6u);
 let own=pF(P_BASE,at);var head=0.0;
 for(var axis=0u;axis<3u;axis++){
  if(axis==f/2u){continue;}
  var rise=0.0;var count=0.0;
  for(var side=0u;side<2u;side++){
   let g=2u*axis+side;
   if(bFaceKind(kinds,g)==K_BAND){let m=pNear0(at,slot,lb,g);if(bLiquid(pool[P_KINDS+m])){rise+=f32(bSign(g))*(pF(P_BASE,m)-own);count+=1.0;}}
  }
  if(count>0.0){head+=rise/count*(f32(lb[axis]&3u)-1.5);}
 }
 return head;
}
// The pool offset of band cell q (UB_NONE_ID: no band cell).
fn pAt(q:vec3i)->u32{if(!pBandCell(q)){return UB_NONE_ID;}return pAddr(ubaSlot(ubaBrickAt(q)),vec3u(q)&vec3u(7u));}
// The band cell across face f of band cell 'at' (brick slot, brick-local
// cell lb): only for a K_BAND face, whose cell is a live brick's (a band
// tile marks its brick), so across a brick face it is the record's neighbour.
fn pNear0(at:u32,slot:u32,lb:vec3u,f:u32)->u32{
 let axis=f/2u;var m=vec3i(lb);m[axis]+=bSign(f);
 if(m[axis]>=0&&m[axis]<8){let step=select(select(64u,8u,axis==1u),1u,axis==0u);return select(at-step,at+step,bSign(f)>0);}
 var d=vec3i(1);d[axis]+=bSign(f);let w=vec3u(m&vec3i(7));
 return pAddr(ubaLevelRecord(0u,slot,u32(d.x+3*d.y+9*d.z)),w);
}
fn pNear(at:u32,p:vec3i,f:u32)->u32{return pNear0(at,at/UBA_STRIDE,vec3u(p&vec3i(7)),f);}
// A unit face (most band faces) loads nothing.
fn pCoefficient(at:u32,kinds:u32,f:u32)->f32{if(bUnit(kinds,f)){return params.solve[1u+f/2u];}return pF(P_C+f*PLANE,at);}
fn pForced(at:u32,f:u32)->f32{return qF(P_U+f*PLANE,at);}
${S?"fn pVolume(at:u32,f:u32)->f32{return qF(P_V+f*PLANE,at);}":""}
// A wall face's native one-cell halo, less the row's start: max(x + lift, -base).
fn pLift(at:u32,f:u32)->f32{return f32(bSign(f))*params.policy.x*pForced(at,f)*params.hDt[f/2u]/params.hDt.w;}
fn pZeroSeam(p:vec3i,word:u32)->f32{return 0.0;}
// The native row (level 2: page block, cell (q/4)&7) of h cell q's tile,
// channel-relative; and the one across face f of h cell p (a seam's).
fn nAt(q:vec3i)->u32{let b=umPageWord(q>>vec3u(5u));let l=(vec3u(q)>>vec3u(2u))&vec3u(7u);return UBA_STRIDE*b+l.x+8u*(l.y+8u*l.z);}
fn nAcross(p:vec3i,f:u32)->u32{var q=p;q[f/2u]+=bSign(f);return nAt(q);}
fn nKind(m:u32,f:u32)->u32{return (m>>(3u*f))&7u;}
// Diagnostics: a native row's class (4 interior, 5 seam, 6 wall).
fn pDiagNative(m:u32)->u32{var c=4u;for(var f=0u;f<6u;f++){let k=nKind(m,f);if(k==N_WALL){return 6u;}if(k==N_SEAM){c=5u;}}return c;}
fn nStride(axis:u32)->u32{return select(select(64u,8u,axis==1u),1u,axis==0u);}
// A page's flags (word 512 of its native mask): bit 0 live (band tiles or
// native rows: the level 2 box and the tail's), bit 1 native rows.
fn nPage(block:u32)->u32{return poolB[PN_M+UBA_STRIDE*block+512u];}
`;
  // A BULK16 store's tail items reach their page neighbours' words (tAt).
  const header=(neighbours=!!BP)=>uniformMixedTopologyWGSL(layout,0)+(neighbours?uniformMixedTopologyNeighboursWGSL():"")+uniformMixedFaceAddressWGSL+uniformMixedSolidWGSL(S?2:undefined,this.solid?.coarse?.count)+params+/* wgsl */`
@group(1) @binding(2) var<storage,read_write> state:array<atomic<u32>>;
const R_PRESSURE:u32=${this.rootSpan.pressure}u;const R_RHS:u32=${this.rootSpan.rhs}u;const R_MINIMUM:u32=${this.rootSpan.minimum}u;
`+tableWGSL+constants+helpers;
  const wet=/* wgsl */`
fn bWet(phi:f32,width:f32)->bool{return ${uniformMixedPressureLiquidWGSL("phi","width*params.policy.w")};}
`;
  const theta=/* wgsl */`
fn bTheta(liquidPhi:f32,airPhi:f32)->f32{return ${uniformMixedSurfaceThetaWGSL("liquidPhi","airPhi")};}
fn bPairTheta(a:f32,wetA:bool,b:f32,wetB:bool)->f32{if(wetA==wetB){return 1.0;}if(wetA){return bTheta(a,b);}return bTheta(b,a);}
// The zero crossing of the vertex field between a liquid and an air centre, linear from each centre to the face's (UniformPressureBand).
fn bFaceTheta(face:f32,a:f32,wetA:bool,b:f32,wetB:bool)->f32{
 if(wetA==wetB){return 1.0;}
 let liquid=select(b,a,wetA);let air=select(a,b,wetA);
 if(face>0.0){return clamp(0.5*liquid/(liquid-face),${UNIFORM_MIXED_THETA_MIN},${(1/UNIFORM_MIXED_THETA_MIN).toFixed(1)});}
 return clamp(0.5+0.5*face/min(face-air,-1e-30),${UNIFORM_MIXED_THETA_MIN},${(1/UNIFORM_MIXED_THETA_MIN).toFixed(1)});
}
`;
  const owner=wet+/* wgsl */`
@group(1) @binding(3) var<storage,read> phi:array<f32>;
@group(1) @binding(6) var vertexPhi:texture_3d<f32>;
fn bCentrePhi(q:vec3i)->f32{let c8=udLoad8_vertexPhi(q,1u);var s=0.0;for(var k=0u;k<8u;k++){s+=c8[k];}return 0.125*s;}
fn bFacePhi(p:vec3i,axis:u32,sign:i32)->f32{
 var o=p;if(sign>0){o[axis]+=1;}let u=(axis+1u)%3u;let v=(axis+2u)%3u;
 var s=0.0;for(var k=0u;k<4u;k++){var c=o;c[u]+=i32(k&1u);c[v]+=i32(k>>1u);s+=textureLoad(vertexPhi,c,0).x;}return 0.25*s;
}
fn bHTile(j:u32)->u32{return umTopology[UM_TILES+j];}
fn bPhiH(q:vec3i)->f32{return phi[umOwnerAt(q).index];}
// UniformPressureBand's face classification by an h neighbour's band phi.
struct BNeighbour {phi:f32,wet:bool,neumann:bool}
fn bNeighbour(q:vec3i)->BNeighbour{
 let n=umOwnerAt(q);let other=phi[n.index];
 if(n.width==1u){let b=bPhiH(q);let wet=bWet(b,1.0);return BNeighbour(b,wet,wet);}
 if(bWet(other,f32(n.width))){return BNeighbour(other,true,true);}
 let own=bCentrePhi(q);let wet=bWet(own,1.0);return BNeighbour(own,wet,wet);
}
`;
  const lanes=/* wgsl */`@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32,@builtin(num_workgroups) groups:vec3u`;
  // The 16h bulk tier (bulkTier "16h"): a member region's 64 native rows take
  // one correction, its 16h row their sum. The region mean of a lane value over
  // a page's 8 regions (64 lanes each), through workgroup array 'red' (512+
  // floats); every lane of the page calls it.
  const bulkMean=(red:string)=>/* wgsl */`
const N_BULK:u32=${1<<22}u;
fn nRegionOf(lane:u32)->u32{return ((lane>>2u)&1u)|(((lane>>5u)&1u)<<1u)|(((lane>>8u)&1u)<<2u);}
fn nRegionLane(lane:u32)->u32{return (lane&3u)|(((lane>>3u)&3u)<<2u)|(((lane>>6u)&3u)<<4u);}
fn nBulkMean(v:f32,bulk:bool,lane:u32)->f32{
 let local=nRegionLane(lane);let i=64u*nRegionOf(lane)+local;
 workgroupBarrier();${red}[i]=select(0.0,v,bulk);workgroupBarrier();
 for(var s=32u;s>0u;s>>=1u){if(local<s){${red}[i]+=${red}[i+s];}workgroupBarrier();}
 return select(v,${red}[i-local]/64.0,bulk);
}
`;
  // pNative's bulk links: a member's link inside its region carries nothing,
  // member to member a 16h coupling (16 4h subfaces at 16h spacing: 16 per
  // subface on the native 64), member to native a 10h one (25.6), the
  // member's pressure carried to the subface's tangential position along the
  // native side's measured slope (the head; bSeamHead's at ratio 4). A
  // member's pressure in its links is its 16h one, its rows' start mean
  // (UniformBulkTier), not the row's own start: each row keeps its offset.
  const bulkNative=B?bulkMean("nRed")+/* wgsl */`
const N_RECORD:u32=${this.bulkWord}u;
const N_HEADS:u32=${UNIFORM_BULK_HEADS(B.regions)}u;const N_MEAN:u32=${UNIFORM_BULK_MEAN(B.regions)}u;
var<workgroup> nRed:array<f32,512>;
var<workgroup> nBulkW:u32;
fn nRegionKey(k:u32)->u32{return 8u*(k>>9u)+nRegionOf(k&511u);}
fn nMember(k:u32)->bool{return (poolB[N_RECORD+nRegionKey(k)]&1u)!=0u;}
fn nBulkStart(k:u32)->f32{if(nMember(k)){return bitcast<f32>(poolB[N_RECORD+N_MEAN+nRegionKey(k)]);}return rootArena[R_PRESSURE+k];}
// A member face's seam head slot: region, face, the subface's tangential cell.
fn nHeadIndex(k:u32,f:u32)->u32{let a=f/2u;let t=umTileCoord(k);return N_HEADS+96u*nRegionKey(k)+16u*f+(t[(a+1u)%3u]&3u)+4u*(t[(a+2u)%3u]&3u);}
// The key across face f of key k, N_KEYS outside (a SOLID page).
fn nNeighbour(k:u32,f:u32)->u32{
 let axis=f/2u;let side=f&1u;let stride=nStride(axis);let lk=k&511u;let l=vec3u(lk&7u,(lk>>3u)&7u,lk>>6u);
 if(l[axis]!=select(0u,7u,side==1u)){return select(k-stride,k+stride,side==1u);}
 let step=select(select(9u,3u,axis==1u),1u,axis==0u);let across=umBlockNeighbourWord(k>>9u,select(13u-step,13u+step,side==1u));
 if(across>=N_SENTINEL){return N_KEYS;}
 return 512u*across+(select(k+7u*stride,k-7u*stride,side==1u)&511u);
}
fn nBulkOpen(n:u32)->bool{
 if(n>=N_KEYS){return false;}
 if(pIndex(SLOTS+n)!=0u||!nLiquid(n)){return false;}
 ${S?"if(rootTopology[n].x<0.99999){return false;}":""}
 return true;
}
// Row q's start slope along both tangential axes of normal a (central over
// open neighbours, native ones unless rows: any row's own start; one-sided
// at an edge), times q's offset from its region's centre: the member
// pressure at q's tangential position.
fn nBulkHead(q:u32,a:u32,rows:bool)->f32{
 let own=rootArena[R_PRESSURE+q];let t=umTileCoord(q);var head=0.0;
 for(var b=0u;b<3u;b++){
  if(b==a){continue;}
  var rise=0.0;var count=0.0;
  for(var side=0u;side<2u;side++){let g=2u*b+side;let n=nNeighbour(q,g);if(nBulkOpen(n)&&(rows||!nMember(n))){rise+=f32(bSign(g))*(rootArena[R_PRESSURE+n]-own);count+=1.0;}}
  if(count>0.0){head+=rise/count*(f32(t[b]&3u)-1.5);}
 }
 return head;
}
// (coupling per V/h4^2, start offset) of native link k -> n across face f.
// A closed native row (a wall's CM11a unknown, open 0) has no open
// neighbours to measure: the member row's own measures the slope there.
// With none, the head was 0 and the wall unknown jumped to the 16h mean at
// entry (+-1.5 rho g 4h), a pressure change no fluid face carries.
fn nBulkLink(k:u32,n:u32,f:u32,member:bool)->vec2f{
 let other=nMember(n);
 if(member&&other){if(nRegionKey(k)==nRegionKey(n)){return vec2f(0.0);}return vec2f(16.0,0.0);}
 if(!member&&!other){return vec2f(64.0,0.0);}
 let native=select(k,n,member);var head=0.0;
 if(${S?"rootTopology[native].x<=1e-5":"false"}){head=nBulkHead(select(n,k,member),f/2u,true);}else{head=nBulkHead(native,f/2u,false);}
 if(member){poolB[N_RECORD+nHeadIndex(k,f)]=bitcast<u32>(head);return vec2f(25.6,-head);}
 return vec2f(25.6,head);
}
`:"";
  // The overlapping box of a brick of level L: the records of its 27
  // neighbours, the pool offset of a box cell, and its 12^3 iterate.
  const box=(L:number)=>/* wgsl */`
var<workgroup> cR:array<u32,28>;
var<workgroup> cM:array<u32,27>;
var<workgroup> cX:array<f32,1728>;
var<workgroup> cN:u32;
fn cRecords(slot:u32,lane:u32){
 workgroupBarrier();
 if(lane<27u){let w=ubaLevelRecord(${L}u,slot,lane);cR[lane]=w;var m=0u;if(w<UBA_SEAM){m=ubaLevelRecord(${L}u,w,28u);}cM[lane]=m;}
 else if(lane==27u){cR[27]=ubaLevelRecord(${L}u,slot,29u);}
 workgroupBarrier();
}
// The pool offset of box cell c (brick-local, -8..15), UB_NONE_ID where no live brick's live tile holds it.
fn cAddr(c:vec3i)->u32{
 let d=vec3u(c+8)>>vec3u(3u);let n=d.x+3u*d.y+9u*d.z;let w=cR[n];let l=vec3u(c&vec3i(7));
 if(w<UBA_SEAM&&((cM[n]>>pTileBit(l))&1u)!=0u){return UBA_STRIDE*w+l.x+8u*(l.y+8u*l.z);}
 return UB_NONE_ID;
}
fn cBox(c:vec3i)->u32{let q=vec3u(c+2);return q.x+12u*(q.y+12u*q.z);}
fn cStaged(k:u32)->vec3i{return vec3i(vec3u(k%12u,(k/12u)%12u,k/144u))-2;}
// The i-th box cell (of 500) of colour col: the box is -1..8, the outer ring stays.
fn cBoxCell(col:u32,i:u32)->vec3i{let y=(i/5u)%10u;let z=i/50u;let x=2u*(i%5u)+((col+y+z+1u)&1u);return vec3i(vec3u(x,y,z))-1;}
fn cCore(lane:u32)->vec3i{return vec3i(vec3u(lane&7u,(lane>>3u)&7u,lane>>6u));}
`;
  // A coarse level's (1 or 2) rows, linear: x = (r + sum k x_n)/d.
  const coarse=(L:1|2,sweeps=schedule.sweeps)=>{const [X,Y,R,D,K,PL]=L===1?["PA_X","PA_Y","PA_R","PA_D","PA_K","PLANE1"]:["PB_X","PB_Y","PB_R","PB_D","PB_K","PLANE2"];return box(L)+/* wgsl */`
const CX:u32=${X};const CY:u32=${Y};const CRH:u32=${R};const CD:u32=${D};const CK:u32=${K};const CPL:u32=${PL};
fn cDiag(at:u32)->f32{if(at==UB_NONE_ID){return 0.0;}return qF(CD,at);}
// A coarse row, held by its lane: diagonal (0: none), RHS, couplings.
struct CCell {d:f32,r:f32,k:array<f32,6>}
fn cCell(at:u32)->CCell{
 var cell:CCell;if(at==UB_NONE_ID){return cell;}
 let d=qF(CD,at);if(d<=0.0){return cell;}
 cell.d=d;cell.r=qF(CRH,at);for(var f=0u;f<6u;f++){cell.k[f]=qF(CK+f*CPL,at);}
 return cell;
}
fn cBoxRow(col:u32,lane:u32)->CCell{if(lane>=500u){return cCell(UB_NONE_ID);}return cCell(cAddr(cBoxCell(col,lane)));}
fn cOff(c:vec3i,cell:CCell)->f32{
 var off=0.0;
 // A select, not a branch: a branch a face serialises the six workgroup
 // loads (twice the sweep's time); a face with no coupling may read a
 // non-row's stale value, which the select drops (0 * NaN would not).
 for(var f=0u;f<6u;f++){var q=c;q[f/2u]+=bSign(f);off+=select(0.0,cell.k[f]*cX[cBox(q)],cell.k[f]!=0.0);}
 return off;
}
// The box sweeps: lane < 500 holds one cell of each colour.
fn cSweeps(red:vec3i,cR0:CCell,black:vec3i,cB0:CCell){
 for(var s=0u;s<${sweeps}u;s++){
  for(var half=0u;half<2u;half++){
   var c=red;var cell=cR0;if(half==1u){c=black;cell=cB0;}
   if(cell.d>0.0){cX[cBox(c)]=(cell.r+cOff(c,cell))/cell.d;}
   workgroupBarrier();
  }
 }
}
`;};
  // Children (2^3 aggregates) of a level summed into the parent's rows: a child's
  // diagonal less its couplings inside the aggregate, its couplings out of it by face.
  const bakeArrays=/* wgsl */`
var<workgroup> gPart:array<f32,512>;
var<workgroup> gCross:array<f32,1536>;
var<workgroup> gRes:array<f32,512>;
// Aggregate A (4^3) of a brick's 512 children: its scaled diagonal and couplings.
fn gChild(A:vec3u,k:u32)->u32{let l=2u*A+vec3u(k&1u,(k>>1u)&1u,k>>2u);return l.x+8u*(l.y+8u*l.z);}
fn gDiag(A:vec3u)->f32{
 var s=0.0;var outward=0.0;
 for(var k=0u;k<8u;k++){let i=gChild(A,k);s+=gPart[i];for(var axis=0u;axis<3u;axis++){outward+=gCross[512u*axis+i];}}
 return pAggregateDiagonal(s,outward,P_SCALE);
}
fn gCouple(A:vec3u,f:u32)->f32{var s=0.0;let axis=f/2u;for(var k=0u;k<8u;k++){if(((k>>axis)&1u)==(f&1u)){s+=gCross[512u*axis+gChild(A,k)];}}return P_SCALE*s;}
fn gSum(A:vec3u)->f32{var s=0.0;for(var k=0u;k<8u;k++){s+=gRes[gChild(A,k)];}return s;}
// A coarse child's part: d less its inner couplings, its outer ones by axis.
fn gCoarseChild(i:u32,l:vec3u,at:u32,d:f32,K:u32,PL:u32){
 var part=0.0;var cross=vec3f(0.0);
 if(at!=UB_NONE_ID&&d>0.0){
  part=d;
  for(var axis=0u;axis<3u;axis++){
   let bit=l[axis]&1u;let inner=2u*axis+1u-bit;let outer=2u*axis+bit;
   var qi=vec3i(l);qi[axis]+=bSign(inner);var qo=vec3i(l);qo[axis]+=bSign(outer);
   // A missing correction row is fixed at zero. Its coefficient remains
   // on the diagonal; it must neither cancel inside this aggregate nor
   // become a link to unrelated live children in the next aggregate.
   if(cDiag(cAddr(qi))>0.0){part-=qF(K+inner*PL,at);}
   if(cDiag(cAddr(qo))>0.0){cross[axis]=qF(K+outer*PL,at);}
  }
 }
 gPart[i]=part;gCross[i]=cross.x;gCross[512u+i]=cross.y;gCross[1024u+i]=cross.z;
}
`;
  const zero=/* wgsl */`var<workgroup> pN:u32;`;
  // ---- level 0 rows, in place (nonlinear rows: walls follow, inside rows clamp).
  const relaxRow=/* wgsl */`
var<workgroup> rN:u32;
// The iterate plane a sweep reads (X or X1); it writes the other. A row
// reads the last sweep's plane except a black row's red neighbours in its own
// tile (this sweep's, written by its own workgroup), never a plane another
// workgroup is writing, so a sweep's result does not depend on the order
// workgroups run.
override pSrc:u32=0u;
fn pPlane(h:u32)->u32{return select(P_X,P_X1,h==1u);}
fn pNeighbourX(at:u32,slot:u32,lb:vec3u,f:u32,black:bool)->f32{
 let axis=f/2u;let m=i32(lb[axis])+bSign(f);
 let own=m>=0&&m<8&&(u32(m)>>2u)==(lb[axis]>>2u);
 return pF(pPlane(select(pSrc,1u-pSrc,black&&own)),pNear0(at,slot,lb,f));
}
// UniformPressureBand's relaxation of a row from its neighbours' iterate; a
// row with walls solves its own halos' clamps exactly (four active-set passes).
fn pRelaxRow(at:u32,slot:u32,lb:vec3u,kinds:u32,black:bool)->f32{
 var off=pF(P_RHS,at);var walls=0u;
 for(var f=0u;f<6u;f++){
  let kind=bFaceKind(kinds,f);
  if(kind==K_BAND){off+=pCoefficient(at,kinds,f)*pNeighbourX(at,slot,lb,f,black);}
  else if(kind==K_SEAM){off+=pF(P_C+f*PLANE,at)*qF(PN_X,nAcross(vec3i(ubaOrigin(ubaBrick(slot))+lb),f));}
  else if(kind==K_WALL){walls|=1u<<f;}
 }
 let d0=pF(P_DIAG,at);var v=off/d0;
 if(walls!=0u||bInside(kinds)){
  let base=pF(P_BASE,at);
  if(walls!=0u){
   for(var it=0u;it<4u;it++){
    var d=d0;var o=off;
    // Ascending bits preserve the original face accumulation order.
    var remaining=walls;while(remaining!=0u){
     let f=firstTrailingBit(remaining);remaining&=remaining-1u;
     let w=pF(P_C+f*PLANE,at);let g=pLift(at,f);
     if(v+g>=-base){d-=w;o+=w*g;}else{o-=w*base;}
    }
    if(d>0.0){v=o/d;}
   }
  }
  if(bInside(kinds)){v=max(v,-base);}
 }
 return v;
}
`;
  // Trilinear prolongation of the parent's correction to child cell c (brick-local): over the parents that exist, renormalised.
  const prolong=(xOf:string,dOf:string)=>/* wgsl */`
fn cProlong(c:vec3i)->f32{
 let P=c>>vec3u(1u);let o=select(vec3i(-1),vec3i(1),(c&vec3i(1))==vec3i(1));var sum=0.0;var total=0.0;
 for(var i=0u;i<8u;i++){
  let d=vec3i(vec3u(i&1u,(i>>1u)&1u,i>>2u));let q=P+d*o;
  let w=select(0.75,0.25,d.x==1)*select(0.75,0.25,d.y==1)*select(0.75,0.25,d.z==1);
  if(${dOf}>0.0){sum+=w*${xOf};total+=w;}
 }
 return select(0.0,sum/total,total>0.0);
}
`;
  // The BULK16 store in pool B (fields.bulkPages): the live slots, each
  // slot's record (BULK16_PAGE), and the addresses the tail levels name a
  // BULK16 cell by. A 16h region (i, 0..7) of slot is BA | slot<<5 | i at
  // levels 4 (and 0: its pressure), BA | slot<<5 | 16 | i at level 3 (the
  // region's 8h iterate, one value: a BULK16 page holds no sub-16h mode),
  // BA | slot<<5 | 8 its page's 32h row. Channel PT_c's word of one is the
  // record's (baWord): level 3's X, Y and R its own, every other channel
  // (D, the couplings) level 4's.
  const store=BP?/* wgsl */`
const BS:u32=${this.storeWord}u;const BS_LIVE:u32=${this.storeWord+BP.layout.live}u;const BS_REC:u32=${this.storeWord+BP.layout.records}u;const BS_STRIDE:u32=${BULK16_PAGE_WORDS}u;
const BS_P:u32=${BULK16_PAGE.pressure}u;const BS_N:u32=${BULK16_PAGE.pressureNext}u;const BS_RHS:u32=${BULK16_PAGE.rhs}u;const BS_U:u32=${BULK16_PAGE.velocity}u;const BS_US:u32=${BULK16_PAGE.velocityNext}u;
const BS_T3:u32=${BULK16_PAGE.tail3}u;const BS_T4:u32=${BULK16_PAGE.tail4}u;const BS_T5:u32=${BULK16_PAGE.tail5}u;
const BA:u32=0x80000000u;const N_BULKN:u32=6u;
fn bsCount()->u32{return poolB[BS+${BULK16_STATUS.live}u];}
fn bsSlot(i:u32)->u32{return poolB[BS_LIVE+i];}
fn bsAt(slot:u32,w:u32)->u32{return BS_REC+BS_STRIDE*slot+w;}
fn bsF(slot:u32,w:u32)->f32{return bitcast<f32>(poolB[bsAt(slot,w)]);}
fn bsSet(slot:u32,w:u32,v:f32){poolB[bsAt(slot,w)]=bitcast<u32>(v);}
// The slot's page in the lattice (umPageWord's): the slot keeps the pool's world page.
fn bsPage(slot:u32)->vec3i{return vec3i(bitcast<i32>(poolB[bsAt(slot,0u)]),bitcast<i32>(poolB[bsAt(slot,1u)]),bitcast<i32>(poolB[bsAt(slot,2u)]))-umFramePage();}
// The slot of a BULK16 page word (a page's, or a block neighbour row's), 0 for any other.
fn bsWordSlot(w:u32)->u32{return select(0u,w&0x3fffffffu,umBulkTag(w));}
fn baOf(slot:u32,l3:bool,i:u32)->u32{return BA|(slot<<5u)|select(0u,16u,l3)|i;}
fn baIs(at:u32)->bool{return at!=UB_NONE_ID&&(at&BA)!=0u;}
fn baSlot(at:u32)->u32{return (at>>5u)&0x3ffffffu;}
fn baWord(base:u32,at:u32)->u32{
 let j=(base-PT_X)/PLANE2;let i=at&15u;var w=BS_T4+8u*j+i;
 if(i==8u){w=BS_T5+j;}else if((at&16u)!=0u&&j<3u){w=BS_T3+8u*j+i;}
 return bsAt(baSlot(at),w);
}
// The region of slot across face f of its region i (none: 8u), and of a 4h tile t of a page across a face of normal a.
fn bsRegion(t:vec3u)->u32{return (t.x>>2u)|((t.y>>2u)<<1u)|((t.z>>2u)<<2u);}
// Region i of slot is open (BULK16_PAGE.open); a closed one is a static wall: no row, no flux.
fn bsOpen(slot:u32,i:u32)->bool{return ((poolB[bsAt(slot,${BULK16_PAGE.open}u)]>>i)&1u)!=0u;}
`:"";
  // A BULK16 tail item: the page list's resident blocks, then the live slots (BA | slot).
  const items=BP?/* wgsl */`
fn tCount(n2:u32)->u32{return n2+bsCount();}
fn tItem(e:u32,n2:u32)->u32{if(e<n2){return ubaLevelSlot(2u,e);}return BA|bsSlot(e-n2);}
fn tF(base:u32,at:u32)->f32{if(baIs(at)){return bitcast<f32>(poolB[baWord(base,at)]);}return qF(base,at);}
fn tSet(base:u32,at:u32,v:f32){if(baIs(at)){poolB[baWord(base,at)]=bitcast<u32>(v);}else{qSet(base,at,v);}}
fn tU(base:u32,at:u32)->u32{if(baIs(at)){return poolB[baWord(base,at)];}return poolB[base+at];}
fn tSetU(base:u32,at:u32,v:u32){if(baIs(at)){poolB[baWord(base,at)]=v;}else{poolB[base+at]=v;}}
`:/* wgsl */`
fn tCount(n2:u32)->u32{return n2;}
fn tItem(e:u32,n2:u32)->u32{return ubaLevelSlot(2u,e);}
fn tF(base:u32,at:u32)->f32{return qF(base,at);}
fn tSet(base:u32,at:u32,v:f32){qSet(base,at,v);}
fn tU(base:u32,at:u32)->u32{return poolB[base+at];}
fn tSetU(base:u32,at:u32,v:u32){poolB[base+at]=v;}
`;
  const tail=store+items+/* wgsl */`
// The tail's levels per page: 3 (8h, 4^3 at word 0), 4 (16h, 2^3 at 64), 5 (32h, one at 72).
fn tSize(L:u32)->u32{return 4u>>(L-3u);}
fn tOffset(L:u32)->u32{return select(select(72u,64u,L==4u),0u,L==3u);}
// Item's level L cell j (a BULK16 item's level 3 cell is its region j's).
fn tCell(item:u32,L:u32,j:u32)->u32{
 ${BP?"if((item&BA)!=0u){let slot=item&0x7fffffffu;if(L==5u){return baOf(slot,false,8u);}return baOf(slot,L==3u,j);}":""}
 return UBA_STRIDE*item+tOffset(L)+j;
}
// The pool offset (or BULK16 address) of level L cell g of item's grid (g
// may lie one page out), UB_NONE_ID outside stored and BULK16 pages. A
// BULK16 item's own grid is levels 4 and 5 only.
fn tAt(block:u32,L:u32,g:vec3i)->u32{
 let s=i32(tSize(L));let d=(g+vec3i(2*s))/s-vec3i(2);let l=g-d*s;let j=u32(l.x+s*(l.y+s*l.z));
 if(all(d==vec3i(0))){return tCell(block,L,j);}
 let k=u32(13+d.x+3*d.y+9*d.z);
 ${BP?`var w=0u;
 if((block&BA)!=0u){w=umPageRawWord(bsPage(block&0x7fffffffu)+d);if(umBlockStored(w)){return UBA_STRIDE*w+tOffset(L)+j;}}
 else{let b=ubaLevelRecord(2u,block,k);if(b!=UB_NONE_ID){return UBA_STRIDE*b+tOffset(L)+j;}w=umBlockNeighbourWord(block,k);}
 let slot=bsWordSlot(w);if(slot==0u){return UB_NONE_ID;}
 if(L==5u){return baOf(slot,false,8u);}
 let r=select(vec3u(l),vec3u(l)>>vec3u(1u),L==3u);return baOf(slot,L==3u,r.x+2u*r.y+4u*r.z);`:`let b=ubaLevelRecord(2u,block,k);if(b==UB_NONE_ID){return UB_NONE_ID;}
 return UBA_STRIDE*b+tOffset(L)+j;`}
}
fn tLocal(j:u32,s:u32)->vec3i{return vec3i(vec3u(j%s,(j/s)%s,j/(s*s)));}
// A page with band tiles or native rows (pNative's flag); every BULK16 item.
fn tPageLive(block:u32)->bool{${BP?"if((block&BA)!=0u){return true;}":""}return (nPage(block)&1u)!=0u;}
`;
  // A BULK16 region's faces (needs store). Region i (0..7, x+2y+4z of its
  // page's 2^3) of slot meets across face f its page's other region, a
  // BULK16 page's region, a resident block's 16 4h rows (subface j: tile
  // bNativeTile, their face f^1 an N_BULKN link to it), or nothing (SOLID:
  // a wall, its flux zero).
  const bulkRows=BP?bulk16FaceIndexWGSL+/* wgsl */`
const B_NONE:u32=0u;const B_REGION:u32=1u;const B_BLOCK:u32=2u;
struct BNear {kind:u32,slot:u32,region:u32,block:u32}
fn bRegionCoord(i:u32)->vec3u{return vec3u(i&1u,(i>>1u)&1u,i>>2u);}
fn bRegionIndex(r:vec3u)->u32{return r.x+2u*r.y+4u*r.z;}
// A closed region (bsOpen), or a face toward one, is a wall: B_NONE.
fn bNear(slot:u32,page:vec3i,i:u32,f:u32)->BNear{
 let a=f/2u;let side=f&1u;var r=bRegionCoord(i);
 if(!bsOpen(slot,i)){return BNear(B_NONE,0u,0u,0u);}
 if(r[a]!=side){r[a]=side;let j=bRegionIndex(r);if(!bsOpen(slot,j)){return BNear(B_NONE,0u,0u,0u);}return BNear(B_REGION,slot,j,0u);}
 var d=vec3i(0);d[a]=bSign(f);let w=umPageRawWord(page+d);r[a]=1u-side;
 let other=bsWordSlot(w);if(other!=0u){let j=bRegionIndex(r);if(!bsOpen(other,j)){return BNear(B_NONE,0u,0u,0u);}return BNear(B_REGION,other,j,0u);}
 if(umBlockStored(w)){return BNear(B_BLOCK,0u,i,w);}
 return BNear(B_NONE,0u,0u,0u);
}
// The 4h tile of the block across face f of region i at subface j (0..15).
fn bNativeTile(i:u32,f:u32,j:u32)->vec3u{
 let a=f/2u;let r=bRegionCoord(i);var t=vec3u(0u);t[a]=select(7u,0u,(f&1u)==1u);
 let u=(a+1u)%3u;let v=(a+2u)%3u;t[u]=4u*r[u]+(j&3u);t[v]=4u*r[v]+(j>>2u);return t;
}
fn bTileLocal(t:vec3u)->u32{return t.x+8u*(t.y+8u*t.z);}
// The native row's link to region i of its face f^1 (active and N_BULKN), its level 2 offset at.
fn bLinked(at:u32,f:u32)->bool{let m=poolB[PN_M+at];return (m&N_ACTIVE)!=0u&&nKind(m,f^1u)==N_BULKN;}
// Region i's 16h coupling to another region (open: whole pages are).
fn bRegionK(f:u32)->f32{let h4=4.0*params.hDt[f/2u];return 256.0/(h4*h4);}
// The record's face of region i's face f (bpFace: its axis, plane 0..2, tangential region).
fn bFace(i:u32,f:u32)->u32{let a=f/2u;let r=bRegionCoord(i);return bpFace(a,r[a]+(f&1u),r[(a+1u)%3u],r[(a+2u)%3u]);}
// Sum k x across face f of region i: level 0's (pressureNext; the native
// rows' x), or level 2's (natives: PB_X; regions: none, their level 3
// correction starts after it).
fn bLinkSum(slot:u32,page:vec3i,i:u32,f:u32,level2:bool)->f32{
 let n=bNear(slot,page,i,f);
 if(n.kind==B_REGION){if(level2){return 0.0;}return bRegionK(f)*bsF(n.slot,BS_N+n.region);}
 if(n.kind==B_BLOCK){
  var sum=0.0;
  for(var j=0u;j<16u;j++){
   let at=UBA_STRIDE*n.block+bTileLocal(bNativeTile(i,f,j));
   if(bLinked(at,f)){sum+=qF(PN_K+(f^1u)*PLANE2,at)*select(qF(PN_X,at),qF(PB_X,at),level2);}
  }
  return sum;
 }
 return 0.0;
}
// Region i's level 0 residual: rhs - d x + sum k x (lanes 0..47 a (region, face)).
var<workgroup> bRed:array<f32,48>;
fn bResidual(slot:u32,lane:u32)->f32{
 let page=bsPage(slot);
 if(lane<48u){bRed[lane]=bLinkSum(slot,page,lane/6u,lane%6u,false);}
 workgroupBarrier();
 var r=0.0;
 if(lane<8u){let s=((bRed[6u*lane]+bRed[6u*lane+1u])+(bRed[6u*lane+2u]+bRed[6u*lane+3u]))+(bRed[6u*lane+4u]+bRed[6u*lane+5u]);
  r=bsF(slot,BS_RHS+lane)-bsF(slot,BS_T4+24u+lane)*bsF(slot,BS_N+lane)+s;}
 workgroupBarrier();
 return r;
}
`:"";
  // Red-black sweeps of a BULK16 page's eight rows (lane < 8: its region's
  // RHS r and diagonal d) against its own (arr: their iterates), as a page's
  // 8h and 16h rows sweep inside it.
  const bulkSweeps=(arr:string,r:string,d:string)=>/* wgsl */`
  for(var s=0u;s<${schedule.sweeps}u;s++){
   for(var col=0u;col<2u;col++){
    // A closed region (d 0) has no row: its iterate stays 0, and no open row couples to it.
    if(lane<8u&&(countOneBits(lane)&1u)==col&&${d}>0.0){
     var off=${r};let c=bRegionCoord(lane);
     for(var f=0u;f<6u;f++){let j=lane^(1u<<(f/2u));if(c[f/2u]!=(f&1u)&&bsOpen(slot,j)){off+=bRegionK(f)*${arr}[j];}}
     ${arr}[lane]=off/${d};
    }
    workgroupBarrier();
   }
  }`;
  // The super-block tail (planUniformPressureSuper): 64h rows a 2^3 group of
  // pages, 128h rows a 4^3 group, keyed by page coordinate from the box's
  // low page. A level's row e: its key (list), its rows' channels (planes of
  // CAP words), its eight children (page blocks at 64h, 64h rows at 128h).
  const W=this.super,superBox=this.simulation.capacity.address.pool.box.origin;
  const superWGSL=W?/* wgsl */`
const S_BOX:vec3u=vec3u(${superBox[0]}u,${superBox[1]}u,${superBox[2]}u);const S_SCALE:f32=${schedule.superScale};
const CAP6:u32=${W.capacity6}u;const CAP7:u32=${W.capacity7}u;
const S_COUNT6:u32=${W.index.count6}u;const S_COUNT7:u32=${W.index.count7}u;const S_LIST6:u32=${W.index.list6}u;const S_LIST7:u32=${W.index.list7}u;const S_MAP6:u32=${W.index.map6}u;const S_MAP7:u32=${W.index.map7}u;
const S6_X:u32=${W.pool.x6}u;const S6_Y:u32=${W.pool.y6}u;const S6_R:u32=${W.pool.r6}u;const S6_D:u32=${W.pool.d6}u;const S6_K:u32=${W.pool.k6}u;const S6_C:u32=${W.pool.c6}u;
const S7_X:u32=${W.pool.x7}u;const S7_R:u32=${W.pool.r7}u;const S7_D:u32=${W.pool.d7}u;const S7_K:u32=${W.pool.k7}u;const S7_C:u32=${W.pool.c7}u;
// A tail item's (a page block, or BA | a BULK16 slot: tItem's) page
// coordinate from the box's low page (mod the window).
fn sRel(item:u32)->vec3i{${BP?"if((item&BA)!=0u){return vec3i((bitcast<vec3u>(bsPage(item&0x7fffffffu))+vec3u(64u)-S_BOX)&vec3u(63u));}":""}return vec3i((umBlockPage(item)+vec3u(64u)-S_BOX)&vec3u(63u));}
// The tail item of the page d (-1..1 an axis) from item's, UB_NONE_ID: none.
// tAt's paths (uniformPressureSuperNeighbour, the host twin): a block's
// level 2 record, else its neighbour word's BULK16 slot; a BULK16 page's
// page word, a stored block or a slot.
fn sNeighbour(item:u32,d:vec3i)->u32{
 let k=u32(13+d.x+3*d.y+9*d.z);
 ${BP?`var w=0u;
 if((item&BA)!=0u){w=umPageRawWord(bsPage(item&0x7fffffffu)+d);if(umBlockStored(w)){return w;}}
 else{let b=ubaLevelRecord(2u,item,k);if(b!=UB_NONE_ID){return b;}w=umBlockNeighbourWord(item,k);}
 let slot=bsWordSlot(w);return select(BA|slot,UB_NONE_ID,slot==0u);`:"return ubaLevelRecord(2u,item,k);"}
}
// The item a 32h cell address (tAt's) belongs to.
fn sItemOf(at:u32)->u32{${BP?"if(baIs(at)){return BA|baSlot(at);}":""}return (at-72u)/UBA_STRIDE;}
fn sPack6(g:vec3i)->u32{return u32(g.x)|(u32(g.y)<<5u)|(u32(g.z)<<10u);}
fn sPack7(g:vec3i)->u32{return u32(g.x)|(u32(g.y)<<4u)|(u32(g.z)<<8u);}
fn sKey6(e:u32)->vec3i{let k=pIndex(S_LIST6+e);return vec3i(vec3u(k&31u,(k>>5u)&31u,k>>10u));}
fn sKey7(e:u32)->vec3i{let k=pIndex(S_LIST7+e);return vec3i(vec3u(k&15u,(k>>4u)&15u,k>>8u));}
// The row of a group key (UB_NONE_ID: none, or outside the window).
fn sRow6(g:vec3i)->u32{if(any(g<vec3i(0))||any(g>=vec3i(32))){return UB_NONE_ID;}let v=pIndex(S_MAP6+sPack6(g));return select(v-1u,UB_NONE_ID,v==0u);}
fn sRow7(g:vec3i)->u32{if(any(g<vec3i(0))||any(g>=vec3i(16))){return UB_NONE_ID;}let v=pIndex(S_MAP7+sPack7(g));return select(v-1u,UB_NONE_ID,v==0u);}
fn sCount6()->u32{return select(min(pIndex(S_COUNT6),CAP6),0u,bConverged());}
fn sCount7()->u32{return select(min(pIndex(S_COUNT7),CAP7),0u,bConverged());}
fn sChild6(e:u32,j:u32)->u32{return poolB[S6_C+j*CAP6+e];}
fn sChild7(e:u32,j:u32)->u32{return poolB[S7_C+j*CAP7+e];}
fn sOctant(j:u32)->vec3u{return vec3u(j&1u,(j>>1u)&1u,j>>2u);}
fn sFace(f:u32)->vec3i{var d=vec3i(0);d[f/2u]=bSign(f);return d;}
// Face f of child j points at a sibling (inside the group): the sibling, else 8.
fn sSibling(j:u32,f:u32)->u32{let a=f/2u;return select(8u,j^(1u<<a),((j>>a)&1u)!=(f&1u));}
fn sParity(g:vec3i)->u32{return u32(g.x+g.y+g.z)&1u;}
fn sWeight(d:vec3i)->f32{return select(0.75,0.25,d.x==1)*select(0.75,0.25,d.y==1)*select(0.75,0.25,d.z==1);}
`:"";
  // The Krylov step (schedule.krylov): flexible GCR(1) with the V-cycle as
  // its preconditioner. kStart after the first cycle: the active set, its
  // frozen diagonal (walls linked or bounded as the iterate leaves them),
  // the true residual on the free rows, the step's start. Then after each
  // later cycle: kApply* z (the cycle's correction, X - KX, 0 off the free
  // set) to A z and the dots' partials, kReduce the step's beta and alpha,
  // kStep the direction p = z - beta p, its image q = A z - beta q, the
  // iterate KX + alpha p and the residual r - alpha q. Rows off the free set
  // keep their start: frozen, as the truncated coarse correction leaves
  // them. Their p is 0 but their q and r are tracked too (outside the dots),
  // so r is a frozen row's true residual, and a frozen row joins the free
  // set when it asks to leave its bound (pCorrectionFree, the cycle's rule).
  // Its p being 0, q = A p stays exact on the enlarged set: no restart.
  // KD is the row's linear diagonal, negative while it is frozen. The step
  // is feasible (Polyak 1969; O'Leary 1980): kBound finds the first free
  // bounded row the direction takes onto its bound, alpha stops there, and
  // that row is frozen on its bound. Its direction was not 0, so the next
  // step restarts (beta 0). It may join again when its residual asks. The band's iterate is plane kPlane,
  // where the cycle's post-smoothing ended. Everything reads the history's
  // stop, as the cycle.
  const KD=KRYLOV_DOTS,[segT,segN]=KRYLOV_SEGMENTS,krylovSlots=BP?Math.min(BP.layout.capacity,KRYLOV_SEGMENTS[2]):0;
  const krylovDots=(seg:number,group="group.x")=>/* wgsl */`
 // The workgroup's dots, one after another through kRed, into its partial.
 for(var d=0u;d<${KD}u;d++){
  kRed[lane]=kAcc[d];workgroupBarrier();
  for(var s=kWidth/2u;s>0u;s>>=1u){if(lane<s){kRed[lane]+=kRed[lane+s];}workgroupBarrier();}
  if(lane==0u){poolB[K_DOT+${KD}u*(${seg}u+${group})+d]=bitcast<u32>(kRed[0]);}
  workgroupBarrier();
 }`;
  // kApplyN's pages, e from its group by stride (its dispatch's width).
  const kApplyNLoop=(stride:string,group="group.x")=>/* wgsl */` for(var e=${group};e<n;e+=${stride}){
  let block=ubaLevelSlot(2u,e);let at2=UBA_STRIDE*block+lane;let d=qF(PN_KD,at2);let m=poolB[PN_M+at2];
  var az=0.0;var q=0.0;var r=0.0;let free=kFreeN(at2,d);
  if(d!=0.0||free){
   az=d*kZn(at2);
   for(var f=0u;f<6u;f++){
    let kind=nKind(m,f);let link=poolB[PN_L+f*PLANE2+at2];
    if(kind==N_NATIVE){az-=qF(PN_K+f*PLANE2,at2)*kZn(link);}
    else if(kind==N_SEAM){let g=f^1u;let axis=f/2u;let u=nStride((axis+1u)%3u);let v=nStride((axis+2u)%3u);
     for(var k=0u;k<16u;k++){let h=link+(k&3u)*u+(k>>2u)*v;if(bFaceKind(pool[P_KINDS+h],g)==K_SEAM){az-=pF(P_C+g*PLANE,h)*kZt(h);}}}${BP?`
    else if(kind==N_BULKN){az-=qF(PN_K+f*PLANE2,at2)*kZb(baSlot(link),link&15u);}`:""}
   }
   qSet(PN_KZ,at2,az);q=qF(PN_KQ,at2);r=qF(PN_KR,at2);
  }${B?`
  // A 16h bulk's members are one row (their correction is one): its
  // residual is theirs summed, every member's, as the verdict measures
  // their mean. Each member enters with the mean and weight 1/64: the
  // region's sum at its inverse volume, 1/4096, as a BULK16 region's.
  if(lane==0u){kFlags=nPage(block);}
  if((workgroupUniformLoad(&kFlags)&4u)!=0u){let bulk=(m&N_BULK)!=0u;az=nBulkMean(az,bulk,lane);q=nBulkMean(q,bulk,lane);r=nBulkMean(r,bulk,lane);}`:""}
  if(free){kAdd(az,q,r,1.0/64.0);}
  workgroupBarrier();
 }
`;
  const krylovWGSL=schedule.krylov?/* wgsl */`
const K_DOT:u32=${this.krylovDotWord}u;const K_ALPHA:u32=${KRYLOV_ALPHA_WORD}u;const K_BETA:u32=${KRYLOV_BETA_WORD}u;const K_L2:u32=${KRYLOV_L2_WORD}u;const K_JOIN:u32=${KRYLOV_JOIN_WORD}u;const K_CROSS:u32=${KRYLOV_CROSS_WORD}u;
const K_LIMIT:u32=${KRYLOV_LIMIT_WORD}u;const K_HIT:u32=${KRYLOV_HIT_WORD}u;const K_RESTART:u32=${KRYLOV_RESTART_WORD}u;const K_RESTARTS:u32=${KRYLOV_RESTARTS_WORD}u;
${BP?`const K_BULK:u32=${this.krylovBulkWord}u;
// A BULK16 slot's Krylov plane v (KRYLOV_PLANES' order) at region i.
fn kbAt(slot:u32,v:u32,i:u32)->u32{return K_BULK+${8*KRYLOV_PLANES.length}u*slot+8u*v+i;}
fn kbF(slot:u32,v:u32,i:u32)->f32{return bitcast<f32>(poolB[kbAt(slot,v,i)]);}
fn kbSet(slot:u32,v:u32,i:u32,x:f32){poolB[kbAt(slot,v,i)]=bitcast<u32>(x);}
// A region's z (every region is free: BULK16 rows have no bound).
fn kZb(slot:u32,i:u32)->f32{return bsF(slot,BS_N+i)-kbF(slot,0u,i);}`:""}
override kPlane:u32=0u;
override kCycle:u32=1u;
var<workgroup> kRed:array<f32,512>;
var<workgroup> kN:u32;
var<workgroup> kFlags:u32;
var<private> kAcc:array<f32,${KD}>;
fn kX()->u32{return select(P_X,P_X1,kPlane==1u);}
// A band row's z: its iterate less the step's start, 0 off the free set.
fn kZt(at:u32)->f32{if(pF(P_KD,at)<=0.0){return 0.0;}return pF(kX(),at)-pF(P_KX,at);}
// A native row in the free set. A 16h bulk's members are one row, the
// region's; their lanes' linear diagonals sum to its diagonal, and a member
// inside its region has 0 (its links there carry nothing). Members are
// unbounded (bulkSelect takes open cells only), so every member is free.
fn kFreeN(at2:u32,d:f32)->bool{return d>0.0${B?"||(d==0.0&&(poolB[PN_M+at2]&(N_BULK|N_ACTIVE))==(N_BULK|N_ACTIVE))":""};}
fn kZn(at2:u32)->f32{if(!kFreeN(at2,qF(PN_KD,at2))){return 0.0;}return qF(PN_X,at2)-qF(PN_KX,at2);}
// The dots (kReduce's order): Az.q, Az.Az, Az.r, q.q, q.r, r.r.
// Weighted by the row's inverse volume (an h row 1, a 4h 1/64, a 16h
// 1/4096): the rows' sums are volume totals, so the 2-norm minimised is the
// divergence field's over the liquid, as the verdict's max norm scales them.
fn kAdd(az:f32,q:f32,r:f32,w:f32){kAcc[0]+=w*az*q;kAcc[1]+=w*az*az;kAcc[2]+=w*az*r;kAcc[3]+=w*q*q;kAcc[4]+=w*q*r;kAcc[5]+=w*r*r;}
fn kClear(){for(var d=0u;d<${KD}u;d++){kAcc[d]=0.0;}}
fn kAlpha()->f32{return bitcast<f32>(pIndex(K_ALPHA));}
fn kBeta()->f32{return bitcast<f32>(pIndex(K_BETA));}
// The feasible step (Polyak; O'Leary): alpha cut at the first free bounded
// row its direction takes onto its bound (kBound's least ratio).
fn kStepAlpha()->f32{let a=kAlpha();return sign(a)*min(abs(a),bitcast<f32>(pIndex(K_LIMIT)));}
// A free bounded row's distance to its bound in units of its direction, when
// the step moves it toward the bound (else none).
fn kRoom(room:f32,dp:f32,alpha:f32)->f32{if(alpha*dp<0.0&&room>=0.0){return room/abs(dp);}return 3.0e38;}

// kStart, kBound and kStep each take every row once (their writes are the
// row's own, their reductions order-free atomics), so one dispatch runs the
// band's tiles (eight to a workgroup, 64 lanes each), the native pages and
// the BULK16 slots: kCounts() loads their counts.
var<workgroup> kNN:u32;
var<workgroup> kNB:u32;
fn kCounts(lane:u32)->vec3u{
 if(lane==0u){kN=pTileLive();kNN=select(ubaLevelCount(2u),0u,bConverged());${BP?"kNB=select(bsCount(),0u,bConverged());":"kNB=0u;"}}
 return vec3u(workgroupUniformLoad(&kN),workgroupUniformLoad(&kNN),workgroupUniformLoad(&kNB));
}
@compute @workgroup_size(${WG}) fn kStart(${lanes}){
 let n=kCounts(lane);let tl=lane&63u;let l=vec3u(tl&3u,(tl>>2u)&3u,tl>>4u);
 for(var j=8u*group.x+(lane>>6u);j<n.x;j+=8u*groups.x){
  let w=pIndex(TLIST+j);let slot=pTileSlot(w);let lb=pTileCell(w,l);let at=pAddr(slot,lb);let p=vec3i(ubaOrigin(ubaBrick(slot))+lb);
  let kinds=pool[P_KINDS+at];let x=pF(kX(),at);var r=0.0;var d=0.0;
  if(pActive(kinds)){
   let base=pF(P_BASE,at);var off=0.0;d=pF(P_DIAG,at);
   for(var f=0u;f<6u;f++){
    let kind=bFaceKind(kinds,f);
    if(kind==K_BAND){off+=pCoefficient(at,kinds,f)*pF(kX(),pNear(at,p,f));}
    else if(kind==K_SEAM){off+=pF(P_C+f*PLANE,at)*qF(PN_X,nAcross(p,f));}
    else if(kind==K_WALL){let c=pF(P_C+f*PLANE,at);let g=pLift(at,f);off+=c*max(x+g,-base);if(x+g>=-base){d-=c;}}
   }
   r=pF(P_RHS,at)+off-pF(P_DIAG,at)*x;
   d=select(-max(d,0.0),d,d>0.0&&pCorrectionFree(bInside(kinds),x+base,r));
  }
  pSet(P_KX,at,x);pSet(P_KQ,at,0.0);pSet(P_KZ,at,0.0);pSet(P_KP,at,0.0);pSet(P_KR,at,r);pSet(P_KD,at,d);
 }
 for(var e=group.x;e<n.y;e+=groups.x){
  let block=ubaLevelSlot(2u,e);let at2=UBA_STRIDE*block+lane;let m=poolB[PN_M+at2];let x=qF(PN_X,at2);var r=0.0;var d=0.0;
  if((nPage(block)&2u)!=0u&&(m&N_ACTIVE)!=0u){
   let base=qF(PN_B,at2);d=qF(PN_D,at2);r=qF(PN_R,at2)-d*x;
   for(var f=0u;f<6u;f++){
    let kind=nKind(m,f);let link=poolB[PN_L+f*PLANE2+at2];let c=qF(PN_K+f*PLANE2,at2);
    if(kind==N_NATIVE){r+=c*qF(PN_X,link);}
    else if(kind==N_SEAM){let g=f^1u;let axis=f/2u;let u=nStride((axis+1u)%3u);let v=nStride((axis+2u)%3u);
     for(var k=0u;k<16u;k++){let h=link+(k&3u)*u+(k>>2u)*v;if(bFaceKind(pool[P_KINDS+h],g)==K_SEAM){r+=pF(P_C+g*PLANE,h)*pF(kX(),h);}}}
    else if(kind==N_WALL){let g=bitcast<f32>(link);r+=c*max(x+g,-base);if(x+g>=-base){d-=c;}}${BP?`
    else if(kind==N_BULKN){r+=c*bsF(baSlot(link),BS_N+(link&15u));}`:""}
   }
   d=select(-max(d,0.0),d,d>0.0&&pCorrectionFree((m&N_CLOSED)!=0u,x+base,r));
  }
  qSet(PN_KX,at2,x);qSet(PN_KQ,at2,0.0);qSet(PN_KZ,at2,0.0);qSet(PN_KP,at2,0.0);qSet(PN_KR,at2,r);qSet(PN_KD,at2,d);
 }${BP?`
 for(var e=group.x;e<n.z;e+=groups.x){
  let slot=bsSlot(e);let r=bResidual(slot,lane);
  if(lane<8u){kbSet(slot,0u,lane,bsF(slot,BS_N+lane));kbSet(slot,1u,lane,0.0);kbSet(slot,2u,lane,0.0);kbSet(slot,3u,lane,0.0);kbSet(slot,4u,lane,r);kbSet(slot,5u,lane,bsF(slot,BS_T4+24u+lane));}
  workgroupBarrier();
 }`:""}
}
@compute @workgroup_size(${TILE_WG}) fn kApplyT(${lanes}){
 if(lane==0u){kN=pTileLive();}
 let n=workgroupUniformLoad(&kN);let l=vec3u(lane&3u,(lane>>2u)&3u,lane>>4u);kClear();
 let kWidth=${TILE_WG}u;
 for(var j=group.x;j<n;j+=groups.x){
  let w=pIndex(TLIST+j);let slot=pTileSlot(w);let lb=pTileCell(w,l);let at=pAddr(slot,lb);let p=vec3i(ubaOrigin(ubaBrick(slot))+lb);
  let d=pF(P_KD,at);
  if(d!=0.0){
   let kinds=pool[P_KINDS+at];var az=d*kZt(at);
   for(var f=0u;f<6u;f++){
    let kind=bFaceKind(kinds,f);
    if(kind==K_BAND){az-=pCoefficient(at,kinds,f)*kZt(pNear(at,p,f));}
    else if(kind==K_SEAM){az-=pF(P_C+f*PLANE,at)*kZn(nAcross(p,f));}
   }
   pSet(P_KZ,at,az);if(d>0.0){kAdd(az,pF(P_KQ,at),pF(P_KR,at),1.0);}
  }
 }
 if(n>0u){${krylovDots(0)}}
}
${BP?`var<workgroup> kLink:array<f32,48>;
// The BULK16 slots' and the native pages' kApply in one dispatch: the
// first K_SLOTS workgroups are kApplyB's, the rest kApplyN's (as many as
// its own dispatch), each with its own partials. The slots go first so
// their longer rows start in the first wave.
const K_SLOTS:u32=${krylovSlots}u;
@compute @workgroup_size(${WG}) fn kApplyNB(${lanes}){
 let gN=groups.x-K_SLOTS;
 if(lane==0u){kN=select(ubaLevelCount(2u),0u,bConverged());kNB=select(bsCount(),0u,bConverged());}
 let nN=workgroupUniformLoad(&kN);let nB=workgroupUniformLoad(&kNB);kClear();
 let kWidth=${WG}u;
 if(group.x>=K_SLOTS){
  let n=nN;
${kApplyNLoop("gN","(group.x-K_SLOTS)")}  if(n>0u){${krylovDots(segT,"(group.x-K_SLOTS)")}}
 }
 else{
  let n=nB;
 for(var e=group.x;e<n;e+=K_SLOTS){
  let slot=bsSlot(e);let page=bsPage(slot);
  // Lanes 0..47 a (region, face): the face's sum of k z across it.
  if(lane<48u){
   let i=lane/6u;let f=lane%6u;let near=bNear(slot,page,i,f);var s=0.0;
   if(near.kind==B_REGION){s=bRegionK(f)*kZb(near.slot,near.region);}
   else if(near.kind==B_BLOCK){for(var k=0u;k<16u;k++){let at=UBA_STRIDE*near.block+bTileLocal(bNativeTile(i,f,k));if(bLinked(at,f)){s+=qF(PN_K+(f^1u)*PLANE2,at)*kZn(at);}}}
   kLink[lane]=s;
  }
  workgroupBarrier();
  if(lane<8u){
   let o=6u*lane;let az=kbF(slot,5u,lane)*kZb(slot,lane)-(((kLink[o]+kLink[o+1u])+(kLink[o+2u]+kLink[o+3u]))+(kLink[o+4u]+kLink[o+5u]));
   kbSet(slot,2u,lane,az);kAdd(az,kbF(slot,1u,lane),kbF(slot,4u,lane),1.0/4096.0);
  }
  workgroupBarrier();
 }
  if(n>0u){${krylovDots(segT+segN)}}
 }
}`:`@compute @workgroup_size(${WG}) fn kApplyN(${lanes}){
 if(lane==0u){kN=select(ubaLevelCount(2u),0u,bConverged());}
 let n=workgroupUniformLoad(&kN);kClear();
 let kWidth=${WG}u;
${kApplyNLoop("groups.x")} if(n>0u){${krylovDots(segT)}}
}`}
// The step from the dots: beta orthogonalises A z against the last image
// (GCR), alpha minimises the new residual's 2-norm. A first step (q = 0)
// takes beta 0. No image (z = 0, or a block A annihilates) takes no step.
@compute @workgroup_size(256) fn kReduce(@builtin(local_invocation_index) lane:u32){
 if(lane==0u){kN=u32(bConverged());}
 if(workgroupUniformLoad(&kN)!=0u){return;}
 for(var d=0u;d<${KD}u;d++){
  var s=0.0;for(var g=lane;g<${KRYLOV_PARTIALS}u;g+=256u){s+=bitcast<f32>(poolB[K_DOT+${KD}u*g+d]);}
  kRed[lane]=s;workgroupBarrier();
  for(var t=128u;t>0u;t>>=1u){if(lane<t){kRed[lane]+=kRed[lane+t];}workgroupBarrier();}
  if(lane==0u){kAcc[d]=kRed[0];}
  workgroupBarrier();
 }
 if(lane==0u){
  let azq=kAcc[0];let azaz=kAcc[1];let azr=kAcc[2];let qq=kAcc[3];let qr=kAcc[4];let rr=kAcc[5];
  // A hit last step froze rows whose direction was not 0: restart (beta 0).
  let restart=pIndex(K_RESTART)!=0u;
  let beta=select(select(0.0,azq/qq,qq>0.0),0.0,restart);
  let nqq=azaz-2.0*beta*azq+beta*beta*qq;let nqr=azr-beta*qr;
  let alpha=select(0.0,nqr/nqq,nqq>0.0);
  atomicStore(&state[K_BETA],bitcast<u32>(beta));atomicStore(&state[K_ALPHA],bitcast<u32>(alpha));
  atomicStore(&state[K_LIMIT],bitcast<u32>(3.0e38));
  if(restart){atomicStore(&state[K_RESTART],0u);atomicAdd(&state[K_RESTARTS],1u);}
  atomicStore(&state[K_L2+kCycle-1u],bitcast<u32>(sqrt(max(rr,0.0))));
  atomicStore(&state[K_L2+kCycle],bitcast<u32>(sqrt(max(rr-2.0*alpha*nqr+alpha*alpha*nqq,0.0))));
 }
}
// The step's first bound hit: the least ratio over the free bounded rows.
@compute @workgroup_size(${WG}) fn kBound(${lanes}){
 let n=kCounts(lane);let tl=lane&63u;let l=vec3u(tl&3u,(tl>>2u)&3u,tl>>4u);
 let alpha=kAlpha();let beta=kBeta();
 for(var j=8u*group.x+(lane>>6u);j<n.x;j+=8u*groups.x){
  let w=pIndex(TLIST+j);let slot=pTileSlot(w);let lb=pTileCell(w,l);let at=pAddr(slot,lb);let kinds=pool[P_KINDS+at];
  if(pActive(kinds)&&bInside(kinds)&&pF(P_KD,at)>0.0){
   let t=kRoom(pF(P_KX,at)+pF(P_BASE,at),kZt(at)-beta*pF(P_KP,at),alpha);
   if(t<abs(alpha)){atomicMin(&state[K_LIMIT],bitcast<u32>(t));}
  }
 }
 for(var e=group.x;e<n.y;e+=groups.x){
  let block=ubaLevelSlot(2u,e);let at2=UBA_STRIDE*block+lane;let m=poolB[PN_M+at2];
  if((nPage(block)&2u)!=0u&&(m&N_ACTIVE)!=0u&&(m&N_CLOSED)!=0u&&qF(PN_KD,at2)>0.0){
   let t=kRoom(qF(PN_KX,at2)+qF(PN_B,at2),kZn(at2)-beta*qF(PN_KP,at2),alpha);
   if(t<abs(alpha)){atomicMin(&state[K_LIMIT],bitcast<u32>(t));}
  }
 }
}
@compute @workgroup_size(${WG}) fn kStep(${lanes}){
 let n=kCounts(lane);let tl=lane&63u;let l=vec3u(tl&3u,(tl>>2u)&3u,tl>>4u);
 let alpha=kStepAlpha();let beta=kBeta();
 for(var j=8u*group.x+(lane>>6u);j<n.x;j+=8u*groups.x){
  let w=pIndex(TLIST+j);let slot=pTileSlot(w);let lb=pTileCell(w,l);let at=pAddr(slot,lb);
  let kinds=pool[P_KINDS+at];
  if(!pActive(kinds)){continue;}
  let start=pF(P_KX,at);var x=start;let d=pF(P_KD,at);let base=pF(P_BASE,at);
  if(d>0.0){
   let dp=kZt(at)-beta*pF(P_KP,at);let q=pF(P_KZ,at)-beta*pF(P_KQ,at);x=start+alpha*dp;
   // The row the step's first hit puts on its bound: frozen there (its
   // direction 0, the next step restarts).
   let hit=bInside(kinds)&&kRoom(start+base,dp,alpha)<=abs(alpha);
   if(hit){x=-base;pSet(P_KD,at,-d);atomicAdd(&state[K_HIT],1u);atomicStore(&state[K_RESTART],1u);}
   pSet(P_KP,at,select(dp,0.0,hit));pSet(P_KQ,at,q);pSet(P_KR,at,pF(P_KR,at)-alpha*q);pSet(P_KX,at,x);
   if(bInside(kinds)&&start+base>=0.0&&x+base<0.0){atomicAdd(&state[K_CROSS],1u);}
  }
  else if(d<0.0){
   let q=pF(P_KZ,at)-beta*pF(P_KQ,at);let r=pF(P_KR,at)-alpha*q;pSet(P_KQ,at,q);pSet(P_KR,at,r);
   if(pCorrectionFree(bInside(kinds),start+base,r)){pSet(P_KD,at,-d);atomicAdd(&state[K_JOIN],1u);}
  }
  pSet(kX(),at,x);
 }
 for(var e=group.x;e<n.y;e+=groups.x){
  let block=ubaLevelSlot(2u,e);let at2=UBA_STRIDE*block+lane;
  let m=poolB[PN_M+at2];
  if((nPage(block)&2u)!=0u&&(m&N_ACTIVE)!=0u){
   let start=qF(PN_KX,at2);var x=start;let d=qF(PN_KD,at2);let base=qF(PN_B,at2);let bounded=(m&N_CLOSED)!=0u;
   if(kFreeN(at2,d)){
    let dp=kZn(at2)-beta*qF(PN_KP,at2);let q=qF(PN_KZ,at2)-beta*qF(PN_KQ,at2);x=start+alpha*dp;
    let hit=bounded&&kRoom(start+base,dp,alpha)<=abs(alpha);
    if(hit){x=-base;qSet(PN_KD,at2,-d);atomicAdd(&state[K_HIT],1u);atomicStore(&state[K_RESTART],1u);}
    qSet(PN_KP,at2,select(dp,0.0,hit));qSet(PN_KQ,at2,q);qSet(PN_KR,at2,qF(PN_KR,at2)-alpha*q);qSet(PN_KX,at2,x);
    if(bounded&&start+base>=0.0&&x+base<0.0){atomicAdd(&state[K_CROSS],1u);}
   }
   else if(d<0.0){
    let q=qF(PN_KZ,at2)-beta*qF(PN_KQ,at2);let r=qF(PN_KR,at2)-alpha*q;qSet(PN_KQ,at2,q);qSet(PN_KR,at2,r);
    if(pCorrectionFree(bounded,start+base,r)){qSet(PN_KD,at2,-d);atomicAdd(&state[K_JOIN],1u);}
   }
   qSet(PN_X,at2,x);
  }
 }${BP?`
 for(var e=group.x;e<n.z;e+=groups.x){
  let slot=bsSlot(e);
  if(lane<8u){
   let dp=kZb(slot,lane)-beta*kbF(slot,3u,lane);let q=kbF(slot,2u,lane)-beta*kbF(slot,1u,lane);let x=kbF(slot,0u,lane)+alpha*dp;
   kbSet(slot,3u,lane,dp);kbSet(slot,1u,lane,q);kbSet(slot,4u,lane,kbF(slot,4u,lane)-alpha*q);kbSet(slot,0u,lane,x);bsSet(slot,BS_N+lane,x);
  }
 }`:""}
}
`:"";
  const sources:Record<string,string>={
   table:uniformMixedTopologyWGSL(layout,0)+/* wgsl */`
@group(1) @binding(2) var<storage,read_write> state:array<atomic<u32>>;
const MASK:u32=${this.maskWord}u;
`+tableWGSL,
   mark:header()+owner+UniformColdFreeze.wgsl(this.cold,1,14)+(this.fields.warmTiers?uniformRegionHeatWGSL(1,25,this.simulation.capacity.tiles/64)+"fn bCoarseWarm(t:u32)->bool{return umRegionTierAtTile(t)==UM_TIER_WARM;}":"fn bCoarseWarm(t:u32)->bool{return false;}")+/* wgsl */`
var<workgroup> member:atomic<u32>;
// UniformPressureBand's list predicate: an h simulation tile with a liquid
// pressure row, neither frozen nor coarse WARM, takes a presentation slot
// and marks its brick (and its bit in the brick's mask).
@compute @workgroup_size(64) fn pMark(${lanes}){
 for(var j=group.x;j<umCounts.x;j+=groups.x){
  let t=bHTile(j);
  if(lane==0u){atomicStore(&member,0u);}
  workgroupBarrier();
  if(bWet(bPhiH(vec3i(umTileCoord(t)*4u+bLocal(lane))),1.0)){atomicStore(&member,1u);}
  if(workgroupUniformLoad(&member)!=0u&&lane==0u&&!umFrozenCell(vec3i(umTileCoord(t)*4u))&&!bCoarseWarm(t)){
   let slot=atomicAdd(&state[0],1u);
   if(slot<params.capacity.x){
    let c=vec3i(umTileCoord(t)*4u);let id=ubaBrickAt(c);
    let bit=pTileBit(vec3u(c)&vec3u(7u));
    atomicStore(&state[SLOTS+t],slot+1u);atomicStore(&state[TLIST+slot],(id<<3u)|bit);ubaMark(id,P_FINE);atomicOr(&state[MASK+id],1u<<bit);
   }else{atomicStore(&state[1],1u);}
  }
 }
}
// The last frame's marks back to zero (encodePrepare, before the header's
// clear): pMark sets no other word of the tile slot map or the brick masks,
// so the clear strides its band tile list, not the lattice. An entry is
// brick slot << 3 | tile bit after the rows; the slot holds its brick until
// ubaUpdate.
@compute @workgroup_size(64) fn pUnmark(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) groups:vec3u){
 for(var j=gid.x;j<pTileCount();j+=64u*groups.x){
  let w=pIndex(TLIST+j);let id=ubaBrick(pTileSlot(w));
  if(id>=${a.bricks}u){continue;}
  let l=id&63u;let b=w&7u;
  let t=512u*(id>>6u)+umTileLocal(2u*vec3u(l&3u,(l>>2u)&3u,l>>4u)+vec3u(b&1u,(b>>1u)&1u,b>>2u));
  atomicStore(&state[SLOTS+t],0u);atomicStore(&state[MASK+id],0u);
 }
}`,
   rows:header()+poolWGSL+theta+owner+UniformColdFreeze.wgsl(this.cold,1,14)+/* wgsl */`
@group(1) @binding(4) var<uniform> authority:vec4f;
@group(1) @binding(7) var velocity:texture_3d<f32>;
@group(1) @binding(8) var<storage,read> negative:array<f32>;
@group(1) @binding(29) var<storage,read> rootPhi:array<f32>;
// A tile across a band face is native where its root row is liquid (the
// root's own test, on the split authority's phi).
fn bNative(q:vec3i)->bool{return bWet(rootPhi[umTileAt(vec3u(q)/4u)],4.0);}
fn bForcedField(p:vec3i,axis:u32,sign:i32)->f32{
 var a=p;if(sign<0){a[axis]-=1;}
 if(umFaceLow(a,axis)){return negative[umLowFaceIndex(vec3u(p),axis)];}
 let raw=textureLoad(velocity,a,0)[axis];
 if(!umFrozenFace(a,axis,1u)){return raw;}
 // Forces retain F on frozen faces. The native RHS reconstructs F+T+g dt;
 // the band must use the same forced flux at their shared pressure seam.
 var q=p;q[axis]+=sign;var wet=bWet(bPhiH(p),1.0);
 if(umCellPresent(q)){wet=wet||bNeighbour(q).wet;}
 return umColdForced(a,axis,1u,raw,wet);
}
override bLocalSolidWork:bool=true;
${S?/* wgsl */`// open: umCellOpen(p), evaluated at the row's one site.
fn bHeadCapacity(p:vec3i,open:f32)->f32{
 if(bLocalSolidWork&&umSolidEnabled()){
  let tile=umTileKey(p>>vec3u(2u));
  if(umKeyStored(tile)){let open=umSolidRecordedOpen(tile,p);if(open>=0.0){return open;}}
 }
 return open;
}`:""}
${zero}
// UniformPressureBand's row assembly, a workgroup a band tile: the kinds,
// coefficients, forced faces${S?", face volumes":""} and RHS of its cells,
// and its list entry (from the brick id to the brick's slot). The cell's phi waits in
// the iterate for the start. A row whose slot held a liquid row last frame
// (and did not just enter) is warm.
// Two variants: pRows reads solid face apertures from the solid record only
// (no geometry inlined: the kernel stays lean) and leaves a tile whose
// apertures the record does not serve (umSolidAperturesServe) to pRowsG,
// which evaluates them, over that list.
const pGeometry:bool=false;
var<workgroup> pRoute:atomic<u32>;var<workgroup> pNeed:atomic<u32>;var<workgroup> pApertureRows:array<u32,4>;
@compute @workgroup_size(${TILE_WG}) fn pRows(${lanes}){
 if(lane==0u){pN=select(min(pIndex(0u),params.capacity.x),pIndex(GCOUNT),pGeometry);if(!pGeometry&&group.x==0u&&pIndex(UBA_FATAL)!=0u){atomicStore(&state[1],${SLOT_OVERFLOW}u);}}
 let n=workgroupUniformLoad(&pN);let l=vec3u(lane&3u,(lane>>2u)&3u,lane>>4u);
 for(var j=group.x;j<n;j+=groups.x){
  let k=select(j,pIndex(GLIST+j),pGeometry);
  let entry=pIndex(TLIST+k);let id=entry>>3u;let bit=entry&7u;let slot=ubaSlot(id);let mask=pIndex(MASK+id);
  // The brick's first band tile clears its other tiles' kinds: a tile is
  // warm only where it held a liquid row last frame.
  if(lane==0u){atomicStore(&pRoute,0u);atomicStore(&pNeed,0u);}
  if((mask&((1u<<bit)-1u))==0u){for(var d=0u;d<8u;d++){if(((mask>>d)&1u)==0u){pool[P_KINDS+pAddr(slot,pTileCell(d,l))]=0u;}}}
  workgroupBarrier();${S?`
  if(!pGeometry){
   // The tiles whose aperture rows its faces read (bit 0 its own, 1 + axis
   // the one below it on that axis), then each one's verdict on four lanes.
   let o=vec3i(ubaOrigin(id)+pTileCell(bit,vec3u(0u)));let c=o+vec3i(l);let clear=bLocalSolidWork&&umSolidRegionClearAtBuild(c);var need=0u;
   for(var f=0u;f<6u;f++){let axis=f/2u;var low=c;if(bSign(f)<0){low[axis]-=1;}var high=low;high[axis]+=1;
    if(!clear||!umSolidValid(low)||!umSolidValid(high)){need|=select(1u,2u<<axis,bSign(f)<0&&l[axis]==0u);}}
   if(need!=0u){atomicOr(&pNeed,need);}
   workgroupBarrier();
   if(lane<4u){var t=o;var axis=0u;if(lane>0u){axis=lane-1u;t[axis]-=1;}pApertureRows[lane]=umSolidApertureRows(t,axis);
    if(((atomicLoad(&pNeed)>>lane)&1u)!=0u){let serve=umSolidAperturesServe(umSolidApertureCell(t,axis));if(serve!=0u){atomicMax(&pRoute,serve);}}}
   let route=workgroupUniformLoad(&pRoute);
   if(route!=0u){
    if(lane==0u){if(route==2u){atomicStore(&state[1],${STALE_APERTURES}u);}else{atomicStore(&state[GLIST+atomicAdd(&state[GCOUNT],1u)],k);}}
    continue;
   }
  }`:""}
  if(lane==0u){atomicStore(&state[TLIST+k],(slot<<3u)|bit);}
  let lb=pTileCell(bit,l);let at=pAddr(slot,lb);
  let warm=(ubaClass(slot)&UBA_ENTERING)==0u&&bLiquid(pool[P_KINDS+at]);
  let p=vec3i(ubaOrigin(id)+lb);
  let own=bPhiH(p);let liquid=bWet(own,1.0);${S?`
  let solidClear=bLocalSolidWork&&umSolidRegionClearAtBuild(p);
  var shut=false;var inside=false;if(!solidClear){shut=!umCellIsOpen(p);inside=umCellInsideSolid(p);}
  // One umCellOpen site for p and one umCellIsOpen site for q (Metal inlines each).
  let open=umCellOpen(p);`:""}
  var kinds=select(0u,0x80000000u,liquid)${S?"|select(0u,0x40000000u,inside)|select(0u,0x20000000u,shut)":""};var diagonal=0.0;var divergence=0.0;
  ${S?`let bodies=liquid&&!solidClear&&umBodyCount()>0u&&umBodyPoseNearTile(umTileKey(p>>vec3u(2u)),${UNIFORM_BODY_FLUX_REACH}u);var capacity=0.0;
  if(bodies||!bLocalSolidWork){capacity=bHeadCapacity(p,open);}`:""}
  for(var f=0u;f<6u;f++){
   let axis=f/2u;let sign=bSign(f);let h=params.hDt[axis];var q=p;q[axis]+=sign;
   let forced=bForcedField(p,axis,sign);var kind=K_GHOST;var coefficient=0.0;var fraction=1.0;
   ${S?`var low=p;if(sign<0){low[axis]-=1;}var high=low;high[axis]+=1;
   var volume=1.0;if(!solidClear||!umSolidValid(low)||!umSolidValid(high)){if(pGeometry){volume=umPressureFaceVRecorded(low,axis);if(volume<0.0){atomicStore(&state[1],${STALE_APERTURES}u);volume=0.0;}}else{volume=umSolidApertureAt(pApertureRows[select(0u,1u+axis,sign<0&&l[axis]==0u)],low,axis);}}var sealed=false;`:""}
   var across=UM_RESIDENT;if(l[axis]==select(0u,3u,sign>0)){across=umCellKind(q);}
   if(across!=UM_RESIDENT){
    if(across==UM_AIR){kind=K_OPEN;coefficient=1.0/(h*h*bTheta(own,0.5*params.policy.w));}
    else{kind=K_WALL;coefficient=0.5/(h*h);fraction=0.5;}
   }
   // Frozen liquid still has native pressure rows. Keep their reciprocal
   // seam links: a Neumann face here also removes the native row's link
   // in pNative, disconnecting its gravity RHS from the live surface.
   else{
    let n=bNeighbour(q);
    var theta=1.0;${S?`
    // q's closure, where a row reads it: a face between liquid and air.
    var acrossShut=false;if(!solidClear&&liquid!=n.wet){acrossShut=!umCellIsOpen(q);}`:""}
    if(liquid!=n.wet){
     if(${S?"!solidClear&&(shut||acrossShut)":"false"}){theta=bPairTheta(own,liquid,n.phi,n.wet);}
     ${S?"else if(inside&&umCellInsideSolid(q)&&open<0.1&&umCellOpen(q)<0.1){theta=1.0;}":""}
     else{theta=bFaceTheta(bFacePhi(p,axis,sign),own,liquid,n.phi,n.wet);}
    }
    if(pBandCell(q)){kind=K_BAND;coefficient=1.0/(h*h*theta);}
    // A seam: the native 4h row's centre lies 2.5h away (a liquid cell meets
    // it at theta 1, an air one at its surface fraction).
    else if(bNative(q)){kind=K_SEAM;coefficient=1.0/(2.5*h*h*select(theta,1.0,liquid));}
    else if(n.neumann){kind=K_NEUMANN;}
    else{coefficient=1.0/(h*h*theta);}
    ${S?"if(!solidClear&&kind!=K_NEUMANN){sealed=(liquid&&shut&&!n.wet)||(!liquid&&n.wet&&acrossShut);}":""}
   }
   ${S?`if(volume<=1e-6||sealed){kind=K_CLOSED;coefficient=0.0;fraction=0.0;}
   else if(kind==K_NEUMANN){fraction=volume;}
   else{if(kind==K_WALL){coefficient=volume/(h*h);}else{coefficient*=volume;}fraction=volume;}
   qSet(P_V+f*PLANE,at,volume);`:""}
   kinds|=(kind<<(3u*f))|select(0u,1u<<(18u+f),kind==K_BAND&&coefficient==params.solve[1u+axis]);qSet(P_U+f*PLANE,at,forced);pSet(P_C+f*PLANE,at,coefficient);
   if(kind!=K_NEUMANN){diagonal+=coefficient;divergence+=f32(sign)*fraction*forced/h;}
   ${S?"if(bodies&&kind!=K_WALL&&kind!=K_OPEN&&(!bLocalSolidWork||capacity!=fraction)){divergence+=f32(sign)*(capacity-fraction)*umSolidFaceVelocity(low,axis)/h;}":""}
  }
  let control=authority.z*${S?"select(1.0,open,umSolidEnabled())":"1.0"};
  kinds|=select(0u,P_ACTIVE,liquid&&diagonal>0.0)|select(0u,P_WARM,warm);
  pSet(P_RHS,at,select(0.0,-params.policy.x*(divergence-control)/params.hDt.w,liquid));pool[P_KINDS+at]=kinds;pSet(P_X,at,own);
 }
}`,
   init:header(true)+poolWGSL+wet+UniformColdFreeze.wgsl(this.cold,1,14)+/* wgsl */`
@group(1) @binding(12) var<storage,read> coarsePressure:array<f32>;
@group(1) @binding(13) var<storage,read> coarsePhi:array<f32>;
fn bCoarseAt(home:u32,page:vec3i,c:vec3i)->vec3i{return umTileNearFromPage(home,page,c);}
fn bCoarse(q:vec3i)->vec3f{
 let i=umTileAt(vec3u(q));
 return select(vec3f(0.0),vec3f(coarsePressure[i],coarsePhi[i],1.0),bWet(coarsePhi[i],4.0)&&umTileOpenAt(q)>1e-5);
}
fn bCoarseKind(q:vec3i)->vec3f{
 let i=umTileAt(vec3u(q));
 let open=umTileOpenAt(q)>1e-5;let wet=bWet(coarsePhi[i],4.0);
 return vec3f(select(0.0,coarsePressure[i],open&&wet),f32(open&&wet),f32(open&&!wet));
}
// UniformPressureBand's carry of a row missing 4h owners from its stencil.
struct BCarry {rise:vec2f,height:vec2f,air:f32}
fn bCoarseCarry(home:u32,page:vec3i,b:vec3i,w:vec3f)->BCarry{
 var carry:BCarry;
 for(var k=0u;k<4u;k++){
  let bit=vec2u(k&1u,k>>1u);let sides=select(1.0-w.xz,w.xz,bit==vec2u(1u));let side=sides.x*sides.y;
  let c=b+vec3i(i32(bit.x),0,i32(bit.y));
  let t0=bCoarseAt(home,page,c);let t1=bCoarseAt(home,page,c+vec3i(0,1,0));let y0=t0.y;let y1=t1.y;
  let low=bCoarseKind(t0);var high=low;if(y1!=y0){high=bCoarseKind(t1);}
  let lower=side*(1.0-w.y);let upper=side*w.y;let liquid=lower*low.y+upper*high.y;
  carry.air+=lower*low.z+upper*high.z;
  carry.height+=vec2f(lower*low.y*f32(y0-b.y)+upper*high.y*f32(y1-b.y),liquid);
  if(y1!=y0&&low.y>0.0&&high.y>0.0){carry.rise+=liquid*vec2f(high.x-low.x,1.0);continue;}
  var own=low;var at=y0;var beyond=select(y0-1,y0+1,y0!=c.y);
  if(y1!=y0){if(high.y>0.0){own=high;at=y1;beyond=y1+1;}else{beyond=y0-1;}}
  if(own.y==0.0){continue;}
  let t2=bCoarseAt(home,page,vec3i(c.x,beyond,c.z));if(t2.y!=beyond){continue;}
  let other=bCoarseKind(t2);
  if(other.y>0.0){carry.rise+=liquid*vec2f((other.x-own.x)*f32(beyond-at),1.0);}
 }
 return carry;
}
${zero}
// The start: last frame's pressure on a warm row, else UniformPressureBand's
// carried trilinear 4h pressure (last frame's presented one); the iterate is
// the correction to it. A Neumann face (a frozen tile's) folds its forced
// flux and cold term into the RHS. pPicard (the kick's re-solve): the start
// is the last solve's pressure and the RHS pRebase's stash.
override pPicard:bool=false;
@compute @workgroup_size(${TILE_WG}) fn pInit(${lanes}){
 if(lane==0u){pN=pTileCount();}
 let n=workgroupUniformLoad(&pN);let l=vec3u(lane&3u,(lane>>2u)&3u,lane>>4u);
 for(var j=group.x;j<n;j+=groups.x){
  let w=pIndex(TLIST+j);let slot=pTileSlot(w);let lb=pTileCell(w,l);let at=pAddr(slot,lb);
  let p=vec3i(ubaOrigin(ubaBrick(slot))+lb);let kinds=pool[P_KINDS+at];
  if(!bLiquid(kinds)){pSet(P_X,at,0.0);pSet(P_X1,at,0.0);pSet(P_BASE,at,0.0);continue;}
  var start=0.0;
  if(pPicard){start=pF(P_BASE,at)+pF(P_X,at);}
  else if((kinds&P_WARM)!=0u){start=pF(P_WARMP,at);}
  else{
   let y=(vec3f(p)+0.5)/4.0-0.5;let b=vec3i(floor(y));let w=y-floor(y);var sum=vec3f(0.0);
   let home=umCellBlock(vec3u(p));let page=p>>vec3u(5u);
   for(var k=0u;k<8u;k++){
    let bit=vec3u(k&1u,(k>>1u)&1u,k>>2u);let weights=select(1.0-w,w,bit==vec3u(1u));
    sum+=weights.x*weights.y*weights.z*bCoarse(bCoarseAt(home,page,b+vec3i(bit)));
   }
   if(sum.z>=0.99999){start=sum.x/sum.z;}
   else if(sum.z>0.0){
    let carry=bCoarseCarry(home,page,b,w);
    if(carry.air>0.0||carry.rise.y==0.0){start=sum.x/sum.z*clamp(pF(P_X,at)*sum.z/min(sum.y,-1e-30),0.0,${(1/UNIFORM_MIXED_THETA_MIN).toFixed(1)});}
    else{
     let mean=sum.x/sum.z;let bound=${(1/UNIFORM_MIXED_THETA_MIN).toFixed(1)}*mean;
     start=clamp(mean+(w.y-carry.height.x/carry.height.y)*carry.rise.x/carry.rise.y,min(bound,0.0),max(bound,0.0));
    }
   }
  }
  if(bInside(kinds)){start=max(start,0.0);}
  pSet(P_BASE,at,start);pSet(P_X,at,0.0);pSet(P_X1,at,0.0);
  if(pPicard){continue;}
  var rhs=pF(P_RHS,at);
  for(var f=0u;f<6u;f++){
   let axis=f/2u;let sign=bSign(f);
   if(bFaceKind(kinds,f)==K_NEUMANN){
    var a=p;if(sign<0){a[axis]-=1;}
    rhs-=params.policy.x*f32(sign)*${S?"pVolume(at,f)*":""}(pForced(at,f)+umColdFaceTerm(a,axis,1u))/(params.hDt[axis]*params.hDt.w);}
  }
  pSet(P_RHS,at,rhs);
 }
}`,
   // After the 4h projection reaches the simulation: a non-band texel below
   // a band cell keeps its projected components for pProject, which writes
   // its band faces (the first band cell along x, then y, then z copies it).
   copy:header()+poolWGSL+zero+/* wgsl */`
@group(1) @binding(11) var velocity:texture_3d<f32>;
@group(1) @binding(23) var copied:texture_storage_3d<rgba32float,write>;
// Snapshot the projected field before the native seam return rewrites it.
// The return must retain projection on its other faces, not read u*.
// A job is 64 tiles of a block, a lane a tile: a 4h tile's lane copies its
// three texels, then the job's h tiles copy their 64 one after another. A
// job a tile left 61 of 64 lanes idle on every 4h tile of every block.
var<workgroup> pCopyFine:array<atomic<u32>,2>;
var<workgroup> pCopyFineTiles:vec2u;
@compute @workgroup_size(${TILE_WG}) fn pCopyNative(${lanes}){
 if(lane==0u){pN=ubaLevelCount(2u)*8u;}
 let n=workgroupUniformLoad(&pN);let l=vec3u(lane&3u,(lane>>2u)&3u,lane>>4u);
 for(var j=group.x;j<n;j+=groups.x){
  if(lane<2u){atomicStore(&pCopyFine[lane],0u);}
  workgroupBarrier();
  let first=512u*ubaLevelSlot(2u,j>>3u)+64u*(j&7u);let tile=first+lane;
  if(umTileWidth(tile)==1u){atomicOr(&pCopyFine[lane>>5u],1u<<(lane&31u));}
  else{let p=vec3i(4u*umTileCoord(tile));for(var a=0u;a<3u;a++){var q=p;q[a]+=3;textureStore(copied,q,/*4h*/textureLoad(velocity,q,0));}}
  workgroupBarrier();
  if(lane==0u){pCopyFineTiles=vec2u(atomicLoad(&pCopyFine[0]),atomicLoad(&pCopyFine[1]));}
  let fine=workgroupUniformLoad(&pCopyFineTiles);
  for(var k=0u;k<64u;k++){
   if(((fine[k>>5u]>>(k&31u))&1u)==0u){continue;}
   let q=vec3i(4u*umTileCoord(first+k))+vec3i(l);textureStore(copied,q,textureLoad(velocity,q,0));
  }
 }
}
@compute @workgroup_size(${TILE_WG}) fn pCopy(${lanes}){
 if(lane==0u){pN=pTileCount();}
 let n=workgroupUniformLoad(&pN);let l=vec3u(lane&3u,(lane>>2u)&3u,lane>>4u);
 for(var j=group.x;j<n;j+=groups.x){
  let w=pIndex(TLIST+j);let slot=pTileSlot(w);let lb=pTileCell(w,l);
  let p=vec3i(ubaOrigin(ubaBrick(slot))+lb);
  for(var axis=0u;axis<3u;axis++){
   var q=p;q[axis]-=1;if(!umCellPresent(q)||pBandCell(q)){continue;}
   var first=true;for(var b=0u;b<axis;b++){var r=q;r[b]+=1;if(pBandCell(r)){first=false;}}
   if(first){textureStore(copied,q,textureLoad(velocity,q,0));}
  }
 }
}`,
   // The native rows (level 2: the root's own rows, key 512 block + cell), a
   // workgroup a page, a lane a cell: every liquid root row outside the band,
   // 64x the root's operator (volume / h^3) about its start (the warm root
   // pressure). A face across a band tile is a seam: the sum of its 16 h
   // faces' couplings, toward their iterate. Writes the page's flags, each
   // native row's level 2 operator and the initial residual.
   native:header(true)+poolWGSL+store+wet+theta+/* wgsl */`
@group(1) @binding(26) var<storage,read_write> rootArena:array<f32>;
@group(1) @binding(29) var<storage,read> rootPhi:array<f32>;
${S?"@group(1) @binding(30) var<storage,read> rootTopology:array<vec4f>;":""}
var<workgroup> nFlags:atomic<u32>;
var<workgroup> nWorst:atomic<u32>;
var<workgroup> nN:u32;
var<workgroup> nPageFlags:u32;
fn nLiquid(k:u32)->bool{return bWet(rootPhi[k],4.0);}
${bulkNative}// umTheta of liquid row k and air row n across face (axis, side).
fn nTheta(k:u32,n:u32,axis:u32,side:u32)->f32{
 ${S?`let low=select(n,k,side==1u);if(arrayLength(&rootTopology)>N_THETA+low){let t=rootTopology[N_THETA+low][axis];if(t>0.0){return t;}}`:""}
 return bTheta(rootPhi[k],rootPhi[n]);
}
// The pool offset of the first of band tile t's 16 h cells on its face g.
fn nSeamFirst(t:u32,g:u32)->u32{
 var c=umTileCoord(t)*4u;if((g&1u)==1u){c[g/2u]+=3u;}
 let q=vec3i(c);return pAddr(ubaSlot(ubaBrickAt(q)),vec3u(q)&vec3u(7u));
}
@compute @workgroup_size(${WG}) fn pNative(${lanes}){
 if(lane==0u){nN=ubaLevelCount(2u);atomicStore(&nWorst,0u);if(group.x==0u){atomicOr(&state[VALID],INITIAL_BIT);}}
 let count=workgroupUniformLoad(&nN);
 let l=vec3u(lane&7u,(lane>>3u)&7u,lane>>6u);
 for(var e=group.x;e<count;e+=groups.x){
  let block=ubaLevelSlot(2u,e);let k=512u*block+lane;let at2=UBA_STRIDE*block+lane;
  if(lane==0u){atomicStore(&nFlags,0u);${B?"var any=0u;for(var r=0u;r<8u;r++){any|=poolB[N_RECORD+8u*block+r];}nBulkW=any&1u;":""}}
  workgroupBarrier();
  var m=0u;var start=0.0;var rhs=0.0;var diagonal=0.0;var walls=0.0;var lifted=0.0;
${B?"  let bulkPage=workgroupUniformLoad(&nBulkW)!=0u;let member=bulkPage&&nMember(k);var bulkDelta=0.0;var rInit=0.0;\n":""}  if(pIndex(SLOTS+k)!=0u){atomicOr(&nFlags,1u);}
  else if(nLiquid(k)){
   start=rootArena[R_PRESSURE+k];rhs=64.0*rootArena[R_RHS+k];
   for(var f=0u;f<6u;f++){
    let axis=f/2u;let side=f&1u;let stride=nStride(axis);let h4=4.0*params.hDt[axis];
    var kind=N_NONE;var c=0.0;var link=0u;var level2=0.0;
${B?"    if(member&&(umTileCoord(k)[axis]&3u)==select(0u,3u,side==1u)){poolB[N_RECORD+nHeadIndex(k,f)]=0x7f61b1e6u;}\n":""}    var n=select(k-stride,k+stride,side==1u);var across=block;
    if(l[axis]==select(0u,7u,side==1u)){
     let step=select(select(9u,3u,axis==1u),1u,axis==0u);
     across=umBlockNeighbourWord(block,select(13u-step,13u+step,side==1u));
     n=512u*min(across,N_SENTINEL)+(select(k+7u*stride,k-7u*stride,side==1u)&511u);
    }
    if(across>=N_SENTINEL){
     // A SOLID page: the root's wall slot, free (held) or bounded at 0.
     let s=N_KEYS+umFaceTile(block,f,l);let fraction=${S?"rootTopology[s].x":"0.5"};
${BP?`     let bslot=bsWordSlot(across);var t=l;t[axis]=select(7u,0u,side==1u);let region=bsRegion(t);
     // A closed region across (bsOpen) is a static wall: the SOLID page's wall slot below.
     if(bslot!=0u&&bsOpen(bslot,region)){
      // A BULK16 page: its region's 16h row across the face, 10h from this
      // row's centre (member to member 16h, as nBulkLink's). The slot's
      // row sees this row's start less its head (a member's: its region's
      // start), q, which the wall slot's bound word keeps for pBulkRows
      // and pWrite; buildRhs left the face's flux in its RHS word.
      if(fraction>1e-6){
       let w=fraction/(h4*h4);
       var q=start-nBulkHead(k,axis,false);c=25.6*w;
       if(member){q=nBulkStart(k);c=16.0*w;}
       kind=N_BULKN;link=baOf(bslot,false,region);level2=c;diagonal+=c;
       rhs+=c*(bsF(bslot,BS_P+region)-q);rootArena[R_MINIMUM+s]=q;atomicOr(&nFlags,8u);
      }
     }else `:""}     if(fraction>1e-6){
      let w=fraction/(h4*h4);let lift=rootArena[R_RHS+s]/w;c=64.0*w;
      if(rootArena[R_MINIMUM+s]<-1e38){kind=N_HELD;rhs+=c*lift;}
      else{kind=N_WALL;diagonal+=c;walls+=c;lifted+=c*max(lift,-start);link=bitcast<u32>(lift);}
     }
    }else{
     let volume=${S?"select(rootTopology[n][axis+1u],rootTopology[k][axis+1u],side==1u)":"1.0"};
     if(volume>1e-6){
      if(pIndex(SLOTS+n)!=0u){
       // A seam: the band tile's h faces toward this row.
       kind=N_SEAM;let g=f^1u;let first=nSeamFirst(n,g);link=first;
       let u=nStride((axis+1u)%3u);let v=nStride((axis+2u)%3u);
       for(var j=0u;j<16u;j++){
        let at=first+(j&3u)*u+(j>>2u)*v;
        if(bFaceKind(pool[P_KINDS+at],g)==K_SEAM){
         let ch=pF(P_C+g*PLANE,at);c+=ch;rhs+=ch*(pF(P_BASE,at)-bSeamHead(at,g)-start);
         // An air subface contributes a zero-pressure boundary to the
         // diagonal, not a coupling to the band's liquid aggregate.
         if(pActive(pool[P_KINDS+at])){level2+=ch;}
        }
       }
       diagonal+=c;
      }else if(nLiquid(n)){${B?
       "kind=N_NATIVE;level2=64.0*volume/(h4*h4);let b=nBulkLink(k,n,f,member);c=b.x*volume/(h4*h4);bulkDelta+=level2-c;diagonal+=c;rhs+=c*(nBulkStart(n)-select(start,nBulkStart(k),member)+b.y);link=UBA_STRIDE*(n>>9u)+(n&511u);":
       "kind=N_NATIVE;c=64.0*volume/(h4*h4);diagonal+=c;level2=c;rhs+=c*(rootArena[R_PRESSURE+n]-start);link=UBA_STRIDE*(n>>9u)+(n&511u);"}}
      else{kind=N_GHOST;c=64.0*volume/(h4*h4*nTheta(k,n,axis,side));diagonal+=c;rhs-=c*start;}
     }
    }
    m|=kind<<(3u*f);
    qSet(PN_K+f*PLANE2,at2,c);poolB[PN_L+f*PLANE2+at2]=link;qSet(PB_K+f*PLANE2,at2,level2);
   }
   if(diagonal${B?"+bulkDelta":""}>0.0){m|=N_ACTIVE;atomicOr(&nFlags,3u);}
   if(rootArena[R_MINIMUM+k]>-1e38){m|=N_CLOSED;}
${B?"   if(member){m|=N_BULK;}\n":""}  }
  poolB[PN_M+at2]=m;qSet(PN_X,at2,0.0);qSet(PN_B,at2,start);qSet(PN_R,at2,rhs);qSet(PN_D,at2,diagonal);
  if((m&N_ACTIVE)!=0u){
   // Level 2's correction operator: the walls follow their row.
   qSet(PB_D,at2,diagonal${B?"+bulkDelta":""}-walls);
   var r=rhs+lifted;if((m&N_CLOSED)!=0u&&start<=0.0&&bResidualMagnitude(r)<0x7f800000u){r=max(r,0.0);}
   ${B?"rInit=r;if((m&N_BULK)==0u){atomicMax(&nWorst,bResidualMagnitude(r*nScale()));pDiag(7u,pDiagNative(m),bResidualMagnitude(r*nScale()));}pDiag(6u,select(pDiagNative(m),3u,(m&N_BULK)!=0u),bResidualMagnitude(1.2e-7*(abs(64.0*rootArena[R_RHS+k])+2.0*diagonal*abs(start))*nScale()));":"atomicMax(&nWorst,bResidualMagnitude(r*nScale()));pDiag(7u,pDiagNative(m),bResidualMagnitude(r*nScale()));"}
  }
  workgroupBarrier();
${B?`  // A member region's residual is its 16h row's: the mean of its lanes'.
  if(bulkPage){let mean=nBulkMean(rInit,(m&N_BULK)!=0u,lane);if((m&N_BULK)!=0u){atomicMax(&nWorst,bResidualMagnitude(mean*nScale()));pDiag(7u,3u,bResidualMagnitude(mean*nScale()));}if(lane==0u){atomicOr(&nFlags,4u);}workgroupBarrier();}
`:""}  if(lane==0u){poolB[PN_M+UBA_STRIDE*block+512u]=atomicLoad(&nFlags);}
 }
 workgroupBarrier();
 // The start meets the root's bound only under its floor: a tenth of itself it never is.
 if(lane==0u){let r=atomicLoad(&nWorst);if(r!=0u){atomicMax(&state[INITIAL],select(r,0x7f7fffffu,bitcast<f32>(r)>nFloor()));atomicMax(&state[NATIVE_INITIAL],r);}}
}
// Rebuild the native correction rows after both residual passes published
// their active sets. Links into held rows remain in the diagonal, with zero
// correction on their other side; physical rows and smoothers stay intact.
@compute @workgroup_size(${WG}) fn pNativeOperator(${lanes}){
 if(lane==0u){nN=select(ubaLevelCount(2u),0u,bConverged());}
 let count=workgroupUniformLoad(&nN);
 for(var e=group.x;e<count;e+=groups.x){
  let block=ubaLevelSlot(2u,e);let at=UBA_STRIDE*block+lane;let m=poolB[PN_M+at];
  if((m&N_ACTIVE)==0u){continue;}
  let free=(m&N_CORRECT)!=0u;var d=qF(PN_D,at);let x=qF(PN_X,at);let base=qF(PN_B,at);
  for(var f=0u;f<6u;f++){
   let kind=nKind(m,f);let c=qF(PN_K+f*PLANE2,at);let link=poolB[PN_L+f*PLANE2+at];var k=0.0;
   if(kind==N_NATIVE){
    k=c;
${B?`    let key=512u*block+lane;let other=512u*(link/UBA_STRIDE)+(link%UBA_STRIDE);let axis=f/2u;
    let volume=${S?"select(rootTopology[other][axis+1u],rootTopology[key][axis+1u],(f&1u)==1u)":"1.0"};
    let h4=4.0*params.hDt[axis];k=64.0*volume/(h4*h4);d+=k-c;
`:""}    if((poolB[PN_M+link]&N_CORRECT)==0u){k=0.0;}
   }else if(kind==N_SEAM){
    let g=f^1u;let axis=f/2u;let u=nStride((axis+1u)%3u);let v=nStride((axis+2u)%3u);
    for(var j=0u;j<16u;j++){let row=link+(j&3u)*u+(j>>2u)*v;let kinds=pool[P_KINDS+row];
     if(bFaceKind(kinds,g)==K_SEAM&&(kinds&P_CORRECT)!=0u){k+=pF(P_C+g*PLANE,row);}
    }
   }else if(kind==N_WALL&&x+bitcast<f32>(link)>=-base){d-=c;}
   qSet(PB_K+f*PLANE2,at,select(0.0,k,free));
  }
  qSet(PB_D,at,select(0.0,d,free));
 }
}
// The native rows' pressures into the root (the 4h projection's), and their wall slots'.
@compute @workgroup_size(${WG}) fn pWrite(${lanes}){
 if(lane==0u){nN=ubaLevelCount(2u);}
 let count=workgroupUniformLoad(&nN);
 let l=vec3u(lane&7u,(lane>>3u)&7u,lane>>6u);
 for(var e=group.x;e<count;e+=groups.x){
  let block=ubaLevelSlot(2u,e);
  if(lane==0u){nPageFlags=nPage(block);}
  if((workgroupUniformLoad(&nPageFlags)&2u)!=0u){
   let at2=UBA_STRIDE*block+lane;let m=poolB[PN_M+at2];
   if((m&N_ACTIVE)!=0u){
    let p=qF(PN_B,at2)+qF(PN_X,at2);rootArena[R_PRESSURE+512u*block+lane]=p;
    for(var f=0u;f<6u;f++){
     let kind=nKind(m,f);
     if(kind==N_WALL||kind==N_HELD){
      let s=N_KEYS+umFaceTile(block,f,l);let wall=p+64.0*rootArena[R_RHS+s]/qF(PN_K+f*PLANE2,at2);
      rootArena[R_PRESSURE+s]=select(max(wall,0.0),wall,kind==N_HELD);
     }${BP?`else if(kind==N_BULKN){
      // The 4h projection reads a wall slot 4h across the face: the BULK16
      // row's pressure in this row's frame, its difference scaled to the
      // row's distance (10h, theta 2.5; a member's 16h, 4), as pNative's c.
      let s=N_KEYS+umFaceTile(block,f,l);let link=poolB[PN_L+f*PLANE2+at2];let r=link&15u;let slot=baSlot(link);
      let across=bsF(slot,BS_P+r)+bsF(slot,BS_N+r)+qF(PN_B,at2)-rootArena[R_MINIMUM+s];
      rootArena[R_PRESSURE+s]=p+(across-p)/select(2.5,4.0,nMember(512u*block+lane));
     }`:""}
    }
   }
  }
  workgroupBarrier();
 }
}`,
   // The native rows' RHS from the simulation's u* (UNIFORM_PRESSURE_ARENA_NATIVE_RHS),
   // a workgroup a page, a lane a row: the root buildRhs's rows in its order
   // (ULP level: the compilers round its divisions apart), for the liquid
   // rows outside the band only. A face's
   // velocity is the transfer to pressure's of its low tile's +face (the 4h
   // anchor, or the mean of its sixteen plane faces against an h tile) or of
   // the row's negative domain wall; a cut face's flux, and a statically cut
   // tile's V-weighted mean, sum sixteen h faces, the workgroup's (a lane an
   // h face), a face direction at a time. Writes each row's RHS, bound and
   // clamped start, its wall slots', and zeroes the air rows' starts.
   nrhs:header(true)+wet+UniformColdFreeze.wgsl(this.cold,1,14)+/* wgsl */`
@group(1) @binding(4) var<uniform> authority:vec4f;
@group(1) @binding(7) var velocity:texture_3d<f32>;
@group(1) @binding(8) var<storage,read> negative:array<f32>;
@group(1) @binding(26) var<storage,read_write> rootArena:array<f32>;
@group(1) @binding(29) var<storage,read> rootPhi:array<f32>;
const N_MASKS:u32=${this.pressure.layout.cellCount}u;
const N_H=vec3f(${this.simulation.layout.lattice.cellSize_m.join(",")});
const N_LISTED:u32=0x40000u;const N_WEIGHTED:u32=0x20000u;
var<workgroup> nN:u32;
var<workgroup> nCount:atomic<u32>;
var<workgroup> nItem:array<u32,${WG}>;var<workgroup> nWord:array<u32,${WG}>;
var<workgroup> nPart:array<vec2f,${WG}>;var<workgroup> nSum:array<vec2f,${WG}>;
fn nLiquid(k:u32)->bool{return bWet(rootPhi[k],4.0);}
// How the transfer to pressure forms tile T's +face on axis a: 0 the 4h
// anchor, 1 the mean of its sixteen plane faces (an h tile, or a 4h tile
// below an h one), 2 their V-weighted mean (a statically cut tile).
fn nFaceMode(T:vec3u,a:u32)->u32{
 let t=umTileAt(T);let cut=umSolidEnabled()&&umSolidStaticCut(t);
 if(umTileWidth(t)==1u){return select(1u,2u,cut);}
 var n=vec3i(T);n[a]+=1;
 if(!umTilePresent(n)||umTileWidth(umTileAt(vec3u(n)))!=1u){return 0u;}
 return select(1u,2u,cut);
}
fn nAnchor(q:vec3i,a:u32)->f32{return ${UNIFORM_DETAIL_4H_LOAD}textureLoad(velocity,q,0)[a];}
// Sixteen faces summed in the transfer's order (tMean): tile T's +plane, or (wall) its negative domain wall.
fn nMean(T:vec3u,a:u32,wall:bool)->f32{
 let u=(a+1u)%3u;let v=(a+2u)%3u;var value=0.0;
 for(var y=0u;y<4u;y++){for(var x=0u;x<4u;x++){
  var q=T*4u;q[u]+=x;q[v]+=y;
  if(wall){value+=negative[umLowFaceIndex(q,a)];}else{q[a]+=3u;value+=textureLoad(velocity,vec3i(q),0)[a];}
 }}
 return value/16.0;
}
// h face j (i+4j) of row k's face f: V_h u_h (plus a moving wall's term)
// and V_h, the root's umCutFace; a weighted mean's part with the mask whole.
fn nPart16(k:u32,f:u32,j:u32,word:u32)->vec2f{
 if(((word>>j)&1u)==0u){return vec2f(0.0);}
 let axis=f/2u;let sign=bSign(f);let a=(axis+1u)%3u;let b=(axis+2u)%3u;
 var c=vec3i(umTileCoord(k)*4u);c[a]+=i32(j&3u);c[b]+=i32(j>>2u);c[axis]+=select(-1,3,sign>0);
 let low=umFaceLow(c,axis);
 var u=0.0;if(low){u=negative[umLowFaceIndex(umLowFaceAbove(c,axis),axis)];}else{u=textureLoad(velocity,c,0)[axis];}
 var volume=0.0;
 if(umSolidEnabled()&&!low){volume=umPatchFaceV(umTileAt(vec3u(c)/4u),c,axis);}else{volume=umPressureFaceV(c,axis);}
 var sum=volume*u;
 if((word&0x10000u)!=0u){var inner=c;if(sign<0){inner[axis]+=1;}let capacity=umCellOpen(inner);if(capacity!=volume){sum+=(capacity-volume)*umSolidFaceVelocity(c,axis);}}
 return vec2f(sum,volume);
}
@compute @workgroup_size(${WG}) fn pRhs(${lanes}){
 if(lane==0u){nN=ubaLevelCount(2u);}
 let count=workgroupUniformLoad(&nN);
 let l=vec3u(lane&7u,(lane>>3u)&7u,lane>>6u);
 let held=-4.0*min(N_H.x,min(N_H.y,N_H.z));
 for(var e=group.x;e<count;e+=groups.x){
  let block=ubaLevelSlot(2u,e);let k=512u*block+lane;
  let T=umTileCoord(k);let O=vec3i(T*4u);
  let liquid=nLiquid(k);let row=liquid&&pIndex(SLOTS+k)==0u;
  // Each face: its velocity, or its sixteen listed (N_LISTED; N_WEIGHTED a weighted mean's).
  var words:array<u32,6>;var velocities:array<f32,6>;
  if(row){for(var f=0u;f<6u;f++){
   let axis=f/2u;let sign=bSign(f);var n=vec3i(T);n[axis]+=sign;let kind=umTileKind(n);
   var cut=umSolidCut(k);if(!cut&&umTilePresent(n)){cut=umSolidCut(umTileAt(vec3u(n)));}
   var word=0u;var source=T;
   if(cut){
    var t=vec3i(T);if(sign<0){t=n;}
    var mask=0xffffu;if(kind==UM_RESIDENT){mask=u32(rootPhi[N_MASKS+${UNIFORM_MIXED_ROOT_TILE_WORDS}u*umTileAt(vec3u(t))+axis])&0xffffu;}
    word=N_LISTED|mask|select(0u,0x10000u,umBodyCount()>0u&&umBodyPoseNearTile(k,${UNIFORM_BODY_FLUX_REACH}u));
   }else if(sign>0||kind==UM_RESIDENT){
    if(sign<0){source=vec3u(n);}
    let mode=nFaceMode(source,axis);var q=vec3i(source*4u);q[axis]+=3;
    if(mode==2u){word=N_LISTED|N_WEIGHTED|0xffffu;}else if(mode==1u){velocities[f]=nMean(source,axis,false);}else{velocities[f]=nAnchor(q,axis);}
   }else if(kind>=UM_SOLID){
    if(umTileWidth(k)==4u){velocities[f]=negative[umLowFaceIndex(T*4u,axis)];}
    else if(umSolidEnabled()&&umSolidStaticCut(k)){word=N_LISTED|N_WEIGHTED|0xffffu;}
    else{velocities[f]=nMean(T,axis,true);}
   }else{
    // An AIR page below: the root's anchor texel there.
    var q=O;q[axis]-=1;velocities[f]=nAnchor(q,axis);
   }
   words[f]=word;
  }}
  var sums:array<vec2f,6>;
  for(var f=0u;f<6u;f++){
   workgroupBarrier();
   if(lane==0u){atomicStore(&nCount,0u);}
   workgroupBarrier();
   if(row&&words[f]!=0u){let i=atomicAdd(&nCount,1u);nItem[i]=lane;nWord[i]=words[f];}
   let listed=workgroupUniformLoad(&nCount);
   for(var first=0u;first<listed;first+=${WG/16}u){
    let item=first+lane/16u;var part=vec2f(0.0);
    if(item<listed){part=nPart16(512u*block+nItem[item],f,lane&15u,nWord[item]);}
    nPart[lane]=part;workgroupBarrier();
    if(lane<${WG/16}u&&first+lane<listed){var sum=vec2f(0.0);for(var j=0u;j<16u;j++){sum+=nPart[16u*lane+j];}nSum[nItem[first+lane]]=sum;}
    workgroupBarrier();
   }
   if(row&&words[f]!=0u){sums[f]=nSum[lane];}
  }
  if(row){
   var terms:array<f32,6>;
   for(var f=0u;f<6u;f++){
    let axis=f/2u;let sign=bSign(f);var n=vec3i(T);n[axis]+=sign;let kind=umTileKind(n);
    // COLD freeze: what u* adds to the stored face (the root's umColdAdd).
    var low=O;if(sign<0){low[axis]-=4;}
    var anchor=O;anchor[axis]+=select(-1,3,sign>0);
    // A BULK16 page across is liquid: its face takes gravity and T as any
    // frozen face (a wall, SOLID, takes neither).
    var add=umColdForced(low,axis,4u,0.0,!umFaceLow(anchor,axis)||kind==UM_BULK16);
    if(kind==UM_SOLID){add-=umColdFaceTerm(low,axis,4u);}
    if(!UM_COLD_FREEZE){add=0.0;}
    let w=words[f];let cut=(w&(N_LISTED|N_WEIGHTED))==N_LISTED;let halo=N_KEYS+umFaceTile(block,f,l);
    var face=velocities[f];if((w&N_WEIGHTED)!=0u){let s=sums[f];face=s.x;if(s.y>0.0){face=s.x/s.y;}}
    var flux=sums[f];
    if(!cut){
     var recordV=1.0;
     if(kind>=UM_SOLID){recordV=umSolidCoarse(halo).x;}else if(kind==UM_RESIDENT){recordV=umSolidCoarse(select(umTileAt(vec3u(n)),k,sign>0))[axis+1u];}
     let weight=recordV*16.0;flux=vec2f(weight*face,weight);
    }
    var total=0.0;total+=flux.x+flux.y*add;
    terms[f]=f32(sign)*total/(64.0*N_H[axis]);
    if(kind>=UM_SOLID){
     // The wall slot (a SOLID or BULK16 page across, as pNative's N_SENTINEL
     // test): its one-cell halo's RHS, bound and start.
     var wall=0.0;if(cut){wall=(flux.x+flux.y*add)/16.0;}else{wall=umSolidCoarse(halo).x*(face+add);}
     let b=f32(sign)*params.policy.x*wall/(4.0*N_H[axis]*params.hDt.w);let bound=select(0.0,-3.402823e38,rootPhi[k]<held);
     rootArena[R_RHS+halo]=b;rootArena[R_MINIMUM+halo]=bound;rootArena[R_PRESSURE+halo]=max(rootArena[R_PRESSURE+halo],bound);
    }
   }
   let divergence=(terms[0]+terms[1])+(terms[4]+terms[5])+(terms[2]+terms[3]);
   let topology=umSolidCoarse(k);
   let value=-params.policy.x*(divergence-authority.z*topology.x)/params.hDt.w;
   let minimum=select(-3.402823e38,0.0,topology.x<=1e-5);
   rootArena[R_RHS+k]=value;rootArena[R_MINIMUM+k]=minimum;rootArena[R_PRESSURE+k]=max(rootArena[R_PRESSURE+k],minimum);
  }else if(!liquid){rootArena[R_PRESSURE+k]=0.0;}
 }
}`,
   // A native row's composite residual: each cycle's after the band's
   // pre-smoothing (level 2's RHS; its largest into the cycle's history
   // word), and the final measure (pLast).
   residual:header()+poolWGSL+store+(B?bulkMean("rRed")+"var<workgroup> rRed:array<f32,512>;\n":"")+(BP?bulkRows:"")+/* wgsl */`
override pCycle:u32=0u;
override pLast:bool=false;
var<workgroup> worst:atomic<u32>;
var<workgroup> rN:u32;
var<workgroup> rFlags:u32;
// Sum c x over a seam's 16 h cells (from the first's pool offset; their face g).
fn nSeamSum(first:u32,f:u32)->f32{
 let g=f^1u;let axis=f/2u;let u=nStride((axis+1u)%3u);let v=nStride((axis+2u)%3u);var s=0.0;
 for(var j=0u;j<16u;j++){let at=first+(j&3u)*u+(j>>2u)*v;if(bFaceKind(pool[P_KINDS+at],g)==K_SEAM){s+=pF(P_C+g*PLANE,at)*pF(P_X,at);}}
 return s;
}
@compute @workgroup_size(${WG}) fn pResidual2(${lanes}){
 if(lane==0u){atomicStore(&worst,0u);rN=select(ubaLevelCount(2u),0u,bConvergedBefore(pCycle));}
 let n=workgroupUniformLoad(&rN);
 for(var e=group.x;e<n;e+=groups.x){
  let block=ubaLevelSlot(2u,e);
  if(lane==0u){rFlags=nPage(block);}
  if((workgroupUniformLoad(&rFlags)&2u)!=0u){
   let at2=UBA_STRIDE*block+lane;let m=poolB[PN_M+at2];var r=0.0;
   if((m&N_ACTIVE)!=0u){
    let x=qF(PN_X,at2);let base=qF(PN_B,at2);r=qF(PN_R,at2)-qF(PN_D,at2)*x;
    for(var f=0u;f<6u;f++){
     let kind=nKind(m,f);let link=poolB[PN_L+f*PLANE2+at2];
     if(kind==N_NATIVE){r+=qF(PN_K+f*PLANE2,at2)*qF(PN_X,link);}
     else if(kind==N_SEAM){r+=nSeamSum(link,f);}
     else if(kind==N_WALL){r+=qF(PN_K+f*PLANE2,at2)*max(x+bitcast<f32>(link),-base);}${BP?`
     else if(kind==N_BULKN){r+=qF(PN_K+f*PLANE2,at2)*bsF(baSlot(link),BS_N+(link&15u));}`:""}
    }
    // Bulk members are fully open (bulkSelect rejects closed/cut cells),
    // hence unbounded: every member lane stays correction-active together.
    if(!pLast){poolB[PN_M+at2]=(m&~N_CORRECT)|select(0u,N_CORRECT,pCorrectionFree((m&N_CLOSED)!=0u,x+base,r));}
    if(pLast&&(m&N_CLOSED)!=0u){
     if(x+base<0.0){atomicAdd(&state[${KRYLOV_BOUND_WORD}u],1u);}
     if(x+base<=0.0&&r*nScale()*nStopScale()>params.solve.x){atomicAdd(&state[${KRYLOV_SIGN_WORD}u],1u);}
    }
    if((m&N_CLOSED)!=0u&&x+base<=0.0&&bResidualMagnitude(r)<0x7f800000u){r=max(r,0.0);}
    ${B?"if((m&N_BULK)==0u){atomicMax(&worst,bResidualMagnitude(r*nScale()));pDiag(pCycle,pDiagNative(m),bResidualMagnitude(r*nScale()*nStopScale()));}":"atomicMax(&worst,bResidualMagnitude(r*nScale()));pDiag(pCycle,pDiagNative(m),bResidualMagnitude(r*nScale()*nStopScale()));"}
   }
${B?`   if((workgroupUniformLoad(&rFlags)&4u)!=0u){
    r=nBulkMean(r,(m&N_BULK)!=0u,lane);if((m&N_BULK)!=0u){atomicMax(&worst,bResidualMagnitude(r*nScale()));pDiag(pCycle,7u,bResidualMagnitude(r*nScale()*nStopScale()));}
    // The check that a region is one pressure: its members' spread about
    // their mean, at the end (0 on every path).
    if(pLast){let xb=qF(PN_X,at2);let mean=nBulkMean(xb,(m&N_BULK)!=0u,lane);if((m&N_BULK)!=0u){atomicMax(&state[${KRYLOV_SPREAD_WORD}u],bResidualMagnitude(xb-mean));}}
   }\n`:""}   if(!pLast){qSet(PB_R,at2,r);}
  }
  workgroupBarrier();
 }
${BP?`  // The BULK16 rows: a region's residual is its row's (64 4h rows' sum,
  // so 1/64 of it on the native scale), into its level 3 RHS.
 if(lane==0u){rN=select(bsCount(),0u,bConvergedBefore(pCycle));}
 let nb=workgroupUniformLoad(&rN);
 for(var e=group.x;e<nb;e+=groups.x){
  let slot=bsSlot(e);let r=bResidual(slot,lane);
  if(lane<8u){atomicMax(&worst,bResidualMagnitude(r/64.0*nScale()));pDiag(pCycle,7u,bResidualMagnitude(r/64.0*nScale()*nStopScale()));if(!pLast){bsSet(slot,BS_T3+16u+lane,r);}}
${UniformPressureArena.diagnostics?"  if(lane<8u&&pCycle==1u&&!pLast){let r0=bsF(slot,BS_RHS+lane);if(abs(r0)/64.0*nScale()>2.5e-7){let q=r/r0;if(q<0.0){pDiag(5u,7u,bitcast<u32>(-q));}else{pDiag(5u,0u,bitcast<u32>(q));}}}\n":""} }
`:""} if(lane==0u&&n>0u){
  let w=atomicLoad(&worst);let word=HISTORY+pCycle;let scaled=bitcast<u32>(bitcast<f32>(w)*nStopScale());if(scaled>pIndex(word)){atomicMax(&state[word],scaled);}
  if(!pLast&&w>pIndex(NATIVE_HISTORY+pCycle)){atomicMax(&state[NATIVE_HISTORY+pCycle],w);}
  if(pLast&&w>pIndex(NATIVE_FINAL)){atomicMax(&state[NATIVE_FINAL],w);}
  if(group.x==0u){atomicOr(&state[VALID],1u<<pCycle);}
 }
}
// A half sweep of the native rows (colour pColour of the 4h lattice) against
// the band's iterate (plane pBand: X after a pre-smoothing, the prolonged
// plane before a post-smoothing) and the other colour's rows: the
// composite's fine smoothing of the rows that level 2 alone corrects
// otherwise (schedule.nativeSweeps). Walls solve their clamps as the band's
// rows do; a closed row stays at or above zero.
override pColour:u32=0u;
override pBand:u32=0u;
@compute @workgroup_size(${WG}) fn pNativeRelax(${lanes}){
 if(lane==0u){rN=select(ubaLevelCount(2u),0u,bConverged());}
 let n=workgroupUniformLoad(&rN);
 let l=vec3u(lane&7u,(lane>>3u)&7u,lane>>6u);
 for(var e=group.x;e<n;e+=groups.x){
  let block=ubaLevelSlot(2u,e);
  if(lane==0u){rFlags=nPage(block);}
  if((workgroupUniformLoad(&rFlags)&2u)!=0u){
   let at2=UBA_STRIDE*block+lane;let m=poolB[PN_M+at2];
   if((m&N_ACTIVE)!=0u&&((l.x+l.y+l.z)&1u)==pColour${B?"&&(m&N_BULK)==0u":""}){
    let base=qF(PN_B,at2);let d0=qF(PN_D,at2);var off=qF(PN_R,at2);var walls=0u;
    for(var f=0u;f<6u;f++){
     let kind=nKind(m,f);let link=poolB[PN_L+f*PLANE2+at2];
     if(kind==N_NATIVE){off+=qF(PN_K+f*PLANE2,at2)*qF(PN_X,link);}${BP?`
     else if(kind==N_BULKN){off+=qF(PN_K+f*PLANE2,at2)*bsF(baSlot(link),BS_N+(link&15u));}`:""}
     else if(kind==N_SEAM){
      let g=f^1u;let axis=f/2u;let u=nStride((axis+1u)%3u);let w=nStride((axis+2u)%3u);let plane=select(P_X,P_X1,pBand==1u);
      for(var j=0u;j<16u;j++){let at=link+(j&3u)*u+(j>>2u)*w;if(bFaceKind(pool[P_KINDS+at],g)==K_SEAM){off+=pF(P_C+g*PLANE,at)*pF(plane,at);}}
     }
     else if(kind==N_WALL){walls|=1u<<f;}
    }
    var v=off/d0;
    if(walls!=0u){
     for(var it=0u;it<4u;it++){
      var d=d0;var o=off;
      var remaining=walls;while(remaining!=0u){
       let f=firstTrailingBit(remaining);remaining&=remaining-1u;
       let w=qF(PN_K+f*PLANE2,at2);let g=bitcast<f32>(poolB[PN_L+f*PLANE2+at2]);
       if(v+g>=-base){d-=w;o+=w*g;}else{o-=w*base;}
      }
      if(d>0.0){v=o/d;}
     }
    }
    if((m&N_CLOSED)!=0u){v=max(v,-base);}
    qSet(PN_X,at2,v);
   }
  }
  workgroupBarrier();
 }
}`,
   // The band's final measure (when no cycle stopped), each band tile's mean
   // pressure into the root (the 4h projection's start there), the header's work counts.
   final:header()+poolWGSL+zero+/* wgsl */`
@group(1) @binding(26) var<storage,read_write> rootArena:array<f32>;
var<workgroup> worst:atomic<u32>;
var<workgroup> pMeasure:u32;
var<workgroup> fSum:array<vec2f,64>;
@compute @workgroup_size(${TILE_WG}) fn pFinal(${lanes}){
 if(lane==0u){atomicStore(&worst,0u);pN=pTileCount();pMeasure=u32(!bConvergedBefore(CYCLES));}
 if(group.x==0u&&lane==0u){atomicStore(&state[BRICKS],pFine());atomicStore(&state[${LEVEL1_WORD}u],ubaLevelCount(1u));atomicStore(&state[${LEVEL2_WORD}u],ubaLevelCount(2u));}
 let n=workgroupUniformLoad(&pN);let measure=workgroupUniformLoad(&pMeasure)!=0u;let l=vec3u(lane&3u,(lane>>2u)&3u,lane>>4u);
 for(var j=group.x;j<n;j+=groups.x){
  let w=pIndex(TLIST+j);let slot=pTileSlot(w);let lb=pTileCell(w,l);let at=pAddr(slot,lb);let id=ubaBrick(slot);
  let p=vec3i(ubaOrigin(id)+lb);let kinds=pool[P_KINDS+at];let own=pF(P_X,at);let base=pF(P_BASE,at);
  if(measure&&pActive(kinds)){
   var off=0.0;var d0=0.0;
   for(var f=0u;f<6u;f++){
    let kind=bFaceKind(kinds,f);let c=pCoefficient(at,kinds,f);d0+=c;
    if(kind==K_BAND){off+=c*pF(P_X,pNear(at,p,f));}
    else if(kind==K_SEAM){off+=c*qF(PN_X,nAcross(p,f));}
    else if(kind==K_WALL){off+=c*max(own+pLift(at,f),-base);}
   }
   if(d0>0.0){
    var r=pF(P_RHS,at)+off-d0*own;
    if(bInside(kinds)&&own+base<=0.0&&r*params.policy.z>params.solve.x){atomicAdd(&state[${KRYLOV_SIGN_WORD}u],1u);}
    if(bInside(kinds)&&own+base<=0.0&&bResidualMagnitude(r)<0x7f800000u){r=max(r,0.0);}
    atomicMax(&worst,bResidualMagnitude(r*params.policy.z));pDiag(CYCLES,pDiagClass(kinds),bResidualMagnitude(r*params.policy.z));
   }
  }
  if(pActive(kinds)&&bInside(kinds)&&own+base<0.0){atomicAdd(&state[${KRYLOV_BOUND_WORD}u],1u);}
  fSum[lane]=select(vec2f(0.0),vec2f(base+own,1.0),bLiquid(kinds));
  workgroupBarrier();
  if(lane==0u){
   var sum=vec2f(0.0);for(var i=0u;i<64u;i++){sum+=fSum[i];}
   let bit=w&7u;let tile=umTileKey(vec3i(ubaOrigin(id)/4u)+vec3i(vec3u(bit&1u,(bit>>1u)&1u,bit>>2u)));
   rootArena[R_PRESSURE+tile]=select(0.0,sum.x/sum.y,sum.y>0.0);
  }
  workgroupBarrier();
 }
 if(lane==0u&&n>0u&&measure){let r=atomicLoad(&worst);if(r>pIndex(HISTORY+CYCLES)){atomicMax(&state[HISTORY+CYCLES],r);}if(group.x==0u){atomicOr(&state[VALID],1u<<CYCLES);}}
}`,
   // The rows against their start (UniformPressureBand.rebase), the initial
   // measure (x = 0: each row's own), each region's predictor residual, and
   // level 1's rows: each brick bakes its parent octant.
   ...(schedule.krylov?{krylov:header()+poolWGSL+store+(B?bulkMean("kRed"):"")+(BP?bulkRows:"")+krylovWGSL}:{}),
   rebase:header()+poolWGSL+/* wgsl */`
// The kick's first solve stashes each row's RHS ahead of its start's terms
// in the warm plane (unread until pProject), its re-solves read it back.
override pStash:u32=0u;
override pCorrection:bool=false;
${this.fields.dynamics?"@group(1) @binding(24) var<storage,read_write> bDynamics:array<atomic<u32>>;":""}
var<workgroup> worst:atomic<u32>;
var<workgroup> tileWorst:atomic<u32>;
var<workgroup> rPart:array<f32,64>;
var<workgroup> rCross:array<f32,192>;
${zero}
// A tile's level-1 cell A (2^3): its 8 children's sum, or the outward
// couplings on side f, scaled.
fn rChild(A:vec3u,k:u32)->u32{let l=2u*A+vec3u(k&1u,(k>>1u)&1u,k>>2u);return l.x+4u*(l.y+4u*l.z);}
fn rDiag(A:vec3u)->f32{
 var s=0.0;var outward=0.0;
 for(var k=0u;k<8u;k++){let i=rChild(A,k);s+=rPart[i];for(var axis=0u;axis<3u;axis++){outward+=rCross[64u*axis+i];}}
 return pAggregateDiagonal(s,outward,P_SCALE);
}
fn rCouple(A:vec3u,f:u32)->f32{var s=0.0;let axis=f/2u;for(var k=0u;k<8u;k++){if(((k>>axis)&1u)==(f&1u)){s+=rCross[64u*axis+rChild(A,k)];}}return P_SCALE*s;}
@compute @workgroup_size(${TILE_WG}) fn pRebase(${lanes}){
 if(lane==0u){pN=pTileCount();atomicStore(&worst,0u);atomicStore(&tileWorst,0u);}
 let n=workgroupUniformLoad(&pN);let l=vec3u(lane&3u,(lane>>2u)&3u,lane>>4u);
 for(var j=group.x;j<n;j+=groups.x){
  let entry=pIndex(TLIST+j);let slot=pTileSlot(entry);let bit=entry&7u;let id=ubaBrick(slot);let lb=pTileCell(entry,l);let at=pAddr(slot,lb);
  let kinds=pool[P_KINDS+at];
  var part=0.0;var cross=vec3f(0.0);
  if(bLiquid(kinds)){
   let p=vec3i(ubaOrigin(id)+lb);let own=pF(P_BASE,at);var sums=vec3f(0.0);var open=0.0;
   var d0=0.0;var walls=0.0;var lifted=0.0;
   for(var axis=0u;axis<3u;axis++){
    var w=vec2f(0.0);var d=vec2f(0.0);
    for(var side=0u;side<2u;side++){
     let f=2u*axis+side;let kind=bFaceKind(kinds,f);let c=pF(P_C+f*PLANE,at);d0+=c;
     // The start's RHS sums (bSeamHead's 4-12 loads a seam face) are the
     // first rebase's alone: a correction rebase rebakes level 1 only.
     if(kind==K_BAND){
      let m=pNear(at,p,f);if(!pCorrection){w[side]=c;d[side]=pF(P_BASE,m)-own;}
      // Level 1: a coupling to an active row, inside the aggregate or out of it.
      if(pActive(pool[P_KINDS+m])&&(!pCorrection||(pool[P_KINDS+m]&P_CORRECT)!=0u)){if(side==(lb[axis]&1u)){cross[axis]=c;}else{part-=c;}}
     }
     else if(kind==K_SEAM){
      // A seam: the native row's start; out of the aggregate (a tile face)
      // toward an active native row, its level 2 cell.
      let m=nAcross(p,f);if(!pCorrection){w[side]=c;d[side]=qF(PN_B,m)+bSeamHead(at,f)-own;}let sc=seamCoarse(c);
      if((poolB[PN_M+m]&N_ACTIVE)!=0u&&(!pCorrection||(poolB[PN_M+m]&N_CORRECT)!=0u)){part+=sc.x;cross[axis]=sc.y;}
     }
     else if(kind==K_GHOST||kind==K_OPEN){open+=c;}
     else if(kind==K_WALL){if(!pCorrection||pF(P_X,at)+pLift(at,f)>=-own){walls+=c;}lifted+=c*max(pLift(at,f),-own);}
    }
    sums[axis]=w.x*(d.x+d.y)+(w.y-w.x)*d.y;
   }
   var given=pF(P_RHS,at);if(pStash==1u){pSet(P_WARMP,at,given);}else if(pStash==2u){given=pF(P_WARMP,at);}
   let rhs=given+((sums.x+sums.z)+sums.y-open*own);if(!pCorrection){pSet(P_RHS,at,rhs);}
   if(pActive(kinds)&&(!pCorrection||(kinds&P_CORRECT)!=0u)){
    // The band's residual of the zero iterate (its halos at their start): the first rebase's measure.
    if(!pCorrection){var r=rhs+lifted;if(bInside(kinds)&&own<=0.0&&bResidualMagnitude(r)<0x7f800000u){r=max(r,0.0);}
    let m=bResidualMagnitude(r*params.policy.z);atomicMax(&worst,m);atomicMax(&tileWorst,m);}
    // The error equation's diagonal: the walls follow their row.
    part+=d0-walls;if(!pCorrection){pSet(P_DIAG,at,d0);}
   }else{part=0.0;cross=vec3f(0.0);}
  }
  rPart[lane]=part;rCross[lane]=cross.x;rCross[64u+lane]=cross.y;rCross[128u+lane]=cross.z;
  workgroupBarrier();
  // Level 1: the tile's 2^3 cells of its parent octant; the brick's first
  // band tile clears the cells of its tiles outside the band.
  let parent=ubaParent(0u,slot);let o=parent.y;let oct=4u*vec3u(o&1u,(o>>1u)&1u,o>>2u);
  if(lane<8u){
   let A=vec3u(lane&1u,(lane>>1u)&1u,lane>>2u);let at1=pAddr(parent.x,oct+2u*(lb>>vec3u(2u))+A);
   qSet(PA_D,at1,rDiag(A));for(var f=0u;f<6u;f++){qSet(PA_K+f*PLANE1,at1,rCouple(A,f));}
  }
  let mask=pIndex(MASK+id);let dead=lane>>3u;
  if((mask&((1u<<bit)-1u))==0u&&((mask>>dead)&1u)==0u){
   let A=2u*vec3u(dead&1u,(dead>>1u)&1u,dead>>2u)+vec3u(lane&1u,(lane>>1u)&1u,(lane>>2u)&1u);let at1=pAddr(parent.x,oct+A);
   qSet(PA_D,at1,0.0);for(var f=0u;f<6u;f++){qSet(PA_K+f*PLANE1,at1,0.0);}
  }
  workgroupBarrier();
  if(lane==0u){
   let r=atomicExchange(&tileWorst,0u);
   ${this.fields.dynamics?`if(!pCorrection){let tile=umTileKey(vec3i(ubaOrigin(id)/4u)+vec3i(vec3u(bit&1u,(bit>>1u)&1u,bit>>2u)));
   let at2=${UNIFORM_REGION_DYNAMICS_WORDS}u*umPageOfTile(tile)+${UNIFORM_REGION_DYNAMICS.pressureResidual}u;if(r>atomicLoad(&bDynamics[at2])){atomicMax(&bDynamics[at2],r);}}`:""}
  }
  workgroupBarrier();
 }
 if(!pCorrection&&lane==0u&&n>0u){let r=atomicLoad(&worst);if(r>pIndex(INITIAL)){atomicMax(&state[INITIAL],r);}if(group.x==0u){atomicOr(&state[VALID],INITIAL_BIT);}}
}`,
   // A red-black sweep of level 0, in place: a workgroup relaxes two band
   // tiles, a lane one red cell and then one black. A tile's black rows see
   // its own red rows of this sweep; across tiles, the last sweep's values
   // (plane pSrc). An active row writes the other plane; the rest hold 0 in
   // both from pInit. pSettle: the shut rows alone, the other active rows
   // carried across (the first cycle's settle sweep under solids).
   relax:header()+poolWGSL+relaxRow+/* wgsl */`
override pSettle:bool=false;
@compute @workgroup_size(${TILE_WG}) fn pRelax(${lanes}){
 if(lane==0u){rN=pTileLive();}
 let n=workgroupUniformLoad(&rN);
 let i=lane&31u;let y=(i>>1u)&3u;let z=i>>3u;
 for(var j0=2u*group.x;j0<n;j0+=2u*groups.x){
  let j=j0+(lane>>5u);
  for(var colour=0u;colour<2u;colour++){
   if(j<n){
    let l=vec3u(2u*(i&1u)+((y+z+colour)&1u),y,z);
    let w=pIndex(TLIST+j);let slot=pTileSlot(w);let lb=pTileCell(w,l);let at=pAddr(slot,lb);let kinds=pool[P_KINDS+at];
    if(pActive(kinds)){
     var x:f32;if(!pSettle||bShut(kinds)){x=pRelaxRow(at,slot,lb,kinds,colour==1u);}else{x=pF(pPlane(pSrc),at);}
     pSet(pPlane(1u-pSrc),at,x);
    }
   }
   storageBarrier();
  }
 }
}`,
   // The exact h residual after pre-smoothing, a workgroup a band tile: its
   // largest liquid row into history word pCycle, the children's sum into
   // level 1's RHS (a tile's 2^3 level 1 cells).
   restrict:header()+poolWGSL+relaxRow+/* wgsl */`
override pCycle:u32=0u;
var<workgroup> worst:atomic<u32>;
var<workgroup> rRes:array<f32,64>;
@compute @workgroup_size(${TILE_WG}) fn pRestrict0(${lanes}){
 if(lane==0u){atomicStore(&worst,0u);rN=select(pTileCount(),0u,bConvergedBefore(pCycle));}
 let n=workgroupUniformLoad(&rN);
 let l=vec3u(lane&3u,(lane>>2u)&3u,lane>>4u);
 for(var j=group.x;j<n;j+=groups.x){
  let w=pIndex(TLIST+j);let slot=pTileSlot(w);let lb=pTileCell(w,l);let at=pAddr(slot,lb);let kinds=pool[P_KINDS+at];var r=0.0;
  if(pActive(kinds)){
   let x=pF(P_X,at);let base=pF(P_BASE,at);var off=0.0;
   for(var f=0u;f<6u;f++){
    let kind=bFaceKind(kinds,f);
    if(kind==K_BAND){off+=pCoefficient(at,kinds,f)*pF(P_X,pNear0(at,slot,lb,f));}
    else if(kind==K_SEAM){off+=pF(P_C+f*PLANE,at)*qF(PN_X,nAcross(vec3i(ubaOrigin(ubaBrick(slot))+lb),f));}
    else if(kind==K_WALL){off+=pF(P_C+f*PLANE,at)*max(x+pLift(at,f),-base);}
   }
   r=pF(P_RHS,at)+off-pF(P_DIAG,at)*x;
   pool[P_KINDS+at]=(kinds&~P_CORRECT)|select(0u,P_CORRECT,pCorrectionFree(bInside(kinds),x+base,r));
   if(bInside(kinds)&&x+base<=0.0&&bResidualMagnitude(r)<0x7f800000u){r=max(r,0.0);}
   atomicMax(&worst,bResidualMagnitude(r*params.policy.z));pDiag(pCycle,pDiagClass(kinds),bResidualMagnitude(r*params.policy.z));
  }
  rRes[lane]=r;workgroupBarrier();
  if(lane<8u){
   let A=vec3u(lane&1u,(lane>>1u)&1u,lane>>2u);var s=0.0;
   for(var k=0u;k<8u;k++){let c=2u*A+vec3u(k&1u,(k>>1u)&1u,k>>2u);s+=rRes[c.x+4u*(c.y+4u*c.z)];}
   let parent=ubaParent(0u,slot);let o=parent.y;
   qSet(PA_R,pAddr(parent.x,4u*vec3u(o&1u,(o>>1u)&1u,o>>2u)+2u*(lb>>vec3u(2u))+A),s);
  }
  workgroupBarrier();
 }
 workgroupBarrier();
 if(lane==0u&&n>0u){let m=atomicLoad(&worst);if(m>pIndex(HISTORY+pCycle)){atomicMax(&state[HISTORY+pCycle],m);}if(group.x==0u){atomicOr(&state[VALID],1u<<pCycle);}}
}`,
   // Level 1 down: the box from zero, its core kept; with pBake level 2's
   // rows (the octant's). pDownR then sums the cores' residual into level
   // 2's RHS.
   downA:header()+poolWGSL+coarse(1)+bakeArrays+/* wgsl */`
override pBake:bool=false;
@compute @workgroup_size(${WG}) fn pDownA(${lanes}){
 if(lane==0u){cN=select(ubaLevelCount(1u),0u,bConverged());}
 let n=workgroupUniformLoad(&cN);
 for(var e=group.x;e<n;e+=groups.x){
  let slot=ubaLevelSlot(1u,e);cRecords(slot,lane);
  for(var k=lane;k<1728u;k+=${WG}u){cX[k]=0.0;}
  let red=cBoxCell(0u,lane);let black=cBoxCell(1u,lane);let cR0=cBoxRow(0u,lane);let cB0=cBoxRow(1u,lane);
  workgroupBarrier();
  cSweeps(red,cR0,black,cB0);
  let c=cCore(lane);let at=cAddr(c);let core=cCell(at);let d=core.d;let x=cX[cBox(c)];
  ubaStore1(CX,slot,lane,x);
  workgroupBarrier();
  if(!pBake){continue;}
  {
   gCoarseChild(lane,vec3u(c),at,d,CK,CPL);
   // Restore a missing level 1 link only when it crosses a real native
   // seam. An absent liquid correction row remains a fixed-zero boundary.
   if(d>0.0){for(var axis=0u;axis<3u;axis++){let f=2u*axis+(u32(c[axis])&1u);var q=c;q[axis]+=bSign(f);let k=core.k[f];if(k>0.0&&cDiag(cAddr(q))<=0.0){
     let hq=2*(vec3i(ubaLevelOrigin(1u,ubaLevelRecord(1u,slot,29u)))+q);
     if(umCellPresent(hq)){let tile=umTileKey(hq/4);let native=UBA_STRIDE*(tile>>9u)+(tile&511u);
      if((poolB[PN_M+native]&N_CORRECT)!=0u){let sc=seamCoarse(k);gPart[lane]+=sc.x;gCross[512u*axis+lane]=sc.y;}
     }
    }}}
  }
  workgroupBarrier();
  if(lane<64u){
   let parent=ubaParent(1u,slot);let A=vec3u(lane&3u,(lane>>2u)&3u,lane>>4u);let o=parent.y;
   let at2=pAddr(parent.x,4u*vec3u(o&1u,(o>>1u)&1u,o>>2u)+A);
   // A native row's level 2 row is its own (pNative, pResidual2).
   if((poolB[PN_M+at2]&N_ACTIVE)==0u){qSet(PB_D,at2,gDiag(A));for(var f=0u;f<6u;f++){qSet(PB_K+f*PLANE2,at2,gCouple(A,f));}}
  }
 }
}
// Level 1's residual against the neighbour bricks' cores (every brick's
// pDownA has stored its own), its children's sums into level 2's RHS.
@compute @workgroup_size(${WG}) fn pDownR(${lanes}){
 if(lane==0u){cN=select(ubaLevelCount(1u),0u,bConverged());}
 let n=workgroupUniformLoad(&cN);
 for(var e=group.x;e<n;e+=groups.x){
  let slot=ubaLevelSlot(1u,e);cRecords(slot,lane);
  let c=cCore(lane);let at=cAddr(c);let core=cCell(at);var r=0.0;
  if(core.d>0.0){
   var off=0.0;
   for(var f=0u;f<6u;f++){if(core.k[f]!=0.0){var q=c;q[f/2u]+=bSign(f);let m=cAddr(q);if(cDiag(m)>0.0){off+=core.k[f]*qF(CX,m);}}}
   r=core.r+off-core.d*qF(CX,at);
  }
  gRes[lane]=r;
  workgroupBarrier();
  if(lane<64u){
   let parent=ubaParent(1u,slot);let A=vec3u(lane&3u,(lane>>2u)&3u,lane>>4u);let o=parent.y;
   let at2=pAddr(parent.x,4u*vec3u(o&1u,(o>>1u)&1u,o>>2u)+A);
   if((poolB[PN_M+at2]&N_ACTIVE)==0u){qSet(PB_R,at2,gSum(A));}
  }
  workgroupBarrier();
 }
}`,
   // Level 2 down (a page a brick): the box from zero, its core kept; with
   // pBake the page's 8h, 16h and 32h rows (all within the page).
   downB:header()+poolWGSL+coarse(2,schedule.coarsestSweeps)+bakeArrays+tail+/* wgsl */`
override pBake:bool=false;
// The parent's rows of aggregate A of a 2s^3 child grid: children k at 2A+k (local index x+2s(y+2s z)).
fn tChild(A:vec3u,k:u32,s:u32)->u32{let l=2u*A+vec3u(k&1u,(k>>1u)&1u,k>>2u);return l.x+2u*s*(l.y+2u*s*l.z);}
// A child row's part and outward couplings (level 3 or 4, within the page) into gPart/gCross at i.
fn tPart(i:u32,l:vec3u,d:f32,k:array<f32,6>){
 var part=0.0;var cross=vec3f(0.0);
 if(d>0.0){part=d;for(var axis=0u;axis<3u;axis++){let bit=l[axis]&1u;part-=k[2u*axis+1u-bit];cross[axis]=k[2u*axis+bit];}}
 gPart[i]=part;gCross[i]=cross.x;gCross[512u+i]=cross.y;gCross[1024u+i]=cross.z;
}
// Faces in mask (a BULK16 page across) keep their coupling unscaled, as a
// seam's does (seamCoarse): the 16h row across stays where it is.
fn tBake(at:u32,A:vec3u,s:u32,mask:u32){
 var d=0.0;var k=array<f32,6>(0.0,0.0,0.0,0.0,0.0,0.0);
 for(var j=0u;j<8u;j++){let i=tChild(A,j,s);d+=gPart[i];for(var axis=0u;axis<3u;axis++){let bit=(j>>axis)&1u;k[2u*axis+bit]+=gCross[512u*axis+i];}}
 var outward=0.0;for(var f=0u;f<6u;f++){if(((mask>>f)&1u)==0u){outward+=k[f];}}
 tSet(PT_D,at,pAggregateDiagonal(d,outward,T_SCALE));for(var f=0u;f<6u;f++){tSet(PT_K+f*PLANE2,at,select(T_SCALE*k[f],k[f],((mask>>f)&1u)!=0u));}
}
@compute @workgroup_size(${WG}) fn pDownB(${lanes}){
 if(lane==0u){cN=select(ubaLevelCount(2u),0u,bConverged());}
 let n=workgroupUniformLoad(&cN);
 for(var e=group.x;e<n;e+=groups.x){
  let slot=ubaLevelSlot(2u,e);
  if(lane==0u){cN=u32(tPageLive(slot));}
  // A page with no rows: its tail rows (read by its neighbours) are none.
  if(workgroupUniformLoad(&cN)==0u){if(pBake&&lane<73u){qSet(PT_D,UBA_STRIDE*slot+lane,0.0);}workgroupBarrier();continue;}
  cRecords(slot,lane);
  for(var k=lane;k<1728u;k+=${WG}u){cX[k]=0.0;}
  let red=cBoxCell(0u,lane);let black=cBoxCell(1u,lane);let cR0=cBoxRow(0u,lane);let cB0=cBoxRow(1u,lane);
  workgroupBarrier();
  cSweeps(red,cR0,black,cB0);
  let c=cCore(lane);let at=cAddr(c);let core=cCell(at);
  ubaStore1(CX,slot,lane,cX[cBox(c)]);
  // With pBake the page's 8h, 16h and 32h rows (aggregates within the page).
  if(pBake){
   workgroupBarrier();
   gCoarseChild(lane,vec3u(c),at,core.d,CK,CPL);
   workgroupBarrier();
   let g=tLocal(lane,4u);let at3=UBA_STRIDE*slot+lane;var d3=0.0;var k3=array<f32,6>(0.0,0.0,0.0,0.0,0.0,0.0);
   if(lane<64u){let A=vec3u(g);d3=gDiag(A);for(var f=0u;f<6u;f++){k3[f]=gCouple(A,f);}
${BP?`    // A BULK16 row across: its children's links, unscaled (gCoarseChild left them in the diagonal).
    if((nPage(slot)&8u)!=0u){for(var f=0u;f<6u;f++){let axis=f/2u;if(u32(g[axis])!=select(0u,3u,(f&1u)==1u)){continue;}
     for(var j=0u;j<8u;j++){if(((j>>axis)&1u)!=(f&1u)){continue;}let l=2u*A+vec3u(j&1u,(j>>1u)&1u,j>>2u);let at=UBA_STRIDE*slot+l.x+8u*(l.y+8u*l.z);let m=poolB[PN_M+at];
      if((m&N_ACTIVE)!=0u&&(m&N_CORRECT)!=0u&&nKind(m,f)==N_BULKN){k3[f]+=qF(PN_K+f*PLANE2,at);}}}}
`:""}    qSet(PT_D,at3,d3);for(var f=0u;f<6u;f++){qSet(PT_K+f*PLANE2,at3,k3[f]);}}
   workgroupBarrier();
   if(lane<64u){tPart(lane,vec3u(g),d3,k3);}
   workgroupBarrier();
   let g4=tLocal(lane,2u);let at4=UBA_STRIDE*slot+64u+lane;
   var mask4=0u;
${BP?`   if(lane<8u){for(var f=0u;f<6u;f++){let axis=f/2u;if(u32(g4[axis])==select(0u,1u,(f&1u)==1u)){let step=select(select(9u,3u,axis==1u),1u,axis==0u);if(bsWordSlot(umBlockNeighbourWord(slot,select(13u-step,13u+step,(f&1u)==1u)))!=0u){mask4|=1u<<f;}}}}
`:""}   if(lane<8u){tBake(at4,vec3u(g4),2u,mask4);}
   workgroupBarrier();
   if(lane<8u){var k4=array<f32,6>(0.0,0.0,0.0,0.0,0.0,0.0);for(var f=0u;f<6u;f++){k4[f]=qF(PT_K+f*PLANE2,at4);}tPart(lane,vec3u(g4),qF(PT_D,at4),k4);}
   workgroupBarrier();
   if(lane==0u){tBake(UBA_STRIDE*slot+72u,vec3u(0u),1u,0u);}
  }
  workgroupBarrier();
 }
}`,
   // Level 2's residual against the neighbour pages' down-leg cores (not
   // the box's own view of them, which the up leg never sees), summed into
   // the page's 8h RHS; then the 8h rows from zero inside the page (the
   // first block-Jacobi iterate).
   down2:header()+poolWGSL+coarse(2,schedule.coarsestSweeps)+bakeArrays+tail+bulkRows+/* wgsl */`
var<workgroup> tX:array<f32,64>;
${BP?"var<workgroup> bBake:array<f32,48>;":""}
@compute @workgroup_size(${WG}) fn pDown2(${lanes}){
 if(lane==0u){cN=select(ubaLevelCount(2u),0u,bConverged());}
 let n=workgroupUniformLoad(&cN);
 for(var e=group.x;e<n;e+=groups.x){
  let slot=ubaLevelSlot(2u,e);
  if(lane==0u){cN=u32(tPageLive(slot));}
  if(workgroupUniformLoad(&cN)==0u){continue;}
  cRecords(slot,lane);
  let c=cCore(lane);let at=cAddr(c);let core=cCell(at);var r=0.0;
  if(core.d>0.0){
   var off=0.0;
   for(var f=0u;f<6u;f++){if(core.k[f]!=0.0){var q=c;q[f/2u]+=bSign(f);let m=cAddr(q);if(m!=UB_NONE_ID){off+=core.k[f]*qF(CX,m);}}}
   r=core.r+off-core.d*qF(CX,at);
  }
  gRes[lane]=r;
  workgroupBarrier();
  let g=tLocal(lane,4u);let at3=UBA_STRIDE*slot+lane;var r3=0.0;var d3=0.0;var k3=array<f32,6>(0.0,0.0,0.0,0.0,0.0,0.0);
  if(lane<64u){
   r3=gSum(vec3u(g));qSet(PT_R,at3,r3);
   d3=qF(PT_D,at3);for(var f=0u;f<6u;f++){k3[f]=qF(PT_K+f*PLANE2,at3);}
   tX[lane]=0.0;
  }
  workgroupBarrier();
  for(var s=0u;s<${schedule.sweeps}u;s++){
   for(var col=0u;col<2u;col++){
    if(lane<64u&&d3>0.0&&u32(g.x+g.y+g.z)%2u==col){
     var off=r3;
     for(var f=0u;f<6u;f++){var q=g;q[f/2u]+=bSign(f);if(all(q>=vec3i(0))&&all(q<vec3i(4))){off+=k3[f]*tX[u32(q.x+4*q.y+16*q.z)];}}
     tX[lane]=off/d3;
    }
    workgroupBarrier();
   }
  }
  if(lane<64u){qSet(PT_X,at3,tX[lane]);}
  workgroupBarrier();
 }
${BP?`  // A BULK16 page: its rows' level 3 RHS takes the level 2 correction of
  // the native rows across (their residual, after it); its 8h iterate
  // (one a region) from zero by sweeps inside the page.
 if(lane==0u){cN=select(bsCount(),0u,bConverged());}
 let nb=workgroupUniformLoad(&cN);
 for(var e=group.x;e<nb;e+=groups.x){
  let slot=bsSlot(e);let page=bsPage(slot);
  if(lane<48u){bRed[lane]=bLinkSum(slot,page,lane/6u,lane%6u,true);}
  workgroupBarrier();
  var r3=0.0;var d=0.0;
  if(lane<8u){r3=bsF(slot,BS_T3+16u+lane)+((bRed[6u*lane]+bRed[6u*lane+1u])+(bRed[6u*lane+2u]+bRed[6u*lane+3u]))+(bRed[6u*lane+4u]+bRed[6u*lane+5u]);
   bsSet(slot,BS_T3+16u+lane,r3);d=bsF(slot,BS_T4+24u+lane);tX[lane]=0.0;}
  workgroupBarrier();
${bulkSweeps("tX","r3","d")}
  if(lane<8u){bsSet(slot,BS_T3+lane,tX[lane]);}
  workgroupBarrier();
 }
  // Each BULK16 page's 16h couplings (to a resident block's 16h row: that
  // row's to it, f^1) and its 32h row, as tBake bakes a page's. Nothing
  // above reads them, nor writes what they read (pDownB's 16h rows, the
  // slot's level 2 diagonal), so the bake shares this pass.
 for(var e=group.x;e<nb;e+=groups.x){
  let slot=bsSlot(e);let page=bsPage(slot);
  if(lane<48u){
   let i=lane/6u;let f=lane%6u;let near=bNear(slot,page,i,f);var k=0.0;
   if(near.kind==B_REGION){k=bRegionK(f);}
   else if(near.kind==B_BLOCK){var g=bRegionCoord(i);g[f/2u]=select(1u,0u,(f&1u)==1u);let at=UBA_STRIDE*near.block+64u+bRegionIndex(g);if(qF(PT_D,at)>0.0){k=qF(PT_K+(f^1u)*PLANE2,at);}}
   bsSet(slot,BS_T4+8u*(4u+f)+i,k);bBake[lane]=k;
  }
  workgroupBarrier();
  if(lane==0u){
   var d=0.0;var k5=array<f32,6>(0.0,0.0,0.0,0.0,0.0,0.0);
   for(var i=0u;i<8u;i++){let r=bRegionCoord(i);d+=bsF(slot,BS_T4+24u+i);for(var f=0u;f<6u;f++){if(r[f/2u]!=(f&1u)){d-=bBake[6u*i+f];}else{k5[f]+=bBake[6u*i+f];}}}
   var outward=0.0;for(var f=0u;f<6u;f++){outward+=k5[f];}
   bsSet(slot,BS_T5+3u,pAggregateDiagonal(d,outward,T_SCALE));for(var f=0u;f<6u;f++){bsSet(slot,BS_T5+4u+f,T_SCALE*k5[f]);}
  }
  workgroupBarrier();
 }
`:""}}`,
   // The 8h level's V-cycles (tailCycles of them, each down3, down4, tail,
   // up4, up3): its residual against the neighbour pages' rows into the
   // page's 16h RHS, the 16h rows from zero inside the page; the 16h
   // residual against the neighbour pages' into the 32h RHS. The 8h iterate
   // alternates between the X and Y channels (pSrcY: read Y), so a page reads
   // its neighbours' while they write the other.
   down3:header()+poolWGSL+tail+bulkRows+/* wgsl */`
override pSrcY:bool=false;
var<workgroup> dR:array<f32,64>;
var<workgroup> dX:array<f32,8>;
var<workgroup> dN:u32;
fn tSrc(at:u32)->f32{return tF(select(PT_X,PT_Y,pSrcY),at);}
@compute @workgroup_size(64) fn pDown3(${lanes}){
 if(lane==0u){dN=select(ubaLevelCount(2u),0u,bConverged());}
 let n=workgroupUniformLoad(&dN);
 for(var e=group.x;e<n;e+=groups.x){
  let slot=ubaLevelSlot(2u,e);
  if(lane==0u){dN=u32(tPageLive(slot));}
  if(workgroupUniformLoad(&dN)==0u){continue;}
  let g=tLocal(lane,4u);let at3=UBA_STRIDE*slot+lane;let d3=qF(PT_D,at3);var res=0.0;
  if(d3>0.0){
   res=qF(PT_R,at3)-d3*tSrc(at3);
   for(var f=0u;f<6u;f++){var q=g;q[f/2u]+=bSign(f);let m=tAt(slot,3u,q);if(m!=UB_NONE_ID&&tF(PT_D,m)>0.0){res+=qF(PT_K+f*PLANE2,at3)*tSrc(m);}}
  }
  dR[lane]=res;
  workgroupBarrier();
  let g4=tLocal(lane,2u);let at4=UBA_STRIDE*slot+64u+lane;var r4=0.0;var d4=0.0;var k4=array<f32,6>(0.0,0.0,0.0,0.0,0.0,0.0);
  if(lane<8u){
   let A=vec3u(g4);
   for(var j=0u;j<8u;j++){let l=2u*A+vec3u(j&1u,(j>>1u)&1u,j>>2u);r4+=dR[l.x+4u*(l.y+4u*l.z)];}
   qSet(PT_R,at4,r4);d4=qF(PT_D,at4);for(var f=0u;f<6u;f++){k4[f]=qF(PT_K+f*PLANE2,at4);}dX[lane]=0.0;
  }
  workgroupBarrier();
  for(var s=0u;s<${schedule.sweeps}u;s++){
   for(var col=0u;col<2u;col++){
    if(lane<8u&&d4>0.0&&u32(g4.x+g4.y+g4.z)%2u==col){
     var off=r4;for(var f=0u;f<6u;f++){var q=g4;q[f/2u]+=bSign(f);if(all(q>=vec3i(0))&&all(q<vec3i(2))){off+=k4[f]*dX[u32(q.x+2*q.y+4*q.z)];}}
     dX[lane]=off/d4;
    }
    workgroupBarrier();
   }
  }
  if(lane<8u){qSet(PT_X,at4,dX[lane]);}
  workgroupBarrier();
 }
${BP?`  // A BULK16 page: its rows' 8h residual (against the 8h rows across:
  // their couplings to it, f^1) is their 16h RHS, then their 16h rows from
  // zero inside the page.
 if(lane==0u){dN=select(bsCount(),0u,bConverged());}
 let nb=workgroupUniformLoad(&dN);
 for(var e=group.x;e<nb;e+=groups.x){
  let slot=bsSlot(e);let page=bsPage(slot);
  if(lane<48u){
   let i=lane/6u;let f=lane%6u;let near=bNear(slot,page,i,f);var t=0.0;
   if(near.kind==B_REGION){t=bRegionK(f)*tSrc(baOf(near.slot,true,near.region));}
   else if(near.kind==B_BLOCK){
    let a=f/2u;let r=bRegionCoord(i);
    for(var j=0u;j<4u;j++){
     var g=vec3u(0u);g[a]=select(3u,0u,(f&1u)==1u);g[(a+1u)%3u]=2u*r[(a+1u)%3u]+(j&1u);g[(a+2u)%3u]=2u*r[(a+2u)%3u]+(j>>1u);
     let at=UBA_STRIDE*near.block+g.x+4u*(g.y+4u*g.z);if(qF(PT_D,at)>0.0){t+=qF(PT_K+(f^1u)*PLANE2,at)*tSrc(at);}
    }
   }
   dR[lane]=t;
  }
  workgroupBarrier();
  var r4=0.0;var d=0.0;
  if(lane<8u){let at=baOf(slot,true,lane);d=bsF(slot,BS_T4+24u+lane);
   r4=tF(PT_R,at)-d*tSrc(at)+((dR[6u*lane]+dR[6u*lane+1u])+(dR[6u*lane+2u]+dR[6u*lane+3u]))+(dR[6u*lane+4u]+dR[6u*lane+5u]);
   bsSet(slot,BS_T4+16u+lane,r4);dX[lane]=0.0;}
  workgroupBarrier();
${bulkSweeps("dX","r4","d")}
  if(lane<8u){bsSet(slot,BS_T4+lane,dX[lane]);}
  workgroupBarrier();
 }
`:""}}`,
   // The 16h residual against the neighbour pages' rows, summed into the page's 32h RHS.
   down4:header()+poolWGSL+tail+/* wgsl */`
var<workgroup> dR:array<f32,8>;
var<workgroup> dN:u32;
var<workgroup> dM:u32;
// Every page item: the resident pages, then the BULK16 pages (tItem).
@compute @workgroup_size(64) fn pDown4(${lanes}){
 if(lane==0u){dM=ubaLevelCount(2u);dN=select(tCount(dM),0u,bConverged());}
 let n=workgroupUniformLoad(&dN);let n2=workgroupUniformLoad(&dM);
 for(var e=group.x;e<n;e+=groups.x){
  let slot=tItem(e,n2);
  if(lane==0u){dN=u32(tPageLive(slot));}
  if(workgroupUniformLoad(&dN)==0u){continue;}
  if(lane<8u){
   let g4=tLocal(lane,2u);let at4=tCell(slot,4u,lane);let d4=tF(PT_D,at4);var res=0.0;
   if(d4>0.0){
    res=tF(PT_R,at4)-d4*tF(PT_X,at4);
    for(var f=0u;f<6u;f++){var q=g4;q[f/2u]+=bSign(f);let m=tAt(slot,4u,q);if(m!=UB_NONE_ID&&tF(PT_D,m)>0.0){res+=tF(PT_K+f*PLANE2,at4)*tF(PT_X,m);}}
   }
   dR[lane]=res;
  }
  workgroupBarrier();
  if(lane==0u){var r5=0.0;for(var j=0u;j<8u;j++){r5+=dR[j];}tSet(PT_R,tCell(slot,5u,0u),r5);}
  workgroupBarrier();
 }
}`,
   ...(BP?{
   // The BULK16 rows, once a solve (after pNative): each region's 16h row
   // of the composite. Its face means take u* (gravity; zero toward a
   // wall); a face toward a region couples at 16h, one toward a resident
   // block through its 16 native rows' links (their flux the wall slot's
   // RHS, buildRhs's), as its 64 4h rows' sum would. After the solve
   // (pBulkProject) the faces between regions project; then (pBulkCommit)
   // the correction joins the pressure.
   bulk:header()+poolWGSL+tail+bulkRows+/* wgsl */`
@group(1) @binding(26) var<storage,read_write> rootArena:array<f32>;
@group(1) @binding(4) var<uniform> authority:vec4f;
var<workgroup> bN:u32;
var<workgroup> bD:array<f32,48>;
var<workgroup> bW:atomic<u32>;
// A kick's re-solve keeps the frame's u* (pBulkProject projected the face means from it).
override pResolve:u32=0u;
var<workgroup> bSpeed:atomic<u32>;
var<workgroup> bError:atomic<u32>;
fn bSum6(v:f32,w:f32,x:f32,y:f32,z:f32,t:f32)->f32{return ((v+w)+(x+y))+(z+t);}
// Momentum at 16h: one semi-Lagrangian trace a face mean (RK2 midpoint, as
// momentum's trace: a face mean moves under half a 16h cell a step, below
// the promote's reach, so one segment is the substep trace). Positions are
// page-local in 16h cells; component a's nodes sit at integer a and
// half-integer tangential coordinates. A node off the page reads the BULK16
// page there, and holds at the page (its nearest own node) elsewhere: the
// rule every resident reader keeps at a BULK16 page, from this side.
fn bMean(s:u32,a:u32,m:vec3i)->f32{return bsF(s,BS_U+12u*a+4u*u32(m[a])+u32(m[(a+1u)%3u])+2u*u32(m[(a+2u)%3u]));}
fn bNode(s:u32,p:vec3i,a:u32,n:vec3i)->f32{
 var hi=vec3i(1);hi[a]=2;var l=n;
 if(any(l<vec3i(0))||any(l>hi)){
  let d=select(select(vec3i(0),vec3i(1),l>hi),vec3i(-1),l<vec3i(0));let o=bsWordSlot(umPageRawWord(p+d));
  let m=l-2*d;if(o!=0u&&all(m>=vec3i(0))&&all(m<=hi)){return bMean(o,a,m);}
  l=clamp(l,vec3i(0),hi);
 }
 return bMean(s,a,l);
}
fn bSample(s:u32,p:vec3i,a:u32,x:vec3f)->f32{
 var q=x-vec3f(0.5);q[a]=x[a];let b=vec3i(floor(q));let f=q-vec3f(b);
 var c:array<f32,8>;
 for(var k=0u;k<8u;k++){c[k]=bNode(s,p,a,b+vec3i(i32(k&1u),i32((k>>1u)&1u),i32(k>>2u)));}
 let y0=mix(mix(c[0],c[1],f.x),mix(c[2],c[3],f.x),f.y);let y1=mix(mix(c[4],c[5],f.x),mix(c[6],c[7],f.x),f.y);
 return mix(y0,y1,f.z);
}
fn bVelocity(s:u32,p:vec3i,x:vec3f)->vec3f{return vec3f(bSample(s,p,0u,x),bSample(s,p,1u,x),bSample(s,p,2u,x));}
// Face mean k (bpFace order) of slot s's page p, advected over dt.
fn bAdvect(s:u32,p:vec3i,k:u32)->f32{
 let a=k/12u;var x=vec3f(0.5);x[a]=f32((k%12u)/4u);x[(a+1u)%3u]+=f32(k&1u);x[(a+2u)%3u]+=f32((k>>1u)&1u);
 let cell=16.0*params.hDt.xyz;let dt=params.hDt.w;
 let mid=x-0.5*dt*bVelocity(s,p,x)/cell;
 return bSample(s,p,a,x-dt*bVelocity(s,p,mid)/cell);
}
@compute @workgroup_size(64) fn pBulkRows(${lanes}){
 if(lane==0u){bN=bsCount();atomicStore(&bW,0u);}
 let n=workgroupUniformLoad(&bN);
 for(var e=group.x;e<n;e+=groups.x){
  let slot=bsSlot(e);let page=bsPage(slot);
  if(lane==0u){atomicStore(&bSpeed,0u);atomicStore(&bError,0u);}
  workgroupBarrier();
  if(lane<36u&&pResolve==0u){
   let a=lane/12u;let plane=(lane%12u)/4u;var es=slot;var ep=page;var ek=lane;
   // Open: the face's regions on this page open (a closed one's faces are
   // walls), and across the page a BULK16 page's open region or a stored block.
   var r=vec3u(0u);r[(a+1u)%3u]=lane&1u;r[(a+2u)%3u]=(lane>>1u)&1u;var lo=r;var hi=r;hi[a]=1u;
   var open=select(bsOpen(slot,bRegionIndex(lo)),true,plane==2u)&&select(bsOpen(slot,bRegionIndex(hi)),true,plane==0u);
   if(plane!=1u){
    var d=vec3i(0);d[a]=select(-1,1,plane==2u);let w=umPageRawWord(page+d);let o=bsWordSlot(w);
    open=open&&select(umBlockStored(w),bsOpen(o,bRegionIndex(select(lo,hi,plane==0u))),o!=0u);
    // A face two BULK16 pages share is traced in the low page's frame by
    // both, so the two copies stay one value.
    if(plane==0u&&bsWordSlot(w)!=0u){es=bsWordSlot(w);ep=page+d;ek=lane+8u;}
   }
   let u=select(0.0,bAdvect(es,ep,ek)+params.hDt.w*params.gravity[a],open);
   bsSet(slot,BS_US+lane,u);atomicMax(&bSpeed,bitcast<u32>(abs(bsF(slot,BS_U+lane))));
  }
  storageBarrier();workgroupBarrier();
  // The bulk record's error test one level up (uniform-bulk-tier): the
  // page's face means, its velocity as a frozen member's stored faces are,
  // on the middle plane against the linear prolongation of the outer two
  // on their axis, under UNIFORM_BULK_TOLERANCE of the page's speed plus
  // UNIFORM_BULK_FLOOR h a step. A member leaves at twice it: the page
  // asks for its block back (its promote word, the next commit's).
  // A line through a closed region has no linear profile to test (its faces are walls).
  if(lane<12u&&pResolve==0u){
   let a=lane/4u;let k=12u*a+(lane&3u);let lo=bsF(slot,BS_U+k);let mid=bsF(slot,BS_U+k+4u);let hi=bsF(slot,BS_U+k+8u);
   var r=vec3u(0u);r[(a+1u)%3u]=lane&1u;r[(a+2u)%3u]=(lane>>1u)&1u;let i=bRegionIndex(r);r[a]=1u;
   if(bsOpen(slot,i)&&bsOpen(slot,bRegionIndex(r))){atomicMax(&bError,bitcast<u32>(abs(mid-0.5*(lo+hi))));}
  }
  workgroupBarrier();
  if(lane==0u&&pResolve==0u){
   let h=min(params.hDt.x,min(params.hDt.y,params.hDt.z));
   let threshold=${UNIFORM_BULK_TOLERANCE}*bitcast<f32>(atomicLoad(&bSpeed))+${UNIFORM_BULK_FLOOR}*h/params.hDt.w;
   // Its promote word says why: 1 the structural scan (bulkPromote), 2 this test
   // (0 when the particles' closure took it back first).
   let at=bsAt(slot,${BULK16_PAGE.promote}u);let error=bitcast<f32>(atomicLoad(&bError));
   if(error>2.0*threshold&&poolB[at]==0u){
    poolB[at]=2u;poolB[bsAt(slot,${BULK16_PAGE.promoteError}u)]=bitcast<u32>(error);poolB[bsAt(slot,${BULK16_PAGE.promoteLevel}u)]=bitcast<u32>(2.0*threshold);
   }
  }
  if(lane<48u){
   let i=lane/6u;let f=lane%6u;let a=f/2u;let near=bNear(slot,page,i,f);let p=bsF(slot,BS_P+i);var d=0.0;var r=0.0;
   if(near.kind==B_REGION){
    let k=bRegionK(f);d=k;
    r=k*(bsF(near.slot,BS_P+near.region)-p)-256.0*params.policy.x*f32(bSign(f))*bsF(slot,BS_US+bFace(i,f))/(params.hDt[a]*params.hDt.w);
   }else if(near.kind==B_BLOCK){
    for(var j=0u;j<16u;j++){
     let t=bNativeTile(i,f,j);let at=UBA_STRIDE*near.block+bTileLocal(t);
     if(bLinked(at,f)){let c=qF(PN_K+(f^1u)*PLANE2,at);let s=N_KEYS+umFaceTile(near.block,f^1u,t);d+=c;r+=c*(rootArena[R_MINIMUM+s]-p)+64.0*rootArena[R_RHS+s];}
    }
   }
   bD[lane]=d;bRed[lane]=r;
  }
  workgroupBarrier();
  if(lane<8u){
   let k=6u*lane;let d=bSum6(bD[k],bD[k+1u],bD[k+2u],bD[k+3u],bD[k+4u],bD[k+5u]);
   let r=bSum6(bRed[k],bRed[k+1u],bRed[k+2u],bRed[k+3u],bRed[k+4u],bRed[k+5u])+4096.0*params.policy.x*authority.z/params.hDt.w;
   bsSet(slot,BS_T4+24u+lane,d);bsSet(slot,BS_RHS+lane,r);
   for(var c=0u;c<3u;c++){bsSet(slot,BS_T3+8u*c+lane,0.0);bsSet(slot,BS_T4+8u*c+lane,0.0);}
   atomicMax(&bW,bResidualMagnitude(r/64.0*nScale()));${UniformPressureArena.diagnostics?"var cls=7u;for(var f=0u;f<6u;f++){let kind=bNear(slot,page,lane,f).kind;if(kind==B_NONE&&cls==7u){cls=1u;}if(kind==B_BLOCK){cls=2u;}}":"let cls=7u;"}pDiag(7u,cls,bResidualMagnitude(r/64.0*nScale()));pDiag(6u,cls,bResidualMagnitude(1.2e-7*2.0*d*abs(bsF(slot,BS_P+lane))/64.0*nScale()));
  }
  workgroupBarrier();
 }
 if(lane==0u){let r=atomicLoad(&bW);if(r!=0u){atomicMax(&state[INITIAL],select(r,0x7f7fffffu,bitcast<f32>(r)>nFloor()));atomicMax(&state[NATIVE_INITIAL],r);}}
}
// The pressure either side of a page's face lane (bpFace order), and whether both are BULK16 regions.
@compute @workgroup_size(64) fn pBulkProject(${lanes}){
 if(lane==0u){bN=bsCount();}
 let n=workgroupUniformLoad(&bN);
 for(var e=group.x;e<n;e+=groups.x){
  let slot=bsSlot(e);
  if(lane<36u){
   let a=lane/12u;let plane=(lane%12u)/4u;var r=vec3u(0u);r[(a+1u)%3u]=lane&1u;r[(a+2u)%3u]=(lane>>1u)&1u;
   var low=slot;var high=slot;var rl=r;var rh=r;rh[a]=1u;
   if(plane!=1u){
    var d=vec3i(0);d[a]=select(-1,1,plane==2u);let other=bsWordSlot(umPageRawWord(bsPage(slot)+d));
    if(plane==0u){low=other;rl[a]=1u;rh[a]=0u;}else{high=other;rl[a]=1u;rh[a]=0u;}
   }
   // A face toward a resident block: its sixteen subfaces' projected flux
   // over the face's area (the reflux sum the region's row balanced, as
   // the bulk tier's face means): each linked row's c (p - across) plus
   // its u* flux (pBulkRows' term), in velocity. Toward a wall: zero.
   if(low!=0u&&high!=0u){
    let jl=bRegionIndex(rl);let jh=bRegionIndex(rh);
    // A face of a closed region is a wall.
    if(!bsOpen(low,jl)||!bsOpen(high,jh)){bsSet(slot,BS_U+lane,0.0);continue;}
    let rise=(bsF(high,BS_P+jh)+bsF(high,BS_N+jh))-(bsF(low,BS_P+jl)+bsF(low,BS_N+jl));
    bsSet(slot,BS_U+lane,bsF(slot,BS_US+lane)-params.hDt.w/params.policy.x*rise/(16.0*params.hDt[a]));
   }else{
    var u=0.0;
    if(plane!=1u){
     let f=2u*a+select(0u,1u,plane==2u);let i=bRegionIndex(select(rh,rl,plane==2u));let near=bNear(slot,bsPage(slot),i,f);
     if(near.kind==B_BLOCK){
      var flux=0.0;
      for(var j=0u;j<16u;j++){
       let t=bNativeTile(i,f,j);let at=UBA_STRIDE*near.block+bTileLocal(t);
       if(bLinked(at,f)){
        let s=N_KEYS+umFaceTile(near.block,f^1u,t);let p=qF(PN_B,at)+qF(PN_X,at);
        let across=bsF(slot,BS_P+i)+bsF(slot,BS_N+i)+qF(PN_B,at)-rootArena[R_MINIMUM+s];
        flux+=qF(PN_K+(f^1u)*PLANE2,at)*(p-across)+64.0*rootArena[R_RHS+s];
       }
      }
      // Into the region through face f: -sign(f) along the axis; a row's
      // flux is 64 rho V u/(4h dt), and the face sixteen subfaces.
      u=-f32(bSign(f))*flux*4.0*params.hDt[a]*params.hDt.w/(1024.0*params.policy.x);
     }
    }
    bsSet(slot,BS_U+lane,u);
   }
  }
 }
}
// Eight live slots a workgroup (a lane a region): each region its own.
@compute @workgroup_size(64) fn pBulkCommit(${lanes}){
 if(lane==0u){bN=bsCount();}
 let n=workgroupUniformLoad(&bN);let i=lane&7u;
 for(var e=8u*group.x+(lane>>3u);e<n;e+=8u*groups.x){
  let slot=bsSlot(e);bsSet(slot,BS_P+i,bsF(slot,BS_P+i)+bsF(slot,BS_N+i));bsSet(slot,BS_N+i,0.0);
 }
}`,
   // Before pUpB: a native row linked to a BULK16 row takes that row's
   // level 3 correction into its level 2 RHS. (Its 16h and 32h rows bake in
   // pDown2.)
   bulkLevels:header()+poolWGSL+tail+bulkRows+/* wgsl */`
var<workgroup> bN:u32;
var<workgroup> bF:u32;
@compute @workgroup_size(${WG}) fn pBulkLift(${lanes}){
 if(lane==0u){bN=select(ubaLevelCount(2u),0u,bConverged());}
 let n=workgroupUniformLoad(&bN);
 for(var e=group.x;e<n;e+=groups.x){
  let block=ubaLevelSlot(2u,e);
  if(lane==0u){bF=nPage(block);}
  if((workgroupUniformLoad(&bF)&8u)!=0u){
   let at2=UBA_STRIDE*block+lane;let m=poolB[PN_M+at2];
   if((m&N_CORRECT)!=0u&&(m&N_ACTIVE)!=0u){
    var lift=0.0;
    for(var f=0u;f<6u;f++){if(nKind(m,f)==N_BULKN){lift+=qF(PN_K+f*PLANE2,at2)*tF(PT_Y,poolB[PN_L+f*PLANE2+at2]|16u);}}
    if(lift!=0.0){qSet(PB_R,at2,qF(PB_R,at2)+lift);}
   }
  }
  workgroupBarrier();
 }
}`,
   }:{}),
   // Level 2's band channels of the held blocks, a workgroup a (block, channel).
   clear2:header()+poolWGSL+/* wgsl */`
const P2_FIRST:u32=${this.level2Clear.first}u;const P2_CHANNELS:u32=${this.level2Clear.channels}u;const P2_CHANNEL:u32=${UBA_STRIDE*this.simulation.capacity.address.blocks}u;
var<workgroup> p2Block:u32;
@compute @workgroup_size(64) fn pClear2(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) l:u32){
 for(var item=group.x;item<P2_CHANNELS*umHeldBlockCount();item+=groups.x){
  if(l==0u){p2Block=umHeldBlock(item/P2_CHANNELS);}
  let base=P2_FIRST+P2_CHANNEL*(item%P2_CHANNELS)+UBA_STRIDE*workgroupUniformLoad(&p2Block);
  for(var k=l;k<UBA_STRIDE;k+=64u){poolB[base+k]=0u;}
 }
}`,
   // 32h, the coarsest level (a row a page with rows): one workgroup solves
   // them by red-black sweeps, row e the page list's e-th page at sX[e+1]
   // (sX[0] stays zero: a missing neighbour's, so the six loads take no
   // branch); a lane keeps its first row in registers and reloads the others.
   tail:header()+poolWGSL+tail+/* wgsl */`
var<workgroup> sX:array<f32,${TAIL_ROWS+1}>;
var<workgroup> tN:u32;
var<workgroup> tM:u32;
// The row (staged index + 1, 0: none) across face f of item block's.
fn sNear(block:u32,f:u32)->u32{
 var d=vec3i(0);d[f/2u]=bSign(f);let near=tAt(block,5u,d);
 if(near==UB_NONE_ID){return 0u;}return tU(PT_Y,near);
}
fn sColour(block:u32)->u32{${BP?"if((block&BA)!=0u){let page=bsPage(block&0x7fffffffu);return u32(page.x+page.y+page.z)&1u;}":""}let page=umBlockPage(block);return (page.x+page.y+page.z)&1u;}
// Row e's update: (r + sum k x_n) / d.
fn sRow(block:u32,at:u32,d:f32)->f32{
 var off=tF(PT_R,at);for(var f=0u;f<6u;f++){off+=tF(PT_K+f*PLANE2,at)*sX[sNear(block,f)];}
 return off/d;
}
// Rows: the page list's, then the BULK16 pages' (tItem).
@compute @workgroup_size(${lanesTail}) fn pTail(@builtin(local_invocation_index) lane:u32){
 if(lane==0u){tM=ubaLevelCount(2u);tN=select(tCount(tM),0u,bConverged());}
 let P=workgroupUniformLoad(&tN);let n2=workgroupUniformLoad(&tM);if(P==0u){return;}
 // A row's staged index + 1, in its page's spare 32h word of the 8h up
 // channel (a BULK16 row's BS_T5+1, where the super-block tail's sUp6 puts
 // its correction: the two tails are exclusive, the constructor's invariant).
 for(var e=lane;e<P;e+=${lanesTail}u){let at=tCell(tItem(e,n2),5u,0u);tSetU(PT_Y,at,select(0u,e+1u,tF(PT_D,at)>0.0));sX[e+1u]=0.0;}
 if(lane==0u){sX[0]=0.0;}
 storageBarrier();workgroupBarrier();
 var d0=0.0;var r0=0.0;var k0=array<f32,6>(0.0,0.0,0.0,0.0,0.0,0.0);var n0=array<u32,6>(0u,0u,0u,0u,0u,0u);var c0=0u;
 if(lane<P){
  let block=tItem(lane,n2);let at=tCell(block,5u,0u);d0=tF(PT_D,at);c0=sColour(block);
  if(d0>0.0){r0=tF(PT_R,at);for(var f=0u;f<6u;f++){n0[f]=sNear(block,f);k0[f]=tF(PT_K+f*PLANE2,at);}}
 }
 for(var sweep=0u;sweep<${TAIL_SWEEPS}u;sweep++){
  for(var col=0u;col<2u;col++){
   if(d0>0.0&&c0==col){var off=r0;for(var f=0u;f<6u;f++){off+=k0[f]*sX[n0[f]];}sX[lane+1u]=off/d0;}
   for(var e=lane+${lanesTail}u;e<P;e+=${lanesTail}u){
    let block=tItem(e,n2);let at=tCell(block,5u,0u);let d=tF(PT_D,at);
    if(d>0.0&&sColour(block)==col){sX[e+1u]=sRow(block,at,d);}
   }
   workgroupBarrier();
  }
 }
 for(var e=lane;e<P;e+=${lanesTail}u){let at=tCell(tItem(e,n2),5u,0u);if(tF(PT_D,at)>0.0){tSet(PT_X,at,sX[e+1u]);}}
}`,
   // 16h up (a page a group): the 32h correction prolonged into the page's
   // 16h rows and their ring (two deep: the neighbours' rows plus their own
   // prolonged correction), the rows' sweeps against the ring, into Y.
   up4:header()+poolWGSL+tail+/* wgsl */`
override pTailY:bool=false;
var<workgroup> vX:array<f32,216>;
var<workgroup> uN:u32;
fn vBox(g:vec3i)->u32{let q=vec3u(g+2);return q.x+6u*(q.y+6u*q.z);}
fn uWeight(d:vec3i)->f32{return select(0.75,0.25,d.x==1)*select(0.75,0.25,d.y==1)*select(0.75,0.25,d.z==1);}
var<workgroup> uM:u32;
@compute @workgroup_size(${TAIL_LANES}) fn pUp4(${lanes}){
 if(lane==0u){uM=ubaLevelCount(2u);uN=select(tCount(uM),0u,bConverged());}
 let n=workgroupUniformLoad(&uN);let n2=workgroupUniformLoad(&uM);
 for(var e=group.x;e<n;e+=groups.x){
  let slot=tItem(e,n2);
  if(lane==0u){uN=u32(tPageLive(slot));}
  if(workgroupUniformLoad(&uN)==0u){continue;}
  if(lane<216u){
   let g=vec3i(vec3u(lane%6u,(lane/6u)%6u,lane/36u))-2;let at=tAt(slot,4u,g);var x=0.0;
   if(at!=UB_NONE_ID&&tF(PT_D,at)>0.0){
    x=tF(PT_X,at);
    let G=g>>vec3u(1u);let o=select(vec3i(-1),vec3i(1),(g&vec3i(1))==vec3i(1));var sum=0.0;var total=0.0;
    for(var i=0u;i<8u;i++){
     let dd=vec3i(vec3u(i&1u,(i>>1u)&1u,i>>2u));let q=G+dd*o;if(any(abs(q)>vec3i(1))){continue;}
     let m=tAt(slot,5u,q);if(m!=UB_NONE_ID&&tF(PT_D,m)>0.0){sum+=uWeight(dd)*tF(select(PT_X,PT_Y,pTailY),m);total+=uWeight(dd);}
    }
    if(total>0.0){x+=sum/total;}
   }
   vX[lane]=x;
  }
  let g4=tLocal(lane,2u);let at4=tCell(slot,4u,lane);var d4=0.0;var r4=0.0;var k4=array<f32,6>(0.0,0.0,0.0,0.0,0.0,0.0);
  if(lane<8u){d4=tF(PT_D,at4);if(d4>0.0){r4=tF(PT_R,at4);for(var f=0u;f<6u;f++){k4[f]=tF(PT_K+f*PLANE2,at4);}}}
  workgroupBarrier();
  for(var s=0u;s<${schedule.sweeps}u;s++){
   for(var col=0u;col<2u;col++){
    if(lane<8u&&d4>0.0&&u32(g4.x+g4.y+g4.z)%2u==col){
     var off=r4;for(var f=0u;f<6u;f++){var q=g4;q[f/2u]+=bSign(f);off+=k4[f]*vX[vBox(q)];}
     vX[vBox(g4)]=off/d4;
    }
    workgroupBarrier();
   }
  }
  if(lane<8u){tSet(PT_Y,at4,vX[vBox(g4)]);}
  workgroupBarrier();
 }
}`,
   // 8h up: the neighbours' 16h results prolonged into the page's 8h rows
   // and their ring (one deep), the rows' sweeps against the ring; the
   // result into the other channel of the 8h iterate.
   up3:header()+poolWGSL+tail+/* wgsl */`
override pSrcY:bool=false;
var<workgroup> uX:array<f32,216>;
var<workgroup> uN:u32;
fn uBox(g:vec3i)->u32{let q=vec3u(g+1);return q.x+6u*(q.y+6u*q.z);}
fn uWeight(d:vec3i)->f32{return select(0.75,0.25,d.x==1)*select(0.75,0.25,d.y==1)*select(0.75,0.25,d.z==1);}
@compute @workgroup_size(${TAIL_LANES}) fn pUp3(${lanes}){
 if(lane==0u){uN=select(ubaLevelCount(2u),0u,bConverged());}
 let n=workgroupUniformLoad(&uN);
 let src=select(PT_X,PT_Y,pSrcY);let dst=select(PT_Y,PT_X,pSrcY);
 for(var e=group.x;e<n;e+=groups.x){
  let slot=ubaLevelSlot(2u,e);
  if(lane==0u){uN=u32(tPageLive(slot));}
  if(workgroupUniformLoad(&uN)==0u){continue;}
  if(lane<216u){
   let g=vec3i(vec3u(lane%6u,(lane/6u)%6u,lane/36u))-1;let at=tAt(slot,3u,g);var x=0.0;
   if(at!=UB_NONE_ID&&tF(PT_D,at)>0.0){
    x=tF(src,at);
    // A BULK16 row's 8h iterate takes its own 16h correction (it holds no finer mode).
    if(${BP?"baIs(at)":"false"}){x+=tF(PT_Y,at&~16u);}
    else{
    let G=g>>vec3u(1u);let o=select(vec3i(-1),vec3i(1),(g&vec3i(1))==vec3i(1));var sum=0.0;var total=0.0;
    for(var i=0u;i<8u;i++){
     let dd=vec3i(vec3u(i&1u,(i>>1u)&1u,i>>2u));let q=G+dd*o;
     let m=tAt(slot,4u,q);if(m!=UB_NONE_ID&&tF(PT_D,m)>0.0){sum+=uWeight(dd)*tF(PT_Y,m);total+=uWeight(dd);}
    }
    if(total>0.0){x+=sum/total;}
    }
   }
   uX[lane]=x;
  }
  let g=tLocal(lane,4u);let at3=UBA_STRIDE*slot+lane;var d3=0.0;var r3=0.0;var k3=array<f32,6>(0.0,0.0,0.0,0.0,0.0,0.0);
  if(lane<64u){d3=qF(PT_D,at3);if(d3>0.0){r3=qF(PT_R,at3);for(var f=0u;f<6u;f++){k3[f]=qF(PT_K+f*PLANE2,at3);}}}
  workgroupBarrier();
  for(var s=0u;s<${schedule.sweeps}u;s++){
   for(var col=0u;col<2u;col++){
    if(lane<64u&&d3>0.0&&u32(g.x+g.y+g.z)%2u==col){
     var off=r3;for(var f=0u;f<6u;f++){var q=g;q[f/2u]+=bSign(f);off+=k3[f]*uX[uBox(q)];}
     uX[uBox(g)]=off/d3;
    }
    workgroupBarrier();
   }
  }
  if(lane<64u){qSet(dst,at3,uX[uBox(g)]);}
  workgroupBarrier();
 }
${BP?`  // A BULK16 page's 8h iterates: their 16h correction added.
 if(lane==0u){uN=select(bsCount(),0u,bConverged());}
 let nb=workgroupUniformLoad(&uN);
 for(var e=group.x;e<nb;e+=groups.x){
  let slot=bsSlot(e);
  if(lane<8u){let at=baOf(slot,true,lane);tSet(dst,at,tF(src,at)+bsF(slot,BS_T4+8u+lane));}
 }
`:""}}`,
   // Level 2 up: the 8h correction prolonged into the page's box, then its sweeps.
   upB:header()+poolWGSL+coarse(2,schedule.coarsestSweeps)+tail+(B?bulkMean("tS")+"var<workgroup> uBulk:u32;\n":"")+/* wgsl */`
var<workgroup> tS:array<f32,1024>;
${prolong("tS[tBox(q)]","tS[512u+tBox(q)]")}
fn tBox(q:vec3i)->u32{let u=vec3u(q+2);return u.x+8u*(u.y+8u*u.z);}
@compute @workgroup_size(${WG}) fn pUpB(${lanes}){
 if(lane==0u){cN=select(ubaLevelCount(2u),0u,bConverged());}
 let n=workgroupUniformLoad(&cN);
 for(var e=group.x;e<n;e+=groups.x){
  let slot=ubaLevelSlot(2u,e);
  if(lane==0u){cN=u32(tPageLive(slot));${B?"uBulk=nPage(slot)&4u;":""}}
  if(workgroupUniformLoad(&cN)==0u){continue;}
  cRecords(slot,lane);
  {let g=vec3i(vec3u(lane&7u,(lane>>3u)&7u,lane>>6u))-2;let m=tAt(slot,3u,g);var x=0.0;var d=0.0;if(m!=UB_NONE_ID){x=tF(PT_Y,m);d=tF(PT_D,m);}tS[lane]=x;tS[512u+lane]=d;}
  workgroupBarrier();
  for(var k=lane;k<1728u;k+=${WG}u){let c=cStaged(k);let at=cAddr(c);var x=0.0;if(at!=UB_NONE_ID){x=qF(CX,at);if(qF(CD,at)>0.0){x+=cProlong(c);}}cX[k]=x;}
  let red=cBoxCell(0u,lane);let black=cBoxCell(1u,lane);let cR0=cBoxRow(0u,lane);let cB0=cBoxRow(1u,lane);
  workgroupBarrier();
  cSweeps(red,cR0,black,cB0);
  // Every native row takes the coupled correction; a closed row stays
  // at or above zero. A native-only stop would invalidate the hierarchy.
  let at2=UBA_STRIDE*slot+lane;let m=poolB[PN_M+at2];
  let y=cX[cBox(cCore(lane))];ubaStore1(CY,slot,lane,y);
${B?"  // A member region takes its lanes' mean correction: one pressure.\n  var yb=y;if(workgroupUniformLoad(&uBulk)!=0u){yb=nBulkMean(y,(m&N_BULK)!=0u,lane);}\n":""}  if((m&N_CORRECT)!=0u){var x=qF(PN_X,at2)+${B?"yb":"y"};if((m&N_CLOSED)!=0u){x=max(x,-qF(PN_B,at2));}qSet(PN_X,at2,x);}
  workgroupBarrier();
 }
${BP?`  // A BULK16 page's rows: their level 3 iterate is their correction.
 if(lane==0u){cN=select(bsCount(),0u,bConverged());}
 let nb=workgroupUniformLoad(&cN);
 for(var e=group.x;e<nb;e+=groups.x){
  let slot=bsSlot(e);
  if(lane<8u){bsSet(slot,BS_N+lane,bsF(slot,BS_N+lane)+bsF(slot,BS_T3+8u+lane));}
 }
`:""}}`,
   // Level 1 up: level 2's correction (the parent octant, two deep) prolonged into the brick's box, then its sweeps.
   upA:header()+poolWGSL+coarse(1)+a.stageWGSL("q",{level:2,region:"octant",channels:[{name:"yb",halo:2,seam:"pZeroSeam"},{name:"db",halo:2,seam:"pZeroSeam"}],workgroup:WG})+
    prolong("ubaW_q[ubaAt_q(q,0u)]","ubaW_q[ubaAt_q(q,1u)]")+/* wgsl */`
@compute @workgroup_size(${WG}) fn pUpA(${lanes}){
 if(lane==0u){cN=select(ubaLevelCount(1u),0u,bConverged());}
 let n=workgroupUniformLoad(&cN);
 for(var e=group.x;e<n;e+=groups.x){
  let slot=ubaLevelSlot(1u,e);let parent=ubaParent(1u,slot);
  ubaStage_q(parent.x,lane,parent.y);
  cRecords(slot,lane);
  // Across a seam (no level 1 row there, or one with no diagonal), the
  // native row's level 2 correction.
  for(var k=lane;k<1728u;k+=${WG}u){let c=cStaged(k);let at=cAddr(c);var x=0.0;if(cDiag(at)>0.0){x=qF(CX,at)+cProlong(c);}else{let P=c>>vec3u(1u);x=select(0.0,ubaW_q[ubaAt_q(P,0u)],ubaW_q[ubaAt_q(P,1u)]>0.0);}cX[k]=x;}
  let red=cBoxCell(0u,lane);let black=cBoxCell(1u,lane);let cR0=cBoxRow(0u,lane);let cB0=cBoxRow(1u,lane);
  workgroupBarrier();
  cSweeps(red,cR0,black,cB0);
  ubaStore1(CY,slot,lane,cX[cBox(cCore(lane))]);
 }
}`,
   // Level 0 up, a workgroup a band tile: level 1's correction prolonged
   // (inside rows clamp) from the 4^3 level 1 cells around the tile's 2^3.
   prolong:header()+poolWGSL+/* wgsl */`
var<workgroup> rN:u32;
var<workgroup> qY:array<f32,64>;
var<workgroup> qD:array<f32,64>;
// The pool offset of level 1 cell r of slot1's brick (-8..15 per axis):
// UB_NONE_ID outside a live brick's live octants.
fn p1At(slot1:u32,r:vec3i)->u32{
 var s=slot1;let d=vec3i(vec3u(r+8)>>vec3u(3u))-1;
 if(any(d!=vec3i(0))){s=ubaLevelRecord(1u,slot1,u32((d.x+1)+3*(d.y+1)+9*(d.z+1)));if(s>=UBA_SEAM){return UB_NONE_ID;}}
 let l=vec3u(r&vec3i(7));let o=l>>vec3u(2u);
 if(((ubaLevelRecord(1u,s,28u)>>(o.x|(o.y<<1u)|(o.z<<2u)))&1u)==0u){return UB_NONE_ID;}
 return pAddr(s,l);
}
// pDst: the plane the corrected iterate goes to (the post-smoothing's first read).
override pDst:u32=0u;
@compute @workgroup_size(${TILE_WG}) fn pProlong0(${lanes}){
 if(lane==0u){rN=pTileLive();}
 let n=workgroupUniformLoad(&rN);
 let l=vec3u(lane&3u,(lane>>2u)&3u,lane>>4u);
 for(var j=group.x;j<n;j+=groups.x){
  let w=pIndex(TLIST+j);let slot=pTileSlot(w);let lb=pTileCell(w,l);
  let parent=ubaParent(0u,slot);let o=parent.y;
  let q=p1At(parent.x,vec3i(4u*vec3u(o&1u,(o>>1u)&1u,o>>2u)+2u*(lb>>vec3u(2u))+l)-1);
  var y=0.0;var d=0.0;if(q!=UB_NONE_ID){y=qF(PA_Y,q);d=qF(PA_D,q);}
  qY[lane]=y;qD[lane]=d;
  workgroupBarrier();
  let at=pAddr(slot,lb);let kinds=pool[P_KINDS+at];
  if(pActive(kinds)){
   let P=vec3i(l>>vec3u(1u))+1;let s=select(vec3i(-1),vec3i(1),(vec3i(l)&vec3i(1))==vec3i(1));var sum=0.0;var total=0.0;
   for(var i=0u;i<8u;i++){
    let e=vec3i(vec3u(i&1u,(i>>1u)&1u,i>>2u));let r=vec3u(P+e*s);let m=r.x+4u*(r.y+4u*r.z);
    let wt=select(0.75,0.25,e.x==1)*select(0.75,0.25,e.y==1)*select(0.75,0.25,e.z==1);
    if(qD[m]>0.0){sum+=wt*qY[m];total+=wt;}
   }
   var x=pF(P_X,at);if((kinds&P_CORRECT)!=0u&&total>0.0){x+=sum/total;}
   if(bInside(kinds)){x=max(x,-pF(P_BASE,at));}
   pSet(select(P_X,P_X1,pDst==1u),at,x);
  }
  workgroupBarrier();
 }
}`,
   project:header()+poolWGSL+zero+UniformColdFreeze.wgsl(this.cold,1,14)+/* wgsl */`
@group(1) @binding(7) var velocity:texture_3d<f32>;
@group(1) @binding(15) var field:texture_3d<f32>;
@group(1) @binding(16) var output:texture_storage_3d<rgba32float,write>;
@group(1) @binding(17) var<storage,read_write> boundary:array<f32>;
@group(1) @binding(21) var<storage,read_write> presented:array<f32>;
// UniformPressureBand's umProject on face f of band cell 'at' at p.
fn bFace(at:u32,p:vec3i,f:u32)->f32{
 let kinds=pool[P_KINDS+at];let kind=bFaceKind(kinds,f);
 if(kind==K_NEUMANN){return pForced(at,f);}
 let axis=f/2u;let sign=bSign(f);let h=params.hDt[axis];let liquid=bLiquid(kinds);
 var anchor=p;if(sign<0){anchor[axis]-=1;}
 // The regional correction owns frozen F+T. Do not bake a second pressure
 // correction (or T) into its stored F when publishing a live band's face.
 if(!umFaceLow(anchor,axis)&&umFrozenFace(anchor,axis,1u)){return textureLoad(velocity,anchor,0)[axis];}
 ${S?"if(kind==K_CLOSED){var low=p;if(sign<0){low[axis]-=1;}return umSolidFaceVelocity(low,axis);}":""}
 let forced=pForced(at,f);let coefficient=pF(P_C+f*PLANE,at)${S?"/pVolume(at,f)":""};let own=pF(P_X,at);let base=pF(P_BASE,at);let scale=params.policy.z;
 if(kind==K_WALL){if(!liquid){return forced;}return forced-scale*f32(sign)*(max(own+pLift(at,f),-base)-own)/h;}
 if(kind==K_OPEN){if(!liquid){return forced;}return forced+scale*f32(sign)*(base+own)*coefficient*h;}
 var other=0.0;var otherBase=0.0;var otherLiquid=false;
 if(kind==K_BAND){let m=pNear(at,p,f);other=pF(P_X,m);otherBase=pF(P_BASE,m);otherLiquid=bLiquid(pool[P_KINDS+m]);}
 else if(kind==K_SEAM){let m=nAcross(p,f);other=qF(PN_X,m);otherBase=qF(PN_B,m)+bSeamHead(at,f);otherLiquid=true;}
 if(!liquid&&!otherLiquid){return 0.0;}
 let value=forced-scale*f32(sign)*((otherBase-base)+(other-own))*coefficient*h;
 return value;
}
// The composite solve couples a native row to the individual h faces.
// Restrict those same projected fluxes, rather than differentiating the
// presentation-only mean pressure of the band tile with a second theta.
fn bRootFace(p:vec3i,axis:u32,sign:i32)->f32{
 var other=p;other[axis]+=4*sign;var anchor=p;anchor[axis]+=select(-1,3,sign>0);
 if(!pBandCell(other)){
  if(umFaceLow(anchor,axis)){return boundary[umLowFaceIndex(vec3u(p),axis)];}
  return /*4h*/textureLoad(field,anchor,0)[axis];
 }
 var sum=0.0;var weight=0.0;let u=(axis+1u)%3u;let v=(axis+2u)%3u;
 let f=2u*axis+select(1u,0u,sign>0);
 for(var j=0u;j<16u;j++){
  var q=other;q[axis]+=select(3,0,sign>0);q[u]+=i32(j&3u);q[v]+=i32(j>>2u);
  let at=pAt(q);let w=pVolume(at,f);sum+=w*bFace(at,q,f);weight+=w;
 }
 return select(0.0,sum/max(weight,1e-30),weight>1e-6);
}
// A job a band tile: its six faces' root tiles. Only a root tile beside the
// band does work, so the band's faces reach every one of them; the root's
// first band face owns it (one visit a root tile). A root face across the
// band takes the volume-weighted mean of the band tile's 16 h fluxes there
// (the composite solve couples the native row to those same projected
// fluxes): the 96 subfaces of a root a lane each, summed in bRootFace's order.
var<workgroup> bRootTests:array<atomic<u32>,6>;
var<workgroup> bRootOwned:u32;
var<workgroup> bRootValues:array<f32,6>;
var<workgroup> bRootSlot:array<u32,6>;
var<workgroup> bRootBits:array<u32,3>;
var<workgroup> bRootW:array<f32,96>;
var<workgroup> bRootV:array<f32,96>;
var<workgroup> bRootCount:u32;
@compute @workgroup_size(${TILE_WG}) fn pProjectNative(${lanes}){
 if(lane==0u){bRootCount=pTileCount();}
 let n=workgroupUniformLoad(&bRootCount);let l=vec3u(lane&3u,(lane>>2u)&3u,lane>>4u);
 for(var j=group.x;j<n;j+=groups.x){
  let w=pIndex(TLIST+j);let origin=vec3i(ubaOrigin(ubaBrick(pTileSlot(w)))+pTileCell(w,vec3u(0u)));
  if(lane<6u){atomicStore(&bRootTests[lane],0u);}
  workgroupBarrier();
  // Face f's root, test t a lane: bit t a band cell across the root's face t, bit 6 no stored tile, a held or a frozen one.
  if(lane<42u){
   let f=lane/7u;let t=lane%7u;var p=origin;p[f/2u]+=4*bSign(f);
   if(t<6u){var q=p;q[t/2u]+=4*bSign(t);if(pBandCell(q)){atomicOr(&bRootTests[f],1u<<t);}}
   else if(!umCellPresent(p)||pIndex(SLOTS+umTileAt(vec3u(p)/4u))!=0u||umFrozenCell(p)){atomicOr(&bRootTests[f],64u);}
  }
  workgroupBarrier();
  if(lane==0u){var owned=0u;for(var f=0u;f<6u;f++){let tests=atomicLoad(&bRootTests[f]);if((tests&64u)==0u&&(tests&((1u<<(f^1u))-1u))==0u){owned|=1u<<f;}}bRootOwned=owned;}
  var owned=workgroupUniformLoad(&bRootOwned);
  while(owned!=0u){
   let f=firstTrailingBit(owned);owned&=owned-1u;
   var p=origin;p[f/2u]+=4*bSign(f);let band=atomicLoad(&bRootTests[f]);
   // A root face a lane: a band neighbour's brick slot, or its 4h (or low boundary) value.
   if(lane<6u){
    let axis=lane/2u;let sign=bSign(lane);var other=p;other[axis]+=4*sign;var slot=UB_NONE_ID;
    if((band&(1u<<lane))!=0u){slot=ubaSlot(ubaBrickAt(other));}
    else{var anchor=p;anchor[axis]+=select(-1,3,sign>0);
     if(umFaceLow(anchor,axis)){bRootValues[lane]=boundary[umLowFaceIndex(vec3u(p),axis)];}else{bRootValues[lane]=/*4h*/textureLoad(field,anchor,0)[axis];}}
    bRootSlot[lane]=slot;
   }
   if(lane<3u){var q=p;q[lane]+=3;bRootBits[lane]=u32(max(0.0,/*4h*/textureLoad(field,q,0).w));}
   workgroupBarrier();
   for(var k=lane;k<96u;k+=64u){
    let g=k/16u;let slot=bRootSlot[g];if(slot==UB_NONE_ID){continue;}
    let axis=g/2u;let sign=bSign(g);let u=(axis+1u)%3u;let v=(axis+2u)%3u;let i=k&15u;
    let face=2u*axis+select(1u,0u,sign>0);
    var q=p;q[axis]+=select(-1,4,sign>0);q[u]+=i32(i&3u);q[v]+=i32(i>>2u);
    let at=pAddr(slot,vec3u(q)&vec3u(7u));bRootW[k]=pVolume(at,face);bRootV[k]=bFace(at,q,face);
   }
   workgroupBarrier();
   if(lane<6u&&bRootSlot[lane]!=UB_NONE_ID){
    var sum=0.0;var weight=0.0;for(var i=0u;i<16u;i++){let wv=bRootW[16u*lane+i];sum+=wv*bRootV[16u*lane+i];weight+=wv;}
    bRootValues[lane]=select(0.0,sum/max(weight,1e-30),weight>1e-6);
   }
   workgroupBarrier();
   let tile=umTileAt(vec3u(p)/4u);
   if(umTileWidth(tile)==1u){
    var value=vec4f(0.0);var bits=0u;
    for(var axis=0u;axis<3u;axis++){
     value[axis]=mix(bRootValues[2u*axis],bRootValues[2u*axis+1u],f32(l[axis]+1u)/4.0);
     if(l[axis]==3u){bits|=bRootBits[axis]&(1u<<axis);}
     if(l[axis]==0u){bits|=bRootBits[axis]&(1u<<(axis+3u));}
    }
    value.w=f32(bits);textureStore(output,p+vec3i(l),value);
   }else if(lane<3u){
    var q=p;q[lane]+=3;var value=vec4f(0.0);value[lane]=bRootValues[2u*lane+1u];value.w=f32(bRootBits[lane]);textureStore(output,q,value);
   }
   workgroupBarrier();
  }
 }
}
// The reference form (a job a band tile and face, bRootFace's serial 16
// subfaces): compiled only for the equivalence check (referenceEntries).
var<workgroup> bRefValues:array<f32,6>;
var<workgroup> bRefBits:array<u32,3>;
var<workgroup> bRefCount:u32;
var<workgroup> bRefTests:atomic<u32>;
@compute @workgroup_size(${TILE_WG}) fn pProjectNativeRef(${lanes}){
 // A job a band tile and face: the root tile across it. Only a root tile
 // beside the band does work, so the band's faces reach every one of them;
 // the root's first band face owns it (one visit a root tile).
 if(lane==0u){bRefCount=6u*pTileCount();}
 let n=workgroupUniformLoad(&bRefCount);let l=vec3u(lane&3u,(lane>>2u)&3u,lane>>4u);
 for(var j=group.x;j<n;j+=groups.x){
  let w=pIndex(TLIST+j/6u);let f=j%6u;
  var p=vec3i(ubaOrigin(ubaBrick(pTileSlot(w)))+pTileCell(w,vec3u(0u)));p[f/2u]+=4*bSign(f);
  // The skip's seven tests a lane each: bit f a band cell across face f, bit 6 no stored tile, a held or a frozen one.
  if(lane==0u){atomicStore(&bRefTests,0u);}
  workgroupBarrier();
  if(lane<6u){var q=p;q[lane/2u]+=4*bSign(lane);if(pBandCell(q)){atomicOr(&bRefTests,1u<<lane);}}
  else if(lane==6u){if(!umCellPresent(p)||pIndex(SLOTS+umTileAt(vec3u(p)/4u))!=0u||umFrozenCell(p)){atomicOr(&bRefTests,64u);}}
  workgroupBarrier();
  if(lane==0u){let tests=atomicLoad(&bRefTests);bRefCount=u32((tests&64u)!=0u||(tests&((1u<<(f^1u))-1u))!=0u);}
  if(workgroupUniformLoad(&bRefCount)!=0u){continue;}
  let tile=umTileAt(vec3u(p)/4u);
  if(lane<6u){bRefValues[lane]=bRootFace(p,lane/2u,bSign(lane));}
  if(lane<3u){var q=p;q[lane]+=3;bRefBits[lane]=u32(max(0.0,/*4h*/textureLoad(field,q,0).w));}
  workgroupBarrier();
  if(umTileWidth(tile)==1u){
   var value=vec4f(0.0);var bits=0u;
   for(var axis=0u;axis<3u;axis++){
    value[axis]=mix(bRefValues[2u*axis],bRefValues[2u*axis+1u],f32(l[axis]+1u)/4.0);
    if(l[axis]==3u){bits|=bRefBits[axis]&(1u<<axis);}
    if(l[axis]==0u){bits|=bRefBits[axis]&(1u<<(axis+3u));}
   }
   value.w=f32(bits);textureStore(output,p+vec3i(l),value);
  }else if(lane<3u){
   var q=p;q[lane]+=3;var value=vec4f(0.0);value[lane]=bRefValues[2u*lane+1u];value.w=f32(bRefBits[lane]);textureStore(output,q,value);
  }
  workgroupBarrier();
 }
}
fn bReleased(at:u32,p:vec3i,f:u32,value:f32)->bool{
 let axis=f/2u;let sign=bSign(f);let kinds=pool[P_KINDS+at];
 ${S?`let volume=pVolume(at,f);
 if(sign>0){
  var q=p;q[axis]+=1;let own=umCellIsOpen(p);
  if(own==umCellIsOpen(q)||volume<=1e-6){return false;}
  var pressure=0.0;let kind=bFaceKind(kinds,f);
  if(kind==K_WALL||kind==K_OPEN){pressure=select(0.0,max(pF(P_X,at)+pLift(at,f),-pF(P_BASE,at))+pF(P_BASE,at),bLiquid(kinds)&&kind==K_WALL);}
  else if(own){let b=pAt(q);if(b!=UB_NONE_ID&&bLiquid(pool[P_KINDS+b])){pressure=pF(P_X,b)+pF(P_BASE,b);}}
  else if(bLiquid(kinds)){pressure=pF(P_X,at)+pF(P_BASE,at);}
  return pressure<=0.0&&select(1.0,-1.0,own)*(value-umSolidFaceVelocity(p,axis))*params.hDt.w>1e-4*params.hDt[axis];
 }
 if(!umCellIsOpen(p)||volume<=1e-6){return false;}`:""}
 if(bFaceKind(kinds,f)!=K_WALL){return false;}
 let pressure=select(0.0,max(pF(P_X,at)+pLift(at,f),-pF(P_BASE,at))+pF(P_BASE,at),bLiquid(kinds));
 return pressure<=0.0&&-f32(sign)*value*params.hDt.w>1e-4*params.hDt[axis];
}
// After the 4h projection reaches the simulation (and pCopy): the band
// faces into the velocity, the band pressures into the stage grids, last
// frame's pressure for the next start.
@compute @workgroup_size(${TILE_WG}) fn pProject(${lanes}){
 if(lane==0u){pN=pTileCount();}
 let n=workgroupUniformLoad(&pN);let l=vec3u(lane&3u,(lane>>2u)&3u,lane>>4u);
 for(var j=group.x;j<n;j+=groups.x){
  let w=pIndex(TLIST+j);let slot=pTileSlot(w);let lb=pTileCell(w,l);let at=pAddr(slot,lb);
  let p=vec3i(ubaOrigin(ubaBrick(slot))+lb);let kinds=pool[P_KINDS+at];let own=pF(P_X,at);let base=pF(P_BASE,at);
  presented[params.capacity.w+64u*j+lane]=base+own;
  pSet(P_WARMP,at,base+own);
  var texel=vec4f(0.0);var released=0u;let lowOpen=umLowOpen(p);
  for(var axis=0u;axis<3u;axis++){
   let value=bFace(at,p,2u*axis+1u);texel[axis]=value;
   if(bReleased(at,p,2u*axis+1u,value)){released|=1u<<axis;}
   if(lowOpen[axis]){
    let low=bFace(at,p,2u*axis);boundary[umLowFaceIndex(vec3u(p),axis)]=low;
    if(bReleased(at,p,2u*axis,low)){released|=1u<<(axis+3u);}
   }
  }
  texel.w=f32(released);textureStore(output,p,texel);
  // A non-band texel below this cell: the first band cell along x, then y, then z writes its band faces.
  for(var axis=0u;axis<3u;axis++){
   var q=p;q[axis]-=1;if(!umCellPresent(q)||pBandCell(q)){continue;}
   var first=true;for(var b=0u;b<axis;b++){var r=q;r[b]+=1;if(pBandCell(r)){first=false;}}
   if(!first){continue;}
   var t=textureLoad(field,q,0);
   for(var b=0u;b<3u;b++){var r=q;r[b]+=1;let m=pAt(r);if(m!=UB_NONE_ID){t[b]=bFace(m,r,2u*b);}}
   textureStore(output,q,t);
  }
 }
}`,
  };
  if(W){
   const base=header()+poolWGSL+tail+superWGSL;
   Object.assign(sources,{
    // Topology contains every physical pressure page, including pages whose
    // correction rows are all held this cycle and may release next cycle.
    // Its group's first page lists the 64h row and its children.
    group6:base+/* wgsl */`
// A thread a tail item (the page list's blocks, then the BULK16 slots).
@compute @workgroup_size(64) fn sGroup6(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) groups:vec3u){
 let n2=ubaLevelCount(2u);let n=select(tCount(n2),0u,bConverged());
 for(var e=gid.x;e<n;e+=64u*groups.x){
  let item=tItem(e,n2);if(!tPageLive(item)){continue;}
  let rel=sRel(item);let o=rel&vec3i(1);let mine=u32(o.x)|(u32(o.y)<<1u)|(u32(o.z)<<2u);
  var kids=array<u32,8>(UB_NONE_ID,UB_NONE_ID,UB_NONE_ID,UB_NONE_ID,UB_NONE_ID,UB_NONE_ID,UB_NONE_ID,UB_NONE_ID);var first=8u;
  for(var j=0u;j<8u;j++){
   var b=item;
   if(j!=mine){b=sNeighbour(item,vec3i(sOctant(j))-o);}
   if(b!=UB_NONE_ID&&tPageLive(b)){kids[j]=b;first=min(first,j);}
  }
  if(first!=mine){continue;}
  let row=atomicAdd(&state[S_COUNT6],1u);if(row>=CAP6){continue;}
  let key=sPack6(rel>>vec3u(1u));atomicStore(&state[S_LIST6+row],key);atomicStore(&state[S_MAP6+key],row+1u);
  for(var j=0u;j<8u;j++){poolB[S6_C+j*CAP6+row]=kids[j];}
 }
}`,
    // A thread a 64h row: its operator from its pages' 32h rows; its 128h
    // group's first 64h row (by octant) lists the 128h row and its children.
    group7:base+/* wgsl */`
override sList:bool=true;
@compute @workgroup_size(64) fn sGroup7(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) groups:vec3u){
 let n=sCount6();
 for(var e=gid.x;e<n;e+=64u*groups.x){
  var d=0.0;var k=array<f32,6>(0.0,0.0,0.0,0.0,0.0,0.0);
  for(var j=0u;j<8u;j++){
   let b=sChild6(e,j);if(b==UB_NONE_ID){continue;}
   let at=tCell(b,5u,0u);var part=tF(PT_D,at);
   for(var axis=0u;axis<3u;axis++){
    let bit=(j>>axis)&1u;
    if(sChild6(e,j^(1u<<axis))!=UB_NONE_ID){part-=tF(PT_K+(2u*axis+1u-bit)*PLANE2,at);}
    k[2u*axis+bit]+=tF(PT_K+(2u*axis+bit)*PLANE2,at);
   }
   d+=part;
  }
  var outward=0.0;for(var f=0u;f<6u;f++){outward+=k[f];}qSet(S6_D,e,pAggregateDiagonal(d,outward,S_SCALE));for(var f=0u;f<6u;f++){qSet(S6_K+f*CAP6,e,S_SCALE*k[f]);}
  if(!sList){continue;}
  let g=sKey6(e);let o=g&vec3i(1);let mine=u32(o.x)|(u32(o.y)<<1u)|(u32(o.z)<<2u);
  var kids=array<u32,8>(UB_NONE_ID,UB_NONE_ID,UB_NONE_ID,UB_NONE_ID,UB_NONE_ID,UB_NONE_ID,UB_NONE_ID,UB_NONE_ID);var first=8u;
  for(var j=0u;j<8u;j++){let r=select(sRow6(g-o+vec3i(sOctant(j))),e,j==mine);if(r!=UB_NONE_ID){kids[j]=r;first=min(first,j);}}
  if(first!=mine){continue;}
  let row=atomicAdd(&state[S_COUNT7],1u);if(row>=CAP7){continue;}
  let key=sPack7(g>>vec3u(1u));atomicStore(&state[S_LIST7+row],key);atomicStore(&state[S_MAP7+key],row+1u);
  for(var j=0u;j<8u;j++){poolB[S7_C+j*CAP7+row]=kids[j];}
 }
}`,
    // A thread a 128h row: its operator from its 64h rows'.
    bake7:base+/* wgsl */`
@compute @workgroup_size(64) fn sBake7(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) groups:vec3u){
 let n=sCount7();
 for(var e=gid.x;e<n;e+=64u*groups.x){
  var d=0.0;var k=array<f32,6>(0.0,0.0,0.0,0.0,0.0,0.0);
  for(var j=0u;j<8u;j++){
   let r=sChild7(e,j);if(r==UB_NONE_ID){continue;}
   var part=qF(S6_D,r);
   for(var axis=0u;axis<3u;axis++){
    let bit=(j>>axis)&1u;
    if(sChild7(e,j^(1u<<axis))!=UB_NONE_ID){part-=qF(S6_K+(2u*axis+1u-bit)*CAP6,r);}
    k[2u*axis+bit]+=qF(S6_K+(2u*axis+bit)*CAP6,r);
   }
   d+=part;
  }
  var outward=0.0;for(var f=0u;f<6u;f++){outward+=k[f];}qSet(S7_D,e,pAggregateDiagonal(d,outward,S_SCALE));for(var f=0u;f<6u;f++){qSet(S7_K+f*CAP7,e,S_SCALE*k[f]);}
 }
}`,
    // A 64h group a workgroup: its pages' 16h residuals (against the
    // neighbour pages' rows, pDown4's) into their 32h RHS, then the 32h rows
    // from zero inside the group (the first block-Jacobi iterate).
    down45:base+/* wgsl */`
var<workgroup> dR:array<f32,64>;
var<workgroup> dX:array<f32,8>;
var<workgroup> dN:u32;
@compute @workgroup_size(64) fn sDown45(${lanes}){
 if(lane==0u){dN=sCount6();}
 let n=workgroupUniformLoad(&dN);
 for(var e=group.x;e<n;e+=groups.x){
  let page=sChild6(e,lane>>3u);var res=0.0;
  if(page!=UB_NONE_ID){
   let c=lane&7u;let g4=tLocal(c,2u);let at4=tCell(page,4u,c);let d4=tF(PT_D,at4);
   if(d4>0.0){
    res=tF(PT_R,at4)-d4*tF(PT_X,at4);
    for(var f=0u;f<6u;f++){var q=g4;q[f/2u]+=bSign(f);let m=tAt(page,4u,q);if(m!=UB_NONE_ID&&tF(PT_D,m)>0.0){res+=tF(PT_K+f*PLANE2,at4)*tF(PT_X,m);}}
   }
  }
  dR[lane]=res;
  workgroupBarrier();
  var at5=0u;var r5=0.0;var d5=0.0;var k5=array<f32,6>(0.0,0.0,0.0,0.0,0.0,0.0);
  if(lane<8u){
   let b=sChild6(e,lane);
   if(b!=UB_NONE_ID){at5=tCell(b,5u,0u);for(var c=0u;c<8u;c++){r5+=dR[8u*lane+c];}tSet(PT_R,at5,r5);d5=tF(PT_D,at5);for(var f=0u;f<6u;f++){k5[f]=tF(PT_K+f*PLANE2,at5);}}
   dX[lane]=0.0;
  }
  workgroupBarrier();
  for(var s=0u;s<${schedule.sweeps}u;s++){
   for(var col=0u;col<2u;col++){
    if(lane<8u&&d5>0.0&&(countOneBits(lane)&1u)==col){
     var off=r5;for(var f=0u;f<6u;f++){let j=sSibling(lane,f);if(j<8u&&sChild6(e,j)!=UB_NONE_ID){off+=k5[f]*dX[j];}}
     dX[lane]=off/d5;
    }
    workgroupBarrier();
   }
  }
  if(lane<8u&&d5>0.0){tSet(PT_X,at5,dX[lane]);}
  workgroupBarrier();
 }
}`,
    // A 128h group a workgroup: its pages' 32h residuals (against the
    // neighbour pages') into their 64h rows' RHS, then the 64h rows from zero
    // inside the group.
    down6:base+/* wgsl */`
var<workgroup> dR:array<f32,64>;
var<workgroup> dX:array<f32,8>;
var<workgroup> dN:u32;
@compute @workgroup_size(64) fn sDown6(${lanes}){
 if(lane==0u){dN=sCount7();}
 let n=workgroupUniformLoad(&dN);
 for(var e=group.x;e<n;e+=groups.x){
  let r6=sChild7(e,lane>>3u);var res=0.0;
  if(r6!=UB_NONE_ID){
   let page=sChild6(r6,lane&7u);
   if(page!=UB_NONE_ID){
    let at5=tCell(page,5u,0u);let d5=tF(PT_D,at5);
    res=tF(PT_R,at5)-d5*tF(PT_X,at5);
    for(var f=0u;f<6u;f++){let m=tAt(page,5u,sFace(f));if(m!=UB_NONE_ID&&tF(PT_D,m)>0.0){res+=tF(PT_K+f*PLANE2,at5)*tF(PT_X,m);}}
   }
  }
  dR[lane]=res;
  workgroupBarrier();
  var row=UB_NONE_ID;var r=0.0;var d=0.0;var k=array<f32,6>(0.0,0.0,0.0,0.0,0.0,0.0);
  if(lane<8u){
   row=sChild7(e,lane);
   if(row!=UB_NONE_ID){for(var j=0u;j<8u;j++){r+=dR[8u*lane+j];}qSet(S6_R,row,r);d=qF(S6_D,row);for(var f=0u;f<6u;f++){k[f]=qF(S6_K+f*CAP6,row);}}
   dX[lane]=0.0;
  }
  workgroupBarrier();
  for(var s=0u;s<${schedule.sweeps}u;s++){
   for(var col=0u;col<2u;col++){
    if(lane<8u&&d>0.0&&(countOneBits(lane)&1u)==col){
     var off=r;for(var f=0u;f<6u;f++){let j=sSibling(lane,f);if(j<8u&&sChild7(e,j)!=UB_NONE_ID){off+=k[f]*dX[j];}}
     dX[lane]=off/d;
    }
    workgroupBarrier();
   }
  }
  if(lane<8u&&d>0.0){qSet(S6_X,row,dX[lane]);}
  workgroupBarrier();
 }
}`,
    // Eight lanes a 128h row: its 64h rows' residuals against the neighbour
    // 64h rows, summed into its RHS.
    down7:base+/* wgsl */`
var<workgroup> dR:array<f32,64>;
var<workgroup> dN:u32;
@compute @workgroup_size(64) fn sDown7(${lanes}){
 if(lane==0u){dN=sCount7();}
 let n=workgroupUniformLoad(&dN);
 for(var first=8u*group.x;first<n;first+=8u*groups.x){
  let e=first+(lane>>3u);var res=0.0;
  if(e<n){
   let r=sChild7(e,lane&7u);
   if(r!=UB_NONE_ID){
    let d=qF(S6_D,r);res=qF(S6_R,r)-d*qF(S6_X,r);let g=sKey6(r);
    for(var f=0u;f<6u;f++){let m=sRow6(g+sFace(f));if(m!=UB_NONE_ID&&qF(S6_D,m)>0.0){res+=qF(S6_K+f*CAP6,r)*qF(S6_X,m);}}
   }
  }
  dR[lane]=res;
  workgroupBarrier();
  if((lane&7u)==0u&&e<n){var sum=0.0;for(var j=0u;j<8u;j++){sum+=dR[lane+j];}qSet(S7_R,e,sum);}
  workgroupBarrier();
 }
}`,
    // 128h, the coarsest level (a row a super-block with rows): one
    // workgroup's red-black sweeps from zero, as pTail's over the 32h rows.
    tail7:base+/* wgsl */`
var<workgroup> sX:array<f32,${TAIL_ROWS+1}>;
var<workgroup> tN:u32;
// The row (index + 1, 0: none) across face f of row e.
fn sNear7(e:u32,f:u32)->u32{let m=sRow7(sKey7(e)+sFace(f));if(m==UB_NONE_ID||qF(S7_D,m)<=0.0){return 0u;}return m+1u;}
fn sRow7Next(e:u32,d:f32)->f32{var off=qF(S7_R,e);for(var f=0u;f<6u;f++){off+=qF(S7_K+f*CAP7,e)*sX[sNear7(e,f)];}return off/d;}
@compute @workgroup_size(${lanesTail}) fn sTail7(@builtin(local_invocation_index) lane:u32){
 if(lane==0u){tN=sCount7();}
 let P=workgroupUniformLoad(&tN);if(P==0u){return;}
 for(var e=lane;e<P;e+=${lanesTail}u){sX[e+1u]=0.0;}
 if(lane==0u){sX[0]=0.0;}
 workgroupBarrier();
 var d0=0.0;var r0=0.0;var k0=array<f32,6>(0.0,0.0,0.0,0.0,0.0,0.0);var n0=array<u32,6>(0u,0u,0u,0u,0u,0u);var c0=0u;
 if(lane<P){d0=qF(S7_D,lane);c0=sParity(sKey7(lane));if(d0>0.0){r0=qF(S7_R,lane);for(var f=0u;f<6u;f++){n0[f]=sNear7(lane,f);k0[f]=qF(S7_K+f*CAP7,lane);}}}
 for(var sweep=0u;sweep<${TAIL_SWEEPS}u;sweep++){
  for(var col=0u;col<2u;col++){
   if(d0>0.0&&c0==col){var off=r0;for(var f=0u;f<6u;f++){off+=k0[f]*sX[n0[f]];}sX[lane+1u]=off/d0;}
   for(var e=lane+${lanesTail}u;e<P;e+=${lanesTail}u){let d=qF(S7_D,e);if(d>0.0&&sParity(sKey7(e))==col){sX[e+1u]=sRow7Next(e,d);}}
   workgroupBarrier();
  }
 }
 for(var e=lane;e<P;e+=${lanesTail}u){if(qF(S7_D,e)>0.0){qSet(S7_X,e,sX[e+1u]);}}
}`,
    // 64h up (a 128h group a workgroup): each 64h row and its ring (the
    // neighbour rows outside the group) take the 128h correction, trilinear
    // and renormalised; the group's rows sweep against the frozen ring, into Y.
    up7:base+/* wgsl */`
var<workgroup> vS:array<f32,56>;
var<workgroup> vX:array<f32,8>;
var<workgroup> uN:u32;
fn sProlong7(g:vec3i)->f32{
 let G=g>>vec3u(1u);let o=select(vec3i(-1),vec3i(1),(g&vec3i(1))==vec3i(1));var sum=0.0;var total=0.0;
 for(var i=0u;i<8u;i++){let dd=vec3i(sOctant(i));let m=sRow7(G+dd*o);if(m!=UB_NONE_ID&&qF(S7_D,m)>0.0){sum+=sWeight(dd)*qF(S7_X,m);total+=sWeight(dd);}}
 return select(0.0,sum/total,total>0.0);
}
@compute @workgroup_size(64) fn sUp7(${lanes}){
 if(lane==0u){uN=sCount7();}
 let n=workgroupUniformLoad(&uN);
 for(var e=group.x;e<n;e+=groups.x){
  // Lane 7c+k: child c's own start (k 0) or its ring's across face k-1.
  if(lane<56u){
   let c=lane/7u;let k=lane%7u;let r=sChild7(e,c);var x=0.0;
   if(r!=UB_NONE_ID){
    var g=sKey6(r);var m=r;
    if(k>0u){if(sSibling(c,k-1u)<8u){m=UB_NONE_ID;}else{g+=sFace(k-1u);m=sRow6(g);}}
    if(m!=UB_NONE_ID&&qF(S6_D,m)>0.0){x=qF(S6_X,m)+sProlong7(g);}
   }
   vS[lane]=x;
  }
  workgroupBarrier();
  var row=UB_NONE_ID;var r=0.0;var d=0.0;var k=array<f32,6>(0.0,0.0,0.0,0.0,0.0,0.0);
  if(lane<8u){row=sChild7(e,lane);if(row!=UB_NONE_ID){r=qF(S6_R,row);d=qF(S6_D,row);for(var f=0u;f<6u;f++){k[f]=qF(S6_K+f*CAP6,row);}}vX[lane]=vS[7u*lane];}
  workgroupBarrier();
  for(var s=0u;s<${schedule.sweeps}u;s++){
   for(var col=0u;col<2u;col++){
    if(lane<8u&&d>0.0&&(countOneBits(lane)&1u)==col){
     var off=r;
     for(var f=0u;f<6u;f++){let j=sSibling(lane,f);if(j<8u){if(sChild7(e,j)!=UB_NONE_ID){off+=k[f]*vX[j];}}else{off+=k[f]*vS[7u*lane+1u+f];}}
     vX[lane]=off/d;
    }
    workgroupBarrier();
   }
  }
  if(lane<8u&&d>0.0){qSet(S6_Y,row,vX[lane]);}
  workgroupBarrier();
 }
}`,
    // 32h up (a 64h group a workgroup): each page's 32h row and its ring take
    // the 64h correction (Y), trilinear and renormalised; the group's rows
    // sweep against the frozen ring, into the 32h Y word pUp4S reads.
    up6:base+/* wgsl */`
var<workgroup> vS:array<f32,56>;
var<workgroup> vX:array<f32,8>;
var<workgroup> uN:u32;
fn sProlong6(g:vec3i)->f32{
 let G=g>>vec3u(1u);let o=select(vec3i(-1),vec3i(1),(g&vec3i(1))==vec3i(1));var sum=0.0;var total=0.0;
 for(var i=0u;i<8u;i++){let dd=vec3i(sOctant(i));let m=sRow6(G+dd*o);if(m!=UB_NONE_ID&&qF(S6_D,m)>0.0){sum+=sWeight(dd)*qF(S6_Y,m);total+=sWeight(dd);}}
 return select(0.0,sum/total,total>0.0);
}
@compute @workgroup_size(64) fn sUp6(${lanes}){
 if(lane==0u){uN=sCount6();}
 let n=workgroupUniformLoad(&uN);
 for(var e=group.x;e<n;e+=groups.x){
  if(lane<56u){
   let c=lane/7u;let k=lane%7u;let page=sChild6(e,c);var x=0.0;
   if(page!=UB_NONE_ID){
    var at=tCell(page,5u,0u);var b=page;
    if(k>0u){if(sSibling(c,k-1u)<8u){at=UB_NONE_ID;}else{at=tAt(page,5u,sFace(k-1u));b=sItemOf(at);}}
    if(at!=UB_NONE_ID&&tF(PT_D,at)>0.0){x=tF(PT_X,at)+sProlong6(sRel(b));}
   }
   vS[lane]=x;
  }
  workgroupBarrier();
  var at5=0u;var r=0.0;var d=0.0;var k=array<f32,6>(0.0,0.0,0.0,0.0,0.0,0.0);
  if(lane<8u){let b=sChild6(e,lane);if(b!=UB_NONE_ID){at5=tCell(b,5u,0u);r=tF(PT_R,at5);d=tF(PT_D,at5);for(var f=0u;f<6u;f++){k[f]=tF(PT_K+f*PLANE2,at5);}}vX[lane]=vS[7u*lane];}
  workgroupBarrier();
  for(var s=0u;s<${schedule.sweeps}u;s++){
   for(var col=0u;col<2u;col++){
    if(lane<8u&&d>0.0&&(countOneBits(lane)&1u)==col){
     var off=r;
     for(var f=0u;f<6u;f++){let j=sSibling(lane,f);if(j<8u){if(sChild6(e,j)!=UB_NONE_ID){off+=k[f]*vX[j];}}else{off+=k[f]*vS[7u*lane+1u+f];}}
     vX[lane]=off/d;
    }
    workgroupBarrier();
   }
  }
  // Into the 32h Y word pUp4S reads (a BULK16 row's BS_T5+1, which the
  // page tail's pTail stages its row index in: the two tails are exclusive).
  if(lane<8u&&d>0.0){tSet(PT_Y,at5,vX[lane]);}
  workgroupBarrier();
 }
}`,
    // After the solve: the map entries this solve's lists wrote, back to
    // none (the counts are cleared before the next solve).
    forget:base+/* wgsl */`
@compute @workgroup_size(64) fn sForget(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) groups:vec3u){
 let n6=min(pIndex(S_COUNT6),CAP6);let n7=min(pIndex(S_COUNT7),CAP7);
 for(var e=gid.x;e<n6;e+=64u*groups.x){atomicStore(&state[S_MAP6+pIndex(S_LIST6+e)],0u);}
 for(var e=gid.x;e<n7;e+=64u*groups.x){atomicStore(&state[S_MAP7+pIndex(S_LIST7+e)],0u);}
}`,
   });
  }
  if(!UNIFORM_PRESSURE_ARENA_NATIVE_RHS)delete sources.nrhs;
  // The page tail and the super-block tail are exclusive: pTail stages a
  // row's index in the 32h Y word sUp6 writes its correction to (a BULK16
  // row's BS_T5+1), so one solve compiles and encodes one of them.
  if(W)delete sources.tail;
  if(("tail" in sources)===!!W||(!W&&"up6" in sources))throw new Error("Pressure arena: the page tail and the super-block tail in one solve");
  else if(!S)throw new Error("The native rows' own RHS reads the coarse solid record (solids are always on)");
  sources.rowsG=sources.rows!.replace("const pGeometry:bool=false;","const pGeometry:bool=true;").replace("fn pRows(","fn pRowsG(");
  const layoutOf:Record<string,string>={bulk:"bulk",nrhs:"nrhs",table:"table",mark:"mark",rows:"rows",rowsG:"rows",init:"init",copy:"copy",native:"native",final:"native",project:"project"};
  const entryOf:Record<string,string[]>={bulk:["pBulkRows","pBulkProject","pBulkCommit"],bulkLevels:["pBulkLift"],nrhs:["pRhs"],table:[...UNIFORM_ARENA_TABLE_ENTRIES,UNIFORM_ARENA_RELEASE_ENTRY],mark:["pMark","pUnmark"],rows:["pRows"],rowsG:["pRowsG"],init:["pInit"],copy:["pCopy","pCopyNative"],native:["pNative","pNativeOperator","pWrite"],residual:["pResidual2","pNativeRelax"],final:["pFinal"],clear2:["pClear2"],
   krylov:["kStart","kApplyT",BP?"kApplyNB":"kApplyN","kReduce","kBound","kStep"],rebase:["pRebase"],relax:["pRelax"],restrict:["pRestrict0"],downA:["pDownA","pDownR"],downB:["pDownB"],down2:["pDown2"],down3:["pDown3"],down4:["pDown4"],tail:["pTail"],up4:["pUp4"],up3:["pUp3"],upB:["pUpB"],upA:["pUpA"],prolong:["pProlong0"],project:["pProject","pProjectNative",...(UniformPressureArena.reference?["pProjectNativeRef"]:[])],
   group6:["sGroup6"],group7:["sGroup7"],bake7:["sBake7"],down45:["sDown45"],down6:["sDown6"],down7:["sDown7"],tail7:["sTail7"],up7:["sUp7"],up6:["sUp6"],forget:["sForget"]};
  type Variant=[string,Record<string,number>];
  const variants:Record<string,Variant[]>={
   pRelax:[["0",{pSrc:0}],["1",{pSrc:1}],["S0",{pSettle:1,pSrc:0}],["S1",{pSettle:1,pSrc:1}]],pProlong0:[["0",{pDst:0}],["1",{pDst:1}]],pRestrict0:Array.from({length:schedule.cycles},(_,k)=>[`@${k}`,{pCycle:k}] as Variant),
   pResidual2:[...Array.from({length:schedule.cycles},(_,k)=>[`@${k}`,{pCycle:k}] as Variant),["F",{pCycle:schedule.cycles,pLast:1}]],pNativeRelax:[["0",{pColour:0}],["1",{pColour:1}],["0X1",{pColour:0,pBand:1}],["1X1",{pColour:1,pBand:1}]],
   ...Object.fromEntries(["kStart","kApplyT","kApplyN","kApplyNB","kBound","kStep"].map(k=>[k,[["0",{kPlane:0}],["1",{kPlane:1}]] as Variant[]])),
   kReduce:Array.from({length:schedule.cycles-1},(_,k)=>[`@${k+1}`,{kCycle:k+1}] as Variant),
   pDownA:[["",{}],["@0",{pBake:1}]],pBulkRows:[["",{}],["P",{pResolve:1}]],pInit:[["",{}],["P",{pPicard:1}]],pRebase:[["",{}],["K",{pStash:1}],["P",{pStash:2}],["C",{pCorrection:1}]],pDownB:[["",{}],["@0",{pBake:1}]],pDown3:[["",{}],["Y",{pSrcY:1}]],pUp3:[["",{}],["Y",{pSrcY:1}]],
   ...(W?{sGroup7:[["",{}],["C",{sList:0}]] as Variant[],pUp4:[["",{}],["S",{pTailY:1}]] as Variant[]}:{}),
  };
  this.verdict=await this.verdictPipeline();
  if(this.launch)this.launcher=await this.launchPipeline();
  await Promise.all(Object.entries(sources).map(async([name,code])=>{
   const module=uniformDetailModule(this.device,{label:`Uniform pressure arena ${name}`,code});
   const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");
   if(errors.length)throw new Error(`Pressure arena ${name}: ${errors.map(m=>`${m.lineNum}: ${m.message}`).join("\n")}`);
   const group=layoutOf[name]??"solver";
   const pipelineLayout=this.device.createPipelineLayout({bindGroupLayouts:[this.simulation.bindLayout,this.layouts.get(group)!,...(this.solid?[this.solid.coarse!.bindLayout]:[])]});
   await Promise.all(entryOf[name]!.flatMap(entry=>(variants[entry]??[["",{}] as Variant]).map(async([suffix,constants])=>{
    this.pipelines.set(entry+suffix,await uniformMixedSolidPipeline(this.solid,s=>uniformDetailPipeline(this.device,this.simulation,{layout:pipelineLayout,
     compute:{module,entryPoint:entry,constants:{umDispatchX:this.simulation.dispatchX,...constants,...s}}}),{solidsOnly:constants.pSettle===1,entry:entry+suffix}));
    this.pipelineGroups.set(entry+suffix,group);
   })));
  }));
  this.bindLaunches();
 }
 private readonly pipelineGroups=new Map<string,string>();
 private verdict?:{pipeline:GPUComputePipeline;group:GPUBindGroup};
 /** The frame's verdict (the old schedule's last gate): the solve's cycle
  * count and residual from the history, a nonfinite residual or a band over
  * capacity latched into the frame status, the 4h projection's gate opened
  * on a healthy record, the acceptance state for the receipt. A solve that
  * ends above its target is not a failure. */
 private async verdictPipeline():Promise<{pipeline:GPUComputePipeline;group:GPUBindGroup}>{
  const f=this.fields,FAIL=UNIFORM_MIXED_FAILURE;
  const module=this.device.createShaderModule({label:"Uniform pressure arena verdict",code:/* wgsl */`
struct BandParams {hDt:vec4f,policy:vec4f,solve:vec4f,capacity:vec4u}
@group(0) @binding(0) var<storage,read_write> state:array<atomic<u32>>;
${uniformMixedFrameStatusWGSL(0,1,"read_write")}
@group(0) @binding(2) var<storage,read_write> support:array<u32>;
@group(0) @binding(3) var<storage,read_write> acceptance:array<u32,8>;
@group(0) @binding(4) var<uniform> params:BandParams;
const HISTORY:u32=${HISTORY_WORD}u;const VALID:u32=${VALID_WORD}u;const INITIAL:u32=${INITIAL_WORD}u;const INITIAL_BIT:u32=${1<<HISTORY}u;
const CYCLES:u32=${this.schedule.cycles}u;const CYCLE:u32=4u;const DONE:u32=${DONE_WORD}u;const SUPPORT:u32=${f.root.supportWord}u;
fn pIndex(i:u32)->u32{return atomicLoad(&state[i]);}
fn bMet(word:u32,mask:u32)->bool{return (pIndex(VALID)&mask)!=0u&&params.solve.x>0.0&&bitcast<f32>(pIndex(word))<=params.solve.x;}
@compute @workgroup_size(1) fn pVerdict(){
 let initial=bMet(INITIAL,INITIAL_BIT);
 var cycle=CYCLES;for(var j=CYCLES;j>0u;j--){if(bMet(HISTORY+j-1u,1u<<(j-1u))){cycle=j-1u;}}
 if(initial){cycle=0u;}
 var residual=pIndex(HISTORY+CYCLES);
 if(cycle<CYCLES){atomicStore(&state[DONE],1u);residual=select(pIndex(HISTORY+cycle),pIndex(INITIAL),initial);}
 // The cycles the solve ran: the frame encodes as many as the solves
 // lately ran, plus one (observeCycles), and each that runs measures its
 // start (its VALID bit). A solve that meets no cycle's measure but meets
 // the final one ran every cycle encoded; one that misses the final too
 // reports CYCLES (it needed more than it ran).
 else if(bMet(HISTORY+CYCLES,1u<<CYCLES)){let ran=countOneBits(pIndex(VALID)&((1u<<CYCLES)-1u));if(ran>0u){cycle=ran;}}
 atomicStore(&state[CYCLE],cycle);atomicStore(&state[2],residual);
 if(!umFrameFailed()){
  if(residual>=0x7f800000u){umLatchFailure(${FAIL.pressureNonfinite}u,residual,0u);}
  if(pIndex(1u)==${STALE_APERTURES}u){umLatchFailure(${FAIL.staleApertures}u,pIndex(0u),0u);}
  else if(pIndex(1u)!=0u){umLatchFailure(${FAIL.bandCapacity}u,pIndex(0u),pIndex(1u));}
 }
 let open=!umFrameFailed();
 support[SUPPORT]=u32(!open);
 if(open){atomicStore(&umStatus[6],atomicLoad(&umStatus[7]));}
 acceptance[0]=residual;acceptance[1]=residual;acceptance[2]=residual;acceptance[4]=0u;acceptance[5]=1u;acceptance[7]=cycle;
}`});
  const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");
  if(errors.length)throw new Error(`Pressure arena verdict: ${errors.map(m=>`${m.lineNum}: ${m.message}`).join("\n")}`);
  const pipeline=await this.device.createComputePipelineAsync({label:"Uniform pressure arena verdict",layout:"auto",compute:{module,entryPoint:"pVerdict"}});
  const group=this.device.createBindGroup({label:"Uniform pressure arena verdict",layout:pipeline.getBindGroupLayout(0),entries:[
   {binding:0,resource:{buffer:this.index}},{binding:1,resource:{buffer:f.status,size:64}},{binding:2,resource:{buffer:f.root.support}},
   {binding:3,resource:{buffer:f.acceptance,size:32}},{binding:4,resource:{buffer:f.params,size:64}}]});
  return {pipeline,group};
 }
 /** pLaunch: the BULK16 solve's launches (LAUNCH) from the level 2 page
  * count (the table's, under its capacity) and the live slot count. */
 private async launchPipeline():Promise<{pipeline:GPUComputePipeline;group:GPUBindGroup}>{
  const blocks=this.simulation.capacity.address.blocks;
  const module=this.device.createShaderModule({label:"Uniform pressure arena BULK16 launches",code:/* wgsl */`
@group(0) @binding(0) var<storage,read> state:array<u32>;
@group(0) @binding(1) var<storage,read> store:array<u32>;
@group(0) @binding(2) var<storage,read_write> launch:array<u32,${LAUNCH.bytes/4}>;
fn put(at:u32,groups:u32){launch[at]=min(groups,${PAGE_GROUPS}u);launch[at+1u]=1u;launch[at+2u]=1u;launch[at+3u]=0u;}
@compute @workgroup_size(1) fn pLaunch(){
 let n2=min(state[${this.arena.pageCountWord(this.tableWord)}u],${blocks}u);let live=store[${BULK16_STATUS.live}u];
 put(${LAUNCH.live/4}u,live);put(${LAUNCH.either/4}u,max(n2,live));put(${LAUNCH.items/4}u,n2+live);put(${LAUNCH.level2/4}u,n2);put(${LAUNCH.live8/4}u,(live+7u)/8u);put(${LAUNCH.items64/4}u,(n2+live+63u)/64u);
}`});
  const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");
  if(errors.length)throw new Error(`Pressure arena launches: ${errors.map(m=>`${m.lineNum}: ${m.message}`).join("\n")}`);
  const pipeline=await this.device.createComputePipelineAsync({label:"Uniform pressure arena BULK16 launches",layout:"auto",compute:{module,entryPoint:"pLaunch"}});
  const group=this.device.createBindGroup({label:"Uniform pressure arena BULK16 launches",layout:pipeline.getBindGroupLayout(0),entries:[
   {binding:0,resource:{buffer:this.index}},{binding:1,resource:{buffer:this.poolB,offset:4*this.storeWord,size:4*BULK16_STATUS.words}},{binding:2,resource:{buffer:this.launch!}}]});
  return {pipeline,group};
 }
 private bindLaunches():void{
  this.launches.clear();
  for(const [name,pipeline] of this.pipelines)this.launches.set(name,{pipeline,group:this.groups.get(this.pipelineGroups.get(name)!)!});
 }
 private dispatch(pass:GPUComputePassEncoder,name:string,launch:UniformArenaLaunch):void{
  const bound=this.launches.get(name);if(!bound)throw new Error(`Pressure arena is not initialized (${name})`);
  if(this.boundPass!==pass||this.boundEpoch!==uniformComputePassEpoch(pass)){this.boundPass=pass;this.boundEpoch=uniformComputePassEpoch(pass);this.boundGroup=undefined;pass.setBindGroup(0,this.simulation.bindGroup);if(this.solid)pass.setBindGroup(2,this.solid.coarse!.bindGroup);}
  const pipeline=this.solid?.select(bound.pipeline)??bound.pipeline;
  if(this.boundGroup!==bound.group){this.boundGroup=bound.group;pass.setBindGroup(1,bound.group.group);}
  if(launch==="tiles"){this.simulation.dispatchBuffered(pass,pipeline,"fine",LIST_GROUPS);return;}
  pass.setPipeline(uniformDetailPick(pipeline));
  if(typeof launch==="number")pass.dispatchWorkgroups(launch);else pass.dispatchWorkgroupsIndirect(this.launch!,launch.indirect);
 }
 /** As the band's: no h tile, or a quiet frame under the freeze, lists none. */
 private get empty():boolean{
  if(this.simulation.capacity.fineTiles===0)return true;
  const ceiling=this.fields.coldFrame?.ceiling;
  return this.cold!==undefined&&ceiling!==undefined&&
   ceiling<(this.fields.warmTiers?UNIFORM_REGION_TIER.hot:UNIFORM_REGION_TIER.warm);
 }
 /** After the split authority (the root's phi) and buildRhs: mark band
  * tiles, update the arena table, assemble the band's rows. The table and
  * the native rows run every frame; an empty band lists no tiles. */
 encodePrepare(encoder:GPUCommandEncoder):void{
  const a=this.arena,step=a.stepClear(this.tableWord);
  if(this.indexClean){
   const unmark=encoder.beginComputePass(UNMARK_PASS);this.dispatch(unmark,"pUnmark",UNIFORM_LIVE_KEY_GROUPS);unmark.end();
   encoder.clearBuffer(this.index,0,4*this.header);
  }else{encoder.clearBuffer(this.index,0,4*this.tableWord);this.indexClean=true;}
  encoder.clearBuffer(this.index,step.offset,step.size);
  const band=!this.empty;
  const pass=encoder.beginComputePass(PREPARE_PASS);
  this.clearLevel2(pass);
  if(band)this.dispatch(pass,"pMark","tiles");
  this.dispatch(pass,"ubaUpdate",UNIFORM_LIVE_KEY_GROUPS);
  this.dispatch(pass,"ubaLists",UNIFORM_LIVE_LANE_GROUPS);
  this.dispatch(pass,"ubaLevel1",Math.min(groupsOf(this.workBricks,64),TABLE_GROUPS));
  // The level 2 slots are the observed blocks (workLevel2), not the pool's: the launch strides the counts.
  this.dispatch(pass,"ubaRecord",Math.min(groupsOf(32*(2*this.workBricks+this.workLevel2),64),4*TABLE_GROUPS));
  if(band){this.dispatch(pass,"pRows",this.tileGroups);this.dispatch(pass,"pRowsG",this.tileGroups);}
  pass.end();
 }
 /** Before the 4h projection: the composite solve (start, native rows,
  * V-cycles, final measure), the native pressures into the root, the verdict.
  * kick: the projected kick has no last frame, so its seam heads (bSeamHead:
  * the start's tangential gradient) start at none; it re-solves
  * UNIFORM_PRESSURE_KICK_RESOLVES times from its own pressure. Each re-solve
  * cuts a resting pool's seam error about 3.3x (rest-step's corner jet: 36
  * mm/s from one solve, 1.0 after three re-solves); a falling one has no
  * gradient to find. Once, at t=0. */
 encodeSolve(encoder:GPUCommandEncoder,kick=false):void{
  this.encodeSolveFrom(encoder,"pInit",kick?"pRebaseK":"pRebase");
  if(!kick)return;
  for(let k=0;k<UNIFORM_PRESSURE_KICK_RESOLVES;k++){
   // The cycle history and the band's level 2 channels start empty, as encodePrepare leaves them.
   encoder.clearBuffer(this.index,4*HISTORY_WORD,4*(GEOMETRY_WORD-HISTORY_WORD));
   const clear=encoder.beginComputePass(PREPARE_PASS);this.clearLevel2(clear);clear.end();
   this.encodeSolveFrom(encoder,"pInitP","pRebaseP");
  }
 }
 /** The held blocks' level 2 band channels to zero (level2Clear). */
 private clearLevel2(pass:GPUComputePassEncoder):void{
  this.dispatch(pass,"pClear2",UNIFORM_LIVE_GROUP_GROUPS);
 }
 private encodeSolveFrom(encoder:GPUCommandEncoder,init:string,rebase:string):void{
  // The page launches stride the composite level 2 (every held block).
  const s=this.schedule,pages=Math.min(this.simulation.capacity.address.blocks,this.pageGroups),g=this.level1Groups;
  const W=this.super,BP=!!this.fields.bulkPages;
  // With BULK16 rows the loops that walk the live slots launch from the
  // counts (pLaunch): a workgroup a live slot, or a level 2 page or a live
  // slot (a loop over both), not the held pages' (fewer than a vast pool's
  // slots). Each reader strides its count, so any launch is the same solve.
  const live:UniformArenaLaunch=BP?{indirect:LAUNCH.live}:pages,either:UniformArenaLaunch=BP?{indirect:LAUNCH.either}:pages;
  const items:UniformArenaLaunch=BP?{indirect:LAUNCH.items}:pages,level2:UniformArenaLaunch=BP?{indirect:LAUNCH.level2}:pages;
  if(W)encoder.clearBuffer(this.index,4*W.index.count6,16);
  // The Krylov dots' partials (a launch past this solve's leaves its own) and header words.
  if(s.krylov){encoder.clearBuffer(this.poolB,4*this.krylovDotWord,4*KRYLOV_DOTS*KRYLOV_PARTIALS);encoder.clearBuffer(this.index,4*KRYLOV_ALPHA_WORD,4*(KRYLOV_RESTARTS_WORD+1-KRYLOV_ALPHA_WORD));}
  encoder.clearBuffer(this.index,4*KRYLOV_SPREAD_WORD,4);
  const pass=encoder.beginComputePass(SOLVE_PASS);
  if(this.launcher){pass.setPipeline(this.launcher.pipeline);pass.setBindGroup(0,this.launcher.group);pass.dispatchWorkgroups(1);this.boundPass=undefined;}
  this.dispatch(pass,init,this.tileGroups);if(UNIFORM_PRESSURE_ARENA_NATIVE_RHS)this.dispatch(pass,"pRhs",pages);this.dispatch(pass,"pNative",pages);if(BP)this.dispatch(pass,init==="pInitP"?"pBulkRowsP":"pBulkRows",live);this.dispatch(pass,rebase,this.tileGroups);
  // One cycle past the most the solve ran lately; a frame needing more ends at its last.
  const cycles=Math.min(s.cycles,this.wideCycles+1);
  // Level 0 ping-pongs X and X1 (pInit zeroes both); a leg of n sweeps from
  // plane src ends in src^(n&1). A pre-smoothing starts from plane n&1 and
  // ends in X for the residual and restriction. The prolongation writes X in
  // place, so a post-smoothing ends where the next pre-smoothing starts,
  // except the last cycle's: it writes plane n&1, and the solve ends in X for
  // the final measure and the projection (as it does when it converges early).
  const odd=s.sweeps&1;
  const sweeps=(settle:boolean,from:number)=>{let src=from;for(let k=settle?-1:0;k<s.sweeps;k++){this.dispatch(pass,(k<0?"pRelaxS":"pRelax")+src,this.relaxGroups);src^=1;}};
  for(let cycle=0;cycle<cycles;cycle++){
   const first=cycle===0;
   const settle=first&&!!this.solid?.present;sweeps(settle,(s.sweeps+(settle?1:0))&1);
   for(let k=0;k<s.nativeSweeps;k++){this.dispatch(pass,"pNativeRelax0",pages);this.dispatch(pass,"pNativeRelax1",pages);}
   this.dispatch(pass,this.residuals[cycle]!,either);this.dispatch(pass,this.restricts[cycle]!,this.tileGroups);
   this.dispatch(pass,"pRebaseC",this.tileGroups);this.dispatch(pass,"pNativeOperator",pages);
   this.dispatch(pass,"pDownA@0",g);this.dispatch(pass,"pDownR",g);this.dispatch(pass,"pDownB@0",pages);this.dispatch(pass,"pDown2",either);
   // Refresh numeric coarse rows each cycle; super-block topology is frame-local.
   if(W){if(first)this.dispatch(pass,"sGroup6",BP?{indirect:LAUNCH.items64}:Math.min(groupsOf(this.simulation.capacity.address.blocks,64),TABLE_GROUPS));this.dispatch(pass,first?"sGroup7":"sGroup7C",groupsOf(W.capacity6,64));this.dispatch(pass,"sBake7",groupsOf(W.capacity7,64));}
   for(let m=0;m<s.tailCycles;m++){
    const y=m%2===1?"Y":"";this.dispatch(pass,"pDown3"+y,either);
    if(W){
     const groups7=Math.min(W.capacity7,PAGE_GROUPS);
     this.dispatch(pass,"sDown45",items);this.dispatch(pass,"sDown6",groups7);this.dispatch(pass,"sDown7",Math.min(groupsOf(W.capacity7,8),PAGE_GROUPS));
     this.dispatch(pass,"sTail7",1);this.dispatch(pass,"sUp7",groups7);this.dispatch(pass,"sUp6",items);this.dispatch(pass,"pUp4S",pages);
    }else{
     // The page tail, never with the super-block tail (the constructor's
     // invariant): its staged index shares sUp6's 32h Y word.
     this.dispatch(pass,"pDown4",items);this.dispatch(pass,"pTail",1);this.dispatch(pass,"pUp4",items);
    }
    this.dispatch(pass,"pUp3"+y,either);
   }
   if(BP)this.dispatch(pass,"pBulkLift",level2);this.dispatch(pass,"pUpB",either);this.dispatch(pass,"pUpA",g);
   const post=cycle===cycles-1?odd:0;this.dispatch(pass,"pProlong0"+post,this.tileGroups);
   // The native rows against the prolonged band (plane post) before its post-smoothing.
   for(let k=0;k<s.nativeSweeps;k++){const band=post?"X1":"";this.dispatch(pass,"pNativeRelax0"+band,pages);this.dispatch(pass,"pNativeRelax1"+band,pages);}
   sweeps(false,post);
   // The Krylov step on the cycle (flexible GCR(1)): the first cycle's end
   // starts it, each later one's correction is its direction.
   if(s.krylov&&cycles>1){
    // kStart, kBound and kStep: the tiles eight to a workgroup, the pages
    // and slots one each. kApplyNB: the slots' groups, then the pages'.
    const end=post^odd,slots=BP?Math.min(this.fields.bulkPages!.layout.capacity,KRYLOV_SEGMENTS[2]):0,rows=Math.max(groupsOf(this.tileGroups,8),pages,slots);
    if(first)this.dispatch(pass,"kStart"+end,rows);
    else{
     this.dispatch(pass,"kApplyT"+end,this.tileGroups);if(BP)this.dispatch(pass,"kApplyNB"+end,pages+slots);else this.dispatch(pass,"kApplyN"+end,pages);
     this.dispatch(pass,`kReduce@${cycle}`,1);
     this.dispatch(pass,"kBound"+end,rows);this.dispatch(pass,"kStep"+end,rows);
    }
   }
  }
  this.dispatch(pass,"pResidual2F",either);this.dispatch(pass,"pFinal",this.tileGroups);this.dispatch(pass,"pWrite",pages);
  if(BP){this.dispatch(pass,"pBulkProject",live);this.dispatch(pass,"pBulkCommit",{indirect:LAUNCH.live8});}
  if(W)this.dispatch(pass,"sForget",groupsOf(W.capacity6,64));
  pass.end();
  const verdict=encoder.beginComputePass(VERDICT_PASS);
  verdict.setPipeline(this.verdict!.pipeline);verdict.setBindGroup(0,this.verdict!.group);verdict.dispatchWorkgroups(1);verdict.end();
 }
 /** After the 4h projection reaches simulation ownership: the band's faces
  * into the velocity, then leaving slots back to the stack. */
 encodeProject(encoder:GPUCommandEncoder):void{
  const pass=encoder.beginComputePass(PROJECT_PASS);
  this.dispatch(pass,"pCopyNative",this.tileGroups);
  this.dispatch(pass,"pProjectNative",this.tileGroups);
  this.dispatch(pass,"pCopy",this.tileGroups);this.dispatch(pass,"pProject",this.tileGroups);
  // Leaving slots return to the stack (after every reader of this frame).
  this.dispatch(pass,"ubaRelease",Math.min(groupsOf(this.workBricks,64),TABLE_GROUPS));
  pass.end();
 }
 /** Equivalence checks (tools/pressure-arena-equivalence.ts): compile the
  * reference forms of rewritten kernels beside them. Off in every solver. */
 static reference=false;
 /** Per-class residual diagnostics in the header (tools; slower). */
 static diagnostics=false;
 /** Equivalence checks: one projection entry (a reference form or its
  * rewrite) over the band as the frame left it. */
 encodeProjectEntry(encoder:GPUCommandEncoder,entry:"pProjectNative"|"pProjectNativeRef"):void{
  const pass=encoder.beginComputePass(PROJECT_PASS);this.dispatch(pass,entry,this.tileGroups);pass.end();
 }
 encodeReceipt(encoder:GPUCommandEncoder,target:GPUBuffer,offset:number):void{encoder.copyBufferToBuffer(this.index,0,target,offset,32);}
 static readonly historyWord=HISTORY_WORD;
 static readonly initialResidualWord=INITIAL_WORD;
 static readonly residualValidityWord=VALID_WORD;
 static readonly initialResidualBit=1<<HISTORY;
 static readonly bricksWord=BRICKS_WORD;
 static readonly level1Word=LEVEL1_WORD;
 static readonly level2Word=LEVEL2_WORD;
 /** The overflow word's value when the band's bricks outnumber the arena's fine slots. */
 static readonly slotOverflow=SLOT_OVERFLOW;
 /** The overflow word's value when a row read a stale solid aperture record. */
 static readonly staleApertures=STALE_APERTURES;
 get slotCapacity():number{return this.arena.fineCapacity;}
 destroy():void{this.index.destroy();this.pool.destroy();this.launch?.destroy();}
}
