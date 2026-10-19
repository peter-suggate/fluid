import { uniformComputePassEpoch } from "./uniform-compute-blit";
import { uniformDetailBindLayout, uniformDetailModule, uniformDetailPipeline, uniformDetailPick, uniformDetailGroup, type UniformDetailGroup } from "./uniform-detail-fields";
import { uniformBufferedWork } from "./uniform-buffered-work";
import {uniformBlockFaceWGSL,uniformFaceCellSlots} from "./uniform-lattice-address";
import type {UniformMixedOwnership} from "./uniform-mixed-ownership";
import {uniformMixedTopologyNeighboursWGSL,uniformMixedTopologyWGSL} from "./uniform-mixed-topology.wgsl";
import {uniformMixedFaceAddressWGSL} from "./uniform-mixed-face-dispatch.wgsl";
import {UNIFORM_MIXED_THETA_MIN,uniformMixedPressureLiquidWGSL,uniformMixedSurfaceThetaWGSL} from "./uniform-mixed-pressure-surface.wgsl";
import {uniformMixedSolidPipeline,uniformMixedSolidWGSL,type UniformMixedSolid} from "./uniform-mixed-solid.wgsl";
import {UNIFORM_BODY_FLUX_REACH,UNIFORM_REGION_DYNAMICS,UNIFORM_REGION_DYNAMICS_WORDS} from "./uniform-mixed-region-dynamics";
import {UNIFORM_COLD_FREEZE,UniformColdFreeze} from "./uniform-cold-freeze";
import type {UniformColdFrame} from "./uniform-cold-frame";
import {UNIFORM_REGION_TIER,uniformRegionHeatWGSL} from "./uniform-region-heat";
import {uniformPressureBandMiddleWGSL} from "./uniform-pressure-band-middle.wgsl";

/** The band's local multigrid: V-cycles h -> 2h (2^3 aggregates per tile) ->
 * 4h (one aggregate per tile), red-black Gauss-Seidel at every level. */
export interface UniformPressureBandSchedule {
 cycles:number;
 /** Red-black h sweeps before and after each coarse correction. */
 fineSweeps:number;
 /** Red-black 2h sweeps before and after the 4h correction. */
 middleSweeps:number;
 /** Red-black sweeps of the per-tile 4h level. */
 coarseSweeps:number;
 /** Their over-relaxation (1: Gauss-Seidel). */
 coarseRelaxation:number;
 /** Each aggregate level's inter-aggregate couplings (and the diagonal share
  * they carry) are scaled by this: piecewise-constant aggregation is twice too
  * stiff for smooth modes (0.5 would recover the rediscretised interior
  * operator). Dirichlet terms stay exact -- scaling them over-corrects small
  * Dirichlet-dominated blobs and stalls the cycle. 0.6 from replaying dumped
  * band systems: a wide shallow pool (hero garden) and a thick long-dam slab
  * converge ~0.2 per cycle at 0.6 but only ~0.5 at 0.5 (0.4 stalls); the
  * cm12 figure-12 surface band is ~0.1 at 0.5 and ~0.2 at 0.6. */
 coarseScale:number;

}
export const UNIFORM_PRESSURE_BAND_SCHEDULE:UniformPressureBandSchedule={cycles:4,fineSweeps:3,middleSweeps:2,coarseSweeps:8,coarseRelaxation:1.4,coarseScale:0.6};

/** Workgroups of the tile classification pass at most: a grid-stride launch
 * over the simulation's h tile worklist, as wide as the ownership's buffered
 * h-tile evidence. */
const LIST_GROUPS=4096;
/** Every band pass strides the band's own compact slot list from a buffered
 * direct launch: one slot per group, eight slots (2h aggregates) per group,
 * or 64 slots (4h aggregates) per group, up to these group counts. Groups
 * past the live count exit at once. Completed-frame evidence sizes the
 * launch with headroom; every kernel still strides the full current list. */
const SLOT_GROUPS=4096,MIDDLE_GROUPS=1024,COARSE_GROUPS=256;
/** An h half-sweep group holds four slots, 32 cells of the swept colour each. */
const CELL_BLOCK=4;
const CELL_GROUPS=2048;
/** The single-workgroup 4h aggregate solve: up to this many lanes (the
 * device's workgroup limit), and the aggregates it solves from registers
 * (fig-9's band holds 3582; a device with less workgroup memory holds
 * fewer). coarseBake stores the operator rows once per frame
 * in colour order (red from the front, black from the back). The register
 * path holds each lane's positions' diagonal, residual, slot and u16-packed
 * neighbour slots, and its first position's couplings (all of them at up to
 * one position per lane); the other couplings stream by position. A larger
 * band streams its rows position by position (contiguous loads, no
 * list-to-slot chain), and keeps as many corrections as the device's
 * workgroup memory holds in workgroup memory, the rest in storage: the same
 * red-black updates in the same one launch, whatever the band's size. A
 * colour's rows couple only to the other colour's, so neither path's order
 * changes a value. */
const COARSE_SOLVE_LANES=1024,COARSE_SOLVE_SLOTS=4096;
/** A band of more slots than this (the buffered estimate) runs the 4h
 * level as a launch per half sweep across workgroups instead (coarseSweep):
 * the same red-black updates. One workgroup streams a large band's rows
 * through at most 1024 threads, 72 microseconds a thousand rows a solve
 * (pool 128^3 all-fine, 17 thousand rows: 1.2 ms of each solve's launch);
 * the `2 coarseSweeps` launches are 5 to 12 microseconds each on the GPU and
 * 7 on the host, so they only win well past the register path's reach. */
const COARSE_WIDE_SLOTS=8192;
/** Threads a capacity cycle's stride launch starts at most. Cycles past
 * the most the band ran in the last BAND_CYCLE_WINDOW receipted frames are
 * capacity: closed (the initial measure converged) in all but a few frames,
 * where each launch costs its fixed few microseconds plus the threads it
 * starts, every one reading the convergence words and returning. A capacity
 * cycle that does open still completes the same frame: every kernel strides
 * the full live list from any width. */
const BAND_CAPACITY_THREADS=16384;
/** Receipted frames whose largest band cycle count keeps that many cycles wide. */
const BAND_CYCLE_WINDOW=4;
/** bandExit "returns": a full V-cycle that cuts the band's worst residual by
 * less than 1/BAND_RETURNS_RATIO, once that residual is at most
 * BAND_RETURNS_RESIDUAL (1/s, h-equivalent), ends the solve. Drifting pool
 * (114 band tiles, h 5 cm, 1/60 s): 11 -> 3.1 -> 0.19 -> 0.03 -> 0.007 ->
 * 0.003 1/s, cycle ratios 0.06, 0.13-0.21, 0.24-0.33, 0.38-0.56, never the
 * 1e-4 floor; the root accepts up to 5 (or a tenth of its start). 0.05 1/s
 * is a 2.5 mm/s worst-cell velocity error at h 5 cm. A bound relative to the
 * band's start let the pool impact stop at 0.2 1/s (its starts spike to
 * 4000 at single surface cells) and the ball sank 2 cm deeper. */
const BAND_RETURNS_RATIO=0.125,BAND_RETURNS_RESIDUAL=0.05;
/** The band's passes, described once: the encode path constructs nothing. */
/** ceil(count/per) of whole numbers in whole-number steps: a fractional
 * quotient is a heap number wherever the caller is not yet optimized. */
const groupsOf=(count:number,per:number):number=>{const padded=count+per-1;return (padded-padded%per)/per;};
const PREPARE_PASS:GPUComputePassDescriptor={label:"Uniform pressure band list and rows"};
const SOLVE_PASS:GPUComputePassDescriptor={label:"Uniform pressure band solve"};
const PROJECT_PASS:GPUComputePassDescriptor={label:"Uniform pressure band projection"};
/** The ten cell planes are field-major over CAP*64 rows: rhs, diagonal,
 * face kinds, six coefficients, and start pressure. Rows are colour-major,
 * so hot red-black sweeps still read 32 contiguous values per plane.
 * The cold face payloads follow: u* (init replaces Neumann faces with the
 * transferred 4h velocity), then CM11a V with solids. Each payload holds
 * three 5x4x4 MAC grids per tile, sharing interior faces between cells.
 * Only the positive face owner and the tile's low boundary write them;
 * Neumann updates occur only on tile boundaries, with one writer each. */
const ROW_WORDS_PER_TILE=10*64+240,SOLID_ROW_WORDS_PER_TILE=10*64+2*240;
/** 2h aggregates: correction, residual, diagonal, six couplings. 4h
 * aggregates: correction and residual by slot, then by colour-ordered
 * position the diagonal and six couplings (2..8), six neighbour slots (+1)
 * by slot (9..14), then by position the slot (15), the residual the large
 * solve gathers per call (16) and the six neighbour slots (+1, 17..22). */
const MIDDLE_FIELDS=9,COARSE_FIELDS=23;
/** Index header words: count, overflow, final residual, fatal, completed
 * cycles, closed (the frame's verdict: nonzero when the pressure schedule's
 * last gate rejected the frame), converged (the convergence word), valid
 * residual bits (history in bits 0..15, initial in bit 16),
 * then the residual after each cycle's pre-smoothing (and after the last
 * cycle) in HISTORY words, then coarseBake's red and black 4h row counts.
 * Then the initial residual, before ordinary smoothing. The tile list and
 * per-tile slot+1 map follow. A cleared residual is not a measurement. */
const HEADER=27,CLOSED_WORD=5,HISTORY_WORD=8,HISTORY=16,RED_WORD=24,BLACK_WORD=25,INITIAL_WORD=26,VALID_WORD=7;

export interface UniformPressureBandFields {
 /** Fuse a small band's 2h smoothing visits; false keeps separate colours. */
 fusedMiddle?:boolean;
 /** Simulation-indexed pressure phi (the simulation authority's output),
  * read by the list and the rows before the split. */
 phi:GPUBufferBinding;
 /** The coarse pressure phi (the split authority's, in pressure indexing),
  * read after the 4h projection. Absent: the split rewrote `phi` itself. */
 coarsePhi?:GPUBufferBinding;
 /** The surface-only authorities' parameters (16 B); word 2 is the volume
  * control rate (1/s). A liquid row's correction is rate x capacity, the
  * simulation authority's own product, formed by the rows. */
 authorityParams:GPUBuffer;
 vertexPhi:GPUTexture;
 /** The forced field u* in simulation ownership (before the split). */
 forced:{velocity:GPUTexture;negative:GPUBuffer};
 /** The 4h-projected field transferred to simulation ownership, and the
  * texture its copy lands in for the band projection's reads. */
 velocity:GPUTexture;negative:GPUBuffer;copy:GPUTexture;
 /** Accepted 4h pressure, indexed by the all-4h pressure ownership. */
 coarsePressure:GPUBufferBinding;
 /** h.xyz, dt; density, openTop (not read: the kinds hold it), dt/density,
  * min h; the residual target
  * (1/s, the h-equivalent pressure tolerance), then 1/h^2 per axis (f32):
  * a face whose coefficient is exactly that is flagged, and its sweeps skip
  * the coefficient load. Then four u32 the band writes itself: its slot
  * capacity, 64 and 8 times it, and the presentation word offset (64 bytes in all). */
 params:GPUBuffer;
 /** The stage grids' band section: the live band pressures land at this
  * word of the buffer (uniform-stage-grids). */
 presentation:{buffer:GPUBuffer;word:number};
 /** Band tiles one frame can hold (1..tiles). Absent: capacityOf at the
  * simulation ownership's h-tile capacity. */
 capacity?:number;
 /** The caller's storage for the rows, at least rowBytes(capacity). The rows
  * live from encodePrepare to encodeSolve and every prepare assembles them
  * anew, so outside that span the buffer is the caller's to overwrite: the
  * frame lends its stage scratch. Absent: the band allocates its own. */
 rows?:GPUBuffer;
 /** UniformMixedRegionDynamics.record: the initial measure publishes each
  * region's largest predictor residual (1/s) into its pressureResidual
  * word, which the frame head's classify cleared. */
 dynamics?:GPUBuffer;
 /** The COLD freeze (UniformColdFreeze): a frozen region's tiles are no
  * band tiles, and a band face to a frozen cell is Neumann at its frozen F
  * plus the region's term T (the rows' flux; the projection writes F back). */
 freeze?:UniformColdFreeze;
 /** Host cold-frame evidence: with the freeze, a quiet frame (every live
  * region COLD, so frozen) lists no band tile, and the band's launches are
  * not encoded; the cleared header and slot map are its whole result. */
 coldFrame?:UniformColdFrame;
 /** NB-FLIP warmResolution "4h": the region heat tiers (UniformRegionHeat.tiers).
  * A WARM region's h tiles are then no band tiles: their velocity stays the
  * 4h interpolation the transfer to simulation wrote, and a band face to
  * one is Neumann at it (the projected 4h flux). */
 warmTiers?:GPUBuffer;
 /** NB-FLIP bandExit "returns": the band stops at diminishing returns
  * (bStop, BAND_RETURNS_RATIO); absent, only at the residual tolerance. */
 returns?:boolean;
}

/** The h pressure band of the two-stage solve: the global solve runs on
 * all-4h ownership, then every h simulation tile holding a liquid pressure row
 * re-solves pressure at h. A tile the simulation holds at h is h in the bulk
 * too: its velocity is only 4h-divergence-free until this solve, and without
 * it V drifts per cell. Its inner boundary is Neumann with the projected 4h
 * face velocity (exactly conservative: a 4h face's flux is the sum of its h
 * faces'), air is ghost-fluid Dirichlet, walls keep the native one-cell halo.
 * The start is the liquid-weighted trilinear 4h pressure; V-cycles of a local
 * multigrid (h, 2h and 4h aggregates of the band's own tiles, homogeneous
 * Neumann wherever the h rows are) converge it however large the band is.
 * No readback on the advance path: the GPU builds a compact tile list and
 * every solver pass is a buffered direct launch striding it.
 * With static solids every face carries the native CM11a dual-cell V (the
 * coefficient V/(h^2 theta), the divergence V u; V=0 faces are closed), rows
 * inside a solid are p_min=0 rows, and positive faces take the native
 * embedded contact release. A Neumann face carries V times the transferred
 * 4h face, the flux the root solved through it (the record V of a 4h face
 * is the mean of its sixteen h V), so an h region beside a cut 4h owner
 * closes on the root's fluxes. A cut tile the simulation holds at 4h is not
 * a band tile: its liquid row is the root's cut owner (record Omega and V),
 * which needs no h band. Header word 3 is reserved (zero). */
export class UniformPressureBand {
 get allocatedBytes():number{return this.index.size+(this.ownRows?.size??0)+this.coarse.size+this.solve.size;}
 /** Header (see HEADER), then the tile list and a per-tile slot+1 map. It is
  * tile-scaled metadata (the list holds every tile, whatever the band's
  * capacity), so the buffer and its word offsets never change. */
 readonly index:GPUBuffer;
 private rows!:GPUBuffer;
 /** The rows the band allocated itself (fields.rows absent). */
 private ownRows?:GPUBuffer;
 private coarse!:GPUBuffer;
 private solve!:GPUBuffer;
 private workSlots=1;
 /** Each slot-list stride's launch width at workSlots (see dispatch), kept
  * with it: every launch of a frame reads one. */
 private cellGroups=1;private slotGroups=1;private middleGroups=1;private coarseGroups=1;
 /** The last BAND_CYCLE_WINDOW receipted cycle counts (observeCycles), and
  * the cycles launched wide: every encoded one until a window is full. */
 private readonly recentCycles=new Uint32Array(BAND_CYCLE_WINDOW);
 private recentCount=0;
 private wideCycles=Number.POSITIVE_INFINITY;
 private middleFusedSlots=0;
 private observedMiddleSlots=Number.POSITIVE_INFINITY;
 private budget(slots:number):void{
  this.workSlots=slots;
  this.cellGroups=Math.min(groupsOf(slots,CELL_BLOCK),CELL_GROUPS);this.slotGroups=Math.min(slots,SLOT_GROUPS);
  this.middleGroups=Math.min(groupsOf(slots,8),MIDDLE_GROUPS);this.coarseGroups=Math.min(groupsOf(slots,64),COARSE_GROUPS);
 }
 /** Fixed-lag evidence, applied once per encoded frame by the frame owner.
  * A smaller budget never limits admission, storage or solver iterations. */
 observeWork(tiles:number,reserve=0):void{this.observedMiddleSlots=tiles;this.budget(uniformBufferedWork(this.workSlots,tiles,this.capacity,reserve));}
 /** Fixed-lag evidence: a receipted frame's band cycle count. Launch width
  * only; forgetCycles (a relayout, a kick) launches every cycle wide again. */
 observeCycles(cycles:number):void{
  this.recentCycles[this.recentCount%BAND_CYCLE_WINDOW]=cycles;this.recentCount++;
  if(this.recentCount<BAND_CYCLE_WINDOW){this.wideCycles=Number.POSITIVE_INFINITY;return;}
  let most=0;for(let i=0;i<BAND_CYCLE_WINDOW;i++)most=Math.max(most,this.recentCycles[i]!);
  this.wideCycles=most;
 }
 forgetCycles():void{this.recentCount=0;this.wideCycles=Number.POSITIVE_INFINITY;}
 /** Band tiles one frame can hold (fields.capacity). More is a fatal receipt
  * (the sticky bandCapacity failure), never a fallback. The shaders read it
  * from the params (BandParams.capacity), so resize compiles nothing. */
 get capacity():number{return this.slots;}
 private slots=0;
 /** The liquid bound: the band capacity of a lattice of `tiles` 4^3 tiles
  * whose owner neither knows nor bounds its layouts' h tiles (a GPU
  * relayout; a host layout passes its own count as fields.capacity, since
  * any of its h tiles may hold a liquid row). Half of them, never
  * fewer than 4096 (a whole 64^3 lattice). The band is the h tiles holding a
  * liquid row, so it is bounded by the liquid's tiles, not by the lattice:
  * measured peaks are 22% (fig-9, 3582 of 16384), 27% (128^3 dam, dynamic,
  * 9k of 32768), 38% (the same, all-h, 12.6k) and 5% (fig7-256, 12.9k h
  * tiles of 262144). Only an all-h layout over a container more than half
  * full of liquid can reach it, and that run fails loudly.
  * fineTiles: the simulation's h-tile capacity. A band tile is an h tile,
  * so the band never needs more slots than that (never fewer than one). */
 static capacityOf(tiles:number,fineTiles=tiles):number{return Math.max(1,Math.min(fineTiles,tiles,Math.max(4096,Math.ceil(tiles/2))));}
 /** The COLD freeze the list, rows and init read (fields.freeze, switch on). */
 private get cold():UniformColdFreeze|undefined{return UNIFORM_COLD_FREEZE?this.fields.freeze:undefined;}
 private readonly layouts=new Map<string,GPUBindGroupLayout>();
 private readonly groups=new Map<string,UniformDetailGroup>();
 private readonly pipelines=new Map<string,GPUComputePipeline>();
 /** The h pressure iterate, the correction to each row's start (the rows' last field): 64 cells per band slot, then the wall halo: a slot a cell of each face of each block (bHalo). */
 get iterate():GPUBuffer{return this.solve;}
 /** Byte offset in index of the per-tile slot+1 map. */
 get slotMapOffset():number{return 4*(HEADER+this.listSlots);}
 /** Tile list words of index: every tile. */
 private readonly listSlots:number;
 /** Each cycle's restriction launch (it writes its own history word). */
 private readonly restricts:readonly string[];
 constructor(private readonly device:GPUDevice,private readonly simulation:UniformMixedOwnership,private readonly pressure:UniformMixedOwnership,private fields:UniformPressureBandFields,
  readonly schedule:UniformPressureBandSchedule=UNIFORM_PRESSURE_BAND_SCHEDULE,
  /** Static solids, with the all-4h record (its cut-tile flags). */
  private readonly solid?:UniformMixedSolid,readonly spatialOrder=true){
  const layout=simulation.layout,tiles=simulation.capacity.address.tileKeys;
  if(pressure.layout.tiles.some(word=>(word&0xc0000000)!==0))throw new Error("The pressure band's global stage must be all-4h");
  if(solid&&!solid.coarse)throw new Error("The solid pressure band needs the all-4h solid record");
  if(schedule.cycles<1||schedule.cycles>=HISTORY)throw new Error(`Pressure band cycles must be 1..${HISTORY-1}`);
  if(schedule.middleSweeps<1)throw new Error("The pressure band's 2h level needs a sweep: its first red half sweep carries the restriction and the prolongation");
  this.restricts=Array.from({length:schedule.cycles},(_,k)=>`restrict@${k}`);
  this.listSlots=tiles;
  this.index=device.createBuffer({label:"Uniform pressure band tiles",size:(HEADER+this.listSlots+tiles+Math.ceil(tiles/256))*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC});
  const texture=(binding:number)=>({binding,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float" as const,viewDimension:"3d" as const}});
  const buffer=(binding:number,type:GPUBufferBindingType="storage")=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type}});
  const params=buffer(0,"uniform");
  // At most six storage buffers here: topology and solids hold four.
  const base=[params,buffer(2),buffer(9,"read-only-storage"),buffer(10),buffer(18)];
  // The dynamics record (24) is the measure's: the projection has no storage to spare.
  const solver=[...base,...(fields.dynamics?[buffer(24)]:[])];
  const cold=UNIFORM_COLD_FREEZE&&fields.freeze?[{binding:14,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float" as const,viewDimension:"2d" as const}}]:[];
  const entries:Record<string,GPUBindGroupLayoutEntry[]>={
   list:[params,buffer(2),buffer(3,"read-only-storage"),texture(6),...cold,...(fields.warmTiers?[buffer(25,"read-only-storage")]:[])],
   copy:[params,buffer(2),texture(22),{binding:23,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"write-only",format:"rgba32float",viewDimension:"3d"}}],
   prep:[params,buffer(2),buffer(3,"read-only-storage"),buffer(4,"uniform"),texture(6),texture(7),buffer(8,"read-only-storage"),buffer(9),buffer(10),...cold],
   init:[params,buffer(2),buffer(9),buffer(10),texture(11),buffer(12,"read-only-storage"),buffer(13,"read-only-storage"),...cold],
   solver,
   project:[...base,texture(15),{binding:16,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"write-only",format:"rgba32float",viewDimension:"3d"}},buffer(17),buffer(21)],
  };
  for(const [name,list] of Object.entries(entries))this.layouts.set(name,uniformDetailBindLayout(device,{label:`Uniform pressure band ${name}`,entries:list}));
  this.allocate();
  this.budget(Math.max(1,Math.min(this.capacity,layout.fineTiles.length)));
 }
 /** Bytes of the rows at `capacity` band tiles, with or without solids. */
 static rowBytes(capacity:number,solid:boolean):number{return 4*(solid?SOLID_ROW_WORDS_PER_TILE:ROW_WORDS_PER_TILE)*capacity;}
 /** Bytes of the capacity-sized buffers at `capacity` band tiles. */
 bytesAt(capacity:number):{rows:number;aggregates:number;iterate:number}{
  const halo=uniformFaceCellSlots(this.simulation.capacity.address.blocks);
  return {rows:UniformPressureBand.rowBytes(capacity,!!this.solid),aggregates:4*(8*MIDDLE_FIELDS+COARSE_FIELDS)*capacity,iterate:4*(64*capacity+halo)};
 }
 /** The capacity-sized buffers (rows, aggregates, iterate) and every group,
  * at fields.capacity. No band buffer carries a frame's state into the next
  * (each solve lists, assembles and starts anew), so a new capacity is a
  * reallocation and a rebind, no copy. */
 private allocate():void{
  const device=this.device,c=this.simulation.capacity;
  const capacity=this.fields.capacity??UniformPressureBand.capacityOf(c.tiles,c.fineTiles);
  if(!Number.isInteger(capacity)||capacity<1||capacity>c.tiles)throw new Error(`The pressure band cannot hold ${capacity} of ${c.tiles} tiles`);
  const rows=capacity*64,bytes=this.bytesAt(capacity);
  const limit=device.limits.maxStorageBufferBindingSize;
  if(bytes.rows>limit)throw new Error(`Pressure band rows (${bytes.rows} bytes) exceed the storage binding limit ${limit}`);
  // The frames encoded on the old buffers are submitted: destroy applies after them.
  for(const b of [this.ownRows,this.coarse,this.solve])b?.destroy();
  this.slots=capacity;
  const lent=this.fields.rows;
  if(lent&&lent.size<bytes.rows)throw new Error(`Pressure band rows (${bytes.rows} bytes) exceed the ${lent.size} bytes lent to them`);
  this.ownRows=lent?undefined:device.createBuffer({label:"Uniform pressure band rows",size:bytes.rows,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
  this.rows=lent??this.ownRows!;
  this.coarse=device.createBuffer({label:"Uniform pressure band aggregates",size:bytes.aggregates,usage:GPUBufferUsage.STORAGE});
  this.solve=device.createBuffer({label:"Uniform pressure band iterate",size:bytes.iterate,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC});
  // BandParams.capacity: slots, rows per field, 2h aggregates per field, presentation offset.
  device.queue.writeBuffer(this.fields.params,48,new Uint32Array([capacity,rows,8*capacity,this.fields.presentation.word]));
  const f=this.fields,cells=c.owners,faces=c.address.lowFaceKeys;
  const scalars=(view:GPUBufferBinding,count:number):GPUBufferBinding=>{
   const offset=view.offset??0;if((view.size??view.buffer.size-offset)<4*count)throw new Error("Pressure band field view is too small");return {buffer:view.buffer,offset,size:4*count};
  };
  const pressure=this.pressure,coarse=pressure.capacity.owners;
  const rowsView={buffer:this.rows,size:bytes.rows};
  const P={0:{buffer:f.params,size:64}},solverResources:Record<number,GPUBindingResource>={...P,2:{buffer:this.index},9:rowsView,10:{buffer:this.solve},18:{buffer:this.coarse}};
  if(f.dynamics)solverResources[24]={buffer:f.dynamics};
  const resources:Record<string,Record<number,GPUBindingResource>>={
   list:{...P,2:{buffer:this.index},3:scalars(f.phi,cells),6:f.vertexPhi},
   copy:{...P,2:{buffer:this.index},22:f.velocity,23:f.copy},
   prep:{...P,2:{buffer:this.index},3:scalars(f.phi,cells),4:{buffer:f.authorityParams,size:16},6:f.vertexPhi,7:f.forced.velocity,8:{buffer:f.forced.negative,size:4*faces},9:rowsView,10:{buffer:this.solve}},
   init:{...P,2:{buffer:this.index},9:rowsView,10:{buffer:this.solve},11:f.velocity,12:scalars(f.coarsePressure,coarse),13:scalars(f.coarsePhi??f.phi,coarse)},
   solver:solverResources,
   project:{...P,2:{buffer:this.index},9:rowsView,10:{buffer:this.solve},18:{buffer:this.coarse},15:f.copy,16:f.velocity,17:{buffer:f.negative,size:4*faces},21:{buffer:f.presentation.buffer}},
  };
  const freeze=this.cold;if(freeze)for(const name of ["list","prep","init"])resources[name]![14]=freeze.regions;
  if(f.warmTiers)resources.list![25]={buffer:f.warmTiers};
  if(f.presentation.buffer.size<4*(f.presentation.word+64*this.capacity))throw new Error("The stage grids cannot hold the pressure band");
  for(const [name,map] of Object.entries(resources))this.groups.set(name,uniformDetailGroup(device,{label:`Uniform pressure band ${name}`,layout:this.layouts.get(name)!,
   entries:Object.entries(map).map(([binding,resource])=>({binding:+binding,resource}))}));
  this.bindLaunches();this.boundPass=undefined;this.boundGroup=undefined;
 }
 /** Between frames, after the simulation ownership's h-tile capacity (and
  * with it the views sized by it) or the band's own changed: reallocate
  * and rebind. */
 resize(fields:Pick<UniformPressureBandFields,"phi"|"presentation"|"capacity"|"rows">):void{
  this.fields={...this.fields,...fields};this.allocate();
  this.budget(Math.max(1,Math.min(this.workSlots,this.capacity)));
 }
 async initialize():Promise<void>{
  const layout=this.simulation.layout,S=!!this.solid,schedule=this.schedule;
  // The coarse solve's workgroup-memory corrections: all the device holds
  // beside its four scalars (Tint pads each to 16 bytes), never fewer than
  // the register path's, whose positions per lane that memory bounds.
  const limits=this.device.limits,lanes=Math.min(COARSE_SOLVE_LANES,limits.maxComputeInvocationsPerWorkgroup,limits.maxComputeWorkgroupSizeX);
  this.middleFusedSlots=Math.floor(lanes/8);
  const memory=4*Math.floor((limits.maxComputeWorkgroupStorageSize-128)/16),held=Math.max(1,Math.floor(Math.min(COARSE_SOLVE_SLOTS,memory)/lanes));
  // Not the band's capacity: that is a run-time word, and the kernel
  // already keeps corrections past SHARED in storage.
  const shared=memory;
  if(shared<held*lanes)throw new Error(`Pressure band coarse solve needs ${4*held*lanes} bytes of workgroup memory`);
  const header=uniformMixedTopologyWGSL(layout,0)+uniformMixedFaceAddressWGSL+uniformMixedSolidWGSL(S?2:undefined,this.solid?.coarse?.count)+/* wgsl */`
struct BandParams {hDt:vec4f,policy:vec4f,solve:vec4f,capacity:vec4u}
// The slot capacity is the host's allocation, read from the params (never
// baked): CAP slots, N = 64 CAP rows per field, M = 8 CAP 2h aggregates.
fn bCap()->u32{return params.capacity.x;}
fn bN()->u32{return params.capacity.y;}
const LIST:u32=${HEADER}u;const SLOTS:u32=${HEADER}u+UM_TILES;const HISTORY:u32=${HISTORY_WORD}u;const CYCLE:u32=4u;const DONE:u32=6u;
const INITIAL:u32=${INITIAL_WORD}u;const VALID:u32=${VALID_WORD}u;const INITIAL_BIT:u32=1u<<${HISTORY}u;
const K_GHOST:u32=0u;const K_BAND:u32=1u;const K_WALL:u32=2u;const K_OPEN:u32=3u;const K_NEUMANN:u32=4u;const K_CLOSED:u32=5u;
fn bLocal(lane:u32)->vec3u{return vec3u(lane%4u,(lane/4u)%4u,lane/16u);}
// A slot's row of local cell l: colour-major, then the half sweep's lane order.
fn bRow(l:vec3u)->u32{return ((l.x+l.y+l.z)&1u)*32u+(l.x>>1u)+2u*l.y+8u*l.z;}
// One face value per tile, with its low and high boundary planes. Keeping
// tile boundaries separate avoids cross-workgroup ownership and lookups.
fn bPackedFace(slot:u32,l:vec3u,f:u32)->u32{
 let a=f/2u;return 240u*slot+80u*a+l[a]+(f&1u)+5u*(l[(a+1u)%3u]+4u*l[(a+2u)%3u]);
}
fn bFaceSlot(c:u32,f:u32)->u32{
 let r=c%64u;let y=(r/2u)%4u;let z=(r/8u)%4u;
 let x=2u*(r&1u)+((r/32u+y+z)&1u);
 return bPackedFace(c/64u,vec3u(x,y,z),f);
}
fn bForcedWord(c:u32,f:u32)->u32{return 10u*bN()+bFaceSlot(c,f);}
fn bVolumeWord(c:u32,f:u32)->u32{return 10u*bN()+240u*bCap()+bFaceSlot(c,f);}

// Face f's coefficient is 1/h^2 (kinds bit 18+f).
fn bUnit(kinds:u32,f:u32)->bool{return (kinds&(1u<<(18u+f)))!=0u;}
fn bMiddleLocal(a:u32)->vec3u{return vec3u(a&1u,(a>>1u)&1u,a>>2u);}
fn bFaceKind(kinds:u32,f:u32)->u32{return (kinds>>(3u*f))&7u;}
fn bLiquid(kinds:u32)->bool{return (kinds&0x80000000u)!=0u;}
// A row inside a solid: the CM11a p_min=0 row.
fn bInside(kinds:u32)->bool{return (kinds&0x40000000u)!=0u;}
// A row of a cell with no capacity (a row by its dual-cell faces): its phi
// continues its open neighbours and is not its own depth.
fn bShut(kinds:u32)->bool{return (kinds&0x20000000u)!=0u;}
fn bSign(f:u32)->i32{return select(-1,1,(f&1u)==1u);}
// Positive float bits have integer order. Treat every nonfinite residual as
// infinity: floating max may otherwise discard a NaN and certify zero.
fn bResidualMagnitude(r:f32)->u32{return min(bitcast<u32>(r)&0x7fffffffu,0x7f800000u);}
`;
  const indexed=(atomic:boolean)=>/* wgsl */`
@group(1) @binding(0) var<uniform> params:BandParams;
@group(1) @binding(2) var<storage,read_write> index:array<${atomic?"atomic<u32>":"u32"}>;
fn bIndex(i:u32)->u32{return ${atomic?"atomicLoad(&index[i])":"index[i]"};}
// A closed band (the verdict rejected the frame) strides no slots: its solve,
// projection and presentation do nothing. Row assembly precedes the verdict.
fn bCount()->u32{return select(min(bIndex(0u),bCap()),0u,bIndex(${CLOSED_WORD}u)!=0u);}
// Only read completed measurements: a dispatch must never test the word it
// is reducing. The following dispatch sees the global maximum and valid bit.
// Zero is a valid residual; zero tolerance disables stopping for diagnostics.
fn bMet(word:u32,mask:u32)->bool{return (bIndex(VALID)&mask)!=0u&&params.solve.x>0.0&&bitcast<f32>(bIndex(word))<=params.solve.x;}
fn bInitialMet()->bool{return bMet(INITIAL,INITIAL_BIT);}
// History word j ends the band: it met the tolerance${this.fields.returns?`, or (bandExit
// "returns") the full cycle before it cut the residual by less than
// 1/BAND_RETURNS_RATIO while the residual is at most BAND_RETURNS_RESIDUAL:
// the slowed cycles chase what the band cannot reach`:""}.
fn bStop(j:u32)->bool{
 if(bMet(HISTORY+j,1u<<j)){return true;}
 ${this.fields.returns?`let both=(1u<<j)|(1u<<(j-1u));
 if(j==0u||(bIndex(VALID)&both)!=both||params.solve.x<=0.0){return false;}
 let r=bitcast<f32>(bIndex(HISTORY+j));
 return r>${BAND_RETURNS_RATIO}*bitcast<f32>(bIndex(HISTORY+j-1u))&&r<=${BAND_RETURNS_RESIDUAL};`:"return false;"}
}
fn bConvergedBefore(k:u32)->bool{if(bInitialMet()){return true;}for(var j=0u;j<min(k,${HISTORY-1}u);j++){if(bStop(j)){return true;}}return false;}
fn bConverged()->bool{return bConvergedBefore(${schedule.cycles}u);}
// Converged launches stride zero slots: the same launches, no work.
fn bLive()->u32{return select(bCount(),0u,bConverged());}
fn bTile(s:u32)->vec3u{return umTileCoord(bIndex(LIST+s));}
// Slot + 1 of the band tile at tile coordinate c, 0 outside the band.
fn bSlotAt(c:vec3i)->u32{if(!umTilePresent(c)){return 0u;}return bIndex(SLOTS+umTileAt(vec3u(c)));}
// Band cell index + 1 at an h position, 0 outside the band.
fn bCellAt(q:vec3i)->u32{
 if(!umCellPresent(q)){return 0u;}
 let u=vec3u(q);let slot=bIndex(SLOTS+umTileAt(u/4u));if(slot==0u){return 0u;}
 return (slot-1u)*64u+bRow(u%4u)+1u;
}
`;
  // umPressureLiquid of a centre whose spacing is width cells.
  const wet=/* wgsl */`
fn bWet(phi:f32,width:f32)->bool{return ${uniformMixedPressureLiquidWGSL("phi","width*params.policy.w")};}
`;
  const theta=/* wgsl */`
// umPressureSurfaceTheta; umPressureTheta of a classified pair.
fn bTheta(liquidPhi:f32,airPhi:f32)->f32{return ${uniformMixedSurfaceThetaWGSL("liquidPhi","airPhi")};}
fn bPairTheta(a:f32,wetA:bool,b:f32,wetB:bool)->f32{if(wetA==wetB){return 1.0;}if(wetA){return bTheta(a,b);}return bTheta(b,a);}
// Theta of a liquid/air pair of h centres whose phi is the vertex field's:
// the zero crossing of that field between them. It is linear from each centre
// to the face's (the mean of the face's four vertices), not from centre to
// centre: fine redistance leaves the interface vertices their advected
// magnitudes, so the slope across the surface is not the slope beyond it. On
// the two centres alone a resting tank's h surface 6e-3 h high read 1e-3 h
// low, and the band filled the rise (uniform-trough-settled-tank).
fn bFaceTheta(face:f32,a:f32,wetA:bool,b:f32,wetB:bool)->f32{
 if(wetA==wetB){return 1.0;}
 let liquid=select(b,a,wetA);let air=select(a,b,wetA);
 if(face>0.0){return clamp(0.5*liquid/(liquid-face),${UNIFORM_MIXED_THETA_MIN},${(1/UNIFORM_MIXED_THETA_MIN).toFixed(1)});}
 return clamp(0.5+0.5*face/min(face-air,-1e-30),${UNIFORM_MIXED_THETA_MIN},${(1/UNIFORM_MIXED_THETA_MIN).toFixed(1)});
}
`;
  // The h solve uses the simulation's actual interface. The 4h solve supplies
  // boundary flux and an initial pressure, not a veto on h air: relabelling
  // a sub-cell pocket as liquid removes its free surface from pressure while
  // advection and rendering keep it, allowing a false cavity to persist.
  const owner=wet+/* wgsl */`
@group(1) @binding(3) var<storage,read> phi:array<f32>;
@group(1) @binding(6) var vertexPhi:texture_3d<f32>;
fn bCentrePhi(q:vec3i)->f32{
 let c8=udLoad8_vertexPhi(q,1u);var s=0.0;for(var k=0u;k<8u;k++){s+=c8[k];}return 0.125*s;
}
// The vertex field at the centre of cell p's face: the mean of its four vertices.
fn bFacePhi(p:vec3i,axis:u32,sign:i32)->f32{
 var o=p;if(sign>0){o[axis]+=1;}let u=(axis+1u)%3u;let v=(axis+2u)%3u;
 var s=0.0;for(var k=0u;k<4u;k++){var c=o;c[u]+=i32(k&1u);c[v]+=i32(k>>1u);s+=textureLoad(vertexPhi,c,0).x;}return 0.25*s;
}
// The simulation's h tile worklist.
fn bHTile(j:u32)->u32{return umTopology[UM_TILES+j];}
`;
  const surface=owner+/* wgsl */`
// Preserve the simulation authority, including its solid continuation.
fn bPhiH(q:vec3i)->f32{return phi[umOwnerAt(q).index];}
// Face classification of row assembly. An h neighbour's band phi decides.
// Inside a coarse owner the neighbour is liquid no band row resolves where
// that owner is liquid or the h centre there is: the face is Neumann, its
// flux the transferred 4h face. The 4h solve owns that flux where either 4h
// owner is a liquid row: projected (liquid beside liquid, or ghost fluid
// beside air). Between two owners that are not it solved none, and the face
// keeps its forced velocity (init's bNeumann). An air h
// centre inside a coarse air owner is the free surface: Dirichlet at that
// centre's phi (a Neumann face there closed the band's lid under a calm 4h
// surface tile's rim owners). Dirichlet at the coarse centre's phi for a wet
// h centre put p=0 beside a liquid cell at its own height: the surface row
// of an h patch in a resting 4h pond lost its support through the seam and
// left at 0.3 m/s on frame 1 (uniform-coarse-solid-rest-dawn).
struct BNeighbour {phi:f32,wet:bool,neumann:bool}
fn bNeighbour(q:vec3i)->BNeighbour{
 let n=umOwnerAt(q);let other=phi[n.index];
 if(n.width==1u){let b=bPhiH(q);let wet=bWet(b,1.0);return BNeighbour(b,wet,wet);}
 if(bWet(other,f32(n.width))){return BNeighbour(other,true,true);}
 let own=bCentrePhi(q);let wet=bWet(own,1.0);return BNeighbour(own,wet,wet);
}
`;
  // Only init writes rows after assembly; every solver launch reads them.
  const rowsOf=(write:boolean)=>/* wgsl */`
@group(1) @binding(9) var<storage,${write?"read_write":"read"}> rows:array<f32>;
fn bRhs(c:u32)->f32{return rows[c];}
fn bDiagonal(c:u32)->f32{return rows[bN()+c];}
fn bKinds(c:u32)->u32{return bitcast<u32>(rows[2u*bN()+c]);}
fn bCoefficient(c:u32,f:u32)->f32{return rows[(3u+f)*bN()+c];}
fn bForced(c:u32,f:u32)->f32{return rows[bForcedWord(c,f)];}
${S?"fn bVolume(c:u32,f:u32)->f32{return rows[bVolumeWord(c,f)];}":""}
// The start pressure of a row: the iterate holds the correction to it.
const B_BASE:u32=9u;
fn bBase(c:u32)->f32{return rows[B_BASE*bN()+c];}
`,rows=rowsOf(false);
  // The iterate and its wall halo, over the block faces. walls: the module
  // also holds a stencil at a wall, and takes the faces with the block
  // neighbour rows.
  const solveOf=(walls:boolean)=>(walls?uniformMixedTopologyNeighboursWGSL():uniformBlockFaceWGSL)+/* wgsl */`
@group(1) @binding(10) var<storage,read_write> solve:array<f32>;
// The native one-cell wall halo at h, after every band cell: a slot a cell
// of each of the six faces of each block (umFaceCell). A wall face is a face
// of its row's own block with a SOLID page across, so every wall face has a
// slot and no two share one, wherever the block's page lies.
fn bHalo(p:vec3i,axis:u32,sign:i32)->u32{let q=vec3u(p);return bN()+umFaceCell(umCellBlock(q),2u*axis+select(0u,1u,sign>0),q);}
// Off-diagonal sum of an h row: band neighbours and the wall halo.
fn bOff(cell:u32,p:vec3i,kinds:u32)->f32{
 var off=0.0;
 for(var f=0u;f<6u;f++){
  let axis=f/2u;let sign=bSign(f);let kind=bFaceKind(kinds,f);
  if(kind==K_BAND){var q=p;q[axis]+=sign;off+=bCoefficient(cell,f)*solve[bCellAt(q)-1u];}
  else if(kind==K_WALL){off+=bCoefficient(cell,f)*solve[bHalo(p,axis,sign)];}
 }
 return off;
}
// The wall halo follows its row: the native Neumann ghost, clamped at 0 of
// the total pressure. The halo holds its value less the row's start.
fn bFollowHalo(cell:u32,p:vec3i,kinds:u32,value:f32){
 // K_WALL is 010 in each three-bit face field. Most rows have no wall:
 // test all six fields together before entering the halo update loop.
 if(((kinds>>1u)&~kinds&~(kinds>>2u)&0x9249u)==0u){return;}
 for(var f=0u;f<6u;f++){
  if(bFaceKind(kinds,f)!=K_WALL){continue;}
  let axis=f/2u;let sign=bSign(f);
  solve[bHalo(p,axis,sign)]=max(value+f32(sign)*params.policy.x*bForced(cell,f)*params.hDt[axis]/params.hDt.w,-bBase(cell));
 }
}
// A liquid row's residual; a p_min=0 row at its bound only counts a positive one.
fn bResidual(cell:u32,p:vec3i,kinds:u32)->f32{
 let r=bRhs(cell)+bOff(cell,p,kinds)-bDiagonal(cell)*solve[cell];
 return select(r,max(r,0.0),bInside(kinds)&&solve[cell]+bBase(cell)<=0.0&&bResidualMagnitude(r)<0x7f800000u);
}
// Band cell + 1 across face f of local cell l of slot s, 0 outside the band.
// A face out of the tile reads the slot's baked neighbour (coarseBake): one
// load shared by the lanes that need it, no tile-map lookup, no barrier.
fn bCellNear(s:u32,l:vec3u,f:u32)->u32{
 let axis=f/2u;var m=vec3i(l);m[axis]+=bSign(f);var slot=s+1u;
 if(m[axis]<0||m[axis]>3){slot=bNearSlot(s,f);m[axis]=(m[axis]+4)%4;}
 if(slot==0u){return 0u;}
 return (slot-1u)*64u+bRow(vec3u(m))+1u;
}
// Neighbours change colour (bit 5 of the local row). Their remaining row
// offsets are fixed, except x's packed parity. Only a tile-boundary face
// needs a neighbour slot load. Emit six faces in the original sum order.
fn bOffNear(cell:u32,s:u32,l:vec3u,p:vec3i,kinds:u32)->f32{
 var off=0.0;let own=cell&~63u;let across=(cell&63u)^32u;
 ${Array.from({length:6},(_,f)=>{
  const axis=f>>1,positive=(f&1)!==0,component=["x","y","z"][axis];
  const delta=axis===0?(positive?"i32(l.x&1u)":"-i32(1u-(l.x&1u))"):String((positive?1:-1)*(axis===1?2:8));
  const wrap=(positive?-1:1)*(axis===0?1:axis===1?6:24);
  return `{let kind=bFaceKind(kinds,${f}u);
  if(kind==K_BAND){
   var neighbor=own+u32(i32(across)+${delta});
   if(l.${component}==${positive?3:0}u){let slot=bNearSlot(s,${f}u);neighbor=0xffffffffu;if(slot!=0u){neighbor=(slot-1u)*64u+u32(i32(across)+${wrap});}}
   var w=params.solve[${axis+1}u];if(!bUnit(kinds,${f}u)){w=bCoefficient(cell,${f}u);}off+=w*solve[neighbor];
  }else if(kind==K_WALL){off+=bCoefficient(cell,${f}u)*solve[bHalo(p,${axis}u,${positive?1:-1})];}}`;
 }).join("\n")}
 return off;
}
fn bResidualNear(cell:u32,s:u32,l:vec3u,p:vec3i,kinds:u32)->f32{
 let r=bRhs(cell)+bOffNear(cell,s,l,p,kinds)-bDiagonal(cell)*solve[cell];
 return select(r,max(r,0.0),bInside(kinds)&&solve[cell]+bBase(cell)<=0.0&&bResidualMagnitude(r)<0x7f800000u);
}
`,solve=solveOf(false);
  const aggregates=/* wgsl */`
@group(1) @binding(18) var<storage,read_write> coarse:array<f32>;
// 2h aggregate a (0..8) of slot s is cell 8s+a; field k at k*M. The per-tile
// 4h aggregate of slot s: field k at 9M+k*CAP+s (9..14 neighbour slot+1, 15 colour).
fn bM(k:u32,c:u32)->u32{return k*params.capacity.z+c;}
fn bC(k:u32,s:u32)->u32{return 9u*params.capacity.z+k*bCap()+s;}
// The 2h aggregate + 1 across face f of aggregate c, 0 outside the band.
fn bMiddleNeighbour(c:u32,f:u32)->u32{
 let s=c/8u;let axis=f/2u;var g=vec3i(bTile(s)*2u+bMiddleLocal(c%8u));g[axis]+=bSign(f);
 if(!umTilePresent(g>>vec3u(1u))){return 0u;}
 let slot=bSlotAt(g/2);if(slot==0u){return 0u;}
 let l=vec3u(g%2);return (slot-1u)*8u+l.x+2u*l.y+4u*l.z+1u;
}
fn bCoarseNeighbour(s:u32,f:u32)->u32{let axis=f/2u;var c=vec3i(bTile(s));c[axis]+=bSign(f);return bSlotAt(c);}
// The slot + 1 across face f of slot s, as coarseBake baked it (bCoarseNeighbour).
fn bNearSlot(s:u32,f:u32)->u32{return bitcast<u32>(coarse[bC(9u+f,s)]);}
// A 64-lane group's eight slots from base: their face-neighbour slots (+1),
// loaded once, and the 2h aggregate + 1 across face f of aggregate c, the
// i-th slot of the group.
var<workgroup> bMiddleSlots:array<u32,48>;
fn bMiddleLoadSlots(base:u32,n:u32,lane:u32){if(lane<48u){let s=base+lane/6u;var slot=0u;if(s<n){slot=bNearSlot(s,lane%6u);}bMiddleSlots[lane]=slot;}}
fn bMiddleNear(i:u32,c:u32,f:u32)->u32{
 let axis=f/2u;var m=vec3i(bMiddleLocal(c%8u));m[axis]+=bSign(f);var slot=c/8u+1u;
 if(m[axis]<0||m[axis]>1){slot=bMiddleSlots[6u*i+f];m[axis]=(m[axis]+2)%2;}
 if(slot==0u){return 0u;}
 return (slot-1u)*8u+u32(m.x+2*m.y+4*m.z)+1u;
}
`;
  const slots=/* wgsl */`@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32,@builtin(num_workgroups) groups:vec3u`;
  const band=header+indexed(false)+rows+solve+aggregates;
  const sweep=(block:number)=>band+/* wgsl */`
override bColour:u32=0u;
// The settling launches after the start relax only the shut rows. The start
// carries the 4h pressure to a row's own depth; a shut row is liquid by its
// open neighbours' phi, so it started at their pressure, one rho g h short
// under a floor. Relaxed with the open rows, half of them (the colour swept
// second) first pulled their open neighbour down by a tenth of that, and
// what four cycles left of the spread error drove a resting pool along a
// sloped floor until the band stopped converging. Their own row from their
// neighbours' start is exact wherever that start is.
override bSettle:u32=0u;
// Half sweeps this launch runs, colours alternating from bColour. More than
// one is a one-group launch (a small band's): the group strides every slot
// and a storage barrier orders its half sweeps as separate launches would.
// That group is wide: one of 32 lanes is serial over the slots.
override bHalves:u32=1u;
var<workgroup> bSweepCount:u32;
// One red-black Gauss-Seidel half sweep at h; the wall halos follow their row.
// 32 lanes relax one slot's 32 cells of this colour: tile origins are even,
// so the colour is the local parity. A group holds ${block} slot${block>1?"s":""} at a time.
@compute @workgroup_size(${32*block}) fn main(${slots}){
 if(lane==0u){bSweepCount=bLive();}
 let n=workgroupUniformLoad(&bSweepCount);let c=lane&31u;
 for(var half=0u;half<bHalves;half++){
  let colour=(bColour+half)&1u;
  for(var s=group.x*${block}u+(lane>>5u);s<n;s+=groups.x*${block}u){
   let y=(c>>1u)&3u;let z=c>>3u;let l=vec3u(2u*(c&1u)+((y+z+colour)&1u),y,z);
   let p=vec3i(bTile(s)*4u+l);let cell=s*64u+colour*32u+c;
   let kinds=bKinds(cell);let diagonal=bDiagonal(cell);
   if(bLiquid(kinds)&&diagonal>0.0&&(bSettle==0u||bShut(kinds))){
    var next=(bRhs(cell)+bOffNear(cell,s,l,p,kinds))/diagonal;if(bInside(kinds)){next=max(next,-bBase(cell));}solve[cell]=next;
    bFollowHalo(cell,p,kinds,next);
   }
  }
  if(bHalves>1u){storageBarrier();}
 }
}`;
  const coarseForms:[string,number,number,number][]=[["coarseSolve",lanes,held,shared]];
  const sources:Record<string,string>={
   list:header+surface+UniformColdFreeze.wgsl(this.cold,1,14)+(this.fields.warmTiers?uniformRegionHeatWGSL(1,25,this.simulation.capacity.tiles/64)+"fn bCoarseWarm(t:u32)->bool{return umRegionTierAtTile(t)==UM_TIER_WARM;}":"fn bCoarseWarm(t:u32)->bool{return false;}")+/* wgsl */`
@group(1) @binding(0) var<uniform> params:BandParams;
@group(1) @binding(2) var<storage,read_write> index:array<atomic<u32>>;
var<workgroup> member:atomic<u32>;
// Every h simulation tile with a liquid pressure row is a band tile: its
// h velocity is divergence-free only after the h solve. A frozen region's
// tile is not (UniformColdFreeze): its velocity holds, the root alone
// answers there. Nor, with warmTiers, is a WARM region's tile: its velocity
// is the 4h interpolation (bCoarseWarm).
@compute @workgroup_size(64) fn main(${slots}){
 for(var j=group.x;j<umCounts.x;j+=groups.x){
  let t=bHTile(j);
  if(lane==0u){atomicStore(&member,0u);}
  workgroupBarrier();
  if(bWet(bPhiH(vec3i(umTileCoord(t)*4u+bLocal(lane))),1.0)){atomicStore(&member,1u);}
  if(workgroupUniformLoad(&member)!=0u&&lane==0u&&!umFrozenCell(vec3i(umTileCoord(t)*4u))&&!bCoarseWarm(t)){
   ${this.spatialOrder?"atomicStore(&index[SLOTS+t],1u);":`let slot=atomicAdd(&index[0],1u);
   if(slot<bCap()){atomicStore(&index[LIST+slot],t);atomicStore(&index[SLOTS+t],slot+1u);}else{atomicStore(&index[1],1u);}`}
  }
 }
}`,

   listPrefix:header+indexed(true)+/* wgsl */`
const B_BLOCKS=(UM_TILES+255u)/256u;
const B_TOTALS=SLOTS+UM_TILES;
// The list is in tile key order: a member's rank among the members below it.
var<workgroup> prefix:array<u32,256>;
@compute @workgroup_size(256) fn main(${slots}){
 let i=256u*group.x+lane;var count=0u;
 if(i<UM_TILES){count=atomicLoad(&index[SLOTS+i]);}
 prefix[lane]=count;workgroupBarrier();
 for(var stride=1u;stride<256u;stride*=2u){
  var add=0u;if(lane>=stride){add=prefix[lane-stride];}
  workgroupBarrier();prefix[lane]+=add;workgroupBarrier();
 }
 if(i<UM_TILES){atomicStore(&index[SLOTS+i],select(0u,prefix[lane],count!=0u));}
 if(lane==255u){atomicStore(&index[B_TOTALS+group.x],prefix[lane]);}
}`,
   listOffsets:header+indexed(true)+/* wgsl */`
const B_BLOCKS=(UM_TILES+255u)/256u;
const B_TOTALS=SLOTS+UM_TILES;
var<workgroup> prefix:array<u32,256>;
@compute @workgroup_size(256) fn main(@builtin(local_invocation_index) lane:u32){
 let per=(B_BLOCKS+255u)/256u;let first=min(B_BLOCKS,lane*per);let last=min(B_BLOCKS,first+per);
 var total=0u;for(var i=first;i<last;i++){total+=atomicLoad(&index[B_TOTALS+i]);}
 prefix[lane]=total;workgroupBarrier();
 for(var stride=1u;stride<256u;stride*=2u){
  var add=0u;if(lane>=stride){add=prefix[lane-stride];}
  workgroupBarrier();prefix[lane]+=add;workgroupBarrier();
 }
 var offset=prefix[lane]-total;
 for(var i=first;i<last;i++){let count=atomicLoad(&index[B_TOTALS+i]);atomicStore(&index[B_TOTALS+i],offset);offset+=count;}
 if(lane==255u){atomicStore(&index[0],prefix[lane]);if(prefix[lane]>bCap()){atomicStore(&index[1],1u);}}
}`,
   listScatter:header+indexed(true)+/* wgsl */`
const B_TOTALS=SLOTS+UM_TILES;
@compute @workgroup_size(256) fn main(${slots}){
 let t=256u*group.x+lane;if(!umTileLive(t)){return;}
 let local=atomicLoad(&index[SLOTS+t]);if(local==0u){return;}
 let slot=atomicLoad(&index[B_TOTALS+t/256u])+local-1u;
 if(slot<bCap()){atomicStore(&index[LIST+slot],t);atomicStore(&index[SLOTS+t],slot+1u);}else{atomicStore(&index[SLOTS+t],0u);}
}`,
   prep:header+indexed(false)+theta+surface+UniformColdFreeze.wgsl(this.cold,1,14)+/* wgsl */`
@group(1) @binding(9) var<storage,read_write> rows:array<f32>;
// The iterate holds each row's phi until init starts it.
@group(1) @binding(10) var<storage,read_write> solve:array<f32>;
// The authorities' parameters: z is the volume control rate (1/s).
@group(1) @binding(4) var<uniform> authority:vec4f;
@group(1) @binding(7) var velocity:texture_3d<f32>;
@group(1) @binding(8) var<storage,read> negative:array<f32>;
fn bForcedField(p:vec3i,axis:u32,sign:i32)->f32{
 var a=p;if(sign<0){a[axis]-=1;}
 if(umFaceLow(a,axis)){return negative[umLowFaceIndex(vec3u(p),axis)];}
 return textureLoad(velocity,a,0)[axis];
}
// This solve follows the current solid rebuild, before any body integration.
// The generic umCellOpen must also serve the frame tail after bodies move;
// here its exact stored h fractions are current, including moving bodies.
override bLocalSolidWork:bool=true;
${S?/* wgsl */`fn bHeadCapacity(p:vec3i)->f32{
 if(bLocalSolidWork&&umSolidEnabled()){
  let tile=umTileKey(p>>vec3u(2u));
  if(umKeyStored(tile)){
   let local=vec3u(p)&vec3u(3u);let cell=local.x+4u*(local.y+4u*local.z);
   return umSolidRecord[umSolidCount()+4u*umSolidTiles()+16u*tile+(cell>>2u)][cell&3u];
  }
 }
 return umCellOpen(p);
}`:""}
// Row assembly, baked once per solve.
@compute @workgroup_size(64) fn main(${slots}){
 let N=bN();let l=bLocal(lane);
 for(var s=group.x;s<bCount();s+=groups.x){
  let p=vec3i(bTile(s)*4u+bLocal(lane));let cell=s*64u+bRow(bLocal(lane));
  let own=bPhiH(p);let liquid=bWet(own,1.0);${S?`// Current 16h geometry certificate: no fractional cell or body in the
  // region/ring. It is consumed only here, before body integration.
  let solidClear=bLocalSolidWork&&umSolidRegionClearAtBuild(p);
  var shut=false;var inside=false;if(!solidClear){shut=!umCellIsOpen(p);inside=umCellInsideSolid(p);}`:""}
  var kinds=select(0u,0x80000000u,liquid)${S?"|select(0u,0x40000000u,inside)|select(0u,0x20000000u,shut)":""};var diagonal=0.0;var divergence=0.0;
  ${S?`// Native divergenceAtWithCapacity: V u + (V_i - V) u_s, u_s the moving wall's.
  // A body's face velocity is zero past 2 hmax of its bounding sphere
  // (umSolidFaceVelocity) and a face lies within 1h of its cell's region:
  // a region whose pose reach exceeds 3h has no moving-wall flux.
  let bodies=liquid&&!solidClear&&umBodyCount()>0u&&umBodyPoseNearTile(umTileKey(p>>vec3u(2u)),${UNIFORM_BODY_FLUX_REACH}u);var capacity=0.0;
  if(bodies||!bLocalSolidWork){capacity=bHeadCapacity(p);}`:""}
  for(var f=0u;f<6u;f++){
   let axis=f/2u;let sign=bSign(f);let h=params.hDt[axis];var q=p;q[axis]+=sign;
   let forced=bForcedField(p,axis,sign);var kind=K_GHOST;var coefficient=0.0;var fraction=1.0;
   ${S?`var low=p;if(sign<0){low[axis]-=1;}var high=low;high[axis]+=1;
   var volume=1.0;if(!solidClear||!umSolidValid(low)||!umSolidValid(high)){volume=umPressureFaceV(low,axis);}var sealed=false;`:""}
   // What lies across the face. A band tile's cells are stored: only a face
   // out of the tile asks.
   var across=UM_RESIDENT;if(l[axis]==select(0u,3u,sign>0)){across=umCellKind(q);}
   if(across!=UM_RESIDENT){
    if(across==UM_AIR){kind=K_OPEN;coefficient=1.0/(h*h*bTheta(own,0.5*params.policy.w));}
    else{kind=K_WALL;coefficient=0.5/(h*h);fraction=0.5;}
   }
   // A face to a frozen cell (only out of the tile: band tiles are live) is
   // Neumann, wet or dry: its velocity holds F, the projection writes it back.
   else if(l[axis]==select(0u,3u,sign>0)&&umFrozenCell(q)){kind=K_NEUMANN;}
   else{
    let n=bNeighbour(q);
    // A closed cell's phi continues its neighbours': no vertex field of its own.
    var theta=1.0;
    if(liquid!=n.wet){
     if(${S?"!solidClear&&(shut||!umCellIsOpen(q))":"false"}){theta=bPairTheta(own,liquid,n.phi,n.wet);}
     ${S?`// Only nearly closed cells use the buried-interface safeguard.
     // A centre inside a solid does not imply a buried free surface: a rigid
     // body can leave substantial open volume in both cells at the waterline.
     // Forcing theta=1 there moves the pressure boundary away from the actual
     // surface (theta=0.5 for a flat grid-aligned pool), so pressure cannot
     // balance gravity and a stationary body continuously generates flow.
     // At the crossing, a near-shut pocket of a smooth wall (open 1e-4..0.1)
     // that stopped its inflow with kPa drove it out through its sliver of
     // aperture at 1/theta (10-80 m/s), and particles took it out of the
     // band. Closed instead, the film of a gentle slope (its last sub-cell
     // layer) could not drain (nbflip-slope-clearance).
     // 0.1 is below a rigid cell's smallest nonzero sampled fraction (1/8).
     else if(inside&&umCellInsideSolid(q)&&umCellOpen(p)<0.1&&umCellOpen(q)<0.1){theta=1.0;}`:""}
     else{theta=bFaceTheta(bFacePhi(p,axis,sign),own,liquid,n.phi,n.wet);}
    }
    if(bCellAt(q)!=0u){kind=K_BAND;coefficient=1.0/(h*h*theta);}
    else if(n.neumann){kind=K_NEUMANN;}
    else{coefficient=1.0/(h*h*theta);}
    ${S?`// A shut row is liquid by the phi its open neighbours continue into it: it
    // has no surface of its own. Its face to an air row is closed. Left as a
    // free-surface face, a shoreline cell under a slope took flux from the
    // air above it at a theta its own column's phi never changed: nothing
    // restored it, and the column's surface sank a micron in 20 steps.
    if(!solidClear&&kind!=K_NEUMANN){sealed=(liquid&&shut&&!n.wet)||(!liquid&&n.wet&&!umCellIsOpen(q));}`:""}
   }
   ${S?`// CM11a: V scales the coefficient and the flux. A Neumann face's flux is
   // V times the transferred 4h face (init), the root's own flux through it:
   // the record V of a 4h face is the mean of its sixteen h V.
   if(volume<=1e-6||sealed){kind=K_CLOSED;coefficient=0.0;fraction=0.0;}
   else if(kind==K_NEUMANN){fraction=volume;}
   else{if(kind==K_WALL){coefficient=volume/(h*h);}else{coefficient*=volume;}fraction=volume;}
   if(sign>0||l[axis]==0u){rows[10u*N+240u*bCap()+bPackedFace(s,l,f)]=volume;}`:""}
   kinds|=(kind<<(3u*f))|select(0u,1u<<(18u+f),kind==K_BAND&&coefficient==params.solve[1u+axis]);if(sign>0||l[axis]==0u){rows[10u*N+bPackedFace(s,l,f)]=forced;}rows[(3u+f)*N+cell]=coefficient;
   if(kind!=K_NEUMANN){diagonal+=coefficient;divergence+=f32(sign)*fraction*forced/h;}
   ${S?"if(bodies&&kind!=K_WALL&&kind!=K_OPEN&&(!bLocalSolidWork||capacity!=fraction)){divergence+=f32(sign)*(capacity-fraction)*umSolidFaceVelocity(low,axis)/h;}":""}
  }
  // Volume control: the simulation authority's correction of a liquid h
  // owner (umAuthorityLiquid is bWet on the same phi words), rate x its
  // umCapacity, formed here bit for bit.
  let control=authority.z*${S?"select(1.0,umCellOpen(p),umSolidEnabled())":"1.0"};
  rows[cell]=select(0.0,-params.policy.x*(divergence-control)/params.hDt.w,liquid);
  rows[N+cell]=diagonal;rows[2u*N+cell]=bitcast<f32>(kinds);solve[cell]=own;
 }
}`,
   // Galerkin aggregation (piecewise-constant prolongation over liquid rows,
   // restriction its transpose): faces inside an aggregate cancel, wall
   // halos follow their row (no correction flux), Neumann faces carry none.
   middleBake:band+/* wgsl */`
@compute @workgroup_size(64) fn main(${slots}){
 for(var c=group.x*64u+lane;c<bCount()*8u;c+=groups.x*64u){
  let s=c/8u;let a=bMiddleLocal(c%8u);let origin=vec3i(bTile(s)*4u);
  var diagonal=0.0;var couple=array<f32,6>(0.0,0.0,0.0,0.0,0.0,0.0);
  for(var k=0u;k<8u;k++){
   let l=a*2u+bMiddleLocal(k);let cell=s*64u+bRow(l);let kinds=bKinds(cell);
   if(!bLiquid(kinds)||bDiagonal(cell)<=0.0){continue;}
   for(var f=0u;f<6u;f++){
    let kind=bFaceKind(kinds,f);let coefficient=bCoefficient(cell,f);
    if(kind==K_GHOST||kind==K_OPEN){diagonal+=coefficient;continue;}
    if(kind!=K_BAND){continue;}
    let axis=f/2u;var m=vec3i(l);m[axis]+=bSign(f);
    // A face to a band row outside the liquid rows is Dirichlet (p=0 there),
    // inside the aggregate too; between liquid rows of one aggregate it
    // cancels.
    let n=bCellAt(origin+m)-1u;
    if(!bLiquid(bKinds(n))||bDiagonal(n)<=0.0){diagonal+=coefficient;continue;}
    if(all(m>=vec3i(0))&&all(m<vec3i(4))&&all(vec3u(m)/2u==a)){continue;}
    couple[f]+=coefficient;
   }
  }
  // Dirichlet terms stay exact; only couplings between aggregates scale.
  let scale=${schedule.coarseScale};var coupled=0.0;
  for(var f=0u;f<6u;f++){coarse[bM(3u+f,c)]=scale*couple[f];coupled+=scale*couple[f];}
  coarse[bM(2u,c)]=diagonal+coupled;
 }
}`,
   coarseBake:header+indexed(true)+rows+solve+aggregates+/* wgsl */`
@compute @workgroup_size(64) fn main(${slots}){
 let n=bCount();
 for(var s=group.x*64u+lane;s<n;s+=groups.x*64u){
  var diagonal=0.0;var couple=array<f32,6>(0.0,0.0,0.0,0.0,0.0,0.0);
  for(var a=0u;a<8u;a++){
   let c=8u*s+a;let at=bMiddleLocal(a);diagonal+=coarse[bM(2u,c)];
   for(var f=0u;f<6u;f++){
    let axis=f/2u;let inside=select(at[axis]==1u,at[axis]==0u,bSign(f)>0);
    let w=coarse[bM(3u+f,c)];diagonal-=w;if(!inside){couple[f]+=w;}
   }
  }
  // diagonal is now the aggregate's Dirichlet part.
  let scale=${schedule.coarseScale};var coupled=0.0;var near:array<u32,6>;
  for(var f=0u;f<6u;f++){couple[f]=scale*couple[f];coupled+=couple[f];near[f]=bCoarseNeighbour(s,f);coarse[bC(9u+f,s)]=bitcast<f32>(near[f]);}
  // A row the solve updates takes its colour's next position.
  let d=diagonal+coupled;if(d<=0.0){continue;}
  let t=bTile(s);var j=0u;
  if(((t.x+t.y+t.z)&1u)==0u){j=atomicAdd(&index[${RED_WORD}u],1u);}else{j=n-1u-atomicAdd(&index[${BLACK_WORD}u],1u);}
  coarse[bC(2u,j)]=d;coarse[bC(15u,j)]=bitcast<f32>(s);
  for(var f=0u;f<6u;f++){coarse[bC(3u+f,j)]=couple[f];coarse[bC(17u+f,j)]=bitcast<f32>(near[f]);}
 }
}`,
   init:header+indexed(false)+rowsOf(true)+solveOf(true)+aggregates+wet+UniformColdFreeze.wgsl(this.cold,1,14)+/* wgsl */`
@group(1) @binding(11) var velocity:texture_3d<f32>;
@group(1) @binding(12) var<storage,read> coarsePressure:array<f32>;
@group(1) @binding(13) var<storage,read> coarsePhi:array<f32>;
// The 4h owner a stencil of a row of block home (its page: page) reads at
// tile c: c itself, stored or not, but the nearest tile of the row's page on
// an axis where a SOLID page lies across that face (umTileNearFromPage): past
// a wall the owner beside it stands again, and an AIR page is air.
fn bCoarseAt(home:u32,page:vec3i,c:vec3i)->vec3i{return umTileNearFromPage(home,page,c);}
// The root's owner of a tile is its key; a tile with nothing stored reads
// block 0's row (far air at zero pressure, UniformMixedRootOwners).
// (pressure, phi) of a liquid 4h owner at tile q, weight 1; zero for an air
// one and for a closed one (its phi continues its liquid neighbours; it has
// no row).
fn bCoarse(q:vec3i)->vec3f{
 let i=umTileAt(vec3u(q));
 return select(vec3f(0.0),vec3f(coarsePressure[i],coarsePhi[i],1.0),bWet(coarsePhi[i],4.0)&&umTileOpenAt(q)>1e-5);
}
// A 4h owner as (pressure, liquid, air): bCoarse's liquid owner, and an open
// one that is not liquid; a closed one is neither.
fn bCoarseKind(q:vec3i)->vec3f{
 let i=umTileAt(vec3u(q));
 let open=umTileOpenAt(q)>1e-5;let wet=bWet(coarsePhi[i],4.0);
 return vec3f(select(0.0,coarsePressure[i],open&&wet),f32(open&&wet),f32(open&&!wet));
}
// What a row missing owners from its stencil b, b+1 (weights w) is carried
// by. rise: the 4h pressure's vertical difference per 4h spacing and its
// weight. Each of the four columns gives its two owners' difference where
// both are liquid, and where one is (the other closed, or the same owner
// past a wall) that owner's against the liquid owner beyond it, one more
// load; a column with neither gives none; columns weigh as their liquid
// owners do in the mean. height: the liquid owners' weighted height above b
// and their weight. air: the air owners' weight.
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
  // One owner for both (a wall holds the other): the owner beyond it is on
  // the side away from that wall.
  var own=low;var at=y0;var beyond=select(y0-1,y0+1,y0!=c.y);
  if(y1!=y0){if(high.y>0.0){own=high;at=y1;beyond=y1+1;}else{beyond=y0-1;}}
  if(own.y==0.0){continue;}
  // No owner beyond a wall.
  let t2=bCoarseAt(home,page,vec3i(c.x,beyond,c.z));if(t2.y!=beyond){continue;}
  let other=bCoarseKind(t2);
  if(other.y>0.0){carry.rise+=liquid*vec2f((other.x-own.x)*f32(beyond-at),1.0);}
 }
 return carry;
}
// The velocity of Neumann face f of band cell 'cell' at p: the transferred 4h
// face, the flux the root solved through it. Where neither 4h owner at the
// face is a liquid row the root solved none: its pressure is zero on both
// sides, and umProject writes the face zero unless detached mass is beside
// it, a wall across the liquid the two owners hold beside their centres.
// There the h face keeps its forced velocity, which the root's zero pressure
// gradient across it leaves as it is. Both cells of a Neumann face are
// stored, so each owner is its own tile's.
fn bNeumann(cell:u32,p:vec3i,f:u32)->f32{
 let axis=f/2u;var q=p;q[axis]+=bSign(f);
 if(bCoarseKind(p>>vec3u(2u)).y==0.0&&bCoarseKind(q>>vec3u(2u)).y==0.0){return bForced(cell,f);}
 var a=p;if(bSign(f)<0){a[axis]-=1;}
 return textureLoad(velocity,a,0)[axis];
}
@compute @workgroup_size(64) fn main(${slots}){
 for(var s=group.x;s<bCount();s+=groups.x){
  let key=bIndex(LIST+s);let p=vec3i(umTileCoord(key)*4u+bLocal(lane));let cell=s*64u+bRow(bLocal(lane));
  let kinds=bKinds(cell);
  // Air rows skip the RHS loop but still keep their Neumann velocities.
  if(!bLiquid(kinds)){
   if(((kinds>>2u)&~kinds&~(kinds>>1u)&0x9249u)!=0u){for(var f=0u;f<6u;f++){
    if(bFaceKind(kinds,f)==K_NEUMANN){rows[bForcedWord(cell,f)]=bNeumann(cell,p,f);}
   }}
   solve[cell]=0.0;rows[B_BASE*bN()+cell]=0.0;continue;
  }
  // Trilinear 4h pressure at the h centre. Where an air or a closed owner
  // is in the stencil the liquid-weighted mean is the pressure at the
  // liquid owners' mean depth, not this cell's: it is carried to the cell's
  // own depth, so that a hydrostatic 4h solution starts every band row on
  // its own hydrostatic value. The unscaled mean started the rows over the
  // highest liquid 4h centre at that centre's pressure, up to 2.5 rho g h
  // too high; what four cycles left of it lifted an h patch's surface at
  // 1e-3 m/s every step of a resting pond.
  // Under an air owner the carry is pressure per depth, the smooth quantity
  // under a free surface: the mean scales by this centre's phi over the
  // same weights' 4h phi, which is zero at the surface whatever the flow.
  // That needs phi to be the depth. With only closed owners missing it need
  // not be: past the redistance band phi is the band value and the ratio is
  // one, and a pond that had ever moved started its promoted rows over a
  // sloped floor up to 2 rho g h off and failed the next root solve. Those
  // rows are carried by the 4h pressure's own vertical difference
  // (bCoarseCarry: rho g dy at rest, nothing in free fall), kept to the
  // mean's sign and the range the ratio has. A row with neither an air
  // owner nor a liquid owner over or under any of its own keeps the ratio.
  let y=(vec3f(p)+0.5)/4.0-0.5;let b=vec3i(floor(y));let w=y-floor(y);var sum=vec3f(0.0);
  // The row's block and page: its stencil stops at their walls (bCoarseAt).
  let home=key>>9u;let page=p>>vec3u(5u);
  for(var k=0u;k<8u;k++){
   let bit=vec3u(k&1u,(k>>1u)&1u,k>>2u);let weights=select(1.0-w,w,bit==vec3u(1u));
   sum+=weights.x*weights.y*weights.z*bCoarse(bCoarseAt(home,page,b+vec3i(bit)));
  }
  // prep left the row's phi in the iterate.
  var start=0.0;
  if(sum.z>=0.99999){start=sum.x/sum.z;}
  else if(sum.z>0.0){
   let carry=bCoarseCarry(home,page,b,w);
   if(carry.air>0.0||carry.rise.y==0.0){start=sum.x/sum.z*clamp(solve[cell]*sum.z/min(sum.y,-1e-30),0.0,${(1/UNIFORM_MIXED_THETA_MIN).toFixed(1)});}
   else{
    let mean=sum.x/sum.z;let bound=${(1/UNIFORM_MIXED_THETA_MIN).toFixed(1)}*mean;
    start=clamp(mean+(w.y-carry.height.x/carry.height.y)*carry.rise.x/carry.rise.y,min(bound,0.0),max(bound,0.0));
   }
  }
  if(bInside(kinds)){start=max(start,0.0);}
  // The start is the row's base; the iterate is the correction to it.
  rows[B_BASE*bN()+cell]=start;solve[cell]=0.0;
  // Neumann faces carry the projected 4h flux into the row.
  var rhs=bRhs(cell);
  for(var f=0u;f<6u;f++){
   let axis=f/2u;let sign=bSign(f);
   if(bFaceKind(kinds,f)==K_NEUMANN){
    // A frozen face's flux composes its region's term; the row keeps F.
    var a=p;if(sign<0){a[axis]-=1;}
    let kept=bNeumann(cell,p,f);rows[bForcedWord(cell,f)]=kept;rhs-=params.policy.x*f32(sign)*${S?"bVolume(cell,f)*":""}(kept+umColdFaceTerm(a,axis,1u))/(params.hDt[axis]*params.hDt.w);}
  }
  rows[cell]=rhs;bFollowHalo(cell,p,kinds,0.0);
 }
}`,
   // The rows against their start. The band's pressure is rho g depth, its
   // rows' balance the difference of neighbours a part in 1e4 of it apart:
   // iterated as the pressure itself, a float32 row holds its divergence to
   // 6 ulp(p)/h^2 only (rho/dt times 2.6e-5 1/s per ulp at 1500 Pa and
   // h = 12.5 mm), one sign over a tile, and the all-4h root answered what
   // the band left on its tiles with seam flux every step: a resting pool
   // with an h box pumped liquid in through the box's seam and out of its
   // surface. The differences of the start are exact; the rows then solve
   // for the correction to it, a few Pa at most.
   rebase:header+indexed(false)+rowsOf(true)+solve+aggregates+/* wgsl */`
@compute @workgroup_size(64) fn main(${slots}){
 for(var s=group.x;s<bCount();s+=groups.x){
  let p=vec3i(bTile(s)*4u+bLocal(lane));let cell=s*64u+bRow(bLocal(lane));let kinds=bKinds(cell);
  if(!bLiquid(kinds)){continue;}
  let own=bBase(cell);var sums=vec3f(0.0);var open=0.0;
  for(var axis=0u;axis<3u;axis++){
   // w0 d0 + w1 d1 as w0 (d0 + d1) + (w1 - w0) d1: equal coefficients
   // leave the second difference of the start, exact where it cancels.
   var w=vec2f(0.0);var d=vec2f(0.0);
   for(var side=0u;side<2u;side++){
    let f=2u*axis+side;let kind=bFaceKind(kinds,f);
    if(kind==K_BAND){var q=p;q[axis]+=bSign(f);w[side]=bCoefficient(cell,f);d[side]=bBase(bCellAt(q)-1u)-own;}
    else if(kind==K_GHOST||kind==K_OPEN){open+=bCoefficient(cell,f);}
   }
   sums[axis]=w.x*(d.x+d.y)+(w.y-w.x)*d.y;
  }
  rows[cell]=bRhs(cell)+((sums.x+sums.z)+sums.y-open*own);
 }
}`,
   sweep:sweep(CELL_BLOCK),
   // h residual into the 2h aggregates, and its largest liquid row into
   // cycle bCycle's history word. With every 2h correction zero, the first
   // red half sweep of the 2h level is local: it runs here.
   restrict:header+indexed(true)+rows+solve+aggregates+/* wgsl */`
override bCycle:u32=0u;
var<workgroup> worst:atomic<u32>;
var<workgroup> bRestrictCount:u32;
var<workgroup> bCellResidual:array<f32,64>;
// A lane per h cell; lanes 0..7 then sum their 2h aggregate's eight cells.
@compute @workgroup_size(64) fn main(${slots}){
 if(lane==0u){atomicStore(&worst,0u);bRestrictCount=select(bCount(),0u,bConvergedBefore(bCycle));}
 let n=workgroupUniformLoad(&bRestrictCount);
 var largest=0u;
 for(var s=group.x;s<n;s+=groups.x){
  workgroupBarrier();
  let l=bLocal(lane);let cell=s*64u+bRow(l);let kinds=bKinds(cell);var r=0.0;
  if(bLiquid(kinds)&&bDiagonal(cell)>0.0){r=bResidualNear(cell,s,l,vec3i(bTile(s)*4u+l),kinds);largest=max(largest,bResidualMagnitude(r));}
  bCellResidual[lane]=r;workgroupBarrier();
  if(lane<8u){
   let a=bMiddleLocal(lane);var sum=0.0;
   for(var k=0u;k<8u;k++){let q=a*2u+bMiddleLocal(k);sum+=bCellResidual[q.x+4u*(q.y+4u*q.z)];}
   let c=8u*s+lane;let g=bTile(s)*2u+a;var e=0.0;
   if(((g.x+g.y+g.z)&1u)==0u){let diagonal=coarse[bM(2u,c)];if(diagonal>0.0){e=(sum+0.0)/diagonal;}}
   coarse[bM(1u,c)]=sum;coarse[bM(0u,c)]=e;
  }
 }
 atomicMax(&worst,bResidualMagnitude(bitcast<f32>(largest)*params.policy.z));
 workgroupBarrier();
 if(lane==0u&&n>0u){atomicMax(&index[HISTORY+bCycle],atomicLoad(&worst));atomicOr(&index[VALID],1u<<bCycle);}
}`,
   // With bProlong (red only), the 4h correction is prolonged on the fly: a
   // red update never reads its own value, and black values are prolonged
   // before any black update reads them.
   middleDownFused:band+uniformPressureBandMiddleWGSL(lanes,2*schedule.middleSweeps-1,1,false),
   middleUpFused:band+uniformPressureBandMiddleWGSL(lanes,2*schedule.middleSweeps,0,true),
   middleSweep:band+/* wgsl */`
override bColour:u32=0u;
override bProlong:bool=false;
// As the h sweep's: half sweeps per launch, colours alternating from
// bColour, more than one in a one-group launch. bProlong is the first's.
override bHalves:u32=1u;
var<workgroup> bMiddleCount:u32;
@compute @workgroup_size(64) fn main(${slots}){
 if(lane==0u){bMiddleCount=bLive();}
 let n=workgroupUniformLoad(&bMiddleCount);
 for(var half=0u;half<bHalves;half++){
  let colour=(bColour+half)&1u;let prolong=bProlong&&half==0u;
  for(var base=group.x*8u;base<n;base+=groups.x*8u){
   workgroupBarrier();
   bMiddleLoadSlots(base,n,lane);
   workgroupBarrier();
   let c=base*8u+lane;
   if(c<n*8u){
    let g=bTile(c/8u)*2u+bMiddleLocal(c%8u);let diagonal=coarse[bM(2u,c)];
    if(((g.x+g.y+g.z)&1u)==colour&&diagonal>0.0){
     var off=0.0;
     for(var f=0u;f<6u;f++){
      let m=bMiddleNear(lane/8u,c,f);
      if(m!=0u){var v=coarse[bM(0u,m-1u)];if(prolong&&coarse[bM(2u,m-1u)]>0.0){v+=coarse[bC(0u,(m-1u)/8u)];}off+=coarse[bM(3u+f,c)]*v;}
     }
     coarse[bM(0u,c)]=(coarse[bM(1u,c)]+off)/diagonal;
    }
   }
  }
  if(bHalves>1u){storageBarrier();}
 }
}`,
   // 2h residual into the tile aggregate (zeroing its correction).
   // A lane per 2h aggregate (eight slots per group), then lanes 0..7 sum
   // their slot's eight in aggregate order.
   middleRestrict:band+/* wgsl */`
var<workgroup> bMiddleCount:u32;
var<workgroup> bMiddleResidual:array<f32,64>;
@compute @workgroup_size(64) fn main(${slots}){
 if(lane==0u){bMiddleCount=bLive();}
 let n=workgroupUniformLoad(&bMiddleCount);
 for(var base=group.x*8u;base<n;base+=groups.x*8u){
  workgroupBarrier();bMiddleLoadSlots(base,n,lane);workgroupBarrier();
  let c=base*8u+lane;var r=0.0;
  if(c<n*8u){
   let diagonal=coarse[bM(2u,c)];
   if(diagonal>0.0){
    var off=0.0;
    for(var f=0u;f<6u;f++){let m=bMiddleNear(lane/8u,c,f);if(m!=0u){off+=coarse[bM(3u+f,c)]*coarse[bM(0u,m-1u)];}}
    r=coarse[bM(1u,c)]+off-diagonal*coarse[bM(0u,c)];
   }
  }
  bMiddleResidual[lane]=r;workgroupBarrier();
  if(lane<8u&&base+lane<n){
   var sum=0.0;for(var a=0u;a<8u;a++){sum+=bMiddleResidual[8u*lane+a];}
   coarse[bC(1u,base+lane)]=sum;coarse[bC(0u,base+lane)]=0.0;
  }
 }
}`,
   // Every red-black sweep of the 4h aggregates in one workgroup: one
   // launch instead of two per sweep, chosen on the GPU from the live count.
   // The rows are coarseBake's colour-ordered positions: red [0,red), black
   // [black,n); a row whose diagonal vanishes has none and keeps its zero
   // correction. Up to SOLVE_SLOTS positions sweep from registers (slots
   // below 2^16: n bounds them); a larger band streams its rows from storage
   // and keeps the first SHARED corrections in workgroup memory, the rest in
   // storage.
   ...Object.fromEntries(coarseForms.map(([name,lanes,held,shared])=>[name,band+/* wgsl */`
const L:u32=${lanes}u;const HELD:u32=${held}u;const SOLVE_SLOTS:u32=L*HELD;const SHARED:u32=${shared}u;
var<workgroup> bSolveCount:u32;
var<workgroup> bRedCount:u32;
var<workgroup> bBlackStart:u32;
var<workgroup> bCorrection:array<f32,SHARED>;
fn bCorrectionAt(s:u32)->f32{if(s<SHARED){return bCorrection[s];}return coarse[bC(0u,s)];}
@compute @workgroup_size(${lanes}) fn main(@builtin(local_invocation_index) lane:u32){
 if(lane==0u){let live=bLive();let r=min(bIndex(${RED_WORD}u),live);bSolveCount=live;bRedCount=r;bBlackStart=live-min(bIndex(${BLACK_WORD}u),live-r);}
 let n=workgroupUniformLoad(&bSolveCount);let red=workgroupUniformLoad(&bRedCount);let black=workgroupUniformLoad(&bBlackStart);
 // No aggregate rows means no red-black updates or barriers to perform.
 if(n==0u){return;}
 if(n<=SOLVE_SLOTS){
  var diagonal:array<f32,HELD>;var residual:array<f32,HELD>;var slot:array<u32,HELD>;var near:array<u32,${3*held}>;var weight:array<f32,6>;
  for(var k=0u;k<HELD;k++){
   let j=lane+k*L;diagonal[k]=0.0;
   if(j<n){bCorrection[j]=coarse[bC(0u,j)];}
   if(j<red||(j>=black&&j<n)){
    let s=bitcast<u32>(coarse[bC(15u,j)]);slot[k]=s;diagonal[k]=coarse[bC(2u,j)];residual[k]=coarse[bC(1u,s)];
    for(var f=0u;f<3u;f++){near[3u*k+f]=bitcast<u32>(coarse[bC(17u+2u*f,j)])|(bitcast<u32>(coarse[bC(18u+2u*f,j)])<<16u);}
    if(k==0u){for(var f=0u;f<6u;f++){weight[f]=coarse[bC(3u+f,j)];}}
   }
  }
  workgroupBarrier();
  for(var sweep=0u;sweep<${schedule.coarseSweeps}u;sweep++){
   for(var c=0u;c<2u;c++){
    for(var k=0u;k<HELD;k++){
     let j=lane+k*L;
     if(diagonal[k]<=0.0||select(j>=red,j<black,c==1u)){continue;}
     var off=0.0;for(var f=0u;f<6u;f++){let m=(near[3u*k+f/2u]>>(16u*(f&1u)))&0xffffu;if(m!=0u){var w=weight[f];if(k!=0u){w=coarse[bC(3u+f,j)];}off+=w*bCorrection[m-1u];}}
     let old=bCorrection[slot[k]];bCorrection[slot[k]]=old+${schedule.coarseRelaxation}*((residual[k]+off)/diagonal[k]-old);
    }
    workgroupBarrier();
   }
  }
  for(var k=0u;k<HELD;k++){let s=lane+k*L;if(s<n){coarse[bC(0u,s)]=bCorrection[s];}}
  return;
 }
 // Each call's residual, gathered into position order once.
 for(var s=lane;s<min(n,SHARED);s+=L){bCorrection[s]=coarse[bC(0u,s)];}
 for(var j=lane;j<n;j+=L){if(j<red||j>=black){coarse[bC(16u,j)]=coarse[bC(1u,bitcast<u32>(coarse[bC(15u,j)]))];}}
 workgroupBarrier();storageBarrier();
 for(var sweep=0u;sweep<${schedule.coarseSweeps}u;sweep++){
  for(var colour=0u;colour<2u;colour++){
   let first=select(0u,black,colour==1u);let last=select(red,n,colour==1u);
   for(var j=first+lane;j<last;j+=L){
    var off=0.0;for(var f=0u;f<6u;f++){let m=bitcast<u32>(coarse[bC(17u+f,j)]);if(m!=0u){off+=coarse[bC(3u+f,j)]*bCorrectionAt(m-1u);}}
    let s=bitcast<u32>(coarse[bC(15u,j)]);let old=bCorrectionAt(s);let next=old+${schedule.coarseRelaxation}*((coarse[bC(16u,j)]+off)/coarse[bC(2u,j)]-old);
    if(s<SHARED){bCorrection[s]=next;}else{coarse[bC(0u,s)]=next;}
   }
   // Corrections past SHARED live in storage: only then must the phase's
   // storage writes be visible to the next.
   workgroupBarrier();if(n>SHARED){storageBarrier();}
  }
 }
 for(var s=lane;s<min(n,SHARED);s+=L){coarse[bC(0u,s)]=bCorrection[s];}
}`])),
   // The same solve for a large band, a launch per half sweep: colour
   // bColour's rows (coarseBake's positions), a row to a thread, each
   // against the other colour's corrections as the launch before left them.
   coarseSweep:band+/* wgsl */`
override bColour:u32=0u;
@compute @workgroup_size(64) fn main(${slots}){
 let live=bLive();let red=min(bIndex(${RED_WORD}u),live);let black=live-min(bIndex(${BLACK_WORD}u),live-red);
 let first=select(0u,black,bColour==1u);let last=select(red,live,bColour==1u);
 for(var j=first+group.x*64u+lane;j<last;j+=groups.x*64u){
  var off=0.0;for(var f=0u;f<6u;f++){let m=bitcast<u32>(coarse[bC(17u+f,j)]);if(m!=0u){off+=coarse[bC(3u+f,j)]*coarse[bC(0u,m-1u)];}}
  let s=bitcast<u32>(coarse[bC(15u,j)]);
  let old=coarse[bC(0u,s)];coarse[bC(0u,s)]=old+${schedule.coarseRelaxation}*((coarse[bC(1u,s)]+off)/coarse[bC(2u,j)]-old);
 }
}`,
   middleProlong:band+/* wgsl */`
@compute @workgroup_size(64) fn main(${slots}){
 let n=bLive();
 for(var s=group.x;s<n;s+=groups.x){
  let l=bLocal(lane);let p=vec3i(bTile(s)*4u+l);let cell=s*64u+bRow(l);let kinds=bKinds(cell);
  if(!bLiquid(kinds)||bDiagonal(cell)<=0.0){continue;}
  let a=l/2u;var next=solve[cell]+coarse[bM(0u,8u*s+a.x+2u*a.y+4u*a.z)];if(bInside(kinds)){next=max(next,-bBase(cell));}
  solve[cell]=next;bFollowHalo(cell,p,kinds,next);
 }
}`,
   // Before ordinary smoothing, or after bCycles encoded cycles.
   measure:header+indexed(true)+rows+solve+aggregates+/* wgsl */`
override bCycles:u32=1u;
override bInitial:bool=false;
var<workgroup> worst:atomic<u32>;
var<workgroup> bMeasureCount:u32;
${this.fields.dynamics?`@group(1) @binding(24) var<storage,read_write> bDynamics:array<atomic<u32>>;
var<workgroup> bTileWorst:atomic<u32>;
// The predictor's residual by region (UNIFORM_REGION_DYNAMICS.pressureResidual):
// the slot's largest, one atomic a tile.
fn bPublishTile(s:u32,lane:u32,r:u32){
 atomicMax(&bTileWorst,r);workgroupBarrier();
 if(lane==0u){
  let at=${UNIFORM_REGION_DYNAMICS_WORDS}u*umPageOfTile(umTileKey(vec3i(bTile(s))))+${UNIFORM_REGION_DYNAMICS.pressureResidual}u;
  let worst=atomicExchange(&bTileWorst,0u);if(worst>atomicLoad(&bDynamics[at])){atomicMax(&bDynamics[at],worst);}
 }
 workgroupBarrier();
}`:""}
// The largest band row residual, in divergence units (1/s), over liquid rows,
// into history word bCycles. A converged solve keeps its restriction's
// residual (nothing moved since): the receipt takes that cycle and sets the
// convergence word.
@compute @workgroup_size(64) fn main(${slots}){
 if(lane==0u){
  atomicStore(&worst,0u);bMeasureCount=bCount();
  if(!bInitial&&bConvergedBefore(bCycles)){bMeasureCount=0u;}
 }
 let n=workgroupUniformLoad(&bMeasureCount);
 var largest=0u;
 for(var s=group.x;s<n;s+=groups.x){
  let l=bLocal(lane);let p=vec3i(bTile(s)*4u+l);let cell=s*64u+bRow(l);
  let kinds=bKinds(cell);var r=0u;if(bLiquid(kinds)&&bDiagonal(cell)>0.0){r=bResidualMagnitude(bResidualNear(cell,s,l,p,kinds)*params.policy.z);}
  largest=max(largest,r);${this.fields.dynamics?"if(bInitial){bPublishTile(s,lane,r);}":""}
 }
 atomicMax(&worst,largest);
 workgroupBarrier();
 if(lane==0u&&n>0u){
  let r=atomicLoad(&worst);
  if(bInitial){atomicMax(&index[INITIAL],r);atomicOr(&index[VALID],INITIAL_BIT);}
  else{atomicMax(&index[2],r);atomicMax(&index[HISTORY+bCycles],r);atomicOr(&index[VALID],1u<<bCycles);}
 }
 if(!bInitial&&group.x==0u&&lane==0u){
  var cycle=bCycles;for(var j=bCycles;j>0u;j--){if(bStop(j-1u)){cycle=j-1u;}}
  if(bInitialMet()||bCount()==0u){cycle=0u;}
  atomicStore(&index[CYCLE],cycle);
  if(cycle<bCycles){atomicStore(&index[DONE],1u);atomicMax(&index[2],select(bIndex(HISTORY+cycle),bIndex(INITIAL),bInitialMet()));}
 }
}`,
   project:band+/* wgsl */`
@group(1) @binding(15) var field:texture_3d<f32>;
@group(1) @binding(16) var output:texture_storage_3d<rgba32float,write>;
@group(1) @binding(17) var<storage,read_write> boundary:array<f32>;
@group(1) @binding(21) var<storage,read_write> presented:array<f32>;
// umProject on one face of band cell 'cell' at p. Init retained transferred
// 4h Neumann velocity in its otherwise unused forced-face row.
fn bFace(cell:u32,p:vec3i,f:u32)->f32{
 let kinds=bKinds(cell);let kind=bFaceKind(kinds,f);
 if(kind==K_NEUMANN){return bForced(cell,f);}
 let axis=f/2u;let sign=bSign(f);let h=params.hDt[axis];let liquid=bLiquid(kinds);
 // A closed face moves with its wall (pressureFaceData's u_s; zero when static).
 ${S?"if(kind==K_CLOSED){var low=p;if(sign<0){low[axis]-=1;}return umSolidFaceVelocity(low,axis);}":""}
 // The gradient coefficient: the row coefficient without its V.
 let forced=bForced(cell,f);let coefficient=bCoefficient(cell,f)${S?"/bVolume(cell,f)":""};let own=solve[cell];let base=bBase(cell);let scale=params.policy.z;
 if(kind==K_WALL){if(!liquid){return forced;}return forced-scale*f32(sign)*(solve[bHalo(p,axis,sign)]-own)/h;}
 if(kind==K_OPEN){if(!liquid){return forced;}return forced+scale*f32(sign)*(base+own)*coefficient*h;}
 var other=0.0;var otherBase=0.0;var otherLiquid=false;
 if(kind==K_BAND){var q=p;q[axis]+=sign;let n=bCellAt(q)-1u;other=solve[n];otherBase=bBase(n);otherLiquid=bLiquid(bKinds(n));}
 if(!liquid&&!otherLiquid){return 0.0;}
 // The start's difference and the correction's, each exact.
 return forced-scale*f32(sign)*((otherBase-base)+(other-own))*coefficient*h;
}
fn bReleased(cell:u32,p:vec3i,f:u32,value:f32)->bool{
 let axis=f/2u;let sign=bSign(f);
 ${S?`let volume=bVolume(cell,f);
 if(sign>0){
  // Native embedded/wall contact where open and closed cells meet, on the
  // separating side's pressure.
  var q=p;q[axis]+=1;let own=umCellIsOpen(p);
  if(own==umCellIsOpen(q)||volume<=1e-6){return false;}
  // Nothing stored across the face: row assembly baked what lies there, a
  // wall (its halo's pressure) or an opening (the ambient 0).
  var pressure=0.0;let kind=bFaceKind(bKinds(cell),f);
  if(kind==K_WALL||kind==K_OPEN){pressure=select(0.0,solve[bHalo(p,axis,sign)]+bBase(cell),bLiquid(bKinds(cell))&&kind==K_WALL);}
  else if(own){let b=bCellAt(q);if(b!=0u&&bLiquid(bKinds(b-1u))){pressure=solve[b-1u]+bBase(b-1u);}}
  else if(bLiquid(bKinds(cell))){pressure=solve[cell]+bBase(cell);}
  // Separation is relative to the wall's own motion (native wall[axis]).
  return pressure<=0.0&&select(1.0,-1.0,own)*(value-umSolidFaceVelocity(p,axis))*params.hDt.w>1e-4*params.hDt[axis];
 }
 if(!umCellIsOpen(p)||volume<=1e-6){return false;}`:""}
 if(bFaceKind(bKinds(cell),f)!=K_WALL){return false;}
 let pressure=select(0.0,solve[bHalo(p,axis,sign)]+bBase(cell),bLiquid(bKinds(cell)));
 return pressure<=0.0&&-f32(sign)*value*params.hDt.w>1e-4*params.hDt[axis];
}
@compute @workgroup_size(128) fn main(${slots}){
 for(var s=group.x;s<bCount();s+=groups.x){
  let origin=vec3i(bTile(s)*4u);
  if(lane<64u){
   // A band cell's own texel: its three positive faces and wall releases.
   let p=origin+vec3i(bLocal(lane));let cell=s*64u+bRow(bLocal(lane));var texel=vec4f(0.0);var released=0u;
   // Pressure publication uses the same row and lane as its projected cell.
   presented[params.capacity.w+s*64u+lane]=bBase(cell)+solve[cell];let lowOpen=umLowOpen(p);
   for(var axis=0u;axis<3u;axis++){
    let value=bFace(cell,p,2u*axis+1u);texel[axis]=value;
    if(bReleased(cell,p,2u*axis+1u,value)){released|=1u<<axis;}
    if(lowOpen[axis]){
     let low=bFace(cell,p,2u*axis);boundary[umLowFaceIndex(vec3u(p),axis)]=low;
     if(bReleased(cell,p,2u*axis,low)){released|=1u<<(axis+3u);}
    }
   }
   texel.w=f32(released);textureStore(output,p,texel);
  }else if(lane<112u){
   // A non-band texel below this tile holds band faces in its positive
   // components. The first band tile along x, then y, then z writes it.
   let axis=(lane-64u)/16u;let k=(lane-64u)%16u;var l=vec3i(0);
   l[axis]=-1;l[(axis+1u)%3u]=i32(k%4u);l[(axis+2u)%3u]=i32(k/4u);let q=origin+l;
   if(!umCellPresent(q)||bCellAt(q)!=0u){continue;}
   var first=true;
   for(var a=0u;a<axis;a++){var r=q;r[a]+=1;if(bCellAt(r)!=0u){first=false;}}
   if(!first){continue;}
   var texel=textureLoad(field,q,0);
   for(var a=0u;a<3u;a++){var r=q;r[a]+=1;let b=bCellAt(r);if(b!=0u){texel[a]=bFace(b-1u,r,2u*a);}}
   textureStore(output,q,texel);
  }
 }
}`,
   // Only non-band texels below the band need retained components: each
   // band cell projects entirely from its rows. Match project's unique
   // seam writer so the copy never reads or writes the full band interior.
   copy:header+indexed(false)+/* wgsl */`
@group(1) @binding(22) var velocity:texture_3d<f32>;
@group(1) @binding(23) var copied:texture_storage_3d<rgba32float,write>;
@compute @workgroup_size(128) fn main(${slots}){
 for(var s=group.x;s<bCount();s+=groups.x){
  if(lane<64u){continue;}
  let origin=vec3i(bTile(s)*4u);var q=origin+vec3i(bLocal(lane));
  if(lane>=64u){
   if(lane>=112u){continue;}
   let axis=(lane-64u)/16u;let k=(lane-64u)%16u;var l=vec3i(0);
   l[axis]=-1;l[(axis+1u)%3u]=i32(k%4u);l[(axis+2u)%3u]=i32(k/4u);q=origin+l;
   if(!umCellPresent(q)||bCellAt(q)!=0u){continue;}
   var first=true;
   for(var a=0u;a<axis;a++){var r=q;r[a]+=1;if(bCellAt(r)!=0u){first=false;}}
   if(!first){continue;}
  }
  textureStore(copied,q,textureLoad(velocity,q,0));
 }
}`,
  };
  const entryOf=(name:string)=>`band${name[0]!.toUpperCase()}${name.slice(1)}`;
  const range=(n:number,from=0)=>Array.from({length:n},(_,i)=>i+from);
  // Specialised launches: red-black colours, each cycle's restriction, the
  // measure after each encodable cycle count.
  const variants:Record<string,[string,Record<string,number>][]>={
   sweep:[...[0,1].map(c=>[`${c}`,{bColour:c}] as [string,Record<string,number>]),...[0,1].map(c=>[`S${c}`,{bColour:c,bSettle:1}] as [string,Record<string,number>])],
   // D: the descent's half sweeps after the restriction, black first. U: the ascent's, the prolonging red first.
   coarseSweep:[0,1].map(c=>[`${c}`,{bColour:c}] as [string,Record<string,number>]),
   middleSweep:[...[0,1].map(c=>[`${c}`,{bColour:c}] as [string,Record<string,number>]),["P",{bColour:0,bProlong:1}]],
   restrict:range(schedule.cycles).map(k=>[`@${k}`,{bCycle:k}]),measure:[["",{bCycles:schedule.cycles}],["Initial",{bInitial:1}]],
  };
  if(!this.spatialOrder)for(const key of ["listPrefix","listOffsets","listScatter"])delete sources[key];
  await Promise.all(Object.entries(sources).map(async([name,code])=>{
   // Each launch's own entry name: per-dispatch timing attributes it.
   const module=uniformDetailModule(this.device,{label:`Uniform pressure band ${name}`,code:code.replace(/fn main\(/,`fn ${entryOf(name)}(`)});
   const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");
   if(errors.length)throw new Error(`Pressure band ${name}: ${errors.map(m=>`${m.lineNum}: ${m.message}`).join("\n")}`);
   const pipelineLayout=this.device.createPipelineLayout({bindGroupLayouts:[this.simulation.bindLayout,this.layouts.get(this.layoutOf(name))!,...(this.solid?[this.solid.coarse!.bindLayout]:[])]});
   // The settle sweeps are dispatched only under solids (encode: solid.present): no solid-free twin.
   await Promise.all((variants[name]??[["",{}]]).map(async([suffix,constants])=>this.pipelines.set(name+suffix,await uniformMixedSolidPipeline(this.solid,s=>uniformDetailPipeline(this.device,this.simulation,{layout:pipelineLayout,
    compute:{module,entryPoint:entryOf(name),constants:{umDispatchX:this.simulation.dispatchX,...constants,...s}}}),{solidsOnly:constants.bSettle===1,entry:name+suffix}))));
  }));
  this.bindLaunches();
 }
 /** Pipeline and group 1 of each compiled launch name, resolved by
  * initialize() and again by allocate() (its groups are new): an encode
  * only looks them up. */
 private readonly launches=new Map<string,{pipeline:GPUComputePipeline;group:UniformDetailGroup}>();
 private bindLaunches():void{
  this.launches.clear();
  for(const [name,pipeline] of this.pipelines)this.launches.set(name,{pipeline,group:this.groups.get(this.layoutOf(name))!});
 }
 /** The pass the band's static groups were last bound in, and its group 1. */
 private boundPass?:GPUComputePassEncoder;
 private boundEpoch=-1;
 private boundGroup?:UniformDetailGroup;
 private layoutOf(name:string):string{const base=name.replace(/([01PDUF]|@\d+)$/,"");if(base==="rebase")return "init";if(base.startsWith("list"))return "list";return ["list","copy","prep","init","project"].includes(base)?base:"solver";}
 /** launch: a fixed group count, the simulation's h tiles (one group per
  * tile of its buffered evidence), or a slot-list stride (one 32-lane group
  * per slot colour, one group per slot, per 8 slots, per 64 slots) capped by
  * the buffered workload estimate and the saturation cap. */
 private dispatch(pass:GPUComputePassEncoder,name:string,launch:number|"tiles"|"cells"|"slots"|"middle"|"coarse",capacity=false):void{
  const bound=this.launches.get(name);if(!bound)throw new Error("Pressure band is not initialized");
  // Groups 0 and 2 are the same for every band launch: bind them once per
  // pass, and group 1 only when the launch's layout changes.
  if(this.boundPass!==pass||this.boundEpoch!==uniformComputePassEpoch(pass)){this.boundPass=pass;this.boundEpoch=uniformComputePassEpoch(pass);this.boundGroup=undefined;pass.setBindGroup(0,this.simulation.bindGroup);if(this.solid)pass.setBindGroup(2,this.solid.coarse!.bindGroup);}
  const pipeline=this.solid?.select(bound.pipeline)??bound.pipeline;
  if(this.boundGroup!==bound.group){this.boundGroup=bound.group;pass.setBindGroup(1,bound.group.group);}
  if(launch==="tiles"){this.simulation.dispatchBuffered(pass,pipeline,"fine",LIST_GROUPS);return;}
  pass.setPipeline(uniformDetailPick(pipeline));
  let groups=typeof launch==="number"?launch:launch==="cells"?this.cellGroups:launch==="slots"?this.slotGroups:launch==="middle"?this.middleGroups:this.coarseGroups;
  // A cells launch is 32 lanes a slot of CELL_BLOCK; the other strides are 64 lanes.
  if(capacity&&typeof launch!=="number")groups=Math.min(groups,Math.floor(BAND_CAPACITY_THREADS/(launch==="cells"?32*CELL_BLOCK:64)));
  pass.dispatchWorkgroups(groups);
 }
 /** `count` red-black sweeps: each colour's launch, colour 0 first. */
 private sweep(pass:GPUComputePassEncoder,colour0:string,colour1:string,launch:"cells"|"middle"|"coarse",count:number,capacity:boolean):void{
  for(let i=0;i<count;i++){this.dispatch(pass,colour0,launch,capacity);this.dispatch(pass,colour1,launch,capacity);}
 }
 /** The ownership holds no h tile (capacity 0, known to the host): the band
  * has no tile, and its launches (each a few microseconds empty) are not
  * encoded. The cleared header is the whole receipt. Likewise a quiet frame
  * under the freeze (UniformPressureBandFields.coldFrame): its frozen
  * regions list no tile. With coarse WARM treatment the same holds up to
  * WARM: frozen COLD tiles and coarse WARM tiles both fail the list predicate.
  * The cold-frame ceiling already handles host triggers, particle evidence
  * and lagged GPU promotions; an unknown ceiling must still run the band. */
 private get empty():boolean{
  if(this.simulation.capacity.fineTiles===0)return true;
  const ceiling=this.fields.coldFrame?.ceiling;
  return this.cold!==undefined&&ceiling!==undefined&&
   ceiling<(this.fields.warmTiers?UNIFORM_REGION_TIER.hot:UNIFORM_REGION_TIER.warm);
 }
 /** Before the split, with the simulation authority's phi and
  * the forced field in simulation ownership: list band tiles, assemble rows
  * and bake the aggregate operators. */
 encodePrepare(encoder:GPUCommandEncoder):void{
  encoder.clearBuffer(this.index,0,4*HEADER);encoder.clearBuffer(this.index,this.slotMapOffset);
  if(this.empty)return;
  const pass=encoder.beginComputePass(PREPARE_PASS);
  this.dispatch(pass,"list","tiles");
  if(this.spatialOrder){const groups=groupsOf(this.simulation.capacity.tiles,256);this.dispatch(pass,"listPrefix",groups);this.dispatch(pass,"listOffsets",1);this.dispatch(pass,"listScatter",groups);}
  this.dispatch(pass,"prep","slots");this.dispatch(pass,"middleBake","middle");this.dispatch(pass,"coarseBake","coarse");pass.end();
 }
 /** After the 4h projection reaches simulation ownership: start from the 4h
  * pressure, run the V-cycles, and project the band faces into the velocity
  * field. Direct launch widths follow lagged evidence; the GPU band count
  * always determines the complete work, including after sudden growth. */
 encodeSolve(encoder:GPUCommandEncoder):void{
  if(this.empty)return;
  const s=this.schedule;const pass=encoder.beginComputePass(SOLVE_PASS);
  const wide=this.workSlots>COARSE_WIDE_SLOTS;
  // The broad relayout reserve sizes parallel grids, not this choice. A
  // fused visit has its own complete strided fallback if the list grows.
  const fusedMiddle=this.fields.fusedMiddle!==false&&this.observedMiddleSlots<=this.middleFusedSlots;
  this.dispatch(pass,"init","slots");this.dispatch(pass,"rebase","slots");
  // Shut rows exist only beside a solid.
  if(this.solid?.present){this.dispatch(pass,"sweepS0","cells");this.dispatch(pass,"sweepS1","cells");}
  // Test the current global pressure predictor, including today's forces and
  // body fluxes. Projection still runs when no fine correction is necessary.
  this.dispatch(pass,"measureInitial","slots");
  // bandExit "returns": one cycle past the most the band ran in the last
  // BAND_CYCLE_WINDOW receipted frames is encoded. A frame that needs more
  // ends at its last encoded cycle and receipts all of them (its measure
  // finds no stop), so the next lagged frames encode every cycle again.
  const cycles=this.fields.returns?Math.min(s.cycles,this.wideCycles+1):s.cycles;
  for(let cycle=0;cycle<cycles;cycle++){
   const c=cycle>=this.wideCycles;
   this.sweep(pass,"sweep0","sweep1","cells",s.fineSweeps,c);
   this.dispatch(pass,this.restricts[cycle]!,"slots",c);
   if(fusedMiddle)this.dispatch(pass,"middleDownFused",1);
   else{this.dispatch(pass,"middleSweep1","middle",c);this.sweep(pass,"middleSweep0","middleSweep1","middle",s.middleSweeps-1,c);}
   this.dispatch(pass,"middleRestrict","middle",c);
   if(wide)this.sweep(pass,"coarseSweep0","coarseSweep1","coarse",s.coarseSweeps,c);else this.dispatch(pass,"coarseSolve",1);
   if(fusedMiddle)this.dispatch(pass,"middleUpFused",1);
   else{this.dispatch(pass,"middleSweepP","middle",c);this.dispatch(pass,"middleSweep1","middle",c);this.sweep(pass,"middleSweep0","middleSweep1","middle",s.middleSweeps-1,c);}
   this.dispatch(pass,"middleProlong","slots",c);this.sweep(pass,"sweep0","sweep1","cells",s.fineSweeps,c);
  }
  // The final measure does work only after a cycle ran.
  this.dispatch(pass,"measure","slots",s.cycles>this.wideCycles);
  pass.end();
  const project=encoder.beginComputePass(PROJECT_PASS);this.dispatch(project,"copy","slots");this.dispatch(project,"project","slots");project.end();
 }
 /** The eight header words, for the frame's receipt. */
 encodeReceipt(encoder:GPUCommandEncoder,target:GPUBuffer,offset:number):void{encoder.copyBufferToBuffer(this.index,0,target,offset,32);}
 /** Diagnostics: the residual history words (before each cycle, then after the last). */
 static readonly historyWord=HISTORY_WORD;
 static readonly initialResidualWord=INITIAL_WORD;
 static readonly residualValidityWord=VALID_WORD;
 static readonly initialResidualBit=1<<HISTORY;
 /** The header word the frame's verdict closes the band with. */
 static readonly closedWord=CLOSED_WORD;
 destroy():void{for(const b of [this.index,this.ownRows,this.coarse,this.solve])b?.destroy();}
}
