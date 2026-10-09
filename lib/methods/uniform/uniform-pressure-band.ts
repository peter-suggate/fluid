import { uniformDetailBindLayout, uniformDetailModule, uniformDetailPipeline, uniformDetailPick, uniformDetailGroup, type UniformDetailGroup } from "./uniform-detail-fields";
import { uniformBufferedWork } from "./uniform-buffered-work";
import type {UniformMixedOwnership} from "./uniform-mixed-ownership";
import {uniformMixedTopologyWGSL} from "./uniform-mixed-topology.wgsl";
import {uniformMixedFaceAddressWGSL} from "./uniform-mixed-face-dispatch.wgsl";
import {UNIFORM_MIXED_THETA_MIN,uniformMixedPressureLiquidWGSL,uniformMixedSurfaceThetaWGSL} from "./uniform-mixed-pressure-surface.wgsl";
import {uniformMixedSolidPipeline,uniformMixedSolidWGSL,type UniformMixedSolid} from "./uniform-mixed-solid.wgsl";

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
 /** Each aggregate level's inter-aggregate couplings (and the diagonal share
  * they carry) are scaled by this: piecewise-constant aggregation is twice too
  * stiff for smooth modes (0.5 would recover the rediscretised interior
  * operator). Dirichlet terms stay exact -- scaling them over-corrects small
  * Dirichlet-dominated blobs and stalls the cycle. 0.6 from replaying dumped
  * band systems: a wide shallow pool (hero garden) and a thick long-dam slab
  * converge ~0.2 per cycle at 0.6 but only ~0.5 at 0.5 (0.4 stalls); the
  * cm12 figure-12 surface band is ~0.1 at 0.5 and ~0.2 at 0.6. */
 coarseScale:number;
 /** A band of at most this many slots (the buffered estimate the launch
  * widths use) runs each level's consecutive half sweeps as one one-group
  * launch: the same kernels in the same order, a storage barrier where a
  * launch boundary was, 35 launches for 95. One group is serial over the
  * slots (128-class pool, band solve of 1 / 4 / 27 band tiles: 0.50 / 0.59 /
  * 1.65 ms fused against 0.88 / 1.03 / 0.99), so only a band of a few tiles
  * gains. 0 never fuses. */
 fusedSlots:number;
}
export const UNIFORM_PRESSURE_BAND_SCHEDULE:UniformPressureBandSchedule={cycles:4,fineSweeps:3,middleSweeps:2,coarseSweeps:16,coarseScale:0.6,fusedSlots:8};

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
/** Slots the one group of a fused h launch relaxes at once: 256 lanes, the
 * workgroup size every WebGPU device has. */
const FUSED_BLOCK=8;
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
/** Narrow forms of that solve, by lanes: a band of at most as many slots
 * (the buffered estimate) launches the narrowest that holds it, a position
 * to a lane. Every barrier of the one workgroup costs its lanes, whether or
 * not they hold a row, and the solve is `2 coarseSweeps` of them. Each
 * form is complete for any band (a larger one streams its rows). */
const COARSE_SMALL_LANES=[64,256];
/** A band of more slots than this (the buffered estimate) runs the 4h
 * level as a launch per half sweep across workgroups instead (coarseSweep):
 * the same red-black updates. One workgroup streams a large band's rows
 * through at most 1024 threads, 72 microseconds a thousand rows a solve
 * (pool 128^3 all-fine, 17 thousand rows: 1.2 ms of each solve's launch);
 * the `2 coarseSweeps` launches are 5 to 12 microseconds each on the GPU and
 * 7 on the host, so they only win well past the register path's reach. */
const COARSE_WIDE_SLOTS=8192;
/** Band rows are field-major over CAP*64 rows: rhs, diagonal, face kinds,
 * the six face coefficients, u* per face, with static solids the CM11a V
 * per face, then the row's start pressure (the iterate is the correction to
 * it). A slot's 64 rows (and iterate values) are colour-major: each
 * red-black half sweep reads 32 contiguous rows per field. */
const ROW_FIELDS=16,SOLID_ROW_FIELDS=22;
/** 2h aggregates: correction, residual, diagonal, six couplings. 4h
 * aggregates: correction and residual by slot, then by colour-ordered
 * position the diagonal and six couplings (2..8), six neighbour slots (+1)
 * by slot (9..14), then by position the slot (15), the residual the large
 * solve gathers per call (16) and the six neighbour slots (+1, 17..22). */
const MIDDLE_FIELDS=9,COARSE_FIELDS=23;
/** Index header words: count, overflow, final residual, fatal, completed
 * cycles, closed (the frame's verdict: nonzero when the pressure schedule's
 * last gate rejected the frame), converged (the convergence word), one spare,
 * then the residual after each cycle's pre-smoothing (and after the last
 * cycle) in HISTORY words, then coarseBake's red and black 4h row counts.
 * The tile list and per-tile slot+1 map follow. */
const HEADER=26,CLOSED_WORD=5,HISTORY_WORD=8,HISTORY=16,RED_WORD=24,BLACK_WORD=25;

export interface UniformPressureBandFields {
 /** Simulation-indexed pressure phi (the simulation authority's output),
  * read by the list and the rows before the split. */
 phi:GPUBufferBinding;
 /** The coarse pressure phi (the split authority's, in pressure indexing),
  * read after the 4h projection. Absent: the split rewrote `phi` itself. */
 coarsePhi?:GPUBufferBinding;
 correction:GPUTexture;
 vertexPhi:GPUTexture;
 /** The forced field u* in simulation ownership (before the split). */
 forced:{velocity:GPUTexture;negative:GPUBuffer};
 /** The 4h-projected field transferred to simulation ownership, and the
  * texture its copy lands in for the band projection's reads. */
 velocity:GPUTexture;negative:GPUBuffer;copy:GPUTexture;
 /** Accepted 4h pressure, indexed by the all-4h pressure ownership. */
 coarsePressure:GPUBufferBinding;
 /** h.xyz, dt; density, openTop, dt/density, min h; the residual target
  * (1/s, the h-equivalent pressure tolerance), then 1/h^2 per axis (f32):
  * a face whose coefficient is exactly that is flagged, and its sweeps skip
  * the coefficient load. Then four u32 the band writes itself: its slot
  * capacity, 64 and 8 times it, and a spare (64 bytes in all). */
 params:GPUBuffer;
 /** The stage grids' band section: the live band pressures land at this
  * word of the buffer (uniform-stage-grids). */
 presentation:{buffer:GPUBuffer;word:number};
 /** Band tiles one frame can hold (1..tiles). Absent: capacityOf at the
  * simulation ownership's h-tile capacity. */
 capacity?:number;
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
 get allocatedBytes():number{return this.index.size+this.rows.size+this.coarse.size+this.solve.size;}
 /** Header (see HEADER), then the tile list and a per-tile slot+1 map. It is
  * tile-scaled metadata (the list holds every tile, whatever the band's
  * capacity), so the buffer and its word offsets never change. */
 readonly index:GPUBuffer;
 private rows!:GPUBuffer;
 private coarse!:GPUBuffer;
 private solve!:GPUBuffer;
 private workSlots:number;
 /** Fixed-lag evidence, applied once per encoded frame by the frame owner.
  * A smaller budget never limits admission, storage or solver iterations. */
 observeWork(tiles:number,reserve=0):void{this.workSlots=uniformBufferedWork(this.workSlots,tiles,this.capacity,reserve);}
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
 private readonly layouts=new Map<string,GPUBindGroupLayout>();
 private readonly groups=new Map<string,UniformDetailGroup>();
 private readonly pipelines=new Map<string,GPUComputePipeline>();
 /** The h pressure iterate, the correction to each row's start (the rows' last field): 64 cells per band slot, then the wall halo. */
 get iterate():GPUBuffer{return this.solve;}
 /** Byte offset in index of the per-tile slot+1 map. */
 get slotMapOffset():number{return 4*(HEADER+this.listSlots);}
 /** Tile list words of index: every tile. */
 private readonly listSlots:number;
 private readonly rowFields:number;
 constructor(private readonly device:GPUDevice,private readonly simulation:UniformMixedOwnership,private readonly pressure:UniformMixedOwnership,private fields:UniformPressureBandFields,
  readonly schedule:UniformPressureBandSchedule=UNIFORM_PRESSURE_BAND_SCHEDULE,
  /** Static solids, with the all-4h record (its cut-tile flags). */
  private readonly solid?:UniformMixedSolid){
  const layout=simulation.layout,tiles=layout.tiles.length;
  if(pressure.layout.tiles.some(word=>(word&0xc0000000)!==0))throw new Error("The pressure band's global stage must be all-4h");
  if(solid&&!solid.coarse)throw new Error("The solid pressure band needs the all-4h solid record");
  if(schedule.cycles<1||schedule.cycles>=HISTORY)throw new Error(`Pressure band cycles must be 1..${HISTORY-1}`);
  if(schedule.middleSweeps<1)throw new Error("The pressure band's 2h level needs a sweep: its first red half sweep carries the restriction and the prolongation");
  this.rowFields=solid?SOLID_ROW_FIELDS:ROW_FIELDS;
  this.listSlots=tiles;
  this.index=device.createBuffer({label:"Uniform pressure band tiles",size:(HEADER+this.listSlots+tiles)*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC});
  const texture=(binding:number)=>({binding,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float" as const,viewDimension:"3d" as const}});
  const buffer=(binding:number,type:GPUBufferBindingType="storage")=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type}});
  const params=buffer(0,"uniform");
  // At most six storage buffers here: topology and solids hold four.
  const solver=[params,buffer(2),buffer(9,"read-only-storage"),buffer(10),buffer(18)];
  const entries:Record<string,GPUBindGroupLayoutEntry[]>={
   list:[params,buffer(2),buffer(3,"read-only-storage"),texture(6)],
   copy:[params,buffer(2),texture(22),{binding:23,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"write-only",format:"rgba32float",viewDimension:"3d"}}],
   prep:[params,buffer(2),buffer(3,"read-only-storage"),texture(4),texture(6),texture(7),buffer(8,"read-only-storage"),buffer(9),buffer(10)],
   init:[params,buffer(2),buffer(9),buffer(10),texture(11),buffer(12,"read-only-storage"),buffer(13,"read-only-storage"),buffer(14,"read-only-storage")],
   solver,
   project:[...solver,texture(15),{binding:16,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"write-only",format:"rgba32float",viewDimension:"3d"}},buffer(17)],
   present:[params,buffer(2),buffer(9,"read-only-storage"),buffer(10),buffer(21)],
  };
  for(const [name,list] of Object.entries(entries))this.layouts.set(name,uniformDetailBindLayout(device,{label:`Uniform pressure band ${name}`,entries:list}));
  this.allocate();
  this.workSlots=Math.max(1,Math.min(this.capacity,layout.fineTiles.length));
 }
 /** Bytes of the capacity-sized buffers at `capacity` band tiles. */
 bytesAt(capacity:number):{rows:number;aggregates:number;iterate:number}{
  const d=this.simulation.capacity.lattice.dimensions,halo=2*(d[0]!*d[1]!+d[0]!*d[2]!+d[1]!*d[2]!);
  return {rows:4*this.rowFields*64*capacity,aggregates:4*(8*MIDDLE_FIELDS+COARSE_FIELDS)*capacity,iterate:4*(64*capacity+halo)};
 }
 /** The capacity-sized buffers (rows, aggregates, iterate) and every group,
  * at fields.capacity. No band buffer carries a frame's state into the next
  * (each solve lists, assembles and starts anew), so a new capacity is a
  * reallocation and a rebind, no copy. */
 private allocate():void{
  const device=this.device,c=this.simulation.capacity,d=c.lattice.dimensions;
  const capacity=this.fields.capacity??UniformPressureBand.capacityOf(c.tiles,c.fineTiles);
  if(!Number.isInteger(capacity)||capacity<1||capacity>c.tiles)throw new Error(`The pressure band cannot hold ${capacity} of ${c.tiles} tiles`);
  const rows=capacity*64,bytes=this.bytesAt(capacity);
  const limit=device.limits.maxStorageBufferBindingSize;
  if(bytes.rows>limit)throw new Error(`Pressure band rows (${bytes.rows} bytes) exceed the storage binding limit ${limit}`);
  // The frames encoded on the old buffers are submitted: destroy applies after them.
  for(const b of [this.rows,this.coarse,this.solve])b?.destroy();
  this.slots=capacity;
  this.rows=device.createBuffer({label:"Uniform pressure band rows",size:bytes.rows,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
  this.coarse=device.createBuffer({label:"Uniform pressure band aggregates",size:bytes.aggregates,usage:GPUBufferUsage.STORAGE});
  this.solve=device.createBuffer({label:"Uniform pressure band iterate",size:bytes.iterate,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC});
  // BandParams.capacity: slots, rows per field, 2h aggregates per field.
  device.queue.writeBuffer(this.fields.params,48,new Uint32Array([capacity,rows,8*capacity,0]));
  const f=this.fields,cells=c.owners,faces=d[0]!*d[1]!+d[0]!*d[2]!+d[1]!*d[2]!;
  const scalars=(view:GPUBufferBinding,count:number):GPUBufferBinding=>{
   const offset=view.offset??0;if((view.size??view.buffer.size-offset)<4*count)throw new Error("Pressure band field view is too small");return {buffer:view.buffer,offset,size:4*count};
  };
  const pressure=this.pressure,coarse=pressure.capacity.tiles;
  const P={0:{buffer:f.params,size:64}},solverResources={...P,2:{buffer:this.index},9:{buffer:this.rows},10:{buffer:this.solve},18:{buffer:this.coarse}};
  const resources:Record<string,Record<number,GPUBindingResource>>={
   list:{...P,2:{buffer:this.index},3:scalars(f.phi,cells),6:f.vertexPhi},
   copy:{...P,2:{buffer:this.index},22:f.velocity,23:f.copy},
   prep:{...P,2:{buffer:this.index},3:scalars(f.phi,cells),4:f.correction,6:f.vertexPhi,7:f.forced.velocity,8:{buffer:f.forced.negative,size:4*faces},9:{buffer:this.rows},10:{buffer:this.solve}},
   init:{...P,2:{buffer:this.index},9:{buffer:this.rows},10:{buffer:this.solve},11:f.velocity,12:scalars(f.coarsePressure,coarse),13:scalars(f.coarsePhi??f.phi,coarse),14:pressure.presentation},
   solver:solverResources,
   project:{...solverResources,15:f.copy,16:f.velocity,17:{buffer:f.negative,size:4*faces}},
   present:{...P,2:{buffer:this.index},9:{buffer:this.rows},10:{buffer:this.solve},21:{buffer:f.presentation.buffer}},
  };
  if(f.presentation.buffer.size<4*(f.presentation.word+64*this.capacity))throw new Error("The stage grids cannot hold the pressure band");
  for(const [name,map] of Object.entries(resources))this.groups.set(name,uniformDetailGroup(device,{label:`Uniform pressure band ${name}`,layout:this.layouts.get(name)!,
   entries:Object.entries(map).map(([binding,resource])=>({binding:+binding,resource}))}));
  this.launches.clear();this.boundPass=undefined;this.boundGroup=undefined;
 }
 /** Between frames, after the simulation ownership's h-tile capacity (and
  * with it the views sized by it) or the band's own changed: reallocate
  * and rebind. */
 resize(fields:Pick<UniformPressureBandFields,"phi"|"presentation"|"capacity">):void{
  this.fields={...this.fields,...fields};this.allocate();
  this.workSlots=Math.max(1,Math.min(this.workSlots,this.capacity));
 }
 async initialize():Promise<void>{
  const layout=this.simulation.layout,S=!!this.solid,schedule=this.schedule;
  // The coarse solve's workgroup-memory corrections: all the device holds
  // beside its four scalars (Tint pads each to 16 bytes), never fewer than
  // the register path's, whose positions per lane that memory bounds.
  const limits=this.device.limits,lanes=Math.min(COARSE_SOLVE_LANES,limits.maxComputeInvocationsPerWorkgroup,limits.maxComputeWorkgroupSizeX);
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
const LIST:u32=${HEADER}u;const SLOTS:u32=${HEADER+this.listSlots}u;const HISTORY:u32=${HISTORY_WORD}u;const CYCLE:u32=4u;const DONE:u32=6u;
const K_GHOST:u32=0u;const K_BAND:u32=1u;const K_WALL:u32=2u;const K_OPEN:u32=3u;const K_NEUMANN:u32=4u;const K_CLOSED:u32=5u;
fn bLocal(lane:u32)->vec3u{return vec3u(lane%4u,(lane/4u)%4u,lane/16u);}
// A slot's row of local cell l: colour-major, then the half sweep's lane order.
fn bRow(l:vec3u)->u32{return ((l.x+l.y+l.z)&1u)*32u+(l.x>>1u)+2u*l.y+8u*l.z;}
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
// The native one-cell wall halo at h, after every band cell.
fn bHalo(p:vec3i,axis:u32,sign:i32)->u32{
 let q=vec3u(p);let side=select(0u,1u,sign>0);var slot=0u;
 if(axis==0u){slot=side*UM_D.y*UM_D.z+q.y+UM_D.y*q.z;}
 else if(axis==1u){slot=2u*UM_D.y*UM_D.z+side*UM_D.x*UM_D.z+q.x+UM_D.x*q.z;}
 else{slot=2u*(UM_D.y*UM_D.z+UM_D.x*UM_D.z)+side*UM_D.x*UM_D.y+q.x+UM_D.x*q.y;}
 return bN()+slot;
}
`;
  const indexed=(atomic:boolean)=>/* wgsl */`
@group(1) @binding(0) var<uniform> params:BandParams;
@group(1) @binding(2) var<storage,read_write> index:array<${atomic?"atomic<u32>":"u32"}>;
fn bIndex(i:u32)->u32{return ${atomic?"atomicLoad(&index[i])":"index[i]"};}
// A closed band (the verdict rejected the frame) strides no slots: its solve,
// projection and presentation do nothing. Row assembly precedes the verdict.
fn bCount()->u32{return select(min(bIndex(0u),bCap()),0u,bIndex(${CLOSED_WORD}u)!=0u);}
// Convergence: a cycle's restriction measured a residual at or below the
// target. Each history word is 0 (cleared with the header) until its cycle's
// restriction has run, and final once it has.
fn bMet(r:u32)->bool{return r!=0u&&bitcast<f32>(r)<=params.solve.x;}
fn bConvergedBefore(k:u32)->bool{for(var j=0u;j<min(k,${HISTORY-1}u);j++){if(bMet(bIndex(HISTORY+j))){return true;}}return false;}
fn bConverged()->bool{return bConvergedBefore(${schedule.cycles}u);}
// Converged launches stride zero slots: the same launches, no work.
fn bLive()->u32{return select(bCount(),0u,bConverged());}
fn bTile(s:u32)->vec3u{return umTileCoord(bIndex(LIST+s));}
// Slot + 1 of the band tile at tile coordinate c, 0 outside the band.
fn bSlotAt(c:vec3i)->u32{if(any(c<vec3i(0))||any(c>=vec3i(UM_T))){return 0u;}return bIndex(SLOTS+umTileAt(vec3u(c)));}
// Band cell index + 1 at an h position, 0 outside the band.
fn bCellAt(q:vec3i)->u32{
 if(any(q<vec3i(0))||any(q>=vec3i(UM_D))){return 0u;}
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
`;
  // The h solve uses the simulation's actual interface. The 4h solve supplies
  // boundary flux and an initial pressure, not a veto on h air: relabelling
  // a sub-cell pocket as liquid removes its free surface from pressure while
  // advection and rendering keep it, allowing a false cavity to persist.
  const owner=wet+/* wgsl */`
@group(1) @binding(3) var<storage,read> phi:array<f32>;
@group(1) @binding(6) var vertexPhi:texture_3d<f32>;
fn bCentrePhi(q:vec3i)->f32{
 var s=0.0;for(var k=0u;k<8u;k++){s+=textureLoad(vertexPhi,q+vec3i(vec3u(k&1u,(k>>1u)&1u,k>>2u)),0).x;}return 0.125*s;
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
// flux the transferred 4h face. The 4h solve owns that flux whatever the two
// 4h phases are: projected (liquid beside liquid, or ghost fluid beside
// air), kept (beside detached mass) or zero (the supported layer of two air
// owners, which the extension moves with the liquid under it). An air h
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
fn bForced(c:u32,f:u32)->f32{return rows[(9u+f)*bN()+c];}
${S?"fn bVolume(c:u32,f:u32)->f32{return rows[(15u+f)*bN()+c];}":""}
// The start pressure of a row: the iterate holds the correction to it.
const B_BASE:u32=${S?21:15}u;
fn bBase(c:u32)->f32{return rows[B_BASE*bN()+c];}
`,rows=rowsOf(false);
  const solve=/* wgsl */`
@group(1) @binding(10) var<storage,read_write> solve:array<f32>;
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
 for(var f=0u;f<6u;f++){
  if(bFaceKind(kinds,f)!=K_WALL){continue;}
  let axis=f/2u;let sign=bSign(f);
  solve[bHalo(p,axis,sign)]=max(value+f32(sign)*params.policy.x*bForced(cell,f)*params.hDt[axis]/params.hDt.w,-bBase(cell));
 }
}
// A liquid row's residual; a p_min=0 row at its bound only counts a positive one.
fn bResidual(cell:u32,p:vec3i,kinds:u32)->f32{
 let r=bRhs(cell)+bOff(cell,p,kinds)-bDiagonal(cell)*solve[cell];
 return select(r,max(r,0.0),bInside(kinds)&&solve[cell]+bBase(cell)<=0.0);
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
// bOff and bResidual with the neighbour slots from bNear.
fn bOffNear(cell:u32,s:u32,l:vec3u,p:vec3i,kinds:u32)->f32{
 var off=0.0;
 for(var f=0u;f<6u;f++){
  let axis=f/2u;let sign=bSign(f);let kind=bFaceKind(kinds,f);
  if(kind==K_BAND){var w=params.solve[1u+axis];if(!bUnit(kinds,f)){w=bCoefficient(cell,f);}off+=w*solve[bCellNear(s,l,f)-1u];}
  else if(kind==K_WALL){off+=bCoefficient(cell,f)*solve[bHalo(p,axis,sign)];}
 }
 return off;
}
fn bResidualNear(cell:u32,s:u32,l:vec3u,p:vec3i,kinds:u32)->f32{
 let r=bRhs(cell)+bOffNear(cell,s,l,p,kinds)-bDiagonal(cell)*solve[cell];
 return select(r,max(r,0.0),bInside(kinds)&&solve[cell]+bBase(cell)<=0.0);
}
`;
  const aggregates=/* wgsl */`
@group(1) @binding(18) var<storage,read_write> coarse:array<f32>;
// 2h aggregate a (0..8) of slot s is cell 8s+a; field k at k*M. The per-tile
// 4h aggregate of slot s: field k at 9M+k*CAP+s (9..14 neighbour slot+1, 15 colour).
fn bM(k:u32,c:u32)->u32{return k*params.capacity.z+c;}
fn bC(k:u32,s:u32)->u32{return 9u*params.capacity.z+k*bCap()+s;}
// The 2h aggregate + 1 across face f of aggregate c, 0 outside the band.
fn bMiddleNeighbour(c:u32,f:u32)->u32{
 let s=c/8u;let axis=f/2u;var g=vec3i(bTile(s)*2u+bMiddleLocal(c%8u));g[axis]+=bSign(f);
 if(g[axis]<0||g[axis]>=2*i32(UM_T[axis])){return 0u;}
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
  const coarseForms:[string,number,number,number][]=[["coarseSolve",lanes,held,shared],
   ...COARSE_SMALL_LANES.filter(n=>n<lanes).map(n=>[`coarseSolve${n}`,n,1,n] as [string,number,number,number])];
  const sources:Record<string,string>={
   list:header+surface+/* wgsl */`
@group(1) @binding(0) var<uniform> params:BandParams;
@group(1) @binding(2) var<storage,read_write> index:array<atomic<u32>>;
var<workgroup> member:atomic<u32>;
// Every h simulation tile with a liquid pressure row is a band tile: its
// h velocity is divergence-free only after the h solve.
@compute @workgroup_size(64) fn main(${slots}){
 for(var j=group.x;j<umCounts.x;j+=groups.x){
  let t=bHTile(j);
  if(lane==0u){atomicStore(&member,0u);}
  workgroupBarrier();
  if(bWet(bPhiH(vec3i(umTileCoord(t)*4u+bLocal(lane))),1.0)){atomicStore(&member,1u);}
  if(workgroupUniformLoad(&member)!=0u&&lane==0u){
   let slot=atomicAdd(&index[0],1u);
   if(slot<bCap()){atomicStore(&index[LIST+slot],t);atomicStore(&index[SLOTS+t],slot+1u);}else{atomicStore(&index[1],1u);}
  }
 }
}`,
   prep:header+indexed(false)+theta+surface+/* wgsl */`
@group(1) @binding(9) var<storage,read_write> rows:array<f32>;
// The iterate holds each row's phi until init starts it.
@group(1) @binding(10) var<storage,read_write> solve:array<f32>;
@group(1) @binding(4) var correction:texture_3d<f32>;
@group(1) @binding(7) var velocity:texture_3d<f32>;
@group(1) @binding(8) var<storage,read> negative:array<f32>;
fn bForcedField(p:vec3i,axis:u32,sign:i32)->f32{
 var a=p;if(sign<0){a[axis]-=1;}
 if(a[axis]<0){return negative[umNegativeBoundaryIndex(vec3u(p),axis)];}
 return textureLoad(velocity,a,0)[axis];
}
// Row assembly, baked once per solve.
@compute @workgroup_size(64) fn main(${slots}){
 let N=bN();
 for(var s=group.x;s<bCount();s+=groups.x){
  let p=vec3i(bTile(s)*4u+bLocal(lane));let cell=s*64u+bRow(bLocal(lane));
  let own=bPhiH(p);let liquid=bWet(own,1.0);${S?"let shut=umCellOpen(p)<=1e-5;":""}
  var kinds=select(0u,0x80000000u,liquid)${S?"|select(0u,0x40000000u,umCellInsideSolid(p))|select(0u,0x20000000u,shut)":""};var diagonal=0.0;var divergence=0.0;
  ${S?`// Native divergenceAtWithCapacity: V u + (V_i - V) u_s, u_s the moving wall's.
  let bodies=liquid&&umBodyCount()>0u;let capacity=select(0.0,umCellOpen(p),bodies);`:""}
  for(var f=0u;f<6u;f++){
   let axis=f/2u;let sign=bSign(f);let h=params.hDt[axis];var q=p;q[axis]+=sign;
   let forced=bForcedField(p,axis,sign);var kind=K_GHOST;var coefficient=0.0;var fraction=1.0;
   ${S?"var low=p;if(sign<0){low[axis]-=1;}let volume=umPressureFaceV(low,axis);var sealed=false;":""}
   if(q[axis]<0||q[axis]>=i32(UM_D[axis])){
    if(axis==1u&&sign>0&&params.policy.y>0.5){kind=K_OPEN;coefficient=1.0/(h*h*bTheta(own,0.5*params.policy.w));}
    else{kind=K_WALL;coefficient=0.5/(h*h);fraction=0.5;}
   }else{
    let n=bNeighbour(q);
    if(bCellAt(q)!=0u){kind=K_BAND;coefficient=1.0/(h*h*bPairTheta(own,liquid,n.phi,n.wet));}
    else if(n.neumann){kind=K_NEUMANN;}
    else{coefficient=1.0/(h*h*bPairTheta(own,liquid,n.phi,n.wet));}
    ${S?`// A shut row is liquid by the phi its open neighbours continue into it: it
    // has no surface of its own. Its face to an air row is closed. Left as a
    // free-surface face, a shoreline cell under a slope took flux from the
    // air above it at a theta its own column's phi never changed: nothing
    // restored it, and the column's surface sank a micron in 20 steps.
    if(kind!=K_NEUMANN){sealed=(liquid&&shut&&!n.wet)||(!liquid&&n.wet&&umCellOpen(q)<=1e-5);}`:""}
   }
   ${S?`// CM11a: V scales the coefficient and the flux. A Neumann face's flux is
   // V times the transferred 4h face (init), the root's own flux through it:
   // the record V of a 4h face is the mean of its sixteen h V.
   if(volume<=1e-6||sealed){kind=K_CLOSED;coefficient=0.0;fraction=0.0;}
   else if(kind==K_NEUMANN){fraction=volume;}
   else{if(kind==K_WALL){coefficient=volume/(h*h);}else{coefficient*=volume;}fraction=volume;}
   rows[(15u+f)*N+cell]=volume;`:""}
   kinds|=(kind<<(3u*f))|select(0u,1u<<(18u+f),kind==K_BAND&&coefficient==params.solve[1u+axis]);rows[(9u+f)*N+cell]=forced;rows[(3u+f)*N+cell]=coefficient;
   if(kind!=K_NEUMANN){diagonal+=coefficient;divergence+=f32(sign)*fraction*forced/h;}
   ${S?"if(bodies&&kind!=K_WALL&&kind!=K_OPEN){divergence+=f32(sign)*(capacity-fraction)*umSolidFaceVelocity(low,axis)/h;}":""}
  }
  rows[cell]=select(0.0,-params.policy.x*(divergence-textureLoad(correction,p,0).x)/params.hDt.w,liquid);
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
   init:header+indexed(false)+rowsOf(true)+solve+aggregates+wet+/* wgsl */`
@group(1) @binding(11) var velocity:texture_3d<f32>;
@group(1) @binding(12) var<storage,read> coarsePressure:array<f32>;
@group(1) @binding(13) var<storage,read> coarsePhi:array<f32>;
@group(1) @binding(14) var<storage,read> coarseTopology:array<u32>;
// (pressure, phi) of a liquid 4h owner, weight 1; zero for an air one and
// for a closed one (its phi continues its liquid neighbours; it has no row).
fn bCoarse(c:vec3i)->vec3f{
 let t=umTileAt(vec3u(clamp(c,vec3i(0),vec3i(UM_T)-vec3i(1))));let i=coarseTopology[t]&0x3fffffffu;
 return select(vec3f(0.0),vec3f(coarsePressure[i],coarsePhi[i],1.0),bWet(coarsePhi[i],4.0)&&umTileOpen(t)>1e-5);
}
// A 4h owner as (pressure, liquid, air): bCoarse's liquid owner, and an open
// one that is not liquid; a closed one is neither.
fn bCoarseKind(c:vec3i)->vec3f{
 let t=umTileAt(vec3u(clamp(c,vec3i(0),vec3i(UM_T)-vec3i(1))));let i=coarseTopology[t]&0x3fffffffu;
 let open=umTileOpen(t)>1e-5;let wet=bWet(coarsePhi[i],4.0);
 return vec3f(select(0.0,coarsePressure[i],open&&wet),f32(open&&wet),f32(open&&!wet));
}
// What a row missing owners from its stencil b, b+1 (weights w) is carried
// by. rise: the 4h pressure's vertical difference per 4h spacing and its
// weight. Each of the four columns gives its two owners' difference where
// both are liquid, and where one is (the other closed, or the same owner
// past the lattice) that owner's against the liquid owner beyond it, one
// more load; a column with neither gives none; columns weigh as their
// liquid owners do in the mean. height: the liquid owners' weighted height
// above b and their weight. air: the air owners' weight.
struct BCarry {rise:vec2f,height:vec2f,air:f32}
fn bCoarseCarry(b:vec3i,w:vec3f)->BCarry{
 var carry:BCarry;let top=i32(UM_T.y)-1;
 for(var k=0u;k<4u;k++){
  let bit=vec2u(k&1u,k>>1u);let sides=select(1.0-w.xz,w.xz,bit==vec2u(1u));let side=sides.x*sides.y;
  let c=b+vec3i(i32(bit.x),0,i32(bit.y));let y0=clamp(c.y,0,top);let y1=clamp(c.y+1,0,top);
  let low=bCoarseKind(vec3i(c.x,y0,c.z));var high=low;if(y1!=y0){high=bCoarseKind(vec3i(c.x,y1,c.z));}
  let lower=side*(1.0-w.y);let upper=side*w.y;let liquid=lower*low.y+upper*high.y;
  carry.air+=lower*low.z+upper*high.z;
  carry.height+=vec2f(lower*low.y*f32(y0-b.y)+upper*high.y*f32(y1-b.y),liquid);
  if(y1!=y0&&low.y>0.0&&high.y>0.0){carry.rise+=liquid*vec2f(high.x-low.x,1.0);continue;}
  var own=low;var at=y0;var beyond=select(y0-1,y0+1,y0==0);
  if(y1!=y0){if(high.y>0.0){own=high;at=y1;beyond=y1+1;}else{beyond=y0-1;}}
  if(own.y==0.0||beyond<0||beyond>top){continue;}
  let other=bCoarseKind(vec3i(c.x,beyond,c.z));
  if(other.y>0.0){carry.rise+=liquid*vec2f((other.x-own.x)*f32(beyond-at),1.0);}
 }
 return carry;
}
@compute @workgroup_size(64) fn main(${slots}){
 for(var s=group.x;s<bCount();s+=groups.x){
  let p=vec3i(bTile(s)*4u+bLocal(lane));let cell=s*64u+bRow(bLocal(lane));
  let kinds=bKinds(cell);
  if(!bLiquid(kinds)){solve[cell]=0.0;rows[B_BASE*bN()+cell]=0.0;continue;}
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
  for(var k=0u;k<8u;k++){
   let bit=vec3u(k&1u,(k>>1u)&1u,k>>2u);let weights=select(1.0-w,w,bit==vec3u(1u));
   sum+=weights.x*weights.y*weights.z*bCoarse(b+vec3i(bit));
  }
  // prep left the row's phi in the iterate.
  var start=0.0;
  if(sum.z>=0.99999){start=sum.x/sum.z;}
  else if(sum.z>0.0){
   let carry=bCoarseCarry(b,w);
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
   if(bFaceKind(kinds,f)==K_NEUMANN){var a=p;if(sign<0){a[axis]-=1;}rhs-=params.policy.x*f32(sign)*${S?"bVolume(cell,f)*":""}textureLoad(velocity,a,0)[axis]/(params.hDt[axis]*params.hDt.w);}
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
   sweep:sweep(CELL_BLOCK),sweepWide:sweep(FUSED_BLOCK),
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
 var largest=0.0;
 for(var s=group.x;s<n;s+=groups.x){
  workgroupBarrier();
  let l=bLocal(lane);let cell=s*64u+bRow(l);let kinds=bKinds(cell);var r=0.0;
  if(bLiquid(kinds)&&bDiagonal(cell)>0.0){r=bResidualNear(cell,s,l,vec3i(bTile(s)*4u+l),kinds);largest=max(largest,abs(r));}
  bCellResidual[lane]=r;workgroupBarrier();
  if(lane<8u){
   let a=bMiddleLocal(lane);var sum=0.0;
   for(var k=0u;k<8u;k++){let q=a*2u+bMiddleLocal(k);sum+=bCellResidual[q.x+4u*(q.y+4u*q.z)];}
   let c=8u*s+lane;let g=bTile(s)*2u+a;var e=0.0;
   if(((g.x+g.y+g.z)&1u)==0u){let diagonal=coarse[bM(2u,c)];if(diagonal>0.0){e=(sum+0.0)/diagonal;}}
   coarse[bM(1u,c)]=sum;coarse[bM(0u,c)]=e;
  }
 }
 atomicMax(&worst,bitcast<u32>(largest*params.policy.z));
 workgroupBarrier();
 if(lane==0u){atomicMax(&index[HISTORY+bCycle],atomicLoad(&worst));}
}`,
   // With bProlong (red only), the 4h correction is prolonged on the fly: a
   // red update never reads its own value, and black values are prolonged
   // before any black update reads them.
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
     bCorrection[slot[k]]=(residual[k]+off)/diagonal[k];
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
    let next=(coarse[bC(16u,j)]+off)/coarse[bC(2u,j)];let s=bitcast<u32>(coarse[bC(15u,j)]);
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
  coarse[bC(0u,s)]=(coarse[bC(1u,s)]+off)/coarse[bC(2u,j)];
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
   // After bCycles encoded cycles.
   measure:header+indexed(true)+rows+solve+aggregates+/* wgsl */`
override bCycles:u32=1u;
var<workgroup> worst:atomic<u32>;
var<workgroup> bMeasureCount:u32;
// The largest band row residual, in divergence units (1/s), over liquid rows,
// into history word bCycles. A converged solve keeps its restriction's
// residual (nothing moved since): the receipt takes that cycle and sets the
// convergence word.
@compute @workgroup_size(64) fn main(${slots}){
 if(lane==0u){atomicStore(&worst,0u);bMeasureCount=select(bCount(),0u,bConvergedBefore(bCycles));}
 let n=workgroupUniformLoad(&bMeasureCount);
 var largest=0.0;
 for(var s=group.x;s<n;s+=groups.x){
  let l=bLocal(lane);let p=vec3i(bTile(s)*4u+l);let cell=s*64u+bRow(l);
  let kinds=bKinds(cell);if(bLiquid(kinds)&&bDiagonal(cell)>0.0){largest=max(largest,abs(bResidualNear(cell,s,l,p,kinds))*params.policy.z);}
 }
 atomicMax(&worst,bitcast<u32>(largest));
 workgroupBarrier();
 if(lane==0u){let r=atomicLoad(&worst);atomicMax(&index[2],r);atomicMax(&index[HISTORY+bCycles],r);}
 if(group.x==0u&&lane==0u){
  var cycle=bCycles;for(var j=bCycles;j>0u;j--){if(bMet(bIndex(HISTORY+j-1u))){cycle=j-1u;}}
  atomicStore(&index[CYCLE],cycle);
  if(cycle<bCycles){atomicStore(&index[DONE],1u);atomicMax(&index[2],bIndex(HISTORY+cycle));}
 }
}`,
   project:band+/* wgsl */`
@group(1) @binding(15) var field:texture_3d<f32>;
@group(1) @binding(16) var output:texture_storage_3d<rgba32float,write>;
@group(1) @binding(17) var<storage,read_write> boundary:array<f32>;
// umProject on one face of band cell 'cell' at p; 'kept' is the transferred
// 4h value, which a Neumann face keeps.
fn bFace(cell:u32,p:vec3i,f:u32,kept:f32)->f32{
 let kinds=bKinds(cell);let kind=bFaceKind(kinds,f);
 if(kind==K_NEUMANN){return kept;}
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
  var q=p;q[axis]+=1;let own=umCellOpen(p)>1e-5;
  if(own==(umCellOpen(q)>1e-5)||volume<=1e-6){return false;}
  var pressure=0.0;
  if(q[axis]>=i32(UM_D[axis])){pressure=select(0.0,solve[bHalo(p,axis,sign)]+bBase(cell),bLiquid(bKinds(cell))&&!(axis==1u&&params.policy.y>0.5));}
  else if(own){let b=bCellAt(q);if(b!=0u&&bLiquid(bKinds(b-1u))){pressure=solve[b-1u]+bBase(b-1u);}}
  else if(bLiquid(bKinds(cell))){pressure=solve[cell]+bBase(cell);}
  // Separation is relative to the wall's own motion (native wall[axis]).
  return pressure<=0.0&&select(1.0,-1.0,own)*(value-umSolidFaceVelocity(p,axis))*params.hDt.w>1e-4*params.hDt[axis];
 }
 if(umCellOpen(p)<=1e-5||volume<=1e-6){return false;}`:""}
 if(bFaceKind(bKinds(cell),f)!=K_WALL){return false;}
 let pressure=select(0.0,solve[bHalo(p,axis,sign)]+bBase(cell),bLiquid(bKinds(cell)));
 return pressure<=0.0&&-f32(sign)*value*params.hDt.w>1e-4*params.hDt[axis];
}
@compute @workgroup_size(128) fn main(${slots}){
 for(var s=group.x;s<bCount();s+=groups.x){
  let origin=vec3i(bTile(s)*4u);
  if(lane<64u){
   // A band cell's own texel: its three positive faces and wall releases.
   let p=origin+vec3i(bLocal(lane));let cell=s*64u+bRow(bLocal(lane));var texel=textureLoad(field,p,0);var released=0u;
   for(var axis=0u;axis<3u;axis++){
    let value=bFace(cell,p,2u*axis+1u,texel[axis]);texel[axis]=value;
    if(bReleased(cell,p,2u*axis+1u,value)){released|=1u<<axis;}
    if(p[axis]==0){
     let low=bFace(cell,p,2u*axis,0.0);boundary[umNegativeBoundaryIndex(vec3u(p),axis)]=low;
     if(bReleased(cell,p,2u*axis,low)){released|=1u<<(axis+3u);}
    }
   }
   texel.w=f32(released);textureStore(output,p,texel);
  }else if(lane<112u){
   // A non-band texel below this tile holds band faces in its positive
   // components. The first band tile along x, then y, then z writes it.
   let axis=(lane-64u)/16u;let k=(lane-64u)%16u;var l=vec3i(0);
   l[axis]=-1;l[(axis+1u)%3u]=i32(k%4u);l[(axis+2u)%3u]=i32(k/4u);let q=origin+l;
   if(q[axis]<0||bCellAt(q)!=0u){continue;}
   var first=true;
   for(var a=0u;a<axis;a++){var r=q;r[a]+=1;if(bCellAt(r)!=0u){first=false;}}
   if(!first){continue;}
   var texel=textureLoad(field,q,0);
   for(var a=0u;a<3u;a++){var r=q;r[a]+=1;let b=bCellAt(r);if(b!=0u){texel[a]=bFace(b-1u,r,2u*a,texel[a]);}}
   textureStore(output,q,texel);
  }
 }
}`,
   // The band projection's reads: each band tile's texels and the texel
   // layer below it, instead of the whole velocity field.
   copy:header+indexed(false)+/* wgsl */`
@group(1) @binding(22) var velocity:texture_3d<f32>;
@group(1) @binding(23) var copied:texture_storage_3d<rgba32float,write>;
@compute @workgroup_size(128) fn main(${slots}){
 for(var s=group.x;s<bCount();s+=groups.x){
  let origin=vec3i(bTile(s)*4u);var q=origin+vec3i(bLocal(lane));
  if(lane>=64u){
   if(lane>=112u){continue;}
   let axis=(lane-64u)/16u;let k=(lane-64u)%16u;var l=vec3i(0);
   l[axis]=-1;l[(axis+1u)%3u]=i32(k%4u);l[(axis+2u)%3u]=i32(k/4u);q=origin+l;
   if(q[axis]<0){continue;}
  }
  textureStore(copied,q,textureLoad(velocity,q,0));
 }
}`,
   present:header+indexed(false)+rows+/* wgsl */`
@group(1) @binding(10) var<storage,read_write> solve:array<f32>;
@group(1) @binding(21) var<storage,read_write> presented:array<f32>;
// The live band pressures (start plus correction) into the stage grids' band section.
@compute @workgroup_size(64) fn main(${slots}){
 for(var s=group.x;s<bCount();s+=groups.x){let cell=s*64u+bRow(bLocal(lane));presented[${this.fields.presentation.word}u+s*64u+lane]=bBase(cell)+solve[cell];}
}`,
  };
  const entryOf=(name:string)=>`band${name[0]!.toUpperCase()}${name.slice(1)}`;
  const range=(n:number,from=0)=>Array.from({length:n},(_,i)=>i+from);
  // Specialised launches: red-black colours, each cycle's restriction, the
  // measure after each encodable cycle count.
  const variants:Record<string,[string,Record<string,number>][]>={
   sweep:[...[0,1].map(c=>[`${c}`,{bColour:c}] as [string,Record<string,number>]),...[0,1].map(c=>[`S${c}`,{bColour:c,bSettle:1}] as [string,Record<string,number>])],
   // F: a cycle's h half sweeps before or after the correction, in one launch. SF: the two settling ones.
   sweepWide:[["F",{bColour:0,bHalves:2*schedule.fineSweeps}],["SF",{bColour:0,bSettle:1,bHalves:2}]],
   // D: the descent's half sweeps after the restriction, black first. U: the ascent's, the prolonging red first.
   coarseSweep:[0,1].map(c=>[`${c}`,{bColour:c}] as [string,Record<string,number>]),
   middleSweep:[...[0,1].map(c=>[`${c}`,{bColour:c}] as [string,Record<string,number>]),["P",{bColour:0,bProlong:1}],["D",{bColour:1,bHalves:2*schedule.middleSweeps-1}],["U",{bColour:0,bProlong:1,bHalves:2*schedule.middleSweeps}]],
   restrict:range(schedule.cycles).map(k=>[`@${k}`,{bCycle:k}]),measure:range(schedule.cycles,1).map(k=>[`@${k}`,{bCycles:k}]),
  };
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
 }
 /** Pipeline and group 1 of each launch name, resolved on first use. */
 private readonly launches=new Map<string,{pipeline:GPUComputePipeline;group:UniformDetailGroup}>();
 /** The pass the band's static groups were last bound in, and its group 1. */
 private boundPass?:GPUComputePassEncoder;
 private boundGroup?:UniformDetailGroup;
 private layoutOf(name:string):string{const base=name.replace(/([01PDUF]|@\d+)$/,"");if(base==="rebase")return "init";return ["list","copy","prep","init","project","present"].includes(base)?base:"solver";}
 /** launch: a fixed group count, the simulation's h tiles (one group per
  * tile of its buffered evidence), or a slot-list stride (one 32-lane group
  * per slot colour, one group per slot, per 8 slots, per 64 slots) capped by
  * the buffered workload estimate and the saturation cap. */
 private dispatch(pass:GPUComputePassEncoder,name:string,launch:number|"tiles"|"cells"|"slots"|"middle"|"coarse"):void{
  let bound=this.launches.get(name);
  if(!bound){const pipeline=this.pipelines.get(name);if(!pipeline)throw new Error("Pressure band is not initialized");
   bound={pipeline,group:this.groups.get(this.layoutOf(name))!};this.launches.set(name,bound);}
  // Groups 0 and 2 are the same for every band launch: bind them once per
  // pass, and group 1 only when the launch's layout changes.
  if(this.boundPass!==pass){this.boundPass=pass;this.boundGroup=undefined;pass.setBindGroup(0,this.simulation.bindGroup);if(this.solid)pass.setBindGroup(2,this.solid.coarse!.bindGroup);}
  const pipeline=this.solid?.select(bound.pipeline)??bound.pipeline;
  if(this.boundGroup!==bound.group){this.boundGroup=bound.group;pass.setBindGroup(1,bound.group.group);}
  if(launch==="tiles"){this.simulation.dispatchBuffered(pass,pipeline,"fine",LIST_GROUPS);return;}
  pass.setPipeline(uniformDetailPick(pipeline));
  const c=this.workSlots;
  pass.dispatchWorkgroups(typeof launch==="number"?launch:launch==="cells"?Math.min(Math.ceil(c/CELL_BLOCK),CELL_GROUPS):launch==="slots"?Math.min(c,SLOT_GROUPS):launch==="middle"?Math.min(Math.ceil(c/8),MIDDLE_GROUPS):Math.min(Math.ceil(c/64),COARSE_GROUPS));
 }
 private sweep(pass:GPUComputePassEncoder,name:string,launch:"cells"|"middle"|"coarse",count:number):void{
  for(let i=0;i<count;i++){this.dispatch(pass,`${name}0`,launch);this.dispatch(pass,`${name}1`,launch);}
 }
 /** The ownership holds no h tile (capacity 0, known to the host): the band
  * has no tile, and its launches (each a few microseconds empty) are not
  * encoded. The cleared header is the whole receipt. */
 private get empty():boolean{return this.simulation.capacity.fineTiles===0;}
 /** Before the split, with the simulation authority's phi and correction and
  * the forced field in simulation ownership: list band tiles, assemble rows
  * and bake the aggregate operators. */
 encodePrepare(encoder:GPUCommandEncoder):void{
  encoder.clearBuffer(this.index,0,4*HEADER);encoder.clearBuffer(this.index,this.slotMapOffset);
  if(this.empty)return;
  const pass=encoder.beginComputePass({label:"Uniform pressure band list and rows"});
  this.dispatch(pass,"list","tiles");
  this.dispatch(pass,"prep","slots");this.dispatch(pass,"middleBake","middle");this.dispatch(pass,"coarseBake","coarse");pass.end();
 }
 /** After the 4h projection reaches simulation ownership: start from the 4h
  * pressure, run the V-cycles, and project the band faces into the velocity
  * field. Direct launch widths follow lagged evidence; the GPU band count
  * always determines the complete work, including after sudden growth. */
 encodeSolve(encoder:GPUCommandEncoder):void{
  if(this.empty)return;
  const s=this.schedule;const pass=encoder.beginComputePass({label:"Uniform pressure band solve"});
  // A small band's launches are a handful of workgroups each: one group
  // then runs a level's consecutive half sweeps in one launch.
  const fused=this.workSlots<=s.fusedSlots;
  const fine=()=>{if(fused)this.dispatch(pass,"sweepWideF",1);else this.sweep(pass,"sweep","cells",s.fineSweeps);};
  // The 4h level: the narrowest one-workgroup form that holds the band, or
  // a large band's launches.
  const wide=this.workSlots>COARSE_WIDE_SLOTS;
  const narrow=COARSE_SMALL_LANES.find(n=>this.workSlots<=n&&this.pipelines.has(`coarseSolve${n}`)),coarse=narrow?`coarseSolve${narrow}`:"coarseSolve";
  this.dispatch(pass,"init","slots");this.dispatch(pass,"rebase","slots");
  // Shut rows exist only beside a solid.
  if(this.solid?.present){if(fused)this.dispatch(pass,"sweepWideSF",1);else{this.dispatch(pass,"sweepS0","cells");this.dispatch(pass,"sweepS1","cells");}}
  for(let cycle=0;cycle<s.cycles;cycle++){
   fine();
   this.dispatch(pass,`restrict@${cycle}`,"slots");
   if(fused)this.dispatch(pass,"middleSweepD",1);else{this.dispatch(pass,"middleSweep1","middle");this.sweep(pass,"middleSweep","middle",s.middleSweeps-1);}
   this.dispatch(pass,"middleRestrict","middle");
   if(wide)this.sweep(pass,"coarseSweep","coarse",s.coarseSweeps);else this.dispatch(pass,coarse,1);
   if(fused)this.dispatch(pass,"middleSweepU",1);else{this.dispatch(pass,"middleSweepP","middle");this.dispatch(pass,"middleSweep1","middle");this.sweep(pass,"middleSweep","middle",s.middleSweeps-1);}
   this.dispatch(pass,"middleProlong","slots");fine();
  }
  this.dispatch(pass,`measure@${s.cycles}`,"slots");
  pass.end();
  const project=encoder.beginComputePass({label:"Uniform pressure band projection"});this.dispatch(project,"copy","slots");this.dispatch(project,"project","slots");this.dispatch(project,"present","slots");project.end();
 }
 /** The eight header words, for the frame's receipt. */
 encodeReceipt(encoder:GPUCommandEncoder,target:GPUBuffer,offset:number):void{encoder.copyBufferToBuffer(this.index,0,target,offset,32);}
 /** Diagnostics: the residual history words (before each cycle, then after the last). */
 static readonly historyWord=HISTORY_WORD;
 /** The header word the frame's verdict closes the band with. */
 static readonly closedWord=CLOSED_WORD;
 destroy():void{for(const b of [this.index,this.rows,this.coarse,this.solve])b.destroy();}
}
