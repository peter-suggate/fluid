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
import {UNIFORM_ARENA_CLASS,UNIFORM_ARENA_RELEASE_ENTRY,UNIFORM_ARENA_TABLE_ENTRIES,UniformBrickArena,type UniformArenaChannel} from "./uniform-brick-arena";

/** The h pressure band on the brick arena (docs/plans/pressure-rfe-speed-of-light-2026-10-20.md
 * §1.6): V-cycles over the arena's levels. Level 0 is the band's h rows in
 * 8^3 bricks, level 1 (2h) and level 2 (4h, a page a brick) are the arena's
 * MG levels, and a one-workgroup tail solves 8h, 16h and 32h per page.
 * Every level smooths with an overlapping box: a brick stages its cells two
 * deep around it, red-black sweeps all but the outer ring and keeps its own
 * 8^3. Restriction sums the children (exact h residuals; the coarse levels'
 * box-local ones); prolongation is trilinear over the parents that exist,
 * renormalised. The coarse operators are the children's aggregation scaled
 * by 0.5 (7-point at every level), baked once a frame by the level below. */
export interface UniformPressureArenaSchedule {
 /** V-cycles at most (NB-FLIP bandCycles). */
 cycles:number;
 /** Red-black sweeps of each box, before and after each coarse correction. */
 sweeps:number;
 /** Red-black sweeps of the 32h level, the coarsest. */
 coarsestSweeps:number;
 /** The aggregated operator's scale (0.5: the rediscretised interior operator). */
 coarseScale:number;
}
export const UNIFORM_PRESSURE_ARENA_SCHEDULE:UniformPressureArenaSchedule={cycles:4,sweeps:3,coarsestSweeps:40,coarseScale:0.5};
/** bandExit "returns": the band's residual target (1/s, h-equivalent). The
 * band stops at it or at the cycle cap; the strict exit keeps the frame's
 * 1e-4 tolerance. 0.05 1/s is a 2.5 mm/s worst-cell velocity error at h 5 cm;
 * the 4h root accepts up to 5. */
export const UNIFORM_PRESSURE_ARENA_RETURNS_RESIDUAL=0.05;

/** Workgroups of the band-tile mark at most (the simulation's buffered h tiles). */
const LIST_GROUPS=4096;
/** Brick and page launches stride their lists from at most this many groups. */
const BRICK_GROUPS=2048,PAGE_GROUPS=1024,TABLE_GROUPS=1024;
/** Brick kernels: a lane a cell of the brick (and of each colour of its box). */
const WG=512;
/** The coarse tail's one workgroup. */
const TAIL_LANES=256;
/** Pages with band rows the coarsest level stages at most (workgroup memory; more is fatal). */
const TAIL_PAGES=496;
const BAND_CYCLE_WINDOW=4;
const groupsOf=(count:number,per:number):number=>{const padded=count+per-1;return (padded-padded%per)/per;};
const PREPARE_PASS:GPUComputePassDescriptor={label:"Uniform pressure arena marks and rows"};
const SOLVE_PASS:GPUComputePassDescriptor={label:"Uniform pressure arena solve"};
/** Header words: band tiles, overflow, final residual, live bricks, completed
 * cycles, closed (the schedule's verdict), converged, valid residual bits
 * (history in bits 0..15, initial in bit 16), the residual after each cycle's
 * pre-smoothing (and after the last cycle) in HISTORY words, two spare, then
 * the initial residual. The tile slot+1 map (presentation slots), the
 * per-brick band-tile masks and the arena table follow. */
const HEADER=27,CLOSED_WORD=5,HISTORY_WORD=8,HISTORY=16,INITIAL_WORD=26,VALID_WORD=7,BRICKS_WORD=3;
/** Kinds bits beyond the band's: an active row (liquid, positive diagonal) and a warm one (its slot held a liquid row last frame). */
const ACTIVE_BIT=1<<27,WARM_BIT=1<<28;

export interface UniformPressureArenaFields {
 phi:GPUBufferBinding;
 coarsePhi?:GPUBufferBinding;
 authorityParams:GPUBuffer;
 vertexPhi:GPUTexture;
 forced:{velocity:GPUTexture;negative:GPUBuffer};
 velocity:GPUTexture;negative:GPUBuffer;copy:GPUTexture;
 coarsePressure:GPUBufferBinding;
 /** As the band's (64 bytes): h.xyz, dt; density, openTop, dt/density, min
  * h; the residual target, then 1/h^2 per axis; four u32 the arena writes:
  * the presentation capacity, two spare, the presentation word. */
 params:GPUBuffer;
 presentation:{buffer:GPUBuffer;word:number};
 /** Band tiles the presentation holds (more is the fatal overflow). */
 capacity:number;
 dynamics?:GPUBuffer;
 freeze?:UniformColdFreeze;
 coldFrame?:UniformColdFrame;
 warmTiers?:GPUBuffer;
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
 private readonly pool:GPUBuffer;
 get allocatedBytes():number{return this.index.size+this.pool.size;}
 private readonly tiles:number;
 private readonly maskWord:number;
 private readonly tableWord:number;
 private readonly level2Clear:{offset:number;size:number};
 private slots=0;
 get capacity():number{return this.slots;}
 get slotMapOffset():number{return 4*HEADER;}
 private workBricks=1;private brickGroups=1;
 private readonly recentCycles=new Uint32Array(BAND_CYCLE_WINDOW);
 private recentCount=0;
 private wideCycles=Number.POSITIVE_INFINITY;
 private readonly layouts=new Map<string,GPUBindGroupLayout>();
 private readonly groups=new Map<string,UniformDetailGroup>();
 private readonly pipelines=new Map<string,GPUComputePipeline>();
 private readonly launches=new Map<string,{pipeline:GPUComputePipeline;group:UniformDetailGroup}>();
 private readonly restricts:readonly string[];
 private boundPass?:GPUComputePassEncoder;
 private boundEpoch=-1;
 private boundGroup?:UniformDetailGroup;
 /** encodePrepare's verdict, which encodeSolve keeps. */
 private skipped=true;
 private get cold():UniformColdFreeze|undefined{return UNIFORM_COLD_FREEZE?this.fields.freeze:undefined;}
 constructor(private readonly device:GPUDevice,private readonly simulation:UniformMixedOwnership,private readonly pressure:UniformMixedOwnership,private fields:UniformPressureArenaFields,
  readonly schedule:UniformPressureArenaSchedule=UNIFORM_PRESSURE_ARENA_SCHEDULE,private readonly solid?:UniformMixedSolid){
  if(pressure.layout.tiles.some(word=>(word&0xc0000000)!==0))throw new Error("The pressure arena's global stage must be all-4h");
  if(solid&&!solid.coarse)throw new Error("The solid pressure arena needs the all-4h solid record");
  if(!Number.isInteger(schedule.cycles)||schedule.cycles<1||schedule.cycles>=HISTORY)throw new Error(`Pressure arena cycles must be 1..${HISTORY-1}`);
  const address=simulation.capacity.address,blocks=address.blocks;
  this.tiles=address.tileKeys;
  const cell=(name:string,level:0|1|2):UniformArenaChannel=>({name,levels:[level],layout:"cell"});
  const six=(prefix:string,level:0|1|2)=>Array.from({length:6},(_,f)=>cell(`${prefix}${f}`,level));
  // Level 2's channels last and contiguous: one clear empties them each frame.
  const channels=[...["x","y","base","warm","rhs","kinds"].map(name=>cell(name,0)),...six("c",0),...six("u",0),...(solid?six("v",0):[]),
   ...["xa","ya","ra","da"].map(name=>cell(name,1)),...six("ka",1),
   ...["xb","yb","rb","db"].map(name=>cell(name,2)),...six("kb",2),...["tx","ty","tr","td"].map(name=>cell(name,2)),...six("tk",2)];
  this.arena=new UniformBrickArena({blocks,channels,lists:[{name:"fine",any:UNIFORM_ARENA_CLASS.fine}],level1From:"fine"});
  const a=this.arena;
  for(const [prefix,level] of [["c",0],["u",0],...(solid?[["v",0]]:[]),["ka",1],["kb",2],["tk",2]] as [string,0|1|2][]){
   for(let f=0;f<6;f++)if(a.base(`${prefix}${f}`,level)!==a.base(`${prefix}0`,level)+f*520*a.capacity(level))throw new Error(`Pressure arena: channel ${prefix} is not contiguous`);
  }
  const first=a.base("xb",2),last=a.base("tk5",2)+520*blocks;
  this.level2Clear={offset:4*first,size:4*(last-first)};
  this.maskWord=HEADER+this.tiles;this.tableWord=this.maskWord+a.bricks;
  const limit=device.limits.maxStorageBufferBindingSize;
  if(4*a.poolWords>limit)throw new Error(`Pressure arena pool (${4*a.poolWords} bytes) exceeds the storage binding limit ${limit}`);
  this.index=device.createBuffer({label:"Uniform pressure arena state",size:4*(this.tableWord+a.tableWords),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC});
  this.pool=device.createBuffer({label:"Uniform pressure arena pool",size:4*a.poolWords,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC});
  for(const w of a.initialWrites(this.tableWord))device.queue.writeBuffer(this.index,w.offset,w.data as Uint32Array<ArrayBuffer>);
  this.restricts=Array.from({length:schedule.cycles},(_,k)=>`pRestrict0@${k}`);
  const texture=(binding:number)=>({binding,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float" as const,viewDimension:"3d" as const}});
  const buffer=(binding:number,type:GPUBufferBindingType="storage")=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type}});
  const write=(binding:number)=>({binding,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"write-only" as const,format:"rgba32float" as const,viewDimension:"3d" as const}});
  const params=buffer(0,"uniform"),cold=UNIFORM_COLD_FREEZE&&fields.freeze?[{binding:14,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float" as const,viewDimension:"2d" as const}}]:[];
  const entries:Record<string,GPUBindGroupLayoutEntry[]>={
   table:[buffer(2)],
   mark:[params,buffer(2),buffer(3,"read-only-storage"),texture(6),...cold,...(fields.warmTiers?[buffer(25,"read-only-storage")]:[])],
   rows:[params,buffer(2),buffer(9),buffer(3,"read-only-storage"),buffer(4,"uniform"),texture(6),texture(7),buffer(8,"read-only-storage"),...cold],
   init:[params,buffer(2),buffer(9),texture(11),buffer(12,"read-only-storage"),buffer(13,"read-only-storage"),write(23),...cold],
   solver:[params,buffer(2),buffer(9),...(fields.dynamics?[buffer(24)]:[])],
   project:[params,buffer(2),buffer(9),texture(15),write(16),buffer(17),buffer(21)],
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
  const P={0:{buffer:f.params,size:64}},S={2:{buffer:this.index}},Q={9:{buffer:this.pool}};
  const resources:Record<string,Record<number,GPUBindingResource|GPUTexture>>={
   table:{...S},
   mark:{...P,...S,3:scalars(f.phi,cells),6:f.vertexPhi},
   rows:{...P,...S,...Q,3:scalars(f.phi,cells),4:{buffer:f.authorityParams,size:16},6:f.vertexPhi,7:f.forced.velocity,8:{buffer:f.forced.negative,size:4*faces}},
   init:{...P,...S,...Q,11:f.velocity,12:scalars(f.coarsePressure,coarse),13:scalars(f.coarsePhi??f.phi,coarse),23:f.copy},
   solver:{...P,...S,...Q},
   project:{...P,...S,...Q,15:f.copy,16:f.velocity,17:{buffer:f.negative,size:4*faces},21:{buffer:f.presentation.buffer}},
  };
  if(f.dynamics)resources.solver![24]={buffer:f.dynamics};
  const freeze=this.cold;if(freeze)for(const name of ["mark","rows","init"])resources[name]![14]=freeze.regions;
  if(f.warmTiers)resources.mark![25]={buffer:f.warmTiers};
  for(const [name,map] of Object.entries(resources))this.groups.set(name,uniformDetailGroup(device,{label:`Uniform pressure arena ${name}`,layout:this.layouts.get(name)!,
   entries:Object.entries(map).map(([binding,resource])=>({binding:+binding,resource}))}));
  this.bindLaunches();this.boundPass=undefined;this.boundGroup=undefined;
 }
 /** Between frames, after the h-tile capacity (the phi view, the stage grids) changed: rebind. The arena's own buffers are lattice-sized. */
 resize(fields:Pick<UniformPressureArenaFields,"phi"|"presentation"|"capacity">):void{this.fields={...this.fields,...fields};this.bind();}
 /** Fixed-lag evidence: band tiles and live bricks of a receipted frame. Launch width only. */
 observeWork(tiles:number,reserve=0,bricks=tiles):void{
  this.workBricks=uniformBufferedWork(this.workBricks,Math.min(bricks,tiles),this.arena.fineCapacity,reserve);
  this.brickGroups=Math.min(this.workBricks,BRICK_GROUPS);
 }
 observeCycles(cycles:number):void{
  this.recentCycles[this.recentCount%BAND_CYCLE_WINDOW]=cycles;this.recentCount++;
  if(this.recentCount<BAND_CYCLE_WINDOW){this.wideCycles=Number.POSITIVE_INFINITY;return;}
  let most=0;for(let i=0;i<BAND_CYCLE_WINDOW;i++)most=Math.max(most,this.recentCycles[i]!);
  this.wideCycles=most;
 }
 forgetCycles():void{this.recentCount=0;this.wideCycles=Number.POSITIVE_INFINITY;}
 /** Bytes of the arena's buffers (lattice-sized: no capacity changes them). */
 get bytes():{state:number;pool:number}{return {state:this.index.size,pool:this.pool.size};}

 async initialize():Promise<void>{
  const layout=this.simulation.layout,S=!!this.solid,schedule=this.schedule,a=this.arena;
  const cap=a.fineCapacity,blocks=this.simulation.capacity.address.blocks;
  const plane0=520*cap,plane2=520*blocks;
  const C=(name:string,level:0|1|2=0)=>a.base(name,level);
  const tableWGSL=a.tableWGSL("state",`${this.tableWord}u`,`atomicLoad(&state[MASK+id])`);
  const constants=/* wgsl */`
const SLOTS:u32=${HEADER}u;const MASK:u32=${this.maskWord}u;const HISTORY:u32=${HISTORY_WORD}u;const CYCLE:u32=4u;const DONE:u32=6u;const BRICKS:u32=${BRICKS_WORD}u;
const INITIAL:u32=${INITIAL_WORD}u;const VALID:u32=${VALID_WORD}u;const INITIAL_BIT:u32=1u<<${HISTORY}u;const CYCLES:u32=${schedule.cycles}u;
const K_GHOST:u32=0u;const K_BAND:u32=1u;const K_WALL:u32=2u;const K_OPEN:u32=3u;const K_NEUMANN:u32=4u;const K_CLOSED:u32=5u;
const P_ACTIVE:u32=${ACTIVE_BIT}u;const P_WARM:u32=${WARM_BIT}u;const P_FINE:u32=${UNIFORM_ARENA_CLASS.fine}u;
const P_X:u32=${C("x")}u;const P_Y:u32=${C("y")}u;const P_BASE:u32=${C("base")}u;const P_WARMP:u32=${C("warm")}u;const P_RHS:u32=${C("rhs")}u;const P_KINDS:u32=${C("kinds")}u;
const P_C:u32=${C("c0")}u;const P_U:u32=${C("u0")}u;${S?`const P_V:u32=${C("v0")}u;`:""}const PLANE:u32=${plane0}u;
const PA_X:u32=${C("xa",1)}u;const PA_Y:u32=${C("ya",1)}u;const PA_R:u32=${C("ra",1)}u;const PA_D:u32=${C("da",1)}u;const PA_K:u32=${C("ka0",1)}u;
const PB_X:u32=${C("xb",2)}u;const PB_Y:u32=${C("yb",2)}u;const PB_R:u32=${C("rb",2)}u;const PB_D:u32=${C("db",2)}u;const PB_K:u32=${C("kb0",2)}u;const PLANE2:u32=${plane2}u;
const PT_X:u32=${C("tx",2)}u;const PT_Y:u32=${C("ty",2)}u;const PT_R:u32=${C("tr",2)}u;const PT_D:u32=${C("td",2)}u;const PT_K:u32=${C("tk0",2)}u;
const P_SCALE:f32=${schedule.coarseScale};
`;
  const params=/* wgsl */`
struct BandParams {hDt:vec4f,policy:vec4f,solve:vec4f,capacity:vec4u}
@group(1) @binding(0) var<uniform> params:BandParams;
`;
  const helpers=/* wgsl */`
fn pIndex(i:u32)->u32{return atomicLoad(&state[i]);}
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
fn pClosed()->bool{return pIndex(${CLOSED_WORD}u)!=0u;}
// Only completed measurements are read (a dispatch never tests the word it reduces).
fn bMet(word:u32,mask:u32)->bool{return (pIndex(VALID)&mask)!=0u&&params.solve.x>0.0&&bitcast<f32>(pIndex(word))<=params.solve.x;}
fn bInitialMet()->bool{return bMet(INITIAL,INITIAL_BIT);}
fn bStop(j:u32)->bool{return bMet(HISTORY+j,1u<<j);}
fn bConvergedBefore(k:u32)->bool{if(bInitialMet()){return true;}for(var j=0u;j<min(k,${HISTORY-1}u);j++){if(bStop(j)){return true;}}return false;}
fn bConverged()->bool{return bConvergedBefore(CYCLES);}
// The fine list (the rows precede the verdict), and the solve's: none when closed.
fn pFine()->u32{return ubaListCount(UBA_LIST_FINE);}
fn pCount()->u32{return select(pFine(),0u,pClosed());}
fn pLive()->u32{return select(pCount(),0u,bConverged());}
`;
  const poolWGSL=/* wgsl */`
@group(1) @binding(9) var<storage,read_write> pool:array<u32>;
${a.poolWGSL("pool")}
fn pF(base:u32,at:u32)->f32{return bitcast<f32>(pool[base+at]);}
fn pSet(base:u32,at:u32,value:f32){pool[base+at]=bitcast<u32>(value);}
fn pAddr(slot:u32,l:vec3u)->u32{return UBA_STRIDE*slot+l.x+8u*(l.y+8u*l.z);}
// The pool offset of band cell q (UB_NONE_ID: no band cell).
fn pAt(q:vec3i)->u32{if(!pBandCell(q)){return UB_NONE_ID;}return pAddr(ubaSlot(ubaBrickAt(q)),vec3u(q)&vec3u(7u));}
// The band cell across face f of band cell 'at' (at p): only for a K_BAND face.
fn pNear(at:u32,p:vec3i,f:u32)->u32{
 let axis=f/2u;var m=p&vec3i(7);m[axis]+=bSign(f);
 if(m[axis]>=0&&m[axis]<8){let step=select(select(64u,8u,axis==1u),1u,axis==0u);return select(at-step,at+step,bSign(f)>0);}
 var q=p;q[axis]+=bSign(f);return pAt(q);
}
// A unit face (most band faces) loads nothing.
fn pCoefficient(at:u32,kinds:u32,f:u32)->f32{if(bUnit(kinds,f)){return params.solve[1u+f/2u];}return pF(P_C+f*PLANE,at);}
fn pForced(at:u32,f:u32)->f32{return pF(P_U+f*PLANE,at);}
${S?"fn pVolume(at:u32,f:u32)->f32{return pF(P_V+f*PLANE,at);}":""}
// A wall face's native one-cell halo, less the row's start: max(x + lift, -base).
fn pLift(at:u32,f:u32)->f32{return f32(bSign(f))*params.policy.x*pForced(at,f)*params.hDt[f/2u]/params.hDt.w;}
// A level 0 row: its pool offset, kinds, rhs, start and diagonal (the band's sum, face by face).
struct PCell {at:u32,kinds:u32,rhs:f32,base:f32,d0:f32}
fn p0Cell(at:u32)->PCell{
 if(at==UB_NONE_ID){return PCell(at,0u,0.0,0.0,0.0);}
 let kinds=pool[P_KINDS+at];if(!pActive(kinds)){return PCell(at,kinds,0.0,0.0,0.0);}
 var d=0.0;for(var f=0u;f<6u;f++){d+=pCoefficient(at,kinds,f);}
 return PCell(at,kinds,pF(P_RHS,at),pF(P_BASE,at),d);
}
fn pZeroSeam(p:vec3i,word:u32)->f32{return 0.0;}
`;
  const header=(neighbours=false)=>uniformMixedTopologyWGSL(layout,0)+(neighbours?uniformMixedTopologyNeighboursWGSL():"")+uniformMixedFaceAddressWGSL+uniformMixedSolidWGSL(S?2:undefined,this.solid?.coarse?.count)+params+/* wgsl */`
@group(1) @binding(2) var<storage,read_write> state:array<atomic<u32>>;
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
  const coarse=(L:1|2)=>{const [X,Y,R,D,K,PL]=L===1?["PA_X","PA_Y","PA_R","PA_D","PA_K","PLANE"]:["PB_X","PB_Y","PB_R","PB_D","PB_K","PLANE2"];return box(L)+/* wgsl */`
const CX:u32=${X};const CY:u32=${Y};const CRH:u32=${R};const CD:u32=${D};const CK:u32=${K};const CPL:u32=${PL};
fn cDiag(at:u32)->f32{if(at==UB_NONE_ID){return 0.0;}return pF(CD,at);}
// A coarse row, held by its lane: diagonal (0: none), RHS, couplings.
struct CCell {d:f32,r:f32,k:array<f32,6>}
fn cCell(at:u32)->CCell{
 var cell:CCell;if(at==UB_NONE_ID){return cell;}
 let d=pF(CD,at);if(d<=0.0){return cell;}
 cell.d=d;cell.r=pF(CRH,at);for(var f=0u;f<6u;f++){cell.k[f]=pF(CK+f*CPL,at);}
 return cell;
}
fn cBoxRow(col:u32,lane:u32)->CCell{if(lane>=500u){return cCell(UB_NONE_ID);}return cCell(cAddr(cBoxCell(col,lane)));}
fn cOff(c:vec3i,cell:CCell)->f32{
 var off=0.0;
 for(var f=0u;f<6u;f++){if(cell.k[f]!=0.0){var q=c;q[f/2u]+=bSign(f);off+=cell.k[f]*cX[cBox(q)];}}
 return off;
}
// The box sweeps: lane < 500 holds one cell of each colour.
fn cSweeps(red:vec3i,cR0:CCell,black:vec3i,cB0:CCell){
 for(var s=0u;s<${schedule.sweeps}u;s++){
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
fn gDiag(A:vec3u)->f32{var s=0.0;for(var k=0u;k<8u;k++){s+=gPart[gChild(A,k)];}return P_SCALE*s;}
fn gCouple(A:vec3u,f:u32)->f32{var s=0.0;let axis=f/2u;for(var k=0u;k<8u;k++){if(((k>>axis)&1u)==(f&1u)){s+=gCross[512u*axis+gChild(A,k)];}}return P_SCALE*s;}
fn gSum(A:vec3u)->f32{var s=0.0;for(var k=0u;k<8u;k++){s+=gRes[gChild(A,k)];}return s;}
// A coarse child's part: d less its inner couplings, its outer ones by axis.
fn gCoarseChild(i:u32,l:vec3u,at:u32,d:f32,K:u32,PL:u32){
 var part=0.0;var cross=vec3f(0.0);
 if(at!=UB_NONE_ID&&d>0.0){
  part=d;
  for(var axis=0u;axis<3u;axis++){let bit=l[axis]&1u;part-=pF(K+(2u*axis+1u-bit)*PL,at);cross[axis]=pF(K+(2u*axis+bit)*PL,at);}
 }
 gPart[i]=part;gCross[i]=cross.x;gCross[512u+i]=cross.y;gCross[1024u+i]=cross.z;
}
`;
  const zero=/* wgsl */`var<workgroup> pN:u32;`;
  // ---- level 0 box (nonlinear rows: walls follow, inside rows clamp).
  const fineBox=box(0)+/* wgsl */`
// UniformPressureBand's relaxation of a row from its box neighbours; a row
// with walls solves its own halos' clamps exactly (four active-set passes).
fn p0Relax(c:vec3i,cell:PCell)->f32{
 var off=cell.rhs;var walls=false;
 for(var f=0u;f<6u;f++){
  let kind=bFaceKind(cell.kinds,f);
  if(kind==K_BAND){var q=c;q[f/2u]+=bSign(f);off+=pCoefficient(cell.at,cell.kinds,f)*cX[cBox(q)];}
  else if(kind==K_WALL){walls=true;}
 }
 var v=off/cell.d0;
 if(walls){
  for(var it=0u;it<4u;it++){
   var d=cell.d0;var o=off;
   for(var f=0u;f<6u;f++){
    if(bFaceKind(cell.kinds,f)!=K_WALL){continue;}
    let w=pF(P_C+f*PLANE,cell.at);let g=pLift(cell.at,f);
    if(v+g>=-cell.base){d-=w;o+=w*g;}else{o-=w*cell.base;}
   }
   if(d>0.0){v=o/d;}
  }
 }
 if(bInside(cell.kinds)){v=max(v,-cell.base);}
 return v;
}
override pSettle:bool=false;
// The box's sweeps (with pSettle: first a red then a black half sweep of the
// shut rows alone, exact in the core: every other row is still zero).
fn p0Sweeps(lane:u32,red:vec3i,cR0:PCell,black:vec3i,cB0:PCell){
 for(var s=0u;s<${schedule.sweeps}u+select(0u,1u,pSettle);s++){
  let settle=pSettle&&s==0u;
  for(var half=0u;half<2u;half++){
   var c=red;var cell=cR0;if(half==1u){c=black;cell=cB0;}
   if(pActive(cell.kinds)&&cell.d0>0.0&&(!settle||bShut(cell.kinds))){cX[cBox(c)]=p0Relax(c,cell);}
   workgroupBarrier();
  }
 }
}
fn p0Preload(col:u32,lane:u32)->PCell{if(lane>=500u){return PCell(UB_NONE_ID,0u,0.0,0.0,0.0);}return p0Cell(cAddr(cBoxCell(col,lane)));}
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
  const tail=/* wgsl */`
// The tail's levels per page: 3 (8h, 4^3 at word 0), 4 (16h, 2^3 at 64), 5 (32h, one at 72).
fn tSize(L:u32)->u32{return 4u>>(L-3u);}
fn tOffset(L:u32)->u32{return select(select(72u,64u,L==4u),0u,L==3u);}
// The pool offset of level L cell g of page block's grid (g may lie one page out), UB_NONE_ID outside stored pages.
fn tAt(block:u32,L:u32,g:vec3i)->u32{
 let s=i32(tSize(L));let d=(g+vec3i(2*s))/s-vec3i(2);let l=g-d*s;var b=block;
 if(any(d!=vec3i(0))){b=ubaLevelRecord(2u,block,u32(13+d.x+3*d.y+9*d.z));if(b==UB_NONE_ID){return UB_NONE_ID;}}
 return UBA_STRIDE*b+tOffset(L)+u32(l.x+s*(l.y+s*l.z));
}
fn tLocal(j:u32,s:u32)->vec3i{return vec3i(vec3u(j%s,(j/s)%s,j/(s*s)));}
fn tPageLive(block:u32)->bool{for(var o=0u;o<8u;o++){if(ubaChild(2u,block,o)!=UB_NONE_ID){return true;}}return false;}
`;
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
    atomicStore(&state[SLOTS+t],slot+1u);ubaMark(id,P_FINE);atomicOr(&state[MASK+id],1u<<pTileBit(vec3u(c)&vec3u(7u)));
   }else{atomicStore(&state[1],1u);}
  }
 }
}`,
   rows:header()+poolWGSL+theta+owner+UniformColdFreeze.wgsl(this.cold,1,14)+/* wgsl */`
@group(1) @binding(4) var<uniform> authority:vec4f;
@group(1) @binding(7) var velocity:texture_3d<f32>;
@group(1) @binding(8) var<storage,read> negative:array<f32>;
fn bForcedField(p:vec3i,axis:u32,sign:i32)->f32{
 var a=p;if(sign<0){a[axis]-=1;}
 if(umFaceLow(a,axis)){return negative[umLowFaceIndex(vec3u(p),axis)];}
 return textureLoad(velocity,a,0)[axis];
}
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
${zero}
// UniformPressureBand's row assembly, a lane a cell of a brick: the kinds,
// coefficients, forced faces${S?", face volumes":""} and RHS of a band tile's
// cells; a cell of another tile has none (kinds 0). The cell's phi waits in
// the iterate for the start. A row whose slot held a liquid row last frame
// (and did not just enter) is warm.
@compute @workgroup_size(${WG}) fn pRows(${lanes}){
 if(lane==0u){pN=pFine();}
 let n=workgroupUniformLoad(&pN);
 for(var e=group.x;e<n;e+=groups.x){
  let slot=ubaListSlot(UBA_LIST_FINE,e);let id=ubaBrick(slot);let lb=vec3u(lane&7u,(lane>>3u)&7u,lane>>6u);let at=UBA_STRIDE*slot+lane;
  let warm=(ubaClass(slot)&UBA_ENTERING)==0u&&bLiquid(pool[P_KINDS+at]);
  if(((pIndex(MASK+id)>>pTileBit(lb))&1u)==0u){pool[P_KINDS+at]=0u;continue;}
  let p=vec3i(ubaOrigin(id)+lb);let l=lb&vec3u(3u);
  let own=bPhiH(p);let liquid=bWet(own,1.0);${S?`
  let solidClear=bLocalSolidWork&&umSolidRegionClearAtBuild(p);
  var shut=false;var inside=false;if(!solidClear){shut=!umCellIsOpen(p);inside=umCellInsideSolid(p);}`:""}
  var kinds=select(0u,0x80000000u,liquid)${S?"|select(0u,0x40000000u,inside)|select(0u,0x20000000u,shut)":""};var diagonal=0.0;var divergence=0.0;
  ${S?`let bodies=liquid&&!solidClear&&umBodyCount()>0u&&umBodyPoseNearTile(umTileKey(p>>vec3u(2u)),${UNIFORM_BODY_FLUX_REACH}u);var capacity=0.0;
  if(bodies||!bLocalSolidWork){capacity=bHeadCapacity(p);}`:""}
  for(var f=0u;f<6u;f++){
   let axis=f/2u;let sign=bSign(f);let h=params.hDt[axis];var q=p;q[axis]+=sign;
   let forced=bForcedField(p,axis,sign);var kind=K_GHOST;var coefficient=0.0;var fraction=1.0;
   ${S?`var low=p;if(sign<0){low[axis]-=1;}var high=low;high[axis]+=1;
   var volume=1.0;if(!solidClear||!umSolidValid(low)||!umSolidValid(high)){volume=umPressureFaceV(low,axis);}var sealed=false;`:""}
   var across=UM_RESIDENT;if(l[axis]==select(0u,3u,sign>0)){across=umCellKind(q);}
   if(across!=UM_RESIDENT){
    if(across==UM_AIR){kind=K_OPEN;coefficient=1.0/(h*h*bTheta(own,0.5*params.policy.w));}
    else{kind=K_WALL;coefficient=0.5/(h*h);fraction=0.5;}
   }
   else if(l[axis]==select(0u,3u,sign>0)&&umFrozenCell(q)){kind=K_NEUMANN;}
   else{
    let n=bNeighbour(q);
    var theta=1.0;
    if(liquid!=n.wet){
     if(${S?"!solidClear&&(shut||!umCellIsOpen(q))":"false"}){theta=bPairTheta(own,liquid,n.phi,n.wet);}
     ${S?"else if(inside&&umCellInsideSolid(q)&&umCellOpen(p)<0.1&&umCellOpen(q)<0.1){theta=1.0;}":""}
     else{theta=bFaceTheta(bFacePhi(p,axis,sign),own,liquid,n.phi,n.wet);}
    }
    if(pBandCell(q)){kind=K_BAND;coefficient=1.0/(h*h*theta);}
    else if(n.neumann){kind=K_NEUMANN;}
    else{coefficient=1.0/(h*h*theta);}
    ${S?"if(!solidClear&&kind!=K_NEUMANN){sealed=(liquid&&shut&&!n.wet)||(!liquid&&n.wet&&!umCellIsOpen(q));}":""}
   }
   ${S?`if(volume<=1e-6||sealed){kind=K_CLOSED;coefficient=0.0;fraction=0.0;}
   else if(kind==K_NEUMANN){fraction=volume;}
   else{if(kind==K_WALL){coefficient=volume/(h*h);}else{coefficient*=volume;}fraction=volume;}
   pSet(P_V+f*PLANE,at,volume);`:""}
   kinds|=(kind<<(3u*f))|select(0u,1u<<(18u+f),kind==K_BAND&&coefficient==params.solve[1u+axis]);pSet(P_U+f*PLANE,at,forced);pSet(P_C+f*PLANE,at,coefficient);
   if(kind!=K_NEUMANN){diagonal+=coefficient;divergence+=f32(sign)*fraction*forced/h;}
   ${S?"if(bodies&&kind!=K_WALL&&kind!=K_OPEN&&(!bLocalSolidWork||capacity!=fraction)){divergence+=f32(sign)*(capacity-fraction)*umSolidFaceVelocity(low,axis)/h;}":""}
  }
  let control=authority.z*${S?"select(1.0,umCellOpen(p),umSolidEnabled())":"1.0"};
  kinds|=select(0u,P_ACTIVE,liquid&&diagonal>0.0)|select(0u,P_WARM,warm);
  pSet(P_RHS,at,select(0.0,-params.policy.x*(divergence-control)/params.hDt.w,liquid));pool[P_KINDS+at]=kinds;pSet(P_X,at,own);
 }
}`,
   init:header(true)+poolWGSL+wet+UniformColdFreeze.wgsl(this.cold,1,14)+/* wgsl */`
@group(1) @binding(11) var velocity:texture_3d<f32>;
@group(1) @binding(12) var<storage,read> coarsePressure:array<f32>;
@group(1) @binding(13) var<storage,read> coarsePhi:array<f32>;
@group(1) @binding(23) var copied:texture_storage_3d<rgba32float,write>;
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
// UniformPressureBand's Neumann face velocity: the transferred 4h face, or the forced one where the root solved no flux.
fn bNeumann(at:u32,p:vec3i,f:u32)->f32{
 let axis=f/2u;var q=p;q[axis]+=bSign(f);
 if(bCoarseKind(p>>vec3u(2u)).y==0.0&&bCoarseKind(q>>vec3u(2u)).y==0.0){return pForced(at,f);}
 var a=p;if(bSign(f)<0){a[axis]-=1;}
 return textureLoad(velocity,a,0)[axis];
}
${zero}
// The start: last frame's pressure on a warm row, else UniformPressureBand's
// carried trilinear 4h pressure; the iterate is the correction to it.
// Neumann faces fold the projected 4h flux into the RHS. A non-band texel
// below a band cell takes the transferred velocity (the projection's other
// components; the first band cell along x, then y, then z copies it).
@compute @workgroup_size(${WG}) fn pInit(${lanes}){
 if(lane==0u){pN=pCount();}
 let n=workgroupUniformLoad(&pN);
 for(var e=group.x;e<n;e+=groups.x){
  let slot=ubaListSlot(UBA_LIST_FINE,e);let id=ubaBrick(slot);let lb=vec3u(lane&7u,(lane>>3u)&7u,lane>>6u);let at=UBA_STRIDE*slot+lane;
  if(((pIndex(MASK+id)>>pTileBit(lb))&1u)==0u){continue;}
  let p=vec3i(ubaOrigin(id)+lb);let kinds=pool[P_KINDS+at];
  for(var axis=0u;axis<3u;axis++){
   var q=p;q[axis]-=1;if(!umCellPresent(q)||pBandCell(q)){continue;}
   var first=true;for(var b=0u;b<axis;b++){var r=q;r[b]+=1;if(pBandCell(r)){first=false;}}
   if(first){textureStore(copied,q,textureLoad(velocity,q,0));}
  }
  if(!bLiquid(kinds)){
   for(var f=0u;f<6u;f++){if(bFaceKind(kinds,f)==K_NEUMANN){pSet(P_U+f*PLANE,at,bNeumann(at,p,f));}}
   pSet(P_X,at,0.0);pSet(P_BASE,at,0.0);continue;
  }
  var start=0.0;
  if((kinds&P_WARM)!=0u){start=pF(P_WARMP,at);}
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
  pSet(P_BASE,at,start);pSet(P_X,at,0.0);
  var rhs=pF(P_RHS,at);
  for(var f=0u;f<6u;f++){
   let axis=f/2u;let sign=bSign(f);
   if(bFaceKind(kinds,f)==K_NEUMANN){
    var a=p;if(sign<0){a[axis]-=1;}
    let kept=bNeumann(at,p,f);pSet(P_U+f*PLANE,at,kept);rhs-=params.policy.x*f32(sign)*${S?"pVolume(at,f)*":""}(kept+umColdFaceTerm(a,axis,1u))/(params.hDt[axis]*params.hDt.w);}
  }
  pSet(P_RHS,at,rhs);
 }
}`,
   // The rows against their start (UniformPressureBand.rebase), the initial
   // measure (x = 0: each row's own), each region's predictor residual, and
   // level 1's rows: each brick bakes its parent octant.
   rebase:header()+poolWGSL+bakeArrays+/* wgsl */`
${this.fields.dynamics?"@group(1) @binding(24) var<storage,read_write> bDynamics:array<atomic<u32>>;":""}
var<workgroup> worst:atomic<u32>;
var<workgroup> tileWorst:array<atomic<u32>,8>;
${zero}
@compute @workgroup_size(${WG}) fn pRebase(${lanes}){
 if(lane==0u){pN=pCount();atomicStore(&worst,0u);}
 if(lane<8u){atomicStore(&tileWorst[lane],0u);}
 let n=workgroupUniformLoad(&pN);
 for(var e=group.x;e<n;e+=groups.x){
  let slot=ubaListSlot(UBA_LIST_FINE,e);let id=ubaBrick(slot);let lb=vec3u(lane&7u,(lane>>3u)&7u,lane>>6u);let at=UBA_STRIDE*slot+lane;
  let band=((pIndex(MASK+id)>>pTileBit(lb))&1u)!=0u;
  var kinds=0u;if(band){kinds=pool[P_KINDS+at];}
  var part=0.0;var cross=vec3f(0.0);
  if(bLiquid(kinds)){
   let p=vec3i(ubaOrigin(id)+lb);let own=pF(P_BASE,at);var sums=vec3f(0.0);var open=0.0;
   var d0=0.0;var walls=0.0;var lifted=0.0;
   for(var axis=0u;axis<3u;axis++){
    var w=vec2f(0.0);var d=vec2f(0.0);
    for(var side=0u;side<2u;side++){
     let f=2u*axis+side;let kind=bFaceKind(kinds,f);let c=pF(P_C+f*PLANE,at);d0+=c;
     if(kind==K_BAND){
      let m=pNear(at,p,f);w[side]=c;d[side]=pF(P_BASE,m)-own;
      // Level 1: a coupling to an active row, inside the aggregate or out of it.
      if(pActive(pool[P_KINDS+m])){if(side==(lb[axis]&1u)){cross[axis]=c;}else{part-=c;}}
     }
     else if(kind==K_GHOST||kind==K_OPEN){open+=c;}
     else if(kind==K_WALL){walls+=c;lifted+=c*max(pLift(at,f),-own);}
    }
    sums[axis]=w.x*(d.x+d.y)+(w.y-w.x)*d.y;
   }
   let rhs=pF(P_RHS,at)+((sums.x+sums.z)+sums.y-open*own);pSet(P_RHS,at,rhs);
   if(pActive(kinds)){
    // The band's residual of the zero iterate (its halos at their start).
    var r=rhs+lifted;if(bInside(kinds)&&own<=0.0&&bResidualMagnitude(r)<0x7f800000u){r=max(r,0.0);}
    let m=bResidualMagnitude(r*params.policy.z);atomicMax(&worst,m);atomicMax(&tileWorst[pTileBit(lb)],m);
    // The error equation's diagonal: the walls follow their row.
    part+=d0-walls;
   }else{part=0.0;cross=vec3f(0.0);}
  }
  gPart[lane]=part;gCross[lane]=cross.x;gCross[512u+lane]=cross.y;gCross[1024u+lane]=cross.z;
  workgroupBarrier();
  if(lane<64u){
   let parent=ubaParent(0u,slot);let A=vec3u(lane&3u,(lane>>2u)&3u,lane>>4u);let o=parent.y;
   let l1=4u*vec3u(o&1u,(o>>1u)&1u,o>>2u)+A;let at1=pAddr(parent.x,l1);
   pSet(PA_D,at1,gDiag(A));for(var f=0u;f<6u;f++){pSet(PA_K+f*PLANE,at1,gCouple(A,f));}
  }
  ${this.fields.dynamics?`else if(lane<72u){
   let t=lane-64u;let tile=umTileKey(vec3i(ubaOrigin(id)/4u)+vec3i(vec3u(t&1u,(t>>1u)&1u,t>>2u)));
   let w=atomicExchange(&tileWorst[t],0u);
   if(((pIndex(MASK+id)>>t)&1u)!=0u){let at2=${UNIFORM_REGION_DYNAMICS_WORDS}u*umPageOfTile(tile)+${UNIFORM_REGION_DYNAMICS.pressureResidual}u;if(w>atomicLoad(&bDynamics[at2])){atomicMax(&bDynamics[at2],w);}}
  }`:"else if(lane<72u){atomicStore(&tileWorst[lane-64u],0u);}"}
  workgroupBarrier();
 }
 workgroupBarrier();
 if(lane==0u&&n>0u){atomicMax(&state[INITIAL],atomicLoad(&worst));atomicOr(&state[VALID],INITIAL_BIT);}
}`,
   smooth:header()+poolWGSL+fineBox+zero+/* wgsl */`
// Pre-smoothing: the brick's box from the current iterate.
@compute @workgroup_size(${WG}) fn pSmooth0(${lanes}){
 if(lane==0u){pN=pLive();}
 let n=workgroupUniformLoad(&pN);
 for(var e=group.x;e<n;e+=groups.x){
  let slot=ubaListSlot(UBA_LIST_FINE,e);cRecords(slot,lane);
  for(var k=lane;k<1728u;k+=${WG}u){let at=cAddr(cStaged(k));var x=0.0;if(at!=UB_NONE_ID){x=pF(P_X,at);}cX[k]=x;}
  let red=cBoxCell(0u,lane);let black=cBoxCell(1u,lane);let cR0=p0Preload(0u,lane);let cB0=p0Preload(1u,lane);
  workgroupBarrier();
  p0Sweeps(lane,red,cR0,black,cB0);
  ubaStore(P_Y,slot,lane,cX[cBox(cCore(lane))]);
 }
}`,
   // The exact h residual after pre-smoothing: its largest liquid row into
   // history word pCycle, the children's sum into level 1's RHS.
   restrict:header()+poolWGSL+box(0)+bakeArrays+zero+/* wgsl */`
override pCycle:u32=0u;
var<workgroup> worst:atomic<u32>;
@compute @workgroup_size(${WG}) fn pRestrict0(${lanes}){
 if(lane==0u){atomicStore(&worst,0u);pN=select(pCount(),0u,bConvergedBefore(pCycle));}
 let n=workgroupUniformLoad(&pN);
 for(var e=group.x;e<n;e+=groups.x){
  let slot=ubaListSlot(UBA_LIST_FINE,e);cRecords(slot,lane);
  for(var k=lane;k<1728u;k+=${WG}u){let at=cAddr(cStaged(k));var x=0.0;if(at!=UB_NONE_ID){x=pF(P_Y,at);}cX[k]=x;}
  workgroupBarrier();
  let c=cCore(lane);let cell=p0Cell(cAddr(c));var r=0.0;
  if(pActive(cell.kinds)&&cell.d0>0.0){
   let x=cX[cBox(c)];var off=0.0;
   for(var f=0u;f<6u;f++){
    let kind=bFaceKind(cell.kinds,f);
    if(kind==K_BAND){var q=c;q[f/2u]+=bSign(f);off+=pCoefficient(cell.at,cell.kinds,f)*cX[cBox(q)];}
    else if(kind==K_WALL){off+=pF(P_C+f*PLANE,cell.at)*max(x+pLift(cell.at,f),-cell.base);}
   }
   r=cell.rhs+off-cell.d0*x;
   if(bInside(cell.kinds)&&x+cell.base<=0.0&&bResidualMagnitude(r)<0x7f800000u){r=max(r,0.0);}
   atomicMax(&worst,bResidualMagnitude(r*params.policy.z));
  }
  gRes[lane]=r;workgroupBarrier();
  if(lane<64u){
   let parent=ubaParent(0u,slot);let A=vec3u(lane&3u,(lane>>2u)&3u,lane>>4u);let o=parent.y;
   pSet(PA_R,pAddr(parent.x,4u*vec3u(o&1u,(o>>1u)&1u,o>>2u)+A),gSum(A));
  }
 }
 workgroupBarrier();
 if(lane==0u&&n>0u){atomicMax(&state[HISTORY+pCycle],atomicLoad(&worst));atomicOr(&state[VALID],1u<<pCycle);}
}`,
   // Level 1 down: the box from zero, the box-local residual's sum into
   // level 2's RHS; with pBake first level 2's rows (the octant's).
   downA:header()+poolWGSL+coarse(1)+bakeArrays+/* wgsl */`
override pBake:bool=false;
@compute @workgroup_size(${WG}) fn pDownA(${lanes}){
 if(lane==0u){cN=select(ubaLevelCount(1u),0u,pClosed()||bConverged());}
 let n=workgroupUniformLoad(&cN);
 for(var e=group.x;e<n;e+=groups.x){
  let slot=ubaLevelSlot(1u,e);cRecords(slot,lane);
  for(var k=lane;k<1728u;k+=${WG}u){cX[k]=0.0;}
  let red=cBoxCell(0u,lane);let black=cBoxCell(1u,lane);let cR0=cBoxRow(0u,lane);let cB0=cBoxRow(1u,lane);
  workgroupBarrier();
  cSweeps(red,cR0,black,cB0);
  let c=cCore(lane);let at=cAddr(c);let core=cCell(at);let d=core.d;let x=cX[cBox(c)];
  ubaStore(CX,slot,lane,x);
  var r=0.0;if(d>0.0){r=core.r+cOff(c,core)-d*x;}
  workgroupBarrier();
  gRes[lane]=r;if(pBake){gCoarseChild(lane,vec3u(c),at,d,CK,CPL);}
  workgroupBarrier();
  if(lane<64u){
   let parent=ubaParent(1u,slot);let A=vec3u(lane&3u,(lane>>2u)&3u,lane>>4u);let o=parent.y;
   let at2=pAddr(parent.x,4u*vec3u(o&1u,(o>>1u)&1u,o>>2u)+A);
   pSet(PB_R,at2,gSum(A));
   if(pBake){pSet(PB_D,at2,gDiag(A));for(var f=0u;f<6u;f++){pSet(PB_K+f*PLANE2,at2,gCouple(A,f));}}
  }
 }
}`,
   // Level 2 down (a page a brick): the box from zero, then the page's 8h
   // rows from zero inside the page, their residual summed into its 16h RHS;
   // with pBake first the page's 8h, 16h and 32h rows (all within the page).
   downB:header()+poolWGSL+coarse(2)+bakeArrays+tail+/* wgsl */`
override pBake:bool=false;
var<workgroup> tX:array<f32,64>;
// The parent's rows of aggregate A of a 2s^3 child grid: children k at 2A+k (local index x+2s(y+2s z)).
fn tChild(A:vec3u,k:u32,s:u32)->u32{let l=2u*A+vec3u(k&1u,(k>>1u)&1u,k>>2u);return l.x+2u*s*(l.y+2u*s*l.z);}
// A child row's part and outward couplings (level 3 or 4, within the page) into gPart/gCross at i.
fn tPart(i:u32,l:vec3u,d:f32,k:array<f32,6>){
 var part=0.0;var cross=vec3f(0.0);
 if(d>0.0){part=d;for(var axis=0u;axis<3u;axis++){let bit=l[axis]&1u;part-=k[2u*axis+1u-bit];cross[axis]=k[2u*axis+bit];}}
 gPart[i]=part;gCross[i]=cross.x;gCross[512u+i]=cross.y;gCross[1024u+i]=cross.z;
}
fn tBake(at:u32,A:vec3u,s:u32){
 var d=0.0;var k=array<f32,6>(0.0,0.0,0.0,0.0,0.0,0.0);
 for(var j=0u;j<8u;j++){let i=tChild(A,j,s);d+=gPart[i];for(var axis=0u;axis<3u;axis++){let bit=(j>>axis)&1u;k[2u*axis+bit]+=gCross[512u*axis+i];}}
 pSet(PT_D,at,P_SCALE*d);for(var f=0u;f<6u;f++){pSet(PT_K+f*PLANE2,at,P_SCALE*k[f]);}
}
@compute @workgroup_size(${WG}) fn pDownB(${lanes}){
 if(lane==0u){cN=select(ubaLevelCount(2u),0u,pClosed()||bConverged());}
 let n=workgroupUniformLoad(&cN);
 for(var e=group.x;e<n;e+=groups.x){
  let slot=ubaLevelSlot(2u,e);
  if(lane==0u){cN=u32(tPageLive(slot));}
  if(workgroupUniformLoad(&cN)==0u){continue;}
  cRecords(slot,lane);
  for(var k=lane;k<1728u;k+=${WG}u){cX[k]=0.0;}
  let red=cBoxCell(0u,lane);let black=cBoxCell(1u,lane);let cR0=cBoxRow(0u,lane);let cB0=cBoxRow(1u,lane);
  workgroupBarrier();
  cSweeps(red,cR0,black,cB0);
  let c=cCore(lane);let at=cAddr(c);let core=cCell(at);let d=core.d;let x=cX[cBox(c)];
  ubaStore(CX,slot,lane,x);
  var r=0.0;if(d>0.0){r=core.r+cOff(c,core)-d*x;}
  workgroupBarrier();
  gRes[lane]=r;if(pBake){gCoarseChild(lane,vec3u(c),at,d,CK,CPL);}
  workgroupBarrier();
  // 8h: a lane a cell of the page's 4^3.
  let g=tLocal(lane,4u);let at3=UBA_STRIDE*slot+lane;var r3=0.0;var d3=0.0;var k3=array<f32,6>(0.0,0.0,0.0,0.0,0.0,0.0);
  if(lane<64u){
   let A=vec3u(g);r3=gSum(A);pSet(PT_R,at3,r3);
   if(pBake){d3=gDiag(A);for(var f=0u;f<6u;f++){k3[f]=gCouple(A,f);}}
   else{d3=pF(PT_D,at3);for(var f=0u;f<6u;f++){k3[f]=pF(PT_K+f*PLANE2,at3);}}
   tX[lane]=0.0;
  }
  workgroupBarrier();
  if(pBake&&lane<64u){pSet(PT_D,at3,d3);for(var f=0u;f<6u;f++){pSet(PT_K+f*PLANE2,at3,k3[f]);}tPart(lane,vec3u(g),d3,k3);}
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
  if(lane<64u){
   let x3=tX[lane];pSet(PT_X,at3,x3);
   var res=0.0;
   if(d3>0.0){res=r3-d3*x3;for(var f=0u;f<6u;f++){var q=g;q[f/2u]+=bSign(f);if(all(q>=vec3i(0))&&all(q<vec3i(4))){res+=k3[f]*tX[u32(q.x+4*q.y+16*q.z)];}}}
   gRes[lane]=res;
  }
  workgroupBarrier();
  // 16h: the page's 2^3 from zero inside the page (with pBake first its rows).
  let g4=tLocal(lane,2u);let at4=UBA_STRIDE*slot+64u+lane;var r4=0.0;var d4=0.0;var k4=array<f32,6>(0.0,0.0,0.0,0.0,0.0,0.0);
  if(lane<8u){
   for(var j=0u;j<8u;j++){r4+=gRes[tChild(vec3u(g4),j,2u)];}
   pSet(PT_R,at4,r4);if(pBake){tBake(at4,vec3u(g4),2u);}
  }
  workgroupBarrier();
  if(lane<8u){
   d4=pF(PT_D,at4);for(var f=0u;f<6u;f++){k4[f]=pF(PT_K+f*PLANE2,at4);}
   tX[lane]=0.0;if(pBake){tPart(lane,vec3u(g4),d4,k4);}
  }
  workgroupBarrier();
  for(var s=0u;s<${schedule.sweeps}u;s++){
   for(var col=0u;col<2u;col++){
    if(lane<8u&&d4>0.0&&u32(g4.x+g4.y+g4.z)%2u==col){
     var off=r4;
     for(var f=0u;f<6u;f++){var q=g4;q[f/2u]+=bSign(f);if(all(q>=vec3i(0))&&all(q<vec3i(2))){off+=k4[f]*tX[u32(q.x+2*q.y+4*q.z)];}}
     tX[lane]=off/d4;
    }
    workgroupBarrier();
   }
  }
  // 32h: the page's RHS (and with pBake its row).
  var res4=0.0;
  if(lane<8u){
   let x4=tX[lane];pSet(PT_X,at4,x4);
   if(d4>0.0){res4=r4-d4*x4;for(var f=0u;f<6u;f++){var q=g4;q[f/2u]+=bSign(f);if(all(q>=vec3i(0))&&all(q<vec3i(2))){res4+=k4[f]*tX[u32(q.x+2*q.y+4*q.z)];}}}
  }
  workgroupBarrier();
  if(lane<8u){gRes[lane]=res4;}
  workgroupBarrier();
  if(lane==0u){
   var r5=0.0;for(var j=0u;j<8u;j++){r5+=gRes[j];}
   pSet(PT_R,UBA_STRIDE*slot+72u,r5);if(pBake){tBake(UBA_STRIDE*slot+72u,vec3u(0u),1u);}
  }
  workgroupBarrier();
 }
}`,
   // 32h, the coarsest level (a row a page with band rows): one workgroup
   // stages the rows and solves them by red-black sweeps.
   tail:header()+poolWGSL+tail+/* wgsl */`
const T_PAGES:u32=${TAIL_PAGES}u;
var<workgroup> sX:array<f32,${TAIL_PAGES}>;
var<workgroup> sR:array<f32,${TAIL_PAGES}>;
var<workgroup> sD:array<f32,${TAIL_PAGES}>;
var<workgroup> sB:array<u32,${TAIL_PAGES}>;
var<workgroup> sK:array<f32,${6*TAIL_PAGES}>;
var<workgroup> sNear:array<u32,${6*TAIL_PAGES}>;
var<workgroup> sCount:atomic<u32>;
var<workgroup> tN:u32;
@compute @workgroup_size(${TAIL_LANES}) fn pTail(@builtin(local_invocation_index) lane:u32){
 if(lane==0u){tN=select(ubaLevelCount(2u),0u,pClosed()||bConverged());atomicStore(&sCount,0u);}
 let P=workgroupUniformLoad(&tN);if(P==0u){return;}
 // A row's staged index + 1, in its page's spare 32h word of the 8h up channel.
 for(var e=lane;e<P;e+=${TAIL_LANES}u){
  let block=ubaLevelSlot(2u,e);let at=UBA_STRIDE*block+72u;let d=pF(PT_D,at);
  if(d<=0.0){continue;}
  let i=atomicAdd(&sCount,1u);
  if(i>=T_PAGES){atomicStore(&state[1],2u);continue;}
  let page=vec3u(umBlockPage(block));
  sX[i]=0.0;sR[i]=pF(PT_R,at);sD[i]=d;sB[i]=block|(((page.x+page.y+page.z)&1u)<<31u);pool[PT_Y+at]=i+1u;
 }
 storageBarrier();workgroupBarrier();
 let count=min(atomicLoad(&sCount),T_PAGES);
 for(var i=lane;i<count;i+=${TAIL_LANES}u){
  let block=sB[i]&0x7fffffffu;let at=UBA_STRIDE*block+72u;
  for(var f=0u;f<6u;f++){
   var d=vec3i(0);d[f/2u]=bSign(f);let near=ubaLevelRecord(2u,block,u32(13+d.x+3*d.y+9*d.z));
   var j=0u;if(near!=UB_NONE_ID&&pF(PT_D,UBA_STRIDE*near+72u)>0.0){j=pool[PT_Y+UBA_STRIDE*near+72u];}
   sNear[6u*i+f]=j;sK[6u*i+f]=select(0.0,pF(PT_K+f*PLANE2,at),j!=0u);
  }
 }
 workgroupBarrier();
 for(var sweep=0u;sweep<${schedule.coarsestSweeps}u;sweep++){
  for(var col=0u;col<2u;col++){
   for(var i=lane;i<count;i+=${TAIL_LANES}u){
    if((sB[i]>>31u)!=col){continue;}
    var off=sR[i];for(var f=0u;f<6u;f++){let j=sNear[6u*i+f];if(j!=0u){off+=sK[6u*i+f]*sX[j-1u];}}
    sX[i]=off/sD[i];
   }
   workgroupBarrier();
  }
 }
 for(var i=lane;i<count;i+=${TAIL_LANES}u){pSet(PT_X,UBA_STRIDE*(sB[i]&0x7fffffffu)+72u,sX[i]);}
}`,
   // 16h and 8h up (a page a group): the 32h correction prolonged into the
   // page's 16h rows and their ring (two deep), the rows' sweeps against the
   // ring; then the 16h correction prolonged into its 8h rows and their ring,
   // and their sweeps.
   upT:header()+poolWGSL+tail+/* wgsl */`
var<workgroup> vX:array<f32,216>;
var<workgroup> vD:array<f32,216>;
var<workgroup> uX:array<f32,216>;
var<workgroup> uN:u32;
fn vBox(g:vec3i)->u32{let q=vec3u(g+2);return q.x+6u*(q.y+6u*q.z);}
fn uBox(g:vec3i)->u32{let q=vec3u(g+1);return q.x+6u*(q.y+6u*q.z);}
// The trilinear weight of parent offset d (0: the cell's own parent).
fn uWeight(d:vec3i)->f32{return select(0.75,0.25,d.x==1)*select(0.75,0.25,d.y==1)*select(0.75,0.25,d.z==1);}
@compute @workgroup_size(${TAIL_LANES}) fn pUpT(${lanes}){
 if(lane==0u){uN=select(ubaLevelCount(2u),0u,pClosed()||bConverged());}
 let n=workgroupUniformLoad(&uN);
 for(var e=group.x;e<n;e+=groups.x){
  let slot=ubaLevelSlot(2u,e);
  if(lane==0u){uN=u32(tPageLive(slot));}
  if(workgroupUniformLoad(&uN)==0u){continue;}
  // 16h: the down leg's rows plus the 32h correction (over the pages one away that exist).
  if(lane<216u){
   let g=vec3i(vec3u(lane%6u,(lane/6u)%6u,lane/36u))-2;let at=tAt(slot,4u,g);var x=0.0;var d=0.0;
   if(at!=UB_NONE_ID){d=pF(PT_D,at);}
   if(d>0.0){
    x=pF(PT_X,at);
    let G=g>>vec3u(1u);let o=select(vec3i(-1),vec3i(1),(g&vec3i(1))==vec3i(1));var sum=0.0;var total=0.0;
    for(var i=0u;i<8u;i++){
     let dd=vec3i(vec3u(i&1u,(i>>1u)&1u,i>>2u));let q=G+dd*o;if(any(abs(q)>vec3i(1))){continue;}
     let m=tAt(slot,5u,q);if(m!=UB_NONE_ID&&pF(PT_D,m)>0.0){sum+=uWeight(dd)*pF(PT_X,m);total+=uWeight(dd);}
    }
    if(total>0.0){x+=sum/total;}
   }
   vX[lane]=x;vD[lane]=d;
  }
  let g4=tLocal(lane,2u);let at4=UBA_STRIDE*slot+64u+lane;var d4=0.0;var r4=0.0;var k4=array<f32,6>(0.0,0.0,0.0,0.0,0.0,0.0);
  if(lane<8u){d4=pF(PT_D,at4);if(d4>0.0){r4=pF(PT_R,at4);for(var f=0u;f<6u;f++){k4[f]=pF(PT_K+f*PLANE2,at4);}}}
  workgroupBarrier();
  let page=vec3u(umBlockPage(slot));
  for(var s=0u;s<${schedule.sweeps}u;s++){
   for(var col=0u;col<2u;col++){
    if(lane<8u&&d4>0.0&&u32(g4.x+g4.y+g4.z)%2u==col){
     var off=r4;for(var f=0u;f<6u;f++){var q=g4;q[f/2u]+=bSign(f);off+=k4[f]*vX[vBox(q)];}
     vX[vBox(g4)]=off/d4;
    }
    workgroupBarrier();
   }
  }
  // 8h: the down leg's rows plus the 16h correction.
  if(lane<216u){
   let g=vec3i(vec3u(lane%6u,(lane/6u)%6u,lane/36u))-1;let at=tAt(slot,3u,g);var x=0.0;
   if(at!=UB_NONE_ID&&pF(PT_D,at)>0.0){
    x=pF(PT_X,at);
    let G=g>>vec3u(1u);let o=select(vec3i(-1),vec3i(1),(g&vec3i(1))==vec3i(1));var sum=0.0;var total=0.0;
    for(var i=0u;i<8u;i++){
     let dd=vec3i(vec3u(i&1u,(i>>1u)&1u,i>>2u));let q=G+dd*o;
     if(vD[vBox(q)]>0.0){sum+=uWeight(dd)*vX[vBox(q)];total+=uWeight(dd);}
    }
    if(total>0.0){x+=sum/total;}
   }
   uX[lane]=x;
  }
  let g=tLocal(lane,4u);let at3=UBA_STRIDE*slot+lane;var d3=0.0;var r3=0.0;var k3=array<f32,6>(0.0,0.0,0.0,0.0,0.0,0.0);
  if(lane<64u){d3=pF(PT_D,at3);if(d3>0.0){r3=pF(PT_R,at3);for(var f=0u;f<6u;f++){k3[f]=pF(PT_K+f*PLANE2,at3);}}}
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
  if(lane<64u){pSet(PT_Y,at3,uX[uBox(g)]);}
  workgroupBarrier();
 }
}`,
   // Level 2 up: the 8h correction prolonged into the page's box, then its sweeps.
   upB:header()+poolWGSL+coarse(2)+tail+/* wgsl */`
var<workgroup> tS:array<f32,1024>;
${prolong("tS[tBox(q)]","tS[512u+tBox(q)]")}
fn tBox(q:vec3i)->u32{let u=vec3u(q+2);return u.x+8u*(u.y+8u*u.z);}
@compute @workgroup_size(${WG}) fn pUpB(${lanes}){
 if(lane==0u){cN=select(ubaLevelCount(2u),0u,pClosed()||bConverged());}
 let n=workgroupUniformLoad(&cN);
 for(var e=group.x;e<n;e+=groups.x){
  let slot=ubaLevelSlot(2u,e);
  if(lane==0u){cN=u32(tPageLive(slot));}
  if(workgroupUniformLoad(&cN)==0u){continue;}
  cRecords(slot,lane);
  {let g=vec3i(vec3u(lane&7u,(lane>>3u)&7u,lane>>6u))-2;let m=tAt(slot,3u,g);var x=0.0;var d=0.0;if(m!=UB_NONE_ID){x=pF(PT_Y,m);d=pF(PT_D,m);}tS[lane]=x;tS[512u+lane]=d;}
  workgroupBarrier();
  for(var k=lane;k<1728u;k+=${WG}u){let c=cStaged(k);let at=cAddr(c);var x=0.0;if(at!=UB_NONE_ID){x=pF(CX,at);if(pF(CD,at)>0.0){x+=cProlong(c);}}cX[k]=x;}
  let red=cBoxCell(0u,lane);let black=cBoxCell(1u,lane);let cR0=cBoxRow(0u,lane);let cB0=cBoxRow(1u,lane);
  workgroupBarrier();
  cSweeps(red,cR0,black,cB0);
  ubaStore(CY,slot,lane,cX[cBox(cCore(lane))]);
  workgroupBarrier();
 }
}`,
   // Level 1 up: level 2's correction (the parent octant, two deep) prolonged into the brick's box, then its sweeps.
   upA:header()+poolWGSL+coarse(1)+a.stageWGSL("q",{level:2,region:"octant",channels:[{name:"yb",halo:2,seam:"pZeroSeam"},{name:"db",halo:2,seam:"pZeroSeam"}],workgroup:WG})+
    prolong("ubaW_q[ubaAt_q(q,0u)]","ubaW_q[ubaAt_q(q,1u)]")+/* wgsl */`
@compute @workgroup_size(${WG}) fn pUpA(${lanes}){
 if(lane==0u){cN=select(ubaLevelCount(1u),0u,pClosed()||bConverged());}
 let n=workgroupUniformLoad(&cN);
 for(var e=group.x;e<n;e+=groups.x){
  let slot=ubaLevelSlot(1u,e);let parent=ubaParent(1u,slot);
  ubaStage_q(parent.x,lane,parent.y);
  cRecords(slot,lane);
  for(var k=lane;k<1728u;k+=${WG}u){let c=cStaged(k);let at=cAddr(c);var x=0.0;if(at!=UB_NONE_ID){x=pF(CX,at);if(pF(CD,at)>0.0){x+=cProlong(c);}}cX[k]=x;}
  let red=cBoxCell(0u,lane);let black=cBoxCell(1u,lane);let cR0=cBoxRow(0u,lane);let cB0=cBoxRow(1u,lane);
  workgroupBarrier();
  cSweeps(red,cR0,black,cB0);
  ubaStore(CY,slot,lane,cX[cBox(cCore(lane))]);
 }
}`,
   // Level 0 up: level 1's correction prolonged (inside rows clamp), then the box's sweeps.
   upZero:header()+poolWGSL+fineBox+zero+a.stageWGSL("q",{level:1,region:"octant",channels:[{name:"ya",halo:2,seam:"pZeroSeam"},{name:"da",halo:2,seam:"pZeroSeam"}],workgroup:WG})+
    prolong("ubaW_q[ubaAt_q(q,0u)]","ubaW_q[ubaAt_q(q,1u)]")+/* wgsl */`
@compute @workgroup_size(${WG}) fn pUp0(${lanes}){
 if(lane==0u){pN=pLive();}
 let n=workgroupUniformLoad(&pN);
 for(var e=group.x;e<n;e+=groups.x){
  let slot=ubaListSlot(UBA_LIST_FINE,e);let parent=ubaParent(0u,slot);
  ubaStage_q(parent.x,lane,parent.y);
  cRecords(slot,lane);
  for(var k=lane;k<1728u;k+=${WG}u){
   let c=cStaged(k);let at=cAddr(c);var x=0.0;
   if(at!=UB_NONE_ID){
    x=pF(P_Y,at);let kinds=pool[P_KINDS+at];
    if(pActive(kinds)){x+=cProlong(c);if(bInside(kinds)){x=max(x,-pF(P_BASE,at));}}
   }
   cX[k]=x;
  }
  let red=cBoxCell(0u,lane);let black=cBoxCell(1u,lane);let cR0=p0Preload(0u,lane);let cB0=p0Preload(1u,lane);
  workgroupBarrier();
  p0Sweeps(lane,red,cR0,black,cB0);
  ubaStore(P_X,slot,lane,cX[cBox(cCore(lane))]);
 }
}`,
   project:header()+poolWGSL+zero+/* wgsl */`
@group(1) @binding(15) var field:texture_3d<f32>;
@group(1) @binding(16) var output:texture_storage_3d<rgba32float,write>;
@group(1) @binding(17) var<storage,read_write> boundary:array<f32>;
@group(1) @binding(21) var<storage,read_write> presented:array<f32>;
var<workgroup> worst:atomic<u32>;
var<workgroup> pMeasure:u32;
// The solution's channel: the last pre-smoothing's where a cycle stopped, else the last post-smoothing's.
var<private> pXF:u32;
fn pStopped()->bool{if(bInitialMet()){return false;}for(var j=0u;j<CYCLES;j++){if(bStop(j)){return true;}}return false;}
// UniformPressureBand's umProject on face f of band cell 'at' at p.
fn bFace(at:u32,p:vec3i,f:u32)->f32{
 let kinds=pool[P_KINDS+at];let kind=bFaceKind(kinds,f);
 if(kind==K_NEUMANN){return pForced(at,f);}
 let axis=f/2u;let sign=bSign(f);let h=params.hDt[axis];let liquid=bLiquid(kinds);
 ${S?"if(kind==K_CLOSED){var low=p;if(sign<0){low[axis]-=1;}return umSolidFaceVelocity(low,axis);}":""}
 let forced=pForced(at,f);let coefficient=pF(P_C+f*PLANE,at)${S?"/pVolume(at,f)":""};let own=pF(pXF,at);let base=pF(P_BASE,at);let scale=params.policy.z;
 if(kind==K_WALL){if(!liquid){return forced;}return forced-scale*f32(sign)*(max(own+pLift(at,f),-base)-own)/h;}
 if(kind==K_OPEN){if(!liquid){return forced;}return forced+scale*f32(sign)*(base+own)*coefficient*h;}
 var other=0.0;var otherBase=0.0;var otherLiquid=false;
 if(kind==K_BAND){let m=pNear(at,p,f);other=pF(pXF,m);otherBase=pF(P_BASE,m);otherLiquid=bLiquid(pool[P_KINDS+m]);}
 if(!liquid&&!otherLiquid){return 0.0;}
 return forced-scale*f32(sign)*((otherBase-base)+(other-own))*coefficient*h;
}
fn bReleased(at:u32,p:vec3i,f:u32,value:f32)->bool{
 let axis=f/2u;let sign=bSign(f);let kinds=pool[P_KINDS+at];
 ${S?`let volume=pVolume(at,f);
 if(sign>0){
  var q=p;q[axis]+=1;let own=umCellIsOpen(p);
  if(own==umCellIsOpen(q)||volume<=1e-6){return false;}
  var pressure=0.0;let kind=bFaceKind(kinds,f);
  if(kind==K_WALL||kind==K_OPEN){pressure=select(0.0,max(pF(pXF,at)+pLift(at,f),-pF(P_BASE,at))+pF(P_BASE,at),bLiquid(kinds)&&kind==K_WALL);}
  else if(own){let b=pAt(q);if(b!=UB_NONE_ID&&bLiquid(pool[P_KINDS+b])){pressure=pF(pXF,b)+pF(P_BASE,b);}}
  else if(bLiquid(kinds)){pressure=pF(pXF,at)+pF(P_BASE,at);}
  return pressure<=0.0&&select(1.0,-1.0,own)*(value-umSolidFaceVelocity(p,axis))*params.hDt.w>1e-4*params.hDt[axis];
 }
 if(!umCellIsOpen(p)||volume<=1e-6){return false;}`:""}
 if(bFaceKind(kinds,f)!=K_WALL){return false;}
 let pressure=select(0.0,max(pF(pXF,at)+pLift(at,f),-pF(P_BASE,at))+pF(P_BASE,at),bLiquid(kinds));
 return pressure<=0.0&&-f32(sign)*value*params.hDt.w>1e-4*params.hDt[axis];
}
// The final measure (when no cycle stopped), the band faces into the
// velocity, the band pressures into the stage grids, last frame's pressure
// for the next start.
@compute @workgroup_size(${WG}) fn pProject(${lanes}){
 if(lane==0u){atomicStore(&worst,0u);pN=pCount();pMeasure=u32(!bConvergedBefore(CYCLES));atomicStore(&state[BRICKS],pFine());}
 pXF=select(P_X,P_Y,pStopped());
 let n=workgroupUniformLoad(&pN);let measure=workgroupUniformLoad(&pMeasure)!=0u;
 for(var e=group.x;e<n;e+=groups.x){
  let slot=ubaListSlot(UBA_LIST_FINE,e);let id=ubaBrick(slot);let lb=vec3u(lane&7u,(lane>>3u)&7u,lane>>6u);let at=UBA_STRIDE*slot+lane;
  if(((pIndex(MASK+id)>>pTileBit(lb))&1u)==0u){continue;}
  let p=vec3i(ubaOrigin(id)+lb);let kinds=pool[P_KINDS+at];let own=pF(pXF,at);let base=pF(P_BASE,at);
  if(measure&&pActive(kinds)){
   var off=0.0;var d0=0.0;
   for(var f=0u;f<6u;f++){
    let kind=bFaceKind(kinds,f);let c=pCoefficient(at,kinds,f);d0+=c;
    if(kind==K_BAND){off+=c*pF(pXF,pNear(at,p,f));}
    else if(kind==K_WALL){off+=c*max(own+pLift(at,f),-base);}
   }
   if(d0>0.0){
    var r=pF(P_RHS,at)+off-d0*own;
    if(bInside(kinds)&&own+base<=0.0&&bResidualMagnitude(r)<0x7f800000u){r=max(r,0.0);}
    atomicMax(&worst,bResidualMagnitude(r*params.policy.z));
   }
  }
  let tile=umTileAt(vec3u(p)/4u);let local=vec3u(p)&vec3u(3u);
  presented[params.capacity.w+(pIndex(SLOTS+tile)-1u)*64u+local.x+4u*local.y+16u*local.z]=base+own;
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
 workgroupBarrier();
 if(lane==0u&&n>0u&&measure){let r=atomicLoad(&worst);atomicMax(&state[2],r);atomicMax(&state[HISTORY+CYCLES],r);atomicOr(&state[VALID],1u<<CYCLES);}
 if(group.x==0u&&lane==0u){
  var cycle=CYCLES;for(var j=CYCLES;j>0u;j--){if(bStop(j-1u)){cycle=j-1u;}}
  if(bInitialMet()||pCount()==0u){cycle=0u;}
  atomicStore(&state[CYCLE],cycle);
  if(cycle<CYCLES){atomicStore(&state[DONE],1u);atomicMax(&state[2],select(pIndex(HISTORY+cycle),pIndex(INITIAL),bInitialMet()));}
 }
}`,
  };
  const layoutOf:Record<string,string>={table:"table",mark:"mark",rows:"rows",init:"init",project:"project"};
  const entryOf:Record<string,string[]>={table:[...UNIFORM_ARENA_TABLE_ENTRIES,UNIFORM_ARENA_RELEASE_ENTRY],mark:["pMark"],rows:["pRows"],init:["pInit"],rebase:["pRebase"],smooth:["pSmooth0"],restrict:["pRestrict0"],
   downA:["pDownA"],downB:["pDownB"],tail:["pTail"],upT:["pUpT"],upB:["pUpB"],upA:["pUpA"],upZero:["pUp0"],project:["pProject"]};
  type Variant=[string,Record<string,number>];
  const variants:Record<string,Variant[]>={
   pSmooth0:[["",{}],["S",{pSettle:1}]],pRestrict0:Array.from({length:schedule.cycles},(_,k)=>[`@${k}`,{pCycle:k}] as Variant),
   pDownA:[["",{}],["@0",{pBake:1}]],pDownB:[["",{}],["@0",{pBake:1}]],
  };
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
 private bindLaunches():void{
  this.launches.clear();
  for(const [name,pipeline] of this.pipelines)this.launches.set(name,{pipeline,group:this.groups.get(this.pipelineGroups.get(name)!)!});
 }
 private dispatch(pass:GPUComputePassEncoder,name:string,launch:number|"tiles"):void{
  const bound=this.launches.get(name);if(!bound)throw new Error(`Pressure arena is not initialized (${name})`);
  if(this.boundPass!==pass||this.boundEpoch!==uniformComputePassEpoch(pass)){this.boundPass=pass;this.boundEpoch=uniformComputePassEpoch(pass);this.boundGroup=undefined;pass.setBindGroup(0,this.simulation.bindGroup);if(this.solid)pass.setBindGroup(2,this.solid.coarse!.bindGroup);}
  const pipeline=this.solid?.select(bound.pipeline)??bound.pipeline;
  if(this.boundGroup!==bound.group){this.boundGroup=bound.group;pass.setBindGroup(1,bound.group.group);}
  if(launch==="tiles"){this.simulation.dispatchBuffered(pass,pipeline,"fine",LIST_GROUPS);return;}
  pass.setPipeline(uniformDetailPick(pipeline));pass.dispatchWorkgroups(launch);
 }
 /** As the band's: no h tile, or a quiet frame under the freeze, lists none. */
 private get empty():boolean{
  if(this.simulation.capacity.fineTiles===0)return true;
  const ceiling=this.fields.coldFrame?.ceiling;
  return this.cold!==undefined&&ceiling!==undefined&&
   ceiling<(this.fields.warmTiers?UNIFORM_REGION_TIER.hot:UNIFORM_REGION_TIER.warm);
 }
 /** Before the split: mark band tiles, update the arena table, assemble the rows. */
 encodePrepare(encoder:GPUCommandEncoder):void{
  encoder.clearBuffer(this.index,0,4*this.tableWord);
  this.skipped=this.empty;if(this.skipped)return;
  const a=this.arena,step=a.stepClear(this.tableWord);
  encoder.clearBuffer(this.index,step.offset,step.size);encoder.clearBuffer(this.pool,this.level2Clear.offset,this.level2Clear.size);
  const blocks=this.simulation.capacity.address.blocks,cap=a.fineCapacity;
  const pass=encoder.beginComputePass(PREPARE_PASS);
  this.dispatch(pass,"pMark","tiles");
  this.dispatch(pass,"ubaUpdate",Math.min(groupsOf(a.bricks,64),TABLE_GROUPS));
  this.dispatch(pass,"ubaLists",Math.min(groupsOf(Math.max(cap,blocks),64),TABLE_GROUPS));
  this.dispatch(pass,"ubaLevel1",Math.min(groupsOf(this.workBricks,64),TABLE_GROUPS));
  this.dispatch(pass,"ubaRecord",Math.min(groupsOf(32*(2*this.workBricks+blocks),64),4*TABLE_GROUPS));
  this.dispatch(pass,"pRows",this.brickGroups);
  pass.end();
 }
 /** After the 4h projection reaches simulation ownership: start, V-cycles, projection. */
 encodeSolve(encoder:GPUCommandEncoder):void{
  if(this.skipped)return;
  const s=this.schedule,pages=Math.min(this.simulation.capacity.address.blocks,PAGE_GROUPS),g=this.brickGroups;
  const pass=encoder.beginComputePass(SOLVE_PASS);
  this.dispatch(pass,"pInit",g);this.dispatch(pass,"pRebase",g);
  // One cycle past the most the band ran lately; a frame needing more ends at its last.
  const cycles=Math.min(s.cycles,this.wideCycles+1);
  for(let cycle=0;cycle<cycles;cycle++){
   const first=cycle===0;
   this.dispatch(pass,first&&this.solid?.present?"pSmooth0S":"pSmooth0",g);
   this.dispatch(pass,this.restricts[cycle]!,g);
   this.dispatch(pass,first?"pDownA@0":"pDownA",g);this.dispatch(pass,first?"pDownB@0":"pDownB",pages);this.dispatch(pass,"pTail",1);
   this.dispatch(pass,"pUpT",pages);this.dispatch(pass,"pUpB",pages);this.dispatch(pass,"pUpA",g);this.dispatch(pass,"pUp0",g);
  }
  this.dispatch(pass,"pProject",g);
  // Leaving slots return to the stack (after every reader of this frame).
  this.dispatch(pass,"ubaRelease",Math.min(groupsOf(this.arena.fineCapacity,64),TABLE_GROUPS));
  pass.end();
 }
 encodeReceipt(encoder:GPUCommandEncoder,target:GPUBuffer,offset:number):void{encoder.copyBufferToBuffer(this.index,0,target,offset,32);}
 static readonly historyWord=HISTORY_WORD;
 static readonly initialResidualWord=INITIAL_WORD;
 static readonly residualValidityWord=VALID_WORD;
 static readonly initialResidualBit=1<<HISTORY;
 static readonly closedWord=CLOSED_WORD;
 static readonly bricksWord=BRICKS_WORD;
 /** The overflow word's value when more band pages reach the coarsest level than it stages. */
 static readonly coarsestOverflow=2;
 static readonly coarsestPages=TAIL_PAGES;
 destroy():void{this.index.destroy();this.pool.destroy();}
}
