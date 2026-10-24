import { uniformWindowAddressWGSL } from "./uniform-window-address";
import {uniformRuntimeConstantsWGSL} from "./uniform-scene-constants";
import { uniformCompileStencilWGSL } from "./uniform-compiled-topology";
import {UNIFORM_MIXED_OVERFLOW_FINE,UNIFORM_MIXED_OVERFLOW_HANGING} from "./uniform-mixed-topology.wgsl";
import type {UniformMixedGenerationBuffers,UniformMixedOwnership} from "./uniform-mixed-ownership";
import {UNIFORM_PAGE_TABLE_WORDS,uniformLatticeAddressWGSL} from "./uniform-lattice-address";
import type {UniformPagePool} from "./uniform-page-pool";
import {UNIFORM_LIVE_GROUP_GROUPS,UNIFORM_LIVE_KEY_GROUPS,uniformLiveListsWGSL} from "./uniform-live-lists";

const BLOCK=256;
/** Work header: the cleared counters [0, CLEARED), then umCounts at COUNTS
 * (256 B aligned: the remap binds them as its target's uniform counts). */
const CLEARED=16,COUNTS=64,RECEIPT=COUNTS+12;
/** Scan categories: h tiles, 4h tiles (one coarse owner each), seam h and 4h
 * tiles, regular 4h tiles. The simulation layout is ungraded h/4h. */
const CATEGORIES=5;
/** Work word counting verifyWords groups (low 16 bits) and groups that
 * found a bad tile word (high 16 bits); the last group seals the build. */
const TICKET=6;
/** Work word counting the h tiles the band and statics ask for (widths). */
const NEED=7;

/** Word layout of a GPU-built generation's changed tiles in the builder's
 * work buffer (UniformMixedGenerationBuffers.changes) for `tiles` tiles.
 * changed: tiles whose width differs from the ownership the generation was
 * built against (count word, then the list); dilated: every tile within
 * Chebyshev tile distance one of a changed tile, each once (the union of
 * their 3x3x3 neighbourhoods). Both lists are unordered and bounded by the
 * tile count; the counts are complete once the builder's encode has run. */
export function uniformMixedChangedTilesWords(tiles:number){
 const changedList=RECEIPT+tiles+CATEGORIES*Math.ceil(tiles/BLOCK);
 return {changedCount:0,dilatedCount:5,changedList,dilatedList:changedList+tiles,words:changedList+2*tiles} as const;
}
/** Read-only access to the changed-tile lists (uniformMixedChangedTilesWords)
 * at @group(group) @binding(binding): umChangedCount/umChangedTile and
 * umDilatedCount/umDilatedTile. */
export function uniformMixedChangedTilesWGSL(tiles:number,group:number,binding:number):string{
 void tiles;
 return /* wgsl */`
@group(${group}) @binding(${binding}) var<storage,read> umChanges:array<u32>;
fn umChangedList()->u32{let n=umChanges[8u];return 76u+n+5u*((n+255u)/256u);}
fn umChangedCount()->u32{return umChanges[0u];}
fn umChangedTile(i:u32)->u32{return umChanges[umChangedList()+i];}
fn umDilatedCount()->u32{return umChanges[5u];}
fn umDilatedTile(i:u32)->u32{return umChanges[umChangedList()+umChanges[8u]+i];}
`;
}

/** Where the fine band lives: one bit per tile from `wordOffset` words into
 * `buffer`. headerWords: words [0, headerWords) of `buffer` are the
 * producer's receipt header (the census's), carried into the builder receipt. */
export interface UniformMixedBandBits {readonly buffer:GPUBuffer;readonly wordOffset:number;readonly headerWords?:number;
 /** The producer's residency audit word (UNIFORM_MIXED_RESIDENCY_AUDIT): non-zero raises the residency fatal. */
 readonly auditWord?:number}

/** An authored region as a box of tiles of the 512-tile window: min inclusive,
 * max exclusive. fine: its tiles are h whatever the band holds. coarse: the
 * band holds none of its tiles that no fine region holds, but for the ones a
 * solid promotes (the census's decide and solidPromote). */
export interface UniformMixedRegion {readonly min:readonly [number,number,number];readonly max:readonly [number,number,number];readonly kind:"fine"|"coarse"}
export const UNIFORM_MIXED_REGION_LIMIT=32;
/** The region table the builder and the census hold: the count and three
 * spare words, then eight words a region (min x, y, z; max x, y, z; kind, 1
 * fine or 2 coarse; spare). */
export const UNIFORM_MIXED_REGION_TABLE_WORDS=4+8*UNIFORM_MIXED_REGION_LIMIT;
export function uniformMixedRegionTable(regions:readonly UniformMixedRegion[]):Uint32Array<ArrayBuffer>{
 if(regions.length>UNIFORM_MIXED_REGION_LIMIT)throw new RangeError(`Uniform mixed regions: ${regions.length} authored, the region table holds ${UNIFORM_MIXED_REGION_LIMIT}`);
 const table=new Uint32Array(UNIFORM_MIXED_REGION_TABLE_WORDS);table[0]=regions.length;
 regions.forEach((r,i)=>{
  const kind=r.kind==="fine"?1:r.kind==="coarse"?2:0;
  if(!kind||r.min.length!==3||r.max.length!==3||[0,1,2].some(a=>!Number.isInteger(r.min[a])||!Number.isInteger(r.max[a])||r.min[a]!<0||r.max[a]!>512||r.min[a]!>=r.max[a]!))
   throw new RangeError(`Uniform mixed regions: region ${i} (${String(r.kind)}, ${r.min.join(",")} to ${r.max.join(",")}) is not a fine or coarse box of tiles of the 512-tile window`);
  table.set([r.min[0],r.min[1],r.min[2],r.max[0],r.max[1],r.max[2],kind,0],4+8*i);
 });
 return table;
}
/** fn name(c:vec3u)->u32 over a region table whose word i is word(i): bit 0
 * where a fine region holds tile c, bit 1 where a coarse one does. An empty
 * table costs its count word. */
export function uniformMixedRegionWGSL(name:string,word:(index:string)=>string):string{
 return /* wgsl */`fn ${name}(local:vec3u)->u32{
 let c=vec3i(local)+8*umFramePage();
 var kinds=0u;let n=${word("0u")};
 for(var i=0u;i<n;i++){
  let e=4u+8u*i;
  if(all(c>=bitcast<vec3i>(vec3u(${word("e")},${word("e+1u")},${word("e+2u")})))&&all(c<bitcast<vec3i>(vec3u(${word("e+3u")},${word("e+4u")},${word("e+5u")})))){kinds|=${word("e+6u")};}
 }
 return kinds;
}`;
}

/** Word offsets of the builder's compact relayout receipt
 * (UniformMixedLayoutBuilder.receipt): what a frame receipt copies instead
 * of the tile words. census: the census header (20 words, see
 * UniformMixedDynamicCensus / uniform-mixed-dynamic.ts HEADER), written by
 * each encode. changed: tiles whose width changed; tiers: h and 4h tiles;
 * seams: seam h and seam 4h tiles (their sum is the hanging slots); counts:
 * umCounts (h, 4h, 0, 8); need: the h tiles the band and statics asked for,
 * and admit: the admission capacity it was held to (all ones: none); a need
 * over it was deferred, and the build kept the generation it was against
 * (tiers, seams and counts are that generation's); all per build.
 * Persistent: generation, bumped by every build that changed a tile and
 * raised no fatal flag; builds, bumped by every build; fatal, sticky
 * UNIFORM_MIXED_RELAYOUT_FATAL bits; fatalBuild, the build number that first
 * raised one. */
export const UNIFORM_MIXED_RELAYOUT_RECEIPT={census:0,changed:20,tiers:21,seams:23,need:25,admit:26,audit:27,counts:28,generation:32,builds:33,fatal:34,fatalBuild:35,words:36} as const;
/** Sticky fatal bits of the relayout receipt, validated on the GPU. A set
 * bit means the built generation must not be adopted or advanced on.
 * hangingCapacity (= UNIFORM_MIXED_OVERFLOW_HANGING): seam tiles exceed the
 * preallocated hanging tap cache (ownership.capacity.hangingSlots); slots
 * past it are left unslotted. tierSum: h + 4h tiles != the tiles of the
 * blocks that hold a page (counted from the block words). tileWords: a
 * tile word, owner index or worklist entry disagrees with the receipt counts
 * (the host's cellCount check). residency: the band producer's residency
 * audit found liquid or near-surface phi in a page the last frame skipped as
 * absent, or a reader left the resident closure (uniformMixedResidencyWord).
 * fineCapacity (= UNIFORM_MIXED_OVERFLOW_FINE): h tiles exceed what the
 * owner-indexed buffers hold (ownership.capacity.fineTiles). Under an
 * admission capacity (setAdmission) a build over it is deferred instead, so
 * the bit then means the kept generation itself is over capacity.
 * The remap latches a set bit into the frame status (markListed). */
export const UNIFORM_MIXED_RELAYOUT_FATAL={hangingCapacity:UNIFORM_MIXED_OVERFLOW_HANGING,tierSum:2,tileWords:4,residency:8,fineCapacity:UNIFORM_MIXED_OVERFLOW_FINE} as const;

interface Level {
 readonly current:UniformMixedOwnership;
 readonly topology:GPUBuffer;readonly support:GPUBuffer;readonly slots:GPUBuffer;readonly work:GPUBuffer;
 readonly params:GPUBuffer;readonly group:GPUBindGroup;
 /** The compact relayout receipt (UNIFORM_MIXED_RELAYOUT_RECEIPT). */
 readonly status:GPUBuffer;
}

/** GPU ownership builder for dynamic coarsening (docs/plans/uniform-dynamic-coarsening.md,
 * phase 4). From the band bits and the fine regions of its region table, it
 * writes exactly the buffers UniformMixedOwnership.update uploads for the
 * ungraded h/4h createUniformMixedLayout(lattice, regions, 4, fine ∪ band).
 * Every launch is a fixed grid over the lattice's tiles;
 * the receipt checks (tier sum, tile words, hanging capacity) run on the GPU
 * into sticky fatal bits of a compact receipt (UNIFORM_MIXED_RELAYOUT_RECEIPT)
 * with a GPU generation counter. There is no host mirror: the frame head
 * remaps on the built buffers and UniformMixedOwnership.adoptGpu copies
 * them; the receipt is the GPU's to consume (diagnostics copy it). */
export class UniformMixedLayoutBuilder {
 readonly allocatedBytes:number;
 /** Tile keys. */
 private readonly tiles:number;
 private readonly blocks:number;
 /** The region table (uniformMixedRegionTable). */
 private readonly statics:GPUBuffer;
 private readonly resources:GPUBindGroupLayout;
 private readonly level:Level;
 private readonly pipelines=new Map<string,GPUComputePipeline>();
 private staticReady=false;
 /** The band producer's receipt header words (UniformMixedBandBits.headerWords),
  * copied from the bound band buffer by scan. */
 private readonly headerWords:number=0;
 /** The compact relayout receipt: copy UNIFORM_MIXED_RELAYOUT_RECEIPT.words
  * words from `offset` bytes. Per-build words are complete once this
  * builder's encode has run; generation, builds and the fatal bits persist. */
 get receipt():{readonly buffer:GPUBuffer;readonly offset:number;readonly words:number}{return this.receiptView??={buffer:this.level.status,offset:0,words:UNIFORM_MIXED_RELAYOUT_RECEIPT.words};}
 private receiptView?:{readonly buffer:GPUBuffer;readonly offset:number;readonly words:number};
 /** The built generation's buffers, complete once encode has run: what
  * UniformMixedOwnership.adoptGpu copies, with no host mirror. */
 get generation():UniformMixedGenerationBuffers{const l=this.level;return this.generationView??={topology:l.topology,support:l.support,slots:l.slots,counts:{buffer:l.work,offset:COUNTS*4},changes:l.work};}
 private generationView?:UniformMixedGenerationBuffers;
 /** One word for a params write (the queue copies it at the call). */
 private readonly word=new Uint32Array(1);
 /** encode's launches in order (admit only under an admission capacity), resolved once. */
 private launches?:readonly GPUComputePipeline[];
 private static readonly PASS:GPUComputePassDescriptor={label:"Uniform mixed layout build"};
 /** ownership: the simulation ownership the built generation replaces.
  * pool: the frame's page pool, whose resident list the per-key launches
  * stride (uniform-live-lists). */
 constructor(private readonly device:GPUDevice,band:UniformMixedBandBits,ownership:UniformMixedOwnership,private readonly pool:UniformPagePool){
  const n=ownership.capacity.tiles;
  if(n!==512*pool.layout.blocks)throw new Error(`Mixed layout builder: ${n} tile keys are not the page pool's ${pool.layout.blocks} blocks`);
  this.tiles=n;this.blocks=Math.ceil(n/BLOCK);
  // The per-tile launches are one dimension of 64-lane groups, and the verify ticket counts them in 16 bits: 8,191 pages of 512 tiles at 65,535 groups.
  const groups=Math.ceil(n/64),most=Math.min(0xffff,device.limits.maxComputeWorkgroupsPerDimension);
  if(groups>most)throw new Error(`Mixed layout builder launches ${groups} groups of 64 tile keys in one dimension, above ${most}: ${n} tile keys exceed ${64*most} (${Math.floor(most/8)} pages of 512)`);
  const storage=(label:string,words:number)=>device.createBuffer({label,size:words*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
  this.statics=storage("Uniform layout builder region table",UNIFORM_MIXED_REGION_TABLE_WORDS);
  const R=UNIFORM_MIXED_RELAYOUT_RECEIPT;
  if(band.headerWords!==undefined){
   if(!Number.isSafeInteger(band.headerWords)||band.headerWords<0||band.headerWords>R.changed-R.census)throw new Error(`Band header of ${band.headerWords} words does not fit the relayout receipt`);
   this.headerWords=band.headerWords;
  }
  this.resources=device.createBindGroupLayout({entries:[
   ...[0,1,2].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage" as const}})),
   {binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},
   ...[4,5,6,7,8].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage" as const}})),
   {binding:9,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},
  ]});
  let bytes=this.statics.size;
  const topology=storage("Uniform layout builder topology",5*n);
  const support=storage("Uniform layout builder support",9*n+24);
  const slots=storage("Uniform layout builder slots",2*n);
  const work=device.createBuffer({label:"Uniform layout builder work",size:uniformMixedChangedTilesWords(n).words*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
  const params=device.createBuffer({label:"Uniform layout builder params",size:24,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
  // Hanging capacity: the ownership's preallocated tap cache.
  // Residency audit word, or none. h-tile capacity: the owner-indexed buffers'.
  // Admission capacity: none (setAdmission).
  const lattice=ownership.capacity.lattice;
  device.queue.writeBuffer(params,0,new Uint32Array([band.wordOffset,ownership.capacity.hangingSlots,band.auditWord??0xffffffff,ownership.capacity.fineTiles,0xffffffff,n]));
  device.queue.writeBuffer(work,COUNTS*4+28,new Uint32Array([n]));
  device.queue.writeBuffer(work,COUNTS*4+32,new Float32Array([...lattice.cellSize_m,Math.min(...lattice.cellSize_m)]));
  const status=device.createBuffer({label:"Uniform layout builder relayout receipt",size:R.words*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
  // Standalone stages visit everything until a frame census: support[0,4n) = 3.
  device.queue.writeBuffer(support,0,new Uint32Array(4*n).fill(3));
  const group=device.createBindGroup({layout:this.resources,entries:[
   {binding:0,resource:{buffer:band.buffer}},{binding:1,resource:{buffer:this.statics}},{binding:2,resource:ownership.presentation},
   {binding:3,resource:{buffer:params}},{binding:4,resource:{buffer:work}},{binding:5,resource:{buffer:topology}},
   {binding:6,resource:{buffer:support}},{binding:7,resource:{buffer:slots}},{binding:8,resource:{buffer:status}},
   {binding:9,resource:{buffer:pool.state}},
  ]});
  bytes+=topology.size+support.size+slots.size+work.size+params.size+status.size;
  this.level={current:ownership,topology,support,slots,work,params,group,status};
  this.capacityRevision=ownership.capacityRevision;
  this.allocatedBytes=bytes;
 }
 async initialize():Promise<void>{
  const n=this.tiles,x=this.level.current.dispatchX;
  const flags=RECEIPT,totals=RECEIPT+n,lists=uniformMixedChangedTilesWords(n);
  const R=UNIFORM_MIXED_RELAYOUT_RECEIPT,F=UNIFORM_MIXED_RELAYOUT_FATAL;
  const module=this.device.createShaderModule({label:"Uniform mixed layout builder",code:uniformRuntimeConstantsWGSL(/* wgsl */`
@group(0) @binding(0) var<storage,read> band:array<u32>;
@group(0) @binding(1) var<storage,read> statics:array<u32>;
@group(0) @binding(2) var<storage,read> current:array<u32>;
struct Params {bandOffset:u32,hangingCapacity:u32,audit:u32,fineCapacity:u32,admitCapacity:u32,tileKeys:u32}
@group(0) @binding(3) var<uniform> params:Params;
@group(0) @binding(4) var<storage,read_write> work:array<atomic<u32>>;
@group(0) @binding(5) var<storage,read_write> topology:array<u32>;
@group(0) @binding(6) var<storage,read_write> support:array<u32>;
@group(0) @binding(7) var<storage,read_write> slots:array<u32>;
// The compact relayout receipt (UNIFORM_MIXED_RELAYOUT_RECEIPT).
@group(0) @binding(8) var<storage,read_write> status:array<atomic<u32>>;
// The pool's live lists: widths, admit, classify, mirror, scatter and
// verifyWords stride its resident blocks (a free tile key writes nothing).
${uniformLiveListsWGSL(0,9,this.pool.layout)}
const R_CHANGED:u32=${R.changed}u;const R_TIERS:u32=${R.tiers}u;const R_SEAMS:u32=${R.seams}u;const R_COUNTS:u32=${R.counts}u;
const R_NEED:u32=${R.need}u;const R_ADMIT:u32=${R.admit}u;const NEED:u32=${NEED}u;
const R_GENERATION:u32=${R.generation}u;const R_BUILDS:u32=${R.builds}u;const R_FATAL:u32=${R.fatal}u;const R_FATAL_BUILD:u32=${R.fatalBuild}u;
const FATAL_TIER_SUM:u32=${F.tierSum}u;const FATAL_TILE_WORDS:u32=${F.tileWords}u;const FATAL_HANGING:u32=${F.hangingCapacity}u;const FATAL_RESIDENCY:u32=${F.residency}u;const FATAL_OWNERS:u32=${F.fineCapacity}u;
const N:u32=${n}u;const X:u32=${x}u;
const BLOCKS:u32=${this.blocks}u;const INF:u32=0xffffffffu;
const FLAGS:u32=${flags}u;const TOTALS:u32=${totals}u;const COUNTS:u32=${COUNTS}u;
const CHANGED:u32=${lists.changedList}u;const DILATED:u32=${lists.dilatedList}u;const DILATED_COUNT:u32=${lists.dilatedCount}u;
// The page table and block records the live ownership holds after its tile words.
${uniformWindowAddressWGSL("current","5u*N","(N/512u)")}
${uniformLatticeAddressWGSL({constant:true,paged:{blocks:"(N/512u)"}})}
${uniformMixedRegionWGSL("regionsAt",i=>`statics[${i}]`)}
// The band is a bit per tile key, and holds no tile of a coarse region but a
// solid's (the census's decide). A tile of a fine region is h without it.
fn fineAt(t:u32)->bool{
 if(((band[params.bandOffset+t/32u]>>(t%32u))&1u)!=0u){return true;}
 return (regionsAt(umTileCoord(t))&1u)!=0u;
}
fn wordWidth(word:u32)->u32{return select(4u,1u,(word&0x80000000u)!=0u);}
fn widthAt(t:u32)->u32{return atomicLoad(&work[FLAGS+t])&7u;}
// Width: h for the band and static tiles, else 4h (ungraded). Staged as a
// plain word in the tile-word plane, which scatter overwrites, so classify's
// 27 neighbour reads are plain loads.
@compute @workgroup_size(64) fn widths(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) groups:vec3u){
 if(gid.x==0u){atomicStore(&work[8u],N);}
 for(var visit=gid.x;visit<512u*ulResidentCount();visit+=64u*groups.x){widthOf(512u*ulResidentBlock(visit>>9u)+(visit&511u));}
}
fn widthOf(t:u32){
 if(!umTileLive(t)){return;}
 let w=select(4u,1u,fineAt(t));
 topology[t]=w;
 if(w==1u){atomicAdd(&work[NEED],1u);}
 if(w!=wordWidth(current[t])){atomicStore(&work[CHANGED+atomicAdd(&work[0],1u)],t);}
}
// Count, then admit: the h tiles this build asks for are known before any
// tile word is written. Over the admission capacity the build keeps the
// generation it is against: the staged widths return to it and no tile is
// changed, so nothing is remapped and the generation does not advance. The
// receipt carries the need; the host grows the capacity between frames.
@compute @workgroup_size(64) fn admit(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) groups:vec3u){
 if(atomicLoad(&work[NEED])<=params.admitCapacity){return;}
 if(gid.x==0u){atomicStore(&work[0],0u);}
 for(var visit=gid.x;visit<512u*ulResidentCount();visit+=64u*groups.x){
  let t=512u*ulResidentBlock(visit>>9u)+(visit&511u);
  if(umTileLive(t)){topology[t]=wordWidth(current[t]);}
 }
}
// The dilated list: each changed tile's 3x3x3 neighbourhood, deduplicated by
// flag bit 16 (classify rewrote the flags; every later reader masks the
// width and regular bits).
// A fixed grid strides the list.
@compute @workgroup_size(64) fn dilate(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) groups:vec3u){
 let changed=atomicLoad(&work[0]);
 for(var i=gid.x;i<changed;i+=64u*groups.x){dilateOf(atomicLoad(&work[CHANGED+i]));}
}
fn dilateOf(t:u32){
 let p=vec3i(umTileCoord(t));
 for(var z=-1;z<=1;z++){for(var y=-1;y<=1;y++){for(var x=-1;x<=1;x++){
  let q=p+vec3i(x,y,z);if(!umTilePresent(q)){continue;}
  let s=umTileAt(vec3u(q));
  if((atomicOr(&work[FLAGS+s],16u)&16u)==0u){atomicStore(&work[DILATED+atomicAdd(&work[DILATED_COUNT],1u)],s);}
 }}}
}
var<workgroup> blockTotals:array<atomic<u32>,${CATEGORIES}>;
fn categories(w:u32,regular:bool)->array<u32,${CATEGORIES}>{
 var c:array<u32,${CATEGORIES}>;
 c[0]=select(0u,1u,w==1u);c[1]=select(0u,1u,w==4u);
 c[2]=select(0u,c[0],!regular);c[3]=select(0u,c[1],!regular);c[4]=select(0u,c[1],regular);
 return c;
}
${uniformCompileStencilWGSL}
// Frozen 3x3x3 stencil masks, compiled recipes and per-block category totals.
// A fixed grid strides the resident blocks' groups of ${BLOCK} keys (two a
// block): a free block's totals are neither written nor read (scan skips it).
@compute @workgroup_size(${BLOCK}) fn classify(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 for(var visit=group.x;visit<2u*ulResidentCount();visit+=groups.x){classifyGroup(2u*ulResidentBlock(visit>>1u)+(visit&1u),lane);workgroupBarrier();}
}
fn classifyGroup(g:u32,lane:u32){
 if(lane<${CATEGORIES}u){atomicStore(&blockTotals[lane],0u);}
 workgroupBarrier();
 let t=g*${BLOCK}u+lane;
 if(umTileLive(t)){
  let w=topology[t];let p=vec3i(umTileCoord(t));
  var maximum=w;var minimum=w;var fine=0u;var valid=0u;
  for(var z=-1;z<=1;z++){for(var y=-1;y<=1;y++){for(var x=-1;x<=1;x++){
   let q=p+vec3i(x,y,z);if(!umTilePresent(q)){continue;}
   valid|=1u<<u32((x+1)+3*((y+1)+3*(z+1)));
   let v=topology[umTileAt(vec3u(q))];maximum=max(maximum,v);minimum=min(minimum,v);
   if(v==1u){fine|=1u<<u32((x+1)+3*((y+1)+3*(z+1)));}
  }}}
  var farPositive=0u;
  for(var axis=0u;axis<3u;axis++){var q=p;q[axis]+=2;if(umTilePresent(q)&&topology[umTileAt(vec3u(q))]==1u){farPositive|=1u<<axis;}}
  topology[2u*N+2u*t]=fine|(maximum<<27u);topology[2u*N+2u*t+1u]=(minimum<<27u)|(farPositive<<24u)|umCompileStencil(fine,valid);
  topology[4u*N+t]=umCompileBlendMask(fine);
  let regular=maximum==minimum;
  atomicStore(&work[FLAGS+t],w|select(0u,8u,regular));
  let c=categories(w,regular);
  for(var k=0u;k<4u;k++){if(c[k]!=0u){atomicAdd(&blockTotals[k],1u);}}
 }
 workgroupBarrier();
 // Regular 4h tiles are the block's non-seam 4h: a key that holds no tile counts nowhere.
 if(lane==0u){atomicStore(&blockTotals[4],atomicLoad(&blockTotals[1])-atomicLoad(&blockTotals[3]));}
 workgroupBarrier();
 if(lane<${CATEGORIES}u){atomicStore(&work[TOTALS+g*${CATEGORIES}u+lane],atomicLoad(&blockTotals[lane]));}
}
var<workgroup> partial:array<u32,${BLOCK}>;
var<workgroup> held:array<u32,${BLOCK}>;
var<workgroup> grand:array<u32,${CATEGORIES}>;
var<workgroup> stored:atomic<u32>;
// Inclusive Hillis-Steele scan of partial[]; every lane must call it.
fn scanPartial(lane:u32){
 for(var offset=1u;offset<${BLOCK}u;offset*=2u){
  var v=0u;if(lane>=offset){v=partial[lane-offset];}
  workgroupBarrier();partial[lane]+=v;workgroupBarrier();
 }
}
// The same scan of partial[] and held[] together.
fn scanPartialHeld(lane:u32){
 for(var offset=1u;offset<${BLOCK}u;offset*=2u){
  var v=0u;var u=0u;if(lane>=offset){v=partial[lane-offset];u=held[lane-offset];}
  workgroupBarrier();partial[lane]+=v;held[lane]+=u;workgroupBarrier();
 }
}
// Exclusive block offsets per category, grand totals, and the headers.
@compute @workgroup_size(${BLOCK}) fn scan(@builtin(local_invocation_index) lane:u32){
${this.headerWords?`// The band producer's receipt header, carried into this build's receipt.
 if(lane<${this.headerWords}u){atomicStore(&status[${R.census}u+lane],band[lane]);}`:""}
 // The stored tiles: 512 a block that holds a page, wherever the page lies.
 if(lane==0u){atomicStore(&stored,0u);}
 workgroupBarrier();
 var tiles=0u;
 for(var b=lane;b<UM_BLOCKS;b+=${BLOCK}u){
  if(umBlockWord(b)==UM_FREE){continue;}
  tiles+=512u;
 }
 atomicAdd(&stored,tiles);
 let per=(BLOCKS+${BLOCK-1}u)/${BLOCK}u;let first=min(BLOCKS,lane*per);let last=min(BLOCKS,first+per);
 for(var k=0u;k<${CATEGORIES}u;k++){
  // A free block's groups hold no tile (classify left their totals).
  var local=0u;for(var b=first;b<last;b++){if(umBlockWord(b>>1u)==UM_FREE){continue;}local+=atomicLoad(&work[TOTALS+b*${CATEGORIES}u+k]);}
  partial[lane]=local;workgroupBarrier();scanPartial(lane);
  var running=partial[lane]-local;
  for(var b=first;b<last;b++){if(umBlockWord(b>>1u)==UM_FREE){continue;}let i=TOTALS+b*${CATEGORIES}u+k;let c=atomicLoad(&work[i]);atomicStore(&work[i],running);running+=c;}
  if(lane==${BLOCK-1}u){grand[k]=partial[lane];}
  workgroupBarrier();
 }
 if(lane==0u){
  let f=grand[0];let c=grand[1];
  // Receipt: [0] changed tiles, [1] h tiles, [2] 4h tiles, [3] seam h,
  // [4] seam 4h, [5] dilated tiles (dilate), [6] the verifyWords ticket,
  // [7] the h tiles asked for (widths); [8,16) stay cleared.
  atomicStore(&work[1],f);atomicStore(&work[2],c);atomicStore(&work[3],grand[2]);atomicStore(&work[4],grand[3]);
  // umCounts (h, 4h, 0, loop bound), then the frame-plan header (update(): header[2], [8..10], [12..14]).
  atomicStore(&work[COUNTS],f);atomicStore(&work[COUNTS+1u],c);atomicStore(&work[COUNTS+2u],0u);atomicStore(&work[COUNTS+3u],8u);
  let merged=f+grand[3]+(grand[4]+63u)/64u;
  let h=4u*N;
  for(var i=0u;i<16u;i++){support[h+i]=0u;}
  support[h+2u]=f;
  support[h+8u]=min(f,X);support[h+9u]=(f+X-1u)/X;support[h+10u]=1u;
  support[h+12u]=min(merged,X);support[h+13u]=(merged+X-1u)/X;support[h+14u]=1u;
  let seam=7u*N+16u;support[seam]=grand[2];support[seam+1u]=grand[3];support[seam+2u]=0u;support[seam+3u]=0u;
  let regular=8u*N+20u;support[regular]=grand[4];support[regular+1u]=0u;support[regular+2u]=0u;support[regular+3u]=0u;
  // The per-build receipt words, then its checks: fatal bits are sticky.
  atomicStore(&status[R_CHANGED],atomicLoad(&work[0]));atomicStore(&status[R_TIERS],f);atomicStore(&status[R_TIERS+1u],c);
  atomicStore(&status[R_SEAMS],grand[2]);atomicStore(&status[R_SEAMS+1u],grand[3]);
  for(var i=R_SEAMS+2u;i<R_COUNTS;i++){atomicStore(&status[i],0u);}
  atomicStore(&status[R_NEED],atomicLoad(&work[NEED]));atomicStore(&status[R_ADMIT],params.admitCapacity);
  atomicStore(&status[R_COUNTS],f);atomicStore(&status[R_COUNTS+1u],c);atomicStore(&status[R_COUNTS+2u],0u);atomicStore(&status[R_COUNTS+3u],8u);
  if(f+c!=atomicLoad(&stored)){atomicOr(&status[R_FATAL],FATAL_TIER_SUM);}
  if(grand[2]+grand[3]>params.hangingCapacity){atomicOr(&status[R_FATAL],FATAL_HANGING);}
  // Known before scatter writes a tile word: every owner-indexed buffer
  // is sized for at most this many h tiles. A need over an admission
  // capacity was deferred (admit); this is the generation kept instead.
  if(f>params.fineCapacity){atomicOr(&status[R_FATAL],FATAL_OWNERS);}
  var audit=0u;if(params.audit!=0xffffffffu){audit=band[params.audit];}
  atomicStore(&status[${UNIFORM_MIXED_RELAYOUT_RECEIPT.audit}u],audit);
  if(audit!=0u){atomicOr(&status[R_FATAL],FATAL_RESIDENCY);}
 }
}
// Owner numbering, worklists and the hanging slot table, in tile key order.
// A fixed grid: a slot no seam tile takes names no tile (every slot of the
// table's N, whatever its key holds), then the resident blocks' groups.
@compute @workgroup_size(${BLOCK}) fn scatter(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 let seamF=atomicLoad(&work[3]);let seamC=atomicLoad(&work[4]);
 for(var s=min(seamF+seamC,params.hangingCapacity)+group.x*${BLOCK}u+lane;s<N;s+=${BLOCK}u*groups.x){slots[N+s]=INF;}
 for(var visit=group.x;visit<2u*ulResidentCount();visit+=groups.x){scatterGroup(2u*ulResidentBlock(visit>>1u)+(visit&1u),lane,seamF);workgroupBarrier();}
}
fn scatterGroup(g:u32,lane:u32,seamF:u32){
 let t=g*${BLOCK}u+lane;let valid=umTileLive(t);
 var w=0u;var regular=false;
 if(valid){let f=atomicLoad(&work[FLAGS+t]);w=f&7u;regular=(f&8u)!=0u;}
 let c=categories(w,regular);
 // One scan of h, seam h and seam 4h packed in 10-bit fields (a block sums
 // at most ${BLOCK}), and one of the keys that hold a tile: a tile's 4h rank
 // is the tiles below it minus its h rank, its regular rank that minus its
 // seam 4h.
 partial[lane]=c[0]|(c[2]<<10u)|(c[3]<<20u);held[lane]=u32(valid);workgroupBarrier();scanPartialHeld(lane);
 if(valid){scatterTile(t,g,w,regular,c,partial[lane],held[lane],seamF);}
}
// A stored tile of the group: packed and heldBelow are its inclusive scans.
fn scatterTile(t:u32,g:u32,w:u32,regular:bool,c:array<u32,${CATEGORIES}>,packed:u32,heldBelow:u32,seamF:u32){
 var rank:array<u32,${CATEGORIES}>;
 rank[0]=(packed&1023u)-c[0];rank[2]=((packed>>10u)&1023u)-c[2];rank[3]=(packed>>20u)-c[3];
 rank[1]=heldBelow-1u-rank[0];rank[4]=rank[1]-rank[3];
 for(var k=0u;k<${CATEGORIES}u;k++){rank[k]+=atomicLoad(&work[TOTALS+g*${CATEGORIES}u+k]);}
 let f=atomicLoad(&work[1]);
 var slot=INF;
 // Slots past the preallocated cache stay unslotted (scan raised the fatal bit).
 if(w==1u){
  topology[t]=(rank[0]*64u)|0x80000000u;topology[N+rank[0]]=t;support[6u*N+16u+rank[0]]=t;
  if(!regular){support[7u*N+20u+rank[2]]=t;slot=rank[2];}
 }else{
  topology[t]=f*64u+rank[1];topology[N+f+rank[1]]=t;
  if(regular){support[8u*N+24u+rank[4]]=t;}else{support[7u*N+20u+seamF+rank[3]]=t;slot=seamF+rank[3];}
 }
 if(slot>=params.hangingCapacity){slot=INF;}
 slots[t]=slot;
 if(slot!=INF){slots[N+slot]=t;}
}
// The detail ring (UNIFORM_MIXED_DETAIL_RING): bit 0 of the second stencil
// word, an h tile within three tiles, from classify's 27-bit h masks of the
// tiles two apart. Each lane writes its own word.
@compute @workgroup_size(64) fn mirror(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) groups:vec3u){
 for(var visit=gid.x;visit<512u*ulResidentCount();visit+=64u*groups.x){mirrorOf(512u*ulResidentBlock(visit>>9u)+(visit&511u));}
}
fn mirrorOf(t:u32){
 if(!umTileLive(t)){return;}
 let p=vec3i(umTileCoord(t));var near=false;
 for(var z=-2;z<=2;z+=2){for(var y=-2;y<=2;y+=2){for(var x=-2;x<=2;x+=2){
  let q=p+vec3i(x,y,z);if(!umTilePresent(q)){continue;}
  near=near||(topology[2u*N+2u*umTileAt(vec3u(q))]&0x7ffffffu)!=0u;
 }}}
 if(near){topology[2u*N+2u*t+1u]|=1u;}
}
// verifyWords checks the scattered tile word against its width and the
// receipt: an h tile's owner base is 64 x its rank below the h count, a 4h
// tile's owner lies in [64f, 64f + c), and its tier list entry names it.
var<workgroup> groupBad:atomic<u32>;
// A fixed grid of VERIFY_GROUPS workgroups, a workgroup a resident z-slab at a time.
const VERIFY_GROUPS:u32=${Math.min(UNIFORM_LIVE_GROUP_GROUPS,Math.ceil(n/64))}u;
@compute @workgroup_size(64) fn verifyWords(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 if(lane==0u){atomicStore(&groupBad,0u);}
 workgroupBarrier();
 for(var slab=group.x;slab<8u*ulResidentCount();slab+=groups.x){
 let t=512u*ulResidentBlock(slab>>3u)+64u*(slab&7u)+lane;
 if(umTileLive(t)){
  let word=topology[t];let f=atomicLoad(&work[1]);let c=atomicLoad(&work[2]);var ok=wordWidth(word)==widthAt(t)&&(word&0x40000000u)==0u;
  if((word&0x80000000u)!=0u){let base=word&0x3fffffffu;let rank=base/64u;ok=ok&&base%64u==0u&&rank<f&&topology[N+rank]==t;}
  else{ok=ok&&word>=64u*f&&word-64u*f<c&&topology[N+f+word-64u*f]==t;}
  if(!ok){atomicStore(&groupBad,1u);}
 }
 }
 workgroupBarrier();
 if(lane!=0u){return;}
 // Each group's verdict and its ticket are one RMW on one word, so the
 // last group's read holds every verdict. It seals the build once every
 // check has run (scan, verifyWords): number the build, latch the first
 // fatal one, else advance the generation if a tile changed.
 let bad=atomicLoad(&groupBad);
 let ticket=atomicAdd(&work[${TICKET}u],1u+(bad<<16u))+1u+(bad<<16u);
 if((ticket&0xffffu)!=VERIFY_GROUPS){return;}
 if((ticket>>16u)!=0u){atomicOr(&status[R_FATAL],FATAL_TILE_WORDS);}
 let build=atomicAdd(&status[R_BUILDS],1u)+1u;
 if(atomicLoad(&status[R_FATAL])!=0u){if(atomicLoad(&status[R_FATAL_BUILD])==0u){atomicStore(&status[R_FATAL_BUILD],build);}}
 else if(atomicLoad(&status[R_CHANGED])!=0u){atomicAdd(&status[R_GENERATION],1u);}
}
`,{
 N:{type:"u32",value:"params.tileKeys"},
 BLOCKS:{type:"u32",value:"(N+255u)/256u"},
 TOTALS:{type:"u32",value:"76u+N"},
 CHANGED:{type:"u32",value:"76u+N+5u*BLOCKS"},
 DILATED:{type:"u32",value:"CHANGED+N"},
 VERIFY_GROUPS:{type:"u32",value:`min(${UNIFORM_LIVE_GROUP_GROUPS}u,(N+63u)/64u)`},
 })});
  const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");
  if(errors.length)throw new Error(errors.map(m=>`${m.lineNum}: ${m.message}`).join("\n"));
  const layout=this.device.createPipelineLayout({bindGroupLayouts:[this.resources]});
  await Promise.all(["widths","admit","classify","mirror","dilate","scan","scatter","verifyWords"].map(async entryPoint=>{this.pipelines.set(entryPoint,await this.device.createComputePipelineAsync({layout,compute:{module,entryPoint}}));}));
 }
 /** The authored regions (uniformMixedRegionTable), the table the census
  * holds too (UniformMixedDynamicClassifier.setRegions): a build makes every
  * tile of a fine region h. Queue-ordered: the next encode's build. */
 setRegions(table:Uint32Array<ArrayBuffer>):void{
  if(table.length!==UNIFORM_MIXED_REGION_TABLE_WORDS||table[0]!>UNIFORM_MIXED_REGION_LIMIT)throw new Error(`Mixed layout builder region table has ${table.length} words and ${table[0]} regions, expected ${UNIFORM_MIXED_REGION_TABLE_WORDS} words and at most ${UNIFORM_MIXED_REGION_LIMIT} regions`);
  this.device.queue.writeBuffer(this.statics,0,table);
  this.staticReady=true;
 }
 /** Seam tiles a built generation may slot in the hanging tap cache; more
  * raises UNIFORM_MIXED_RELAYOUT_FATAL.hangingCapacity. Default and upper
  * bound: the ownership's preallocated cache (capacity.hangingSlots). A
  * lower value only exercises the overflow path. */
 setHangingCapacity(slots:number):void{
  const most=this.level.current.capacity.hangingSlots;
  if(!Number.isSafeInteger(slots)||slots<0||slots>most)throw new Error(`Mixed layout builder hanging capacity must be an integer in 0..${most}: ${slots}`);
  this.device.queue.writeBuffer(this.level.params,4,new Uint32Array([slots]));
 }
 /** h tiles a built generation may hold; more raises
  * UNIFORM_MIXED_RELAYOUT_FATAL.fineCapacity. Default and upper bound: the
  * ownership's capacity.fineTiles (rewritten by the next encode after that
  * is reserved anew). A lower value only exercises the overflow path. */
 setFineCapacity(tiles:number):void{
  const most=this.level.current.capacity.fineTiles;
  if(!Number.isSafeInteger(tiles)||tiles<0||tiles>most)throw new Error(`Mixed layout builder h-tile capacity must be an integer in 0..${most}: ${tiles}`);
  this.device.queue.writeBuffer(this.level.params,12,new Uint32Array([tiles]));
 }
 /** h tiles a build may ask for before it is deferred (the host's
  * reservation): a build over it keeps the generation it is against and
  * reports its need (UNIFORM_MIXED_RELAYOUT_RECEIPT.need), where without an
  * admission capacity it raises fineCapacity. undefined: none, a layout
  * whose capacity is every tile the pool can store. Queue-ordered: the next
  * encode's build. */
 setAdmission(tiles?:number):void{
  if(tiles!==undefined&&(!Number.isSafeInteger(tiles)||tiles<0||tiles>this.tiles))throw new Error(`Mixed layout builder admission capacity must be an integer in 0..${this.tiles}: ${tiles}`);
  this.admission=tiles;this.word[0]=tiles??0xffffffff;this.device.queue.writeBuffer(this.level.params,16,this.word);
 }
 private admission?:number;
 /** Zero the build's counters: encode once before each encode, in the same
  * encoder, ahead of the band producer's pass (no pass between them touches
  * the work buffer), so the clear joins the frame's first blit run. */
 encodeClear(encoder:GPUCommandEncoder):void{encoder.clearBuffer(this.level.work,0,CLEARED*4);this.cleared=true;}
 private cleared=false;
 /** The ownership capacity the params' hanging and h-tile bounds were written at. */
 private capacityRevision:number;
 /** encode's launches in order, resolved once (apart from encode: a closure there would cost every encode a context). */
 private buildLaunches():readonly GPUComputePipeline[]{return ["widths","admit","classify","mirror","dilate","scan","scatter","verifyWords"].map(entry=>this.pipelines.get(entry)!);}
 /** Encode after the band bits are written, while the ownership is still
  * the generation to compare against; encodeClear must precede it. */
 encode(encoder:GPUCommandEncoder):void{
  if(this.pipelines.size!==8)throw new Error("Mixed layout builder is not initialized");
  if(!this.staticReady)throw new Error("Mixed layout builder has no region table");
  if(!this.cleared)throw new Error("Mixed layout builder encoded without encodeClear");
  this.cleared=false;
  // The per-key launches: small fixed grids over the resident blocks (and
  // dilate over the changed list); classify and scatter a workgroup per
  // 256 keys of a resident block at a time (scan alone spans them all).
  const groups64=(this.tiles+63)>>>6,keys=Math.min(UNIFORM_LIVE_KEY_GROUPS,groups64),verify=Math.min(UNIFORM_LIVE_GROUP_GROUPS,groups64),wide=Math.min(UNIFORM_LIVE_GROUP_GROUPS,this.blocks);
  const level=this.level;
  // The ownership's capacity was reserved anew: this build's bounds follow
  // (queue-ordered, ahead of this encoder's submit).
  if(level.current.capacityRevision!==this.capacityRevision){
   this.capacityRevision=level.current.capacityRevision;
   this.word[0]=level.current.capacity.hangingSlots;this.device.queue.writeBuffer(level.params,4,this.word);
   this.word[0]=level.current.capacity.fineTiles;this.device.queue.writeBuffer(level.params,12,this.word);
  }
  const launches=this.launches??=this.buildLaunches();
  const pass=encoder.beginComputePass(UniformMixedLayoutBuilder.PASS);pass.setBindGroup(0,level.group);
  // admit is launched only under an admission capacity: without one it could never act.
  pass.setPipeline(launches[0]!);pass.dispatchWorkgroups(keys);
  if(this.admission!==undefined){pass.setPipeline(launches[1]!);pass.dispatchWorkgroups(keys);}
  pass.setPipeline(launches[2]!);pass.dispatchWorkgroups(wide);
  pass.setPipeline(launches[3]!);pass.dispatchWorkgroups(keys);
  pass.setPipeline(launches[4]!);pass.dispatchWorkgroups(keys);
  pass.setPipeline(launches[5]!);pass.dispatchWorkgroups(1);
  pass.setPipeline(launches[6]!);pass.dispatchWorkgroups(wide);
  pass.setPipeline(launches[7]!);pass.dispatchWorkgroups(verify);
  pass.end();
 }
 destroy():void{
  this.statics.destroy();
  const l=this.level;l.topology.destroy();l.support.destroy();l.slots.destroy();l.work.destroy();l.params.destroy();l.status.destroy();
 }
}
