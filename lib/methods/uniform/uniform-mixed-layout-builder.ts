import { uniformCompileStencilWGSL } from "./uniform-compiled-topology";
import {UNIFORM_MIXED_OVERFLOW_FINE,UNIFORM_MIXED_OVERFLOW_HANGING} from "./uniform-mixed-topology.wgsl";
import type {UniformMixedGenerationBuffers,UniformMixedOwnership} from "./uniform-mixed-ownership";

const BLOCK=256;
/** Work header: the cleared counters [0, CLEARED), then umCounts at COUNTS
 * (256 B aligned: the remap binds them as its target's uniform counts). */
const CLEARED=16,COUNTS=64,RECEIPT=COUNTS+4;
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
 const w=uniformMixedChangedTilesWords(tiles);
 return /* wgsl */`
@group(${group}) @binding(${binding}) var<storage,read> umChanges:array<u32>;
fn umChangedCount()->u32{return umChanges[${w.changedCount}u];}
fn umChangedTile(i:u32)->u32{return umChanges[${w.changedList}u+i];}
fn umDilatedCount()->u32{return umChanges[${w.dilatedCount}u];}
fn umDilatedTile(i:u32)->u32{return umChanges[${w.dilatedList}u+i];}
`;
}

/** Where the fine band lives: one bit per tile from `wordOffset` words into
 * `buffer`. headerWords: words [0, headerWords) of `buffer` are the
 * producer's receipt header (the census's), carried into the builder receipt. */
export interface UniformMixedBandBits {readonly buffer:GPUBuffer;readonly wordOffset:number;readonly headerWords?:number;
 /** The producer's residency audit word (UNIFORM_MIXED_RESIDENCY_AUDIT): non-zero raises the residency fatal. */
 readonly auditWord?:number}

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
export const UNIFORM_MIXED_RELAYOUT_RECEIPT={census:0,changed:20,tiers:21,seams:23,need:25,admit:26,counts:28,generation:32,builds:33,fatal:34,fatalBuild:35,words:36} as const;
/** Sticky fatal bits of the relayout receipt, validated on the GPU. A set
 * bit means the built generation must not be adopted or advanced on.
 * hangingCapacity (= UNIFORM_MIXED_OVERFLOW_HANGING): seam tiles exceed the
 * preallocated hanging tap cache (ownership.capacity.hangingSlots); slots
 * past it are left unslotted. tierSum: h + 4h tiles != tiles. tileWords: a
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
 * phase 4). From the band bits and a static fine mask (solids and fine-only
 * regions), it writes exactly the buffers UniformMixedOwnership.update
 * uploads for the ungraded h/4h createUniformMixedLayout(lattice, regions,
 * 4, static ∪ band). Every launch is a fixed grid over the lattice's tiles;
 * the receipt checks (tier sum, tile words, hanging capacity) run on the GPU
 * into sticky fatal bits of a compact receipt (UNIFORM_MIXED_RELAYOUT_RECEIPT)
 * with a GPU generation counter. There is no host mirror: the frame head
 * remaps on the built buffers and UniformMixedOwnership.adoptGpu copies
 * them; the receipt is the GPU's to consume (diagnostics copy it). */
export class UniformMixedLayoutBuilder {
 readonly allocatedBytes:number;
 private readonly tiles:number;
 private readonly blocks:number;
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
 get receipt():{readonly buffer:GPUBuffer;readonly offset:number;readonly words:number}{return {buffer:this.level.status,offset:0,words:UNIFORM_MIXED_RELAYOUT_RECEIPT.words};}
 /** The built generation's buffers, complete once encode has run: what
  * UniformMixedOwnership.adoptGpu copies, with no host mirror. */
 get generation():UniformMixedGenerationBuffers{const l=this.level;return {topology:l.topology,support:l.support,slots:l.slots,counts:{buffer:l.work,offset:COUNTS*4},changes:l.work};}
 /** ownership: the simulation ownership the built generation replaces. */
 constructor(private readonly device:GPUDevice,band:UniformMixedBandBits,ownership:UniformMixedOwnership){
  const n=ownership.capacity.tiles;
  this.tiles=n;this.blocks=Math.ceil(n/BLOCK);
  if(this.blocks>device.limits.maxComputeWorkgroupsPerDimension)throw new Error("Mixed layout builder tile count exceeds one dispatch dimension");
  const storage=(label:string,words:number)=>device.createBuffer({label,size:words*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
  // Static fine words, then static coarse words (coarse-only regions mask the band).
  this.statics=storage("Uniform layout builder static masks",2*Math.ceil(n/32));
  const R=UNIFORM_MIXED_RELAYOUT_RECEIPT;
  if(band.headerWords!==undefined){
   if(!Number.isSafeInteger(band.headerWords)||band.headerWords<0||band.headerWords>R.changed-R.census)throw new Error(`Band header of ${band.headerWords} words does not fit the relayout receipt`);
   this.headerWords=band.headerWords;
  }
  this.resources=device.createBindGroupLayout({entries:[
   ...[0,1,2].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage" as const}})),
   {binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},
   ...[4,5,6,7,8].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage" as const}})),
  ]});
  let bytes=this.statics.size;
  const topology=storage("Uniform layout builder topology",5*n);
  const support=storage("Uniform layout builder support",9*n+24);
  const slots=storage("Uniform layout builder slots",2*n);
  const work=device.createBuffer({label:"Uniform layout builder work",size:uniformMixedChangedTilesWords(n).words*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
  const params=device.createBuffer({label:"Uniform layout builder params",size:32,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
  // Hanging capacity: the ownership's preallocated tap cache.
  // Residency audit word, or none. h-tile capacity: the owner-indexed buffers'.
  // Admission capacity: none (setAdmission).
  device.queue.writeBuffer(params,0,new Uint32Array([band.wordOffset,ownership.capacity.hangingSlots,band.auditWord??0xffffffff,ownership.capacity.fineTiles,0xffffffff,0,0,0]));
  const status=device.createBuffer({label:"Uniform layout builder relayout receipt",size:R.words*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
  // Standalone stages visit everything until a frame census: support[0,4n) = 3.
  device.queue.writeBuffer(support,0,new Uint32Array(4*n).fill(3));
  const group=device.createBindGroup({layout:this.resources,entries:[
   {binding:0,resource:{buffer:band.buffer}},{binding:1,resource:{buffer:this.statics}},{binding:2,resource:ownership.presentation},
   {binding:3,resource:{buffer:params}},{binding:4,resource:{buffer:work}},{binding:5,resource:{buffer:topology}},
   {binding:6,resource:{buffer:support}},{binding:7,resource:{buffer:slots}},{binding:8,resource:{buffer:status}},
  ]});
  bytes+=topology.size+support.size+slots.size+work.size+params.size+status.size;
  this.level={current:ownership,topology,support,slots,work,params,group,status};
  this.capacityRevision=ownership.capacityRevision;
  this.allocatedBytes=bytes;
 }
 async initialize():Promise<void>{
  const lattice=this.level.current.capacity.lattice,n=this.tiles,x=this.level.current.dispatchX;
  const T=lattice.dimensions.map(d=>d/4);
  const flags=RECEIPT,totals=RECEIPT+n,lists=uniformMixedChangedTilesWords(n);
  const R=UNIFORM_MIXED_RELAYOUT_RECEIPT,F=UNIFORM_MIXED_RELAYOUT_FATAL;
  if(Math.ceil(n/64)>0xffff)throw new Error(`Mixed layout builder verifies ${Math.ceil(n/64)} groups, above the 16-bit ticket`);
  const module=this.device.createShaderModule({label:"Uniform mixed layout builder",code:/* wgsl */`
@group(0) @binding(0) var<storage,read> band:array<u32>;
@group(0) @binding(1) var<storage,read> statics:array<u32>;
@group(0) @binding(2) var<storage,read> current:array<u32>;
struct Params {bandOffset:u32,hangingCapacity:u32,audit:u32,fineCapacity:u32,admitCapacity:u32}
@group(0) @binding(3) var<uniform> params:Params;
@group(0) @binding(4) var<storage,read_write> work:array<atomic<u32>>;
@group(0) @binding(5) var<storage,read_write> topology:array<u32>;
@group(0) @binding(6) var<storage,read_write> support:array<u32>;
@group(0) @binding(7) var<storage,read_write> slots:array<u32>;
// The compact relayout receipt (UNIFORM_MIXED_RELAYOUT_RECEIPT).
@group(0) @binding(8) var<storage,read_write> status:array<atomic<u32>>;
const R_CHANGED:u32=${R.changed}u;const R_TIERS:u32=${R.tiers}u;const R_SEAMS:u32=${R.seams}u;const R_COUNTS:u32=${R.counts}u;
const R_NEED:u32=${R.need}u;const R_ADMIT:u32=${R.admit}u;const NEED:u32=${NEED}u;
const R_GENERATION:u32=${R.generation}u;const R_BUILDS:u32=${R.builds}u;const R_FATAL:u32=${R.fatal}u;const R_FATAL_BUILD:u32=${R.fatalBuild}u;
const FATAL_TIER_SUM:u32=${F.tierSum}u;const FATAL_TILE_WORDS:u32=${F.tileWords}u;const FATAL_HANGING:u32=${F.hangingCapacity}u;const FATAL_RESIDENCY:u32=${F.residency}u;const FATAL_OWNERS:u32=${F.fineCapacity}u;
const N:u32=${n}u;const T=vec3u(${T.map(v=>`${v}u`).join(",")});const X:u32=${x}u;
const BLOCKS:u32=${this.blocks}u;const INF:u32=0xffffffffu;
const FLAGS:u32=${flags}u;const TOTALS:u32=${totals}u;const COUNTS:u32=${COUNTS}u;
const CHANGED:u32=${lists.changedList}u;const DILATED:u32=${lists.dilatedList}u;const DILATED_COUNT:u32=${lists.dilatedCount}u;
fn coord(t:u32)->vec3u{return vec3u(t%T.x,(t/T.x)%T.y,t/(T.x*T.y));}
fn key(p:vec3u)->u32{return p.x+T.x*(p.y+T.y*p.z);}
fn inside(q:vec3i)->bool{return all(q>=vec3i(0))&&all(q<vec3i(T));}
const WORDS:u32=${Math.ceil(n/32)}u;
fn fineAt(t:u32)->bool{return ((((band[params.bandOffset+t/32u]&~statics[WORDS+t/32u])|statics[t/32u])>>(t%32u))&1u)!=0u;}
fn wordWidth(word:u32)->u32{return select(4u,1u,(word&0x80000000u)!=0u);}
fn widthAt(t:u32)->u32{return atomicLoad(&work[FLAGS+t])&7u;}
// Width: h for the band and static tiles, else 4h (ungraded). Staged as a
// plain word in the tile-word plane, which scatter overwrites, so classify's
// 27 neighbour reads are plain loads.
@compute @workgroup_size(64) fn widths(@builtin(global_invocation_id) gid:vec3u){
 let t=gid.x;if(t>=N){return;}
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
@compute @workgroup_size(64) fn admit(@builtin(global_invocation_id) gid:vec3u){
 if(atomicLoad(&work[NEED])<=params.admitCapacity){return;}
 let t=gid.x;if(t>=N){return;}
 topology[t]=wordWidth(current[t]);
 if(t==0u){atomicStore(&work[0],0u);}
}
// The dilated list: each changed tile's 3x3x3 neighbourhood, deduplicated by
// flag bit 16 (classify rewrote the flags; every later reader masks the
// width and regular bits).
@compute @workgroup_size(64) fn dilate(@builtin(global_invocation_id) gid:vec3u){
 let i=gid.x;if(i>=atomicLoad(&work[0])){return;}
 let p=vec3i(coord(atomicLoad(&work[CHANGED+i])));
 for(var z=-1;z<=1;z++){for(var y=-1;y<=1;y++){for(var x=-1;x<=1;x++){
  let q=p+vec3i(x,y,z);if(!inside(q)){continue;}
  let s=key(vec3u(q));
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
@compute @workgroup_size(${BLOCK}) fn classify(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 if(lane<${CATEGORIES}u){atomicStore(&blockTotals[lane],0u);}
 workgroupBarrier();
 let t=group.x*${BLOCK}u+lane;
 if(t<N){
  let w=topology[t];let p=vec3i(coord(t));
  var maximum=w;var minimum=w;var fine=0u;var valid=0u;
  for(var z=-1;z<=1;z++){for(var y=-1;y<=1;y++){for(var x=-1;x<=1;x++){
   let q=p+vec3i(x,y,z);if(!inside(q)){continue;}
   valid|=1u<<u32((x+1)+3*((y+1)+3*(z+1)));
   let v=topology[key(vec3u(q))];maximum=max(maximum,v);minimum=min(minimum,v);
   if(v==1u){fine|=1u<<u32((x+1)+3*((y+1)+3*(z+1)));}
  }}}
  topology[2u*N+2u*t]=fine|(maximum<<27u);topology[2u*N+2u*t+1u]=(minimum<<27u)|umCompileStencil(fine,valid);
  topology[4u*N+t]=umCompileBlendMask(fine);
  let regular=maximum==minimum;
  atomicStore(&work[FLAGS+t],w|select(0u,8u,regular));
  let c=categories(w,regular);
  for(var k=0u;k<4u;k++){if(k!=1u&&c[k]!=0u){atomicAdd(&blockTotals[k],1u);}}
 }
 workgroupBarrier();
 // 4h tiles are the block's other tiles; regular 4h ones its non-seam 4h.
 if(lane==0u){let c1=min(N-group.x*${BLOCK}u,${BLOCK}u)-atomicLoad(&blockTotals[0]);atomicStore(&blockTotals[1],c1);atomicStore(&blockTotals[4],c1-atomicLoad(&blockTotals[3]));}
 workgroupBarrier();
 if(lane<${CATEGORIES}u){atomicStore(&work[TOTALS+group.x*${CATEGORIES}u+lane],atomicLoad(&blockTotals[lane]));}
}
var<workgroup> partial:array<u32,${BLOCK}>;
var<workgroup> grand:array<u32,${CATEGORIES}>;
// Inclusive Hillis-Steele scan of partial[]; every lane must call it.
fn scanPartial(lane:u32){
 for(var offset=1u;offset<${BLOCK}u;offset*=2u){
  var v=0u;if(lane>=offset){v=partial[lane-offset];}
  workgroupBarrier();partial[lane]+=v;workgroupBarrier();
 }
}
// Exclusive block offsets per category, grand totals, and the headers.
@compute @workgroup_size(${BLOCK}) fn scan(@builtin(local_invocation_index) lane:u32){
${this.headerWords?`// The band producer's receipt header, carried into this build's receipt.
 if(lane<${this.headerWords}u){atomicStore(&status[${R.census}u+lane],band[lane]);}`:""}
 let per=(BLOCKS+${BLOCK-1}u)/${BLOCK}u;let first=min(BLOCKS,lane*per);let last=min(BLOCKS,first+per);
 for(var k=0u;k<${CATEGORIES}u;k++){
  var local=0u;for(var b=first;b<last;b++){local+=atomicLoad(&work[TOTALS+b*${CATEGORIES}u+k]);}
  partial[lane]=local;workgroupBarrier();scanPartial(lane);
  var running=partial[lane]-local;
  for(var b=first;b<last;b++){let i=TOTALS+b*${CATEGORIES}u+k;let c=atomicLoad(&work[i]);atomicStore(&work[i],running);running+=c;}
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
  if(f+c!=N){atomicOr(&status[R_FATAL],FATAL_TIER_SUM);}
  if(grand[2]+grand[3]>params.hangingCapacity){atomicOr(&status[R_FATAL],FATAL_HANGING);}
  // Known before scatter writes a tile word: every owner-indexed buffer
  // is sized for at most this many h tiles. A need over an admission
  // capacity was deferred (admit); this is the generation kept instead.
  if(f>params.fineCapacity){atomicOr(&status[R_FATAL],FATAL_OWNERS);}
  if(params.audit!=0xffffffffu&&band[params.audit]!=0u){atomicOr(&status[R_FATAL],FATAL_RESIDENCY);}
 }
}
// Owner numbering, worklists and the hanging slot table, in tile key order.
@compute @workgroup_size(${BLOCK}) fn scatter(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 let t=group.x*${BLOCK}u+lane;let valid=t<N;
 var w=0u;var regular=false;
 if(valid){let f=atomicLoad(&work[FLAGS+t]);w=f&7u;regular=(f&8u)!=0u;}
 let c=categories(w,regular);
 // One scan of h, seam h and seam 4h packed in 10-bit fields (a block sums
 // at most ${BLOCK}); the valid tiles below a valid lane are its lane, so its
 // 4h rank is lane minus its h rank and its regular rank that minus its seam 4h.
 partial[lane]=c[0]|(c[2]<<10u)|(c[3]<<20u);workgroupBarrier();scanPartial(lane);
 if(!valid){return;}
 let packed=partial[lane];var rank:array<u32,${CATEGORIES}>;
 rank[0]=(packed&1023u)-c[0];rank[2]=((packed>>10u)&1023u)-c[2];rank[3]=(packed>>20u)-c[3];
 rank[1]=lane-rank[0];rank[4]=rank[1]-rank[3];
 for(var k=0u;k<${CATEGORIES}u;k++){rank[k]+=atomicLoad(&work[TOTALS+group.x*${CATEGORIES}u+k]);}
 let f=atomicLoad(&work[1]);
 let seamF=atomicLoad(&work[3]);let seamC=atomicLoad(&work[4]);
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
 if(t>=min(seamF+seamC,params.hangingCapacity)){slots[N+t]=INF;}
}
// The detail ring (UNIFORM_MIXED_DETAIL_RING): bit 0 of the second stencil
// word, an h tile within three tiles, from classify's 27-bit h masks of the
// tiles two apart. Each lane writes its own word.
@compute @workgroup_size(64) fn mirror(@builtin(global_invocation_id) gid:vec3u){
 let t=gid.x;if(t>=N){return;}
 let p=vec3i(coord(t));var near=false;
 for(var z=-2;z<=2;z+=2){for(var y=-2;y<=2;y+=2){for(var x=-2;x<=2;x+=2){
  let q=p+vec3i(x,y,z);if(!inside(q)){continue;}
  near=near||(topology[2u*N+2u*key(vec3u(q))]&0x7ffffffu)!=0u;
 }}}
 if(near){topology[2u*N+2u*t+1u]|=1u;}
}
// verifyWords checks the scattered tile word against its width and the
// receipt: an h tile's owner base is 64 x its rank below the h count, a 4h
// tile's owner lies in [64f, 64f + c), and its tier list entry names it.
var<workgroup> groupBad:atomic<u32>;
const VERIFY_GROUPS:u32=${Math.ceil(n/64)}u;
@compute @workgroup_size(64) fn verifyWords(@builtin(global_invocation_id) gid:vec3u,@builtin(local_invocation_index) lane:u32){
 if(lane==0u){atomicStore(&groupBad,0u);}
 workgroupBarrier();
 let t=gid.x;
 if(t<N){
  let word=topology[t];let f=atomicLoad(&work[1]);let c=atomicLoad(&work[2]);var ok=wordWidth(word)==widthAt(t)&&(word&0x40000000u)==0u;
  if((word&0x80000000u)!=0u){let base=word&0x3fffffffu;let rank=base/64u;ok=ok&&base%64u==0u&&rank<f&&topology[N+rank]==t;}
  else{ok=ok&&word>=64u*f&&word-64u*f<c&&topology[N+f+word-64u*f]==t;}
  if(!ok){atomicStore(&groupBad,1u);}
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
`});
  const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");
  if(errors.length)throw new Error(errors.map(m=>`${m.lineNum}: ${m.message}`).join("\n"));
  const layout=this.device.createPipelineLayout({bindGroupLayouts:[this.resources]});
  await Promise.all(["widths","admit","classify","mirror","dilate","scan","scatter","verifyWords"].map(async entryPoint=>{this.pipelines.set(entryPoint,await this.device.createComputePipelineAsync({layout,compute:{module,entryPoint}}));}));
 }
 /** Static fine tiles (fine-only regions), one byte per tile. coarse: tiles a coarse-only
  * region holds at 4h; band bits there are dropped. Liquid-conditional solid
  * promotion arrives in the band bits, so the host leaves every tile a solid
  * could promote out of `coarse`. A tile in both masks is fine. */
 setStatic(fine:Uint8Array,coarse?:Uint8Array):void{
  if(fine.length!==this.tiles||(coarse&&coarse.length!==this.tiles))throw new Error("Static masks do not match the tile lattice");
  const count=Math.ceil(this.tiles/32),words=new Uint32Array(2*count);
  for(let t=0;t<fine.length;t++){if(fine[t])words[t>>5]!|=1<<(t&31);if(coarse?.[t])words[count+(t>>5)]!|=1<<(t&31);}
  this.device.queue.writeBuffer(this.statics,0,words);
  this.staticWords=words;this.staticReady=true;
 }
 /** The words setStatic wrote, kept so an edit moves them by tiles. */
 private staticWords?:Uint32Array<ArrayBuffer>;
 /** setStatic for a fine mask that differs from the one held at `tiles` at
  * most (a tile may repeat, or not have changed): the words those tiles are
  * in are rewritten, as runs joined across short gaps, so the buffer holds
  * what setStatic(fine) with the same coarse mask would have written. */
 editStatic(tiles:readonly number[],fine:Uint8Array):void{
  const words=this.staticWords;
  if(!words||fine.length!==this.tiles)throw new Error("A static mask edit needs a static mask of the tile lattice");
  const dirty:number[]=[];
  for(const t of tiles){const w=t>>5,bit=1<<(t&31),before=words[w]!,after=(fine[t]?before|bit:before&~bit)>>>0;if(after!==before){words[w]=after;dirty.push(w);}}
  dirty.sort((a,b)=>a-b);
  for(let i=0;i<dirty.length;){
   let j=i;while(j+1<dirty.length&&dirty[j+1]!-dirty[j]!<=64)j++;
   this.device.queue.writeBuffer(this.statics,4*dirty[i]!,words,dirty[i]!,dirty[j]!-dirty[i]!+1);i=j+1;
  }
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
  * whose capacity is every tile. Queue-ordered: the next encode's build. */
 setAdmission(tiles?:number):void{
  if(tiles!==undefined&&(!Number.isSafeInteger(tiles)||tiles<0||tiles>this.tiles))throw new Error(`Mixed layout builder admission capacity must be an integer in 0..${this.tiles}: ${tiles}`);
  this.admission=tiles;this.device.queue.writeBuffer(this.level.params,16,new Uint32Array([tiles??0xffffffff]));
 }
 private admission?:number;
 /** Zero the build's counters: encode once before each encode, in the same
  * encoder, ahead of the band producer's pass (no pass between them touches
  * the work buffer), so the clear joins the frame's first blit run. */
 encodeClear(encoder:GPUCommandEncoder):void{encoder.clearBuffer(this.level.work,0,CLEARED*4);this.cleared=true;}
 private cleared=false;
 /** The ownership capacity the params' hanging and h-tile bounds were written at. */
 private capacityRevision:number;
 /** Encode after the band bits are written, while the ownership is still
  * the generation to compare against; encodeClear must precede it. */
 encode(encoder:GPUCommandEncoder):void{
  if(this.pipelines.size!==8)throw new Error("Mixed layout builder is not initialized");
  if(!this.staticReady)throw new Error("Mixed layout builder has no static fine mask");
  if(!this.cleared)throw new Error("Mixed layout builder encoded without encodeClear");
  this.cleared=false;
  const n=this.tiles,groups64=Math.ceil(n/64);
  const level=this.level;
  // The ownership's capacity was reserved anew: this build's bounds follow
  // (queue-ordered, ahead of this encoder's submit).
  if(level.current.capacityRevision!==this.capacityRevision){
   this.capacityRevision=level.current.capacityRevision;
   this.device.queue.writeBuffer(level.params,4,new Uint32Array([level.current.capacity.hangingSlots]));
   this.device.queue.writeBuffer(level.params,12,new Uint32Array([level.current.capacity.fineTiles]));
  }
  const pass=encoder.beginComputePass({label:"Uniform mixed layout build"});pass.setBindGroup(0,level.group);
  // admit is launched only under an admission capacity: without one it could never act.
  for(const [entry,groups] of [["widths",groups64],...(this.admission===undefined?[]:[["admit",groups64] as const]),["classify",this.blocks],["mirror",groups64],["dilate",groups64],["scan",1],["scatter",this.blocks],["verifyWords",groups64]] as const){
   pass.setPipeline(this.pipelines.get(entry)!);pass.dispatchWorkgroups(groups);
  }
  pass.end();
 }
 destroy():void{
  this.statics.destroy();
  const l=this.level;l.topology.destroy();l.support.destroy();l.slots.destroy();l.work.destroy();l.params.destroy();l.status.destroy();
 }
}
