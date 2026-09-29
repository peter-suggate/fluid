import {uniformMixedLayoutFromTiles,type UniformMixedLayout} from "./uniform-mixed-layout";
import {UNIFORM_MIXED_OVERFLOW_HANGING} from "./uniform-mixed-topology.wgsl";
import type {UniformMixedBuiltOwnership,UniformMixedGenerationBuffers,UniformMixedOwnership} from "./uniform-mixed-ownership";

const BLOCK=256;
const RECEIPT=16;
/** Scan categories: h tiles, 4h tiles (one coarse owner each), seam h and 4h
 * tiles, regular 4h tiles. The simulation layout is ungraded h/4h. */
const CATEGORIES=5;

/** Where the fine band lives: one bit per tile from `wordOffset` words into
 * `buffer`. headerWords: words [0, headerWords) of `buffer` are the
 * producer's receipt header (the census's), carried into the builder receipt. */
export interface UniformMixedBandBits {readonly buffer:GPUBuffer;readonly wordOffset:number;readonly headerWords?:number}

/** Word offsets of the builder's compact relayout receipt
 * (UniformMixedLayoutBuilder.receipt): what a frame receipt copies instead
 * of the tile words. census: the census header (20 words, see
 * UniformMixedDynamicCensus / uniform-mixed-dynamic.ts HEADER), written by
 * each encode. changed: tiles whose width changed; tiers: h and 4h tiles;
 * seams: seam h and seam 4h tiles (their sum is the hanging slots); counts:
 * umCounts (h, 4h, 0, 8); all per build. Persistent: generation, bumped by
 * every build that changed a tile and raised no fatal flag; builds, bumped
 * by every build; fatal, sticky UNIFORM_MIXED_RELAYOUT_FATAL bits; fatalBuild,
 * the build number that first raised one. */
export const UNIFORM_MIXED_RELAYOUT_RECEIPT={census:0,changed:20,tiers:21,seams:23,counts:28,generation:32,builds:33,fatal:34,fatalBuild:35,words:36} as const;
/** Sticky fatal bits of the relayout receipt, validated on the GPU. A set
 * bit means the built generation must not be adopted or advanced on.
 * hangingCapacity (= UNIFORM_MIXED_OVERFLOW_HANGING): seam tiles exceed the
 * preallocated hanging tap cache (ownership.capacity.hangingSlots); slots
 * past it are left unslotted. tierSum: h + 4h tiles != tiles. tileWords: a
 * tile word, owner index or worklist entry disagrees with the receipt counts
 * (the host's cellCount check). Each build mirrors the whole word into the
 * simulation ownership's sticky overflow word (ownership.overflowOffset). */
export const UNIFORM_MIXED_RELAYOUT_FATAL={hangingCapacity:UNIFORM_MIXED_OVERFLOW_HANGING,tierSum:2,tileWords:4} as const;

export interface UniformMixedBuiltLevel extends UniformMixedBuiltOwnership {
 /** Tiles whose width differs from the ownership this level was built against. */
 readonly changedTiles:number;
 /** h and 4h tile counts from the receipt, indexed by tier. */
 readonly tierCounts:readonly [number,number];
 /** The receipt's GPU generation after this build. */
 readonly generation:number;
}

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
 * with a GPU generation counter. read() is the host mirror: it maps the
 * receipt and the tile words to rebuild the UniformMixedLayout;
 * UniformMixedOwnership.adopt copies the rest on the GPU. */
export class UniformMixedLayoutBuilder {
 readonly allocatedBytes:number;
 private readonly tiles:number;
 private readonly blocks:number;
 private readonly statics:GPUBuffer;
 private readonly readback:GPUBuffer;
 private readonly resources:GPUBindGroupLayout;
 private readonly level:Level;
 private readonly pipelines=new Map<string,GPUComputePipeline>();
 private regions:UniformMixedLayout["regions"]=[];
 private staticReady=false;
 private encoded=false;
 /** The band producer's receipt header (UniformMixedBandBits.headerWords). */
 private readonly header?:{readonly buffer:GPUBuffer;readonly words:number};
 /** The compact relayout receipt: copy UNIFORM_MIXED_RELAYOUT_RECEIPT.words
  * words from `offset` bytes. Per-build words are complete once this
  * builder's encode has run; generation, builds and the fatal bits persist. */
 get receipt():{readonly buffer:GPUBuffer;readonly offset:number;readonly words:number}{return {buffer:this.level.status,offset:0,words:UNIFORM_MIXED_RELAYOUT_RECEIPT.words};}
 /** The built generation's buffers, complete once encode has run: what
  * UniformMixedOwnership.adoptGpu copies, with no host mirror. */
 get generation():UniformMixedGenerationBuffers{const l=this.level;return {topology:l.topology,support:l.support,slots:l.slots,counts:{buffer:l.work,offset:48}};}
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
   if(band.headerWords)this.header={buffer:band.buffer,words:band.headerWords};
  }
  this.readback=device.createBuffer({label:"Uniform layout builder receipt",size:(R.words+n)*4,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  this.resources=device.createBindGroupLayout({entries:[
   ...[0,1,2].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage" as const}})),
   {binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},
   ...[4,5,6,7,8].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage" as const}})),
  ]});
  let bytes=this.statics.size+this.readback.size;
  const topology=storage("Uniform layout builder topology",4*n);
  const support=storage("Uniform layout builder support",9*n+24);
  const slots=storage("Uniform layout builder slots",2*n);
  const work=storage("Uniform layout builder work",RECEIPT+n+CATEGORIES*this.blocks);
  const params=device.createBuffer({label:"Uniform layout builder params",size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
  // Hanging capacity: the ownership's preallocated tap cache.
  device.queue.writeBuffer(params,0,new Uint32Array([band.wordOffset,ownership.capacity.hangingSlots,0,0]));
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
  this.allocatedBytes=bytes;
 }
 async initialize():Promise<void>{
  const lattice=this.level.current.capacity.lattice,n=this.tiles,x=this.level.current.dispatchX;
  const T=lattice.dimensions.map(d=>d/4);
  const flags=RECEIPT,totals=RECEIPT+n;
  const R=UNIFORM_MIXED_RELAYOUT_RECEIPT,F=UNIFORM_MIXED_RELAYOUT_FATAL;
  const module=this.device.createShaderModule({label:"Uniform mixed layout builder",code:/* wgsl */`
@group(0) @binding(0) var<storage,read> band:array<u32>;
@group(0) @binding(1) var<storage,read> statics:array<u32>;
@group(0) @binding(2) var<storage,read> current:array<u32>;
struct Params {bandOffset:u32,hangingCapacity:u32}
@group(0) @binding(3) var<uniform> params:Params;
@group(0) @binding(4) var<storage,read_write> work:array<atomic<u32>>;
@group(0) @binding(5) var<storage,read_write> topology:array<u32>;
@group(0) @binding(6) var<storage,read_write> support:array<u32>;
@group(0) @binding(7) var<storage,read_write> slots:array<u32>;
// The compact relayout receipt (UNIFORM_MIXED_RELAYOUT_RECEIPT).
@group(0) @binding(8) var<storage,read_write> status:array<atomic<u32>>;
const R_CHANGED:u32=${R.changed}u;const R_TIERS:u32=${R.tiers}u;const R_SEAMS:u32=${R.seams}u;const R_COUNTS:u32=${R.counts}u;
const R_GENERATION:u32=${R.generation}u;const R_BUILDS:u32=${R.builds}u;const R_FATAL:u32=${R.fatal}u;const R_FATAL_BUILD:u32=${R.fatalBuild}u;
const FATAL_TIER_SUM:u32=${F.tierSum}u;const FATAL_TILE_WORDS:u32=${F.tileWords}u;const FATAL_HANGING:u32=${F.hangingCapacity}u;
const N:u32=${n}u;const T=vec3u(${T.map(v=>`${v}u`).join(",")});const X:u32=${x}u;
const BLOCKS:u32=${this.blocks}u;const INF:u32=0xffffffffu;
const FLAGS:u32=${flags}u;const TOTALS:u32=${totals}u;
fn coord(t:u32)->vec3u{return vec3u(t%T.x,(t/T.x)%T.y,t/(T.x*T.y));}
fn key(p:vec3u)->u32{return p.x+T.x*(p.y+T.y*p.z);}
fn inside(q:vec3i)->bool{return all(q>=vec3i(0))&&all(q<vec3i(T));}
const WORDS:u32=${Math.ceil(n/32)}u;
fn fineAt(t:u32)->bool{return ((((band[params.bandOffset+t/32u]&~statics[WORDS+t/32u])|statics[t/32u])>>(t%32u))&1u)!=0u;}
fn wordWidth(word:u32)->u32{return select(4u,1u,(word&0x80000000u)!=0u);}
fn widthAt(t:u32)->u32{return atomicLoad(&work[FLAGS+t])&7u;}
// Width: h for the band and static tiles, else 4h (ungraded).
@compute @workgroup_size(64) fn widths(@builtin(global_invocation_id) gid:vec3u){
 let t=gid.x;if(t>=N){return;}
 let w=select(4u,1u,fineAt(t));
 atomicStore(&work[FLAGS+t],w);
 if(w!=wordWidth(current[t])){atomicAdd(&work[0],1u);}
}
var<workgroup> blockTotals:array<atomic<u32>,${CATEGORIES}>;
fn categories(w:u32,regular:bool)->array<u32,${CATEGORIES}>{
 var c:array<u32,${CATEGORIES}>;
 c[0]=select(0u,1u,w==1u);c[1]=select(0u,1u,w==4u);
 c[2]=select(0u,c[0],!regular);c[3]=select(0u,c[1],!regular);c[4]=select(0u,c[1],regular);
 return c;
}
// Frozen 3x3x3 stencil masks and per-block category totals.
@compute @workgroup_size(${BLOCK}) fn classify(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 if(lane<${CATEGORIES}u){atomicStore(&blockTotals[lane],0u);}
 workgroupBarrier();
 let t=group.x*${BLOCK}u+lane;
 if(t<N){
  let w=widthAt(t);let p=vec3i(coord(t));
  var maximum=w;var minimum=w;var fine=0u;
  for(var z=-1;z<=1;z++){for(var y=-1;y<=1;y++){for(var x=-1;x<=1;x++){
   let q=p+vec3i(x,y,z);if(!inside(q)){continue;}
   let v=widthAt(key(vec3u(q)));maximum=max(maximum,v);minimum=min(minimum,v);
   if(v==1u){fine|=1u<<u32((x+1)+3*((y+1)+3*(z+1)));}
  }}}
  topology[2u*N+2u*t]=fine|(maximum<<27u);topology[2u*N+2u*t+1u]=minimum<<27u;
  let regular=maximum==minimum;
  atomicStore(&work[FLAGS+t],w|select(0u,8u,regular));
  let c=categories(w,regular);
  for(var k=0u;k<${CATEGORIES}u;k++){if(c[k]!=0u){atomicAdd(&blockTotals[k],c[k]);}}
 }
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
  // [4] seam 4h; [5,12) stay cleared.
  atomicStore(&work[1],f);atomicStore(&work[2],c);atomicStore(&work[3],grand[2]);atomicStore(&work[4],grand[3]);
  // umCounts (h, 4h, 0, loop bound), then the frame-plan header (update(): header[2], [8..10], [12..14]).
  atomicStore(&work[12],f);atomicStore(&work[13],c);atomicStore(&work[14],0u);atomicStore(&work[15],8u);
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
  atomicStore(&status[R_COUNTS],f);atomicStore(&status[R_COUNTS+1u],c);atomicStore(&status[R_COUNTS+2u],0u);atomicStore(&status[R_COUNTS+3u],8u);
  if(f+c!=N){atomicOr(&status[R_FATAL],FATAL_TIER_SUM);}
  if(grand[2]+grand[3]>params.hangingCapacity){atomicOr(&status[R_FATAL],FATAL_HANGING);}
 }
}
// Owner numbering, worklists and the hanging slot table, in tile key order.
@compute @workgroup_size(${BLOCK}) fn scatter(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 let t=group.x*${BLOCK}u+lane;let valid=t<N;
 var w=0u;var regular=false;
 if(valid){let f=atomicLoad(&work[FLAGS+t]);w=f&7u;regular=(f&8u)!=0u;}
 let c=categories(w,regular);
 var rank:array<u32,${CATEGORIES}>;
 for(var k=0u;k<${CATEGORIES}u;k++){
  partial[lane]=c[k];workgroupBarrier();scanPartial(lane);
  rank[k]=atomicLoad(&work[TOTALS+group.x*${CATEGORIES}u+k])+partial[lane]-c[k];
  workgroupBarrier();
 }
 if(!valid){return;}
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
// verifyWords checks the scattered tile word against its width and the
// receipt: an h tile's owner base is 64 x its rank below the h count, a 4h
// tile's owner lies in [64f, 64f + c), and its tier list entry names it.
@compute @workgroup_size(64) fn verifyWords(@builtin(global_invocation_id) gid:vec3u){
 let t=gid.x;if(t>=N){return;}
 let word=topology[t];let f=atomicLoad(&work[1]);let c=atomicLoad(&work[2]);var ok=wordWidth(word)==widthAt(t)&&(word&0x40000000u)==0u;
 if((word&0x80000000u)!=0u){let base=word&0x3fffffffu;let rank=base/64u;ok=ok&&base%64u==0u&&rank<f&&topology[N+rank]==t;}
 else{ok=ok&&word>=64u*f&&word-64u*f<c&&topology[N+f+word-64u*f]==t;}
 if(!ok){atomicOr(&status[R_FATAL],FATAL_TILE_WORDS);}
}
// Every check of this build has run (scan, verifyWords): number the build,
// latch the first fatal one, else advance the generation if a tile changed.
@compute @workgroup_size(1) fn sealBuild(){
 let build=atomicAdd(&status[R_BUILDS],1u)+1u;
 if(atomicLoad(&status[R_FATAL])!=0u){if(atomicLoad(&status[R_FATAL_BUILD])==0u){atomicStore(&status[R_FATAL_BUILD],build);}}
 else if(atomicLoad(&status[R_CHANGED])!=0u){atomicAdd(&status[R_GENERATION],1u);}
}
`});
  const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");
  if(errors.length)throw new Error(errors.map(m=>`${m.lineNum}: ${m.message}`).join("\n"));
  const layout=this.device.createPipelineLayout({bindGroupLayouts:[this.resources]});
  await Promise.all(["widths","classify","scan","scatter","verifyWords","sealBuild"].map(async entryPoint=>{this.pipelines.set(entryPoint,await this.device.createComputePipelineAsync({layout,compute:{module,entryPoint}}));}));
 }
 /** Static fine tiles (fine-only regions), one byte per tile, and the
  * snapped regions the built layouts report. coarse: tiles a coarse-only
  * region holds at 4h; band bits there are dropped. Liquid-conditional solid
  * promotion arrives in the band bits, so the host leaves every tile a solid
  * could promote out of `coarse`. A tile in both masks is fine. */
 setStatic(fine:Uint8Array,regions:UniformMixedLayout["regions"],coarse?:Uint8Array):void{
  if(fine.length!==this.tiles||(coarse&&coarse.length!==this.tiles))throw new Error("Static masks do not match the tile lattice");
  const count=Math.ceil(this.tiles/32),words=new Uint32Array(2*count);
  for(let t=0;t<fine.length;t++){if(fine[t])words[t>>5]!|=1<<(t&31);if(coarse?.[t])words[count+(t>>5)]!|=1<<(t&31);}
  this.device.queue.writeBuffer(this.statics,0,words);
  this.regions=regions;this.staticReady=true;
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
 /** Encode after the band bits are written, while the ownership is still
  * the generation to compare against. readback: copy the receipt and the
  * tile words for read() (the host mirror); without it nothing is read
  * back, and the receipt (receipt) is the GPU's to consume. */
 encode(encoder:GPUCommandEncoder,readback=true):void{
  if(this.pipelines.size!==6)throw new Error("Mixed layout builder is not initialized");
  if(!this.staticReady)throw new Error("Mixed layout builder has no static fine mask");
  const n=this.tiles,groups64=Math.ceil(n/64);
  const level=this.level;
  encoder.clearBuffer(level.work,0,RECEIPT*4);
  if(this.header)encoder.copyBufferToBuffer(this.header.buffer,0,level.status,UNIFORM_MIXED_RELAYOUT_RECEIPT.census*4,this.header.words*4);
  const pass=encoder.beginComputePass({label:"Uniform mixed layout build"});pass.setBindGroup(0,level.group);
  for(const [entry,groups] of [["widths",groups64],["classify",this.blocks],["scan",1],["scatter",this.blocks],["verifyWords",groups64],["sealBuild",1]] as const){
   pass.setPipeline(this.pipelines.get(entry)!);pass.dispatchWorkgroups(groups);
  }
  pass.end();
  // The ownership's sticky overflow word: only the builder writes it, and
  // the receipt's fatal word is sticky, so the copy never clears a bit.
  encoder.copyBufferToBuffer(level.status,UNIFORM_MIXED_RELAYOUT_RECEIPT.fatal*4,level.current.support,level.current.overflowOffset,4);
  if(!readback)return;
  const words=UNIFORM_MIXED_RELAYOUT_RECEIPT.words;
  encoder.copyBufferToBuffer(level.status,0,this.readback,0,words*4);
  encoder.copyBufferToBuffer(level.topology,0,this.readback,words*4,n*4);
  this.encoded=true;
 }
 /** Map the generation built by the last submitted encode(readback): the
  * host mirror. Throws on any fatal bit the GPU checks raised. */
 async read():Promise<UniformMixedBuiltLevel>{
  if(!this.encoded)throw new Error("Mixed layout builder was not encoded with a readback");
  this.encoded=false;
  await this.readback.mapAsync(GPUMapMode.READ);
  const words=new Uint32Array(this.readback.getMappedRange()).slice();this.readback.unmap();
  const n=this.tiles,level=this.level,lattice=level.current.capacity.lattice,R=UNIFORM_MIXED_RELAYOUT_RECEIPT;
  const r=words.subarray(0,R.words),tiles=words.slice(R.words,R.words+n);
  const fatal=r[R.fatal]!;
  if(fatal){
   const causes=Object.entries(UNIFORM_MIXED_RELAYOUT_FATAL).filter(([,bit])=>fatal&bit).map(([name])=>name);
   throw new Error(`Mixed layout builder fatal (${causes.join(", ")}) at build ${r[R.fatalBuild]}: receipt ${[...r.subarray(R.changed)]}`);
  }
  // The receipt's tier counts; the tile lists verify them when built.
  const layout=uniformMixedLayoutFromTiles(lattice,tiles,this.regions,[r[R.tiers]!,r[R.tiers+1]!]);
  const seams=[r[R.seams]!,r[R.seams+1]!] as const;
  return {changedTiles:r[R.changed]!,tierCounts:[r[R.tiers]!,r[R.tiers+1]!],layout,seamCounts:seams,hangingSlots:seams[0]+seams[1],generation:r[R.generation]!,
   source:this.generation};
 }
 destroy():void{
  this.statics.destroy();this.readback.destroy();
  const l=this.level;l.topology.destroy();l.support.destroy();l.slots.destroy();l.work.destroy();l.params.destroy();l.status.destroy();
 }
}
