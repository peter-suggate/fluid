import {uniformMixedLayoutFromTiles,type UniformMixedLayout} from "./uniform-mixed-layout";
import type {UniformMixedBuiltOwnership,UniformMixedOwnership} from "./uniform-mixed-ownership";

const BLOCK=256;
const RECEIPT=16;
/** Scan categories: h tiles, 4h tiles (one coarse owner each), seam h and 4h
 * tiles, regular 4h tiles. The simulation layout is ungraded h/4h. */
const CATEGORIES=5;

/** Where the fine band lives: one bit per tile from `wordOffset` words into `buffer`. */
export interface UniformMixedBandBits {readonly buffer:GPUBuffer;readonly wordOffset:number}

export interface UniformMixedBuiltLevel extends UniformMixedBuiltOwnership {
 /** Tiles whose width differs from the ownership this level was built against. */
 readonly changedTiles:number;
 /** h, tier-1 and 4h tile counts from the receipt (the tier-1 slot is empty, reserved). */
 readonly tierCounts:readonly [number,number,number];
}

interface Level {
 readonly current:UniformMixedOwnership;
 readonly topology:GPUBuffer;readonly support:GPUBuffer;readonly slots:GPUBuffer;readonly work:GPUBuffer;
 readonly params:GPUBuffer;readonly group:GPUBindGroup;
}

/** GPU ownership builder for dynamic coarsening (docs/plans/uniform-dynamic-coarsening.md,
 * phase 4). From the band bits and a static fine mask (solids and fine-only
 * regions), it writes exactly the buffers UniformMixedOwnership.update
 * uploads for the ungraded h/4h createUniformMixedLayout(lattice, regions,
 * 4, static ∪ band). The host reads back a 16-word receipt and the tile
 * words; UniformMixedOwnership.adopt copies the rest on the GPU. */
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
 /** ownership: the simulation ownership the built generation replaces. */
 constructor(private readonly device:GPUDevice,band:UniformMixedBandBits,ownership:UniformMixedOwnership){
  const n=ownership.layout.tiles.length;
  this.tiles=n;this.blocks=Math.ceil(n/BLOCK);
  if(this.blocks>device.limits.maxComputeWorkgroupsPerDimension)throw new Error("Mixed layout builder tile count exceeds one dispatch dimension");
  const storage=(label:string,words:number)=>device.createBuffer({label,size:words*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
  // Static fine words, then static coarse words (coarse-only regions mask the band).
  this.statics=storage("Uniform layout builder static masks",2*Math.ceil(n/32));
  this.readback=device.createBuffer({label:"Uniform layout builder receipt",size:(RECEIPT+n)*4,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  this.resources=device.createBindGroupLayout({entries:[
   ...[0,1,2].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage" as const}})),
   {binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},
   ...[4,5,6,7].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage" as const}})),
  ]});
  let bytes=this.statics.size+this.readback.size;
  const topology=storage("Uniform layout builder topology",4*n);
  const support=storage("Uniform layout builder support",9*n+24);
  const slots=storage("Uniform layout builder slots",2*n);
  const work=storage("Uniform layout builder work",RECEIPT+n+CATEGORIES*this.blocks+2*n);
  const params=device.createBuffer({label:"Uniform layout builder params",size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
  device.queue.writeBuffer(params,0,new Uint32Array([band.wordOffset,0,0,0]));
  // Standalone stages visit everything until a frame census: support[0,4n) = 3.
  device.queue.writeBuffer(support,0,new Uint32Array(4*n).fill(3));
  const group=device.createBindGroup({layout:this.resources,entries:[
   {binding:0,resource:{buffer:band.buffer}},{binding:1,resource:{buffer:this.statics}},{binding:2,resource:ownership.presentation},
   {binding:3,resource:{buffer:params}},{binding:4,resource:{buffer:work}},{binding:5,resource:{buffer:topology}},
   {binding:6,resource:{buffer:support}},{binding:7,resource:{buffer:slots}},
  ]});
  bytes+=topology.size+support.size+slots.size+work.size+params.size;
  this.level={current:ownership,topology,support,slots,work,params,group};
  this.allocatedBytes=bytes;
 }
 async initialize():Promise<void>{
  const lattice=this.level.current.layout.lattice,n=this.tiles,x=this.level.current.dispatchX;
  const T=lattice.dimensions.map(d=>d/4);
  const flags=RECEIPT,totals=RECEIPT+n,distanceA=totals+CATEGORIES*this.blocks,distanceB=distanceA+n;
  const module=this.device.createShaderModule({label:"Uniform mixed layout builder",code:/* wgsl */`
@group(0) @binding(0) var<storage,read> band:array<u32>;
@group(0) @binding(1) var<storage,read> statics:array<u32>;
@group(0) @binding(2) var<storage,read> current:array<u32>;
struct Params {bandOffset:u32}
@group(0) @binding(3) var<uniform> params:Params;
@group(0) @binding(4) var<storage,read_write> work:array<atomic<u32>>;
@group(0) @binding(5) var<storage,read_write> topology:array<u32>;
@group(0) @binding(6) var<storage,read_write> support:array<u32>;
@group(0) @binding(7) var<storage,read_write> slots:array<u32>;
const N:u32=${n}u;const T=vec3u(${T.map(v=>`${v}u`).join(",")});const X:u32=${x}u;
const BLOCKS:u32=${this.blocks}u;const INF:u32=0xffffffffu;
const FLAGS:u32=${flags}u;const TOTALS:u32=${totals}u;const DA:u32=${distanceA}u;const DB:u32=${distanceB}u;
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
  // Receipt: [1] h tiles, [3] 4h tiles, [5] seam h, [7] seam 4h; the tier-1
  // words [2] and [6] are empty (reserved) and stay cleared.
  atomicStore(&work[1],f);atomicStore(&work[3],c);atomicStore(&work[5],grand[2]);atomicStore(&work[7],grand[3]);
  // Counts (tier-1 slot empty), then the frame-plan header (update(): header[2], [8..10], [12..14]).
  atomicStore(&work[12],f);atomicStore(&work[13],0u);atomicStore(&work[14],c);atomicStore(&work[15],8u);
  let merged=f+grand[3]+(grand[4]+63u)/64u;
  let h=4u*N;
  for(var i=0u;i<16u;i++){support[h+i]=0u;}
  support[h+2u]=f;
  support[h+8u]=min(f,X);support[h+9u]=(f+X-1u)/X;support[h+10u]=1u;
  support[h+12u]=min(merged,X);support[h+13u]=(merged+X-1u)/X;support[h+14u]=1u;
  let seam=7u*N+16u;support[seam]=grand[2];support[seam+1u]=0u;support[seam+2u]=grand[3];support[seam+3u]=0u;
  let regular=8u*N+20u;support[regular]=0u;support[regular+1u]=grand[4];support[regular+2u]=0u;support[regular+3u]=0u;
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
 let seamF=atomicLoad(&work[5]);let seamC=atomicLoad(&work[7]);
 var slot=INF;
 if(w==1u){
  topology[t]=(rank[0]*64u)|0x80000000u;topology[N+rank[0]]=t;support[6u*N+16u+rank[0]]=t;
  if(!regular){support[7u*N+20u+rank[2]]=t;slot=rank[2];}
 }else{
  topology[t]=f*64u+rank[1];topology[N+f+rank[1]]=t;
  if(regular){support[8u*N+24u+rank[4]]=t;}else{support[7u*N+20u+seamF+rank[3]]=t;slot=seamF+rank[3];}
 }
 slots[t]=slot;
 if(slot!=INF){slots[N+slot]=t;}
 if(t>=seamF+seamC){slots[N+t]=INF;}
}
// Exact chessboard distance to a non-h tile, separable: x, then y, then z.
@compute @workgroup_size(64) fn distance0(@builtin(global_invocation_id) gid:vec3u){
 let t=gid.x;if(t>=N){return;}let p=coord(t);var d=INF;
 for(var i=0u;i<T.x;i++){if(widthAt(key(vec3u(i,p.y,p.z)))!=1u){d=min(d,u32(abs(i32(i)-i32(p.x))));}}
 atomicStore(&work[DA+t],d);
}
@compute @workgroup_size(64) fn distance1(@builtin(global_invocation_id) gid:vec3u){
 let t=gid.x;if(t>=N){return;}let p=coord(t);var d=INF;
 for(var i=0u;i<T.y;i++){let s=atomicLoad(&work[DA+key(vec3u(p.x,i,p.z))]);if(s!=INF){d=min(d,max(u32(abs(i32(i)-i32(p.y))),s));}}
 atomicStore(&work[DB+t],d);
}
@compute @workgroup_size(64) fn distance2(@builtin(global_invocation_id) gid:vec3u){
 let t=gid.x;if(t>=N){return;}let p=coord(t);var d=INF;
 for(var i=0u;i<T.z;i++){let s=atomicLoad(&work[DB+key(vec3u(p.x,p.y,i))]);if(s!=INF){d=min(d,max(u32(abs(i32(i)-i32(p.z))),s));}}
 support[4u*N+16u+t]=d;
}
`});
  const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");
  if(errors.length)throw new Error(errors.map(m=>`${m.lineNum}: ${m.message}`).join("\n"));
  const layout=this.device.createPipelineLayout({bindGroupLayouts:[this.resources]});
  for(const entryPoint of ["widths","classify","scan","scatter","distance0","distance1","distance2"])
   this.pipelines.set(entryPoint,await this.device.createComputePipelineAsync({layout,compute:{module,entryPoint}}));
 }
 /** Static fine tiles (solid promotion and fine-only regions), one byte per
  * tile, and the snapped regions the built layouts report. coarse: tiles a
  * coarse-only region holds at 4h; band bits there are dropped. A tile in
  * both masks is fine (solid promotion wins). */
 setStatic(fine:Uint8Array,regions:UniformMixedLayout["regions"],coarse?:Uint8Array):void{
  if(fine.length!==this.tiles||(coarse&&coarse.length!==this.tiles))throw new Error("Static masks do not match the tile lattice");
  const count=Math.ceil(this.tiles/32),words=new Uint32Array(2*count);
  for(let t=0;t<fine.length;t++){if(fine[t])words[t>>5]!|=1<<(t&31);if(coarse?.[t])words[count+(t>>5)]!|=1<<(t&31);}
  this.device.queue.writeBuffer(this.statics,0,words);
  this.regions=regions;this.staticReady=true;
 }
 /** Encode after the band bits are written, while the ownership is still
  * the generation to compare against. */
 encode(encoder:GPUCommandEncoder):void{
  if(this.pipelines.size!==7)throw new Error("Mixed layout builder is not initialized");
  if(!this.staticReady)throw new Error("Mixed layout builder has no static fine mask");
  const n=this.tiles,groups64=Math.ceil(n/64);
  const level=this.level;
  encoder.clearBuffer(level.work,0,RECEIPT*4);
  const pass=encoder.beginComputePass({label:"Uniform mixed layout build"});pass.setBindGroup(0,level.group);
  for(const [entry,groups] of [["widths",groups64],["classify",this.blocks],["scan",1],["scatter",this.blocks],["distance0",groups64],["distance1",groups64],["distance2",groups64]] as const){
   pass.setPipeline(this.pipelines.get(entry)!);pass.dispatchWorkgroups(groups);
  }
  pass.end();
  encoder.copyBufferToBuffer(level.work,0,this.readback,0,RECEIPT*4);
  encoder.copyBufferToBuffer(level.topology,0,this.readback,RECEIPT*4,n*4);
  this.encoded=true;
 }
 /** Map the generation built by the last submitted encode(). */
 async read():Promise<UniformMixedBuiltLevel>{
  if(!this.encoded)throw new Error("Mixed layout builder was not encoded");
  this.encoded=false;
  await this.readback.mapAsync(GPUMapMode.READ);
  const words=new Uint32Array(this.readback.getMappedRange()).slice();this.readback.unmap();
  const n=this.tiles,level=this.level,lattice=level.current.layout.lattice;
  const r=words.subarray(0,RECEIPT),tiles=words.slice(RECEIPT,RECEIPT+n);
  if(r[1]!+r[3]!!==n)throw new Error(`Mixed layout builder receipt is inconsistent: ${[...r]}`);
  const layout=uniformMixedLayoutFromTiles(lattice,tiles,this.regions);
  if(layout.cellCount!==64*r[1]!+r[3]!)throw new Error("Mixed layout builder tile words disagree with its receipt");
  return {changedTiles:r[0]!,tierCounts:[r[1]!,0,r[3]!],layout,seamCounts:[r[5]!,0,r[7]!],hangingSlots:r[5]!+r[7]!,
   source:{topology:level.topology,support:level.support,slots:level.slots,counts:{buffer:level.work,offset:48}}};
 }
 destroy():void{
  this.statics.destroy();this.readback.destroy();
  const l=this.level;l.topology.destroy();l.support.destroy();l.slots.destroy();l.work.destroy();l.params.destroy();
 }
}
