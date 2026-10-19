/** The core's velocity pair (docs/plans/brick-arena-core-2026-10-20.md):
 * one brick arena (uniform-brick-arena.ts) in one buffer, the table at word
 * 0 and the pool after it, holding two face channels a live level 0 brick:
 * u (the extended, projected field the next move's G2P and surface advect
 * sample) and uStar (the pre-pressure field the snapshot takes, for the
 * FLIP delta), and a plane channel for the faces just below a brick; and a
 * 4h store a block (after the pool): the 4h cache block's texels, -1..8 an
 * axis, u and uStar.
 *
 * ucPair is umSampleVelocity term for term, bit for bit: the h channels hold
 * the unit taps' faces (the unit halo texels the h interpolant reads), the
 * 4h store the 4h interpolant's taps, and the slot's tile words (uplane
 * words 384..391) the h interpolant's weight (support bit, width, blend
 * mask). Checked against the sampler on the pour, frames 5-95 at 1/60 s:
 * zero difference in fine, pure 4h and blended tiles, no misses.
 *
 * Layout: channel index i = x + 8y + 64z of a brick holds its cell's +axis
 * face (anchor a: the face at a + e_axis, as textureLoad(velocity, a)[axis]
 * and umSampleVelocityFine read it). The plane channel holds, a field and
 * an axis, the 64 faces at anchor origin[axis] - 1 (u + 8v over the other
 * two axes in cyclic order): a tap there reads it where the brick below is
 * not live (a low wall plane: the native negative boundary table today).
 *
 * Per step, in the frame (UniformMixedFrame):
 * 1. encodeFill, at the head after the 4h cache and the hanging taps, before
 *    the move: u at every live slot's faces and planes (dense copies of the
 *    page's unit halo), the tile words, and the 4h store (the cache's
 *    blocks).
 * 2. encodeClear: the marks (the slot table, records, live list and counts
 *    stand, so the move's G2P and the surface advect still resolve through
 *    them; the counts are cleared just before the build).
 * 3. The markers: NB-FLIP marks every brick with a current slab and every
 *    spill record's brick UBA_PARTICLES_CORE after SORT (markWGSL).
 * 4. encodeUpdate, after the phi stage: the surface arena's crossing bricks
 *    (NB-FLIP's brick table marks, UB_ARENA_CORE, of this step's
 *    redistance) marked UBA_ARENA_CORE, the table build (cores dilate into
 *    their one-brick ring), then the leaving slots released. The next
 *    step's consumers (the move's G2P) find those slots in that table.
 * 5. The snapshot writes uStar (velocityWGSL's ucStore, ucStorePlane and
 *    ucStoreCoarse, star = true).
 * Both fields then stand until the next move's G2P reads them (ucPair).
 *
 * A pop from an empty free stack is fatal (the receipt word, frame
 * diagnostics): the capacity is paid upfront, never grown. */
import { UniformBrickArena, UNIFORM_ARENA_CLASS, UNIFORM_ARENA_TABLE_ENTRIES, UNIFORM_ARENA_RELEASE_ENTRY } from "./uniform-brick-arena";
import { uniformDetailBindLayout, uniformDetailGroup, uniformDetailModule, uniformDetailPipeline, type UniformDetailGroup } from "./uniform-detail-fields";
import type { UniformMixedOwnership } from "./uniform-mixed-ownership";
import { uniformMixedTopologyWGSL } from "./uniform-mixed-topology.wgsl";
import { uniformUnitHaloWGSL } from "./uniform-mixed-velocity-sampling.wgsl";
import { UNIFORM_BRICK_CLASS } from "./uniform-brick-table";
import { UNIFORM_COARSE_CACHE_EDGE, UNIFORM_COARSE_CACHE_REACH, uniformCoarseCacheWGSL } from "./uniform-coarse-velocity-cache";

const CORE_FILL_PASS:GPUComputePassDescriptor={label:"Core velocity fill"},CORE_TABLE_PASS:GPUComputePassDescriptor={label:"Core velocity table"};
/** Workgroups of a grid-stride launch. */
const CORE_GRID=1024;

/** NB-FLIP's brick table (uniform-brick-table.ts): its buffer and first word. */
export interface UniformCoreSurfaceMarks{readonly buffer:GPUBuffer;readonly word:number}
export interface UniformCoreVelocityFields{velocity:GPUTexture;coarseVelocity:GPUTexture;negative:GPUBuffer;unitVelocity:GPUTexture;unitHalo:GPUTexture}

export class UniformCoreVelocity{
 readonly arena:UniformBrickArena;
 /** The table at word 0, the pool from poolBase. */
 readonly buffer:GPUBuffer;
 readonly poolBase:number;
 /** The 4h store's first word: a block's cache texels (-1..8 an axis), u
  * then uStar, three components each. */
 readonly coarseBase:number;
 private readonly resources:GPUBindGroupLayout;
 private table?:Map<string,GPUComputePipeline>;
 private fill?:GPUComputePipeline;
 private fillCoarse?:GPUComputePipeline;
 private readonly brickGroups:number;
 private readonly slotGroups:number;
 private readonly recordGroups:number;
 private readonly coarseGroups:number;
 private readonly markGroups:number;
 /** surfaceMarks: the surface arena's brick marks, copied into the table
  * each step (encodeUpdate). capacity: level 0 slots (default every brick
  * of the lattice: no overflow). */
 constructor(private readonly device:GPUDevice,private readonly ownership:UniformMixedOwnership,private readonly surfaceMarks:UniformCoreSurfaceMarks,capacity?:number){
  const blocks=ownership.capacity.address.blocks;
  this.arena=new UniformBrickArena({blocks,fineCapacity:capacity===undefined?undefined:Math.min(capacity,64*blocks),lists:[],channels:[
   {name:"u",levels:[0],layout:"face"},{name:"ustar",levels:[0],layout:"face"},{name:"uplane",levels:[0],layout:"cell"},
  ]});
  this.poolBase=Math.ceil(this.arena.tableWords/64)*64;
  this.coarseBase=Math.ceil((this.poolBase+this.arena.poolWords)/64)*64;
  const bytes=4*(this.coarseBase+6*UNIFORM_COARSE_CACHE_EDGE**3*blocks),limit=Math.min(device.limits.maxStorageBufferBindingSize,device.limits.maxBufferSize);
  if(bytes>limit)throw new Error(`Core velocity arena needs ${bytes} bytes for ${this.arena.fineCapacity} slots, exceeding the device buffer limit ${limit}`);
  this.buffer=device.createBuffer({label:"Core velocity arena",size:bytes,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC});
  for(const write of this.arena.initialWrites(0))device.queue.writeBuffer(this.buffer,write.offset,write.data as Uint32Array<ArrayBuffer>);
  const cap=this.arena.fineCapacity;
  this.brickGroups=Math.min(CORE_GRID,Math.ceil(this.arena.bricks/64));
  this.slotGroups=Math.min(CORE_GRID,Math.ceil(Math.max(cap,blocks)/64));
  this.recordGroups=Math.min(CORE_GRID,Math.ceil(32*(2*cap+blocks)/64));
  this.coarseGroups=Math.min(CORE_GRID,Math.ceil(UNIFORM_COARSE_CACHE_EDGE**3*blocks/64));
  this.markGroups=this.brickGroups;
  this.resources=uniformDetailBindLayout(device,{entries:[
   ...[2,3,11,14].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float" as const,viewDimension:"3d" as const}})),
   {binding:5,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},
   {binding:16,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},
   {binding:17,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},
  ]},[14]);
 }
 /** Bytes of the buffer, for a consumer's binding (offset 0, whole buffer). */
 get bytes():number{return this.buffer.size;}
 /** The fatal word's byte offset (0, or UBA_SEAM | the brick refused a slot). */
 get fatalOffset():number{return this.arena.fatalOffset(0);}

 /** The read library of a module binding the buffer as name, an
  * array<u32> (read_write when writable: then also the stores). Needs the
  * lattice address.
  * - ucPair(p, lo, hi): u and uStar at p (in cells), umSampleVelocity's
  *   value, held to the sample's walls lo, hi (its umHold:
  *   umPointBounds(p, 8)); hit is false where p's brick or a weighted h tap
  *   has no live slot or plane.
  * - ucVelocity(p, lo, hi, star): one field, w = 1 on a hit.
  * - ucLiveCount, ucLiveSlot(e), ucOrigin(slot): the live slots.
  * - ucStore(slot, i, axis, star, value), ucStorePlane(slot, uv, axis,
  *   star, value): a face and a plane face; ucStoreCoarse(block, t, axis,
  *   star, value): a 4h store texel (t -1..8 an axis) (writable only).
  * - ucWithStar (private, default true): false loads u alone. */
 velocityWGSL(name:string,writable:boolean):string{
  const component=520*this.arena.fineCapacity,base=(channel:string)=>this.poolBase+this.arena.base(channel,0);
  const blocks=this.ownership.capacity.address.blocks,edge=UNIFORM_COARSE_CACHE_EDGE;
  if(this.arena.base("u",0,2)!==this.arena.base("u",0,0)+2*component||this.arena.base("ustar",0,2)!==this.arena.base("ustar",0,0)+2*component)throw new Error("Core velocity components are not consecutive");
  return this.arena.readerWGSL(name,0)+/* wgsl */`
const UC_U:u32=${base("u")}u;
const UC_STAR:u32=${base("ustar")}u;
const UC_PLANE:u32=${base("uplane")}u;
const UC_COMPONENT:u32=${component}u;
fn ucLoad(at:u32,slot:u32,i:u32)->f32{return bitcast<f32>(${name}[at+520u*slot+i]);}
fn ucFaceBase(axis:u32,star:bool)->u32{return select(UC_U,UC_STAR,star)+axis*UC_COMPONENT;}
fn ucLiveCount()->u32{return ubrLiveCount();}
fn ucLiveSlot(e:u32)->u32{return ubrLiveSlot(e);}
fn ucOrigin(slot:u32)->vec3i{return vec3i(ubrOrigin(ubrBrick(slot)));}
${writable?`fn ucStore(slot:u32,i:u32,axis:u32,star:bool,value:f32){${name}[ucFaceBase(axis,star)+520u*slot+i]=bitcast<u32>(value);}
fn ucStorePlane(slot:u32,uv:u32,axis:u32,star:bool,value:f32){${name}[UC_PLANE+520u*slot+select(0u,192u,star)+64u*axis+uv]=bitcast<u32>(value);}`:""}
struct UCPair{u:vec3f,uStar:vec3f,hit:bool}
var<private> ucHit:bool=true;
// false: ucPair leaves uStar zero and loads only u (a reader of u alone).
var<private> ucWithStar:bool=true;
const UC_CORE:u32=${UNIFORM_ARENA_CLASS.particlesCore}u;
// A slot of any of these classes is filled whole; a particle ring slot only near its cores.
const UC_FULL:u32=${UNIFORM_ARENA_CLASS.particlesCore|UNIFORM_ARENA_CLASS.arenaCore|UNIFORM_ARENA_CLASS.arena}u;
// The 4h store: block b's cache texel t (-1..8 an axis) of a component.
const UC_COARSE:u32=${this.coarseBase}u;
const UC_BLOCKS:u32=${blocks}u;
fn ucCoarseIndex(block:u32,t:vec3i,axis:u32,star:bool)->u32{return UC_COARSE+((select(0u,3u,star)+axis)*UC_BLOCKS+block)*${edge**3}u+u32(t.x+1)+${edge}u*u32(t.y+1)+${edge*edge}u*u32(t.z+1);}
// A slot's tile words (uplane words 384 + tile x + 2y + 4z): the velocity
// blend mask, UC_SUPPORT (support bit 0) and UC_UNIT (width 1).
const UC_TILE_WORD:u32=384u;
const UC_SUPPORT:u32=0x08000000u;
const UC_UNIT:u32=0x10000000u;
fn ucRecordIndex(d:vec3i)->u32{return u32(d.x+1+3*(d.y+1)+9*(d.z+1));}
// Face anchor a on axis, both fields, through the record of home (the slot
// of cell c0's brick). Below a brick that is not live, its plane.
fn ucTap(home:u32,c0:vec3i,a:vec3i,axis:u32)->vec2f{
 let d=(a>>vec3u(3u))-(c0>>vec3u(3u));let l=vec3u(a&vec3i(7));
 if(any(abs(d)>vec3i(1))){ucHit=false;return vec2f(0.0);}
 let w=ubrRecord(home,ucRecordIndex(d));
 if(w<UBR_SEAM){let i=l.x+8u*(l.y+8u*l.z);return vec2f(ucLoad(ucFaceBase(axis,false),w,i),select(0.0,ucLoad(ucFaceBase(axis,true),w,i),ucWithStar));}
 var e=d;e[axis]+=1;
 if(l[axis]!=7u||e[axis]>1){ucHit=false;return vec2f(0.0);}
 let v=ubrRecord(home,ucRecordIndex(e));
 if(v>=UBR_SEAM){ucHit=false;return vec2f(0.0);}
 let uv=l[(axis+1u)%3u]+8u*l[(axis+2u)%3u];
 return vec2f(ucLoad(UC_PLANE,v,64u*axis+uv),select(0.0,ucLoad(UC_PLANE,v,192u+64u*axis+uv),ucWithStar));
}
fn ucLerp8(v:array<f32,8>,f:vec3f)->f32{
 let a=(1.0-f.x)*v[0]+f.x*v[1];let b=(1.0-f.x)*v[2]+f.x*v[3];
 let c=(1.0-f.x)*v[4]+f.x*v[5];let d=(1.0-f.x)*v[6]+f.x*v[7];
 return (1.0-f.z)*((1.0-f.y)*a+f.y*b)+f.z*((1.0-f.y)*c+f.y*d);
}
// umSampleVelocity at p, term for term: the sample's tile (p held to lo,
// hi, its walls: umHold) gives the h interpolant's weight f (its blend);
// the h interpolant's taps are the unit taps' faces of the slots, the 4h
// one's the 4h store of the tile's block, at today's plain offsets.
fn ucPair(p:vec3f,lo:vec3f,hi:vec3f)->UCPair{
 ucHit=true;let q=clamp(p,lo,hi);let c0=vec3i(clamp(floor(q),lo,hi-vec3f(1.0)));
 let id=ubrBrickAt(c0);let home=ubrSlotOf(id);
 if(home==UBR_NONE){return UCPair(vec3f(0.0),vec3f(0.0),false);}
 let tile=c0>>vec3u(2u);
 let info=bitcast<u32>(ucLoad(UC_PLANE,home,UC_TILE_WORD+u32(tile.x&1)+2u*u32(tile.y&1)+4u*u32(tile.z&1)));
 var f=0.0;
 if((info&UC_SUPPORT)!=0u){
  if((info&UC_UNIT)!=0u){f=1.0;}
  else{
   var remaining=info&0x07ffffffu;
   while(remaining!=0u){
    let bit=firstTrailingBit(remaining);remaining&=remaining-1u;
    let t=tile+vec3i(i32(bit%3u),i32((bit/3u)%3u),i32(bit/9u))-vec3i(1);
    let origin=vec3f(t)*4.0;let delta=max(vec3f(0),max(origin-q,q-origin-vec3f(4.0)));
    let distance=max(delta.x,max(delta.y,delta.z));
    f=max(f,max(0.0,1.0-distance/2.0));
   }
  }
 }
 let coarse=max(0.0,1.0-f);let block=id>>6u;let local=p/4.0-vec3f(tile-(tile&vec3i(7)));
 var u=vec3f(0.0);var s=vec3f(0.0);
 for(var axis=0u;axis<3u;axis++){
  var offset=vec3f(0.5);offset[axis]=1.0;
  if(f>0.0){
   var lower=lo;lower[axis]-=1.0;
   let r=clamp(p-offset,lower,hi-vec3f(1.0));let base=vec3i(floor(r));let fraction=fract(r);
   var tu:array<f32,8>;var ts:array<f32,8>;
   for(var k=0u;k<8u;k++){
    let bit=vec3i(i32(k&1u),i32((k>>1u)&1u),i32(k>>2u));
    let w=select(vec3f(1.0)-fraction,fraction,bit==vec3i(1));
    tu[k]=0.0;ts[k]=0.0;
    if(w.x*w.y*w.z>0.0){let v=ucTap(home,c0,base+bit,axis);tu[k]=v.x;ts[k]=v.y;}
   }
   u[axis]+=f*ucLerp8(tu,fraction);s[axis]+=f*ucLerp8(ts,fraction);
  }
  if(coarse>0.0){
   let r=clamp(local-offset,vec3f(-1.0),vec3f(${UNIFORM_COARSE_CACHE_REACH}));let at=vec3i(floor(r));let fraction=fract(r);
   var tu:array<f32,8>;var ts:array<f32,8>;
   for(var k=0u;k<8u;k++){
    let t=at+vec3i(i32(k&1u),i32((k>>1u)&1u),i32(k>>2u));
    tu[k]=bitcast<f32>(${name}[ucCoarseIndex(block,t,axis,false)]);ts[k]=select(0.0,bitcast<f32>(${name}[ucCoarseIndex(block,t,axis,true)]),ucWithStar);
   }
   u[axis]+=coarse*ucLerp8(tu,fraction);s[axis]+=coarse*ucLerp8(ts,fraction);
  }
 }
 return UCPair(u,s,ucHit);
}
fn ucVelocity(p:vec3f,lo:vec3f,hi:vec3f,star:bool)->vec4f{let pair=ucPair(p,lo,hi);return vec4f(select(pair.u,pair.uStar,star),select(0.0,1.0,pair.hit));}
${writable?`fn ucStoreCoarse(block:u32,t:vec3i,axis:u32,star:bool,value:f32){${name}[ucCoarseIndex(block,t,axis,star)]=bitcast<u32>(value);}`:""}
`;}
 /** A marker's library (a binding of the buffer as array<atomic<u32>> read_write). */
 markWGSL(name:string):string{return this.arena.markWGSL(name,0);}

 bind(f:UniformCoreVelocityFields):UniformDetailGroup{
  return uniformDetailGroup(this.device,{layout:this.resources,entries:[
   {binding:2,resource:f.velocity},{binding:3,resource:f.coarseVelocity},{binding:11,resource:f.unitVelocity},{binding:14,resource:f.unitHalo},
   {binding:5,resource:{buffer:f.negative}},{binding:16,resource:{buffer:this.buffer}},{binding:17,resource:{buffer:this.surfaceMarks.buffer}},
  ]});
 }
 async initialize():Promise<void>{
  const topology=uniformMixedTopologyWGSL(this.ownership.capacity,0);
  const tableModule=uniformDetailModule(this.device,{specializeWorldGeometry:true,label:"Core velocity table",code:topology+/* wgsl */`
@group(1) @binding(16) var<storage,read_write> ucTable:array<atomic<u32>>;
@group(1) @binding(17) var<storage,read> ucSurfaceMarks:array<u32>;
${this.arena.tableWGSL("ucTable","0u")}
// A thread a brick: the surface arena's crossing bricks (the brick table's
// marks, UB_ARENA_CORE) are arena cores here too.
@compute @workgroup_size(64) fn ucMarkSurface(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) groups:vec3u){
 for(var id=gid.x;id<UBA_BRICKS;id+=64u*groups.x){
  if((ucSurfaceMarks[${this.surfaceMarks.word+4}u+id]&${UNIFORM_BRICK_CLASS.arenaCore}u)!=0u){ubaMark(id,${UNIFORM_ARENA_CLASS.arenaCore}u);}
 }
}
`});
  const fillModule=uniformDetailModule(this.device,{specializeWorldGeometry:true,label:"Core velocity fill",code:topology+/* wgsl */`
@group(1) @binding(3) var coarseVelocity:texture_3d<f32>;
@group(1) @binding(14) var unitHalo:texture_3d<f32>;
@group(1) @binding(16) var<storage,read_write> ucArena:array<u32>;
${uniformUnitHaloWGSL}
${uniformCoarseCacheWGSL}
${this.velocityWGSL("ucArena",true)}
// A workgroup (256 lanes) a live slot: dense copies of the unit taps' halo block of the
// slot's page (the texels umSampleVelocity's h interpolant reads), one load
// a cell for its three faces. Items 0..511 are the cells (x fastest); a
// particle ring slot (no full class) copies only the cells within one of a
// particle core brick, the only ones a G2P tap reaches. Items 512..703 are
// the planes, an axis's 64, only where the brick below along the axis is not
// live. Lanes 0..7 write the tile words ucPair's weight reads.
var<workgroup> ucFillCount:u32;
var<workgroup> ucFillNeed:array<u32,27>;
@compute @workgroup_size(256) fn ucFill(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 if(lane==0u){ucFillCount=ucLiveCount();}
 let n=workgroupUniformLoad(&ucFillCount);
 for(var e=group.x;e<n;e+=groups.x){
  let slot=ucLiveSlot(e);let origin=ucOrigin(slot);let id=ubrBrick(slot);
  let full=(ubrClass(slot)&UC_FULL)!=0u;
  let halo=umUnitHaloOrigin(id>>6u)-((origin>>vec3u(5u))<<vec3u(5u));
  workgroupBarrier();
  // A neighbour's word: bit 0 a particle core brick, bit 1 a live slot.
  if(lane<27u){let w=ubrRecord(slot,lane);var need=0u;if(w<UBR_SEAM){need=2u|select(0u,1u,(ubrClass(w)&UC_CORE)!=0u);}ucFillNeed[lane]=need;}
  if(lane<8u){
   let key=umTileAt(vec3u(origin>>vec3u(2u))+vec3u(lane&1u,(lane>>1u)&1u,lane>>2u));
   let info=(umVelocityBlendMask(key)&0x07ffffffu)|select(0u,UC_SUPPORT,(umTileSupport(key)&1u)!=0u)|select(0u,UC_UNIT,umTileWidth(key)==1u);
   ucArena[UC_PLANE+520u*slot+UC_TILE_WORD+lane]=info;
  }
  workgroupBarrier();
  for(var j=lane;j<704u;j+=256u){
   if(j<512u){
    let l=vec3u(j&7u,(j>>3u)&7u,j>>6u);
    if(!full){
     let lo=vec3i(select(vec3u(0u),vec3u(1u),l==vec3u(0u)));let hi=vec3i(select(vec3u(0u),vec3u(1u),l==vec3u(7u)));var near=false;
     for(var dz=-lo.z;dz<=hi.z;dz++){for(var dy=-lo.y;dy<=hi.y;dy++){for(var dx=-lo.x;dx<=hi.x;dx++){near=near||(ucFillNeed[ucRecordIndex(vec3i(dx,dy,dz))]&1u)!=0u;}}}
     if(!near){continue;}
    }
    let v=textureLoad(unitHalo,origin+vec3i(l)+halo,0);
    for(var a=0u;a<3u;a++){ucStore(slot,j,a,false,v[a]);}
   }else{
    let k=j-512u;let axis=k/64u;let uv=k%64u;
    var d=vec3i(0);d[axis]=-1;if((ucFillNeed[ucRecordIndex(d)]&2u)!=0u){continue;}
    var l=vec3i(0);l[(axis+1u)%3u]=i32(uv&7u);l[(axis+2u)%3u]=i32(uv>>3u);l[axis]=-1;
    ucStorePlane(slot,uv,axis,false,textureLoad(unitHalo,origin+l+halo,0)[axis]);
   }
  }
 }
}
// A thread a texel of the 4h store: the cache block's texels, copied.
@compute @workgroup_size(64) fn ucFillCoarse(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) groups:vec3u){
 for(var i=gid.x;i<UC_BLOCKS*${UNIFORM_COARSE_CACHE_EDGE**3}u;i+=64u*groups.x){
  let block=i/${UNIFORM_COARSE_CACHE_EDGE**3}u;let r=i%${UNIFORM_COARSE_CACHE_EDGE**3}u;
  let t=vec3i(i32(r%${UNIFORM_COARSE_CACHE_EDGE}u),i32((r/${UNIFORM_COARSE_CACHE_EDGE}u)%${UNIFORM_COARSE_CACHE_EDGE}u),i32(r/${UNIFORM_COARSE_CACHE_EDGE**2}u))-vec3i(1);
  let v=textureLoad(coarseVelocity,umCoarseOrigin(block)+t,0);
  for(var a=0u;a<3u;a++){ucStoreCoarse(block,t,a,false,v[a]);}
 }
}
`});
  for(const module of [tableModule,fillModule]){
   const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");
   if(errors.length)throw new Error(errors.map(m=>`${m.lineNum}: ${m.message}`).join("\n"));
  }
  const tableLayout=this.device.createPipelineLayout({bindGroupLayouts:[this.ownership.bindLayout,this.resources]});
  const fillLayout=this.device.createPipelineLayout({bindGroupLayouts:[this.ownership.bindLayout,this.resources]});
  const constants={umDispatchX:this.ownership.dispatchX};
  const entries=["ucMarkSurface",...UNIFORM_ARENA_TABLE_ENTRIES.filter(entry=>entry!=="ubaLevel1"),UNIFORM_ARENA_RELEASE_ENTRY];
  const table=await Promise.all(entries.map(entryPoint=>uniformDetailPipeline(this.device,this.ownership,{layout:tableLayout,compute:{module:tableModule,entryPoint,constants}})));
  this.table=new Map(entries.map((entry,i)=>[entry,table[i]!]));
  [this.fill,this.fillCoarse]=await Promise.all(["ucFill","ucFillCoarse"].map(entryPoint=>uniformDetailPipeline(this.device,this.ownership,{layout:fillLayout,compute:{module:fillModule,entryPoint,constants}})));
 }
 /** u at every live slot (the head, after the 4h cache and hanging taps). */
 encodeFill(encoder:GPUCommandEncoder,group:UniformDetailGroup):void{
  if(!this.fill)throw new Error("Core velocity is not initialized");
  const pass=encoder.beginComputePass(CORE_FILL_PASS);
  pass.setBindGroup(0,this.ownership.bindGroup);pass.setBindGroup(1,group.group);
  pass.setPipeline(this.fill);pass.dispatchWorkgroups(Math.min(CORE_GRID,this.arena.fineCapacity));
  pass.setPipeline(this.fillCoarse!);pass.dispatchWorkgroups(this.coarseGroups);pass.end();
 }
 /** The marks, before the step's markers. The counts stand until the
  * build (encodeUpdate): the live list serves the step's readers. */
 encodeClear(encoder:GPUCommandEncoder):void{const range=this.arena.markClear(0);encoder.clearBuffer(this.buffer,range.offset,range.size);}
 /** The surface marks, the table build after the markers, then the leaving slots released. */
 encodeUpdate(encoder:GPUCommandEncoder,group:UniformDetailGroup):void{
  const table=this.table;if(!table)throw new Error("Core velocity is not initialized");
  const counts=this.arena.countClear(0);encoder.clearBuffer(this.buffer,counts.offset,counts.size);
  const pass=encoder.beginComputePass(CORE_TABLE_PASS);
  pass.setBindGroup(0,this.ownership.bindGroup);pass.setBindGroup(1,group.group);
  const run=(entry:string,groups:number)=>{pass.setPipeline(table.get(entry)!);pass.dispatchWorkgroups(groups);};
  run("ucMarkSurface",this.markGroups);run("ubaUpdate",this.brickGroups);run("ubaLists",this.slotGroups);run("ubaRecord",this.recordGroups);run(UNIFORM_ARENA_RELEASE_ENTRY,this.slotGroups);
  pass.end();
 }
 destroy():void{this.buffer.destroy();}
}
