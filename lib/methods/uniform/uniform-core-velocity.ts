/** The core's velocity pair (docs/plans/brick-arena-core-2026-10-20.md):
 * one brick arena (uniform-brick-arena.ts) in one buffer, the table at word
 * 0 and the pool after it, holding two face channels a live level 0 brick:
 * u (the extended, projected field the next move's G2P and surface advect
 * sample) and uStar (the pre-pressure field the snapshot takes, for the
 * FLIP delta), and a plane channel for the faces just below a brick.
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
 *    the move: u at every live slot's faces and planes, through the mixed
 *    sampler at the face (exact at h faces).
 * 2. encodeClear: the marks and counts (the slot table and records stand,
 *    so the move's G2P still resolves through them).
 * 3. The markers: NB-FLIP marks every brick with a current slab and every
 *    spill record's brick UBA_PARTICLES_CORE after SORT (markWGSL).
 * 4. encodeUpdate, after the phi stage: the table build (particle cores
 *    dilate into their one-brick ring), then the leaving slots released.
 * 5. The snapshot writes uStar (velocityWGSL's ucStore, star = true).
 * Both fields then stand until the next move's G2P reads them (ucPair).
 *
 * A pop from an empty free stack is fatal (the receipt word, frame
 * diagnostics): the capacity is paid upfront, never grown. */
import { UniformBrickArena, UNIFORM_ARENA_CLASS, UNIFORM_ARENA_TABLE_ENTRIES, UNIFORM_ARENA_RELEASE_ENTRY } from "./uniform-brick-arena";
import { uniformDetailBindLayout, uniformDetailGroup, uniformDetailModule, uniformDetailPipeline, type UniformDetailGroup } from "./uniform-detail-fields";
import type { UniformMixedOwnership } from "./uniform-mixed-ownership";
import { uniformMixedTopologyWGSL } from "./uniform-mixed-topology.wgsl";
import { uniformMixedFaceAddressWGSL } from "./uniform-mixed-face-dispatch.wgsl";
import { uniformMixedVelocitySamplingSource } from "./uniform-mixed-velocity-sampling.wgsl";

const CORE_FILL_PASS:GPUComputePassDescriptor={label:"Core velocity fill"},CORE_TABLE_PASS:GPUComputePassDescriptor={label:"Core velocity table"};
/** Workgroups of a grid-stride launch. */
const CORE_GRID=1024;

export interface UniformCoreVelocityFields{velocity:GPUTexture;coarseVelocity:GPUTexture;negative:GPUBuffer;unitVelocity:GPUTexture;unitHalo:GPUTexture}

export class UniformCoreVelocity{
 readonly arena:UniformBrickArena;
 /** The table at word 0, the pool from poolBase. */
 readonly buffer:GPUBuffer;
 readonly poolBase:number;
 private readonly resources:GPUBindGroupLayout;
 private table?:Map<string,GPUComputePipeline>;
 private fill?:GPUComputePipeline;
 private readonly brickGroups:number;
 private readonly slotGroups:number;
 private readonly recordGroups:number;
 /** capacity: level 0 slots (default every brick of the lattice: no overflow). */
 constructor(private readonly device:GPUDevice,private readonly ownership:UniformMixedOwnership,capacity?:number){
  const blocks=ownership.capacity.address.blocks;
  this.arena=new UniformBrickArena({blocks,fineCapacity:capacity===undefined?undefined:Math.min(capacity,64*blocks),lists:[],channels:[
   {name:"u",levels:[0],layout:"face"},{name:"ustar",levels:[0],layout:"face"},{name:"uplane",levels:[0],layout:"cell"},
  ]});
  this.poolBase=Math.ceil(this.arena.tableWords/64)*64;
  const bytes=4*(this.poolBase+this.arena.poolWords),limit=Math.min(device.limits.maxStorageBufferBindingSize,device.limits.maxBufferSize);
  if(bytes>limit)throw new Error(`Core velocity arena needs ${bytes} bytes for ${this.arena.fineCapacity} slots, exceeding the device buffer limit ${limit}`);
  this.buffer=device.createBuffer({label:"Core velocity arena",size:bytes,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC});
  for(const write of this.arena.initialWrites(0))device.queue.writeBuffer(this.buffer,write.offset,write.data as Uint32Array<ArrayBuffer>);
  const cap=this.arena.fineCapacity;
  this.brickGroups=Math.min(CORE_GRID,Math.ceil(this.arena.bricks/64));
  this.slotGroups=Math.min(CORE_GRID,Math.ceil(Math.max(cap,blocks)/64));
  this.recordGroups=Math.min(CORE_GRID,Math.ceil(32*(2*cap+blocks)/64));
  this.resources=uniformDetailBindLayout(device,{entries:[
   ...[2,3,11,14].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float" as const,viewDimension:"3d" as const}})),
   {binding:5,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},
   {binding:16,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},
  ]},[14]);
 }
 /** Bytes of the buffer, for a consumer's binding (offset 0, whole buffer). */
 get bytes():number{return this.buffer.size;}
 /** The fatal word's byte offset (0, or UBA_SEAM | the brick refused a slot). */
 get fatalOffset():number{return this.arena.fatalOffset(0);}

 /** The read library of a module binding the buffer as name, an
  * array<u32> (read_write when writable: then also the stores). Needs the
  * lattice address.
  * - ucPair(p, lo, hi): u and uStar at p (in cells) by the MAC trilinear
  *   rule of umSampleVelocityFine, held to the sample's walls lo, hi (its
  *   umHold); hit is false where a weighted tap has no live slot or plane.
  * - ucVelocity(p, lo, hi, star): one field, w = 1 on a hit.
  * - ucLiveCount, ucLiveSlot(e), ucOrigin(slot): the live slots.
  * - ucStore(slot, i, axis, star, value), ucStorePlane(slot, uv, axis,
  *   star, value): a face and a plane face (writable only). */
 velocityWGSL(name:string,writable:boolean):string{
  const component=520*this.arena.fineCapacity,base=(channel:string)=>this.poolBase+this.arena.base(channel,0);
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
const UC_CORE:u32=${UNIFORM_ARENA_CLASS.particlesCore}u;
// A slot of any of these classes is filled whole; a particle ring slot only near its cores.
const UC_FULL:u32=${UNIFORM_ARENA_CLASS.particlesCore|UNIFORM_ARENA_CLASS.arenaCore|UNIFORM_ARENA_CLASS.arena}u;
fn ucRecordIndex(d:vec3i)->u32{return u32(d.x+1+3*(d.y+1)+9*(d.z+1));}
// Face anchor a on axis, both fields, through the record of home (the slot
// of cell c0's brick). Below a brick that is not live, its plane.
fn ucTap(home:u32,c0:vec3i,a:vec3i,axis:u32)->vec2f{
 let d=(a>>vec3u(3u))-(c0>>vec3u(3u));let l=vec3u(a&vec3i(7));
 if(any(abs(d)>vec3i(1))){ucHit=false;return vec2f(0.0);}
 let w=ubrRecord(home,ucRecordIndex(d));
 if(w<UBR_SEAM){let i=l.x+8u*(l.y+8u*l.z);return vec2f(ucLoad(ucFaceBase(axis,false),w,i),ucLoad(ucFaceBase(axis,true),w,i));}
 var e=d;e[axis]+=1;
 if(l[axis]!=7u||e[axis]>1){ucHit=false;return vec2f(0.0);}
 let v=ubrRecord(home,ucRecordIndex(e));
 if(v>=UBR_SEAM){ucHit=false;return vec2f(0.0);}
 let uv=l[(axis+1u)%3u]+8u*l[(axis+2u)%3u];
 return vec2f(ucLoad(UC_PLANE,v,64u*axis+uv),ucLoad(UC_PLANE,v,192u+64u*axis+uv));
}
fn ucLerp8(v:array<f32,8>,f:vec3f)->f32{
 let a=(1.0-f.x)*v[0]+f.x*v[1];let b=(1.0-f.x)*v[2]+f.x*v[3];
 let c=(1.0-f.x)*v[4]+f.x*v[5];let d=(1.0-f.x)*v[6]+f.x*v[7];
 return (1.0-f.z)*((1.0-f.y)*a+f.y*b)+f.z*((1.0-f.y)*c+f.y*d);
}
fn ucPair(p:vec3f,lo:vec3f,hi:vec3f)->UCPair{
 ucHit=true;let c0=vec3i(floor(p));
 let home=ubrSlotOf(ubrBrickAt(c0));
 if(home==UBR_NONE){return UCPair(vec3f(0.0),vec3f(0.0),false);}
 var u=vec3f(0.0);var s=vec3f(0.0);
 for(var axis=0u;axis<3u;axis++){
  var offset=vec3f(0.5);offset[axis]=1.0;var lower=lo;lower[axis]-=1.0;
  let q=clamp(p-offset,lower,hi-vec3f(1.0));let base=vec3i(floor(q));let f=fract(q);
  var tu:array<f32,8>;var ts:array<f32,8>;
  for(var k=0u;k<8u;k++){
   let bit=vec3i(i32(k&1u),i32((k>>1u)&1u),i32(k>>2u));
   let w=select(vec3f(1.0)-f,f,bit==vec3i(1));
   tu[k]=0.0;ts[k]=0.0;
   if(w.x*w.y*w.z>0.0){let v=ucTap(home,c0,base+bit,axis);tu[k]=v.x;ts[k]=v.y;}
  }
  u[axis]=ucLerp8(tu,f);s[axis]=ucLerp8(ts,f);
 }
 return UCPair(u,s,ucHit);
}
fn ucVelocity(p:vec3f,lo:vec3f,hi:vec3f,star:bool)->vec4f{let pair=ucPair(p,lo,hi);return vec4f(select(pair.u,pair.uStar,star),select(0.0,1.0,pair.hit));}
`;}
 /** A marker's library (a binding of the buffer as array<atomic<u32>> read_write). */
 markWGSL(name:string):string{return this.arena.markWGSL(name,0);}

 bind(f:UniformCoreVelocityFields):UniformDetailGroup{
  return uniformDetailGroup(this.device,{layout:this.resources,entries:[
   {binding:2,resource:f.velocity},{binding:3,resource:f.coarseVelocity},{binding:11,resource:f.unitVelocity},{binding:14,resource:f.unitHalo},
   {binding:5,resource:{buffer:f.negative}},{binding:16,resource:{buffer:this.buffer}},
  ]});
 }
 async initialize():Promise<void>{
  const topology=uniformMixedTopologyWGSL(this.ownership.capacity,0);
  const tableModule=uniformDetailModule(this.device,{specializeWorldGeometry:true,label:"Core velocity table",code:topology+/* wgsl */`
@group(1) @binding(16) var<storage,read_write> ucTable:array<atomic<u32>>;
${this.arena.tableWGSL("ucTable","0u")}
`});
  const fillModule=uniformDetailModule(this.device,{specializeWorldGeometry:true,label:"Core velocity fill",code:topology+/* wgsl */`
@group(1) @binding(2) var velocity:texture_3d<f32>;
@group(1) @binding(3) var coarseVelocity:texture_3d<f32>;
@group(1) @binding(5) var<storage,read> negative:array<f32>;
@group(1) @binding(11) var unitVelocity:texture_3d<f32>;
@group(1) @binding(14) var unitHalo:texture_3d<f32>;
@group(1) @binding(16) var<storage,read_write> ucArena:array<u32>;
${uniformMixedFaceAddressWGSL}
fn umLoadMixedFace(anchor:vec3i,axis:u32)->f32{
 if(umFaceLow(anchor,axis)){return negative[umLowFaceIndex(umLowFaceAbove(anchor,axis),axis)];}
 return textureLoad(velocity,anchor,0)[axis];
}
${uniformMixedVelocitySamplingSource(false,"coarseVelocity","velocity",2,"unitVelocity",undefined,"unitHalo")}
${this.velocityWGSL("ucArena",true)}
// The mixed field at face point p as cell c's page sees it: its tile, 4h
// cache block and unit halo home are c's (a face on a page boundary takes
// the side of the cell that holds it), its walls and weights p's.
fn ucFieldAt(p:vec3f,c:vec3i,axis:u32)->f32{
 umHold=umPointBounds(p,8.0);
 let tile=vec3i(umBoundedCell(vec3f(c)+vec3f(0.5),umHold)>>vec3u(2u));
 let s=UMSampleTile(clamp(p,umHold.lo,umHold.hi),tile,umTileAt(vec3u(tile)));
 let fineTile=umCachedFineTile(s.key);umUnitHome=umUnitHaloHome(s);
 var weights=1.0;if(!fineTile){weights=umVelocitySamplingWeightsAt(s);}
 return umSampleVelocityAxis(p,s,fineTile,weights,axis);
}
// A workgroup a live slot. Items 0..511 are its cells (x fastest): a cell
// of a cached unit tile or a slotted seam tile (its hanging taps) takes its
// three faces in one load of the unit taps' halo, exactly what a unit
// interpolant's tap reads (the halo's other texels may be stale); any other
// its three through ucFieldAt. A particle ring brick (no full class)
// fills only the cells within one of a particle core brick: no G2P tap
// reaches its others. Items 512..703 are the planes, an axis's 64, only
// where the brick below along the axis is not live. One sampler call site.
var<workgroup> ucFillCount:u32;
var<workgroup> ucFillNeed:array<u32,27>;
@compute @workgroup_size(64) fn ucFill(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 if(lane==0u){ucFillCount=ucLiveCount();}
 let n=workgroupUniformLoad(&ucFillCount);
 for(var e=group.x;e<n;e+=groups.x){
  let slot=ucLiveSlot(e);let origin=ucOrigin(slot);
  let full=(ubrClass(slot)&UC_FULL)!=0u;
  workgroupBarrier();
  // A neighbour's word: bit 0 a particle core brick, bit 1 a live slot.
  if(lane<27u){let w=ubrRecord(slot,lane);var need=0u;if(w<UBR_SEAM){need=2u|select(0u,1u,(ubrClass(w)&UC_CORE)!=0u);}ucFillNeed[lane]=need;}
  workgroupBarrier();
  for(var j=lane;j<704u;j+=64u){
   var a0=0u;var a1=3u;var anchor=origin;var at=j;let plane=j>=512u;
   if(!plane){
    let l=vec3u(j&7u,(j>>3u)&7u,j>>6u);anchor+=vec3i(l);
    if(!full){
     let lo=vec3i(select(vec3u(0u),vec3u(1u),l==vec3u(0u)));let hi=vec3i(select(vec3u(0u),vec3u(1u),l==vec3u(7u)));var near=false;
     for(var dz=-lo.z;dz<=hi.z;dz++){for(var dy=-lo.y;dy<=hi.y;dy++){for(var dx=-lo.x;dx<=hi.x;dx++){near=near||(ucFillNeed[ucRecordIndex(vec3i(dx,dy,dz))]&1u)!=0u;}}}
     if(!near){continue;}
    }
    let key=umTileAt(vec3u(anchor)>>vec3u(2u));
    if(umCachedFineTile(key)||umHanging[key]!=UM_NO_SLOT){
     let v=textureLoad(unitHalo,anchor+umUnitHaloOrigin(key>>9u)-((anchor>>vec3u(5u))<<vec3u(5u)),0);
     for(var a=0u;a<3u;a++){ucStore(slot,j,a,false,v[a]);}continue;
    }
   }else{
    let k=j-512u;a0=k/64u;a1=a0+1u;at=k%64u;
    var d=vec3i(0);d[a0]=-1;if((ucFillNeed[ucRecordIndex(d)]&2u)!=0u){continue;}
    var l=vec3u(0u);l[(a0+1u)%3u]=at&7u;l[(a0+2u)%3u]=at>>3u;anchor+=vec3i(l);anchor[a0]-=1;
   }
   for(var axis=a0;axis<a1;axis++){
    var p=vec3f(anchor)+vec3f(0.5);p[axis]+=0.5;
    var cell=anchor;if(plane){cell[axis]+=1;}
    let value=ucFieldAt(p,cell,axis);
    if(plane){ucStorePlane(slot,at,axis,false,value);}else{ucStore(slot,at,axis,false,value);}
   }
  }
 }
}
`});
  for(const module of [tableModule,fillModule]){
   const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");
   if(errors.length)throw new Error(errors.map(m=>`${m.lineNum}: ${m.message}`).join("\n"));
  }
  const tableLayout=this.device.createPipelineLayout({bindGroupLayouts:[this.ownership.bindLayout,this.resources]});
  const fillLayout=this.device.createPipelineLayout({bindGroupLayouts:[this.ownership.bindLayout,this.resources,this.ownership.hangingLayout]});
  const constants={umDispatchX:this.ownership.dispatchX};
  const entries=[...UNIFORM_ARENA_TABLE_ENTRIES.filter(entry=>entry!=="ubaLevel1"),UNIFORM_ARENA_RELEASE_ENTRY];
  const table=await Promise.all(entries.map(entryPoint=>uniformDetailPipeline(this.device,this.ownership,{layout:tableLayout,compute:{module:tableModule,entryPoint,constants}})));
  this.table=new Map(entries.map((entry,i)=>[entry,table[i]!]));
  this.fill=await uniformDetailPipeline(this.device,this.ownership,{layout:fillLayout,compute:{module:fillModule,entryPoint:"ucFill",constants}});
 }
 /** u at every live slot (the head, after the 4h cache and hanging taps). */
 encodeFill(encoder:GPUCommandEncoder,group:UniformDetailGroup):void{
  if(!this.fill)throw new Error("Core velocity is not initialized");
  const pass=encoder.beginComputePass(CORE_FILL_PASS);
  pass.setBindGroup(0,this.ownership.bindGroup);pass.setBindGroup(1,group.group);pass.setBindGroup(2,this.ownership.hangingGroup);
  pass.setPipeline(this.fill);pass.dispatchWorkgroups(Math.min(CORE_GRID,this.arena.fineCapacity));pass.end();
 }
 /** The marks and counts, before the step's markers. */
 encodeClear(encoder:GPUCommandEncoder):void{const range=this.arena.stepClear(0);encoder.clearBuffer(this.buffer,range.offset,range.size);}
 /** The table build after the markers, then the leaving slots released. */
 encodeUpdate(encoder:GPUCommandEncoder,group:UniformDetailGroup):void{
  const table=this.table;if(!table)throw new Error("Core velocity is not initialized");
  const pass=encoder.beginComputePass(CORE_TABLE_PASS);
  pass.setBindGroup(0,this.ownership.bindGroup);pass.setBindGroup(1,group.group);
  const run=(entry:string,groups:number)=>{pass.setPipeline(table.get(entry)!);pass.dispatchWorkgroups(groups);};
  run("ubaUpdate",this.brickGroups);run("ubaLists",this.slotGroups);run("ubaRecord",this.recordGroups);run(UNIFORM_ARENA_RELEASE_ENTRY,this.slotGroups);
  pass.end();
 }
 destroy():void{this.buffer.destroy();}
}
