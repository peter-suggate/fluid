/** The persistent fine brick store: h storage only where h bricks exist
 * (docs/plans/tiered-adaptivity-huge-grids-2026-10-24.md §2.1-2.2, Phase 2;
 * brick-arena-core-2026-10-20.md §2.3 and §3). A brick is 8³ h cells, 64 a
 * page (id 64·block + local); a live brick holds a slot of a fixed-capacity
 * pool for as long as it stays marked, and its channel values with it.
 *
 * The slots, the free stack, the enter and leave lists, the class lists and
 * the 27-neighbour records are the brick arena's level 0
 * (uniform-brick-arena.ts: ubaUpdate, ubaLists, ubaRecord, ubaRelease, the
 * fatal word): the store adds no table of its own, so the pressure arena's
 * slots and these are one design.
 *
 * Channels are of two kinds:
 * - SoA (the arena's pool): 512 values a slot at a 520-word stride, read by
 *   ubaLoad and staged by the arena's halo helper;
 * - sampled (u, phi): r32float 3D atlases, one a component and bank, of
 *   UFB_EDGE³ texels a slot: the slot's 8³ owned values at local 0..7 and a
 *   one-texel apron at -1 and 8 holding the neighbours' values (or the
 *   channel's seam value), so a hardware trilinear tap anywhere in the brick
 *   reads one slot's texels. A writer stores the core; its epilogue
 *   (apronWGSL) refreshes the aprons. One texture a component keeps every
 *   apron refresh a read_write r32float pass and every tap an r32float
 *   sample (2x the rgba32float filter rate).
 *
 * Layouts are the arena's, owned high: owned index i of a brick at origin o
 * (its first cell) is cell o+i, the +x/+y/+z face of cell o+i, or lattice
 * vertex o+i+1. A trilinear tap at position x (in h, lattice coordinates)
 * reads owned index x-o-δ, δ = (0.5,0.5,0.5) for a cell, (1,0.5,0.5) for
 * an x face (and its cyclic forms), (1,1,1) for a vertex: inside
 * [-1,8] for any x in the brick, which is what the apron holds.
 *
 * Migration: gather (paged -> slot) and scatter (slot -> paged) entries
 * (copyWGSL) keep the paged store valid while its readers move, a channel
 * at a time (write-through). The store is allocated upfront at its
 * capacity: fluid.tiers.bricks, or the t=0 census's bricks x 1.5 with a
 * floor of UNIFORM_FINE_BRICK_FLOOR (§2.7). A pop from an empty stack is the
 * arena's fatal word: no growth, no recovery. */
import { UniformBrickArena, type UniformArenaChannel, type UniformArenaLayout, type UniformArenaList } from "./uniform-brick-arena";

/** Texels of a slot's atlas box on an axis: 8 owned and a one-texel apron each side. */
export const UFB_EDGE=10;
/** Texels of a slot's apron (UFB_EDGE³ - 8³). */
export const UFB_SHELL=UFB_EDGE**3-512;
/** The capacity floor (§2.7) and the default growth factor over the t=0 census. */
export const UNIFORM_FINE_BRICK_FLOOR=256,UNIFORM_FINE_BRICK_GROWTH=1.5;
/** Default particles per fine cell (the pour runs at about 2). */
export const UNIFORM_FINE_PARTICLES_PER_CELL=4;

export interface UniformFineBrickChannel{
 name:string;
 layout:UniformArenaLayout;
 /** An atlas channel (hardware-filtered taps); otherwise an SoA pool channel. */
 sampled?:boolean;
 banks?:1|2;
}
export interface UniformFineBrickStoreOptions{
 /** Blocks of the lattice (UM_BLOCKS). */
 blocks:number;
 /** Slots (uniformFineBrickCapacity). */
 capacity:number;
 channels:readonly UniformFineBrickChannel[];
 /** Class lists of the live slots (the arena's). */
 lists:readonly UniformArenaList[];
 /** The 3D texture edge the atlas must fit (device.limits.maxTextureDimension3D). */
 maxTextureDimension3D?:number;
}
/** An atlas channel's WGSL binding in an apron or copy entry: var names a
 * texture_storage_3d<r32float,read_write> the module binds. */
export interface UniformFineBrickAtlasBinding{name:string;component?:number;bank?:0|1;binding:string}
/** An apron's seam: a WGSL fn(p:vec3i,word:u32)->f32 giving the value of
 * owned index p (absolute, in h) where the neighbour word is not a live
 * slot with that tile live (UB_NONE_ID, UBA_SEAM | block, or a slot whose
 * tile mask bit is clear). */
export interface UniformFineBrickApronChannel extends UniformFineBrickAtlasBinding{seam:string}
/** A channel copied between the store and the paged store: fn is a WGSL
 * fn(p:vec3i)->f32 (gather) or fn(p:vec3i,value:f32) (scatter) over owned
 * index p; binding names the atlas var of a sampled channel. */
export interface UniformFineBrickCopied{name:string;component?:number;bank?:0|1;fn:string;binding?:string}

/** The scene's tiers (SceneDescription.fluid.tiers). */
export interface UniformFineBrickTiers{readonly bricks?:number;readonly particlesPerFineCell?:number}
/** Slots of the store: the scene's fluid.tiers.bricks, or the t=0 census's
 * live bricks x 1.5 with a floor of 256, never more than the lattice's
 * 64·blocks bricks (which cannot overflow). */
export function uniformFineBrickCapacity(blocks:number,censusBricks:number,tiers:UniformFineBrickTiers={}):number{
 const bricks=64*blocks;
 if(tiers.bricks!==undefined){
  if(!Number.isSafeInteger(tiers.bricks)||tiers.bricks<1)throw new RangeError(`fluid.tiers.bricks ${tiers.bricks} is not a positive integer`);
  return Math.min(bricks,tiers.bricks);
 }
 if(!Number.isSafeInteger(censusBricks)||censusBricks<0)throw new RangeError(`Fine brick census ${censusBricks} is not a count`);
 return Math.min(bricks,Math.max(UNIFORM_FINE_BRICK_FLOOR,Math.ceil(UNIFORM_FINE_BRICK_GROWTH*censusBricks)));
}
/** Particle capacity of a brick store (§2.2): particles per fine cell x slots x 512. */
export function uniformFineBrickParticleCapacity(capacity:number,tiers:UniformFineBrickTiers={}):number{
 const per=tiers.particlesPerFineCell??UNIFORM_FINE_PARTICLES_PER_CELL;
 if(!(per>0))throw new RangeError(`fluid.tiers.particlesPerFineCell ${per} is not positive`);
 return Math.ceil(per*512*capacity);
}
/** Slots of the atlas grid an axis (x, y, z): the most compact box of at
 * least capacity slots whose texture fits maxDimension on every axis. */
export function uniformFineBrickAtlasGrid(capacity:number,maxDimension=2048):[number,number,number]{
 const most=Math.floor(maxDimension/UFB_EDGE);
 if(!(capacity>=1)||capacity>most**3)throw new RangeError(`A fine brick atlas of ${capacity} slots does not fit ${maxDimension}-texel 3D textures`);
 const x=Math.min(most,Math.ceil(Math.cbrt(capacity))),y=Math.min(most,Math.ceil(Math.sqrt(capacity/x))),z=Math.ceil(capacity/(x*y));
 return [x,y,z];
}
/** The atlas texel of owned index l (-1..8 an axis) of a slot, on the host (ufbAtlasTexel). */
export function uniformFineBrickAtlasTexel(grid:readonly number[],slot:number,l:readonly number[]):[number,number,number]{
 const gx=grid[0]!,gy=grid[1]!,s=[slot%gx,Math.floor(slot/gx)%gy,Math.floor(slot/(gx*gy))];
 return [UFB_EDGE*s[0]!+1+l[0]!,UFB_EDGE*s[1]!+1+l[1]!,UFB_EDGE*s[2]!+1+l[2]!];
}
/** The apron's k-th texel (0..487), as owned index -1..8 an axis: the z = -1
 * and z = 8 planes, then per z = 0..7 the y = -1 and y = 8 rows and the
 * x = -1 and x = 8 ends of every other row. The host twin of ufbShell. */
export function uniformFineBrickShell(k:number):[number,number,number]{
 if(k<200){const q=k%100;return [q%10-1,Math.floor(q/10)-1,k<100?-1:8];}
 const j=k-200,z=Math.floor(j/36),r=j%36;
 if(r<20)return [r%10-1,r<10?-1:8,z];
 const q=r-20;return [(q&1)?8:-1,q>>1,z];
}

interface Atlas{readonly name:string;readonly component:number;readonly bank:number;readonly label:string}
export class UniformFineBrickStore{
 /** The slot table and SoA pool. Its level 0 is the store; level 2 (the page) is the 4h block. */
 readonly arena:UniformBrickArena;
 readonly capacity:number;
 /** Slots of the atlas grid an axis and its texture size. */
 readonly atlasGrid:readonly [number,number,number];
 readonly atlasExtent:readonly [number,number,number];
 readonly atlases:readonly Atlas[];
 private readonly sampled=new Map<string,UniformFineBrickChannel>();
 constructor(readonly options:UniformFineBrickStoreOptions){
  this.capacity=options.capacity;
  const pool:UniformArenaChannel[]=[];
  const atlases:Atlas[]=[];
  for(const channel of options.channels){
   if(this.sampled.has(channel.name)||pool.some(c=>c.name===channel.name))throw new Error(`Fine brick store: channel ${channel.name} twice`);
   if(!channel.sampled){pool.push({name:channel.name,levels:[0],layout:channel.layout,banks:channel.banks});continue;}
   this.sampled.set(channel.name,channel);
   for(let bank=0;bank<(channel.banks??1);bank++)for(let component=0;component<(channel.layout==="face"?3:1);component++)
    atlases.push({name:channel.name,component,bank,label:`Uniform fine brick atlas ${channel.name}${channel.layout==="face"?` ${"xyz"[component]}`:""}${channel.banks===2?` bank ${bank}`:""}`});
  }
  this.arena=new UniformBrickArena({blocks:options.blocks,fineCapacity:options.capacity,channels:pool,lists:options.lists});
  this.atlases=atlases;
  this.atlasGrid=uniformFineBrickAtlasGrid(options.capacity,options.maxTextureDimension3D);
  this.atlasExtent=[UFB_EDGE*this.atlasGrid[0],UFB_EDGE*this.atlasGrid[1],UFB_EDGE*this.atlasGrid[2]];
 }
 /** Bytes the store allocates: the table, the SoA pool and every atlas. */
 get allocatedBytes():number{return 4*(this.arena.tableWords+this.arena.poolWords)+4*this.atlases.length*this.atlasExtent[0]*this.atlasExtent[1]*this.atlasExtent[2];}
 /** Bytes a slot costs in channels (pool and atlases), without the table. */
 get slotBytes():number{return 4*this.arena.poolWords/this.capacity+4*UFB_EDGE**3*this.atlases.length;}
 private atlas(name:string,component=0,bank=0):Atlas{
  const a=this.atlases.find(t=>t.name===name&&t.component===component&&t.bank===bank);
  if(!a)throw new Error(`Fine brick store: no atlas ${name} component ${component} bank ${bank}`);
  return a;
 }
 /** The atlases, allocated upfront (the only textures the store makes), keyed name/component/bank. */
 createAtlases(device:GPUDevice):Map<string,GPUTexture>{
  const out=new Map<string,GPUTexture>();
  for(const a of this.atlases)out.set(`${a.name}/${a.component}/${a.bank}`,device.createTexture({label:a.label,size:[...this.atlasExtent],dimension:"3d",format:"r32float",
   usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.STORAGE_BINDING|GPUTextureUsage.COPY_SRC|GPUTextureUsage.COPY_DST}));
  return out;
 }
 /** Atlas addressing: ufbAtlasTexel(slot, l) is the texel of owned index l
  * (-1..8 an axis) of a slot; ufbAtlasPoint(slot, l) the normalized
  * coordinate of a fractional owned index, for textureSampleLevel with a
  * linear clamp sampler; ufbShell(k) the apron's k-th owned index. */
 atlasWGSL():string{
  const [gx,gy]=this.atlasGrid,e=this.atlasExtent;
  return /* wgsl */`
const UFB_EDGE:u32=${UFB_EDGE}u;
const UFB_SHELL:u32=${UFB_SHELL}u;
const UFB_ATLAS_EXTENT:vec3f=vec3f(${e[0]}.0,${e[1]}.0,${e[2]}.0);
fn ufbAtlasCore(slot:u32)->vec3i{return i32(UFB_EDGE)*vec3i(vec3u(slot%${gx}u,(slot/${gx}u)%${gy}u,slot/${gx*gy}u))+vec3i(1);}
fn ufbAtlasTexel(slot:u32,l:vec3i)->vec3i{return ufbAtlasCore(slot)+l;}
fn ufbAtlasPoint(slot:u32,l:vec3f)->vec3f{return (vec3f(ufbAtlasCore(slot))+l+vec3f(0.5))/UFB_ATLAS_EXTENT;}
fn ufbShell(k:u32)->vec3i{
 if(k<200u){let q=k%100u;return vec3i(i32(q%10u)-1,i32(q/10u)-1,select(8,-1,k<100u));}
 let j=k-200u;let z=i32(j/36u);let r=j%36u;
 if(r<20u){return vec3i(i32(r%10u)-1,select(8,-1,r<10u),z);}
 let q=r-20u;return vec3i(select(-1,8,(q&1u)!=0u),i32(q>>1u),z);
}
`;}
 /** The slot list a churn or apron entry strides: "enter", "leave", "live" or a class list by name. */
 private list(list:string):[string,string]{
  if(list==="enter")return ["ubaCount(UBA_ENTER_COUNT)","ubaEnterSlot(e)"];
  if(list==="leave")return ["ubaCount(UBA_LEAVE_COUNT)","ubaLeaveSlot(e)"];
  if(list==="live")return ["ubaCount(UBA_LIVE_COUNT)","atomicLoad(&ufbTable[UBA_LIVE+e])"];
  const j=this.options.lists.findIndex(item=>item.name===list);
  if(j<0)throw new Error(`Fine brick store: no list ${list}`);
  return [`ubaListCount(${j}u)`,`ubaListSlot(${j}u,e)`];
 }
 /** The writer's epilogue: a workgroup a slot of a list, each channel's
  * apron from the neighbours' owned values (a live neighbour slot with the
  * tile live), else its seam fn. The table is the arena's tableWGSL over
  * var ufbTable. Run after every core store of the list's slots and their
  * neighbours (a pass boundary); its loads read only cores and its stores
  * write only aprons, so the atlas binds read_write once. */
 apronWGSL(entry:string,options:{list:string;channels:readonly UniformFineBrickApronChannel[];workgroup:number}):string{
  const {channels,workgroup}=options,[count,slotOf]=this.list(options.list);
  for(const ch of channels)this.atlas(ch.name,ch.component,ch.bank);
  const body=channels.map(ch=>`if(w<UBA_SEAM&&((m>>t)&1u)!=0u){textureStore(${ch.binding},ufbAtlasTexel(slot,l),vec4f(textureLoad(${ch.binding},ufbAtlasTexel(w,s)).x));}
   else{textureStore(${ch.binding},ufbAtlasTexel(slot,l),vec4f(${ch.seam}(origin+l,w)));}`).join("\n   ");
  return /* wgsl */`
var<workgroup> ufbR_${entry}:array<u32,27>;
var<workgroup> ufbM_${entry}:array<u32,27>;
var<workgroup> ufbN_${entry}:u32;
@compute @workgroup_size(${workgroup}) fn ${entry}(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 // The list's count, uniform across the workgroup (the loop holds barriers).
 if(lane==0u){ufbN_${entry}=${count};}
 let n=workgroupUniformLoad(&ufbN_${entry});
 for(var e=group.x;e<n;e+=groups.x){
  let slot=${slotOf};
  workgroupBarrier();
  if(lane<27u){let w=ubaRecordWord(slot,lane);ufbR_${entry}[lane]=w;var m=0u;if(w<UBA_SEAM){m=ubaRecordWord(w,28u);}ufbM_${entry}[lane]=m;}
  workgroupBarrier();
  let origin=vec3i(ubaOrigin(ubaBrick(slot)));
  for(var k=lane;k<UFB_SHELL;k+=${workgroup}u){
   let l=ufbShell(k);let d=vec3u(l+8)>>vec3u(3u);let nb=d.x+3u*d.y+9u*d.z;
   let w=ufbR_${entry}[nb];let m=ufbM_${entry}[nb];let s=l&vec3i(7);let t=(u32(s.x)>>2u)|((u32(s.y)>>2u)<<1u)|((u32(s.z)>>2u)<<2u);
   ${body}
  }
 }
}
`;}
 /** A churn skeleton entry: a workgroup a slot of a list, each channel's
  * 512 owned values through its fn. gather: fn(p) into the slot (an atlas
  * channel's core texels); scatter: the slot into fn(p,value). An atlas
  * gather leaves the aprons to the epilogue (apronWGSL). */
 copyWGSL(entry:string,options:{list:string;direction:"gather"|"scatter";channels:readonly UniformFineBrickCopied[];workgroup:number}):string{
  const {direction,channels,workgroup}=options,[count,slotOf]=this.list(options.list);
  const body=channels.map(ch=>{
   if(this.sampled.has(ch.name)){
    this.atlas(ch.name,ch.component,ch.bank);if(!ch.binding)throw new Error(`Fine brick store: atlas channel ${ch.name} copied without its binding`);
    return direction==="gather"?`textureStore(${ch.binding},ufbAtlasTexel(slot,l),vec4f(${ch.fn}(p)));`:`${ch.fn}(p,textureLoad(${ch.binding},ufbAtlasTexel(slot,l)).x);`;
   }
   const base=`UBA_${ch.name.toUpperCase()}_L0_${ch.component??0}_${ch.bank??0}`;this.arena.base(ch.name,0,ch.component,ch.bank);
   return direction==="gather"?`ubaStore(${base},slot,k,${ch.fn}(p));`:`${ch.fn}(p,ubaLoad(${base},slot,k));`;
  }).join("");
  return /* wgsl */`
@compute @workgroup_size(${workgroup}) fn ${entry}(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 let n=${count};
 for(var e=group.x;e<n;e+=groups.x){
  let slot=${slotOf};let origin=vec3i(ubaOrigin(ubaBrick(slot)));
  for(var k=lane;k<512u;k+=${workgroup}u){let l=vec3i(vec3u(k&7u,(k>>3u)&7u,k>>6u));let p=origin+l;${body}}
 }
}
`;}
 /** The table and pool WGSL a store module includes: the arena's table
  * over var ufbTable (array<atomic<u32>>, the table at word base), its
  * pool accessors over var ufbPool, and the atlas addressing. */
 moduleWGSL(base:number,tileMask="0xffu"):string{
  return this.arena.tableWGSL("ufbTable",`${base}u`,tileMask)+(this.arena.poolWords?this.arena.poolWGSL("ufbPool"):"")+this.atlasWGSL();
 }
}
