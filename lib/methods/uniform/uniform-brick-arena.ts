/** The brick arena core (docs/plans/brick-arena-core-2026-10-20.md §3): a
 * persistent brick table, SoA channels in one pool, one halo-staging helper
 * per kernel, the churn gather/scatter skeletons, and the MG levels.
 *
 * Levels: a level-ℓ brick is 8³ cells of width 2^ℓ h. Level 0 bricks are
 * the brick table's (uniform-brick-table.ts): id 64·block + local. Level 1
 * keys are 8·block + local (2³ a page), level 2 is the page (slot = block).
 *
 * Persistence (level 0): a brick keeps its slot, and so its channel values,
 * while it stays marked. Per step: clear the marks and counts (stepClear),
 * launch the markers, then ubaUpdate (a thread a brick id: dilate, take a
 * slot from the free stack for an entering brick, list a leaving one),
 * ubaLists (the live and class lists, the page list), ubaLevel1 (level 1
 * slots of the parents of one level 0 list's bricks: scratch, rebuilt every
 * step), ubaRecord (level 0, 1 and 2 records); the stages; the churn scatter
 * over the leave list; then ubaRelease (leaving slots back onto the free
 * stack). A pop from an empty stack stores UBA_SEAM | brick id in the fatal
 * word (fail-fast: no growth, no recovery); the host's frame diagnostics
 * throw on it.
 *
 * Records, 32 words a slot of every level: 0..26 neighbour words (offset
 * x+3y+9z of (-1..1)³), 27 parent (slot | octant<<29, UB_NONE at level 2 or
 * with no parent), 28 the mask of h-owned tiles (level 0) or live child
 * octants (level 1; 0xff at level 2), 29 the key. A neighbour word is a
 * live slot of the same level; UBA_SEAM | block where the page is stored but
 * the brick is not live (the stage's seam rule gives its values, from the
 * coarser level); UB_NONE outside stored pages. A dead tile or octant of a
 * live brick (mask bit clear) is a seam too.
 *
 * Channels: 512 values a brick (x fastest, owned high: a cell, its +x/+y/+z
 * faces and its +++ vertex) at a slot stride of 520 words, so the 27
 * neighbours of a brick do not alias the same cache sets (a 512 stride
 * stages 2.5× slower). One pool, array<u32>, f32 by bitcast. */
import { UNIFORM_BRICK_CLASS } from "./uniform-brick-table";

/** Class bits of the arena: the brick table's, the arena's own, and the
 * per-step churn flags (set by ubaUpdate, cleared by the next). */
export const UNIFORM_ARENA_CLASS={...UNIFORM_BRICK_CLASS,fine:128,surface:256,liquid:512,dirty:1024,entering:1<<30,leaving:1<<31} as const;
/** The table build, in order, after the markers. */
export const UNIFORM_ARENA_TABLE_ENTRIES=["ubaUpdate","ubaLists","ubaLevel1","ubaRecord"] as const;
/** After the step's churn scatter. */
export const UNIFORM_ARENA_RELEASE_ENTRY="ubaRelease";
export const UBA_STRIDE=520;
const UBA_RECORD=32,UBA_HEADER=32,UBA_BANKS=16,UBA_COUNTS=2+UBA_BANKS,UBA_LISTS=9;
export type UniformArenaLevel=0|1|2;
export type UniformArenaLayout="cell"|"face"|"vertex";
export interface UniformArenaChannel{name:string;levels:readonly UniformArenaLevel[];layout:UniformArenaLayout;banks?:1|2}
export interface UniformArenaList{name:string;any:number}
export interface UniformArenaOptions{
 /** Blocks of the lattice (UM_BLOCKS). */
 blocks:number;
 /** Level 0 slots (and level 1 slots); default every brick of the lattice (cannot overflow). */
 fineCapacity?:number;
 channels:readonly UniformArenaChannel[];
 /** Compact level 0 slot lists by class: a live slot joins each list whose bits it has any of. */
 lists:readonly UniformArenaList[];
 /** The level 0 list whose bricks' parents are level 1's bricks (none: no level 1). */
 level1From?:string;
}
/** A channel staged by a kernel, with its own halo (default the stage's);
 * seam names a WGSL fn(p:vec3i,word:u32)->f32 giving the value of owned
 * index p (absolute, in the level's cells; a vertex index p is lattice
 * vertex p+1) where the source is not a live brick's live tile: word is the
 * neighbour word (UB_NONE, UBA_SEAM|block, or a live slot whose tile is
 * dead). */
export interface UniformArenaStaged{name:string;component?:number;bank?:0|1;halo?:number;seam:string}
/** A channel copied by a churn skeleton; fn names a WGSL fn(p:vec3i)->f32
 * (gather) or fn(p:vec3i,value:f32) (scatter) over owned index p. */
export interface UniformArenaCopied{name:string;component?:number;bank?:0|1;fn:string}

export class UniformBrickArena{
 readonly bricks:number;
 readonly fineCapacity:number;
 readonly tableWords:number;
 readonly poolWords:number;
 private readonly bases=new Map<string,number>();
 private readonly bankWords=new Map<string,number>();
 private readonly lists:readonly UniformArenaList[];
 private readonly level1:number;
 constructor(private readonly options:UniformArenaOptions){
  this.bricks=64*options.blocks;
  this.fineCapacity=options.fineCapacity??this.bricks;
  if(this.fineCapacity<1||this.fineCapacity>this.bricks){throw new Error(`Brick arena capacity ${this.fineCapacity} outside 1..${this.bricks}`);}
  if(options.lists.length>UBA_LISTS){throw new Error(`Brick arena: ${options.lists.length} lists, at most ${UBA_LISTS}`);}
  this.lists=options.lists;
  this.level1=options.level1From===undefined?-1:options.lists.findIndex(list=>list.name===options.level1From);
  if(options.level1From!==undefined&&this.level1<0){throw new Error(`Brick arena: no list ${options.level1From} for level 1`);}
  let words=0,banks=0;
  for(const channel of options.channels){
   const components=channel.layout==="face"?3:1,count=channel.banks??1;
   if(count===2){if(banks===UBA_BANKS){throw new Error(`Brick arena: more than ${UBA_BANKS} two-bank channels`);}this.bankWords.set(channel.name,banks++);}
   for(const level of channel.levels){
    if(level===1&&this.level1<0){throw new Error(`Brick arena: channel ${channel.name} at level 1 without level1From`);}
    for(let bank=0;bank<count;bank++)for(let component=0;component<components;component++){
     this.bases.set(`${channel.name}/${level}/${component}/${bank}`,words);words+=UBA_STRIDE*this.capacity(level);
    }
   }
  }
  this.poolWords=words;
  const l=this.layout;this.tableWords=l.pages+options.blocks;
 }
 /** Slots of a level. */
 capacity(level:UniformArenaLevel):number{return level===2?this.options.blocks:this.fineCapacity;}
 /** The pool word of a channel's first slot. */
 base(name:string,level:UniformArenaLevel,component=0,bank=0):number{
  const base=this.bases.get(`${name}/${level}/${component}/${bank}`);
  if(base===undefined){throw new Error(`Brick arena: no channel ${name} at level ${level} component ${component} bank ${bank}`);}
  return base;
 }
 /** Table word offsets, from the table's base word. The step clear covers
  * the counts, the marks and the level 1 table, contiguous. */
 private get layout(){
  const cap=this.fineCapacity,blocks=this.options.blocks;
  const mark=UBA_HEADER,table1=mark+this.bricks,table=table1+8*blocks,brick=table+this.bricks,klass=brick+cap,record=klass+cap,free=record+UBA_RECORD*cap;
  const enter=free+cap,leave=enter+cap,live=leave+cap,lists=live+cap,brick1=lists+this.lists.length*cap,record1=brick1+cap,record2=record1+UBA_RECORD*cap,pages=record2+UBA_RECORD*blocks;
  return {mark,table1,table,brick,klass,record,free,enter,leave,live,lists,brick1,record1,record2,pages};
 }
 /** Words written once at attach (base: the table's first word): an empty
  * table and a full free stack. The only host writes the table takes. */
 initialWrites(base:number):{offset:number;data:Uint32Array}[]{
  const l=this.layout,cap=this.fineCapacity;
  const header=new Uint32Array(UBA_HEADER);header[1]=cap;
  const none=new Uint32Array(this.bricks+cap).fill(0xffffffff);
  const free=new Uint32Array(cap);for(let i=0;i<cap;i++){free[i]=cap-1-i;}
  return [{offset:4*base,data:header},{offset:4*(base+l.table),data:none},{offset:4*(base+l.free),data:free}];
 }
 /** The byte range each step clears before its markers: the counts, the marks and the level 1 table. */
 stepClear(base:number):{offset:number;size:number}{return {offset:4*(base+UBA_COUNTS),size:4*(this.layout.table-UBA_COUNTS)};}
 /** stepClear in two parts, for a table whose live list is read between
  * the markers and the build: the marks and the level 1 table (before the
  * markers), and the counts (just before the build). */
 markClear(base:number):{offset:number;size:number}{return {offset:4*(base+UBA_HEADER),size:4*(this.layout.table-UBA_HEADER)};}
 countClear(base:number):{offset:number;size:number}{return {offset:4*(base+UBA_COUNTS),size:4*(UBA_HEADER-UBA_COUNTS)};}
 /** The byte offset of the fatal word (0, or UBA_SEAM | the brick id refused a slot). */
 fatalOffset(base:number):number{return 4*base;}

 /** The table: constants, accessors and the entries. table: an
  * array<atomic<u32>> the module binds; base: a WGSL u32 expression for the
  * table's first word. Needs the lattice address; tileMask: a WGSL u32
  * expression of id, the level 0 brick's h-owned tiles. */
 tableWGSL(table:string,base:string,tileMask="0xffu"):string{
  const l=this.layout,cap=this.fineCapacity,c=UNIFORM_ARENA_CLASS,blocks=this.options.blocks;
  const count=(k:number)=>`${base}+${UBA_COUNTS+k}u`;
  const word=(name:keyof typeof l)=>`${base}+${l[name]}u`;
  return /* wgsl */`
const UBA_BRICKS:u32=${this.bricks}u;
const UBA_CAPACITY:u32=${cap}u;
const UBA_PAGES_CAPACITY:u32=${blocks}u;
const UBA_SEAM:u32=0x80000000u;
const UB_NONE_ID:u32=0xffffffffu;
const UBA_FATAL:u32=${base};
const UBA_FREE_TOP:u32=${base}+1u;
const UBA_BANK:u32=${base}+2u;
const UBA_ENTER_COUNT:u32=${count(0)};
const UBA_LEAVE_COUNT:u32=${count(1)};
const UBA_LIVE_COUNT:u32=${count(2)};
const UBA_LEVEL1_COUNT:u32=${count(3)};
const UBA_PAGE_COUNT:u32=${count(4)};
const UBA_LIST_COUNT:u32=${count(5)};
const UBA_MARK:u32=${word("mark")};
const UBA_TABLE1:u32=${word("table1")};
const UBA_TABLE:u32=${word("table")};
const UBA_BRICK:u32=${word("brick")};
const UBA_CLASS:u32=${word("klass")};
const UBA_RECORD:u32=${word("record")};
const UBA_FREE:u32=${word("free")};
const UBA_ENTER:u32=${word("enter")};
const UBA_LEAVE:u32=${word("leave")};
const UBA_LIVE:u32=${word("live")};
const UBA_LISTS:u32=${word("lists")};
const UBA_BRICK1:u32=${word("brick1")};
const UBA_RECORD1:u32=${word("record1")};
const UBA_RECORD2:u32=${word("record2")};
const UBA_PAGES:u32=${word("pages")};
const UBA_ENTERING:u32=${c.entering>>>0}u;
const UBA_LEAVING:u32=${c.leaving>>>0}u;
const UBA_DIRTY:u32=${c.dirty}u;
const UBA_CORES:u32=${c.arenaCore|c.particlesCore}u;
${this.lists.map((list,j)=>`const UBA_LIST_${list.name.replace(/[A-Z]/g,ch=>`_${ch}`).toUpperCase()}:u32=${j}u;`).join("")}
// Level 0 bricks: the brick of cell c (UB_NONE_ID where its page holds nothing), a brick's first cell.
fn ubaBrickAt(c:vec3i)->u32{let b=umPageWord(c>>vec3u(5u));if(!umBlockStored(b)){return UB_NONE_ID;}let l=(vec3u(c)>>vec3u(3u))&vec3u(3u);return 64u*b+(l.x|(l.y<<2u)|(l.z<<4u));}
fn ubaOrigin(id:u32)->vec3u{let l=id&63u;return (umBlockPage(id>>6u)<<vec3u(5u))+8u*vec3u(l&3u,(l>>2u)&3u,l>>4u);}
// A level's brick key (record word 29) to its first cell, in the level's cells.
fn ubaLevelOrigin(level:u32,key:u32)->vec3u{
 if(level==0u){return ubaOrigin(key);}
 if(level==1u){let l=key&7u;return (umBlockPage(key>>3u)<<vec3u(4u))+8u*vec3u(l&1u,(l>>1u)&1u,l>>2u);}
 return umBlockPage(key)<<vec3u(3u);
}
fn ubaMark(id:u32,bits:u32){if((atomicLoad(&${table}[UBA_MARK+id])&bits)!=bits){atomicOr(&${table}[UBA_MARK+id],bits);}}
fn ubaSlot(id:u32)->u32{return atomicLoad(&${table}[UBA_TABLE+id]);}
fn ubaBrick(slot:u32)->u32{return atomicLoad(&${table}[UBA_BRICK+slot]);}
fn ubaClass(slot:u32)->u32{return atomicLoad(&${table}[UBA_CLASS+slot]);}
fn ubaLevelRecord(level:u32,slot:u32,k:u32)->u32{
 let at=select(select(UBA_RECORD2,UBA_RECORD1,level==1u),UBA_RECORD,level==0u);
 return atomicLoad(&${table}[at+${UBA_RECORD}u*slot+k]);
}
fn ubaRecordWord(slot:u32,k:u32)->u32{return ubaLevelRecord(0u,slot,k);}
fn ubaCount(word:u32)->u32{return min(atomicLoad(&${table}[word]),UBA_CAPACITY);}
fn ubaListCount(j:u32)->u32{return ubaCount(UBA_LIST_COUNT+j);}
fn ubaListSlot(j:u32,e:u32)->u32{return atomicLoad(&${table}[UBA_LISTS+UBA_CAPACITY*j+e]);}
fn ubaEnterSlot(e:u32)->u32{return atomicLoad(&${table}[UBA_ENTER+e]);}
fn ubaLeaveSlot(e:u32)->u32{return atomicLoad(&${table}[UBA_LEAVE+e]);}
// Level 1 slots are 0 .. count; level 2's are the page list's blocks.
fn ubaLevelCount(level:u32)->u32{if(level==1u){return ubaCount(UBA_LEVEL1_COUNT);}return min(atomicLoad(&${table}[UBA_PAGE_COUNT]),UBA_PAGES_CAPACITY);}
fn ubaLevelSlot(level:u32,e:u32)->u32{if(level==1u){return e;}return atomicLoad(&${table}[UBA_PAGES+e]);}
// MG: the parent (slot, octant), and the child of an octant (UB_NONE_ID: none).
fn ubaParent(level:u32,slot:u32)->vec2u{let w=ubaLevelRecord(level,slot,27u);if(w==UB_NONE_ID){return vec2u(UB_NONE_ID,0u);}return vec2u(w&0x1fffffffu,w>>29u);}
fn ubaChild(level:u32,slot:u32,octant:u32)->u32{
 if(level==2u){let s=atomicLoad(&${table}[UBA_TABLE1+8u*slot+octant]);return select(s-1u,UB_NONE_ID,s==0u);}
 let key=ubaLevelRecord(1u,slot,29u);let l=2u*vec3u(key&1u,(key>>1u)&1u,(key>>2u)&1u)+vec3u(octant&1u,(octant>>1u)&1u,octant>>2u);
 return atomicLoad(&${table}[UBA_TABLE+64u*(key>>3u)+(l.x|(l.y<<2u)|(l.z<<4u))]);
}
fn ubaParentKey(id:u32)->u32{let l=id&63u;return 8u*(id>>6u)+(((l>>1u)&1u)|(((l>>3u)&1u)<<1u)|(((l>>5u)&1u)<<2u));}
fn ubaOctant(id:u32)->u32{let l=id&63u;return (l&1u)|(((l>>2u)&1u)<<1u)|(((l>>4u)&1u)<<2u);}
// The parent cell of a child brick's cell c.
fn ubaOctantCell(octant:u32,c:vec3u)->vec3u{return 4u*vec3u(octant&1u,(octant>>1u)&1u,octant>>2u)+(c>>vec3u(1u));}
// A two-bank channel's current bank; a stage flips it from one lane of a later pass.
fn ubaBank(j:u32)->u32{return atomicLoad(&${table}[UBA_BANK+j])&1u;}
fn ubaFlip(j:u32){atomicXor(&${table}[UBA_BANK+j],1u);}
fn ubaAppend(count:u32,list:u32,slot:u32){let e=atomicAdd(&${table}[count],1u);atomicStore(&${table}[list+e],slot);}
fn ubaPageLive(block:u32)->bool{return umBlockStored(block)&&umBlockWord(block)!=UM_FREE;}
// A thread a brick id: dilate the core marks; an entering brick takes a
// slot, a leaving one is listed (its slot stays until ubaRelease), a dirty
// live one is listed for the gather again.
@compute @workgroup_size(64) fn ubaUpdate(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) groups:vec3u){
 for(var id=gid.x;id<UBA_BRICKS;id+=64u*groups.x){
  var bits=0u;
  if(ubaPageLive(id>>6u)){
   bits=atomicLoad(&${table}[UBA_MARK+id]);let origin=vec3i(ubaOrigin(id));
   for(var n=0u;n<27u;n++){
    let other=ubaBrickAt(origin+8*(vec3i(umCorner(n,3u))-1));
    if(other!=UB_NONE_ID){bits|=(atomicLoad(&${table}[UBA_MARK+other])&UBA_CORES)<<1u;}
   }
  }
  let old=atomicLoad(&${table}[UBA_TABLE+id]);
  if(bits==0u){if(old!=UB_NONE_ID){atomicStore(&${table}[UBA_CLASS+old],UBA_LEAVING);ubaAppend(UBA_LEAVE_COUNT,UBA_LEAVE,old);}continue;}
  if(old!=UB_NONE_ID){atomicStore(&${table}[UBA_CLASS+old],bits);if((bits&UBA_DIRTY)!=0u){ubaAppend(UBA_ENTER_COUNT,UBA_ENTER,old);}continue;}
  let top=atomicSub(&${table}[UBA_FREE_TOP],1u);
  if(top-1u>=UBA_CAPACITY){atomicAdd(&${table}[UBA_FREE_TOP],1u);atomicCompareExchangeWeak(&${table}[UBA_FATAL],0u,UBA_SEAM|id);continue;}
  let slot=atomicLoad(&${table}[UBA_FREE+top-1u]);
  atomicStore(&${table}[UBA_TABLE+id],slot);atomicStore(&${table}[UBA_BRICK+slot],id);atomicStore(&${table}[UBA_CLASS+slot],bits|UBA_ENTERING);
  ubaAppend(UBA_ENTER_COUNT,UBA_ENTER,slot);
 }
}
// A thread a slot: the live list and the class lists; a thread a block: the page list.
@compute @workgroup_size(64) fn ubaLists(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) groups:vec3u){
 for(var slot=gid.x;slot<max(UBA_CAPACITY,UBA_PAGES_CAPACITY);slot+=64u*groups.x){
  if(slot<UBA_PAGES_CAPACITY&&ubaPageLive(slot)){ubaAppend(UBA_PAGE_COUNT,UBA_PAGES,slot);}
  if(slot>=UBA_CAPACITY){continue;}
  let bits=atomicLoad(&${table}[UBA_CLASS+slot]);if(bits==0u||(bits&UBA_LEAVING)!=0u){continue;}
  ubaAppend(UBA_LIVE_COUNT,UBA_LIVE,slot);
  ${this.lists.map((list,j)=>`if((bits&${list.any}u)!=0u){ubaAppend(UBA_LIST_COUNT+${j}u,UBA_LISTS+${j*cap}u,slot);}`).join("\n  ")}
 }
}
// A thread a brick of the level 1 list: its parent takes a level 1 slot
// (table1: slot+1 by level 1 key, cleared every step).
@compute @workgroup_size(64) fn ubaLevel1(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) groups:vec3u){
 ${this.level1<0?"":`let n=ubaListCount(${this.level1}u);
 for(var e=gid.x;e<n;e+=64u*groups.x){
  let key=ubaParentKey(ubaBrick(ubaListSlot(${this.level1}u,e)));
  if(atomicOr(&${table}[UBA_TABLE1+key],UBA_SEAM)!=0u){continue;}
  let slot=atomicAdd(&${table}[UBA_LEVEL1_COUNT],1u);
  atomicStore(&${table}[UBA_TABLE1+key],slot+1u);atomicStore(&${table}[UBA_BRICK1+slot],key);
 }`}
}
// A level 1 key's slot (UB_NONE_ID: none).
fn ubaSlot1(key:u32)->u32{let s=atomicLoad(&${table}[UBA_TABLE1+key]);return select(s-1u,UB_NONE_ID,s==0u);}
// A thread a (slot, record word) of level 0, then level 1, then level 2.
@compute @workgroup_size(64) fn ubaRecord(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) groups:vec3u){
 let n0=${UBA_RECORD}u*ubaCount(UBA_LIVE_COUNT);let n1=n0+${UBA_RECORD}u*ubaLevelCount(1u);let n2=n1+${UBA_RECORD}u*ubaLevelCount(2u);
 for(var i=gid.x;i<n2;i+=64u*groups.x){
  var word=0u;
  if(i<n0){
   let slot=atomicLoad(&${table}[UBA_LIVE+i/${UBA_RECORD}u]);let k=i%${UBA_RECORD}u;let id=atomicLoad(&${table}[UBA_BRICK+slot]);
   if(k<27u){
    let other=ubaBrickAt(vec3i(ubaOrigin(id))+8*(vec3i(umCorner(k,3u))-1));
    word=UB_NONE_ID;
    if(other!=UB_NONE_ID){
     let s=atomicLoad(&${table}[UBA_TABLE+other]);
     word=select(UBA_SEAM|(other>>6u),s,s!=UB_NONE_ID&&(atomicLoad(&${table}[UBA_CLASS+s])&UBA_LEAVING)==0u);
    }
   }else if(k==27u){let p=ubaSlot1(ubaParentKey(id));word=select(p|(ubaOctant(id)<<29u),UB_NONE_ID,p==UB_NONE_ID);}
   else if(k==28u){word=${tileMask};}else if(k==29u){word=id;}
   atomicStore(&${table}[UBA_RECORD+${UBA_RECORD}u*slot+k],word);
  }else if(i<n1){
   let j=i-n0;let slot=j/${UBA_RECORD}u;let k=j%${UBA_RECORD}u;let key=atomicLoad(&${table}[UBA_BRICK1+slot]);let block=key>>3u;
   if(k<27u){
    let c=vec3i(ubaLevelOrigin(1u,key))+8*(vec3i(umCorner(k,3u))-1);let b=umPageWord(c>>vec3u(4u));
    word=UB_NONE_ID;
    if(umBlockStored(b)){let l=(vec3u(c)>>vec3u(3u))&vec3u(1u);let s=ubaSlot1(8u*b+(l.x|(l.y<<1u)|(l.z<<2u)));word=select(s,UBA_SEAM|b,s==UB_NONE_ID);}
   }else if(k==27u){word=block|((key&7u)<<29u);}
   else if(k==28u){
    let l=key&7u;let first=64u*block+2u*((l&1u)|(((l>>1u)&1u)<<2u)|((l>>2u)<<4u));
    for(var o=0u;o<8u;o++){
     let s=atomicLoad(&${table}[UBA_TABLE+first+(o&1u)+(((o>>1u)&1u)<<2u)+((o>>2u)<<4u)]);
     ${this.level1<0?"":`if(s!=UB_NONE_ID&&(atomicLoad(&${table}[UBA_CLASS+s])&${this.lists[this.level1]!.any}u)!=0u&&(atomicLoad(&${table}[UBA_CLASS+s])&UBA_LEAVING)==0u){word|=1u<<o;}`}
    }
   }else if(k==29u){word=key;}
   atomicStore(&${table}[UBA_RECORD1+j],word);
  }else{
   let j=i-n1;let block=atomicLoad(&${table}[UBA_PAGES+j/${UBA_RECORD}u]);let k=j%${UBA_RECORD}u;
   if(k<27u){let b=umPageWord(vec3i(umBlockPage(block))+vec3i(umCorner(k,3u))-1);word=select(UB_NONE_ID,b,umBlockStored(b));}
   else if(k==27u){word=UB_NONE_ID;}else if(k==28u){word=0xffu;}else if(k==29u){word=block;}
   atomicStore(&${table}[UBA_RECORD2+${UBA_RECORD}u*block+k],word);
  }
 }
}
// A thread a leave entry, after the churn scatter: the slot back on the stack.
@compute @workgroup_size(64) fn ubaRelease(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) groups:vec3u){
 let n=ubaCount(UBA_LEAVE_COUNT);
 for(var e=gid.x;e<n;e+=64u*groups.x){
  let slot=atomicLoad(&${table}[UBA_LEAVE+e]);let id=atomicLoad(&${table}[UBA_BRICK+slot]);
  atomicStore(&${table}[UBA_TABLE+id],UB_NONE_ID);atomicStore(&${table}[UBA_BRICK+slot],UB_NONE_ID);atomicStore(&${table}[UBA_CLASS+slot],0u);
  let top=atomicAdd(&${table}[UBA_FREE_TOP],1u);atomicStore(&${table}[UBA_FREE+top],slot);
 }
}
`;}

 /** A pass that only reads the table, or marks it, from another module:
  * plain loads of a buffer the module binds as array<u32> (read or
  * read_write), the table at word base. ubrSlotOf: a brick id's live slot
  * (UB_NONE_ID: none); ubrRecord: record word k of a level 0 slot; the
  * live list. The table is stable between one build and the next. */
 readerWGSL(buffer:string,base:number):string{
  const l=this.layout,count=(k:number)=>base+UBA_COUNTS+k;
  return /* wgsl */`
const UBR_NONE:u32=0xffffffffu;
const UBR_SEAM:u32=0x80000000u;
const UBR_CAPACITY:u32=${this.fineCapacity}u;
fn ubrBrickAt(c:vec3i)->u32{let b=umPageWord(c>>vec3u(5u));if(!umBlockStored(b)){return UBR_NONE;}let l=(vec3u(c)>>vec3u(3u))&vec3u(3u);return 64u*b+(l.x|(l.y<<2u)|(l.z<<4u));}
fn ubrOrigin(id:u32)->vec3u{let l=id&63u;return (umBlockPage(id>>6u)<<vec3u(5u))+8u*vec3u(l&3u,(l>>2u)&3u,l>>4u);}
fn ubrSlotOf(id:u32)->u32{if(id==UBR_NONE){return UBR_NONE;}return ${buffer}[${base+l.table}u+id];}
fn ubrBrick(slot:u32)->u32{return ${buffer}[${base+l.brick}u+slot];}
fn ubrClass(slot:u32)->u32{return ${buffer}[${base+l.klass}u+slot];}
fn ubrRecord(slot:u32,k:u32)->u32{return ${buffer}[${base+l.record}u+${UBA_RECORD}u*slot+k];}
fn ubrLiveCount()->u32{return min(${buffer}[${count(2)}u],UBR_CAPACITY);}
fn ubrLiveSlot(e:u32)->u32{return ${buffer}[${base+l.live}u+e];}
`;}
 /** A marker in another module: the table's mark words in a buffer the
  * module binds as array<atomic<u32>> read_write, the table at word base.
  * Marks land between the step clear and the table build. */
 markWGSL(buffer:string,base:number):string{
  const c=UNIFORM_ARENA_CLASS;
  return /* wgsl */`
${Object.entries(c).filter(([name])=>name!=="entering"&&name!=="leaving").map(([name,bit])=>`const UBA_${name.replace(/[A-Z]/g,ch=>`_${ch}`).toUpperCase()}:u32=${bit}u;`).join("")}
fn ubaMark(id:u32,bits:u32){let at=${base+this.layout.mark}u+id;if((atomicLoad(&${buffer}[at])&bits)!=bits){atomicOr(&${buffer}[at],bits);}}
`;}
 /** Channel bases (from pool word base) and the pool accessors. pool: an array<u32> the module binds. */
 poolWGSL(pool:string,base=0):string{
  const constants=[...this.bases].map(([key,word])=>{const [name,level,component,bank]=key.split("/");return `const UBA_${name!.toUpperCase()}_L${level}_${component}_${bank}:u32=${base+word}u;`;}).join("");
  const banks=[...this.bankWords].map(([name,j])=>`const UBA_BANK_${name.toUpperCase()}:u32=${j}u;`).join("");
  return /* wgsl */`
const UBA_STRIDE:u32=${UBA_STRIDE}u;
${constants}${banks}
fn ubaLoad(base:u32,slot:u32,i:u32)->f32{return bitcast<f32>(${pool}[base+UBA_STRIDE*slot+i]);}
fn ubaStore(base:u32,slot:u32,i:u32,value:f32){${pool}[base+UBA_STRIDE*slot+i]=bitcast<u32>(value);}
`;}
 /** A channel's base expression: a two-bank channel's bank picks by the table's bank word (bank 0 current, 1 the other). */
 private baseExpression(name:string,level:UniformArenaLevel,component=0,bank:0|1=0):string{
  const j=this.bankWords.get(name),key=name.toUpperCase();this.base(name,level,component,0);
  if(j===undefined){if(bank!==0){throw new Error(`Brick arena: ${name} has one bank`);}return `UBA_${key}_L${level}_${component}_0`;}
  return `select(UBA_${key}_L${level}_${component}_0,UBA_${key}_L${level}_${component}_1,(ubaBank(${j}u)^${bank}u)!=0u)`;
 }

 /** The halo-staging helper of one kernel (one call site). ubaStage_K(slot,
  * lane, octant) fills ubaW_K with each staged channel's box of owned
  * indices -halo .. n+halo around the brick (region "brick", n 8; octant
  * unused) or around one octant of it (region "octant", n 4: cells
  * 4·octant ..); ubaAt_K(c,j) indexes channel j's box at c, relative to the
  * region's first cell. Seam values come from each channel's seam fn and are
  * never stored. */
 stageWGSL(kernel:string,options:{level?:UniformArenaLevel;region?:"brick"|"octant";channels:readonly UniformArenaStaged[];halo?:number;workgroup:number}):string{
  const {channels,workgroup}=options,level=options.level??0,n=options.region==="octant"?4:8,k=kernel;
  const halos=channels.map(ch=>ch.halo??options.halo??1);
  for(const halo of halos){if(!Number.isInteger(halo)||halo<0||halo>n/2){throw new Error(`Brick arena staging halo ${halo} outside 0..${n/2}`);}}
  const offsets:number[]=[];let words=0;
  for(const halo of halos){offsets.push(words);words+=(n+2*halo)**3;}
  const loops=channels.map((ch,j)=>{
   const halo=halos[j]!,e=n+2*halo,e3=e**3,at=offsets[j]!;
   return `
 for(var k=lane;k<${e3}u;k+=${workgroup}u){
  let c=vec3i(vec3u(k%${e}u,(k/${e}u)%${e}u,k/${e*e}u))-${halo}+shift;let d=vec3u(c+8)>>vec3u(3u);let n=d.x+3u*d.y+9u*d.z;
  let w=ubaR_${k}[n];let l=vec3u(c&vec3i(7));let t=(l.x>>2u)|((l.y>>2u)<<1u)|((l.z>>2u)<<2u);
  if(w<UBA_SEAM&&((ubaM_${k}[n]>>t)&1u)!=0u){ubaW_${k}[${at}u+k]=ubaLoad(${this.baseExpression(ch.name,level,ch.component,ch.bank)},w,l.x+8u*(l.y+8u*l.z));}
  else{ubaW_${k}[${at}u+k]=${ch.seam}(origin+c,w);}
 }`;}).join("");
  const at=channels.map((_,j)=>`if(j==${j}u){let q=vec3u(c+${halos[j]});return ${offsets[j]}u+q.x+${n+2*halos[j]!}u*(q.y+${n+2*halos[j]!}u*q.z);}`).join("");
  return /* wgsl */`
var<workgroup> ubaW_${k}:array<f32,${words}>;
var<workgroup> ubaR_${k}:array<u32,28>;
var<workgroup> ubaM_${k}:array<u32,27>;
fn ubaAt_${k}(c:vec3i,j:u32)->u32{${at}return 0u;}
fn ubaStage_${k}(slot:u32,lane:u32,octant:u32){
 workgroupBarrier();
 if(lane<27u){let w=ubaLevelRecord(${level}u,slot,lane);ubaR_${k}[lane]=w;var m=0u;if(w<UBA_SEAM){m=ubaLevelRecord(${level}u,w,28u);}ubaM_${k}[lane]=m;}
 else if(lane==27u){ubaR_${k}[27]=ubaLevelRecord(${level}u,slot,29u);}
 workgroupBarrier();
 let shift=${n===4?"4*vec3i(vec3u(octant&1u,(octant>>1u)&1u,octant>>2u))":"vec3i(0)"};
 let origin=vec3i(ubaLevelOrigin(${level}u,ubaR_${k}[27]));${loops}
 workgroupBarrier();
}
`;}

 /** A churn skeleton entry (level 0): a workgroup a slot of a list (the
  * enter list, the leave list, or a class list by name), each copied
  * channel's 512 values through its fn. gather: fn(p) into the slot;
  * scatter: the slot into fn(p,value). */
 copyWGSL(entry:string,options:{list:string;direction:"gather"|"scatter";channels:readonly UniformArenaCopied[];workgroup:number}):string{
  const {list,direction,channels,workgroup}=options;
  const j=this.lists.findIndex(item=>item.name===list);
  const [count,slotOf]=list==="enter"?["ubaCount(UBA_ENTER_COUNT)","ubaEnterSlot(e)"]:list==="leave"?["ubaCount(UBA_LEAVE_COUNT)","ubaLeaveSlot(e)"]
   :j>=0?[`ubaListCount(${j}u)`,`ubaListSlot(${j}u,e)`]:(()=>{throw new Error(`Brick arena: no list ${list}`);})();
  const body=channels.map(ch=>{const base=this.baseExpression(ch.name,0,ch.component,ch.bank);
   return direction==="gather"?`ubaStore(${base},slot,k,${ch.fn}(p));`:`${ch.fn}(p,ubaLoad(${base},slot,k));`;}).join("");
  return /* wgsl */`
@compute @workgroup_size(${workgroup}) fn ${entry}(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 let n=${count};
 for(var e=group.x;e<n;e+=groups.x){
  let slot=${slotOf};let origin=vec3i(ubaOrigin(ubaBrick(slot)));
  for(var k=lane;k<512u;k+=${workgroup}u){let p=origin+vec3i(vec3u(k&7u,(k>>3u)&7u,k>>6u));${body}}
 }
}
`;}
}
