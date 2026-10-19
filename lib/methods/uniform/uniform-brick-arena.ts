/** The brick arena core (docs/plans/brick-arena-core-2026-10-20.md §3): a
 * persistent brick table, SoA channels in one pool, one halo-staging helper
 * per kernel, and the churn gather/scatter skeletons. Bricks and keys are
 * the brick table's (uniform-brick-table.ts): 8³ cells of a page, id
 * 64·block + local. Level 2 (4h, a brick a page) has identity slots
 * (slot = block) and no table.
 *
 * Persistence: a brick keeps its slot, and so its channel values, while it
 * stays marked. Per step: clear the marks and counts (stepClear), launch the
 * markers, then ubaUpdate (a thread a brick id: dilate, take a slot from the
 * free stack for an entering brick, list a leaving one), ubaLists (the live
 * and class lists), ubaRecord (27 neighbour words, tile mask, brick id of
 * every live slot); the stages; the churn scatter over the leave list; then
 * ubaRelease (leaving slots back onto the free stack). A pop from an empty
 * stack stores the brick id in the fatal word (fail-fast: no growth, no
 * recovery); the host's frame diagnostics throw on it.
 *
 * Neighbour words: a live slot of the same level; UBA_SEAM | block where the
 * neighbour's page is stored but the brick is not live (values come from the
 * coarser level: the stage's seam rule); UB_NONE outside stored pages.
 *
 * Channels: 512 values a brick (x fastest, owned high: a cell, its +x/+y/+z
 * faces and its +++ vertex) at a slot stride of 520 words, so the 27
 * neighbours of a brick do not alias the same cache sets (a 512 stride
 * stages 2.5× slower). One pool, array<u32>, f32 by bitcast. */
import { UNIFORM_BRICK_CLASS } from "./uniform-brick-table";

/** Class bits of the arena: the brick table's, the arena's own, and the
 * per-step churn flags (set by ubaUpdate, cleared by the next). */
export const UNIFORM_ARENA_CLASS={...UNIFORM_BRICK_CLASS,fine:128,surface:256,liquid:512,dirty:1024,entering:1<<30,leaving:1<<31} as const;
export const UNIFORM_ARENA_TABLE_ENTRIES=["ubaUpdate","ubaLists","ubaRecord"] as const;
export const UNIFORM_ARENA_RELEASE_ENTRY="ubaRelease";
export const UBA_STRIDE=520;
const UBA_RECORD=32,UBA_HEADER=32,UBA_BANKS=16,UBA_LISTS=11;
export type UniformArenaLayout="cell"|"face"|"vertex";
export interface UniformArenaChannel{name:string;level:0|2;layout:UniformArenaLayout;banks?:1|2}
export interface UniformArenaList{name:string;any:number}
export interface UniformArenaOptions{
 /** Blocks of the lattice (UM_BLOCKS). */
 blocks:number;
 /** Level 0 slots; default every brick of the lattice (cannot overflow). */
 fineCapacity?:number;
 channels:readonly UniformArenaChannel[];
 /** Compact slot lists by class: a live slot joins each list whose bits it has any of. */
 lists:readonly UniformArenaList[];
}
/** A channel staged by a kernel; seam names a WGSL fn(p:vec3i,word:u32)->f32
 * giving the value of owned index p (absolute; a vertex index p is lattice
 * vertex p+1) where the source is not a live h-owned brick: word is the
 * neighbour word (UB_NONE, UBA_SEAM|block, or a live slot whose tile is not
 * h-owned). */
export interface UniformArenaStaged{name:string;component?:number;bank?:0|1;seam:string}
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
 constructor(private readonly options:UniformArenaOptions){
  this.bricks=64*options.blocks;
  this.fineCapacity=options.fineCapacity??this.bricks;
  if(this.fineCapacity<1||this.fineCapacity>this.bricks){throw new Error(`Brick arena capacity ${this.fineCapacity} outside 1..${this.bricks}`);}
  if(options.lists.length>UBA_LISTS){throw new Error(`Brick arena: ${options.lists.length} lists, at most ${UBA_LISTS}`);}
  this.lists=options.lists;
  let words=0,banks=0;
  for(const channel of options.channels){
   const capacity=channel.level===0?this.fineCapacity:options.blocks;
   const components=channel.layout==="face"?3:1,count=channel.banks??1;
   if(count===2){if(banks===UBA_BANKS){throw new Error(`Brick arena: more than ${UBA_BANKS} two-bank channels`);}this.bankWords.set(channel.name,banks++);}
   for(let bank=0;bank<count;bank++)for(let component=0;component<components;component++){
    this.bases.set(`${channel.name}/${component}/${bank}`,words);words+=UBA_STRIDE*capacity;
   }
  }
  this.poolWords=words;
  const cap=this.fineCapacity;
  this.tableWords=UBA_HEADER+2*this.bricks+cap*(1+1+UBA_RECORD+1+1+1+1+this.lists.length);
 }
 /** The pool word of a channel's first slot. */
 base(name:string,component=0,bank=0):number{
  const base=this.bases.get(`${name}/${component}/${bank}`);
  if(base===undefined){throw new Error(`Brick arena: no channel ${name} component ${component} bank ${bank}`);}
  return base;
 }
 /** Table word offsets, from the table's base word. */
 private get layout(){
  const cap=this.fineCapacity,mark=UBA_HEADER,table=mark+this.bricks,brick=table+this.bricks,klass=brick+cap,record=klass+cap,free=record+UBA_RECORD*cap;
  const enter=free+cap,leave=enter+cap,live=leave+cap,lists=live+cap;
  return {mark,table,brick,klass,record,free,enter,leave,live,lists};
 }
 /** Words written once at attach (base: the table's first word): an empty
  * table and a full free stack. The only host writes the table takes. */
 initialWrites(base:number):{offset:number;data:Uint32Array<ArrayBuffer>}[]{
  const l=this.layout,cap=this.fineCapacity;
  const header=new Uint32Array(UBA_HEADER);header[1]=cap;
  const none=new Uint32Array(this.bricks+cap).fill(0xffffffff);
  const free=new Uint32Array(cap);for(let i=0;i<cap;i++){free[i]=cap-1-i;}
  return [{offset:4*base,data:header},{offset:4*(base+l.table),data:none},{offset:4*(base+l.free),data:free}];
 }
 /** The byte range each step clears before its markers: the counts and the marks. */
 stepClear(base:number):{offset:number;size:number}{return {offset:4*(base+2+UBA_BANKS),size:4*(UBA_HEADER-2-UBA_BANKS+this.bricks)};}
 /** The byte offset of the fatal word (0, or 0x80000000 | the brick id refused a slot). */
 fatalOffset(base:number):number{return 4*base;}

 /** The persistent table: constants, accessors and the four entries.
  * table: an array<atomic<u32>> the module binds; base: a WGSL u32
  * expression for the table's first word. Needs the lattice address;
  * tileMask: a WGSL u32 expression of id, the brick's h-owned tiles. */
 tableWGSL(table:string,base:string,tileMask="0xffu"):string{
  const l=this.layout,cap=this.fineCapacity,c=UNIFORM_ARENA_CLASS;
  const count=(k:number)=>`${base}+${2+UBA_BANKS+k}u`;
  return /* wgsl */`
const UBA_BRICKS:u32=${this.bricks}u;
const UBA_CAPACITY:u32=${cap}u;
const UBA_SEAM:u32=0x80000000u;
const UBA_FATAL:u32=${base};
const UBA_FREE_TOP:u32=${base}+1u;
const UBA_BANK:u32=${base}+2u;
const UBA_ENTER_COUNT:u32=${count(0)};
const UBA_LEAVE_COUNT:u32=${count(1)};
const UBA_LIVE_COUNT:u32=${count(2)};
const UBA_LIST_COUNT:u32=${count(3)};
const UBA_MARK:u32=${base}+${l.mark}u;
const UBA_TABLE:u32=${base}+${l.table}u;
const UBA_BRICK:u32=${base}+${l.brick}u;
const UBA_CLASS:u32=${base}+${l.klass}u;
const UBA_RECORD:u32=${base}+${l.record}u;
const UBA_FREE:u32=${base}+${l.free}u;
const UBA_ENTER:u32=${base}+${l.enter}u;
const UBA_LEAVE:u32=${base}+${l.leave}u;
const UBA_LIVE:u32=${base}+${l.live}u;
const UBA_LISTS:u32=${base}+${l.lists}u;
const UBA_ENTERING:u32=${c.entering>>>0}u;
const UBA_LEAVING:u32=${c.leaving>>>0}u;
const UBA_DIRTY:u32=${c.dirty}u;
const UBA_CORES:u32=${c.arenaCore|c.particlesCore}u;
${this.lists.map((list,j)=>`const UBA_LIST_${list.name.replace(/[A-Z]/g,ch=>`_${ch}`).toUpperCase()}:u32=${j}u;`).join("")}
fn ubaBrickAt(c:vec3i)->u32{let b=umPageWord(c>>vec3u(5u));if(!umBlockStored(b)){return UB_NONE_ID;}let l=(vec3u(c)>>vec3u(3u))&vec3u(3u);return 64u*b+(l.x|(l.y<<2u)|(l.z<<4u));}
const UB_NONE_ID:u32=0xffffffffu;
fn ubaOrigin(id:u32)->vec3u{let l=id&63u;return (umBlockPage(id>>6u)<<vec3u(5u))+8u*vec3u(l&3u,(l>>2u)&3u,l>>4u);}
fn ubaMark(id:u32,bits:u32){if((atomicLoad(&${table}[UBA_MARK+id])&bits)!=bits){atomicOr(&${table}[UBA_MARK+id],bits);}}
fn ubaSlot(id:u32)->u32{return atomicLoad(&${table}[UBA_TABLE+id]);}
fn ubaBrick(slot:u32)->u32{return atomicLoad(&${table}[UBA_BRICK+slot]);}
fn ubaClass(slot:u32)->u32{return atomicLoad(&${table}[UBA_CLASS+slot]);}
// Record words: 0..26 neighbours (x+3y+9z of (-1..1)³), 28 tile mask, 29 brick id.
fn ubaRecordWord(slot:u32,k:u32)->u32{return atomicLoad(&${table}[UBA_RECORD+${UBA_RECORD}u*slot+k]);}
fn ubaCount(word:u32)->u32{return min(atomicLoad(&${table}[word]),UBA_CAPACITY);}
fn ubaListCount(j:u32)->u32{return ubaCount(UBA_LIST_COUNT+j);}
fn ubaListSlot(j:u32,e:u32)->u32{return atomicLoad(&${table}[UBA_LISTS+UBA_CAPACITY*j+e]);}
fn ubaEnterSlot(e:u32)->u32{return atomicLoad(&${table}[UBA_ENTER+e]);}
fn ubaLeaveSlot(e:u32)->u32{return atomicLoad(&${table}[UBA_LEAVE+e]);}
// A two-bank channel's current bank; a stage flips it from one lane of a later pass.
fn ubaBank(j:u32)->u32{return atomicLoad(&${table}[UBA_BANK+j])&1u;}
fn ubaFlip(j:u32){atomicXor(&${table}[UBA_BANK+j],1u);}
fn ubaAppend(count:u32,list:u32,slot:u32){let e=atomicAdd(&${table}[count],1u);atomicStore(&${table}[list+e],slot);}
// A thread a brick id: dilate the core marks; an entering brick takes a
// slot, a leaving one is listed (its slot stays until ubaRelease), a dirty
// live one is listed for the gather again.
@compute @workgroup_size(64) fn ubaUpdate(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) groups:vec3u){
 for(var id=gid.x;id<UBA_BRICKS;id+=64u*groups.x){
  let block=id>>6u;var bits=0u;
  if(umBlockStored(block)&&umBlockWord(block)!=UM_FREE){
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
// A thread a slot: the live list and the class lists.
@compute @workgroup_size(64) fn ubaLists(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) groups:vec3u){
 for(var slot=gid.x;slot<UBA_CAPACITY;slot+=64u*groups.x){
  let bits=atomicLoad(&${table}[UBA_CLASS+slot]);if(bits==0u||(bits&UBA_LEAVING)!=0u){continue;}
  ubaAppend(UBA_LIVE_COUNT,UBA_LIVE,slot);
  ${this.lists.map((list,j)=>`if((bits&${list.any}u)!=0u){ubaAppend(UBA_LIST_COUNT+${j}u,UBA_LISTS+${j*cap}u,slot);}`).join("\n  ")}
 }
}
// A thread a (live slot, record word).
@compute @workgroup_size(64) fn ubaRecord(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) groups:vec3u){
 let n=${UBA_RECORD}u*ubaCount(UBA_LIVE_COUNT);
 for(var i=gid.x;i<n;i+=64u*groups.x){
  let slot=atomicLoad(&${table}[UBA_LIVE+i/${UBA_RECORD}u]);let k=i%${UBA_RECORD}u;let id=atomicLoad(&${table}[UBA_BRICK+slot]);
  var word=0u;
  if(k<27u){
   let other=ubaBrickAt(vec3i(ubaOrigin(id))+8*(vec3i(umCorner(k,3u))-1));
   word=UB_NONE_ID;
   if(other!=UB_NONE_ID){
    let s=atomicLoad(&${table}[UBA_TABLE+other]);
    word=select(UBA_SEAM|(other>>6u),s,s!=UB_NONE_ID&&(atomicLoad(&${table}[UBA_CLASS+s])&UBA_LEAVING)==0u);
   }
  }else if(k==28u){word=${tileMask};}else if(k==29u){word=id;}
  atomicStore(&${table}[UBA_RECORD+i],word);
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
// MG indices: a level-0 brick's level-1 key and octant, and the parent cell of child cell c.
fn ubaParentKey(id:u32)->u32{let l=id&63u;return 8u*(id>>6u)+(((l>>1u)&1u)|(((l>>3u)&1u)<<1u)|(((l>>5u)&1u)<<2u));}
fn ubaOctant(id:u32)->u32{let l=id&63u;return (l&1u)|(((l>>2u)&1u)<<1u)|(((l>>4u)&1u)<<2u);}
fn ubaOctantCell(octant:u32,c:vec3u)->vec3u{return 4u*vec3u(octant&1u,(octant>>1u)&1u,octant>>2u)+(c>>vec3u(1u));}
`;}

 /** Channel bases and the pool accessors. pool: an array<u32> the module binds. */
 poolWGSL(pool:string):string{
  const constants=[...this.bases].map(([key,base])=>{const [name,component,bank]=key.split("/");return `const UBA_${name!.toUpperCase()}_${component}_${bank}:u32=${base}u;`;}).join("");
  const banks=[...this.bankWords].map(([name,j])=>`const UBA_BANK_${name.toUpperCase()}:u32=${j}u;`).join("");
  return /* wgsl */`
const UBA_STRIDE:u32=${UBA_STRIDE}u;
${constants}${banks}
fn ubaLoad(base:u32,slot:u32,i:u32)->f32{return bitcast<f32>(${pool}[base+UBA_STRIDE*slot+i]);}
fn ubaStore(base:u32,slot:u32,i:u32,value:f32){${pool}[base+UBA_STRIDE*slot+i]=bitcast<u32>(value);}
`;}
 /** A channel's base expression: a two-bank channel's bank picks by the table's bank word (bank 0 current, 1 the other). */
 private baseExpression(name:string,component=0,bank:0|1=0):string{
  const j=this.bankWords.get(name),key=name.toUpperCase();this.base(name,component,0);
  if(j===undefined){if(bank!==0){throw new Error(`Brick arena: ${name} has one bank`);}return `UBA_${key}_${component}_0`;}
  return `select(UBA_${key}_${component}_0,UBA_${key}_${component}_1,(ubaBank(${j}u)^${bank}u)!=0u)`;
 }

 /** The halo-staging helper of one kernel (one call site): ubaStage_K(slot,
  * lane) fills ubaW_K with the brick's box of owned indices -halo ..
  * 8+halo of each staged channel, ubaAt_K(c,j) indexes it. Seam values
  * come from each channel's seam fn and are never stored. */
 stageWGSL(kernel:string,options:{channels:readonly UniformArenaStaged[];halo:number;workgroup:number}):string{
  const {channels,halo,workgroup}=options;
  if(!Number.isInteger(halo)||halo<0||halo>4){throw new Error(`Brick arena staging halo ${halo} outside 0..4`);}
  const e=8+2*halo,e3=e*e*e,k=kernel;
  const live=channels.map((ch,j)=>`ubaW_${k}[k+${j*e3}u]=ubaLoad(${this.baseExpression(ch.name,ch.component,ch.bank)},w,i);`).join("");
  const seam=channels.map((ch,j)=>`ubaW_${k}[k+${j*e3}u]=${ch.seam}(origin+c,w);`).join("");
  return /* wgsl */`
const UBA_E_${k}:u32=${e}u;
var<workgroup> ubaW_${k}:array<f32,${e3*channels.length}>;
var<workgroup> ubaR_${k}:array<u32,28>;
var<workgroup> ubaM_${k}:array<u32,27>;
fn ubaAt_${k}(c:vec3i,j:u32)->u32{let q=vec3u(c+${halo});return q.x+${e}u*(q.y+${e}u*q.z)+${e3}u*j;}
fn ubaStage_${k}(slot:u32,lane:u32){
 workgroupBarrier();
 if(lane<27u){let w=ubaRecordWord(slot,lane);ubaR_${k}[lane]=w;var m=0u;if(w<UBA_SEAM){m=ubaRecordWord(w,28u);}ubaM_${k}[lane]=m;}
 else if(lane==27u){ubaR_${k}[27]=ubaRecordWord(slot,29u);}
 workgroupBarrier();
 let origin=vec3i(ubaOrigin(ubaR_${k}[27]));
 for(var k=lane;k<${e3}u;k+=${workgroup}u){
  let c=vec3i(vec3u(k%${e}u,(k/${e}u)%${e}u,k/${e*e}u))-${halo};let d=vec3u(c+8)>>vec3u(3u);let n=d.x+3u*d.y+9u*d.z;
  let w=ubaR_${k}[n];let l=vec3u(c&vec3i(7));let t=(l.x>>2u)|((l.y>>2u)<<1u)|((l.z>>2u)<<2u);
  if(w<UBA_SEAM&&((ubaM_${k}[n]>>t)&1u)!=0u){let i=l.x+8u*(l.y+8u*l.z);${live}}else{${seam}}
 }
 workgroupBarrier();
}
`;}

 /** A churn skeleton entry: a workgroup a slot of a list (the enter list,
  * the leave list, or a class list by name), each copied channel's 512
  * values through its fn. gather: fn(p) into the slot; scatter: the slot
  * into fn(p,value). */
 copyWGSL(entry:string,options:{list:string;direction:"gather"|"scatter";channels:readonly UniformArenaCopied[];workgroup:number}):string{
  const {list,direction,channels,workgroup}=options;
  const j=this.lists.findIndex(item=>item.name===list);
  const [count,slotOf]=list==="enter"?["ubaCount(UBA_ENTER_COUNT)","ubaEnterSlot(e)"]:list==="leave"?["ubaCount(UBA_LEAVE_COUNT)","ubaLeaveSlot(e)"]
   :j>=0?[`ubaListCount(${j}u)`,`ubaListSlot(${j}u,e)`]:(()=>{throw new Error(`Brick arena: no list ${list}`);})();
  const body=channels.map(ch=>{const base=this.baseExpression(ch.name,ch.component,ch.bank);
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
