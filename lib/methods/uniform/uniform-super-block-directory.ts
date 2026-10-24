/** The super-block directory (docs/plans/tiered-adaptivity-huge-grids-2026-10-24.md
 * §2.1, the storage half of Phase 3): the page table one level up, and the
 * 16h tier's storage. Built and tested, not yet wired.
 *
 * Three levels, each a 64-word child table:
 * - top: a 64³ toroidal window of super-blocks (128³ h, 4³ pages), the
 *   window's 6-bit packing as today's page table. Window super-block s
 *   lives at entry s & 63 per axis, so the window spans 64·128 = 8192 h an
 *   axis (today's page table: 64·32 = 2048). Word: AIR (0), SOLID
 *   (0xffffffff) or the super slot that holds it;
 * - a super slot: 64 child words, one a page (x | y<<2 | z<<4 of its page
 *   in the super-block): AIR, SOLID or the pool block that holds it (today's
 *   page table word);
 * - a block: 64 child words, one a brick, the brick slot or NONE. That is
 *   the brick arena's table (UBA_TABLE + 64·block + i, uniform-brick-arena.ts),
 *   unchanged: the directory does not duplicate it.
 *
 * Slot 0 is the AIR sentinel (every child AIR) and slot capacity+1 the SOLID
 * sentinel (every child SOLID), so a page's word is two dependent loads with
 * no branch: child[64·min(top, SOLID slot) + page & 3] (usdPageWord). That
 * is umPageWord's replacement during the migration (§2.1: inner loops keep
 * to records and staging; the tree is walked by the record builder,
 * staging, particles and the renderer).
 *
 * A block's word (usdBlockPage, today's umBlockWord) packs its window page
 * at 8 bits an axis (x | y<<8 | z<<16), since the window holds 256 pages an
 * axis.
 *
 * Lifetime (GPU, fixed launches, fail-fast as the brick arena and the page
 * pool): consumers append super-block coordinates (usdRequest) during a
 * step; the commit then runs
 *   usdRecycle  one workgroup: the previous commit's retired slots join the
 *               free stack (one commit of quarantine), the commit's counts clear;
 *   usdClaim    a thread a request: an AIR entry takes a free slot (the first
 *               request for it wins the entry by compare-and-swap); a request
 *               for a held entry keeps it; one whose entry holds another
 *               super-block is the alias fatal, one that finds the stack
 *               empty the exhausted fatal (no growth);
 *   usdPlace    a workgroup a request: a claimed slot's 64 child words start
 *               AIR, its record and entry are written;
 *   usdAge      a workgroup a slot: a slot with no stored child and no
 *               request ages, and at idleSteps retires (its entry back to
 *               AIR, the slot quarantined); the live list is rebuilt.
 * usdSetPage / usdClearPage name a page's block in its live super slot (the
 * page pool's claim and retirement, once the pool commits through it).
 *
 * The 16h tier: SoA channels a super slot (8³ cells of 16h, the brick
 * arena's 520-word stride and layouts), indexed by slot (usdLoad/usdStore).
 *
 * The pressure tail (TAIL_ROWS, uniform-pressure-arena.ts) is one row a
 * super-block: the capacity is capped at USD_TAIL_ROWS. */
import { UBA_STRIDE, type UniformArenaLayout } from "./uniform-brick-arena";

/** Top window edge (super-blocks), pages a super-block edge, h cells a super-block edge. */
export const USD_WINDOW=64,USD_PAGES=4,USD_CELLS=128;
/** h cells an axis the window spans. */
export const USD_AXIS_CELLS=USD_WINDOW*USD_CELLS;
/** Words of no storage, in the top table and in a child table. */
export const USD_AIR=0,USD_SOLID=0xffffffff;
/** A block word of a block that holds no page. */
export const USD_FREE=0xffffffff;
/** The pressure tail's rows: one a super-block. */
export const USD_TAIL_ROWS=4096;
/** Sticky fatal bits (header word 0). exhausted: a claim found the free
 * stack empty. alias: a request's entry holds another super-block.
 * requests: more requests in a step than the list holds. */
export const USD_FATAL={exhausted:1,alias:2,requests:4} as const;
/** Record states. */
export const USD_STATE={free:0,live:1,pending:2,sentinel:3} as const;
/** Header words. */
export const USD_HEADER={fatal:0,fatalEntry:1,free:2,pending:3,live:4,requests:5,claimed:6,retired:7,generation:8,words:16} as const;
/** Record words a slot: super coordinate x, y, z (i32), state, idle commits, kept. */
const RECORD=8;
/** A claim in flight: the request index under this bit (never a slot: slots stay under USD_TAIL_ROWS + 2). */
const MARK=0x80000000;
/** The default capacity floor, and the growth over a census (§2.7). */
export const UNIFORM_SUPER_BLOCK_FLOOR=8,UNIFORM_SUPER_BLOCK_GROWTH=1.5;

export interface UniformSuperBlockChannel{name:string;layout:UniformArenaLayout;banks?:1|2}
export interface UniformSuperBlockDirectoryOptions{
 /** Super slots, the sentinels apart: 1..USD_TAIL_ROWS. */
 capacity:number;
 /** Pool blocks (UM_BLOCKS): one block word each. */
 blocks:number;
 /** A slot with no stored child and no request retires at the idleSteps-th commit in a row. */
 idleSteps:number;
 /** Requests a step may append (default: capacity). */
 requestCapacity?:number;
 /** The 16h channels, one SoA pool. */
 channels?:readonly UniformSuperBlockChannel[];
}
/** A page of a seed: its window page coordinate (signed) and its block. */
export interface UniformSuperBlockSeedPage{readonly x:number;readonly y:number;readonly z:number;readonly block:number}
/** Kinds of a page without a block (the page pool's UNIFORM_PAGE_POOL_KIND.air / .solid). */
export const USD_KIND={air:1,solid:2} as const;

/** Super slots of a scene: tiers.superBlocks, else the census × 1.5 over a
 * floor of UNIFORM_SUPER_BLOCK_FLOOR, at most USD_TAIL_ROWS. */
export function uniformSuperBlockCapacity(census:number,tiers?:{readonly superBlocks?:number}):number{
 const want=tiers?.superBlocks??Math.max(UNIFORM_SUPER_BLOCK_FLOOR,Math.ceil(UNIFORM_SUPER_BLOCK_GROWTH*census));
 if(!Number.isSafeInteger(want)||want<1)throw new RangeError(`Super-block capacity must be a positive integer: ${want}`);
 return Math.min(USD_TAIL_ROWS,want);
}
/** Top entry of super-block s (window coordinates, any sign). */
export function uniformSuperBlockEntry(x:number,y:number,z:number):number{return (x&63)|((y&63)<<6)|((z&63)<<12);}
/** A page's child index in its super-block. */
export function uniformSuperBlockChild(x:number,y:number,z:number):number{return (x&3)|((y&3)<<2)|((z&3)<<4);}
/** The block word of a block holding page (x,y,z): the window page at 8 bits an axis. */
export function uniformSuperBlockBlockWord(x:number,y:number,z:number):number{return (x&255)|((y&255)<<8)|((z&255)<<16);}

export class UniformSuperBlockDirectory{
 readonly capacity:number;
 readonly blocks:number;
 readonly idleSteps:number;
 readonly requestCapacity:number;
 /** Slots including the two sentinels. */
 readonly slots:number;
 readonly solidSlot:number;
 readonly stateWords:number;
 readonly poolWords:number;
 private readonly bases=new Map<string,number>();
 constructor(private readonly options:UniformSuperBlockDirectoryOptions){
  const {capacity,blocks,idleSteps}=options;
  if(!Number.isSafeInteger(capacity)||capacity<1||capacity>USD_TAIL_ROWS)throw new RangeError(`Super-block directory capacity ${capacity} outside 1..${USD_TAIL_ROWS} (the pressure tail's rows)`);
  if(!Number.isSafeInteger(blocks)||blocks<3)throw new RangeError(`Super-block directory blocks must be an integer from 3: ${blocks}`);
  if(!Number.isSafeInteger(idleSteps)||idleSteps<1)throw new RangeError(`Super-block directory idle steps must be a positive integer: ${idleSteps}`);
  this.capacity=capacity;this.blocks=blocks;this.idleSteps=idleSteps;
  this.requestCapacity=options.requestCapacity??capacity;
  if(!Number.isSafeInteger(this.requestCapacity)||this.requestCapacity<1||this.requestCapacity>=1<<24)throw new RangeError(`Super-block directory request capacity outside 1..2^24: ${this.requestCapacity}`);
  this.slots=capacity+2;this.solidSlot=capacity+1;
  this.stateWords=this.layout.words;
  let words=0;
  for(const ch of options.channels??[]){
   const components=ch.layout==="face"?3:1;
   for(let bank=0;bank<(ch.banks??1);bank++)for(let component=0;component<components;component++){this.bases.set(`${ch.name}/${component}/${bank}`,words);words+=UBA_STRIDE*this.slots;}
  }
  this.poolWords=words;
 }
 /** Word offsets of the state buffer. */
 get layout(){
  const top=USD_HEADER.words,child=top+USD_WINDOW**3,records=child+64*this.slots,free=records+RECORD*this.slots,pending=free+this.capacity;
  const live=pending+this.capacity,claimed=live+this.capacity,claimSlot=claimed+this.capacity,requests=claimSlot+this.requestCapacity;
  const blockWords=requests+4*this.requestCapacity;
  return {top,child,records,free,pending,live,claimed,claimSlot,requests,blockWords,words:blockWords+this.blocks} as const;
 }
 /** Bytes of the directory and of its 16h channels. */
 get allocatedBytes():number{return 4*(this.stateWords+this.poolWords);}
 /** The pool word of a 16h channel's slot 0. */
 base(name:string,component=0,bank=0):number{
  const base=this.bases.get(`${name}/${component}/${bank}`);
  if(base===undefined)throw new Error(`Super-block directory: no channel ${name} component ${component} bank ${bank}`);
  return base;
 }

 /** The state at scene build: every super-block holding a seeded page takes
  * a slot (ascending, in the order of first appearance), its pages' words
  * the blocks; its other pages take other(page) (USD_KIND, window page
  * coordinates). A super-block with no seeded page is AIR or SOLID when its
  * 64 pages all are, else it takes a slot too. Entries never named stay
  * as `outside` gives for super-block coordinates (default SOLID). Throws on
  * a block out of range or named twice, two pages on one entry, two
  * super-blocks on one entry, and more super-blocks than slots. */
 seed(pages:readonly UniformSuperBlockSeedPage[],other:(x:number,y:number,z:number)=>number,superBlocks:readonly {readonly x:number;readonly y:number;readonly z:number}[]=[]):Uint32Array<ArrayBuffer>{
  const l=this.layout,state=new Uint32Array(l.words),signed=new Int32Array(state.buffer);
  const top=state.subarray(l.top,l.child);
  top.fill(USD_SOLID);
  state.fill(USD_SOLID,l.child+64*this.solidSlot,l.child+64*this.slots);
  state.fill(USD_FREE,l.blockWords,l.blockWords+this.blocks);
  state[l.records+RECORD*0+3]=state[l.records+RECORD*this.solidSlot+3]=USD_STATE.sentinel;
  let used=0;
  const key=(x:number,y:number,z:number)=>`${x},${y},${z}`;
  const held=new Map<number,string>();
  const take=(sx:number,sy:number,sz:number):number=>{
   const entry=uniformSuperBlockEntry(sx,sy,sz),name=key(sx,sy,sz),at=held.get(entry);
   if(at!==undefined){if(at!==name)throw new Error(`Super-block directory seed: super-block (${name}) aliases (${at}) in the ${USD_WINDOW}³ window`);return top[entry]!;}
   if(used>=this.capacity)throw new RangeError(`Super-block directory seed needs more than its ${this.capacity} slots`);
   const slot=++used;held.set(entry,name);top[entry]=slot;
   const r=l.records+RECORD*slot;signed[r]=sx;signed[r+1]=sy;signed[r+2]=sz;state[r+3]=USD_STATE.live;
   for(let c=0;c<64;c++){const px=4*sx+(c&3),py=4*sy+((c>>2)&3),pz=4*sz+(c>>4);state[l.child+64*slot+c]=other(px,py,pz)===USD_KIND.solid?USD_SOLID:USD_AIR;}
   return slot;
  };
  const named=new Set<number>();
  pages.forEach(({x,y,z,block},i)=>{
   if((x|0)!==x||(y|0)!==y||(z|0)!==z)throw new RangeError(`Super-block directory seed page ${i}: coordinates must be 32-bit integers`);
   if(!Number.isSafeInteger(block)||block<1||block>this.blocks-2)throw new RangeError(`Super-block directory seed page ${i} (${x}, ${y}, ${z}): block ${block} outside 1..${this.blocks-2}`);
   if(named.has(block))throw new Error(`Super-block directory seed page ${i} (${x}, ${y}, ${z}): block ${block} named twice`);
   named.add(block);
   const slot=take(x>>2,y>>2,z>>2),at=l.child+64*slot+uniformSuperBlockChild(x,y,z);
   if(state[at]!==USD_AIR&&state[at]!==USD_SOLID)throw new Error(`Super-block directory seed page ${i} (${x}, ${y}, ${z}): its child word holds block ${state[at]}`);
   state[at]=block;state[l.blockWords+block]=uniformSuperBlockBlockWord(x,y,z);
  });
  // Super-blocks the caller names without a seeded page (the box's): AIR or
  // SOLID when uniform, else a slot.
  for(const {x,y,z} of superBlocks){
   const entry=uniformSuperBlockEntry(x,y,z);if(held.has(entry)){take(x,y,z);continue;}
   let kinds=0;
   for(let c=0;c<64;c++)kinds|=other(4*x+(c&3),4*y+((c>>2)&3),4*z+(c>>4))===USD_KIND.solid?2:1;
   if(kinds===3)take(x,y,z);else top[entry]=kinds===2?USD_SOLID:USD_AIR;
  }
  for(let s=1;s<=this.capacity;s++){
   if(state[l.records+RECORD*s+3]===USD_STATE.live)state[l.live+state[USD_HEADER.live]++]=s;
  }
  // The free stack pops from its top: the highest slots first, so the next claim takes used+1.
  for(let s=this.capacity;s>used;s--)state[l.free+state[USD_HEADER.free]++]=s;
  return state;
 }
 /** Host lookup of a page's word in a state image (usdPageWord). */
 pageWord(state:ArrayLike<number>,x:number,y:number,z:number):number{
  const l=this.layout,t=state[l.top+uniformSuperBlockEntry(x>>2,y>>2,z>>2)]!;
  const slot=t>=MARK?this.solidSlot:t;
  return state[l.child+64*slot+uniformSuperBlockChild(x,y,z)]!;
 }

 /** The directory: constants, lookups, the request and page fns, and the
  * commit entries. state: an array<atomic<u32>> the module binds, the
  * directory at word 0. */
 wgsl(state:string):string{
  const l=this.layout,H=USD_HEADER,F=USD_FATAL,S=USD_STATE;
  return /* wgsl */`
const USD_AIR:u32=0u;
const USD_SOLID:u32=0xffffffffu;
const USD_FREE:u32=0xffffffffu;
const USD_MARK:u32=${MARK}u;
// A claim word: the request won its entry but found the stack empty.
const USD_RESET:u32=0xfffffffeu;
const USD_CAPACITY:u32=${this.capacity}u;
const USD_SOLID_SLOT:u32=${this.solidSlot}u;
const USD_REQUESTS:u32=${this.requestCapacity}u;
const USD_IDLE:u32=${this.idleSteps}u;
const USD_TOP:u32=${l.top}u;
const USD_CHILD:u32=${l.child}u;
const USD_RECORDS:u32=${l.records}u;
const USD_FREE_STACK:u32=${l.free}u;
const USD_PENDING:u32=${l.pending}u;
const USD_LIVE:u32=${l.live}u;
const USD_CLAIMED:u32=${l.claimed}u;
const USD_CLAIM_SLOT:u32=${l.claimSlot}u;
const USD_REQUEST_LIST:u32=${l.requests}u;
const USD_BLOCK_WORDS:u32=${l.blockWords}u;
fn usdEntry(s:vec3i)->u32{let q=vec3u(s&vec3i(63));return q.x|(q.y<<6u)|(q.z<<12u);}
fn usdChildIndex(p:vec3i)->u32{let q=vec3u(p&vec3i(3));return q.x|(q.y<<2u)|(q.z<<4u);}
// The super slot of window super-block s: 0 (AIR) and USD_SOLID_SLOT are the sentinels.
fn usdSuperSlot(s:vec3i)->u32{return min(atomicLoad(&${state}[USD_TOP+usdEntry(s)]),USD_SOLID_SLOT);}
// A page's word, as umPageWord: AIR, SOLID or its block.
fn usdPageWord(p:vec3i)->u32{return atomicLoad(&${state}[USD_CHILD+64u*usdSuperSlot(p>>vec3u(2u))+usdChildIndex(p)]);}
// A block's window page, 8 bits an axis.
fn usdBlockPage(b:u32)->vec3u{let w=atomicLoad(&${state}[USD_BLOCK_WORDS+b]);return vec3u(w&255u,(w>>8u)&255u,(w>>16u)&255u);}
fn usdRecordCoord(slot:u32)->vec3i{let r=USD_RECORDS+${RECORD}u*slot;return vec3i(bitcast<i32>(atomicLoad(&${state}[r])),bitcast<i32>(atomicLoad(&${state}[r+1u])),bitcast<i32>(atomicLoad(&${state}[r+2u])));}
fn usdFatal(bit:u32,entry:u32){if(atomicOr(&${state}[${H.fatal}u],bit)==0u){atomicStore(&${state}[${H.fatalEntry}u],entry);}}
// A consumer asks for super-block s this step (a claim if its entry is AIR, a keep if it holds s).
fn usdRequest(s:vec3i){
 let e=atomicAdd(&${state}[${H.requests}u],1u);
 if(e>=USD_REQUESTS){usdFatal(${F.requests}u,usdEntry(s));return;}
 let r=USD_REQUEST_LIST+4u*e;atomicStore(&${state}[r],bitcast<u32>(s.x));atomicStore(&${state}[r+1u],bitcast<u32>(s.y));atomicStore(&${state}[r+2u],bitcast<u32>(s.z));
}
fn usdRequestCoord(e:u32)->vec3i{let r=USD_REQUEST_LIST+4u*e;return vec3i(bitcast<i32>(atomicLoad(&${state}[r])),bitcast<i32>(atomicLoad(&${state}[r+1u])),bitcast<i32>(atomicLoad(&${state}[r+2u])));}
// A page's block in its live super slot (false: the super-block holds no slot), and its block word.
fn usdSetPage(p:vec3i,block:u32)->bool{
 let slot=usdSuperSlot(p>>vec3u(2u));if(slot==0u||slot==USD_SOLID_SLOT){return false;}
 atomicStore(&${state}[USD_CHILD+64u*slot+usdChildIndex(p)],block);
 let q=vec3u(p&vec3i(255));atomicStore(&${state}[USD_BLOCK_WORDS+block],q.x|(q.y<<8u)|(q.z<<16u));
 return true;
}
fn usdClearPage(p:vec3i){
 let slot=usdSuperSlot(p>>vec3u(2u));if(slot==0u||slot==USD_SOLID_SLOT){return;}
 let at=USD_CHILD+64u*slot+usdChildIndex(p);let block=atomicLoad(&${state}[at]);
 if(block!=USD_AIR&&block!=USD_SOLID){atomicStore(&${state}[USD_BLOCK_WORDS+block],USD_FREE);atomicStore(&${state}[at],USD_AIR);}
}
fn usdLiveCount()->u32{return min(atomicLoad(&${state}[${H.live}u]),USD_CAPACITY);}
fn usdLiveSlot(e:u32)->u32{return atomicLoad(&${state}[USD_LIVE+e]);}
// One workgroup: the previous commit's retired slots join the free stack; the counts clear.
@compute @workgroup_size(64) fn usdRecycle(@builtin(local_invocation_index) lane:u32){
 let free=atomicLoad(&${state}[${H.free}u]);let pending=atomicLoad(&${state}[${H.pending}u]);
 for(var e=lane;e<pending;e+=64u){
  let slot=atomicLoad(&${state}[USD_PENDING+e]);
  atomicStore(&${state}[USD_RECORDS+${RECORD}u*slot+3u],${S.free}u);
  atomicStore(&${state}[USD_FREE_STACK+free+e],slot);
 }
 storageBarrier();
 if(lane==0u){
  atomicStore(&${state}[${H.free}u],free+pending);atomicStore(&${state}[${H.pending}u],0u);
  atomicStore(&${state}[${H.live}u],0u);atomicStore(&${state}[${H.claimed}u],0u);atomicStore(&${state}[${H.retired}u],0u);
  atomicAdd(&${state}[${H.generation}u],1u);
 }
}
// A thread a request: an AIR entry is won by compare-and-swap (USD_MARK | request) and takes a slot.
@compute @workgroup_size(64) fn usdClaim(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) groups:vec3u){
 let n=min(atomicLoad(&${state}[${H.requests}u]),USD_REQUESTS);
 for(var e=gid.x;e<n;e+=64u*groups.x){
  atomicStore(&${state}[USD_CLAIM_SLOT+e],USD_FREE);
  let s=usdRequestCoord(e);let entry=usdEntry(s);
  var word=atomicLoad(&${state}[USD_TOP+entry]);
  if(word==USD_SOLID){continue;}
  if(word==USD_AIR){
   // A weak compare-and-swap may fail spuriously: it re-reads AIR and tries again.
   for(var attempt=0u;attempt<64u&&word==USD_AIR;attempt++){
    let swap=atomicCompareExchangeWeak(&${state}[USD_TOP+entry],USD_AIR,USD_MARK|e);
    word=select(swap.old_value,USD_MARK|e,swap.exchanged);
   }
   if(word==(USD_MARK|e)){
    let top=atomicSub(&${state}[${H.free}u],1u);
    if(top==0u||top>USD_CAPACITY){atomicAdd(&${state}[${H.free}u],1u);usdFatal(${F.exhausted}u,entry);atomicStore(&${state}[USD_CLAIM_SLOT+e],USD_RESET);continue;}
    atomicStore(&${state}[USD_CLAIM_SLOT+e],atomicLoad(&${state}[USD_FREE_STACK+top-1u]));
    continue;
   }
   if(word==USD_AIR){usdFatal(${F.alias}u,entry);continue;}
  }
  if(word>=USD_MARK){
   // Claimed this commit by request word & ~USD_MARK: the same super-block is a duplicate.
   if(any(usdRequestCoord(word&~USD_MARK)!=s)){usdFatal(${F.alias}u,entry);}
   continue;
  }
  if(any(usdRecordCoord(word)!=s)){usdFatal(${F.alias}u,entry);continue;}
  atomicStore(&${state}[USD_RECORDS+${RECORD}u*word+5u],1u);
 }
}
// A workgroup a request, a lane a child word: a won entry takes its slot (or
// goes back to AIR when the stack was empty). The claim word decides, never
// the entry, which lane 0 rewrites.
@compute @workgroup_size(64) fn usdPlace(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 let n=min(atomicLoad(&${state}[${H.requests}u]),USD_REQUESTS);
 for(var e=group.x;e<n;e+=groups.x){
  let slot=atomicLoad(&${state}[USD_CLAIM_SLOT+e]);
  if(slot==USD_FREE){continue;}
  let s=usdRequestCoord(e);let entry=usdEntry(s);
  if(slot==USD_RESET){if(lane==0u){atomicStore(&${state}[USD_TOP+entry],USD_AIR);}continue;}
  atomicStore(&${state}[USD_CHILD+64u*slot+lane],USD_AIR);
  if(lane==0u){
   let r=USD_RECORDS+${RECORD}u*slot;
   atomicStore(&${state}[r],bitcast<u32>(s.x));atomicStore(&${state}[r+1u],bitcast<u32>(s.y));atomicStore(&${state}[r+2u],bitcast<u32>(s.z));
   atomicStore(&${state}[r+3u],${S.live}u);atomicStore(&${state}[r+4u],0u);atomicStore(&${state}[r+5u],1u);
   let c=atomicAdd(&${state}[${H.claimed}u],1u);atomicStore(&${state}[USD_CLAIMED+c],slot);
   atomicStore(&${state}[USD_TOP+entry],slot);
  }
 }
}
var<workgroup> usdStored:atomic<u32>;
// A workgroup a slot, a lane a child word: a live slot with no stored child
// and no request ages; at USD_IDLE it retires (entry AIR, slot pending).
// The live list is rebuilt; the request list empties.
@compute @workgroup_size(64) fn usdAge(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 if(group.x==0u&&lane==0u){atomicStore(&${state}[${H.requests}u],0u);}
 for(var slot=group.x+1u;slot<=USD_CAPACITY;slot+=groups.x){
  if(lane==0u){atomicStore(&usdStored,0u);}
  workgroupBarrier();
  let r=USD_RECORDS+${RECORD}u*slot;
  let live=atomicLoad(&${state}[r+3u])==${S.live}u;
  let w=atomicLoad(&${state}[USD_CHILD+64u*slot+lane]);
  if(live&&w!=USD_AIR&&w!=USD_SOLID){atomicOr(&usdStored,1u);}
  workgroupBarrier();
  if(lane==0u&&live){
   let stored=atomicLoad(&usdStored);
   let kept=atomicExchange(&${state}[r+5u],0u);
   var idle=0u;if(stored==0u&&kept==0u){idle=atomicLoad(&${state}[r+4u])+1u;}
   atomicStore(&${state}[r+4u],idle);
   if(idle>=USD_IDLE){
    atomicStore(&${state}[USD_TOP+usdEntry(usdRecordCoord(slot))],USD_AIR);atomicStore(&${state}[r+3u],${S.pending}u);
    let p=atomicAdd(&${state}[${H.pending}u],1u);atomicStore(&${state}[USD_PENDING+p],slot);atomicAdd(&${state}[${H.retired}u],1u);
   }else{
    let e=atomicAdd(&${state}[${H.live}u],1u);atomicStore(&${state}[USD_LIVE+e],slot);
   }
  }
  workgroupBarrier();
 }
}
`;}
 /** The commit's entries, in order, and their launch shapes (workgroups). */
 get commitEntries():readonly {entry:string;workgroups:number}[]{
  return [{entry:"usdRecycle",workgroups:1},{entry:"usdClaim",workgroups:Math.ceil(this.requestCapacity/64)},
   {entry:"usdPlace",workgroups:Math.min(this.requestCapacity,1024)},{entry:"usdAge",workgroups:Math.min(this.capacity,1024)}];
 }
 /** The 16h channels' bases (from pool word base) and accessors. pool: an array<u32> the module binds. */
 poolWGSL(pool:string,base=0):string{
  const constants=[...this.bases].map(([key,word])=>{const [name,component,bank]=key.split("/");return `const USD_${name!.toUpperCase()}_${component}_${bank}:u32=${base+word}u;`;}).join("");
  return /* wgsl */`
const USD_STRIDE:u32=${UBA_STRIDE}u;
${constants}
fn usdLoad(base:u32,slot:u32,i:u32)->f32{return bitcast<f32>(${pool}[base+USD_STRIDE*slot+i]);}
fn usdStore(base:u32,slot:u32,i:u32,value:f32){${pool}[base+USD_STRIDE*slot+i]=bitcast<u32>(value);}
`;}
}
