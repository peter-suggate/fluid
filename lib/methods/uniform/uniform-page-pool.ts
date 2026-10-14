/** GPU page pool of the domain-less uniform solver: `blocks` storage blocks
 * of 32^3 cells. Block 0 is the AIR sentinel and block `blocks`-1 the SOLID
 * sentinel; blocks 1..blocks-2 are the slots, claimed on demand anywhere in
 * an unbounded signed page lattice and retired when idle. A toroidal page
 * table of `table`^3 words is the only directory: page q lives at entry
 * q & (table-1) per axis, so two resident pages `table` apart on an axis
 * alias one entry, which is fatal. A table word is 0 (AIR: no storage),
 * 0xffffffff (SOLID: no storage) or the block 1..blocks-2 that holds the
 * page, so min(word, blocks-1) is the block a read resolves to. SOLID is
 * written by the host seed alone: no kernel writes it, and a SOLID entry is
 * never requested, claimed or retired. Everything is GPU resident; the host
 * seeds the pool at scene build (uniformPagePoolSeed, UniformPagePool.seed),
 * queues page coordinates (request), fixes the pinned pages (pin) and
 * encodes fixed launches (encodeCommit). The mixed frame owns the solver's
 * (UniformMixedFrame.pagePool): the census asks for its residency
 * certificate's pages and one commit follows every census.
 *
 * Three buffers, all created once: state, work and the published block record.
 * Publication is independent of whether the scene names a seed box.
 *
 * state (UniformPagePool.state; consumers bind it read-only, array<u32>):
 *   [0,16)   status (UNIFORM_PAGE_POOL_STATUS), rewritten by every commit:
 *            0 generation       commits made since the seed
 *            1 resident         length of the resident list
 *            2 claimed          length of the new list (this commit's claims)
 *            3 free             free blocks, the quarantined apart
 *            4 pending          blocks this commit retired (quarantined); the next
 *                               commit may claim them and the free ones
 *            5 fatal            sticky UNIFORM_PAGE_POOL_FATAL bits
 *            6..8 fatalPage     the first offending page coordinate (i32 x 3)
 *            9 fatalGeneration  the generation that first raised a bit
 *            10 fatalFirst      the bit raised first (fatalPage is its page)
 *            11 requests        distinct requests this commit listed
 *            12 fatalDetail     a number that goes with fatalFirst: the pages
 *                               resident before the commit (exhausted), the
 *                               certificate's reach (closureCap), else 0
 *            13..15 reserved
 *   table    [16, 16+table^3): entry x | y<<b | z<<2b (b = log2 table) of the
 *            masked page coordinate; 0 AIR, 0xffffffff SOLID, else the block
 *   window   32 runtime words after the directory: F, revision, delta, fatal,
 *            boundary counter, world policy/box and the disabled probe command
 *   blocks   one word per block, after the runtime header: the world page residue
 *            x | y<<6 | z<<12 (each coordinate & (table-1)) of the page a
 *            resident block holds; 0xffffffff for a free or quarantined block
 *            and for the two sentinels
 *   records  8 words per block (UNIFORM_PAGE_POOL_RECORD): page x,y,z (i32),
 *            state (UNIFORM_PAGE_POOL_BLOCK), generation of the last claim (0:
 *            seeded), idle commits, 2 reserved. A free or quarantined block
 *            keeps the page it last held.
 *   resident `blocks`-2 words: the resident blocks, ascending, [0, status.resident)
 *   new      `blocks`-2 words: the blocks this commit claimed, [0, status.claimed),
 *            unordered; a consumer initialises those pages. The list is this
 *            commit's alone: empty at generation 0, so seeded residents are
 *            never in it and whoever seeds the pool initialises their blocks.
 * Word offsets: uniformPagePoolLayout(options).state.
 *
 * published: 32 header words [edge, blocks, table edge, directory offset,
 * resident count, record stride=4, version=2, reserved, 12 frame lanes,
 * signed world resident bounds at words 20..22 / 24..26, padding], then one record [signed world page x,y,z, state] per pool block:
 * the allocator's own record, frame invariant (a reader subtracts the directory header's F for lattice pages). State is 1 for
 * resident, 0 otherwise; SOLID remains a directory state (2 in the overlay).
 * The overlay copies the directory only while visible, after these records.
 *
 * The directory, runtime header and block words (layout.tail.words) are the
 * tail a reader of another buffer needs. initialize(mirrors) takes up to
 * UNIFORM_PAGE_POOL_MIRRORS buffers and a word offset in each: the seed
 * uploads the tail there and the commit kernels store every table and block
 * word they change there as well as in state, so a mirror equals the state's
 * tail after every commit with nothing copied.
 *
 * work (UniformPagePool.work; a requesting consumer binds it read_write,
 * array<atomic<u32>>; nothing else of it is a consumer's):
 *   [0,16)   header (UNIFORM_PAGE_POOL_WORK): 0 requests appended this step
 *            (may pass the list capacity), 1 host commands uploaded, 2 sticky
 *            fatal bits, 3 the bit raised first (the lock on words 4..6 and
 *            9), 4..6 its page, 7 blocks popped by the commit under way, 8
 *            ring head, 9 the first bit's detail, 10 pinned pages, 11..15
 *            reserved
 *   host     4 words per host command (x,y,z,0), `hostCapacity` of them
 *   pins     4 words per pinned page (x,y,z,0), `pinCapacity` of them: written
 *            by pin() at scene build and asked for by every commit
 *   list     4 words per distinct request (x,y,z,0), `requestCapacity` of them
 *   keep     one word per block: non-zero when upKeep marked it this step
 *   ring     layout.ringCapacity words (the power of two that holds the
 *            slots), a queue of blocks: [head, head+free) are free and the
 *            `pending` after them are quarantined
 *   claims   3 words per table entry: the x, y and z wrap tags
 *            ((q>>b)<<1 | 1) of the page asked for there this step, 0 unasked
 * Word offsets: uniformPagePoolLayout(options).work.
 *
 * A step: consumer kernels call upRequest / upKeep (uniformPagePoolWGSL), the
 * host queues request(x,y,z), then encodeCommit, which asks for the host
 * commands and the pinned pages itself. A request toward a SOLID
 * entry does nothing; one for a resident page keeps it. Otherwise each axis
 * tag is compare-and-swapped into the entry's claim words: the invocation
 * that stores the x tag appends the page to the list, any other with the
 * same tags is a duplicate, and one that meets a different tag found an
 * alias. So a listed request owns its table entry until the commit ends. The
 * commit pops one block per listed request off the ring head (first-in
 * first-out; which request gets which block is a race), ages the residents
 * it did not just claim (kept: idle 0, else idle+1), retires those at
 * `idleSteps` by clearing their entry and block word and queuing them behind
 * the free blocks, and rebuilds the resident list. A block retired by commit
 * n is therefore not claimable in commit n, and only once every older free
 * block is gone from commit n+1. Claims precede retirement, so a request
 * that aliases a page retiring in the same commit is still an alias. A
 * pinned page is asked for by every commit, so it never ages.
 *
 * A fatal leaves the pool consistent and the commits running: an exhausted,
 * aliased or overflowed request is simply not resident, and a certificate
 * over the cap asks for nothing. The bits are sticky and the first raised
 * keeps its page; uniformPagePoolFatalMessage reads them from the receipt. */
import { uniformMovingWindowDecisionWGSL } from "./uniform-moving-window";
import { UNIFORM_WINDOW_HEADER_WORDS, uniformWindowDerivedWords, uniformWindowWorldWGSL } from "./uniform-window-address";
import type { GPUFluidVolumePageSource } from "../../core/method-view-records";

const RECORD=8;
export const UNIFORM_PAGE_POOL_EDGE=32;
/** upBlockOf of a page that is not resident. */
export const UNIFORM_PAGE_POOL_NONE=0xffffffff;
/** The table words of a page without storage; any other word is its block. */
export const UNIFORM_PAGE_POOL_AIR=0,UNIFORM_PAGE_POOL_SOLID=0xffffffff;
/** The block word of a block that holds no page: free, quarantined or a sentinel. */
export const UNIFORM_PAGE_POOL_FREE=0xffffffff;
/** What a table entry is (upKind). */
export const UNIFORM_PAGE_POOL_KIND={resident:0,air:1,solid:2} as const;
/** Mirrors initialize() takes: each is one storage binding of the commit kernels. */
export const UNIFORM_PAGE_POOL_MIRRORS=4;
export const UNIFORM_PAGE_POOL_STATUS={generation:0,resident:1,claimed:2,free:3,pending:4,fatal:5,fatalPage:6,fatalGeneration:9,fatalFirst:10,requests:11,fatalDetail:12,words:16} as const;
/** Sticky fatal bits of the status. exhausted: a request found no free block.
 * alias: a request's entry is held by, or was asked for in the same step
 * for, a page with another coordinate. requests: more distinct requests in a
 * step than the list holds (requestCapacity); the surplus was dropped.
 * closureCap: a residency certificate's request (rsAsk,
 * uniform-mixed-residency.ts) was given a reach over closureCap and asked
 * for nothing. closure: it met a page seeded by the liquid whose pool page
 * was not resident before the commit (it is asked for all the same). edge:
 * a certificate reached past the frame (window pages 0..table-1 an axis)
 * toward a page that is not SOLID: the liquid has reached the edge of the
 * world's frame, and the page is not asked for. */
export const UNIFORM_PAGE_POOL_FATAL={exhausted:1,alias:2,requests:4,closureCap:8,closure:16,edge:32} as const;
/** Record states; sentinel: block 0 and the last block, which never change state. */
export const UNIFORM_PAGE_POOL_BLOCK={free:0,resident:1,pending:2,sentinel:3} as const;
export const UNIFORM_PAGE_POOL_RECORD={page:0,state:3,generation:4,idle:5,words:RECORD} as const;
export const UNIFORM_PAGE_POOL_WORK={requests:0,host:1,flags:2,first:3,firstPage:4,pops:7,head:8,detail:9,pins:10,words:16} as const;
/** The published record (file header): header words and the offset of the flags. */
export const UNIFORM_PAGE_POOL_PUBLISHED={edge:0,blocks:1,table:2,directory:3,resident:4,stride:5,version:6,frame:8,low:20,high:24,records:32,recordWords:4} as const;

/** The scene's bounding box in pages: its low corner and its extent. */
export interface UniformPagePoolBox {readonly origin:readonly [number,number,number];readonly pages:readonly [number,number,number]}
export interface UniformPagePoolOptions {
 /** Page table edge in pages, a power of two up to 64 (default 64): the virtual window is table*32 cells an axis. */
 readonly table?:number;
 /** Runtime world classification: 1 closed tank, 2 open top, 3 open ground. */
 readonly worldPolicy?:number;
 /** Storage blocks, the two sentinels included (NB); any integer from 3. blocks-2 pages can be resident. */
 readonly blocks:number;
 /** A resident page is retired by the idleSteps-th commit in a row that finds it unkept (the commit that claims it does not count). */
 readonly idleSteps:number;
 /** Distinct requests a step may list (default: blocks-2). */
 readonly requestCapacity?:number;
 /** Host commands a step may queue (default 64). */
 readonly hostCapacity?:number;
 /** Pages pin() may hold (default 64). */
 readonly pinCapacity?:number;
 /** The largest reach a residency certificate's request accepts (UP_CLOSURE_CAP), in tiles (default 16). */
 readonly closureCap?:number;
 /** The scene's box: what a box seed fills (uniformPagePoolBoxSeed) and a
  * source's pages are clipped to (uniformPagePoolCellBoxPages). With it the
  * pool publishes the overlay's page record; no kernel reads it. */
 readonly box?:UniformPagePoolBox;
}
export type UniformPagePoolLayout=ReturnType<typeof uniformPagePoolLayout>;

const powerOfTwo=(n:number)=>Number.isSafeInteger(n)&&n>0&&(n&(n-1))===0;
const int32=(n:number)=>(n|0)===n;
/** Sizes and word offsets of the pool's buffers and of the tail (see the file header). */
export function uniformPagePoolLayout(options:UniformPagePoolOptions){
 const table=options.table??64,blocks=options.blocks,slots=blocks-2,idleSteps=options.idleSteps,requests=options.requestCapacity??slots,host=options.hostCapacity??64,pins=options.pinCapacity??64,closureCap=options.closureCap??16;
 // A block word packs the window page at six bits an axis.
 if(!powerOfTwo(table)||table<2||table>64)throw new RangeError(`Uniform page table edge must be a power of two in 2..64: ${table}`);
 if(!Number.isSafeInteger(blocks)||blocks<3||blocks>65536)throw new RangeError(`Uniform page pool blocks must be an integer in 3..65536, two of them the sentinels: ${blocks}`);
 if(!Number.isSafeInteger(idleSteps)||idleSteps<1)throw new RangeError(`Uniform page pool idle steps must be a positive integer: ${idleSteps}`);
 if(!Number.isSafeInteger(requests)||requests<1||!Number.isSafeInteger(host)||host<1||!Number.isSafeInteger(pins)||pins<1)throw new RangeError(`Uniform page pool request capacities must be positive integers: ${requests}, ${host}, ${pins}`);
 if(!Number.isSafeInteger(closureCap)||closureCap<0)throw new RangeError(`Uniform page pool closure cap must be a non-negative integer: ${closureCap}`);
 // A box wider than the table would alias itself.
 const origin:readonly [number,number,number]=options.box?[...options.box.origin]:[0,0,0],extent:readonly [number,number,number]=options.box?[...options.box.pages]:[0,0,0];
 if(options.box&&!(origin.every(int32)&&extent.every(n=>Number.isSafeInteger(n)&&n>=1&&n<=table)))throw new RangeError(`Uniform page pool box must have a 32-bit integer origin and 1..${table} pages an axis: origin ${origin}, pages ${extent}`);
 const boxPages=extent[0]*extent[1]*extent[2];
 let ringCapacity=1;while(ringCapacity<slots)ringCapacity*=2;
 const entries=table**3,tableAt=UNIFORM_PAGE_POOL_STATUS.words,blockWords=tableAt+entries+UNIFORM_WINDOW_HEADER_WORDS,records=blockWords+blocks,resident=records+RECORD*blocks,claimed=resident+slots;
 const hostList=UNIFORM_PAGE_POOL_WORK.words,pinList=hostList+4*host,list=pinList+4*pins,keep=list+4*requests,ring=keep+blocks,claims=ring+ringCapacity;
 return {table,tableBits:Math.log2(table),entries,blocks,slots,solidBlock:blocks-1,ringCapacity,idleSteps,requests,host,pins,closureCap,
  /** The box (origin and extent in pages) and its page count; count 0: no box. */
  box:{origin,pages:extent,count:boxPages},
  state:{table:tableAt,window:tableAt+entries,blocks:blockWords,records,resident,claimed,words:claimed+slots},
  /** Word offsets inside the tail, wherever it lies: state.table in the state buffer, a mirror's offset in a mirror. */
  tail:{table:0,window:entries,blocks:entries+UNIFORM_WINDOW_HEADER_WORDS,words:entries+UNIFORM_WINDOW_HEADER_WORDS+blocks},
  worldPolicy:options.worldPolicy??0,
  work:{hostList,pins:pinList,list,keep,ring,claims,bounds:claims+3*entries,words:claims+3*entries+8},
  /** One coordinate/state record per block, independent of the seed box. */
  published:{records:UNIFORM_PAGE_POOL_PUBLISHED.records,words:UNIFORM_PAGE_POOL_PUBLISHED.records+4*blocks}} as const;
}

/** A page of a seed: its signed page coordinate and the block that holds it. */
export interface UniformPagePoolSeedPage {readonly x:number;readonly y:number;readonly z:number;readonly block:number}
/** The host copy of a seeded pool, generation 0. Read-only: the arrays are
 * what the buffers were written from. table, blocks, tail and resident are
 * views of state. */
export interface UniformPagePoolSeed {
 readonly layout:UniformPagePoolLayout;
 /** The whole state buffer. */
 readonly state:Uint32Array<ArrayBuffer>;
 /** The page table, layout.entries words. */
 readonly table:Uint32Array<ArrayBuffer>;
 /** The block words, layout.blocks of them. */
 readonly blocks:Uint32Array<ArrayBuffer>;
 /** Table then block words, layout.tail.words: what a mirror holds at its offset. */
 readonly tail:Uint32Array<ArrayBuffer>;
 /** The resident blocks, ascending. */
 readonly resident:Uint32Array<ArrayBuffer>;
 /** The free ring from its head (word 0), layout.ringCapacity words: the `free` first are the free blocks, ascending. */
 readonly ring:Uint32Array<ArrayBuffer>;
 readonly free:number;
}
/** Entry of page (x,y,z) in the table. */
export function uniformPagePoolEntry(layout:UniformPagePoolLayout,x:number,y:number,z:number):number{
 const mask=layout.table-1,bits=layout.tableBits;return (x&mask)|((y&mask)<<bits)|((z&mask)<<(2*bits));
}
/** Block word of a block that holds page (x,y,z). */
export function uniformPagePoolBlockWord(layout:UniformPagePoolLayout,x:number,y:number,z:number):number{
 const mask=layout.table-1;return (x&mask)|((y&mask)<<6)|((z&mask)<<12);
}
/** The whole pool state for a given resident set, built on the host at scene
 * build: `pages` are resident in the blocks they name (any order, any
 * blocks of 1..blocks-2, holes allowed), and `other(x,y,z)` gives
 * UNIFORM_PAGE_POOL_KIND.air or .solid for every other table entry, by its
 * window page coordinate (0..table-1 an axis). Blocks no page names are
 * free, in ascending order from the ring head. Throws on a coordinate that
 * is not a 32-bit integer, a block out of range or named twice, a page
 * listed twice, and two pages that alias one entry. */
export function uniformPagePoolSeed(layout:UniformPagePoolLayout,pages:readonly UniformPagePoolSeedPage[],other:(x:number,y:number,z:number)=>number):UniformPagePoolSeed{
 const l=layout,S=UNIFORM_PAGE_POOL_STATUS,R=UNIFORM_PAGE_POOL_RECORD,B=UNIFORM_PAGE_POOL_BLOCK,K=UNIFORM_PAGE_POOL_KIND;
 const state=new Uint32Array(l.state.words),signed=new Int32Array(state.buffer),ring=new Uint32Array(l.ringCapacity);
 const table=state.subarray(l.state.table,l.state.table+l.entries),blocks=state.subarray(l.state.blocks,l.state.blocks+l.blocks);
 const record=(block:number)=>l.state.records+RECORD*block,held=(block:number)=>{const r=record(block);return `(${signed[r]}, ${signed[r+1]}, ${signed[r+2]})`;};
 state[l.state.window+12]=l.worldPolicy;state.set(l.box.origin,l.state.window+13);state.set(l.box.pages,l.state.window+16);state.set(uniformWindowDerivedWords(l.worldPolicy,l.box.origin,l.box.pages),l.state.window+24);
 blocks.fill(UNIFORM_PAGE_POOL_FREE);state[record(0)+R.state]=state[record(l.solidBlock)+R.state]=B.sentinel;
 pages.forEach(({x,y,z,block},i)=>{
  const name=()=>`Uniform page pool seed page ${i} (${x}, ${y}, ${z})`;
  if((x|0)!==x||(y|0)!==y||(z|0)!==z)throw new RangeError(`${name()}: coordinates must be 32-bit integers`);
  if(!Number.isSafeInteger(block)||block<1||block>l.blocks-2)throw new RangeError(`${name()}: block ${block} is outside 1..${l.blocks-2}`);
  const r=record(block),entry=uniformPagePoolEntry(l,x,y,z),at=table[entry]!;
  if(state[r+R.state]===B.resident)throw new Error(`${name()}: block ${block} already holds page ${held(block)}`);
  if(at){
   const q=record(at);
   if(signed[q]===x&&signed[q+1]===y&&signed[q+2]===z)throw new Error(`${name()}: listed twice (blocks ${at} and ${block})`);
   throw new Error(`${name()}: aliases page ${held(at)} of block ${at} in a table of edge ${l.table}`);
  }
  table[entry]=block;blocks[block]=uniformPagePoolBlockWord(l,x,y,z);
  signed[r]=x;signed[r+1]=y;signed[r+2]=z;state[r+R.state]=B.resident;
 });
 let resident=0,free=0;
 for(let block=1;block<=l.blocks-2;block++){if(state[record(block)+R.state]===B.resident)state[l.state.resident+resident++]=block;else ring[free++]=block;}
 for(let z=0,entry=0;z<l.table;z++)for(let y=0;y<l.table;y++)for(let x=0;x<l.table;x++,entry++){
  if(table[entry])continue;
  const kind=other(x,y,z);
  if(kind===K.solid)table[entry]=UNIFORM_PAGE_POOL_SOLID;
  else if(kind!==K.air)throw new RangeError(`Uniform page pool seed: entry (${x}, ${y}, ${z}) must be classified air (${K.air}) or solid (${K.solid}): ${kind}`);
 }
 state[S.resident]=resident;state[S.free]=free;
 return {layout:l,state,table,blocks,tail:state.subarray(l.state.table,l.state.table+l.tail.words),resident:state.subarray(l.state.resident,l.state.resident+resident),ring,free};
}
/** A resident set inside the layout's box. */
export interface UniformPagePoolBoxSeedOptions {
 /** Which box pages are resident, by box coordinate (0..pages-1 an axis); default: all. */
 readonly resident?:(x:number,y:number,z:number)=>boolean;
 /** A number that fixes a shuffle of the blocks; omitted: the blocks are in scan order. */
 readonly permute?:number;
 /** The kind of an entry outside the box, by window page coordinate (default SOLID). */
 readonly outside?:(x:number,y:number,z:number)=>number;
}
/** The resident pages of the layout's box as seed pages, in scan order (x
 * fastest). Without `permute` their blocks are 1, 2, ... in that order, so
 * the whole box is the identity 1 + x + pages.x*(y + pages.y*z). With it
 * they are the first blocks of a shuffle of 1..blocks-2 that the number
 * fixes: out of scan order and, in a pool larger than the set, with free
 * holes among them. */
export function uniformPagePoolBoxPages(layout:UniformPagePoolLayout,options:UniformPagePoolBoxSeedOptions={}):UniformPagePoolSeedPage[]{
 const {origin,pages:[px,py,pz],count}=layout.box,order=Array.from({length:layout.slots},(_,i)=>i+1),pages:UniformPagePoolSeedPage[]=[];
 if(!count)throw new Error("Uniform page pool layout has no box");
 if(options.permute!==undefined){
  // Fisher-Yates over a mulberry32 stream.
  let s=options.permute|0;
  const next=()=>{s=(s+0x6d2b79f5)|0;let t=Math.imul(s^(s>>>15),1|s);t=(t+Math.imul(t^(t>>>7),61|t))^t;return ((t^(t>>>14))>>>0)/4294967296;};
  for(let i=order.length-1;i>0;i--){const j=Math.floor(next()*(i+1)),o=order[i]!;order[i]=order[j]!;order[j]=o;}
 }
 for(let z=0;z<pz;z++)for(let y=0;y<py;y++)for(let x=0;x<px;x++){
  if(options.resident&&!options.resident(x,y,z))continue;
  if(pages.length>=layout.slots)throw new RangeError(`Uniform page pool box seed holds more pages than the pool's ${layout.slots} blocks`);
  pages.push({x:origin[0]+x,y:origin[1]+y,z:origin[2]+z,block:order[pages.length]!});
 }
 return pages;
}
/** The seed of a resident set inside the layout's box
 * (uniformPagePoolBoxPages): the box's other pages are AIR and every entry
 * outside the box is what `outside` gives (default SOLID). */
export function uniformPagePoolBoxSeed(layout:UniformPagePoolLayout,options:UniformPagePoolBoxSeedOptions={}):UniformPagePoolSeed{
 const {origin,pages}=layout.box,mask=layout.table-1,K=UNIFORM_PAGE_POOL_KIND,outside=options.outside;
 const inside=(x:number,y:number,z:number)=>((x-origin[0])&mask)<pages[0]&&((y-origin[1])&mask)<pages[1]&&((z-origin[2])&mask)<pages[2];
 return uniformPagePoolSeed(layout,uniformPagePoolBoxPages(layout,options),(x,y,z)=>inside(x,y,z)?K.air:outside?outside(x,y,z):K.solid);
}
/** Overlay publication of the pool's block records, including signed pages. */
export function uniformPagePoolPublished(seed:UniformPagePoolSeed,layout:UniformPagePoolLayout=seed.layout):Uint32Array<ArrayBuffer>{
 const l=layout,P=UNIFORM_PAGE_POOL_PUBLISHED,record=new Uint32Array(l.published.words);
 record[P.edge]=UNIFORM_PAGE_POOL_EDGE;record[P.blocks]=l.blocks;record[P.table]=l.table;
 record[P.directory]=l.published.words;record[P.resident]=seed.resident.length;
 record[P.stride]=4;record[P.version]=2;
 const signed=new Int32Array(record.buffer);signed.set([2147483647,2147483647,2147483647],P.low);signed.set([-2147483647,-2147483647,-2147483647],P.high);
 for(const b of seed.resident){
  const r=l.state.records+RECORD*b;
  for(let a=0;a<3;a++){const q=seed.state[r+a]!|0;signed[P.low+a]=Math.min(signed[P.low+a]!,q);signed[P.high+a]=Math.max(signed[P.high+a]!,q+1);}
 }
 for(let b=0;b<l.blocks;b++){
  const r=l.state.records+RECORD*b,o=P.records+4*b;
  record.set(seed.state.subarray(r,r+3),o);
  record[o+3]=seed.state[r+3]===UNIFORM_PAGE_POOL_BLOCK.resident?1:0;
 }
 return record;
}
/** The pages that overlap the cell box [low, high) grown by `pad` cells an
 * axis, clipped to the layout's box when it has one: what a source box
 * pins. Cells are window cells (page*32 + local). */
export function uniformPagePoolCellBoxPages(layout:UniformPagePoolLayout,low:readonly [number,number,number],high:readonly [number,number,number],pad=0):{x:number;y:number;z:number}[]{
 return cellBoxPages(layout.box,low,high,pad);
}
function cellBoxPages(box:UniformPagePoolLayout["box"],low:readonly number[],high:readonly number[],pad:number):{x:number;y:number;z:number}[]{
 const {origin,pages:extent,count}=box,E=UNIFORM_PAGE_POOL_EDGE,pages:{x:number;y:number;z:number}[]=[];
 const a=[0,1,2].map(k=>{const q=Math.floor((low[k]!-pad)/E);return count?Math.max(q,origin[k]!):q;}),b=[0,1,2].map(k=>{const q=Math.floor((high[k]!+pad-1)/E);return count?Math.min(q,origin[k]!+extent[k]!-1):q;});
 for(let z=a[2]!;z<=b[2]!;z++)for(let y=a[1]!;y<=b[1]!;y++)for(let x=a[0]!;x<=b[0]!;x++)pages.push({x,y,z});
 return pages;
}
/** Cells of margin the host seed keeps around the initial liquid, so the
 * first census finds its certificate stored: four tiles to the far corner
 * of a 4h tile the census calls near (a corner within 16 h of the surface)
 * and that tile's own width, the certificate's closure at rest (its margin
 * of five tiles and one of travel), and three tiles of rounding to its
 * 16-cell pages. A page the first commit asks for past it is claimed like
 * any other. */
export const UNIFORM_PAGE_POOL_SEED_MARGIN=56;
/** Cells a pinned source box, or a requested drop's, is grown by: one tile.
 * The census marks a source's tiles band where they are stored, so the
 * certificate asks for their closure itself. */
export const UNIFORM_PAGE_POOL_PIN_MARGIN=4;
/** The scheduled drop whose pages a step asks the pool for one step early:
 * its index in `schedule`, or -1. The step ends at `end_s` after `dt_s`;
 * `longest_s` is the longest step the solver takes. The drop is the earliest
 * one after this step (its own takes those up to end_s, with the slack of
 * the solver's due test) and within a longest step of its end. The step
 * that adds a drop takes no other, so the step before a drop's always
 * names it: its pages are stored by that commit, and the census of the
 * drop's own step, which runs before its commit, finds the drop's tiles
 * stored and marks them band (UNIFORM_PAGE_POOL_PIN_MARGIN). A drop of the
 * first step has no step before it: the kick's commit stores its pages
 * ahead of the first frame's census. Allocates nothing. */
export function uniformPagePoolDropAhead(schedule:readonly {readonly time_s:number}[],end_s:number,dt_s:number,longest_s:number):number{
 const slack=1e-6,after=end_s+slack*dt_s,until=end_s+(1+slack)*longest_s;
 let ahead=-1,first=Infinity;
 for(let index=0;index<schedule.length;index++){
  const time_s=schedule[index]!.time_s;
  if(time_s>after&&time_s<=until&&time_s<first){first=time_s;ahead=index;}
 }
 return ahead;
}
/** A box of lattice cells, [low, high). */
export interface UniformPagePoolCellBox {readonly low:readonly [number,number,number];readonly high:readonly [number,number,number]}
/** A scene's stored set at build (uniformPagePoolResidency). */
export interface UniformPagePoolResidency {
 /** The stored set follows the census: pages are claimed as its certificate
  * reaches them and retired when idle. False: every seeded page is pinned,
  * so no commit claims or retires one. */
 readonly live:boolean;
 /** The box pages seeded AIR, by box page coordinate; every other is resident. */
 readonly air:[number,number,number][];
 /** The pinned pages (UniformPagePool.pin), by box page coordinate. */
 readonly pins:{x:number;y:number;z:number}[];
}
/** The one rule for a scene's stored set, from its box in cells, its page
 * pool record (SceneDescription.fluid.pagePool), its sources' cell boxes
 * (what writes liquid every step: the inflow plug) and whether a census
 * runs every step (follows).
 * Live: a census. The pinned pages are then the sources' alone (grown by
 * UNIFORM_PAGE_POOL_PIN_MARGIN): every other page retires when no request
 * keeps it and is claimed again when one arrives. A drop is not a source:
 * its pages are requested around its own step (uniformPagePoolDropAhead)
 * and retire like any other once its liquid has left them. Otherwise every
 * resident page is pinned.
 * Resident at build: the box less record.air where the record lists one;
 * otherwise, live, the pages within UNIFORM_PAGE_POOL_SEED_MARGIN cells of a
 * wet cell of the initial liquid and the pinned; otherwise every box page.
 * liquid: the initial cell volumes, x fastest, asked for only by the
 * wet-cell case. origin: the box's low page in the window. The sources'
 * cell boxes and the pins are the window's, the AIR pages the box's. */
export function uniformPagePoolResidency(cells:readonly [number,number,number],record:{readonly blocks?:number;readonly air?:readonly (readonly [number,number,number])[]}|undefined,sources:readonly UniformPagePoolCellBox[],follows:boolean,liquid:()=>ArrayLike<number>,origin:readonly [number,number,number]=[0,0,0]):UniformPagePoolResidency{
 const E=UNIFORM_PAGE_POOL_EDGE,pages:[number,number,number]=[Math.ceil(cells[0]/E),Math.ceil(cells[1]/E),Math.ceil(cells[2]/E)],[px,py,pz]=pages,count=px*py*pz;
 const live=follows,resident=new Uint8Array(count).fill(1),pinned=new Uint8Array(count);
 const coordinates=(flags:Uint8Array,set:number)=>{const out:[number,number,number][]=[];for(let z=0,i=0;z<pz;z++)for(let y=0;y<py;y++)for(let x=0;x<px;x++,i++)if(flags[i]===set)out.push([x,y,z]);return out;};
 if(live){
  for(const {x,y,z} of uniformPagePoolSourcePages(cells,sources,origin))pinned[x-origin[0]+px*(y-origin[1]+py*(z-origin[2]))]=1;
  if(!record?.air){resident.set(wetPages(cells,pages,liquid(),UNIFORM_PAGE_POOL_SEED_MARGIN));for(let i=0;i<count;i++)if(pinned[i])resident[i]=1;}
 }
 for(const [x,y,z] of record?.air??[]){
  if(!(Number.isInteger(x)&&Number.isInteger(y)&&Number.isInteger(z)&&x>=0&&y>=0&&z>=0&&x<px&&y<py&&z<pz))throw new RangeError(`Uniform page pool: AIR page (${x}, ${y}, ${z}) is outside the box of ${px}x${py}x${pz} pages`);
  resident[x+px*(y+py*z)]=0;
 }
 if(!live)pinned.set(resident);
 return {live,air:coordinates(resident,0),pins:coordinates(pinned,1).map(([x,y,z])=>({x:x+origin[0],y:y+origin[1],z:z+origin[2]}))};
}
/** The box pages a live pool pins for its sources' cell boxes, each grown by
 * UNIFORM_PAGE_POOL_PIN_MARGIN: every page once, in the window (the box's
 * low page at origin). */
export function uniformPagePoolSourcePages(cells:readonly [number,number,number],sources:readonly UniformPagePoolCellBox[],origin:readonly [number,number,number]=[0,0,0]):{x:number;y:number;z:number}[]{
 const E=UNIFORM_PAGE_POOL_EDGE,pages:[number,number,number]=[Math.ceil(cells[0]/E),Math.ceil(cells[1]/E),Math.ceil(cells[2]/E)],[px,py,pz]=pages,box={origin,pages,count:px*py*pz},seen=new Uint8Array(box.count);
 return sources.flatMap(s=>cellBoxPages(box,s.low,s.high,UNIFORM_PAGE_POOL_PIN_MARGIN)).filter(({x,y,z})=>{const i=x-origin[0]+px*(y-origin[1]+py*(z-origin[2])),first=!seen[i];seen[i]=1;return first;});
}
/** One flag per box page: a wet cell (volume above zero) lies within `margin`
 * cells of it on every axis. Each page keeps the box of its wet cells, so a
 * page is tested against boxes, not cells: never fewer pages than the cells give. */
function wetPages(cells:readonly [number,number,number],pages:readonly [number,number,number],volume:ArrayLike<number>,margin:number):Uint8Array{
 const E=UNIFORM_PAGE_POOL_EDGE,[nx,ny,nz]=cells,[px,py,pz]=pages,count=px*py*pz,low=new Int32Array(3*count).fill(0x7fffffff),high=new Int32Array(3*count).fill(-1),flags=new Uint8Array(count);
 if(volume.length!==nx*ny*nz)throw new RangeError(`Uniform page pool: the initial liquid holds ${volume.length} cells, not the lattice's ${nx*ny*nz}`);
 for(let z=0,i=0;z<nz;z++)for(let y=0;y<ny;y++)for(let x=0;x<nx;x++,i++){
  if(!(volume[i]!>0))continue;
  const p=3*(Math.floor(x/E)+px*(Math.floor(y/E)+py*Math.floor(z/E)));
  if(x<low[p]!)low[p]=x;if(y<low[p+1]!)low[p+1]=y;if(z<low[p+2]!)low[p+2]=z;
  if(x>high[p]!)high[p]=x;if(y>high[p+1]!)high[p+1]=y;if(z>high[p+2]!)high[p+2]=z;
 }
 const reach=Math.ceil(margin/E),gap=(p:number,k:number,a:number)=>Math.max(low[p+k]!-(a+E-1),a-high[p+k]!,0);
 for(let z=0,i=0;z<pz;z++)for(let y=0;y<py;y++)for(let x=0;x<px;x++,i++){
  near:for(let sz=Math.max(z-reach,0);sz<=Math.min(z+reach,pz-1);sz++)for(let sy=Math.max(y-reach,0);sy<=Math.min(y+reach,py-1);sy++)for(let sx=Math.max(x-reach,0);sx<=Math.min(x+reach,px-1);sx++){
   const p=3*(sx+px*(sy+py*sz));
   if(high[p]!>=0&&Math.max(gap(p,0,E*x),gap(p,1,E*y),gap(p,2,E*z))<=margin){flags[i]=1;break near;}
  }
 }
 return flags;
}
/** The message of the pool's first fatal, from the status words at word
 * `offset` of `status` (a mapped receipt of encodeReceipt); null, with
 * nothing allocated, while no bit is set. */
export function uniformPagePoolFatalMessage(layout:UniformPagePoolLayout,status:ArrayLike<number>,offset=0):string|null{
 const S=UNIFORM_PAGE_POOL_STATUS,F=UNIFORM_PAGE_POOL_FATAL;
 if(!status[offset+S.fatal])return null;
 const first=status[offset+S.fatalFirst]!,commit=status[offset+S.fatalGeneration]!,detail=status[offset+S.fatalDetail]!,at=offset+S.fatalPage;
 const page=`page (${status[at]!|0}, ${status[at+1]!|0}, ${status[at+2]!|0})`;
 switch(first){
  case F.exhausted:return `Page pool exhausted at commit ${commit}: ${page} requested with ${detail} of ${layout.slots} blocks resident and the other ${layout.slots-detail} claimed by the same commit. Raise pagePool.memory_MiB or pagePool.blocks (about 10 MiB a block, UNIFORM_PAGE_BLOCK_MiB) or shrink the scene.`;
  case F.requests:return `More than claimsPerStep (${layout.requests}) pages were requested in commit ${commit} (first dropped: ${page}).`;
  case F.alias:return `Two pages name one page table entry (${page}, commit ${commit}): the resident set is wider than the table's ${layout.table} pages, or an allocator bug.`;
  case F.closureCap:return `Departure travel gives a closure of ${detail} tiles, over the cap of ${layout.closureCap} (commit ${commit}, ${page}): a sample is moving faster than the residency certificate covers.`;
  case F.closure:return `Liquid or band tile found in ${page}, which was not resident at the start of commit ${commit}.`;
  case F.edge:return `Liquid exceeds the moving fluid frame at ${page} (commit ${commit}): residents and requests must fit ${layout.table-4} pages per axis with the window margin; an invalid forced shift or signed coordinate overflow also raises this fatal.`;
  default:return `Page pool fatal bits ${status[offset+S.fatal]} (first ${first}) at commit ${commit}, ${page}.`;
 }
}

const READERS=/* wgsl */`
fn upEntry(q:vec3i)->u32{let m=vec3u(q)&vec3u(UP_TABLE-1u);return m.x|(m.y<<UP_TABLE_BITS)|(m.z<<(2u*UP_TABLE_BITS));}
// The word at q's entry: UP_AIR_WORD, UP_SOLID_WORD or the block resident there (q's own unless a page a table apart holds the entry).
fn upWord(q:vec3i)->u32{
 let raw=upState[UP_AT_TABLE+upEntry(q)];if(upPolicy()==0u){return raw;}
 if(upWorldSolid(q)){return UP_SOLID_WORD;}
 return select(raw,UP_AIR_WORD,raw==UP_SOLID_WORD);
}
// A block with a page slot: neither sentinel.
fn upStored(block:u32)->bool{return block-1u<UP_SOLID_BLOCK-1u;}
// UP_RESIDENT, UP_AIR or UP_SOLID: what q's entry is, without the coordinate check.
fn upKind(q:vec3i)->u32{let word=upWord(q);return select(select(UP_RESIDENT,UP_SOLID,word==UP_SOLID_WORD),UP_AIR,word==UP_AIR_WORD);}
fn upPageOf(block:u32)->vec3i{let r=UP_AT_RECORDS+8u*block;return vec3i(bitcast<i32>(upState[r]),bitcast<i32>(upState[r+1u]),bitcast<i32>(upState[r+2u]));}
// The block a read of q resolves to, without the coordinate check: block 0 for AIR, UP_SOLID_BLOCK for SOLID.
fn upBlockWrapped(q:vec3i)->u32{return min(upWord(q),UP_SOLID_BLOCK);}
// The block that holds page q itself, else UP_NONE.
fn upBlockOf(q:vec3i)->u32{let block=upWord(q);if(!upStored(block)){return UP_NONE;}return select(UP_NONE,block,all(upPageOf(block)==q));}
// The block's word: the window page of the page it holds (upWindowPage), or UP_FREE.
fn upBlockWord(block:u32)->u32{return upState[UP_AT_BLOCKS+block];}
fn upWindowPage(word:u32)->vec3i{return vec3i((vec3u(word&63u,(word>>6u)&63u,(word>>12u)&63u)-bitcast<vec3u>(upFramePage()))&vec3u(63u));}
fn upGeneration()->u32{return upState[0u];}
fn upResidentCount()->u32{return upState[1u];}
fn upResidentBlock(i:u32)->u32{return upState[UP_AT_RESIDENT+i];}
// The blocks the last commit claimed, [0, upNewCount()), and the page each
// now holds: what a page-initialise launch strides. Seeded residents are not
// in the new list (it is empty until the first commit claims).
fn upNewCount()->u32{return upState[2u];}
fn upNewBlock(i:u32)->u32{return upState[UP_AT_NEW+i];}
fn upNewPage(i:u32)->vec3i{return upPageOf(upNewBlock(i))-upFramePage();}
fn upFatal()->u32{return upState[5u];}
`;
const REQUESTERS=/* wgsl */`
// The resident block is in use this step: the next commit resets its idle count. Nothing for a sentinel or UP_NONE.
fn upKeep(block:u32){if(block<UP_BLOCKS){atomicStore(&upWork[UP_WORK_KEEP+block],1u);}}
// Raise a sticky fatal bit (UP_FATAL_*); the first raised keeps its page and its detail.
fn upFailWith(bit:u32,q:vec3i,detail:u32){
 atomicOr(&upWork[2u],bit);
 var seen=atomicLoad(&upWork[3u]);
 while(seen==0u){
  let r=atomicCompareExchangeWeak(&upWork[3u],0u,bit);
  if(r.exchanged){atomicStore(&upWork[4u],bitcast<u32>(q.x));atomicStore(&upWork[5u],bitcast<u32>(q.y));atomicStore(&upWork[6u],bitcast<u32>(q.z));atomicStore(&upWork[UP_WORK_DETAIL],detail);return;}
  seen=r.old_value;
 }
}
fn upFail(bit:u32,q:vec3i){upFailWith(bit,q,0u);}
// 0: the claim word already held the tag; 1: this call stored it; 2: it holds another.
fn upClaimWord(at:u32,tag:u32)->u32{
 var seen=atomicLoad(&upWork[at]);
 while(seen==0u){let r=atomicCompareExchangeWeak(&upWork[at],0u,tag);if(r.exchanged){return 1u;}seen=r.old_value;}
 return select(2u,0u,seen==tag);
}
// Ask for page q to be resident after the next commit, from any number of
// invocations at once: nothing toward a SOLID entry, a resident page is
// kept, an absent one listed once.
fn upInclude(q:vec3i){
 if(upPolicy()==0u){return;}
 let key=bitcast<vec3u>(q)^vec3u(0x80000000u);
 for(var a=0u;a<3u;a++){atomicMin(&upWork[UP_WORK_BOUNDS+a],key[a]);atomicMax(&upWork[UP_WORK_BOUNDS+3u+a],key[a]);}
 atomicOr(&upWork[UP_WORK_BOUNDS+6u],1u);
}
fn upRequest(q:vec3i){
 let e=upEntry(q);let held=upWord(q);
 if(held==UP_SOLID_WORD){return;}
 upInclude(q);
 if(held!=UP_AIR_WORD&&all(upPageOf(held)==q)){upKeep(held);return;}
 let tag=(vec3u(q>>vec3u(UP_TABLE_BITS))<<vec3u(1u))|vec3u(1u);
 let c=UP_WORK_CLAIMS+3u*e;
 let x=upClaimWord(c,tag.x);if(x==2u){upFail(select(UP_FATAL_ALIAS,UP_FATAL_EDGE,upPolicy()!=0u),q);return;}
 let y=upClaimWord(c+1u,tag.y);let z=upClaimWord(c+2u,tag.z);
 if(y==2u||z==2u){upFail(select(UP_FATAL_ALIAS,UP_FATAL_EDGE,upPolicy()!=0u),q);}
 if(x!=1u){return;}
 let i=atomicAdd(&upWork[0u],1u);
 if(i>=UP_REQUESTS){upFail(UP_FATAL_REQUESTS,q);return;}
 let a=UP_WORK_LIST+4u*i;
 atomicStore(&upWork[a],bitcast<u32>(q.x));atomicStore(&upWork[a+1u],bitcast<u32>(q.y));atomicStore(&upWork[a+2u],bitcast<u32>(q.z));
}
`;
function source(l:UniformPagePoolLayout,group:number,state:number,work:number|undefined,access:"read"|"read_write"):string{
 const F=UNIFORM_PAGE_POOL_FATAL,K=UNIFORM_PAGE_POOL_KIND;
 return /* wgsl */`
const UP_EDGE:u32=${UNIFORM_PAGE_POOL_EDGE}u;const UP_BLOCKS:u32=${l.blocks}u;const UP_SOLID_BLOCK:u32=${l.solidBlock}u;const UP_TABLE:u32=${l.table}u;const UP_NONE:u32=0xffffffffu;
const UP_AIR_WORD:u32=0u;const UP_SOLID_WORD:u32=0xffffffffu;const UP_FREE:u32=0xffffffffu;
const UP_RESIDENT:u32=${K.resident}u;const UP_AIR:u32=${K.air}u;const UP_SOLID:u32=${K.solid}u;
const UP_TABLE_BITS:u32=${l.tableBits}u;const UP_REQUESTS:u32=${l.requests}u;const UP_CLOSURE_CAP:u32=${l.closureCap}u;
const UP_AT_WINDOW:u32=${l.state.window}u;const UP_WORK_BOUNDS:u32=${l.work.bounds}u;
const UP_AT_TABLE:u32=${l.state.table}u;const UP_AT_BLOCKS:u32=${l.state.blocks}u;const UP_AT_RECORDS:u32=${l.state.records}u;const UP_AT_RESIDENT:u32=${l.state.resident}u;const UP_AT_NEW:u32=${l.state.claimed}u;
const UP_WORK_LIST:u32=${l.work.list}u;const UP_WORK_KEEP:u32=${l.work.keep}u;const UP_WORK_CLAIMS:u32=${l.work.claims}u;const UP_WORK_DETAIL:u32=${UNIFORM_PAGE_POOL_WORK.detail}u;
const UP_FATAL_EXHAUSTED:u32=${F.exhausted}u;const UP_FATAL_ALIAS:u32=${F.alias}u;const UP_FATAL_REQUESTS:u32=${F.requests}u;const UP_FATAL_CLOSURE_CAP:u32=${F.closureCap}u;const UP_FATAL_CLOSURE:u32=${F.closure}u;const UP_FATAL_EDGE:u32=${F.edge}u;
@group(${group}) @binding(${state}) var<storage,${access}> upState:array<u32>;
${uniformWindowWorldWGSL(i=>`upState[UP_AT_WINDOW+${i}]`,"up")}
${READERS}${work===undefined?"":`@group(${group}) @binding(${work}) var<storage,read_write> upWork:array<atomic<u32>>;
${REQUESTERS}`}`;
}
/** What a consumer includes to read the pool: the state buffer (read-only)
 * at @group(group) @binding(state), with constants UP_EDGE, UP_BLOCKS,
 * UP_SOLID_BLOCK, UP_TABLE, UP_NONE, UP_FREE, UP_CLOSURE_CAP, the table words UP_AIR_WORD
 * and UP_SOLID_WORD and the kinds UP_RESIDENT, UP_AIR and UP_SOLID, and
 * readers upKind, upWord, upBlockOf, upBlockWrapped, upStored, upPageOf,
 * upBlockWord / upWindowPage, upGeneration, upResidentCount /
 * upResidentBlock, upNewCount / upNewBlock / upNewPage and upFatal. With `work`, the work buffer (read_write) at
 * that binding of the same group, the requesters upRequest and upKeep, and
 * upFail / upFailWith with the bits UP_FATAL_*. */
export function uniformPagePoolWGSL(layout:UniformPagePoolLayout,group:number,state:number,work?:number):string{
 return source(layout,group,state,work,"read");
}
/** A buffer that carries the tail (layout.tail) at word `offset`: the commit
 * kernels bind it writable, the seed is uploaded to it. */
export interface UniformPagePoolMirror {readonly buffer:GPUBuffer;readonly offset:number}

/** The pool's buffers and its commit. Construct (every entry AIR, every
 * block free), seed() a resident set, pin() the pages that stay, await
 * initialize, then per step: consumer launches that request and keep,
 * request() for host commands, encodeCommit, consumer launches that read the
 * new generation. */
export class UniformPagePool {
 readonly layout:UniformPagePoolLayout;
 /** Status, page table, block words, block records, resident and new lists (file header). */
 readonly state:GPUBuffer;
 /** Requests, keeps, the free ring and claim words (file header). */
 readonly work:GPUBuffer;
 /** The overlay's page record, a word a block (file header); absent without a box. */
 readonly published?:GPUBuffer;
 /** `published` as the solver hands it out (GPUFluid.volumePageSource). */
 readonly volumePageSource?:GPUFluidVolumePageSource;
 readonly allocatedBytes:number;
 /** Host commands queued for the next commit, four words each. */
 private readonly host:Int32Array<ArrayBuffer>;
 private hostCount=0;
 /** The pinned pages, four words each: the host copy of the work buffer's pin list. */
 private readonly pins:Int32Array<ArrayBuffer>;
 private pinCount=0;
 /** One word for the host command count (the queue copies it at the call). */
 private readonly word=new Uint32Array(1);
 private readonly hostGroups:number;
 private readonly claimGroups:number;
 private readonly publishGroups:number;
 private current:UniformPagePoolSeed;
 private mirrors:readonly UniformPagePoolMirror[]=[];
 private started=false;
 private windowPipeline?:GPUComputePipeline;
 private hostPipeline?:GPUComputePipeline;
 private claimPipeline?:GPUComputePipeline;
 private slotsPipeline?:GPUComputePipeline;
 private publishPipeline?:GPUComputePipeline;
 /** State read-only (the host command launch is a consumer); state and
  * mirrors writable (the commit); state read-only beside the published record. */
 private reader?:GPUBindGroup;
 private writer?:GPUBindGroup;
 private publisher?:GPUBindGroup;
 private static readonly PASS:GPUComputePassDescriptor={label:"Uniform page pool commit"};
 constructor(private readonly device:GPUDevice,options:UniformPagePoolOptions){
  const l=this.layout=uniformPagePoolLayout(options);
  const most=Math.min(device.limits.maxStorageBufferBindingSize,device.limits.maxBufferSize);
  if(4*Math.max(l.state.words,l.work.words)>most)throw new RangeError(`Uniform page pool of table ${l.table} and ${l.blocks} blocks exceeds the device buffer limit`);
  const usage=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST;
  this.state=device.createBuffer({label:"Uniform page pool state",size:4*l.state.words,usage});
  this.work=device.createBuffer({label:"Uniform page pool work",size:4*l.work.words,usage});
  if(l.published.words){
   this.published=device.createBuffer({label:"Uniform page pool published pages",size:4*l.published.words,usage});
   this.volumePageSource={records:{buffer:this.published},directory:{buffer:this.state,offset:4*l.state.table,size:4*(l.entries+UNIFORM_WINDOW_HEADER_WORDS)}};
  }
  this.allocatedBytes=this.state.size+this.work.size+(this.published?.size??0);
  this.host=new Int32Array(4*l.host);this.pins=new Int32Array(4*l.pins);
  // Publication visits blocks; the address-window directory is copied only by a visible overlay.
  this.hostGroups=Math.min(Math.ceil(Math.max(l.host,l.pins)/64),16);this.claimGroups=Math.min(Math.ceil(l.requests/64),16);this.publishGroups=Math.ceil(l.blocks/64);
  this.current=uniformPagePoolSeed(l,[],()=>UNIFORM_PAGE_POOL_KIND.air);this.upload();
 }
 /** The host copy of the pool at generation 0: the last seed, or the empty pool. */
 get seeded():UniformPagePoolSeed{return this.current;}
 /** Replace the whole pool with a seed built for this layout
  * (uniformPagePoolSeed): both buffers, the published record and the tail
  * of every mirror, at generation 0 with no fatal, request, keep or host
  * command pending. The pinned pages stay. Scene build only: it allocates. */
 seed(seed:UniformPagePoolSeed):void{
  const l=this.layout,s=seed.layout;
  if(s.table!==l.table||s.blocks!==l.blocks||seed.state.length!==l.state.words||seed.ring.length!==l.ringCapacity)throw new Error(`Uniform page pool seed is of table ${s.table} and ${s.blocks} blocks; the pool is of table ${l.table} and ${l.blocks} blocks`);
  this.current=seed;this.hostCount=0;this.upload();
 }
 private upload():void{
  const l=this.layout,seed=this.current,queue=this.device.queue,work=new Uint32Array(l.work.words);
  work.fill(0xffffffff,l.work.bounds,l.work.bounds+3);
  work.set(seed.ring,l.work.ring);work[UNIFORM_PAGE_POOL_WORK.pins]=this.pinCount;work.set(new Uint32Array(this.pins.buffer,0,4*this.pinCount),l.work.pins);
  queue.writeBuffer(this.state,0,seed.state);queue.writeBuffer(this.work,0,work);
  if(this.published)queue.writeBuffer(this.published,0,uniformPagePoolPublished(seed,l));
  for(const m of this.mirrors)queue.writeBuffer(m.buffer,4*m.offset,seed.tail);
 }
 /** Fix the pinned pages (source boxes): every commit from now on
  * asks for each, exactly as upRequest asks, so a pinned page is claimed by
  * the next commit unless resident already and never retires. Replaces the
  * list; the pages are written to the work buffer here, once, and the
  * commits read them there. Scene build only. */
 pin(pages:readonly {readonly x:number;readonly y:number;readonly z:number}[]):void{
  const l=this.layout,queue=this.device.queue;
  if(pages.length>l.pins)throw new Error(`Uniform page pool pins exceed ${l.pins}: ${pages.length}`);
  for(const {x,y,z} of pages)if(!int32(x)||!int32(y)||!int32(z))throw new RangeError(`Uniform page coordinate must be 32-bit integers: ${x}, ${y}, ${z}`);
  pages.forEach(({x,y,z},i)=>{this.pins[4*i]=x;this.pins[4*i+1]=y;this.pins[4*i+2]=z;});
  this.pinCount=this.word[0]=pages.length;
  queue.writeBuffer(this.work,4*UNIFORM_PAGE_POOL_WORK.pins,this.word);
  if(pages.length)queue.writeBuffer(this.work,4*l.work.pins,this.pins,0,4*pages.length);
 }
 /** Pages pin() holds. */
 get pinned():number{return this.pinCount;}
 /** Compile the commit, every pipeline it will ever launch. Each mirror's
  * tail is written from the seed here and kept equal to the state's by the
  * commit kernels from then on; a mirror needs STORAGE and COPY_DST usage
  * and costs the commit one storage binding. */
 async initialize(mirrors:readonly UniformPagePoolMirror[]=[],frameParams?:GPUBuffer):Promise<void>{
  const device=this.device,l=this.layout,W=UNIFORM_PAGE_POOL_WORK,S=UNIFORM_PAGE_POOL_STATUS,R=UNIFORM_PAGE_POOL_RECORD,B=UNIFORM_PAGE_POOL_BLOCK;
  if(this.started)throw new Error("Uniform page pool is already initialized");
  const most=Math.min(UNIFORM_PAGE_POOL_MIRRORS,device.limits.maxStorageBuffersPerShaderStage-2-(frameParams?1:0)),needed=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST;
  if(mirrors.length>most)throw new RangeError(`Uniform page pool takes ${most} mirrors at most: ${mirrors.length}`);
  mirrors.forEach((m,i)=>{
   if(!Number.isSafeInteger(m.offset)||m.offset<0||4*(m.offset+l.tail.words)>m.buffer.size)throw new RangeError(`Uniform page pool mirror ${i}: ${l.tail.words} words at word ${m.offset} do not fit its buffer of ${m.buffer.size} bytes`);
   if((m.buffer.usage&needed)!==needed)throw new Error(`Uniform page pool mirror ${i}: its buffer needs STORAGE and COPY_DST usage`);
   if(m.buffer===this.state||m.buffer===this.work||mirrors.findIndex(o=>o.buffer===m.buffer)!==i)throw new Error(`Uniform page pool mirror ${i}: its buffer is bound to the commit already`);
  });
  this.started=true;this.mirrors=mirrors.map(({buffer,offset})=>({buffer,offset}));
  for(const m of this.mirrors)device.queue.writeBuffer(m.buffer,4*m.offset,this.current.tail);
  const puts=(at:number)=>this.mirrors.map((m,i)=>`upMirror${i}[${m.offset+at}u+i]=word;`).join("");
  const compile=async(label:string,code:string)=>{
   const module=device.createShaderModule({label,code});
   const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");
   if(errors.length)throw new Error(`${label}\n${errors.map(m=>`${m.lineNum}: ${m.message}`).join("\n")}`);
   return module;
  };
  const windowBody=uniformMovingWindowDecisionWGSL.slice(uniformMovingWindowDecisionWGSL.indexOf("@compute"))
   .replaceAll("mwState[","upState[UP_AT_WINDOW+").replaceAll("mwBounds[","upWork[UP_WORK_BOUNDS+")
   .replaceAll("mwForce.boundary","upState[UP_AT_WINDOW+20u]")
   .replaceAll("mwForce.x","bitcast<i32>(upState[UP_AT_WINDOW+21u])")
   .replaceAll("mwForce.y","bitcast<i32>(upState[UP_AT_WINDOW+22u])")
   .replaceAll("mwForce.z","bitcast<i32>(upState[UP_AT_WINDOW+23u])")
   .replace("upState[UP_AT_WINDOW+20u]==upState[UP_AT_WINDOW+8u]+1u",frameParams?"upState[UP_AT_WINDOW+20u]==u32(bitcast<f32>(upParams[53u]))":"upState[UP_AT_WINDOW+20u]==upState[UP_AT_WINDOW+8u]+1u")
   .replace("@compute @workgroup_size(1) fn mwDecide()", "fn upDecide()")
   .replace("let old=", "if(upPolicy()==0u){return;}let old=")
   // A finite tank already has walls on every side; only the probe moves it.
   .replace("if(forced){", "if(upPolicy()==1u||(a==1u&&upPolicy()==3u&&lo==((upState[UP_AT_WINDOW+14u])^0x80000000u)&&hi>=old[a]&&hi-old[a]<62u)){next[a]=old[a];}if(forced){");
  const [hostModule,publishModule,commitModule]=await Promise.all([compile("Uniform page pool host commands",/* wgsl */`${source(l,0,0,1,"read")}
fn upAsk(a:u32){upRequest(vec3i(bitcast<i32>(atomicLoad(&upWork[a])),bitcast<i32>(atomicLoad(&upWork[a+1u])),bitcast<i32>(atomicLoad(&upWork[a+2u]))));}
// The host commands and the pinned pages ask as any consumer does, so they deduplicate with the kernels' requests.
@compute @workgroup_size(64) fn upHost(@builtin(global_invocation_id) gid:vec3u){
 for(var i=gid.x;i<upResidentCount();i+=${64*this.hostGroups}u){upInclude(upPageOf(upResidentBlock(i)));}
 let count=min(atomicLoad(&upWork[${W.host}u]),${l.host}u);let pins=min(atomicLoad(&upWork[${W.pins}u]),${l.pins}u);
 for(var i=gid.x;i<count;i+=${64*this.hostGroups}u){upAsk(${l.work.hostList}u+4u*i);}
 for(var i=gid.x;i<pins;i+=${64*this.hostGroups}u){upAsk(${l.work.pins}u+4u*i);}
}`),this.published?compile("Uniform page pool publish",/* wgsl */`${source(l,0,0,undefined,"read")}
@group(0) @binding(1) var<storage,read_write> upPublished:array<u32>;
// One lane per block: the signed coordinate and current resident state.
@compute @workgroup_size(64) fn upPublish(@builtin(global_invocation_id) gid:vec3u){
 let b=gid.x;if(b>=UP_BLOCKS){return;}
 if(b==0u){
  let count=upResidentCount();upPublished[${UNIFORM_PAGE_POOL_PUBLISHED.resident}u]=count;
  var low=vec3i(2147483647);var high=vec3i(-2147483647);
  for(var i=0u;i<count;i++){let q=upPageOf(upResidentBlock(i));low=min(low,q);high=max(high,q+vec3i(1));}
  for(var a=0u;a<3u;a++){upPublished[20u+a]=bitcast<u32>(low[a]);upPublished[24u+a]=bitcast<u32>(high[a]);}
 }
 let r=UP_AT_RECORDS+8u*b;let o=${l.published.records}u+4u*b;
 let q=upPageOf(b);upPublished[o]=bitcast<u32>(q.x);upPublished[o+1u]=bitcast<u32>(q.y);upPublished[o+2u]=bitcast<u32>(q.z);
 upPublished[o+3u]=select(0u,1u,upState[r+3u]==${UNIFORM_PAGE_POOL_BLOCK.resident}u);
}`):undefined,compile("Uniform page pool commit",/* wgsl */`${source(l,0,0,1,"read_write")}
${this.mirrors.map((_,i)=>`@group(0) @binding(${2+i}) var<storage,read_write> upMirror${i}:array<u32>;`).join("\n")}
const S_FREE:u32=${S.free}u;const S_PENDING:u32=${S.pending}u;const S_FATAL:u32=${S.fatal}u;
const W_POPS:u32=${W.pops}u;const W_HEAD:u32=${W.head}u;const W_RING:u32=${l.work.ring}u;const RING_MASK:u32=${l.ringCapacity-1}u;
const R_STATE:u32=${R.state}u;const R_GENERATION:u32=${R.generation}u;const R_IDLE:u32=${R.idle}u;
const FREE:u32=${B.free}u;const RESIDENT:u32=${B.resident}u;const PENDING:u32=${B.pending}u;
const IDLE_STEPS:u32=${l.idleSteps}u;const CLAIM_WIDTH:u32=${64*this.claimGroups}u;
// The only stores of table and block words: to the state and to every mirror.
fn upPutEntry(i:u32,word:u32){upState[UP_AT_TABLE+i]=word;${puts(l.tail.table)}}
fn upPutBlock(i:u32,word:u32){upState[UP_AT_BLOCKS+i]=word;${puts(l.tail.blocks)}}
fn upPutWindow(i:u32,word:u32){upState[UP_AT_WINDOW+i]=word;${puts(l.tail.window)}}
${frameParams?`@group(0) @binding(${2+this.mirrors.length}) var<storage,read_write> upParams:array<u32>;`:""}
${windowBody}
@compute @workgroup_size(1) fn upWindow(){
 upDecide();
 ${frameParams?`if(upState[UP_AT_WINDOW+20u]==u32(bitcast<f32>(upParams[53u]))){upState[UP_AT_WINDOW+20u]=0u;}`:""}
 if(upState[UP_AT_WINDOW+7u]!=0u){upFail(UP_FATAL_EDGE,upFramePage());}
 for(var i=0u;i<12u;i++){upPutWindow(i,upState[UP_AT_WINDOW+i]);}
 ${frameParams?`for(var a=0u;a<3u;a++){
  let d=f32(bitcast<i32>(upState[UP_AT_WINDOW+4u+a]))*32.0;
  if(d!=0.0){upParams[8u+a]=bitcast<u32>(bitcast<f32>(upParams[8u+a])+d*bitcast<f32>(upParams[4u+a]));upParams[32u+a]=bitcast<u32>(bitcast<f32>(upParams[32u+a])-d);}
 }`:""}
}
fn upBlockWordOf(q:vec3i)->u32{let m=vec3u(q)&vec3u(UP_TABLE-1u);return m.x|(m.y<<6u)|(m.z<<12u);}
// upClaim: one block per listed request, popped off the ring head; the
// blocks free or quarantined before this commit are all claimable. Each
// request owns its table entry, its block's words and its new-list word. A
// request whose entry is SOLID by now is dropped.
@compute @workgroup_size(64) fn upClaim(@builtin(global_invocation_id) gid:vec3u){
 let asked=atomicLoad(&upWork[0u]);
 // A list that overflowed left claim words no listed request clears.
 if(asked>UP_REQUESTS){for(var w=gid.x;w<${3*l.entries}u;w+=CLAIM_WIDTH){atomicStore(&upWork[UP_WORK_CLAIMS+w],0u);}}
 let count=min(asked,UP_REQUESTS);let generation=upState[0u]+1u;
 let head=atomicLoad(&upWork[W_HEAD]);let available=upState[S_FREE]+upState[S_PENDING];
 for(var i=gid.x;i<count;i+=CLAIM_WIDTH){
  let a=UP_WORK_LIST+4u*i;
  let q=vec3i(bitcast<i32>(atomicLoad(&upWork[a])),bitcast<i32>(atomicLoad(&upWork[a+1u])),bitcast<i32>(atomicLoad(&upWork[a+2u])));
  let e=upEntry(q);let c=UP_WORK_CLAIMS+3u*e;
  atomicStore(&upWork[c],0u);atomicStore(&upWork[c+1u],0u);atomicStore(&upWork[c+2u],0u);
  let held=upWord(q);
  if(held==UP_SOLID_WORD||atomicLoad(&upWork[2u])!=0u){continue;}
  if(held!=UP_AIR_WORD){if(any(upPageOf(held)!=q)){upFail(UP_FATAL_ALIAS,q);}continue;}
  let pop=atomicAdd(&upWork[W_POPS],1u);
  if(pop>=available){upFailWith(UP_FATAL_EXHAUSTED,q,upState[${S.resident}u]);continue;}
  let block=atomicLoad(&upWork[W_RING+((head+pop)&RING_MASK)]);let r=UP_AT_RECORDS+8u*block;
  upState[r]=bitcast<u32>(q.x);upState[r+1u]=bitcast<u32>(q.y);upState[r+2u]=bitcast<u32>(q.z);
  upState[r+R_STATE]=RESIDENT;upState[r+R_GENERATION]=generation;upState[r+R_IDLE]=0u;
  upPutEntry(e,block);upPutBlock(block,upBlockWordOf(q));upState[UP_AT_NEW+pop]=block;
 }
}
// upSlots: one workgroup over every block, each lane a run of them. Ages and
// retires the residents this commit did not claim, frees the blocks the last
// commit quarantined, then lists the residents (ascending) and queues the
// retired behind the ring's free blocks; the last lane seals the status. The
// sentinels' records are neither resident nor quarantined, so they only have
// their keep word cleared.
const LANES:u32=256u;const PER:u32=${Math.ceil(l.blocks/256)}u;
var<workgroup> upScan:array<vec2u,256>;
@compute @workgroup_size(256) fn upSlots(@builtin(local_invocation_index) lane:u32){
 let first=min(lane*PER,UP_BLOCKS);let last=min(first+PER,UP_BLOCKS);
 let generation=upState[0u]+1u;let head=atomicLoad(&upWork[W_HEAD]);let available=upState[S_FREE]+upState[S_PENDING];
 // x: residents, y: retired.
 var count=vec2u(0u);
 for(var s=first;s<last;s++){
  let r=UP_AT_RECORDS+8u*s;let kept=atomicExchange(&upWork[UP_WORK_KEEP+s],0u);let state=upState[r+R_STATE];
  if(state==PENDING){upState[r+R_STATE]=FREE;continue;}
  if(state!=RESIDENT){continue;}
  if(upState[r+R_GENERATION]!=generation){
   let idle=select(upState[r+R_IDLE]+1u,0u,kept!=0u);upState[r+R_IDLE]=idle;
   if(idle>=IDLE_STEPS){
    // Only an entry that names this block is cleared.
    let e=upEntry(upPageOf(s));if(upState[UP_AT_TABLE+e]==s){upPutEntry(e,UP_AIR_WORD);}
    upPutBlock(s,UP_FREE);upState[r+R_STATE]=PENDING;count.y+=1u;continue;
   }
  }
  count.x+=1u;
 }
 upScan[lane]=count;storageBarrier();workgroupBarrier();
 // Inclusive Hillis-Steele scan over the lanes.
 for(var stride=1u;stride<LANES;stride*=2u){
  var add=vec2u(0u);if(lane>=stride){add=upScan[lane-stride];}
  workgroupBarrier();upScan[lane]+=add;workgroupBarrier();
 }
 var at=upScan[lane]-count;
 for(var s=first;s<last;s++){
  let state=upState[UP_AT_RECORDS+8u*s+R_STATE];
  if(state==RESIDENT){upState[UP_AT_RESIDENT+at.x]=s;at.x+=1u;}
  else if(state==PENDING){atomicStore(&upWork[W_RING+((head+available+at.y)&RING_MASK)],s);at.y+=1u;}
 }
 if(lane==LANES-1u){
  let pops=min(atomicLoad(&upWork[W_POPS]),available);let flags=atomicLoad(&upWork[${W.flags}u]);
  if(upState[S_FATAL]==0u&&flags!=0u){
   for(var k=0u;k<3u;k++){upState[${S.fatalPage}u+k]=atomicLoad(&upWork[${W.firstPage}u+k]);}
   upState[${S.fatalGeneration}u]=generation;upState[${S.fatalFirst}u]=atomicLoad(&upWork[${W.first}u]);upState[${S.fatalDetail}u]=atomicLoad(&upWork[${W.detail}u]);
  }
  upState[S_FATAL]=flags;upState[${S.generation}u]=generation;upState[${S.resident}u]=at.x;upState[${S.claimed}u]=pops;
  upState[S_FREE]=available-pops;upState[S_PENDING]=at.y;upState[${S.requests}u]=min(atomicLoad(&upWork[0u]),UP_REQUESTS);
  atomicStore(&upWork[W_HEAD],(head+pops)&RING_MASK);atomicStore(&upWork[W_POPS],0u);
  atomicStore(&upWork[${W.requests}u],0u);atomicStore(&upWork[${W.host}u],0u);
 }
}`)]);
  const binding=(at:number,type:GPUBufferBindingType):GPUBindGroupLayoutEntry=>({binding:at,visibility:GPUShaderStage.COMPUTE,buffer:{type}});
  const readerLayout=device.createBindGroupLayout({entries:[binding(0,"read-only-storage"),binding(1,"storage")]});
  const writerLayout=device.createBindGroupLayout({entries:[binding(0,"storage"),binding(1,"storage"),...this.mirrors.map((_,i)=>binding(2+i,"storage")),...(frameParams?[binding(2+this.mirrors.length,"storage")]:[])]});
  const entries=[{binding:0,resource:{buffer:this.state}},{binding:1,resource:{buffer:this.work}}];
  this.reader=device.createBindGroup({layout:readerLayout,entries});
  this.writer=device.createBindGroup({layout:writerLayout,entries:[...entries,...this.mirrors.map((m,i)=>({binding:2+i,resource:{buffer:m.buffer}})),...(frameParams?[{binding:2+this.mirrors.length,resource:{buffer:frameParams}}]:[])]});
  // The published record takes the work buffer's place beside the read-only state.
  if(this.published)this.publisher=device.createBindGroup({layout:readerLayout,entries:[entries[0]!,{binding:1,resource:{buffer:this.published}}]});
  const reading=device.createPipelineLayout({bindGroupLayouts:[readerLayout]}),writing=device.createPipelineLayout({bindGroupLayouts:[writerLayout]});
  [this.windowPipeline,this.hostPipeline,this.claimPipeline,this.slotsPipeline,this.publishPipeline]=await Promise.all([
   device.createComputePipelineAsync({label:"Uniform page pool upWindow",layout:writing,compute:{module:commitModule,entryPoint:"upWindow"}}),
   device.createComputePipelineAsync({label:"Uniform page pool upHost",layout:reading,compute:{module:hostModule,entryPoint:"upHost"}}),
   device.createComputePipelineAsync({label:"Uniform page pool upClaim",layout:writing,compute:{module:commitModule,entryPoint:"upClaim"}}),
   device.createComputePipelineAsync({label:"Uniform page pool upSlots",layout:writing,compute:{module:commitModule,entryPoint:"upSlots"}}),
   publishModule?device.createComputePipelineAsync({label:"Uniform page pool upPublish",layout:reading,compute:{module:publishModule,entryPoint:"upPublish"}}):undefined,
  ]);
 }
 /** Queue a host command (sources, drops, edits): page (x,y,z) is asked for
  * in the next encodeCommit, exactly as upRequest asks. */
 request(x:number,y:number,z:number):void{
  if(this.hostCount>=this.layout.host)throw new Error(`Uniform page pool host commands exceed ${this.layout.host} a step`);
  if(!int32(x)||!int32(y)||!int32(z))throw new RangeError(`Uniform page coordinate must be 32-bit integers: ${x}, ${y}, ${z}`);
  const a=4*this.hostCount++;this.host[a]=x;this.host[a+1]=y;this.host[a+2]=z;
 }
 /** Turn the step's requests into claims, retire the idle and publish the
  * next generation: after the consumer launches that request and keep, in
  * the same encoder or a later one. The queued host commands are uploaded
  * at this call (queue-ordered, ahead of the encoder's submit), so an
  * encoder holds one commit with host commands at most. Every mirror and
  * the published record are current when the pass ends. */
 encodeCommit(encoder:GPUCommandEncoder):void{
  if(!this.slotsPipeline)throw new Error("Uniform page pool is not initialized");
  const host=this.hostCount;
  if(host){
   this.word[0]=host;this.device.queue.writeBuffer(this.work,4*UNIFORM_PAGE_POOL_WORK.host,this.word);
   this.device.queue.writeBuffer(this.work,4*this.layout.work.hostList,this.host,0,4*host);this.hostCount=0;
  }
  const pass=encoder.beginComputePass(UniformPagePool.PASS);
  // Moving worlds also gather the signed coordinates of every old resident.
  // This existing small launch carries the bounds; it never scans field cells.
  if(host||this.pinCount||this.layout.worldPolicy){pass.setPipeline(this.hostPipeline!);pass.setBindGroup(0,this.reader!);pass.dispatchWorkgroups(this.hostGroups);}
  // The request list is complete here and no block is popped yet: a launch
  // over the list that settles a requested entry (to SOLID) belongs between
  // upHost and upClaim.
  pass.setBindGroup(0,this.writer!);
  pass.setPipeline(this.windowPipeline!);pass.dispatchWorkgroups(1);
  pass.setPipeline(this.claimPipeline!);pass.dispatchWorkgroups(this.claimGroups);
  pass.setPipeline(this.slotsPipeline);pass.dispatchWorkgroups(1);
  if(this.publishPipeline){pass.setPipeline(this.publishPipeline);pass.setBindGroup(0,this.publisher!);pass.dispatchWorkgroups(this.publishGroups);}
  pass.end();
 }
 /** Copy the status (UNIFORM_PAGE_POOL_STATUS.words words) to `target` at
  * `offset` bytes: the host's only view of the pool, to map when it likes. */
 encodeReceipt(encoder:GPUCommandEncoder,target:GPUBuffer,offset:number):void{
  encoder.copyBufferToBuffer(this.state,0,target,offset,4*UNIFORM_PAGE_POOL_STATUS.words);
 }
 destroy():void{this.state.destroy();this.work.destroy();this.published?.destroy();}
}
