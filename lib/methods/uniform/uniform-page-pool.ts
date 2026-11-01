/** GPU page pool of the domain-less uniform solver: `blocks` storage blocks
 * of 32^3 cells. Block 0 is the AIR sentinel and block `blocks`-1 the SOLID
 * sentinel; blocks 1..blocks-2 are the slots, claimed on demand anywhere in
 * an unbounded signed page lattice and retired when idle. A two-level
 * toroidal directory (tiered-adaptivity §2.1, uniform-window-address.ts) is
 * the only index: a top table of (table/4)^3 words, one a super-block of 4^3
 * pages at entry (q>>2) & (table/4-1) per axis, then UNIFORM_SUPER_BLOCK_SLOTS
 * child tables of 64 words, one a page of a super-block slot. A top word is
 * 0 (AIR), 0xffffffff (SOLID) or the slot 1..4094 that holds the
 * super-block; slot 0 is all AIR and slot 4095 all SOLID, so a page's word
 * is child[min(top, 4095)*64 + (q&3 packed x|y<<2|z<<4)], two dependent
 * reads. Two resident pages `table` apart on an axis meet in one entry (two
 * super-blocks table/4 apart), which is fatal. A page word is 0 (AIR: no
 * storage), 0xffffffff (SOLID: no storage) or the block 1..blocks-2 that
 * holds the page, so min(word, blocks-1) is the block a read resolves to.
 * SOLID is written by the host seed alone: no kernel writes it, and a SOLID
 * page is never requested, claimed or retired. Everything is GPU resident;
 * the host seeds the pool at scene build (uniformPagePoolSeed,
 * UniformPagePool.seed), queues page coordinates (request), fixes the pinned
 * pages (pin) and encodes fixed launches (encodeCommit). The mixed frame owns
 * the solver's (UniformMixedFrame.pagePool): the census asks for its
 * residency certificate's pages and one commit follows every census.
 *
 * A super-block slot is claimed by the commit that claims the first page of a
 * super-block without one (its 64 children start as the top word it
 * replaces, its background), counts its resident pages, and retires, its top
 * word back to the background, at the commit that retires its last; like a
 * block it is quarantined for a commit before reuse. A seeded super-block
 * whose pages without storage are a mix of AIR and SOLID has no background:
 * its slot is fixed and never retires. `superBlocks` slots (1..superBlocks)
 * are claimable; claiming past them is the superBlocks fatal.
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
 *                               super-block slots held (superBlocks), the
 *                               certificate's reach (closureCap), else 0
 *            13 superResident   super-block slots held, the fixed among them
 *            14 superFree       free super-block slots, the quarantined apart
 *            15 superPending    super-block slots this commit retired
 *   table    [16, 16+directory): the top table, entry x | y<<b | z<<2b (b =
 *            log2 table/4) of the masked super-block coordinate, then the
 *            4096 child tables of 64 words, then the flat window: 64^3
 *            words, the raw word of lattice pages -1..62 an axis under the
 *            frame (uniform-window-address.ts), rewritten whole by upFlat at
 *            the end of every commit
 *   window   32 runtime words after the directory: F, revision, delta, fatal,
 *            boundary counter, world policy/box and the disabled probe command
 *   blocks   one word per block, after the runtime header: the world page residue
 *            x | y<<8 | z<<16 (each coordinate & (table-1)) of the page a
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
 *   retired  `blocks`-2 words: the blocks this commit retired, ascending,
 *            [0, status.pending); empty at generation 0. With the resident
 *            list it names every block whose liveness this commit changed
 *            (uniform-live-lists).
 *   supers   8 words per super-block slot (UNIFORM_PAGE_POOL_SUPER), 4096:
 *            super-block x,y,z (i32, page >> 2), state (UNIFORM_PAGE_POOL_BLOCK,
 *            fixed for a seeded mixed one), generation of the claim, the
 *            background top word, 2 reserved
 * Word offsets: uniformPagePoolLayout(options).state.
 *
 * published: 32 header words [edge, blocks, top table edge, directory offset,
 * resident count, record stride=4, version=3, reserved, 12 frame lanes,
 * signed world resident bounds at words 20..22 / 24..26, padding], then one record [signed world page x,y,z, state] per pool block:
 * the allocator's own record, frame invariant (a reader subtracts the directory header's F for lattice pages). State is 1 for
 * resident, 0 otherwise; SOLID remains a directory state (2 in the overlay).
 * The overlay copies the directory only while visible, after these records.
 *
 * The directory, runtime header and block words (layout.tail.words) are the
 * tail a reader of another buffer needs. initialize(mirrors) takes up to
 * UNIFORM_PAGE_POOL_MIRRORS buffers and a word offset in each: the seed
 * uploads the tail there and the commit kernels store every directory and
 * block word they change there as well as in state, so a mirror equals the
 * state's tail after every commit with nothing copied.
 *
 * work (UniformPagePool.work; a requesting consumer binds it read_write,
 * array<atomic<u32>>; nothing else of it is a consumer's):
 *   [0,16)   header (UNIFORM_PAGE_POOL_WORK): 0 requests appended this step
 *            (may pass the list capacity), 1 host commands uploaded, 2 sticky
 *            fatal bits, 3 the bit raised first (the lock on words 4..6 and
 *            9), 4..6 its page, 7 blocks popped by the commit under way, 8
 *            ring head, 9 the first bit's detail, 10 pinned pages, 11
 *            super-block slots popped, 12 super ring head, 13..15 reserved
 *   host     4 words per host command (x,y,z,0), `hostCapacity` of them
 *   pins     4 words per pinned page (x,y,z,0), `pinCapacity` of them: written
 *            by pin() at scene build and asked for by every commit
 *   list     4 words per distinct request (x,y,z,0), `requestCapacity` of them
 *   keep     one word per block: non-zero when upKeep marked it this step;
 *            a BULK16 word when upDemote gave the page to that slot
 *   ring     layout.ringCapacity words (the power of two that holds the
 *            slots), a queue of blocks: [head, head+free) are free and the
 *            `pending` after them are quarantined
 *   claims   5 words per top entry: the x, y and z wrap tags
 *            ((q>>log2 table)<<1 | 1) of the super-block asked for there this
 *            step (0 unasked), then a 64-bit mask of its pages asked for
 *   supers   4096 words, the super ring: [head, head+superFree) are free
 *            slots and the superPending after them quarantined
 *   counts   4096 words, the resident pages of each super-block slot
 * Word offsets: uniformPagePoolLayout(options).work.
 *
 * A step: consumer kernels call upRequest / upKeep (uniformPagePoolWGSL), the
 * host queues request(x,y,z), then encodeCommit, which asks for the host
 * commands and the pinned pages itself. A request toward a SOLID page does
 * nothing; one for a resident page keeps it. Otherwise each axis tag of its
 * super-block is compare-and-swapped into its top entry's claim words (one
 * that meets a different tag found an alias), then the page's bit is set in
 * the entry's mask: the invocation that sets it appends the page to the
 * list, any other is a duplicate. So a listed request owns its page word
 * until the commit ends. upClaimSuper gives each top entry asked for a slot
 * where it has none (one held by another super-block is an alias), popped
 * off the super ring; upClaim pops one block per listed request off the ring
 * head (first-in first-out; which request gets which block is a race);
 * upSlots ages the residents it did not just claim (kept: idle 0, else
 * idle+1), retires those at `idleSteps` by clearing their page word and
 * block word and queuing them behind the free blocks, and rebuilds the
 * resident list; upSuperSlots retires every slot left without a resident
 * page. A block retired by commit n is therefore not claimable in commit n,
 * and only once every older free block is gone from commit n+1. Claims
 * precede retirement, so a request that aliases a page retiring in the same
 * commit is still an alias. A pinned page is asked for by every commit, so
 * it never ages.
 *
 * A fatal leaves the pool consistent and the commits running: an exhausted,
 * aliased or overflowed request is simply not resident, and a certificate
 * over the cap asks for nothing. The bits are sticky and the first raised
 * keeps its page; uniformPagePoolFatalMessage reads them from the receipt. */
import { UNIFORM_MOVING_WINDOW_INNER, UNIFORM_MOVING_WINDOW_MARGIN, uniformMovingWindowDecisionWGSL } from "./uniform-moving-window";
import { UNIFORM_DIRECTORY_FLAT_WORDS, UNIFORM_SUPER_BLOCK_CHILDREN, UNIFORM_SUPER_BLOCK_SLOTS, UNIFORM_WINDOW_HEADER_WORDS, UNIFORM_WINDOW_PAGES, uniformDirectoryFlat, uniformDirectoryWord, uniformSuperBlockLocal, uniformWindowDerivedWords, uniformWindowWorldWGSL } from "./uniform-window-address";
import type { GPUFluidVolumePageSource } from "../../core/method-view-records";
import type { LatticePages } from "../../core/initial-liquid-lattice";
import { bulk16PageWord } from "./uniform-bulk-pages-reference";
import { RESIDENCY_STAGES, residencyFootprintContains } from "./uniform-local-residency";

const RECORD=8;
/** upFlat's workgroups a row of its two-dimensional launch. */
const FLAT_ROW=64;
export const UNIFORM_PAGE_POOL_EDGE=32;
/** upBlockOf of a page that is not resident. */
export const UNIFORM_PAGE_POOL_NONE=0xffffffff;
/** The table words of a page without storage; any other word is its block. */
export const UNIFORM_PAGE_POOL_AIR=0,UNIFORM_PAGE_POOL_SOLID=0xffffffff;
/** The block word of a block that holds no page: free, quarantined or a sentinel. */
export const UNIFORM_PAGE_POOL_FREE=0xffffffff;
/** What a table entry is (upKind). */
export const UNIFORM_PAGE_POOL_KIND={resident:0,air:1,solid:2,bulk:3} as const;
/** Mirrors initialize() takes: each is one storage binding of the commit kernels. */
export const UNIFORM_PAGE_POOL_MIRRORS=4;
export const UNIFORM_PAGE_POOL_STATUS={generation:0,resident:1,claimed:2,free:3,pending:4,fatal:5,fatalPage:6,fatalGeneration:9,fatalFirst:10,requests:11,fatalDetail:12,superResident:13,superFree:14,superPending:15,words:16} as const;
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
 * world's frame, and the page is not asked for. superBlocks: a page's
 * super-block needed a slot and none of the `superBlocks` was free. */
export const UNIFORM_PAGE_POOL_FATAL={exhausted:1,alias:2,requests:4,closureCap:8,closure:16,edge:32,superBlocks:64} as const;
/** Record states; sentinel: block 0 and the last block (and super-block
 * slots 0, 4095 and those past superBlocks), which never change state;
 * fixed: a seeded super-block slot that never retires. */
export const UNIFORM_PAGE_POOL_BLOCK={free:0,resident:1,pending:2,sentinel:3,fixed:4} as const;
export const UNIFORM_PAGE_POOL_RECORD={page:0,state:3,generation:4,idle:5,words:RECORD} as const;
/** A super-block slot's record: its super-block (page >> 2), state, generation of the claim and background top word. */
export const UNIFORM_PAGE_POOL_SUPER={block:0,state:3,generation:4,background:5,words:RECORD} as const;
export const UNIFORM_PAGE_POOL_WORK={requests:0,host:1,flags:2,first:3,firstPage:4,pops:7,head:8,detail:9,pins:10,superPops:11,superHead:12,words:16} as const;
/** The published record (file header): header words and the offset of the flags. */
export const UNIFORM_PAGE_POOL_PUBLISHED={edge:0,blocks:1,table:2,directory:3,resident:4,stride:5,version:6,frame:8,low:20,high:24,records:32,recordWords:4} as const;

/** The scene's bounding box in pages: its low corner and its extent. */
export interface UniformPagePoolBox {readonly origin:readonly [number,number,number];readonly pages:readonly [number,number,number]}
export interface UniformPagePoolOptions {
 /** The directory's period in pages, a power of two in 4..256 (default
  * UNIFORM_WINDOW_PAGES, 256): the virtual window is table*32 cells an axis,
  * the top table table/4 super-blocks an axis. */
 readonly table?:number;
 /** Super-block slots a commit may hold, the fixed seeded ones among them: 1..4094 (default 4094). */
 readonly superBlocks?:number;
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
 /** Frames past the inner window's 60 pages an axis may be placed in the full
  * one (default: no box, or a box past 60 pages on an axis). A narrow pool
  * refuses them (edge fatal), as the 64-page table did: its stored pages stay
  * in the flat window, and its readers' lookups compile to that one read
  * (umFlatWindow). */
 readonly wide?:boolean;
}
export type UniformPagePoolLayout=ReturnType<typeof uniformPagePoolLayout>;

const powerOfTwo=(n:number)=>Number.isSafeInteger(n)&&n>0&&(n&(n-1))===0;
const int32=(n:number)=>(n|0)===n;
/** Sizes and word offsets of the pool's buffers and of the tail (see the file header). */
export function uniformPagePoolLayout(options:UniformPagePoolOptions){
 const table=options.table??UNIFORM_WINDOW_PAGES,blocks=options.blocks,slots=blocks-2,idleSteps=options.idleSteps,requests=options.requestCapacity??slots,host=options.hostCapacity??64,pins=options.pinCapacity??64,closureCap=options.closureCap??16;
 const superBlocks=options.superBlocks??UNIFORM_SUPER_BLOCK_SLOTS-2,SUPERS=UNIFORM_SUPER_BLOCK_SLOTS;
 // A block word packs the window page at eight bits an axis.
 if(!powerOfTwo(table)||table<4||table>UNIFORM_WINDOW_PAGES)throw new RangeError(`Uniform page table edge must be a power of two in 4..${UNIFORM_WINDOW_PAGES}: ${table}`);
 if(!Number.isSafeInteger(superBlocks)||superBlocks<1||superBlocks>SUPERS-2)throw new RangeError(`Uniform page pool super-blocks must be an integer in 1..${SUPERS-2}: ${superBlocks}`);
 if(!Number.isSafeInteger(blocks)||blocks<3||blocks>65536)throw new RangeError(`Uniform page pool blocks must be an integer in 3..65536, two of them the sentinels: ${blocks}`);
 if(!Number.isSafeInteger(idleSteps)||idleSteps<1)throw new RangeError(`Uniform page pool idle steps must be a positive integer: ${idleSteps}`);
 if(!Number.isSafeInteger(requests)||requests<1||!Number.isSafeInteger(host)||host<1||!Number.isSafeInteger(pins)||pins<1)throw new RangeError(`Uniform page pool request capacities must be positive integers: ${requests}, ${host}, ${pins}`);
 if(!Number.isSafeInteger(closureCap)||closureCap<0)throw new RangeError(`Uniform page pool closure cap must be a non-negative integer: ${closureCap}`);
 // A box wider than the table would alias itself.
 const origin:readonly [number,number,number]=options.box?[...options.box.origin]:[0,0,0],extent:readonly [number,number,number]=options.box?[...options.box.pages]:[0,0,0];
 if(options.box&&!(origin.every(int32)&&extent.every(n=>Number.isSafeInteger(n)&&n>=1&&n<=table)))throw new RangeError(`Uniform page pool box must have a 32-bit integer origin and 1..${table} pages an axis: origin ${origin}, pages ${extent}`);
 const boxPages=extent[0]*extent[1]*extent[2],wide=options.wide??(!options.box||extent.some(n=>n>UNIFORM_MOVING_WINDOW_INNER-2*UNIFORM_MOVING_WINDOW_MARGIN));
 let ringCapacity=1;while(ringCapacity<slots)ringCapacity*=2;
 const edge=table/4,entries=edge**3,flat=entries+UNIFORM_SUPER_BLOCK_CHILDREN*SUPERS,directory=flat+UNIFORM_DIRECTORY_FLAT_WORDS,tableAt=UNIFORM_PAGE_POOL_STATUS.words,blockWords=tableAt+directory+UNIFORM_WINDOW_HEADER_WORDS,records=blockWords+blocks,resident=records+RECORD*blocks,claimed=resident+slots,retired=claimed+slots,supers=retired+slots;
 const hostList=UNIFORM_PAGE_POOL_WORK.words,pinList=hostList+4*host,list=pinList+4*pins,keep=list+4*requests,ring=keep+blocks,claims=ring+ringCapacity,superRing=claims+5*entries,counts=superRing+SUPERS;
 return {table,tableBits:Math.log2(table),
  /** The top table: `edge` super-blocks an axis, `entries` words; `directory` words of top and child tables and the flat window. */
  edge,edgeBits:Math.log2(edge),entries,directory,superBlocks,blocks,slots,solidBlock:blocks-1,ringCapacity,idleSteps,requests,host,pins,closureCap,
  /** The box (origin and extent in pages) and its page count; count 0: no box. */
  box:{origin,pages:extent,count:boxPages},wide,
  state:{table:tableAt,window:tableAt+directory,blocks:blockWords,records,resident,claimed,retired,supers,words:supers+RECORD*SUPERS},
  /** Word offsets inside the tail, wherever it lies: state.table in the state buffer, a mirror's offset in a mirror. */
  tail:{table:0,children:entries,flat,window:directory,blocks:directory+UNIFORM_WINDOW_HEADER_WORDS,words:directory+UNIFORM_WINDOW_HEADER_WORDS+blocks},
  worldPolicy:options.worldPolicy??0,
  work:{hostList,pins:pinList,list,keep,ring,claims,superRing,counts,bounds:counts+SUPERS,words:counts+SUPERS+8},
  /** One coordinate/state record per block, independent of the seed box. */
  published:{records:UNIFORM_PAGE_POOL_PUBLISHED.records,words:UNIFORM_PAGE_POOL_PUBLISHED.records+4*blocks}} as const;
}

/** A page of a seed: its signed page coordinate and the block that holds it,
 * or (block 0) the BULK16 slot that holds it: its entry is that slot's page
 * word (bulk16PageWord) and it counts in its super-block like a resident
 * page, as a demote leaves it. */
export interface UniformPagePoolSeedPage {readonly x:number;readonly y:number;readonly z:number;readonly block:number;readonly slot?:number}
/** The host copy of a seeded pool, generation 0. Read-only: the arrays are
 * what the buffers were written from. table, blocks, tail and resident are
 * views of state. */
export interface UniformPagePoolSeed {
 readonly layout:UniformPagePoolLayout;
 /** The whole state buffer. */
 readonly state:Uint32Array<ArrayBuffer>;
 /** The directory, layout.directory words: the top table then the child tables (uniformDirectoryWord reads a page's word). */
 readonly table:Uint32Array<ArrayBuffer>;
 /** The block words, layout.blocks of them. */
 readonly blocks:Uint32Array<ArrayBuffer>;
 /** Directory, runtime header then block words, layout.tail.words: what a mirror holds at its offset. */
 readonly tail:Uint32Array<ArrayBuffer>;
 /** The resident blocks, ascending. */
 readonly resident:Uint32Array<ArrayBuffer>;
 /** The free ring from its head (word 0), layout.ringCapacity words: the `free` first are the free blocks, ascending. */
 readonly ring:Uint32Array<ArrayBuffer>;
 readonly free:number;
 /** The super ring from its head, UNIFORM_SUPER_BLOCK_SLOTS words: the superFree first are the free slots, ascending. */
 readonly superRing:Uint32Array<ArrayBuffer>;
 readonly superFree:number;
 /** The resident pages of each super-block slot (the work buffer's counts). */
 readonly superCount:Uint32Array<ArrayBuffer>;
}
/** Top entry of page (x,y,z): its super-block's, masked. */
export function uniformPagePoolEntry(layout:UniformPagePoolLayout,x:number,y:number,z:number):number{
 const mask=layout.edge-1,bits=layout.edgeBits;return ((x>>2)&mask)|(((y>>2)&mask)<<bits)|(((z>>2)&mask)<<(2*bits));
}
/** The word of page (x,y,z) in a directory of the layout (seed.table, or a tail at word `at`). */
export function uniformPagePoolWord(layout:UniformPagePoolLayout,table:ArrayLike<number>,x:number,y:number,z:number,at=0):number{
 return uniformDirectoryWord(table,x,y,z,layout.edge,at);
}
/** Block word of a block that holds page (x,y,z). */
export function uniformPagePoolBlockWord(layout:UniformPagePoolLayout,x:number,y:number,z:number):number{
 const mask=layout.table-1;return ((x&mask)|((y&mask)<<8)|((z&mask)<<16))>>>0;
}
/** The whole pool state for a given resident set, built on the host at scene
 * build: `pages` are resident in the blocks they name (any order, any
 * blocks of 1..blocks-2, holes allowed), and `other(x,y,z)` gives
 * UNIFORM_PAGE_POOL_KIND.air or .solid for every other page of the window,
 * by its window page coordinate (0..table-1 an axis); without it every other
 * page is AIR. Blocks no page names are free, in ascending order from the
 * ring head. Super-block slots go to the super-blocks with a resident page in
 * the order their first is listed, then to the mixed AIR and SOLID ones in
 * entry order (fixed). Throws on a coordinate that is not a 32-bit integer, a
 * block out of range or named twice, a page listed twice, two pages that
 * alias one entry and more slots than superBlocks. */
export function uniformPagePoolSeed(layout:UniformPagePoolLayout,pages:readonly UniformPagePoolSeedPage[],other?:(x:number,y:number,z:number)=>number):UniformPagePoolSeed{
 const l=layout,S=UNIFORM_PAGE_POOL_STATUS,R=UNIFORM_PAGE_POOL_RECORD,B=UNIFORM_PAGE_POOL_BLOCK,K=UNIFORM_PAGE_POOL_KIND,U=UNIFORM_PAGE_POOL_SUPER,SUPERS=UNIFORM_SUPER_BLOCK_SLOTS,C=UNIFORM_SUPER_BLOCK_CHILDREN;
 const state=new Uint32Array(l.state.words),signed=new Int32Array(state.buffer),ring=new Uint32Array(l.ringCapacity),superRing=new Uint32Array(SUPERS),superCount=new Uint32Array(SUPERS);
 const table=state.subarray(l.state.table,l.state.table+l.directory),blocks=state.subarray(l.state.blocks,l.state.blocks+l.blocks);
 const record=(block:number)=>l.state.records+RECORD*block,held=(block:number)=>{const r=record(block);return `(${signed[r]}, ${signed[r+1]}, ${signed[r+2]})`;};
 const superRecord=(slot:number)=>l.state.supers+RECORD*slot,first=new Uint32Array(SUPERS);
 state[l.state.window+12]=l.worldPolicy;state.set(l.box.origin,l.state.window+13);state.set(l.box.pages,l.state.window+16);state.set(uniformWindowDerivedWords(l.worldPolicy,l.box.origin,l.box.pages),l.state.window+24);
 blocks.fill(UNIFORM_PAGE_POOL_FREE);state[record(0)+R.state]=state[record(l.solidBlock)+R.state]=B.sentinel;
 for(let s=0;s<SUPERS;s++)state[superRecord(s)+U.state]=B.sentinel;
 table.fill(UNIFORM_PAGE_POOL_SOLID,l.entries+C*(SUPERS-1),l.directory);
 let slots=0;
 const claim=(entry:number,state_:number,background:number,sb:readonly number[])=>{
  if(++slots>l.superBlocks)throw new RangeError(`Uniform page pool seed needs more than its ${l.superBlocks} super-block slots`);
  const r=superRecord(slots);signed.set(sb,r);state[r+U.state]=state_;state[r+U.background]=background;table[entry]=slots;
  return slots;
 };
 pages.forEach((page,i)=>{const {x,y,z,block}=page;
  const name=()=>`Uniform page pool seed page ${i} (${x}, ${y}, ${z})`;
  if((x|0)!==x||(y|0)!==y||(z|0)!==z)throw new RangeError(`${name()}: coordinates must be 32-bit integers`);
  const bulk=block===0&&page.slot!==undefined?bulk16PageWord(page.slot):0;
  if(!bulk&&(!Number.isSafeInteger(block)||block<1||block>l.blocks-2))throw new RangeError(`${name()}: block ${block} is outside 1..${l.blocks-2}`);
  const r=record(block),entry=uniformPagePoolEntry(l,x,y,z),sb=[x>>2,y>>2,z>>2];
  if(!bulk&&state[r+R.state]===B.resident)throw new Error(`${name()}: block ${block} already holds page ${held(block)}`);
  let slot=table[entry]!;
  if(slot){const q=superRecord(slot);if(signed[q]!==sb[0]||signed[q+1]!==sb[1]||signed[q+2]!==sb[2])throw new Error(`${name()}: aliases ${first[slot]?`page ${held(first[slot]!)} of block ${first[slot]}`:`super-block (${signed[q]}, ${signed[q+1]}, ${signed[q+2]})`} in a table of edge ${l.table}`);}
  else{slot=claim(entry,B.resident,UNIFORM_PAGE_POOL_AIR,sb);first[slot]=block;}
  const child=l.entries+C*slot+uniformSuperBlockLocal(x,y,z),at=table[child]!;
  if(at)throw new Error(`${name()}: listed twice (words ${at} and ${bulk||block})`);
  superCount[slot]!++;
  if(bulk){table[child]=bulk;return;}
  table[child]=block;blocks[block]=uniformPagePoolBlockWord(l,x,y,z);
  signed[r]=x;signed[r+1]=y;signed[r+2]=z;state[r+R.state]=B.resident;
 });
 let resident=0,free=0;
 for(let block=1;block<=l.blocks-2;block++){if(state[record(block)+R.state]===B.resident)state[l.state.resident+resident++]=block;else ring[free++]=block;}
 // The pages without storage: a super-block of one kind is that top word (or a resident one's background), a mix is a fixed slot.
 const kinds=new Uint8Array(C);
 if(other)for(let z=0,entry=0;z<l.edge;z++)for(let y=0;y<l.edge;y++)for(let x=0;x<l.edge;x++,entry++){
  const slot=table[entry]!;let air=0,solid=0;
  for(let k=0;k<C;k++){
   if(slot&&table[l.entries+C*slot+k])continue;
   const px=4*x+(k&3),py=4*y+((k>>2)&3),pz=4*z+(k>>4),kind=other(px,py,pz);
   if(kind===K.solid)solid++;else if(kind===K.air)air++;
   else throw new RangeError(`Uniform page pool seed: page (${px}, ${py}, ${pz}) must be classified air (${K.air}) or solid (${K.solid}): ${kind}`);
   kinds[k]=kind;
  }
  if(!solid)continue;
  if(!air&&!slot){table[entry]=UNIFORM_PAGE_POOL_SOLID;continue;}
  const s=slot||claim(entry,B.fixed,UNIFORM_PAGE_POOL_AIR,[x,y,z]);
  if(!air)state[superRecord(s)+U.background]=UNIFORM_PAGE_POOL_SOLID;else state[superRecord(s)+U.state]=B.fixed;
  for(let k=0;k<C;k++)if(!(slot&&table[l.entries+C*slot+k])&&kinds[k]===K.solid)table[l.entries+C*s+k]=UNIFORM_PAGE_POOL_SOLID;
 }
 let superFree=0;
 for(let s=slots+1;s<=l.superBlocks;s++){superRing[superFree++]=s;state[superRecord(s)+U.state]=B.free;}
 state[S.resident]=resident;state[S.free]=free;state[S.superResident]=slots;state[S.superFree]=superFree;
 uniformDirectoryFlat(table,[signed[l.state.window]!,signed[l.state.window+1]!,signed[l.state.window+2]!],l.edge);
 return {layout:l,state,table,blocks,tail:state.subarray(l.state.table,l.state.table+l.tail.words),resident:state.subarray(l.state.resident,l.state.resident+resident),ring,free,superRing,superFree,superCount};
}
/** A resident set inside the layout's box. */
export interface UniformPagePoolBoxSeedOptions {
 /** Which box pages are resident, by box coordinate (0..pages-1 an axis); default: all. */
 readonly resident?:(x:number,y:number,z:number)=>boolean;
 /** The box pages to walk, by box index ascending, in place of the whole box (each still asked `resident`). */
 readonly pages?:readonly number[];
 /** A number that fixes a shuffle of the blocks; omitted: the blocks are in scan order. */
 readonly permute?:number;
 /** The kind of an entry outside the box, by window page coordinate (default SOLID). */
 readonly outside?:(x:number,y:number,z:number)=>number;
 /** The BULK16 slot of a resident box page held in one (no block), by box coordinate; 0: a block. */
 readonly bulk?:(x:number,y:number,z:number)=>number;
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
 let blocks=0;
 if(options.permute!==undefined){
  // Fisher-Yates over a mulberry32 stream.
  let s=options.permute|0;
  const next=()=>{s=(s+0x6d2b79f5)|0;let t=Math.imul(s^(s>>>15),1|s);t=(t+Math.imul(t^(t>>>7),61|t))^t;return ((t^(t>>>14))>>>0)/4294967296;};
  for(let i=order.length-1;i>0;i--){const j=Math.floor(next()*(i+1)),o=order[i]!;order[i]=order[j]!;order[j]=o;}
 }
 const visit=(x:number,y:number,z:number)=>{
  if(options.resident&&!options.resident(x,y,z))return;
  const slot=options.bulk?.(x,y,z)??0;
  if(slot){pages.push({x:origin[0]+x,y:origin[1]+y,z:origin[2]+z,block:0,slot});return;}
  if(blocks>=layout.slots)throw new RangeError(`Uniform page pool box seed holds more pages than the pool's ${layout.slots} blocks`);
  pages.push({x:origin[0]+x,y:origin[1]+y,z:origin[2]+z,block:order[blocks++]!});
 };
 if(options.pages)for(const i of options.pages)visit(i%px,Math.floor(i/px)%py,Math.floor(i/(px*py)));
 else for(let z=0;z<pz;z++)for(let y=0;y<py;y++)for(let x=0;x<px;x++)visit(x,y,z);
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
 record[P.edge]=UNIFORM_PAGE_POOL_EDGE;record[P.blocks]=l.blocks;record[P.table]=l.edge;
 record[P.directory]=l.published.words;record[P.resident]=seed.resident.length;
 record[P.stride]=4;record[P.version]=3;
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
 /** The box pages resident at build, by box page coordinate, in box index
  * order (x fastest); every other box page is seeded AIR. */
 readonly stored:[number,number,number][];
 /** The pinned pages (UniformPagePool.pin), by box page coordinate. */
 readonly pins:{x:number;y:number;z:number}[];
 /** The liquid's surface at build where it was walked (an open tile rule):
  * by box page index, the 4h tile bounds [x0,y0,z0,x1,y1,z1] (inclusive) of
  * its wet tiles beside open dry ones. */
 readonly surface?:ReadonlyMap<number,readonly number[]>;
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
 * With `openTile` (whether a box 4h tile holds an open cell) the margin is
 * measured through open tiles (V2d): a page past a solid wall is not
 * reachable air (uniformPagePoolReachablePages).
 * liquid: the initial cell volumes, x fastest or by page (LatticePages of
 * UNIFORM_PAGE_POOL_EDGE cells), asked for only by the wet-cell case. origin: the box's low page in the window. The sources'
 * cell boxes and the pins are the window's, the AIR pages the box's. */
export function uniformPagePoolResidency(cells:readonly [number,number,number],record:{readonly blocks?:number;readonly air?:readonly (readonly [number,number,number])[]}|undefined,sources:readonly UniformPagePoolCellBox[],follows:boolean,liquid:()=>ArrayLike<number>|LatticePages,origin:readonly [number,number,number]=[0,0,0],openTile?:(x:number,y:number,z:number)=>boolean):UniformPagePoolResidency{
 const E=UNIFORM_PAGE_POOL_EDGE,pages:[number,number,number]=[Math.ceil(cells[0]/E),Math.ceil(cells[1]/E),Math.ceil(cells[2]/E)],[px,py,pz]=pages,count=px*py*pz;
 const live=follows,pinned=new Set<number>();
 let surface:Map<number,number[]>|undefined;
 // The resident set by box index: every box page (less record.air), or, live with no record.air, the wet pages' margin.
 let resident:Set<number>|undefined;
 if(live){
  for(const {x,y,z} of uniformPagePoolSourcePages(cells,sources,origin))pinned.add(x-origin[0]+px*(y-origin[1]+py*(z-origin[2])));
  if(!record?.air){
   const volume=liquid();resident=wetPages(cells,pages,volume,UNIFORM_PAGE_POOL_SEED_MARGIN);
   surface=new Map();
   if(openTile){const reached=uniformPagePoolReachablePages(cells,pages,volume,UNIFORM_PAGE_POOL_SEED_MARGIN,openTile,surface);for(const i of resident)if(!reached.has(i))resident.delete(i);}
   // No solid: the surface alone (no walk), every dry tile open.
   else uniformPagePoolReachablePages(cells,pages,volume,0,()=>true,surface);
   for(const i of pinned)resident.add(i);
  }
 }
 const air=new Set<number>();
 for(const [x,y,z] of record?.air??[]){
  if(!(Number.isInteger(x)&&Number.isInteger(y)&&Number.isInteger(z)&&x>=0&&y>=0&&z>=0&&x<px&&y<py&&z<pz))throw new RangeError(`Uniform page pool: AIR page (${x}, ${y}, ${z}) is outside the box of ${px}x${py}x${pz} pages`);
  air.add(x+px*(y+py*z));
 }
 const keys=resident?[...resident].sort((a,b)=>a-b):Array.from({length:count},(_,i)=>i).filter(i=>!air.has(i));
 const at=(i:number):[number,number,number]=>[i%px,Math.floor(i/px)%py,Math.floor(i/(px*py))];
 const stored=keys.map(at),pins=(live?[...pinned].sort((a,b)=>a-b):keys).map(i=>{const [x,y,z]=at(i);return {x:x+origin[0],y:y+origin[1],z:z+origin[2]};});
 return {live,stored,pins,surface:record?.air?undefined:surface};
}
/** The box pages a live pool pins for its sources' cell boxes, each grown by
 * UNIFORM_PAGE_POOL_PIN_MARGIN: every page once, in the window (the box's
 * low page at origin). */
export function uniformPagePoolSourcePages(cells:readonly [number,number,number],sources:readonly UniformPagePoolCellBox[],origin:readonly [number,number,number]=[0,0,0]):{x:number;y:number;z:number}[]{
 const E=UNIFORM_PAGE_POOL_EDGE,pages:[number,number,number]=[Math.ceil(cells[0]/E),Math.ceil(cells[1]/E),Math.ceil(cells[2]/E)],[px,py,pz]=pages,box={origin,pages,count:px*py*pz},seen=new Uint8Array(box.count);
 return sources.flatMap(s=>cellBoxPages(box,s.low,s.high,UNIFORM_PAGE_POOL_PIN_MARGIN)).filter(({x,y,z})=>{const i=x-origin[0]+px*(y-origin[1]+py*(z-origin[2])),first=!seen[i];seen[i]=1;return first;});
}
/** The box pages (by box index) a wet cell (volume above zero) lies within
 * `margin` cells of on every axis. Each wet page keeps the box of its wet
 * cells, so a page is tested against boxes, not cells: never fewer pages than
 * the cells give. Work and memory follow the wet pages and their margin; by
 * page, only the pages that hold liquid are read. */
function wetPages(cells:readonly [number,number,number],pages:readonly [number,number,number],volume:ArrayLike<number>|LatticePages,margin:number):Set<number>{
 const E=UNIFORM_PAGE_POOL_EDGE,[nx,ny,nz]=cells,[px,py,pz]=pages,bounds=new Map<number,Int32Array>(),flags=new Set<number>();
 // A wet page's cell box: low x, y, z then high x, y, z (inclusive).
 const wet=(p:number,x:number,y:number,z:number)=>{
  let b=bounds.get(p);if(!b)bounds.set(p,b=Int32Array.of(x,y,z,x,y,z));
  if(x<b[0]!)b[0]=x;if(y<b[1]!)b[1]=y;if(z<b[2]!)b[2]=z;
  if(x>b[3]!)b[3]=x;if(y>b[4]!)b[4]=y;if(z>b[5]!)b[5]=z;
 };
 if("edge" in volume){
  if(volume.edge!==E||volume.cells.some((n,a)=>n!==cells[a]))throw new RangeError(`Uniform page pool: the initial liquid's pages are of ${volume.edge} cells over ${volume.cells.join("x")}, not ${E} over the lattice's ${cells.join("x")}`);
  for(const [i,page] of volume.at){
   const ox=E*(i%px),oy=E*(Math.floor(i/px)%py),oz=E*Math.floor(i/(px*py));
   for(let z=0,j=0;z<E;z++)for(let y=0;y<E;y++)for(let x=0;x<E;x++,j++)if(page[j]!>0&&ox+x<nx&&oy+y<ny&&oz+z<nz)wet(i,ox+x,oy+y,oz+z);
  }
 }else{
  if(volume.length!==nx*ny*nz)throw new RangeError(`Uniform page pool: the initial liquid holds ${volume.length} cells, not the lattice's ${nx*ny*nz}`);
  for(let z=0,i=0;z<nz;z++)for(let y=0;y<ny;y++)for(let x=0;x<nx;x++,i++)if(volume[i]!>0)wet(Math.floor(x/E)+px*(Math.floor(y/E)+py*Math.floor(z/E)),x,y,z);
 }
 const reach=Math.ceil(margin/E),gap=(b:Int32Array,k:number,a:number)=>Math.max(b[k]!-(a+E-1),a-b[k+3]!,0);
 for(const [i,b] of bounds){
  const sx=i%px,sy=Math.floor(i/px)%py,sz=Math.floor(i/(px*py));
  for(let z=Math.max(sz-reach,0);z<=Math.min(sz+reach,pz-1);z++)for(let y=Math.max(sy-reach,0);y<=Math.min(sy+reach,py-1);y++)for(let x=Math.max(sx-reach,0);x<=Math.min(sx+reach,px-1);x++)
   if(Math.max(gap(b,0,E*x),gap(b,1,E*y),gap(b,2,E*z))<=margin)flags.add(x+px*(y+py*z));
 }
 return flags;
}
/** The tile offsets the page request (lrAsk) asks for around a wet tile, as
 * (dx, dy, dz, surface only) quads: the transport and pressure stages' footprints
 * (their seeds are the wet tiles), and the surface stage's, whose seeds are the
 * crossings, here the wet tiles beside open dry ones. Particles' footprint is
 * transport's; ownership's is the tile itself; legacy is not staged. */
const WET_FOOTPRINT=(()=>{
 const stage=(name:string)=>RESIDENCY_STAGES.findIndex(s=>s.name===name),wet=[stage("transport"),stage("pressure"),stage("particles")],surface=stage("surface");
 const R=Math.max(...[...wet,surface].map(s=>RESIDENCY_STAGES[s]!.core+RESIDENCY_STAGES[s]!.faces)),out:number[]=[];
 for(let dz=-R;dz<=R;dz++)for(let dy=-R;dy<=R;dy++)for(let dx=-R;dx<=R;dx++){
  const d=[dx,dy,dz];
  if(wet.some(s=>residencyFootprintContains(s,d)))out.push(dx,dy,dz,0);
  else if(residencyFootprintContains(surface,d))out.push(dx,dy,dz,1);
 }
 return out;
})();
/** The box pages (by box index) a 4h tile of which lies within ceil(margin / 4)
 * tile steps (26-connected) of a wet tile (a cell's volume above zero)
 * through open tiles (`openTile`): the margin's reachable air (V2d). A
 * wall's closed tiles stop it, so a page past the wall is not reached. Work
 * and memory follow the wet tiles and the margin's band, not the box.
 * The wet tiles' stage footprints (WET_FOOTPRINT) flag their pages whether
 * open or solid: the request asks for them, so a wall's page within a
 * footprint of the liquid is stored. */
export function uniformPagePoolReachablePages(cells:readonly [number,number,number],pages:readonly [number,number,number],volume:ArrayLike<number>|LatticePages,margin:number,openTile:(x:number,y:number,z:number)=>boolean,surface?:Map<number,number[]>):Set<number>{
 const E=UNIFORM_PAGE_POOL_EDGE,T=E/4,[nx,ny,nz]=cells,[px,py]=pages,flags=new Set<number>();
 const [tx,ty,tz]=[Math.ceil(nx/4),Math.ceil(ny/4),Math.ceil(nz/4)];
 // Each wet page's wet tiles (a byte a tile of the page).
 const wetTiles=new Map<number,Uint8Array>();
 const mark=(x:number,y:number,z:number)=>{
  const i=Math.floor(x/E)+px*(Math.floor(y/E)+py*Math.floor(z/E));let m=wetTiles.get(i);
  if(!m){m=new Uint8Array(T*T*T);wetTiles.set(i,m);}
  m[((x%E)>>2)+T*(((y%E)>>2)+T*((z%E)>>2))]=1;
 };
 if("edge" in volume){
  for(const [i,page] of volume.at){
   const ox=E*(i%px),oy=E*(Math.floor(i/px)%py),oz=E*Math.floor(i/(px*py));
   for(let z=0,j=0;z<E;z++)for(let y=0;y<E;y++)for(let x=0;x<E;x++,j++)if(page[j]!>0&&ox+x<nx&&oy+y<ny&&oz+z<nz)mark(ox+x,oy+y,oz+z);
  }
 }else for(let z=0,i=0;z<nz;z++)for(let y=0;y<ny;y++)for(let x=0;x<nx;x++,i++)if(volume[i]!>0)mark(x,y,z);
 const wet=(x:number,y:number,z:number)=>wetTiles.get(Math.floor(x/T)+px*(Math.floor(y/T)+py*Math.floor(z/T)))?.[x%T+T*(y%T+T*(z%T))]===1;
 const page=(x:number,y:number,z:number)=>Math.floor(x/T)+px*(Math.floor(y/T)+py*Math.floor(z/T));
 // The wet tiles beside a tile that is not wet start the walk.
 let frontier:number[]=[];const seen=new Set<number>(),key=(x:number,y:number,z:number)=>x+tx*(y+ty*z);
 for(const [i,m] of wetTiles){
  flags.add(i);const ox=T*(i%px),oy=T*(Math.floor(i/px)%py),oz=T*Math.floor(i/(px*py));
  for(let c=0;c<m.length;c++){
   if(!m[c])continue;const x=ox+c%T,y=oy+Math.floor(c/T)%T,z=oz+Math.floor(c/(T*T));
   let front=false,air=false;
   edge:for(let dz=-1;dz<=1;dz++)for(let dy=-1;dy<=1;dy++)for(let dx=-1;dx<=1;dx++){
    const X=x+dx,Y=y+dy,Z=z+dz;
    if(X>=0&&Y>=0&&Z>=0&&X<tx&&Y<ty&&Z<tz&&!wet(X,Y,Z)){front=true;if(!surface||openTile(X,Y,Z)){air=true;break edge;}}
   }
   if(front){
    frontier.push(x,y,z);
    // The request's footprints, solid or not: commit 1 claims these pages before it retires any.
    for(let f=0;f<WET_FOOTPRINT.length;f+=4)if(air||!WET_FOOTPRINT[f+3]){
     const X=x+WET_FOOTPRINT[f]!,Y=y+WET_FOOTPRINT[f+1]!,Z=z+WET_FOOTPRINT[f+2]!;
     if(X>=0&&Y>=0&&Z>=0&&X<tx&&Y<ty&&Z<tz)flags.add(page(X,Y,Z));
    }
   }
   // The surface: a wet tile beside open dry tiles (not a wall's), its page's tile bounds.
   if(air&&surface){const b=surface.get(i);if(!b)surface.set(i,[x,y,z,x,y,z]);else{b[0]=Math.min(b[0]!,x);b[1]=Math.min(b[1]!,y);b[2]=Math.min(b[2]!,z);b[3]=Math.max(b[3]!,x);b[4]=Math.max(b[4]!,y);b[5]=Math.max(b[5]!,z);}}
  }
 }
 for(let step=0;step<Math.ceil(margin/4)&&frontier.length;step++){
  const next:number[]=[];
  for(let f=0;f<frontier.length;f+=3)for(let dz=-1;dz<=1;dz++)for(let dy=-1;dy<=1;dy++)for(let dx=-1;dx<=1;dx++){
   const X=frontier[f]!+dx,Y=frontier[f+1]!+dy,Z=frontier[f+2]!+dz;
   if(X<0||Y<0||Z<0||X>=tx||Y>=ty||Z>=tz||wet(X,Y,Z))continue;
   const k=key(X,Y,Z);if(seen.has(k))continue;seen.add(k);
   if(!openTile(X,Y,Z))continue;
   flags.add(page(X,Y,Z));next.push(X,Y,Z);
  }
  frontier=next;
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
  case F.exhausted:return `Page pool exhausted at commit ${commit}: ${page} requested with ${detail} of ${layout.slots} blocks resident and the other ${layout.slots-detail} claimed by the same commit. Raise pagePool.memory_MiB or pagePool.blocks (about 10 MiB a block, UNIFORM_PAGE_BLOCK_MiB) or shrink the scene. The pool rule holds a block for each BULK16 page within v T_hold (the scene's speed over the 30-frame hold) plus the closure margin of the pages stored at build, and for the ballistic columns: a scene that brings the surface to more deep water than that must author pagePool.blocks.`;
  case F.requests:return `More than claimsPerStep (${layout.requests}) pages were requested in commit ${commit} (first dropped: ${page}).`;
  case F.alias:return `Two pages name one page directory entry (${page}, commit ${commit}): the resident set is wider than the directory's ${layout.table} pages (super-blocks of 4 pages ${layout.edge} apart share a top entry), or an allocator bug.`;
  case F.superBlocks:return `Super-block slots exhausted at commit ${commit}: ${page} needed one with ${detail} of ${layout.superBlocks} held before the commit and the rest claimed by it. Raise pagePool.superBlocks (at most ${UNIFORM_SUPER_BLOCK_SLOTS-2}, a slot for each 4x4x4 pages holding a resident page) or shrink the scene.`;
  case F.closureCap:return `Departure travel gives a closure of ${detail} tiles, over the cap of ${layout.closureCap} (commit ${commit}, ${page}): a sample is moving faster than the residency certificate covers.`;
  case F.closure:return `Liquid or band tile found in ${page}, which was not resident at the start of commit ${commit}.`;
  case F.edge:return `Liquid exceeds the moving fluid frame at ${page} (commit ${commit}): residents and requests must fit ${layout.table-4} pages per axis with the window margin; an invalid forced shift or signed coordinate overflow also raises this fatal.`;
  default:return `Page pool fatal bits ${status[offset+S.fatal]} (first ${first}) at commit ${commit}, ${page}.`;
 }
}

const READERS=/* wgsl */`
// The top entry of q's super-block, and q's word in its child table.
fn upEntry(q:vec3i)->u32{let m=vec3u(q>>vec3u(2u))&vec3u(UP_TOP-1u);return m.x|(m.y<<UP_TOP_BITS)|(m.z<<(2u*UP_TOP_BITS));}
fn upLocal(q:vec3i)->u32{let l=vec3u(q)&vec3u(3u);return l.x|(l.y<<2u)|(l.z<<4u);}
fn upChild(slot:u32,q:vec3i)->u32{return UP_TOP_WORDS+64u*min(slot,UP_SUPER_LAST)+upLocal(q);}
fn upSuperStored(slot:u32)->bool{return slot-1u<UP_SUPER_LAST-1u;}
fn upSuperOf(slot:u32)->vec3i{let r=UP_AT_SUPERS+8u*slot;return vec3i(bitcast<i32>(upState[r]),bitcast<i32>(upState[r+1u]),bitcast<i32>(upState[r+2u]));}
// The word of q's page: UP_AIR_WORD, UP_SOLID_WORD or the block resident there (q's own unless a page a table apart holds the word).
fn upWord(q:vec3i)->u32{
 let raw=upState[UP_AT_TABLE+upChild(upState[UP_AT_TABLE+upEntry(q)],q)];if(upPolicy()==0u){return raw;}
 if(upWorldSolid(q)){return UP_SOLID_WORD;}
 return select(raw,UP_AIR_WORD,raw==UP_SOLID_WORD);
}
// A block with a page slot: neither sentinel.
fn upStored(block:u32)->bool{return block-1u<UP_SOLID_BLOCK-1u;}
// UP_RESIDENT, UP_AIR or UP_SOLID: what q's entry is, without the coordinate check.
fn upBulkWord(word:u32)->bool{return (word>>30u)==2u&&(word&0x3fffffffu)!=0u;}
fn upKind(q:vec3i)->u32{let word=upWord(q);return select(select(select(UP_RESIDENT,UP_SOLID,word==UP_SOLID_WORD),UP_AIR,word==UP_AIR_WORD),UP_BULK,upBulkWord(word));}
fn upPageOf(block:u32)->vec3i{let r=UP_AT_RECORDS+8u*block;return vec3i(bitcast<i32>(upState[r]),bitcast<i32>(upState[r+1u]),bitcast<i32>(upState[r+2u]));}
// The block a read of q resolves to, without the coordinate check: block 0 for AIR, UP_SOLID_BLOCK for SOLID.
fn upBlockWrapped(q:vec3i)->u32{return min(upWord(q),UP_SOLID_BLOCK);}
// The block that holds page q itself, else UP_NONE.
fn upBlockOf(q:vec3i)->u32{let block=upWord(q);if(!upStored(block)){return UP_NONE;}return select(UP_NONE,block,all(upPageOf(block)==q));}
// The block's word: the window page of the page it holds (upWindowPage), or UP_FREE.
fn upBlockWord(block:u32)->u32{return upState[UP_AT_BLOCKS+block];}
fn upWindowPage(word:u32)->vec3i{return vec3i((vec3u(word&255u,(word>>8u)&255u,(word>>16u)&255u)-bitcast<vec3u>(upFramePage()))&vec3u(UP_TABLE-1u));}
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
fn upKeep(block:u32){if(block<UP_BLOCKS){atomicMax(&upWork[UP_WORK_KEEP+block],1u);}}
// The resident block's page leaves the 4h pool for BULK16 slot word \`word\`
// (BULK16_PAGE_TAG|slot): the next commit writes the word over the page's
// entry and retires the block (upSlots). It outranks a keep.
fn upDemote(block:u32,word:u32){if(upStored(block)&&upBulkWord(word)){atomicMax(&upWork[UP_WORK_KEEP+block],word);}}
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
 let held=upWord(q);
 if(held==UP_SOLID_WORD){return;}
 upInclude(q);
 // A BULK16 page is liquid held at 16h: only upPromote gives it a block.
 if(upBulkWord(held)){return;}
 if(held!=UP_AIR_WORD&&all(upPageOf(held)==q)){upKeep(held);return;}
 upList(q);
}
// Ask for BULK16 page q to take a block again at the next commit (its slot
// is the caller's to prolong and retire once the commit lists the block).
fn upPromote(q:vec3i){if(upBulkWord(upWord(q))){upList(q);}}
// List absent page q once for the next commit's claims.
fn upList(q:vec3i){
 // The super-block's wrap tags claim its top entry; its page's bit lists it once.
 let tag=(vec3u(q>>vec3u(UP_TABLE_BITS))<<vec3u(1u))|vec3u(1u);
 let c=UP_WORK_CLAIMS+5u*upEntry(q);
 let x=upClaimWord(c,tag.x);let y=upClaimWord(c+1u,tag.y);let z=upClaimWord(c+2u,tag.z);
 if(x==2u||y==2u||z==2u){upFail(select(UP_FATAL_ALIAS,UP_FATAL_EDGE,upPolicy()!=0u),q);return;}
 let l=upLocal(q);let bit=1u<<(l&31u);
 if((atomicOr(&upWork[c+3u+(l>>5u)],bit)&bit)!=0u){return;}
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
const UP_TOP:u32=${l.edge}u;const UP_TOP_BITS:u32=${l.edgeBits}u;const UP_TOP_WORDS:u32=${l.entries}u;const UP_SUPER_LAST:u32=${UNIFORM_SUPER_BLOCK_SLOTS-1}u;const UP_AT_SUPERS:u32=${l.state.supers}u;
const UP_AIR_WORD:u32=0u;const UP_SOLID_WORD:u32=0xffffffffu;const UP_FREE:u32=0xffffffffu;
const UP_RESIDENT:u32=${K.resident}u;const UP_AIR:u32=${K.air}u;const UP_SOLID:u32=${K.solid}u;const UP_BULK:u32=${K.bulk}u;
const UP_TABLE_BITS:u32=${l.tableBits}u;const UP_REQUESTS:u32=${l.requests}u;const UP_CLOSURE_CAP:u32=${l.closureCap}u;
const UP_AT_WINDOW:u32=${l.state.window}u;const UP_WORK_BOUNDS:u32=${l.work.bounds}u;
const UP_AT_TABLE:u32=${l.state.table}u;const UP_AT_BLOCKS:u32=${l.state.blocks}u;const UP_AT_RECORDS:u32=${l.state.records}u;const UP_AT_RESIDENT:u32=${l.state.resident}u;const UP_AT_NEW:u32=${l.state.claimed}u;const UP_AT_RETIRED:u32=${l.state.retired}u;
const UP_WORK_LIST:u32=${l.work.list}u;const UP_WORK_KEEP:u32=${l.work.keep}u;const UP_WORK_CLAIMS:u32=${l.work.claims}u;const UP_WORK_DETAIL:u32=${UNIFORM_PAGE_POOL_WORK.detail}u;
const UP_FATAL_EXHAUSTED:u32=${F.exhausted}u;const UP_FATAL_ALIAS:u32=${F.alias}u;const UP_FATAL_REQUESTS:u32=${F.requests}u;const UP_FATAL_CLOSURE_CAP:u32=${F.closureCap}u;const UP_FATAL_CLOSURE:u32=${F.closure}u;const UP_FATAL_EDGE:u32=${F.edge}u;const UP_FATAL_SUPER:u32=${F.superBlocks}u;
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
 * that binding of the same group, the requesters upRequest and upKeep, the
 * BULK16 transitions upDemote and upPromote, and
 * upFail / upFailWith with the bits UP_FATAL_*. */
export function uniformPagePoolWGSL(layout:UniformPagePoolLayout,group:number,state:number,work?:number):string{
 return source(layout,group,state,work,"read");
}
/** A buffer that carries the tail (layout.tail) at word `offset`: the commit
 * kernels bind it writable, the seed is uploaded to it. An optional
 * headerUniform (offset in bytes) receives a copy of the window header. */
export interface UniformPagePoolMirror {readonly buffer:GPUBuffer;readonly offset:number;readonly headerUniform?:{readonly buffer:GPUBuffer;readonly offset:number}}

/** A buffer's words rounded up to a whole 8 bytes, and an array padded to
 * them (UniformPagePool's buffers: see its constructor). */
const evenWords=(words:number):number=>words+(words&1);
const evenArray=(a:Uint32Array<ArrayBuffer>):Uint32Array<ArrayBuffer>=>{if(!(a.length&1))return a;const b=new Uint32Array(a.length+1);b.set(a);return b;};
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
 /** Changes of the stored block set the host has encoded: one a commit
  * (its resident and retired lists name every block it changed), two an
  * upload (a seed may change any block). UniformLiveVisit reads it. */
 private changes=0;
 get liveRevision():number{return this.changes;}
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
 private claimSuperPipeline?:GPUComputePipeline;
 private claimPipeline?:GPUComputePipeline;
 private slotsPipeline?:GPUComputePipeline;
 private superSlotsPipeline?:GPUComputePipeline;
 private flatPipeline?:GPUComputePipeline;
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
  // Whole 8-byte sizes: Dawn (Metal) aborts a writeBuffer past 4 MiB whose
  // size is an odd number of words (UploadData; a 33-block pool's work
  // buffer, 1,319,637 words), and upload() writes each buffer whole.
  this.state=device.createBuffer({label:"Uniform page pool state",size:4*evenWords(l.state.words),usage});
  this.work=device.createBuffer({label:"Uniform page pool work",size:4*evenWords(l.work.words),usage});
  if(l.published.words){
   this.published=device.createBuffer({label:"Uniform page pool published pages",size:4*evenWords(l.published.words),usage});
   this.volumePageSource={records:{buffer:this.published},directory:{buffer:this.state,offset:4*l.state.table,size:4*(l.directory+UNIFORM_WINDOW_HEADER_WORDS)}};
  }
  this.allocatedBytes=this.state.size+this.work.size+(this.published?.size??0);
  this.host=new Int32Array(4*l.host);this.pins=new Int32Array(4*l.pins);
  // Publication visits blocks; the address-window directory is copied only by a visible overlay.
  this.hostGroups=Math.min(Math.ceil(Math.max(l.host,l.pins)/64),16);this.claimGroups=Math.min(Math.ceil(l.requests/64),16);this.publishGroups=Math.ceil(l.blocks/64);
  this.current=uniformPagePoolSeed(l,[]);this.upload();
 }
 /** The host copy of the pool at generation 0: the last seed, or the empty pool. */
 get seeded():UniformPagePoolSeed{return this.current;}
 /** Replace the whole pool with a seed built for this layout
  * (uniformPagePoolSeed): both buffers, the published record and the tail
  * of every mirror, at generation 0 with no fatal, request, keep or host
  * command pending. The pinned pages stay. Scene build only: it allocates. */
 seed(seed:UniformPagePoolSeed):void{
  const l=this.layout,s=seed.layout;
  if(s.table!==l.table||s.blocks!==l.blocks||s.superBlocks!==l.superBlocks||seed.state.length!==l.state.words||seed.ring.length!==l.ringCapacity)throw new Error(`Uniform page pool seed is of table ${s.table} and ${s.blocks} blocks; the pool is of table ${l.table} and ${l.blocks} blocks`);
  this.current=seed;this.hostCount=0;this.upload();
 }
 private upload():void{
  const l=this.layout,seed=this.current,queue=this.device.queue,work=new Uint32Array(evenWords(l.work.words));
  this.changes+=2;
  work.fill(0xffffffff,l.work.bounds,l.work.bounds+3);
  work.set(seed.ring,l.work.ring);work.set(seed.superRing,l.work.superRing);work.set(seed.superCount,l.work.counts);work[UNIFORM_PAGE_POOL_WORK.pins]=this.pinCount;work.set(new Uint32Array(this.pins.buffer,0,4*this.pinCount),l.work.pins);
  queue.writeBuffer(this.state,0,evenArray(seed.state));queue.writeBuffer(this.work,0,work);
  if(this.published)queue.writeBuffer(this.published,0,evenArray(uniformPagePoolPublished(seed,l)));
  for(const m of this.mirrors){
   queue.writeBuffer(m.buffer,4*m.offset,seed.tail);
   if(m.headerUniform)queue.writeBuffer(m.headerUniform.buffer,m.headerUniform.offset,seed.state.subarray(l.state.window,l.state.window+UNIFORM_WINDOW_HEADER_WORDS));
  }
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
  * and costs the commit one storage binding. Optional headerUniform receives
  * the 32-word runtime window on initialize, seed and after each commit. */
 async initialize(mirrors:readonly UniformPagePoolMirror[]=[],frameParams?:GPUBuffer):Promise<void>{
  const device=this.device,l=this.layout,W=UNIFORM_PAGE_POOL_WORK,S=UNIFORM_PAGE_POOL_STATUS,R=UNIFORM_PAGE_POOL_RECORD,B=UNIFORM_PAGE_POOL_BLOCK;
  if(this.started)throw new Error("Uniform page pool is already initialized");
  const most=Math.min(UNIFORM_PAGE_POOL_MIRRORS,device.limits.maxStorageBuffersPerShaderStage-2-(frameParams?1:0)),needed=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST;
  if(mirrors.length>most)throw new RangeError(`Uniform page pool takes ${most} mirrors at most: ${mirrors.length}`);
  mirrors.forEach((m,i)=>{
   if(!Number.isSafeInteger(m.offset)||m.offset<0||4*(m.offset+l.tail.words)>m.buffer.size)throw new RangeError(`Uniform page pool mirror ${i}: ${l.tail.words} words at word ${m.offset} do not fit its buffer of ${m.buffer.size} bytes`);
   if((m.buffer.usage&needed)!==needed)throw new Error(`Uniform page pool mirror ${i}: its buffer needs STORAGE and COPY_DST usage`);
   if(m.headerUniform){
    const h=m.headerUniform;
    if(!Number.isSafeInteger(h.offset)||h.offset<0||h.offset%4||h.offset+4*UNIFORM_WINDOW_HEADER_WORDS>h.buffer.size)throw new RangeError(`Uniform page pool mirror ${i}: header does not fit its target`);
    if(!(h.buffer.usage&GPUBufferUsage.COPY_DST))throw new Error(`Uniform page pool mirror ${i}: header target needs COPY_DST usage`);
    if(h.buffer===this.state||h.buffer===this.work||h.buffer===this.published||mirrors.some(o=>o.buffer===h.buffer))throw new Error(`Uniform page pool mirror ${i}: header target aliases pool storage`);
    if(mirrors.slice(0,i).some(o=>o.headerUniform?.buffer===h.buffer&&o.headerUniform.offset<h.offset+4*UNIFORM_WINDOW_HEADER_WORDS&&h.offset<o.headerUniform.offset+4*UNIFORM_WINDOW_HEADER_WORDS))throw new Error(`Uniform page pool mirror ${i}: header targets overlap`);
   }
   if(m.buffer===this.state||m.buffer===this.work||mirrors.findIndex(o=>o.buffer===m.buffer)!==i)throw new Error(`Uniform page pool mirror ${i}: its buffer is bound to the commit already`);
  });
  this.started=true;this.mirrors=mirrors.map(({buffer,offset,headerUniform})=>({buffer,offset,headerUniform:headerUniform?{...headerUniform}:undefined}));
  for(const m of this.mirrors){
   device.queue.writeBuffer(m.buffer,4*m.offset,this.current.tail);
   if(m.headerUniform)device.queue.writeBuffer(m.headerUniform.buffer,m.headerUniform.offset,this.current.state.subarray(l.state.window,l.state.window+UNIFORM_WINDOW_HEADER_WORDS));
  }
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
   .replace(`hi-lo>=${UNIFORM_WINDOW_PAGES-2*UNIFORM_MOVING_WINDOW_MARGIN}u`,`hi-lo>=${(l.wide?UNIFORM_WINDOW_PAGES:UNIFORM_MOVING_WINDOW_INNER)-2*UNIFORM_MOVING_WINDOW_MARGIN}u`)
   .replace("@compute @workgroup_size(1) fn mwDecide()", "fn upDecide()")
   .replace("let old=", "if(upPolicy()==0u){return;}let old=")
   // A finite tank already has walls on every side; only the probe moves it.
   .replace("if(forced){", "if(upPolicy()==1u||(a==1u&&upPolicy()==3u&&lo==((upState[UP_AT_WINDOW+14u])^0x80000000u)&&hi>=old[a]&&hi-old[a]<span-2u)){next[a]=old[a];}if(forced){");
  if(!windowBody.includes(`hi-lo>=${(l.wide?UNIFORM_WINDOW_PAGES:UNIFORM_MOVING_WINDOW_INNER)-2*UNIFORM_MOVING_WINDOW_MARGIN}u`))throw new Error("Uniform page pool: the window decision's width test moved");
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
const S_SUPER_RESIDENT:u32=${S.superResident}u;const S_SUPER_FREE:u32=${S.superFree}u;const S_SUPER_PENDING:u32=${S.superPending}u;
const W_POPS:u32=${W.pops}u;const W_HEAD:u32=${W.head}u;const W_RING:u32=${l.work.ring}u;const RING_MASK:u32=${l.ringCapacity-1}u;
const W_SUPER_POPS:u32=${W.superPops}u;const W_SUPER_HEAD:u32=${W.superHead}u;const W_SUPER_RING:u32=${l.work.superRing}u;const W_SUPER_COUNT:u32=${l.work.counts}u;const SUPER_MASK:u32=${UNIFORM_SUPER_BLOCK_SLOTS-1}u;
const R_STATE:u32=${R.state}u;const R_GENERATION:u32=${R.generation}u;const R_IDLE:u32=${R.idle}u;const U_BACKGROUND:u32=${UNIFORM_PAGE_POOL_SUPER.background}u;
const FREE:u32=${B.free}u;const RESIDENT:u32=${B.resident}u;const PENDING:u32=${B.pending}u;const FIXED:u32=${B.fixed}u;
const IDLE_STEPS:u32=${l.idleSteps}u;const CLAIM_WIDTH:u32=${64*this.claimGroups}u;
// The only stores of directory and block words: to the state and to every mirror.
fn upPutEntry(i:u32,word:u32){upState[UP_AT_TABLE+i]=word;${puts(l.tail.table)}}
fn upPutBlock(i:u32,word:u32){upState[UP_AT_BLOCKS+i]=word;${puts(l.tail.blocks)}}
fn upPutWindow(i:u32,word:u32){upState[UP_AT_WINDOW+i]=word;${puts(l.tail.window)}}
fn upPutFlat(i:u32,word:u32){upState[UP_AT_TABLE+${l.tail.flat}u+i]=word;${puts(l.tail.flat)}}
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
fn upBlockWordOf(q:vec3i)->u32{let m=vec3u(q)&vec3u(UP_TABLE-1u);return m.x|(m.y<<8u)|(m.z<<16u);}
fn upListed(i:u32)->vec3i{let a=UP_WORK_LIST+4u*i;return vec3i(bitcast<i32>(atomicLoad(&upWork[a])),bitcast<i32>(atomicLoad(&upWork[a+1u])),bitcast<i32>(atomicLoad(&upWork[a+2u])));}
// q's top entry holds a slot for q's super-block: its own, or a fixed one, which every super-block of the entry shares.
fn upSuperHolds(slot:u32,q:vec3i)->bool{return upSuperStored(slot)&&(upState[UP_AT_SUPERS+8u*slot+R_STATE]==FIXED||all(upSuperOf(slot)==(q>>vec3u(2u))));}
// upClaimSuper: the first listed request of each top entry clears the
// entry's claim words and, where the entry holds no slot, pops one off the
// super ring for the request's super-block: its children start as the top
// word it replaces. An entry held by another super-block is an alias.
@compute @workgroup_size(64) fn upClaimSuper(@builtin(global_invocation_id) gid:vec3u){
 let asked=atomicLoad(&upWork[0u]);
 // A list that overflowed left claim words no listed request clears.
 if(asked>UP_REQUESTS){for(var w=gid.x;w<${5*l.entries}u;w+=CLAIM_WIDTH){atomicStore(&upWork[UP_WORK_CLAIMS+w],0u);}}
 let count=min(asked,UP_REQUESTS);let generation=upState[0u]+1u;
 let head=atomicLoad(&upWork[W_SUPER_HEAD]);let available=upState[S_SUPER_FREE]+upState[S_SUPER_PENDING];
 for(var i=gid.x;i<count;i+=CLAIM_WIDTH){
  let q=upListed(i);let e=upEntry(q);let c=UP_WORK_CLAIMS+5u*e;
  if(atomicExchange(&upWork[c],0u)==0u){continue;}
  for(var k=1u;k<5u;k++){atomicStore(&upWork[c+k],0u);}
  if(atomicLoad(&upWork[2u])!=0u){continue;}
  let top=upState[UP_AT_TABLE+e];
  if(upSuperStored(top)){if(!upSuperHolds(top,q)){upFail(select(UP_FATAL_ALIAS,UP_FATAL_EDGE,upPolicy()!=0u),q);}continue;}
  let pop=atomicAdd(&upWork[W_SUPER_POPS],1u);
  if(pop>=available){upFailWith(UP_FATAL_SUPER,q,upState[S_SUPER_RESIDENT]);continue;}
  let slot=atomicLoad(&upWork[W_SUPER_RING+((head+pop)&SUPER_MASK)]);let r=UP_AT_SUPERS+8u*slot;let sb=q>>vec3u(2u);
  upState[r]=bitcast<u32>(sb.x);upState[r+1u]=bitcast<u32>(sb.y);upState[r+2u]=bitcast<u32>(sb.z);
  upState[r+R_STATE]=RESIDENT;upState[r+R_GENERATION]=generation;upState[r+U_BACKGROUND]=top;
  for(var k=0u;k<64u;k++){upPutEntry(UP_TOP_WORDS+64u*slot+k,top);}
  upPutEntry(e,slot);
 }
}
// upClaim: one block per listed request, popped off the ring head; the
// blocks free or quarantined before this commit are all claimable. Each
// request owns its page word, its block's words and its new-list word. A
// request whose page is SOLID by now, or whose super-block has no slot, is
// dropped.
@compute @workgroup_size(64) fn upClaim(@builtin(global_invocation_id) gid:vec3u){
 let count=min(atomicLoad(&upWork[0u]),UP_REQUESTS);let generation=upState[0u]+1u;
 let head=atomicLoad(&upWork[W_HEAD]);let available=upState[S_FREE]+upState[S_PENDING];
 for(var i=gid.x;i<count;i+=CLAIM_WIDTH){
  let q=upListed(i);let slot=upState[UP_AT_TABLE+upEntry(q)];
  let held=upWord(q);
  if(held==UP_SOLID_WORD||atomicLoad(&upWork[2u])!=0u||!upSuperHolds(slot,q)){continue;}
  // A promoted BULK16 page already counts in its super-block (upSlots kept it there).
  let bulk=upBulkWord(held);
  if(held!=UP_AIR_WORD&&!bulk){if(any(upPageOf(held)!=q)){upFail(UP_FATAL_ALIAS,q);}continue;}
  let pop=atomicAdd(&upWork[W_POPS],1u);
  if(pop>=available){upFailWith(UP_FATAL_EXHAUSTED,q,upState[${S.resident}u]);continue;}
  let block=atomicLoad(&upWork[W_RING+((head+pop)&RING_MASK)]);let r=UP_AT_RECORDS+8u*block;
  upState[r]=bitcast<u32>(q.x);upState[r+1u]=bitcast<u32>(q.y);upState[r+2u]=bitcast<u32>(q.z);
  upState[r+R_STATE]=RESIDENT;upState[r+R_GENERATION]=generation;upState[r+R_IDLE]=0u;
  upPutEntry(upChild(slot,q),block);if(!bulk){atomicAdd(&upWork[W_SUPER_COUNT+slot],1u);}
  upPutBlock(block,upBlockWordOf(q));upState[UP_AT_NEW+pop]=block;
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
   // A demoted page: its entry takes the BULK16 word, which keeps the page
   // in its super-block's count, and the block retires.
   if(upBulkWord(kept)){
    let q=upPageOf(s);let at=upChild(upState[UP_AT_TABLE+upEntry(q)],q);
    if(upState[UP_AT_TABLE+at]==s){upPutEntry(at,kept);}
    upPutBlock(s,UP_FREE);upState[r+R_STATE]=PENDING;count.y+=1u;continue;
   }
   let idle=select(upState[r+R_IDLE]+1u,0u,kept!=0u);upState[r+R_IDLE]=idle;
   if(idle>=IDLE_STEPS){
    // Only a word that names this block is cleared; its slot counts one page fewer.
    let q=upPageOf(s);let slot=upState[UP_AT_TABLE+upEntry(q)];let at=upChild(slot,q);
    if(upState[UP_AT_TABLE+at]==s){upPutEntry(at,UP_AIR_WORD);}
    if(upSuperStored(slot)){atomicSub(&upWork[W_SUPER_COUNT+slot],1u);}
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
  else if(state==PENDING){atomicStore(&upWork[W_RING+((head+available+at.y)&RING_MASK)],s);upState[UP_AT_RETIRED+at.y]=s;at.y+=1u;}
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
}
// upSuperSlots: one workgroup over every super-block slot, after upSlots.
// Frees the slots the last commit quarantined and retires the resident ones
// left without a page, their top entry back to the background; then queues
// the retired behind the super ring's free slots and seals the super status.
const SUPER_PER:u32=${UNIFORM_SUPER_BLOCK_SLOTS/256}u;
@compute @workgroup_size(256) fn upSuperSlots(@builtin(local_invocation_index) lane:u32){
 let first=lane*SUPER_PER;let last=first+SUPER_PER;
 let head=atomicLoad(&upWork[W_SUPER_HEAD]);let available=upState[S_SUPER_FREE]+upState[S_SUPER_PENDING];
 // x: held, y: retired.
 var count=vec2u(0u);
 for(var s=first;s<last;s++){
  let r=UP_AT_SUPERS+8u*s;let state=upState[r+R_STATE];
  if(state==PENDING){upState[r+R_STATE]=FREE;continue;}
  if(state==FIXED){count.x+=1u;continue;}
  if(state!=RESIDENT){continue;}
  if(atomicLoad(&upWork[W_SUPER_COUNT+s])==0u){
   let e=upEntry(4*upSuperOf(s));if(upState[UP_AT_TABLE+e]==s){upPutEntry(e,upState[r+U_BACKGROUND]);}
   upState[r+R_STATE]=PENDING;count.y+=1u;continue;
  }
  count.x+=1u;
 }
 upScan[lane]=count;workgroupBarrier();
 for(var stride=1u;stride<LANES;stride*=2u){
  var add=vec2u(0u);if(lane>=stride){add=upScan[lane-stride];}
  workgroupBarrier();upScan[lane]+=add;workgroupBarrier();
 }
 var at=upScan[lane]-count;
 for(var s=first;s<last;s++){
  if(upState[UP_AT_SUPERS+8u*s+R_STATE]==PENDING){atomicStore(&upWork[W_SUPER_RING+((head+available+at.y)&SUPER_MASK)],s);at.y+=1u;}
 }
 if(lane==LANES-1u){
  let pops=min(atomicLoad(&upWork[W_SUPER_POPS]),available);
  upState[S_SUPER_RESIDENT]=at.x;upState[S_SUPER_FREE]=available-pops;upState[S_SUPER_PENDING]=at.y;
  atomicStore(&upWork[W_SUPER_HEAD],(head+pops)&SUPER_MASK);atomicStore(&upWork[W_SUPER_POPS],0u);
 }
}
// upFlat: the flat window under the frame upWindow left, after the commit's
// last directory write: one lane a word.
@compute @workgroup_size(64) fn upFlat(@builtin(global_invocation_id) gid:vec3u){
 let i=gid.x+${64*FLAT_ROW}u*gid.y;let q=vec3i(vec3u(i&63u,(i>>6u)&63u,i>>12u))-vec3i(1)+upFramePage();
 upPutFlat(i,upState[UP_AT_TABLE+upChild(upState[UP_AT_TABLE+upEntry(q)],q)]);
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
  [this.windowPipeline,this.hostPipeline,this.claimSuperPipeline,this.claimPipeline,this.slotsPipeline,this.superSlotsPipeline,this.flatPipeline,this.publishPipeline]=await Promise.all([
   device.createComputePipelineAsync({label:"Uniform page pool upWindow",layout:writing,compute:{module:commitModule,entryPoint:"upWindow"}}),
   device.createComputePipelineAsync({label:"Uniform page pool upHost",layout:reading,compute:{module:hostModule,entryPoint:"upHost"}}),
   device.createComputePipelineAsync({label:"Uniform page pool upClaimSuper",layout:writing,compute:{module:commitModule,entryPoint:"upClaimSuper"}}),
   device.createComputePipelineAsync({label:"Uniform page pool upClaim",layout:writing,compute:{module:commitModule,entryPoint:"upClaim"}}),
   device.createComputePipelineAsync({label:"Uniform page pool upSlots",layout:writing,compute:{module:commitModule,entryPoint:"upSlots"}}),
   device.createComputePipelineAsync({label:"Uniform page pool upSuperSlots",layout:writing,compute:{module:commitModule,entryPoint:"upSuperSlots"}}),
   device.createComputePipelineAsync({label:"Uniform page pool upFlat",layout:writing,compute:{module:commitModule,entryPoint:"upFlat"}}),
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
  this.changes++;
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
  pass.setPipeline(this.claimSuperPipeline!);pass.dispatchWorkgroups(this.claimGroups);
  pass.setPipeline(this.claimPipeline!);pass.dispatchWorkgroups(this.claimGroups);
  pass.setPipeline(this.slotsPipeline);pass.dispatchWorkgroups(1);
  pass.setPipeline(this.superSlotsPipeline!);pass.dispatchWorkgroups(1);
  pass.setPipeline(this.flatPipeline!);pass.dispatchWorkgroups(FLAT_ROW,UNIFORM_DIRECTORY_FLAT_WORDS/(64*FLAT_ROW));
  if(this.publishPipeline){pass.setPipeline(this.publishPipeline);pass.setBindGroup(0,this.publisher!);pass.dispatchWorkgroups(this.publishGroups);}
  pass.end();
  // Queue-ordered publication uses the existing uniform binding and no new compute pass.
  for(const m of this.mirrors)if(m.headerUniform)encoder.copyBufferToBuffer(this.state,4*this.layout.state.window,m.headerUniform.buffer,m.headerUniform.offset,4*UNIFORM_WINDOW_HEADER_WORDS);
 }
 /** Copy the status (UNIFORM_PAGE_POOL_STATUS.words words) to `target` at
  * `offset` bytes: the host's only view of the pool, to map when it likes. */
 encodeReceipt(encoder:GPUCommandEncoder,target:GPUBuffer,offset:number):void{
  encoder.copyBufferToBuffer(this.state,0,target,offset,4*UNIFORM_PAGE_POOL_STATUS.words);
 }
 destroy():void{this.state.destroy();this.work.destroy();this.published?.destroy();}
}
/** The pool rule (docs/plans/bulk16-plan-2026-10-28.md, M4b step 3): the
 * census (the box pages resident at build, `stored`, each once) plus the
 * ballistic column reserve, plus the two sentinel blocks. The
 * reserve is, for each page column holding a resident page, the pages above
 * its highest resident one up to `reachPage` (the page holding the census
 * liquid's top plus its ballistic head, uniformCensusReach), at most
 * `ceilingPage` (the box's top page under a closed top, the frame's under an
 * open one), and at least the one block a pool holds besides its sentinels
 * (uniformPagePoolLayout's minimum: a box with no liquid at build). Exhaustion
 * stays the pool's fatal. Work and memory follow the stored pages. */
export function uniformPagePoolRuleBlocks(pages:readonly [number,number,number],stored:readonly (readonly [number,number,number])[],reachPage:number,ceilingPage:number):number{
 const reach=Math.min(Math.floor(reachPage),ceilingPage);let reserve=0;
 for(const top of columnTops(pages,stored).values())reserve+=Math.max(0,reach-top);
 return Math.max(1,stored.length+reserve)+2;
}
/** Each stored page column's highest stored page, by column (x + pages.x z). */
function columnTops(pages:readonly [number,number,number],stored:readonly (readonly [number,number,number])[]):Map<number,number>{
 const tops=new Map<number,number>();
 for(const [x,y,z] of stored){const c=x+pages[0]*z;tops.set(c,Math.max(tops.get(c)??-1,y));}
 return tops;
}
/** The pool rule's pages (uniformPagePoolRuleBlocks without the sentinels)
 * by box page index: the census pages (1) and the reserve above each
 * column's top one (2), the reserve's pages past the box's top (an open top)
 * not held. */
export function uniformPagePoolRulePages(pages:readonly [number,number,number],stored:readonly (readonly [number,number,number])[],reachPage:number,ceilingPage:number):Map<number,1|2>{
 const [px,py]=pages,flags=new Map<number,1|2>();
 for(const [x,y,z] of stored)flags.set(x+px*(y+py*z),1);
 const reach=Math.min(Math.floor(reachPage),ceilingPage,py-1);
 for(const [c,top] of columnTops(pages,stored)){
  const x=c%px,z=(c-x)/px;
  for(let y=top+1;y<=reach;y++)flags.set(x+px*(y+py*z),2);
 }
 return flags;
}
/** The tiles of the box's outer layer of 4h tiles (those beside its walls)
 * that lie in a held page (`flags`'s keys, box page indices): `tiles`, the box's
 * tiles an axis, UNIFORM_PAGE_POOL_EDGE / 4 a page edge. */
export function uniformBoxShellTilesIn(pages:readonly [number,number,number],flags:ReadonlyMap<number,number>,tiles:readonly [number,number,number]):number{
 const T=UNIFORM_PAGE_POOL_EDGE/4,[px,py]=pages;
 // On axis a, a page's tiles inside the box, and those of them off the box's two outer layers.
 const span=(a:number,p:number)=>{const lo=T*p,hi=Math.min(T*(p+1),tiles[a]!);return [Math.max(0,hi-lo),Math.max(0,Math.min(hi,tiles[a]!-1)-Math.max(lo,1))];};
 let n=0;
 for(const i of flags.keys()){
  const x=i%px,y=Math.floor(i/px)%py,z=Math.floor(i/(px*py));
  const [ax,ix]=span(0,x),[ay,iy]=span(1,y),[az,iz]=span(2,z);n+=ax*ay*az-ix*iy*iz;
 }
 return n;
}
