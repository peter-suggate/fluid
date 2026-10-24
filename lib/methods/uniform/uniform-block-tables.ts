import { UNIFORM_PAGE_WINDOW } from "./uniform-lattice-address";
import { UNIFORM_PAGE_POOL_FREE } from "./uniform-page-pool";

/** Per-block tables: what a kernel asks about a box of tiles, kept by block.
 *
 * A table holds a fixed number of words for every block of the pool, at the
 * block's index, so its size follows the pool and nothing in it knows the
 * scene's box. A query takes an inclusive tile box in lattice tiles, holds it
 * inside the 64-page window (a page outside the window is SOLID and stores
 * nothing) and visits the pages the box overlaps through the page table, a
 * word a page. Two tables:
 *
 *  tile bits     one bit a tile key, asked "is a bit set in this box"
 *                (uniformTileBitsWGSL): 16 words of bits, key order; a slabs
 *                word, bit z set where the block's z-slab z holds a bit; an
 *                around word, bit i set where the page at row word i of the
 *                block's neighbour row (uniformBlockNeighbourWGSL) holds a
 *                bit. An AIR page holds none, or (air) counts as a set bit.
 *  bound cubes   six ordered keys (three minima, three maxima) over aligned
 *                cubes of a block's tiles (uniformBoundCubesWGSL): 64 cubes of
 *                side 2, 8 of side 4 and the page, six words each.
 *
 * The including module declares the page table and block records
 * (uniformLatticeAddressWGSL), the block neighbour rows where it asks for tile
 * bits, and umDispatchX. Every function here has a host twin of the same
 * integer arithmetic below. */

/** Words a block of a tile-bit table. */
export const UNIFORM_TILE_BITS_WORDS = 18;
/** Words a block of a bound-cube table: 73 cubes of six keys. */
export const UNIFORM_BOUND_CUBE_WORDS = 438;
/** Pages a box query visits through the page table before it walks the blocks instead. */
const BOX_PAGES = 64;

export interface UniformTileBitsSource {
  /** Prefix of every name this emits: nameIn, nameAny, nameBits, nameBlocks. */
  readonly name: string;
  /** The table word `index` of block `block`, both WGSL u32 expressions. */
  readonly word: (block: string, index: string) => string;
  /** A statement storing `value` there. */
  readonly store: (block: string, index: string, value: string) => string;
  /** A bool expression: the bit of the live tile key `tile`. */
  readonly flag: (tile: string) => string;
  /** An AIR page counts as a set bit, and a box of more than 64 pages as holding one. */
  readonly air: boolean;
}

/** A tile-bit table's two build entries and its query. Both stride the
 * pool's resident list (uniform-live-lists, which the module includes) over
 * any fixed grid.
 *
 * nameBits: a workgroup a z-slab of one resident block writes the block's 16
 * bit words. nameBlocks: a lane a resident block writes its slabs and around
 * words from them and the neighbour rows. A free block's words are stale and
 * no query reads them.
 *
 * nameAny(home, low, high): a bit is set in the inclusive tile box. home is a
 * stored block: where the box lies in its page and the 26 around it, the
 * around word answers an empty box in one read and names the pages worth a
 * visit. A larger box reads the page table, and one of more than 64 pages
 * walks the block records instead (air: it holds a bit). */
export function uniformTileBitsWGSL(o: UniformTileBitsSource): string {
  const n = o.name;
  return /* wgsl */ `
var<workgroup> ${n}Slab:array<atomic<u32>,2>;
@compute @workgroup_size(64) fn ${n}Bits(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 for(var slab=group.x;slab<8u*ulResidentCount();slab+=groups.x){
 let first=512u*ulResidentBlock(slab>>3u)+64u*(slab&7u);let tile=first+lane;
 if(lane<2u){atomicStore(&${n}Slab[lane],0u);}
 workgroupBarrier();
 if(umTileLive(tile)){if(${o.flag("tile")}){atomicOr(&${n}Slab[lane>>5u],1u<<(lane&31u));}}
 workgroupBarrier();
 // The slab's two words, by its first two lanes.
 if(lane<2u){${o.store("(first>>9u)", "(((first&511u)>>5u)+lane)", `atomicLoad(&${n}Slab[lane])`)}}
 }
}
@compute @workgroup_size(64) fn ${n}Blocks(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) groups:vec3u){
 for(var i=gid.x;i<ulResidentCount();i+=64u*groups.x){${n}Block(ulResidentBlock(i));}
}
fn ${n}Block(b:u32){
 if(umBlockWord(b)==UM_FREE){return;}
 var slabs=0u;
 for(var z=0u;z<8u;z++){if((${o.word("b", "(2u*z)")}|${o.word("b", "(2u*z+1u)")})!=0u){slabs|=1u<<z;}}
 var around=0u;
 for(var i=0u;i<27u;i++){
  let w=umBlockNeighbourWord(b,i);${o.air ? `
  if(w==0u){around|=1u<<i;continue;}` : ""}
  if(!umBlockStored(w)){continue;}
  for(var j=0u;j<16u;j++){if(${o.word("w", "j")}!=0u){around|=1u<<i;break;}}
 }
 ${o.store("b", "16u", "slabs")}${o.store("b", "17u", "around")}
}
// A bit of block blk in the tile box [low, high], in tiles of its page.
fn ${n}In(blk:u32,low:vec3i,high:vec3i)->bool{
 let slabs=${o.word("blk", "16u")};if(slabs==0u){return false;}
 let a=vec3u(max(low,vec3i(0)));let b=vec3u(min(high,vec3i(7)));
 let row=((2u<<(b.x-a.x))-1u)<<a.x;var mask=vec2u(0u);
 for(var y=a.y;y<=b.y;y++){mask[y>>2u]|=row<<(8u*(y&3u));}
 for(var z=a.z;z<=b.z;z++){
  if(((slabs>>z)&1u)==0u){continue;}
  if(((${o.word("blk", "(2u*z)")}&mask.x)|(${o.word("blk", "(2u*z+1u)")}&mask.y))!=0u){return true;}
 }
 return false;
}
fn ${n}Any(home:u32,low:vec3i,high:vec3i)->bool{
 let a=max(low,vec3i(0));let b=min(high,vec3i(${8 * UNIFORM_PAGE_WINDOW - 1}));
 if(any(b<a)){return false;}
 let pa=a>>vec3u(3u);let pb=b>>vec3u(3u);let hp=vec3i(umBlockPage(home));
 let near=all(pa>=hp-vec3i(1))&&all(pb<=hp+vec3i(1));
 var around=0u;
 if(near){around=${o.word("home", "17u")};if(around==0u){return false;}}
 let n=pb-pa+vec3i(1);
 if(n.x*n.y*n.z>${BOX_PAGES}){${o.air ? `return true;` : `
  for(var blk=1u;blk+1u<UM_BLOCKS;blk++){
   if(umBlockWord(blk)==UM_FREE){continue;}
   let q=vec3i(umBlockPage(blk));if(any(q<pa)||any(q>pb)){continue;}
   if(${n}In(blk,a-8*q,b-8*q)){return true;}
  }
  return false;
 `}}
 for(var z=pa.z;z<=pb.z;z++){for(var y=pa.y;y<=pb.y;y++){for(var x=pa.x;x<=pb.x;x++){
  let q=vec3i(x,y,z);var w=0u;
  if(near){
   let i=u32((x-hp.x+1)+3*(y-hp.y+1)+9*(z-hp.z+1));
   if(((around>>i)&1u)==0u){continue;}
   w=umBlockNeighbourWord(home,i);
  }else{w=umPageWord(q);}${o.air ? `
  if(w==0u){return true;}` : ""}
  if(!umBlockStored(w)){continue;}
  if(${n}In(w,a-8*q,b-8*q)){return true;}
 }}}
 return false;
}
`;
}

export interface UniformBoundCubesSource {
  /** Key k (0..2 minima, 3..5 maxima) of the tile key `tile`: a WGSL u32
   * expression, of no module-scope name `keys` (the fold's own). */
  readonly tile: (tile: string, k: string) => string;
  /** The table word `index`, and a statement storing `value` there. */
  readonly word: (index: string) => string;
  readonly store: (index: string, value: string) => string;
  /** Cubes or tiles a box query reads at most before it takes the next level. */
  readonly budget: number;
}

/** A bound-cube table's build entry and its query.
 *
 * boundCubes: a workgroup a resident block (striding the pool's resident
 * list, uniform-live-lists, over any fixed grid). Its 64 lanes fold the
 * block's tiles into its 64 cubes of side 2, eight of them those into its
 * cubes of side 4, and one those into the page's. A free block's words are
 * stale and no query reads them.
 *
 * rangeKeys(a, b): the keys of every stored tile in the inclusive tile box,
 * from the aligned cubes of the smallest side that cover it within the
 * budget: side 1 is the tiles themselves, and above the page's side the pages
 * are walked from the block records. A cover overhangs the box by under its
 * side on each face, so the keys bound at least the box and never less; an
 * unstored tile bounds nothing. */
export function uniformBoundCubesWGSL(o: UniformBoundCubesSource): string {
  const fold = (min: (k: string) => string, max: (k: string) => string) => `for(var k=0u;k<3u;k++){keys[k]=min(keys[k],${min("k")});keys[3u+k]=max(keys[3u+k],${max("(3u+k)")});}`;
  const cube = (index: string) => fold(k => o.word(`${index}+${k}`), k => o.word(`${index}+${k}`));
  return /* wgsl */ `
fn identityKeys()->array<u32,6>{return array<u32,6>(0xffffffffu,0xffffffffu,0xffffffffu,0u,0u,0u);}
// First word of cube c of level j of block blk: level 1 its 64 cubes of side
// 2, level 2 its 8 of side 4, level 3 the page; c in cubes of that side.
fn boundCubeIndex(blk:u32,j:u32,c:vec3u)->u32{
 let s=3u-j;return ${UNIFORM_BOUND_CUBE_WORDS}u*blk+6u*(select(select(72u,64u,j==2u),0u,j==1u)+(c.x|(c.y<<s)|(c.z<<(2u*s))));
}
var<workgroup> boundPairs:array<u32,384>;
var<workgroup> boundFours:array<u32,48>;
@compute @workgroup_size(64) fn boundCubes(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 for(var resident=group.x;resident<ulResidentCount();resident+=groups.x){
 let blk=ulResidentBlock(resident);let stored=umBlockWord(blk)!=UM_FREE;
 var keys=identityKeys();
 if(stored){
  let c=vec3u(lane&3u,(lane>>2u)&3u,lane>>4u);
  for(var i=0u;i<8u;i++){
   let t=512u*blk+umTileLocal(2u*c+vec3u(i&1u,(i>>1u)&1u,i>>2u));
   ${fold(k => o.tile("t", k), k => o.tile("t", k))}
  }
  for(var k=0u;k<6u;k++){${o.store(`${UNIFORM_BOUND_CUBE_WORDS}u*blk+6u*lane+k`, "keys[k]")}}
 }
 for(var k=0u;k<6u;k++){boundPairs[6u*lane+k]=keys[k];}
 workgroupBarrier();
 keys=identityKeys();
 if(lane<8u){
  let c=vec3u(lane&1u,(lane>>1u)&1u,lane>>2u);
  for(var i=0u;i<8u;i++){
   let q=2u*c+vec3u(i&1u,(i>>1u)&1u,i>>2u);let l=6u*(q.x|(q.y<<2u)|(q.z<<4u));
   ${fold(k => `boundPairs[l+${k}]`, k => `boundPairs[l+${k}]`)}
  }
  if(stored){for(var k=0u;k<6u;k++){${o.store(`${UNIFORM_BOUND_CUBE_WORDS}u*blk+6u*(64u+lane)+k`, "keys[k]")}}}
  for(var k=0u;k<6u;k++){boundFours[6u*lane+k]=keys[k];}
 }
 workgroupBarrier();
 if(lane==0u&&stored){
  keys=identityKeys();
  for(var i=0u;i<8u;i++){let l=6u*i;${fold(k => `boundFours[l+${k}]`, k => `boundFours[l+${k}]`)}}
  for(var k=0u;k<6u;k++){${o.store(`${UNIFORM_BOUND_CUBE_WORDS}u*blk+432u+k`, "keys[k]")}}
 }
 // Lane 0's reads of the fours land before the next block's.
 workgroupBarrier();
 }
}
fn rangeKeys(low:vec3i,high:vec3i)->array<u32,6>{
 var keys=identityKeys();
 let a=max(low,vec3i(0));let b=min(high,vec3i(${8 * UNIFORM_PAGE_WINDOW - 1}));
 if(any(b<a)){return keys;}
 var j=0u;
 loop{
  let n=(b>>vec3u(j))-(a>>vec3u(j))+vec3i(1);
  if(j>=3u||n.x*n.y*n.z<=${o.budget}){break;}
  j++;
 }
 let pa=a>>vec3u(3u);let pb=b>>vec3u(3u);let n=pb-pa+vec3i(1);
 if(n.x*n.y*n.z>${Math.max(o.budget, BOX_PAGES)}){
  for(var blk=1u;blk+1u<UM_BLOCKS;blk++){
   if(umBlockWord(blk)==UM_FREE){continue;}
   let q=vec3i(umBlockPage(blk));if(any(q<pa)||any(q>pb)){continue;}
   let i=${UNIFORM_BOUND_CUBE_WORDS}u*blk+432u;${cube("i")}
  }
  return keys;
 }
 for(var z=pa.z;z<=pb.z;z++){for(var y=pa.y;y<=pb.y;y++){for(var x=pa.x;x<=pb.x;x++){
  let q=vec3i(x,y,z);let blk=umPageWord(q);if(!umBlockStored(blk)){continue;}
  let la=vec3u(max(a-8*q,vec3i(0)))>>vec3u(j);let lb=vec3u(min(b-8*q,vec3i(7)))>>vec3u(j);
  for(var cz=la.z;cz<=lb.z;cz++){for(var cy=la.y;cy<=lb.y;cy++){for(var cx=la.x;cx<=lb.x;cx++){
   if(j==0u){let t=512u*blk+(cx|(cy<<3u)|(cz<<6u));${fold(k => o.tile("t", k), k => o.tile("t", k))}}
   else{let i=boundCubeIndex(blk,j,vec3u(cx,cy,cz));${cube("i")}}
  }}}
 }}}
 return keys;
}
`;
}

const stored = (word: number, blocks: number): boolean => word >= 1 && word < blocks - 1;
const pageWord = (table: ArrayLike<number>, x: number, y: number, z: number): number => table[x | (y << 6) | (z << 12)]!;

/** nameBits and nameBlocks on the host: the table of a page table's blocks
 * (records.length of them) from flag(tile key) over the keys of stored
 * blocks. rows: the block neighbour rows (uniformBlockNeighbourWords). */
export function uniformTileBitsBuild(records: ArrayLike<number>, rows: ArrayLike<number>, flag: (tile: number) => boolean, air: boolean): Uint32Array<ArrayBuffer> {
  const blocks = records.length, W = UNIFORM_TILE_BITS_WORDS, words = new Uint32Array(W * blocks);
  for (let tile = 0; tile < 512 * blocks; tile++) {
    if (records[tile >>> 9] !== UNIFORM_PAGE_POOL_FREE && flag(tile)) words[W * (tile >>> 9) + ((tile & 511) >>> 5)]! |= 1 << (tile & 31);
  }
  for (let b = 0; b < blocks; b++) {
    if (records[b] === UNIFORM_PAGE_POOL_FREE) continue;
    let slabs = 0, around = 0;
    for (let z = 0; z < 8; z++) if ((words[W * b + 2 * z]! | words[W * b + 2 * z + 1]!) !== 0) slabs |= 1 << z;
    for (let i = 0; i < 27; i++) {
      const w = rows[27 * b + i]!;
      if (air && w === 0) { around |= 1 << i; continue; }
      if (!stored(w, blocks)) continue;
      for (let j = 0; j < 16; j++) if (words[W * w + j] !== 0) { around |= 1 << i; break; }
    }
    words[W * b + 16] = slabs; words[W * b + 17] = around;
  }
  return words;
}
function tileBitsIn(words: ArrayLike<number>, blk: number, low: readonly number[], high: readonly number[]): boolean {
  const W = UNIFORM_TILE_BITS_WORDS, slabs = words[W * blk + 16]!;
  if (slabs === 0) return false;
  const ax = Math.max(low[0]!, 0), ay = Math.max(low[1]!, 0), az = Math.max(low[2]!, 0), bx = Math.min(high[0]!, 7), by = Math.min(high[1]!, 7), bz = Math.min(high[2]!, 7);
  const row = ((2 << (bx - ax)) - 1) << ax;
  const mask = [0, 0];
  for (let y = ay; y <= by; y++) mask[y >> 2]! |= row << (8 * (y & 3));
  for (let z = az; z <= bz; z++) {
    if (((slabs >>> z) & 1) === 0) continue;
    if (((words[W * blk + 2 * z]! & mask[0]!) | (words[W * blk + 2 * z + 1]! & mask[1]!)) !== 0) return true;
  }
  return false;
}
/** nameAny on the host, over a table uniformTileBitsBuild made. */
export function uniformTileBitsAny(words: ArrayLike<number>, table: ArrayLike<number>, records: ArrayLike<number>, rows: ArrayLike<number>, home: number, low: readonly number[], high: readonly number[], air: boolean): boolean {
  const blocks = records.length, last = 8 * UNIFORM_PAGE_WINDOW - 1;
  const a = [0, 1, 2].map(k => Math.max(low[k]!, 0)), b = [0, 1, 2].map(k => Math.min(high[k]!, last));
  if (b.some((v, k) => v < a[k]!)) return false;
  const pa = a.map(v => v >> 3), pb = b.map(v => v >> 3), r = records[home]!, hp = [r & 63, (r >>> 6) & 63, (r >>> 12) & 63];
  const near = pa.every((v, k) => v >= hp[k]! - 1) && pb.every((v, k) => v <= hp[k]! + 1);
  let around = 0;
  if (near) { around = words[UNIFORM_TILE_BITS_WORDS * home + 17]!; if (around === 0) return false; }
  const local = (q: readonly number[], blk: number) => tileBitsIn(words, blk, a.map((v, k) => v - 8 * q[k]!), b.map((v, k) => v - 8 * q[k]!));
  if ((pb[0]! - pa[0]! + 1) * (pb[1]! - pa[1]! + 1) * (pb[2]! - pa[2]! + 1) > BOX_PAGES) {
    if (air) return true;
    for (let blk = 1; blk + 1 < blocks; blk++) {
      const record = records[blk]!;
      if (record === UNIFORM_PAGE_POOL_FREE) continue;
      const q = [record & 63, (record >>> 6) & 63, (record >>> 12) & 63];
      if (q.some((v, k) => v < pa[k]! || v > pb[k]!)) continue;
      if (local(q, blk)) return true;
    }
    return false;
  }
  for (let z = pa[2]!; z <= pb[2]!; z++) for (let y = pa[1]!; y <= pb[1]!; y++) for (let x = pa[0]!; x <= pb[0]!; x++) {
    let w: number;
    if (near) {
      const i = (x - hp[0]! + 1) + 3 * (y - hp[1]! + 1) + 9 * (z - hp[2]! + 1);
      if (((around >>> i) & 1) === 0) continue;
      w = rows[27 * home + i]!;
    } else w = pageWord(table, x, y, z);
    if (air && w === 0) return true;
    if (!stored(w, blocks)) continue;
    if (local([x, y, z], w)) return true;
  }
  return false;
}

/** boundCubes on the host: the table of records.length blocks from
 * key(tile key, k), k in 0..2 a minimum and 3..5 a maximum. */
export function uniformBoundCubesBuild(records: ArrayLike<number>, key: (tile: number, k: number) => number): Uint32Array<ArrayBuffer> {
  const blocks = records.length, W = UNIFORM_BOUND_CUBE_WORDS, words = new Uint32Array(W * blocks);
  const fold = (into: number, from: (k: number) => number) => { for (let k = 0; k < 3; k++) { words[into + k] = Math.min(words[into + k]!, from(k)); words[into + 3 + k] = Math.max(words[into + 3 + k]!, from(3 + k)); } };
  for (let blk = 0; blk < blocks; blk++) {
    if (records[blk] === UNIFORM_PAGE_POOL_FREE) continue;
    for (let c = 0; c < 73; c++) words.fill(0xffffffff, W * blk + 6 * c, W * blk + 6 * c + 3);
    for (let lane = 0; lane < 64; lane++) for (let i = 0; i < 8; i++) {
      const x = 2 * (lane & 3) + (i & 1), y = 2 * ((lane >> 2) & 3) + ((i >> 1) & 1), z = 2 * (lane >> 4) + (i >> 2), t = 512 * blk + (x | (y << 3) | (z << 6));
      fold(W * blk + 6 * lane, k => key(t, k));
    }
    for (let lane = 0; lane < 8; lane++) for (let i = 0; i < 8; i++) {
      const x = 2 * (lane & 1) + (i & 1), y = 2 * ((lane >> 1) & 1) + ((i >> 1) & 1), z = 2 * (lane >> 2) + (i >> 2), l = W * blk + 6 * (x | (y << 2) | (z << 4));
      fold(W * blk + 6 * (64 + lane), k => words[l + k]!);
    }
    for (let i = 0; i < 8; i++) fold(W * blk + 432, k => words[W * blk + 6 * (64 + i) + k]!);
  }
  return words;
}
/** rangeKeys on the host, written to out; level: the side's power the cover took (3 where it walked the blocks too). */
export function uniformBoundCubeRange(cubes: ArrayLike<number>, key: (tile: number, k: number) => number, table: ArrayLike<number>, records: ArrayLike<number>, low: readonly number[], high: readonly number[], budget: number, out: number[]): { readonly level: number; readonly reads: number } {
  const blocks = records.length, W = UNIFORM_BOUND_CUBE_WORDS, last = 8 * UNIFORM_PAGE_WINDOW - 1;
  out.length = 6; out.fill(0xffffffff, 0, 3); out.fill(0, 3, 6);
  const a = [0, 1, 2].map(k => Math.max(low[k]!, 0)), b = [0, 1, 2].map(k => Math.min(high[k]!, last));
  if (b.some((v, k) => v < a[k]!)) return { level: 0, reads: 0 };
  const fold = (from: (k: number) => number) => { for (let k = 0; k < 3; k++) { out[k] = Math.min(out[k]!, from(k)); out[3 + k] = Math.max(out[3 + k]!, from(3 + k)); } };
  let j = 0, reads = 0;
  for (;;) {
    const n = [0, 1, 2].map(k => (b[k]! >> j) - (a[k]! >> j) + 1);
    if (j >= 3 || n[0]! * n[1]! * n[2]! <= budget) break;
    j++;
  }
  const pa = a.map(v => v >> 3), pb = b.map(v => v >> 3);
  if ((pb[0]! - pa[0]! + 1) * (pb[1]! - pa[1]! + 1) * (pb[2]! - pa[2]! + 1) > Math.max(budget, BOX_PAGES)) {
    for (let blk = 1; blk + 1 < blocks; blk++) {
      const record = records[blk]!;
      if (record === UNIFORM_PAGE_POOL_FREE) continue;
      const q = [record & 63, (record >>> 6) & 63, (record >>> 12) & 63];
      if (q.some((v, k) => v < pa[k]! || v > pb[k]!)) continue;
      reads++; fold(k => cubes[W * blk + 432 + k]!);
    }
    return { level: 3, reads };
  }
  for (let z = pa[2]!; z <= pb[2]!; z++) for (let y = pa[1]!; y <= pb[1]!; y++) for (let x = pa[0]!; x <= pb[0]!; x++) {
    const blk = pageWord(table, x, y, z);
    if (!stored(blk, blocks)) continue;
    const q = [x, y, z], la = a.map((v, k) => Math.max(v - 8 * q[k]!, 0) >> j), lb = b.map((v, k) => Math.min(v - 8 * q[k]!, 7) >> j), s = 3 - j;
    for (let cz = la[2]!; cz <= lb[2]!; cz++) for (let cy = la[1]!; cy <= lb[1]!; cy++) for (let cx = la[0]!; cx <= lb[0]!; cx++) {
      reads++;
      if (j === 0) { const t = 512 * blk + (cx | (cy << 3) | (cz << 6)); fold(k => key(t, k)); }
      else { const i = W * blk + 6 * ((j === 1 ? 0 : j === 2 ? 64 : 72) + (cx | (cy << s) | (cz << (2 * s)))); fold(k => cubes[i + k]!); }
    }
  }
  return { level: j, reads };
}
