/** The residency certificate's page arithmetic (uniformMixedResidencyWord in
 * uniform-mixed-topology.wgsl.ts): the kernels' WGSL and their host twins,
 * the same integer arithmetic.
 *
 * A residency page is 16 cells (4 tiles) an axis, eight a pool block: page
 * 8·block + octant, the octant x | y<<1 | z<<2 of the block's halves. Its
 * coordinate (16-cell pages of the 64-page window, 0..127 an axis) is twice
 * its block's page plus the octant; tile lane l of it (umCorner(l, 4)) is a
 * key of its block.
 *
 * One census certifies in three steps around the page commit:
 *  seed     per page of a block that holds one, the box of its seed tiles
 *           (uniformResidencySeedWord); 0 for any other page.
 *  request  each seeded page asks the pool for every 32-cell page of the
 *           window with a tile within the closure radius (Chebyshev, in
 *           tiles) of its seed box (uniformResidencyRequests). The commit
 *           follows: an AIR page among them is claimed, a SOLID one is not.
 *  mark     through the page table the commit left: a page is resident when
 *           its block holds a page and a seed lies within the radius of one
 *           of its tiles (uniformResidencyMark). A block the commit claimed
 *           was seeded 0 while it was free, and a seeded block asked for its
 *           own page, so every seed word a mark reads is this census's.
 * A radius over UNIFORM_RESIDENCY_PAGE_REACH pages makes every page of a
 * block that holds one resident, and the request then reaches that far. */
import { UNIFORM_PAGE_POOL_FREE } from "./uniform-page-pool";

/** Pages the mark scans around a page, and the request's reach in pages.
 * A live pool's closure cap is this in tiles: a travel of 108 cells a step.
 * A jet in a wall corner is a real 45 to 78 cells a step
 * (high-resolution-dam-break, 1/60 s and 1/30 s). */
export const UNIFORM_RESIDENCY_PAGE_REACH = 8;
/** Bits of a seed word above its packed tile box (low then high corner, in
 * tiles of the page, three bits a coordinate): the page holds a seed; one of
 * its seeds is the liquid's. */
export const UNIFORM_RESIDENCY_SEEDED = 1 << 18, UNIFORM_RESIDENCY_LIQUID = 1 << 19;
/** A tile's seed flags (uniformResidencySeedWord): it seeds; it is the liquid's. */
export const UNIFORM_RESIDENCY_SEED = { seed: 1, liquid: 2 } as const;

/** The stored set: the page table and the block words (a topology's tail). */
export interface UniformResidencyPages { readonly table: ArrayLike<number>; readonly blocks: ArrayLike<number> }

/** umPageOfTile: the page of a tile key. */
export function uniformResidencyPageOfTile(key: number): number { return 8 * (key >>> 9) + (((key >>> 2) & 1) | (((key >>> 5) & 1) << 1) | (((key >>> 8) & 1) << 2)); }
/** umPageTile: tile lane `lane` of a page. */
export function uniformResidencyPageTile(page: number, lane: number): number {
  return 512 * (page >>> 3) + ((4 * (page & 1) + (lane & 3)) | ((4 * ((page >>> 1) & 1) + ((lane >>> 2) & 3)) << 3) | ((4 * ((page >>> 2) & 1) + (lane >>> 4)) << 6));
}
/** The block holds a page: neither sentinel, nor free. */
export function uniformResidencyBlockHeld(pages: UniformResidencyPages, block: number): boolean {
  return block >= 1 && block < pages.blocks.length - 1 && pages.blocks[block] !== UNIFORM_PAGE_POOL_FREE;
}
/** The page at a 16-cell page coordinate of the window (umFarAirOwner's lookup): a page of block 0 or of the last block, never resident, where nothing is stored. */
export function uniformResidencyPageAt(pages: UniformResidencyPages, x: number, y: number, z: number): number {
  const word = pages.table[((x >> 1) & 63) | (((y >> 1) & 63) << 6) | (((z >> 1) & 63) << 12)]!;
  return 8 * Math.min(word, pages.blocks.length - 1) + ((x & 1) | ((y & 1) << 1) | ((z & 1) << 2));
}
/** umPageCoord: a page's 16-cell page coordinate, into `out`. */
export function uniformResidencyPageCoord(pages: UniformResidencyPages, page: number, out: number[]): number[] {
  const r = pages.blocks[page >>> 3]!;
  out[0] = 2 * (r & 63) + (page & 1);out[1] = 2 * ((r >>> 6) & 63) + ((page >>> 1) & 1);out[2] = 2 * ((r >>> 12) & 63) + ((page >>> 2) & 1);
  return out;
}
/** pageSeed's fold over a page's 64 tile lanes: `flags(lane)` is the tile's UNIFORM_RESIDENCY_SEED bits. */
export function uniformResidencySeedWord(flags: (lane: number) => number): number {
  const lo = [3, 3, 3], hi = [0, 0, 0];
  let found = false, liquid = false;
  for (let lane = 0; lane < 64; lane++) {
    const f = flags(lane);
    if (!(f & UNIFORM_RESIDENCY_SEED.seed)) continue;
    const c = [lane & 3, (lane >>> 2) & 3, lane >>> 4];
    for (let k = 0; k < 3; k++) { lo[k] = Math.min(lo[k]!, c[k]!);hi[k] = Math.max(hi[k]!, c[k]!); }
    found = true;liquid ||= (f & UNIFORM_RESIDENCY_SEED.liquid) !== 0;
  }
  if (!found) return 0;
  return (lo[0]! | (lo[1]! << 3) | (lo[2]! << 6) | (hi[0]! << 9) | (hi[1]! << 12) | (hi[2]! << 15) | UNIFORM_RESIDENCY_SEEDED | (liquid ? UNIFORM_RESIDENCY_LIQUID : 0)) >>> 0;
}
/** rsSeedBox: the window tiles the seed word of the page at 16-cell page coordinate `s` spans, low corner then high, into `out`. */
function seedBox(s: readonly number[], word: number, out: number[]): number[] {
  const x = 4 * s[0]!, y = 4 * s[1]!, z = 4 * s[2]!;
  out[0] = x + (word & 7);out[1] = y + ((word >>> 3) & 7);out[2] = z + ((word >>> 6) & 7);
  out[3] = x + ((word >>> 9) & 7);out[4] = y + ((word >>> 12) & 7);out[5] = z + ((word >>> 15) & 7);
  return out;
}
/** pageRequest on the host: `ask(x, y, z)` for every 32-cell page a seeded
 * page asks for, in window pages; a page two seeds reach is asked twice.
 * seeds: a word a page (pageSeed). radius: the closure radius in tiles.
 * Returns false, having asked nothing, where the radius is over `cap` (the
 * pool's closureCap) and a page is seeded. */
export function uniformResidencyRequests(pages: UniformResidencyPages, seeds: ArrayLike<number>, radius: number, cap: number, ask: (x: number, y: number, z: number) => void): boolean {
  const box: number[] = [], s: number[] = [], d = Math.min(radius, 4 * UNIFORM_RESIDENCY_PAGE_REACH);
  for (let page = 0; page < 8 * pages.blocks.length; page++) if (seeds[page] && radius > cap) return false;
  for (let page = 0; page < 8 * pages.blocks.length; page++) {
    const word = seeds[page]!;
    if (!word) continue;
    seedBox(uniformResidencyPageCoord(pages, page, s), word, box);
    // Past the frame (window pages 0..63) nothing is asked: the kernel raises the edge fatal there unless the entry is SOLID.
    const a = [0, 1, 2].map(k => Math.max(box[k]! - d, 0) >> 3), b = [0, 1, 2].map(k => Math.min(box[3 + k]! + d, 511) >> 3);
    for (let z = a[2]!; z <= b[2]!; z++) for (let y = a[1]!; y <= b[1]!; y++) for (let x = a[0]!; x <= b[0]!; x++) ask(x, y, z);
  }
  return true;
}
/** rsResident: one page of the mark. */
export function uniformResidencyResident(pages: UniformResidencyPages, seeds: ArrayLike<number>, radius: number, page: number): boolean {
  if (!uniformResidencyBlockHeld(pages, page >>> 3)) return false;
  const reach = (radius + 3) >> 2;
  if (reach > UNIFORM_RESIDENCY_PAGE_REACH) return true;
  const q = uniformResidencyPageCoord(pages, page, []), box: number[] = [];
  const a = q.map(v => Math.max(v - reach, 0) >> 1), b = q.map(v => Math.min(v + reach, 127) >> 1);
  for (let z = a[2]!; z <= b[2]!; z++) for (let y = a[1]!; y <= b[1]!; y++) for (let x = a[0]!; x <= b[0]!; x++) {
    const n = pages.table[x | (y << 6) | (z << 12)]!;
    if (!(n >= 1 && n < pages.blocks.length - 1)) continue;
    for (let o = 0; o < 8; o++) {
      const s = [2 * x + (o & 1), 2 * y + ((o >> 1) & 1), 2 * z + (o >> 2)];
      if (s.some((v, k) => v < q[k]! - reach || v > q[k]! + reach)) continue;
      const word = seeds[8 * n + o]!;
      if (!word) continue;
      seedBox(s, word, box);
      let gap = 0;
      for (let k = 0; k < 3; k++) gap = Math.max(gap, box[k]! - (4 * q[k]! + 3), 4 * q[k]! - box[3 + k]!);
      if (gap <= radius) return true;
    }
  }
  return false;
}
/** pageMark then pageCompact on the host: the flag of every page into
 * `flags` and the resident pages, ascending, into `list`; returns their
 * count. */
export function uniformResidencyMark(pages: UniformResidencyPages, seeds: ArrayLike<number>, radius: number, flags: Uint32Array, list: Uint32Array): number {
  let count = 0;
  for (let page = 0; page < 8 * pages.blocks.length; page++) {
    flags[page] = uniformResidencyResident(pages, seeds, radius, page) ? 1 : 0;
    if (flags[page]) list[count++] = page;
  }
  return count;
}

/** rsSeedBox: what the request and the mark both read of a seed word. Once a module. */
export const uniformResidencySeedBoxWGSL = /* wgsl */ `
const RS_REACH:u32=${UNIFORM_RESIDENCY_PAGE_REACH}u;
struct RSBox{lo:vec3i,hi:vec3i}
// The window tiles the seed word of the page at 16-cell page coordinate s spans.
fn rsSeedBox(s:vec3i,word:u32)->RSBox{
 return RSBox(4*s+vec3i(i32(word&7u),i32((word>>3u)&7u),i32((word>>6u)&7u)),4*s+vec3i(i32((word>>9u)&7u),i32((word>>12u)&7u),i32((word>>15u)&7u)));
}`;
/** rsAsk(page, word, r), for a module that includes a topology,
 * uniformPagePoolWGSL with `work` and uniformResidencySeedBoxWGSL: a seeded page (its seed word) asks
 * the pool for every 32-cell page of the window with a tile within r tiles
 * of its seed box, at most RS_REACH pages' tiles away. Returns false, having
 * raised closureCap with r as its detail and asked nothing, where r is over
 * UP_CLOSURE_CAP. A seed of the liquid's whose own page the pool does not
 * hold raises closure and is asked for all the same. A page past the frame
 * (the window's pages 0..63 an axis) is never asked for: a SOLID entry there
 * is a wall, and any other raises the edge fatal. */
export const uniformResidencyRequestWGSL = /* wgsl */ `
fn rsAsk(page:u32,word:u32,r:u32)->bool{
 let own=vec3i(umBlockPage(page>>3u));
 if(r>UP_CLOSURE_CAP){upFailWith(UP_FATAL_CLOSURE_CAP,own,r);return false;}
 if((word&${UNIFORM_RESIDENCY_LIQUID}u)!=0u&&upBlockOf(own)==UP_NONE){upFail(UP_FATAL_CLOSURE,own);}
 let box=rsSeedBox(vec3i(umPageCoord(page)),word);let d=vec3i(i32(min(r,4u*RS_REACH)));
 let a=(box.lo-d)>>vec3u(3u);let b=(box.hi+d)>>vec3u(3u);
 for(var z=a.z;z<=b.z;z++){for(var y=a.y;y<=b.y;y++){for(var x=a.x;x<=b.x;x++){
  let q=vec3i(x,y,z);
  // Past the frame: a SOLID entry is the wall it always was; anything else is the frame's edge.
  if(any(q<vec3i(0))||any(q>vec3i(i32(UP_TABLE)-1))){if(upKind(q)!=UP_SOLID){upFail(UP_FATAL_EDGE,q);}continue;}
  upRequest(q);
 }}}
 return true;
}`;
/** rsResident(page, r), for a module that includes a topology and
 * uniformResidencySeedBoxWGSL: the mark of one page under closure radius r
 * (tiles), through the page table as it stands. `seed(page)` is the WGSL of
 * a page's seed word. */
export function uniformResidencyMarkWGSL(seed: (page: string) => string): string {
  return /* wgsl */ `
fn rsResident(page:u32,r:u32)->bool{
 let block=page>>3u;if(!umBlockStored(block)||umBlockWord(block)==UM_FREE){return false;}
 let reach=i32((r+3u)/4u);if(reach>i32(RS_REACH)){return true;}
 let q=vec3i(umPageCoord(page));let qa=4*q;let qb=qa+vec3i(3);
 // The 32-cell pages of the window that hold a page within reach.
 let a=max(q-vec3i(reach),vec3i(0))>>vec3u(1u);let b=min(q+vec3i(reach),vec3i(127))>>vec3u(1u);
 for(var z=a.z;z<=b.z;z++){for(var y=a.y;y<=b.y;y++){for(var x=a.x;x<=b.x;x++){
  let n=umPageWord(vec3i(x,y,z));if(!umBlockStored(n)){continue;}
  for(var o=0u;o<8u;o++){
   let s=2*vec3i(x,y,z)+vec3i(vec3u(o&1u,(o>>1u)&1u,o>>2u));
   if(any(s<q-vec3i(reach))||any(s>q+vec3i(reach))){continue;}
   let word=${seed("(8u*n+o)")};if(word==0u){continue;}
   let box=rsSeedBox(s,word);
   let gap=max(max(box.lo-qb,qa-box.hi),vec3i(0));
   if(max(gap.x,max(gap.y,gap.z))<=i32(r)){return true;}
  }
 }}}
 return false;
}`;
}
