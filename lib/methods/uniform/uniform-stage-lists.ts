/** Live lists for stage launches (tiered-adaptivity §2.5): no launch, clear
 * or copy of a stage is sized by the pool's capacity. Each is a direct
 * grid-stride over a list whose count the GPU holds, launched at a small
 * fixed grid: an empty workgroup of it exits at once.
 *
 * The live tiles are the ownership's own worklists (uniformMixedTopologyWGSL):
 * the h list then the 4h list, umCounts.x + umCounts.y tiles, each in tile
 * key order. Every stored tile is in exactly one of them (the layout
 * builder's FATAL_TIER_SUM), so a stride over them visits exactly the tiles
 * a stride over every key that tests umTileLive keeps.
 *
 * A held block (one whose record holds a page) has all 512 of its tiles
 * live, so the held blocks are the blocks of the live tiles of rank 512 r in
 * key order: the r-th held block is the block of the (512 r)-th smallest key
 * of the two sorted lists, found by a binary search of their merge. Its 16h
 * pages are 8 a block (umPageOfTile's order). Nothing is built or stored.
 *
 * The frame head's launches stride the pool's own lists instead
 * (uniform-live-lists.ts): they run around the commit and bind the pool. A
 * stage binds only its ownership, and these are that ownership's: the held
 * blocks of the simulation ownership are the pool's resident blocks once the
 * commit has named the stored set, but a standalone ownership has no pool,
 * and the resident pressure ownership's lists are its static layout's (a
 * stage over it scans every page). */

/** The fixed grid of a stage's list launch: past it a workgroup strides. */
export const UNIFORM_STAGE_LIST_GRID = 1024;

/** Workgroups of a list launch over at most `jobs` jobs: never more than the
 * fixed grid, and one at least. */
export function uniformStageListGroups(jobs: number, grid = UNIFORM_STAGE_LIST_GRID): number {
  return Math.max(1, Math.min(grid, Math.ceil(jobs)));
}

/** Needs uniformMixedTopologyWGSL in the module (umTopology, umCounts). */
export const uniformStageListsWGSL = /* wgsl */ `
// The live tiles: the h list then the 4h list (uniform-stage-lists.ts).
fn umLiveTileCount()->u32{return umCounts.x+umCounts.y;}
fn umLiveTile(j:u32)->u32{return umTopology[UM_TILES+j];}
// Live tile j, or none past the list.
fn umLiveTileOr(j:u32,none:u32)->u32{if(j>=umLiveTileCount()){return none;}return umLiveTile(j);}
// The k-th smallest live tile key (k < umLiveTileCount()): i keys of the h
// list and k - i of the 4h list precede it, i the first with h[i] past the
// 4h list's (k - i - 1)-th.
fn umLiveTileRank(k:u32)->u32{
 let x=umCounts.x;let y=umCounts.y;let h=UM_TILES;let c=UM_TILES+x;
 var lo=select(0u,k-y,k>y);var hi=min(k,x);
 while(lo<hi){let mid=(lo+hi)/2u;if(umTopology[h+mid]<umTopology[c+k-mid-1u]){lo=mid+1u;}else{hi=mid;}}
 var key=0xffffffffu;if(lo<x){key=umTopology[h+lo];}if(k-lo<y){key=min(key,umTopology[c+k-lo]);}
 return key;
}
// The held blocks, in block order, and their 16h pages (8 a block).
fn umHeldBlockCount()->u32{return umLiveTileCount()/512u;}
fn umHeldBlock(r:u32)->u32{return umLiveTileRank(512u*r)>>9u;}
fn umHeldPageCount()->u32{return 8u*umHeldBlockCount();}
fn umHeldPage(j:u32)->u32{return 8u*umHeldBlock(j>>3u)+(j&7u);}
`;
