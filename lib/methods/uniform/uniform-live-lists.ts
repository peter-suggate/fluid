import type { UniformPagePoolLayout } from "./uniform-page-pool";
import { UNIFORM_PAGE_POOL_STATUS } from "./uniform-page-pool";

/** Live lists (docs/plans/tiered-adaptivity-huge-grids-2026-10-24.md §2.5):
 * a launch over pool storage visits the blocks whose words can matter, never
 * the reservation. The lists are the pool's own (UniformPagePool.state,
 * bound read-only): the resident blocks, ascending, then the blocks the
 * latest commit retired, ascending. Their counts live on the GPU, and a
 * launch is a small fixed grid striding them (no indirect dispatch).
 *
 * Resident only (ulResidentCount, ulResidentBlock): for a launch that
 * writes nothing for a key whose block holds no page (umTileLive false).
 *
 * Visit (ulVisitCount, ulVisitBlock, ulVisitRegion): for a launch whose
 * writes for a block that holds no page are a fixed point, the same words
 * whatever ran before (a free region's reset state, a free key's zero rows).
 * Such a block needs one visit after it leaves, and none while it stays
 * free. The resident blocks and the latest commit's retired ones give that
 * visit whenever at most one commit passed since the launch's last encode.
 * UniformLiveVisit counts the pool's changes on the host and asks for a
 * sweep of every block otherwise (the first encode, a seed, a skipped
 * commit): the launch then carries grid.y = 2 (uniformLiveDispatch), which
 * ulSweep reads; the workgroups of the second row exit at once.
 *
 * The region of a visit index is 8 a block plus its octant, so regions and
 * tile keys keep the lists' ascending order. */
export function uniformLiveListsWGSL(group:number,binding:number,layout:UniformPagePoolLayout):string{
 return `@group(${group}) @binding(${binding}) var<storage,read> ulPool:array<u32>;\n${uniformLiveListsOfWGSL("ulPool",layout)}`;
}
/** The same over a pool state the module already binds, by its name. */
export function uniformLiveListsOfWGSL(state:string,layout:UniformPagePoolLayout):string{
 const S=UNIFORM_PAGE_POOL_STATUS;
 return /* wgsl */`
const UL_BLOCKS:u32=${layout.blocks}u;
fn ulSweep(groups:vec3u)->bool{return groups.y>1u;}
fn ulResidentCount()->u32{return ${state}[${S.resident}u];}
fn ulResidentBlock(i:u32)->u32{return ${state}[${layout.state.resident}u+i];}
fn ulVisitCount(sweep:bool)->u32{if(sweep){return UL_BLOCKS;}return ${state}[${S.resident}u]+${state}[${S.pending}u];}
fn ulVisitBlock(i:u32,sweep:bool)->u32{
 if(sweep){return i;}
 let resident=${state}[${S.resident}u];
 if(i<resident){return ${state}[${layout.state.resident}u+i];}
 return ${state}[${layout.state.retired}u+i-resident];
}
fn ulVisitRegion(i:u32,sweep:bool)->u32{return 8u*ulVisitBlock(i>>3u,sweep)+(i&7u);}
`;}

/** Workgroups of a lane-per-item visit (64 lanes each); any positive grid
 * is correct. */
export const UNIFORM_LIVE_LANE_GROUPS=64;
/** Workgroups of a lane-per-tile-key visit (512 keys a block). */
export const UNIFORM_LIVE_KEY_GROUPS=256;
/** Workgroups of a workgroup-per-item visit. */
export const UNIFORM_LIVE_GROUP_GROUPS=256;

/** The host half of a visit launch: one per launch site (or group of
 * launches encoded together). reset: a revision of something else that
 * writes the launch's free-block words wholesale (a host upload); any
 * change of it sweeps too. */
export class UniformLiveVisit{
 private seen=Number.NEGATIVE_INFINITY;
 private seenReset=Number.NaN;
 constructor(private readonly pool:{readonly liveRevision:number},private readonly reset?:()=>number){}
 /** At each encode of the launch: whether it must sweep every block. */
 sweep():boolean{
  const revision=this.pool.liveRevision,reset=this.reset?this.reset():0;
  const sweep=revision-this.seen>1||reset!==this.seenReset;
  this.seen=revision;this.seenReset=reset;return sweep;
 }
}

/** A visit launch of x workgroups; a sweep doubles the grid's rows (ulSweep). */
export function uniformLiveDispatch(pass:GPUComputePassEncoder,x:number,sweep:boolean):void{pass.dispatchWorkgroups(x,sweep?2:1);}
