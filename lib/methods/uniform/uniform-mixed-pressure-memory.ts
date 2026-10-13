import type {UniformLatticeAddress} from "./uniform-lattice-address";
import type {UniformMixedLayout} from "./uniform-mixed-layout";
import {uniformMixedPressureStorage} from "./uniform-mixed-pressure-boundary.wgsl";
import {uniformMixedPageCount} from "./uniform-mixed-topology.wgsl";

export interface UniformMixedMemoryRange {offset:number;size:number}
const aligned=(bytes:number)=>Math.ceil(bytes/256)*256;
/** The authority workspace (UniformMixedPressureAuthority.scratchBytes) both
 * authorities fit: the simulation authority's, a job per tile at most (every
 * tile h); and the split's on the all-4h owners, a job per 64 owners and per
 * page plus its cut word per owner. Fixed in the lattice: no h-tile capacity.
 * The root's owners are the box's tiles, which a pool smaller than the box
 * has more of than tile keys. */
function frozenBytes(address:UniformLatticeAddress):number{
 const tiles=Math.max(address.tileKeys,address.boxTiles),split=Math.ceil(tiles/64)+uniformMixedPageCount({dimensions:address.cells});
 return Math.max(8*(1+2*(tiles+Math.ceil(tiles/1024))),8*(1+2*(split+Math.ceil(split/1024))+tiles));
}
/** Wall slots of the all-4h root: a tile face on each of the six planes of
 * the static box (uniformMixedPressureStorage at width 4). */
const rootWallSlots=(address:UniformLatticeAddress)=>{const t=address.tiles;return 2*(t[0]*t[1]+t[0]*t[2]+t[1]*t[2]);};
/** Arena bytes of the root planUniformMixedPressureMemory lays out on a
 * lattice: seven fields of the all-4h live words (an owner per box tile,
 * whatever the pool stores of it, plus its wall slots) and the authority
 * workspace. */
export function uniformMixedPressureRootBytes(address:UniformLatticeAddress):number{
 const words=address.boxTiles+rootWallSlots(address);
 return 7*aligned(4*words)+aligned(frozenBytes(address));
}
/** The all-4h pressure root's arena fields: pressure, the RHS and a
 * Full-Cycle's correction RHS, the bound and its shifted copy, the residual
 * the acceptance reduces, and the Full-Cycle backup. The frozen workspace
 * is borrowed by authority. Every correction level
 * lives in the native hierarchy (n/4 and below). Phi is the frame's own
 * buffer, not arena scratch. `layout` is the all-4h pressure layout: the
 * root holds its live words only, whatever the simulation layout. `address`
 * is the lattice the arena range was sized on (uniformMixedPressureRootBytes). */
export function planUniformMixedPressureMemory(layout:UniformMixedLayout,address:UniformLatticeAddress,arenaOffset:number,arenaBytes:number){
 const storage=uniformMixedPressureStorage(layout);
 if(storage.width!==4||arenaOffset%256!==0)throw new Error("The mixed pressure root is planned on the all-4h layout at an aligned arena offset");
 let cursor=arenaOffset;
 const allocate=(size:number):UniformMixedMemoryRange=>{cursor=aligned(cursor);const range={offset:cursor,size};cursor+=size;return range;};
 const count=storage.count;
 const frozen=allocate(frozenBytes(address)),residual=allocate(count*4);
 const root={pressure:allocate(count*4),rhs:[allocate(count*4),allocate(count*4)] as const,minimum:[allocate(count*4),allocate(count*4)] as const,
  frozen,residual};
 const backup=allocate(count*4);
 if(cursor>arenaOffset+arenaBytes)
  throw new Error(`Unified pressure scratch exceeds its arena range: ${cursor-arenaOffset}/${arenaBytes} bytes`);
 return {root,backup,bytes:cursor-arenaOffset};
}
