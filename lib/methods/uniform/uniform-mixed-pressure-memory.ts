import type {UniformMixedLayout} from "./uniform-mixed-layout";
import {uniformMixedAllCoarseLayout} from "./uniform-mixed-layout";
import {uniformMixedPressureStorage} from "./uniform-mixed-pressure-boundary.wgsl";

export interface UniformMixedMemoryRange {offset:number;size:number}
/** Pressure lifetimes: slopes, frozen RHS and residuals are recomputed after
 * returning from a child solve, so both levels share those three workspaces.
 * All live pressure/RHS/minimum/phi fields remain disjoint. Level 0's phi uses
 * the idle native conditioning buffer; the native 4h hierarchy stays intact.
 * Both levels are all-4h. Level 0 keeps the capacity of the simulation
 * layout (its phi range also carries the static solid record and the frame's
 * presentation reads it whole). With static solids, level 1 also carries one
 * vec4 (open, V+) record per owner and halo slot. */
export function planUniformMixedPressureMemory(layout:UniformMixedLayout,arenaPrefixBytes:number,conditioningBytes:number,solid=false){
 const layouts=[layout,uniformMixedAllCoarseLayout(layout)];
 let cursor=0;
 const allocate=(size:number):UniformMixedMemoryRange=>{cursor=Math.ceil(cursor/256)*256;const range={offset:cursor,size};cursor+=size;return range;};
 const rootCount=uniformMixedPressureStorage(layout).count;
 const slopes=allocate(layout.cellCount*16),frozen=allocate(rootCount*4),residual=allocate(rootCount*4);
 const levels=layouts.map((l,i)=>{
  const count=uniformMixedPressureStorage(l).count;
  return {pressure:allocate(count*4),rhs:[allocate(count*4),allocate(count*4)] as const,
   minimum:Array.from({length:i===0?2:1},()=>allocate(count*4)),
   phi:i===0?{offset:0,size:layout.cellCount*4}:allocate(l.cellCount*4),
   slopes:{...slopes,size:l.cellCount*16},frozen:{...frozen,size:count*4},residual:{...residual,size:count*4},
   topology:solid&&i>0?allocate(count*16):undefined};
 });
 const backup=allocate(rootCount*4);
 if(levels[0]!.phi.size>conditioningBytes||cursor>arenaPrefixBytes)
  throw new Error(`Unified pressure scratch exceeds borrowed fields: ${cursor}/${arenaPrefixBytes} arena bytes, ${levels[0]!.phi.size}/${conditioningBytes} conditioning bytes`);
 return {layouts,levels,backup,bytes:cursor};
}
