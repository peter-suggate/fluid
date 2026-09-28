import type {UniformMixedLayout} from "./uniform-mixed-layout";
import {uniformMixedPressureStorage} from "./uniform-mixed-pressure-boundary.wgsl";

export interface UniformMixedMemoryRange {offset:number;size:number}
/** The all-4h pressure root's arena fields: pressure, the RHS and a
 * Full-Cycle's correction RHS, the bound and its shifted copy, the residual
 * the acceptance reduces, and the Full-Cycle backup. The frozen workspace
 * is borrowed by authority. Every correction level
 * lives in the native hierarchy (n/4 and below). Phi uses the idle native
 * conditioning buffer; with static solids its range also carries the all-4h
 * solid record. Capacity follows the simulation layout; the frame presents
 * only the live all-4h words. */
export function planUniformMixedPressureMemory(layout:UniformMixedLayout,arenaPrefixBytes:number,conditioningBytes:number){
 let cursor=0;
 const allocate=(size:number):UniformMixedMemoryRange=>{cursor=Math.ceil(cursor/256)*256;const range={offset:cursor,size};cursor+=size;return range;};
 const count=uniformMixedPressureStorage(layout).count;
 const frozen=allocate(count*4),residual=allocate(count*4);
 const root={pressure:allocate(count*4),rhs:[allocate(count*4),allocate(count*4)] as const,minimum:[allocate(count*4),allocate(count*4)] as const,
  phi:{offset:0,size:layout.cellCount*4},frozen,residual};
 const backup=allocate(count*4);
 if(root.phi.size>conditioningBytes||cursor>arenaPrefixBytes)
  throw new Error(`Unified pressure scratch exceeds borrowed fields: ${cursor}/${arenaPrefixBytes} arena bytes, ${root.phi.size}/${conditioningBytes} conditioning bytes`);
 return {root,backup,bytes:cursor};
}
