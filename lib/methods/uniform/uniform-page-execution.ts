import type { UniformPageDomain } from "./uniform-page-domain";

/** Prove complete rectangular residency once, before compiling a native plan.
 * Missing, duplicate, translated and out-of-range page records are rejected.
 * A changed catalogue must compile a new plan before it is published; this is
 * not a dynamic membership bypass. Production catalogues are all-resident.
 */
export function uniformPageHasRectangularCoverage(domain: UniformPageDomain): boolean {
  if(![16,32].includes(domain.edge) || !Number.isSafeInteger(domain.capacity)
    || domain.capacity<1 || domain.words.length<16+18*domain.capacity
    || domain.words[9]!==domain.edge)return false;
  const dimensions=Array.from(domain.words.subarray(12,15));
  if(dimensions.some(n=>n===0))return false;
  const grid=dimensions.map(n=>Math.ceil(n/domain.edge));
  const expected=grid[0]!*grid[1]!*grid[2]!;
  if(domain.count!==expected || domain.words[8]!==expected || domain.capacity<expected)return false;
  const seen=new Set<number>();
  for(let i=0;i<domain.count;i++){
    const slot=domain.words[16+16*domain.capacity+i]!;
    if(slot>=domain.capacity)return false;
    const at=16+16*slot;
    const [x,y,z]=Array.from(domain.words.subarray(at,at+3));
    if(x!>=grid[0]! || y!>=grid[1]! || z!>=grid[2]!)return false;
    seen.add(x!+grid[0]!*(y!+grid[1]!*z!));
  }
  return seen.size===expected;
}
