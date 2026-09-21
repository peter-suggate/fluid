import type { UniformPageDomain } from "./uniform-page-domain";

/** Compile immutable accepted residency into an execution layout. The native
 * case is a page payload with field-specific vertex/pressure halos, not an atlas
 * with a second page allocated for each halo plane. No runtime address adapter
 * is needed when this payload is the entire accepted domain.
 *
 * A changed accepted catalogue must compile a new plan before it is published.
 * Current production catalogues are immutable and all-resident.
 */
export function uniformPageHasNativeCoordinates(domain: UniformPageDomain): boolean {
  if (domain.count !== 1 || domain.capacity !== 1) return false;
  const slot = domain.words[16 + 16 * domain.capacity]!;
  const record = 16 + 16 * slot;
  return slot === 0 && domain.words[record] === 0
    && domain.words[record + 1] === 0 && domain.words[record + 2] === 0
    && domain.words[8] === 1
    && [12, 13, 14].every(i => domain.words[i]! > 0 && domain.words[i]! <= domain.edge);
}

/** Prove complete rectangular residency once, before compiling a native plan.
 * Missing, duplicate, translated and out-of-range page records are rejected.
 * Like the single-page specialization, this plan must be replaced before a
 * changed catalogue can be published; it is not a dynamic membership bypass.
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
