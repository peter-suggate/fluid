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
