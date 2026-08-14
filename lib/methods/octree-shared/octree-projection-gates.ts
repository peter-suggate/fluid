/**
 * Process-environment arms for the octree projection.
 *
 * Each gate is read from the environment at construction or encode time rather
 * than from a scene field, because each exists to make an interleaved A/B
 * possible out of one shader module: two processes, one build, different arms.
 * Routing them through the scene document instead would change the authored
 * thing being measured.
 *
 * The default of every gate is the shipped behavior, and a gate defaults on
 * only where the two arms provably write the same words to the same cells --
 * the individual comments record what each arm was measured against, including
 * the one that is off because it currently leaves symmetric-expansion inert.
 */

/** Read at encode time so benchmark processes can select attribution without
 * changing construction or numerical behavior. */
export function octreeFineEngineSplitsEnabled(
  environment?: Readonly<Record<string, string | undefined>>,
): boolean {
  const resolved = environment
    ?? (typeof process !== "undefined" ? process.env : undefined);
  return resolved?.FLUID_ENGINE_SPLIT !== "collapsed";
}

/**
 * Whole-page owner fills inside the split materializer.
 *
 * Selects the page-claimed form of splitLeaf over the per-cell walk. Both write
 * the same cells the same words through the same idempotent atomicMin; the page
 * form only stops the inner loop from recomputing a page-invariant constant 512
 * times through three runtime integer divisions. Off restores the original walk
 * verbatim, including its per-cell membership load, so an interleaved A/B can
 * score them from one shader module in separate processes.
 */
export function octreeGradingPageFillEnabled(
  environment?: Readonly<Record<string, string | undefined>>,
): boolean {
  const resolved = environment
    ?? (typeof process !== "undefined" ? process.env : undefined);
  return resolved?.FLUID_OCTREE_GRADING_PAGE_FILL !== "0";
}

/**
 * Losing askers take one page of the split they asked for.
 *
 * Requires the page-claimed materializer above; with the page fill off this has
 * nothing to divide. The write set is unchanged either way -- the same cells
 * receive the same words through the same idempotent atomicMin -- so this only
 * chooses how many lanes carry it.
 */
export function octreeGradingSplitHelpersEnabled(
  environment?: Readonly<Record<string, string | undefined>>,
): boolean {
  const resolved = environment
    ?? (typeof process !== "undefined" ? process.env : undefined);
  return resolved?.FLUID_OCTREE_GRADING_SPLIT_HELPERS !== "0";
}

/**
 * Restore the per-cell membership load inside the split materializer.
 *
 * Off by default because the bit it preserves is provably clear on every cell
 * the topology candidate view can address; see splitOwnerWord. Kept as an arm
 * so the interleaved A/B can price the load rather than assert it.
 */
export function octreeGradingMembershipLoadEnabled(
  environment?: Readonly<Record<string, string | undefined>>,
): boolean {
  const resolved = environment
    ?? (typeof process !== "undefined" ? process.env : undefined);
  return resolved?.FLUID_OCTREE_GRADING_MEMBERSHIP_LOAD === "1";
}

/**
 * Size the topology delta tile by the largest leaf the domain can hold.
 *
 * Default OFF: it currently leaves symmetric-expansion inert. See the comment
 * at the assignment for what has already been ruled out.
 */
export function octreeTopologyTileClampEnabled(
  environment?: Readonly<Record<string, string | undefined>>,
): boolean {
  const resolved = environment
    ?? (typeof process !== "undefined" ? process.env : undefined);
  return resolved?.FLUID_OCTREE_TOPOLOGY_TILE_CLAMP === "1";
}
