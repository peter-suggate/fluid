/**
 * Binding 15 is one "lane scratch" slot with three published ABIs: the Power
 * bulk worklist, the Power corrected coarse-phi directory, and the Losasso
 * coarse-phi arena. The shared projection shader dispatches on the header
 * word, so that word cannot live inside either lane package — the engine has
 * to be able to name it without naming a lane.
 */

/** Header word 0 of the Losasso coarse-phi arena view of binding 15. */
export const OCTREE_LOSASSO_COARSE_PHI_MAGIC = 0x4c50_4849;
