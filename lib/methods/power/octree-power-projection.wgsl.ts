import {
  octreeProjectionShaderSource,
  type OctreeProjectionLane,
} from "../octree-shared/octree-projection.wgsl";

export const POWER_PROJECTION_LANE: OctreeProjectionLane = {
  compactSurfaceRows: `  if (fineSummaryFactor != 1u && size <= 2u) {
    return false;
  }`,
  mixedRingRepair:
    `  if (owner.size >= 2u && owner.size <= 16u && isOrigin(gid, owner)) { repairPaperMixedNeighbors(gid, owner.size); }`,
};

/** The projection module, assembled with the Power topology clauses. */
export const octreePowerProjectionShader = octreeProjectionShaderSource(POWER_PROJECTION_LANE);
