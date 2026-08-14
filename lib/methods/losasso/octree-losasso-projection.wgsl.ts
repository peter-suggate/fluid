import {
  octreeProjectionShaderSource,
  type OctreeProjectionLane,
} from "../octree-shared/octree-projection.wgsl";

export const LOSASSO_PROJECTION_LANE: OctreeProjectionLane = {
  compactSurfaceRows:
    `  // Losasso: size-two rows inside the two-fine-cell shell split to unit rows.`,
  mixedRingRepair:
    `  // Losasso's native face graph represents every strict 2:1 transition.
  // The stronger Power/Delaunay mixed-ring repair would split legal coarse
  // neighbours and manufacture an unnecessary intermediate-resolution shell.`,
};

/** The projection module, assembled with the Losasso topology clauses. */
export const octreeLosassoProjectionShader = octreeProjectionShaderSource(LOSASSO_PROJECTION_LANE);
