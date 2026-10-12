import type { SolidWorldVoxelPatch } from "../../solid-world";
import { inscribedQuadricRasterizer } from "../inscribed-quadric";
import { SOLID_PATCH_ANALYTIC_SHAPE, SOLID_PATCH_CYLINDER_KIND, type SolidPatchKind } from "../kind";

/** The axis a cylinder patch runs along: up, unless it says otherwise. */
export const solidPatchCylinderAxis = (patch: Pick<SolidWorldVoxelPatch, "axis">): 0 | 1 | 2 => patch.axis ?? 1;

/**
 * The elliptic cylinder inscribed in the box, the box's full length along
 * `axis`. Its end caps are faces of the box, so every layer along the axis is
 * the same ellipse; see `../inscribed-quadric`.
 */
export const cylinderSolidPatchKind: SolidPatchKind = {
  id: SOLID_PATCH_CYLINDER_KIND,
  analyticShape: SOLID_PATCH_ANALYTIC_SHAPE.cylinder,
  validate(patch) {
    if (patch.axis !== undefined && patch.axis !== 0 && patch.axis !== 1 && patch.axis !== 2) {
      throw new RangeError(`Solid cylinder patch axis must be 0, 1 or 2, received ${String(patch.axis)}`);
    }
  },
  fold: (patch, word) => word(solidPatchCylinderAxis(patch)),
  rasterizer: (patch) => inscribedQuadricRasterizer(patch, solidPatchCylinderAxis(patch)),
};
