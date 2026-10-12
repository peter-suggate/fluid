import { inscribedQuadricRasterizer } from "../inscribed-quadric";
import { SOLID_PATCH_ANALYTIC_SHAPE, SOLID_PATCH_ELLIPSOID_KIND, type SolidPatchKind } from "../kind";

/** The ellipsoid inscribed in the box; see `../inscribed-quadric` for its distance and its one-cell axes. */
export const ellipsoidSolidPatchKind: SolidPatchKind = {
  id: SOLID_PATCH_ELLIPSOID_KIND,
  analyticShape: SOLID_PATCH_ANALYTIC_SHAPE.ellipsoid,
  validate() {},
  // The box is the whole shape.
  fold() {},
  rasterizer: (patch) => inscribedQuadricRasterizer(patch, undefined),
};
