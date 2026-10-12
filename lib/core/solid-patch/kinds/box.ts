import { SOLID_PATCH_ANALYTIC_SHAPE, SOLID_PATCH_BOX_KIND, type SolidPatchKind,
  type SolidPatchRasterizer } from "../kind";

/** Every row whole. Nothing is evaluated, so a box costs its pages. */
const WHOLE: SolidPatchRasterizer = { row: () => "full" };

/** The whole box: every cell covered. What a patch is when it names no kind. */
export const boxSolidPatchKind: SolidPatchKind = {
  id: SOLID_PATCH_BOX_KIND,
  analyticShape: SOLID_PATCH_ANALYTIC_SHAPE.box,
  validate() {},
  fold() {},
  rasterizer: () => WHOLE,
};
