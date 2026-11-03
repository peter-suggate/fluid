import { inscribedQuadricRasterizer } from "../inscribed-quadric";
import { resizedSubcellPatch, subcellInsets } from "../ellipsoid-bounds";
import { SOLID_PATCH_ANALYTIC_SHAPE, SOLID_PATCH_ELLIPSOID_KIND, type SolidPatchKind } from "../kind";

/** Optional payload insets keep a subcell ellipsoid inside its integer coverage
 * box. Legacy patches remain inscribed, with the original one-cell axes. */
export const ellipsoidSolidPatchKind: SolidPatchKind = {
  id: SOLID_PATCH_ELLIPSOID_KIND,
  analyticShape: SOLID_PATCH_ANALYTIC_SHAPE.ellipsoid,
  validate(patch) { subcellInsets(patch); },
  fold(patch, word) {
    if (patch.payload === undefined) return;
    word(0x53554243);
    for (const value of new Uint32Array(new Float64Array(subcellInsets(patch)).buffer)) word(value);
  },
  resized: resizedSubcellPatch,
  rasterizer: (patch) => inscribedQuadricRasterizer(patch, undefined),
};
