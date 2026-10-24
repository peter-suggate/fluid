import { inscribedQuadricRasterizer } from "../inscribed-quadric";
import { ellipsoidInsets } from "../ellipsoid-bounds";
import { SOLID_PATCH_ANALYTIC_SHAPE, SOLID_PATCH_ELLIPSOID_KIND, type SolidPatchKind } from "../kind";

/** Optional payload insets keep a subcell ellipsoid inside its integer coverage
 * box. Legacy patches remain inscribed, with the original one-cell axes. */
export const ellipsoidSolidPatchKind: SolidPatchKind = {
  id: SOLID_PATCH_ELLIPSOID_KIND,
  analyticShape: SOLID_PATCH_ANALYTIC_SHAPE.ellipsoid,
  validate(patch) { ellipsoidInsets(patch); },
  fold(patch, word) {
    if (patch.payload === undefined) return;
    word(0x53554243);
    for (const value of new Uint32Array(new Float64Array(ellipsoidInsets(patch)).buffer)) word(value);
  },
  resized(patch, minimum, maximumExclusive) {
    if (patch.payload === undefined) return { ...patch, minimum, maximumExclusive };
    const insets = ellipsoidInsets(patch).map((v, i) => {
      const a = i % 3;
      return v * (maximumExclusive[a]! - minimum[a]!) / (patch.maximumExclusive[a]! - patch.minimum[a]!);
    });
    return { ...patch, minimum, maximumExclusive, payload: JSON.stringify(insets) };
  },
  rasterizer: (patch) => inscribedQuadricRasterizer(patch, undefined),
};
