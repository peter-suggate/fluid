import { coverageForSignedDistance } from "../inscribed-quadric";
import { resizedSubcellPatch, subcellBounds, subcellInsets } from "../ellipsoid-bounds";
import { SOLID_PATCH_CUBOID_KIND, type SolidPatchKind, type SolidPatchRasterizer } from "../kind";

/**
 * An axis-aligned box whose faces stand at fractions of a cell: the subcell
 * insets in `payload` (`../ellipsoid-bounds`) inside the patch's whole-cell
 * box. A box centred on the exact point a hand pressed, rather than snapped to
 * the lattice.
 *
 * Its own kind rather than a payload on `box`, because a box is read as whole
 * cells in many places — the solver's fast regions, the vessel outline, the
 * lattice — and a fractional one must never be mistaken for one of those.
 *
 * Distance is exact: the box's signed distance at each cell centre, so a face
 * a quarter-cell into a layer covers it by a quarter, and the edges and corners
 * round as the same field does on the GPU's sculpt record for a box.
 */
export const cuboidSolidPatchKind: SolidPatchKind = {
  id: SOLID_PATCH_CUBOID_KIND,
  validate(patch) { subcellInsets(patch); },
  fold(patch, word) {
    if (patch.payload === undefined) return;
    word(0x53554243);
    for (const value of new Uint32Array(new Float64Array(subcellInsets(patch)).buffer)) word(value);
  },
  resized: resizedSubcellPatch,
  rasterizer(patch): SolidPatchRasterizer {
    const { minimum, maximumExclusive } = subcellBounds(patch);
    const centre = minimum.map((v, a) => (v + maximumExclusive[a]!) / 2);
    const half = minimum.map((v, a) => (maximumExclusive[a]! - v) / 2);
    return {
      row(y, z, x0, x1, out) {
        const qy = Math.abs(y + 0.5 - centre[1]!) - half[1]!, qz = Math.abs(z + 0.5 - centre[2]!) - half[2]!;
        // Every cell of this row is at least this far outside: half a cell clears it.
        if (Math.max(qy, qz) >= 0.5) return "empty";
        let covered = 0, whole = 0;
        const count = x1 - x0;
        for (let index = 0; index < count; index += 1) {
          const qx = Math.abs(x0 + index + 0.5 - centre[0]!) - half[0]!;
          const distance = Math.hypot(Math.max(qx, 0), Math.max(qy, 0), Math.max(qz, 0)) + Math.min(Math.max(qx, qy, qz), 0);
          const coverage = coverageForSignedDistance(distance);
          out.coverage[index] = coverage;
          out.signedDistanceQ8[index] = coverage === 255 ? -128
            : Math.max(-0x8000, Math.min(0x7fff, Math.round(256 * distance)));
          out.materialId[index] = 0;
          if (coverage > 0) covered += 1;
          if (coverage === 255) whole += 1;
        }
        return covered === 0 ? "empty" : whole === count ? "full" : "cells";
      },
    };
  },
};
