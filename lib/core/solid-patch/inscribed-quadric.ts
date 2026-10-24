import type { SolidWorldCoordinate, SolidWorldVoxelPatch } from "../solid-world";
import type { SolidPatchRasterizer } from "./kind";
import { ellipsoidBounds } from "./ellipsoid-bounds";

/**
 * The round shapes a patch box inscribes: an ellipsoid over its round axes and
 * a prism along its flat ones. All three axes round is the ellipsoid; one flat
 * is an elliptic cylinder along it.
 *
 * An axis the box is one cell wide on is always flat. A radius of half a cell
 * is not a curve the lattice can hold — it would thin every cell of the layer
 * to a sliver — so a one-cell-thick ellipsoid is a disc, and a one-cell box is
 * a whole cell.
 *
 * Distance is the first-order estimate (k0 - 1) / |grad k0| with
 * k0 = |p / r|, which is k0 (k0 - 1) / k1 with k1 = |p / r^2| (Quilez's
 * ellipsoid bound). It is exact for equal radii. Against the true distance,
 * over the half-cell band coverage is read from, it is off by at most 0.04 of
 * a cell up to a 4:1 box and 0.13 at 16:1, where the error sits at the tips.
 * It is also a closed form, so the renderer's analytic record can evaluate the
 * very same expression.
 */

/** Half sizes of the box in cells, and which axes are flat. */
interface InscribedQuadric {
  readonly centre: SolidWorldCoordinate;
  readonly half: SolidWorldCoordinate;
  /** 1 / radius on a round axis, 0 on a flat one. */
  readonly inverseRadius: SolidWorldCoordinate;
  /** The smallest round radius; infinite when every axis is flat. */
  readonly minimumRadius: number;
}

function inscribedQuadric(patch: Pick<SolidWorldVoxelPatch, "minimum" | "maximumExclusive" | "payload">,
  flatAxis: number | undefined): InscribedQuadric {
  const subcell = flatAxis === undefined && patch.payload !== undefined;
  if (subcell) {
    patch = { ...patch, ...ellipsoidBounds(patch as SolidWorldVoxelPatch) };
  }
  const half = patch.minimum.map((value, axis) => 0.5 * (patch.maximumExclusive[axis]! - value));
  const round = half.map((value, axis) => axis !== flatAxis && (value > 0.5 || subcell));
  return {
    centre: patch.minimum.map((value, axis) => value + half[axis]!) as unknown as SolidWorldCoordinate,
    half: half as unknown as SolidWorldCoordinate,
    inverseRadius: half.map((value, axis) => round[axis] ? 1 / value : 0) as unknown as SolidWorldCoordinate,
    minimumRadius: Math.min(...half.map((value, axis) => round[axis] ? value : Number.POSITIVE_INFINITY)),
  };
}

/**
 * Signed distance in cells from a lattice point to the inscribed shape,
 * negative inside. `flatAxis` is the cylinder's axis; absent is the ellipsoid.
 *
 * This is the whole definition of both shapes, for any point. The rasterizer
 * below is the same expression at cell centres, where the flat axes' slab is
 * always at least half a cell away and so never decides anything.
 */
export function inscribedQuadricSignedDistance_cells(
  patch: Pick<SolidWorldVoxelPatch, "minimum" | "maximumExclusive" | "payload">,
  flatAxis: number | undefined,
  point: SolidWorldCoordinate,
): number {
  const shape = inscribedQuadric(patch, flatAxis);
  let s0 = 0, s1 = 0, slab = Number.NEGATIVE_INFINITY;
  for (let axis = 0; axis < 3; axis += 1) {
    const offset = point[axis]! - shape.centre[axis]!, inverse = shape.inverseRadius[axis]!;
    if (inverse === 0) slab = Math.max(slab, Math.abs(offset) - shape.half[axis]!);
    s0 += (offset * inverse) ** 2;
    s1 += (offset * inverse * inverse) ** 2;
  }
  if (shape.minimumRadius === Number.POSITIVE_INFINITY) return slab;
  const k0 = Math.sqrt(s0);
  return Math.max(slab, s1 > 0 ? k0 * (k0 - 1) / Math.sqrt(s1) : -shape.minimumRadius);
}

/** Coverage of a cell whose centre is `distance` cells from the surface. */
export const coverageForSignedDistance = (distance: number): number =>
  Math.round(255 * Math.max(0, Math.min(1, 0.5 - distance)));

export function inscribedQuadricRasterizer(patch: SolidWorldVoxelPatch,
  flatAxis: number | undefined): SolidPatchRasterizer {
  const { centre, inverseRadius: [ix, iy, iz], minimumRadius } = inscribedQuadric(patch, flatAxis);
  // Every axis flat: the shape is its box.
  if (minimumRadius === Number.POSITIVE_INFINITY) return { row: () => "full" };
  // The cell centre nearest the shape's own: on it for an odd width, half a cell off for an even one.
  const centreOffset = Math.abs(centre[0] - Math.floor(centre[0]) - 0.5);
  return {
    row(y, z, x0, x1, out) {
      const qy = (y + 0.5 - centre[1]) * iy, qz = (z + 0.5 - centre[2]) * iz;
      const s0 = qy * qy + qz * qz, s1 = (qy * iy) ** 2 + (qz * iz) ** 2;
      const first = x0 + 0.5 - centre[0], last = x1 - 0.5 - centre[0];
      // k0 is 1/minimumRadius-Lipschitz, so (1 - k0) * minimumRadius never
      // overstates how deep a cell is, nor (k0 - 1) * minimumRadius how far
      // out; and the estimate below is never shallower than either. Along a
      // row k0 only grows away from the centre, so its ends settle the row.
      const farthest = Math.max(Math.abs(first), Math.abs(last)) * ix;
      if ((1 - Math.sqrt(s0 + farthest * farthest)) * minimumRadius >= 0.5) return "full";
      const nearest = (first > 0 ? first : last < 0 ? -last : centreOffset) * ix;
      if ((Math.sqrt(s0 + nearest * nearest) - 1) * minimumRadius >= 0.5) return "empty";
      let covered = 0, whole = 0;
      const count = x1 - x0;
      for (let index = 0; index < count; index += 1) {
        const qx = (first + index) * ix;
        const k0 = Math.sqrt(s0 + qx * qx), k1 = Math.sqrt(s1 + (qx * ix) ** 2);
        const distance = k1 > 0 ? k0 * (k0 - 1) / k1 : -minimumRadius;
        const coverage = coverageForSignedDistance(distance);
        out.coverage[index] = coverage;
        // Half a cell inside is as deep as a covered cell is ever recorded: what a box writes.
        out.signedDistanceQ8[index] = coverage === 255 ? -128
          : Math.max(-0x8000, Math.min(0x7fff, Math.round(256 * distance)));
        out.materialId[index] = 0;
        if (coverage > 0) covered += 1;
        if (coverage === 255) whole += 1;
      }
      return covered === 0 ? "empty" : whole === count ? "full" : "cells";
    },
  };
}
