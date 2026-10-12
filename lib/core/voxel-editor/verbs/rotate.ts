import type { VoxelRegionVerbPlugin } from "../verb";
import { blockOfRegion, contentsReplacementPatches, quarterTurnedRegion, rotateBlock90 } from "../voxel-grid";
import { verbAxis } from "./flip";
import { assertInsideLattice, blockSolidCount, plural, transformUnavailable } from "./region";

const AXES = ["X", "Y", "Z"] as const;

/**
 * Turn the selection's solids a quarter turn about the centre of its box.
 *
 * Right-handed about the chosen axis: about Y, +Z swings to +X. The box turns
 * with its contents — its two sizes across the axis trade — and the selection
 * follows. Where the turned box lands when the centre is not a point the
 * lattice can turn about is `quarterTurnedMinimum`'s rule: half a cell aside,
 * always to the same vertex, so four turns come home.
 */
export const rotateVerb: VoxelRegionVerbPlugin = {
  id: "rotate",
  version: 1,
  ui: {
    label: "Rotate", hint: "Turn the selection's voxels a quarter turn about the centre of its box",
    group: "Transform", order: 11,
    icon: "M19 12a7 7 0 1 1-2.5-5.4M17 3v4h-4",
    variants: AXES.map((name, axis) => ({
      id: name.toLowerCase(), label: name, hint: `Turn the selection's voxels a quarter turn about ${name}`, values: { axis },
    })),
  },
  unavailable: transformUnavailable,
  run: ({ scene, reader, region, values }) => {
    const axis = verbAxis(values, "Rotate");
    const contents = blockOfRegion(reader, region);
    const turned = quarterTurnedRegion(region, axis);
    const placed = { minimum: turned.minimum, block: rotateBlock90(contents, axis, 1) };
    assertInsideLattice({ scene }, placed);
    return {
      patches: contentsReplacementPatches(reader, region, placed),
      region: turned,
      caption: `Rotated ${plural(blockSolidCount(contents), "voxel")} about ${AXES[axis]}`,
    };
  },
};
