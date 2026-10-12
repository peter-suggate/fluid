import type { VoxelRegionVerbPlugin } from "../verb";
import { blockOfRegion, contentsReplacementPatches, peeledBlock, solidRegionOfBlock } from "../voxel-grid";
import { blockSolidCount, plural, transformUnavailable } from "./region";

/**
 * Peel one layer of voxels off every exposed side of the selection's solids.
 *
 * The inverse of Grow by the same rule: a voxel goes when any of its six face
 * neighbours is empty, so a side standing against other solids is left alone.
 * The selection afterwards is exactly what remains. A piece too thin to have
 * anything under its skin is refused — that is a Clear, and saying so is
 * better than a shrink that quietly deletes the lot.
 */
export const shrinkVerb: VoxelRegionVerbPlugin = {
  id: "shrink",
  version: 1,
  ui: {
    label: "Shrink", hint: "Peel one layer of voxels off every exposed side", group: "Edit", order: 5, menu: "Shape",
    icon: "M5 5h14v14H5zM12 9v2M12 13v2M9 12h2M13 12h2",
  },
  unavailable: transformUnavailable,
  run: ({ reader, region }) => {
    const contents = blockOfRegion(reader, region);
    const halo = reader.read([region.minimum[0] - 1, region.minimum[1] - 1, region.minimum[2] - 1],
      [region.maximumExclusive[0] + 1, region.maximumExclusive[1] + 1, region.maximumExclusive[2] + 1]);
    const peeled = peeledBlock(contents, halo);
    const remaining = solidRegionOfBlock(peeled, region.minimum);
    if (!remaining) throw new Error("Every voxel of this selection is on its surface: shrinking would remove them all. Use Clear for that.");
    const removed = blockSolidCount(contents) - blockSolidCount(peeled);
    if (removed === 0) throw new Error("No voxel of this selection has an exposed side to peel.");
    return {
      patches: contentsReplacementPatches(reader, region, { minimum: region.minimum, block: peeled }),
      region: remaining,
      caption: `Shrank by ${plural(removed, "voxel")}`,
    };
  },
};
