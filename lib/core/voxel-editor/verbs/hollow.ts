import type { ToolNumberControl } from "../plugin";
import type { VoxelRegionVerbPlugin } from "../verb";
import { blockOfRegion, contentsReplacementPatches, hollowBlock } from "../voxel-grid";
import { blockSolidCount, plural, transformUnavailable } from "./region";

const wallControl: ToolNumberControl = {
  id: "wall", label: "Hollow wall · voxels", short: "Wall", min: 1, max: 16, step: 1, initial: 1,
};

/**
 * Remove the inside of the selection's solids, leaving a skin.
 *
 * The skin is `wall` cells thick, counted across edges and corners
 * (`blockInterior`), and the selection's own box counts as open air — so a
 * selection cut through a larger solid is skinned on the cut faces too, and
 * what is left is a closed shell of exactly what was selected.
 */
export const hollowVerb: VoxelRegionVerbPlugin = {
  id: "hollow",
  version: 1,
  ui: {
    label: "Hollow", group: "Edit", order: 3, menu: "Shape",
    hint: "Remove the inside and keep a skin. A wall of 1 can vanish under smooth rendering; 2 is safe",
    icon: "M12 4a8 8 0 1 0 0 16a8 8 0 1 0 0-16ZM12 9a3 3 0 1 0 0 6a3 3 0 1 0 0-6Z",
  },
  controls: [wallControl],
  unavailable: transformUnavailable,
  run: ({ reader, region, values }) => {
    const wall = values[wallControl.id]!;
    const contents = blockOfRegion(reader, region);
    const hollowed = hollowBlock(contents, wall);
    const removed = blockSolidCount(contents) - blockSolidCount(hollowed);
    if (removed === 0) {
      throw new Error(`No voxel of this selection is more than ${plural(wall, "voxel")} from its surface: there is no inside to remove.`);
    }
    return {
      patches: contentsReplacementPatches(reader, region, { minimum: region.minimum, block: hollowed }),
      caption: `Hollowed out ${plural(removed, "voxel")}`,
    };
  },
};
