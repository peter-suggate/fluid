import { materialControl } from "../controls";
import type { VoxelRegionVerbPlugin } from "../verb";
import { regionCellCount, regionPatches } from "../voxel-grid";
import { plural, solidEditUnavailable } from "./region";

/** Make every cell of the selection whole and solid, in the chosen material — the cells that were solid already included. */
export const fillVerb: VoxelRegionVerbPlugin = {
  id: "fill",
  version: 1,
  ui: {
    label: "Fill", hint: "Make every cell of the selection solid", group: "Edit", order: 1,
    icon: "M5 5h14v14H5zM9 9h6v6H9z",
  },
  controls: [materialControl],
  unavailable: solidEditUnavailable,
  run: ({ region, materialId }) => ({
    patches: regionPatches("fill", region, materialId),
    caption: `Filled ${plural(regionCellCount(region), "voxel")}`,
  }),
};
