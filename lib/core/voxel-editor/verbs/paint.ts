import { materialControl } from "../controls";
import type { VoxelRegionVerbPlugin } from "../verb";
import { regionPatches } from "../voxel-grid";
import { VOXEL_PALETTE } from "../../voxel-palette";
import { contentsUnavailable, solidsOf } from "./region";

const PAINT_ICON = "M12 3c3 4 6 7 6 10a6 6 0 1 1-12 0c0-3 3-6 6-10Z";

/**
 * Give the solids inside the selection a material. Shape is untouched, and air stays air.
 *
 * Run plainly it paints with the editor's chosen swatch — the bar's button,
 * beside the swatches themselves. On the ring it opens one wedge per palette
 * material instead, because a ring has no swatch row to choose from first.
 */
export const paintVerb: VoxelRegionVerbPlugin = {
  id: "paint",
  version: 1,
  ui: {
    label: "Paint", hint: "Repaint the solid voxels in the selection", group: "Edit", order: 2,
    icon: PAINT_ICON,
    variants: VOXEL_PALETTE.map((entry) => ({
      id: `material-${entry.id}`, label: entry.label, hint: `Repaint the selection's solid voxels as ${entry.label}`,
      icon: PAINT_ICON, swatch: entry.color, values: { [materialControl.id]: entry.id }, surface: "ring" as const,
    })),
  },
  controls: [materialControl],
  unavailable: contentsUnavailable,
  run: (context) => ({
    patches: regionPatches("paint", context.region, context.materialId),
    caption: `Painted ${solidsOf(context)}`,
  }),
};
