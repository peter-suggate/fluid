import type { VoxelRegionVerbPlugin } from "../verb";
import { regionPatches } from "../voxel-grid";
import { contentsUnavailable, solidsOf } from "./region";

/**
 * Remove the solids inside the selection.
 *
 * The verb the swept box was introduced for, and the meaning of Delete while
 * one stands. Withheld over air: a clear that removes nothing would still be an
 * undo entry.
 */
export const clearVerb: VoxelRegionVerbPlugin = {
  id: "clear",
  version: 1,
  ui: {
    label: "Clear", hint: "Remove the solid voxels in the selection", group: "Edit", order: 0, tone: "danger",
    icon: "M5 7h14M10 7V4h4v3M7 7l1 13h8l1-13",
    keys: ["Delete", "Backspace"],
  },
  unavailable: contentsUnavailable,
  run: (context) => ({
    patches: regionPatches("clear", context.region),
    caption: `Cleared ${solidsOf(context)}`,
  }),
};
