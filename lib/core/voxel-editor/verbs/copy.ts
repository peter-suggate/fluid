import type { VoxelClipboard, VoxelRegionVerbPlugin, VoxelRegionVerbTarget } from "../verb";
import { blockOfRegion, croppedBlock } from "../voxel-grid";
import { blockSolidCount, contentsUnreadable, plural } from "./region";

/**
 * The selection's solids as a clipboard holds them: cut down to the box around
 * the solid cells, so what later follows the pointer is the piece and not the
 * margin of air that happened to be selected with it.
 */
export function copiedContents(target: Pick<VoxelRegionVerbTarget, "reader" | "region">):
  { readonly clipboard: VoxelClipboard; readonly count: number } {
  const solid = croppedBlock(blockOfRegion(target.reader, target.region));
  if (!solid) throw new Error("No solid voxels in this selection.");
  const { minimum } = target.region;
  return {
    clipboard: { block: solid.block,
      minimum: [minimum[0] + solid.offset[0], minimum[1] + solid.offset[1], minimum[2] + solid.offset[2]] },
    count: blockSolidCount(solid.block),
  };
}

/**
 * Lift a copy of the selection's solids onto the clipboard.
 *
 * Not an edit: nothing is written and nothing is recorded, so it is offered
 * even where solid edits are not — a copy made under running water can still
 * be pasted once the water is off.
 */
export const copyVerb: VoxelRegionVerbPlugin = {
  id: "copy",
  version: 1,
  ui: {
    label: "Copy", hint: "Copy the selection's voxels to the clipboard", group: "Clipboard", order: 20, menu: "Clipboard",
    icon: "M9 9h11v11H9zM5 15V4h11",
    keys: ["Mod+c"],
  },
  unavailable: contentsUnreadable,
  run: (context) => {
    const { clipboard, count } = copiedContents(context);
    return { patches: [], clipboard, caption: `Copied ${plural(count, "voxel")}` };
  },
};
