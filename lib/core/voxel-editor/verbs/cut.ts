import type { VoxelRegionVerbPlugin } from "../verb";
import { regionPatches } from "../voxel-grid";
import { copiedContents } from "./copy";
import { plural, transformUnavailable } from "./region";

/**
 * Copy, then clear: the selection's solids go to the clipboard and off the scene.
 *
 * One edit and one undo entry — the clear. The clipboard is not document data,
 * so undoing a cut puts the voxels back and leaves the copy where it is. The
 * selection stays standing over the hole, which is where a paste would go.
 */
export const cutVerb: VoxelRegionVerbPlugin = {
  id: "cut",
  version: 1,
  ui: {
    label: "Cut", hint: "Move the selection's voxels to the clipboard", group: "Clipboard", order: 21, menu: "Clipboard",
    tone: "danger",
    icon: "M20 5L8.5 15.5M20 19L8.5 8.5M6 4.5a2.5 2.5 0 1 0 0 5a2.5 2.5 0 1 0 0-5ZM6 14.5a2.5 2.5 0 1 0 0 5a2.5 2.5 0 1 0 0-5Z",
    keys: ["Mod+x"],
  },
  unavailable: transformUnavailable,
  run: (context) => {
    const { clipboard, count } = copiedContents(context);
    return { patches: regionPatches("clear", context.region), clipboard, caption: `Cut ${plural(count, "voxel")}` };
  },
};
