import type { VoxelRegionVerbPlugin } from "../verb";
import { solidEditUnavailable } from "./region";

/** The tool that stamps the clipboard where the pointer is, by its registry id. */
export const PASTE_TOOL_ID = "pattern";

/**
 * Hand the clipboard to the pointer.
 *
 * A paste has to be told where, and the answer is a click — so this verb
 * writes nothing and arms the tool that stamps the clipboard under the
 * pointer. A standing selection stays where it is, as it does when any tool
 * is armed. It needs no selection to run: what it needs is something copied.
 */
export const pasteVerb: VoxelRegionVerbPlugin = {
  id: "paste",
  version: 1,
  ui: {
    label: "Paste", hint: "Stamp the clipboard's voxels where you click", group: "Clipboard", order: 22, menu: "Clipboard",
    icon: "M8 5H6v15h12V5h-2M9 3h6v4H9z",
    keys: ["Mod+v"],
    standalone: true,
  },
  unavailable: (target) => solidEditUnavailable(target)
    ?? (target.clipboard ? undefined : "Nothing copied yet: copy or cut a selection first."),
  run: () => ({ patches: [], armTool: PASTE_TOOL_ID, caption: "Click to stamp the copied voxels" }),
};
