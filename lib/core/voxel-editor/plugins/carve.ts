import type { VoxelToolPlugin } from "../plugin";
import { beginShapeGesture, voxelToolUnavailable, shellControl, sizeControl, depthControl, mirrorControl } from "../geometry";

export const carveTool: VoxelToolPlugin = {
  id: "carve", version: 1,
  ui: { label: "Carve", hint: "Drag across solids to cut them away.", group: "Subtract", order: 1,
    icon: "M4 12h16", controls: [shellControl, sizeControl, depthControl, mirrorControl] },
  unavailable: voxelToolUnavailable,
  begin: (context) => beginShapeGesture(context, "clear", "brush", "box"),
};
