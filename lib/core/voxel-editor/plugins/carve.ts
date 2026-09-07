import type { VoxelToolPlugin } from "../plugin";
import { beginShapeGesture, voxelToolUnavailable, shellControl, sizeControl, depthControl, planeControl, mirrorControl } from "../geometry";

export const carveTool: VoxelToolPlugin = {
  id: "carve", version: 1,
  ui: { label: "Carve", hint: "Drag to remove solid voxels. Depth cuts into the picked face.", group: "Subtract", order: 1,
    icon: "M4 12h16", controls: [shellControl, sizeControl, depthControl, planeControl, mirrorControl] },
  unavailable: voxelToolUnavailable,
  begin: (context) => beginShapeGesture(context, "clear", "brush", "box"),
};
