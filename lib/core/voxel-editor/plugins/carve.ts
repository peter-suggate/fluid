import type { VoxelToolPlugin } from "../plugin";
import { beginShapeGesture, sizeControl, depthControl, planeControl, mirrorControl } from "../geometry";

export const carveTool: VoxelToolPlugin = {
  id: "carve", version: 1,
  ui: { label: "Carve", hint: "Drag to remove solid voxels. Depth cuts into the picked face.", group: "Subtract", order: 1,
    icon: "M4 12h16", controls: [sizeControl, depthControl, planeControl, mirrorControl] },
  begin: (context) => beginShapeGesture(context, "clear", "brush", "box"),
};
