import type { VoxelToolPlugin } from "../plugin";
import { beginShapeGesture, sizeControl, depthControl, planeControl, mirrorControl } from "../geometry";

export const wallTool: VoxelToolPlugin = {
  id: "wall", version: 1,
  ui: { label: "Wall", hint: "Drag a straight wall between two points. Width and depth set its section.", group: "Construct", order: 6,
    icon: "M4 20 20 4M4 14 14 4", controls: [sizeControl, depthControl, planeControl, mirrorControl] },
  begin: (context) => beginShapeGesture(context, "fill", "line", "box"),
};
