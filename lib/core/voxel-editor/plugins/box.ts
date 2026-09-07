import type { VoxelToolPlugin } from "../plugin";
import { beginShapeGesture, depthControl, planeControl, mirrorControl } from "../geometry";

export const boxTool: VoxelToolPlugin = {
  id: "box", version: 1,
  ui: { label: "Box", hint: "Drag a rectangular footprint. Depth extrudes it from the starting face.", group: "Construct", order: 2,
    icon: "M4 4h16v16H4Z", controls: [depthControl, planeControl, mirrorControl] },
  begin: (context) => beginShapeGesture(context, "fill", "box", "box"),
};
