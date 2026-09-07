import type { VoxelToolPlugin } from "../plugin";
import { beginShapeGesture, sizeControl, depthControl, planeControl, mirrorControl } from "../geometry";

export const drillTool: VoxelToolPlugin = {
  id: "drill", version: 1,
  ui: { label: "Drill", hint: "Place a cylindrical cut along the picked face normal.", group: "Subtract", order: 5,
    icon: "M7 3h10v18H7ZM4 12h16", controls: [sizeControl, depthControl, planeControl, mirrorControl] },
  begin: (context) => beginShapeGesture(context, "clear", "stamp", "cylinder"),
};
