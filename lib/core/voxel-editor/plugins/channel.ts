import type { VoxelToolPlugin } from "../plugin";
import { beginShapeGesture, sizeControl, depthControl, planeControl, mirrorControl } from "../geometry";

export const channelTool: VoxelToolPlugin = {
  id: "channel", version: 1,
  ui: { label: "Channel", hint: "Drag a straight channel through solids. Width and depth set its section.", group: "Subtract", order: 7,
    icon: "M4 4v16h16M8 4v12h12", controls: [sizeControl, depthControl, planeControl, mirrorControl] },
  begin: (context) => beginShapeGesture(context, "clear", "line", "box"),
};
