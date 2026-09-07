import type { VoxelToolPlugin } from "../plugin";
import { beginShapeGesture, voxelToolUnavailable, shellControl, sizeControl, depthControl, planeControl, mirrorControl } from "../geometry";

export const buildTool: VoxelToolPlugin = {
  id: "build", version: 1,
  ui: { label: "Build", hint: "Drag to build on a solid face. Empty space uses the construction height.", group: "Construct", order: 0,
    icon: "M12 4v16M4 12h16", controls: [shellControl, sizeControl, depthControl, planeControl, mirrorControl] },
  unavailable: voxelToolUnavailable,
  begin: (context) => beginShapeGesture(context, "fill", "brush", "box"),
};
