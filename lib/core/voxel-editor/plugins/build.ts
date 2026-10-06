import type { VoxelToolPlugin } from "../plugin";
import { beginShapeGesture, voxelToolUnavailable, shellControl, sizeControl, depthControl, mirrorControl } from "../geometry";

export const buildTool: VoxelToolPlugin = {
  id: "build", version: 1,
  ui: { label: "Build", hint: "Drag across a solid face to build on it.", group: "Construct", order: 0,
    icon: "M12 4v16M4 12h16", controls: [shellControl, sizeControl, depthControl, mirrorControl] },
  unavailable: voxelToolUnavailable,
  begin: (context) => beginShapeGesture(context, "fill", "brush", "box"),
};
