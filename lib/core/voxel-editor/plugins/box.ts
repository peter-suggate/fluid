import type { VoxelToolPlugin } from "../plugin";
import { beginShapeGesture, voxelToolUnavailable, shellControl, depthControl, planeControl, mirrorControl } from "../geometry";

export const boxTool: VoxelToolPlugin = {
  id: "box", version: 1,
  ui: { label: "Box", hint: "Drag a rectangular footprint. Depth extrudes it from the starting face.", group: "Construct", order: 2,
    icon: "M4 4h16v16H4Z", controls: [shellControl, depthControl, planeControl, mirrorControl] },
  unavailable: voxelToolUnavailable,
  begin: (context) => beginShapeGesture(context, "fill", "box", "box"),
};
