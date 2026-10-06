import type { VoxelToolPlugin } from "../plugin";
import { beginShapeGesture, voxelToolUnavailable, shellControl, depthControl, mirrorControl } from "../geometry";

export const cutTool: VoxelToolPlugin = {
  id: "cut", version: 1,
  ui: { label: "Cut", hint: "Drag a rectangular cut. Depth extends into the starting face.", group: "Subtract", order: 3,
    icon: "M4 4h16v16H4ZM8 12h8", controls: [shellControl, depthControl, mirrorControl] },
  unavailable: voxelToolUnavailable,
  begin: (context) => beginShapeGesture(context, "clear", "box", "box"),
};
