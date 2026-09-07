import type { VoxelToolPlugin } from "../plugin";
import { beginShapeGesture, voxelToolUnavailable, shellControl, sizeControl, planeControl, mirrorControl } from "../geometry";

export const sphereTool: VoxelToolPlugin = {
  id: "sphere", version: 1,
  ui: { label: "Sphere", hint: "Place a voxel sphere; drag to move it along the starting plane.", group: "Construct", order: 4,
    icon: "M12 3a9 9 0 1 0 0 18a9 9 0 1 0 0-18", controls: [shellControl, { ...sizeControl, initial: 5 }, planeControl, mirrorControl] },
  unavailable: voxelToolUnavailable,
  begin: (context) => beginShapeGesture(context, "fill", "stamp", "sphere"),
};
