import type { VoxelToolPlugin } from "../plugin";
import { beginFluidShapeGesture, fluidToolDefaults, fluidHeightControl, fluidRemoveControl, fluidSizeControl, fluidToolUnavailable } from "../fluid-geometry";
export const fluidBallTool: VoxelToolPlugin = {
  id: "fluid-ball", version: 1, execution: "release",
  ui: { label: "Water ball", hint: "Drag to position a ball of water. Release to apply.", group: "Fluid", order: 20,
    notice: "Changes moving water; scene Undo does not reverse fluid motion.",
    icon: "M12 3a9 9 0 1 0 0 18a9 9 0 1 0 0-18M6 14c3-3 9 3 12-1",
    controls: [fluidSizeControl, fluidRemoveControl, fluidHeightControl] },
  defaults: (scene, values) => fluidToolDefaults(scene, values, "ball"),
  unavailable: context => fluidToolUnavailable(context, "ball"),
  begin: context => beginFluidShapeGesture(context, "ball"),
};
