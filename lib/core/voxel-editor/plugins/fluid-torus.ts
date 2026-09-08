import type { VoxelToolPlugin } from "../plugin";
import { beginFluidShapeGesture, fluidToolDefaults, fluidHeightControl, fluidRemoveControl, fluidSizeControl, fluidToolUnavailable, fluidTubeControl } from "../fluid-geometry";
export const fluidTorusTool: VoxelToolPlugin = {
  id: "fluid-torus", version: 1, execution: "release",
  ui: { label: "Water torus", hint: "Place a horizontal ring of water. Release to apply.", group: "Fluid", order: 22,
    notice: "Changes moving water; scene Undo does not reverse fluid motion.",
    icon: "M3 12a9 6 0 1 0 18 0a9 6 0 1 0-18 0M8 12a4 2 0 1 0 8 0a4 2 0 1 0-8 0",
    controls: [{ ...fluidSizeControl, label: "Outer diameter · voxels", initial: 8 }, fluidTubeControl, fluidRemoveControl, fluidHeightControl] },
  defaults: (scene, values) => fluidToolDefaults(scene, values, "torus"),
  unavailable: fluidToolUnavailable,
  begin: context => beginFluidShapeGesture(context, "torus"),
};
