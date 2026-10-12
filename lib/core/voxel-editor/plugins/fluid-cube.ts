import type { VoxelToolPlugin } from "../plugin";
import { FLUID_OPPOSITE, beginFluidShapeGesture, fluidToolDefaults, fluidHeightControl, fluidRemoveControl, fluidSizeControl, fluidToolUnavailable } from "../fluid-geometry";
export const fluidCubeTool: VoxelToolPlugin = {
  id: "fluid-cube", version: 1, execution: "release",
  ui: { label: "Water cube", hint: "Drag to position a cube of water. Release to apply.", group: "Fluid", order: 21,
    notice: "Changes moving water; scene Undo does not reverse fluid motion.", opposite: FLUID_OPPOSITE,
    icon: "M4 4h16v16H4ZM4 13c5-3 11 3 16 0",
    controls: [{ ...fluidSizeControl, label: "Edge · voxels" }, fluidRemoveControl, fluidHeightControl] },
  defaults: (scene, values) => fluidToolDefaults(scene, values, "cube"),
  unavailable: fluidToolUnavailable,
  begin: context => beginFluidShapeGesture(context, "cube"),
};
