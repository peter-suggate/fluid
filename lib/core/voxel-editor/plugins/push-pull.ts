import type { VoxelToolPlugin } from "../plugin";
import { beginPushPullGesture, voxelToolUnavailable, shellControl, planeControl, mirrorControl } from "../geometry";

export const pushPullTool: VoxelToolPlugin = {
  id: "push-pull", version: 1,
  ui: { label: "Push / pull", hint: "Drag a footprint on a face and let go. Then pull out to build or push in to carve, and click.", group: "Construct", order: -1,
    icon: "M5 14h14v6H5ZM12 12V3M8.5 6.5 12 3l3.5 3.5", controls: [shellControl, planeControl, mirrorControl] },
  unavailable: voxelToolUnavailable,
  begin: beginPushPullGesture,
};
