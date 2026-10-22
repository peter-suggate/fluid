import { createVoxelToolRegistry } from "./plugin";
import { voxelBrushTool } from "./plugins/voxel";
import { penTool } from "./plugins/pen";
import { faceTool } from "./plugins/face";
import { boxTool } from "./plugins/box";
import { lineTool } from "./plugins/line";
import { centerTool } from "./plugins/center";
import { patternTool } from "./plugins/pattern";

import { marqueeTool } from "./plugins/marquee";
import { wandTool } from "./plugins/wand";

import { fluidBallTool } from "./plugins/fluid-ball";
import { fluidCubeTool } from "./plugins/fluid-cube";
import { fluidTorusTool } from "./plugins/fluid-torus";

export const voxelTools = createVoxelToolRegistry([penTool, voxelBrushTool, faceTool, boxTool, lineTool, centerTool, patternTool, marqueeTool, wandTool, fluidBallTool, fluidCubeTool, fluidTorusTool]);
