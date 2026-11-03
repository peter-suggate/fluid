import { createVoxelToolRegistry } from "./plugin";
import { penTool } from "./plugins/pen";
import { faceTool } from "./plugins/face";
import { lineTool } from "./plugins/line";
import { shapeTool } from "./plugins/shape";
import { patternTool } from "./plugins/pattern";

import { marqueeTool } from "./plugins/marquee";
import { wandTool } from "./plugins/wand";

import { fluidBallTool } from "./plugins/fluid-ball";
import { fluidCubeTool } from "./plugins/fluid-cube";
import { fluidTorusTool } from "./plugins/fluid-torus";

export const voxelTools = createVoxelToolRegistry([penTool, shapeTool, faceTool, lineTool, patternTool, marqueeTool, wandTool, fluidBallTool, fluidCubeTool, fluidTorusTool]);
