import { createVoxelToolRegistry } from "./plugin";
import { penTool } from "./plugins/pen";
import { shapeTool } from "./plugins/shape";
import { pasteTool } from "./plugins/paste";

import { selectTool } from "./plugins/select";

import { fluidBallTool } from "./plugins/fluid-ball";
import { fluidCubeTool } from "./plugins/fluid-cube";
import { fluidTorusTool } from "./plugins/fluid-torus";

export const voxelTools = createVoxelToolRegistry([penTool, shapeTool, pasteTool, selectTool, fluidBallTool, fluidCubeTool, fluidTorusTool]);
