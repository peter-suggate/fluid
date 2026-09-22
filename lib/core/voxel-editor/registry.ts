import { createVoxelToolRegistry } from "./plugin";
import { pushPullTool } from "./plugins/push-pull";
import { buildTool } from "./plugins/build";
import { carveTool } from "./plugins/carve";
import { boxTool } from "./plugins/box";
import { cutTool } from "./plugins/cut";
import { sphereTool } from "./plugins/sphere";
import { drillTool } from "./plugins/drill";
import { wallTool } from "./plugins/wall";
import { channelTool } from "./plugins/channel";

import { fluidBallTool } from "./plugins/fluid-ball";
import { fluidCubeTool } from "./plugins/fluid-cube";
import { fluidTorusTool } from "./plugins/fluid-torus";

export const voxelTools = createVoxelToolRegistry([pushPullTool, buildTool, carveTool, boxTool, cutTool, sphereTool, drillTool, wallTool, channelTool, fluidBallTool, fluidCubeTool, fluidTorusTool]);
