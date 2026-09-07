import { createVoxelToolRegistry } from "./plugin";
import { buildTool } from "./plugins/build";
import { carveTool } from "./plugins/carve";
import { boxTool } from "./plugins/box";
import { cutTool } from "./plugins/cut";
import { sphereTool } from "./plugins/sphere";
import { drillTool } from "./plugins/drill";
import { wallTool } from "./plugins/wall";
import { channelTool } from "./plugins/channel";

export const voxelTools = createVoxelToolRegistry([buildTool, carveTool, boxTool, cutTool, sphereTool, drillTool, wallTool, channelTool]);
