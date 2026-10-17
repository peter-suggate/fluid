/** The product scene, including the catalog environment and authored lighting. */
import "../blender/register-assets";
import { getScenePreset } from "../../lib/core/scenes";
export const createScene = () => getScenePreset("backyard-pool-party").create();
export const camera = getScenePreset("backyard-pool-party").camera;
