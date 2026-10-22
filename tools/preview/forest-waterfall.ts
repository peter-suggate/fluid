/** The standard registered waterfall scene, through the production renderer. */
import { registerSceneMeshAssets } from "../blender/register-assets";
import { getScenePreset } from "../../lib/core/scenes";
export const createScene = () => getScenePreset("forest-waterfall").create();
export const camera = getScenePreset("forest-waterfall").camera;
await registerSceneMeshAssets(createScene());
