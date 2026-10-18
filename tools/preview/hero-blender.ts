/** Production renderer, catalog lighting and the new garden camera. */
import { registerSceneMeshAssets } from "../blender/register-assets";
import { getScenePreset } from "../../lib/core/scenes";
export const createScene = () => getScenePreset("hero-garden-hose-x10-blender").create();
export const camera = getScenePreset("hero-garden-hose-x10-blender").camera;
await registerSceneMeshAssets(createScene());
