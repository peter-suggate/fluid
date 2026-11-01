/** Production scene and moving Blender skins, without simulated water in this dry capture. */
import { getScenePreset } from '../../lib/core/scenes';
export const createScene = () => getScenePreset('paddleboats').create();
export const camera = getScenePreset('paddleboats').camera;
