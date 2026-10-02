/** Stored terrain shrub scatter through the production renderer. */
import { createScene as oakScene } from "./oak-v2";
import { heroPreviewCamera } from "./hero-still";
export const createScene = oakScene;
export const camera = { ...heroPreviewCamera(), distance_m: 3.4, elevation_rad: .48,
  target_m: { x: 0, y: .15, z: -.5 } };
