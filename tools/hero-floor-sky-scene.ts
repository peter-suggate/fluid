/** Empty-sky floor lane for benchmark-svo-dry-frame-gpu.ts. */
import type { CameraState, SceneDescription } from "../lib/core/model";
import { heroGardenCamera } from "../lib/core/hero-garden-scene";
import { createHeroGardenHoseStressScene } from "../lib/core/hero-garden-stress-scene";

export function createScene(): SceneDescription {
  return createHeroGardenHoseStressScene({ recordMultiplier: 1 });
}

export const camera: Partial<CameraState> = {
  ...heroGardenCamera,
  elevation_rad: -0.5,
  distance_m: 1.4,
  target_m: { x: 0, y: 50, z: 0 },
};
