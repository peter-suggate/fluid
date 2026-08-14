/** Distant-content floor lane for benchmark-svo-dry-frame-gpu.ts. */
import type { CameraState, SceneDescription } from "../lib/core/model";
import { heroGardenCamera } from "../lib/core/hero-garden-scene";
import { createHeroGardenHoseStressScene } from "../lib/core/hero-garden-stress-scene";

export function createScene(): SceneDescription {
  return createHeroGardenHoseStressScene({
    recordMultiplier: Number(process.env.FLUID_HERO_FAR_MULTIPLIER ?? 1),
  });
}

export const camera: Partial<CameraState> = {
  ...heroGardenCamera,
  distance_m: Number(process.env.FLUID_HERO_FAR_DISTANCE_M ?? 30),
  target_m: { ...(heroGardenCamera.target_m ?? { x: 0, y: 0.35, z: 0 }) },
};
