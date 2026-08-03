/**
 * The stone set in the hero pond: boulders, pebble beds and stepping stones.
 *
 *     FLUID_SVO_DRY_FRAME_SCENE_MODULE=tools/preview/stone-set.ts
 *
 * The hero scene already publishes the set, so this module adds *nothing* — it is
 * the hero document with the camera swung round and pulled in so the near-left
 * bank fills the frame. Appending `stoneSet(...)` here, which is what it used to
 * do, published every stone twice at the same coordinates: invisible in the frame
 * and quietly doubling the primitive count the frame was being judged on.
 *
 * `tools/preview/hero-still.ts` is where the judgement that matters is made, at
 * the hero camera. This crop is for reading the beds and the wading path close
 * up, where the hero framing spends most of its width on water these stones are
 * not in.
 */
import { heroPreviewCamera, heroPreviewScene } from "./hero-still";

export const createScene = () => heroPreviewScene();

export const camera = {
  ...heroPreviewCamera(),
  azimuth_rad: 1.00,
  elevation_rad: 0.50,
  distance_m: 0.80,
  target_m: { x: -0.18, y: 0.30, z: 0.08 },
};
