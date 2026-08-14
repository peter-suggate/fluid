/**
 * The `garden-svo-lighting` study seen from a fully zoomed-out orbit, for
 * `benchmark-svo-dry-frame-gpu.ts` and the xctrace render profiler.
 *
 * The scene itself is untouched — this lane only moves the camera, because the
 * question it answers is why the app stays slow at the far end of the zoom
 * range where almost nothing subtends more than a pixel.
 */
import type { CameraState, SceneDescription } from "../lib/core/model";
import { getScenePreset } from "../lib/core/scenes";

const preset = getScenePreset("garden-svo-lighting");

export function createScene(): SceneDescription {
  return preset.create();
}

/**
 * `zoomCamera` in lib/math.ts clamps orbit distance to [0.65, 12] m, so 12 is
 * exactly what the app reaches when the user scrolls out until it stops.
 */
export const camera: Partial<CameraState> = {
  ...preset.camera,
  distance_m: Number(process.env.FLUID_GARDEN_FAR_DISTANCE_M ?? 12),
  target_m: { ...(preset.camera?.target_m ?? { x: 0, y: 0.26, z: 0 }) },
};
