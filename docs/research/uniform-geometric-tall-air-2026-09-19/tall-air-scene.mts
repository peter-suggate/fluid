/**
 * The tall-air A/B fixture: one scene, one dial.
 *
 * Arm A is `multiple = 1`; every other arm is the identical problem in a taller
 * container. The cell size, the absolute reservoir box, the floor footprint,
 * the boundary kinds and dt are all fixed, so the only thing that changes
 * between arms is how much EMPTY AIR sits above the liquid.
 *
 * The reservoir is authored in absolute metres through
 * `fluid.initialDamBreakDimensions_m` (lib/core/initial-fluid.ts:29). Without
 * that the fill-fraction path (`damBreakFractions`, :17) would make the taller
 * arm's dam proportionally taller, and the arms would not be the same problem.
 *
 * The tank shell is re-authored after the height moves, because
 * `solidVoxelShellForScene` bakes voxel indices against the final lattice.
 */
import { createMinimalPowerDamBreak64Scene } from "../../../lib/core/scenes";
import { solidVoxelEditsForScene, solidVoxelShellForScene } from "../../../lib/core/scene-lattice";
import type { SceneDescription } from "../../../lib/core/model";

/**
 * 0.4 x 0.2 x 0.8 m: half the tank's width, a quarter of arm A's height, the
 * full depth. A quarter-height column keeps the collapse and its far-wall
 * run-up well clear of arm A's 0.8 m closed lid, so both arms solve the same
 * physical problem and neither one's liquid ever reaches the top.
 */
export const TALL_AIR_RESERVOIR_M = { x: 0.4, y: 0.2, z: 0.8 } as const;

export function tallAirScene(multiple: number): SceneDescription {
  const scene = createMinimalPowerDamBreak64Scene();
  scene.solidVoxels = solidVoxelEditsForScene(scene);
  scene.sceneId = `tall-air-dam-${multiple}x`;
  scene.duration_s = 60;
  scene.container = { ...scene.container, height_m: 0.8 * multiple, top: "closed" };
  scene.fluid.initialDamBreakDimensions_m = { ...TALL_AIR_RESERVOIR_M };
  scene.solidVoxels = [...solidVoxelShellForScene(scene), ...scene.solidVoxels];
  return scene;
}
