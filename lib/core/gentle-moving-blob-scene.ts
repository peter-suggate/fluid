import { cloneScene, defaultScene, type SceneDescription } from "./model";

/**
 * Small curved liquid body for exercising the real sparse geometric advance.
 *
 * There is deliberately no refinement region: the curved interface goes
 * through the ordinary coarse-first sparse policy and its natural 2:1 seams.
 * The initial velocity starts the motion, then the projected field and
 * transport evolve without forcing, sources, moving solids, or intervention.
 */
export function createGentleMovingBlobScene(): SceneDescription {
  const scene = cloneScene(defaultScene);
  scene.sceneId = "gentle-moving-blob";
  scene.duration_s = 6;
  scene.rigidBodies = [];
  scene.solidVoxels = [];
  delete scene.terrain;
  scene.container = {
    ...scene.container,
    width_m: 1.6,
    height_m: 1.2,
    depth_m: 0.4,
    fillFraction: 0,
    top: "closed",
    fluidWallMode: "free-slip",
  };
  scene.voxelDomain = { finestCellSize_m: 0.05, brickSize_cells: 8 };
  scene.fluid.initialCondition = "tank-fill";
  delete scene.fluid.initialDamBreakDimensions_m;
  delete scene.fluid.initialDamBreakOrigin_m;
  delete scene.fluid.initialBrickSeeds_m;
  delete scene.fluid.initialBrickSeedsAdditive;
  delete scene.fluid.initialHeightField;
  delete scene.fluid.inflow;
  delete scene.fluid.refinementRegions;
  delete scene.fluid.refinementKeyframes;
  scene.fluid.initialLiquidVolumes = [{
    shape: "sphere",
    center_m: { x: -0.25, y: 0.6, z: 0 },
    radius_m: 0.3,
  }];
  scene.fluid.initialVelocity_m_s = { x: 0.08, y: 0, z: 0 };
  scene.fluid.gravity_m_s2 = { x: 0, y: 0, z: 0 };
  scene.fluid.dynamicViscosity_Pa_s = 0;
  scene.fluid.surfaceTension_N_m = 0;
  scene.numerics.fixedDt_s = scene.numerics.maxDt_s = 1 / 30;
  return scene;
}
