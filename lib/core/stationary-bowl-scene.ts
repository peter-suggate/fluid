import { cloneScene, defaultScene, type SceneDescription } from "./model";

/** The UI counterpart of the density-first Dawn grid-imprint fixture. */
export function createStationaryBowlScene(curvatureMultiplier: 1 | 2 = 1): SceneDescription {
  const scene = cloneScene(defaultScene);
  scene.sceneId = curvatureMultiplier === 2 ? "stationary-bowl-2x" : "stationary-bowl";
  scene.rigidBodies = [];
  scene.solidVoxels = [];
  delete scene.terrain;
  scene.container = { ...scene.container, width_m: 2.4, height_m: 1.6, depth_m: 2,
    fillFraction: 17.3 / 32, top: "closed", fluidWallMode: "free-slip", depthBoundary: "closed" };
  scene.voxelDomain = { finestCellSize_m: .05, brickSize_cells: 8 };
  scene.fluid.initialCondition = "tank-fill";
  delete scene.fluid.initialBrickSeeds_m;
  delete scene.fluid.initialBrickSeedsAdditive;
  delete scene.fluid.initialLiquidVolumes;
  delete scene.fluid.initialDamBreakDimensions_m;
  delete scene.fluid.initialDamBreakOrigin_m;
  delete scene.fluid.inflow;
  // H in finest cells = 17.3 + .003 ((x-24)^2 + .7 (z-20)^2).
  scene.fluid.initialHeightField = { kind: "quadratic", baseHeight_m: .865,
    center_m: { x: 0, z: 0 }, curvatureX_mInv: .06 * curvatureMultiplier,
    curvatureZ_mInv: .042 * curvatureMultiplier };
  scene.fluid.gravity_m_s2 = { x: 0, y: 0, z: 0 };
  scene.fluid.dynamicViscosity_Pa_s = 0;
  scene.fluid.surfaceTension_N_m = 0;
  scene.fluid.refinementRegions = [{ id: "bowl-resolution", rule: "minimum-cell-size",
    minimumCellSize_cells: 4, maximumCellSize_cells: 4,
    min_m: { x: -1.2, y: 0, z: -1 }, max_m: { x: 1.2, y: 1.6, z: 1 } }];
  if (curvatureMultiplier === 2) scene.fluid.refinementRegions = [];
  scene.numerics.fixedDt_s = scene.numerics.maxDt_s = 1 / 60;
  return scene;
}
