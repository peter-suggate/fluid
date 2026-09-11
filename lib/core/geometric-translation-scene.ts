import { cloneScene, defaultScene, type SceneDescription } from "./model";
import { solidVoxelShellForScene } from "./scene-lattice";

export const GEOMETRIC_TRANSLATION_CELL_SIZE_M = 0.05;
export const GEOMETRIC_TRANSLATION_SPEED_M_S = 6;
export const GEOMETRIC_TRANSLATION_DT_S = 1 / 60;
export const GEOMETRIC_TRANSLATION_DURATION_S = 0.05;
export const GEOMETRIC_TRANSLATION_BLOCK = Object.freeze({
  origin_m: Object.freeze({ x: 0.1, y: 0, z: 0 }),
  dimensions_m: Object.freeze({ x: 0.2, y: 0.4, z: 0.4 }),
});
export const GEOMETRIC_TRANSLATION_DETACHED_BLOCK = Object.freeze({
  origin_m: Object.freeze({ x: 0.1, y: 0.1, z: 0 }),
  dimensions_m: Object.freeze({ x: 0.2, y: 0.2, z: 0.4 }),
});

export interface GeometricUniformTranslationSceneOptions {
  readonly speed_m_s?: number;
  readonly dt_s?: number;
  readonly duration_s?: number;
  readonly detached?: boolean;
}

/**
 * Small all-fine manufactured solution for geometric volume transport.
 *
 * Before the translated block meets a wall, zero gravity and a spatially
 * uniform velocity make the exact Euler solution a rigid translation. The
 * whole-domain width-one region removes resolution changes from that claim;
 * deterministic probes additionally freeze its accepted topology after
 * construction.
 */
export function createGeometricUniformTranslationScene(
  options: GeometricUniformTranslationSceneOptions = {},
): SceneDescription {
  const speed_m_s = options.speed_m_s ?? GEOMETRIC_TRANSLATION_SPEED_M_S;
  const dt_s = options.dt_s ?? GEOMETRIC_TRANSLATION_DT_S;
  const duration_s = options.duration_s ?? GEOMETRIC_TRANSLATION_DURATION_S;
  const block = options.detached
    ? GEOMETRIC_TRANSLATION_DETACHED_BLOCK : GEOMETRIC_TRANSLATION_BLOCK;
  if (!(Number.isFinite(speed_m_s) && speed_m_s > 0)) {
    throw new RangeError("translation speed must be finite and positive");
  }
  if (!(Number.isFinite(dt_s) && dt_s > 0)) {
    throw new RangeError("translation timestep must be finite and positive");
  }
  if (!(Number.isFinite(duration_s) && duration_s > 0)) {
    throw new RangeError("translation duration must be finite and positive");
  }
  const finalMaximumX_m = block.origin_m.x
    + block.dimensions_m.x + speed_m_s * duration_s;
  if (finalMaximumX_m >= 0.8) {
    throw new RangeError("translation duration reaches the downstream wall");
  }
  const scene = cloneScene(defaultScene);
  scene.sceneId = "geometric-uniform-translation";
  scene.duration_s = duration_s;
  scene.rigidBodies = [];
  scene.container = { ...scene.container, width_m: 0.8, height_m: 0.4,
    depth_m: 0.4, fillFraction: block.dimensions_m.x * block.dimensions_m.y
      * block.dimensions_m.z / (0.8 * 0.4 * 0.4), top: "closed",
    fluidWallMode: "free-slip" };
  scene.voxelDomain = { finestCellSize_m: GEOMETRIC_TRANSLATION_CELL_SIZE_M,
    brickSize_cells: 8 };
  scene.fluid.initialCondition = "dam-break";
  scene.fluid.initialDamBreakOrigin_m = {
    ...block.origin_m,
  };
  scene.fluid.initialDamBreakDimensions_m = {
    ...block.dimensions_m,
  };
  scene.fluid.initialVelocity_m_s = { x: speed_m_s, y: 0, z: 0 };
  scene.fluid.gravity_m_s2 = { x: 0, y: 0, z: 0 };
  scene.fluid.dynamicViscosity_Pa_s = 0;
  scene.fluid.surfaceTension_N_m = 0;
  delete scene.fluid.initialBrickSeeds_m;
  delete scene.fluid.initialBrickSeedsAdditive;
  delete scene.fluid.initialLiquidVolumes;
  delete scene.fluid.inflow;
  scene.fluid.refinementRegions = [{
    id: "geometric-uniform-translation-all-fine",
    rule: "minimum-cell-size",
    minimumCellSize_cells: 1,
    maximumCellSize_cells: 1,
    // Include one finest-cell margin around every vessel face so clamping or
    // boundary ownership cannot leave a coarse edge brick.
    min_m: { x: -0.45, y: -0.05, z: -0.25 },
    max_m: { x: 0.45, y: 0.45, z: 0.25 },
  }];
  scene.numerics.fixedDt_s = scene.numerics.maxDt_s = dt_s;
  scene.solidVoxels = [...solidVoxelShellForScene(scene)];
  return scene;
}
