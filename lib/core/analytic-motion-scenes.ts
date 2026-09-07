import { cloneScene, defaultScene, type SceneDescription } from "./model";

export type AnalyticMotion = "translation" | "free-fall";

/** Two-dimensional rectangular liquid body, extruded between free-slip depth walls. */
export function createAnalyticMotionScene(motion: AnalyticMotion): SceneDescription {
  const scene = cloneScene(defaultScene);
  scene.sceneId = `coarse-surface-${motion}`;
  scene.duration_s = motion === "translation" ? .5 : .3;
  scene.rigidBodies = [];
  scene.solidVoxels = [];
  delete scene.terrain;
  scene.container = { ...scene.container, width_m: 1.6, height_m: 1.6, depth_m: .4,
    fillFraction: .125, top: "closed", fluidWallMode: "free-slip", depthBoundary: "closed" };
  scene.voxelDomain = { finestCellSize_m: .05, brickSize_cells: 8 };
  scene.fluid.initialCondition = "dam-break";
  scene.fluid.initialDamBreakDimensions_m = { x: .8, y: .4, z: .4 };
  scene.fluid.initialDamBreakOrigin_m = { x: .4, y: .8, z: 0 };
  delete scene.fluid.initialBrickSeeds_m;
  delete scene.fluid.initialBrickSeedsAdditive;
  delete scene.fluid.initialLiquidVolumes;
  delete scene.fluid.initialHeightField;
  delete scene.fluid.inflow;
  scene.fluid.initialVelocity_m_s = { x: 0, y: motion === "translation" ? -.4 : 0, z: 0 };
  scene.fluid.gravity_m_s2 = { x: 0, y: motion === "free-fall" ? -9.81 : 0, z: 0 };
  scene.fluid.dynamicViscosity_Pa_s = 0;
  scene.fluid.surfaceTension_N_m = 0;
  // Fixed width-8 / width-4 halves isolate mixed-resolution physics from the
  // selector. Both regions meet on x=0 and cut the moving horizontal surface.
  scene.fluid.refinementRegions = [
    { id: "coarse-half", rule: "minimum-cell-size", minimumCellSize_cells: 8,
      maximumCellSize_cells: 8, min_m: { x: -.8, y: 0, z: -.2 }, max_m: { x: 0, y: 1.6, z: .2 } },
    { id: "fine-half", rule: "minimum-cell-size", minimumCellSize_cells: 4,
      maximumCellSize_cells: 4, min_m: { x: 0, y: 0, z: -.2 }, max_m: { x: .8, y: 1.6, z: .2 } },
  ];
  scene.numerics.fixedDt_s = scene.numerics.maxDt_s = 1 / 60;
  return scene;
}

/** Same falling body, with a coarse/fine swap and return while it is moving. */
export function createRerungFreeFallScene(): SceneDescription {
  const scene = createAnalyticMotionScene("free-fall");
  scene.sceneId = "coarse-surface-free-fall-rerung";
  const original = structuredClone(scene.fluid.refinementRegions!);
  const swapped = original.map(region => ({ ...region,
    minimumCellSize_cells: region.minimumCellSize_cells === 8 ? 4 : 8,
    maximumCellSize_cells: region.maximumCellSize_cells === 8 ? 4 : 8 }));
  scene.fluid.refinementKeyframes = [
    { time_s: 0, regions: original },
    { time_s: .1, regions: swapped },
    { time_s: .2, regions: original },
  ];
  return scene;
}

export const STANDING_WAVE = Object.freeze({ length_m: 1.6, depth_m: .6, amplitude_m: .03, gravity_m_s2: 9.81 });
export function standingWaveOmega() {
  const {length_m:L,depth_m:H,gravity_m_s2:g}=STANDING_WAVE;
  const k=Math.PI/L;
  return Math.sqrt(g*k*Math.tanh(k*H));
}

/** Fundamental inviscid seiche: initially at maximum displacement and at rest. */
export function createStandingWaveScene(live: boolean): SceneDescription {
  const scene = createAnalyticMotionScene("free-fall");
  const {length_m:L,depth_m:H,amplitude_m:A}=STANDING_WAVE;
  scene.sceneId = live ? "coarse-surface-standing-wave-live" : "coarse-surface-standing-wave";
  scene.container.height_m = 1.2;
  scene.container.fillFraction = H / scene.container.height_m;
  scene.duration_s = 4 * Math.PI / standingWaveOmega();
  scene.fluid.initialCondition = "tank-fill";
  delete scene.fluid.initialDamBreakDimensions_m;
  delete scene.fluid.initialDamBreakOrigin_m;
  scene.fluid.initialHeightField = { kind: "cosine", baseHeight_m: H, amplitude_m: A,
    wavelength_m: 2*L, originX_m: -L/2 };
  scene.fluid.refinementRegions = live ? [] : [
    { id: "wave-coarse", rule: "minimum-cell-size", minimumCellSize_cells: 4,
      maximumCellSize_cells: 4, min_m: { x: -.8, y: 0, z: -.2 }, max_m: { x: -.4, y: 1.2, z: .2 } },
    { id: "wave-fine", rule: "minimum-cell-size", minimumCellSize_cells: 2,
      maximumCellSize_cells: 2, min_m: { x: -.4, y: 0, z: -.2 }, max_m: { x: .8, y: 1.2, z: .2 } },
  ];
  return scene;
}
