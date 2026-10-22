import { cloneScene, defaultScene, type SceneDescription } from "./model";
import { prescribedOrbitPosition } from "./prescribed-rigid-motion";

export const CHURNING_TANK_SCENE_ID = "churning-tank";

/** One complete motor-driven stirring session, followed by free settling. */
export function createChurningTankScene(): SceneDescription {
  const scene = cloneScene(defaultScene);
  scene.sceneId = CHURNING_TANK_SCENE_ID;
  scene.duration_s = 20;
  scene.container = { width_m: 2.4, height_m: 1.6, depth_m: 2.4,
    fillFraction: 0.5, top: "open", fluidWallMode: "no-slip" };
  scene.voxelDomain = { finestCellSize_m: 0.025, brickSize_cells: 8 };
  scene.nominalResolution = { length_m: 0.025 };
  scene.solidVoxels = [];
  delete scene.terrain;
  scene.fluid = { density_kg_m3: 998.2, dynamicViscosity_Pa_s: 0.001002,
    surfaceTension_N_m: 0.072, gravity_m_s2: { x: 0, y: -9.80665, z: 0 },
    initialCondition: "tank-fill", initialVelocity_m_s: { x: 0, y: 0, z: 0 },
    // The sustained paddle splash needs more than the default single air
    // layer above the tank, including the advection sampling halo. 64 live
    // pages plus two sentinels reserve room for the complete session.
    pagePool: { blocks: 66 } };
  // Keep the full motor speed and fine grid at the normal fluid clock.
  // Regression coverage also exercises 1/30 without internal substeps.
  scene.numerics.fixedDt_s = scene.numerics.maxDt_s = 1 / 60;
  const prescribedMotion = {
    kind: "orbit" as const, center_m: { x: 0, y: 1.08, z: 0 }, radius_m: 0.72,
    phase_rad: 0, angularSpeed_rad_s: Math.PI,
    startTime_s: 1, rampUp_s: 2, steady_s: 10, rampDown_s: 3,
  };
  scene.rigidBodies = [{
    id: "churning-rod", name: "Spatula handle", shape: "cylinder", motion: "static",
    dimensions_m: { x: 0.055, y: 0.64, z: 0.055 }, density_kg_m3: 7800,
    position_m: prescribedOrbitPosition(prescribedMotion, 0), orientation: { w: 1, x: 0, y: 0, z: 0 },
    linearVelocity_m_s: { x: 0, y: 0, z: 0 }, angularVelocity_rad_s: { x: 0, y: 0, z: 0 },
    restitution: 0, friction: 0.5, prescribedMotion,
  }];
  // The blade overlaps the handle and stays broadside to its circular travel.
  // Both pieces share one motor schedule and displace the actual fluid.
  const bladeMotion = { ...prescribedMotion, center_m: { x: 0, y: 0.45, z: 0 }, rotateWithOrbit: true };
  scene.rigidBodies.push({
    ...scene.rigidBodies[0]!, id: "churning-blade", name: "Spatula blade", shape: "box",
    dimensions_m: { x: 0.46, y: 0.74, z: 0.10 },
    position_m: prescribedOrbitPosition(bladeMotion, 0), prescribedMotion: bladeMotion,
  });
  scene.surfaceStyle = "smooth";
  return scene;
}
