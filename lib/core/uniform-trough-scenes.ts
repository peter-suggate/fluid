import { cloneScene, defaultScene, type SceneDescription } from "./model";
import { sceneLatticeDimensions } from "./scene-lattice-dimensions";
import { boxSolidVoxelShell } from "./solid-world";
import { VOXEL_MATERIAL_IDS } from "./voxel-scene";

export type UniformTroughMode = "dam-break" | "settled-tank" | "hose-fill";

/** Matched 3D vessels for transport, hydrostatics, and source growth studies. */
export function createUniformTroughScene(
  mode: UniformTroughMode,
  cellSize_m = 0.05,
): SceneDescription {
  const scene = cloneScene(defaultScene);
  scene.sceneId = `uniform-trough-${mode}`;
  scene.duration_s = 20;
  scene.rigidBodies = [];
  scene.container = {
    ...scene.container,
    shape: "box",
    vessel: "none",
    width_m: 3.2,
    height_m: 1.2,
    depth_m: 1.2,
    top: "open",
    depthBoundary: "closed",
    fluidWallMode: "free-slip",
    // Keep 0.2 m of settled water above the raised solid floor.
    fillFraction: mode === "hose-fill" ? 0
      : (0.2 + (mode === "settled-tank" ? cellSize_m : 0)) / 1.2,
  };
  scene.voxelDomain = { finestCellSize_m: cellSize_m, brickSize_cells: 8 };
  // The authored voxel vessel replaces the generated glass tank shell.
  // Rebuild them at the requested lattice so resolution changes keep the vessel.
  scene.solidVoxels = boxSolidVoxelShell(sceneLatticeDimensions(scene), {
    top: "open", materialId: VOXEL_MATERIAL_IDS.box,
  }).map(patch => ({
    ...patch,
    // Lift the whole vessel one voxel: floor top is now above the stage.
    minimum: [patch.minimum[0], patch.minimum[1] + 1, patch.minimum[2]],
    maximumExclusive: [patch.maximumExclusive[0], patch.maximumExclusive[1] + 1,
      patch.maximumExclusive[2]],
  }));
  scene.fluid = {
    density_kg_m3: 998.2,
    dynamicViscosity_Pa_s: 0,
    surfaceTension_N_m: 0,
    gravity_m_s2: { x: 0, y: -9.80665, z: 0 },
    initialCondition: mode === "dam-break" ? "dam-break" : "tank-fill",
  };
  if (mode === "dam-break") {
    // Starts immediately after gate removal, spanning the trough's full depth.
    scene.fluid.initialDamBreakDimensions_m = { x: 0.8, y: 0.8, z: 1.2 };
    scene.fluid.initialDamBreakOrigin_m = { x: 0, y: cellSize_m, z: 0 };
  } else if (mode === "hose-fill") {
    scene.fluid.inflow = {
      center_m: { x: -1.3, y: 0.9 + cellSize_m, z: 0 },
      radius_m: 0.1,
      length_m: 0.15,
      velocity_m_s: { x: 1.5, y: 0, z: 0 },
      start_s: 0,
      end_s: 12,
      ramp_s: 0.5,
    };
  }
  scene.numerics.fixedDt_s = scene.numerics.maxDt_s = 1 / 60;
  return scene;
}
