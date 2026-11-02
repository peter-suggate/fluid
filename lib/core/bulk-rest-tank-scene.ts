import { cloneScene, defaultScene, type CameraState, type SceneDescription } from "./model";
import { LATTICE_PAGE_CELLS } from "./scene-lattice-dimensions";

export const BULK_REST_TANK_ID = "bulk-rest-tank";
export const BULK_REST_TANK_CELL_M = 0.01;
export const BULK_REST_TANK_CAMERA: Partial<CameraState> = {
  distance_m: 7, target_m: { x: 0, y: 1.2, z: 0 }, azimuth_rad: 0.5, elevation_rad: 0.3,
};

/** Cells of liquid above the top BULK16 page's top face. A whole page joins
 * the 16h tier only with its 16-cell grading region and the 5.5 h depth test
 * above it (docs/plans/bulk16-plan-2026-10-28.md, "BULK16 coverage of the
 * catalog"), so 20 cells is the least that admits it; 24 keeps a margin. */
export const BULK_REST_TANK_SURFACE_CELLS = 24;

/** The pure BULK16 benchmark: a page-aligned box of still water with no solid
 * voxels, no refinement regions and no bodies. Only the box's own walls bound
 * it. The free surface sits `BULK_REST_TANK_SURFACE_CELLS` into the top liquid
 * page layer, so every page below that layer can be a whole-16h page; one air
 * page layer sits above it. `pagesX/Y/Z` are the box's liquid pages a side
 * (`pagesY` liquid layers, the top one holding the surface). */
export function createBulkRestTankScene(cellSize_m = BULK_REST_TANK_CELL_M,
  { pagesX = 8, pagesY = 8, pagesZ = 4 }: { readonly pagesX?: number; readonly pagesY?: number; readonly pagesZ?: number } = {}): SceneDescription {
  for (const [name, n] of [["pagesX", pagesX], ["pagesY", pagesY], ["pagesZ", pagesZ]] as const)
    if (!Number.isSafeInteger(n) || n < 1) throw new Error(`createBulkRestTankScene: ${name} must be a positive integer (got ${n})`);
  const P = LATTICE_PAGE_CELLS, page_m = P * cellSize_m;
  const scene = cloneScene(defaultScene);
  scene.sceneId = BULK_REST_TANK_ID;
  scene.duration_s = 20;
  scene.container = {
    width_m: pagesX * page_m, height_m: (pagesY + 1) * page_m, depth_m: pagesZ * page_m,
    fillFraction: 0, top: "open", fluidWallMode: "free-slip",
  };
  scene.voxelDomain = { finestCellSize_m: cellSize_m, brickSize_cells: 8 };
  scene.nominalResolution = { length_m: cellSize_m };
  scene.rigidBodies = [];
  scene.solidVoxels = [];
  delete scene.terrain;
  delete scene.solidBoundary;
  const surface_m = ((pagesY - 1) * P + BULK_REST_TANK_SURFACE_CELLS) * cellSize_m;
  const hw = scene.container.width_m / 2, hd = scene.container.depth_m / 2;
  scene.fluid = {
    density_kg_m3: 998.2, dynamicViscosity_Pa_s: 0,
    surfaceTension_N_m: 0, gravity_m_s2: { x: 0, y: -9.80665, z: 0 },
    initialCondition: "tank-fill", initialVelocity_m_s: { x: 0, y: 0, z: 0 },
    initialLiquidVolumes: [{ shape: "box", min_m: { x: -hw, y: 0, z: -hd }, max_m: { x: hw, y: surface_m, z: hd } }],
    // No pagePool: the pool by rule (uniformPagePoolRule).
  };
  scene.numerics.fixedDt_s = scene.numerics.maxDt_s = 1 / 60;
  scene.surfaceStyle = "smooth";
  return scene;
}
