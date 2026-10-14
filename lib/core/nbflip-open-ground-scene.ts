import type { CameraState, SceneDescription } from "./model";
import { cellPoint, nbflipDomain } from "./nbflip-paper-scenes";

/**
 * A column of water released on open ground: the open world's probe.
 *
 * The container has no walls (`container.walls: "none"`). It is only where
 * the water starts, a column two pages wide, two deep and one high, on the
 * plane y = 0 that is the whole world's ground. When it is released it
 * slumps and spreads on every side, and the narrow-band FLIP solver claims
 * pages wherever it goes: past the container in x and z, and above it where
 * it splashes. Nothing holds it but the ground.
 *
 * The pool is sized from a memory budget. A flood that needs more pages than
 * the budget buys stops at the pool's exhausted fatal, and one that reaches
 * the edge of the solver's window at its edge fatal: neither reallocates.
 *
 * Coarse cells keep the flood inside the budget: a column `COLUMN` cells
 * tall spreads about 2 sqrt(g COLUMN / h) t cells in t seconds, so a
 * quarter-metre cell holds two seconds of it to a few pages on each side.
 */
export const NBFLIP_OPEN_GROUND_SCENE_ID = "nbflip-open-ground";
/** The container: the column's footprint and one page of height, in cells. */
export const NBFLIP_OPEN_GROUND_GRID = [64, 32, 64] as const;
export const NBFLIP_OPEN_GROUND_CELL_SIZE_M = 0.25;
/** The water column's height in cells. */
export const NBFLIP_OPEN_GROUND_COLUMN = 16;
export const NBFLIP_OPEN_GROUND_STEP_HZ = 60;
export const NBFLIP_OPEN_GROUND_SECONDS = 6;
/** The page pool's budget. */
export const NBFLIP_OPEN_GROUND_MEMORY_MIB = 1280;

export function createNbflipOpenGround(stepHz = NBFLIP_OPEN_GROUND_STEP_HZ): SceneDescription {
  const grid = NBFLIP_OPEN_GROUND_GRID, h = NBFLIP_OPEN_GROUND_CELL_SIZE_M;
  const scene = nbflipDomain(NBFLIP_OPEN_GROUND_SCENE_ID, grid, h, NBFLIP_OPEN_GROUND_SECONDS * stepHz, 0, grid, 1 / stepHz);
  scene.solidVoxels = [];
  scene.container = { ...scene.container, top: "open", walls: "none", vessel: "none", fillFraction: 0 };
  scene.fluid.initialVelocity_m_s = { x: 0, y: 0, z: 0 };
  scene.fluid.initialLiquidVolumes = [{
    shape: "box",
    min_m: cellPoint(grid, h, [0, 0, 0]),
    max_m: cellPoint(grid, h, [grid[0], NBFLIP_OPEN_GROUND_COLUMN, grid[2]]),
  }];
  scene.fluid.pagePool = { memory_MiB: NBFLIP_OPEN_GROUND_MEMORY_MIB };
  return scene;
}

/** From above and to the side, far enough back to see the flood reach a few pages out. */
export function nbflipOpenGroundCamera(): Partial<CameraState> {
  return { azimuth_rad: 0.6, elevation_rad: 0.7, distance_m: 70, target_m: { x: 0, y: 0, z: 0 } };
}
