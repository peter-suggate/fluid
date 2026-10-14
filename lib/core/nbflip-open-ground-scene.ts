import type { CameraState, SceneDescription } from "./model";
import { cellPoint, nbflipDomain } from "./nbflip-paper-scenes";

/**
 * A column of water released on open ground: the open world's probe.
 *
 * The container has no walls (`container.walls: "none"`): it only places
 * the world. The column stands on the plane y = 0 that is the whole world's
 * ground, `MARGIN` cells in from the container's sides. When it is released
 * it slumps and spreads on every side, and the narrow-band FLIP solver claims
 * pages wherever it goes. Nothing holds it but the ground. The solver builds
 * its start of run over the column and a margin (`sceneFluidBox`), so where
 * the container's faces are does not matter: the `faces` variant lays the
 * same column against them (margin 0, the container its footprint), and the
 * `slope` variant raises the ground into a gentle analytic incline that runs
 * on past the container, rising to -x.
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
export const NBFLIP_OPEN_GROUND_FACES_SCENE_ID = "nbflip-open-ground-faces";
export const NBFLIP_OPEN_GROUND_SLOPE_SCENE_ID = "nbflip-open-ground-slope";
/** The container: one page of height, in cells. */
export const NBFLIP_OPEN_GROUND_GRID = [64, 32, 64] as const;
export const NBFLIP_OPEN_GROUND_CELL_SIZE_M = 0.25;
/** The water column's height in cells. */
export const NBFLIP_OPEN_GROUND_COLUMN = 16;
/** Cells between the column and the container's sides. */
export const NBFLIP_OPEN_GROUND_MARGIN = 8;
/** The column's side in cells: the same in every variant. */
export const NBFLIP_OPEN_GROUND_SIDE = NBFLIP_OPEN_GROUND_GRID[0] - 2 * NBFLIP_OPEN_GROUND_MARGIN;
export const NBFLIP_OPEN_GROUND_STEP_HZ = 60;
export const NBFLIP_OPEN_GROUND_SECONDS = 6;
/** The page pool's budget. */
export const NBFLIP_OPEN_GROUND_MEMORY_MIB = 1792;
/** The slope variant's incline: one wide mound, centred off to -x, whose flank the column stands on. */
export const NBFLIP_OPEN_GROUND_SLOPE = { center_m: { x: -48, z: 0 }, radius_m: { x: 60, z: 60 }, amount_m: 6, flat: 0 } as const;

export type NbflipOpenGroundVariant = "column" | "faces" | "slope";

export function createNbflipOpenGround(stepHz = NBFLIP_OPEN_GROUND_STEP_HZ, variant: NbflipOpenGroundVariant = "column"): SceneDescription {
  const h = NBFLIP_OPEN_GROUND_CELL_SIZE_M, m = variant === "faces" ? 0 : NBFLIP_OPEN_GROUND_MARGIN;
  const grid = [NBFLIP_OPEN_GROUND_SIDE + 2 * m, NBFLIP_OPEN_GROUND_GRID[1], NBFLIP_OPEN_GROUND_SIDE + 2 * m] as const;
  const id = variant === "faces" ? NBFLIP_OPEN_GROUND_FACES_SCENE_ID : variant === "slope" ? NBFLIP_OPEN_GROUND_SLOPE_SCENE_ID : NBFLIP_OPEN_GROUND_SCENE_ID;
  const scene = nbflipDomain(id, grid, h, NBFLIP_OPEN_GROUND_SECONDS * stepHz, 0, grid, 1 / stepHz);
  scene.solidVoxels = [];
  scene.container = { ...scene.container, top: "open", walls: "none", vessel: "none", fillFraction: 0 };
  scene.fluid.initialVelocity_m_s = { x: 0, y: 0, z: 0 };
  scene.fluid.initialLiquidVolumes = [{
    shape: "box",
    min_m: cellPoint(grid, h, [m, 0, m]),
    max_m: cellPoint(grid, h, [grid[0] - m, NBFLIP_OPEN_GROUND_COLUMN, grid[2] - m]),
  }];
  if (variant === "slope") {
    const s = NBFLIP_OPEN_GROUND_SLOPE;
    scene.terrain = { baseHeight_m: 0, features: [{ kind: "mound", center_m: { ...s.center_m }, radius_m: { ...s.radius_m }, amount_m: s.amount_m, flat: s.flat }] };
  }
  scene.fluid.pagePool = { memory_MiB: NBFLIP_OPEN_GROUND_MEMORY_MIB };
  return scene;
}

/** From above and to the side, far enough back to see the flood reach a few pages out. */
export function nbflipOpenGroundCamera(): Partial<CameraState> {
  return { azimuth_rad: 0.6, elevation_rad: 0.7, distance_m: 70, target_m: { x: 0, y: 0, z: 0 } };
}
