import type { SceneDescription } from "./model";
import { cellPoint, nbflipDomain } from "./nbflip-paper-scenes";
import {
  NBFLIP_RAMP_CELL_SIZE_M, NBFLIP_RAMP_GRID, NBFLIP_RAMP_LENGTH,
  NBFLIP_RAMP_STEP_HZ,
} from "./nbflip-ramp-scene";

export const NBFLIP_SLOPE_CLEARANCE_SCENE_ID = "nbflip-slope-clearance";
/** A 16 × 12 × 16 cm body released just above the incline, clear of the walls. */
export const NBFLIP_SLOPE_CLEARANCE_MIN = [4, 17, 12] as const;
export const NBFLIP_SLOPE_CLEARANCE_MAX = [12, 23, 20] as const;
export const NBFLIP_SLOPE_CLEARANCE_INITIAL_CELLS = 8 * 6 * 8;
export const NBFLIP_SLOPE_CLEARANCE_SECONDS = 6;
export const NBFLIP_SLOPE_CLEARANCE_RUN = 16;
/** A short, continuous spill apron joins the incline to the lower catch basin. */
export const NBFLIP_SLOPE_CLEARANCE_PIT_START = NBFLIP_RAMP_LENGTH + 4;

/** Height in cells, at a continuous x coordinate measured from the left edge. */
export function nbflipSlopeClearanceBed(x: number): number {
  if (x <= NBFLIP_RAMP_LENGTH) return 17 - x / NBFLIP_SLOPE_CLEARANCE_RUN;
  const foot = 17 - NBFLIP_RAMP_LENGTH / NBFLIP_SLOPE_CLEARANCE_RUN;
  return Math.max(2, foot - (x - NBFLIP_RAMP_LENGTH) * (foot - 2) / 4);
}

/**
 * Reference transit for a frictionless parcel released at the body's uphill
 * edge on the corresponding smooth incline: s = L / cos(theta), a = g sin(theta).
 * About 2.45 s. This is a scale estimate, not an exact free-surface solution;
 * six seconds allows spreading and a drainage tail. No residue is the target
 * for this free-slip, zero-surface-tension model, not a real wetting surface.
 */
export function nbflipSlopeClearanceTransitSeconds(): number {
  const slope = 1 / NBFLIP_SLOPE_CLEARANCE_RUN;
  const length = (NBFLIP_RAMP_LENGTH - NBFLIP_SLOPE_CLEARANCE_MIN[0]) * NBFLIP_RAMP_CELL_SIZE_M;
  return Math.sqrt(2 * length * (1 + slope * slope) / (9.81 * slope));
}

/**
 * A continuous 1:16 heightfield with a finite release. Bilinear interpolation
 * reproduces the inclined plane exactly, without rounding heights to treads.
 * NB-FLIP still rasterizes solid occupancy and bakes its bed-gradient lane;
 * this scene does not change that solver's collision discretization.
 * Clearance means no water at x < 96 cells; water remains in the lower pit,
 * so losing liquid cannot masquerade as successfully draining the slope.
 */
export function createNbflipSlopeClearance(stepHz = NBFLIP_RAMP_STEP_HZ): SceneDescription {
  const grid = NBFLIP_RAMP_GRID, h = NBFLIP_RAMP_CELL_SIZE_M, [nx, , nz] = grid;
  const scene = nbflipDomain(NBFLIP_SLOPE_CLEARANCE_SCENE_ID, grid, h,
    NBFLIP_SLOPE_CLEARANCE_SECONDS * stepHz, 0, grid, 1 / stepHz);
  scene.terrain = {
    baseHeight_m: 0, features: [],
    grid: {
      kind: "grid", origin_m: { x: -nx * h / 2, z: -nz * h / 2 },
      spacing_m: h, size: { nx: nx + 1, nz: nz + 1 },
      heights_m: Array.from({ length: (nx + 1) * (nz + 1) }, (_, i) => nbflipSlopeClearanceBed(i % (nx + 1)) * h),
    },
  };
  scene.fluid.pagePool = { blocks: nx / 32 + 2 };
  scene.fluid.initialVelocity_m_s = { x: 0, y: 0, z: 0 };
  scene.fluid.initialLiquidVolumes = [{
    shape: "box",
    min_m: cellPoint(NBFLIP_RAMP_GRID, NBFLIP_RAMP_CELL_SIZE_M, NBFLIP_SLOPE_CLEARANCE_MIN),
    max_m: cellPoint(NBFLIP_RAMP_GRID, NBFLIP_RAMP_CELL_SIZE_M, NBFLIP_SLOPE_CLEARANCE_MAX),
  }];
  return scene;
}
