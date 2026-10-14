import type { CameraState, SceneDescription } from "./model";
import { cellPoint, nbflipDomain } from "./nbflip-paper-scenes";
import { VOXEL_MATERIAL_IDS } from "./voxel-scene";

/**
 * A gentle voxel ramp that drains into a pit: the measured probe for liquid
 * left behind on a slope.
 *
 * A ramp of whole voxels is a staircase of level treads, each `run` cells
 * long. A stream is laid on its top for a moment and then stops; what is
 * still on the ramp some seconds later is the residue. On a real ramp of this
 * fall the water would be gone in about three seconds.
 *
 *  - **channel** is the water slide's: one cell of fall in twenty, between
 *    walls fourteen cells apart.
 *  - **staircase** is steeper and open: one in sixteen, the full width of the
 *    box.
 *  - **flat** is the control: the staircase's stream on a level floor at the
 *    ramp's top height. Nothing drains it, and a slope term must not change it.
 *  - **basin** is the guard: the channel dammed at its foot and filled with
 *    still water two cells over its top tread. It must stay still.
 *
 * Lengths below are in cells unless they say otherwise.
 */

export type NbflipRampVariant = "channel" | "staircase" | "flat" | "basin";
export const NBFLIP_RAMP_VARIANTS = ["channel", "staircase", "flat", "basin"] as const satisfies readonly NbflipRampVariant[];
export const nbflipRampSceneId = (variant: NbflipRampVariant) => `nbflip-ramp-${variant}`;

/** Four pages of 32 cells in a row. */
export const NBFLIP_RAMP_GRID = [128, 32, 32] as const;
export const NBFLIP_RAMP_CELL_SIZE_M = 0.02;
export const NBFLIP_RAMP_STEP_HZ = 60;
/** The ramp's length along x; the pit takes the rest of the box. */
export const NBFLIP_RAMP_LENGTH = 96;
/** The ramp's height at x = 0, and the pit's floor. */
const TOP = 17, PIT_FLOOR = 2;
/** The channel's walls: solid for z below the first and from the second on, this far over the ramp's top. */
const CHANNEL = [9, 23] as const, WALL = 9;
/** The basin's dam, from the ramp's foot along x, and its waterline. */
const DAM = 4;
export const NBFLIP_RAMP_BASIN_LEVEL = 18;
/** How long a run lasts after its stream stops, and how long the basin stands. */
export const NBFLIP_RAMP_DRAIN_SECONDS = 6;
export const NBFLIP_RAMP_BASIN_SECONDS = 4;

/** A variant's fall (one cell in `run`), channel walls, and stream: radius, speed in m/s and duration. */
export function nbflipRampShape(variant: NbflipRampVariant) {
  const walled = variant === "channel" || variant === "basin";
  return {
    run: walled ? 20 : 16,
    level: variant === "flat",
    wall: walled ? CHANNEL : undefined,
    stream: variant === "basin" ? undefined : walled ? { radius: 4, speed_m_s: 1.92, seconds: 1.5 } : { radius: 3, speed_m_s: 1.44, seconds: 2 },
  };
}

/** The top of the solid under a column of cells: the ramp's tread, then the pit's floor. */
export function nbflipRampBed(variant: NbflipRampVariant, x: number): number {
  if (x >= NBFLIP_RAMP_LENGTH) return PIT_FLOOR;
  const { run, level } = nbflipRampShape(variant);
  return Math.floor(TOP - (level ? 0.5 : x + 0.5) / run);
}

/** What a variant's stream emits, in cells. */
export function nbflipRampEmittedCells(variant: NbflipRampVariant): number {
  const stream = nbflipRampShape(variant).stream;
  return stream ? Math.PI * stream.radius ** 2 * stream.speed_m_s * stream.seconds / NBFLIP_RAMP_CELL_SIZE_M : 0;
}

/** A variant at `stepHz` steps a second: the stream's slug is one step long, so it is the time step's. */
export function createNbflipRamp(variant: NbflipRampVariant, stepHz = NBFLIP_RAMP_STEP_HZ): SceneDescription {
  const grid = NBFLIP_RAMP_GRID, h = NBFLIP_RAMP_CELL_SIZE_M, dt = 1 / stepHz, [nx, , nz] = grid;
  const { wall, stream } = nbflipRampShape(variant);
  const seconds = stream ? stream.seconds + NBFLIP_RAMP_DRAIN_SECONDS : NBFLIP_RAMP_BASIN_SECONDS;
  const scene = nbflipDomain(nbflipRampSceneId(variant), grid, h, Math.round(seconds * stepHz), 0, grid, dt);
  const fill = (minimum: readonly [number, number, number], maximumExclusive: readonly [number, number, number]) =>
    scene.solidVoxels.push({ operation: "fill", minimum, maximumExclusive, materialId: VOXEL_MATERIAL_IDS.container });
  // One box for each tread, then the pit's floor.
  for (let x = 0, end = 0; x < nx; x = end) {
    const top = nbflipRampBed(variant, x);
    for (end = x + 1; end < nx && nbflipRampBed(variant, end) === top; end++);
    fill([x, 0, 0], [end, top, nz]);
  }
  if (wall) {
    fill([0, 0, 0], [NBFLIP_RAMP_LENGTH, TOP + WALL, wall[0]]);
    fill([0, 0, wall[1]], [NBFLIP_RAMP_LENGTH, TOP + WALL, nz]);
  }
  if (stream) {
    // The stream lies on the top tread and leaves along the ramp.
    scene.fluid.inflow = {
      center_m: cellPoint(grid, h, [6, nbflipRampBed(variant, 6) + stream.radius + 0.5, nz / 2]),
      radius_m: stream.radius * h,
      length_m: stream.speed_m_s * dt,
      velocity_m_s: { x: stream.speed_m_s, y: 0, z: 0 },
      start_s: 0,
      end_s: stream.seconds,
      ramp_s: 0,
    };
  } else {
    fill([NBFLIP_RAMP_LENGTH, 0, 0], [NBFLIP_RAMP_LENGTH + DAM, TOP + WALL, nz]);
    scene.fluid.initialLiquidVolumes = [{
      shape: "box",
      min_m: cellPoint(grid, h, [0, 0, wall![0]]),
      max_m: cellPoint(grid, h, [NBFLIP_RAMP_LENGTH, NBFLIP_RAMP_BASIN_LEVEL, wall![1]]),
    }];
  }
  // The box is four pages and the water reaches every one: with the two blocks that hold no page, six.
  scene.fluid.pagePool = { blocks: 6 };
  return scene;
}

/** From the side and a little above, the ramp falling from left to right. */
export function nbflipRampCamera(): Partial<CameraState> {
  return { azimuth_rad: 0.3, elevation_rad: 0.45, distance_m: 3.4, target_m: { x: 0, y: 0.25, z: 0 } };
}
