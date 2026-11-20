/**
 * Pour solver: the baked static structure of `nbflip-figure-7-pour` (docs/plans/pour-rewrite-2026-11-20.md §2.1).
 *
 * The solver supports exactly one scene. Everything static is decided here, once, on the CPU: the lattice, the
 * glass (the brick engine's own voxel mask, as an exact integer annulus rule that construction re-verifies cell by
 * cell against brickStaticMask), the brick classes and the inflow's per-step schedule. Construction fails
 * explicitly for any other scene or any change to the geometry or inflow this bake covers.
 *
 * Units: positions and level-set values in cells (h = 22.18 mm), velocities in m/s.
 * Storage is dense and brick-major: a brick is 8^3 cells, its 512 cells contiguous, and the brick table is
 * 16 x 32 x 16 (brick id bx | by << 4 | bz << 9).
 */
import type { FluidInflow, SceneDescription } from "../../core/model";
import { averageInflowStrength, createInflowGridBoundary, inflowOutletCenter } from "../../core/inflow-boundary";
import { inflowStrength } from "../../core/initial-fluid";
import { NBFLIP_POUR_CELL_SIZE_M, NBFLIP_POUR_GRID, NBFLIP_POUR_SCENE_ID } from "../../core/nbflip-paper-scenes";
import { sceneLatticeDimensions } from "../../core/scene-lattice-dimensions";

export const NX = 128, NY = 256, NZ = 128;
export const CELLS = NX * NY * NZ;
export const BRICK = 8;
export const BX = NX / BRICK, BY = NY / BRICK, BZ = NZ / BRICK;
export const BRICKS = BX * BY * BZ;
export const CELL_M = NBFLIP_POUR_CELL_SIZE_M;
export const DT = 1 / 60;
export const GRAVITY = 9.81;
/** Level-set magnitude clamp, in cells. */
export const PHI_MAX = 6;
/** The glass, as integers: with s = (2x + 1 - NX)^2 + (2z + 1 - NZ)^2 (four times the squared radius of a cell
 *  centre from the axis), cell (x, y, z) is solid iff y < RIM_Y and s in [WALL_S0, WALL_S1], or y == RIM_Y and s in
 *  [RIM_S0, RIM_S1]. Rows above RIM_Y are open; the glass's floor is the box's floor. */
export const RIM_Y = 191, WALL_S0 = 10486, WALL_S1 = 11577, RIM_S0 = 10594, RIM_S1 = 11482;

/** Brick class bits (static). No pour brick is wholly solid (the wall is 2.6 cells), so none is DEAD today. */
export const CLASS_DEAD = 1;

export function brickId(bx: number, by: number, bz: number): number { return bx | (by << 4) | (bz << 9); }
export function cellIndex(x: number, y: number, z: number): number {
  return (brickId(x >> 3, y >> 3, z >> 3) << 9) | ((x & 7) | ((y & 7) << 3) | ((z & 7) << 6));
}
export function glassSolid(x: number, y: number, z: number): boolean {
  if (y > RIM_Y) return false;
  const dx = 2 * x + 1 - NX, dz = 2 * z + 1 - NZ, s = dx * dx + dz * dz;
  return y < RIM_Y ? s >= WALL_S0 && s <= WALL_S1 : s >= RIM_S0 && s <= RIM_S1;
}
export function cellOpen(x: number, y: number, z: number): boolean {
  if (x < 0 || y < 0 || z < 0 || x >= NX || y >= NY || z >= NZ) return false;
  return !glassSolid(x, y, z);
}

/** The inflow in the solver's units (cells, m/s), as the brick engine forms it (brick-source.ts brickInflow). */
export interface PourInflow {
  readonly outlet: readonly [number, number, number];
  readonly radius: number;
  readonly velocity_mps: readonly [number, number, number];
  readonly scale: number;
  readonly axis: 0 | 1 | 2;
  readonly face: number;
  /** Cells a second at full strength (|v| / h). */
  readonly speed: number;
  /** Cells^3 a second at full strength (brick-source.ts brickInflow's flow: the grid boundary's rate / h^3). */
  readonly flow: number;
  /** The plug's cell box (lo inclusive, hi exclusive), the brick engine's pinned source box. */
  readonly box: readonly [readonly number[], readonly number[]];
  /** Per step s (the step (s dt, (s + 1) dt]): strength and sweep (cells) as f32, then flags (1 on, 2 continues) as
   *  u32 bits. Off past the table. */
  readonly schedule: Float32Array<ArrayBuffer>;
  /** The schedule's words as u32 (the flags read here, the GPU table's upload). */
  readonly scheduleWords: Uint32Array<ArrayBuffer>;
  readonly steps: number;
}
/** The brick engine's volume control time for this scene (s): the method's default volumeControlSeconds, which the
 *  pour's profile keeps (velocity-kernels.wgsl.ts V_CONTROL = 1 / it). */
export const VOLUME_CONTROL_S = 0.1;
/** The brick engine's plug overlap (cells), brick-source.ts SRC_PLUG_OVERLAP. */
export const PLUG_OVERLAP = 2;

export function bakeInflow(scene: SceneDescription, inflow: FluidInflow): PourInflow {
  const h = CELL_M, origin = [-scene.container.width_m / 2, 0, -scene.container.depth_m / 2];
  const grid = createInflowGridBoundary(inflow, scene.container, [NX, NY, NZ]);
  const out = inflowOutletCenter(inflow);
  const outlet = [(out.x - origin[0]!) / h, (out.y - origin[1]!) / h, (out.z - origin[2]!) / h] as const;
  const v = [inflow.velocity_m_s.x, inflow.velocity_m_s.y, inflow.velocity_m_s.z] as const, speed = Math.hypot(...v);
  const axis = grid.axis as 0 | 1 | 2, lattice = [NX, NY, NZ];
  const face = Math.min(Math.max(Math.round(outlet[axis]) - 1, 0), lattice[axis]! - 2);
  if (face !== grid.faceIndex) throw new Error(`pour: inflow face ${face} differs from the grid boundary's ${grid.faceIndex}`);
  const d = v.map(x => x / speed), reach = speed * DT / h + PLUG_OVERLAP, r = inflow.radius_m / h;
  const end = outlet.map((x, a) => x + reach * d[a]!);
  const lo = outlet.map((x, a) => Math.min(Math.max(Math.floor(Math.min(x, end[a]!) - r), 0), lattice[a]!));
  const hi = outlet.map((x, a) => Math.min(Math.max(Math.ceil(Math.max(x, end[a]!) + r), 0), lattice[a]!));
  // The schedule runs to two steps past the inflow's end; every later step is off.
  const steps = Math.ceil(inflow.end_s / DT) + 2;
  if (3 * steps > BRICKS) throw new Error(`pour: an inflow schedule of ${steps} steps exceeds its table`);
  const schedule = new Float32Array(3 * steps), scheduleWords = new Uint32Array(schedule.buffer);
  for (let s = 0; s < steps; s++) {
    const t = s * DT, strength = averageInflowStrength(inflow, t, t + DT);
    // brick-source.ts brickInflowStep's flags: on, and continues (the nozzle ran before this step).
    const continues = inflow.enabled !== false && t > inflow.start_s + 0.5 * DT && inflowStrength(t, inflow.start_s, inflow.end_s, inflow.ramp_s) > 0;
    schedule[3 * s] = strength; schedule[3 * s + 1] = (speed / h) * strength * DT;
    scheduleWords[3 * s + 2] = (strength > 0 ? 1 : 0) | (continues ? 2 : 0);
  }
  return { outlet, radius: r, velocity_mps: v, scale: grid.apertureScale, axis, face, speed: speed / h, flow: grid.flowRate_m3_s / h ** 3, box: [lo, hi], schedule, scheduleWords, steps };
}

export interface PourBake {
  /** Static class bits per brick. */
  readonly brickClass: Uint32Array<ArrayBuffer>;
  readonly inflow: PourInflow;
}

/** Fail explicitly unless `scene` is exactly the baked pour. `solid` is the brick engine's voxel mask (brickStaticMask). */
export function assertPourScene(scene: SceneDescription, solid: (x: number, y: number, z: number) => boolean): void {
  const fail = (why: string): never => { throw new Error(`Pour solver supports only the ${NBFLIP_POUR_SCENE_ID} scene: ${why}`); };
  if (scene.sceneId !== NBFLIP_POUR_SCENE_ID) fail(`scene id ${scene.sceneId}`);
  const [nx, ny, nz] = sceneLatticeDimensions(scene);
  if (nx !== NX || ny !== NY || nz !== NZ || String(NBFLIP_POUR_GRID) !== String([NX, NY, NZ])) fail(`lattice ${nx}x${ny}x${nz}`);
  if (Math.abs(scene.voxelDomain.finestCellSize_m - CELL_M) > 1e-12) fail(`cell size ${scene.voxelDomain.finestCellSize_m}`);
  if (Math.abs(scene.container.width_m - NX * CELL_M) > 1e-9 || Math.abs(scene.container.height_m - NY * CELL_M) > 1e-9
    || Math.abs(scene.container.depth_m - NZ * CELL_M) > 1e-9) fail("container size");
  if (scene.container.top !== "closed") fail("container top must be closed");
  if (scene.container.fluidWallMode !== "free-slip") fail("walls must be free-slip");
  if ((scene.container.shape ?? "box") !== "box") fail("container must be a box");
  if ((scene.rigidBodies ?? []).length) fail("rigid bodies are not supported");
  const g = scene.fluid.gravity_m_s2;
  if (g.x !== 0 || g.z !== 0 || Math.abs(g.y + GRAVITY) > 1e-9) fail(`gravity must be (0,-${GRAVITY},0)`);
  if (scene.fluid.dynamicViscosity_Pa_s !== 0 || scene.fluid.surfaceTension_N_m !== 0) fail("inviscid, tension-free only");
  const f = scene.fluid;
  if (scene.container.fillFraction !== 0 || f.initialLiquidVolumes?.length || f.initialHeightField || f.initialDamBreakDimensions_m
    || f.initialBrickSeeds_m?.length) fail("the box must start empty");
  const v = scene.fluid.initialVelocity_m_s ?? { x: 0, y: 0, z: 0 };
  if (v.x !== 0 || v.y !== 0 || v.z !== 0) fail("initial velocity must be zero");
  const inflow = scene.fluid.inflow;
  if (!inflow) fail("the inflow is missing");
  if (inflow!.ramp_s !== 0 || inflow!.start_s !== 0) fail("the inflow must start at 0 with no ramp");
  if (inflow!.enabled === false) fail("the inflow must be enabled");
  if (f.scheduledDrops?.length || f.refinementRegions?.length) fail("drops and refinement regions are not supported");
  // The glass: every cell of the box, the integer rule against the brick engine's mask.
  for (let z = 0; z < NZ; z++) for (let y = 0; y < NY; y++) for (let x = 0; x < NX; x++)
    if (solid(x, y, z) !== glassSolid(x, y, z)) fail(`the glass differs from the baked rule at cell (${x}, ${y}, ${z})`);
}

/** Bake static classes and the inflow schedule. */
export function bakePour(scene: SceneDescription): PourBake {
  const brickClass = new Uint32Array(BRICKS);
  for (let bz = 0; bz < BZ; bz++) for (let by = 0; by < BY; by++) for (let bx = 0; bx < BX; bx++) {
    // DEAD: no vertex of the brick is a corner of an open cell.
    let touches = false;
    for (let z = bz * 8; z < bz * 8 + 8 && !touches; z++) for (let y = by * 8; y < by * 8 + 8 && !touches; y++)
      for (let x = bx * 8; x < bx * 8 + 8 && !touches; x++)
        for (let c = 0; c < 8 && !touches; c++)
          if (cellOpen(x - (c & 1), y - ((c >> 1) & 1), z - ((c >> 2) & 1))) touches = true;
    brickClass[brickId(bx, by, bz)] = touches ? 0 : CLASS_DEAD;
  }
  return { brickClass, inflow: bakeInflow(scene, scene.fluid.inflow!) };
}
