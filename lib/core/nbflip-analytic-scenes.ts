import type { SceneDescription } from "./model";
import type { SolidWorldVoxelPatch } from "./solid-world";
import { cellPoint, nbflipDomain } from "./nbflip-paper-scenes";
import { VOXEL_MATERIAL_IDS } from "./voxel-scene";

/**
 * Small, isolated NB-FLIP scenes whose motion has an analytic (or classic
 * experimental) answer. Each isolates one behaviour of the method in a few
 * dozen frames on one shared lattice, so a probe builds its pipelines once:
 * a closed free-slip box of 64 x 64 x 32 cells at 1 cm, stepped at 1/60 s,
 * every axis whole 32-cell pages; the liquid spans the depth (z), so the
 * flow is two-dimensional.
 *
 *  - **rest**: a still pool. Exact: no motion, a level surface at 0.24 m,
 *    hydrostatic pressure, every region COLD after the start-up transient.
 *  - **rest-step**: the same pool over a submerged step and around a pier
 *    that pierces the surface. Exact as rest: solids change nothing.
 *  - **free-fall**: a block falling from rest, no pool. Exact: a rigid
 *    translation x0 + g t^2 / 2 with no deformation and zero pressure, until
 *    it nears the floor. Vertical: the solver's forces take gravity's y alone.
 *  - **standing-wave**: the fundamental seiche of a 0.64 m tank 0.24 m deep,
 *    2 cm amplitude. Linear theory: the end heights follow A cos(wt) with
 *    w^2 = g k tanh(k H), k = pi / L; the middle stays at H.
 *  - **dam-break**: a column a = 16 cells wide and 2a tall at the left wall.
 *    Martin & Moyce (1952), n^2 = 2: the front's Z = x/a against
 *    T = t sqrt(2g/a).
 *  - **mound**: a cylinder of liquid released onto a pool (Ferstl et al.
 *    Figure 4 in two dimensions). Inviscid with dissipative transport: the
 *    total energy KE + PE must never rise, and the surface returns to rest.
 */
export const NBFLIP_ANALYTIC_GRID = [64, 64, 32] as const;
export const NBFLIP_ANALYTIC_CELL_M = 0.01;
export const NBFLIP_ANALYTIC_DT_S = 1 / 60;
export const NBFLIP_ANALYTIC_GRAVITY_M_S2 = 9.81;
export const NBFLIP_ANALYTIC_POOL_CELLS = 24;

export const NBFLIP_ANALYTIC_CASES = ["rest", "rest-step", "free-fall", "standing-wave", "dam-break", "mound"] as const;
export type NbflipAnalyticCase = (typeof NBFLIP_ANALYTIC_CASES)[number];
export const nbflipAnalyticSceneId = (name: NbflipAnalyticCase) => `nbflip-analytic-${name}`;

/** Frames each case runs: long enough to show its answer, no longer. */
export const NBFLIP_ANALYTIC_FRAMES: Readonly<Record<NbflipAnalyticCase, number>> = {
  "rest": 60, "rest-step": 60, "free-fall": 16, "standing-wave": 60, "dam-break": 18, "mound": 120,
};

/** free-fall: the block's cells (x, y) and the tilted gravity. */
export const NBFLIP_ANALYTIC_FALL_BLOCK = [[24, 36], [40, 52]] as const;
export const NBFLIP_ANALYTIC_FALL_GRAVITY = { x: 0, y: -NBFLIP_ANALYTIC_GRAVITY_M_S2, z: 0 } as const;
/** rest-step: a submerged step and a pier through the surface, in cells. */
export const NBFLIP_ANALYTIC_STEP = [[24, 0], [40, 12]] as const;
export const NBFLIP_ANALYTIC_PIER = [[46, 0], [50, 40]] as const;
/** standing-wave amplitude, m. */
export const NBFLIP_ANALYTIC_WAVE_AMPLITUDE_M = 0.02;
/** dam-break column width a, cells (height 2a). */
export const NBFLIP_ANALYTIC_DAM_CELLS = 16;
/** mound: pool depth and the cylinder's centre height and radius, cells. */
export const NBFLIP_ANALYTIC_MOUND = { pool: 16, centre: 30, radius: 8 } as const;

export function nbflipAnalyticWaveOmega(): number {
  const L = NBFLIP_ANALYTIC_GRID[0] * NBFLIP_ANALYTIC_CELL_M, H = NBFLIP_ANALYTIC_POOL_CELLS * NBFLIP_ANALYTIC_CELL_M, k = Math.PI / L;
  return Math.sqrt(NBFLIP_ANALYTIC_GRAVITY_M_S2 * k * Math.tanh(k * H));
}

/**
 * Martin & Moyce (1952), Phil. Trans. R. Soc. A 244, Plate 2, n^2 = 2: the
 * surge front's Z = x/a against T = t sqrt(2g/a), as tabulated in the SPH and
 * MPS literature that validates against it (Koshizuka & Oka 1996, Fig. 5).
 */
export const MARTIN_MOYCE_N2_2: readonly (readonly [T: number, Z: number])[] = [
  [0.41, 1.11], [0.84, 1.22], [1.19, 1.44], [1.43, 1.67], [1.63, 1.89], [1.83, 2.11], [1.98, 2.33],
  [2.20, 2.56], [2.32, 2.78], [2.51, 3.00], [2.65, 3.22], [2.83, 3.44], [2.98, 3.67], [3.11, 3.89],
];
/** The experiment's front position Z at T, linearly interpolated (undefined outside the table). */
export function martinMoyceFront(T: number): number | undefined {
  const table = MARTIN_MOYCE_N2_2;
  if (T < table[0]![0] || T > table[table.length - 1]![0]) return undefined;
  for (let i = 1; i < table.length; i++) {
    const [t0, z0] = table[i - 1]!, [t1, z1] = table[i]!;
    if (T <= t1) return z0 + (z1 - z0) * (T - t0) / (t1 - t0);
  }
  return undefined;
}

const solidCells = (min: readonly [number, number], max: readonly [number, number]): SolidWorldVoxelPatch => ({
  operation: "fill", minimum: [min[0], min[1], 0], maximumExclusive: [max[0], max[1], NBFLIP_ANALYTIC_GRID[2]], materialId: VOXEL_MATERIAL_IDS.container,
});
/** A box of liquid spanning the depth, between cell corners (x, y). */
const liquidSlab = (min: readonly [number, number], max: readonly [number, number]) => {
  const grid = NBFLIP_ANALYTIC_GRID, h = NBFLIP_ANALYTIC_CELL_M;
  return { shape: "box" as const, min_m: cellPoint(grid, h, [min[0], min[1], 0]), max_m: cellPoint(grid, h, [max[0], max[1], grid[2]]) };
};

export function createNbflipAnalyticScene(name: NbflipAnalyticCase): SceneDescription {
  const grid = NBFLIP_ANALYTIC_GRID, h = NBFLIP_ANALYTIC_CELL_M;
  const pool = name === "rest" || name === "rest-step" || name === "standing-wave" ? NBFLIP_ANALYTIC_POOL_CELLS
    : name === "mound" ? NBFLIP_ANALYTIC_MOUND.pool : 0;
  const scene = nbflipDomain(nbflipAnalyticSceneId(name), grid, h, NBFLIP_ANALYTIC_FRAMES[name], pool, grid, NBFLIP_ANALYTIC_DT_S, NBFLIP_ANALYTIC_GRAVITY_M_S2);
  scene.solidVoxels = [];
  if (name === "rest-step") scene.solidVoxels = [solidCells(...NBFLIP_ANALYTIC_STEP), solidCells(...NBFLIP_ANALYTIC_PIER)];
  if (name === "free-fall") {
    scene.fluid.gravity_m_s2 = { ...NBFLIP_ANALYTIC_FALL_GRAVITY };
    scene.fluid.initialLiquidVolumes = [liquidSlab(...NBFLIP_ANALYTIC_FALL_BLOCK)];
  }
  if (name === "standing-wave") {
    const L = grid[0] * h;
    scene.fluid.initialHeightField = { kind: "cosine", baseHeight_m: pool * h, amplitude_m: NBFLIP_ANALYTIC_WAVE_AMPLITUDE_M,
      wavelength_m: 2 * L, originX_m: -L / 2 };
  }
  if (name === "dam-break") {
    const a = NBFLIP_ANALYTIC_DAM_CELLS;
    scene.fluid.initialLiquidVolumes = [liquidSlab([0, 0], [a, 2 * a])];
  }
  if (name === "mound") {
    const { centre, radius } = NBFLIP_ANALYTIC_MOUND;
    scene.fluid.initialLiquidVolumes = [{ shape: "cylinder", center_m: cellPoint(grid, h, [grid[0] / 2, centre, grid[2] / 2]),
      radius_m: radius * h, halfHeight_m: grid[2] * h / 2 }];
  }
  return scene;
}
