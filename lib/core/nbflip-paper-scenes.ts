import { cloneScene, defaultScene, type CameraState, type InitialLiquidVolume, type SceneDescription, type ScheduledLiquidDrop } from "./model";
import type { LiquidExtrusion } from "./liquid-extrusion";
import type { MethodProfile } from "./method-contract";
import type { SolidWorldVoxelPatch } from "./solid-world";
import { VOXEL_MATERIAL_IDS } from "./voxel-scene";

/**
 * Scenes of Ferstl, Ando, Wojtan, Westermann & Thuerey, *Narrow Band FLIP for
 * Liquid Simulations* (Eurographics 2016) --
 * `docs/papers/narrow-band-flip-2026/ferstl-2016-nbflip/` -- rebuilt from the
 * paper's supplemental video (https://youtu.be/fEj2Hw7Vxd8, 24 fps).
 *
 * The paper publishes each scene's grid and little else, so every scene below
 * says what was measured from the footage and how. What they share comes from
 * the authors' own mantaflow example of the method, `scenes/flip05_nbflip.py`:
 *
 *  - **One video frame is one solver frame.** The step here is one frame.
 *  - **Gravity is 0.003 of the largest grid dimension**, in cells per frame
 *    squared: mantaflow scales its `gravity = (0, -0.003, 0)` by that
 *    dimension. Each scene's fall was also measured, and is stated beside it.
 *    An inviscid solve has no other scale, so the cell size is whatever makes
 *    that fall terrestrial gravity at 24 frames a second.
 *  - **The box is closed and its walls are free-slip.**
 *  - **Shapes sit on round fractions of the grid.**
 *
 * mantaflow's outermost layer of cells is the wall, so its liquid has two
 * cells fewer on each axis than the published grid. The footage cannot tell
 * the two apart -- both fit the Teaser's frame to 0.89 px -- and the scenes
 * here take the published grid as the liquid's box: this method's 4h tiles
 * need every axis to be a multiple of four.
 *
 * They run on Uniform Narrow-band FLIP because that method is this paper's.
 * Its particle band is 4h where the mantaflow scenes used 3h.
 */

export const NBFLIP_FRAME_RATE_HZ = 24;
const GRAVITY_M_S2 = 9.81;
/** The cell at which a fall of `cellsPerFrame2` is terrestrial gravity. */
const cellSizeForFall_m = (cellsPerFrame2: number) => GRAVITY_M_S2 / (cellsPerFrame2 * NBFLIP_FRAME_RATE_HZ ** 2);
/** mantaflow's gravity for a grid, in cells per frame squared. */
const mantaflowFall = (grid: readonly [number, number, number]) => 0.003 * Math.max(...grid);

/**
 * Pins the scene's own step -- the method's default is a sixtieth of a second.
 * Grid coverage is the method's default: dynamic, with the tiles the surface
 * crosses at the published cell size and the rest of the liquid at 4h.
 */
export const NBFLIP_METHOD_PROFILE: MethodProfile = {
  methodId: "uniform-narrow-band-flip", quality: "balanced", overrides: { timeStep: "scene" },
};

/**
 * A closed, free-slip box of `grid` cells (width, height, depth) holding a
 * pool `poolCells` deep, stepped once a frame for `frames` frames. Surface
 * tension is zeroed because the paper's solver has none.
 */
function nbflipDomain(
  sceneId: string, grid: readonly [number, number, number], cellSize_m: number, frames: number, poolCells: number,
): SceneDescription {
  const scene = cloneScene(defaultScene);
  const [nx, ny, nz] = grid, h = cellSize_m, dt = 1 / NBFLIP_FRAME_RATE_HZ;
  scene.sceneId = sceneId;
  scene.randomSeed = 2016;
  scene.duration_s = frames * dt;
  scene.solidVoxels = [];
  scene.rigidBodies = [];
  scene.container = {
    ...scene.container,
    width_m: nx * h,
    height_m: ny * h,
    depth_m: nz * h,
    fillFraction: poolCells / ny,
    top: "closed",
    fluidWallMode: "free-slip",
  };
  scene.voxelDomain = { ...scene.voxelDomain, finestCellSize_m: h };
  scene.nominalResolution = { length_m: h };
  scene.numerics = { ...scene.numerics, fixedDt_s: dt, maxDt_s: dt };
  scene.fluid = {
    ...scene.fluid,
    gravity_m_s2: { x: 0, y: -GRAVITY_M_S2, z: 0 },
    surfaceTension_N_m: 0,
    initialCondition: "tank-fill",
  };
  delete scene.fluid.initialDamBreakDimensions_m;
  delete scene.fluid.initialDamBreakOrigin_m;
  delete scene.fluid.initialBrickSeeds_m;
  delete scene.fluid.initialBrickSeedsAdditive;
  delete scene.fluid.initialLiquidVolumes;
  delete scene.fluid.inflow;
  delete scene.terrain;
  return scene;
}

/**
 * A camera on the domain's centre plane, as a pinhole fit of the footage gives
 * it: `height` cells above the floor and `back` cells in front of the domain's
 * centre, pitched down, with its focal length in pixels of the 720 px frame.
 */
function footageCamera(cellSize_m: number, height: number, back: number, pitch_deg: number, focal_px: number): Partial<CameraState> {
  const pitch_rad = pitch_deg * Math.PI / 180;
  return {
    azimuth_rad: 0,
    elevation_rad: pitch_rad,
    distance_m: back / Math.cos(pitch_rad) * cellSize_m,
    target_m: { x: 0, y: (height - back * Math.tan(pitch_rad)) * cellSize_m, z: 0 },
    tanHalfFov: 360 / focal_px,
  };
}

/** A box of liquid between two corners given in cells from the domain's minimum corner. */
function liquidBox(grid: readonly [number, number, number], cellSize_m: number, min: readonly [number, number, number], max: readonly [number, number, number]): InitialLiquidVolume {
  const at = (cells: readonly [number, number, number]) => ({ x: (cells[0] - grid[0] / 2) * cellSize_m, y: cells[1] * cellSize_m, z: (cells[2] - grid[2] / 2) * cellSize_m });
  return { shape: "box", min_m: at(min), max_m: at(max) };
}

/**
 * An upright solid cylinder standing on the floor, as voxel runs: every cell
 * whose centre its circle covers, one run per row of z.
 */
function cylinderVoxels(centreX: number, centreZ: number, radius: number, height: number): SolidWorldVoxelPatch[] {
  const runs: SolidWorldVoxelPatch[] = [];
  for (let z = Math.floor(centreZ - radius); z < centreZ + radius; z++) {
    const half = Math.sqrt(radius ** 2 - (z + 0.5 - centreZ) ** 2);
    const from = Math.ceil(centreX - half - 0.5), to = Math.floor(centreX + half - 0.5) + 1;
    if (to > from) runs.push({ operation: "fill", minimum: [from, 0, z], maximumExclusive: [to, height, z + 1], materialId: VOXEL_MATERIAL_IDS.cylinder });
  }
  return runs;
}

/**
 * Figure 8, "Letters".
 *
 * What is published and what is not, because the difference decides how much
 * of the scene below is a reconstruction:
 *
 *  - **Published exactly**, in Table 1 and Sec. 5: the grid, 256 x 192 x 128;
 *    "a series of letter-shaped drops falling into a basin"; a band of R = 3h
 *    with eight particles per cell. The grid is the constant below and nothing
 *    here rounds or reinterprets it.
 *  - **Not published at all**: which axis is which, the pool depth, the cell
 *    size, the time step, gravity, the typeface, and where and when each
 *    letter appears. Every one of those is measured from the paper's
 *    supplemental video (https://youtu.be/ETr1Fptm6Z8, the single-view NB-FLIP
 *    take that opens at video frame 4343 of a 24 fps encode) and stated here
 *    in whole cells and whole frames wherever the footage supports that.
 *
 * How each measurement was made:
 *
 *  - **Axes.** A pinhole camera fitted to the six visible corners of the pool
 *    closes to 0.84 px rms with x = 256 across, y = 192 up and z = 128 deep.
 *    The other assignment fits worse, and its 128-cell ceiling would not hold
 *    the letters at the heights they spawn at.
 *  - **Pool.** The same fit puts the rest surface 36.7 cells above the floor.
 *  - **Schedule.** Letters A to I appear at rest every ten frames, starting
 *    with A in the take's first frame, and each then falls freely.
 *  - **Scale.** Their tracked fall is 0.72 to 0.82 cells per frame squared;
 *    0.8 is used. Only that number is dynamically meaningful -- an inviscid
 *    solve has no other scale -- so the step is one frame, and the cell size
 *    is whatever makes 0.8 cells per frame squared equal terrestrial gravity.
 *  - **Placement.** Each letter's centre sits on fifths of the domain: x
 *    cycles centre, left, right; z and y are per letter, below.
 *  - **Letterforms.** The nearest match is DejaVu Sans (the face Blender's
 *    built-in font derives from) with its outline grown outward and
 *    extruded, which is what a bevelled text object makes. It is a match and
 *    not an identification: the footage's H is about a fifth wider and its E
 *    and F about two cells narrower than these outlines.
 */

/** Table 1: "Fig. 8 / Letters, 256 x 192 x 128". Width, height, depth. */
export const NBFLIP_LETTERS_GRID = [256, 192, 128] as const;
/** The supplemental video's frame rate; one solver step is one frame. */
export const NBFLIP_LETTERS_FRAME_RATE_HZ = NBFLIP_FRAME_RATE_HZ;
export const NBFLIP_LETTERS_TIME_STEP_S = 1 / NBFLIP_LETTERS_FRAME_RATE_HZ;
/** Measured free fall of the letters, in cells per frame squared. */
export const NBFLIP_LETTERS_GRAVITY_CELLS_PER_FRAME2 = 0.8;
export const NBFLIP_LETTERS_GRAVITY_M_S2 = 9.81;
/** The cell at which the measured fall is terrestrial gravity: about 21.3 mm. */
export const NBFLIP_LETTERS_CELL_SIZE_M = NBFLIP_LETTERS_GRAVITY_M_S2
  / (NBFLIP_LETTERS_GRAVITY_CELLS_PER_FRAME2 * NBFLIP_LETTERS_FRAME_RATE_HZ ** 2);
/** Rest depth of the basin, in cells. */
export const NBFLIP_LETTERS_POOL_CELLS = 37;
/** Frames between one letter appearing and the next. */
export const NBFLIP_LETTERS_INTERVAL_FRAMES = 10;
/** Frames from A appearing to the end of the take. */
export const NBFLIP_LETTERS_FRAMES = 193;

/** Cap height of the raw outline, before it is grown, in cells. */
const CAP_HEIGHT_CELLS = 35.5;
/** How far the outline is grown: stems come to 12.8 cells and letters to 43.5 tall. */
const OUTLINE_OFFSET_CELLS = 4;
/** Front to back, 13 cells, with the faces' edges rounded over 4. */
const HALF_DEPTH_CELLS = 6.5;
const EDGE_RADIUS_CELLS = 4;

/**
 * Where each letter's centre spawns, as fractions of the domain: across (x),
 * up (y) and toward the camera (z). In drop order.
 */
const LETTER_CENTRES: readonly (readonly [letter: keyof typeof LETTER_OUTLINES, x: number, y: number, z: number])[] = [
  ["A", 0.5, 0.6, 0.5],
  ["B", 0.2, 0.55, 0.8],
  ["C", 0.8, 0.55, 0.5],
  ["D", 0.5, 0.55, 0.2],
  ["E", 0.2, 0.6, 0.8],
  ["F", 0.8, 0.55, 0.5],
  ["G", 0.5, 0.6, 0.8],
  ["H", 0.2, 0.6, 0.2],
  ["I", 0.8, 0.55, 0.5],
];

/**
 * DejaVu Sans capitals as closed contours, flat `[x0, y0, x1, y1, ...]` in
 * units of the cap height and centred on each glyph's bounding box. The
 * quadratic splines are flattened to within 0.4% of the cap height, a seventh
 * of a cell at this size. Fill is even-odd: the second contour of A, B and D
 * is a counter.
 */
const LETTER_OUTLINES = {
  A: [
    [
    0, 0.3667, -0.1835, -0.1309, 0.1842, -0.1309,
    ],
    [
    -0.0764, 0.5, 0.077, 0.5, 0.4581, -0.5, 0.3175, -0.5, 0.2264, -0.2435, -0.2244, -0.2435, -0.3155, -0.5, -0.4581,
    -0.5,
    ],
  ],
  B: [
    [
    -0.2194, -0.0224, -0.2194, -0.3888, -0.0023, -0.3888, 0.0642, -0.3838, 0.1181, -0.3687, 0.1594, -0.3436, 0.1886,
    -0.3081, 0.2061, -0.262, 0.212, -0.2053, 0.2061, -0.1482, 0.1886, -0.1021, 0.1594, -0.067, 0.1181, -0.0422,
    0.0642, -0.0274, -0.0023, -0.0224,
    ],
    [
    -0.2194, 0.3888, -0.2194, 0.0874, -0.0191, 0.0874, 0.0674, 0.0967, 0.1286, 0.1246, 0.165, 0.1716, 0.1772, 0.2381,
    0.165, 0.3043, 0.1286, 0.3513, 0.0674, 0.3794, -0.0191, 0.3888,
    ],
    [
    -0.3547, 0.5, -0.009, 0.5, 0.0862, 0.4929, 0.1657, 0.4714, 0.2294, 0.4357, 0.2759, 0.3868, 0.3038, 0.3259,
    0.3131, 0.2528, 0.3024, 0.1705, 0.2703, 0.1068, 0.2173, 0.0628, 0.1443, 0.0392, 0.2059, 0.0197, 0.2576, -0.0101,
    0.2994, -0.0502, 0.3301, -0.0993, 0.3485, -0.1559, 0.3547, -0.22, 0.3445, -0.3026, 0.3142, -0.3715, 0.2636,
    -0.427, 0.1943, -0.4676, 0.1079, -0.4919, 0.0044, -0.5, -0.3547, -0.5,
    ],
  ],
  C: [
    [
    0.4032, 0.4236, 0.4032, 0.281, 0.3326, 0.3366, 0.2575, 0.3761, 0.1777, 0.3997, 0.0931, 0.4076, -0.0123, 0.3959,
    -0.0998, 0.361, -0.1695, 0.3027, -0.2201, 0.2225, -0.2504, 0.1215, -0.2605, -0.0003, -0.2504, -0.1218, -0.2201,
    -0.2226, -0.1695, -0.3027, -0.0998, -0.361, -0.0123, -0.3959, 0.0931, -0.4076, 0.1777, -0.3997, 0.2575, -0.3761,
    0.3326, -0.3366, 0.4032, -0.281, 0.4032, -0.4223, 0.3301, -0.4645, 0.2528, -0.4946, 0.1712, -0.5127, 0.0851,
    -0.5188, -0.0224, -0.5101, -0.1179, -0.484, -0.2013, -0.4406, -0.2726, -0.3798, -0.3297, -0.3039, -0.3706,
    -0.2154, -0.3951, -0.1142, -0.4032, -0.0003, -0.3951, 0.1138, -0.3706, 0.2153, -0.3297, 0.3039, -0.2726, 0.3798,
    -0.2013, 0.4406, -0.1179, 0.484, -0.0224, 0.5101, 0.0851, 0.5188, 0.1722, 0.5128, 0.2542, 0.495, 0.3311, 0.4653,
    ],
  ],
  D: [
    [
    -0.285, 0.3888, -0.285, -0.3888, -0.1216, -0.3888, -0.025, -0.383, 0.0577, -0.3654, 0.1265, -0.3361, 0.1815,
    -0.295, 0.2236, -0.2414, 0.2536, -0.1741, 0.2716, -0.0934, 0.2776, 0.001, 0.2716, 0.0948, 0.2536, 0.1751, 0.2236,
    0.2419, 0.1815, 0.2954, 0.1265, 0.3363, 0.0577, 0.3655, -0.025, 0.383, -0.1216, 0.3888,
    ],
    [
    -0.4203, 0.5, -0.1423, 0.5, -0.0067, 0.4924, 0.1097, 0.4698, 0.2067, 0.432, 0.2843, 0.3791, 0.3438, 0.3101,
    0.3863, 0.2241, 0.4118, 0.1211, 0.4203, 0.001, 0.4118, -0.1197, 0.3861, -0.2232, 0.3434, -0.3096, 0.2837,
    -0.3788, 0.2058, -0.4318, 0.1088, -0.4697, -0.0072, -0.4924, -0.1423, -0.5, -0.4203, -0.5,
    ],
  ],
  E: [
    [
    -0.3222, 0.5, 0.3101, 0.5, 0.3101, 0.3861, -0.1869, 0.3861, -0.1869, 0.0901, 0.2894, 0.0901, 0.2894, -0.0238,
    -0.1869, -0.0238, -0.1869, -0.3861, 0.3222, -0.3861, 0.3222, -0.5, -0.3222, -0.5,
    ],
  ],
  F: [
    [
    -0.2873, 0.5, 0.2873, 0.5, 0.2873, 0.3861, -0.152, 0.3861, -0.152, 0.0914, 0.2445, 0.0914, 0.2445, -0.0224,
    -0.152, -0.0224, -0.152, -0.5, -0.2873, -0.5,
    ],
  ],
  G: [
    [
    0.3027, -0.3567, 0.3027, -0.0881, 0.0817, -0.0881, 0.0817, 0.0231, 0.4367, 0.0231, 0.4367, -0.4062, 0.3543,
    -0.455, 0.2639, -0.4903, 0.1663, -0.5116, 0.0623, -0.5188, -0.0485, -0.5102, -0.1466, -0.4845, -0.2319, -0.4417,
    -0.3044, -0.3818, -0.3623, -0.3066, -0.4036, -0.2179, -0.4284, -0.1159, -0.4367, -0.0003, -0.4284, 0.1155,
    -0.4036, 0.2178, -0.3623, 0.3065, -0.3044, 0.3818, -0.2319, 0.4417, -0.1466, 0.4845, -0.0485, 0.5102, 0.0623,
    0.5188, 0.1577, 0.5127, 0.2482, 0.4946, 0.3328, 0.4648, 0.4106, 0.4236, 0.4106, 0.2796, 0.3332, 0.3354, 0.2512,
    0.3754, 0.1646, 0.3995, 0.0737, 0.4076, -0.0385, 0.3962, -0.1304, 0.362, -0.2019, 0.3051, -0.2531, 0.2256,
    -0.2838, 0.1238, -0.294, -0.0003, -0.2838, -0.1241, -0.2531, -0.2257, -0.2019, -0.3051, -0.1304, -0.362, -0.0385,
    -0.3962, 0.0737, -0.4076, 0.1415, -0.4045, 0.2016, -0.3952, 0.255, -0.3794,
    ],
  ],
  H: [
    [
    -0.3811, 0.5, -0.2458, 0.5, -0.2458, 0.0901, 0.2458, 0.0901, 0.2458, 0.5, 0.3811, 0.5, 0.3811, -0.5, 0.2458,
    -0.5, 0.2458, -0.0238, -0.2458, -0.0238, -0.2458, -0.5, -0.3811, -0.5,
    ],
  ],
  I: [
    [
    -0.0676, 0.5, 0.0676, 0.5, 0.0676, -0.5, -0.0676, -0.5,
    ],
  ],
} as const satisfies Record<string, readonly (readonly number[])[]>;

/** One letter as a liquid extrusion, its centre at the given cell coordinates. */
function letterExtrusion(letter: keyof typeof LETTER_OUTLINES, centreCells: readonly [number, number, number]): LiquidExtrusion {
  const h = NBFLIP_LETTERS_CELL_SIZE_M, cap = CAP_HEIGHT_CELLS * h;
  return {
    shape: "extrusion",
    contours_m: LETTER_OUTLINES[letter].map((contour) => contour.map((value, index) => centreCells[index % 2]! * h + value * cap)),
    centerZ_m: centreCells[2] * h,
    halfDepth_m: HALF_DEPTH_CELLS * h,
    offset_m: OUTLINE_OFFSET_CELLS * h,
    edgeRadius_m: EDGE_RADIUS_CELLS * h,
  };
}

/**
 * The nine letters on their schedule.
 *
 * A scheduled drop is added by the step that ends at its time, after that
 * step's advection: it leaves the step in place, carrying one step of gravity.
 * The footage holds a letter still for frames 10k and 10k + 1, so the letter
 * is given the time of frame 10k + 1 and its fall then matches frame for
 * frame; only the footage's first still frame is without it.
 */
export function nbflipLettersDrops(): ScheduledLiquidDrop[] {
  const [nx, ny, nz] = NBFLIP_LETTERS_GRID;
  return LETTER_CENTRES.map(([letter, x, y, z], index) => ({
    time_s: (NBFLIP_LETTERS_INTERVAL_FRAMES * index + 1) * NBFLIP_LETTERS_TIME_STEP_S,
    volume: letterExtrusion(letter, [(x - 0.5) * nx, y * ny, (z - 0.5) * nz]),
  }));
}

export const NBFLIP_LETTERS_SCENE_ID = "nbflip-figure-8-letters";

export const NBFLIP_LETTERS_METHOD_PROFILE = NBFLIP_METHOD_PROFILE;

export function createNbflipLetters(): SceneDescription {
  const scene = nbflipDomain(NBFLIP_LETTERS_SCENE_ID, NBFLIP_LETTERS_GRID, NBFLIP_LETTERS_CELL_SIZE_M, NBFLIP_LETTERS_FRAMES, NBFLIP_LETTERS_POOL_CELLS);
  scene.fluid.scheduledDrops = nbflipLettersDrops();
  return scene;
}

/**
 * The footage's camera, from the same pinhole fit that fixed the axes: 244
 * cells up and 469 in front of the domain's centre line, pitched down 19.4
 * degrees, with a 1445 px focal length on a 720 px frame.
 */
export function nbflipLettersCamera(): Partial<CameraState> {
  return footageCamera(NBFLIP_LETTERS_CELL_SIZE_M, 244.38, 469.35, 19.4, 1445.5);
}

/**
 * Figures 1 and 9, "Teaser": "a breaking dam hitting a row of cylinders".
 *
 *  - **Published**, in Table 1: the grid, 200 x 200 x 200.
 *  - **Measured**, from the single-view take that opens at video frame 340,
 *    with a pinhole camera fitted to thirteen edges of the pool and the dam.
 *    The round numbers below reproject onto those edges to 1.05 px rms; the
 *    same fit with every length free closes to 0.67 px and lands within a
 *    cell of each.
 *
 * What the fit gives, in cells from the back-left corner of the floor:
 *
 *  - **Pool.** 40 deep, a fifth of the grid (39.2 to 40.3 by three readings).
 *  - **Dam.** A block 60 x 80 x 60 standing on the pool in the back-left
 *    corner: three tenths of the grid across and deep, and up to six tenths.
 *    Fifty or sixty-seven across fit five to eight times worse.
 *  - **Cylinders.** Seven, of radius 5, from the floor to one cell above the
 *    dam's top. Their centres step 10 across and 25 toward the camera, from
 *    (50, 175) at the front to (110, 25) at the back. A radius of 4.5 or 5.5
 *    fits the silhouettes half again as badly.
 *  - **Scale.** The dam's top and two wall jets fall at 0.60 +- 0.04 cells
 *    per frame squared, which is mantaflow's 0.003 x 200.
 *  - **Start.** The take's first frame is not the release: the dam's top has
 *    already dropped a cell one frame later, as it would a frame and a half
 *    into its fall. This scene's frame n is the take's frame n - 1.5.
 *
 * Liquid never reaches the ceiling in the footage, so closing the top is the
 * solver's convention and not something the take shows.
 */

/** Table 1: "Fig. 9 / Teaser, 200^3". */
export const NBFLIP_TEASER_GRID = [200, 200, 200] as const;
export const NBFLIP_TEASER_CELL_SIZE_M = cellSizeForFall_m(mantaflowFall(NBFLIP_TEASER_GRID));
/** Rest depth of the pool, in cells. */
export const NBFLIP_TEASER_POOL_CELLS = 40;
/** The dam: its far corner from the back-left corner of the floor, in cells. It stands on the pool. */
export const NBFLIP_TEASER_DAM_CELLS = [60, 120, 60] as const;
export const NBFLIP_TEASER_CYLINDER_COUNT = 7;
export const NBFLIP_TEASER_CYLINDER_RADIUS_CELLS = 5;
export const NBFLIP_TEASER_CYLINDER_HEIGHT_CELLS = 121;
/** Frames from the take's first frame to the end of its fade. */
export const NBFLIP_TEASER_FRAMES = 248;
export const NBFLIP_TEASER_SCENE_ID = "nbflip-figure-9-teaser";

/** Centre of cylinder `index`, front to back, in cells across (x) and toward the camera (z). */
export const nbflipTeaserCylinderCentre = (index: number): readonly [x: number, z: number] => [50 + 10 * index, 175 - 25 * index];

export function createNbflipTeaser(): SceneDescription {
  const grid = NBFLIP_TEASER_GRID, h = NBFLIP_TEASER_CELL_SIZE_M, [damX, damY, damZ] = NBFLIP_TEASER_DAM_CELLS;
  const scene = nbflipDomain(NBFLIP_TEASER_SCENE_ID, grid, h, NBFLIP_TEASER_FRAMES, NBFLIP_TEASER_POOL_CELLS);
  scene.fluid.initialLiquidVolumes = [liquidBox(grid, h, [0, NBFLIP_TEASER_POOL_CELLS, 0], [damX, damY, damZ])];
  scene.solidVoxels = Array.from({ length: NBFLIP_TEASER_CYLINDER_COUNT }, (_, index) =>
    cylinderVoxels(...nbflipTeaserCylinderCentre(index), NBFLIP_TEASER_CYLINDER_RADIUS_CELLS, NBFLIP_TEASER_CYLINDER_HEIGHT_CELLS)).flat();
  return scene;
}

/**
 * The take's camera, static throughout: on the centre plane, 309 cells up and
 * 431 in front of the domain's centre, pitched down 29.4 degrees, with a
 * 1400 px focal length on the 720 px frame.
 */
export function nbflipTeaserCamera(): Partial<CameraState> {
  return footageCamera(NBFLIP_TEASER_CELL_SIZE_M, 308.8, 430.7, 29.4, 1400);
}
