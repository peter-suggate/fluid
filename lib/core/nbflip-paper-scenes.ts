import { cloneScene, defaultCamera, defaultScene, type CameraState, type InitialLiquidVolume, type SceneDescription, type ScheduledLiquidDrop } from "./model";
import { cameraPosition } from "./math";
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
 *  - **A frame is one solver frame**, of one to two steps. The step here is
 *    one frame. The long takes show every frame and the two small ones every
 *    second frame.
 *  - **Gravity is 0.003 of the largest grid dimension**, in cells per frame
 *    squared: mantaflow scales its `gravity = (0, -0.003, 0)` by that
 *    dimension. Each scene's fall was also measured, and is stated beside it.
 *    An inviscid solve has no other scale, so the cell size is whatever makes
 *    that fall terrestrial gravity at the rate the footage plays its frames.
 *  - **The box is closed and its walls are free-slip.**
 *  - **Shapes sit on round fractions of the grid.**
 *
 * mantaflow's outermost layer of cells is the wall, so its liquid has two
 * cells fewer on each axis than the published grid. In the large scenes the
 * footage cannot tell the two apart -- both fit the Teaser's frame to
 * 0.89 px -- and they take the published grid as the liquid's box: this
 * method's 4h tiles need every axis to be a multiple of four. The two small
 * scenes, where two cells are a sixteenth and a thirty-second of the box,
 * carry the wall layer as solid cells.
 *
 * They run on Uniform Narrow-band FLIP because that method is this paper's.
 * Its particle band is 4h where the mantaflow scenes used 3h.
 */

export const NBFLIP_FRAME_RATE_HZ = 24;
const GRAVITY_M_S2 = 9.81;
/** The cell at which a fall of `cellsPerFrame2` is terrestrial gravity. */
const cellSizeForFall_m = (cellsPerFrame2: number, framesPerSecond = NBFLIP_FRAME_RATE_HZ) => GRAVITY_M_S2 / (cellsPerFrame2 * framesPerSecond ** 2);
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
 * pool `poolCells` deep, stepped once a frame of `dt` for `frames` frames.
 * Surface tension is zeroed because the paper's solvers have none.
 *
 * `lattice` is the solver's grid where that has to be larger than the box:
 * the box then stands on the floor in the middle of it, and everything
 * outside the box is solid.
 */
function nbflipDomain(
  sceneId: string, grid: readonly [number, number, number], cellSize_m: number, frames: number, poolCells: number,
  lattice: readonly [number, number, number] = grid, dt = 1 / NBFLIP_FRAME_RATE_HZ, gravity_m_s2 = GRAVITY_M_S2,
): SceneDescription {
  const scene = cloneScene(defaultScene);
  const [nx, ny, nz] = lattice, h = cellSize_m;
  const [marginX, marginZ] = latticeMargin(grid, lattice);
  scene.sceneId = sceneId;
  scene.randomSeed = 2016;
  scene.duration_s = frames * dt;
  scene.solidVoxels = ([
    [[0, 0, 0], [marginX, ny, nz]], [[nx - marginX, 0, 0], [nx, ny, nz]],
    [[0, 0, 0], [nx, ny, marginZ]], [[0, 0, nz - marginZ], [nx, ny, nz]],
    [[0, grid[1], 0], [nx, ny, nz]],
  ] as const).filter(([minimum, maximumExclusive]) => minimum.every((value, axis) => value < maximumExclusive[axis]!))
    .map(([minimum, maximumExclusive]) => ({ operation: "fill", minimum, maximumExclusive, materialId: VOXEL_MATERIAL_IDS.container }));
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
    gravity_m_s2: { x: 0, y: -gravity_m_s2, z: 0 },
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

/** Cells of solid on either side of a box centred in a larger lattice, across and deep. */
function latticeMargin(grid: readonly [number, number, number], lattice: readonly [number, number, number]): readonly [x: number, z: number] {
  return [(lattice[0] - grid[0]) / 2, (lattice[2] - grid[2]) / 2];
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

/** Where a footage camera stands. */
const footageEye = (camera: Partial<CameraState>) => cameraPosition({ ...defaultCamera, ...camera });

/** A box of liquid between two corners given in cells from the domain's minimum corner. */
function liquidBox(grid: readonly [number, number, number], cellSize_m: number, min: readonly [number, number, number], max: readonly [number, number, number]): InitialLiquidVolume {
  return { shape: "box", min_m: cellPoint(grid, cellSize_m, min), max_m: cellPoint(grid, cellSize_m, max) };
}

/** A point given in cells from the domain's minimum corner. */
const cellPoint = (grid: readonly [number, number, number], cellSize_m: number, cells: readonly [number, number, number]) =>
  ({ x: (cells[0] - grid[0] / 2) * cellSize_m, y: cells[1] * cellSize_m, z: (cells[2] - grid[2] / 2) * cellSize_m });

/** mantaflow's `initDomain(boundaryWidth=0)`: the outermost layer of cells, on all six sides, is wall. */
function wallLayer([nx, ny, nz]: readonly [number, number, number]): SolidWorldVoxelPatch[] {
  return ([
    [[0, 0, 0], [1, ny, nz]], [[nx - 1, 0, 0], [nx, ny, nz]], [[0, 0, 0], [nx, 1, nz]],
    [[0, ny - 1, 0], [nx, ny, nz]], [[0, 0, 0], [nx, ny, 1]], [[0, 0, nz - 1], [nx, ny, nz]],
  ] as const).map(([minimum, maximumExclusive]) => ({ operation: "fill", minimum, maximumExclusive, materialId: VOXEL_MATERIAL_IDS.container }));
}

/**
 * An upright solid cylinder standing on the floor, as voxel runs: every cell
 * whose centre it covers, one run per row of z. In lattice cells.
 */
function cylinderVoxels(centreX: number, centreZ: number, radius: number, top: number): SolidWorldVoxelPatch[] {
  const runs: SolidWorldVoxelPatch[] = [], height = Math.round(top);
  for (let z = Math.floor(centreZ - radius); z < centreZ + radius; z++) {
    const half = Math.sqrt(radius ** 2 - (z + 0.5 - centreZ) ** 2);
    const from = Math.ceil(centreX - half - 0.5), to = Math.floor(centreX + half - 0.5) + 1;
    if (to > from) runs.push({ operation: "fill", minimum: [from, 0, z], maximumExclusive: [to, height, z + 1], materialId: VOXEL_MATERIAL_IDS.cylinder });
  }
  return runs;
}

/**
 * An upright tube standing on the floor, as voxel runs: every cell whose
 * centre lies between its two radii. In lattice cells.
 */
function tubeVoxels(centreX: number, centreZ: number, inner: number, outer: number, top: number): SolidWorldVoxelPatch[] {
  const runs: SolidWorldVoxelPatch[] = [], height = Math.round(top);
  const chord = (radius: number, z: number): readonly [from: number, to: number] => {
    const half = Math.sqrt(Math.max(0, radius ** 2 - (z + 0.5 - centreZ) ** 2));
    return [Math.ceil(centreX - half - 0.5), Math.floor(centreX + half - 0.5) + 1];
  };
  for (let z = Math.floor(centreZ - outer); z < centreZ + outer; z++) {
    const [from, to] = chord(outer, z), [hollowFrom, hollowTo] = Math.abs(z + 0.5 - centreZ) < inner ? chord(inner, z) : [to, to];
    for (const [minimum, maximumExclusive] of [[from, hollowFrom], [hollowTo, to]] as const) {
      if (maximumExclusive > minimum) runs.push({ operation: "fill", minimum: [minimum, 0, z], maximumExclusive: [maximumExclusive, height, z + 1], materialId: VOXEL_MATERIAL_IDS.container });
    }
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
/**
 * The solver's grid. Its pressure hierarchy halves the 4h lattice until that
 * is a few cells across, which 200 = 8 x 25 does not allow and 224 = 32 x 7
 * does: the published box stands in the middle of this one, behind solid.
 */
export const NBFLIP_TEASER_LATTICE = [224, 224, 224] as const;
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
  const scene = nbflipDomain(NBFLIP_TEASER_SCENE_ID, grid, h, NBFLIP_TEASER_FRAMES, NBFLIP_TEASER_POOL_CELLS, NBFLIP_TEASER_LATTICE);
  const [marginX, marginZ] = latticeMargin(grid, NBFLIP_TEASER_LATTICE);
  scene.fluid.initialLiquidVolumes = [liquidBox(grid, h, [0, NBFLIP_TEASER_POOL_CELLS, 0], [damX, damY, damZ])];
  for (let index = 0; index < NBFLIP_TEASER_CYLINDER_COUNT; index++) {
    const [x, z] = nbflipTeaserCylinderCentre(index);
    scene.solidVoxels.push(...cylinderVoxels(x + marginX, z + marginZ, NBFLIP_TEASER_CYLINDER_RADIUS_CELLS, NBFLIP_TEASER_CYLINDER_HEIGHT_CELLS));
  }
  scene.cutaway = { eye_m: footageEye(nbflipTeaserCamera()) };
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

/**
 * Figure 10, "Dam": a breaking dam in a channel of cylinders.
 *
 * This one ran in a different solver -- "an advanced FLIP implementation",
 * Ando et al. 2012 -- so nothing of mantaflow's carries over, and its band is
 * the 4h this method uses.
 *
 *  - **Published**, in Table 1: the grid, 256 x 128 x 64.
 *  - **Measured**, from the four takes that open at video frame 2719, with a
 *    pinhole camera fitted to the corners of the block and the sheet. The
 *    round numbers below reproject onto eleven of them to 0.86 px rms.
 *
 * What the fit gives, in cells from the back-left corner of the floor:
 *
 *  - **Axes.** 256 along, 128 up and 64 deep. With the depth free the fit
 *    gives 64.5; 128 deep and 64 tall misses by 12 px, and the block alone is
 *    taller than 64.
 *  - **Liquid.** Fractions of the 256, not of each axis: a block two tenths
 *    long (51.1 to 51.3 measured) and three tenths tall (76.4 to 77.1)
 *    against the left wall, over a sheet six hundredths deep (15.2 to 15.5).
 *    Both span the channel.
 *  - **Cylinders.** Eight, of radius 2.5, as tall as the block, in rows of
 *    three, two and three: at four, six and eight tenths of the length in the
 *    rows three and seven tenths of the way back, and at five and seven
 *    tenths in the row between. Each measures within 0.7 cells of that.
 *  - **Frames.** The two side-by-side takes show 362 frames, one to a video
 *    frame; the single-view takes show every second one.
 *  - **Scale.** The block's top and the plume off the end wall fall at
 *    0.174 +- 0.008 cells per frame squared. That is a metre-long channel
 *    under 9.8 m/s^2 at 120 frames a second, to four figures, and the scene
 *    is authored as that. mantaflow's rule would give 0.768 per video frame
 *    of the single-view takes, which the fall rules out at 0.70 +- 0.03.
 *
 * Whether there is a ceiling the footage does not show: the plume leaves the
 * top of the frame at the end wall. It is closed here, at the grid's top.
 */

/** Table 1: "Fig. 10 / Dam, 256 x 128 x 64". Length, height, depth. */
export const NBFLIP_DAM_GRID = [256, 128, 64] as const;
/** A metre of channel. */
export const NBFLIP_DAM_CELL_SIZE_M = 1 / NBFLIP_DAM_GRID[0];
export const NBFLIP_DAM_FRAME_RATE_HZ = 120;
export const NBFLIP_DAM_GRAVITY_M_S2 = 9.8;
/** Depth of the sheet, the block's length and its height, as fractions of the channel's length. */
export const NBFLIP_DAM_SHEET = 0.06;
export const NBFLIP_DAM_BLOCK = [0.2, 0.3] as const;
/** Each cylinder's centre, as fractions of the length and of the depth. */
export const NBFLIP_DAM_CYLINDERS: readonly (readonly [x: number, z: number])[] = [
  [0.4, 0.3], [0.6, 0.3], [0.8, 0.3], [0.5, 0.5], [0.7, 0.5], [0.4, 0.7], [0.6, 0.7], [0.8, 0.7],
];
export const NBFLIP_DAM_CYLINDER_RADIUS_CELLS = 2.5;
export const NBFLIP_DAM_FRAMES = 362;
export const NBFLIP_DAM_SCENE_ID = "nbflip-figure-10-dam";

export function createNbflipDam(): SceneDescription {
  const grid = NBFLIP_DAM_GRID, [nx, , nz] = grid, h = NBFLIP_DAM_CELL_SIZE_M;
  const sheet = NBFLIP_DAM_SHEET * nx, length = NBFLIP_DAM_BLOCK[0] * nx, top = NBFLIP_DAM_BLOCK[1] * nx;
  const scene = nbflipDomain(NBFLIP_DAM_SCENE_ID, grid, h, NBFLIP_DAM_FRAMES, sheet, grid, 1 / NBFLIP_DAM_FRAME_RATE_HZ, NBFLIP_DAM_GRAVITY_M_S2);
  scene.fluid.initialLiquidVolumes = [liquidBox(grid, h, [0, sheet, 0], [length, top, nz])];
  scene.solidVoxels = NBFLIP_DAM_CYLINDERS.flatMap(([x, z]) => cylinderVoxels(x * nx, z * nz, NBFLIP_DAM_CYLINDER_RADIUS_CELLS, top));
  return scene;
}

/**
 * The takes' camera, static throughout: on the centre plane, 384 cells up and
 * 416 in front of the channel's centre line -- one and a half and one and
 * three quarter channel lengths from the back-left corner of the floor --
 * pitched down 40.6 degrees at that corner's edge, with a 2388 px focal
 * length on the 720 px frame.
 */
export function nbflipDamCamera(): Partial<CameraState> {
  return footageCamera(NBFLIP_DAM_CELL_SIZE_M, 384, 416, 40.63, 2388.5);
}

/**
 * Figure 7, "Pour": a stream from an inclined pipe fills a glass.
 *
 *  - **Published**, in Table 1: the grid, 128 x 128 x 256.
 *  - **Measured**, from the single-view take whose liquid first moves at video
 *    frame 2073, with a pinhole camera fitted to the glass's rim, its floor
 *    and the cell faces that show as stripes in the take's FLIP half.
 *
 * What the fit gives, in cells from the back-left corner of the floor:
 *
 *  - **Axes.** 256 is up.
 *  - **Glass.** The obstacle shows through the liquid as a circle of radius
 *    50 to 51.6 on the box's axis: four tenths of the grid. The glass drawn
 *    around it has a wall 2.6 thick and a rim 189.8 +- 1.6 above its floor,
 *    three quarters of the grid.
 *  - **Pipe.** Drawn, and not known to be simulated, so it is not here. Its
 *    axis falls three in five (30.9 +- 0.4 degrees below level) on the plane
 *    through the glass's axis; its bore is 19.0 +- 0.5 in radius, 0.15 of the
 *    grid, and its mouth is centred 214.5 above the glass's floor and 33 short
 *    of the glass's axis.
 *  - **Stream.** Liquid travels along that axis at 4.1 +- 0.1 cells per frame
 *    and falls freely from a plane 5.5 +- 1.5 beyond the mouth: (3.5, -2.1) is
 *    seven tenths of (5, -3). It crosses that plane for 126 +- 1.5 frames.
 *  - **Scale.** The stream's shape and its head's timing give a fall of
 *    0.77 +- 0.03 cells per frame squared; mantaflow's is 0.003 x 256.
 *  - **Start.** The take's first frame already shows 12 to 14 cells of liquid
 *    past the mouth, 1.8 frames of this stream. This scene's frame n is the
 *    take's frame n - 1.8.
 *
 * Two things the footage does not settle. The glass's floor may stand up to
 * 25 cells above the box's; it is put on it. And the glass in the footage
 * fills to 147 cells, 2.07 times what this stream carries into it (71): the
 * stream is authored as measured, not scaled to reach that level.
 */

/** Table 1: "Fig. 7 / Pour, 128^2 x 256", with the long axis up. */
export const NBFLIP_POUR_GRID = [128, 256, 128] as const;
export const NBFLIP_POUR_CELL_SIZE_M = cellSizeForFall_m(mantaflowFall(NBFLIP_POUR_GRID));
/** Inner radius of the glass, as a fraction of the grid's width. */
export const NBFLIP_POUR_GLASS_RADIUS = 0.4;
/** Thickness of the glass's wall, in cells. */
export const NBFLIP_POUR_GLASS_WALL_CELLS = 2.6;
/** Height of the glass's rim, as a fraction of the grid's height. */
export const NBFLIP_POUR_GLASS_RIM = 0.75;
/** Radius of the stream, as a fraction of the grid's width. */
export const NBFLIP_POUR_STREAM_RADIUS = 0.15;
/** Centre of the plane the stream falls freely from, in cells across and up. */
export const NBFLIP_POUR_RELEASE_CELLS = [31 + 5.5 * 5 / Math.sqrt(34), 214.5 - 5.5 * 3 / Math.sqrt(34)] as const;
/** The stream's velocity there, in cells per frame across and up. */
export const NBFLIP_POUR_STREAM_CELLS_PER_FRAME = [3.5, -2.1] as const;
/** Frames the stream runs for. */
export const NBFLIP_POUR_STREAM_FRAMES = 126;
/** Frames from the stream's start to the end of the take's fade. */
export const NBFLIP_POUR_FRAMES = 250;
export const NBFLIP_POUR_SCENE_ID = "nbflip-figure-7-pour";

export function createNbflipPour(): SceneDescription {
  const grid = NBFLIP_POUR_GRID, [nx, ny, nz] = grid, h = NBFLIP_POUR_CELL_SIZE_M, dt = 1 / NBFLIP_FRAME_RATE_HZ;
  const scene = nbflipDomain(NBFLIP_POUR_SCENE_ID, grid, h, NBFLIP_POUR_FRAMES, 0);
  const inner = NBFLIP_POUR_GLASS_RADIUS * nx;
  scene.solidVoxels.push(...tubeVoxels(nx / 2, nz / 2, inner, inner + NBFLIP_POUR_GLASS_WALL_CELLS, NBFLIP_POUR_GLASS_RIM * ny));
  // The footage draws the glass cut open toward its camera.
  scene.cutaway = { eye_m: footageEye(nbflipPourCamera()) };
  const [x, y] = NBFLIP_POUR_RELEASE_CELLS, [u, v] = NBFLIP_POUR_STREAM_CELLS_PER_FRAME;
  scene.fluid.inflow = {
    center_m: cellPoint(grid, h, [x, y, nz / 2]),
    radius_m: NBFLIP_POUR_STREAM_RADIUS * nx * h,
    length_m: Math.hypot(u, v) * h,
    velocity_m_s: { x: u * h / dt, y: v * h / dt, z: 0 },
    start_s: 0,
    end_s: NBFLIP_POUR_STREAM_FRAMES * dt,
    ramp_s: 0,
  };
  return scene;
}

/**
 * The take's camera, static throughout: on the centre plane, 274 cells above
 * the glass's floor and 511 in front of its axis, pitched down 17.6 degrees,
 * with a 1403 px focal length on the 720 px frame.
 */
export function nbflipPourCamera(): Partial<CameraState> {
  return footageCamera(NBFLIP_POUR_CELL_SIZE_M, 273.8, 510.6, 17.6, 1403);
}

/**
 * The video's "Coupling Comparison": the two small scenes that set the
 * paper's velocity blend (Eq. 3) beside the naive one (Eq. 2) and regular
 * FLIP. Both are mantaflow scenes in the mould of `flip05_nbflip.py`, and
 * both play two solver frames to a video frame: 250 video frames from the
 * held still to black, against the 500 frames of Figure 4's energy plot.
 * Their step here is a forty-eighth of a second, so they run at the
 * footage's speed.
 *
 * The outermost layer of cells is wall, as mantaflow has it, so the liquid's
 * box is 30 and 62 cells across. Heights below count from the bottom of the
 * grid, a cell under the floor.
 */
export const NBFLIP_COUPLING_FRAME_RATE_HZ = 2 * NBFLIP_FRAME_RATE_HZ;
export const NBFLIP_COUPLING_FRAMES = 500;

/**
 * Figure 4, "Oscillating Surface": a mound on a pool, left to slosh.
 *
 *  - **Published**: "a 32^3 simulation of surface waves", four frames of it
 *    (Fig. 4) and its kinetic energy over 500 frames.
 *  - **Measured**, from the take that opens at video frame 1216, with a
 *    pinhole camera fitted to six corners of the pool to 0.47 px rms on the
 *    360 px sub-view.
 *
 *  - **Frames.** The figure's "Frame 170" is video frame 1216 + 85 in all
 *    three sub-views (cross-correlation 0.96, against 0.87 to 0.91 at
 *    1216 + 170).
 *  - **Pool.** Its depth is 0.290 of its width: 8.7 cells over the floor,
 *    where three tenths of the grid is 8.6.
 *  - **Mound.** A sphere centred in plan with its top at 15.1 +- 0.15 and its
 *    centre about two cells under the surface. A quarter of the grid up and
 *    0.225 of it in radius puts the top at 15.2 and fits the outline to
 *    0.69 px; the unconstrained fit, 7.4 up and 7.7 in radius, reaches
 *    0.54 px, and the footage does not separate the two. A hemisphere
 *    sitting on the surface is ruled out at 2.1 px.
 *  - **Scale.** mantaflow's rule gives 0.096 cells per frame squared. The
 *    slosh is the only clock in the take and it runs slow of that: the
 *    corners rise every 49.3 to 49.7 frames where linear waves under 0.096
 *    would take 45.5, which is a fall of 0.076 to 0.088. The other small
 *    scene confirms the rule ballistically, so the rule stands here and the
 *    period is the number a run is checked against.
 *
 * The liquid never comes within 18 cells of the ceiling.
 */

/** Sec. 3.1: "a 32^3 simulation". */
export const NBFLIP_WAVES_GRID = [32, 32, 32] as const;
/** 0.096 cells per frame squared. */
export const NBFLIP_WAVES_CELL_SIZE_M = cellSizeForFall_m(mantaflowFall(NBFLIP_WAVES_GRID), NBFLIP_COUPLING_FRAME_RATE_HZ);
/** The pool's surface, and the mound's centre height and radius, as fractions of the grid. */
export const NBFLIP_WAVES_POOL = 0.3;
export const NBFLIP_WAVES_MOUND = [0.25, 0.225] as const;
export const NBFLIP_WAVES_SCENE_ID = "nbflip-figure-4-oscillating-surface";

export function createNbflipWaves(): SceneDescription {
  const grid = NBFLIP_WAVES_GRID, [n] = grid, h = NBFLIP_WAVES_CELL_SIZE_M, [centre, radius] = NBFLIP_WAVES_MOUND;
  const scene = nbflipDomain(NBFLIP_WAVES_SCENE_ID, grid, h, NBFLIP_COUPLING_FRAMES, NBFLIP_WAVES_POOL * n, grid, 1 / NBFLIP_COUPLING_FRAME_RATE_HZ);
  scene.solidVoxels = wallLayer(grid);
  scene.cutaway = { eye_m: footageEye(nbflipWavesCamera()) };
  scene.fluid.initialLiquidVolumes = [{ shape: "sphere", center_m: cellPoint(grid, h, [n / 2, centre * n, n / 2]), radius_m: radius * n * h }];
  return scene;
}

/**
 * The sub-views' camera, static: on the centre plane, 49 cells up and 66.7 in
 * front of the pool's centre, pitched down 30.2 degrees, with a 1354 px focal
 * length on the 720 px frame (677 px on a sub-view).
 */
export function nbflipWavesCamera(): Partial<CameraState> {
  return footageCamera(NBFLIP_WAVES_CELL_SIZE_M, 48.98, 66.71, 30.23, 1354);
}

/**
 * "Simple Breaking Dam", 64^3: in the video only, between Figures 4 and 7.
 *
 *  - **Published**: nothing but the title card's grid.
 *  - **Measured**, from the take that opens at video frame 1662, with a
 *    pinhole camera fitted to seven corners of the block and the basin to
 *    0.22 px rms on the 360 px sub-view.
 *
 *  - **Liquid.** A block against the left wall and the wall nearest the
 *    camera, over a basin. Its free faces measure 39.0 across, 31.5 up and
 *    18.1 from the far wall, and the basin 6.5 deep: six tenths, a half and
 *    three tenths of the grid, and a tenth, to 0.6, 0.5, 1.1 and 0.1 cells
 *    (1.1 px rms). It is not `flip05_nbflip.py`'s dam, whose fractions miss
 *    by 22 px, and a block spanning the depth misses by 6.7 px.
 *  - **Scale.** The block's free top edge and the last liquid to leave the
 *    ceiling both fall at 0.76 +- 0.06 cells per video frame squared: 0.19
 *    per solver frame at two to a video frame, for mantaflow's 0.192.
 *  - **Ceiling.** Closed, at the grid's top: the sheet that runs up the
 *    right wall stops on the box's projected top edge 26 to 28 frames in and
 *    hangs there for twelve.
 */

export const NBFLIP_SIMPLE_DAM_GRID = [64, 64, 64] as const;
/** 0.192 cells per frame squared. */
export const NBFLIP_SIMPLE_DAM_CELL_SIZE_M = cellSizeForFall_m(mantaflowFall(NBFLIP_SIMPLE_DAM_GRID), NBFLIP_COUPLING_FRAME_RATE_HZ);
/** The basin's surface, and the block's far corner and near one, as fractions of the grid. */
export const NBFLIP_SIMPLE_DAM_BASIN = 0.1;
export const NBFLIP_SIMPLE_DAM_BLOCK = [[0, 0.1, 0.3], [0.6, 0.5, 1]] as const;
export const NBFLIP_SIMPLE_DAM_SCENE_ID = "nbflip-simple-breaking-dam";

export function createNbflipSimpleDam(): SceneDescription {
  const grid = NBFLIP_SIMPLE_DAM_GRID, [n] = grid, h = NBFLIP_SIMPLE_DAM_CELL_SIZE_M;
  const scene = nbflipDomain(NBFLIP_SIMPLE_DAM_SCENE_ID, grid, h, NBFLIP_COUPLING_FRAMES, NBFLIP_SIMPLE_DAM_BASIN * n, grid, 1 / NBFLIP_COUPLING_FRAME_RATE_HZ);
  const cells = (corner: readonly [number, number, number]) => [corner[0] * n, corner[1] * n, corner[2] * n] as const;
  scene.solidVoxels = wallLayer(grid);
  scene.cutaway = { eye_m: footageEye(nbflipSimpleDamCamera()) };
  scene.fluid.initialLiquidVolumes = [liquidBox(grid, h, cells(NBFLIP_SIMPLE_DAM_BLOCK[0]), cells(NBFLIP_SIMPLE_DAM_BLOCK[1]))];
  return scene;
}

/**
 * The sub-views' camera, static: on the centre plane, 98 cells up and 138.7
 * in front of the box's centre, pitched down 29.0 degrees, with a 1401 px
 * focal length on the 720 px frame.
 */
export function nbflipSimpleDamCamera(): Partial<CameraState> {
  return footageCamera(NBFLIP_SIMPLE_DAM_CELL_SIZE_M, 98, 138.72, 29.03, 1401);
}
