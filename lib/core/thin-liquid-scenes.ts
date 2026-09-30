import { cloneScene, defaultScene, type InitialLiquidVolume, type SceneDescription } from "./model";
import { studioStageSceneryGraph } from "./studio-stage-scene";

/**
 * Thin-liquid ladders: splash-sized features on the CM12 lattice.
 *
 * Figure 9's splashes -- drops, sheets and wall films one or two cells thick --
 * lose their level set while their conservative volume survives, so they
 * vanish from the render. These scenes isolate that on features whose exact
 * motion is known: every body starts at rest in free fall, so until it reaches
 * the pool it should keep its shape and fall as y0 - g t^2 / 2. Anything else
 * before impact is numerical. Each ladder varies one size across otherwise
 * identical bodies, spaced so a per-body census can bin them by position.
 *
 * Shared with Figure 9: dx = 0.05 m, dt = 1/30 s, g = 10 m/s^2, closed top,
 * free-slip walls, no surface tension. A 6-cell pool catches every body, so
 * impacts make crowns and secondary drops into liquid rather than a sub-cell
 * film on a dry floor (corner-brick-drop covers that one).
 *
 * All three share one 64x64x48-cell grid, so a probe running them on one
 * device compiles the solver once. Sizes are in cells. The layout constants
 * are exported for probes. Sheet and
 * film faces lie on cell faces: the initial V samples each cell at 8 points
 * (+-0.4 cell), so a face inside a cell rounds V to a half cell while phi keeps
 * the exact box, and the two fields would disagree before the first step.
 */
export const THIN_LIQUID_CELL_M = 0.05;
export const THIN_LIQUID_POOL_CELLS = 6;
export const THIN_LIQUID_GRID_CELLS = Object.freeze([64, 64, 48] as const);
/** Radii of the droplet ladder, one drop per 16-cell x bin. */
export const THIN_DROPLET_RADII_CELLS = Object.freeze([1, 1.5, 2, 3]);
export const THIN_DROPLET_BIN_CELLS = 16;
/** Whole-cell thicknesses of the sheet and film ladders, one body per bin. */
export const THIN_SHEET_THICKNESS_CELLS = Object.freeze([1, 2, 3, 4]);
export const THIN_SHEET_BIN_CELLS = 16;
export const THIN_FILM_BIN_CELLS = 12;
/** Release height of every body's centre, in cells above the floor. */
export const THIN_RELEASE_HEIGHT_CELLS = 40;

function domain(id: string): SceneDescription {
  const cells = THIN_LIQUID_GRID_CELLS;
  const scene = cloneScene(defaultScene);
  const h = THIN_LIQUID_CELL_M;
  scene.solidVoxels = [];
  scene.sceneId = id;
  scene.randomSeed = 2012;
  scene.duration_s = 3;
  scene.container = {
    ...scene.container,
    width_m: cells[0] * h, height_m: cells[1] * h, depth_m: cells[2] * h,
    fillFraction: THIN_LIQUID_POOL_CELLS / cells[1],
    top: "closed", fluidWallMode: "free-slip",
  };
  scene.voxelDomain = { ...scene.voxelDomain, finestCellSize_m: h };
  scene.nominalResolution = { length_m: h };
  scene.numerics = { ...scene.numerics, fixedDt_s: 1 / 30, maxDt_s: 1 / 30 };
  scene.fluid = { ...scene.fluid, gravity_m_s2: { x: 0, y: -10, z: 0 }, surfaceTension_N_m: 0, initialCondition: "tank-fill" };
  delete scene.fluid.initialDamBreakDimensions_m;
  delete scene.fluid.initialDamBreakOrigin_m;
  delete scene.fluid.initialBrickSeeds_m;
  delete scene.fluid.initialBrickSeedsAdditive;
  delete scene.fluid.inflow;
  delete scene.terrain;
  scene.rigidBodies = [];
  return scene;
}

/** Container-centred metres of a cell coordinate measured from the -x/floor/-z corner. */
function at(scene: SceneDescription, x: number, y: number, z: number) {
  const c = scene.container, h = THIN_LIQUID_CELL_M;
  return { x: x * h - 0.5 * c.width_m, y: y * h, z: z * h - 0.5 * c.depth_m };
}

function box(scene: SceneDescription, min: readonly number[], max: readonly number[]): InitialLiquidVolume {
  return { shape: "box", min_m: at(scene, min[0]!, min[1]!, min[2]!), max_m: at(scene, max[0]!, max[1]!, max[2]!) };
}

function withStage(scene: SceneDescription): SceneDescription {
  scene.scenery = studioStageSceneryGraph(scene);
  return scene;
}

/**
 * Four drops of radius 1 to 3 cells, one per 16-cell x bin, released together.
 * The smallest is below what a vertex level set can hold for a single step of
 * resampling; the largest is a Figure 9 splash drop.
 */
export function createThinDropletLadderScene(): SceneDescription {
  const w = THIN_DROPLET_BIN_CELLS;
  const scene = domain("thin-droplet-ladder");
  scene.fluid.initialLiquidVolumes = THIN_DROPLET_RADII_CELLS.map((radius, i) => ({
    shape: "sphere" as const,
    center_m: at(scene, (i + 0.5) * w, THIN_RELEASE_HEIGHT_CELLS, 24),
    radius_m: radius * THIN_LIQUID_CELL_M,
  }));
  return withStage(scene);
}

/**
 * Horizontal 12x12-cell sheets 1 to 4 cells thick, one per 16-cell x bin,
 * falling face-down: every step resamples them across vertex planes, the
 * motion that thins a Figure 9 crown sheet.
 */
export function createThinSheetLadderScene(): SceneDescription {
  const w = THIN_SHEET_BIN_CELLS;
  const scene = domain("thin-sheet-ladder");
  const y = THIN_RELEASE_HEIGHT_CELLS;
  scene.fluid.initialLiquidVolumes = THIN_SHEET_THICKNESS_CELLS.map((t, i) =>
    box(scene, [i * w + 2, y, 18], [i * w + 14, y + t, 30]));
  return withStage(scene);
}

/**
 * Vertical films 1 to 4 cells thick, 30 cells tall and 10 wide, one per
 * 12-cell z bin: each lies against the -x wall and has a free-standing twin
 * from x = 36, so a 32-cell x bin separates them. Both should fall unchanged
 * (free-slip, gravity along the film); any difference between the twins is
 * the wall's.
 */
export function createThinWallFilmScene(): SceneDescription {
  const w = THIN_FILM_BIN_CELLS;
  const scene = domain("thin-wall-films");
  const top = THIN_RELEASE_HEIGHT_CELLS + 15, bottom = THIN_RELEASE_HEIGHT_CELLS - 15;
  scene.fluid.initialLiquidVolumes = THIN_SHEET_THICKNESS_CELLS.flatMap((t, i) => {
    const z0 = i * w + 1, z1 = z0 + 10;
    // The wall film's box runs through the wall: phi on the wall plane is
    // then the liquid's continuation (-t), not 0 on the box face.
    return [box(scene, [-t, bottom, z0], [t, top, z1]), box(scene, [36, bottom, z0], [36 + t, top, z1])];
  });
  return withStage(scene);
}
