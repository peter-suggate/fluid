import { createInteractivePlayground } from "../interaction/playground";
import type { SceneRenderProfile } from "./scene-render-profile";
import { cloneScene, defaultCamera, defaultScene, type CameraState, type SceneDescription } from "./model";
import { applyHeroGardenNodeOverrides } from "./hero-garden-overrides";
import { createMassConservingFigure9DamBreak, createPaperScenario } from "./paper-scenarios";
import {
  NBFLIP_LETTERS_GRID, NBFLIP_LETTERS_INTERVAL_FRAMES, NBFLIP_LETTERS_METHOD_PROFILE, NBFLIP_LETTERS_SCENE_ID, NBFLIP_METHOD_PROFILE,
  NBFLIP_DAM_CYLINDERS, NBFLIP_DAM_GRID, NBFLIP_DAM_SCENE_ID, NBFLIP_SIMPLE_DAM_GRID, NBFLIP_SIMPLE_DAM_SCENE_ID,
  NBFLIP_POUR_GRID, NBFLIP_POUR_SCENE_ID, NBFLIP_POUR_STREAM_FRAMES,
  NBFLIP_TEASER_CYLINDER_COUNT, NBFLIP_TEASER_GRID, NBFLIP_TEASER_SCENE_ID, NBFLIP_WAVES_GRID, NBFLIP_WAVES_SCENE_ID,
  createNbflipDam, createNbflipLetters, createNbflipPour, createNbflipPourPreview, createNbflipSimpleDam, createNbflipTeaser, createNbflipWaves,
  nbflipDamCamera, nbflipLettersCamera, nbflipPourCamera, nbflipSimpleDamCamera, nbflipTeaserCamera, nbflipWavesCamera,
} from "./nbflip-paper-scenes";
import {
  NBFLIP_SLIDE_GRID, NBFLIP_SLIDE_LONG_BENDS, NBFLIP_SLIDE_LONG_GRID, NBFLIP_SLIDE_LONG_SCENE_ID, NBFLIP_SLIDE_LONG_STREAM_SECONDS,
  NBFLIP_SLIDE_SCENE_ID, NBFLIP_SLIDE_STREAM_SECONDS, NBFLIP_SLIDE_TURNS,
  createNbflipWaterSlide, createNbflipWaterSlideLong, createNbflipWaterSlidePreview, createNbflipWaterSlideLongPreview, nbflipWaterSlideCamera, nbflipWaterSlideLongCamera,
} from "./nbflip-water-slide-scene";
import {
  NBFLIP_THEME_PARK_GRID, NBFLIP_THEME_PARK_SCENE_ID, NBFLIP_THEME_PARK_STREAM_SECONDS,
  createNbflipThemeParkSlide, nbflipThemeParkCamera,
} from "./nbflip-theme-park-slide-scene";
import {
  NBFLIP_RAMP_BASIN_LEVEL, NBFLIP_RAMP_DRAIN_SECONDS, NBFLIP_RAMP_GRID, NBFLIP_RAMP_LENGTH, NBFLIP_RAMP_VARIANTS,
  createNbflipRamp, nbflipRampBed, nbflipRampCamera, nbflipRampSceneId, nbflipRampShape,
} from "./nbflip-ramp-scene";
import { NBFLIP_SLOPE_CLEARANCE_SCENE_ID, createNbflipSlopeClearance } from "./nbflip-slope-clearance-scene";
import {
  NBFLIP_OPEN_GROUND_CELL_SIZE_M, NBFLIP_OPEN_GROUND_COLUMN, NBFLIP_OPEN_GROUND_MEMORY_MIB, NBFLIP_OPEN_GROUND_SCENE_ID,
  NBFLIP_OPEN_GROUND_FACES_SCENE_ID, NBFLIP_OPEN_GROUND_SLOPE_SCENE_ID, NBFLIP_OPEN_GROUND_SIDE, NBFLIP_OPEN_GROUND_STEP_HZ,
  createNbflipOpenGround, nbflipOpenGroundCamera,
} from "./nbflip-open-ground-scene";
import { CM12_FIGURES, CM12_SLAB_DEPTH_CELLS, cm12Camera, cm12Grid, cm12Scene, createCm12Figure7_256 } from "./cm12-paper-scenes";
import { createThinDropletLadderScene, createThinSheetLadderScene, createThinWallFilmScene } from "./thin-liquid-scenes";
import { applyGardenPool, GARDEN_DAM_BRICK_SEED_M, GARDEN_WATERLINE_M, gardenPoolTerrain } from "./garden-scene";
import {
  createHeroGardenHoseScene,
  HERO_GARDEN_POND_CELL_M,
  heroGardenCamera,
  type HeroGardenHoseOptions,
} from "./hero-garden-scene";
import {
  createHeroGardenHoseStressScene,
  HERO_GARDEN_STRESS_MAXIMUM_MULTIPLIER,
} from "./hero-garden-stress-scene";
import { studioStageCamera, studioStageSceneryGraph } from "./studio-stage-scene";
import { createAnalyticMotionScene, createRerungFreeFallScene, createStandingWaveScene } from "./analytic-motion-scenes";
import { createGeometricUniformTranslationScene } from "./geometric-translation-scene";
import { createGentleMovingBlobScene } from "./gentle-moving-blob-scene";
import { CHURNING_TANK_SCENE_ID, createChurningTankScene } from "./churning-tank-scene";
import { NBFLIP_STAIR_CUBE_ID, createNbflipStairCube, nbflipStairCubeCamera } from "./nbflip-stair-cube-scene";
import { createStationaryBowlScene } from "./stationary-bowl-scene";
import { HERO_GARDEN_BLENDER_ID, createHeroGardenBlenderScene, heroGardenBlenderCamera } from "./hero-garden-blender-scene";
import { BACKYARD_POOL_DRIFT_ID, BACKYARD_POOL_PARTY_ID, backyardPoolPartyCamera, createBackyardPoolDriftScene, createBackyardPoolPartyScene } from "./backyard-pool-party-scene";
import { FOREST_WATERFALL_ID, createForestWaterfallScene, forestWaterfallCamera } from "./forest-waterfall-scene";
import { createUniformTroughScene, UNIFORM_TROUGH_CAMERA } from "./uniform-trough-scenes";
import { withHeroLayout } from "./voxel-scenery/hero-layout";
import { terrainHeightAt, type TerrainDescription, type TerrainGrid } from "./terrain";
import type { EnvironmentId } from "./environments";
import { sceneDamBreakFractions } from "./initial-fluid";
import { solidVoxelEditsForScene, solidVoxelShellForScene } from "./scene-lattice";
import { sceneWithEnvironment } from "./scenery-presets";
import { withSceneryNodes } from "./scenery-edit";
import type { SceneryGraph } from "./scenery-graph";
import {
  defineScene,
  presentationModeForScene,
  sceneCardForDefinition,
  sceneDocument,
  type SceneCard,
  type SceneDefinition,
  type SceneLattice,
  type ScenePresentationMode,
  type SceneVariant,
} from "./scene-definition";

/** Preset factories author only their extra solid edits; the catalog compiler
 * adds the ordinary tank shell once, after the final lattice is known. */
function sceneBody(): SceneDescription {
  const scene = cloneScene(defaultScene);
  scene.solidVoxels = [];
  return scene;
}

/**
 * The preset view of a catalog entry.
 *
 * Retained as a projection of `SceneDefinition` rather than a second source of
 * truth: forty-eight modules import `scenePresets`/`getScenePreset`, and a
 * scene's identity now lives in one place regardless of which shape a caller
 * happens to want. `group` is the definition's shelf and is therefore a plain
 * string — it used to be a closed union of four labels written for us, which is
 * exactly the thing the audience model replaces.
 */
export interface ScenePreset {
  id: string;
  name: string;
  group: string;
  description: string;
  create(): SceneDescription;
  camera?: Partial<CameraState>;
  /** Art-directed background that is part of this preset's presentation. */
  background: EnvironmentId;
  /** Whether the authored dry world is presented behind the raster water. */
  presentationMode: ScenePresentationMode;
  renderProfile?: SceneRenderProfile;
}

/**
 * A deliberately small uniform-solid transfer demonstration.
 *
 * The authored fluid lattice is only the left tank. Ordinary SolidWorld
 * voxels extend a catch basin beyond its positive-X wall, and a generic clear
 * edit cuts a six-by-four floor-level opening through that wall. The solver
 * therefore has no pre-authored fluid owner on the far side: it must create
 * signed pages along the escaping fluid course, while the wider voxel basin is
 * the finite catastrophe boundary.
 */
function createExternalVoxelPoolTransferScene(sceneId: string): SceneDescription {
  const cellSize_m = 0.05;
  const scene = sceneBody();
  scene.sceneId = sceneId;
  scene.duration_s = 4;
  scene.container = {
    ...scene.container,
    width_m: 8 * cellSize_m,
    height_m: 12 * cellSize_m,
    depth_m: 16 * cellSize_m,
    fillFraction: 0.75,
    top: "closed",
    fluidWallMode: "free-slip",
  };
  scene.voxelDomain = { finestCellSize_m: cellSize_m, brickSize_cells: 8 };
  scene.fluid.initialCondition = "tank-fill";
  delete scene.fluid.initialDamBreakDimensions_m;
  delete scene.fluid.initialDamBreakOrigin_m;
  scene.fluid.surfaceTension_N_m = 0;
  scene.solidVoxels = [
    // The catalog compiler authors the small tank shell first. This clear is
    // therefore literally a missing SolidWorld voxel region, not an opening
    // descriptor consumed by the solver.
    { operation: "clear", minimum: [8, 0, 5], maximumExclusive: [9, 4, 11] },
    // Catch-basin floor, depth walls, and far wall. The small tank's shell
    // closes the near end except for the authored opening above.
    { operation: "fill", minimum: [8, -1, 0], maximumExclusive: [32, 0, 16],
      materialId: 2 },
    { operation: "fill", minimum: [8, 0, -1], maximumExclusive: [32, 16, 0],
      materialId: 2 },
    { operation: "fill", minimum: [8, 0, 16], maximumExclusive: [32, 16, 17],
      materialId: 2 },
    { operation: "fill", minimum: [32, 0, 0], maximumExclusive: [33, 16, 16],
      materialId: 2 },
    // The initial tank is twelve cells high but the connectivity unit is an
    // eight-cell page. Extend only its vertical side voxels through the
    // remainder of the second page so empty support cannot route around a
    // wall. The catalog compiler supplies the small tank's ceiling at y=12.
    { operation: "fill", minimum: [-1, 12, 0], maximumExclusive: [0, 16, 16],
      materialId: 2 },
    { operation: "fill", minimum: [8, 12, 0], maximumExclusive: [9, 16, 16],
      materialId: 2 },
    { operation: "fill", minimum: [0, 12, -1], maximumExclusive: [8, 16, 0],
      materialId: 2 },
    { operation: "fill", minimum: [0, 12, 16], maximumExclusive: [8, 16, 17],
      materialId: 2 },
  ];
  delete scene.fluid.initialBrickSeeds_m;
  delete scene.fluid.initialBrickSeedsAdditive;
  delete scene.fluid.initialLiquidVolumes;
  delete scene.fluid.inflow;

  scene.rigidBodies = [];
  scene.numerics.fixedDt_s = scene.numerics.maxDt_s = 1 / 30;
  return scene;
}

export function createBoundedPoolTransferScene(): SceneDescription {
  return createExternalVoxelPoolTransferScene("bounded-pool-transfer");
}

/** Finest cell shared with the whole hydrostatic/dam family, so a wall measured
 * here is comparable to theirs: only the lattice *extent* differs. */
export const POWER_DROPLET_CELL_SIZE_M = 0.05;


export const POWER_DROPLET_RESERVOIR_M = Object.freeze({ x: 0.25, y: 0.2, z: 0.25 });

/** 5 x 4 x 5 cells of the authored reservoir. The invariant of the family. */
export const POWER_DROPLET_LIQUID_CELLS = 100;

/** Container edges of the geometric domain sweep, in finest cells. */
export const POWER_DROPLET_EDGE_CELLS = Object.freeze([64, 128, 240, 256] as const);
export type PowerDropletEdgeCells = typeof POWER_DROPLET_EDGE_CELLS[number];

export const POWER_FILL_EDGE_CELLS = 256;

/**
 * The three reservoirs, in finest cells, and why they are these three.
 *
 * Each is the previous one doubled on every axis: 5x4x5 -> 10x8x10 -> 20x16x20,
 * so liquid occupancy is exactly 100 -> 800 -> 6,400 and each step is exactly
 * 8x. A geometric sweep with a constant ratio is what makes the classification
 * mechanical rather than a judgement call — over one step, a flat pass reads
 * 1x, a linear pass reads 8x, and an area-shaped pass reads 4x — and holding
 * the 5:4:5 aspect ratio constant keeps the surface/volume law fixed too, so a
 * pass that tracks free-surface area rather than volume is still readable
 * (4x per step) instead of being confounded by a reshaped box.
 *
 * 6,400 rather than a round 8,000 for the top member: 20x20x20 would be 8,000
 * cells but a 1.25x-taller box, which breaks both the exact 8x volume ratio and
 * the fixed aspect ratio and buys nothing. The ratio is the instrument.
 *
 * The first member is the droplet family's own 5x4x5 reservoir, reproduced
 * exactly rather than approximated, so the fill sweep's low end is literally
 * the droplet sweep's high end.
 */
export const POWER_FILL_RESERVOIR_CELLS = Object.freeze([
  Object.freeze({ x: 5, y: 4, z: 5 }),
  Object.freeze({ x: 10, y: 8, z: 10 }),
  Object.freeze({ x: 20, y: 16, z: 20 }),
] as const);

/** Liquid cells per member: the family's independent variable, and the axis
 * every per-pass cost in the sweep is regressed against. */
export const POWER_FILL_LIQUID_CELLS = Object.freeze([100, 800, 6_400] as const);
export type PowerFillLiquidCells = typeof POWER_FILL_LIQUID_CELLS[number];

/** The reservoir a member is built from, in finest cells. Keyed by the liquid
 * count because that count is the family's independent variable and the thing
 * its scene ids are named for. */
export function powerFillReservoirCells(liquidCells: PowerFillLiquidCells) {
  const index = POWER_FILL_LIQUID_CELLS.indexOf(liquidCells);
  const reservoir = POWER_FILL_RESERVOIR_CELLS[index];
  if (!reservoir) throw new RangeError(`No authored fill reservoir for ${liquidCells} liquid cells`);
  return reservoir;
}


export function createPowerFillScene(liquidCells: PowerFillLiquidCells): SceneDescription {
  const scene = sceneBody();
  scene.sceneId = `power-fill-${POWER_FILL_EDGE_CELLS}-${liquidCells}`;
  scene.duration_s = 1;
  scene.rigidBodies = [];
  const extent_m = POWER_FILL_EDGE_CELLS * POWER_DROPLET_CELL_SIZE_M;
  const cells = powerFillReservoirCells(liquidCells);
  const reservoir = {
    x: cells.x * POWER_DROPLET_CELL_SIZE_M,
    y: cells.y * POWER_DROPLET_CELL_SIZE_M,
    z: cells.z * POWER_DROPLET_CELL_SIZE_M,
  };
  scene.container = {
    ...scene.container,
    width_m: extent_m,
    height_m: extent_m,
    depth_m: extent_m,
    // The expression `validateScene` re-derives, so its 1e-9 equality is exact.
    fillFraction: (reservoir.x * reservoir.y * reservoir.z) / (extent_m * extent_m * extent_m),
    top: "closed",
    fluidWallMode: "free-slip",
  };
  scene.voxelDomain = { finestCellSize_m: POWER_DROPLET_CELL_SIZE_M, brickSize_cells: 8 };
  scene.fluid.initialCondition = "dam-break";
  scene.fluid.initialDamBreakDimensions_m = reservoir;
  // The four deletions that keep `analyticSparseBootstrap` true.
  delete scene.fluid.initialDamBreakOrigin_m;
  delete scene.fluid.initialBrickSeeds_m;
  delete scene.fluid.initialBrickSeedsAdditive;
  delete scene.fluid.inflow;
  scene.fluid.surfaceTension_N_m = 0;
  scene.numerics.fixedDt_s = scene.numerics.maxDt_s = 1 / 30;
  scene.solidVoxels = [...solidVoxelShellForScene(scene), ...scene.solidVoxels];
  return scene;
}

/** World-space centre of the single seeded 8-cubed fluid brick (the -x/-z quadrant). */
export const BRICK_QUAD_DAM_SEED_M = { x: -0.2, y: 0.2, z: -0.2 };

/** A suspended liquid sphere impacts a broad pool on a 128×96×128 lattice. */
export function createCoarseFirstPoolImpactScene(): SceneDescription {
  const scene = sceneBody();
  scene.sceneId = "coarse-first-pool-impact";
  scene.duration_s = 4;
  scene.rigidBodies = [];
  scene.container = { ...scene.container, width_m: 6.4, height_m: 4.8,
    depth_m: 6.4, fillFraction: 1 / 3, top: "closed", fluidWallMode: "free-slip" };
  scene.voxelDomain = { finestCellSize_m: 0.05, brickSize_cells: 8 };
  scene.fluid.initialCondition = "tank-fill";
  scene.fluid.initialLiquidVolumes = [
    { shape: "sphere", center_m: { x: 0, y: 3.65, z: 0 }, radius_m: 1.0 },
  ];
  delete scene.fluid.initialBrickSeeds_m;
  delete scene.fluid.initialBrickSeedsAdditive;
  delete scene.fluid.inflow;
  scene.fluid.surfaceTension_N_m = 0;
  scene.fluid.dynamicViscosity_Pa_s = 0;
  scene.numerics.fixedDt_s = scene.numerics.maxDt_s = 1 / 30;
  return scene;
}

/** Half the linear resolution: the identical 6.4 x 4.8 x 6.4 m tank and the
 * identical falling ball, on 0.1 m finest cells (64x48x64). Physical dimensions
 * are shared with the full scene so the family varies resolution alone. */
export function createCoarseFirstPoolImpactHalfScene(): SceneDescription {
  const scene = createCoarseFirstPoolImpactScene();
  scene.sceneId = "coarse-first-pool-impact-half";
  scene.voxelDomain = { ...scene.voxelDomain, finestCellSize_m: 0.1 };
  return scene;
}

/** Z-invariant disk/pool cross-section, eight fine cells deep with free-slip
 * end walls. Extend the cylinder beyond the walls so its caps cannot alter phi. */
export function createCoarseFirstPoolImpactHalfSlabScene(): SceneDescription {
  const scene = createCoarseFirstPoolImpactHalfScene();
  scene.sceneId = "coarse-first-pool-impact-half-slab";
  scene.container.depth_m = 0.8;
  scene.fluid.initialLiquidVolumes = [{ shape: "cylinder",
    center_m: { x: 0, y: 3.65, z: 0 }, radius_m: 1, halfHeight_m: 2 }];
  return scene;
}

/** Quarter the linear resolution: the same tank and ball on 0.2 m finest cells
 * (32x24x32). */
export function createCoarseFirstPoolImpactQuarterScene(): SceneDescription {
  const scene = createCoarseFirstPoolImpactScene();
  scene.sceneId = "coarse-first-pool-impact-quarter";
  scene.voxelDomain = { ...scene.voxelDomain, finestCellSize_m: 0.2 };
  return scene;
}

export function createFallingWaterTorusScene(): SceneDescription {
  const scene = createCoarseFirstPoolImpactScene();
  scene.sceneId = "falling-water-torus";
  scene.duration_s = 3;
  scene.container = { ...scene.container, width_m: 1.6, height_m: 1.6,
    depth_m: 1.6, fillFraction: 0 };
  scene.fluid.initialLiquidVolumes = [{ shape: "torus",
    center_m: { x: 0, y: 1.0, z: 0 }, radius_m: 0.4, tubeRadius_m: 0.125 }];
  return scene;
}

export function createTinyHydrostaticScene(): SceneDescription {
  const scene = sceneBody();
  scene.sceneId = "tiny-hydrostatic-two-level";
  scene.duration_s = 0.1;
  scene.rigidBodies = [];
  scene.container = {
    ...scene.container,
    width_m: 0.8,
    height_m: 0.8,
    depth_m: 0.8,
    fillFraction: 0.75,
    top: "closed",
    fluidWallMode: "free-slip",
  };
  scene.voxelDomain = { finestCellSize_m: 0.05, brickSize_cells: 8 };
  scene.fluid.initialCondition = "tank-fill";
  scene.fluid.surfaceTension_N_m = 0;
  delete scene.fluid.initialBrickSeeds_m;
  delete scene.fluid.initialBrickSeedsAdditive;
  delete scene.fluid.inflow;
  scene.numerics.fixedDt_s = scene.numerics.maxDt_s = 1 / 30;
  return scene;
}

/**
 * A larger hydrostatic oracle with an intentionally cell-cut free surface.
 * Its 32x24x16 lattice is deep enough for a materially larger adaptive
 * pressure layout while remaining small enough for an isolated Dawn smoke.
 */
export function createLargeHydrostaticScene(): SceneDescription {
  const scene = sceneBody();
  scene.sceneId = "large-hydrostatic-offset";
  scene.duration_s = 0.1;
  scene.rigidBodies = [];
  scene.container = {
    ...scene.container,
    width_m: 1.6,
    height_m: 1.2,
    depth_m: 0.8,
    // 61/96 of 1.2 m = 0.7625 m = 15.25 cells at h = 0.05 m.
    fillFraction: 61 / 96,
    top: "closed",
    fluidWallMode: "free-slip",
  };
  scene.voxelDomain = { finestCellSize_m: 0.05, brickSize_cells: 8 };
  scene.fluid.initialCondition = "tank-fill";
  scene.fluid.surfaceTension_N_m = 0;
  delete scene.fluid.initialBrickSeeds_m;
  delete scene.fluid.initialBrickSeedsAdditive;
  delete scene.fluid.inflow;
  scene.numerics.fixedDt_s = scene.numerics.maxDt_s = 1 / 30;
  return scene;
}


export function createMinimalPowerDamBreakScene(): SceneDescription {
  const scene = sceneBody();
  scene.sceneId = "minimal-power-dam-break";
  scene.duration_s = 0.1;
  scene.rigidBodies = [];
  scene.container = {
    ...scene.container,
    width_m: 0.8,
    height_m: 0.8,
    depth_m: 0.8,
    fillFraction: 23 / 64,
    top: "closed",
    fluidWallMode: "free-slip",
  };
  scene.voxelDomain = { finestCellSize_m: 0.05, brickSize_cells: 8 };
  scene.fluid.initialCondition = "dam-break";
  delete scene.fluid.initialBrickSeeds_m;
  delete scene.fluid.initialBrickSeedsAdditive;
  scene.fluid.surfaceTension_N_m = 0;
  delete scene.fluid.inflow;
  scene.numerics.fixedDt_s = scene.numerics.maxDt_s = 1 / 30;
  scene.solidVoxels = [...solidVoxelShellForScene(scene), ...scene.solidVoxels];
  return scene;
}


export function createSymmetricExpansionScene(): SceneDescription {
  const scene = sceneBody();
  scene.sceneId = "symmetric-expansion";
  scene.duration_s = 1;
  scene.rigidBodies = [];
  scene.container = {
    ...scene.container,
    width_m: 1.6,
    height_m: 0.8,
    depth_m: 1.6,
    // One centred 0.8 x 0.4 x 0.8 m body in a 1.6 x 0.8 x 1.6 m tank.
    fillFraction: 1 / 8,
    top: "closed",
    fluidWallMode: "free-slip",
  };
  scene.voxelDomain = { finestCellSize_m: 0.05, brickSize_cells: 8 };
  scene.fluid.initialCondition = "tank-fill";
  scene.fluid.initialBrickSeeds_m = [
    { x: -0.2, y: 0.2, z: -0.2 }, { x: 0.2, y: 0.2, z: -0.2 },
    { x: -0.2, y: 0.2, z: 0.2 }, { x: 0.2, y: 0.2, z: 0.2 },
  ];
  delete scene.fluid.initialBrickSeedsAdditive;
  delete scene.fluid.initialDamBreakDimensions_m;
  delete scene.fluid.inflow;
  scene.fluid.dynamicViscosity_Pa_s = 0;
  scene.fluid.surfaceTension_N_m = 0;
  scene.numerics.fixedDt_s = scene.numerics.maxDt_s = 1 / 30;
  scene.solidVoxels = [...solidVoxelShellForScene(scene), ...scene.solidVoxels];
  return scene;
}


export function createMinimalPowerDamBreak64Scene(): SceneDescription {
  const scene = createMinimalPowerDamBreakScene();
  scene.solidVoxels = solidVoxelEditsForScene(scene);
  scene.sceneId = "minimal-power-dam-break-64";
  scene.voxelDomain = {
    ...scene.voxelDomain,
    finestCellSize_m: scene.voxelDomain.finestCellSize_m / 4,
  };
  scene.solidVoxels = [...solidVoxelShellForScene(scene), ...scene.solidVoxels];
  return scene;
}

/**
 * A watchable 128-cubed version of the analytic mini dam. It keeps the exact
 * 0.8 m physical tank and reservoir while resolving each axis at 6.25 mm.
 * Coarse-only surface transport makes that resolution available to the liquid
 * instead of spending the frame and memory budget on a second 4x lattice.
 */
export function createHighResolutionDamBreakScene(): SceneDescription {
  const scene = createMinimalPowerDamBreakScene();
  scene.solidVoxels = solidVoxelEditsForScene(scene);
  scene.sceneId = "high-resolution-dam-break";
  scene.duration_s = 20;
  scene.voxelDomain = {
    ...scene.voxelDomain,
    finestCellSize_m: scene.voxelDomain.finestCellSize_m / 8,
  };
  scene.solidVoxels = [...solidVoxelShellForScene(scene), ...scene.solidVoxels];
  return scene;
}

/**
 * The depth-uniform extrusion of the 2D case the advance lab already runs.
 *
 * The lab seeds its Rust 2D world from `water-box-dam-break` by cutting one
 * plane through the tank's depth, which leaves a 12x14-cell column standing in
 * a 24x16-cell box. That 2D answer is only comparable with a 3D one if the
 * third axis contributes nothing, and in the 0.8 m tank it contributes a great
 * deal: the column is narrower than the tank in z, so the 3D run spreads
 * sideways into air the 2D run has no room for. This scene is the same x-y
 * geometry on the same 50 mm lattice, extruded through `CM12_SLAB_DEPTH_CELLS`
 * with the reservoir spanning the whole slab, so every z plane holds exactly
 * the plane the lab solves and the only remaining difference is the solver.
 *
 * Eight cells because that is the depth the CM12 figures reconstruct their own
 * 2D cases at — one B8 brick, and the minimum the dense pressure hierarchy will
 * construct at all.
 *
 * Two departures from the 3D tank are what make this an extrusion rather than
 * merely a thin tank:
 *
 *  - Free-slip on all six walls. The 2D world has no tangential wall condition
 *    to disagree with: its `BoundaryMode::Closed` pins the normal component and
 *    leaves the tangent free. A no-slip slab would differ from the 2D run in
 *    the boundary layer along the floor and end walls, which is a fact about
 *    `fluidWallMode` rather than about dimensionality.
 *  - `depthBoundary: "symmetry"`, the catalog's statement that a depth is the
 *    thickness of the plane being looked at rather than an axis with water in
 *    it. The uniform reference reads it for its z faces, and the editor reads
 *    it to extrude a dropped ball into a disk spanning the slab instead of
 *    scaling one by a depth this case does not have.
 *
 * Surface tension is zeroed for the reason the CM12 figures zero it: there is
 * no capillary term on the 2D lane, so leaving it on would make the comparison
 * partly a measurement of that term.
 */
export function createDamBreakSlabScene(): SceneDescription {
  const scene = sceneBody();
  scene.sceneId = "water-box-dam-break-slab";
  scene.rigidBodies = [];
  const depth_m = CM12_SLAB_DEPTH_CELLS * scene.voxelDomain.finestCellSize_m;
  scene.container = {
    ...scene.container,
    depth_m,
    top: "closed",
    fluidWallMode: "free-slip",
    depthBoundary: "symmetry",
  };
  // The 3D tank's reservoir face, carried through the full slab: 12x14 cells of
  // liquid, which is the 168 fine-cell areas the 2D lane conserves.
  scene.fluid.initialDamBreakDimensions_m = { x: 0.6, y: 0.7, z: depth_m };
  scene.container.fillFraction = 0.6 * 0.7 * depth_m
    / (scene.container.width_m * scene.container.height_m * scene.container.depth_m);
  scene.fluid.surfaceTension_N_m = 0;
  return scene;
}

/** The same physical mini dam on the midpoint 32³ coarse-only lattice. */
export function createMinimalPowerDamBreak32Scene(): SceneDescription {
  const scene = createMinimalPowerDamBreakScene();
  scene.solidVoxels = solidVoxelEditsForScene(scene);
  scene.sceneId = "minimal-power-dam-break-32";
  scene.voxelDomain = {
    ...scene.voxelDomain,
    finestCellSize_m: scene.voxelDomain.finestCellSize_m / 2,
  };
  scene.solidVoxels = [...solidVoxelShellForScene(scene), ...scene.solidVoxels];
  return scene;
}


export function createSparseCM12LongDamBreakScene(): SceneDescription {
  const scene = createMinimalPowerDamBreakScene();
  scene.solidVoxels = solidVoxelEditsForScene(scene);
  scene.sceneId = "sparse-cm12-long-dam-break";
  scene.duration_s = 4;
  scene.container = {
    ...scene.container,
    width_m: 2.4,
    height_m: 1.2,
    depth_m: 0.4,
    fillFraction: 5 / 72,
    top: "closed",
    fluidWallMode: "free-slip",
  };
  // This is deliberately one DETAIL rung above the original 96x48x16
  // validation scene. The physical tank and reservoir stay unchanged while
  // the 12.5 mm lattice makes the sparse 1/2/4/8 hierarchy do meaningful work
  // at a scale where a blanket-fine policy is visibly and measurably costly.
  scene.voxelDomain = { finestCellSize_m: 0.0125, brickSize_cells: 8 };
  scene.fluid.initialDamBreakDimensions_m = { x: 0.4, y: 0.5, z: 0.4 };
  delete scene.fluid.initialDamBreakOrigin_m;
  scene.numerics.fixedDt_s = scene.numerics.maxDt_s = 1 / 30;
  scene.solidVoxels = [...solidVoxelShellForScene(scene), ...scene.solidVoxels];
  return scene;
}

/** Shallow, initially resting dry-bed release for pre-impact Ritter metrics.
 * The eight-cell transverse width keeps the production B8 path supported.
 * See docs/geometric-dam-reference.md for coordinates and validity windows. */
export function createSparseGeometricRitterDamBreakScene(): SceneDescription {
  const scene = createMinimalPowerDamBreakScene();
  scene.sceneId = "sparse-geometric-ritter-dam-break";
  scene.duration_s = 2;
  scene.container = { ...scene.container, width_m: 4, height_m: 0.2,
    depth_m: 0.1, fillFraction: 0.3, top: "closed", fluidWallMode: "free-slip" };
  scene.voxelDomain = { finestCellSize_m: 0.0125, brickSize_cells: 8 };
  scene.fluid.initialDamBreakDimensions_m = { x: 2.4, y: 0.1, z: 0.1 };
  delete scene.fluid.initialDamBreakOrigin_m;
  scene.fluid.initialVelocity_m_s = { x: 0, y: 0, z: 0 };
  scene.fluid.gravity_m_s2 = { x: 0, y: -9.81, z: 0 };
  scene.fluid.dynamicViscosity_Pa_s = 0;
  scene.fluid.surfaceTension_N_m = 0;
  // Replace the inherited mini shell after changing the physical lattice.
  scene.solidVoxels = [...solidVoxelShellForScene(scene)];
  return scene;
}

/**
 * Chentanez & Mueller's tall-cell Flood case uses a 64 x (32 + 2) x 256
 * simulation grid over uneven terrain (`docs/papers/tallCells.pdf`, Table 2
 * and Figure 1). Our sparse world does not compress a whole water column into
 * one tall cell, so this reconstruction rotates the paper's 256-cell travel
 * axis onto +X and gives the terrain and falling reservoir a physical height of
 * its own. The important experiment remains the same: a small active body moves
 * through a much larger world while solid ground varies by metres beneath it.
 *
 * The vertical extent is the one axis sized by the scene rather than the paper.
 * It has to hold the shelf, the reservoir standing on it, and the air the
 * released sheet throws, and every one of those cells is paid for in domain
 * bricks — so it is the smallest multiple of the 8-cell brick that clears the
 * reservoir's crown with room to spare: 112 cells is 5.6 m over a 4 m shelf
 * carrying a 0.8 m block, and the runout below has 5.5 m of air above it.
 */
export const TALL_CELLS_FLOOD_CELL_SIZE_M = 0.05;
export const TALL_CELLS_FLOOD_GRID = Object.freeze([256, 112, 64] as const);
export const TALL_CELLS_FLOOD_UPHILL_HEIGHT_M = 4;
export const TALL_CELLS_FLOOD_DOWNHILL_HEIGHT_M = 0.1;
/**
 * How much flat ground the far end keeps.
 *
 * The slope is short so that it can be steep, and what it gives up is spent
 * here: the front leaves the toe at several metres a second and needs somewhere
 * to spread rather than a wall to pile against. Four metres is a third of the
 * travel axis.
 */
export const TALL_CELLS_FLOOD_RUNOUT_M = 4;

export const TALL_CELLS_FLOOD_SLOPE_EASE_EXPONENT = 2;
/**
 * Peak height of the transverse bank, at the domain's z edges and mid-slope.
 *
 * A sixteenth of the drop, which is what the previous 0.12 m was against the
 * previous 1.9 m: enough crossfall to keep the released sheet in the middle of
 * the course, not enough to make the hillside a flume.
 */
export const TALL_CELLS_FLOOD_CHANNEL_BANK_M = 0.25;
export const TALL_CELLS_FLOOD_RESERVOIR_M = Object.freeze({
  x: 1.2,
  y: 0.8,
  z: 2.4,
});

/**
 * The Flood benchmark's set: its own ground, standing on the house stage floor.
 *
 * The scene authors its graph rather than letting `finishSceneDocument` attach
 * the Stage environment's, because that set is a floor *and a practical*, and a
 * hillside wants the first without the second. The lamp was once the only thing
 * lighting a 12.8 m landscape: everything outside its beam was the stage rig's
 * near-black fill, the beam's own centre blew the ground out, and the cone sat
 * in frame as a stair-stepped white blob. A landscape is lit by the sky, so the
 * light is a sun — `scene.lighting` in `createTallCellsHillsideDamBreakScene`
 * raises the directional (which is also the water's only key, via
 * `resolveWaterKeyLight`) and gives the environment a daylight hemisphere.
 *
 * The floor is the house set's own node, solved for this container by
 * `studioStageSceneryGraph`, so the hillside stands on what every other stage
 * scene stands on. It used to be left out on cost: at the solver's 50 mm cell a
 * plate spanning the paper domain dominated sparse-scene startup. The
 * environment ladder draws that plate at the rung its own thickness allows, and
 * the CM12 figures carry the same floor round the same 12.8 m footprint.
 */
function tallCellsFloodScenery(scene: SceneDescription): SceneryGraph {
  const stage = studioStageSceneryGraph(scene);
  return {
    palettes: { stage: stage.palettes.stage },
    nodes: [
      {
        kind: "terrain-shell",
        id: "shell",
        // Not porcelain: a 0.90-albedo ground clips the moment it is lit like
        // outdoors, and this is a hillside rather than a studio prop.
        materialModel: "garden-terrain",
      },
      ...stage.nodes.filter((node) => node.id === "stage/floor"),
    ],
  };
}

/**
 * Authoring samples for one voxel solid: a flat launch shelf, a steep central
 * drop that eases into a long flat runout, and a transverse bank. The bank
 * makes this uneven ground rather than an inclined container floor, while the
 * shelf stays exactly level under the whole reservoir footprint so the initial
 * water block rests on solid voxels everywhere.
 *
 * All three sections are one expression of `progress`, which is clamped, so the
 * shelf and the runout are the clamped ends of the slope rather than separate
 * pieces that could drift apart from it.
 */
export function createTallCellsFloodTerrain(): TerrainDescription {
  const [nx, , nz] = TALL_CELLS_FLOOD_GRID;
  const width_m = nx * TALL_CELLS_FLOOD_CELL_SIZE_M;
  const depth_m = nz * TALL_CELLS_FLOOD_CELL_SIZE_M;
  const grid: TerrainGrid = {
    kind: "grid",
    origin_m: { x: -0.5 * width_m, z: -0.5 * depth_m },
    spacing_m: TALL_CELLS_FLOOD_CELL_SIZE_M,
    size: { nx: nx + 1, nz: nz + 1 },
    heights_m: [],
  };
  const slopeStartX_m = grid.origin_m.x + TALL_CELLS_FLOOD_RESERVOIR_M.x;
  const slopeEndX_m = 0.5 * width_m - TALL_CELLS_FLOOD_RUNOUT_M;
  const slopeRun_m = slopeEndX_m - slopeStartX_m;
  for (let z = 0; z <= nz; z += 1) {
    const worldZ_m = grid.origin_m.z + z * grid.spacing_m;
    const transverse = Math.abs(worldZ_m) / (0.5 * depth_m);
    for (let x = 0; x <= nx; x += 1) {
      const worldX_m = grid.origin_m.x + x * grid.spacing_m;
      const progress = Math.max(0, Math.min(1,
        (worldX_m - slopeStartX_m) / slopeRun_m));
      // Written as a two-term blend rather than `uphill + fall * drop` so both
      // ends land on their constant exactly: the shelf under the reservoir has
      // to be bit-flat, and the runout has to agree with `baseHeight_m`.
      const fall = (1 - progress) ** TALL_CELLS_FLOOD_SLOPE_EASE_EXPONENT;
      const slopeHeight_m = TALL_CELLS_FLOOD_UPHILL_HEIGHT_M * fall
        + TALL_CELLS_FLOOD_DOWNHILL_HEIGHT_M * (1 - fall);
      // Zero on both shelves, strongest halfway down the hill. This shallow
      // crossfall keeps the released sheet in view without becoming a vessel.
      const slopeWindow = 4 * progress * (1 - progress);
      const channelBank_m = TALL_CELLS_FLOOD_CHANNEL_BANK_M * slopeWindow
        * transverse * transverse;
      grid.heights_m.push(slopeHeight_m + channelBank_m);
    }
  }
  return {
    baseHeight_m: TALL_CELLS_FLOOD_DOWNHILL_HEIGHT_M,
    features: [],
    grid,
  };
}

/** A hillside dam break with a lattice-aligned reservoir and a dry slope. */
export function createTallCellsHillsideDamBreakScene(): SceneDescription {
  const scene = sceneBody();
  const [nx, ny, nz] = TALL_CELLS_FLOOD_GRID;
  const cell_m = TALL_CELLS_FLOOD_CELL_SIZE_M;
  const reservoir = TALL_CELLS_FLOOD_RESERVOIR_M;
  scene.sceneId = "tall-cells-hillside-dam-break";
  scene.randomSeed = 2011;
  scene.duration_s = 10;
  scene.container = {
    ...scene.container,
    width_m: nx * cell_m,
    height_m: ny * cell_m,
    depth_m: nz * cell_m,
    fillFraction: reservoir.x * reservoir.y * reservoir.z
      / (nx * cell_m * ny * cell_m * nz * cell_m),
    top: "closed",
    fluidWallMode: "free-slip",
    vessel: "none",
  };
  scene.voxelDomain = { finestCellSize_m: cell_m, brickSize_cells: 8 };
  scene.nominalResolution = { length_m: cell_m };
  scene.terrain = createTallCellsFloodTerrain();
  scene.scenery = tallCellsFloodScenery(scene);
  // Daylight, over the stage's base rig.
  //
  // `environment: "stage"` is kept for its floor and its absence of a room, but
  // its rig is a dark room under one practical, and this scene has no practical
  // and no room. Left alone it rendered as a black frame
  // with a clipped white hillside in it. Every field below is therefore a
  // deliberate override of `STUDIO_STAGE_DRY_SCENE_LIGHTING`; `svoSceneLighting`
  // merges per field, so the stage's tone curve is all that survives.
  //
  // The directional is doing two jobs and both want it near daylight strength:
  // it is the sun the hillside is modelled by, and it is the water's only key
  // (`resolveWaterKeyLight` — the raster water pipeline never sees the light
  // table). Aimed high and from the front-left of the travel axis so the slope
  // has a lit face and a shaded one, and so the released sheet catches a
  // highlight on its way downhill.
  //
  // The hemisphere is the other half of an outdoor frame, and on a set with no
  // walls it is also the backdrop: a miss shades as `svoEnvironmentPrefilteredSpecular`,
  // so `upper` and `lower` are literally the sky and the ground haze the camera
  // sees past the terrain. Cool above, warm and dim below.
  scene.lighting = {
    directional: {
      direction: [-0.45, 0.80, 0.35],
      colorLinear: [1, 0.90, 0.74],
      intensity: 2.8,
    },
    environment: {
      diffuseScale: 1.2,
      specularScale: 1,
      upperRadianceLinear: [0.34, 0.44, 0.64],
      lowerRadianceLinear: [0.030, 0.024, 0.018],
      accentRadianceLinear: [0.36, 0.25, 0.15],
    },
    // Exposure is the camera, not the rig: the sun above is a physical
    // brightness and this is the stop that puts the lit hillside in the middle
    // of the range instead of on the shoulder.
    grade: {
      toneCurve: "aces",
      exposure: 0.58,
      whiteBalance: [1.08, 1, 0.92],
    },
  };
  scene.fluid.initialCondition = "dam-break";
  scene.fluid.initialDamBreakDimensions_m = { ...reservoir };
  scene.fluid.initialDamBreakOrigin_m = {
    x: 0,
    y: TALL_CELLS_FLOOD_UPHILL_HEIGHT_M,
    z: 0.4,
  };
  delete scene.fluid.initialBrickSeeds_m;
  delete scene.fluid.initialBrickSeedsAdditive;
  delete scene.fluid.initialLiquidVolumes;
  delete scene.fluid.inflow;
  scene.fluid.dynamicViscosity_Pa_s = 0;
  scene.fluid.surfaceTension_N_m = 0;
  // `vessel: "none"` hides the outline; it does not make the Flood benchmark
  // an unbounded fluid domain. Keep the box shell in SolidWorld so its fixed
  // side/end walls use the same editable voxel authority as every ordinary
  // tank (including clear patches cut into it). Like theirs it is physics
  // only: presentation leaves the uncut shell out of the render field.
  scene.solidVoxels = [...solidVoxelShellForScene(scene), ...scene.solidVoxels];
  scene.fluid.optics = {
    absorption_mInv: [0.045, 0.009, 0.006],
    scatter: [0.0012, 0.0055, 0.0049],
  };
  scene.rigidBodies = [];
  // Section 4 uses a 1/30 s simulation step for every example.
  scene.numerics.fixedDt_s = scene.numerics.maxDt_s = 1 / 30;
  return scene;
}

/**
 * A watchable, room-sized version of the minimal dam break. The
 * footprint is four times longer and four times wider, while the tank is only
 * 25% taller, making the container exactly 20x larger by volume. The reservoir
 * keeps the mini scene's absolute dimensions, volume, corner placement, and
 * lattice spacing; only the empty space around it grows.
 */
export function createLargePowerDamBreakScene(): SceneDescription {
  const scene = createMinimalPowerDamBreakScene();
  scene.solidVoxels = solidVoxelEditsForScene(scene);
  const miniDam = sceneDamBreakFractions(scene);
  const initialDamBreakDimensions_m = {
    x: miniDam.width * scene.container.width_m,
    y: miniDam.height * scene.container.height_m,
    z: miniDam.depth * scene.container.depth_m,
  };
  scene.sceneId = "large-power-dam-break";
  scene.container = {
    ...scene.container,
    width_m: 3.2,
    height_m: 1.0,
    depth_m: 3.2,
    fillFraction: scene.container.fillFraction / 20,
  };
  scene.fluid.initialDamBreakDimensions_m = initialDamBreakDimensions_m;
  scene.solidVoxels = [...solidVoxelShellForScene(scene), ...scene.solidVoxels];
  return scene;
}


export function createLargePowerHydrostaticScene(): SceneDescription {
  const mini = createMinimalPowerDamBreakScene();
  const scene = createLargePowerDamBreakScene();
  scene.sceneId = "large-power-hydrostatic";
  scene.duration_s = 1;
  scene.container = {
    ...scene.container,
    fillFraction: 1 / 80,
  };
  // Tank-fill always spans the whole 64x64 floor. A one-cell-deep 32x32
  // footprint quarters volume without dropping below the first cell centre.
  scene.fluid.initialCondition = "dam-break";
  scene.fluid.initialDamBreakDimensions_m = { x: 1.6, y: 0.05, z: 1.6 };
  delete scene.fluid.initialBrickSeeds_m;
  delete scene.fluid.initialBrickSeedsAdditive;
  delete scene.fluid.inflow;
  return scene;
}


export function createDeepPowerHydrostaticScene(): SceneDescription {
  const scene = createLargePowerHydrostaticScene();
  scene.solidVoxels = solidVoxelEditsForScene(scene);
  scene.sceneId = "deep-power-hydrostatic";
  scene.container = {
    ...scene.container,
    height_m: 2.4,
    // 5/6 of 48 cells is exactly 40; the nearest cell centres are 0.5 cells
    // clear on both sides, so the wet layer count cannot round either way.
    fillFraction: 5 / 6,
  };
  // A full-footprint tank fill, not the 20x lane's corner reservoir: the whole
  // point is that no liquid cell is near a lateral free surface.
  scene.fluid.initialCondition = "tank-fill";
  delete scene.fluid.initialDamBreakDimensions_m;
  scene.solidVoxels = [...solidVoxelShellForScene(scene), ...scene.solidVoxels];
  return scene;
}


export function createPowerDropletScene(edgeCells: PowerDropletEdgeCells): SceneDescription {
  const scene = sceneBody();
  scene.sceneId = `power-droplet-${edgeCells}`;
  scene.duration_s = 1;
  scene.rigidBodies = [];
  const extent_m = edgeCells * POWER_DROPLET_CELL_SIZE_M;
  const reservoir = POWER_DROPLET_RESERVOIR_M;
  scene.container = {
    ...scene.container,
    width_m: extent_m,
    height_m: extent_m,
    depth_m: extent_m,
    // The same expression `validateScene` re-derives, so the equality it
    // checks is exact rather than merely close.
    fillFraction: (reservoir.x * reservoir.y * reservoir.z) / (extent_m * extent_m * extent_m),
    top: "closed",
    fluidWallMode: "free-slip",
  };
  scene.voxelDomain = { finestCellSize_m: POWER_DROPLET_CELL_SIZE_M, brickSize_cells: 8 };
  scene.fluid.initialCondition = "dam-break";
  scene.fluid.initialDamBreakDimensions_m = { ...reservoir };
  delete scene.fluid.initialDamBreakOrigin_m;
  delete scene.fluid.initialBrickSeeds_m;
  delete scene.fluid.initialBrickSeedsAdditive;
  delete scene.fluid.inflow;
  scene.fluid.surfaceTension_N_m = 0;
  scene.numerics.fixedDt_s = scene.numerics.maxDt_s = 1 / 30;
  return scene;
}

/**
 * Bet-4 work-verdict scene: a genuinely volumetric pool, but bounded enough to
 * run as a one-step shipping-path gate. The topology-aligned 64x64x48 lattice
 * has 56 wet layers, so the regular deep interior dominates the free-surface
 * and solid-contact bands. It deliberately has no wave seed: D4 correctness remains
 * the separate symmetric-expansion gate, while this scene measures only avoided
 * machinery.
 */
export function createPowerHybridDeepOceanScene(): SceneDescription {
  const scene = sceneBody();
  scene.sceneId = "power-hybrid-deep-ocean";
  scene.rigidBodies = [];
  scene.container = { ...scene.container, width_m: 3.2, height_m: 3.2, depth_m: 2.4,
    fillFraction: 0.875, top: "closed", fluidWallMode: "no-slip" };
  scene.voxelDomain = { finestCellSize_m: 0.05, brickSize_cells: 8 };
  scene.fluid.initialCondition = "tank-fill";
  scene.fluid.surfaceTension_N_m = 0;
  delete scene.fluid.initialBrickSeeds_m;
  delete scene.fluid.initialBrickSeedsAdditive;
  delete scene.fluid.inflow;
  scene.numerics.fixedDt_s = scene.numerics.maxDt_s = 1 / 30;
  return scene;
}

/**
 * Analytic free-fall oracles for unilateral wall contact. A single 8-cubed
 * fluid brick is seeded flush against the closed lid of a 1.2 x 0.8 x 1.2 m
 * tank (24x16x24 cells at 0.05 m, a 3x2x3 brick lattice). No pressure field
 * can support liquid hanging from a ceiling — that would require tension —
 * so from t=0 the body must be in free fall: its centre of mass obeys
 * y(t) = 0.6 m - g t^2 / 2 until the lower face reaches the floor at about
 * 0.29 s. Any measured deviation is wall adhesion introduced by the
 * discretization, isolated from dam-break dynamics.
 *
 * The four variants form a 2x2 over the two contact kinds, which is what
 * makes them attributive rather than merely reproductive:
 *
 *              | no vertical walls   | two vertical walls
 *   lid        | ceiling-slab-drop   | corner-brick-drop
 *   mid-air    | midair-brick-drop   | midair-corner-drop
 *
 * The mid-air pair hangs the same brick in the middle layer of a taller
 * (24-cubed) tank so it touches no ceiling at all. Comparing a row isolates
 * what the vertical walls do; comparing a column isolates what the lid does.
 * `midair-brick-drop` touches nothing and is the zero-contact control: any
 * deviation it shows is the scheme's own transient, not adhesion.
 */
export type FreeFallDropSceneId = "ceiling-slab-drop" | "corner-brick-drop"
  | "midair-brick-drop" | "midair-corner-drop";

function createFreeFallDropScene(id: FreeFallDropSceneId): SceneDescription {
  const lidAttached = id === "ceiling-slab-drop" || id === "corner-brick-drop";
  const cornered = id === "corner-brick-drop" || id === "midair-corner-drop";
  const finestCellSize_m = 0.05;
  const dimensions_cells = { x: 24, y: lidAttached ? 16 : 24, z: 24 };
  const scene = sceneBody();
  scene.sceneId = id;
  scene.duration_s = 0.5;
  scene.rigidBodies = [];
  scene.container = {
    ...scene.container,
    // Derive every physical extent from the authored integer grid. Decimal
    // literals such as 1.2 / 24 and 0.8 / 16 have different binary rounding;
    // multiplying by the common authored spacing ensures even the smallest
    // rounded axis spacing spans a lid-flush analytic box on every axis.
    width_m: dimensions_cells.x * finestCellSize_m,
    // The lid variants seed the top brick of a two-brick column; the mid-air
    // variants seed the middle brick of a three-brick column. Both leave the
    // same eight cells of clearance beneath, so all four share one analytic
    // trajectory and one impact time.
    height_m: dimensions_cells.y * finestCellSize_m,
    depth_m: dimensions_cells.z * finestCellSize_m,
    // Ignored: the seeded brick replaces the base fill entirely.
    fillFraction: 512 / (24 * (lidAttached ? 16 : 24) * 24),
    top: "closed",
    fluidWallMode: "free-slip",
  };
  scene.voxelDomain = { finestCellSize_m, brickSize_cells: 8 };
  scene.fluid.initialCondition = "dam-break";
  scene.fluid.initialBrickSeeds_m = [{
    x: cornered ? -0.55 : 0,
    y: lidAttached ? 0.75 : 0.6,
    z: cornered ? -0.55 : 0,
  }];
  delete scene.fluid.initialBrickSeedsAdditive;
  scene.fluid.surfaceTension_N_m = 0;
  delete scene.fluid.inflow;
  scene.numerics.fixedDt_s = scene.numerics.maxDt_s = 1 / 30;
  return scene;
}

export function createCeilingSlabDropScene(): SceneDescription {
  return createFreeFallDropScene("ceiling-slab-drop");
}

export function createCornerBrickDropScene(): SceneDescription {
  return createFreeFallDropScene("corner-brick-drop");
}

export function createMidairBrickDropScene(): SceneDescription {
  return createFreeFallDropScene("midair-brick-drop");
}

export function createMidairCornerDropScene(): SceneDescription {
  return createFreeFallDropScene("midair-corner-drop");
}

export type RigidCouplingOracleSceneId = "rigid-hydrostatic" | "rigid-float" | "rigid-sink";

/** Three small analytic sphere/tank scenes used by the Dawn coupling oracle. */
export function createRigidCouplingOracleScene(id: RigidCouplingOracleSceneId): SceneDescription {
  const scene = sceneBody();
  scene.sceneId = id;
  scene.duration_s = id === "rigid-hydrostatic" ? 0.5 : id === "rigid-float" ? 2 : 1;
  scene.container = { ...scene.container, width_m: 0.8, height_m: 0.8, depth_m: 0.8,
    fillFraction: 0.5, top: "closed", fluidWallMode: "free-slip" };
  scene.voxelDomain = { finestCellSize_m: 0.05, brickSize_cells: 8 };
  scene.fluid.initialCondition = "tank-fill";
  scene.fluid.surfaceTension_N_m = 0;
  delete scene.fluid.initialBrickSeeds_m;
  delete scene.fluid.initialBrickSeedsAdditive;
  delete scene.fluid.inflow;
  delete scene.terrain;
  scene.numerics.fixedDt_s = scene.numerics.maxDt_s = 1 / 30;
  const radius = id === "rigid-float" ? 0.15 : 0.10;
  scene.rigidBodies = [{
    id: `${id}-sphere`, name: id, shape: "sphere",
    dimensions_m: { x: radius, y: radius, z: radius },
    density_kg_m3: id === "rigid-float" ? 500 : id === "rigid-sink" ? 2400 : 1000,
    position_m: { x: 0, y: id === "rigid-hydrostatic" ? 0.20 : id === "rigid-float" ? 0.40 : 0.60, z: 0 },
    orientation: { w: 1, x: 0, y: 0, z: 0 },
    linearVelocity_m_s: { x: 0, y: 0, z: 0 },
    angularVelocity_rad_s: { x: 0, y: 0, z: 0 },
    restitution: 0.05, friction: 0.3,
    motion: id === "rigid-hydrostatic" ? "static" : "dynamic",
  }];
  return scene;
}

export const createRigidHydrostaticScene = () => createRigidCouplingOracleScene("rigid-hydrostatic");
export const createRigidFloatScene = () => createRigidCouplingOracleScene("rigid-float");
export const createRigidSinkScene = () => createRigidCouplingOracleScene("rigid-sink");

/**
 * A tank sized so the finest solver grid is exactly 16x8x16 cells: a 2x2 x/z
 * arrangement of 8-cubed fluid bricks at one brick of height. Water starts as
 * a full-height column filling exactly one brick quadrant and dam-breaks
 * across the brick boundaries into the other three, which makes it the
 * minimal watchable scenario for brick residency activation, the sparse brick
 * atlas, and seam quality.
 */
export function createBrickQuadDamBreakScene(): SceneDescription {
  const scene = sceneBody();
  scene.sceneId = "brick-quad-dam-break";
  scene.rigidBodies = [];
  scene.container = { ...scene.container, width_m: 0.8, height_m: 0.4, depth_m: 0.8, fillFraction: 0.25, top: "closed", fluidWallMode: "no-slip" };
  scene.fluid.initialCondition = "dam-break";
  scene.fluid.initialBrickSeeds_m = [{ ...BRICK_QUAD_DAM_SEED_M }];
  delete scene.fluid.inflow;
  // The 0.05 m scene lattice gives exactly 16x8x16 cells: one 8-cell brick
  // high and a 2x2 brick footprint.
  scene.voxelDomain = { finestCellSize_m: 0.05, brickSize_cells: 8 };
  return scene;
}

/** Brick tiers occupied by each reservoir of the twin-dam scene: two bricks in
 * x, one in y, one in z, anchored at diagonally opposite floor corners. */
const TWIN_DAM_BRICK_TIERS = Object.freeze({
  x: Object.freeze([[0, 1], [5, 6]] as const),
  y: Object.freeze([0] as const),
  z: Object.freeze([0, 1] as const),
});


export function createTwinDamCollisionScene(): SceneDescription {
  const scene = sceneBody();
  scene.sceneId = "twin-dam-collision";
  scene.rigidBodies = [];
  scene.container = {
    ...scene.container,
    width_m: 2.8,
    height_m: 0.8,
    depth_m: 0.8,
    // Both reservoirs together: 2 x (0.8 x 0.4 x 0.4 m) of 2.8 x 0.8 x 0.8 m.
    // Seeds replace the base condition, so this only keeps fill-derived
    // diagnostics consistent with the water actually present.
    fillFraction: 1 / 7,
    top: "closed",
    fluidWallMode: "no-slip",
  };
  scene.fluid.initialCondition = "dam-break";
  // The authored lattice gives exactly 56x16x16 cells of 0.05 m.
  scene.voxelDomain = { finestCellSize_m: 0.05, brickSize_cells: 8 };
  const brick = 8 * scene.voxelDomain.finestCellSize_m;
  const seeds: { x: number; y: number; z: number }[] = [];
  // One seed per occupied brick, placed at the brick centre so the solver's
  // own rounding resolves it to the intended tier.
  TWIN_DAM_BRICK_TIERS.x.forEach((xTiers, dam) => {
    for (const xTier of xTiers) for (const yTier of TWIN_DAM_BRICK_TIERS.y) {
      seeds.push({
        x: -0.5 * scene.container.width_m + (xTier + 0.5) * brick,
        y: (yTier + 0.5) * brick,
        z: -0.5 * scene.container.depth_m + (TWIN_DAM_BRICK_TIERS.z[dam] + 0.5) * brick,
      });
    }
  });
  scene.fluid.initialBrickSeeds_m = seeds;
  delete scene.fluid.initialBrickSeedsAdditive;
  delete scene.fluid.inflow;
  return scene;
}


export function createOceanSeicheScene(): SceneDescription {
  const scene = sceneBody();
  scene.sceneId = "ocean-seiche";
  scene.rigidBodies = [];
  scene.container = { ...scene.container, width_m: 8.0, height_m: 2.4, depth_m: 2.0, fillFraction: 0.75, top: "closed", fluidWallMode: "no-slip" };
  scene.fluid.initialCondition = "tank-fill";
  // A long gravity wave has no meaningful capillary scale; keep the scene in
  // the same physical scope as the deep-water A/B preset.
  scene.fluid.surfaceTension_N_m = 0;
  delete scene.fluid.inflow;
  // The authored scene lattice gives exactly 320x96x80 cells of 0.025 m.
  scene.voxelDomain = { finestCellSize_m: 0.025, brickSize_cells: 8 };
  // The raised slab: brick tiers x {0,1}, y tier 9 (cells 72..79 — directly
  // on the 72-cell pool surface), and every z tier. Seeds are the world-space
  // centres of those 8-cubed bricks at the exact grid above.
  const h = 0.025, brick = 8 * h;
  const seeds: { x: number; y: number; z: number }[] = [];
  for (let zTier = 0; zTier < 10; zTier += 1) {
    const z = -1.0 + (zTier + 0.5) * brick;
    seeds.push({ x: -4.0 + 0.5 * brick, y: 9.5 * brick, z }, { x: -4.0 + 1.5 * brick, y: 9.5 * brick, z });
  }
  scene.fluid.initialBrickSeeds_m = seeds;
  scene.fluid.initialBrickSeedsAdditive = true;
  return scene;
}

const paperCamera: Partial<CameraState> = { distance_m: 2.45, target_m: { x: 0, y: 0.42, z: 0 } };
// Close and low: the pond fills the frame with the cloud trees and mushrooms
// crowding its banks, while the banks still occlude the far waterline so the
// water reads as inset into the ground.
const gardenCamera: Partial<CameraState> = { azimuth_rad: 0.58, elevation_rad: 0.38, distance_m: 2.95, target_m: { x: 0, y: 0.26, z: 0 } };

export function createGardenPondScene(): SceneDescription {
  const scene = applyGardenPool(sceneBody());
  scene.sceneId = "garden-pond-still";
  scene.fluid.initialCondition = "tank-fill";
  const terrain = gardenPoolTerrain();
  const beach = { x: 0.25, z: -0.55 };
  const stone = (index: number, x: number, z: number) => ({
    id: `garden-stone-${index}`, name: `Stepping stone ${index}`, shape: "cylinder" as const,
    dimensions_m: { x: 0.13, y: 0.06, z: 0.13 }, density_kg_m3: 2600,
    position_m: { x, y: terrainHeightAt(terrain, x, z) + 0.03, z },
    orientation: { w: 1, x: 0, y: 0, z: 0 },
    linearVelocity_m_s: { x: 0, y: 0, z: 0 }, angularVelocity_rad_s: { x: 0, y: 0, z: 0 },
    restitution: 0.05, friction: 0.9, motion: "static" as const
  });
  scene.rigidBodies = [
    {
      id: "garden-cork-ball", name: "Cork ball", shape: "sphere",
      dimensions_m: { x: 0.09, y: 0.09, z: 0.09 }, density_kg_m3: 240,
      position_m: { x: -0.35, y: GARDEN_WATERLINE_M + 0.25, z: -0.12 },
      orientation: { w: 1, x: 0, y: 0, z: 0 },
      linearVelocity_m_s: { x: 0.1, y: 0, z: 0.05 }, angularVelocity_rad_s: { x: 0, y: 1.2, z: 0 },
      restitution: 0.4, friction: 0.35
    },
    stone(1, beach.x - 0.18, beach.z - 0.1), stone(2, beach.x + 0.08, beach.z + 0.04), stone(3, beach.x + 0.34, beach.z + 0.16)
  ];
  return scene;
}

/** Dry acceptance scene for the sparse renderer; no fluid solver is created. */
export function createGardenSvoLightingScene(): SceneDescription {
  const scene = createGardenPondScene();
  scene.sceneId = "garden-svo-lighting-study";
  scene.systems = { ...scene.systems, fluid: false };
  scene.container.fillFraction = 0;
  scene.voxelDomain = { finestCellSize_m: 0.025, brickSize_cells: 8 };
  // The lamppost remains authored emissive geometry. Key, environment balance,
  // and display grade are the shared dry-SVO defaults, never a scene override.
  delete scene.fluid.initialBrickSeeds_m;
  delete scene.fluid.initialBrickSeedsAdditive;
  delete scene.fluid.inflow;
  // Remove the floating cork. Static stepping stones provide crisp contact
  // and penumbra references without relying on rigid/fluid coupling.
  scene.rigidBodies = scene.rigidBodies.filter(({ id }) => id !== "garden-cork-ball");
  // The study needs an actual fixture: the garden has trees, mushrooms and
  // pebbles but no lamp. It is added to *this scene's* scenery rather than to
  // the garden every scene shares, which is what a document-owned description
  // buys — the shared set used to carry the lamppost behind a check on this
  // scene's own id. Just inside the right bank, so the inverse-square pool
  // reaches the pond while the pole and cap read as a silhouette.
  return withSceneryNodes(sceneWithEnvironment(scene, "garden"), [{
    kind: "group", id: "lamppost", place: { position: { x: .32, y: 0, z: .24 }, anchor: "floor" },
    children: [
      { kind: "cylinder", id: "lamppost/base", group: "lamp-fixture", tags: ["lamppost", "fixture"], place: { position: { x: 0, y: .02, z: 0 } }, radius: .05, halfHeight: .02, material: { palette: "stone", value: .708 } },
      { kind: "cylinder", id: "lamppost/pole", group: "lamp-fixture", tags: ["lamppost", "fixture"], place: { position: { x: 0, y: .17, z: 0 } }, radius: .014, halfHeight: .13, material: { palette: "stone", value: .6 } },
      // The only saturated surface anywhere in the garden, and it is the light
      // itself rather than a painted one.
      { kind: "ellipsoid", id: "lamppost/lantern", group: "emissive-fixture", tags: ["lamppost", "lantern", "fixture", "light", "point-light"], place: { position: { x: 0, y: .35, z: 0 } }, radius: { x: .05, y: .06, z: .05 }, material: { colorLinear: [1, .48, .19], emission: 11 } },
      { kind: "cylinder", id: "lamppost/cap", group: "lamp-fixture", tags: ["lamppost", "fixture"], place: { position: { x: 0, y: .43, z: 0 } }, radius: .06, halfHeight: .012, material: { palette: "stone", value: .828 } },
    ],
  }]);
}

/**
 * The delta a WebGPU smoke lane runs the scene with, declared beside the scene
 * it is a delta *of*.
 *
 * Every one of these existed already, as a forked factory in the smoke catalog
 * that rebuilt the scene from `defaultScene` under the catalog's own name — and
 * so shipped with no scenery, and drifted apart from the scene it claimed to
 * be. Written here the difference is the only thing stated, a test can pin it
 * field by field, and the environment still arrives from the definition.
 *
 * `build` hands out a fresh document, so an `apply` mutates it the way every
 * factory above does.
 */
function smokeVariant(description: string, apply: (scene: SceneDescription) => void): SceneVariant {
  return { id: "gpu-smoke", description, apply: (scene) => { apply(scene); return scene; } };
}

/** Pin the outer step: a lane's `exactSteps`/`simulatedTime_s` contract is only
 * exact when the document cannot take a larger step than its fixed one. */
function pinStep(scene: SceneDescription, dt_s: number): void {
  scene.numerics.fixedDt_s = scene.numerics.maxDt_s = dt_s;
}


export function createHeroGardenHoseSceneWithSet(
  options: HeroGardenHoseOptions = {},
): SceneDescription {
  const scene = createHeroGardenHoseScene(options);
  // Appending rather than mutating matters: `scene.scenery` is composed fresh
  // per document, and `withHeroLayout` reads the same vessel and waterline the
  // factory just solved.
  if (!scene.scenery) return scene;
  const composed = withHeroLayout(scene);
  // And the shape lab's overrides once more, over the nodes the layout added —
  // see `lib/hero-garden-overrides.ts`. Idempotent on the ones already replaced.
  return { ...scene, scenery: { ...composed, nodes: [...applyHeroGardenNodeOverrides(composed.nodes)] } };
}

function createHeroGardenHoseFillingScene(lattice?: SceneLattice): SceneDescription {
  const scene = createHeroGardenHoseSceneWithSet({ ...lattice, water: true });
  // Compose props against the authored waterline before emptying the basin.
  // The fluid solver stays enabled so the hose supplies all water over time.
  scene.container.fillFraction = 0;
  return scene;
}

export type SparseCM12ComplexitySceneId =
  | "empty-16"
  | "full-16"
  | "cut-mixed-seam-16"
  | "double-x-32"
  | "planar-interface-16"
  | "first-split-bundle"
  | "recursive-split"
  | "frontier-create"
  | "frontier-advance"
  | "row-only-replace"
  | "retire-reuse"
  | "symmetric-2d"
  | "symmetric-3d"
  | "brick-quad"
  | "long-dam";

export type SparseCM12ComplexityScene = Readonly<{
  id: SparseCM12ComplexitySceneId;
  ordinal: number;
  title: string;
  introducedFeature: string;
  defaultSteps: number;
}>;

/** A geometric progression from small still tanks to moving water fronts. */
export const SPARSE_CM12_COMPLEXITY_SCENES: readonly SparseCM12ComplexityScene[] = Object.freeze([
  { id: "empty-16", ordinal: 0, title: "Empty 16³-cell world",
    introducedFeature: "an empty tank", defaultSteps: 2 },
  { id: "full-16", ordinal: 1, title: "Smallest fully wet quiescent world",
    introducedFeature: "a completely filled still tank", defaultSteps: 2 },
  { id: "cut-mixed-seam-16", ordinal: 2, title: "One-cell water seed",
    introducedFeature: "a small seed of water", defaultSteps: 2 },
  { id: "double-x-32", ordinal: 3, title: "Doubled x extent",
    introducedFeature: "a tank doubled along its x axis", defaultSteps: 2 },
  { id: "planar-interface-16", ordinal: 4, title: "Planar water surface",
    introducedFeature: "a flat half-filled tank", defaultSteps: 2 },
  { id: "first-split-bundle", ordinal: 5, title: "Water above a two-level floor",
    introducedFeature: "falling water above a stepped floor", defaultSteps: 8 },
  { id: "recursive-split", ordinal: 6, title: "Falling water block",
    introducedFeature: "a falling block deforming under gravity", defaultSteps: 40 },
  { id: "frontier-create", ordinal: 7, title: "Dam front",
    introducedFeature: "water advancing across a dry floor", defaultSteps: 160 },
  { id: "frontier-advance", ordinal: 8, title: "Longer front travel",
    introducedFeature: "a longer observation of the advancing water front", defaultSteps: 240 },
  { id: "row-only-replace", ordinal: 9, title: "Shallow tank motion",
    introducedFeature: "water motion in a shallow tank", defaultSteps: 40 },
  { id: "retire-reuse", ordinal: 10, title: "Extended tank motion",
    introducedFeature: "a longer observation of water motion", defaultSteps: 300 },
  { id: "symmetric-2d", ordinal: 11, title: "Two-dimensional symmetric expansion",
    introducedFeature: "symmetric expansion in a shallow tank", defaultSteps: 20 },
  { id: "symmetric-3d", ordinal: 12, title: "Three-dimensional D4 symmetric expansion",
    introducedFeature: "the production D4 correctness oracle", defaultSteps: 20 },
  { id: "brick-quad", ordinal: 13, title: "Brick-quad water motion",
    introducedFeature: "water crossing the four brick quadrants", defaultSteps: 20 },
  { id: "long-dam", ordinal: 14, title: "Long dam traversal",
    introducedFeature: "a water front traversing a long tank", defaultSteps: 30 },
]);

const SPARSE_CM12_LADDER_CELL_M = 0.05;

function createSparseCM12LadderBox(
  id: SparseCM12ComplexitySceneId,
  dimensions: readonly [number, number, number],
  fillFraction: number,
  options: Readonly<{ gravity?: boolean; dam?: readonly [number, number, number] }> = {},
): SceneDescription {
  const scene = sceneBody();
  scene.sceneId = `sparse-cm12-ladder-${id}`;
  scene.duration_s = 1;
  scene.rigidBodies = [];
  scene.container = {
    ...scene.container,
    width_m: dimensions[0] * SPARSE_CM12_LADDER_CELL_M,
    height_m: dimensions[1] * SPARSE_CM12_LADDER_CELL_M,
    depth_m: dimensions[2] * SPARSE_CM12_LADDER_CELL_M,
    fillFraction,
    top: "closed",
    fluidWallMode: "free-slip",
  };
  scene.voxelDomain = { finestCellSize_m: SPARSE_CM12_LADDER_CELL_M, brickSize_cells: 8 };
  scene.fluid.initialCondition = options.dam ? "dam-break" : "tank-fill";
  if (options.dam) {
    scene.fluid.initialDamBreakDimensions_m = {
      x: options.dam[0] * SPARSE_CM12_LADDER_CELL_M,
      y: options.dam[1] * SPARSE_CM12_LADDER_CELL_M,
      z: options.dam[2] * SPARSE_CM12_LADDER_CELL_M,
    };
  } else {
    delete scene.fluid.initialDamBreakDimensions_m;
  }
  delete scene.fluid.initialDamBreakOrigin_m;
  delete scene.fluid.initialBrickSeeds_m;
  delete scene.fluid.initialBrickSeedsAdditive;
  delete scene.fluid.initialLiquidVolumes;
  delete scene.fluid.inflow;
  scene.fluid.dynamicViscosity_Pa_s = 0;
  scene.fluid.surfaceTension_N_m = 0;
  if (options.gravity === false) scene.fluid.gravity_m_s2 = { x: 0, y: 0, z: 0 };
  scene.numerics.fixedDt_s = scene.numerics.maxDt_s = 1 / 30;
  return scene;
}

/** One construction route for both Dawn ladder runs and UI scene cards. */
export function createSparseCM12ComplexityScene(id: SparseCM12ComplexitySceneId): SceneDescription {
  switch (id) {
    case "empty-16":
      return createSparseCM12LadderBox(id, [16, 16, 16], 0, { gravity: false });
    case "full-16":
      return createSparseCM12LadderBox(id, [16, 16, 16], 1, { gravity: false });
    case "cut-mixed-seam-16":
      return createSparseCM12LadderBox(id, [16, 16, 16], 1 / 4096,
        { gravity: false, dam: [1, 1, 1] });
    case "double-x-32":
      return createSparseCM12LadderBox(id, [32, 16, 16], 1 / 4096,
        { gravity: false, dam: [2, 1, 1] });
    case "planar-interface-16":
      return createSparseCM12LadderBox(id, [16, 16, 16], 0.5, { gravity: false });
    case "first-split-bundle":
    case "recursive-split":
    case "row-only-replace":
      // These are deliberately the same physical scene observed for different
      // transactions and durations by the ladder.
      return createSparseCM12LadderBox("first-split-bundle", [32, 32, 16], 1 / 32,
        { gravity: true, dam: [8, 8, 16] });
    case "frontier-create":
    case "frontier-advance":
    case "retire-reuse":
      return createExternalVoxelPoolTransferScene(`sparse-cm12-ladder-${id}`);
    case "symmetric-2d": {
      const scene = createSparseCM12LadderBox(id, [32, 16, 16], 1 / 8,
        { gravity: true, dam: [8, 8, 16] });
      scene.container.depthBoundary = "symmetry";
      return scene;
    }
    case "symmetric-3d":
      return createSymmetricExpansionScene();
    case "brick-quad":
      return createBrickQuadDamBreakScene();
    case "long-dam":
      return createSparseCM12LongDamBreakScene();
  }
}

/**
 * The scene catalog.
 *
 * One list, one identity per scene, one accessor (`sceneDocument`). Order is
 * the order the library offers them within a shelf, and the first entry is what
 * a cold load opens on.
 *
 * `audience` is the load-bearing new field. "Comparisons" used to hold thirteen
 * analytic oracles beside the three paper figures, and a `<select>` showed them
 * all to whoever arrived — a third of the product's first impression was
 * ceiling-drop adhesion tests. They are how the physics is trusted and they
 * stay; they are simply the last section the page reads, not the first.
 */
export const SCENE_CATALOG: readonly SceneDefinition[] = Object.freeze([
  defineScene({
    id: "water-box-dam-break",
    name: "Water box · dam break",
    blurb: "A collapsing water column, staged: a bare floor, one lamp above it, and nothing else. The tank stands in the pool of light and the wave throws its own shadow across the boards. Drop bodies in and watch them take it.",
    audience: "explore",
    shelf: "Tanks",
    environment: "stage",
    build: () => {
      const scene = sceneBody();
      scene.rigidBodies = [];
      scene.fluid.initialDamBreakDimensions_m = { x: 0.6, y: 0.7, z: 0.5 };
      scene.container.fillFraction = 0.6 * 0.7 * 0.5
        / (scene.container.width_m * scene.container.height_m * scene.container.depth_m);
      return scene;
    },
    camera: studioStageCamera,
  }),
  defineScene({
    id: CHURNING_TANK_SCENE_ID,
    name: "Churning tank",
    blurb: "A broad spatula vigorously churns a square tank of water, circling every two seconds with its blade facing into the flow. Water resolves at 2.5 cm. Watch a full 20-second session: one second at rest, two seconds accelerating, ten seconds churning, three seconds slowing down, and four seconds settling. Press Play to start.",
    audience: "explore", shelf: "Tanks", environment: "stage",
    build: createChurningTankScene,
    camera: { distance_m: 5.2, target_m: { x: 0, y: 0.65, z: 0 }, elevation_rad: 0.62, azimuth_rad: 0.7 },
    methodProfile: { methodId: "uniform-narrow-band-flip", quality: "balanced",
      overrides: { rigidCoupling: "on" } },
  }),
  ...([false, true] as const).map(wet => defineScene({
    id: wet ? "interactive-pool" : "interactive-playground",
    name: wet ? "Interactive pool" : "Interactive playground",
    blurb: "A controllable character, a crate, a switch, a lift and a trigger. Entity controls are supplied by the interactive editor.",
    audience: "explore", shelf: "Interactive", environment: "stage",
    build: () => createInteractivePlayground(wet),
    camera: { target_m: { x: 0, y: 1, z: 0 }, distance_m: 10 },
    methodProfile: wet ? { methodId: "uniform-narrow-band-flip", quality: "balanced", overrides: { timeStep: "scene", rigidCoupling: "on" } } : undefined,
  })),
  ...(["dam-break", "settled-tank", "hose-fill"] as const).map(mode => defineScene({
    id: `uniform-trough-${mode}`,
    name: `Voxel bath · ${mode === "dam-break" ? "dam break" : mode === "settled-tank" ? "settled tank" : "hose fill"}`,
    blurb: mode === "dam-break"
      ? "A reservoir collapses inside a voxel bath with rounded corners and curved, tapered sides. Uniform geometric transport on a 128×48×48 grid at 2.5 cm spacing."
      : mode === "settled-tank"
        ? "The same curved voxel bath and dam-break water volume at rest, with a flat waterline. A uniform-geometric hydrostatic stability scene on a 256×96×96 grid at 1.25 cm spacing."
        : "The same curved voxel bath starts completely empty on a 128×48×48 grid. A horizontal hose runs for 12 seconds, followed by eight seconds of settling, using uniform-geometric.",
    audience: "explore",
    shelf: "Uniform geometric · 3D",
    environment: "stage",
    presentationMode: "full-scene",
    containerShell: "authored",
    build: () => createUniformTroughScene(mode),
    buildAt: lattice => createUniformTroughScene(mode, lattice.cellSize_m),
    camera: UNIFORM_TROUGH_CAMERA,
  })),
  defineScene({
    id: "gentle-moving-blob",
    name: "Gentle moving blob",
    blurb: "A fully contained 0.3 m sphere drifts gently sideways through air while the sparse grid adapts around it. Zero gravity, viscosity, surface tension, sources and moving solids isolate transport shape loss.",
    audience: "validation",
    shelf: "Transport lab",
    environment: "stage",
    build: createGentleMovingBlobScene,
    camera: { distance_m: 2.5, target_m: { x: -0.05, y: 0.6, z: 0 } },
  }),
  defineScene({
    id: "water-box-tank-fill",
    name: "Water box · settled tank",
    blurb: "The same container starting from a settled fill; drop bodies into calm water.",
    audience: "explore",
    shelf: "Tanks",
    environment: "bathhouse",
    build: () => {
      const scene = sceneBody();
      scene.sceneId = "interactive-water-box-settled";
      // Catalog tanks use a physical ceiling; author dropped bodies inside it.
      scene.container.top = "closed";
      for (const body of scene.rigidBodies) body.position_m.y = 0.6;
      scene.fluid.initialCondition = "tank-fill";
      return scene;
    },
    variants: {
      "gpu-smoke": smokeVariant(
        "Hydrostatic equilibrium alone: no bodies to stir it, no capillary force, "
        + "a 70% fill deep enough to hold a still interior, and a pinned 1/120 s step.",
        (scene) => {
          scene.rigidBodies = [];
          scene.container.fillFraction = 0.7;
          scene.fluid.surfaceTension_N_m = 0;
          pinStep(scene, 1 / 120);
        }),
    },
  }),
  defineScene({
    id: "bounded-pool-transfer",
    name: "Water transfer · bounded pool",
    blurb: "A compact reservoir drains through a floor-level opening in a voxel dam. A wider closed voxel pool catches the jet while fluid pages appear only along the advancing front.",
    audience: "explore",
    shelf: "Tanks",
    environment: "stage",
    build: createBoundedPoolTransferScene,
    camera: {
      azimuth_rad: 0.18,
      elevation_rad: 0.48,
      distance_m: 2.15,
      target_m: { x: -0.02, y: 0.2, z: 0 },
    },
  }),
  defineScene({
    id: "high-resolution-dam-break",
    name: "Dam break · high resolution 128³",
    blurb: "A 0.8 m analytic water column resolved on a 128³ lattice, carried on the sparse 4³/8³ brick atlas so the full frame budget stays on simulation and rendering.",
    audience: "explore",
    shelf: "Tanks",
    environment: "stage",
    build: createHighResolutionDamBreakScene,
    camera: { distance_m: 1.9, target_m: { x: 0, y: 0.3, z: 0 } },
  }),
  defineScene({
    id: "twin-dam-collision",
    name: "Twin dams · corner collision",
    blurb: "A wide tank with a reservoir on each diagonally opposite floor corner. Both release at once, run 1.2 m down the long axis, and meet mid-tank at an angle.",
    audience: "explore",
    shelf: "Tanks",
    environment: "stage",
    build: createTwinDamCollisionScene,
    camera: { azimuth_rad: 0.5, elevation_rad: 0.32, distance_m: 3.2, target_m: { x: 0, y: 0.2, z: 0 } },
    variants: {
      "gpu-smoke": smokeVariant(
        "The collision measured without capillary force, at the 4 ms step the oblique "
        + "front needs to stay inside the lane's component-CFL bound.",
        (scene) => {
          scene.fluid.surfaceTension_N_m = 0;
          pinStep(scene, 0.004);
        }),
    },
  }),
  defineScene({
    id: "ocean-seiche",
    name: "Ocean · rolling wave",
    blurb: "A broad 8 m tank of deep calm water; a raised slab along one wall releases a long wave that ripples across and reflects. The quiescent deep interior rides coarse 4³ bricks while the moving surface is promoted to 8³.",
    audience: "explore",
    shelf: "Open water",
    // The station's circular pressure-hull ribs scale from the largest scene
    // axis. This tank is deliberately long and shallow, so those ribs become
    // enormous low arcs behind the glass and their refraction reads as a hole
    // through the deep water. Keep this volume against the uncluttered room;
    // the wave and its macro-rung transitions remain the only silhouettes.
    environment: "stage",
    build: createOceanSeicheScene,
    camera: { azimuth_rad: 0.35, elevation_rad: 0.32, distance_m: 9.0, target_m: { x: 0, y: 1.1, z: 0 } },
    variants: {
      "gpu-smoke": smokeVariant(
        "A 5 ms outer step: the wave-profile lane observes six seconds, and the "
        + "browser's 8 ms cap would let the 4.2 m/s front cross more than a cell a step.",
        (scene) => pinStep(scene, 0.005)),
    },
  }),
  defineScene({
    id: "tall-cells-hillside-dam-break",
    name: "Tall cells · hillside dam break",
    blurb: "A paper-inspired reservoir releases from a 4 m shelf, over a 46-degree crest that eases into a 4 m runout. Height samples populate one unified voxel solid; the 1.8-million-cell logical world starts with only a 3×2×6-tile water body so the travelling front exercises dynamic sparse-tile creation and retirement.",
    audience: "study",
    shelf: "Tall-cell studies",
    environment: "stage",
    presentationMode: "full-scene",
    build: createTallCellsHillsideDamBreakScene,
    // Aimed at the drop rather than at the domain.
    //
    // The azimuth and elevation are unchanged — a three-quarter view from off
    // the downhill end, so the front comes towards the lens with the slope
    // modelled by the sun — but the subject is no longer a 1.9 m fall. The
    // target sits at mid-drop (the shelf is 4 m, the toe 0.1 m) instead of near
    // the old shelf's foot, which was low enough on this hillside to put the
    // reservoir off the top of the frame.
    //
    // The distance moves barely at all, and that is a fact about the framing
    // rather than an oversight: what binds this shot is the 12.8 m travel axis
    // seen close to side-on, not the height. The minimum enclosing distance for
    // the whole terrain-plus-reservoir box goes from 9.6 m to 9.9 m at a 1.4
    // aspect, so 15.0 m holds the same ~1.5x of air around the subject that
    // 14.5 m did before.
    camera: {
      azimuth_rad: 0.48,
      elevation_rad: 0.3,
      distance_m: 15.0,
      target_m: { x: 0, y: 2.2, z: 0 },
    },
  }),
  defineScene({
    id: BACKYARD_POOL_PARTY_ID,
    name: "Backyard pool party",
    blurb: "A sunlit backyard pool with Bub, a little controllable robot. Explore the patio, hold Space at the ladder to climb, then run and somersault off the diving board with F.",
    audience: "explore",
    shelf: "Garden",
    environment: "garden",
    containerShell: "authored",
    presentationMode: "full-scene",
    build: createBackyardPoolPartyScene,
    camera: backyardPoolPartyCamera,
  }),
  defineScene({
    id: FOREST_WATERFALL_ID,
    name: "Stillwater Grove · forest waterfall",
    blurb: "A nineteen-metre waterfall cascades over mossy rock ledges into a broad forest pool. Living water and splashes come from the narrow-band FLIP solver, surrounded by Blender-authored woodland.",
    audience: "explore",
    shelf: "Garden",
    environment: "garden",
    containerShell: "authored",
    presentationMode: "full-scene",
    methodProfile: NBFLIP_METHOD_PROFILE,
    renderProfile: { environmentRefinementDepth: 0 },
    build: createForestWaterfallScene,
    camera: forestWaterfallCamera,
  }),
  defineScene({
    id: BACKYARD_POOL_DRIFT_ID,
    name: "Backyard pool · gentle drift",
    blurb: "The same backyard pool barely disturbed: the ball floats and drifts slowly across the water, and a single drip lands at the far end. A gentle wake and weak ripples spread over an otherwise still pool.",
    audience: "explore",
    shelf: "Garden",
    environment: "garden",
    containerShell: "authored",
    presentationMode: "full-scene",
    build: createBackyardPoolDriftScene,
    camera: backyardPoolPartyCamera,
  }),
  defineScene({
    id: "hero-garden-hose",
    name: "Porcelain pond · hose filling",
    blurb: "An empty raised porcelain pond gradually filling from a garden hose. The coping, the basin and the ground are one generated heightfield, so the water meets the rim exactly where the rim is.",
    audience: "explore",
    shelf: "Garden",
    environment: "garden",
    // The set *is* the scene here — a generated heightfield vessel, its coping,
    // its beds and its props — alongside the pond water and hose inflow.
    presentationMode: "full-scene",
    // The set is composed here rather than inside `createHeroGardenHoseScene`
    // because the layout depends on the vessel — it seats every prop through
    // `pondVesselHeightAt` and runs its pebble beds along `pondVesselPlanCurve`
    // — and a scene module that imported it back would close a cycle that dies
    // in the temporal dead zone. Vessel, then set, then catalog is the one
    // ordering with no arrow pointing backwards. Appending rather than mutating
    // matters too: `scene.scenery` is a module-level constant shared by every
    // document the factory produces.
    build: () => createHeroGardenHoseFillingScene(),
    // The one route to this scene at another lattice, and the reason it is a
    // member of the definition rather than a call somewhere: the terrain bake,
    // the layout composed against it and every generator's legibility floor are
    // resolved while the document is built, so the sizes have to arrive before
    // it exists. `tools/run-svo-dry-render-smoke.ts` already reached the factory
    // directly for `FLUID_SVO_DRY_SMOKE_REFINEMENT`; this is the same call, from
    // the side the product loads scenes on.
    buildAt: (lattice: SceneLattice) => createHeroGardenHoseFillingScene(lattice),
    camera: heroGardenCamera,
    // No GPU lane yet, and so deliberately no variant: an authored variant that
    // no lane claims is a fork by another name. The settling lane returns with
    // the inner-face fix that phase 1 is currently blocked on — see
    // docs/HERO_GARDEN_HOSE_SCENE_PLAN.md.
  }),
  defineScene({
    id: "hero-garden-hose-x10",
    name: "Porcelain pond · ten times the set",
    blurb: "The filled porcelain pond with miniature birch, pine and willow, plus azalea, juniper and dense boxwood and hebe bushes around the original oak.",
    // The catalog presents the filled pond; dry render measurements use the
    // same stress factory with its default water-off setting.
    audience: "validation",
    shelf: "Garden",
    environment: "garden",
    // Present the authored pond water from the first frame. The stress factory
    // remains dry by default for render-only benchmark lanes.
    presentationMode: "full-scene",
    // `heroGardenCamera`, unchanged and deliberately so — W3's gate is stated
    // against an *unchanged camera*, and a stress scene that reframed itself
    // could not discharge it. The factory composes the blocking layout itself,
    // so unlike the hero above there is nothing to append here.
    build: () => createHeroGardenHoseStressScene({
      recordMultiplier: HERO_GARDEN_STRESS_MAXIMUM_MULTIPLIER,
      detailCellSize_m: HERO_GARDEN_POND_CELL_M,
      water: true,
      pondTank: true,
    }),
    buildAt: (lattice: SceneLattice) => createHeroGardenHoseStressScene({
      recordMultiplier: HERO_GARDEN_STRESS_MAXIMUM_MULTIPLIER,
      ...lattice,
      detailCellSize_m: lattice.detailCellSize_m ?? Math.min(lattice.cellSize_m, HERO_GARDEN_POND_CELL_M),
      water: true,
      pondTank: true,
    }),
    camera: heroGardenCamera,
  }),
  defineScene({
    id: HERO_GARDEN_BLENDER_ID,
    name: "Porcelain pond · Blender garden",
    blurb: "A colorful Blender garden: green and silver sage planting, warm limestone, a porcelain pond and a brass hose nozzle around finely resolved living water.",
    audience: "explore",
    shelf: "Garden",
    environment: "garden",
    containerShell: "authored",
    presentationMode: "full-scene",
    build: createHeroGardenBlenderScene,
    buildAt: createHeroGardenBlenderScene,
    camera: heroGardenBlenderCamera,
  }),
  defineScene({
    id: "garden-pond",
    name: "Garden pond · still water",
    blurb: "A white-clay pond settled to its waterline, ringed by cloud trees and oversized mushrooms. A cork ball bobs over the deep end; stepping stones cross the beach shelf.",
    audience: "explore",
    shelf: "Garden",
    environment: "garden",
    presentationMode: "full-scene",
    build: createGardenPondScene,
    camera: gardenCamera,
    variants: {
      "gpu-smoke": smokeVariant(
        "Rest on terrain with nothing floating in it: the cork ball and stepping stones "
        + "are removed so a residual speed is the pond's, not a body's, with no capillary "
        + "force and a pinned 1/120 s step.",
        (scene) => {
          scene.rigidBodies = [];
          scene.fluid.surfaceTension_N_m = 0;
          pinStep(scene, 1 / 120);
        }),
    },
  }),
  defineScene({
    id: "garden-dam-break",
    name: "Garden pond · dam break",
    blurb: "One resident fluid brick releases on the upper lawn, vacates its source region, and activates neighbouring bricks as it washes into the pond.",
    audience: "explore",
    shelf: "Garden",
    environment: "garden",
    presentationMode: "full-scene",
    build: () => {
      const scene = applyGardenPool(createPaperScenario("dam-break-boxes"), { fillFraction: 0.16 });
      scene.sceneId = "garden-pond-dam-break";
      scene.fluid.initialBrickSeeds_m = [{ ...GARDEN_DAM_BRICK_SEED_M }];
      return scene;
    },
    camera: gardenCamera,
    variants: {
      "gpu-smoke": smokeVariant(
        "The brick-migration lane attributes residency to the water alone, so the paper "
        + "box stack this scene inherits is removed; no-slip banks and a 4 ms step keep "
        + "the released body's arrival where the hook's core-brick counts were measured.",
        (scene) => {
          scene.rigidBodies = [];
          scene.container.fluidWallMode = "no-slip";
          pinStep(scene, 0.004);
        }),
    },
  }),
  defineScene({
    id: "garden-hose",
    name: "Garden pond · hose fill",
    blurb: "The pond starts as a puddle in the deep end and a hose arcs water in until the banks fill to the waterline.",
    audience: "explore",
    shelf: "Garden",
    environment: "garden",
    presentationMode: "full-scene",
    build: () => {
      const scene = applyGardenPool(createPaperScenario("hose-tank"), { fillFraction: 0.08 });
      scene.sceneId = "garden-pond-hose-fill";
      if (scene.fluid.inflow) scene.fluid.inflow.end_s = 30;
      return scene;
    },
    camera: gardenCamera,
  }),
  defineScene({
    id: "mass-conserving-figure-9-dam-break",
    name: "CM12 Figure 9 · dam phase",
    blurb: "Published 128×128×64, dx=.05 m, dt=1/30 s parameters; explicit reconstructed dam geometry for Dawn conformance.",
    audience: "study",
    shelf: "Paper figures",
    environment: "night-lab",
    build: () => createMassConservingFigure9DamBreak(),
    camera: paperCamera,
  }),
  defineScene({
    id: "hose-tank",
    name: "Hose-filled tank",
    blurb: "Paper-inspired legacy demo · a continuous jet fills a shallow tank.",
    audience: "study",
    shelf: "Paper figures",
    environment: "conservatory",
    presentationMode: "full-scene",
    build: () => createPaperScenario("hose-tank"),
    camera: paperCamera,
  }),
  defineScene({
    id: "dam-break-boxes",
    name: "Dam break + box stack",
    blurb: "Paper-inspired legacy demo · a dam break strikes a stack of rigid boxes.",
    audience: "study",
    shelf: "Paper figures",
    environment: "concrete-gallery",
    build: () => createPaperScenario("dam-break-boxes"),
    camera: paperCamera,
  }),
  defineScene({
    id: "sphere-jet",
    name: "Jet past sphere",
    blurb: "Paper-inspired legacy demo · an inlet jet flows past a static sphere.",
    audience: "study",
    shelf: "Paper figures",
    environment: "night-lab",
    presentationMode: "full-scene",
    build: () => createPaperScenario("sphere-jet"),
    camera: paperCamera,
  }),
  // Chentanez & Muller 2012, every figure that is a simulation. Figure 4 is a
  // 1D schematic of the sharpening scheme and Figure 10 is a plot, so neither
  // is a scene. See lib/cm12-paper-scenes.ts for what each entry reproduces
  // exactly and what it reconstructs.
  ...CM12_FIGURES.map((figure) => {
    const grid = cm12Grid(figure);
    const twoDimensional = figure.grid[2] === undefined;
    const published = twoDimensional
      ? `${figure.grid[0]}\u00d7${figure.grid[1]} (2D, run as a ${grid[2]}-cell slab)`
      : grid.join("\u00d7");
    return defineScene({
      id: figure.id,
      name: `CM12 Figure ${figure.figure} \u00b7 ${figure.name}`,
      blurb: `${figure.blurb} Published ${published} cells at dx=.05 m, dt=1/30 s, g=10 m/s\u00b2, D=2.1${
        figure.cfl === undefined ? "" : `; CFL ${figure.cfl}`}.`,
      audience: "study",
      shelf: "CM12 paper figures",
      environment: "stage",
      build: () => cm12Scene(figure.id),
      camera: cm12Camera(cm12Scene(figure.id)),
    });
  }),
  // Ferstl et al. 2016, Figure 8. See lib/core/nbflip-paper-scenes.ts for what
  // is published (the grid) and what is measured from the paper's video.
  defineScene({
    id: NBFLIP_LETTERS_SCENE_ID,
    name: "NB-FLIP Figure 8 \u00b7 Letters",
    blurb: `Letter-shaped drops A to I fall into a basin, one every ${NBFLIP_LETTERS_INTERVAL_FRAMES} frames. Published ${NBFLIP_LETTERS_GRID.join("\u00d7")} cells; pool depth, letterforms, placement and the 24 fps schedule are reconstructed from the paper's video.`,
    audience: "study",
    shelf: "Paper figures",
    environment: "stage",
    methodProfile: NBFLIP_LETTERS_METHOD_PROFILE,
    build: createNbflipLetters,
    camera: nbflipLettersCamera(),
  }),
  defineScene({
    id: NBFLIP_POUR_SCENE_ID,
    name: "NB-FLIP Figure 7 \u00b7 Pour",
    blurb: `A stream from an inclined pipe pours into an empty glass for ${NBFLIP_POUR_STREAM_FRAMES} frames. Published ${NBFLIP_POUR_GRID.join("\u00d7")} cells; the glass, stream and camera are reconstructed from the paper's video, whose glass fills twice as far as this stream carries.`,
    audience: "study",
    shelf: "Paper figures",
    environment: "stage",
    methodProfile: NBFLIP_METHOD_PROFILE,
    build: createNbflipPour,
    preview: createNbflipPourPreview,
    camera: nbflipPourCamera(),
  }),
  defineScene({
    id: NBFLIP_TEASER_SCENE_ID,
    name: "NB-FLIP Figure 9 \u00b7 Teaser",
    blurb: `A dam breaks from the back corner of a pool onto a diagonal row of ${NBFLIP_TEASER_CYLINDER_COUNT} cylinders. Published ${NBFLIP_TEASER_GRID.join("\u00d7")} cells; the pool, dam, cylinders and camera are reconstructed from the paper's video.`,
    audience: "study",
    shelf: "Paper figures",
    environment: "stage",
    methodProfile: { ...NBFLIP_METHOD_PROFILE, overrides: {} },
    build: createNbflipTeaser,
    camera: nbflipTeaserCamera(),
  }),
  defineScene({
    id: NBFLIP_DAM_SCENE_ID,
    name: "NB-FLIP Figure 10 \u00b7 Dam",
    blurb: `A block of liquid collapses down a shallow channel through ${NBFLIP_DAM_CYLINDERS.length} thin cylinders and up the end wall. Published ${NBFLIP_DAM_GRID.join("\u00d7")} cells; the block, sheet, cylinders, camera and the 120 Hz step are reconstructed from the paper's video.`,
    audience: "study",
    shelf: "Paper figures",
    environment: "stage",
    methodProfile: NBFLIP_METHOD_PROFILE,
    build: createNbflipDam,
    camera: nbflipDamCamera(),
  }),
  defineScene({
    id: NBFLIP_WAVES_SCENE_ID,
    name: "NB-FLIP Figure 4 \u00b7 Oscillating Surface",
    blurb: `A mound on a shallow pool collapses and sloshes to rest over 500 frames: the paper's test of its particle-to-grid velocity blend. Published ${NBFLIP_WAVES_GRID.join("\u00d7")} cells; the pool, mound and camera are reconstructed from the paper's video.`,
    audience: "study",
    shelf: "Paper figures",
    environment: "stage",
    methodProfile: NBFLIP_METHOD_PROFILE,
    build: createNbflipWaves,
    camera: nbflipWavesCamera(),
  }),
  defineScene({
    id: NBFLIP_SIMPLE_DAM_SCENE_ID,
    name: "NB-FLIP Video \u00b7 Simple Breaking Dam",
    blurb: `A corner block collapses across a shallow basin, runs up the far wall to the ceiling and falls back. ${NBFLIP_SIMPLE_DAM_GRID.join("\u00d7")} cells, from the paper's video alone: it is in no figure.`,
    audience: "study",
    shelf: "Paper figures",
    environment: "stage",
    methodProfile: NBFLIP_METHOD_PROFILE,
    build: createNbflipSimpleDam,
    camera: nbflipSimpleDamCamera(),
  }),
  defineScene({
    id: NBFLIP_STAIR_CUBE_ID,
    name: "NB-FLIP · Stair Cube",
    blurb: "Clear water sloshes inside an invisible sealed vessel as it tips down three charcoal beams onto a gridded floor. An eight-second reconstruction of Ballistic Simulations’ APIC test.",
    audience: "explore", shelf: "Tanks", environment: "stage",
    renderProfile: { fluidSurfaceRenderMode: "shaded", fluidParticleDisplay: "off" },
    methodProfile: { ...NBFLIP_METHOD_PROFILE, overrides: { ...NBFLIP_METHOD_PROFILE.overrides, rigidCoupling: "on" } },
    build: createNbflipStairCube, camera: nbflipStairCubeCamera,
  }),
  // Not a paper scene: the same closed box and method, with liquid that starts
  // as nothing and grows. See lib/core/nbflip-water-slide-scene.ts.
  defineScene({
    id: NBFLIP_SLIDE_SCENE_ID,
    name: "NB-FLIP · Water Slide",
    blurb: `A hose runs for ${NBFLIP_SLIDE_STREAM_SECONDS} seconds down a continuous slide that winds inward for ${NBFLIP_SLIDE_TURNS} turns and spills into a pool at its centre, where the water comes to rest. The box starts empty: ${NBFLIP_SLIDE_GRID.join("×")} cells at 2 cm.`,
    audience: "explore",
    shelf: "Tanks",
    environment: "stage",
    containerShell: "authored",
    methodProfile: NBFLIP_METHOD_PROFILE,
    build: createNbflipWaterSlide,
    preview: createNbflipWaterSlidePreview,
    camera: nbflipWaterSlideCamera(),
  }),
  defineScene({
    id: NBFLIP_SLIDE_LONG_SCENE_ID,
    name: "NB-FLIP · Long Water Slide",
    blurb: `A hose runs for ${NBFLIP_SLIDE_LONG_STREAM_SECONDS} seconds down a continuous flume that snakes through ${NBFLIP_SLIDE_LONG_BENDS} alternating bends and winds once around a pool, where the water comes to rest. The box starts empty and stays mostly air: ${NBFLIP_SLIDE_LONG_GRID.join("×")} cells at 2 cm.`,
    audience: "explore",
    shelf: "Tanks",
    environment: "stage",
    methodProfile: NBFLIP_METHOD_PROFILE,
    build: createNbflipWaterSlideLong,
    preview: createNbflipWaterSlideLongPreview,
    camera: nbflipWaterSlideLongCamera(),
  }),
  defineScene({
    id: NBFLIP_THEME_PARK_SCENE_ID,
    name: "NB-FLIP · Theme Park Water Slide",
    blurb: `A white-clay water park with oak, birch, pine, willow and low shrubs, visitors, food stalls, a toilet block and picnic terraces around an elevated two-turn spiral and splash pool. The hose runs for ${NBFLIP_THEME_PARK_STREAM_SECONDS} seconds: ${NBFLIP_THEME_PARK_GRID.join("×")} cells at 2 cm, starting empty.`,
    audience: "explore",
    shelf: "Tanks",
    environment: "stage",
    methodProfile: NBFLIP_METHOD_PROFILE,
    build: createNbflipThemeParkSlide,
    presentationMode: "full-scene",
    camera: nbflipThemeParkCamera(),
  }),
  defineScene({
    id: NBFLIP_OPEN_GROUND_SCENE_ID,
    name: "NB-FLIP · Open ground",
    blurb: `A water column ${NBFLIP_OPEN_GROUND_COLUMN * NBFLIP_OPEN_GROUND_CELL_SIZE_M} m tall and ${NBFLIP_OPEN_GROUND_SIDE * NBFLIP_OPEN_GROUND_CELL_SIZE_M} m square released on open ground: no walls anywhere, so it floods out on every side and the solver claims pages wherever it goes. The page pool is a ${NBFLIP_OPEN_GROUND_MEMORY_MIB} MiB budget; running out of it is a named fatal. Cells of ${NBFLIP_OPEN_GROUND_CELL_SIZE_M * 100} cm.`,
    audience: "validation",
    shelf: "Tanks",
    environment: "stage",
    methodProfile: NBFLIP_METHOD_PROFILE,
    build: createNbflipOpenGround,
    camera: nbflipOpenGroundCamera(),
  }),
  defineScene({
    id: NBFLIP_OPEN_GROUND_FACES_SCENE_ID,
    name: "NB-FLIP · Open ground, column at the faces",
    blurb: "The open-ground column laid against its container's faces: the container is the column's footprint. Liquid against a face of an open world's container is no different from liquid inside it; nothing is lost at the first step.",
    audience: "validation",
    shelf: "Tanks",
    environment: "stage",
    methodProfile: NBFLIP_METHOD_PROFILE,
    build: () => createNbflipOpenGround(NBFLIP_OPEN_GROUND_STEP_HZ, "faces"),
    camera: nbflipOpenGroundCamera(),
  }),
  defineScene({
    id: NBFLIP_OPEN_GROUND_SLOPE_SCENE_ID,
    name: "NB-FLIP · Open ground on a slope",
    blurb: "The open-ground column released on the flank of a wide analytic mound: the ground rises gently to -x past the container, and the water runs down it and floods out over the plane.",
    audience: "validation",
    shelf: "Tanks",
    environment: "stage",
    methodProfile: NBFLIP_METHOD_PROFILE,
    build: () => createNbflipOpenGround(NBFLIP_OPEN_GROUND_STEP_HZ, "slope"),
    camera: nbflipOpenGroundCamera(),
  }),
  defineScene({
    id: NBFLIP_SLOPE_CLEARANCE_SCENE_ID,
    name: "NB-FLIP · Slope clearance",
    blurb: "A small 3.072 L body is released from rest just above a continuous 1:16 incline and drains into a lower catch basin. Target: the entire slope dry within 6 simulated seconds, with the water retained below. A simple test of bed gradients and lingering residue; 128×32×32 cells at 2 cm.",
    audience: "validation",
    shelf: "Thin liquid",
    environment: "stage",
    methodProfile: NBFLIP_METHOD_PROFILE,
    build: createNbflipSlopeClearance,
    camera: nbflipRampCamera(),
  }),
  // The slide's slope on its own, as measured probes for what a gentle voxel
  // ramp leaves behind. See lib/core/nbflip-ramp-scene.ts.
  ...NBFLIP_RAMP_VARIANTS.map(variant => {
    const { run, stream } = nbflipRampShape(variant), size = `${NBFLIP_RAMP_GRID.join("×")} cells at 2 cm`;
    const drains = `What is still on the ramp ${NBFLIP_RAMP_DRAIN_SECONDS} seconds after the stream stops is residue`;
    return defineScene({
      id: nbflipRampSceneId(variant),
      name: `NB-FLIP ramp · ${variant}`,
      blurb: variant === "channel" ? `A stream runs for ${stream!.seconds} seconds down a walled voxel ramp falling one cell in ${run}, the water slide's slope, into a pit. ${drains}: a real ramp would be dry. ${size}.`
        : variant === "staircase" ? `A stream runs for ${stream!.seconds} seconds down an open voxel ramp falling one cell in ${run}, into a pit. ${drains}: a real ramp would be dry. ${size}.`
        : variant === "flat" ? `The staircase's stream on a level floor: the control. Nothing drains a level floor, so what it holds ${NBFLIP_RAMP_DRAIN_SECONDS} seconds after the stream stops must not change when a slope does. ${size}.`
        : `The channel's treads dammed at their foot under still water, ${NBFLIP_RAMP_BASIN_LEVEL - nbflipRampBed(variant, 0)} cells over the top one and ${NBFLIP_RAMP_BASIN_LEVEL - nbflipRampBed(variant, NBFLIP_RAMP_LENGTH - 1)} over the lowest: the guard. It must stay still and level. ${size}.`,
      audience: "validation",
      shelf: "Thin liquid",
      environment: "stage",
      methodProfile: NBFLIP_METHOD_PROFILE,
      build: () => createNbflipRamp(variant),
      camera: nbflipRampCamera(),
    });
  }),
  defineScene({
    id: "cm12-figure-7-256",
    name: "CM12 Figure 7 · Ball drop into an empty tank · 256³",
    blurb: "Figure 7 at twice the linear resolution: the same 6.4 m tank and 1 m radius ball on 256³ cells at dx=0.025 m, dt=1/30 s and g=10 m/s². Uniform Geometric performance scene; the paper uses 128³.",
    audience: "study",
    shelf: "Method comparisons",
    environment: "stage",
    build: createCm12Figure7_256,
    camera: cm12Camera(createCm12Figure7_256()),
  }),
  // Thin-liquid ladders (lib/core/thin-liquid-scenes.ts): splash-sized bodies
  // in free fall, one size per bin, for Uniform Geometric's thin-feature loss.
  ...([
    ["thin-droplet-ladder", "Thin liquid · droplet ladder", "Six drops of radius 1 to 4 cells fall together into a 6-cell pool. Until impact each should keep its shape and fall as y0 − gt²/2; lost level set before impact is numerical.", createThinDropletLadderScene],
    ["thin-sheet-ladder", "Thin liquid · falling sheets", "Horizontal sheets 0.5 to 3 cells thick fall face-down into a 6-cell pool, resampled across vertex planes every step like a Figure 9 crown sheet. Exact: rigid free fall until impact.", createThinSheetLadderScene],
    ["thin-wall-films", "Thin liquid · wall films", "Vertical films 0.5 to 3 cells thick slide down the free-slip −x wall beside free-standing twins. Both should fall unchanged; any difference between the twins is the wall treatment's.", createThinWallFilmScene],
  ] as const).map(([id, name, blurb, build]) => defineScene({
    id, name, blurb,
    audience: "validation",
    shelf: "Thin liquid",
    environment: "stage",
    build,
    camera: cm12Camera(build()),
  })),
  defineScene({
    id: "deep-water-ab",
    name: "Deep-water A/B",
    blurb: "20 m tank at 80% fill, 1/30 s paper step, σ = 0 · isolates the grid method.",
    audience: "study",
    shelf: "Method comparisons",
    environment: "research-station",
    build: () => {
      const scene = sceneBody();
      scene.sceneId = "deep-water-grid-comparison";
      scene.container.height_m = 20;
      scene.container.fillFraction = 0.8;
      scene.fluid.initialCondition = "tank-fill";
      // The tall-cell paper does not include a capillary-force discretization.
      // Keep the A/B preset within that shared physical scope so the grid and
      // pressure methods are the only variables in the comparison.
      scene.fluid.surfaceTension_N_m = 0;
      scene.numerics.fixedDt_s = 1 / 30;
      scene.numerics.maxDt_s = scene.numerics.fixedDt_s;
      scene.rigidBodies = [];
      return scene;
    },
    variants: {
      "gpu-smoke": smokeVariant(
        "The equilibrium lane runs one paper step per outer step: the browser's 4x GPU "
        + "cap would let a 20 m column settle four times as far between observations.",
        (scene) => pinStep(scene, scene.numerics.fixedDt_s)),
    },
  }),
  defineScene({
    id: "water-box-dam-break-slab",
    name: "Dam break · 2D slab",
    blurb: "The water box dam break with its depth cut to one 8-cell brick and the reservoir spanning it, so every plane through the tank is the 2D case the advance lab solves. Free-slip walls and σ = 0 leave dimensionality as the only difference.",
    audience: "study",
    shelf: "Method comparisons",
    environment: "stage",
    build: createDamBreakSlabScene,
    // Square on to the slab: the subject is one x-y plane, and an oblique
    // framing of an 0.4 m depth reads as a tank seen from the corner rather
    // than as the figure the 2D lane draws.
    camera: {
      azimuth_rad: 0,
      elevation_rad: 0.06,
      distance_m: 1.8,
      target_m: { x: 0, y: 0.3, z: 0 },
    },
  }),
  defineScene({
    id: "garden-svo-lighting",
    name: "Garden · SVO lighting study",
    blurb: "A fluid-free dusk garden lit by a warm lamppost for validating SVO mip-cone lighting, soft shadows, and ambient occlusion without initializing simulation authority.",
    audience: "study",
    shelf: "Rendering",
    environment: "garden",
    // Fluid-free by construction: without the dry world there is nothing to light.
    presentationMode: "full-scene",
    build: createGardenSvoLightingScene,
    camera: gardenCamera,
  }),
  defineScene({
    id: "geometric-uniform-translation",
    name: "Geometric · uniform high-speed translation",
    blurb: "A liquid plug moves at 6 m/s with zero gravity, viscosity or surface tension. It should translate two cells per step without changing shape or speed; the short run ends before wall contact.",
    audience: "validation", shelf: "Analytic motion", environment: "stage",
    build: () => createGeometricUniformTranslationScene(),
    camera: { distance_m: 1.3, target_m: { x: 0, y: .2, z: 0 }, elevation_rad: .3, azimuth_rad: .45 },
  }),
  ...(["translation", "free-fall"] as const).map(motion => defineScene({
    id: `coarse-surface-${motion}`,
    name: motion === "translation" ? "Coarse surface · translation" : "Coarse surface · free fall",
    blurb: motion === "translation"
      ? "A small liquid rectangle moves down at 0.4 m/s with zero gravity. Its flat surface crosses fixed coarse and fine halves: expect unchanged shape and 0.20 m descent after 0.5 s."
      : "The same mixed-resolution rectangle falls from rest. Before contact, expect y = y₀ − ½gt², uniform downward velocity and no deformation. The 0.3 s run ends above the floor.",
    audience: "validation", shelf: "Analytic motion", environment: "stage",
    build: () => createAnalyticMotionScene(motion),
    camera: { distance_m: 2.6, target_m: { x: 0, y: .8, z: 0 }, elevation_rad: .3, azimuth_rad: .45 },
  })),
  defineScene({
    id: "coarse-surface-free-fall-rerung", name: "Coarse surface · free fall with refinement",
    blurb: "A falling rectangle crosses the same tank at the finest-cell resolution. Expect ballistic motion before floor contact.",
    audience: "validation", shelf: "Analytic motion", environment: "stage",
    build: createRerungFreeFallScene,
    camera: { distance_m: 2.6, target_m: { x: 0, y: .8, z: 0 }, elevation_rad: .3, azimuth_rad: .45 },
  }),
  ...([false, true] as const).map(live => defineScene({
    id: live ? "coarse-surface-standing-wave-live" : "coarse-surface-standing-wave",
    name: live ? "Coarse surface · standing wave, live refinement" : "Coarse surface · standing wave",
    blurb: live
      ? "A gentle standing wave with narrow-band refinement following the moving surface. Compare its period and amplitude with the fixed-resolution wave; watch the surface refine and coarsen."
      : "A 3 cm cosine disturbance sloshes across a narrow 1.6 m tank, with the same authored geometry. Expect one oscillation in about 1.58 s and unchanged mean depth.",
    audience: "validation", shelf: "Analytic motion", environment: "stage",
    build: () => createStandingWaveScene(live),
    camera: { distance_m: 2.6, target_m: { x: 0, y: .5, z: 0 }, elevation_rad: .3, azimuth_rad: .45 },
  })),
  ...([1, 2] as const).map((curvatureMultiplier) => defineScene({
    id: curvatureMultiplier === 2 ? "stationary-bowl-2x" : "stationary-bowl",
    name: curvatureMultiplier === 2 ? "Stationary bowl · 2× curvature" : "Stationary bowl",
    blurb: curvatureMultiplier === 2
      ? "The stationary bowl with twice the curvature in both horizontal directions, the same center height and no enforced refinement regions, for inspecting surface reconstruction artifacts."
      : "A shallow curved water surface held still by zero gravity for inspecting surface reconstruction and grid ripples.",
    audience: "study",
    shelf: "Surface reconstruction",
    environment: "stage",
    build: () => createStationaryBowlScene(curvatureMultiplier),
    camera: { distance_m: 4.6, target_m: { x: 0, y: .65, z: 0 }, elevation_rad: .58, azimuth_rad: .72 },
  })),
  defineScene({
    id: "brick-quad-dam-break",
    name: "Brick quad · dam break",
    blurb: "A 2x2 four-brick tank: one brick quadrant of water releases and crosses every brick boundary, exercising cross-brick transport, residency activation, and seam quality.",
    audience: "validation",
    shelf: "Brick residency",
    environment: "stage",
    build: createBrickQuadDamBreakScene,
    camera: { distance_m: 1.9, target_m: { x: 0, y: 0.2, z: 0 } },
    variants: {
      "gpu-smoke": smokeVariant(
        "Seam quality is the measurement, so capillary force is removed and the outer "
        + "step is pinned to 4 ms: a doubled cap moves the front across a brick boundary "
        + "in fewer steps than the coverage checkpoints resolve.",
        (scene) => {
          scene.fluid.surfaceTension_N_m = 0;
          pinStep(scene, 0.004);
        }),
    },
  }),
  defineScene({
    id: "hydrostatic-power-two-level",
    name: "Water · tiny hydrostatic",
    blurb: "A settled 16³ tank for checking hydrostatic pressure.",
    audience: "validation",
    shelf: "Hydrostatic oracles",
    environment: "stage",
    build: createTinyHydrostaticScene,
    camera: { distance_m: 1.85, target_m: { x: 0, y: 0.35, z: 0 } },
  }),
  defineScene({
    id: "hydrostatic-power-large-offset",
    name: "Water · larger hydrostatic",
    blurb: "A settled 32×24×16 tank with a free surface that cuts through the finest cells.",
    audience: "validation",
    shelf: "Hydrostatic oracles",
    environment: "stage",
    build: createLargeHydrostaticScene,
    camera: { distance_m: 2.75, target_m: { x: 0, y: 0.5, z: 0 } },
  }),
  defineScene({
    id: "large-power-hydrostatic",
    name: "Water · 20× hydrostatic",
    blurb: "The 20× dam tank with a representable one-cell-deep 32×32 sparse pool (1,024 finest-cell volumes), completing the large-scene/minimal-liquid benchmark cell.",
    audience: "validation",
    shelf: "Hydrostatic oracles",
    environment: "stage",
    build: createLargePowerHydrostaticScene,
    camera: { distance_m: 6.4, target_m: { x: 0, y: 0.2, z: 0 } },
  }),
  defineScene({
    id: "deep-power-hydrostatic",
    name: "Water · deep hydrostatic",
    blurb: "A still 64×48×64 tank filled 2 m deep, with 163,840 liquid cells for inspecting a broad interior and its surface.",
    audience: "validation",
    shelf: "Hydrostatic oracles",
    environment: "stage",
    build: createDeepPowerHydrostaticScene,
    camera: { distance_m: 7.4, target_m: { x: 0, y: 1, z: 0 } },
  }),
  ...POWER_DROPLET_EDGE_CELLS.map((edgeCells) => defineScene({
    id: `power-droplet-${edgeCells}`,
    name: `Water · droplet in a ${edgeCells}³ domain`,
    blurb: `One hundred cells of water — a 5×4×5 corner puddle — in a ${edgeCells}³ container (${(edgeCells * POWER_DROPLET_CELL_SIZE_M).toFixed(1)} m on a side). The fluid is identical across the family, so the only thing that changes between these scenes is how much empty space the solver is asked to carry. Wall against domain volume across the sweep is the domain tax, measured directly.`,
    audience: "validation",
    // Its own shelf rather than the hydrostatic one: these four are a single
    // instrument read across its members, not four scenes anyone would open
    // for their own sake. The shelf is the unit of meaning here.
    shelf: "Domain-tax sweep",
    environment: "stage",
    build: () => createPowerDropletScene(edgeCells),
    camera: {
      distance_m: edgeCells * POWER_DROPLET_CELL_SIZE_M * 1.6,
      target_m: { x: 0, y: edgeCells * POWER_DROPLET_CELL_SIZE_M * 0.1, z: 0 },
    },
  })),
  ...POWER_FILL_LIQUID_CELLS.map((liquidCells) => {
    const cells = powerFillReservoirCells(liquidCells);
    return defineScene({
      id: `power-fill-${POWER_FILL_EDGE_CELLS}-${liquidCells}`,
      name: `Water · ${liquidCells.toLocaleString("en-US")}-cell fill of a ${POWER_FILL_EDGE_CELLS}³ domain`,
      blurb: `${liquidCells.toLocaleString("en-US")} cells of water — a ${cells.x}×${cells.y}×${cells.z} corner reservoir — in a fixed ${POWER_FILL_EDGE_CELLS}³ container (${(POWER_FILL_EDGE_CELLS * POWER_DROPLET_CELL_SIZE_M).toFixed(1)} m on a side) with identical geometry across the family. Only the amount of water changes.`,
      audience: "validation",
      // Its own shelf, beside the domain-tax sweep: these three are one
      // instrument read across its members, and the pair of shelves is the
      // pair of axes.
      shelf: "Live-occupancy sweep",
      environment: "stage",
      build: () => createPowerFillScene(liquidCells),
      camera: {
        distance_m: POWER_FILL_EDGE_CELLS * POWER_DROPLET_CELL_SIZE_M * 1.6,
        target_m: { x: 0, y: POWER_FILL_EDGE_CELLS * POWER_DROPLET_CELL_SIZE_M * 0.1, z: 0 },
      },
    });
  }),
  defineScene({
    id: "power-hybrid-deep-ocean",
    name: "Water · hybrid deep ocean",
    blurb: "A 64×64×48 pool filled to 56 wet layers, with no wave seed at all. Its broad interior provides a still-water pressure and rendering case.",
    audience: "validation",
    shelf: "Hydrostatic oracles",
    environment: "stage",
    build: createPowerHybridDeepOceanScene,
    camera: { distance_m: 7.4, target_m: { x: 0, y: 1.4, z: 0 } },
  }),
  ...SPARSE_CM12_COMPLEXITY_SCENES.map((rung) => defineScene({
    id: `sparse-cm12-ladder-${rung.id}`,
    name: `Water ${String(rung.ordinal).padStart(2, "0")} · ${rung.title}`,
    blurb: `${rung.introducedFeature}. Suggested observation: ${rung.defaultSteps} solver ${rung.defaultSteps === 1 ? "step" : "steps"}.`,
    audience: "validation",
    shelf: "Geometry ladder",
    environment: "stage",
    build: () => createSparseCM12ComplexityScene(rung.id),
  })),
  defineScene({
    id: "minimal-power-dam-break",
    name: "Minimal dam break",
    blurb: "The analytic 12.5%-volume dam initializer collapses inside a 16³ tank, providing a small dynamic water scene.",
    audience: "validation",
    shelf: "Dam-break ladder",
    environment: "stage",
    build: createMinimalPowerDamBreakScene,
    camera: { distance_m: 1.9, target_m: { x: 0, y: 0.3, z: 0 } },
  }),
  defineScene({
    id: "minimal-power-dam-break-32",
    name: "Minimal dam break 32³",
    blurb: "An analytic mini dam at 0.025 m resolution.",
    audience: "validation",
    shelf: "Dam-break ladder",
    environment: "stage",
    build: createMinimalPowerDamBreak32Scene,
    camera: { distance_m: 1.9, target_m: { x: 0, y: 0.3, z: 0 } },
  }),
  defineScene({
    id: "falling-water-torus",
    name: "Falling water torus",
    blurb: "A suspended ring of water drops onto a dry floor. Watch the hole deform as the ring flattens and spreads on impact.",
    audience: "explore", shelf: "Tanks", environment: "stage",
    build: createFallingWaterTorusScene,
    camera: { distance_m: 3, target_m: { x: 0, y: 0.6, z: 0 } },
  }),
  defineScene({
    id: "coarse-first-pool-impact",
    name: "Water · ball into still pool",
    blurb: "A liquid ball falls into a broad still pool. Watch the impact and spreading surface.",
    audience: "validation", shelf: "Dam-break ladder", environment: "stage",
    build: createCoarseFirstPoolImpactScene,
    camera: { distance_m: 11, target_m: { x: 0, y: 1.8, z: 0 } },
  }),
  defineScene({
    id: "coarse-first-pool-impact-half",
    name: "Water · ball into still pool (half resolution)",
    blurb: "The same 6.4 × 4.8 × 6.4 m pool impact at half the linear resolution: 0.1 m finest cells on a 64×48×64 lattice, one eighth of the cells. Identical physical dimensions and falling ball, so only resolution varies across the family.",
    audience: "validation", shelf: "Dam-break ladder", environment: "stage",
    build: createCoarseFirstPoolImpactHalfScene,
    camera: { distance_m: 11, target_m: { x: 0, y: 1.8, z: 0 } },
  }),
  defineScene({
    id: "coarse-first-pool-impact-half-slab",
    name: "Water · disk into still pool (half-resolution slab)",
    blurb: "A 6.4 × 4.8 × 0.8 m extrusion for 2D comparison: 64×48×8 finest cells at 0.1 m, a 1 m radius disk centred at 3.65 m above a 1.6 m pool, and closed free-slip walls. Matches the half-resolution pool's XY cross-section, gravity and timestep; compare the centre slice and volume per 0.8 m depth.",
    audience: "validation", shelf: "Dam-break ladder", environment: "stage",
    build: createCoarseFirstPoolImpactHalfSlabScene,
    camera: { azimuth_rad: 0, elevation_rad: 0, distance_m: 10,
      target_m: { x: 0, y: 2.4, z: 0 } },
  }),
  defineScene({
    id: "coarse-first-pool-impact-quarter",
    name: "Water · ball into still pool (quarter resolution)",
    blurb: "The same 6.4 × 4.8 × 6.4 m pool impact at one quarter the linear resolution: 0.2 m finest cells on a 32×24×32 lattice, one sixty-fourth of the cells. Identical physical dimensions and falling ball.",
    audience: "validation", shelf: "Dam-break ladder", environment: "stage",
    build: createCoarseFirstPoolImpactQuarterScene,
    camera: { distance_m: 11, target_m: { x: 0, y: 1.8, z: 0 } },
  }),
  defineScene({
    id: "minimal-power-dam-break-64",
    name: "Minimal dam break 64³",
    blurb: "An analytic mini dam at 0.0125 m resolution.",
    audience: "validation",
    shelf: "Dam-break ladder",
    environment: "stage",
    build: createMinimalPowerDamBreak64Scene,
    camera: { distance_m: 1.9, target_m: { x: 0, y: 0.3, z: 0 } },
  }),
  defineScene({
    id: "sparse-geometric-ritter-dam-break",
    name: "Ritter dam break",
    blurb: "A shallow 4 m free-slip flume with a 0.1 m head. Accepted-volume depth and pre-impact sensor arrivals can be compared with Ritter's dry-bed solution; the later wall impact is a separate physical observation.",
    audience: "validation",
    shelf: "Dam-break ladder",
    environment: "stage",
    build: createSparseGeometricRitterDamBreakScene,
    camera: { distance_m: 5.2, target_m: { x: 0.4, y: 0.1, z: 0 } },
  }),
  defineScene({
    id: "sparse-cm12-long-dam-break",
    name: "Water · long-tank dam break",
    blurb: "A tall 192×96×32 tank with a full-width reservoir at one end. Watch the water front travel along its dry floor.",
    audience: "validation",
    shelf: "Dam-break ladder",
    environment: "stage",
    build: createSparseCM12LongDamBreakScene,
    camera: { distance_m: 4.35, target_m: { x: 0, y: 0.5, z: 0 } },
  }),
  defineScene({
    id: "large-power-dam-break",
    name: "Water · 20× dam break",
    blurb: "The mini dam break's exact water block in a tank with 20× the volume: 4× longer, 4× wider, and 25% taller, at the same finest-cell resolution.",
    audience: "validation",
    shelf: "Dam-break ladder",
    environment: "stage",
    build: createLargePowerDamBreakScene,
    camera: { distance_m: 6.4, target_m: { x: 0, y: 0.45, z: 0 } },
  }),
  defineScene({
    id: "symmetric-expansion",
    name: "Water · symmetric expansion",
    blurb: "A central 2×1×2-brick water body collapses in a 32×16×32 tank. Its four matching walls make symmetry easy to inspect.",
    audience: "validation",
    shelf: "Symmetry",
    environment: "stage",
    build: createSymmetricExpansionScene,
    camera: { distance_m: 2.5, target_m: { x: 0, y: 0.25, z: 0 } },
  }),
  defineScene({
    id: "sparse-cm12-symmetric-expansion",
    name: "Water · central expansion",
    blurb: "A central water block expands across a 32×16×32 tank with four matching walls.",
    audience: "validation",
    shelf: "Symmetry",
    environment: "stage",
    build: createSymmetricExpansionScene,
    camera: { distance_m: 2.5, target_m: { x: 0, y: 0.25, z: 0 } },
  }),
  defineScene({
    id: "ceiling-slab-drop",
    name: "Water · ceiling drop oracle",
    blurb: "One 8³ fluid brick seeded flush under the closed lid, touching nothing else. The exact answer is free fall (y = y₀ − gt²/2, impact ≈0.29 s); any hesitation is wall adhesion.",
    audience: "validation",
    shelf: "Free-fall contact",
    environment: "stage",
    build: createCeilingSlabDropScene,
    camera: { distance_m: 2.4, target_m: { x: 0, y: 0.4, z: 0 } },
  }),
  defineScene({
    id: "rigid-hydrostatic",
    name: "Rigid coupling · hydrostatic sphere",
    blurb: "A fully submerged static sphere measures pressure buoyancy, volume displacement, and publication liveness.",
    audience: "validation", shelf: "Rigid coupling", environment: "stage",
    build: createRigidHydrostaticScene,
    camera: { distance_m: 1.8, target_m: { x: 0, y: 0.3, z: 0 } },
  }),
  defineScene({
    id: "rigid-float",
    name: "Rigid coupling · buoyant sphere",
    blurb: "A half-density sphere released at the waterline checks settling, drag, and plunge-through.",
    audience: "validation", shelf: "Rigid coupling", environment: "stage",
    build: createRigidFloatScene,
    camera: { distance_m: 1.8, target_m: { x: 0, y: 0.4, z: 0 } },
  }),
  defineScene({
    id: "rigid-sink",
    name: "Rigid coupling · dense sphere drop",
    blurb: "A dense sphere enters from above and checks bounded splash, conserved displacement, and floor arrival.",
    audience: "validation", shelf: "Rigid coupling", environment: "stage",
    build: createRigidSinkScene,
    camera: { distance_m: 1.8, target_m: { x: 0, y: 0.4, z: 0 } },
  }),
  defineScene({
    id: "corner-brick-drop",
    name: "Water · corner drop oracle",
    blurb: "One 8³ fluid brick seeded into a top corner: lid, two walls, and their vertical edge seam. Frictionless walls exert only normal force, so the exact answer is still free fall.",
    audience: "validation",
    shelf: "Free-fall contact",
    environment: "stage",
    build: createCornerBrickDropScene,
    camera: { distance_m: 2.4, target_m: { x: 0, y: 0.4, z: 0 } },
  }),
  defineScene({
    id: "midair-brick-drop",
    name: "Water · mid-air drop control",
    blurb: "The same 8³ brick hanging in open space, touching no boundary at all. The zero-contact control: whatever it deviates from free fall is the scheme's own transient, not adhesion.",
    audience: "validation",
    shelf: "Free-fall contact",
    environment: "stage",
    build: createMidairBrickDropScene,
    camera: { distance_m: 3, target_m: { x: 0, y: 0.6, z: 0 } },
  }),
  defineScene({
    id: "midair-corner-drop",
    name: "Water · mid-air corner oracle",
    blurb: "The same 8³ brick against two vertical walls and their seam, but clear of the lid. Gravity is tangential to both walls, so free-slip walls cannot slow it: this isolates seam adhesion from ceiling adhesion.",
    audience: "validation",
    shelf: "Free-fall contact",
    environment: "stage",
    build: createMidairCornerDropScene,
    camera: { distance_m: 3, target_m: { x: 0, y: 0.6, z: 0 } },
  }),
]);

export function getSceneDefinition(id: string): SceneDefinition {
  return findSceneDefinition(id) ?? SCENE_CATALOG[0];
}

/**
 * Exact lookup, with no fallback.
 *
 * A saved scene records the preset it came from, and that id may name a scene
 * this build no longer has. Falling back would silently frame it with another
 * scene's camera and pin another scene's solver profile, so callers that are
 * asking "was this authored here?" need to be able to hear no.
 */
export function findSceneDefinition(id: string): SceneDefinition | undefined {
  return SCENE_CATALOG.find((definition) => definition.id === id);
}

/** Library cards for the authored catalog, in catalog order. */
export const sceneCatalogCards: readonly SceneCard[] =
  Object.freeze(SCENE_CATALOG.map(sceneCardForDefinition));

/**
 * The preset projection.
 *
 * `create` is `sceneDocument`, so a preset, a library card, a saved scene and a
 * GPU smoke lane all obtain the identical document — which was not previously
 * true: the smoke catalog's local factories assigned `environment` without
 * copying its scenery, so ten shared ids named two different scenes.
 */
export const scenePresets: ReadonlyArray<ScenePreset> = SCENE_CATALOG.map((definition) => {
  /**
   * Lazy, because the answer needs the document and the document is not free.
   *
   * `presentationModeForScene` reads the lattice the house set has to fit into,
   * which means building the scene; doing that for all fifty-odd definitions at
   * import would put every factory in the catalog on the client's startup path
   * to answer a question about the four it is asked about. Memoised because the
   * viewport reads this every animation frame.
   */
  let resolved: ScenePresentationMode | undefined;
  return {
    id: definition.id,
    name: definition.name,
    group: definition.shelf,
    description: definition.blurb,
    background: definition.environment,
    get presentationMode(): ScenePresentationMode {
      return resolved ??= presentationModeForScene(definition, sceneDocument(definition));
    },
    camera: definition.camera,
    renderProfile: definition.renderProfile,
    create: () => sceneDocument(definition),
  };
});

export const defaultScenePresetId = scenePresets[0].id;

export function getScenePreset(id: string): ScenePreset {
  return scenePresets.find((preset) => preset.id === id) ?? scenePresets[0];
}

export function cameraForPreset(preset: ScenePreset): CameraState {
  return { ...defaultCamera, ...preset.camera };
}
