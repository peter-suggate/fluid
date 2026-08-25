import { cloneScene, defaultCamera, defaultScene, DEFAULT_GPU_CPU_TIMESTEP_RATIO, type CameraState, type SceneDescription } from "./model";
import { applyHeroGardenNodeOverrides } from "./hero-garden-overrides";
import { createMassConservingFigure9DamBreak, createPaperScenario } from "./paper-scenarios";
import { CM12_FIGURES, cm12Camera, cm12Grid, cm12MethodProfile, cm12Scene } from "./cm12-paper-scenes";
import { applyGardenPool, GARDEN_DAM_BRICK_SEED_M, GARDEN_WATERLINE_M, gardenPoolTerrain } from "./garden-scene";
import {
  createHeroGardenHoseScene,
  heroGardenCamera,
  type HeroGardenHoseOptions,
} from "./hero-garden-scene";
import {
  createHeroGardenHoseStressScene,
  HERO_GARDEN_STRESS_MAXIMUM_MULTIPLIER,
} from "./hero-garden-stress-scene";
import { studioStageCamera } from "./studio-stage-scene";
import { withHeroLayout } from "./voxel-scenery/hero-layout";
import { terrainHeightAt, type TerrainDescription, type TerrainGrid } from "./terrain";
import type { EnvironmentId } from "./environments";
import type { MethodProfile } from "./method-contract";
import { sceneDamBreakFractions } from "./initial-fluid";
import { boxTankWallFieldForScene } from "./scene-lattice";
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
  /** Exact solver profile required for a numerical comparison/validation preset. */
  methodProfile?: MethodProfile;
  /** Art-directed background that is part of this preset's presentation. */
  background: EnvironmentId;
  /** Whether the authored dry world is presented behind the raster water. */
  presentationMode: ScenePresentationMode;
}

export const POWER_VALIDATION_METHOD_PROFILE: MethodProfile = Object.freeze({
  // Historical profile name retained for callers: this is a Losasso profile.
  // Product scenes start on Losasso; the smoke catalog opts into the frozen
  // Power Liquids method explicitly, by naming it.
  methodId: "losasso",
  quality: "balanced",
  overrides: Object.freeze({
    maximumLeafSize: "16",
    interfaceRefinementBandCells: 3,
    globalFineLevelSetFactor: "1",
  }),
});

/**
 * Canonical Sparse CM12 front-propagation lane: adaptive policy, paper step.
 *
 * Every scene that runs a CM12 lane runs it at the paper's 1/30 s, this one
 * included. Sec. 3.5 sharpening only balances transport diffusion at that
 * per-step dose, so a scene-step profile was asking the gate to certify a
 * front the shipped method never propagates. The scene keeps its authored
 * 4 ms `fixedDt_s` for the other methods that open it; the A/B tool in
 * tools/run-sparse-cm12-long-dam-ab-dawn.ts still opts into the scene step
 * explicitly, which is what a matched-dt comparison lane is for.
 */
export const SPARSE_CM12_LONG_DAM_METHOD_PROFILE: MethodProfile = Object.freeze({
  methodId: "adaptive-mass",
  quality: "balanced",
  overrides: Object.freeze({
    resolutionMode: "adaptive",
    // These thresholds remain the long-dam activity tuning when that optional
    // criterion is selected; the product default stays Surface distance.
    finestTravelCells: 4,
    fourTravelCells: 2,
    twoTravelCells: 1,
    timeStep: "paper",
  }),
});

/** The production UI tuple exercised by every Sparse CM12 complexity rung. */
export const SPARSE_CM12_COMPLEXITY_LADDER_METHOD_PROFILE: MethodProfile = Object.freeze({
  methodId: "adaptive-mass",
  quality: "balanced",
  overrides: Object.freeze({
    brickFineResolution: "8",
    resolutionMode: "adaptive",
    maximumMacroSpanBricks: "auto",
    selectorMode: "surface",
    receiverFloor: "auto",
    surfaceFineRings: 1,
    receiverSupportRings: 9,
    timeStep: "paper",
    pressureIterations: 64,
    pressureRelativeTolerance: 4e-6,
  }),
});

/** Conservative profile for the small bounded transfer demonstration. */
export const BOUNDED_POOL_TRANSFER_METHOD_PROFILE: MethodProfile = Object.freeze({
  methodId: "adaptive-mass",
  quality: "balanced",
  overrides: Object.freeze({
    brickFineResolution: "8",
    resolutionMode: "adaptive",
    maximumMacroSpanBricks: "auto",
    selectorMode: "surface",
    receiverFloor: "auto",
    surfaceFineRings: 1,
    receiverSupportRings: 2,
    timeStep: "scene",
    pressureIterations: 64,
    pressureRelativeTolerance: 4e-6,
  }),
});

/**
 * A deliberately bounded fluid-transfer demonstration.
 *
 * The outer container is the receiving pool from t=0 and a compact water
 * column occupies its left end. There is deliberately no rigid divider: this
 * is the smallest useful baseline for watching region A wet region B without
 * conflating transport, dynamic sparse residency, and solid voxelization.
 * Every possible fluid destination is inside the authored 32x12x16 lattice.
 */
export function createBoundedPoolTransferScene(): SceneDescription {
  const cellSize_m = 0.05;
  const scene = cloneScene(defaultScene);
  scene.sceneId = "bounded-pool-transfer";
  scene.duration_s = 4;
  scene.container = {
    ...scene.container,
    width_m: 32 * cellSize_m,
    height_m: 12 * cellSize_m,
    depth_m: 16 * cellSize_m,
    // 8x9x16 initially wet cells inside a 32x12x16 pool.
    fillFraction: (8 * 9 * 16) / (32 * 12 * 16),
    top: "open",
    fluidWallMode: "free-slip",
  };
  scene.voxelDomain = { finestCellSize_m: cellSize_m, brickSize_cells: 8 };
  scene.container.wallField = boxTankWallFieldForScene(scene);
  scene.fluid.initialCondition = "dam-break";
  scene.fluid.initialDamBreakDimensions_m = {
    x: 8 * cellSize_m,
    y: 9 * cellSize_m,
    z: 16 * cellSize_m,
  };
  delete scene.fluid.initialDamBreakOrigin_m;
  scene.fluid.surfaceTension_N_m = 0;
  delete scene.fluid.initialBrickSeeds_m;
  delete scene.fluid.initialBrickSeedsAdditive;
  delete scene.fluid.initialLiquidVolumes;
  delete scene.fluid.inflow;

  scene.rigidBodies = [];
  scene.numerics.fixedDt_s = scene.numerics.maxDt_s = 0.004;
  return scene;
}

/** Canonical factor-one adaptive LoSasso profile for interactive water.
 * This exact tuple is exercised by Dawn raster lanes before it is offered by
 * the dam-break or symmetry presets. */
export const ADAPTIVE_LOSASSO_UI_METHOD_PROFILE: MethodProfile = Object.freeze({
  methodId: "losasso",
  quality: "balanced",
  overrides: Object.freeze({
    losassoVelocityExtension: "causal-front",
    // The visible D4 gate is 32x16x32, so 16 is its largest common dyadic leaf.
    maximumLeafSize: "16",
    interfaceRefinementBandCells: 4,
    globalFineLevelSetFactor: "1",
  }),
});

/** Reproducible Aanjaneya et al. 2017 comparison tuple.
 *
 * Section 5 keeps a separate uniformly high-resolution sparse level set,
 * typically 4× or 8× the background octree. Factor four is the canonical
 * benchmark here; band reach four is the existing symmetric-expansion oracle.
 */
export const POWER2017_FACTOR4_BENCHMARK_METHOD_PROFILE: MethodProfile = Object.freeze({
  methodId: "power-liquids",
  quality: "balanced",
  overrides: Object.freeze({
    maximumLeafSize: "32",
    interfaceRefinementBandCells: 4,
    // Every-advance topology is a construction constant of that method, not an
    // authored value, so this profile no longer states it.
    globalFineLevelSetFactor: "4",
  }),
});

/** Historical oracle name retained for focused symmetry callers. */
export const SYMMETRIC_EXPANSION_METHOD_PROFILE = ADAPTIVE_LOSASSO_UI_METHOD_PROFILE;

/** Canonical Sparse CM12 product path for the same D4 expansion oracle.
 *
 * Keep the octree/LoSasso card above as its independent comparison lane. The
 * Sparse CM12 card names B8/P8 and the interactive production pressure
 * budget. The stricter 108-iteration accuracy oracle remains a separate Dawn
 * lane; making that offline budget the UI default needlessly drops the scene
 * below real time.
 */
export const SPARSE_CM12_SYMMETRIC_EXPANSION_METHOD_PROFILE:
MethodProfile = Object.freeze({
  methodId: "adaptive-mass",
  quality: "balanced",
  overrides: Object.freeze({
    resolutionMode: "adaptive",
    brickFineResolution: "8",
    presentationPageResolution: "8",
    selectorMode: "surface",
    timeStep: "paper",
    pressureIterations: 64,
  }),
});

/**
 * Shared coarse-only dam experiment. The surface field and pressure octree use
 * the same lattice; size-32 leaves preserve a materially coarse far field while
 * three grading layers expose the intermediate dyadic shells.
 */
export const COARSE_ONLY_POWER_DAM_METHOD_PROFILE: MethodProfile = Object.freeze({
  methodId: "losasso",
  quality: "balanced",
  overrides: Object.freeze({
    maximumLeafSize: "32",
    interfaceRefinementBandCells: 3,
    surfaceRefinementGradingLayers: 3,
    globalFineLevelSetFactor: "1",
  }),
});

/** Fixed WebGPU pool for the room-sized scene's physically unchanged water.
 * The initial paper-style publication occupies 3,114 pages, but Section 5
 * rebuilds around the deformed surface: a floor-spanning sheet needs the
 * 64 x 64 brick footprint times band-1's seven recurring layers, or 28,672
 * pages. One additional footprint rounds that physical bound to 32,768 while
 * exact active residency remains the rebuilt band rather than the whole pool. */
export const LARGE_POWER_DAM_FINE_BRICK_CAPACITY = 32_768;

/** The room-sized comparison keeps empty regions aggressively coarse. */
export const LARGE_POWER_DAM_METHOD_PROFILE: MethodProfile = Object.freeze({
  ...POWER_VALIDATION_METHOD_PROFILE,
  overrides: Object.freeze({
    ...POWER_VALIDATION_METHOD_PROFILE.overrides,
    // 64x20x64: four is the largest dyadic leaf dividing every axis.
    maximumLeafSize: "4",
    interfaceRefinementBandCells: 1,
    // The authored 1,472-cell reservoir publishes 1,185 rows at t=0. An
    // 8,192-row footprint pool leaves >6.9x growth headroom while the existing
    // overflow receipt rejects an over-budget generation.
    pressureRowCapacity: 8_192,
    globalFineLevelSetMaximumBricks: LARGE_POWER_DAM_FINE_BRICK_CAPACITY,
  }),
});

/** Footprint-budgeted allocation for the quarter-volume large still-water
 * lane. Cold publication uses 1,028 pressure rows and recurring fine residency
 * is 6,405 bricks. Its fine-band reach needs 5,456 non-liquid support records;
 * 4,096 rows author 6,144 while the existing GPU overflow sentinels reject any
 * generation that exceeds either reserve. */
export const LARGE_POWER_HYDROSTATIC_PRESSURE_ROW_CAPACITY = 4_096;
export const LARGE_POWER_HYDROSTATIC_METHOD_PROFILE: MethodProfile = Object.freeze({
  ...POWER_VALIDATION_METHOD_PROFILE,
  overrides: Object.freeze({
    ...POWER_VALIDATION_METHOD_PROFILE.overrides,
    // This scene shares the 64x20x64 container with the large dam.
    maximumLeafSize: "4",
    interfaceRefinementBandCells: 1,
    pressureRowCapacity: LARGE_POWER_HYDROSTATIC_PRESSURE_ROW_CAPACITY,
  }),
});

/**
 * Row reserve for the deep still-water lane (`deep-power-hydrostatic`).
 *
 * A CPU model of the shipped refinement rule — unit owners inside the three-
 * cell closed-wall strips (`OCTREE_POWER_BOUNDARY_STRIP_MIN_CELLS`) and across
 * the free surface, `interfaceRefinementBandCells + surfaceRefinementGrading-
 * Layers * (size - 2)` of outward distance protection, then 2:1 balance —
 * publishes 57,776 rows on this 64x48x64 tank: 50,944 unit, 6,048 size-2, 752
 * size-4 and 32 size-8 leaves. That is 0.353 rows per liquid cell, against
 * `large-power-hydrostatic`'s 1.004. The same model reproduces the ocean
 * lane's measured 363,336-row frontier bank to within 5%, and it deliberately
 * over-refines (it ignores that a leaf outside the published fine narrow band
 * has no summary to refine against at all), so 57,776 is an upper bound.
 *
 * `planOctreePressureCapacity`'s own scene-shaped request here is 59,392
 * (15,360 surface + 43,764 closed-wall strip + 48 coarse rows, 256-aligned) —
 * only 2.8% above that bound, which is not enough for a settling transient.
 * 65,536 is the smallest 256-aligned power of two clearing both by >= 1.10x.
 * It is not an air-support allowance: above `domain / 5` rows
 * `octreeAirSupportFootprintCapacity` returns the dense domain-sized support
 * reserve (196,608) at any capacity in this range, which already covers the
 * 64 x 4 x 64 = 16,384-cell air reach above the free surface. The GPU overflow
 * receipt still rejects any generation that exceeds the reserve.
 */
export const DEEP_POWER_HYDROSTATIC_PRESSURE_ROW_CAPACITY = 65_536;

/**
 * Physical single-interface fine-band reserve for the same lane.
 *
 * The deep tank's fluid footprint *is* the tank, so the footprint-shaped
 * default (`planFluidFootprintFineNarrowBandBrickCapacity`, which reserves all
 * three exposed faces of the authored liquid box) asks for 152,064 of the
 * 196,608 logical fine bricks — 77% of a dense lattice, for a scene whose
 * interface is one horizontal plane. The physical bound is that plane:
 * 64 x 64 = 4,096 fine bricks of interface area times the eleven-layer
 * capacity band that `planFineLevelSetCapacityDilationBrickRings(4, 1, 4)`
 * (five rings) implies is 45,056 resident bricks, and the planner's own 1.5x
 * deformation safety rounds that to 67,584 — exactly
 * `planGlobalFineNarrowBandBrickCapacity([64, 48, 64], 5).maximumResidentBricks`,
 * which the scene test pins. Authoring it removes 2.25x of reserve rather than
 * adding any.
 */
export const DEEP_POWER_HYDROSTATIC_FINE_BRICK_CAPACITY = 67_584;

/** The deep still-water lane keeps the 20x still lane's interface and fine
 * surface settings. Its 64x48x64 domain can use a larger compatible leaf than
 * the shallow 64x20x64 tank. */
export const DEEP_POWER_HYDROSTATIC_METHOD_PROFILE: MethodProfile = Object.freeze({
  ...LARGE_POWER_HYDROSTATIC_METHOD_PROFILE,
  overrides: Object.freeze({
    ...LARGE_POWER_HYDROSTATIC_METHOD_PROFILE.overrides,
    maximumLeafSize: "16",
    pressureRowCapacity: DEEP_POWER_HYDROSTATIC_PRESSURE_ROW_CAPACITY,
    globalFineLevelSetMaximumBricks: DEEP_POWER_HYDROSTATIC_FINE_BRICK_CAPACITY,
  }),
});

// ---- the droplet-in-a-vast-domain family ------------------------------------
//
// `wall(scene) ~= intercept(fluid footprint, solver constants) + slope * domain`.
// Every other authored lane moves both terms at once, so neither can be read.
// This family pins the fluid term to ~zero — one hundred liquid cells, always
// the same hundred — and sweeps only the container. The measured ms/advance vs
// nx*ny*nz curve therefore *is* the domain tax: its slope is what the program
// attacks, and its intercept belongs to the solve-constant work that the
// persistent-MGPCG program already owns. See
// `docs/POWER_LIQUIDS_DROPLET_VAST_DOMAIN_PLAN.md`.

/** Finest cell shared with the whole hydrostatic/dam family, so a wall measured
 * here is comparable to theirs: only the lattice *extent* differs. */
export const POWER_DROPLET_CELL_SIZE_M = 0.05;

/**
 * The reservoir, and the reason it is a corner reservoir.
 *
 * 0.25 x 0.2 x 0.25 m is 5 x 4 x 5 = 100 finest cells at 0.05 m, and it is
 * authored with **no** `initialDamBreakOrigin_m`. That is not a stylistic
 * choice: `analyticSparseBootstrap` (`lib/webgpu-octree.ts`) requires no brick
 * seeds, no bodies, no terrain and a reservoir anchored at the container
 * corner. Any offset origin, or an equivalent `initialBrickSeeds_m` droplet,
 * flips the whole bootstrap dense — `Float32Array(nx*ny*nz)`, a full-domain
 * signed-distance transform, a 67-134 MB phi texture and a persistent
 * `cellCount * 4 B` surface-state allocation — which is precisely the
 * domain-shaped cost this family exists to measure, moved into the setup where
 * it cannot be separated. A free-floating droplet is a later, deliberate
 * variant with those costs budgeted; it is not the v1 scene.
 */
export const POWER_DROPLET_RESERVOIR_M = Object.freeze({ x: 0.25, y: 0.2, z: 0.25 });

/** 5 x 4 x 5 cells of the authored reservoir. The invariant of the family. */
export const POWER_DROPLET_LIQUID_CELLS = 100;

/** Container edges of the swept family, in finest cells.
 *
 * 64 is the control (3.2 m — the large-dam footprint, so its intercept is
 * directly comparable to that lane's). 240 is the vast end at 12 m: 13,824,000
 * cells, whose recurring fine-band ladder issues 54,000 workgroups and clears
 * the 65,535 device dispatch limit. 256 is 16,777,216 cells = 65,536
 * workgroups, one over that limit, and did not run at all until the recurring
 * ladder was routed through `planFineLevelSetDispatch2D`; it is kept in the
 * family as the regression test for the scene that could not exist. */
export const POWER_DROPLET_EDGE_CELLS = Object.freeze([64, 128, 240, 256] as const);
export type PowerDropletEdgeCells = typeof POWER_DROPLET_EDGE_CELLS[number];

/**
 * Row reserve for every droplet lane — deliberately independent of N.
 *
 * The planner's footprint-shaped default collapses to 256 rows at 100 liquid
 * cells (`planOctreePressureCapacity`: `maximumLiquidCells * 2`, 256-aligned).
 * That is too tight the moment the three-cell closed-wall strip and 2:1
 * balance publish around a reservoir sitting in the container corner, and an
 * under-authored value does not merely reserve less — it dies at the
 * `cold-topology` / `pressureCapacityOverflow` gates, the way
 * `large-power-dam-break` dies at its planner default.
 *
 * The derivation, in rows:
 *
 *   liquid rows        5 * 4 * 5 unit owners                     =   100
 *   wall strip         the reservoir touches three closed walls;
 *                      the 3-cell strip around a 5x4x5 corner
 *                      block is bounded by 11 * 10 * 11 - 100    = 1,110
 *   surface band       band 1 plus the Section 5 transport reach
 *                      of 4 cells, i.e. a further four layers
 *                      outward: 19 * 18 * 19 - 11 * 10 * 11      = 5,288 cells,
 *                      of which only wet leaves publish rows
 *   2:1 balance        dry, coarsening outward: no rows
 *
 * Every term is a *local* shell around a fixed 100-cell body, so none of them
 * moves with N — which is the whole point of the instrument. 4,096 covers the
 * liquid plus wall terms with the surface band entirely dry, with 3.7x
 * headroom, and the t=0 `pressureRequiredRows` readback verifies it. The GPU
 * overflow receipt still rejects any generation that exceeds the reserve.
 */
export const POWER_DROPLET_PRESSURE_ROW_CAPACITY = 4_096;

/**
 * Fine narrow-band brick reserve, likewise independent of N.
 *
 * At `globalFineLevelSetFactor: "4"` and a 4-cubed brick resolution the fine
 * lattice has exactly one brick per finest cell, so the *logical* lattice is
 * nx*ny*nz — 13.8M bricks at 240 — while the resident set is the band around
 * 100 cells. The footprint-shaped default asks for ~1,073 resident bricks for
 * this reservoir; 4,096 authors 3.8x of that for the slump transient and is
 * still four orders of magnitude below the logical lattice. There is no
 * environment path for this knob (`lib/methods/octree.ts` reads it only from
 * authored method values), so a lane that wants it must carry it in the smoke
 * catalog overrides.
 */
export const POWER_DROPLET_FINE_BRICK_CAPACITY = 4_096;

/** Every surface discretization knob of the 20x still lane — interface band 1,
 * fine factor 4 — so the droplet sweep's intercept is comparable to the
 * hydrostatic family's wall. Only the two footprint reserves are re-authored,
 * and both are constants of the family rather than functions of N. */
export const POWER_DROPLET_METHOD_PROFILE: MethodProfile = Object.freeze({
  ...LARGE_POWER_HYDROSTATIC_METHOD_PROFILE,
  overrides: Object.freeze({
    ...LARGE_POWER_HYDROSTATIC_METHOD_PROFILE.overrides,
    // All swept cube edges (64, 128, 240, 256) are divisible by 16.
    maximumLeafSize: "16",
    pressureRowCapacity: POWER_DROPLET_PRESSURE_ROW_CAPACITY,
    globalFineLevelSetMaximumBricks: POWER_DROPLET_FINE_BRICK_CAPACITY,
  }),
});

// ---- the live-occupancy fill family: the droplet sweep's dual ---------------
//
// The droplet family pins the fluid and sweeps the container, so it measures
// `slope * domain`. It has run out of resolving power on the other term:
// *every* pass driven by live work is flat across that sweep by construction,
// so "flat in N" cannot tell correctly live-shaped work apart from
// capacity-shaped work or from fixed overhead. At 256 cubed the two largest
// GPU labels — resident grading closure and structured advection — are both
// flat, and one of them (advection, `publishUnionDispatch` in
// `lib/webgpu-octree-structured-dynamics.ts`) is provably live-count-driven.
// Flatness in N is therefore not evidence of anything.
//
// This family is the dual instrument. The container is FIXED at 256 cubed and
// only the reservoir moves, so a per-pass cost read across its members answers
// the question the droplet sweep cannot:
//
//   flat in live occupancy        -> fixed overhead; deletable, and the only
//                                    place a 5x frame win can come from
//   linear in live occupancy      -> honest work; attack per-item cost
//   superlinear in live occupancy -> algorithmic defect
//
// Read together the two families span the plane: droplet holds fluid and moves
// domain, fill holds domain and moves fluid.

/** Container edge for every member of the fill family, in finest cells.
 *
 * Fixed on purpose, and fixed at the *largest* droplet member so the two
 * instruments share a container: `power-fill-256-100` and `power-droplet-256`
 * are the same 256-cubed box holding the same hundred cells, differing only in
 * their authored reserves. That coincidence is the capacity control — see
 * `POWER_FILL_PRESSURE_ROW_CAPACITY`. */
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

/**
 * Row reserve for every fill lane — deliberately independent of the reservoir.
 *
 * A capacity that grew with the member would be a hidden confound: a pass that
 * is capacity-shaped would then read as live-shaped and the instrument would
 * report the opposite of the truth. So one value covers all three, authored for
 * the 6,400-cell member and carried unchanged by the 100-cell one.
 *
 * The derivation, in rows, for the largest member (the same shape as
 * `POWER_DROPLET_PRESSURE_ROW_CAPACITY`, evaluated on a 20x16x20 corner block):
 *
 *   liquid rows        20 * 16 * 20 unit owners                  =  6,400
 *   wall strip         three closed walls; the 3-cell strip
 *                      around the block is 26 * 22 * 26 - 6,400  =  8,472
 *   surface band       band 1 plus the Section 5 transport reach,
 *                      dry at t=0 and wetting as the block slumps
 *   2:1 balance        dry, coarsening outward: no rows
 *
 * 14,872 liquid-plus-wall rows against `planOctreePressureCapacity`'s own
 * footprint-shaped ask of 12,800. 65,536 is 4.4x the derivation and 5.1x the
 * planner, matching the droplet family's 3.4x headroom over its own
 * liquid-plus-wall term with extra room for a reservoir eight times deeper that
 * genuinely spreads. It is also a magnitude the codebase already runs:
 * `deep-power-hydrostatic` authors exactly 65,536 rows.
 *
 * Raising a reserve is not inert here ([[capacity-is-not-inert]]) — a lane has
 * failed at t=0 under a LARGER row capacity — which is precisely why this
 * number is also an experiment. `power-fill-256-100` is the same hundred cells
 * in the same 256-cubed container as `power-droplet-256`, differing ONLY in
 * carrying 65,536/65,536 where that lane carries 4,096/4,096. Diffing the two
 * per-label captures is a direct measurement of capacity-shaped GPU cost at
 * identical live occupancy, with no attribution guesswork at all.
 */
export const POWER_FILL_PRESSURE_ROW_CAPACITY = 65_536;

/**
 * Fine narrow-band brick reserve, likewise constant across the family.
 *
 * `planFluidFootprintFineNarrowBandBrickCapacity` asks 1,073 / 4,290 / 17,160
 * bricks for the three reservoirs. 65,536 is 3.8x the largest ask — the same
 * ratio the droplet family authors over its own (4,096 against 1,073) — and is
 * still 256x below the 16,777,216-brick logical lattice a 256-cubed container
 * carries at fine factor 4. Constant across members for the same reason as the
 * row reserve, and 16x the droplet family's value so the capacity control above
 * moves both reserves by the same factor.
 *
 * There is no environment path for this knob (`lib/methods/octree.ts` reads it
 * only from authored method values), so it must be carried by the authored
 * profile and by the smoke-catalog overrides; the benchmark reaches it through
 * the latter.
 */
export const POWER_FILL_FINE_BRICK_CAPACITY = 65_536;

/** Every surface discretization knob of the droplet family — interface band 1,
 * fine factor 4, 0.05 m cells, 0.004 s steps — so a per-label capture on a fill
 * lane is directly comparable to a droplet one. Only the two reserves differ,
 * and both are constants of the family rather than functions of the member. */
export const POWER_FILL_METHOD_PROFILE: MethodProfile = Object.freeze({
  ...LARGE_POWER_HYDROSTATIC_METHOD_PROFILE,
  overrides: Object.freeze({
    ...LARGE_POWER_HYDROSTATIC_METHOD_PROFILE.overrides,
    // The fill family is a fixed 256-cubed domain.
    maximumLeafSize: "32",
    pressureRowCapacity: POWER_FILL_PRESSURE_ROW_CAPACITY,
    globalFineLevelSetMaximumBricks: POWER_FILL_FINE_BRICK_CAPACITY,
  }),
});

/** The reservoir a member is built from, in finest cells. Keyed by the liquid
 * count because that count is the family's independent variable and the thing
 * its scene ids are named for. */
export function powerFillReservoirCells(liquidCells: PowerFillLiquidCells) {
  const index = POWER_FILL_LIQUID_CELLS.indexOf(liquidCells);
  const reservoir = POWER_FILL_RESERVOIR_CELLS[index];
  if (!reservoir) throw new RangeError(`No authored fill reservoir for ${liquidCells} liquid cells`);
  return reservoir;
}

/**
 * `liquidCells` cells of water in a fixed 256-cubed container.
 *
 * One factory for the same reason the droplet family has one: the only authored
 * difference between members must be the reservoir, or the measured slope is a
 * scene difference rather than a live-occupancy law.
 *
 * The corner dam-break with **no** `initialDamBreakOrigin_m` is mandatory and
 * not stylistic. `analyticSparseBootstrap` (`lib/webgpu-octree.ts`) requires no
 * brick seeds, no bodies, no terrain and a reservoir anchored at the container
 * corner; any offset origin flips the bootstrap dense — `Float32Array(nx*ny*nz)`
 * over 16.7M cells, a full-domain signed-distance transform and a ~134 MB phi
 * texture — which at 256 cubed would swamp the very measurement this family
 * exists to take. See `POWER_DROPLET_RESERVOIR_M` for the same argument at the
 * other end of the plane.
 */
export function createPowerFillScene(liquidCells: PowerFillLiquidCells): SceneDescription {
  const scene = cloneScene(defaultScene);
  scene.sceneId = `power-fill-${POWER_FILL_EDGE_CELLS}-${liquidCells}`;
  scene.duration_s = 1;
  scene.rigidBodies = [];
  const extent_m = POWER_FILL_EDGE_CELLS * POWER_DROPLET_CELL_SIZE_M;
  const cells = powerFillReservoirCells(liquidCells);
  // Built from the authored integer lattice times one spacing, never decimal
  // literals: the power catalog requires isotropic finest cells to 1e-5.
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
  scene.numerics.fixedDt_s = scene.numerics.maxDt_s = 0.004;
  return scene;
}

/** The Bet-4 work verdict is about avoided machinery, not surface accuracy: a
 * factor-1 surface keeps the frame's cost in the pressure path the census
 * measures, and band 1 keeps the refined shell off the deep interior. */
export const POWER_HYBRID_DEEP_OCEAN_METHOD_PROFILE: MethodProfile = Object.freeze({
  methodId: "losasso",
  quality: "balanced",
  overrides: Object.freeze({
    // 64x64x48: sixteen is the largest common dyadic leaf.
    maximumLeafSize: "16",
    interfaceRefinementBandCells: 1,
    globalFineLevelSetFactor: "1",
  }),
});

/** Product profile for the 320x96x80 rolling-wave scene. The old leaf-32 UI
 * suggestion could not divide the 80-cell depth; leaf 16 is the largest
 * Losasso hierarchy root common to all axes. */
export const OCEAN_SEICHE_METHOD_PROFILE: MethodProfile = Object.freeze({
  methodId: "losasso",
  quality: "balanced",
  overrides: Object.freeze({
    maximumLeafSize: "16",
  }),
});

/** The ceiling-drop UI oracle uses the narrow interface reach and the common
 * leaf-8 Losasso hierarchy shared by all four free-fall scenes. */
export const CEILING_DROP_METHOD_PROFILE: MethodProfile = Object.freeze({
  ...POWER_VALIDATION_METHOD_PROFILE,
  overrides: Object.freeze({
    ...POWER_VALIDATION_METHOD_PROFILE.overrides,
    // Both 24x16x24 and 24-cubed free-fall variants admit leaf 8.
    maximumLeafSize: "8",
    interfaceRefinementBandCells: 1,
  }),
});

/** Minimal 16-cubed Losasso coupling oracle profile. */
export const RIGID_COUPLING_ORACLE_METHOD_PROFILE: MethodProfile = Object.freeze({
  ...POWER_VALIDATION_METHOD_PROFILE,
  overrides: Object.freeze({
    ...POWER_VALIDATION_METHOD_PROFILE.overrides,
    maximumLeafSize: "8",
    interfaceRefinementBandCells: 4,
    globalFineLevelSetFactor: "1",
  }),
});

/** The larger offset tank needs one additional interface-support cell to keep
 * its adaptive Section 5 band inside complete catalog support. This mirrors the
 * isolated Dawn oracle instead of inheriting the tiny 16-cubed profile. */
export const LARGE_HYDROSTATIC_POWER_METHOD_PROFILE: MethodProfile = Object.freeze({
  ...POWER_VALIDATION_METHOD_PROFILE,
  overrides: Object.freeze({
    ...POWER_VALIDATION_METHOD_PROFILE.overrides,
    // 32x24x16: eight is the largest common dyadic leaf.
    maximumLeafSize: "8",
    interfaceRefinementBandCells: 4,
  }),
});

/** World-space centre of the single seeded 8-cubed fluid brick (the -x/-z quadrant). */
export const BRICK_QUAD_DAM_SEED_M = { x: -0.2, y: 0.2, z: -0.2 };

/**
 * Smallest authored hydrostatic oracle that still leaves room for adaptive
 * pressure cells away from the closed walls and planar free surface. The
 * 0.05 m lattice resolves the 0.8 m cube as exactly 16 cells per axis.
 */
export function createTinyHydrostaticScene(): SceneDescription {
  const scene = cloneScene(defaultScene);
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
  scene.numerics.fixedDt_s = scene.numerics.maxDt_s = 0.004;
  return scene;
}

/**
 * A larger hydrostatic oracle with an intentionally cell-cut free surface.
 * Its 32x24x16 lattice is deep enough for a materially larger adaptive
 * pressure layout while remaining small enough for an isolated Dawn smoke.
 */
export function createLargeHydrostaticScene(): SceneDescription {
  const scene = cloneScene(defaultScene);
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
  scene.numerics.fixedDt_s = scene.numerics.maxDt_s = 0.004;
  return scene;
}

/**
 * A minimal three-dimensional analytic dam break on the 16-cubed paper path.
 * A 23/64-domain reservoir has an exact ten-cell footprint: the smallest
 * clean rational fill that leaves a two-cell liquid interior after the
 * four-cell interface refinement band.
 * Keeping the authored dam initializer (rather than imported brick geometry)
 * preserves authoritative generalized-face projection at t=0.
 */
export function createMinimalPowerDamBreakScene(): SceneDescription {
  const scene = cloneScene(defaultScene);
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
  scene.numerics.fixedDt_s = scene.numerics.maxDt_s = 0.004;
  return scene;
}

/**
 * Smallest brick-authored power-octree scene with an exact horizontal D4
 * symmetry frame. A 32-cell dyadic domain and a centred 2x1x2 brick body are
 * the minimum that leave an equal one-brick gap to all four walls; a 24-cell
 * domain centres one brick geometrically but introduces a non-dyadic root
 * decomposition that is itself not an invariant of the octree algorithm.
 */
export function createSymmetricExpansionScene(): SceneDescription {
  const scene = cloneScene(defaultScene);
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
  // This is the Losasso-2004 fidelity oracle.  That method solves inviscid
  // Euler flow; zero the inherited water viscosity rather than presenting a
  // damped scene parameter that the reduced backend does not consume.
  scene.fluid.dynamicViscosity_Pa_s = 0;
  scene.fluid.surfaceTension_N_m = 0;
  scene.numerics.fixedDt_s = scene.numerics.maxDt_s = 0.004;
  return scene;
}

/**
 * The minimal dam break at four times the linear resolution. All physical
 * geometry, fluid properties, and time integration settings are inherited
 * unchanged; only the finest octree/surface lattice changes from 16³ to 64³.
 */
export function createMinimalPowerDamBreak64Scene(): SceneDescription {
  const scene = createMinimalPowerDamBreakScene();
  scene.sceneId = "minimal-power-dam-break-64";
  scene.voxelDomain = {
    ...scene.voxelDomain,
    finestCellSize_m: scene.voxelDomain.finestCellSize_m / 4,
  };
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
  scene.sceneId = "high-resolution-dam-break";
  scene.duration_s = 20;
  scene.voxelDomain = {
    ...scene.voxelDomain,
    finestCellSize_m: scene.voxelDomain.finestCellSize_m / 8,
  };
  return scene;
}

/** The same physical mini dam on the midpoint 32³ coarse-only lattice. */
export function createMinimalPowerDamBreak32Scene(): SceneDescription {
  const scene = createMinimalPowerDamBreakScene();
  scene.sceneId = "minimal-power-dam-break-32";
  scene.voxelDomain = {
    ...scene.voxelDomain,
    finestCellSize_m: scene.voxelDomain.finestCellSize_m / 2,
  };
  return scene;
}

/**
 * Canonical Sparse CM12 traversal scene. A full-width reservoir starts at the
 * negative end of a 192x96x32 tank, leaving twenty brick columns for the front to
 * cross before it reaches the far wall. The doubled vertical air column makes
 * sparse omission visually and numerically explicit; the authored DETAIL x2
 * keeps the initial liquid block physical size unchanged. The narrow transverse section
 * keeps a long residency test affordable while still exercising genuine 3D
 * pressure, transport, and 2:1 face ports.
 */
export function createSparseCM12LongDamBreakScene(): SceneDescription {
  const scene = createMinimalPowerDamBreakScene();
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
  scene.numerics.fixedDt_s = scene.numerics.maxDt_s = 0.004;
  return scene;
}

/**
 * Chentanez & Mueller's tall-cell Flood case uses a 64 x (32 + 2) x 256
 * simulation grid over uneven terrain (`docs/papers/tallCells.pdf`, Table 2
 * and Figure 1). Our sparse world does not compress a whole water column into
 * one tall cell, so this reconstruction rotates the paper's 256-cell travel
 * axis onto +X and gives the terrain and falling reservoir a 64-cell physical
 * height. The important experiment remains the same: a small active body moves
 * through a much larger world while solid ground varies by metres beneath it.
 */
export const TALL_CELLS_FLOOD_CELL_SIZE_M = 0.05;
export const TALL_CELLS_FLOOD_GRID = Object.freeze([256, 64, 64] as const);
export const TALL_CELLS_FLOOD_UPHILL_HEIGHT_M = 2;
export const TALL_CELLS_FLOOD_DOWNHILL_HEIGHT_M = 0.1;
export const TALL_CELLS_FLOOD_RESERVOIR_M = Object.freeze({
  x: 1.2,
  y: 0.8,
  z: 2.4,
});

/**
 * The Flood benchmark is the terrain, not a room around the terrain.
 *
 * Giving the scene its own shell also prevents `finishSceneDocument` from
 * attaching the Stage environment's scaled studio set. On this 12.8 m domain
 * that set grows the sparse presentation world far beyond the simulation
 * footprint and makes opening the scene allocate hundreds of megabytes of
 * unrelated dry-world lighting data before the first frame.
 */
const TALL_CELLS_FLOOD_SCENERY: SceneryGraph = Object.freeze({
  palettes: Object.freeze({}),
  nodes: Object.freeze([
    Object.freeze({
      kind: "terrain-shell" as const,
      id: "shell",
      materialModel: "porcelain" as const,
    }),
    // The ordinary stage graph cannot be attached here: its floor spans the
    // complete 12.8 m paper domain and used to dominate sparse-scene startup.
    // Retain only its practical. This bounded reflector covers the hillside,
    // publishes the same physical spot record to SVO lighting, and gives the
    // water compositor a real fixture key instead of the stage's dim blue fill.
    Object.freeze({
      kind: "cone" as const,
      id: "hillside/key",
      tags: Object.freeze(["lamp", "fixture", "light", "spot-light"]),
      place: Object.freeze({
        position: Object.freeze({ x: 0, y: 8.5, z: 0 }),
        units: "metres" as const,
      }),
      baseRadius: 1.5,
      topRadius: 0.2,
      halfHeight: 0.5,
      material: Object.freeze({
        colorLinear: Object.freeze([1, 0.955, 0.88] as const),
        emission: 800,
      }),
    }),
  ]),
});

/**
 * Authoring samples for one voxel solid: a flat launch shelf, a long central
 * drainage slope, and a short runout. The gentle transverse rise makes this
 * uneven ground rather than an inclined container floor, while keeping the
 * upper shelf flat so the initial water block rests on solid voxels everywhere.
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
  const slopeEndX_m = 0.5 * width_m - 0.8;
  const slopeRun_m = slopeEndX_m - slopeStartX_m;
  for (let z = 0; z <= nz; z += 1) {
    const worldZ_m = grid.origin_m.z + z * grid.spacing_m;
    const transverse = Math.abs(worldZ_m) / (0.5 * depth_m);
    for (let x = 0; x <= nx; x += 1) {
      const worldX_m = grid.origin_m.x + x * grid.spacing_m;
      const progress = Math.max(0, Math.min(1,
        (worldX_m - slopeStartX_m) / slopeRun_m));
      const slopeHeight_m = TALL_CELLS_FLOOD_UPHILL_HEIGHT_M
        + progress * (TALL_CELLS_FLOOD_DOWNHILL_HEIGHT_M
          - TALL_CELLS_FLOOD_UPHILL_HEIGHT_M);
      // Zero on both shelves, strongest halfway down the hill. This shallow
      // crossfall keeps the released sheet in view without becoming a vessel.
      const slopeWindow = 4 * progress * (1 - progress);
      const channelBank_m = 0.12 * slopeWindow * transverse * transverse;
      grid.heights_m.push(slopeHeight_m + channelBank_m);
    }
  }
  return {
    solidRepresentation: "voxel",
    baseHeight_m: TALL_CELLS_FLOOD_DOWNHILL_HEIGHT_M,
    features: [],
    grid,
  };
}

/**
 * Paper-inspired hillside dam break for the vast sparse-world path.
 *
 * The reservoir is exactly 3 x 2 x 6 B8 tiles and its origin is aligned to the
 * 50 mm lattice. Everything downhill begins dry. Fluid residency therefore
 * starts local even though the logical domain contains 1,048,576 cells, and a
 * travelling front has to create and later retire topology while terrain
 * remains authoritative for solid contact and rendering.
 */
export function createTallCellsHillsideDamBreakScene(): SceneDescription {
  const scene = cloneScene(defaultScene);
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
    top: "open",
    fluidWallMode: "free-slip",
    vessel: "none",
  };
  scene.voxelDomain = { finestCellSize_m: cell_m, brickSize_cells: 8 };
  scene.container.wallField = boxTankWallFieldForScene(scene);
  scene.nominalResolution = { length_m: cell_m };
  scene.terrain = createTallCellsFloodTerrain();
  scene.scenery = TALL_CELLS_FLOOD_SCENERY;
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
  // Preserve the clear-water look used by the small tank scenes. Optical
  // coefficients are rates per metre; applying their tank-scale defaults to a
  // 2.4 m reservoir makes an otherwise identical medium strongly cyan. This
  // ten-times-lower rate gives the large paper figure the same shallow optical
  // depth without changing the shared material used by existing scenes.
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
  return scene;
}

/**
 * The still-water partner of the room-sized dam-break scene. The tank and
 * lattice are identical to `large-power-dam-break`, while the initial liquid
 * is a full-footprint hydrostatic slab with exactly the same represented
 * order of volume as the mini/large dam reservoir. The mathematically exact
 * equal-volume slab is thinner than half a finest cell and therefore has no
 * liquid cell centre for the sparse bootstrap to seed. Use the smallest
 * representable slab instead. It is one 0.05 m cell deep with a 32x32-cell
 * footprint: 1,024 of 81,920 cells. This is one quarter of the earlier
 * full-footprint one-cell slab and keeps the benchmark genuinely sparse.
 */
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

/**
 * The 20x still-water tank made deep: the only lane that can express Bet 4.2.
 *
 * `large-power-hydrostatic` is one finest cell deep, so every one of its liquid
 * cells touches the floor *and* the free surface. It publishes 1.004 rows per
 * liquid cell and zero `regularInterior` rows, which makes the paper's Section
 * 8 interior-coarsening cost model unmeasurable on it — a hybrid discretization
 * that skips the power machinery on regular interior rows has nothing to skip.
 * This scene changes exactly two authored numbers: tank height 1.0 -> 2.4 m and
 * fill 1/80 -> 5/6. Footprint (3.2 x 3.2 m), lattice (0.05 m finest cell, 8-cell
 * bricks), closed lid, free-slip walls, zero surface tension and the 0.004 s
 * step are the 20x family's, unchanged, so the two lanes differ only in depth.
 *
 * Geometry, in finest cells. The domain is 64 x 48 x 64 = 196,608 and 5/6 of 48
 * puts the flat free surface exactly on the y = 40 cell boundary, so the liquid
 * is 64 x 40 x 64 = 163,840 cells (2.0 m of water). Classify those against the
 * two reaches the solver actually uses:
 *
 *   - closed-wall shell: `OCTREE_POWER_BOUNDARY_STRIP_MIN_CELLS` = 3 cells
 *     inside each of the four vertical walls and the floor are split to unit
 *     owners (the lid strip is dry).
 *   - free-surface band: the Section 5 transport band is
 *     `max(2, band) * fineFactor` = 8 fine cells = 2 coarse cells, plus one
 *     extension layer and one interpolation halo — the same reach = 4 the
 *     large-hydro air-support budget is derived from.
 *
 *   interior     = (64 - 2*3) * (40 - 3 - 4) * (64 - 2*3)
 *                = 58 * 33 * 58            = 111,012 cells = 67.8%
 *   surface band = 64 * 4 * 64             =  16,384 cells = 10.0%
 *   wall-only    = 163,840 - 111,012 - 16,384 = 36,444 cells = 22.2%
 *
 * Two thirds is the target, and 40 cells is close to the shallowest depth that
 * reaches it: the shell is 6 cells wide horizontally and 7 cells tall, so the
 * interior fraction is (1 - 6/64)^2 * (1 - 7/H) and drops below 2/3 for any
 * H < 38. There is no cheap deep scene — that is the finding, not a shortcut
 * being missed. Below the shell the octree coarsens: the CPU leaf model behind
 * `DEEP_POWER_HYDROSTATIC_PRESSURE_ROW_CAPACITY` puts 112,896 of the 163,840
 * liquid cells (69%) inside 6,832 coarse leaves, i.e. under 0.36 rows per cell.
 */
export function createDeepPowerHydrostaticScene(): SceneDescription {
  const scene = createLargePowerHydrostaticScene();
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
  return scene;
}

/**
 * One hundred cells of water in a container of `edgeCells` cubed.
 *
 * The whole family is one factory because the only authored difference between
 * its members is the container edge — if the reservoir, lattice, lid, wall
 * mode, surface tension or timestep could drift between two entries, the
 * measured slope would be a scene difference rather than a domain tax.
 *
 * Extents are `edgeCells * 0.05 m` computed from the authored integer lattice,
 * not decimal literals: the power catalog requires isotropic finest cells to
 * within 1e-5 (`lib/webgpu-octree.ts`), and a cube built by multiplying one
 * spacing by one integer is isotropic by construction. `fillFraction` is
 * likewise computed from the reservoir rather than written down, because
 * `validateScene` requires it to equal the authored dam-break volume fraction
 * to within 1e-9 and that fraction changes with every N (3.81e-4 at 64 down to
 * 5.96e-6 at 256).
 *
 * The physics is deliberately boring. A corner puddle five cells wide slumps
 * for a few steps, wets the floor and two walls, and settles; there is no
 * front to track, no splash, and no interface worth refining. That is the
 * instrument: whatever the wall does across the sweep is not the fluid.
 */
export function createPowerDropletScene(edgeCells: PowerDropletEdgeCells): SceneDescription {
  const scene = cloneScene(defaultScene);
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
  // The four deletions that keep `analyticSparseBootstrap` true. See
  // `POWER_DROPLET_RESERVOIR_M` for what each one would otherwise cost.
  delete scene.fluid.initialDamBreakOrigin_m;
  delete scene.fluid.initialBrickSeeds_m;
  delete scene.fluid.initialBrickSeedsAdditive;
  delete scene.fluid.inflow;
  scene.fluid.surfaceTension_N_m = 0;
  scene.numerics.fixedDt_s = scene.numerics.maxDt_s = 0.004;
  return scene;
}

/**
 * Bet-4 work-verdict scene: a genuinely volumetric pool, but bounded enough to
 * run as a one-step shipping-path gate. The topology-aligned 64x64x48 lattice
 * has 56 wet layers, so the regular deep interior dominates the free-surface
 * and tank-wall bands. It deliberately has no wave seed: D4 correctness remains
 * the separate symmetric-expansion gate, while this scene measures only avoided
 * machinery.
 */
export function createPowerHybridDeepOceanScene(): SceneDescription {
  const scene = cloneScene(defaultScene);
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
  scene.numerics.fixedDt_s = scene.numerics.maxDt_s = 0.004;
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
  const scene = cloneScene(defaultScene);
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
  scene.numerics.fixedDt_s = scene.numerics.maxDt_s = 0.004;
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
  const scene = cloneScene(defaultScene);
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
  scene.numerics.fixedDt_s = scene.numerics.maxDt_s = 0.004;
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
  const scene = cloneScene(defaultScene);
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

/**
 * A wide tank holding two reservoirs on diagonally opposite floor corners.
 * The finest solver grid is exactly 56x16x16 cells of 0.05 m (7x2x2 fluid
 * bricks); each dam is a 2x1x1-brick slab (0.8 x 0.4 x 0.4 m) leaving three
 * bricks — 1.2 m — of dry floor between them and one brick of headroom under
 * the closed lid. Released together they run down the long axis, and because
 * they are offset in z the fronts meet mid-tank at an angle rather than
 * head-on, so the collision produces an oblique churn instead of a symmetric
 * standing wall.
 *
 * The reservoirs are authored as explicit brick seeds rather than the analytic
 * `dam-break` initializer, which only ever builds one corner-anchored box.
 * Seeded geometry therefore takes the dense host bootstrap that terrain and
 * rigid-body scenes already use.
 *
 * Both extents are solver limits rather than taste, and both are pinned by
 * tests:
 *  - 14336 cells keeps the octree lane inside the bounded SPGrid V-cycle. At
 *    this domain size `planOctreePressureCapacity` saturates at the cell
 *    count, so the pressure plan is the tank volume itself. The V-cycle's row
 *    ceiling was 16,384 when this tank was authored; it is now
 *    `SPGRID_MAXIMUM_ROW_CAPACITY`, and the test below reads that constant
 *    rather than the number.
 *  - Eight-cell reservoirs stay inside the tall-cell method's default twelve
 *    regular layers. Columns taller than that band are compressed by the
 *    remesh, which loses most of their volume over half a second.
 */
export function createTwinDamCollisionScene(): SceneDescription {
  const scene = cloneScene(defaultScene);
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

/**
 * A wide ocean tank sized so the finest solver grid is exactly 320x96x80
 * cells of 0.025 m (40x12x10 fluid bricks). The pool fills to 72 cells
 * (1.8 m); a 2x1x10-brick slab of extra water (0.4 m wide, 0.2 m tall, full
 * depth extent) rests on the surface along the -x wall. Releasing it launches
 * a long gravity wave (~sqrt(gH) = 4.2 m/s) that crosses the tank in ~1.9 s
 * and reflects. The calm deep interior is exactly what large octree leaves
 * coarsen best: below the graded surface band the water collapses into
 * 16-cubed and 32-cubed pressure cells when the octree method's maximum leaf
 * is raised to 32.
 */
export function createOceanSeicheScene(): SceneDescription {
  const scene = cloneScene(defaultScene);
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
  const scene = applyGardenPool(cloneScene(defaultScene));
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

/**
 * The hero garden, vessel and set, as one document.
 *
 * The set is composed here rather than inside `createHeroGardenHoseScene`
 * because the layout depends on the vessel — it seats every prop through
 * `pondVesselHeightAt` and runs its pebble beds along `pondVesselPlanCurve` —
 * and a scene module that imported it back would close a cycle that dies in the
 * temporal dead zone. Vessel, then set, then catalog is the one ordering with no
 * arrow pointing backwards.
 *
 * Exported, and with the factory's own options, because a lane that wants this
 * scene at another lattice has exactly one place to ask. The alternative was the
 * one `tools/run-svo-dry-render-smoke.ts` used to take: build the preset, then
 * overwrite `voxelDomain.finestCellSize_m` on the finished document. That moves
 * the octree and nothing else — the heightfield is already baked at whatever
 * spacing the default asked for and every generator has already expanded at the
 * 25 mm leaf — so the lattice got finer and nothing that feeds it did. A size
 * has to be an *input* to construction, which is what this signature is for.
 * `FLUID_SVO_DRY_SMOKE_RECORD_MULTIPLIER` already set the precedent by calling
 * `createHeroGardenHoseStressScene` rather than editing a built scene.
 */
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

export type SparseCM12ComplexitySceneId =
  | "empty-16"
  | "full-16"
  | "cut-receiver-seam-16"
  | "double-x-32"
  | "planar-interface-16"
  | "first-split-bundle"
  | "recursive-split"
  | "receiver-create"
  | "receiver-activate"
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

/**
 * The visible projection of the Dawn complexity ladder. The scene factory
 * below is shared by the browser and the probe; these labels are kept here too
 * so adding a rung cannot leave it hidden from the scene library.
 */
export const SPARSE_CM12_COMPLEXITY_SCENES: readonly SparseCM12ComplexityScene[] = Object.freeze([
  { id: "empty-16", ordinal: 0, title: "Empty complete B16/P16 domain",
    introducedFeature: "construction and advance with no accepted fluid ownership", defaultSteps: 2 },
  { id: "full-16", ordinal: 1, title: "Smallest fully wet quiescent domain",
    introducedFeature: "one complete B16/P16 macro owner, one accepted cell and no pressure rows", defaultSteps: 2 },
  { id: "cut-receiver-seam-16", ordinal: 2, title: "Smallest production cut/receiver/seam bundle",
    introducedFeature: "smallest one-cell cut, dry receivers, cross-owner rows and mixed seams", defaultSteps: 2 },
  { id: "double-x-32", ordinal: 3, title: "Doubled x extent",
    introducedFeature: "double domain/reservoir x extent with the same owner closure", defaultSteps: 2 },
  { id: "planar-interface-16", ordinal: 4, title: "Planar interface fanout",
    introducedFeature: "larger free-surface support and row fanout", defaultSteps: 2 },
  { id: "first-split-bundle", ordinal: 5, title: "First production split bundle",
    introducedFeature: "first accepted moving rung/split/receiver transaction", defaultSteps: 8 },
  { id: "recursive-split", ordinal: 6, title: "Recursive split lineage",
    introducedFeature: "a child created in one generation later becomes a split parent", defaultSteps: 40 },
  { id: "receiver-create", ordinal: 7, title: "Receiver creation beyond the support halo",
    introducedFeature: "front travel beyond rings9 commits receiver creation", defaultSteps: 160 },
  { id: "receiver-activate", ordinal: 8, title: "Receiver activation",
    introducedFeature: "a later moving front consumes an existing receiver", defaultSteps: 240 },
  { id: "row-only-replace", ordinal: 9, title: "Row-only seam replacement",
    introducedFeature: "row pages change while accepted cell IDs and scalar bits remain exact", defaultSteps: 40 },
  { id: "retire-reuse", ordinal: 10, title: "Stable-slot retirement and reuse",
    introducedFeature: "a retired owner/page slot reappears at a newer generation", defaultSteps: 300 },
  { id: "symmetric-2d", ordinal: 11, title: "Two-dimensional symmetric expansion",
    introducedFeature: "symmetric front evolution with a depth symmetry boundary", defaultSteps: 20 },
  { id: "symmetric-3d", ordinal: 12, title: "Three-dimensional D4 symmetric expansion",
    introducedFeature: "the production D4 correctness oracle", defaultSteps: 20 },
  { id: "brick-quad", ordinal: 13, title: "Brick-quad moving topology",
    introducedFeature: "four-owner x/z-transpose topology and receiver churn", defaultSteps: 20 },
  { id: "long-dam", ordinal: 14, title: "Long dam traversal",
    introducedFeature: "large sparse moving front with repeated topology transactions", defaultSteps: 30 },
]);

const SPARSE_CM12_LADDER_CELL_M = 0.05;

function createSparseCM12LadderBox(
  id: SparseCM12ComplexitySceneId,
  dimensions: readonly [number, number, number],
  fillFraction: number,
  options: Readonly<{ gravity?: boolean; dam?: readonly [number, number, number] }> = {},
): SceneDescription {
  const scene = cloneScene(defaultScene);
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
  scene.numerics.fixedDt_s = scene.numerics.maxDt_s = 0.004;
  return scene;
}

/** One construction route for both Dawn ladder runs and UI scene cards. */
export function createSparseCM12ComplexityScene(id: SparseCM12ComplexitySceneId): SceneDescription {
  switch (id) {
    case "empty-16":
      return createSparseCM12LadderBox(id, [16, 16, 16], 0, { gravity: false });
    case "full-16":
      return createSparseCM12LadderBox(id, [16, 16, 16], 1, { gravity: false });
    case "cut-receiver-seam-16":
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
    case "receiver-create":
    case "receiver-activate":
    case "retire-reuse":
      return createSparseCM12LadderBox("receiver-activate", [64, 32, 16], 1 / 64,
        { gravity: true, dam: [8, 8, 16] });
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
      const scene = cloneScene(defaultScene);
      scene.rigidBodies = [];
      return scene;
    },
    camera: studioStageCamera,
  }),
  defineScene({
    id: "water-box-tank-fill",
    name: "Water box · settled tank",
    blurb: "The same container starting from a settled fill; drop bodies into calm water.",
    audience: "explore",
    shelf: "Tanks",
    environment: "bathhouse",
    build: () => {
      const scene = cloneScene(defaultScene);
      scene.sceneId = "interactive-water-box-settled";
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
    blurb: "A compact water column releases into a much wider closed voxel pool. The receiving area is finite and present from frame zero, making sparse wet-region activation easy to watch and safe to run.",
    audience: "explore",
    shelf: "Tanks",
    environment: "stage",
    methodProfile: BOUNDED_POOL_TRANSFER_METHOD_PROFILE,
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
    blurb: "A paper-inspired reservoir releases from a 2 m shelf down a 256-cell uneven slope. Height samples populate one unified voxel solid; the million-cell logical world starts with only a 3×2×6-tile water body so the travelling front exercises dynamic sparse-tile creation and retirement.",
    audience: "study",
    shelf: "Tall-cell studies",
    environment: "stage",
    presentationMode: "full-scene",
    methodProfile: SPARSE_CM12_COMPLEXITY_LADDER_METHOD_PROFILE,
    build: createTallCellsHillsideDamBreakScene,
    camera: {
      azimuth_rad: 0.48,
      elevation_rad: 0.3,
      distance_m: 14.5,
      target_m: { x: 0, y: 1.0, z: 0 },
    },
  }),
  defineScene({
    id: "hero-garden-hose",
    name: "Porcelain pond · hose filling",
    blurb: "A raised porcelain pond with a hose over it. The coping, the basin and the ground are one generated heightfield, so the water meets the rim exactly where the rim is. Opens dry — turn Water on under Scene configuration · Fluid to fill it.",
    audience: "explore",
    shelf: "Garden",
    environment: "garden",
    // The set *is* the scene here — a generated heightfield vessel, its coping,
    // its beds and its props — and it opens dry, so the default fluid-only path
    // would present an empty frame.
    presentationMode: "full-scene",
    // Dry by default. The set is what is ready to be looked at; the water is
    // still in bring-up, and a scene that opens is worth more than a scene that
    // opens correctly. `systems.fluid` is the switch, and nothing else differs.
    //
    // The set is composed here rather than inside `createHeroGardenHoseScene`
    // because the layout depends on the vessel — it seats every prop through
    // `pondVesselHeightAt` and runs its pebble beds along `pondVesselPlanCurve`
    // — and a scene module that imported it back would close a cycle that dies
    // in the temporal dead zone. Vessel, then set, then catalog is the one
    // ordering with no arrow pointing backwards. Appending rather than mutating
    // matters too: `scene.scenery` is a module-level constant shared by every
    // document the factory produces.
    build: () => createHeroGardenHoseSceneWithSet(),
    // The one route to this scene at another lattice, and the reason it is a
    // member of the definition rather than a call somewhere: the terrain bake,
    // the layout composed against it and every generator's legibility floor are
    // resolved while the document is built, so the sizes have to arrive before
    // it exists. `tools/run-svo-dry-render-smoke.ts` already reached the factory
    // directly for `FLUID_SVO_DRY_SMOKE_REFINEMENT`; this is the same call, from
    // the side the product loads scenes on.
    buildAt: (lattice: SceneLattice) => createHeroGardenHoseSceneWithSet(lattice),
    camera: heroGardenCamera,
    // No GPU lane yet, and so deliberately no variant: an authored variant that
    // no lane claims is a fork by another name. The settling lane returns with
    // the inner-face fix that phase 1 is currently blocked on — see
    // docs/HERO_GARDEN_HOSE_SCENE_PLAN.md.
  }),
  defineScene({
    id: "hero-garden-hose-x10",
    name: "Porcelain pond · ten times the set",
    blurb: "The same pond, the same camera, and ten times as many authored records standing on its banks. The acceptance scene for raster visibility: the hero frame spends 85 % of itself on a pass whose cost tracks record count, and this is where that stops being an argument.",
    // Disclosed rather than offered: it is a measurement rig, and it opens at
    // ~5 000 records on a path that is currently 405 ms at 501.
    //
    // It also does not open *yet*, and that is the point of registering it. The
    // live sparse world refuses 5 039 records at construction — its PBR material
    // table holds 32 + 4 096 (`webgpu-octree-sparse-bricks.ts:229`) — so the
    // acceptance scene names a capacity that has to be raised rather than
    // quietly sitting under it. 8x is the largest rung that draws today; the
    // smoke lane's FLUID_SVO_DRY_SMOKE_RECORD_MULTIPLIER walks the sweep.
    audience: "validation",
    shelf: "Garden",
    environment: "garden",
    // The dry world is the measurement, so it has to be constructed and drawn.
    presentationMode: "full-scene",
    // `heroGardenCamera`, unchanged and deliberately so — W3's gate is stated
    // against an *unchanged camera*, and a stress scene that reframed itself
    // could not discharge it. The factory composes the blocking layout itself,
    // so unlike the hero above there is nothing to append here.
    build: () => createHeroGardenHoseStressScene({
      recordMultiplier: HERO_GARDEN_STRESS_MAXIMUM_MULTIPLIER,
    }),
    buildAt: (lattice: SceneLattice) => createHeroGardenHoseStressScene({
      recordMultiplier: HERO_GARDEN_STRESS_MAXIMUM_MULTIPLIER,
      ...lattice,
    }),
    camera: heroGardenCamera,
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
      methodProfile: cm12MethodProfile(figure),
      build: () => cm12Scene(figure.id),
      camera: cm12Camera(cm12Scene(figure.id)),
    });
  }),
  defineScene({
    id: "deep-water-ab",
    name: "Deep-water A/B",
    blurb: "20 m tank at 80% fill, 1/30 s paper step, σ = 0 · isolates the grid method.",
    audience: "study",
    shelf: "Method comparisons",
    environment: "research-station",
    build: () => {
      const scene = cloneScene(defaultScene);
      scene.sceneId = "deep-water-grid-comparison";
      scene.container.height_m = 20;
      scene.container.fillFraction = 0.8;
      scene.fluid.initialCondition = "tank-fill";
      // The tall-cell paper does not include a capillary-force discretization.
      // Keep the A/B preset within that shared physical scope so the grid and
      // pressure methods are the only variables in the comparison.
      scene.fluid.surfaceTension_N_m = 0;
      scene.numerics.fixedDt_s = 1 / 30;
      scene.numerics.maxDt_s = scene.numerics.fixedDt_s * DEFAULT_GPU_CPU_TIMESTEP_RATIO;
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
    name: "Octree · tiny hydrostatic",
    blurb: "A 16³ settled tank for the first coarse-pressure oracle. Leaf 16 is the largest hierarchy root that divides the authored domain, while interface band 3 keeps the surface support explicit.",
    audience: "validation",
    shelf: "Hydrostatic oracles",
    environment: "stage",
    methodProfile: POWER_VALIDATION_METHOD_PROFILE,
    build: createTinyHydrostaticScene,
    camera: { distance_m: 1.85, target_m: { x: 0, y: 0.35, z: 0 } },
  }),
  defineScene({
    id: "hydrostatic-power-large-offset",
    name: "Octree · larger hydrostatic",
    blurb: "A 32x24x16 settled tank with a cell-cut free surface. Leaf 8 is the largest hierarchy root shared by all three axes, while exercising a larger adaptive pressure layout than the tiny oracle.",
    audience: "validation",
    shelf: "Hydrostatic oracles",
    environment: "stage",
    methodProfile: LARGE_HYDROSTATIC_POWER_METHOD_PROFILE,
    build: createLargeHydrostaticScene,
    camera: { distance_m: 2.75, target_m: { x: 0, y: 0.5, z: 0 } },
  }),
  defineScene({
    id: "large-power-hydrostatic",
    name: "Octree · 20× hydrostatic",
    blurb: "The 20× dam tank with a representable one-cell-deep 32×32 sparse pool (1,024 finest-cell volumes), completing the large-scene/minimal-liquid benchmark cell.",
    audience: "validation",
    shelf: "Hydrostatic oracles",
    environment: "stage",
    methodProfile: LARGE_POWER_HYDROSTATIC_METHOD_PROFILE,
    build: createLargePowerHydrostaticScene,
    camera: { distance_m: 6.4, target_m: { x: 0, y: 0.2, z: 0 } },
  }),
  defineScene({
    id: "deep-power-hydrostatic",
    name: "Octree · deep hydrostatic",
    blurb: "The 20× still tank filled 2 m deep (64×48×64, 163,840 liquid cells). 67.8% of its liquid is interior — away from every wall strip and the surface band — so it is the only authored lane on which interior coarsening, and the Section 8 hybrid cost model, can be measured at all.",
    audience: "validation",
    shelf: "Hydrostatic oracles",
    environment: "stage",
    methodProfile: DEEP_POWER_HYDROSTATIC_METHOD_PROFILE,
    build: createDeepPowerHydrostaticScene,
    camera: { distance_m: 7.4, target_m: { x: 0, y: 1, z: 0 } },
  }),
  ...POWER_DROPLET_EDGE_CELLS.map((edgeCells) => defineScene({
    id: `power-droplet-${edgeCells}`,
    name: `Octree · droplet in a ${edgeCells}³ domain`,
    blurb: `One hundred cells of water — a 5×4×5 corner puddle — in a ${edgeCells}³ container (${(edgeCells * POWER_DROPLET_CELL_SIZE_M).toFixed(1)} m on a side). The fluid is identical across the family, so the only thing that changes between these scenes is how much empty space the solver is asked to carry. Wall against domain volume across the sweep is the domain tax, measured directly.`,
    audience: "validation",
    // Its own shelf rather than the hydrostatic one: these four are a single
    // instrument read across its members, not four scenes anyone would open
    // for their own sake. The shelf is the unit of meaning here.
    shelf: "Domain-tax sweep",
    environment: "stage",
    methodProfile: POWER_DROPLET_METHOD_PROFILE,
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
      name: `Octree · ${liquidCells.toLocaleString("en-US")}-cell fill of a ${POWER_FILL_EDGE_CELLS}³ domain`,
      blurb: `${liquidCells.toLocaleString("en-US")} cells of water — a ${cells.x}×${cells.y}×${cells.z} corner reservoir — in a fixed ${POWER_FILL_EDGE_CELLS}³ container (${(POWER_FILL_EDGE_CELLS * POWER_DROPLET_CELL_SIZE_M).toFixed(1)} m on a side) at a capacity shared with every member. The container, the lattice and both reserves are identical across the family, so the only thing that changes is how much live water there is. A GPU pass measured across the sweep is flat (fixed overhead), linear (honest work) or superlinear (a defect), and nothing else.`,
      audience: "validation",
      // Its own shelf, beside the domain-tax sweep: these three are one
      // instrument read across its members, and the pair of shelves is the
      // pair of axes.
      shelf: "Live-occupancy sweep",
      environment: "stage",
      methodProfile: POWER_FILL_METHOD_PROFILE,
      build: () => createPowerFillScene(liquidCells),
      camera: {
        distance_m: POWER_FILL_EDGE_CELLS * POWER_DROPLET_CELL_SIZE_M * 1.6,
        target_m: { x: 0, y: POWER_FILL_EDGE_CELLS * POWER_DROPLET_CELL_SIZE_M * 0.1, z: 0 },
      },
    });
  }),
  defineScene({
    id: "power-hybrid-deep-ocean",
    name: "Octree · hybrid deep ocean",
    blurb: "A 64×64×48 pool filled to 56 wet layers, with no wave seed at all. The regular deep interior dominates the free-surface and wall bands, which is what makes one step of it a verdict on the machinery a hybrid discretization avoids rather than on surface accuracy.",
    audience: "validation",
    shelf: "Hydrostatic oracles",
    environment: "stage",
    methodProfile: POWER_HYBRID_DEEP_OCEAN_METHOD_PROFILE,
    build: createPowerHybridDeepOceanScene,
    camera: { distance_m: 7.4, target_m: { x: 0, y: 1.4, z: 0 } },
  }),
  ...SPARSE_CM12_COMPLEXITY_SCENES.map((rung) => defineScene({
    id: `sparse-cm12-ladder-${rung.id}`,
    name: `Sparse CM12 ${String(rung.ordinal).padStart(2, "0")} · ${rung.title}`,
    blurb: `${rung.introducedFeature}. Dawn's default observation window is ${rung.defaultSteps} solver ${rung.defaultSteps === 1 ? "step" : "steps"}.`,
    audience: "validation",
    shelf: "Sparse CM12 complexity ladder",
    environment: "stage",
    methodProfile: SPARSE_CM12_COMPLEXITY_LADDER_METHOD_PROFILE,
    build: () => createSparseCM12ComplexityScene(rung.id),
  })),
  defineScene({
    id: "minimal-power-dam-break",
    name: "Minimal dam break",
    blurb: "The analytic 12.5%-volume dam initializer collapses inside a 16³ tank, providing the smallest dynamic solver-comparison scene.",
    audience: "validation",
    shelf: "Dam-break ladder",
    environment: "stage",
    methodProfile: POWER_VALIDATION_METHOD_PROFILE,
    build: createMinimalPowerDamBreakScene,
    camera: { distance_m: 1.9, target_m: { x: 0, y: 0.3, z: 0 } },
  }),
  defineScene({
    id: "minimal-power-dam-break-32",
    name: "Minimal dam break 32³",
    blurb: "The same 0.8 m analytic mini dam at 0.025 m resolution, suitable for comparing the Uniform reference lane with the retained Octree profile.",
    audience: "validation",
    shelf: "Dam-break ladder",
    environment: "stage",
    methodProfile: COARSE_ONLY_POWER_DAM_METHOD_PROFILE,
    build: createMinimalPowerDamBreak32Scene,
    camera: { distance_m: 1.9, target_m: { x: 0, y: 0.3, z: 0 } },
  }),
  defineScene({
    id: "minimal-power-dam-break-64",
    name: "Minimal dam break 64³",
    blurb: "The same 0.8 m analytic mini dam at 0.0125 m resolution, suitable for comparing the Uniform reference lane with the retained Octree profile.",
    audience: "validation",
    shelf: "Dam-break ladder",
    environment: "stage",
    methodProfile: COARSE_ONLY_POWER_DAM_METHOD_PROFILE,
    build: createMinimalPowerDamBreak64Scene,
    camera: { distance_m: 1.9, target_m: { x: 0, y: 0.3, z: 0 } },
  }),
  defineScene({
    id: "sparse-cm12-long-dam-break",
    name: "Sparse CM12 · long-tank dam break",
    blurb: "A 192x96x32 tall, narrow tank with a full-width reservoir at the negative end. Its doubled empty air column showcases sparse omission while the canonical gate follows the front across twenty initially dry brick columns and checks resident 2:1 transitions, conservation, and Uniform comparison checkpoints.",
    audience: "validation",
    shelf: "Dam-break ladder",
    environment: "stage",
    methodProfile: SPARSE_CM12_LONG_DAM_METHOD_PROFILE,
    build: createSparseCM12LongDamBreakScene,
    camera: { distance_m: 4.35, target_m: { x: 0, y: 0.5, z: 0 } },
  }),
  defineScene({
    id: "large-power-dam-break",
    name: "Octree · 20× dam break",
    blurb: "The mini dam break's exact water block in a tank with 20× the volume: 4× longer, 4× wider, and 25% taller, using the compatible leaf-4 hierarchy and a band-1 interface.",
    audience: "validation",
    shelf: "Dam-break ladder",
    environment: "stage",
    methodProfile: LARGE_POWER_DAM_METHOD_PROFILE,
    build: createLargePowerDamBreakScene,
    camera: { distance_m: 6.4, target_m: { x: 0, y: 0.45, z: 0 } },
  }),
  defineScene({
    id: "symmetric-expansion",
    name: "Octree · symmetric expansion",
    blurb: "One exact central 2×1×2-brick water body collapses across the minimum dyadic 32×16×32 tank using the default factor-1 coarse surface and level-4 coupled reach. Dawn checks D4 symmetry of volume, velocity, pressure, topology, and four-wall contact after every step.",
    audience: "validation",
    shelf: "Symmetry",
    environment: "stage",
    // Watch this one first if the stage ever eats a publication. The set's floor
    // puts its top face on y=0 because that is a cell plane, and this oracle's
    // centred seeded bricks share that plane; against the room shell this scene
    // used to be presented in, the floor could win the depth test and turn a
    // healthy water publication into an apparently empty tank. The house set
    // builds no room faces at all now and the failure has not been seen since,
    // but this is the scene it was seen on, and it is the one where a rendering
    // artifact would be read as a physics result.
    methodProfile: SYMMETRIC_EXPANSION_METHOD_PROFILE,
    build: createSymmetricExpansionScene,
    camera: { distance_m: 2.5, target_m: { x: 0, y: 0.25, z: 0 } },
  }),
  defineScene({
    id: "sparse-cm12-symmetric-expansion",
    name: "Sparse CM12 · symmetric expansion",
    blurb: "The exact 32×16×32 D4 expansion oracle on the production Sparse CM12 B16/P16 resident. The card uses the real-time production pressure budget; a separate 108-iteration Dawn lane preserves the stricter accuracy oracle.",
    audience: "validation",
    shelf: "Symmetry",
    environment: "stage",
    methodProfile: SPARSE_CM12_SYMMETRIC_EXPANSION_METHOD_PROFILE,
    build: createSymmetricExpansionScene,
    camera: { distance_m: 2.5, target_m: { x: 0, y: 0.25, z: 0 } },
  }),
  defineScene({
    id: "ceiling-slab-drop",
    name: "Octree · ceiling drop oracle",
    blurb: "One 8³ fluid brick seeded flush under the closed lid, touching nothing else. The exact answer is free fall (y = y₀ − gt²/2, impact ≈0.29 s); any hesitation is wall adhesion.",
    audience: "validation",
    shelf: "Free-fall contact",
    environment: "stage",
    methodProfile: CEILING_DROP_METHOD_PROFILE,
    build: createCeilingSlabDropScene,
    camera: { distance_m: 2.4, target_m: { x: 0, y: 0.4, z: 0 } },
  }),
  defineScene({
    id: "rigid-hydrostatic",
    name: "Rigid coupling · hydrostatic sphere",
    blurb: "A fully submerged static sphere measures pressure buoyancy, volume displacement, and publication liveness.",
    audience: "validation", shelf: "Rigid coupling", environment: "stage",
    methodProfile: RIGID_COUPLING_ORACLE_METHOD_PROFILE,
    build: createRigidHydrostaticScene,
    camera: { distance_m: 1.8, target_m: { x: 0, y: 0.3, z: 0 } },
  }),
  defineScene({
    id: "rigid-float",
    name: "Rigid coupling · buoyant sphere",
    blurb: "A half-density sphere released at the waterline checks settling, drag, and plunge-through.",
    audience: "validation", shelf: "Rigid coupling", environment: "stage",
    methodProfile: RIGID_COUPLING_ORACLE_METHOD_PROFILE,
    build: createRigidFloatScene,
    camera: { distance_m: 1.8, target_m: { x: 0, y: 0.4, z: 0 } },
  }),
  defineScene({
    id: "rigid-sink",
    name: "Rigid coupling · dense sphere drop",
    blurb: "A dense sphere enters from above and checks bounded splash, conserved displacement, and floor arrival.",
    audience: "validation", shelf: "Rigid coupling", environment: "stage",
    methodProfile: RIGID_COUPLING_ORACLE_METHOD_PROFILE,
    build: createRigidSinkScene,
    camera: { distance_m: 1.8, target_m: { x: 0, y: 0.4, z: 0 } },
  }),
  defineScene({
    id: "corner-brick-drop",
    name: "Octree · corner drop oracle",
    blurb: "One 8³ fluid brick seeded into a top corner: lid, two walls, and their vertical edge seam. Frictionless walls exert only normal force, so the exact answer is still free fall.",
    audience: "validation",
    shelf: "Free-fall contact",
    environment: "stage",
    methodProfile: CEILING_DROP_METHOD_PROFILE,
    build: createCornerBrickDropScene,
    camera: { distance_m: 2.4, target_m: { x: 0, y: 0.4, z: 0 } },
  }),
  defineScene({
    id: "midair-brick-drop",
    name: "Octree · mid-air drop control",
    blurb: "The same 8³ brick hanging in open space, touching no boundary at all. The zero-contact control: whatever it deviates from free fall is the scheme's own transient, not adhesion.",
    audience: "validation",
    shelf: "Free-fall contact",
    environment: "stage",
    methodProfile: CEILING_DROP_METHOD_PROFILE,
    build: createMidairBrickDropScene,
    camera: { distance_m: 3, target_m: { x: 0, y: 0.6, z: 0 } },
  }),
  defineScene({
    id: "midair-corner-drop",
    name: "Octree · mid-air corner oracle",
    blurb: "The same 8³ brick against two vertical walls and their seam, but clear of the lid. Gravity is tangential to both walls, so free-slip walls cannot slow it: this isolates seam adhesion from ceiling adhesion.",
    audience: "validation",
    shelf: "Free-fall contact",
    environment: "stage",
    methodProfile: CEILING_DROP_METHOD_PROFILE,
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
    methodProfile: definition.methodProfile,
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
