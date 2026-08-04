import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { validateScene } from "../lib/model";
import { getSceneWebGPUSmokeLane } from "../lib/scene-webgpu-smoke-catalog";
import {
  DEEP_POWER_HYDROSTATIC_FINE_BRICK_CAPACITY,
  DEEP_POWER_HYDROSTATIC_METHOD_PROFILE,
  DEEP_POWER_HYDROSTATIC_PRESSURE_ROW_CAPACITY,
  LARGE_POWER_HYDROSTATIC_METHOD_PROFILE,
  createDeepPowerHydrostaticScene,
  createLargePowerHydrostaticScene,
  getScenePreset,
} from "../lib/scenes";
import { createTallCellLayout, type GPUQuality } from "../lib/tall-cell-grid";
import {
  OCTREE_POWER_BOUNDARY_STRIP_MIN_CELLS,
  planFluidFootprintFineNarrowBandBrickCapacity,
  planGlobalFineNarrowBandBrickCapacity,
  planOctreeFluidFootprintBudget,
  planOctreePressureCapacity,
} from "../lib/webgpu-octree";
import {
  octreeAirSupportFootprintCapacity,
  planOctreeAirVelocitySupportGPU,
} from "../lib/webgpu-octree-air-velocity-support-gpu";
import {
  planFineLevelSetBandFineCells,
  planFineLevelSetCapacityDilationBrickRings,
} from "../lib/webgpu-octree-fine-levelset-topology";
import { POWER_DAM_LANE_ENVIRONMENT } from "../tools/power-dam-lane-environment";

const DIMENSIONS = { nx: 64, ny: 48, nz: 64 } as const;
const GRID = [DIMENSIONS.nx, DIMENSIONS.ny, DIMENSIONS.nz] as const;
const DOMAIN_CELLS = DIMENSIONS.nx * DIMENSIONS.ny * DIMENSIONS.nz;
const WET_LAYERS = 40;
const LIQUID_CELLS = DIMENSIONS.nx * WET_LAYERS * DIMENSIONS.nz;
const FINE_FACTOR = 4;
const BAND_CELLS = 1;

/** The two reaches that decide whether a liquid cell can ever be coarsened:
 * the closed-wall strip that is split to unit owners, and the air-support reach
 * below the free surface (transport band + one extension layer + one
 * interpolation halo), which is the same reach the 20x still lane's air-support
 * budget is derived from. */
function reaches() {
  const strip = Math.max(OCTREE_POWER_BOUNDARY_STRIP_MIN_CELLS, BAND_CELLS);
  const bands = planFineLevelSetBandFineCells(BAND_CELLS, FINE_FACTOR);
  const surface = bands.transportBandFineCells / FINE_FACTOR + 1 + 1;
  return { strip, surface };
}

/** Interior / surface-band / wall-only split of a full-footprint tank fill. */
function classifyTankFill(nx: number, nz: number, wet: number) {
  const { strip, surface } = reaches();
  const interior = Math.max(0, nx - 2 * strip) * Math.max(0, wet - strip - surface)
    * Math.max(0, nz - 2 * strip);
  const band = nx * Math.min(wet, surface) * nz;
  return { interior, band, wall: nx * wet * nz - interior - band };
}

test("the deep lane changes only depth against the 20x still lane", () => {
  const shallow = createLargePowerHydrostaticScene();
  const deep = createDeepPowerHydrostaticScene();
  assert.deepEqual(validateScene(deep), []);
  assert.equal(deep.sceneId, "deep-power-hydrostatic");
  // Everything that is not depth must be byte-identical to the lane it is
  // meant to be compared against, or the pair measures two things at once.
  assert.deepEqual({ ...deep.container, height_m: shallow.container.height_m,
    fillFraction: shallow.container.fillFraction }, shallow.container);
  assert.deepEqual(deep.voxelDomain, shallow.voxelDomain);
  assert.equal(deep.numerics.maxDt_s, shallow.numerics.maxDt_s);
  assert.equal(deep.numerics.fixedDt_s, shallow.numerics.fixedDt_s);
  assert.equal(deep.fluid.surfaceTension_N_m, 0);
  assert.deepEqual(deep.rigidBodies, []);
  assert.equal(deep.fluid.inflow, undefined);
  // A full-footprint fill, not the shallow lane's corner reservoir.
  assert.equal(deep.fluid.initialCondition, "tank-fill");
  assert.equal(deep.fluid.initialDamBreakDimensions_m, undefined);
  assert.equal(deep.fluid.initialBrickSeeds_m, undefined);
  assert.equal(deep.container.height_m, 2.4);
  assert.equal(deep.container.fillFraction, 5 / 6);

  for (const quality of ["balanced", "high", "ultra"] as GPUQuality[]) {
    const layout = createTallCellLayout(deep, quality);
    assert.deepEqual([layout.nx, layout.fineNy, layout.nz], [...GRID]);
  }
  const footprint = planOctreeFluidFootprintBudget(deep, DIMENSIONS);
  assert.equal(footprint.initialLiquidCells, LIQUID_CELLS);
  assert.equal(footprint.inflowLiquidCells, 0);
  // 5/6 of 48 lands exactly on the y = 40 cell boundary; the nearest cell
  // centres are half a cell clear on both sides, so this cannot round either way.
  assert.deepEqual(footprint.maximumCell.map((value, axis) =>
    value - footprint.minimumCell[axis]!), [64, WET_LAYERS, 64]);
});

test("two thirds of the deep lane's liquid is interior; none of the shallow lane's is", () => {
  const { strip, surface } = reaches();
  assert.equal(strip, 3);
  assert.equal(surface, 4);

  const deep = classifyTankFill(DIMENSIONS.nx, DIMENSIONS.nz, WET_LAYERS);
  // interior = (64 - 2*3) * (40 - 3 - 4) * (64 - 2*3) = 58 * 33 * 58
  assert.deepEqual(deep, { interior: 111_012, band: 16_384, wall: 36_444 });
  assert.equal(deep.interior + deep.band + deep.wall, LIQUID_CELLS);
  assert.ok(deep.interior / LIQUID_CELLS > 2 / 3,
    `interior fraction ${deep.interior / LIQUID_CELLS} must clear two thirds`);

  // The reason this scene had to be authored: the 20x still lane's slab is one
  // cell deep, so every one of its liquid cells is inside both reaches at once
  // and a hybrid discretization that skips interior machinery skips nothing.
  const shallow = classifyTankFill(32, 32, 1);
  assert.equal(shallow.interior, 0);
  assert.equal(shallow.band, 1_024);

  // The shell is 6 cells wide horizontally and 7 tall, so a shallower column
  // cannot reach two thirds however wide the tank is made.
  const ceiling = (1 - 6 / DIMENSIONS.nx) * (1 - 6 / DIMENSIONS.nz);
  assert.ok(ceiling * (1 - 7 / 37) < 2 / 3);
  assert.ok(ceiling * (1 - 7 / WET_LAYERS) > 2 / 3);
});

test("the deep lane's reserves are derived from its fluid footprint, not raised to fit", () => {
  const scene = createDeepPowerHydrostaticScene();
  const footprint = planOctreeFluidFootprintBudget(scene, DIMENSIONS);
  const planned = planOctreePressureCapacity(DIMENSIONS, 32, BAND_CELLS, undefined, true,
    scene.container.fillFraction, Number.MAX_SAFE_INTEGER, footprint).rowCapacity;
  // 15,360 surface + 43,764 closed-wall strip + 48 coarse rows, 256-aligned.
  assert.equal(planned, 59_392);
  // The authored reserve clears the planner's own request without inflating it:
  // a still lane needs settling headroom, not a second capacity policy.
  assert.ok(DEEP_POWER_HYDROSTATIC_PRESSURE_ROW_CAPACITY >= planned);
  assert.ok(DEEP_POWER_HYDROSTATIC_PRESSURE_ROW_CAPACITY <= 1.25 * planned);
  // And it is far below one row per liquid cell, which is the whole claim: if
  // the deep interior did not coarsen, this arena could not hold the solve.
  assert.ok(DEEP_POWER_HYDROSTATIC_PRESSURE_ROW_CAPACITY < 0.5 * LIQUID_CELLS);

  const rings = planFineLevelSetCapacityDilationBrickRings(4, BAND_CELLS, FINE_FACTOR);
  const single = planGlobalFineNarrowBandBrickCapacity([...GRID], rings);
  const footprintShaped = planFluidFootprintFineNarrowBandBrickCapacity([...GRID],
    [64, WET_LAYERS, 64], rings);
  // The tank's fluid footprint *is* the tank, so the footprint-shaped default
  // reserves all three faces of a box that has only one free surface. Authoring
  // the single-interface plan removes reserve rather than adding any.
  assert.equal(DEEP_POWER_HYDROSTATIC_FINE_BRICK_CAPACITY, single.maximumResidentBricks);
  assert.equal(DEEP_POWER_HYDROSTATIC_FINE_BRICK_CAPACITY, 67_584);
  assert.ok(footprintShaped.maximumResidentBricks
    >= 2 * DEEP_POWER_HYDROSTATIC_FINE_BRICK_CAPACITY);
  assert.ok(DEEP_POWER_HYDROSTATIC_FINE_BRICK_CAPACITY < single.logicalBrickCount);

  // Air support has no scene dial at this size: above domain/5 rows the
  // corridor arena is the dense domain-sized one either way, and it already
  // covers the 64 x 4 x 64 air reach above the free surface.
  const support = octreeAirSupportFootprintCapacity(
    DEEP_POWER_HYDROSTATIC_PRESSURE_ROW_CAPACITY, DOMAIN_CELLS);
  assert.equal(support, DOMAIN_CELLS);
  const arena = planOctreeAirVelocitySupportGPU(DEEP_POWER_HYDROSTATIC_PRESSURE_ROW_CAPACITY,
    DEEP_POWER_HYDROSTATIC_PRESSURE_ROW_CAPACITY * 30, [...GRID], 256, support);
  const airReachCells = DIMENSIONS.nx * reaches().surface * DIMENSIONS.nz;
  assert.equal(airReachCells, 16_384);
  assert.ok(arena.identityCapacity
    >= DEEP_POWER_HYDROSTATIC_PRESSURE_ROW_CAPACITY + airReachCells);
});

test("the deep lane shares surface settings while using its compatible hierarchy root", () => {
  const preset = getScenePreset("deep-power-hydrostatic");
  assert.equal(preset.id, "deep-power-hydrostatic");
  assert.equal(preset.create().sceneId, "deep-power-hydrostatic");
  assert.equal(preset.methodProfile, DEEP_POWER_HYDROSTATIC_METHOD_PROFILE);
  const shallow = LARGE_POWER_HYDROSTATIC_METHOD_PROFILE.overrides;
  const deep = DEEP_POWER_HYDROSTATIC_METHOD_PROFILE.overrides;
  assert.equal(DEEP_POWER_HYDROSTATIC_METHOD_PROFILE.methodId,
    LARGE_POWER_HYDROSTATIC_METHOD_PROFILE.methodId);
  assert.equal(DEEP_POWER_HYDROSTATIC_METHOD_PROFILE.quality,
    LARGE_POWER_HYDROSTATIC_METHOD_PROFILE.quality);
  assert.equal(shallow.maximumLeafSize, "4");
  assert.equal(deep.maximumLeafSize, "16");
  assert.equal(deep.interfaceRefinementBandCells, shallow.interfaceRefinementBandCells);
  assert.equal(deep.globalFineLevelSetFactor, shallow.globalFineLevelSetFactor);
  // The two footprint-derived reserves and the domain-compatible root differ.
  assert.deepEqual(
    Object.keys(deep).filter((key) => deep[key] !== shallow[key]).sort(),
    ["globalFineLevelSetMaximumBricks", "maximumLeafSize", "pressureRowCapacity"]);
});

test("both deep smoke lanes carry the interior-coarsening claim as a named gate", () => {
  const expectedRowGate = 61_440;
  for (const laneId of ["default", "runtime-240"] as const) {
    const lane = getSceneWebGPUSmokeLane("deep-power-hydrostatic", laneId);
    assert.equal(lane.methods[0]?.overrides.pressureRowCapacity,
      DEEP_POWER_HYDROSTATIC_PRESSURE_ROW_CAPACITY);
    assert.equal(lane.methods[0]?.overrides.globalFineLevelSetMaximumBricks,
      DEEP_POWER_HYDROSTATIC_FINE_BRICK_CAPACITY);
    assert.equal(lane.methods[0]?.overrides.interfaceRefinementBandCells, BAND_CELLS);
    assert.equal(lane.methods[0]?.overrides.globalFineLevelSetFactor, "4");
    assert.ok(lane.acceptance.some(({ id, expected }) =>
      id === "expected-grid" && Array.isArray(expected)
      && expected.every((value, axis) => value === GRID[axis])));
    const coarsening = lane.acceptance.find(({ id }) => id === "deep-hydrostatic-interior-coarsening");
    assert.ok(coarsening, `${laneId} must gate on published rows`);
    assert.equal(coarsening.metric, "methods.octree.info.pressureRequiredRows");
    assert.equal(coarsening.expected, expectedRowGate);
    assert.ok(lane.acceptance.some(({ id, expected }) =>
      id === "deep-hydrostatic-row-arena-fits" && expected === false));
    // 0.375 rows per liquid cell, against the shallow lane's 1.004.
    assert.equal(expectedRowGate / LIQUID_CELLS, 0.375);
    assert.ok(expectedRowGate < DEEP_POWER_HYDROSTATIC_PRESSURE_ROW_CAPACITY);
  }
  const cold = getSceneWebGPUSmokeLane("deep-power-hydrostatic");
  assert.deepEqual(cold.stop, { simulatedTime_s: 0.004, exactSteps: 1, maxDt_s: 0.004 });
  const still = getSceneWebGPUSmokeLane("deep-power-hydrostatic", "runtime-240");
  assert.deepEqual(still.stop, { simulatedTime_s: 0.96, exactSteps: 240, maxDt_s: 0.004 });
  assert.ok(still.acceptance.some(({ id, expected }) =>
    id === "deep-hydrostatic-volume-drift" && expected === 1e-4));
});

test("the deep benchmark lane pins the scene, the still step count and the Bet-4 census", () => {
  const deep = POWER_DAM_LANE_ENVIRONMENT["deep-hydrostatic"];
  const shallow = POWER_DAM_LANE_ENVIRONMENT["large-hydrostatic"];
  assert.deepEqual(deep, {
    FLUID_SCENE: "deep-power-hydrostatic", FLUID_TARGET_S: "0.96",
    FLUID_MAX_DT: "0.004", FLUID_ORACLE_STEPS: "240", FLUID_EXPECT_EXACT_STEPS: "240",
    FLUID_EXPECT_GRID: "64,48,64", FLUID_MAXIMUM_LEAF_SIZE: "32",
    FLUID_OCTREE_INTERFACE_BAND: "1", FLUID_OCTREE_GLOBAL_FINE_FACTOR: "4",
    FLUID_POWER_HYBRID_CENSUS: "1",
    FLUID_PRESSURE_ROW_CAPACITY: String(DEEP_POWER_HYDROSTATIC_PRESSURE_ROW_CAPACITY),
  });
  // A benchmark lane runs its own environment, not the smoke catalog's authored
  // method profile: any lane whose scene authors a non-default row capacity must
  // restate it here or the benchmark silently runs a differently-sized solver.
  // `--lane=large` died at t=0 for exactly this reason.
  assert.equal(POWER_DAM_LANE_ENVIRONMENT.large.FLUID_PRESSURE_ROW_CAPACITY, "8192");
  assert.equal(shallow.FLUID_PRESSURE_ROW_CAPACITY, "4096");
  assert.equal(deep.FLUID_PRESSURE_ROW_CAPACITY,
    String(DEEP_POWER_HYDROSTATIC_PRESSURE_ROW_CAPACITY));
  // The deep lane's other reserve is a reduction below the planner default, so
  // it stays correct on the benchmark path without using the diagnostic-only
  // `FLUID_OCTREE_GLOBAL_FINE_MAXIMUM_BRICKS` A/B override.
  const rings = planFineLevelSetCapacityDilationBrickRings(4, BAND_CELLS, FINE_FACTOR);
  assert.ok(DEEP_POWER_HYDROSTATIC_FINE_BRICK_CAPACITY
    < planFluidFootprintFineNarrowBandBrickCapacity([...GRID], [64, WET_LAYERS, 64], rings)
      .maximumResidentBricks);

  // Same 240 still steps, timestep, band, fine factor and leaf size as the
  // lane it is measured against; only the scene, its grid and its reserve differ.
  for (const key of ["FLUID_TARGET_S", "FLUID_MAX_DT", "FLUID_ORACLE_STEPS",
    "FLUID_EXPECT_EXACT_STEPS", "FLUID_MAXIMUM_LEAF_SIZE", "FLUID_OCTREE_INTERFACE_BAND",
    "FLUID_OCTREE_GLOBAL_FINE_FACTOR"] as const) {
    assert.equal(deep[key], shallow[key], `deep-hydrostatic must match large-hydrostatic on ${key}`);
  }
  // No diagnostic load beyond the terminal census: this is a throughput lane,
  // so none of the ocean lane's per-step audits or its lowered binding tier may
  // leak in. The census itself is one readback after the measured window.
  for (const forbidden of ["FLUID_POWER_GENERATION_AUDIT", "FLUID_POWER_STAGE_AUDIT",
    "FLUID_STABILITY_ENVELOPE", "FLUID_WEBGPU_MAX_STORAGE_BINDING_BYTES"]) {
    assert.equal(Object.hasOwn(deep, forbidden), false,
      `a throughput lane must not carry ${forbidden}`);
  }

  const packageJson = JSON.parse(readFileSync(new URL("../package.json", import.meta.url), "utf8")) as {
    scripts: Record<string, string>;
  };
  assert.equal(packageJson.scripts["benchmark:power-dam-deep-hydrostatic"],
    "node --import tsx tools/benchmark-power-dam.ts --lane=deep-hydrostatic");
  for (const [script, lane] of [
    ["test:webgpu:deep-power-hydrostatic", "default"],
    ["test:webgpu:deep-power-hydrostatic-240", "runtime-240"],
  ] as const) {
    const command = packageJson.scripts[script];
    assert.match(command, /FLUID_SCENE=deep-power-hydrostatic/);
    assert.match(command, new RegExp(`FLUID_LANE=${lane}`));
    assert.match(command, /FLUID_EXPECT_GRID=64,48,64/);
    assert.match(command, /FLUID_POWER_HYBRID_CENSUS=1/);
    // The isolated runner refuses anything above its 240 s safety envelope.
    assert.match(command, /FLUID_WEBGPU_SMOKE_TIMEOUT_MS=240000/);
    assert.match(command, /run-webgpu-smoke-isolated\.ts$/);
  }
});
