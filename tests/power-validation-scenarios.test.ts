import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { damBreakFractions } from "../lib/initial-fluid";
import { validateScene } from "../lib/model";
import {
  LARGE_HYDROSTATIC_POWER_METHOD_PROFILE,
  POWER_VALIDATION_METHOD_PROFILE,
  createLargeHydrostaticScene,
  createMinimalPowerDamBreakScene,
  getScenePreset,
} from "../lib/scenes";
import { createTallCellLayout, type GPUQuality } from "../lib/tall-cell-grid";
import { createSmokeScenario, isSmokeScenarioId } from "../tools/webgpu-smoke-scenarios";

test("larger hydrostatic oracle is a 32x24x16 body-free tank with a quarter-cell surface offset", () => {
  const scene = createLargeHydrostaticScene();
  assert.deepEqual(validateScene(scene), []);
  assert.deepEqual(scene.container, {
    width_m: 1.6,
    height_m: 1.2,
    depth_m: 0.8,
    fillFraction: 61 / 96,
    top: "open",
    fluidWallMode: "free-slip",
  });
  assert.equal(scene.container.height_m * scene.container.fillFraction, 0.7625);
  assert.ok(Math.abs(scene.container.height_m * scene.container.fillFraction / 0.05 - 15.25) < 1e-12);
  assert.equal(scene.fluid.initialCondition, "tank-fill");
  assert.equal(scene.fluid.surfaceTension_N_m, 0);
  assert.equal(scene.fluid.inflow, undefined);
  assert.equal(scene.fluid.initialBrickSeeds_m, undefined);
  assert.deepEqual(scene.rigidBodies, []);
  assert.deepEqual(scene.voxelDomain, { finestCellSize_m: 0.05, brickSize_cells: 8 });
  assert.equal(scene.numerics.fixedDt_s, 0.004);
  assert.equal(scene.numerics.maxDt_s, 0.004);

  for (const quality of ["balanced", "high", "ultra"] as GPUQuality[]) {
    const layout = createTallCellLayout(scene, quality);
    assert.deepEqual([layout.nx, layout.fineNy, layout.nz], [32, 24, 16]);
  }
});

test("minimal power dam uses a two-level authoritative analytic initializer in a 16-cubed tank", () => {
  const scene = createMinimalPowerDamBreakScene();
  assert.deepEqual(validateScene(scene), []);
  assert.deepEqual(scene.container, {
    width_m: 0.8,
    height_m: 0.8,
    depth_m: 0.8,
    fillFraction: 23 / 64,
    top: "open",
    fluidWallMode: "free-slip",
  });
  assert.equal(scene.fluid.initialCondition, "dam-break");
  assert.equal(scene.fluid.surfaceTension_N_m, 0);
  assert.equal(scene.fluid.inflow, undefined);
  assert.equal(scene.fluid.initialBrickSeeds_m, undefined);
  assert.equal(scene.fluid.initialBrickSeedsAdditive, undefined);
  assert.deepEqual(scene.rigidBodies, []);
  assert.deepEqual(scene.voxelDomain, { finestCellSize_m: 0.05, brickSize_cells: 8 });
  assert.equal(scene.numerics.fixedDt_s, 0.004);
  assert.equal(scene.numerics.maxDt_s, 0.004);

  const dam = damBreakFractions(scene.container.fillFraction);
  assert.equal(dam.width, 5 / 8);
  assert.equal(dam.depth, 5 / 8);
  assert.equal(dam.width * 16, 10);
  assert.equal(dam.depth * 16, 10);
  assert.ok(Math.abs(dam.width * dam.height * dam.depth - scene.container.fillFraction) < 1e-12);
});

test("both power-validation scenes are shared by presets and the smoke registry", () => {
  for (const id of ["hydrostatic-power-large-offset", "minimal-power-dam-break"] as const) {
    const preset = getScenePreset(id);
    assert.equal(preset.id, id);
    assert.ok(isSmokeScenarioId(id));
    const smoke = createSmokeScenario(id);
    assert.deepEqual(validateScene(smoke.scene), []);
    assert.deepEqual(smoke.scene, preset.create(), `${id} Dawn scene must be the UI preset scene`);
    assert.equal(smoke.methodProfile, preset.methodProfile,
      `${id} Dawn method profile must be the UI preset profile`);
  }
  const hydro = createSmokeScenario("hydrostatic-power-large-offset");
  assert.equal(hydro.oracleSteps, 1);
  assert.equal(hydro.target_s, 0.004);
  const dam = createSmokeScenario("minimal-power-dam-break");
  assert.equal(dam.oracleSteps, 50);
  assert.equal(dam.target_s, 0.5);
});

test("power-validation UI presets carry the exact authoritative Dawn method profile", () => {
  assert.deepEqual(POWER_VALIDATION_METHOD_PROFILE, {
    methodId: "octree",
    quality: "balanced",
    overrides: {
      maximumLeafSize: "2",
      interfaceRefinementBandCells: 3,
      faceVelocityTransport: "on",
      globalFineLevelSetFactor: "4",
      powerDiagramProjection: "authoritative",
    },
  });
  for (const id of ["hydrostatic-power-two-level", "minimal-power-dam-break"] as const) {
    assert.equal(getScenePreset(id).methodProfile, POWER_VALIDATION_METHOD_PROFILE);
  }
  assert.deepEqual(LARGE_HYDROSTATIC_POWER_METHOD_PROFILE, {
    ...POWER_VALIDATION_METHOD_PROFILE,
    overrides: { ...POWER_VALIDATION_METHOD_PROFILE.overrides, interfaceRefinementBandCells: 4 },
  });
  assert.equal(getScenePreset("hydrostatic-power-large-offset").methodProfile,
    LARGE_HYDROSTATIC_POWER_METHOD_PROFILE);
});

test("isolated Dawn commands pin the authored adaptive power configurations", () => {
  const packageJson = JSON.parse(readFileSync(new URL("../package.json", import.meta.url), "utf8")) as {
    scripts: Record<string, string>;
  };
  const hydro = packageJson.scripts["test:webgpu:hydrostatic-power-large-offset"];
  assert.match(hydro, /FLUID_SCENE=hydrostatic-power-large-offset/);
  assert.match(hydro, /FLUID_EXPECT_GRID=32,24,16/);
  assert.match(hydro, /FLUID_EXPECT_EXACT_STEPS=1/);
  assert.match(hydro, /FLUID_MAXIMUM_LEAF_SIZE=2/);
  assert.match(hydro, /FLUID_OCTREE_INTERFACE_BAND=4/);

  const dam = packageJson.scripts["test:webgpu:minimal-power-dam-break"];
  assert.match(dam, /FLUID_SCENE=minimal-power-dam-break/);
  assert.match(dam, /FLUID_TARGET_S=2(?:\s|$)/);
  assert.match(dam, /FLUID_ORACLE_STEPS=500/);
  assert.match(dam, /FLUID_EXPECT_EXACT_STEPS=500/);
  assert.match(dam, /FLUID_CHECKPOINT_EVERY_S=0\.1/);
  assert.match(dam, /FLUID_EXPECT_GRID=16,16,16/);
  assert.match(dam, /FLUID_MAXIMUM_LEAF_SIZE=2/);
  assert.match(dam, /FLUID_OCTREE_INTERFACE_BAND=3/);
  assert.match(dam, /FLUID_POWER_AUDIT_EVERY_STEPS=1/);
  assert.match(dam, /FLUID_WEBGPU_SMOKE_TIMEOUT_MS=240000/);
  assert.doesNotMatch(dam, /FLUID_DISABLE_TIMESTAMPS=1/,
    "the exhaustive regression must use the UI's timestamp-enabled submission path");

  const motion = packageJson.scripts["test:webgpu:minimal-power-dam-break-motion"];
  assert.match(motion, /FLUID_TARGET_S=0\.5/);
  assert.match(motion, /FLUID_CHECKPOINT_EVERY_S=0\.016/);
  assert.match(motion, /FLUID_MIN_PEAK_SPEED_M_S=0\.1/);
  assert.match(motion, /FLUID_MIN_DAM_SPREAD_M=0\.05/);
  assert.match(motion, /FLUID_EXPECT_GRID=16,16,16/);
  assert.match(motion, /FLUID_RASTER_CHECKPOINTS=1/);
  assert.match(motion, /FLUID_GLOBAL_FINE_GENERATION_TRANSITION=1/);
  assert.match(motion, /FLUID_WEBGPU_SMOKE_TIMEOUT_MS=240000/);
  assert.doesNotMatch(motion, /FLUID_(?:QUALITY|MAX_DT|VOXEL_CELL_SIZE|MAXIMUM_LEAF_SIZE|OCTREE_INTERFACE_BAND|OCTREE_GLOBAL_FINE_FACTOR)=/,
    "UI-parity smoke must consume the preset profile instead of duplicating solver settings");
  assert.doesNotMatch(motion, /FLUID_(?:STABILITY_ENVELOPE|DISABLE_TIMESTAMPS|POWER_GENERATION_AUDIT|POWER_STAGE_AUDIT)=/,
    "UI-parity smoke must not add a per-step audit/readback cadence absent from the UI");

  for (const command of [hydro, dam]) {
    assert.match(command, /FLUID_STABILITY_ENVELOPE=1/);
    assert.match(command, /FLUID_RASTER_CHECKPOINTS=1/);
    assert.match(command, /FLUID_GLOBAL_FINE_GENERATION_TRANSITION=1/);
    assert.match(command, /FLUID_OCTREE_GLOBAL_FINE_FACTOR=4/);
    assert.match(command, /FLUID_POWER_GENERATION_AUDIT=1/);
    assert.match(command, /run-webgpu-smoke-isolated\.ts$/);
  }
});

test("moving dam Dawn regression crosses the rejected generation and checks open-surface peeling", () => {
  const packageJson = JSON.parse(readFileSync(new URL("../package.json", import.meta.url), "utf8")) as {
    scripts: Record<string, string>;
  };
  const command = packageJson.scripts["test:webgpu:minimal-power-dam-break-motion"];
  assert.match(command, /FLUID_TARGET_S=0\.5/,
    "the regression must pass the former t=0.412 s failure instead of stopping at 0.4 s");
  assert.match(command, /FLUID_CHECKPOINT_EVERY_S=0\.016/,
    "the rejected generation must be bracketed by fenced raster/publication checkpoints");
  const smoke = readFileSync(new URL("../tools/run-webgpu-smoke.ts", import.meta.url), "utf8");
  assert.match(smoke, /powerTransitionWitness = await readPowerTransitionWitness/,
    "the long dynamic run retains historical topology only for failure localization");
  assert.match(smoke,
    /expectedLeafSizes\.every\(\(size\) => \(sizes\[size\] \?\? 0\) > 0\)/,
    "the regression must require both leaf scales in the live final topology");
  assert.doesNotMatch(smoke,
    /dynamicTransitionWitness\?\.(?:leafSizeHistogram|transitionFaceCount|obliqueTransitionFaceCount|maximumTransitionSizeRatio|transitionRowCount|transitionTetrahedronCount)\s*\?\?/,
    "a historical transition must never replace final topology diagnostics");
  assert.match(smoke, /maximumProjectedVariationalResidual[\s\S]*?<= 1e-6/,
    "moving free-surface incompressibility must be gated by the Eq. (3)-form operator residual");
  assert.doesNotMatch(smoke, /minimal dam per-step power-cell divergence/,
    "the unweighted full-cell divergence reduction is not the paper's cut-cell variational residual");
  assert.match(smoke, /observed\?\.backOnlyInterfacePixels === 0/);
  assert.match(smoke, /reverse\?\.backOnlyInterfacePixels === 0/);
  assert.match(smoke,
    /observed\?\.damExposedCornerCapPixels\?\.\[0\][\s\S]*observed\?\.damExposedCornerCapPixels\?\.\[1\]/,
    "the t=0 Dawn raster must require both exposed dam faces at their shared +x/+z corner");
  assert.match(smoke, /damBreakFractions\(scene\.container\.fillFraction\)/,
    "the raster oracle must inspect the authored dam corner rather than only the tank-wall corner");
  assert.doesNotMatch(smoke, /minimumPairedFraction/,
    "an uncapped liquid\/air surface must not be judged as a closed wall-capped solid");
});
