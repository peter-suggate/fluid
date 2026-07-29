import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { STRUCTURED_GENERATION_AUDIT_SNAPSHOT } from "../lib/structured-authority-audit";
import { getSceneWebGPUSmokeLane } from "../lib/scene-webgpu-smoke-catalog";
import { damBreakFractions } from "../lib/initial-fluid";
import { getMethod, resolveMethodValues } from "../lib/methods";
import { validateScene } from "../lib/model";
import { OCTREE_SECTION43_MINI_SHELL_DEPTH, OCTREE_SECTION43_PRODUCTION_SHELL_DEPTH,
  planOctreeSolveTail } from "../lib/octree-solve-tail-policy";
import {
  CEILING_DROP_METHOD_PROFILE,
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
    top: "closed",
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
    top: "closed",
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
    assert.deepEqual(smoke.methodProfile, preset.methodProfile,
      `${id} Dawn method profile must be the UI preset profile`);
  }
  const hydro = createSmokeScenario("hydrostatic-power-large-offset");
  assert.equal(hydro.oracleSteps, 1);
  assert.equal(hydro.target_s, 0.004);
  const dam = createSmokeScenario("minimal-power-dam-break");
  assert.equal(dam.oracleSteps, 500);
  assert.equal(dam.target_s, 2);
});

test("power-validation UI presets carry the exact authoritative Dawn method profile", () => {
  assert.deepEqual(POWER_VALIDATION_METHOD_PROFILE, {
    methodId: "octree",
    quality: "balanced",
    overrides: {
      maximumLeafSize: "2",
      interfaceRefinementBandCells: 3,
      globalFineLevelSetFactor: "4",
    },
  });
  for (const id of ["hydrostatic-power-two-level", "minimal-power-dam-break"] as const) {
    assert.equal(getScenePreset(id).methodProfile, POWER_VALIDATION_METHOD_PROFILE);
  }
  assert.deepEqual(LARGE_HYDROSTATIC_POWER_METHOD_PROFILE, {
    ...POWER_VALIDATION_METHOD_PROFILE,
    overrides: { ...POWER_VALIDATION_METHOD_PROFILE.overrides,
      interfaceRefinementBandCells: 4 },
  });
  assert.equal(getScenePreset("hydrostatic-power-large-offset").methodProfile,
    LARGE_HYDROSTATIC_POWER_METHOD_PROFILE);
  assert.deepEqual(CEILING_DROP_METHOD_PROFILE, {
    ...POWER_VALIDATION_METHOD_PROFILE,
    overrides: { ...POWER_VALIDATION_METHOD_PROFILE.overrides,
      interfaceRefinementBandCells: 1 },
  });
  assert.equal(getScenePreset("ceiling-slab-drop").methodProfile,
    CEILING_DROP_METHOD_PROFILE);
  for (const id of ["corner-brick-drop", "midair-brick-drop", "midair-corner-drop"] as const) {
    assert.equal(getScenePreset(id).methodProfile, POWER_VALIDATION_METHOD_PROFILE,
      `${id} must retain the shared band-3 validation profile`);
  }
});

test("ceiling UI resolves band 1 and k=4 while larger authored profiles retain k=8", () => {
  const planPreset = (id: "ceiling-slab-drop" | "hydrostatic-power-large-offset" | "ocean-seiche") => {
    const preset = getScenePreset(id);
    const scene = preset.create();
    const profile = preset.methodProfile;
    const method = getMethod(profile?.methodId ?? "octree");
    const quality = profile?.quality ?? "balanced";
    const values = resolveMethodValues(method, quality, profile?.overrides ?? {});
    const layout = createTallCellLayout(scene, quality);
    return {
      values,
      policy: planOctreeSolveTail({
        finestDimensions: [layout.nx, layout.fineNy, layout.nz],
        maximumLeafSize: Number(values.maximumLeafSize) as 2 | 4 | 8 | 16 | 32,
        initialCondition: scene.fluid.initialCondition,
        hasInflow: scene.fluid.inflow !== undefined,
        hasTerrain: false,
        movingRigidBodyCount: scene.rigidBodies.filter(({ motion }) => motion !== "static").length,
        closedTop: scene.container.top === "closed",
        requestedRelativeTolerance: scene.numerics.pressureRelativeTolerance,
      }),
      dimensions: [layout.nx, layout.fineNy, layout.nz],
    };
  };

  const ceiling = planPreset("ceiling-slab-drop");
  assert.deepEqual(ceiling.dimensions, [24, 16, 24]);
  assert.equal(ceiling.values.interfaceRefinementBandCells, 1);
  assert.equal(ceiling.values.globalFineLevelSetFactor, "4");
  assert.equal(ceiling.values.maximumLeafSize, "2");
  assert.equal(ceiling.policy.boundarySmoothingIterations,
    OCTREE_SECTION43_MINI_SHELL_DEPTH);

  for (const id of ["hydrostatic-power-large-offset", "ocean-seiche"] as const) {
    assert.equal(planPreset(id).policy.boundarySmoothingIterations,
      OCTREE_SECTION43_PRODUCTION_SHELL_DEPTH, `${id} must retain k=8`);
  }
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

  const motion = packageJson.scripts["test:webgpu:minimal-power-dam-break-motion"];
  assert.match(motion, /FLUID_TARGET_S=0\.5/);
  assert.match(motion, /FLUID_CHECKPOINT_EVERY_S=0\.016/);
  assert.doesNotMatch(motion, /FLUID_MIN_PEAK_SPEED_M_S|FLUID_MIN_DAM_SPREAD_M/,
    "motion thresholds belong to the scene hook parameters, not the command environment");
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

test("moving dam Dawn regression crosses the former rejection and checks structured authority plus open-surface peeling", () => {
  const packageJson = JSON.parse(readFileSync(new URL("../package.json", import.meta.url), "utf8")) as {
    scripts: Record<string, string>;
  };
  const command = packageJson.scripts["test:webgpu:minimal-power-dam-break-motion"];
  assert.match(command, /FLUID_TARGET_S=0\.5/,
    "the regression must pass the former t=0.412 s failure instead of stopping at 0.4 s");
  assert.match(command, /FLUID_CHECKPOINT_EVERY_S=0\.016/,
    "the rejected generation must be bracketed by fenced raster/publication checkpoints");
  const motion = getSceneWebGPUSmokeLane("minimal-power-dam-break", "motion");
  assert.equal(motion.stop.simulatedTime_s, 0.5);
  assert.equal(motion.collect.checkpointEvery_s, 0.016);
  const motionParameters = motion.hooks.find((hook) => hook.id === "minimal-dam-motion")?.parameters;
  assert.deepEqual(motionParameters && {
    minimumPeakSpeed_m_s: motionParameters.minimumPeakSpeed_m_s,
    minimumLateralSpread_m: motionParameters.minimumLateralSpread_m,
  }, {
    minimumPeakSpeed_m_s: 0.1,
    minimumLateralSpread_m: 0.05,
  });

  const complete = getSceneWebGPUSmokeLane("minimal-power-dam-break");
  assert.equal(complete.collect.structuredValidation, true,
    "every audited moving generation must use the direct structured authority contract");
  assert.equal(complete.acceptance.find((rule) => rule.id === "minimal-power-variational-residual")?.expected, 3.5e-6,
    "moving free-surface incompressibility must be gated by the Eq. (3)-form operator residual");
  const rasterParameters = complete.hooks.find((hook) => hook.id === "water-raster-integrity")?.parameters;
  assert.deepEqual(rasterParameters && {
    initialDamCornerCaps: rasterParameters.initialDamCornerCaps,
    maximumBackOnlyPixels: rasterParameters.maximumBackOnlyPixels,
    preImpactHoleLimitPixels: rasterParameters.preImpactHoleLimitPixels,
    terraceEdgeFractionMaximum: rasterParameters.terraceEdgeFractionMaximum,
    separatingCeilingContact: rasterParameters.separatingCeilingContact,
    ceilingWetCellLimits: rasterParameters.ceilingWetCellLimits,
    ceilingContactPixelLimits: rasterParameters.ceilingContactPixelLimits,
  }, {
    initialDamCornerCaps: true,
    maximumBackOnlyPixels: 2,
    preImpactHoleLimitPixels: 2,
    terraceEdgeFractionMaximum: 0.12,
    separatingCeilingContact: true,
    ceilingWetCellLimits: [
      { before_s: 1.6, maximum: 9 },
      { before_s: 1.7, maximum: 5 },
      { before_s: 2, maximum: 1 },
      { maximum: 3 },
    ],
    ceilingContactPixelLimits: [
      { before_s: 1.5, maximum: 30 },
      { before_s: 1.6, maximum: 18 },
      { maximum: 0 },
    ],
  });
  const completeMotionParameters = complete.hooks.find((hook) => hook.id === "minimal-dam-motion")?.parameters;
  assert.deepEqual(completeMotionParameters && {
    minimumPeakSpeed_m_s: completeMotionParameters.minimumPeakSpeed_m_s,
    minimumLateralSpread_m: completeMotionParameters.minimumLateralSpread_m,
    maximumMechanicalEnergyRetention: completeMotionParameters.maximumMechanicalEnergyRetention,
    maximumRitterCelerityRatio: completeMotionParameters.maximumRitterCelerityRatio,
  }, {
    minimumPeakSpeed_m_s: 0.1,
    minimumLateralSpread_m: 0.05,
    maximumMechanicalEnergyRetention: 1.5,
    maximumRitterCelerityRatio: 1.35,
  });
});

test("per-generation Dawn audit records one coherent structured epoch", () => {
  const smoke = readFileSync(new URL("../tools/webgpu-smoke-executor.ts", import.meta.url), "utf8");
  // The per-buffer copies moved into one shared ABI writer that the browser's
  // snapshot ring also calls, so the harness and the UI record byte-identical
  // bytes. The exact widths are still asserted -- on the writer, where they
  // are now defined once instead of at each call site.
  const audit = readFileSync(
    new URL("../lib/structured-authority-audit.ts", import.meta.url), "utf8");
  assert.match(smoke, /encodeStructuredAuditRecordCopies\(auditEncoder, \{[\s\S]*?structuredVelocityControl:[\s\S]*?structuredBoundaryControl:/,
    "each audited step must enqueue the structured velocity and boundary controls");
  assert.match(audit,
    /copy\("structured", sources\.structuredVelocityControl,\s*layout\.structuredOffsetBytes, layout\.structuredBytes\)/,
    "the writer must copy the exact six-word structured velocity control");
  assert.match(audit,
    /copy\("boundary", sources\.structuredBoundaryControl,\s*layout\.boundaryOffsetBytes, layout\.boundaryBytes\)/,
    "the writer must copy the exact seven-word structured boundary control");
  assert.equal(STRUCTURED_GENERATION_AUDIT_SNAPSHOT.structuredBytes, 24);
  assert.equal(STRUCTURED_GENERATION_AUDIT_SNAPSHOT.boundaryBytes, 28);
  assert.match(smoke,
    /expectedStructuredEpoch: snapshot\.structured\.epoch,[\s\S]*structured: snapshot\.structured,[\s\S]*boundary: snapshot\.boundary/,
    "admission must validate velocity, boundary, and fine publication as one coherent epoch");
  assert.match(smoke, /previousAuditedPowerGeneration = snapshot\.structured\.epoch/,
    "the next audit must advance from the accepted structured epoch");
});
