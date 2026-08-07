import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  DEFAULT_SVO_RENDER_TUNING,
  SVO_CONE_RADIANCE_RECONSTRUCTION_CODES,
  SVO_ENVIRONMENT_BRICK_REFINEMENT_MAXIMUM,
  SVO_ENVIRONMENT_REFINEMENT_DEPTH_DEFAULT,
  SVO_ENVIRONMENT_REFINEMENT_DEPTH_MAXIMUM,
  SVO_LOD_FIXED_LEVEL_MAXIMUM,
  SVO_LOD_SCREEN_SPACE_PIXELS_MAXIMUM,
  SVO_PRIMARY_LEAF_VISIT_HARD_LIMIT,
  SVO_RENDER_QUALITY_PRESETS,
  SVO_RENDER_TUNING_PRESETS,
  normalizeSvoRenderTuning,
  svoSceneryDetailCellSize_m,
  svoSceneryRefinementDepth,
  type SvoRenderQualityPreset,
  type SvoRenderTuningPreset,
} from "../lib/svo-render-tuning";
import { SVO_SCREEN_SPACE_TERMINATION_CONTRACT } from "../lib/svo-screen-space-termination";
import { svoDrySceneShader } from "../lib/webgpu-svo-dry-scene";

test("primary leaf tuning reaches the audited shader ceiling without recompilation", () => {
  assert.ok(SVO_RENDER_TUNING_PRESETS.performance.primaryLeafVisits
    < DEFAULT_SVO_RENDER_TUNING.primaryLeafVisits);
  assert.ok(DEFAULT_SVO_RENDER_TUNING.primaryLeafVisits
    < SVO_RENDER_TUNING_PRESETS.quality.primaryLeafVisits);
  assert.ok(SVO_RENDER_TUNING_PRESETS.quality.primaryLeafVisits
    <= SVO_PRIMARY_LEAF_VISIT_HARD_LIMIT);
  assert.equal(normalizeSvoRenderTuning({
    ...DEFAULT_SVO_RENDER_TUNING,
    primaryLeafVisits: SVO_PRIMARY_LEAF_VISIT_HARD_LIMIT + 1,
  }).primaryLeafVisits, SVO_PRIMARY_LEAF_VISIT_HARD_LIMIT);
  assert.match(svoDrySceneShader,
    new RegExp(`leafVisit<${SVO_PRIMARY_LEAF_VISIT_HARD_LIMIT}u&&leafVisit<leafBudget`));
});

test("the quality ladder is a ladder of pairs, and its rungs agree with the tuning projection", () => {
  // Every rung names both halves, because they are not independent: under
  // `cones` the visibility budgets are barely reached, and under `exact` they
  // are the whole cost of a shadow.
  const rungs = Object.keys(SVO_RENDER_QUALITY_PRESETS) as SvoRenderQualityPreset[];
  assert.deepEqual(rungs, ["performance", "balanced", "quality", "reference"]);
  for (const rung of rungs) {
    assert.equal(SVO_RENDER_QUALITY_PRESETS[rung].tuning, SVO_RENDER_TUNING_PRESETS[rung],
      `${rung} must project to the same tuning object, not a second copy that can drift`);
  }
  // `reference` is `quality`'s work with the cone tier switched off. Same
  // sliders, so the difference in the frame is attributable to one thing.
  assert.equal(SVO_RENDER_QUALITY_PRESETS.reference.tuning, SVO_RENDER_QUALITY_PRESETS.quality.tuning);
  assert.equal(SVO_RENDER_QUALITY_PRESETS.reference.coneTracingMode, "exact");
  for (const rung of ["performance", "balanced", "quality"] as const) {
    assert.equal(SVO_RENDER_QUALITY_PRESETS[rung].coneTracingMode, "cones", rung);
  }
  // Exact visibility is what spends these, so the top rung must not ship the
  // budgets that were tuned for a tier that never reached them.
  assert.ok(SVO_RENDER_QUALITY_PRESETS.reference.tuning.visibilityLeafVisits
    > SVO_RENDER_QUALITY_PRESETS.performance.tuning.visibilityLeafVisits);
});

test("level of detail defaults to the shipping screen-space policy in every preset", () => {
  // Zero, which the contract defines as exact: no node is ever collapsed, so
  // the default frame is the reference image rather than an approximation of it.
  //
  // The machinery is still compiled and the slider still live — this is the
  // runtime threshold, not the compile flag — so raising it restores the saving.
  // What it stopped buying is quality: a sub-threshold brick draws as one voxel,
  // and once the primary began shading a smooth reconstructed normal that tier
  // was the only part of the frame still terracing.
  assert.equal(DEFAULT_SVO_RENDER_TUNING.lodScreenSpacePixels, 0);
  assert.ok(SVO_SCREEN_SPACE_TERMINATION_CONTRACT.defaultThresholdPixels > 0,
    "the contract keeps its authored threshold so raising the slider has a derived value to return to");
  // `fixed-level` is a debugging view. A preset reaching it would mean shipping
  // a frame whose detail is pinned to a level nobody chose for that scene.
  for (const preset of Object.keys(SVO_RENDER_TUNING_PRESETS) as SvoRenderTuningPreset[]) {
    assert.equal(SVO_RENDER_TUNING_PRESETS[preset].lodMode, "screen-space", preset);
  }
  assert.equal(normalizeSvoRenderTuning({
    ...DEFAULT_SVO_RENDER_TUNING,
    lodMode: "screen-space-ish" as never,
  }).lodMode, "screen-space");
});

test("a zero screen-space threshold survives normalization as exact traversal", () => {
  // The acceptance lanes ask for zero to get the unapproximated reference
  // image. Any clamp that nudged it to a small positive value would give them
  // a subtly different frame under the name of the exact one, so this is the
  // one bound in the whole tuning table that must not be inclusive-above-zero.
  assert.equal(normalizeSvoRenderTuning({
    ...DEFAULT_SVO_RENDER_TUNING,
    lodScreenSpacePixels: 0,
  }).lodScreenSpacePixels, 0);
  assert.equal(normalizeSvoRenderTuning({
    ...DEFAULT_SVO_RENDER_TUNING,
    lodScreenSpacePixels: SVO_LOD_SCREEN_SPACE_PIXELS_MAXIMUM + 10,
  }).lodScreenSpacePixels, SVO_LOD_SCREEN_SPACE_PIXELS_MAXIMUM);
  // Continuous, not integral: this is the control that sweeps the cost curve.
  assert.equal(normalizeSvoRenderTuning({
    ...DEFAULT_SVO_RENDER_TUNING,
    lodScreenSpacePixels: 2.5,
  }).lodScreenSpacePixels, 2.5);
});

test("the fixed level clamps to the addressable hierarchy and starts at its floor of effect", () => {
  assert.equal(SVO_LOD_FIXED_LEVEL_MAXIMUM, 21);
  // Defaulting to the deepest level keeps a mode switch from changing the image
  // by itself — otherwise the debugging tool becomes the thing under suspicion.
  assert.equal(DEFAULT_SVO_RENDER_TUNING.lodFixedLevel, SVO_LOD_FIXED_LEVEL_MAXIMUM);
  assert.equal(normalizeSvoRenderTuning({
    ...DEFAULT_SVO_RENDER_TUNING,
    lodFixedLevel: SVO_LOD_FIXED_LEVEL_MAXIMUM + 4,
  }).lodFixedLevel, SVO_LOD_FIXED_LEVEL_MAXIMUM);
  assert.equal(normalizeSvoRenderTuning({
    ...DEFAULT_SVO_RENDER_TUNING,
    lodFixedLevel: -3,
  }).lodFixedLevel, 0);
});

test("the render panel exposes the level-of-detail mode and both of its predicates", () => {
  const panel = readFileSync(new URL("../components/VisualPanel.tsx", import.meta.url), "utf8");
  assert.match(panel, /Level of detail/);
  assert.match(panel, /label="LOD threshold"/);
  // "Fixed LOD level" said where descent stopped and nothing about what the
  // number meant. Levels at or below the scene's brick depth all render the
  // same image — the raster primary instances one proxy per brick — so the
  // coarse half of the slider is a plateau, and three levels past the brick
  // depth is already one cell. A user must not have to find either by
  // experiment, and a hover tooltip is not where they would look, so both
  // facts are asserted as visible caption text rather than as a `hint`.
  assert.match(panel, /label="Detail level"/);
  assert.doesNotMatch(panel, /Fixed LOD level/);
  const caption = panel.match(/<p className="control-caption">[^]*?<\/p>/)?.[0] ?? "";
  assert.match(caption, /0 coarsest → 21 exact voxels/);
  assert.match(caption, /plateau/);
  assert.match(caption, /no aggregate is ever drawn larger than this/i,
    "the screen-space predicate states the guarantee it makes, not just its units");
  // The pre-existing depth slider is a work budget that reports exhaustion, not
  // a coarser surface. Two "levels" sliders that mean different things is a
  // support burden; they are now separated — one leads Detail, the other sits
  // under Advanced — and the one that is a budget still says so.
  assert.match(panel, /A budget, not a LOD control/);
  const lead = panel.indexOf('className="control-lead"');
  const depth = panel.indexOf('label="Maximum traversal depth"');
  assert.ok(lead > 0 && depth > lead, "the LOD predicate leads; the work budget is elsewhere");
});

test("environment bricks default one refinement level deeper than the previous plan", () => {
  assert.equal(SVO_ENVIRONMENT_BRICK_REFINEMENT_MAXIMUM, 1);
  assert.equal(DEFAULT_SVO_RENDER_TUNING.environmentBrickRefinementLevels, 1);
  assert.equal(normalizeSvoRenderTuning({
    ...DEFAULT_SVO_RENDER_TUNING,
    environmentBrickRefinementLevels: 0,
  }).environmentBrickRefinementLevels, 0);
  assert.equal(normalizeSvoRenderTuning({
    ...DEFAULT_SVO_RENDER_TUNING,
    environmentBrickRefinementLevels: 2,
  }).environmentBrickRefinementLevels, 1);
  const panel = readFileSync(new URL("../components/VisualPanel.tsx", import.meta.url), "utf8");
  assert.match(panel, /Environment brick refinement/);
});

test("cone prepass tuning preserves every supported spatial rate", () => {
  assert.equal(DEFAULT_SVO_RENDER_TUNING.coneLightingScale, 0.5,
    "balanced production uses the accepted 2x2 visibility tier");
  assert.equal(SVO_RENDER_TUNING_PRESETS.performance.coneLightingScale, 0.25,
    "the 4x4 relight tier remains an explicit performance choice");
  assert.equal(SVO_RENDER_TUNING_PRESETS.quality.coneLightingScale, 0.5,
    "quality retains the accepted 2x2 visibility error bar");
  for (const coneLightingScale of [1, 0.5, 0.25, 0.125] as const) {
    assert.equal(normalizeSvoRenderTuning({
      ...DEFAULT_SVO_RENDER_TUNING,
      coneLightingScale,
    }).coneLightingScale, coneLightingScale);
  }
  const panel = readFileSync(new URL("../components/VisualPanel.tsx", import.meta.url), "utf8");
  for (const label of ["FULL", "2×2", "4×4", "8×8"]) assert.ok(panel.includes(`label: "${label}"`));
  const renderer = readFileSync(new URL("../lib/webgpu-svo-dry-scene.ts", import.meta.url), "utf8");
  assert.match(renderer, /conePipelineBundles = new Map<SvoConeLightingScale/,
    "scale-specific pipelines remain resident instead of replacing one global variant");
  assert.match(renderer, /Promise\.all\(\(\[0\.25, 0\.5\] as const\)/,
    "the production moving and settled scales are both prewarmed");
});

test("stationary primary visibility reuse is explicit and defaults off", () => {
  assert.equal(DEFAULT_SVO_RENDER_TUNING.stationaryPrimaryReuseEnabled, false);
  assert.equal(SVO_RENDER_TUNING_PRESETS.performance.stationaryPrimaryReuseEnabled, false);
  assert.equal(SVO_RENDER_TUNING_PRESETS.balanced.stationaryPrimaryReuseEnabled, false);
  assert.equal(SVO_RENDER_TUNING_PRESETS.quality.stationaryPrimaryReuseEnabled, false);
  assert.equal(normalizeSvoRenderTuning({
    ...DEFAULT_SVO_RENDER_TUNING,
    stationaryPrimaryReuseEnabled: true,
  }).stationaryPrimaryReuseEnabled, true);
});

test("the render panel is four groups deep and only the two tuning groups open", () => {
  const panel = readFileSync(new URL("../components/VisualPanel.tsx", import.meta.url), "utf8");
  // Nine peer sections, five of them open, is what this panel grew into once
  // "add another group" became the answer to "where does this control live?".
  // The count is asserted rather than described because re-nesting is cheap and
  // silent: a fifth group costs nothing to add and undoes the whole point.
  const groups = [...panel.matchAll(/<ControlGroup title="([^"]+)"([^>]*)>/g)]
    .map(([, title, attributes]) => ({ title, open: attributes.includes(" open") }));
  assert.deepEqual(groups.map(({ title }) => title), ["Detail", "Lighting", "Diagnostics", "Advanced"]);
  assert.deepEqual(groups.filter(({ open }) => open).map(({ title }) => title), ["Detail", "Lighting"]);
});

test("calibration knobs stay reachable under Advanced instead of ranking with the levers", () => {
  const panel = readFileSync(new URL("../components/VisualPanel.tsx", import.meta.url), "utf8");
  assert.match(panel, /label="Shadow cone aperture"[^>]*value=\{tuning\.shadowConeAperture\}/);
  assert.match(panel, /updateTuning\("shadowConeAperture", value\)/);
  // Demoted, never deleted: none of these has an environment override or a
  // benchmark flag, so removing the control would make the field editable only
  // by recompiling `DEFAULT_SVO_RENDER_TUNING`. What earns the demotion is that
  // no preset moves any of them — and the ones presets do move, they move
  // together, which is what the PROFILE strip already is.
  const advanced = panel.indexOf('<ControlGroup title="Advanced"');
  assert.ok(advanced > 0, "the Advanced group exists");
  for (const label of [
    "Shadow cone aperture", "Shadow strength", "Shadow origin bias",
    "AO strength", "AO radius", "AO cone aperture",
    "Direct key", "GI cone aperture", "GI cones",
    "Normal escape", "Emitter clearance", "Cone step budget", "Shaded lights",
    "Maximum traversal depth", "Maximum node visits", "Maximum leaf visits",
    "Visibility nodes", "Visibility leaves", "Visibility voxel work", "Intersections",
    "Environment brick refinement", "Lighting reconstruction",
  ]) {
    const at = panel.indexOf(`>${label}<`) >= 0 ? panel.indexOf(`>${label}<`) : panel.indexOf(`label="${label}"`);
    assert.ok(at > advanced, `${label} is surfaced under Advanced`);
  }
  // The levers stay above it. Each is measured: resolution is pixel-linear,
  // raster/traced is 29.0 against 49.6 ms at 1500x1500, stationary reuse is
  // 47.6 -> 20.7 ms, and the screen-space threshold is 24.1 -> 5.0 ms on the
  // 4 Mpx far arm. Nothing under Advanced has a number like that attached.
  for (const label of ["Render resolution", "LOD threshold", "GI bounce", "GI occlusion", "Diffuse environment"]) {
    const at = panel.indexOf(`label="${label}"`);
    assert.ok(at > 0 && at < advanced, `${label} stays a first-class control`);
  }
  for (const label of ["Primary visibility", "Level of detail", "Lighting visibility", "Cone prepass"]) {
    const at = panel.indexOf(`>${label}<`);
    assert.ok(at > 0 && at < advanced, `${label} stays a first-class control`);
  }
});

test("global illumination exposes image-shaping controls with cinematic balanced defaults", () => {
  assert.deepEqual({
    bounce: DEFAULT_SVO_RENDER_TUNING.giBounceStrength,
    occlusion: DEFAULT_SVO_RENDER_TUNING.giOcclusionStrength,
    environment: DEFAULT_SVO_RENDER_TUNING.giEnvironmentStrength,
    direct: DEFAULT_SVO_RENDER_TUNING.giDirectStrength,
    aperture: DEFAULT_SVO_RENDER_TUNING.giConeAperture,
    cones: DEFAULT_SVO_RENDER_TUNING.giConeCount,
  }, { bounce: 1.8, occlusion: 0.65, environment: 0.85, direct: 1, aperture: 1.05, cones: 4 });
  const normalized = normalizeSvoRenderTuning({
    ...DEFAULT_SVO_RENDER_TUNING,
    giBounceStrength: 99, giOcclusionStrength: -1, giEnvironmentStrength: 99,
    giDirectStrength: -1, giConeAperture: 99, giConeCount: 2,
  });
  assert.deepEqual({
    bounce: normalized.giBounceStrength, occlusion: normalized.giOcclusionStrength,
    environment: normalized.giEnvironmentStrength, direct: normalized.giDirectStrength,
    aperture: normalized.giConeAperture, cones: normalized.giConeCount,
  }, { bounce: 4, occlusion: 0, environment: 2, direct: 0, aperture: 1.4, cones: 3 });
  const upgraded = normalizeSvoRenderTuning({
    ...DEFAULT_SVO_RENDER_TUNING,
    giBounceStrength: undefined, giOcclusionStrength: undefined, giEnvironmentStrength: undefined,
    giDirectStrength: undefined, giConeAperture: undefined, giConeCount: undefined,
  } as unknown as typeof DEFAULT_SVO_RENDER_TUNING);
  assert.equal(upgraded.giBounceStrength, DEFAULT_SVO_RENDER_TUNING.giBounceStrength,
    "an older persisted tuning object upgrades to the image-forward GI defaults");
  assert.equal(upgraded.giConeCount, DEFAULT_SVO_RENDER_TUNING.giConeCount);
  const panel = readFileSync(new URL("../components/VisualPanel.tsx", import.meta.url), "utf8");
  for (const label of ["GI bounce", "GI occlusion", "Diffuse environment", "Direct key", "GI cones", "GI cone aperture"]) {
    assert.ok(panel.includes(`label="${label}"`), `${label} is live-tunable`);
  }
  assert.match(svoDrySceneShader, /if\(!finiteRadiance\|\|!finiteVisibility\)\{[^]*return DryGlobalIllumination\(vec3f\(0\.0\),1\.0,0u\);\}[^]*indirect\+=result\.radiance\*weight;[^]*let visibleThroughStatic=result\.transmittance;[^]*visibility\+=select\(visibleThroughStatic,0\.0,rigidBlocked\)\*weight/,
    "one validated GI gather must supply both bounced light and broad occlusion");
  assert.match(svoDrySceneShader, /direct\*directScale\+indirectDiffuse/);
  assert.match(svoDrySceneShader, /let directScale=dry\.giLighting\.w/,
    "live analytic direct lighting must not change when derived GI pages become ready");
  assert.doesNotMatch(svoDrySceneShader, /directScale=select\(1\.0,dry\.giLighting\.w,globalIllumination\)/,
    "derived radiance readiness cannot stand in for a baked direct-light contribution");
});

test("retired temporal and interlaced-shadow tuning cannot re-enter a preset", () => {
  for (const preset of Object.values(SVO_RENDER_TUNING_PRESETS)) {
    assert.equal("temporalEnabled" in preset, false);
    assert.equal("checkerboardShadowsEnabled" in preset, false);
    assert.equal("temporalMaximumSamples" in preset, false);
  }
  const panel = readFileSync(new URL("../components/VisualPanel.tsx", import.meta.url), "utf8");
  assert.doesNotMatch(panel, /Interlaced shadows|Temporal resolve|Maximum history|Variance clamp/);
});

test("radiance reconstruction modes normalize and remain available for live visual A/B", () => {
  assert.deepEqual(SVO_CONE_RADIANCE_RECONSTRUCTION_CODES, {
    nearest: 0, "gated-linear": 1, "joint-bilateral": 2, "wide-relight": 3, "full-res-relight": 4,
  });
  for (const coneRadianceReconstruction of ["nearest", "gated-linear", "joint-bilateral", "wide-relight", "full-res-relight"] as const) {
    assert.equal(normalizeSvoRenderTuning({
      ...DEFAULT_SVO_RENDER_TUNING,
      coneRadianceReconstruction,
    }).coneRadianceReconstruction, coneRadianceReconstruction);
  }
  assert.equal(normalizeSvoRenderTuning({
    ...DEFAULT_SVO_RENDER_TUNING,
    coneRadianceReconstruction: "invalid" as never,
  }).coneRadianceReconstruction, "full-res-relight");
  assert.equal(DEFAULT_SVO_RENDER_TUNING.coneRadianceReconstruction, "full-res-relight",
    "the measured split relight path is the production default");
  const panel = readFileSync(new URL("../components/VisualPanel.tsx", import.meta.url), "utf8");
  for (const label of ["EXACT", "LINEAR", "BILAT", "WIDE", "RELIGHT"]) assert.ok(panel.includes(`label: "${label}"`));
});

/**
 * The refinement depth has exactly one home, and it is the document.
 *
 * It used to be two numbers — a `SvoRenderTuning` field that divided the tree's
 * cell, and `voxelDomain.detailCellSize_m`, which every scenery generator is
 * expanded against. Nothing held them in step, so the render panel's slider
 * built a finer tree over a set still authored at the coarse size: on
 * `hero-garden-hose` the stepping-stone treads stayed 12-17 voxels across while
 * the tuning claimed a 0.78 mm leaf. These pin the collapse, not the arithmetic.
 */
test("the refinement depth is not a tuning field any more", () => {
  assert.equal("environmentRefinementDepth" in DEFAULT_SVO_RENDER_TUNING, false);
  for (const preset of Object.keys(SVO_RENDER_TUNING_PRESETS) as SvoRenderTuningPreset[]) {
    assert.equal("environmentRefinementDepth" in SVO_RENDER_TUNING_PRESETS[preset], false, preset);
  }
  // Normalization must not resurrect it from an old persisted or linked value.
  assert.equal("environmentRefinementDepth" in normalizeSvoRenderTuning({
    ...DEFAULT_SVO_RENDER_TUNING,
    environmentRefinementDepth: 3,
  } as never), false);
  // And the panel's slider must go through the controller's one setter rather
  // than write tuning, or the two numbers come straight back with the symptom.
  const panel = readFileSync(new URL("../components/VisualPanel.tsx", import.meta.url), "utf8");
  assert.doesNotMatch(panel, /updateTuning\("environmentRefinementDepth"/);
  assert.match(panel, /simulation\.setEnvironmentRefinementDepth\(/);
  // Disabled only by water. Gating it on the factory taking a lattice killed the
  // control on 37 of 39 presets, where a finer leaf still resolves the analytic
  // proxies genuinely better — see `setEnvironmentRefinementDepth`.
  assert.match(panel, /disabled=\{!sceneIsDry\}/);
});

test("the document's depth and its detail cell are exact inverses", () => {
  for (const finestCellSize_m of [0.025, 0.00625, 0.1]) {
    for (let depth = 0; depth <= SVO_ENVIRONMENT_REFINEMENT_DEPTH_MAXIMUM; depth += 1) {
      const detailCellSize_m = svoSceneryDetailCellSize_m(finestCellSize_m, {
        environmentRefinementDepth: depth, fluid: false,
      });
      assert.equal(
        svoSceneryRefinementDepth({ finestCellSize_m, detailCellSize_m }, { fluid: false }),
        depth, `${finestCellSize_m} m at depth ${depth}`);
    }
  }
  // An absent detail cell is depth zero, which is what an unauthored document
  // says: the set was expanded at the lattice's own cell.
  assert.equal(svoSceneryRefinementDepth({ finestCellSize_m: 0.00625 }, { fluid: false }), 0);
  // Zero on a wet scene however fine the document claims to be — a solver brick
  // pins its node, so there is nowhere to descend. Same rule as the forward
  // function, which is why they cannot disagree about a fluid scene either.
  assert.equal(svoSceneryRefinementDepth(
    { finestCellSize_m: 0.00625, detailCellSize_m: 0.00078125 }, { fluid: true }), 0);
  // Clamped, so an externally supplied lattice cannot ask for an unbounded one.
  assert.equal(svoSceneryRefinementDepth(
    { finestCellSize_m: 0.025, detailCellSize_m: 0.025 / 2 ** 9 }, { fluid: false }),
    SVO_ENVIRONMENT_REFINEMENT_DEPTH_MAXIMUM);
});

test("a dry document opens at the default refinement rung, through its own factory", async () => {
  assert.equal(SVO_ENVIRONMENT_REFINEMENT_DEPTH_DEFAULT, 1,
    "dry SVO scenes default to one refinement level");
  const { getScenePreset, findSceneDefinition } = await import("../lib/scenes");
  const { sceneDefinitionTakesLattice } = await import("../lib/scene-definition");
  const hero = getScenePreset("hero-garden-hose").create();
  assert.equal(hero.systems?.fluid, false, "the hero opens dry, which is what makes a depth legal");
  assert.equal(
    svoSceneryRefinementDepth(hero.voxelDomain, { fluid: false }),
    SVO_ENVIRONMENT_REFINEMENT_DEPTH_DEFAULT,
    "the document itself carries the rung, because the renderer reads it from there");
  // Through `buildAt`, not written on afterwards: the terrain pitch follows the
  // detail cell, so a document that acquired one after construction would claim
  // a leaf its ground was never sampled for.
  const { terrainSampleShape } = await import("../lib/terrain");
  const ground = terrainSampleShape(hero.terrain);
  assert.ok(ground, "the hero has ground to check the pitch of");
  assert.ok(ground.spacing_m <= (hero.voxelDomain.detailCellSize_m ?? hero.voxelDomain.finestCellSize_m),
    `terrain sampled at ${ground.spacing_m} m must not be coarser than the ${hero.voxelDomain.detailCellSize_m} m leaf`);
  // A definition with no lattice input opens at its own, and must not have a
  // finer detail cell invented for it — there is no factory to re-run.
  for (const preset of ["garden-svo-lighting"]) {
    const definition = findSceneDefinition(preset);
    if (definition && sceneDefinitionTakesLattice(definition)) continue;
    const scene = getScenePreset(preset).create();
    assert.equal(scene.voxelDomain.detailCellSize_m, undefined,
      `${preset} has no buildAt, so nothing may author a detail cell its generators did not run at`);
  }
});
