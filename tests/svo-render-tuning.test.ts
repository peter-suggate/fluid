import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  DEFAULT_SVO_RENDER_TUNING,
  SVO_CONE_RADIANCE_RECONSTRUCTION_CODES,
  SVO_ENVIRONMENT_BRICK_REFINEMENT_MAXIMUM,
  SVO_PRIMARY_LEAF_VISIT_HARD_LIMIT,
  SVO_RENDER_TUNING_PRESETS,
  normalizeSvoRenderTuning,
} from "../lib/svo-render-tuning";
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

test("shadow cone aperture is exposed in the initially open cone controls", () => {
  const panel = readFileSync(new URL("../components/VisualPanel.tsx", import.meta.url), "utf8");
  assert.match(panel, /ControlGroup title="Cone tracing"[^>]* open>/);
  assert.match(panel, /label="Shadow cone aperture"[^>]*value=\{tuning\.shadowConeAperture\}/);
  assert.match(panel, /updateTuning\("shadowConeAperture", value\)/);
});

test("global illumination exposes image-shaping controls with cinematic balanced defaults", () => {
  assert.deepEqual({
    bounce: DEFAULT_SVO_RENDER_TUNING.giBounceStrength,
    occlusion: DEFAULT_SVO_RENDER_TUNING.giOcclusionStrength,
    environment: DEFAULT_SVO_RENDER_TUNING.giEnvironmentStrength,
    direct: DEFAULT_SVO_RENDER_TUNING.giDirectStrength,
    aperture: DEFAULT_SVO_RENDER_TUNING.giConeAperture,
    cones: DEFAULT_SVO_RENDER_TUNING.giConeCount,
  }, { bounce: 1.5, occlusion: 0.82, environment: 0.65, direct: 0.9, aperture: 1.05, cones: 4 });
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
  assert.match(svoDrySceneShader, /indirect\+=select\(vec3f\(0\.0\),result\.radiance,finiteRadiance\)\*weight;[^]*visibility\+=select\(1\.0,result\.transmittance,finiteVisibility\)\*weight/,
    "one fail-soft GI gather must supply both bounced light and broad occlusion");
  assert.match(svoDrySceneShader, /direct\*directScale\+indirectDiffuse/);
});

test("temporal history caps are halved consistently across quality presets", () => {
  assert.equal(SVO_RENDER_TUNING_PRESETS.performance.temporalMaximumSamples, 16);
  assert.equal(DEFAULT_SVO_RENDER_TUNING.temporalMaximumSamples, 32);
  assert.equal(SVO_RENDER_TUNING_PRESETS.quality.temporalMaximumSamples, 48);
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
