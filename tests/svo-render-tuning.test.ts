import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  DEFAULT_SVO_RENDER_TUNING,
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

test("cone prepass tuning preserves every supported spatial rate", () => {
  for (const coneLightingScale of [1, 0.5, 0.25, 0.125] as const) {
    assert.equal(normalizeSvoRenderTuning({
      ...DEFAULT_SVO_RENDER_TUNING,
      coneLightingScale,
    }).coneLightingScale, coneLightingScale);
  }
  const panel = readFileSync(new URL("../components/VisualPanel.tsx", import.meta.url), "utf8");
  for (const label of ["FULL", "2×2", "4×4", "8×8"]) assert.ok(panel.includes(`label: "${label}"`));
});
