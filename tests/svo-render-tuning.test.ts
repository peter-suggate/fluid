import assert from "node:assert/strict";
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
