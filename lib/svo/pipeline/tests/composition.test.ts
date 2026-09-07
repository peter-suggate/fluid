import { SVO_RADIANCE_RECONSTRUCTION_OPTIONS } from "../../features/radiance/definition";
import assert from "node:assert/strict";
import test from "node:test";
import { composeFeatures, compositionUpdateImpact } from "../../../framework/composition";
import { createUIStore } from "../../../core/stores/ui-store";
import { resolveSvoPipelineComposition, SVO_PIPELINE_FEATURES } from "../composition";
import { SVO_PRIMARY_VISIBILITY_OPTIONS } from "../../features/primary-visibility/definition";
import { SVO_LIGHTING_VISIBILITY_OPTIONS } from "../../features/lighting-visibility/definition";

// These are supported compositions, not an accidental coupling between shadow
// quality and primary visibility. Exercise every product-visible combination.
test("SVO primary and lighting dimensions compose independently", () => {
  for (const primary of SVO_PRIMARY_VISIBILITY_OPTIONS) {
    for (const lighting of SVO_LIGHTING_VISIBILITY_OPTIONS) {
      for (const reconstruction of SVO_RADIANCE_RECONSTRUCTION_OPTIONS) {
      const resolved = resolveSvoPipelineComposition({ primaryTraversal: primary.value, coneTracingMode: lighting.value, coneRadianceReconstruction: reconstruction.value });
      assert.deepEqual(resolved.composition.variants.map(v => [v.point, v.id]), [
        ["svo.primary-visibility", primary.value], ["svo.lighting-visibility", lighting.value],
        ["svo.radiance-reconstruction", reconstruction.value],
      ]);
      assert.equal(resolved.primaryTraversal, primary.value);
      assert.equal(resolved.coneTracingMode, lighting.value);
      assert.equal(resolved.coneRadianceReconstruction, reconstruction.value);
      }
    }
  }
});

test("SVO resource requirements fail closed when a provider is removed or incompatible", () => {
  assert.throws(() => composeFeatures({ features: SVO_PIPELINE_FEATURES.filter(feature => feature.id !== "svo.scene-publication") }), /missing|Missing/);
  assert.throws(() => composeFeatures({ features: SVO_PIPELINE_FEATURES.map(feature => feature.id === "svo.scene-publication"
    ? { ...feature, outputs: [{ id: "svo.scene", representation: "wrong", lifetime: "generation" as const }] }
    : feature) }), /matching representation and lifetime/);
  assert.throws(() => resolveSvoPipelineComposition({ primaryTraversal: "unknown" as "mesh" }), /supported variant/);
});

test("SVO variants declare rebuild versus live impact", () => {
  const initial = resolveSvoPipelineComposition().composition;
  assert.equal(compositionUpdateImpact(initial, resolveSvoPipelineComposition({ coneTracingMode: "off" }).composition), "live");
  assert.equal(compositionUpdateImpact(initial, resolveSvoPipelineComposition({ primaryTraversal: "traced" }).composition), "rebuild");
});

test("primary-work selection is remembered across representations and isolated by session", () => {
  const first = createUIStore();
  const second = createUIStore();
  first.getState().setSvoStageView("primary-node-visits");
  first.getState().setSvoStageView("off");
  assert.equal(first.getState().svoLastPrimaryWorkView, "primary-node-visits");
  assert.equal(second.getState().svoLastPrimaryWorkView, "primary-work");
  first.getState().setSvoStageView("primary-depth");
  assert.equal(first.getState().svoLastPrimaryWorkView, "primary-node-visits");
});

test("invalid reconstruction edits do not mutate accepted settings", () => {
  const store = createUIStore();
  const before = store.getState().svoRenderTuning;
  assert.throws(() => store.getState().setSvoRenderTuning({ ...before, coneRadianceReconstruction: "invalid" as "nearest" }), /supported variant/);
  assert.equal(store.getState().svoRenderTuning, before);
  store.getState().setSvoRenderTuning(current => ({ ...current, coneRadianceReconstruction: "nearest" }));
  assert.equal(store.getState().svoRenderTuning.coneRadianceReconstruction, "nearest");
});
