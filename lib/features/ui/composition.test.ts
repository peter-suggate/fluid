import "../../methods";
import { registeredSimulationMethods } from "../../core/method-registry";
import assert from "node:assert/strict";
import test from "node:test";
import { composeFeatureUI } from "./composition";

test("UI installation follows active method and fluid capability", () => {
  const adaptive = composeFeatureUI("adaptive-mass", true);
  assert.ok(adaptive.placements.some(p => p.slot === "scene.adaptivity"));
  const uniform = composeFeatureUI("uniform", true);
  assert.ok(!uniform.placements.some(p => p.slot === "scene.adaptivity"));
  assert.ok(uniform.placements.some(p => p.slot === "scene.physics"));
  const dry = composeFeatureUI("adaptive-mass", false);
  assert.ok(!dry.placements.some(p => p.slot === "scene.physics" || p.slot === "scene.adaptivity"));
  assert.ok(dry.placements.some(p => p.slot === "scene.visibility"));
});

test("UI composition resolves the active variants rather than their defaults", () => {
  const resolved = composeFeatureUI("adaptive-mass", true, {
    "svo.primary-visibility": "traced",
    "svo.lighting-visibility": "cones",
  }, { selectorMode: "activity" });
  assert.equal(resolved.variants.find(v => v.point === "simulation.adaptive-mass.adaptivity")?.id, "activity");
});

test("every registered method supplies its complete active composition to the UI", () => {
  for (const method of registeredSimulationMethods()) {
    const resolved = composeFeatureUI(method.id, true);
    const expected = method.resolveComposition({});
    for (const feature of expected.features) assert.ok(resolved.features.some(candidate => candidate.id === feature.id));
    for (const variant of expected.variants) assert.ok(resolved.variants.some(candidate => candidate.point === variant.point && candidate.id === variant.id));
  }
});
