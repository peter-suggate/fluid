import "../../methods";
import { registeredSimulationMethods } from "../../core/method-registry";
import assert from "node:assert/strict";
import test from "node:test";
import { composeFeatureUI } from "./composition";

test("UI installation follows active method and fluid capability", () => {
  const uniform = composeFeatureUI("uniform-volume", true);
  assert.ok(!uniform.placements.some(p => p.slot === "scene.adaptivity"));
  assert.ok(uniform.placements.some(p => p.slot === "scene.physics"));
  const dry = composeFeatureUI("uniform-volume", false);
  assert.ok(!dry.placements.some(p => p.slot === "scene.physics" || p.slot === "scene.adaptivity"));
  assert.ok(dry.placements.some(p => p.slot === "scene.visibility"));
});

test("the maintained method supplies its complete active composition to the UI", () => {
  for (const method of registeredSimulationMethods().filter(method => method.id === "uniform-volume")) {
    const resolved = composeFeatureUI(method.id, true);
    const expected = method.resolveComposition({});
    for (const feature of expected.features) assert.ok(resolved.features.some(candidate => candidate.id === feature.id));
    for (const variant of expected.variants) assert.ok(resolved.variants.some(candidate => candidate.point === variant.point && candidate.id === variant.id));
  }
});
