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

test("the view rail's sections come from installed features, with the controls placed in them", () => {
  for (const fluid of [true, false]) {
    const composition = composeFeatureUI("uniform-volume", fluid);
    const lighting = composition.sections.find(section => section.surface === "view-rail" && section.id === "lighting");
    assert.equal(lighting?.feature, "svo.lighting-visibility");
    assert.ok(composition.placements.some(p => p.slot === lighting?.slot && p.control === "far-ao-taps" && p.presentation === "compact"));
    const performance = composition.sections.find(section => section.surface === "view-rail" && section.id === "performance");
    assert.equal(performance?.feature, "view.rail");
    assert.ok(composition.placements.some(p => p.slot === performance?.slot && p.feature === "svo.lighting-visibility" && p.control === "still-view-reuse"));
    assert.ok(!composition.placements.some(p => p.slot === lighting?.slot && p.control === "still-view-reuse"));
    const diagnostics = composition.sections.find(section => section.surface === "view-rail" && section.id === "ao-diagnostics");
    assert.ok(diagnostics);
    assert.ok(composition.placements.some(p => p.slot === diagnostics.slot && p.control === "ao-tap-overlay"));
    // A control has one home on the rail; a second section would only repeat it.
    const railSlots = new Set(composition.sections.filter(section => section.surface === "view-rail").map(section => section.slot));
    const railHomes = composition.placements.filter(p => railSlots.has(p.slot)).map(p => `${p.feature}/${p.control}`);
    assert.deepEqual(railHomes, [...new Set(railHomes)]);
  }
});

test("the study rail holds the clock, the method's tuning and the layers, and nothing on the scene strip repeats them", () => {
  const composition = composeFeatureUI("uniform-narrow-band-flip", true, {}, { detailPolicy: "dynamic" });
  const study = composition.sections.filter(section => section.surface === "study-rail");
  assert.deepEqual(study.map(section => section.id), ["simulation", "tune", "layers"]);
  const clock = study.find(section => section.id === "simulation")!;
  assert.deepEqual(composition.placements.filter(p => p.slot === clock.slot).map(p => `${p.feature}/${p.control}`),
    ["simulation/step", "simulation/cost", "simulation/more"]);
  const layers = study.find(section => section.id === "layers")!;
  const inLayers = composition.placements.filter(p => p.slot === layers.slot).map(p => `${p.feature}/${p.control}`);
  // The particle display is a layer: it stands first in the layer card.
  assert.deepEqual(inLayers, ["presentation.surface-display/particles", "presentation.visual-layers/layers"]);
  assert.ok(!composition.placements.some(p => p.slot.startsWith("scene.") && p.control === "particles"));
  assert.ok(composition.placements.some(p => p.slot === "rail.legend" && p.feature === "presentation.visual-layers"));
  // Off Dynamic the dials would move nothing, so the method installs no Tune.
  const requested = composeFeatureUI("uniform-narrow-band-flip", true, {}, { detailPolicy: "requested" });
  assert.deepEqual(requested.sections.filter(section => section.surface === "study-rail").map(section => section.id), ["simulation", "layers"]);
  // A dry document has no solve to tune and no fields to draw.
  assert.ok(!composeFeatureUI("uniform-narrow-band-flip", false).sections.some(section => section.surface === "study-rail"));
});
