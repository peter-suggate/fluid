import assert from "node:assert/strict";
import test from "node:test";
import { defaultScene } from "../lib/model";
import { createPaperScenario, paperScenarios } from "../lib/paper-scenarios";
import { scenePresets } from "../lib/scenes";

test("water-tank scenes default to a closed top", () => {
  assert.equal(defaultScene.container.top, "closed");

  for (const preset of scenePresets.filter(({ group }) => group !== "Garden")) {
    assert.equal(preset.create().container.top, "closed", `${preset.id} must default to a closed tank`);
  }

  for (const scenario of paperScenarios) {
    assert.equal(createPaperScenario(scenario.id).container.top, "closed", `${scenario.id} must default to a closed tank`);
  }
});

test("garden scenes remain open environments", () => {
  for (const preset of scenePresets.filter(({ group }) => group === "Garden")) {
    assert.equal(preset.create().container.top, "open", `${preset.id} must keep an open top`);
  }
});
