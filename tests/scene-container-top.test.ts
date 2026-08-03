import assert from "node:assert/strict";
import test from "node:test";
import { defaultScene } from "../lib/model";
import { createPaperScenario, paperScenarios } from "../lib/paper-scenarios";
import { scenePresets } from "../lib/scenes";

/**
 * Split on the environment, not on the shelf.
 *
 * This used to partition by `preset.group === "Garden"`, which read a display
 * label as though it were a physical fact — so the moment the lighting study
 * moved to a Rendering shelf, an open-topped garden was asserted to be a closed
 * tank. The environment is the art direction the open top belongs to, and it
 * lives in the document.
 */
const isGarden = (preset: (typeof scenePresets)[number]) => preset.background === "garden";

test("water-tank scenes default to a closed top", () => {
  assert.equal(defaultScene.container.top, "closed");

  for (const preset of scenePresets.filter((candidate) => !isGarden(candidate))) {
    assert.equal(preset.create().container.top, "closed", `${preset.id} must default to a closed tank`);
  }

  for (const scenario of paperScenarios) {
    assert.equal(createPaperScenario(scenario.id).container.top, "closed", `${scenario.id} must default to a closed tank`);
  }
});

test("garden scenes remain open environments", () => {
  const gardens = scenePresets.filter(isGarden);
  assert.ok(gardens.length > 0, "the partition must actually cover something");
  for (const preset of gardens) {
    assert.equal(preset.create().container.top, "open", `${preset.id} must keep an open top`);
  }
});
