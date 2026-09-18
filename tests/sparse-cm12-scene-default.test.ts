import assert from "node:assert/strict";
import test from "node:test";

import "../lib/methods";
import { defaultMethodId, getMethod } from "../lib/core/method-registry";
import { resolveMethodValues } from "../lib/core/method-contract";
import { scenePresets } from "../lib/core/scenes";
import { parseQueryState } from "../lib/core/url-state";

test("every scene opens with Uniform Geometric unless the URL explicitly chooses a method", () => {
  assert.equal(defaultMethodId(), "uniform-volume");
  assert.equal(resolveMethodValues(getMethod("uniform-volume"), "balanced", {}).velocityTransport, "semi-lagrangian");
  assert.equal(resolveMethodValues(getMethod("uniform-volume"), "balanced", {}).liquidCapacityBalancing, "off");
  assert.equal(parseQueryState("?method=adaptive-volume").methodId, "adaptive-volume");
  for (const scene of scenePresets) {
    assert.equal(parseQueryState(`?scene=${encodeURIComponent(scene.id)}`).methodId,
      "uniform-volume", scene.id);
  }
});
