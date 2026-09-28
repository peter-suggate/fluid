import assert from "node:assert/strict";
import test from "node:test";

import "../lib/methods";
import { defaultMethodId } from "../lib/core/method-registry";
import { scenePresets } from "../lib/core/scenes";
import { parseQueryState } from "../lib/core/url-state";

test("every scene opens with Uniform Geometric unless the URL explicitly chooses a method", () => {
  assert.equal(defaultMethodId(), "uniform-volume");
  assert.equal(parseQueryState("?method=adaptive-volume").methodId, "adaptive-volume");
  for (const scene of scenePresets) {
    assert.equal(parseQueryState(`?scene=${encodeURIComponent(scene.id)}`).methodId,
      "uniform-volume", scene.id);
  }
});
