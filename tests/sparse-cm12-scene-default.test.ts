import assert from "node:assert/strict";
import test from "node:test";

import "../lib/methods";
import { defaultMethodId } from "../lib/core/method-registry";
import { scenePresets } from "../lib/core/scenes";
import { parseQueryState } from "../lib/core/url-state";

test("every scene opens with Sparse CM12 unless the URL explicitly chooses a method", () => {
  assert.equal(defaultMethodId(), "adaptive-volume");
  for (const scene of scenePresets) {
    assert.equal(parseQueryState(`?scene=${encodeURIComponent(scene.id)}`).methodId,
      "adaptive-volume", scene.id);
  }
});
