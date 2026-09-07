import assert from "node:assert/strict";
import test from "node:test";

import { RENDER_PIPELINE_NODES } from "../../../pipeline/render-pipeline-graph";
import { DEFAULT_SVO_RENDER_TUNING } from "../../../pipeline/svo-render-tuning";

test("stationary primary reuse has no product tuning or frame-graph node", () => {
  assert.equal("stationaryPrimaryReuseEnabled" in DEFAULT_SVO_RENDER_TUNING, false);
  assert.equal(RENDER_PIPELINE_NODES.some(({ id }) => id === ("stationary-reuse" as never)), false);
});
