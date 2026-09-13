import assert from "node:assert/strict";
import test from "node:test";

import type { AdvanceView } from "../lib/physics-wasm/advance-view";
import { advancePresentationReady, advancePresentationRevision } from "./playback";

const view = (commandSequence: number, frame: number): AdvanceView => ({
  revision: { schemaVersion: 1, dimension: 2, runEpoch: 7, commandSequence, frame,
    time: frame / 30, injections: 0, topologyGeneration: 2, fieldRevision: frame,
    surfaceRevision: frame, memoryEpoch: 0 },
} as AdvanceView);

test("play waits for each immutable Wasm publication to reach the canvas", () => {
  const first = view(1, 0);
  const painted = advancePresentationRevision(first);
  assert.equal(advancePresentationReady(painted, first), true);
  assert.equal(advancePresentationReady(painted, view(2, 1)), false,
    "the next play tick must not outrun the publication React painted");
});
