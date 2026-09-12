import assert from "node:assert/strict";
import test from "node:test";

import { productionSceneSliceSeedById } from
  "../lib/methods/adaptive-volume/advance-slice/production-scene-slice";
import { advanceSlice, createAdvanceSlice } from
  "../lib/methods/adaptive-volume/advance-slice/slice-solver";
import { slicePresentationReady, slicePresentationRevision } from "./playback";

test("play waits for each mutable slice revision to reach the canvas", () => {
  const slice = createAdvanceSlice(
    productionSceneSliceSeedById("coarse-first-pool-impact-half"));
  let painted = slicePresentationRevision(slice);
  assert.equal(slicePresentationReady(painted, slice), true);

  advanceSlice(slice, { pressureIterations: 4 });
  assert.equal(slicePresentationReady(painted, slice), false,
    "the next play tick must not overwrite a frame React has not painted");

  painted = slicePresentationRevision(slice);
  assert.equal(slicePresentationReady(painted, slice), true,
    "painting the accepted RDF/VOF revision reopens play admission");
});
