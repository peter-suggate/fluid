import assert from "node:assert/strict";
import test from "node:test";

import {
  OCTREE_REGULAR_BAND_INCIDENCE_PER_ROW,
  OCTREE_REGULAR_BAND_OWNED_FACES_PER_ROW,
  planOctreeRegularFaceBand,
  summarizeOctreeFaceBandPhi,
} from "../lib/octree-face-band";

test("regular face-band capacity is bounded by wet rows plus active fine bricks", () => {
  const factor4 = planOctreeRegularFaceBand(100, 20, 4, 4);
  assert.equal(factor4.ownerCandidatesPerBrick, 1);
  assert.equal(factor4.rowCapacity, 120);
  assert.equal(OCTREE_REGULAR_BAND_OWNED_FACES_PER_ROW, 12);
  assert.equal(OCTREE_REGULAR_BAND_INCIDENCE_PER_ROW, 24);
  assert.equal(factor4.faceCapacity, 120 * OCTREE_REGULAR_BAND_OWNED_FACES_PER_ROW);
  assert.equal(factor4.allocatedBytes, factor4.faceBytes + factor4.incidenceBytes + factor4.phiBytes);
  assert.equal(planOctreeRegularFaceBand(100, 20, 4, 8).ownerCandidatesPerBrick, 1);
  const wideBrick = planOctreeRegularFaceBand(100, 20, 8, 4);
  assert.equal(wideBrick.ownerCandidatesPerBrick, 8);
  assert.equal(wideBrick.rowCapacity, 260);
});

test("fine-cell phi summary preserves a mixed-sign interval", () => {
  assert.deepEqual(summarizeOctreeFaceBandPhi([-0.4, 0.2, 0.7]), {
    representativePhi: 0.2,
    minimumPhi: -0.4,
    maximumPhi: 0.7,
  });
  assert.deepEqual(summarizeOctreeFaceBandPhi([0.2, -0.2]), {
    representativePhi: -0.2,
    minimumPhi: -0.2,
    maximumPhi: 0.2,
  });
});
