import assert from "node:assert/strict";
import test from "node:test";

import {
  OCTREE_FACE_MARCH_INVALID,
  OCTREE_REGULAR_BAND_INCIDENCE_PER_ROW,
  OCTREE_REGULAR_BAND_OWNED_FACES_PER_ROW,
  fastMarchOctreeFaceVelocity,
  planOctreeRegularFaceBand,
  summarizeOctreeFaceBandPhi,
} from "../lib/octree-face-fast-march";

test("regular face-band capacity is bounded by wet rows plus active fine bricks", () => {
  const factor4 = planOctreeRegularFaceBand(100, 20, 4, 4);
  assert.equal(factor4.ownerCandidatesPerBrick, 1);
  assert.equal(factor4.rowCapacity, 120);
  assert.equal(OCTREE_REGULAR_BAND_OWNED_FACES_PER_ROW, 12);
  assert.equal(OCTREE_REGULAR_BAND_INCIDENCE_PER_ROW, 24);
  assert.equal(factor4.faceCapacity, 120 * OCTREE_REGULAR_BAND_OWNED_FACES_PER_ROW);
  const splitFactor8 = planOctreeRegularFaceBand(100, 20, 4, 8);
  assert.equal(splitFactor8.ownerCandidatesPerBrick, 1);
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
  }, "equal-distance ties prefer the smaller signed phi deterministically");
});

test("face fast march copies the closest-to-surface incident velocity deterministically", () => {
  const result = fastMarchOctreeFaceVelocity(4, [
    { negativeRow: 0, positiveRow: 1, phi: 0.05, normalVelocity: 3 },
    { negativeRow: 1, positiveRow: 2, phi: 0.15 },
    { negativeRow: 2, positiveRow: 3, phi: 0.25 },
    { negativeRow: 1, positiveRow: OCTREE_FACE_MARCH_INVALID, phi: 0.1, normalVelocity: -2 },
  ]);
  assert.deepEqual(Array.from(result.velocities), [3, 3, 3, -2]);
  assert.deepEqual(Array.from(result.parents), [0, 0, 1, 3]);
  assert.deepEqual(Array.from(result.graphDistance), [0, 1, 2, 0]);
  assert.equal(result.maximumGraphDistance, 2);
  assert.equal(result.acceptedCount, 4);
  assert.deepEqual(Array.from(result.unresolvedFaces), []);
});

test("face fast march reports a disconnected or non-monotone band instead of fabricating velocity", () => {
  const result = fastMarchOctreeFaceVelocity(4, [
    { negativeRow: 0, positiveRow: 1, phi: 0.2, normalVelocity: 1 },
    { negativeRow: 1, positiveRow: 2, phi: 0.1 },
    { negativeRow: 3, positiveRow: OCTREE_FACE_MARCH_INVALID, phi: 0.3 },
  ]);
  assert.equal(result.acceptedCount, 1);
  assert.deepEqual(Array.from(result.unresolvedFaces), [1, 2]);
  assert.equal(result.parents[1], OCTREE_FACE_MARCH_INVALID);
});
