import assert from "node:assert/strict";
import test from "node:test";
import {
  OCTREE_POWER_SAME_OR_COARSER_FLAG,
  sitesForSameOrCoarserPowerDescriptor,
} from "../lib/octree-power-descriptor";
import {
  OCTREE_CUBE_TRANSFORMS,
  auditOctreePowerTopology,
  composeCubeTransforms,
  enforcePaperCompatibleOctreeGrading,
  inverseCubeTransform,
  octreePowerCoarseMaskNeedsAcuteRepair,
  transformPowerVector,
} from "../lib/octree-power-topology";

test("co-spherical same/coarser masks no longer trigger a topology split", () => {
  for (const mask of [25, 42, 52, 57, 58, 60]) {
    assert.equal(octreePowerCoarseMaskNeedsAcuteRepair(mask), false, `mask ${mask}`);
    const descriptor = (OCTREE_POWER_SAME_OR_COARSER_FLAG | (mask << 3)) >>> 0;
    const leaves = sitesForSameOrCoarserPowerDescriptor(descriptor).map((site) => ({
      key: site.key, origin: site.origin, size: site.size,
    }));
    const before = auditOctreePowerTopology(leaves);
    assert.equal(before.strictlyObtuseSameOrCoarserLeaves, 0, `mask ${mask}`);
    assert.deepEqual(before.acuteRepairCoarseLeaves, [], `mask ${mask}`);
    assert.equal(before.ordinaryTwoToOne, true, `mask ${mask} remains ordinarily 2:1`);
    assert.equal(before.paperCompatible, true, `mask ${mask} is covered by the row-local Delaunay catalog`);

    const repaired = enforcePaperCompatibleOctreeGrading(leaves, 100);
    assert.equal(repaired.refinedParents, 0, `mask ${mask}`);
    assert.equal(repaired.iterations, 0, `mask ${mask}`);
    assert.equal(repaired.audit.strictlyObtuseSameOrCoarserLeaves, 0, `mask ${mask}`);
    assert.equal(repaired.audit.paperCompatible, true, `mask ${mask}`);
    assert.deepEqual(repaired.leaves, [...leaves].sort((a, b) => a.key.localeCompare(b.key)),
      `mask ${mask} must not propagate refinement`);
  }
});

test("topology audit distinguishes ordinary 2:1 balance from paper grading", () => {
  const audit = auditOctreePowerTopology([
    { key: "coarse", origin: [0, 4, 0], size: 4 },
    { key: "middle", origin: [4, 4, 0], size: 2 },
    { key: "fine-a", origin: [6, 4, 0], size: 1 },
    { key: "fine-b", origin: [6, 5, 0], size: 1 },
  ]);
  assert.equal(audit.liveLeafCount, 4);
  assert.equal(audit.ordinaryTwoToOne, true);
  assert.equal(audit.paperCompatible, false);
  assert.equal(audit.mixedFinerAndCoarserLeaves, 1);
  assert.equal(audit.maximumFaceNeighborLevelDifference, 1);
  assert.deepEqual(audit.countsBySize, { "1": 2, "2": 1, "4": 1 });
  assert.equal(audit.leaves.find((leaf) => leaf.key === "middle")?.gradingCase, "mixed");
});

test("paper grading deterministically refines coarse neighbors until mixed 1-rings disappear", () => {
  const input = [
    { key: "coarse", origin: [0, 4, 0] as const, size: 4 },
    { key: "middle", origin: [4, 4, 0] as const, size: 2 },
    { key: "fine-a", origin: [6, 4, 0] as const, size: 1 },
    { key: "fine-b", origin: [6, 5, 0] as const, size: 1 },
  ];
  const result = enforcePaperCompatibleOctreeGrading(input, 2);
  assert.equal(result.accepted, true);
  assert.equal(result.audit.paperCompatible, true);
  assert.equal(result.audit.ordinaryTwoToOne, true);
  assert.equal(result.iterations, 1);
  assert.equal(result.refinedParents, 1);
  assert.equal(result.leaves.length, 11);
  const volume = (leaves: readonly { size: number }[]) => leaves.reduce((sum, leaf) => sum + leaf.size ** 3, 0);
  assert.equal(volume(result.leaves), volume(input));
  assert.deepEqual(enforcePaperCompatibleOctreeGrading([...input].reverse(), 2).leaves, result.leaves);

  const rejected = enforcePaperCompatibleOctreeGrading(input);
  assert.equal(rejected.accepted, false);
  assert.match(rejected.rejectionReason!, /leaf growth/);
});

test("paper grading rebalances a propagated ordinary 2:1 violation", () => {
  const result = enforcePaperCompatibleOctreeGrading([
    { key: "outer", origin: [-8, 0, 0], size: 8 },
    { key: "coarse", origin: [0, 0, 0], size: 4 },
    { key: "middle", origin: [4, 0, 0], size: 2 },
    { key: "fine-a", origin: [6, 0, 0], size: 1 },
    { key: "fine-b", origin: [6, 1, 0], size: 1 },
  ], 20);
  assert.equal(result.accepted, true);
  assert.equal(result.audit.ordinaryTwoToOne, true);
  assert.equal(result.audit.paperCompatible, true);
  assert.ok(result.refinedParents >= 2, "strong grading should trigger an ordinary-balance propagation");
  assert.equal(result.leaves.some((leaf) => leaf.key === "outer"), false);
});

test("topology audit reports edge-neighbor level differences", () => {
  const audit = auditOctreePowerTopology([
    { key: "large", origin: [0, 0, 0], size: 4 },
    { key: "small", origin: [4, 4, 0], size: 1 },
  ]);
  assert.equal(audit.maximumEdgeNeighborLevelDifference, 2);
  assert.equal(audit.ordinaryTwoToOne, false);
  assert.equal(audit.neighbors[0].kind, "edge");
});

test("topology audit rejects overlap and non-dyadic alignment", () => {
  assert.throws(() => auditOctreePowerTopology([
    { key: "a", origin: [0, 0, 0], size: 2 }, { key: "b", origin: [1, 1, 1], size: 1 },
  ]), /overlap/);
  assert.throws(() => auditOctreePowerTopology([{ key: "a", origin: [1, 0, 0], size: 2 }]), /aligned/);
});

test("cube transform catalog contains 24 rotations and 24 reflections with exact inverses", () => {
  assert.equal(OCTREE_CUBE_TRANSFORMS.length, 48);
  assert.equal(OCTREE_CUBE_TRANSFORMS.filter((transform) => transform.determinant === 1).length, 24);
  assert.equal(OCTREE_CUBE_TRANSFORMS.filter((transform) => transform.determinant === -1).length, 24);
  const sample = [2, -3, 5] as const;
  for (const transform of OCTREE_CUBE_TRANSFORMS) {
    assert.deepEqual(transformPowerVector(transformPowerVector(sample, transform), inverseCubeTransform(transform)), sample);
  }
});

test("cube transforms compose exactly inside the finite group", () => {
  const sample = [2, -3, 5] as const;
  for (const first of OCTREE_CUBE_TRANSFORMS) for (const second of OCTREE_CUBE_TRANSFORMS) {
    assert.deepEqual(transformPowerVector(sample, composeCubeTransforms(first, second)),
      transformPowerVector(transformPowerVector(sample, first), second));
  }
});
