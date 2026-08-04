import assert from "node:assert/strict";
import test from "node:test";

import { environmentIds } from "../lib/environments";
import { cloneScene, defaultScene, type Vec3 } from "../lib/model";
import {
  SVO_PRIMITIVE_CANDIDATE_LEAF_SENTINEL,
  SVO_PRIMITIVE_CANDIDATE_MAXIMUM_LEAVES,
  SVO_PRIMITIVE_CANDIDATE_MAXIMUM_NODES,
  SVO_PRIMITIVE_CANDIDATE_MAXIMUM_STACK,
  buildSvoPrimitiveCandidates,
  createSvoPrimitiveCandidateRefitPlan,
  packSvoPrimitiveCandidateArena,
  querySvoPrimitiveCandidates,
  refitSvoPrimitiveCandidates,
  refitSvoPrimitiveCandidatesIncremental,
  svoPrimitiveCandidateBounds,
  traceSvoPrimitiveCandidates,
} from "../lib/svo-primitive-candidates";
import { intersectSvoPrimitive, type SvoFinitePrimitiveDescriptor } from "../lib/svo-primitive-abi";
import { packSvoPrimitiveRecords } from "../lib/svo-primitive-abi";
import { buildSvoScenePrimitives } from "../lib/svo-scene-primitives";

const identity = { w: 1, x: 0, y: 0, z: 0 };

function normalize(value: Vec3): Vec3 {
  const length = Math.hypot(value.x, value.y, value.z);
  return { x: value.x / length, y: value.y / length, z: value.z / length };
}

function rotate(q: { w: number; x: number; y: number; z: number }, v: Vec3): Vec3 {
  const tx = 2 * (q.y * v.z - q.z * v.y), ty = 2 * (q.z * v.x - q.x * v.z), tz = 2 * (q.x * v.y - q.y * v.x);
  return {
    x: v.x + q.w * tx + (q.y * tz - q.z * ty),
    y: v.y + q.w * ty + (q.z * tx - q.x * tz),
    z: v.z + q.w * tz + (q.x * ty - q.y * tx),
  };
}

/**
 * A point inside the solid. Aiming at the centre is only a hit for the convex
 * kinds: a ray at the middle of a torus goes straight through its hole, which
 * is the shape doing its job rather than a gap in coverage.
 */
function interiorTarget(descriptor: SvoFinitePrimitiveDescriptor): Vec3 {
  if (descriptor.kind !== "torus") return descriptor.center_m;
  const offset = rotate(descriptor.orientation ?? identity, { x: descriptor.majorRadius_m, y: 0, z: 0 });
  return {
    x: descriptor.center_m.x + offset.x,
    y: descriptor.center_m.y + offset.y,
    z: descriptor.center_m.z + offset.z,
  };
}

test("candidate bounds conservatively cover smooth, sharp, rotated, and subcell primitives", () => {
  const descriptors: SvoFinitePrimitiveDescriptor[] = [
    { kind: "sphere", primitiveId: 1, materialId: 2, ownerId: 1, center_m: { x: -2, y: 0, z: 0 }, radius_m: .7 },
    { kind: "box", primitiveId: 2, materialId: 3, ownerId: 2, center_m: { x: 0, y: 0, z: 0 }, halfExtents_m: { x: .8, y: .6, z: .5 }, orientation: { w: Math.SQRT1_2, x: 0, y: Math.SQRT1_2, z: 0 } },
    { kind: "box", primitiveId: 3, materialId: 3, ownerId: 3, center_m: { x: 2, y: 0, z: 0 }, halfExtents_m: { x: .4, y: .003, z: .2 }, orientation: identity },
    { kind: "ellipsoid", primitiveId: 4, materialId: 4, ownerId: 4, center_m: { x: 0, y: 2, z: 0 }, radii_m: { x: .8, y: .3, z: .5 }, orientation: identity },
    { kind: "torus", primitiveId: 5, materialId: 5, ownerId: 5, center_m: { x: -2, y: 2, z: 0 }, majorRadius_m: .6, minorRadius_m: .12, orientation: { w: Math.SQRT1_2, x: Math.SQRT1_2, y: 0, z: 0 } },
    { kind: "cone", primitiveId: 6, materialId: 6, ownerId: 6, center_m: { x: 2, y: 2, z: 0 }, baseRadius_m: .3, topRadius_m: .45, halfHeight_m: .4, orientation: identity },
  ];
  const publication = buildSvoPrimitiveCandidates(descriptors);
  for (let primitiveIndex = 0; primitiveIndex < descriptors.length; primitiveIndex += 1) {
    const descriptor = descriptors[primitiveIndex];
    const bounds = svoPrimitiveCandidateBounds(descriptor);
    const center = descriptor.center_m;
    const diagonal = Math.hypot(
      bounds.maximum_m.x - bounds.minimum_m.x,
      bounds.maximum_m.y - bounds.minimum_m.y,
      bounds.maximum_m.z - bounds.minimum_m.z,
    );
    const offset = normalize({ x: 1 + primitiveIndex, y: 2, z: 3 - primitiveIndex * .2 });
    const origin_m = {
      x: center.x + offset.x * (diagonal + 1),
      y: center.y + offset.y * (diagonal + 1),
      z: center.z + offset.z * (diagonal + 1),
    };
    const target = interiorTarget(descriptor);
    const direction = normalize({ x: target.x - origin_m.x, y: target.y - origin_m.y, z: target.z - origin_m.z });
    const ray = { origin_m, direction };
    assert.ok(intersectSvoPrimitive(descriptor, ray), `exact ray hits primitive ${primitiveIndex}`);
    assert.ok(querySvoPrimitiveCandidates(publication, ray).primitiveIndices.includes(primitiveIndex),
      `candidate traversal retains primitive ${primitiveIndex}`);
  }
});

test("offline candidate utility has no false negatives for shipped analytic catalogs", () => {
  const scene = cloneScene(defaultScene);
  for (const environmentId of environmentIds) {
    const built = buildSvoScenePrimitives(scene, { environmentId });
    const descriptors = built.descriptors as readonly SvoFinitePrimitiveDescriptor[];
    const primitiveCandidates = buildSvoPrimitiveCandidates(descriptors, { skippedOwnerId: built.openShellOwnerId });
    built.descriptors.forEach((descriptor, primitiveIndex) => {
      assert.notEqual(descriptor.kind, "terrain-heightfield");
      const finite = descriptor as SvoFinitePrimitiveDescriptor;
      if (finite.ownerId === built.openShellOwnerId) {
        assert.ok(!primitiveCandidates.nodes.some((node) => node.rightChildIndex === SVO_PRIMITIVE_CANDIDATE_LEAF_SENTINEL && node.leftOrPrimitiveIndex === primitiveIndex));
        return;
      }
      const bounds = svoPrimitiveCandidateBounds(finite);
      const center = finite.center_m;
      const span = Math.hypot(
        bounds.maximum_m.x - bounds.minimum_m.x,
        bounds.maximum_m.y - bounds.minimum_m.y,
        bounds.maximum_m.z - bounds.minimum_m.z,
      );
      const origin_m = { x: center.x + span + 1, y: center.y + .37 * (span + 1), z: center.z + .23 * (span + 1) };
      const target = interiorTarget(finite);
      const direction = normalize({ x: target.x - origin_m.x, y: target.y - origin_m.y, z: target.z - origin_m.z });
      const ray = { origin_m, direction };
      assert.ok(intersectSvoPrimitive(finite, ray), `${environmentId} exact hit ${primitiveIndex}`);
      assert.ok(querySvoPrimitiveCandidates(primitiveCandidates, ray).primitiveIndices.includes(primitiveIndex),
        `${environmentId} candidate retains ${primitiveIndex}`);
    });
  }
});

test("overlapping equal-depth owners retain the original lowest-index tie rule", () => {
  const descriptors: SvoFinitePrimitiveDescriptor[] = [
    { kind: "sphere", primitiveId: 20, materialId: 2, ownerId: 20, center_m: { x: 0, y: 0, z: 0 }, radius_m: 1 },
    { kind: "sphere", primitiveId: 10, materialId: 3, ownerId: 10, center_m: { x: 0, y: 0, z: 0 }, radius_m: 1 },
    { kind: "box", primitiveId: 30, materialId: 4, ownerId: 30, center_m: { x: 0, y: 0, z: 0 }, halfExtents_m: { x: 1, y: 1, z: 1 }, orientation: identity },
  ];
  const publication = buildSvoPrimitiveCandidates(descriptors);
  const result = traceSvoPrimitiveCandidates(publication, descriptors, {
    origin_m: { x: 0, y: 0, z: 4 }, direction: { x: 0, y: 0, z: -1 },
  });
  assert.equal(result.primitiveIndex, 0);
  assert.equal(result.hit?.ownerId, 20);
  assert.equal(result.candidateIntersections, 3);
});

test("live refit moves bounds without rebuilding BVH topology", () => {
  const rest: SvoFinitePrimitiveDescriptor[] = [
    { kind: "sphere", primitiveId: 1, materialId: 2, ownerId: 1, center_m: { x: 0, y: 0, z: 0 }, radius_m: .5 },
    { kind: "box", primitiveId: 2, materialId: 3, ownerId: 2, center_m: { x: 2, y: 0, z: 0 }, halfExtents_m: { x: .5, y: .5, z: .5 }, orientation: identity },
  ];
  const built = buildSvoPrimitiveCandidates(rest);
  const moved: SvoFinitePrimitiveDescriptor[] = [
    { ...rest[0], center_m: { x: 10, y: 0, z: 0 } },
    rest[1],
  ];
  const refit = refitSvoPrimitiveCandidates(moved, built);
  assert.deepEqual(refit.nodes.map(({ leftOrPrimitiveIndex, rightChildIndex }) => [leftOrPrimitiveIndex, rightChildIndex]),
    built.nodes.map(({ leftOrPrimitiveIndex, rightChildIndex }) => [leftOrPrimitiveIndex, rightChildIndex]));
  assert.ok(!querySvoPrimitiveCandidates(refit, { origin_m: { x: 0, y: 0, z: 2 }, direction: { x: 0, y: 0, z: -1 } }).primitiveIndices.includes(0));
  assert.ok(querySvoPrimitiveCandidates(refit, { origin_m: { x: 10, y: 0, z: 2 }, direction: { x: 0, y: 0, z: -1 } }).primitiveIndices.includes(0));
});

test("incremental live refit touches only a dirty leaf and its ancestor closure", () => {
  const rest: SvoFinitePrimitiveDescriptor[] = Array.from({ length: 8 }, (_, index) => ({
    kind: "sphere", primitiveId: index + 1, materialId: 2, ownerId: index,
    center_m: { x: index * 2, y: 0, z: 0 }, radius_m: .5,
  }));
  const built = buildSvoPrimitiveCandidates(rest);
  const plan = createSvoPrimitiveCandidateRefitPlan(built);
  const before = new Uint32Array(plan.publication.packedRecords);
  const moved = [...rest];
  moved[5] = { ...rest[5], center_m: { x: 10, y: 3, z: 0 } };
  const incremental = refitSvoPrimitiveCandidatesIncremental(moved, plan, [5]);
  const full = refitSvoPrimitiveCandidates(moved, built);
  assert.deepEqual(incremental.publication.packedRecords, full.packedRecords);
  assert.ok(incremental.dirtyNodeIndices.length <= Math.ceil(Math.log2(rest.length)) + 1);
  const dirty = new Set(incremental.dirtyNodeIndices);
  for (let nodeIndex = 0; nodeIndex < built.nodes.length; nodeIndex += 1) {
    if (dirty.has(nodeIndex)) continue;
    const start = nodeIndex * 16;
    assert.deepEqual(incremental.publication.packedRecords.slice(start, start + 16), before.slice(start, start + 16));
  }
});

test("balanced BVH work is bounded and empty rays perform zero exact intersections", () => {
  // A catalog filled exactly to capacity, so the balanced tree it produces is
  // the widest one the shader ever traverses. Derived from the cap rather than
  // hardcoded: the two have to move together or this stops testing the bound.
  const descriptors: SvoFinitePrimitiveDescriptor[] = Array.from({ length: SVO_PRIMITIVE_CANDIDATE_MAXIMUM_LEAVES }, (_, index) => ({
    kind: index % 2 === 0 ? "sphere" : "box",
    primitiveId: index + 1,
    materialId: 2,
    ownerId: index,
    center_m: { x: index * 3, y: 0, z: 0 },
    ...(index % 2 === 0
      ? { radius_m: .5 }
      : { halfExtents_m: { x: .5, y: .01, z: .5 }, orientation: identity }),
  })) as SvoFinitePrimitiveDescriptor[];
  const publication = buildSvoPrimitiveCandidates(descriptors);
  assert.equal(publication.nodes.length, SVO_PRIMITIVE_CANDIDATE_MAXIMUM_NODES);
  const result = traceSvoPrimitiveCandidates(publication, descriptors, {
    origin_m: { x: 0, y: 100, z: 0 }, direction: { x: 1, y: 0, z: 0 },
  });
  assert.equal(result.hit, null);
  assert.equal(result.candidateIntersections, 0);
  assert.ok(result.nodeVisits <= SVO_PRIMITIVE_CANDIDATE_MAXIMUM_NODES);
  assert.ok(querySvoPrimitiveCandidates(publication, {
    origin_m: { x: -2, y: 0, z: 0 }, direction: { x: 1, y: 0, z: 0 },
  }).maximumStackDepth <= SVO_PRIMITIVE_CANDIDATE_MAXIMUM_STACK);
  assert.throws(() => buildSvoPrimitiveCandidates([...descriptors, descriptors[0]]),
    new RegExp(`1-${SVO_PRIMITIVE_CANDIDATE_MAXIMUM_LEAVES} primitives`));
});

test("packed candidate records preserve the shared stride and fail closed on descriptor mismatch", () => {
  const descriptors: SvoFinitePrimitiveDescriptor[] = [
    { kind: "box", primitiveId: 1, materialId: 2, ownerId: 1, center_m: { x: 0, y: 0, z: 0 }, halfExtents_m: { x: 1, y: .01, z: 1 }, orientation: identity },
  ];
  const publication = buildSvoPrimitiveCandidates(descriptors);
  assert.equal(publication.packedRecords.byteLength, 64);
  assert.equal(publication.packedRecords[7], SVO_PRIMITIVE_CANDIDATE_LEAF_SENTINEL);
  assert.equal(publication.packedRecords[3], 0);
  assert.match(publication.cacheKey, /^svo-primitive-candidates-v1:/);
  const arena = packSvoPrimitiveCandidateArena(packSvoPrimitiveRecords(descriptors), publication);
  assert.equal(arena.primitiveCount, 1);
  assert.equal(arena.candidateRecordOffset, 1);
  assert.equal(arena.candidateNodeCount, 1);
  assert.equal(arena.packedRecords.byteLength, 128);
  const tamperedRecords = Uint32Array.from(publication.packedRecords);
  tamperedRecords[0] ^= 1;
  assert.throws(() => packSvoPrimitiveCandidateArena(packSvoPrimitiveRecords(descriptors), {
    ...publication,
    packedRecords: tamperedRecords,
  }), /packed records do not match/);
  assert.throws(() => packSvoPrimitiveCandidateArena(new Uint32Array(0), publication), /does not match primitive records/);
  assert.throws(() => traceSvoPrimitiveCandidates(publication, [], {
    origin_m: { x: 0, y: 1, z: 0 }, direction: { x: 0, y: -1, z: 0 },
  }), /descriptor count mismatch/);
});
