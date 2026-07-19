import assert from "node:assert/strict";
import test from "node:test";
import {
  SVO_PRIMITIVE_MOTION_FLAGS,
  SVO_PRIMITIVE_MOTION_STRIDE_BYTES,
  SVO_PRIMITIVE_MOTION_WORDS,
  createSvoPrimitiveMotionRecord,
  intersectSvoPrimitiveMotion,
  packSvoPrimitiveMotionRecords,
  svoPrimitiveBoundingRadius,
  svoPrimitiveMotionWGSL,
  svoPrimitiveSurfaceVelocity,
  svoPrimitiveSweptBounds,
  svoPrimitiveTemporalMotionLimit,
  unpackSvoPrimitiveMotionRecords,
  type SvoPrimitiveMotionInput,
} from "../lib/svo-primitive-motion";
import { SVO_PRIMITIVE_FEATURES, type SvoFinitePrimitiveDescriptor } from "../lib/svo-primitive-abi";

const identity = { w: 1, x: 0, y: 0, z: 0 } as const;

function close(actual: number, expected: number, tolerance = 1e-6) {
  assert.ok(Math.abs(actual - expected) <= tolerance, `${actual} should be within ${tolerance} of ${expected}`);
}

function closeVector(actual: readonly number[], expected: readonly number[], tolerance = 1e-6) {
  actual.forEach((value, index) => close(value, expected[index], tolerance));
}

function motion(
  primitive: SvoFinitePrimitiveDescriptor,
  overrides: Partial<SvoPrimitiveMotionInput> = {},
): SvoPrimitiveMotionInput {
  const position = [primitive.center_m.x, primitive.center_m.y, primitive.center_m.z] as const;
  return {
    primitive,
    previous: { position_m: position, orientation: primitive.kind === "sphere" ? identity : primitive.orientation ?? identity, revision: 1, localTopologyGeneration: 9 },
    current: { position_m: position, orientation: primitive.kind === "sphere" ? identity : primitive.orientation ?? identity, revision: 2, localTopologyGeneration: 9 },
    deltaTime_s: 0.5,
    cellSize_m: 1,
    ...overrides,
  };
}

const sphere: SvoFinitePrimitiveDescriptor = {
  kind: "sphere", primitiveId: 11, materialId: 23, ownerId: 41,
  center_m: { x: 1, y: 2, z: 3 }, radius_m: 0.2,
};

test("128-byte sidecar packs normalized current/previous transforms and stable publication identity", () => {
  const input = motion(sphere, {
    previous: { position_m: [0.9, 2, 3], orientation: { w: 2, x: 0, y: 0, z: 0 }, revision: 6, localTopologyGeneration: 12 },
    current: { position_m: [1, 2, 3], orientation: { w: 3, x: 0, y: 0, z: 0 }, revision: 7, localTopologyGeneration: 12 },
    deltaTime_s: 0.25,
    cellSize_m: 0.2,
  });
  const packed = packSvoPrimitiveMotionRecords([input]);
  assert.equal(SVO_PRIMITIVE_MOTION_STRIDE_BYTES, 128);
  assert.equal(SVO_PRIMITIVE_MOTION_WORDS, 32);
  assert.equal(packed.byteLength, 128);
  const [record] = unpackSvoPrimitiveMotionRecords(packed);
  closeVector(record.currentPosition_m, [1, 2, 3]);
  closeVector(record.previousPosition_m, [0.9, 2, 3]);
  close(record.currentOrientation.w, 1);
  close(record.previousOrientation.w, 1);
  closeVector(record.linearVelocity_m_s, [0.4, 0, 0]);
  assert.deepEqual({
    primitive: record.primitiveId, material: record.materialId, owner: record.ownerId,
    currentRevision: record.currentRevision, previousRevision: record.previousRevision,
    currentGeneration: record.currentLocalTopologyGeneration, previousGeneration: record.previousLocalTopologyGeneration,
  }, {
    primitive: 11, material: 23, owner: 41,
    currentRevision: 7, previousRevision: 6,
    currentGeneration: 12, previousGeneration: 12,
  });
  close(record.temporalMotionLimit_m, 0.4);
  assert.equal(record.velocityValid, true);
});

test("static and quaternion-sign-wrap publications produce zero valid motion on the shortest arc", () => {
  const staticRecord = createSvoPrimitiveMotionRecord(motion(sphere));
  assert.equal(staticRecord.velocityValid, true);
  assert.ok((staticRecord.flags & SVO_PRIMITIVE_MOTION_FLAGS.staticMotion) !== 0);
  assert.deepEqual(staticRecord.linearVelocity_m_s, [0, 0, 0]);
  assert.deepEqual(staticRecord.angularVelocity_rad_s, [0, 0, 0]);

  const wrapped = createSvoPrimitiveMotionRecord(motion(sphere, {
    previous: { position_m: [1, 2, 3], orientation: { w: -1, x: 0, y: 0, z: 0 }, revision: 0xffff_ffff, localTopologyGeneration: 9 },
    current: { position_m: [1, 2, 3], orientation: { w: 1, x: 0, y: 0, z: 0 }, revision: 0, localTopologyGeneration: 9 },
  }));
  assert.equal(wrapped.velocityValid, true, "uint32 revision wrap remains adjacent");
  assert.ok((wrapped.flags & SVO_PRIMITIVE_MOTION_FLAGS.shortestArcFlip) !== 0);
  close(wrapped.angularDisplacement_rad, 0);
  closeVector(wrapped.angularVelocity_rad_s, [0, 0, 0]);
});

test("world-space surface velocity combines translation with shortest-arc angular motion", () => {
  const angle = Math.PI / 2;
  const rotatingSphere: SvoFinitePrimitiveDescriptor = { ...sphere, center_m: { x: 0, y: 0, z: 0 }, radius_m: 0.1 };
  const record = createSvoPrimitiveMotionRecord(motion(rotatingSphere, {
    previous: { position_m: [-0.1, 0, 0], orientation: identity, revision: 3, localTopologyGeneration: 5 },
    current: {
      position_m: [0, 0, 0],
      orientation: { w: Math.cos(angle / 2), x: 0, y: 0, z: Math.sin(angle / 2) },
      revision: 4,
      localTopologyGeneration: 5,
    },
    deltaTime_s: 1,
    cellSize_m: 1,
  }));
  assert.equal(record.velocityValid, true);
  closeVector(record.linearVelocity_m_s, [0.1, 0, 0]);
  closeVector(record.angularVelocity_rad_s, [0, 0, angle]);
  const velocity = svoPrimitiveSurfaceVelocity(record, [0.1, 0, 0]);
  assert.equal(velocity.valid, true);
  closeVector(velocity.velocity_m_s, [0.1, angle * 0.1, 0]);
});

test("generation, revision, and temporal-limit discontinuities invalidate velocity fail-closed", () => {
  const generation = createSvoPrimitiveMotionRecord(motion(sphere, {
    previous: { position_m: [1, 2, 3], orientation: identity, revision: 1, localTopologyGeneration: 8 },
  }));
  assert.equal(generation.continuityReason, "generation-change");
  assert.equal(generation.velocityValid, false);

  const revision = createSvoPrimitiveMotionRecord(motion(sphere, {
    previous: { position_m: [0.9, 2, 3], orientation: identity, revision: 2, localTopologyGeneration: 9 },
    current: { position_m: [1, 2, 3], orientation: identity, revision: 2, localTopologyGeneration: 9 },
  }));
  assert.equal(revision.continuityReason, "revision-discontinuity");

  const teleport = createSvoPrimitiveMotionRecord(motion(sphere, {
    previous: { position_m: [0, 2, 3], orientation: identity, revision: 1, localTopologyGeneration: 9 },
    cellSize_m: 0.1,
  }));
  assert.equal(svoPrimitiveTemporalMotionLimit(0.1), 0.2);
  assert.equal(teleport.continuityReason, "teleport");
  assert.ok((teleport.flags & SVO_PRIMITIVE_MOTION_FLAGS.teleport) !== 0);
  assert.deepEqual(svoPrimitiveSurfaceVelocity(teleport, [1.2, 2, 3]), { velocity_m_s: [0, 0, 0], valid: false });
});

test("temporal motion limit accepts bounded movement and rejects the next larger displacement", () => {
  const primitive: SvoFinitePrimitiveDescriptor = { ...sphere, radius_m: 0.01, center_m: { x: 0, y: 0, z: 0 } };
  const withTranslation = (distance: number) => createSvoPrimitiveMotionRecord(motion(primitive, {
    previous: { position_m: [-distance, 0, 0], orientation: identity, revision: 1, localTopologyGeneration: 9 },
    current: { position_m: [0, 0, 0], orientation: identity, revision: 2, localTopologyGeneration: 9 },
    cellSize_m: 0.1,
  }));
  assert.equal(withTranslation(0.199).velocityValid, true);
  assert.equal(withTranslation(0.201).continuityReason, "teleport");
  assert.equal(svoPrimitiveTemporalMotionLimit(1), 0.5, "world cap dominates large cells");
});

test("swept bounds conservatively contain every primitive kind across translation and rotation", () => {
  const primitives: SvoFinitePrimitiveDescriptor[] = [
    { kind: "sphere", primitiveId: 1, materialId: 2, center_m: { x: 1, y: 0, z: 0 }, radius_m: 1 },
    { kind: "box", primitiveId: 2, materialId: 2, center_m: { x: 1, y: 0, z: 0 }, halfExtents_m: { x: 1, y: 2, z: 3 } },
    { kind: "capsule", primitiveId: 3, materialId: 2, center_m: { x: 1, y: 0, z: 0 }, radius_m: 0.5, segmentHalfLength_m: 2 },
    { kind: "cylinder", primitiveId: 4, materialId: 2, center_m: { x: 1, y: 0, z: 0 }, radius_m: 2, halfHeight_m: 3 },
    { kind: "ellipsoid", primitiveId: 5, materialId: 2, center_m: { x: 1, y: 0, z: 0 }, radii_m: { x: 1, y: 4, z: 2 } },
  ];
  const radii = [1, Math.sqrt(14), 2.5, Math.sqrt(13), 4];
  primitives.forEach((primitive, index) => {
    close(svoPrimitiveBoundingRadius(primitive), radii[index]);
    const record = createSvoPrimitiveMotionRecord(motion(primitive, {
      previous: { position_m: [-2, 0, 0], orientation: identity, revision: 1, localTopologyGeneration: 9 },
      current: { position_m: [1, 0, 0], orientation: { w: Math.SQRT1_2, x: 0, y: Math.SQRT1_2, z: 0 }, revision: 2, localTopologyGeneration: 9 },
      cellSize_m: 100,
    }));
    const bounds = svoPrimitiveSweptBounds(record);
    closeVector(bounds.minimum, [-2 - radii[index], -radii[index], -radii[index]]);
    closeVector(bounds.maximum, [1 + radii[index], radii[index], radii[index]]);
  });
});

test("exact motion intersections support all finite shapes and preserve sharp box features", () => {
  const primitives: SvoFinitePrimitiveDescriptor[] = [
    { kind: "sphere", primitiveId: 1, materialId: 10, ownerId: 0, center_m: { x: 0, y: 0, z: 0 }, radius_m: 1 },
    { kind: "box", primitiveId: 2, materialId: 11, ownerId: 1, center_m: { x: 0, y: 0, z: 0 }, halfExtents_m: { x: 1, y: 1, z: 1 } },
    { kind: "capsule", primitiveId: 3, materialId: 12, ownerId: 2, center_m: { x: 0, y: 0, z: 0 }, radius_m: 1, segmentHalfLength_m: 1 },
    { kind: "cylinder", primitiveId: 4, materialId: 13, ownerId: 3, center_m: { x: 0, y: 0, z: 0 }, radius_m: 1, halfHeight_m: 1 },
    { kind: "ellipsoid", primitiveId: 5, materialId: 14, ownerId: 4, center_m: { x: 0, y: 0, z: 0 }, radii_m: { x: 1, y: 2, z: 0.5 } },
  ];
  primitives.forEach((primitive) => {
    const record = createSvoPrimitiveMotionRecord(motion(primitive));
    const hit = intersectSvoPrimitiveMotion(primitive, record, {
      origin_m: { x: -4, y: 0, z: 0 }, direction: { x: 7, y: 0, z: 0 }, tMax_m: 8,
    });
    assert.ok(hit, `${primitive.kind} should intersect`);
    assert.equal(hit.primitiveKind, primitive.kind);
    assert.equal(hit.motionValid, true);
    assert.equal(hit.localTopologyGeneration, 9);
    closeVector(hit.surfaceVelocity_m_s, [0, 0, 0]);
  });
  const box = primitives[1];
  const boxRecord = createSvoPrimitiveMotionRecord(motion(box));
  const corner = intersectSvoPrimitiveMotion(box, boxRecord, {
    origin_m: { x: 2, y: 2, z: 2 }, direction: { x: -1, y: -1, z: -1 }, tMax_m: 5,
  });
  assert.ok(corner);
  assert.equal(corner.featureId, SVO_PRIMITIVE_FEATURES.boxFaceX);
  assert.deepEqual(corner.normal, { x: 1, y: 0, z: 0 });
});

test("identity mismatch and malformed publications fail at the adapter boundary", () => {
  const record = createSvoPrimitiveMotionRecord(motion(sphere));
  assert.throws(() => intersectSvoPrimitiveMotion({ ...sphere, primitiveId: 12 }, record, {
    origin_m: { x: -2, y: 2, z: 3 }, direction: { x: 1, y: 0, z: 0 },
  }), /identity/);
  assert.throws(() => createSvoPrimitiveMotionRecord(motion(sphere, { deltaTime_s: 0 })), /delta time/);
  assert.throws(() => unpackSvoPrimitiveMotionRecords(new Uint32Array(31)), /partial record/);
});

test("WGSL sidecar is binding-free and carries shortest arc, exact velocity, bounds, IDs, and generation", () => {
  assert.doesNotMatch(svoPrimitiveMotionWGSL, /@group|@binding/);
  assert.match(svoPrimitiveMotionWGSL, /struct SvoPrimitiveMotionRecord\{currentPositionDt:vec4f,previousPositionRadius:vec4f,currentOrientation:vec4f,previousOrientation:vec4f/);
  assert.match(svoPrimitiveMotionWGSL, /dot\(previous,current\)<0\.0/);
  assert.match(svoPrimitiveMotionWGSL, /cross\(record\.angularVelocityAngle\.xyz,radius\)/);
  assert.match(svoPrimitiveMotionWGSL, /min\(record\.currentPositionDt\.xyz,record\.previousPositionRadius\.xyz\)-radius/);
  assert.match(svoPrimitiveMotionWGSL, /record\.identityRevision\.y&0xffffu/);
  assert.match(svoPrimitiveMotionWGSL, /return record\.publication\.x/);
});
