import assert from "node:assert/strict";
import test from "node:test";
import {
  SVO_PRIMITIVE_FEATURES,
  intersectPackedSvoPrimitiveRecords,
  intersectSvoPrimitive,
  intersectSvoPrimitives,
  packSvoPrimitiveRecords,
  type SvoFinitePrimitiveDescriptor,
  type SvoPrimitiveRayHit,
} from "../lib/svo-primitive-abi";

const close = (actual: number, expected: number, tolerance = 1e-8) => {
  assert.ok(Math.abs(actual - expected) <= tolerance, `${actual} should be within ${tolerance} of ${expected}`);
};

const closeVector = (
  actual: { x: number; y: number; z: number },
  expected: { x: number; y: number; z: number },
  tolerance = 1e-8,
) => {
  close(actual.x, expected.x, tolerance);
  close(actual.y, expected.y, tolerance);
  close(actual.z, expected.z, tolerance);
};

function expectHit(hit: SvoPrimitiveRayHit | null): SvoPrimitiveRayHit {
  assert.ok(hit);
  return hit;
}

const identity = (primitiveId: number, materialId = 16, ownerId = primitiveId) => ({ primitiveId, materialId, ownerId });

test("sphere rays return metre distances for normalized and non-normalized directions", () => {
  const sphere: SvoFinitePrimitiveDescriptor = {
    ...identity(11, 23, 41), kind: "sphere", center_m: { x: 1, y: 0, z: 0 }, radius_m: 2,
  };
  const unit = expectHit(intersectSvoPrimitive(sphere, { origin_m: { x: -4, y: 0, z: 0 }, direction: { x: 1, y: 0, z: 0 } }));
  const scaled = expectHit(intersectSvoPrimitive(sphere, { origin_m: { x: -4, y: 0, z: 0 }, direction: { x: 20, y: 0, z: 0 } }));
  close(unit.t_m, 3);
  close(scaled.t_m, 3);
  closeVector(unit.position_m, { x: -1, y: 0, z: 0 });
  closeVector(unit.normal, { x: -1, y: 0, z: 0 });
  assert.equal(unit.normalPolicy, "smooth");
  assert.equal(unit.featureId, SVO_PRIMITIVE_FEATURES.smooth);
  assert.deepEqual(
    { primitiveId: unit.primitiveId, materialId: unit.materialId, ownerId: unit.ownerId },
    { primitiveId: 11, materialId: 23, ownerId: 41 },
  );
});

test("sphere intersection selects an inside exit and retains exact tangent contact", () => {
  const sphere: SvoFinitePrimitiveDescriptor = {
    ...identity(1), kind: "sphere", center_m: { x: 0, y: 0, z: 0 }, radius_m: 1,
  };
  const inside = expectHit(intersectSvoPrimitive(sphere, { origin_m: { x: 0, y: 0, z: 0 }, direction: { x: 4, y: 0, z: 0 } }));
  close(inside.t_m, 1);
  closeVector(inside.normal, { x: 1, y: 0, z: 0 });

  const tangent = expectHit(intersectSvoPrimitive(sphere, { origin_m: { x: -2, y: 1, z: 0 }, direction: { x: 1, y: 0, z: 0 } }));
  close(tangent.t_m, 2);
  closeVector(tangent.position_m, { x: 0, y: 1, z: 0 });
  closeVector(tangent.normal, { x: 0, y: 1, z: 0 });
  assert.equal(intersectSvoPrimitive(sphere, {
    origin_m: { x: -2, y: 1.0001, z: 0 }, direction: { x: 1, y: 0, z: 0 },
  }), null);
});

test("oriented boxes retain authored hard normals and stable edge/corner ties", () => {
  const rotated: SvoFinitePrimitiveDescriptor = {
    ...identity(2, 17), kind: "box", center_m: { x: 0, y: 0, z: 0 },
    halfExtents_m: { x: 2, y: 1, z: 1 },
    orientation: { w: Math.SQRT1_2, x: 0, y: 0, z: Math.SQRT1_2 },
  };
  const face = expectHit(intersectSvoPrimitive(rotated, { origin_m: { x: 0, y: -4, z: 0 }, direction: { x: 0, y: 3, z: 0 } }));
  close(face.t_m, 2);
  closeVector(face.position_m, { x: 0, y: -2, z: 0 });
  closeVector(face.normal, { x: 0, y: -1, z: 0 });
  assert.equal(face.normalPolicy, "hard-feature");
  assert.equal(face.featureId, SVO_PRIMITIVE_FEATURES.boxFaceX);

  const cube: SvoFinitePrimitiveDescriptor = {
    ...identity(3, 17), kind: "box", center_m: { x: 0, y: 0, z: 0 }, halfExtents_m: { x: 1, y: 1, z: 1 },
  };
  const corner = expectHit(intersectSvoPrimitive(cube, { origin_m: { x: 2, y: 2, z: 2 }, direction: { x: -1, y: -1, z: -1 } }));
  close(corner.t_m, Math.sqrt(3));
  closeVector(corner.position_m, { x: 1, y: 1, z: 1 });
  closeVector(corner.normal, { x: 1, y: 0, z: 0 });
  assert.equal(corner.featureId, SVO_PRIMITIVE_FEATURES.boxFaceX);

  const edge = expectHit(intersectSvoPrimitive(cube, { origin_m: { x: 2, y: 2, z: 0 }, direction: { x: -1, y: -1, z: 0 } }));
  assert.equal(edge.featureId, SVO_PRIMITIVE_FEATURES.boxFaceX);
  closeVector(edge.normal, { x: 1, y: 0, z: 0 });
  const inside = expectHit(intersectSvoPrimitive(cube, { origin_m: { x: 0, y: 0, z: 0 }, direction: { x: 0, y: -1, z: 0 } }));
  close(inside.t_m, 1);
  closeVector(inside.normal, { x: 0, y: -1, z: 0 });
});

test("capsules select barrel and hemispheres smoothly, including grazing contact", () => {
  const capsule: SvoFinitePrimitiveDescriptor = {
    ...identity(4, 18), kind: "capsule", center_m: { x: 0, y: 0, z: 0 }, radius_m: 0.5, segmentHalfLength_m: 1,
  };
  const barrel = expectHit(intersectSvoPrimitive(capsule, { origin_m: { x: -2, y: 0, z: 0 }, direction: { x: 1, y: 0, z: 0 } }));
  close(barrel.t_m, 1.5);
  closeVector(barrel.normal, { x: -1, y: 0, z: 0 });
  const cap = expectHit(intersectSvoPrimitive(capsule, { origin_m: { x: 0, y: 3, z: 0 }, direction: { x: 0, y: -2, z: 0 } }));
  close(cap.t_m, 1.5);
  closeVector(cap.normal, { x: 0, y: 1, z: 0 });
  const tangent = expectHit(intersectSvoPrimitive(capsule, { origin_m: { x: -2, y: 1.5, z: 0 }, direction: { x: 1, y: 0, z: 0 } }));
  close(tangent.t_m, 2);
  closeVector(tangent.normal, { x: 0, y: 1, z: 0 });
  assert.equal(tangent.normalPolicy, "smooth");
});

test("capped cylinders prefer cap normals at rims and support rotated inside exits", () => {
  const cylinder: SvoFinitePrimitiveDescriptor = {
    ...identity(5, 19), kind: "cylinder", center_m: { x: 0, y: 0, z: 0 }, radius_m: 1, halfHeight_m: 2,
  };
  const side = expectHit(intersectSvoPrimitive(cylinder, { origin_m: { x: -3, y: 0, z: 0 }, direction: { x: 1, y: 0, z: 0 } }));
  close(side.t_m, 2);
  closeVector(side.normal, { x: -1, y: 0, z: 0 });
  assert.equal(side.featureId, SVO_PRIMITIVE_FEATURES.cylinderSide);
  const rim = expectHit(intersectSvoPrimitive(cylinder, { origin_m: { x: 1, y: 4, z: 0 }, direction: { x: 0, y: -1, z: 0 } }));
  close(rim.t_m, 2);
  closeVector(rim.normal, { x: 0, y: 1, z: 0 });
  assert.equal(rim.featureId, SVO_PRIMITIVE_FEATURES.cylinderCap);

  const rotated: SvoFinitePrimitiveDescriptor = {
    ...cylinder, primitiveId: 6,
    orientation: { w: Math.SQRT1_2, x: 0, y: 0, z: -Math.SQRT1_2 },
  };
  const cap = expectHit(intersectSvoPrimitive(rotated, { origin_m: { x: 4, y: 0, z: 0 }, direction: { x: -5, y: 0, z: 0 } }));
  close(cap.t_m, 2);
  closeVector(cap.normal, { x: 1, y: 0, z: 0 });
  assert.equal(cap.featureId, SVO_PRIMITIVE_FEATURES.cylinderCap);
  const inside = expectHit(intersectSvoPrimitive(rotated, { origin_m: { x: 0, y: 0, z: 0 }, direction: { x: -1, y: 0, z: 0 } }));
  close(inside.t_m, 2);
  closeVector(inside.normal, { x: -1, y: 0, z: 0 });
});

test("ellipsoid rays solve the exact rotated quadratic and gradient normal", () => {
  const ellipsoid: SvoFinitePrimitiveDescriptor = {
    ...identity(7, 32), kind: "ellipsoid", center_m: { x: 0, y: 0, z: 0 }, radii_m: { x: 2, y: 1, z: 0.5 },
    orientation: { w: Math.SQRT1_2, x: 0, y: 0, z: Math.SQRT1_2 },
  };
  const hit = expectHit(intersectSvoPrimitive(ellipsoid, { origin_m: { x: 0, y: 5, z: 0 }, direction: { x: 0, y: -9, z: 0 } }));
  close(hit.t_m, 3);
  closeVector(hit.position_m, { x: 0, y: 2, z: 0 });
  closeVector(hit.normal, { x: 0, y: 1, z: 0 });
  assert.equal(hit.normalPolicy, "smooth");
  const inside = expectHit(intersectSvoPrimitive(ellipsoid, { origin_m: { x: 0, y: 0, z: 0 }, direction: { x: 0, y: 1, z: 0 } }));
  close(inside.t_m, 2);
});

test("nearest selection is geometric for descriptor and packed-record inputs", () => {
  const farther: SvoFinitePrimitiveDescriptor = {
    ...identity(80, 16, 800), kind: "sphere", center_m: { x: 3, y: 0, z: 0 }, radius_m: 1,
  };
  const nearer: SvoFinitePrimitiveDescriptor = {
    ...identity(70, 17, 700), kind: "box", center_m: { x: 0, y: 0, z: 0 }, halfExtents_m: { x: 1, y: 1, z: 1 },
  };
  const ray = { origin_m: { x: -5, y: 0, z: 0 }, direction: { x: 4, y: 0, z: 0 } };
  const described = expectHit(intersectSvoPrimitives([farther, nearer], ray));
  assert.equal(described.primitiveId, 70);
  close(described.t_m, 4);

  const packed = expectHit(intersectPackedSvoPrimitiveRecords(packSvoPrimitiveRecords([farther, nearer]), ray));
  assert.equal(packed.primitiveId, 70);
  assert.equal(packed.materialId, 17);
  assert.equal(packed.ownerId, 700);
  close(packed.t_m, 4);
});

test("packed and described hit oracles agree for every finite primitive kind", () => {
  const cases: Array<{ primitive: SvoFinitePrimitiveDescriptor; origin_m: { x: number; y: number; z: number } }> = [
    {
      primitive: { ...identity(101, 16), kind: "sphere", center_m: { x: 0, y: 0, z: 0 }, radius_m: 1 },
      origin_m: { x: -3, y: 0.2, z: 0 },
    },
    {
      primitive: {
        ...identity(102, 17), kind: "box", center_m: { x: 0, y: 0, z: 0 }, halfExtents_m: { x: 1, y: 0.8, z: 0.6 },
        orientation: { w: Math.cos(0.2), x: 0, y: Math.sin(0.2), z: 0 },
      },
      origin_m: { x: -3, y: 0.1, z: 0 },
    },
    {
      primitive: { ...identity(103, 18), kind: "capsule", center_m: { x: 0, y: 0, z: 0 }, radius_m: 0.5, segmentHalfLength_m: 1 },
      origin_m: { x: -3, y: 0.3, z: 0 },
    },
    {
      primitive: { ...identity(104, 19), kind: "cylinder", center_m: { x: 0, y: 0, z: 0 }, radius_m: 0.75, halfHeight_m: 1 },
      origin_m: { x: -3, y: 0.3, z: 0 },
    },
    {
      primitive: { ...identity(105, 32), kind: "ellipsoid", center_m: { x: 0, y: 0, z: 0 }, radii_m: { x: 1.5, y: 0.8, z: 0.4 } },
      origin_m: { x: -3, y: 0.2, z: 0 },
    },
  ];
  for (const { primitive, origin_m } of cases) {
    const ray = { origin_m, direction: { x: 7, y: 0, z: 0 } };
    const described = expectHit(intersectSvoPrimitive(primitive, ray));
    const packed = expectHit(intersectPackedSvoPrimitiveRecords(packSvoPrimitiveRecords([primitive]), ray));
    close(packed.t_m, described.t_m, 2e-6);
    closeVector(packed.position_m, described.position_m, 2e-6);
    closeVector(packed.normal, described.normal, 2e-6);
    assert.equal(packed.featureId, described.featureId);
    assert.equal(packed.primitiveKind, primitive.kind);
    assert.equal(packed.primitiveId, primitive.primitiveId);
  }
});

test("ray intervals, invalid rays, and separate terrain handling fail deterministically", () => {
  const sphere: SvoFinitePrimitiveDescriptor = {
    ...identity(9), kind: "sphere", center_m: { x: 0, y: 0, z: 0 }, radius_m: 1,
  };
  const ray = { origin_m: { x: -3, y: 0, z: 0 }, direction: { x: 1, y: 0, z: 0 }, tMin_m: 2.5 };
  const farSurface = expectHit(intersectSvoPrimitive(sphere, ray));
  close(farSurface.t_m, 4);
  assert.equal(intersectSvoPrimitive(sphere, { ...ray, tMax_m: 3.5 }), null);
  assert.throws(() => intersectSvoPrimitive(sphere, { origin_m: { x: 0, y: 0, z: 0 }, direction: { x: 0, y: 0, z: 0 } }), /non-zero/);
  assert.throws(() => intersectSvoPrimitive(sphere, {
    origin_m: { x: 0, y: 0, z: 0 }, direction: { x: 1, y: 0, z: 0 }, tMin_m: -1,
  }), /non-negative/);

  const terrain = { kind: "terrain-heightfield" as const, primitiveId: 10, materialId: 2, terrainReference: 3 };
  assert.equal(intersectSvoPrimitives([terrain], ray), null, "heightfields remain owned by the separate terrain tracer");
});
