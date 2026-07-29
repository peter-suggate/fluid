import assert from "node:assert/strict";
import test from "node:test";
import {
  SVO_PRIMITIVE_FEATURES,
  intersectPackedSvoPrimitiveRecords,
  intersectSvoPrimitive,
  intersectSvoPrimitives,
  packSvoPrimitiveRecords,
  sampleSvoPrimitive,
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
    {
      // Off the axis, so the ray meets the tube rather than the hole.
      primitive: { ...identity(106, 33), kind: "torus", center_m: { x: 0, y: 0, z: 0 }, majorRadius_m: 1, minorRadius_m: 0.3 },
      origin_m: { x: -3, y: 0.1, z: 0 },
    },
    {
      primitive: { ...identity(107, 34), kind: "cone", center_m: { x: 0, y: 0, z: 0 }, baseRadius_m: 1, topRadius_m: 0.4, halfHeight_m: 0.9 },
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

test("marched kinds land on their own exact surface with gradient-consistent normals", () => {
  const torus: SvoFinitePrimitiveDescriptor = {
    ...identity(201, 33), kind: "torus", center_m: { x: 0.4, y: -0.2, z: 0.1 }, majorRadius_m: 0.9, minorRadius_m: 0.22,
    orientation: { w: Math.cos(0.3), x: Math.sin(0.3), y: 0, z: 0 },
  };
  const cone: SvoFinitePrimitiveDescriptor = {
    ...identity(202, 34), kind: "cone", center_m: { x: -0.3, y: 0.5, z: 0.2 },
    baseRadius_m: 0.7, topRadius_m: 0.25, halfHeight_m: 0.8,
    orientation: { w: Math.cos(0.2), x: 0, y: 0, z: Math.sin(0.2) },
  };
  for (const primitive of [torus, cone]) {
    let hits = 0;
    for (let index = 0; index < 512; index += 1) {
      // Deterministic directions over the sphere; the origin sits well outside.
      const u = ((Math.imul(index + 1, 2246822519) >>> 8) & 0xffff) / 0xffff;
      const v = ((Math.imul(index + 7, 3266489917) >>> 8) & 0xffff) / 0xffff;
      const theta = 2 * Math.PI * u, phi = Math.acos(2 * v - 1);
      const direction = { x: Math.sin(phi) * Math.cos(theta), y: Math.cos(phi), z: Math.sin(phi) * Math.sin(theta) };
      const distance = 2.5;
      const origin_m = {
        x: primitive.center_m.x - direction.x * distance,
        y: primitive.center_m.y - direction.y * distance,
        z: primitive.center_m.z - direction.z * distance,
      };
      const hit = intersectSvoPrimitive(primitive, { origin_m, direction, tMin_m: 0, tMax_m: 2 * distance });
      if (!hit) continue;
      hits += 1;
      // The sphere trace accepts inside a band proportional to the distance
      // travelled, which is a constant angular error rather than a fixed one.
      const band = 2e-4 * Math.max(1, hit.t_m);
      close(sampleSvoPrimitive(primitive, hit.position_m).signedDistance_m, 0, band);
      close(Math.hypot(hit.normal.x, hit.normal.y, hit.normal.z), 1, 1e-9);
      const step = 1e-5;
      const gradient = (["x", "y", "z"] as const).map((axis) => {
        const ahead = { ...hit.position_m, [axis]: hit.position_m[axis] + step };
        const behind = { ...hit.position_m, [axis]: hit.position_m[axis] - step };
        return (sampleSvoPrimitive(primitive, ahead).signedDistance_m - sampleSvoPrimitive(primitive, behind).signedDistance_m) / (2 * step);
      });
      const length = Math.hypot(...gradient);
      if (hit.normalPolicy !== "smooth" || !(length > 1e-6)) continue;
      const alignment = (gradient[0] * hit.normal.x + gradient[1] * hit.normal.y + gradient[2] * hit.normal.z) / length;
      close(alignment, 1, 1e-6);
    }
    assert.ok(hits > 64, `${primitive.kind} should be hit by most rays aimed through it, got ${hits}`);
  }

  // The hole is the whole point of a ring: a ray down its axis finds nothing.
  const axisRay = {
    origin_m: { x: 0, y: 3, z: 0 }, direction: { x: 0, y: -1, z: 0 }, tMin_m: 0, tMax_m: 6,
  };
  const flat: SvoFinitePrimitiveDescriptor = {
    ...identity(203, 33), kind: "torus", center_m: { x: 0, y: 0, z: 0 }, majorRadius_m: 1, minorRadius_m: 0.25,
  };
  assert.equal(intersectSvoPrimitive(flat, axisRay), null);
  const throughTube = expectHit(intersectSvoPrimitive(flat, { ...axisRay, origin_m: { x: 1, y: 3, z: 0 } }));
  close(throughTube.t_m, 2.75, 4e-4);
  closeVector(throughTube.normal, { x: 0, y: 1, z: 0 }, 1e-3);
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
