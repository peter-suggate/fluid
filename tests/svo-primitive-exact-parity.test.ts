import assert from "node:assert/strict";
import test from "node:test";

import { sampleSvoImplicit } from "../lib/svo-implicit-reference";
import {
  SVO_ELLIPSOID_CLOSEST_POINT_ITERATIONS,
  SVO_PRIMITIVE_FLAGS,
  SVO_PRIMITIVE_RECORD_WORDS,
  intersectSvoPrimitive,
  packSvoPrimitiveRecords,
  sampleSvoPrimitive,
  svoPrimitiveWGSL,
  type SvoEllipsoidPrimitive,
} from "../lib/svo-primitive-abi";

const close = (actual: number, expected: number, tolerance = 1e-9): void => {
  assert.ok(Math.abs(actual - expected) <= tolerance, `${actual} should be within ${tolerance} of ${expected}`);
};

const closeVector = (
  actual: { x: number; y: number; z: number } | null,
  expected: { x: number; y: number; z: number } | null,
  tolerance = 1e-9,
): void => {
  assert.equal(actual === null, expected === null);
  if (!actual || !expected) return;
  close(actual.x, expected.x, tolerance);
  close(actual.y, expected.y, tolerance);
  close(actual.z, expected.z, tolerance);
};

const ellipsoid: SvoEllipsoidPrimitive = {
  kind: "ellipsoid",
  primitiveId: 77,
  materialId: 32,
  ownerId: 9,
  center_m: { x: 0.3, y: -0.2, z: 0.4 },
  radii_m: { x: 2, y: 1, z: 0.5 },
  orientation: { w: Math.cos(0.35), x: 0, y: Math.sin(0.35), z: 0 },
};

test("bounded ellipsoid closest points match the exact reference off axis, inside, and after rotation", () => {
  const reference = {
    kind: "ellipsoid" as const,
    center_m: ellipsoid.center_m,
    radii_m: ellipsoid.radii_m,
    orientation: ellipsoid.orientation,
  };
  const points = [
    { x: 2.7, y: 1.1, z: -0.4 },
    { x: 0.7, y: 0.1, z: 0.2 },
    { x: -1.2, y: -0.8, z: 1.4 },
    { x: 0.3, y: -0.2, z: 0.4 },
  ];
  for (const point of points) {
    const sample = sampleSvoPrimitive(ellipsoid, point);
    const exact = sampleSvoImplicit(reference, point);
    close(sample.signedDistance_m, exact.signedDistance_m, 2e-12);
    closeVector(sample.normal, exact.normal, 2e-12);
  }
  const centre = sampleSvoPrimitive(ellipsoid, ellipsoid.center_m);
  close(centre.signedDistance_m, -0.5);
  assert.equal(centre.normal, null, "the interior medial axis must not invent a smooth normal");
});

test("ellipsoid analytic rays preserve tangent contact, inside exits, and grazing misses", () => {
  const axisAligned: SvoEllipsoidPrimitive = {
    ...ellipsoid,
    center_m: { x: 0, y: 0, z: 0 },
    orientation: { w: 1, x: 0, y: 0, z: 0 },
  };
  const tangent = intersectSvoPrimitive(axisAligned, {
    origin_m: { x: -3, y: 1, z: 0 }, direction: { x: 7, y: 0, z: 0 },
  });
  assert.ok(tangent);
  close(tangent.t_m, 3);
  closeVector(tangent.normal, { x: 0, y: 1, z: 0 });
  assert.equal(intersectSvoPrimitive(axisAligned, {
    origin_m: { x: -3, y: 1.0001, z: 0 }, direction: { x: 1, y: 0, z: 0 },
  }), null);
  const inside = intersectSvoPrimitive(axisAligned, {
    origin_m: { x: 0, y: 0, z: 0 }, direction: { x: 0, y: 0, z: -4 },
  });
  assert.ok(inside);
  close(inside.t_m, 0.5);
  closeVector(inside.normal, { x: 0, y: 0, z: -1 });
});

test("the ABI advertises exact ellipsoid distance and WGSL uses only bounded closest-point work", () => {
  const packed = packSvoPrimitiveRecords([ellipsoid]);
  assert.equal(packed[SVO_PRIMITIVE_RECORD_WORDS - 2], SVO_PRIMITIVE_FLAGS.exactDistance);
  assert.equal(SVO_ELLIPSOID_CLOSEST_POINT_ITERATIONS, 64);
  assert.match(svoPrimitiveWGSL, /fn svoEllipsoidClosestPoint_m/);
  assert.match(svoPrimitiveWGSL, new RegExp(
    `iteration < ${SVO_ELLIPSOID_CLOSEST_POINT_ITERATIONS}u`,
  ));
  assert.match(svoPrimitiveWGSL, /if \(any\(radii_m <= vec3f\(0\.0\)\)\) \{ return 3\.402823e38; \}/,
    "malformed GPU dimensions must return the invalid positive sentinel");
  assert.doesNotMatch(svoPrimitiveWGSL, /k0 \* \(k0 - 1\.0\) \/ k1/,
    "the compact sign-correct approximation must not remain in the shared ABI");
  assert.match(svoPrimitiveWGSL, /if \(closest\.w > 0\.5\) \{ return vec4f\(0\.0\); \}/,
    "ambiguous interior normals must remain invalid rather than flickering");
});
