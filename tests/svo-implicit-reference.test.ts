import assert from "node:assert/strict";
import test from "node:test";
import type { RigidBodyDescription, Vec3 } from "../lib/model";
import {
  sampleSvoImplicit,
  svoImplicitReferenceForRigidBody,
  type SvoEllipsoidReference,
  type SvoImplicitSample,
} from "../lib/svo-implicit-reference";
import { terrainHeightAt, terrainNormalAt, type TerrainDescription } from "../lib/terrain";

const close = (actual: number, expected: number, tolerance = 1e-10) => {
  assert.ok(Math.abs(actual - expected) <= tolerance, `${actual} should be within ${tolerance} of ${expected}`);
};

function closeVec(actual: Vec3 | null, expected: Vec3, tolerance = 1e-10): void {
  assert.ok(actual, "expected a well-defined normal");
  close(actual.x, expected.x, tolerance);
  close(actual.y, expected.y, tolerance);
  close(actual.z, expected.z, tolerance);
}

function expectSample(sample: SvoImplicitSample, distance_m: number, normal: Vec3, tolerance = 1e-10): void {
  close(sample.signedDistance_m, distance_m, tolerance);
  closeVec(sample.normal, normal, tolerance);
}

test("sphere reference has metric distance, negative interior, and an undefined centre normal", () => {
  const sphere = { kind: "sphere" as const, center_m: { x: 1, y: 2, z: 3 }, radius_m: 2 };
  expectSample(sampleSvoImplicit(sphere, { x: 4, y: 2, z: 3 }), 1, { x: 1, y: 0, z: 0 });
  expectSample(sampleSvoImplicit(sphere, { x: 1, y: 1, z: 3 }), -1, { x: 0, y: -1, z: 0 });
  const centre = sampleSvoImplicit(sphere, sphere.center_m);
  close(centre.signedDistance_m, -2);
  assert.equal(centre.normal, null);
});

test("oriented box preserves exact faces, edges, corners, and ambiguous medial normals", () => {
  const quarterTurn = { w: Math.SQRT1_2, x: 0, y: 0, z: Math.SQRT1_2 };
  const box = {
    kind: "box" as const, center_m: { x: 0, y: 0, z: 0 }, orientation: quarterTurn,
    halfExtents_m: { x: 1, y: 2, z: 3 },
  };
  expectSample(sampleSvoImplicit(box, { x: 0, y: 2, z: 0 }), 1, { x: 0, y: 1, z: 0 });
  expectSample(sampleSvoImplicit(box, { x: -1.5, y: 0, z: 0 }), -0.5, { x: -1, y: 0, z: 0 });
  close(sampleSvoImplicit(box, { x: -3, y: 2, z: 3 }).signedDistance_m, Math.SQRT2);
  assert.equal(sampleSvoImplicit(box, { x: 0, y: 0, z: 0 }).normal, null);
});

test("capsule uses a Y-axis segment and remains smooth across the cap join", () => {
  const capsule = {
    kind: "capsule" as const, center_m: { x: 0, y: 0, z: 0 },
    radius_m: 1, segmentHalfLength_m: 2,
  };
  expectSample(sampleSvoImplicit(capsule, { x: 0, y: 4, z: 0 }), 1, { x: 0, y: 1, z: 0 });
  expectSample(sampleSvoImplicit(capsule, { x: 0.5, y: 2, z: 0 }), -0.5, { x: 1, y: 0, z: 0 });
  close(sampleSvoImplicit(capsule, { x: 0, y: 0, z: 0 }).signedDistance_m, -1);
  assert.equal(sampleSvoImplicit(capsule, { x: 0, y: 0, z: 0 }).normal, null);
});

test("capped cylinder distinguishes cap, side, and sharp rim normals", () => {
  const cylinder = {
    kind: "cylinder" as const, center_m: { x: 0, y: 0, z: 0 },
    radius_m: 1, halfHeight_m: 2,
  };
  expectSample(sampleSvoImplicit(cylinder, { x: 0, y: 3, z: 0 }), 1, { x: 0, y: 1, z: 0 });
  expectSample(sampleSvoImplicit(cylinder, { x: 2, y: 0, z: 0 }), 1, { x: 1, y: 0, z: 0 });
  const cornerNormal = Math.SQRT1_2;
  expectSample(sampleSvoImplicit(cylinder, { x: 2, y: 3, z: 0 }), Math.SQRT2,
    { x: cornerNormal, y: cornerNormal, z: 0 });
  assert.equal(sampleSvoImplicit(cylinder, { x: 1, y: 2, z: 0 }).normal, null,
    "the authored hard rim must not receive an averaged face normal");
});

test("ellipsoid computes exact Euclidean distance and outward normals", () => {
  const ellipsoid: SvoEllipsoidReference = {
    kind: "ellipsoid", center_m: { x: 0, y: 0, z: 0 }, radii_m: { x: 2, y: 1, z: 0.5 },
  };
  expectSample(sampleSvoImplicit(ellipsoid, { x: 3, y: 0, z: 0 }), 1, { x: 1, y: 0, z: 0 });
  const centre = sampleSvoImplicit(ellipsoid, { x: 0, y: 0, z: 0 });
  close(centre.signedDistance_m, -0.5);
  assert.equal(centre.normal, null, "two closest poles make the centre normal ambiguous");

  const surface = { x: 2 / Math.sqrt(2), y: 1 / Math.sqrt(2), z: 0 };
  const outward = (() => {
    const x = surface.x / 4, y = surface.y;
    const magnitude = Math.hypot(x, y);
    return { x: x / magnitude, y: y / magnitude, z: 0 };
  })();
  const offset_m = 0.37;
  const query = {
    x: surface.x + outward.x * offset_m,
    y: surface.y + outward.y * offset_m,
    z: 0,
  };
  expectSample(sampleSvoImplicit(ellipsoid, query), offset_m, outward, 2e-10);
});

test("terrain heightfield mirrors the shared terrain zero set, sign, and normal", () => {
  const terrain: TerrainDescription = {
    baseHeight_m: 0.3,
    features: [{
      kind: "mound", center_m: { x: 0, z: 0 }, radius_m: { x: 1, z: 0.5 },
      amount_m: 0.2, flat: 0,
    }],
  };
  const x = 0.4, z = 0.1, height = terrainHeightAt(terrain, x, z);
  const reference = { kind: "terrain-heightfield" as const, terrain, normalEpsilon_m: 1e-4 };
  expectSample(sampleSvoImplicit(reference, { x, y: height + 0.25, z }), 0.25,
    terrainNormalAt(terrain, x, z, 1e-4));
  close(sampleSvoImplicit(reference, { x, y: height - 0.1, z }).signedDistance_m, -0.1);
  close(sampleSvoImplicit(reference, { x, y: height, z }).signedDistance_m, 0);
});

test("rigid-body adapter preserves the repository's dimension conventions", () => {
  const base = {
    id: "body", name: "Body", density_kg_m3: 1, position_m: { x: 1, y: 2, z: 3 },
    orientation: { w: 1, x: 0, y: 0, z: 0 }, linearVelocity_m_s: { x: 0, y: 0, z: 0 },
    angularVelocity_rad_s: { x: 0, y: 0, z: 0 }, restitution: 0, friction: 0,
  } satisfies Omit<RigidBodyDescription, "shape" | "dimensions_m">;
  const sphere = svoImplicitReferenceForRigidBody({ ...base, shape: "sphere", dimensions_m: { x: 0.4, y: 9, z: 9 } });
  assert.deepEqual(sphere, { kind: "sphere", center_m: base.position_m, radius_m: 0.4 });
  const box = svoImplicitReferenceForRigidBody({ ...base, shape: "box", dimensions_m: { x: 2, y: 4, z: 6 } });
  assert.equal(box.kind, "box");
  if (box.kind === "box") assert.deepEqual(box.halfExtents_m, { x: 1, y: 2, z: 3 });
  const capsule = svoImplicitReferenceForRigidBody({ ...base, shape: "capsule", dimensions_m: { x: 0.5, y: 4, z: 0.5 } });
  assert.equal(capsule.kind, "capsule");
  if (capsule.kind === "capsule") close(capsule.segmentHalfLength_m, 2);
  const cylinder = svoImplicitReferenceForRigidBody({ ...base, shape: "cylinder", dimensions_m: { x: 0.5, y: 4, z: 0.5 } });
  assert.equal(cylinder.kind, "cylinder");
  if (cylinder.kind === "cylinder") close(cylinder.halfHeight_m, 2);
});

test("implicit references reject degenerate dimensions and orientations", () => {
  assert.throws(() => sampleSvoImplicit({
    kind: "sphere", center_m: { x: 0, y: 0, z: 0 }, radius_m: 0,
  }, { x: 0, y: 0, z: 0 }), /positive/);
  assert.throws(() => sampleSvoImplicit({
    kind: "box", center_m: { x: 0, y: 0, z: 0 }, halfExtents_m: { x: 1, y: 1, z: 1 },
    orientation: { w: 0, x: 0, y: 0, z: 0 },
  }, { x: 0, y: 0, z: 0 }), /nonzero length/);
});
