import assert from "node:assert/strict";
import test from "node:test";
import { sampleSvoImplicit, type SvoImplicitReference } from "../lib/svo-implicit-reference";
import { VOXEL_MATERIAL_IDS } from "../lib/voxel-scene";
import {
  SVO_PRIMITIVE_FEATURES,
  SVO_PRIMITIVE_FLAGS,
  SVO_PRIMITIVE_INVALID_REFERENCE,
  SVO_PRIMITIVE_KINDS,
  SVO_PRIMITIVE_RECORD_STRIDE_BYTES,
  SVO_PRIMITIVE_RECORD_WORDS,
  canonicalSvoPrimitive,
  packSvoPrimitiveRecords,
  sampleSvoPrimitive,
  svoPrimitiveForRigidBody,
  svoPrimitiveWGSL,
  unpackSvoPrimitiveRecords,
  type SvoPrimitiveDescriptor,
} from "../lib/svo-primitive-abi";

const close = (actual: number, expected: number, tolerance = 1e-6) => {
  assert.ok(Math.abs(actual - expected) <= tolerance, `${actual} should be within ${tolerance} of ${expected}`);
};

const closeVector = (
  actual: { x: number; y: number; z: number } | null,
  expected: { x: number; y: number; z: number },
  tolerance = 1e-6,
) => {
  assert.ok(actual);
  close(actual.x, expected.x, tolerance);
  close(actual.y, expected.y, tolerance);
  close(actual.z, expected.z, tolerance);
};

const descriptors: SvoPrimitiveDescriptor[] = [
  { kind: "sphere", primitiveId: 101, materialId: 16, ownerId: 1, center_m: { x: 1, y: 2, z: 3 }, radius_m: 0.75 },
  {
    kind: "box", primitiveId: 102, materialId: 17, ownerId: 2, center_m: { x: -1, y: -2, z: -3 },
    halfExtents_m: { x: 1, y: 2, z: 3 }, orientation: { w: 2, x: 0, y: 0, z: 2 },
  },
  {
    kind: "capsule", primitiveId: 103, materialId: 18, ownerId: 3, center_m: { x: 0, y: 1, z: 0 },
    radius_m: 0.4, segmentHalfLength_m: 1.2,
  },
  {
    kind: "cylinder", primitiveId: 104, materialId: 19, ownerId: 4, center_m: { x: 0, y: 0, z: 1 },
    radius_m: 0.5, halfHeight_m: 1.5, orientation: { w: 1, x: 0, y: 0, z: 0 },
  },
  {
    kind: "ellipsoid", primitiveId: 105, materialId: 32, ownerId: 5, center_m: { x: 2, y: 0, z: -1 },
    radii_m: { x: 2, y: 1, z: 0.5 },
  },
  { kind: "terrain-heightfield", primitiveId: 106, materialId: 2, terrainReference: 7, normalEpsilon_m: 0.025 },
];

test("primitive ABI uses four aligned lanes and stable kind values", () => {
  assert.equal(SVO_PRIMITIVE_RECORD_STRIDE_BYTES, 64);
  assert.equal(SVO_PRIMITIVE_RECORD_WORDS, 16);
  assert.deepEqual(SVO_PRIMITIVE_KINDS, {
    sphere: 1, box: 2, capsule: 3, cylinder: 4, ellipsoid: 5, terrainHeightfield: 6,
  });

  const packed = packSvoPrimitiveRecords(descriptors);
  const floats = new Float32Array(packed.buffer);
  assert.equal(packed.byteLength, descriptors.length * SVO_PRIMITIVE_RECORD_STRIDE_BYTES);
  assert.deepEqual([...floats.slice(0, 3)], [1, 2, 3]);
  assert.equal(packed[3], SVO_PRIMITIVE_KINDS.sphere);
  assert.deepEqual([...floats.slice(4, 7)], [0.75, 0, 0]);
  assert.equal(packed[7], (1 << 16) | 16);
  assert.deepEqual([...floats.slice(8, 12)], [0, 0, 0, 1], "sphere orientation is canonical identity xyzw");
  assert.deepEqual([...packed.slice(12, 16)], [101, SVO_PRIMITIVE_INVALID_REFERENCE, SVO_PRIMITIVE_FLAGS.exactDistance, 0]);

  const box = SVO_PRIMITIVE_RECORD_WORDS;
  assert.equal(packed[box + 3], SVO_PRIMITIVE_KINDS.box);
  close(floats[box + 10], Math.SQRT1_2);
  close(floats[box + 11], Math.SQRT1_2);
  assert.equal(packed[box + 14], SVO_PRIMITIVE_FLAGS.exactDistance | SVO_PRIMITIVE_FLAGS.hardFeatures);

  const terrain = 5 * SVO_PRIMITIVE_RECORD_WORDS;
  assert.equal(packed[terrain + 3], SVO_PRIMITIVE_KINDS.terrainHeightfield);
  assert.equal(packed[terrain + 13], 7);
  assert.equal(packed[terrain + 14], SVO_PRIMITIVE_FLAGS.externalTerrain);
});

test("packing is deterministic and unpacking preserves normalized metre records", () => {
  const first = packSvoPrimitiveRecords(descriptors);
  const second = packSvoPrimitiveRecords(descriptors.map((descriptor) => ({ ...descriptor })));
  assert.deepEqual(first, second);
  const unpacked = unpackSvoPrimitiveRecords(first);
  assert.equal(unpacked.length, descriptors.length);

  const sphere = unpacked[0];
  assert.equal(sphere.kind, "sphere");
  if (sphere.kind !== "sphere") return;
  close(sphere.radius_m, 0.75);
  assert.deepEqual(sphere.center_m, { x: 1, y: 2, z: 3 });
  assert.equal(sphere.primitiveId, 101);
  assert.equal(sphere.materialId, 16);
  assert.equal(sphere.ownerId, 1);

  const box = unpacked[1];
  assert.equal(box.kind, "box");
  if (box.kind !== "box") return;
  close(box.orientation?.w ?? 0, Math.SQRT1_2);
  close(box.orientation?.z ?? 0, Math.SQRT1_2);
  assert.deepEqual(box.halfExtents_m, { x: 1, y: 2, z: 3 });

  const terrain = unpacked[5];
  assert.equal(terrain.kind, "terrain-heightfield");
  if (terrain.kind !== "terrain-heightfield") return;
  assert.equal(terrain.terrainReference, 7);
  close(terrain.normalEpsilon_m ?? 0, 0.025);
  assert.equal(terrain.ownerId, 0xffff);

  assert.throws(() => unpackSvoPrimitiveRecords(new Uint32Array(15)), /partial record/);
  const unknown = new Uint32Array(first.slice(0, SVO_PRIMITIVE_RECORD_WORDS));
  unknown[3] = 99;
  assert.throws(() => unpackSvoPrimitiveRecords(unknown), /Unknown SVO primitive kind 99/);
});

test("primitive validation rejects lossy identity, invalid geometry, and degenerate orientations", () => {
  assert.throws(() => canonicalSvoPrimitive({
    kind: "sphere", primitiveId: -1, materialId: 16, center_m: { x: 0, y: 0, z: 0 }, radius_m: 1,
  }), /Primitive ID/);
  assert.throws(() => canonicalSvoPrimitive({
    kind: "sphere", primitiveId: 1, materialId: 0, center_m: { x: 0, y: 0, z: 0 }, radius_m: 1,
  }), /nonzero uint16/);
  assert.throws(() => canonicalSvoPrimitive({
    kind: "sphere", primitiveId: 1, materialId: 16, ownerId: 0x1_0000, center_m: { x: 0, y: 0, z: 0 }, radius_m: 1,
  }), /owner ID/i);
  assert.throws(() => canonicalSvoPrimitive({
    kind: "box", primitiveId: 1, materialId: 17, center_m: { x: 0, y: 0, z: 0 },
    halfExtents_m: { x: 1, y: 0, z: 1 }, orientation: { w: 1, x: 0, y: 0, z: 0 },
  }), /positive/);
  assert.throws(() => canonicalSvoPrimitive({
    kind: "cylinder", primitiveId: 1, materialId: 19, center_m: { x: 0, y: 0, z: 0 }, radius_m: 1, halfHeight_m: 1,
    orientation: { w: 0, x: 0, y: 0, z: 0 },
  }), /nonzero length/);
  assert.throws(() => canonicalSvoPrimitive({
    kind: "terrain-heightfield", primitiveId: 1, materialId: 2, terrainReference: SVO_PRIMITIVE_INVALID_REFERENCE,
  }), /invalid sentinel/);
});

test("rigid-body adapter preserves the repository's dimension semantics", () => {
  const base = {
    position_m: { x: 1, y: 2, z: 3 }, orientation: { w: 1, x: 0, y: 0, z: 0 },
  };
  const sphere = svoPrimitiveForRigidBody({ ...base, shape: "sphere", dimensions_m: { x: 0.6, y: 0.6, z: 0.6 } }, 10, 2);
  assert.equal(sphere.kind, "sphere");
  assert.equal(sphere.radius_m, 0.6);
  assert.equal(sphere.materialId, VOXEL_MATERIAL_IDS.sphere);

  const box = svoPrimitiveForRigidBody({ ...base, shape: "box", dimensions_m: { x: 2, y: 4, z: 6 } }, 11, 3);
  assert.equal(box.kind, "box");
  if (box.kind === "box") assert.deepEqual(box.halfExtents_m, { x: 1, y: 2, z: 3 });

  const capsule = svoPrimitiveForRigidBody({ ...base, shape: "capsule", dimensions_m: { x: 0.4, y: 2.4, z: 0.4 } }, 12, 4);
  assert.equal(capsule.kind, "capsule");
  if (capsule.kind === "capsule") {
    assert.equal(capsule.radius_m, 0.4);
    assert.equal(capsule.segmentHalfLength_m, 1.2);
  }

  const cylinder = svoPrimitiveForRigidBody({ ...base, shape: "cylinder", dimensions_m: { x: 0.5, y: 3, z: 0.5 } }, 13, 5);
  assert.equal(cylinder.kind, "cylinder");
  if (cylinder.kind === "cylinder") assert.equal(cylinder.halfHeight_m, 1.5);
});

test("CPU evaluation keeps spheres smooth and selects one hard feature for boxes", () => {
  const sphere: SvoPrimitiveDescriptor = {
    kind: "sphere", primitiveId: 1, materialId: 16, center_m: { x: 0, y: 0, z: 0 }, radius_m: 2,
  };
  const sphereSample = sampleSvoPrimitive(sphere, { x: Math.SQRT2, y: Math.SQRT2, z: 0 });
  close(sphereSample.signedDistance_m, 0);
  closeVector(sphereSample.normal, { x: Math.SQRT1_2, y: Math.SQRT1_2, z: 0 });
  assert.equal(sphereSample.featureId, SVO_PRIMITIVE_FEATURES.smooth);

  const box: SvoPrimitiveDescriptor = {
    kind: "box", primitiveId: 2, materialId: 17, center_m: { x: 0, y: 0, z: 0 },
    halfExtents_m: { x: 1, y: 1, z: 1 },
  };
  const face = sampleSvoPrimitive(box, { x: 1, y: 0.8, z: 0 });
  close(face.signedDistance_m, 0);
  closeVector(face.normal, { x: 1, y: 0, z: 0 });
  assert.equal(face.featureId, SVO_PRIMITIVE_FEATURES.boxFaceX);

  const edge = sampleSvoPrimitive(box, { x: 1, y: 1, z: 0 });
  close(edge.signedDistance_m, 0);
  closeVector(edge.normal, { x: 1, y: 0, z: 0 });
  assert.equal(edge.featureId, SVO_PRIMITIVE_FEATURES.boxFaceX, "the stable tie does not average adjacent face normals");

  const rotated: SvoPrimitiveDescriptor = {
    ...box, primitiveId: 3, orientation: { w: Math.SQRT1_2, x: 0, y: 0, z: Math.SQRT1_2 },
  };
  const rotatedFace = sampleSvoPrimitive(rotated, { x: 0, y: 1, z: 0 });
  closeVector(rotatedFace.normal, { x: 0, y: 1, z: 0 });
});

test("capsule, capped-cylinder, ellipsoid, and terrain evaluations preserve zero sets and features", () => {
  const capsule: SvoPrimitiveDescriptor = {
    kind: "capsule", primitiveId: 1, materialId: 18, center_m: { x: 0, y: 0, z: 0 }, radius_m: 0.5, segmentHalfLength_m: 1,
  };
  close(sampleSvoPrimitive(capsule, { x: 0.5, y: 0, z: 0 }).signedDistance_m, 0);
  close(sampleSvoPrimitive(capsule, { x: 0, y: 1.5, z: 0 }).signedDistance_m, 0);

  const cylinder: SvoPrimitiveDescriptor = {
    kind: "cylinder", primitiveId: 2, materialId: 19, center_m: { x: 0, y: 0, z: 0 }, radius_m: 1, halfHeight_m: 2,
  };
  const side = sampleSvoPrimitive(cylinder, { x: 1, y: 0, z: 0 });
  closeVector(side.normal, { x: 1, y: 0, z: 0 });
  assert.equal(side.featureId, SVO_PRIMITIVE_FEATURES.cylinderSide);
  const cap = sampleSvoPrimitive(cylinder, { x: 0.5, y: 2, z: 0 });
  closeVector(cap.normal, { x: 0, y: 1, z: 0 });
  assert.equal(cap.featureId, SVO_PRIMITIVE_FEATURES.cylinderCap);
  const rim = sampleSvoPrimitive(cylinder, { x: 1, y: 2, z: 0 });
  closeVector(rim.normal, { x: 0, y: 1, z: 0 });
  assert.equal(rim.featureId, SVO_PRIMITIVE_FEATURES.cylinderCap, "rim tie selects one feature instead of averaging");

  const ellipsoid: SvoPrimitiveDescriptor = {
    kind: "ellipsoid", primitiveId: 3, materialId: 32, center_m: { x: 0, y: 0, z: 0 }, radii_m: { x: 2, y: 1, z: 0.5 },
  };
  close(sampleSvoPrimitive(ellipsoid, { x: 2, y: 0, z: 0 }).signedDistance_m, 0);
  closeVector(sampleSvoPrimitive(ellipsoid, { x: 2, y: 0, z: 0 }).normal, { x: 1, y: 0, z: 0 });

  const terrain: SvoPrimitiveDescriptor = {
    kind: "terrain-heightfield", primitiveId: 4, materialId: 2, terrainReference: 9, normalEpsilon_m: 0.01,
  };
  const terrainDescription = { baseHeight_m: 0.4, features: [] };
  const terrainSample = sampleSvoPrimitive(terrain, { x: 1, y: 0.4, z: -1 }, (reference) => {
    assert.equal(reference, 9);
    return terrainDescription;
  });
  close(terrainSample.signedDistance_m, 0);
  closeVector(terrainSample.normal, { x: 0, y: 1, z: 0 });
  assert.equal(terrainSample.featureId, SVO_PRIMITIVE_FEATURES.terrain);
  assert.throws(() => sampleSvoPrimitive(terrain, { x: 0, y: 0, z: 0 }), /terrain resolver/);
});

test("packed-record CPU evaluation agrees numerically with the canonical implicit reference", () => {
  const cases: Array<{
    descriptor: Exclude<SvoPrimitiveDescriptor, { kind: "terrain-heightfield" }>;
    reference: Exclude<SvoImplicitReference, { kind: "terrain-heightfield" }>;
    point: { x: number; y: number; z: number };
  }> = [
    {
      descriptor: { kind: "sphere", primitiveId: 1, materialId: 16, center_m: { x: 1, y: 0, z: 0 }, radius_m: 2 },
      reference: { kind: "sphere", center_m: { x: 1, y: 0, z: 0 }, radius_m: 2 },
      point: { x: 4, y: 4, z: 0 },
    },
    {
      descriptor: { kind: "box", primitiveId: 2, materialId: 17, center_m: { x: 0, y: 0, z: 0 }, halfExtents_m: { x: 1, y: 2, z: 3 } },
      reference: { kind: "box", center_m: { x: 0, y: 0, z: 0 }, halfExtents_m: { x: 1, y: 2, z: 3 } },
      point: { x: 2, y: 3, z: 3 },
    },
    {
      descriptor: { kind: "capsule", primitiveId: 3, materialId: 18, center_m: { x: 0, y: 0, z: 0 }, radius_m: 0.5, segmentHalfLength_m: 1 },
      reference: { kind: "capsule", center_m: { x: 0, y: 0, z: 0 }, radius_m: 0.5, segmentHalfLength_m: 1 },
      point: { x: 0.75, y: 1.5, z: 0 },
    },
    {
      descriptor: { kind: "cylinder", primitiveId: 4, materialId: 19, center_m: { x: 0, y: 0, z: 0 }, radius_m: 1, halfHeight_m: 2 },
      reference: { kind: "cylinder", center_m: { x: 0, y: 0, z: 0 }, radius_m: 1, halfHeight_m: 2 },
      point: { x: 1.5, y: 2.5, z: 0 },
    },
    {
      descriptor: { kind: "ellipsoid", primitiveId: 5, materialId: 32, center_m: { x: 0, y: 0, z: 0 }, radii_m: { x: 2, y: 1, z: 0.5 } },
      reference: { kind: "ellipsoid", center_m: { x: 0, y: 0, z: 0 }, radii_m: { x: 2, y: 1, z: 0.5 } },
      point: { x: 2.4, y: 0.7, z: -0.2 },
    },
  ];

  for (const entry of cases) {
    const unpacked = unpackSvoPrimitiveRecords(packSvoPrimitiveRecords([entry.descriptor]))[0];
    const abi = sampleSvoPrimitive(unpacked, entry.point);
    const reference = sampleSvoImplicit(entry.reference, entry.point);
    close(abi.signedDistance_m, reference.signedDistance_m, 2e-6);
  }
});

test("WGSL mirror preserves integer identity and explicit hard-feature normal selection", () => {
  assert.match(svoPrimitiveWGSL, /struct SvoPrimitiveRecord \{[\s\S]*centerKind: vec4u,[\s\S]*dimensionsIdentity: vec4u,[\s\S]*orientation: vec4f,[\s\S]*metadata: vec4u/);
  assert.match(svoPrimitiveWGSL, /bitcast<vec3f>\(record\.centerKind\.xyz\)/);
  assert.match(svoPrimitiveWGSL, /record\.dimensionsIdentity\.w & 0xffffu/);
  assert.match(svoPrimitiveWGSL, /record\.dimensionsIdentity\.w >> 16u/);
  assert.match(svoPrimitiveWGSL, /fn svoPrimitiveDistance_m/);
  assert.match(svoPrimitiveWGSL, /fn svoBoxFeatureNormal/);
  assert.match(svoPrimitiveWGSL, /if \(q\.y > q\.x\) \{ axis = 1u; \}/);
  assert.match(svoPrimitiveWGSL, /if \(q\.z > q\[axis\]\) \{ axis = 2u; \}/);
  assert.match(svoPrimitiveWGSL, /fn svoCylinderFeatureNormal/);
  assert.match(svoPrimitiveWGSL, /if \(capDistance >= radialDistance\)/);
  assert.match(svoPrimitiveWGSL, /worldPoint_m\.y - terrainHeight_m/);
  assert.doesNotMatch(svoPrimitiveWGSL, /mix\([^\n]*SVO_FEATURE_BOX/);
});
