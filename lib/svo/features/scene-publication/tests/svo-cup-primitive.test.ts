import assert from "node:assert/strict";
import test from "node:test";

import { cupDistance_m, cupWallThickness_m } from "../../../../core/scene-shape";
import type { Vec3 } from "../../../../core/model";
import { SVO_PRIMITIVE_FEATURES, canonicalSvoPrimitive, intersectSvoPrimitive, packSvoPrimitiveRecords, sampleSvoPrimitive, svoPrimitiveForRigidBody, unpackSvoPrimitiveRecords, type SvoCupPrimitive } from "../../../contracts/svo-primitive-abi";
import { sparseScenePrimitiveFeatureRadius, sparseScenePrimitiveForSvoDescriptor } from "../../../../core/webgpu-sparse-scene-proxies";

/**
 * The cup is the first shape in this repository whose interior is not solid,
 * and a hollow shape is only worth anything if every consumer agrees about the
 * hollow part. Its field is declared once, in `lib/core/scene-shape.ts`; the
 * solvers reach it through `sceneShapeWgsl` and the renderer through the SVO
 * ABI, and this file is what holds the second path to the first.
 *
 * The failure it exists to catch is specific and silent: a cup whose cavity is
 * filled draws exactly like a cup, and only stops being one when it is dipped.
 */

/** Outer radius, HALF height, wall — the render ABI's convention. */
const CUP = {
  primitiveId: 1, materialId: 20, ownerId: 3,
  kind: "cup", center_m: { x: 0, y: 0, z: 0 },
  radius_m: 0.15, halfHeight_m: 0.13, wallThickness_m: 0.02,
} as const satisfies SvoCupPrimitive;

/** The same cup in the shared table's convention, which carries the FULL height. */
const SHARED: Vec3 = { x: CUP.radius_m, y: 2 * CUP.halfHeight_m, z: CUP.wallThickness_m };

function samplePoints(): Vec3[] {
  const axis = [-0.21, -0.15, -0.13, -0.08, -0.02, 0, 0.02, 0.08, 0.13, 0.15, 0.21];
  const points: Vec3[] = [];
  for (const x of axis) for (const y of axis) for (const z of axis) points.push({ x, y, z });
  return points;
}

test("the render ABI evaluates the shared cup field, not a second copy of it", () => {
  for (const point of samplePoints()) {
    assert.equal(sampleSvoPrimitive(CUP, point).signedDistance_m, cupDistance_m(SHARED, point),
      `at ${JSON.stringify(point)}`);
  }
});

test("a cup's three authored surfaces each get their own normal", () => {
  const wall = cupWallThickness_m(SHARED);
  // Just outside the outer wall, just inside the cavity wall, just above the
  // inner floor. Each point is nearest a different one of the three surfaces.
  const outside = sampleSvoPrimitive(CUP, { x: CUP.radius_m + 0.01, y: 0, z: 0 });
  assert.ok(outside.normal && outside.normal.x > 0.99, "the outer wall faces away from the axis");
  assert.equal(outside.featureId, SVO_PRIMITIVE_FEATURES.cylinderSide);

  const cavityWall = sampleSvoPrimitive(CUP, { x: CUP.radius_m - wall - 0.005, y: 0.05, z: 0 });
  assert.ok(cavityWall.normal && cavityWall.normal.x < -0.99, "the cavity wall faces the axis");
  assert.equal(cavityWall.featureId, SVO_PRIMITIVE_FEATURES.cylinderSide);

  const floor = sampleSvoPrimitive(CUP, { x: 0, y: -CUP.halfHeight_m + wall + 0.005, z: 0 });
  assert.ok(floor.normal && floor.normal.y > 0.99, "the inner floor faces up");
  assert.equal(floor.featureId, SVO_PRIMITIVE_FEATURES.cylinderCap);
});

test("a ray down the axis lands on the inner floor, not on a filled interior", () => {
  const wall = cupWallThickness_m(SHARED);
  const hit = intersectSvoPrimitive(CUP,
    { origin_m: { x: 0, y: 1, z: 0 }, direction: { x: 0, y: -1, z: 0 }, tMin_m: 0, tMax_m: 10 });
  assert.ok(hit, "the axis ray must hit the cup");
  // A solid cylinder would answer at the rim, one wall above where the floor is.
  const floorY = -CUP.halfHeight_m + wall;
  assert.ok(Math.abs(hit.position_m.y - floorY) < 1e-4,
    `expected the floor at y=${floorY}, got y=${hit.position_m.y}`);
  assert.ok(hit.normal && hit.normal.y > 0.99, "the floor faces up");
});

test("a cup survives the record round trip and refuses a wall that would close it", () => {
  // A record holds f32s, so the round trip is exact only up to the arena's own
  // precision — the contract is that nothing else moves.
  const [recovered] = unpackSvoPrimitiveRecords(packSvoPrimitiveRecords([CUP]));
  assert.deepEqual(recovered, canonicalSvoPrimitive({
    ...CUP,
    radius_m: Math.fround(CUP.radius_m),
    halfHeight_m: Math.fround(CUP.halfHeight_m),
    wallThickness_m: Math.fround(CUP.wallThickness_m),
  }));

  // The shared field clamps a wall this thick; the ABI refuses it instead, so a
  // record can never describe a cup other than the one that gets drawn.
  assert.throws(() => canonicalSvoPrimitive({ ...CUP, wallThickness_m: CUP.radius_m }), /open cavity/);
});

test("a cup body publishes as a cup, converting the one height convention", () => {
  const published = svoPrimitiveForRigidBody(
    { shape: "cup", dimensions_m: SHARED, position_m: { x: 1, y: 2, z: 3 }, orientation: { x: 0, y: 0, z: 0, w: 1 } },
    7, 9);
  assert.equal(published.kind, "cup");
  assert.deepEqual(published, canonicalSvoPrimitive({
    ...CUP, primitiveId: 7, ownerId: 9, materialId: published.materialId,
    center_m: { x: 1, y: 2, z: 3 }, orientation: { x: 0, y: 0, z: 0, w: 1 },
  }));
});

test("the live voxelizer knows a cup's wall is its finest feature", () => {
  const live = sparseScenePrimitiveForSvoDescriptor(canonicalSvoPrimitive(CUP));
  assert.equal(live.kind, "cup");
  // Without this the centre sample of a cell straddling the wall reads the
  // cavity, and the cup voxelizes as a ring of holes.
  assert.equal(sparseScenePrimitiveFeatureRadius(live), CUP.wallThickness_m);
});
