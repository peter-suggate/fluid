import assert from "node:assert/strict";
import test from "node:test";
import { SPARSE_BRICK_NO_OWNER } from "../lib/sparse-brick-octree";
import {
  SVO_PICKING_DEFAULT_COINCIDENCE_EPSILON_M,
  SVO_PICKING_DEFAULT_MAXIMUM_ANALYTIC_TESTS,
  SVO_PICKING_MAXIMUM_ANALYTIC_TESTS,
  pickSvoScene,
  resolveNearestSvoPickingCandidate,
  svoPickingInteractionForHit,
  svoPickingWGSL,
  type SvoPickingCandidate,
} from "../lib/svo-picking";

function close(actual: number, expected: number, tolerance = 1e-4) {
  assert.ok(Math.abs(actual - expected) <= tolerance, `${actual} should be within ${tolerance} of ${expected}`);
}

const ray = { origin_m: [-5, 0, 0] as const, direction: [10, 0, 0] as const, tMax_m: 20 };

test("exact primitive picking returns world data and maps only rigid owner indices to body interaction state", () => {
  const result = pickSvoScene({
    primitives: [{
      kind: "box", primitiveId: 10, materialId: 22, ownerId: 0,
      center_m: { x: 0, y: 0, z: 0 }, halfExtents_m: { x: 1, y: 1, z: 1 },
    }],
    primitiveLocalTopologyGeneration: 7,
    rigidBodyIds: ["box-a"],
  }, ray);
  assert.equal(result.status, "hit");
  if (result.status !== "hit") return;
  close(result.hit.distance_m, 4);
  assert.deepEqual(result.hit.position_m, [-1, 0, 0]);
  assert.deepEqual(result.hit.geometricNormal, [-1, 0, 0]);
  assert.deepEqual({
    material: result.hit.materialId, owner: result.hit.ownerId, source: result.hit.source,
    before: result.hit.mediumBefore, after: result.hit.mediumAfter, generation: result.hit.localTopologyGeneration,
  }, { material: 22, owner: 0, source: "primitive", before: "air", after: "opaque", generation: 7 });
  assert.deepEqual(result.interaction, { kind: "rigid", rigidBodyIndex: 0, rigidBodyId: "box-a" });

  const environment = pickSvoScene({
    primitives: [{
      kind: "sphere", primitiveId: 11, materialId: 23, ownerId: 4,
      center_m: { x: 0, y: 0, z: 0 }, radius_m: 1,
    }],
    rigidBodyIds: ["only-rigid"],
  }, ray);
  assert.equal(environment.status, "hit");
  if (environment.status === "hit") assert.deepEqual(environment.interaction, { kind: "none", reason: "environment-owner" });
});

test("inside-origin exits and exact grazing primitive contacts remain pickable", () => {
  const primitive = {
    kind: "sphere" as const, primitiveId: 1, materialId: 2, ownerId: 0,
    center_m: { x: 0, y: 0, z: 0 }, radius_m: 1,
  };
  const inside = pickSvoScene({ primitives: [primitive], rigidBodyIds: ["sphere"] }, {
    origin_m: [0, 0, 0], direction: [4, 0, 0], tMax_m: 2,
  });
  assert.equal(inside.status, "hit");
  if (inside.status === "hit") { close(inside.hit.distance_m, 1); assert.deepEqual(inside.hit.geometricNormal, [1, 0, 0]); }
  const grazing = pickSvoScene({ primitives: [primitive], rigidBodyIds: ["sphere"] }, {
    origin_m: [-2, 1, 0], direction: [1, 0, 0], tMax_m: 4,
  });
  assert.equal(grazing.status, "hit");
  if (grazing.status === "hit") { close(grazing.hit.distance_m, 2); assert.deepEqual(grazing.hit.geometricNormal, [0, 1, 0]); }
});

test("analytic terrain returns opaque material identity but never a selectable rigid owner", () => {
  const result = pickSvoScene({
    terrain: { description: { baseHeight_m: 0.25, features: [] }, sceneScale_m: 4, materialId: 51, localTopologyGeneration: 3 },
    rigidBodyIds: ["body"],
  }, { origin_m: [0, 1, 0], direction: [0, -2, 0], tMax_m: 3 });
  assert.equal(result.status, "hit");
  if (result.status !== "hit") return;
  close(result.hit.position_m[1], 0.25, 5e-4);
  assert.equal(result.hit.ownerId, SPARSE_BRICK_NO_OWNER);
  assert.equal(result.hit.source, "terrain");
  assert.equal(result.hit.localTopologyGeneration, 3);
  assert.deepEqual(result.interaction, { kind: "none", reason: "terrain" });
  assert.ok(result.work.terrainHeightEvaluations > 0);
});

test("candidate ties and interaction policy are deterministic independent of input order", () => {
  const candidate = (source: SvoPickingCandidate["source"], sourceIndex: number, distance_m = 2): SvoPickingCandidate => ({
    distance_m, position_m: [0, 0, 0], geometricNormal: [0, 1, 0], materialId: 1,
    ownerId: source === "primitive" ? 5 : SPARSE_BRICK_NO_OWNER,
    mediumBefore: "air", mediumAfter: "opaque",
    boundaryMedium: "opaque",
    source, featureId: 0, localTopologyGeneration: 0, sourceIndex,
  });
  const glass = candidate("thinGlass", 0, 2);
  const terrain = candidate("terrain", 0, 2);
  const primitive = candidate("primitive", 1, 2.000001);
  assert.equal(resolveNearestSvoPickingCandidate([glass, terrain, primitive])?.source, "primitive");
  assert.equal(resolveNearestSvoPickingCandidate([primitive, terrain, glass])?.source, "primitive");
  assert.equal(resolveNearestSvoPickingCandidate([candidate("primitive", 2), candidate("primitive", 1)])?.sourceIndex, 1);
  assert.deepEqual(svoPickingInteractionForHit({ ...primitive, ownerId: SPARSE_BRICK_NO_OWNER }, ["a"]), { kind: "none", reason: "no-owner" });
});

test("analytic caps and malformed rays terminate explicitly", () => {
  const primitive = {
    kind: "sphere" as const, primitiveId: 1, materialId: 2, ownerId: 0,
    center_m: { x: 0, y: 0, z: 0 }, radius_m: 1,
  };
  const exhausted = pickSvoScene({ primitives: [primitive, primitive] }, ray, { maximumAnalyticTests: 1 });
  assert.equal(exhausted.status, "work-exhausted");
  if (exhausted.status === "work-exhausted") assert.equal(exhausted.reason, "analytic-work-exhausted");
  assert.throws(() => pickSvoScene({}, { ...ray, direction: [0, 0, 0] }), /non-zero/);
  assert.throws(() => pickSvoScene({}, ray, { maximumAnalyticTests: SVO_PICKING_MAXIMUM_ANALYTIC_TESTS + 1 }), /analytic-test cap/);
});

test("binding-free WGSL adapter fixes tie priority, fail status, and rigid-owner filtering", () => {
  assert.equal(SVO_PICKING_DEFAULT_COINCIDENCE_EPSILON_M, 1e-5);
  assert.equal(SVO_PICKING_DEFAULT_MAXIMUM_ANALYTIC_TESTS, 4_096);
  assert.equal(SVO_PICKING_MAXIMUM_ANALYTIC_TESTS, 65_536);
  assert.doesNotMatch(svoPickingWGSL, /@group|@binding/);
  assert.match(svoPickingWGSL, /candidate\.source<best\.source/);
  assert.match(svoPickingWGSL, /candidate\.source==best\.source&&candidate\.sourceIndex<best\.sourceIndex/);
  assert.match(svoPickingWGSL, /hit\.source==SVO_PICK_SOURCE_TERRAIN/);
  assert.match(svoPickingWGSL, /hit\.ownerId>=rigidOwnerCount/);
  assert.match(svoPickingWGSL, /SVO_PICK_STATUS_EXHAUSTED:u32=2u/);
  assert.match(svoPickingWGSL, /SVO_PICK_STATUS_INVALID:u32=3u/);
});
