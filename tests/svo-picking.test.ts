import assert from "node:assert/strict";
import test from "node:test";
import { SPARSE_BRICK_GPU_LAYOUT, SPARSE_BRICK_NO_OWNER, packSparseBrickPlan, planSparseBrickOctree } from "../lib/sparse-brick-octree";
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
import type { SvoStructuralFluidPackedFixture } from "../lib/svo-fluid-structural-sampling";
import { FLUID_BRICK_RESIDENT } from "../lib/webgpu-fluid-brick-residency";
import { SPARSE_VOXEL_PUBLICATION_STATE, SPARSE_VOXEL_VALID_FIELDS } from "../lib/webgpu-voxel-debug";

function close(actual: number, expected: number, tolerance = 1e-4) {
  assert.ok(Math.abs(actual - expected) <= tolerance, `${actual} should be within ${tolerance} of ${expected}`);
}

function sampledFixture(phiAt: (position_m: readonly [number, number, number]) => number): SvoStructuralFluidPackedFixture {
  const dimensionsCells = [8, 4, 4] as const;
  const cellSize_m = [1, 1, 1] as const;
  const brickSize = 4;
  const coordinates = [{ x: 0, y: 0, z: 0 }, { x: 1, y: 0, z: 0 }];
  const plan = planSparseBrickOctree(coordinates, { brickSize, maximumDepth: 1 });
  const packed = packSparseBrickPlan(plan, 13);
  const geometry = new Float32Array(plan.voxelCount * 4);
  for (const leaf of plan.leaves) {
    for (let z = 0; z < brickSize; z += 1) for (let y = 0; y < brickSize; y += 1) for (let x = 0; x < brickSize; x += 1) {
      const cell = [leaf.coordinate.x * brickSize + x, y, z] as const;
      const position = cell.map((component) => component + 0.5) as [number, number, number];
      geometry[(leaf.voxelOffset + x + y * brickSize + z * brickSize ** 2) * 4] = phiAt(position);
    }
  }
  const control = new Uint32Array(32);
  control[SPARSE_BRICK_GPU_LAYOUT.controlWords.publishedNodes] = plan.nodes.length;
  control[SPARSE_BRICK_GPU_LAYOUT.controlWords.publishedLeaves] = plan.leaves.length;
  control[SPARSE_BRICK_GPU_LAYOUT.controlWords.publishedVoxels] = plan.voxelCount;
  control[SPARSE_BRICK_GPU_LAYOUT.controlWords.brickSize] = brickSize;
  const publicationState = new Uint32Array(SPARSE_VOXEL_PUBLICATION_STATE.strideBytes / 4);
  publicationState[SPARSE_VOXEL_PUBLICATION_STATE.completeGeneration] = 13;
  publicationState[SPARSE_VOXEL_PUBLICATION_STATE.validFields] = SPARSE_VOXEL_VALID_FIELDS.topology | SPARSE_VOXEL_VALID_FIELDS.coarseFluid;
  publicationState[SPARSE_VOXEL_PUBLICATION_STATE.topologyRevision] = 4;
  publicationState[SPARSE_VOXEL_PUBLICATION_STATE.coarseFluidRevision] = 12;
  return {
    control, nodes: packed.nodes, leaves: packed.leaves, geometry,
    fluidLeafStates: new Uint32Array(plan.leaves.length).fill(FLUID_BRICK_RESIDENT),
    publicationState,
    domain: { worldOrigin_m: [0, 0, 0], cellSize_m, dimensionsCells, brickSize, maximumDepth: 1 },
    expectedCompleteGeneration: 13,
  };
}

function copied(source: SvoStructuralFluidPackedFixture): SvoStructuralFluidPackedFixture {
  return {
    ...source,
    control: new Uint32Array(source.control), nodes: new Uint32Array(source.nodes), leaves: new Uint32Array(source.leaves),
    geometry: new Float32Array(source.geometry), fluidLeafStates: new Uint32Array(source.fluidLeafStates),
    publicationState: new Uint32Array(source.publicationState),
  };
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

test("coincident solid, glass, and fluid resolve by authored opaque priority", () => {
  const fluid = sampledFixture((position) => position[0] - 4);
  const pane = {
    paneId: 8, materialId: 31, ownerId: SPARSE_BRICK_NO_OWNER,
    center_m: [4, 1.5, 1.5] as const,
    orientation: { w: Math.SQRT1_2, x: 0, y: Math.SQRT1_2, z: 0 },
    halfExtent_m: [1, 1] as const, thickness_m: 0.01, absorption_mInv: [0, 0, 0] as const,
  };
  const common = {
    thinGlass: [pane], structuralFluid: { source: fluid, materialId: 40 }, rigidBodyIds: ["box"],
  };
  const pickRay = { origin_m: [0.5, 1.5, 1.5] as const, direction: [1, 0, 0] as const, tMax_m: 7 };
  const all = pickSvoScene({
    ...common,
    primitives: [{
      kind: "box", primitiveId: 1, materialId: 20, ownerId: 0,
      center_m: { x: 4.5, y: 1.5, z: 1.5 }, halfExtents_m: { x: 0.5, y: 0.5, z: 0.5 },
    }],
  }, pickRay, { coincidenceEpsilon_m: 0.05 });
  assert.equal(all.status, "hit");
  if (all.status === "hit") {
    assert.equal(all.hit.source, "primitive");
    assert.deepEqual(all.interaction, { kind: "rigid", rigidBodyIndex: 0, rigidBodyId: "box" });
  }

  const transparent = pickSvoScene(common, pickRay, { coincidenceEpsilon_m: 0.05 });
  assert.equal(transparent.status, "hit");
  if (transparent.status === "hit") {
    assert.equal(transparent.hit.source, "thinGlass", "analytic glass wins a coincident sampled-fluid boundary");
    assert.deepEqual(transparent.interaction, { kind: "none", reason: "no-owner" });
  }
});

test("inside coarse fluid returns the exit/media transition but never impersonates an owner", () => {
  const fluid = sampledFixture((position) => position[0] - 4);
  const result = pickSvoScene({
    structuralFluid: { source: fluid, materialId: 40 }, rigidBodyIds: ["body"], incidentMedium: "air",
  }, { origin_m: [1.5, 1.5, 1.5], direction: [2, 0, 0], tMax_m: 6 });
  assert.equal(result.status, "hit");
  if (result.status !== "hit") return;
  close(result.hit.position_m[0], 4, 2e-5);
  assert.equal(result.hit.mediumBefore, "water");
  assert.equal(result.hit.mediumAfter, "air");
  assert.equal(result.hit.ownerId, SPARSE_BRICK_NO_OWNER);
  assert.equal(result.hit.localTopologyGeneration, 13);
  assert.deepEqual(result.interaction, { kind: "none", reason: "fluid" });
  assert.ok(result.work.fluidTopologyNodeVisits > 0);
});

test("stale, nonresident, and exhausted fluid fail closed ahead of otherwise valid rigid hits", () => {
  const source = sampledFixture((position) => position[0] - 4);
  const primitive = {
    kind: "sphere" as const, primitiveId: 1, materialId: 2, ownerId: 0,
    center_m: { x: 6, y: 1.5, z: 1.5 }, radius_m: 0.5,
  };
  const pickingRay = { origin_m: [0.5, 1.5, 1.5] as const, direction: [1, 0, 0] as const, tMax_m: 7 };

  const stale = copied(source);
  stale.expectedCompleteGeneration = 14;
  const staleResult = pickSvoScene({ primitives: [primitive], structuralFluid: { source: stale, materialId: 40 } }, pickingRay);
  assert.deepEqual({ status: staleResult.status, reason: "reason" in staleResult ? staleResult.reason : undefined }, {
    status: "invalid", reason: "fluid-stale-generation",
  });

  const retired = copied(source);
  retired.fluidLeafStates[0] = 0;
  const retiredResult = pickSvoScene({ primitives: [primitive], structuralFluid: { source: retired, materialId: 40 } }, pickingRay);
  assert.deepEqual({ status: retiredResult.status, reason: "reason" in retiredResult ? retiredResult.reason : undefined }, {
    status: "invalid", reason: "fluid-nonresident",
  });

  const exhausted = pickSvoScene({
    primitives: [primitive],
    structuralFluid: { source, materialId: 40, options: { maximumSteps: 1, maximumNodeVisits: 2_000, step_m: 0.1 } },
  }, pickingRay);
  assert.deepEqual({ status: exhausted.status, reason: "reason" in exhausted ? exhausted.reason : undefined }, {
    status: "work-exhausted", reason: "fluid-work-exhausted",
  });
});

test("candidate ties and interaction policy are deterministic independent of input order", () => {
  const candidate = (source: SvoPickingCandidate["source"], sourceIndex: number, distance_m = 2): SvoPickingCandidate => ({
    distance_m, position_m: [0, 0, 0], geometricNormal: [0, 1, 0], materialId: 1,
    ownerId: source === "primitive" ? 5 : SPARSE_BRICK_NO_OWNER,
    mediumBefore: "air", mediumAfter: source === "structuralCoarseFluid" ? "water" : "opaque",
    boundaryMedium: source === "structuralCoarseFluid" ? "water" : "opaque",
    source, featureId: 0, localTopologyGeneration: 0, sourceIndex,
  });
  const fluid = candidate("structuralCoarseFluid", 0, 1.999999);
  const glass = candidate("thinGlass", 0, 2);
  const terrain = candidate("terrain", 0, 2);
  const primitive = candidate("primitive", 1, 2.000001);
  assert.equal(resolveNearestSvoPickingCandidate([fluid, glass, terrain, primitive])?.source, "primitive");
  assert.equal(resolveNearestSvoPickingCandidate([primitive, terrain, glass, fluid])?.source, "primitive");
  assert.equal(resolveNearestSvoPickingCandidate([candidate("primitive", 2), candidate("primitive", 1)])?.sourceIndex, 1);
  assert.deepEqual(svoPickingInteractionForHit(fluid, ["a", "b", "c"]), { kind: "none", reason: "fluid" });
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
  assert.match(svoPickingWGSL, /hit\.source==SVO_PICK_SOURCE_FLUID_COARSE\|\|hit\.source==SVO_PICK_SOURCE_TERRAIN/);
  assert.match(svoPickingWGSL, /hit\.ownerId>=rigidOwnerCount/);
  assert.match(svoPickingWGSL, /SVO_PICK_STATUS_EXHAUSTED:u32=2u/);
  assert.match(svoPickingWGSL, /SVO_PICK_STATUS_INVALID:u32=3u/);
});
