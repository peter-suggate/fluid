import assert from "node:assert/strict";
import test from "node:test";

// A renderer resolves a method by id on any path that reaches a scene, and the
// catalog cases below build scenes without ever creating a solver.
import "../lib/methods";
import { presentationModeForScene, SCENE_ENVIRONMENT_PAYLOAD_BUDGET_BYTES } from "../lib/core/scene-definition";
import { SCENE_CATALOG } from "../lib/core/scenes";
import { studioStageSceneryGraph } from "../lib/core/studio-stage-scene";
import { buildEnvironmentProxyCatalog, environmentProxyPrimitives } from "../lib/core/voxel-environments";
import type { EnvironmentProxyPrimitive } from "../lib/core/voxel-environments";
import {
  createSvoEnvironmentCoarsening,
  environmentProxyFeatureSize_m,
  SVO_ENVIRONMENT_FEATURE_VOXELS,
  solidWorldTerrainSurfaceCoarseningRegions,
  svoEnvironmentCoarseningPower,
  svoEnvironmentPayloadBytes,
} from "../lib/svo/svo-environment-coarsening";
import { sceneDocument } from "../lib/core/scene-definition";
import { createTallCellsHillsideDamBreakScene, getScenePreset } from "../lib/core/scenes";
import { solidWorldForScene } from "../lib/core/solid-world";

/**
 * The environment's resolution ladder, and the gate that reads the same rule.
 *
 * What this is here to catch is one failure with two faces, both of which were
 * live before the ladder existed. `ocean-seiche` — a 8 m tank on a 25 mm
 * lattice — drew its house set at the solver's own cell and claimed 62,752
 * environment bricks and a 1,959 MiB tree for a plate, a stem and a lamp cone,
 * so it was held back to `fluid-only` and had no floor at all. The CM12 paper
 * figures carry their own room shell, seventy metres of it at 50 mm, and the
 * only thing that ever kept them from trying to build it was a budget that
 * happened to say no while measuring a different set entirely.
 *
 * So the cases run in both directions: the rule must coarsen the house set far
 * enough that a large tank can afford it, and the gate must still refuse a set
 * that no device can hold. A change that relaxes one of those usually breaks
 * the other, which is why they are asserted together.
 */

const material = { colorLinear: [1, 1, 1] as const, roughness: 1, metallic: 0, emission: 0 };

/** A box proxy with an AABB stated independently of its half sizes. */
function boxProxy(
  halfSize: readonly [number, number, number],
  center: readonly [number, number, number],
  aabbHalf: readonly [number, number, number] = halfSize,
): EnvironmentProxyPrimitive {
  return {
    kind: "box", key: `box-${center.join(",")}`, ownerIndex: 0, group: "test", tags: [],
    center_m: { x: center[0], y: center[1], z: center[2] },
    halfSize_m: { x: halfSize[0], y: halfSize[1], z: halfSize[2] },
    material: material as never,
    aabb_m: {
      min: { x: center[0] - aabbHalf[0], y: center[1] - aabbHalf[1], z: center[2] - aabbHalf[2] },
      max: { x: center[0] + aabbHalf[0], y: center[1] + aabbHalf[1], z: center[2] + aabbHalf[2] },
    },
  } as EnvironmentProxyPrimitive;
}

test("a planar terminal is sized by its in-plane feature, not analytic thickness", () => {
  // Thickness survives in the exact terminal record, so voxel resolution is
  // responsible only for locating the finite patch in its two broad axes.
  const plate = boxProxy([10, 0.025, 10], [0, 0, 0]);
  const turned = boxProxy([10, 0.025, 10], [0, 0, 0], [10.02, 7.1, 10.02]);
  assert.equal(environmentProxyFeatureSize_m(plate), 20);
  assert.equal(environmentProxyFeatureSize_m(turned), 20);
});

test("coarsening stops at the brick, so every brick plane stays a voxel plane", () => {
  // `planSparseSceneDomain` aligns the lattice minimum to the brick, so a leaf
  // whose voxels are 2^p cells lands on whole voxels exactly while 2^p divides
  // the brick. Past that the stage floor's top face would no longer fall on
  // y = 0, which is a visible seam rather than a lost detail.
  assert.equal(svoEnvironmentCoarseningPower(8), 3);
  assert.equal(svoEnvironmentCoarseningPower(4), 2);
  assert.equal(svoEnvironmentCoarseningPower(1), 0);
  assert.throws(() => svoEnvironmentCoarseningPower(6), RangeError);
  assert.throws(() => svoEnvironmentCoarseningPower(0), RangeError);
});

test("a node splits for the finest solid in it and for nothing else", () => {
  // Level 1 of this table holds 1.6 m nodes, so its voxels are 0.2 m and the
  // finest feature it records is 0.8 m — the boundary both slabs are placed on.
  const coarsening = (primitives: readonly EnvironmentProxyPrimitive[], crowdingTarget = 8) =>
    createSvoEnvironmentCoarsening({
      primitives, worldOrigin_m: [0, 0, 0], brickSize: 8, maximumDepth: 3, crowdingTarget,
      nodeEdge_m: [[3.2, 3.2, 3.2], [1.6, 1.6, 1.6], [0.8, 0.8, 0.8], [0.4, 0.4, 0.4]],
    });
  const node = { x: 0, y: 0, z: 0 };

  const thick = coarsening([boxProxy([0.8, 0.4, 0.8], [0.8, 0.4, 0.8])]);
  assert.equal(thick.refineEnvironmentLeaf(1, node), false);
  assert.equal(thick.statistics.resolvedLeaves, 1);

  const thin = coarsening([boxProxy([0.8, 0.025, 0.8], [0.8, 0.4, 0.8])]);
  assert.equal(thin.refineEnvironmentLeaf(1, node), false);
  assert.equal(thin.statistics.resolvedLeaves, 1);

  // The same thin plate is not in the node above it at all. Bounds overlap is
  // inclusive on purpose, so this probes a node that clears the plate rather
  // than one that shares a face with it.
  assert.equal(thin.refineEnvironmentLeaf(1, { x: 0, y: 1, z: 0 }), false);
  assert.equal(thin.statistics.emptyLeaves, 1);

  // A small box without the decisive slab aspect ratio still resolves its
  // actual thinnest feature and therefore splits.
  const ordinaryThinBox = coarsening([boxProxy([0.19, 0.025, 0.19], [0.8, 0.4, 0.8])]);
  assert.equal(ordinaryThinBox.refineEnvironmentLeaf(1, node), true);
  assert.equal(ordinaryThinBox.statistics.featureSplits, 1);

  // A leaf binds a bounded number of solids and drops the surplus silently, so
  // a coarse leaf that gathers a crowd is worse than the leaves it replaced.
  const crowd = coarsening(
    [0, 1, 2].map((index) => boxProxy([0.8, 0.4, 0.8], [0.8, 0.4, 0.8 + index * 0.01])), 2);
  assert.equal(crowd.refineEnvironmentLeaf(1, node), true);
  assert.equal(crowd.statistics.crowdingSplits, 1);

  const terrainPage = createSvoEnvironmentCoarsening({
    primitives: [],
    regions: [{ minimum_m: [0.01, 0.01, 0.01], maximum_m: [1.59, 1.59, 1.59],
      feature_m: 0 }],
    worldOrigin_m: [0, 0, 0], brickSize: 8, maximumDepth: 3, crowdingTarget: 8,
    nodeEdge_m: [[3.2, 3.2, 3.2], [1.6, 1.6, 1.6], [0.8, 0.8, 0.8], [0.4, 0.4, 0.4]],
  });
  assert.equal(terrainPage.refineEnvironmentLeaf(1, node), true,
    "an exposed terrain page must retain the source-cell step size");
  assert.equal(terrainPage.refineEnvironmentLeaf(1, { x: 1, y: 1, z: 1 }), false,
    "terrain refinement must remain local to its exposed page");
});

test("only exposed terrain pages opt out of voxel coarsening", () => {
  const hillside = createTallCellsHillsideDamBreakScene();
  const hillsideWorld = solidWorldForScene(hillside);
  const regions = solidWorldTerrainSurfaceCoarseningRegions(hillside, hillsideWorld);
  assert.ok(regions.length > 0);
  assert.ok(regions.length < hillsideWorld.pages.length,
    "buried terrain pages must remain eligible for coarsening");
  assert.ok(regions.every(({ feature_m }) => feature_m === 0));

  const waterBox = getScenePreset("water-box").create();
  assert.deepEqual(solidWorldTerrainSurfaceCoarseningRegions(
    waterBox, solidWorldForScene(waterBox)), [],
  "terrain-free tank floors stay on their exact planar path");
});

test("the planar terminal ladder keeps the ocean-scale set within budget", () => {
  // The same set, on the tank it was authored around and on one twelve times
  // wider. Without a ladder the second costs the square of the first; with one
  // both land in the same order of magnitude, because the plate that grew is
  // also the plate that may be drawn coarser.
  const bytes = (id: string) => {
    const definition = SCENE_CATALOG.find((entry) => entry.id === id)!;
    const scene = sceneDocument(definition);
    return svoEnvironmentPayloadBytes(
      environmentProxyPrimitives(buildEnvironmentProxyCatalog(scene, scene.environment ?? "default")),
      { cellSize_m: scene.voxelDomain.finestCellSize_m, brickSize: scene.voxelDomain.brickSize_cells });
  };
  const reference = bytes("water-box-dam-break");
  const ocean = bytes("ocean-seiche");
  assert.ok(reference > 0);
  assert.ok(ocean < 32 * 1024 ** 2, `ocean-seiche claims ${ocean} B`);
  assert.ok(ocean < SCENE_ENVIRONMENT_PAYLOAD_BUDGET_BYTES);
});

test("automatic stage presentation follows its set budget; explicit modes remain explicit", () => {
  // The budget is a ceiling on a ceiling, so what matters is not that it is
  // exact but that nothing sits near it. Measured across the catalog the house
  // set costs at most 30 MiB and an authored room shell at least 887 MiB.
  const affordable: string[] = [], unaffordable: string[] = [], explicit: string[] = [];
  for (const definition of SCENE_CATALOG) {
    if (definition.environment !== "stage") continue;
    const scene = sceneDocument(definition);
    const bytes = svoEnvironmentPayloadBytes(
      environmentProxyPrimitives(buildEnvironmentProxyCatalog(scene, scene.environment ?? "default")),
      { cellSize_m: scene.voxelDomain.finestCellSize_m, brickSize: scene.voxelDomain.brickSize_cells });
    if (definition.presentationMode) {
      explicit.push(`${definition.id} ${(bytes / 1024 ** 2).toFixed(1)} MiB`);
      assert.equal(presentationModeForScene(definition, scene),
        definition.presentationMode, definition.id);
      continue;
    }
    (bytes <= SCENE_ENVIRONMENT_PAYLOAD_BUDGET_BYTES ? affordable : unaffordable)
      .push(`${definition.id} ${(bytes / 1024 ** 2).toFixed(1)} MiB`);
    assert.equal(presentationModeForScene(definition, scene),
      bytes <= SCENE_ENVIRONMENT_PAYLOAD_BUDGET_BYTES ? "full-scene" : "fluid-only",
      definition.id);
  }
  assert.ok(affordable.length > 20, affordable.join(", "));
  assert.ok(affordable.length + unaffordable.length > 30);
  // The hillside is the cheapest explicit set in the catalog and now costs
  // nothing at all: its scenery is one terrain shell, which is analytic
  // metadata rather than a proxy, and the emissive practical it used to hang
  // over the slope is gone in favour of an authored sun.
  assert.ok(explicit.some((entry) => entry.startsWith("tall-cells-hillside-dam-break 0.0 MiB")),
    `expected the explicitly full-scene sparse hillside set, got ${explicit.join(", ")}`);
});

test("ocean-seiche presents the floor it was missing", () => {
  const definition = SCENE_CATALOG.find((entry) => entry.id === "ocean-seiche")!;
  const scene = sceneDocument(definition);
  assert.equal(presentationModeForScene(definition, scene), "full-scene");
  assert.deepEqual((scene.scenery?.nodes ?? []).map((node) => node.id),
    ["shell", "stage/floor", "lamp/stem", "lamp/reflector"]);
  // Every level offered is a level the predicate may decline, so the constant
  // the floor is expressed in is asserted rather than implied by the counts.
  assert.equal(SVO_ENVIRONMENT_FEATURE_VOXELS, 4);
});
