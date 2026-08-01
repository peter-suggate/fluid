import assert from "node:assert/strict";
import test from "node:test";

import type { SceneDescription } from "../lib/model";
import type { SparseSceneDomainPlan } from "../lib/sparse-scene-domain";
import { buildSvoSceneLights, canonicalSvoLightRecord } from "../lib/svo-light-abi";
import { buildSvoStaticNodeMipPublication } from "../lib/svo-static-node-mips";
import { planSvoStaticShadowField } from "../lib/svo-static-shadow-field";
import {
  buildSvoStaticShadowVisibleProofs,
  svoDirectionalPageBeamIntersectsAabb,
  svoFinitePageBeamIntersectsAabb,
  svoStaticShadowLocalTraceReach_m,
} from "../lib/svo-static-shadow-visible-proofs";
import type { EnvironmentProxyPrimitive } from "../lib/voxel-environments";

function scene(): SceneDescription {
  return {
    schemaVersion: "1.0.0", sceneId: "shadow-proof-test", environment: "default", randomSeed: 1, duration_s: 1,
    container: { width_m: 24, height_m: 8, depth_m: 24, fillFraction: 0, top: "open", fluidWallMode: "free-slip" },
    voxelDomain: { finestCellSize_m: 1, brickSize_cells: 8 },
    fluid: { density_kg_m3: 1_000, dynamicViscosity_Pa_s: .001, surfaceTension_N_m: .07, gravity_m_s2: { x: 0, y: -9.81, z: 0 }, initialCondition: "tank-fill" },
    nominalResolution: { length_m: 1 },
    numerics: { fixedDt_s: .01, maxDt_s: .01, pressureRelativeTolerance: 1e-5, pressureMaxIterations: 10 },
    rigidBodies: [],
  };
}

function domain(): SparseSceneDomainPlan {
  const point = (x: number, y: number, z: number) => ({ x, y, z });
  return {
    brickSize: 8, cellSize_m: [1, 1, 1], worldOrigin_m: point(0, 0, 0),
    solverGridOriginCells: [0, 0, 0], solverDimensionsCells: [24, 8, 24], sceneDimensionsCells: [24, 8, 24],
    brickDimensions: [3, 1, 3],
    solverBounds_m: { min: point(0, 0, 0), max: point(24, 8, 24) },
    worldBounds_m: { min: point(0, 0, 0), max: point(24, 8, 24) },
    solverBrickCoordinates: [], environmentBrickCoordinates: [], proxyBrickCoordinates: [], coordinates: [],
  };
}

function box(key: string, x: number, z: number): EnvironmentProxyPrimitive {
  return {
    kind: "box", key, ownerIndex: 0, group: "stone", tags: [],
    center_m: { x: x + 4, y: 4, z: z + 4 },
    halfSize_m: { x: 3, y: 3, z: 3 },
    material: { colorLinear: [1, 1, 1], emission: 0, roughness: .8 },
    aabb_m: { min: { x: x + 1, y: 1, z: z + 1 }, max: { x: x + 7, y: 7, z: z + 7 } },
  };
}

function build(proxies: readonly EnvironmentProxyPrimitive[], direction: readonly [number, number, number], capacity = 64) {
  const description = scene();
  const mips = buildSvoStaticNodeMipPublication(description, domain(), proxies, {
    generation: 4, levelCount: 1, capacity, samplesPerAxis: 1,
  });
  const lights = buildSvoSceneLights(description, {
    revision: 7, maximumRecords: 1, directionalDirection: direction,
  });
  const plan = planSvoStaticShadowField(mips.plan, lights);
  // One diagonal excuses the receiver's own page and nothing else, which is the
  // narrowest sound reach and keeps these beam assertions about the beam. The
  // fixture's pages are 8 m across, so the production default of two diagonals
  // would put this entire 24 m scene inside the local trace.
  return {
    mips, lights, plan,
    publication: buildSvoStaticShadowVisibleProofs(mips, lights.records, plan, { localTraceReachPageDiagonals: 1 }),
  };
}

test("parallel page beam uses an exact Minkowski ray/slab intersection", () => {
  const receiver = { minimum: [0, 0, 0] as const, maximum: [1, 1, 1] as const };
  assert.equal(svoDirectionalPageBeamIntersectsAabb(receiver,
    { minimum: [2, 0, 0], maximum: [3, 1, 1] }, [1, 0, 0]), true);
  assert.equal(svoDirectionalPageBeamIntersectsAabb(receiver,
    { minimum: [2, 2, 0], maximum: [3, 3, 1] }, [1, 0, 0]), false);
  assert.equal(svoDirectionalPageBeamIntersectsAabb(receiver,
    { minimum: [-3, 0, 0], maximum: [-2, 1, 1] }, [1, 0, 0]), false);
});

test("directional producer proves only pages with no remote swept-volume blocker", () => {
  const result = build([box("receiver", 0, 0), box("ahead", 8, 0)], [1, 0, 0]);
  const byPage = new Set(result.publication.proofs.map(({ page }) => page));
  assert.equal(byPage.has(result.plan.pages[0].keyString), false, "the page behind the blocker stays exact");
  assert.equal(byPage.has(result.plan.pages[1].keyString), true, "the front page has no remote blocker");
  assert.equal(result.publication.requiresExactLocalPageTrace, true);
  assert.equal(result.publication.requiresCurrentFrameRigidOverlay, true);
  assert.equal(result.publication.semantics, "remote-static-hard-ray-visible");
  assert.equal(result.publication.maximumCertifiedConeApertureRadians, 0);
  assert.equal(result.publication.includesNodeMipSamplingSupport, false);
});

test("off-axis pages do not create false blocker dependencies", () => {
  const result = build([box("receiver", 0, 0), box("off-axis", 8, 8)], [1, 0, 0]);
  assert.equal(result.publication.proofs.length, 2);
  assert.equal(result.publication.visibleFraction, 1);
});

test("capacity omissions fail closed with no visible certificates", () => {
  const result = build([
    box("a", 0, 0), box("b", 8, 0), box("c", 16, 0), box("d", 0, 8),
  ], [1, 0, 0], 1);
  assert.ok(result.mips.omittedBasePageCount > 0);
  assert.equal(result.publication.proofs.length, 0);
  assert.equal(result.publication.failClosedReason, "omitted-static-pages");
});

test("a caller-supplied complete proxy cover safely recovers proofs when sampled pages were omitted", () => {
  const proxies = [box("a", 0, 0), box("b", 8, 0), box("c", 16, 0), box("d", 0, 8)];
  // Capacity retains Morton-first page a. Toward -X it has no remote blocker;
  // the omitted +X pages are nevertheless present in the supplied proof cover.
  const result = build(proxies, [-1, 0, 0], 1);
  const publication = buildSvoStaticShadowVisibleProofs(result.mips, result.lights.records, result.plan, {
    localTraceReachPageDiagonals: 1,
    completeStaticBlockerBounds: proxies.map(({ aabb_m }) => ({
      minimum: [aabb_m.min.x, aabb_m.min.y, aabb_m.min.z],
      maximum: [aabb_m.max.x, aabb_m.max.y, aabb_m.max.z],
    })),
  });
  assert.equal(publication.failClosedReason, undefined);
  assert.ok(publication.proofs.length > 0);
});

test("topology or light revision mismatches cannot produce reusable proofs", () => {
  const first = build([box("a", 0, 0)], [1, 0, 0]);
  const changedLights = buildSvoSceneLights(scene(), {
    revision: first.lights.revision + 1, maximumRecords: 1, directionalDirection: [1, 0, 0],
  });
  assert.throws(() => buildSvoStaticShadowVisibleProofs(first.mips, changedLights.records, first.plan),
    /exact authored-light revision/);
});

test("the local-trace reach is published and must cover at least the receiver's own page", () => {
  const first = build([box("a", 0, 0)], [1, 0, 0]);
  const pageDiagonal = Math.hypot(...first.mips.basePageSize_m);
  assert.equal(svoStaticShadowLocalTraceReach_m(first.mips.basePageSize_m), pageDiagonal);
  // Coarser receiver levels cover proportionally more world per page.
  assert.equal(svoStaticShadowLocalTraceReach_m(first.mips.basePageSize_m, 2), pageDiagonal * 4);
  assert.equal(
    buildSvoStaticShadowVisibleProofs(first.mips, first.lights.records, first.plan,
      { localTraceReachPageDiagonals: 1 }).localTraceReach_m,
    pageDiagonal,
  );
  // The default keeps the whole 26-page neighbourhood inside the local trace.
  assert.equal(
    buildSvoStaticShadowVisibleProofs(first.mips, first.lights.records, first.plan).localTraceReach_m,
    pageDiagonal * 2,
  );
  assert.throws(() => buildSvoStaticShadowVisibleProofs(first.mips, first.lights.records, first.plan,
    { localTraceReachPageDiagonals: 0.5 }), /at least one page diagonal/);
});

test("reach excuses exactly the blockers the local trace is guaranteed to reach", () => {
  // A corner neighbour's farthest point is exactly two page diagonals away, so
  // two diagonals is the smallest reach that owns the full neighbourhood — and
  // one diagonal must own nothing beyond the receiver page itself.
  const blocked = build([box("receiver", 0, 0), box("ahead", 8, 0)], [1, 0, 0]);
  assert.equal(blocked.publication.proofs.length, 1, "one diagonal leaves the blocked page exact");
  const excused = buildSvoStaticShadowVisibleProofs(blocked.mips, blocked.lights.records, blocked.plan,
    { localTraceReachPageDiagonals: 2 });
  assert.equal(excused.proofs.length, 2,
    "two diagonals hand the adjacent blocker to the local trace, so both pages certify");
});

test("finite emitters use the hull margin, not the union of receiver and emitter bounds", () => {
  const receiver = { minimum: [0, 0, 0] as const, maximum: [2, 2, 2] as const };
  const light = canonicalSvoLightRecord({
    lightId: 5, ownerId: 0, revision: 1, kind: "point",
    position_m: [1, 40, 1], range_m: 100, direction: [0, -1, 0],
    colorLinear: [1, 1, 1], intensity: 1,
    axisU: [1, 0, 0], halfWidth_m: 0, axisV: [0, 0, 1], halfHeight_m: 0, radius_m: .5,
    sourceKey: "test/point",
  });
  // Directly between the receiver and a ceiling fixture: a genuine blocker.
  assert.equal(svoFinitePageBeamIntersectsAabb(receiver,
    { minimum: [0, 10, 0], maximum: [2, 12, 2] }, light), true);
  // Beside that column. The union AABB of receiver and emitter spans the whole
  // shaft between them and would have called this a blocker too, which is what
  // reduced finite lights to certifying nothing at all in a furnished room.
  assert.equal(svoFinitePageBeamIntersectsAabb(receiver,
    { minimum: [20, 10, 0], maximum: [22, 12, 2] }, light), false);
});
