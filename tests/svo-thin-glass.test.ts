import assert from "node:assert/strict";
import test from "node:test";

import {
  canonicalSvoThinGlassPane,
  evaluateSvoThinGlassOptics,
  intersectSvoThinGlassPane,
  packSvoThinGlassPanes,
  SVO_THIN_GLASS_FEATURES,
  SVO_THIN_GLASS_RECORD_STRIDE_BYTES,
  SVO_THIN_GLASS_REFINEMENT,
  svoThinGlassBounds,
  svoThinGlassMediaBoundary,
  svoThinGlassWGSL,
  unpackSvoThinGlassPanes,
  type SvoThinGlassPane,
} from "../lib/svo-thin-glass";
import { resolveSvoMediumBoundaryGroup } from "../lib/svo-media";

const pane: SvoThinGlassPane = {
  paneId: 42,
  materialId: 1,
  ownerId: 19,
  center_m: [1, 2, 3],
  halfExtent_m: [2, 1],
  thickness_m: 0.012,
  indexOfRefraction: 1.52,
  absorption_mInv: [0.8, 0.2, 0.1],
  edgeEpsilon_m: 1e-5,
  maximumOpticalPath_m: 0.4,
};

test("packed pane ABI preserves stable owner, material, geometry, and optical semantics", () => {
  assert.equal(SVO_THIN_GLASS_RECORD_STRIDE_BYTES, 80);
  const packed = packSvoThinGlassPanes([pane]);
  assert.equal(packed.byteLength, 80);
  const unpacked = unpackSvoThinGlassPanes(packed)[0];
  assert.equal(unpacked.paneId, pane.paneId);
  assert.equal(unpacked.materialId, pane.materialId);
  assert.equal(unpacked.ownerId, pane.ownerId);
  assert.deepEqual(unpacked.center_m, pane.center_m);
  assert.ok(Math.abs(unpacked.thickness_m - pane.thickness_m) < 1e-8);
  assert.ok(Math.abs(unpacked.indexOfRefraction - (pane.indexOfRefraction ?? 0)) < 1e-6);
  assert.throws(() => canonicalSvoThinGlassPane({ ...pane, materialId: 0 }), /nonzero uint16/);
});

test("oriented finite panes return two-sided normals with normal-incidence identity", () => {
  const front = intersectSvoThinGlassPane(pane, { origin_m: [1, 2, 5], direction: [0, 0, -2], tMax_m: 4 });
  const back = intersectSvoThinGlassPane(pane, { origin_m: [1, 2, 1], direction: [0, 0, 1], tMax_m: 4 });
  assert.ok(front && back);
  assert.equal(front?.t_m, 2);
  assert.deepEqual(front?.geometricNormal, [0, 0, 1]);
  assert.equal(front?.frontFacing, true);
  assert.deepEqual(back?.geometricNormal, [0, 0, -1]);
  assert.equal(back?.frontFacing, false);
  assert.equal(front?.paneId, 42);
  assert.equal(front?.materialId, 1);
  assert.equal(front?.ownerId, 19);

  const rotated = { ...pane, orientation: { w: Math.SQRT1_2, x: 0, y: Math.SQRT1_2, z: 0 } };
  const side = intersectSvoThinGlassPane(rotated, { origin_m: [4, 2, 3], direction: [-1, 0, 0], tMax_m: 5 });
  assert.ok(side);
  assert.ok(Math.abs((side?.authoredNormal[0] ?? 0) - 1) < 1e-12);
});

test("grazing intersections are finite and optical length is explicitly capped", () => {
  const direction = [1, 0, -1e-3] as const;
  const hit = intersectSvoThinGlassPane({ ...pane, center_m: [0, 0, 0], halfExtent_m: [2, 2] }, {
    origin_m: [-1, 0, 0.001], direction, tMax_m: 3,
  });
  assert.ok(hit);
  assert.equal(hit?.opticalPath_m, pane.maximumOpticalPath_m);
  assert.equal(intersectSvoThinGlassPane(pane, {
    origin_m: [1, 2, 4], direction: [1, 0, -1e-8], tMax_m: 10,
  }), null, "bounded parallel threshold rejects unstable near-coplanar rays");
});

test("edge and corner ownership is deterministic and inclusive within one epsilon", () => {
  const hitAt = (x: number, y: number) => intersectSvoThinGlassPane(pane, {
    origin_m: [1 + x, 2 + y, 4], direction: [0, 0, -1], tMax_m: 2,
  });
  assert.equal(hitAt(0, 0)?.featureId, SVO_THIN_GLASS_FEATURES.face);
  assert.equal(hitAt(2, 0)?.featureId, SVO_THIN_GLASS_FEATURES.edgeX);
  assert.equal(hitAt(0, 1)?.featureId, SVO_THIN_GLASS_FEATURES.edgeY);
  assert.equal(hitAt(2, 1)?.featureId, SVO_THIN_GLASS_FEATURES.corner);
  assert.equal(hitAt(2 + 0.5e-5, 1 + 0.5e-5)?.featureId, SVO_THIN_GLASS_FEATURES.corner);
  assert.equal(hitAt(2 + 2e-5, 0), null);
});

test("thickness, custom IOR, and absorption map into the thin-wall media contract once", () => {
  const hit = intersectSvoThinGlassPane(pane, { origin_m: [1, 2, 5], direction: [0, 0, -1], tMax_m: 4 });
  assert.ok(hit);
  const optics = evaluateSvoThinGlassOptics(pane, hit!, "water");
  assert.equal(optics.opticalPath_m, pane.thickness_m);
  assert.ok(optics.netTransmittance[0] < optics.netTransmittance[1]);
  const boundary = svoThinGlassMediaBoundary(pane, hit!);
  assert.equal(boundary.thinWall, true);
  assert.equal(boundary.thinWallIor, 1.52);
  assert.equal(boundary.boundaryId, 42);
  assert.deepEqual(boundary.thinWallTint, optics.absorptionTint);
});

test("thin pane composes with coincident water exit ordering without entering glass", () => {
  const hit = intersectSvoThinGlassPane(pane, { origin_m: [1, 2, 5], direction: [0, 0, -1], tMax_m: 4 });
  assert.ok(hit);
  const glass = svoThinGlassMediaBoundary(pane, hit!);
  const waterExit = { t_m: glass.t_m + 0.5e-5, medium: "water" as const, geometricNormal: [0, 0, -1] as const };
  const resolved = resolveSvoMediumBoundaryGroup(["air", "water"], [glass, waterExit], [0, 0, -1], 1e-5);
  assert.equal(resolved.from, "water");
  assert.equal(resolved.to, "air");
  assert.deepEqual(resolved.nextStack, ["air"]);
  assert.equal(resolved.thinWalls.length, 1);
});

test("conservative bounds cover the oriented slab and require bounded analytic refinement", () => {
  const bounds = svoThinGlassBounds(pane, [0.1, 0.2, 0.4]);
  assert.deepEqual(bounds.exact_m.minimum, [-1, 1, 2.994]);
  assert.deepEqual(bounds.exact_m.maximum, [3, 3, 3.006]);
  assert.ok(bounds.conservative_m.minimum.every((value, axis) => value < bounds.exact_m.minimum[axis]));
  assert.ok(bounds.conservative_m.maximum.every((value, axis) => value > bounds.exact_m.maximum[axis]));
  assert.equal(bounds.maximumRefinementIterations, 6);
  assert.equal(bounds.analyticFinalHitRequired, true);
  assert.equal(SVO_THIN_GLASS_REFINEMENT.analyticFinalHitRequired, true);
  assert.ok(Math.abs(bounds.refinementTolerance_m - 1e-4) < 1e-12);
});

test("WGSL reference is binding-free, finite, two-sided, and refinement-bounded", () => {
  assert.match(svoThinGlassWGSL, /struct SvoThinGlassRecord/);
  assert.match(svoThinGlassWGSL, /fn svoThinGlassIntersect/);
  assert.match(svoThinGlassWGSL, /normalCosine<minimumRayCosine/);
  assert.match(svoThinGlassWGSL, /let normal=select\(-authored,authored,front\)/);
  assert.match(svoThinGlassWGSL, /SVO_THIN_GLASS_FEATURE_CORNER/);
  assert.match(svoThinGlassWGSL, /SVO_THIN_GLASS_MAX_REFINEMENT:u32=6u/);
  assert.match(svoThinGlassWGSL, /fn svoThinGlassBounds/);
  assert.match(svoThinGlassWGSL, /fn svoThinGlassOptics/);
  assert.match(svoThinGlassWGSL, /fn svoThinGlassMaterialId/);
  assert.match(svoThinGlassWGSL, /fn svoThinGlassOwnerId/);
  assert.doesNotMatch(svoThinGlassWGSL, /@group|@binding/);
});
