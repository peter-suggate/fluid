import assert from "node:assert/strict";
import test from "node:test";
import "../lib/methods";
import { cloneScene, defaultScene } from "../lib/core/model";
import { bakeProceduralSolid, proceduralSolidField } from "../lib/core/procedural-solid";
import { sceneryConstructionKey } from "../lib/core/scenery-construction-key";
import { sampleSolidWorld, solidWorldContentStamp, solidWorldForScene, withSolidWorldPatches } from "../lib/core/solid-world";
import { buildSvoSolidWorldPlanarBoundaryCatalog, svoRenderResidual } from "../lib/svo/features/scene-publication/svo-planar-boundary";
import { createForestWaterfallScene } from "../lib/core/forest-waterfall-scene";

function fixture() {
  const scene = cloneScene(defaultScene);
  scene.container = { ...scene.container, width_m: 1.6, height_m: 1.6, depth_m: 1.6, fillFraction: 0 };
  scene.voxelDomain = { finestCellSize_m: .1, brickSize_cells: 8 };
  scene.terrain = { baseHeight_m: .2, features: [], grid: { kind: "grid", origin_m: { x: -.8, z: -.8 },
    spacing_m: 1.6, size: { nx: 2, nz: 2 }, heights_m: [.2, .2, .2, .2] } };
  Object.assign(scene, bakeProceduralSolid([16, 16, 16], [.1, .1, .1],
    ([x, y, z]) => Math.hypot(x - 8, y - 9, z - 8) - 4, 2));
  scene.scenery = { palettes: {}, nodes: [{ kind: "terrain-shell", id: "ground", materialModel: "garden-terrain" }] };
  scene.rigidBodies = [];
  return scene;
}

test("scenery-owned sampled solids retain collision and terrain while omitting both duplicate render representations", () => {
  const visible = fixture(), scene = { ...visible,
    scenery: { ...visible.scenery!, solidBoundarySurface: "scenery" as const } };
  assert.equal(solidWorldContentStamp(scene), solidWorldContentStamp(visible));
  assert.notEqual(sceneryConstructionKey(scene), sceneryConstructionKey(visible));
  const world = solidWorldForScene(scene), at = [8, 9, 8] as const;
  assert.ok(sampleSolidWorld(world, at).solidFraction > 0);
  assert.ok(proceduralSolidField(scene));
  const original = svoRenderResidual(visible, world), hidden = svoRenderResidual(scene, world);
  assert.ok(original.solidBoundary);
  assert.equal(hidden.solidBoundary, undefined, "terrain renderer receives no duplicate sampled-boundary shader");
  assert.equal(sampleSolidWorld(hidden.residual, at).solidFraction, 0);
  assert.ok(sampleSolidWorld(hidden.residual, [8, 0, 8]).solidFraction > 0, "native terrain stays visible");
  assert.equal(hidden.residual.patches.length, 0);
  assert.equal(svoRenderResidual(scene, world), hidden, "hidden-boundary snapshots still reuse their residual");
  assert.ok(svoRenderResidual(visible, world).solidBoundary, "toggling ownership restores the original field");
  const catalog = buildSvoSolidWorldPlanarBoundaryCatalog(scene, world.patches);
  assert.equal(catalog.residualExcludedPatchIndices.size, scene.solidBoundary!.patchCount);
  assert.equal(catalog.sources.length, 0, "hidden collider patches cannot become visible planar primitives");
});

test("replacement scenery hides only the valid generated prefix, preserving later edits and invalidated geometry", () => {
  const base = fixture(), scene = { ...base,
    scenery: { ...base.scenery!, solidBoundarySurface: "scenery" as const } };
  const world = solidWorldForScene(scene), hidden = svoRenderResidual(scene, world);
  const patch = { operation: "fill" as const, minimum: [1, 6, 1] as const,
    maximumExclusive: [3, 8, 3] as const, materialId: 9 };
  const edited = { ...scene, solidVoxels: [...scene.solidVoxels, patch] };
  const next = svoRenderResidual(edited, withSolidWorldPatches(world, [patch]), hidden);
  assert.equal(sampleSolidWorld(next.residual, [2, 7, 2]).materialId, 9);
  assert.equal(sampleSolidWorld(next.residual, [8, 9, 8]).solidFraction, 0);
  assert.deepEqual(next.residual.patches, [patch]);
  const invalid = { ...scene, solidVoxels: scene.solidVoxels.map((p, i) => i ? p : { ...p, materialId: 9 }) };
  assert.equal(proceduralSolidField(invalid), undefined);
  const invalidWorld = solidWorldForScene(invalid);
  assert.equal(svoRenderResidual(invalid, invalidWorld).residual.patches.length, invalid.solidVoxels.length,
    "a stale prefix is never silently hidden");
});

test("forest waterfall publishes native terrain with Blender-owned rock surfaces and unchanged live collision geometry", () => {
  const scene = createForestWaterfallScene(), world = solidWorldForScene(scene);
  assert.equal(scene.scenery!.solidBoundarySurface, "scenery");
  assert.ok(scene.terrain!.grid);
  assert.ok(scene.solidBoundary!.bricks.length > 0);
  const residual = svoRenderResidual(scene, world);
  assert.equal(residual.solidBoundary, undefined);
  assert.equal(residual.residual.patches.length, 0);
  assert.ok(residual.residual.ground);
  assert.ok(proceduralSolidField(scene), "the live solver retains the exact mesh-derived ledges");
});
