import assert from "node:assert/strict";
import test from "node:test";
import { createSolidWorld, SOLID_WORLD_TERRAIN_MATERIAL_ID } from "../lib/core/solid-world";
import { retainedSceneDensity } from "../lib/methods/adaptive-mass/sparse-cm12-retained-scene-density";
import { compileRetainedScenePreparationCache } from "../lib/methods/adaptive-mass/sparse-cm12-retained-preparation-cache";

const h = .125, dimensions = [8, 8, 8] as const;
const field = retainedSceneDensity({ generation: 1, transitionWidth: h,
  domain: { lower: [-.5, 0, -.5], upper: [.5, 1, .5] },
  primitives: [{ kind: "quadratic-height", center: [0, h / 2, 0], curvature: [0, 0, 0] }] });

test("frozen preparation snapshots reuse all physical integrals across native repartitioning", () => {
  const world = createSolidWorld();
  const initial = compileRetainedScenePreparationCache(field, dimensions, h, world, { rigid: true });
  const cloned = structuredClone(initial);
  const prepared = compileRetainedScenePreparationCache(field, dimensions, h, world, { previous: cloned, rigid: true });
  assert.equal(prepared.unrestrictedMeans, cloned.unrestrictedMeans);
  assert.equal(prepared.openMeans, cloned.openMeans);
  assert.equal(prepared.subcellMoments, cloned.subcellMoments);
  assert.notEqual(prepared.unrestrictedMeans.buffer, initial.unrestrictedMeans.buffer, "worker owns its cloned storage");
  assert.equal(initial.unrestrictedMeans[0], .5);
  assert.ok(Object.isFrozen(prepared));
});

test("a changed static support reuses the seed field and preserves independent exact open moments", () => {
  const initial = compileRetainedScenePreparationCache(field, dimensions, h, createSolidWorld(), { rigid: true });
  const world = createSolidWorld([{ operation: "fill", minimum: [0, 0, 0], maximumExclusive: [1, 1, 1],
    materialId: SOLID_WORLD_TERRAIN_MATERIAL_ID }]);
  world.pages[0].solidFraction[0] = 128;
  const edited = compileRetainedScenePreparationCache(field, dimensions, h, world, { previous: initial, rigid: true });
  assert.equal(edited.unrestrictedMeans, initial.unrestrictedMeans);
  assert.notEqual(edited.openMeans, initial.openMeans);
  assert.equal(edited.subcellMoments!.receipt.recomputedCells, 1);
  const fraction = 127 / 255;
  assert.ok(Math.abs(edited.openMeans.effectiveMeans[0] - fraction ** 2 / 2) < 1e-8);
  assert.ok(Math.abs(edited.openMeans.openFractions[0] - fraction) < 1e-8);
  assert.equal(initial.openMeans.effectiveMeans[0], .5);
  assert.equal(initial.subcellMoments!.openVolumes[0], .125);
});

test("preparation cannot reuse a same-generation cache from another numeric field or lattice", () => {
  const world = createSolidWorld(), initial = compileRetainedScenePreparationCache(field, dimensions, h, world);
  const other = retainedSceneDensity({ ...field, primitives: [] });
  assert.throws(() => compileRetainedScenePreparationCache(other, dimensions, h, world, { previous: initial }), /numeric field and physical lattice/);
  assert.throws(() => compileRetainedScenePreparationCache(field, [16, 16, 16], h / 2, world, { previous: initial }), /numeric field and physical lattice/);
  const malformed = { ...initial, unrestrictedMeans: new Float32Array(1) };
  assert.throws(() => compileRetainedScenePreparationCache(field, dimensions, h, world, { previous: malformed }), /numeric field and physical lattice/);
});

test("later rigid preparation adds subcell geometry without recomputing unrestricted or open moments", () => {
  const world = createSolidWorld(), initial = compileRetainedScenePreparationCache(field, dimensions, h, world);
  const rigid = compileRetainedScenePreparationCache(field, dimensions, h, world, { previous: initial, rigid: true });
  assert.equal(rigid.unrestrictedMeans, initial.unrestrictedMeans);
  assert.equal(rigid.openMeans, initial.openMeans);
  assert.equal(initial.subcellMoments, undefined);
  assert.equal(rigid.subcellMoments!.seedAmounts[0], .09375);
});
