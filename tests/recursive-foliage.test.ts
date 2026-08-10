import assert from "node:assert/strict";
import test from "node:test";

import { validateSceneryGraph, type SceneryRecursiveShapeNode } from "../lib/scenery-graph";
import {
  activeFoliageLeaves,
  foliagePadClusterNode,
  refineFoliageShape,
  refineFoliageShapeToDepth,
} from "../lib/voxel-scenery/recursive-foliage";

const ROOT: SceneryRecursiveShapeNode = {
  kind: "recursive-shape",
  family: "foliage-pad",
  id: "tree/foliage/crown-a",
  seed: 0x51a9,
  place: { units: "metres", position: { x: 0, y: 0.7, z: 0 } },
  form: {
    radii_m: [0.22, 0.13, 0.18], flatten: 0.72, edgeLobes: 8,
    lobeDepth: 0.48, topBias: 0.64, undersideCut: 0.52, blockJitter: 0.38,
    density: {
      clusterPeriod_m: 0.065, dotSpacing_m: 0.020, threshold: 0.5008,
      clusterWeight: 0.72, detailWeight: 0.28, interiorBias: 0.092,
    },
  },
  split: {
    pattern: "ring", childCount: 4, childScale: 0.61, spread: 0.72,
    overlap: 0.31, verticalBias: 0.42, flattening: 0.58, jitter: 0.28,
  },
  material: { palette: "clay", value: 0.94, surface: "foliage" },
};

test("recursive foliage materializes stable document-owned children", () => {
  const a = refineFoliageShapeToDepth(ROOT, 2);
  const b = refineFoliageShapeToDepth(ROOT, 2);
  assert.deepEqual(a, b);
  assert.equal(activeFoliageLeaves(a).length, 16);
  assert.deepEqual(
    activeFoliageLeaves(a).map(({ id }) => id),
    activeFoliageLeaves(b).map(({ id }) => id),
  );
  assert.deepEqual(JSON.parse(JSON.stringify(a)), a);
});

test("refinement preserves existing descendants until regeneration is explicit", () => {
  const refined = refineFoliageShape(ROOT);
  assert.equal(refineFoliageShape(refined), refined);
  const edited = { ...refined, children: refined.children?.map((child, index) => index === 0
    ? { ...child, form: { ...child.form, flatten: 0.5 } }
    : child) };
  assert.equal(refineFoliageShape(edited), edited);
  assert.equal(edited.children?.[0].form.flatten, 0.5);
});

test("one active foliage leaf compiles to one bounded clustered-noise density record", () => {
  const cluster = foliagePadClusterNode(ROOT);
  assert.equal(cluster.kind, "cluster");
  assert.equal(cluster.field, "noise-foliage");
  assert.equal(cluster.id, ROOT.id);
  assert.equal(cluster.group, ROOT.id);
  assert.deepEqual(cluster.lobe, { x: 0.22, y: 0.13 * 0.72, z: 0.18 });
  assert.equal(cluster.clusterPeriod, ROOT.form.density.clusterPeriod_m);
  assert.equal(cluster.detailPeriod, ROOT.form.density.dotSpacing_m);
  assert.equal(cluster.threshold, ROOT.form.density.threshold);
  assert.equal(cluster.interiorBias, ROOT.form.density.interiorBias);
  assert.ok(cluster.clusterPeriod > cluster.detailPeriod * 3);
  assert.ok(cluster.detailPeriod >= 0.020);
  assert.equal(cluster.clusterWeight + cluster.detailWeight, 1);
  assert.ok(cluster.threshold > 0.45 && cluster.threshold < 0.6);
  assert.ok(cluster.interiorBias > 0);
});

test("graph validation walks descendants and rejects malformed recursive forms", () => {
  const tree = refineFoliageShape(ROOT);
  assert.deepEqual(validateSceneryGraph({
    palettes: { clay: { tint: [1, 0.98, 0.92] } },
    nodes: [
      { kind: "terrain-shell", id: "shell", materialModel: "porcelain" },
      tree,
    ],
  }), []);

  const malformed = {
    ...ROOT,
    form: { ...ROOT.form, radii_m: [0.2, 0, 0.1] },
  } as unknown as SceneryRecursiveShapeNode;
  assert.match(validateSceneryGraph({
    palettes: { clay: { tint: [1, 1, 1] } },
    nodes: [{ kind: "terrain-shell", id: "shell", materialModel: "porcelain" }, malformed],
  }).join("\n"), /radii must be three positive/);
});
