import assert from "node:assert/strict";
import test from "node:test";

import type { SceneryRecursiveShapeNode } from "../lib/scenery-graph";
import { shapeLabRecords, shapeLabWorld } from "../lib/shape-lab/specimens";

test("Shape Lab enumerates and isolates individual recursive foliage leaves", () => {
  const world = shapeLabWorld({ depth: 0 });
  const leaf = world.allNodes.find((node): node is SceneryRecursiveShapeNode =>
    node.kind === "recursive-shape" && !node.children?.length);
  assert.ok(leaf);
  const specimen = world.specimens.find(({ id }) => id === leaf.id);
  assert.ok(specimen);
  assert.equal(specimen.depth, 2, "the shipped hero stops at its authored canopy density volume");
  assert.equal(specimen.parentId, specimen.nodePath.at(-1));
  const isolated = shapeLabRecords(world, specimen.nodeIds);
  assert.equal(isolated.records.length, 1);
  assert.match(isolated.records[0].key, new RegExp(`${leaf.id.replaceAll("/", "\\/")}$`));
  assert.equal(isolated.records[0].descriptor.kind, "smooth-union-cluster");
  assert.ok(isolated.records[0].descriptor.kind === "smooth-union-cluster");
  assert.equal(isolated.records[0].descriptor.packing?.field, "noise-foliage");
});

test("a nested whole-node override keeps ancestors and changes only that leaf", () => {
  const pristine = shapeLabWorld({ depth: 0 });
  const leaf = pristine.allNodes.find((node): node is SceneryRecursiveShapeNode =>
    node.kind === "recursive-shape" && !node.children?.length);
  assert.ok(leaf);
  const before = shapeLabRecords(pristine, [leaf.id]).records;
  const edited: SceneryRecursiveShapeNode = {
    ...leaf,
    form: { ...leaf.form, flatten: leaf.form.flatten * 0.7 },
  };
  const world = shapeLabWorld({ depth: 0, nodeOverrides: { [leaf.id]: edited } });
  const after = shapeLabRecords(world, [leaf.id]).records;
  assert.equal(after.length, 1);
  assert.equal(after[0].key, before[0].key);
  assert.notDeepEqual(after[0].descriptor, before[0].descriptor);
});

test("Shape Lab density edits reach the published noise packing", () => {
  const pristine = shapeLabWorld({ depth: 0 });
  const leaf = pristine.allNodes.find((node): node is SceneryRecursiveShapeNode =>
    node.kind === "recursive-shape" && !node.children?.length);
  assert.ok(leaf);
  const edited: SceneryRecursiveShapeNode = {
    ...leaf,
    form: {
      ...leaf.form,
      density: {
        ...leaf.form.density,
        threshold: 0.63,
        interiorBias: 0.17,
        clusterPeriod_m: 0.12,
        dotSpacing_m: 0.022,
        clusterWeight: 0.66,
        detailWeight: 0.34,
      },
    },
  };
  const world = shapeLabWorld({ depth: 0, nodeOverrides: { [leaf.id]: edited } });
  const descriptor = shapeLabRecords(world, [leaf.id]).records[0].descriptor;
  assert.equal(descriptor.kind, "smooth-union-cluster");
  assert.deepEqual(descriptor.packing, {
    field: "noise-foliage",
    smoothRadius_m: 0,
    seed: leaf.seed,
    clusterPeriod_m: 0.12,
    detailPeriod_m: 0.022,
    threshold: 0.63,
    clusterWeight: 0.66,
    detailWeight: 0.34,
    interiorBias: 0.17,
  });
});

test("assembled tree context contains structure plus the recursive frontier", () => {
  const world = shapeLabWorld({ depth: 0 });
  const tree = world.specimens.find(({ id }) => id === "tree");
  assert.ok(tree);
  const records = shapeLabRecords(world, tree.nodeIds).records;
  assert.equal(records.length, 7, "six structural sweeps plus one clustered-noise canopy");
  assert.ok(records.some(({ key }) => key.includes("tree/structure/trunk")));
  assert.ok(records.some(({ key }) => key.includes("tree/foliage/canopy")));
  assert.equal(records.filter(({ descriptor }) =>
    descriptor.kind === "smooth-union-cluster" && descriptor.packing?.field === "noise-foliage").length, 1);
});
