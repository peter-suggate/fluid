import assert from "node:assert/strict";
import test from "node:test";

import { heroGardenCloudTree } from "../lib/hero-garden-tree";
import { cloneScene, defaultScene } from "../lib/model";
import { addSceneryNode } from "../lib/editor-scenery";
import { findSceneryNode } from "../lib/scenery-edit";
import {
  canopyDials,
  canopyPads,
  sceneCanopyPads,
  sceneCanopyQuery,
  withCanopyDials,
  withSceneCanopyQuery,
} from "../lib/tree-canopy-controls";
import { getScenePreset } from "../lib/scenes";
import { parseQueryState, serializeQueryState } from "../lib/url-state";
import type { SceneryRecursiveShapeNode } from "../lib/scenery-graph";

/**
 * The canopy dials are a projection of the six-number density field onto the
 * three moves that stay tree-like. What matters is that the projection is
 * honest: the authored hero canopy reads back as an interior dial position (so
 * selecting the tree changes nothing), setting a dial reads back as itself,
 * and every reachable setting stays inside the document validator's ranges —
 * a slider that can author an unpublishable node is a trap, not a control.
 */

function treeScene() {
  return addSceneryNode(cloneScene(defaultScene), heroGardenCloudTree());
}

test("the hero tree exposes its canopy pads through the scene", () => {
  const scene = treeScene();
  const pads = sceneCanopyPads(scene, "tree");
  assert.equal(pads.length, 1);
  assert.equal(pads[0]!.id, "tree/foliage/canopy");
  // A non-foliage prop offers no pads, which is what keeps leaf dials off
  // lanterns in the viewport.
  assert.equal(sceneCanopyPads(scene, "no-such-node").length, 0);
});

test("the authored canopy reads back as interior dial positions", () => {
  const [pad] = canopyPads(heroGardenCloudTree());
  const dials = canopyDials(pad!);
  // Interior, not pinned: a dial born at 0 or 1 can only be moved one way.
  for (const value of Object.values(dials)) {
    assert.ok(value > 0.1 && value < 0.9, `dial pinned at ${value}`);
  }
});

test("dials round-trip through the document", () => {
  const set = { coverage: 0.25, clumpSize: 0.8, breakup: 0.6 };
  const edited = withCanopyDials(treeScene(), "tree", set);
  const [pad] = sceneCanopyPads(edited, "tree");
  const read = canopyDials(pad!);
  assert.ok(Math.abs(read.coverage - set.coverage) < 1e-9);
  assert.ok(Math.abs(read.clumpSize - set.clumpSize) < 1e-9);
  assert.ok(Math.abs(read.breakup - set.breakup) < 1e-9);
});

test("every reachable setting satisfies the density validator's ranges", () => {
  for (const coverage of [0, 0.5, 1]) for (const clumpSize of [0, 1]) for (const breakup of [0, 1]) {
    const edited = withCanopyDials(treeScene(), "tree", { coverage, clumpSize, breakup });
    const [pad] = sceneCanopyPads(edited, "tree");
    const density = pad!.form.density;
    const unit = (value: number) => Number.isFinite(value) && value >= 0 && value <= 1;
    assert.ok(density.clusterPeriod_m > 0 && Number.isFinite(density.clusterPeriod_m));
    assert.ok(density.dotSpacing_m > 0 && Number.isFinite(density.dotSpacing_m));
    assert.ok(unit(density.threshold), `threshold ${density.threshold}`);
    assert.ok(unit(density.clusterWeight), `clusterWeight ${density.clusterWeight}`);
    assert.ok(unit(density.detailWeight), `detailWeight ${density.detailWeight}`);
    assert.ok(unit(density.interiorBias), `interiorBias ${density.interiorBias}`);
    assert.ok(density.clusterWeight + density.detailWeight > 0);
  }
});

test("editing dials keeps the envelope, the seed and the leaf spacing", () => {
  const edited = withCanopyDials(treeScene(), "tree", { coverage: 0, clumpSize: 1, breakup: 1 });
  const before = canopyPads(heroGardenCloudTree())[0]!;
  const after = sceneCanopyPads(edited, "tree")[0]!;
  assert.deepEqual(after.form.radii_m, before.form.radii_m);
  assert.equal(after.seed, before.seed);
  assert.equal(after.form.density.dotSpacing_m, before.form.density.dotSpacing_m);
  // The structural sweeps are untouched — the dials own foliage, not wood.
  const tree = findSceneryNode(edited, "tree");
  assert.ok(tree && tree.kind === "group");
});

test("canopy dials round-trip through the compact query form", () => {
  const scene = withCanopyDials(treeScene(), "tree", { coverage: 0.31, clumpSize: 0.74, breakup: 0.12 });
  const query = sceneCanopyQuery(scene);
  assert.equal(query, "tree~0.31,0.74,0.12");
  const restored = withSceneCanopyQuery(treeScene(), query);
  const dials = canopyDials(sceneCanopyPads(restored, "tree")[0]!);
  assert.ok(Math.abs(dials.coverage - 0.31) < 1e-9);
  assert.ok(Math.abs(dials.clumpSize - 0.74) < 1e-9);
  assert.ok(Math.abs(dials.breakup - 0.12) < 1e-9);
  // A carried value re-reads as the same string, so applying it never makes
  // the scene look edited on the next diff.
  assert.equal(sceneCanopyQuery(restored), query);
  // Junk keys neither throw nor touch the document.
  assert.equal(withSceneCanopyQuery(scene, "no-such-node~1,1,1;garbage"), scene);
});

test("an edited canopy rides the URL and an untouched one stays out of it", () => {
  const method = { methodId: "octree" as const, quality: "balanced" as const, overrides: {} };
  const preset = getScenePreset("hero-garden-hose");
  const pristine = new URLSearchParams(serializeQueryState("", {
    presetId: "hero-garden-hose", scene: preset.create(),
  }, method));
  assert.equal(pristine.get("canopy"), null, "authored dials must not dirty the URL");

  const edited = withCanopyDials(preset.create(), "tree", { coverage: 0.2, clumpSize: 0.9, breakup: 0.5 });
  const query = serializeQueryState("", { presetId: "hero-garden-hose", scene: edited }, method);
  assert.equal(new URLSearchParams(query).get("canopy"), "tree~0.2,0.9,0.5");

  const rehydrated = parseQueryState(`?${query}`);
  const dials = canopyDials(sceneCanopyPads(rehydrated.scene, "tree")[0]!);
  assert.ok(Math.abs(dials.coverage - 0.2) < 1e-9);
  assert.ok(Math.abs(dials.clumpSize - 0.9) < 1e-9);
  assert.ok(Math.abs(dials.breakup - 0.5) < 1e-9);
});

test("canopy dials survive a lattice re-author carried by the same URL", () => {
  // The user's exact loop: thin the tree, then change the environment level.
  // The link carries both, and hydration applies the dials to the document the
  // lattice actually rebuilt.
  const base = getScenePreset("hero-garden-hose").create();
  const halfLattice = {
    ...base.voxelDomain,
    detailCellSize_m: (base.voxelDomain.detailCellSize_m ?? base.voxelDomain.finestCellSize_m) / 2,
  };
  const parsed = parseQueryState(
    `?scene=hero-garden-hose&canopy=${encodeURIComponent("tree~0.15,0.8,0.4")}`
    + `&scene.voxelDomain=${encodeURIComponent(JSON.stringify(halfLattice))}`);
  assert.equal(parsed.scene.voxelDomain.detailCellSize_m, halfLattice.detailCellSize_m,
    "the lattice request must have re-authored the document");
  const dials = canopyDials(sceneCanopyPads(parsed.scene, "tree")[0]!);
  assert.ok(Math.abs(dials.coverage - 0.15) < 1e-9, `coverage ${dials.coverage}`);
  assert.ok(Math.abs(dials.clumpSize - 0.8) < 1e-9, `clumpSize ${dials.clumpSize}`);
  assert.ok(Math.abs(dials.breakup - 0.4) < 1e-9, `breakup ${dials.breakup}`);
});

test("a split crown moves all of its pads together", () => {
  const tree = heroGardenCloudTree();
  const canopy = canopyPads(tree)[0]!;
  const child = (suffix: string): SceneryRecursiveShapeNode => ({
    ...canopy,
    id: `${canopy.id}/${suffix}`,
    children: undefined,
  });
  const split = {
    ...tree,
    children: tree.children.map((group) => group.id !== "tree/foliage" || group.kind !== "group"
      ? group
      : { ...group, children: [{ ...canopy, children: [child("a"), child("b")] }] }),
  };
  const scene = addSceneryNode(cloneScene(defaultScene), split);
  assert.equal(sceneCanopyPads(scene, "tree").length, 2);
  const edited = withCanopyDials(scene, "tree", { coverage: 0.9, clumpSize: 0.2, breakup: 0.4 });
  const pads = sceneCanopyPads(edited, "tree");
  assert.equal(pads.length, 2);
  const [first, second] = pads.map((pad) => canopyDials(pad));
  assert.deepEqual(first, second);
  assert.ok(Math.abs(first!.coverage - 0.9) < 1e-9);
});
