import assert from "node:assert/strict";
import test from "node:test";
import { authoredFluidEdits, authoredFluidVolumes, packAuthoredFluidVolume } from "../lib/core/authored-fluid-edit";
import { cloneScene, defaultScene } from "../lib/core/model";
import { editorFluidLattice, fluidBrickCenter } from "../lib/core/editor-fluid";

test("painted rectangular bodies collapse to one resident shape operation", () => {
  const scene = cloneScene(defaultScene);
  scene.fluid.initialBrickSeeds_m = [];
  scene.fluid.initialBrickSeedsAdditive = false;
  scene.fluid.initialLiquidVolumes = [];
  const lattice = editorFluidLattice(scene);
  for (let z = 0; z < 2; z++) for (let y = 0; y < 2; y++) for (let x = 0; x < 2; x++) {
    scene.fluid.initialBrickSeeds_m.push(fluidBrickCenter(lattice, { x, y, z }));
  }
  const volumes = authoredFluidVolumes(scene);
  assert.equal(volumes.length, 1, "eight painted bricks become one box");
  assert.equal(volumes[0]!.shape, "box");
  const same = structuredClone(scene);
  assert.deepEqual(authoredFluidEdits(scene, same), [], "worker clones must not replay fluid");
  same.fluid.density_kg_m3 += 10;
  assert.deepEqual(authoredFluidEdits(scene, same), [], "scalar edits leave occupancy alone");
});

test("moving one analytic volume leaves other authored bodies out of the edit list", () => {
  const before = cloneScene(defaultScene);
  before.fluid.initialLiquidVolumes = [{ shape: "sphere", center_m: { x: 0, y: 1, z: 0 }, radius_m: .1 }];
  const after = structuredClone(before);
  const sphere = after.fluid.initialLiquidVolumes![0]!;
  assert.equal(sphere.shape, "sphere");
  if (sphere.shape === "sphere") sphere.center_m.x = .4;
  const edits = authoredFluidEdits(before, after);
  assert.deepEqual(edits.map(e => [e.operation, e.volume.shape]), [["remove", "sphere"], ["add", "sphere"]]);
  assert.throws(() => packAuthoredFluidVolume({ shape: "sphere", center_m: { x: NaN, y: 0, z: 0 }, radius_m: 1 }, "add", .1, [0, 0, 0]), RangeError);
});
