import assert from "node:assert/strict";
import test from "node:test";
import { createNbflipSlopeClearance } from "../lib/core/nbflip-slope-clearance-scene";
import { SolidOccupancyMask } from "../lib/core/solid-occupancy-mask";
import { fluidSolidWorldForScene, sampleSolidWorld, sceneWithSolidStroke } from "../lib/core/solid-world";
import { uniformLatticeAddress } from "../lib/methods/uniform/uniform-lattice-address";
import { uniformSolidMaskPages } from "../lib/methods/uniform/uniform-mixed-solid.wgsl";

test("native terrain stays out of voxel occupancy, including pages materialized by edits", () => {
  let scene = createNbflipSlopeClearance();
  const mask = new SolidOccupancyMask(uniformSolidMaskPages(uniformLatticeAddress([128, 32, 32], false)));
  const check = (filled: boolean) => {
    const world = fluidSolidWorldForScene(scene);
    mask.update(world);
    assert.ok(sampleSolidWorld(world, [8, 16, 16]).solidFraction > 0, "the terrain intersects this cell");
    assert.equal(sampleSolidWorld(world, [8, 16, 16], true).solidFraction, 0);
    assert.equal(mask.solid(8, 16, 16), false, "fractional terrain has its own capacity and contact");
    assert.equal(mask.solid(8, 4, 16), false, "even deep ground is supplied by the terrain lane");
    assert.equal(mask.solid(8, 20, 16), filled, "authored voxel obstacle is retained");
    assert.equal(sampleSolidWorld(world, [8, 20, 16], true).solidFraction > 0, filled);
    assert.equal(mask.solid(-1, 20, 16), true, "container wall remains closed");
  };
  check(false);
  scene = sceneWithSolidStroke(scene, [{ operation: "fill", minimum: [8, 20, 16], maximumExclusive: [9, 21, 17] }]);
  check(true);
  scene = sceneWithSolidStroke(scene, [{ operation: "clear", minimum: [8, 20, 16], maximumExclusive: [9, 21, 17] }]);
  check(false);
});
