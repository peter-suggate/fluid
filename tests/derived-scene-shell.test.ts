import assert from "node:assert/strict";
import test from "node:test";

import { sampleSolidWorld, solidWorldForScene } from "../lib/core/solid-world";
import { solidVoxelEditsForScene } from "../lib/core/scene-lattice";
import {
  createDeepPowerHydrostaticScene,
  createHighResolutionDamBreakScene,
  createLargePowerDamBreakScene,
  createMinimalPowerDamBreakScene,
  createMinimalPowerDamBreak32Scene,
  createMinimalPowerDamBreak64Scene,
  createSparseCM12LongDamBreakScene,
} from "../lib/core/scenes";

test("the source Mini16 dam carries its closed solid shell", () => {
  const world = solidWorldForScene(createMinimalPowerDamBreakScene());
  assert.equal(sampleSolidWorld(world, [-1, 8, 8]).solidFraction, 1);
  assert.equal(sampleSolidWorld(world, [16, 8, 8]).solidFraction, 1);
  assert.equal(sampleSolidWorld(world, [8, -1, 8]).solidFraction, 1);
  assert.equal(sampleSolidWorld(world, [8, 16, 8]).solidFraction, 1);
  assert.equal(sampleSolidWorld(world, [8, 8, -1]).solidFraction, 1);
  assert.equal(sampleSolidWorld(world, [8, 8, 16]).solidFraction, 1);
  assert.equal(sampleSolidWorld(world, [8, 8, 8]).solidFraction, 0);
});

test("derived dam scenes replace their inherited shell on the final lattice", () => {
  const scenes = [
    createMinimalPowerDamBreak32Scene(),
    createMinimalPowerDamBreak64Scene(),
    createHighResolutionDamBreakScene(),
    createSparseCM12LongDamBreakScene(),
    createLargePowerDamBreakScene(),
    createDeepPowerHydrostaticScene(),
  ];

  for (const scene of scenes) {
    assert.deepEqual(solidVoxelEditsForScene(scene), [],
      `${scene.sceneId} retained a shell from its source scene as authored geometry`);
  }
});

test("Sparse CM12 dam scenes have no hidden 16-cubed vessel", () => {
  const mini64 = solidWorldForScene(createMinimalPowerDamBreak64Scene());
  assert.equal(sampleSolidWorld(mini64, [16, 8, 8]).solidFraction, 0,
    "the inherited 16-cubed right wall must not survive inside the 64-cubed tank");
  assert.equal(sampleSolidWorld(mini64, [64, 8, 8]).solidFraction, 1,
    "the 64-cubed tank must retain its actual right wall");

  const longDam = solidWorldForScene(createSparseCM12LongDamBreakScene());
  assert.equal(sampleSolidWorld(longDam, [16, 8, 8]).solidFraction, 0,
    "the inherited mini-dam wall must not obstruct the long tank");
  assert.equal(sampleSolidWorld(longDam, [192, 8, 8]).solidFraction, 1,
    "the long tank must retain its actual far wall");
});
