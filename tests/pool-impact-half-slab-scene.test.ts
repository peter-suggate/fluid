import assert from "node:assert/strict";
import test from "node:test";
import { getSceneDefinition } from "../lib/core/scenes";
import { sceneDocument } from "../lib/core/scene-definition";
import { sceneLatticeDimensions } from "../lib/core/scene-lattice";
import { initialLiquidVolumesSignedDistance } from "../lib/core/initial-fluid";
import { labAuthoredScene } from "../advance-lab/lab-scenes";

test("half-pool slab keeps the XY physics and seeds the same disk through its depth", () => {
  const id = "coarse-first-pool-impact-half-slab";
  const slab = sceneDocument(getSceneDefinition(id));
  const original = sceneDocument(getSceneDefinition("coarse-first-pool-impact-half"));
  assert.deepEqual(sceneLatticeDimensions(slab), [64, 48, 8]);
  assert.equal(slab.container.width_m, original.container.width_m);
  assert.equal(slab.container.height_m, original.container.height_m);
  assert.equal(slab.container.fillFraction, original.container.fillFraction);
  assert.equal(slab.container.fluidWallMode, "free-slip");
  assert.equal(slab.container.top, "closed");
  assert.deepEqual(slab.numerics, original.numerics);
  assert.deepEqual(labAuthoredScene(id)?.document, slab);
  for (const [x, y] of [[0, 3.65], [.8, 3.65], [1.2, 3.65], [0, 2.8]]) {
    const expected = Math.hypot(x!, y! - 3.65) - 1;
    for (const z of [-.4, -.35, 0, .35, .4]) {
      const distance = initialLiquidVolumesSignedDistance(slab, { x: x!, y: y!, z });
      assert.notEqual(distance, undefined);
      assert.ok(Math.abs(distance! - expected) < 1e-12);
    }
  }
});
