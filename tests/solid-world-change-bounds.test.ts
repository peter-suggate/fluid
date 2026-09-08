import assert from "node:assert/strict";
import test from "node:test";
import { createEmptyScene } from "../lib/core/empty-scene";
import { sceneWithSolidStroke, solidWorldForScene } from "../lib/core/solid-world";
import { solidWorldChangeBounds } from "../lib/core/solid-world-change-bounds";

test("an eight-voxel Box allocates only its narrow changed footprint and undo allocates nothing", () => {
  const scene = createEmptyScene();
  const base = solidWorldForScene(scene);
  const edited = sceneWithSolidStroke(scene, [{ operation: "fill", minimum: [16, 2, 16], maximumExclusive: [17, 10, 17], materialId: 2 }]);
  const next = solidWorldForScene(edited);
  const bounds = solidWorldChangeBounds(scene, base, next);
  assert.equal(bounds.addedBounds.length, 2, "one tight bound for each of two touched eight-cell pages");
  const volume = bounds.addedBounds.reduce((sum, b) => sum + b.maximum.reduce((v, maximum, axis) => v * (maximum - b.minimum[axis]!), 1), 0);
  assert.ok(Math.abs(volume - 8 * scene.voxelDomain.finestCellSize_m ** 3) < 1e-12,
    "allocating entire pages would multiply work by128 for this narrow stroke");
  const undo = solidWorldChangeBounds(scene, next, base);
  assert.deepEqual(undo.dirtyBounds, bounds.dirtyBounds);
  assert.deepEqual(undo.addedBounds, []);
  assert.deepEqual(solidWorldChangeBounds(scene, next, next), { dirtyBounds: [], addedBounds: [] });
});

test("negative coordinates and material-only voxel changes remain local", () => {
  const scene = createEmptyScene();
  const filled = sceneWithSolidStroke(scene, [{ operation: "fill", minimum: [-3, -2, -1], maximumExclusive: [-2, -1, 0], materialId: 2 }]);
  const painted = sceneWithSolidStroke(filled, [{ operation: "fill", minimum: [-3, -2, -1], maximumExclusive: [-2, -1, 0], materialId: 3 }]);
  const bounds = solidWorldChangeBounds(scene, solidWorldForScene(filled), solidWorldForScene(painted));
  assert.equal(bounds.dirtyBounds.length, 1); assert.equal(bounds.addedBounds.length, 1);
  assert.deepEqual(bounds.dirtyBounds, bounds.addedBounds);
  for (let axis = 0; axis < 3; axis++) {
    assert.ok(Math.abs(bounds.dirtyBounds[0]!.maximum[axis]! - bounds.dirtyBounds[0]!.minimum[axis]!
      - scene.voxelDomain.finestCellSize_m) < 1e-12);
  }
});
