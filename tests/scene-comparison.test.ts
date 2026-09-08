import assert from "node:assert/strict";
import test from "node:test";
import { canonicalScene, cloneScene } from "../lib/core/model";
import { createEmptyScene } from "../lib/core/empty-scene";
import { sceneEqualExcept } from "../lib/core/scene-comparison";

test("history comparison matches canonical documents across clones and optional keys", () => {
  const base = createEmptyScene();
  const clone = cloneScene(base);
  assert.equal(sceneEqualExcept(base, clone, ["solidVoxels"]), true);
  clone.solidVoxels = [];
  assert.equal(sceneEqualExcept(base, clone, ["solidVoxels"]), true);
  clone.container.width_m += .1;
  assert.equal(sceneEqualExcept(base, clone, ["solidVoxels"]), false);
  const optional = { ...base, scenery: undefined };
  const serialized = cloneScene(optional);
  assert.equal(sceneEqualExcept(optional, serialized, []), canonicalScene(optional) === canonicalScene(serialized));
});

test("history comparison examines every cloned heightfield value without shallow assumptions", () => {
  const base = createEmptyScene();
  base.terrain = { baseHeight_m: 0, features: [], heightField: { columns: 128, rows: 128,
    heights_m: Array.from({ length: 16384 }, (_, index) => index / 16384) } } as typeof base.terrain;
  const clone = cloneScene(base);
  assert.equal(sceneEqualExcept(base, clone, ["solidVoxels"]), true);
  const field = (clone.terrain as unknown as { heightField: { heights_m: number[] } }).heightField;
  field.heights_m[field.heights_m.length - 1] += .01;
  assert.equal(sceneEqualExcept(base, clone, ["solidVoxels"]), false);
});
