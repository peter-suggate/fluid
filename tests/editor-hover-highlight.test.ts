import assert from "node:assert/strict";
import test from "node:test";

import { hoverSceneAt } from "../lib/editor-hover";
import { sceneryHighlightRange } from "../lib/editor-scenery";
import { cloneScene, defaultScene, type SceneDescription } from "../lib/model";
import { getScenePreset } from "../lib/scenes";
import { svoOwnerIdForEnvironmentProxy } from "../lib/svo-scene-primitives";
import { buildEnvironmentProxyCatalog } from "../lib/voxel-environments";
import { svoDrySceneShader } from "../lib/webgpu-svo-dry-scene";

/**
 * The hover outline.
 *
 * Hovering used to produce a seven-pixel chip in the DOM and nothing in the
 * scene, so on a crowded set there was no way to tell which of four overlapping
 * mushrooms a click would take. The rim answers that in the render itself,
 * which is the only place it can be answered without lying about occlusion.
 */

/** World bounds of one described object, so a ray can be aimed straight at it. */
function boundsOf(scene: SceneDescription, nodeId: string) {
  const catalog = buildEnvironmentProxyCatalog(scene, scene.environment ?? "default");
  const span = catalog.spans.find((candidate) => candidate.nodeId === nodeId)!;
  const min = { x: Infinity, y: Infinity, z: Infinity };
  const max = { x: -Infinity, y: -Infinity, z: -Infinity };
  for (const primitive of catalog.primitives.slice(span.from, span.to)) {
    for (const axis of ["x", "y", "z"] as const) {
      min[axis] = Math.min(min[axis], primitive.aabb_m.min[axis]);
      max[axis] = Math.max(max[axis], primitive.aabb_m.max[axis]);
    }
  }
  return { min, max };
}

test("hovering scenery names the described object, not the primitive under the ray", () => {
  const scene = getScenePreset("garden-svo-lighting").create();
  const bounds = boundsOf(scene, "lamppost");
  const hover = hoverSceneAt(scene, [], {
    origin: { x: .5 * (bounds.min.x + bounds.max.x), y: bounds.max.y + .2, z: .5 * (bounds.min.z + bounds.max.z) },
    direction: { x: 0, y: -1, z: 0 },
  });
  assert.equal(hover?.kind, "scenery");
  assert.equal(hover?.sceneryNodeId, "lamppost");
});

test("a hovered object resolves to the contiguous owner range the shader compares against", () => {
  const scene = getScenePreset("garden-svo-lighting").create();
  const catalog = buildEnvironmentProxyCatalog(scene, "garden");
  const range = sceneryHighlightRange(scene, "lamppost");
  assert.ok(range);
  assert.equal(range!.last - range!.first, 3, "the lamppost is four primitives");

  // The published range is in the numbering the *shader* reads back off a hit,
  // not the catalog's dense index. Those differ by the rigid-body count, and a
  // range in the wrong one still looks plausible — it is contiguous, it is the
  // right length — while lighting the object next door.
  assert.ok(scene.rigidBodies.length > 0, "the garden has bodies, so the two numberings differ here");
  const owners = catalog.primitives
    .filter((primitive) => {
      const ownerId = svoOwnerIdForEnvironmentProxy(scene, primitive);
      return ownerId >= range!.first && ownerId <= range!.last;
    })
    .map(({ key }) => key);
  // Contiguity is what lets the shader answer with two comparisons instead of a
  // per-primitive lookup, and it holds because expansion is depth-first in
  // document order.
  assert.deepEqual(owners, [
    "garden/lamppost/base", "garden/lamppost/pole", "garden/lamppost/lantern", "garden/lamppost/cap",
  ]);

  const tree = sceneryHighlightRange(scene, "tree-hero")!;
  assert.ok(tree.last - tree.first > 20, "a grown tree is one range too, however many parts it grew");
  assert.equal(sceneryHighlightRange(scene, "no-such-node"), undefined);
});

test("scenery answers the cursor only where a click would select it", () => {
  const scene = getScenePreset("garden-svo-lighting").create();
  const bounds = boundsOf(scene, "lamppost");
  const ray = {
    origin: { x: .5 * (bounds.min.x + bounds.max.x), y: bounds.max.y + .2, z: .5 * (bounds.min.z + bounds.max.z) },
    direction: { x: 0, y: -1, z: 0 },
  };
  assert.equal(hoverSceneAt(scene, [], ray)?.kind, "scenery");
  // Withheld, the ray falls through to the ground the lamppost stands on rather
  // than reporting a hit no armed tool would act on.
  assert.equal(hoverSceneAt(scene, [], ray, { scenery: false })?.kind, "terrain");
});

// The shader-side rim is gone. It keyed off the voxel's owner id, and a voxel
// no longer names the object it belongs to — the identity word's high half now
// carries the baked surface normal. CPU-side hover resolution is untouched and
// still tested above; only the pixel it used to tint has no way back to a scene
// node. `uniforms.highlight` is written and read by nothing.
test("an environment with no scenery hovered leaves the floor as the answer", () => {
  const scene = cloneScene(defaultScene);
  scene.environment = "default";
  // Straight down the middle of the tank from just above it: the studio's rig
  // is higher than this, so the container floor is what the cursor is over.
  const hover = hoverSceneAt(scene, [], {
    origin: { x: 0, y: scene.container.height_m + .2, z: 0 },
    direction: { x: 0, y: -1, z: 0 },
  });
  assert.equal(hover?.kind, "floor");
});
