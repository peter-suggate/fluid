import assert from "node:assert/strict";
import test from "node:test";

import { entityAtRay, findEntity } from "../lib/editor-entity-catalog";
import { boxCenter, type EditorEntityContext } from "../lib/editor-entity";
import {
  addSceneryNode,
  createSceneryNodeAt,
  sceneryEntity,
  scenerySelectionId,
} from "../lib/editor-scenery";
import { cloneScene, defaultScene, validateScene, type SceneDescription } from "../lib/model";
import { getScenePreset } from "../lib/scenes";
import { findSceneryNode, sceneSceneryGraph } from "../lib/scenery-edit";
import { buildEnvironmentProxyCatalog } from "../lib/voxel-environments";

/**
 * Scenery is editable the same way everything else is.
 *
 * The behaviour under test is the one the old prop editor could not offer: the
 * objects a *preset* placed — a garden's lantern, a conservatory's pot — are
 * clickable, movable and removable, because they are ordinary nodes of the
 * document rather than the output of a generator that ran once.
 */

const context = (scene: SceneDescription): EditorEntityContext => ({ scene, bodies: [] });

function gardenScene(): SceneDescription {
  return getScenePreset("garden-svo-lighting").create();
}

/** World bounds of everything one described object expanded to. */
function boundsOf(scene: SceneDescription, nodeId: string) {
  const catalog = buildEnvironmentProxyCatalog(scene, scene.environment ?? "default");
  const span = catalog.spans.find((candidate) => candidate.nodeId === nodeId);
  assert.ok(span && span.to > span.from, `${nodeId} publishes primitives`);
  const min = { x: Infinity, y: Infinity, z: Infinity };
  const max = { x: -Infinity, y: -Infinity, z: -Infinity };
  for (const primitive of catalog.primitives.slice(span!.from, span!.to)) {
    for (const axis of ["x", "y", "z"] as const) {
      min[axis] = Math.min(min[axis], primitive.aabb_m.min[axis]);
      max[axis] = Math.max(max[axis], primitive.aabb_m.max[axis]);
    }
  }
  return { min, max };
}

const centreOf = (scene: SceneDescription, nodeId: string) => boxCenter(boundsOf(scene, nodeId));

/**
 * A ray straight down onto one object's crown.
 *
 * From above rather than from the side because the garden is a crowded set and
 * the point here is which object a hit belongs to, not which of two overlapping
 * silhouettes is nearer.
 */
function rayOnto(scene: SceneDescription, nodeId: string) {
  const bounds = boundsOf(scene, nodeId);
  const centre = boxCenter(bounds);
  return { origin: { x: centre.x, y: bounds.max.y + 0.2, z: centre.z }, direction: { x: 0, y: -1, z: 0 } };
}

test("a click on a preset's own scenery selects the whole described object", () => {
  const scene = gardenScene();
  // The lamppost is four primitives under one node. A ray at its lantern must
  // select the lamppost, not the lantern: an object is what the document says
  // it is, not whichever bead the ray happened to reach.
  const hit = entityAtRay(context(scene), rayOnto(scene, "lamppost"));
  assert.equal(hit?.selection.kind, "scenery");
  assert.equal(hit?.selection.id, scenerySelectionId("lamppost"));

  const entity = findEntity(context(scene), hit!.selection);
  assert.ok(entity, "the selection resolves to an entity with handles");
  assert.ok(entity!.handles.length > 0);
  assert.ok(entity!.remove, "a preset's scenery can be removed like anything else");
});

test("a grown tree is one click target, not thirty", () => {
  const scene = gardenScene();
  const catalog = buildEnvironmentProxyCatalog(scene, "garden");
  const span = catalog.spans.find(({ nodeId }) => nodeId === "tree-hero");
  assert.ok(span && span.to - span.from > 20, "the specimen tree expands to many primitives");
  const hit = entityAtRay(context(scene), rayOnto(scene, "tree-hero"));
  assert.equal(hit?.selection.id, scenerySelectionId("tree-hero"));
});

test("moving scenery writes the placement its own units are authored in", () => {
  const scene = gardenScene();
  const entity = findEntity(context(scene), { kind: "scenery", id: scenerySelectionId("lamppost") });
  assert.ok(entity);
  const before = sceneSceneryGraph(scene).nodes.find(({ id }) => id === "lamppost")!;
  const move = entity!.handles.find(({ id }) => id === "move")!;
  const target = { x: entity!.frame.origin_m.x + 0.3, y: entity!.frame.origin_m.y, z: entity!.frame.origin_m.z };
  const next = move.drag(target, undefined) as SceneDescription;
  const after = sceneSceneryGraph(next).nodes.find(({ id }) => id === "lamppost")!;
  const s = Math.max(scene.container.width_m, scene.container.height_m, scene.container.depth_m);
  // Authored in scene-scale fractions, so the offset is the world move divided
  // by the environment scale — which is what keeps it attached when the
  // container is resized.
  assert.ok(Math.abs((after.place!.position!.x - before.place!.position!.x) - 0.3 / s) < 1e-9);
  assert.equal(after.place!.position!.y, before.place!.position!.y, "a locked axis holds");
  assert.equal(after.place!.anchor, "floor", "moving an object does not change what it stands on");
  assert.deepEqual(validateScene(next), [], "the edited document stays valid");
  // The move landed the object where the pointer asked.
  assert.ok(Math.abs(centreOf(next, "lamppost").x - target.x) < 1e-9);
});

test("removing scenery drops exactly one node and leaves the rest addressable", () => {
  const scene = gardenScene();
  const entity = findEntity(context(scene), { kind: "scenery", id: scenerySelectionId("lamppost") });
  const next = entity!.remove!();
  assert.equal(findSceneryNode(next, "lamppost"), undefined);
  assert.equal(
    sceneSceneryGraph(next).nodes.length,
    sceneSceneryGraph(scene).nodes.length - 1,
  );
  assert.ok(findSceneryNode(next, "tree-hero"), "its neighbours are untouched");
  assert.deepEqual(validateScene(next), []);
});

test("a placed prop is an ordinary node, in metres because a click chose a world point", () => {
  const scene = cloneScene(defaultScene);
  scene.environment = "default";
  const node = createSceneryNodeAt(scene, "box", { x: .2, y: 0, z: -.1 }, { x: 0, y: 1, z: 0 });
  assert.equal(node.place?.units, "metres");
  assert.equal(node.place?.anchor, undefined, "a world point is not re-resolved against a datum");
  const next = addSceneryNode(scene, node);
  assert.deepEqual(validateScene(next), []);
  const placed = findEntity(context(next), { kind: "scenery", id: scenerySelectionId(node.id) });
  assert.ok(placed, "it is selectable the moment it exists");
  // It sits on the surface rather than half through it.
  assert.ok(centreOf(next, node.id).y > 0);
});

test("the room itself is not an object in it", () => {
  const scene = cloneScene(defaultScene);
  scene.environment = "conservatory";
  const ids = sceneryEntity.instances(context(scene)).map(({ selection }) => selection.id);
  assert.equal(ids.includes(scenerySelectionId("shell")), false,
    "shell faces enclose the scene; a click on the floor must not put a gizmo on the room");
  assert.equal(ids.includes(scenerySelectionId("glazing/pane-left-low")), false,
    "a pane publishes no proxy, so there is nothing under the pointer to grab");
  assert.ok(ids.includes(scenerySelectionId("glazing/frame-0")), "its frame, however, is right there");
});
