import assert from "node:assert/strict";
import test from "node:test";
import { EDITOR_PROBES, targetAtRay } from "../lib/core/editor-probe-catalog";
import { TANK_SELECTION_ID } from "../lib/core/editor-tank";
import { cloneScene, defaultScene, type SceneDescription } from "../lib/core/model";
import { sceneCellSizes_m } from "../lib/core/scene-lattice";
import type { EditorEntityContext, EditorRay } from "../lib/core/editor-entity";

function context(scene: SceneDescription): EditorEntityContext {
  // `pickingAvailable: true` stands in for a fenced presentation, which is what
  // the entity probe gates on. Every other probe reads the document alone.
  return { scene, bodies: [], pickingAvailable: true };
}

/** A ray from a point on a sphere around the scene, aimed at the given target. */
function rayFrom(from: EditorRay["origin"], to: EditorRay["origin"]): EditorRay {
  const d = { x: to.x - from.x, y: to.y - from.y, z: to.z - from.z };
  const length = Math.hypot(d.x, d.y, d.z);
  return { origin: from, direction: { x: d.x / length, y: d.y / length, z: d.z / length } };
}

test("exactly one probe is the fallback, and it is last", () => {
  const fallbacks = EDITOR_PROBES.filter((probe) => probe.fallback);
  assert.equal(fallbacks.length, 1, "the total-ness of targetAtRay rests on there being one");
  assert.equal(EDITOR_PROBES.at(-1), fallbacks[0]);
});

test("probe ids are unique, so targetActionsAt can resolve a target's own probe", () => {
  const ids = EDITOR_PROBES.map((probe) => probe.id);
  assert.equal(new Set(ids).size, ids.length);
});

// The guiding principle of INTERACT, asserted as a property rather than as a
// list of covered cases: sweep a fan of rays over the scene from several camera
// positions — including from inside the tank and from below the floor, aimed at
// open sky — and every single one names something.
test("every ray from every direction lands on a target", () => {
  const scene = cloneScene(defaultScene);
  const c = scene.container;
  const centre = { x: 0, y: 0.5 * c.height_m, z: 0 };
  const radius = 3 * Math.max(c.width_m, c.height_m, c.depth_m);
  const eyes = [
    { x: radius, y: 0.6 * radius, z: radius },
    { x: -radius, y: 0.2 * radius, z: 0 },
    { x: 0, y: -radius, z: 0.4 * radius },
    { x: 0, y: 2 * radius, z: 0 },
    centre,
  ];
  let sampled = 0;
  for (const eye of eyes) {
    for (let u = -1; u <= 1; u += 0.25) {
      for (let v = -1; v <= 1; v += 0.25) {
        const aim = { x: centre.x + u * radius, y: centre.y + v * radius, z: centre.z + u * v * radius };
        if (aim.x === eye.x && aim.y === eye.y && aim.z === eye.z) continue;
        const target = targetAtRay(context(scene), rayFrom(eye, aim));
        assert.ok(target, `no target for ray from ${JSON.stringify(eye)}`);
        assert.ok(target.label.length > 0);
        assert.ok(Number.isFinite(target.point_m.x + target.point_m.y + target.point_m.z),
          "a target's point must be a real place, since verbs are aimed at it");
        sampled += 1;
      }
    }
  }
  assert.ok(sampled > 300, `expected a broad sweep, sampled ${sampled}`);
});

test("a ray down the middle of an empty tank reaches a wall, not the room", () => {
  const scene = cloneScene(defaultScene);
  scene.solidVoxels = [];
  const c = scene.container;
  const target = targetAtRay(context(scene), {
    origin: { x: 0, y: 0.5 * c.height_m, z: -2 * c.depth_m },
    direction: { x: 0, y: 0, z: 1 },
  });
  assert.equal(target.kind, "tank-wall");
  assert.equal(target.id, "+z");
  // A wall belongs to the vessel, so clicking one selects the tank; the finer
  // address stays on the target for the ring and the drag.
  assert.deepEqual(target.selection, { kind: "tank", id: TANK_SELECTION_ID });
  assert.equal(target.highlight.kind, "quad");
});

test("an occupied voxel outranks the wall behind it", () => {
  const scene = cloneScene(defaultScene);
  const [hx, hy, hz] = sceneCellSizes_m(scene);
  scene.solidVoxels = [{
    operation: "fill", minimum: [3, 4, 2], maximumExclusive: [4, 5, 3], materialId: 2,
  }];
  const originX = -0.5 * scene.container.width_m;
  const originZ = -0.5 * scene.container.depth_m;
  const target = targetAtRay(context(scene), {
    origin: { x: originX + 2 * hx, y: 4.5 * hy, z: originZ + 2.5 * hz },
    direction: { x: 1, y: 0, z: 0 },
  });
  assert.equal(target.kind, "solid-voxel");
  assert.equal(target.id, "3,4,2");
  assert.equal(target.highlight.kind, "box");
  // The face is what a drag from here locks to; without it a sweep would burrow
  // into the solid instead of running across the surface being looked at.
  assert.equal(target.detail?.faceAxis, 0);
  assert.equal(target.detail?.faceSign, -1);
});

// From *inside* the tank every ray meets a wall, so the room fallback is only
// reachable from outside it, aimed away. That is the case that used to render as
// a dead pixel with no chip and no highlight.
test("a ray aimed at open sky still has a target, and it is the room", () => {
  const scene = cloneScene(defaultScene);
  const c = scene.container;
  const target = targetAtRay(context(scene), {
    origin: { x: 0, y: 4 * c.height_m, z: 4 * c.depth_m },
    direction: { x: 0, y: 0.7071, z: 0.7071 },
  });
  assert.equal(target.kind, "room");
  // Nothing in the document is the room, so a click on it selects nothing —
  // which is exactly how clicking empty space deselects.
  assert.equal(target.selection, undefined);
  assert.ok(Number.isFinite(target.point_m.y));
});

test("the fallback never outranks a real hit even though it is infinitely far", () => {
  const scene = cloneScene(defaultScene);
  scene.solidVoxels = [];
  const c = scene.container;
  const target = targetAtRay(context(scene), {
    origin: { x: 0, y: 0.5 * c.height_m, z: 0 },
    direction: { x: 0, y: -1, z: 0 },
  });
  assert.notEqual(target.kind, "room");
  assert.ok(Number.isFinite(target.distance_m));
});

// The bug this exists to hold shut: a container's shell is authored into
// `scene.solidVoxels` like any other patch, so the solid world cannot tell it
// from a dam. Stopping on the first solid cell answered "a voxel of the front
// pane" for every pixel of the tank, and since a bare voxel named nothing to
// select, the tank and the water were both unclickable from any camera.
test("the vessel is glass: a ray from outside reaches what is behind the near wall", () => {
  const scene = cloneScene(defaultScene);
  const c = scene.container;
  const from = { x: 0.9 * c.width_m, y: 0.75 * c.height_m, z: 1.6 * c.depth_m };

  const water = targetAtRay(context(scene), rayFrom(from, { x: 0, y: 0.25 * c.height_m, z: 0 }));
  assert.equal(water.kind, "entity");
  assert.deepEqual(water.selection, { kind: "fluid-body", id: "fluid-body" });

  // …and with the water transparent, the tank behind it is reachable too, which
  // is the second half of "select the water tank and the fluid".
  const tank = targetAtRay(context(scene),
    rayFrom(from, { x: 0, y: 0.25 * c.height_m, z: 0 }), water.selection);
  assert.deepEqual(tank.selection, { kind: "tank", id: TANK_SELECTION_ID });
});

// Glass, not a hole: the wall is still solid, and a ray that has nothing behind
// the shell must not fall through the vessel to the room.
test("seeing through the shell does not delete it", () => {
  const scene = cloneScene(defaultScene);
  const c = scene.container;
  const target = targetAtRay(context(scene), rayFrom(
    { x: 0, y: 0.95 * c.height_m, z: 1.6 * c.depth_m },
    { x: 0, y: 0.95 * c.height_m, z: 0 }));
  assert.notEqual(target.kind, "room");
  assert.deepEqual(target.selection, { kind: "tank", id: TANK_SELECTION_ID });
});

// An authored solid is not the vessel and must still be pickable — the skip is
// exactly the shell, never "solids near the wall".
test("a solid the reader authored is still the nearest thing", () => {
  const scene = cloneScene(defaultScene);
  // No authored water, so the fluid body cannot stand in front of the block.
  scene.fluid = { ...scene.fluid, initialCondition: "tank-fill" };
  scene.container = { ...scene.container, fillFraction: 0 };
  const [, hy, hz] = sceneCellSizes_m(scene);
  scene.solidVoxels = [...(scene.solidVoxels ?? []),
    { operation: "fill", minimum: [4, 4, 4], maximumExclusive: [8, 8, 8], materialId: 2 }];
  const c = scene.container;
  const target = targetAtRay(context(scene), {
    origin: { x: -2 * c.width_m, y: 6 * hy, z: -0.5 * c.depth_m + 6 * hz },
    direction: { x: 1, y: 0, z: 0 },
  });
  assert.equal(target.kind, "solid-voxel");
  assert.deepEqual(target.selection, { kind: "tank", id: TANK_SELECTION_ID },
    "a solid names the vessel whose solid world it is a cell of");
});
