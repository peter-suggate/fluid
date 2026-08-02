import assert from "node:assert/strict";
import test from "node:test";
import {
  boxCorners,
  boxEdgeSegment,
  boxHandles,
  boxResizeDrag,
  entityHandleAtPointer,
  frameDirectionToLocal,
  frameRayToLocal,
  frameToLocal,
  frameToWorld,
  handleIsInert,
  handleWorldEnds,
  handleWorldPosition,
  intersectBox,
  moveHandles,
  pickRoomInterior,
  pickSolidBox,
  resizeBox,
  BOX_EDGES,
  WORLD_FRAME,
  type BoxExtent,
  type BoxResizePolicy,
  type EditorEntity,
  type EditorFrame,
} from "../lib/editor-entity";
import { projectToViewport } from "../lib/webgpu-camera";
import { sides } from "./helpers/editor-entities";
import type { CameraState, Quaternion, Vec3 } from "../lib/model";

const UNIT: BoxExtent = { min: { x: -1, y: 0, z: -1 }, max: { x: 1, y: 2, z: 1 } };
const FREE: BoxResizePolicy = { minimum_m: [0, 0, 0] };
const CAMERA: CameraState = {
  azimuth_rad: 0.72, elevation_rad: 0.42, distance_m: 6, target_m: { x: 0, y: 1, z: 0 },
};

/** A quaternion for a rotation about y, so a frame can be given a real pose. */
function yaw(radians: number): Quaternion {
  return { w: Math.cos(radians / 2), x: 0, y: Math.sin(radians / 2), z: 0 };
}

function entity(overrides: Partial<EditorEntity> = {}): EditorEntity {
  const box = overrides.box ?? UNIT;
  return {
    selection: { kind: "scenery", id: "test" },
    label: "TEST",
    tone: "prop",
    frame: WORLD_FRAME,
    draftSubject: "scenery",
    editLabel: () => "Test",
    ...overrides,
    box,
    handles: overrides.handles ?? boxHandles(box, { drag: () => ({}) }),
  };
}

test("a box carries a handle for every face, edge, and corner", () => {
  const handles = boxHandles(UNIT, { drag: () => ({}) });
  assert.equal(handles.length, 26);
  assert.equal(handles.filter((handle) => handle.kind === "face").length, 6);
  assert.equal(handles.filter((handle) => handle.kind === "edge").length, 12);
  assert.equal(handles.filter((handle) => handle.kind === "corner").length, 8);
  assert.equal(new Set(handles.map((handle) => handle.id)).size, 26, "handle ids must be unique");

  const maxX = handles.find((handle) => handle.id === "+00")!;
  assert.deepEqual(maxX.position_m, { x: 1, y: 1, z: 0 });
  assert.deepEqual(maxX.axes, ["x"]);
  assert.equal(maxX.label, "face · X");

  const corner = handles.find((handle) => handle.id === "-+-")!;
  assert.deepEqual(corner.position_m, { x: -1, y: 2, z: -1 });
  assert.deepEqual(corner.axes, ["x", "y", "z"]);
  assert.equal(corner.label, "corner · X Y Z");
});

test("handles the entity does not offer are never built", () => {
  const handles = boxHandles(UNIT, { grabbable: (side) => side.y !== "min", drag: () => ({}) });
  assert.equal(handles.length, 26 - 9, "a whole face of the box goes: 1 face, 4 edges, 4 corners");
  assert.ok(!handles.some((handle) => handle.id.includes("-") && handle.id[1] === "-"));
});

test("an edge handle spans the whole edge it grabs", () => {
  const box = { min: { x: -1, y: 0, z: -2 }, max: { x: 1, y: 3, z: 2 } };
  // "+0-" fixes max x and min z, leaving y free: the edge runs the box's height.
  const segment = boxEdgeSegment(box, sides("+0-"));
  assert.ok(segment);
  assert.deepEqual(segment.from, { x: 1, y: 0, z: -2 });
  assert.deepEqual(segment.to, { x: 1, y: 3, z: -2 });

  // Only edges have a segment; faces and corners are points.
  assert.equal(boxEdgeSegment(box, sides("+00")), undefined);
  assert.equal(boxEdgeSegment(box, sides("+++")), undefined);

  for (const handle of boxHandles(box, { drag: () => ({}) }).filter((entry) => entry.kind === "edge")) {
    assert.ok(handle.segment, `${handle.id} must span an edge`);
    // Exactly one axis varies along an edge — the one the handle leaves free.
    const varying = (["x", "y", "z"] as const)
      .filter((axis) => handle.segment!.from[axis] !== handle.segment!.to[axis]);
    assert.equal(varying.length, 1);
    assert.ok(!handle.axes.includes(varying[0]!));
    // And both ends lie on the box it came from.
    for (const end of [handle.segment.from, handle.segment.to]) {
      for (const axis of ["x", "y", "z"] as const) {
        assert.ok(end[axis] === box.min[axis] || end[axis] === box.max[axis],
          `${handle.id} endpoint is off the box on ${axis}`);
      }
    }
  }
});

test("a free axis holds its opposite side, so a face drag reads as moving that face", () => {
  const dragged = resizeBox(UNIT, sides("+00"), { x: 3, y: 0, z: 0 }, FREE);
  assert.equal(dragged.max.x, 3);
  assert.equal(dragged.min.x, UNIT.min.x, "the opposite side is the anchor");
  assert.deepEqual([dragged.min.y, dragged.max.y], [UNIT.min.y, UNIT.max.y]);
});

test("a side pushed through the box stops at the minimum rather than inverting", () => {
  const policy: BoxResizePolicy = { minimum_m: [0.5, 0.5, 0.5] };
  const squashed = resizeBox(UNIT, sides("+00"), { x: -50, y: 0, z: 0 }, policy);
  assert.equal(squashed.max.x - squashed.min.x, 0.5);
  const pulled = resizeBox(UNIT, sides("-00"), { x: 50, y: 0, z: 0 }, policy);
  assert.equal(pulled.max.x - pulled.min.x, 0.5);
});

test("limits clamp the box, and snapping is measured from their minimum corner", () => {
  const policy: BoxResizePolicy = {
    minimum_m: [0, 0, 0],
    snap_m: [0.25, 0.25, 0.25],
    limits: { min: { x: -2, y: 0, z: -2 }, max: { x: 2, y: 4, z: 2 } },
  };
  assert.equal(resizeBox(UNIT, sides("+00"), { x: 99, y: 0, z: 0 }, policy).max.x, 2);
  const snapped = resizeBox(UNIT, sides("+00"), { x: 1.31, y: 0, z: 0 }, policy);
  assert.equal(snapped.max.x, 1.25, "1.31 lands on the quarter-metre grid measured from -2");
});

test("linked axes share one extent and resize about the centre", () => {
  // The sphere case: the schema stores one radius, so there is no side to hold
  // still and every linked axis has to follow the one that was dragged.
  const box: BoxExtent = { min: { x: -1, y: -1, z: -1 }, max: { x: 1, y: 1, z: 1 } };
  const policy: BoxResizePolicy = { minimum_m: [0.1, 0.1, 0.1], links: [["x", "y", "z"]] };
  const grown = resizeBox(box, sides("+00"), { x: 3, y: 0, z: 0 }, policy);
  for (const axis of ["x", "y", "z"] as const) {
    assert.equal(grown.min[axis], -3, `${axis} min`);
    assert.equal(grown.max[axis], 3, `${axis} max`);
  }

  // The cylinder case: x and z are one radius, y is free.
  const cylinder: BoxResizePolicy = { minimum_m: [0.1, 0.1, 0.1], links: [["x", "z"]] };
  const widened = resizeBox(box, sides("+00"), { x: 2, y: 0, z: 0 }, cylinder);
  assert.deepEqual([widened.min.x, widened.max.x], [-2, 2]);
  assert.deepEqual([widened.min.z, widened.max.z], [-2, 2], "the cross section stays circular");
  assert.deepEqual([widened.min.y, widened.max.y], [-1, 1], "the free axis is untouched");
});

test("a corner on linked axes does not jump the moment it is touched", () => {
  // The corner of a square cross section stands at r·√2 from the axis, so
  // reading the radius radially would inflate it by 41% before the pointer had
  // moved at all. The largest requested half-extent is stable at rest.
  const box: BoxExtent = { min: { x: -1, y: -1, z: -1 }, max: { x: 1, y: 1, z: 1 } };
  const policy: BoxResizePolicy = { minimum_m: [0.1, 0.1, 0.1], links: [["x", "z"]] };
  const held = resizeBox(box, sides("+0+"), { x: 1, y: 0, z: 1 }, policy);
  assert.deepEqual([held.min.x, held.max.x], [-1, 1], "an untouched corner must not resize");
  const grown = resizeBox(box, sides("+0+"), { x: 2, y: 0, z: 1 }, policy);
  assert.deepEqual([grown.min.x, grown.max.x], [-2, 2], "and either axis may lead");
});

test("a linked group never drops below its minimum", () => {
  const box: BoxExtent = { min: { x: -1, y: -1, z: -1 }, max: { x: 1, y: 1, z: 1 } };
  const policy: BoxResizePolicy = { minimum_m: [0.4, 0.4, 0.4], links: [["x", "y", "z"]] };
  const collapsed = resizeBox(box, sides("+00"), { x: 0, y: 0, z: 0 }, policy);
  for (const axis of ["x", "y", "z"] as const) {
    assert.equal(collapsed.max[axis] - collapsed.min[axis], 0.4, `${axis} floor`);
  }
});

test("a handle applies the lock itself, so the drag maths never sees one", () => {
  let received: BoxExtent | undefined;
  const handles = boxHandles(UNIT, {
    drag: boxResizeDrag(UNIT, FREE, (box) => { received = box; return {}; }),
  });
  const corner = handles.find((handle) => handle.id === "+++")!;

  corner.drag({ x: 5, y: 5, z: 5 }, ["y"]);
  assert.deepEqual(received, { min: UNIT.min, max: { x: 1, y: 5, z: 1 } },
    "only the locked axis may move");

  received = undefined;
  const face = handles.find((handle) => handle.id === "+00")!;
  assert.equal(face.drag({ x: 5, y: 5, z: 5 }, ["y"]), undefined,
    "a lock owning none of the handle's axes proposes nothing at all");
  assert.equal(received, undefined);
  assert.equal(handleIsInert(face, ["y"]), true);
  assert.equal(handleIsInert(corner, ["y"]), false);
});

test("a move handle keeps the axes the lock leaves and holds the rest", () => {
  let moved: Vec3 | undefined;
  const handles = moveHandles({ x: 1, y: 2, z: 3 }, (position) => { moved = position; return {}; });
  const centre = handles.find((handle) => handle.id === "move")!;
  assert.equal(centre.kind, "center");
  assert.equal(centre.space, "world", "a move acts on the frame, so it speaks world coordinates");

  centre.drag({ x: 9, y: 9, z: 9 }, undefined);
  assert.deepEqual(moved, { x: 9, y: 9, z: 9 });

  centre.drag({ x: 9, y: 9, z: 9 }, ["y"]);
  assert.deepEqual(moved, { x: 1, y: 9, z: 3 }, "a locked move slides rather than jumping");

  const arrow = handles.find((handle) => handle.id === "move-x")!;
  assert.deepEqual(arrow.axes, ["x"]);
  arrow.drag({ x: 9, y: 9, z: 9 }, undefined);
  assert.deepEqual(moved, { x: 9, y: 2, z: 3 });
  assert.equal(arrow.drag({ x: 9, y: 9, z: 9 }, ["y"]), undefined);
});

test("the world frame is the identity, and a posed frame round-trips", () => {
  const point = { x: 0.3, y: -1.2, z: 4 };
  assert.deepEqual(frameToWorld(WORLD_FRAME, point), point);
  assert.deepEqual(frameToLocal(WORLD_FRAME, point), point);

  const frame: EditorFrame = { origin_m: { x: 2, y: 1, z: -3 }, orientation: yaw(Math.PI / 3) };
  const world = frameToWorld(frame, point);
  const back = frameToLocal(frame, world);
  for (const axis of ["x", "y", "z"] as const) {
    assert.ok(Math.abs(back[axis] - point[axis]) < 1e-9, `${axis} round trip`);
  }
});

test("a quarter turn about y maps the entity's x onto the world's -z", () => {
  const frame: EditorFrame = { origin_m: { x: 0, y: 0, z: 0 }, orientation: yaw(Math.PI / 2) };
  const world = frameToWorld(frame, { x: 1, y: 0, z: 0 });
  assert.ok(Math.abs(world.x) < 1e-9 && Math.abs(world.z + 1) < 1e-9,
    `expected roughly (0,0,-1), got (${world.x}, ${world.y}, ${world.z})`);
  // A direction is rotated and never translated, which is what lets the camera
  // plane be expressed in entity space during a drag.
  const local = frameDirectionToLocal(
    { origin_m: { x: 50, y: 50, z: 50 }, orientation: yaw(Math.PI / 2) }, { x: 0, y: 1, z: 0 });
  assert.ok(Math.abs(local.y - 1) < 1e-9, "y is the rotation axis, so it survives untouched");
});

test("a ray resolves into entity space, translation and all", () => {
  const frame: EditorFrame = { origin_m: { x: 5, y: 0, z: 0 }, orientation: yaw(Math.PI / 2) };
  const local = frameRayToLocal(frame, { origin: { x: 5, y: 1, z: 0 }, direction: { x: 0, y: 0, z: 1 } });
  assert.ok(Math.abs(local.origin.x) < 1e-9 && Math.abs(local.origin.y - 1) < 1e-9,
    "a ray starting at the entity's own origin starts at the local origin");
  // The frame turns the entity's +x onto the world's -z, so the inverse brings
  // the world's +z back onto the entity's -x.
  assert.ok(Math.abs(local.direction.x + 1) < 1e-9,
    `expected the local direction to be -x, got x=${local.direction.x}`);
});

test("a rotated entity's handles are drawn where its box actually is", () => {
  const frame: EditorFrame = { origin_m: { x: 0, y: 1, z: 0 }, orientation: yaw(Math.PI / 2) };
  const rotated = entity({ frame, box: { min: { x: -2, y: -0.5, z: -0.5 }, max: { x: 2, y: 0.5, z: 0.5 } } });
  const maxX = rotated.handles.find((handle) => handle.id === "+00")!;
  const world = handleWorldPosition(rotated, maxX);
  assert.ok(Math.abs(world.z + 2) < 1e-9,
    `the long axis should point along world -z, got z=${world.z}`);
  assert.ok(Math.abs(world.y - 1) < 1e-9, "and the frame origin carries it");
});

test("an arm resolves to a constant on-screen length, so it grows with distance", () => {
  const moving = entity({ handles: moveHandles({ x: 0, y: 0, z: 0 }, () => ({})) });
  const arrow = moving.handles.find((handle) => handle.id === "move-x")!;
  const near = handleWorldEnds(moving, arrow, 2);
  const far = handleWorldEnds(moving, arrow, 20);
  assert.ok(near && far);
  assert.ok(far.to.x > near.to.x * 5, "ten times the depth is about ten times the arm");
  assert.deepEqual(near.from, { x: 0, y: 0, z: 0 });
});

test("corners beat edges beat faces, because the corner could not be reached otherwise", () => {
  const subject = entity();
  const corner = subject.handles.find((handle) => handle.id === "+++")!;
  const at = projectToViewport(handleWorldPosition(subject, corner), CAMERA, 1200, 800);
  const pointer = { x: at.leftFraction * 1200, y: at.topFraction * 800 };
  const pick = entityHandleAtPointer([subject], CAMERA, 1200, 800, pointer);
  assert.ok(pick);
  assert.equal(pick.handle.id, "+++", "the three faces meeting here must not win");
});

test("an earlier entity beats a later one outright, whatever the later one offers", () => {
  const front = entity({ selection: { kind: "fluid-body", id: "fluid-body" }, tone: "fluid" });
  const behind = entity({ selection: { kind: "tank", id: "tank" }, tone: "tank" });
  const corner = front.handles.find((handle) => handle.id === "+++")!;
  const at = projectToViewport(handleWorldPosition(front, corner), CAMERA, 1200, 800);
  const pick = entityHandleAtPointer([front, behind], CAMERA, 1200, 800,
    { x: at.leftFraction * 1200, y: at.topFraction * 800 });
  assert.equal(pick?.entity.selection.kind, "fluid-body");
});

test("an edge is grabbable along its length, not just at its midpoint", () => {
  const subject = entity();
  // Find an edge whose whole span projects in front of the camera, then aim a
  // quarter of the way along it — nowhere near the midpoint a square would use.
  const edge = subject.handles.filter((handle) => handle.kind === "edge")
    .map((handle) => ({
      handle,
      ends: [handle.segment!.from, handle.segment!.to]
        .map((point) => projectToViewport(point, CAMERA, 1200, 800)),
    }))
    .find(({ ends }) => ends.every((end) => end.visible && end.depth_m > 1e-6));
  assert.ok(edge, "fixture must present at least one fully visible edge");

  const [from, to] = edge.ends.map((end) => ({ x: end.leftFraction * 1200, y: end.topFraction * 800 }));
  const quarter = { x: from!.x + 0.25 * (to!.x - from!.x), y: from!.y + 0.25 * (to!.y - from!.y) };
  const pick = entityHandleAtPointer([subject], CAMERA, 1200, 800, quarter);
  assert.ok(pick, "a point on the drawn edge must grab something");
  assert.equal(pick.handle.id, edge.handle.id, "and it must be the edge under the pointer");
});

test("a solid is picked where the ray enters it; a room where it leaves", () => {
  const ray = { origin: { x: 0, y: 1, z: -5 }, direction: { x: 0, y: 0, z: 1 } };
  const span = intersectBox(ray, UNIT);
  assert.ok(span);
  assert.ok(Math.abs(span.near_m - 4) < 1e-9 && Math.abs(span.far_m - 6) < 1e-9);
  assert.ok(Math.abs(pickSolidBox(ray, UNIT)! - 4) < 1e-9);
  assert.ok(Math.abs(pickRoomInterior(ray, UNIT)! - 6) < 1e-9,
    "the wall you can see from inside a tank is the far one");

  // From inside the box, a solid has no near crossing left to use.
  const inside = { origin: { x: 0, y: 1, z: 0 }, direction: { x: 0, y: 0, z: 1 } };
  assert.ok(Math.abs(pickSolidBox(inside, UNIT)! - 1) < 1e-9);

  const missed = { origin: { x: 0, y: 9, z: -5 }, direction: { x: 0, y: 0, z: 1 } };
  assert.equal(intersectBox(missed, UNIT), undefined);
  // A box entirely behind the ray is not a hit either.
  const behind = { origin: { x: 0, y: 1, z: 9 }, direction: { x: 0, y: 0, z: 1 } };
  assert.equal(pickSolidBox(behind, UNIT), undefined);
});

test("the box outline is the twelve edges of the eight corners", () => {
  assert.equal(boxCorners(UNIT).length, 8);
  assert.equal(BOX_EDGES.length, 12);
  const corners = boxCorners(UNIT);
  for (const [from, to] of BOX_EDGES) {
    const differing = (["x", "y", "z"] as const)
      .filter((axis) => corners[from]![axis] !== corners[to]![axis]);
    assert.equal(differing.length, 1, "an edge of a box varies on exactly one axis");
  }
});
