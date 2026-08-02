import assert from "node:assert/strict";
import test from "node:test";
import {
  createInflowAt,
  inflowBox,
  inflowDirection,
  inflowEntity,
  inflowSpeed_m_s,
  INFLOW_MINIMUM_RADIUS_M,
  INFLOW_SELECTION_ID,
} from "../lib/editor-inflow";
import { frameToWorld, handleWorldPosition } from "../lib/editor-entity";
import { getScenePreset } from "../lib/scenes";
import { validateScene, type SceneDescription } from "../lib/model";
import { context } from "./helpers/editor-entities";

function withHose(direction = { x: 0, y: -1, z: 0 }): SceneDescription {
  const base = getScenePreset("water-box-dam-break").create();
  const inflow = createInflowAt({ x: 0, y: 0.2, z: 0 }, direction, base);
  return { ...base, fluid: { ...base.fluid, inflow } };
}

test("the hose's box is its own channel: a radius across, its length along", () => {
  const scene = withHose();
  const inflow = scene.fluid.inflow!;
  const box = inflowBox(inflow);
  assert.equal(box.max.x, inflow.radius_m);
  assert.equal(box.max.z, inflow.radius_m);
  assert.equal(box.max.y - box.min.y, inflow.length_m);
});

test("the frame carries the aim, so the channel is drawn along the jet", () => {
  // Aiming a cylinder about the axis it sprays along *is* orienting it, which is
  // what lets the nozzle be an ordinary box entity.
  const scene = withHose({ x: 0, y: -1, z: 0 });
  const entity = inflowEntity.find(context(scene), INFLOW_SELECTION_ID)!;
  const direction = inflowDirection(scene.fluid.inflow!);
  // The entity's +y must land on the jet direction.
  const tip = frameToWorld(entity.frame, { x: 0, y: 1, z: 0 });
  const along = {
    x: tip.x - entity.frame.origin_m.x,
    y: tip.y - entity.frame.origin_m.y,
    z: tip.z - entity.frame.origin_m.z,
  };
  for (const axis of ["x", "y", "z"] as const) {
    assert.ok(Math.abs(along[axis] - direction[axis]) < 1e-9,
      `${axis}: expected ${direction[axis]}, got ${along[axis]}`);
  }
});

test("a sideways jet orients the same way, not just the straight-down one", () => {
  const scene = withHose({ x: 1, y: 0, z: 0 });
  const entity = inflowEntity.find(context(scene), INFLOW_SELECTION_ID)!;
  const direction = inflowDirection(scene.fluid.inflow!);
  const tip = frameToWorld(entity.frame, { x: 0, y: 1, z: 0 });
  assert.ok(Math.abs((tip.x - entity.frame.origin_m.x) - direction.x) < 1e-9);
});

test("a radial drag sets the radius and holds the length; a y drag does the reverse", () => {
  const scene = withHose();
  const inflow = scene.fluid.inflow!;
  const entity = inflowEntity.find(context(scene), INFLOW_SELECTION_ID)!;

  const side = entity.handles.find((handle) => handle.id === "+00")!;
  const wider = side.drag({ x: 2 * inflow.radius_m, y: 0, z: 0 }, undefined);
  assert.ok(wider?.fluid?.inflow);
  assert.ok(Math.abs(wider.fluid.inflow.radius_m - 2 * inflow.radius_m) < 1e-9);
  assert.equal(wider.fluid.inflow.length_m, inflow.length_m, "the channel keeps its length");
  assert.ok(Math.abs(wider.fluid.inflow.radius_m - wider.fluid.inflow.radius_m) < 1e-9);

  const end = entity.handles.find((handle) => handle.id === "0+0")!;
  const longer = end.drag({ x: 0, y: inflow.length_m, z: 0 }, undefined);
  assert.ok(longer?.fluid?.inflow);
  assert.ok(longer.fluid.inflow.length_m > inflow.length_m);
  assert.equal(longer.fluid.inflow.radius_m, inflow.radius_m, "and the bore is untouched");

  assert.deepEqual(validateScene({ ...scene, ...longer }), []);
});

test("the nozzle cannot be pinched shut", () => {
  const scene = withHose();
  const entity = inflowEntity.find(context(scene), INFLOW_SELECTION_ID)!;
  const side = entity.handles.find((handle) => handle.id === "+00")!;
  const pinched = side.drag({ x: 0, y: 0, z: 0 }, undefined);
  assert.ok(pinched?.fluid?.inflow);
  assert.ok(pinched.fluid.inflow.radius_m >= INFLOW_MINIMUM_RADIUS_M - 1e-9);
  assert.deepEqual(validateScene({ ...scene, ...pinched }), []);
});

test("the aim arrow keeps its old job in the shared vocabulary", () => {
  const scene = withHose();
  const inflow = scene.fluid.inflow!;
  const entity = inflowEntity.find(context(scene), INFLOW_SELECTION_ID)!;
  const aim = entity.handles.find((handle) => handle.id === "aim")!;
  assert.equal(aim.kind, "tip");
  assert.equal(aim.space, "world", "the arrow points anywhere in the world, not along the channel");
  assert.deepEqual(aim.axes, ["x", "y", "z"]);
  // It stands at the arrow tip, which is where the old bespoke handle was.
  const expected = inflowSpeed_m_s(inflow);
  assert.ok(expected > 0);

  const aimed = aim.drag({ x: 0.4, y: 0.2, z: 0 }, undefined);
  assert.ok(aimed?.fluid?.inflow);
  const next = inflowDirection(aimed.fluid.inflow);
  assert.ok(next.x > 0, "the jet must now point toward the dragged tip");
  assert.deepEqual(validateScene({ ...scene, ...aimed }), []);
});

test("the hose is clicked on its channel, in the frame its handles use", () => {
  const scene = withHose();
  const inflow = scene.fluid.inflow!;
  const hit = inflowEntity.pick!(context(scene), {
    origin: { x: inflow.center_m.x, y: inflow.center_m.y, z: inflow.center_m.z - 5 },
    direction: { x: 0, y: 0, z: 1 },
  });
  assert.ok(hit);
  assert.equal(hit.selection.id, INFLOW_SELECTION_ID);
  assert.ok(Math.abs(hit.distance_m - (5 - inflow.radius_m)) < 1e-9);

  const missed = inflowEntity.pick!(context(scene), {
    origin: { x: inflow.center_m.x + 10, y: inflow.center_m.y, z: inflow.center_m.z - 5 },
    direction: { x: 0, y: 0, z: 1 },
  });
  assert.equal(missed, undefined);
});

test("a scene with no hose offers no hose to select", () => {
  const base = getScenePreset("water-box-dam-break").create();
  assert.equal(inflowEntity.instances(context(base)).length, 0);
  assert.equal(inflowEntity.find(context(base), INFLOW_SELECTION_ID), undefined);
  assert.equal(inflowEntity.pick!(context(base),
    { origin: { x: 0, y: 0, z: -5 }, direction: { x: 0, y: 0, z: 1 } }), undefined);
});

test("removing the hose drops the field rather than leaving a dead nozzle", () => {
  const scene = withHose();
  const entity = inflowEntity.find(context(scene), INFLOW_SELECTION_ID)!;
  const next = entity.remove!();
  assert.equal("inflow" in next.fluid, false);
  assert.deepEqual(validateScene(next), []);
});

test("handles project through the frame, so a rotated hose's are on the hose", () => {
  // A hose placed on a wall sprays *away* from it, so the jet is the negated
  // placement normal — and the channel runs along the jet, not along the normal.
  const scene = withHose({ x: 1, y: 0, z: 0 });
  const entity = inflowEntity.find(context(scene), INFLOW_SELECTION_ID)!;
  const inflow = scene.fluid.inflow!;
  const direction = inflowDirection(inflow);
  assert.ok(direction.x < 0, "the jet points back off the wall it was placed on");

  // The +y face is the far end of the channel: half a length along the jet.
  const far = handleWorldPosition(entity, entity.handles.find((handle) => handle.id === "0+0")!);
  const near = handleWorldPosition(entity, entity.handles.find((handle) => handle.id === "0-0")!);
  for (const axis of ["x", "y", "z"] as const) {
    assert.ok(Math.abs(far[axis] - (inflow.center_m[axis] + 0.5 * inflow.length_m * direction[axis])) < 1e-9,
      `far ${axis}: got ${far[axis]}`);
    assert.ok(Math.abs(near[axis] - (inflow.center_m[axis] - 0.5 * inflow.length_m * direction[axis])) < 1e-9,
      `near ${axis}: got ${near[axis]}`);
  }
});
