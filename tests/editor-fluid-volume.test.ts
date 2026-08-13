import assert from "node:assert/strict";
import test from "node:test";
import { validateScene, type InitialLiquidVolume, type SceneDescription } from "../lib/model";
import { getScenePreset } from "../lib/scenes";
import { EDITOR_ENTITIES } from "../lib/editor-entity-catalog";
import { fluidBodyEntity, fluidWaterVolume_m3 } from "../lib/editor-fluid-body";
import {
  addFluidBall,
  defaultFluidBallRadius_m,
  fluidDropVolume,
  fluidVolumeBodies,
  fluidVolumeBox,
  fluidVolumeVolume_m3,
  pickFluidVolume,
  FLUID_VOLUME_PREFIX,
} from "../lib/editor-fluid-volume";
import { context, entityFor } from "./helpers/editor-entities";

/**
 * The balls of liquid, as things the editor can reach.
 *
 * The point of every case here is that a ball is not a new kind of selectable
 * object: it answers under the `fluid-body` kind, it is numbered in with the
 * reservoir and the painted blobs, and the edits it accepts write back a
 * document `validateScene` still accepts.
 */

function preset(id: string): SceneDescription {
  return getScenePreset(id).create();
}

/** Figure 7 is the lone falling ball in a dry box — one volume, no reservoir. */
const FIGURE_7 = "cm12-figure-7";
/** Figure 9 carries a dam *and* a ball, so its bodies have to be numbered. */
const FIGURE_9 = "cm12-figure-9";
/** Figure 8's liquid is a hemisphere, whose bounds are not its ball's. */
const FIGURE_8 = "cm12-figure-8";

function volumes(scene: SceneDescription): readonly InitialLiquidVolume[] {
  return scene.fluid.initialLiquidVolumes ?? [];
}

function apply(scene: SceneDescription, patch: Partial<SceneDescription> | undefined): SceneDescription {
  assert.ok(patch, "the handle should have proposed a scene");
  return { ...scene, ...patch };
}

test("a paper figure's ball is one of the scene's bodies of water", () => {
  const scene = preset(FIGURE_7);
  const bodies = fluidVolumeBodies(scene);
  assert.equal(bodies.length, 1);
  assert.equal(bodies[0]!.id, `${FLUID_VOLUME_PREFIX}0`);
  const entity = entityFor(scene, { kind: "fluid-body", id: bodies[0]!.id });
  assert.equal(entity.tone, "fluid");
  // The dry box has no reservoir, so this ball is the only water and takes the
  // unnumbered label.
  assert.equal(entity.label, "WATER");
});

test("a ball is numbered in with the reservoir it shares a scene with", () => {
  const scene = preset(FIGURE_9);
  const instances = fluidBodyEntity.instances(context(scene));
  assert.equal(instances.length, 2, "the dam and the ball");
  assert.deepEqual(instances.map((entity) => entity.label), ["WATER 1", "WATER 2"]);
  assert.equal(instances[1]!.selection.id, `${FLUID_VOLUME_PREFIX}0`);
});

test("a click reaches the ball, and passes through the corner of its box", () => {
  const scene = preset(FIGURE_7);
  const ball = volumes(scene)[0]!;
  assert.equal(ball.shape, "sphere");
  const box = fluidVolumeBox(ball);
  const centreHit = pickFluidVolume(
    { origin: { x: ball.center_m.x, y: ball.center_m.y, z: ball.center_m.z + 40 }, direction: { x: 0, y: 0, z: -1 } },
    ball);
  assert.ok(centreHit !== undefined, "a ray at the centre meets the ball");
  // Straight down the box's corner column: inside the gizmo, outside the water.
  const cornerMiss = pickFluidVolume(
    { origin: { x: box.max.x, y: box.max.y, z: box.max.z + 40 }, direction: { x: 0, y: 0, z: -1 } },
    ball);
  assert.equal(cornerMiss, undefined);
});

test("the fluid-body pick answers for a ball, so a click selects it", () => {
  const scene = preset(FIGURE_7);
  const ball = volumes(scene)[0]!;
  assert.equal(ball.shape, "sphere");
  const hit = fluidBodyEntity.pick?.(context(scene), {
    origin: { x: ball.center_m.x, y: ball.center_m.y, z: ball.center_m.z + 40 },
    direction: { x: 0, y: 0, z: -1 },
  });
  assert.deepEqual(hit?.selection, { kind: "fluid-body", id: `${FLUID_VOLUME_PREFIX}0` });
});

test("a corner drag resizes a ball and leaves it a ball", () => {
  const scene = preset(FIGURE_7);
  const before = volumes(scene)[0]!;
  assert.equal(before.shape, "sphere");
  const entity = entityFor(scene, { kind: "fluid-body", id: `${FLUID_VOLUME_PREFIX}0` });
  const corner = entity.handles.find((handle) => handle.id === "+++");
  assert.ok(corner, "a ball offers resize handles");
  const reach = before.radius_m * 2;
  const next = apply(scene, corner.drag({
    x: before.center_m.x + reach, y: before.center_m.y + reach, z: before.center_m.z + reach,
  }, undefined));
  const after = volumes(next)[0]!;
  assert.equal(after.shape, "sphere");
  assert.ok(after.radius_m > before.radius_m, "the ball should have grown");
  assert.deepEqual(after.center_m, before.center_m, "a linked resize is about the centre");
  assert.deepEqual(validateScene(next), []);
});

test("a move writes a ball back at its own index, leaving its siblings alone", () => {
  const scene = preset("cm12-figure-3");
  const before = volumes(scene);
  assert.ok(before.length >= 3, "figure 3 throws several balls");
  const entity = entityFor(scene, { kind: "fluid-body", id: `${FLUID_VOLUME_PREFIX}1` });
  const centre = entity.handles.find((handle) => handle.space === "world");
  assert.ok(centre, "a ball can be moved");
  // Figure 3 is a 2D case, so its balls are disks extruded through the slab.
  // A move must hold whatever shape the author wrote, and the slab is the
  // whole depth, so the drag stays on its mid-plane.
  const target = { x: 0.25, y: 1.5, z: 0 };
  const next = apply(scene, centre.drag(target, undefined));
  const after = volumes(next);
  assert.equal(after.length, before.length);
  after.forEach((volume, index) => {
    if (index === 1) return;
    assert.deepEqual(volume, before[index], `volume ${index} should not have moved`);
  });
  const moved = after[1]!;
  assert.equal(moved.shape, before[1]!.shape, "a move must not reshape the volume");
  assert.ok(moved.shape !== "box");
  assert.deepEqual(moved.center_m, target);
  assert.deepEqual(validateScene(next), []);
});

test("a 2D figure's disk is picked and bounded as the disk it is", () => {
  // The 2D reconstructions extrude their ball through the slab, so the volume
  // the editor draws and clicks has to be that column and not the ball the 3D
  // figures use — otherwise Figure 2's water reads as a fraction of its size.
  const scene = preset("cm12-figure-2");
  const body = fluidVolumeBodies(scene)[0]!;
  const disk = body.volume;
  assert.equal(disk.shape, "cylinder");
  assert.equal(body.box.max.z - body.box.min.z, scene.container.depth_m,
    "the disk spans the slab");
  assert.equal(body.box.max.x - body.box.min.x, 2 * disk.radius_m);
  assert.equal(body.resizable, false, "a slab-spanning disk is sized by its radius, not by a face drag");
  // Down the axis, through the flat cap.
  assert.ok(pickFluidVolume(
    { origin: { ...disk.center_m, z: disk.center_m.z + 40 }, direction: { x: 0, y: 0, z: -1 } },
    disk) !== undefined);
  // Along the slab, past the rim: inside the box, outside the water.
  assert.equal(pickFluidVolume(
    { origin: { x: disk.center_m.x + 0.99 * disk.radius_m, y: body.box.max.y, z: 40 },
      direction: { x: 0, y: 0, z: -1 } },
    disk), undefined);
  assert.ok(Math.abs(fluidVolumeVolume_m3(disk)
    - Math.PI * disk.radius_m ** 2 * scene.container.depth_m) < 1e-12);
});

test("a move clamps the centre to the container rather than the whole ball", () => {
  const scene = preset(FIGURE_7);
  const entity = entityFor(scene, { kind: "fluid-body", id: `${FLUID_VOLUME_PREFIX}0` });
  const centre = entity.handles.find((handle) => handle.space === "world");
  assert.ok(centre);
  const c = scene.container;
  const next = apply(scene, centre.drag({ x: 500, y: 500, z: 500 }, undefined));
  const moved = volumes(next)[0]!;
  assert.equal(moved.shape, "sphere");
  assert.deepEqual(moved.center_m, { x: c.width_m / 2, y: c.height_m, z: c.depth_m / 2 });
  // A ball resting against a wall overhangs it, which the schema allows and a
  // scene that starts half-submerged needs.
  assert.deepEqual(validateScene(next), []);
});

test("a dome moves but is never resized by the box around it", () => {
  const scene = preset(FIGURE_8);
  const dome = volumes(scene)[0]!;
  assert.equal(dome.shape, "hemisphere");
  const body = fluidVolumeBodies(scene)[0]!;
  assert.equal(body.resizable, false);
  const entity = entityFor(scene, { kind: "fluid-body", id: body.id });
  assert.equal(entity.handles.every((handle) => handle.space === "world"), true,
    "a dome offers move handles only");
  assert.ok(entity.fields?.some((field) => field.id === "radius"),
    "and takes its radius from the field instead");
});

test("a dome's bounds are the half it keeps, not the ball it came from", () => {
  const scene = preset(FIGURE_8);
  const dome = volumes(scene)[0]!;
  assert.equal(dome.shape, "hemisphere");
  const box = fluidVolumeBox(dome);
  // Figure 8 keeps x < 0, so the +x side of the box is the cut plane.
  assert.ok(Math.abs(box.max.x - dome.center_m.x) < 1e-9, "the cut face is at the centre");
  assert.ok(Math.abs(box.min.x - (dome.center_m.x - dome.radius_m)) < 1e-9);
  for (const axis of ["y", "z"] as const) {
    assert.ok(Math.abs(box.max[axis] - (dome.center_m[axis] + dome.radius_m)) < 1e-9, axis);
    assert.ok(Math.abs(box.min[axis] - (dome.center_m[axis] - dome.radius_m)) < 1e-9, axis);
  }
});

test("dropping a ball appends one, and the id it reports resolves to it", () => {
  const scene = preset(FIGURE_7);
  const before = volumes(scene).length;
  const radius_m = defaultFluidBallRadius_m(scene);
  assert.ok(radius_m > 0);
  const centre = { x: 0, y: 0.75 * scene.container.height_m, z: 0 };
  const dropped = addFluidBall(scene, centre, radius_m);
  const next: SceneDescription = { ...scene, fluid: dropped.fluid };
  assert.equal(volumes(next).length, before + 1);
  assert.deepEqual(volumes(next).slice(0, before), volumes(scene));
  const entity = entityFor(next, { kind: "fluid-body", id: dropped.id });
  assert.equal(entity.label, "WATER 2", "it is numbered in with the ball already there");
  assert.deepEqual(validateScene(next), []);
});

test("a drop into a 2D case is the disk that case's water is", () => {
  // Figure 2 is a 2D reconstruction run as an 8-cell slab. A ball dropped into
  // it would be water the case cannot hold: not extruded through the slab, and
  // sized against a depth that is the thickness of the plane rather than a
  // dimension of the scene — which is a two-cell ball.
  const scene = preset("cm12-figure-2");
  assert.ok(scene.container.depth_m < scene.container.width_m / 4, "figure 2 is a slab");
  const radius_m = defaultFluidBallRadius_m(scene);
  assert.ok(radius_m > 4 * scene.voxelDomain.finestCellSize_m,
    "the standard drop is sized across the plane, not through the slab");
  const dropped = fluidDropVolume(scene, { x: 0.5, y: 4, z: 0 }, radius_m);
  assert.equal(dropped.shape, "cylinder");
  assert.equal(dropped.shape === "cylinder" && dropped.halfHeight_m, scene.container.depth_m / 2);
  assert.deepEqual(validateScene({ ...scene, fluid: addFluidBall(scene, { x: 0.5, y: 4, z: 0 }, radius_m).fluid }), []);
  // A real volume still gets a ball.
  assert.equal(fluidDropVolume(preset(FIGURE_7), { x: 0, y: 4, z: 0 }, radius_m).shape, "sphere");
});

test("a ball dropped past a wall lands on it instead of failing validation", () => {
  const scene = preset(FIGURE_7);
  const dropped = addFluidBall(scene, { x: 1e3, y: -1e3, z: 0 }, defaultFluidBallRadius_m(scene));
  assert.deepEqual(validateScene({ ...scene, fluid: dropped.fluid }), []);
});

test("removing the only volume drops the field rather than leaving it empty", () => {
  const scene = preset(FIGURE_7);
  const entity = entityFor(scene, { kind: "fluid-body", id: `${FLUID_VOLUME_PREFIX}0` });
  assert.ok(entity.remove, "a ball can be deleted");
  const next = entity.remove();
  assert.equal(next.fluid.initialLiquidVolumes, undefined);
  assert.deepEqual(validateScene(next), []);
});

test("the water readout counts a ball as a ball, not as the box around it", () => {
  const scene = preset(FIGURE_7);
  const ball = volumes(scene)[0]!;
  assert.equal(ball.shape, "sphere");
  const expected = (4 / 3) * Math.PI * ball.radius_m ** 3;
  assert.ok(Math.abs(fluidVolumeVolume_m3(ball) - expected) < 1e-12);
  // The dry box holds nothing else, so the scene's water is exactly the ball.
  assert.ok(Math.abs(fluidWaterVolume_m3(scene) - expected) < 1e-9);
});

test("balls are surfaced by the same definition as every other body of water", () => {
  const kinds = EDITOR_ENTITIES.filter((definition) => definition.kind === "fluid-body");
  assert.equal(kinds.length, 1, "one definition owns every body of water");
  assert.equal(kinds[0]!.surfacedBy("select", undefined), true);
});
