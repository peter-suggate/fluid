import assert from "node:assert/strict";
import test from "node:test";
import { validateScene, type SceneDescription, type Vec3 } from "../lib/model";
import { getScenePreset } from "../lib/scenes";
import { sceneCellSizes_m } from "../lib/scene-lattice";
import {
  dragFluidBodyBox,
  fluidBodyBox,
  fluidBodyBoxPatch,
  fluidBodyBoxVolume_m3,
  fluidBodyEntity,
  fluidBodyLimits,
  moveFluidBodyBox,
  scaleFluidBodyBox,
  scaleFluidBodyVolume,
  FLUID_BODY_SELECTION_ID,
} from "../lib/editor-fluid-body";
import { damBreakBoxContains, sceneDamBreakBox } from "../lib/initial-fluid";
import { context, entityFor, sides } from "./helpers/editor-entities";

function preset(id: string): SceneDescription {
  return getScenePreset(id).create();
}

const WATER = { kind: "fluid-body", id: FLUID_BODY_SELECTION_ID } as const;

function applied(scene: SceneDescription, box: ReturnType<typeof fluidBodyBox>): SceneDescription {
  assert.ok(box);
  return { ...scene, ...fluidBodyBoxPatch(scene, box) };
}

test("a face drag moves only the side it owns", () => {
  const scene = preset("water-box-dam-break");
  const box = fluidBodyBox(scene);
  assert.ok(box);
  const target = { x: box.min.x + 0.5 * (scene.container.width_m), y: 99, z: 99 };
  const dragged = dragFluidBodyBox(box, sides("+00"), target, scene);
  assert.equal(dragged.min.x, box.min.x);
  assert.deepEqual([dragged.min.y, dragged.max.y], [box.min.y, box.max.y]);
  assert.deepEqual([dragged.min.z, dragged.max.z], [box.min.z, box.max.z]);
  assert.ok(dragged.max.x > box.max.x, "the +x face should have moved outward");
});

test("a corner drag moves all three of its sides", () => {
  const scene = preset("water-box-dam-break");
  const box = fluidBodyBox(scene);
  assert.ok(box);
  const limits = fluidBodyLimits(scene);
  const target = { x: limits.max.x, y: limits.max.y, z: limits.max.z };
  const dragged = dragFluidBodyBox(box, sides("+++"), target, scene);
  assert.deepEqual(dragged.min, box.min, "the opposite corner is an anchor");
  assert.ok(dragged.max.x > box.max.x && dragged.max.y > box.max.y && dragged.max.z > box.max.z);
});

test("drags snap to the finest lattice, so a moved handle always moves the water", () => {
  const scene = preset("water-box-dam-break");
  const box = fluidBodyBox(scene);
  assert.ok(box);
  const [cellX] = sceneCellSizes_m(scene);
  const limits = fluidBodyLimits(scene);
  const dragged = dragFluidBodyBox(box, sides("+00"), { x: box.max.x + cellX * 1.4, y: 0, z: 0 }, scene);
  const offset = (dragged.max.x - limits.min.x) / cellX;
  assert.ok(Math.abs(offset - Math.round(offset)) < 1e-9, `${offset} should be a whole number of cells`);
});

test("a side pushed through the body stops at one cell rather than inverting it", () => {
  const scene = preset("water-box-dam-break");
  const box = fluidBodyBox(scene);
  assert.ok(box);
  const [cellX] = sceneCellSizes_m(scene);
  const dragged = dragFluidBodyBox(box, sides("+00"), { x: box.min.x - 10, y: 0, z: 0 }, scene);
  assert.ok(dragged.max.x > dragged.min.x);
  assert.ok(Math.abs((dragged.max.x - dragged.min.x) - cellX) < 1e-9);
});

test("handles never push the body outside the container", () => {
  const scene = preset("water-box-dam-break");
  const box = fluidBodyBox(scene);
  assert.ok(box);
  const limits = fluidBodyLimits(scene);
  const dragged = dragFluidBodyBox(box, sides("+++"), { x: 99, y: 99, z: 99 }, scene);
  const pulled = dragFluidBodyBox(dragged, sides("---"), { x: -99, y: -99, z: -99 }, scene);
  for (const axis of ["x", "y", "z"] as const) {
    assert.ok(pulled.min[axis] >= limits.min[axis] - 1e-9, `min ${axis}`);
    assert.ok(pulled.max[axis] <= limits.max[axis] + 1e-9, `max ${axis}`);
  }
  assert.deepEqual(validateScene(applied(scene, pulled)), []);
});

test("a box still in the corner stays anchored, keeping the closed-form GPU seed", () => {
  const scene = preset("water-box-dam-break");
  const box = fluidBodyBox(scene);
  assert.ok(box);
  const grown = dragFluidBodyBox(box, sides("0+0"), { x: 0, y: scene.container.height_m, z: 0 }, scene);
  const patch = fluidBodyBoxPatch(scene, grown);
  assert.equal(patch.fluid.initialDamBreakOrigin_m, undefined,
    "an anchored reservoir must not author an origin — that would cost the analytic bootstrap");
  assert.deepEqual(validateScene({ ...scene, ...patch }), []);
});

test("a box dragged off the corner authors an origin the seeding honours", () => {
  const scene = preset("water-box-dam-break");
  const box = fluidBodyBox(scene);
  assert.ok(box);
  const [cellX] = sceneCellSizes_m(scene);
  const moved = dragFluidBodyBox(box, sides("-00"), { x: box.min.x + 2 * cellX, y: 0, z: 0 }, scene);
  const next = applied(scene, moved);
  const origin = next.fluid.initialDamBreakOrigin_m;
  assert.ok(origin);
  assert.ok(Math.abs(origin.x - 2 * cellX) < 1e-9);
  assert.deepEqual(validateScene(next), []);

  // The seeding test must now exclude the cells the body was pulled off.
  const dam = sceneDamBreakBox(next);
  const nx = Math.round(next.container.width_m / next.voxelDomain.finestCellSize_m);
  assert.equal(damBreakBoxContains(dam, 0.5 / nx, 0.5, 0.5), false, "the vacated column must be dry");
  assert.equal(damBreakBoxContains(dam, (2.5) / nx, 0.5 * (dam.min.y + dam.max.y) / 1, 0.5), false);
  const inside = 0.5 * (dam.min.x + dam.max.x);
  assert.equal(damBreakBoxContains(dam, inside, 0.5 * (dam.min.y + dam.max.y), 0.5 * (dam.min.z + dam.max.z)), true);
});

test("an unmoved corner-anchored reservoir seeds exactly as it did before origins existed", () => {
  const scene = preset("water-box-dam-break");
  const dam = sceneDamBreakBox(scene);
  assert.deepEqual(dam.min, { x: 0, y: 0, z: 0 });
  const size = { x: dam.max.x, y: dam.max.y, z: dam.max.z };
  for (const fraction of [0.01, 0.2, 0.5, 0.99]) {
    assert.equal(
      damBreakBoxContains(dam, fraction, fraction, fraction),
      fraction <= size.x && fraction <= size.y && fraction <= size.z,
      `fraction ${fraction} must match the legacy corner test`);
  }
});

test("reshaping a filled tank keeps exactly the water it had", () => {
  const scene = preset("water-box-tank-fill");
  assert.equal(scene.fluid.initialCondition, "tank-fill");
  const box = fluidBodyBox(scene);
  assert.ok(box);
  const c = scene.container;
  assert.ok(Math.abs(fluidBodyBoxVolume_m3(box)
    - c.width_m * c.height_m * c.depth_m * c.fillFraction) < 1e-9);
  const next = applied(scene, box);
  assert.equal(next.fluid.initialCondition, "dam-break");
  assert.ok(Math.abs(next.container.fillFraction - c.fillFraction) < 1e-9,
    "the conversion must not change how much water there is");
  assert.deepEqual(validateScene(next), []);
});

test("grow and shrink scale about the centre and stay inside the tank", () => {
  const scene = preset("water-box-dam-break");
  const box = fluidBodyBox(scene);
  assert.ok(box);
  const limits = fluidBodyLimits(scene);
  const smaller = scaleFluidBodyBox(box, 0.5, scene);
  assert.ok(fluidBodyBoxVolume_m3(smaller) < fluidBodyBoxVolume_m3(box));
  const bigger = scaleFluidBodyBox(box, 2, scene);
  assert.ok(fluidBodyBoxVolume_m3(bigger) > fluidBodyBoxVolume_m3(box));
  for (const candidate of [smaller, bigger]) {
    for (const axis of ["x", "y", "z"] as const) {
      assert.ok(candidate.min[axis] >= limits.min[axis] - 1e-9);
      assert.ok(candidate.max[axis] <= limits.max[axis] + 1e-9);
    }
    assert.deepEqual(validateScene(applied(scene, candidate)), []);
  }
});

test("the water control doubles the water, not the edge lengths", () => {
  const scene = preset("water-box-dam-break");
  const box = fluidBodyBox(scene);
  assert.ok(box);
  const before = fluidBodyBoxVolume_m3(box);
  const doubled = scaleFluidBodyVolume(box, 2, scene);
  // Snapping to whole cells keeps this from being exact, but ×2 must read as
  // twice the water beside a litres readout — never eight times it.
  const ratio = fluidBodyBoxVolume_m3(doubled) / before;
  assert.ok(ratio > 1.5 && ratio < 2.6, `volume ratio ${ratio} should be about 2`);
  const halved = scaleFluidBodyVolume(box, 0.5, scene);
  const shrunkRatio = fluidBodyBoxVolume_m3(halved) / before;
  assert.ok(shrunkRatio > 0.3 && shrunkRatio < 0.7, `volume ratio ${shrunkRatio} should be about 0.5`);
});

test("repeated growth saturates at the container instead of failing", () => {
  const scene = preset("water-box-dam-break");
  let box = fluidBodyBox(scene);
  assert.ok(box);
  for (let step = 0; step < 6; step += 1) box = scaleFluidBodyBox(box, 2, scene);
  const limits = fluidBodyLimits(scene);
  for (const axis of ["x", "y", "z"] as const) {
    assert.ok(Math.abs(box.min[axis] - limits.min[axis]) < 1e-9);
    assert.ok(Math.abs(box.max[axis] - limits.max[axis]) < 1e-9);
  }
  assert.deepEqual(validateScene(applied(scene, box)), []);
});

test("a move slides the body without resizing it, and stops at the wall", () => {
  const scene = preset("water-box-dam-break");
  const box = fluidBodyBox(scene);
  assert.ok(box);
  const limits = fluidBodyLimits(scene);
  const size = fluidBodyBoxVolume_m3(box);

  const moved = moveFluidBodyBox(box, { x: 0.05, y: 0.12, z: -0.03 }, scene);
  assert.ok(Math.abs(fluidBodyBoxVolume_m3(moved) - size) < 1e-9, "a move must not change the volume");

  // Pushed hard against a wall it stops there rather than being squashed into it.
  const pinned = moveFluidBodyBox(box, { x: 99, y: 99, z: 99 }, scene);
  assert.ok(Math.abs(fluidBodyBoxVolume_m3(pinned) - size) < 1e-9);
  for (const axis of ["x", "y", "z"] as const) {
    assert.ok(pinned.max[axis] <= limits.max[axis] + 1e-9, `max ${axis}`);
    assert.ok(pinned.min[axis] >= limits.min[axis] - 1e-9, `min ${axis}`);
  }
});

test("a render-only scene has no shapeable body", () => {
  const scene = preset("water-box-dam-break");
  assert.equal(fluidBodyBox({ ...scene, systems: { fluid: false } }), undefined);
  assert.equal(fluidBodyEntity.instances(context({ ...scene, systems: { fluid: false } })).length, 0,
    "and therefore offers no entity to select");
});

test("the box round-trips: authoring it and reading it back returns the same box", () => {
  const scene = preset("water-box-dam-break");
  const box = fluidBodyBox(scene);
  assert.ok(box);
  const [cellX, , cellZ] = sceneCellSizes_m(scene);
  const moved = dragFluidBodyBox(
    dragFluidBodyBox(box, sides("-0-"),
      { x: box.min.x + cellX, y: 0, z: box.min.z + cellZ } as Vec3, scene),
    sides("+0+"),
    { x: box.max.x + cellX, y: 0, z: box.max.z + cellZ } as Vec3, scene);
  const next = applied(scene, moved);
  const reread = fluidBodyBox(next);
  assert.ok(reread);
  for (const axis of ["x", "y", "z"] as const) {
    assert.ok(Math.abs(reread.min[axis] - moved.min[axis]) < 1e-9, `min ${axis}`);
    assert.ok(Math.abs(reread.max[axis] - moved.max[axis]) < 1e-9, `max ${axis}`);
  }
});

test("the water body is clicked on its seed box, and the box is what the handles hold", () => {
  const scene = preset("water-box-dam-break");
  const box = fluidBodyBox(scene)!;
  const centre = {
    x: 0.5 * (box.min.x + box.max.x),
    y: 0.5 * (box.min.y + box.max.y),
    z: 0.5 * (box.min.z + box.max.z),
  };
  const hit = fluidBodyEntity.pick!(context(scene),
    { origin: { x: centre.x, y: centre.y, z: box.min.z - 5 }, direction: { x: 0, y: 0, z: 1 } });
  assert.ok(hit, "a ray through the reservoir must reach it");
  assert.equal(hit.selection.id, FLUID_BODY_SELECTION_ID);
  assert.ok(Math.abs(hit.distance_m - 5) < 1e-6, "and the hit is where the ray enters the box");

  const missed = fluidBodyEntity.pick!(context(scene),
    { origin: { x: centre.x, y: box.max.y + 10, z: box.min.z - 5 }, direction: { x: 0, y: 0, z: 1 } });
  assert.equal(missed, undefined, "a ray passing over the reservoir must not select it");

  assert.deepEqual(entityFor(scene, WATER).box, box);
});
