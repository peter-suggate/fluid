import assert from "node:assert/strict";
import test from "node:test";
import { fluidBodyEntity } from "../lib/core/editor-fluid-body";
import { tankEntity, TANK_SELECTION_ID } from "../lib/core/editor-tank";
import { getScenePreset } from "../lib/core/scenes";
import type { EditorEntityContext } from "../lib/core/editor-entity";
import type { SceneDescription } from "../lib/core/model";

/**
 * Where the water's own settings are, and that they are somewhere.
 *
 * Density, viscosity, surface tension and gravity used to hang off the tank,
 * because the tank was the only thing on the strip that could hold them. They
 * belong to the water — a reader who wants thicker water selects the water —
 * but the water is an *object*, and a scene can have none: an empty tank, a
 * document whose brick seeds replaced the base condition, a renderer-only set.
 * Moving them created a way for four settings to leave the editor entirely, and
 * nothing about that failure is visible: the strip simply has one fewer row.
 *
 * So this asserts the invariant rather than the layout — the material is
 * reachable in every scene, from the water where there is water and from the
 * tank where there is none, and never from both at once.
 */

const context = (scene: SceneDescription): EditorEntityContext =>
  ({ scene, pickingAvailable: true, bodies: [] });

const MATERIAL_FIELDS = ["density", "viscosity", "surface-tension", "gravity"];

/** Every group of an entity, by id, or undefined when it has no such entity. */
function groupIds(entity: { readonly groups?: readonly { readonly id: string }[] } | undefined) {
  return entity?.groups?.map((group) => group.id);
}

function materialFieldsOf(groups: readonly { readonly id: string; readonly fields?: readonly { readonly id: string }[] }[] | undefined) {
  const material = groups?.find((group) => MATERIAL_FIELDS
    .every((id) => group.fields?.some((field) => field.id === id)));
  return material?.fields?.map((field) => field.id);
}

test("the water carries its own material, and the tank does not repeat it", () => {
  const scene = getScenePreset("water-box-dam-break").create();
  const water = fluidBodyEntity.find(context(scene), "fluid-body");
  assert.ok(water, "a filled tank has a body of water to select");
  assert.deepEqual(materialFieldsOf(water.groups), MATERIAL_FIELDS);
  // What it is comes before where it is: the start condition is a choice on the
  // water itself, not a setting of the vessel around it.
  assert.ok(water.choices?.some((choice) => choice.id === "initial-condition"));

  const tank = tankEntity.find(context(scene), TANK_SELECTION_ID);
  assert.ok(tank);
  assert.equal(materialFieldsOf(tank.groups), undefined,
    "the tank must not offer a second copy of the water's material");
  assert.deepEqual(groupIds(tank), ["container", "domain"]);
});

test("an empty tank keeps the water's settings, because nothing else can hold them", () => {
  const preset = getScenePreset("water-box-dam-break").create();
  const scene: SceneDescription = {
    ...preset,
    container: { ...preset.container, fillFraction: 0 },
    fluid: (() => {
      const { initialDamBreakDimensions_m: _drop, ...fluid } = preset.fluid;
      return fluid;
    })(),
  };
  assert.equal(fluidBodyEntity.find(context(scene), "fluid-body"), undefined,
    "the premise: an unfilled tank has no body of water to select");

  const tank = tankEntity.find(context(scene), TANK_SELECTION_ID);
  assert.ok(tank);
  assert.deepEqual(materialFieldsOf(tank.groups), MATERIAL_FIELDS);
  assert.ok(tank.groups?.some((group) => group.id === "fluid"
    && group.choices?.some((choice) => choice.id === "initial-condition")),
  "and how the water starts, so a tank can be given water that behaves as it should");
});

test("the position of a body of water is a pane, not three rows of the column", () => {
  const scene = getScenePreset("water-box-dam-break").create();
  const water = fluidBodyEntity.find(context(scene), "fluid-body");
  assert.ok(water);
  // The handles are already on screen saying where the box is. Exact entry is
  // the fallback for that gesture, one click back, rather than three lines of a
  // column that has better things to promote.
  assert.equal(water.fields, undefined);
  const place = water.groups?.find((group) => group.id === "place");
  assert.deepEqual(place?.fields?.map((field) => field.id), ["x", "y", "z"]);
});
