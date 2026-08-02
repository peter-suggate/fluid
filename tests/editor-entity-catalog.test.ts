import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import { EDITOR_ENTITIES, entityAtRay, findEntity, surfacedEntities } from "../lib/editor-entity-catalog";
import { FLUID_BODY_SELECTION_ID } from "../lib/editor-fluid-body";
import { TANK_SELECTION_ID, tankBox } from "../lib/editor-tank";
import { getScenePreset } from "../lib/scenes";
import { context } from "./helpers/editor-entities";
import type { SceneDescription } from "../lib/model";

function preset(id: string): SceneDescription {
  return getScenePreset(id).create();
}

const WATER = { kind: "fluid-body", id: FLUID_BODY_SELECTION_ID } as const;
const TANK = { kind: "tank", id: TANK_SELECTION_ID } as const;

test("every entity is composed once, under its own kind", () => {
  const kinds = EDITOR_ENTITIES.map((definition) => definition.kind);
  assert.equal(new Set(kinds).size, kinds.length, "a kind is a selection kind, so it must be unique");
  assert.ok(kinds.includes("fluid-body"));
  assert.ok(kinds.includes("tank"));
});

test("the water body is composed before the tank that encloses it", () => {
  const kinds = EDITOR_ENTITIES.map((definition) => definition.kind);
  assert.ok(kinds.indexOf("fluid-body") < kinds.indexOf("tank"),
    "a pointer within tolerance of both is over the body actually visible there");
});

test("the catalog is a static composition, not a mutable registry", () => {
  // As with the resource plugins: import order and hot reload must not be able
  // to change what the editor offers.
  const source = readFileSync(new URL("../lib/editor-entity-catalog.ts", import.meta.url), "utf8");
  assert.match(source, /EDITOR_ENTITIES[^=]*= Object\.freeze\(\[/, "the catalog itself must be frozen");
  assert.doesNotMatch(source.replace(/\/\*[\s\S]*?\*\//g, ""), /EDITOR_ENTITIES\.(push|splice|unshift)/,
    "there must be no runtime registration");
  assert.equal(Object.isFrozen(EDITOR_ENTITIES), true);
  assert.throws(() => (EDITOR_ENTITIES as unknown as unknown[]).push({}));
});

test("every entity that can be selected can also be clicked", () => {
  // An entity whose handles appear only once it is selected, and which no click
  // can select, is unreachable — so a missing `pick` is a bug, not a choice.
  for (const definition of EDITOR_ENTITIES) {
    assert.ok(definition.pick, `${definition.kind} offers no way to select it`);
  }
});

test("handles belong to the selection and to nothing else", () => {
  const scene = preset("water-box-dam-break");
  assert.deepEqual(surfacedEntities(context(scene), "select", undefined), [],
    "with nothing selected the scene must be free of handles");

  const water = surfacedEntities(context(scene), "select", WATER);
  assert.equal(water.length, 1);
  assert.equal(water[0]!.selection.kind, "fluid-body");
  assert.ok(water[0]!.handles.length > 0);

  const tank = surfacedEntities(context(scene), "select", TANK);
  assert.equal(tank.length, 1);
  assert.equal(tank[0]!.selection.kind, "tank", "and only the one that is selected");
});

test("a tool that does not surface an entity draws none of its handles", () => {
  const scene = preset("water-box-dam-break");
  assert.deepEqual(surfacedEntities(context(scene), "fluid-paint", WATER), [],
    "painting water must not put resize handles under the brush");
});

test("a selection resolves whether or not the tool surfaces it", () => {
  // A selection outlives the tool that made it: the flyout and the keyboard both
  // have to resolve one after the mode that drew its handles has been left.
  const scene = preset("water-box-dam-break");
  assert.equal(findEntity(context(scene), WATER)?.label, "WATER");
  assert.equal(findEntity(context(scene), TANK)?.label, "TANK");
  assert.equal(findEntity(context(scene), undefined), undefined);
  assert.equal(findEntity(context(scene), { kind: "tank", id: "not-the-tank" }), undefined);
});

test("a click resolves to whatever is nearest, so the tank is what is left", () => {
  const scene = preset("water-box-dam-break");
  const box = tankBox(scene);
  // Down the far +x +z column of the tank: the water is in the -x -z corner, so
  // this meets only the floor. It starts just above the tank and not further,
  // because the studio hangs a softbox two metres up and that softbox is
  // selectable scenery like anything else — a ray from higher would reach it
  // first, correctly.
  const corner = entityAtRay(context(scene), {
    origin: { x: 0.45 * box.max.x, y: box.max.y + 0.5, z: 0.45 * box.max.z },
    direction: { x: 0, y: -1, z: 0 },
  });
  assert.equal(corner?.selection.kind, "tank", "the floor is the tank, and it is behind everything");

  // Through the reservoir itself, the water is in front and wins. Started
  // inside the studio rather than five metres outside it: the cyclorama stands
  // between, and it is scenery a click can now reach.
  const water = entityAtRay(context(scene), {
    origin: { x: 0.9 * box.min.x, y: 0.1 * box.max.y, z: box.min.z - 0.5 },
    direction: { x: 0, y: 0, z: 1 },
  });
  assert.equal(water?.selection.kind, "fluid-body");
});

test("a ray that leaves the tank entirely selects nothing, which is how a click deselects", () => {
  const scene = preset("water-box-dam-break");
  const box = tankBox(scene);
  // Up and out past the lighting rig, which stops over the tank: everything the
  // studio contains is behind this ray, and nothing is in front of it.
  const sky = entityAtRay(context(scene), {
    origin: { x: box.max.x + 0.4, y: box.max.y + 1, z: box.max.z + 0.6 },
    direction: { x: 0, y: 1, z: 0 },
  });
  assert.equal(sky, undefined);
});
