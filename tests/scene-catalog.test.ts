import assert from "node:assert/strict";
import test from "node:test";
import { cameraPosition, pan } from "../lib/math";
import { canonicalScene, parseScene, serializeScene, validateScene } from "../lib/model";
import { SCENERY_GENERATORS } from "../lib/scenery-generators";
import { SVO_PRIMITIVE_CANDIDATE_MAXIMUM_LEAVES } from "../lib/svo-primitive-candidates";
import { SCENERY_GENERATOR_IDS, validateSceneryGraph, type SceneryGraph } from "../lib/scenery-graph";
import {
  SCENE_AUDIENCES,
  defineScene,
  duplicateSceneDefinitionIds,
  sceneDefinitionCamera,
  sceneDocument,
  sceneShelves,
  type SceneAudience,
  type SceneDefinition,
} from "../lib/scene-definition";
import {
  SCENE_CATALOG,
  defaultScenePresetId,
  getSceneDefinition,
  getScenePreset,
  sceneCatalogCards,
  scenePresets,
} from "../lib/scenes";
import { buildEnvironmentProxyCatalog, environmentProxyPrimitives } from "../lib/voxel-environments";

const AUDIENCE_IDS = new Set<SceneAudience>(SCENE_AUDIENCES.map(({ id }) => id));

/** A minimal well-formed definition, so a rejection test isolates one field. */
function definition(patch: Partial<SceneDefinition> = {}): SceneDefinition {
  return defineScene({
    id: "probe",
    name: "Probe",
    blurb: "A scene that exists only to exercise the guard.",
    audience: "explore",
    shelf: "Tanks",
    environment: "default",
    build: () => getScenePreset("water-box-dam-break").create(),
    ...patch,
  });
}

test("every scene has one identity", () => {
  assert.deepEqual(duplicateSceneDefinitionIds(SCENE_CATALOG), []);
  assert.equal(new Set(SCENE_CATALOG.map(({ name }) => name)).size, SCENE_CATALOG.length,
    "two scenes with the same name are indistinguishable on a card");
});

test("every catalog entry builds a document the schema accepts", () => {
  for (const entry of SCENE_CATALOG) {
    assert.deepEqual(validateScene(sceneDocument(entry)), [], `${entry.id} must build a valid document`);
  }
});

test("every catalog document owns its scenery rather than naming a background", () => {
  // The whole reason the smoke catalog's local factories diverged: assigning
  // `environment` names a background without copying the graph it implies.
  for (const entry of SCENE_CATALOG) {
    const scene = sceneDocument(entry);
    assert.equal(scene.environment, entry.environment, `${entry.id} must carry its environment id`);
    assert.ok(scene.scenery, `${entry.id} must carry the scenery graph its environment seeds`);
  }
});

test("every water scene has an enclosing SVO shell unless terrain is its enclosure", () => {
  for (const entry of SCENE_CATALOG) {
    const scene = sceneDocument(entry);
    if (scene.systems?.fluid === false) continue;
    const shell = scene.scenery?.nodes.find((node) =>
      node.kind === "room-shell" || node.kind === "terrain-shell");
    if (scene.terrain) {
      assert.equal(shell?.kind, "terrain-shell", `${entry.id} must publish its authored terrain as the SVO shell`);
    } else {
      assert.equal(shell?.kind, "room-shell", `${entry.id} must place water in a bounded SVO room`);
    }
  }
});

test("the default water-box cameras start inside the white room with panning headroom", () => {
  for (const id of ["water-box-dam-break", "twin-dam-collision"]) {
    const entry = getSceneDefinition(id);
    const scene = sceneDocument(entry);
    const shell = buildEnvironmentProxyCatalog(scene, entry.environment).shell;
    assert.equal(shell.kind, "room");
    if (shell.kind !== "room") continue;
    const camera = sceneDefinitionCamera(entry);
    for (const candidate of [camera, pan(camera, 100, 0), pan(camera, -100, 0), pan(camera, 0, 100), pan(camera, 0, -100)]) {
      const position = cameraPosition(candidate);
      for (const axis of ["x", "y", "z"] as const) {
        assert.ok(position[axis] > shell.bounds_m.min[axis] && position[axis] < shell.bounds_m.max[axis],
          `${id} camera must remain inside the room after a modest ${axis}-axis pan`);
      }
    }
  }
});

test("building a scene twice yields the same document", () => {
  for (const entry of SCENE_CATALOG) {
    assert.equal(canonicalScene(sceneDocument(entry)), canonicalScene(sceneDocument(entry)),
      `${entry.id} must not depend on call order or ambient state`);
  }
});

test("every scene declares a known audience, and the oracles stay in validation", () => {
  for (const entry of SCENE_CATALOG) {
    assert.ok(AUDIENCE_IDS.has(entry.audience), `${entry.id} declares an unknown audience`);
    assert.ok(entry.shelf.trim().length > 0, `${entry.id} needs a shelf`);
    // A guard rather than taste: an analytic oracle that lands on the Explore
    // shelf is thirteen ceiling-drop tests back in the product's front door.
    if (entry.name.startsWith("Octree · ")) {
      assert.equal(entry.audience, "validation", `${entry.id} is an oracle and belongs behind the disclosure`);
    }
  }
  assert.ok(SCENE_CATALOG.some((entry) => entry.audience === "explore"));
  assert.ok(SCENE_CATALOG.some((entry) => entry.audience === "study"));
  assert.ok(SCENE_CATALOG.some((entry) => entry.audience === "validation"));
  assert.equal(SCENE_AUDIENCES.find(({ id }) => id === "validation")?.disclosed, true,
    "validation scenes are disclosed, never removed");
});

test("a numerical scene carries the solver profile it requires", () => {
  // Loading a saved scene used to drop this, so a validation document could run
  // under whatever method happened to be selected.
  for (const entry of SCENE_CATALOG) {
    if (entry.audience !== "validation") continue;
    if (!entry.name.startsWith("Octree · ")) continue;
    assert.ok(entry.methodProfile, `${entry.id} is a power-diagram oracle and must pin its profile`);
    assert.equal(entry.methodProfile?.methodId, "octree");
  }
});

test("the preset projection agrees with the catalog", () => {
  assert.equal(scenePresets.length, SCENE_CATALOG.length);
  for (const [index, entry] of SCENE_CATALOG.entries()) {
    const preset = scenePresets[index];
    assert.equal(preset.id, entry.id, "preset order is catalog order");
    assert.equal(preset.group, entry.shelf);
    assert.equal(preset.description, entry.blurb);
    assert.equal(preset.background, entry.environment);
    assert.equal(preset.presentationMode, entry.presentationMode ?? "full-scene");
    assert.deepEqual(preset.methodProfile, entry.methodProfile);
    assert.equal(canonicalScene(preset.create()), canonicalScene(sceneDocument(entry)),
      `${entry.id} must be one document however it is reached`);
  }
});

test("the cold-load scene is pinned", () => {
  // `lib/gpu-startup.ts` names this id when it decides whether bring-up needs an
  // explicit start, so renaming or reordering it is a runtime behaviour change.
  assert.equal(defaultScenePresetId, "water-box-dam-break");
  assert.equal(SCENE_CATALOG[0].id, defaultScenePresetId);
  assert.equal(getScenePreset(defaultScenePresetId).presentationMode, "fluid-only",
    "the performance scene must bypass dry-world construction and presentation");
});

test("an unknown id falls back rather than throwing", () => {
  assert.equal(getSceneDefinition("no-such-scene").id, SCENE_CATALOG[0].id);
  assert.equal(getScenePreset("no-such-scene").id, scenePresets[0].id);
});

test("a card opens the same document, camera and profile as its definition", () => {
  assert.equal(sceneCatalogCards.length, SCENE_CATALOG.length);
  for (const [index, card] of sceneCatalogCards.entries()) {
    const entry = SCENE_CATALOG[index];
    const opening = card.open();
    assert.equal(card.id, entry.id);
    assert.equal(card.source, "catalog");
    assert.equal(opening.presetId, entry.id);
    assert.equal(canonicalScene(opening.scene), canonicalScene(sceneDocument(entry)));
    // Loading a preset applied these and loading a saved scene did not; one card
    // cannot afford two behaviours. The camera arrives resolved, so a scene that
    // declares none still restores the default view rather than inheriting the
    // framing of whatever was open before.
    assert.deepEqual(opening.camera, sceneDefinitionCamera(entry));
    assert.deepEqual(opening.methodProfile, entry.methodProfile);
  }
});

test("shelves group in catalog order, within one audience", () => {
  const shelves = sceneShelves(sceneCatalogCards.filter(({ audience }) => audience === "explore"));
  assert.ok(shelves.length > 1, "Explore is more than one shelf");
  assert.equal(new Set(shelves.map(({ shelf }) => shelf)).size, shelves.length, "a shelf appears once");
  for (const { shelf, cards } of shelves) {
    assert.ok(cards.length > 0);
    for (const card of cards) {
      assert.equal(card.audience, "explore");
      assert.equal(card.shelf, shelf);
    }
  }
  assert.equal(
    shelves.flatMap(({ cards }) => cards.map(({ id }) => id)).join(),
    sceneCatalogCards.filter(({ audience }) => audience === "explore").map(({ id }) => id).join(),
    "grouping must not reorder the catalog",
  );
});

test("a malformed definition fails at definition time, not on a blank card", () => {
  assert.throws(() => definition({ id: " " }), /non-empty id/);
  assert.throws(() => definition({ name: "" }), /needs a name/);
  assert.throws(() => definition({ blurb: "  " }), /needs a blurb/);
  assert.throws(() => definition({ shelf: "" }), /needs a shelf/);
  assert.throws(
    () => definition({ environment: "atlantis" as SceneDefinition["environment"] }),
    /unknown environment/,
  );
  assert.throws(() => definition({
    variants: { smoke: { id: "gpu-smoke", description: "mismatched", apply: (scene) => scene } },
  }), /differs from variant id/);
  assert.throws(() => definition({
    variants: { smoke: { id: "smoke", description: "", apply: (scene) => scene } },
  }), /needs a description/);
});

test("a variant is a delta over its own scene, and an unknown one is refused", () => {
  const entry = definition({
    variants: {
      still: {
        id: "still",
        description: "Zeroed surface tension, as the GPU lanes run it.",
        apply: (scene) => ({ ...scene, fluid: { ...scene.fluid, surfaceTension_N_m: 0 } }),
      },
    },
  });
  assert.equal(sceneDocument(entry, "still").fluid.surfaceTension_N_m, 0);
  // Everything else is the base document, including the scenery the definition
  // attaches after the delta runs.
  assert.ok(sceneDocument(entry, "still").scenery);
  assert.equal(sceneDocument(entry, "still").environment, "default");
  assert.throws(() => sceneDocument(entry, "no-such-variant"), /has no variant/);
});

test("definitions are frozen, so nothing can rewrite the catalog at runtime", () => {
  assert.ok(Object.isFrozen(SCENE_CATALOG));
  for (const entry of SCENE_CATALOG) assert.ok(Object.isFrozen(entry), `${entry.id} must be frozen`);
});

test("the scenery generator catalog is frozen and total over the ids a document may name", () => {
  assert.ok(Object.isFrozen(SCENERY_GENERATORS));
  assert.ok(Object.isFrozen(SCENERY_GENERATOR_IDS));
  assert.deepEqual(Object.keys(SCENERY_GENERATORS).sort(), [...SCENERY_GENERATOR_IDS].sort(),
    "the schema's id list and the catalog are one vocabulary, not two");
  for (const id of SCENERY_GENERATOR_IDS) {
    const generator = SCENERY_GENERATORS[id];
    assert.equal(generator.id, id, "an entry must answer to the key it is filed under");
    assert.equal(typeof generator.grow, "function");
    assert.equal(typeof generator.needsVessel, "boolean");
  }
  // No mutable `register()`, stated as the property rather than as the absence
  // of a function: import order and hot reload cannot change what a document is
  // allowed to say.
  assert.throws(() => {
    (SCENERY_GENERATORS as Record<string, unknown>)["floret-canopy"] = SCENERY_GENERATORS.bonsai;
  }, TypeError);
  const graph: SceneryGraph = {
    palettes: {},
    nodes: [
      { kind: "terrain-shell", id: "shell", materialModel: "porcelain" },
      // Cast because the whole point is a document that got past the compiler by
      // not being compiled: a parsed scene is `JSON.parse` output.
      { kind: "generator", id: "ghost", generator: "floret-canopy", seed: 1, params: {} } as never,
    ],
  };
  assert.deepEqual(validateSceneryGraph(graph), ["Scenery node ghost names unknown generator floret-canopy"]);
});

test("the hero set is a description of itself, and survives a save unchanged", () => {
  const entry = getSceneDefinition("hero-garden-hose");
  const scene = sceneDocument(entry);
  const scenery = scene.scenery;
  assert.ok(scenery);
  assert.deepEqual(
    scenery.nodes.filter((node) => node.kind === "generator").map((node) => node.generator),
    ["pond-stone-set", "rosette", "rosette", "rosette"],
    "the stones and plants are named rather than baked while the tree slot is empty");

  // The property this phase exists for. Baked, the same set is 684 nodes and
  // 884 kB of ellipsoid centres — every one of which `cloneScene` copies on
  // every edit and `localStorage` would have to hold — and re-seeding it is a
  // factory re-run that discards whatever the user changed. Described, it is
  // three nodes and a vessel. The ceiling is deliberately loose: it catches a
  // regression to baking, not an extra prop.
  // Recursive foliage is intentionally materialized as individually editable
  // document shapes. That makes the graph larger than an opaque generator but
  // still an order of magnitude smaller than a primitive bake, and every byte
  // remains high-level form, split and placement data.
  assert.ok(JSON.stringify(scenery).length < 150_000,
    `the hero scenery graph must stay a description, not a bake (${JSON.stringify(scenery).length} bytes)`);

  const before = environmentProxyPrimitives(buildEnvironmentProxyCatalog(scene, entry.environment));
  // A real set, and one that fits. The tree slot is intentionally empty while
  // the recursive replacement is authored, so this floor covers the remaining
  // stone arrangement, hose and plants. It catches a generator that quietly
  // stopped growing without preserving a record budget for a deleted species.
  // Re-baseline both bounds from the accepted frame when the new tree lands.
  assert.ok(before.length > 200, `the generators publish a stub: ${before.length} primitives`);
  assert.ok(before.length < 0.6 * SVO_PRIMITIVE_CANDIDATE_MAXIMUM_LEAVES,
    `the hero set publishes ${before.length} of ${SVO_PRIMITIVE_CANDIDATE_MAXIMUM_LEAVES} records`);
  // Through the real save path, because that is what a generator node has to
  // survive: a document is stored as raw JSON and read back with `parseScene`.
  const reloaded = parseScene(serializeScene(scene));
  const after = environmentProxyPrimitives(buildEnvironmentProxyCatalog(reloaded, entry.environment));
  assert.deepEqual(after.map(({ key }) => key), before.map(({ key }) => key),
    "owner order is the GPU material table's own addressing; it cannot move across a save");
  assert.deepEqual(after.map(({ center_m }) => center_m), before.map(({ center_m }) => center_m));
});
