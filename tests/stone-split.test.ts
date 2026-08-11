import assert from "node:assert/strict";
import test from "node:test";

import { HERO_GARDEN_SET_SEED, HERO_GARDEN_VESSEL } from "../lib/hero-garden-scene";
import { getScenePreset } from "../lib/scenes";
import { findSceneryNode } from "../lib/scenery-edit";
import {
  growSceneryGenerator,
  type SceneryGeneratorRequest,
} from "../lib/scenery-generators";
import {
  sceneStoneNode,
  sceneStoneQuery,
  stoneDials,
  withSceneStoneQuery,
  withStoneDials,
  withStoneSeedRerolled,
} from "../lib/stone-look-controls";
import { parseQueryState, serializeQueryState } from "../lib/url-state";
import { stoneSetBoulderPlan } from "../lib/voxel-scenery/stone-set";

/**
 * The pond stone set is split so each boulder is its own selectable node. Two
 * things carry the split: the split document must publish exactly what the
 * unsplit set published (owner indices are material addresses, so a changed
 * node count or order restyles the scene), and the look dials over each
 * boulder must be an honest projection — authored forms read back interior,
 * settings round-trip, and no reachable setting leaves the family's band.
 */

const heroWaterline = (): number => {
  const beds = findSceneryNode(getScenePreset("hero-garden-hose").create(), "stone");
  assert.ok(beds && beds.kind === "generator" && beds.generator === "pond-stone-set");
  return beds.params.waterline_m;
};

const request = (key: string, seed: number): SceneryGeneratorRequest => ({
  key,
  seed,
  groundHeightAt: () => 0,
  detailCellSize_m: 0.025,
  vessel: () => HERO_GARDEN_VESSEL,
});

test("the split arrangement publishes exactly what the whole set did", () => {
  const waterline_m = heroWaterline();
  const unsplit = growSceneryGenerator(
    "pond-stone-set", { waterline_m }, request("stone", HERO_GARDEN_SET_SEED));
  const plan = stoneSetBoulderPlan({
    vessel: HERO_GARDEN_VESSEL, waterline_m, seed: HERO_GARDEN_SET_SEED,
  });
  const split = [
    ...plan.flatMap((boulder) => growSceneryGenerator(
      "capped-boulder",
      { ...boulder.form, capRadius_m: boulder.capRadius_m, at_m: boulder.at_m },
      request(`stone/boulder-${boulder.index}`, boulder.seed),
    )),
    ...growSceneryGenerator(
      "pond-stone-set", { waterline_m, families: ["beds"] }, request("stone", HERO_GARDEN_SET_SEED)),
    ...growSceneryGenerator(
      "pond-stone-set", { waterline_m, families: ["path"] }, request("stone-path", HERO_GARDEN_SET_SEED)),
  ];
  // The path family now publishes under its own node id; the rename is the one
  // permitted difference, so compare with its prefix folded back.
  const fold = (nodes: unknown) => JSON.parse(JSON.stringify(nodes)
    .replaceAll("\"stone-path/", "\"stone/")
    .replaceAll("\"stone-path\"", "\"stone\""));
  assert.deepEqual(fold(split), fold(unsplit));
  assert.ok(plan.length > 0, "the split must actually contain boulders");
});

test("the hero document authors each boulder as its own top-level node", () => {
  const scene = getScenePreset("hero-garden-hose").create();
  const plan = stoneSetBoulderPlan({
    vessel: HERO_GARDEN_VESSEL, waterline_m: heroWaterline(), seed: HERO_GARDEN_SET_SEED,
  });
  assert.equal(plan.length, 4, "all four hero boulders stand on dry bank");
  const ids = scene.scenery!.nodes.map((node) => node.id);
  for (const boulder of plan) {
    const id = `stone/boulder-${boulder.index}`;
    assert.ok(ids.includes(id), `${id} must be a top-level (selectable) node`);
    const node = sceneStoneNode(scene, id);
    assert.ok(node, `${id} must be a capped-boulder generator node`);
    assert.equal(node.seed, boulder.seed, "the set's own seed formula");
    assert.deepEqual(node.params.at_m, boulder.at_m);
    assert.equal(node.params.capRadius_m, boulder.capRadius_m);
  }
  // Publication order is the owner order: boulders, then beds, then path.
  const stoneIds = ids.filter((id) => id.startsWith("stone"));
  assert.deepEqual(stoneIds, [
    "stone/boulder-0", "stone/boulder-1", "stone/boulder-2", "stone/boulder-3",
    "stone", "stone-path",
  ]);
});

test("each authored boulder form reads back as interior dial positions", () => {
  const scene = getScenePreset("hero-garden-hose").create();
  for (let index = 0; index < 4; index += 1) {
    const node = sceneStoneNode(scene, `stone/boulder-${index}`)!;
    const dials = stoneDials(node.params);
    for (const [name, value] of Object.entries(dials)) {
      assert.ok(value > 0.05 && value < 0.95,
        `boulder-${index} ${name} pinned at ${value}`);
    }
  }
});

test("stone dials round-trip through the document", () => {
  const set = { size: 0.35, squash: 0.85, lip: 0.15 };
  const edited = withStoneDials(
    getScenePreset("hero-garden-hose").create(), "stone/boulder-1", set);
  const read = stoneDials(sceneStoneNode(edited, "stone/boulder-1")!.params);
  assert.ok(Math.abs(read.size - set.size) < 1e-9);
  assert.ok(Math.abs(read.squash - set.squash) < 1e-9);
  assert.ok(Math.abs(read.lip - set.lip) < 1e-9);
});

test("every reachable setting stays inside the family's band and still grows", () => {
  const scene = getScenePreset("hero-garden-hose").create();
  for (const size of [0, 0.5, 1]) for (const squash of [0, 1]) for (const lip of [0, 1]) {
    const edited = withStoneDials(scene, "stone/boulder-0", { size, squash, lip });
    const params = sceneStoneNode(edited, "stone/boulder-0")!.params;
    assert.ok(params.capRadius_m >= 0.02 && params.capRadius_m <= 0.12);
    assert.ok(params.capFlatten >= 0.34 && params.capFlatten <= 0.80);
    assert.ok(params.stemHeightShare >= 0.15 && params.stemHeightShare <= 1);
    assert.ok(params.stemBaseShare > params.stemTopShare,
      "the foot must stay wider than the waist");
    // The setting must actually grow into geometry, not merely validate.
    const nodes = growSceneryGenerator("capped-boulder", params, request("probe", 7));
    assert.ok(nodes.length > 0);
  }
});

test("dials leave the numbers they do not own untouched", () => {
  const scene = getScenePreset("hero-garden-hose").create();
  const before = sceneStoneNode(scene, "stone/boulder-2")!.params;
  const edited = withStoneDials(scene, "stone/boulder-2", { size: 0.9, squash: 0.1, lip: 0.8 });
  const after = sceneStoneNode(edited, "stone/boulder-2")!.params;
  assert.deepEqual(after.at_m, before.at_m, "a look dial never moves the stone");
  assert.equal(after.plinthShare, before.plinthShare);
  assert.equal(after.capSeatShare, before.capSeatShare);
  assert.equal(after.lean_rad, before.lean_rad);
});

test("stone looks round-trip through the compact query form", () => {
  const base = getScenePreset("hero-garden-hose").create();
  const edited = withStoneSeedRerolled(
    withStoneDials(base, "stone/boulder-3", { size: 0.62, squash: 0.4, lip: 0.75 }),
    "stone/boulder-3");
  const query = sceneStoneQuery(edited);
  const restored = withSceneStoneQuery(getScenePreset("hero-garden-hose").create(), query);
  const dials = stoneDials(sceneStoneNode(restored, "stone/boulder-3")!.params);
  assert.ok(Math.abs(dials.size - 0.62) < 1e-9);
  assert.ok(Math.abs(dials.squash - 0.4) < 1e-9);
  assert.ok(Math.abs(dials.lip - 0.75) < 1e-9);
  assert.equal(
    sceneStoneNode(restored, "stone/boulder-3")!.seed,
    sceneStoneNode(edited, "stone/boulder-3")!.seed,
    "the re-rolled individual must survive the round-trip");
  // Applying a carried value re-reads as the same string.
  assert.equal(sceneStoneQuery(restored), query);
  // Junk neither throws nor touches the document.
  assert.equal(withSceneStoneQuery(edited, "no-such~1,1,1,5;garbage"), edited);
});

test("an edited stone rides the URL and an untouched set stays out of it", () => {
  const method = { methodId: "octree" as const, quality: "balanced" as const, overrides: {} };
  const preset = getScenePreset("hero-garden-hose");
  const pristine = new URLSearchParams(serializeQueryState("", {
    presetId: "hero-garden-hose", scene: preset.create(),
  }, method));
  assert.equal(pristine.get("stones"), null, "authored looks must not dirty the URL");

  const edited = withStoneDials(
    preset.create(), "stone/boulder-0", { size: 0.5, squash: 0.9, lip: 0.3 });
  const query = serializeQueryState("", { presetId: "hero-garden-hose", scene: edited }, method);
  const stones = new URLSearchParams(query).get("stones");
  assert.ok(stones?.includes("stone/boulder-0~0.5,0.9,0.3"), `stones key was ${stones}`);

  const rehydrated = parseQueryState(`?${query}`);
  const dials = stoneDials(sceneStoneNode(rehydrated.scene, "stone/boulder-0")!.params);
  assert.ok(Math.abs(dials.size - 0.5) < 1e-9);
  assert.ok(Math.abs(dials.squash - 0.9) < 1e-9);
  assert.ok(Math.abs(dials.lip - 0.3) < 1e-9);
});

test("stone looks survive a lattice re-author carried by the same URL", () => {
  // Same loop as the canopy: reshape a stone, then change the environment
  // level. Hydration applies the looks to the document the lattice rebuilt.
  const base = getScenePreset("hero-garden-hose").create();
  const halfLattice = {
    ...base.voxelDomain,
    detailCellSize_m: (base.voxelDomain.detailCellSize_m ?? base.voxelDomain.finestCellSize_m) / 2,
  };
  const query = sceneStoneQuery(withStoneDials(
    base, "stone/boulder-2", { size: 0.25, squash: 0.7, lip: 0.6 }));
  const parsed = parseQueryState(
    `?scene=hero-garden-hose&stones=${encodeURIComponent(query)}`
    + `&scene.voxelDomain=${encodeURIComponent(JSON.stringify(halfLattice))}`);
  assert.equal(parsed.scene.voxelDomain.detailCellSize_m, halfLattice.detailCellSize_m,
    "the lattice request must have re-authored the document");
  const dials = stoneDials(sceneStoneNode(parsed.scene, "stone/boulder-2")!.params);
  assert.ok(Math.abs(dials.size - 0.25) < 1e-9, `size ${dials.size}`);
  assert.ok(Math.abs(dials.squash - 0.7) < 1e-9, `squash ${dials.squash}`);
  assert.ok(Math.abs(dials.lip - 0.6) < 1e-9, `lip ${dials.lip}`);
});
