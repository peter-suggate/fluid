import assert from "node:assert/strict";
import test from "node:test";

import type { EnvironmentId } from "../lib/environments";
import { HERO_GARDEN_VESSEL, HERO_GARDEN_WATERLINE_M } from "../lib/hero-garden-scene";
import type { SceneDescription, Vec3 } from "../lib/model";
import { getScenePreset } from "../lib/scenes";
import { validateSceneryGraph } from "../lib/scenery-graph";
import { SVO_PRIMITIVE_CANDIDATE_MAXIMUM_LEAVES } from "../lib/svo-primitive-candidates";
import { buildEnvironmentProxyCatalog, type EnvironmentProxyPrimitive } from "../lib/voxel-environments";
import {
  HERO_LAYOUT_CONTAINER_BOUNDS,
  heroGardenInWater,
  heroGardenLayout,
  heroGardenLayoutInput,
  heroGardenLayoutPlacements,
  heroGardenLayoutWorld,
  heroGardenPlanDistance_m,
  layoutBounds,
  layoutLeafCount,
  layoutPlaceholderNodes,
  layoutSeat_m,
  layoutStatistics,
  offsetRail,
  withHeroLayout,
  type LayoutOptions,
  type LayoutPlacement,
  type LayoutSpecies,
  type LayoutWorld,
} from "../lib/voxel-scenery/hero-layout";

/**
 * Oracles for the hero scene's composition.
 *
 * The subject is a *layout*, so the properties worth pinning are the ones an eye
 * cannot check quickly and a moved control point can silently break: that every
 * object is standing on the ground the solver will collide with, that nothing but
 * the stepping stones is in the water, that nothing has walked out of the domain,
 * and that the crude stand-ins have not quietly promised the detail generators
 * more room than the composition has.
 *
 * Everything is measured through the real publication path —
 * `buildEnvironmentProxyCatalog` over the document the scene library builds —
 * rather than by re-deriving positions here. A test that recomputed the placement
 * would pass while the renderer drew something else.
 */

/** The hero document, built once: the terrain bake is not cheap and never varies. */
let heroScene: SceneDescription | undefined;
function hero(): SceneDescription {
  return heroScene ??= getScenePreset("hero-garden-hose").create();
}

let heroWorld: LayoutWorld | undefined;
function world(): LayoutWorld {
  return heroWorld ??= heroGardenLayoutWorld(hero());
}

const placements = (): readonly LayoutPlacement[] => heroGardenLayout();

function layoutProxies(options: LayoutOptions = {}): readonly EnvironmentProxyPrimitive[] {
  const scene = hero();
  const document: SceneDescription = { ...scene, scenery: withHeroLayout(scene, options) };
  const catalog = buildEnvironmentProxyCatalog(document, (scene.environment ?? "garden") as EnvironmentId);
  return catalog.primitives.filter((primitive) => primitive.tags.includes("layout"));
}

/** The proxies one row published, found by the key every one of its nodes is prefixed with. */
function proxiesFor(row: LayoutPlacement, all: readonly EnvironmentProxyPrimitive[]): readonly EnvironmentProxyPrimitive[] {
  return all.filter((primitive) => primitive.key.includes(`/${row.key}/`));
}

const speciesRows = (species: LayoutSpecies) => placements().filter((row) => row.species === species);

test("the hero table carries the object families the plan calls for, once each", () => {
  // The counts in docs/HERO_GARDEN_HOSE_SCENE_PLAN.md §1, which is what the
  // detail generators are being budgeted and built against.
  assert.equal(speciesRows("mushroom-boulder").length, 4);
  assert.equal(speciesRows("stepping-disc").length, 5);
  assert.equal(speciesRows("rosette").length, 4);
  assert.equal(speciesRows("bonsai").length, 1);
  assert.equal(speciesRows("swept-tube").length, 1);
  assert.equal(speciesRows("pebble-bed").length, 3);
  assert.equal(placements().length, 18);

  const keys = new Set(placements().map((row) => row.key));
  assert.equal(keys.size, placements().length, "keys are the generators' handles and must be unique");
  for (const row of placements()) {
    assert.ok(row.detailBudget > 0, `${row.key} needs a detail budget`);
    assert.equal(row.run !== undefined, row.species === "pebble-bed", `${row.key} carries a run exactly when it is a bed`);
  }
});

test("the table is a placement authority, not a pond ornament", () => {
  // The whole point of taking a rail: the same rows lay out around a different
  // outline without being rewritten. A pond-shaped table would be unable to.
  const square: readonly (readonly [number, number])[] = [
    [-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5],
  ];
  const rows = heroGardenLayoutPlacements({ copingRail: square, copingHalfWidth_m: 0.04 });
  assert.equal(rows.length, placements().length, "the same table, laid around a different outline");
  const bed = rows.find((row) => row.run);
  assert.ok(bed?.run, "the beds must follow the outline they were handed");
  assert.notDeepEqual(bed.run.path_m, speciesRows("pebble-bed")[0].run!.path_m,
    "a different outline must produce a different bed");

  // And every id is derived from the row's key, so a species can be instantiated
  // more than once in one scene.
  for (const node of layoutPlaceholderNodes(placements(), world())) {
    assert.ok(node.id.startsWith("layout/"), `${node.id} must be keyed by its row`);
  }
});

test("every ground-standing row is seated on the terrain the renderer draws", () => {
  for (const row of placements()) {
    if (row.datum !== "ground") continue;
    const seat = layoutSeat_m(row, world());
    const ground = world().groundHeightAt(row.at_m[0], row.at_m[1]);
    assert.equal(seat.y, ground + row.lift_m, `${row.key} must resolve its datum through the ground query`);
    // A seat is allowed to bed a little into the ground and not to float above
    // it. Twenty millimetres is under a solver cell; anything larger is an
    // authored y wearing a heightfield's name.
    assert.ok(row.lift_m <= 0.002 && row.lift_m >= -0.02, `${row.key} lift ${row.lift_m} is not a bedding offset`);
  }
});

test("the stand-ins meet the ground they stand on", () => {
  const proxies = layoutProxies();
  for (const row of placements()) {
    if (row.authority === "authored" || row.run) continue;
    const mine = proxiesFor(row, proxies);
    assert.equal(mine.length, layoutLeafCount(row, world()), `${row.key} published the wrong number of leaves`);
    const lowest = Math.min(...mine.map((primitive) => primitive.aabb_m.min.y));
    const ground = world().groundHeightAt(row.at_m[0], row.at_m[1]);
    const datum = row.datum === "level" ? world().level_m : ground;
    // Down to the basin floor for a stepping stone's support, and never floating.
    assert.ok(lowest <= datum + 0.006 && lowest >= Math.min(ground, datum) - 0.14,
      `${row.key} bottoms out at ${lowest.toFixed(4)} against a datum of ${datum.toFixed(4)}`);
  }
});

test("every bed stone lies on the ground under its own centre", () => {
  const proxies = layoutProxies();
  for (const row of placements()) {
    if (!row.run) continue;
    const mine = proxiesFor(row, proxies);
    assert.equal(mine.length, layoutLeafCount(row, world()), `${row.key} published the wrong number of stones`);
    for (const stone of mine) {
      // Per stone, not per bed: a bed runs a metre and a half over ground that
      // rises to a terrace and falls to the coping's foot, so one shared datum
      // would bury half of it and float the rest.
      const ground = world().groundHeightAt(stone.center_m.x, stone.center_m.z);
      const offset = stone.aabb_m.min.y - ground;
      assert.ok(offset <= 0.001 && offset >= -0.04, `${stone.key} sits ${offset.toFixed(4)} m off its own ground`);
    }
  }
});

test("nothing but the stepping stones stands in the water", () => {
  const proxies = layoutProxies();
  for (const row of placements()) {
    if (row.authority === "authored" || row.species === "stepping-disc") continue;
    for (const primitive of proxiesFor(row, proxies)) {
      const { min, max } = primitive.aabb_m;
      if (min.y >= HERO_GARDEN_WATERLINE_M) continue;
      // Sample the footprint rather than testing a corner: the water's plan is a
      // wandering curve and a box can straddle it without any corner being wet.
      for (let i = 0; i <= 4; i += 1) {
        for (let j = 0; j <= 4; j += 1) {
          const x = min.x + (max.x - min.x) * (i / 4);
          const z = min.z + (max.z - min.z) * (j / 4);
          assert.ok(!heroGardenInWater(x, min.y, z),
            `${primitive.key} reaches into the pond at (${x.toFixed(3)}, ${z.toFixed(3)})`);
        }
      }
    }
  }
});

test("each stepping stone breaks the surface and clears the basin wall", () => {
  // Asserted on the rows rather than on published proxies, because the stone
  // families are `authored` and the layout stands in for none of them: what is
  // being pinned is the *placement* the generator will be handed.
  for (const row of speciesRows("stepping-disc")) {
    const seat = layoutSeat_m(row, world());
    assert.ok(seat.y < HERO_GARDEN_WATERLINE_M && seat.y + row.size_m[1] > HERO_GARDEN_WATERLINE_M,
      `${row.key} must be cut by the waterline, not floating over it or drowned under it`);
    // The inner face is near vertical, so a disc whose rim overlapped it would
    // read as set into the wall rather than as standing in the water.
    const radius = .5 * row.size_m[0];
    const distance = heroGardenPlanDistance_m(row.at_m[0], row.at_m[1]);
    assert.ok(distance + radius < -HERO_GARDEN_VESSEL.rimHalfWidth_m,
      `${row.key} overlaps the coping's inner foot by ${(distance + radius + HERO_GARDEN_VESSEL.rimHalfWidth_m).toFixed(4)} m`);
  }
});

test("no stand-in is drawn on top of something the scene already publishes", () => {
  // The one failure mode this arrangement invites: two authorities drawing the
  // same object. The species in this directory are landing one at a time, and as
  // each lands the hero scene starts publishing it for real — so the matching row
  // must become `authored` and stand in for nothing, or the object appears twice.
  //
  // Pinned as geometry rather than as a list of names, because a list is a second
  // place to remember and it will be forgotten on the day it matters. What
  // "drawn twice" means is an object of about the same size in about the same
  // place, so that is what is measured — per *object*, since a duplicate shares a
  // silhouette and not a primitive count. A plant nestled among pebbles is not
  // a duplicate and must not read as one, which the size test is what excludes.
  const scene = hero();
  const groups = (primitives: readonly EnvironmentProxyPrimitive[]) => {
    const boxes = new Map<string, { min: Vec3; max: Vec3 }>();
    for (const p of primitives) {
      const box = boxes.get(p.group);
      boxes.set(p.group, box ? {
        min: { x: Math.min(box.min.x, p.aabb_m.min.x), y: Math.min(box.min.y, p.aabb_m.min.y), z: Math.min(box.min.z, p.aabb_m.min.z) },
        max: { x: Math.max(box.max.x, p.aabb_m.max.x), y: Math.max(box.max.y, p.aabb_m.max.y), z: Math.max(box.max.z, p.aabb_m.max.z) },
      } : { min: { ...p.aabb_m.min }, max: { ...p.aabb_m.max } });
    }
    return boxes;
  };
  const all = buildEnvironmentProxyCatalog(
    { ...scene, scenery: withHeroLayout(scene) }, (scene.environment ?? "garden") as EnvironmentId,
  ).primitives;
  const mine = groups(all.filter((p) => p.tags.includes("layout")));
  const theirs = groups(all.filter((p) => !p.tags.includes("layout")));
  const span = (box: { min: Vec3; max: Vec3 }) => ({
    x: box.max.x - box.min.x, y: box.max.y - box.min.y, z: box.max.z - box.min.z,
  });
  const volume = (box: { min: Vec3; max: Vec3 }) => { const s = span(box); return s.x * s.y * s.z; };
  const centre = (box: { min: Vec3; max: Vec3 }) => ({
    x: .5 * (box.min.x + box.max.x), y: .5 * (box.min.y + box.max.y), z: .5 * (box.min.z + box.max.z),
  });
  for (const [key, box] of mine) {
    for (const [other, published] of theirs) {
      // Four times the volume either way is a generous "about the same size", and
      // it is three orders of magnitude away from a rosette beside a pebble bed.
      const ratio = volume(box) / Math.max(1e-12, volume(published));
      if (ratio < .25 || ratio > 4) continue;
      const a = centre(box), b = centre(published);
      const diagonal = .5 * (Math.hypot(span(box).x, span(box).y, span(box).z)
        + Math.hypot(span(published).x, span(published).y, span(published).z));
      assert.ok(Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z) > .25 * diagonal,
        `${key} stands in for ${other}, which the scene already publishes`);
    }
  }
  for (const row of placements()) {
    if (row.authority !== "authored") continue;
    assert.equal(layoutLeafCount(row, world(), "blocking"), 0, `${row.key} is authored and must emit nothing`);
  }
});

test("nothing the layout places is buried in something else", () => {
  // A separate property from duplication, and the one that made the rosettes
  // move: a plant may nestle among stones and must not be inside one.
  const scene = hero();
  const all = buildEnvironmentProxyCatalog(
    { ...scene, scenery: withHeroLayout(scene) }, (scene.environment ?? "garden") as EnvironmentId,
  ).primitives;
  const published = all.filter((p) => !p.tags.includes("layout"));
  for (const standIn of all.filter((p) => p.tags.includes("layout"))) {
    const box = standIn.aabb_m;
    const size = { x: box.max.x - box.min.x, y: box.max.y - box.min.y, z: box.max.z - box.min.z };
    const mineVolume = size.x * size.y * size.z;
    for (const other of published) {
      const overlap = ["x", "y", "z"].reduce((product, axis) => product * Math.max(0,
        Math.min(box.max[axis as "x"], other.aabb_m.max[axis as "x"])
        - Math.max(box.min[axis as "x"], other.aabb_m.min[axis as "x"])), 1);
      assert.ok(overlap < .4 * mineVolume,
        `${standIn.key} is buried in ${other.key}`);
    }
  }
});

test("nothing the layout places escapes the container", () => {
  const { min: floor, max: lid } = HERO_LAYOUT_CONTAINER_BOUNDS;
  for (const row of placements()) {
    if (row.authority === "authored") continue;
    const bounds = layoutBounds(row, world());
    assert.ok(bounds.min.x >= floor.x && bounds.max.x <= lid.x
      && bounds.min.y >= floor.y && bounds.max.y <= lid.y
      && bounds.min.z >= floor.z && bounds.max.z <= lid.z,
      `${row.key} envelope ${JSON.stringify(bounds)} leaves the container`);
  }
  for (const options of [{}, { fidelity: "budget" as const }]) {
    for (const primitive of layoutProxies(options)) {
      const { min, max } = primitive.aabb_m;
      assert.ok(min.x >= floor.x && max.x <= lid.x && min.y >= floor.y && max.y <= lid.y
        && min.z >= floor.z && max.z <= lid.z,
        `${primitive.key} leaves the container at ${JSON.stringify(primitive.aabb_m)}`);
    }
  }
});

test("a stand-in stays inside the envelope its generator is promised", () => {
  // The envelope is a contract in both directions, and this is the direction that
  // is easy to break by accident: a stand-in that overflows it teaches the
  // composition that there is room where there is not.
  const proxies = layoutProxies();
  const slack = 0.003;
  for (const row of placements()) {
    if (row.authority === "authored") continue;
    const bounds = layoutBounds(row, world());
    for (const primitive of proxiesFor(row, proxies)) {
      // The submerged supports are scaffolding for the dry render, not part of
      // the object being handed on, so they are outside the contract by design.
      if (primitive.tags.includes("scaffold")) continue;
      const { min, max } = primitive.aabb_m;
      assert.ok(min.x >= bounds.min.x - slack && max.x <= bounds.max.x + slack
        && min.y >= bounds.min.y - slack && max.y <= bounds.max.y + slack
        && min.z >= bounds.min.z - slack && max.z <= bounds.max.z + slack,
        `${primitive.key} overflows the ${row.key} envelope`);
    }
  }
});

test("the same table publishes the same nodes every time", () => {
  for (const options of [{}, { fidelity: "budget" as const }, { include: ["mushroom-boulder" as const] }]) {
    assert.deepEqual(
      layoutPlaceholderNodes(placements(), world(), options),
      layoutPlaceholderNodes(heroGardenLayout(), heroGardenLayoutWorld(hero()), options),
      "emission is the generators' seed contract and must be pure",
    );
  }
  const outline = heroGardenLayoutInput().copingRail;
  assert.deepEqual(
    offsetRail(outline, { fromTurn: 0.1, toTurn: 0.4, offset_m: 0.05, samples: 12 }),
    offsetRail(outline, { fromTurn: 0.1, toTurn: 0.4, offset_m: 0.05, samples: 12 }),
  );
});

test("the layout composes onto the hero graph without breaking it, and only once", () => {
  const scene = hero();
  for (const options of [{}, { fidelity: "budget" as const }]) {
    const scenery = withHeroLayout(scene, options);
    assert.deepEqual(validateSceneryGraph(scenery), []);
    // The scene library already composes the blocking set in, so re-composing
    // must replace rather than append; two sets sharing one id space would be a
    // duplicate-key error and a doubled primitive count.
    assert.deepEqual(withHeroLayout({ ...scene, scenery }, options).nodes, scenery.nodes);
  }
});

test("the scene the product opens is the scene the layout describes", () => {
  // The wiring the UI depends on: `hero-garden-hose` carries the set, so what the
  // preview renders and what the app shows cannot drift apart.
  const nodes = hero().scenery?.nodes ?? [];
  const composed = nodes.filter((node) => node.id.startsWith("layout/")).length;
  assert.equal(composed, layoutPlaceholderNodes(placements(), world()).length);
  assert.ok(composed > 0, "the hero scene preset must carry the layout");
});

test("the statistics count what is emitted, and the finished scene fits under the ceiling", () => {
  for (const fidelity of ["blocking", "budget"] as const) {
    const statistics = layoutStatistics(placements(), world(), { fidelity });
    assert.equal(statistics.placeholderLeaves, layoutProxies({ fidelity }).length,
      `${fidelity} statistics must count the leaves the catalog actually publishes`);
    assert.equal(statistics.placements, placements().length);
    assert.equal(statistics.ceiling, SVO_PRIMITIVE_CANDIDATE_MAXIMUM_LEAVES);
    assert.equal(statistics.detailedLeaves + statistics.headroom, statistics.ceiling);
  }

  const statistics = layoutStatistics(placements(), world());
  // The budgets as handed out: stone set 1200, bonsai 1500, planting 400, plus
  // the hose the scene already publishes.
  const budget = (species: LayoutSpecies) => statistics.bySpecies.find((row) => row.species === species)!.detailedLeaves;
  assert.equal(budget("mushroom-boulder") + budget("pebble-bed") + budget("stepping-disc"), 1_200);
  assert.equal(budget("bonsai"), 1_500);
  assert.equal(budget("rosette"), 400);
  assert.equal(budget("swept-tube"), 6);
  assert.equal(statistics.detailedLeaves, 3_106);
  // The whole point of publishing this number: a scene over the ceiling loses its
  // candidate BVH entirely, and the dry-scene contract then refuses it.
  assert.ok(statistics.headroom > 0, "the finished hero scene must fit under the primitive ceiling");
});

test("budget fidelity puts each row's whole detail allowance on the lattice", () => {
  const statistics = layoutStatistics(placements(), world(), { fidelity: "budget" });
  const authored = placements().filter((row) => row.authority === "authored")
    .reduce((total, row) => total + row.detailBudget, 0);
  assert.equal(statistics.placeholderLeaves, statistics.detailedLeaves - authored,
    "the load test must be the same size as the scene it stands in for");
  assert.ok(statistics.placeholderLeaves <= SVO_PRIMITIVE_CANDIDATE_MAXIMUM_LEAVES,
    "the load test must itself be publishable");
});
