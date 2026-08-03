import assert from "node:assert/strict";
import test from "node:test";

import {
  HERO_GARDEN_CONTAINER,
  HERO_GARDEN_VESSEL,
  HERO_GARDEN_WATERLINE_M,
} from "../lib/hero-garden-scene";
import type { Quaternion, Vec3 } from "../lib/model";
import { quaternionRotate } from "../lib/rigid-body";
import {
  isSceneryPrimitiveNode,
  validateSceneryGraph,
  walkSceneryNodes,
  type SceneryGraph,
  type SceneryNode,
  type SceneryPrimitiveNode,
} from "../lib/scenery-graph";
import { pondVesselHeightAt, pondVesselPlanCurve, pondVesselPlanDistance } from "../lib/voxel-scenery/pond-vessel";
import {
  BOULDER_MUSHROOM_CAP,
  BOULDER_ROUNDED_COBBLE,
  cappedBoulderNodes,
  PEBBLE_FINE_SHINGLE,
  PEBBLE_GRADED_COBBLE,
  planRail,
  STEPPING_DISC,
  steppingStoneNodes,
  stoneSet,
  stoneSetBoulderNodes,
  stoneSetPebbleNodes,
  stoneSetSteppingNodes,
  pebbleBedNodes,
  type StoneSetSpec,
} from "../lib/voxel-scenery/stone-set";

/**
 * The stone set's CPU oracles.
 *
 * Two things are being pinned here and they want keeping apart. The **species** —
 * a capped boulder, a pebble bed, a stepping path — are reusable generators that
 * take an outline and a ground query and know nothing about ponds; their oracles
 * build them on shapes that are not this pond. The **arrangement** is the hero
 * garden's own set, and everything asserted about it is asserted against the
 * *vessel*, never against a coordinate copied out of the generator. A test that
 * pinned the boulders to the metres they happen to land on would pass for as long
 * as nobody touched the pond and would say nothing about whether the stones still
 * follow it — which is the one property this arrangement exists to have.
 */

const SPEC: StoneSetSpec = {
  vessel: HERO_GARDEN_VESSEL,
  waterline_m: HERO_GARDEN_WATERLINE_M,
  seed: 0x5701_e5,
};

const CURVE = pondVesselPlanCurve(SPEC.vessel);
const groundAt = (x: number, z: number): number => pondVesselHeightAt(SPEC.vessel, CURVE, x, z);

/** A published primitive, resolved to the world the expander would put it in. */
interface ResolvedStone {
  readonly id: string;
  readonly group: string;
  readonly node: SceneryPrimitiveNode;
  readonly center_m: Vec3;
  /** Half extents of the world axis-aligned box the primitive fits inside. */
  readonly halfExtent_m: Vec3;
}

const V = (x: number, y: number, z: number): Vec3 => ({ x, y, z });

interface Frame {
  readonly origin_m: Vec3;
  readonly quaternion: Quaternion;
  readonly group: string;
}

const IDENTITY: Quaternion = { w: 1, x: 0, y: 0, z: 0 };

/**
 * Resolve the nodes exactly as `expandSceneryGraph` would, on this terrain.
 *
 * Deliberately a re-implementation of the placement rules rather than a call into
 * the expander: the expander needs an environment context, a builder and a whole
 * scene, and what these oracles are about is the geometry, not the plumbing that
 * carries it. The two rules that matter are here — a terrain anchor replaces the
 * vertical datum with the ground under the node's own place, and a group's
 * orientation rotates its children's offsets.
 */
function resolve(nodes: readonly SceneryNode[], ground: (x: number, z: number) => number = groundAt): ResolvedStone[] {
  const out: ResolvedStone[] = [];
  const visit = (list: readonly SceneryNode[], parent: Frame): void => {
    for (const node of list) {
      const place = node.place ?? {};
      const local = place.position ?? V(0, 0, 0);
      const rotated = quaternionRotate(parent.quaternion, local);
      let origin_m = V(
        parent.origin_m.x + rotated.x,
        parent.origin_m.y + rotated.y,
        parent.origin_m.z + rotated.z,
      );
      if (place.anchor === "terrain") {
        const [gx, gz] = place.ground ?? [local.x, local.z];
        origin_m = V(origin_m.x, ground(gx, gz) + local.y, origin_m.z);
      }
      const quaternion = place.orientation ? multiply(parent.quaternion, place.orientation) : parent.quaternion;
      const group = node.group ?? (node.kind === "group" ? node.id : parent.group);
      if (node.kind === "group") { visit(node.children, { origin_m, quaternion, group }); continue; }
      if (!isSceneryPrimitiveNode(node)) continue;
      out.push({ id: node.id, group, node, center_m: origin_m, halfExtent_m: halfExtentOf(node, quaternion) });
    }
  };
  visit(nodes, { origin_m: V(0, 0, 0), quaternion: IDENTITY, group: "scenery" });
  return out;
}

/** The parts of a published object, by node kind rather than by assertion. */
function childrenOf(node: SceneryNode): readonly SceneryNode[] {
  assert.equal(node.kind, "group", `${node.id} should have been published as a group`);
  return node.kind === "group" ? node.children : [];
}

function partOf(node: SceneryNode, suffix: string): SceneryNode {
  const part = childrenOf(node).find((child) => child.id.endsWith(suffix));
  assert.ok(part, `${node.id} is missing its ${suffix}`);
  return part;
}

function multiply(a: Quaternion, b: Quaternion): Quaternion {
  return {
    w: a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
    x: a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
    y: a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
    z: a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
  };
}

/** Conservative world half extents: the local box, rotated by absolute value. */
function halfExtentOf(node: SceneryPrimitiveNode, quaternion: Quaternion): Vec3 {
  const local = (() => {
    switch (node.kind) {
      case "ellipsoid": return node.radius;
      case "cylinder": return V(node.radius, node.halfHeight, node.radius);
      case "cone": return V(Math.max(node.baseRadius, node.topRadius), node.halfHeight, Math.max(node.baseRadius, node.topRadius));
      case "box": return node.halfSize;
      case "torus": return V(node.majorRadius + node.minorRadius, node.minorRadius, node.majorRadius + node.minorRadius);
      case "capsule": return V(node.radius, node.radius, node.radius);
    }
  })();
  const axis = (v: Vec3) => quaternionRotate(quaternion, v);
  const x = axis(V(local.x, 0, 0)), y = axis(V(0, local.y, 0)), z = axis(V(0, 0, local.z));
  return V(
    Math.abs(x.x) + Math.abs(y.x) + Math.abs(z.x),
    Math.abs(x.y) + Math.abs(y.y) + Math.abs(z.y),
    Math.abs(x.z) + Math.abs(y.z) + Math.abs(z.z),
  );
}

/**
 * The published primitives, folded back into the objects they belong to.
 *
 * A boulder is four records and one stone; a pebble is one of each. A pebble is
 * therefore keyed by its own id and everything else by the group it declares.
 */
function objects(nodes: readonly SceneryNode[]): Map<string, ResolvedStone[]> {
  const out = new Map<string, ResolvedStone[]>();
  for (const stone of resolve(nodes)) {
    const key = stone.id.includes("/pebble-") ? stone.id : stone.group;
    const list = out.get(key);
    if (list) list.push(stone); else out.set(key, [stone]);
  }
  return out;
}

const leavesOf = (nodes: readonly SceneryNode[]): SceneryPrimitiveNode[] => {
  const out: SceneryPrimitiveNode[] = [];
  for (const { node } of walkSceneryNodes(nodes)) if (isSceneryPrimitiveNode(node)) out.push(node);
  return out;
};

// ---------------------------------------------------------------------------
// The species, on geometry that is not this pond
// ---------------------------------------------------------------------------

/** A rounded rectangle of an outline, so nothing below can be a pond in disguise. */
function trackOutline(halfX: number, halfZ: number, samples = 96): readonly (readonly [number, number])[] {
  const points: (readonly [number, number])[] = [];
  for (let index = 0; index < samples; index += 1) {
    const angle = 2 * Math.PI * index / samples;
    const c = Math.cos(angle), s = Math.sin(angle);
    points.push([halfX * Math.sign(c) * Math.abs(c) ** 0.6, halfZ * Math.sign(s) * Math.abs(s) ** 0.6]);
  }
  return points;
}

test("a species takes geometry, not a scene", () => {
  // The property the whole file was reshaped for. A bed given an outline, a flat
  // ground query and a key must grow on it — no pond, no vessel, no waterline.
  const rail = planRail(trackOutline(0.9, 0.5));
  const bed = pebbleBedNodes({
    ...PEBBLE_GRADED_COBBLE,
    key: "courtyard/verge",
    seed: 11,
    rail,
    groundHeightAt: () => 0.12,
    start_m: 0.02,
    direction: 1,
    widthAt: () => 0.10,
  });
  const leaves = leavesOf(bed);
  assert.ok(leaves.length > 60, `a 3 m verge grew only ${leaves.length} stones`);
  // Ids are unique scene-wide, so a generator that does not prefix them can be
  // instantiated exactly once. Two beds side by side is the test of that.
  const second = pebbleBedNodes({
    ...PEBBLE_GRADED_COBBLE,
    key: "courtyard/kerb",
    seed: 11,
    rail,
    groundHeightAt: () => 0.12,
    start_m: 0.14,
    direction: 1,
    widthAt: () => 0.06,
  });
  const ids = new Set([...leavesOf(bed), ...leavesOf(second)].map((leaf) => leaf.id));
  assert.equal(ids.size, leaves.length + leavesOf(second).length, "two beds collided on an id");
  for (const leaf of leaves) assert.ok(leaf.id.startsWith("courtyard/verge/"), `${leaf.id} does not carry its key`);

  // ...and the ground it was given is the ground it sits on, whatever that is.
  const sloped = pebbleBedNodes({
    ...PEBBLE_GRADED_COBBLE,
    key: "courtyard/verge",
    seed: 11,
    rail,
    groundHeightAt: (x) => 0.12 + 0.25 * x,
    start_m: 0.02,
    direction: 1,
    widthAt: () => 0.10,
  });
  const tilted = resolve(sloped, (x) => 0.12 + 0.25 * x);
  assert.ok(tilted.some((stone) => Math.abs(stone.center_m.y - 0.12) > 0.05),
    "the bed ignored the ground query it was handed");
});

test("a form carries the silhouette, and a spec only carries the placement", () => {
  const cap = cappedBoulderNodes({ ...BOULDER_MUSHROOM_CAP, key: "a", at_m: [0, 0], seed: 3 });
  const cobble = cappedBoulderNodes({ ...BOULDER_ROUNDED_COBBLE, key: "b", at_m: [0, 0], seed: 3 });
  const overhang = (nodes: readonly SceneryNode[]): number => {
    const capPart = partOf(nodes[0], "/cap"), stem = partOf(nodes[0], "/stem");
    assert.ok(capPart.kind === "ellipsoid" && stem.kind === "cone");
    return capPart.radius.x / stem.topRadius;
  };
  assert.ok(overhang(cap) > 1.7, `the capped form overhangs only ${overhang(cap).toFixed(2)}x`);
  assert.ok(overhang(cobble) < 1.3, `the cobble form overhangs ${overhang(cobble).toFixed(2)}x, which is a cap`);
  // Two specimens of one form differ only by the seed and where they stand.
  const here = cappedBoulderNodes({ ...BOULDER_MUSHROOM_CAP, key: "a", at_m: [0.4, -0.2], seed: 3 });
  assert.equal(JSON.stringify(leavesOf(cap).map((leaf) => leaf.kind)), JSON.stringify(leavesOf(here).map((leaf) => leaf.kind)));

  // The same for the beds: a named form has to be a different *stone*, not a
  // renamed copy of its neighbour. The shingle is the finer of the two by a
  // clear margin at both ends of its grade, and it fines further as it runs down
  // into the water where the cobble bed holds its grain.
  assert.ok(PEBBLE_FINE_SHINGLE.radiusRich_m < 0.9 * PEBBLE_GRADED_COBBLE.radiusRich_m,
    "the shingle form is not finer than the cobble form");
  assert.ok(PEBBLE_FINE_SHINGLE.crossGrade < 1 && PEBBLE_GRADED_COBBLE.crossGrade > 1,
    "the two bed forms grade the same way across the band");
});

test("a stepping path lays plates on the bed it is given", () => {
  // A straight run over a bed that falls away, which is the case the hero shelf
  // is a curved instance of. Every plate has to hold its freeboard while the bed
  // under it drops, and its footing has to reach the bed.
  const level = 0.30;
  const bed = (x: number): number => 0.30 - 0.30 * Math.max(0, x);
  const path = steppingStoneNodes({
    ...STEPPING_DISC,
    key: "ford",
    seed: 5,
    path: [[-0.1, 0], [0.5, 0]],
    count: 5,
    radiusStart_m: 0.05,
    radiusEnd_m: 0.04,
    stride_m: 0.02,
    groundHeightAt: (x) => bed(x),
    level_m: level,
  });
  assert.equal(path.length, 5);
  // Resolved, not read off the node. A stone is a group with a yaw on it and its
  // parts are offsets inside that group; reading the offsets as though they were
  // world metres is precisely the mistake that put five plates on the far side of
  // the hero basin while every number about them looked right.
  const parts = resolve(path, () => 0);
  for (const stone of path) {
    const tread = partOf(stone, "/tread"), footing = partOf(stone, "/footing");
    assert.ok(tread.kind === "cylinder" && footing.kind === "cone");
    const at = parts.find((part) => part.id === `${stone.id}/tread`)!;
    const foot = parts.find((part) => part.id === `${stone.id}/footing`)!;
    const top = at.center_m.y + tread.halfHeight;
    assert.ok(top >= level + 0.008, `${stone.id} is drowned`);
    assert.ok(foot.center_m.y - footing.halfHeight <= bed(at.center_m.x) + 1e-9, `${stone.id} hangs over the bed`);
    // A plate, not a drum: nothing wider than the tread it carries.
    assert.ok(footing.baseRadius <= tread.radius, `${stone.id} wears a collar wider than its plate`);
    // ...and the yaw is about the stone's own axis, not about the world origin.
    assert.ok(Math.hypot(at.center_m.x - foot.center_m.x, at.center_m.z - foot.center_m.z) < 1e-9,
      `${stone.id} has its tread and its footing in different places`);
  }
});

// ---------------------------------------------------------------------------
// The hero arrangement
// ---------------------------------------------------------------------------

test("the set is one seed in and the same stones out", () => {
  assert.equal(JSON.stringify(stoneSet(SPEC)), JSON.stringify(stoneSet(SPEC)),
    "the same spec must build the same nodes: the publication cache is keyed on this geometry");
  // ...and a different seed must actually move stone, or "another garden" is a
  // rename rather than a re-generation.
  const other = stoneSet({ ...SPEC, seed: SPEC.seed + 1 });
  assert.notEqual(JSON.stringify(other), JSON.stringify(stoneSet(SPEC)));
  assert.ok(Math.abs(leavesOf(other).length - leavesOf(stoneSet(SPEC)).length) < 60,
    "a re-seed must be a sibling, not a different quantity of set");
});

test("the set fits its share of the scene's primitive budget, and its bricks", () => {
  const leaves = leavesOf(stoneSet(SPEC));
  // 4 096 leaves is the whole scene's candidate ceiling and this class was given
  // 1 200 of it. The lower bound is the reference's own count: it shows something
  // like a hundred and fifty stones in four clusters, and a set that published
  // half that would have lost one of them.
  assert.ok(leaves.length <= 1_200, `the stone set publishes ${leaves.length} leaves, over its 1 200 budget`);
  assert.ok(leaves.length >= 110, `the stone set publishes only ${leaves.length} leaves`);
  assert.equal(leavesOf(stoneSetBoulderNodes(SPEC)).length, 16, "four boulders of four parts each");
  assert.equal(leavesOf(stoneSetSteppingNodes(SPEC)).length, 15, "five plates of three parts each");
  const pebbles = leavesOf(stoneSetPebbleNodes(SPEC)).length;
  assert.ok(pebbles >= 80 && pebbles <= 320, `${pebbles} pebbles, outside the 80-320 the reference reads as`);

  // The binding limit is not the scene total: the lighting hierarchy takes
  // `OCTREE_LIVE_SCENE_CANDIDATES_PER_BRICK = 64` primitives per 200 mm brick and
  // drops the surplus *silently*, so the densest part of a bed is the part that
  // stops being lit. A continuous double band round the coping broke this in
  // three bricks; clusters and a coarser grain are what fixed it. See
  // `docs/SVO_FINE_VOXEL_CAPACITY.md`.
  const perBrick = new Map<string, number>();
  for (const stone of resolve(stoneSet(SPEC))) {
    const key = [stone.center_m.x, stone.center_m.y, stone.center_m.z]
      .map((value) => Math.floor(value / 0.2)).join(",");
    perBrick.set(key, (perBrick.get(key) ?? 0) + 1);
  }
  const worst = Math.max(...perBrick.values());
  assert.ok(worst <= 64, `one 200 mm brick holds ${worst} of this set's primitives, over the 64 the hierarchy binds`);
});

test("every node is legal scenery, in the palettes the hero scene declares", () => {
  const graph: SceneryGraph = {
    palettes: { clay: { tint: [1, 0.985, 0.955] }, stone: { tint: [0.972, 0.984, 1] } },
    nodes: [{ kind: "terrain-shell", id: "shell", materialModel: "porcelain" }, ...stoneSet(SPEC)],
  };
  assert.deepEqual(validateSceneryGraph(graph), []);
  for (const leaf of leavesOf(graph.nodes)) {
    assert.ok("palette" in leaf.material && leaf.material.palette === "stone",
      `${leaf.id} must be stone by palette; the set is monochrome by construction`);
    // The band the whole set is authored in. Outside it a stone stops reading as
    // the same fired white as the vessel and starts reading as a different
    // material, which is the one thing a monochrome set cannot survive.
    assert.ok(leaf.material.value >= 0.62 && leaf.material.value <= 0.92,
      `${leaf.id} has value ${leaf.material.value}, outside the set's [0.62, 0.92]`);
  }
});

test("the material closure resolves to stone, not to organic", () => {
  // `svoMaterialFunctionIdForEnvironmentProxy` picks a procedural surface by regex
  // over the group and tags, and it tests `mushroom` on the same line as `hose`
  // and `cloth`. A boulder described as a mushroom would be shaded as one. Every
  // node here therefore declares a group carrying the word the stone rule
  // matches — declares it rather than inheriting it, so a caller's own key cannot
  // decide what the surface is.
  for (const nodes of [stoneSet(SPEC), cappedBoulderNodes({ ...BOULDER_MUSHROOM_CAP, key: "q", at_m: [0, 0], seed: 1 })]) {
    for (const { node, path } of walkSceneryNodes(nodes)) {
      const group = node.group ?? path[path.length - 1] ?? node.id;
      const semantic = `${group} ${(node.tags ?? []).join(" ")}`;
      assert.ok(/stone|pebble/.test(semantic), `${node.id} would not reach the stone closure: "${semantic}"`);
      assert.ok(!/mushroom|organic|hose|rope|cable/.test(semantic),
        `${node.id} names a surface the organic rule claims: "${semantic}"`);
    }
  }
});

test("every stone stands inside the container, under its ceiling", () => {
  const half = { x: 0.5 * HERO_GARDEN_CONTAINER.width_m, z: 0.5 * HERO_GARDEN_CONTAINER.depth_m };
  for (const stone of resolve(stoneSet(SPEC))) {
    assert.ok(Math.abs(stone.center_m.x) + stone.halfExtent_m.x <= half.x,
      `${stone.id} reaches x = ${(stone.center_m.x + stone.halfExtent_m.x).toFixed(3)}, outside the container`);
    assert.ok(Math.abs(stone.center_m.z) + stone.halfExtent_m.z <= half.z,
      `${stone.id} reaches z = ${(stone.center_m.z + stone.halfExtent_m.z).toFixed(3)}, outside the container`);
    assert.ok(stone.center_m.y + stone.halfExtent_m.y <= HERO_GARDEN_CONTAINER.height_m,
      `${stone.id} reaches y = ${(stone.center_m.y + stone.halfExtent_m.y).toFixed(3)}, through the ceiling`);
    assert.ok(stone.center_m.y - stone.halfExtent_m.y >= 0, `${stone.id} reaches below the container floor`);
  }
});

test("every bedded stone is bedded: into the ground, and not swallowed by it", () => {
  // The two ways a generated set fails visibly. An object whose lowest point is
  // above the ground floats, and shows daylight under it from any camera near the
  // horizon; one whose highest point is below the ground is invisible and costs a
  // primitive record for nothing.
  //
  // The property belongs to the *object*, not to its parts — a boulder's cap is
  // supposed to be six centimetres clear of the bank, which is what its stem is
  // for. So the parts are folded back into the thing they were published as, and
  // the ground is taken under that thing's own footing.
  for (const [id, parts] of objects([...stoneSetBoulderNodes(SPEC), ...stoneSetPebbleNodes(SPEC)])) {
    const bottom = Math.min(...parts.map((part) => part.center_m.y - part.halfExtent_m.y));
    const top = Math.max(...parts.map((part) => part.center_m.y + part.halfExtent_m.y));
    const lowest = parts.reduce((a, b) =>
      a.center_m.y - a.halfExtent_m.y <= b.center_m.y - b.halfExtent_m.y ? a : b);
    const ground = groundAt(lowest.center_m.x, lowest.center_m.z);
    assert.ok(top > ground + 0.0015, `${id} is buried: its top is ${((top - ground) * 1e3).toFixed(1)} mm over the ground`);
    // A mounded bed rests its upper course on the stones below rather than on the
    // plaster, so the floor here is one stone's depth rather than zero.
    assert.ok(bottom < ground + 0.7 * lowest.halfExtent_m.y,
      `${id} floats: its base is ${((bottom - ground) * 1e3).toFixed(1)} mm over the ground`);
  }
});

test("no bedded stone stands in the water", () => {
  // Scenery is invisible to the solver, so a boulder or a pebble below the
  // still-water level is one the water pours straight through. This is the
  // assertion that keeps the shoreline honest, and it is deliberately made on the
  // stone rather than on the ground under it: a pebble bedded on a bank a
  // millimetre clear of the water still has most of itself under the water.
  for (const stone of resolve([...stoneSetBoulderNodes(SPEC), ...stoneSetPebbleNodes(SPEC)])) {
    assert.ok(stone.center_m.y > HERO_GARDEN_WATERLINE_M,
      `${stone.id} sits ${((HERO_GARDEN_WATERLINE_M - stone.center_m.y) * 1e3).toFixed(1)} mm under the waterline`);
    assert.ok(groundAt(stone.center_m.x, stone.center_m.z) > HERO_GARDEN_WATERLINE_M,
      `${stone.id} stands on ground the pond covers`);
  }
});

test("the boulders and beds follow the coping rather than sitting on it", () => {
  const rimHalfWidth = SPEC.vessel.rimHalfWidth_m;
  const boulders = resolve(stoneSetBoulderNodes(SPEC));
  assert.equal(stoneSetBoulderNodes(SPEC).length, 4, "the reference's set is four");
  for (const stone of boulders) {
    const distance = pondVesselPlanDistance(CURVE, stone.center_m.x, stone.center_m.z);
    // Outside the widest the coping's section ever swells to. A boulder standing
    // on the rim would be a boulder the water has to be told about.
    assert.ok(distance > 1.25 * rimHalfWidth,
      `${stone.id} stands ${(distance * 1e3).toFixed(0)} mm out, on the coping`);
    // ...and near enough that the group reads as belonging to this pond.
    assert.ok(distance < 0.42, `${stone.id} stands ${(distance * 1e3).toFixed(0)} mm out, adrift of the pond`);
  }
  // The graded set: strictly descending, and a range wide enough to read as one.
  const caps = stoneSetBoulderNodes(SPEC)
    .filter((node) => node.id.endsWith("/boulder-0") || node.id.includes("/boulder-"))
    .map((node) => {
      const cap = partOf(node, "/cap");
      return cap.kind === "ellipsoid" ? cap.radius.x : 0;
    });
  for (let index = 1; index < caps.length; index += 1) {
    assert.ok(caps[index] < caps[index - 1], `boulder ${index} is not smaller than the one before it`);
  }
  assert.ok(caps[0] / caps[caps.length - 1] > 1.8, `the set grades only ${(caps[0] / caps[caps.length - 1]).toFixed(2)}x`);
  // The reference's largest cap is about a sixth of the pond's width and sits
  // behind the near coping. Twice that and the group stops being a family on the
  // bank and becomes four foreground objects, which is what the second pass did.
  assert.ok(caps[0] * 2 < SPEC.vessel.radius_m[0] * 2 / 5,
    `the largest cap is ${(caps[0] * 2e3).toFixed(0)} mm across, over a fifth of the pond`);

  // Every pebble hugs one foot of the coping or the other: nothing on the crest,
  // nothing stranded out on open plaster.
  for (const stone of resolve(stoneSetPebbleNodes(SPEC))) {
    const distance = pondVesselPlanDistance(CURVE, stone.center_m.x, stone.center_m.z);
    assert.ok(Math.abs(distance) > 0.55 * rimHalfWidth, `${stone.id} is bedded on the coping's crown`);
    assert.ok(distance < 0.44, `${stone.id} is ${(distance * 1e3).toFixed(0)} mm out on bare ground`);
  }
});

test("the beds are clusters, not a band round the whole coping", () => {
  // The correction that mattered most, and the one a count alone would not catch:
  // the reference banks its stones in three or four places and leaves most of the
  // rim bare. Measured on the rail's own turn — a set that ringed the pond would
  // put stones in every bucket, and this one has to leave a run of them empty.
  const occupied = new Set<number>();
  for (const stone of resolve(stoneSetPebbleNodes(SPEC))) {
    occupied.add(Math.floor(40 * ((Math.atan2(stone.center_m.z, stone.center_m.x) / (2 * Math.PI) + 1) % 1)));
  }
  assert.ok(occupied.size <= 26, `pebbles reach ${occupied.size} of 40 bearings; that is a band, not clusters`);
  assert.ok(occupied.size >= 10, `pebbles reach only ${occupied.size} of 40 bearings; a cluster has gone missing`);
  // ...and the empty run has to be *contiguous*, or what the frame shows is a bed
  // with holes in it rather than a bare stretch of coping.
  let longestGap = 0, gap = 0;
  for (let bucket = 0; bucket < 80; bucket += 1) {
    if (occupied.has(bucket % 40)) gap = 0; else longestGap = Math.max(longestGap, gap += 1);
  }
  assert.ok(longestGap >= 5, `the longest bare run of coping is ${longestGap} of 40 bearings`);
});

test("a boulder is a cap on a stem, and the cap overhangs it", () => {
  // The three proportions that make the silhouette. Read off the reference: the
  // cap is a little under twice the stem's width, flattened to about a third of
  // its own, and its rim stands clear of the stem all the way round — which is
  // what puts the dark undercut line under every one of these stones.
  for (const boulder of stoneSetBoulderNodes(SPEC)) {
    const cap = partOf(boulder, "/cap");
    const stem = partOf(boulder, "/stem");
    assert.ok(cap.kind === "ellipsoid" && stem.kind === "cone", `${boulder.id} needs an ellipsoid cap on a cone stem`);
    assert.ok(cap.radius.x / stem.topRadius > 1.7 && cap.radius.x / stem.topRadius < 2.3,
      `${boulder.id} cap-to-stem is ${(cap.radius.x / stem.topRadius).toFixed(2)}, outside the reference's ~1.9`);
    assert.ok(cap.radius.y / cap.radius.x > 0.30 && cap.radius.y / cap.radius.x < 0.45,
      `${boulder.id} cap flatten is ${(cap.radius.y / cap.radius.x).toFixed(2)}, outside the reference's ~0.37`);
    assert.ok(stem.baseRadius > stem.topRadius, `${boulder.id} stem must taper upward`);
    // The cap has to sit down over the stem, or the two meet in a seam the
    // voxelizer decides the ownership of rather than fusing into one stone.
    const capBottom = (cap.place?.position?.y ?? 0) - cap.radius.y;
    const stemTop = (stem.place?.position?.y ?? 0) + stem.halfHeight;
    assert.ok(capBottom < stemTop, `${boulder.id} cap balances on its stem instead of sitting over it`);
  }
});

test("the plates wade over the shore, not over the basin floor", () => {
  // The intent that changed when the pond grew a beach, and the reason this
  // assertion is no longer "the ground under each stone is below the waterline".
  // The reference's stones are *plates*: thin discs whose tops sit just proud of
  // the surface and which show nothing underneath. Laid on the basin floor they
  // become drums on 110 mm plinths — legal, invisible under water, and fully on
  // show on a scene that opens dry, which this one does.
  const steps = stoneSetSteppingNodes(SPEC);
  assert.equal(steps.length, 5, "the reference lays five");
  const placed = resolve(steps);
  const treads: { x: number; z: number; radius: number }[] = [];
  let footing_m = 0, exposed_m = 0;
  for (const step of steps) {
    const footing = partOf(step, "/footing");
    const tread = partOf(step, "/tread");
    assert.ok(footing.kind === "cone" && tread.kind === "cylinder", `${step.id} is a tread on a tapered footing`);
    const treadAt = placed.find((part) => part.id === `${step.id}/tread`)!.center_m;
    const footingAt = placed.find((part) => part.id === `${step.id}/footing`)!.center_m;
    const ground = groundAt(treadAt.x, treadAt.z);
    assert.ok(ground < SPEC.vessel.groundHeight_m - 0.03, `${step.id} stands on the bank rather than in the pond`);
    assert.ok(footingAt.y - footing.halfHeight < ground, `${step.id} hangs in the water instead of standing on the bed`);
    assert.ok(Math.hypot(treadAt.x - footingAt.x, treadAt.z - footingAt.z) < 1e-9,
      `${step.id} was spun about the world origin rather than about its own axis`);
    const top = treadAt.y + tread.halfHeight;
    assert.ok(top > HERO_GARDEN_WATERLINE_M + 0.008,
      `${step.id} tread is drowned: ${((top - HERO_GARDEN_WATERLINE_M) * 1e3).toFixed(1)} mm of freeboard`);
    assert.ok(top < HERO_GARDEN_WATERLINE_M + 0.05,
      `${step.id} stands ${((top - HERO_GARDEN_WATERLINE_M) * 1e3).toFixed(0)} mm proud; a stepping stone is not a plinth`);
    // A plate is thin: the reference's are about a fifth of their own width.
    assert.ok(2 * tread.halfHeight < 0.30 * tread.radius * 2,
      `${step.id} is ${(2e3 * tread.halfHeight).toFixed(0)} mm thick on a ${(2e3 * tread.radius).toFixed(0)} mm plate`);
    // Inside the basin by a whole tread, so no plate grows out of the coping.
    const distance = pondVesselPlanDistance(CURVE, treadAt.x, treadAt.z);
    assert.ok(distance < -(1.22 * SPEC.vessel.rimHalfWidth_m + tread.radius),
      `${step.id} overlaps the coping it stepped off`);
    footing_m += 2 * footing.halfHeight;
    exposed_m += Math.max(0, footingAt.y + footing.halfHeight - ground);
    treads.push({ x: treadAt.x, z: treadAt.z, radius: tread.radius });
  }
  // ...and taken over the path, most of the footing is *buried by the shelf*
  // rather than hidden by water. That is the whole difference between a plate on
  // a shore and a drum in a pond, and it is a property of the arrangement rather
  // than of any one stone.
  assert.ok(exposed_m < 0.5 * footing_m,
    `${(1e2 * exposed_m / footing_m).toFixed(0)}% of the path's footing stands out of the bed`);

  // A path, not a heap: consecutive plates clear one another, and they shrink as
  // they wade in.
  for (let index = 1; index < treads.length; index += 1) {
    const a = treads[index - 1], b = treads[index];
    assert.ok(b.radius < a.radius, `plate ${index} is not smaller than the one before it`);
    const gap = Math.hypot(a.x - b.x, a.z - b.z) - a.radius - b.radius;
    assert.ok(gap > 0.012, `plates ${index - 1} and ${index} are ${(gap * 1e3).toFixed(0)} mm apart`);
    assert.ok(gap < 0.14, `plates ${index - 1} and ${index} are ${(gap * 1e3).toFixed(0)} mm apart, not a stride`);
  }
});

test("the beds are packed, not scattered", () => {
  // The distinction the plan got wrong and the reference settles: a Poisson
  // scatter leaves ground between its stones and these beds have none. Measured
  // as the share of pebbles with a neighbour inside a stone's width — a scatter
  // tuned to the same count comes out under a half, and a packing near one. The
  // floor is under one because the arrangement now ends deliberately in singles:
  // the reference's lower-right scatter is a dozen stones with plaster between
  // them, and those have no neighbour by construction.
  const pebbles = resolve(stoneSetPebbleNodes(SPEC));
  let touching = 0;
  for (const [index, stone] of pebbles.entries()) {
    const reach = stone.halfExtent_m.x;
    const near = pebbles.some((other, otherIndex) => otherIndex !== index
      && Math.hypot(stone.center_m.x - other.center_m.x, stone.center_m.z - other.center_m.z)
        < reach + other.halfExtent_m.x);
    if (near) touching += 1;
  }
  const share = touching / pebbles.length;
  assert.ok(share > 0.8, `only ${(share * 100).toFixed(0)}% of pebbles touch a neighbour; that is a scatter`);
});

test("the set moves when the pond does", () => {
  // The property the rail exists for. A pond a centimetre wider must carry its
  // stones with it; a set that stayed put would be a set authored in world metres
  // wearing a generator's clothes.
  const wider = { ...SPEC, vessel: { ...SPEC.vessel, radius_m: [0.56, 0.42] as const } };
  const before = resolve(stoneSetBoulderNodes(SPEC));
  const after = resolve(stoneSetBoulderNodes(wider));
  for (const [index, stone] of before.entries()) {
    const moved = Math.hypot(stone.center_m.x - after[index].center_m.x, stone.center_m.z - after[index].center_m.z);
    assert.ok(moved > 0.02, `${stone.id} did not follow the pond outward: ${(moved * 1e3).toFixed(1)} mm`);
  }
  assert.ok(leavesOf(stoneSet(wider)).length > 0.7 * leavesOf(stoneSet(SPEC)).length,
    "a bigger pond must not collapse the beds that follow it");
  // The path is solved against the shore rather than authored across it, so a
  // pond with no beach at all still lays five plates and none of them is stranded.
  const walled = { ...SPEC, vessel: { ...SPEC.vessel, beach: undefined } };
  assert.equal(stoneSetSteppingNodes(walled).length, 5, "a walled pond lost its path");
});
