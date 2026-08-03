import assert from "node:assert/strict";
import test from "node:test";

import { SVO_MATERIAL_FUNCTION_IDS, svoMaterialFunctionIdForEnvironmentProxy } from "../lib/svo-material-abi";
import {
  isSceneryPrimitiveNode,
  sceneryNodeGroup,
  validateSceneryGraph,
  walkSceneryNodes,
  type SceneryNode,
} from "../lib/scenery-graph";
import { SVO_PRIMITIVE_CANDIDATE_MAXIMUM_LEAVES } from "../lib/svo-primitive-candidates";
import {
  BONSAI_COURTYARD_STANDARD,
  BONSAI_POND_CANOPY,
  BONSAI_SHELF_MINIATURE,
  bonsaiFloretVoxelSpan,
  bonsaiNodes,
  planBonsai,
  type BonsaiForm,
  type BonsaiSpec,
} from "../lib/voxel-scenery/bonsai";
import {
  pondVesselHeightAt,
  pondVesselPlanCurve,
  pondVesselPlanDistance,
  type PondVesselSpec,
} from "../lib/voxel-scenery/pond-vessel";

/**
 * The bank the hero specimen stands on, spelled here rather than imported from
 * `lib/hero-garden-scene`.
 *
 * The generator is a species and must not know what it is standing beside, so
 * neither must its oracles: everything below reaches the vessel through the same
 * `(x, z) => height` the generator takes. What this vessel *is* for is the two
 * questions only a real bank can answer — that the toes land on ground that
 * falls away under them, and that the crown reaches over water without touching
 * the coping.
 */
const VESSEL: PondVesselSpec = {
  center_m: [0, 0],
  radius_m: [0.52, 0.38],
  groundHeight_m: 0.30,
  basinDepth_m: 0.155,
  rimHeight_m: 0.055,
  rimHalfWidth_m: 0.055,
  innerFace_m: 0.035,
  lobes: 7,
  wobble: 0.08,
  sectionHeightVariation: 0.16,
  sectionWidthVariation: 0.22,
  relief_m: 0.0025,
  seed: 0x9a7de11,
  terraces: [{ center_m: [0.62, 0.30], radius_m: [0.44, 0.34], height_m: 0.075, rotation_rad: -0.35, flat: 0.30 }],
};

const CURVE = pondVesselPlanCurve(VESSEL);
const ground = (x: number, z: number): number => pondVesselHeightAt(VESSEL, CURVE, x, z);

/** The hero placement: back-right terrace, leaning out over the pond's centre. */
const STAND_M = [0.60, 0.26] as const;

function specFor(form: BonsaiForm, overrides: Partial<BonsaiSpec> = {}): BonsaiSpec {
  return {
    ...form,
    key: "bonsai",
    at_m: STAND_M,
    groundHeightAt: ground,
    lean: [-STAND_M[0], -STAND_M[1]],
    seed: 0x8017a1,
    ...overrides,
  };
}

const FORMS: readonly (readonly [string, BonsaiForm])[] = [
  ["pond canopy", BONSAI_POND_CANOPY],
  ["courtyard standard", BONSAI_COURTYARD_STANDARD],
  ["shelf miniature", BONSAI_SHELF_MINIATURE],
];

/** Primitive leaves under the specimen's group, which is what the budget counts. */
function leaves(nodes: readonly SceneryNode[]): SceneryNode[] {
  return [...walkSceneryNodes(nodes)].map(({ node }) => node).filter(isSceneryPrimitiveNode);
}

test("one seed grows one tree, every time", () => {
  const spec = specFor(BONSAI_POND_CANOPY);
  assert.deepEqual(bonsaiNodes(spec), bonsaiNodes(spec));
  // Identity is not enough on its own: the plan caches nothing and shares no
  // mutable state, so the real risk is a hash reached through a floating index
  // rather than an integer. Serialize it, because that is what the publication
  // cache's static revision effectively hashes.
  assert.equal(JSON.stringify(bonsaiNodes(spec)), JSON.stringify(bonsaiNodes({ ...spec })));
});

test("a re-seed grows a sibling, not a different species", () => {
  const base = planBonsai(specFor(BONSAI_POND_CANOPY));
  for (const seed of [1, 7, 0x51ed, 0x7fff_ffff]) {
    const sibling = planBonsai(specFor(BONSAI_POND_CANOPY, { seed }));
    assert.notDeepEqual(sibling.nodes, base.nodes, `seed ${seed} produced the identical tree`);
    // Counts are structural and the seed only moves jitter inside slots, so the
    // budget cannot depend on which seed was picked by more than the culls do.
    // Culling is geometric, so the count really does swing with the seed. What
    // must hold is the budget, for every seed and not just the chosen one — a
    // form tuned on its own seed overran on a re-seed, which is how this
    // assertion came to exist.
    assert.ok(sibling.leafCount <= BONSAI_POND_CANOPY.maximumLeaves,
      `seed ${seed} published ${sibling.leafCount}, over the form's budget`);
    const drift = Math.abs(sibling.leafCount - base.leafCount) / base.leafCount;
    assert.ok(drift < 0.15, `seed ${seed} changed the leaf count by ${(100 * drift).toFixed(1)}%`);
    assert.ok(sibling.floretCoverage > 1.9, `seed ${seed} covers only ${sibling.floretCoverage.toFixed(2)}x`);
    for (const axis of ["x", "y", "z"] as const) {
      const here = sibling.bounds_m.max[axis] - sibling.bounds_m.min[axis];
      const there = base.bounds_m.max[axis] - base.bounds_m.min[axis];
      assert.ok(Math.abs(here - there) / there < 0.10, `seed ${seed} changed the ${axis} extent by ${(100 * Math.abs(here - there) / there).toFixed(1)}%`);
    }
  }
});

test("every form stays inside its own leaf budget, and the scene's", () => {
  for (const [name, form] of FORMS) {
    const plan = planBonsai(specFor(form));
    assert.ok(plan.leafCount <= form.maximumLeaves, `${name} published ${plan.leafCount} of ${form.maximumLeaves}`);
    assert.ok(plan.leafCount < SVO_PRIMITIVE_CANDIDATE_MAXIMUM_LEAVES,
      `${name} alone would fill the scene-wide candidate index`);
    assert.equal(plan.leafCount, leaves(plan.nodes).length, `${name} miscounted its own leaves`);
  }
  // The hero share, pinned. The scene budget is 4 096 across every object class
  // and this is the one that could eat it, so a re-tune that quietly doubles the
  // canopy has to come here and say so.
  assert.equal(BONSAI_POND_CANOPY.maximumLeaves, 1_500);
  assert.ok(planBonsai(specFor(BONSAI_POND_CANOPY)).leafCount > 1_300,
    "the pond form should be spending most of its share; a large drop means the culls broke");
});

test("the budget is enforced rather than merely documented", () => {
  assert.throws(
    () => planBonsai(specFor(BONSAI_POND_CANOPY, { maximumLeaves: 200 })),
    /over its 200-leaf budget/,
    "a form that overspends must fail loudly, not silently take another object's records",
  );
});

test("the florets cover the crown at the overlap that fuses them", () => {
  // One is tangent florets with gaps between; two is where they stop reading as
  // separate beads. Every shipped form has to clear two, and this is the oracle
  // that says so without a render — the coverage a form reaches for a given
  // record count is the whole argument about this object.
  for (const [name, form] of FORMS) {
    const { floretCoverage } = planBonsai(specFor(form));
    assert.ok(floretCoverage > 2, `${name} covers only ${floretCoverage.toFixed(2)}x; its florets will read as beads`);
    assert.ok(floretCoverage < 4, `${name} covers ${floretCoverage.toFixed(2)}x; records are being spent under the surface`);
  }
});

test("the crown is a wide flat plate that overhangs where it is told to", () => {
  const plan = planBonsai(specFor(BONSAI_POND_CANOPY));
  const { min, max } = plan.bounds_m;
  const span = { x: max.x - min.x, y: max.y - min.y, z: max.z - min.z };
  // Authored as 0.68 x 0.60 across a 0.17 m plate. The extent is measured off
  // the published primitives, so this is the assertion that the lobe layout,
  // the nodule cap and the floret seating together land on the authored
  // envelope rather than a third again of it, which two earlier versions did.
  assert.ok(Math.abs(span.x - 2 * BONSAI_POND_CANOPY.crownRadius_m[0]) < 0.05, `crown x span ${span.x.toFixed(3)}`);
  assert.ok(Math.abs(span.z - 2 * BONSAI_POND_CANOPY.crownRadius_m[1]) < 0.05, `crown z span ${span.z.toFixed(3)}`);
  const top = ground(...STAND_M) + BONSAI_POND_CANOPY.height_m;
  assert.ok(Math.abs(max.y - top) < 0.02, `crown top ${max.y.toFixed(3)} against an authored ${top.toFixed(3)}`);
  assert.ok(span.x / span.y > 1.2, "the specimen is taller than it is wide; the plate has become a ball");

  // The overhang, as the reference shows it: canopy standing over open water.
  const overWater = plan.lobes.filter(({ center_m }) => pondVesselPlanDistance(CURVE, center_m.x, center_m.z) < -VESSEL.rimHalfWidth_m);
  assert.ok(overWater.length >= 3, `only ${overWater.length} lobes reach past the coping's inner foot`);
});

test("the crown clears the coping it overhangs", () => {
  const plan = planBonsai(specFor(BONSAI_POND_CANOPY));
  const crest = VESSEL.groundHeight_m + VESSEL.rimHeight_m * (1 + VESSEL.sectionHeightVariation);
  // Nothing in the canopy may come down onto the rim it reaches over. Checked
  // against the crest's own upper bound rather than a sample, because the
  // section swells as it runs and a sample would pass on the thin quarter.
  for (const node of leaves(plan.nodes)) {
    if (!node.tags?.includes("floret") && !node.tags?.includes("crown")) continue;
    const position = node.place?.position;
    assert.ok(position, "every canopy part is placed");
    const reach = node.kind === "ellipsoid" ? Math.max(node.radius.x, node.radius.y, node.radius.z) : 0;
    const world = ground(...STAND_M) + position.y - reach;
    assert.ok(world > crest, `${node.id} reaches down to ${world.toFixed(3)}, under the crest at ${crest.toFixed(3)}`);
  }
});

test("the root fingers meet the bank they run out over", () => {
  const plan = planBonsai(specFor(BONSAI_POND_CANOPY));
  assert.equal(plan.rootFeet_m.length, BONSAI_POND_CANOPY.roots);
  let fall = 0;
  for (const foot of plan.rootFeet_m) {
    const under = ground(foot.x, foot.z);
    // Bedded, not floating and not buried: a toe sits a fraction of a stump
    // radius into the ground so no rim shows where it lands, and the *depth* is
    // what is pinned, because the ground under a toe is not the ground under the
    // stand — which is the entire reason this generator takes a height query.
    const depth = under - foot.y;
    assert.ok(depth > 0 && depth < 0.5 * BONSAI_POND_CANOPY.boleRadius_m,
      `a toe sits ${(1000 * depth).toFixed(1)} mm into the bank`);
    fall = Math.max(fall, Math.abs(under - ground(...STAND_M)));
  }
  // The terrace falls away toward the coping inside the fingers' own reach, so a
  // generator that had assumed a plane would have left toes hanging in air by
  // this much. Guards the height query against being quietly dropped.
  assert.ok(fall > 0.02, `the bank only falls ${(1000 * fall).toFixed(0)} mm under the root spread; the oracle has stopped testing anything`);
});

test("a specimen is one selectable object built from porcelain", () => {
  const nodes = bonsaiNodes(specFor(BONSAI_POND_CANOPY));
  assert.equal(nodes.length, 1, "a specimen publishes one group");
  const [root] = nodes;
  assert.equal(root.kind, "group");
  assert.equal(root.id, "bonsai");
  for (const { node, path } of walkSceneryNodes(nodes)) {
    if (node === root) continue;
    assert.ok(node.id.startsWith("bonsai/"), `${node.id} is not under its key`);
    // The group is inherited from the enclosing node, so the whole tree is one
    // click target and one gizmo.
    assert.equal(sceneryNodeGroup(node, path), "bonsai");
    // And the surface closure is chosen by a regex over that group plus the
    // tags. This specimen is one fired ceramic surface from the toes up; if a
    // rename ever lets it match the wood or foliage pattern first it will
    // silently restyle, which is exactly the kind of change no render review
    // catches.
    assert.equal(
      svoMaterialFunctionIdForEnvironmentProxy({ group: "bonsai", tags: node.tags ?? [] }),
      SVO_MATERIAL_FUNCTION_IDS.ceramic,
      `${node.id} does not resolve to the ceramic closure`,
    );
  }
});

test("two specimens can stand in one scene", () => {
  // The point of a key. Ids are unique scene-wide, so a generator that hard-codes
  // them can be instantiated exactly once — and a grove is the normal case.
  const graph = {
    palettes: { clay: { tint: [1, 0.985, 0.955] as const }, stone: { tint: [0.972, 0.984, 1] as const } },
    nodes: [
      { kind: "terrain-shell", id: "shell", materialModel: "porcelain" } as const,
      ...bonsaiNodes(specFor(BONSAI_POND_CANOPY, { key: "bank-bonsai" })),
      ...bonsaiNodes(specFor(BONSAI_SHELF_MINIATURE, { key: "shelf-bonsai", at_m: [-0.5, 0.3], seed: 12 })),
    ],
  };
  assert.deepEqual(validateSceneryGraph(graph), []);
});

test("the form refuses shapes it cannot grow", () => {
  assert.throws(() => planBonsai(specFor(BONSAI_POND_CANOPY, { trunks: 1 })), /at least two trunks/);
  assert.throws(() => planBonsai(specFor(BONSAI_POND_CANOPY, { roots: 2 })), /at least three root fingers/);
  assert.throws(() => planBonsai(specFor(BONSAI_POND_CANOPY, { lobes: 2 })), /at least three canopy lobes/);
  assert.throws(() => planBonsai(specFor(BONSAI_POND_CANOPY, { height_m: 0 })), /positive height/);
  assert.throws(() => planBonsai(specFor(BONSAI_POND_CANOPY, { floretFlatten: 1.4 })), /flatten must be in/);
  // The stump has to end before the trunks fork, and the two are authored as
  // independent fractions of the height, so nothing but a check couples them.
  assert.throws(() => planBonsai(specFor(BONSAI_POND_CANOPY, { stumpShare: 0.5, forkShare: 0.2 })), /fork share/);
});

test("halving the floret quadruples what the surface costs", () => {
  // The measurement the whole verdict rests on, as an oracle rather than an
  // assertion about pixels: coverage is records times floret area over envelope
  // area, so holding coverage while halving the floret is a four-fold bill. It
  // is why the reference's own ~7 mm floret is out of reach — at this crown's
  // envelope it would want tens of thousands of records against a scene-wide
  // ceiling of 4 096.
  const coarse = planBonsai(specFor(BONSAI_POND_CANOPY, { maximumLeaves: 40_000 }));
  const fine = planBonsai(specFor(BONSAI_POND_CANOPY, {
    floretRadius_m: 0.5 * BONSAI_POND_CANOPY.floretRadius_m,
    floretsPerLobe: 4 * BONSAI_POND_CANOPY.floretsPerLobe,
    maximumLeaves: 40_000,
  }));
  // Four times the florets at half the width does not even hold coverage: the
  // occlusion cull takes a larger share of small florets, because "buried deeper
  // than my own depth" is a lower bar for a shallower cap. So the true bill for
  // halving the grain is *more* than four-fold.
  assert.ok(fine.floretCoverage < coarse.floretCoverage,
    `four times the florets at half the width held coverage at ${fine.floretCoverage.toFixed(2)}; the cull was expected to bite harder`);
  assert.ok(fine.leafCount > 2.5 * coarse.leafCount, "the record count did not scale with the count of florets");
  assert.ok(fine.leafCount > 0.8 * SVO_PRIMITIVE_CANDIDATE_MAXIMUM_LEAVES,
    "one halving should already take most of the whole scene's candidate index");
});

test("the floret's span on a lattice is reported, and is not the constraint", () => {
  // Kept because it is the first thing anyone asks. At the hero lattice a floret
  // is a little over two cells across and the miniature's is under one — and
  // both render identically silhouetted, because a scenery primitive's primary
  // visibility is a BVH over exact records and never consults the per-voxel
  // material owner. The number bounds how coarsely the cone tracer lights the
  // crown, not what shape it is.
  assert.ok(Math.abs(bonsaiFloretVoxelSpan(BONSAI_POND_CANOPY, 0.025) - 2.34) < 0.02);
  assert.ok(bonsaiFloretVoxelSpan(BONSAI_SHELF_MINIATURE, 0.025) < 1);
  assert.throws(() => bonsaiFloretVoxelSpan(BONSAI_POND_CANOPY, 0), /positive cell size/);
});
