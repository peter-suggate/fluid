import assert from "node:assert/strict";
import test from "node:test";

import { createHeroGardenHoseScene, HERO_GARDEN_VESSEL } from "../lib/hero-garden-scene";
import { growSceneryGenerator } from "../lib/scenery-generators";
import { SVO_MATERIAL_FUNCTION_IDS, svoMaterialFunctionIdForEnvironmentProxy } from "../lib/svo-material-abi";
import {
  isSceneryPrimitiveNode,
  sceneryNodeGroup,
  validateSceneryGraph,
  walkSceneryNodes,
  type SceneryNode,
} from "../lib/scenery-graph";
import { intersectSvoPrimitive, validateSvoClusterPacking } from "../lib/svo-primitive-abi";
import { SVO_PRIMITIVE_CANDIDATE_MAXIMUM_LEAVES } from "../lib/svo-primitive-candidates";
import {
  BONSAI_COURTYARD_STANDARD,
  BONSAI_POND_CANOPY,
  BONSAI_SHELF_MINIATURE,
  BONSAI_DEFAULT_LEAF_SIZE_M,
  bonsaiCanopyLadder,
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

/**
 * The pond form as it was before the crown became an aggregate.
 *
 * The explicit construction is still shipped and still has to hold, and the
 * argument the aggregate wins on is a comparison against it. So the assertions
 * about records-per-unit-of-surface run here, against a form that still spells
 * its florets, rather than being deleted along with the canopy that stopped
 * needing them.
 */
const POND_FLORETS: BonsaiForm = {
  ...BONSAI_POND_CANOPY,
  canopy: "florets",
  // The crown the explicit construction was tuned for, which is not the one the
  // aggregate ships. A cast floret field covers a plate of *many* heads; the
  // aggregate's crown is four nearly crown-sized plates, and a floret field
  // spread over four heads is not a comparison, it is a different object. These
  // are the head count and overlap the explicit build shipped with.
  lobes: 18,
  lobeFill: 1.34,
  lobeSwell: 0.58,
  // Its own grain, too. `canopyGrainAcross` used to be the number both crowns
  // were spelled with; it is the aggregate's alone now, and the explicit build
  // is spelled with `floretRadius_m`, so what this arm needs is the head layout
  // it was tuned against and nothing the aggregate has since moved.
  canopyGrainAcross: 64,
};

/**
 * A swept run's control point in the specimen's own frame.
 *
 * `sweptTubeNodes` fits a frame to each run and places the record with *both* a
 * position and an orientation, so a control point is in that rotated frame and
 * adding the position alone lands somewhere else entirely — which is how the
 * coping oracle below first came back reporting a root station 83 mm off the
 * axis when it was the one authored *on* the axis.
 */
function pointInSpecimen(node: SceneryNode, position: { x: number; y: number; z: number }): { x: number; y: number; z: number } {
  const place = node.place;
  const origin = place?.position ?? { x: 0, y: 0, z: 0 };
  const q = place?.orientation;
  if (!q) return { x: origin.x + position.x, y: origin.y + position.y, z: origin.z + position.z };
  // v + 2 * cross(q.xyz, cross(q.xyz, v) + q.w * v), the standard rotation.
  const tx = 2 * (q.y * position.z - q.z * position.y);
  const ty = 2 * (q.z * position.x - q.x * position.z);
  const tz = 2 * (q.x * position.y - q.y * position.x);
  return {
    x: origin.x + position.x + q.w * tx + (q.y * tz - q.z * ty),
    y: origin.y + position.y + q.w * ty + (q.z * tx - q.x * tz),
    z: origin.z + position.z + q.w * tz + (q.x * ty - q.y * tx),
  };
}

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
    // Coverage means the same thing under either canopy — how many times over
    // the crown's envelope is covered by the surfaces published on it — but the
    // surface is a floret in one and a whole lobe in the other, so the floors
    // differ. An aggregate needs its lobes merely to overlap; a floret field
    // needs roughly twice its own area before the beads fuse.
    const floor = BONSAI_POND_CANOPY.canopy === "aggregate" ? 1.8 : 1.9;
    assert.ok(sibling.floretCoverage > floor, `seed ${seed} covers only ${sibling.floretCoverage.toFixed(2)}x`);
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
  // The aggregate's whole argument, as a number. Spelled out, this crown is
  // about 1 160 records — better than a quarter of the scene-wide 4 096 — and it
  // is now one record a head plus a six-record bed plus the trunk. The explicit
  // form still spends its share, which is what the budget ceiling above is
  // guarding.
  assert.ok(planBonsai(specFor(POND_FLORETS)).leafCount > 900,
    "the explicit form should be spending most of its share; a large drop means the culls broke");
  const aggregate = planBonsai(specFor(BONSAI_POND_CANOPY)).leafCount;
  assert.ok(aggregate < 150, `the aggregate crown published ${aggregate}; it is meant to collapse the floret field`);
  // One record a head, not two, and that is what the closure core paid for.
  //
  // A lattice cluster cannot be both granular and closed — it only stops letting
  // rays through at `r >= P*sqrt(3)/2`, where the packing is a solid the
  // envelope's `max` cuts off flush — so it needed an ellipsoid core a
  // granulation depth beneath it, and that core is what the frame's black pits
  // were the bottom of. `seeded-lobes` is closed by construction and silhouettes
  // on its own lobes, so the record it freed went to *more heads* instead.
  const crown = leaves(planBonsai(specFor(BONSAI_POND_CANOPY)).nodes)
    .filter((node) => node.tags?.includes("crown"));
  const heads = crown.filter((node) => node.id.includes("/lobe-"));
  const bed = crown.filter((node) => node.id.includes("/bed-"));
  assert.equal(heads.length, BONSAI_POND_CANOPY.lobes);
  assert.ok(heads.every((node) => node.kind === "cluster" && node.field === "seeded-lobes"),
    "a canopy head that is not a seeded-lobes cluster has gone back to needing a closure core");
  // The bed is the one place an exact ellipsoid is the right answer, and the
  // count is small: six records close a plate that twenty-four heads leave 15%
  // open to a vertical ray. See the emission for the measurement.
  assert.ok(bed.length > 0 && bed.length <= BONSAI_POND_CANOPY.lobes / 3,
    `the canopy bed is ${bed.length} records against ${heads.length} heads`);
  assert.ok(bed.every((node) => node.kind === "ellipsoid"),
    "the bed exists to be closed; a procedural field fills some fraction of its envelope and an ellipsoid fills all of it");
  // And nothing below the crown is a cone. A hard union is C0 but not C1, so a
  // chain of them steps its shading normal at every junction and a taper renders
  // as stacked faceted pipes; every run is one swept record now.
  assert.equal(leaves(planBonsai(specFor(BONSAI_POND_CANOPY)).nodes).filter((node) => node.kind === "cone").length, 0);
});

test("every published scale clears the leaf that has to draw it", () => {
  // The band law, which is the constraint this whole object was rebuilt under.
  // Shading is the voxel cell's surface rather than an analytic intersection
  // with the record, so a feature whose period is under about two leaves renders
  // as aliasing and one over about three renders as geometry. The crown that
  // shipped before had its three lattice octaves at 45, 22.5 and 11.25 mm
  // against a 25 mm leaf — 1.8, 0.9 and 0.45 — and not one of them could draw.
  const ladder = bonsaiCanopyLadder(BONSAI_POND_CANOPY);
  assert.equal(ladder.leafSize_m, BONSAI_DEFAULT_LEAF_SIZE_M);
  assert.ok(ladder.headLeaves >= 3,
    `a canopy head spans ${ladder.headLeaves.toFixed(1)} leaves; under three it is not a head, it is noise`);
  assert.ok(ladder.floretLeaves >= 2,
    `a floret spans ${ladder.floretLeaves.toFixed(1)} leaves; under two it does not render as a floret at all`);

  // The head is published *coarser* than the plate on purpose while the leaf is
  // the shading resolution — a legible head at 24% of the crown beats a faithful
  // one at 17% that is marginal — and this pins how much coarser, so "slightly"
  // cannot drift into "a disc". Nine heads, which is what shipped, measured 2.6.
  const coarsening = ladder.headWidth_m / ladder.plateHeadWidth_m;
  assert.ok(coarsening > 1.15 && coarsening < 1.75,
    `the head is ${coarsening.toFixed(2)} times the plate's own; that is either marginal or a disc`);

  // Three scales, each the next one's cluster, which is what makes this a
  // Romanesco rather than a lumpy ball. Two florets across a head is not a
  // cluster of anything.
  assert.ok(ladder.headWidth_m / ladder.floretWidth_m > 2,
    "a head has to be a cluster of florets, not a pair of them");
  assert.ok(ladder.crownWidth_m / ladder.headWidth_m > 3,
    "the crown has to be a cluster of heads");

  // And the ladder converges on the plate as the leaf shrinks rather than being
  // re-authored — the whole reason the floret is derived and not a number in the
  // form. At a 6 mm leaf nothing is floored and the floret is the plate's own.
  for (const leaf of [BONSAI_DEFAULT_LEAF_SIZE_M, 0.0125, 0.006]) {
    const fine = bonsaiCanopyLadder(BONSAI_POND_CANOPY, leaf);
    assert.ok(fine.floretWidth_m <= ladder.floretWidth_m + 1e-9,
      `a ${1000 * leaf} mm leaf published a coarser floret than a ${1000 * BONSAI_DEFAULT_LEAF_SIZE_M} mm one`);
  }
  const converged = bonsaiCanopyLadder(BONSAI_POND_CANOPY, 0.006);
  assert.ok(Math.abs(converged.floretWidth_m - converged.plateFloretWidth_m) < 1e-9,
    `at a 6 mm leaf the floret is ${(1000 * converged.floretWidth_m).toFixed(1)} mm and the plate asks for ${(1000 * converged.plateFloretWidth_m).toFixed(1)}`);

  // Every form's ladder is solved against its own crown rather than re-authored,
  // which is the point of counting florets instead of measuring them. The small
  // forms cannot reach the floret floor — a 0.25 m crown has nowhere to put a
  // 75 mm floret — and what they must not do is fail silently, so the ABI's own
  // containment cap is asserted rather than the floor.
  for (const [name, form] of FORMS) {
    const each = bonsaiCanopyLadder(form);
    assert.ok(each.headLeaves >= 3, `${name}'s head spans only ${each.headLeaves.toFixed(1)} leaves`);
    assert.ok(each.floretSpan <= 0.47,
      `${name} publishes florets at ${each.floretSpan.toFixed(2)} of their head; the ABI clip will slice them`);
    assert.ok(Math.abs(each.plateFloretWidth_m * form.canopyGrainAcross - each.crownWidth_m) < 1e-9,
      `${name} does not carry its own authored grain`);
  }

  // The published floret in leaves, through the function the rest of the code
  // asks with, at the leaf sizes that matter.
  assert.ok(bonsaiFloretVoxelSpan(BONSAI_POND_CANOPY, BONSAI_DEFAULT_LEAF_SIZE_M) >= 2.9);
  assert.ok(bonsaiFloretVoxelSpan(BONSAI_POND_CANOPY, 0.006) >= 4.9);
});

test("every swept run publishes a packing the render ABI will accept", () => {
  // This one is a world-build failure rather than a bad frame, and it reaches
  // the eye as `GPU initialization failed` naming a segment index — nowhere near
  // the numbers that caused it. A sweep segment may not taper faster than it
  // runs (a round cone that does is one ball inside the other and the closed-form
  // distance degenerates), and the base's flare is authored steep enough to hit
  // that on any form whose stump is short against its bole. Checked here across
  // every form and several seeds, through the ABI's own validator, so a re-tune
  // of the flare fails on this bench instead of on a device.
  for (const [name, form] of FORMS) {
    for (const seed of [0x8017a1, 1, 7, 0x51ed, 0x7fff_ffff]) {
      for (const node of leaves(planBonsai(specFor(form, { seed })).nodes)) {
        if (node.kind !== "cluster" || node.field !== "tapered-sweep") continue;
        assert.doesNotThrow(
          () => validateSvoClusterPacking({
            field: "tapered-sweep",
            smoothRadius_m: node.smoothRadius,
            seed: node.seed >>> 0,
            points: node.points.map((point) => ({ position_m: point.position, radius_m: point.radius })),
          }, node.lobe),
          `${name} seed ${seed}: ${node.id}`,
        );
      }
    }
  }
});

test("the buttresses are rounded roots on the bank, not a collar of lumps over the water", () => {
  // Three properties, and the object failed all three in one round: a root as
  // thin as a leaf reads as a row of disconnected cell faces, roots that touch
  // all the way round read as one lumpy collar, and a root over the coping is a
  // root growing out of open water.
  //
  // This used to assert a *fin*: several `seeded-lobes` envelopes chained along
  // each bearing, a quarter as thick as they were tall. Both halves of that
  // argument went with analytic shading — see the emission — and what replaced
  // it is one round swept run per bearing, which is also what the plate shows.
  for (const seed of [0x8017a1, 1, 7, 0x51ed, 0x7fff_ffff]) {
    const plan = planBonsai(specFor(BONSAI_POND_CANOPY, { seed }));
    // By tag, not by field. Every run below the crown is a `tapered-sweep` now,
    // so a filter on the field alone picks up the trunks and the limbs too.
    const roots = leaves(plan.nodes).filter((node) => node.tags?.includes("root"));
    assert.ok(roots.every((node) => node.kind === "cluster" && node.field === "tapered-sweep"),
      "a buttress that is not a swept run has a hard union at every junction of it");
    assert.ok(roots.length >= 3, `seed ${seed}: ${roots.length} root runs; the bearings have stopped publishing`);

    for (const root of roots) {
      if (root.kind !== "cluster" || root.field !== "tapered-sweep") continue;
      // Every station over a leaf and a half across at the shoulder, tapering to
      // the toe, which is where a root is meant to disappear into the ground it
      // is bedded in. Under a leaf anywhere but the last station is a run the
      // voxel surface cannot carry.
      root.points.forEach((point, index) => {
        const across_mm = 2000 * point.radius;
        const floor_mm = index >= root.points.length - 1 ? 25 : 1.5 * 25;
        assert.ok(across_mm >= floor_mm,
          `seed ${seed}: ${root.id} station ${index} is ${across_mm.toFixed(0)} mm across, under ${floor_mm} mm`);
      });
      // And tapering rather than parallel-sided, which is the difference between
      // a root and a length of pipe.
      const first = root.points[0].radius;
      const last = root.points[root.points.length - 1].radius;
      assert.ok(last < first, `seed ${seed}: ${root.id} does not taper along its run`);
    }

    // Air between them, where a real buttress system has its deep V valleys. The
    // merge is meant to happen only inside the base.
    //
    // Sampled at one fixed radius from the axis rather than at each root's own
    // mid-station, and both halves of that matter. Per record is wrong because a
    // run whose best single envelope wastes too much volume is *split*, so one
    // bearing can publish two records a few degrees apart and comparing those
    // measures the splitter. Per root's own mid-station is wrong because a
    // bearing whose bank runs out publishes a stub whose stations are all inside
    // the base — its median station sits 30 mm from the axis and 25 mm thick, and
    // subtends fifty-six degrees on its own, which came back as *minus* sixty-one
    // degrees of air between neighbours. A stub that never leaves the flare
    // cannot make a collar, so the sample skips it.
    const AIR_RADIUS_M = 1.8 * BONSAI_POND_CANOPY.boleRadius_m;
    const byBearing = new Map<string, { radial: number; radius: number; angle: number }[]>();
    for (const root of roots) {
      if (root.kind !== "cluster" || root.field !== "tapered-sweep") continue;
      const bearing = /root-\d+/.exec(root.id)?.[0] ?? root.id;
      const stations = byBearing.get(bearing) ?? [];
      for (const point of root.points) {
        const at = pointInSpecimen(root, point.position);
        stations.push({ radial: Math.hypot(at.x, at.z), radius: point.radius, angle: Math.atan2(at.z, at.x) });
      }
      byBearing.set(bearing, stations);
    }
    const mids = [...byBearing.values()]
      .map((stations) => stations.filter((station) => station.radial >= AIR_RADIUS_M)
        .sort((a, b) => a.radial - b.radial)[0])
      .filter((station): station is NonNullable<typeof station> => station !== undefined)
      .map((station) => ({ angle: station.angle, radius: station.radial, half: station.radius }))
      .sort((a, b) => a.angle - b.angle);
    // Three bearings clear 1.8 bole radii at every seed tried; at 2.0 the bank
    // has run out under two of them and the sample stops meaning anything, which
    // is why the radius is 1.8 and not further.
    assert.ok(mids.length >= 3, `seed ${seed}: only ${mids.length} bearings reach ${(1000 * AIR_RADIUS_M).toFixed(0)} mm out`);
    let narrowest = Infinity;
    for (let i = 0; i < mids.length; i += 1) {
      const here = mids[i], next = mids[(i + 1) % mids.length];
      const between = ((next.angle - here.angle) + 2 * Math.PI) % (2 * Math.PI);
      const subtended = Math.asin(Math.min(1, here.half / here.radius)) + Math.asin(Math.min(1, next.half / next.radius));
      narrowest = Math.min(narrowest, (between - subtended) * 180 / Math.PI);
    }
    // Nine degrees is the worst of eight seeds at this radius, which is 14 mm of
    // gap — a notch rather than a valley, and that is correct here: the roots are
    // *meant* to fuse inside the flare and separate beyond it. Measured at 2.0
    // and 2.4 bole radii, where the run has left the base, the same seeds open to
    // 18 and 21 degrees. What this bar catches is the failure the fins had, which
    // was zero and negative — one lumpy collar all the way round.
    assert.ok(narrowest > 6, `seed ${seed}: only ${narrowest.toFixed(0)} degrees of air between adjacent roots; they will fuse into a collar`);

    // And on the bank. Every station's own ball against the coping's outer foot
    // — a root may reach the foot and no further, whatever `rootReach` asked for.
    // The stations bound the run exactly, because a sweep is the union of balls
    // about them; the fitted envelope is much larger and would fail a root that
    // stops well short.
    for (const root of roots) {
      if (root.kind !== "cluster" || root.field !== "tapered-sweep") continue;
      for (const point of root.points) {
        const at = pointInSpecimen(root, point.position);
        for (let i = 0; i < 24; i += 1) {
          const angle = 2 * Math.PI * i / 24;
          const x = STAND_M[0] + at.x + point.radius * Math.cos(angle);
          const z = STAND_M[1] + at.z + point.radius * Math.sin(angle);
          assert.ok(pondVesselPlanDistance(CURVE, x, z) > VESSEL.rimHalfWidth_m,
            `seed ${seed}: ${root.id} reaches over the coping`);
        }
      }
    }
  }
});

test("the budget is enforced rather than merely documented", () => {
  assert.throws(
    () => planBonsai(specFor(POND_FLORETS, { maximumLeaves: 200 })),
    /over its 200-leaf budget/,
    "a form that overspends must fail loudly, not silently take another object's records",
  );
});

test("the florets cover the crown at the overlap that fuses them", () => {
  // One is tangent florets with gaps between; two is where they stop reading as
  // separate beads. Every shipped form has to clear two, and this is the oracle
  // that says so without a render — the coverage a form reaches for a given
  // record count is the whole argument about this object.
  for (const [name, form] of [...FORMS, ["pond canopy, florets", POND_FLORETS] as const]) {
    const { floretCoverage } = planBonsai(specFor(form));
    if (form.canopy === "aggregate") {
      // An aggregate covers its own head by construction, so what this ratio
      // measures for it is whether the *heads* overlap: whether the crown is one
      // fused mass with clefts in it, or a heap of separate balls with sky
      // between them. Above four and a half it would be paying for heads buried
      // inside other heads, which for an aggregate is wasted march rather than
      // wasted records.
      assert.ok(floretCoverage > 1.8, `${name} heads cover only ${floretCoverage.toFixed(2)}x; the crown will show sky`);
      assert.ok(floretCoverage < 5, `${name} covers ${floretCoverage.toFixed(2)}x; masses are buried inside masses`);
      continue;
    }
    assert.ok(floretCoverage > 2, `${name} covers only ${floretCoverage.toFixed(2)}x; its florets will read as beads`);
    assert.ok(floretCoverage < 4, `${name} covers ${floretCoverage.toFixed(2)}x; records are being spent under the surface`);
  }
});

test("the crown is a wide flat plate that overhangs where it is told to", () => {
  // Measured off the canopy's own extent rather than the specimen's. Every
  // question here is about the crown, and against the whole specimen all three
  // answers move when the trunk or the bank under it does.
  for (const seed of [0x8017a1, 1, 7, 0x51ed, 0x7fff_ffff]) {
    const plan = planBonsai(specFor(BONSAI_POND_CANOPY, { seed }));
    const { min, max } = plan.crownBounds_m;
    const span = { x: max.x - min.x, y: max.y - min.y, z: max.z - min.z };
    // Authored as 0.74 x 0.64 across a 0.135 m plate, and the band is one-sided
    // on purpose. Every head is inset by its own plan radius, so nothing may
    // reach past `crownRadius_m` — an overrun is a bug, and two earlier versions
    // had one at a third again of the authored crown. Falling short is
    // structural: a finite set of round heads inscribed in a disc spans less
    // than the disc, because whichever head is nearest the axis is a few degrees
    // off it. A tenth is what eighteen heads leave; a fifth would mean the rim
    // has stopped being covered.
    for (const [axis, authored] of [["x", 2 * BONSAI_POND_CANOPY.crownRadius_m[0]], ["z", 2 * BONSAI_POND_CANOPY.crownRadius_m[1]]] as const) {
      const reached = span[axis] / authored;
      assert.ok(reached <= 1.001, `seed ${seed}: crown ${axis} span ${span[axis].toFixed(3)} overruns its authored ${authored.toFixed(3)}`);
      assert.ok(reached > 0.84, `seed ${seed}: crown ${axis} span ${span[axis].toFixed(3)} reaches only ${(100 * reached).toFixed(0)}% of ${authored.toFixed(3)}`);
    }
    const top = ground(...STAND_M) + BONSAI_POND_CANOPY.height_m;
    assert.ok(Math.abs(max.y - top) < 0.02, `seed ${seed}: crown top ${max.y.toFixed(3)} against an authored ${top.toFixed(3)}`);
    // The plate, as the one proportion the reference is unambiguous about. This
    // is the crown's own aspect, so it is a real bound on the shape rather than
    // on how tall the tree happens to be — a mound of equal balls comes out
    // under two however wide the layout is.
    assert.ok(span.x / span.y > 3, `seed ${seed}: crown is ${(span.x / span.y).toFixed(1)} times as wide as thick; the plate has become a mound`);
  }

  // The overhang, as the reference shows it: canopy standing over open water.
  // Measured off the crown's own plan extent rather than off the list of masses,
  // because the crown is now a handful of nearly crown-sized plates whose
  // centres all sit near its middle — counting centres said "no overhang" of a
  // canopy reaching a quarter of a metre past the coping.
  const plan = planBonsai(specFor(BONSAI_POND_CANOPY));
  const { min, max } = plan.crownBounds_m;
  let overWater = 0, sampled = 0;
  for (let i = 0; i < 24; i += 1) {
    for (let j = 0; j < 24; j += 1) {
      const x = min.x + (max.x - min.x) * (i + 0.5) / 24;
      const z = min.z + (max.z - min.z) * (j + 0.5) / 24;
      // Inside the crown's own plan ellipse, so the corners of the box do not count.
      const cx = 2 * ((x - min.x) / (max.x - min.x)) - 1;
      const cz = 2 * ((z - min.z) / (max.z - min.z)) - 1;
      if (cx * cx + cz * cz > 1) continue;
      sampled += 1;
      if (pondVesselPlanDistance(CURVE, x, z) < -VESSEL.rimHalfWidth_m) overWater += 1;
    }
  }
  assert.ok(overWater / sampled > 0.2,
    `only ${(100 * overWater / sampled).toFixed(0)}% of the crown's plan reaches past the coping's inner foot`);
});

test("the crown's rim is a cloud edge rather than an ellipse", () => {
  // The defect this replaced: a lattice cluster *fills* its envelope, so at the
  // boundary the clipping `max` is what draws — the rim you see is the ellipsoid
  // and not the field. No amount of tuning the packing touches it, because the
  // clipping surface is doing the drawing. The only fix is compositional: the
  // union of several plates has an irregular rim exactly when their rims are in
  // different places, so this measures the union's own outline against the
  // ellipse it is inscribed in.
  const reachAlong = (lobes: ReturnType<typeof planBonsai>["lobes"], centre: { x: number; z: number }, angle: number): number => {
    const d = { x: Math.cos(angle), z: Math.sin(angle) };
    let far = 0;
    for (const lobe of lobes) {
      // Largest t with (centre + t d) inside this plate's plan ellipse.
      const ox = (centre.x - lobe.center_m.x) / lobe.radius_m.x;
      const oz = (centre.z - lobe.center_m.z) / lobe.radius_m.z;
      const dx = d.x / lobe.radius_m.x;
      const dz = d.z / lobe.radius_m.z;
      const a = dx * dx + dz * dz;
      const b = 2 * (ox * dx + oz * dz);
      const c = ox * ox + oz * oz - 1;
      const discriminant = b * b - 4 * a * c;
      if (discriminant <= 0) continue;
      far = Math.max(far, (-b + Math.sqrt(discriminant)) / (2 * a));
    }
    return far;
  };
  for (const seed of [0x8017a1, 1, 7, 0x51ed, 0x7fff_ffff]) {
    const plan = planBonsai(specFor(BONSAI_POND_CANOPY, { seed }));
    const { min, max } = plan.crownBounds_m;
    const centre = { x: 0.5 * (min.x + max.x), z: 0.5 * (min.z + max.z) };
    const semi = { x: 0.5 * (max.x - min.x), z: 0.5 * (max.z - min.z) };
    let low = Infinity, high = 0;
    for (let step = 0; step < 180; step += 1) {
      const angle = 2 * Math.PI * step / 180;
      // Against the *inscribing* ellipse at that bearing, so an elliptical crown
      // scores a flat 1 at every angle however eccentric it is.
      const ellipse = 1 / Math.hypot(Math.cos(angle) / semi.x, Math.sin(angle) / semi.z);
      const ratio = reachAlong(plan.lobes, centre, angle) / ellipse;
      low = Math.min(low, ratio);
      high = Math.max(high, ratio);
    }
    // A single plate, or several nested concentrically, measures 1.00 to 1.00 —
    // which is what a lily pad looks like. A cloud edge scallops in between the
    // plates that reach furthest.
    assert.ok(high > 0.97, `seed ${seed}: the rim never reaches its own bounds (${high.toFixed(2)})`);
    assert.ok(low < 0.86, `seed ${seed}: the rim is an ellipse to within ${(100 * (1 - low)).toFixed(0)}%; it will read as machined`);
    assert.ok(low > 0.55, `seed ${seed}: the rim scallops to ${(100 * low).toFixed(0)}%; the crown has come apart into separate heads`);
  }
});

test("the crown clears the coping it overhangs", () => {
  const plan = planBonsai(specFor(BONSAI_POND_CANOPY));
  const crest = VESSEL.groundHeight_m + VESSEL.rimHeight_m * (1 + VESSEL.sectionHeightVariation);
  // Nothing in the canopy may come down onto the rim it reaches over. Checked
  // against the crest's own upper bound rather than a sample, because the
  // section swells as it runs and a sample would pass on the thin quarter.
  //
  // Per node and per *axis*: every canopy record is an unrotated ellipsoid or an
  // unrotated cluster envelope, so what it reaches down to is its own y half-
  // axis. Taking the largest of the three said a 0.33 m wide plate hung 0.33 m
  // below its own centre, which on a crown five times as wide as it is thick is
  // wrong by a factor of five and was about to fail a crown with a quarter of a
  // metre of clearance.
  for (const node of leaves(plan.nodes)) {
    if (!node.tags?.includes("floret") && !node.tags?.includes("crown")) continue;
    const position = node.place?.position;
    assert.ok(position, "every canopy part is placed");
    assert.ok(!node.place?.orientation, `${node.id} is rotated; its y half-axis is no longer its downward reach`);
    const reach = node.kind === "ellipsoid" ? node.radius.y : node.kind === "cluster" ? node.lobe.y : 0;
    const world = ground(...STAND_M) + position.y - reach;
    assert.ok(world > crest, `${node.id} reaches down to ${world.toFixed(3)}, under the crest at ${crest.toFixed(3)}`);
  }
  // And the whole canopy's own floor, which is the number the scene cares about.
  assert.ok(plan.crownBounds_m.min.y > crest,
    `the crown's underside is at ${plan.crownBounds_m.min.y.toFixed(3)}, under the crest at ${crest.toFixed(3)}`);
});

test("the buttress roots meet the bank they run out over, and grip it", () => {
  const plan = planBonsai(specFor(BONSAI_POND_CANOPY));
  // At most one per authored bearing, and a bearing whose bank runs out is left
  // bare rather than given a buttress hanging over the water — so this is a
  // range, not an equality.
  assert.ok(plan.rootFeet_m.length <= BONSAI_POND_CANOPY.roots);
  assert.ok(plan.rootFeet_m.length >= 3, `only ${plan.rootFeet_m.length} bearings kept a buttress`);
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
  // The terrace falls away toward the coping inside the buttresses' own reach,
  // so a generator that had assumed a plane would have left toes hanging in air
  // by this much. Guards the height query against being quietly dropped.
  // Bounded on both sides now, and the pair is the whole story. Below: the
  // terrace really does fall away inside the buttresses' own reach, so a
  // generator that had assumed a plane would leave toes hanging in air by this
  // much — the guard against the height query being quietly dropped. Above: no
  // toe is allowed to follow the bank further down than the run may descend
  // before it has left it, which is what stops a root growing over the pond.
  assert.ok(fall > 0.015, `the bank only falls ${(1000 * fall).toFixed(0)} mm under the root spread; the oracle has stopped testing anything`);
  assert.ok(fall < 0.9 * BONSAI_POND_CANOPY.boleRadius_m,
    `a toe follows the bank ${(1000 * fall).toFixed(0)} mm down; the reach clamp has stopped biting`);

  // The grip. A buttress is not a finger: the plate it makes has to be wider
  // than the stump it leaves, or the base reads as a post with tabs beside it —
  // which is what the first version of this looked like. Measured as the
  // published toes' own spread against the stump's widest section.
  const spread = Math.max(...plan.rootFeet_m.map((foot) =>
    Math.hypot(foot.x - STAND_M[0], foot.z - STAND_M[1])));
  assert.ok(spread > 2 * BONSAI_POND_CANOPY.boleRadius_m,
    `the buttresses reach ${(1000 * spread).toFixed(0)} mm against a ${(1000 * BONSAI_POND_CANOPY.boleRadius_m).toFixed(0)} mm bole; the flare has gone`);
  // And thick where they leave, not just long — but measured in leaves rather
  // than as a share of the stump, which is the change round roots forced.
  //
  // Under fins `rootWidth` was a fin *height* and had to be near one, because a
  // fin leaves the axis inside the stump and anything much thinner emerged as a
  // rib on the flare rather than as part of it. Under round roots it is a
  // *radius*, and a root as thick as the bole it leaves is not a root — the plate
  // has them at about half. What has to hold instead is the floor the leaf sets:
  // the shoulder over a leaf and a half, which is asserted station by station in
  // the test above, and the toe still bedded rather than floating, which is
  // asserted from `rootFeet_m` at the top of this one.
  assert.ok(BONSAI_POND_CANOPY.rootWidth > 0.45 && BONSAI_POND_CANOPY.rootWidth < 0.85,
    "a root as thick as its own stump is a buttress fin again; one under half a leaf is nothing");
  assert.ok(2000 * BONSAI_POND_CANOPY.rootWidth * BONSAI_POND_CANOPY.boleRadius_m >= 1.5 * 25,
    "the shoulder of a root has to clear a leaf and a half");
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
  assert.throws(() => planBonsai(specFor(BONSAI_POND_CANOPY, { roots: 2 })), /at least three root buttresses/);
  assert.throws(() => planBonsai(specFor(BONSAI_POND_CANOPY, { rootTaper: 1 })), /buttress taper/);
  assert.throws(() => planBonsai(specFor(BONSAI_POND_CANOPY, { canopyGrainAcross: 2 })), /florets across/);
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
  // Against the explicit form, because this law *is* the explicit form: it is
  // the economy the aggregate was built to escape. Under `canopy: "aggregate"`
  // halving the floret costs nothing at all — the packing is evaluated in the
  // shader and the record count does not move — which is the same measurement
  // read from the other side, and is asserted below.
  const coarse = planBonsai(specFor(POND_FLORETS, { maximumLeaves: 40_000 }));
  const fine = planBonsai(specFor(POND_FLORETS, {
    floretRadius_m: 0.5 * POND_FLORETS.floretRadius_m,
    floretsPerLobe: 4 * POND_FLORETS.floretsPerLobe,
    maximumLeaves: 40_000,
  }));
  // Four times the florets at half the width does not even hold coverage: the
  // occlusion cull takes a larger share of small florets, because "buried deeper
  // than my own depth" is a lower bar for a shallower cap. So the true bill for
  // halving the grain is *more* than four-fold.
  assert.ok(fine.floretCoverage < coarse.floretCoverage,
    `four times the florets at half the width held coverage at ${fine.floretCoverage.toFixed(2)}; the cull was expected to bite harder`);
  assert.ok(fine.leafCount > 2.3 * coarse.leafCount, "the record count did not scale with the count of florets");
  // And the aggregate's side of it: refining the grain is free. Not the same
  // knob — `canopyGrainAcross` is what an aggregate crown is spelled with — and
  // that is the point: under the aggregate the number the whole economy above is
  // expressed in has stopped being a cost at all.
  assert.equal(
    planBonsai(specFor(BONSAI_POND_CANOPY, { canopyGrainAcross: 2 * BONSAI_POND_CANOPY.canopyGrainAcross })).leafCount,
    planBonsai(specFor(BONSAI_POND_CANOPY)).leafCount,
    "an aggregate's grain must cost no records at all; that is the whole trade",
  );
  // Against 2 458 — six tenths of the 4 096 the scene-wide candidate index held
  // when this was written — rather than against the constant, which W0 of the
  // raster-visibility program raised to 16 384 for the `hero-garden-hose-x10`
  // acceptance scene. The claim being made is about the *record budget a set is
  // authored inside of*, and that is still the hero's: a crown allowed to spend
  // 2 600 records on one halving is the economy the aggregate exists to escape,
  // whatever headroom the arena has since been given for a ten-times scene.
  assert.ok(fine.leafCount > 0.6 * 4_096,
    "one halving should already take most of the record budget the hero set is authored inside of");
  assert.ok(SVO_PRIMITIVE_CANDIDATE_MAXIMUM_LEAVES >= 4_096,
    "the candidate index may grow, but this bound was measured against its 4 096");
});

test("the floret's span in leaves is the constraint, and is reported", () => {
  // This test used to assert the opposite, and the sentence it asserted was
  // measured: "the hero crown's floret is half a cell across on a 25 mm lattice,
  // and it still silhouettes exactly, because a scenery primitive's primary
  // visibility is a BVH over exact records and never consults the per-voxel
  // material owner". True then. Analytic shading is gone — the primary returns
  // the voxel cell's surface — so the leaf is the geometric resolution and this
  // number is the whole constraint on the object rather than a curiosity.
  //
  // The hero crown's published floret is three leaves at 25 mm, where the crown
  // that shipped was 0.35 of one.
  assert.ok(bonsaiFloretVoxelSpan(BONSAI_POND_CANOPY, 0.025) >= 2.9,
    `the hero floret spans ${bonsaiFloretVoxelSpan(BONSAI_POND_CANOPY, 0.025).toFixed(2)} leaves`);
  // And it *follows* the leaf rather than being a fixed length, so the same form
  // resolves further as the leaf shrinks instead of being re-authored. Halving
  // the leaf does not double the span, because the floret shrinks with it until
  // it reaches the plate's own proportion and then stops.
  const coarse = bonsaiFloretVoxelSpan(BONSAI_POND_CANOPY, 0.025);
  const fine = bonsaiFloretVoxelSpan(BONSAI_POND_CANOPY, 0.0125);
  assert.ok(fine >= coarse, "a finer leaf must not draw a coarser floret");
  assert.ok(fine < 2 * coarse, "the floret is following the leaf down, not standing still");
  // The explicit crown is a fixed authored length and does not follow anything,
  // which is the difference the aggregate exists for.
  assert.ok(Math.abs(bonsaiFloretVoxelSpan(POND_FLORETS, 0.025) - 2.34) < 0.02);
  assert.equal(bonsaiFloretVoxelSpan(POND_FLORETS, 0.0125), 2 * bonsaiFloretVoxelSpan(POND_FLORETS, 0.025));
  assert.throws(() => bonsaiFloretVoxelSpan(BONSAI_POND_CANOPY, 0), /positive cell size/);
});

/**
 * The convergence is *connected*, which for two years it was not.
 *
 * `bonsaiCanopyLadder` was built so that a shrinking leaf carries the published
 * floret down to the plate's own proportion with nothing re-authored — 75 mm at
 * a 25 mm leaf, 37.5 at 12.5, and the plate's 30 mm from 6.25 mm down. Every
 * part of that existed except the wire: `BonsaiSpec.leafSize_m` is optional, no
 * document passed one, and the hero garden therefore drew 75 mm florets no
 * matter what lattice it was rendered on.
 *
 * The leaf now comes from `SceneryGeneratorRequest.detailCellSize_m`, which is
 * the scene's own. This walks the hero document's own bonsai node through
 * `growSceneryGenerator` at each rung and reads the floret back off the records,
 * so it is checking what the scene publishes rather than what the ladder
 * returns.
 */
test("the hero document's specimen draws at the scene's own leaf", () => {
  // Built here rather than read off the hero document, because the hero set's
  // tree is a procedural oak since the cutover and this species no longer stands
  // in it. What the test is actually about is unchanged and is nothing to do
  // with that scene: a specimen resolves its canopy ladder against the leaf it
  // is handed. Reading the node out of the garden only ever supplied params.
  const node = {
    kind: "generator", id: "bonsai", generator: "bonsai",
    seed: 0x8017a1,
    params: { ...BONSAI_POND_CANOPY, canopy: "aggregate", at_m: [0.55, -0.15], lean: [-0.55, 0.15] },
  } as const satisfies Extract<SceneryNode, { kind: "generator" }>;
  const form = node.params as { readonly lobes: number };
  /** Published floret width, in metres: `lobeSpan` is a share of the envelope. */
  const floretWidth_m = (detailCellSize_m: number): number => {
    const grown = growSceneryGenerator(node.generator, node.params, {
      key: node.id,
      seed: node.seed,
      groundHeightAt: () => HERO_GARDEN_VESSEL.groundHeight_m,
      detailCellSize_m,
      vessel: () => { throw new Error("a bonsai names no vessel"); },
    });
    const heads: number[] = [];
    for (const { node: child } of walkSceneryNodes(grown)) {
      if (child.kind !== "cluster" || child.field !== "seeded-lobes") continue;
      if (!child.id.includes("lobe")) continue;
      heads.push((child.lobeSpan ?? 0) * 2 * child.lobe.x);
    }
    assert.ok(heads.length === form.lobes, "the crown must publish one seeded-lobe head per lobe");
    return Math.max(...heads);
  };
  // The authored default is untouched, which is the regression bar.
  assert.ok(Math.abs(floretWidth_m(BONSAI_DEFAULT_LEAF_SIZE_M) - 0.075) < 1e-9,
    "at the authored 25 mm leaf the crown must still publish its 75 mm floret");
  // And it converges, monotonically, onto the plate's own 30 mm — which is
  // `crownWidth / canopyGrainAcross` and therefore a *proportion*, so it stops
  // there rather than continuing to shrink with the lattice.
  const ladder = [0.025, 0.0125, 0.00625, 0.003125, 0.0015625].map(floretWidth_m);
  for (let index = 1; index < ladder.length; index += 1) {
    assert.ok(ladder[index] <= ladder[index - 1] + 1e-12, "a finer leaf must never publish a coarser floret");
  }
  assert.ok(Math.abs(ladder[2] - 0.030) < 1e-9, "by a 6.25 mm leaf the specimen has become the plate");
  assert.ok(Math.abs(ladder[4] - 0.030) < 1e-9, "and stays there: 30 mm is a proportion, not a resolution");
  // Free, and that is why it can be taken now. The floret is a packing parameter
  // of a record that already exists, so a converged crown is the same records.
  const counts = [0.025, 0.00625].map((leaf) => {
    let total = 0;
    for (const { node: child } of walkSceneryNodes(growSceneryGenerator(node.generator, node.params, {
      key: node.id, seed: node.seed, groundHeightAt: () => HERO_GARDEN_VESSEL.groundHeight_m,
      detailCellSize_m: leaf, vessel: () => { throw new Error("a bonsai names no vessel"); },
    }))) if (isSceneryPrimitiveNode(child)) total += 1;
    return total;
  });
  assert.equal(counts[0], counts[1], "converging the crown must not cost a record");
});
